# This file is part of Patsy
# Copyright (C) 2011-2015 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

# Tests for patsy/shared_build.py: the optional shared build, in which
# identical stateful-transform subexpressions share learned state within a
# single fit, and each batch of matrix building evaluates each shared
# subexpression only once.

import numpy as np
import pytest

from patsy import PatsyError
from patsy.build import design_matrix_builders, build_design_matrices
from patsy.desc import ModelDesc, Term, INTERCEPT
from patsy.design_info import DesignInfo
from patsy.eval import EvalFactor
from patsy.highlevel import dmatrix, dmatrices, incr_dbuilders
from patsy.shared_build import (
    shared_design_matrix_builders,
    build_shared_design_matrices,
)
from patsy.state import stateful_transform
from patsy.util import have_pandas

if have_pandas:
    import pandas


def _shared_builders(formula, data, eval_env=1, **kwargs):
    desc = ModelDesc.from_formula(formula)
    return shared_design_matrix_builders(
        [desc.lhs_termlist, desc.rhs_termlist], lambda: iter([data]), eval_env, **kwargs
    )


def _native_builders(formula, data, eval_env=1, **kwargs):
    desc = ModelDesc.from_formula(formula)
    return design_matrix_builders(
        [desc.lhs_termlist, desc.rhs_termlist], lambda: iter([data]), eval_env, **kwargs
    )


def _get_plan(design_info):
    plans = set()
    for factor_info in design_info.factor_infos.values():
        plan = factor_info.state.get("shared_plan")
        if plan is not None:
            plans.add(id(plan))
            plan_obj = plan
    assert len(plans) <= 1
    return plan_obj


def _make_counting_transform():
    # A stateful transform that records exactly what it memorizes, so tests
    # can check that shared objects see each row exactly once. Returns the
    # formula-visible function and the implementation class (for
    # isinstance checks).
    class Counter(object):
        def __init__(self):
            self.rows = 0
            self.chunks = 0
            self.finish_calls = 0
            self.total = 0.0

        def memorize_chunk(self, x):
            x = np.asarray(x, dtype=float)
            self.chunks += 1
            self.rows += x.shape[0]
            self.total += np.sum(x)

        def memorize_finish(self):
            self.finish_calls += 1

        def transform(self, x):
            return np.asarray(x, dtype=float) - self.total / self.rows

    return stateful_transform(Counter), Counter


def test_shared_state_is_actually_shared():
    counter, Counter = _make_counting_transform()
    data = {"x": [1.0, 2.0, 3.0, 4.0], "y": [4.0, 3.0, 2.0, 1.0]}
    # counter(x) occurs in three different factors
    builders = _shared_builders(
        "y ~ counter(x) + np.log(counter(x) + 10) + I(counter(x) ** 2)", data
    )
    plan = _get_plan(builders[1])
    counters = [o for o in plan.transforms.values() if isinstance(o, Counter)]
    # one shared object, and it memorized each row exactly once
    assert len(counters) == 1
    assert counters[0].rows == 4
    assert counters[0].chunks == 1
    assert counters[0].finish_calls == 1
    # all three factors refer to the same shared object
    used = [
        factor_info.state["shared_objs"]
        for factor_info in builders[1].factor_infos.values()
    ]
    assert used[0] == used[1] == used[2] and len(used[0]) == 1

    # the native build gives each occurrence its own object
    native = _native_builders(
        "y ~ counter(x) + np.log(counter(x) + 10) + I(counter(x) ** 2)", data
    )
    native_objs = set()
    for factor_info in native[1].factor_infos.values():
        native_objs.update(id(o) for o in factor_info.state["transforms"].values())
    assert len(native_objs) == 3


def test_no_double_accumulation_with_chunks():
    counter, Counter = _make_counting_transform()
    chunk1 = {"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]}
    chunk2 = {"x": [4.0, 5.0], "y": [4.0, 5.0]}
    chunks = [chunk1, chunk2]
    desc = ModelDesc.from_formula("y ~ counter(x) + np.log(counter(x) + 10)")
    builders = shared_design_matrix_builders(
        [desc.lhs_termlist, desc.rhs_termlist], lambda: iter(chunks), 0
    )
    plan = _get_plan(builders[1])
    (obj,) = plan.transforms.values()
    # 5 rows total, seen exactly once each, over 2 chunks
    assert obj.rows == 5
    assert obj.chunks == 2
    assert obj.finish_calls == 1
    # and the learned state matches the native incremental build
    n_lhs, n_rhs = incr_dbuilders(desc, lambda: iter(chunks))
    n_objs = [
        o
        for fi in n_rhs.factor_infos.values()
        for o in fi.state["transforms"].values()
        if isinstance(o, Counter)
    ]
    # native builds one Counter per factor; all see the same total
    assert len(n_objs) == 2
    for n_obj in n_objs:
        assert obj.total == n_obj.total


def test_nested_dependencies_advance_normally():
    counter, Counter = _make_counting_transform()
    data = {"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [5.0, 4.0, 3.0, 2.0, 1.0]}
    builders = _shared_builders(
        "y ~ counter(counter(x)) + counter(x) + bs(center(x), df=3)", data
    )
    plan = _get_plan(builders[1])
    # counter(counter(x)) needs two passes; bs(center(x)) needs two passes
    assert len(plan.pass_bins) == 2
    counters = [
        o for o in plan.transforms.values() if isinstance(o, Counter)
    ]
    # inner and outer counter are distinct objects, each saw the rows once
    assert len(counters) == 2
    for obj in counters:
        assert obj.rows == 5
        assert obj.chunks == 1
        assert obj.finish_calls == 1
    # inner transforms are in an earlier bin than their consumers
    from patsy.state import Center

    bin_of = {}
    for i, pass_bin in enumerate(plan.pass_bins):
        for name in pass_bin:
            bin_of[name] = i
    for name, obj in plan.transforms.items():
        if isinstance(obj, Center):
            center_bin = bin_of[name]
        elif type(obj).__name__ == "BS":
            bs_bin = bin_of[name]
    assert center_bin < bs_bin
    # matrices match the native build
    native = _native_builders(
        "y ~ counter(counter(x)) + counter(x) + bs(center(x), df=3)", data
    )
    got = build_shared_design_matrices(builders, data)
    want = build_design_matrices(native, data)
    for got_m, want_m in zip(got, want):
        assert np.allclose(np.asarray(got_m), np.asarray(want_m))


def test_matches_native_mixed_builtins():
    rng = np.random.RandomState(42)
    data = {
        "x": rng.normal(size=100),
        "z": rng.normal(size=100),
        "a": ["a", "b", "c"] * 33 + ["a"],
    }
    data["y"] = data["x"] + rng.normal(size=100)
    formula = (
        "y ~ center(x) + standardize(x) + scale(z, ddof=1) "
        "+ bs(x, df=5) + np.log(center(z) + 10) "
        "+ C(a) + center(x):C(a) + cr(z, df=4)"
    )
    shared = _shared_builders(formula, data)
    native = _native_builders(formula, data)
    # real DesignInfo objects with identical structure
    assert all(isinstance(b, DesignInfo) for b in shared)
    for s, n in zip(shared, native):
        assert s.column_names == n.column_names
        assert s.term_names == n.term_names
    got = build_shared_design_matrices(shared, data)
    want = build_design_matrices(native, data)
    for got_m, want_m in zip(got, want):
        assert np.allclose(np.asarray(got_m), np.asarray(want_m))
        assert got_m.design_info.column_names == want_m.design_info.column_names
    # categorical contrasts are preserved
    for s, n in zip(shared, native):
        for term in s.term_codings:
            for s_sub, n_sub in zip(s.term_codings[term], n.term_codings[term]):
                for f in s_sub.contrast_matrices:
                    assert np.array_equal(
                        s_sub.contrast_matrices[f].matrix,
                        n_sub.contrast_matrices[f].matrix,
                    )


def test_different_parameters_do_not_share():
    data = {
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "y": [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    }
    builders = _shared_builders(
        "y ~ bs(x, df=3) + bs(x, df=4) + standardize(x) "
        "+ standardize(x, ddof=1) + center(x) + center(y)",
        data,
    )
    plan = _get_plan(builders[1])
    from patsy.state import Center, Standardize

    types = [type(o).__name__ for o in plan.transforms.values()]
    # 2 distinct bs (different df), 2 distinct standardize (different ddof),
    # 2 distinct center (different input)
    assert types.count("BS") == 2
    assert types.count("Standardize") == 2
    assert types.count("Center") == 2
    # but identical parameters do share, even across very different factors
    builders2 = _shared_builders(
        "y ~ bs(x, df=3) + I(bs(x, df=3) ** 2) + standardize(x) "
        "+ np.log(standardize(x) + 10)",
        data,
    )
    plan2 = _get_plan(builders2[1])
    types2 = [type(o).__name__ for o in plan2.transforms.values()]
    assert types2.count("BS") == 1
    assert types2.count("Standardize") == 1
    # and results still match the native build
    native2 = _native_builders(
        "y ~ bs(x, df=3) + I(bs(x, df=3) ** 2) + standardize(x) "
        "+ np.log(standardize(x) + 10)",
        data,
    )
    got = build_shared_design_matrices(builders2, data)
    want = build_design_matrices(native2, data)
    for got_m, want_m in zip(got, want):
        assert np.allclose(np.asarray(got_m), np.asarray(want_m))


def test_new_batch_does_not_reuse_cached_values():
    rng = np.random.RandomState(0)
    fit_data = {"x": rng.normal(size=50), "y": rng.normal(size=50)}
    builders = _shared_builders(
        "y ~ center(x) + np.log(center(x) + 10) + bs(center(x), df=4)", fit_data
    )
    batch1 = {"x": rng.normal(size=10), "y": rng.normal(size=10)}
    batch2 = {"x": rng.normal(size=10), "y": rng.normal(size=10)}
    got1 = build_shared_design_matrices(builders, batch1)
    got2 = build_shared_design_matrices(builders, batch2)
    # a second batch must reflect the second batch's data, not the first's
    native = _native_builders(
        "y ~ center(x) + np.log(center(x) + 10) + bs(center(x), df=4)", fit_data
    )
    want1 = build_design_matrices(native, batch1)
    want2 = build_design_matrices(native, batch2)
    for got_m, want_m in zip(got1 + got2, want1 + want2):
        assert np.allclose(np.asarray(got_m), np.asarray(want_m))
    assert not np.allclose(np.asarray(got2[1]), np.asarray(got1[1]))


def test_independent_fits_do_not_share_state():
    data1 = {"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]}
    data2 = {"x": [10.0, 20.0, 30.0], "y": [10.0, 20.0, 30.0]}
    builders1 = _shared_builders("y ~ center(x) + standardize(x)", data1)
    builders2 = _shared_builders("y ~ center(x) + standardize(x)", data2)
    plan1 = _get_plan(builders1[1])
    plan2 = _get_plan(builders2[1])
    assert plan1 is not plan2
    # each fit matches a native fit on the same data
    for builders, data in ((builders1, data1), (builders2, data2)):
        native = _native_builders("y ~ center(x) + standardize(x)", data)
        got = build_shared_design_matrices(builders, data)
        want = build_design_matrices(native, data)
        for got_m, want_m in zip(got, want):
            assert np.allclose(np.asarray(got_m), np.asarray(want_m))
    # and the two fits really did learn different state
    (c1,) = [
        o for o in plan1.transforms.values() if type(o).__name__ == "Center"
    ]
    (c2,) = [
        o for o in plan2.transforms.values() if type(o).__name__ == "Center"
    ]
    assert c1._count == 3 and c2._count == 3
    assert not np.allclose(c1._sum, c2._sum)


def test_term_subsets_still_work():
    rng = np.random.RandomState(1)
    data = {
        "x": rng.normal(size=30),
        "z": rng.normal(size=30),
        "y": rng.normal(size=30),
    }
    formula = "y ~ center(x) + standardize(z) + bs(x, df=4)"
    shared = _shared_builders(formula, data)
    native = _native_builders(formula, data)
    for which in ("0 + center(x)", "standardize(z)", "0 + bs(x, df=4)"):
        s_sub = shared[1].subset(which)
        n_sub = native[1].subset(which)
        assert s_sub.column_names == n_sub.column_names
        got = build_shared_design_matrices([s_sub], data)[0]
        want = build_design_matrices([n_sub], data)[0]
        assert np.allclose(np.asarray(got), np.asarray(want))
    # a subset build must not touch variables used only by dropped terms
    new_data = {"x": rng.normal(size=7)}  # note: no 'z'
    s_sub = shared[1].subset("0 + center(x)")
    n_sub = native[1].subset("0 + center(x)")
    got = build_shared_design_matrices([s_sub], new_data)[0]
    want = build_design_matrices([n_sub], new_data)[0]
    assert np.allclose(np.asarray(got), np.asarray(want))


def test_joint_na_and_original_rows():
    # NAs in non-transformed columns; the transformed columns are clean, so
    # the memorized state is well-defined and rows drop jointly
    data = {
        "x": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0],
        "w": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "v": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        "y": [2.0, 1.0, np.nan, 4.0, 5.0, 6.0],
    }
    formula = "y ~ x + center(w) + standardize(v) + center(w):standardize(v)"
    shared = _shared_builders(formula, data)
    native = _native_builders(formula, data)
    got = build_shared_design_matrices(shared, data)
    want = build_design_matrices(native, data)
    # joint NA drop: rows 1 and 2 are dropped from all matrices
    for got_m, want_m in zip(got, want):
        assert got_m.shape[0] == want_m.shape[0] == 4
        assert np.allclose(np.asarray(got_m), np.asarray(want_m))
    # NA_action="raise" behaves the same way
    pytest.raises(
        PatsyError,
        build_shared_design_matrices,
        shared,
        data,
        NA_action="raise",
    )
    # with pandas input, the surviving original rows keep their index
    if have_pandas:
        pdf = pandas.DataFrame(data, index=[10, 11, 12, 13, 14, 15])
        shared_p = _shared_builders(formula, pdf)
        native_p = _native_builders(formula, pdf)
        got_p = build_shared_design_matrices(shared_p, pdf, return_type="dataframe")
        want_p = build_design_matrices(native_p, pdf, return_type="dataframe")
        for got_m, want_m in zip(got_p, want_p):
            assert isinstance(got_m, pandas.DataFrame)
            assert list(got_m.index) == [10, 13, 14, 15]
            assert list(got_m.index) == list(want_m.index)
            assert list(got_m.columns) == list(want_m.columns)
            assert np.allclose(np.asarray(got_m), np.asarray(want_m))


def test_categorical_contrasts_and_column_names():
    rng = np.random.RandomState(7)
    data = {
        "x": rng.normal(size=60),
        "a": ["a", "b", "c"] * 20,
        "y": rng.normal(size=60),
    }
    formula = "y ~ C(a, Treatment(reference='b')) + center(x):C(a) + bs(x, df=4)"
    shared = _shared_builders(formula, data)
    native = _native_builders(formula, data)
    assert shared[1].column_names == native[1].column_names
    got = build_shared_design_matrices(shared, data)
    want = build_design_matrices(native, data)
    for got_m, want_m in zip(got, want):
        assert np.allclose(np.asarray(got_m), np.asarray(want_m))


def test_plain_build_design_matrices_on_shared_builders():
    # DesignInfo objects from a shared build must work with the regular
    # build_design_matrices() too (it just doesn't share evaluations).
    rng = np.random.RandomState(3)
    data = {"x": rng.normal(size=40), "y": rng.normal(size=40)}
    formula = "y ~ center(x) + np.log(center(x) + 10) + bs(center(x), df=4)"
    shared = _shared_builders(formula, data)
    native = _native_builders(formula, data)
    got = build_design_matrices(shared, data)
    want = build_design_matrices(native, data)
    for got_m, want_m in zip(got, want):
        assert np.allclose(np.asarray(got_m), np.asarray(want_m))


def test_shared_build_on_native_design_infos():
    # build_shared_design_matrices() also accepts DesignInfo objects that
    # came from the regular builders, and then behaves exactly like
    # build_design_matrices().
    rng = np.random.RandomState(4)
    data = {"x": rng.normal(size=40), "y": rng.normal(size=40)}
    formula = "y ~ center(x) + bs(x, df=4)"
    native = _native_builders(formula, data)
    got = build_shared_design_matrices(native, data)
    want = build_design_matrices(native, data)
    for got_m, want_m in zip(got, want):
        assert np.allclose(np.asarray(got_m), np.asarray(want_m))


def test_shared_evaluation_actually_evaluates_once():
    # the batch entry point must evaluate each shared subexpression once,
    # not once per factor
    calls = []

    class Transform(object):
        def memorize_chunk(self, x):
            pass

        def memorize_finish(self):
            pass

        def transform(self, x):
            calls.append(len(x))
            return np.asarray(x, dtype=float)

    from patsy.state import stateful_transform as st

    tracked = st(Transform)
    data = {"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]}
    builders = _shared_builders(
        "y ~ tracked(x) + np.log(tracked(x) + 10) + I(tracked(x) ** 2)", data
    )
    del calls[:]
    build_shared_design_matrices(builders, data)
    assert calls == [3]
    # while the plain builder evaluates it once per factor
    del calls[:]
    build_design_matrices(builders, data)
    assert calls == [3, 3, 3]


def test_return_type_dataframe_and_dtype():
    if not have_pandas:
        pytest.skip("pandas not installed")
    rng = np.random.RandomState(5)
    pdf = pandas.DataFrame(
        {"x": rng.normal(size=20), "y": rng.normal(size=20)},
        index=np.arange(100, 120),
    )
    formula = "y ~ center(x) + bs(x, df=3)"
    shared = _shared_builders(formula, pdf)
    native = _native_builders(formula, pdf)
    got = build_shared_design_matrices(
        shared, pdf, return_type="dataframe", dtype=np.dtype(np.float32)
    )
    want = build_design_matrices(
        native, pdf, return_type="dataframe", dtype=np.dtype(np.float32)
    )
    for got_m, want_m in zip(got, want):
        assert isinstance(got_m, pandas.DataFrame)
        assert got_m.values.dtype == np.float32
        assert list(got_m.index) == list(want_m.index)
        assert list(got_m.columns) == list(want_m.columns)
        assert np.allclose(np.asarray(got_m), np.asarray(want_m))
        assert got_m.design_info.column_names == want_m.design_info.column_names


def test_eval_env_lookups():
    # variables resolved from the eval environment (not the data) work the
    # same as in the native build
    offset = 10.0
    data = {"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]}
    formula = "y ~ center(x) + I(center(x) + offset)"
    shared = _shared_builders(formula, data)
    native = _native_builders(formula, data)
    got = build_shared_design_matrices(shared, data)
    want = build_design_matrices(native, data)
    for got_m, want_m in zip(got, want):
        assert np.allclose(np.asarray(got_m), np.asarray(want_m))


def test_lhs_rhs_share_one_fit():
    # the lhs and rhs termlists of a single call form one fit: a transform
    # appearing on both sides is shared, and both matrices are correct
    counter, Counter = _make_counting_transform()
    data = {"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 6.0, 8.0]}
    builders = _shared_builders("counter(y) ~ counter(x) + I(counter(x) ** 2)", data)
    lhs_plan = _get_plan(builders[0])
    rhs_plan = _get_plan(builders[1])
    assert lhs_plan is rhs_plan
    objs = [o for o in lhs_plan.transforms.values() if isinstance(o, Counter)]
    # one counter for x (shared between the two rhs factors), one for y
    assert len(objs) == 2
    native = _native_builders("counter(y) ~ counter(x) + I(counter(x) ** 2)", data)
    got = build_shared_design_matrices(builders, data)
    want = build_design_matrices(native, data)
    for got_m, want_m in zip(got, want):
        assert np.allclose(np.asarray(got_m), np.asarray(want_m))


def test_multiple_formulas_one_fit():
    # several termlists (e.g. several formulas) built in one call share
    # state across all of them
    counter, Counter = _make_counting_transform()
    data = {"x": [1.0, 2.0, 3.0], "z": [3.0, 2.0, 1.0]}
    desc1 = ModelDesc.from_formula("~ counter(x)")
    desc2 = ModelDesc.from_formula("~ counter(x) + counter(z)")
    builders = shared_design_matrix_builders(
        [desc1.rhs_termlist, desc2.rhs_termlist], lambda: iter([data]), 0
    )
    assert len(builders) == 2
    plan = _get_plan(builders[0])
    assert plan is _get_plan(builders[1])
    counters = [o for o in plan.transforms.values() if isinstance(o, Counter)]
    # counter(x) shared across the two formulas; counter(z) separate
    assert len(counters) == 2
    native1 = design_matrix_builders(
        [desc1.rhs_termlist], lambda: iter([data]), 0
    )
    native2 = design_matrix_builders(
        [desc2.rhs_termlist], lambda: iter([data]), 0
    )
    got = build_shared_design_matrices(builders, data)
    want = build_design_matrices(list(native1) + list(native2), data)
    for got_m, want_m in zip(got, want):
        assert np.allclose(np.asarray(got_m), np.asarray(want_m))


def test_incremental_multi_pass_over_chunks():
    # nested transforms force multiple passes over an incremental data
    # source; results must match the native incremental builders
    rng = np.random.RandomState(6)
    full = {"x": rng.normal(size=30), "y": rng.normal(size=30)}
    chunks = [
        {k: v[i : i + 10] for k, v in full.items()} for i in range(0, 30, 10)
    ]
    formula = "y ~ bs(center(x), df=4) + center(x)"
    desc = ModelDesc.from_formula(formula)
    shared = shared_design_matrix_builders(
        [desc.lhs_termlist, desc.rhs_termlist], lambda: iter(chunks), 0
    )
    n_lhs, n_rhs = incr_dbuilders(desc, lambda: iter(chunks))
    got = build_shared_design_matrices(shared, full)
    want = build_design_matrices([n_lhs, n_rhs], full)
    for got_m, want_m in zip(got, want):
        assert np.allclose(np.asarray(got_m), np.asarray(want_m))
    assert got[1].design_info.column_names == want[1].design_info.column_names


def test_no_stateful_transforms():
    # formulas without any stateful transforms work unchanged
    data = {
        "x": np.asarray([1.0, 2.0, 3.0]),
        "a": ["a", "b", "a"],
        "y": np.asarray([1.0, 2.0, 3.0]),
    }
    formula = "y ~ x + C(a) + np.log(x) + I(x ** 2)"
    shared = _shared_builders(formula, data)
    native = _native_builders(formula, data)
    assert shared[1].column_names == native[1].column_names
    got = build_shared_design_matrices(shared, data)
    want = build_design_matrices(native, data)
    for got_m, want_m in zip(got, want):
        assert np.allclose(np.asarray(got_m), np.asarray(want_m))


def test_intercept_only_and_empty():
    data = {"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]}
    shared = _shared_builders("y ~ 1", data)
    native = _native_builders("y ~ 1", data)
    got = build_shared_design_matrices(shared, data)
    want = build_design_matrices(native, data)
    for got_m, want_m in zip(got, want):
        assert np.allclose(np.asarray(got_m), np.asarray(want_m))
        assert got_m.shape == want_m.shape


def test_errors_match_native():
    # bad return_type
    data = {"x": [1.0, 2.0], "y": [1.0, 2.0]}
    shared = _shared_builders("y ~ center(x)", data)
    pytest.raises(
        PatsyError, build_shared_design_matrices, shared, data, return_type="foo"
    )
    # malformed input to the builders fails the same way as the native
    # builders (same code path)
    for fn in (shared_design_matrix_builders, design_matrix_builders):
        pytest.raises(AttributeError, fn, "not a termlist", lambda: iter([]), 0)
    # a bad formula still produces a PatsyError
    pytest.raises(PatsyError, _shared_builders, "y ~ center(x", data)
    pytest.raises(PatsyError, _shared_builders, "y ~ no_such_function(x)", data)


def test_custom_factor_objects():
    # Factors implementing the low-level factor protocol directly (rather
    # than being EvalFactor expressions) participate in shared builds via
    # the regular per-factor protocol, and everything still works.
    class CustomFactor(object):
        # centers its input using state memorized over the data
        def __init__(self, varname):
            self._varname = varname
            self.origin = None
            self.memorize_chunks = 0

        def name(self):
            return "custom(%s)" % (self._varname,)

        def memorize_passes_needed(self, state, eval_env):
            state["sum"] = 0.0
            state["n"] = 0
            return 1

        def memorize_chunk(self, state, which_pass, data):
            self.memorize_chunks += 1
            state["sum"] += np.sum(np.asarray(data[self._varname], dtype=float))
            state["n"] += len(data[self._varname])

        def memorize_finish(self, state, which_pass):
            pass

        def eval(self, state, data):
            return np.asarray(data[self._varname], dtype=float) - (
                state["sum"] / state["n"]
            )

    data = {"x": [1.0, 2.0, 3.0], "y": [3.0, 2.0, 1.0]}
    f = CustomFactor("x")
    termlists = [
        [Term([EvalFactor("y")])],
        [INTERCEPT, Term([f]), Term([EvalFactor("center(x)")])],
    ]
    shared = shared_design_matrix_builders(termlists, lambda: iter([data]), 0)
    assert f.memorize_chunks == 1
    native = design_matrix_builders(termlists, lambda: iter([data]), 0)
    got = build_shared_design_matrices(shared, data)
    want = build_design_matrices(native, data)
    for got_m, want_m in zip(got, want):
        assert np.allclose(np.asarray(got_m), np.asarray(want_m))
    assert got[1].design_info.column_names == want[1].design_info.column_names


def test_dmatrix_etc_unchanged():
    # the classic high-level API is untouched and produces the same results
    # as before (sanity check that the refactor changed nothing)
    rng = np.random.RandomState(8)
    data = {"x": rng.normal(size=25), "y": rng.normal(size=25)}
    y, x = dmatrices("y ~ center(x) + bs(x, df=4)", data)
    m = dmatrix("center(x) + bs(x, df=4)", data)
    assert np.allclose(np.asarray(x), np.asarray(m))
    assert x.design_info.column_names == m.design_info.column_names
