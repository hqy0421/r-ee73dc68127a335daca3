# This file is part of Patsy
# Copyright (C) 2011-2015 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

# This file defines the optional "shared build" interface, an alternative to
# design_matrix_builders()/build_design_matrices(). Within a single shared
# fit, stateful transform subexpressions (like center(x) or bs(x, df=5)) that
# have identical input expressions and identical literal arguments are
# evaluated -- and memorized -- only once, no matter how many different
# factors they appear in. Nested transforms still memorize in dependency
# order, and no transform ever sees the same row of data twice. The resulting
# DesignInfo objects are ordinary DesignInfo objects: they can be used with
# the regular build_design_matrices(), with DesignInfo.subset(), or with the
# batch entry point build_shared_design_matrices() defined here, which
# additionally evaluates each shared subexpression only once per batch.

# These are made available in the patsy.* namespace
__all__ = ["shared_design_matrix_builders", "build_shared_design_matrices"]

import numpy as np

from patsy import PatsyError
from patsy.build import (
    _build_design_matrices,
    _design_infos_from_factor_states,
    _eval_factor,
    _finish_eval_factor,
)
from patsy.compat import call_and_wrap_exc
from patsy.eval import (
    EvalEnvironment,
    EvalFactor,
    VarLookupDict,
    _builtins_dict,
    annotated_tokens,
    ast_names,
    capture_obj_method_calls,
    has_bare_variable_reference,
    replace_bare_funcalls,
)
from patsy.missing import NAAction
from patsy.tokens import pretty_untokenize
from patsy.util import have_pandas


def _replace_bare_refs(code, mapping):
    # Replaces references of the form '<name>.<something>' -- which is the
    # only way our generated transform-object names ever appear -- leaving
    # other uses of the same names (e.g. keyword-argument names) alone.
    if not mapping:
        return code
    tokens = list(annotated_tokens(code))
    out = []
    for i, (token_type, token, origin, props) in enumerate(tokens):
        if (
            props["bare_ref"]
            and token in mapping
            and i + 1 < len(tokens)
            and tokens[i + 1][1] == "."
        ):
            token = mapping[token]
        out.append((token_type, token))
    return pretty_untokenize(out)


def test__replace_bare_refs():
    assert _replace_bare_refs("a.transform(x) + b.c(d)", {"a": "g"}) == (
        "g.transform(x) + b.c(d)"
    )
    assert _replace_bare_refs("foo.bar(foo.baz(quux))", {"foo": "g"}) == (
        "g.bar(g.baz(quux))"
    )
    # plain variable references and keyword-argument names are untouched
    assert _replace_bare_refs("aa + a", {"a": "b"}) == "aa + a"
    assert _replace_bare_refs("f(a=a.b)", {"a": "X"}) == "f(a=X.b)"


class _SharedTransformPlan(object):
    """Holds the stateful-transform pool for a single shared fit.

    A fresh plan is created by every call to
    :func:`shared_design_matrix_builders`, and is reachable from the
    resulting :class:`DesignInfo` objects (via their factor states) so that
    :func:`build_shared_design_matrices` can find it again. Because plans are
    never reused across fits, memorized state cannot leak from one fit (or
    one data batch) into another.
    """

    def __init__(self):
        # (transform class, canonical argument code) -> shared object name.
        # The canonical argument code has nested stateful transforms already
        # rewritten to use their shared names, so that, e.g., the two
        # occurrences of bs(center(x), df=5) in different factors produce
        # textually identical keys, while bs(x, df=5) and bs(x, df=6) do
        # not share.
        self._registry = {}
        self._counter = 0
        # shared object name -> stateful transform object
        self.transforms = {}
        # shared object name -> code for '<name>.transform(<args>)'
        self.transform_call_code = {}
        # shared object name -> code for '<name>.memorize_chunk(<args>)'
        self.memorize_code = {}
        # shared object name -> factor/eval_env that first introduced it
        # (used for error messages, and to evaluate the code above in the
        # same environment that the regular builder would have used)
        self.obj_origins = {}
        self.obj_eval_envs = {}
        # list of sets of shared object names, in dependency order: every
        # object in bin i depends only on objects in bins < i.
        self.pass_bins = []

    def shared_object_for(self, key, token, factor, eval_env):
        if key in self._registry:
            return self._registry[key]
        transform_class, args_code = key
        obj_name = "_patsy_shared%s__%s__" % (self._counter, token)
        self._counter += 1
        self._registry[key] = obj_name
        self.transforms[obj_name] = transform_class()
        self.transform_call_code[obj_name] = obj_name + ".transform" + args_code
        self.memorize_code[obj_name] = obj_name + ".memorize_chunk" + args_code
        self.obj_origins[obj_name] = factor
        self.obj_eval_envs[obj_name] = eval_env
        return obj_name

    def finish_setup(self):
        # Sort the shared objects into dependency bins. This is the same
        # binning that EvalFactor.memorize_passes_needed performs within a
        # single factor, applied here to the whole shared pool at once, so
        # that nested transforms (e.g. the center() inside
        # bs(center(x), df=5)) are memorized in an earlier pass than their
        # consumers no matter which factors they came from.
        pass_bins = []
        unsorted = set(self.memorize_code)
        while unsorted:
            pass_bin = set()
            for obj_name in unsorted:
                other_objs = unsorted.difference([obj_name])
                if not has_bare_variable_reference(
                    other_objs, self.memorize_code[obj_name]
                ):
                    pass_bin.add(obj_name)
            assert pass_bin
            unsorted.difference_update(pass_bin)
            pass_bins.append(pass_bin)
        self.pass_bins = pass_bins

    def eval_memorize_chunk(self, obj_name, data):
        inner_namespace = VarLookupDict([data, self.transforms])
        return call_and_wrap_exc(
            "Error evaluating factor",
            self.obj_origins[obj_name],
            self.obj_eval_envs[obj_name].eval,
            self.memorize_code[obj_name],
            inner_namespace=inner_namespace,
        )


def _shared_memorize_setup(factor, state, eval_env, plan):
    # This plays the same role as EvalFactor.memorize_passes_needed, except
    # that instead of every call site getting its own fresh transform
    # object, call sites with identical input expressions and literal
    # arguments are mapped onto a single shared object from `plan`.
    eval_env = eval_env.with_outer_namespace(_builtins_dict)
    env_namespace = eval_env.namespace
    subset_names = [name for name in ast_names(factor.code) if name in env_namespace]
    state["eval_env"] = eval_env.subset(subset_names)

    # Phase 1: give every stateful-transform call site in this factor a
    # unique temporary name (we cannot pick final names yet, because the
    # sharing key for an outer call mentions the shared names of the inner
    # calls, which are not known until the inner calls are resolved).
    temp_counter = [0]
    temp_objs = {}  # temp name -> (token, transform class)

    def temp_name_maker(token):
        value = env_namespace.get(token)
        if hasattr(value, "__patsy_stateful_transform__"):
            temp_name = "_patsy_sharedtmp%s__%s__" % (temp_counter[0], token)
            temp_counter[0] += 1
            temp_objs[temp_name] = (token, value.__patsy_stateful_transform__)
            return temp_name + ".transform"
        else:
            return token

    temp_code = replace_bare_funcalls(factor.code, temp_name_maker)
    # paranoia (mirrors EvalFactor.memorize_passes_needed): our generated
    # names must not collide with names the user wrote themselves
    if has_bare_variable_reference(temp_objs, factor.code):
        raise PatsyError("names of this form are reserved for internal use", factor)

    # Phase 2: map each temporary name onto a shared object, resolving
    # objects that contain nested stateful transforms inside-out, so that
    # the argument code used as the sharing key is canonical.
    remaining = {}  # temp name -> '<temp>.transform(<args>)' code
    for temp_name in temp_objs:
        transform_calls = capture_obj_method_calls(temp_name, temp_code)
        assert len(transform_calls) == 1
        call_name, call_code = transform_calls[0]
        assert call_name == temp_name + ".transform"
        remaining[temp_name] = call_code
    temp_to_shared = {}
    while remaining:
        ready = [
            temp_name
            for temp_name, call_code in remaining.items()
            if not has_bare_variable_reference(
                [other for other in remaining if other != temp_name], call_code
            )
        ]
        assert ready
        for temp_name in ready:
            call_code = remaining.pop(temp_name)
            # Canonicalize: refer to nested transforms by their shared names
            call_code = _replace_bare_refs(call_code, temp_to_shared)
            token, transform_class = temp_objs[temp_name]
            prefix = temp_name + ".transform"
            assert call_code.startswith(prefix + "(")
            args_code = call_code[len(prefix) :]
            key = (transform_class, args_code)
            temp_to_shared[temp_name] = plan.shared_object_for(
                key, token, factor, state["eval_env"]
            )
    if has_bare_variable_reference(set(temp_to_shared.values()), factor.code):
        raise PatsyError("names of this form are reserved for internal use", factor)

    state["transforms"] = plan.transforms
    state["eval_code"] = _replace_bare_refs(temp_code, temp_to_shared)
    state["shared_plan"] = plan
    # The shared objects this factor actually refers to; used to avoid
    # computing unneeded shared values when building from a subset of terms.
    state["shared_objs"] = frozenset(temp_to_shared.values())


def _shared_factors_memorize(factors, data_iter_maker, eval_env):
    # A drop-in replacement for patsy.build._factors_memorize in which
    # EvalFactor factors draw their stateful transforms from a single shared
    # pool, so each distinct subexpression memorizes each data chunk exactly
    # once, no matter how many factors contain it.
    plan = _SharedTransformPlan()
    factor_states = {}
    other_factors_passes = {}
    for factor in factors:
        state = {}
        if isinstance(factor, EvalFactor):
            _shared_memorize_setup(factor, state, eval_env, plan)
        else:
            # Non-EvalFactor factors (e.g. user-defined factor objects in a
            # hand-built ModelDesc) use the regular per-factor protocol.
            passes = factor.memorize_passes_needed(state, eval_env)
            if passes > 0:
                other_factors_passes[factor] = passes
        factor_states[factor] = state
    plan.finish_setup()
    which_pass = 0
    memorize_needed = set(other_factors_passes)
    while which_pass < len(plan.pass_bins) or memorize_needed:
        if which_pass < len(plan.pass_bins):
            shared_bin = plan.pass_bins[which_pass]
        else:
            shared_bin = ()
        for data in data_iter_maker():
            for obj_name in shared_bin:
                plan.eval_memorize_chunk(obj_name, data)
            for factor in memorize_needed:
                factor.memorize_chunk(factor_states[factor], which_pass, data)
        for obj_name in shared_bin:
            plan.transforms[obj_name].memorize_finish()
        for factor in list(memorize_needed):
            factor.memorize_finish(factor_states[factor], which_pass)
            if which_pass == other_factors_passes[factor] - 1:
                memorize_needed.remove(factor)
        which_pass += 1
    return factor_states


def shared_design_matrix_builders(
    termlists, data_iter_maker, eval_env, NA_action="drop"
):
    """Construct several :class:`DesignInfo` objects from termlists, sharing
    stateful transforms between factors.

    This is a drop-in alternative to :func:`design_matrix_builders`; see that
    function's documentation for the meaning of the arguments. The only
    difference in behavior is that within this one call, stateful transform
    subexpressions with identical input expressions and identical literal
    arguments -- e.g. two occurrences of ``center(x)``, or of
    ``bs(x, df=5)``, appearing in any of the factors of any of the termlists
    -- share a single transform object. The shared object memorizes each
    data chunk exactly once (nested transforms still memorize in dependency
    order, in multiple passes if needed), so the resulting learned state
    (means, standard deviations, spline knots, ...) is identical to what
    :func:`design_matrix_builders` would produce.

    :returns: A list of ordinary :class:`DesignInfo` objects, one for each
      termlist passed in. They can be used with
      :func:`build_design_matrices`, with :meth:`DesignInfo.subset`, and
      with :func:`build_shared_design_matrices`.

    Each call to this function performs an independent fit: no memorized
    state is shared with, or leaked into, any other call.

    .. versionadded:: 1.1.0
    """
    eval_env = EvalEnvironment.capture(eval_env, reference=1)
    if isinstance(NA_action, str):
        NA_action = NAAction(NA_action)
    all_factors = set()
    for termlist in termlists:
        for term in termlist:
            all_factors.update(term.factors)
    factor_states = _shared_factors_memorize(all_factors, data_iter_maker, eval_env)
    return _design_infos_from_factor_states(
        termlists, all_factors, factor_states, data_iter_maker, NA_action
    )


class _PrecomputedTransform(object):
    # Stands in for a stateful transform object whose value for the current
    # batch has already been computed; .transform() simply returns it. The
    # arguments are ignored -- they were already evaluated once, inside the
    # single shared evaluation that produced the value.
    def __init__(self, value):
        self._value = value

    def transform(self, *args, **kwargs):
        return self._value


def _precompute_shared_values(plan, needed, data):
    # Evaluate each needed shared subexpression exactly once for this batch,
    # in dependency order (innermost transforms first). Names that have not
    # been computed yet (including the object currently being evaluated)
    # resolve to the real transform objects; names that have been computed
    # resolve to _PrecomputedTransform wrappers, so each shared
    # subexpression is only ever evaluated once. The results live only in
    # this local dict, so nothing can leak into the next batch.
    wrappers = dict(plan.transforms)
    for pass_bin in plan.pass_bins:
        for obj_name in pass_bin:
            if obj_name not in needed:
                continue
            inner_namespace = VarLookupDict([data, wrappers])
            value = call_and_wrap_exc(
                "Error evaluating factor",
                plan.obj_origins[obj_name],
                plan.obj_eval_envs[obj_name].eval,
                plan.transform_call_code[obj_name],
                inner_namespace=inner_namespace,
            )
            wrappers[obj_name] = _PrecomputedTransform(value)
    return wrappers


def build_shared_design_matrices(
    design_infos, data, NA_action="drop", return_type="matrix", dtype=np.dtype(float)
):
    """Construct several design matrices from :class:`DesignInfo` objects,
    evaluating shared subexpressions only once.

    This is a batch-oriented alternative to :func:`build_design_matrices`;
    see that function's documentation for the meaning of the arguments and
    return values. It accepts any :class:`DesignInfo` objects -- including
    ones produced by the regular :func:`design_matrix_builders`, by
    :func:`dmatrix`/:func:`dmatrices`, or by :meth:`DesignInfo.subset` --
    and produces exactly the same matrices as
    :func:`build_design_matrices` would (same columns, same rows, same
    handling of missing values and pandas indexes).

    The difference is purely in the amount of work done: for
    :class:`DesignInfo` objects that came from
    :func:`shared_design_matrix_builders`, each shared stateful
    subexpression (e.g. a ``center(x)`` that occurs in several different
    factors) is evaluated only once for this batch of data, and the value is
    reused everywhere it occurs. These computed values are local to this
    call -- they are never cached between calls, so building a new batch of
    data always reflects that batch's data.

    .. versionadded:: 1.1.0
    """
    if isinstance(NA_action, str):
        NA_action = NAAction(NA_action)
    if return_type == "dataframe" and not have_pandas:
        raise PatsyError("pandas.DataFrame was requested, but pandas is not installed")
    if return_type not in ("matrix", "dataframe"):
        raise PatsyError(
            "unrecognized output type %r, should be "
            "'matrix' or 'dataframe'" % (return_type,)
        )
    # Find the shared plans (if any) carried by these DesignInfo objects,
    # and work out which shared subexpressions the requested factors
    # actually need (a DesignInfo produced by DesignInfo.subset() may not
    # use all of them).
    plan_to_needed = {}
    for design_info in design_infos:
        for factor_info in design_info.factor_infos.values():
            state = factor_info.state
            if isinstance(state, dict):
                plan = state.get("shared_plan")
                if plan is not None:
                    needed = plan_to_needed.setdefault(plan, set())
                    needed.update(state.get("shared_objs", ()))
    # Evaluate each needed shared subexpression once for this batch.
    plan_to_wrappers = {}
    for plan, needed in plan_to_needed.items():
        plan_to_wrappers[plan] = _precompute_shared_values(plan, needed, data)

    def factor_eval_fn(factor_info):
        state = factor_info.state
        plan = state.get("shared_plan") if isinstance(state, dict) else None
        if plan is None:
            # Factors that did not come from a shared build (native
            # DesignInfo objects, user-defined factor objects, ...) are
            # evaluated exactly as usual.
            return _eval_factor(factor_info, data, NA_action)
        factor = factor_info.factor
        inner_namespace = VarLookupDict([data, plan_to_wrappers[plan]])
        result = call_and_wrap_exc(
            "Error evaluating factor",
            factor,
            state["eval_env"].eval,
            state["eval_code"],
            inner_namespace=inner_namespace,
        )
        return _finish_eval_factor(factor_info, result, NA_action)

    return _build_design_matrices(
        design_infos, data, NA_action, return_type, dtype, factor_eval_fn
    )
