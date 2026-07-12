"""2D evaluator for the compiled backend.

All component functions live in ``trspecfit.functions.energy`` as the
single source of truth.  Peak functions broadcast naturally with
``(n_time, 1)`` params and ``(1, n_energy)`` energy.  Background
functions (Offset, LinBack, Shirley) accept optional or axis-agnostic
signatures that work for both 1D and 2D evaluation.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from trspecfit.functions import time as fcts_time
from trspecfit.graph_ir import (
    OP_DISPATCH,
    PROFILE_DISPATCH,
    ConvKernelKind,
    DynFuncKind,
    ExprNodeKind,
    ParamSourceKind,
    ScheduledPlan2D,
)
from trspecfit.utils.arrays import ConvOperator, conv_matrix_apply

# ---------------------------------------------------------------------------
# Dynamics dispatch table
# ---------------------------------------------------------------------------

DYNAMICS_DISPATCH: dict[int, tuple] = {
    DynFuncKind.EXPFUN: (fcts_time.expFun, 3),
    DynFuncKind.SINFUN: (fcts_time.sinFun, 4),
    DynFuncKind.LINFUN: (fcts_time.linFun, 2),
    DynFuncKind.SINDIVX: (fcts_time.sinDivX, 3),
    DynFuncKind.ERFFUN: (fcts_time.erfFun, 3),
    DynFuncKind.SQRTFUN: (fcts_time.sqrtFun, 2),
    DynFuncKind.STEPFUN: (fcts_time.stepFun, 2),
}

# Convolution kernel dispatch.  Kernel functions are elementwise in
# their first argument, so they evaluate directly on the precomputed
# deduplicated dt values of the kernel-matrix operator
# (plan.conv_operator.dt_unique).  Mirrors MCP's Component.convolve.
CONV_KERNEL_DISPATCH: dict[int, Callable] = {
    ConvKernelKind.GAUSSCONV: fcts_time.gaussCONV,
    ConvKernelKind.EXPSYMCONV: fcts_time.expSymCONV,
    ConvKernelKind.EXPDECAYCONV: fcts_time.expDecayCONV,
    ConvKernelKind.EXPRISECONV: fcts_time.expRiseCONV,
    ConvKernelKind.BOXCONV: fcts_time.boxCONV,
}

# Edge-mass dispatch: exact analytic exterior masses per kernel
# (edge-value padding), keyed by the same enum as the kernel dispatch.
# Callables live in functions/time.py (CONV_EDGE_MASS); the plan itself
# stores only numeric kernel ids, keeping it serializable.
CONV_EDGE_MASS_DISPATCH: dict[int, Callable] = {
    ConvKernelKind.GAUSSCONV: fcts_time.CONV_EDGE_MASS["gaussCONV"],
    ConvKernelKind.EXPSYMCONV: fcts_time.CONV_EDGE_MASS["expSymCONV"],
    ConvKernelKind.EXPDECAYCONV: fcts_time.CONV_EDGE_MASS["expDecayCONV"],
    ConvKernelKind.EXPRISECONV: fcts_time.CONV_EDGE_MASS["expRiseCONV"],
    ConvKernelKind.BOXCONV: fcts_time.CONV_EDGE_MASS["boxCONV"],
}


# ---------------------------------------------------------------------------
# Shared RPN expression evaluator
# ---------------------------------------------------------------------------


#
def eval_expr_program(
    instructions: np.ndarray,
    traces: np.ndarray,
) -> np.ndarray:
    """Evaluate a compiled RPN program against the trace matrix.

    Works for both plan initialization and hot-path evaluation.
    Each PARAM_REF pushes a *view* of its ``(n_time,)`` trace row and
    constants stay scalar; every operator allocates a fresh array, so
    the views are never written to. Callers must not mutate the result
    in place (it may alias a *traces* row).

    Parameters
    ----------
    instructions
        Compiled RPN instruction array (one program's slice of the
        packed ``expr_instructions``).
    traces
        ``(n_params, n_time)`` trace matrix (current state).

    Returns
    -------
    ndarray
        ``(n_time,)`` result.
    """

    n_time = traces.shape[1]
    stack: list[np.ndarray | np.float64] = []
    instr = instructions
    n_instr = len(instr) // 2

    for i in range(n_instr):
        kind = int(instr[2 * i])
        operand = instr[2 * i + 1]

        if kind == ExprNodeKind.CONST:
            stack.append(np.int64(operand).view(np.float64))

        elif kind == ExprNodeKind.PARAM_REF:
            stack.append(traces[int(operand), :])

        elif kind == ExprNodeKind.ADD:
            b, a = stack.pop(), stack.pop()
            stack.append(a + b)

        elif kind == ExprNodeKind.SUB:
            b, a = stack.pop(), stack.pop()
            stack.append(a - b)

        elif kind == ExprNodeKind.MUL:
            b, a = stack.pop(), stack.pop()
            stack.append(a * b)

        elif kind == ExprNodeKind.DIV:
            b, a = stack.pop(), stack.pop()
            stack.append(a / b)

        elif kind == ExprNodeKind.NEG:
            stack.append(-stack.pop())

        elif kind == ExprNodeKind.POW:
            b, a = stack.pop(), stack.pop()
            stack.append(a**b)

    assert len(stack) == 1
    result = stack[0]
    if not isinstance(result, np.ndarray):  # constant-only program
        return np.full(n_time, float(result), dtype=np.float64)
    return result


# ---------------------------------------------------------------------------
# Shared trace resolution
# ---------------------------------------------------------------------------


#
def resolve_param_traces(
    traces: np.ndarray,
    resolution_kinds: np.ndarray,
    resolution_indices: np.ndarray,
    dyn_group_target_row: np.ndarray,
    dyn_group_base_row: np.ndarray,
    dyn_group_indptr: np.ndarray,
    dyn_sub_func_id: np.ndarray,
    dyn_sub_n_params: np.ndarray,
    dyn_sub_param_rows: np.ndarray,
    dyn_sub_time_axes: np.ndarray,
    dyn_sub_masks: np.ndarray,
    expr_target_rows: np.ndarray,
    expr_instructions: np.ndarray,
    expr_indptr: np.ndarray,
    conv_target_rows: np.ndarray,
    conv_func_ids: np.ndarray,
    conv_param_indptr: np.ndarray,
    conv_param_rows: np.ndarray,
    conv_operator: ConvOperator | None,
) -> None:
    """Resolve dynamics, expressions, and convolutions into *traces* in place.

    Dynamics groups, expressions, and resolved-trace convolutions are
    interleaved in topological order so that downstream consumers see the
    fully resolved trace (base + dynamics + expressions + IRF).  A dynamics
    group evaluates all substeps (e.g. two expFun in a bi-exponential) and
    sums them: target = base + sum(traces).  Expression-valued dynamics
    params are resolved before the group that consumes them.

    Shared between the hot path (``evaluate_2d``) and compile-time trace
    initialization in ``schedule_2d``.
    """

    for step in range(len(resolution_kinds)):
        kind = int(resolution_kinds[step])
        idx = int(resolution_indices[step])
        if kind == 0:  # dynamics group
            target = int(dyn_group_target_row[idx])
            base = int(dyn_group_base_row[idx])
            traces[target, :] = traces[base, :]
            s_start = int(dyn_group_indptr[idx])
            s_end = int(dyn_group_indptr[idx + 1])
            for s in range(s_start, s_end):
                func_id = int(dyn_sub_func_id[s])
                func, _n_par = DYNAMICS_DISPATCH[func_id]
                n_par = int(dyn_sub_n_params[s])
                param_rows = dyn_sub_param_rows[s, :n_par]
                # Reading t=0 is exact: substep and kernel param rows are
                # time-constant by construction — dynamics-model expressions
                # cannot reference cross-model (potentially time-varying)
                # parameters; add_dynamics rejects them.
                dyn_params = [float(traces[int(row), 0]) for row in param_rows]
                traces[target, :] += (
                    func(dyn_sub_time_axes[s], *dyn_params) * dyn_sub_masks[s]
                )
        elif kind == 1:  # expression
            target = int(expr_target_rows[idx])
            traces[target, :] = eval_expr_program(
                expr_instructions[expr_indptr[idx] : expr_indptr[idx + 1]],
                traces,
            )
        else:  # kind == 2: resolved-trace convolution
            assert conv_operator is not None  # type guard: set when steps exist
            target = int(conv_target_rows[idx])
            func_id = int(conv_func_ids[idx])
            kernel_func = CONV_KERNEL_DISPATCH[func_id]
            edge_mass_func = CONV_EDGE_MASS_DISPATCH[func_id]
            p_start = int(conv_param_indptr[idx])
            p_end = int(conv_param_indptr[idx + 1])
            # t=0 read is exact; same time-constant invariant as above
            kernel_params = [
                float(traces[int(conv_param_rows[j]), 0]) for j in range(p_start, p_end)
            ]
            # companion first: it validates the parameters, so the
            # kernel body never sees a nonpositive width
            mass_left, mass_right = edge_mass_func(
                conv_operator.dt_left, conv_operator.dt_right, *kernel_params
            )
            kernel_values = kernel_func(conv_operator.dt_unique, *kernel_params)
            traces[target, :] = conv_matrix_apply(
                conv_operator, kernel_values, mass_left, mass_right, traces[target, :]
            )


# ---------------------------------------------------------------------------
# Profile evaluation helpers (2D)
# ---------------------------------------------------------------------------


#
def _evaluate_profile_sample_values_2d(
    aux_axis: np.ndarray,
    traces: np.ndarray,
    profile_sample_base_rows: np.ndarray,
    profile_sample_component_indptr: np.ndarray,
    profile_component_func_ids: np.ndarray,
    profile_component_param_indptr: np.ndarray,
    profile_component_param_rows: np.ndarray,
) -> np.ndarray:
    """Evaluate lowered PROFILE_SAMPLE groups into ``(n_groups, n_time, n_aux)``.

    Profile functions broadcast naturally: ``aux_axis`` is shaped
    ``(1, n_aux)`` and each param trace is ``(n_time, 1)``, yielding
    ``(n_time, n_aux)`` per function call.
    """

    n_groups = len(profile_sample_base_rows)
    n_time = traces.shape[1]
    n_aux = len(aux_axis)
    if n_groups == 0:
        return np.zeros((0, n_time, n_aux), dtype=np.float64)

    aux_2d = aux_axis[np.newaxis, :]  # (1, n_aux)
    sample_values = np.empty((n_groups, n_time, n_aux), dtype=np.float64)

    for group_idx in range(n_groups):
        base_row = int(profile_sample_base_rows[group_idx])
        # base trace -> (n_time, 1) broadcast into the output row
        values = sample_values[group_idx]
        values[:] = traces[base_row, :][:, np.newaxis]

        comp_start = int(profile_sample_component_indptr[group_idx])
        comp_end = int(profile_sample_component_indptr[group_idx + 1])
        for comp_idx in range(comp_start, comp_end):
            func = PROFILE_DISPATCH[int(profile_component_func_ids[comp_idx])]
            param_start = int(profile_component_param_indptr[comp_idx])
            param_end = int(profile_component_param_indptr[comp_idx + 1])
            params = [
                traces[int(row), :][:, np.newaxis]  # (n_time, 1)
                for row in profile_component_param_rows[param_start:param_end]
            ]
            values += np.asarray(func(aux_2d, *params), dtype=np.float64)

    return sample_values


#
def _evaluate_profile_expr_values_2d(
    traces: np.ndarray,
    profile_sample_values: np.ndarray,
    n_params: int,
    profile_expr_instructions: np.ndarray,
    profile_expr_indptr: np.ndarray,
) -> np.ndarray:
    """Evaluate lowered per-sample profile expressions over (n_time, n_aux).

    Builds a virtual trace matrix ``(n_params + n_groups, n_time * n_aux)``
    so the standard RPN evaluator can be reused unchanged.
    """

    n_exprs = len(profile_expr_indptr) - 1
    if n_exprs == 0:
        n_time = traces.shape[1]
        n_aux = profile_sample_values.shape[2] if profile_sample_values.size else 0
        return np.zeros((0, n_time, n_aux), dtype=np.float64)

    n_time = traces.shape[1]
    n_aux = profile_sample_values.shape[2]
    n_groups = profile_sample_values.shape[0]
    n_cols = n_time * n_aux

    # Virtual trace: regular params repeated across aux, profile samples
    # flattened from (n_groups, n_time, n_aux) -> (n_groups, n_time*n_aux).
    virtual = np.empty((n_params + n_groups, n_cols), dtype=np.float64)
    # broadcast-write params across aux in place (no np.repeat temporary)
    virtual_params = virtual[:n_params, :].reshape(n_params, n_time, n_aux)
    virtual_params[:] = traces[:, :, np.newaxis]
    if n_groups > 0:
        virtual[n_params:, :] = profile_sample_values.reshape(n_groups, n_cols)

    expr_values = np.empty((n_exprs, n_time, n_aux), dtype=np.float64)
    for expr_idx in range(n_exprs):
        start = int(profile_expr_indptr[expr_idx])
        end = int(profile_expr_indptr[expr_idx + 1])
        result = eval_expr_program(
            profile_expr_instructions[start:end], virtual
        )  # (n_time*n_aux,)
        expr_values[expr_idx] = result.reshape(n_time, n_aux)

    return expr_values


#
def _evaluate_profiled_op_2d(
    energy: np.ndarray,
    kind: int,
    param_source_kinds: np.ndarray,
    param_indices: np.ndarray,
    traces: np.ndarray,
    profile_sample_values: np.ndarray,
    profile_expr_values: np.ndarray,
    peak_sum: np.ndarray,
    *,
    needs_spectrum: bool,
    n_aux: int,
) -> np.ndarray:
    """Evaluate one profiled 2D op: loop over aux points, average.

    Param sources are resolved to ``(n_time, n_aux)`` views once,
    outside the loop. The per-aux loop is deliberate: vectorizing over
    aux in a single call only wins when profiled params enter the
    function linearly (amplitude-only profiles, where broadcasting
    keeps the transcendental part at ``(n_time, 1, n_energy)``); with a
    profiled position or width the energy function materializes full
    ``(n_time, n_aux, n_energy)`` temporaries and measures ~60% slower
    (example 04: profiled x0, n_aux=50, 175x280 grid).
    """

    func, _needs = OP_DISPATCH[kind]
    n_time = traces.shape[1]
    n_energy = energy.shape[-1]

    # Resolve each param source once (scalars as no-copy broadcast views)
    sources: list[np.ndarray] = []
    for source_kind, source_idx in zip(
        param_source_kinds,
        param_indices,
        strict=True,
    ):
        sk = int(source_kind)
        si = int(source_idx)
        if sk == int(ParamSourceKind.SCALAR):
            source = np.broadcast_to(traces[si, :][:, np.newaxis], (n_time, n_aux))
        elif sk == int(ParamSourceKind.PROFILE_SAMPLE):
            source = profile_sample_values[si]  # (n_time, n_aux)
        else:
            source = profile_expr_values[si]  # (n_time, n_aux)
        sources.append(source)

    accumulated = np.zeros((n_time, n_energy), dtype=np.float64)
    for aux_i in range(n_aux):
        params = [s[:, aux_i, np.newaxis] for s in sources]
        if needs_spectrum:
            accumulated += func(energy, *params, peak_sum)
        else:
            accumulated += func(energy, *params)

    accumulated /= n_aux
    return accumulated


# ---------------------------------------------------------------------------
# Core 2D evaluator
# ---------------------------------------------------------------------------


#
def evaluate_2d(plan: ScheduledPlan2D, theta: np.ndarray) -> np.ndarray:
    """Evaluate the compiled 2D model at optimizer parameters *theta*.

    Parameters
    ----------
    plan
        Immutable compiled execution schedule from ``schedule_2d``.
    theta
        ``(n_opt,)`` optimizer parameter vector.  Order must match
        ``plan.opt_param_names``.

    Returns
    -------
    ndarray
        ``(n_time, n_energy)`` model spectrum.

    Raises
    ------
    ValueError
        If ``len(theta) != len(plan.opt_indices)``.
    """

    # --- theta contract check ---
    if len(theta) != len(plan.opt_indices):
        raise ValueError(
            f"theta length {len(theta)} does not match "
            f"plan.opt_indices length {len(plan.opt_indices)}"
        )

    # 1a. Copy trace matrix -> scratch
    traces = plan.param_traces_init.copy()

    # 1b. Broadcast optimizer params
    traces[plan.opt_indices, :] = theta[:, np.newaxis]

    # 1c+d. Resolve dynamics groups, expressions, and trace convolutions
    # in interleaved topological order.
    resolve_param_traces(
        traces,
        plan.resolution_kinds,
        plan.resolution_indices,
        plan.dyn_group_target_row,
        plan.dyn_group_base_row,
        plan.dyn_group_indptr,
        plan.dyn_sub_func_id,
        plan.dyn_sub_n_params,
        plan.dyn_sub_param_rows,
        plan.dyn_sub_time_axes,
        plan.dyn_sub_masks,
        plan.expr_target_rows,
        plan.expr_instructions,
        plan.expr_indptr,
        plan.conv_target_rows,
        plan.conv_func_ids,
        plan.conv_param_indptr,
        plan.conv_param_rows,
        plan.conv_operator,
    )

    # 1e. Profile evaluation (after parameter resolution).
    profile_sample_values = _evaluate_profile_sample_values_2d(
        plan.aux_axis,
        traces,
        plan.profile_sample_base_rows,
        plan.profile_sample_component_indptr,
        plan.profile_component_func_ids,
        plan.profile_component_param_indptr,
        plan.profile_component_param_rows,
    )
    profile_expr_values = _evaluate_profile_expr_values_2d(
        traces,
        profile_sample_values,
        plan.n_params,
        plan.profile_expr_instructions,
        plan.profile_expr_indptr,
    )

    # 2. Component evaluation
    energy = plan.energy[np.newaxis, :]  # (1, n_energy)
    result = plan.cached_result.copy()
    peak_sum = plan.cached_peak_sum.copy()

    for op_idx in range(plan.n_ops):
        if plan.op_is_constant[op_idx]:
            continue
        kind = int(plan.op_kinds[op_idx])
        start = int(plan.op_param_indptr[op_idx])
        end = int(plan.op_param_indptr[op_idx + 1])
        needs_spectrum = bool(plan.op_needs_spectrum[op_idx])
        is_pre = bool(plan.op_is_pre_spectrum[op_idx])

        if plan.op_is_profiled[op_idx]:
            component = _evaluate_profiled_op_2d(
                energy,
                kind,
                plan.op_param_source_kinds[start:end],
                plan.op_param_indices[start:end],
                traces,
                profile_sample_values,
                profile_expr_values,
                peak_sum,
                needs_spectrum=needs_spectrum,
                n_aux=plan.n_aux,
            )
        else:
            param_rows = plan.op_param_indices[start:end]
            # Gather params as (n_time, 1) columns
            params: list[np.ndarray] = [
                traces[int(row), :][:, np.newaxis] for row in param_rows
            ]
            func, _needs = OP_DISPATCH[kind]
            if needs_spectrum:
                component = func(energy, *params, peak_sum)
            else:
                component = func(energy, *params)

        result += component
        if is_pre:
            peak_sum += component

    return result
