"""2D evaluator for the compiled backend.

All component functions live in ``trspecfit.functions.energy`` as the
single source of truth.  Peak functions broadcast naturally with
``(n_time, 1)`` params and ``(1, n_energy)`` energy.  Background
functions (Offset, LinBack, Shirley) accept optional or axis-agnostic
signatures that work for both 1D and 2D evaluation.
"""

from __future__ import annotations

import numpy as np

from trspecfit.functions import time as fcts_time
from trspecfit.graph_ir import (
    OP_DISPATCH,
    PROFILE_DISPATCH,
    ConvKernelKind,
    DynFuncKind,
    ExprNodeKind,
    ExprProgram,
    ParamSourceKind,
    ScheduledPlan2D,
)
from trspecfit.utils.arrays import my_conv

# ---------------------------------------------------------------------------
# Dynamics dispatch table
# ---------------------------------------------------------------------------

DYNAMICS_DISPATCH: dict[int, tuple] = {
    DynFuncKind.EXPFUN: (fcts_time.expFun, 4),
    DynFuncKind.SINFUN: (fcts_time.sinFun, 5),
    DynFuncKind.LINFUN: (fcts_time.linFun, 3),
    DynFuncKind.SINDIVX: (fcts_time.sinDivX, 4),
    DynFuncKind.ERFFUN: (fcts_time.erfFun, 4),
    DynFuncKind.SQRTFUN: (fcts_time.sqrtFun, 3),
}

# Convolution kernel dispatch: kernel function evaluated on the frozen
# kernel-time support with per-theta kernel parameters.  Mirrors MCP's
# Model.combine(...) path.
CONV_KERNEL_DISPATCH: dict[int, tuple] = {
    ConvKernelKind.GAUSSCONV: (fcts_time.gaussCONV, 1),
}


# ---------------------------------------------------------------------------
# Shared RPN expression evaluator
# ---------------------------------------------------------------------------


#
def eval_expr_program(
    program: ExprProgram,
    traces: np.ndarray,
) -> np.ndarray:
    """Evaluate an RPN ExprProgram against the trace matrix.

    Works for both plan initialization and hot-path evaluation.
    Each PARAM_REF reads a full ``(n_time,)`` row from *traces*;
    constants are broadcast to ``(n_time,)`` via ``np.full``.

    Parameters
    ----------
    program
        Compiled RPN instruction array.
    traces
        ``(n_params, n_time)`` trace matrix (current state).

    Returns
    -------
    ndarray
        ``(n_time,)`` result.
    """

    n_time = traces.shape[1]
    stack: list[np.ndarray] = []
    instr = program.instructions
    n_instr = len(instr) // 2

    for i in range(n_instr):
        kind = ExprNodeKind(instr[2 * i])
        operand = instr[2 * i + 1]

        if kind == ExprNodeKind.CONST:
            val = np.int64(operand).view(np.float64)
            stack.append(np.full(n_time, val, dtype=np.float64))

        elif kind == ExprNodeKind.PARAM_REF:
            stack.append(traces[int(operand), :].copy())

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
    return stack[0]


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
        # base trace -> (n_time, 1) -> broadcast to (n_time, n_aux)
        values = np.broadcast_to(
            traces[base_row, :][:, np.newaxis], (n_time, n_aux)
        ).copy()

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

        sample_values[group_idx] = values

    return sample_values


#
def _evaluate_profile_expr_values_2d(
    traces: np.ndarray,
    profile_sample_values: np.ndarray,
    n_params: int,
    profile_expr_programs: list[ExprProgram],
) -> np.ndarray:
    """Evaluate lowered per-sample profile expressions over (n_time, n_aux).

    Builds a virtual trace matrix ``(n_params + n_groups, n_time * n_aux)``
    so the standard RPN evaluator can be reused unchanged.
    """

    n_exprs = len(profile_expr_programs)
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
    virtual[:n_params, :] = np.repeat(traces, n_aux, axis=1)
    if n_groups > 0:
        virtual[n_params:, :] = profile_sample_values.reshape(n_groups, n_cols)

    expr_values = np.empty((n_exprs, n_time, n_aux), dtype=np.float64)
    for expr_idx, program in enumerate(profile_expr_programs):
        result = eval_expr_program(program, virtual)  # (n_time*n_aux,)
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
    """Evaluate one profiled 2D op: loop over aux points, average."""

    func, _needs = OP_DISPATCH[kind]
    n_time = traces.shape[1]
    n_energy = energy.shape[-1]
    accumulated = np.zeros((n_time, n_energy), dtype=np.float64)

    for aux_i in range(n_aux):
        params: list[np.ndarray] = []
        for source_kind, source_idx in zip(
            param_source_kinds,
            param_indices,
            strict=True,
        ):
            sk = int(source_kind)
            si = int(source_idx)
            if sk == int(ParamSourceKind.SCALAR):
                param = traces[si, :][:, np.newaxis]  # (n_time, 1)
            elif sk == int(ParamSourceKind.PROFILE_SAMPLE):
                param = profile_sample_values[si, :, aux_i][
                    :, np.newaxis
                ]  # (n_time, 1)
            else:
                param = profile_expr_values[si, :, aux_i][:, np.newaxis]  # (n_time, 1)
            params.append(param)

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

    # 1c+d. Resolve dynamics groups and expressions in interleaved topo
    # order.  A dynamics group evaluates all substeps (e.g. two expFun
    # in a bi-exponential) and sums them: target = base + sum(traces).
    # Expression-valued dynamics params are resolved before the group
    # that consumes them.
    for step in range(len(plan.resolution_kinds)):
        kind = int(plan.resolution_kinds[step])
        idx = int(plan.resolution_indices[step])
        if kind == 0:  # dynamics group
            target = int(plan.dyn_group_target_row[idx])
            base = int(plan.dyn_group_base_row[idx])
            traces[target, :] = traces[base, :]
            s_start = int(plan.dyn_group_indptr[idx])
            s_end = int(plan.dyn_group_indptr[idx + 1])
            for s in range(s_start, s_end):
                func_id = int(plan.dyn_sub_func_id[s])
                func, _n_par = DYNAMICS_DISPATCH[func_id]
                n_par = int(plan.dyn_sub_n_params[s])
                param_rows = plan.dyn_sub_param_rows[s, :n_par]
                dyn_params = [float(traces[int(row), 0]) for row in param_rows]
                traces[target, :] += (
                    func(plan.dyn_sub_time_axes[s], *dyn_params) * plan.dyn_sub_masks[s]
                )
        elif kind == 1:  # expression
            target = int(plan.expr_target_rows[idx])
            traces[target, :] = eval_expr_program(plan.expr_programs[idx], traces)
        else:  # kind == 2: resolved-trace convolution
            target = int(plan.conv_target_rows[idx])
            func_id = int(plan.conv_func_ids[idx])
            kernel_func, _k_par = CONV_KERNEL_DISPATCH[func_id]
            p_start = int(plan.conv_param_indptr[idx])
            p_end = int(plan.conv_param_indptr[idx + 1])
            kernel_params = [
                float(traces[int(plan.conv_param_rows[j]), 0])
                for j in range(p_start, p_end)
            ]
            s_start = int(plan.conv_support_indptr[idx])
            s_end = int(plan.conv_support_indptr[idx + 1])
            support = plan.conv_support_values[s_start:s_end]
            kernel = kernel_func(support, *kernel_params)
            traces[target, :] = my_conv(plan.time, traces[target, :], kernel)

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
        plan.profile_expr_programs,
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
