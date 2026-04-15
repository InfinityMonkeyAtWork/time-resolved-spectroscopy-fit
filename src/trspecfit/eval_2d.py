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
    DynFuncKind,
    ExprNodeKind,
    ExprProgram,
    ScheduledPlan2D,
)

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
                traces[target, :] += func(plan.time, *dyn_params)
        else:  # expression
            target = int(plan.expr_target_rows[idx])
            traces[target, :] = eval_expr_program(plan.expr_programs[idx], traces)

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
        param_rows = plan.op_param_indices[start:end]

        # Gather params as (n_time, 1) columns
        params: list[np.ndarray] = [
            traces[int(row), :][:, np.newaxis] for row in param_rows
        ]

        needs_spectrum = bool(plan.op_needs_spectrum[op_idx])
        is_pre = bool(plan.op_is_pre_spectrum[op_idx])

        func, _needs = OP_DISPATCH[kind]
        if needs_spectrum:
            # Spectrum-fed op (Shirley): pass peak_sum as extra arg
            component = func(energy, *params, peak_sum)
        else:
            component = func(energy, *params)

        result += component
        if is_pre:
            peak_sum += component

    return result
