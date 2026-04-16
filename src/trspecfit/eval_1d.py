"""1D evaluator for the compiled backend.

All component functions live in ``trspecfit.functions.energy`` as the
single source of truth.  For 1D evaluation, parameters are plain
scalars (no ``(n_time, 1)`` broadcasting needed).
"""

from __future__ import annotations

import numpy as np

from trspecfit.graph_ir import (
    OP_DISPATCH,
    ExprNodeKind,
    ExprProgram,
    ScheduledPlan1D,
)

# ---------------------------------------------------------------------------
# Scalar RPN expression evaluator
# ---------------------------------------------------------------------------


#
def eval_expr_program_1d(
    program: ExprProgram,
    values: np.ndarray,
) -> float:
    """Evaluate an RPN ExprProgram against a scalar parameter vector.

    Parameters
    ----------
    program
        Compiled RPN instruction array.
    values
        ``(n_params,)`` scalar parameter vector.

    Returns
    -------
    float
        Scalar result.
    """

    stack: list[float] = []
    instr = program.instructions
    n_instr = len(instr) // 2

    for i in range(n_instr):
        kind = ExprNodeKind(instr[2 * i])
        operand = instr[2 * i + 1]

        if kind == ExprNodeKind.CONST:
            stack.append(float(np.int64(operand).view(np.float64)))

        elif kind == ExprNodeKind.PARAM_REF:
            stack.append(float(values[int(operand)]))

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
# Core 1D evaluator
# ---------------------------------------------------------------------------


#
def evaluate_1d(plan: ScheduledPlan1D, theta: np.ndarray) -> np.ndarray:
    """Evaluate the compiled 1D model at optimizer parameters *theta*.

    Parameters
    ----------
    plan
        Immutable compiled execution schedule from ``schedule_1d``.
    theta
        ``(n_opt,)`` optimizer parameter vector.  Order must match
        ``plan.opt_param_names``.

    Returns
    -------
    ndarray
        ``(n_energy,)`` model spectrum.

    Raises
    ------
    ValueError
        If ``len(theta) != len(plan.opt_indices)``.
    """

    if len(theta) != len(plan.opt_indices):
        raise ValueError(
            f"theta length {len(theta)} does not match "
            f"plan.opt_indices length {len(plan.opt_indices)}"
        )

    # 1a. Copy parameter values -> scratch
    values = plan.param_values_init.copy()

    # 1b. Write optimizer params
    values[plan.opt_indices] = theta

    # 1c. Resolve expressions in topological order
    for i in range(plan.n_expressions):
        target = int(plan.expr_target_indices[i])
        values[target] = eval_expr_program_1d(plan.expr_programs[i], values)

    # 2. Component evaluation
    energy = plan.energy
    result = plan.cached_result.copy()
    peak_sum = plan.cached_peak_sum.copy()

    for op_idx in range(plan.n_ops):
        if plan.op_is_constant[op_idx]:
            continue
        kind = int(plan.op_kinds[op_idx])
        start = int(plan.op_param_indptr[op_idx])
        end = int(plan.op_param_indptr[op_idx + 1])
        param_rows = plan.op_param_indices[start:end]

        params = [float(values[int(row)]) for row in param_rows]

        needs_spectrum = bool(plan.op_needs_spectrum[op_idx])
        is_pre = bool(plan.op_is_pre_spectrum[op_idx])

        func, _needs = OP_DISPATCH[kind]
        if needs_spectrum:
            component = func(energy, *params, peak_sum)
        else:
            component = func(energy, *params)

        result += component
        if is_pre:
            peak_sum += component

    return result
