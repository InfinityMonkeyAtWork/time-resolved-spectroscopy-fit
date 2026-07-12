"""Experimental JAX evaluator backend for 2D scheduled plans.

Phase B slice of the JAX track (docs/design/jax-planning.md): static
component ops, dynamics groups, and arithmetic expressions.  Profiles,
convolution, subcycle dynamics, and Voigt are gated out by
``can_lower_jax_2d``; such graphs run on the compiled NumPy evaluator
instead.

The backend compiles one jitted function per plan via trace-time
unrolling: ``make_evaluator_2d_jax(plan)`` walks the schedule arrays
with host-side Python control flow while tracing, so the compiled
XLA program contains no dispatch, and only ``theta`` is traced.

Importing this module enables JAX 64-bit mode globally
(``jax_enable_x64``); parity with the float64 NumPy evaluator requires
it.

JAX is an optional dependency: ``pip install "trspecfit[jax]"``.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from trspecfit.graph_ir import (
    DynFuncKind,
    ExprNodeKind,
    OpKind,
    ScheduledPlan2D,
)

if TYPE_CHECKING:
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import erf as _jax_erf

    _HAVE_JAX = True
else:
    try:
        import jax
        import jax.numpy as jnp
        from jax.scipy.special import erf as _jax_erf
    except ImportError:  # pragma: no cover - exercised only without [jax]
        jax = None
        jnp = None
        _jax_erf = None
        _HAVE_JAX = False
    else:
        # Parity contract with the float64 NumPy evaluator.
        jax.config.update("jax_enable_x64", True)
        _HAVE_JAX = True


#
def _require_jax() -> None:
    """Raise a helpful ImportError when JAX is not installed."""

    if not _HAVE_JAX:
        raise ImportError(
            "The JAX evaluator backend requires jax; "
            'install it with: pip install "trspecfit[jax]"'
        )


# ---------------------------------------------------------------------------
# Component-op kernels (jnp mirrors of functions/energy.py bodies)
# ---------------------------------------------------------------------------
# Same broadcasting contract as the NumPy evaluator: *x* is
# ``(1, n_energy)``, params are ``(n_time, 1)`` columns, *spectrum* is
# ``(n_time, n_energy)``.


#
def _Offset(x, y0, spectrum=None):
    return jnp.ones_like(x) * y0


#
def _Shirley(x, pShirley, spectrum):
    flipped = jnp.flip(spectrum, axis=-1)
    return pShirley * jnp.flip(jnp.cumsum(flipped, axis=-1), axis=-1)


#
def _LinBack(x, m, b, xStart, xStop, spectrum=None):
    # No xStart < xStop validation here: parameters are traced values,
    # so the ordering cannot be checked at trace time.  The NumPy
    # reference path validates; keep fit bounds ordered.
    y = m * (x - xStart) + b
    y_stop = m * (xStop - xStart) + b
    return jnp.where(x < xStart, b, jnp.where(x > xStop, y_stop, y))


#
def _Gauss(x, A, x0, SD):
    return A * jnp.exp(-1 / 2 * ((x - x0) / SD) ** 2)


#
def _GaussAsym(x, A, x0, SD, ratio):
    return jnp.where(x < x0, _Gauss(x, A, x0, SD), _Gauss(x, A, x0, SD * ratio))


#
def _Lorentz(x, A, x0, W):
    return A / (1 + ((x - x0) / W * 2) ** 2)


#
def _GLS(x, A, x0, F, m):
    u2 = ((x - x0) / F) ** 2
    return A * ((1 - m) * jnp.exp(-u2 * 4 * jnp.log(2)) + m / (1 + 4 * u2))


#
def _GLP(x, A, x0, F, m):
    u2 = ((x - x0) / F) ** 2
    return A * jnp.exp(-u2 * 4 * jnp.log(2) * (1 - m)) / (1 + 4 * m * u2)


#
def _DS(x, A, x0, F, alpha):
    dx = x - x0
    return (
        A
        * jnp.cos(jnp.pi * alpha / 2 + (1 - alpha) * jnp.arctan(dx / F))
        / (F**2 + dx**2) ** ((1 - alpha) / 2)
    )


JAX_OP_DISPATCH: dict[int, Callable] = {
    int(OpKind.GAUSS): _Gauss,
    int(OpKind.GAUSS_ASYM): _GaussAsym,
    int(OpKind.LORENTZ): _Lorentz,
    int(OpKind.GLS): _GLS,
    int(OpKind.GLP): _GLP,
    int(OpKind.DS): _DS,
    int(OpKind.OFFSET): _Offset,
    int(OpKind.LINBACK): _LinBack,
    int(OpKind.SHIRLEY): _Shirley,
}


# ---------------------------------------------------------------------------
# Dynamics kernels (jnp mirrors of functions/time.py bodies)
# ---------------------------------------------------------------------------


#
def _stepFun(t, A, t0):
    return jnp.where(t < t0, 0.0, A)


#
def _linFun(t, m, t0):
    return jnp.where(t < t0, 0.0, m * (t - t0))


#
def _expFun(t, A, tau, t0):
    return jnp.where(t < t0, 0.0, A * jnp.exp(-1 / tau * (t - t0)))


#
def _sinFun(t, A, f, phi, t0):
    return jnp.where(t < t0, 0.0, A * jnp.sin(2 * jnp.pi * f * (t - t0) + phi))


#
def _sinDivX(t, A, f, t0):
    return jnp.where(t < t0, 0.0, A * jnp.sinc(2 * f * (t - t0)))


#
def _erfFun(t, A, SD, t0):
    return A / 2 * (1 + _jax_erf((t - t0) / (SD * jnp.sqrt(2.0))))


#
def _sqrtFun(t, A, t0):
    return A * jnp.sqrt(jnp.clip(t - t0, 0))


JAX_DYNAMICS_DISPATCH: dict[int, Callable] = {
    int(DynFuncKind.EXPFUN): _expFun,
    int(DynFuncKind.SINFUN): _sinFun,
    int(DynFuncKind.LINFUN): _linFun,
    int(DynFuncKind.SINDIVX): _sinDivX,
    int(DynFuncKind.ERFFUN): _erfFun,
    int(DynFuncKind.SQRTFUN): _sqrtFun,
    int(DynFuncKind.STEPFUN): _stepFun,
}


# ---------------------------------------------------------------------------
# Expression evaluation (trace-time unrolled RPN)
# ---------------------------------------------------------------------------


#
def _eval_expr_rows(instructions: np.ndarray, rows: list, n_time: int):
    """Evaluate one packed RPN program over per-parameter trace rows.

    Mirrors ``eval_expr_program``; PARAM_REF pushes the ``(n_time,)``
    row (a traced value), constants stay Python floats until an
    operator combines them.
    """

    stack: list = []
    n_instr = len(instructions) // 2

    for i in range(n_instr):
        kind = int(instructions[2 * i])
        operand = instructions[2 * i + 1]

        if kind == ExprNodeKind.CONST:
            stack.append(float(np.int64(operand).view(np.float64)))
        elif kind == ExprNodeKind.PARAM_REF:
            stack.append(rows[int(operand)])
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
    if isinstance(result, float):  # constant-only program
        return jnp.full((n_time,), result, dtype=jnp.float64)
    return result


# ---------------------------------------------------------------------------
# Evaluator factory
# ---------------------------------------------------------------------------


#
def _check_plan_supported(plan: ScheduledPlan2D) -> None:
    """Reject plan features outside the Phase B JAX slice.

    Callers should gate at the graph level with ``can_lower_jax_2d``;
    this is the defensive plan-level check for plans built directly.
    """

    unsupported: list[str] = []
    if plan.n_conv_steps > 0:
        unsupported.append("convolution steps")
    if plan.n_profile_samples > 0 or plan.n_profile_exprs > 0:
        unsupported.append("profile-varying parameters")
    for op_idx in range(plan.n_ops):
        if int(plan.op_kinds[op_idx]) not in JAX_OP_DISPATCH:
            unsupported.append(f"op kind {OpKind(int(plan.op_kinds[op_idx])).name}")
    if unsupported:
        raise ValueError(
            "Plan is not supported by the JAX evaluator: "
            + ", ".join(sorted(set(unsupported)))
            + ". Gate with can_lower_jax_2d(graph) and use the NumPy "
            "evaluator for unsupported models."
        )


#
def make_evaluator_2d_jax(
    plan: ScheduledPlan2D,
) -> Callable[[np.ndarray], np.ndarray]:
    """Compile a jitted JAX evaluator for a 2D scheduled plan.

    Parameters
    ----------
    plan : ScheduledPlan2D
        Compiled 2D execution schedule (from ``schedule_2d``).  The
        source graph must pass ``can_lower_jax_2d``.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        ``evaluate(theta) -> (n_time, n_energy)`` with the same theta
        contract as ``evaluate_2d(plan, theta)``.  The first call
        triggers XLA compilation; subsequent calls reuse it.

    Raises
    ------
    ImportError
        If jax is not installed.
    ValueError
        If the plan uses features outside the JAX slice.
    """

    _require_jax()
    _check_plan_supported(plan)

    n_opt = len(plan.opt_indices)
    n_time = plan.n_time

    #
    def _evaluate(theta):
        # Per-parameter trace rows; list mutation is trace-time only.
        rows = [jnp.asarray(plan.param_traces_init[i]) for i in range(plan.n_params)]
        for k in range(n_opt):
            rows[int(plan.opt_indices[k])] = jnp.broadcast_to(theta[k], (n_time,))

        # Interleaved dynamics/expression resolution in schedule order.
        # Convolution steps (kind 2) are excluded by _check_plan_supported.
        for step in range(len(plan.resolution_kinds)):
            kind = int(plan.resolution_kinds[step])
            idx = int(plan.resolution_indices[step])
            if kind == 0:  # dynamics group
                target = int(plan.dyn_group_target_row[idx])
                acc = rows[int(plan.dyn_group_base_row[idx])]
                s_start = int(plan.dyn_group_indptr[idx])
                s_end = int(plan.dyn_group_indptr[idx + 1])
                for s in range(s_start, s_end):
                    func = JAX_DYNAMICS_DISPATCH[int(plan.dyn_sub_func_id[s])]
                    n_par = int(plan.dyn_sub_n_params[s])
                    # t=0 read is exact: substep param rows are
                    # time-constant by construction (see eval_2d).
                    dyn_params = [
                        rows[int(row)][0] for row in plan.dyn_sub_param_rows[s, :n_par]
                    ]
                    acc = acc + func(
                        jnp.asarray(plan.dyn_sub_time_axes[s]), *dyn_params
                    ) * jnp.asarray(plan.dyn_sub_masks[s])
                rows[target] = acc
            else:  # kind == 1: expression
                target = int(plan.expr_target_rows[idx])
                start = int(plan.expr_indptr[idx])
                end = int(plan.expr_indptr[idx + 1])
                rows[target] = _eval_expr_rows(
                    plan.expr_instructions[start:end], rows, n_time
                )

        # Component evaluation, unrolled in schedule order.
        energy = jnp.asarray(plan.energy)[jnp.newaxis, :]
        result = jnp.asarray(plan.cached_result)
        peak_sum = jnp.asarray(plan.cached_peak_sum)
        for op_idx in range(plan.n_ops):
            if plan.op_is_constant[op_idx]:
                continue
            func = JAX_OP_DISPATCH[int(plan.op_kinds[op_idx])]
            start = int(plan.op_param_indptr[op_idx])
            end = int(plan.op_param_indptr[op_idx + 1])
            params = [
                rows[int(row)][:, jnp.newaxis]
                for row in plan.op_param_indices[start:end]
            ]
            if plan.op_needs_spectrum[op_idx]:
                component = func(energy, *params, peak_sum)
            else:
                component = func(energy, *params)
            result = result + component
            if plan.op_is_pre_spectrum[op_idx]:
                peak_sum = peak_sum + component

        return result

    jitted = jax.jit(_evaluate)

    #
    def evaluate(theta: np.ndarray) -> np.ndarray:
        # Host-side checks stay outside the jitted region.
        theta_arr = np.asarray(theta, dtype=np.float64)
        if theta_arr.shape != (n_opt,):
            raise ValueError(
                f"theta shape {theta_arr.shape} does not match "
                f"plan.opt_indices length {n_opt}"
            )
        return np.asarray(jitted(theta_arr))

    return evaluate
