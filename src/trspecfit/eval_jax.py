"""Experimental JAX evaluator backend for 2D scheduled plans.

Covers the full lowered 2D surface (docs/design/jax-planning.md,
Phases B + C): static component ops, dynamics groups, arithmetic
expressions, subcycle dynamics, profile-varying parameters,
kernel-matrix convolution, and Voigt (via a Weideman rational
approximation of ``wofz``).  ``can_lower_jax_2d`` gates entry; graphs
it rejects run on the compiled NumPy evaluator.

The backend compiles one jitted function per plan via trace-time
unrolling: ``make_evaluator_2d_jax(plan)`` walks the schedule arrays
with host-side Python control flow while tracing, so the compiled
XLA program contains no dispatch, and only ``theta`` is traced.

Value checks that the NumPy path performs on parameter-dependent
quantities (LinBack ordering, convolution kernel positivity) cannot
run on traced values and are omitted here; keep fit bounds ordered
and kernel widths positive.

Importing this module enables JAX 64-bit mode globally
(``jax_enable_x64``); parity with the float64 NumPy evaluator requires
it.

JAX is an optional dependency: ``pip install "trspecfit[jax]"``.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from trspecfit.graph_ir import (
    ConvKernelKind,
    DynFuncKind,
    ExprNodeKind,
    OpKind,
    ParamSourceKind,
    ProfileFuncKind,
    ScheduledPlan2D,
)

if TYPE_CHECKING:
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import erf as _jax_erf
    from jax.scipy.special import erfc as _jax_erfc

    _HAVE_JAX = True
else:
    try:
        import jax
        import jax.numpy as jnp
        from jax.scipy.special import erf as _jax_erf
        from jax.scipy.special import erfc as _jax_erfc
    except ImportError:  # pragma: no cover - exercised only without [jax]
        jax = None
        jnp = None
        _jax_erf = None
        _jax_erfc = None
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


#
def _weideman_coeffs(n_terms: int) -> tuple[float, np.ndarray]:
    """Precompute Weideman (1994) rational-approximation coefficients.

    Host-side NumPy, evaluated once at import.  ``_wofz`` uses the
    result to approximate the Faddeeva function ``w(z)`` for
    ``Im(z) > 0`` (always true for Voigt: z carries ``+i W/2``).
    """

    m = 2 * n_terms
    k = np.arange(-m + 1, m)
    L = np.sqrt(n_terms / np.sqrt(2.0))
    t = L * np.tan(k * np.pi / (2 * m))
    f = np.concatenate(([0.0], np.exp(-(t**2)) * (L**2 + t**2)))
    a = np.real(np.fft.fft(np.fft.fftshift(f))) / (2 * m)
    return float(L), a[1 : n_terms + 1][::-1]


# 64 terms: max relative error vs scipy wofz < 1e-13 over the physical
# Voigt domain (checked in tests/test_evaluate_jax.py).
_WEIDEMAN_L, _WEIDEMAN_A = _weideman_coeffs(64)


#
def _wofz(z):
    """Faddeeva function ``w(z)`` for ``Im(z) > 0`` (Weideman 1994)."""

    iz = 1j * z
    Z = (_WEIDEMAN_L + iz) / (_WEIDEMAN_L - iz)
    p = jnp.zeros_like(Z)
    for coeff in _WEIDEMAN_A:  # Horner, unrolled at trace time
        p = p * Z + coeff
    return 2 * p / (_WEIDEMAN_L - iz) ** 2 + (1 / jnp.sqrt(jnp.pi)) / (_WEIDEMAN_L - iz)


#
def _Voigt(x, A, x0, SD, W):
    scale = SD * jnp.sqrt(2.0)
    voigt = jnp.real(_wofz(((x - x0) + 1j * (W / 2)) / scale))
    peak_voigt = jnp.real(_wofz(1j * (W / 2) / scale))
    return A * voigt / peak_voigt


JAX_OP_DISPATCH: dict[int, Callable] = {
    int(OpKind.GAUSS): _Gauss,
    int(OpKind.GAUSS_ASYM): _GaussAsym,
    int(OpKind.LORENTZ): _Lorentz,
    int(OpKind.VOIGT): _Voigt,
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
# Profile kernels (jnp mirrors of functions/profile.py bodies)
# ---------------------------------------------------------------------------


#
def _pExpDecay(x, A, tau):
    return A * jnp.exp(-x / tau)


#
def _pLinear(x, m, b):
    return m * x + b


#
def _pGauss(x, A, x0, SD):
    return A * jnp.exp(-0.5 * ((x - x0) / SD) ** 2)


JAX_PROFILE_DISPATCH: dict[int, Callable] = {
    int(ProfileFuncKind.PEXPDECAY): _pExpDecay,
    int(ProfileFuncKind.PLINEAR): _pLinear,
    int(ProfileFuncKind.PGAUSS): _pGauss,
}


# ---------------------------------------------------------------------------
# Convolution kernels and edge-mass companions (functions/time.py mirrors)
# ---------------------------------------------------------------------------
# The NumPy companions validate kernel parameters (strictly positive,
# finite) on every call; traced values cannot be validated, so the JAX
# mirrors omit that backstop.


#
def _gaussCONV(x, SD):
    return jnp.exp(-1 / 2 * (x / SD) ** 2)


#
def _expSymCONV(x, tau):
    return jnp.exp(-1 / tau * jnp.abs(x))


#
def _expDecayCONV(x, tau):
    return jnp.where(x < 0, 0.0, _expSymCONV(x, tau))


#
def _expRiseCONV(x, tau):
    return jnp.where(x > 0, 0.0, _expSymCONV(x, tau))


#
def _boxCONV(x, width):
    return jnp.where(jnp.abs(x) <= width / 2, 1.0, 0.0)


#
def _gaussCONV_edge_mass(dt_left, dt_right, SD):
    scale = SD * jnp.sqrt(jnp.pi / 2)
    M_L = scale * _jax_erfc(dt_left / (jnp.sqrt(2.0) * SD))
    M_R = scale * _jax_erfc(-dt_right / (jnp.sqrt(2.0) * SD))
    return M_L, M_R


#
def _expSymCONV_edge_mass(dt_left, dt_right, tau):
    return tau * jnp.exp(-dt_left / tau), tau * jnp.exp(dt_right / tau)


#
def _expDecayCONV_edge_mass(dt_left, dt_right, tau):
    return tau * jnp.exp(-dt_left / tau), jnp.zeros_like(dt_right)


#
def _expRiseCONV_edge_mass(dt_left, dt_right, tau):
    return jnp.zeros_like(dt_left), tau * jnp.exp(dt_right / tau)


#
def _boxCONV_edge_mass(dt_left, dt_right, width):
    M_L = jnp.clip(width / 2 - dt_left, 0.0, width)
    M_R = jnp.clip(dt_right + width / 2, 0.0, width)
    return M_L, M_R


JAX_CONV_KERNEL_DISPATCH: dict[int, Callable] = {
    int(ConvKernelKind.GAUSSCONV): _gaussCONV,
    int(ConvKernelKind.EXPSYMCONV): _expSymCONV,
    int(ConvKernelKind.EXPDECAYCONV): _expDecayCONV,
    int(ConvKernelKind.EXPRISECONV): _expRiseCONV,
    int(ConvKernelKind.BOXCONV): _boxCONV,
}

JAX_CONV_EDGE_MASS_DISPATCH: dict[int, Callable] = {
    int(ConvKernelKind.GAUSSCONV): _gaussCONV_edge_mass,
    int(ConvKernelKind.EXPSYMCONV): _expSymCONV_edge_mass,
    int(ConvKernelKind.EXPDECAYCONV): _expDecayCONV_edge_mass,
    int(ConvKernelKind.EXPRISECONV): _expRiseCONV_edge_mass,
    int(ConvKernelKind.BOXCONV): _boxCONV_edge_mass,
}


# ---------------------------------------------------------------------------
# Expression evaluation (trace-time unrolled RPN)
# ---------------------------------------------------------------------------


#
def _eval_expr_rows(instructions: np.ndarray, rows: list, shape: tuple):
    """Evaluate one packed RPN program over per-parameter trace rows.

    Mirrors ``eval_expr_program``; PARAM_REF pushes the *shape*-shaped
    row (a traced value), constants stay Python floats until an
    operator combines them.  ``rows`` entries are ``(n_time,)`` for
    plan expressions and ``(n_time, n_aux)`` for per-sample profile
    expressions.
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
        return jnp.full(shape, result, dtype=jnp.float64)
    return result


# ---------------------------------------------------------------------------
# Evaluator factory
# ---------------------------------------------------------------------------


#
def _check_plan_supported(plan: ScheduledPlan2D) -> None:
    """Reject plan features the JAX backend has no kernel for.

    Callers should gate at the graph level with ``can_lower_jax_2d``;
    this is the defensive plan-level check for plans built directly.
    """

    unsupported: list[str] = []
    for op_idx in range(plan.n_ops):
        if int(plan.op_kinds[op_idx]) not in JAX_OP_DISPATCH:
            unsupported.append(f"op kind {OpKind(int(plan.op_kinds[op_idx])).name}")
    for step in range(plan.n_conv_steps):
        if int(plan.conv_func_ids[step]) not in JAX_CONV_KERNEL_DISPATCH:
            unsupported.append(
                f"conv kernel {ConvKernelKind(int(plan.conv_func_ids[step])).name}"
            )
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

        # Interleaved dynamics/expression/convolution resolution in
        # schedule order.
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
            elif kind == 1:  # expression
                target = int(plan.expr_target_rows[idx])
                start = int(plan.expr_indptr[idx])
                end = int(plan.expr_indptr[idx + 1])
                rows[target] = _eval_expr_rows(
                    plan.expr_instructions[start:end], rows, (n_time,)
                )
            else:  # kind == 2: resolved-trace convolution
                operator = plan.conv_operator
                assert operator is not None  # type guard: set when steps exist
                target = int(plan.conv_target_rows[idx])
                func_id = int(plan.conv_func_ids[idx])
                p_start = int(plan.conv_param_indptr[idx])
                p_end = int(plan.conv_param_indptr[idx + 1])
                # t=0 read is exact; same time-constant invariant as above
                kernel_params = [
                    rows[int(plan.conv_param_rows[j])][0] for j in range(p_start, p_end)
                ]
                mass_left, mass_right = JAX_CONV_EDGE_MASS_DISPATCH[func_id](
                    jnp.asarray(operator.dt_left),
                    jnp.asarray(operator.dt_right),
                    *kernel_params,
                )
                kernel_values = JAX_CONV_KERNEL_DISPATCH[func_id](
                    jnp.asarray(operator.dt_unique), *kernel_params
                )
                # conv_matrix_apply without its host-side value checks
                interior = kernel_values[operator.gather_idx] * jnp.asarray(
                    operator.quad_weights
                )
                row_sums = interior.sum(axis=1) + mass_left + mass_right
                y = rows[target]
                y_conv = interior @ y + mass_left * y[0] + mass_right * y[-1]
                rows[target] = y_conv / row_sums

        # Profile sample groups -> (n_time, n_aux) per group, then
        # per-sample profile expressions over broadcast virtual rows.
        n_aux = plan.n_aux
        aux_2d = jnp.asarray(plan.aux_axis)[jnp.newaxis, :]
        profile_samples: list = []
        for g in range(plan.n_profile_samples):
            base_row = int(plan.profile_sample_base_rows[g])
            value = jnp.broadcast_to(rows[base_row][:, jnp.newaxis], (n_time, n_aux))
            c_start = int(plan.profile_sample_component_indptr[g])
            c_end = int(plan.profile_sample_component_indptr[g + 1])
            for c in range(c_start, c_end):
                func = JAX_PROFILE_DISPATCH[int(plan.profile_component_func_ids[c])]
                p_start = int(plan.profile_component_param_indptr[c])
                p_end = int(plan.profile_component_param_indptr[c + 1])
                params = [
                    rows[int(row)][:, jnp.newaxis]
                    for row in plan.profile_component_param_rows[p_start:p_end]
                ]
                value = value + func(aux_2d, *params)
            profile_samples.append(value)

        profile_exprs: list = []
        if plan.n_profile_exprs > 0:
            virtual_rows = [
                jnp.broadcast_to(row[:, jnp.newaxis], (n_time, n_aux)) for row in rows
            ] + profile_samples
            for e in range(plan.n_profile_exprs):
                start = int(plan.profile_expr_indptr[e])
                end = int(plan.profile_expr_indptr[e + 1])
                profile_exprs.append(
                    _eval_expr_rows(
                        plan.profile_expr_instructions[start:end],
                        virtual_rows,
                        (n_time, n_aux),
                    )
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
            if plan.op_is_profiled[op_idx]:
                # Vectorized over aux (axis 1), then averaged.  The
                # NumPy path loops per aux point to avoid materialized
                # (n_time, n_aux, n_energy) temporaries; under XLA the
                # fused form wins on simplicity and lets the compiler
                # decide.
                sources = []
                for sk, si in zip(
                    plan.op_param_source_kinds[start:end],
                    plan.op_param_indices[start:end],
                    strict=True,
                ):
                    if int(sk) == int(ParamSourceKind.SCALAR):
                        source = jnp.broadcast_to(
                            rows[int(si)][:, jnp.newaxis], (n_time, n_aux)
                        )
                    elif int(sk) == int(ParamSourceKind.PROFILE_SAMPLE):
                        source = profile_samples[int(si)]
                    else:
                        source = profile_exprs[int(si)]
                    sources.append(source[:, :, jnp.newaxis])  # (n_time, n_aux, 1)
                if plan.op_needs_spectrum[op_idx]:
                    component = func(
                        energy[jnp.newaxis, :, :],
                        *sources,
                        peak_sum[:, jnp.newaxis, :],
                    ).mean(axis=1)
                else:
                    component = func(energy[jnp.newaxis, :, :], *sources).mean(axis=1)
            else:
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
