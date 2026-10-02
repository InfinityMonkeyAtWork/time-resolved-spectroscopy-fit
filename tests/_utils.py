"""Shared helpers across the test suite.

Plain module-level functions, not pytest fixtures. Promote a helper here
when the same setup logic is duplicated in two or more test files.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import copy
import warnings
from typing import Any

import numpy as np

from trspecfit import Project, Simulator, fitlib


#
def make_project(
    *,
    name: str = "test",
    spec_fun_str: str = "fit_model_gir",
    show_output: int = 0,
):
    """Create a Project pointing at tests/ for YAML access.

    Defaults to ``show_output=0`` (silent) so test output stays clean.  Pass
    ``show_output=1`` for tests that exercise display/plot behavior.

    Fits never write to disk (v0.14.0); tests that exercise persistence call
    ``save_fits`` / ``export_fit`` explicitly with a ``tmp_path``.
    """

    project = Project(path="tests", name=name)
    project.show_output = show_output
    project.spec_fun_str = spec_fun_str
    return project


#
def extract_truth_pars(model) -> dict[str, float]:
    """Return ``{name: value}`` for all non-expression parameters."""

    return {
        name: model._lmfit_pars[name].value
        for name in model.parameter_names
        if model._lmfit_pars[name].expr is None
    }


#
def simulate_clean(model, *, seed: int = 42) -> np.ndarray:
    """Simulate noiseless 2D data from a truth model."""

    sim = Simulator(
        model=model,
        detection="analog",
        noise_level=0.0,
        noise_type="none",
        seed=seed,
    )
    clean, _, _ = sim.simulate_2d()
    return clean


#
def simulate_noisy(model, *, noise_level: float = 0.01, seed: int = 42) -> np.ndarray:
    """Simulate Gaussian-noisy 2D data from a truth model."""

    sim = Simulator(
        model=model,
        detection="analog",
        noise_level=noise_level,
        noise_type="gaussian",
        seed=seed,
    )
    _, noisy, _ = sim.simulate_2d()
    return noisy


#
def assert_recovery_exact(
    truth_pars: dict[str, float],
    fitted_pars: Any,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> None:
    """Assert exact (clean-data) parameter recovery.

    ``fitted_pars`` is an ``lmfit.Parameters`` object or anything indexable
    by name returning an object with a ``.value`` attribute.
    """

    for name, true_val in truth_pars.items():
        fit_val = fitted_pars[name].value
        assert np.isclose(true_val, fit_val, rtol=rtol, atol=atol), (
            f"{name}: true={true_val:.6f}, fit={fit_val:.6f}"
        )


#
def assert_recovery_within(
    truth_pars: dict[str, float],
    fitted_pars: Any,
    *,
    rel_tol: float = 0.05,
    skip_zero_threshold: float = 1e-6,
) -> None:
    """Assert noisy-fit recovery within ``rel_tol``, skipping near-zero truth values."""

    for name, true_val in truth_pars.items():
        if abs(true_val) < skip_zero_threshold:
            continue
        fit_val = fitted_pars[name].value
        rel_err = abs(fit_val - true_val) / abs(true_val)
        assert rel_err < rel_tol, (
            f"{name}: true={true_val:.4f}, fit={fit_val:.4f}, rel_err={rel_err:.1%}"
        )


#
def capture_fit_wrapper(monkeypatch):
    """Record every ``fitlib.fit_wrapper`` call and run the real one."""

    calls: list[dict] = []
    real = fitlib.fit_wrapper

    def _recording(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(fitlib, "fit_wrapper", _recording)
    return calls


#
def residual_jacobian_fd(par, *, const, args, noise, rel_step: float = 1e-6):
    """Central-difference residual Jacobian, columns in lmfit vary order."""

    columns = []
    for name in [name for name in par if par[name].vary]:
        perturbed = copy.deepcopy(par)
        value = perturbed[name].value
        step = rel_step * max(1.0, abs(value))
        perturbed[name].value = value + step
        res_plus = np.asarray(
            fitlib.residual_fun(perturbed, *const, "lmfit", args, noise=noise)
        )
        perturbed[name].value = value - step
        res_minus = np.asarray(
            fitlib.residual_fun(perturbed, *const, "lmfit", args, noise=noise)
        )
        columns.append((res_plus - res_minus) / (2 * step))
    return np.stack(columns, axis=1)


#
def reload_model(file, model_yaml: str, model_info: str):
    """Replace a loaded model with the same-named entry of *model_yaml*.

    The YAML is the source of truth: a variant of a model (a fixed
    parameter, a moved bound, a shifted seed) is another YAML entry or
    file, loaded in place of the current one. ``load_model`` warns when it
    replaces a model; that warning is the point here, not a finding.
    """

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*already exists on file.*", category=UserWarning
        )
        return file.load_model(model_yaml=model_yaml, model_info=model_info)
