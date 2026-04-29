"""Workflow registry for the roundtrip test matrix.

Each workflow encapsulates the canonical "fit through this API" sequence
plus the rule for where the fitted ``lmfit_pars`` live afterwards. The
test entry point treats every workflow uniformly: pass a fit file, get a
``FitResult`` back.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from trspecfit import File

from .families import Family


#
#
@dataclass(frozen=True)
class FitResult:
    """Normalized output of a workflow runner."""

    params: Any  # lmfit.Parameters or any name -> {.value} mapping


#
#
@dataclass(frozen=True)
class Workflow:
    """Static metadata + runner for one fit workflow API."""

    id: str
    description: str
    requires_baseline: bool
    requires_2d_data: bool
    run: Callable[..., FitResult]


# ---- B: fit_baseline ----


#
def _run_baseline(
    file: File, family: Family, model_name: str, variant: str
) -> FitResult:
    file.fit_baseline(model_name=model_name, stages=2, try_ci=0)
    assert file.model_base is not None  # type guard
    return FitResult(params=file.model_base.result[1].params)


# ---- Sp: fit_spectrum ----


#
def _run_spectrum(
    file: File, family: Family, model_name: str, variant: str
) -> FitResult:
    assert file.time is not None  # type guard
    mid = len(file.time) // 2
    file.fit_spectrum(
        model_name=model_name,
        time_point=int(mid),
        time_type="ind",
        stages=2,
        show_plot=False,
        try_ci=0,
    )
    assert file.model_spec is not None  # type guard
    return FitResult(params=file.model_spec.result[1].params)


# ---- SbS: fit_slice_by_slice ----


#
def _run_sbs(file: File, family: Family, model_name: str, variant: str) -> FitResult:
    file.fit_slice_by_slice(
        model_name=model_name,
        stages=2,
        n_workers=1,
        seed_source="model",
        seed_adapt=None,
        try_ci=0,
    )
    mid = len(file.results_sbs) // 2
    return FitResult(params=file.results_sbs[mid][1].params)


# ---- 2D: fit_baseline + (re-add dynamics) + fit_2d ----


#
def _run_2d(file: File, family: Family, model_name: str, variant: str) -> FitResult:
    file.fit_baseline(model_name=model_name, stages=2, try_ci=0)
    if family.add_dynamics is not None:
        family.add_dynamics(file, variant)
    file.fit_2d(model_name=model_name, stages=2, try_ci=0)
    assert file.model_2d is not None  # type guard
    return FitResult(params=file.model_2d.result[1].params)


# ---- registry ----


WORKFLOWS: dict[str, Workflow] = {
    "B": Workflow(
        id="B",
        description="File.fit_baseline()",
        requires_baseline=False,
        requires_2d_data=False,
        run=_run_baseline,
    ),
    "Sp": Workflow(
        id="Sp",
        description="File.fit_spectrum() at the middle time index",
        requires_baseline=False,
        requires_2d_data=True,
        run=_run_spectrum,
    ),
    "SbS": Workflow(
        id="SbS",
        description="File.fit_slice_by_slice() (serial, seed_source='model')",
        requires_baseline=False,
        requires_2d_data=True,
        run=_run_sbs,
    ),
    "2D": Workflow(
        id="2D",
        description="File.fit_baseline() + File.fit_2d()",
        requires_baseline=True,
        requires_2d_data=True,
        run=_run_2d,
    ),
}
