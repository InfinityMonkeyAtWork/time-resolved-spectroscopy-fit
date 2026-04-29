"""Focused MCMC and worker variants for the roundtrip suite.

Per the matrix doc, MCMC and worker mode are *focused* secondary
requirements, not full axes on the main matrix.  This module covers them
with a small set of dedicated tests:

- MC1: MCMC with workers=1 — exercises MCMC sampling itself.
- MC2: MCMC with workers=2 — exercises pickling / process boundary.

Worker variants for ``fit_slice_by_slice`` are deferred until the API
exposes a parallel-versus-serial semantic difference; today
``n_workers=1`` and ``n_workers=2`` both go through ``ProcessPoolExecutor``
once n_workers > 1, so a focused W2 test is sufficient.
"""

from __future__ import annotations

import pytest
from _utils import (
    assert_recovery_exact,
    extract_truth_pars,
    make_project,
    simulate_clean,
)

from trspecfit.utils.lmfit import MC

from .families import FAMILIES

pytestmark = pytest.mark.slow


_MC_KWARGS = {
    "use_mc": 1,
    "steps": 50,
    "nwalkers": 32,
    "burn": 5,
    "thin": 1,
}


#
def _build_fit_file_for_baseline(family_id: str):
    """Common setup: build truth, simulate, build a baseline-ready fit file."""

    family = FAMILIES[family_id]
    truth_project = make_project(name="mc_truth", spec_fun_str="fit_model_mcp")
    truth_file = family.build_truth(truth_project, variant="default")
    data = simulate_clean(truth_file.model_active)

    fit_project = make_project(name="mc_fit", spec_fun_str="fit_model_gir")
    fit_kwargs = {
        "data": data,
        "energy": truth_file.energy,
        "time": truth_file.time,
        "variant": "default",
    }
    if family.needs_aux:
        fit_kwargs["aux"] = truth_file.aux_axis
    fit_file = family.build_fit(fit_project, **fit_kwargs)
    return fit_file, family


# ---- MC1: serial MCMC on F1 baseline ----


#
def test_mc1_baseline_f1():
    """MCMC with workers=1 on the simplest baseline fit must run without crashing."""

    fit_file, family = _build_fit_file_for_baseline("F1")
    mc = MC(workers=1, **_MC_KWARGS)
    fit_file.fit_baseline(
        model_name=family.model_name("default"),
        stages=1,
        try_ci=0,
        mc_settings=mc,
    )
    assert fit_file.model_base is not None  # type guard


# ---- MC2: parallel MCMC on F1 baseline ----


#
def test_mc2_baseline_f1():
    """MCMC with workers=2 must not hit pickling / serialization errors."""

    fit_file, family = _build_fit_file_for_baseline("F1")
    mc = MC(workers=2, **_MC_KWARGS)
    fit_file.fit_baseline(
        model_name=family.model_name("default"),
        stages=1,
        try_ci=0,
        mc_settings=mc,
    )
    assert fit_file.model_base is not None  # type guard


# ---- MC2 on an expression-sensitive case (F10 baseline) ----


#
def test_mc2_baseline_f10_expression():
    """MCMC workers=2 on an expression+profile family — pickling stress test."""

    fit_file, family = _build_fit_file_for_baseline("F10")
    mc = MC(workers=2, **_MC_KWARGS)
    fit_file.fit_baseline(
        model_name=family.model_name("default"),
        stages=1,
        try_ci=0,
        mc_settings=mc,
    )
    assert fit_file.model_base is not None  # type guard


# ---- MC2 on a 2D varying case (F3) ----


#
def test_mc2_2d_f3():
    """MCMC workers=2 through fit_2d on a standard-dynamics family."""

    fit_file, family = _build_fit_file_for_baseline("F3")
    fit_file.fit_baseline(model_name=family.model_name("default"), stages=2, try_ci=0)
    assert family.add_dynamics is not None
    family.add_dynamics(fit_file, "default")

    mc = MC(workers=2, **_MC_KWARGS)
    fit_file.fit_2d(
        model_name=family.model_name("default"),
        stages=1,
        try_ci=0,
        mc_settings=mc,
    )
    assert fit_file.model_2d is not None  # type guard


# ---- W2: parallel SbS on F1 ----


#
def test_w2_sbs_f1():
    """fit_slice_by_slice with n_workers=2 must complete without pickling errors."""

    fit_file, family = _build_fit_file_for_baseline("F1")
    fit_file.fit_slice_by_slice(
        model_name=family.model_name("default"),
        stages=1,
        n_workers=2,
        seed_source="model",
        seed_adapt=None,
        try_ci=0,
    )
    assert len(fit_file.results_sbs) == len(fit_file.time)


# ---- W2 on a profile family: pickles Par.p_model into workers ----


#
def test_w2_sbs_f6_profile():
    """SbS workers=2 on a profile-bearing family.

    Per-slice fits are 1D, so dynamics aren't inside the worker model,
    but profiles are: ``Par.p_model`` (a Profile sub-Model) must survive
    the worker pickle boundary or this fails. F1 doesn't exercise that
    path.
    """

    fit_file, family = _build_fit_file_for_baseline("F6")
    fit_file.fit_slice_by_slice(
        model_name=family.model_name("default"),
        stages=1,
        n_workers=2,
        seed_source="model",
        seed_adapt=None,
        try_ci=0,
    )
    assert len(fit_file.results_sbs) == len(fit_file.time)


# ---- SbS seed combos: explicit seed + baseline+argmax_shift ----


#
def _build_sbs_truth_and_fit(family_id: str):
    """Build a (truth_file, fit_file, family) triple for SbS seed-combo tests."""

    family = FAMILIES[family_id]
    truth_project = make_project(name="sbs_truth", spec_fun_str="fit_model_mcp")
    truth_file = family.build_truth(truth_project, variant="default")
    data = simulate_clean(truth_file.model_active)

    fit_project = make_project(name="sbs_fit", spec_fun_str="fit_model_gir")
    fit_kwargs = {
        "data": data,
        "energy": truth_file.energy,
        "time": truth_file.time,
        "variant": "default",
    }
    if family.needs_aux:
        fit_kwargs["aux"] = truth_file.aux_axis
    fit_file = family.build_fit(fit_project, **fit_kwargs)
    return truth_file, fit_file, family


#
def test_sbs_explicit_seed_f1():
    """seed_source='explicit' + seed_adapt=None must accept seed_values and recover."""

    truth_file, fit_file, family = _build_sbs_truth_and_fit("F1")
    truth_pars = extract_truth_pars(truth_file.model_active)
    seed_values = {name: val for name, val in truth_pars.items()}

    fit_file.fit_slice_by_slice(
        model_name=family.model_name("default"),
        stages=2,
        n_workers=1,
        seed_source="explicit",
        seed_values=seed_values,
        seed_adapt=None,
        try_ci=0,
    )

    mid = len(fit_file.results_sbs) // 2
    assert_recovery_exact(truth_pars, fit_file.results_sbs[mid][1].params)


#
def test_sbs_baseline_argmax_shift_f1():
    """Production-default SbS: seed_source='baseline' + seed_adapt='argmax_shift'."""

    truth_file, fit_file, family = _build_sbs_truth_and_fit("F1")
    truth_pars = extract_truth_pars(truth_file.model_active)

    fit_file.fit_baseline(model_name=family.model_name("default"), stages=2, try_ci=0)
    fit_file.fit_slice_by_slice(
        model_name=family.model_name("default"),
        stages=2,
        n_workers=1,
        seed_source="baseline",
        seed_adapt="argmax_shift",
        try_ci=0,
    )

    mid = len(fit_file.results_sbs) // 2
    assert_recovery_exact(truth_pars, fit_file.results_sbs[mid][1].params)
