"""
Integration tests for ``FitResults.plot_fit(full_range=True)``.

Builds a real fit for each fit type with a fit window strictly inside the
full energy (and time, for 2d) axis, saves to an archive, reloads with no
live ``File`` in memory, then verifies the full-range reconstruction:
real data across the whole persisted axis, ``fit``/``components`` equal
to the slot's own cropped arrays inside the window and ``NaN`` (never a
fabricated value) outside it.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from typing import Any

import numpy as np
import pytest
from _utils import make_project, simulate_noisy
from roundtrip.families import FAMILIES

from trspecfit import FitResults
from trspecfit.utils.arrays import resolve_time_selection
from trspecfit.utils.fit_io import SavedFile


#
def _build_fit_file(family_id: str, *, spec_fun_str: str = "fit_model_gir"):
    """(truth_file, fit_file, family) for a family, with noisy data.

    Mirrors the setup pattern in test_fit_archive_roundtrip.py.
    """

    family = FAMILIES[family_id]
    truth_project = make_project(name="fr_truth", spec_fun_str=spec_fun_str)
    truth_file = family.build_truth(truth_project, variant="default")
    data = simulate_noisy(truth_file.model_active, noise_level=0.01)

    fit_project = make_project(name="fr_fit", spec_fun_str=spec_fun_str)
    fit_project.show_output = 0
    fit_kwargs: dict[str, Any] = {
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
def _narrow_energy_limits(fit_file) -> list[float]:
    """A fit window strictly inside the full energy axis (exercises NaN padding)."""

    e = fit_file.energy
    return [float(e[5]), float(e[-6])]


#
def _narrow_time_limits(fit_file) -> list[float]:
    t = fit_file.time
    return [float(t[2]), float(t[-3])]


# ---------------------------------------------------------------------------
# baseline
# ---------------------------------------------------------------------------


#
def test_baseline_full_range_reconstructs_real_data(tmp_path) -> None:
    _, fit_file, family = _build_fit_file("F1")
    fit_file.set_fit_limits(_narrow_energy_limits(fit_file), show_plot=False)
    fit_file.fit_baseline(model_name=family.model_name("default"), stages=1, try_ci=0)

    archive_path = tmp_path / "baseline.fit.h5"
    fit_file.p.save_fits(archive_path, show_output=0)
    loaded = FitResults.load(archive_path)  # no live File
    slot = next(iter(loaded))
    provider = loaded._provider_for(slot)

    e_lim = slot.selection["e_lim"]
    b0, b1 = slot.selection["base_t_ind"]
    expected_obs = np.mean(provider.data_raw[b0:b1, :], axis=0)

    full_obs = loaded._full_observed_for(slot, provider)
    np.testing.assert_allclose(full_obs, expected_obs)

    energy_full, _ = loaded._axes_for(slot, full_range=True)
    assert energy_full is not None
    assert len(energy_full) == len(provider.energy)
    assert len(energy_full) > slot.observed.shape[0]  # window is strictly narrower

    fit_padded = loaded._pad_axis(np.asarray(slot.fit), len(energy_full), e_lim, axis=0)
    assert np.all(np.isnan(fit_padded[: e_lim[0]]))
    assert np.all(np.isnan(fit_padded[e_lim[1] :]))
    np.testing.assert_array_equal(fit_padded[e_lim[0] : e_lim[1]], slot.fit)

    comp_padded = loaded._pad_axis(slot.components, len(energy_full), e_lim, axis=1)
    assert np.all(np.isnan(comp_padded[:, : e_lim[0]]))
    np.testing.assert_array_equal(comp_padded[:, e_lim[0] : e_lim[1]], slot.components)

    # full end-to-end call must not raise
    loaded.plot_fit(
        model=family.model_name("default"),
        fit_type="baseline",
        full_range=True,
        show_plot=False,
    )


#
def test_live_full_range_uses_slot_correction_state() -> None:
    """A live-session slot renders against its own captured correction
    state, not the file's current one.

    Fit raw data, then subtract_dark(): the pre-correction slot's
    full-range observed must reconstruct from ``data_raw`` with the
    slot's ``dark=None`` — serving the file's current corrected ``data``
    would silently shift the observations outside the fit window.
    """

    _, fit_file, family = _build_fit_file("F1")
    fit_file.set_fit_limits(_narrow_energy_limits(fit_file), show_plot=False)
    fit_file.fit_baseline(model_name=family.model_name("default"), stages=1, try_ci=0)
    assert fit_file.data_raw is not None  # type guard
    assert fit_file.energy is not None  # type guard
    raw = fit_file.data_raw.copy()
    fit_file.subtract_dark(np.full(fit_file.energy.size, 0.1))

    results = fit_file.p.results  # post-correction session state
    slot = next(iter(results))
    assert slot.dark is None  # fitted before the correction
    # The provider is the captured SavedFile, never the live File —
    # completed fits read the same context as after FitResults.load.
    provider = results._provider_for(slot)
    assert isinstance(provider, SavedFile)
    np.testing.assert_array_equal(np.asarray(provider.data_raw), raw)

    b0, b1 = slot.selection["base_t_ind"]
    expected_obs = np.mean(raw[b0:b1, :], axis=0)
    full_obs = results._full_observed_for(slot, provider)
    assert full_obs is not None  # type guard
    np.testing.assert_allclose(full_obs, expected_obs)
    # The file's current corrected data would not match — the dark moved it.
    corrected_obs = np.mean(np.asarray(fit_file.data)[b0:b1, :], axis=0)
    assert not np.allclose(full_obs, corrected_obs)


#
def test_completed_fit_rendering_insulated_from_live_mutation() -> None:
    """In-place mutation of the live file after a fit never reaches
    completed-fit rendering: the provider is the captured SavedFile,
    which holds its own frozen copy of the fit-time arrays. (Parity with
    a loaded archive is covered by the roundtrip tests.)"""

    _, fit_file, family = _build_fit_file("F1")
    fit_file.set_fit_limits(_narrow_energy_limits(fit_file), show_plot=False)
    fit_file.fit_baseline(model_name=family.model_name("default"), stages=1, try_ci=0)
    assert fit_file.data_raw is not None  # type guard
    raw = fit_file.data_raw.copy()

    fit_file.data_raw[:] = fit_file.data_raw + 5.0  # unsupported, but must not leak

    results = fit_file.p.results  # fresh snapshot, taken after the mutation
    slot = next(iter(results))
    provider = results._provider_for(slot)
    assert isinstance(provider, SavedFile)
    np.testing.assert_array_equal(np.asarray(provider.data_raw), raw)

    b0, b1 = slot.selection["base_t_ind"]
    full_obs = results._full_observed_for(slot, provider)
    assert full_obs is not None  # type guard
    np.testing.assert_allclose(full_obs, np.mean(raw[b0:b1, :], axis=0))


# ---------------------------------------------------------------------------
# spectrum
# ---------------------------------------------------------------------------


#
@pytest.mark.parametrize(
    ("kwargs", "expected_ref"),
    [
        pytest.param({"time_point": 10, "time_type": "ind"}, "point", id="time_point"),
        pytest.param({"time_type": "abs"}, "range", id="time_range"),
    ],
)
def test_spectrum_full_range_reconstructs_real_data(
    tmp_path, kwargs, expected_ref
) -> None:
    _, fit_file, family = _build_fit_file("F1")
    fit_file.set_fit_limits(_narrow_energy_limits(fit_file), show_plot=False)
    if expected_ref == "range":
        t = fit_file.time
        kwargs = {**kwargs, "time_range": (float(t[2]), float(t[5]))}
    fit_file.fit_spectrum(
        family.model_name("default"),
        stages=1,
        try_ci=0,
        show_plot=False,
        **kwargs,
    )

    archive_path = tmp_path / "spectrum.fit.h5"
    fit_file.p.save_fits(archive_path, show_output=0)
    loaded = FitResults.load(archive_path)
    slot = next(iter(loaded))
    provider = loaded._provider_for(slot)

    if expected_ref == "point":
        expected_obs = provider.data_raw[10, :]
    else:
        ind = resolve_time_selection(
            provider.time,
            kwargs["time_range"][0],
            kwargs["time_range"][1],
            time_type="abs",
        )
        expected_obs = np.mean(provider.data_raw[ind[0] : ind[1], :], axis=0)

    full_obs = loaded._full_observed_for(slot, provider)
    np.testing.assert_allclose(full_obs, expected_obs)

    loaded.plot_fit(
        model=family.model_name("default"),
        fit_type="spectrum",
        full_range=True,
        show_plot=False,
    )


# ---------------------------------------------------------------------------
# sbs
# ---------------------------------------------------------------------------


#
@pytest.mark.slow
def test_sbs_full_range_reconstructs_real_data(tmp_path) -> None:
    _, fit_file, family = _build_fit_file("F1", spec_fun_str="fit_model_mcp")
    fit_file.set_fit_limits(_narrow_energy_limits(fit_file), show_plot=False)
    fit_file.fit_baseline(model_name=family.model_name("default"), stages=1, try_ci=0)
    fit_file.fit_slice_by_slice(
        family.model_name("default"),
        n_workers=1,
        seed_source="model",
        seed_adapt=None,
        try_ci=0,
    )

    archive_path = tmp_path / "sbs.fit.h5"
    fit_file.p.save_fits(archive_path, show_output=0)
    loaded = FitResults.load(archive_path)
    slot = next(s for s in loaded if s.fit_type == "sbs")
    provider = loaded._provider_for(slot)

    full_obs = loaded._full_observed_for(slot, provider)
    np.testing.assert_array_equal(full_obs, provider.data_raw)

    e_lim = slot.selection["e_lim"]
    fit_padded = loaded._pad_axis(
        np.asarray(slot.fit), provider.data_raw.shape[1], e_lim, axis=1
    )
    assert np.all(np.isnan(fit_padded[:, : e_lim[0]]))
    np.testing.assert_array_equal(fit_padded[:, e_lim[0] : e_lim[1]], slot.fit)

    loaded.plot_fit(
        model=family.model_name("default"),
        fit_type="sbs",
        full_range=True,
        show_plot=False,
    )


# ---------------------------------------------------------------------------
# 2d
# ---------------------------------------------------------------------------


#
def test_2d_full_range_reconstructs_real_data(tmp_path) -> None:
    _, fit_file, family = _build_fit_file("F3", spec_fun_str="fit_model_mcp")
    fit_file.set_fit_limits(
        _narrow_energy_limits(fit_file),
        time_limits=_narrow_time_limits(fit_file),
        show_plot=False,
    )
    fit_file.fit_baseline(model_name=family.model_name("default"), stages=1, try_ci=0)
    assert family.add_dynamics is not None  # type guard
    family.add_dynamics(fit_file, "default")
    fit_file.fit_2d(family.model_name("default"), stages=1, try_ci=0)

    archive_path = tmp_path / "2d.fit.h5"
    fit_file.p.save_fits(archive_path, fit_type="2d", show_output=0)
    loaded = FitResults.load(archive_path)
    slot = next(iter(loaded))
    provider = loaded._provider_for(slot)

    full_obs = loaded._full_observed_for(slot, provider)
    np.testing.assert_array_equal(full_obs, provider.data_raw)

    e_lim = slot.selection["e_lim"]
    t_lim = slot.selection["t_lim"]
    fit_padded = loaded._pad_axis(
        np.asarray(slot.fit), provider.data_raw.shape[0], t_lim, axis=0
    )
    fit_padded = loaded._pad_axis(fit_padded, provider.data_raw.shape[1], e_lim, axis=1)
    assert np.all(np.isnan(fit_padded[: t_lim[0], :]))
    np.testing.assert_array_equal(
        fit_padded[t_lim[0] : t_lim[1], e_lim[0] : e_lim[1]], slot.fit
    )

    loaded.plot_fit(
        model=family.model_name("default"),
        fit_type="2d",
        full_range=True,
        show_plot=False,
    )


# ---------------------------------------------------------------------------
# graceful fallback
# ---------------------------------------------------------------------------


#
def test_full_range_falls_back_to_cropped_view_without_provider(tmp_path) -> None:
    """No axes provider for the slot's file -> full_range=True degrades to
    the cropped view instead of raising."""

    _, fit_file, family = _build_fit_file("F1")
    fit_file.set_fit_limits(_narrow_energy_limits(fit_file), show_plot=False)
    fit_file.fit_baseline(model_name=family.model_name("default"), stages=1, try_ci=0)

    archive_path = tmp_path / "baseline.fit.h5"
    fit_file.p.save_fits(archive_path, show_output=0)
    loaded = FitResults.load(archive_path)
    slot = next(iter(loaded))

    orphan = FitResults(slots=[slot], files=None)
    # must not raise
    orphan.plot_fit(
        model=family.model_name("default"),
        fit_type="baseline",
        full_range=True,
        show_plot=False,
    )
