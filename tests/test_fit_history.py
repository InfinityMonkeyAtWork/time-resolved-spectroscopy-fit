"""
Tests for the in-memory fit-history layer:

- Project._fit_history accumulation as fits complete.
- SavedFitSlot field correctness (observed/fit shape, residual reconstruction,
  metrics match lmfit, identity hashes, selection capture).
- Project.results snapshot semantics (immutability after access).
- FitResults find / get / files / models / iteration.
- SbS extraction survives the seed-template restoration at the end of
  fit_slice_by_slice.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from _utils import make_project, simulate_clean, simulate_noisy

from trspecfit import File, FitResults
from trspecfit.utils.fit_io import (
    SavedFitSlot,
    build_selection_json,
    compute_file_fingerprint,
    compute_history_key,
)


#
def _make_truth_file(project):
    energy = np.linspace(83, 87, 30)
    time = np.linspace(-2, 10, 24)
    file = File(parent_project=project, name="truth")
    file.energy = energy
    file.time = time
    file.dim = 2
    file.load_model(
        model_yaml="models/file_energy.yaml",
        model_info="single_glp",
    )
    file.add_time_dependence(
        target_model="single_glp",
        target_parameter="GLP_01_A",
        dynamics_yaml="models/file_time.yaml",
        dynamics_model=["MonoExpPos"],
    )
    return file


#
def _make_fit_file(project, data, energy, time, *, name="fit"):
    file = File(
        parent_project=project,
        name=name,
        data=data,
        energy=energy.copy(),
        time=time.copy(),
    )
    file.load_model(
        model_yaml="models/file_energy.yaml",
        model_info="single_glp",
    )
    return file


#
def _setup_baseline_fit():
    """Run a baseline fit and return (project, file). Uses noisy data so
    chi2 > 0 and AIC/BIC are finite (clean data gives chi2=0 -> log(0)=nan)."""

    truth_project = make_project(name="truth")
    truth = _make_truth_file(truth_project)
    data = simulate_noisy(truth.model_active, noise_level=0.01)

    project = make_project(name="fit")
    file = _make_fit_file(project, data, truth.energy, truth.time)
    file.define_baseline(time_start=0, time_stop=3, time_type="ind", show_plot=False)
    file.fit_baseline(model_name="single_glp", stages=2, try_ci=0)
    return project, file


#
# --- identity helpers --------------------------------------------------------
#


#
class TestIdentityHelpers:
    """compute_file_fingerprint, build_selection_json, compute_history_key."""

    #
    def test_fingerprint_changes_when_data_differs(self):
        rng = np.random.default_rng(0)
        e = np.linspace(0, 1, 20)
        t = np.linspace(0, 5, 10)
        d1 = rng.standard_normal((10, 20))
        d2 = rng.standard_normal((10, 20))
        fp1 = compute_file_fingerprint(data=d1, energy=e, time=t)
        fp2 = compute_file_fingerprint(data=d2, energy=e, time=t)
        assert fp1["data_sha256"] != fp2["data_sha256"]
        assert fp1["energy_sha256"] == fp2["energy_sha256"]
        assert fp1["shape"] == fp2["shape"] == (10, 20)

    #
    def test_fingerprint_handles_1d_no_time(self):
        d = np.arange(10, dtype=float)
        e = np.linspace(0, 1, 10)
        fp = compute_file_fingerprint(data=d, energy=e, time=None)
        assert fp["time_sha256"] == ""
        assert fp["shape"] == (10,)

    #
    def test_selection_json_is_deterministic(self):
        a = build_selection_json("baseline", base_t_ind=[0, 5], e_lim=[10, 20])
        b = build_selection_json("baseline", e_lim=[10, 20], base_t_ind=[0, 5])
        assert a == b  # sorted keys

    #
    def test_history_key_changes_with_selection(self):
        fp = {
            "data_sha256": "x",
            "energy_sha256": "y",
            "time_sha256": "z",
            "shape": (5,),
        }
        s1 = build_selection_json("spectrum", time_point=0.5, e_lim=None)
        s2 = build_selection_json("spectrum", time_point=1.5, e_lim=None)
        k1 = compute_history_key(
            file_fingerprint=fp,
            model_name="m",
            fit_type="spectrum",
            selection_json=s1,
        )
        k2 = compute_history_key(
            file_fingerprint=fp,
            model_name="m",
            fit_type="spectrum",
            selection_json=s2,
        )
        assert k1 != k2


#
# --- baseline slot extraction ------------------------------------------------
#


#
class TestBaselineSlot:
    #
    def test_history_grows_by_one_after_fit(self):
        project, _ = _setup_baseline_fit()
        assert len(project._fit_history) == 1

    #
    def test_slot_basic_fields(self):
        project, file = _setup_baseline_fit()
        slot = project._fit_history[0]
        assert isinstance(slot, SavedFitSlot)
        assert slot.fit_type == "baseline"
        assert slot.model_name == "single_glp"
        assert slot.file_name == file.name
        assert slot.observed.shape == slot.fit.shape
        assert slot.observed.size > 0

    #
    def test_residual_matches_observed_minus_fit(self):
        """Invariant: residuals = observed - fit, with no recipe replay."""

        project, _ = _setup_baseline_fit()
        slot = project._fit_history[0]
        residual = slot.observed - slot.fit
        # chi2 in metrics should match sum of squared residuals.
        assert slot.metrics["chi2"] == pytest.approx(float(np.sum(residual**2)))

    #
    def test_metrics_keys_present(self):
        project, _ = _setup_baseline_fit()
        slot = project._fit_history[0]
        assert set(slot.metrics.keys()) == {"chi2", "chi2_red", "r2", "aic", "bic"}
        assert all(np.isfinite(v) for v in slot.metrics.values())

    #
    def test_selection_captures_base_t_ind(self):
        project, _ = _setup_baseline_fit()
        slot = project._fit_history[0]
        # define_baseline(time_start=0, time_stop=3, time_type="ind") yields
        # the inclusive index range [0, 3] -> exclusive slice [0, 4).
        assert slot.selection["base_t_ind"] == [0, 4]

    #
    def test_history_key_is_stable(self):
        project, _ = _setup_baseline_fit()
        slot = project._fit_history[0]
        # Recompute and verify it matches.
        k = compute_history_key(
            file_fingerprint=slot.file_fingerprint,
            model_name=slot.model_name,
            fit_type=slot.fit_type,
            selection_json=slot.selection_json,
        )
        assert k == slot.history_key


#
# --- spectrum slot extraction ------------------------------------------------
#


#
class TestSpectrumSlot:
    #
    def test_spectrum_slot_records_time_point(self):
        truth_project = make_project(name="truth")
        truth = _make_truth_file(truth_project)
        data = simulate_clean(truth.model_active)

        project = make_project(name="fit")
        file = _make_fit_file(project, data, truth.energy, truth.time)
        file.fit_spectrum(
            "single_glp",
            time_point=5,
            time_type="ind",
            stages=1,
            show_plot=False,
        )

        assert len(project._fit_history) == 1
        slot = project._fit_history[0]
        assert slot.fit_type == "spectrum"
        assert slot.selection["time_point"] == 5
        assert slot.selection["time_range"] is None
        assert slot.selection["time_type"] == "ind"

    #
    def test_refit_at_different_time_point_creates_distinct_slots(self):
        """selection_json includes time_point, so refits don't collide."""

        truth_project = make_project(name="truth")
        truth = _make_truth_file(truth_project)
        data = simulate_clean(truth.model_active)

        project = make_project(name="fit")
        file = _make_fit_file(project, data, truth.energy, truth.time)
        file.fit_spectrum(
            "single_glp",
            time_point=5,
            time_type="ind",
            stages=1,
            show_plot=False,
        )
        file.fit_spectrum(
            "single_glp",
            time_point=10,
            time_type="ind",
            stages=1,
            show_plot=False,
        )
        keys = {s.history_key for s in project._fit_history}
        assert len(keys) == 2  # different selections -> different keys


#
# --- SbS slot extraction ------------------------------------------------------
#


#
class TestSbSSlot:
    #
    def test_sbs_slot_per_slice_metrics(self):
        truth_project = make_project(name="truth")
        truth = _make_truth_file(truth_project)
        data = simulate_clean(truth.model_active)

        project = make_project(name="fit")
        project.spec_fun_str = "fit_model_mcp"
        file = _make_fit_file(project, data, truth.energy, truth.time)
        file.fit_slice_by_slice(
            "single_glp",
            n_workers=1,
            seed_source="model",
            seed_adapt=None,
        )

        assert len(project._fit_history) == 1
        slot = project._fit_history[0]
        assert slot.fit_type == "sbs"
        # observed / fit are 2D, one row per time slice.
        assert slot.observed.ndim == 2
        assert slot.observed.shape == slot.fit.shape
        assert slot.observed.shape[0] == len(file.time)
        # Metrics are per-slice arrays.
        for k in ("chi2", "chi2_red", "r2", "aic", "bic"):
            assert isinstance(slot.metrics[k], np.ndarray)
            assert slot.metrics[k].shape == (len(file.time),)

    #
    def test_sbs_slot_survives_seed_template_restoration(self):
        """
        SbS ends with model_sbs.update_value(seed_template, par_select='all'),
        which would blow away live model state. The slot must already hold a
        complete snapshot before that happens.
        """

        truth_project = make_project(name="truth")
        truth = _make_truth_file(truth_project)
        data = simulate_clean(truth.model_active)

        project = make_project(name="fit")
        project.spec_fun_str = "fit_model_mcp"
        file = _make_fit_file(project, data, truth.energy, truth.time)
        file.fit_slice_by_slice(
            "single_glp",
            n_workers=1,
            seed_source="model",
            seed_adapt=None,
        )

        # After fit_slice_by_slice returns, the seed-template restoration has
        # already run. The slot must still hold valid, finite per-slice metrics
        # (built before the restoration via copied snapshot args).
        slot = project._fit_history[0]
        assert np.all(np.isfinite(slot.metrics["chi2"]))
        assert slot.params.shape[0] == len(file.time)


#
# --- 2D slot extraction -------------------------------------------------------
#


#
class TestTwoDSlot:
    #
    def test_2d_slot_basic_fields(self):
        truth_project = make_project(name="truth")
        truth = _make_truth_file(truth_project)
        data = simulate_clean(truth.model_active)

        project = make_project(name="fit")
        file = _make_fit_file(project, data, truth.energy, truth.time)
        file.define_baseline(
            time_start=0,
            time_stop=3,
            time_type="ind",
            show_plot=False,
        )
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
        # Reload as a 2D model with dynamics for fit_2d.
        file.add_time_dependence(
            target_model="single_glp",
            target_parameter="GLP_01_A",
            dynamics_yaml="models/file_time.yaml",
            dynamics_model=["MonoExpPos"],
        )
        file.fit_2d("single_glp", stages=1, try_ci=0)

        twod_slots = [s for s in project._fit_history if s.fit_type == "2d"]
        assert len(twod_slots) == 1
        slot = twod_slots[0]
        assert slot.observed.ndim == 2
        assert slot.observed.shape == slot.fit.shape
        # Residual reconstruction.
        residual = slot.observed - slot.fit
        assert slot.metrics["chi2"] == pytest.approx(float(np.sum(residual**2)))


#
# --- Project.results snapshot semantics ---------------------------------------
#


#
class TestResultsSnapshot:
    #
    def test_results_returns_fresh_wrapper(self):
        project, _ = _setup_baseline_fit()
        r1 = project.results
        r2 = project.results
        assert r1 is not r2  # fresh wrapper per access
        assert len(r1) == len(r2) == 1

    #
    def test_returned_results_is_frozen_against_subsequent_fits(self):
        """A captured FitResults does not see new history entries."""

        truth_project = make_project(name="truth")
        truth = _make_truth_file(truth_project)
        data = simulate_clean(truth.model_active)

        project = make_project(name="fit")
        file = _make_fit_file(project, data, truth.energy, truth.time)
        file.define_baseline(
            time_start=0,
            time_stop=3,
            time_type="ind",
            show_plot=False,
        )
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
        r1 = project.results
        assert len(r1) == 1
        # Run a second fit (different selection -> new slot).
        file.fit_spectrum(
            "single_glp",
            time_point=10,
            time_type="ind",
            stages=1,
            show_plot=False,
        )
        # r1 still sees only the first slot.
        assert len(r1) == 1
        # New access reflects both.
        assert len(project.results) == 2


#
# --- FitResults query API -----------------------------------------------------
#


#
def _slot_stub(*, file_name="f1", model_name="m", fit_type="baseline"):
    """Build a minimal SavedFitSlot for query-API tests (no real fit)."""

    fp = {"data_sha256": "a", "energy_sha256": "b", "time_sha256": "c", "shape": (3,)}
    selection: dict = {"base_t_ind": [0, 1], "e_lim": None}
    selection_json = build_selection_json("baseline", **selection)
    history_key = compute_history_key(
        file_fingerprint=fp,
        model_name=model_name,
        fit_type=fit_type,
        selection_json=selection_json,
    )
    import pandas as pd

    return SavedFitSlot(
        file_fingerprint=fp,
        file_name=file_name,
        model_name=model_name,
        fit_type=fit_type,
        selection=selection,
        selection_json=selection_json,
        observed_sha256="z",
        history_key=history_key,
        params=pd.DataFrame(),
        metrics={"chi2": 0.0, "chi2_red": 0.0, "r2": 1.0, "aic": 0.0, "bic": 0.0},
        observed=np.zeros(3),
        fit=np.zeros(3),
        fit_alg="leastsq",
        yaml_filename=None,
        timestamp="2026-04-30T00:00:00+00:00",
    )


#
class TestFitResultsQueryAPI:
    #
    def test_files_and_models_unique_in_order(self):
        slots = [
            _slot_stub(file_name="A", model_name="m1"),
            _slot_stub(file_name="B", model_name="m1"),
            _slot_stub(file_name="A", model_name="m2"),
        ]
        r = FitResults(slots=slots)
        assert r.files() == ["A", "B"]
        assert r.models() == ["m1", "m2"]
        assert r.models(file="A") == ["m1", "m2"]
        assert r.models(file="B") == ["m1"]

    #
    def test_find_filters_combine(self):
        slots = [
            _slot_stub(file_name="A", model_name="m1", fit_type="baseline"),
            _slot_stub(file_name="A", model_name="m2", fit_type="baseline"),
            _slot_stub(file_name="A", model_name="m1", fit_type="2d"),
        ]
        r = FitResults(slots=slots)
        assert len(r.find(model="m1")) == 2
        assert len(r.find(model="m1", fit_type="baseline")) == 1

    #
    def test_get_raises_on_zero_or_multi(self):
        slots = [
            _slot_stub(file_name="A", model_name="m1", fit_type="baseline"),
            _slot_stub(file_name="A", model_name="m1", fit_type="baseline"),
        ]
        r = FitResults(slots=slots)
        with pytest.raises(LookupError, match="2 slots match"):
            r.get(file="A", model="m1", fit_type="baseline")
        with pytest.raises(LookupError, match="No slot matches"):
            r.get(file="A", model="m_missing", fit_type="baseline")

    #
    def test_iteration(self):
        slots = [_slot_stub(file_name=f"f{i}") for i in range(3)]
        r = FitResults(slots=slots)
        assert len(list(r)) == 3
