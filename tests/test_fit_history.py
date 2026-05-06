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
import pandas as pd
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
    def test_file_fingerprint_tracks_corrections(self):
        """File.fingerprint() must reflect the current ``self.data``.

        Regression: an earlier cache held the pre-correction hash so slots
        recorded after subtract_dark / calibrate_data inherited a stale
        history_key.
        """

        rng = np.random.default_rng(0)
        energy = np.linspace(83, 87, 30)
        time = np.linspace(-2, 10, 24)
        raw = rng.standard_normal((24, 30))
        project = make_project(name="fp")
        file = File(
            parent_project=project,
            name="fp",
            data=raw,
            energy=energy,
            time=time,
        )

        fp_raw = file.fingerprint()
        file.subtract_dark(np.full(30, 0.1))
        fp_after_dark = file.fingerprint()
        file.calibrate_data(np.full(30, 1.5))
        fp_after_cal = file.fingerprint()
        file.reset_dark()
        file.reset_calibration()
        fp_reset = file.fingerprint()

        assert fp_raw["data_sha256"] != fp_after_dark["data_sha256"]
        assert fp_after_dark["data_sha256"] != fp_after_cal["data_sha256"]
        # Resetting both corrections restores the raw data hash.
        assert fp_reset["data_sha256"] == fp_raw["data_sha256"]

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
            file_name="f1",
            model_name="m",
            fit_type="spectrum",
            selection_json=s1,
        )
        k2 = compute_history_key(
            file_fingerprint=fp,
            file_name="f1",
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
            file_name=slot.file_name,
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
            try_ci=0,
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
            try_ci=0,
            show_plot=False,
        )
        file.fit_spectrum(
            "single_glp",
            time_point=10,
            time_type="ind",
            stages=1,
            try_ci=0,
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
            try_ci=0,
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
            try_ci=0,
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
# --- MCMC payload capture ----------------------------------------------------
#


#
class TestMcmcPayload:
    """fit_wrapper's emcee outputs (result[3]/[4]) flow into SavedFitSlot.mcmc.

    Without this wiring the slot's ``mcmc`` field stays None even when MCMC
    actually ran — see _mcmc_payload in utils/fit_io.py.
    """

    #
    @pytest.mark.slow
    def test_baseline_slot_captures_mcmc(self):
        from trspecfit.utils.lmfit import MC

        truth_project = make_project(name="truth")
        truth = _make_truth_file(truth_project)
        data = simulate_noisy(truth.model_active, noise_level=0.01)

        project = make_project(name="fit")
        file = _make_fit_file(project, data, truth.energy, truth.time)
        file.define_baseline(
            time_start=0, time_stop=3, time_type="ind", show_plot=False
        )
        # nwalkers > 2 * n_params for emcee's red-blue move.
        mc = MC(use_mc=1, steps=20, nwalkers=32, burn=5, thin=1, workers=1)
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0, mc_settings=mc)

        slot = project._fit_history[0]
        assert slot.mcmc is not None
        assert set(slot.mcmc.keys()) == {"flatchain", "ci", "lnsigma"}
        assert slot.mcmc["flatchain"] is not None
        assert slot.mcmc["ci"] is not None
        assert slot.mcmc["lnsigma"] is not None

    #
    def test_baseline_slot_mcmc_none_when_mcmc_skipped(self):
        project, _ = _setup_baseline_fit()  # try_ci=0, no MCMC
        slot = project._fit_history[0]
        assert slot.mcmc is None


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
            try_ci=0,
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
def _slot_stub(
    *,
    file_name="f1",
    model_name="m",
    fit_type="baseline",
    metrics=None,
    observed_sha256="z",
    fingerprint=None,
    selection=None,
):
    """Build a minimal SavedFitSlot for query-API tests (no real fit)."""

    fp = fingerprint or {
        "data_sha256": "a",
        "energy_sha256": "b",
        "time_sha256": "c",
        "shape": (3,),
    }
    if selection is None:
        selection = (
            {"base_t_ind": [0, 1], "e_lim": None} if fit_type == "baseline" else {}
        )
    selection_json = build_selection_json(fit_type, **selection)
    history_key = compute_history_key(
        file_fingerprint=fp,
        file_name=file_name,
        model_name=model_name,
        fit_type=fit_type,
        selection_json=selection_json,
    )
    if metrics is None:
        metrics = {"chi2": 0.0, "chi2_red": 0.0, "r2": 1.0, "aic": 0.0, "bic": 0.0}
    import pandas as pd

    return SavedFitSlot(
        file_fingerprint=fp,
        file_name=file_name,
        model_name=model_name,
        fit_type=fit_type,
        selection=selection,
        selection_json=selection_json,
        observed_sha256=observed_sha256,
        history_key=history_key,
        params=pd.DataFrame(),
        metrics=metrics,
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


#
class TestFitResultsCompareModels:
    """Tests for FitResults.compare_models: scalar, sbs aggregation, checks."""

    #
    @staticmethod
    def _scalar_metrics(*, chi2_red, r2, aic, bic, chi2=None):
        """Build a metrics dict with the canonical 5 keys."""

        return {
            "chi2": float(chi2) if chi2 is not None else float(chi2_red),
            "chi2_red": float(chi2_red),
            "r2": float(r2),
            "aic": float(aic),
            "bic": float(bic),
        }

    #
    def test_default_returns_columns_and_one_row_per_slot(self):
        slots = [
            _slot_stub(
                file_name="A",
                model_name="m1",
                fit_type="baseline",
                metrics=self._scalar_metrics(chi2_red=1.5, r2=0.9, aic=10.0, bic=12.0),
            ),
            _slot_stub(
                file_name="A",
                model_name="m2",
                fit_type="baseline",
                metrics=self._scalar_metrics(chi2_red=0.8, r2=0.95, aic=8.0, bic=10.0),
            ),
        ]
        df = FitResults(slots=slots).compare_models()
        assert list(df.columns) == [
            "file",
            "model",
            "fit_type",
            "selection_json",
            "chi2_red",
            "r2",
            "aic",
            "bic",
        ]
        assert len(df) == 2
        assert set(df["model"]) == {"m1", "m2"}
        assert df.loc[df["model"] == "m1", "aic"].iloc[0] == 10.0
        assert df.loc[df["model"] == "m2", "aic"].iloc[0] == 8.0

    #
    def test_filters_by_file_models_and_fit_type(self):
        slots = [
            _slot_stub(
                file_name="A",
                model_name="m1",
                fit_type="baseline",
                metrics=self._scalar_metrics(chi2_red=1.0, r2=0.9, aic=1, bic=1),
            ),
            _slot_stub(
                file_name="B",
                model_name="m1",
                fit_type="baseline",
                metrics=self._scalar_metrics(chi2_red=1.0, r2=0.9, aic=1, bic=1),
                fingerprint={
                    "data_sha256": "B",
                    "energy_sha256": "b",
                    "time_sha256": "c",
                    "shape": (3,),
                },
            ),
            _slot_stub(
                file_name="A",
                model_name="m2",
                fit_type="2d",
                selection={"e_lim": None, "t_lim": None},
                metrics=self._scalar_metrics(chi2_red=1.0, r2=0.9, aic=1, bic=1),
            ),
        ]
        r = FitResults(slots=slots)
        assert len(r.compare_models(file="A")) == 2
        assert len(r.compare_models(file="A", models=["m1"])) == 1
        assert len(r.compare_models(fit_type="baseline")) == 2
        assert len(r.compare_models(fit_type=["baseline", "2d"])) == 3

    #
    def test_custom_metrics_subset(self):
        slot = _slot_stub(
            file_name="A",
            model_name="m1",
            fit_type="baseline",
            metrics=self._scalar_metrics(
                chi2=2.0,
                chi2_red=0.5,
                r2=0.99,
                aic=5.0,
                bic=7.0,
            ),
        )
        df = FitResults(slots=[slot]).compare_models(metrics=["chi2", "r2"])
        assert list(df.columns) == [
            "file",
            "model",
            "fit_type",
            "selection_json",
            "chi2",
            "r2",
        ]
        assert df["chi2"].iloc[0] == 2.0
        assert df["r2"].iloc[0] == 0.99

    #
    def test_unknown_metric_raises_keyerror(self):
        slot = _slot_stub(
            metrics=self._scalar_metrics(chi2_red=1, r2=1, aic=1, bic=1),
        )
        with pytest.raises(KeyError, match="bogus"):
            FitResults(slots=[slot]).compare_models(metrics=["bogus"])

    #
    def test_sbs_aggregation_modes(self):
        per_slice = {
            "chi2": np.array([1.0, 2.0, 3.0]),
            "chi2_red": np.array([0.1, 0.2, 0.3]),
            "r2": np.array([0.9, 0.8, 0.95]),
            "aic": np.array([10.0, 20.0, 30.0]),
            "bic": np.array([12.0, 22.0, 32.0]),
        }
        slot = _slot_stub(
            file_name="A",
            model_name="m_sbs",
            fit_type="sbs",
            selection={"e_lim": None, "t_lim": None},
            metrics=per_slice,
        )
        r = FitResults(slots=[slot])

        df_med = r.compare_models(sbs_aggregation="median")
        assert df_med["chi2_red"].iloc[0] == pytest.approx(0.2)
        assert df_med["aic"].iloc[0] == pytest.approx(20.0)

        df_mean = r.compare_models(sbs_aggregation="mean")
        assert df_mean["chi2_red"].iloc[0] == pytest.approx(0.2)
        assert df_mean["r2"].iloc[0] == pytest.approx((0.9 + 0.8 + 0.95) / 3)

        df_sum = r.compare_models(sbs_aggregation="sum")
        assert df_sum["aic"].iloc[0] == pytest.approx(60.0)
        assert df_sum["bic"].iloc[0] == pytest.approx(66.0)

    #
    def test_sbs_long_mode_emits_per_slice_rows(self):
        per_slice = {
            "chi2": np.array([1.0, 2.0]),
            "chi2_red": np.array([0.1, 0.2]),
            "r2": np.array([0.9, 0.8]),
            "aic": np.array([10.0, 20.0]),
            "bic": np.array([12.0, 22.0]),
        }
        sbs_slot = _slot_stub(
            file_name="A",
            model_name="m_sbs",
            fit_type="sbs",
            selection={"e_lim": None, "t_lim": None},
            metrics=per_slice,
        )
        baseline_slot = _slot_stub(
            file_name="B",
            model_name="m_base",
            fit_type="baseline",
            metrics=self._scalar_metrics(chi2_red=0.5, r2=0.99, aic=5, bic=7),
            fingerprint={
                "data_sha256": "B",
                "energy_sha256": "b",
                "time_sha256": "c",
                "shape": (3,),
            },
        )
        df = FitResults(slots=[sbs_slot, baseline_slot]).compare_models(
            sbs_aggregation="long"
        )
        assert "slice_index" in df.columns
        sbs_rows = df[df["model"] == "m_sbs"]
        assert len(sbs_rows) == 2
        assert list(sbs_rows["slice_index"]) == [0, 1]
        assert sbs_rows["aic"].tolist() == [10.0, 20.0]
        baseline_rows = df[df["model"] == "m_base"]
        assert len(baseline_rows) == 1
        assert pd.isna(baseline_rows["slice_index"].iloc[0])

    #
    def test_observed_mismatch_raises(self):
        """Two slots on same (file, fit_type) with different observed_sha256 → raise."""

        slots = [
            _slot_stub(
                file_name="A",
                model_name="m1",
                fit_type="baseline",
                metrics=self._scalar_metrics(chi2_red=1, r2=1, aic=1, bic=1),
                observed_sha256="hash_A",
            ),
            _slot_stub(
                file_name="A",
                model_name="m2",
                fit_type="baseline",
                selection={"base_t_ind": [0, 5], "e_lim": None},
                metrics=self._scalar_metrics(chi2_red=1, r2=1, aic=1, bic=1),
                observed_sha256="hash_B",
            ),
        ]
        r = FitResults(slots=slots)
        with pytest.raises(ValueError, match="observed_sha256"):
            r.compare_models(file="A", fit_type="baseline")

    #
    def test_observed_mismatch_allowed_across_different_fit_types(self):
        """Same file, different fit_type — observed differs legitimately, no raise."""

        slots = [
            _slot_stub(
                file_name="A",
                model_name="m1",
                fit_type="baseline",
                metrics=self._scalar_metrics(chi2_red=1, r2=1, aic=1, bic=1),
                observed_sha256="hash_A",
            ),
            _slot_stub(
                file_name="A",
                model_name="m1",
                fit_type="2d",
                selection={"e_lim": None, "t_lim": None},
                metrics=self._scalar_metrics(chi2_red=2, r2=0.5, aic=5, bic=7),
                observed_sha256="hash_B",
            ),
        ]
        df = FitResults(slots=slots).compare_models(file="A")
        assert len(df) == 2

    #
    def test_observed_mismatch_allowed_across_different_files(self):
        """Same fit_type on different files — observed differs legitimately."""

        slots = [
            _slot_stub(
                file_name="A",
                model_name="m1",
                fit_type="baseline",
                metrics=self._scalar_metrics(chi2_red=1, r2=1, aic=1, bic=1),
                observed_sha256="hash_A",
            ),
            _slot_stub(
                file_name="B",
                model_name="m1",
                fit_type="baseline",
                metrics=self._scalar_metrics(chi2_red=2, r2=0.5, aic=5, bic=7),
                observed_sha256="hash_B",
                fingerprint={
                    "data_sha256": "B",
                    "energy_sha256": "b",
                    "time_sha256": "c",
                    "shape": (3,),
                },
            ),
        ]
        df = FitResults(slots=slots).compare_models(fit_type="baseline")
        assert len(df) == 2

    #
    def test_observed_mismatch_allowed_across_replicate_files(self):
        """Two distinct files with byte-identical raw arrays but different names.

        Project identity treats them as separate files (history_key folds in
        file_name), so a fit_type-wide compare must not collapse them and
        falsely raise on observed_sha256.
        """

        shared_fp = {
            "data_sha256": "same",
            "energy_sha256": "same",
            "time_sha256": "same",
            "shape": (3,),
        }
        slots = [
            _slot_stub(
                file_name="rep_A",
                model_name="m1",
                fit_type="baseline",
                metrics=self._scalar_metrics(chi2_red=1, r2=1, aic=1, bic=1),
                observed_sha256="hash_A",
                fingerprint=shared_fp,
            ),
            _slot_stub(
                file_name="rep_B",
                model_name="m1",
                fit_type="baseline",
                metrics=self._scalar_metrics(chi2_red=2, r2=0.5, aic=5, bic=7),
                observed_sha256="hash_B",
                fingerprint=shared_fp,
            ),
        ]
        df = FitResults(slots=slots).compare_models(fit_type="baseline")
        assert len(df) == 2
        assert set(df["file"]) == {"rep_A", "rep_B"}

    #
    def test_file_arg_accepts_object_with_name_attr(self):
        slot = _slot_stub(
            file_name="A",
            model_name="m1",
            metrics=self._scalar_metrics(chi2_red=1, r2=1, aic=1, bic=1),
        )

        class _Stub:
            name = "A"

        df = FitResults(slots=[slot]).compare_models(file=_Stub())
        assert len(df) == 1
        assert df["file"].iloc[0] == "A"

    #
    def test_file_arg_invalid_type_raises(self):
        slot = _slot_stub()
        with pytest.raises(TypeError, match="file must be"):
            FitResults(slots=[slot]).compare_models(file=42)

    #
    def test_empty_match_returns_empty_dataframe(self):
        slot = _slot_stub(file_name="A", model_name="m1")
        df = FitResults(slots=[slot]).compare_models(file="missing")
        assert df.empty
        assert "model" in df.columns

    #
    def test_unknown_sbs_aggregation_raises(self):
        per_slice = {
            "chi2": np.array([1.0]),
            "chi2_red": np.array([0.1]),
            "r2": np.array([0.9]),
            "aic": np.array([10.0]),
            "bic": np.array([12.0]),
        }
        slot = _slot_stub(
            file_name="A",
            model_name="m_sbs",
            fit_type="sbs",
            selection={"e_lim": None, "t_lim": None},
            metrics=per_slice,
        )
        with pytest.raises(ValueError, match="unknown sbs_aggregation"):
            FitResults(slots=[slot]).compare_models(sbs_aggregation="bogus")  # type: ignore[arg-type]


#
class TestFitResultsPlotResiduals:
    """Smoke tests for FitResults.plot_residuals — figure construction only."""

    #
    @staticmethod
    def _slot_with_arrays(
        *,
        file_name="A",
        model_name="m1",
        fit_type="baseline",
        observed,
        fit,
        selection=None,
    ):
        """Build a slot with custom observed/fit arrays for plotting."""

        slot = _slot_stub(
            file_name=file_name,
            model_name=model_name,
            fit_type=fit_type,
            selection=selection,
        )
        return SavedFitSlot(
            file_fingerprint=slot.file_fingerprint,
            file_name=slot.file_name,
            model_name=slot.model_name,
            fit_type=slot.fit_type,
            selection=slot.selection,
            selection_json=slot.selection_json,
            observed_sha256=slot.observed_sha256,
            history_key=slot.history_key,
            params=slot.params,
            metrics=slot.metrics,
            observed=np.asarray(observed),
            fit=np.asarray(fit),
            fit_alg=slot.fit_alg,
            yaml_filename=slot.yaml_filename,
            timestamp=slot.timestamp,
        )

    #
    def test_1d_fit_returns_figure(self):
        slot_a = self._slot_with_arrays(
            model_name="m1",
            observed=np.linspace(0, 1, 30),
            fit=np.linspace(0, 1, 30) + 0.05,
        )
        slot_b = self._slot_with_arrays(
            model_name="m2",
            observed=np.linspace(0, 1, 30),
            fit=np.linspace(0, 1, 30) - 0.02,
        )
        fig = FitResults(slots=[slot_a, slot_b]).plot_residuals(
            file="A", show_plot=False
        )
        assert fig is not None
        assert len(fig.axes) >= 4

    #
    def test_2d_fit_returns_figure(self):
        obs = np.random.RandomState(0).randn(8, 12)
        fit = obs + np.random.RandomState(1).randn(8, 12) * 0.1
        slot = self._slot_with_arrays(
            model_name="m_2d",
            fit_type="2d",
            selection={"e_lim": None, "t_lim": None},
            observed=obs,
            fit=fit,
        )
        fig = FitResults(slots=[slot]).plot_residuals(file="A", show_plot=False)
        assert fig is not None
        assert len(fig.axes) >= 1

    #
    def test_no_match_raises(self):
        slot = self._slot_with_arrays(
            observed=np.zeros(5),
            fit=np.zeros(5),
        )
        with pytest.raises(LookupError, match="No slots match"):
            FitResults(slots=[slot]).plot_residuals(file="missing", show_plot=False)

    #
    def test_mixed_fit_types_requires_disambiguation(self):
        slot_b = self._slot_with_arrays(
            model_name="m1",
            fit_type="baseline",
            observed=np.zeros(5),
            fit=np.zeros(5),
        )
        slot_2d = self._slot_with_arrays(
            model_name="m2",
            fit_type="2d",
            selection={"e_lim": None, "t_lim": None},
            observed=np.zeros((3, 5)),
            fit=np.zeros((3, 5)),
        )
        r = FitResults(slots=[slot_b, slot_2d])
        with pytest.raises(ValueError, match="span fit_types"):
            r.plot_residuals(file="A", show_plot=False)
        # Disambiguating works:
        fig = r.plot_residuals(file="A", fit_type="baseline", show_plot=False)
        assert fig is not None

    #
    def test_missing_file_arg_raises(self):
        slot = self._slot_with_arrays(observed=np.zeros(5), fit=np.zeros(5))
        with pytest.raises(ValueError, match="requires file"):
            FitResults(slots=[slot]).plot_residuals(
                file=None,
                show_plot=False,  # type: ignore[arg-type]
            )
