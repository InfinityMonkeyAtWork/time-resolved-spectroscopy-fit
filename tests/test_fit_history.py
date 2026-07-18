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
    _compute_sigma_eff,
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
        # chi2_raw in metrics should match sum of squared residuals (the
        # lmfit-unweighted SSE diagnostic; chi2 is the σ-calibrated form).
        assert slot.metrics["chi2_raw"] == pytest.approx(float(np.sum(residual**2)))

    #
    def test_metrics_keys_present(self):
        project, _ = _setup_baseline_fit()
        slot = project._fit_history[0]
        assert set(slot.metrics.keys()) == {
            "chi2_raw",
            "chi2_red_raw",
            "chi2",
            "chi2_red",
            "r2",
            "aic",
            "bic",
        }
        # Raw + dimensionless metrics are always finite for a successful fit.
        for k in ("chi2_raw", "chi2_red_raw", "r2", "aic", "bic"):
            assert np.isfinite(slot.metrics[k])
        # Calibrated metrics are NaN when no sigma was set on the file.
        assert np.isnan(slot.metrics["chi2"])
        assert np.isnan(slot.metrics["chi2_red"])

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
    @pytest.mark.slow
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
        # The slot-backed accessor serves the wide per-slice params frame.
        sbs_df = file.get_fit_results(fit_type="sbs")
        pd.testing.assert_frame_equal(sbs_df, slot.params)
        assert len(sbs_df) == len(file.time)
        # Shared per-parameter metadata, column-aligned with the wide frame.
        assert slot.params_meta is not None  # type guard
        assert list(slot.params_meta.columns) == ["name", "vary", "min", "max", "expr"]
        assert list(slot.params_meta["name"]) == list(slot.params.columns)
        assert bool(slot.params_meta["vary"].any())
        # Per-slice stderr mirrors the wide params frame's shape.
        assert slot.params_stderr is not None  # type guard
        assert slot.params_stderr.shape == slot.params.shape
        assert list(slot.params_stderr.columns) == list(slot.params.columns)
        # Provenance records the SbS seeding recipe.
        assert slot.fit_settings is not None  # type guard
        assert slot.fit_settings["seed_source"] == "model"
        assert slot.fit_settings["seed_adapt"] is None
        assert slot.fit_settings["seed_values"] is None
        assert slot.fit_settings["stages"] == 1

    #
    @pytest.mark.slow
    def test_sbs_slot_records_explicit_dict_seed(self):
        """Explicit dict seeds land in fit_settings as the normalized,
        parameter-ordered float list (regression: the raw dict used to be
        np.asarray()'d, which raised TypeError at slot capture)."""

        truth_project = make_project(name="truth")
        truth = _make_truth_file(truth_project)
        data = simulate_clean(truth.model_active)

        project = make_project(name="fit")
        project.spec_fun_str = "fit_model_mcp"
        file = _make_fit_file(project, data, truth.energy, truth.time)
        model = file.model_active
        seed_values = {
            name: model.lmfit_pars[name].value for name in model.parameter_names
        }
        file.fit_slice_by_slice(
            "single_glp",
            n_workers=1,
            seed_source="explicit",
            seed_values=seed_values,
            seed_adapt=None,
            try_ci=0,
        )

        settings = project._fit_history[0].fit_settings
        assert settings is not None  # type guard
        assert settings["seed_source"] == "explicit"
        assert settings["seed_values"] == [
            float(seed_values[name]) for name in model.parameter_names
        ]

    #
    @pytest.mark.slow
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
        assert np.all(np.isfinite(slot.metrics["chi2_raw"]))
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
        assert slot.metrics["chi2_raw"] == pytest.approx(float(np.sum(residual**2)))


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
    def test_baseline_slot_captures_mcmc(self, tmp_path):
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
        assert set(slot.mcmc.keys()) == {
            "flatchain",
            "ci",
            "lnsigma",
            "acceptance_fraction",
        }
        assert slot.mcmc["flatchain"] is not None
        assert slot.mcmc["ci"] is not None
        assert slot.mcmc["lnsigma"] is not None
        # emcee's acceptance fraction is per-walker.
        acceptance = slot.mcmc["acceptance_fraction"]
        assert acceptance is not None  # type guard
        assert acceptance.shape == (32,)
        # MCMC settings land in the fit_settings provenance.
        assert slot.fit_settings is not None  # type guard
        assert slot.fit_settings["mc"]["steps"] == 20
        assert slot.fit_settings["mc"]["nwalkers"] == 32
        assert slot.fit_settings["mc"]["burn"] == 5

        # acceptance_fraction survives the archive round-trip (schema 3).
        archive_path = tmp_path / "mcmc.fit.h5"
        project.save_fits(archive_path, show_output=0)
        loaded_results = FitResults.load(archive_path)
        loaded = next(iter(loaded_results))
        assert loaded.mcmc is not None  # type guard
        np.testing.assert_array_equal(loaded.mcmc["acceptance_fraction"], acceptance)

        # ... and the slot-backed accessor serves it from the loaded archive.
        mcmc_res = loaded_results.get_mcmc(file="fit", fit_type="baseline")
        assert mcmc_res.acceptance_fraction is not None  # type guard
        np.testing.assert_array_equal(mcmc_res.acceptance_fraction, acceptance)
        assert not mcmc_res.table.empty
        assert not mcmc_res.flatchain.empty

        # plot_mcmc reproduces the fit-time diagnostics from the persisted
        # payload — live history and loaded archive alike.
        import matplotlib.pyplot as plt

        n_figs = len(plt.get_fignums())
        file.plot_mcmc(fit_type="baseline", show_plot=False)
        loaded_results.plot_mcmc(file="fit", fit_type="baseline", show_plot=False)
        assert len(plt.get_fignums()) == n_figs

    #
    def test_baseline_slot_mcmc_none_when_mcmc_skipped(self):
        project, _ = _setup_baseline_fit()  # try_ci=0, no MCMC
        slot = project._fit_history[0]
        assert slot.mcmc is None


#
# --- fit_settings provenance ---------------------------------------------------
#


#
class TestFitSettingsProvenance:
    """Every fit type records its optimizer configuration in the slot."""

    #
    def test_baseline_slot_records_fit_settings(self):
        project, _ = _setup_baseline_fit()  # stages=2, try_ci=0
        slot = project._fit_history[0]
        assert slot.fit_settings == {
            "stages": 2,
            "fit_alg_1": "Nelder",
            "fit_alg_2": "leastsq",
            "try_ci": 0,
        }
        # Non-sbs slots carry no sbs-only payloads.
        assert slot.params_meta is None
        assert slot.params_stderr is None

    #
    def test_fit_settings_records_custom_algorithms(self):
        truth_project = make_project(name="truth")
        truth = _make_truth_file(truth_project)
        data = simulate_noisy(truth.model_active, noise_level=0.01)
        project = make_project(name="fit")
        file = _make_fit_file(project, data, truth.energy, truth.time)
        file.define_baseline(
            time_start=0, time_stop=3, time_type="ind", show_plot=False
        )
        file.fit_baseline(
            model_name="single_glp", stages=1, fit_alg_1="leastsq", try_ci=0
        )
        settings = project._fit_history[0].fit_settings
        assert settings is not None  # type guard
        assert settings["fit_alg_1"] == "leastsq"
        assert settings["stages"] == 1


#
# --- slot-backed get_* accessors ----------------------------------------------
#


#
class TestSlotBackedAccessors:
    """FitResults.get_fit_results / get_correlations / get_conf_intervals /
    get_mcmc read the latest matching SavedFitSlot; the File.get_* methods
    are thin sugar delegating with file=self."""

    #
    def test_file_sugar_matches_fitresults_accessor(self):
        project, file = _setup_baseline_fit()
        via_file = file.get_fit_results(fit_type="baseline")
        via_results = project.results.get_fit_results(file=file, fit_type="baseline")
        pd.testing.assert_frame_equal(via_file, via_results)
        # ... and both match the slot payload.
        pd.testing.assert_frame_equal(via_file, project._fit_history[0].params)

    #
    def test_returned_frame_is_a_copy(self):
        """Accessors hand out copies — mutating the return value must not
        desynchronize the persisted slot."""

        project, file = _setup_baseline_fit()
        df = file.get_fit_results(fit_type="baseline")
        df.loc[0, "value"] = -999.0
        assert project._fit_history[0].params.loc[0, "value"] != -999.0

    #
    def test_get_mcmc_payload_is_a_copy(self):
        """get_mcmc must not alias the slot's stored arrays/frames —
        np.asarray on an ndarray is a no-copy passthrough (regression)."""

        import dataclasses

        project, _ = _setup_baseline_fit()
        payload = {
            "flatchain": pd.DataFrame({"GLP_01_A": [1.0, 2.0]}),
            "ci": pd.DataFrame({"par[v]/sigma[>]": ["GLP_01_A"], "best fit": [1.5]}),
            "lnsigma": None,
            "acceptance_fraction": np.array([0.3, 0.4]),
        }
        slot = dataclasses.replace(project._fit_history[0], mcmc=payload)
        res = FitResults(slots=[slot]).get_mcmc(fit_type="baseline")

        assert res.acceptance_fraction is not None  # type guard
        res.acceptance_fraction[0] = -1.0
        res.flatchain.loc[0, "GLP_01_A"] = -999.0
        res.table.loc[0, "best fit"] = -999.0
        np.testing.assert_array_equal(
            payload["acceptance_fraction"], np.array([0.3, 0.4])
        )
        assert payload["flatchain"].loc[0, "GLP_01_A"] == 1.0
        assert payload["ci"].loc[0, "best fit"] == 1.5

    #
    def test_latest_slot_wins_after_refit(self):
        project, file = _setup_baseline_fit()
        assert file.data_base is not None  # type guard
        # Refit against rescaled data: same (file, model, fit_type, selection)
        # → a second slot appends, and the accessors must serve the newer one.
        file.data_base = file.data_base * 1.5
        file.fit_baseline(model_name="single_glp", stages=2, try_ci=0)
        assert len(project._fit_history) == 2

        df = file.get_fit_results(fit_type="baseline")
        pd.testing.assert_frame_equal(df, project._fit_history[-1].params)
        first_values = project._fit_history[0].params["value"].to_numpy()
        assert not np.allclose(df["value"].to_numpy(), first_values)

    #
    def test_get_correlations_raises_without_covariance(self):
        """A slot with correl=None (covariance-less optimizer, project joint
        fit) must produce a clear error, not a fabricated identity matrix."""

        import dataclasses

        project, _ = _setup_baseline_fit()
        slot = dataclasses.replace(project._fit_history[0], correl=None)
        results = FitResults(slots=[slot])
        with pytest.raises(ValueError, match="reported no covariance"):
            results.get_correlations(fit_type="baseline")

    #
    def test_get_mcmc_tolerates_missing_acceptance(self):
        """Slots loaded from schema-2 archives carry acceptance_fraction=None;
        get_mcmc must still serve table/flatchain."""

        import dataclasses

        project, _ = _setup_baseline_fit()
        v2_payload = {
            "flatchain": pd.DataFrame({"GLP_01_A": [1.0, 2.0]}),
            "ci": None,
            "lnsigma": None,
            "acceptance_fraction": None,
        }
        slot = dataclasses.replace(project._fit_history[0], mcmc=v2_payload)
        res = FitResults(slots=[slot]).get_mcmc(fit_type="baseline")
        assert res.acceptance_fraction is None
        assert res.table.empty
        assert list(res.flatchain.columns) == ["GLP_01_A"]


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
    sigma_data=float("nan"),
    noise_type=None,
    sigma_source="user_supplied",
    sigma_type="constant",
):
    """Build a minimal SavedFitSlot for query-API tests (no real fit).

    ``sigma_data`` defaults to ``NaN`` (file had no sigma set); pass a
    positive number to exercise the σ-calibrated code paths. ``noise_type``
    follows from ``sigma_data`` when omitted (``"gaussian"`` if finite,
    ``"unknown"`` otherwise).
    """

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
    sigma_data_f = float(sigma_data)
    is_unset = not np.isfinite(sigma_data_f)
    sigma_eff = (
        float("nan")
        if is_unset
        else _compute_sigma_eff(fit_type, selection, sigma_data_f)
    )
    if noise_type is None:
        noise_type = "unknown" if is_unset else "gaussian"
    if metrics is None:
        metrics = {
            "chi2_raw": 0.0,
            "chi2_red_raw": 0.0,
            "chi2": float("nan") if is_unset else 0.0,
            "chi2_red": float("nan") if is_unset else 0.0,
            "r2": 1.0,
            "aic": 0.0,
            "bic": 0.0,
        }
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
        noise_type=noise_type,
        sigma_source=sigma_source,
        sigma_type=sigma_type,
        sigma_data=sigma_data_f,
        sigma_eff=sigma_eff,
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
    def _scalar_metrics(*, chi2_red_raw, r2, aic, bic, chi2_raw=None):
        """Build a metrics dict with the 7-key schema.

        Calibrated ``chi2`` / ``chi2_red`` are populated as NaN — slots
        built via this helper represent the "no σ set on file" case.
        Tests that need calibrated values should pass ``sigma_data`` to
        ``_slot_stub`` and build the per-key dict by hand.
        """

        chi2_raw_v = float(chi2_raw) if chi2_raw is not None else float(chi2_red_raw)
        return {
            "chi2_raw": chi2_raw_v,
            "chi2_red_raw": float(chi2_red_raw),
            "chi2": float("nan"),
            "chi2_red": float("nan"),
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
                metrics=self._scalar_metrics(
                    chi2_red_raw=1.5, r2=0.9, aic=10.0, bic=12.0
                ),
            ),
            _slot_stub(
                file_name="A",
                model_name="m2",
                fit_type="baseline",
                metrics=self._scalar_metrics(
                    chi2_red_raw=0.8, r2=0.95, aic=8.0, bic=10.0
                ),
            ),
        ]
        df = FitResults(slots=slots).compare_models()
        assert list(df.columns) == [
            "file",
            "model",
            "fit_type",
            "selection_json",
            "chi2_red_raw",
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
                metrics=self._scalar_metrics(chi2_red_raw=1.0, r2=0.9, aic=1, bic=1),
            ),
            _slot_stub(
                file_name="B",
                model_name="m1",
                fit_type="baseline",
                metrics=self._scalar_metrics(chi2_red_raw=1.0, r2=0.9, aic=1, bic=1),
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
                metrics=self._scalar_metrics(chi2_red_raw=1.0, r2=0.9, aic=1, bic=1),
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
                chi2_raw=2.0,
                chi2_red_raw=0.5,
                r2=0.99,
                aic=5.0,
                bic=7.0,
            ),
        )
        df = FitResults(slots=[slot]).compare_models(metrics=["chi2_raw", "r2"])
        assert list(df.columns) == [
            "file",
            "model",
            "fit_type",
            "selection_json",
            "chi2_raw",
            "r2",
        ]
        assert df["chi2_raw"].iloc[0] == 2.0
        assert df["r2"].iloc[0] == 0.99

    #
    def test_unknown_metric_raises_keyerror(self):
        slot = _slot_stub(
            metrics=self._scalar_metrics(chi2_red_raw=1, r2=1, aic=1, bic=1),
        )
        with pytest.raises(KeyError, match="bogus"):
            FitResults(slots=[slot]).compare_models(metrics=["bogus"])

    #
    def test_sbs_aggregation_modes(self):
        # No σ → calibrated columns are absent from the default; assertions
        # target the raw column. (See TestFitResultsCompareModelsSigmaColumns
        # for the σ-calibrated equivalents.)
        per_slice = {
            "chi2_raw": np.array([1.0, 2.0, 3.0]),
            "chi2_red_raw": np.array([0.1, 0.2, 0.3]),
            "chi2": np.array([float("nan")] * 3),
            "chi2_red": np.array([float("nan")] * 3),
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
        assert df_med["chi2_red_raw"].iloc[0] == pytest.approx(0.2)
        assert df_med["aic"].iloc[0] == pytest.approx(20.0)

        df_mean = r.compare_models(sbs_aggregation="mean")
        assert df_mean["chi2_red_raw"].iloc[0] == pytest.approx(0.2)
        assert df_mean["r2"].iloc[0] == pytest.approx((0.9 + 0.8 + 0.95) / 3)

        df_sum = r.compare_models(sbs_aggregation="sum")
        assert df_sum["aic"].iloc[0] == pytest.approx(60.0)
        assert df_sum["bic"].iloc[0] == pytest.approx(66.0)
        # chi2_red_raw in sum mode is aggregate-reduced-chi-square:
        # Σchi2_raw / ΣDoF with DoF_i = chi2_raw_i / chi2_red_raw_i = [10, 10, 10],
        # so aggregate = 6 / 30 = 0.2 (not Σ chi2_red_raw = 0.6).
        assert df_sum["chi2_red_raw"].iloc[0] == pytest.approx(0.2)

    #
    def test_sbs_long_mode_emits_per_slice_rows(self):
        per_slice = {
            "chi2_raw": np.array([1.0, 2.0]),
            "chi2_red_raw": np.array([0.1, 0.2]),
            "chi2": np.array([float("nan"), float("nan")]),
            "chi2_red": np.array([float("nan"), float("nan")]),
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
            metrics=self._scalar_metrics(chi2_red_raw=0.5, r2=0.99, aic=5, bic=7),
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
        baseline_rows = df.loc[df["model"] == "m_base"]
        assert len(baseline_rows) == 1
        assert pd.isna(baseline_rows["slice_index"].iloc[0])

    #
    def test_sbs_long_mode_is_slice_major(self):
        """Long-form interleaves models by slice so head() compares them."""

        per_slice = {
            "chi2_raw": np.array([1.0, 2.0, 3.0]),
            "chi2_red_raw": np.array([0.1, 0.2, 0.3]),
            "chi2": np.array([float("nan")] * 3),
            "chi2_red": np.array([float("nan")] * 3),
            "r2": np.array([0.9, 0.8, 0.7]),
            "aic": np.array([10.0, 20.0, 30.0]),
            "bic": np.array([12.0, 22.0, 32.0]),
        }
        slot_A = _slot_stub(
            file_name="F",
            model_name="sbsA",
            fit_type="sbs",
            selection={"e_lim": None, "t_lim": None},
            metrics=per_slice,
        )
        slot_B = _slot_stub(
            file_name="F",
            model_name="sbsB",
            fit_type="sbs",
            selection={"e_lim": None, "t_lim": None},
            metrics=per_slice,
        )
        # Slot order is A, B; slice-major output groups both models per slice.
        df = FitResults(slots=[slot_A, slot_B]).compare_models(
            fit_type="sbs", sbs_aggregation="long"
        )
        assert list(df["slice_index"]) == [0, 0, 1, 1, 2, 2]
        # Stable sort preserves slot order (A before B) within each slice.
        assert list(df["model"]) == ["sbsA", "sbsB"] * 3

    #
    def test_observed_mismatch_raises(self):
        """Two slots on same (file, fit_type) with different observed_sha256 → raise."""

        slots = [
            _slot_stub(
                file_name="A",
                model_name="m1",
                fit_type="baseline",
                metrics=self._scalar_metrics(chi2_red_raw=1, r2=1, aic=1, bic=1),
                observed_sha256="hash_A",
            ),
            _slot_stub(
                file_name="A",
                model_name="m2",
                fit_type="baseline",
                selection={"base_t_ind": [0, 5], "e_lim": None},
                metrics=self._scalar_metrics(chi2_red_raw=1, r2=1, aic=1, bic=1),
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
                metrics=self._scalar_metrics(chi2_red_raw=1, r2=1, aic=1, bic=1),
                observed_sha256="hash_A",
            ),
            _slot_stub(
                file_name="A",
                model_name="m1",
                fit_type="2d",
                selection={"e_lim": None, "t_lim": None},
                metrics=self._scalar_metrics(chi2_red_raw=2, r2=0.5, aic=5, bic=7),
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
                metrics=self._scalar_metrics(chi2_red_raw=1, r2=1, aic=1, bic=1),
                observed_sha256="hash_A",
            ),
            _slot_stub(
                file_name="B",
                model_name="m1",
                fit_type="baseline",
                metrics=self._scalar_metrics(chi2_red_raw=2, r2=0.5, aic=5, bic=7),
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
                metrics=self._scalar_metrics(chi2_red_raw=1, r2=1, aic=1, bic=1),
                observed_sha256="hash_A",
                fingerprint=shared_fp,
            ),
            _slot_stub(
                file_name="rep_B",
                model_name="m1",
                fit_type="baseline",
                metrics=self._scalar_metrics(chi2_red_raw=2, r2=0.5, aic=5, bic=7),
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
            metrics=self._scalar_metrics(chi2_red_raw=1, r2=1, aic=1, bic=1),
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
            "chi2_raw": np.array([1.0]),
            "chi2_red_raw": np.array([0.1]),
            "chi2": np.array([float("nan")]),
            "chi2_red": np.array([float("nan")]),
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
class TestFitResultsCompareModelsSigmaColumns:
    """Stable column semantics around the file's persistent σ.

    Covers:

    - Default column set switches dynamically: 4 cols without σ, 6 cols
      with (``chi2_red_raw`` is always present; ``sigma_eff`` + ``chi2_red``
      appear only when at least one matched slot carries a finite σ).
    - Explicit request for ``chi2`` / ``chi2_red`` with no σ raises a clear
      ``KeyError`` pointing at ``file.set_sigma(...)`` / the raw column.
    - Sum-mode aggregation of ``chi2_red_raw`` and ``chi2_red`` uses
      ``Σnumerator / ΣDoF`` (not ``np.nansum`` of per-slice values) for the
      "≈ 1 for a good fit" reading.
    - The 4 sigma fields on the slot dataclass round-trip through both
      scalar and long output modes.
    """

    #
    @staticmethod
    def _scalar_metrics(*, chi2_red_raw, r2=0.9, aic=10.0, bic=12.0, sigma_eff=None):
        """Build a 7-key metrics dict.

        ``chi2_raw = chi2_red_raw`` for stub purposes (DoF=1 by construction);
        calibrated fields are computed from ``sigma_eff`` when given.
        """

        chi2_raw = float(chi2_red_raw)
        if sigma_eff is None or not np.isfinite(sigma_eff):
            chi2 = float("nan")
            chi2_red = float("nan")
        else:
            chi2 = chi2_raw / sigma_eff**2
            chi2_red = float(chi2_red_raw) / sigma_eff**2
        return {
            "chi2_raw": chi2_raw,
            "chi2_red_raw": float(chi2_red_raw),
            "chi2": chi2,
            "chi2_red": chi2_red,
            "r2": float(r2),
            "aic": float(aic),
            "bic": float(bic),
        }

    #
    def test_default_columns_without_sigma(self):
        """No σ on any slot → calibrated columns are absent from the default."""

        slot = _slot_stub(
            file_name="A",
            model_name="m",
            fit_type="2d",
            selection={"e_lim": None, "t_lim": None},
            metrics=self._scalar_metrics(chi2_red_raw=0.05),
        )
        df = FitResults(slots=[slot]).compare_models()
        assert list(df.columns) == [
            "file",
            "model",
            "fit_type",
            "selection_json",
            "chi2_red_raw",
            "r2",
            "aic",
            "bic",
        ]
        assert "chi2_red" not in df.columns
        assert "sigma_eff" not in df.columns
        assert "chi2" not in df.columns

    #
    def test_default_columns_with_sigma(self):
        """σ on the slot → default set adds sigma_eff + chi2_red."""

        slot = _slot_stub(
            file_name="A",
            model_name="m",
            fit_type="2d",
            selection={"e_lim": None, "t_lim": None},
            sigma_data=0.2,
            metrics=self._scalar_metrics(chi2_red_raw=0.04, sigma_eff=0.2),
        )
        df = FitResults(slots=[slot]).compare_models()
        assert list(df.columns) == [
            "file",
            "model",
            "fit_type",
            "selection_json",
            "chi2_red_raw",
            "sigma_eff",
            "chi2_red",
            "r2",
            "aic",
            "bic",
        ]
        assert df["sigma_eff"].iloc[0] == pytest.approx(0.2)
        assert df["chi2_red"].iloc[0] == pytest.approx(0.04 / 0.2**2)
        assert df["chi2_red_raw"].iloc[0] == pytest.approx(0.04)

    #
    def test_baseline_sigma_eff_uses_n_avg_correction(self):
        """Slot stub mirrors the live ``_compute_sigma_eff`` correction."""

        slot = _slot_stub(
            file_name="A",
            model_name="m",
            fit_type="baseline",
            selection={"base_t_ind": [0, 5], "e_lim": None},
            sigma_data=0.2,
        )
        # _slot_stub computed sigma_eff = 0.2 / sqrt(5) at construction.
        expected = 0.2 / np.sqrt(5)
        assert slot.sigma_eff == pytest.approx(expected)
        df = FitResults(slots=[slot]).compare_models()
        assert df["sigma_eff"].iloc[0] == pytest.approx(expected)

    #
    def test_explicit_calibrated_request_without_sigma_raises(self):
        """``metrics=['chi2_red']`` with no σ → KeyError pointing at set_sigma."""

        slot = _slot_stub(
            file_name="A",
            model_name="m",
            fit_type="2d",
            selection={"e_lim": None, "t_lim": None},
            metrics=self._scalar_metrics(chi2_red_raw=0.05),
        )
        with pytest.raises(KeyError, match="file.set_sigma"):
            FitResults(slots=[slot]).compare_models(metrics=["chi2_red"])
        with pytest.raises(KeyError, match="file.set_sigma"):
            FitResults(slots=[slot]).compare_models(metrics=["chi2"])

    #
    def test_explicit_raw_request_works_without_sigma(self):
        """``metrics=['chi2_red_raw']`` always works — raw is always populated."""

        slot = _slot_stub(
            file_name="A",
            model_name="m",
            fit_type="2d",
            selection={"e_lim": None, "t_lim": None},
            metrics=self._scalar_metrics(chi2_red_raw=0.05),
        )
        df = FitResults(slots=[slot]).compare_models(metrics=["chi2_red_raw"])
        assert df["chi2_red_raw"].iloc[0] == pytest.approx(0.05)

    #
    def test_sigma_eff_broadcast_in_long_mode(self):
        """SbS in ``long`` mode: every slice row gets the slot's σ_eff."""

        per_slice = {
            "chi2_raw": np.array([1.0, 2.0, 3.0]),
            "chi2_red_raw": np.array([0.04, 0.05, 0.06]),
            "chi2": np.array([1.0 / 0.04, 2.0 / 0.04, 3.0 / 0.04]),
            "chi2_red": np.array([1.0, 1.25, 1.5]),
            "r2": np.array([0.9, 0.8, 0.85]),
            "aic": np.array([10.0, 20.0, 30.0]),
            "bic": np.array([12.0, 22.0, 32.0]),
        }
        slot = _slot_stub(
            file_name="A",
            model_name="m_sbs",
            fit_type="sbs",
            selection={"e_lim": None, "t_lim": None},
            metrics=per_slice,
            sigma_data=0.2,
        )
        df = FitResults(slots=[slot]).compare_models(sbs_aggregation="long")
        assert len(df) == 3
        # Per-slot scalar broadcast to every slice row.
        assert df["sigma_eff"].tolist() == [pytest.approx(0.2)] * 3

    #
    def test_sbs_sum_chi2_red_raw_aggregates_via_dof(self):
        """sum-mode ``chi2_red_raw`` = Σ chi2_raw / Σ DoF (not nansum)."""

        # DoF_i = chi2_raw_i / chi2_red_raw_i = [10, 15] → ΣDoF = 25, Σchi2_raw = 40.
        per_slice = {
            "chi2_raw": np.array([10.0, 30.0]),
            "chi2_red_raw": np.array([1.0, 2.0]),
            "chi2": np.array([float("nan"), float("nan")]),
            "chi2_red": np.array([float("nan"), float("nan")]),
            "r2": np.array([0.9, 0.8]),
            "aic": np.array([10.0, 20.0]),
            "bic": np.array([12.0, 22.0]),
        }
        slot = _slot_stub(
            file_name="A",
            model_name="m_sbs",
            fit_type="sbs",
            selection={"e_lim": None, "t_lim": None},
            metrics=per_slice,
        )
        df = FitResults(slots=[slot]).compare_models(
            sbs_aggregation="sum",
            metrics=["chi2_raw", "chi2_red_raw", "aic", "bic"],
        )
        assert df["chi2_red_raw"].iloc[0] == pytest.approx(40.0 / 25.0)
        # chi2_raw / aic / bic still nansum'd.
        assert df["chi2_raw"].iloc[0] == pytest.approx(40.0)
        assert df["aic"].iloc[0] == pytest.approx(30.0)
        assert df["bic"].iloc[0] == pytest.approx(34.0)

    #
    def test_sbs_sum_chi2_red_uses_calibrated_numerator(self):
        """sum-mode ``chi2_red`` = Σ chi2 / Σ DoF; equals chi2_red_raw / σ²."""

        sigma = 0.5
        per_slice_raw = 0.04
        n_slices = 4
        chi2_raw = np.full(n_slices, per_slice_raw * 100.0)  # DoF = 100 each
        chi2_red_raw = np.full(n_slices, per_slice_raw)
        chi2 = chi2_raw / sigma**2
        chi2_red = chi2_red_raw / sigma**2
        per_slice = {
            "chi2_raw": chi2_raw,
            "chi2_red_raw": chi2_red_raw,
            "chi2": chi2,
            "chi2_red": chi2_red,
            "r2": np.full(n_slices, 0.99),
            "aic": np.full(n_slices, -10.0),
            "bic": np.full(n_slices, -8.0),
        }
        slot = _slot_stub(
            file_name="A",
            model_name="m",
            fit_type="sbs",
            selection={"e_lim": None, "t_lim": None},
            metrics=per_slice,
            sigma_data=sigma,
        )
        df = FitResults(slots=[slot]).compare_models(sbs_aggregation="sum")
        # aggregate raw = per_slice_raw (constant per slice)
        assert df["chi2_red_raw"].iloc[0] == pytest.approx(per_slice_raw)
        # aggregate calibrated = per_slice_raw / σ²
        assert df["chi2_red"].iloc[0] == pytest.approx(per_slice_raw / sigma**2)


#
class TestPlotFitAPI:
    """FitResults.plot_fit / plot_param_evolution and the File.* sugar."""

    #
    def test_plot_fit_1d_uses_real_axes_and_config(self):
        import matplotlib.pyplot as plt

        _, file = _setup_baseline_fit()
        file.plot_fit(fit_type="baseline")  # show under Agg keeps the fig live
        fig = plt.gcf()
        try:
            line_x = fig.axes[0].lines[0].get_xdata()
            np.testing.assert_array_equal(line_x, np.asarray(file.energy))
            assert fig.axes[1].get_xlabel() == file.plot_config.x_label
        finally:
            plt.close("all")

    #
    def test_plot_fit_2d_passes_real_axes(self, monkeypatch):
        from unittest.mock import MagicMock

        from trspecfit import fitlib

        mock_2d = MagicMock()
        monkeypatch.setattr(fitlib, "plt_fit_res_2d", mock_2d)

        project, file = _setup_baseline_fit()
        file.add_time_dependence(
            target_model="single_glp",
            target_parameter="GLP_01_A",
            dynamics_yaml="models/file_time.yaml",
            dynamics_model=["MonoExpPos"],
        )
        file.fit_2d("single_glp", stages=1, try_ci=0)
        mock_2d.reset_mock()

        project.results.plot_fit(file=file, fit_type="2d", show_plot=False)
        assert mock_2d.call_count == 1
        kwargs = mock_2d.call_args.kwargs
        np.testing.assert_array_equal(kwargs["x"], np.asarray(file.energy))
        np.testing.assert_array_equal(kwargs["y"], np.asarray(file.time))
        assert kwargs["save_img"] == -2  # show_plot=False

    #
    def test_plot_fit_from_loaded_archive_has_axes(self, tmp_path):
        import matplotlib.pyplot as plt

        project, file = _setup_baseline_fit()
        archive_path = tmp_path / "plot.fit.h5"
        project.save_fits(archive_path, show_output=0)

        loaded = FitResults.load(archive_path)
        loaded.plot_fit(file=file.name, fit_type="baseline")
        fig = plt.gcf()
        try:
            line_x = fig.axes[0].lines[0].get_xdata()
            np.testing.assert_array_equal(line_x, np.asarray(file.energy))
        finally:
            plt.close("all")

    #
    @staticmethod
    def _fake_sbs_results(*, vary=(True, False, True)):
        """FitResults around a synthetic SbS slot (wide params + metadata)."""

        import dataclasses

        project, _ = _setup_baseline_fit()
        wide = pd.DataFrame(
            {
                "A": [1.0, 2.0, 3.0],
                "B": [4.0, 5.0, 6.0],
                "C": [7.0, 8.0, 9.0],
            }
        )
        meta = pd.DataFrame(
            {
                "name": ["A", "B", "C"],
                "vary": list(vary),
                "min": [0.0] * 3,
                "max": [10.0] * 3,
                "expr": [None] * 3,
            }
        )
        fake = dataclasses.replace(
            project._fit_history[0], fit_type="sbs", params=wide, params_meta=meta
        )
        return FitResults(slots=[fake])

    #
    def test_plot_param_evolution_defaults_to_varied(self, monkeypatch):
        from unittest.mock import MagicMock

        from trspecfit import fitlib

        mock_pars = MagicMock()
        monkeypatch.setattr(fitlib, "plt_fit_res_pars", mock_pars)

        results = self._fake_sbs_results(vary=(True, False, True))
        results.plot_param_evolution(show_plot=False)
        assert mock_pars.call_count == 1
        kwargs = mock_pars.call_args.kwargs
        assert list(kwargs["df"].columns) == ["A", "C"]  # varied only
        # No axes provider on this FitResults -> index fallback.
        np.testing.assert_array_equal(kwargs["x"], np.arange(3))

    #
    def test_plot_param_evolution_explicit_and_missing_params(self, monkeypatch):
        from unittest.mock import MagicMock

        from trspecfit import fitlib

        mock_pars = MagicMock()
        monkeypatch.setattr(fitlib, "plt_fit_res_pars", mock_pars)

        results = self._fake_sbs_results()
        results.plot_param_evolution(params=["B"], show_plot=False)
        assert list(mock_pars.call_args.kwargs["df"].columns) == ["B"]
        with pytest.raises(KeyError, match="not in this SbS fit"):
            results.plot_param_evolution(params=["nope"], show_plot=False)

    #
    def test_plot_param_evolution_all_fixed_plots_nothing(self, monkeypatch):
        from unittest.mock import MagicMock

        from trspecfit import fitlib

        mock_pars = MagicMock()
        monkeypatch.setattr(fitlib, "plt_fit_res_pars", mock_pars)

        results = self._fake_sbs_results(vary=(False, False, False))
        results.plot_param_evolution(show_plot=False)
        assert mock_pars.call_count == 0

    #
    def test_plot_residuals_uses_energy_axis_with_provider(self):
        project, file = _setup_baseline_fit()
        fig = project.results.plot_residuals(file=file.name, show_plot=False)
        assert fig.axes[1].get_xlabel() == "energy"


#
class TestPlotMcmc:
    """FitResults.plot_mcmc renders diagnostics from the slot's mcmc payload
    (synthetic slots here; the live-fit + loaded-archive path is covered in
    TestMcmcPayload)."""

    #
    @staticmethod
    def _mcmc_results(*, with_acceptance=True):
        import dataclasses

        n = 40
        flatchain = pd.DataFrame(
            {
                "GLP_01_A": np.linspace(0.9, 1.1, n),
                "__lnsigma": np.linspace(-2.1, -1.9, n),
            }
        )
        ci = pd.DataFrame(
            {
                "par[v]/sigma[>]": ["GLP_01_A", "__lnsigma"],
                "-1.0": [0.95, -2.05],
                "best fit": [1.0, -2.0],
                "+1.0": [1.05, -1.95],
            }
        )
        mcmc = {
            "flatchain": flatchain,
            "ci": ci,
            "lnsigma": -2.0,
            "acceptance_fraction": (np.full(8, 0.4) if with_acceptance else None),
        }
        return FitResults(slots=[dataclasses.replace(_slot_stub(), mcmc=mcmc)])

    #
    def test_renders_acceptance_and_corner(self):
        import matplotlib.pyplot as plt

        results = self._mcmc_results()
        plt.close("all")
        results.plot_mcmc(file="f1", fit_type="baseline")  # show under Agg
        try:
            assert len(plt.get_fignums()) == 2
        finally:
            plt.close("all")

    #
    def test_skips_acceptance_when_absent(self):
        import matplotlib.pyplot as plt

        # schema-2 archives did not store acceptance_fraction.
        results = self._mcmc_results(with_acceptance=False)
        plt.close("all")
        results.plot_mcmc(file="f1", fit_type="baseline")
        try:
            assert len(plt.get_fignums()) == 1  # corner only
        finally:
            plt.close("all")

    #
    def test_show_plot_false_leaves_no_figures(self):
        import matplotlib.pyplot as plt

        results = self._mcmc_results()
        n_figs = len(plt.get_fignums())
        results.plot_mcmc(file="f1", fit_type="baseline", show_plot=False)
        assert len(plt.get_fignums()) == n_figs

    #
    def test_raises_without_mcmc_payload(self):
        results = FitResults(slots=[_slot_stub()])
        with pytest.raises(ValueError, match="No MCMC results"):
            results.plot_mcmc(file="f1", fit_type="baseline", show_plot=False)


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
            noise_type=slot.noise_type,
            sigma_source=slot.sigma_source,
            sigma_type=slot.sigma_type,
            sigma_data=slot.sigma_data,
            sigma_eff=slot.sigma_eff,
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


#
# --- multi-fit history accumulation + snapshot-collapse on save -------------
#


#
class TestHistoryAccumulationAndSnapshot:
    """Multi-fit history accumulation, in-session multi-version visibility,
    and snapshot-collapse-on-save.

    Scenario: fit modelA-baseline, fit modelB-baseline, refit modelA-baseline.
    History has *all three* slots; ``Project.results`` exposes them.
    ``save_fits`` (snapshot mode) collapses to two — one per ``history_key``,
    latest wins.
    """

    #
    @staticmethod
    def _two_model_fit_file(project):
        """Build a fit file with two distinct energy models registered.

        Both ``single_glp`` and ``two_glp_expr_amplitude`` fit cleanly on
        the [82, 92] axis; quality-of-fit is irrelevant here — what matters
        is that both ``model_name`` strings produce valid baseline slots
        with distinct ``history_key`` values.
        """

        truth_project = make_project(name="truth_two_model")
        truth = File(
            parent_project=truth_project,
            name="truth",
            energy=np.linspace(82, 92, 30),
            time=np.linspace(-2, 10, 24),
        )
        truth.dim = 2
        truth.load_model(model_yaml="models/file_energy.yaml", model_info="single_glp")
        data = simulate_noisy(truth.model_active, noise_level=0.01)

        file = File(
            parent_project=project,
            name="fit_two_model",
            data=data,
            energy=truth.energy.copy(),
            time=truth.time.copy(),
        )
        file.load_model(model_yaml="models/file_energy.yaml", model_info="single_glp")
        file.load_model(
            model_yaml="models/file_energy.yaml",
            model_info="two_glp_expr_amplitude",
        )
        file.define_baseline(
            time_start=0, time_stop=3, time_type="ind", show_plot=False
        )
        return file

    #
    def test_history_holds_all_completed_fits(self):
        """fit modelA → fit modelB → refit modelA accumulates 3 slots."""

        project = make_project(name="acc")
        file = self._two_model_fit_file(project)

        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
        file.fit_baseline(model_name="two_glp_expr_amplitude", stages=1, try_ci=0)
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)

        assert len(project._fit_history) == 3
        names_in_order = [s.model_name for s in project._fit_history]
        assert names_in_order == [
            "single_glp",
            "two_glp_expr_amplitude",
            "single_glp",
        ]

    #
    def test_results_exposes_all_history_entries(self):
        """``Project.results`` mirrors ``_fit_history`` slot-for-slot."""

        project = make_project(name="acc_results")
        file = self._two_model_fit_file(project)

        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
        file.fit_baseline(model_name="two_glp_expr_amplitude", stages=1, try_ci=0)
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)

        results = project.results
        assert len(results) == 3
        # find() exposes both refits when narrowed to modelA.
        single_glp_slots = results.find(
            file=file.name, model="single_glp", fit_type="baseline"
        )
        assert len(single_glp_slots) == 2
        # The two refits share a history_key (same fit_view, same model).
        assert single_glp_slots[0].history_key == single_glp_slots[1].history_key
        # The cross-model slot has a distinct history_key.
        cross = results.find(
            file=file.name,
            model="two_glp_expr_amplitude",
            fit_type="baseline",
        )
        assert len(cross) == 1
        assert cross[0].history_key != single_glp_slots[0].history_key

    #
    def test_save_fits_collapses_refits_to_latest_per_key(self, tmp_path):
        """Snapshot save keeps one slot per ``history_key`` (latest wins)."""

        project = make_project(name="acc_save")
        file = self._two_model_fit_file(project)

        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
        file.fit_baseline(model_name="two_glp_expr_amplitude", stages=1, try_ci=0)
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
        # _fit_history has 3; the two single_glp slots share a history_key.
        assert len(project._fit_history) == 3

        # Stamp the duplicate-key slots with deterministic sentinels so the
        # latest-wins assertion does not depend on second-resolution wall
        # clocks. ``_now_iso()`` is per-second, so two fits inside the same
        # second would silently weaken the assertion (both "earlier" and
        # "later" timestamps would compare equal). ``dataclasses.replace``
        # works on the frozen SavedFitSlot.
        from dataclasses import replace

        project._fit_history[0] = replace(
            project._fit_history[0], timestamp="2026-01-01T00:00:00+00:00"
        )
        project._fit_history[2] = replace(
            project._fit_history[2], timestamp="2026-01-01T00:00:01+00:00"
        )

        archive_path = tmp_path / "snapshot.fit.h5"
        project.save_fits(archive_path, show_output=0)
        loaded = FitResults.load(archive_path)
        # Snapshot collapses the duplicate-key pair → 2 distinct slots.
        assert len(loaded) == 2
        keys_in_archive = {s.history_key for s in loaded}
        assert keys_in_archive == {
            project._fit_history[0].history_key,
            project._fit_history[1].history_key,
        }

        # Latest-wins: collapse must keep the third fit (slot[2]), not the
        # first (slot[0]) — proved by the sentinel timestamp regardless of
        # clock resolution.
        loaded_single = next(s for s in loaded if s.model_name == "single_glp")
        assert loaded_single.timestamp == "2026-01-01T00:00:01+00:00"


#
# --- selection-identity: refits with different views → distinct slots --------
#


#
class TestSelectionIdentity:
    """Refits with different fit-view selections must produce distinct
    ``history_key`` values and survive snapshot save as separate slots.

    Covers each fit_type's selection-identity field:

    - baseline: ``base_t_ind`` (time window averaged for ``data_base``)
    - sbs:      ``e_lim`` / ``t_lim``
    - 2d:       ``e_lim`` / ``t_lim``
    - spectrum: ``time_point`` is already covered in ``TestSpectrumSlot``
    """

    #
    @staticmethod
    def _basic_2d_fit_file(project):
        """1D-fittable 2D file with single_glp and a wide enough baseline."""

        truth_project = make_project(name="truth_sel")
        truth = _make_truth_file(truth_project)
        data = simulate_noisy(truth.model_active, noise_level=0.01)

        file = _make_fit_file(project, data, truth.energy, truth.time)
        return file

    #
    def test_baseline_refit_with_different_base_t_ind_distinct(self, tmp_path):
        """Different ``base_t_ind`` → distinct ``history_key``; snapshot keeps both."""

        project = make_project(name="sel_base")
        file = self._basic_2d_fit_file(project)

        file.define_baseline(
            time_start=0, time_stop=3, time_type="ind", show_plot=False
        )
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
        first_key = project._fit_history[0].history_key

        file.define_baseline(
            time_start=0, time_stop=2, time_type="ind", show_plot=False
        )
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)

        keys = [s.history_key for s in project._fit_history]
        assert keys[0] != keys[1]
        assert keys[0] == first_key
        # selection captures the inclusive→exclusive index slice.
        assert project._fit_history[0].selection["base_t_ind"] == [0, 4]
        assert project._fit_history[1].selection["base_t_ind"] == [0, 3]

        # Snapshot save preserves both — no collapse since keys differ.
        archive_path = tmp_path / "base_t_ind.fit.h5"
        project.save_fits(archive_path, show_output=0)
        loaded = FitResults.load(archive_path)
        assert len(loaded) == 2
        assert {s.history_key for s in loaded} == set(keys)

    #
    @pytest.mark.slow
    def test_sbs_refit_with_different_e_lim_distinct(self, tmp_path):
        """SbS refit with a different ``e_lim`` → distinct slots."""

        project = make_project(name="sel_sbs")
        project.spec_fun_str = "fit_model_mcp"
        file = self._basic_2d_fit_file(project)

        file.fit_slice_by_slice(
            "single_glp",
            n_workers=1,
            seed_source="model",
            seed_adapt=None,
            try_ci=0,
        )
        # Refit with a tighter e_lim. Set both index and absolute parallels.
        file.e_lim = [5, 25]
        file.e_lim_abs = [float(file.energy[5]), float(file.energy[24])]
        file.fit_slice_by_slice(
            "single_glp",
            n_workers=1,
            seed_source="model",
            seed_adapt=None,
            try_ci=0,
        )

        keys = [s.history_key for s in project._fit_history]
        assert len(keys) == 2
        assert keys[0] != keys[1]
        # File constructor pre-fills e_lim with the full range via
        # set_fit_limits, so the first fit's selection is not None.
        assert project._fit_history[0].selection["e_lim"] == [0, len(file.energy)]
        assert project._fit_history[1].selection["e_lim"] == [5, 25]

        archive_path = tmp_path / "sbs_e_lim.fit.h5"
        project.save_fits(archive_path, show_output=0)
        loaded = FitResults.load(archive_path)
        assert len(loaded) == 2

    #
    @pytest.mark.slow
    def test_2d_refit_with_different_t_lim_distinct(self, tmp_path):
        """2D refit with a different ``t_lim`` → distinct slots."""

        project = make_project(name="sel_2d")
        file = self._basic_2d_fit_file(project)
        file.define_baseline(
            time_start=0, time_stop=3, time_type="ind", show_plot=False
        )
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
        # Add dynamics so fit_2d is valid.
        file.add_time_dependence(
            target_model="single_glp",
            target_parameter="GLP_01_A",
            dynamics_yaml="models/file_time.yaml",
            dynamics_model=["MonoExpPos"],
        )
        file.fit_2d("single_glp", stages=1, try_ci=0)
        # Refit with a tighter t_lim covering the post-trigger half.
        file.t_lim = [4, 24]
        file.t_lim_abs = [float(file.time[4]), float(file.time[23])]
        file.fit_2d("single_glp", stages=1, try_ci=0)

        twod_slots = [s for s in project._fit_history if s.fit_type == "2d"]
        assert len(twod_slots) == 2
        assert twod_slots[0].history_key != twod_slots[1].history_key
        # File constructor pre-fills t_lim with the full range; the second
        # fit narrows it. The two distinct t_lim values must produce two
        # distinct history_keys.
        assert twod_slots[0].selection["t_lim"] == [0, len(file.time)]
        assert twod_slots[1].selection["t_lim"] == [4, 24]

        archive_path = tmp_path / "2d_t_lim.fit.h5"
        project.save_fits(archive_path, fit_type="2d", show_output=0)
        loaded = FitResults.load(archive_path)
        assert len(loaded) == 2
