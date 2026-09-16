"""MC settings: derived defaults, resolution into a copy, seed, provenance.

The ``MC`` a caller passes is a request. ``fitlib.fit_wrapper`` resolves the
knobs left at None from the optimizer result into a copy, which is returned
as ``FitOutput.mc_settings`` and recorded in the slot's ``fit_settings["mc"]``.
"""

import numpy as np
import pandas as pd
import pytest
from _utils import make_project, simulate_noisy

from trspecfit import File
from trspecfit.utils.lmfit import MC

# short chains: these tests pin the plumbing, not posterior quality
_CHAIN = {"steps": 20, "burn": 5, "thin": 1, "workers": 1}


#
def _make_truth_file(project):
    file = File(parent_project=project, name="truth")
    file.energy = np.linspace(83, 87, 30)
    file.time = np.linspace(-2, 10, 24)
    file.dim = 2
    file.load_model(model_yaml="models/file_energy.yaml", model_info="single_glp")
    file.add_time_dependence(
        target_model="single_glp",
        target_parameter="GLP_01_A",
        dynamics_yaml="models/file_time.yaml",
        dynamics_model=["MonoExpPos"],
    )
    return file


#
def _baseline_ready_file():
    """(project, file) with noisy 2D data and a defined baseline, not yet fit."""

    truth = _make_truth_file(make_project(name="truth"))
    data = simulate_noisy(truth.model_active, noise_level=0.01)
    project = make_project(name="fit")
    file = File(
        parent_project=project,
        name="fit",
        data=data,
        energy=truth.energy.copy(),
        time=truth.time.copy(),
    )
    file.load_model(model_yaml="models/file_energy.yaml", model_info="single_glp")
    file.define_baseline(time_start=0, time_stop=3, time_type="ind", show_plot=False)
    return project, file


#
def _fit_output(file):
    assert file.model_base is not None  # type guard
    assert file.model_base.result is not None  # type guard
    return file.model_base.result


#
#
class TestMCConstruction:
    #
    def test_derivable_knobs_default_to_none(self):
        mc = MC()
        assert mc.use_mc == 0
        assert mc.nwalkers is None
        assert (mc.sigma_ini, mc.sigma_min, mc.sigma_max) == (None, None, None)
        assert mc.seed is None

    #
    def test_use_mc_is_stored_under_its_own_name(self):
        mc = MC(use_mc=2)
        assert mc.use_mc == 2
        assert not hasattr(mc, "use_emcee")

    #
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"use_mc": 3}, "use_mc must be 0"),
            ({"nwalkers": 1}, "nwalkers must be >= 2"),
            ({"seed": -1}, "seed must be a non-negative int"),
            ({"seed": True}, "seed must be a non-negative int"),
            ({"sigma_min": 2.0, "sigma_max": 1.0}, "sigma_min must be <"),
            ({"sigma_ini": 10.0, "sigma_max": 2.0}, "sigma_ini must lie within"),
            ({"sigma_ini": 0.0}, "sigma_ini must be a positive"),
        ],
    )
    def test_rejects_inconsistent_settings(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            MC(**kwargs)


#
#
class TestMCResolve:
    #
    def test_derives_sigma_bounds_and_walker_floor(self):
        ran = MC(use_mc=1).resolve(sigma_fit=0.3, n_dim=4)
        assert ran.sigma_ini == 0.3
        assert ran.sigma_min == pytest.approx(0.003)
        assert ran.sigma_max == pytest.approx(30.0)
        assert ran.nwalkers == 20  # floor: 2 * 4 = 8 is too few for mixing

    #
    def test_walkers_follow_the_dimension_count(self):
        assert MC(use_mc=1).resolve(sigma_fit=0.3, n_dim=15).nwalkers == 30

    #
    def test_explicit_values_are_kept_and_the_request_is_untouched(self):
        mc = MC(use_mc=1, nwalkers=50, sigma_ini=0.5, sigma_min=0.01, sigma_max=5.0)
        ran = mc.resolve(sigma_fit=0.3, n_dim=4)
        assert ran is not mc
        assert (ran.nwalkers, ran.sigma_ini) == (50, 0.5)
        assert (ran.sigma_min, ran.sigma_max) == (0.01, 5.0)
        assert (mc.nwalkers, mc.sigma_ini) == (50, 0.5)

    #
    def test_derived_start_is_checked_against_explicit_bounds(self):
        mc = MC(use_mc=1, sigma_min=0.01, sigma_max=1.0)
        assert mc.resolve(sigma_fit=0.3, n_dim=2).sigma_ini == 0.3
        with pytest.raises(ValueError, match="derived from the fit"):
            mc.resolve(sigma_fit=5.0, n_dim=2)

    #
    def test_explicit_walkers_below_the_emcee_minimum_raise(self):
        with pytest.raises(ValueError, match=r"2 \* n_dim = 6"):
            MC(use_mc=1, nwalkers=4).resolve(sigma_fit=0.3, n_dim=3)

    #
    def test_weighted_sampling_carries_no_sigma(self):
        ran = MC(use_mc=1, is_weighted=True, sigma_ini=0.5).resolve(
            sigma_fit=0.3, n_dim=4
        )
        assert (ran.sigma_ini, ran.sigma_min, ran.sigma_max) == (None, None, None)
        assert ran.nwalkers == 20

    #
    def test_zero_residual_cannot_seed_the_noise_scale(self):
        with pytest.raises(ValueError, match="noiseless"):
            MC(use_mc=1).resolve(sigma_fit=0.0, n_dim=2)


#
#
@pytest.mark.slow
class TestResolutionInFit:
    #
    def test_derived_knobs_come_from_the_optimizer_result(self):
        project, file = _baseline_ready_file()
        mc = MC(use_mc=1, **_CHAIN)
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0, mc_settings=mc)

        out = _fit_output(file)
        ran = out.mc_settings
        assert ran is not None  # type guard
        sigma_fit = np.sqrt(out.par_fin.chisqr / out.par_fin.ndata)
        assert ran.sigma_ini == pytest.approx(sigma_fit)
        assert ran.sigma_min == pytest.approx(sigma_fit / 100)
        assert ran.sigma_max == pytest.approx(sigma_fit * 100)
        assert ran.nwalkers == max(20, 2 * (out.par_fin.nvarys + 1))
        # ...and they are what emcee sampled
        assert out.emcee_fin is not None  # type guard
        lnsigma = out.emcee_fin.params["__lnsigma"]
        assert lnsigma.min == pytest.approx(np.log(sigma_fit / 100))
        assert lnsigma.max == pytest.approx(np.log(sigma_fit * 100))
        assert out.emcee_fin.acceptance_fraction.shape == (ran.nwalkers,)
        # the request is untouched
        assert mc.nwalkers is None
        assert mc.sigma_ini is None

    #
    def test_provenance_records_what_ran(self):
        project, file = _baseline_ready_file()
        mc = MC(use_mc=1, seed=5, **_CHAIN)
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0, mc_settings=mc)

        ran = _fit_output(file).mc_settings
        assert ran is not None  # type guard
        slot = project._fit_history[0]
        assert slot.fit_settings is not None  # type guard
        assert slot.fit_settings["mc"] == {
            "use_mc": 1,
            "steps": 20,
            "nwalkers": ran.nwalkers,
            "burn": 5,
            "thin": 1,
            "ntemps": 1,
            "is_weighted": False,
            "sigma_ini": ran.sigma_ini,
            "sigma_min": ran.sigma_min,
            "sigma_max": ran.sigma_max,
            "seed": 5,
        }
        # the sampler seed is not the optimizer seed
        assert "seed" not in {k for k in slot.fit_settings if k != "mc"}

    #
    def test_weighted_run_omits_sigma_from_provenance(self):
        project, file = _baseline_ready_file()
        mc = MC(use_mc=1, is_weighted=True, nwalkers=32, **_CHAIN)
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0, mc_settings=mc)

        ran = _fit_output(file).mc_settings
        assert ran is not None  # type guard
        assert ran.sigma_ini is None
        slot = project._fit_history[0]
        assert slot.fit_settings is not None  # type guard
        block = slot.fit_settings["mc"]
        assert block["nwalkers"] == 32
        assert block["is_weighted"] is True
        assert not {"sigma_ini", "sigma_min", "sigma_max", "seed"} & set(block)

    #
    def test_seed_makes_the_chain_reproducible(self):
        def flatchain(seed):
            _, file = _baseline_ready_file()
            mc = MC(use_mc=1, seed=seed, **_CHAIN)
            file.fit_baseline(
                model_name="single_glp", stages=1, try_ci=0, mc_settings=mc
            )
            out = _fit_output(file)
            assert out.emcee_fin is not None  # type guard
            return out.emcee_fin.flatchain

        pd.testing.assert_frame_equal(flatchain(3), flatchain(3))
        assert not flatchain(3).equals(flatchain(4))

    #
    def test_ci_fallback_runs_mcmc_without_mutating_the_request(self, monkeypatch):
        """``use_mc=2`` falls back to MCMC when conf_interval cannot run; the
        decision is local to the call, so the caller's MC still says 2 and
        the next slice/fit decides afresh."""

        from trspecfit import fitlib

        monkeypatch.setattr(fitlib, "_result_errorbars", lambda result: False)
        project, file = _baseline_ready_file()
        mc = MC(use_mc=2, **_CHAIN)
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=1, mc_settings=mc)

        out = _fit_output(file)
        assert out.conf_ci.empty
        assert out.emcee_fin is not None
        assert out.mc_settings is not None  # type guard
        assert out.mc_settings.use_mc == 2
        assert mc.use_mc == 2
        slot = project._fit_history[0]
        assert slot.fit_settings is not None  # type guard
        assert slot.fit_settings["mc"]["use_mc"] == 2

    #
    def test_ci_success_skips_mcmc_and_records_no_block(self):
        project, file = _baseline_ready_file()
        mc = MC(use_mc=2, **_CHAIN)
        file.fit_baseline(model_name="single_glp", stages=2, try_ci=1, mc_settings=mc)

        out = _fit_output(file)
        assert not out.conf_ci.empty
        assert out.emcee_fin is None
        assert out.mc_settings is None
        slot = project._fit_history[0]
        assert slot.fit_settings is not None  # type guard
        assert "mc" not in slot.fit_settings

    #
    def test_explicit_walkers_below_minimum_raise_before_sampling(self):
        _, file = _baseline_ready_file()
        mc = MC(use_mc=1, nwalkers=2, **_CHAIN)
        with pytest.raises(ValueError, match="below emcee's minimum"):
            file.fit_baseline(
                model_name="single_glp", stages=1, try_ci=0, mc_settings=mc
            )

    #
    def test_sbs_slot_records_slice_zero_settings(self):
        project, file = _baseline_ready_file()
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
        mc = MC(use_mc=1, **_CHAIN)
        file.fit_slice_by_slice(
            model_name="single_glp", stages=1, try_ci=0, mc_settings=mc
        )

        slice0 = file.results_sbs[0].mc_settings
        assert slice0 is not None  # type guard
        slot = project._fit_history[-1]
        assert slot.fit_type == "sbs"
        assert slot.fit_settings is not None  # type guard
        assert slot.fit_settings["mc"]["sigma_ini"] == slice0.sigma_ini
        assert slot.fit_settings["mc"]["nwalkers"] == slice0.nwalkers
        # each slice resolves its own start from its own residual
        starts = {r.mc_settings.sigma_ini for r in file.results_sbs if r.mc_settings}
        assert len(starts) > 1
        assert mc.sigma_ini is None
