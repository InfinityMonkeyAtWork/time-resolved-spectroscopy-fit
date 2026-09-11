"""Tests for trspecfit.sensitivity — Fisher information and photon budgets."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from _utils import make_project

from trspecfit import File, sensitivity


#
def _gauss_model(*, energy=None):
    """Single Gauss on a wide window: the case with a closed-form bound."""

    project = make_project(name="sens")
    energy = np.arange(80.0, 90.0, 0.02) if energy is None else energy
    file = File(parent_project=project, name=f"g{energy.size}", energy=energy)
    file.load_model(
        model_yaml="models/sensitivity_energy.yaml",
        model_info="gauss_free",
    )
    model = file.model_active
    model.energy = energy.copy()
    return model


#
def _model(model_info, *, name, energy=None):
    """Load one of the sensitivity fixtures with its axes set."""

    project = make_project(name=f"sens_{name}")
    energy = np.arange(80.0, 92.0, 0.02) if energy is None else energy
    file = File(parent_project=project, name=name, energy=energy)
    file.load_model(
        model_yaml="models/sensitivity_energy.yaml",
        model_info=model_info,
    )
    model = file.model_active
    model.energy = energy.copy()
    return model


#
#
class TestAnalyticAgreement:
    """The bound must reproduce the one case with a closed-form answer."""

    #
    def test_gaussian_position_bound(self):
        """For a Gauss peak, sigma(x0) = SD / sqrt(N) to numerical precision.

        Position is orthogonal to amplitude and width for a symmetric peak,
        so the marginal bound equals the closed form even with A and SD free.
        """

        model = _gauss_model()
        sd = model.lmfit_pars["Gauss_01_SD"].value

        for counts in (1e3, 1e4, 1e5):
            bound = sensitivity.crb(model, counts)["Gauss_01_x0"]
            assert bound == pytest.approx(sd / np.sqrt(counts), rel=1e-4)

    #
    def test_position_is_uncorrelated_for_symmetric_peak(self):
        """Odd and even moments do not mix: x0 pays no correlation penalty."""

        model = _gauss_model()
        frame = sensitivity.sensitivity_report(model, counts=1e4)
        row = frame.loc[frame["name"] == "Gauss_01_x0"].iloc[0]
        assert row["correlation_penalty"] == pytest.approx(1.0, rel=1e-6)


#
#
class TestScaling:
    """Bounds scale as counts**-0.5 exactly, and the inverse is consistent."""

    #
    def test_inverse_square_root_scaling(self):
        """Quadrupling the budget halves every bound."""

        model = _gauss_model()
        low = sensitivity.crb(model, 1e4)
        high = sensitivity.crb(model, 4e4)
        for name, value in low.items():
            assert value / high[name] == pytest.approx(2.0, rel=1e-9)

    #
    def test_counts_required_round_trip(self):
        """counts_required inverts crb: feed its answer back and recover it."""

        model = _gauss_model()
        target = 0.004
        counts = sensitivity.counts_required(model, "Gauss_01_x0", target)
        achieved = sensitivity.crb(model, counts)["Gauss_01_x0"]
        assert achieved == pytest.approx(target, rel=1e-6)

    #
    def test_counts_required_independent_of_reference(self):
        """The reference budget only sets the numerical scale."""

        model = _gauss_model()
        a = sensitivity.counts_required(model, "Gauss_01_x0", 0.004, counts_ref=1e3)
        b = sensitivity.counts_required(model, "Gauss_01_x0", 0.004, counts_ref=1e6)
        assert a == pytest.approx(b, rel=1e-6)


#
#
class TestSelectionSemantics:
    """par_names selects output; it must not silently shrink the matrix."""

    #
    def test_subset_matches_full_report(self):
        """A single-parameter request returns the marginal bound, not the
        optimistic one that assumes every other parameter is known."""

        model = _gauss_model()
        full = sensitivity.crb(model, 1e4)
        subset = sensitivity.crb(model, 1e4, ["Gauss_01_A"])
        assert subset["Gauss_01_A"] == pytest.approx(full["Gauss_01_A"], rel=1e-12)

    #
    def test_marginal_is_never_better_than_uncorrelated(self):
        """Marginalising over nuisance parameters can only cost precision."""

        model = _gauss_model()
        marginal = sensitivity.crb(model, 1e4, marginal=True)
        optimistic = sensitivity.crb(model, 1e4, marginal=False)
        for name, value in marginal.items():
            assert value >= optimistic[name] * (1 - 1e-9)

    #
    def test_fisher_span_restriction_is_optimistic(self):
        """Restricting the matrix span is documented as assuming the excluded
        parameters are known, so it must match marginal=False."""

        model = _gauss_model()
        info, names, _ = sensitivity.fisher_matrix(model, 1e4, ["Gauss_01_A"])
        restricted = float(1.0 / np.sqrt(info[0, 0]))
        optimistic = sensitivity.crb(model, 1e4, marginal=False)["Gauss_01_A"]
        assert names == ["Gauss_01_A"]
        assert restricted == pytest.approx(optimistic, rel=1e-9)


#
#
class TestParameterFiltering:
    """Only parameters the measurement can constrain may be bounded."""

    #
    def test_fixed_parameter_excluded(self):
        """vary=False parameters do not appear in the report."""

        model = _gauss_model()
        model.lmfit_pars["Gauss_01_SD"].vary = False
        frame = sensitivity.sensitivity_report(model, counts=1e4)
        assert "Gauss_01_SD" not in set(frame["name"])
        assert "Gauss_01_x0" in set(frame["name"])

    #
    def test_fixed_parameter_request_raises(self):
        """Asking for a fixed parameter is an error, not a silent omission."""

        model = _gauss_model()
        model.lmfit_pars["Gauss_01_SD"].vary = False
        with pytest.raises(ValueError, match="fixed or expression-defined"):
            sensitivity.crb(model, 1e4, ["Gauss_01_SD"])

    #
    def test_unknown_parameter_raises(self):
        model = _gauss_model()
        with pytest.raises(ValueError, match="Unknown parameter"):
            sensitivity.crb(model, 1e4, ["not_a_parameter"])

    #
    def test_fixing_a_nuisance_improves_the_bound(self):
        """Holding a correlated parameter known can only help."""

        model = _gauss_model()
        loose = sensitivity.crb(model, 1e4)["Gauss_01_A"]
        model.lmfit_pars["Gauss_01_SD"].vary = False
        tight = sensitivity.crb(model, 1e4)["Gauss_01_A"]
        assert tight <= loose * (1 + 1e-9)


#
#
class TestValidation:
    """Guard rails on the inputs."""

    #
    @pytest.mark.parametrize("counts", [0.0, -1.0])
    def test_non_positive_counts_raises(self, counts):
        model = _gauss_model()
        with pytest.raises(ValueError, match="counts must be positive"):
            sensitivity.crb(model, counts)

    #
    @pytest.mark.parametrize("target", [0.0, -0.1])
    def test_non_positive_target_raises(self, target):
        model = _gauss_model()
        with pytest.raises(ValueError, match="target_sigma must be positive"):
            sensitivity.counts_required(model, "Gauss_01_x0", target)

    #
    def test_all_parameters_fixed_raises(self):
        model = _gauss_model()
        for name in model.parameter_names:
            model.lmfit_pars[name].vary = False
        with pytest.raises(ValueError, match="no free parameters"):
            sensitivity.crb(model, 1e4)


#
#
class TestReportShape:
    """The report is a usable table with the metadata attached."""

    #
    def test_columns_and_attrs(self):
        model = _gauss_model()
        frame = sensitivity.sensitivity_report(model, counts=1e4)
        assert list(frame.columns) == [
            "name",
            "value",
            "sigma",
            "sigma_uncorrelated",
            "correlation_penalty",
            "rel_precision",
        ]
        for key in ("counts", "counts_in_peaks", "n_bins", "condition_number"):
            assert key in frame.attrs
        assert frame.attrs["counts"] == pytest.approx(1e4)

    #
    def test_report_agrees_with_crb(self):
        model = _gauss_model()
        frame = sensitivity.sensitivity_report(model, counts=1e4)
        bounds = sensitivity.crb(model, 1e4)
        for _, row in frame.iterrows():
            assert row["sigma"] == pytest.approx(bounds[row["name"]], rel=1e-12)

    #
    def test_background_free_model_puts_all_counts_in_peaks(self):
        """gauss_free has no background component."""

        model = _gauss_model()
        frame = sensitivity.sensitivity_report(model, counts=1e4)
        assert frame.attrs["counts_in_peaks"] == pytest.approx(1e4, rel=1e-6)

    #
    def test_background_reduces_counts_in_peaks(self):
        """With a background, only part of the budget lands in the peaks."""

        model = _model("gauss_with_background", name="bg")
        frame = sensitivity.sensitivity_report(model, counts=1e4)
        assert 0.0 < frame.attrs["counts_in_peaks"] < 1e4

    #
    def test_derived_doublet_adds_no_free_parameters(self):
        """Expression-defined components contribute information, not freedom."""

        single = sensitivity.crb(_gauss_model(), 1e4)
        doublet = sensitivity.crb(_model("gauss_doublet", name="pair"), 1e4)
        assert set(doublet) == {"Gauss_01_A", "Gauss_01_x0", "Gauss_01_SD"}
        # Same total budget split across two tied lines: the shift bound is
        # set by total counts, so it should land close to the single-peak case.
        assert doublet["Gauss_01_x0"] == pytest.approx(single["Gauss_01_x0"], rel=0.15)


#
#
class TestModelIsUnchanged:
    """Finite differencing must not leave the model perturbed."""

    #
    def test_parameter_values_restored(self):
        model = _gauss_model()
        before = {n: model.lmfit_pars[n].value for n in model.parameter_names}
        sensitivity.sensitivity_report(model, counts=1e4)
        after = {n: model.lmfit_pars[n].value for n in model.parameter_names}
        assert before == after

    #
    def test_repeated_calls_agree(self):
        """No drift from accumulated finite-difference steps."""

        model = _gauss_model()
        first = sensitivity.crb(model, 1e4)
        second = sensitivity.crb(model, 1e4)
        assert first == pytest.approx(second, rel=1e-12)


#
#
class TestTwoDimensional:
    """Dynamics parameters get bounds through the same code path."""

    #
    def test_dynamics_parameter_bounded(self):
        """A time constant is just another parameter to the Fisher matrix."""

        project = make_project(name="sens_2d")
        energy = np.arange(83.0, 87.0, 0.05)
        time = np.linspace(-1.0, 8.0, 40)
        file = File(parent_project=project, name="d2", energy=energy, time=time)
        file.energy = energy.copy()
        file.time = time.copy()
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
        model = file.model_active
        model.energy = energy.copy()
        model.time = time.copy()

        frame = sensitivity.sensitivity_report(model, counts=1e6)
        tau = [n for n in frame["name"] if n.endswith("tau")]
        assert tau, f"no time constant among {list(frame['name'])}"
        row = frame.loc[frame["name"] == tau[0]].iloc[0]
        assert np.isfinite(row["sigma"])
        assert row["sigma"] > 0

    #
    def test_dynamics_bound_scales_with_counts(self):
        project = make_project(name="sens_2d_scale")
        energy = np.arange(83.0, 87.0, 0.05)
        time = np.linspace(-1.0, 8.0, 40)
        file = File(parent_project=project, name="d3", energy=energy, time=time)
        file.energy = energy.copy()
        file.time = time.copy()
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
        model = file.model_active
        model.energy = energy.copy()
        model.time = time.copy()

        low = sensitivity.crb(model, 1e6)
        high = sensitivity.crb(model, 4e6)
        for name, value in low.items():
            if np.isfinite(value) and value > 0:
                assert value / high[name] == pytest.approx(2.0, rel=1e-6)
