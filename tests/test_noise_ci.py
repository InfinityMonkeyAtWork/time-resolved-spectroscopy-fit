"""Profiled confidence intervals under a declared noise model.

``fitlib.fit_wrapper`` hands ``lmfit.conf_interval`` a chi-square threshold
(``chi2.cdf(delta_chisqr, nfix)``) whenever the fitted view carries a
declared noise model, and keeps lmfit's F-test for ``'unknown'``. The F-test
statistic is the *ratio* ``new_chi / best_chi - 1``, so it profiles out the
very noise scale ``File.set_noise`` fixes and cannot see it: the two
thresholds agree only at ``chi2_red ~ 1``. These tests pin the interval to
the declared noise instead.

``conf_interval`` refuses fewer than two varying parameters, so every case
fits the four-parameter ``single_glp`` to data generated from that same
model.
"""

import numpy as np
import pytest
from _utils import make_project

from trspecfit import File
from trspecfit.functions.energy import GLP

_ENERGY = np.linspace(83.0, 87.0, 30)
_TIME = np.linspace(0.0, 5.0, 4)
# offset from the YAML start values so the optimizer has somewhere to go
_TRUTH = {"A": 18.5, "x0": 85.2, "F": 1.1, "m": 0.35}
_SIGMA = 0.2


#
def _clean_data():
    """2D data whose every slice is exactly the fitted GLP model."""

    return np.tile(GLP(_ENERGY, **_TRUTH), (len(_TIME), 1))


#
def _fit_with_ci(data, *, noise_type, **noise_kwargs):
    """Fit slice 0 with 1-sigma profile CIs under the declared noise."""

    project = make_project(name="noise_ci")
    file = File(
        parent_project=project, name="ci", data=data, energy=_ENERGY, time=_TIME
    )
    file.load_model(model_yaml="models/file_energy.yaml", model_info="single_glp")
    file.define_baseline(0, 1, time_type="ind", show_plot=False)
    file.set_noise(noise_type, **noise_kwargs)
    file.fit_baseline(model_name="single_glp", stages=2, try_ci=1, ci_sigmas=[1.0])
    return file


#
def _half_widths(file):
    """``{parameter: (lower, upper)}`` 1-sigma half-widths of the profiled CI."""

    ci = file.get_confidence_intervals().set_index("par[v]/sigma[>]")
    assert not ci.empty
    return {
        par_name: (row["best fit"] - row["-1.0"], row["+1.0"] - row["best fit"])
        for par_name, row in ci.iterrows()
    }


#
def _stderrs(file):
    """``{parameter: stderr}`` of the same fit."""

    pars = file.get_parameters().set_index("name")
    return {par_name: float(pars.loc[par_name, "stderr"]) for par_name in pars.index}


#
def _chi2_red(file):
    return file.p._fit_history[-1].metrics["chi2_red"]


#
#
class TestChiSquareThreshold:
    #
    def test_interval_scales_with_the_declared_sigma(self):
        """Sigma declared 2x too small halves every 1-sigma half-width.

        lmfit's F-test would not move at all: its statistic
        ``new_chi / best_chi - 1`` is invariant under a constant rescaling
        of the residual.
        """

        rng = np.random.default_rng(5)
        clean = _clean_data()
        data = clean + _SIGMA * rng.standard_normal(clean.shape)
        honest = _half_widths(_fit_with_ci(data, noise_type="gaussian", sigma=_SIGMA))
        tight = _half_widths(
            _fit_with_ci(data, noise_type="gaussian", sigma=_SIGMA / 2)
        )

        for par_name, (lower, upper) in honest.items():
            lower_tight, upper_tight = tight[par_name]
            # measured 1.96-2.04 over the four GLP parameters; the spread is
            # the profile asymmetry, which the narrower interval sees less of
            assert lower / lower_tight == pytest.approx(2.0, rel=0.08)
            assert upper / upper_tight == pytest.approx(2.0, rel=0.08)

    #
    def test_near_noiseless_data_keeps_the_interval_at_stderr(self):
        """chi2_red << 1: finite intervals, each of them one stderr wide.

        The residual scatter is 50x below the declared sigma, which is the
        regime where the F-test collapses the interval to the scatter
        (measured: 2-9% of stderr) instead of the declared noise.
        """

        rng = np.random.default_rng(5)
        clean = _clean_data()
        data = clean + 1e-3 * rng.standard_normal(clean.shape)
        file = _fit_with_ci(data, noise_type="gaussian", sigma=0.05)
        assert _chi2_red(file) < 0.01

        widths, stderrs = _half_widths(file), _stderrs(file)
        assert set(widths) == set(stderrs)
        for par_name, (lower, upper) in widths.items():
            assert np.isfinite(lower) and np.isfinite(upper)
            # measured 0.991-1.009 (GLP is linear in A, near-linear in the rest)
            assert lower / stderrs[par_name] == pytest.approx(1.0, rel=0.05)
            assert upper / stderrs[par_name] == pytest.approx(1.0, rel=0.05)

    #
    def test_poisson_counts_give_intervals_near_stderr(self):
        """Realistic counting data: the profiled interval agrees with the
        curvature of the same weighted chi-square."""

        rng = np.random.default_rng(5)
        scale = 20.0
        file = _fit_with_ci(
            rng.poisson(scale * _clean_data()) / scale,
            noise_type="poisson",
            scale=scale,
        )
        assert 0.5 < _chi2_red(file) < 2.0

        widths, stderrs = _half_widths(file), _stderrs(file)
        for par_name, (lower, upper) in widths.items():
            assert np.isfinite(lower) and np.isfinite(upper)
            # measured 0.96-1.07 at 370 peak counts; the band covers the
            # profile asymmetry from about 90 peak counts upward, below which
            # the asymmetry is the interval's real shape, not an error
            assert lower / stderrs[par_name] == pytest.approx(1.0, rel=0.2)
            assert upper / stderrs[par_name] == pytest.approx(1.0, rel=0.2)

    #
    def test_model_mismatch_on_near_noiseless_data_completes(self):
        """A misspecified model on near-noiseless data with a declared sigma.

        The residual is then structured rather than random and the F-test's
        ratio statistic is not monotonic along the profile, so lmfit's
        bracketing fails (``f(a) and f(b) must have different signs``). The
        chi-square threshold profiles the declared sigma and completes.
        """

        energy = np.linspace(83.0, 87.0, 24)
        time = np.linspace(0.0, 5.0, 6)
        gauss = 20.0 * np.exp(-0.5 * ((energy - 85.0) / 0.6) ** 2) + 0.5
        data = gauss[None, :] * (1.0 + 0.05 * np.arange(len(time)))[:, None]
        project = make_project(name="noise_ci")
        file = File(
            parent_project=project, name="ci", data=data, energy=energy, time=time
        )
        file.load_model(model_yaml="models/file_energy.yaml", model_info="single_glp")
        file.define_baseline(0, 2, time_type="ind", show_plot=False)
        file.set_fit_limits([84.0, 86.0], show_plot=False)
        file.set_sigma(0.5)
        file.fit_baseline(model_name="single_glp", try_ci=1)

        assert _chi2_red(file) < 0.01
        widths = _half_widths(file)
        # the bounded shape parameter may run into its bound (lmfit reports
        # -inf there); the free peak parameters must have finite intervals
        for par_name in ("GLP_01_A", "GLP_01_x0"):
            lower, upper = widths[par_name]
            assert np.isfinite(lower) and np.isfinite(upper)
            assert lower > 0 and upper > 0
