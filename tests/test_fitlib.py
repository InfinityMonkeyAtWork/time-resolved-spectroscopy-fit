"""Unit tests for fitlib bridge functions."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from _utils import make_project

from trspecfit import File, fitlib


#
def _make_1d_model_file():
    """File with a loaded 1D energy model (public API)."""

    project = make_project(name="fitlib")
    file = File(parent_project=project, energy=np.linspace(80, 90, 101))
    file.load_model(model_yaml="models/file_energy.yaml", model_info="single_glp")
    assert file.model_active is not None  # type guard
    return file


#
#
class TestComputeFitMetricsUndefinedCount:
    """``n_free_pars=None`` marks the parameter count as undefined — the
    per-file projections of a project-level joint fit, whose joint count
    does not decompose by file."""

    #
    def test_count_dependent_metrics_nan_others_unchanged(self):
        observed = np.array([1.0, 2.0, 3.0, 4.0])
        fit = np.array([1.1, 1.9, 3.2, 3.8])
        m_none = fitlib.compute_fit_metrics(
            observed=observed, fit=fit, n_free_pars=None, sigma_eff=0.5
        )
        for key in ("chi2_red_raw", "chi2_red", "aic", "bic"):
            assert np.isnan(m_none[key]), key
        m_two = fitlib.compute_fit_metrics(
            observed=observed, fit=fit, n_free_pars=2, sigma_eff=0.5
        )
        for key in ("chi2_raw", "chi2", "r2"):
            assert np.isfinite(m_none[key]), key
            assert m_none[key] == m_two[key], key


#
#
class TestResultsToFit2D:
    """DataFrame-path column handling in results_to_fit_2d."""

    #
    def _const_args(self, file):
        """const/args as the SbS fit path builds them (per-slice, MCP)."""

        model = file.model_active
        assert model is not None  # type guard
        data = np.zeros_like(np.asarray(file.energy))
        const = (file.energy, data, "fit_model_mcp", 0, [], [])
        args = (model, 1)
        return model, const, args

    #
    def test_parameter_names_selects_and_orders_columns(self):
        """Extra columns and scrambled order are handled via parameter_names."""

        file = _make_1d_model_file()
        model, const, args = self._const_args(file)

        values = [model.lmfit_pars[n].value for n in model.parameter_names]
        df_pars = pd.DataFrame([values, values], columns=model.parameter_names)
        df_extra = df_pars.copy()
        df_extra["chi2"] = [0.1, 0.2]  # non-parameter column
        # scramble column order on top of the extra column
        df_extra = df_extra[["chi2", *reversed(model.parameter_names)]]

        fit_ref = fitlib.results_to_fit_2d(df_pars, const, args)
        fit_sel = fitlib.results_to_fit_2d(
            df_extra, const, args, parameter_names=model.parameter_names
        )

        np.testing.assert_allclose(fit_sel, fit_ref)
        assert fit_ref.shape == (2, len(file.energy))

    #
    def test_missing_parameter_column_raises(self):
        """Requesting a parameter column absent from the DataFrame fails."""

        file = _make_1d_model_file()
        model, const, args = self._const_args(file)

        values = [model.lmfit_pars[n].value for n in model.parameter_names]
        df_pars = pd.DataFrame([values], columns=model.parameter_names)
        df_missing = df_pars.drop(columns=model.parameter_names[:1])

        with pytest.raises(KeyError):
            fitlib.results_to_fit_2d(
                df_missing, const, args, parameter_names=model.parameter_names
            )
