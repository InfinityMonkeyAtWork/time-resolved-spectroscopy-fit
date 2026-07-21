"""Unit tests for fitlib bridge functions."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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


#
class TestPltFitRes2dNanAware:
    """plt_fit_res_2d must handle a NaN-padded fit array (full_range mode)
    without warning/crashing, and report min/max from the real values."""

    #
    def test_nan_padded_fit_renders_without_warning(self, recwarn):
        rng = np.random.default_rng(0)
        data = rng.random((6, 8))
        fit = np.full((6, 8), np.nan)
        fit[2:4, 3:6] = data[2:4, 3:6] * 0.9  # the "fit window"

        fitlib.plt_fit_res_2d(
            data=data,
            fit=fit,
            x_lim=[3, 6],
            y_lim=[2, 4],
            save_img=0,
        )
        fig = plt.gcf()
        try:
            assert not any("All-NaN" in str(w.message) for w in recwarn.list)
            fit_ax = next(
                ax for ax in fig.axes if ax.get_title().startswith("Fit [min:")
            )
            expected_min = np.nanmin(fit)
            assert f"{expected_min:.3E}" in fit_ax.get_title()
        finally:
            plt.close("all")
