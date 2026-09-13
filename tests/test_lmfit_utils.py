"""Unit tests for trspecfit.utils.lmfit helpers not covered elsewhere."""

import typing

import lmfit
import pandas as pd
from lmfit.minimizer import MinimizerResult

from trspecfit.utils import lmfit as ulmfit
from trspecfit.utils.lmfit import (
    FitOutput,
    list_of_par_ini_to_df,
    restore_true_init_values,
)


#
def _make_fit_output(
    fin_values: dict[str, float], ini_values: dict[str, float]
) -> FitOutput:
    """Minimal FitOutput with real lmfit.Parameters for par_ini/par_fin.params."""

    par_fin_params = lmfit.Parameters()
    for name, value in fin_values.items():
        par_fin_params.add(name, value=value)
    par_fin = typing.cast("ulmfit.TypedMinimizerResult", MinimizerResult())
    par_fin.params = par_fin_params

    par_ini = lmfit.Parameters()
    for name, value in ini_values.items():
        par_ini.add(name, value=value)

    return FitOutput(
        par_ini=par_ini,
        par_fin=par_fin,
        conf_ci=pd.DataFrame(),
        emcee_fin=None,
        emcee_ci=pd.DataFrame(),
    )


#
class TestRestoreTrueInitValues:
    """restore_true_init_values — corrects a two-stage fit result's
    init_value back to the true pre-fit seed (fitlib.fit_wrapper's
    par_ini), which lmfit otherwise resets to stage 1's output."""

    #
    def test_overwrites_init_value_from_par_ini(self):
        result_params = lmfit.Parameters()
        result_params.add("A", value=5.0)
        result_params.add("B", value=10.0)
        # simulate lmfit's prepare_fit() stamping stage-1's output as
        # stage-2's init_value
        result_params["A"].init_value = 5.0
        result_params["B"].init_value = 10.0

        par_ini = lmfit.Parameters()
        par_ini.add("A", value=1.0)
        par_ini.add("B", value=2.0)

        restore_true_init_values(result_params, par_ini)

        assert result_params["A"].init_value == 1.0
        assert result_params["B"].init_value == 2.0

    #
    def test_leaves_value_and_stderr_untouched(self):
        result_params = lmfit.Parameters()
        result_params.add("A", value=5.0)
        result_params["A"].init_value = 5.0
        result_params["A"].stderr = 0.1

        par_ini = lmfit.Parameters()
        par_ini.add("A", value=1.0)

        restore_true_init_values(result_params, par_ini)

        assert result_params["A"].value == 5.0
        assert result_params["A"].stderr == 0.1

    #
    def test_ignores_names_absent_from_par_ini(self):
        """A parameter present in result_params but not par_ini is left alone
        (defensive; shouldn't happen in practice since both come from the
        same fit_wrapper call)."""

        result_params = lmfit.Parameters()
        result_params.add("A", value=5.0)
        result_params["A"].init_value = 5.0

        par_ini = lmfit.Parameters()  # empty

        restore_true_init_values(result_params, par_ini)

        assert result_params["A"].init_value == 5.0


#
class TestListOfParIniToDf:
    """list_of_par_ini_to_df — per-slice true initial-guess values, the
    SbS companion to list_of_par_stderr_to_df; reads FitOutput.par_ini
    directly (unaffected by fitlib.fit_wrapper's init_value correction)."""

    #
    def test_shape_and_columns_match_par_fin(self):
        results = [
            _make_fit_output({"A": 5.0, "B": 10.0}, {"A": 1.0, "B": 2.0}),
            _make_fit_output({"A": 6.0, "B": 11.0}, {"A": 1.5, "B": 2.5}),
        ]

        df = list_of_par_ini_to_df(results)

        assert list(df.columns) == ["A", "B"]
        assert len(df) == 2

    #
    def test_values_are_the_true_seed_not_the_fit_result(self):
        results = [_make_fit_output({"A": 5.0}, {"A": 1.0})]

        df = list_of_par_ini_to_df(results)

        assert df.iloc[0]["A"] == 1.0
