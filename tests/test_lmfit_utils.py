"""Unit tests for trspecfit.utils.lmfit helpers not covered elsewhere."""

import lmfit

from trspecfit.utils.lmfit import restore_true_init_values


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
