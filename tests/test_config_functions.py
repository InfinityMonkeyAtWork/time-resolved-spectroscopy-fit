"""Unit tests for get_function_parameters() from config/functions.py.

Tests that the parameter introspection correctly strips the axis argument
(x or t) and the 'spectrum' argument from background functions, returning
only the user-facing parameter names that appear in YAML model definitions.
"""

import pytest

from trspecfit.config.functions import get_function_parameters


#
class TestGetFunctionParameters:
    """Test parameter introspection for energy, time, profile, convolution functions."""

    #
    # energy: peak functions — first arg (x) stripped
    #
    def test_GLP(self):
        assert get_function_parameters("GLP") == ["A", "x0", "F", "m"]

    def test_Gauss(self):
        assert get_function_parameters("Gauss") == ["A", "x0", "SD"]

    def test_Lorentz(self):
        assert get_function_parameters("Lorentz") == ["A", "x0", "W"]

    def test_Voigt(self):
        assert get_function_parameters("Voigt") == ["A", "x0", "SD", "W"]

    def test_DS(self):
        assert get_function_parameters("DS") == ["A", "x0", "F", "alpha"]

    #
    # energy: background functions — x stripped AND spectrum stripped
    #
    def test_Offset_strips_spectrum(self):
        assert get_function_parameters("Offset") == ["y0"]

    def test_Shirley_strips_spectrum(self):
        assert get_function_parameters("Shirley") == ["pShirley"]

    def test_LinBack_strips_spectrum(self):
        assert get_function_parameters("LinBack") == ["pLinear"]

    def test_LinBackRev_strips_spectrum(self):
        assert get_function_parameters("LinBackRev") == ["pLinear"]

    #
    # time: dynamics functions — first arg (t) stripped
    #
    def test_expFun(self):
        assert get_function_parameters("expFun") == ["A", "tau", "t0", "y0"]

    def test_sinFun(self):
        assert get_function_parameters("sinFun") == ["A", "f", "phi", "t0", "y0"]

    def test_none_has_no_params(self):
        """none() takes only t, so parameters should be empty."""
        assert get_function_parameters("none") == []

    #
    # time: convolution kernels — first arg (x) stripped
    #
    def test_gaussCONV(self):
        assert get_function_parameters("gaussCONV") == ["SD"]

    def test_voigtCONV(self):
        assert get_function_parameters("voigtCONV") == ["SD", "W"]

    #
    # profile functions — first arg (x) stripped
    #
    def test_exp_decay(self):
        assert get_function_parameters("exp_decay") == ["A", "tau"]

    def test_linear(self):
        assert get_function_parameters("linear") == ["m", "b"]

    #
    # unknown function
    #
    def test_unknown_returns_empty(self):
        assert get_function_parameters("nonexistent_function_xyz") == []


if __name__ == "__main__":
    pytest.main([__file__])
