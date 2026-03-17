"""Unit tests for get_function_parameters() from config/functions.py.

Tests that the parameter introspection correctly strips the axis argument
(x or t) and the 'spectrum' argument from background functions, returning
only the user-facing parameter names that appear in YAML model definitions.
"""

import inspect

import pytest

from trspecfit.config.functions import get_function_parameters
from trspecfit.functions import energy, profile, time


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
        assert get_function_parameters("LinBack") == ["m", "b", "xStart", "xStop"]

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
    def test_pExpDecay(self):
        assert get_function_parameters("pExpDecay") == ["A", "tau"]

    def test_pLinear(self):
        assert get_function_parameters("pLinear") == ["m", "b"]

    #
    # unknown function
    #
    def test_unknown_returns_empty(self):
        assert get_function_parameters("nonexistent_function_xyz") == []


#
#
class TestNoUnderscoresInNames:
    """Underscores are reserved as YAML separators (numbering, names, parameters).

    No public function name or parameter name in any function module may
    contain an underscore.
    """

    MODULES = [energy, time, profile]
    INTERNAL_SUFFIXES = ("_kernel_width",)

    #
    def _public_callables(self):
        """Yield (module, name, func) for every public callable."""

        for mod in self.MODULES:
            for name in dir(mod):
                if name.startswith("_"):
                    continue
                obj = getattr(mod, name)
                if callable(obj):
                    yield mod, name, obj

    #
    def test_no_underscore_in_function_names(self):
        for mod, name, _func in self._public_callables():
            if any(name.endswith(s) for s in self.INTERNAL_SUFFIXES):
                continue
            assert "_" not in name, f"{mod.__name__}.{name} contains an underscore"

    #
    def test_profile_functions_start_with_p(self):
        for _mod, name, _func in self._public_callables():
            if _mod is not profile:
                continue
            assert name.startswith("p"), f"profile.{name} must start with 'p'"

    #
    def test_no_underscore_in_parameter_names(self):
        for mod, name, func in self._public_callables():
            if any(name.endswith(s) for s in self.INTERNAL_SUFFIXES):
                continue
            sig = inspect.signature(func)
            for par_name in sig.parameters:
                assert "_" not in par_name, (
                    f"{mod.__name__}.{name}() parameter '{par_name}' "
                    "contains an underscore"
                )


if __name__ == "__main__":
    pytest.main([__file__])
