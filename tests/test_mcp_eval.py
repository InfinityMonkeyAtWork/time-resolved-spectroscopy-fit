"""Evaluation-focused MCP tests.

These tests complement parser tests by exercising model evaluation paths
that should catch regressions in dependent/time-dependent parameter handling.
"""

import numpy as np

from trspecfit import File, Project


def _build_file_with_axes() -> File:
    """Create a test File with deterministic axes for model evaluation."""
    project = Project(path="tests")
    file = File(parent_project=project)
    file.energy = np.linspace(80, 90, 201)
    file.time = np.linspace(-10, 100, 111)
    return file


def _find_parameter(model, name: str):
    """Return parameter object by full name from a loaded model."""
    for comp in model.components:
        for par in comp.pars:
            if par.name == name:
                return par
    raise AssertionError(f"Parameter '{name}' not found")


def test_eval_energy_expression_value1d():
    """Dependent expression parameters should evaluate correctly in 1D."""
    file = _build_file_with_axes()
    file.load_model(
        model_yaml="test_models_energy.yaml",
        model_info=["energy_expression"],
        debug=False,
    )

    model = file.model_active
    assert model is not None

    value_1d = model.create_value1D(return1D=1)
    assert value_1d is not None
    assert value_1d.shape == file.energy.shape
    assert np.isfinite(value_1d).all()

    p_x0_1 = _find_parameter(model, "GLP_01_x0")
    p_x0_2 = _find_parameter(model, "GLP_02_x0")

    v1 = p_x0_1.value(t_ind=0)
    v2 = p_x0_2.value(t_ind=0)
    assert v1[0] is not None
    assert v2[0] is not None
    assert np.isclose(v2[0], v1[0] + 3.6)


def test_eval_time_dependent_expression_value2d():
    """Expressions depending on time-dependent parameters should evaluate in 2D."""
    file = _build_file_with_axes()
    file.load_model(
        model_yaml="test_models_energy.yaml",
        model_info=["energy_expression"],
        debug=False,
    )
    file.add_time_dependence(
        model_yaml="test_models_time.yaml",
        model_info=["MonoExpPosIRF"],
        par_name="GLP_01_x0",
    )

    model = file.model_active
    assert model is not None
    assert model.dim == 2

    p_x0_1 = _find_parameter(model, "GLP_01_x0")
    p_x0_2 = _find_parameter(model, "GLP_02_x0")

    v1_t0 = p_x0_1.value(t_ind=0)
    v2_t0 = p_x0_2.value(t_ind=0)
    assert v1_t0[0] is not None
    assert v2_t0[0] is not None
    assert np.isclose(v2_t0[0], v1_t0[0] + 3.6)

    model.create_value2D()
    assert model.value2D is not None
    assert model.value2D.shape == (len(file.time), len(file.energy))
    assert np.isfinite(model.value2D).all()
