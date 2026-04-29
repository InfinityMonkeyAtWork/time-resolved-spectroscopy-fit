"""``Model.update_value`` matrix coverage for non-static families.

Mirrors ``test_pickle.py``. ``update_value`` is the dual of evaluation:
it writes positional values back into ``lmfit_pars`` along the same
flattening that ``parameter_names`` defines, and runs every fit
iteration plus every SbS slice reset. The single-static-case coverage
in ``tests/test_mcp_library.py`` and ``tests/test_gir_integration.py``
won't surface ordering / dropped-par / cache-invalidation bugs that
only appear once Components grow Dynamics or Profile sub-Models.
"""

from __future__ import annotations

import numpy as np
import pytest
from _utils import make_project

from .families import FAMILIES

pytestmark = pytest.mark.slow


_FAMILIES = ("F3", "F6", "F8", "F12")


#
def _build_truth_model(family_id: str):
    """Build a truth file for ``family_id`` and return its active model."""

    family = FAMILIES[family_id]
    project = make_project(name=f"upd_{family_id}", spec_fun_str="fit_model_mcp")
    truth_file = family.build_truth(project, variant="default")
    assert truth_file.model_active is not None  # type guard
    return truth_file.model_active


#
def _extract_values(model) -> list[float]:
    """Return current ``lmfit_pars`` values in ``parameter_names`` order."""

    return [model.lmfit_pars[n].value for n in model.parameter_names]


#
@pytest.mark.parametrize("family_id", _FAMILIES)
def test_update_value_all_identity(family_id):
    """``update_value(current, 'all')`` is a no-op on the value list."""

    model = _build_truth_model(family_id)
    expected = _extract_values(model)

    model.update_value(expected, par_select="all")
    actual = _extract_values(model)

    assert actual == expected


#
@pytest.mark.parametrize("family_id", _FAMILIES)
def test_update_value_list_identity(family_id):
    """``par_select=[names]`` no-op matches ``par_select='all'`` no-op."""

    model = _build_truth_model(family_id)
    expected = _extract_values(model)

    model.update_value(expected, par_select=list(model.parameter_names))
    actual = _extract_values(model)

    assert actual == expected


#
@pytest.mark.parametrize("family_id", _FAMILIES)
def test_update_value_no_op_preserves_value_2d(family_id):
    """Re-writing the same values must not perturb ``value_2d``.

    Catches stale-cache bugs in the evaluator: ``update_value`` mutates
    ``lmfit_pars`` but doesn't itself trigger evaluation, so any cached
    component state must be invalidated correctly on next eval.
    """

    model = _build_truth_model(family_id)
    model.create_value_2d()
    assert model.value_2d is not None  # type guard
    expected = model.value_2d.copy()

    model.update_value(_extract_values(model), par_select="all")
    model.create_value_2d()
    np.testing.assert_array_equal(model.value_2d, expected)


#
@pytest.mark.parametrize("family_id", _FAMILIES)
def test_update_value_perturbation_propagates(family_id):
    """Perturbing one vary non-expr par must change ``value_2d``.

    Catches "``update_value`` silently dropped a par" bugs: if the
    targeted slot wasn't actually written, the evaluator would return
    the baseline output and the test would fail.
    """

    model = _build_truth_model(family_id)
    model.create_value_2d()
    assert model.value_2d is not None  # type guard
    baseline = model.value_2d.copy()

    target_idx = None
    for i, name in enumerate(model.parameter_names):
        lp = model.lmfit_pars[name]
        if lp.expr is None and lp.vary:
            target_idx = i
            break
    assert target_idx is not None, f"{family_id}: no vary non-expr par to perturb"

    new_values = _extract_values(model)
    old = new_values[target_idx]
    new_values[target_idx] = old * 1.01 if old != 0.0 else 0.01

    model.update_value(new_values, par_select="all")
    model.create_value_2d()
    assert not np.array_equal(model.value_2d, baseline)
