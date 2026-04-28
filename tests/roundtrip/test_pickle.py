"""Pickle / deepcopy evaluator-parity for non-static roundtrip families.

Complements ``TestMCPPickling`` in ``tests/test_mcp_library.py``, which
covers static models only (``single_glp``, no ``t_vary``, no ``p_vary``).
Here we exercise families that carry a ``Par.t_model`` (Dynamics),
``Par.p_model`` (Profile), or both, and assert that the restored Model's
2D evaluator is byte-identical to the original. This locks down the
contract that ``Par.t_model`` / ``Par.p_model`` survive the
``parent_model`` nulling baked into the pickle hooks.
"""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest
from _utils import make_project

from .families import FAMILIES

pytestmark = pytest.mark.slow


# Cover the four non-static regimes:
# - F3:  top-level dynamics only (Par.t_model)
# - F6:  top-level profile only (Par.p_model x2)
# - F8:  profile-internal dynamics (Par on a Profile component carries t_model)
# - F12: mixed expression referencing a profiled and a time-dep base par
_FAMILIES = ("F3", "F6", "F8", "F12")


#
def _build_truth_model(family_id: str):
    """Build a truth file for ``family_id`` and return its active model."""

    family = FAMILIES[family_id]
    project = make_project(name=f"pickle_{family_id}", spec_fun_str="fit_model_mcp")
    truth_file = family.build_truth(project, variant="default")
    assert truth_file.model_active is not None  # type guard
    return truth_file.model_active


#
@pytest.mark.parametrize("family_id", _FAMILIES)
def test_pickle_value_2d_parity(family_id):
    """Pickled non-static Model reproduces ``value_2d`` byte-for-byte."""

    model = _build_truth_model(family_id)
    model.create_value_2d()
    assert model.value_2d is not None  # type guard
    expected = model.value_2d.copy()

    restored = pickle.loads(pickle.dumps(model))
    restored.create_value_2d()
    np.testing.assert_array_equal(restored.value_2d, expected)


#
@pytest.mark.parametrize("family_id", _FAMILIES)
def test_deepcopy_value_2d_parity(family_id):
    """``copy.deepcopy`` non-static Model reproduces ``value_2d`` byte-for-byte."""

    model = _build_truth_model(family_id)
    model.create_value_2d()
    assert model.value_2d is not None  # type guard
    expected = model.value_2d.copy()

    clone = copy.deepcopy(model)
    clone.create_value_2d()
    np.testing.assert_array_equal(clone.value_2d, expected)
