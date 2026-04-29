"""``parameter_names`` / ``lmfit_pars`` ordering invariants across topology.

``parameter_names`` is the positional contract that ``update_value``,
``ulmfit.par_extract``, SbS seed templates, and the GIR ``theta_indices``
all rely on. Topology matters because the flattening crosses
Components × ``t_model`` × ``p_model`` and the order is non-obvious for
F8 (profile-internal dynamics) and F12 (mixed expressions). These tests
pin two invariants single-static-model coverage can't:

- ``list(model.lmfit_pars)`` iterates in ``model.parameter_names`` order.
- Rebuilding the same family produces an identical ``parameter_names``
  list (deterministic flattening).
"""

from __future__ import annotations

import pytest
from _utils import make_project

from .families import FAMILIES

pytestmark = pytest.mark.slow


_FAMILIES = ("F3", "F6", "F8", "F12")


#
def _build_truth_model(family_id: str, suffix: str):
    """Build a truth file for ``family_id`` and return its active model."""

    family = FAMILIES[family_id]
    project = make_project(
        name=f"ord_{family_id}_{suffix}", spec_fun_str="fit_model_mcp"
    )
    truth_file = family.build_truth(project, variant="default")
    assert truth_file.model_active is not None  # type guard
    return truth_file.model_active


#
@pytest.mark.parametrize("family_id", _FAMILIES)
def test_lmfit_pars_iter_matches_parameter_names(family_id):
    """``list(model.lmfit_pars)`` must equal ``model.parameter_names``.

    ``update_value(par_select='all')`` enumerates ``lmfit_pars`` and
    indexes positionally into the input list; if the iteration order
    drifts from ``parameter_names``, every consumer that built its
    input via ``parameter_names`` (par_extract, GIR theta_indices, SbS
    seeds) silently writes to the wrong slot.
    """

    model = _build_truth_model(family_id, "single")
    assert list(model.lmfit_pars) == list(model.parameter_names)


#
@pytest.mark.parametrize("family_id", _FAMILIES)
def test_parameter_names_deterministic_across_rebuild(family_id):
    """Rebuilding a family twice produces an identical ``parameter_names``."""

    model_a = _build_truth_model(family_id, "a")
    model_b = _build_truth_model(family_id, "b")
    assert list(model_a.parameter_names) == list(model_b.parameter_names)
