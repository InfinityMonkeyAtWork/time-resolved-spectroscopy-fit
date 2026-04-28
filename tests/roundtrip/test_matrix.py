"""Single-file roundtrip matrix.

Every cell follows the same shape: build a truth file, simulate clean 2D
data, build a fit file, run the workflow, assert exact recovery on all
non-expression parameters.  Backend (M/G/C) is selected by setting
``project.spec_fun_str``; the truth project always uses MCP so the
reference data is independent of the path under test.
"""

from __future__ import annotations

import pytest
from _utils import (
    assert_recovery_exact,
    extract_truth_pars,
    make_project,
    simulate_clean,
)

from .families import FAMILIES
from .matrix import cell_id, iter_cells
from .workflows import WORKFLOWS

pytestmark = pytest.mark.slow


_CELLS = list(iter_cells())


#
@pytest.mark.parametrize("cell", _CELLS, ids=[cell_id(c) for c in _CELLS])
def test_roundtrip_cell(cell):
    """Simulate from truth, fit through ``cell`` workflow + backend, recover."""

    family = FAMILIES[cell.family_id]
    workflow = WORKFLOWS[cell.workflow_id]
    model_name = family.model_name(cell.variant)

    truth_project = make_project(name="truth", spec_fun_str="fit_model_mcp")
    truth_file = family.build_truth(truth_project, variant=cell.variant)
    truth_pars = extract_truth_pars(truth_file.model_active)

    data = simulate_clean(truth_file.model_active)

    fit_project = make_project(name="fit", spec_fun_str=cell.spec_fun_str)
    fit_kwargs = {
        "data": data,
        "energy": truth_file.energy,
        "time": truth_file.time,
        "variant": cell.variant,
    }
    if family.needs_aux:
        fit_kwargs["aux"] = truth_file.aux_axis

    fit_file = family.build_fit(fit_project, **fit_kwargs)
    result = workflow.run(fit_file, family, model_name, cell.variant)

    assert_recovery_exact(truth_pars, result.params)
