"""Noisy roundtrips — second-layer coverage on top of the clean matrix.

Per the matrix doc, noisy is *not* an axis on the main matrix: the
assertion shape differs (5% relative tolerance + skip near-zero), and
parametrising it doubles every cell's runtime for marginal gain. Instead,
hand-pick representative cells that exercise distinct code paths
(plain energy / dynamics / profile-only / profile-internal dynamics).
"""

from __future__ import annotations

import pytest
from _utils import (
    assert_recovery_within,
    extract_truth_pars,
    make_project,
    simulate_noisy,
)

from .families import FAMILIES
from .matrix import ResolvedCell, cell_id
from .workflows import WORKFLOWS

pytestmark = pytest.mark.slow


_NOISY_CELLS = [
    ResolvedCell(family_id="F3", workflow_id="2D", backend="G", variant="default"),
    ResolvedCell(family_id="F6", workflow_id="B", backend="G", variant="default"),
    ResolvedCell(family_id="F8", workflow_id="2D", backend="G", variant="default"),
]


#
@pytest.mark.parametrize("cell", _NOISY_CELLS, ids=[cell_id(c) for c in _NOISY_CELLS])
def test_noisy_roundtrip(cell):
    """Noisy fit should recover truth parameters within 5% relative error."""

    family = FAMILIES[cell.family_id]
    workflow = WORKFLOWS[cell.workflow_id]
    model_name = family.model_name(cell.variant)

    truth_project = make_project(name="truth_noisy", spec_fun_str="fit_model_mcp")
    truth_file = family.build_truth(truth_project, variant=cell.variant)
    truth_pars = extract_truth_pars(truth_file.model_active)

    data = simulate_noisy(truth_file.model_active, noise_level=0.01)

    fit_project = make_project(name="fit_noisy", spec_fun_str=cell.spec_fun_str)
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

    assert_recovery_within(truth_pars, result.params, rel_tol=0.05)
