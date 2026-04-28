"""Static matrix declaration for the roundtrip test suite.

The :data:`MATRIX` list mirrors the per-cell coverage in
``docs/design/roundtrip_test_matrix.md``.  Adding a new family/workflow
combination is a one-line edit here.

:func:`iter_cells` expands ``MATRIX`` into one :class:`ResolvedCell` per
``(family, workflow, backend, variant)`` tuple — that is the form
:func:`pytest.mark.parametrize` consumes.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .families import FAMILIES

BACKEND_TO_SPEC_FUN = {
    "M": "fit_model_mcp",
    "G": "fit_model_gir",
    "C": "fit_model_compare",
}

_ALL_BACKENDS = ("M", "G", "C")


#
#
@dataclass(frozen=True)
class Cell:
    """One row of the matrix: a (family, workflow) pair with its backends."""

    family_id: str
    workflow_id: str
    backends: tuple[str, ...] = _ALL_BACKENDS
    variants: tuple[str, ...] | None = None  # None -> use family.variants


#
#
@dataclass(frozen=True)
class ResolvedCell:
    """A fully expanded cell — what each parametrized test invocation runs."""

    family_id: str
    workflow_id: str
    backend: str
    variant: str

    @property
    def spec_fun_str(self) -> str:
        return BACKEND_TO_SPEC_FUN[self.backend]


# ---- the matrix ----
#
# Mirrored from docs/design/roundtrip_test_matrix.md.  Each row corresponds
# to one (family, workflow) cell with all its applicable backends.

MATRIX: list[Cell] = [
    # F1 — plain energy: B, Sp, SbS
    Cell("F1", "B"),
    Cell("F1", "Sp"),
    Cell("F1", "SbS"),
    # F2 — static expressions: B, Sp, SbS across direct/fan_out/forward_ref
    Cell("F2", "B"),
    Cell("F2", "Sp"),
    Cell("F2", "SbS"),
    # F3 — top-level standard dynamics: 2D
    Cell("F3", "2D"),
    # F4 — IRF dynamics: 2D
    Cell("F4", "2D"),
    # F5 — subcycle dynamics: 2D
    Cell("F5", "2D"),
    # F6 — top-level profile only: B, Sp, SbS
    Cell("F6", "B"),
    Cell("F6", "Sp"),
    Cell("F6", "SbS"),
    # F7 — profile + separate dynamics: 2D
    Cell("F7", "2D"),
    # F8 — profile-internal dynamics: 2D
    Cell("F8", "2D"),
    # F9 — expr -> time-dep base par: 2D
    Cell("F9", "2D"),
    # F10 — expr -> profiled base par: B, Sp, SbS
    Cell("F10", "B"),
    Cell("F10", "Sp"),
    Cell("F10", "SbS"),
    # F11 — expr -> profiled base par with profile-internal dynamics: 2D
    Cell("F11", "2D"),
    # F12 — mixed expr: 2D
    Cell("F12", "2D"),
]


#
def iter_cells() -> Iterable[ResolvedCell]:
    """Expand :data:`MATRIX` into per-test ``ResolvedCell`` instances."""

    for cell in MATRIX:
        family = FAMILIES[cell.family_id]
        variants = cell.variants if cell.variants is not None else family.variants
        for variant in variants:
            for backend in cell.backends:
                yield ResolvedCell(
                    family_id=cell.family_id,
                    workflow_id=cell.workflow_id,
                    backend=backend,
                    variant=variant,
                )


#
def cell_id(cell: ResolvedCell) -> str:
    """Stable, greppable test ID — e.g. ``F3-2D-M`` or ``F2[fan_out]-B-C``."""

    family_part = (
        cell.family_id
        if cell.variant == "default"
        else f"{cell.family_id}[{cell.variant}]"
    )
    return f"{family_part}-{cell.workflow_id}-{cell.backend}"
