"""
Normalize .ipynb cell sources to Jupyter's canonical list-of-strings shape.

Runs as a pre-commit hook (see .pre-commit-config.yaml). Some tools — notably
Claude Code's NotebookEdit — write inserted/replaced cell sources as a single
JSON string with embedded ``\\n``, while Jupyter's canonical on-disk format is
a list with one element per line. The mixed representation makes ``git diff``
render touched cells as a single blob change rather than line-by-line, which
compounds across many small notebook edits.

This script renormalizes any string-sourced cells to the canonical shape and
relies on ``nbformat.write`` for the str → list-of-strings split (it writes
``"a\\nb\\n"`` as ``["a\\n", "b\\n"]`` and ``""`` as ``[]`` automatically), plus
the trailing-newline + key-order canonical formatting.

A fast JSON-level pre-check decides whether the notebook needs any work at
all: ``nbformat.read`` collapses every cell source — list **or** string on
disk — to an in-memory Python ``str``, so we cannot tell from the parsed
object which cells were already canonical. We inspect the raw JSON instead
and skip the nbformat round-trip entirely when no cell sources are stored as
strings. That keeps mtime stable on already-canonical notebooks and avoids
churning the working tree on repeat pre-commit runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import nbformat


#
def normalize(path: Path) -> bool:
    """Normalize one notebook in place; return True if anything changed."""

    # Pre-check at the JSON layer. nbformat.read would mask the on-disk
    # shape by collapsing every cell source to ``str``, so we can't decide
    # from the parsed object whether the file needs work. Read the raw JSON
    # cheaply and short-circuit when nothing is non-canonical.
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    if not any(isinstance(c.get("source"), str) for c in raw.get("cells", [])):
        return False

    # At least one cell is stored as a JSON string. Round-trip through
    # nbformat — its write path is what does the canonical splitting (incl.
    # empty source -> ``[]``, not ``[""]``) plus trailing newline and key
    # ordering. No per-cell munging needed here.
    nb = nbformat.read(path, as_version=4)
    nbformat.write(nb, path)
    return True


#
def main(argv: list[str]) -> int:
    for arg in argv:
        normalize(Path(arg))
    # Exit 0 either way; pre-commit detects modified files on its own and
    # fails the commit accordingly (same pattern as nbstripout / ruff format).
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
