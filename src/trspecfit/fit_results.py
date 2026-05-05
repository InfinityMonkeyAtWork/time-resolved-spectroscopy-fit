"""
``FitResults`` — inspection / comparison artifact for completed fits.

Two construction paths:

1. **Loaded from disk** — ``FitResults.load(path)`` deserializes an HDF5
   fit archive (see ``docs/design/fit_archive_schema.md``).
2. **In-memory view** — ``Project.results`` property wraps
   ``Project._fit_history``.

A ``FitResults`` is **immutable after construction**: its slot list is frozen
at the moment of construction. ``Project.results`` returns a fresh wrapper per
access (``FitResults(slots=list(self._fit_history))``); subsequent fits append
to ``_fit_history`` and do **not** affect previously-returned ``FitResults``.

Identity is internally keyed by file fingerprint (multi-sha) + model name +
fit_type + selection_json. Name-based query inputs (``file=...``,
``model=...``) resolve to fingerprint at lookup time.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from os import PathLike
from typing import Literal

from trspecfit.utils.fit_io import SavedFitSlot, read_archive

FitType = Literal["baseline", "spectrum", "sbs", "2d"]


#
def _to_str_set(arg: str | Sequence[str] | None) -> set[str] | None:
    """Normalize a string-or-sequence filter arg to a set; ``None`` → no filter."""

    if arg is None:
        return None
    if isinstance(arg, str):
        return {arg}
    return set(arg)


#
class FitResults:
    """
    Immutable view over a list of ``SavedFitSlot``.

    Construction is positional-only (``FitResults(slots=...)``); users normally
    obtain instances via ``Project.results`` or ``FitResults.load(path)``.
    """

    #
    def __init__(self, *, slots: list[SavedFitSlot]) -> None:
        self._slots: tuple[SavedFitSlot, ...] = tuple(slots)

    #
    @classmethod
    def load(
        cls,
        filepath: PathLike | str,
        *,
        file: str | Sequence[str] | None = None,
        model: str | Sequence[str] | None = None,
        fit_type: FitType | Sequence[FitType] | None = None,
    ) -> FitResults:
        """
        Load a fit archive from disk and wrap its slots in a ``FitResults``.

        Calls ``utils.fit_io.read_archive`` and flattens the returned
        ``SavedProject.files[*].slots`` into a single sequence in archive
        (file, then slot) order. Independent of any live ``Project``: the
        returned object is a snapshot of the archive at read time.

        Optional ``file`` / ``model`` / ``fit_type`` arguments accept either
        a single string or a list of strings; only matching slots are
        kept. Filters are AND-combined and operate on the slot's display
        fields (``file_name``, ``model_name``, ``fit_type``).
        """

        project = read_archive(filepath)
        files_filter = _to_str_set(file)
        models_filter = _to_str_set(model)
        types_filter = _to_str_set(fit_type)
        slots: list[SavedFitSlot] = []
        for sf in project.files:
            if files_filter is not None and sf.name not in files_filter:
                continue
            for slot in sf.slots:
                if models_filter is not None and slot.model_name not in models_filter:
                    continue
                if types_filter is not None and slot.fit_type not in types_filter:
                    continue
                slots.append(slot)
        return cls(slots=slots)

    #
    def __iter__(self) -> Iterator[SavedFitSlot]:
        return iter(self._slots)

    #
    def __len__(self) -> int:
        return len(self._slots)

    #
    def __repr__(self) -> str:
        n = len(self._slots)
        files = self.files()
        return (
            f"FitResults({n} slot{'s' if n != 1 else ''}, "
            f"{len(files)} file{'s' if len(files) != 1 else ''})"
        )

    #
    def files(self) -> list[str]:
        """
        List unique file names across slots (insertion order).

        Names are display strings (``SavedFitSlot.file_name``); identity is
        fingerprint-based internally.
        """

        seen: dict[str, None] = {}
        for slot in self._slots:
            seen.setdefault(slot.file_name, None)
        return list(seen.keys())

    #
    def models(self, *, file: str | None = None) -> list[str]:
        """
        List unique model names. If ``file`` is given, restrict to that file.
        """

        seen: dict[str, None] = {}
        for slot in self._slots:
            if file is not None and slot.file_name != file:
                continue
            seen.setdefault(slot.model_name, None)
        return list(seen.keys())

    #
    def find(
        self,
        *,
        file: str | None = None,
        model: str | None = None,
        fit_type: FitType | None = None,
    ) -> list[SavedFitSlot]:
        """
        Return all slots matching the given filters (AND-combined).

        Filters operate on display fields (``file_name``, ``model_name``,
        ``fit_type``). Returns slots in history order (oldest first).
        """

        out: list[SavedFitSlot] = []
        for slot in self._slots:
            if file is not None and slot.file_name != file:
                continue
            if model is not None and slot.model_name != model:
                continue
            if fit_type is not None and slot.fit_type != fit_type:
                continue
            out.append(slot)
        return out

    #
    def get(
        self,
        *,
        file: str,
        model: str,
        fit_type: FitType,
    ) -> SavedFitSlot:
        """
        Return the unique slot matching ``(file, model, fit_type)``.

        Raises ``LookupError`` if 0 or >1 slots match. For multi-match
        scenarios (e.g. refits with different selections), use ``find`` and
        narrow further on ``slot.selection``.
        """

        matches = self.find(file=file, model=model, fit_type=fit_type)
        if not matches:
            raise LookupError(
                f"No slot matches file={file!r}, model={model!r}, "
                f"fit_type={fit_type!r}."
            )
        if len(matches) > 1:
            raise LookupError(
                f"{len(matches)} slots match file={file!r}, model={model!r}, "
                f"fit_type={fit_type!r}; use find() and narrow on .selection."
            )
        return matches[0]
