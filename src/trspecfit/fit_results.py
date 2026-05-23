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
from typing import Any, Literal

import numpy as np
import pandas as pd

from trspecfit.utils.fit_io import SavedFile, SavedFitSlot, read_archive

FitType = Literal["baseline", "spectrum", "sbs", "2d"]
SbsAggregation = Literal["median", "mean", "sum", "long"]

# Default columns. Dynamic: which set is used depends on whether any matched
# slot carries a finite ``sigma_data``. ``chi2_red_raw`` is the lmfit-unweighted
# diagnostic (always populated); ``chi2_red`` is the σ-calibrated value
# (≈ 1 for a fit at the noise floor) and is only meaningful when a sigma was
# set on the File at fit time.
DEFAULT_METRICS_NO_SIGMA: tuple[str, ...] = ("chi2_red_raw", "r2", "aic", "bic")
DEFAULT_METRICS_WITH_SIGMA: tuple[str, ...] = (
    "chi2_red_raw",
    "sigma_eff",
    "chi2_red",
    "r2",
    "aic",
    "bic",
)
# Calibrated columns are only available when a sigma was supplied at fit time.
# Explicit requests for these in ``metrics=[...]`` raise a clear KeyError when
# the matched slot set has no sigma, pointing the user at the raw alternative.
_CALIBRATED_KEYS: frozenset[str] = frozenset({"chi2", "chi2_red"})
_CALIBRATED_TO_RAW: dict[str, str] = {"chi2": "chi2_raw", "chi2_red": "chi2_red_raw"}


#
def _to_str_set(arg: str | Sequence[str] | None) -> set[str] | None:
    """Normalize a string-or-sequence filter arg to a set; ``None`` → no filter."""

    if arg is None:
        return None
    if isinstance(arg, str):
        return {arg}
    return set(arg)


#
def _fp_key(fingerprint: dict[str, Any]) -> tuple[Any, ...]:
    """Hashable key for a file fingerprint (mirrors trspecfit._fp_key)."""

    return (
        fingerprint["data_sha256"],
        fingerprint["energy_sha256"],
        fingerprint["time_sha256"],
        tuple(int(x) for x in fingerprint["shape"]),
    )


#
def _resolve_file_arg(file: Any) -> str | None:
    """
    Normalize ``file`` filter input to a name string.

    Accepts ``None`` (no filter), a string, or any object exposing ``.name``
    (covers live ``trspecfit.File`` and ``SavedFile``). Avoids importing
    ``trspecfit.File`` to keep this module a leaf in the import graph.
    """

    if file is None:
        return None
    if isinstance(file, str):
        return file
    if isinstance(file, SavedFile):
        return file.name
    name = getattr(file, "name", None)
    if isinstance(name, str):
        return name
    raise TypeError(
        f"file must be str, SavedFile, or have a .name attribute; "
        f"got {type(file).__name__}"
    )


#
def _has_any_sigma(slots: Sequence[SavedFitSlot]) -> bool:
    """True if at least one slot carries a finite ``sigma_data``."""

    return any(np.isfinite(s.sigma_data) for s in slots)


#
def _resolve_metric_keys(
    metrics: Sequence[str] | None, slots: list[SavedFitSlot]
) -> tuple[str, ...]:
    """
    Pick the metric columns for a ``compare_models()`` call.

    ``metrics=None`` → dynamic defaults: ``DEFAULT_METRICS_WITH_SIGMA`` when at
    least one matched slot has a sigma, ``DEFAULT_METRICS_NO_SIGMA`` otherwise.

    ``metrics=[...]`` → explicit. If the request includes a calibrated metric
    (``chi2`` / ``chi2_red``) and no matched slot has a sigma, raise a clear
    ``KeyError`` pointing the user at ``file.set_sigma()`` or the raw
    alternative — neither silently-NaN columns nor renamed-raw columns.
    """

    has_sigma = _has_any_sigma(slots)
    if metrics is None:
        return DEFAULT_METRICS_WITH_SIGMA if has_sigma else DEFAULT_METRICS_NO_SIGMA
    metric_keys = tuple(metrics)
    if not has_sigma:
        bad = next((k for k in metric_keys if k in _CALIBRATED_KEYS), None)
        if bad is not None:
            raise KeyError(
                f"Metric {bad!r} requires sigma_data, but none of the matched "
                f"slots carry a sigma. Call file.set_sigma(...) on the live "
                f"file and re-run the fit, or request "
                f"{_CALIBRATED_TO_RAW[bad]!r} for the raw (uncalibrated) "
                f"value."
            )
    return metric_keys


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

    #
    def compare_models(
        self,
        file: Any = None,
        *,
        models: Sequence[str] | None = None,
        fit_type: FitType | Sequence[FitType] | None = None,
        metrics: Sequence[str] | None = None,
        sbs_aggregation: SbsAggregation = "median",
    ) -> pd.DataFrame:
        """
        Compare fit-quality metrics across slots.

        Filters slots by ``(file, models, fit_type)``, then returns a
        ``pd.DataFrame`` with one row per slot (or per slice in ``"long"``
        mode) and one column per metric.

        Default column set is **dynamic** based on whether any matched
        slot carries a sigma (set via ``File.set_sigma()`` before the fit):

        - no sigma:  ``chi2_red_raw, r2, aic, bic``
        - with sigma: ``chi2_red_raw, sigma_eff, chi2_red, r2, aic, bic``

        ``chi2_red_raw`` is always present (the lmfit-unweighted diagnostic);
        ``chi2_red`` is the σ-calibrated value (≈ 1 for a fit at the noise
        floor). Names are stable — the same column always carries the same
        kind of value across calls, sessions, and loaded archives. There is
        no per-call ``sigma=`` kwarg by design; persistent state on the File
        is the only sigma source.

        Parameters
        ----------
        file : str | SavedFile | trspecfit.File | None
            Filter to a single file. Accepts a name string, a ``SavedFile``,
            or any object with a ``.name`` attribute (so the live
            ``File.compare_models`` delegate can pass ``self``).
        models : sequence of str, optional
            Restrict to these model names.
        fit_type : str or sequence, optional
            Restrict to these fit types.
        metrics : sequence of str, optional
            Metric keys to include as columns. Defaults to the dynamic set
            above. Valid keys: ``chi2_raw``, ``chi2_red_raw``, ``chi2``,
            ``chi2_red``, ``r2``, ``aic``, ``bic``, ``sigma_eff``. Requesting
            ``chi2`` or ``chi2_red`` when no matched slot has a sigma raises
            ``KeyError`` with a pointer to ``File.set_sigma()`` or the raw
            alternative.
        sbs_aggregation : {"median", "mean", "sum", "long"}, default "median"
            How to collapse per-slice SbS metrics to a comparable value:

            - ``"median"`` — robust scalar via ``np.nanmedian``.
            - ``"mean"``   — scalar via ``np.nanmean``.
            - ``"sum"``    — ``np.nansum`` for additive metrics (``chi2``,
              ``chi2_raw``, ``aic``, ``bic``). ``chi2_red`` and
              ``chi2_red_raw`` instead aggregate as ``Σnumerator / ΣDoF``
              (per-slice DoF recovered from ``chi2_raw / chi2_red_raw``)
              so the canonical "≈ 1 for a good fit" reading is preserved.
              ``r2`` is still nansum'd; treat it as informational in sum
              mode (no per-slice SST is stored to compute an aggregate r²).
            - ``"long"``   — one row per slice. Adds a ``slice_index``
              column (NaN for non-SbS rows). ``sigma_eff`` is broadcast
              from the slot's scalar to every slice row.

        Returns
        -------
        pd.DataFrame
            Columns: ``file``, ``model``, ``fit_type``, ``selection_json``,
            optionally ``slice_index``, then one column per requested
            metric. Empty DataFrame if no slots match the filter.

        Raises
        ------
        ValueError
            If two or more slots in the filtered result share
            ``(file_fingerprint, fit_type)`` but disagree on
            ``observed_sha256``. Same fit type on the same file must run
            against the same observed grid for AIC/BIC comparisons to be
            meaningful — typically this happens when the user mixes refits
            with different ``e_lim`` / ``t_lim`` / ``base_t_ind`` /
            ``time_point``.
        KeyError
            If ``metrics`` requests ``chi2`` / ``chi2_red`` when no matched
            slot has a sigma, or any other unknown metric key for at least
            one slot.
        """

        file_name = _resolve_file_arg(file)
        models_filter = _to_str_set(models)
        types_filter = _to_str_set(fit_type)

        matched: list[SavedFitSlot] = []
        for slot in self._slots:
            if file_name is not None and slot.file_name != file_name:
                continue
            if models_filter is not None and slot.model_name not in models_filter:
                continue
            if types_filter is not None and slot.fit_type not in types_filter:
                continue
            matched.append(slot)

        self._check_observed_consistency(matched)
        metric_keys = _resolve_metric_keys(metrics, matched)

        if sbs_aggregation == "long":
            return self._compare_rows_long(matched, metric_keys)
        return self._compare_rows_scalar(matched, metric_keys, sbs_aggregation)

    #
    @staticmethod
    def _check_observed_consistency(slots: list[SavedFitSlot]) -> None:
        """
        Raise if two slots in the same ``(file_fingerprint, file_name, fit_type)``
        group disagree on ``observed_sha256``.

        Different ``observed`` arrays mean different ndata or different data
        views — AIC/BIC/chi2 across them are not comparable. Catches
        e_lim/t_lim/base_t_ind/time_point mismatches via the data hash even
        when ``selection_json`` would also differ.

        ``file_name`` is part of the grouping key (not just fingerprint)
        because Project identity treats two ``Project.files`` with
        byte-identical raw arrays but different names as distinct files
        (matches ``history_key`` / ``archive_slot_key`` semantics, which
        also fold ``file_name`` in). A project-wide
        ``compare_models(fit_type=...)`` across replicate files would
        otherwise raise a false "different data views" error.
        """

        groups: dict[tuple[Any, str, str], list[SavedFitSlot]] = {}
        for slot in slots:
            key = (_fp_key(slot.file_fingerprint), slot.file_name, slot.fit_type)
            groups.setdefault(key, []).append(slot)
        for (_fp, file_name, ft), group in groups.items():
            shas = {s.observed_sha256 for s in group}
            if len(shas) > 1:
                names = sorted({s.model_name for s in group})
                raise ValueError(
                    f"Cannot compare fit_type={ft!r} on file="
                    f"{file_name!r}: {len(shas)} distinct "
                    f"observed_sha256 across {len(group)} slot(s) "
                    f"(models={names}). Slots fit against different data "
                    f"views — narrow the filter (or restrict on selection "
                    f"via find()) so all compared slots share the same "
                    f"observed grid."
                )

    #
    @staticmethod
    def _aggregate_sbs(values: np.ndarray, mode: SbsAggregation) -> float:
        """
        Collapse a per-slice metric array to a scalar using ``mode``.

        ``"long"`` is handled separately by the caller and is not a valid
        input here.
        """

        arr = np.asarray(values, dtype=float)
        if mode == "median":
            return float(np.nanmedian(arr))
        if mode == "mean":
            return float(np.nanmean(arr))
        if mode == "sum":
            return float(np.nansum(arr))
        raise ValueError(f"unknown sbs_aggregation: {mode!r}")

    #
    @staticmethod
    def _aggregate_sbs_reduced_sum(slot: SavedFitSlot, key: str) -> float:
        """
        Aggregate reduced χ² for sum-mode SbS — handles both raw and σ-calibrated.

        For ``key="chi2_red_raw"``: returns ``Σ chi2_raw / Σ DoF``.
        For ``key="chi2_red"``:     returns ``Σ chi2 / Σ DoF`` (NaN when σ
        was unset, since per-slice ``chi2`` is then NaN).

        Per-slice DoF is recovered from the always-populated raw columns
        (``DoF = chi2_raw / chi2_red_raw``). Treating the SbS result as
        one composite fit with total DoF = Σ DoF_per_slice preserves the
        canonical "good fit ≈ 1" reading. The naive ``np.nansum`` of
        per-slice reduced χ² would otherwise grow linearly with the number
        of slices and break the comparison.

        Returns ``NaN`` if total DoF is non-positive (degenerate fit) or
        the numerator is non-finite.
        """

        chi2_raw_arr = np.asarray(slot.metrics["chi2_raw"], dtype=float)
        chi2_red_raw_arr = np.asarray(slot.metrics["chi2_red_raw"], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            dof_arr = np.where(
                chi2_red_raw_arr != 0,
                chi2_raw_arr / chi2_red_raw_arr,
                np.nan,
            )
        total_dof = float(np.nansum(dof_arr))
        if not (total_dof > 0):
            return float("nan")
        numerator_key = "chi2_raw" if key == "chi2_red_raw" else "chi2"
        numerator_arr = np.asarray(slot.metrics[numerator_key], dtype=float)
        total_num = float(np.nansum(numerator_arr))
        if not np.isfinite(total_num):
            return float("nan")
        return total_num / total_dof

    #
    @staticmethod
    def _slot_metric(slot: SavedFitSlot, key: str) -> Any:
        """Look up ``key`` on the slot, with sigma_eff handled as a special case.

        ``sigma_eff`` lives as a top-level field on ``SavedFitSlot`` (not in
        the metrics dict) because it's noise metadata, not a fit-quality
        metric. Every other key reads from ``slot.metrics`` with a clear
        ``KeyError`` if absent.
        """

        if key == "sigma_eff":
            return float(slot.sigma_eff)
        if key not in slot.metrics:
            raise KeyError(
                f"metric {key!r} not present in slot "
                f"(file={slot.file_name!r}, model={slot.model_name!r}, "
                f"fit_type={slot.fit_type!r}); available: "
                f"{sorted(slot.metrics.keys())} (plus 'sigma_eff')"
            )
        return slot.metrics[key]

    #
    def _compare_rows_scalar(
        self,
        slots: list[SavedFitSlot],
        metric_keys: tuple[str, ...],
        sbs_aggregation: SbsAggregation,
    ) -> pd.DataFrame:
        """One row per slot; SbS per-slice metrics collapsed via ``sbs_aggregation``."""

        rows: list[dict[str, Any]] = []
        for slot in slots:
            row: dict[str, Any] = {
                "file": slot.file_name,
                "model": slot.model_name,
                "fit_type": slot.fit_type,
                "selection_json": slot.selection_json,
            }
            for key in metric_keys:
                if key == "sigma_eff":
                    # Per-slot scalar; SbS doesn't aggregate it (one σ per fit).
                    row[key] = float(slot.sigma_eff)
                    continue
                value = self._slot_metric(slot, key)
                if slot.fit_type == "sbs":
                    if sbs_aggregation == "sum" and key in (
                        "chi2_red",
                        "chi2_red_raw",
                    ):
                        # Treat the SbS fit as one composite fit: aggregate
                        # reduced chi-square = Σnumerator / ΣDoF. The naive
                        # nansum of per-slice reduced χ² would grow linearly
                        # with N_slices and lose the "≈ 1" reading.
                        row[key] = self._aggregate_sbs_reduced_sum(slot, key)
                    else:
                        row[key] = self._aggregate_sbs(value, sbs_aggregation)
                else:
                    row[key] = float(value)
            rows.append(row)
        columns = ["file", "model", "fit_type", "selection_json", *metric_keys]
        return pd.DataFrame(rows, columns=columns)

    #
    def plot_residuals(
        self,
        *,
        file: Any,
        models: Sequence[str] | None = None,
        fit_type: FitType | None = None,
        show_plot: bool = True,
        figsize: tuple[float, float] | None = None,
    ) -> Any:
        """
        Plot observed/fit/residual for the selected slots side-by-side.

        Smoke-test-grade visualization: x-axis is array index (no energy /
        time labels), since slots do not carry the parent file's axes.
        Users wanting publication-quality plots should build their own from
        ``slot.observed`` / ``slot.fit``.

        Parameters
        ----------
        file : str | SavedFile | trspecfit.File
            Required. Filter to a single file.
        models : sequence of str, optional
            Which models to compare. ``None`` plots every model that fit
            this file.
        fit_type : str, optional
            Required if the matched slots span more than one fit type.
        show_plot : bool, default True
            Set ``False`` in tests to close the figure without displaying.
        figsize : (w, h), optional
            Forwarded to ``plt.subplots``. Defaults scale with the number
            of compared models.

        Returns
        -------
        matplotlib.figure.Figure

        Raises
        ------
        LookupError
            If no slots match the filter.
        ValueError
            If matched slots span more than one ``fit_type`` and
            ``fit_type`` was not given.
        """

        file_name = _resolve_file_arg(file)
        if file_name is None:
            raise ValueError("plot_residuals requires file=...")

        models_filter = _to_str_set(models)
        matched: list[SavedFitSlot] = []
        for slot in self._slots:
            if slot.file_name != file_name:
                continue
            if models_filter is not None and slot.model_name not in models_filter:
                continue
            if fit_type is not None and slot.fit_type != fit_type:
                continue
            matched.append(slot)

        if not matched:
            raise LookupError(
                f"No slots match file={file_name!r}, models={models}, "
                f"fit_type={fit_type!r}."
            )

        fit_types = {s.fit_type for s in matched}
        if len(fit_types) > 1:
            raise ValueError(
                f"Matched slots span fit_types={sorted(fit_types)}; "
                f"pass fit_type=... to disambiguate."
            )
        ft = next(iter(fit_types))

        if ft in ("baseline", "spectrum"):
            return self._plot_residuals_1d(
                matched, file_name, show_plot=show_plot, figsize=figsize
            )
        return self._plot_residuals_2d(
            matched, file_name, show_plot=show_plot, figsize=figsize
        )

    #
    @staticmethod
    def _plot_residuals_1d(
        slots: list[SavedFitSlot],
        file_name: str,
        *,
        show_plot: bool,
        figsize: tuple[float, float] | None,
    ) -> Any:
        """1D fits: top row = observed + fit; bottom row = residual."""

        import matplotlib.pyplot as plt

        n = len(slots)
        fig, axs = plt.subplots(
            2,
            n,
            figsize=figsize or (4.0 * max(n, 1), 5.0),
            squeeze=False,
            sharex="col",
        )
        for col, slot in enumerate(slots):
            obs = np.asarray(slot.observed).ravel()
            fit = np.asarray(slot.fit).ravel()
            x = np.arange(obs.size)
            axs[0, col].plot(x, obs, "k.", ms=3, label="observed")
            axs[0, col].plot(x, fit, "-", lw=1.5, label="fit")
            axs[0, col].set_title(f"{slot.model_name} ({slot.fit_type})")
            axs[0, col].legend(fontsize="small")
            axs[1, col].plot(x, obs - fit, "-", lw=1.0)
            axs[1, col].axhline(0, color="gray", lw=0.5)
            axs[1, col].set_xlabel("index")
            if col == 0:
                axs[0, col].set_ylabel("intensity")
                axs[1, col].set_ylabel("residual")
        fig.suptitle(f"Residuals — {file_name}")
        fig.tight_layout()
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        return fig

    #
    @staticmethod
    def _plot_residuals_2d(
        slots: list[SavedFitSlot],
        file_name: str,
        *,
        show_plot: bool,
        figsize: tuple[float, float] | None,
    ) -> Any:
        """SbS / 2D fits: residual heatmaps side-by-side, shared diverging scale."""

        import matplotlib.pyplot as plt

        residuals = [np.asarray(slot.observed) - np.asarray(slot.fit) for slot in slots]
        global_max = 0.0
        for res in residuals:
            if res.size:
                local = float(np.nanmax(np.abs(res)))
                if local > global_max:
                    global_max = local
        if global_max == 0.0:
            global_max = 1.0

        n = len(slots)
        fig, axs = plt.subplots(
            1,
            n,
            figsize=figsize or (5.0 * max(n, 1), 4.0),
            squeeze=False,
        )
        im = None
        for col, (slot, res) in enumerate(zip(slots, residuals, strict=True)):
            im = axs[0, col].imshow(
                res,
                aspect="auto",
                cmap="RdBu_r",
                vmin=-global_max,
                vmax=global_max,
                origin="lower",
            )
            axs[0, col].set_title(f"{slot.model_name} ({slot.fit_type})")
            axs[0, col].set_xlabel("energy index")
            if col == 0:
                axs[0, col].set_ylabel("time / slice index")
        if im is not None:
            fig.colorbar(im, ax=axs[0, :].tolist(), shrink=0.85)
        fig.suptitle(f"Residuals — {file_name}")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        return fig

    #
    def _compare_rows_long(
        self,
        slots: list[SavedFitSlot],
        metric_keys: tuple[str, ...],
    ) -> pd.DataFrame:
        """
        One row per slice for SbS slots; one row total for non-SbS slots.

        Adds a ``slice_index`` column. Non-SbS rows get ``slice_index = pd.NA``;
        SbS rows enumerate slice indices. ``sigma_eff`` is a per-fit scalar
        and is broadcast to every slice row of an SbS slot.
        """

        rows: list[dict[str, Any]] = []
        for slot in slots:
            base: dict[str, Any] = {
                "file": slot.file_name,
                "model": slot.model_name,
                "fit_type": slot.fit_type,
                "selection_json": slot.selection_json,
            }
            if slot.fit_type == "sbs":
                # Use any non-sigma_eff key to determine n_slices (sigma_eff
                # is a scalar). Fall back to the first array metric stored.
                size_probe = next(
                    (k for k in metric_keys if k != "sigma_eff" and k in slot.metrics),
                    None,
                )
                if size_probe is None:
                    # All requested keys are sigma_eff or missing; treat the
                    # SbS slot as a single row using slot.fit's row count.
                    n_slices = int(np.asarray(slot.fit).shape[0])
                else:
                    n_slices = int(np.asarray(slot.metrics[size_probe]).size)
                for i in range(n_slices):
                    row = {**base, "slice_index": i}
                    for key in metric_keys:
                        if key == "sigma_eff":
                            row[key] = float(slot.sigma_eff)
                            continue
                        arr = np.asarray(self._slot_metric(slot, key))
                        row[key] = float(arr[i])
                    rows.append(row)
            else:
                row = {**base, "slice_index": pd.NA}
                for key in metric_keys:
                    if key == "sigma_eff":
                        row[key] = float(slot.sigma_eff)
                        continue
                    row[key] = float(self._slot_metric(slot, key))
                rows.append(row)
        columns = [
            "file",
            "model",
            "fit_type",
            "selection_json",
            "slice_index",
            *metric_keys,
        ]
        return pd.DataFrame(rows, columns=columns)
