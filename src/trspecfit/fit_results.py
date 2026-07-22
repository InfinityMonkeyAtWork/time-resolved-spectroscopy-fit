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
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from trspecfit.config.plot import PlotConfig
from trspecfit.utils.arrays import resolve_time_selection
from trspecfit.utils.fit_io import SavedFile, SavedFitSlot, read_archive
from trspecfit.utils.lmfit import MCMCResult

FitType = Literal["baseline", "spectrum", "sbs", "2d"]
SbsAggregation = Literal["median", "mean", "sum", "long"]

# Fit entry point per fit_type — used in "not fit yet" error messages.
_FIT_METHOD_BY_TYPE: dict[str, str] = {
    "baseline": "fit_baseline",
    "spectrum": "fit_spectrum",
    "sbs": "fit_slice_by_slice",
    "2d": "fit_2d",
}

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
def _slot_title(slot: SavedFitSlot) -> str:
    """Plot title from persisted slot metadata: file, model, yaml stem, time."""

    title = f'{slot.file_name} - "{slot.model_name}" ({slot.fit_type})'
    if slot.yaml_filename:
        title += f" [{slot.yaml_filename}]"
    if slot.fit_type == "spectrum":
        time_point = slot.selection.get("time_point")
        time_range = slot.selection.get("time_range")
        time_type = slot.selection.get("time_type", "abs")
        if time_point is not None:
            title += f", t={time_point} ({time_type})"
        elif time_range is not None:
            title += f", t in {time_range} ({time_type})"
    return title


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

    Construction is keyword-only (``FitResults(slots=...)``); users normally
    obtain instances via ``Project.results`` or ``FitResults.load(path)``.

    ``files`` optionally supplies per-file axes / plot-config providers —
    ``SavedFile`` records (load path) or live ``trspecfit.File`` objects
    (``Project.results``), matched to slots by fingerprint. The plot
    methods use them to label real energy/time axes; without a provider
    they fall back to array-index axes.
    """

    #
    def __init__(
        self,
        *,
        slots: list[SavedFitSlot],
        files: Sequence[Any] | None = None,
    ) -> None:
        self._slots: tuple[SavedFitSlot, ...] = tuple(slots)
        self._files_by_fp: dict[tuple[Any, ...], Any] = {}
        for f in files or ():
            fp = self._provider_fp_key(f)
            if fp is not None:
                self._files_by_fp[fp] = f

    #
    @staticmethod
    def _provider_fp_key(f: Any) -> tuple[Any, ...] | None:
        """
        Fingerprint key for an axes provider, or ``None`` if unavailable.

        ``SavedFile.fingerprint`` is a dict attribute; the live
        ``trspecfit.File.fingerprint`` is a method that raises when the
        file has no data — such files produced no slots, so skipping them
        is safe.
        """

        fingerprint = getattr(f, "fingerprint", None)
        if callable(fingerprint):
            try:
                fingerprint = fingerprint()
            except ValueError:
                return None
        if isinstance(fingerprint, dict):
            return _fp_key(fingerprint)
        return None

    #
    def _provider_for(self, slot: SavedFitSlot) -> Any | None:
        """Axes/data provider (``SavedFile`` or live ``File``) for this slot's file."""

        return self._files_by_fp.get(_fp_key(slot.file_fingerprint))

    #
    def _axes_for(
        self, slot: SavedFitSlot, *, full_range: bool = False
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        ``(energy, time)`` on the slot's grid, or ``None`` where unknown.

        Crops the provider's full axes to the slot's selection (same rule
        as ``fit_io._slot_axes``, tolerant of missing providers/axes).
        Pass ``full_range=True`` to skip the crop and return the
        provider's full, uncropped axes instead.
        """

        provider = self._provider_for(slot)
        energy = getattr(provider, "energy", None) if provider is not None else None
        time = getattr(provider, "time", None) if provider is not None else None
        if energy is not None:
            energy = np.asarray(energy)
            if not full_range:
                e_lim = slot.selection.get("e_lim")
                if e_lim:
                    energy = energy[int(e_lim[0]) : int(e_lim[1])]
        if time is not None:
            time = np.asarray(time)
            if time.ndim == 0 or time.size == 0:
                time = None
            elif not full_range and slot.fit_type == "2d":
                t_lim = slot.selection.get("t_lim")
                if t_lim:
                    time = time[int(t_lim[0]) : int(t_lim[1])]
        return energy, time

    #
    def _full_observed_for(
        self, slot: SavedFitSlot, provider: Any
    ) -> np.ndarray | None:
        """
        Real, full-range observed array for ``slot``, re-derived from the
        provider's full ``data`` — or ``None`` if reconstruction isn't
        possible (missing provider/selection fields; caller falls back to
        the cropped ``slot.observed``).

        ``sbs``/``2d`` need no re-derivation — the provider's full ``data``
        already matches (sbs never crops time; 2d's slot is a direct
        sub-slice of it). ``baseline``/``spectrum`` re-derive the
        time-reduced spectrum (average or single row) from the persisted
        selection, mirroring ``File.fit_baseline``/``File.fit_spectrum``.
        """

        data = getattr(provider, "data", None)
        if data is None:
            return None
        data = np.asarray(data)

        if slot.fit_type in ("sbs", "2d"):
            return data

        if slot.fit_type == "baseline":
            base_t_ind = slot.selection.get("base_t_ind")
            if not base_t_ind:
                return None
            return np.asarray(
                np.mean(data[int(base_t_ind[0]) : int(base_t_ind[1]), :], axis=0)
            )

        if slot.fit_type == "spectrum":
            time = getattr(provider, "time", None)
            if time is None:
                return None
            time = np.asarray(time)
            time_point = slot.selection.get("time_point")
            time_range = slot.selection.get("time_range")
            time_type = slot.selection.get("time_type", "abs")
            try:
                if time_point is not None:
                    ind = resolve_time_selection(
                        time, time_point, time_point, time_type=time_type
                    )
                    return np.asarray(data[ind[0], :])
                if time_range is not None:
                    ind = resolve_time_selection(
                        time, time_range[0], time_range[1], time_type=time_type
                    )
                    return np.asarray(np.mean(data[ind[0] : ind[1], :], axis=0))
            except ValueError:
                return None
            return None

        return None

    #
    @staticmethod
    def _pad_axis(
        cropped: np.ndarray, full_len: int, lim: list[int] | None, *, axis: int
    ) -> np.ndarray:
        """
        Return ``cropped`` embedded in a ``NaN``-filled array of length
        ``full_len`` along ``axis``, at the ``lim`` (``[start, stop)``)
        window. ``lim is None`` means that axis is already full range
        (the fit ran on the whole axis) — ``cropped`` is returned as-is.

        Used to place a slot's cropped ``fit``/``components`` into the
        full-range grid without fabricating values outside the fit
        window — the padding is ``NaN``, never a placeholder.
        """

        if lim is None:
            return cropped
        shape = list(cropped.shape)
        shape[axis] = full_len
        full = np.full(tuple(shape), np.nan, dtype=np.result_type(cropped, float))
        idx: list[slice] = [slice(None)] * cropped.ndim
        idx[axis] = slice(int(lim[0]), int(lim[1]))
        full[tuple(idx)] = cropped
        return full

    #
    def _config_for(self, slot: SavedFitSlot, config: Any) -> Any:
        """
        Resolve the ``PlotConfig`` for a plot call.

        Explicit ``config=`` wins; otherwise the live file's
        ``plot_config`` when this ``FitResults`` was built from a Project;
        default ``PlotConfig()`` for loaded archives (styling is
        deliberately not persisted).
        """

        if config is not None:
            return config
        provider = self._provider_for(slot)
        live_config = getattr(provider, "plot_config", None)
        return live_config if live_config is not None else PlotConfig()

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
        # SavedFiles are kept (unfiltered) as axes providers for the plot
        # methods; slots are matched to them by fingerprint at plot time.
        return cls(slots=slots, files=list(project.files))

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
    def _latest_slot(
        self,
        *,
        file: Any,
        model: str | None,
        fit_type: str,
    ) -> SavedFitSlot:
        """
        Return the most recent slot matching the filters.

        ``find()`` preserves history order, so the last match is the latest
        fit — mirroring the live-model convention where each ``fit_*`` call
        overwrites the previous result of that type. Raises ``ValueError``
        (not ``LookupError``) with a "run fit_x() first" hint, matching the
        long-standing accessor contract on ``File``.
        """

        if fit_type not in _FIT_METHOD_BY_TYPE:
            raise ValueError(
                f"Unknown fit_type={fit_type!r}; "
                "use 'baseline', 'spectrum', 'sbs', or '2d'."
            )
        file_name = _resolve_file_arg(file)
        matches = self.find(
            file=file_name, model=model, fit_type=cast(FitType, fit_type)
        )
        if not matches:
            parts = [
                f"{k}={v!r}"
                for k, v in (("file", file_name), ("model", model))
                if v is not None
            ]
            detail = f" ({', '.join(parts)})" if parts else ""
            raise ValueError(
                f"No {fit_type} fit results{detail}. "
                f"Run {_FIT_METHOD_BY_TYPE[fit_type]}() first."
            )
        return matches[-1]

    #
    def get_fit_results(
        self,
        *,
        file: Any = None,
        model: str | None = None,
        fit_type: FitType = "baseline",
    ) -> pd.DataFrame:
        """
        Return the fitted parameters of the latest matching fit.

        Reads the persisted slot (``SavedFitSlot.params``) — works identically
        on ``Project.results`` and on archives loaded via
        :meth:`FitResults.load`.

        Parameters
        ----------
        file : str | SavedFile | trspecfit.File | None
            Filter to a single file (name string or object with ``.name``).
        model : str, optional
            Filter to a single model name.
        fit_type : {'baseline', 'spectrum', 'sbs', '2d'}, default='baseline'
            Which fit type to read. When several slots match, the most
            recent fit wins.

        Returns
        -------
        pd.DataFrame
            For 'baseline', 'spectrum', and '2d': one row per parameter with
            columns ``['name', 'value', 'stderr', 'init_value', 'min', 'max',
            'vary', 'expr']``.
            For 'sbs': one row per time slice with columns = parameter names.

        Raises
        ------
        ValueError
            If no matching fit has been performed (or loaded) yet.
        """

        slot = self._latest_slot(file=file, model=model, fit_type=fit_type)
        return slot.params.copy()

    #
    def get_correlations(
        self,
        *,
        file: Any = None,
        model: str | None = None,
        fit_type: FitType = "baseline",
    ) -> pd.DataFrame:
        """
        Return the parameter correlation matrix of the latest matching fit.

        Reads the persisted slot (``SavedFitSlot.correl``). For SbS fits the
        matrix is slice 0's (the representative slice, like ``conf_ci`` and
        ``mcmc``).

        Parameters
        ----------
        file : str | SavedFile | trspecfit.File | None
            Filter to a single file (name string or object with ``.name``).
        model : str, optional
            Filter to a single model name.
        fit_type : {'baseline', 'spectrum', 'sbs', '2d'}, default='baseline'
            Which fit type to read (latest matching fit wins).

        Returns
        -------
        pd.DataFrame
            Square matrix indexed by the varying parameter names: 1.0 on the
            diagonal, lmfit's pairwise correlations off-diagonal.

        Raises
        ------
        ValueError
            If no matching fit exists, or the fit produced no covariance
            (e.g. Nelder without numdifftools installed, or a project-level
            joint fit).
        """

        slot = self._latest_slot(file=file, model=model, fit_type=fit_type)
        if slot.correl is None:
            raise ValueError(
                f"No correlation matrix for the {fit_type} fit: the optimizer "
                "reported no covariance. Use a covariance-producing method "
                "(e.g. leastsq, or Nelder with numdifftools installed); "
                "project-level joint fits do not decompose per file."
            )
        return slot.correl.copy()

    #
    def get_conf_intervals(
        self,
        *,
        file: Any = None,
        model: str | None = None,
        fit_type: FitType = "baseline",
    ) -> pd.DataFrame:
        """
        Return the profiled confidence-interval table of the latest matching
        fit.

        Reads the persisted slot (``SavedFitSlot.conf_ci``). Populated only
        when the fit ran with ``try_ci=1`` (otherwise an empty DataFrame).
        For SbS fits the table is slice 0's.

        Parameters
        ----------
        file : str | SavedFile | trspecfit.File | None
            Filter to a single file (name string or object with ``.name``).
        model : str, optional
            Filter to a single model name.
        fit_type : {'baseline', 'spectrum', 'sbs', '2d'}, default='baseline'
            Which fit type to read (latest matching fit wins).

        Returns
        -------
        pd.DataFrame
            The conf_interval table, or an empty DataFrame if ``try_ci`` was
            off.

        Raises
        ------
        ValueError
            If no matching fit has been performed (or loaded) yet.
        """

        slot = self._latest_slot(file=file, model=model, fit_type=fit_type)
        if slot.conf_ci is None:
            return pd.DataFrame()
        return slot.conf_ci.copy()

    #
    def get_mcmc(
        self,
        *,
        file: Any = None,
        model: str | None = None,
        fit_type: FitType = "baseline",
    ) -> MCMCResult:
        """
        Return the MCMC outputs (quantile table, chain, acceptance) of the
        latest matching fit.

        Reads the persisted slot (``SavedFitSlot.mcmc``). Available only when
        the fit ran with ``mc_settings`` enabling MCMC. For SbS fits the
        payload is slice 0's.

        Parameters
        ----------
        file : str | SavedFile | trspecfit.File | None
            Filter to a single file (name string or object with ``.name``).
        model : str, optional
            Filter to a single model name.
        fit_type : {'baseline', 'spectrum', 'sbs', '2d'}, default='baseline'
            Which fit type to read (latest matching fit wins).

        Returns
        -------
        MCMCResult
            Bundle of ``table`` (posterior quantiles), ``flatchain``, and
            ``acceptance_fraction`` (``None`` for slots loaded from schema-2
            archives, which did not store it).

        Raises
        ------
        ValueError
            If no matching fit exists, or the fit had no MCMC step.
        """

        slot = self._latest_slot(file=file, model=model, fit_type=fit_type)
        if slot.mcmc is None:
            raise ValueError(
                f"No MCMC results for the {fit_type} fit. Re-run with "
                "mc_settings=MC(use_mc=1, ...)."
            )
        flatchain = slot.mcmc.get("flatchain")
        ci = slot.mcmc.get("ci")
        acceptance = slot.mcmc.get("acceptance_fraction")
        return MCMCResult(
            table=ci.copy() if ci is not None else pd.DataFrame(),
            flatchain=flatchain.copy() if flatchain is not None else pd.DataFrame(),
            acceptance_fraction=(
                np.asarray(acceptance).copy() if acceptance is not None else None
            ),
        )

    #
    def plot_fit(
        self,
        *,
        file: Any = None,
        model: str | None = None,
        fit_type: FitType = "baseline",
        config: Any = None,
        show_plot: bool = True,
        full_range: bool | None = None,
        show_init: bool | None = None,
    ) -> None:
        """
        Plot the latest matching fit: observed, fit, and residual.

        Reads the persisted slot. 1D fits (baseline / spectrum) render an
        observed+fit panel over a residual panel vs energy; 2D fits and SbS
        render the data/fit/residual maps via ``fitlib.plt_fit_res_2d``.
        Real axes are used when this ``FitResults`` carries an axes
        provider for the slot's file (always the case for
        ``Project.results`` and ``FitResults.load``); otherwise array
        indices.

        Parameters
        ----------
        file : str | SavedFile | trspecfit.File | None
            Filter to a single file (name string or object with ``.name``).
        model : str, optional
            Filter to a single model name.
        fit_type : {'baseline', 'spectrum', 'sbs', '2d'}, default='baseline'
            Which fit type to plot (latest matching fit wins).
        config : PlotConfig, optional
            Styling override. Default: the live file's ``plot_config``
            when available, else ``PlotConfig()``.
        show_plot : bool, default True
            Set ``False`` to build without displaying (tests / batch use).
        full_range : bool, optional
            Show the full, uncropped data range (re-derived from the
            file's persisted raw data) instead of just the fit-limits
            window. Fit/residual/components are drawn only inside the
            fit window (``NaN`` outside — never a fabricated value),
            with dashed lines marking the window boundary, matching
            ``describe_model``'s pre-fit view. Falls back to the
            cropped, fit-limits-only view when reconstruction isn't
            possible (e.g. no axes provider for this slot's file).
            Default: ``config.full_range`` (``PlotConfig`` field,
            itself ``Project.full_range``-backed; ``True`` out of the
            box) — pass explicitly to override for one call.
        show_init : bool, optional
            Draw the dotted-gold initial-guess overlay (1D fit types
            only — baseline/spectrum) when the slot has a persisted
            ``fit_ini``. Default: ``config.show_init`` (``PlotConfig``
            field, itself ``Project.show_init``-backed; ``True`` out of
            the box) — pass explicitly to override for one call.

        Raises
        ------
        ValueError
            If no matching fit has been performed (or loaded) yet.
        """

        slot = self._latest_slot(file=file, model=model, fit_type=fit_type)
        cfg = self._config_for(slot, config)
        if full_range is None:
            full_range = bool(getattr(cfg, "full_range", False))
        if show_init is None:
            show_init = bool(getattr(cfg, "show_init", False))
        observed = np.asarray(slot.observed)
        fit = np.asarray(slot.fit)
        components = slot.components
        fit_ini = slot.fit_ini
        e_lim: list[int] | None = None
        t_lim: list[int] | None = None

        if full_range:
            provider = self._provider_for(slot)
            full_observed = (
                self._full_observed_for(slot, provider)
                if provider is not None
                else None
            )
            energy, time = self._axes_for(slot, full_range=True)
            if full_observed is not None and energy is not None:
                n_e_full = energy.shape[0]
                e_lim = slot.selection.get("e_lim")
                if slot.fit_type == "2d":
                    t_lim = slot.selection.get("t_lim")
                    n_t_full = time.shape[0] if time is not None else fit.shape[0]
                    fit = self._pad_axis(fit, n_t_full, t_lim, axis=0)
                    fit = self._pad_axis(fit, n_e_full, e_lim, axis=1)
                elif slot.fit_type == "sbs":
                    fit = self._pad_axis(fit, n_e_full, e_lim, axis=1)
                else:  # baseline / spectrum
                    fit = self._pad_axis(fit, n_e_full, e_lim, axis=0)
                    if fit_ini is not None:
                        fit_ini = self._pad_axis(fit_ini, n_e_full, e_lim, axis=0)
                if components is not None:
                    comp_axis = 2 if slot.fit_type == "sbs" else 1
                    components = self._pad_axis(
                        components, n_e_full, e_lim, axis=comp_axis
                    )
                observed = full_observed
            else:
                energy, time = self._axes_for(slot)
        else:
            energy, time = self._axes_for(slot)

        if slot.fit_type in ("2d", "sbs"):
            from trspecfit import fitlib

            fitlib.plt_fit_res_2d(
                data=observed,
                fit=fit,
                x=energy,
                y=time,
                x_lim=e_lim,
                y_lim=t_lim,
                config=cfg,
                save_img=0 if show_plot else -2,
                title=_slot_title(slot),
            )
            return
        self._plot_fit_1d(
            slot,
            energy=energy,
            config=cfg,
            show_plot=show_plot,
            observed=observed,
            fit=fit,
            components=components,
            roi=e_lim,
            fit_ini=fit_ini,
            show_init=show_init,
        )

    #
    @staticmethod
    def _plot_fit_1d(
        slot: SavedFitSlot,
        *,
        energy: np.ndarray | None,
        config: Any,
        show_plot: bool,
        observed: np.ndarray | None = None,
        fit: np.ndarray | None = None,
        components: np.ndarray | None = None,
        roi: list[int] | None = None,
        fit_ini: np.ndarray | None = None,
        show_init: bool = True,
    ) -> Any:
        """
        Observed + fit (with components, when persisted) over a residual panel.

        ``observed``/``fit``/``components`` default to the slot's own
        (cropped) arrays; pass overrides to render a full-range
        reconstruction instead (see ``FitResults.plot_fit``'s
        ``full_range``). ``roi`` draws dashed boundary lines at the given
        ``[start, stop)`` index window (full-range mode only). ``fit_ini``
        (default: the slot's own, when ``show_init`` and persisted) draws
        the dotted-gold initial-guess overlay.
        """

        import matplotlib.pyplot as plt

        obs = np.asarray(observed if observed is not None else slot.observed).ravel()
        fit_arr = np.asarray(fit if fit is not None else slot.fit).ravel()
        comps = components if components is not None else slot.components
        ini = fit_ini if fit_ini is not None else slot.fit_ini
        if energy is not None and energy.size == obs.size:
            x = energy
            x_label = getattr(config, "x_label", "energy")
        else:
            x = np.arange(obs.size)
            x_label = "index"
        fig, (ax_fit, ax_res) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(6.0, 5.0),
            height_ratios=[3, 1],
        )
        ax_fit.plot(x, obs, "k.", ms=3, label="observed")
        if show_init and ini is not None:
            # NaN entries (full-range mode, outside the fit window) leave a
            # gap rather than a fabricated value, matching fit/components.
            ax_fit.plot(
                x,
                np.asarray(ini).ravel(),
                color="#FFD700",
                linestyle=":",
                linewidth=2,
                label="initial guess",
            )
        if comps is not None:
            # schema >= 4: render the persisted per-component decomposition,
            # matching fitlib.plt_fit_res_1d's live visual style. NaN
            # entries (full-range mode, outside the fit window) leave a
            # gap rather than a fabricated value.
            colors = list(
                plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])
            )
            names = slot.component_names or [
                f"component {i}" for i in range(comps.shape[0])
            ]
            for p, (peak, name) in enumerate(zip(comps, names, strict=True)):
                color = colors[p % len(colors)]
                ax_fit.plot(
                    x, peak, color=color, linestyle="-", linewidth=2, label=name
                )
                ax_fit.fill_between(x, 0, peak, facecolor=color, alpha=0.5)
            ax_fit.plot(x, fit_arr, "-", lw=1.5, color="#000000", label="fit")
        else:
            ax_fit.plot(x, fit_arr, "-", lw=1.5, label="fit")
        ax_fit.set_ylabel("intensity")
        ax_fit.legend(fontsize="small")
        ax_fit.set_title(_slot_title(slot))
        ax_res.plot(x, obs - fit_arr, "-", lw=1.0)
        ax_res.axhline(0, color="gray", lw=0.5)
        ax_res.set_xlabel(x_label)
        ax_res.set_ylabel("residual")
        if roi is not None and len(roi) == 2 and x.size:
            x_start = x[roi[0]]
            x_end = x[roi[1] - 1] if roi[1] > 0 else x[-1]
            for ax in (ax_fit, ax_res):
                ax.axvline(x_start, color="#A9A9A9", linestyle="--")
                ax.axvline(x_end, color="#A9A9A9", linestyle="--")
        if getattr(config, "x_dir", "def") == "rev":
            ax_res.invert_xaxis()
        fig.tight_layout()
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        return fig

    #
    def plot_mcmc(
        self,
        *,
        file: Any = None,
        model: str | None = None,
        fit_type: FitType = "baseline",
        show_plot: bool = True,
    ) -> None:
        """
        Plot the MCMC diagnostics of the latest matching fit.

        Renders the two figures ``fit_wrapper`` shows at fit time, from the
        persisted slot (``SavedFitSlot.mcmc``): the per-walker acceptance
        fraction and the corner plot of the posterior samples. Works
        identically on ``Project.results`` and on archives loaded via
        :meth:`FitResults.load`. The acceptance panel is skipped for slots
        loaded from schema-2 archives (which did not store
        ``acceptance_fraction``).

        Parameters
        ----------
        file : str | SavedFile | trspecfit.File | None
            Filter to a single file (name string or object with ``.name``).
        model : str, optional
            Filter to a single model name.
        fit_type : {'baseline', 'spectrum', 'sbs', '2d'}, default='baseline'
            Which fit to plot (latest matching fit wins). For SbS fits the
            payload is slice 0's.
        show_plot : bool, default True
            Set ``False`` to build without displaying (tests / batch use).

        Raises
        ------
        ValueError
            If no matching fit exists, or the fit had no MCMC step.
        """

        import corner
        import matplotlib.pyplot as plt

        mcmc = self.get_mcmc(file=file, model=model, fit_type=fit_type)
        if mcmc.acceptance_fraction is not None:
            fig_walker, ax = plt.subplots(1, 1, dpi=75)
            ax.plot(mcmc.acceptance_fraction, "o")
            ax.set_xlabel("Walker number")
            ax.set_ylabel("Acceptance fraction")
            if show_plot:
                plt.show()
            else:
                plt.close(fig_walker)
        if not mcmc.flatchain.empty:
            var_names = list(mcmc.flatchain.columns)
            truths = None
            if not mcmc.table.empty:
                best = dict(
                    zip(
                        mcmc.table.iloc[:, 0],
                        mcmc.table["best fit"],
                        strict=True,
                    )
                )
                truths = [best.get(name) for name in var_names]
            fig_corner = plt.figure(figsize=(10, 10))
            corner.corner(
                mcmc.flatchain,
                labels=var_names,
                truths=truths,
                fig=fig_corner,
            )
            if show_plot:
                plt.show()
            else:
                plt.close(fig_corner)

    #
    def plot_param_evolution(
        self,
        *,
        file: Any = None,
        model: str | None = None,
        params: Sequence[str] | None = None,
        config: Any = None,
        show_plot: bool = True,
    ) -> None:
        """
        Plot per-parameter evolution vs time for the latest matching SbS fit.

        One panel per parameter, values from the slot's wide per-slice
        ``params`` frame, x-axis from the file's time axis (array index if
        no axes provider is available).

        Parameters
        ----------
        file : str | SavedFile | trspecfit.File | None
            Filter to a single file (name string or object with ``.name``).
        model : str, optional
            Filter to a single model name.
        params : sequence of str, optional
            Which parameters to plot. Default: the varied parameters (from
            the slot's ``params_meta``); for slots loaded from schema-2
            archives (no ``params_meta``), every parameter. Plots nothing
            if the default resolves to an empty set (all-fixed model).
        config : PlotConfig, optional
            Styling override (see :meth:`plot_fit`).
        show_plot : bool, default True
            Set ``False`` to build without displaying.

        Raises
        ------
        ValueError
            If no matching SbS fit exists.
        KeyError
            If ``params`` names a parameter the fit does not have.
        """

        slot = self._latest_slot(file=file, model=model, fit_type="sbs")
        if params is None:
            if slot.params_meta is not None:
                params = [
                    str(name)
                    for name, vary in zip(
                        slot.params_meta["name"],
                        slot.params_meta["vary"],
                        strict=True,
                    )
                    if vary
                ]
            else:
                params = [str(c) for c in slot.params.columns]
        else:
            params = [str(p) for p in params]
            missing = [p for p in params if p not in slot.params.columns]
            if missing:
                raise KeyError(
                    f"Parameter(s) {missing} not in this SbS fit; available: "
                    f"{list(slot.params.columns)}"
                )
        if not params:
            return
        cfg = self._config_for(slot, config)
        _, time = self._axes_for(slot)
        n_slices = len(slot.params)
        x = (
            np.asarray(time)[:n_slices]
            if time is not None and np.asarray(time).size >= n_slices
            else np.arange(n_slices)
        )
        from trspecfit import fitlib

        fitlib.plt_fit_res_pars(
            df=slot.params.loc[:, params],
            x=x,
            config=cfg,
            save_img=0 if show_plot else -2,
        )

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
              column (NaN for non-SbS rows). Rows are emitted slice-major
              (ascending ``slice_index`` with competing models adjacent at
              each slice; non-SbS rows last) so ``head()`` compares models
              at the same slice. ``sigma_eff`` is broadcast from the slot's
              scalar to every slice row.

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

        Uses real energy/time axes when this ``FitResults`` carries an
        axes provider for the file (always the case for ``Project.results``
        and ``FitResults.load``); falls back to array indices otherwise.
        For single-fit plots with full styling, see :meth:`plot_fit`.

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
    def _plot_residuals_1d(
        self,
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
            energy, _ = self._axes_for(slot)
            if energy is not None and energy.size == obs.size:
                x, x_label = energy, "energy"
            else:
                x, x_label = np.arange(obs.size), "index"
            axs[0, col].plot(x, obs, "k.", ms=3, label="observed")
            axs[0, col].plot(x, fit, "-", lw=1.5, label="fit")
            axs[0, col].set_title(f"{slot.model_name} ({slot.fit_type})")
            axs[0, col].legend(fontsize="small")
            axs[1, col].plot(x, obs - fit, "-", lw=1.0)
            axs[1, col].axhline(0, color="gray", lw=0.5)
            axs[1, col].set_xlabel(x_label)
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
    def _plot_residuals_2d(
        self,
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
            energy, time = self._axes_for(slot)
            extent = None
            x_label, y_label = "energy index", "time / slice index"
            if (
                energy is not None
                and time is not None
                and res.ndim == 2
                and energy.size == res.shape[1]
                and time.size >= res.shape[0]
            ):
                time_view = np.asarray(time)[: res.shape[0]]
                extent = (
                    float(energy[0]),
                    float(energy[-1]),
                    float(time_view[0]),
                    float(time_view[-1]),
                )
                x_label, y_label = "energy", "time"
            im = axs[0, col].imshow(
                res,
                aspect="auto",
                cmap="RdBu_r",
                vmin=-global_max,
                vmax=global_max,
                origin="lower",
                extent=extent,
            )
            axs[0, col].set_title(f"{slot.model_name} ({slot.fit_type})")
            axs[0, col].set_xlabel(x_label)
            if col == 0:
                axs[0, col].set_ylabel(y_label)
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

        Rows are emitted **slice-major**: ascending ``slice_index`` with all
        competing models at a given slice adjacent, so ``head()`` compares
        models at the same slice instead of scrolling through one model's
        full time series first. The sort is stable, so models keep their
        original (slot) order within a slice; non-SbS rows sort to the end.
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
        df = pd.DataFrame(rows, columns=columns)
        # Slice-major ordering so head()/eyeballing compares competing models
        # at the same slice. Coerce slice_index to a numeric key (NA -> NaN)
        # rather than sorting the mixed int/NA column directly, which can trip
        # pandas' "boolean value of NA is ambiguous". Stable sort preserves the
        # original model order within a slice; na_position pushes non-SbS rows
        # to the end.
        sort_key = pd.to_numeric(df["slice_index"], errors="coerce")
        return (
            df.assign(_sort_key=sort_key)
            .sort_values("_sort_key", kind="stable", na_position="last")
            .drop(columns="_sort_key")
            .reset_index(drop=True)
        )
