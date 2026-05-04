"""
Fit-result persistence: dataclasses, identity helpers, slot extractors.

This module owns the data model for completed fits. ``SavedFitSlot`` is the
first-class owner of ``observed`` / ``fit`` / ``metrics`` and the identity
fields (``observed_sha256``, ``selection_json``, ``history_key``) — neither
``Model`` nor ``File`` carries those concerns. Each fit code path captures
snapshot args at fit completion and calls the matching ``_slot_from_<fit_type>``
helper, which builds the slot in one shot.

Pipeline:

    fit path -> snapshot args -> _slot_from_<fit_type> -> SavedFitSlot
                                                            |
                                                            v
                                                  Project._fit_history
                                                            |
                                                            v
                                          Project.results / save / export

Helpers receive plain copied snapshot args (numpy arrays, primitives,
DataFrames) — never live ``Model`` or ``File`` references — so they cannot be
broken by post-fit cleanup that overwrites live state.
"""

import datetime
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Literal, cast

import h5py
import numpy as np
import pandas as pd

from trspecfit.fitlib import compute_fit_metrics

FitType = Literal["baseline", "spectrum", "sbs", "2d"]
SCHEMA_VERSION = "1"


#
def _as_group(obj: Any) -> h5py.Group:
    """
    Narrow an h5py lookup result (``Group | Dataset | Datatype | Link``) to
    ``h5py.Group``, raising if it isn't one. Used at archive-traversal sites
    to give pyright a stable type without sprinkling ``cast`` everywhere.
    """

    if not isinstance(obj, h5py.Group):
        raise TypeError(
            f"expected h5py.Group at archive path, got {type(obj).__name__}"
        )
    return obj


#
@dataclass(frozen=True)
class SavedFitSlot:
    """
    One completed fit result for a (file, model, fit_type, selection) tuple.

    Immutable after construction. Built once at fit completion by
    ``_slot_from_<fit_type>`` and appended to ``Project._fit_history``.

    Attributes
    ----------
    file_fingerprint : dict
        ``{"data_sha256", "energy_sha256", "time_sha256", "shape"}`` — used to
        match this slot back to its source file across sessions.
    file_name : str
        Display name of the file (``File.name``). Identity uses fingerprint;
        ``file_name`` is metadata only.
    model_name : str
    fit_type : {"baseline", "spectrum", "sbs", "2d"}
    selection : dict
        Fit-view identity. Shape depends on ``fit_type``:

        - baseline: ``{"base_t_ind", "e_lim"}``
        - spectrum: ``{"time_point", "time_range", "time_type", "e_lim"}``
        - sbs:      ``{"e_lim", "t_lim"}``
        - 2d:       ``{"e_lim", "t_lim"}``

    selection_json : str
        Deterministic JSON of ``selection`` (sorted keys); used in
        ``history_key`` so refits with different selections do not collide.
    observed_sha256 : str
        Hash of ``observed.tobytes()`` — defensive cross-check guarding against
        silent grid drift if ``selection`` ever fails to capture a view detail.
    history_key : str
        ``sha256(file_fingerprint | model_name | fit_type | selection_json)``.
        Used by snapshot collapse and in-session dedup.
    params : pd.DataFrame
        ``[name, value, init_value, stderr, min, max, vary, expr]``. For SbS,
        a per-slice DataFrame (one row per slice, columns are param values).
    metrics : dict
        ``{"chi2", "chi2_red", "r2", "aic", "bic"}``. Scalar floats for
        baseline/spectrum/2d. For SbS, each value is a 1D ``np.ndarray`` of
        length ``n_slices``.
    observed : np.ndarray
        Data view that was fit against (cropped to ``e_lim`` / ``t_lim`` where
        applicable). ``observed.shape == fit.shape`` always.
    fit : np.ndarray
        Model evaluated at final params on the same grid as ``observed``.
    fit_alg : str
        Optimizer name (e.g. ``"Nelder"``, ``"leastsq"``). For two-stage fits,
        the final stage's algorithm.
    yaml_filename : str | None
        YAML file stem for human reference. Not promised to round-trip.
    timestamp : str
        ISO 8601 UTC timestamp of slot construction.
    conf_ci : pd.DataFrame | None
    mcmc : dict | None
        ``{"flatchain", "ci", "lnsigma"}`` if MCMC ran, else ``None``.
    """

    file_fingerprint: dict[str, Any]
    file_name: str
    model_name: str
    fit_type: FitType
    selection: dict[str, Any]
    selection_json: str
    observed_sha256: str
    history_key: str
    params: pd.DataFrame
    metrics: dict[str, Any]
    observed: np.ndarray
    fit: np.ndarray
    fit_alg: str
    yaml_filename: str | None
    timestamp: str
    conf_ci: pd.DataFrame | None = None
    mcmc: dict[str, Any] | None = None


#
@dataclass(frozen=True)
class SavedFile:
    """
    Archive-side container for a single file's raw data, identity, and slots.

    Used by both writer and reader. The writer assembles ``SavedFile``
    records from a Project + filtered slot list before serializing; the
    reader returns them as the contents of the loaded archive.

    Attributes
    ----------
    name : str
        ``File.name``.
    original_path : str
        Absolute path of the source data file at save time. May not exist
        on the loading machine; used as a tie-break for matching only.
    dim : int
        1 or 2.
    shape : tuple[int, ...]
        ``data.shape``.
    fingerprint : dict
        ``{"data_sha256", "energy_sha256", "time_sha256", "shape"}``.
        Authoritative file identity across machines.
    data : np.ndarray
    energy : np.ndarray
    time : np.ndarray
        Empty array for 1D files.
    e_lim, t_lim : list[int] | None
        ``[start, stop)`` index slices, or ``None``.
    slots : tuple[SavedFitSlot, ...]
        Slots belonging to this file. Tuple (not list) to keep the record
        immutable; the writer accumulates slots into a list and freezes
        on construction.
    """

    name: str
    original_path: str
    dim: int
    shape: tuple[int, ...]
    fingerprint: dict[str, Any]
    data: np.ndarray
    energy: np.ndarray
    time: np.ndarray
    e_lim: list[int] | None
    t_lim: list[int] | None
    slots: tuple[SavedFitSlot, ...]


#
@dataclass(frozen=True)
class SavedProject:
    """
    Top-level archive container.

    The writer takes a ``SavedProject`` and serializes it to HDF5; the
    reader does the inverse. Construction is positional-only — callers
    typically build via ``build_saved_project_from_slots`` rather than
    instantiating directly.

    Attributes
    ----------
    name : str
        Project name.
    trspecfit_version : str
    schema_version : str
        Currently ``"1"``. Bumped on incompatible schema changes.
    timestamp_created : str
        ISO 8601 UTC; first archive-write time.
    timestamp_updated : str
        ISO 8601 UTC; most recent archive-write time. Equal to
        ``timestamp_created`` on the initial save.
    files : tuple[SavedFile, ...]
    """

    name: str
    trspecfit_version: str
    schema_version: str
    timestamp_created: str
    timestamp_updated: str
    files: tuple[SavedFile, ...]


#
# --- identity helpers --------------------------------------------------------
#


#
def compute_file_fingerprint(
    *,
    data: np.ndarray,
    energy: np.ndarray,
    time: np.ndarray | None,
) -> dict[str, Any]:
    """
    Multi-sha fingerprint identifying a File's content.

    Returns ``{"data_sha256", "energy_sha256", "time_sha256", "shape"}``.
    ``time_sha256`` is ``""`` for 1D files (no time axis). Multiple shas plus
    shape avoid the "identical replicate files share data hash" collision.
    """

    data_arr = np.ascontiguousarray(data)
    energy_arr = np.ascontiguousarray(energy)
    fp: dict[str, Any] = {
        "data_sha256": hashlib.sha256(data_arr.tobytes()).hexdigest(),
        "energy_sha256": hashlib.sha256(energy_arr.tobytes()).hexdigest(),
        "shape": tuple(int(x) for x in data_arr.shape),
    }
    if time is None:
        fp["time_sha256"] = ""
    else:
        time_arr = np.ascontiguousarray(time)
        fp["time_sha256"] = hashlib.sha256(time_arr.tobytes()).hexdigest()
    return fp


#
def compute_observed_sha256(observed: np.ndarray) -> str:
    """Hash the observed array (defensive cross-check for grid drift)."""

    return hashlib.sha256(np.ascontiguousarray(observed).tobytes()).hexdigest()


#
def build_selection_json(fit_type: FitType, **fields: Any) -> str:
    """
    Deterministic JSON serialization of a slot's selection dict.

    Sorted keys + ``default=_json_default`` ensure equivalent selections
    produce identical strings (and therefore identical history keys).
    """

    return json.dumps(fields, sort_keys=True, default=_json_default)


#
def _json_default(obj: Any) -> Any:
    """JSON fallback for numpy scalars / arrays / tuples."""

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON-serializable")


#
def compute_history_key(
    *,
    file_fingerprint: dict[str, Any],
    model_name: str,
    fit_type: FitType,
    selection_json: str,
) -> str:
    """
    In-memory canonical slot key.

    ``sha256(file_fingerprint | model_name | fit_type | selection_json)``.
    Slots with the same key represent re-fits of the same view; snapshot save
    keeps only the latest per key.
    """

    fp_json = json.dumps(file_fingerprint, sort_keys=True, default=_json_default)
    payload = f"{fp_json}|{model_name}|{fit_type}|{selection_json}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


#
def compute_archive_slot_key(
    *,
    file_ref: str,
    model_name: str,
    fit_type: FitType,
    selection_json: str,
) -> str:
    """
    On-disk canonical slot key.

    ``sha256(file_ref | model_name | fit_type | selection_json)``. Differs
    from ``history_key`` only in the file-identity token: in-memory uses the
    multi-sha fingerprint; on-disk uses the archive-local positional path
    (e.g. ``"files/000000"``). See ``docs/design/fit_archive_schema.md``.
    """

    payload = f"{file_ref}|{model_name}|{fit_type}|{selection_json}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


#
def _now_iso() -> str:
    """Current UTC timestamp in ISO 8601 (seconds precision)."""

    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


#
def _mcmc_payload(
    emcee_fin: Any,
    emcee_ci: pd.DataFrame,
) -> dict[str, Any] | None:
    """
    Build the ``mcmc`` slot payload from ``fit_wrapper``'s emcee outputs.

    Returns ``None`` if MCMC did not run (``emcee_fin is None``). Otherwise
    returns ``{"flatchain", "ci", "lnsigma"}`` matching ``SavedFitSlot.mcmc``.
    Frames are copied so the slot is invariant to subsequent state changes.
    """

    if emcee_fin is None:
        return None
    flatchain = getattr(emcee_fin, "flatchain", None)
    if isinstance(flatchain, pd.DataFrame):
        flatchain_out: pd.DataFrame | None = flatchain.copy()
    else:
        flatchain_out = None
    params = getattr(emcee_fin, "params", None)
    lnsigma_par = params.get("__lnsigma") if params is not None else None
    lnsigma = float(lnsigma_par.value) if lnsigma_par is not None else None
    ci_out = emcee_ci.copy() if not emcee_ci.empty else None
    return {"flatchain": flatchain_out, "ci": ci_out, "lnsigma": lnsigma}


#
# --- per-fit-type slot extractors -------------------------------------------
#


#
def _slot_from_baseline(
    *,
    file_fingerprint: dict[str, Any],
    file_name: str,
    model_name: str,
    fit_alg: str,
    yaml_filename: str | None,
    params_df: pd.DataFrame,
    observed: np.ndarray,
    fit: np.ndarray,
    base_t_ind: list[int],
    e_lim: list[int] | None,
    n_free_pars: int,
    conf_ci: pd.DataFrame | None = None,
    mcmc: dict[str, Any] | None = None,
) -> SavedFitSlot:
    """
    Build a SavedFitSlot for a completed baseline fit.

    Caller passes already-copied snapshot args (no live Model references) so
    the helper is invariant to post-fit cleanup.
    """

    selection = {
        "base_t_ind": list(base_t_ind),
        "e_lim": list(e_lim) if e_lim else None,
    }
    return _build_slot(
        file_fingerprint=file_fingerprint,
        file_name=file_name,
        model_name=model_name,
        fit_type="baseline",
        selection=selection,
        params=params_df,
        observed=observed,
        fit=fit,
        n_free_pars=n_free_pars,
        fit_alg=fit_alg,
        yaml_filename=yaml_filename,
        conf_ci=conf_ci,
        mcmc=mcmc,
    )


#
def _slot_from_spectrum(
    *,
    file_fingerprint: dict[str, Any],
    file_name: str,
    model_name: str,
    fit_alg: str,
    yaml_filename: str | None,
    params_df: pd.DataFrame,
    observed: np.ndarray,
    fit: np.ndarray,
    time_point: float | None,
    time_range: list[float] | None,
    time_type: str,
    e_lim: list[int] | None,
    n_free_pars: int,
    conf_ci: pd.DataFrame | None = None,
    mcmc: dict[str, Any] | None = None,
) -> SavedFitSlot:
    """Build a SavedFitSlot for a completed spectrum fit."""

    selection = {
        "time_point": time_point,
        "time_range": list(time_range) if time_range else None,
        "time_type": time_type,
        "e_lim": list(e_lim) if e_lim else None,
    }
    return _build_slot(
        file_fingerprint=file_fingerprint,
        file_name=file_name,
        model_name=model_name,
        fit_type="spectrum",
        selection=selection,
        params=params_df,
        observed=observed,
        fit=fit,
        n_free_pars=n_free_pars,
        fit_alg=fit_alg,
        yaml_filename=yaml_filename,
        conf_ci=conf_ci,
        mcmc=mcmc,
    )


#
def _slot_from_sbs(
    *,
    file_fingerprint: dict[str, Any],
    file_name: str,
    model_name: str,
    fit_alg: str,
    yaml_filename: str | None,
    params_df: pd.DataFrame,
    observed: np.ndarray,
    fit: np.ndarray,
    e_lim: list[int] | None,
    t_lim: list[int] | None,
    n_free_pars: int,
    conf_ci: pd.DataFrame | None = None,
    mcmc: dict[str, Any] | None = None,
) -> SavedFitSlot:
    """
    Build a SavedFitSlot for a completed slice-by-slice fit.

    ``observed`` and ``fit`` are 2D arrays (slices x energy_in_lim). ``metrics``
    values are per-slice 1D arrays. ``params_df`` is the SbS DataFrame
    (one row per slice).
    """

    selection = {
        "e_lim": list(e_lim) if e_lim else None,
        "t_lim": list(t_lim) if t_lim else None,
    }
    metrics = _per_slice_metrics(observed=observed, fit=fit, n_free_pars=n_free_pars)
    selection_json = build_selection_json("sbs", **selection)
    history_key = compute_history_key(
        file_fingerprint=file_fingerprint,
        model_name=model_name,
        fit_type="sbs",
        selection_json=selection_json,
    )
    return SavedFitSlot(
        file_fingerprint=dict(file_fingerprint),
        file_name=file_name,
        model_name=model_name,
        fit_type="sbs",
        selection=selection,
        selection_json=selection_json,
        observed_sha256=compute_observed_sha256(observed),
        history_key=history_key,
        params=params_df,
        metrics=metrics,
        observed=np.asarray(observed),
        fit=np.asarray(fit),
        fit_alg=fit_alg,
        yaml_filename=yaml_filename,
        timestamp=_now_iso(),
        conf_ci=conf_ci,
        mcmc=mcmc,
    )


#
def _slot_from_2d(
    *,
    file_fingerprint: dict[str, Any],
    file_name: str,
    model_name: str,
    fit_alg: str,
    yaml_filename: str | None,
    params_df: pd.DataFrame,
    observed: np.ndarray,
    fit: np.ndarray,
    e_lim: list[int] | None,
    t_lim: list[int] | None,
    n_free_pars: int,
    conf_ci: pd.DataFrame | None = None,
    mcmc: dict[str, Any] | None = None,
) -> SavedFitSlot:
    """Build a SavedFitSlot for a completed 2D global fit."""

    selection = {
        "e_lim": list(e_lim) if e_lim else None,
        "t_lim": list(t_lim) if t_lim else None,
    }
    return _build_slot(
        file_fingerprint=file_fingerprint,
        file_name=file_name,
        model_name=model_name,
        fit_type="2d",
        selection=selection,
        params=params_df,
        observed=observed,
        fit=fit,
        n_free_pars=n_free_pars,
        fit_alg=fit_alg,
        yaml_filename=yaml_filename,
        conf_ci=conf_ci,
        mcmc=mcmc,
    )


#
# --- internal builders ------------------------------------------------------
#


#
def _build_slot(
    *,
    file_fingerprint: dict[str, Any],
    file_name: str,
    model_name: str,
    fit_type: FitType,
    selection: dict[str, Any],
    params: pd.DataFrame,
    observed: np.ndarray,
    fit: np.ndarray,
    n_free_pars: int,
    fit_alg: str,
    yaml_filename: str | None,
    conf_ci: pd.DataFrame | None,
    mcmc: dict[str, Any] | None,
) -> SavedFitSlot:
    """Shared scalar-metric path for baseline / spectrum / 2d."""

    metrics = compute_fit_metrics(observed=observed, fit=fit, n_free_pars=n_free_pars)
    selection_json = build_selection_json(fit_type, **selection)
    history_key = compute_history_key(
        file_fingerprint=file_fingerprint,
        model_name=model_name,
        fit_type=fit_type,
        selection_json=selection_json,
    )
    return SavedFitSlot(
        file_fingerprint=dict(file_fingerprint),
        file_name=file_name,
        model_name=model_name,
        fit_type=fit_type,
        selection=selection,
        selection_json=selection_json,
        observed_sha256=compute_observed_sha256(observed),
        history_key=history_key,
        params=params,
        metrics=metrics,
        observed=np.asarray(observed),
        fit=np.asarray(fit),
        fit_alg=fit_alg,
        yaml_filename=yaml_filename,
        timestamp=_now_iso(),
        conf_ci=conf_ci,
        mcmc=mcmc,
    )


#
def _per_slice_metrics(
    *,
    observed: np.ndarray,
    fit: np.ndarray,
    n_free_pars: int,
) -> dict[str, np.ndarray]:
    """Compute per-slice metrics for SbS (one row per time slice)."""

    obs = np.asarray(observed)
    fit_arr = np.asarray(fit)
    if obs.ndim != 2 or fit_arr.shape != obs.shape:
        raise ValueError(
            f"SbS observed/fit must be 2D and matching shapes; "
            f"got observed{obs.shape}, fit{fit_arr.shape}"
        )
    n_slices = obs.shape[0]
    out: dict[str, list[float]] = {
        "chi2": [],
        "chi2_red": [],
        "r2": [],
        "aic": [],
        "bic": [],
    }
    for i in range(n_slices):
        m = compute_fit_metrics(
            observed=obs[i], fit=fit_arr[i], n_free_pars=n_free_pars
        )
        for k in out:
            out[k].append(m[k])
    return {k: np.array(v) for k, v in out.items()}


#
# --- history collapse -------------------------------------------------------
#


#
def collapse_history_to_snapshot(slots: list[SavedFitSlot]) -> list[SavedFitSlot]:
    """
    Keep the latest slot per ``history_key`` (snapshot semantics).

    Used by ``Project.save_fits`` (and any other consumer that wants
    "current state" rather than "every iteration").
    """

    latest: dict[str, SavedFitSlot] = {}
    for slot in slots:
        latest[slot.history_key] = slot
    return list(latest.values())


#
# --- archive lookup helpers -------------------------------------------------
#


#
def _find_file_by_fingerprint(
    archive: h5py.File | h5py.Group,
    fingerprint: dict[str, Any],
    *,
    name: str | None = None,
    original_path: str | None = None,
) -> h5py.Group | None:
    """
    Look up a file group inside an archive.

    Matches on the file fingerprint (``data_sha256`` + ``energy_sha256`` +
    ``time_sha256`` + ``shape``). When ``name`` and/or ``original_path`` are
    given, the candidate group's metadata attrs must also match those
    values; this is how the writer enforces the
    ``(fingerprint, name, original_path)`` identity rule from
    ``docs/design/fit_archive_schema.md``. Read-side callers may omit the
    tie-break args for fingerprint-only matching.

    Returns the first matching ``files/<id>/`` group in positional-key
    order, or ``None`` if no candidate satisfies all supplied predicates.
    """

    files_obj = archive.get("files")
    if files_obj is None:
        return None
    files_group = _as_group(files_obj)
    for key in sorted(files_group.keys()):
        fg = _as_group(files_group[key])
        meta = _as_group(fg["metadata"])
        if str(meta.attrs.get("data_sha256", "")) != fingerprint["data_sha256"]:
            continue
        if str(meta.attrs.get("energy_sha256", "")) != fingerprint["energy_sha256"]:
            continue
        if str(meta.attrs.get("time_sha256", "")) != fingerprint["time_sha256"]:
            continue
        archived_shape = tuple(int(x) for x in meta.attrs.get("shape", []))
        if archived_shape != tuple(fingerprint["shape"]):
            continue
        if name is not None and str(meta.attrs.get("name", "")) != name:
            continue
        if original_path is not None:
            if str(meta.attrs.get("original_path", "")) != original_path:
                continue
        return fg
    return None


#
def _find_slot_by_archive_key(
    file_group: h5py.Group,
    archive_slot_key: str,
) -> h5py.Group | None:
    """
    Look up a slot inside a file group by its ``archive_slot_key``.

    Used by ``Project.save_fits`` to detect an existing slot at the same
    canonical identity, so it can apply the slot-scoped overwrite policy.
    Returns ``None`` if no slot under ``file_group/slots/`` carries the
    given key.
    """

    slots_obj = file_group.get("slots")
    if slots_obj is None:
        return None
    slots_group = _as_group(slots_obj)
    for key in sorted(slots_group.keys()):
        slot_group = _as_group(slots_group[key])
        meta_obj = slot_group.get("metadata")
        if meta_obj is None:
            continue
        meta = _as_group(meta_obj)
        if str(meta.attrs.get("archive_slot_key", "")) == archive_slot_key:
            return slot_group
    return None


#
def _next_positional_key(parent: h5py.Group) -> str:
    """Smallest unused six-digit zero-padded key in ``parent``."""

    used = {int(k) for k in parent.keys() if k.isdigit()}
    n = 0
    while n in used:
        n += 1
    return f"{n:06d}"


#
# --- DataFrame encoding (per fit_archive_schema.md "DataFrame encoding") ----
#

TypeTag = Literal["str", "float64", "bool"]
_VLEN_STR = h5py.string_dtype(encoding="utf-8")


#
def _infer_type_tag(series: pd.Series) -> TypeTag:
    """
    Map a pandas Series to one of ``{"str", "float64", "bool"}``.

    Integer columns are promoted to ``float64`` (the schema only emits
    bool, float, str). Object-dtype columns are inspected sample-wise.
    """

    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_numeric_dtype(series):
        return "float64"
    non_na = series.dropna()
    if len(non_na) == 0:
        return "str"
    sample = non_na.iloc[0]
    if isinstance(sample, bool | np.bool_):
        return "bool"
    if isinstance(sample, int | float | np.integer | np.floating):
        return "float64"
    return "str"


#
def _pack_for_dtype(value: Any, tag: TypeTag) -> Any:
    """Coerce a scalar to the storage dtype, mapping None/NaN to a default."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        if tag == "str":
            return ""
        if tag == "float64":
            return np.nan
        return False
    if tag == "str":
        return str(value)
    if tag == "float64":
        return float(value)
    return bool(value)


#
def _encode_dataframe(
    group: h5py.Group,
    name: str,
    df: pd.DataFrame,
    *,
    type_tags: Sequence[TypeTag] | None = None,
) -> h5py.Dataset:
    """
    Write a DataFrame to ``group/name`` using the schema's encoding rule.

    If every column's tag is ``"float64"``, the result is a 2D ``float64``
    dataset of shape ``(n_rows, n_cols)`` with attr ``columns``
    (all-numeric form). Otherwise it is a 1D structured dataset of shape
    ``(n_rows,)`` with positional ``c000000, c000001, ...`` fields, plus
    attrs ``columns`` and ``dtypes`` (heterogeneous form).

    ``type_tags`` may be supplied when the caller knows the schema; if
    ``None``, tags are inferred per column.
    """

    columns = [str(c) for c in df.columns]
    if type_tags is None:
        tags: list[TypeTag] = [
            _infer_type_tag(cast(pd.Series, df[c])) for c in df.columns
        ]
    else:
        if len(type_tags) != len(columns):
            raise ValueError(
                f"type_tags length {len(type_tags)} does not match "
                f"DataFrame column count {len(columns)}"
            )
        tags = list(type_tags)

    n_rows = len(df)
    if all(t == "float64" for t in tags):
        values = df.to_numpy(dtype=np.float64, copy=True)
        if values.ndim == 1:
            values = values.reshape(n_rows, len(columns))
        ds = group.create_dataset(name, data=values)
        ds.attrs["columns"] = np.array(columns, dtype=_VLEN_STR)
        return ds

    field_keys = [f"c{i:06d}" for i in range(len(columns))]
    field_dtypes: list[tuple[str, Any]] = []
    for key, tag in zip(field_keys, tags, strict=True):
        if tag == "str":
            field_dtypes.append((key, _VLEN_STR))
        elif tag == "float64":
            field_dtypes.append((key, "f8"))
        else:
            field_dtypes.append((key, "?"))
    arr = np.empty(n_rows, dtype=field_dtypes)
    for col_name, key, tag in zip(columns, field_keys, tags, strict=True):
        col = df[col_name]
        arr[key] = [_pack_for_dtype(v, tag) for v in col]
    ds = group.create_dataset(name, data=arr)
    ds.attrs["columns"] = np.array(columns, dtype=_VLEN_STR)
    ds.attrs["dtypes"] = np.array(tags, dtype=_VLEN_STR)
    return ds


#
# --- HDF5 writer ------------------------------------------------------------
#

# Per fit_archive_schema.md "params dataset" — long format for non-sbs fits.
_PARAMS_LONG_TYPE_TAGS: list[TypeTag] = [
    "str",  # name
    "float64",  # value
    "float64",  # stderr
    "float64",  # init_value
    "float64",  # min
    "float64",  # max
    "bool",  # vary
    "str",  # expr
]
_METRICS_KEYS = ("chi2", "chi2_red", "r2", "aic", "bic")


#
def _all_float64_tags(n: int) -> list[TypeTag]:
    """All-float64 tag list, typed properly for ``_encode_dataframe``."""

    return ["float64"] * n


#
def write_archive(
    filepath: PathLike | str,
    *,
    project: SavedProject,
    overwrite: bool = False,
) -> None:
    """
    Serialize a ``SavedProject`` to an HDF5 archive.

    See ``docs/design/fit_archive_schema.md`` for the on-disk layout.
    Behavior:

    - **Append-mode by default.** If ``filepath`` exists, files and slots are
      added in place. The archive's ``timestamp_created`` is preserved;
      ``timestamp_updated`` is rewritten on every save. To start fresh, pass
      a new path (or remove the existing file first).
    - **Slot-scoped overwrite.** A slot collision (same ``archive_slot_key``
      already in the archive) raises ``FileExistsError`` unless
      ``overwrite=True``, in which case the existing slot is deleted and
      replaced. Other slots in the archive are untouched.
    - **Pre-check on append.** All slot collisions are detected before any
      mutation, so a single conflicting slot does not leave half the
      payload written.
    """

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "a") as archive:
        is_new = _classify_archive_for_write(archive, project, path=path)
        if not is_new and not overwrite:
            _precheck_slot_collisions(archive, project)
        _write_top_metadata(archive, project, is_new=is_new)
        files_group = archive.require_group("files")
        for sf in project.files:
            file_group = _find_file_by_fingerprint(
                archive,
                sf.fingerprint,
                name=sf.name,
                original_path=sf.original_path,
            )
            if file_group is None:
                key = _next_positional_key(files_group)
                file_group = files_group.create_group(key)
                _write_file_payload(file_group, sf)
            file_ref = _file_ref(file_group)
            for slot in sf.slots:
                _write_slot(file_group, file_ref, slot, overwrite=overwrite)


#
def _file_ref(file_group: h5py.Group) -> str:
    """Archive-local path used as a slot's ``file_ref`` attr."""

    name = cast(str | None, file_group.name)
    assert name is not None  # type guard
    return name.lstrip("/")


#
def _classify_archive_for_write(
    archive: h5py.File,
    project: SavedProject,
    *,
    path: Path,
) -> bool:
    """
    Decide whether the open archive needs initialization or is appendable.

    Returns ``True`` if the writer should treat this as a new archive
    (write all top-level metadata attrs), ``False`` if it is an existing
    fit archive whose ``schema_version`` matches and we should append.

    Raises ``ValueError`` if the file is non-empty but missing fit-archive
    metadata (foreign HDF5 or partially-written archive), or if its
    ``schema_version`` does not match the writer's. h5py's ``"a"`` mode
    creates the file on open, so an empty file at this point is either a
    brand-new archive or an empty stub the user created elsewhere; both
    are safe to initialize.
    """

    meta_obj = archive.get("metadata")
    meta = _as_group(meta_obj) if meta_obj is not None else None
    if meta is not None and "schema_version" in meta.attrs:
        existing = str(meta.attrs["schema_version"])
        if existing != project.schema_version:
            raise ValueError(
                f"Archive at {path} has schema_version {existing!r} which "
                f"does not match writer's {project.schema_version!r}; cannot "
                f"append. Choose a new path."
            )
        return False
    if len(archive.keys()) > 0:
        raise ValueError(
            f"File at {path} exists but is not a recognized trspecfit fit "
            f"archive (missing metadata/schema_version). Choose a different "
            f"path or remove the existing file."
        )
    return True


#
def _precheck_slot_collisions(archive: h5py.File, project: SavedProject) -> None:
    """
    Raise ``FileExistsError`` if any slot would collide with an existing one.

    Runs before any mutation so partial writes cannot happen on append.
    """

    for sf in project.files:
        existing_fg = _find_file_by_fingerprint(
            archive,
            sf.fingerprint,
            name=sf.name,
            original_path=sf.original_path,
        )
        if existing_fg is None:
            continue
        file_ref = _file_ref(existing_fg)
        for slot in sf.slots:
            key = compute_archive_slot_key(
                file_ref=file_ref,
                model_name=slot.model_name,
                fit_type=slot.fit_type,
                selection_json=slot.selection_json,
            )
            if _find_slot_by_archive_key(existing_fg, key) is not None:
                raise FileExistsError(
                    f"Slot already exists in archive for "
                    f"file={sf.name!r}, model={slot.model_name!r}, "
                    f"fit_type={slot.fit_type!r}; pass overwrite=True to replace."
                )


#
def _write_top_metadata(
    archive: h5py.File, project: SavedProject, *, is_new: bool
) -> None:
    """Write or update the top-level ``metadata`` group attrs."""

    meta = archive.require_group("metadata")
    meta.attrs["trspecfit_version"] = project.trspecfit_version
    meta.attrs["timestamp_updated"] = project.timestamp_updated
    if is_new:
        meta.attrs["project_name"] = project.name
        meta.attrs["timestamp_created"] = project.timestamp_created
        meta.attrs["schema_version"] = project.schema_version


#
def _write_file_payload(file_group: h5py.Group, sf: SavedFile) -> None:
    """Write file metadata, raw arrays, and an empty ``slots/`` subgroup."""

    meta = file_group.create_group("metadata")
    meta.attrs["name"] = sf.name
    meta.attrs["original_path"] = sf.original_path
    meta.attrs["dim"] = int(sf.dim)
    meta.attrs["shape"] = np.array(sf.shape, dtype=np.int64)
    meta.attrs["data_sha256"] = sf.fingerprint["data_sha256"]
    meta.attrs["energy_sha256"] = sf.fingerprint["energy_sha256"]
    meta.attrs["time_sha256"] = sf.fingerprint["time_sha256"]
    if sf.e_lim is not None:
        meta.attrs["e_lim"] = np.array(sf.e_lim, dtype=np.int64)
    if sf.t_lim is not None:
        meta.attrs["t_lim"] = np.array(sf.t_lim, dtype=np.int64)
    # Preserve source dtype: the file's fingerprint hashes the original
    # bytes, so casting on write would invalidate `data_sha256` /
    # `energy_sha256` / `time_sha256` for non-float64 inputs.
    file_group.create_dataset("energy", data=np.ascontiguousarray(sf.energy))
    file_group.create_dataset("time", data=np.ascontiguousarray(sf.time))
    file_group.create_dataset("data", data=np.ascontiguousarray(sf.data))
    file_group.create_group("slots")


#
def _write_slot(
    file_group: h5py.Group,
    file_ref: str,
    slot: SavedFitSlot,
    *,
    overwrite: bool,
) -> None:
    """Append (or replace, if ``overwrite``) one slot under ``file_group``."""

    archive_slot_key = compute_archive_slot_key(
        file_ref=file_ref,
        model_name=slot.model_name,
        fit_type=slot.fit_type,
        selection_json=slot.selection_json,
    )
    slots_group = file_group.require_group("slots")
    existing = _find_slot_by_archive_key(file_group, archive_slot_key)
    if existing is not None:
        if not overwrite:
            # Should have been caught by _precheck_slot_collisions; defense in
            # depth in case the writer is called directly without precheck.
            raise FileExistsError(
                f"Slot exists for model={slot.model_name!r}, "
                f"fit_type={slot.fit_type!r}; pass overwrite=True."
            )
        existing_name = cast(str | None, existing.name)
        assert existing_name is not None  # type guard
        del slots_group[existing_name.rsplit("/", 1)[-1]]

    key = _next_positional_key(slots_group)
    slot_group = slots_group.create_group(key)
    _write_slot_metadata(slot_group, slot, file_ref, archive_slot_key)
    _write_slot_params(slot_group, slot)
    # Preserve source dtype: the slot's `observed_sha256` hashes the original
    # bytes, so casting on write would invalidate the cross-check for slots
    # whose observed array was non-float64 (e.g. sbs slice from float32 data).
    slot_group.create_dataset("observed", data=np.ascontiguousarray(slot.observed))
    slot_group.create_dataset("fit", data=np.ascontiguousarray(slot.fit))
    if slot.fit_type == "sbs":
        _write_metrics_per_slice(slot_group, slot.metrics)
    if slot.conf_ci is not None:
        _encode_dataframe(slot_group, "conf_ci", slot.conf_ci)
    if slot.mcmc is not None:
        _write_mcmc_group(slot_group, slot.mcmc)


#
def _write_slot_metadata(
    slot_group: h5py.Group,
    slot: SavedFitSlot,
    file_ref: str,
    archive_slot_key: str,
) -> None:
    """Identity + provenance + (non-sbs) scalar metric attrs."""

    meta = slot_group.create_group("metadata")
    meta.attrs["file_ref"] = file_ref
    meta.attrs["model_name"] = slot.model_name
    meta.attrs["fit_type"] = slot.fit_type
    meta.attrs["selection_json"] = slot.selection_json
    meta.attrs["archive_slot_key"] = archive_slot_key
    meta.attrs["history_key"] = slot.history_key
    meta.attrs["observed_sha256"] = slot.observed_sha256
    meta.attrs["fit_alg"] = slot.fit_alg
    if slot.yaml_filename is not None:
        meta.attrs["yaml_filename"] = slot.yaml_filename
    meta.attrs["timestamp"] = slot.timestamp
    if slot.fit_type != "sbs":
        for k in _METRICS_KEYS:
            meta.attrs[k] = float(slot.metrics[k])


#
def _write_slot_params(slot_group: h5py.Group, slot: SavedFitSlot) -> None:
    """``params`` dataset; layout depends on ``fit_type``."""

    if slot.fit_type == "sbs":
        n_cols = len(slot.params.columns)
        _encode_dataframe(
            slot_group,
            "params",
            slot.params,
            type_tags=_all_float64_tags(n_cols),
        )
    else:
        _encode_dataframe(
            slot_group,
            "params",
            slot.params,
            type_tags=_PARAMS_LONG_TYPE_TAGS,
        )


#
def _write_metrics_per_slice(
    slot_group: h5py.Group,
    metrics: dict[str, Any],
) -> None:
    """1D structured dataset (chi2, chi2_red, r2, aic, bic) for sbs fits."""

    arrays = {k: np.asarray(metrics[k], dtype=np.float64) for k in _METRICS_KEYS}
    n = len(arrays[_METRICS_KEYS[0]])
    dtype = [(k, "f8") for k in _METRICS_KEYS]
    out = np.empty(n, dtype=dtype)
    for k in _METRICS_KEYS:
        out[k] = arrays[k]
    slot_group.create_dataset("metrics_per_slice", data=out)


#
def _write_mcmc_group(slot_group: h5py.Group, mcmc: dict[str, Any]) -> None:
    """``mcmc/`` subgroup: flatchain (always), ci (optional), lnsigma attr."""

    mcmc_group = slot_group.create_group("mcmc")
    lnsigma = mcmc.get("lnsigma")
    mcmc_group.attrs["lnsigma"] = (
        float(lnsigma) if lnsigma is not None else float("nan")
    )
    flatchain = mcmc.get("flatchain")
    if flatchain is None:
        flatchain = pd.DataFrame()
    # Always go through _encode_dataframe so a 0-row chain with named
    # columns still records (0, n_cols) + the parameter labels, instead
    # of collapsing to (0, 0). pd.DataFrame().to_numpy() yields (0, 0)
    # for the no-MCMC case, which is the same on-disk shape as before.
    _encode_dataframe(
        mcmc_group,
        "flatchain",
        flatchain,
        type_tags=_all_float64_tags(len(flatchain.columns)),
    )
    ci = mcmc.get("ci")
    if ci is not None:
        _encode_dataframe(mcmc_group, "ci", ci)
