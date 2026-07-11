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

from trspecfit.fitlib import (
    compute_fit_metrics,
    plt_fit_res_2d,
    plt_fit_res_pars,
)
from trspecfit.utils.hdf5 import require_dataset, require_group

FitType = Literal["baseline", "spectrum", "sbs", "2d"]
SCHEMA_VERSION = "2"

# Default noise metadata used when no σ has been set on the File. Mirrors the
# project.yaml defaults defined in ``Project._set_defaults``; if you change one,
# change the other.
NOISE_TYPE_UNKNOWN = "unknown"
NOISE_TYPE_GAUSSIAN = "gaussian"
SIGMA_SOURCE_USER = "user_supplied"
SIGMA_TYPE_CONSTANT = "constant"


#
def normalize_sigma_data(value: Any) -> float:
    """
    Coerce a user-provided ``sigma_data`` to a storage float.

    Accepts ``None`` or ``NaN`` (both returned as ``NaN`` — the "unset"
    marker) or a finite positive number; otherwise raises a clear
    ``ValueError``. NaN-tolerance lets the same function validate both
    raw user input (where ``None`` arrives from YAML ``null``) and the
    in-memory representation (where ``NaN`` already means unset), so
    re-coercing a default value is a safe no-op. Centralizes the
    validation used by ``Project`` YAML loading and ``File.set_sigma``.
    """

    if value is None:
        return float("nan")
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"sigma_data must be None or a finite positive number; got {value!r}"
        ) from exc
    if np.isnan(v):
        return float("nan")
    if not (np.isfinite(v) and v > 0):
        raise ValueError(
            f"sigma_data must be None or a finite positive number; got {value!r}"
        )
    return v


#
def validate_noise_metadata(
    *,
    noise_type: str,
    sigma_source: str,
    sigma_type: str,
) -> None:
    """
    Validate the noise-schema discriminator fields against v1's strict subset.

    v1 supports ``noise_type ∈ {"gaussian", "unknown"}``, ``sigma_source ==
    "user_supplied"``, and ``sigma_type == "constant"``. Future passes will
    relax these (Poisson-derived σ, per-spectrum σ, etc.), but every value
    on disk now must round-trip cleanly through this check.
    """

    if noise_type not in (NOISE_TYPE_GAUSSIAN, NOISE_TYPE_UNKNOWN):
        raise ValueError(
            f"noise_type must be 'gaussian' or 'unknown'; got {noise_type!r}"
        )
    if sigma_source != SIGMA_SOURCE_USER:
        raise ValueError(
            f"sigma_source must be 'user_supplied' (v1); got {sigma_source!r}"
        )
    if sigma_type != SIGMA_TYPE_CONSTANT:
        raise ValueError(f"sigma_type must be 'constant' (v1); got {sigma_type!r}")


#
def _compute_sigma_eff(
    fit_type: FitType,
    selection: dict[str, Any],
    sigma_data: float,
) -> float:
    """
    Effective σ on a slot's fit data view, given the File's per-pixel σ.

    Baseline fits average ``base_t_ind[1] - base_t_ind[0]`` time slices, so
    the per-row noise on ``data_base`` is ``σ_data / √N_avg``. SbS, 2D, and
    spectrum fits operate on per-pixel data → no scaling. ``time_range``
    averaging in ``spectrum`` is *not* auto-corrected in v1 (users
    averaging a spectrum must pre-scale the σ they pass to
    ``File.set_sigma()``).
    """

    if not np.isfinite(sigma_data) or sigma_data <= 0:
        return float("nan")
    if fit_type == "baseline":
        base_t_ind = selection.get("base_t_ind")
        if base_t_ind is not None and len(base_t_ind) == 2:
            n_avg = int(base_t_ind[1]) - int(base_t_ind[0])
            if n_avg > 1:
                return float(sigma_data / np.sqrt(n_avg))
    return float(sigma_data)


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
        ``sha256(file_fingerprint | file_name | model_name | fit_type |
        selection_json)``. ``file_name`` is included so two distinct
        ``Project.files`` with byte-identical raw arrays do not collapse
        into one slot. Used by snapshot collapse and in-session dedup.
    params : pd.DataFrame
        ``[name, value, init_value, stderr, min, max, vary, expr]``. For SbS,
        a per-slice DataFrame (one row per slice, columns are param values).
    metrics : dict
        ``{"chi2_raw", "chi2_red_raw", "chi2", "chi2_red", "r2", "aic",
        "bic"}``. Scalar floats for baseline/spectrum/2d. For SbS, each
        value is a 1D ``np.ndarray`` of length ``n_slices``. ``chi2_raw``
        and ``chi2_red_raw`` are the unweighted lmfit-convention diagnostics
        (always populated). ``chi2`` and ``chi2_red`` are the σ-calibrated
        versions (``≈ 1`` for a fit at the noise floor) and are ``NaN``
        when no sigma was supplied at fit time.
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
    noise_type : str
        Statistical noise assumption captured from the File at fit time —
        ``"gaussian"`` or ``"unknown"``. v1 only supports those two values;
        ``"unknown"`` records "no σ was supplied" without claiming a
        distribution.
    sigma_source : str
        How ``sigma_data`` was obtained. v1 supports ``"user_supplied"``
        only; future passes will add ``"estimated_from_data"`` etc.
    sigma_type : str
        Shape/layout of ``sigma_data``. v1 supports ``"constant"`` only;
        ``"per_spectrum"`` / ``"per_point"`` are reserved for future work.
    sigma_data : float
        File-level per-pixel noise σ at fit time. ``NaN`` when no sigma
        was set on the File (``noise_type == "unknown"``).
    sigma_eff : float
        Effective σ on this slot's fit data view. Equals ``sigma_data``
        for SbS / 2D / spectrum; equals ``sigma_data / √N_avg`` for
        baseline (``N_avg`` = number of time slices averaged into
        ``data_base``). ``NaN`` when ``sigma_data`` is ``NaN``.
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
    noise_type: str
    sigma_source: str
    sigma_type: str
    sigma_data: float
    sigma_eff: float
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
    file_name: str,
    model_name: str,
    fit_type: FitType,
    selection_json: str,
) -> str:
    """
    In-memory canonical slot key.

    ``sha256(file_fingerprint | file_name | model_name | fit_type | selection_json)``.
    ``file_name`` is included so two distinct ``Project.files`` with
    byte-identical raw arrays (same fingerprint, different names) do not
    collapse into a single slot during snapshot save. Project enforces
    unique ``File.name`` within a session, so name suffices as the
    disambiguator (the archive's full identity is
    ``(fingerprint, name, original_path)``; in-memory we only need
    ``name`` to break the fingerprint tie).

    Slots with the same key represent re-fits of the same view of the
    same file; snapshot save keeps only the latest per key.
    """

    fp_json = json.dumps(file_fingerprint, sort_keys=True, default=_json_default)
    payload = f"{fp_json}|{file_name}|{model_name}|{fit_type}|{selection_json}"
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
    noise_type: str,
    sigma_source: str,
    sigma_type: str,
    sigma_data: float,
    conf_ci: pd.DataFrame | None = None,
    mcmc: dict[str, Any] | None = None,
) -> SavedFitSlot:
    """
    Build a SavedFitSlot for a completed baseline fit.

    Caller passes already-copied snapshot args (no live Model references) so
    the helper is invariant to post-fit cleanup. The noise metadata is also
    a snapshot of the File's σ state at fit completion — subsequent calls
    to ``File.set_sigma`` do not retroactively rewrite the slot.
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
        noise_type=noise_type,
        sigma_source=sigma_source,
        sigma_type=sigma_type,
        sigma_data=sigma_data,
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
    noise_type: str,
    sigma_source: str,
    sigma_type: str,
    sigma_data: float,
    conf_ci: pd.DataFrame | None = None,
    mcmc: dict[str, Any] | None = None,
) -> SavedFitSlot:
    """Build a SavedFitSlot for a completed spectrum fit.

    v1 does not auto-correct σ for ``time_range`` averaging — users fitting
    an averaged spectrum should pre-scale the σ they pass to
    ``File.set_sigma()``.
    """

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
        noise_type=noise_type,
        sigma_source=sigma_source,
        sigma_type=sigma_type,
        sigma_data=sigma_data,
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
    noise_type: str,
    sigma_source: str,
    sigma_type: str,
    sigma_data: float,
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
    sigma_eff = _compute_sigma_eff("sbs", selection, sigma_data)
    metrics = _per_slice_metrics(
        observed=observed,
        fit=fit,
        n_free_pars=n_free_pars,
        sigma_eff=sigma_eff if np.isfinite(sigma_eff) else None,
    )
    selection_json = build_selection_json("sbs", **selection)
    history_key = compute_history_key(
        file_fingerprint=file_fingerprint,
        file_name=file_name,
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
        noise_type=noise_type,
        sigma_source=sigma_source,
        sigma_type=sigma_type,
        sigma_data=float(sigma_data),
        sigma_eff=float(sigma_eff),
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
    noise_type: str,
    sigma_source: str,
    sigma_type: str,
    sigma_data: float,
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
        noise_type=noise_type,
        sigma_source=sigma_source,
        sigma_type=sigma_type,
        sigma_data=sigma_data,
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
    noise_type: str,
    sigma_source: str,
    sigma_type: str,
    sigma_data: float,
) -> SavedFitSlot:
    """Shared scalar-metric path for baseline / spectrum / 2d."""

    sigma_eff = _compute_sigma_eff(fit_type, selection, sigma_data)
    metrics = compute_fit_metrics(
        observed=observed,
        fit=fit,
        n_free_pars=n_free_pars,
        sigma_eff=sigma_eff if np.isfinite(sigma_eff) else None,
    )
    selection_json = build_selection_json(fit_type, **selection)
    history_key = compute_history_key(
        file_fingerprint=file_fingerprint,
        file_name=file_name,
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
        noise_type=noise_type,
        sigma_source=sigma_source,
        sigma_type=sigma_type,
        sigma_data=float(sigma_data),
        sigma_eff=float(sigma_eff),
        conf_ci=conf_ci,
        mcmc=mcmc,
    )


#
def _per_slice_metrics(
    *,
    observed: np.ndarray,
    fit: np.ndarray,
    n_free_pars: int,
    sigma_eff: float | None = None,
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
    out: dict[str, list[float]] = {k: [] for k in _METRICS_KEYS}
    for i in range(n_slices):
        m = compute_fit_metrics(
            observed=obs[i],
            fit=fit_arr[i],
            n_free_pars=n_free_pars,
            sigma_eff=sigma_eff,
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
    files_group = require_group(files_obj, "files")
    for key in sorted(files_group.keys()):
        fg = require_group(files_group[key], f"files/{key}")
        meta = require_group(fg["metadata"], f"files/{key}/metadata")
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
    slots_group = require_group(slots_obj, "slots")
    for key in sorted(slots_group.keys()):
        slot_group = require_group(slots_group[key], f"slots/{key}")
        meta_obj = slot_group.get("metadata")
        if meta_obj is None:
            continue
        meta = require_group(meta_obj, f"slots/{key}/metadata")
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
_METRICS_KEYS = (
    "chi2_raw",
    "chi2_red_raw",
    "chi2",
    "chi2_red",
    "r2",
    "aic",
    "bic",
)


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
    meta = require_group(meta_obj, "metadata") if meta_obj is not None else None
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
    # Noise metadata snapshot at fit time — see SavedFitSlot docstring.
    meta.attrs["noise_type"] = slot.noise_type
    meta.attrs["sigma_source"] = slot.sigma_source
    meta.attrs["sigma_type"] = slot.sigma_type
    meta.attrs["sigma_data"] = float(slot.sigma_data)
    meta.attrs["sigma_eff"] = float(slot.sigma_eff)
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


#
# --- HDF5 reader ------------------------------------------------------------
#


#
def _attr_str(value: Any) -> str:
    """Normalize an h5py attr value to ``str`` (handles bytes from vlen-str)."""

    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


#
def _to_str_value(value: Any) -> str:
    """Coerce a single vlen-str field/element to ``str``."""

    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


#
def _decode_dataframe(ds: h5py.Dataset) -> pd.DataFrame:
    """
    Inverse of ``_encode_dataframe``.

    Reads the schema's two DataFrame forms back into a ``pd.DataFrame``:

    - **All-numeric form**: 2D ``float64`` dataset with ``columns`` attr.
    - **Heterogeneous form**: 1D structured dataset with positional
      ``c000000``-fields, ``columns`` attr, and ``dtypes`` attr.

    Generic decoder: returns ``""`` / ``NaN`` exactly as stored. The
    schema's ``""`` ↔ ``None`` and ``NaN`` ↔ ``None`` mappings are slot-
    specific (e.g. long-form params ``stderr`` / ``expr``) and applied by
    ``_read_slot``, not here, since other DataFrames (sbs ``params``,
    ``conf_ci``, ``mcmc`` chain/ci) treat the literal values as data.
    """

    columns_attr = ds.attrs["columns"]
    columns = [_to_str_value(c) for c in np.asarray(columns_attr).ravel()]

    if ds.dtype.fields is None:
        values = ds[...]
        return pd.DataFrame(values, columns=columns)

    arr = ds[...]
    type_tags = [_to_str_value(t) for t in np.asarray(ds.attrs["dtypes"]).ravel()]
    field_keys = [f"c{i:06d}" for i in range(len(columns))]
    cols_data: dict[str, list[Any]] = {}
    for col_label, key, tag in zip(columns, field_keys, type_tags, strict=True):
        col = arr[key]
        if tag == "str":
            cols_data[col_label] = [_to_str_value(v) for v in col]
        elif tag == "float64":
            cols_data[col_label] = [float(v) for v in col]
        elif tag == "bool":
            cols_data[col_label] = [bool(v) for v in col]
        else:
            raise ValueError(f"unknown column dtype tag {tag!r} on {ds.name}")
    return pd.DataFrame(cols_data, columns=columns)


#
def _restore_long_params_nones(params: pd.DataFrame) -> None:
    """
    Map ``NaN`` → ``None`` for ``stderr`` and ``""`` → ``None`` for ``expr``,
    in place, on a long-form params DataFrame.

    Mirrors the writer's encoding (``_pack_for_dtype``) so a round-trip
    matches what ``utils/lmfit.py:par_to_df(..., col_type="min")`` produced
    in-session: lmfit yields ``stderr=None`` when uncomputed and
    ``expr=None`` when no expression is set.
    """

    if "stderr" in params.columns:
        params["stderr"] = [None if pd.isna(v) else v for v in params["stderr"]]
    if "expr" in params.columns:
        params["expr"] = [None if v == "" else v for v in params["expr"]]


#
def _read_metrics_per_slice(ds: h5py.Dataset) -> dict[str, np.ndarray]:
    """Decode the sbs ``metrics_per_slice`` structured dataset."""

    arr = ds[...]
    return {k: np.asarray(arr[k], dtype=np.float64) for k in _METRICS_KEYS}


#
def _read_mcmc_group(group: h5py.Group) -> dict[str, Any]:
    """
    Inverse of ``_write_mcmc_group``.

    Returns ``{"flatchain", "ci", "lnsigma"}`` matching the writer's
    payload. ``lnsigma`` NaN maps back to ``None``; ``ci`` is ``None`` if
    the optional dataset was not written.
    """

    flatchain_obj = group.get("flatchain")
    if flatchain_obj is None:
        flatchain: pd.DataFrame | None = None
    else:
        flatchain = _decode_dataframe(require_dataset(flatchain_obj, "mcmc/flatchain"))
    ci_obj = group.get("ci")
    ci = (
        _decode_dataframe(require_dataset(ci_obj, "mcmc/ci"))
        if ci_obj is not None
        else None
    )
    lnsigma_attr = group.attrs.get("lnsigma")
    lnsigma: float | None
    if lnsigma_attr is None:
        lnsigma = None
    else:
        v = float(np.asarray(lnsigma_attr).item())
        lnsigma = None if np.isnan(v) else v
    return {"flatchain": flatchain, "ci": ci, "lnsigma": lnsigma}


#
def _read_slot(
    slot_group: h5py.Group,
    *,
    file_fingerprint: dict[str, Any],
    file_name: str,
) -> SavedFitSlot:
    """Decode one slot group into a ``SavedFitSlot``."""

    meta = require_group(slot_group["metadata"], "metadata")
    a = meta.attrs
    fit_type = cast(FitType, _attr_str(a["fit_type"]))
    selection_json = _attr_str(a["selection_json"])
    model_name = _attr_str(a["model_name"])

    params = _decode_dataframe(require_dataset(slot_group["params"], "params"))
    if fit_type != "sbs":
        # Restore the schema's "" ↔ None / NaN ↔ None mappings for long-form
        # params. sbs params is wide-form numeric and carries no None
        # semantics, so this only applies to baseline / spectrum / 2d.
        _restore_long_params_nones(params)
    observed = np.asarray(require_dataset(slot_group["observed"], "observed")[...])
    fit_arr = np.asarray(require_dataset(slot_group["fit"], "fit")[...])

    metrics: dict[str, Any]
    if fit_type == "sbs":
        metrics = _read_metrics_per_slice(
            require_dataset(slot_group["metrics_per_slice"], "metrics_per_slice")
        )
    else:
        metrics = {k: float(np.asarray(a[k]).item()) for k in _METRICS_KEYS}

    conf_ci_obj = slot_group.get("conf_ci")
    conf_ci = (
        _decode_dataframe(require_dataset(conf_ci_obj, "conf_ci"))
        if conf_ci_obj is not None
        else None
    )
    mcmc_obj = slot_group.get("mcmc")
    mcmc = (
        _read_mcmc_group(require_group(mcmc_obj, "mcmc"))
        if mcmc_obj is not None
        else None
    )

    yaml_filename = _attr_str(a["yaml_filename"]) if "yaml_filename" in a else None
    selection = json.loads(selection_json)

    # history_key is recomputed per schema; on-disk value is debug-only.
    history_key = compute_history_key(
        file_fingerprint=file_fingerprint,
        file_name=file_name,
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
        observed_sha256=_attr_str(a["observed_sha256"]),
        history_key=history_key,
        params=params,
        metrics=metrics,
        observed=observed,
        fit=fit_arr,
        fit_alg=_attr_str(a["fit_alg"]),
        yaml_filename=yaml_filename,
        timestamp=_attr_str(a["timestamp"]),
        noise_type=_attr_str(a["noise_type"]),
        sigma_source=_attr_str(a["sigma_source"]),
        sigma_type=_attr_str(a["sigma_type"]),
        sigma_data=float(np.asarray(a["sigma_data"]).item()),
        sigma_eff=float(np.asarray(a["sigma_eff"]).item()),
        conf_ci=conf_ci,
        mcmc=mcmc,
    )


#
def _read_file(file_group: h5py.Group) -> SavedFile:
    """Decode one file group into a ``SavedFile``."""

    meta = require_group(file_group["metadata"], "metadata")
    a = meta.attrs
    name = _attr_str(a["name"])
    original_path = _attr_str(a["original_path"])
    dim = int(np.asarray(a["dim"]).item())
    shape = tuple(int(x) for x in np.asarray(a["shape"]).ravel())
    fingerprint: dict[str, Any] = {
        "data_sha256": _attr_str(a["data_sha256"]),
        "energy_sha256": _attr_str(a["energy_sha256"]),
        "time_sha256": _attr_str(a["time_sha256"]),
        "shape": shape,
    }
    e_lim = [int(x) for x in np.asarray(a["e_lim"]).ravel()] if "e_lim" in a else None
    t_lim = [int(x) for x in np.asarray(a["t_lim"]).ravel()] if "t_lim" in a else None

    data = np.asarray(require_dataset(file_group["data"], "data")[...])
    energy = np.asarray(require_dataset(file_group["energy"], "energy")[...])
    time = np.asarray(require_dataset(file_group["time"], "time")[...])

    slot_records: list[SavedFitSlot] = []
    slots_obj = file_group.get("slots")
    if slots_obj is not None:
        slots_group = require_group(slots_obj, "slots")
        for key in sorted(slots_group.keys()):
            sg = require_group(slots_group[key], f"slots/{key}")
            slot_records.append(
                _read_slot(sg, file_fingerprint=fingerprint, file_name=name)
            )

    return SavedFile(
        name=name,
        original_path=original_path,
        dim=dim,
        shape=shape,
        fingerprint=fingerprint,
        data=data,
        energy=energy,
        time=time,
        e_lim=e_lim,
        t_lim=t_lim,
        slots=tuple(slot_records),
    )


#
def read_archive(filepath: PathLike | str) -> SavedProject:
    """
    Deserialize an HDF5 fit archive into a ``SavedProject``.

    Inverse of ``write_archive``. Does not touch any live ``Project``,
    ``File``, or ``Model`` state — the returned ``SavedProject`` is a
    standalone, immutable view of the archive's contents at read time.

    Raises ``ValueError`` if ``schema_version`` does not match the
    reader's ``SCHEMA_VERSION``.
    """

    path = Path(filepath)
    with h5py.File(path, "r") as archive:
        meta = require_group(archive["metadata"], "metadata")
        ma = meta.attrs
        schema_version = _attr_str(ma["schema_version"])
        if schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Archive at {path} has schema_version {schema_version!r}; "
                f"this reader supports {SCHEMA_VERSION!r}."
            )
        files_obj = archive.get("files")
        files: list[SavedFile] = []
        if files_obj is not None:
            files_group = require_group(files_obj, "files")
            for key in sorted(files_group.keys()):
                fg = require_group(files_group[key], f"files/{key}")
                files.append(_read_file(fg))

        return SavedProject(
            name=_attr_str(ma["project_name"]),
            trspecfit_version=_attr_str(ma["trspecfit_version"]),
            schema_version=schema_version,
            timestamp_created=_attr_str(ma["timestamp_created"]),
            timestamp_updated=_attr_str(ma["timestamp_updated"]),
            files=tuple(files),
        )


#
# --- CSV / PNG export -------------------------------------------------------
#


#
def _slot_axes(
    slot: SavedFitSlot,
    saved_file: SavedFile,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return ``(energy, time)`` axes matching the slot's grid.

    The slot's ``observed`` / ``fit`` are stored on the cropped fit grid
    (``e_lim`` / ``t_lim`` applied per fit type). This rebuilds the
    matching axes from the parent ``SavedFile`` so CSV outputs and 2D
    plots line up with the array shapes.
    """

    energy = np.asarray(saved_file.energy)
    time = np.asarray(saved_file.time)
    e_lim = slot.selection.get("e_lim")
    if e_lim:
        energy = energy[int(e_lim[0]) : int(e_lim[1])]
    if slot.fit_type == "2d":
        t_lim = slot.selection.get("t_lim")
        if t_lim:
            time = time[int(t_lim[0]) : int(t_lim[1])]
    return energy, time


#
def _slot_dir_name(slot: SavedFitSlot, suffix_with_hash: bool) -> str:
    """
    Output-directory name for one slot.

    Default form is ``{model_name}__{fit_type}``. When the same
    ``(file, model, fit_type)`` triple appears more than once in the
    snapshot (different selections), all of its slots get an
    ``__{history_key[:8]}`` suffix so each lands in a distinct directory.
    """

    base = f"{slot.model_name}__{slot.fit_type}"
    if suffix_with_hash:
        base = f"{base}__{slot.history_key[:8]}"
    return base


#
def _resolve_export_dirs(
    saved_files: Sequence[SavedFile],
    root: Path,
) -> dict[int, Path]:
    """
    Map ``id(slot) -> output directory`` for every slot in ``saved_files``.

    Two-tier disambiguation:

    1. **Across files:** when two or more ``SavedFile`` records share a
       ``name``, every entry in the colliding group gets a positional
       ordinal suffix (``__000``, ``__001``, ...) keyed by its position
       within ``saved_files``. Ordinals are unique even for byte-identical
       ``SavedFile`` records, so a content-hash suffix would not be
       sufficient — two records with identical fingerprint *and*
       ``original_path`` would still collide.
    2. **Within a file:** ``(model_name, fit_type)`` collisions get the
       slot's ``history_key[:8]`` suffix on the slot directory.

    Together these guarantee every slot resolves to a unique path, so the
    pre-check / overwrite logic can rely on path identity == slot identity.
    """

    name_indices: dict[str, list[int]] = {}
    for i, sf in enumerate(saved_files):
        name_indices.setdefault(sf.name, []).append(i)

    file_dir_for: dict[int, Path] = {}
    for indices in name_indices.values():
        if len(indices) == 1:
            i = indices[0]
            file_dir_for[i] = root / saved_files[i].name
        else:
            for ordinal, i in enumerate(indices):
                file_dir_for[i] = root / f"{saved_files[i].name}__{ordinal:03d}"

    out: dict[int, Path] = {}
    for i, sf in enumerate(saved_files):
        file_dir = file_dir_for[i]
        groups: dict[tuple[str, str], list[SavedFitSlot]] = {}
        for slot in sf.slots:
            groups.setdefault((slot.model_name, slot.fit_type), []).append(slot)
        for slots in groups.values():
            need_hash = len(slots) > 1
            for slot in slots:
                out[id(slot)] = file_dir / _slot_dir_name(slot, need_hash)
    return out


#
def _precheck_export_collisions(
    slot_dirs: dict[int, Path],
    overwrite: bool,
) -> None:
    """
    Refuse to start the export if any target directory already has content.

    Mirrors the pre-check in ``write_archive``: collect every conflict
    before mutating the filesystem so a single blocker does not leave a
    half-written tree. Empty directories are tolerated.
    """

    if overwrite:
        return
    conflicts: list[Path] = []
    for path in slot_dirs.values():
        if path.exists() and any(path.iterdir()):
            conflicts.append(path)
    if conflicts:
        joined = "\n  ".join(str(p) for p in conflicts)
        raise FileExistsError(
            f"export_fits: {len(conflicts)} target director"
            f"{'y' if len(conflicts) == 1 else 'ies'} already exist and are "
            f"non-empty. Pass overwrite=True to replace, or choose a fresh "
            f"root path. Conflicts:\n  {joined}"
        )


#
def _clear_directory(path: Path) -> None:
    """Remove every entry under ``path`` (one level deep is sufficient)."""

    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            for sub in child.rglob("*"):
                if sub.is_file() or sub.is_symlink():
                    sub.unlink()
            for sub in sorted(
                (p for p in child.rglob("*") if p.is_dir()),
                key=lambda p: len(p.parts),
                reverse=True,
            ):
                sub.rmdir()
            child.rmdir()
        else:
            child.unlink()


#
def _write_csv_array(
    path: Path,
    array: np.ndarray,
    *,
    num_fmt: str,
    delim: str,
) -> None:
    """``np.savetxt`` with the project's number format and delimiter."""

    np.savetxt(path, np.asarray(array), fmt=num_fmt, delimiter=delim)


#
def _write_csv_dataframe(
    path: Path,
    df: pd.DataFrame,
    *,
    num_fmt: str,
    delim: str,
    index: bool = False,
) -> None:
    """``pd.DataFrame.to_csv`` with project formatting defaults."""

    df.to_csv(path, index=index, float_format=num_fmt, sep=delim)


#
def _metrics_to_dataframe(metrics: dict[str, Any]) -> pd.DataFrame:
    """
    Render a slot's ``metrics`` dict to a tidy DataFrame.

    Scalar metrics → single-row DataFrame with one column per metric. SbS
    metrics (per-slice arrays) → multi-row DataFrame indexed by slice.
    """

    sample = next(iter(metrics.values()))
    if isinstance(sample, np.ndarray):
        n = len(sample)
        df = pd.DataFrame({"slice": np.arange(n)})
        for key in _METRICS_KEYS:
            if key in metrics:
                df[key] = np.asarray(metrics[key])
        return df
    return pd.DataFrame(
        {key: [float(metrics[key])] for key in _METRICS_KEYS if key in metrics}
    )


#
def _export_1d_slot(
    slot: SavedFitSlot,
    saved_file: SavedFile,
    slot_dir: Path,
    *,
    num_fmt: str,
    delim: str,
) -> None:
    """Write CSVs for a 1D slot (baseline / spectrum)."""

    energy, _ = _slot_axes(slot, saved_file)
    fit_1d = pd.DataFrame(
        {
            "energy": energy,
            "observed": np.asarray(slot.observed),
            "fit": np.asarray(slot.fit),
            "residual": np.asarray(slot.observed) - np.asarray(slot.fit),
        }
    )
    _write_csv_dataframe(slot_dir / "fit_1d.csv", fit_1d, num_fmt=num_fmt, delim=delim)


#
def _export_2d_slot(
    slot: SavedFitSlot,
    saved_file: SavedFile,
    slot_dir: Path,
    *,
    num_fmt: str,
    delim: str,
    plot_config: Any,
) -> None:
    """Write CSVs and the data/fit/residual map PNG for a 2D slot."""

    energy, time = _slot_axes(slot, saved_file)
    _write_csv_array(
        slot_dir / "fit_2d.csv", np.asarray(slot.fit), num_fmt=num_fmt, delim=delim
    )
    _write_csv_array(
        slot_dir / "observed_2d.csv",
        np.asarray(slot.observed),
        num_fmt=num_fmt,
        delim=delim,
    )
    _write_csv_array(slot_dir / "energy.csv", energy, num_fmt=num_fmt, delim=delim)
    _write_csv_array(slot_dir / "time.csv", time, num_fmt=num_fmt, delim=delim)
    plt_fit_res_2d(
        data=np.asarray(slot.observed),
        fit=np.asarray(slot.fit),
        x=energy,
        y=time,
        config=plot_config,
        save_img=-1,  # save without display; bulk export should not pop figures
        save_path=slot_dir,
    )


#
def _export_sbs_param_evolution(
    slot: SavedFitSlot,
    saved_file: SavedFile,
    slot_dir: Path,
    *,
    num_fmt: str,
    delim: str,
    plot_config: Any,
) -> None:
    """
    Write ``fit_pars.csv`` (per-slice param values) and per-parameter PNGs.

    Mirrors ``fitlib.results_to_df``'s output shape: columns are
    ``[index, time, par1, par2, ...]``. Per-parameter PNGs are emitted
    only for parameters that varied at fit time (vary=True).
    """

    params_per_slice = slot.params
    _, time = _slot_axes(slot, saved_file)
    n_slices = len(params_per_slice)
    fit_pars = params_per_slice.copy()
    time_label = (
        getattr(plot_config, "y_label", "time") if plot_config is not None else "time"
    )
    fit_pars.insert(0, time_label, np.asarray(time)[:n_slices])
    fit_pars.insert(0, "index", np.arange(n_slices))
    _write_csv_dataframe(
        slot_dir / "fit_pars.csv", fit_pars, num_fmt=num_fmt, delim=delim
    )

    par_cols = list(params_per_slice.columns)
    if not par_cols:
        return
    plt_fit_res_pars(
        df=params_per_slice.loc[:, par_cols],
        x=np.asarray(time)[:n_slices],
        config=plot_config,
        save_img=-1,
        save_path=slot_dir,
    )


#
def _export_slot(
    slot: SavedFitSlot,
    saved_file: SavedFile,
    slot_dir: Path,
    *,
    num_fmt: str,
    delim: str,
    plot_config: Any,
) -> None:
    """Write one slot's CSV/PNG payload into ``slot_dir`` (must exist)."""

    _write_csv_dataframe(
        slot_dir / "params.csv", slot.params, num_fmt=num_fmt, delim=delim
    )
    metrics_df = _metrics_to_dataframe(slot.metrics)
    metrics_filename = (
        "metrics_per_slice.csv" if slot.fit_type == "sbs" else "metrics.csv"
    )
    _write_csv_dataframe(
        slot_dir / metrics_filename, metrics_df, num_fmt=num_fmt, delim=delim
    )
    if slot.conf_ci is not None and not slot.conf_ci.empty:
        _write_csv_dataframe(
            slot_dir / "conf_ci.csv", slot.conf_ci, num_fmt=num_fmt, delim=delim
        )
    if slot.mcmc is not None:
        mcmc_dir = slot_dir / "mcmc"
        mcmc_dir.mkdir(parents=True, exist_ok=True)
        flatchain = slot.mcmc.get("flatchain")
        if isinstance(flatchain, pd.DataFrame) and not flatchain.empty:
            _write_csv_dataframe(
                mcmc_dir / "flatchain.csv", flatchain, num_fmt=num_fmt, delim=delim
            )
        ci = slot.mcmc.get("ci")
        if isinstance(ci, pd.DataFrame) and not ci.empty:
            _write_csv_dataframe(mcmc_dir / "ci.csv", ci, num_fmt=num_fmt, delim=delim)

    if slot.fit_type in ("baseline", "spectrum"):
        _export_1d_slot(slot, saved_file, slot_dir, num_fmt=num_fmt, delim=delim)
    elif slot.fit_type == "2d":
        _export_2d_slot(
            slot,
            saved_file,
            slot_dir,
            num_fmt=num_fmt,
            delim=delim,
            plot_config=plot_config,
        )
    elif slot.fit_type == "sbs":
        _export_2d_slot(
            slot,
            saved_file,
            slot_dir,
            num_fmt=num_fmt,
            delim=delim,
            plot_config=plot_config,
        )
        _export_sbs_param_evolution(
            slot,
            saved_file,
            slot_dir,
            num_fmt=num_fmt,
            delim=delim,
            plot_config=plot_config,
        )
    else:
        raise ValueError(f"unsupported fit_type for export: {slot.fit_type!r}")


#
def _resolve_plot_config(
    plot_config: Any,
    file_name: str,
) -> Any:
    """
    Pick the ``PlotConfig`` for a single ``SavedFile``.

    Accepts a ``PlotConfig`` (used for every file), a ``dict`` keyed by
    ``SavedFile.name`` (per-file lookup with default fallback for missing
    keys), or ``None`` (default ``PlotConfig`` for everything).
    """

    if isinstance(plot_config, dict):
        cfg = plot_config.get(file_name)
        if cfg is not None:
            return cfg
        plot_config = None
    if plot_config is None:
        from trspecfit.config.plot import PlotConfig

        return PlotConfig()
    return plot_config


#
def write_csv_export(
    root: PathLike | str,
    *,
    project: SavedProject,
    num_fmt: str = "%.6e",
    delim: str = ",",
    plot_config: Any = None,
    overwrite: bool = False,
) -> int:
    """
    Serialize a ``SavedProject`` to a CSV/PNG export tree.

    Layout: ``<root>/<file_name>/<model_name>__<fit_type>[__<hash>]/``.
    The ``__<hash>`` suffix appears only when more than one slot shares
    the ``(file, model, fit_type)`` triple (i.e. multiple selections in
    the snapshot); the suffix is the first 8 chars of ``history_key``.
    When two ``SavedFile`` records share a ``name``, every entry in the
    colliding group gets a positional ordinal suffix (``__000``,
    ``__001``, ...) so byte-identical records (same fingerprint *and*
    ``original_path``) still resolve to distinct directories.

    Parameters
    ----------
    root : path
        Output directory; created if missing.
    project : SavedProject
        Already filtered + collapsed by the caller (see
        ``Project.export_fits``).
    num_fmt, delim : str
        Number format and delimiter for ``np.savetxt`` /
        ``DataFrame.to_csv``.
    plot_config : PlotConfig | dict[str, PlotConfig] | None
        Drives PNG styling.

        - ``PlotConfig`` — applied to every file.
        - ``dict[file_name, PlotConfig]`` — per-file lookup; missing keys
          fall back to a default ``PlotConfig``.
        - ``None`` — default ``PlotConfig`` for all files.

        ``Project.export_fits`` builds the dict form by reading each live
        ``File.plot_config``, so per-file styling is preserved.
    overwrite : bool, default False
        Per-slot directory: a non-empty target dir raises
        ``FileExistsError`` unless True. Pre-checked across all slots
        before any writes.

    Returns
    -------
    int
        Number of slot directories written.
    """

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    slot_dirs = _resolve_export_dirs(project.files, root_path)
    _precheck_export_collisions(slot_dirs, overwrite=overwrite)

    written_paths: set[Path] = set()
    n_written = 0
    for sf in project.files:
        sf_plot_config = _resolve_plot_config(plot_config, sf.name)
        for slot in sf.slots:
            slot_dir = slot_dirs[id(slot)]
            if slot_dir in written_paths:
                # _resolve_export_dirs is supposed to return a unique path
                # per slot; this asserts the invariant rather than silently
                # overwriting earlier slots' output (the bug fixed by the
                # SavedFile-name disambiguation above).
                raise RuntimeError(
                    f"export_fits internal error: two slots resolved to the "
                    f"same output directory {slot_dir!s}. Please report."
                )
            written_paths.add(slot_dir)
            if slot_dir.exists() and any(slot_dir.iterdir()):
                # overwrite=True path; pre-check has already ruled out the
                # overwrite=False case.
                _clear_directory(slot_dir)
            slot_dir.mkdir(parents=True, exist_ok=True)
            _export_slot(
                slot,
                sf,
                slot_dir,
                num_fmt=num_fmt,
                delim=delim,
                plot_config=sf_plot_config,
            )
            n_written += 1
    return n_written
