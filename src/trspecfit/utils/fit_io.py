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
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from trspecfit.fitlib import compute_fit_metrics

FitType = Literal["baseline", "spectrum", "sbs", "2d"]


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
