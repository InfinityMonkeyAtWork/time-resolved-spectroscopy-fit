"""
Fit-result persistence: dataclasses, identity helpers, slot extractors.

This module owns the data model for completed fits. ``SavedFitSlot`` is the
first-class owner of ``observed`` / ``fit`` / ``metrics`` and the identity
fields (``handle``, ``optimization_hash``, ``fit_view_sha256``) — neither
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

import copy
import datetime
import hashlib
import json
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

import h5py
import numpy as np
import pandas as pd

from trspecfit.config.plot import PlotConfig
from trspecfit.fitlib import (
    compute_fit_metrics,
    plt_fit_res_2d,
    plt_fit_res_pars,
)
from trspecfit.utils import lmfit as ulmfit
from trspecfit.utils.hdf5 import require_dataset, require_group
from trspecfit.utils.lmfit import MCMCResult

PathLike = str | Path
FitType = Literal["baseline", "spectrum", "sbs", "2d"]
SCHEMA_VERSION = "7"
# Schema 7 is a clean break (identity-hashed slots under project/, raw-data
# file groups, per-slot corrections, joint sidecar): pre-7 archives are not
# readable — re-fit and re-save. See docs/design/fit_archive_schema.md.
SUPPORTED_READ_VERSIONS = ("7",)

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
#
class ModelYamlRecord(NamedTuple):
    """
    One YAML snippet of a fitted model's provenance.

    Stored on ``SavedFitSlot.model_yaml``, one record per snippet — the
    energy model, each dynamics attachment (one per element of a
    multi-cycle sequence), each profile attachment.

    Attributes
    ----------
    role : str
        ``"energy"`` | ``"dynamics"`` | ``"profile"``.
    name : str
        The YAML top-level key.
    source_file : str
        YAML filename the snippet came from. Snippets of one fit may span
        several files.
    target_par : str | None
        Parameter the attachment binds to; ``None`` for the energy model.
    sequence_index : int | None
        Position in a multi-cycle dynamics sequence (0 is the global
        element); ``None`` for energy and profile snippets.
    text : str
        Verbatim slice of the source file at the key's top-level
        boundaries (``uparsing.dump_yaml_subtrees``) — comments and
        ordering preserved byte-for-byte. Not a parser round-trip: the
        numbering feature allows duplicate component keys, which YAML
        loaders reject or silently collapse.
    """

    role: str
    name: str
    source_file: str
    target_par: str | None
    sequence_index: int | None
    text: str


#
@dataclass(frozen=True)
class SavedFitSlot:
    """
    One completed fit result for a single file view.

    Immutable after construction. Built once at fit completion by
    ``_slot_from_<fit_type>`` and appended to ``Project._fit_history``.
    Identity is ``handle`` (fit_archive_principles.md, Principle 3);
    equal handles mean the same optimization on the same file.

    Attributes
    ----------
    handle : str
        The slot's primary key: ``compute_slot_handle`` over
        ``(optimization_hash, file_name)``. Joint-bundle siblings share
        one ``optimization_hash`` and get distinct handles through
        ``file_name``.
    optimization_hash : str
        Identity of the optimization that produced this slot —
        ``compute_optimization_hash`` over ``input_files``, ``fit_type``,
        ``model_structure``, the parameter table, the quantized initial
        state, and the optimizer settings.
    input_files : str
        Canonical ``encode_input_files`` JSON: scope (``"file"`` or
        ``"project"``) plus one ``(file_name, version_stamp,
        selection_json)`` entry per participating file, sorted by file
        name. The version stamp folds the correction state (``dark``,
        ``calibration``) into fit identity.
    model_structure : str
        Canonical ``encode_model_structure`` JSON: top-level YAML model
        names per file, dynamics attachments as flat ordered
        submodel-name tuples (order assigns subcycles), exact-text
        frequency.
    fit_view_sha256 : str
        Comparability hash of the exact fit view (observed + axes) —
        ``compute_fit_view_sha256``. Deliberately **not** an input to
        ``optimization_hash``: it is the independent cross-check that
        metric comparisons only happen across identical views.
    file_name : str
        The file's identity (``File.name`` — unique within the Project and
        guarded against reassignment).
    model_name : str
    fit_type : {"baseline", "spectrum", "sbs", "2d"}
    selection : dict
        Fit-view identity. Shape depends on ``fit_type``:

        - baseline: ``{"base_t_ind", "e_lim"}``
        - spectrum: ``{"time_point", "time_range", "time_type", "e_lim"}``
        - sbs:      ``{"e_lim", "t_lim"}``
        - 2d:       ``{"e_lim", "t_lim"}``

    selection_json : str
        Deterministic JSON of ``selection`` (sorted keys); enters
        ``optimization_hash`` through this file's ``input_files`` entry,
        so refits with different selections do not collide.
    params : pd.DataFrame
        ``[name, value, init_value, stderr, min, max, vary, expr]``. For SbS,
        a per-slice DataFrame (one row per slice, columns are param values).
    metrics : dict
        ``{"chi2_raw", "chi2_red_raw", "chi2", "chi2_red", "r2", "aic",
        "bic"}``. Scalar floats for baseline/spectrum/2d. For SbS, each
        value is a 1D ``np.ndarray`` of length ``n_slices``. ``chi2_raw``
        and ``chi2_red_raw`` are the unweighted lmfit-convention
        diagnostics. ``chi2`` and ``chi2_red`` are the σ-calibrated
        versions (``≈ 1`` for a fit at the noise floor) and are ``NaN``
        when no sigma was supplied at fit time. For the per-file 2d
        projections of a project-level joint fit, the count-dependent
        metrics ``chi2_red_raw`` / ``chi2_red`` / ``aic`` / ``bic`` are
        ``NaN`` — the joint parameter count does not decompose by file
        (the joint record owns the whole-objective values).
    observed : np.ndarray
        Data view that was fit against (cropped to ``e_lim`` / ``t_lim`` where
        applicable). ``observed.shape == fit.shape`` always.
    fit : np.ndarray
        Model evaluated at final params on the same grid as ``observed``.
    fit_alg : str
        Optimizer name (e.g. ``"Nelder"``, ``"leastsq"``). For two-stage fits,
        the final stage's algorithm.
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
    dark : np.ndarray | None
        Dark/background spectrum in force on the file at fit time, shape
        ``(n_energy,)``; ``None`` when no dark correction was applied.
        Per-slot (not per-file) because corrections are fit-time state:
        two slots on one file may have consumed different correction
        states.
    calibration : np.ndarray | None
        Sensitivity-calibration spectrum in force at fit time, shape
        ``(n_energy,)``; ``None`` when no calibration was applied.
    model_yaml : tuple[ModelYamlRecord, ...] | None
        YAML snippet provenance of the fitted model, one record per
        snippet (see ``ModelYamlRecord``). ``None`` for a model built
        programmatically with no YAML source.
    label : str | None
        User-facing label. On disk a mutable attr — settable on an
        existing archive without rewriting the slot. ``None`` when never
        set.
    joint_ref : str | None
        ``optimization_hash`` of the owning ``JointFitResult`` — a
        content-addressed reference, never a positional path. Present
        exactly when ``input_files`` scope is ``"project"`` (a one-file
        project fit included).
    params_meta : pd.DataFrame | None
        SbS only: shared per-parameter metadata ``[name, vary, min, max,
        expr]`` — the columns that are slice-invariant by construction
        (one model, one vary set for every slice). ``None`` for other fit
        types, whose long-form ``params`` already carry these.
    params_stderr : pd.DataFrame | None
        SbS only: per-slice parameter standard errors, same shape/columns
        as the wide ``params`` frame; ``NaN`` where the optimizer reported
        none. ``None`` for other fit types (long-form ``params`` has a
        ``stderr`` column).
    fit_settings : dict | None
        Optimizer-configuration provenance: ``{"stages", "fit_alg_1",
        "fit_alg_2", "try_ci"}``, plus ``{"seed_source", "seed_adapt",
        "seed_values"}`` for SbS and an ``"mc"`` sub-dict (steps, walkers,
        burn, thin, ntemps, is_weighted, sigma bounds) when MCMC was
        enabled. Deliberately excludes execution details that cannot
        change the result (worker counts). The result-shaping core
        (stages, per-stage methods, backend, seed, Jacobian) also enters
        ``optimization_hash`` via ``encode_optimizer_settings``; the
        post-fit analysis knobs (``try_ci``, ``mc``) do not — rerunning
        CI or MCMC enriches the same fit rather than minting a new one.
    conf_ci : pd.DataFrame | None
    correl : pd.DataFrame | None
        Varying-parameter correlation matrix (index == columns == varying
        parameter names, 1.0 on the diagonal). ``None`` when the optimizer
        reported no covariance (e.g. Nelder without numdifftools) and for
        project-level joint fits, whose joint covariance does not decompose
        per file. For SbS, captured from slice 0 (the representative slice,
        mirroring ``conf_ci`` / ``mcmc``).
    mcmc : dict | None
        ``{"flatchain", "ci", "lnsigma", "acceptance_fraction"}`` if MCMC
        ran, else ``None``. ``acceptance_fraction`` is emcee's per-walker
        array.
    components : np.ndarray | None
        Per-component fit curves on the same grid as ``fit``, evaluated at
        final params. ``None`` for ``fit_type == "2d"`` (no per-component
        concept there).
        Shape ``(n_components, energy_in_lim)`` for baseline/spectrum;
        ``(n_slices, n_components, energy_in_lim)`` for sbs.
    component_names : list of str | None
        Component labels, same order as ``components``' component axis —
        ``[comp.name for comp in model.components]`` at fit time. Persisted
        explicitly rather than parsed from ``params.name`` because a
        static (``dim == 1``) attached ``par_profile`` splices a nested
        model's parameters into its host component's name block without
        adding a distinct ``model.components`` entry, breaking any
        prefix-based re-derivation. ``None`` exactly when ``components``
        is ``None``.
    fit_ini : np.ndarray | None
        Model evaluated at the true pre-fit seed (``FitOutput.par_ini``),
        on the same grid as ``fit``. ``None`` on the project-level
        joint-fit path (no per-file ``par_ini`` there). Shape matches
        ``fit``: for sbs,
        ``(n_slices, energy_in_lim)`` (every slice, not just slice 0 —
        the per-slice seed is already available at slot-construction
        time at no extra cost).
    params_init : pd.DataFrame | None
        SbS only: per-slice true initial-guess values, same shape/columns
        as the wide ``params`` frame's value columns (mirrors
        ``params_stderr``). ``None`` for other fit types (long-form
        ``params`` already has an ``init_value`` column).
    """

    handle: str
    optimization_hash: str
    input_files: str
    model_structure: str
    fit_view_sha256: str
    file_name: str
    model_name: str
    fit_type: FitType
    selection: dict[str, Any]
    selection_json: str
    params: pd.DataFrame
    metrics: dict[str, Any]
    observed: np.ndarray
    fit: np.ndarray
    fit_alg: str
    timestamp: str
    noise_type: str
    sigma_source: str
    sigma_type: str
    sigma_data: float
    sigma_eff: float
    dark: np.ndarray | None = None
    calibration: np.ndarray | None = None
    model_yaml: tuple[ModelYamlRecord, ...] | None = None
    label: str | None = None
    joint_ref: str | None = None
    conf_ci: pd.DataFrame | None = None
    correl: pd.DataFrame | None = None
    mcmc: dict[str, Any] | None = None
    params_meta: pd.DataFrame | None = None
    params_stderr: pd.DataFrame | None = None
    fit_settings: dict[str, Any] | None = None
    components: np.ndarray | None = None
    component_names: list[str] | None = None
    fit_ini: np.ndarray | None = None
    params_init: pd.DataFrame | None = None


#
@dataclass(frozen=True)
class SavedFile:
    """
    Archive-side container for a single file's raw data, identity, and slots.

    Used by both writer and reader. The writer assembles ``SavedFile``
    records from a Project + filtered slot list before serializing; the
    reader returns them as the contents of the loaded archive. All arrays
    are copies with the write flag cleared — the ownership boundary that
    makes the record a snapshot rather than a view (Principle 4). The
    payload is captured once, when the file produces its first slot.

    Attributes
    ----------
    name : str
        ``File.name`` — file identity within the archive
        (fit_archive_principles.md, Principle 1). Write-side, an incoming
        file matching an existing record by name but differing in
        ``file_content_hash`` raises; read-side, a hash difference is
        reportable staleness, not a lookup failure.
    original_path : str
        Absolute path of the source data file at save time. May not exist
        on the loading machine; never participates in matching.
    dim : int
        1 or 2.
    shape : tuple[int, ...]
        ``data_raw.shape``.
    file_content_hash : str
        ``compute_file_content_hash`` over dtype, shape, ``data_raw``,
        ``energy``, ``time``, ``aux_axis`` (an absent axis contributes a
        fixed sentinel, distinguishable from a zero-length one).
    data_raw : np.ndarray
        The raw data as loaded, before any correction. Corrected data is
        reconstructed on demand from a slot's ``dark`` / ``calibration``.
    energy : np.ndarray
    time : np.ndarray
        Empty array for 1D files.
    slots : tuple[SavedFitSlot, ...]
        Slots belonging to this file. Tuple (not list) to keep the record
        immutable; the writer accumulates slots into a list and freezes
        on construction.
    aux_axis : np.ndarray | None
        Auxiliary physical axis (``File.aux_axis``, e.g. depth) for
        ``par_profile``-attached models. ``None`` when the file has none.
    """

    name: str
    original_path: str
    dim: int
    shape: tuple[int, ...]
    file_content_hash: str
    data_raw: np.ndarray
    energy: np.ndarray
    time: np.ndarray
    slots: tuple[SavedFitSlot, ...]
    aux_axis: np.ndarray | None = None


#
@dataclass(frozen=True)
class SavedProject:
    """
    Top-level archive container.

    The writer takes a ``SavedProject`` and serializes it to HDF5; the
    reader does the inverse. Callers typically build via
    ``Project._build_saved_project_from_history`` rather than
    instantiating directly.

    Attributes
    ----------
    name : str
        Project name. Written once; an append under a different project
        name raises.
    trspecfit_version : str
    schema_version : str
        Currently ``"7"``. Bumped on incompatible schema changes.
    timestamp_created : str
        ISO 8601 UTC; first archive-write time.
    timestamp_updated : str
        ISO 8601 UTC; most recent archive-write time. Equal to
        ``timestamp_created`` on the initial save.
    plot_config : PlotConfig
        The project's rendering configuration; round-trips via
        ``PlotConfig.to_json`` / ``from_json`` and is rewritten on every
        save. One rule: a ``FitResults`` renders with its project's
        config.
    files : tuple[SavedFile, ...]
    joint : tuple[JointFitResult, ...]
        Joint optimization records, one per ``Project.fit_2d`` run. Each
        record's projections hold the same slot objects stored under
        ``files``; every projection slot's ``joint_ref`` equals its
        record's ``optimization_hash``.
    """

    name: str
    trspecfit_version: str
    schema_version: str
    timestamp_created: str
    timestamp_updated: str
    plot_config: PlotConfig
    files: tuple[SavedFile, ...]
    joint: tuple["JointFitResult", ...]


#
@dataclass(frozen=True)
class JointFitProjection:
    """
    One file's view of a project-level joint fit.

    Attributes
    ----------
    parameter_map : Mapping[str, str]
        Combined (optimizer) parameter name -> local (per-file model)
        parameter name, total in both directions for this file: every
        combined parameter that feeds this file appears as a key, every
        local model parameter appears as a value. Readers look names up
        here; they never parse the ``fileNN_`` prefix convention.
    slot : SavedFitSlot
        The per-file 2d slot capturing this file's observed/fitted
        arrays, projected parameter values, selection, noise metadata,
        and per-file residual metrics. ``slot.file_name`` is the
        projection's captured identity (no second copy is stored).
        Projections never carry ``conf_ci`` / ``correl`` / ``mcmc`` —
        joint uncertainty lives on the ``JointFitResult`` only.
    """

    parameter_map: Mapping[str, str]
    slot: SavedFitSlot


#
@dataclass(frozen=True)
class JointFitResult:
    """
    In-memory record of one successful ``Project.fit_2d`` optimization.

    One optimization produces one ``JointFitResult`` plus one projection
    per participating file, published together as one bundle (a one-file
    project fit is still a joint result: it came through the
    project-scoped path). The record owns everything belonging to the
    optimization as a whole; the projections own the per-file payloads.
    Returned by ``Project.fit_2d`` and queryable via
    ``FitResults.find_joint`` / ``get_joint``.

    Attributes
    ----------
    model_name : str
        The common model name fitted on every file.
    optimization_hash : str
        Identity of the joint optimization (fit_archive_principles.md,
        Principle 3). Every projection slot's ``joint_ref`` holds this
        value — the content-addressed bundle reference.
    input_files : str
        Canonical ``encode_input_files`` JSON, scope ``"project"``, one
        entry per participating file. Identical on every projection slot
        in the bundle.
    model_structure : str
        Canonical ``encode_model_structure`` JSON for the common model
        across the participating files.
    projections : tuple[JointFitProjection, ...]
        One per participating file, in canonical (sorted) file-name
        order. The parameter map, not tuple position, carries the
        association with optimizer parameters.
    params : pd.DataFrame
        Authoritative combined parameter table (``par_to_df`` ``"min"``
        columns) in optimizer order. ``init_value`` is the effective
        optimizer-entry value — after baseline-result injection,
        project-sharing resolution, and expression evaluation — not the
        value authored in YAML.
    metrics : Mapping[str, float]
        Whole-objective metrics; see ``_joint_result_from_project_fit``.
    fit_alg : str
        Final-stage optimizer method.
    fit_settings : Mapping[str, Any]
        Optimizer-configuration provenance (see ``build_fit_settings``).
    timestamp : str
        ISO 8601 UTC timestamp of record construction.
    conf_ci : pd.DataFrame | None
        Joint profiled confidence intervals; ``None`` when CI was
        skipped or failed.
    correl : pd.DataFrame | None
        Joint varying-parameter correlation matrix; ``None`` when the
        optimizer produced no covariance. With ``stderr`` in ``params``
        this recovers covariance as ``correl(i,j)·stderr(i)·stderr(j)``
        (covariance itself is deliberately not stored — storing both
        invites disagreement).
    mcmc : MCMCResult | None
        The joint posterior; its ``lnsigma`` is a single nuisance scale
        over the concatenated residual — never a per-file σ, never
        back-filled into a projection, never used to calibrate metrics.
    label : str | None
        User-facing label. On disk a mutable attr — settable on an
        existing archive without rewriting the record. ``None`` when
        never set.
    """

    model_name: str
    optimization_hash: str
    input_files: str
    model_structure: str
    projections: tuple[JointFitProjection, ...]
    params: pd.DataFrame
    metrics: Mapping[str, float]
    fit_alg: str
    fit_settings: Mapping[str, Any]
    timestamp: str
    conf_ci: pd.DataFrame | None = None
    correl: pd.DataFrame | None = None
    mcmc: MCMCResult | None = None
    label: str | None = None

    #
    @property
    def files(self) -> tuple[str, ...]:
        """Participating file names, derived from the projections."""

        return tuple(p.slot.file_name for p in self.projections)


#
# --- identity helpers --------------------------------------------------------
#


#
def build_selection_json(fit_type: FitType, **fields: Any) -> str:
    """
    Deterministic JSON serialization of a slot's selection dict.

    Sorted keys + ``default=_json_default`` ensure equivalent selections
    produce identical strings (and therefore identical identity hashes —
    the string enters ``optimization_hash`` via ``input_files``).
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
# --- schema-7 identity hashes (fit_archive_principles.md, Principle 3) --------
#

# Quantization applies to the initial-state matrix only; every other hash
# input is exact (principles §Quantization).
INITIAL_STATE_SIG_DIGITS = 9

_INPUT_FILE_SCOPES = ("file", "project")

# Fixed sentinel for an absent array (e.g. no aux axis): a JSON string where
# arrays encode as lists, so "no axis" never collides with a zero-length one.
_ABSENT_ARRAY = "absent"


#
def _sha256_of_json(payload: Any) -> str:
    """Sha256 over the compact, key-sorted JSON encoding of ``payload``."""

    text = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


#
def _array_record(arr: np.ndarray | None) -> list[Any] | str:
    """Tagged array encoding: ``[dtype, shape, content sha256]`` or sentinel."""

    if arr is None:
        return _ABSENT_ARRAY
    a = np.ascontiguousarray(arr)
    return [
        str(a.dtype),
        [int(s) for s in a.shape],
        hashlib.sha256(a.tobytes()).hexdigest(),
    ]


#
def _exact_float_text(value: float) -> str:
    """Exact canonical decimal text (shortest round-trip ``repr``)."""

    return repr(float(value))


#
def _quantized_float_text(value: float) -> str:
    """
    Canonical decimal text at ``INITIAL_STATE_SIG_DIGITS`` significant
    digits, with ``-0.0`` normalized to ``0``. The hash consumes this text,
    never re-parsed float bytes.
    """

    f = float(value)
    if f == 0.0:
        f = 0.0
    return f"{f:.{INITIAL_STATE_SIG_DIGITS}g}"


#
def compute_file_content_hash(
    *,
    data_raw: np.ndarray,
    energy: np.ndarray,
    time: np.ndarray | None,
    aux_axis: np.ndarray | None,
) -> str:
    """
    Single content hash of a file's immutable payload (schema 7).

    One sha256 over a tagged record of ``data_raw`` (uncorrected),
    ``energy``, ``time``, and ``aux_axis``, each contributing dtype,
    shape, and content. ``None`` axes contribute a fixed sentinel, so a
    missing axis is distinguishable from a zero-length one.
    """

    return _sha256_of_json(
        [
            "file_content",
            _array_record(data_raw),
            _array_record(energy),
            _array_record(time),
            _array_record(aux_axis),
        ]
    )


#
def compute_file_version_stamp(
    *,
    file_content_hash: str,
    dark: np.ndarray | None,
    calibration: np.ndarray | None,
) -> str:
    """
    Version stamp of the data state a fit consumed (schema 7).

    ``sha256(file_content_hash + dark + calibration)`` as a tagged record:
    the correction state folds into **fit** identity here while file
    identity stays the guarded name (Principle 3).
    """

    return _sha256_of_json(
        [
            "version_stamp",
            file_content_hash,
            _array_record(dark),
            _array_record(calibration),
        ]
    )


#
def encode_input_files(
    *,
    scope: str,
    entries: Sequence[tuple[str, str, str]],
) -> str:
    """
    Canonical ``input_files`` JSON: ``[scope, [[name, stamp, selection]..]]``.

    ``entries`` are ``(file_name, version_stamp, selection_json)`` tuples;
    they are sorted by file name (canonical order, principles §Composite
    keys). Duplicate file names raise — file identity is the unique name.
    Stored verbatim as the slot attr and consumed by
    ``compute_optimization_hash``.
    """

    if scope not in _INPUT_FILE_SCOPES:
        raise ValueError(
            f"input_files scope must be one of {_INPUT_FILE_SCOPES}, got {scope!r}"
        )
    ordered = sorted(entries, key=lambda e: e[0])
    names = [name for name, _, _ in ordered]
    if len(set(names)) != len(names):
        raise ValueError(f"input_files entries contain duplicate file names: {names}")
    payload = [scope, [[name, stamp, selection] for name, stamp, selection in ordered]]
    return json.dumps(payload, separators=(",", ":"))


#
def encode_model_structure(
    entries: Sequence[
        tuple[str, Sequence[str], Sequence[tuple[str, Sequence[str], float]]]
    ],
) -> str:
    """
    Canonical ``model_structure`` JSON (schema 7).

    ``entries`` are ``(file_name, energy_models, dynamics)`` per file —
    one for file scope, N for a joint fit. Names throughout are
    **top-level YAML model names**, never component names or the joined
    composite name — component-level structure enters identity through
    the parameter table instead. ``energy_models`` is the ordered
    composition list (order is identity). Each dynamics attachment is
    ``(target_par, submodels, frequency)`` where ``submodels`` is the
    flat ordered tuple of submodel names — order is what assigns
    subcycles (principles §model_structure). Files and attachments sort
    canonically (by file name / target parameter); model and submodel
    order is preserved. ``frequency`` encodes as exact decimal text.
    """

    file_names = [name for name, _, _ in entries]
    if len(set(file_names)) != len(file_names):
        raise ValueError(
            f"model_structure entries contain duplicate file names: {file_names}"
        )
    encoded_files = []
    for file_name, energy_models, dynamics in sorted(entries, key=lambda e: e[0]):
        targets = [target for target, _, _ in dynamics]
        if len(set(targets)) != len(targets):
            raise ValueError(
                f"model_structure for {file_name!r} contains duplicate dynamics "
                f"targets: {targets}"
            )
        encoded_dynamics = [
            [
                target,
                [str(name) for name in submodels],
                _exact_float_text(frequency),
            ]
            for target, submodels, frequency in sorted(dynamics, key=lambda d: d[0])
        ]
        encoded_files.append([file_name, [list(energy_models), encoded_dynamics]])
    return json.dumps(encoded_files, separators=(",", ":"))


#
def encode_optimizer_settings(
    *,
    stages: int,
    fit_alg_1: str,
    fit_alg_2: str,
    backend: str,
    seed: int | None = None,
    jac_fun_name: str | None = None,
) -> str:
    """
    Keyed optimizer-settings record: only what was actually in force.

    ``stages``, ``fit_alg_1``, and the **effective** evaluator ``backend``
    are always keyed; ``fit_alg_2`` only when ``stages == 2``; ``seed``
    only when supplied; ``jac_fun_name`` (``module.qualname``) only when
    an analytic Jacobian was applied — lmfit forwards ``Dfun`` solely for
    ``leastsq``, so it is keyed iff some stage in force uses ``leastsq``.
    """

    if stages not in (1, 2):
        raise ValueError(f"stages must be 1 or 2, got {stages!r}")
    settings: dict[str, Any] = {
        "stages": int(stages),
        "fit_alg_1": str(fit_alg_1),
        "backend": str(backend),
    }
    if stages == 2:
        settings["fit_alg_2"] = str(fit_alg_2)
    if seed is not None:
        settings["seed"] = int(seed)
    leastsq_in_force = fit_alg_1 == "leastsq" or (
        stages == 2 and fit_alg_2 == "leastsq"
    )
    if jac_fun_name is not None and leastsq_in_force:
        settings["jac_fun"] = str(jac_fun_name)
    return json.dumps(settings, separators=(",", ":"), sort_keys=True)


#
def compute_optimization_hash(
    *,
    input_files_json: str,
    fit_type: FitType,
    model_structure_json: str,
    parameter_metadata: Sequence[tuple[str, float, float, bool, str | None]],
    initial_state: np.ndarray | Sequence[Sequence[float]],
    optimizer_settings_json: str,
) -> str:
    """
    Identity hash of one optimization: complete input, nothing else.

    ``parameter_metadata`` rows are ``(name, min, max, vary, expr)`` in
    **model order** — order is identity, never sorted. Bounds encode
    exact; ``expr`` is ``None`` for non-expression parameters. For a
    joint fit the rows are the **combined** table, where project sharing
    is visible in the names. ``initial_state`` is the ``(n_slices,
    n_par)`` optimizer-entry value matrix (one row except SbS), the only
    quantized input. Shared by all projections of a joint fit.
    """

    matrix = np.atleast_2d(np.asarray(initial_state, dtype=float))
    if matrix.shape[1] != len(parameter_metadata):
        raise ValueError(
            f"initial_state has {matrix.shape[1]} columns but "
            f"parameter_metadata has {len(parameter_metadata)} rows"
        )
    metadata_rows = [
        [str(name), _exact_float_text(lo), _exact_float_text(hi), bool(vary), expr]
        for name, lo, hi, vary, expr in parameter_metadata
    ]
    matrix_rows = [[_quantized_float_text(v) for v in row] for row in matrix]
    return _sha256_of_json(
        [
            "optimization",
            input_files_json,
            fit_type,
            model_structure_json,
            metadata_rows,
            matrix_rows,
            optimizer_settings_json,
        ]
    )


#
def compute_slot_handle(*, optimization_hash: str, file_name: str) -> str:
    """
    Stored slot handle: ``sha256(optimization_hash + file_name)``, tagged.

    Full 64-hex is authoritative on disk; display abbreviates to 8 and
    lookup prefix-matches (query layer). Joint siblings share the
    ``optimization_hash`` and differ here by file name.
    """

    return _sha256_of_json(["handle", optimization_hash, file_name])


#
def compute_fit_view_sha256(
    *,
    observed: np.ndarray,
    energy: np.ndarray,
    time: np.ndarray | None,
    aux_axis: np.ndarray | None,
) -> str:
    """
    Comparability hash of what the optimizer saw (schema 7).

    Tagged record of the ``observed`` array plus the **selected** energy
    and time coordinates and the aux axis. ``time`` is ``None`` iff the
    file has no time axis; ``aux_axis`` is ``None`` iff the file has none
    — presence is a file property, never conditioned on whether a model
    consumes the axis. Deliberately **not** an input to
    ``compute_optimization_hash``: the view is implied there by
    ``(version_stamp, selection)``, and this hash exists to cross-check
    that derivation against the arrays actually used.
    """

    return _sha256_of_json(
        [
            "fit_view",
            _array_record(observed),
            _array_record(energy),
            _array_record(time),
            _array_record(aux_axis),
        ]
    )


#
def _frozen_copy(arr: np.ndarray) -> np.ndarray:
    """Copy with the write flag cleared — the snapshot ownership boundary."""

    out = np.array(arr, copy=True)
    out.flags.writeable = False
    return out


#
def capture_saved_file(
    *,
    name: str,
    original_path: str,
    dim: int,
    data_raw: np.ndarray,
    energy: np.ndarray,
    time: np.ndarray | None,
    aux_axis: np.ndarray | None,
    file_content_hash: str,
) -> SavedFile:
    """
    Copy-and-freeze the immutable ``SavedFile`` payload (no slots yet).

    Called when a file produces its first slot — the single point where
    file content crosses the capture boundary (Principle 4). Save-time
    assembly attaches slots via ``dataclasses.replace``; nothing is read
    from live state at save time.
    """

    return SavedFile(
        name=name,
        original_path=original_path,
        dim=int(dim),
        shape=tuple(data_raw.shape),
        file_content_hash=file_content_hash,
        data_raw=_frozen_copy(data_raw),
        energy=_frozen_copy(energy),
        time=_frozen_copy(time if time is not None else np.empty(0)),
        slots=(),
        aux_axis=_frozen_copy(aux_axis) if aux_axis is not None else None,
    )


#
def params_identity(
    params: Any,
) -> tuple[list[tuple[str, float, float, bool, str | None]], list[float]]:
    """
    ``(parameter_metadata, initial_values)`` from optimizer-entry Parameters.

    Rows are ``(name, min, max, vary, expr)`` in model order (lmfit
    preserves insertion order); values are the optimizer-entry values —
    ``FitOutput.par_ini`` is the pre-fit deepcopy, so ``.value`` is the
    effective entry value after baseline injection, sharing resolution,
    and expression evaluation. Feeds ``compute_optimization_hash``.
    """

    rows: list[tuple[str, float, float, bool, str | None]] = []
    values: list[float] = []
    for p in params.values():
        rows.append(
            (
                str(p.name),
                float(p.min),
                float(p.max),
                bool(p.vary),
                p.expr if p.expr else None,
            )
        )
        values.append(float(p.value))
    return rows, values


#
def optimizer_settings_from_provenance(fit_settings: Mapping[str, Any]) -> str:
    """
    ``encode_optimizer_settings`` JSON from a ``build_fit_settings`` dict.

    One source: the provenance dict records everything result-shaping,
    and the encoder applies the conditional identity keying (``fit_alg_2``
    iff two stages, ``seed`` / ``jac_fun`` iff supplied).
    """

    return encode_optimizer_settings(
        stages=int(fit_settings["stages"]),
        fit_alg_1=str(fit_settings["fit_alg_1"]),
        fit_alg_2=str(fit_settings["fit_alg_2"]),
        backend=str(fit_settings["backend"]),
        seed=fit_settings.get("seed"),
        jac_fun_name=fit_settings.get("jac_fun"),
    )


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
    returns ``{"flatchain", "ci", "lnsigma", "acceptance_fraction"}`` matching
    ``SavedFitSlot.mcmc``. Frames and arrays are copied so the slot is
    invariant to subsequent state changes.
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
    acceptance = getattr(emcee_fin, "acceptance_fraction", None)
    acceptance_out = (
        np.array(acceptance, dtype=np.float64) if acceptance is not None else None
    )
    return {
        "flatchain": flatchain_out,
        "ci": ci_out,
        "lnsigma": lnsigma,
        "acceptance_fraction": acceptance_out,
    }


#
def mcmc_result_from_payload(payload: dict[str, Any]) -> MCMCResult:
    """
    Build an ``MCMCResult`` from a persisted mcmc payload dict.

    The single decoder for the ``{"flatchain", "ci", "lnsigma",
    "acceptance_fraction"}`` payload produced by ``_mcmc_payload`` — used by
    both the per-file slot path (``FitResults.get_mcmc``) and the
    project-level joint path (``JointFitResult.mcmc``). Frames and arrays
    are copied so the returned bundle never aliases the stored record.
    """

    flatchain = payload.get("flatchain")
    ci = payload.get("ci")
    acceptance = payload.get("acceptance_fraction")
    lnsigma = payload.get("lnsigma")
    return MCMCResult(
        table=ci.copy() if ci is not None else pd.DataFrame(),
        flatchain=flatchain.copy() if flatchain is not None else pd.DataFrame(),
        acceptance_fraction=(
            np.asarray(acceptance).copy() if acceptance is not None else None
        ),
        lnsigma=float(lnsigma) if lnsigma is not None else None,
    )


#
def build_fit_settings(
    *,
    stages: int,
    backend: str,
    fit_wrapper_kwargs: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """
    Assemble the provenance dict stored as ``SavedFitSlot.fit_settings``.

    Records the optimizer configuration that can influence the fit result:
    stage count, per-stage methods, the **effective** evaluator backend
    (what the dispatch site actually ran, never the requested string), the
    analytic Jacobian's qualified name when one was supplied, the
    optimizer RNG seed when one was supplied (forwarded to the stage-1
    method by ``fitlib.fit_wrapper``), the
    profiled-CI request, MCMC sampling settings (when enabled), plus any
    fit-type-specific extras the caller passes verbatim (e.g. SbS
    ``seed_source`` / ``seed_adapt`` / ``seed_values`` — ``None`` values
    are kept: "no seed adaptation" is provenance too). Execution details
    that cannot change the result (SbS / emcee worker counts) are
    deliberately excluded — serial and parallel dispatch are pinned
    result-identical by test. ``optimizer_settings_from_provenance``
    derives the identity-keyed subset from this dict.

    Defaults mirror ``fitlib.fit_wrapper``'s signature; if you change one,
    change the other.
    """

    kwargs = fit_wrapper_kwargs or {}
    settings: dict[str, Any] = {
        "stages": int(stages),
        "fit_alg_1": str(kwargs.get("fit_alg_1", "Nelder")),
        "fit_alg_2": str(kwargs.get("fit_alg_2", "leastsq")),
        "backend": str(backend),
        "try_ci": int(kwargs.get("try_ci", 1)),
    }
    jac_fun = kwargs.get("jac_fun")
    if jac_fun is not None:
        settings["jac_fun"] = f"{jac_fun.__module__}.{jac_fun.__qualname__}"
    seed = kwargs.get("seed")
    if seed is not None:
        settings["seed"] = int(seed)
    mc = kwargs.get("mc_settings")
    # MC stores its use_mc constructor arg as the use_emcee attribute.
    if mc is not None and getattr(mc, "use_emcee", False):
        settings["mc"] = {
            "use_mc": int(mc.use_emcee),
            "steps": int(mc.steps),
            "nwalkers": int(mc.nwalkers),
            "burn": int(mc.burn),
            "thin": int(mc.thin),
            "ntemps": int(mc.ntemps),
            "is_weighted": bool(mc.is_weighted),
            "sigma_ini": float(mc.sigma_ini),
            "sigma_min": float(mc.sigma_min),
            "sigma_max": float(mc.sigma_max),
        }
    settings.update(extra)
    return settings


#
# --- per-fit-type slot extractors -------------------------------------------
#


#
def _slot_from_baseline(
    *,
    file_name: str,
    model_name: str,
    fit_alg: str,
    params_df: pd.DataFrame,
    observed: np.ndarray,
    fit: np.ndarray,
    base_t_ind: list[int],
    e_lim: list[int] | None,
    n_free_pars: int | None,
    version_stamp: str,
    model_structure: str,
    parameter_metadata: Sequence[tuple[str, float, float, bool, str | None]],
    initial_state: np.ndarray | Sequence[Sequence[float]],
    fit_settings: dict[str, Any],
    energy: np.ndarray,
    time: np.ndarray | None,
    aux_axis: np.ndarray | None,
    dark: np.ndarray | None,
    calibration: np.ndarray | None,
    model_yaml: tuple[ModelYamlRecord, ...] | None,
    noise_type: str,
    sigma_source: str,
    sigma_type: str,
    sigma_data: float,
    conf_ci: pd.DataFrame | None = None,
    correl: pd.DataFrame | None = None,
    mcmc: dict[str, Any] | None = None,
    components: np.ndarray | None = None,
    component_names: list[str] | None = None,
    fit_ini: np.ndarray | None = None,
) -> SavedFitSlot:
    """
    Build a SavedFitSlot for a completed baseline fit.

    Caller passes already-copied snapshot args (no live Model references) so
    the helper is invariant to post-fit cleanup. The noise metadata is also
    a snapshot of the File's σ state at fit completion — subsequent calls
    to ``File.set_sigma`` do not retroactively rewrite the slot. ``energy``
    / ``time`` are the **selected** view coordinates (here: e_lim-cropped
    energy and the time slices averaged into the baseline), feeding
    ``compute_fit_view_sha256``.
    """

    selection = {
        "base_t_ind": list(base_t_ind),
        "e_lim": list(e_lim) if e_lim else None,
    }
    return _build_slot(
        file_name=file_name,
        model_name=model_name,
        fit_type="baseline",
        selection=selection,
        params=params_df,
        observed=observed,
        fit=fit,
        n_free_pars=n_free_pars,
        fit_alg=fit_alg,
        version_stamp=version_stamp,
        model_structure=model_structure,
        parameter_metadata=parameter_metadata,
        initial_state=initial_state,
        joint_identity=None,
        energy=energy,
        time=time,
        aux_axis=aux_axis,
        dark=dark,
        calibration=calibration,
        model_yaml=model_yaml,
        conf_ci=conf_ci,
        correl=correl,
        mcmc=mcmc,
        fit_settings=fit_settings,
        noise_type=noise_type,
        sigma_source=sigma_source,
        sigma_type=sigma_type,
        sigma_data=sigma_data,
        components=components,
        component_names=component_names,
        fit_ini=fit_ini,
    )


#
def _slot_from_spectrum(
    *,
    file_name: str,
    model_name: str,
    fit_alg: str,
    params_df: pd.DataFrame,
    observed: np.ndarray,
    fit: np.ndarray,
    time_point: float | None,
    time_range: list[float] | None,
    time_type: str,
    e_lim: list[int] | None,
    n_free_pars: int | None,
    version_stamp: str,
    model_structure: str,
    parameter_metadata: Sequence[tuple[str, float, float, bool, str | None]],
    initial_state: np.ndarray | Sequence[Sequence[float]],
    fit_settings: dict[str, Any],
    energy: np.ndarray,
    time: np.ndarray | None,
    aux_axis: np.ndarray | None,
    dark: np.ndarray | None,
    calibration: np.ndarray | None,
    model_yaml: tuple[ModelYamlRecord, ...] | None,
    noise_type: str,
    sigma_source: str,
    sigma_type: str,
    sigma_data: float,
    conf_ci: pd.DataFrame | None = None,
    correl: pd.DataFrame | None = None,
    mcmc: dict[str, Any] | None = None,
    components: np.ndarray | None = None,
    component_names: list[str] | None = None,
    fit_ini: np.ndarray | None = None,
) -> SavedFitSlot:
    """Build a SavedFitSlot for a completed spectrum fit.

    v1 does not auto-correct σ for ``time_range`` averaging — users fitting
    an averaged spectrum should pre-scale the σ they pass to
    ``File.set_sigma()``. ``energy`` / ``time`` are the **selected** view
    coordinates (e_lim-cropped energy; the selected time point/range).
    """

    selection = {
        "time_point": time_point,
        "time_range": list(time_range) if time_range else None,
        "time_type": time_type,
        "e_lim": list(e_lim) if e_lim else None,
    }
    return _build_slot(
        file_name=file_name,
        model_name=model_name,
        fit_type="spectrum",
        selection=selection,
        params=params_df,
        observed=observed,
        fit=fit,
        n_free_pars=n_free_pars,
        fit_alg=fit_alg,
        version_stamp=version_stamp,
        model_structure=model_structure,
        parameter_metadata=parameter_metadata,
        initial_state=initial_state,
        joint_identity=None,
        energy=energy,
        time=time,
        aux_axis=aux_axis,
        dark=dark,
        calibration=calibration,
        model_yaml=model_yaml,
        conf_ci=conf_ci,
        correl=correl,
        mcmc=mcmc,
        fit_settings=fit_settings,
        noise_type=noise_type,
        sigma_source=sigma_source,
        sigma_type=sigma_type,
        sigma_data=sigma_data,
        components=components,
        component_names=component_names,
        fit_ini=fit_ini,
    )


#
def _slot_from_sbs(
    *,
    file_name: str,
    model_name: str,
    fit_alg: str,
    params_df: pd.DataFrame,
    observed: np.ndarray,
    fit: np.ndarray,
    e_lim: list[int] | None,
    t_lim: list[int] | None,
    n_free_pars: int | None,
    version_stamp: str,
    model_structure: str,
    parameter_metadata: Sequence[tuple[str, float, float, bool, str | None]],
    initial_state: np.ndarray | Sequence[Sequence[float]],
    fit_settings: dict[str, Any],
    energy: np.ndarray,
    time: np.ndarray | None,
    aux_axis: np.ndarray | None,
    dark: np.ndarray | None,
    calibration: np.ndarray | None,
    model_yaml: tuple[ModelYamlRecord, ...] | None,
    noise_type: str,
    sigma_source: str,
    sigma_type: str,
    sigma_data: float,
    conf_ci: pd.DataFrame | None = None,
    correl: pd.DataFrame | None = None,
    mcmc: dict[str, Any] | None = None,
    params_meta: pd.DataFrame | None = None,
    params_stderr: pd.DataFrame | None = None,
    components: np.ndarray | None = None,
    component_names: list[str] | None = None,
    fit_ini: np.ndarray | None = None,
    params_init: pd.DataFrame | None = None,
) -> SavedFitSlot:
    """
    Build a SavedFitSlot for a completed slice-by-slice fit.

    ``observed`` and ``fit`` are 2D arrays (slices x energy_in_lim).
    ``metrics`` values are per-slice 1D arrays. ``params_df`` is the SbS
    DataFrame (one row per slice). ``components`` is 3D
    ``(n_slices, n_components, energy_in_lim)`` when provided.
    ``initial_state`` is the per-slice seed matrix (one row per slice) —
    the only place the initial-state matrix has more than one row.
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
    input_files, optimization_hash, handle, joint_ref = _slot_identity(
        file_name=file_name,
        fit_type="sbs",
        selection_json=selection_json,
        version_stamp=version_stamp,
        model_structure=model_structure,
        parameter_metadata=parameter_metadata,
        initial_state=initial_state,
        fit_settings=fit_settings,
        joint_identity=None,
    )
    return SavedFitSlot(
        handle=handle,
        optimization_hash=optimization_hash,
        input_files=input_files,
        model_structure=model_structure,
        fit_view_sha256=compute_fit_view_sha256(
            observed=observed, energy=energy, time=time, aux_axis=aux_axis
        ),
        file_name=file_name,
        model_name=model_name,
        fit_type="sbs",
        selection=selection,
        selection_json=selection_json,
        params=params_df,
        metrics=metrics,
        observed=_frozen_copy(np.asarray(observed)),
        fit=_frozen_copy(np.asarray(fit)),
        fit_alg=fit_alg,
        timestamp=_now_iso(),
        noise_type=noise_type,
        sigma_source=sigma_source,
        sigma_type=sigma_type,
        sigma_data=float(sigma_data),
        sigma_eff=float(sigma_eff),
        dark=_frozen_copy(dark) if dark is not None else None,
        calibration=_frozen_copy(calibration) if calibration is not None else None,
        model_yaml=model_yaml or None,
        joint_ref=joint_ref,
        conf_ci=conf_ci,
        correl=correl,
        mcmc=mcmc,
        params_meta=params_meta,
        params_stderr=params_stderr,
        fit_settings=fit_settings,
        components=_frozen_copy(components) if components is not None else None,
        component_names=component_names,
        fit_ini=_frozen_copy(fit_ini) if fit_ini is not None else None,
        params_init=params_init,
    )


#
def _slot_from_2d(
    *,
    file_name: str,
    model_name: str,
    fit_alg: str,
    params_df: pd.DataFrame,
    observed: np.ndarray,
    fit: np.ndarray,
    e_lim: list[int] | None,
    t_lim: list[int] | None,
    n_free_pars: int | None,
    model_structure: str,
    fit_settings: dict[str, Any],
    energy: np.ndarray,
    time: np.ndarray | None,
    aux_axis: np.ndarray | None,
    dark: np.ndarray | None,
    calibration: np.ndarray | None,
    model_yaml: tuple[ModelYamlRecord, ...] | None,
    noise_type: str,
    sigma_source: str,
    sigma_type: str,
    sigma_data: float,
    version_stamp: str | None = None,
    parameter_metadata: Sequence[tuple[str, float, float, bool, str | None]]
    | None = None,
    initial_state: np.ndarray | Sequence[Sequence[float]] | None = None,
    joint_identity: tuple[str, str] | None = None,
    conf_ci: pd.DataFrame | None = None,
    correl: pd.DataFrame | None = None,
    mcmc: dict[str, Any] | None = None,
    fit_ini: np.ndarray | None = None,
) -> SavedFitSlot:
    """Build a SavedFitSlot for a completed 2D global fit.

    Two identity modes: a file-scope fit passes ``version_stamp`` /
    ``parameter_metadata`` / ``initial_state`` and the local chain is
    computed here; a project-level joint projection passes
    ``joint_identity = (optimization_hash, input_files_json)`` computed
    once for the whole bundle — the slot inherits it and gets its
    ``joint_ref`` and per-file ``handle`` from it. ``n_free_pars`` is
    ``None`` on the joint path (the joint parameter count does not
    decompose by file), which makes the count-dependent metrics ``NaN`` —
    see ``compute_fit_metrics``.
    """

    selection = {
        "e_lim": list(e_lim) if e_lim else None,
        "t_lim": list(t_lim) if t_lim else None,
    }
    return _build_slot(
        file_name=file_name,
        model_name=model_name,
        fit_type="2d",
        selection=selection,
        params=params_df,
        observed=observed,
        fit=fit,
        n_free_pars=n_free_pars,
        fit_alg=fit_alg,
        version_stamp=version_stamp,
        model_structure=model_structure,
        parameter_metadata=parameter_metadata,
        initial_state=initial_state,
        joint_identity=joint_identity,
        energy=energy,
        time=time,
        aux_axis=aux_axis,
        dark=dark,
        calibration=calibration,
        model_yaml=model_yaml,
        conf_ci=conf_ci,
        correl=correl,
        mcmc=mcmc,
        fit_settings=fit_settings,
        noise_type=noise_type,
        sigma_source=sigma_source,
        sigma_type=sigma_type,
        sigma_data=sigma_data,
        fit_ini=fit_ini,
    )


#
def _joint_result_from_project_fit(
    *,
    model_name: str,
    optimization_hash: str,
    input_files: str,
    model_structure: str,
    mapping: Sequence[tuple[str, int, str]],
    slots: Sequence[SavedFitSlot],
    fit_output: ulmfit.FitOutput,
    fit_settings: dict[str, Any] | None,
) -> JointFitResult:
    """
    Build the ``JointFitResult`` for a completed ``Project.fit_2d``.

    ``slots`` are the per-file projection slots in file-index order
    (matching ``mapping``'s ``file_idx``); ``fit_output`` is the joint
    ``fitlib.fit_wrapper`` result. The identity fields are the bundle's:
    every slot already carries ``optimization_hash`` as its ``joint_ref``
    (built with ``joint_identity``). Frames and arrays are copied here —
    no lmfit reference survives into the record. Raises ``ValueError``
    when the pieces do not form one consistent bundle; the caller
    publishes nothing in that case.

    Whole-objective metrics come from ``compute_fit_metrics`` over the
    concatenated prediction (not lmfit's ``aic``/``bic``), so joint and
    per-file numbers stay comparable by construction. ``chi2`` is the sum
    of the projections' σ-calibrated values (it sums cleanly across
    heterogeneous per-file noise scales; NaN unless every file's σ is
    valid), ``chi2_red`` divides it by the joint DoF, and ``r2`` is NaN —
    it would depend on an arbitrary global mean across separate
    measurements.
    """

    par_fin = fit_output.par_fin
    params_df = ulmfit.par_to_df(par_fin.params, col_type="min")

    # Per-file combined -> local maps, keyed by file index.
    maps_by_idx: dict[int, dict[str, str]] = {}
    for combined_name, file_idx, local_name in mapping:
        maps_by_idx.setdefault(int(file_idx), {})[combined_name] = local_name
    if sorted(maps_by_idx) != list(range(len(slots))):
        raise ValueError(
            f"Joint-fit capture: mapping covers file indices "
            f"{sorted(maps_by_idx)} but {len(slots)} projection slot(s) "
            f"were built."
        )

    projections: list[JointFitProjection] = []
    for file_idx, slot in enumerate(slots):
        if slot.model_name != model_name:
            raise ValueError(
                f"Joint-fit capture: slot for file {slot.file_name!r} "
                f"records model {slot.model_name!r}, expected "
                f"{model_name!r}."
            )
        parameter_map = maps_by_idx[file_idx]
        if set(parameter_map.values()) != set(slot.params["name"]):
            raise ValueError(
                f"Joint-fit capture: the parameter map for file "
                f"{slot.file_name!r} does not cover its model parameters "
                f"exactly."
            )
        projections.append(JointFitProjection(parameter_map=parameter_map, slot=slot))
    # Canonical file-name order; the map, not tuple position, carries the
    # association with optimizer parameters.
    projections.sort(key=lambda p: p.slot.file_name)

    concat_observed = np.concatenate([np.asarray(s.observed).ravel() for s in slots])
    concat_fit = np.concatenate([np.asarray(s.fit).ravel() for s in slots])
    joint_nvarys = int(getattr(par_fin, "nvarys", 0))
    metrics = compute_fit_metrics(
        observed=concat_observed,
        fit=concat_fit,
        n_free_pars=joint_nvarys,
    )
    chi2 = float(sum(float(s.metrics["chi2"]) for s in slots))
    dof = concat_observed.size - joint_nvarys
    metrics["chi2"] = chi2
    metrics["chi2_red"] = chi2 / dof if dof > 0 else float("nan")
    metrics["r2"] = float("nan")

    conf_ci = fit_output.conf_ci
    mcmc_payload = _mcmc_payload(fit_output.emcee_fin, fit_output.emcee_ci)
    return JointFitResult(
        model_name=model_name,
        optimization_hash=optimization_hash,
        input_files=input_files,
        model_structure=model_structure,
        projections=tuple(projections),
        params=params_df,
        metrics=metrics,
        fit_alg=str(getattr(par_fin, "method", "unknown")),
        fit_settings=copy.deepcopy(fit_settings) if fit_settings else {},
        timestamp=_now_iso(),
        conf_ci=conf_ci.copy() if not conf_ci.empty else None,
        correl=ulmfit.correl_from_result(par_fin),
        mcmc=(
            mcmc_result_from_payload(mcmc_payload) if mcmc_payload is not None else None
        ),
    )


#
# --- internal builders ------------------------------------------------------
#


#
def _slot_identity(
    *,
    file_name: str,
    fit_type: FitType,
    selection_json: str,
    version_stamp: str | None,
    model_structure: str,
    parameter_metadata: Sequence[tuple[str, float, float, bool, str | None]] | None,
    initial_state: np.ndarray | Sequence[Sequence[float]] | None,
    fit_settings: Mapping[str, Any],
    joint_identity: tuple[str, str] | None,
) -> tuple[str, str, str, str | None]:
    """
    ``(input_files, optimization_hash, handle, joint_ref)`` for one slot.

    File scope: the local identity chain is computed here from the
    captured primitives. Joint scope: ``joint_identity`` is the bundle's
    ``(optimization_hash, input_files_json)``, computed once by
    ``Project.fit_2d`` — the slot inherits it, and ``joint_ref`` points
    back at it.
    """

    if joint_identity is not None:
        optimization_hash, input_files = joint_identity
        joint_ref: str | None = optimization_hash
    else:
        if version_stamp is None or parameter_metadata is None or initial_state is None:
            raise ValueError(
                "file-scope slot capture requires version_stamp, "
                "parameter_metadata, and initial_state (only a joint "
                "projection may omit them, via joint_identity)"
            )
        input_files = encode_input_files(
            scope="file", entries=[(file_name, version_stamp, selection_json)]
        )
        optimization_hash = compute_optimization_hash(
            input_files_json=input_files,
            fit_type=fit_type,
            model_structure_json=model_structure,
            parameter_metadata=parameter_metadata,
            initial_state=initial_state,
            optimizer_settings_json=optimizer_settings_from_provenance(fit_settings),
        )
        joint_ref = None
    handle = compute_slot_handle(
        optimization_hash=optimization_hash, file_name=file_name
    )
    return input_files, optimization_hash, handle, joint_ref


#
def _build_slot(
    *,
    file_name: str,
    model_name: str,
    fit_type: FitType,
    selection: dict[str, Any],
    params: pd.DataFrame,
    observed: np.ndarray,
    fit: np.ndarray,
    n_free_pars: int | None,
    fit_alg: str,
    version_stamp: str | None,
    model_structure: str,
    parameter_metadata: Sequence[tuple[str, float, float, bool, str | None]] | None,
    initial_state: np.ndarray | Sequence[Sequence[float]] | None,
    joint_identity: tuple[str, str] | None,
    energy: np.ndarray,
    time: np.ndarray | None,
    aux_axis: np.ndarray | None,
    dark: np.ndarray | None,
    calibration: np.ndarray | None,
    model_yaml: tuple[ModelYamlRecord, ...] | None,
    conf_ci: pd.DataFrame | None,
    correl: pd.DataFrame | None,
    mcmc: dict[str, Any] | None,
    fit_settings: dict[str, Any],
    noise_type: str,
    sigma_source: str,
    sigma_type: str,
    sigma_data: float,
    components: np.ndarray | None = None,
    component_names: list[str] | None = None,
    fit_ini: np.ndarray | None = None,
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
    input_files, optimization_hash, handle, joint_ref = _slot_identity(
        file_name=file_name,
        fit_type=fit_type,
        selection_json=selection_json,
        version_stamp=version_stamp,
        model_structure=model_structure,
        parameter_metadata=parameter_metadata,
        initial_state=initial_state,
        fit_settings=fit_settings,
        joint_identity=joint_identity,
    )
    return SavedFitSlot(
        handle=handle,
        optimization_hash=optimization_hash,
        input_files=input_files,
        model_structure=model_structure,
        fit_view_sha256=compute_fit_view_sha256(
            observed=observed, energy=energy, time=time, aux_axis=aux_axis
        ),
        file_name=file_name,
        model_name=model_name,
        fit_type=fit_type,
        selection=selection,
        selection_json=selection_json,
        params=params,
        metrics=metrics,
        observed=_frozen_copy(np.asarray(observed)),
        fit=_frozen_copy(np.asarray(fit)),
        fit_alg=fit_alg,
        timestamp=_now_iso(),
        noise_type=noise_type,
        sigma_source=sigma_source,
        sigma_type=sigma_type,
        sigma_data=float(sigma_data),
        sigma_eff=float(sigma_eff),
        dark=_frozen_copy(dark) if dark is not None else None,
        calibration=_frozen_copy(calibration) if calibration is not None else None,
        model_yaml=model_yaml or None,
        joint_ref=joint_ref,
        conf_ci=conf_ci,
        correl=correl,
        mcmc=mcmc,
        fit_settings=fit_settings,
        components=_frozen_copy(components) if components is not None else None,
        component_names=component_names,
        fit_ini=_frozen_copy(fit_ini) if fit_ini is not None else None,
    )


#
def _per_slice_metrics(
    *,
    observed: np.ndarray,
    fit: np.ndarray,
    n_free_pars: int | None,
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
    return {k: _frozen_copy(np.array(v)) for k, v in out.items()}


#
# --- history collapse -------------------------------------------------------
#


#
def _divergence_remedy(fit_settings: Mapping[str, Any] | None) -> str:
    """
    Actionable tail for a same-handle divergence error.

    A recorded seed means the user already asked for reproducibility —
    re-recommending one would be noise. The seed reaches only the
    ``fit_alg_1`` stage (the two-stage contract designates stage 2 as
    deterministic refinement, but ``fit_alg_2`` stays free-form), so the
    remaining divergence sources are a stochastic second stage or an
    environment difference between the runs.
    """

    if fit_settings and fit_settings.get("seed") is not None:
        hint = (
            "A seed was supplied and reaches only the fit_alg_1 stage — "
            "check for a stochastic fit_alg_2 (or an environment "
            "difference between the runs)"
        )
    else:
        hint = (
            "Pin an optimizer seed to make re-runs reproducible (a "
            "seeded run is a distinct configuration that never collides)"
        )
    return (
        f"{hint}, or pass overwrite=True to keep only the latest run. "
        f"Nothing is recomputed: every run remains in the in-session "
        f"history."
    )


#
def collapse_history_to_snapshot(
    slots: list[SavedFitSlot], *, overwrite: bool = False
) -> list[SavedFitSlot]:
    """
    Keep the latest slot per ``handle`` (exact-duplicate dedup).

    Two slots share a handle only when they are the same optimization on
    the same file — a re-run with an unchanged seed and settings. Distinct
    variants (different initial state, settings, or corrections) have
    distinct handles and are all kept; choosing among them is the query
    layer's ``select=``, never a silent collapse.

    Collapse applies the same divergence rule as the archive boundary
    (fit_archive_principles.md §"One rule, both boundaries"): if two
    same-handle slots differ in fitted values beyond the named
    equivalence tolerances, the configuration is not deterministic —
    raises ``FileExistsError`` (the archive's collision type, so one
    ``except`` clause covers both boundaries) unless ``overwrite=True``,
    which keeps the latest run and warns about the replacement. Nothing
    is ever recomputed: every run stays in the in-session history.
    """

    latest: dict[str, SavedFitSlot] = {}
    for slot in slots:
        stored = latest.get(slot.handle)
        if stored is not None and not _fitted_values_equivalent(
            stored.params, slot.params, is_long_params=slot.fit_type != "sbs"
        ):
            if not overwrite:
                raise FileExistsError(
                    f"Two fits of file {slot.file_name!r} (model "
                    f"{slot.model_name!r}, fit_type {slot.fit_type!r}) share "
                    f"handle {slot.handle[:8]} but differ in fitted values — "
                    f"the optimizer configuration is not deterministic "
                    f"(identical inputs produced different optima). "
                    + _divergence_remedy(slot.fit_settings)
                )
            warnings.warn(
                f"Divergent re-runs under handle {slot.handle[:8]} (file "
                f"{slot.file_name!r}): keeping the latest (timestamp "
                f"{slot.timestamp}) and dropping the earlier result from "
                f"this save. Every run remains in the in-session history.",
                UserWarning,
                stacklevel=2,
            )
        latest[slot.handle] = slot
    return list(latest.values())


#
def collapse_joint_history_to_snapshot(
    records: Sequence[JointFitResult], *, overwrite: bool = False
) -> list[JointFitResult]:
    """
    Keep the latest joint record per ``optimization_hash``.

    Same divergence rule as ``collapse_history_to_snapshot``, applied to
    the combined parameter table: two same-hash joint records whose
    fitted values disagree raise unless ``overwrite=True`` (their
    projection slots share handles pairwise, so the slot-level collapse
    enforces the same rule per file).
    """

    latest: dict[str, JointFitResult] = {}
    for jr in records:
        stored = latest.get(jr.optimization_hash)
        if stored is not None and not _fitted_values_equivalent(
            stored.params, jr.params, is_long_params=True
        ):
            if not overwrite:
                raise FileExistsError(
                    f"Two joint fits of model {jr.model_name!r} share "
                    f"optimization hash {jr.optimization_hash[:8]} but "
                    f"differ in fitted values — the optimizer configuration "
                    f"is not deterministic (identical inputs produced "
                    f"different optima). " + _divergence_remedy(jr.fit_settings)
                )
            warnings.warn(
                f"Divergent joint re-runs under optimization hash "
                f"{jr.optimization_hash[:8]} (model {jr.model_name!r}): "
                f"keeping the latest (timestamp {jr.timestamp}) and "
                f"dropping the earlier result from this save. Every run "
                f"remains in the in-session history.",
                UserWarning,
                stacklevel=2,
            )
        latest[jr.optimization_hash] = jr
    return list(latest.values())


# ``select=`` keywords on save_fits / export_fits — a label may not shadow
# them, and resolve_fit_reference never receives them (callers intercept).
SELECT_RESERVED: frozenset[str] = frozenset({"all", "latest", "best"})


#
def resolve_fit_reference(
    ref: str,
    *,
    slots: Sequence[SavedFitSlot],
    joint_records: Sequence[JointFitResult] = (),
    labels: bool = True,
) -> SavedFitSlot | JointFitResult:
    """
    Resolve a user-supplied fit reference to one slot or joint record.

    ``ref`` matches a slot by ``handle`` prefix or exact ``label``, and a
    joint record by ``optimization_hash`` prefix or exact ``label``
    (git-style: any unambiguous prefix works; tables display the first 8
    hex chars). Matching is case-insensitive for hex prefixes, exact for
    labels; ``labels=False`` restricts to prefixes (the ``handle=``
    accessor kwarg). Several history entries sharing one handle (exact
    re-runs) count as a single target and resolve to the latest entry.

    Raises
    ------
    LookupError
        If nothing matches, or the reference is ambiguous — the message
        lists every distinct candidate with its short id and context.
    """

    prefix = ref.lower()
    by_handle: dict[str, SavedFitSlot] = {}
    for slot in slots:
        if slot.handle.startswith(prefix) or (
            labels and slot.label is not None and slot.label == ref
        ):
            by_handle[slot.handle] = slot  # latest entry per handle wins
    by_hash: dict[str, JointFitResult] = {}
    for jr in joint_records:
        if jr.optimization_hash.startswith(prefix) or (
            labels and jr.label is not None and jr.label == ref
        ):
            by_hash[jr.optimization_hash] = jr

    n_targets = len(by_handle) + len(by_hash)
    if n_targets == 0:
        raise LookupError(
            f"No fit matches reference {ref!r} — expected a slot-handle "
            f"prefix, a joint-optimization-hash prefix, or an exact label."
        )
    if n_targets > 1:
        candidates = [
            f"slot {s.handle[:8]} (file={s.file_name!r}, "
            f"model={s.model_name!r}, fit_type={s.fit_type!r})"
            for s in by_handle.values()
        ] + [
            f"joint {jr.optimization_hash[:8]} (model={jr.model_name!r}, "
            f"files={list(jr.files)})"
            for jr in by_hash.values()
        ]
        raise LookupError(
            f"Reference {ref!r} is ambiguous — {n_targets} fits match: "
            + "; ".join(candidates)
            + ". Use a longer prefix."
        )
    if by_handle:
        return next(iter(by_handle.values()))
    return next(iter(by_hash.values()))


#
def set_fit_label(target: SavedFitSlot | JointFitResult, label: str) -> None:
    """
    Set the user-facing ``label`` on a slot or joint record.

    ``label`` is the one deliberately mutable display field on the
    otherwise frozen records (on disk it is a rewritable attr; see the
    archive collision rules) — this is its sanctioned in-session mutator,
    used by ``FitResults.label``. Every live view sees the change, since
    ``Project._fit_history`` and ``FitResults`` share the record objects.
    """

    if not isinstance(label, str) or not label:
        raise ValueError("label must be a non-empty string")
    if label in SELECT_RESERVED:
        raise ValueError(
            f"label {label!r} is reserved by select= on save_fits / "
            f"export_fits; pick another label."
        )
    object.__setattr__(target, "label", label)


# Offered ``by=`` criteria for ``select="best"`` (principles §"Pruning and
# selection" — the settled table). Raw χ² and r² are deliberately not
# offered: a fit with more free parameters almost always wins on them while
# being the worse model. ``chi2_red`` requires σ and ranks by |x − 1| — a
# σ-calibrated fit at the noise floor sits at ≈ 1 and overfitting drives it
# *below* 1, so smallest-wins would select the most overfit variant.
_BEST_BY_KEYS: frozenset[str] = frozenset({"aic", "bic", "chi2_red", "chi2_red_raw"})


#
def _slot_metric_scalar(slot: SavedFitSlot, key: str) -> float:
    """
    One comparable number for ranking: the metric, SbS collapsed to its
    per-slice ``nanmedian``. NaN when absent or undefined.
    """

    if key not in slot.metrics:
        return float("nan")
    arr = np.asarray(slot.metrics[key], dtype=float)
    if arr.ndim == 0:
        return float(arr)
    if arr.size == 0 or bool(np.isnan(arr).all()):
        return float("nan")
    return float(np.nanmedian(arr))


#
def _best_score(slot: SavedFitSlot, by: str) -> float:
    """
    Ranking score for ``select="best"`` — smaller is always better.

    ``chi2_red`` scores as ``|x − 1|`` (distance from the noise floor);
    every other offered criterion is minimized directly. NaN when the
    metric is undefined on the slot.
    """

    value = _slot_metric_scalar(slot, by)
    if by == "chi2_red":
        return abs(value - 1.0)
    return value


#
def select_snapshot_slots(
    snapshot: list[SavedFitSlot],
    *,
    select: str,
    by: str | None = None,
    history_order: Sequence[SavedFitSlot] = (),
) -> list[SavedFitSlot]:
    """
    Apply the ``select=`` keyword rule to a collapsed snapshot.

    ``"all"`` keeps every variant. ``"latest"`` and ``"best"`` resolve
    within each ``(file_name, model_name, fit_type)`` group — one winner
    per group. ``"latest"`` picks the most recent *run* (by position in
    ``history_order``, the pre-collapse filtered history — more robust
    than timestamp strings). ``"best"`` requires ``by=``, one of the
    offered criteria (principles §"Pruning and selection"): ``aic``,
    ``bic``, ``chi2_red_raw`` minimize; ``chi2_red`` ranks by |x − 1| and
    requires a σ consistent across the group. Raw χ² and r² are not
    offered — they reward extra free parameters.

    Three rankings are refused because each would silently pick a wrong
    winner: a group spanning more than one ``fit_view_sha256`` (metrics
    of different fit views must never be ranked against one another —
    the same rule ``compare_models`` enforces), ``chi2_red`` over a
    group mixing finite ``sigma_eff`` values, and any ``by=`` undefined
    on every slot of a group. Reference-style ``select`` values never
    reach this function — callers resolve them via
    ``resolve_fit_reference`` first.
    """

    if select == "all":
        return snapshot
    if select not in ("latest", "best"):
        raise ValueError(f"unknown select keyword: {select!r}")

    groups: dict[tuple[str, str, str], list[SavedFitSlot]] = {}
    for slot in snapshot:
        key = (slot.file_name, slot.model_name, slot.fit_type)
        groups.setdefault(key, []).append(slot)

    winners: set[str] = set()
    if select == "latest":
        rank = {s.handle: i for i, s in enumerate(history_order)}
        for group in groups.values():
            winners.add(max(group, key=lambda s: rank.get(s.handle, -1)).handle)
    else:
        if by is None:
            raise ValueError(
                f'select="best" requires by= (one of {sorted(_BEST_BY_KEYS)}).'
            )
        if by not in _BEST_BY_KEYS:
            raise ValueError(
                f"by={by!r} is not an offered selection criterion; use one "
                f"of {sorted(_BEST_BY_KEYS)}. Raw χ² and r² are deliberately "
                f"not offered — a fit with more free parameters almost "
                f"always wins on them while being the worse model."
            )
        for gkey, group in groups.items():
            views = {s.fit_view_sha256 for s in group}
            if len(views) > 1:
                raise ValueError(
                    f"select='best' over group (file={gkey[0]!r}, "
                    f"model={gkey[1]!r}, fit_type={gkey[2]!r}) spans "
                    f"{len(views)} distinct fit views — slots fit against "
                    f"different data views (changed fit limits or "
                    f"correction state) must never be ranked against one "
                    f"another. Narrow the filter, pick an explicit "
                    f"handle/label, or use select='latest'."
                )
            if by == "chi2_red":
                sigmas = sorted(
                    {float(s.sigma_eff) for s in group if np.isfinite(s.sigma_eff)}
                )
                if len(sigmas) > 1:
                    raise ValueError(
                        f"select='best' by='chi2_red' over group "
                        f"(file={gkey[0]!r}, model={gkey[1]!r}, "
                        f"fit_type={gkey[2]!r}) mixes sigma_eff values "
                        f"{sigmas} — values scaled by different σ are not "
                        f"comparable, and a ranking would silently pick a "
                        f"wrong winner. Rank by 'chi2_red_raw' instead, or "
                        f"narrow the filter to one σ."
                    )
            scored = [
                (s, _best_score(s, by))
                for s in group
                if not np.isnan(_best_score(s, by))
            ]
            if not scored:
                raise ValueError(
                    f"select='best' by={by!r}: the metric is undefined on "
                    f"every slot of group (file={gkey[0]!r}, "
                    f"model={gkey[1]!r}, fit_type={gkey[2]!r})"
                    + (
                        " — σ-scaled metrics need a sigma set at fit time "
                        "(File.set_sigma)."
                        if by == "chi2_red"
                        else "."
                    )
                )
            winners.add(min(scored, key=lambda sv: sv[1])[0].handle)

    return [s for s in snapshot if s.handle in winners]


#
def joint_comparability(record: JointFitResult) -> tuple[tuple[str, str], ...]:
    """
    Comparability key of a joint fit: sorted ``(file_name,
    fit_view_sha256)`` pairs over the projections.

    A canonical *tuple of pairs*, never a set of view hashes — a set
    would discard association and multiplicity, collapsing two
    byte-identical files under different names into one entry and making
    a two-file joint fit look like a one-file one. Computed on read from
    the projections; there is no stored field.
    """

    return tuple(
        sorted((p.slot.file_name, p.slot.fit_view_sha256) for p in record.projections)
    )


#
# --- archive lookup helpers -------------------------------------------------
#


#
def _find_file_by_name(
    project_group: h5py.Group,
    name: str,
) -> h5py.Group | None:
    """
    Look up a file group by its ``name`` attr under ``project/files/``.

    The name is the file's identity (fit_archive_principles.md,
    Principle 1); ``file_content_hash`` is deliberately not matched here —
    content divergence is the writer's integrity raise or the reader's
    staleness report, never a lookup miss.

    Returns the first matching ``files/<id>/`` group in positional-key
    order, or ``None``.
    """

    files_obj = project_group.get("files")
    if files_obj is None:
        return None
    files_group = require_group(files_obj, "files")
    for key in sorted(files_group.keys()):
        fg = require_group(files_group[key], f"files/{key}")
        meta = require_group(fg["metadata"], f"files/{key}/metadata")
        if str(meta.attrs.get("name", "")) == name:
            return fg
    return None


#
def _find_slot_by_handle(
    file_group: h5py.Group,
    handle: str,
) -> h5py.Group | None:
    """
    Look up a slot inside a file group by its stored ``handle``.

    The handle is the slot's primary key — stored, never recomputed on
    read. Returns ``None`` if no slot under ``file_group/slots/`` carries
    it.
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
        if str(meta.attrs.get("handle", "")) == handle:
            return slot_group
    return None


#
def _find_joint_by_hash(
    joint_parent: h5py.Group,
    optimization_hash: str,
) -> h5py.Group | None:
    """
    Look up a joint record under ``project/joint/`` by its
    ``optimization_hash`` attr.
    """

    for key in sorted(joint_parent.keys()):
        jg = require_group(joint_parent[key], f"joint/{key}")
        meta = require_group(jg["metadata"], f"joint/{key}/metadata")
        if str(meta.attrs.get("optimization_hash", "")) == optimization_hash:
            return jg
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
        ds = _create_array_dataset(group, name, values)
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
    ds = _create_array_dataset(group, name, arr)
    ds.attrs["columns"] = np.array(columns, dtype=_VLEN_STR)
    ds.attrs["dtypes"] = np.array(tags, dtype=_VLEN_STR)
    return ds


#
# --- HDF5 writer ------------------------------------------------------------
#

_ARCHIVE_FORMAT = "trspecfit-fit-archive"
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
# SbS shared per-parameter metadata (slice-invariant columns only).
_PARAMS_META_TYPE_TAGS: list[TypeTag] = [
    "str",  # name
    "bool",  # vary
    "float64",  # min
    "float64",  # max
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
# Residual-only metrics are defined for every non-sbs slot. The
# degrees-of-freedom metrics divide by (or penalize) a parameter count,
# which does not decompose per file for a joint fit — omitted on disk for
# joint projections, rehydrated as NaN by the reader.
_METRICS_RESIDUAL_ONLY = ("chi2_raw", "chi2", "r2")
_METRICS_DOF = ("chi2_red_raw", "chi2_red", "aic", "bic")
# Whole-objective metrics on a joint record; r2 is structurally undefined
# for a joint result and omitted on disk.
_JOINT_METRICS_KEYS = ("chi2_raw", "chi2_red_raw", "chi2", "chi2_red", "aic", "bic")
# Optional post-fit payloads that merge individually under the four-case
# collision rules (see write_archive).
_RESULT_ATTACHMENTS = ("conf_ci", "correl", "mcmc")
# Fitted-value equivalence for handle collisions
# (fit_archive_principles.md §"Equivalence is defined, not loose"):
# |a − b| <= atol + rtol·|b|, per parameter matched by name. The atol term
# is not optional — a parameter converging to ±1e-15 across two runs
# differs by 200% relatively and by nothing that matters. The one chosen
# (not derived) number pair in the design; tune here, with evidence.
_PARAMS_EQUIV_RTOL = 1e-6
_PARAMS_EQUIV_ATOL = 1e-12


#
def _all_float64_tags(n: int) -> list[TypeTag]:
    """All-float64 tag list, typed properly for ``_encode_dataframe``."""

    return ["float64"] * n


#
def _input_files_scope(input_files_json: str) -> str:
    """Scope field (``"file"`` | ``"project"``) of an ``input_files`` JSON."""

    return str(json.loads(input_files_json)[0])


#
def _create_array_dataset(group: h5py.Group, name: str, data: Any) -> h5py.Dataset:
    """
    Create an array dataset with lossless compression (gzip + shuffle).

    Empty arrays are stored uncompressed: HDF5 filters require chunked
    storage, and h5py cannot chunk a zero-size dataset.
    """

    arr = np.ascontiguousarray(data)
    if arr.size == 0:
        return group.create_dataset(name, data=arr)
    return group.create_dataset(name, data=arr, compression="gzip", shuffle=True)


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

    - **Append-mode by default.** If ``filepath`` exists, files, slots, and
      joint records are added in place. The archive's ``timestamp_created``
      is preserved; ``timestamp_updated`` and ``project/plot_config`` are
      rewritten on every save. To start fresh, pass a new path.
    - **One project per archive.** The project name is written once; an
      append under a different project name raises ``ValueError``.
    - **File-content integrity.** File groups are matched by name; an
      incoming file whose ``file_content_hash`` differs from its stored
      group raises ``ValueError`` in **both** overwrite modes, before any
      mutation — ``overwrite=`` is slot-scoped and does not authorize
      filing fits under another measurement's data.
    - **Bundle integrity.** Every joint record's projection slots must be
      part of the write (or already stored), every project-scoped slot's
      ``joint_ref`` must resolve to a joint record, and each projection's
      parameter map must be total in both directions; violations raise
      ``ValueError`` before any mutation, regardless of ``overwrite``.
    - **Collision rules** for a slot (same stored ``handle``) or joint
      record (same stored ``optimization_hash``): if the stored ``params``
      differ from the incoming ones, the write is a hard conflict and
      requires ``overwrite=True`` (full replacement). If they agree, the
      attachments (``conf_ci`` / ``correl`` / ``mcmc``) merge
      individually — absent in the archive: written freely (enrichment);
      stored but absent incoming: kept, never deleted; present on both
      sides: requires ``overwrite=True`` and is replaced individually.
      ``label`` is mutable and rewritten whenever the incoming record
      carries one. All collisions are detected before any mutation, so a
      conflicting append leaves the archive byte-untouched.
    """

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "a") as archive:
        is_new = _classify_archive_for_write(archive, project, path=path)
        _precheck_bundle_integrity(archive, project, archive_is_new=is_new)
        if not is_new:
            _precheck_file_content(archive, project, path=path)
            if not overwrite:
                _precheck_collisions(archive, project)
        _write_top_metadata(archive, project, is_new=is_new)
        project_group = archive.require_group("project")
        if is_new:
            project_group.attrs["name"] = project.name
        project_group.attrs["plot_config"] = project.plot_config.to_json()
        files_group = project_group.require_group("files")
        for sf in project.files:
            file_group = _find_file_by_name(project_group, sf.name)
            if file_group is None:
                key = _next_positional_key(files_group)
                file_group = files_group.create_group(key)
                _write_file_payload(file_group, sf)
            for slot in sf.slots:
                _write_slot(file_group, slot, overwrite=overwrite)
        if project.joint:
            joint_parent = project_group.require_group("joint")
            for jr in project.joint:
                _write_joint(joint_parent, jr, overwrite=overwrite)


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
    metadata (foreign HDF5 or partially-written archive), if its
    ``schema_version`` does not match the writer's, or if it stores a
    different project's fits (one project per archive). h5py's ``"a"`` mode
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
        project_obj = archive.get("project")
        if project_obj is None:
            raise ValueError(
                f"File at {path} has fit-archive metadata but no project/ "
                f"group — a partially written archive. Choose a different "
                f"path or remove the existing file."
            )
        stored_name = str(require_group(project_obj, "project").attrs.get("name", ""))
        if stored_name != project.name:
            raise ValueError(
                f"Archive at {path} stores project {stored_name!r}; cannot "
                f"append project {project.name!r} — one project per archive. "
                f"Choose a new path."
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
def _precheck_bundle_integrity(
    archive: h5py.File,
    project: SavedProject,
    *,
    archive_is_new: bool,
) -> None:
    """
    Raise ``ValueError`` on any joint-bundle integrity violation.

    Checks, per ``fit_archive_schema_plan.md`` §``project/``:

    - the scope/reference invariant — a slot carries ``joint_ref`` exactly
      when its ``input_files`` scope is ``"project"``, and a joint
      record's scope is always ``"project"``;
    - resolution in both directions — every projection slot is part of
      this write (or already stored) and carries its record's
      ``optimization_hash`` as ``joint_ref``; every project-scoped slot's
      ``joint_ref`` resolves to a joint record. Partial bundles are not
      representable;
    - parameter-map totality in both directions — each map's value set
      equals its projection slot's parameter names, and the union of all
      key sets equals the combined parameter names.

    Runs before any mutation, regardless of ``overwrite``.
    """

    written_handles = {slot.handle for sf in project.files for slot in sf.slots}
    written_joint = {jr.optimization_hash for jr in project.joint}
    stored_handles, stored_joint = _stored_identity_sets(archive, archive_is_new)

    for sf in project.files:
        for slot in sf.slots:
            scope = _input_files_scope(slot.input_files)
            if (scope == "project") != (slot.joint_ref is not None):
                joint_ref_state = "set" if slot.joint_ref is not None else "absent"
                raise ValueError(
                    f"Slot {slot.handle[:8]} on file {slot.file_name!r} "
                    f"violates the joint-reference invariant: input_files "
                    f"scope is {scope!r} but joint_ref is {joint_ref_state}."
                )
            if (
                slot.joint_ref is not None
                and slot.joint_ref not in written_joint
                and slot.joint_ref not in stored_joint
            ):
                raise ValueError(
                    f"Slot {slot.handle[:8]} on file {slot.file_name!r} "
                    f"references joint record {slot.joint_ref[:8]}, which is "
                    f"neither part of this save nor stored in the archive — "
                    f"a joint bundle must be saved whole."
                )

    for jr in project.joint:
        scope = _input_files_scope(jr.input_files)
        if scope != "project":
            raise ValueError(
                f"Joint record {jr.optimization_hash[:8]} has input_files "
                f"scope {scope!r}; joint optimizations are always "
                f"project-scoped."
            )
        pairs: list[tuple[str, Mapping[str, str], pd.DataFrame]] = []
        for proj in jr.projections:
            slot = proj.slot
            if slot.joint_ref != jr.optimization_hash:
                raise ValueError(
                    f"Joint record {jr.optimization_hash[:8]}: projection "
                    f"slot {slot.handle[:8]} (file {slot.file_name!r}) "
                    f"carries joint_ref "
                    f"{'absent' if slot.joint_ref is None else slot.joint_ref[:8]} "
                    f"instead of its record's optimization_hash."
                )
            if slot.handle not in written_handles and slot.handle not in stored_handles:
                raise ValueError(
                    f"Joint record {jr.optimization_hash[:8]} declares "
                    f"projection slot {slot.handle[:8]} (file "
                    f"{slot.file_name!r}), which is neither part of this "
                    f"save nor stored in the archive — a joint bundle must "
                    f"be saved whole."
                )
            pairs.append((slot.file_name, proj.parameter_map, slot.params))
        _assert_parameter_maps_consistent(
            context=f"Joint record {jr.optimization_hash[:8]}",
            combined_params=jr.params,
            projections=pairs,
        )


#
def _assert_parameter_maps_consistent(
    *,
    context: str,
    combined_params: pd.DataFrame,
    projections: Sequence[tuple[str, Mapping[str, str], pd.DataFrame]],
) -> None:
    """
    Raise ``ValueError`` unless every parameter map is total in both
    directions and every projected value agrees with the combined table.

    ``projections`` holds ``(file_name, parameter_map, slot_params)``
    triples. Local half: each map's value set equals its slot's parameter
    names. Combined half: the union of all key sets equals the combined
    parameter names exactly. Value half: each mapped slot value equals
    the combined table's value within the named equivalence tolerances —
    a projection that disagrees with the authoritative combined result is
    a corrupt bundle, not a variant. Shared by the writer's bundle
    precheck and the reader — this is what makes the materialized-view
    invariant checkable rather than asserted (fit_archive_schema_plan.md
    §Projection records).
    """

    combined_values = {
        str(n): float(v)
        for n, v in zip(combined_params["name"], combined_params["value"], strict=True)
    }
    mapped: set[str] = set()
    for file_name, parameter_map, slot_params in projections:
        local_values = {
            str(n): float(v)
            for n, v in zip(slot_params["name"], slot_params["value"], strict=True)
        }
        values = set(parameter_map.values())
        if values != set(local_values):
            raise ValueError(
                f"{context}: the parameter map for file {file_name!r} is "
                f"not total against its slot "
                f"(map-only: {sorted(values - set(local_values))}, "
                f"slot-only: {sorted(set(local_values) - values)})."
            )
        disagreeing = sorted(
            combined_name
            for combined_name, local_name in parameter_map.items()
            if combined_name in combined_values
            and not np.isclose(
                local_values[local_name],
                combined_values[combined_name],
                rtol=_PARAMS_EQUIV_RTOL,
                atol=_PARAMS_EQUIV_ATOL,
                equal_nan=True,
            )
        )
        if disagreeing:
            raise ValueError(
                f"{context}: file {file_name!r} projected parameter values "
                f"disagree with the combined table for {disagreeing} "
                f"(beyond rtol={_PARAMS_EQUIV_RTOL}, "
                f"atol={_PARAMS_EQUIV_ATOL})."
            )
        mapped |= set(parameter_map.keys())
    if mapped != set(combined_values):
        raise ValueError(
            f"{context}: projection parameter maps do not cover the "
            f"combined parameter table exactly "
            f"(unmapped: {sorted(set(combined_values) - mapped)}, "
            f"stray: {sorted(mapped - set(combined_values))})."
        )


#
def _stored_identity_sets(
    archive: h5py.File,
    archive_is_new: bool,
) -> tuple[set[str], set[str]]:
    """All stored slot handles and joint optimization hashes."""

    stored_handles: set[str] = set()
    stored_joint: set[str] = set()
    if archive_is_new:
        return stored_handles, stored_joint
    project_group = require_group(archive["project"], "project")
    files_obj = project_group.get("files")
    if files_obj is not None:
        files_group = require_group(files_obj, "files")
        for fkey in files_group.keys():
            fg = require_group(files_group[fkey], f"files/{fkey}")
            slots_obj = fg.get("slots")
            if slots_obj is None:
                continue
            slots_group = require_group(slots_obj, f"files/{fkey}/slots")
            for skey in slots_group.keys():
                sg = require_group(slots_group[skey], f"slots/{skey}")
                smeta = require_group(sg["metadata"], f"slots/{skey}/metadata")
                stored_handles.add(str(smeta.attrs.get("handle", "")))
    joint_obj = project_group.get("joint")
    if joint_obj is not None:
        joint_group = require_group(joint_obj, "joint")
        for jkey in joint_group.keys():
            jg = require_group(joint_group[jkey], f"joint/{jkey}")
            jmeta = require_group(jg["metadata"], f"joint/{jkey}/metadata")
            stored_joint.add(str(jmeta.attrs.get("optimization_hash", "")))
    return stored_handles, stored_joint


#
def _precheck_file_content(
    archive: h5py.File, project: SavedProject, *, path: Path
) -> None:
    """
    Raise ``ValueError`` if an incoming file's content differs from the
    stored group it would append into.

    File groups are matched by name (Principle 1), so without this check
    a reused name — or in-place mutation of ``data_raw`` — would file new
    slots under another data payload's group. The check is unconditional:
    ``overwrite=`` is slot-scoped and does not authorize re-associating a
    measurement. Runs before any mutation, so a failed append leaves the
    archive byte-untouched.
    """

    project_group = require_group(archive["project"], "project")
    for sf in project.files:
        existing_fg = _find_file_by_name(project_group, sf.name)
        if existing_fg is None:
            continue
        meta = require_group(existing_fg["metadata"], "metadata")
        if _attr_str(meta.attrs["file_content_hash"]) != sf.file_content_hash:
            raise ValueError(
                f"Archive {path} already stores file {sf.name!r} with "
                f"different content (file_content_hash mismatch). Appending "
                f"would file new fits under data they did not run against — "
                f"the name was reused for a different measurement, or the "
                f"raw data was mutated in place. Save to a new archive path."
            )


#
def _precheck_collisions(archive: h5py.File, project: SavedProject) -> None:
    """
    Raise ``FileExistsError`` on any collision that would need ``overwrite``.

    Applies the four-case collision rules (see ``write_archive``) to every
    incoming slot and joint record before any mutation, so a single
    conflict cannot leave half the payload written.
    """

    project_group = require_group(archive["project"], "project")
    for sf in project.files:
        existing_fg = _find_file_by_name(project_group, sf.name)
        if existing_fg is None:
            continue
        for slot in sf.slots:
            existing = _find_slot_by_handle(existing_fg, slot.handle)
            if existing is None:
                continue
            differ, both = _result_conflicts(
                existing,
                params=slot.params,
                is_long_params=slot.fit_type != "sbs",
                attachments=_slot_attachments(slot),
            )
            _raise_for_conflicts(
                differ,
                both,
                context=(
                    f"Slot {slot.handle[:8]} (file={slot.file_name!r}, "
                    f"model={slot.model_name!r}, fit_type={slot.fit_type!r})"
                ),
            )
    joint_obj = project_group.get("joint")
    if joint_obj is None:
        return
    joint_parent = require_group(joint_obj, "joint")
    for jr in project.joint:
        existing_jg = _find_joint_by_hash(joint_parent, jr.optimization_hash)
        if existing_jg is None:
            continue
        differ, both = _result_conflicts(
            existing_jg,
            params=jr.params,
            is_long_params=True,
            attachments=_joint_attachments(jr),
        )
        _raise_for_conflicts(
            differ, both, context=f"Joint record {jr.optimization_hash[:8]}"
        )


#
def _raise_for_conflicts(
    differ: bool,
    both: Sequence[str],
    *,
    context: str,
) -> None:
    """``FileExistsError`` for the two collision cases that need opt-in."""

    if differ:
        raise FileExistsError(
            f"{context} exists in the archive with differing fitted "
            f"parameters; pass overwrite=True to replace it."
        )
    if both:
        raise FileExistsError(
            f"{context} already stores {', '.join(both)}; pass "
            f"overwrite=True to replace (absent attachments enrich "
            f"without overwrite)."
        )


#
def _slot_attachments(slot: SavedFitSlot) -> dict[str, Any]:
    """A slot's optional post-fit payloads, keyed by dataset/group name."""

    return {"conf_ci": slot.conf_ci, "correl": slot.correl, "mcmc": slot.mcmc}


#
def _joint_attachments(jr: JointFitResult) -> dict[str, Any]:
    """A joint record's optional payloads, mcmc converted to the wire dict."""

    mcmc = _payload_from_mcmc_result(jr.mcmc) if jr.mcmc is not None else None
    return {"conf_ci": jr.conf_ci, "correl": jr.correl, "mcmc": mcmc}


#
def _payload_from_mcmc_result(mcmc: MCMCResult) -> dict[str, Any]:
    """Inverse of ``mcmc_result_from_payload`` — the writer-side encoding."""

    return {
        "flatchain": mcmc.flatchain,
        "ci": mcmc.table if not mcmc.table.empty else None,
        "lnsigma": mcmc.lnsigma,
        "acceptance_fraction": mcmc.acceptance_fraction,
    }


#
def _result_conflicts(
    existing: h5py.Group,
    *,
    params: pd.DataFrame,
    is_long_params: bool,
    attachments: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    """
    Classify a handle collision against the stored group.

    Returns ``(params_differ, attachments_present_on_both_sides)`` — the
    inputs to the four-case collision rules. ``params`` equality is
    value-level with ``NaN == NaN``, after the same ``None`` restoration
    the reader applies, so re-saving an identical result classifies as
    agreement rather than a byte-level mismatch.
    """

    stored = _decode_dataframe(require_dataset(existing["params"], "params"))
    if is_long_params:
        _restore_long_params_nones(stored)
    differ = not _fitted_values_equivalent(
        stored, params, is_long_params=is_long_params
    )
    both = [k for k, v in attachments.items() if v is not None and k in existing]
    return differ, both


#
def _fitted_values_equivalent(
    stored: pd.DataFrame,
    incoming: pd.DataFrame,
    *,
    is_long_params: bool,
) -> bool:
    """
    Fitted-parameter equivalence for handle collisions.

    Per fit_archive_principles.md §"Equivalence is defined, not loose":
    compare fitted **values only**, per parameter matched by name, with
    ``|a − b| <= _PARAMS_EQUIV_ATOL + _PARAMS_EQUIV_RTOL·|b|`` and
    ``NaN == NaN``. Everything else in the frame is either guaranteed
    identical by the matching hash (bounds, vary, expr, quantized init)
    or legitimately non-identical across re-runs of the same optimum —
    ``stderr`` is not hashed, so numdifftools present vs absent, or a CI
    re-run, must not read as a divergent optimum and force an
    ``overwrite=True`` that would replace stored attachments.

    Long-form frames match rows by the ``name`` column; SbS wide-form
    frames match columns by parameter name, row-for-row (slice order is
    positional identity).
    """

    if is_long_params:
        a = {
            str(n): float(v)
            for n, v in zip(stored["name"], stored["value"], strict=True)
        }
        b = {
            str(n): float(v)
            for n, v in zip(incoming["name"], incoming["value"], strict=True)
        }
        if a.keys() != b.keys():
            return False
        names = list(b)
        stored_values = np.array([a[n] for n in names])
        incoming_values = np.array([b[n] for n in names])
    else:
        stored_cols = sorted(str(c) for c in stored.columns)
        incoming_cols = sorted(str(c) for c in incoming.columns)
        if stored_cols != incoming_cols or len(stored) != len(incoming):
            return False
        stored_values = stored[incoming_cols].to_numpy(dtype=np.float64)
        incoming_values = incoming[incoming_cols].to_numpy(dtype=np.float64)
    return bool(
        np.allclose(
            stored_values,
            incoming_values,
            rtol=_PARAMS_EQUIV_RTOL,
            atol=_PARAMS_EQUIV_ATOL,
            equal_nan=True,
        )
    )


#
def _merge_result_group(
    existing: h5py.Group,
    *,
    attachments: Mapping[str, Any],
    both: Sequence[str],
    label: str | None,
    overwrite: bool,
    context: str,
) -> None:
    """
    Enrich a stored result whose fitted parameters agree with the incoming
    record.

    Attachments merge individually: absent-in-archive ones are written,
    stored ones the incoming record lacks are kept, and present-on-both
    ones require ``overwrite`` (each is then replaced in place). ``label``
    is mutable and rewritten whenever the incoming record carries one.
    """

    if both and not overwrite:
        # Should have been caught by _precheck_collisions; defense in
        # depth in case the writer is called directly without precheck.
        _raise_for_conflicts(False, both, context=context)
    for name, value in attachments.items():
        if value is None:
            continue
        if name in existing:
            del existing[name]
        _write_result_attachment(existing, name, value)
    if label is not None:
        meta = require_group(existing["metadata"], "metadata")
        meta.attrs["label"] = label


#
def _write_result_attachment(group: h5py.Group, name: str, value: Any) -> None:
    """Write one ``conf_ci`` / ``correl`` / ``mcmc`` attachment payload."""

    if name == "conf_ci":
        _encode_dataframe(group, "conf_ci", value)
    elif name == "correl":
        # Square all-float matrix; index == columns, so only the columns
        # attr is stored and the reader restores the index from it.
        _encode_dataframe(
            group,
            "correl",
            value,
            type_tags=_all_float64_tags(len(value.columns)),
        )
    elif name == "mcmc":
        _write_mcmc_group(group, value)
    else:
        raise AssertionError(f"unknown result attachment {name!r}")


#
def _write_top_metadata(
    archive: h5py.File, project: SavedProject, *, is_new: bool
) -> None:
    """Write or update the top-level ``metadata`` group attrs."""

    meta = archive.require_group("metadata")
    meta.attrs["trspecfit_version"] = project.trspecfit_version
    meta.attrs["timestamp_updated"] = project.timestamp_updated
    if is_new:
        meta.attrs["format"] = _ARCHIVE_FORMAT
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
    meta.attrs["file_content_hash"] = sf.file_content_hash
    # Preserve source dtype: `file_content_hash` covers dtype and the
    # original bytes, so casting on write would invalidate it for
    # non-float64 inputs.
    _create_array_dataset(file_group, "energy", sf.energy)
    _create_array_dataset(file_group, "time", sf.time)
    _create_array_dataset(file_group, "data_raw", sf.data_raw)
    if sf.aux_axis is not None:
        _create_array_dataset(file_group, "aux_axis", sf.aux_axis)
    file_group.create_group("slots")


#
def _write_slot(
    file_group: h5py.Group,
    slot: SavedFitSlot,
    *,
    overwrite: bool,
) -> None:
    """Append one slot under ``file_group``, applying the collision rules."""

    slots_group = file_group.require_group("slots")
    existing = _find_slot_by_handle(file_group, slot.handle)
    if existing is not None:
        attachments = _slot_attachments(slot)
        differ, both = _result_conflicts(
            existing,
            params=slot.params,
            is_long_params=slot.fit_type != "sbs",
            attachments=attachments,
        )
        if not differ:
            _merge_result_group(
                existing,
                attachments=attachments,
                both=both,
                label=slot.label,
                overwrite=overwrite,
                context=f"Slot {slot.handle[:8]} (file={slot.file_name!r})",
            )
            return
        if not overwrite:
            # Should have been caught by _precheck_collisions; defense in
            # depth in case the writer is called directly without precheck.
            _raise_for_conflicts(
                differ,
                both,
                context=f"Slot {slot.handle[:8]} (file={slot.file_name!r})",
            )
        existing_name = cast(str | None, existing.name)
        assert existing_name is not None  # type guard
        del slots_group[existing_name.rsplit("/", 1)[-1]]

    key = _next_positional_key(slots_group)
    slot_group = slots_group.create_group(key)
    _write_slot_metadata(slot_group, slot)
    _write_slot_params(slot_group, slot)
    # Preserve source dtype: the slot's `fit_view_sha256` covers dtype and
    # the original bytes, so casting on write would invalidate the
    # cross-check for slots whose observed array was non-float64 (e.g. sbs
    # slice from float32 data).
    _create_array_dataset(slot_group, "observed", slot.observed)
    _create_array_dataset(slot_group, "fit", slot.fit)
    if slot.dark is not None:
        _create_array_dataset(slot_group, "dark", slot.dark)
    if slot.calibration is not None:
        _create_array_dataset(slot_group, "calibration", slot.calibration)
    if slot.fit_type == "sbs":
        _write_metrics_per_slice(slot_group, slot.metrics)
    if slot.params_meta is not None:
        _encode_dataframe(
            slot_group,
            "params_meta",
            slot.params_meta,
            type_tags=_PARAMS_META_TYPE_TAGS,
        )
    if slot.params_stderr is not None:
        _encode_dataframe(
            slot_group,
            "params_stderr",
            slot.params_stderr,
            type_tags=_all_float64_tags(len(slot.params_stderr.columns)),
        )
    if slot.params_init is not None:
        _encode_dataframe(
            slot_group,
            "params_init",
            slot.params_init,
            type_tags=_all_float64_tags(len(slot.params_init.columns)),
        )
    for name, value in _slot_attachments(slot).items():
        if value is not None:
            _write_result_attachment(slot_group, name, value)
    if slot.components is not None:
        _create_array_dataset(slot_group, "components", slot.components)
        assert slot.component_names is not None  # type guard
        _create_array_dataset(
            slot_group,
            "component_names",
            np.array(slot.component_names, dtype=_VLEN_STR),
        )
    if slot.fit_ini is not None:
        _create_array_dataset(slot_group, "fit_ini", slot.fit_ini)
    if slot.model_yaml is not None:
        _write_model_yaml(slot_group, slot.model_yaml)


#
def _write_slot_metadata(slot_group: h5py.Group, slot: SavedFitSlot) -> None:
    """Identity + provenance + noise + (non-sbs) scalar metric attrs."""

    meta = slot_group.create_group("metadata")
    meta.attrs["handle"] = slot.handle
    meta.attrs["optimization_hash"] = slot.optimization_hash
    meta.attrs["input_files"] = slot.input_files
    meta.attrs["model_structure"] = slot.model_structure
    meta.attrs["fit_view_sha256"] = slot.fit_view_sha256
    meta.attrs["fit_type"] = slot.fit_type
    meta.attrs["model_name"] = slot.model_name
    meta.attrs["fit_alg"] = slot.fit_alg
    if slot.fit_settings is not None:
        meta.attrs["fit_settings"] = json.dumps(slot.fit_settings, sort_keys=True)
    meta.attrs["timestamp"] = slot.timestamp
    if slot.label is not None:
        meta.attrs["label"] = slot.label
    if slot.joint_ref is not None:
        meta.attrs["joint_ref"] = slot.joint_ref
    # Noise metadata snapshot at fit time — see SavedFitSlot docstring.
    meta.attrs["noise_type"] = slot.noise_type
    meta.attrs["sigma_source"] = slot.sigma_source
    meta.attrs["sigma_type"] = slot.sigma_type
    meta.attrs["sigma_data"] = float(slot.sigma_data)
    meta.attrs["sigma_eff"] = float(slot.sigma_eff)
    if slot.fit_type != "sbs":
        for k in _METRICS_RESIDUAL_ONLY:
            meta.attrs[k] = float(slot.metrics[k])
        # The DoF metrics are omitted for joint projections — keyed on
        # scope, not fit_type, so a future project-level SbS needs no
        # change here.
        if _input_files_scope(slot.input_files) == "file":
            for k in _METRICS_DOF:
                meta.attrs[k] = float(slot.metrics[k])


#
def _write_model_yaml(
    slot_group: h5py.Group,
    records: Sequence[ModelYamlRecord],
) -> None:
    """
    ``model_yaml/`` group: one scalar vlen-utf8 dataset per snippet.

    Snippet metadata rides as attrs on its own dataset, so a mismatched
    record is structurally unrepresentable. Scalar datasets cannot be
    chunked, hence never compressed — the one exception to the
    compressed-datasets convention.
    """

    my_group = slot_group.create_group("model_yaml")
    for index, rec in enumerate(records):
        ds = my_group.create_dataset(f"{index:06d}", data=rec.text, dtype=_VLEN_STR)
        ds.attrs["role"] = rec.role
        ds.attrs["name"] = rec.name
        ds.attrs["source_file"] = rec.source_file
        if rec.target_par is not None:
            ds.attrs["target_par"] = rec.target_par
        if rec.sequence_index is not None:
            ds.attrs["sequence_index"] = np.int64(rec.sequence_index)


#
def _write_joint(
    joint_parent: h5py.Group,
    jr: JointFitResult,
    *,
    overwrite: bool,
) -> None:
    """
    Append one joint record under ``project/joint/``, applying the
    collision rules.
    """

    existing = _find_joint_by_hash(joint_parent, jr.optimization_hash)
    attachments = _joint_attachments(jr)
    if existing is not None:
        differ, both = _result_conflicts(
            existing,
            params=jr.params,
            is_long_params=True,
            attachments=attachments,
        )
        if not differ:
            _merge_result_group(
                existing,
                attachments=attachments,
                both=both,
                label=jr.label,
                overwrite=overwrite,
                context=f"Joint record {jr.optimization_hash[:8]}",
            )
            return
        if not overwrite:
            # Should have been caught by _precheck_collisions; defense in
            # depth in case the writer is called directly without precheck.
            _raise_for_conflicts(
                differ,
                both,
                context=f"Joint record {jr.optimization_hash[:8]}",
            )
        existing_name = cast(str | None, existing.name)
        assert existing_name is not None  # type guard
        del joint_parent[existing_name.rsplit("/", 1)[-1]]

    key = _next_positional_key(joint_parent)
    jg = joint_parent.create_group(key)
    meta = jg.create_group("metadata")
    meta.attrs["optimization_hash"] = jr.optimization_hash
    meta.attrs["input_files"] = jr.input_files
    meta.attrs["model_structure"] = jr.model_structure
    meta.attrs["model_name"] = jr.model_name
    meta.attrs["projections"] = _projections_json(jr)
    if jr.label is not None:
        meta.attrs["label"] = jr.label
    meta.attrs["fit_alg"] = jr.fit_alg
    meta.attrs["fit_settings"] = json.dumps(dict(jr.fit_settings), sort_keys=True)
    meta.attrs["timestamp"] = jr.timestamp
    # r2 is structurally undefined for a joint result — omitted on disk,
    # rehydrated as NaN by the reader.
    for k in _JOINT_METRICS_KEYS:
        meta.attrs[k] = float(jr.metrics[k])
    _encode_dataframe(jg, "params", jr.params, type_tags=_PARAMS_LONG_TYPE_TAGS)
    for name, value in attachments.items():
        if value is not None:
            _write_result_attachment(jg, name, value)


#
def _projections_json(jr: JointFitResult) -> str:
    """
    Canonical ``projections`` attr: records sorted by file name.

    The parameter map — not record order and never a ``fileNN_`` prefix —
    carries the association with the combined optimizer parameters
    (fit_archive_schema_plan.md §Projection records).
    """

    records = [
        {
            "file_name": proj.slot.file_name,
            "handle": proj.slot.handle,
            "parameter_map": dict(sorted(proj.parameter_map.items())),
        }
        for proj in sorted(jr.projections, key=lambda p: p.slot.file_name)
    ]
    return json.dumps(records, sort_keys=True, separators=(",", ":"))


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
    _create_array_dataset(slot_group, "metrics_per_slice", out)


#
def _write_mcmc_group(slot_group: h5py.Group, mcmc: dict[str, Any]) -> None:
    """``mcmc/`` subgroup: flatchain (always), ci / acceptance_fraction
    (optional), lnsigma attr."""

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
    acceptance = mcmc.get("acceptance_fraction")
    if acceptance is not None:
        _create_array_dataset(
            mcmc_group,
            "acceptance_fraction",
            np.asarray(acceptance, dtype=np.float64),
        )


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
def _read_array(dataset: h5py.Dataset) -> np.ndarray:
    """
    Read a dataset into a read-only array.

    The write flag is cleared so the returned records are snapshots, not
    views anything downstream can mutate (Principle 4 — the same
    ownership boundary capture enforces on the write side).
    """

    arr = np.asarray(dataset[...])
    arr.flags.writeable = False
    return arr


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
    return {
        k: _frozen_copy(np.asarray(arr[k], dtype=np.float64)) for k in _METRICS_KEYS
    }


#
def _read_mcmc_group(group: h5py.Group) -> dict[str, Any]:
    """
    Inverse of ``_write_mcmc_group``.

    Returns ``{"flatchain", "ci", "lnsigma", "acceptance_fraction"}``
    matching the writer's payload. ``lnsigma`` NaN maps back to ``None``;
    ``ci`` and ``acceptance_fraction`` are ``None`` if the optional
    dataset was not written.
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
    acc_obj = group.get("acceptance_fraction")
    acceptance = (
        _read_array(require_dataset(acc_obj, "mcmc/acceptance_fraction"))
        if acc_obj is not None
        else None
    )
    return {
        "flatchain": flatchain,
        "ci": ci,
        "lnsigma": lnsigma,
        "acceptance_fraction": acceptance,
    }


#
def _read_result_attachments(
    group: h5py.Group,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict[str, Any] | None]:
    """
    Decode the optional ``conf_ci`` / ``correl`` / ``mcmc`` payloads.

    Shared by the slot and joint readers — the attachment encodings are
    identical on both group kinds.
    """

    conf_ci_obj = group.get("conf_ci")
    conf_ci = (
        _decode_dataframe(require_dataset(conf_ci_obj, "conf_ci"))
        if conf_ci_obj is not None
        else None
    )
    correl_obj = group.get("correl")
    correl: pd.DataFrame | None = None
    if correl_obj is not None:
        correl = _decode_dataframe(require_dataset(correl_obj, "correl"))
        # Square matrix stores only column labels; index == columns.
        correl.index = pd.Index(correl.columns)
    mcmc_obj = group.get("mcmc")
    mcmc = (
        _read_mcmc_group(require_group(mcmc_obj, "mcmc"))
        if mcmc_obj is not None
        else None
    )
    return conf_ci, correl, mcmc


#
def _selection_json_for(input_files: str, *, file_name: str, handle: str) -> str:
    """
    This file's selection from the slot's ``input_files`` entries.

    Schema 7 stores selection only inside ``input_files`` (where it is
    hashed into identity); a slot's own fit-view selection is the entry
    matching its file name.
    """

    entries = json.loads(input_files)[1]
    for name, _stamp, selection_json in entries:
        if name == file_name:
            return str(selection_json)
    raise ValueError(
        f"Slot {handle[:8]} on file {file_name!r}: input_files has no "
        f"entry for its own file — corrupt archive."
    )


#
def _read_model_yaml(group: h5py.Group) -> tuple[ModelYamlRecord, ...]:
    """Decode the ``model_yaml/`` snippet group, in positional-key order."""

    records: list[ModelYamlRecord] = []
    for key in sorted(group.keys()):
        ds = require_dataset(group[key], f"model_yaml/{key}")
        da = ds.attrs
        records.append(
            ModelYamlRecord(
                role=_attr_str(da["role"]),
                name=_attr_str(da["name"]),
                source_file=_attr_str(da["source_file"]),
                target_par=(
                    _attr_str(da["target_par"]) if "target_par" in da else None
                ),
                sequence_index=(
                    int(np.asarray(da["sequence_index"]).item())
                    if "sequence_index" in da
                    else None
                ),
                text=_to_str_value(ds[()]),
            )
        )
    return tuple(records)


#
def _read_slot(slot_group: h5py.Group, *, file_name: str) -> SavedFitSlot:
    """Decode one slot group into a ``SavedFitSlot``."""

    meta = require_group(slot_group["metadata"], "metadata")
    a = meta.attrs
    fit_type = cast(FitType, _attr_str(a["fit_type"]))
    handle = _attr_str(a["handle"])
    input_files = _attr_str(a["input_files"])
    selection_json = _selection_json_for(
        input_files, file_name=file_name, handle=handle
    )

    params = _decode_dataframe(require_dataset(slot_group["params"], "params"))
    if fit_type != "sbs":
        # Restore the schema's "" ↔ None / NaN ↔ None mappings for long-form
        # params. sbs params is wide-form numeric and carries no None
        # semantics, so this only applies to baseline / spectrum / 2d.
        _restore_long_params_nones(params)

    metrics: dict[str, Any]
    if fit_type == "sbs":
        metrics = _read_metrics_per_slice(
            require_dataset(slot_group["metrics_per_slice"], "metrics_per_slice")
        )
    else:
        # Attrs omitted on disk (the DoF metrics of a joint projection)
        # rehydrate as NaN.
        metrics = {
            k: (float(np.asarray(a[k]).item()) if k in a else float("nan"))
            for k in _METRICS_KEYS
        }

    conf_ci, correl, mcmc = _read_result_attachments(slot_group)
    params_meta_obj = slot_group.get("params_meta")
    params_meta: pd.DataFrame | None = None
    if params_meta_obj is not None:
        params_meta = _decode_dataframe(require_dataset(params_meta_obj, "params_meta"))
        # Same "" ↔ None mapping as long-form params (expr column).
        _restore_long_params_nones(params_meta)
    params_stderr_obj = slot_group.get("params_stderr")
    params_stderr = (
        _decode_dataframe(require_dataset(params_stderr_obj, "params_stderr"))
        if params_stderr_obj is not None
        else None
    )
    params_init_obj = slot_group.get("params_init")
    params_init = (
        _decode_dataframe(require_dataset(params_init_obj, "params_init"))
        if params_init_obj is not None
        else None
    )
    fit_settings = (
        json.loads(_attr_str(a["fit_settings"])) if "fit_settings" in a else None
    )
    components_obj = slot_group.get("components")
    components: np.ndarray | None = None
    component_names: list[str] | None = None
    if components_obj is not None:
        components = _read_array(require_dataset(components_obj, "components"))
        names_obj = require_dataset(slot_group["component_names"], "component_names")
        component_names = [_to_str_value(v) for v in names_obj[...]]
    fit_ini_obj = slot_group.get("fit_ini")
    fit_ini = (
        _read_array(require_dataset(fit_ini_obj, "fit_ini"))
        if fit_ini_obj is not None
        else None
    )
    dark_obj = slot_group.get("dark")
    dark = (
        _read_array(require_dataset(dark_obj, "dark")) if dark_obj is not None else None
    )
    calibration_obj = slot_group.get("calibration")
    calibration = (
        _read_array(require_dataset(calibration_obj, "calibration"))
        if calibration_obj is not None
        else None
    )
    model_yaml_obj = slot_group.get("model_yaml")
    model_yaml = (
        _read_model_yaml(require_group(model_yaml_obj, "model_yaml"))
        if model_yaml_obj is not None
        else None
    )

    return SavedFitSlot(
        handle=handle,
        optimization_hash=_attr_str(a["optimization_hash"]),
        input_files=input_files,
        model_structure=_attr_str(a["model_structure"]),
        fit_view_sha256=_attr_str(a["fit_view_sha256"]),
        file_name=file_name,
        model_name=_attr_str(a["model_name"]),
        fit_type=fit_type,
        selection=json.loads(selection_json),
        selection_json=selection_json,
        params=params,
        metrics=metrics,
        observed=_read_array(require_dataset(slot_group["observed"], "observed")),
        fit=_read_array(require_dataset(slot_group["fit"], "fit")),
        fit_alg=_attr_str(a["fit_alg"]),
        timestamp=_attr_str(a["timestamp"]),
        noise_type=_attr_str(a["noise_type"]),
        sigma_source=_attr_str(a["sigma_source"]),
        sigma_type=_attr_str(a["sigma_type"]),
        sigma_data=float(np.asarray(a["sigma_data"]).item()),
        sigma_eff=float(np.asarray(a["sigma_eff"]).item()),
        dark=dark,
        calibration=calibration,
        model_yaml=model_yaml,
        label=_attr_str(a["label"]) if "label" in a else None,
        joint_ref=_attr_str(a["joint_ref"]) if "joint_ref" in a else None,
        conf_ci=conf_ci,
        correl=correl,
        mcmc=mcmc,
        params_meta=params_meta,
        params_stderr=params_stderr,
        fit_settings=fit_settings,
        components=components,
        component_names=component_names,
        fit_ini=fit_ini,
        params_init=params_init,
    )


#
def _read_file(file_group: h5py.Group) -> SavedFile:
    """Decode one file group into a ``SavedFile``."""

    meta = require_group(file_group["metadata"], "metadata")
    a = meta.attrs
    name = _attr_str(a["name"])

    aux_axis_obj = file_group.get("aux_axis")
    aux_axis = (
        _read_array(require_dataset(aux_axis_obj, "aux_axis"))
        if aux_axis_obj is not None
        else None
    )

    slot_records: list[SavedFitSlot] = []
    slots_obj = file_group.get("slots")
    if slots_obj is not None:
        slots_group = require_group(slots_obj, "slots")
        for key in sorted(slots_group.keys()):
            sg = require_group(slots_group[key], f"slots/{key}")
            slot_records.append(_read_slot(sg, file_name=name))

    return SavedFile(
        name=name,
        original_path=_attr_str(a["original_path"]),
        dim=int(np.asarray(a["dim"]).item()),
        shape=tuple(int(x) for x in np.asarray(a["shape"]).ravel()),
        file_content_hash=_attr_str(a["file_content_hash"]),
        data_raw=_read_array(require_dataset(file_group["data_raw"], "data_raw")),
        energy=_read_array(require_dataset(file_group["energy"], "energy")),
        time=_read_array(require_dataset(file_group["time"], "time")),
        slots=tuple(slot_records),
        aux_axis=aux_axis,
    )


#
def _read_joint_records(
    project_group: h5py.Group,
    files: Sequence[SavedFile],
) -> tuple[JointFitResult, ...]:
    """
    Decode ``project/joint/`` into ``JointFitResult`` records.

    Projection slots resolve by ``handle`` to the slot objects already
    decoded under ``files`` — content addressing, never a positional
    path. Unresolved handles, ``joint_ref`` mismatches, and non-total
    parameter maps raise ``ValueError``, mirroring the writer's
    bundle-integrity checks. Metric attrs omitted on disk (``r2``)
    rehydrate as ``NaN``.
    """

    joint_obj = project_group.get("joint")
    joint_groups: list[tuple[str, h5py.Group]] = []
    if joint_obj is not None:
        joint_root = require_group(joint_obj, "joint")
        joint_groups = [
            (key, require_group(joint_root[key], f"joint/{key}"))
            for key in sorted(joint_root.keys())
        ]
    slots_by_handle = {slot.handle: slot for sf in files for slot in sf.slots}

    records: list[JointFitResult] = []
    for key, jg in joint_groups:
        meta = require_group(jg["metadata"], f"joint/{key}/metadata")
        a = meta.attrs
        optimization_hash = _attr_str(a["optimization_hash"])
        context = f"Joint record {optimization_hash[:8]}"

        params = _decode_dataframe(require_dataset(jg["params"], "params"))
        _restore_long_params_nones(params)
        metrics = {
            k: (float(np.asarray(a[k]).item()) if k in a else float("nan"))
            for k in _METRICS_KEYS
        }
        conf_ci, correl, mcmc_payload = _read_result_attachments(jg)
        mcmc = (
            mcmc_result_from_payload(mcmc_payload) if mcmc_payload is not None else None
        )

        projections: list[JointFitProjection] = []
        for rec in json.loads(_attr_str(a["projections"])):
            handle = str(rec["handle"])
            slot = slots_by_handle.get(handle)
            if slot is None:
                raise ValueError(
                    f"{context} declares projection slot {handle[:8]} "
                    f"(file {rec['file_name']!r}), which is not stored in "
                    f"the archive — corrupt bundle."
                )
            if slot.joint_ref != optimization_hash:
                raise ValueError(
                    f"{context}: projection slot {handle[:8]} (file "
                    f"{slot.file_name!r}) carries joint_ref "
                    f"{'absent' if slot.joint_ref is None else slot.joint_ref[:8]} "
                    f"instead of its record's optimization_hash."
                )
            projections.append(
                JointFitProjection(parameter_map=dict(rec["parameter_map"]), slot=slot)
            )
        _assert_parameter_maps_consistent(
            context=context,
            combined_params=params,
            projections=[
                (p.slot.file_name, p.parameter_map, p.slot.params) for p in projections
            ],
        )

        records.append(
            JointFitResult(
                model_name=_attr_str(a["model_name"]),
                optimization_hash=optimization_hash,
                input_files=_attr_str(a["input_files"]),
                model_structure=_attr_str(a["model_structure"]),
                projections=tuple(projections),
                params=params,
                metrics=metrics,
                fit_alg=_attr_str(a["fit_alg"]),
                fit_settings=json.loads(_attr_str(a["fit_settings"])),
                timestamp=_attr_str(a["timestamp"]),
                conf_ci=conf_ci,
                correl=correl,
                mcmc=mcmc,
                label=_attr_str(a["label"]) if "label" in a else None,
            )
        )

    # Mirror the writer's slot→joint half of the bundle invariant: a
    # project-scoped slot must reference a stored joint record (and only
    # project-scoped slots carry one). The joint→slot half is enforced
    # per record above; without this pass, deleting joint/ from an
    # archive would leave its projections loading silently.
    known_joint = {record.optimization_hash for record in records}
    for sf in files:
        for slot in sf.slots:
            scope = _input_files_scope(slot.input_files)
            if (scope == "project") != (slot.joint_ref is not None):
                joint_ref_state = "set" if slot.joint_ref is not None else "absent"
                raise ValueError(
                    f"Slot {slot.handle[:8]} on file {slot.file_name!r} "
                    f"violates the joint-reference invariant: input_files "
                    f"scope is {scope!r} but joint_ref is {joint_ref_state} "
                    f"— corrupt bundle."
                )
            if slot.joint_ref is not None and slot.joint_ref not in known_joint:
                raise ValueError(
                    f"Slot {slot.handle[:8]} on file {slot.file_name!r} "
                    f"references joint record {slot.joint_ref[:8]}, which "
                    f"is not stored in the archive — corrupt bundle."
                )
    return tuple(records)


#
def read_archive(filepath: PathLike | str) -> SavedProject:
    """
    Deserialize an HDF5 fit archive into a ``SavedProject``.

    Inverse of ``write_archive``. Does not touch any live ``Project``,
    ``File``, or ``Model`` state — the returned ``SavedProject`` is a
    standalone, immutable view of the archive's contents at read time
    (arrays come back read-only). Joint records rehydrate with their
    projection slots resolved by ``handle`` — the same objects as in
    ``files[*].slots`` — and are validated like the writer validates on
    save. Metric attrs omitted on disk (a joint projection's DoF metrics,
    the joint record's ``r2``) come back as ``NaN``.

    Raises ``ValueError`` if ``schema_version`` is not in
    ``SUPPORTED_READ_VERSIONS`` — pre-7 archives are not readable
    (re-fit and re-save under schema 7).
    """

    path = Path(filepath)
    with h5py.File(path, "r") as archive:
        meta = require_group(archive["metadata"], "metadata")
        ma = meta.attrs
        schema_version = _attr_str(ma["schema_version"])
        if schema_version not in SUPPORTED_READ_VERSIONS:
            supported = ", ".join(repr(v) for v in SUPPORTED_READ_VERSIONS)
            raise ValueError(
                f"Archive at {path} has schema_version {schema_version!r}; "
                f"this reader supports {supported}."
            )
        project_group = require_group(archive["project"], "project")
        pa = project_group.attrs

        files: list[SavedFile] = []
        files_obj = project_group.get("files")
        if files_obj is not None:
            files_group = require_group(files_obj, "files")
            for key in sorted(files_group.keys()):
                fg = require_group(files_group[key], f"files/{key}")
                files.append(_read_file(fg))

        return SavedProject(
            name=_attr_str(pa["name"]),
            trspecfit_version=_attr_str(ma["trspecfit_version"]),
            schema_version=schema_version,
            timestamp_created=_attr_str(ma["timestamp_created"]),
            timestamp_updated=_attr_str(ma["timestamp_updated"]),
            plot_config=PlotConfig.from_json(_attr_str(pa["plot_config"])),
            files=tuple(files),
            joint=_read_joint_records(project_group, files),
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
    ``__{handle[:8]}`` suffix so each lands in a distinct directory.
    """

    base = f"{slot.model_name}__{slot.fit_type}"
    if suffix_with_hash:
        base = f"{base}__{slot.handle[:8]}"
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
       sufficient — two records with identical ``file_content_hash`` *and*
       ``original_path`` would still collide.
    2. **Within a file:** ``(model_name, fit_type)`` collisions get the
       slot's ``handle[:8]`` suffix on the slot directory.

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
    plot_config: PlotConfig,
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
    plot_config: PlotConfig,
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
    plot_config: PlotConfig,
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
def write_csv_export(
    root: PathLike | str,
    *,
    project: SavedProject,
    num_fmt: str = "%.6e",
    delim: str = ",",
    plot_config: PlotConfig | None = None,
    overwrite: bool = False,
) -> int:
    """
    Serialize a ``SavedProject`` to a CSV/PNG export tree.

    Layout: ``<root>/<file_name>/<model_name>__<fit_type>[__<hash>]/``.
    The ``__<hash>`` suffix appears only when more than one slot shares
    the ``(file, model, fit_type)`` triple (i.e. multiple selections in
    the snapshot); the suffix is the first 8 chars of ``handle``.
    When two ``SavedFile`` records share a ``name``, every entry in the
    colliding group gets a positional ordinal suffix (``__000``,
    ``__001``, ...) so byte-identical records (same content hash *and*
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
    plot_config : PlotConfig | None
        Drives PNG styling; one config for every file (presentation is
        project-owned). ``Project.export_fits`` passes the project's
        config; ``None`` falls back to default ``PlotConfig()``.
    overwrite : bool, default False
        Per-slot directory: a non-empty target dir raises
        ``FileExistsError`` unless True. Pre-checked across all slots
        before any writes.

    Returns
    -------
    int
        Number of slot directories written.
    """

    if plot_config is not None and not isinstance(plot_config, PlotConfig):
        # Fail before any filesystem mutation (mkdir / overwrite clearing) —
        # a partial export after a destructive clear is the worst outcome.
        raise TypeError(
            f"plot_config must be a PlotConfig or None, got "
            f"{type(plot_config).__name__}. Per-file config dicts were "
            f"removed in v0.14.0 — presentation is project-owned (one "
            f"config for every file)."
        )

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    slot_dirs = _resolve_export_dirs(project.files, root_path)
    _precheck_export_collisions(slot_dirs, overwrite=overwrite)

    written_paths: set[Path] = set()
    n_written = 0
    for sf in project.files:
        sf_plot_config = plot_config if plot_config is not None else PlotConfig()
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
