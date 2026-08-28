"""
Round-trip tests for the fit archive: save → load → field equality.

For each (model family, fit type), run the fit, write the archive via
``Project.save_fits``, read it back via ``FitResults.load`` (and via
``Project.load_fits``), and verify every user-visible slot field is
reconstructed exactly. Also exercises the design invariant that
``observed - fit`` reproduces residuals for any fit_type without reading
``file.data``.

Coverage matrix
---------------
- F1  (basic):              baseline, spectrum, sbs
- F3  (basic + dynamics):   2d
- F6  (profile-only):       baseline, spectrum, sbs
- F8  (profile + dynamics): baseline, 2d

The matrix covers basic / profile / profile+dynamics × applicable fit
types — and aligns with
``tests/roundtrip/matrix.py`` for B/Sp/SbS on F1/F6 (the families that
support 1D fits). F6 has no top-level dynamics so its 2d slot would
have nothing extra to assert vs F3; F8 covers the full "profile + 2d
dynamics" payload.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from typing import Any

import numpy as np
import pandas as pd
import pytest
from _utils import make_project, simulate_noisy
from roundtrip.families import FAMILIES

from trspecfit import File, FitResults
from trspecfit.utils.fit_io import SavedFitSlot


#
def _build_fit_file(family_id: str, *, spec_fun_str: str = "fit_model_gir"):
    """Build (truth_file, fit_file, family) for a family, with noisy data.

    Mirrors the setup pattern used in roundtrip/test_focused.py: truth
    file + simulator → noisy data → empty fit file with the same model.
    Noise is small (0.01) so fits converge but chi2/AIC/BIC stay finite
    (clean data → chi2=0 → log(0)=nan in AIC).
    """

    family = FAMILIES[family_id]
    truth_project = make_project(name="rt_truth", spec_fun_str=spec_fun_str)
    truth_file = family.build_truth(truth_project, variant="default")
    data = simulate_noisy(truth_file.model_active, noise_level=0.01)

    fit_project = make_project(name="rt_fit", spec_fun_str=spec_fun_str)
    fit_kwargs: dict[str, Any] = {
        "data": data,
        "energy": truth_file.energy,
        "time": truth_file.time,
        "variant": "default",
    }
    if family.needs_aux:
        fit_kwargs["aux"] = truth_file.aux_axis
    fit_file = family.build_fit(fit_project, **fit_kwargs)
    return truth_file, fit_file, family


#
def _save_load_one(project, archive_path) -> tuple[SavedFitSlot, FitResults]:
    """Save → load and return (loaded slot, FitResults) for the single in-memory slot.

    Asserts that exactly one slot survives the round-trip — keeps the
    individual tests focused on field equality rather than count book-
    keeping.
    """

    project.save_fits(archive_path, show_output=0)
    loaded = FitResults.load(archive_path)
    assert len(loaded) == 1, f"expected 1 loaded slot, got {len(loaded)}"
    return next(iter(loaded)), loaded


#
def _assert_slot_round_tripped(loaded: SavedFitSlot, original: SavedFitSlot) -> None:
    """Assert every persisted SavedFitSlot field round-trips exactly.

    Covers the identity chain (hashes, version stamps, selection), arrays,
    metrics, params, and provenance. Also verifies the design invariant that
    ``observed - fit`` reproduces residuals on the loaded slot alone (no
    ``file.data`` lookup) — both via direct subtraction and against the
    stored chi2.
    """

    # --- identity ------------------------------------------------------
    assert loaded.file_name == original.file_name
    assert loaded.model_name == original.model_name
    assert loaded.fit_type == original.fit_type
    assert loaded.selection_json == original.selection_json
    assert loaded.selection == original.selection
    assert loaded.handle == original.handle
    assert loaded.optimization_hash == original.optimization_hash
    assert loaded.input_files == original.input_files
    assert loaded.model_structure == original.model_structure
    assert loaded.fit_view_sha256 == original.fit_view_sha256
    assert loaded.joint_ref == original.joint_ref

    # --- arrays --------------------------------------------------------
    np.testing.assert_array_equal(loaded.observed, original.observed)
    np.testing.assert_array_equal(loaded.fit, original.fit)
    assert loaded.observed.shape == loaded.fit.shape
    assert loaded.observed.dtype == original.observed.dtype
    assert loaded.fit.dtype == original.fit.dtype

    # --- metrics -------------------------------------------------------
    assert set(loaded.metrics.keys()) == set(original.metrics.keys())
    metric_keys = ("chi2_raw", "chi2_red_raw", "chi2", "chi2_red", "r2", "aic", "bic")
    if original.fit_type == "sbs":
        for k in metric_keys:
            # equal_nan=True so NaN-valued calibrated metrics (when no σ was
            # set on the file at fit time) round-trip as exact NaN matches.
            np.testing.assert_allclose(
                loaded.metrics[k], original.metrics[k], rtol=0, atol=0, equal_nan=True
            )
            # Per-slice metric arrays are frozen like every read array.
            assert not loaded.metrics[k].flags.writeable
    else:
        for k in metric_keys:
            orig_v = original.metrics[k]
            loaded_v = loaded.metrics[k]
            if isinstance(orig_v, float) and np.isnan(orig_v):
                assert np.isnan(loaded_v)
            else:
                assert loaded_v == pytest.approx(orig_v, rel=0, abs=0)

    # --- noise metadata -----------------------------------------------
    assert loaded.noise_type == original.noise_type
    assert loaded.sigma_source == original.sigma_source
    assert loaded.sigma_type == original.sigma_type
    for name in ("sigma_data", "sigma_eff"):
        orig_v = getattr(original, name)
        loaded_v = getattr(loaded, name)
        if np.isnan(orig_v):
            assert np.isnan(loaded_v), f"{name} NaN round-trip failed"
        else:
            assert loaded_v == pytest.approx(orig_v, rel=0, abs=0)

    # --- params --------------------------------------------------------
    _assert_params_equal(loaded.params, original.params, fit_type=original.fit_type)

    # --- uncertainty payloads (None ↔ None or exact) --------------------
    _assert_optional_df_equal(loaded.conf_ci, original.conf_ci, label="conf_ci")
    _assert_optional_df_equal(loaded.correl, original.correl, label="correl")
    _assert_optional_df_equal(
        loaded.params_meta, original.params_meta, label="params_meta"
    )
    _assert_optional_df_equal(
        loaded.params_stderr, original.params_stderr, label="params_stderr"
    )
    assert loaded.fit_settings == original.fit_settings
    if original.correl is not None:
        assert loaded.correl is not None  # type guard
        # The square matrix persists only column labels; the reader must
        # restore index == columns.
        assert list(loaded.correl.index) == list(loaded.correl.columns)
    if original.mcmc is None:
        assert loaded.mcmc is None
    else:
        assert loaded.mcmc is not None  # type guard
        assert set(loaded.mcmc.keys()) == set(original.mcmc.keys())
        _assert_optional_df_equal(
            loaded.mcmc["flatchain"], original.mcmc["flatchain"], label="flatchain"
        )
        _assert_optional_df_equal(loaded.mcmc["ci"], original.mcmc["ci"], label="ci")
        assert loaded.mcmc["lnsigma"] == original.mcmc["lnsigma"]
        orig_acc = original.mcmc["acceptance_fraction"]
        if orig_acc is None:
            assert loaded.mcmc["acceptance_fraction"] is None
        else:
            np.testing.assert_array_equal(loaded.mcmc["acceptance_fraction"], orig_acc)

    # --- provenance ----------------------------------------------------
    assert loaded.fit_alg == original.fit_alg
    assert loaded.model_yaml == original.model_yaml
    assert loaded.label == original.label
    assert loaded.timestamp == original.timestamp

    # --- correction snapshots -------------------------------------------
    for corr in ("dark", "calibration"):
        orig_corr = getattr(original, corr)
        loaded_corr = getattr(loaded, corr)
        if orig_corr is None:
            assert loaded_corr is None
        else:
            np.testing.assert_array_equal(loaded_corr, orig_corr)

    # --- components (None for 2d) ----------------------------------------
    if original.fit_type == "2d":
        assert original.components is None
        assert loaded.components is None
        assert original.component_names is None
        assert loaded.component_names is None
    else:
        assert original.components is not None
        assert loaded.components is not None
        np.testing.assert_array_equal(loaded.components, original.components)
        assert loaded.component_names == original.component_names
        if original.fit_type == "sbs":
            assert loaded.components.shape[0] == loaded.observed.shape[0]  # n_slices
            assert loaded.components.shape[2] == loaded.observed.shape[1]  # n_energy
            assert loaded.components.shape[1] == len(loaded.component_names)
            recon = np.sum(loaded.components, axis=1)
        else:
            assert loaded.components.shape[1] == loaded.observed.shape[0]
            assert loaded.components.shape[0] == len(loaded.component_names)
            recon = np.sum(loaded.components, axis=0)
        # Components must sum back to the persisted fit curve.
        np.testing.assert_allclose(recon, loaded.fit, rtol=1e-8, atol=1e-8)

    # --- fit_ini / params_init -------------------------------------------
    # None only on the project-level joint-fit path; every family/fit_type
    # exercised here goes through a per-File fit method, so fit_ini is
    # always populated.
    assert original.fit_ini is not None
    assert loaded.fit_ini is not None
    np.testing.assert_array_equal(loaded.fit_ini, original.fit_ini)
    assert loaded.fit_ini.shape == loaded.fit.shape
    if original.fit_type == "sbs":
        assert original.params_init is not None
        assert loaded.params_init is not None
        _assert_optional_df_equal(
            loaded.params_init, original.params_init, label="params_init"
        )
    else:
        assert original.params_init is None
        assert loaded.params_init is None

    # --- residual reconstruction (design invariant) --------------------
    # chi2_raw is the lmfit-unweighted SSE diagnostic; chi2 is σ-calibrated
    # and NaN when no σ was set on the file, so we cross-check against the raw
    # column (always populated and grid-derived).
    residual = loaded.observed - loaded.fit
    assert residual.shape == loaded.observed.shape
    if loaded.fit_type == "sbs":
        assert residual.ndim == 2
        for i in range(residual.shape[0]):
            assert loaded.metrics["chi2_raw"][i] == pytest.approx(
                float(np.sum(residual[i] ** 2))
            )
    else:
        assert loaded.metrics["chi2_raw"] == pytest.approx(float(np.sum(residual**2)))


#
def _assert_optional_df_equal(
    loaded: pd.DataFrame | None, original: pd.DataFrame | None, *, label: str
) -> None:
    """None ↔ None, or column labels + cell values equal (str cells exact,
    float cells exact-or-NaN-matched)."""

    if original is None:
        assert loaded is None, f"{label}: orig=None, loaded is not None"
        return
    assert loaded is not None, f"{label}: orig is a DataFrame, loaded=None"
    assert list(loaded.columns) == list(original.columns), label
    assert len(loaded) == len(original), label
    for col in original.columns:
        for o, ll in zip(original[col].to_list(), loaded[col].to_list(), strict=True):
            if isinstance(o, float) and np.isnan(o):
                assert isinstance(ll, float) and np.isnan(ll), f"{label}.{col}"
            elif isinstance(o, float):
                assert ll == pytest.approx(o, rel=0, abs=0), f"{label}.{col}"
            else:
                assert ll == o, f"{label}.{col}"


#
def _assert_params_equal(
    loaded: pd.DataFrame, original: pd.DataFrame, *, fit_type: str
) -> None:
    """Compare params DataFrames column-wise.

    Handles two layouts:

    - **long-form** (baseline / spectrum / 2d): mixed-dtype columns
      including ``expr`` (str | None) and ``stderr`` (float | None).
      ``_restore_long_params_nones`` in the reader maps ``""`` → ``None``
      and ``NaN`` → ``None`` so the round-tripped frame matches the
      lmfit-original.
    - **wide-form** (sbs): all-float columns, one row per slice.

    Compared column-by-column rather than via ``assert_frame_equal``
    because the writer round-trips through structured arrays — minor
    dtype quirks (object vs string) on the ``expr`` column would
    otherwise fail an exact frame-equality check despite values matching.
    """

    assert list(loaded.columns) == list(original.columns)
    assert len(loaded) == len(original)
    for col in original.columns:
        orig_vals = original[col].to_list()
        load_vals = loaded[col].to_list()
        assert len(orig_vals) == len(load_vals)
        for o, ll in zip(orig_vals, load_vals, strict=True):
            if isinstance(o, float) and np.isnan(o):
                # Both should be NaN (or None ↔ None handled below).
                assert isinstance(ll, float) and np.isnan(ll), (
                    f"col {col!r}: orig=NaN, loaded={ll!r}"
                )
            elif o is None:
                assert ll is None, f"col {col!r}: orig=None, loaded={ll!r}"
            elif isinstance(o, float):
                assert ll == pytest.approx(o, rel=0, abs=0), (
                    f"col {col!r}: orig={o!r}, loaded={ll!r}"
                )
            else:
                assert ll == o, f"col {col!r}: orig={o!r}, loaded={ll!r}"


# ---------------------------------------------------------------------------
# baseline round-trip
# ---------------------------------------------------------------------------


#
@pytest.mark.parametrize("family_id", ["F1", "F6", "F8"])
def test_baseline_roundtrip(family_id: str, tmp_path) -> None:
    """basic / profile / profile+dynamics × baseline."""

    _, fit_file, family = _build_fit_file(family_id)
    fit_file.fit_baseline(model_name=family.model_name("default"), stages=1, try_ci=0)
    project = fit_file.p
    archive_path = tmp_path / "baseline.fit.h5"

    loaded_slot, loaded_results = _save_load_one(project, archive_path)
    original = project._fit_history[0]
    assert original.fit_type == "baseline"
    _assert_slot_round_tripped(loaded_slot, original)

    # --- per-file aux_axis (None unless the family needs it) --------------
    provider = loaded_results._provider_for(loaded_slot)
    if family.needs_aux:
        assert fit_file.aux_axis is not None
        np.testing.assert_array_equal(provider.aux_axis, fit_file.aux_axis)
    else:
        assert provider.aux_axis is None


# ---------------------------------------------------------------------------
# spectrum round-trip
# ---------------------------------------------------------------------------


#
@pytest.mark.parametrize("family_id", ["F1", "F6"])
def test_spectrum_roundtrip(family_id: str, tmp_path) -> None:
    """basic / profile × spectrum: 1D fit at a single time point.

    F6 covers the profile path through ``fit_spectrum``: profiles
    propagate into the per-spectrum lmfit params (one ``pExpDecay`` /
    ``pLinear`` parameter set per profiled base parameter), and the
    serialized params DataFrame must round-trip without losing those
    rows or their min/max/expr metadata.
    """

    _, fit_file, family = _build_fit_file(family_id)
    fit_file.fit_spectrum(
        family.model_name("default"),
        time_point=10,
        time_type="ind",
        stages=1,
        try_ci=0,
        show_plot=False,
    )
    project = fit_file.p
    archive_path = tmp_path / "spectrum.fit.h5"

    loaded_slot, _ = _save_load_one(project, archive_path)
    original = project._fit_history[0]
    assert original.fit_type == "spectrum"
    assert loaded_slot.selection["time_point"] == 10
    assert loaded_slot.selection["time_type"] == "ind"
    _assert_slot_round_tripped(loaded_slot, original)


# ---------------------------------------------------------------------------
# slice-by-slice round-trip
# ---------------------------------------------------------------------------


#
@pytest.mark.slow
@pytest.mark.parametrize("family_id", ["F1", "F6"])
def test_sbs_roundtrip(family_id: str, tmp_path) -> None:
    """basic / profile × slice-by-slice (per-slice metrics, wide-form params)."""

    _, fit_file, family = _build_fit_file(family_id, spec_fun_str="fit_model_mcp")
    fit_file.fit_slice_by_slice(
        family.model_name("default"),
        stages=1,
        n_workers=1,
        seed_source="model",
        seed_adapt=None,
        try_ci=0,
    )
    project = fit_file.p
    archive_path = tmp_path / "sbs.fit.h5"

    loaded_slot, _ = _save_load_one(project, archive_path)
    original = project._fit_history[0]
    assert original.fit_type == "sbs"
    # Per-slice metrics are arrays sized to the time axis.
    assert loaded_slot.metrics["chi2"].shape == (len(fit_file.time),)
    _assert_slot_round_tripped(loaded_slot, original)

    # Cross-check params_init against each slice's true seed straight from
    # the live SbS results — joint validation that this feature and the
    # fitlib.fit_wrapper stage-2 init_value fix agree with each other.
    assert loaded_slot.params_init is not None  # type guard
    for i, result in enumerate(fit_file.results_sbs):
        assert result.par_ini is not None  # type guard
        for name in loaded_slot.params_init.columns:
            assert loaded_slot.params_init.iloc[i][name] == pytest.approx(
                result.par_ini[name].value
            )


# ---------------------------------------------------------------------------
# 2D round-trip
# ---------------------------------------------------------------------------


#
def _fit_2d_with_dynamics(family_id: str):
    """Run the baseline → add_dynamics → fit_2d pipeline for a 2D family.

    Mirrors the standard 2D workflow used in test_focused.py: fit the
    baseline first to seed amplitudes, attach dynamics on the fit-side
    file, then run the joint 2D fit.
    """

    _, fit_file, family = _build_fit_file(family_id)
    fit_file.fit_baseline(model_name=family.model_name("default"), stages=1, try_ci=0)
    assert family.add_dynamics is not None  # type guard
    family.add_dynamics(fit_file, "default")
    fit_file.fit_2d(model_name=family.model_name("default"), stages=1, try_ci=0)
    return fit_file, family


#
@pytest.mark.parametrize("family_id", ["F3", "F8"])
def test_2d_roundtrip(family_id: str, tmp_path) -> None:
    """basic+dynamics / profile+dynamics × 2d.

    The 2D slot lives alongside the baseline slot in ``_fit_history``;
    this test saves only the 2d slot via the ``fit_type`` filter so the
    round-trip is unambiguous.
    """

    fit_file, _ = _fit_2d_with_dynamics(family_id)
    project = fit_file.p
    # _fit_history holds [baseline, 2d]; filter to just the 2d slot.
    archive_path = tmp_path / "2d.fit.h5"
    project.save_fits(archive_path, fit_type="2d", show_output=0)
    loaded = FitResults.load(archive_path)
    assert len(loaded) == 1
    loaded_slot = next(iter(loaded))

    original = next(s for s in project._fit_history if s.fit_type == "2d")
    assert loaded_slot.observed.ndim == 2
    _assert_slot_round_tripped(loaded_slot, original)


# ---------------------------------------------------------------------------
# load entry-point parity
# ---------------------------------------------------------------------------


#
def test_project_load_fits_matches_fitresults_load(tmp_path) -> None:
    """``Project.load_fits`` is documented as a thin delegate to ``FitResults.load``.

    Verify both entry points return field-equal slot lists for the same
    archive — guards against drift if either path adds incidental
    transformations later.
    """

    _, fit_file, family = _build_fit_file("F1")
    fit_file.fit_baseline(model_name=family.model_name("default"), stages=1, try_ci=0)
    project = fit_file.p
    archive_path = tmp_path / "parity.fit.h5"
    project.save_fits(archive_path, show_output=0)

    via_class = list(FitResults.load(archive_path))
    via_project = list(project.load_fits(archive_path, show_output=0))
    assert len(via_class) == len(via_project) == 1
    _assert_slot_round_tripped(via_project[0], via_class[0])


# ---------------------------------------------------------------------------
# multi-file + multi-fit-type archive round-trip
# ---------------------------------------------------------------------------


#
@pytest.mark.slow
def test_multi_slot_roundtrip(tmp_path) -> None:
    """Archive with multiple slots from one file (baseline + spectrum + sbs).

    Exercises the writer's per-file slot-list handling and the reader's
    flatten-into-FitResults order. All three slots must be recoverable
    field-by-field, not just by count.
    """

    _, fit_file, family = _build_fit_file("F1", spec_fun_str="fit_model_mcp")
    fit_file.fit_baseline(model_name=family.model_name("default"), stages=1, try_ci=0)
    fit_file.fit_spectrum(
        family.model_name("default"),
        time_point=10,
        time_type="ind",
        stages=1,
        try_ci=0,
        show_plot=False,
    )
    fit_file.fit_slice_by_slice(
        family.model_name("default"),
        stages=1,
        n_workers=1,
        seed_source="model",
        seed_adapt=None,
        try_ci=0,
    )
    project = fit_file.p
    assert len(project._fit_history) == 3

    archive_path = tmp_path / "multi.fit.h5"
    project.save_fits(archive_path, show_output=0)

    loaded = FitResults.load(archive_path)
    assert len(loaded) == 3

    # Match loaded slots to originals by handle (order-independent).
    by_key = {s.handle: s for s in loaded}
    for original in project._fit_history:
        assert original.handle in by_key
        _assert_slot_round_tripped(by_key[original.handle], original)


# ---------------------------------------------------------------------------
# correlation-matrix round-trip
# ---------------------------------------------------------------------------


#
def test_correl_roundtrip(tmp_path) -> None:
    """leastsq always produces covariance, so the slot must capture the
    correlation matrix and round-trip it with index == columns intact.

    (The parametrized round-trips above use Nelder, where covariance —
    and therefore ``correl`` — depends on numdifftools being installed;
    this test pins the deterministic path.)
    """

    _, fit_file, family = _build_fit_file("F1")
    fit_file.fit_baseline(
        model_name=family.model_name("default"),
        stages=1,
        fit_alg_1="leastsq",
        try_ci=0,
    )
    project = fit_file.p
    original = project._fit_history[0]
    assert original.correl is not None  # type guard
    n_vary = int(original.params["vary"].sum())
    assert original.correl.shape == (n_vary, n_vary)
    np.testing.assert_allclose(np.diag(original.correl.to_numpy()), 1.0)

    loaded_slot, _ = _save_load_one(project, tmp_path / "correl.fit.h5")
    _assert_slot_round_tripped(loaded_slot, original)


# ---------------------------------------------------------------------------
# joint-bundle round-trip
# ---------------------------------------------------------------------------


#
def _build_joint_project(*, noise_level: float = 0.05):
    """Two-file project ready for ``Project.fit_2d`` (shared tau, per-file A).

    Compact version of ``test_project_fit._make_shared_tau_project``:
    per-file baseline fit (result injection), then the shared 2D model
    with dynamics. Noisy data so leastsq produces a covariance.
    """

    project = make_project(name="joint_roundtrip")
    for i, (amplitude, seed) in enumerate([(20.0, 42), (14.0, 43)]):
        truth_project = make_project(name="truth")
        truth = File(parent_project=truth_project)
        truth.energy = np.linspace(83, 87, 30)
        truth.time = np.linspace(-2, 10, 24)
        truth.dim = 2
        truth.load_model(
            model_yaml="models/project_energy.yaml", model_info="project_glp"
        )
        truth.add_time_dependence(
            target_model="project_glp",
            target_parameter="GLP_01_x0",
            dynamics_yaml="models/project_time.yaml",
            dynamics_model=["MonoExpProject"],
        )
        truth.model_active.lmfit_pars["GLP_01_A"].value = amplitude
        data = simulate_noisy(truth.model_active, noise_level=noise_level, seed=seed)
        file = File(
            parent_project=project,
            name=f"file_{i}",
            data=data,
            energy=truth.energy.copy(),
            time=truth.time.copy(),
        )
        file.load_model(
            model_yaml="models/project_energy.yaml", model_info="project_glp_base"
        )
        file.define_baseline(
            time_start=0, time_stop=3, time_type="ind", show_plot=False
        )
        file.fit_baseline(model_name="project_glp_base", stages=1, try_ci=0)
        file.load_model(
            model_yaml="models/project_energy.yaml", model_info="project_glp"
        )
        file.add_time_dependence(
            target_model="project_glp",
            target_parameter="GLP_01_x0",
            dynamics_yaml="models/project_time.yaml",
            dynamics_model=["MonoExpProject"],
        )
    return project


#
@pytest.mark.slow
def test_joint_bundle_roundtrip(tmp_path) -> None:
    """A project-level joint fit round-trips as one bundle.

    One ``joint/`` record plus one projection per file: identity fields,
    combined parameter table, whole-objective metrics, joint correlation,
    parameter maps, and the projection slots resolving to the same
    objects stored under ``files``. Also exercises save-filter bundle
    expansion: saving ``file="file_0"`` must pull in the sibling
    projection and the joint record (a joint bundle is saved whole),
    while file_1's baseline fit stays filtered out.
    """

    project = _build_joint_project()
    record = project.fit_2d(model_name="project_glp", stages=2, try_ci=0)
    assert record.correl is not None  # type guard

    archive_path = tmp_path / "joint.fit.h5"
    project.save_fits(archive_path, file="file_0", show_output=0)

    loaded = FitResults.load(archive_path)
    assert {(s.file_name, s.fit_type) for s in loaded} == {
        ("file_0", "baseline"),
        ("file_0", "2d"),
        ("file_1", "2d"),
    }
    (jrec,) = loaded.find_joint()
    assert loaded.get_joint(model="project_glp") is jrec
    assert jrec.optimization_hash == record.optimization_hash
    assert jrec.input_files == record.input_files
    assert jrec.model_structure == record.model_structure
    assert jrec.fit_alg == record.fit_alg
    _assert_params_equal(jrec.params, record.params, fit_type="2d")
    _assert_optional_df_equal(jrec.correl, record.correl, label="joint.correl")
    # The six whole-objective metrics round-trip; r2 is omitted on disk
    # and rehydrates as NaN (it is NaN on the live record too).
    for key in ("chi2_raw", "chi2_red_raw", "chi2", "chi2_red", "aic", "bic"):
        live_v, loaded_v = record.metrics[key], jrec.metrics[key]
        assert (np.isnan(live_v) and np.isnan(loaded_v)) or loaded_v == live_v, key
    assert np.isnan(jrec.metrics["r2"])

    # Projections resolve by handle to the same slot objects under files;
    # each carries the bundle reference and its per-file payload.
    loaded_2d = {s.file_name: s for s in loaded if s.fit_type == "2d"}
    for live_proj, loaded_proj in zip(
        record.projections, jrec.projections, strict=True
    ):
        slot = loaded_proj.slot
        assert slot is loaded_2d[slot.file_name]
        assert slot.handle == live_proj.slot.handle
        assert slot.joint_ref == jrec.optimization_hash
        assert loaded_proj.parameter_map == dict(live_proj.parameter_map)
        # The map recovers local names from combined ones by lookup.
        assert set(loaded_proj.parameter_map.values()) == set(slot.params["name"])
        # Count-dependent metrics do not decompose by file: NaN on disk
        # and after rehydration alike.
        for key in ("chi2_red_raw", "chi2_red", "aic", "bic"):
            assert np.isnan(slot.metrics[key]), key
        assert np.isfinite(slot.metrics["chi2_raw"])


# ---------------------------------------------------------------------------
# schema-version compatibility
# ---------------------------------------------------------------------------


#
def test_reader_rejects_unknown_schema_version(tmp_path) -> None:
    """Versions outside SUPPORTED_READ_VERSIONS raise a clear ValueError."""

    import h5py

    from trspecfit.utils.hdf5 import require_group

    _, fit_file, family = _build_fit_file("F1")
    fit_file.fit_baseline(model_name=family.model_name("default"), stages=1, try_ci=0)
    archive_path = tmp_path / "v1.fit.h5"
    fit_file.p.save_fits(archive_path, show_output=0)
    with h5py.File(archive_path, "r+") as h5:
        require_group(h5["metadata"], "metadata").attrs["schema_version"] = "1"

    with pytest.raises(ValueError, match=r"schema_version '1'"):
        FitResults.load(archive_path)
