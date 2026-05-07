"""
``Project.export_fits`` parity vs the legacy ``_save_sbs_fit_legacy`` /
``_save_2d_fit_legacy`` paths.

PLAN ask: same column shapes as the old ``save_sbs_fit`` / ``save_2d_fit``
outputs. The new export tree is slot-driven and richer (observed_2d.csv,
metrics.csv, params.csv land alongside the legacy files), but for the
artifacts that *do* overlap — ``fit_pars.csv``, ``fit_2d.csv``,
``energy.csv``, ``time.csv`` — column / shape parity must hold so users
can keep their downstream CSV-consuming pipelines unchanged.

Strategy: redirect the project's auto-save directory into ``tmp_path``,
run the fit (which triggers the legacy auto-save via
``_save_sbs_fit_legacy`` / ``_save_2d_fit_legacy``), then call
``project.export_fits`` into a sibling directory. Read both trees and
diff column names + shapes.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import numpy as np
import pandas as pd
from _utils import make_project, simulate_noisy

from trspecfit import File

_MODEL_YAML = "models/file_energy.yaml"
_TIME_YAML = "models/file_time.yaml"
_ENERGY_AXIS = np.linspace(83, 87, 30)
_TIME_AXIS = np.linspace(-2, 10, 24)


#
def _truth_2d_data():
    """Simulate noisy 2D data from a single_glp + MonoExpPos truth model.

    Reused across the SbS and 2D parity tests so both compare against
    the same data.
    """

    truth_project = make_project(name="parity_truth")
    truth = File(
        parent_project=truth_project,
        name="truth",
        energy=_ENERGY_AXIS,
        time=_TIME_AXIS,
    )
    truth.dim = 2
    truth.load_model(model_yaml=_MODEL_YAML, model_info="single_glp")
    truth.add_time_dependence(
        target_model="single_glp",
        target_parameter="GLP_01_A",
        dynamics_yaml=_TIME_YAML,
        dynamics_model=["MonoExpPos"],
    )
    return simulate_noisy(truth.model_active, noise_level=0.01)


#
def _make_parity_fit_file(*, name: str, tmp_path: Path, spec_fun_str: str):
    """Build a fit-side project + file with auto-save redirected into ``tmp_path``.

    Setting ``path_results`` after construction reroutes the legacy
    auto-save (``create_model_path`` builds paths under
    ``project.path_results``) into the test-scoped ``tmp_path / "legacy"``
    tree, so the test has full control over both outputs and the source
    repo stays untouched.
    """

    data = _truth_2d_data()
    project = make_project(name=name, spec_fun_str=spec_fun_str)
    project.path_results = tmp_path / "legacy"
    file = File(
        parent_project=project,
        name="fit",
        data=data,
        energy=_ENERGY_AXIS.copy(),
        time=_TIME_AXIS.copy(),
    )
    file.load_model(model_yaml=_MODEL_YAML, model_info="single_glp")
    return project, file


# ---------------------------------------------------------------------------
# SbS parity
# ---------------------------------------------------------------------------


#
def test_sbs_export_parity(tmp_path):
    """SbS exports: fit_pars.csv / fit_2d.csv / energy.csv / time.csv match.

    Verifies the new ``Project.export_fits`` SbS output has the same
    column names and shapes as the legacy ``_save_sbs_fit_legacy`` files
    that the auto-export path inside ``fit_slice_by_slice`` writes today.
    """

    project, file = _make_parity_fit_file(
        name="parity_sbs", tmp_path=tmp_path, spec_fun_str="fit_model_mcp"
    )
    file.fit_slice_by_slice(
        "single_glp",
        n_workers=1,
        seed_source="model",
        seed_adapt=None,
        try_ci=0,
    )

    legacy_dir = project.path_results / file.name / "sbs" / "single_glp"
    new_root = tmp_path / "new"
    project.export_fits(new_root, show_output=0)
    new_dir = new_root / file.name / "single_glp__sbs"

    # --- fit_pars.csv: per-slice param values with [index, time, par...] cols.
    # Legacy ``df.to_csv()`` emits a redundant pandas auto-index column;
    # the new export writes ``index=False``, so the legacy file has one
    # extra unnamed leading column. Strip "Unnamed: 0" before comparing
    # so the parity check focuses on the meaningful columns.
    legacy_fp = pd.read_csv(legacy_dir / "fit_pars.csv")
    if legacy_fp.columns[0].startswith("Unnamed"):
        legacy_fp = legacy_fp.drop(columns=legacy_fp.columns[0])
    new_fp = pd.read_csv(new_dir / "fit_pars.csv")
    assert list(legacy_fp.columns) == list(new_fp.columns), (
        f"fit_pars.csv columns differ:\n  legacy={list(legacy_fp.columns)}\n"
        f"  new={list(new_fp.columns)}"
    )
    assert legacy_fp.shape == new_fp.shape
    # Per-slice param values must match — both go through
    # ``list_of_par_to_df`` on the same fit results.
    for col in legacy_fp.columns:
        np.testing.assert_allclose(
            legacy_fp[col].to_numpy(dtype=float),
            new_fp[col].to_numpy(dtype=float),
            rtol=0,
            atol=0,
        )

    # --- fit_2d.csv: stacked per-slice fit spectra (n_time × n_energy).
    # Both paths re-evaluate the model at each slice's final params via
    # ``residual_fun(..., res_type="fit")`` (legacy through
    # ``results_to_fit_2d``, new through ``_slot_from_sbs``'s captured
    # ``fit`` array), so values must match exactly. Asserting shape alone
    # would let a bug that wrote the right-sized wrong matrix slip through.
    legacy_2d = np.loadtxt(legacy_dir / "fit_2d.csv", delimiter=project.delim)
    new_2d = np.loadtxt(new_dir / "fit_2d.csv", delimiter=project.delim)
    assert legacy_2d.shape == new_2d.shape == (len(file.time), len(file.energy))
    np.testing.assert_allclose(legacy_2d, new_2d, rtol=0, atol=0)

    # --- axis sidecars
    legacy_e = np.loadtxt(legacy_dir / "energy.csv", delimiter=project.delim)
    new_e = np.loadtxt(new_dir / "energy.csv", delimiter=project.delim)
    assert legacy_e.shape == new_e.shape == (len(file.energy),)
    np.testing.assert_array_equal(legacy_e, new_e)

    legacy_t = np.loadtxt(legacy_dir / "time.csv", delimiter=project.delim)
    new_t = np.loadtxt(new_dir / "time.csv", delimiter=project.delim)
    assert legacy_t.shape == new_t.shape == (len(file.time),)
    np.testing.assert_array_equal(legacy_t, new_t)

    # --- the same parameter PNGs exist in both trees
    legacy_pngs = {p.name for p in legacy_dir.glob("*.png")}
    new_pngs = {p.name for p in new_dir.glob("*.png")}
    # Per-parameter PNGs are emitted only for varied parameters; both
    # trees go through the same plt_fit_res_pars helper, so the per-
    # parameter set must match. The legacy path also emits an extra
    # "*_par_fin*" / fit-quality figure in some pipelines, so check
    # subset rather than equality.
    per_param_legacy = {n for n in legacy_pngs if "GLP_01_" in n}
    per_param_new = {n for n in new_pngs if "GLP_01_" in n}
    assert per_param_legacy == per_param_new


# ---------------------------------------------------------------------------
# 2D parity
# ---------------------------------------------------------------------------


#
def test_2d_export_parity(tmp_path):
    """2D exports: fit_2d.csv / energy.csv / time.csv shapes match.

    The legacy ``_save_2d_fit_legacy`` writes fit_2d / energy / time CSVs
    plus a residual-map PNG. The new export adds ``observed_2d.csv``,
    ``params.csv``, ``metrics.csv`` (no legacy counterparts), but the
    shared CSVs must keep identical shapes / values.
    """

    project, file = _make_parity_fit_file(
        name="parity_2d", tmp_path=tmp_path, spec_fun_str="fit_model_gir"
    )
    file.define_baseline(time_start=0, time_stop=3, time_type="ind", show_plot=False)
    file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
    file.add_time_dependence(
        target_model="single_glp",
        target_parameter="GLP_01_A",
        dynamics_yaml=_TIME_YAML,
        dynamics_model=["MonoExpPos"],
    )
    file.fit_2d("single_glp", stages=1, try_ci=0)

    legacy_dir = project.path_results / file.name / "2d" / "single_glp"
    new_root = tmp_path / "new"
    project.export_fits(new_root, fit_type="2d", show_output=0)
    new_dir = new_root / file.name / "single_glp__2d"

    # --- fit_2d.csv
    legacy_2d = np.loadtxt(legacy_dir / "fit_2d.csv", delimiter=project.delim)
    new_2d = np.loadtxt(new_dir / "fit_2d.csv", delimiter=project.delim)
    assert legacy_2d.shape == new_2d.shape == (len(file.time), len(file.energy))
    # Both go through ``residual_fun(..., res_type="fit")`` on the same
    # final params; values should match to numerical precision.
    np.testing.assert_allclose(legacy_2d, new_2d, rtol=0, atol=0)

    # --- axis sidecars (identical writers, identical inputs)
    legacy_e = np.loadtxt(legacy_dir / "energy.csv", delimiter=project.delim)
    new_e = np.loadtxt(new_dir / "energy.csv", delimiter=project.delim)
    np.testing.assert_array_equal(legacy_e, new_e)

    legacy_t = np.loadtxt(legacy_dir / "time.csv", delimiter=project.delim)
    new_t = np.loadtxt(new_dir / "time.csv", delimiter=project.delim)
    np.testing.assert_array_equal(legacy_t, new_t)

    # --- residual-map PNG present in both
    assert (legacy_dir / "2D_data_fit_res.png").exists()
    assert (new_dir / "2D_data_fit_res.png").exists()


#
def test_2d_export_includes_new_artifacts(tmp_path):
    """Sanity-check the additive payload the new export emits over legacy.

    Documents the new artifacts (``observed_2d.csv``, ``params.csv``,
    ``metrics.csv``) so a future change that drops one fails loudly.
    """

    project, file = _make_parity_fit_file(
        name="new_only", tmp_path=tmp_path, spec_fun_str="fit_model_gir"
    )
    file.define_baseline(time_start=0, time_stop=3, time_type="ind", show_plot=False)
    file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
    file.add_time_dependence(
        target_model="single_glp",
        target_parameter="GLP_01_A",
        dynamics_yaml=_TIME_YAML,
        dynamics_model=["MonoExpPos"],
    )
    file.fit_2d("single_glp", stages=1, try_ci=0)

    new_root = tmp_path / "new"
    project.export_fits(new_root, fit_type="2d", show_output=0)
    new_dir = new_root / file.name / "single_glp__2d"

    assert (new_dir / "observed_2d.csv").exists()
    assert (new_dir / "params.csv").exists()
    assert (new_dir / "metrics.csv").exists()

    # observed_2d should have the same shape as fit_2d.
    fit_2d = np.loadtxt(new_dir / "fit_2d.csv", delimiter=project.delim)
    obs_2d = np.loadtxt(new_dir / "observed_2d.csv", delimiter=project.delim)
    assert fit_2d.shape == obs_2d.shape

    # metrics.csv: one row, the canonical 5 metrics as columns.
    metrics_df = pd.read_csv(new_dir / "metrics.csv")
    assert list(metrics_df.columns) == ["chi2", "chi2_red", "r2", "aic", "bic"]
    assert len(metrics_df) == 1
