"""
Auto-export vs explicit ``Project.export_fits`` parity.

Both paths route through the same slot exporter (``fit_io._export_slot``)
since the legacy ``_save_*_fit_legacy`` savers were removed: auto-export
inside ``fit_slice_by_slice`` / ``fit_2d`` writes the grouped slot tree
under ``Project.path_results``, and an explicit ``export_fits`` call
writes it under the caller's root. The artifacts must be identical — a
divergence means one of the paths grew its own writer again.

Strategy: redirect ``path_results`` into ``tmp_path``, run the fit with
``auto_export=True`` (which writes the auto tree), then call
``project.export_fits`` into a sibling directory and diff the trees.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
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
    """Build a fit-side project + file with auto-export redirected into
    ``tmp_path / "auto"`` (auto-export writes the slot tree under
    ``project.path_results``), so the test has full control over both
    outputs and the source repo stays untouched.
    """

    data = _truth_2d_data()
    # export parity is this test's subject, so opt back into auto-export
    project = make_project(name=name, spec_fun_str=spec_fun_str, auto_export=True)
    project.path_results = tmp_path / "auto"
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
@pytest.mark.slow
def test_sbs_export_parity(tmp_path):
    """SbS auto-export tree is identical to an explicit ``export_fits`` tree.

    Both must go through ``fit_io._export_slot``; the file sets and every
    shared artifact's values are compared.
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

    auto_dir = project.path_results / file.name / "single_glp__sbs"
    new_root = tmp_path / "new"
    project.export_fits(new_root, show_output=0)
    new_dir = new_root / file.name / "single_glp__sbs"

    # --- identical artifact sets (both trees written by _export_slot)
    auto_names = {p.name for p in auto_dir.rglob("*") if p.is_file()}
    new_names = {p.name for p in new_dir.rglob("*") if p.is_file()}
    assert auto_names == new_names
    assert "fit_pars.csv" in auto_names
    assert "fit_2d.csv" in auto_names

    # --- fit_pars.csv: per-slice param values with [index, time, par...] cols
    auto_fp = pd.read_csv(auto_dir / "fit_pars.csv")
    new_fp = pd.read_csv(new_dir / "fit_pars.csv")
    assert list(auto_fp.columns) == list(new_fp.columns)
    assert auto_fp.shape == new_fp.shape
    for col in auto_fp.columns:
        np.testing.assert_allclose(
            auto_fp[col].to_numpy(dtype=float),
            new_fp[col].to_numpy(dtype=float),
            rtol=0,
            atol=0,
        )

    # --- fit_2d.csv: stacked per-slice fit spectra (n_time × n_energy),
    # both from the slot's captured ``fit`` array.
    auto_2d = np.loadtxt(auto_dir / "fit_2d.csv", delimiter=project.delim)
    new_2d = np.loadtxt(new_dir / "fit_2d.csv", delimiter=project.delim)
    assert auto_2d.shape == new_2d.shape == (len(file.time), len(file.energy))
    np.testing.assert_allclose(auto_2d, new_2d, rtol=0, atol=0)

    # --- axis sidecars
    for name, axis in (("energy.csv", file.energy), ("time.csv", file.time)):
        auto_ax = np.loadtxt(auto_dir / name, delimiter=project.delim)
        new_ax = np.loadtxt(new_dir / name, delimiter=project.delim)
        assert auto_ax.shape == new_ax.shape == (len(axis),)
        np.testing.assert_array_equal(auto_ax, new_ax)


# ---------------------------------------------------------------------------
# 2D parity
# ---------------------------------------------------------------------------


#
@pytest.mark.slow
def test_2d_export_parity(tmp_path):
    """2D auto-export tree is identical to an explicit ``export_fits`` tree."""

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

    auto_dir = project.path_results / file.name / "single_glp__2d"
    new_root = tmp_path / "new"
    project.export_fits(new_root, fit_type="2d", show_output=0)
    new_dir = new_root / file.name / "single_glp__2d"

    # --- identical artifact sets
    auto_names = {p.name for p in auto_dir.rglob("*") if p.is_file()}
    new_names = {p.name for p in new_dir.rglob("*") if p.is_file()}
    assert auto_names == new_names

    # --- fit_2d.csv (both from the slot's captured ``fit`` array)
    auto_2d = np.loadtxt(auto_dir / "fit_2d.csv", delimiter=project.delim)
    new_2d = np.loadtxt(new_dir / "fit_2d.csv", delimiter=project.delim)
    assert auto_2d.shape == new_2d.shape == (len(file.time), len(file.energy))
    np.testing.assert_allclose(auto_2d, new_2d, rtol=0, atol=0)

    # --- axis sidecars (identical writers, identical inputs)
    for name in ("energy.csv", "time.csv"):
        auto_ax = np.loadtxt(auto_dir / name, delimiter=project.delim)
        new_ax = np.loadtxt(new_dir / name, delimiter=project.delim)
        np.testing.assert_array_equal(auto_ax, new_ax)

    # --- residual-map PNG present in both
    assert (auto_dir / "2D_data_fit_res.png").exists()
    assert (new_dir / "2D_data_fit_res.png").exists()


#
@pytest.mark.slow
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

    # metrics.csv: one row, the canonical 7-key stable schema as columns
    # (raw + σ-calibrated chi² flavors, plus dimensionless r2/aic/bic).
    # Calibrated chi2 / chi2_red are NaN here since the file has no σ set.
    metrics_df = pd.read_csv(new_dir / "metrics.csv")
    assert list(metrics_df.columns) == [
        "chi2_raw",
        "chi2_red_raw",
        "chi2",
        "chi2_red",
        "r2",
        "aic",
        "bic",
    ]
    assert len(metrics_df) == 1
