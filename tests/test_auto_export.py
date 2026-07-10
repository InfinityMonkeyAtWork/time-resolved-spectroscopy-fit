"""Tests for the ``Project.auto_export`` toggle.

``auto_export=False`` must suppress the automatic CSV / PNG side effects
inside fit_baseline / fit_spectrum / fit_slice_by_slice / fit_2d while
preserving in-memory state (``_fit_history``, ``Model.result``) and the
explicit File.export_fit / Project.export_fits / save_fits paths.
"""

from unittest.mock import MagicMock

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from _utils import make_project, simulate_noisy

from trspecfit import File, fitlib
from trspecfit.utils.lmfit import MC


#
def _make_truth_file(project):
    energy = np.linspace(83, 87, 30)
    time = np.linspace(-2, 10, 24)
    file = File(parent_project=project, name="truth")
    file.energy = energy
    file.time = time
    file.dim = 2
    file.load_model(model_yaml="models/file_energy.yaml", model_info="single_glp")
    file.add_time_dependence(
        target_model="single_glp",
        target_parameter="GLP_01_A",
        dynamics_yaml="models/file_time.yaml",
        dynamics_model=["MonoExpPos"],
    )
    return file


#
def _make_fit_file(project, data, energy, time, *, name="fit"):
    file = File(
        parent_project=project,
        name=name,
        data=data,
        energy=energy.copy(),
        time=time.copy(),
    )
    file.load_model(model_yaml="models/file_energy.yaml", model_info="single_glp")
    return file


#
def _list_files(root):
    """Return the set of files (recursive) under ``root``; empty if missing."""

    if not root.exists():
        return set()
    return {p for p in root.rglob("*") if p.is_file()}


#
def _baseline_setup(tmp_path, *, auto_export: bool):
    truth_project = make_project(name="truth")
    truth = _make_truth_file(truth_project)
    data = simulate_noisy(truth.model_active, noise_level=0.01)
    project = make_project(name="fit", auto_export=auto_export)
    project.path_results = tmp_path / "auto"
    file = _make_fit_file(project, data, truth.energy, truth.time)
    file.define_baseline(time_start=0, time_stop=3, time_type="ind", show_plot=False)
    return project, file


#
class TestProjectDefault:
    """``auto_export`` defaults to ``True`` for backward compatibility."""

    #
    def test_default_is_true(self):
        project = make_project(name="default")
        # _utils.make_project respects its kwarg default, which mirrors the
        # production default declared in Project._set_defaults.
        assert project.auto_export is True

    #
    def test_toggle_can_be_flipped_after_init(self):
        project = make_project(name="flip")
        project.auto_export = False
        assert project.auto_export is False


#
class TestAutoExportFalseSuppressesSideEffects:
    """Auto-CSV/PNG side effects are skipped when ``auto_export=False``."""

    #
    def test_baseline_writes_nothing(self, tmp_path):
        project, file = _baseline_setup(tmp_path, auto_export=False)
        file.fit_baseline(model_name="single_glp", stages=2, try_ci=0)

        # In-memory state is intact.
        assert file.model_base.result is not None
        assert file.model_base.result[1] != []
        assert len(project._fit_history) == 1
        assert project._fit_history[0].fit_type == "baseline"

        # No CSV / PNG hit disk (create_model_path makes empty dirs only).
        assert _list_files(tmp_path / "auto") == set()

    #
    def test_2d_writes_nothing(self, tmp_path):
        project, file = _baseline_setup(tmp_path, auto_export=False)
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
        file.add_time_dependence(
            target_model="single_glp",
            target_parameter="GLP_01_A",
            dynamics_yaml="models/file_time.yaml",
            dynamics_model=["MonoExpPos"],
        )
        file.fit_2d("single_glp", stages=1, try_ci=0)

        assert file.model_2d.result is not None
        assert file.model_2d.result[1] != []
        assert any(slot.fit_type == "2d" for slot in project._fit_history)
        assert _list_files(tmp_path / "auto") == set()


#
class TestAutoExportTrueWritesFiles:
    """Default behavior (``auto_export=True``) preserves on-disk side effects."""

    #
    def test_baseline_writes_files(self, tmp_path):
        project, file = _baseline_setup(tmp_path, auto_export=True)
        file.fit_baseline(model_name="single_glp", stages=2, try_ci=0)
        outputs = _list_files(tmp_path / "auto")
        # At least one CSV/PNG should be auto-written by the baseline path.
        assert outputs, "expected at least one auto-written file"


#
class TestExplicitPathsStillWrite:
    """Explicit save / export bypass ``auto_export`` entirely."""

    #
    def test_export_fits_writes_under_auto_export_false(self, tmp_path):
        project, file = _baseline_setup(tmp_path, auto_export=False)
        file.fit_baseline(model_name="single_glp", stages=2, try_ci=0)

        # Auto path stayed silent.
        assert _list_files(tmp_path / "auto") == set()

        # Explicit CSV/PNG export still writes.
        explicit_root = tmp_path / "explicit_csv"
        project.export_fits(explicit_root, show_output=0)
        assert _list_files(explicit_root), (
            "project.export_fits must write even with auto_export=False"
        )

    #
    def test_save_fits_writes_under_auto_export_false(self, tmp_path):
        project, file = _baseline_setup(tmp_path, auto_export=False)
        file.fit_baseline(model_name="single_glp", stages=2, try_ci=0)

        archive = tmp_path / "explicit.fit.h5"
        project.save_fits(archive, show_output=0)
        assert archive.exists()
        assert archive.stat().st_size > 0


#
class TestPlotHelperSkipped:
    """``auto_export=False`` + silent mode must skip ``plt_fit_res_1d``
    entirely — not just suppress its save. Guards against future regressions
    where figures get built and immediately closed (the SbS hot path is the
    expensive case)."""

    #
    def test_baseline_skips_plot_when_silent_and_no_export(self, tmp_path, monkeypatch):
        mock = MagicMock()
        monkeypatch.setattr(fitlib, "plt_fit_res_1d", mock)

        project, file = _baseline_setup(tmp_path, auto_export=False)
        # show_output default in make_project is 0 (silent).
        file.fit_baseline(model_name="single_glp", stages=2, try_ci=0)

        assert mock.call_count == 0

    #
    def test_baseline_plots_when_verbose_even_without_export(
        self, tmp_path, monkeypatch
    ):
        mock = MagicMock()
        monkeypatch.setattr(fitlib, "plt_fit_res_1d", mock)

        truth_project = make_project(name="truth")
        truth = _make_truth_file(truth_project)
        data = simulate_noisy(truth.model_active, noise_level=0.01)
        project = make_project(name="fit", auto_export=False, show_output=1)
        project.path_results = tmp_path / "auto"
        file = _make_fit_file(project, data, truth.energy, truth.time)
        file.define_baseline(
            time_start=0, time_stop=3, time_type="ind", show_plot=False
        )
        file.fit_baseline(model_name="single_glp", stages=2, try_ci=0)

        # The plot still runs so the user sees results inline; only the
        # disk write was suppressed.
        assert mock.call_count == 1

    #
    def test_fit_2d_silent_mode_prints_nothing(self, tmp_path, capsys):
        """fit_2d honors show_output=0: no timing line, no params display.

        Regression: time_display and display(params) ran whenever
        stages >= 1, regardless of show_output.
        """

        project, file = _baseline_setup(tmp_path, auto_export=False)
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
        file.add_time_dependence(
            target_model="single_glp",
            target_parameter="GLP_01_A",
            dynamics_yaml="models/file_time.yaml",
            dynamics_model=["MonoExpPos"],
        )
        capsys.readouterr()  # drop setup output
        file.fit_2d("single_glp", stages=1, try_ci=0)
        assert capsys.readouterr().out == ""

    #
    @pytest.mark.slow
    def test_mcmc_silent_mode_no_output_no_figures(self, tmp_path, capsys):
        """MCMC honors silent mode: no progress banner, no shown figures.

        Regression: the emcee progress banner and progress=True ran
        unconditionally, and with show_output=0, save_output=0 the walker
        and corner figures reached _finalize_plot(0), i.e. plt.show(),
        and were left open.
        """

        project, file = _baseline_setup(tmp_path, auto_export=False)
        mc = MC(use_mc=1, steps=20, nwalkers=32, burn=5, thin=1)
        n_figs = len(plt.get_fignums())
        capsys.readouterr()  # drop setup output
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0, mc_settings=mc)

        # emcee prints its own short-chain autocorrelation notice for the
        # deliberately tiny chain; only our banner is under test here
        assert "Progress of lmfit.emcee" not in capsys.readouterr().out
        assert len(plt.get_fignums()) == n_figs

    #
    def test_sbs_skips_per_slice_plot_when_no_export(self, tmp_path, monkeypatch):
        mock = MagicMock()
        monkeypatch.setattr(fitlib, "plt_fit_res_1d", mock)

        project, file = _baseline_setup(tmp_path, auto_export=False)
        # spec_fun_str defaults to "fit_model_gir"; SbS does not lower, so
        # use the interpreter path. n_workers=1 keeps the call in-process
        # so the monkeypatch sees it.
        project.spec_fun_str = "fit_model_mcp"
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
        file.fit_slice_by_slice(
            "single_glp",
            n_workers=1,
            seed_source="model",
            seed_adapt=None,
            try_ci=0,
        )

        assert mock.call_count == 0


#
class TestVerboseDisplayWithoutExport:
    """``show_output>=1`` + ``auto_export=False`` shows the data/fit/residual
    maps inline (via ``_save_{sbs,2d}_fit_legacy(save_files=False)``) but writes
    no files — the interactive-display path mirroring fit_baseline. Guards the
    save_files=False branch added to fit_slice_by_slice / fit_2d."""

    #
    def _verbose_no_export_setup(self, tmp_path):
        truth_project = make_project(name="truth")
        truth = _make_truth_file(truth_project)
        data = simulate_noisy(truth.model_active, noise_level=0.01)
        project = make_project(name="fit", auto_export=False, show_output=1)
        project.path_results = tmp_path / "auto"
        file = _make_fit_file(project, data, truth.energy, truth.time)
        file.define_baseline(
            time_start=0, time_stop=3, time_type="ind", show_plot=False
        )
        return project, file

    #
    def test_sbs_displays_but_writes_nothing(self, tmp_path, monkeypatch):
        # plt_fit_res_2d runs in the main process (after fitting), so the
        # monkeypatch is visible regardless of worker path; n_workers=1 keeps
        # the run cheap and deterministic.
        mock_2d = MagicMock()
        monkeypatch.setattr(fitlib, "plt_fit_res_2d", mock_2d)

        project, file = self._verbose_no_export_setup(tmp_path)
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
        file.fit_slice_by_slice(
            "single_glp",
            n_workers=1,
            seed_source="model",
            seed_adapt=None,
            try_ci=0,
        )

        # Display branch ran (maps shown) but nothing hit disk.
        assert mock_2d.call_count == 1
        assert _list_files(tmp_path / "auto") == set()

    #
    def test_2d_displays_but_writes_nothing(self, tmp_path, monkeypatch):
        mock_2d = MagicMock()
        monkeypatch.setattr(fitlib, "plt_fit_res_2d", mock_2d)

        project, file = self._verbose_no_export_setup(tmp_path)
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
        file.add_time_dependence(
            target_model="single_glp",
            target_parameter="GLP_01_A",
            dynamics_yaml="models/file_time.yaml",
            dynamics_model=["MonoExpPos"],
        )
        file.fit_2d("single_glp", stages=1, try_ci=0)

        assert mock_2d.call_count == 1
        assert _list_files(tmp_path / "auto") == set()
