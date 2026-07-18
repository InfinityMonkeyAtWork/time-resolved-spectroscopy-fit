"""Fits never write to disk (v0.14.0).

fit_baseline / fit_spectrum / fit_slice_by_slice / fit_2d compute, display
(per ``show_output``), and capture fit slots — persistence is only ever
the explicit ``save_fits`` (HDF5) / ``export_fits`` (CSV/PNG) calls. The
write-nothing tests run each fit with the working directory pointed at an
empty ``tmp_path`` so any accidental relative-path write is caught. The
display/silent guardrail matrix (plot helpers skipped when silent, shown
when verbose, never written) lives here too, as does the loud failure on
removed ``project.yaml`` keys.
"""

import pathlib
from unittest.mock import MagicMock

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from _utils import simulate_noisy

from trspecfit import File, Project, fitlib
from trspecfit.utils.lmfit import MC

TESTS_DIR = pathlib.Path(__file__).resolve().parent


#
def _make_abs_project(*, name="fit", show_output=0):
    """Project anchored at the tests dir by absolute path, so tests can
    chdir into a tmp_path without breaking YAML/model resolution."""

    project = Project(path=TESTS_DIR, name=name)
    project.show_output = show_output
    project.spec_fun_str = "fit_model_gir"
    return project


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
def _baseline_setup(tmp_path, monkeypatch, *, show_output=0):
    """Build a fit-ready project/file and chdir into an empty tmp_path so
    any file a fit method writes (absolute or relative) is detectable."""

    truth_project = _make_abs_project(name="truth")
    truth = _make_truth_file(truth_project)
    data = simulate_noisy(truth.model_active, noise_level=0.01)
    project = _make_abs_project(name="fit", show_output=show_output)
    file = _make_fit_file(project, data, truth.energy, truth.time)
    file.define_baseline(time_start=0, time_stop=3, time_type="ind", show_plot=False)
    monkeypatch.chdir(tmp_path)
    return project, file


#
def _add_dynamics(file):
    file.add_time_dependence(
        target_model="single_glp",
        target_parameter="GLP_01_A",
        dynamics_yaml="models/file_time.yaml",
        dynamics_model=["MonoExpPos"],
    )


#
class TestDefaults:
    #
    def test_project_name_default_is_placeholder(self):
        project = Project(path=TESTS_DIR, config_file=None)
        assert project.name == "my_project"


#
class TestRemovedConfigKeys:
    """Removed ``project.yaml`` keys fail loudly instead of being silently
    ignored — an old config relying on them would otherwise change behavior
    without a trace."""

    #
    @pytest.mark.parametrize("key", ["auto_export", "path_results"])
    def test_removed_key_raises(self, tmp_path, key):
        (tmp_path / "project.yaml").write_text(f"{key}: false\n")
        with pytest.raises(ValueError, match=f"'{key}' was removed"):
            Project(path=tmp_path)


#
class TestFitsWriteNothing:
    """Every fit method leaves the filesystem untouched while keeping the
    in-memory state (``Model.result``, fit slots) intact."""

    #
    def test_baseline_writes_nothing(self, tmp_path, monkeypatch):
        project, file = _baseline_setup(tmp_path, monkeypatch)
        file.fit_baseline(model_name="single_glp", stages=2, try_ci=0)

        assert file.model_base.result is not None
        assert file.model_base.result.par_fin.success
        assert len(project._fit_history) == 1
        assert project._fit_history[0].fit_type == "baseline"
        assert _list_files(tmp_path) == set()

    #
    def test_spectrum_writes_nothing(self, tmp_path, monkeypatch):
        project, file = _baseline_setup(tmp_path, monkeypatch)
        file.fit_spectrum(
            "single_glp", time_point=0, time_type="ind", stages=1, try_ci=0
        )

        assert any(slot.fit_type == "spectrum" for slot in project._fit_history)
        assert _list_files(tmp_path) == set()

    #
    def test_sbs_writes_nothing(self, tmp_path, monkeypatch):
        project, file = _baseline_setup(tmp_path, monkeypatch)
        # SbS does not lower on the GIR path; use the interpreter. Serial
        # (n_workers=1) keeps the run cheap and in-process.
        project.spec_fun_str = "fit_model_mcp"
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
        file.fit_slice_by_slice(
            "single_glp",
            n_workers=1,
            seed_source="model",
            seed_adapt=None,
            try_ci=0,
        )

        assert any(slot.fit_type == "sbs" for slot in project._fit_history)
        assert _list_files(tmp_path) == set()

    #
    def test_2d_writes_nothing(self, tmp_path, monkeypatch):
        project, file = _baseline_setup(tmp_path, monkeypatch)
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
        _add_dynamics(file)
        file.fit_2d("single_glp", stages=1, try_ci=0)

        assert file.model_2d.result is not None
        assert file.model_2d.result.par_fin.success
        assert any(slot.fit_type == "2d" for slot in project._fit_history)
        assert _list_files(tmp_path) == set()


#
class TestExplicitPathsStillWrite:
    """``save_fits`` / ``export_fits`` are the only persistence paths."""

    #
    def test_export_fits_writes(self, tmp_path, monkeypatch):
        project, file = _baseline_setup(tmp_path, monkeypatch)
        file.fit_baseline(model_name="single_glp", stages=2, try_ci=0)

        explicit_root = tmp_path / "explicit_csv"
        project.export_fits(explicit_root, show_output=0)
        assert _list_files(explicit_root)

    #
    def test_save_fits_writes(self, tmp_path, monkeypatch):
        project, file = _baseline_setup(tmp_path, monkeypatch)
        file.fit_baseline(model_name="single_glp", stages=2, try_ci=0)

        archive = tmp_path / "explicit.fit.h5"
        project.save_fits(archive, show_output=0)
        assert archive.exists()
        assert archive.stat().st_size > 0


#
class TestPlotHelperSkipped:
    """Silent mode must skip ``plt_fit_res_1d`` entirely — not just
    suppress its display. Guards against future regressions where figures
    get built and immediately closed (the SbS hot path is the expensive
    case)."""

    #
    def test_baseline_skips_plot_when_silent(self, tmp_path, monkeypatch):
        mock = MagicMock()
        monkeypatch.setattr(fitlib, "plt_fit_res_1d", mock)

        project, file = _baseline_setup(tmp_path, monkeypatch)
        # show_output defaults to 0 (silent) in _baseline_setup.
        file.fit_baseline(model_name="single_glp", stages=2, try_ci=0)

        assert mock.call_count == 0

    #
    def test_baseline_plots_when_verbose(self, tmp_path, monkeypatch):
        mock = MagicMock()
        monkeypatch.setattr(fitlib, "plt_fit_res_1d", mock)

        project, file = _baseline_setup(tmp_path, monkeypatch, show_output=1)
        file.fit_baseline(model_name="single_glp", stages=2, try_ci=0)

        # The plot runs so the user sees results inline; nothing hits disk.
        assert mock.call_count == 1
        assert _list_files(tmp_path) == set()

    #
    def test_fit_2d_silent_mode_prints_nothing(self, tmp_path, monkeypatch, capsys):
        """fit_2d honors show_output=0: no timing line, no params display.

        Regression: time_display and display(params) ran whenever
        stages >= 1, regardless of show_output.
        """

        project, file = _baseline_setup(tmp_path, monkeypatch)
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
        _add_dynamics(file)
        capsys.readouterr()  # drop setup output
        file.fit_2d("single_glp", stages=1, try_ci=0)
        assert capsys.readouterr().out == ""

    #
    @pytest.mark.slow
    def test_mcmc_silent_mode_no_output_no_figures(self, tmp_path, capsys, monkeypatch):
        """MCMC honors silent mode: no progress banner, no figures built.

        Regression: the emcee progress banner and progress=True ran
        unconditionally, and with show_output=0 the walker and corner
        figures were built, shown, and left open. When silent the figures
        must not be constructed at all.
        """

        mock_corner = MagicMock()
        monkeypatch.setattr(fitlib, "corner", mock_corner)

        project, file = _baseline_setup(tmp_path, monkeypatch)
        mc = MC(use_mc=1, steps=20, nwalkers=32, burn=5, thin=1)
        n_figs = len(plt.get_fignums())
        capsys.readouterr()  # drop setup output
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0, mc_settings=mc)

        # emcee prints its own short-chain autocorrelation notice for the
        # deliberately tiny chain; only our banner is under test here
        assert "Progress of lmfit.emcee" not in capsys.readouterr().out
        assert len(plt.get_fignums()) == n_figs
        assert mock_corner.corner.call_count == 0
        assert _list_files(tmp_path) == set()

    #
    def test_sbs_never_plots_per_slice_during_fit(self, tmp_path, monkeypatch):
        """The SbS fit loop builds no per-slice figures; per-slice panels
        are on-demand via File.plot_sbs_slices."""

        mock = MagicMock()
        monkeypatch.setattr(fitlib, "plt_fit_res_1d", mock)

        project, file = _baseline_setup(tmp_path, monkeypatch)
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
class TestPlotSbsSlices:
    """File.plot_sbs_slices: on-demand per-slice diagnostics from the live
    SbS fit state — display-only by default, PNGs only on explicit
    ``save_path``."""

    #
    def _sbs_fit(self, tmp_path, monkeypatch):
        project, file = _baseline_setup(tmp_path, monkeypatch)
        project.spec_fun_str = "fit_model_mcp"
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
        file.fit_slice_by_slice(
            "single_glp",
            n_workers=1,
            seed_source="model",
            seed_adapt=None,
            try_ci=0,
        )
        return project, file

    #
    def test_display_only_writes_nothing(self, tmp_path, monkeypatch):
        _, file = self._sbs_fit(tmp_path, monkeypatch)
        plt.close("all")
        file.plot_sbs_slices(slices=[0, 1])  # show under Agg
        try:
            assert len(plt.get_fignums()) == 2
            assert _list_files(tmp_path) == set()
        finally:
            plt.close("all")

    #
    def test_save_path_writes_one_png_per_slice(self, tmp_path, monkeypatch):
        project, file = self._sbs_fit(tmp_path, monkeypatch)
        out = tmp_path / "slices"
        file.plot_sbs_slices(slices=[0, 2], save_path=out, show_plot=False)
        expected = {str(project.da_slices_fmt % s) + ".png" for s in (0, 2)}
        assert {p.name for p in out.iterdir()} == expected

    #
    def test_raises_without_live_results(self, tmp_path, monkeypatch):
        _, file = _baseline_setup(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="fit_slice_by_slice"):
            file.plot_sbs_slices(show_plot=False)

    #
    def test_raises_on_model_mismatch(self, tmp_path, monkeypatch):
        _, file = self._sbs_fit(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="most recent"):
            file.plot_sbs_slices(model="other_model", show_plot=False)

    #
    def test_raises_on_out_of_range_slice(self, tmp_path, monkeypatch):
        _, file = self._sbs_fit(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="out of range"):
            file.plot_sbs_slices(slices=[9999], show_plot=False)


#
class TestVerboseDisplay:
    """``show_output>=1`` shows the data/fit/residual maps inline via the
    plot API but writes no files — guards the display path in
    fit_slice_by_slice / fit_2d."""

    #
    def test_sbs_displays_but_writes_nothing(self, tmp_path, monkeypatch):
        # plt_fit_res_2d runs in the main process (after fitting), so the
        # monkeypatch is visible regardless of worker path; n_workers=1 keeps
        # the run cheap and deterministic.
        mock_2d = MagicMock()
        monkeypatch.setattr(fitlib, "plt_fit_res_2d", mock_2d)

        project, file = _baseline_setup(tmp_path, monkeypatch, show_output=1)
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
        assert _list_files(tmp_path) == set()

    #
    def test_2d_displays_but_writes_nothing(self, tmp_path, monkeypatch):
        mock_2d = MagicMock()
        monkeypatch.setattr(fitlib, "plt_fit_res_2d", mock_2d)

        project, file = _baseline_setup(tmp_path, monkeypatch, show_output=1)
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
        _add_dynamics(file)
        file.fit_2d("single_glp", stages=1, try_ci=0)

        assert mock_2d.call_count == 1
        assert _list_files(tmp_path) == set()
