"""Tests for Project-level 2D fitting with shared/independent parameters.

Simulates identical data for two files from a known model, then fits them
simultaneously at the Project level with ``tau`` shared (project-vary)
and ``A`` independent (file-vary).
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from _utils import make_project, simulate_clean

from trspecfit import File


#
def _make_truth_file(*, amplitude=20.0, x0_shift=3.0, tau=5.0):
    """Create a file with known parameters for data generation.

    Uses a throwaway project so truth files don't pollute the fit project.
    """

    truth_project = make_project(name="truth")

    energy = np.linspace(83, 87, 30)
    time_ax = np.linspace(-2, 10, 24)

    file = File(parent_project=truth_project)
    file.energy = energy
    file.time = time_ax
    file.dim = 2

    file.load_model(
        model_yaml="models/project_energy.yaml",
        model_info="project_glp",
    )
    file.add_time_dependence(
        target_model="project_glp",
        target_parameter="GLP_01_x0",
        dynamics_yaml="models/project_time.yaml",
        dynamics_model=["MonoExpProject"],
    )

    # Override to truth values
    model = file.model_active
    model.lmfit_pars["GLP_01_A"].value = amplitude
    model.lmfit_pars["GLP_01_x0"].value = 85.0
    model.lmfit_pars["GLP_01_F"].value = 1.0
    model.lmfit_pars["GLP_01_m"].value = 0.3
    model.lmfit_pars["GLP_01_x0_expFun_01_A"].value = x0_shift
    model.lmfit_pars["GLP_01_x0_expFun_01_tau"].value = tau
    model.lmfit_pars["GLP_01_x0_expFun_01_t0"].value = 0.0
    model.lmfit_pars["GLP_01_x0_expFun_01_y0"].value = 0.0

    return file


#
def _make_fit_file(project, data, energy, time_ax, *, name="test"):
    """Create a fresh file with baseline + 2D model, ready to fit.

    Mirrors real workflow: load baseline model → define_baseline →
    fit_baseline → load 2D model → add_time_dependence.
    """

    file = File(
        parent_project=project,
        name=name,
        data=data,
        energy=energy.copy(),
        time=time_ax.copy(),
    )

    # Step 1: baseline model (energy-only, all vary=True)
    file.load_model(
        model_yaml="models/project_energy.yaml",
        model_info="project_glp_base",
    )
    file.define_baseline(time_start=0, time_stop=3, time_type="ind", show_plot=False)
    file.fit_baseline(model_name="project_glp_base", stages=2, try_ci=0)

    # Step 2: 2D model with dynamics (project/file/static vary levels)
    file.load_model(
        model_yaml="models/project_energy.yaml",
        model_info="project_glp",
    )
    file.add_time_dependence(
        target_model="project_glp",
        target_parameter="GLP_01_x0",
        dynamics_yaml="models/project_time.yaml",
        dynamics_model=["MonoExpProject"],
    )

    return file


#
#
class TestProjectFitClean:
    """Project-level fit on noiseless data — non-trivial roundtrips."""

    #
    @pytest.mark.slow
    def test_biexp_expr_t0_roundtrip(self):
        """Bi-exponential with t0 expression — constraint holds through fit."""

        TRUE_T0 = 3.0
        TRUE_TAU1 = 2.0
        TRUE_TAU2 = 20.0

        project = make_project(name="project_fit")

        # --- build truth files with bi-exponential + Gaussian IRF on x0 ---
        # gaussCONV smooths the hard step at t0 into a smooth onset,
        # making t0 a well-defined Gaussian center for the optimizer.
        truth_files = []
        for amp, dx0_1, dx0_2, seed in [
            (20.0, 2.0, 1.0, 42),
            (15.0, 1.5, 0.8, 43),
        ]:
            tp = make_project(name="truth")
            tf = File(parent_project=tp)
            tf.energy = np.linspace(80, 90, 50)
            tf.time = np.linspace(-5, 50, 120)
            tf.dim = 2
            tf.load_model(
                model_yaml="models/project_energy.yaml",
                model_info="project_glp",
            )
            tf.add_time_dependence(
                target_model="project_glp",
                target_parameter="GLP_01_x0",
                dynamics_yaml="models/project_time.yaml",
                dynamics_model=["BiExpProject"],
            )
            m = tf.model_active
            m.lmfit_pars["GLP_01_A"].value = amp
            m.lmfit_pars["GLP_01_x0"].value = 85.0
            m.lmfit_pars["GLP_01_F"].value = 1.0
            m.lmfit_pars["GLP_01_m"].value = 0.3
            m.lmfit_pars["GLP_01_x0_expFun_01_A"].value = dx0_1
            m.lmfit_pars["GLP_01_x0_expFun_01_tau"].value = TRUE_TAU1
            m.lmfit_pars["GLP_01_x0_expFun_01_t0"].value = TRUE_T0
            m.lmfit_pars["GLP_01_x0_expFun_01_y0"].value = 0.0
            m.lmfit_pars["GLP_01_x0_expFun_02_A"].value = dx0_2
            m.lmfit_pars["GLP_01_x0_expFun_02_tau"].value = TRUE_TAU2
            m.lmfit_pars["GLP_01_x0_expFun_02_t0"].value = TRUE_T0
            m.lmfit_pars["GLP_01_x0_expFun_02_y0"].value = 0.0
            truth_files.append((tf, seed))

        # --- simulate and build fit files ---
        for file_idx, (tf, seed) in enumerate(truth_files):
            clean = simulate_clean(tf.model_active, seed=seed)

            ff = File(
                parent_project=project,
                name=f"file_{file_idx}",
                data=clean,
                energy=tf.energy.copy(),
                time=tf.time.copy(),
            )
            ff.load_model(
                model_yaml="models/project_energy.yaml",
                model_info="project_glp_base",
            )
            ff.define_baseline(
                time_start=-5,
                time_stop=0,
                time_type="abs",
                show_plot=False,
            )
            ff.fit_baseline(model_name="project_glp_base", stages=2, try_ci=0)
            ff.load_model(
                model_yaml="models/project_energy.yaml",
                model_info="project_glp",
            )
            ff.add_time_dependence(
                target_model="project_glp",
                target_parameter="GLP_01_x0",
                dynamics_yaml="models/project_time.yaml",
                dynamics_model=["BiExpProject"],
            )

        # --- project-level fit ---
        project.fit_2d(model_name="project_glp", stages=2, try_ci=0)

        # --- assertions ---
        for i, f in enumerate(project.files):
            m = f.select_model("project_glp")
            assert m is not None  # type guard

            t0_01 = m.lmfit_pars["GLP_01_x0_expFun_01_t0"].value
            t0_02 = m.lmfit_pars["GLP_01_x0_expFun_02_t0"].value

            # Expression constraint: t0_02 == t0_01
            assert t0_01 == t0_02, (
                f"file {i}: t0 expression broken: "
                f"expFun_01_t0={t0_01}, expFun_02_t0={t0_02}"
            )
            # Recovery of true value
            assert np.isclose(t0_01, TRUE_T0, atol=0.1), (
                f"file {i}: t0 recovery: true={TRUE_T0}, fit={t0_01:.4f}"
            )

        # t0 is project-vary — same across both files
        m0 = project.files[0].select_model("project_glp")
        m1 = project.files[1].select_model("project_glp")
        assert m0 is not None  # type guard
        assert m1 is not None  # type guard
        t0_f0 = m0.lmfit_pars["GLP_01_x0_expFun_01_t0"].value
        t0_f1 = m1.lmfit_pars["GLP_01_x0_expFun_01_t0"].value
        assert t0_f0 == t0_f1, f"t0 should be project-shared: {t0_f0} != {t0_f1}"


#
#
class TestVaryLevelParsing:
    """Test that project/file/static vary levels are correctly parsed."""

    #
    def test_vary_levels_on_par(self):
        """Par.vary_level is set from YAML."""

        project = make_project(name="project_fit")
        file = File(parent_project=project)
        file.energy = np.linspace(83, 87, 10)
        file.time = np.linspace(-2, 10, 10)
        file.dim = 2

        file.load_model(
            model_yaml="models/project_energy.yaml",
            model_info="project_glp",
        )

        model = file.model_active
        all_pars = model.get_all_parameters()
        par_levels = {p.name: p.vary_level for p in all_pars}

        assert par_levels["GLP_01_A"] == "file"
        assert par_levels["GLP_01_x0"] == "file"
        assert par_levels["GLP_01_F"] == "static"
        assert par_levels["GLP_01_m"] == "static"

    #
    def test_vary_levels_map(self):
        """Model.get_vary_levels returns correct levels for all params."""

        project = make_project(name="project_fit")
        file = File(parent_project=project)
        file.energy = np.linspace(83, 87, 10)
        file.time = np.linspace(-2, 10, 10)
        file.dim = 2

        file.load_model(
            model_yaml="models/project_energy.yaml",
            model_info="project_glp",
        )
        file.add_time_dependence(
            target_model="project_glp",
            target_parameter="GLP_01_x0",
            dynamics_yaml="models/project_time.yaml",
            dynamics_model=["MonoExpProject"],
        )

        model = file.model_active
        levels = model.get_vary_levels()

        assert levels["GLP_01_A"] == "file"
        assert levels["GLP_01_x0"] == "file"
        assert levels["GLP_01_F"] == "static"
        assert levels["GLP_01_m"] == "static"
        # Dynamics sub-model params
        assert levels["GLP_01_x0_expFun_01_A"] == "file"
        assert levels["GLP_01_x0_expFun_01_tau"] == "project"
        assert levels["GLP_01_x0_expFun_01_t0"] == "static"
        assert levels["GLP_01_x0_expFun_01_y0"] == "static"

    #
    def test_vary_levels_profile_with_dynamics(self):
        """get_vary_levels includes dynamics params nested under a profile param.

        Profile param GLP_01_A_pLinear_01_m gets MonoExpProject dynamics.
        The resulting dynamics params (expFun_01_tau at 'project',
        expFun_01_A at 'file') must appear in get_vary_levels(), not be
        silently absent (which would cause _build_fit_params to freeze them).
        """

        project = make_project(name="project_fit")
        file = File(parent_project=project, aux_axis=np.linspace(0, 4, 5))
        file.energy = np.linspace(83, 87, 10)
        file.time = np.linspace(-2, 10, 10)
        file.dim = 2

        file.load_model(
            model_yaml="models/project_energy.yaml",
            model_info="project_glp",
        )
        file.add_par_profile(
            target_model="project_glp",
            target_parameter="GLP_01_A",
            profile_yaml="models/file_profile.yaml",
            profile_model=["profile_pLinear"],
        )
        file.add_time_dependence(
            target_model="project_glp",
            target_parameter="GLP_01_A_pLinear_01_m",
            dynamics_yaml="models/project_time.yaml",
            dynamics_model=["MonoExpProject"],
        )

        model = file.model_active
        levels = model.get_vary_levels()

        # Nested dynamics params must appear — not fall through to "static" default
        tau_name = "GLP_01_A_pLinear_01_m_expFun_01_tau"
        A_name = "GLP_01_A_pLinear_01_m_expFun_01_A"
        assert tau_name in model.lmfit_pars  # confirm they're in the model
        assert A_name in model.lmfit_pars
        assert levels[tau_name] == "project"
        assert levels[A_name] == "file"

    #
    def test_file_level_fit_treats_project_as_vary(self):
        """File-level fitting treats both 'project' and 'file' as vary=True."""

        project = make_project(name="project_fit")
        file = File(parent_project=project)
        file.energy = np.linspace(83, 87, 10)
        file.time = np.linspace(-2, 10, 10)
        file.dim = 2

        file.load_model(
            model_yaml="models/project_energy.yaml",
            model_info="project_glp",
        )

        model = file.model_active
        # "file" and "project" vary levels should map to lmfit vary=True
        assert model.lmfit_pars["GLP_01_A"].vary is True
        assert model.lmfit_pars["GLP_01_x0"].vary is True
        # "static" should map to vary=False
        assert model.lmfit_pars["GLP_01_F"].vary is False
        assert model.lmfit_pars["GLP_01_m"].vary is False


#
#
class TestBuildFitParams:
    """Test Project._build_fit_params assembly logic."""

    #
    def test_combined_params_structure(self):
        """Combined params have prefixed file-vary and unprefixed project-vary."""

        project = make_project(name="project_fit")

        for i in range(2):
            f = File(parent_project=project, name=f"file_{i}")
            f.energy = np.linspace(83, 87, 10)
            f.time = np.linspace(-2, 10, 10)
            f.dim = 2
            f.load_model(
                model_yaml="models/project_energy.yaml",
                model_info="project_glp",
            )
            f.add_time_dependence(
                target_model="project_glp",
                target_parameter="GLP_01_x0",
                dynamics_yaml="models/project_time.yaml",
                dynamics_model=["MonoExpProject"],
            )

        combined, info = project._build_fit_params(model_name="project_glp")

        # project-vary: tau appears once (not prefixed)
        assert "GLP_01_x0_expFun_01_tau" in combined
        # file-vary: A appears twice (prefixed per file)
        assert "file00_GLP_01_A" in combined
        assert "file01_GLP_01_A" in combined
        # file-vary dynamics A
        assert "file00_GLP_01_x0_expFun_01_A" in combined
        assert "file01_GLP_01_x0_expFun_01_A" in combined
        # static: F appears per file but vary=False
        assert combined["file00_GLP_01_F"].vary is False
        assert combined["file01_GLP_01_F"].vary is False
        # project tau should be vary=True
        assert combined["GLP_01_x0_expFun_01_tau"].vary is True

    #
    def test_expressions_rewritten_with_prefix(self):
        """Expressions referencing file-vary params get file prefix."""

        project = make_project(name="project_fit")

        for i in range(2):
            f = File(parent_project=project, name=f"file_{i}")
            f.energy = np.linspace(83, 87, 10)
            f.time = np.linspace(-2, 10, 10)
            f.dim = 2
            f.load_model(
                model_yaml="models/project_energy.yaml",
                model_info="project_glp_expr",
            )

        combined, info = project._build_fit_params(
            model_name="project_glp_expr",
        )

        # GLP_02_A is an expression "GLP_01_A * 0.5"
        # For file00 it should become "file00_GLP_01_A * 0.5"
        par_00 = combined["file00_GLP_02_A"]
        par_01 = combined["file01_GLP_02_A"]
        assert par_00.expr == "file00_GLP_01_A * 0.5"
        assert par_01.expr == "file01_GLP_01_A * 0.5"

    #
    def test_expr_referencing_project_vary_stays_unprefixed(self):
        """Expression referencing a project-vary param keeps unprefixed name."""

        project = make_project(name="project_fit")

        for i in range(2):
            f = File(parent_project=project, name=f"file_{i}")
            f.energy = np.linspace(83, 87, 10)
            f.time = np.linspace(-2, 10, 10)
            f.dim = 2
            f.load_model(
                model_yaml="models/project_energy.yaml",
                model_info="project_glp",
            )
            f.add_time_dependence(
                target_model="project_glp",
                target_parameter="GLP_01_x0",
                dynamics_yaml="models/project_time.yaml",
                dynamics_model=["BiExpProject"],
            )

        combined, info = project._build_fit_params(model_name="project_glp")

        # expFun_01_t0 is project-vary → unprefixed
        t0_name = "GLP_01_x0_expFun_01_t0"
        assert t0_name in combined
        assert combined[t0_name].vary is True

        # expFun_02_t0 is an expression referencing expFun_01_t0
        # The reference should stay unprefixed (project-vary target)
        t0_02_file00 = combined["file00_GLP_01_x0_expFun_02_t0"]
        t0_02_file01 = combined["file01_GLP_01_x0_expFun_02_t0"]
        assert t0_02_file00.expr == t0_name
        assert t0_02_file01.expr == t0_name

    #
    def test_project_vary_initial_value_conflict_warns(self):
        """Warn when project-vary param has different initial values across files."""

        project = make_project(name="project_fit")

        for i in range(2):
            f = File(parent_project=project, name=f"file_{i}")
            f.energy = np.linspace(83, 87, 10)
            f.time = np.linspace(-2, 10, 10)
            f.dim = 2
            f.load_model(
                model_yaml="models/project_energy.yaml",
                model_info="project_glp",
            )
            f.add_time_dependence(
                target_model="project_glp",
                target_parameter="GLP_01_x0",
                dynamics_yaml="models/project_time.yaml",
                dynamics_model=["MonoExpProject"],
            )

        # Manually set different initial tau on file 1
        model1 = project.files[1].select_model("project_glp")
        assert model1 is not None  # type guard
        model1.lmfit_pars["GLP_01_x0_expFun_01_tau"].value = 99.0

        with pytest.warns(UserWarning, match="different initial values"):
            project._build_fit_params(model_name="project_glp")

    #
    def test_project_vary_bound_conflict_raises(self):
        """Raise when project-vary param has different min or max across files."""

        project = make_project(name="project_fit")

        for i in range(2):
            f = File(parent_project=project, name=f"file_{i}")
            f.energy = np.linspace(83, 87, 10)
            f.time = np.linspace(-2, 10, 10)
            f.dim = 2
            f.load_model(
                model_yaml="models/project_energy.yaml",
                model_info="project_glp",
            )
            f.add_time_dependence(
                target_model="project_glp",
                target_parameter="GLP_01_x0",
                dynamics_yaml="models/project_time.yaml",
                dynamics_model=["MonoExpProject"],
            )

        model1 = project.files[1].select_model("project_glp")
        assert model1 is not None  # type guard
        model1.lmfit_pars["GLP_01_x0_expFun_01_tau"].min = 0.01
        model1.lmfit_pars["GLP_01_x0_expFun_01_tau"].max = 999.0

        with pytest.raises(ValueError, match="different min bounds"):
            project._build_fit_params(model_name="project_glp")


#
#
class TestProjectFitLifecycle:
    """Project.fit_2d() populates file.model_2d so the standard post-fit
    API (get_fit_results, save_2d_fit) works on project-fitted files."""

    #
    @pytest.mark.slow
    def test_model_2d_set_after_project_fit(self):
        """file.model_2d is set on every file after Project.fit_2d()."""

        project = make_project(name="project_fit")
        truth = _make_truth_file()
        clean = simulate_clean(truth.model_active)

        for i in range(2):
            _make_fit_file(project, clean, truth.energy, truth.time, name=f"file_{i}")

        project.fit_2d(model_name="project_glp", stages=2, try_ci=0)

        for f in project.files:
            assert f.model_2d is not None

    #
    @pytest.mark.slow
    def test_get_fit_results_2d_works_after_project_fit(self):
        """get_fit_results("2d") returns a DataFrame on project-fitted files."""

        project = make_project(name="project_fit")
        truth = _make_truth_file()
        clean = simulate_clean(truth.model_active)

        for i in range(2):
            _make_fit_file(project, clean, truth.energy, truth.time, name=f"file_{i}")

        project.fit_2d(model_name="project_glp", stages=2, try_ci=0)

        for f in project.files:
            df = f.get_fit_results(fit_type="2d")
            assert df is not None
            assert "GLP_01_x0_expFun_01_tau" in df["name"].values

    #
    @pytest.mark.slow
    def test_save_2d_fit_works_after_project_fit(self, tmp_path):
        """save_2d_fit() runs without error on project-fitted files."""

        project = make_project(name="project_fit")
        truth = _make_truth_file()
        clean = simulate_clean(truth.model_active)

        for i in range(2):
            _make_fit_file(project, clean, truth.energy, truth.time, name=f"file_{i}")

        project.fit_2d(model_name="project_glp", stages=2, try_ci=0)

        for f in project.files:
            f.save_2d_fit(save_path=tmp_path)  # must not raise

    #
    @pytest.mark.slow
    def test_fit_history_populated_after_project_fit(self):
        """Project.fit_2d() appends a 2D slot per file to _fit_history."""

        project = make_project(name="project_fit")
        truth = _make_truth_file()
        clean = simulate_clean(truth.model_active)

        for i in range(2):
            _make_fit_file(project, clean, truth.energy, truth.time, name=f"file_{i}")

        project.fit_2d(model_name="project_glp", stages=2, try_ci=0)

        twod_slots = [s for s in project._fit_history if s.fit_type == "2d"]
        assert len(twod_slots) == len(project.files)
        # Slots are tagged with each file's name, observed/fit grids align,
        # and conf_ci is absent (joint covariance does not decompose per file).
        slot_files = {s.file_name for s in twod_slots}
        assert slot_files == {f.name for f in project.files}
        for slot in twod_slots:
            assert slot.observed.ndim == 2
            assert slot.observed.shape == slot.fit.shape
            assert slot.conf_ci is None

    #
    @pytest.mark.slow
    def test_num_fmt_and_delim_propagate_to_csv_outputs(self):
        """Custom num_fmt/delim on the Project flow into fit-CSV writes.

        Covers two pandas ``to_csv`` paths exercised by fit_baseline:
        - fit_wrapper -> ``{model}_par_fin.csv``
        - save_baseline_fit -> ``fit_1d.csv``
        """

        project = make_project(name="num_fmt_test")
        project.num_fmt = "%.3f"
        project.delim = ";"

        truth = _make_truth_file()
        clean = simulate_clean(truth.model_active)
        _make_fit_file(project, clean, truth.energy, truth.time, name="file_fmt")

        f = project.files[0]
        base_dir = project.path_results / f.name / "baseline" / "project_glp_base"

        # fit_wrapper writes <model>_par_fin.csv via pandas to_csv
        par_fin_lines = (
            (base_dir / "project_glp_base_par_fin.csv").read_text().splitlines()
        )
        assert ";" in par_fin_lines[0]  # custom delimiter on header
        value_field = par_fin_lines[1].split(";")[1]
        # %.3f -> fixed-point; %.6e fallback would contain 'e'
        assert "." in value_field and "e" not in value_field.lower(), value_field

        # save_baseline_fit writes fit_1d.csv via pandas to_csv
        fit_1d_lines = (base_dir / "fit_1d.csv").read_text().splitlines()
        assert fit_1d_lines[0].startswith("energy;sum;")
        energy_field = fit_1d_lines[1].split(";")[0]
        assert "." in energy_field and "e" not in energy_field.lower(), energy_field
