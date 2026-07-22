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
def _make_truth_file(
    *, amplitude=20.0, x0_shift=3.0, tau=5.0, energy=None, time_ax=None
):
    """Create a file with known parameters for data generation.

    Uses a throwaway project so truth files don't pollute the fit project.
    Pass ``energy``/``time_ax`` to override the default grids (used by
    heterogeneous-grid tests).
    """

    truth_project = make_project(name="truth")

    energy = np.linspace(83, 87, 30) if energy is None else energy
    time_ax = np.linspace(-2, 10, 24) if time_ax is None else time_ax

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
            m.lmfit_pars["GLP_01_x0_expFun_02_A"].value = dx0_2
            m.lmfit_pars["GLP_01_x0_expFun_02_tau"].value = TRUE_TAU2
            m.lmfit_pars["GLP_01_x0_expFun_02_t0"].value = TRUE_T0
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
    API (get_fit_results, export_fit) works on project-fitted files."""

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
    def test_export_fit_works_after_project_fit(self, tmp_path):
        """Slot export runs without error on project-fitted files."""

        project = make_project(name="project_fit")
        truth = _make_truth_file()
        clean = simulate_clean(truth.model_active)

        for i in range(2):
            _make_fit_file(project, clean, truth.energy, truth.time, name=f"file_{i}")

        project.fit_2d(model_name="project_glp", stages=2, try_ci=0)

        for f in project.files:
            f.export_fit(tmp_path, fit_type="2d", show_output=0)
            slot_dir = tmp_path / f.name / "project_glp__2d"
            assert (slot_dir / "fit_2d.csv").exists()

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
    def test_num_fmt_and_delim_propagate_to_csv_outputs(self, tmp_path):
        """Custom num_fmt/delim on the Project flow into explicit CSV writes
        (``save_baseline_fit`` -> ``fit_1d.csv``)."""

        project = make_project(name="num_fmt_test")
        project.num_fmt = "%.3f"
        project.delim = ";"

        truth = _make_truth_file()
        clean = simulate_clean(truth.model_active)
        _make_fit_file(project, clean, truth.energy, truth.time, name="file_fmt")

        f = project.files[0]
        base_dir = tmp_path / "base"
        f.save_baseline_fit(save_path=base_dir)

        fit_1d_lines = (base_dir / "fit_1d.csv").read_text().splitlines()
        assert fit_1d_lines[0].startswith("energy;sum;")
        energy_field = fit_1d_lines[1].split(";")[0]
        # %.3f -> fixed-point; %.6e fallback would contain 'e'
        assert "." in energy_field and "e" not in energy_field.lower(), energy_field


#
#
class TestPackProjectTheta:
    """Test spectra.pack_project_theta index-array assembly."""

    #
    def _stub_plan(self, opt_param_names):
        import types

        return types.SimpleNamespace(opt_param_names=opt_param_names)

    #
    def test_synthetic_shared_and_file_params(self):
        """Shared param gathers to one theta_c slot; plan order preserved."""

        mapping = [
            ("tau", 0, "tau"),
            ("file00_A", 0, "A"),
            ("file00_F", 0, "F"),
            ("tau", 1, "tau"),
            ("file01_A", 1, "A"),
            ("file01_F", 1, "F"),
        ]
        par_names = ["tau", "file00_A", "file00_F", "file01_A", "file01_F"]
        var_names = ["tau", "file00_A", "file01_A"]
        # opt order differs between the plans on purpose
        plans = [self._stub_plan(["A", "tau"]), self._stub_plan(["tau", "A"])]

        from trspecfit import spectra

        theta_c_indices, plan_gathers = spectra.pack_project_theta(
            plans,
            mapping=mapping,
            par_names=par_names,
            var_names=var_names,
        )

        assert theta_c_indices.tolist() == [0, 1, 3]
        assert plan_gathers[0].tolist() == [1, 0]
        assert plan_gathers[1].tolist() == [0, 2]

        # end-to-end gather: full combined vector -> per-plan theta
        par_full = np.array([5.0, 20.0, 0.1, 30.0, 0.2])
        theta_c = par_full[theta_c_indices]
        assert theta_c[plan_gathers[0]].tolist() == [20.0, 5.0]
        assert theta_c[plan_gathers[1]].tolist() == [5.0, 30.0]

    #
    def test_missing_mapping_entry_raises(self):
        """Plan opt param without a mapping entry is an internal error."""

        from trspecfit import spectra

        with pytest.raises(RuntimeError, match="no combined-parameter mapping"):
            spectra.pack_project_theta(
                [self._stub_plan(["GLP_01_A"])],
                mapping=[],
                par_names=[],
                var_names=[],
            )

    #
    def test_non_varying_counterpart_raises(self):
        """Plan opt param mapping to a static combined param is an error."""

        from trspecfit import spectra

        with pytest.raises(RuntimeError, match="not\\s+varying"):
            spectra.pack_project_theta(
                [self._stub_plan(["GLP_01_F"])],
                mapping=[("file00_GLP_01_F", 0, "GLP_01_F")],
                par_names=["file00_GLP_01_F"],
                var_names=[],
            )

    #
    def test_unconsumed_varying_param_raises(self):
        """A varying combined param feeding no plan is an error."""

        from trspecfit import spectra

        with pytest.raises(RuntimeError, match="feed no plan"):
            spectra.pack_project_theta(
                [self._stub_plan(["GLP_01_A"])],
                mapping=[("file00_GLP_01_A", 0, "GLP_01_A")],
                par_names=["file00_GLP_01_A", "orphan"],
                var_names=["file00_GLP_01_A", "orphan"],
            )

    #
    def test_real_models_roundtrip(self):
        """Packing built from real plans reproduces name-based distribution."""

        from trspecfit import spectra
        from trspecfit.graph_ir import build_graph, can_lower_2d, schedule_2d

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
        par_names = info["par_names"]
        var_names = [n for n in par_names if combined[n].vary]

        plans = []
        for model in info["models"]:
            graph = build_graph(model)
            assert can_lower_2d(graph)
            plans.append(schedule_2d(graph))

        theta_c_indices, plan_gathers = spectra.pack_project_theta(
            plans,
            mapping=info["mapping"],
            par_names=par_names,
            var_names=var_names,
        )

        # gathered values must equal the name-based lookup per file
        remaps = [{}, {}]
        for combined_name, file_idx, local_name in info["mapping"]:
            remaps[file_idx][local_name] = combined_name
        par_full = np.array([combined[n].value for n in par_names])
        theta_c = par_full[theta_c_indices]
        for file_idx, plan in enumerate(plans):
            expected = [
                combined[remaps[file_idx][local]].value
                for local in plan.opt_param_names
            ]
            assert theta_c[plan_gathers[file_idx]].tolist() == expected

        # shared tau lands on the same theta_c slot for both files
        tau_name = "GLP_01_x0_expFun_01_tau"
        tau_slot = var_names.index(tau_name)
        for file_idx, plan in enumerate(plans):
            local_tau = [
                local for local, comb in remaps[file_idx].items() if comb == tau_name
            ]
            assert len(local_tau) == 1
            opt_pos = plan.opt_param_names.index(local_tau[0])
            assert plan_gathers[file_idx][opt_pos] == tau_slot


#
def _make_shared_tau_project(*, spec_fun_str, grids=None, show_output=0):
    """Build a ready-to-fit 2-file project with shared tau, per-file A.

    Simulates noiseless data from two truth files (differing amplitudes,
    identical tau) and assembles fit files via the real workflow.
    ``grids`` optionally gives per-file ``(energy, time_ax)`` pairs for
    heterogeneous-grid tests. ``show_output`` is applied after setup so
    baseline fits stay silent.
    """

    if grids is None:
        grids = [(None, None), (None, None)]
    amplitudes = [20.0, 14.0]
    seeds = [42, 43]

    project = make_project(
        name=f"jax_project_{spec_fun_str}",
        spec_fun_str=spec_fun_str,
    )
    for i, ((energy, time_ax), amplitude, seed) in enumerate(
        zip(grids, amplitudes, seeds, strict=True)
    ):
        truth = _make_truth_file(amplitude=amplitude, energy=energy, time_ax=time_ax)
        data = simulate_clean(truth.model_active, seed=seed)
        _make_fit_file(project, data, truth.energy, truth.time, name=f"file_{i}")
    project.show_output = show_output
    return project


#
#
class TestProjectFitJax:
    """Project.fit_2d dispatch: fused JAX backend and interpreter fallback.

    The fallback tests monkeypatch the gate/factory seams because the
    JAX slice currently covers the full lowered 2D surface — no public
    YAML construct builds a 2D model that fails ``can_lower_jax_2d``.
    They run without jax installed; only the tests that execute the
    fused path importorskip.
    """

    TRUE_TAU = 5.0

    #
    def test_jax_parity_with_interpreter(self, capsys):
        """Both backends converge to the same parameters on clean data."""

        pytest.importorskip("jax")

        results = {}
        for spec_fun_str in ("fit_model_gir", "fit_model_jax"):
            project = _make_shared_tau_project(spec_fun_str=spec_fun_str, show_output=1)
            # exercise per-file fit windows on one file
            project.files[0].e_lim = [2, 28]
            project.files[0].t_lim = [1, 23]
            project.fit_2d(model_name="project_glp", stages=2, try_ci=0)

            backend = "JAX" if spec_fun_str == "fit_model_jax" else "interpreter"
            assert f"({backend} backend)" in capsys.readouterr().out

            per_file = []
            for f in project.files:
                m = f.select_model("project_glp")
                assert m is not None  # type guard
                per_file.append({n: m.lmfit_pars[n].value for n in m.parameter_names})
            results[spec_fun_str] = per_file

        for file_idx, (pars_gir, pars_jax) in enumerate(
            zip(results["fit_model_gir"], results["fit_model_jax"], strict=True)
        ):
            for name, value in pars_gir.items():
                assert np.isclose(pars_jax[name], value, rtol=1e-6, atol=1e-9), (
                    f"file {file_idx} par {name}: gir={value!r} jax={pars_jax[name]!r}"
                )

        tau_jax = results["fit_model_jax"][0]["GLP_01_x0_expFun_01_tau"]
        assert np.isclose(tau_jax, self.TRUE_TAU, atol=0.01)

    #
    def test_heterogeneous_grids_fuse(self, capsys):
        """Files with different energy/time grids fit on the fused path."""

        pytest.importorskip("jax")

        grids = [
            (np.linspace(83, 87, 30), np.linspace(-2, 10, 24)),
            (np.linspace(83.2, 86.8, 37), np.linspace(-1.5, 9, 19)),
        ]
        project = _make_shared_tau_project(
            spec_fun_str="fit_model_jax", grids=grids, show_output=1
        )
        project.fit_2d(model_name="project_glp", stages=2, try_ci=0)
        assert "(JAX backend)" in capsys.readouterr().out

        m0 = project.files[0].select_model("project_glp")
        m1 = project.files[1].select_model("project_glp")
        assert m0 is not None  # type guard
        assert m1 is not None  # type guard
        tau_0 = m0.lmfit_pars["GLP_01_x0_expFun_01_tau"].value
        tau_1 = m1.lmfit_pars["GLP_01_x0_expFun_01_tau"].value
        assert tau_0 == tau_1  # project-shared
        assert np.isclose(tau_0, self.TRUE_TAU, atol=0.05)
        A_0 = m0.lmfit_pars["GLP_01_A"].value
        A_1 = m1.lmfit_pars["GLP_01_A"].value
        assert np.isclose(A_0, 20.0, atol=0.1)
        assert np.isclose(A_1, 14.0, atol=0.1)

    #
    def test_fallback_when_one_file_not_jax_lowerable(self, monkeypatch, capsys):
        """One file failing the JAX gate sends the whole project to MCP."""

        from trspecfit import graph_ir

        project = _make_shared_tau_project(spec_fun_str="fit_model_jax", show_output=1)

        # First file passes the real gate, second is rejected.
        real_gate = graph_ir.can_lower_jax_2d
        gate_calls: list[bool] = []

        def fail_from_second_call(graph):
            gate_calls.append(True)
            if len(gate_calls) >= 2:
                return False
            return real_gate(graph)

        monkeypatch.setattr(
            "trspecfit.graph_ir.can_lower_jax_2d", fail_from_second_call
        )

        project.fit_2d(model_name="project_glp", stages=1, try_ci=0)
        assert "(interpreter backend)" in capsys.readouterr().out
        assert len(gate_calls) >= 2

        # The interpreter path still recovers the shared tau.
        m = project.files[0].select_model("project_glp")
        assert m is not None  # type guard
        tau_fit = m.lmfit_pars["GLP_01_x0_expFun_01_tau"].value
        assert np.isclose(tau_fit, self.TRUE_TAU, atol=0.05)

    #
    def test_fallback_when_jax_unavailable(self, monkeypatch, capsys):
        """Factory raising ImportError (jax missing) falls back to MCP."""

        def raise_import_error(*args, **kwargs):
            raise ImportError("jax is not installed")

        project = _make_shared_tau_project(spec_fun_str="fit_model_jax", show_output=1)
        monkeypatch.setattr(
            "trspecfit.eval_jax.make_project_evaluator_2d_jax", raise_import_error
        )

        project.fit_2d(model_name="project_glp", stages=1, try_ci=0)
        assert "(interpreter backend)" in capsys.readouterr().out
