"""Project-level roundtrip matrix.

Project-level fitting uses ``Project.fit_2d`` which is wired through
``fit_project_mcp``.  Per the matrix doc, PF cells are M-only today; if
project-level GIR ever lands, the ``backends`` field on each ``PFCell``
upgrades to ``("M", "G", "C")`` with no other shape change.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from _utils import make_project, simulate_clean

from trspecfit import File

pytestmark = pytest.mark.slow


_PROJECT_ENERGY_YAML = "models/project_energy.yaml"
_PROJECT_TIME_YAML = "models/project_time.yaml"


# ---- PF1: shared plain dynamics ----


#
def _build_pf1_truth(*, amplitude: float, x0_shift: float, tau: float):
    project = make_project(name="pf1_truth")
    file = File(
        parent_project=project,
        energy=np.linspace(83, 87, 30),
        time=np.linspace(-2, 10, 24),
    )
    file.load_model(model_yaml=_PROJECT_ENERGY_YAML, model_info="project_glp")
    file.add_time_dependence(
        target_model="project_glp",
        target_parameter="GLP_01_x0",
        dynamics_yaml=_PROJECT_TIME_YAML,
        dynamics_model=["MonoExpProject"],
    )
    model = file.model_active
    assert model is not None
    model.lmfit_pars["GLP_01_A"].value = amplitude
    model.lmfit_pars["GLP_01_x0"].value = 85.0
    model.lmfit_pars["GLP_01_F"].value = 1.0
    model.lmfit_pars["GLP_01_m"].value = 0.3
    model.lmfit_pars["GLP_01_x0_expFun_01_A"].value = x0_shift
    model.lmfit_pars["GLP_01_x0_expFun_01_tau"].value = tau
    return file


#
def _build_pf1_fit(project, data, energy, time, *, name: str) -> File:
    file = File(
        parent_project=project,
        name=name,
        data=data,
        energy=energy.copy(),
        time=time.copy(),
    )
    file.load_model(model_yaml=_PROJECT_ENERGY_YAML, model_info="project_glp_base")
    file.define_baseline(time_start=0, time_stop=3, time_type="ind", show_plot=False)
    file.fit_baseline(model_name="project_glp_base", stages=2, try_ci=0)
    file.load_model(model_yaml=_PROJECT_ENERGY_YAML, model_info="project_glp")
    file.add_time_dependence(
        target_model="project_glp",
        target_parameter="GLP_01_x0",
        dynamics_yaml=_PROJECT_TIME_YAML,
        dynamics_model=["MonoExpProject"],
    )
    return file


#
def _run_pf1():
    # Different amplitude AND different x0_shift per file exercise the
    # file-vary semantics; identical tau exercises the project-vary semantics.
    truth_a = _build_pf1_truth(amplitude=20.0, x0_shift=3.0, tau=5.0)
    truth_b = _build_pf1_truth(amplitude=15.0, x0_shift=2.0, tau=5.0)
    data_a = simulate_clean(truth_a.model_active, seed=42)
    data_b = simulate_clean(truth_b.model_active, seed=43)

    project = make_project(name="pf1_fit")
    fit_a = _build_pf1_fit(project, data_a, truth_a.energy, truth_a.time, name="file_a")
    fit_b = _build_pf1_fit(project, data_b, truth_b.energy, truth_b.time, name="file_b")
    project.fit_2d(model_name="project_glp", stages=2, try_ci=0)

    cases = (
        (fit_a, 20.0, 3.0),
        (fit_b, 15.0, 2.0),
    )
    models = []
    for fit_file, truth_amp, truth_shift in cases:
        model = fit_file.select_model("project_glp")
        assert model is not None
        assert np.isclose(model.lmfit_pars["GLP_01_A"].value, truth_amp, rtol=1e-3)
        assert np.isclose(
            model.lmfit_pars["GLP_01_x0_expFun_01_A"].value, truth_shift, rtol=1e-3
        )
        assert np.isclose(
            model.lmfit_pars["GLP_01_x0_expFun_01_tau"].value, 5.0, rtol=1e-3
        )
        models.append(model)

    # Project-vary invariant: tau is shared exactly across files.
    tau_a = models[0].lmfit_pars["GLP_01_x0_expFun_01_tau"].value
    tau_b = models[1].lmfit_pars["GLP_01_x0_expFun_01_tau"].value
    assert tau_a == tau_b, f"tau should be project-shared: {tau_a} != {tau_b}"


# ---- PF2: project-level expressions ----


#
def _build_pf2_truth(*, amplitude: float, tau: float):
    project = make_project(name="pf2_truth")
    file = File(
        parent_project=project,
        energy=np.linspace(83, 90, 40),
        time=np.linspace(-2, 10, 24),
    )
    file.load_model(model_yaml=_PROJECT_ENERGY_YAML, model_info="project_glp_expr")
    file.add_time_dependence(
        target_model="project_glp_expr",
        target_parameter="GLP_01_x0",
        dynamics_yaml=_PROJECT_TIME_YAML,
        dynamics_model=["MonoExpProject"],
    )
    model = file.model_active
    assert model is not None
    model.lmfit_pars["GLP_01_A"].value = amplitude
    model.lmfit_pars["GLP_01_x0"].value = 85.0
    model.lmfit_pars["GLP_01_F"].value = 1.0
    model.lmfit_pars["GLP_01_m"].value = 0.3
    model.lmfit_pars["GLP_02_x0"].value = 86.5
    model.lmfit_pars["GLP_02_F"].value = 1.0
    model.lmfit_pars["GLP_02_m"].value = 0.3
    model.lmfit_pars["GLP_01_x0_expFun_01_A"].value = 3.0
    model.lmfit_pars["GLP_01_x0_expFun_01_tau"].value = tau
    return file


#
def _build_pf2_fit(project, data, energy, time, *, name: str) -> File:
    file = File(
        parent_project=project,
        name=name,
        data=data,
        energy=energy.copy(),
        time=time.copy(),
    )
    file.load_model(model_yaml=_PROJECT_ENERGY_YAML, model_info="project_glp_expr_base")
    file.define_baseline(time_start=0, time_stop=3, time_type="ind", show_plot=False)
    file.fit_baseline(model_name="project_glp_expr_base", stages=2, try_ci=0)
    file.load_model(model_yaml=_PROJECT_ENERGY_YAML, model_info="project_glp_expr")
    file.add_time_dependence(
        target_model="project_glp_expr",
        target_parameter="GLP_01_x0",
        dynamics_yaml=_PROJECT_TIME_YAML,
        dynamics_model=["MonoExpProject"],
    )
    return file


#
def _run_pf2():
    truth_a = _build_pf2_truth(amplitude=20.0, tau=4.0)
    truth_b = _build_pf2_truth(amplitude=14.0, tau=4.0)
    data_a = simulate_clean(truth_a.model_active)
    data_b = simulate_clean(truth_b.model_active)

    project = make_project(name="pf2_fit")
    fit_a = _build_pf2_fit(project, data_a, truth_a.energy, truth_a.time, name="file_a")
    fit_b = _build_pf2_fit(project, data_b, truth_b.energy, truth_b.time, name="file_b")
    project.fit_2d(model_name="project_glp_expr", stages=2, try_ci=0)

    for fit_file, truth_amp in ((fit_a, 20.0), (fit_b, 14.0)):
        model = fit_file.select_model("project_glp_expr")
        assert model is not None
        a1 = model.lmfit_pars["GLP_01_A"].value
        a2 = model.lmfit_pars["GLP_02_A"].value
        assert np.isclose(a1, truth_amp, rtol=1e-3)
        assert np.isclose(a2, 0.5 * a1, rtol=1e-6), (
            "expression GLP_02_A = GLP_01_A * 0.5 must hold"
        )
        assert np.isclose(
            model.lmfit_pars["GLP_01_x0_expFun_01_tau"].value, 4.0, rtol=1e-3
        )


# ---- PF3: shared dynamics with IRF (BiExpProject — bi-exponential + gaussCONV) ----


#
def _build_pf3_truth(*, amplitude: float, t0: float, tau1: float, tau2: float):
    project = make_project(name="pf3_truth")
    file = File(
        parent_project=project,
        energy=np.linspace(80, 90, 40),
        time=np.linspace(-5, 50, 60),
    )
    file.load_model(model_yaml=_PROJECT_ENERGY_YAML, model_info="project_glp")
    file.add_time_dependence(
        target_model="project_glp",
        target_parameter="GLP_01_x0",
        dynamics_yaml=_PROJECT_TIME_YAML,
        dynamics_model=["BiExpProject"],
    )
    model = file.model_active
    assert model is not None
    model.lmfit_pars["GLP_01_A"].value = amplitude
    model.lmfit_pars["GLP_01_x0"].value = 85.0
    model.lmfit_pars["GLP_01_F"].value = 1.0
    model.lmfit_pars["GLP_01_m"].value = 0.3
    model.lmfit_pars["GLP_01_x0_expFun_01_A"].value = 2.0
    model.lmfit_pars["GLP_01_x0_expFun_01_tau"].value = tau1
    model.lmfit_pars["GLP_01_x0_expFun_01_t0"].value = t0
    model.lmfit_pars["GLP_01_x0_expFun_02_A"].value = 1.0
    model.lmfit_pars["GLP_01_x0_expFun_02_tau"].value = tau2
    return file


#
def _build_pf3_fit(project, data, energy, time, *, name: str) -> File:
    file = File(
        parent_project=project,
        name=name,
        data=data,
        energy=energy.copy(),
        time=time.copy(),
    )
    file.load_model(model_yaml=_PROJECT_ENERGY_YAML, model_info="project_glp_base")
    file.define_baseline(time_start=0, time_stop=3, time_type="ind", show_plot=False)
    file.fit_baseline(model_name="project_glp_base", stages=2, try_ci=0)
    file.load_model(model_yaml=_PROJECT_ENERGY_YAML, model_info="project_glp")
    file.add_time_dependence(
        target_model="project_glp",
        target_parameter="GLP_01_x0",
        dynamics_yaml=_PROJECT_TIME_YAML,
        dynamics_model=["BiExpProject"],
    )
    return file


#
def _run_pf3():
    truth_a = _build_pf3_truth(amplitude=20.0, t0=3.0, tau1=2.0, tau2=20.0)
    truth_b = _build_pf3_truth(amplitude=15.0, t0=3.0, tau1=2.0, tau2=20.0)
    data_a = simulate_clean(truth_a.model_active)
    data_b = simulate_clean(truth_b.model_active)

    project = make_project(name="pf3_fit")
    fit_a = _build_pf3_fit(project, data_a, truth_a.energy, truth_a.time, name="file_a")
    fit_b = _build_pf3_fit(project, data_b, truth_b.energy, truth_b.time, name="file_b")
    project.fit_2d(model_name="project_glp", stages=2, try_ci=0)

    for fit_file, truth_amp in ((fit_a, 20.0), (fit_b, 15.0)):
        model = fit_file.select_model("project_glp")
        assert model is not None
        assert np.isclose(model.lmfit_pars["GLP_01_A"].value, truth_amp, rtol=1e-2)
        assert np.isclose(
            model.lmfit_pars["GLP_01_x0_expFun_01_tau"].value, 2.0, rtol=1e-2
        )
        assert np.isclose(
            model.lmfit_pars["GLP_01_x0_expFun_02_tau"].value, 20.0, rtol=1e-2
        )


# ---- registry + parametrize ----


#
#
@dataclass(frozen=True)
class PFCell:
    """One project-level matrix cell."""

    id: str
    description: str
    backends: tuple[str, ...]
    run: Callable[[], None]


PROJECT_MATRIX: list[PFCell] = [
    PFCell("PF1", "Shared plain dynamics", ("M",), _run_pf1),
    PFCell("PF2", "Project-level expressions", ("M",), _run_pf2),
    PFCell(
        "PF3", "Shared dynamics with IRF (BiExpProject + gaussCONV)", ("M",), _run_pf3
    ),
    # PF4 (shared subcycle dynamics) deferred — no project-subcycle fixture yet.
]


#
def _iter_project_cells() -> Iterable[tuple[str, str, Callable[[], None]]]:
    for cell in PROJECT_MATRIX:
        for backend in cell.backends:
            yield (cell.id, backend, cell.run)


_PROJECT_CELLS = list(_iter_project_cells())


#
@pytest.mark.parametrize(
    "pf_id, backend, runner",
    _PROJECT_CELLS,
    ids=[f"{pf_id}-{backend}" for pf_id, backend, _ in _PROJECT_CELLS],
)
def test_project_roundtrip(pf_id, backend, runner):
    """Project-level fits must recover shared and per-file truth parameters."""

    runner()
