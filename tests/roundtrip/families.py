"""Family registry for the roundtrip test matrix.

Each family declares how to build a *truth* file (used to simulate clean
data) and a *fit* file (the empty file the workflow runner fits against
the simulated data). Builders are lazy: nothing is constructed at import
time. Each family also exposes ``model_name(variant)`` so the workflow
runner can pass the right ``model_name`` to ``fit_*`` even for families
whose YAML key depends on the variant (e.g. F2's expression variants).

The two builders share a private recipe helper that applies the model,
profiles, and dynamics in the same order on both files. Truth/fit
asymmetry (``define_baseline`` is fit-side only, etc.) is handled by the
workflow runner, not by the builders.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from trspecfit import File

_ENERGY_YAML = "models/file_energy.yaml"
_TIME_YAML = "models/file_time.yaml"
_PROFILE_YAML = "models/file_profile.yaml"


#
#
@dataclass(frozen=True)
class Family:
    """Static metadata + lazy builders for one model family."""

    id: str
    description: str
    model_name: Callable[[str], str]  # variant -> YAML model key
    needs_aux: bool
    needs_time: bool
    single_cycle_only: bool
    build_truth: Callable[..., File]
    build_fit: Callable[..., File]
    add_dynamics: Callable[[File, str], None] | None = None
    variants: tuple[str, ...] = field(default_factory=lambda: ("default",))


# ---- axis defaults ----


#
def _energy_axis_glp() -> np.ndarray:
    return np.linspace(83, 87, 30)


#
def _energy_axis_two_glp() -> np.ndarray:
    return np.linspace(82, 92, 30)


#
def _energy_axis_gauss() -> np.ndarray:
    return np.linspace(81, 89, 50)


#
def _time_axis() -> np.ndarray:
    return np.linspace(-2, 10, 24)


#
def _aux_axis() -> np.ndarray:
    return np.linspace(0, 8, 20)


# ---- F1: Plain energy (single GLP, no profile, no dynamics) ----


#
def _f1_apply(file: File) -> None:
    file.load_model(model_yaml=_ENERGY_YAML, model_info="single_glp")


#
def _f1_truth(project, *, variant: str = "default") -> File:
    file = File(
        parent_project=project,
        name="truth",
        energy=_energy_axis_glp(),
        time=_time_axis(),
    )
    _f1_apply(file)
    return file


#
def _f1_fit(project, *, data, energy, time, aux=None, variant: str = "default") -> File:
    file = File(
        parent_project=project,
        name="fit",
        data=data,
        energy=energy.copy(),
        time=time.copy(),
    )
    _f1_apply(file)
    file.define_baseline(time_start=0, time_stop=3, time_type="ind", show_plot=False)
    return file


# ---- F2: Static expressions on a plain energy model ----


_F2_VARIANT_TO_KEY = {
    "direct": "energy_expression",
    "fan_out": "expression_fan_out",
    "forward_ref": "energy_expression_forward_reference",
}


#
def _f2_apply(file: File, variant: str) -> None:
    file.load_model(model_yaml=_ENERGY_YAML, model_info=_F2_VARIANT_TO_KEY[variant])


#
def _f2_truth(project, *, variant: str) -> File:
    file = File(
        parent_project=project,
        name="truth",
        energy=_energy_axis_two_glp(),
        time=_time_axis(),
    )
    _f2_apply(file, variant)
    return file


#
def _f2_fit(project, *, data, energy, time, aux=None, variant: str) -> File:
    file = File(
        parent_project=project,
        name="fit",
        data=data,
        energy=energy.copy(),
        time=time.copy(),
    )
    _f2_apply(file, variant)
    file.define_baseline(time_start=0, time_stop=3, time_type="ind", show_plot=False)
    return file


# ---- F3: Top-level standard dynamics ----


#
def _f3_apply(file: File) -> None:
    file.load_model(model_yaml=_ENERGY_YAML, model_info="single_glp")


#
def _f3_truth(project, *, variant: str = "default") -> File:
    file = File(
        parent_project=project,
        name="truth",
        energy=_energy_axis_glp(),
        time=_time_axis(),
    )
    _f3_apply(file)
    file.add_time_dependence(
        target_model="single_glp",
        target_parameter="GLP_01_A",
        dynamics_yaml=_TIME_YAML,
        dynamics_model=["MonoExpPos"],
    )
    return file


#
def _f3_fit(project, *, data, energy, time, aux=None, variant: str = "default") -> File:
    file = File(
        parent_project=project,
        name="fit",
        data=data,
        energy=energy.copy(),
        time=time.copy(),
    )
    _f3_apply(file)
    file.define_baseline(time_start=0, time_stop=3, time_type="ind", show_plot=False)
    return file


#
def _f3_add_dynamics(file: File, variant: str) -> None:
    file.add_time_dependence(
        target_model="single_glp",
        target_parameter="GLP_01_A",
        dynamics_yaml=_TIME_YAML,
        dynamics_model=["MonoExpPos"],
    )


# ---- F4: IRF dynamics ----


#
def _f4_truth(project, *, variant: str = "default") -> File:
    file = File(
        parent_project=project,
        name="truth",
        energy=_energy_axis_glp(),
        time=_time_axis(),
    )
    file.load_model(model_yaml=_ENERGY_YAML, model_info="single_glp")
    file.add_time_dependence(
        target_model="single_glp",
        target_parameter="GLP_01_A",
        dynamics_yaml=_TIME_YAML,
        dynamics_model=["MonoExpPosIRF"],
    )
    return file


#
def _f4_fit(project, *, data, energy, time, aux=None, variant: str = "default") -> File:
    file = File(
        parent_project=project,
        name="fit",
        data=data,
        energy=energy.copy(),
        time=time.copy(),
    )
    file.load_model(model_yaml=_ENERGY_YAML, model_info="single_glp")
    file.define_baseline(time_start=0, time_stop=3, time_type="ind", show_plot=False)
    return file


#
def _f4_add_dynamics(file: File, variant: str) -> None:
    file.add_time_dependence(
        target_model="single_glp",
        target_parameter="GLP_01_A",
        dynamics_yaml=_TIME_YAML,
        dynamics_model=["MonoExpPosIRF"],
    )


# ---- F5: Subcycle dynamics ----


_F5_DYNAMICS = ["ModelNone", "MonoExpNeg", "MonoExpPosExpr"]
_F5_FREQ = 10.0


#
def _f5_truth(project, *, variant: str = "default") -> File:
    file = File(
        parent_project=project,
        name="truth",
        energy=_energy_axis_glp(),
        time=_time_axis(),
    )
    file.load_model(model_yaml=_ENERGY_YAML, model_info="single_glp")
    file.add_time_dependence(
        target_model="single_glp",
        target_parameter="GLP_01_A",
        dynamics_yaml=_TIME_YAML,
        dynamics_model=_F5_DYNAMICS,
        frequency=_F5_FREQ,
    )
    return file


#
def _f5_fit(project, *, data, energy, time, aux=None, variant: str = "default") -> File:
    file = File(
        parent_project=project,
        name="fit",
        data=data,
        energy=energy.copy(),
        time=time.copy(),
    )
    file.load_model(model_yaml=_ENERGY_YAML, model_info="single_glp")
    file.define_baseline(time_start=0, time_stop=3, time_type="ind", show_plot=False)
    return file


#
def _f5_add_dynamics(file: File, variant: str) -> None:
    file.add_time_dependence(
        target_model="single_glp",
        target_parameter="GLP_01_A",
        dynamics_yaml=_TIME_YAML,
        dynamics_model=_F5_DYNAMICS,
        frequency=_F5_FREQ,
    )


# ---- F6: Top-level profile only (two profiles on single_gauss) ----


#
def _f6_apply(file: File) -> None:
    file.load_model(model_yaml=_ENERGY_YAML, model_info="single_gauss")
    file.add_par_profile(
        target_model="single_gauss",
        target_parameter="Gauss_01_x0",
        profile_yaml=_PROFILE_YAML,
        profile_model=["roundtrip_pLinear_x0"],
    )
    file.add_par_profile(
        target_model="single_gauss",
        target_parameter="Gauss_01_A",
        profile_yaml=_PROFILE_YAML,
        profile_model=["roundtrip_pExpDecay_A"],
    )


#
def _f6_truth(project, *, variant: str = "default") -> File:
    file = File(
        parent_project=project,
        name="truth",
        energy=_energy_axis_gauss(),
        time=_time_axis(),
        aux_axis=_aux_axis(),
    )
    _f6_apply(file)
    return file


#
def _f6_fit(project, *, data, energy, time, aux, variant: str = "default") -> File:
    file = File(
        parent_project=project,
        name="fit",
        data=data,
        energy=energy.copy(),
        time=time.copy(),
        aux_axis=aux.copy(),
    )
    _f6_apply(file)
    file.define_baseline(time_start=0, time_stop=3, time_type="ind", show_plot=False)
    return file


# ---- F7: Profile + separate dynamics on different params ----


#
def _f7_apply(file: File) -> None:
    file.load_model(model_yaml=_ENERGY_YAML, model_info="single_gauss")
    file.add_par_profile(
        target_model="single_gauss",
        target_parameter="Gauss_01_A",
        profile_yaml=_PROFILE_YAML,
        profile_model=["roundtrip_pExpDecay_A"],
    )


#
def _f7_truth(project, *, variant: str = "default") -> File:
    file = File(
        parent_project=project,
        name="truth",
        energy=_energy_axis_gauss(),
        time=_time_axis(),
        aux_axis=_aux_axis(),
    )
    _f7_apply(file)
    file.add_time_dependence(
        target_model="single_gauss",
        target_parameter="Gauss_01_x0",
        dynamics_yaml=_TIME_YAML,
        dynamics_model=["MonoExpPos"],
    )
    return file


#
def _f7_fit(project, *, data, energy, time, aux, variant: str = "default") -> File:
    file = File(
        parent_project=project,
        name="fit",
        data=data,
        energy=energy.copy(),
        time=time.copy(),
        aux_axis=aux.copy(),
    )
    _f7_apply(file)
    file.define_baseline(time_start=0, time_stop=3, time_type="ind", show_plot=False)
    return file


#
def _f7_add_dynamics(file: File, variant: str) -> None:
    file.add_time_dependence(
        target_model="single_gauss",
        target_parameter="Gauss_01_x0",
        dynamics_yaml=_TIME_YAML,
        dynamics_model=["MonoExpPos"],
    )


# ---- F8: Profile-internal dynamics ----


#
def _f8_apply(file: File) -> None:
    file.load_model(model_yaml=_ENERGY_YAML, model_info="single_gauss")
    file.add_par_profile(
        target_model="single_gauss",
        target_parameter="Gauss_01_x0",
        profile_yaml=_PROFILE_YAML,
        profile_model=["roundtrip_pLinear_x0"],
    )
    file.add_par_profile(
        target_model="single_gauss",
        target_parameter="Gauss_01_A",
        profile_yaml=_PROFILE_YAML,
        profile_model=["roundtrip_pExpDecay_A"],
    )


#
def _f8_truth(project, *, variant: str = "default") -> File:
    file = File(
        parent_project=project,
        name="truth",
        energy=_energy_axis_gauss(),
        time=_time_axis(),
        aux_axis=_aux_axis(),
    )
    _f8_apply(file)
    file.add_time_dependence(
        target_model="single_gauss",
        target_parameter="Gauss_01_A_pExpDecay_01_A",
        dynamics_yaml=_TIME_YAML,
        dynamics_model=["MonoExpPosStrong"],
    )
    return file


#
def _f8_fit(project, *, data, energy, time, aux, variant: str = "default") -> File:
    file = File(
        parent_project=project,
        name="fit",
        data=data,
        energy=energy.copy(),
        time=time.copy(),
        aux_axis=aux.copy(),
    )
    _f8_apply(file)
    file.define_baseline(time_start=0, time_stop=3, time_type="ind", show_plot=False)
    return file


#
def _f8_add_dynamics(file: File, variant: str) -> None:
    file.add_time_dependence(
        target_model="single_gauss",
        target_parameter="Gauss_01_A_pExpDecay_01_A",
        dynamics_yaml=_TIME_YAML,
        dynamics_model=["MonoExpPosStrong"],
    )


# ---- F9: Expression -> time-dependent base parameter ----


#
def _f9_apply(file: File) -> None:
    file.load_model(model_yaml=_ENERGY_YAML, model_info="two_glp_expr_amplitude")


#
def _f9_truth(project, *, variant: str = "default") -> File:
    file = File(
        parent_project=project,
        name="truth",
        energy=_energy_axis_two_glp(),
        time=_time_axis(),
    )
    _f9_apply(file)
    file.add_time_dependence(
        target_model="two_glp_expr_amplitude",
        target_parameter="GLP_01_A",
        dynamics_yaml=_TIME_YAML,
        dynamics_model=["MonoExpPos"],
    )
    return file


#
def _f9_fit(project, *, data, energy, time, aux=None, variant: str = "default") -> File:
    file = File(
        parent_project=project,
        name="fit",
        data=data,
        energy=energy.copy(),
        time=time.copy(),
    )
    _f9_apply(file)
    file.define_baseline(time_start=0, time_stop=3, time_type="ind", show_plot=False)
    return file


#
def _f9_add_dynamics(file: File, variant: str) -> None:
    file.add_time_dependence(
        target_model="two_glp_expr_amplitude",
        target_parameter="GLP_01_A",
        dynamics_yaml=_TIME_YAML,
        dynamics_model=["MonoExpPos"],
    )


# ---- F10: Expression -> profiled base parameter ----


#
def _f10_apply(file: File) -> None:
    file.load_model(model_yaml=_ENERGY_YAML, model_info="two_glp_expr_amplitude")
    file.add_par_profile(
        target_model="two_glp_expr_amplitude",
        target_parameter="GLP_01_A",
        profile_yaml=_PROFILE_YAML,
        profile_model=["roundtrip_pExpDecay_A"],
    )


#
def _f10_truth(project, *, variant: str = "default") -> File:
    file = File(
        parent_project=project,
        name="truth",
        energy=_energy_axis_two_glp(),
        time=_time_axis(),
        aux_axis=_aux_axis(),
    )
    _f10_apply(file)
    return file


#
def _f10_fit(project, *, data, energy, time, aux, variant: str = "default") -> File:
    file = File(
        parent_project=project,
        name="fit",
        data=data,
        energy=energy.copy(),
        time=time.copy(),
        aux_axis=aux.copy(),
    )
    _f10_apply(file)
    file.define_baseline(time_start=0, time_stop=3, time_type="ind", show_plot=False)
    return file


# ---- F11: Expression -> profiled base par with profile-internal dynamics ----


#
def _f11_apply(file: File) -> None:
    file.load_model(model_yaml=_ENERGY_YAML, model_info="two_glp_expr_amplitude")
    file.add_par_profile(
        target_model="two_glp_expr_amplitude",
        target_parameter="GLP_01_A",
        profile_yaml=_PROFILE_YAML,
        profile_model=["roundtrip_pExpDecay_A"],
    )


#
def _f11_truth(project, *, variant: str = "default") -> File:
    file = File(
        parent_project=project,
        name="truth",
        energy=_energy_axis_two_glp(),
        time=_time_axis(),
        aux_axis=_aux_axis(),
    )
    _f11_apply(file)
    file.add_time_dependence(
        target_model="two_glp_expr_amplitude",
        target_parameter="GLP_01_A_pExpDecay_01_A",
        dynamics_yaml=_TIME_YAML,
        dynamics_model=["MonoExpPosStrong"],
    )
    return file


#
def _f11_fit(project, *, data, energy, time, aux, variant: str = "default") -> File:
    file = File(
        parent_project=project,
        name="fit",
        data=data,
        energy=energy.copy(),
        time=time.copy(),
        aux_axis=aux.copy(),
    )
    _f11_apply(file)
    file.define_baseline(time_start=0, time_stop=3, time_type="ind", show_plot=False)
    return file


#
def _f11_add_dynamics(file: File, variant: str) -> None:
    file.add_time_dependence(
        target_model="two_glp_expr_amplitude",
        target_parameter="GLP_01_A_pExpDecay_01_A",
        dynamics_yaml=_TIME_YAML,
        dynamics_model=["MonoExpPosStrong"],
    )


# ---- F12: Mixed expression referencing both profiled and time-dep base pars ----


#
def _f12_apply(file: File) -> None:
    file.load_model(
        model_yaml=_ENERGY_YAML, model_info="two_glp_mixed_profile_dynamics"
    )
    file.add_par_profile(
        target_model="two_glp_mixed_profile_dynamics",
        target_parameter="GLP_01_A",
        profile_yaml=_PROFILE_YAML,
        profile_model=["roundtrip_pExpDecay_A"],
    )


#
def _f12_truth(project, *, variant: str = "default") -> File:
    file = File(
        parent_project=project,
        name="truth",
        energy=_energy_axis_two_glp(),
        time=_time_axis(),
        aux_axis=_aux_axis(),
    )
    _f12_apply(file)
    file.add_time_dependence(
        target_model="two_glp_mixed_profile_dynamics",
        target_parameter="GLP_01_x0",
        dynamics_yaml=_TIME_YAML,
        dynamics_model=["MonoExpPos"],
    )
    return file


#
def _f12_fit(project, *, data, energy, time, aux, variant: str = "default") -> File:
    file = File(
        parent_project=project,
        name="fit",
        data=data,
        energy=energy.copy(),
        time=time.copy(),
        aux_axis=aux.copy(),
    )
    _f12_apply(file)
    file.define_baseline(time_start=0, time_stop=3, time_type="ind", show_plot=False)
    return file


#
def _f12_add_dynamics(file: File, variant: str) -> None:
    file.add_time_dependence(
        target_model="two_glp_mixed_profile_dynamics",
        target_parameter="GLP_01_x0",
        dynamics_yaml=_TIME_YAML,
        dynamics_model=["MonoExpPos"],
    )


# ---- registry ----


FAMILIES: dict[str, Family] = {
    "F1": Family(
        id="F1",
        description="Plain energy (single_glp, no profile, no dynamics)",
        model_name=lambda v: "single_glp",
        needs_aux=False,
        needs_time=True,
        single_cycle_only=False,
        build_truth=_f1_truth,
        build_fit=_f1_fit,
    ),
    "F2": Family(
        id="F2",
        description="Static energy expressions (direct/fan_out/forward_ref)",
        model_name=lambda v: _F2_VARIANT_TO_KEY[v],
        needs_aux=False,
        needs_time=True,
        single_cycle_only=False,
        build_truth=_f2_truth,
        build_fit=_f2_fit,
        variants=("direct", "fan_out", "forward_ref"),
    ),
    "F3": Family(
        id="F3",
        description="Top-level standard dynamics (single_glp + MonoExpPos on GLP_01_A)",
        model_name=lambda v: "single_glp",
        needs_aux=False,
        needs_time=True,
        single_cycle_only=False,
        build_truth=_f3_truth,
        build_fit=_f3_fit,
        add_dynamics=_f3_add_dynamics,
    ),
    "F4": Family(
        id="F4",
        description="Top-level IRF dynamics (single_glp + MonoExpPosIRF on GLP_01_A)",
        model_name=lambda v: "single_glp",
        needs_aux=False,
        needs_time=True,
        single_cycle_only=False,
        build_truth=_f4_truth,
        build_fit=_f4_fit,
        add_dynamics=_f4_add_dynamics,
    ),
    "F5": Family(
        id="F5",
        description=(
            "Subcycle dynamics "
            "(single_glp + ModelNone/MonoExpNeg/MonoExpPosExpr, freq=10)"
        ),
        model_name=lambda v: "single_glp",
        needs_aux=False,
        needs_time=True,
        single_cycle_only=False,
        build_truth=_f5_truth,
        build_fit=_f5_fit,
        add_dynamics=_f5_add_dynamics,
    ),
    "F6": Family(
        id="F6",
        description=(
            "Top-level profile only (single_gauss + pLinear on x0 + pExpDecay on A)"
        ),
        model_name=lambda v: "single_gauss",
        needs_aux=True,
        needs_time=True,
        single_cycle_only=False,
        build_truth=_f6_truth,
        build_fit=_f6_fit,
    ),
    "F7": Family(
        id="F7",
        description="Profile + separate dynamics (pExpDecay on A, MonoExpPos on x0)",
        model_name=lambda v: "single_gauss",
        needs_aux=True,
        needs_time=True,
        single_cycle_only=False,
        build_truth=_f7_truth,
        build_fit=_f7_fit,
        add_dynamics=_f7_add_dynamics,
    ),
    "F8": Family(
        id="F8",
        description=(
            "Profile-internal dynamics (two profiles + MonoExpPosStrong on pExpDecay A)"
        ),
        model_name=lambda v: "single_gauss",
        needs_aux=True,
        needs_time=True,
        single_cycle_only=True,
        build_truth=_f8_truth,
        build_fit=_f8_fit,
        add_dynamics=_f8_add_dynamics,
    ),
    "F9": Family(
        id="F9",
        description=(
            "Expr -> time-dep base par (two_glp_expr_amplitude + dyn on GLP_01_A)"
        ),
        model_name=lambda v: "two_glp_expr_amplitude",
        needs_aux=False,
        needs_time=True,
        single_cycle_only=False,
        build_truth=_f9_truth,
        build_fit=_f9_fit,
        add_dynamics=_f9_add_dynamics,
    ),
    "F10": Family(
        id="F10",
        description=(
            "Expr -> profiled base par (two_glp_expr_amplitude + profile on GLP_01_A)"
        ),
        model_name=lambda v: "two_glp_expr_amplitude",
        needs_aux=True,
        needs_time=True,
        single_cycle_only=False,
        build_truth=_f10_truth,
        build_fit=_f10_fit,
    ),
    "F11": Family(
        id="F11",
        description="Expr -> profiled base par with profile-internal dynamics",
        model_name=lambda v: "two_glp_expr_amplitude",
        needs_aux=True,
        needs_time=True,
        single_cycle_only=True,
        build_truth=_f11_truth,
        build_fit=_f11_fit,
        add_dynamics=_f11_add_dynamics,
    ),
    "F12": Family(
        id="F12",
        description="Mixed expr (profile on GLP_01_A + dynamics on GLP_01_x0)",
        model_name=lambda v: "two_glp_mixed_profile_dynamics",
        needs_aux=True,
        needs_time=True,
        single_cycle_only=False,
        build_truth=_f12_truth,
        build_fit=_f12_fit,
        add_dynamics=_f12_add_dynamics,
    ),
}
