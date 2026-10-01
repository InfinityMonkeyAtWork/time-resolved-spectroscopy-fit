"""
Behavioral probes for the ownership contract, rules 1 to 3
(docs/design/api_ownership_contract.md; check 21 of docs/ai/code-review.md).

Reading the code shows what a method does; these probes show what the File
does afterwards: a caller mutates across the ownership boundary, or assigns
an owned attribute, and the File's state is what the contract says.
"""

import numpy as np
import pytest
from _utils import make_project

from trspecfit import File


#
def make_2d_file(*, name: str = "owned") -> tuple[File, dict[str, np.ndarray]]:
    """A 2D File built from caller arrays that the caller keeps and may edit."""

    project = make_project()
    arrays = {
        "data": np.arange(12.0).reshape(3, 4) + 10.0,
        "energy": np.array([80.0, 81.0, 82.0, 83.0]),
        "time": np.array([-1.0, 0.0, 1.0]),
        "aux_axis": np.array([0.0, 0.5]),
    }
    file = File(parent_project=project, name=name, **arrays)
    return file, arrays


#
#
class TestInputsAreOwnedAtConstruction:
    """Rule 1: inputs are copied, frozen, and replaced only by a new File."""

    #
    def test_caller_edits_after_construction_do_not_reach_the_file(self):
        file, arrays = make_2d_file()
        before = {k: v.copy() for k, v in arrays.items()}
        for arr in arrays.values():
            arr[...] = -999.0
        np.testing.assert_array_equal(file.data, before["data"])
        np.testing.assert_array_equal(file.data_raw, before["data"])
        np.testing.assert_array_equal(file.energy, before["energy"])
        np.testing.assert_array_equal(file.time, before["time"])
        np.testing.assert_array_equal(file.aux_axis, before["aux_axis"])

    #
    @pytest.mark.parametrize("attr", ["data", "data_raw", "energy", "time", "aux_axis"])
    def test_handed_out_arrays_are_read_only(self, attr):
        file, _ = make_2d_file()
        with pytest.raises(ValueError, match="read-only"):
            getattr(file, attr)[0] = 0.0

    #
    def test_synthesized_axes_are_read_only_too(self):
        project = make_project()
        file = File(parent_project=project, data=np.ones((2, 3)))
        with pytest.raises(ValueError, match="read-only"):
            file.energy[0] = 5.0
        with pytest.raises(ValueError, match="read-only"):
            file.time[0] = 5.0

    #
    def test_models_share_the_frozen_axes(self):
        """Models hold the file's axes by reference; frozen arrays make that safe."""

        file, _ = make_2d_file()
        file.load_model(
            model_yaml="models/file_energy.yaml", model_info="simple_energy"
        )
        model = file.model_active
        assert model is not None  # type guard
        assert model.energy is file.energy
        with pytest.raises(ValueError, match="read-only"):
            model.energy[0] = 0.0


#
#
class TestOwnedAttributesRefuseAssignment:
    """Rule 8 precedent: refuse at the point of misuse and name the route."""

    #
    @pytest.mark.parametrize(
        ("attr", "route"),
        [
            ("data", "correction methods"),
            ("data_raw", "construct a new File"),
            ("dim", "construct a new File"),
            ("energy", "construct a new File"),
            ("time", "construct a new File"),
            ("aux_axis", "construct a new File"),
            ("dark", "subtract_dark()"),
            ("calibration", "calibrate_data()"),
            ("data_base", "define_baseline()"),
            ("base_t_ind", "define_baseline()"),
            ("base_t_abs", "define_baseline()"),
            ("e_lim", "set_fit_limits()"),
            ("e_lim_abs", "set_fit_limits()"),
            ("t_lim", "set_fit_limits()"),
            ("t_lim_abs", "set_fit_limits()"),
            ("noise", "set_noise()"),
        ],
    )
    def test_assignment_raises_and_names_the_route(self, attr, route):
        file, _ = make_2d_file()
        before = getattr(file, attr)
        with pytest.raises(AttributeError, match=f"File.{attr} is owned") as info:
            setattr(file, attr, None)
        assert route in str(info.value)
        assert getattr(file, attr) is before

    #
    @pytest.mark.parametrize("attr", ["e_lim", "e_lim_abs", "t_lim", "t_lim_abs"])
    def test_fit_windows_cannot_be_edited_in_place(self, attr):
        file, _ = make_2d_file()
        file.set_fit_limits([81.0, 82.0], time_limits=[0.0, 1.0], show_plot=False)
        window = getattr(file, attr)
        assert len(window) == 2
        with pytest.raises(TypeError):
            window[0] = 0  # type: ignore[index]

    #
    def test_baseline_window_cannot_be_edited_in_place(self):
        file, _ = make_2d_file()
        file.define_baseline(-1.0, 0.0, show_plot=False)
        assert file.base_t_ind == (0, 2)
        assert file.base_t_abs == (-1.0, 0.0)
        with pytest.raises(TypeError):
            file.base_t_ind[0] = 1  # type: ignore[index]
        with pytest.raises(ValueError, match="read-only"):
            file.data_base[0] = 0.0


#
#
class TestCorrectionsAreOperationsOnOwnedArrays:
    """Rule 2: the File copies the correction it is given; data is derived."""

    #
    def test_caller_edits_after_subtract_dark_do_not_reach_the_file(self):
        file, arrays = make_2d_file()
        dark = np.array([1.0, 2.0, 3.0, 4.0])
        file.subtract_dark(dark)
        expected = arrays["data"] - dark
        dark[...] = 100.0
        np.testing.assert_array_equal(file.dark, [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(file.data, expected)
        with pytest.raises(ValueError, match="read-only"):
            file.dark[0] = 0.0

    #
    def test_caller_edits_after_calibrate_data_do_not_reach_the_file(self):
        file, arrays = make_2d_file()
        calibration = np.array([1.0, 2.0, 4.0, 8.0])
        file.calibrate_data(calibration)
        expected = arrays["data"] / calibration
        calibration[...] = 1.0
        np.testing.assert_array_equal(file.calibration, [1.0, 2.0, 4.0, 8.0])
        np.testing.assert_array_equal(file.data, expected)

    #
    def test_corrected_data_and_baseline_are_recomputed_read_only(self):
        file, arrays = make_2d_file()
        file.define_baseline(-1.0, 0.0, show_plot=False)
        file.subtract_dark(np.ones(4))
        np.testing.assert_array_equal(
            file.data_base, (arrays["data"] - 1.0)[0:2].mean(axis=0)
        )
        with pytest.raises(ValueError, match="read-only"):
            file.data[0, 0] = 0.0
        np.testing.assert_array_equal(file.data_raw, arrays["data"])


#
def make_model_file(*, model: str = "simple_energy", with_aux: bool = False) -> File:
    """A 2D File with a loaded energy model (and an aux axis for profiles)."""

    project = make_project()
    energy = np.linspace(80, 90, 101)
    time = np.linspace(-5.0, 20.0, 26)
    data = np.random.default_rng(7).normal(size=(time.size, energy.size)) + 5.0
    aux_axis = np.linspace(0.0, 5.0, 6) if with_aux else None
    file = File(
        parent_project=project, data=data, energy=energy, time=time, aux_axis=aux_axis
    )
    file.load_model(model_yaml="models/file_energy.yaml", model_info=model)
    return file


#
def model_state(model) -> tuple:
    """Everything an attachment writes, plus the model's settled evaluation.

    The first evaluation of a model with chained expressions is not the
    settled one (lmfit resolves a chain on the second pass), so evaluate
    twice and keep the second.
    """

    for _ in range(2):
        if model.dim == 2:
            model.create_value_2d()
            value = np.array(model.value_2d)
        else:
            value = np.array(model.create_value_1d(return_1d=1))
    flags = [
        (
            par.name,
            par.t_vary,
            par.t_model,
            par.p_vary,
            par.p_model,
            par.expr_refs_time_dep,
            par.expr_refs_profile_dep,
            tuple(par.expr_refs),
            len(par.lmfit_par_list),
        )
        for par in model.get_all_parameters()
    ]
    return value, list(model.parameter_names), model.get_vary_levels(), flags, model.dim


#
def assert_same_state(before: tuple, after: tuple) -> None:
    np.testing.assert_array_equal(after[0], before[0])
    assert after[1:] == before[1:]


#
def add_dynamics(file: File, *, model: str, parameter: str, dynamics, **kwargs):
    file.add_time_dependence(
        target_model=model,
        target_parameter=parameter,
        dynamics_yaml="models/file_time.yaml",
        dynamics_model=dynamics,
        **kwargs,
    )


#
def add_profile(file: File, *, model: str, parameter: str, profile: str):
    file.add_par_profile(
        target_model=model,
        target_parameter=parameter,
        profile_yaml="models/file_profile.yaml",
        profile_model=[profile],
    )


#
#
class TestReplacementIsAtomic:
    """Rule 4: a failed reload leaves the previous model usable and unchanged."""

    #
    def test_failed_reload_of_the_same_name_keeps_the_old_model(self, tmp_path):
        file = make_model_file()
        old = file.model_active
        assert old is not None  # type guard
        before = model_state(old)
        broken = tmp_path / "broken.yaml"
        broken.write_text(
            "simple_energy:\n    Offset:\n      y0: [2, True, 0, 5]\n"
            "    NoSuchFunction:\n      A: [1, True]\n"
        )
        with pytest.raises(ValueError):
            file.load_model(model_yaml=broken, model_info="simple_energy")
        assert file.models == [old]
        assert file.model_active is old
        assert_same_state(before, model_state(old))
        file.define_baseline(-5.0, 0.0, show_plot=False)
        file.fit_baseline(model_name="simple_energy", stages=1, try_ci=0)

    #
    def test_successful_reload_replaces_and_warns(self):
        file = make_model_file()
        old = file.model_active
        with pytest.warns(UserWarning, match="replacing the live model"):
            file.load_model(
                model_yaml="models/file_energy.yaml", model_info="simple_energy"
            )
        assert len(file.models) == 1
        assert file.model_active is not old

    #
    def test_removing_the_active_model_clears_the_reference(self):
        file = make_model_file()
        file.delete_model()
        assert file.models == []
        assert file.model_active is None
        file.load_model(
            model_yaml="models/file_energy.yaml", model_info="simple_energy"
        )
        file.reset_models()
        assert file.model_active is None


#
#
class TestAttachmentIsAtomic:
    """Rule 4: a rejected attachment leaves both models as they were."""

    #
    def test_unknown_parameter(self):
        file = make_model_file()
        before = model_state(file.model_active)
        with pytest.raises(ValueError, match="not found"):
            add_dynamics(
                file,
                model="simple_energy",
                parameter="GLP_01_nope",
                dynamics=["MonoExpPos"],
            )
        assert_same_state(before, model_state(file.model_active))

    #
    def test_expression_linked_parameter(self):
        file = make_model_file(model="two_glp_expr_amplitude")
        before = model_state(file.model_active)
        with pytest.raises(ValueError, match="expression parameter"):
            add_dynamics(
                file,
                model="two_glp_expr_amplitude",
                parameter="GLP_02_A",
                dynamics=["MonoExpPos"],
            )
        assert_same_state(before, model_state(file.model_active))

    #
    def test_already_attached_parameter(self):
        file = make_model_file()
        add_dynamics(
            file, model="simple_energy", parameter="GLP_01_A", dynamics=["MonoExpPos"]
        )
        before = model_state(file.model_active)
        assert before[4] == 2
        with pytest.raises(ValueError, match="already has time dependence"):
            add_dynamics(
                file,
                model="simple_energy",
                parameter="GLP_01_A",
                dynamics=["MonoExpPos"],
            )
        assert_same_state(before, model_state(file.model_active))

    #
    def test_invalid_frequency_touches_neither_model(self):
        file = make_model_file()
        model = file.model_active
        assert model is not None  # type guard
        before = model_state(model)
        with pytest.raises(ValueError, match="single dynamics model"):
            add_dynamics(
                file,
                model="simple_energy",
                parameter="GLP_01_A",
                dynamics=["MonoExpPos"],
                frequency=0.1,
            )
        assert_same_state(before, model_state(model))
        # the advanced route hands the candidate in: it is untouched too
        dyn = file.load_model(
            "models/file_time.yaml", ["MonoExpPos"], "GLP_01_A", model_type="dynamics"
        )
        with pytest.raises(ValueError, match="single dynamics model"):
            model.add_dynamics(dyn, frequency=0.1)
        assert dyn.parent_model is None
        assert dyn.frequency == -1
        assert dyn.time_norm is None
        assert_same_state(before, model_state(model))

    #
    def test_transitive_chain_is_rolled_back(self):
        """The analysis rewrites every parameter's flags before it rejects."""

        file = make_model_file(model="expression_chain")
        model = file.model_active
        assert model is not None  # type guard
        before = model_state(model)
        dyn = file.load_model(
            "models/file_time.yaml",
            ["MonoExpPosIRF"],
            "GLP_01_A",
            model_type="dynamics",
        )
        with pytest.raises(ValueError, match="indirectly references"):
            model.add_dynamics(dyn)
        assert_same_state(before, model_state(model))
        assert dyn.parent_model is None
        ci, pi = model.find_par_by_name("GLP_01_A")
        target = model.components[ci].pars[pi]
        assert (target.t_vary, target.t_model) == (False, None)
        assert all(not p.expr_refs_time_dep for p in model.get_all_parameters())

    #
    def test_rollback_restores_candidate_timing_after_set_frequency(self):
        """A positive frequency writes the candidate's timing arrays before
        the attachment can be rejected; the rollback puts them back."""

        file = make_model_file(model="expression_chain")
        model = file.model_active
        assert model is not None  # type guard
        before = model_state(model)
        dyn = file.load_model(
            "models/file_time.yaml",
            ["IRF", "MonoExpPos"],  # two entries: a frequency is allowed
            "GLP_01_A",
            model_type="dynamics",
        )
        timing_before = (dyn.frequency, dyn.time_norm, dyn.n_sub, dyn.n_counter)
        components_before = [(c.time_n_sub, c.time_norm) for c in dyn.components]
        with pytest.raises(ValueError, match="indirectly references"):
            model.add_dynamics(dyn, frequency=0.1)
        assert_same_state(before, model_state(model))
        assert (dyn.frequency, dyn.time_norm, dyn.n_sub, dyn.n_counter) == timing_before
        for comp, (time_n_sub, time_norm) in zip(
            dyn.components, components_before, strict=True
        ):
            assert comp.time_n_sub is time_n_sub
            assert comp.time_norm is time_norm
        assert dyn.parent_model is None

    #
    def test_profile_rejections_touch_neither_model(self):
        no_aux = make_model_file()
        before = model_state(no_aux.model_active)
        with pytest.raises(ValueError, match="aux_axis is not set"):
            add_profile(
                no_aux,
                model="simple_energy",
                parameter="GLP_01_A",
                profile="profile_pExpDecay",
            )
        assert_same_state(before, model_state(no_aux.model_active))

        chain = make_model_file(model="expression_chain", with_aux=True)
        model = chain.model_active
        assert model is not None  # type guard
        before = model_state(model)
        prof = chain.load_model(
            "models/file_profile.yaml",
            ["profile_pLinear"],
            "GLP_01_A",
            model_type="profile",
        )
        aux_before = prof.aux_axis  # the file's axis, given at load
        with pytest.raises(ValueError, match="indirectly references"):
            model.add_profile(prof)
        assert_same_state(before, model_state(model))
        assert prof.parent_model is None
        assert prof.aux_axis is aux_before
        ci, pi = model.find_par_by_name("GLP_01_A")
        assert model.components[ci].pars[pi].p_vary is False

    #
    def test_already_profiled_parameter(self):
        file = make_model_file(with_aux=True)
        add_profile(
            file,
            model="simple_energy",
            parameter="GLP_01_A",
            profile="profile_pExpDecay",
        )
        before = model_state(file.model_active)
        with pytest.raises(ValueError, match="already has a profile"):
            add_profile(
                file,
                model="simple_energy",
                parameter="GLP_01_A",
                profile="profile_pExpDecay",
            )
        assert_same_state(before, model_state(file.model_active))
