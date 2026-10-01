"""
Behavioral probes for the ownership contract, rules 1 to 3
(docs/design/api_ownership_contract.md; check 21 of docs/ai/code-review.md).

Reading the code shows what a method does; these probes show what the File
does afterwards: a caller mutates across the ownership boundary, or assigns
an owned attribute, and the File's state is what the contract says.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest
from _utils import make_project, simulate_noisy

from trspecfit import File, FitResults
from trspecfit.utils import fit_io


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
    def test_unknown_name_in_a_candidate_expression_leaves_the_model_unchanged(self):
        """The candidate fails at its own load, before anything is attached."""

        file = make_model_file()
        before = model_state(file.model_active)
        with pytest.raises(ValueError, match="references an unknown parameter"):
            add_dynamics(
                file,
                model="simple_energy",
                parameter="GLP_01_A",
                dynamics=["CrossModelExpr"],
            )
        assert_same_state(before, model_state(file.model_active))

        with_aux = make_model_file(with_aux=True)
        before = model_state(with_aux.model_active)
        with pytest.raises(ValueError, match="references an unknown parameter"):
            add_profile(
                with_aux,
                model="simple_energy",
                parameter="GLP_01_x0",
                profile="profile_pLinear_unknown_name",
            )
        assert_same_state(before, model_state(with_aux.model_active))

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


#
def make_fitted_file(*, try_ci: int = 0) -> tuple[File, fit_io.SavedFitSlot]:
    """A baseline fit on simulated single-peak data (``try_ci=1`` for intervals)."""

    energy = np.linspace(83, 87, 30)
    time = np.linspace(-2, 10, 24)
    truth = File(
        parent_project=make_project(name="truth"),
        name="truth",
        energy=energy,
        time=time,
    )
    truth.load_model(model_yaml="models/file_energy.yaml", model_info="single_glp")
    truth.add_time_dependence(
        target_model="single_glp",
        target_parameter="GLP_01_A",
        dynamics_yaml="models/file_time.yaml",
        dynamics_model=["MonoExpPos"],
    )
    data = simulate_noisy(truth.model_active, noise_level=0.01)
    project = make_project()
    file = File(parent_project=project, name="fit", data=data, energy=energy, time=time)
    file.load_model(model_yaml="models/file_energy.yaml", model_info="single_glp")
    file.define_baseline(0, 3, time_type="ind", show_plot=False)
    file.fit_baseline(model_name="single_glp", stages=1, try_ci=try_ci)
    slot = project.results.find(file="fit", fit_type="baseline")[-1]
    return file, slot


#
def record_content(slot: fit_io.SavedFitSlot) -> dict:
    """The container content of a slot, as independent copies."""

    assert slot.fit_settings is not None  # type guard
    assert slot.component_names is not None  # type guard
    return {
        "params": slot.params.copy(),
        "metrics": dict(slot.metrics),
        "fit_settings": dict(slot.fit_settings),
        "component_names": list(slot.component_names),
        "selection": dict(slot.selection),
        "conf_ci": None if slot.conf_ci is None else slot.conf_ci.copy(),
    }


#
def assert_same_content(slot: fit_io.SavedFitSlot, before: dict) -> None:
    pd.testing.assert_frame_equal(slot.params, before["params"])
    assert slot.metrics == before["metrics"]
    assert slot.fit_settings == before["fit_settings"]
    assert slot.component_names == before["component_names"]
    assert slot.selection == before["selection"]
    if before["conf_ci"] is None:
        assert slot.conf_ci is None
    else:
        assert slot.conf_ci is not None  # type guard
        pd.testing.assert_frame_equal(slot.conf_ci, before["conf_ci"])


#
#
class TestResultRecordsAreSnapshots:
    """Rule 5: every public read is a detached copy; the record never changes."""

    #
    def test_editing_what_a_read_returned_does_not_reach_the_record(self, tmp_path):
        file, slot = make_fitted_file(try_ci=1)
        before = record_content(slot)
        assert before["conf_ci"] is not None  # try_ci=1 produced intervals

        frame = slot.params
        frame.loc[0, "value"] = -999.0
        assert frame.loc[0, "value"] == -999.0  # the edit took, on the copy
        slot.metrics["chi2"] = -1.0
        slot.fit_settings["stages"] = 99  # type: ignore[index]
        slot.component_names.append("intruder")  # type: ignore[union-attr]
        slot.selection["e_lim"] = [0, 1]
        intervals = slot.conf_ci
        assert intervals is not None  # type guard
        intervals.iloc[0, 1] = -999.0
        assert intervals.iloc[0, 1] == -999.0
        assert_same_content(slot, before)

        # the same record through the query surface, and the next archive
        again = file.p.results.get(file="fit", model="single_glp", fit_type="baseline")
        assert again is slot
        assert_same_content(again, before)
        path = tmp_path / "snapshot.fit.h5"
        file.p.save_fits(path, show_output=0)
        loaded = FitResults.load(path).find(file="fit", fit_type="baseline")[-1]
        assert loaded.handle == slot.handle
        pd.testing.assert_frame_equal(loaded.params, before["params"])
        assert loaded.fit_settings == before["fit_settings"]
        assert loaded.component_names == before["component_names"]
        for key, value in before["metrics"].items():
            assert loaded.metrics[key] == pytest.approx(value, nan_ok=True)

    #
    def test_editing_the_live_result_after_the_fit_does_not_reach_the_slot(self):
        """Copy on set: the slot's conf_ci was the live model result's frame."""

        file, slot = make_fitted_file(try_ci=1)
        before = record_content(slot)
        live = file.model_base.result  # type: ignore[union-attr]
        assert live is not None  # type guard
        live_intervals = live.conf_ci
        live_intervals.iloc[0, 1] = -999.0
        assert live_intervals.iloc[0, 1] == -999.0  # the live object did change
        assert_same_content(slot, before)

    #
    def test_a_container_shared_between_records_is_copied_at_capture(self):
        """Copy on set at the record level: the joint fit builds its
        projection slots from one settings dict."""

        _, slot = make_fitted_file()
        shared = {"stages": 1, "mc": {"use_mc": 0}}
        frame = slot.params.copy()
        first = dataclasses.replace(slot, fit_settings=shared, params=frame)
        second = dataclasses.replace(slot, fit_settings=shared, params=frame)
        shared["stages"] = 99
        shared["mc"]["use_mc"] = 5
        frame.loc[0, "value"] = -999.0
        for record in (first, second):
            assert record.fit_settings == {"stages": 1, "mc": {"use_mc": 0}}
            pd.testing.assert_frame_equal(record.params, slot.params)

    #
    def test_nested_mcmc_and_joint_containers_are_detached(self):
        _, slot = make_fitted_file()
        payload = {
            "flatchain": pd.DataFrame({"GLP_01_A": [1.0, 2.0]}),
            "ci": pd.DataFrame({"par[v]/sigma[>]": ["GLP_01_A"], "best fit": [1.5]}),
            "lnsigma": None,
            "acceptance_fraction": np.array([0.3, 0.4]),
        }
        with_mcmc = dataclasses.replace(slot, mcmc=payload)
        payload["flatchain"].loc[0, "GLP_01_A"] = -999.0
        read = with_mcmc.mcmc
        assert read is not None  # type guard
        read["flatchain"].loc[1, "GLP_01_A"] = -999.0
        assert read["flatchain"].loc[1, "GLP_01_A"] == -999.0
        chain = with_mcmc.mcmc["flatchain"]  # type: ignore[index]
        assert list(chain["GLP_01_A"]) == [1.0, 2.0]
        assert not with_mcmc.mcmc["acceptance_fraction"].flags.writeable  # type: ignore[index]

        result = fit_io.mcmc_result_from_payload(payload)
        chain = result.flatchain
        chain.loc[0, "GLP_01_A"] = 7.0
        assert chain.loc[0, "GLP_01_A"] == 7.0
        assert list(result.flatchain["GLP_01_A"]) == [-999.0, 2.0]

        parameter_map = {"GLP_01_A": "file00_GLP_01_A"}
        projection = fit_io.JointFitProjection(slot=slot, parameter_map=parameter_map)
        parameter_map["GLP_01_A"] = "edited"
        projection.parameter_map["GLP_01_A"] = "edited again"  # type: ignore[index]
        assert projection.parameter_map == {"GLP_01_A": "file00_GLP_01_A"}
        joint = fit_io.JointFitResult(
            model_name="single_glp",
            optimization_hash="0" * 64,
            input_files="",
            model_structure="",
            projections=(projection,),
            fit_alg="leastsq",
            timestamp="t",
            params=slot.params,
            metrics={"chi2": 1.0},
            fit_settings={"stages": 1},
        )
        joint_frame = joint.params
        joint_frame.loc[0, "value"] = -999.0
        assert joint_frame.loc[0, "value"] == -999.0
        joint.metrics["chi2"] = -1.0  # type: ignore[index]
        pd.testing.assert_frame_equal(joint.params, slot.params)
        assert joint.metrics == {"chi2": 1.0}

    #
    def test_detach_covers_mappings_tuples_and_direct_arrays(self):
        """Direct construction with a Mapping subclass, a tuple holding a
        dict, or a writable array is detached at the record boundary too."""

        from collections import UserDict

        _, slot = make_fitted_file()
        mapping = UserDict({"GLP_01_A": "file00_GLP_01_A"})
        projection = fit_io.JointFitProjection(slot=slot, parameter_map=mapping)
        mapping["GLP_01_A"] = "edited"
        assert projection.parameter_map == {"GLP_01_A": "file00_GLP_01_A"}

        settings = {"window": ({"e_lim": [0, 5]},)}
        record = dataclasses.replace(slot, fit_settings=settings)
        settings["window"][0]["e_lim"].append(99)
        read = record.fit_settings
        assert read is not None  # type guard
        read["window"][0]["e_lim"].append(98)
        assert record.fit_settings == {"window": ({"e_lim": [0, 5]},)}

        acceptance = np.array([0.3, 0.4])
        result = fit_io.MCMCResult(
            table=pd.DataFrame(),
            flatchain=pd.DataFrame(),
            acceptance_fraction=acceptance,
        )
        acceptance[0] = -1.0
        assert result.acceptance_fraction is not None  # type guard
        assert list(result.acceptance_fraction) == [0.3, 0.4]
        assert not result.acceptance_fraction.flags.writeable
