"""
Unit tests for the schema-7 identity and comparability hashes.

Pure-function tests with no I/O (schema plan §Execution order step 4):
the identity family (``file_content_hash`` → ``version_stamp`` →
``input_files`` → ``optimization_hash`` → ``handle``) and the independent
comparability hash (``fit_view_sha256``). End-to-end "a config change
mints a new slot" tests arrive with capture (step 8).
"""

import numpy as np
import pytest

from trspecfit.utils.fit_io import (
    _quantized_float_text,
    compute_file_content_hash,
    compute_file_version_stamp,
    compute_fit_view_sha256,
    compute_optimization_hash,
    compute_slot_handle,
    encode_input_files,
    encode_model_structure,
    encode_optimizer_settings,
)


#
def make_content_kwargs(**overrides):
    """Baseline file-content inputs; override single fields per test."""

    kwargs = {
        "data_raw": np.arange(6.0).reshape(2, 3),
        "energy": np.arange(3.0),
        "time": np.arange(2.0),
        "aux_axis": None,
    }
    kwargs.update(overrides)
    return kwargs


#
def make_optimization_kwargs(**overrides):
    """Baseline optimization-hash inputs; override single fields per test."""

    kwargs = {
        "input_files_json": encode_input_files(
            scope="file", entries=[("A", "stamp-a", '{"e_lim":[0,3]}')]
        ),
        "fit_type": "2d",
        "model_structure_json": encode_model_structure(
            [("A", ["peaks"], [("GLP_01_A", ["IRF", "MonoExpNeg"], -1.0)])]
        ),
        "parameter_metadata": [
            ("GLP_01_A", 0.0, 10.0, True, None),
            ("GLP_01_x0", -1.0, 1.0, False, None),
        ],
        "initial_state": [[1.0, 0.5]],
        "optimizer_settings_json": encode_optimizer_settings(
            stages=1, fit_alg_1="Nelder", fit_alg_2="leastsq", backend="mcp"
        ),
    }
    kwargs.update(overrides)
    return kwargs


#
#
class TestFileContentHash:
    #
    def test_deterministic(self):
        assert compute_file_content_hash(
            **make_content_kwargs()
        ) == compute_file_content_hash(**make_content_kwargs())

    #
    def test_dtype_is_identity(self):
        h64 = compute_file_content_hash(**make_content_kwargs())
        h32 = compute_file_content_hash(
            **make_content_kwargs(data_raw=np.arange(6, dtype=np.float32).reshape(2, 3))
        )
        assert h64 != h32

    #
    def test_absent_aux_differs_from_zero_length(self):
        absent = compute_file_content_hash(**make_content_kwargs(aux_axis=None))
        empty = compute_file_content_hash(**make_content_kwargs(aux_axis=np.array([])))
        assert absent != empty

    #
    def test_absent_time_differs_from_zero_length(self):
        absent = compute_file_content_hash(**make_content_kwargs(time=None))
        empty = compute_file_content_hash(**make_content_kwargs(time=np.array([])))
        assert absent != empty

    #
    def test_each_array_content_is_identity(self):
        base = compute_file_content_hash(**make_content_kwargs())
        data = compute_file_content_hash(
            **make_content_kwargs(data_raw=np.arange(6.0).reshape(2, 3) + 1.0)
        )
        energy = compute_file_content_hash(
            **make_content_kwargs(energy=np.arange(3.0) + 0.1)
        )
        time = compute_file_content_hash(
            **make_content_kwargs(time=np.arange(2.0) + 0.1)
        )
        assert len({base, data, energy, time}) == 4

    #
    def test_aux_axis_values_are_identity(self):
        base = compute_file_content_hash(**make_content_kwargs(aux_axis=np.arange(2.0)))
        other = compute_file_content_hash(
            **make_content_kwargs(aux_axis=np.arange(2.0) + 0.1)
        )
        assert base != other


#
#
class TestFileVersionStamp:
    #
    def test_correction_changes_stamp(self):
        base = compute_file_version_stamp(
            file_content_hash="c" * 64, dark=np.zeros(3), calibration=np.ones(3)
        )
        darker = compute_file_version_stamp(
            file_content_hash="c" * 64, dark=np.full(3, 0.1), calibration=np.ones(3)
        )
        recal = compute_file_version_stamp(
            file_content_hash="c" * 64, dark=np.zeros(3), calibration=np.full(3, 2.0)
        )
        assert len({base, darker, recal}) == 3

    #
    def test_content_hash_changes_stamp(self):
        a = compute_file_version_stamp(
            file_content_hash="a" * 64, dark=None, calibration=None
        )
        b = compute_file_version_stamp(
            file_content_hash="b" * 64, dark=None, calibration=None
        )
        assert a != b


#
#
class TestInputFilesEncoding:
    #
    def test_sorted_by_file_name(self):
        entries = [("B", "sb", "{}"), ("A", "sa", "{}")]
        assert encode_input_files(
            scope="project", entries=entries
        ) == encode_input_files(scope="project", entries=list(reversed(entries)))

    #
    def test_scope_is_identity(self):
        entries = [("A", "sa", "{}")]
        assert encode_input_files(scope="file", entries=entries) != encode_input_files(
            scope="project", entries=entries
        )

    #
    def test_invalid_scope_raises(self):
        with pytest.raises(ValueError, match="scope"):
            encode_input_files(scope="global", entries=[("A", "sa", "{}")])

    #
    def test_duplicate_file_name_raises(self):
        with pytest.raises(ValueError, match="duplicate"):
            encode_input_files(
                scope="project", entries=[("A", "s1", "{}"), ("A", "s2", "{}")]
            )


#
#
class TestModelStructureEncoding:
    #
    def test_model_name_boundary_framing(self):
        # ['IRF', 'MonoExp_Neg'] vs. one model literally named
        # 'IRF_MonoExp_Neg' — moving a character across a field boundary
        # must change the encoding (schema plan §Test coverage).
        two = encode_model_structure([("A", ["IRF", "MonoExp_Neg"], [])])
        one = encode_model_structure([("A", ["IRF_MonoExp_Neg"], [])])
        assert two != one

    #
    def test_energy_model_order_is_identity(self):
        ab = encode_model_structure([("A", ["peaks", "background"], [])])
        ba = encode_model_structure([("A", ["background", "peaks"], [])])
        assert ab != ba

    #
    def test_energy_model_rename_is_identity(self):
        # Renaming a top-level YAML model mints a new identity even when
        # its components are unchanged — the encoding carries model names,
        # never component names.
        renamed = encode_model_structure([("A", ["peaks_v2"], [])])
        assert encode_model_structure([("A", ["peaks"], [])]) != renamed

    #
    def test_per_file_frequency_is_identity(self):
        mixed = encode_model_structure(
            [
                ("A", ["peaks"], [("GLP_01_x0", ["decay"], 0.25)]),
                ("B", ["peaks"], [("GLP_01_x0", ["decay"], 0.5)]),
            ]
        )
        uniform = encode_model_structure(
            [
                ("A", ["peaks"], [("GLP_01_x0", ["decay"], 0.25)]),
                ("B", ["peaks"], [("GLP_01_x0", ["decay"], 0.25)]),
            ]
        )
        assert mixed != uniform

    #
    def test_files_and_attachments_sort_canonically(self):
        dyn_a = ("GLP_01_A", ["IRF"], -1.0)
        dyn_x0 = ("GLP_01_x0", ["decay"], -1.0)
        one = encode_model_structure(
            [("B", ["peaks"], [dyn_a, dyn_x0]), ("A", ["peaks"], [])]
        )
        two = encode_model_structure(
            [("A", ["peaks"], []), ("B", ["peaks"], [dyn_x0, dyn_a])]
        )
        assert one == two

    #
    def test_submodel_order_is_identity(self):
        # Submodel order is what assigns subcycles (principles
        # §model_structure) — it is preserved, never sorted.
        one = encode_model_structure(
            [("A", ["peaks"], [("GLP_01_A", ["IRF", "MonoExpNeg"], -1.0)])]
        )
        other = encode_model_structure(
            [("A", ["peaks"], [("GLP_01_A", ["MonoExpNeg", "IRF"], -1.0)])]
        )
        assert one != other

    #
    def test_submodel_rename_is_identity(self):
        renamed = encode_model_structure(
            [("A", ["peaks"], [("GLP_01_A", ["IRF_v2"], -1.0)])]
        )
        assert (
            encode_model_structure([("A", ["peaks"], [("GLP_01_A", ["IRF"], -1.0)])])
            != renamed
        )

    #
    def test_duplicate_dynamics_target_raises(self):
        with pytest.raises(ValueError, match="duplicate dynamics"):
            encode_model_structure(
                [
                    (
                        "A",
                        ["peaks"],
                        [
                            ("GLP_01_A", ["IRF"], -1.0),
                            ("GLP_01_A", ["decay"], -1.0),
                        ],
                    )
                ]
            )


#
#
class TestOptimizerSettingsEncoding:
    #
    def test_fit_alg_2_keyed_only_for_two_stages(self):
        one_stage_a = encode_optimizer_settings(
            stages=1, fit_alg_1="Nelder", fit_alg_2="leastsq", backend="mcp"
        )
        one_stage_b = encode_optimizer_settings(
            stages=1, fit_alg_1="Nelder", fit_alg_2="powell", backend="mcp"
        )
        assert one_stage_a == one_stage_b
        two_stage_a = encode_optimizer_settings(
            stages=2, fit_alg_1="Nelder", fit_alg_2="leastsq", backend="mcp"
        )
        two_stage_b = encode_optimizer_settings(
            stages=2, fit_alg_1="Nelder", fit_alg_2="powell", backend="mcp"
        )
        assert two_stage_a != two_stage_b

    #
    def test_seed_keyed_only_when_supplied(self):
        without = encode_optimizer_settings(
            stages=1, fit_alg_1="Nelder", fit_alg_2="leastsq", backend="mcp"
        )
        with_seed = encode_optimizer_settings(
            stages=1, fit_alg_1="Nelder", fit_alg_2="leastsq", backend="mcp", seed=7
        )
        assert without != with_seed
        assert "seed" not in without

    #
    def test_jac_fun_keyed_only_when_applied(self):
        base = dict(stages=1, fit_alg_1="Nelder", fit_alg_2="leastsq", backend="mcp")
        no_leastsq = encode_optimizer_settings(
            **base, jac_fun_name="trspecfit.fitlib.jacobian_fun"
        )
        assert no_leastsq == encode_optimizer_settings(**base)
        applied = encode_optimizer_settings(
            stages=2,
            fit_alg_1="Nelder",
            fit_alg_2="leastsq",
            backend="mcp",
            jac_fun_name="trspecfit.fitlib.jacobian_fun",
        )
        plain = encode_optimizer_settings(
            stages=2, fit_alg_1="Nelder", fit_alg_2="leastsq", backend="mcp"
        )
        assert applied != plain

    #
    def test_backend_is_identity(self):
        mcp = encode_optimizer_settings(
            stages=1, fit_alg_1="Nelder", fit_alg_2="leastsq", backend="mcp"
        )
        jax = encode_optimizer_settings(
            stages=1, fit_alg_1="Nelder", fit_alg_2="leastsq", backend="jax"
        )
        assert mcp != jax

    #
    def test_invalid_stages_raises(self):
        with pytest.raises(ValueError, match="stages"):
            encode_optimizer_settings(
                stages=3, fit_alg_1="Nelder", fit_alg_2="leastsq", backend="mcp"
            )


#
#
class TestOptimizationHash:
    #
    def test_deterministic(self):
        assert compute_optimization_hash(
            **make_optimization_kwargs()
        ) == compute_optimization_hash(**make_optimization_kwargs())

    #
    def test_vary_flip_is_identity(self):
        flipped = make_optimization_kwargs(
            parameter_metadata=[
                ("GLP_01_A", 0.0, 10.0, True, None),
                ("GLP_01_x0", -1.0, 1.0, True, None),
            ]
        )
        assert compute_optimization_hash(
            **make_optimization_kwargs()
        ) != compute_optimization_hash(**flipped)

    #
    def test_bound_change_is_identity(self):
        widened = make_optimization_kwargs(
            parameter_metadata=[
                ("GLP_01_A", 0.0, 20.0, True, None),
                ("GLP_01_x0", -1.0, 1.0, False, None),
            ]
        )
        assert compute_optimization_hash(
            **make_optimization_kwargs()
        ) != compute_optimization_hash(**widened)

    #
    def test_expr_is_identity(self):
        with_expr = make_optimization_kwargs(
            parameter_metadata=[
                ("GLP_01_A", 0.0, 10.0, True, None),
                ("GLP_01_x0", -1.0, 1.0, False, "GLP_01_A / 2"),
            ]
        )
        assert compute_optimization_hash(
            **make_optimization_kwargs()
        ) != compute_optimization_hash(**with_expr)

    #
    def test_parameter_order_is_identity(self):
        reordered = make_optimization_kwargs(
            parameter_metadata=[
                ("GLP_01_x0", -1.0, 1.0, False, None),
                ("GLP_01_A", 0.0, 10.0, True, None),
            ],
            initial_state=[[0.5, 1.0]],
        )
        assert compute_optimization_hash(
            **make_optimization_kwargs()
        ) != compute_optimization_hash(**reordered)

    #
    def test_initial_value_quantization_boundary(self):
        base = compute_optimization_hash(
            **make_optimization_kwargs(initial_state=[[1.23456789, 0.5]])
        )
        tenth_digit = compute_optimization_hash(
            **make_optimization_kwargs(initial_state=[[1.234567891, 0.5]])
        )
        ninth_digit = compute_optimization_hash(
            **make_optimization_kwargs(initial_state=[[1.23456788, 0.5]])
        )
        assert base == tenth_digit
        assert base != ninth_digit

    #
    def test_negative_zero_initial_value_normalized(self):
        pos = compute_optimization_hash(
            **make_optimization_kwargs(initial_state=[[0.0, 0.5]])
        )
        neg = compute_optimization_hash(
            **make_optimization_kwargs(initial_state=[[-0.0, 0.5]])
        )
        assert pos == neg

    #
    def test_fit_type_is_identity(self):
        assert compute_optimization_hash(
            **make_optimization_kwargs()
        ) != compute_optimization_hash(**make_optimization_kwargs(fit_type="sbs"))

    #
    def test_column_mismatch_raises(self):
        with pytest.raises(ValueError, match="columns"):
            compute_optimization_hash(
                **make_optimization_kwargs(initial_state=[[1.0, 0.5, 3.0]])
            )

    #
    def test_sbs_slice_rows_are_identity(self):
        one_row = compute_optimization_hash(
            **make_optimization_kwargs(initial_state=[[1.0, 0.5]])
        )
        two_rows = compute_optimization_hash(
            **make_optimization_kwargs(initial_state=[[1.0, 0.5], [1.0, 0.6]])
        )
        assert one_row != two_rows

    #
    def test_input_files_alone_are_identity(self):
        other = make_optimization_kwargs(
            input_files_json=encode_input_files(
                scope="file", entries=[("A", "stamp-b", '{"e_lim":[0,3]}')]
            )
        )
        assert compute_optimization_hash(
            **make_optimization_kwargs()
        ) != compute_optimization_hash(**other)

    #
    def test_model_structure_alone_is_identity(self):
        other = make_optimization_kwargs(
            model_structure_json=encode_model_structure([("A", ["peaks"], [])])
        )
        assert compute_optimization_hash(
            **make_optimization_kwargs()
        ) != compute_optimization_hash(**other)

    #
    def test_optimizer_settings_alone_are_identity(self):
        other = make_optimization_kwargs(
            optimizer_settings_json=encode_optimizer_settings(
                stages=2, fit_alg_1="Nelder", fit_alg_2="leastsq", backend="mcp"
            )
        )
        assert compute_optimization_hash(
            **make_optimization_kwargs()
        ) != compute_optimization_hash(**other)


#
#
class TestSlotHandle:
    #
    def test_file_name_differentiates_joint_siblings(self):
        shared = "f" * 64
        handle_a = compute_slot_handle(optimization_hash=shared, file_name="A")
        handle_b = compute_slot_handle(optimization_hash=shared, file_name="B")
        assert handle_a != handle_b

    #
    def test_full_width_hex(self):
        handle = compute_slot_handle(optimization_hash="f" * 64, file_name="A")
        assert len(handle) == 64
        assert set(handle) <= set("0123456789abcdef")


#
#
class TestFitViewSha256:
    #
    def test_window_is_comparability(self):
        full = compute_fit_view_sha256(
            observed=np.arange(4.0),
            energy=np.arange(4.0),
            time=None,
            aux_axis=None,
        )
        cropped = compute_fit_view_sha256(
            observed=np.arange(3.0),
            energy=np.arange(3.0),
            time=None,
            aux_axis=None,
        )
        assert full != cropped

    #
    def test_model_independent_by_construction(self):
        # A profile and a non-profile model on identical observations must
        # share a view — the function takes no model input, so identical
        # arrays are the whole contract.
        kwargs = {
            "observed": np.arange(4.0),
            "energy": np.arange(4.0),
            "time": None,
            "aux_axis": np.arange(2.0),
        }
        assert compute_fit_view_sha256(**kwargs) == compute_fit_view_sha256(**kwargs)

    #
    def test_aux_axis_presence_is_comparability(self):
        without = compute_fit_view_sha256(
            observed=np.arange(4.0), energy=np.arange(4.0), time=None, aux_axis=None
        )
        with_aux = compute_fit_view_sha256(
            observed=np.arange(4.0),
            energy=np.arange(4.0),
            time=None,
            aux_axis=np.arange(2.0),
        )
        assert without != with_aux

    #
    def test_observed_dtype_is_comparability(self):
        f64 = compute_fit_view_sha256(
            observed=np.arange(4.0), energy=np.arange(4.0), time=None, aux_axis=None
        )
        f32 = compute_fit_view_sha256(
            observed=np.arange(4, dtype=np.float32),
            energy=np.arange(4.0),
            time=None,
            aux_axis=None,
        )
        assert f64 != f32

    #
    def test_observed_values_alone_are_comparability(self):
        base = compute_fit_view_sha256(
            observed=np.arange(4.0), energy=np.arange(4.0), time=None, aux_axis=None
        )
        other = compute_fit_view_sha256(
            observed=np.arange(4.0) + 1.0,
            energy=np.arange(4.0),
            time=None,
            aux_axis=None,
        )
        assert base != other

    #
    def test_energy_coordinates_alone_are_comparability(self):
        base = compute_fit_view_sha256(
            observed=np.arange(4.0), energy=np.arange(4.0), time=None, aux_axis=None
        )
        shifted = compute_fit_view_sha256(
            observed=np.arange(4.0),
            energy=np.arange(4.0) + 0.1,
            time=None,
            aux_axis=None,
        )
        assert base != shifted

    #
    def test_time_coordinates_alone_are_comparability(self):
        base = compute_fit_view_sha256(
            observed=np.arange(6.0).reshape(2, 3),
            energy=np.arange(3.0),
            time=np.array([0.0, 1.0]),
            aux_axis=None,
        )
        other = compute_fit_view_sha256(
            observed=np.arange(6.0).reshape(2, 3),
            energy=np.arange(3.0),
            time=np.array([0.0, 2.0]),
            aux_axis=None,
        )
        assert base != other

    #
    def test_aux_axis_values_alone_are_comparability(self):
        base = compute_fit_view_sha256(
            observed=np.arange(4.0),
            energy=np.arange(4.0),
            time=None,
            aux_axis=np.arange(2.0),
        )
        other = compute_fit_view_sha256(
            observed=np.arange(4.0),
            energy=np.arange(4.0),
            time=None,
            aux_axis=np.arange(2.0) + 0.1,
        )
        assert base != other


#
#
class TestQuantizedFloatText:
    #
    def test_negative_zero_normalized(self):
        assert _quantized_float_text(-0.0) == "0"
        assert _quantized_float_text(-0.0) == _quantized_float_text(0.0)

    #
    def test_nine_significant_digits(self):
        assert _quantized_float_text(1.234567891) == "1.23456789"
        assert _quantized_float_text(123456789.4) == "123456789"
        assert _quantized_float_text(123456789.6) == "123456790"

    #
    def test_non_finite_values_stable(self):
        assert _quantized_float_text(float("inf")) == "inf"
        assert _quantized_float_text(float("-inf")) == "-inf"
        assert _quantized_float_text(float("nan")) == "nan"
