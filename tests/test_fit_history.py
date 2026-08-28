"""
Tests for the in-memory fit-history layer:

- Project._fit_history accumulation as fits complete.
- SavedFitSlot field correctness (observed/fit shape, residual reconstruction,
  metrics match lmfit, selection capture).
- Schema-7 identity at capture: the handle chain is a pure function of the
  captured state; correction variants coexist; exact re-runs dedup on save.
- Copy-and-freeze ownership at the capture boundary.
- Project.results snapshot semantics (immutability after access).
- FitResults find / get / files / models / iteration.
- SbS extraction survives the seed-template restoration at the end of
  fit_slice_by_slice.

Writer/reader record contracts (collision table, bundle integrity, layout)
live in ``test_fit_archive_writer.py``; full-fit archive round-trips in
``test_fit_archive_roundtrip.py``; no-I/O hash-function properties in
``test_fit_identity_hashes.py``.
"""

import matplotlib

matplotlib.use("Agg")

import dataclasses

import numpy as np
import pandas as pd
import pytest
from _utils import make_project, simulate_clean, simulate_noisy

from trspecfit import File, FitResults
from trspecfit.utils.fit_io import (
    JointFitProjection,
    JointFitResult,
    SavedFile,
    SavedFitSlot,
    _compute_sigma_eff,
    build_selection_json,
    capture_saved_file,
    collapse_history_to_snapshot,
    collapse_joint_history_to_snapshot,
    compute_file_content_hash,
    compute_optimization_hash,
    compute_slot_handle,
    encode_input_files,
    encode_model_structure,
    encode_optimizer_settings,
    joint_comparability,
    optimizer_settings_from_provenance,
    read_archive,
    resolve_fit_reference,
    select_snapshot_slots,
    set_fit_label,
)


#
def _make_truth_file(project):
    energy = np.linspace(83, 87, 30)
    time = np.linspace(-2, 10, 24)
    file = File(parent_project=project, name="truth")
    file.energy = energy
    file.time = time
    file.dim = 2
    file.load_model(
        model_yaml="models/file_energy.yaml",
        model_info="single_glp",
    )
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
    file.load_model(
        model_yaml="models/file_energy.yaml",
        model_info="single_glp",
    )
    return file


#
def _setup_baseline_fit():
    """Run a baseline fit and return (project, file). Uses noisy data so
    chi2 > 0 and AIC/BIC are finite (clean data gives chi2=0 -> log(0)=nan)."""

    truth_project = make_project(name="truth")
    truth = _make_truth_file(truth_project)
    data = simulate_noisy(truth.model_active, noise_level=0.01)

    project = make_project(name="fit")
    file = _make_fit_file(project, data, truth.energy, truth.time)
    file.define_baseline(time_start=0, time_stop=3, time_type="ind", show_plot=False)
    file.fit_baseline(model_name="single_glp", stages=2, try_ci=0)
    return project, file


#
def _fit_file_with_seed():
    """(project, file, model, seed values) ready for a baseline fit.

    Fits write their output back into the live model, so an exact re-run
    needs the captured seed restored via ``model.update_value(seed)``.
    """

    truth_project = make_project(name="truth")
    truth = _make_truth_file(truth_project)
    data = simulate_noisy(truth.model_active, noise_level=0.01)
    project = make_project(name="fit")
    file = _make_fit_file(project, data, truth.energy, truth.time)
    file.define_baseline(time_start=0, time_stop=3, time_type="ind", show_plot=False)
    model = next(m for m in file.models if m.name == "single_glp")
    seed = [p.value for p in model.lmfit_pars.values()]
    return project, file, model, seed


#
def test_correction_refit_archives_both_variants(tmp_path):
    """A fit before a data correction and its refit after are distinct
    variants that archive side by side.

    Schema 7 stores immutable ``data_raw`` once and per-slot
    ``dark``/``calibration``: the version stamp folds the correction into
    each slot's identity, so nothing is stale and nothing is skipped
    (the schema-6 warn-and-skip is gone). The archive hashes what it
    stores — the raw payload verifies against ``file_content_hash``.
    """

    project, file = _setup_baseline_fit()
    assert file.energy is not None  # type guard
    assert file.data_raw is not None  # type guard
    raw = file.data_raw.copy()
    dark = np.full(file.energy.size, 0.1)
    file.subtract_dark(dark)
    file.fit_baseline(model_name="single_glp", stages=2, try_ci=0)
    s_raw, s_cor = project._fit_history
    assert s_raw.handle != s_cor.handle
    # The version stamp (inside input_files) carries the correction state.
    assert s_raw.input_files != s_cor.input_files
    assert s_raw.dark is None
    assert s_cor.dark is not None  # type guard
    np.testing.assert_array_equal(s_cor.dark, dark)

    archive_path = tmp_path / "variants.fit.h5"
    project.save_fits(archive_path, show_output=0)
    saved = read_archive(archive_path)
    sf = saved.files[0]
    np.testing.assert_array_equal(sf.data_raw, raw)
    recomputed = compute_file_content_hash(
        data_raw=sf.data_raw, energy=sf.energy, time=sf.time, aux_axis=sf.aux_axis
    )
    assert recomputed == sf.file_content_hash
    assert {s.handle for s in sf.slots} == {s_raw.handle, s_cor.handle}
    loaded = FitResults.load(archive_path)
    assert len(loaded) == 2


#
def test_correction_reversal_restores_identical_input_files():
    """reset_dark() restores the raw version stamp exactly.

    The reversal refit shares ``input_files`` with the original raw fit
    (same content hash, same absent corrections, same selection); the
    three fits stay distinct slots regardless, because each refit seeds
    from the previous output (initial state is identity).
    """

    project, file = _setup_baseline_fit()
    assert file.energy is not None  # type guard
    file.subtract_dark(np.full(file.energy.size, 0.1))
    file.fit_baseline(model_name="single_glp", stages=2, try_ci=0)
    file.reset_dark()
    file.fit_baseline(model_name="single_glp", stages=2, try_ci=0)
    s_raw, s_cor, s_back = project._fit_history
    assert s_back.input_files == s_raw.input_files
    assert s_back.input_files != s_cor.input_files
    assert s_back.dark is None
    assert len({s.handle for s in project._fit_history}) == 3


#
def test_append_same_name_different_content_raises(tmp_path):
    """Appending a same-name file with different content is an integrity error.

    Without the guard, write_archive files the new slots under the first
    name-matched group — the old data, axes, and content hash. The raise
    is unconditional (overwrite= is slot-scoped and does not authorize
    re-associating a measurement) and pre-mutation: the failed append
    leaves the archive byte-identical.
    """

    project_a, _ = _setup_baseline_fit()
    archive_path = tmp_path / "append.fit.h5"
    project_a.save_fits(archive_path, show_output=0)
    before = read_archive(archive_path)
    before_bytes = archive_path.read_bytes()

    # A second session: same project name, same file name, different data.
    truth_project = make_project(name="truth")
    truth = _make_truth_file(truth_project)
    data = simulate_noisy(truth.model_active, noise_level=0.01)
    project_b = make_project(name="fit")
    file_b = _make_fit_file(project_b, data * 1.1, truth.energy, truth.time)
    file_b.define_baseline(time_start=0, time_stop=3, time_type="ind", show_plot=False)
    file_b.fit_baseline(model_name="single_glp", stages=2, try_ci=0)
    for overwrite in (False, True):
        with pytest.raises(ValueError, match="file_content_hash mismatch"):
            project_b.save_fits(archive_path, overwrite=overwrite, show_output=0)
    # Diagnostic (decoded) checks first, so a real mutation is named...
    after = read_archive(archive_path)
    assert after.timestamp_updated == before.timestamp_updated
    assert len(after.files) == len(before.files) == 1
    assert len(after.files[0].slots) == len(before.files[0].slots) == 1
    # ...then the full promised contract: byte-identical afterwards (a
    # pass above with a diff here would point at h5py rewriting
    # internals on the write-mode open, not at a payload mutation).
    assert archive_path.read_bytes() == before_bytes


#
def test_same_name_providers_resolve_by_parent_association():
    """Each archived slot resolves to its own SavedFile's axes.

    The schema-7 writer refuses same-name file groups, but the provider
    association is structural, never name-guessed: a SavedFile owns the
    slots read from its group. A purely name-keyed provider map would
    collapse duplicates (last one wins), silently serving one record's
    axes/data to another record's slots.
    """

    slot_a = _slot_stub(file_name="dup", model_name="m1")
    slot_b = _slot_stub(file_name="dup", model_name="m2")
    e_a = np.linspace(0.0, 1.0, 5)
    e_b = np.linspace(10.0, 20.0, 5)

    def saved_file(energy, slot):
        data = np.zeros(5)
        return SavedFile(
            name="dup",
            original_path="dup.h5",
            dim=1,
            shape=(5,),
            file_content_hash=compute_file_content_hash(
                data_raw=data, energy=energy, time=np.array([]), aux_axis=None
            ),
            data_raw=data,
            energy=energy,
            time=np.array([], dtype=np.float64),
            slots=(slot,),
        )

    sf_a = saved_file(e_a, slot_a)
    sf_b = saved_file(e_b, slot_b)
    results = FitResults(slots=[slot_a, slot_b], files=[sf_a, sf_b])
    # Invariant check on the association: a name-keyed map would send
    # slot_a to sf_b (the later same-name provider).
    assert results._provider_for(slot_a) is sf_a
    assert results._provider_for(slot_b) is sf_b
    # A slot not owned by any archive record (copied/reconstructed —
    # unsupported) gets no provider (index-axes fallback), never a
    # same-name guess.
    slot_c = _slot_stub(file_name="dup", model_name="m1")
    results_c = FitResults(slots=[slot_c], files=[sf_a, sf_b])
    assert results_c._provider_for(slot_c) is None


#
# --- identity helpers --------------------------------------------------------
#


#
class TestIdentityCapture:
    """The handle chain is a pure function of captured state.

    Fit-level identity: what a user-visible change to the optimization
    setup does to the slot minted by a real fit. No-I/O hash-function
    properties (framing, ordering, per-input isolation) live in
    ``test_fit_identity_hashes.py``.
    """

    #
    def test_selection_json_is_deterministic(self):
        a = build_selection_json("baseline", base_t_ind=[0, 5], e_lim=[10, 20])
        b = build_selection_json("baseline", e_lim=[10, 20], base_t_ind=[0, 5])
        assert a == b  # sorted keys

    #
    def test_identity_chain_recomputes_from_slot_fields(self):
        """handle and optimization_hash recompute exactly from the
        persisted slot payload — capture stored the true hash inputs
        (parameter table, initial state, settings), nothing hidden."""

        project, _ = _setup_baseline_fit()
        slot = project._fit_history[0]
        assert slot.fit_settings is not None  # type guard
        rows = [
            (
                str(rec["name"]),
                float(rec["min"]),
                float(rec["max"]),
                bool(rec["vary"]),
                rec["expr"] if rec["expr"] else None,
            )
            for rec in slot.params.to_dict("records")
        ]
        recomputed = compute_optimization_hash(
            input_files_json=slot.input_files,
            fit_type=slot.fit_type,
            model_structure_json=slot.model_structure,
            parameter_metadata=rows,
            initial_state=np.asarray([slot.params["init_value"].to_numpy(dtype=float)]),
            optimizer_settings_json=optimizer_settings_from_provenance(
                slot.fit_settings
            ),
        )
        assert recomputed == slot.optimization_hash
        assert (
            compute_slot_handle(optimization_hash=recomputed, file_name=slot.file_name)
            == slot.handle
        )

    #
    def test_identical_rerun_shares_handle(self):
        """An exact re-run (same seed, settings, view, data) is not a new
        variant — it shares the handle and dedups at collapse."""

        project, file, model, seed = _fit_file_with_seed()
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
        model.update_value(seed)  # fits write back; restore the exact seed
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
        first, second = project._fit_history
        assert first.optimization_hash == second.optimization_hash
        assert first.handle == second.handle

    #
    def test_vary_flip_mints_distinct_slot(self):
        project, file, model, seed = _fit_file_with_seed()
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
        model.update_value(seed)
        model.lmfit_pars["GLP_01_x0"].vary = False
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
        first, second = project._fit_history
        assert first.optimization_hash != second.optimization_hash
        assert first.handle != second.handle

    #
    def test_bound_change_mints_distinct_slot(self):
        project, file, model, seed = _fit_file_with_seed()
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
        model.update_value(seed)
        par_A = model.lmfit_pars["GLP_01_A"]
        new_max = float(par_A.value) * 10.0 + 7.0
        assert new_max != par_A.max  # the change must actually change it
        par_A.max = new_max
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
        first, second = project._fit_history
        assert first.optimization_hash != second.optimization_hash
        assert first.handle != second.handle


#
class TestOptimizerSeed:
    """The optimizer seed is a keyed input when supplied (principles
    §"A slot is a configuration, not an execution"): forwarded to the
    ``fit_alg_1`` stage, part of the hash when given, absent otherwise.
    No capability table of seed-accepting methods exists — SciPy's own
    TypeError surfaces a seed on a method that cannot consume it."""

    #
    @staticmethod
    def _de_baseline(seed):
        """One seeded differential_evolution baseline fit on a fresh,
        identically-built project; returns its slot."""

        project, file, _, _ = _fit_file_with_seed()
        file.fit_baseline(
            model_name="single_glp",
            stages=1,
            fit_alg_1="differential_evolution",
            seed=seed,
            try_ci=0,
        )
        return project._fit_history[0]

    #
    def test_seeded_stochastic_rerun_is_the_same_configuration(self):
        """differential_evolution seed=42, re-run → same handle, same
        values (the enrich / no-op scenario row); collapse dedups
        silently."""

        first = self._de_baseline(42)
        second = self._de_baseline(42)
        assert first.handle == second.handle
        np.testing.assert_array_equal(
            first.params["value"].to_numpy(dtype=float),
            second.params["value"].to_numpy(dtype=float),
        )
        assert first.fit_settings is not None  # type guard
        assert first.fit_settings["seed"] == 42
        assert collapse_history_to_snapshot([first, second]) == [second]

    #
    def test_seeded_and_unseeded_are_distinct_configurations(self):
        """unseeded, then seed=42 → the hashes differ → two slots; the
        seed enters fit_settings only when supplied."""

        seeded = self._de_baseline(42)
        unseeded = self._de_baseline(None)
        assert seeded.handle != unseeded.handle
        assert unseeded.fit_settings is not None  # type guard
        assert "seed" not in unseeded.fit_settings
        assert len(collapse_history_to_snapshot([seeded, unseeded])) == 2

    #
    def test_seed_on_a_method_that_cannot_consume_it_raises(self):
        """leastsq takes no seed; the library's TypeError is the surface
        (no capability table), and no slot is captured from a failed
        fit."""

        project, file, _, _ = _fit_file_with_seed()
        with pytest.raises(TypeError, match="seed"):
            file.fit_baseline(
                model_name="single_glp",
                stages=1,
                fit_alg_1="leastsq",
                seed=42,
                try_ci=0,
            )
        assert project._fit_history == []

    #
    def test_two_stage_seed_goes_to_the_global_stage_only(self):
        """stages=2 differential_evolution + leastsq with a seed: the
        deterministic local stage receives none (it would reject it) and
        the seed is keyed once for the configuration."""

        project, file, _, _ = _fit_file_with_seed()
        file.fit_baseline(
            model_name="single_glp",
            stages=2,
            fit_alg_1="differential_evolution",
            fit_alg_2="leastsq",
            seed=7,
            try_ci=0,
        )
        slot = project._fit_history[0]
        assert slot.fit_settings is not None  # type guard
        assert slot.fit_settings["seed"] == 7
        assert slot.fit_alg == "leastsq"  # the final stage's method


#
class TestCaptureOwnership:
    """Copy-and-freeze at the capture boundary (Principle 4)."""

    #
    def test_slot_arrays_frozen_and_independent_of_live_file(self):
        project, file = _setup_baseline_fit()
        slot = project._fit_history[0]
        assert not slot.observed.flags.writeable
        assert not slot.fit.flags.writeable
        observed_before = slot.observed.copy()
        assert file.data_base is not None  # type guard
        file.data_base *= 3.0
        np.testing.assert_array_equal(slot.observed, observed_before)

    #
    def test_captured_payload_frozen_and_independent(self):
        project, file = _setup_baseline_fit()
        captured = project._captured_files[file.name]
        assert not captured.data_raw.flags.writeable
        raw_before = captured.data_raw.copy()
        assert file.data_raw is not None  # type guard
        file.data_raw[0, 0] += 1.0  # the live array stays the user's
        np.testing.assert_array_equal(captured.data_raw, raw_before)

    #
    def test_captured_empty_time_axis_is_frozen(self):
        """A 1D file has no time axis; the synthesized empty array is
        still part of the frozen record."""

        sf = capture_saved_file(
            name="one_d",
            original_path="x",
            dim=1,
            data_raw=np.zeros(3),
            energy=np.arange(3.0),
            time=None,
            aux_axis=None,
            file_content_hash="0" * 64,
        )
        assert sf.time.shape == (0,)
        assert not sf.time.flags.writeable

    #
    def test_data_raw_mutation_after_capture_raises_on_next_fit(self):
        """In-place raw-data mutation would put slots from different
        measurements under one name; the next fit refuses."""

        project, file = _setup_baseline_fit()
        assert file.data_raw is not None  # type guard
        file.data_raw[0, 0] += 1.0
        with pytest.raises(RuntimeError, match="in-place mutation is not supported"):
            file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)


#
# --- baseline slot extraction ------------------------------------------------
#


#
class TestBaselineSlot:
    #
    def test_history_grows_by_one_after_fit(self):
        project, _ = _setup_baseline_fit()
        assert len(project._fit_history) == 1

    #
    def test_slot_basic_fields(self):
        project, file = _setup_baseline_fit()
        slot = project._fit_history[0]
        assert isinstance(slot, SavedFitSlot)
        assert slot.fit_type == "baseline"
        assert slot.model_name == "single_glp"
        assert slot.file_name == file.name
        assert slot.observed.shape == slot.fit.shape
        assert slot.observed.size > 0

    #
    def test_residual_matches_observed_minus_fit(self):
        """Invariant: residuals = observed - fit, with no recipe replay."""

        project, _ = _setup_baseline_fit()
        slot = project._fit_history[0]
        residual = slot.observed - slot.fit
        # chi2_raw in metrics should match sum of squared residuals (the
        # lmfit-unweighted SSE diagnostic; chi2 is the σ-calibrated form).
        assert slot.metrics["chi2_raw"] == pytest.approx(float(np.sum(residual**2)))

    #
    def test_metrics_keys_present(self):
        project, _ = _setup_baseline_fit()
        slot = project._fit_history[0]
        assert set(slot.metrics.keys()) == {
            "chi2_raw",
            "chi2_red_raw",
            "chi2",
            "chi2_red",
            "r2",
            "aic",
            "bic",
        }
        # Raw + dimensionless metrics are always finite for a successful fit.
        for k in ("chi2_raw", "chi2_red_raw", "r2", "aic", "bic"):
            assert np.isfinite(slot.metrics[k])
        # Calibrated metrics are NaN when no sigma was set on the file.
        assert np.isnan(slot.metrics["chi2"])
        assert np.isnan(slot.metrics["chi2_red"])

    #
    def test_selection_captures_base_t_ind(self):
        project, _ = _setup_baseline_fit()
        slot = project._fit_history[0]
        # define_baseline(time_start=0, time_stop=3, time_type="ind") yields
        # the inclusive index range [0, 3] -> exclusive slice [0, 4).
        assert slot.selection["base_t_ind"] == [0, 4]

    #
    def test_stages2_init_value_is_true_seed_not_stage1_output(self):
        """Regression: lmfit resets init_value at the start of every
        optimization stage, and fit_wrapper's stage 2 starts from stage
        1's output — so a two-stage result's init_value would silently
        become stage 1's output unless restore_true_init_values corrects
        it before the slot is built. _setup_baseline_fit already uses
        stages=2."""

        truth_project = make_project(name="truth")
        truth = _make_truth_file(truth_project)
        data = simulate_noisy(truth.model_active, noise_level=0.01)

        project = make_project(name="fit")
        file = _make_fit_file(project, data, truth.energy, truth.time)
        file.define_baseline(
            time_start=0, time_stop=3, time_type="ind", show_plot=False
        )

        model = next(m for m in file.models if m.name == "single_glp")
        true_seed = {name: par.value for name, par in model.lmfit_pars.items()}
        file.fit_baseline(model_name="single_glp", stages=2, try_ci=0)

        slot = project._fit_history[0]
        persisted_init = dict(
            zip(slot.params["name"], slot.params["init_value"], strict=True)
        )
        assert persisted_init.keys() == true_seed.keys()
        for name, seed_value in true_seed.items():
            assert persisted_init[name] == pytest.approx(seed_value)

    #
    def test_stages1_init_value_is_true_seed(self):
        """stages=1 already had correct init_value (no intermediate stage to
        taint it); confirm the added correction call doesn't change that."""

        truth_project = make_project(name="truth")
        truth = _make_truth_file(truth_project)
        data = simulate_noisy(truth.model_active, noise_level=0.01)

        project = make_project(name="fit")
        file = _make_fit_file(project, data, truth.energy, truth.time)
        file.define_baseline(
            time_start=0, time_stop=3, time_type="ind", show_plot=False
        )

        model = next(m for m in file.models if m.name == "single_glp")
        true_seed = {name: par.value for name, par in model.lmfit_pars.items()}
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)

        slot = project._fit_history[0]
        persisted_init = dict(
            zip(slot.params["name"], slot.params["init_value"], strict=True)
        )
        for name, seed_value in true_seed.items():
            assert persisted_init[name] == pytest.approx(seed_value)

    #
    def test_stages2_fit_wrapper_result_and_report_show_true_seed(self, capsys):
        """The fix lives in fitlib.fit_wrapper itself (not the slot
        extractors), so it must be visible on two things the slot layer
        doesn't touch: FitOutput.par_fin.params directly (any direct
        consumer, not just code that passes through
        _append_baseline_slot), and lmfit.report_fit's own printed
        "(init = ...)" annotation for the local-optimization stage
        (previously showed stage 1's output, contradicting the archive)."""

        truth_project = make_project(name="truth")
        truth = _make_truth_file(truth_project)
        data = simulate_noisy(truth.model_active, noise_level=0.01)

        project = make_project(name="fit", show_output=1)
        file = _make_fit_file(project, data, truth.energy, truth.time)
        file.define_baseline(
            time_start=0, time_stop=3, time_type="ind", show_plot=False
        )

        model = next(m for m in file.models if m.name == "single_glp")
        true_seed = {name: par.value for name, par in model.lmfit_pars.items()}
        capsys.readouterr()  # drop setup output
        file.fit_baseline(model_name="single_glp", stages=2, try_ci=0)
        printed = capsys.readouterr().out

        result = file.model_base.result
        assert result is not None
        for name, seed_value in true_seed.items():
            assert result.par_fin.params[name].init_value == pytest.approx(seed_value)

        # lmfit.report_fit's "(init = <value>)" for the *local optimization*
        # stage must show the true seed, not stage 1's output. lmfit
        # formats it with .7g (printfuncs.py:177).
        local_section = printed.split("Results local optimization fit")[1]
        for seed_value in true_seed.values():
            assert f"(init = {seed_value:.7g})" in local_section


#
# --- spectrum slot extraction ------------------------------------------------
#


#
class TestSpectrumSlot:
    #
    def test_spectrum_slot_records_time_point(self):
        truth_project = make_project(name="truth")
        truth = _make_truth_file(truth_project)
        data = simulate_clean(truth.model_active)

        project = make_project(name="fit")
        file = _make_fit_file(project, data, truth.energy, truth.time)
        file.fit_spectrum(
            "single_glp",
            time_point=5,
            time_type="ind",
            stages=1,
            try_ci=0,
            show_plot=False,
        )

        assert len(project._fit_history) == 1
        slot = project._fit_history[0]
        assert slot.fit_type == "spectrum"
        assert slot.selection["time_point"] == 5
        assert slot.selection["time_range"] is None
        assert slot.selection["time_type"] == "ind"

    #
    def test_refit_at_different_time_point_creates_distinct_slots(self):
        """selection_json includes time_point, so refits don't collide."""

        truth_project = make_project(name="truth")
        truth = _make_truth_file(truth_project)
        data = simulate_clean(truth.model_active)

        project = make_project(name="fit")
        file = _make_fit_file(project, data, truth.energy, truth.time)
        file.fit_spectrum(
            "single_glp",
            time_point=5,
            time_type="ind",
            stages=1,
            try_ci=0,
            show_plot=False,
        )
        file.fit_spectrum(
            "single_glp",
            time_point=10,
            time_type="ind",
            stages=1,
            try_ci=0,
            show_plot=False,
        )
        handles = {s.handle for s in project._fit_history}
        assert len(handles) == 2  # different selections -> different identities


#
# --- SbS slot extraction ------------------------------------------------------
#


#
class TestSbSSlot:
    #
    @pytest.mark.slow
    def test_sbs_slot_per_slice_metrics(self):
        truth_project = make_project(name="truth")
        truth = _make_truth_file(truth_project)
        data = simulate_clean(truth.model_active)

        project = make_project(name="fit")
        project.spec_fun_str = "fit_model_mcp"
        file = _make_fit_file(project, data, truth.energy, truth.time)
        file.fit_slice_by_slice(
            "single_glp",
            n_workers=1,
            seed_source="model",
            seed_adapt=None,
            try_ci=0,
        )

        assert len(project._fit_history) == 1
        slot = project._fit_history[0]
        assert slot.fit_type == "sbs"
        # observed / fit are 2D, one row per time slice.
        assert slot.observed.ndim == 2
        assert slot.observed.shape == slot.fit.shape
        assert slot.observed.shape[0] == len(file.time)
        # Metrics are per-slice arrays, frozen like every captured array.
        for k in ("chi2", "chi2_red", "r2", "aic", "bic"):
            assert isinstance(slot.metrics[k], np.ndarray)
            assert slot.metrics[k].shape == (len(file.time),)
            assert not slot.metrics[k].flags.writeable
        # The slot-backed accessor serves the wide per-slice params frame.
        sbs_df = file.get_parameters(fit_type="sbs")
        pd.testing.assert_frame_equal(sbs_df, slot.params)
        assert len(sbs_df) == len(file.time)
        # Shared per-parameter metadata, column-aligned with the wide frame.
        assert slot.params_meta is not None  # type guard
        assert list(slot.params_meta.columns) == ["name", "vary", "min", "max", "expr"]
        assert list(slot.params_meta["name"]) == list(slot.params.columns)
        assert bool(slot.params_meta["vary"].any())
        # Per-slice stderr mirrors the wide params frame's shape.
        assert slot.params_stderr is not None  # type guard
        assert slot.params_stderr.shape == slot.params.shape
        assert list(slot.params_stderr.columns) == list(slot.params.columns)
        # Provenance records the SbS seeding recipe.
        assert slot.fit_settings is not None  # type guard
        assert slot.fit_settings["seed_source"] == "model"
        assert slot.fit_settings["seed_adapt"] is None
        assert slot.fit_settings["seed_values"] is None
        assert slot.fit_settings["stages"] == 1

    #
    @pytest.mark.slow
    def test_sbs_slot_records_explicit_dict_seed(self):
        """Explicit dict seeds land in fit_settings as the normalized,
        parameter-ordered float list (regression: the raw dict used to be
        np.asarray()'d, which raised TypeError at slot capture)."""

        truth_project = make_project(name="truth")
        truth = _make_truth_file(truth_project)
        data = simulate_clean(truth.model_active)

        project = make_project(name="fit")
        project.spec_fun_str = "fit_model_mcp"
        file = _make_fit_file(project, data, truth.energy, truth.time)
        model = file.model_active
        seed_values = {
            name: model.lmfit_pars[name].value for name in model.parameter_names
        }
        file.fit_slice_by_slice(
            "single_glp",
            n_workers=1,
            seed_source="explicit",
            seed_values=seed_values,
            seed_adapt=None,
            try_ci=0,
        )

        settings = project._fit_history[0].fit_settings
        assert settings is not None  # type guard
        assert settings["seed_source"] == "explicit"
        assert settings["seed_values"] == [
            float(seed_values[name]) for name in model.parameter_names
        ]

    #
    @pytest.mark.slow
    def test_sbs_slot_survives_seed_template_restoration(self):
        """
        SbS ends with model_sbs.update_value(seed_template, par_select='all'),
        which would blow away live model state. The slot must already hold a
        complete snapshot before that happens.
        """

        truth_project = make_project(name="truth")
        truth = _make_truth_file(truth_project)
        data = simulate_clean(truth.model_active)

        project = make_project(name="fit")
        project.spec_fun_str = "fit_model_mcp"
        file = _make_fit_file(project, data, truth.energy, truth.time)
        file.fit_slice_by_slice(
            "single_glp",
            n_workers=1,
            seed_source="model",
            seed_adapt=None,
            try_ci=0,
        )

        # After fit_slice_by_slice returns, the seed-template restoration has
        # already run. The slot must still hold valid, finite per-slice metrics
        # (built before the restoration via copied snapshot args).
        slot = project._fit_history[0]
        assert np.all(np.isfinite(slot.metrics["chi2_raw"]))
        assert slot.params.shape[0] == len(file.time)


#
# --- 2D slot extraction -------------------------------------------------------
#


#
class TestTwoDSlot:
    #
    def test_2d_slot_basic_fields(self):
        truth_project = make_project(name="truth")
        truth = _make_truth_file(truth_project)
        data = simulate_clean(truth.model_active)

        project = make_project(name="fit")
        file = _make_fit_file(project, data, truth.energy, truth.time)
        file.define_baseline(
            time_start=0,
            time_stop=3,
            time_type="ind",
            show_plot=False,
        )
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
        # Reload as a 2D model with dynamics for fit_2d.
        file.add_time_dependence(
            target_model="single_glp",
            target_parameter="GLP_01_A",
            dynamics_yaml="models/file_time.yaml",
            dynamics_model=["MonoExpPos"],
        )
        file.fit_2d("single_glp", stages=1, try_ci=0)

        twod_slots = [s for s in project._fit_history if s.fit_type == "2d"]
        assert len(twod_slots) == 1
        slot = twod_slots[0]
        assert slot.observed.ndim == 2
        assert slot.observed.shape == slot.fit.shape
        # Residual reconstruction.
        residual = slot.observed - slot.fit
        assert slot.metrics["chi2_raw"] == pytest.approx(float(np.sum(residual**2)))


#
# --- MCMC payload capture ----------------------------------------------------
#


#
class TestMcmcPayload:
    """fit_wrapper's emcee outputs (emcee_fin/emcee_ci) flow into SavedFitSlot.mcmc.

    Without this wiring the slot's ``mcmc`` field stays None even when MCMC
    actually ran — see _mcmc_payload in utils/fit_io.py.
    """

    #
    @pytest.mark.slow
    def test_baseline_slot_captures_mcmc(self, tmp_path):
        from trspecfit.utils.lmfit import MC

        truth_project = make_project(name="truth")
        truth = _make_truth_file(truth_project)
        data = simulate_noisy(truth.model_active, noise_level=0.01)

        project = make_project(name="fit")
        file = _make_fit_file(project, data, truth.energy, truth.time)
        file.define_baseline(
            time_start=0, time_stop=3, time_type="ind", show_plot=False
        )
        # nwalkers > 2 * n_params for emcee's red-blue move.
        mc = MC(use_mc=1, steps=20, nwalkers=32, burn=5, thin=1, workers=1)
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0, mc_settings=mc)

        slot = project._fit_history[0]
        assert slot.mcmc is not None
        assert set(slot.mcmc.keys()) == {
            "flatchain",
            "ci",
            "lnsigma",
            "acceptance_fraction",
        }
        assert slot.mcmc["flatchain"] is not None
        assert slot.mcmc["ci"] is not None
        assert slot.mcmc["lnsigma"] is not None
        # emcee's acceptance fraction is per-walker.
        acceptance = slot.mcmc["acceptance_fraction"]
        assert acceptance is not None  # type guard
        assert acceptance.shape == (32,)
        # MCMC settings land in the fit_settings provenance.
        assert slot.fit_settings is not None  # type guard
        assert slot.fit_settings["mc"]["steps"] == 20
        assert slot.fit_settings["mc"]["nwalkers"] == 32
        assert slot.fit_settings["mc"]["burn"] == 5

        # acceptance_fraction survives the archive round-trip.
        archive_path = tmp_path / "mcmc.fit.h5"
        project.save_fits(archive_path, show_output=0)
        loaded_results = FitResults.load(archive_path)
        loaded = next(iter(loaded_results))
        assert loaded.mcmc is not None  # type guard
        np.testing.assert_array_equal(loaded.mcmc["acceptance_fraction"], acceptance)

        # ... and the slot-backed accessor serves it from the loaded archive,
        # including the persisted lnsigma noise scale.
        mcmc_res = loaded_results.get_mcmc(file="fit", fit_type="baseline")
        assert mcmc_res.acceptance_fraction is not None  # type guard
        np.testing.assert_array_equal(mcmc_res.acceptance_fraction, acceptance)
        assert not mcmc_res.table.empty
        assert not mcmc_res.flatchain.empty
        assert mcmc_res.lnsigma == pytest.approx(slot.mcmc["lnsigma"])

        # plot_mcmc reproduces the fit-time diagnostics from the persisted
        # payload — live history and loaded archive alike.
        import matplotlib.pyplot as plt

        n_figs = len(plt.get_fignums())
        file.plot_mcmc(fit_type="baseline", show_plot=False)
        loaded_results.plot_mcmc(file="fit", fit_type="baseline", show_plot=False)
        assert len(plt.get_fignums()) == n_figs

    #
    @pytest.mark.slow
    def test_weighted_mcmc_has_no_lnsigma(self):
        """__lnsigma only enters lmfit's log-probability for unweighted
        sampling — a weighted run (is_weighted=True) must not add and sample
        a likelihood-free nuisance dimension."""

        from trspecfit.utils.lmfit import MC

        truth_project = make_project(name="truth")
        truth = _make_truth_file(truth_project)
        data = simulate_noisy(truth.model_active, noise_level=0.01)

        project = make_project(name="fit")
        file = _make_fit_file(project, data, truth.energy, truth.time)
        file.define_baseline(
            time_start=0, time_stop=3, time_type="ind", show_plot=False
        )
        mc = MC(
            use_mc=1,
            steps=20,
            nwalkers=32,
            burn=5,
            thin=1,
            workers=1,
            is_weighted=True,
        )
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0, mc_settings=mc)

        slot = project._fit_history[0]
        assert slot.mcmc is not None  # type guard
        assert slot.mcmc["lnsigma"] is None
        assert "__lnsigma" not in slot.mcmc["flatchain"].columns
        assert "__lnsigma" not in list(slot.mcmc["ci"].iloc[:, 0])
        res = project.results.get_mcmc(file="fit", fit_type="baseline")
        assert res.lnsigma is None
        assert "__lnsigma" not in res.flatchain.columns

    #
    def test_baseline_slot_mcmc_none_when_mcmc_skipped(self):
        project, _ = _setup_baseline_fit()  # try_ci=0, no MCMC
        slot = project._fit_history[0]
        assert slot.mcmc is None


#
# --- fit_settings provenance ---------------------------------------------------
#


#
class TestFitSettingsProvenance:
    """Every fit type records its optimizer configuration in the slot."""

    #
    def test_baseline_slot_records_fit_settings(self):
        project, _ = _setup_baseline_fit()  # stages=2, try_ci=0
        slot = project._fit_history[0]
        # backend is the *effective* evaluator: the 1D baseline fit under
        # the default spec_fun_str="fit_model_gir" runs the compiled plan.
        assert slot.fit_settings == {
            "stages": 2,
            "fit_alg_1": "Nelder",
            "fit_alg_2": "leastsq",
            "backend": "fit_model_gir",
            "try_ci": 0,
        }
        # Non-sbs slots carry no sbs-only payloads.
        assert slot.params_meta is None
        assert slot.params_stderr is None

    #
    def test_fit_settings_records_custom_algorithms(self):
        truth_project = make_project(name="truth")
        truth = _make_truth_file(truth_project)
        data = simulate_noisy(truth.model_active, noise_level=0.01)
        project = make_project(name="fit")
        file = _make_fit_file(project, data, truth.energy, truth.time)
        file.define_baseline(
            time_start=0, time_stop=3, time_type="ind", show_plot=False
        )
        file.fit_baseline(
            model_name="single_glp", stages=1, fit_alg_1="leastsq", try_ci=0
        )
        settings = project._fit_history[0].fit_settings
        assert settings is not None  # type guard
        assert settings["fit_alg_1"] == "leastsq"
        assert settings["stages"] == 1


#
# --- slot-backed get_* accessors ----------------------------------------------
#


#
class TestSlotBackedAccessors:
    """FitResults.get_parameters / get_correlations / get_confidence_intervals /
    get_mcmc read the latest matching SavedFitSlot; the File.get_* methods
    are thin sugar delegating with file=self."""

    #
    def test_file_sugar_matches_fitresults_accessor(self):
        project, file = _setup_baseline_fit()
        via_file = file.get_parameters(fit_type="baseline")
        via_results = project.results.get_parameters(file=file, fit_type="baseline")
        pd.testing.assert_frame_equal(via_file, via_results)
        # ... and both match the slot payload.
        pd.testing.assert_frame_equal(via_file, project._fit_history[0].params)

    #
    def test_returned_frame_is_a_copy(self):
        """Accessors hand out copies — mutating the return value must not
        desynchronize the persisted slot."""

        project, file = _setup_baseline_fit()
        df = file.get_parameters(fit_type="baseline")
        df.loc[0, "value"] = -999.0
        assert project._fit_history[0].params.loc[0, "value"] != -999.0

    #
    def test_get_mcmc_payload_is_a_copy(self):
        """get_mcmc must not alias the slot's stored arrays/frames —
        np.asarray on an ndarray is a no-copy passthrough (regression)."""

        import dataclasses

        project, _ = _setup_baseline_fit()
        payload = {
            "flatchain": pd.DataFrame({"GLP_01_A": [1.0, 2.0]}),
            "ci": pd.DataFrame({"par[v]/sigma[>]": ["GLP_01_A"], "best fit": [1.5]}),
            "lnsigma": None,
            "acceptance_fraction": np.array([0.3, 0.4]),
        }
        slot = dataclasses.replace(project._fit_history[0], mcmc=payload)
        res = FitResults(slots=[slot]).get_mcmc(fit_type="baseline")

        assert res.acceptance_fraction is not None  # type guard
        res.acceptance_fraction[0] = -1.0
        res.flatchain.loc[0, "GLP_01_A"] = -999.0
        res.table.loc[0, "best fit"] = -999.0
        np.testing.assert_array_equal(
            payload["acceptance_fraction"], np.array([0.3, 0.4])
        )
        assert payload["flatchain"].loc[0, "GLP_01_A"] == 1.0
        assert payload["ci"].loc[0, "best fit"] == 1.5

    #
    def test_latest_slot_wins_after_refit(self):
        project, file = _setup_baseline_fit()
        assert file.data_base is not None  # type guard
        # Refit against rescaled data: same (file, model, fit_type, selection)
        # → a second slot appends, and the accessors must serve the newer one.
        file.data_base = file.data_base * 1.5
        file.fit_baseline(model_name="single_glp", stages=2, try_ci=0)
        assert len(project._fit_history) == 2

        df = file.get_parameters(fit_type="baseline")
        pd.testing.assert_frame_equal(df, project._fit_history[-1].params)
        first_values = project._fit_history[0].params["value"].to_numpy()
        assert not np.allclose(df["value"].to_numpy(), first_values)

    #
    def test_get_correlations_raises_without_covariance(self):
        """A slot with correl=None (covariance-less optimizer, project joint
        fit) must produce a clear error, not a fabricated identity matrix."""

        import dataclasses

        project, _ = _setup_baseline_fit()
        slot = dataclasses.replace(project._fit_history[0], correl=None)
        results = FitResults(slots=[slot])
        with pytest.raises(ValueError, match="reported no covariance"):
            results.get_correlations(fit_type="baseline")

    #
    def test_get_mcmc_serves_lnsigma(self):
        """The persisted lnsigma noise scale reaches the MCMCResult — the
        accessor previously dropped it (lossy for every fit type)."""

        import dataclasses

        payload = {
            "flatchain": pd.DataFrame({"GLP_01_A": [1.0, 2.0]}),
            "ci": None,
            "lnsigma": -2.0,
            "acceptance_fraction": None,
        }
        slot = dataclasses.replace(_slot_stub(), mcmc=payload)
        res = FitResults(slots=[slot]).get_mcmc(fit_type="baseline")
        assert res.lnsigma == -2.0

    #
    def test_get_mcmc_tolerates_missing_acceptance(self):
        """A payload may carry acceptance_fraction=None (the sampler
        reported none); get_mcmc must still serve table/flatchain."""

        import dataclasses

        project, _ = _setup_baseline_fit()
        payload = {
            "flatchain": pd.DataFrame({"GLP_01_A": [1.0, 2.0]}),
            "ci": None,
            "lnsigma": None,
            "acceptance_fraction": None,
        }
        slot = dataclasses.replace(project._fit_history[0], mcmc=payload)
        res = FitResults(slots=[slot]).get_mcmc(fit_type="baseline")
        assert res.acceptance_fraction is None
        assert res.table.empty
        assert list(res.flatchain.columns) == ["GLP_01_A"]


#
# --- Project.results snapshot semantics ---------------------------------------
#


#
class TestResultsSnapshot:
    #
    def test_results_returns_fresh_wrapper(self):
        project, _ = _setup_baseline_fit()
        r1 = project.results
        r2 = project.results
        assert r1 is not r2  # fresh wrapper per access
        assert len(r1) == len(r2) == 1

    #
    def test_returned_results_is_frozen_against_subsequent_fits(self):
        """A captured FitResults does not see new history entries."""

        truth_project = make_project(name="truth")
        truth = _make_truth_file(truth_project)
        data = simulate_clean(truth.model_active)

        project = make_project(name="fit")
        file = _make_fit_file(project, data, truth.energy, truth.time)
        file.define_baseline(
            time_start=0,
            time_stop=3,
            time_type="ind",
            show_plot=False,
        )
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
        r1 = project.results
        assert len(r1) == 1
        # Run a second fit (different selection -> new slot).
        file.fit_spectrum(
            "single_glp",
            time_point=10,
            time_type="ind",
            stages=1,
            try_ci=0,
            show_plot=False,
        )
        # r1 still sees only the first slot.
        assert len(r1) == 1
        # New access reflects both.
        assert len(project.results) == 2


#
# --- FitResults query API -----------------------------------------------------
#


#
def _stub_optimization_hash(*, input_files, fit_type, model_structure):
    """Real hash over a one-parameter stub table (shared by the stubs)."""

    return compute_optimization_hash(
        input_files_json=input_files,
        fit_type=fit_type,
        model_structure_json=model_structure,
        parameter_metadata=[("p", 0.0, 1.0, True, None)],
        initial_state=np.array([[0.5]]),
        optimizer_settings_json=encode_optimizer_settings(
            stages=1, fit_alg_1="leastsq", fit_alg_2="leastsq", backend="fit_model_mcp"
        ),
    )


#
def _slot_stub(
    *,
    file_name="f1",
    model_name="m",
    fit_type="baseline",
    metrics=None,
    fit_view_sha256="z",
    selection=None,
    sigma_data=float("nan"),
    noise_type=None,
    sigma_source="user_supplied",
    sigma_type="constant",
    input_files=None,
    joint_ref=None,
    optimization_hash=None,
):
    """Build a minimal SavedFitSlot for query-API tests (no real fit).

    The identity chain uses the real hash functions over a stub parameter
    table, so distinct (file, model, fit_type, selection) stubs get
    distinct handles. Joint-projection stubs pass a shared
    ``optimization_hash`` (siblings differ only through ``file_name``).
    ``sigma_data`` defaults to ``NaN`` (file had no sigma set); pass a
    positive number to exercise the σ-calibrated code paths.
    ``noise_type`` follows from ``sigma_data`` when omitted
    (``"gaussian"`` if finite, ``"unknown"`` otherwise).
    """

    if selection is None:
        selection = (
            {"base_t_ind": [0, 1], "e_lim": None} if fit_type == "baseline" else {}
        )
    selection_json = build_selection_json(fit_type, **selection)
    if input_files is None:
        input_files = encode_input_files(
            scope="file", entries=[(file_name, "0" * 64, selection_json)]
        )
    model_structure = encode_model_structure([(file_name, [model_name], [])])
    if optimization_hash is None:
        optimization_hash = _stub_optimization_hash(
            input_files=input_files, fit_type=fit_type, model_structure=model_structure
        )
    handle = compute_slot_handle(
        optimization_hash=optimization_hash, file_name=file_name
    )
    sigma_data_f = float(sigma_data)
    is_unset = not np.isfinite(sigma_data_f)
    sigma_eff = (
        float("nan")
        if is_unset
        else _compute_sigma_eff(fit_type, selection, sigma_data_f)
    )
    if noise_type is None:
        noise_type = "unknown" if is_unset else "gaussian"
    if metrics is None:
        metrics = {
            "chi2_raw": 0.0,
            "chi2_red_raw": 0.0,
            "chi2": float("nan") if is_unset else 0.0,
            "chi2_red": float("nan") if is_unset else 0.0,
            "r2": 1.0,
            "aic": 0.0,
            "bic": 0.0,
        }
    return SavedFitSlot(
        handle=handle,
        optimization_hash=optimization_hash,
        input_files=input_files,
        model_structure=model_structure,
        fit_view_sha256=fit_view_sha256,
        file_name=file_name,
        model_name=model_name,
        fit_type=fit_type,
        selection=selection,
        selection_json=selection_json,
        params=pd.DataFrame(),
        metrics=metrics,
        observed=np.zeros(3),
        fit=np.zeros(3),
        fit_alg="leastsq",
        timestamp="2026-04-30T00:00:00+00:00",
        noise_type=noise_type,
        sigma_source=sigma_source,
        sigma_type=sigma_type,
        sigma_data=sigma_data_f,
        sigma_eff=sigma_eff,
        joint_ref=joint_ref,
    )


#
def _joint_record_stub(
    *,
    model_name="m",
    file_names=("f1", "f2"),
    mcmc=None,
    timestamp="2026-04-30T00:00:00+00:00",
):
    """Build a minimal JointFitResult for query-API tests (no real fit)."""

    names = sorted(file_names)
    selection = {"e_lim": None, "t_lim": None}
    selection_json = build_selection_json("2d", **selection)
    input_files = encode_input_files(
        scope="project", entries=[(n, "0" * 64, selection_json) for n in names]
    )
    model_structure = encode_model_structure([(n, [model_name], []) for n in names])
    optimization_hash = _stub_optimization_hash(
        input_files=input_files, fit_type="2d", model_structure=model_structure
    )
    projections = tuple(
        JointFitProjection(
            parameter_map={},
            slot=_slot_stub(
                file_name=name,
                model_name=model_name,
                fit_type="2d",
                selection=selection,
                input_files=input_files,
                joint_ref=optimization_hash,
                optimization_hash=optimization_hash,
            ),
        )
        for name in names
    )
    return JointFitResult(
        model_name=model_name,
        optimization_hash=optimization_hash,
        input_files=input_files,
        model_structure=model_structure,
        projections=projections,
        params=pd.DataFrame(),
        metrics={},
        fit_alg="leastsq",
        fit_settings={},
        timestamp=timestamp,
        mcmc=mcmc,
    )


#
class TestFitResultsJointQueryAPI:
    """find_joint / get_joint / plot_joint_mcmc read the joint records
    carried alongside the per-file slots; iteration and len() stay
    slot-only (a joint record would otherwise count one optimization
    N+1 times)."""

    #
    def test_find_joint_filters_in_history_order(self):
        r1 = _joint_record_stub(model_name="m1")
        r2 = _joint_record_stub(model_name="m2")
        r3 = _joint_record_stub(model_name="m1")
        results = FitResults(slots=[], joint=[r1, r2, r3])
        assert results.find_joint() == [r1, r2, r3]
        assert results.find_joint(model="m1") == [r1, r3]
        assert results.find_joint(model="nope") == []

    #
    def test_find_joint_files_matches_full_participant_set(self):
        record = _joint_record_stub(file_names=("b", "a"))
        results = FitResults(slots=[], joint=[record])
        # canonicalized: order and duplicates don't matter
        assert results.find_joint(files=["b", "a"]) == [record]
        assert results.find_joint(files=["a", "b", "a"]) == [record]
        # a strict subset or superset does not match
        assert results.find_joint(files="a") == []
        assert results.find_joint(files=["a", "b", "c"]) == []

    #
    def test_find_joint_files_accepts_name_objects(self):
        record = _joint_record_stub(file_names=("f1",))

        class _Named:
            name = "f1"

        results = FitResults(slots=[], joint=[record])
        assert results.find_joint(files=_Named()) == [record]

    #
    def test_get_joint_raises_on_zero_and_multiple(self):
        record = _joint_record_stub(model_name="m1")
        twin = _joint_record_stub(model_name="m1")
        results = FitResults(slots=[], joint=[record, twin])
        with pytest.raises(LookupError, match="2 joint fit records"):
            results.get_joint(model="m1")
        with pytest.raises(LookupError, match="No joint fit record"):
            results.get_joint(model="nope")
        only = FitResults(slots=[], joint=[record])
        assert only.get_joint(model="m1") is record

    #
    def test_iteration_and_len_stay_slot_only(self):
        slot = _slot_stub()
        record = _joint_record_stub()
        results = FitResults(slots=[slot], joint=[record])
        assert len(results) == 1
        assert list(results) == [slot]
        assert "1 joint fit" in repr(results)

    #
    def test_plot_joint_mcmc_no_record_raises(self):
        results = FitResults(slots=[])
        with pytest.raises(ValueError, match="No project-level joint fit"):
            results.plot_joint_mcmc()

    #
    def test_plot_joint_mcmc_without_chain_raises(self):
        results = FitResults(slots=[], joint=[_joint_record_stub()])
        with pytest.raises(ValueError, match="No MCMC results for the joint fit"):
            results.plot_joint_mcmc()

    #
    def test_plot_joint_mcmc_renders_latest_chain(self):
        import matplotlib.pyplot as plt

        from trspecfit.utils.lmfit import MCMCResult

        n = 30
        mcmc = MCMCResult(
            table=pd.DataFrame(
                {"par[v]/sigma[>]": ["tau", "__lnsigma"], "best fit": [5.0, -2.0]}
            ),
            flatchain=pd.DataFrame(
                {
                    "tau": np.linspace(4.9, 5.1, n),
                    "__lnsigma": np.linspace(-2.1, -1.9, n),
                }
            ),
            acceptance_fraction=np.full(8, 0.35),
            lnsigma=-2.0,
        )
        older = _joint_record_stub()
        latest = _joint_record_stub(mcmc=mcmc)
        results = FitResults(slots=[], joint=[older, latest])
        n_figs = len(plt.get_fignums())
        # latest matching record wins; suppressed figures are closed again
        results.plot_joint_mcmc(show_plot=False)
        assert len(plt.get_fignums()) == n_figs


#
class TestFitResultsQueryAPI:
    #
    def test_files_and_models_unique_in_order(self):
        slots = [
            _slot_stub(file_name="A", model_name="m1"),
            _slot_stub(file_name="B", model_name="m1"),
            _slot_stub(file_name="A", model_name="m2"),
        ]
        r = FitResults(slots=slots)
        assert r.files() == ["A", "B"]
        assert r.models() == ["m1", "m2"]
        assert r.models(file="A") == ["m1", "m2"]
        assert r.models(file="B") == ["m1"]

    #
    def test_find_filters_combine(self):
        slots = [
            _slot_stub(file_name="A", model_name="m1", fit_type="baseline"),
            _slot_stub(file_name="A", model_name="m2", fit_type="baseline"),
            _slot_stub(file_name="A", model_name="m1", fit_type="2d"),
        ]
        r = FitResults(slots=slots)
        assert len(r.find(model="m1")) == 2
        assert len(r.find(model="m1", fit_type="baseline")) == 1

    #
    def test_get_raises_on_zero_or_multi(self):
        slots = [
            _slot_stub(file_name="A", model_name="m1", fit_type="baseline"),
            _slot_stub(file_name="A", model_name="m1", fit_type="baseline"),
        ]
        r = FitResults(slots=slots)
        with pytest.raises(LookupError, match="2 slots match"):
            r.get(file="A", model="m1", fit_type="baseline")
        with pytest.raises(LookupError, match="No slot matches"):
            r.get(file="A", model="m_missing", fit_type="baseline")

    #
    def test_iteration(self):
        slots = [_slot_stub(file_name=f"f{i}") for i in range(3)]
        r = FitResults(slots=slots)
        assert len(list(r)) == 3


#
class TestFitResultsCompareModels:
    """Tests for FitResults.compare_models: scalar, sbs aggregation, checks."""

    #
    @staticmethod
    def _scalar_metrics(*, chi2_red_raw, r2, aic, bic, chi2_raw=None):
        """Build a metrics dict with the 7-key schema.

        Calibrated ``chi2`` / ``chi2_red`` are populated as NaN — slots
        built via this helper represent the "no σ set on file" case.
        Tests that need calibrated values should pass ``sigma_data`` to
        ``_slot_stub`` and build the per-key dict by hand.
        """

        chi2_raw_v = float(chi2_raw) if chi2_raw is not None else float(chi2_red_raw)
        return {
            "chi2_raw": chi2_raw_v,
            "chi2_red_raw": float(chi2_red_raw),
            "chi2": float("nan"),
            "chi2_red": float("nan"),
            "r2": float(r2),
            "aic": float(aic),
            "bic": float(bic),
        }

    #
    def test_default_returns_columns_and_one_row_per_slot(self):
        slots = [
            _slot_stub(
                file_name="A",
                model_name="m1",
                fit_type="baseline",
                metrics=self._scalar_metrics(
                    chi2_red_raw=1.5, r2=0.9, aic=10.0, bic=12.0
                ),
            ),
            _slot_stub(
                file_name="A",
                model_name="m2",
                fit_type="baseline",
                metrics=self._scalar_metrics(
                    chi2_red_raw=0.8, r2=0.95, aic=8.0, bic=10.0
                ),
            ),
        ]
        df = FitResults(slots=slots).compare_models()
        assert list(df.columns) == [
            "file",
            "model",
            "fit_type",
            "handle",
            "selection_json",
            "chi2_red_raw",
            "r2",
            "aic",
            "bic",
        ]
        assert len(df) == 2
        assert set(df["model"]) == {"m1", "m2"}
        assert df.loc[df["model"] == "m1", "aic"].iloc[0] == 10.0
        assert df.loc[df["model"] == "m2", "aic"].iloc[0] == 8.0

    #
    def test_filters_by_file_models_and_fit_type(self):
        slots = [
            _slot_stub(
                file_name="A",
                model_name="m1",
                fit_type="baseline",
                metrics=self._scalar_metrics(chi2_red_raw=1.0, r2=0.9, aic=1, bic=1),
            ),
            _slot_stub(
                file_name="B",
                model_name="m1",
                fit_type="baseline",
                metrics=self._scalar_metrics(chi2_red_raw=1.0, r2=0.9, aic=1, bic=1),
            ),
            _slot_stub(
                file_name="A",
                model_name="m2",
                fit_type="2d",
                selection={"e_lim": None, "t_lim": None},
                metrics=self._scalar_metrics(chi2_red_raw=1.0, r2=0.9, aic=1, bic=1),
            ),
        ]
        r = FitResults(slots=slots)
        assert len(r.compare_models(file="A")) == 2
        assert len(r.compare_models(file="A", models=["m1"])) == 1
        assert len(r.compare_models(fit_type="baseline")) == 2
        assert len(r.compare_models(fit_type=["baseline", "2d"])) == 3

    #
    def test_custom_metrics_subset(self):
        slot = _slot_stub(
            file_name="A",
            model_name="m1",
            fit_type="baseline",
            metrics=self._scalar_metrics(
                chi2_raw=2.0,
                chi2_red_raw=0.5,
                r2=0.99,
                aic=5.0,
                bic=7.0,
            ),
        )
        df = FitResults(slots=[slot]).compare_models(metrics=["chi2_raw", "r2"])
        assert list(df.columns) == [
            "file",
            "model",
            "fit_type",
            "handle",
            "selection_json",
            "chi2_raw",
            "r2",
        ]
        assert df["chi2_raw"].iloc[0] == 2.0
        assert df["r2"].iloc[0] == 0.99

    #
    def test_unknown_metric_raises_keyerror(self):
        slot = _slot_stub(
            metrics=self._scalar_metrics(chi2_red_raw=1, r2=1, aic=1, bic=1),
        )
        with pytest.raises(KeyError, match="bogus"):
            FitResults(slots=[slot]).compare_models(metrics=["bogus"])

    #
    def test_sbs_aggregation_modes(self):
        # No σ → calibrated columns are absent from the default; assertions
        # target the raw column. (See TestFitResultsCompareModelsSigmaColumns
        # for the σ-calibrated equivalents.)
        per_slice = {
            "chi2_raw": np.array([1.0, 2.0, 3.0]),
            "chi2_red_raw": np.array([0.1, 0.2, 0.3]),
            "chi2": np.array([float("nan")] * 3),
            "chi2_red": np.array([float("nan")] * 3),
            "r2": np.array([0.9, 0.8, 0.95]),
            "aic": np.array([10.0, 20.0, 30.0]),
            "bic": np.array([12.0, 22.0, 32.0]),
        }
        slot = _slot_stub(
            file_name="A",
            model_name="m_sbs",
            fit_type="sbs",
            selection={"e_lim": None, "t_lim": None},
            metrics=per_slice,
        )
        r = FitResults(slots=[slot])

        df_med = r.compare_models(sbs_aggregation="median")
        assert df_med["chi2_red_raw"].iloc[0] == pytest.approx(0.2)
        assert df_med["aic"].iloc[0] == pytest.approx(20.0)

        df_mean = r.compare_models(sbs_aggregation="mean")
        assert df_mean["chi2_red_raw"].iloc[0] == pytest.approx(0.2)
        assert df_mean["r2"].iloc[0] == pytest.approx((0.9 + 0.8 + 0.95) / 3)

        df_sum = r.compare_models(sbs_aggregation="sum")
        assert df_sum["aic"].iloc[0] == pytest.approx(60.0)
        assert df_sum["bic"].iloc[0] == pytest.approx(66.0)
        # chi2_red_raw in sum mode is aggregate-reduced-chi-square:
        # Σchi2_raw / ΣDoF with DoF_i = chi2_raw_i / chi2_red_raw_i = [10, 10, 10],
        # so aggregate = 6 / 30 = 0.2 (not Σ chi2_red_raw = 0.6).
        assert df_sum["chi2_red_raw"].iloc[0] == pytest.approx(0.2)

    #
    def test_sbs_long_mode_emits_per_slice_rows(self):
        per_slice = {
            "chi2_raw": np.array([1.0, 2.0]),
            "chi2_red_raw": np.array([0.1, 0.2]),
            "chi2": np.array([float("nan"), float("nan")]),
            "chi2_red": np.array([float("nan"), float("nan")]),
            "r2": np.array([0.9, 0.8]),
            "aic": np.array([10.0, 20.0]),
            "bic": np.array([12.0, 22.0]),
        }
        sbs_slot = _slot_stub(
            file_name="A",
            model_name="m_sbs",
            fit_type="sbs",
            selection={"e_lim": None, "t_lim": None},
            metrics=per_slice,
        )
        baseline_slot = _slot_stub(
            file_name="B",
            model_name="m_base",
            fit_type="baseline",
            metrics=self._scalar_metrics(chi2_red_raw=0.5, r2=0.99, aic=5, bic=7),
        )
        df = FitResults(slots=[sbs_slot, baseline_slot]).compare_models(
            sbs_aggregation="long"
        )
        assert "slice_index" in df.columns
        sbs_rows = df[df["model"] == "m_sbs"]
        assert len(sbs_rows) == 2
        assert list(sbs_rows["slice_index"]) == [0, 1]
        assert sbs_rows["aic"].tolist() == [10.0, 20.0]
        baseline_rows = df.loc[df["model"] == "m_base"]
        assert len(baseline_rows) == 1
        assert pd.isna(baseline_rows["slice_index"].iloc[0])

    #
    def test_sbs_long_mode_is_slice_major(self):
        """Long-form interleaves models by slice so head() compares them."""

        per_slice = {
            "chi2_raw": np.array([1.0, 2.0, 3.0]),
            "chi2_red_raw": np.array([0.1, 0.2, 0.3]),
            "chi2": np.array([float("nan")] * 3),
            "chi2_red": np.array([float("nan")] * 3),
            "r2": np.array([0.9, 0.8, 0.7]),
            "aic": np.array([10.0, 20.0, 30.0]),
            "bic": np.array([12.0, 22.0, 32.0]),
        }
        slot_A = _slot_stub(
            file_name="F",
            model_name="sbsA",
            fit_type="sbs",
            selection={"e_lim": None, "t_lim": None},
            metrics=per_slice,
        )
        slot_B = _slot_stub(
            file_name="F",
            model_name="sbsB",
            fit_type="sbs",
            selection={"e_lim": None, "t_lim": None},
            metrics=per_slice,
        )
        # Slot order is A, B; slice-major output groups both models per slice.
        df = FitResults(slots=[slot_A, slot_B]).compare_models(
            fit_type="sbs", sbs_aggregation="long"
        )
        assert list(df["slice_index"]) == [0, 0, 1, 1, 2, 2]
        # Stable sort preserves slot order (A before B) within each slice.
        assert list(df["model"]) == ["sbsA", "sbsB"] * 3

    #
    def test_fit_view_mismatch_raises(self):
        """Two slots on same (file, fit_type) with different fit_view_sha256
        fit against different data views — their metrics must not be
        compared."""

        slots = [
            _slot_stub(
                file_name="A",
                model_name="m1",
                fit_type="baseline",
                metrics=self._scalar_metrics(chi2_red_raw=1, r2=1, aic=1, bic=1),
                fit_view_sha256="hash_A",
            ),
            _slot_stub(
                file_name="A",
                model_name="m2",
                fit_type="baseline",
                selection={"base_t_ind": [0, 5], "e_lim": None},
                metrics=self._scalar_metrics(chi2_red_raw=1, r2=1, aic=1, bic=1),
                fit_view_sha256="hash_B",
            ),
        ]
        r = FitResults(slots=slots)
        with pytest.raises(ValueError, match="fit_view_sha256"):
            r.compare_models(file="A", fit_type="baseline")

    #
    def test_fit_view_mismatch_allowed_across_different_fit_types(self):
        """Same file, different fit_type — views differ legitimately, no raise."""

        slots = [
            _slot_stub(
                file_name="A",
                model_name="m1",
                fit_type="baseline",
                metrics=self._scalar_metrics(chi2_red_raw=1, r2=1, aic=1, bic=1),
                fit_view_sha256="hash_A",
            ),
            _slot_stub(
                file_name="A",
                model_name="m1",
                fit_type="2d",
                selection={"e_lim": None, "t_lim": None},
                metrics=self._scalar_metrics(chi2_red_raw=2, r2=0.5, aic=5, bic=7),
                fit_view_sha256="hash_B",
            ),
        ]
        df = FitResults(slots=slots).compare_models(file="A")
        assert len(df) == 2

    #
    def test_fit_view_mismatch_allowed_across_different_files(self):
        """Same fit_type on different files — views differ legitimately."""

        slots = [
            _slot_stub(
                file_name="A",
                model_name="m1",
                fit_type="baseline",
                metrics=self._scalar_metrics(chi2_red_raw=1, r2=1, aic=1, bic=1),
                fit_view_sha256="hash_A",
            ),
            _slot_stub(
                file_name="B",
                model_name="m1",
                fit_type="baseline",
                metrics=self._scalar_metrics(chi2_red_raw=2, r2=0.5, aic=5, bic=7),
                fit_view_sha256="hash_B",
            ),
        ]
        df = FitResults(slots=slots).compare_models(fit_type="baseline")
        assert len(df) == 2

    #
    def test_replicate_files_group_by_name_not_view(self):
        """Two distinct files with byte-identical observations share a
        fit_view_sha256; grouping is by (file_name, fit_type), so a
        fit_type-wide compare keeps them as two rows rather than
        collapsing them into one comparability group."""

        slots = [
            _slot_stub(
                file_name="rep_A",
                model_name="m1",
                fit_type="baseline",
                metrics=self._scalar_metrics(chi2_red_raw=1, r2=1, aic=1, bic=1),
                fit_view_sha256="same_view",
            ),
            _slot_stub(
                file_name="rep_B",
                model_name="m1",
                fit_type="baseline",
                metrics=self._scalar_metrics(chi2_red_raw=2, r2=0.5, aic=5, bic=7),
                fit_view_sha256="same_view",
            ),
        ]
        df = FitResults(slots=slots).compare_models(fit_type="baseline")
        assert len(df) == 2
        assert set(df["file"]) == {"rep_A", "rep_B"}

    #
    def test_file_arg_accepts_object_with_name_attr(self):
        slot = _slot_stub(
            file_name="A",
            model_name="m1",
            metrics=self._scalar_metrics(chi2_red_raw=1, r2=1, aic=1, bic=1),
        )

        class _Stub:
            name = "A"

        df = FitResults(slots=[slot]).compare_models(file=_Stub())
        assert len(df) == 1
        assert df["file"].iloc[0] == "A"

    #
    def test_file_arg_invalid_type_raises(self):
        slot = _slot_stub()
        with pytest.raises(TypeError, match="file must be"):
            FitResults(slots=[slot]).compare_models(file=42)

    #
    def test_empty_match_returns_empty_dataframe(self):
        slot = _slot_stub(file_name="A", model_name="m1")
        df = FitResults(slots=[slot]).compare_models(file="missing")
        assert df.empty
        assert "model" in df.columns

    #
    def test_unknown_sbs_aggregation_raises(self):
        per_slice = {
            "chi2_raw": np.array([1.0]),
            "chi2_red_raw": np.array([0.1]),
            "chi2": np.array([float("nan")]),
            "chi2_red": np.array([float("nan")]),
            "r2": np.array([0.9]),
            "aic": np.array([10.0]),
            "bic": np.array([12.0]),
        }
        slot = _slot_stub(
            file_name="A",
            model_name="m_sbs",
            fit_type="sbs",
            selection={"e_lim": None, "t_lim": None},
            metrics=per_slice,
        )
        with pytest.raises(ValueError, match="unknown sbs_aggregation"):
            FitResults(slots=[slot]).compare_models(sbs_aggregation="bogus")  # type: ignore[arg-type]


#
class TestFitResultsCompareModelsSigmaColumns:
    """Stable column semantics around the file's persistent σ.

    Covers:

    - Default column set switches dynamically: 4 cols without σ, 6 cols
      with (``chi2_red_raw`` is present whenever any matched row defines
      it; ``sigma_eff`` + ``chi2_red`` appear only when at least one
      matched slot carries a finite σ).
    - Explicit request for ``chi2`` / ``chi2_red`` with no σ raises a clear
      ``KeyError`` pointing at ``file.set_sigma(...)`` / the raw column.
    - Sum-mode aggregation of ``chi2_red_raw`` and ``chi2_red`` uses
      ``Σnumerator / ΣDoF`` (not ``np.nansum`` of per-slice values) for the
      "≈ 1 for a good fit" reading.
    - The 4 sigma fields on the slot dataclass round-trip through both
      scalar and long output modes.
    """

    #
    @staticmethod
    def _scalar_metrics(*, chi2_red_raw, r2=0.9, aic=10.0, bic=12.0, sigma_eff=None):
        """Build a 7-key metrics dict.

        ``chi2_raw = chi2_red_raw`` for stub purposes (DoF=1 by construction);
        calibrated fields are computed from ``sigma_eff`` when given.
        """

        chi2_raw = float(chi2_red_raw)
        if sigma_eff is None or not np.isfinite(sigma_eff):
            chi2 = float("nan")
            chi2_red = float("nan")
        else:
            chi2 = chi2_raw / sigma_eff**2
            chi2_red = float(chi2_red_raw) / sigma_eff**2
        return {
            "chi2_raw": chi2_raw,
            "chi2_red_raw": float(chi2_red_raw),
            "chi2": chi2,
            "chi2_red": chi2_red,
            "r2": float(r2),
            "aic": float(aic),
            "bic": float(bic),
        }

    #
    def test_default_columns_without_sigma(self):
        """No σ on any slot → calibrated columns are absent from the default."""

        slot = _slot_stub(
            file_name="A",
            model_name="m",
            fit_type="2d",
            selection={"e_lim": None, "t_lim": None},
            metrics=self._scalar_metrics(chi2_red_raw=0.05),
        )
        df = FitResults(slots=[slot]).compare_models()
        assert list(df.columns) == [
            "file",
            "model",
            "fit_type",
            "handle",
            "selection_json",
            "chi2_red_raw",
            "r2",
            "aic",
            "bic",
        ]
        assert "chi2_red" not in df.columns
        assert "sigma_eff" not in df.columns
        assert "chi2" not in df.columns

    #
    def test_default_columns_with_sigma(self):
        """σ on the slot → default set adds sigma_eff + chi2_red."""

        slot = _slot_stub(
            file_name="A",
            model_name="m",
            fit_type="2d",
            selection={"e_lim": None, "t_lim": None},
            sigma_data=0.2,
            metrics=self._scalar_metrics(chi2_red_raw=0.04, sigma_eff=0.2),
        )
        df = FitResults(slots=[slot]).compare_models()
        assert list(df.columns) == [
            "file",
            "model",
            "fit_type",
            "handle",
            "selection_json",
            "chi2_red_raw",
            "sigma_eff",
            "chi2_red",
            "r2",
            "aic",
            "bic",
        ]
        assert df["sigma_eff"].iloc[0] == pytest.approx(0.2)
        assert df["chi2_red"].iloc[0] == pytest.approx(0.04 / 0.2**2)
        assert df["chi2_red_raw"].iloc[0] == pytest.approx(0.04)

    #
    def test_baseline_sigma_eff_uses_n_avg_correction(self):
        """Slot stub mirrors the live ``_compute_sigma_eff`` correction."""

        slot = _slot_stub(
            file_name="A",
            model_name="m",
            fit_type="baseline",
            selection={"base_t_ind": [0, 5], "e_lim": None},
            sigma_data=0.2,
        )
        # _slot_stub computed sigma_eff = 0.2 / sqrt(5) at construction.
        expected = 0.2 / np.sqrt(5)
        assert slot.sigma_eff == pytest.approx(expected)
        df = FitResults(slots=[slot]).compare_models()
        assert df["sigma_eff"].iloc[0] == pytest.approx(expected)

    #
    def test_explicit_calibrated_request_without_sigma_raises(self):
        """``metrics=['chi2_red']`` with no σ → KeyError pointing at set_sigma."""

        slot = _slot_stub(
            file_name="A",
            model_name="m",
            fit_type="2d",
            selection={"e_lim": None, "t_lim": None},
            metrics=self._scalar_metrics(chi2_red_raw=0.05),
        )
        with pytest.raises(KeyError, match="file.set_sigma"):
            FitResults(slots=[slot]).compare_models(metrics=["chi2_red"])
        with pytest.raises(KeyError, match="file.set_sigma"):
            FitResults(slots=[slot]).compare_models(metrics=["chi2"])

    #
    def test_explicit_raw_request_works_without_sigma(self):
        """``metrics=['chi2_red_raw']`` works without σ — raw needs no calibration."""

        slot = _slot_stub(
            file_name="A",
            model_name="m",
            fit_type="2d",
            selection={"e_lim": None, "t_lim": None},
            metrics=self._scalar_metrics(chi2_red_raw=0.05),
        )
        df = FitResults(slots=[slot]).compare_models(metrics=["chi2_red_raw"])
        assert df["chi2_red_raw"].iloc[0] == pytest.approx(0.05)

    #
    def test_sigma_eff_broadcast_in_long_mode(self):
        """SbS in ``long`` mode: every slice row gets the slot's σ_eff."""

        per_slice = {
            "chi2_raw": np.array([1.0, 2.0, 3.0]),
            "chi2_red_raw": np.array([0.04, 0.05, 0.06]),
            "chi2": np.array([1.0 / 0.04, 2.0 / 0.04, 3.0 / 0.04]),
            "chi2_red": np.array([1.0, 1.25, 1.5]),
            "r2": np.array([0.9, 0.8, 0.85]),
            "aic": np.array([10.0, 20.0, 30.0]),
            "bic": np.array([12.0, 22.0, 32.0]),
        }
        slot = _slot_stub(
            file_name="A",
            model_name="m_sbs",
            fit_type="sbs",
            selection={"e_lim": None, "t_lim": None},
            metrics=per_slice,
            sigma_data=0.2,
        )
        df = FitResults(slots=[slot]).compare_models(sbs_aggregation="long")
        assert len(df) == 3
        # Per-slot scalar broadcast to every slice row.
        assert df["sigma_eff"].tolist() == [pytest.approx(0.2)] * 3

    #
    def test_sbs_sum_chi2_red_raw_aggregates_via_dof(self):
        """sum-mode ``chi2_red_raw`` = Σ chi2_raw / Σ DoF (not nansum)."""

        # DoF_i = chi2_raw_i / chi2_red_raw_i = [10, 15] → ΣDoF = 25, Σchi2_raw = 40.
        per_slice = {
            "chi2_raw": np.array([10.0, 30.0]),
            "chi2_red_raw": np.array([1.0, 2.0]),
            "chi2": np.array([float("nan"), float("nan")]),
            "chi2_red": np.array([float("nan"), float("nan")]),
            "r2": np.array([0.9, 0.8]),
            "aic": np.array([10.0, 20.0]),
            "bic": np.array([12.0, 22.0]),
        }
        slot = _slot_stub(
            file_name="A",
            model_name="m_sbs",
            fit_type="sbs",
            selection={"e_lim": None, "t_lim": None},
            metrics=per_slice,
        )
        df = FitResults(slots=[slot]).compare_models(
            sbs_aggregation="sum",
            metrics=["chi2_raw", "chi2_red_raw", "aic", "bic"],
        )
        assert df["chi2_red_raw"].iloc[0] == pytest.approx(40.0 / 25.0)
        # chi2_raw / aic / bic still nansum'd.
        assert df["chi2_raw"].iloc[0] == pytest.approx(40.0)
        assert df["aic"].iloc[0] == pytest.approx(30.0)
        assert df["bic"].iloc[0] == pytest.approx(34.0)

    #
    def test_sbs_sum_chi2_red_uses_calibrated_numerator(self):
        """sum-mode ``chi2_red`` = Σ chi2 / Σ DoF; equals chi2_red_raw / σ²."""

        sigma = 0.5
        per_slice_raw = 0.04
        n_slices = 4
        chi2_raw = np.full(n_slices, per_slice_raw * 100.0)  # DoF = 100 each
        chi2_red_raw = np.full(n_slices, per_slice_raw)
        chi2 = chi2_raw / sigma**2
        chi2_red = chi2_red_raw / sigma**2
        per_slice = {
            "chi2_raw": chi2_raw,
            "chi2_red_raw": chi2_red_raw,
            "chi2": chi2,
            "chi2_red": chi2_red,
            "r2": np.full(n_slices, 0.99),
            "aic": np.full(n_slices, -10.0),
            "bic": np.full(n_slices, -8.0),
        }
        slot = _slot_stub(
            file_name="A",
            model_name="m",
            fit_type="sbs",
            selection={"e_lim": None, "t_lim": None},
            metrics=per_slice,
            sigma_data=sigma,
        )
        df = FitResults(slots=[slot]).compare_models(sbs_aggregation="sum")
        # aggregate raw = per_slice_raw (constant per slice)
        assert df["chi2_red_raw"].iloc[0] == pytest.approx(per_slice_raw)
        # aggregate calibrated = per_slice_raw / σ²
        assert df["chi2_red"].iloc[0] == pytest.approx(per_slice_raw / sigma**2)


#
#
class TestResolveFitReference:
    """Prefix / label resolution semantics (B9 matrix rows).

    Real-fit end-to-end use (``get(handle=)``, ``select=``,
    ``drop_fits``) lives in ``test_fit_query.py``; these pin the
    resolver's contract on stubs with real handle chains.
    """

    #
    def test_unambiguous_prefix_resolves(self):
        a = _slot_stub(model_name="mA")
        b = _slot_stub(model_name="mB")
        assert a.handle != b.handle
        got = resolve_fit_reference(a.handle[:8], slots=[a, b])
        assert got is a

    #
    def test_full_handle_resolves(self):
        a = _slot_stub()
        assert resolve_fit_reference(a.handle, slots=[a]) is a

    #
    def test_ambiguous_prefix_raises(self):
        """The empty prefix matches every handle — the degenerate ambiguity."""

        a = _slot_stub(model_name="mA")
        b = _slot_stub(model_name="mB")
        with pytest.raises(LookupError, match="ambiguous"):
            resolve_fit_reference("", slots=[a, b])

    #
    def test_prefix_matching_nothing_raises(self):
        a = _slot_stub()
        with pytest.raises(LookupError, match="No fit matches"):
            resolve_fit_reference("no-such-fit", slots=[a])

    #
    def test_exact_label_resolves(self):
        a = _slot_stub(model_name="mA")
        b = _slot_stub(model_name="mB")
        set_fit_label(a, "final")
        assert resolve_fit_reference("final", slots=[a, b]) is a

    #
    def test_duplicate_label_is_ambiguous(self):
        a = _slot_stub(model_name="mA")
        b = _slot_stub(model_name="mB")
        set_fit_label(a, "dup")
        set_fit_label(b, "dup")
        with pytest.raises(LookupError, match="ambiguous"):
            resolve_fit_reference("dup", slots=[a, b])

    #
    def test_labels_flag_restricts_to_prefixes(self):
        """``handle=`` accessors resolve prefixes only, never labels."""

        a = _slot_stub()
        set_fit_label(a, "final")
        with pytest.raises(LookupError, match="No fit matches"):
            resolve_fit_reference("final", slots=[a], labels=False)

    #
    def test_same_handle_reruns_resolve_to_latest_entry(self):
        a1 = _slot_stub()
        a2 = _slot_stub()
        assert a1.handle == a2.handle
        assert resolve_fit_reference(a1.handle[:8], slots=[a1, a2]) is a2

    #
    def test_joint_hash_prefix_resolves_the_record(self):
        record = _joint_record_stub()
        slots = [p.slot for p in record.projections]
        got = resolve_fit_reference(
            record.optimization_hash[:8], slots=slots, joint_records=[record]
        )
        assert got is record

    #
    def test_reserved_and_empty_labels_rejected(self):
        a = _slot_stub()
        with pytest.raises(ValueError, match="reserved"):
            set_fit_label(a, "latest")
        with pytest.raises(ValueError, match="non-empty"):
            set_fit_label(a, "")


#
#
class TestSelectSnapshotSlots:
    """``select=`` keyword semantics on collapsed snapshots (B9 rows)."""

    #
    @staticmethod
    def _variant(
        *, e_lo, metrics, sigma_data=float("nan"), model_name="m", fit_view="z"
    ):
        """One stub variant; ``e_lo`` varies the selection to mint a
        distinct handle within the same (file, model, fit_type) group."""

        return _slot_stub(
            model_name=model_name,
            selection={"base_t_ind": [0, 1], "e_lim": [e_lo, 20]},
            metrics=metrics,
            sigma_data=sigma_data,
            fit_view_sha256=fit_view,
        )

    #
    @staticmethod
    def _metrics(**overrides):
        base = {
            "chi2_raw": 1.0,
            "chi2_red_raw": 1.0,
            "chi2": float("nan"),
            "chi2_red": float("nan"),
            "r2": 0.9,
            "aic": 0.0,
            "bic": 0.0,
        }
        base.update(overrides)
        return base

    #
    def test_all_keeps_every_variant(self):
        a = self._variant(e_lo=0, metrics=self._metrics())
        b = self._variant(e_lo=1, metrics=self._metrics())
        assert select_snapshot_slots([a, b], select="all") == [a, b]

    #
    def test_latest_picks_newest_run_per_group(self):
        a = self._variant(e_lo=0, metrics=self._metrics())
        b = self._variant(e_lo=1, metrics=self._metrics())
        c = self._variant(e_lo=0, metrics=self._metrics(), model_name="m2")
        got = select_snapshot_slots([a, b, c], select="latest", history_order=[a, b, c])
        # One winner per (file, model, fit_type) group: b is m's newest
        # run, c is m2's only one.
        assert got == [b, c]

    #
    def test_best_lower_wins_on_chi2_red_raw(self):
        a = self._variant(e_lo=0, metrics=self._metrics(chi2_red_raw=1.0))
        b = self._variant(e_lo=1, metrics=self._metrics(chi2_red_raw=2.0))
        got = select_snapshot_slots([a, b], select="best", by="chi2_red_raw")
        assert got == [a]

    #
    def test_best_chi2_red_ranks_by_distance_to_one(self):
        """chi2_red is never minimized: overfitting drives it *below* 1,
        so smallest-wins would select the most overfit variant. The
        winner is the fit closest to the noise floor (|x − 1|)."""

        a = self._variant(e_lo=0, metrics=self._metrics(chi2_red=0.5), sigma_data=1.0)
        b = self._variant(e_lo=1, metrics=self._metrics(chi2_red=1.2), sigma_data=1.0)
        got = select_snapshot_slots([a, b], select="best", by="chi2_red")
        assert got == [b]  # |1.2 − 1| beats |0.5 − 1|; min-wins would pick a

    #
    def test_raw_chi2_and_r2_are_not_offered(self):
        """Principles §Pruning and selection: a fit with more free
        parameters almost always wins on raw χ² or r² while being the
        worse model — they are not selection criteria."""

        a = self._variant(e_lo=0, metrics=self._metrics())
        for by in ("chi2_raw", "chi2", "r2"):
            with pytest.raises(ValueError, match="not an offered"):
                select_snapshot_slots([a], select="best", by=by)

    #
    def test_best_sbs_ranks_by_per_slice_median(self):
        a = _slot_stub(
            fit_type="sbs",
            selection={"e_lim": [0, 20]},
            metrics={"chi2_red_raw": np.array([1.0, 3.0, 1.2])},
        )
        b = _slot_stub(
            fit_type="sbs",
            selection={"e_lim": [1, 20]},
            metrics={"chi2_red_raw": np.array([2.0, 2.1, 2.2])},
        )
        got = select_snapshot_slots([a, b], select="best", by="chi2_red_raw")
        assert got == [a]  # median 1.2 beats 2.1

    #
    def test_best_sigma_scaled_over_mixed_sigma_raises(self):
        """The reversed-ranking fixture: σ-scaled ranking would pick the
        raw loser, so the mix must raise instead of ranking."""

        a = self._variant(
            e_lo=0,
            metrics=self._metrics(
                chi2_raw=100.0, chi2_red_raw=1.0, chi2=100.0, chi2_red=100.0
            ),
            sigma_data=1.0,
        )
        b = self._variant(
            e_lo=1,
            metrics=self._metrics(
                chi2_raw=120.0, chi2_red_raw=1.2, chi2=1.2, chi2_red=1.2
            ),
            sigma_data=10.0,
        )
        with pytest.raises(ValueError, match="sigma_eff values"):
            select_snapshot_slots([a, b], select="best", by="chi2_red")
        # The raw ranking stays available and picks the true winner.
        got = select_snapshot_slots([a, b], select="best", by="chi2_red_raw")
        assert got == [a]

    #
    def test_best_mixed_fit_views_refuse_to_rank(self):
        """Metrics of different fit views (changed limits or correction
        state) must never be ranked against one another — the same rule
        compare_models enforces. 'latest' stays available: it picks by
        recency, not by metric."""

        a = self._variant(e_lo=0, metrics=self._metrics(aic=1.0), fit_view="v1")
        b = self._variant(e_lo=1, metrics=self._metrics(aic=2.0), fit_view="v2")
        with pytest.raises(ValueError, match="fit views"):
            select_snapshot_slots([a, b], select="best", by="aic")
        got = select_snapshot_slots([a, b], select="latest", history_order=[a, b])
        assert got == [b]

    #
    def test_best_metric_undefined_everywhere_raises(self):
        a = self._variant(e_lo=0, metrics=self._metrics(aic=float("nan")))
        b = self._variant(e_lo=1, metrics=self._metrics(aic=float("nan")))
        with pytest.raises(ValueError, match="undefined"):
            select_snapshot_slots([a, b], select="best", by="aic")

    #
    def test_best_requires_and_validates_by(self):
        a = self._variant(e_lo=0, metrics=self._metrics())
        with pytest.raises(ValueError, match="requires by="):
            select_snapshot_slots([a], select="best")
        with pytest.raises(ValueError, match="not an offered"):
            select_snapshot_slots([a], select="best", by="sigma_eff")
        with pytest.raises(ValueError, match="unknown select keyword"):
            select_snapshot_slots([a], select="bets")


#
#
class TestCompareModelsSigmaTiers:
    """σ gates comparison within a shared view (schema plan §σ tiers)."""

    #
    @staticmethod
    def _mixed_sigma_pair():
        """Two models on one file/view: σ=1 vs σ=10."""

        a = _slot_stub(
            model_name="m1",
            sigma_data=1.0,
            metrics={
                "chi2_raw": 100.0,
                "chi2_red_raw": 1.0,
                "chi2": 100.0,
                "chi2_red": 100.0,
                "r2": 0.9,
                "aic": 1.0,
                "bic": 1.0,
            },
        )
        b = _slot_stub(
            model_name="m2",
            sigma_data=10.0,
            metrics={
                "chi2_raw": 120.0,
                "chi2_red_raw": 1.2,
                "chi2": 1.2,
                "chi2_red": 1.2,
                "r2": 0.88,
                "aic": 2.0,
                "bic": 2.0,
            },
        )
        return a, b

    #
    def test_mixed_sigma_drops_calibrated_from_defaults(self):
        a, b = self._mixed_sigma_pair()
        df = FitResults(slots=[a, b]).compare_models()
        assert "chi2_red" not in df.columns
        assert "sigma_eff" in df.columns  # the mix stays visible
        assert "chi2_red_raw" in df.columns

    #
    def test_mixed_sigma_explicit_request_raises_naming_both(self):
        a, b = self._mixed_sigma_pair()
        with pytest.raises(ValueError, match=r"1\.0.*10\.0"):
            FitResults(slots=[a, b]).compare_models(metrics=["chi2_red"])

    #
    def test_raw_metrics_still_compare_across_the_mix(self):
        a, b = self._mixed_sigma_pair()
        df = FitResults(slots=[a, b]).compare_models(metrics=["chi2_raw", "r2"])
        assert len(df) == 2
        assert list(df["chi2_raw"]) == [100.0, 120.0]

    #
    def test_finite_sigma_next_to_unset_is_not_a_conflict(self):
        a, _ = self._mixed_sigma_pair()
        c = _slot_stub(model_name="m3")  # no sigma set
        df = FitResults(slots=[a, c]).compare_models()
        assert "chi2_red" in df.columns
        assert np.isnan(df["chi2_red"].iloc[1])


#
#
class TestCompareModelsColumnDrop:
    """All-undefined default columns are dropped; explicit ones render."""

    #
    @staticmethod
    def _projection_like(model_name="m"):
        """Metrics as on a joint projection: count-dependent ones NaN."""

        return _slot_stub(
            model_name=model_name,
            fit_type="2d",
            selection={"e_lim": None, "t_lim": None},
            metrics={
                "chi2_raw": 5.0,
                "chi2_red_raw": float("nan"),
                "chi2": float("nan"),
                "chi2_red": float("nan"),
                "r2": 0.9,
                "aic": float("nan"),
                "bic": float("nan"),
            },
        )

    #
    def test_all_nan_default_columns_dropped(self):
        df = FitResults(slots=[self._projection_like()]).compare_models()
        assert "aic" not in df.columns
        assert "bic" not in df.columns
        assert "chi2_red_raw" not in df.columns
        assert "r2" in df.columns

    #
    def test_explicit_request_keeps_all_nan_column(self):
        df = FitResults(slots=[self._projection_like()]).compare_models(metrics=["aic"])
        assert "aic" in df.columns
        assert np.isnan(df["aic"].iloc[0])

    #
    def test_one_defined_row_keeps_the_column(self):
        full = _slot_stub(
            model_name="m2", fit_type="2d", selection={"e_lim": None, "t_lim": None}
        )
        df = FitResults(slots=[self._projection_like(), full]).compare_models()
        assert "aic" in df.columns
        assert np.isnan(df["aic"].iloc[0]) and df["aic"].iloc[1] == 0.0

    #
    def test_handle_column_is_the_short_prefix(self):
        slot = _slot_stub()
        df = FitResults(slots=[slot]).compare_models()
        assert df["handle"].iloc[0] == slot.handle[:8]


#
#
class TestJointComparabilityKey:
    """The joint comparability key is a tuple of pairs, never a set."""

    #
    def test_identical_views_keep_multiplicity(self):
        record = _joint_record_stub()
        key = joint_comparability(record)
        # Both stub projections share fit_view_sha256 "z"; a set of view
        # hashes would collapse them into one entry and make the two-file
        # joint fit look one-file.
        assert key == (("f1", "z"), ("f2", "z"))


#
#
class TestDiffAndVariantsInputs:
    """Cross-model diffs and identity-input visibility (review rows)."""

    #
    @staticmethod
    def _params(names, values):
        return pd.DataFrame(
            {
                "name": names,
                "value": values,
                "stderr": [0.1] * len(names),
                "init_value": values,
                "min": [0.0] * len(names),
                "max": [10.0] * len(names),
                "vary": [True] * len(names),
                "expr": [None] * len(names),
            }
        )

    #
    def test_cross_model_diff_reports_added_and_removed_params(self):
        """Disjoint parameter sets (Gauss vs GLP) diff cleanly: the
        missing side renders NA in both input and result sections —
        never a crash."""

        a = dataclasses.replace(
            _slot_stub(model_name="gauss"),
            params=self._params(["Gauss_01_A", "Gauss_01_x0"], [1.0, 2.0]),
        )
        b = dataclasses.replace(
            _slot_stub(model_name="glp"),
            params=self._params(["GLP_01_A", "GLP_01_m"], [1.5, 0.3]),
        )
        d = FitResults(slots=[a, b]).diff(a.handle[:8], b.handle[:8])
        rows = {(r["section"], r["field"]) for _, r in d.iterrows()}
        assert ("identity", "model") in rows
        assert ("result", "Gauss_01_A") in rows
        assert ("result", "GLP_01_m") in rows
        added_row = d[d["field"] == "GLP_01_m"].iloc[0]
        assert pd.isna(added_row[a.handle[:8]])
        assert added_row[b.handle[:8]] == 0.3

    #
    def test_correction_values_show_as_digests(self):
        """Two variants differing only in dark *values* show a differing
        input column — a boolean applied/not-applied cell would hide the
        difference and leave the variant table empty."""

        base = _slot_stub()
        a = dataclasses.replace(base, dark=np.array([0.1, 0.2, 0.3]))
        b = dataclasses.replace(base, dark=np.array([0.4, 0.5, 0.6]))
        df = FitResults(slots=[a, b]).variants(file="f1", model="m")
        assert "dark" in df.columns
        digest_a, digest_b = df["dark"].iloc[0], df["dark"].iloc[1]
        assert digest_a != digest_b
        assert len(digest_a) == len(digest_b) == 8
        # A no-correction slot renders as missing, not as a digest.
        df2 = FitResults(slots=[base, a]).variants(file="f1", model="m")
        assert pd.isna(df2["dark"].iloc[0])

    #
    def test_joint_diff_decodes_input_files(self):
        """Per-file version stamps (correction/content state) and
        selections are visible in a bundle-level diff — the combined
        table alone would hide them."""

        ja = dataclasses.replace(
            _joint_record_stub(), params=self._params(["tau"], [2.0])
        )
        selection_json = build_selection_json("2d", e_lim=None, t_lim=None)
        changed = encode_input_files(
            scope="project",
            entries=[
                ("f1", "0" * 64, selection_json),
                ("f2", "1" * 64, selection_json),
            ],
        )
        jb = dataclasses.replace(ja, optimization_hash="f" * 64, input_files=changed)
        r = FitResults(slots=[], joint=[ja, jb])
        d = r.diff(ja.optimization_hash[:8], "ffffffff")
        rows = {(row["section"], row["field"]) for _, row in d.iterrows()}
        assert ("input", "f2.version_stamp") in rows
        stamp_row = d[d["field"] == "f2.version_stamp"].iloc[0]
        assert stamp_row[ja.optimization_hash[:8]] == "0" * 8
        assert stamp_row["ffffffff"] == "1" * 8


#
class TestPlotFitAPI:
    """FitResults.plot_fit / plot_param_evolution and the File.* sugar."""

    #
    def test_plot_fit_1d_uses_real_axes_and_config(self):
        import matplotlib.pyplot as plt

        _, file = _setup_baseline_fit()
        file.plot_fit(fit_type="baseline")  # show under Agg keeps the fig live
        fig = plt.gcf()
        try:
            line_x = fig.axes[0].lines[0].get_xdata()
            np.testing.assert_array_equal(line_x, np.asarray(file.energy))
            assert fig.axes[1].get_xlabel() == file.p.plot_config.x_label
        finally:
            plt.close("all")

    #
    def test_plot_fit_2d_passes_real_axes(self, monkeypatch):
        from unittest.mock import MagicMock

        from trspecfit import fitlib

        mock_2d = MagicMock()
        monkeypatch.setattr(fitlib, "plt_fit_res_2d", mock_2d)

        project, file = _setup_baseline_fit()
        file.add_time_dependence(
            target_model="single_glp",
            target_parameter="GLP_01_A",
            dynamics_yaml="models/file_time.yaml",
            dynamics_model=["MonoExpPos"],
        )
        file.fit_2d("single_glp", stages=1, try_ci=0)
        mock_2d.reset_mock()

        project.results.plot_fit(file=file, fit_type="2d", show_plot=False)
        assert mock_2d.call_count == 1
        kwargs = mock_2d.call_args.kwargs
        np.testing.assert_array_equal(kwargs["x"], np.asarray(file.energy))
        np.testing.assert_array_equal(kwargs["y"], np.asarray(file.time))
        assert kwargs["save_img"] == -2  # show_plot=False

    #
    def test_plot_fit_from_loaded_archive_has_axes(self, tmp_path):
        import matplotlib.pyplot as plt

        project, file = _setup_baseline_fit()
        archive_path = tmp_path / "plot.fit.h5"
        project.save_fits(archive_path, show_output=0)

        loaded = FitResults.load(archive_path)
        loaded.plot_fit(file=file.name, fit_type="baseline")
        fig = plt.gcf()
        try:
            line_x = fig.axes[0].lines[0].get_xdata()
            np.testing.assert_array_equal(line_x, np.asarray(file.energy))
        finally:
            plt.close("all")

    #
    def test_plot_fit_1d_renders_components_when_present(self):
        """components/component_names present -> one line +
        one fill_between per component, plus the observed/fit lines."""

        import dataclasses

        import matplotlib.pyplot as plt

        obs = np.array([1.0, 2.0, 3.0, 4.0])
        comp_a = np.array([0.6, 1.2, 1.8, 2.4])
        comp_b = np.array([0.4, 0.8, 1.2, 1.6])
        slot = dataclasses.replace(
            _slot_stub(),
            observed=obs,
            fit=comp_a + comp_b,
            components=np.stack([comp_a, comp_b], axis=0),
            component_names=["peak_a", "peak_b"],
        )
        fig = FitResults._plot_fit_1d(slot, energy=None, config=None, show_plot=False)
        try:
            ax_fit = fig.axes[0]
            labels = [line.get_label() for line in ax_fit.lines]
            assert "peak_a" in labels
            assert "peak_b" in labels
            assert "fit" in labels
            assert "observed" in labels
            # one fill_between per component
            assert len(ax_fit.collections) == 2
            fit_line = next(line for line in ax_fit.lines if line.get_label() == "fit")
            np.testing.assert_array_equal(fit_line.get_ydata(), comp_a + comp_b)
        finally:
            plt.close(fig)

    #
    def test_plot_fit_1d_falls_back_to_lean_when_components_none(self):
        """Slots without persisted components keep the sum-only rendering."""

        import matplotlib.pyplot as plt

        slot = _slot_stub()
        assert slot.components is None
        fig = FitResults._plot_fit_1d(slot, energy=None, config=None, show_plot=False)
        try:
            ax_fit = fig.axes[0]
            labels = [line.get_label() for line in ax_fit.lines]
            assert labels == ["observed", "fit"]
            assert len(ax_fit.collections) == 0
        finally:
            plt.close(fig)

    #
    def test_plot_fit_1d_full_range_masks_outside_roi_and_draws_boundary(self):
        """full_range overrides: NaN outside the ROI leaves a gap (never a
        fabricated value); roi draws dashed boundary lines on both panels."""

        import matplotlib.pyplot as plt

        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        obs_full = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        fit_full = np.array([np.nan, np.nan, 2.9, 3.9, np.nan, np.nan])
        slot = _slot_stub()
        fig = FitResults._plot_fit_1d(
            slot,
            energy=x,
            config=None,
            show_plot=False,
            observed=obs_full,
            fit=fit_full,
            roi=[2, 4],
        )
        try:
            ax_fit, ax_res = fig.axes
            fit_line = next(line for line in ax_fit.lines if line.get_label() == "fit")
            np.testing.assert_array_equal(fit_line.get_ydata(), fit_full)
            # residual is NaN wherever fit is NaN (obs - NaN = NaN), no
            # special-cased padding logic needed; residual is plotted first,
            # before the boundary vlines
            res_line = ax_res.lines[0]
            np.testing.assert_array_equal(
                np.isnan(res_line.get_ydata()), np.isnan(fit_full)
            )
            # dashed boundary lines at the ROI edges, on both panels
            dashed_x = {
                float(line.get_xdata()[0])
                for line in ax_fit.lines
                if line.get_linestyle() == "--"
            }
            assert dashed_x == {x[2], x[3]}
        finally:
            plt.close(fig)

    #
    def test_plot_fit_1d_full_range_defaults_match_cropped_view(self):
        """Omitting the overrides (full_range=False path) renders exactly
        the slot's own cropped arrays — unchanged from before full_range
        existed."""

        import dataclasses

        import matplotlib.pyplot as plt

        obs = np.array([1.0, 2.0, 3.0])
        fit = np.array([1.1, 1.9, 3.2])
        slot = dataclasses.replace(_slot_stub(), observed=obs, fit=fit)
        fig = FitResults._plot_fit_1d(slot, energy=None, config=None, show_plot=False)
        try:
            ax_fit, ax_res = fig.axes
            fit_line = next(line for line in ax_fit.lines if line.get_label() == "fit")
            np.testing.assert_array_equal(fit_line.get_ydata(), fit)
            assert not any(line.get_linestyle() == "--" for line in ax_fit.lines)
        finally:
            plt.close(fig)

    #
    def test_plot_fit_1d_renders_initial_guess_when_present_and_shown(self):
        """fit_ini + show_init=True (default) -> dotted-gold "initial
        guess" line, drawn alongside observed/fit."""

        import dataclasses

        import matplotlib.pyplot as plt

        obs = np.array([1.0, 2.0, 3.0])
        fit = np.array([1.1, 1.9, 3.2])
        fit_ini = np.array([0.5, 1.0, 1.5])
        slot = dataclasses.replace(_slot_stub(), observed=obs, fit=fit, fit_ini=fit_ini)
        fig = FitResults._plot_fit_1d(slot, energy=None, config=None, show_plot=False)
        try:
            ax_fit = fig.axes[0]
            labels = [line.get_label() for line in ax_fit.lines]
            assert "initial guess" in labels
            ini_line = next(
                line for line in ax_fit.lines if line.get_label() == "initial guess"
            )
            np.testing.assert_array_equal(ini_line.get_ydata(), fit_ini)
            assert ini_line.get_linestyle() == ":"
            assert ini_line.get_color() == "#FFD700"
        finally:
            plt.close(fig)

    #
    def test_plot_fit_1d_omits_initial_guess_when_show_init_false(self):
        """A persisted fit_ini is not drawn when show_init=False."""

        import dataclasses

        import matplotlib.pyplot as plt

        slot = dataclasses.replace(_slot_stub(), fit_ini=np.array([0.5, 1.0, 1.5]))
        fig = FitResults._plot_fit_1d(
            slot, energy=None, config=None, show_plot=False, show_init=False
        )
        try:
            labels = [line.get_label() for line in fig.axes[0].lines]
            assert "initial guess" not in labels
        finally:
            plt.close(fig)

    #
    def test_plot_fit_1d_omits_initial_guess_when_absent(self):
        """show_init=True (default) with no persisted fit_ini draws nothing
        extra (fit_ini is None on joint-projection slots)."""

        import matplotlib.pyplot as plt

        slot = _slot_stub()
        assert slot.fit_ini is None
        fig = FitResults._plot_fit_1d(slot, energy=None, config=None, show_plot=False)
        try:
            labels = [line.get_label() for line in fig.axes[0].lines]
            assert "initial guess" not in labels
        finally:
            plt.close(fig)

    #
    def test_plot_fit_1d_full_range_pads_fit_ini_with_nan(self):
        """full_range mode: an explicit fit_ini override (NaN outside the
        fit window, mirroring fit/components) renders with the same gaps."""

        import matplotlib.pyplot as plt

        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        obs_full = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        fit_full = np.array([np.nan, np.nan, 2.9, 3.9, np.nan, np.nan])
        fit_ini_full = np.array([np.nan, np.nan, 0.8, 1.6, np.nan, np.nan])
        slot = _slot_stub()
        fig = FitResults._plot_fit_1d(
            slot,
            energy=x,
            config=None,
            show_plot=False,
            observed=obs_full,
            fit=fit_full,
            fit_ini=fit_ini_full,
            roi=[2, 4],
        )
        try:
            ax_fit = fig.axes[0]
            ini_line = next(
                line for line in ax_fit.lines if line.get_label() == "initial guess"
            )
            np.testing.assert_array_equal(ini_line.get_ydata(), fit_ini_full)
        finally:
            plt.close(fig)

    #
    @staticmethod
    def _fake_sbs_results(*, vary=(True, False, True)):
        """FitResults around a synthetic SbS slot (wide params + metadata)."""

        import dataclasses

        project, _ = _setup_baseline_fit()
        wide = pd.DataFrame(
            {
                "A": [1.0, 2.0, 3.0],
                "B": [4.0, 5.0, 6.0],
                "C": [7.0, 8.0, 9.0],
            }
        )
        meta = pd.DataFrame(
            {
                "name": ["A", "B", "C"],
                "vary": list(vary),
                "min": [0.0] * 3,
                "max": [10.0] * 3,
                "expr": [None] * 3,
            }
        )
        fake = dataclasses.replace(
            project._fit_history[0], fit_type="sbs", params=wide, params_meta=meta
        )
        return FitResults(slots=[fake])

    #
    def test_plot_param_evolution_defaults_to_varied(self, monkeypatch):
        from unittest.mock import MagicMock

        from trspecfit import fitlib

        mock_pars = MagicMock()
        monkeypatch.setattr(fitlib, "plt_fit_res_pars", mock_pars)

        results = self._fake_sbs_results(vary=(True, False, True))
        results.plot_param_evolution(show_plot=False)
        assert mock_pars.call_count == 1
        kwargs = mock_pars.call_args.kwargs
        assert list(kwargs["df"].columns) == ["A", "C"]  # varied only
        # No axes provider on this FitResults -> index fallback.
        np.testing.assert_array_equal(kwargs["x"], np.arange(3))

    #
    def test_plot_param_evolution_explicit_and_missing_params(self, monkeypatch):
        from unittest.mock import MagicMock

        from trspecfit import fitlib

        mock_pars = MagicMock()
        monkeypatch.setattr(fitlib, "plt_fit_res_pars", mock_pars)

        results = self._fake_sbs_results()
        results.plot_param_evolution(params=["B"], show_plot=False)
        assert list(mock_pars.call_args.kwargs["df"].columns) == ["B"]
        with pytest.raises(KeyError, match="not in this SbS fit"):
            results.plot_param_evolution(params=["nope"], show_plot=False)

    #
    def test_plot_param_evolution_all_fixed_plots_nothing(self, monkeypatch):
        from unittest.mock import MagicMock

        from trspecfit import fitlib

        mock_pars = MagicMock()
        monkeypatch.setattr(fitlib, "plt_fit_res_pars", mock_pars)

        results = self._fake_sbs_results(vary=(False, False, False))
        results.plot_param_evolution(show_plot=False)
        assert mock_pars.call_count == 0

    #
    def test_plot_residuals_uses_energy_axis_with_provider(self):
        project, file = _setup_baseline_fit()
        fig = project.results.plot_residuals(file=file.name, show_plot=False)
        assert fig.axes[1].get_xlabel() == "energy"


#
class TestPlotMcmc:
    """FitResults.plot_mcmc renders diagnostics from the slot's mcmc payload
    (synthetic slots here; the live-fit + loaded-archive path is covered in
    TestMcmcPayload)."""

    #
    @staticmethod
    def _mcmc_results(*, with_acceptance=True):
        import dataclasses

        n = 40
        flatchain = pd.DataFrame(
            {
                "GLP_01_A": np.linspace(0.9, 1.1, n),
                "__lnsigma": np.linspace(-2.1, -1.9, n),
            }
        )
        ci = pd.DataFrame(
            {
                "par[v]/sigma[>]": ["GLP_01_A", "__lnsigma"],
                "-1.0": [0.95, -2.05],
                "best fit": [1.0, -2.0],
                "+1.0": [1.05, -1.95],
            }
        )
        mcmc = {
            "flatchain": flatchain,
            "ci": ci,
            "lnsigma": -2.0,
            "acceptance_fraction": (np.full(8, 0.4) if with_acceptance else None),
        }
        return FitResults(slots=[dataclasses.replace(_slot_stub(), mcmc=mcmc)])

    #
    def test_renders_acceptance_and_corner(self):
        import matplotlib.pyplot as plt

        results = self._mcmc_results()
        plt.close("all")
        results.plot_mcmc(file="f1", fit_type="baseline")  # show under Agg
        try:
            assert len(plt.get_fignums()) == 2
        finally:
            plt.close("all")

    #
    def test_skips_acceptance_when_absent(self):
        import matplotlib.pyplot as plt

        results = self._mcmc_results(with_acceptance=False)
        plt.close("all")
        results.plot_mcmc(file="f1", fit_type="baseline")
        try:
            assert len(plt.get_fignums()) == 1  # corner only
        finally:
            plt.close("all")

    #
    def test_show_plot_false_leaves_no_figures(self):
        import matplotlib.pyplot as plt

        results = self._mcmc_results()
        n_figs = len(plt.get_fignums())
        results.plot_mcmc(file="f1", fit_type="baseline", show_plot=False)
        assert len(plt.get_fignums()) == n_figs

    #
    def test_raises_without_mcmc_payload(self):
        results = FitResults(slots=[_slot_stub()])
        with pytest.raises(ValueError, match="No MCMC results"):
            results.plot_mcmc(file="f1", fit_type="baseline", show_plot=False)


#
class TestFitResultsPlotResiduals:
    """Smoke tests for FitResults.plot_residuals — figure construction only."""

    #
    @staticmethod
    def _slot_with_arrays(
        *,
        file_name="A",
        model_name="m1",
        fit_type="baseline",
        observed,
        fit,
        selection=None,
    ):
        """Build a slot with custom observed/fit arrays for plotting."""

        import dataclasses

        slot = _slot_stub(
            file_name=file_name,
            model_name=model_name,
            fit_type=fit_type,
            selection=selection,
        )
        return dataclasses.replace(
            slot, observed=np.asarray(observed), fit=np.asarray(fit)
        )

    #
    def test_1d_fit_returns_figure(self):
        slot_a = self._slot_with_arrays(
            model_name="m1",
            observed=np.linspace(0, 1, 30),
            fit=np.linspace(0, 1, 30) + 0.05,
        )
        slot_b = self._slot_with_arrays(
            model_name="m2",
            observed=np.linspace(0, 1, 30),
            fit=np.linspace(0, 1, 30) - 0.02,
        )
        fig = FitResults(slots=[slot_a, slot_b]).plot_residuals(
            file="A", show_plot=False
        )
        assert fig is not None
        assert len(fig.axes) >= 4

    #
    def test_2d_fit_returns_figure(self):
        obs = np.random.RandomState(0).randn(8, 12)
        fit = obs + np.random.RandomState(1).randn(8, 12) * 0.1
        slot = self._slot_with_arrays(
            model_name="m_2d",
            fit_type="2d",
            selection={"e_lim": None, "t_lim": None},
            observed=obs,
            fit=fit,
        )
        fig = FitResults(slots=[slot]).plot_residuals(file="A", show_plot=False)
        assert fig is not None
        assert len(fig.axes) >= 1

    #
    def test_no_match_raises(self):
        slot = self._slot_with_arrays(
            observed=np.zeros(5),
            fit=np.zeros(5),
        )
        with pytest.raises(LookupError, match="No slots match"):
            FitResults(slots=[slot]).plot_residuals(file="missing", show_plot=False)

    #
    def test_mixed_fit_types_requires_disambiguation(self):
        slot_b = self._slot_with_arrays(
            model_name="m1",
            fit_type="baseline",
            observed=np.zeros(5),
            fit=np.zeros(5),
        )
        slot_2d = self._slot_with_arrays(
            model_name="m2",
            fit_type="2d",
            selection={"e_lim": None, "t_lim": None},
            observed=np.zeros((3, 5)),
            fit=np.zeros((3, 5)),
        )
        r = FitResults(slots=[slot_b, slot_2d])
        with pytest.raises(ValueError, match="span fit_types"):
            r.plot_residuals(file="A", show_plot=False)
        # Disambiguating works:
        fig = r.plot_residuals(file="A", fit_type="baseline", show_plot=False)
        assert fig is not None

    #
    def test_missing_file_arg_raises(self):
        slot = self._slot_with_arrays(observed=np.zeros(5), fit=np.zeros(5))
        with pytest.raises(ValueError, match="requires file"):
            FitResults(slots=[slot]).plot_residuals(
                file=None,
                show_plot=False,  # type: ignore[arg-type]
            )


#
# --- multi-fit history accumulation + snapshot-collapse on save -------------
#


#
class TestHistoryAccumulationAndSnapshot:
    """Multi-fit history accumulation, in-session multi-version visibility,
    and snapshot-collapse-on-save.

    Scenario: fit modelA-baseline, fit modelB-baseline, re-run
    modelA-baseline. History has *all three* slots; ``Project.results``
    exposes them. ``save_fits`` (snapshot mode) collapses exact re-runs —
    one slot per ``handle``, latest wins; distinct variants all survive.
    """

    #
    @staticmethod
    def _two_model_fit_file(project):
        """Build a fit file with two distinct energy models registered.

        Both ``single_glp`` and ``two_glp_expr_amplitude`` fit cleanly on
        the [82, 92] axis; quality-of-fit is irrelevant here — what matters
        is that both ``model_name`` strings produce valid baseline slots
        with distinct identities.
        """

        truth_project = make_project(name="truth_two_model")
        truth = File(
            parent_project=truth_project,
            name="truth",
            energy=np.linspace(82, 92, 30),
            time=np.linspace(-2, 10, 24),
        )
        truth.dim = 2
        truth.load_model(model_yaml="models/file_energy.yaml", model_info="single_glp")
        data = simulate_noisy(truth.model_active, noise_level=0.01)

        file = File(
            parent_project=project,
            name="fit_two_model",
            data=data,
            energy=truth.energy.copy(),
            time=truth.time.copy(),
        )
        file.load_model(model_yaml="models/file_energy.yaml", model_info="single_glp")
        file.load_model(
            model_yaml="models/file_energy.yaml",
            model_info="two_glp_expr_amplitude",
        )
        file.define_baseline(
            time_start=0, time_stop=3, time_type="ind", show_plot=False
        )
        return file

    #
    def test_history_holds_all_completed_fits(self):
        """fit modelA → fit modelB → refit modelA accumulates 3 slots."""

        project = make_project(name="acc")
        file = self._two_model_fit_file(project)

        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
        file.fit_baseline(model_name="two_glp_expr_amplitude", stages=1, try_ci=0)
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)

        assert len(project._fit_history) == 3
        names_in_order = [s.model_name for s in project._fit_history]
        assert names_in_order == [
            "single_glp",
            "two_glp_expr_amplitude",
            "single_glp",
        ]

    #
    def test_results_exposes_all_history_entries(self):
        """``Project.results`` mirrors ``_fit_history`` slot-for-slot."""

        project = make_project(name="acc_results")
        file = self._two_model_fit_file(project)

        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
        file.fit_baseline(model_name="two_glp_expr_amplitude", stages=1, try_ci=0)
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)

        results = project.results
        assert len(results) == 3
        # find() exposes both refits when narrowed to modelA.
        single_glp_slots = results.find(
            file=file.name, model="single_glp", fit_type="baseline"
        )
        assert len(single_glp_slots) == 2
        # The refit is a distinct variant (it seeds from the first fit's
        # output), so it keeps its own handle — nothing is hidden.
        assert single_glp_slots[0].handle != single_glp_slots[1].handle
        # The cross-model slot has a distinct optimization identity.
        cross = results.find(
            file=file.name,
            model="two_glp_expr_amplitude",
            fit_type="baseline",
        )
        assert len(cross) == 1
        assert cross[0].optimization_hash != single_glp_slots[0].optimization_hash

    #
    def test_save_fits_collapses_exact_reruns_to_latest(self, tmp_path):
        """Snapshot save keeps one slot per ``handle`` (latest wins).

        Only an exact re-run (same seed, settings, view, data) shares a
        handle — the seed is restored before the third fit to construct
        one. Distinct variants never collapse (see TestSelectionIdentity
        and the correction-variant tests).
        """

        project = make_project(name="acc_save")
        file = self._two_model_fit_file(project)
        model = next(m for m in file.models if m.name == "single_glp")
        seed = [p.value for p in model.lmfit_pars.values()]

        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
        file.fit_baseline(model_name="two_glp_expr_amplitude", stages=1, try_ci=0)
        model.update_value(seed)  # fits write back; restore the exact seed
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
        # _fit_history has 3; the two single_glp slots share a handle.
        assert len(project._fit_history) == 3
        assert project._fit_history[0].handle == project._fit_history[2].handle

        # Stamp the duplicate-key slots with deterministic sentinels so the
        # latest-wins assertion does not depend on second-resolution wall
        # clocks. ``_now_iso()`` is per-second, so two fits inside the same
        # second would silently weaken the assertion (both "earlier" and
        # "later" timestamps would compare equal). ``dataclasses.replace``
        # works on the frozen SavedFitSlot.
        from dataclasses import replace

        project._fit_history[0] = replace(
            project._fit_history[0], timestamp="2026-01-01T00:00:00+00:00"
        )
        project._fit_history[2] = replace(
            project._fit_history[2], timestamp="2026-01-01T00:00:01+00:00"
        )

        archive_path = tmp_path / "snapshot.fit.h5"
        project.save_fits(archive_path, show_output=0)
        loaded = FitResults.load(archive_path)
        # Snapshot collapses the duplicate-handle pair → 2 distinct slots.
        assert len(loaded) == 2
        handles_in_archive = {s.handle for s in loaded}
        assert handles_in_archive == {
            project._fit_history[0].handle,
            project._fit_history[1].handle,
        }

        # Latest-wins: collapse must keep the third fit (slot[2]), not the
        # first (slot[0]) — proved by the sentinel timestamp regardless of
        # clock resolution.
        loaded_single = next(s for s in loaded if s.model_name == "single_glp")
        assert loaded_single.timestamp == "2026-01-01T00:00:01+00:00"


#
# --- in-session collapse applies the archive collision rule ------------------
#


#
class TestCollapseCollisionRule:
    """Divergent fitted values under one handle raise at collapse;
    ``overwrite=True`` keeps the latest and warns — the same rule and the
    same flag as the archive boundary (fit_archive_principles.md §"One
    rule, both boundaries")."""

    #
    @staticmethod
    def _divergent_pair(value_second=2.0):
        """Two same-handle slots whose fitted values differ."""

        import dataclasses

        base = _slot_stub()
        first = dataclasses.replace(
            base,
            params=pd.DataFrame({"name": ["p"], "value": [1.0]}),
            timestamp="2026-01-01T00:00:00+00:00",
        )
        second = dataclasses.replace(
            base,
            params=pd.DataFrame({"name": ["p"], "value": [value_second]}),
            timestamp="2026-01-01T00:00:01+00:00",
        )
        return first, second

    #
    def test_divergent_same_handle_raises(self):
        first, second = self._divergent_pair()
        with pytest.raises(FileExistsError, match="not deterministic"):
            collapse_history_to_snapshot([first, second])

    #
    def test_divergent_same_handle_overwrite_keeps_latest_and_warns(self):
        first, second = self._divergent_pair()
        with pytest.warns(UserWarning, match="keeping the latest"):
            out = collapse_history_to_snapshot([first, second], overwrite=True)
        assert out == [second]

    #
    def test_agreeing_rerun_dedups_silently(self):
        """Within-tolerance values are the same optimum — no raise, no
        warning, latest kept."""

        import warnings

        first, second = self._divergent_pair(value_second=1.0 + 1e-9)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = collapse_history_to_snapshot([first, second])
        assert out == [second]

    #
    def test_divergence_hint_matches_seed_state(self):
        """Unseeded divergence points at pinning a seed; seeded
        divergence points at a stochastic fit_alg_2 instead of
        re-recommending the seed the user already supplied."""

        import dataclasses

        first, second = self._divergent_pair()
        with pytest.raises(FileExistsError, match="Pin an optimizer seed"):
            collapse_history_to_snapshot([first, second])
        seeded = [
            dataclasses.replace(s, fit_settings={"seed": 42}) for s in (first, second)
        ]
        with pytest.raises(FileExistsError, match="stochastic fit_alg_2"):
            collapse_history_to_snapshot(seeded)

    #
    def test_divergent_joint_records_follow_the_same_rule(self):
        import dataclasses

        base = _joint_record_stub(model_name="m")
        first = dataclasses.replace(
            base, params=pd.DataFrame({"name": ["p"], "value": [1.0]})
        )
        second = dataclasses.replace(
            base, params=pd.DataFrame({"name": ["p"], "value": [2.0]})
        )
        with pytest.raises(FileExistsError, match="not deterministic"):
            collapse_joint_history_to_snapshot([first, second])
        with pytest.warns(UserWarning, match="keeping the latest"):
            out = collapse_joint_history_to_snapshot([first, second], overwrite=True)
        assert out == [second]

    #
    def test_save_fits_divergent_rerun_raises_and_overwrite_resolves(self, tmp_path):
        """End-to-end: the raise and its resolution both come from the
        same ``save_fits`` call; nothing is recomputed and both runs stay
        in the in-session history."""

        import dataclasses

        project, file, model, seed = _fit_file_with_seed()
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
        model.update_value(seed)  # exact re-run: same handle
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
        # Force divergence under the shared handle — the in-session stand-in
        # for an unseeded stochastic optimizer (deterministic algorithms
        # cannot produce it).
        rerun = project._fit_history[1]
        diverged_params = rerun.params.copy()
        diverged_params["value"] = diverged_params["value"] * 1.5
        project._fit_history[1] = dataclasses.replace(rerun, params=diverged_params)

        archive_path = tmp_path / "divergent.fit.h5"
        with pytest.raises(FileExistsError, match="pass\\s+overwrite=True"):
            project.save_fits(archive_path, show_output=0)
        assert len(project._fit_history) == 2  # nothing dropped by the raise

        with pytest.warns(UserWarning, match="keeping the latest"):
            project.save_fits(archive_path, overwrite=True, show_output=0)
        loaded = FitResults.load(archive_path)
        (slot,) = iter(loaded)
        np.testing.assert_allclose(
            slot.params["value"].to_numpy(dtype=float),
            diverged_params["value"].to_numpy(dtype=float),
        )


#
# --- selection-identity: refits with different views → distinct slots --------
#


#
class TestSelectionIdentity:
    """Refits with different fit-view selections must produce distinct
    ``handle`` values and survive snapshot save as separate slots.

    Covers each fit_type's selection-identity field:

    - baseline: ``base_t_ind`` (time window averaged for ``data_base``)
    - sbs:      ``e_lim`` / ``t_lim``
    - 2d:       ``e_lim`` / ``t_lim``
    - spectrum: ``time_point`` is already covered in ``TestSpectrumSlot``
    """

    #
    @staticmethod
    def _basic_2d_fit_file(project):
        """1D-fittable 2D file with single_glp and a wide enough baseline."""

        truth_project = make_project(name="truth_sel")
        truth = _make_truth_file(truth_project)
        data = simulate_noisy(truth.model_active, noise_level=0.01)

        file = _make_fit_file(project, data, truth.energy, truth.time)
        return file

    #
    def test_baseline_refit_with_different_base_t_ind_distinct(self, tmp_path):
        """Different ``base_t_ind`` → distinct ``handle``; snapshot keeps both."""

        project = make_project(name="sel_base")
        file = self._basic_2d_fit_file(project)

        file.define_baseline(
            time_start=0, time_stop=3, time_type="ind", show_plot=False
        )
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
        first_handle = project._fit_history[0].handle

        file.define_baseline(
            time_start=0, time_stop=2, time_type="ind", show_plot=False
        )
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)

        handles = [s.handle for s in project._fit_history]
        assert handles[0] != handles[1]
        assert handles[0] == first_handle
        # selection captures the inclusive→exclusive index slice.
        assert project._fit_history[0].selection["base_t_ind"] == [0, 4]
        assert project._fit_history[1].selection["base_t_ind"] == [0, 3]

        # Snapshot save preserves both — no collapse since handles differ.
        archive_path = tmp_path / "base_t_ind.fit.h5"
        project.save_fits(archive_path, show_output=0)
        loaded = FitResults.load(archive_path)
        assert len(loaded) == 2
        assert {s.handle for s in loaded} == set(handles)

    #
    @pytest.mark.slow
    def test_sbs_refit_with_different_e_lim_distinct(self, tmp_path):
        """SbS refit with a different ``e_lim`` → distinct slots."""

        project = make_project(name="sel_sbs")
        project.spec_fun_str = "fit_model_mcp"
        file = self._basic_2d_fit_file(project)

        file.fit_slice_by_slice(
            "single_glp",
            n_workers=1,
            seed_source="model",
            seed_adapt=None,
            try_ci=0,
        )
        # Refit with a tighter e_lim. Set both index and absolute parallels.
        file.e_lim = [5, 25]
        file.e_lim_abs = [float(file.energy[5]), float(file.energy[24])]
        file.fit_slice_by_slice(
            "single_glp",
            n_workers=1,
            seed_source="model",
            seed_adapt=None,
            try_ci=0,
        )

        handles = [s.handle for s in project._fit_history]
        assert len(handles) == 2
        assert handles[0] != handles[1]
        # File constructor pre-fills e_lim with the full range via
        # set_fit_limits, so the first fit's selection is not None.
        assert project._fit_history[0].selection["e_lim"] == [0, len(file.energy)]
        assert project._fit_history[1].selection["e_lim"] == [5, 25]

        archive_path = tmp_path / "sbs_e_lim.fit.h5"
        project.save_fits(archive_path, show_output=0)
        loaded = FitResults.load(archive_path)
        assert len(loaded) == 2

    #
    @pytest.mark.slow
    def test_2d_refit_with_different_t_lim_distinct(self, tmp_path):
        """2D refit with a different ``t_lim`` → distinct slots."""

        project = make_project(name="sel_2d")
        file = self._basic_2d_fit_file(project)
        file.define_baseline(
            time_start=0, time_stop=3, time_type="ind", show_plot=False
        )
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
        # Add dynamics so fit_2d is valid.
        file.add_time_dependence(
            target_model="single_glp",
            target_parameter="GLP_01_A",
            dynamics_yaml="models/file_time.yaml",
            dynamics_model=["MonoExpPos"],
        )
        file.fit_2d("single_glp", stages=1, try_ci=0)
        # Refit with a tighter t_lim covering the post-trigger half.
        file.t_lim = [4, 24]
        file.t_lim_abs = [float(file.time[4]), float(file.time[23])]
        file.fit_2d("single_glp", stages=1, try_ci=0)

        twod_slots = [s for s in project._fit_history if s.fit_type == "2d"]
        assert len(twod_slots) == 2
        assert twod_slots[0].handle != twod_slots[1].handle
        # File constructor pre-fills t_lim with the full range; the second
        # fit narrows it. The two distinct t_lim values must produce two
        # distinct handles.
        assert twod_slots[0].selection["t_lim"] == [0, len(file.time)]
        assert twod_slots[1].selection["t_lim"] == [4, 24]

        archive_path = tmp_path / "2d_t_lim.fit.h5"
        project.save_fits(archive_path, fit_type="2d", show_output=0)
        loaded = FitResults.load(archive_path)
        assert len(loaded) == 2
