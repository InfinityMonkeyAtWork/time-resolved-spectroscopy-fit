"""
End-to-end tests for the B9 query layer on real fits (public API):

- ``FitResults.variants()`` — input table of one model's runs, constant
  columns suppressed.
- ``FitResults.diff()`` — pairwise input/output diff; bundle-level for
  joint fits.
- ``handle=`` accessor pinning, mutually exclusive with the filter trio.
- ``FitResults.set_label()`` — post-hoc labels: selection, escalation to the
  joint record, archive round-trip.
- ``select=`` / ``by=`` on ``save_fits`` / ``export_fits`` ("all" /
  "latest" / "best" / reference), including joint-bundle expansion.
- ``Project.drop_fits()`` — pruning slots and whole joint bundles.

Stub-level unit rows (prefix-resolution semantics, σ tiers,
``select_snapshot_slots`` ranking, comparability keys) live in
``test_fit_history.py``.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from _utils import make_project, simulate_noisy

from trspecfit import File, FitResults


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
def _two_variant_baseline():
    """(project, file, handle_A, handle_B): two baseline variants of one
    model in one (file, model, fit_type) group.

    Run A fits freely; run B restores the seed, then shifts ``GLP_01_x0``
    and fixes it (``vary=False``) — a distinct configuration under schema
    7 and a deterministically *worse* fit, so ``select="best"`` has an
    unambiguous winner.
    """

    truth_project = make_project(name="truth")
    truth = _make_truth_file(truth_project)
    data = simulate_noisy(truth.model_active, noise_level=0.01)

    project = make_project(name="fit")
    file = File(
        parent_project=project,
        name="fit",
        data=data,
        energy=truth.energy.copy(),
        time=truth.time.copy(),
    )
    file.load_model(
        model_yaml="models/file_energy.yaml",
        model_info="single_glp",
    )
    file.define_baseline(time_start=0, time_stop=3, time_type="ind", show_plot=False)
    model = next(m for m in file.models if m.name == "single_glp")
    seed = [p.value for p in model.lmfit_pars.values()]

    file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
    handle_a = project.results.find(
        file="fit", model="single_glp", fit_type="baseline"
    )[-1].handle

    model.update_value(seed)  # fits write back; restore the exact seed
    model.lmfit_pars["GLP_01_x0"].value += 0.5
    model.lmfit_pars["GLP_01_x0"].vary = False
    file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
    handle_b = project.results.find(
        file="fit", model="single_glp", fit_type="baseline"
    )[-1].handle

    assert handle_a != handle_b  # distinct configurations, distinct slots
    return project, file, handle_a, handle_b


#
def _build_joint_project(*, noise_level: float = 0.05):
    """Two-file project ready for ``Project.fit_2d`` (shared tau,
    per-file A) — the ``test_fit_archive_roundtrip`` fixture shape."""

    project = make_project(name="joint_query")
    for i, (amplitude, seed) in enumerate([(20.0, 42), (14.0, 43)]):
        truth_project = make_project(name="truth")
        truth = File(parent_project=truth_project)
        truth.energy = np.linspace(83, 87, 30)
        truth.time = np.linspace(-2, 10, 24)
        truth.dim = 2
        truth.load_model(
            model_yaml="models/project_energy.yaml", model_info="project_glp"
        )
        truth.add_time_dependence(
            target_model="project_glp",
            target_parameter="GLP_01_x0",
            dynamics_yaml="models/project_time.yaml",
            dynamics_model=["MonoExpProject"],
        )
        truth.model_active.lmfit_pars["GLP_01_A"].value = amplitude
        data = simulate_noisy(truth.model_active, noise_level=noise_level, seed=seed)
        file = File(
            parent_project=project,
            name=f"file_{i}",
            data=data,
            energy=truth.energy.copy(),
            time=truth.time.copy(),
        )
        file.load_model(
            model_yaml="models/project_energy.yaml", model_info="project_glp_base"
        )
        file.define_baseline(
            time_start=0, time_stop=3, time_type="ind", show_plot=False
        )
        file.fit_baseline(model_name="project_glp_base", stages=1, try_ci=0)
        file.load_model(
            model_yaml="models/project_energy.yaml", model_info="project_glp"
        )
        file.add_time_dependence(
            target_model="project_glp",
            target_parameter="GLP_01_x0",
            dynamics_yaml="models/project_time.yaml",
            dynamics_model=["MonoExpProject"],
        )
    return project


#
#
class TestVariantsTable:
    #
    def test_vary_flip_shows_only_differing_inputs(self):
        project, file, handle_a, handle_b = _two_variant_baseline()
        df = project.results.variants(
            file=file, model="single_glp", fit_type="baseline"
        )

        assert list(df["handle"]) == [handle_a[:8], handle_b[:8]]
        assert list(df.columns[:2]) == ["handle", "timestamp"]  # no labels yet
        # The flipped parameter surfaces; the shift surfaces as .init.
        assert list(df["GLP_01_x0.vary"]) == [True, False]
        assert "GLP_01_x0.init" in df.columns
        # Constant inputs are suppressed — the table IS the difference.
        assert "fit_alg_1" not in df.columns
        assert "stages" not in df.columns
        assert "GLP_01_A.vary" not in df.columns

    #
    def test_multiple_groups_raise(self):
        project, file, _, _ = _two_variant_baseline()
        file.fit_spectrum(model_name="single_glp", time_point=5, stages=1, try_ci=0)
        with pytest.raises(ValueError, match="one model"):
            project.results.variants(file=file, model="single_glp")

    #
    def test_no_match_raises(self):
        project, _, _, _ = _two_variant_baseline()
        with pytest.raises(LookupError, match="No slots match"):
            project.results.variants(file="nope")


#
#
class TestHandleAccessors:
    #
    def test_get_by_prefix_and_mutual_exclusion(self):
        project, _, handle_a, handle_b = _two_variant_baseline()
        results = project.results

        assert results.get(handle=handle_b[:10]).handle == handle_b
        with pytest.raises(TypeError, match="cannot be combined"):
            results.get(handle=handle_b[:10], model="single_glp")
        with pytest.raises(TypeError, match="requires either"):
            results.get(file="fit")
        # Multi-match filter errors point at variants() + handle=.
        with pytest.raises(LookupError, match="variants"):
            results.get(file="fit", model="single_glp", fit_type="baseline")

        params_a = results.get_parameters(handle=handle_a[:10])
        vary_x0 = params_a.loc[params_a["name"] == "GLP_01_x0", "vary"].iloc[0]
        assert bool(vary_x0) is True

    #
    def test_plot_fit_accepts_handle(self):
        project, _, handle_a, _ = _two_variant_baseline()
        results = project.results
        results.plot_fit(handle=handle_a[:10], show_plot=False)
        with pytest.raises(TypeError, match="cannot be combined"):
            results.plot_fit(handle=handle_a[:10], fit_type="baseline")

    #
    def test_handle_never_matches_a_label(self):
        project, _, handle_a, _ = _two_variant_baseline()
        results = project.results
        results.set_label(handle_a[:8], "keeper")
        with pytest.raises(LookupError, match="No fit matches"):
            results.get(handle="keeper")


#
#
class TestDiff:
    #
    def test_reports_input_and_result_rows(self):
        project, _, handle_a, handle_b = _two_variant_baseline()
        d = project.results.diff(handle_a[:8], handle_b[:8])

        assert list(d.columns) == ["section", "field", handle_a[:8], handle_b[:8]]
        by_field = {(row["section"], row["field"]): row for _, row in d.iterrows()}
        vary_row = by_field[("input", "GLP_01_x0.vary")]
        assert vary_row[handle_a[:8]] is True
        assert vary_row[handle_b[:8]] is False
        assert ("input", "GLP_01_x0.init") in by_field
        # Fitted values differ beyond tolerance → result rows exist.
        assert any(section == "result" for section, _ in by_field)
        # Identity fields agree within the group → no identity rows.
        assert not any(section == "identity" for section, _ in by_field)

    #
    def test_same_reference_raises(self):
        project, _, handle_a, _ = _two_variant_baseline()
        with pytest.raises(ValueError, match="same fit"):
            project.results.diff(handle_a[:8], handle_a)


#
#
class TestLabelFlow:
    #
    def test_label_selects_and_round_trips(self, tmp_path):
        project, file, handle_a, handle_b = _two_variant_baseline()
        results = project.results
        results.set_label(handle_b[:8], "keeper")

        df = results.variants(file=file, model="single_glp", fit_type="baseline")
        assert pd.isna(df["label"].iloc[0])  # unlabeled row
        assert df["label"].iloc[1] == "keeper"

        with pytest.raises(ValueError, match="reserved"):
            results.set_label(handle_b[:8], "latest")
        # Relabel by the current label — labels are ordinary references.
        results.set_label("keeper", "keeper-2")

        archive = tmp_path / "labeled.fit.h5"
        project.save_fits(archive, select="keeper-2", show_output=0)
        loaded = FitResults.load(archive)
        assert [s.handle for s in loaded] == [handle_b]
        assert next(iter(loaded)).label == "keeper-2"

    #
    def test_select_ref_excludes_filters(self, tmp_path):
        project, _, _, _ = _two_variant_baseline()
        with pytest.raises(ValueError, match="cannot be.*combined"):
            project.save_fits(tmp_path / "x.fit.h5", select="deadbeef", file="fit")


#
#
class TestSelectOnSaveAndExport:
    #
    def test_latest_and_best_pick_one_winner(self, tmp_path):
        project, _, handle_a, handle_b = _two_variant_baseline()

        latest = tmp_path / "latest.fit.h5"
        project.save_fits(latest, select="latest", show_output=0)
        assert [s.handle for s in FitResults.load(latest)] == [handle_b]

        best = tmp_path / "best.fit.h5"
        project.save_fits(best, select="best", by="chi2_red_raw", show_output=0)
        assert [s.handle for s in FitResults.load(best)] == [handle_a]

    #
    def test_by_validation(self, tmp_path):
        project, _, _, _ = _two_variant_baseline()
        with pytest.raises(ValueError, match="requires by="):
            project.save_fits(tmp_path / "a.fit.h5", select="best")
        with pytest.raises(ValueError, match="only meaningful"):
            project.save_fits(tmp_path / "b.fit.h5", select="latest", by="r2")

    #
    def test_export_defaults_to_latest(self, tmp_path):
        project, _, handle_a, handle_b = _two_variant_baseline()

        root_latest = tmp_path / "latest"
        project.export_fits(root_latest, show_output=0)
        dirs = sorted(p.name for p in (root_latest / "fit").iterdir())
        assert dirs == ["single_glp__baseline"]  # one winner, no hash suffix

        root_all = tmp_path / "all"
        project.export_fits(root_all, select="all", show_output=0)
        dirs = sorted(p.name for p in (root_all / "fit").iterdir())
        assert dirs == sorted(
            f"single_glp__baseline__{h[:8]}" for h in (handle_a, handle_b)
        )

    #
    def test_handle_prefix_exports_one_exact_run(self, tmp_path):
        project, _, handle_a, _ = _two_variant_baseline()
        root = tmp_path / "one"
        project.export_fits(root, select=handle_a[:8], show_output=0)
        dirs = sorted(p.name for p in (root / "fit").iterdir())
        assert dirs == ["single_glp__baseline"]


#
#
class TestDropFits:
    #
    def test_drop_by_prefix_prunes_the_session(self, tmp_path):
        project, _, handle_a, handle_b = _two_variant_baseline()
        project.drop_fits(handle_a[:8], show_output=0)

        remaining = project.results.find(
            file="fit", model="single_glp", fit_type="baseline"
        )
        assert [s.handle for s in remaining] == [handle_b]

        archive = tmp_path / "pruned.fit.h5"
        project.save_fits(archive, show_output=0)  # select="all"
        assert [s.handle for s in FitResults.load(archive)] == [handle_b]

    #
    def test_unknown_reference_raises(self):
        project, _, _, _ = _two_variant_baseline()
        with pytest.raises(LookupError, match="No fit matches"):
            project.drop_fits("no-such-fit")


#
#
class TestFileApiParity:
    """The File convenience wrappers expose the new selectors."""

    #
    def test_file_save_fit_forwards_select(self, tmp_path):
        _, file, _, handle_b = _two_variant_baseline()
        archive = tmp_path / "file_latest.fit.h5"
        file.save_fit(archive, select="latest", show_output=0)
        assert [s.handle for s in FitResults.load(archive)] == [handle_b]

    #
    def test_file_accessors_pin_owned_handles_only(self):
        project, file, handle_a, _ = _two_variant_baseline()
        # A second file with its own fit — its handle must be rejected by
        # the first file's accessors: handles are project-wide, and a
        # File accessor silently answering for another file's fit would
        # be worse than requiring project.results.
        assert file.data_raw is not None  # type guard
        assert file.energy is not None  # type guard
        assert file.time is not None  # type guard
        file_2 = File(
            parent_project=project,
            name="fit2",
            data=file.data_raw.copy(),
            energy=file.energy.copy(),
            time=file.time.copy(),
        )
        file_2.load_model(model_yaml="models/file_energy.yaml", model_info="single_glp")
        file_2.define_baseline(
            time_start=0, time_stop=3, time_type="ind", show_plot=False
        )
        file_2.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
        handle_2 = project.results.find(file="fit2")[-1].handle

        params_a = file.get_parameters(handle=handle_a[:10])
        vary_x0 = params_a.loc[params_a["name"] == "GLP_01_x0", "vary"].iloc[0]
        assert bool(vary_x0) is True
        file.plot_fit(handle=handle_a[:10], show_plot=False)

        with pytest.raises(ValueError, match="belongs to file"):
            file.get_parameters(handle=handle_2[:10])
        with pytest.raises(ValueError, match="belongs to file"):
            file.plot_fit(handle=handle_2[:10], show_plot=False)
        assert not file_2.get_parameters(handle=handle_2[:10]).empty


#
#
class TestCorrectionVariantVisibility:
    #
    def test_dark_value_change_shows_as_digest_column(self):
        """Three runs differing only in correction state — none, dark A,
        dark B — mint three slots, and the variant table shows the
        difference as distinct dark digests (booleans would render the
        last two identical)."""

        truth_project = make_project(name="truth")
        truth = _make_truth_file(truth_project)
        data = simulate_noisy(truth.model_active, noise_level=0.01)
        project = make_project(name="fit")
        file = File(
            parent_project=project,
            name="fit",
            data=data,
            energy=truth.energy.copy(),
            time=truth.time.copy(),
        )
        file.load_model(model_yaml="models/file_energy.yaml", model_info="single_glp")
        file.define_baseline(
            time_start=0, time_stop=3, time_type="ind", show_plot=False
        )
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
        assert file.energy is not None  # type guard
        file.subtract_dark(np.full(file.energy.size, 0.1))
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)
        file.reset_dark()
        file.subtract_dark(np.full(file.energy.size, 0.2))
        file.fit_baseline(model_name="single_glp", stages=1, try_ci=0)

        df = project.results.variants(
            file=file, model="single_glp", fit_type="baseline"
        )
        assert len(df) == 3
        darks = list(df["dark"])
        assert pd.isna(darks[0])  # uncorrected run
        assert isinstance(darks[1], str) and isinstance(darks[2], str)
        assert darks[1] != darks[2]


#
#
@pytest.mark.slow
class TestJointBundleQueries:
    #
    def test_bundle_label_select_diff_and_drop(self, tmp_path):
        """One fixture, every joint matrix row: projection labels escalate
        to the record; ``select=`` on a projection expands to the whole
        bundle; ``drop_fits`` on a projection raises and on the joint hash
        drops the bundle whole; two bundles differing in shared-parameter
        state show it in a bundle-level diff."""

        project = _build_joint_project()
        record_1 = project.fit_2d(model_name="project_glp", stages=2, try_ci=0)
        hash_1 = record_1.optimization_hash
        results = project.results
        projection = next(
            s for s in results.find(fit_type="2d") if s.joint_ref == hash_1
        )

        # Labels live on the joint record — a projection ref escalates.
        results.set_label(projection.handle[:8], "joint-run")
        assert record_1.label == "joint-run"
        assert projection.label is None

        # select= on a single projection expands to the whole bundle.
        archive = tmp_path / "bundle.fit.h5"
        project.save_fits(archive, select=projection.handle[:8], show_output=0)
        loaded = FitResults.load(archive)
        assert {(s.file_name, s.fit_type) for s in loaded} == {
            ("file_0", "2d"),
            ("file_1", "2d"),
        }
        assert len(loaded.find_joint()) == 1
        assert loaded.find_joint()[0].label == "joint-run"

        # drop() on one projection raises, naming the bundle.
        with pytest.raises(ValueError, match=hash_1[:8]):
            project.drop_fits(projection.handle[:8])

        # Second bundle: clamp the shared tau via its bound — a
        # shared-parameter state change and a distinct configuration.
        tau_combined = next(
            str(n)
            for n in record_1.params["name"]
            if "tau" in str(n) and not str(n).startswith("file0")
        )
        tau_fitted = float(
            record_1.params.loc[record_1.params["name"] == tau_combined, "value"].iloc[
                0
            ]
        )
        for file in project.files:
            model = next(m for m in file.models if m.name == "project_glp")
            key = next(k for k in model.lmfit_pars if "tau" in k)
            model.lmfit_pars[key].value = tau_fitted * 0.5
            model.lmfit_pars[key].max = tau_fitted * 0.6
        record_2 = project.fit_2d(model_name="project_glp", stages=2, try_ci=0)
        hash_2 = record_2.optimization_hash
        assert hash_2 != hash_1

        # Bundle-level diff: the bound change is an input row, the
        # clamped shared value a result row, and — comparability keys
        # matching — the whole-objective metrics compare as metric rows.
        d = project.results.diff(hash_1[:8], hash_2[:8])
        assert list(d.columns) == ["section", "field", hash_1[:8], hash_2[:8]]
        fields = {(row["section"], row["field"]) for _, row in d.iterrows()}
        assert ("input", f"{tau_combined}.max") in fields
        assert ("result", tau_combined) in fields
        assert any(section == "metric" for section, _ in fields)
        assert not any(section == "comparability" for section, _ in fields)

        # Dropping the joint hash removes the record and every projection.
        project.drop_fits(hash_2[:8], show_output=0)
        after = project.results
        assert [jr.optimization_hash for jr in after.find_joint()] == [hash_1]
        assert not [s for s in after.find(fit_type="2d") if s.joint_ref == hash_2]
