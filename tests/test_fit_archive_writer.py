"""
Writer/reader contract on hand-built schema-7 records (no fits).

Covers the parts of the archive contract that are about the *records*,
not about capture: on-disk layout and compression, the four-case
collision table on ``handle`` / ``optimization_hash`` (one test per row —
the distinctions are exactly what a two-case implementation would lose),
append-integrity raises (project name, file content), the joint sidecar,
bundle-integrity prechecks, and the reader's mirror validation. Real-fit
round-trips live in ``test_fit_archive_roundtrip.py``; in-session capture
and history semantics live in ``test_fit_history.py``.
"""

from __future__ import annotations

import dataclasses
import json
import shutil

import h5py
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from trspecfit.config.plot import PlotConfig
from trspecfit.utils import fit_io
from trspecfit.utils.hdf5 import require_dataset, require_group
from trspecfit.utils.lmfit import MCMCResult

_METRIC_KEYS = ("chi2_raw", "chi2_red_raw", "chi2", "chi2_red", "r2", "aic", "bic")
_DOF_KEYS = ("chi2_red_raw", "chi2_red", "aic", "bic")
_TS = "2026-08-14T00:00:00+00:00"


#
def _make_params(names, values):
    return pd.DataFrame(
        {
            "name": names,
            "value": values,
            "stderr": [0.1] * len(names),
            "init_value": [v * 0.9 for v in values],
            "min": [-10.0] * len(names),
            "max": [10.0] * len(names),
            "vary": [True] * len(names),
            "expr": [None] * len(names),
        }
    )


#
def _make_file_and_slot(
    file_name,
    *,
    scope="file",
    joint_hash=None,
    joint_entries=None,
    seed=0,
    fit_type="spectrum",
    **slot_overrides,
):
    """(SavedFile, SavedFitSlot, version_stamp) with a real identity chain."""

    rng = np.random.default_rng(seed)
    energy = np.linspace(0.0, 10.0, 50)
    data = rng.normal(size=50)
    fch = fit_io.compute_file_content_hash(
        data_raw=data, energy=energy, time=np.array([]), aux_axis=None
    )
    stamp = fit_io.compute_file_version_stamp(
        file_content_hash=fch, dark=None, calibration=None
    )
    sel = json.dumps({"e_lim": None}, sort_keys=True)
    if scope == "file":
        in_files = fit_io.encode_input_files(
            scope="file", entries=[(file_name, stamp, sel)]
        )
    else:
        in_files = fit_io.encode_input_files(scope="project", entries=joint_entries)
    ms = fit_io.encode_model_structure([(file_name, ["peaks"], [])])
    opt = fit_io.encode_optimizer_settings(
        stages=1, fit_alg_1="Nelder", fit_alg_2="leastsq", backend="fit_model_mcp"
    )
    names = ["GLP_01_A", "GLP_01_x0"]
    if joint_hash is None:
        oh = fit_io.compute_optimization_hash(
            input_files_json=in_files,
            fit_type=fit_type,
            model_structure_json=ms,
            parameter_metadata=[(n, -10.0, 10.0, True, None) for n in names],
            initial_state=np.array([[0.9, 4.5]]),
            optimizer_settings_json=opt,
        )
    else:
        oh = joint_hash
    handle = fit_io.compute_slot_handle(optimization_hash=oh, file_name=file_name)
    fv = fit_io.compute_fit_view_sha256(
        observed=data, energy=energy, time=None, aux_axis=None
    )
    fields = dict(
        handle=handle,
        optimization_hash=oh,
        input_files=in_files,
        model_structure=ms,
        fit_view_sha256=fv,
        file_name=file_name,
        model_name="peaks",
        fit_type=fit_type,
        selection={"e_lim": None},
        selection_json=sel,
        params=_make_params(names, [1.0, 5.0]),
        metrics={k: 1.0 for k in _METRIC_KEYS},
        observed=data,
        fit=data * 0.99,
        fit_alg="Nelder",
        timestamp=_TS,
        noise_type="unknown",
        sigma_source="user_supplied",
        sigma_type="constant",
        sigma_data=float("nan"),
        sigma_eff=float("nan"),
        fit_settings={"stages": 1, "fit_alg_1": "Nelder", "try_ci": 0},
        joint_ref=joint_hash,
        model_yaml=(
            fit_io.ModelYamlRecord(
                "energy", "peaks", "models.yaml", None, None, "peaks:\n  GLP: {}\n"
            ),
        ),
    )
    fields.update(slot_overrides)
    slot = fit_io.SavedFitSlot(**fields)
    sf = fit_io.SavedFile(
        name=file_name,
        original_path=f"/data/{file_name}.h5",
        dim=1,
        shape=data.shape,
        file_content_hash=fch,
        data_raw=data,
        energy=energy,
        time=np.array([]),
        slots=(slot,),
    )
    return sf, slot, stamp


#
def _make_project(files, joint=()):
    return fit_io.SavedProject(
        name="proj",
        trspecfit_version="0.14.0",
        schema_version="7",
        timestamp_created=_TS,
        timestamp_updated=_TS,
        plot_config=PlotConfig(),
        files=tuple(files),
        joint=tuple(joint),
    )


#
def _mcmc_payload(seed=7):
    rng = np.random.default_rng(seed)
    return {
        "flatchain": pd.DataFrame(
            rng.normal(size=(20, 2)), columns=["GLP_01_A", "GLP_01_x0"]
        ),
        "ci": pd.DataFrame({"name": ["GLP_01_A"], "p50": [1.0]}),
        "lnsigma": 0.5,
        "acceptance_fraction": rng.uniform(0.2, 0.6, size=8),
    }


#
def _make_joint_bundle():
    """Two projection files + one JointFitResult with real joint identity."""

    sel = json.dumps({"e_lim": None}, sort_keys=True)
    _, _, stamp_a = _make_file_and_slot("A", seed=1)
    _, _, stamp_b = _make_file_and_slot("B", seed=2)
    entries = [("A", stamp_a, sel), ("B", stamp_b, sel)]
    in_files = fit_io.encode_input_files(scope="project", entries=entries)
    ms = fit_io.encode_model_structure([("A", ["peaks"], []), ("B", ["peaks"], [])])
    opt = fit_io.encode_optimizer_settings(
        stages=1, fit_alg_1="Nelder", fit_alg_2="leastsq", backend="fit_model_mcp"
    )
    combined = ["file00_GLP_01_A", "file01_GLP_01_A", "GLP_01_x0"]
    jrh = fit_io.compute_optimization_hash(
        input_files_json=in_files,
        fit_type="2d",
        model_structure_json=ms,
        parameter_metadata=[(n, -10.0, 10.0, True, None) for n in combined],
        initial_state=np.array([[0.9, 0.9, 4.5]]),
        optimizer_settings_json=opt,
    )
    # DoF metrics do not decompose per file on a joint projection.
    nan_dof = {k: (float("nan") if k in _DOF_KEYS else 1.0) for k in _METRIC_KEYS}
    sf_a, slot_a, _ = _make_file_and_slot(
        "A",
        seed=1,
        scope="project",
        joint_hash=jrh,
        joint_entries=entries,
        fit_type="2d",
        metrics=nan_dof,
    )
    sf_b, slot_b, _ = _make_file_and_slot(
        "B",
        seed=2,
        scope="project",
        joint_hash=jrh,
        joint_entries=entries,
        fit_type="2d",
        metrics=nan_dof,
        # B's projected amplitude must agree with the combined table's
        # file01_GLP_01_A (1.1) — the bundle validator checks values now.
        params=_make_params(["GLP_01_A", "GLP_01_x0"], [1.1, 5.0]),
    )
    map_a = {"file00_GLP_01_A": "GLP_01_A", "GLP_01_x0": "GLP_01_x0"}
    map_b = {"file01_GLP_01_A": "GLP_01_A", "GLP_01_x0": "GLP_01_x0"}
    jr = fit_io.JointFitResult(
        model_name="peaks",
        optimization_hash=jrh,
        input_files=in_files,
        model_structure=ms,
        projections=(
            fit_io.JointFitProjection(parameter_map=map_a, slot=slot_a),
            fit_io.JointFitProjection(parameter_map=map_b, slot=slot_b),
        ),
        params=_make_params(combined, [1.0, 1.1, 5.0]),
        metrics={**{k: 2.0 for k in _METRIC_KEYS}, "r2": float("nan")},
        fit_alg="Nelder",
        fit_settings={"stages": 1, "fit_alg_1": "Nelder", "try_ci": 0},
        timestamp=_TS,
    )
    return sf_a, sf_b, jr, (map_a, map_b), combined


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------


#
def test_fresh_write_layout_retired_attrs_and_compression(tmp_path):
    """project/ hierarchy, identity attrs, no retired schema-6 fields,
    gzip+shuffle on non-empty arrays, uncompressed model_yaml snippets."""

    path = tmp_path / "one.fit.h5"
    sf, slot, _ = _make_file_and_slot("A")
    fit_io.write_archive(path, project=_make_project([sf]))
    with h5py.File(path) as archive:
        assert set(archive.keys()) == {"metadata", "project"}
        meta = dict(require_group(archive["metadata"], "metadata").attrs)
        assert meta["format"] == "trspecfit-fit-archive"
        assert meta["schema_version"] == "7"
        project_group = require_group(archive["project"], "project")
        assert project_group.attrs["name"] == "proj"
        PlotConfig.from_json(project_group.attrs["plot_config"])  # decodable
        fg = require_group(project_group["files/000000"], "files/000000")
        file_meta = dict(require_group(fg["metadata"], "file metadata").attrs)
        assert file_meta["file_content_hash"] == sf.file_content_hash
        assert "data_sha256" not in file_meta and "e_lim" not in file_meta
        data_raw = require_dataset(fg["data_raw"], "data_raw")
        assert data_raw.compression == "gzip" and data_raw.shuffle
        time_ds = require_dataset(fg["time"], "time")
        assert time_ds.shape == (0,) and time_ds.compression is None
        assert "data" not in fg
        sg = require_group(fg["slots/000000"], "slots/000000")
        slot_meta = dict(require_group(sg["metadata"], "slot metadata").attrs)
        assert slot_meta["handle"] == slot.handle
        assert slot_meta["optimization_hash"] == slot.optimization_hash
        assert slot_meta["fit_view_sha256"] == slot.fit_view_sha256
        for gone in (
            "selection_json",
            "history_key",
            "observed_sha256",
            "archive_slot_key",
            "file_ref",
            "yaml_filename",
            "label",
            "joint_ref",
        ):
            assert gone not in slot_meta, gone
        # File scope: all seven metric attrs present.
        for k in _METRIC_KEYS:
            assert slot_meta[k] == 1.0
        assert require_dataset(sg["observed"], "observed").compression == "gzip"
        assert require_dataset(sg["params"], "params").compression == "gzip"
        snippet = require_dataset(sg["model_yaml/000000"], "model_yaml/000000")
        assert bytes(snippet[()]).decode() == "peaks:\n  GLP: {}\n"
        assert snippet.attrs["role"] == "energy"
        assert snippet.attrs["source_file"] == "models.yaml"
        assert "target_par" not in snippet.attrs
        assert "sequence_index" not in snippet.attrs
        assert snippet.compression is None


#
def test_read_arrays_come_back_frozen(tmp_path):
    """Every array the reader hands out is a read-only copy (Principle 4)."""

    path = tmp_path / "frozen.fit.h5"
    sf, _, _ = _make_file_and_slot(
        "A",
        dark=np.full(50, 0.1),
        calibration=np.full(50, 1.5),
        mcmc=_mcmc_payload(),
    )
    fit_io.write_archive(path, project=_make_project([sf]))
    loaded = fit_io.read_archive(path)
    (lf,) = loaded.files
    (ls,) = lf.slots
    assert ls.mcmc is not None  # type guard
    for arr in (
        lf.data_raw,
        lf.energy,
        lf.time,
        ls.observed,
        ls.fit,
        ls.dark,
        ls.calibration,
        ls.mcmc["acceptance_fraction"],
    ):
        assert arr is not None  # type guard
        assert not arr.flags.writeable


# ---------------------------------------------------------------------------
# collision table (one test per row)
# ---------------------------------------------------------------------------


#
def test_identical_resave_is_a_noop(tmp_path):
    path = tmp_path / "one.fit.h5"
    sf, _, _ = _make_file_and_slot("A")
    project = _make_project([sf])
    fit_io.write_archive(path, project=project)
    fit_io.write_archive(path, project=project)
    with h5py.File(path) as archive:
        assert len(require_group(archive["project/files/000000/slots"], "slots")) == 1


#
def test_absent_to_present_enriches_without_overwrite(tmp_path):
    """Row 2: running MCMC/CI after a save attaches with no overwrite=;
    label is mutable alongside."""

    path = tmp_path / "one.fit.h5"
    sf, slot, _ = _make_file_and_slot("A")
    fit_io.write_archive(path, project=_make_project([sf]))
    enriched = dataclasses.replace(
        slot,
        conf_ci=pd.DataFrame({"name": ["GLP_01_A"], "low": [0.5], "high": [1.5]}),
        mcmc=_mcmc_payload(),
        label="with-ci",
    )
    fit_io.write_archive(
        path, project=_make_project([dataclasses.replace(sf, slots=(enriched,))])
    )
    with h5py.File(path) as archive:
        sg = require_group(archive["project/files/000000/slots/000000"], "slot")
        assert "conf_ci" in sg and "mcmc" in sg
        assert require_group(sg["metadata"], "slot metadata").attrs["label"] == (
            "with-ci"
        )


#
def test_present_on_both_and_differing_requires_overwrite(tmp_path):
    """Row 4 (attachment level): a shorter re-run over a longer archived
    chain raises without overwrite=True and leaves the archived chain
    intact; overwrite=True replaces it."""

    path = tmp_path / "one.fit.h5"
    sf, slot, _ = _make_file_and_slot("A")
    long_chain = _mcmc_payload(seed=7)
    with_long = dataclasses.replace(slot, mcmc=long_chain)
    fit_io.write_archive(
        path, project=_make_project([dataclasses.replace(sf, slots=(with_long,))])
    )
    short_chain = {**_mcmc_payload(seed=8)}
    short_chain["flatchain"] = short_chain["flatchain"].iloc[:5]
    with_short = dataclasses.replace(slot, mcmc=short_chain)
    project_short = _make_project([dataclasses.replace(sf, slots=(with_short,))])
    with pytest.raises(FileExistsError, match="pass\\s+overwrite=True"):
        fit_io.write_archive(path, project=project_short)
    # The archived (long) chain is intact after the failed save.
    loaded = fit_io.read_archive(path)
    stored = loaded.files[0].slots[0].mcmc
    assert stored is not None  # type guard
    assert_frame_equal(stored["flatchain"], long_chain["flatchain"])
    fit_io.write_archive(path, project=project_short, overwrite=True)
    loaded = fit_io.read_archive(path)
    stored = loaded.files[0].slots[0].mcmc
    assert stored is not None  # type guard
    assert len(stored["flatchain"]) == 5


#
def test_stderr_drift_and_tiny_value_drift_are_not_conflicts(tmp_path):
    """Fitted-value equivalence is per parameter within named tolerances.

    ``stderr`` is not hashed and legitimately drifts across re-runs of
    the same optimum (numdifftools present vs absent, a CI re-run); a
    value nudge within rtol=1e-6/atol=1e-12 is numerical noise. Neither
    may classify as a hard conflict — that would demand overwrite=True,
    which replaces the whole slot including stored attachments."""

    path = tmp_path / "one.fit.h5"
    sf, slot, _ = _make_file_and_slot("A")
    fit_io.write_archive(path, project=_make_project([sf]))
    drifted_params = _make_params(["GLP_01_A", "GLP_01_x0"], [1.0 + 1e-9, 5.0])
    drifted_params["stderr"] = [0.5, 0.5]
    drifted = dataclasses.replace(slot, params=drifted_params)
    # No raise, no overwrite needed: this classifies as agreement.
    fit_io.write_archive(
        path, project=_make_project([dataclasses.replace(sf, slots=(drifted,))])
    )
    loaded = fit_io.read_archive(path)
    (stored,) = loaded.files[0].slots
    # Agreement keeps the stored record (no partial update).
    assert stored.params.loc[0, "value"] == 1.0
    assert stored.params.loc[0, "stderr"] == 0.1


#
def test_value_drift_beyond_tolerance_is_a_conflict(tmp_path):
    """A relative value change of 1e-3 is a divergent optimum, not noise."""

    path = tmp_path / "one.fit.h5"
    sf, slot, _ = _make_file_and_slot("A")
    fit_io.write_archive(path, project=_make_project([sf]))
    diverged = dataclasses.replace(
        slot, params=_make_params(["GLP_01_A", "GLP_01_x0"], [1.0 + 1e-3, 5.0])
    )
    with pytest.raises(FileExistsError, match="differing fitted parameters"):
        fit_io.write_archive(
            path, project=_make_project([dataclasses.replace(sf, slots=(diverged,))])
        )


#
def test_present_to_absent_keeps_stored_attachment(tmp_path):
    """Row 3: re-saving a fit that has no MCMC does not delete an
    archived chain."""

    path = tmp_path / "one.fit.h5"
    sf, slot, _ = _make_file_and_slot("A")
    with_mcmc = dataclasses.replace(slot, mcmc=_mcmc_payload())
    fit_io.write_archive(
        path, project=_make_project([dataclasses.replace(sf, slots=(with_mcmc,))])
    )
    fit_io.write_archive(path, project=_make_project([sf]))  # slot without mcmc
    with h5py.File(path) as archive:
        sg = require_group(archive["project/files/000000/slots/000000"], "slot")
        assert "mcmc" in sg


#
def test_diverging_params_hard_conflict_and_full_replace(tmp_path):
    """Row 1: same handle + differing fitted parameters raises;
    overwrite=True resolves it without a refit — the incoming record is
    written whole (its own MCMC attachment survives the resolution,
    proving nothing was recomputed), and the old attachments are gone."""

    path = tmp_path / "one.fit.h5"
    sf, slot, _ = _make_file_and_slot("A")
    with_ci = dataclasses.replace(
        slot, conf_ci=pd.DataFrame({"name": ["GLP_01_A"], "low": [0.5], "high": [1.5]})
    )
    fit_io.write_archive(
        path, project=_make_project([dataclasses.replace(sf, slots=(with_ci,))])
    )
    incoming_chain = _mcmc_payload(seed=9)
    diverged = dataclasses.replace(
        slot,
        params=_make_params(["GLP_01_A", "GLP_01_x0"], [2.0, 5.0]),
        mcmc=incoming_chain,
    )
    project_new = _make_project([dataclasses.replace(sf, slots=(diverged,))])
    with pytest.raises(FileExistsError, match="differing fitted parameters"):
        fit_io.write_archive(path, project=project_new)
    fit_io.write_archive(path, project=project_new, overwrite=True)
    loaded = fit_io.read_archive(path)
    (stored,) = loaded.files[0].slots
    assert stored.params.loc[0, "value"] == 2.0
    assert stored.conf_ci is None  # full replacement: old attachment dropped
    assert stored.mcmc is not None  # type guard
    assert_frame_equal(stored.mcmc["flatchain"], incoming_chain["flatchain"])


# ---------------------------------------------------------------------------
# append integrity
# ---------------------------------------------------------------------------


#
def test_project_name_mismatch_raises(tmp_path):
    path = tmp_path / "one.fit.h5"
    sf, _, _ = _make_file_and_slot("A")
    project = _make_project([sf])
    fit_io.write_archive(path, project=project)
    with pytest.raises(ValueError, match="one project per archive"):
        fit_io.write_archive(path, project=dataclasses.replace(project, name="other"))


#
def test_same_name_different_content_raises_pre_mutation(tmp_path):
    path = tmp_path / "one.fit.h5"
    sf, _, _ = _make_file_and_slot("A")
    fit_io.write_archive(path, project=_make_project([sf]))
    before_bytes = path.read_bytes()
    sf_mut, _, _ = _make_file_and_slot("A", seed=99)  # same name, other data
    for overwrite in (False, True):
        with pytest.raises(ValueError, match="file_content_hash mismatch"):
            fit_io.write_archive(
                path, project=_make_project([sf_mut]), overwrite=overwrite
            )
    assert path.read_bytes() == before_bytes


# ---------------------------------------------------------------------------
# joint sidecar
# ---------------------------------------------------------------------------


#
def test_joint_sidecar_layout_and_dof_omission(tmp_path):
    """One joint/ record: projections JSON with parameter maps, six
    whole-objective metrics (r2 omitted), joint_ref + DoF omission on
    every projection slot."""

    path = tmp_path / "joint.fit.h5"
    sf_a, sf_b, jr, (map_a, _), _ = _make_joint_bundle()
    fit_io.write_archive(path, project=_make_project([sf_a, sf_b], joint=[jr]))
    with h5py.File(path) as archive:
        jg = require_group(archive["project/joint/000000"], "joint/000000")
        joint_meta = dict(require_group(jg["metadata"], "joint metadata").attrs)
        assert joint_meta["optimization_hash"] == jr.optimization_hash
        assert joint_meta["model_name"] == "peaks"
        assert "r2" not in joint_meta
        for k in ("chi2_raw", "chi2_red_raw", "chi2", "chi2_red", "aic", "bic"):
            assert joint_meta[k] == 2.0
        records = json.loads(joint_meta["projections"])
        assert [r["file_name"] for r in records] == ["A", "B"]
        assert records[0]["handle"] == sf_a.slots[0].handle
        assert records[0]["parameter_map"] == map_a
        assert "params" in jg
        for file_key in ("000000", "000001"):
            slot_meta = dict(
                require_group(
                    archive[f"project/files/{file_key}/slots/000000/metadata"],
                    "slot metadata",
                ).attrs
            )
            assert slot_meta["joint_ref"] == jr.optimization_hash
            assert slot_meta["chi2_raw"] == 1.0 and slot_meta["r2"] == 1.0
            for gone in _DOF_KEYS:
                assert gone not in slot_meta, gone


#
def test_joint_attachments_enrich_and_collide_and_roundtrip(tmp_path):
    """Joint correl/mcmc follow the same per-attachment rules and
    round-trip through the reader."""

    path = tmp_path / "joint.fit.h5"
    sf_a, sf_b, jr, _, combined = _make_joint_bundle()
    project = _make_project([sf_a, sf_b], joint=[jr])
    fit_io.write_archive(path, project=project)
    fit_io.write_archive(path, project=project)  # idempotent re-save
    chain = _mcmc_payload(seed=11)
    chain["flatchain"].columns = combined[:2]
    jr_rich = dataclasses.replace(
        jr,
        correl=pd.DataFrame(np.eye(3), index=combined, columns=combined),
        mcmc=fit_io.mcmc_result_from_payload(chain),
    )
    project_rich = dataclasses.replace(project, joint=(jr_rich,))
    fit_io.write_archive(path, project=project_rich)  # absent -> present
    with pytest.raises(FileExistsError, match="pass\\s+overwrite=True"):
        fit_io.write_archive(path, project=project_rich)  # both present
    loaded = fit_io.read_archive(path)
    (jrec,) = loaded.joint
    assert jrec.optimization_hash == jr.optimization_hash
    assert jrec.correl is not None  # type guard
    assert list(jrec.correl.index) == combined
    assert jrec.mcmc is not None  # type guard
    assert isinstance(jrec.mcmc, MCMCResult)
    assert_frame_equal(jrec.mcmc.flatchain, chain["flatchain"])
    assert jrec.mcmc.lnsigma == 0.5
    assert np.isnan(jrec.metrics["r2"])  # r2 omitted on disk -> NaN
    # Projection slots resolve by handle to the same objects under files.
    assert jrec.projections[0].slot is loaded.files[0].slots[0]
    assert jrec.projections[1].slot is loaded.files[1].slots[0]
    for k in _DOF_KEYS:
        assert np.isnan(jrec.projections[0].slot.metrics[k])


# ---------------------------------------------------------------------------
# bundle integrity (writer precheck + reader mirror)
# ---------------------------------------------------------------------------


#
def test_bundle_integrity_raises_and_archive_untouched(tmp_path):
    path = tmp_path / "broken.fit.h5"
    sf_a, sf_b, jr, (_, map_b), _ = _make_joint_bundle()

    # Projection slots without their joint record.
    with pytest.raises(ValueError, match="saved whole"):
        fit_io.write_archive(path, project=_make_project([sf_a, sf_b], joint=[]))

    # Non-total parameter map (a local param is unmapped).
    bad_map = {"file00_GLP_01_A": "GLP_01_A"}  # missing GLP_01_x0
    slot_a, slot_b = (proj.slot for proj in jr.projections)
    jr_bad = dataclasses.replace(
        jr,
        projections=(
            fit_io.JointFitProjection(parameter_map=bad_map, slot=slot_a),
            fit_io.JointFitProjection(parameter_map=map_b, slot=slot_b),
        ),
    )
    with pytest.raises(ValueError, match="not total against its slot"):
        fit_io.write_archive(path, project=_make_project([sf_a, sf_b], joint=[jr_bad]))

    # Missing projection: combined params no longer covered.
    jr_missing = dataclasses.replace(jr, projections=jr.projections[:1])
    with pytest.raises(ValueError, match="do not cover the combined parameter"):
        fit_io.write_archive(
            path, project=_make_project([sf_a, sf_b], joint=[jr_missing])
        )

    # A projection whose parameter table disagrees with the combined result.
    slot_a_bad = dataclasses.replace(
        jr.projections[0].slot,
        params=_make_params(["GLP_01_A", "GLP_01_x0"], [2.0, 5.0]),
    )
    jr_disagree = dataclasses.replace(
        jr,
        projections=(
            fit_io.JointFitProjection(
                parameter_map=jr.projections[0].parameter_map, slot=slot_a_bad
            ),
            jr.projections[1],
        ),
    )
    sf_a_bad = dataclasses.replace(sf_a, slots=(slot_a_bad,))
    with pytest.raises(ValueError, match="disagree with the combined table"):
        fit_io.write_archive(
            path, project=_make_project([sf_a_bad, sf_b], joint=[jr_disagree])
        )

    # scope <-> joint_ref invariant: file-scoped slot with a joint_ref.
    sf, slot, _ = _make_file_and_slot("A")
    slot_bad = dataclasses.replace(slot, joint_ref="deadbeef" * 8)
    with pytest.raises(ValueError, match="joint-reference invariant"):
        fit_io.write_archive(
            path,
            project=_make_project([dataclasses.replace(sf, slots=(slot_bad,))]),
        )

    # All prechecks fired before any payload mutation: the h5py open may
    # have created the container file, but nothing was written into it.
    if path.exists():
        with h5py.File(path) as archive:
            assert len(archive.keys()) == 0


#
def test_reader_raises_on_dangling_projection_handle(tmp_path):
    path = tmp_path / "joint.fit.h5"
    sf_a, sf_b, jr, _, _ = _make_joint_bundle()
    fit_io.write_archive(path, project=_make_project([sf_a, sf_b], joint=[jr]))
    broken = tmp_path / "dangling.fit.h5"
    shutil.copy(path, broken)
    with h5py.File(broken, "a") as archive:
        del archive["project/files/000000/slots/000000"]
    with pytest.raises(ValueError, match="not stored in the archive"):
        fit_io.read_archive(broken)


#
def test_reader_raises_on_missing_joint_group(tmp_path):
    """The slot→joint half of the mirror: project-scoped slots whose
    joint records were deleted must not load silently."""

    path = tmp_path / "joint.fit.h5"
    sf_a, sf_b, jr, _, _ = _make_joint_bundle()
    fit_io.write_archive(path, project=_make_project([sf_a, sf_b], joint=[jr]))
    broken = tmp_path / "no_joint.fit.h5"
    shutil.copy(path, broken)
    with h5py.File(broken, "a") as archive:
        del archive["project/joint"]
    with pytest.raises(ValueError, match="not stored in the archive"):
        fit_io.read_archive(broken)
