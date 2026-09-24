"""Declared noise in metrics, slots, identity and the archive.

Phase C of the noise-model work: the view-reduced model that weighted the
residual reaches the slot (``sigma_eff`` / ``noise_scale`` / ``sigma`` and
the noise-scaled metrics), the *declared* model reaches
``optimization_hash`` through the ``input_files`` entry, and both survive a
round trip through the HDF5 archive. Comparison keys on the noise model, so
a group mixing two of them refuses to rank.
"""

import json

import matplotlib
import numpy as np
import pytest
from _utils import make_project

matplotlib.use("Agg")

from trspecfit import File  # noqa: E402
from trspecfit.fit_results import FitResults  # noqa: E402
from trspecfit.utils.fit_io import (  # noqa: E402
    encode_input_files,
    select_snapshot_slots,
)
from trspecfit.utils.noise import NoiseModel  # noqa: E402

_ENERGY_YAML = "models/file_energy.yaml"
_MODEL = "single_glp"
_PROJECT_ENERGY_YAML = "models/project_energy.yaml"
_PROJECT_TIME_YAML = "models/project_time.yaml"


#
def _peak_data(*, n_time=6, n_energy=24, amplitude=20.0):
    """``(data, energy, time)`` of a strictly positive peak, rising in time."""

    energy = np.linspace(83.0, 87.0, n_energy)
    time = np.linspace(0.0, 5.0, n_time)
    peak = amplitude * np.exp(-0.5 * ((energy - 85.0) / 0.6) ** 2) + 0.5
    data = peak[None, :] * (1.0 + 0.05 * np.arange(n_time))[:, None]
    return data, energy, time


#
def _fit_ready_file(*, project=None, name="noise", n_time=6):
    """File with positive 2D data, a GLP model, a baseline and an e_lim window."""

    project = make_project(name="noise_archive") if project is None else project
    data, energy, time = _peak_data(n_time=n_time)
    file = File(parent_project=project, name=name, data=data, energy=energy, time=time)
    file.load_model(model_yaml=_ENERGY_YAML, model_info=_MODEL)
    file.define_baseline(0, 3, time_type="ind", show_plot=False)
    file.set_fit_limits([84.0, 86.0], show_plot=False)
    return file


#
def _baseline_slot(declare=None, *, project=None, name="noise"):
    """Fit the baseline of a fresh file under *declare* and return its slot."""

    file = _fit_ready_file(project=project, name=name)
    if declare is not None:
        declare(file)
    file.fit_baseline(_MODEL, stages=1, try_ci=0)
    return file.p._fit_history[-1]


#
def _sbs_slot(declare=None):
    """Fit slice-by-slice under *declare* and return the slot."""

    file = _fit_ready_file(name="sbs")
    if declare is not None:
        declare(file)
    file.fit_slice_by_slice(
        _MODEL, n_workers=1, seed_source="model", seed_adapt=None, try_ci=0
    )
    return file.p._fit_history[-1]


#
def _per_point_sigma(file, *, hot=0.7, base=0.5):
    """Per-point σ over the file's data, one pixel different from the rest."""

    assert file.data is not None  # type guard
    sigma = np.full(file.data.shape, base)
    sigma[0, 0] = hot
    return sigma


#
def _entries(input_files):
    """The decoded ``input_files`` entry list."""

    return json.loads(input_files)[1]


#
def _saved_slot(tmp_path, slot_file, *, name="archive"):
    """Save *slot_file*'s project and read back its single slot."""

    path = tmp_path / f"{name}.h5"
    slot_file.p.save_fits(path)
    return list(FitResults.load(path))[-1]


#
#
class TestDeclaredNoiseEntersTheHash:
    """The declared model is keyed; ``unknown`` hashes exactly as before."""

    #
    def test_declaring_sigma_changes_the_hash(self):
        plain = _baseline_slot()
        weighted = _baseline_slot(lambda f: f.set_sigma(0.5))
        assert plain.optimization_hash != weighted.optimization_hash

    #
    def test_two_constant_sigmas_differ(self):
        a = _baseline_slot(lambda f: f.set_sigma(0.5))
        b = _baseline_slot(lambda f: f.set_sigma(0.25))
        assert a.optimization_hash != b.optimization_hash

    #
    def test_two_poisson_scales_differ(self):
        a = _baseline_slot(lambda f: f.set_noise("poisson", scale=1.0))
        b = _baseline_slot(lambda f: f.set_noise("poisson", scale=2.0))
        assert a.optimization_hash != b.optimization_hash

    #
    def test_per_point_sigma_differing_in_one_element_differs(self):
        a = _baseline_slot(lambda f: f.set_noise("gaussian", sigma=_per_point_sigma(f)))
        b = _baseline_slot(
            lambda f: f.set_noise("gaussian", sigma=_per_point_sigma(f, hot=0.71))
        )
        assert a.optimization_hash != b.optimization_hash

    #
    def test_sigma_outside_the_fit_window_still_changes_the_hash(self):
        """The *declared* σ is keyed, not its view: the view is already
        implied by the selection, so keying it would double-count."""

        def hot_outside(file):
            assert file.data is not None  # type guard
            sigma = np.full(file.data.shape, 0.5)
            sigma[-1, -1] = 0.9  # outside base_t_ind and e_lim
            file.set_noise("gaussian", sigma=sigma)

        a = _baseline_slot(lambda f: f.set_noise("gaussian", sigma=_per_point_sigma(f)))
        b = _baseline_slot(hot_outside)
        assert a.optimization_hash != b.optimization_hash

    #
    def test_unknown_entry_keeps_three_elements(self):
        slot = _baseline_slot()
        entries = _entries(slot.input_files)
        assert [len(e) for e in entries] == [3]

    #
    def test_declared_entry_carries_the_noise_record(self):
        slot = _baseline_slot(lambda f: f.set_noise("poisson", scale=2.0))
        (entry,) = _entries(slot.input_files)
        assert len(entry) == 4
        assert entry[3] == {"kind": "poisson", "scale": "2.0"}

    #
    def test_unknown_encodes_exactly_like_an_omitted_model(self):
        """A hand-built three-element entry is the ``unknown`` encoding —
        archives written before the noise model keep their hashes."""

        three = encode_input_files(scope="file", entries=[("A", "0" * 64, "{}")])
        unknown = encode_input_files(
            scope="file",
            entries=[("A", "0" * 64, "{}", NoiseModel(kind="unknown"))],
        )
        assert three == unknown
        assert _entries(three) == [["A", "0" * 64, "{}"]]


#
#
class TestSigmaEffAndMetrics:
    """``sigma_eff`` is the constant-Gaussian view reduction, nothing else."""

    #
    def test_baseline_reduces_constant_sigma_by_sqrt_n(self):
        slot = _baseline_slot(lambda f: f.set_sigma(0.5))
        assert slot.sigma_data == pytest.approx(0.5)
        # define_baseline(0, 3) averages base_t_ind [0, 4) — four slices
        assert slot.sigma_eff == pytest.approx(0.5 / np.sqrt(4))
        assert slot.sigma_type == "constant"
        assert slot.sigma is None
        assert np.isnan(slot.noise_scale)

    #
    def test_chi2_is_chi2_raw_over_sigma_eff_squared(self):
        slot = _baseline_slot(lambda f: f.set_sigma(0.5))
        assert slot.metrics["chi2"] == pytest.approx(
            slot.metrics["chi2_raw"] / slot.sigma_eff**2
        )

    #
    def test_weighted_aic_and_bic_are_the_summed_objective(self):
        slot = _baseline_slot(lambda f: f.set_sigma(0.5))
        n_free = int((slot.params["vary"]).sum())
        chi2 = slot.metrics["chi2"]
        ndata = slot.observed.size
        assert slot.metrics["aic"] == pytest.approx(chi2 + 2 * n_free)
        assert slot.metrics["bic"] == pytest.approx(chi2 + np.log(ndata) * n_free)
        assert slot.metrics["chi2_red"] == pytest.approx(chi2 / (ndata - n_free))

    #
    def test_sigma_eff_nan_for_poisson_per_point_and_unknown(self):
        poisson = _baseline_slot(lambda f: f.set_noise("poisson", scale=2.0))
        per_point = _baseline_slot(
            lambda f: f.set_noise("gaussian", sigma=_per_point_sigma(f))
        )
        unknown = _baseline_slot()
        for slot in (poisson, per_point, unknown):
            assert np.isnan(slot.sigma_eff), slot.noise_type
            assert np.isnan(slot.sigma_data), slot.noise_type
        assert poisson.noise_scale == pytest.approx(2.0 * 4)  # four averaged slices
        assert per_point.sigma is not None
        assert np.isnan(unknown.metrics["chi2"])

    #
    def test_unknown_keeps_the_profiled_aic(self):
        slot = _baseline_slot()
        ndata = slot.observed.size
        n_free = int((slot.params["vary"]).sum())
        profiled = ndata * np.log(slot.metrics["chi2_raw"] / ndata) + 2 * n_free
        assert slot.metrics["aic"] == pytest.approx(profiled)

    #
    def test_sbs_stores_the_per_slice_view_sigma(self):
        slot = _sbs_slot(lambda f: f.set_noise("gaussian", sigma=_per_point_sigma(f)))
        assert slot.sigma is not None
        assert slot.sigma.shape == slot.observed.shape
        # slice 0 carries the one deviating pixel, inside the e_lim window
        assert slot.metrics["chi2"][0] != pytest.approx(slot.metrics["chi2"][1])


#
#
class TestArchiveRoundTrip:
    """``noise_scale`` and the ``sigma`` dataset survive save → load."""

    #
    def test_per_point_sigma_round_trips(self, tmp_path):
        file = _fit_ready_file()
        sigma = _per_point_sigma(file)
        file.set_noise("gaussian", sigma=sigma)
        file.fit_baseline(_MODEL, stages=1, try_ci=0)
        original = file.p._fit_history[-1]
        loaded = _saved_slot(tmp_path, file)
        assert loaded.sigma is not None
        assert loaded.sigma.shape == original.observed.shape
        np.testing.assert_allclose(loaded.sigma, original.sigma)
        assert np.isnan(loaded.sigma_data)
        assert loaded.sigma_type == "per_point"
        assert loaded.noise_type == "gaussian"

    #
    def test_poisson_scale_round_trips_and_chi2_is_the_deviance(self, tmp_path):
        file = _fit_ready_file()
        file.set_noise("poisson", scale=2.0)
        file.fit_baseline(_MODEL, stages=1, try_ci=0)
        loaded = _saved_slot(tmp_path, file)
        assert loaded.noise_scale == pytest.approx(2.0 * 4)
        assert np.isnan(loaded.sigma_eff)
        assert loaded.sigma is None
        view = NoiseModel(kind="poisson", scale=loaded.noise_scale)
        deviance = float(np.sum(view.apply(loaded.observed, loaded.fit) ** 2))
        assert loaded.metrics["chi2"] == pytest.approx(deviance)

    #
    def test_per_point_chi2_recomputes_from_the_stored_sigma(self, tmp_path):
        file = _fit_ready_file()
        file.set_noise("gaussian", sigma=_per_point_sigma(file))
        file.fit_baseline(_MODEL, stages=1, try_ci=0)
        loaded = _saved_slot(tmp_path, file)
        assert loaded.sigma is not None  # type guard
        view = NoiseModel(kind="gaussian", sigma=loaded.sigma)
        weighted = float(np.sum(view.apply(loaded.observed, loaded.fit) ** 2))
        assert loaded.metrics["chi2"] == pytest.approx(weighted)

    #
    def test_unknown_slot_reads_back_without_the_new_fields(self, tmp_path):
        file = _fit_ready_file()
        file.fit_baseline(_MODEL, stages=1, try_ci=0)
        loaded = _saved_slot(tmp_path, file)
        assert loaded.noise_type == "unknown"
        assert np.isnan(loaded.noise_scale)
        assert loaded.sigma is None


#
#
class TestNoiseKeyGatesComparison:
    """Metrics in different noise units are never ranked against each other."""

    #
    @staticmethod
    def _pair(first, second):
        """One file refit under two noise declarations — same view, two keys."""

        file = _fit_ready_file(name="f")
        slots = []
        for declare in (first, second):
            if declare is not None:
                declare(file)
            else:
                file.set_noise("unknown")
            file.fit_baseline(_MODEL, stages=1, try_ci=0)
            slots.append(file.p._fit_history[-1])
        assert slots[0].fit_view_sha256 == slots[1].fit_view_sha256
        return file.p, slots

    #
    @pytest.mark.parametrize("by", ["aic", "chi2_red"])
    def test_best_refuses_two_poisson_scales(self, by):
        _, slots = self._pair(
            lambda f: f.set_noise("poisson", scale=1.0),
            lambda f: f.set_noise("poisson", scale=2.0),
        )
        with pytest.raises(ValueError, match="mixes noise models"):
            select_snapshot_slots(slots, select="best", by=by)
        # the raw ranking stays available across the mix
        assert len(select_snapshot_slots(slots, select="best", by="chi2_red_raw")) == 1

    #
    @pytest.mark.parametrize("by", ["aic", "chi2_red"])
    def test_best_refuses_unknown_next_to_a_declared_model(self, by):
        _, slots = self._pair(None, lambda f: f.set_sigma(0.5))
        with pytest.raises(ValueError, match="mixes noise models"):
            select_snapshot_slots(slots, select="best", by=by)

    #
    def test_compare_models_drops_the_scaled_columns_for_mixed_scales(self):
        project, _ = self._pair(
            lambda f: f.set_noise("poisson", scale=1.0),
            lambda f: f.set_noise("poisson", scale=2.0),
        )
        df = project.results.compare_models()
        assert "chi2_red" not in df.columns
        # sigma_eff is NaN for both Poisson slots, so the all-NaN column
        # drop removes it too — chi2_red_raw is what stays comparable
        assert "chi2_red_raw" in df.columns
        assert "aic" not in df.columns and "bic" not in df.columns
        with pytest.raises(ValueError, match="mixes noise models"):
            project.results.compare_models(metrics=["chi2_red"])
        with pytest.raises(ValueError, match="mixes noise models"):
            project.results.compare_models(metrics=["aic"])

    #
    def test_compare_models_drops_the_scaled_columns_for_unknown_mix(self):
        project, _ = self._pair(None, lambda f: f.set_sigma(0.5))
        df = project.results.compare_models()
        assert "chi2_red" not in df.columns
        assert "aic" not in df.columns
        assert "sigma_eff" in df.columns  # the mix stays visible
        with pytest.raises(ValueError, match="unknown"):
            project.results.compare_models(metrics=["chi2"])


#
#
class TestJointRecordMetrics:
    """The joint record's metrics are the weighted objective, summed."""

    #
    @staticmethod
    def _joint_project():
        """Two-file project ready for a joint 2D fit."""

        project = make_project(name="noise_joint_archive")
        for name, t_window in (("b_file", [0.0, 5.0]), ("a_file", [1.0, 4.0])):
            data, energy, time = _peak_data(n_time=8, n_energy=16)
            file = File(
                parent_project=project,
                name=name,
                data=data,
                energy=energy,
                time=time,
            )
            file.load_model(
                model_yaml=_PROJECT_ENERGY_YAML, model_info="project_glp_base"
            )
            file.define_baseline(0, 1, time_type="ind", show_plot=False)
            file.set_fit_limits([84.0, 86.0], time_limits=t_window, show_plot=False)
            file.fit_baseline(model_name="project_glp_base", stages=1, try_ci=0)
            file.load_model(model_yaml=_PROJECT_ENERGY_YAML, model_info="project_glp")
            file.add_time_dependence(
                target_model="project_glp",
                target_parameter="GLP_01_x0",
                dynamics_yaml=_PROJECT_TIME_YAML,
                dynamics_model=["MonoExpProject"],
            )
        return project

    #
    def test_joint_chi2_is_the_sum_over_files(self):
        project = self._joint_project()
        project.files[0].set_sigma(0.3)
        project.files[1].set_noise("poisson", scale=2.0)
        record = project.fit_2d(model_name="project_glp", stages=1, try_ci=0)
        per_file = sum(float(p.slot.metrics["chi2"]) for p in record.projections)
        chi2 = float(record.metrics["chi2"])
        assert chi2 == pytest.approx(per_file)
        nvarys = int(record.params["vary"].sum())
        ndata = sum(p.slot.observed.size for p in record.projections)
        assert record.metrics["aic"] == pytest.approx(chi2 + 2 * nvarys)
        assert record.metrics["bic"] == pytest.approx(chi2 + np.log(ndata) * nvarys)
        assert record.metrics["chi2_red"] == pytest.approx(chi2 / (ndata - nvarys))

    #
    def test_joint_entries_carry_each_file_noise(self):
        project = self._joint_project()
        project.files[0].set_sigma(0.3)
        project.files[1].set_noise("poisson", scale=2.0)
        record = project.fit_2d(model_name="project_glp", stages=1, try_ci=0)
        entries = {e[0]: e for e in _entries(record.input_files)}
        assert entries["b_file"][3] == {"kind": "gaussian", "sigma": "0.3"}
        assert entries["a_file"][3] == {"kind": "poisson", "scale": "2.0"}

    #
    def test_unknown_joint_keeps_the_profiled_form(self):
        project = self._joint_project()
        record = project.fit_2d(model_name="project_glp", stages=1, try_ci=0)
        assert np.isnan(record.metrics["chi2"])
        assert np.isfinite(record.metrics["aic"])
        assert [len(e) for e in _entries(record.input_files)] == [3, 3]
