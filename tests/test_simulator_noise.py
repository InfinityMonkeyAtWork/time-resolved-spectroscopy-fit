"""The noise declaration the simulator hands to a fit.

``Simulator.noise_model`` reports the ``File.set_noise`` arguments for the
noise the simulator actually drew, so a fit of simulated data can be weighted
by the truth instead of an estimate. Two properties of that report are worth
testing: it has to describe the *statistics of the simulated array* — the
counting variance really is ``signal / scale`` — and it has to stay attached
to the data it was drawn for, unaffected by noise settings changed afterwards.
"""

import h5py
import matplotlib
import numpy as np
import pytest
from _utils import make_project

matplotlib.use("Agg")

from trspecfit import File, Simulator  # noqa: E402
from trspecfit.utils.hdf5 import require_dataset  # noqa: E402

_ENERGY = np.linspace(82.0, 88.0, 41)
_TIME = np.linspace(-2.0, 10.0, 17)
_ENERGY_YAML = "models/eval_2d_energy.yaml"
_TIME_YAML = "models/file_time.yaml"
_MODEL = "offset_only"
_DYNAMICS = "MonoExpPos"
_SEED = 11

# Peak decaying over the time axis on a flat background of 1.0. The background
# keeps every pixel at ~90 expected counts or more under _COUNTS, so no bin is
# in the small-count regime where the variance estimator itself gets noisy.
_TRUTH = {"Offset_y0": 1.0, "GLP_01_A": 16.0, "GLP_01_A_expFun_01_A": 4.0}
_COUNTS = 20000
# smaller budget for the bleach model, whose signal is a few units per pixel
_BLEACH_COUNTS = 5000

# Realizations behind the variance checks. The per-pixel ratio var/mean is a
# sample variance of ~400 draws, so its relative scatter is sqrt(2 / (n - 1)).
_N_REALIZATIONS = 400
_RATIO_SD = np.sqrt(2.0 / (_N_REALIZATIONS - 1))


#
def _peak_file(*, name, data=None, time=True):
    """File carrying the peak-on-background model on the shared axes."""

    project = make_project(name=name)
    file = File(
        parent_project=project,
        name=name,
        data=data,
        energy=_ENERGY.copy(),
        time=_TIME.copy() if time else None,
    )
    file.load_model(model_yaml=_ENERGY_YAML, model_info=_MODEL)
    return file


#
def _truth_model(*, name, dynamics=True):
    """Truth model at the operating point, with or without the dynamics."""

    file = _peak_file(name=name)
    if dynamics:
        file.add_time_dependence(
            target_model=_MODEL,
            target_parameter="GLP_01_A",
            dynamics_yaml=_TIME_YAML,
            dynamics_model=[_DYNAMICS],
        )
    model = file.model_active
    assert model is not None  # type guard
    for par, value in _TRUTH.items():
        if par in model.parameter_names:
            model._lmfit_pars[par].value = value
    return model


#
def _bleach_model(*, name):
    """Truth model whose clean signal is negative everywhere.

    A negative background of -3 with a peak of 2 on top: a bleach, which no
    counting draw describes.
    """

    file = _peak_file(name=name)
    model = file.model_active
    assert model is not None  # type guard
    model._lmfit_pars["Offset_y0"].set(min=-10.0, value=-3.0)
    model._lmfit_pars["GLP_01_A"].value = 2.0
    return model


#
def _counting_simulator(model, *, counts_per_delay=_COUNTS, seed=_SEED):
    """Photon-counting simulator on *model*."""

    return Simulator(
        model=model,
        detection="photon_counting",
        counts_per_delay=counts_per_delay,
        seed=seed,
    )


#
def _analog_simulator(model, *, noise_level=0.05, noise_type="gaussian", seed=_SEED):
    """Analog simulator on *model*."""

    return Simulator(
        model=model,
        detection="analog",
        noise_level=noise_level,
        noise_type=noise_type,
        seed=seed,
    )


#
def _ratio_of_variance_to_mean(noisy_list):
    """Per-pixel ``var / mean`` over a stack of realizations."""

    stack = np.stack(noisy_list)
    return stack.var(axis=0, ddof=1) / stack.mean(axis=0)


#
#
class TestCountingDeclaresItsScale:
    """Counting output is counts / scale, and the declaration says so.

    Each pixel holds ``Poisson(scale * clean) / scale``, whose variance is
    ``clean / scale`` and whose mean is ``clean``: the ratio of the two is
    ``1 / scale`` at every pixel, independent of the signal. That ratio is
    what a fit weighted by the declaration assumes, so measuring it on the
    simulated array tests the declared number against the data it describes.
    """

    #
    def test_2d_scale_is_counts_over_the_mean_row_total(self):
        model = _truth_model(name="scale_2d")
        sim = _counting_simulator(model)
        clean, _noisy, _noise = sim.simulate_2d()
        expected = _COUNTS / float(np.mean(np.sum(np.abs(clean), axis=1)))
        noise_model = sim.noise_model
        assert noise_model is not None  # type guard
        assert noise_model["noise_type"] == "poisson"
        assert noise_model["scale"] == pytest.approx(expected, rel=1e-12)
        # the budget buys one mean row, so the window holds n_time budgets
        assert noise_model["scale"] * float(np.sum(clean)) == pytest.approx(
            _COUNTS * len(_TIME), rel=1e-12
        )

    #
    def test_2d_variance_over_mean_is_one_over_scale(self):
        model = _truth_model(name="var_2d")
        sim = _counting_simulator(model)
        _clean, noisy_list, _noise_list = sim.simulate_n(
            n=_N_REALIZATIONS, dim=2, show_progress=False
        )
        noise_model = sim.noise_model
        assert noise_model is not None  # type guard
        target = 1.0 / noise_model["scale"]
        ratio = _ratio_of_variance_to_mean(noisy_list)
        deviation = np.abs(ratio / target - 1.0)
        # 5 sigma of the per-pixel estimator over 697 pixels; the worst pixel
        # measured 0.27 (3.9 sigma) over eight seeds
        assert deviation.max() < 5.0 * _RATIO_SD
        # pooling the pixels averages that scatter down by sqrt(n_pixels);
        # measured within 0.7% over the same eight seeds
        pooled = np.mean(np.stack(noisy_list).var(axis=0, ddof=1)) / np.mean(
            np.stack(noisy_list).mean(axis=0)
        )
        assert pooled == pytest.approx(
            target, rel=4.0 * _RATIO_SD / np.sqrt(ratio.size)
        )

    #
    def test_2d_declaration_is_accepted_by_the_file(self):
        model = _truth_model(name="accept_2d")
        sim = _counting_simulator(model)
        _clean, noisy, _noise = sim.simulate_2d()
        noise_model = sim.noise_model
        assert noise_model is not None  # type guard
        file = _peak_file(name="accept_2d_fit", data=noisy)
        file.set_noise(**noise_model)
        assert file.noise_type == "poisson"
        assert file.noise_scale == pytest.approx(noise_model["scale"], rel=1e-12)

    #
    def test_1d_scale_is_counts_over_the_total_signal(self):
        model = _truth_model(name="scale_1d", dynamics=False)
        sim = _counting_simulator(model)
        clean, noisy, _noise = sim.simulate_1d()
        expected = _COUNTS / float(np.sum(np.abs(clean)))
        noise_model = sim.noise_model
        assert noise_model is not None  # type guard
        assert noise_model["noise_type"] == "poisson"
        assert noise_model["scale"] == pytest.approx(expected, rel=1e-12)
        file = _peak_file(name="accept_1d", data=noisy, time=False)
        file.set_noise(**noise_model)
        assert file.noise_scale == pytest.approx(expected, rel=1e-12)

    #
    def test_1d_variance_over_mean_is_one_over_scale(self):
        model = _truth_model(name="var_1d", dynamics=False)
        sim = _counting_simulator(model)
        _clean, noisy_list, _noise_list = sim.simulate_n(
            n=_N_REALIZATIONS, dim=1, show_progress=False
        )
        noise_model = sim.noise_model
        assert noise_model is not None  # type guard
        target = 1.0 / noise_model["scale"]
        ratio = _ratio_of_variance_to_mean(noisy_list)
        deviation = np.abs(ratio / target - 1.0)
        # same 5 sigma bound on 41 pixels; worst measured 0.24 over eight seeds
        assert deviation.max() < 5.0 * _RATIO_SD
        assert float(np.mean(ratio)) == pytest.approx(
            target, rel=4.0 * _RATIO_SD / np.sqrt(ratio.size)
        )


#
#
class TestAnalogGaussianDeclaresItsSigma:
    """The analog gaussian pathway has one constant sigma for the array."""

    #
    def test_declaration_and_sigma_data_agree_with_the_definition(self):
        model = _truth_model(name="gauss_sigma")
        sim = _analog_simulator(model, noise_level=0.1)
        clean, _noisy, _noise = sim.simulate_2d()
        expected = 0.1 * float(np.max(np.abs(clean)))
        assert sim.noise_model == {"noise_type": "gaussian", "sigma": expected}
        assert sim.sigma_data == pytest.approx(expected, rel=1e-12)

    #
    def test_declaration_is_accepted_as_a_constant_sigma(self):
        model = _truth_model(name="gauss_accept")
        sim = _analog_simulator(model, noise_level=0.1)
        _clean, noisy, _noise = sim.simulate_2d()
        noise_model = sim.noise_model
        assert noise_model is not None  # type guard
        file = _peak_file(name="gauss_accept_fit", data=noisy)
        file.set_noise(**noise_model)
        assert file.noise_type == "gaussian"
        assert file.sigma_type == "constant"
        assert file.sigma_data == pytest.approx(noise_model["sigma"], rel=1e-12)


#
#
class TestSnapshotBelongsToTheSimulatedData:
    """Changing the noise settings does not rewrite what was already drawn.

    The saved arrays carry the noise of the draw that produced them; a
    declaration recomputed from the current settings would silently misweight
    a fit of data simulated before the change.
    """

    #
    def test_new_noise_level_leaves_the_snapshot_alone(self):
        model = _truth_model(name="stale_level")
        sim = _analog_simulator(model, noise_level=0.1)
        clean, _noisy, _noise = sim.simulate_2d()
        simulated = 0.1 * float(np.max(np.abs(clean)))

        sim.set_noise_level(0.4)
        assert sim.sigma_data == pytest.approx(simulated, rel=1e-12)
        assert sim.noise_model == {"noise_type": "gaussian", "sigma": simulated}

        sim.simulate_2d()
        assert sim.sigma_data == pytest.approx(4.0 * simulated, rel=1e-12)

    #
    def test_new_noise_type_leaves_the_snapshot_alone(self):
        model = _truth_model(name="stale_type")
        sim = _analog_simulator(model, noise_level=0.1)
        clean, _noisy, _noise = sim.simulate_2d()
        simulated = 0.1 * float(np.max(np.abs(clean)))

        sim.set_noise_type("none")
        assert sim.sigma_data == pytest.approx(simulated, rel=1e-12)
        assert sim.noise_model == {"noise_type": "gaussian", "sigma": simulated}

        sim.simulate_2d()
        assert sim.sigma_data is None
        assert sim.noise_model is None

    #
    def test_saved_sigma_is_the_simulated_one(self, tmp_path, monkeypatch):
        model = _truth_model(name="stale_save")
        sim = _analog_simulator(model, noise_level=0.1)
        clean, _noisy, _noise = sim.simulate_2d()
        simulated = 0.1 * float(np.max(np.abs(clean)))

        sim.set_noise_level(0.4)
        # save_data always writes below cwd/simulated_data
        monkeypatch.chdir(tmp_path)
        sim.save_data(filepath="stale.h5", show_output=0)
        with h5py.File(tmp_path / "simulated_data" / "stale.h5", "r") as f:
            saved = f["metadata"].attrs["sigma_data"]
        assert saved == pytest.approx(simulated, rel=1e-12)

    #
    def test_saved_settings_are_the_simulated_ones(self, tmp_path, monkeypatch):
        """Draw, change the noise level and type, save: the file describes
        the draw, not the current settings."""

        model = _truth_model(name="stale_settings")
        sim = _analog_simulator(model, noise_level=0.1)
        clean, _noisy, _noise = sim.simulate_2d()
        simulated = 0.1 * float(np.max(np.abs(clean)))
        sim.set_noise_level(0.4)
        sim.set_noise_type("poisson")
        monkeypatch.chdir(tmp_path)
        sim.save_data(filepath="stale_settings.h5", show_output=0)
        with h5py.File(tmp_path / "simulated_data" / "stale_settings.h5", "r") as f:
            attrs = dict(f["metadata"].attrs)
        assert attrs["noise_level"] == pytest.approx(0.1)
        assert attrs["noise_type"] == "gaussian"
        assert attrs["sigma_data"] == pytest.approx(simulated, rel=1e-12)
        assert attrs["seed"] == _SEED

    #
    def test_a_standalone_add_noise_does_not_touch_the_record(
        self, tmp_path, monkeypatch
    ):
        """add_noise on the caller's array is a helper: the stored
        simulation and its record are untouched, and save_data writes
        the record of the stored arrays."""

        model = _truth_model(name="helper_draw")
        sim = _analog_simulator(model, noise_level=0.1)
        clean, noisy, _noise = sim.simulate_2d()
        declared, sigma = sim.noise_model, sim.sigma_data
        sim.set_noise_level(0.4)
        other_noisy, _other_noise = sim.add_noise(2.0 * clean, dim=2)
        assert not np.array_equal(other_noisy, noisy)
        assert sim.noise_model == declared and sim.sigma_data == sigma
        assert sim.data_noisy is noisy
        monkeypatch.chdir(tmp_path)
        sim.save_data(filepath="helper_draw.h5", show_output=0)
        with h5py.File(tmp_path / "simulated_data" / "helper_draw.h5", "r") as f:
            attrs = dict(f["metadata"].attrs)
            saved_noisy = require_dataset(f["simulated_data/000000"], "000000")[()]
        assert attrs["noise_level"] == pytest.approx(0.1)
        assert attrs["sigma_data"] == pytest.approx(sigma, rel=1e-12)
        np.testing.assert_array_equal(saved_noisy, noisy)

    #
    def test_a_new_clean_array_starts_a_new_simulation(self, tmp_path, monkeypatch):
        """generate_clean_data replaces the stored simulation as a unit: the
        old noisy arrays and record do not describe the new clean array."""

        model = _truth_model(name="new_clean")
        sim = _analog_simulator(model, noise_level=0.1)
        sim.simulate_2d()
        sim.generate_clean_data(dim=2)
        assert sim.data_noisy is None and sim.noise is None
        assert sim.noise_model is None and sim.sigma_data is None
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError):
            sim.save_data(filepath="new_clean.h5", show_output=0)


#
#
class TestTheDrawIsTheDeclaredModel:
    """The simulator draws from the NoiseModel it declares, as lmfit fits
    weight with it; replaying the draw from the declaration reproduces the
    noisy array exactly, for the same seed."""

    #
    def test_counting_draw_is_poisson_on_clean_times_scale(self):
        model = _truth_model(name="replay_counting")
        sim = _counting_simulator(model)
        clean, noisy, noise = sim.simulate_2d()
        declared = sim.noise_model
        assert declared is not None  # type guard
        scale = declared["scale"]
        rng = np.random.default_rng(_SEED)
        np.testing.assert_array_equal(noisy, rng.poisson(clean * scale) / scale)
        np.testing.assert_array_equal(noise, noisy - clean)

    #
    def test_analog_gaussian_draw_is_normal_with_the_declared_sigma(self):
        model = _truth_model(name="replay_gaussian")
        sim = _analog_simulator(model, noise_level=0.1)
        clean, noisy, noise = sim.simulate_2d()
        declared = sim.noise_model
        assert declared is not None  # type guard
        rng = np.random.default_rng(_SEED)
        np.testing.assert_array_equal(
            noise, rng.normal(0, declared["sigma"], clean.shape)
        )
        np.testing.assert_array_equal(noisy, clean + noise)

    #
    def test_analog_poisson_draw_is_poisson_on_clean_times_scale(self):
        model = _truth_model(name="replay_analog_poisson")
        sim = _analog_simulator(model, noise_level=0.02, noise_type="poisson")
        clean, noisy, noise = sim.simulate_2d()
        declared = sim.noise_model
        assert declared is not None  # type guard
        scale = declared["scale"]
        rng = np.random.default_rng(_SEED)
        np.testing.assert_array_equal(noise, rng.poisson(clean * scale) / scale - clean)
        np.testing.assert_array_equal(noisy, clean + noise)

    #
    def test_declaration_round_trips_into_the_file_noise_model(self):
        model = _truth_model(name="round_trip")
        for sim in (
            _counting_simulator(model),
            _analog_simulator(model, noise_level=0.1),
            _analog_simulator(model, noise_level=0.02, noise_type="poisson"),
        ):
            _clean, noisy, _noise = sim.simulate_2d()
            declared = sim.noise_model
            assert declared is not None  # type guard
            file = _peak_file(
                name=f"round_trip_{sim.detection}_{sim.noise_type}", data=noisy
            )
            file.set_noise(**declared)
            drawn = sim._noise_applied.model  # the declared model itself
            assert drawn is not None  # type guard
            assert file.noise.kind == drawn.kind
            assert file.noise.scale == drawn.scale
            assert file.noise.sigma == drawn.sigma


#
#
class TestSignalsNoModelDescribesAreRefused:
    """A count cannot be negative, and a gaussian draw needs a sigma: the
    simulator refuses what no NoiseModel describes instead of drawing
    something a fit cannot weight."""

    #
    @pytest.mark.parametrize("dim", [1, 2])
    def test_counting_refuses_a_negative_signal(self, dim):
        model = _bleach_model(name=f"refuse_counting_{dim}")
        sim = _counting_simulator(model, counts_per_delay=_BLEACH_COUNTS)
        with pytest.raises(ValueError, match="non-negative"):
            sim.simulate_1d() if dim == 1 else sim.simulate_2d()
        assert sim.data_noisy is None and sim.noise_model is None  # nothing drawn

    #
    def test_analog_poisson_refuses_a_negative_signal(self):
        model = _bleach_model(name="refuse_analog_poisson")
        sim = _analog_simulator(model, noise_level=0.02, noise_type="poisson")
        with pytest.raises(ValueError, match="non-negative"):
            sim.simulate_2d()
        assert sim.data_noisy is None and sim.noise_model is None  # nothing drawn

    #
    def test_analog_poisson_refuses_a_negative_noise_level(self):
        model = _truth_model(name="refuse_negative_level")
        sim = _analog_simulator(model, noise_level=-0.02, noise_type="poisson")
        with pytest.raises(ValueError, match="negative"):
            sim.simulate_2d()
        assert sim.data_noisy is None and sim.noise_model is None  # nothing drawn

    #
    def test_analog_gaussian_refuses_a_zero_sigma(self):
        model = _truth_model(name="refuse_zero_sigma")
        sim = _analog_simulator(model, noise_level=0.0)
        with pytest.raises(ValueError, match="noise_type='none'"):
            sim.simulate_2d()
        assert sim.data_noisy is None and sim.noise_model is None  # nothing drawn

    #
    def test_a_gaussian_draw_on_a_negative_signal_is_fine(self):
        """Gaussian noise describes signed data; only counts are non-negative."""

        model = _bleach_model(name="gaussian_bleach")
        sim = _analog_simulator(model, noise_level=0.1)
        clean, noisy, _noise = sim.simulate_2d()
        assert clean.max() < 0.0
        declared = sim.noise_model
        assert declared is not None  # type guard
        assert declared["noise_type"] == "gaussian"


#
#
class TestOtherPathways:
    """The analog poisson scale, and the cases without a declaration."""

    #
    def test_analog_poisson_scale_follows_the_noise_level(self):
        model = _truth_model(name="analog_poisson")
        sim = _analog_simulator(model, noise_level=0.02, noise_type="poisson")
        sim.simulate_2d()
        assert sim.noise_model == {
            "noise_type": "poisson",
            "scale": 1.0 / (0.02 + 1e-10),
        }
        assert sim.sigma_data is None

    #
    def test_no_noise_has_nothing_to_declare(self):
        model = _truth_model(name="no_noise")
        sim = _analog_simulator(model, noise_level=0.0, noise_type="none")
        sim.simulate_2d()
        assert sim.noise_model is None
        assert sim.sigma_data is None

    #
    def test_nothing_is_declared_before_the_first_simulation(self):
        model = _truth_model(name="unsimulated")
        sim = _counting_simulator(model)
        assert sim.noise_model is None
        assert sim.sigma_data is None
