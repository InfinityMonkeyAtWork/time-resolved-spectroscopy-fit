# Active Plan: noise-model weighting + sensitivity (branch `sigma-weighting`)

Target: v0.17.0 (main released 0.16.0 for export-by-handle on 2026-09-21). Started 2026-09-17. TODO items covered: "Weight the residual by
per-point sigma", "Future `sigma_type` expansion", the `Simulator.sigma_data`
snapshot case inside the array-mutation item, and the unmerged
`trspecfit.sensitivity` module (first commit on this branch, 26 tests green).

Goal in one line: quoted `stderr`, CI and MCMC widths follow from a declared
noise model instead of assuming uniform Gaussian noise, and the Cramér-Rao
bound from `sensitivity` becomes the independent check that they are right.

## Design (settled 2026-09-17, revised after review 2026-09-21)

**Knobs.** `File.set_noise(noise_type, *, sigma=None, scale=None)`:

| `noise_type` | value | residual per point | covariance |
|---|---|---|---|
| `unknown` (default) | none | `d - m` (today) | lmfit, redchi-scaled |
| `gaussian` | `sigma`: scalar or array broadcastable to `data.shape` | `(d - m) / σ` | `scale_covar=False` |
| `poisson` | `scale`: counts per data unit (default 1 = data are counts) | deviance: `sign(d-m)·sqrt(2·scale·(m - d + d·ln(d/m)))` | `scale_covar=False` |

`set_sigma(x)` stays as the Gaussian shortcut. The Poisson deviance is the exact
likelihood: zero-count bins need no floor, `emcee(is_weighted=True)` samples the
exact posterior, and at `d = m` the Gauss-Newton matrix equals the Poisson
Fisher matrix that `sensitivity.fisher_matrix` computes independently.

Rejected alternative (measured 2026-09-21, peak + flat background toy, 200
pixels, 400 seeds): frozen data-based weights `σ² = max(d, 1 count)/scale`
give the same scatter/`stderr` as the deviance for peak parameters, but the
fitted background level lands 2.5 quoted `stderr` below the truth at 21
counts per background pixel and 0.7 at 215 (bias ≈ one count per pixel, so
bias/`stderr` ~ sqrt(n_pixels/counts)); the deviance stays within 0.1
`stderr` at every level. Error bars must cover the truth, and frozen weights
would need ~1000 counts per background pixel to do so. Cost of the deviance:
the floor policy, the Jacobian limit branch, one extra forward evaluation per
Jacobian call.

**Residual evaluation policy.** `poisson` requires `data >= 0` on the file's
`data` array, checked once at `set_noise` (the fit window does not exist yet;
every later view is a slice or mean of that array, and corrections raise, see
below) → `ValueError` with a hint to use `gaussian` for dark-subtracted or
difference data. Inside the residual `m` has a floor `ε/scale` (a fixed
fraction ε of one count, in data units). Where `d > 0` the residual below the
floor is the linear extension with the slope at the floor, so residual and
Jacobian stay consistent and `leastsq` keeps a nonzero gradient that pushes
`m` back up (a hard clamp would leave a large residual with a zero analytic
Jacobian and stall `Dfun`). Where `d = 0`, `m ≤ 0` scores like `m = 0` (a
plateau with zero slope, exactly what the likelihood says). Nothing raises
inside the optimizer, so `leastsq` never aborts and emcee walkers never die.
The deviance term is `scipy.special.kl_div(d, m)`, exact at `d = 0`.

**Jacobian factor.** `∂r/∂m = −sqrt(scale/2)·|m − d| / (m·sqrt(D))` with the
analytic limit `−sqrt(scale/m)` on the branch `|m − d| ≤ 1e-6·m` (covers
`d = m` exactly, hence noiseless data; the closed form loses digits there).
Gaussian: `−1/σ`. Below the floor (`d > 0`) the factor is its value at the
floor. At `d = 0` the residual has no floor (the deviance is finite at
`m = 0`), so the factor is the exact `−sqrt(scale/(2m))` for `m > 0` and `0`
on the `m ≤ 0` plateau; flooring it there would make `JᵀJ` claim
information from bins whose residual is flat (review, 2026-09-21). The factor multiplies the model Jacobian `∂m/∂θ`; `jacobian_fun` and
`jacobian_fun_project` already return `−∂m/∂θ` (the unweighted residual
Jacobian), so their output is multiplied by `−∂r/∂m`, i.e. `1/σ` for
Gaussian. Applied in numpy after one forward evaluation of `m`; no JAX code
changes. lmfit's finite-difference Jacobian (no `Dfun`) needs nothing. The
finite-difference test in A3 is the sign check.

**Reductions to the fit view.** Baseline over `n` slices: Gaussian
`sqrt(Σ_t σ²)/n` per pixel (constant σ → `σ/√n` as today); Poisson `scale·n`
(sum of Poisson is Poisson, so the baseline fit stays exact). `fit_spectrum`
with `time_range` gets the same reduction (documented gap today). SbS takes the
slice, 2D the fit window. Joint fits concatenate per-file residuals, each in
likelihood units, so files may mix `gaussian` and `poisson` at no extra cost;
mixing `unknown` with a weighted file raises.

**Corrections.** `subtract_dark` / `calibrate_data` / `reset_dark` /
`reset_calibration` raise `ValueError` while a weighted model is declared:
σ describes the `data` view and nothing is propagated. Drop it with
`set_noise('unknown')`, correct, re-declare. No clearing, no warning, no
stale state a later fit could miss.

**Identity and storage.** Any declared noise model (kind, `scale`, σ scalar or
array digest) enters `optimization_hash` through the per-file `input_files`
entry, inside the same JSON payload so float encoding is shared. This includes
constant σ: with `scale_covar=False` it changes the covariance, and a same-hash
append compares fitted values, finds them equal, and returns without
refreshing stored `stderr`, so an unkeyed σ would leave stale error bars in
the archive. `unknown` hashes as today (existing archives unaffected). Slot gains `noise_scale` attr and an optional view-aligned `sigma`
dataset (per-point Gaussian); schema stays 7, additive. `sigma_eff` stays what
it is today: the view reduction of a constant Gaussian σ (σ, or `σ/√n` for
baseline / `time_range` means); NaN for `poisson`, per-point σ and `unknown`.
The identity `chi2_red = chi2_red_raw / sigma_eff²` holds exactly where
`sigma_eff` is finite and is not claimed elsewhere.

**Metrics.** `chi2_raw` unweighted (as now). `chi2` = Σ weighted residual²
(Gaussian) or the deviance (Poisson), taken from the weighted residual
directly: per slice for SbS, summed over files for joint. With a declared noise
model AIC/BIC use `chi2 + 2k` / `chi2 + k·ln n`, including the joint record
(replaces today's profiled form on the concatenated residual); `unknown` keeps
today's profiled form. `chi2_red`, `aic` and `bic` are therefore comparable
only within one noise model: `select='best'` by any of the three, and
`compare_models`' conflict detection, key on (kind, `noise_scale`, σ digest)
with `unknown` as its own key, and refuse a mixed group (today `aic`/`bic`
have no σ check at all, and the finite-`sigma_eff` test lets two Poisson
scales or two σ arrays pass as comparable). `fit_results._has_any_sigma` and
the other `isfinite(sigma_data)` presence signals key on
`noise_type != 'unknown'`.

**MCMC and CI.** Any declared noise forces `is_weighted=True` (the weighted
residual is the log-likelihood); the `MC` σ knobs raise under declared noise;
`is_weighted=True` under `unknown` raises. `conf_interval` gets a `prob_func`
under declared noise: the χ² threshold `chi2.cdf(Δchisqr, nfix)` (lmfit's
default F-test profiles out a variance the noise model already fixes, so the
two agree only at `chi2_red ≈ 1`). `unknown` keeps the F-test.

**Simulator.** Snapshot the scale factor `add_noise` actually applied
(counting: `counts_per_delay / total_signal` in 1D, `counts_per_delay /
mean_row_total` in 2D; analog poisson: `1/(noise_level + 1e-10)`) and σ for
analog gaussian. `Simulator.noise_model` returns the `set_noise(**...)`
kwargs: `poisson` when the noisy output is non-negative, else per-point
`gaussian` with `σ = sqrt(|clean|/scale)` (signed bleach output).
`save_data` writes `noise_type` / `noise_scale` next to the `clean_data` it
already stores, which together determine that σ map; no σ dataset.
Deferred (2026-09-22): the file-format half moves to the simulator
schema-compliance work (TODO item 1), so no field is added to a layout about
to be redesigned; the declaration is derivable from the stored detection
metadata and `clean_data` (counting: `counts_per_delay / mean row total`;
analog gaussian: `noise_level · max|clean|`; analog poisson:
`1/(noise_level + 1e-10)`), so nothing a 0.17.0 file stores is lost. The
snapshot helper computes the declaration from exactly those inputs, so the
compliance work reuses it for writing and for reading older files.

**Scope kept out.** Noise estimation helpers (temporal-RMS σ, photon-transfer
`scale`; notebook 12 takes `scale` from the simulator snapshot); automatic
noise detection; propagating σ through corrections; compound (Poisson +
read-noise) likelihood; per-slice-stderr trace fitting (gets its weighting
hook from this work). `sigma_source` stays `user_supplied`. Recorded in TODO at
the end.

## Decisions taken in this plan (veto while reviewing)

- New module `utils/noise.py` owns `NoiseModel` (kind, σ, scale), view
  reductions, `apply()` / `jacobian_factor()`, validation, and the per-file
  list applied per segment for joint fits. `fit_io` keeps its constants as
  aliases.
- Noise reaches `residual_fun` / both Jacobians as `noise=` via lmfit
  `fcn_kws` (lmfit forwards `userkws` to the objective and `Dfun`); `const`
  stays untouched.
- Poisson Jacobian factor needs `m`: one extra forward evaluation per Jacobian
  call (simplest); value-and-jacobian fusion is a later optimization if
  profiling asks for it.
- `input_files` entries grow a 4th element `noise_json`; 3-element entries
  read as `unknown`.
- Names: API `scale`, attr/YAML/slot `noise_scale`; `sigma_type` derived:
  `constant` | `per_point`.
- Per-point σ is broadcast to `data.shape` at `set_noise`, copied and frozen.

## Steps

### A. Noise model core
- [x] A1 `utils/noise.py`: `NoiseModel`, `for_view()`, `apply()`,
      `jacobian_factor()`, `data >= 0` validation, per-segment application.
- [x] A2 `fitlib.residual_fun(noise=)`, `fit_wrapper(noise=)` → `fcn_kws`,
      `scale_covar=False` when weighted; `jacobian_fun` /
      `jacobian_fun_project` apply the factor.
- [x] A3 Tests `tests/test_noise_model.py`: deviance limits (`d=0`, `d=m`),
      sum equals the deviance formula, floor policy (`d > 0`: linear below
      the floor with nonzero slope; `d = 0`: equals `m = 0`), Gaussian
      broadcasting, full residual Jacobian vs finite differences (sign) incl.
      the limit branch and below the floor, segment concatenation.

### B. File API and reductions
- [x] B1 `File.set_noise`, `set_sigma` delegating, `File.noise` attribute with
      `noise_type` / `sigma_data` read-only properties for existing callers.
- [x] B2 Project defaults `noise_type` / `sigma_data` / `noise_scale` in
      `_set_defaults` + project.yaml key handling.
- [x] B3 Corrections raise while a weighted model is declared.
- [x] B4 Wire view reductions into `fit_baseline`, `fit_spectrum`, `fit_sbs`,
      `fit_2d`; pass `noise=` to `fit_wrapper`. The slot dict keeps reporting
      the File-level noise through the new properties; threading the view
      noise into slots and metrics is C.
- [x] B5 `Project.fit_2d`: per-file view noise list in `Project.files`
      order (the order of `concat_data`, not the sorted `input_files`); mixed
      `unknown`/weighted raises.
- [x] B6 Tests: validation and domain errors, corrections raise, reductions
      pinned (constant, per-point, Poisson baseline), joint mixed-unknown
      raises, joint Gaussian+Poisson runs, `jacobian_fun_project` with
      `SegmentedNoise` vs finite differences (untested after A).

### C. Identity, metrics, archive
- [x] C1 `compute_fit_metrics(noise=)`: `chi2` from the weighted residual,
      AIC/BIC forms, `sigma_eff` as the constant-Gaussian view reduction (NaN
      otherwise); SbS per-slice and joint callers. The joint helper today
      computes profiled AIC/BIC from the concatenated raw residual and only
      then overwrites `chi2`: replace those with the summed-objective forms.
- [x] C2 `SavedFitSlot` + builders: `noise_scale`, `sigma` array, `sigma_data`
      NaN for arrays; `_has_any_sigma` and other presence signals key on
      `noise_type`.
- [x] C3 `encode_input_files` 4th element; `compute_optimization_hash`
      unchanged in signature; reader fallback for 3-element entries; check
      `variants()` / `diff()` parsing.
- [x] C4 Writer/reader: `noise_scale` attr, optional `sigma` dataset.
- [x] C5 Noise-model key for `select='best'` (`chi2_red`, `aic`, `bic`) and
      for `compare_models` conflict detection (`_sigma_conflicts` and the
      σ-scaled column drop), replacing the finite-`sigma_eff` tests. Docs:
      `fit_archive_principles.md` amend the "σ is post-hoc" rule and the
      attachment table (dated; the noise model is keyed because the writer
      does not refresh stored `stderr` on a same-hash append);
      `fit_archive_schema.md` slot attrs/dataset, the additive note and the
      value sets `noise_type ∈ {unknown, gaussian, poisson}` /
      `sigma_type ∈ {constant, per_point}` that `validate_noise_metadata`
      accepts since B; `compare_models` docstring.
- [x] C6 Tests: hash changes with any declared noise (incl. constant σ) and
      not with `unknown`; roundtrip with per-point σ and Poisson scale;
      3-element `input_files` reads; `sigma_eff` finite only for constant
      Gaussian and the identity pinned there; `select='best'` by `aic` and
      `compare_models` refuse a group mixing two Poisson scales, and one
      mixing `unknown` with a declared model.

### D. MCMC and CI
- [x] D1 `MC.resolve(..., weighted)`: declared noise forces `is_weighted`,
      rejects σ knobs; `is_weighted=True` under `unknown` raises (at
      sampling time, like the `nwalkers` check). A bool, not the planned
      `noise_kind`: a joint `SegmentedNoise` has no single kind and
      `fit_wrapper` already derives `weighted` for `scale_covar`.
- [x] D2 `conf_interval(prob_func=χ² threshold)` under declared noise.
      lmfit 1.3.4's `prob_func(best_fit, new_fit)` takes the two results,
      so `fitlib._chi2_compare` derives `nfix` from their `nvarys`. The
      phase-C bracketing failure (`f(a)` and `f(b)` have the same sign)
      reproduces with a misspecified model on near-noiseless data and a
      constant σ: the F-test ratio is not monotonic along the profile
      there; the χ² threshold completes. Pinned in D4.
- [x] D3 Tests in `test_mc_settings.py`: forced/raised cases; seeded short
      chain on Poisson data: posterior widths / `stderr` measured 0.93–1.11
      over 200–800 steps, asserted 0.85–1.15.
- [x] D4 CI tests in `test_noise_ci.py` (`conf_interval` refuses fewer than
      two varying parameters): constant σ declared 2× too small halves the
      χ²-threshold CI (measured 1.96–2.04; the F-test ratio stays at 1.00);
      near-noiseless data with a declared σ gives half-widths equal to
      `stderr` (F-test: 2–9% of it); Poisson counts give half-widths within
      7% of `stderr` (the 20% band holds from ~90 peak counts upward; below
      that the profile asymmetry is the interval's real shape); the phase-C
      misspecified-model case completes.

### E. Verification against the bound
- [x] E1 Deterministic, noiseless `d = m` (limit branch): `JᵀJ` from the
      noise-weighted Jacobian equals `sensitivity.fisher_matrix(model,
      counts=scale·Σm)`: 2D analytic Jacobian to 4.4e-7 (entries scaled by
      the diagonals), 1D finite-difference Jacobian to 2e-6 on the diagonal.
      `d = m` must hold bit for bit: `kl_div` evaluates with absolute error
      ~eps·m at `d ≈ m` and the square root turns that into a residual
      floor ~sqrt(scale·eps·m) ≈ 4e-7 σ per point; irrelevant for data
      (residuals are O(1)), so the tests feed the backend's own curve back
      as data.
- [x] E2 Seeded noisy fit: `stderr` within 10% of the CRB at 1.4e6 counts
      (one-seed spot check; measured max deviation 2–9% over five seeds,
      the scatter almost all in `tau`). The population statement lives in
      E3: the 256-seed mean of `stderr` within 2% of the CRB (measured
      1.000–1.009).
- [x] E3 Slow test reproducing the 2026-09-15 control: 256 seeds, seed
      scatter / `stderr` 0.93–1.06 under `poisson` (band 0.85–1.15);
      1.39–1.79 on the peak parameters under `unknown` and 0.60 on the
      background offset (one misweighting, not a missing global factor;
      both bounds belong to this peak-to-background contrast). 23 s.

### F. Simulator
- [x] F1 Snapshot the applied scale factor and σ at `add_noise` time;
      `Simulator.noise_model`; `sigma_data` kept (now the recorded value,
      which also fixes the stale-σ case in `save_data` and the sweep).
      Module-private `_NoiseSnapshot` built by `_snapshot_noise(detection
      settings, clean array)`, the derivation the deferred F2 reuses. The
      poisson-vs-per-point choice keys on the clean array's sign, not on
      one realization (a negative pixel can draw zero counts).
- [x] F2 Deferred (2026-09-22) to the simulator schema-compliance work
      (TODO item 1): `save_data` and the sweep output carry the noise
      declaration in the compliant form; derivable meanwhile from the
      stored detection metadata and `clean_data` (see Design, Simulator).
- [x] F3 Tests (`tests/test_simulator_noise.py`): variance/mean of counting
      output equals `1/scale` (400 realizations; per-pixel band 5 sd of the
      variance estimator, pooled band 4 sd/√n_pixels; worst pixel measured
      3.9 sd over 8 seeds); stale case (`set_noise_level` after `simulate`)
      no longer changes the snapshot, incl. the saved `sigma_data` attr;
      signed output yields the per-point Gaussian form (σ = 0 pixels are
      rejected by `set_noise`, so the bleach model carries a background).

### G. Sensitivity module
- [x] G1 Docstring: fix the `counts` convention for 2D (total over the window
      = `counts_per_delay × n_time`); separate the two facts about unweighted
      fits: measured on the E3 geometry (256 seeds) the unweighted scatter is
      0.98–1.24× the bound (the old "10–20% above" was the wrong emphasis),
      while the quoted `stderr` is 1.4–1.8× too small on peak parameters and
      too large on the background; state the advanced tier (`mcp.Model` in,
      `docs/stability.md` makes no commitment for it yet).
- [x] G2 `sphinx -W` clean build succeeds with `docs/api/sensitivity.rst`.

### H. Examples and docs
Teaching split (decided 2026-09-22): 01–04 stay about fit mechanics and
declare no noise; 12 is the correct path in one line; a new 13 shows what
the `unknown` default gets wrong on the same data, with every comparison
computed live. No separate Gaussian notebook: 10 already declares a known
Gaussian σ where the noise is Gaussian, and the model-free σ estimate is one
fallback cell in 13. The generators store data in model units, so the
counting scale is `counts_per_delay / mean row total` of the data (0.07%
from the truth total); raw counts take the default `scale=1`.

- [x] H1 Notebook 12 rewrite, the correct path: §0 re-run 01; §1 one
      `set_noise('poisson', scale=...)` line (comment: why the scale, and
      that raw counts need none) and the χ²_red ≈ 1 gate; §2 one fit, three
      tiers (`stderr`, χ²-threshold CI, weighted MCMC with no nuisance);
      §3 side by side against the truth and against the Cramér-Rao bound
      (`sensitivity.crb` at the budget); §4 the 2D chain and how to read
      its convergence (the off-scale demo moves to 13); tips. 35 → 32
      cells, executes in 1 min 52 s (2D chain 85 s). `stderr`/CRB 0.99–1.00
      on four parameters, 0.91/1.08 on `F`/`m` from evaluating the bound at
      the truth instead of the fitted point (1.000 at the fitted point).
      The gate reads the slot's `chi2_red` via `results.find(...)[-1]`:
      01's slots are `unknown`, so `compare_models` withholds the calibrated
      columns for the mixed group (13's lesson). `/check-example` passes
      except the three forward links to 13 until H1b lands.
- [ ] H1b New notebook 13, when the noise model is missing: `%run` 12 under
      `%%capture` (as 12 runs 01) so 12's fits are live and no number is
      copied; open with the warning that `unknown` is the default; refit
      under `set_noise('unknown')` with 12's settings and table the three
      tiers next to 12's (consistently too small on the peak, too large on
      the background, and why: one variance spread over a window whose
      counting noise peaks with the signal); `compare_models` refusing to
      rank across noise models; the `__lnsigma` nuisance and the off-scale
      unconverged chain; the baseline-block σ estimate as the fallback for
      non-counting data, one cell, with the gate flagging a wrong σ. Mirrors
      12 section for section. Register in the examples README and the docs
      examples index. `/check-example`.
- [x] H2 Notebook 01: one sentence at the first fit report (error bars are
      lmfit's default; 12 makes them right). Notebook 10 re-executed (its
      `set_sigma` weights the fits since B): no quoted number changed (the
      prose quotes inputs and verdicts only), no conclusion flipped; AIC/BIC
      printouts moved from the profiled to the weighted form and the
      misfitting models' `stderr` lost their redchi inflation; three prose
      edits (`set_sigma` weights the residual; `chi2_red_raw` is the
      unweighted diagnostic; σ is File state, `set_sigma` is no longer the
      only entry point) and the YAML comments say shift-and-decay like the
      notebook. `/check-example` both. 02/03/04 unchanged.
- [ ] H3 CHANGELOG 0.17.0 (incl. behaviour changes since B: `set_sigma`
      now weights the fit; corrections refuse to run under declared noise;
      a project default `sigma_data` without `noise_type: gaussian` raises;
      `noise_type` / `sigma_data` / `sigma_type` are read-only properties;
      declared noise forces weighted MCMC and its σ knobs raise,
      `MC(is_weighted=True)` under `unknown` raises; `conf_interval` uses
      the χ² threshold under declared noise);
      notebooks 12 and 13; TODO.md: retire the two noise items, note the
      simulator snapshot fix, add follow-ups (noise estimation helpers,
      compound likelihood, σ through corrections, declaring noise in the
      counting notebooks 01/02/03/04/20/21 once uncertainties enter them).
- [ ] H4 `pyproject.toml` version 0.17.0 at commit time.

### I. Verify pass
- [ ] `pytest -q`, `pytest -m slow` for E3, pre-commit (ruff, mypy
      `--no-incremental`, pyright), `sphinx -W`, execute 03 and 12 then
      `scripts/normalize_notebooks.py`.
