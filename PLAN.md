# Active Plan: noise-model weighting + sensitivity (branch `sigma-weighting`)

Target: v0.17.0 (main released 0.16.0 for export-by-handle on 2026-09-21). Started 2026-09-17. TODO items covered: "Weight the residual by
per-point sigma", "Future `sigma_type` expansion", the `Simulator.sigma_data`
snapshot case inside the array-mutation item, and the unmerged
`trspecfit.sensitivity` module (first commit on this branch, 26 tests green).

Goal in one line: quoted `stderr`, CI and MCMC widths follow from a declared
noise model instead of assuming uniform Gaussian noise, and the Cramér-Rao
bound from `sensitivity` becomes the independent check that they are right.

## Design (settled 2026-09-17)

**Knobs.** `File.set_noise(noise_type, *, sigma=None, scale=None)`:

| `noise_type` | value | residual per point | covariance |
|---|---|---|---|
| `unknown` (default) | none | `d - m` (today) | lmfit, redchi-scaled |
| `gaussian` | `sigma`: scalar or array broadcastable to `data.shape` | `(d - m) / σ` | `scale_covar=False` |
| `poisson` | `scale`: counts per data unit (default 1 = data are counts) | deviance: `sign(d-m)·sqrt(2·scale·(m - d + d·ln(d/m)))` | `scale_covar=False` |

`set_sigma(x)` stays as the Gaussian shortcut. The Poisson deviance is the exact
likelihood: zero-count bins need no floor, `emcee(is_weighted=True)` samples the
exact posterior, and the Gauss-Newton covariance equals the inverse Poisson
Fisher matrix, which `sensitivity.fisher_matrix` computes independently.

**Reductions to the fit view.** Baseline over `n` slices: Gaussian
`sqrt(Σ_t σ²)/n` per pixel (constant σ → `σ/√n` as today); Poisson `scale·n`
(sum of Poisson is Poisson, so the baseline fit stays exact). `fit_spectrum`
with `time_range` gets the same reduction (documented gap today). SbS takes the
slice, 2D the fit window. Joint fits concatenate per-file residuals, which are
already in likelihood units; files may mix `gaussian` and `poisson`, but mixing
`unknown` with a weighted file raises.

**Domain.** `poisson` requires `data >= 0` on the fit view and `m > 0` wherever
`d > 0`; violated → `ValueError` with a hint to use `gaussian` for
dark-subtracted or difference data. Any later `subtract_dark` /
`calibrate_data` / reset clears the noise model with a warning (σ describes the
`data` view; nothing is propagated through corrections).

**Identity and storage.** A noise model that weights the residual moves the
optimum, so it enters `optimization_hash` through the per-file `input_files`
entry. `unknown` hashes as today (existing archives unaffected). Slot gains
`noise_scale` attr and an optional view-aligned `sigma` dataset (per-point
Gaussian); schema stays 7, additive. `sigma_eff` keeps its documented identity
`chi2_red = chi2_red_raw / sigma_eff²` by definition: `sqrt(chi2_raw / chi2)`.

**Metrics.** `chi2_raw` unweighted (as now). `chi2` = Σ weighted residual²
(Gaussian) or the deviance (Poisson). With a declared noise model AIC/BIC use
`chi2 + 2k` / `chi2 + k·ln n`; with `unknown` they keep today's profiled form.
`compare_models` already refuses to mix `sigma_eff` tiers within a group.

**MCMC.** `poisson` forces `is_weighted=True` and rejects explicit `sigma_*`
knobs. `gaussian` keeps the `is_weighted` default False, where `__lnsigma` is a
multiplicative scale factor on the supplied σ (starts near 1 via the existing
RMS derivation). `is_weighted=True` with `unknown` noise raises.

**Estimation.** `File.estimate_noise()` over the baseline block (signal-free in
time by construction): Gaussian σ = RMS temporal std over the fit window
(notebook 12's formula); Poisson `scale` = slope of temporal variance vs
temporal mean across pixels (photon-transfer method, also a noise-type
diagnostic). Explicit call, returns values, never sets them.

**Scope kept out.** Automatic noise detection; propagating σ through
corrections; compound (Poisson + read-noise) likelihood; per-slice-stderr
trace fitting (gets its weighting hook from this work). Recorded in TODO at
the end.

## Decisions taken in this plan (veto while reviewing)

- New module `utils/noise.py` owns `NoiseModel` (kind, σ, scale), view
  reductions, `apply()` / `jacobian_factor()`, validation, segmented joint form.
  `fit_io` keeps its constants as aliases.
- Noise reaches `residual_fun` / both Jacobians as `noise=` via lmfit
  `fcn_kws` (lmfit forwards `userkws` to the objective and `Dfun`); `const`
  stays untouched.
- Poisson Jacobian factor needs `m`: one extra forward evaluation per Jacobian
  call on the JAX path (simplest); value-and-jacobian fusion is a later
  optimization if profiling asks for it.
- `input_files` entries grow a 4th element `noise_json`; 3-element entries
  read as `unknown`.
- Names: API `scale`, attr/YAML/slot `noise_scale`; `sigma_source` gains
  `estimated` and `simulated`; `sigma_type` derived: `constant` | `per_point`.
- Per-point σ is broadcast to `data.shape` at `set_noise`, copied and frozen.

## Steps

### A. Noise model core
- [ ] A1 `utils/noise.py`: `NoiseModel`, `for_view()`, `apply()`,
      `jacobian_factor()`, domain validation, `SegmentedNoise` for joint.
- [ ] A2 `fitlib.residual_fun(noise=)`, `fit_wrapper(noise=)` → `fcn_kws`,
      `scale_covar=False` when weighted; `jacobian_fun` /
      `jacobian_fun_project` apply the factor.
- [ ] A3 Tests `tests/test_noise_model.py`: deviance limits (`d=0`, `d=m`),
      sum equals the deviance formula, Gaussian broadcasting, `jacobian_factor`
      vs finite differences, segmented concatenation.

### B. File API and reductions
- [ ] B1 `File.set_noise`, `set_sigma` delegating, `File.noise` attribute with
      `noise_type` / `sigma_data` read-only properties for existing callers.
- [ ] B2 Project defaults `noise_type` / `sigma_data` / `noise_scale` in
      `_set_defaults` + project.yaml key handling.
- [ ] B3 Corrections clear the noise model with a warning.
- [ ] B4 `File.estimate_noise()`.
- [ ] B5 Wire view reductions into `fit_baseline`, `fit_spectrum`, `fit_sbs`,
      `fit_2d`; pass `noise=` to `fit_wrapper`; capture into the slot dict.
- [ ] B6 `Project.fit_2d`: per-file view noise → `SegmentedNoise`; mixed
      `unknown`/weighted raises.
- [ ] B7 Tests: validation and domain errors, corrections clear, reductions
      pinned (constant, per-point, Poisson baseline), `estimate_noise`
      recovers σ and `scale` on simulated data, joint mixed raises, joint
      Gaussian+Poisson runs.

### C. Identity, metrics, archive
- [ ] C1 `compute_fit_metrics(noise=)`: `chi2`, AIC/BIC forms, `sigma_eff`
      definition; SbS per-slice and joint callers.
- [ ] C2 `SavedFitSlot` + builders: `noise_scale`, `sigma` array, `sigma_data`
      NaN for arrays.
- [ ] C3 `encode_input_files` 4th element; `compute_optimization_hash`
      unchanged in signature; reader fallback for 3-element entries; check
      `variants()` / `diff()` parsing.
- [ ] C4 Writer/reader: `noise_scale` attr, optional `sigma` dataset.
- [ ] C5 Docs: `fit_archive_principles.md` amend the "σ is post-hoc" rule
      (dated); `fit_archive_schema.md` slot attrs/dataset and the additive
      note; `compare_models` docstring for `sigma_eff`.
- [ ] C6 Tests: hash changes with noise and not with `unknown`; roundtrip with
      per-point σ and Poisson scale; 3-element `input_files` reads; the
      `sigma_eff` identity pinned for all three kinds.

### D. MCMC and CI
- [ ] D1 `MC.resolve(..., noise_kind)`: Poisson forces `is_weighted`, rejects
      σ knobs; `is_weighted=True` under `unknown` raises; Gaussian
      `sigma_fit` from the weighted residual.
- [ ] D2 Confirm `conf_interval` needs no change (F-test on the objective's
      chisqr; deviance is asymptotically χ²).
- [ ] D3 Tests in `test_mc_settings.py`: forced/raised cases; seeded short
      chain on Poisson data: posterior widths vs `stderr` within tolerance.

### E. Verification against the bound
- [ ] E1 Deterministic: `JᵀJ` from the noise-weighted Jacobian at the truth
      equals `sensitivity.fisher_matrix(model, counts=scale·Σm)` to tight
      relative tolerance, 1D and 2D.
- [ ] E2 Seeded noisy fit: `stderr` within ~10% of the CRB.
- [ ] E3 Slow test reproducing the 2026-09-15 control: seed scatter /
      `stderr` in a band around 1 under `poisson`, vs the known ~1.6 under
      `unknown`.

### F. Simulator
- [ ] F1 Snapshot the noise model at simulate time: `scale` for
      `photon_counting` (and analog `poisson`, `1/noise_level`), σ for analog
      `gaussian`; `Simulator.noise_model` for `File.set_noise(**...)`;
      `sigma_data` kept.
- [ ] F2 `save_data` and the ML export write `noise_scale`.
- [ ] F3 Tests: variance/mean of counting output equals `1/scale`; stale case
      (`set_noise_level` after `simulate`) no longer changes the snapshot.

### G. Sensitivity module
- [ ] G1 Fix the `counts` convention docstring for 2D (total over the window
      = `counts_per_delay × n_time`); state the advanced tier in the module
      docstring.
- [ ] G2 `sphinx -W` build with `docs/api/sensitivity.rst`.

### H. Examples and docs
- [ ] H1 Notebook 12: keep §1–§3 under `unknown` (the off-scale MC demo needs
      `__lnsigma`); add a section that declares `poisson` via
      `estimate_noise`, refits, and compares `stderr` with and without the
      noise model and against `sensitivity_report`. `/check-example`.
- [ ] H2 Notebook 03: `set_noise('poisson', ...)` before the fits; verdict
      prose updated. `/check-example`.
- [ ] H3 CHANGELOG 0.17.0; TODO.md: retire the two noise items, note the
      simulator snapshot fix, add follow-ups (compound likelihood, σ through
      corrections, other counting notebooks 01/04/20/21).
- [ ] H4 `pyproject.toml` version 0.17.0 at commit time.

### I. Verify pass
- [ ] `pytest -q`, `pytest -m slow` for E3, pre-commit (ruff, mypy
      `--no-incremental`, pyright), `sphinx -W`, execute 03 and 12 then
      `scripts/normalize_notebooks.py`.
