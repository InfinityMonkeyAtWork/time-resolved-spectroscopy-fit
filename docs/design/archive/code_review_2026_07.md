---
orphan: true
---

# Full-Repo Code Review — July 2026

> Archived on 2026-07-10 after every finding was resolved: fixed on the
> `fix-conv-kernels` branch, declined with rationale recorded inline, or
> moved to a `TODO.md` item. The full-scope review protocol developed for
> this run now lives in `docs/ai/code-review.md`.

Reviewed at commit `a1df0cc` using the checklist in `docs/ai/code-review.md`
(full scope). Method: repo-wide grep checks run once globally; read-the-code
checks run per architecture chunk (A authoring, B compiled hot path,
C fitting/bridge, D persistence, E simulation/plotting/config, F tests light
pass) by parallel subagents. Findings verified by spot-checks where marked.

Scope: deep review of `src/`, light pass on `tests/` (edge-case coverage,
patterns, parity verification). Notebooks, YAML assets, docs excluded.

**Triage key** (fill in during triage): `[F]` fix now, `[L]` later, `[W]` won't fix.

## Summary table

| # | Check | Status | Issues |
|---|-------|--------|--------|
| 1 | Bugs & correctness | **FAIL** | 4 FAIL, 12 WARN, 6 INFO |
| 2 | Performance | WARN | 12 WARN, 4 INFO |
| 3 | Broad exceptions | WARN | 1 WARN, 1 INFO |
| 4 | God classes / long methods | INFO | size mostly justified |
| 5 | Dead code | PASS | 1 INFO |
| 6 | Typing / modern Python syntax | PASS | no `Optional[`/`Union[` |
| 7 | Docstring coverage | WARN | 1 WARN, 2 INFO |
| 8 | Abstractions & duplication | WARN | 9 WARN, 5 INFO |
| 9 | Numpy anti-patterns | INFO | 1 INFO |
| 10 | Fragile array comparisons | PASS | only zero-guards found |
| 11 | Ignored warnings | PASS | none in src; 1 scoped pytest mark |
| 12 | Plotting mixed with logic | WARN | 8 WARN, 1 INFO |
| 13 | Separation of concerns | WARN | 8 WARN, 4 INFO |
| 14 | Missing `__repr__` | INFO | 8 dataclasses/classes |
| 15 | Global mutable state | INFO | 1 documented worker pattern |
| 16 | Security | PASS | safe YAML loaders, no eval/pickle/shell |
| 17 | Edge-case tests | INFO | 3 concrete gaps |
| 18 | GIR / MCP parity | WARN | core parity strong; 6 verified gaps |
| 19 | Two-layer design compliance | WARN | 4 WARN, 3 INFO |
| 20 | `can_lower_*` hygiene | INFO | 2 implicit handlings, otherwise complete |

## FAIL findings (check 1 unless noted)

All four spot-verified against source.

- [x] `src/trspecfit/simulator.py:1304` — `plot_comparison` builds the plot
  title via `self.get_snr(...)` *before* the `data_clean is None`
  auto-simulate guard at :1310, so calling it on a fresh Simulator raises
  instead of simulating first. Fix: move title construction below the
  auto-simulate block.
- [x] `src/trspecfit/utils/arrays.py:241` — `sign_change(ignore_zeros=True)`
  infinite-loops on all-zero input: `asign` stays all-zero so the
  `while sz.any()` loop never terminates. Fix: bail out (return zeros) when
  `asign.any()` is False.
- [x] `src/trspecfit/utils/arrays.py:383` — `my_conv` computes
  `x_arr[1] - x_arr[0]` with no `len(x) >= 2` guard; single-element input
  raises `IndexError`. Called from the hot-path 2D evaluator. Fix: guard or
  document the precondition at the call boundary.
- [x] `src/trspecfit/utils/plot.py:703` — `plot_1d(y_norm=1)` divides by
  `np.max(y_data - np.min(y_data))`, which is 0 for a constant trace
  (produces inf/NaN plot silently). Fix: guard the zero-range case.

## WARN findings

### Check 1 — Bugs and correctness

- [x] `src/trspecfit/mcp.py:992` — `create_value_2d(t_ind=[start,stop])`
  passes loop index `ti` (0..n-1) to `create_value_1d` instead of
  `start+ti`; partial-range evaluation is wrong. Latent: never called with
  `t_ind` in-repo.
- [x] `src/trspecfit/mcp.py:2248` — `Par.value` returns `-1.0` and prints
  when `t_vary` is set but `t_model` is None, instead of raising.
- [x] `src/trspecfit/trspecfit.py:1702`, `:2313`, `:2383` — `File.describe`,
  `define_baseline`, and `set_fit_limits` mutate `self.energy`/`self.time`
  in-place when axes are missing (side effects in inspection/setup paths).
- [x] `src/trspecfit/eval_2d.py:362` and `src/trspecfit/graph_ir.py:2916` —
  `time[1] - time[0]` assumes `n_time >= 2`; a single-point time axis raises
  `IndexError`. Cross-reference: no test covers this (check 17).
- [x] `src/trspecfit/eval_2d.py:343` — dynamics substeps read only
  `traces[row, 0]`; a time-varying expression row would silently use its
  t=0 value.
  (Investigated 2026-07-10: latent — rows feeding dynamics substeps and
  conv kernels are time-constant by construction. Dynamics-model
  expressions are evaluated in the dynamics model's own lmfit namespace,
  so cross-model references (the only route to a time-varying row) fail
  at `add_time_dependence` before any graph build or fit; probed all
  three routes (direct t_vary ref, static ref, via top-level expression
  param). Hardened: `add_dynamics` now re-raises the deep lmfit
  `NameError` as a clear `ValueError`; the invariant is documented at
  both t=0 read sites; and
  `test_dynamics_expression_cross_model_reference_raises` pins the
  rejection so the reads get revisited if cross-model refs are ever
  allowed. Follow-up considered and declined 2026-07-10: the rejection is
  not atomic (target par keeps t_vary/t_model if the caller catches the
  ValueError), but the leaked state re-raises on every subsequent
  evaluation (verified: create_value_1d/2d, par.value) — loud, not
  silent. The authoring API is non-atomic throughout (add_profile,
  load_model), so a one-off rollback would imply a transactional
  guarantee the layer doesn't have; recovery is reloading the model.)
- [x] `src/trspecfit/functions/energy.py:144` — `LinBack` raises `ValueError`
  (xStart >= xStop) inside the hot-path numeric body; validation belongs in
  the authoring layer (also flagged under check 19).
  (Resolved under check 19, 2026-07-10: guard kept by design — see the
  check-19 note.)
- [x] `src/trspecfit/fitlib.py:1097` — `results_to_fit_2d` DataFrame path
  passes `iloc[i].values` (all columns); extra non-parameter columns break
  the fit unless the caller pre-filters (the public API does).
  (2026-07-10: added optional `parameter_names` kwarg that selects and
  orders DataFrame columns (KeyError on missing); the SbS caller now
  passes it instead of pre-filtering. Tested incl. scrambled column
  order.)
- [x] `src/trspecfit/utils/fit_io.py:796` and `:171` — `SavedFitSlot` stores
  `params`/`conf_ci`/`selection`/`metrics` by reference; `frozen=True`
  blocks reassignment but not in-place DataFrame/dict mutation, weakening
  the snapshot contract.
  (Declined 2026-07-10: a user mutating `file.data`/results in place breaks
  far more than slots (fingerprints, fit limits, cached evaluations), so a
  slot-level defensive copy papers over one symptom. Deferred to a TODO.md
  item on a systemic stance — read-only arrays, copy-on-set, ownership
  contract, or save-time re-hash.)
- [x] `src/trspecfit/utils/sweep.py:334` — `_generate_random` silently skips
  unknown spec types, yielding incomplete configs.
  (2026-07-10: `else: raise ValueError` naming parameter and type,
  mirroring the grid path's `_sample_distribution` which already raised.)
- [x] `src/trspecfit/utils/sweep.py:400` and `:309` — `get_n_configs()` on
  empty `parameter_specs` raises `ValueError` from `max()`; empty sweep
  yields one `{}` config.
  (2026-07-10: `_require_parameters()` guard in both `get_n_configs` and
  `__iter__` with a clear "No parameters added" error.)
- [x] `src/trspecfit/utils/arrays.py:388` — `my_conv` normalizes by
  `np.sum(kernel_arr)` with no zero-sum guard.
  (2026-07-10: raises `ValueError` on zero or non-finite kernel sum,
  naming the likely cause (kernel width collapsed below the axis step);
  one scalar comparison on an already-computed sum, no hot-path cost.)
- [x] `src/trspecfit/simulator.py:982` — `set_noise_type` docstring lists
  `'uniform'` but only poisson/gaussian/none are handled.
  (2026-07-10: docstring fixed to poisson/gaussian/none; setter AND
  constructor now validate via a shared `_validated_noise_type` helper, so
  a typo fails at construction/set time instead of at the first
  `add_noise`. The broader noise-language cleanup stays a TODO.md item.)

### Check 2 — Performance (hot path and bulk loops)

- [x] `src/trspecfit/eval_2d.py:111`, `:108` — RPN evaluator copies the full
  trace row on every `PARAM_REF` and allocates `np.full(n_time, ...)` per
  constant instruction, per residual evaluation.
  (Fixed in `b139601`: parameter rows are views, constants remain scalar,
  and only constant-only programs allocate the required output row.)
- [x] `src/trspecfit/eval_2d.py:174-176`, `:222` — per-eval
  `broadcast_to(...).copy()` per profile-sample group; `np.repeat(traces,
  n_aux, axis=1)` builds a large matrix for profile expressions each eval.
  (Fixed in `b139601`: both broadcasts now write directly into their
  preallocated destination buffers.)
- [x] `src/trspecfit/eval_2d.py:255-279` — profiled ops loop `n_aux` times
  calling energy functions instead of a vectorized aux broadcast.
  (Evaluated both ways 2026-07-10 after fixing the benchmark harness to
  actually attach example 04's profiles: aux vectorization is ~4.5x faster
  only when profiled params enter the function linearly (amplitude-only),
  and ~60% *slower* with a profiled position (example 04, n_aux=50) because
  the full `(n_time, n_aux, n_energy)` temporaries materialize inside the
  energy functions. Kept the loop; param-source resolution hoisted out of
  it. See `_evaluate_profiled_op_2d` docstring.)
- [x] `src/trspecfit/fitlib.py:233`, `:238` — `residual_fun` resolves
  `fit_fun_str` via `getattr(spectra, ...)` and calls `par_extract` on every
  residual evaluation; both hoistable to setup.
  (Evaluated 2026-07-10, not changed: measured 0.07 + 2.5 us per call vs
  1-8.5 ms per model eval, i.e. <0.3%; hoisting would change the `const`
  contract across all five fit entry points for no observable gain.)
- [x] `src/trspecfit/spectra.py:292` — `fit_project_mcp` rebuilds the
  `par_lookup` dict and runs full MCP `create_value_2d` per file on every
  project-level residual (also check 19).
  (Declined 2026-07-10: MCP stays the readable reference layer; no
  micro-optimization there. The real fix is a lowered GIR project
  evaluator — deliberately a later TODO.)
- [x] `src/trspecfit/fitlib.py:782` — MCMC walker/corner figures always built
  when `use_emcee==1`, even with `show_output=0, save_output=0`.
  (Figure construction now skipped entirely when neither shown nor saved.)
- [x] `src/trspecfit/utils/arrays.py:383`, `:388` — `my_conv` copies
  x/y/kernel and recomputes `np.sum(kernel)` on every hot-path call;
  padded x array is built then discarded.
  (`b139601` removes the discarded padded-x construction and normalizes the
  short kernel instead of dividing the padded signal. The live kernel changes
  each evaluation, so its sum is still computed once per call; evaluator
  float64 inputs pass through `np.asarray` without copies. `pad_x_y` is now
  unused repo-wide — decision 2026-07-10: keeping it.)
- [x] `src/trspecfit/simulator.py:1984`, `:748`, `:798` — per-config HDF5
  open/close in sweep append; `simulate_n` appends per iteration instead of
  pre-allocating; Poisson scale factor recomputed per `add_noise` call.
  (2026-07-10: fixed the HDF5 part — `simulate_parameter_sweep` opens the
  file once and passes the handle to `_initialize_sweep_hdf5` /
  `_append_config_to_hdf5`, with a per-config `flush()` preserving
  interrupted-sweep durability. The `simulate_n` pre-allocation and Poisson
  scale-factor items were declined: they only speed up the analog path
  and/or couple `simulate_n` to photon-sampling internals.)

### Check 3 — Broad exceptions

- [x] `src/trspecfit/trspecfit.py:825` — config-loading `except Exception`
  only prints when `show_output >= 1`; with `show_output=0` errors are
  swallowed silently. (`mcp.py:1545`, `:2368` re-raise as `ValueError` —
  fine. `trspecfit.py:3290` `except BaseException` cancels futures and
  re-raises — fine. `utils/lmfit.py:125` warns — acceptable.)
  (Fixed in `f0e5e66`: missing files retain the designed fallback; all other
  load/validation failures are re-raised as `ValueError`.)

### Check 7 — Docstring coverage

- [x] `src/trspecfit/functions/profile.py:47`, `:70`, `:92` — `pExpDecay`,
  `pLinear`, `pGauss` lack `Returns` sections (user-facing `functions/`).

### Check 8 — Abstractions and duplication

- [x] `src/trspecfit/eval_1d.py:27` vs `graph_ir.py:3048` — scalar RPN
  evaluator implemented twice (`eval_expr_program_1d` vs
  `_eval_expr_scalar`).
  (Deduped 2026-07-10: `eval_1d` now imports `_eval_expr_scalar` from
  `graph_ir`, matching the other runtime helpers it already imports;
  `eval_expr_program_1d` deleted.)
- [x] `src/trspecfit/graph_ir.py:2433-2657` vs `:3407-3608`, and
  `:2707-2835` vs `:3659-3793` — PROFILE_SAMPLE/EXPR compilation and
  PROFILE_AVERAGE op-wiring largely copy-pasted between `schedule_2d` and
  `schedule_1d`.
  (2026-07-10: sample/expr compilation extracted into
  `_compile_profile_groups` and op scheduling into
  `_schedule_component_ops` (NamedTuple returns); each shared body is the
  superset — 2D-only PPT/CONVOLUTION walk-backs never fire on 1D graphs,
  1D-only "not scalar-lowerable"/"Non-scalar parameter source" errors
  never fire on 2D, where every node has a row. Net −223 lines.)
- [x] `src/trspecfit/graph_ir.py:2866` vs `eval_2d.py:31` — `_DYN_DISPATCH`
  mirrors `DYNAMICS_DISPATCH` with separate function references.
  (Resolved 2026-07-10 together with the item below — the mirror only
  existed to serve the duplicated loop.)
- [x] `src/trspecfit/graph_ir.py:2882-2921` vs `eval_2d.py:329-365` —
  resolution loop duplicated at compile-time init vs hot-path eval.
  (Extracted 2026-07-10 into `eval_2d.resolve_param_traces`; `evaluate_2d`
  and `schedule_2d` init both call it. Array-argument signature keeps it
  usable at compile time before the plan exists.)
- [x] `src/trspecfit/simulator.py:769`, `:1650`, `:1680` — near-identical
  1D/2D analog-noise generators; HDF5 metadata serialization duplicated
  across `_save_hdf5` / `_initialize_sweep_hdf5`; params-to-JSON loop
  repeated in three HDF5 writers.
  (2026-07-10: `_generate_noise_analog_1d/2d` were byte-identical (all
  numpy ops shape-agnostic) — merged into `_generate_noise_analog`; axes
  write, detection/seed attrs, and full params-to-JSON extracted into
  `_write_axes_hdf5` / `_write_detection_metadata` /
  `_model_parameters_json`. The value-only params loop in
  `_append_config_to_hdf5` is a different output and stays. The
  `_sample_photons_1d/2d` pair genuinely differs — total vs per-row
  normalization — and stays split.)

### Check 12 — Plotting mixed with logic

- [x] `src/trspecfit/fitlib.py:749`, `:764`, `:781` — MCMC progress print
  and `progress=True` run unconditionally; with `show_output=0,
  save_output=0` MCMC plots still reach `plt.show()` via
  `_finalize_plot(0, ...)`.
  (Fixed in `f0e5e66`; the separate cost of constructing unsaved MCMC figures
  remains open under check 2.)
- [x] `src/trspecfit/trspecfit.py:4061` — `File.fit_2d` calls `time_display`
  and `display(params)` whenever `stages>=1`, ignoring `show_output`
  (fixed in `f0e5e66` together with the same pattern in
  `fit_slice_by_slice`).
- [x] `src/trspecfit/trspecfit.py:2335`, `:2425` — `define_baseline` /
  `set_fit_limits` plot on `show_plot=True` default without consulting
  `Project.show_output`. (Fixed in `f0e5e66`.)
- [x] `src/trspecfit/simulator.py:1304`, `:1310` — `plot_comparison` mixes
  SNR computation and auto-simulate side effects into the plot path
  (see also the FAIL above).
  (2026-07-10: the auto-simulate is deliberate (documented, regression
  test) and the SNR title is the feature. The actionable part was the
  hand-rolled 3-panel loop duplicating `plot_2d_grid` — now delegated
  (`columns=3`; grid gained `columns`, x/y-lim, and ticksize support so
  nothing was lost). Panel size now follows the grid convention.)

### Check 13 — Separation of concerns

- [x] `src/trspecfit/utils/fit_io.py:35`, `:1139` — raw `h5py` throughout
  instead of the `utils/hdf5.py` helpers the architecture doc mandates
  (`require_group`/`require_dataset`/`json_loads_attr`). Either migrate or
  amend the contract in `repo_architecture.md`.
  (Both, 2026-07-10: fit_io's local `_as_group`/`_as_dataset` duplicated
  the shared helpers — deleted; all 25 read sites now use
  `require_group`/`require_dataset` with real archive paths in errors.
  Contract amended: helpers cover *reads* (type narrowing + JSON attrs);
  writes have no wrapper and use `h5py` directly.)
- [x] `src/trspecfit/simulator.py:1620` and `utils/sweep.py:482` — same raw
  `h5py.File` usage in `_save_hdf5`/`_initialize_sweep_hdf5` and
  `SweepDataset` reads.
  (Stale finding, 2026-07-10: `SweepDataset` reads already go through
  `require_group`/`require_dataset`/`json_loads_attr`, and the simulator
  sites are write-only — covered by the amended contract above.)
- [x] `src/trspecfit/fitlib.py:446` — `fit_wrapper` combines optimization,
  CI, MCMC, plotting, CSV/TXT I/O, and notebook display in one ~207-line
  function.
  (Declined 2026-07-10: decomposition not wanted; it stays one function
  unless a concrete need arises.)
- [x] `src/trspecfit/simulator.py:1347`, `utils/plot.py:246`, `:516` —
  hardcoded `figsize`s and reference-line colors outside `PlotConfig`
  (PlotConfig is documented as the single source of truth for styling).
  (2026-07-10: added `refline_color`/`refline_style` (unified default
  grey `#808080` dotted — 2D reflines were black, 1D vlines dashed) and
  `panel_size` (default 4×3 per panel) to PlotConfig; `plot_1d`/`plot_2d`
  accept them as per-call kwargs, `plot_2d_grid` reads them from config.
  The simulator figsize disappeared via the plot_comparison delegation
  below.)

### Check 18 — GIR/MCP parity (verified against tests by chunk F)

Core parity is strong: 2D baseline/IRF/subcycle/profile residual+compare,
1D compare/residual, and end-to-end `fit_model_compare` through `fit_2d`,
`fit_baseline`, `fit_spectrum`, `fit_slice_by_slice` (serial + 2 workers)
all exist in `tests/test_gir_integration.py`; roundtrip matrix backend `C`
adds broad coverage. Verified gaps at the `fit_model_compare` level:

- [x] Energy shapes `GaussAsym`, `Lorentz`, `Voigt`, `GLS`, `DS`, `LinBack`:
  evaluator-level parity only (`tests/test_evaluate_2d.py:146-171`), no
  pipeline/compare coverage. (`Gauss` and `Shirley` partially covered via
  roundtrip families and profile tests.)
  (`test_compare_mode_energy_shapes`, parametrized over all 7 incl. Shirley.)
- [x] Dynamics `sinFun`, `linFun`, `sinDivX`, `erfFun`, `sqrtFun`,
  `stepFun`: pure-math tests only (`tests/test_functions_time.py`); only
  `expFun` has parity coverage.
  (`test_residual_same_gir_vs_mcp_dynamics`, parametrized over all 6.)
- [x] Profile `pGauss`: pure-math only; no profile YAML fixture.
  (`profile_pGauss` fixture; compare-mode profile 1D/2D tests parametrized.)
- [x] Multi-substep dynamics without subcycles (`frequency=-1`/omitted):
  no parity test (all multi-substep tests use `frequency=10`).
  (`test_residual_same_gir_vs_mcp_multi_substep_single_cycle` via
  `BiExpSharedT0`.)
- [x] Chained CONVOLUTION nodes: structural gate test only
  (`tests/test_graph_ir.py:1434`); no compare-backend coverage.
  (`MonoExpPosDoubleIRF` fixture; parity test pins `n_conv_steps == 2`.)
- [x] TIME_1D-only domain models: graph tests only
  (`tests/test_graph_ir.py:1101`); MCP fallback untested through the fit
  pipeline. (`test_time_1d_dynamics_model_mcp_fallback` at the
  residual_fun/fit_model_gir-fallback level; standalone trace *fitting* is
  itself still a deferred TODO, so no deeper pipeline entry exists to test.)
- [x] `test_compare_mode_irf` covers only `MonoExpPosIRF`; the residual
  variant is parametrized over all 7 kernels but compare mode is not.
  (Both now share the `_IRF_KERNEL_MODELS` list.)
- Extra (2026-07-10): `test_constant_profiled_op_folds_into_cache` — a
  fully-fixed profiled op (compile-time constant-op branch in `schedule_2d`)
  had no fixture; only the type checkers caught a stale call there during
  the check-2 batch.

### Check 19 — Two-layer design compliance

- [x] `src/trspecfit/spectra.py:244` — `fit_project_mcp` is fully
  interpreter/MCP with per-call dict distribution; belongs on a
  setup/compile path or a lowered multi-file evaluator.
  (Declined as an MCP change 2026-07-10 — see the check-2 note; the
  lowered multi-file evaluator is the future fix.)
- [x] `src/trspecfit/fitlib.py:233` — string-based `getattr` dispatch per
  residual call on the bridge (same fix as the check 2 item).
  (Closed 2026-07-10 by the check-2 measurement above: <0.3% of a residual
  call; hoisting would ripple through the `const` contract of all five fit
  entry points for no observable gain.)
- [x] `src/trspecfit/functions/energy.py:144-152` — `LinBack` validation and
  error formatting in a numeric body called from the residual loop.
  (Closed 2026-07-10, guard kept by design: xStart/xStop can be fit
  parameters or expressions, so the optimizer can violate the ordering
  mid-fit and setup-time validation cannot replace the runtime check. The
  happy path costs one `np.any` comparison; formatting only runs on the
  failure branch. No clamping — the error now nudges users to set min/max
  bounds on xStart/xStop instead.)
- Hot evaluators themselves are clean: no `isinstance`/model-structure
  branching in `eval_1d.py`/`eval_2d.py` inner loops (PASS).

### Chunk F — test-pattern compliance (CLAUDE.md)

- [x] `tests/test_gir_integration.py:1087` — uses internal
  `File._build_1d_dispatch_args`; could mask validation bugs on the public
  dispatch path.
  (Closed 2026-07-10 as within the CLAUDE.md invariant-check exception:
  the test pins the private dispatch-args contract used by three
  trspecfit.py call sites, while `test_gir_baseline_writes_back` covers
  the same machinery through the public path. Docstring now says so.)
- [x] `tests/test_plotting.py:671` (also `:713`, `:726`, `:740`, `:753`) —
  `component.plot()` without `save_img=-2`/`show_plot=False`; mitigated by
  Agg backend + `plt.close("all")` but not per convention.
  (Closed 2026-07-10 as by-design: these tests assert on the live axes, so
  `save_img=-2` would close the figure before the assertions; Agg +
  `plt.close("all")` is the suppression. CLAUDE.md's Plots rule now carves
  out figure-inspection tests. The `# type guard` INFO item was swept
  suite-wide in the same pass.)

## INFO findings (no action required; note during triage)

- Check 1: `spectra.py:132` GIR fast path leaves `model.lmfit_pars` stale
  mid-fit; `fitlib.py:840` docstring promises `emcee_fin=[]` but returns
  `None`; `fit_io.py:807` `observed`/`fit` rely on callers copying;
  `fit_io.py:1601` reader trusts on-disk sha256; `fit_io.py:345`/`:336`
  stale docstrings (schema version "1" vs "2"; nonexistent builder name);
  `mcp.py:1728` `Component.value` reassigns `self.time` for conv kernels.
- Check 4: `File` has 36 public methods; `fit_slice_by_slice` 234 lines;
  `fit_wrapper` ~207 lines; `schedule_2d` ~1020 / `schedule_1d` ~616 lines
  (justified compiler monoliths); `fit_io.py` ~2200-line module. All judged
  justified by the respective reviewers.
- Check 5: `trspecfit.py:4132` single commented-out `dpi_plot` line ("NOT
  AVAILABLE YET").
- Check 7: 12 public items in `trspecfit.py` and 2 in `mcp.py` lack
  Parameters/Returns sections (list in chunk A transcript); worst offenders
  `load_fits:559`, `save_fit:2907`, `export_fit:2935`, `compare_models:4340`.
- Check 8: `_append_baseline_slot`/`_append_spectrum_slot` share ~30-line
  extraction block (`trspecfit.py:3419`); GIR dispatch duplicated between
  1D helper and `File.fit_2d:3993`; `par_create`/`par_construct` share
  branching (`utils/lmfit.py:113`); `_as_group`/`_as_dataset` duplicate
  `hdf5.py` helpers (`fit_io.py:145`); `_attr_str`/`_to_str_value`
  near-identical (`fit_io.py:1434`).
- Check 9: `utils/plot.py:670` `np.arange(0, n, 1)` → `np.arange(n)`.
- Check 13: `graph_ir.py:3171-3285` runtime helpers called from `eval_1d`
  (layer blur); lazy `eval_2d` import inside `schedule_2d` (`:2876`);
  CSV export invokes `fitlib` plotting (`fit_io.py:1953`); `fitlib.py:34`
  imports `IPython.display`.
- Check 14: missing `__repr__` on `GraphIR`, `ScheduledPlan1D/2D`,
  `GraphNode`, `ExprProgram`, `SavedFitSlot`, `SavedFile`, `SavedProject`,
  `PlotConfig`, `par_dummy`.
- Check 15: `utils/sbs.py:155` worker-process globals — documented,
  standard `ProcessPoolExecutor` initializer pattern; acceptable.
- Check 17 (test gaps): no single-point time axis test anywhere (directly
  covers the `time[1]-time[0]` WARN above) — covered since 2026-07-09
  (`test_graph_ir.py`, `test_mcp_eval.py`); no single-element energy/time
  arrays through the fit pipeline; no NaN/Inf-in-data tests at the public
  `File`/`Project` fit level — both covered since 2026-07-09
  (`test_fit_validation.py`; non-finite fit-window data now raises a
  clear error from `fit_wrapper`).
- Check 20: `NodeKind.SUM` has no explicit gate (always structural);
  `DomainKind.TIME_1D` rejected by domain check rather than node-kind set.
  Both implicit but correct; all other kinds handled explicitly.
- Chunk F: several `assert model is not None` without `# type guard`
  comment (`test_gir_integration.py:57` etc.); many `test_plotting.py`
  calls use `save_img=0` instead of `-2`.

## Suggested triage order

1. The four FAILs (small, local fixes; `sign_change` and `my_conv` are
   pure-math and easy to test).
2. Check 1 WARNs that affect correctness of public behavior:
   `mcp.py:992` t_ind indexing, axis-mutating describe/setup methods,
   single-point time axis (fix together with the check 17 test gap).
3. Silent-mode violations (checks 3 and 12): config-load swallow,
   MCMC prints/plots, `fit_2d` display calls — one themed pass.
4. Hot-path performance batch (check 2, eval_2d + residual_fun + my_conv).
5. Parity-coverage batch (check 18): extend the roundtrip/parametrized
   fixtures; cheap wins are parametrizing `test_compare_mode_irf` and adding
   a pGauss profile fixture.
6. Structural/duplication items (checks 8, 13) as deliberate refactors,
   one per PR.
