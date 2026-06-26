# Active Plan — Examples Upgrade

Reorganize `examples/` around user-workflow tracks per
[docs/design/examples_upgrade.md](docs/design/examples_upgrade.md). Branch:
`upgrade-examples`.

## Status

- [x] Save/load branch merged; minimal save/load coverage already in
  `01_basic_fitting` and `10_model_comparison`.
- [x] Rename existing notebooks (03, 04, 05, data_generation/* directories).
- [x] Split notebook 10 (comparison-only) and lift persistence content into 11.
- [x] Create new notebook 11 (`%cd` + `%run` preamble under `%%capture`).
- [x] Extend notebook 01 with `get_fit_results` / `export_fit` / `save_fit`
  + `auto_export` opt-out story (casual-user exit path). *Refined during
  cleanup (6eed236): notebook 01 keeps only `get_fit_results` and the
  `auto_export: False` opt-out; `export_fit` / `save_fit` live in
  notebook 11 exclusively.*
- [x] Create new notebook 20 (bridge — multi-file separate fits) using 21's
  6-file dataset.
- [x] Add `fitting_workflows/README.md` documenting 0x / 1x / 2x legend.
- [x] Update navigation in `examples/README.md`, `docs/examples/index.rst`,
  `docs/quickstart.md`, `docs/installation.md`, `docs/ai/benchmark.md`.
- [x] Smoke-test notebook paths and run `pytest -q` (822 passed, 161 deselected).
- [x] `sphinx-build -W` clean.
- [ ] Commit and open PR (pending user review of the rendered notebooks).

## [DONE] Review 12_uncertainty_mcmc

Content was strong (the three-rung ladder + truth check). Library work
(the user's, pre-this-session) was the `ci_sigmas` rename and the new
`MC(sigma_ini/sigma_min/sigma_max)` knobs feeding `__lnsigma` — full suite
826 passed, slow `test_mc_sigma_settings_reach_lnsigma` passes. Two
execution blockers found and fixed in the notebook:

- [x] **Preamble**: already quotes the `%cd -q "../01_basic_fitting"` path
  (the IPython 9.x tokenizer fix from 11's review) — chains off **01**, not
  10. SbS spawn-pool fix also applies via the shared `%run`.
- [x] **Wrong YAML library (cells 3 + 14)**: cell 14 read the truth YAML
  with PyYAML `yaml.safe_load`, the only place in the repo that bypasses the
  project's ruamel stack. PyYAML mis-parses `8E-2` (no decimal point in the
  exponent) as the **string** `'8E-2'` → `TypeError` in the pull. The
  library parser (ruamel, `parsing.py`) parses it as `0.08` correctly, as do
  all the other `1E-2`/`-1E-3` example YAML values. Fixed by switching the
  notebook to `ruamel.yaml.YAML(typ='safe')`. (Decision: notebook-only;
  parser already accepts the format.)
- [x] **Hardening (follow-up, same session)**: closed the related latent gap —
  `parsing.py` value-validation silently skipped numeric checks when `value`
  wasn't int/float, so a non-numeric scalar (`m: ["abc", True]`) slipped past
  the bounds checks and failed cryptically downstream. Added a numeric guard
  raising `ModelValidationError` (hint points to the single-element `["expr"]`
  form for the string case). Verified safe: expressions are exclusively the
  1-element form, so no false positives. Test
  `test_non_numeric_value_raises` + `non_numeric_value` model in
  `tests/models/file_energy.yaml`.
- [x] **`__lnsigma` leaked into the truth table (cells 9 + 14)**: cell 14
  read `result[1].params` *after* the MCMC re-fit replaced
  `file.model_base.result`, so (a) the post-MCMC `__lnsigma` row had no truth
  → `KeyError`, and (b) the "stderr" column was MCMC posterior std, making
  the three-rungs comparison circular. Fixed by capturing `base_pars` (the
  leastsq value + covariance stderr) in cell 9 before MCMC overwrites it.
  Table now shows 6 model params, three genuinely distinct-source columns
  that agree, pulls all ≤ 1.14σ.
- [x] Executed end-to-end via nbconvert: exit 0, zero error outputs, ~45 s
  wall. No stray artifacts in git status.
- [x] **Root-cause library fix — `fit_wrapper` leaked `__lnsigma` into
  `result[1]`**: the `KeyError` above was a symptom. `_result_params(par_fin)`
  returns the *live* `par_fin.params`, and `fit_wrapper` did
  `.add("__lnsigma")` on it in place (`fitlib.py:734`), so the MCMC nuisance
  parameter leaked into the stored leastsq result (`result[1]`) after every
  MCMC fit. Fixed with `copy.deepcopy` before `.add` — emcee gets the copy,
  `result[1]` stays model-only. Regression test
  `test_lnsigma_does_not_leak_into_leastsq_result`. Blast-radius audit:
  numerically SAFE everywhere (`compare_models`/χ²/AIC/BIC use the frozen
  `result.nvarys`, not a param recount; `get_fit_results` baseline/spec/2d and
  save/export filter by `model.parameter_names`; save-path `lnsigma` is read
  from `emcee_fin` = `result[3]`). The leak was real but display-only:
  `display(result[1].params)` (trspecfit.py:2646/2888/4064) and
  `get_fit_results('sbs')` / SbS plots (only if SbS forwarded `mc_settings`)
  showed a spurious `__lnsigma` row. All fixed by the deepcopy.
- [x] **Readability pass (cell altitude)**: notebook was the most syntax-dense
  of the set (median 10 lines/code-cell vs ~6 elsewhere; 7/11 cells >8 lines).
  Split multi-step cells into one-idea cells, lifted the `one_sigma_halfwidth`
  helper to setup — median back to ~6, no logic change.
- [x] **Live result accessors (library)**: added `File.get_correlations`,
  `File.get_conf_intervals`, `File.get_mcmc` (returns new
  `ulmfit.MCMCResult` dataclass: `table`/`flatchain`/`acceptance_fraction`),
  reading `model.result` directly — so the notebook no longer needs raw
  `result[1..4]` indexing. Live-only by design; persisting `correl` /
  `acceptance_fraction` into slots is a TODO ("results-data ownership
  boundary"). 5 new tests in `test_mcp_library.py`.
- [x] **Notebook rewrite to one "kitchen-sink" fit**: a single
  `fit_baseline(stages=2, try_ci=1, mc_settings=mc)` now produces all three
  rungs (one unambiguous slot, no `base_pars`/`ci_table` capture). Three
  breakout sections read each rung via the `get_*` accessors — zero raw
  `result[i]` indexing remains. Corner plot regenerated under rung 3 via an
  explicit `corner.corner(get_mcmc(...).flatchain)` one-liner (getters stay
  pure; verbosity still gates the fit-cell plots). Re-executed: exit 0, table
  + pulls unchanged (≤1.14σ), 7 plots, unconverged 2D symptom intact.

## [DONE] Review 11_save_load_export

The notebook content was already strong; both findings were execution
blockers in the `%run` preamble, not content problems:

- [x] **Library fix — SbS spawn pool under `%run notebook.ipynb`** (was a
  TODO.md item, now removed): spawn workers re-ran `__main__` — the notebook
  JSON — and died (`NameError: name 'null'` → `BrokenProcessPool`).
  `utils/sbs.py::sanitized_spawn_main()` now hides a non-`.py`
  `__main__.__file__` for the pool's lifetime (workers only need the
  initializer-installed model). Regression test:
  `tests/roundtrip/test_focused.py::test_w2_sbs_with_notebook_main_file`.
- [x] **IPython 9.x `%%capture` tokenizer crash**: `%%capture` tokenizes the
  cell body as Python to detect a trailing semicolon; the bare path in
  `%cd -q ../10_model_comparison` reads as a malformed number (`10_model`).
  Fixed by quoting the path (+ explanatory comment in the cell).
- [x] Verified the preamble's artifact claim: `save_fit`/`export_fit`
  resolve relative paths against the process cwd, so after `%cd -q -` all
  artifacts land in 11's dir; 10's dir stays clean. (The stale `.fit.h5`
  found in 10 came from an interactive session, not the committed
  notebook.)
- [x] `.gitignore`: added 11's `winner_base/` / `winner_2d/` export trees
  (`*.fit.h5` was already covered) so a notebook run leaves git status
  clean.
- [x] Added the missing Next Steps links (12, 20) to the Tips cell.
- [x] Executed end-to-end: exit 0, zero warnings, ~31 s wall (matches the
  stated 30-40 s). Heartbeat reports all 6 slots. Full suite 825 passed.
- Note for 12's review: it uses the same `%run` preamble — both fixes apply;
  verify its preamble quotes the path too.

## [DONE] Review 10_model_comparison

Already strong from the comparison-only split; the criteria review found only
checklist deviations:

- [x] `try_ci=0` (+ pointer comment) on both baseline fits — removes the two
  CI tables, the only flagged output in the run.
- [x] `Path.cwd()` instead of `os.getcwd()`; dropped `import os`.
- [x] YAMLs: fixed parameters normalized to `[v, False]` (stale bounds
  stripped).
- [x] Re-executed end-to-end: exit 0, zero warnings, ~30 s wall. Verdicts
  unchanged (baseline: baseA chi2_red 1.00 vs baseB 5.40; 2D: m2dA 1.01 vs
  m2dB 4.75).
- [x] No side-effect files. Deleted two stale gitignored `.fit.h5` leftovers
  (`comparison.fit.h5`, `winner_base.fit.h5`) from an earlier notebook-11
  run; 10 itself does not recreate them.
- Criterion 2 amended below with the inline-synthetic variant.

Carry-over for 11's review: the notes below claim 11's artifacts land in
`11_save_load_export/`, but the stale `.fit.h5` files were found in 10's
directory — verify where `save_fit` resolves its path when 11 runs the
`%cd`/`%run` preamble (it may use `project.path`, i.e. 10's dir).

## [DONE] Upgrade 04_parameter_profiles

Core message: a parameter can vary along an auxiliary physical axis via a
profile model (`File.add_par_profile`); profile parameters are regular fit
parameters and can themselves be time-dependent (spectral diffusion,
auto-promotes to dim=2). Scenario: time-resolved XPS depth profiling — IMFP
amplitude profile + band bending position profile; photovoltage collapses the
band bending to flat-band at t=0, recovery tau=100 ps.

- [x] Generator rewritten Simulator-style (`data/generate_data.ipynb` +
  `models_{energy,profile,time}_truth.yaml`), save→reload axes pattern.
  Photon counting, counts_per_delay=20000, seed 42, peak SNR ~40.
- [x] Teaching structure: standard vs profile-aware baseline (effective vs
  physical values), SbS with exactly one free par (the gradient, via the
  shared `BandBending` profile entry; `IMFPfixed` pinned variant), 2D fit
  with dynamics on the profile par `GLP_01_x0_pLinear_01_m`.
- [x] YAMLs rewritten with 01-style comments; profile YAML keys are
  free-form (loader names the Profile by `par_name`) — descriptive names
  `IMFP` / `BandBending` + pinned variants.
- [x] `project.yaml`: `auto_export: False` + comment, eV/ps labels, XPS
  x_dir rev.
- [x] Executed generator + example end-to-end: exit 0, zero
  warnings/errors. Truth recovery: m -0.499/-0.5, SD 10.10/10,
  A_collapse 0.498/0.5, tau 99.0/100, x0 99.501/99.5, pExp A 100.8/100.
  Full suite 825 passed.

Two design lessons recorded:
- Exact scaling degeneracy `(A, tau, m) -> (cA, tau/c, cm)` of depth-averaged
  spectra: the IMFP tau must be fixed (from tables, standard XPS practice) —
  taught in the notebook and commented in `models_profile.yaml`.
- Library pitfall (TODO.md "Conv kernel support"): `create_t_kernel` sizes the
  conv kernel from the INITIAL parameter value; a too-small init silently
  truncates the kernel and biases the fitted width (bit 03 mildly too).
  Workaround: init conv widths generously (commented in `models_time.yaml`).

## [DONE] Rewrite 03_multi_cycle_dynamics

Old notebook taught wrong `frequency` semantics and used real data that can't
showcase the topic. Verified semantics (`mcp.py` `normalize_time`): `frequency`
= 1/T of the full repeating period; the N-1 non-global `model_info` entries
split each period equally (subcycle duration = `1/(frequency*(N-1))`), each
evaluated on a local clock that resets at its subcycle start; t < 0 is
baseline (`n_sub=0`).

Teaching message: pass a *list* of models to `add_time_dependence` — element 0
is global (IRF home), elements 1+ alternate as subcycles; plus `frequency`
semantics and cross-subcycle expression links (callback to 02).

Design: 01's system (GLP + LinBack, arb. units), synthetic via
`data/generate_data.ipynb` + truth YAMLs. Dynamics
`['IRF', 'MonoExpNeg', 'MonoExpPos']`, `frequency=0.25` (period 4, two 2-unit
subcycles). MonoExpPos amplitude linked (`-expFun_01_A`), taus independent
(truth: 0.4 vs 0.8); IRF SD 0.15 (visible, ~3 time steps).

- [x] Fix `t_label` ms -> s (03's time.csv was already in seconds).
- [x] `data/generate_data.ipynb` + `models_energy_truth.yaml` /
  `models_time_truth.yaml`; regenerate CSVs (old real-data CSVs are replaced;
  same campaign data lives on in 02).
- [x] Rewrite `models_energy.yaml` / `models_time.yaml` (01-style comments,
  drop unused ModelNone, drop doublet).
- [x] `project.yaml`: `auto_export: False` + comment, arb-unit labels.
- [x] Rewrite `example.ipynb` in 01's structure: roadmap, baseline excluding
  t=0, `try_ci=0`, SbS section says what to look for (sawtooth), corrected
  frequency narrative + local-clock note, results vs truth, pruned Tips
  (only real model names), Next Steps links, no `MC` import, `Path.cwd()`.
- [x] Library fix enabling conv-only global element: the validator's
  conv-cannot-be-last rule now applies only to multi-component models
  (`utils/parsing.py`), so `['IRF', ...]` works as documented in
  `Dynamics.set_frequency`. Guard test added in `test_model_parser.py`;
  full suite passes (825).
- [x] Execute generator + example end-to-end; verify all nine criteria.
  Stock-init fit recovers truth: SD 0.148/0.15, A -1.98/-2, tau 0.406/0.4
  and 0.804/0.8. Clean run, no warnings, suite green.

Pitfall discovered (recorded in the generator notebook): subcycle masks
switch discretely at period boundaries, so the simulation must run on the
*reloaded* CSV axes — generating on `np.arange` axes and fitting on their
`%.6e`-rounded reload flips the subcycle assignment of boundary-exact
samples and visibly biases the fit (tau2 0.8 -> 1.0 before the fix).

## Gold-standard example criteria (distilled from 01)

Bar for every `fitting_workflows` notebook before the PR. Review each
notebook against this list.

1. **Runs clean end-to-end.** `jupyter nbconvert --execute` from the project
   venv exits 0 with zero warnings/errors in cell outputs. Unavoidable
   warnings get an explaining markdown note; avoidable ones get fixed at the
   source (cf. `try_ci=0` in 01's baseline fit).
2. **Truth-anchored.** Synthetic data with committed `*_truth.yaml` files,
   regenerable via `data/generate_data.ipynb`. The closing section quotes the
   truth values so the reader can verify the fit recovers them.
   *Real-data variant (02):* measured data is deliberately kept when known
   physics anchors the fit instead — state that the data is real, document
   its provenance, and compare results to literature values in the closing
   section (02: Au 4f7/2 at 84.0 eV, 3.67 eV splitting, 3:4 ratio).
   *Inline-synthetic variant (10):* generation may stay inline when the
   tunable ground truth is itself the teaching device (edit a constant,
   re-run, watch the metrics react) — keep the truth constants in one
   labeled cell and print them. Simulator or plain numpy both work; pick
   whichever makes the truth most legible in that notebook.
3. **Self-contained directory.** `data/`, model YAMLs, `project.yaml`; no
   dependence on having run another notebook first (exception: 11's
   documented `%run` preamble).
4. **No surprise side-effects.** `auto_export: False` with an explanatory
   comment in `project.yaml`, unless persistence/export is the topic.
5. **Why-driven narrative.** Opening cell gives a numbered roadmap; each
   section explains why the step exists (e.g. 01's "Why global?"), not just
   what the next call does.
6. **Deliberate kwargs.** Non-obvious arguments carry a short comment; don't
   lean on defaults that produce unexplained output.
7. **Scope discipline.** One topic per notebook. Adjacent topics are
   delegated via valid relative links; close with Tips + Next Steps. No
   content the reader can't act on within this notebook.
8. **Commented YAMLs.** Model files say what each block is for and point at
   the `functions/` source for available functions/parameters.
9. **Stripped outputs.** `nbstripout` stays; rendered outputs arrive via the
   Read the Docs follow-up below, not via committed outputs.

## Follow-up — Render executed examples on Read the Docs

Goal: let users browse fully rendered example notebooks (plots, fit tables)
without installing anything. Keep git sources stripped (`nbstripout` stays);
outputs are generated at docs build time, so they are always in sync with the
code. Executed notebook 01 is ~4.2 MB vs 14 KB stripped — do NOT commit
outputs. Start AFTER the notebook content work above is done.

- [ ] Add example notebooks to the docs as rendered pages. They live outside
  `docs/`, so use `nbsphinx-link`: one `.nblink` file per notebook in
  `docs/examples/` pointing at `../../examples/.../example.ipynb`, listed in
  a toctree in `docs/examples/index.rst` (currently only plain reST links to
  the `.ipynb` sources — nothing is rendered today).
- [ ] Flip `nbsphinx_execute` from `"never"` to `"auto"` in `docs/conf.py` so
  stripped notebooks are executed during the Sphinx build.
- [ ] Time all nine notebooks; opt slow ones (likely `12_uncertainty_mcmc`)
  out of build-time execution via notebook metadata
  (`"nbsphinx": {"execute": "never"}`) or trim their runtime.
- [ ] Verify `.readthedocs.yaml` installs the package with all fitting deps
  (required for execution) and that total build time fits RTD limits.
- [ ] Keep `sphinx-build -W` clean; watch the notebook 11 `%cd`/`%run`
  preamble under build-time execution.
- [ ] Add "browse the rendered examples" links to `README.md` and
  `examples/README.md` once the RTD pages exist.

## Target Layout

```text
examples/
  fitting_workflows/
    01_basic_fitting/                 # fit + get_fit_results (export → 11)
    02_dependent_parameters/          # unchanged
    03_multi_cycle_dynamics/          # was 03_multi_cycle
    04_parameter_profiles/            # was 04_par_profiles
    10_model_comparison/              # trimmed to comparison-only
    11_save_load_export/              # NEW (%cd + %run + %%capture preamble)
    20_fit_each_separately/           # NEW (bridge: multi-file separate fits)
    21_project_level_shared_fit/      # was 05_project_level_fitting
  synthetic_data/                     # was data_generation
    01_simulator/                     # was simulator
    02_ml_training_data/              # was ml_training
```

## Notes / decisions inherited from (or refined past) the design doc

- 0x = single-file fitting skills; 1x = post-fit work; 2x = multi-file.
- Notebook 11 preamble: `%%capture` wrapping `%cd -q ../10_model_comparison`,
  `%run example.ipynb`, `%cd -q -`. `%%capture` (the design doc's stated
  fallback) is the default mechanism in the implementation — notebook 10
  keeps `show_output: 1` so it stays interactive when opened directly. The
  `%cd` dance ensures notebook 10's `project.yaml` and model YAMLs resolve;
  artifacts that notebook 11 writes (`comparison.fit.h5`, `winner_*`) land
  in `11_save_load_export/`. A short heartbeat cell after the preamble
  prints file/model/slot counts so the reader sees the fits landed.
- Notebook 10's `project.yaml` keeps `auto_export: false` so the six
  baseline/SbS/2D fits don't each spawn a CSV/PNG dump and drown out the
  comparison narrative.
- Notebook 20 reuses 21's data (6 files) via relative path; copies only
  the small YAMLs. `compare_models(file=...)` is single-target — the
  whole-batch survey calls `compare_models()` with no `file=` filter.
- Benchmark harness discovers folders by `NN_` prefix. Updated:
  `--example 0` (batch mode) iterates examples 1–4; project-level
  (`21_project_level_shared_fit`) and the new 1x notebooks sit outside the
  harness. Table in `docs/ai/benchmark.md` updated accordingly.
