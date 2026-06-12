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

## [ACTIVE] Rewrite 03_multi_cycle_dynamics

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
