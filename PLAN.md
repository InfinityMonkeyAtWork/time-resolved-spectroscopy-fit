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
  + `auto_export` opt-out story (casual-user exit path).
- [x] Create new notebook 20 (bridge — multi-file separate fits) using 21's
  6-file dataset.
- [x] Add `fitting_workflows/README.md` documenting 0x / 1x / 2x legend.
- [x] Update navigation in `examples/README.md`, `docs/examples/index.rst`,
  `docs/quickstart.md`, `docs/installation.md`, `docs/ai/benchmark.md`.
- [x] Smoke-test notebook paths and run `pytest -q` (822 passed, 161 deselected).
- [x] `sphinx-build -W` clean.
- [ ] Commit and open PR (pending user review of the rendered notebooks).

## Target Layout

```text
examples/
  fitting_workflows/
    01_basic_fitting/                 # extended with save/export sections
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
