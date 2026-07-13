# Stability and Deprecation Policy

`trspecfit` is pre-1.0: the public API is still being shaped by real usage.
This page states what can change and what you can rely on, so adopting the
package early is safe on both sides.

## Before v1.0.0

Any part of the API may change between minor releases, without a deprecation
cycle. What you can rely on:

- **The changelog is complete.** Renames, removals, and behavior changes are
  listed in `CHANGELOG.md` under the release that made them; skim it before
  upgrading.
- **Schemas refuse rather than misread.** The fit archive (`.fit.h5`) carries
  a schema version and refuses to read or append across mismatched versions.
  A format-version key for model YAML files is planned so future syntax
  changes can warn or migrate instead of silently misparsing old files.
- **Version numbers signal risk.** Patch releases (`0.x.y` → `0.x.y+1`)
  contain only fixes and backwards-compatible additions; anything breaking
  lands in a minor release (`0.x` → `0.x+1`).

Pin to a minor version (e.g. `trspecfit>=0.12,<0.13`) where you need
reproducibility. Model YAML files and `.fit.h5` archives you accumulate are
treated as long-lived artifacts on our side.

## From v1.0.0 on

The user API — `Project`, `File`, `FitResults`, `Simulator`, `PlotConfig`
(the top-level exports) and the YAML model format — gains a deprecation
cycle: before a public name is removed or renamed, the old name keeps
working and emits a `DeprecationWarning` pointing to the replacement for
at least six months and at least one intervening minor release, whichever
is longer.

The rest of the importable surface (e.g. the model-building layer in
`trspecfit.mcp`) is not yet classified. An API-tier guide separating stable,
advanced, and internal modules is planned before v1.0.0; once it exists, the
stability commitment for the advanced tier will be stated there. Compiled
internals (`graph_ir`, `eval_1d`, `eval_2d`, `eval_jax`, and low-level
parsing/HDF5 helpers) carry no stability guarantees at any version.
