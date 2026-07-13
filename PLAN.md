# PLAN: Project-level shared fits on the compiled JAX backend

Branch: `jax-project-fit`. Design source of truth:
[docs/design/project-level-fits.md](docs/design/project-level-fits.md)
(direction settled 2026-07-11). Goal: replace the interpreter path in
`fit_project_mcp()` (`src/trspecfit/spectra.py:349`) with lowered
per-file plans evaluated through a fused JAX residual with a joint
analytic Jacobian, wired into `Project.fit_2d()`
(`src/trspecfit/trspecfit.py:3977`).

## Status

- [x] Branch created; CI fix (release workflow `[dev,jax]`) landed as
  first commit (`043ee08`).
- [x] Phase 0 — recon & decisions (2026-07-13, findings below)
- [x] Phase 1 — theta packing (`spectra.pack_project_theta` +
  `TestPackProjectTheta`, 5 tests)
- [x] Phase 2 — fused JAX residual + joint Jacobian (factories +
  dispatch entries + 6 tests)
- [x] Phase 3 — wiring into `Project.fit_2d()` (dispatch + gating
  helper + whole-project fallback)
- [ ] Phase 4 — tests
- [ ] Phase 5 — benchmark & docs

## Phase 0 — recon & decisions (DONE 2026-07-13)

Findings (file:line refs current as of `043ee08`):

- **Combined-par machinery reusable as-is.**
  `Project._build_fit_params` (`trspecfit.py:1090`) builds the combined
  `lmfit.Parameters` plus `mapping` = `(proj_name, file_idx,
  local_name)` and per-file `name_remap` (local -> combined name;
  file/static prefixed `file{idx:02d}_`). Shared params validate
  identical bounds, warn on differing initial values. Keep all of it;
  the JAX path adds index-array packing on top.
- **Expressions need no combined-level work on the JAX path.** Local
  YAML expressions are lowered into plans (ExprNode); plan opt params
  are exactly the locally *varying* non-expr params
  (`graph_ir.py:2154`). The combined-level expr rewrite in
  `_build_fit_params` pass 2 only feeds the MCP distribution path and
  result write-back.
- **Vary levels map cleanly onto plan opt params.** Local lmfit
  `vary` is True for both `project` and `file` levels, False for
  `static` (`utils/lmfit.py:40`), so a per-file plan built from the
  local model has opt = project+file params, statics baked in as
  constants. Build order matters: baseline-fixed values are written
  into each model *before* plan build (`Project.fit_2d` already does
  this at `trspecfit.py:1291` before `_build_fit_params`).
- **Single-file template.** Dispatch block in `File.fit_2d`
  (`trspecfit.py:4014`-`4062`): `build_graph` -> `can_lower_2d` ->
  `schedule_2d` -> `can_lower_jax_2d` -> `make_evaluator_2d_jax` +
  `make_jacobian_2d_jax`, args convention `(evaluator, jacobian,
  theta_indices, model, dim)`, `jac_fun=fitlib.jacobian_fun` ->
  `Dfun` (leastsq stages only, `fitlib.py:753`). The fused factory
  can reuse `eval_jax._build_evaluate_2d(plan)` per plan inside one
  traced function; jit + jacfwd the fused function once.
- **Windows**: `fit_project_mcp` applies per-file `e_lim`/`t_lim` as
  raw index slices post-eval (`spectra.py:349`); `Project.fit_2d`
  slices data identically and passes empty lims to `residual_fun`.
  The fused traced function applies the same static slices while
  tracing — no dynamic shapes.
- **Jacobian shape**: jacfwd of the fused (windowed, flattened,
  concatenated) function w.r.t. the combined varying theta gives
  `(n_res_total, n_opt_combined)` directly — the shared-param columns
  spanning files come out automatically; block sparsity is XLA's
  problem. A project variant of `fitlib.jacobian_fun` reorders
  columns against the combined lmfit varying names.
- **Weighted residuals**: `sigma_type` is hard-locked to `"constant"`
  and sigma never enters `residual_fun` (metadata only) — constant
  sigma doesn't change the leastsq solution. Nothing to design in
  now; per-file weights would enter the fused function as static
  per-file scale arrays later.

Decisions:

- **Gate**: fused JAX path only when `Project.spec_fun_str ==
  "fit_model_jax"` AND jax importable AND every file's graph passes
  `can_lower_2d` + `can_lower_jax_2d`; any miss -> whole-project
  fallback to the existing MCP path (no mixed backends, no NumPy-plan
  per-file loop — interpreter stays the fallback, per design doc).
- **New spectra entry** `fit_project_jax` following the
  `fit_model_jax` dispatch convention, plus a project-aware Jacobian
  in `fitlib`; result distribution / slot lifecycle
  (`trspecfit.py:1361`-`1414`) unchanged.
- **Constraint carried over**: fused closures don't pickle — no
  parallel-worker MCMC on the project JAX path (same as single-file).

## Phase 1 — theta packing (DONE 2026-07-13)

- [x] `spectra.pack_project_theta(plans, *, mapping, par_names,
  var_names)` returns `theta_c_indices` (positions of varying params
  in the full combined vector) plus per-file `plan_gathers`
  (`theta_f = theta_c[plan_gathers[i]]` in `plan.opt_param_names`
  order). Raises `RuntimeError` on unmapped plan opt params,
  non-varying counterparts, or varying params feeding no plan.
- [x] Tests in `tests/test_project_fit.py::TestPackProjectTheta`:
  synthetic shared/file/static packing with differing per-plan opt
  order, three error paths, and a real-model roundtrip (2-file
  project, shared tau slot identity, gather == name-based lookup).

## Phase 2 — fused JAX residual + joint Jacobian (DONE 2026-07-13)

- [x] `eval_jax._build_project_fused_2d` + public
  `make_project_evaluator_2d_jax` / `make_project_jacobian_2d_jax`
  (`plans, *, plan_gathers, windows, n_theta`): per-plan traced eval
  via `_build_evaluate_2d`, static-slice windows, flatten + concat;
  jit / jit(jacfwd). Gather length/range validated at build time.
- [x] `spectra.fit_project_jax` dispatch entry, args convention
  `(evaluator, jacobian, theta_c_indices, var_names, dim)`; no
  fallback branch (Project gates before choosing the const).
- [x] `fitlib.jacobian_fun_project`: negated fused Jacobian, columns
  reordered theta_c order -> lmfit varying order, mismatch guarded.
- [x] Tests `tests/test_evaluate_jax.py::TestProjectFused` (6):
  evaluator parity vs per-file `evaluate_2d` (incl. jit reuse), joint
  Jacobian vs central FD, shared-column spans both files while
  file-vary columns stay block-local, gather validation errors,
  theta-length error.
- [x] `vmap` batching for homogeneous series deferred (already a
  TODO.md follow-on).

## Phase 3 — wiring into `Project.fit_2d()` (DONE 2026-07-13)

- [x] `Project._build_project_jax_args(combined_pars,
  project_fit_info, windows)` gating helper: per-file `build_graph`
  -> `can_lower_2d` + `can_lower_jax_2d` (any miss -> None), plans
  via `schedule_2d`, `pack_project_theta`, fused factories wrapped in
  `try/except ImportError` (jax missing -> None). Returns the
  `fit_project_jax` args tuple or None.
- [x] Dispatch in `Project.fit_2d`: fused path only when
  `self.spec_fun_str == "fit_model_jax"` and the helper returns args;
  sets `fit_fun_str = "fit_project_jax"` and
  `fit_wrapper_kwargs.setdefault("jac_fun",
  fitlib.jacobian_fun_project)`. Otherwise interpreter const
  (`fit_project_mcp`) unchanged. `show_output` line now reports the
  backend (`JAX` / `interpreter`).
- [x] Data slicing now uses `fitlib._fit_window_slices` and the same
  `windows` list feeds the fused evaluator — data and prediction
  vectors provably share windows.
- [x] Result distribution / slot lifecycle untouched (per-file slots
  still re-evaluate via `fit_model_mcp`).
- [x] Sanity-checked end-to-end (scratch script): 2-file project,
  shared tau, per-file windows on one file, `stages=2` — MCP vs JAX
  final params agree (rtol 1e-6), tau recovered, both backend prints
  observed.
- [x] Same constraints as single-file JAX: no parallel-worker MCMC
  (closures don't pickle), runtime value checks absent.

## Phase 4 — tests

- [ ] Parity: project fit via JAX path vs current MCP path on a small
  multi-file project (shared + per-file params), tolerances per
  existing JAX parity tests in `tests/test_evaluate_2d.py`.
- [ ] Fallback: project with one non-lowerable file uses MCP path.
- [ ] Heterogeneous files (different grids) still fuse (no vmap).
- [ ] `pytest.importorskip("jax")` so min-versions CI stays green.

## Phase 5 — benchmark & docs

- [ ] Benchmark vs interpreter path (docs/ai/benchmark recipe);
  measure compile time growth with file count (open question in
  design doc).
- [ ] Changelog entry; update `docs/design/repo_architecture.md` and
  the design note; bump version (minor -> 0.13.0) before release.
