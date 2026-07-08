# PLAN: time-functions fixes + y0 overhaul

Branch: `fix-time-functions`. Source: audit report (`~/Desktop/functions bugs.txt`),
all findings verified. Expanded scope (agreed 2026-07-06): remove `y0` from all
dynamics primitives in favor of a dedicated step function, plus the Voigt
normalization fix and the remaining docstring corrections.

## Design decisions

- **Remove `y0`** from all six dynamics functions: `linFun`, `expFun`, `sinFun`,
  `sinDivX`, `erfFun`, `sqrtFun`. Rationale: `y0` breaks the causality convention
  (`f(t<t0) = 0`) inconsistently across functions (sqrtFun/erfFun leak `y0` before
  `t0`; siblings clamp to 0), and offsets compose naturally since dynamics
  components sum (`mcp.Model.combine`, comp_type "add").
- **Add `stepFun(t, A, t0)`**: `np.where(t < t0, 0.0, A)` — the causal offset
  primitive. Auto-registered by `config/functions.py` discovery (no `CONV` suffix).
  `erfFun` (sans `y0`) is its Gaussian-broadened counterpart: rises 0 → A.
- **Energy-domain `Offset(x, y0)` is untouched** — different axis, different
  convention; all nonzero `y0` values in repo YAMLs belong to it.
- **Breaking change, no deprecation shim**: pre-1.0; YAML validation
  (`utils/parsing.py` param-count/name check against introspected signatures)
  rejects stale `y0:` entries with a clear error. Changelog documents migration
  (delete `y0: [0, ...]` lines; nonzero baseline → add `stepFun` component).
- Removing `y0` **subsumes** audit items #1 (sqrtFun t<t0 — `clip(0)` alone is
  now correct), #4 (erfFun y0 description), and most of #5/#7.

## Tasks

### Part A — y0 overhaul (`src/trspecfit/functions/time.py`)

- [x] Add `stepFun(t, A, t0)` with full NumPy docstring.
- [x] Remove `y0` from the six signatures + bodies; rewrite each docstring
      (expFun A<0 case: jumps to −|A| at t0, rises toward 0; erfFun: 0 → A).
- [x] Module docstring: convention becomes `func(t, par1, ..., t0)`; drop the
      "Offset Convention" section; carve out erfFun/stepFun edge behavior at t0
      (erfFun is nonzero slightly before t0 — smooth step exception).
- [x] YAML migration — delete `y0:` lines + fix signature comments:
      `tests/models/file_time.yaml`, `tests/models/project_time.yaml`, and
      `examples/**/models_time*.yaml` incl. `data/*_truth.yaml` (01, 02, 03, 04,
      10, 20, 21, synthetic_data 01/02). All values are 0 → no data regeneration.
- [x] Tests: `test_functions_time.py` (drop y0 kwargs; add stepFun tests; assert
      sqrtFun/all primitives are 0 before t0), `test_config_functions.py`
      (signature asserts), `test_graph_ir.py` (param counts, `*_expFun_01_y0`
      node names), `test_model_parser.py` (`par_dict["y0"]`),
      `test_project_fit.py` (lmfit par names), `test_mcp_library.py` (add_pars).
- [x] Docs: `docs/design/repo_architecture.md` (~L216 convention),
      `docs/ai/add-function.md` (~L135), `docs/design/lowered_evaluator.md`
      (signature/node-name references). Notebook markdown tables mentioning
      dynamics y0: `01_basic_fitting`, `21_multi_file_shared_fit`
      `data/generate_data.ipynb` (04's table row is energy Offset — keep).

### Part B — remaining audit fixes

- [x] #2 Voigt normalization — `functions/energy.py:272-274`: replace
      `np.max(voigt, axis=-1, keepdims=True)` with analytic peak
      `np.real(wofz(1j * (W / 2) / SD / np.sqrt(2)))`; add grid-independence
      test (value at fixed x unchanged when window/grid changes).
- [x] #3 expRiseCONV docstring — `time.py`: "Causal" → anti-causal mirror of
      expDecayCONV (kernel nonzero for x ≤ 0).
- [x] #6 boxCONV docstring — remove false "(with smooth edges)".
- [x] #8 GLP/GLS — document `m ∈ [0, 1]`; note m<0 divide-by-zero-crossing.
- [x] #9 DS — document asymmetric tail extends toward x < x0 (increasing-KE
      convention; binding-energy axis sees it mirrored).

## Verification

- `pytest -q`
- Repo-wide grep: `y0` (only energy `Offset` + profile contexts remain),
  `stepFun` (registered, documented, tested)
- Sphinx `-W` build
- `CHANGELOG.md`: breaking-change entry with YAML migration note
