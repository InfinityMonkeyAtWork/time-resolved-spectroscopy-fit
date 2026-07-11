---
orphan: true
---

# Planning Note: Kernel-Matrix Convolution

## Summary

Replace the 1D-kernel-array convolution
([`my_conv`](../../src/trspecfit/utils/arrays.py) plus the per-theta
`conv_kernel_support` rebuild) with a quadrature-weighted kernel-matrix
operator, on both the mcp and GIR paths, in one branch. Two motivations:

1. **Correctness.** Sample-index convolution is silently wrong on
   non-uniform time axes. Measured on example 21's axis
   (steps 0.5 → 2.0) with its truth parameters (2026-07-10): up to ~7% of
   the trace maximum versus the exact continuous convolution, worst just
   past the step change.
2. **Architecture.** It removes theta-dependent kernel array shapes — the
   main jit blocker for the JAX track
   ([jax-planning.md](jax-planning.md)) — and retires the SciPy
   convolution dependency in the lowered path.

The mcp and GIR changes cannot be split across branches: parity tests
assert the two paths agree to 1e-10, so the convolution semantics must
change everywhere at once.


## The defect today

- Kernels are sampled at `t_step = time[1] - time[0]`
  ([`Component.create_t_kernel`](../../src/trspecfit/mcp.py),
  `resolve_param_traces` in
  [`eval_2d.py`](../../src/trspecfit/eval_2d.py), and the schedule-time
  trace init in [`graph_ir.py`](../../src/trspecfit/graph_ir.py)).
- `my_conv` convolves by sample index; the axis itself never enters the
  computation. On a non-uniform axis the effective IRF width scales with
  the local step (4x wider in example 21's coarse region), and points
  near the step change see a blend of both regimes.
- Nothing guards against this: a non-uniform time axis flows into the
  convolution path silently.
- The shipped example is self-consistent — its data was generated through
  the same operator, so parameter recovery round-trips. Real measured
  data on a fine-around-t0 + coarse-tail axis would produce biased fits,
  concentrated in the IRF width and everything near the step change.


## The operator

For a monotonic time axis `t` (length `n_t`) and kernel function `g`
with fitted parameters `theta_k`:

- `dt[i, j] = t_i - t_j` — theta-independent, built once.
- `w_j` = trapezoid quadrature weights (`np.gradient(t)`) —
  theta-independent, built once.
- Per evaluation: `K[i, j] = g(dt[i, j]; theta_k) * w_j`, rows
  normalized to sum to 1; then `y_conv = K @ y`.

Properties:

- Exact on any monotonic axis; on a uniform axis the weights cancel in
  the row normalization and the result matches the current path up to
  the (small) truncation of the current finite support.
- Static shapes, no support truncation at all — this supersedes the
  dynamic-support machinery from the `fix-conv-kernels` branch rather
  than layering on top of it.
- Differentiable and jit-friendly: per-theta work is an elementwise
  kernel evaluation plus a matmul.
- O(n_t^2) per convolution node per evaluation. At the current
  n_t ~ hundreds this is sub-millisecond; see the benchmark gate below.


## Generality: one operator for 1D and 2D

Call-site inventory — every convolution in the package acts on a 1D
trace sampled on the time axis:

- **mcp (single site).** `Model._combine_component` in
  [`mcp.py`](../../src/trspecfit/mcp.py) handles `comp_type == "conv"`
  for standalone `TIME_1D` models and for dynamics traces inside 2D
  models alike — parameter time dependence evaluates via
  `par.t_model.create_value_1d()`, i.e. the 2D case reuses the 1D
  time-trace evaluator. Replacing this one site covers both.
- **GIR (two sites).** The `kind == 2` branch of
  `resolve_param_traces` in
  [`eval_2d.py`](../../src/trspecfit/eval_2d.py), and the schedule-time
  trace initialization in
  [`graph_ir.py`](../../src/trspecfit/graph_ir.py). The plan gains the
  precomputed `dt` matrix and weights; the per-theta support recompute
  disappears.
- **Future.** The 1D time-trace fitting item in `TODO.md` inherits
  correct convolution on measured (typically non-uniform) delay axes
  automatically — the matrix operator is an enabler for that feature,
  not just compatible with it.
- Energy-domain convolution remains undefined (explicitly rejected in
  `_combine_component`); out of scope here, though the same helper
  would apply if it is ever defined.


## Design decisions

1. **Edge policy.** `my_conv` pads the signal with edge values
   (`mode="edge"`), convolves, and crops. Matrix equivalents:
   (a) row normalization only — kernel mass falling outside the window
   is renormalized over interior samples; or (b) edge-mass
   accumulation — kernel mass beyond each end of the axis is added to
   the first/last column, reproducing edge-padding semantics.
   Recommendation: (b). It matches current behavior on uniform axes
   (signal assumed constant beyond the window — the right assumption
   for saturating or decaying traces) and minimizes golden-value churn.
   Decide before implementation and test both edge rows explicitly.
2. **Kernel bodies evaluate on `dt` directly.** Registry kernel
   functions (`gaussCONV`, …) are elementwise in their first argument,
   so they apply to the `dt` matrix unchanged; no new numeric bodies.
   The `*_kernel_width` helpers and `conv_kernel_support` existed only
   to size the 1D support and are removed together with the
   dynamic-support recompute and its tests.
3. **Normalization.** Per-row, generalizing the current
   kernel-divided-by-sum normalization to non-uniform sampling.
4. **Validation.** Keep a monotonic-axis check at the authoring layer;
   non-uniform spacing stops being a (silent) error condition and
   becomes supported.


## Implementation order and blast radius

1. `utils/arrays.py`: matrix-convolution helper (theta-independent
   builder + per-theta apply), unit-tested against an analytic
   Gaussian-times-exponential reference, uniform-axis agreement with
   the current path within truncation tolerance, and a dense
   continuous-convolution reference on a non-uniform axis.
2. mcp path: `_combine_component` and the `Component` kernel handling
   (`create_t_kernel` path).
3. GIR: plan fields (`dt`, weights; kernel function ids stay), the
   `kind == 2` branch of `resolve_param_traces`, and the schedule-time
   init convolution.
4. Parity: mcp vs GIR tests keep asserting 1e-10 — both paths must
   consume the same helper.
5. Regenerate example 21's data (`generate_data.ipynb`; the old
   operator's artifact is baked into the CSVs) and re-run affected
   examples per `docs/ai/check-example.md`. Update roundtrip/golden
   values once.
6. Benchmark before/after per `docs/ai/benchmark.md`. The O(n_t^2)
   cost is expected to be noise at current sizes; if long time axes
   ever make it matter, the escape hatch is a banded matrix with a
   static support cap derived from parameter bounds (shapes stay
   static).


## Relationship to the JAX track

Land this before Phase B/C of [jax-planning.md](jax-planning.md). It
removes two blockers listed there: the SciPy convolution utilities in
the lowered path, and the theta-dependent kernel shapes introduced by
the dynamic-support fix. After this change, porting convolution to JAX
is an elementwise kernel evaluation plus a matmul.


## Success criteria

- Non-uniform-axis convolution matches a dense continuous-convolution
  reference to floating-point tolerance.
- Uniform-axis results match the previous path within the documented
  truncation tolerance; examples and golden values updated exactly once.
- mcp/GIR parity retained at 1e-10.
- No per-theta array shapes remain anywhere in the convolution path.
- Benchmarks show no regression beyond noise, or document the banded
  fallback and its trigger.
