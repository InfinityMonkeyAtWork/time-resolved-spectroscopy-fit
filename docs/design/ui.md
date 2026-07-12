---
orphan: true
---

# Design Note: Interactive UI Backend Requirements

Forward-looking note (2026-07-11) collecting backend facts and open
design choices for a future interactive UI (live fit preview while the
user edits a model: initial guesses, fit limits, fixed values). Written
while the JAX backend landed; nothing here is implemented UI work.

## Latency picture per interaction

The UI's hot loop is: user tweaks something -> re-evaluate (or re-fit)
-> redraw. What each tweak costs on the compiled backends:

| User action | JAX backend cost | Recompile? |
|---|---|---|
| Change a varying parameter's value/guess | one jitted call (~0.6-10 ms for 2D) | no — theta is the traced input |
| Change a fit limit (`e_lim`/`t_lim`) | free on the compiled side | no — windowing is host-side slicing of the full-grid output; axes are never cropped |
| Change a *fixed* parameter's value | re-schedule + re-trace (~100-500 ms hiccup) | yes — fixed values are baked into the plan and become XLA constants |
| Flip a `vary` flag | re-schedule + re-trace | yes — theta layout changes |
| Change model structure (components, dynamics, profiles) | re-schedule + re-trace | yes — unavoidable, the plan itself changes |

## Known fixes / prerequisites for fluid interaction

- **Full-parameter-vector evaluator variant.** Trace the evaluator over
  the full parameter vector instead of theta (a small
  `make_evaluator_2d_jax` variant). Fixed-value edits and vary-flips
  then become plain input changes; only true structure changes
  recompile. Negligible runtime cost.
- **Session-level evaluator caching.** `File.fit_2d` currently rebuilds
  graph + plan + evaluator on every call. A UI must hold the compiled
  evaluator/Jacobian across interactions (the factory API supports this
  directly) and invalidate only on structure changes.
- **Persistent compilation cache.** `jax_compilation_cache_dir` makes
  XLA compiles survive process restarts, so even a session's first
  render can be warm.

## 1D does not need JAX for interactivity

Measured on the NumPy GIR 1D path (2001-point grid, 2026-07-11):
24 us/eval (GLP) to 188 us/eval (Voigt via scipy `wofz`). A full 1D
leastsq fit is single-digit-to-tens of milliseconds — far below
perception, with no compile hiccups ever. The interaction where the
backend genuinely matters is **2D fit preview** (seconds -> sub-second),
which the JAX backend covers. Keep 1D on the NumPy path.

## SbS throughput: `n_workers` (today) vs `vmap` (possible future)

Slice-by-slice fitting is the other latency-relevant workload (a UI
would want a full SbS refresh after a seed change). Two very different
parallelization models:

- **`fit_slice_by_slice(n_workers=N)` — what exists.** Process-level
  parallelism over whole *fits*: each worker owns a pickled model copy
  and runs complete lmfit optimizations per slice. Backend-agnostic and
  robust; scales ~min(N_cores, n_slices) minus process spawn/pickle
  overhead, which is why it only pays off above ~20 slices. Keeps all
  lmfit machinery (per-slice stderr from the covariance, etc.).
- **`vmap` over slices — what JAX could add.** Array-level batching of
  the *evaluator*: all slices evaluated in one XLA program with no
  per-process copies or pickling. But vmap alone does not parallelize
  the *fits* — each slice is an independent lmfit optimization with its
  own iteration trajectory, and lmfit cannot step hundreds of
  optimizations in lockstep. Exploiting vmap for SbS therefore requires
  a **batched least-squares solver** (a JAX-native LM stepping all
  slices simultaneously until each converges) — i.e., the Phase E
  "replace lmfit" decision from
  [jax-planning.md](jax-planning.md), scoped to the SbS inner loop.

Notably, SbS is the one workload where lmfit itself plausibly *is* the
bottleneck (per-eval cost is tens of microseconds, so per-iteration
Python/lmfit overhead dominates) — making a batched SbS solver the
natural Phase E pilot if throughput ever demands it. It must reproduce
the per-slice error bars that downstream SbS analysis consumes before
it can replace the default. The two models compose poorly (process
pools each re-import jax/XLA), so it would be either/or per fit.
