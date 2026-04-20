---
orphan: true
---

# Archived Decision: Numba vs JAX for the Lowered Evaluator

> Archived on 2026-04-17 after the backend decision closed.
> Keep [../lowered_evaluator.md](../lowered_evaluator.md) as the long-lived
> design/spec and `PLAN.md` at the repo root as the implementation
> tracker. This file preserves the benchmark data and tradeoff analysis behind
> the decision.

## Context

Phase 4 of the execution plan landed the Graph IR (GIR) and the lowered
2D evaluator. On the default benchmark example
(`02_dependent_parameters`), the GIR path is roughly 3× faster per call
than the MCP interpreter while staying numerically identical
(`max |diff| ~ 1e-13`). The question this document is trying to answer
is what the next backend step should be:

- **Numba** — incremental acceleration. Keep the NumPy-style evaluator
  and `@njit` the hot loops.
- **JAX** — a larger rewrite of the evaluator into a pure-array,
  mutation-free function that can be fully JIT-compiled, with optional
  follow-on work for analytic Jacobians (`jax.jacrev`) and eventually
  GPU acceleration.

We are on the `np-to-jax` branch, which signals an initial lean toward
JAX, but the decision is not made. The purpose of this document is to
(a) lay out the real work/speedup tradeoff as precisely as the current
information allows, (b) call out the numbers that are shakiest, and
(c) specify a benchmark plan that should collapse the decision into
something defensible.

## Current state of the lowered evaluator

Relevant entry points (for grounding the analysis below):

- `evaluate_2d` at [src/trspecfit/eval_2d.py:115](../../../src/trspecfit/eval_2d.py#L115),
  with two hot loops:
  - Resolution loop at [eval_2d.py:155–173](../../../src/trspecfit/eval_2d.py#L155-L173):
    dynamics groups + inline expression evaluation.
  - Component loop at [eval_2d.py:180–205](../../../src/trspecfit/eval_2d.py#L180-L205):
    peak/background op dispatch via `OP_DISPATCH`.
- `eval_expr_program` RPN evaluator at [eval_2d.py:43–106](../../../src/trspecfit/eval_2d.py#L43-L106).
- `ScheduledPlan2D` at [src/trspecfit/graph_ir.py:399](../../../src/trspecfit/graph_ir.py#L399).
  Most fields are already flat NumPy arrays (`opt_indices`,
  `dyn_group_target_row`, `op_param_indices`, ...); the one notable
  exception is `expr_programs: list[ExprProgram]` at
  [graph_ir.py:432](../../../src/trspecfit/graph_ir.py#L432), which holds
  Python-wrapped instruction arrays.
- Voigt / Faddeeva via `scipy.special.wofz` at two sites only:
  [functions/energy.py:272](../../../src/trspecfit/functions/energy.py#L272)
  and [functions/time.py:357](../../../src/trspecfit/functions/time.py#L357).
- Plan construction: `build_graph` + `schedule_2d` at
  [src/trspecfit/trspecfit.py:2755–2757](../../../src/trspecfit/trspecfit.py#L2755-L2757),
  once per `File.fit_2d` call.
- Outer optimizer: `lmfit.Minimizer(residual_fun, par_ini, ...)` at
  [src/trspecfit/fitlib.py:574](../../../src/trspecfit/fitlib.py#L574),
  with no `Dfun` currently wired.

A few properties of the evaluator that matter for the Numba/JAX
comparison:

- The evaluator is **mutation-heavy**. `traces[...] = ...`, `+=`, and
  `.copy()` show up roughly eight times between
  [eval_2d.py:145](../../../src/trspecfit/eval_2d.py#L145) and
  [eval_2d.py:205](../../../src/trspecfit/eval_2d.py#L205). JAX's pure-array
  model requires each of these to become a functional update.
- The evaluator has **no theta-dependent branches**. Every `for` and
  `if` depends on `plan` state that is fixed at `schedule_2d` time.
  Good for either backend: branches are plan-dependent, so they can be
  unrolled at trace time (JAX) or turned into straight-line integer
  dispatch (Numba).
- `DYNAMICS_DISPATCH` at [eval_2d.py:27](../../../src/trspecfit/eval_2d.py#L27)
  and `OP_DISPATCH` imported from `graph_ir` both map integer kinds
  to Python callables that live in `src/trspecfit/functions/energy.py`
  and `src/trspecfit/functions/time.py`. Neither Numba nor JAX can
  follow a Python-callable table inside a compiled region, so both
  require some form of flattening or upstream decoration.

## Measured benchmarks (2026-04-15)

Two of the variance-reducing benchmarks listed further down have been
run; this section records the measurements and the consequences for the
analysis below. The raw flamegraph from py-spy is at
[docs/design/archive/numba_vs_jax_benchmarks.svg](numba_vs_jax_benchmarks.svg);
the cProfile and nfev captures were produced by
`.claude/skills/benchmark/benchmark_gir.py` with the `--profile` and
`--nfev` flags.

### Benchmark #1: per-call profile of GIR (example 02)

Grid 212 × 1131. Before any further optimization the GIR per-call was
10.58 ms (cProfile over 1000 calls). Time breakdown by function
(`tottime` = self time, excluding sub-calls):

| Bucket                                | % of wall time | Note                                                              |
|---------------------------------------|----------------|-------------------------------------------------------------------|
| `GLP` peak function                   | **72.3%**      | Pure numpy: `np.exp` + arithmetic. Two peaks × 1020 calls         |
| `Shirley` (incl. `cumsum`)            | **11.0%**      | `flip + cumsum + flip`. Pure numpy                                |
| `evaluate_2d` self-time (dispatch)    | **7.2%**       | Resolution loop, op-dispatch loop, param list comprehensions      |
| `ndarray.copy`                        | **6.2%**       | 7144 copies across [eval_2d.py:145, 177, 178, 80](../../../src/trspecfit/eval_2d.py#L145) |
| `fit_model_gir` wrapper               | **2.3%**       | Entry overhead                                                    |
| `eval_expr_program` self              | **0.2%**       | RPN evaluator is not a hotspot                                    |
| Everything else                       | <1%            | Enum lookups, list pops, `np.full`, etc.                          |

The dominant finding is that **Python dispatch is ~10% of per-call
time, not 40%**. The 40% threshold I originally wrote into the decision
criteria below is not met. The time is overwhelmingly in NumPy
compute, concentrated in `GLP`. This reframes the decision: a
compiled-backend win on this workload comes from kernel fusion of
GLP-like expressions, not from eliminating interpreter overhead.

### GLP fusion rewrite (free, no backend)

The original GLP at [energy.py:342–346](../../../src/trspecfit/functions/energy.py#L342-L346)
computed `((x - x0) / F) ** 2` twice — once in the `np.exp` numerator
and once in the denominator. Factoring it into a shared `u2` variable:

| Metric                          | Before    | After    | Ratio   |
|---------------------------------|-----------|----------|---------|
| GIR per-call (example 02)       | 10.58 ms  | 8.38 ms  | 1.26×   |
| GLP self-time (per 1000 calls)  | 7.81 s    | 5.87 s   | 1.33×   |
| GIR vs MCP speedup              | ~3.1×     | 3.82×    | --      |

Parity preserved (`max |diff| = 1.14e-13`); all 60
`tests/test_functions_energy.py` cases pass. This is also a
**lower-bound signal on how much kernel fusion is worth on this
workload**: ~1/3 of GLP time was the redundant quadratic. Further
fusion (folding `np.exp`, the division, and the multiplies into a
single pass over memory) is exactly what Numba and especially
JAX-via-XLA are designed to do — but the remaining ceiling is
constrained by the cost of `np.exp` itself, which is hardware-bound
the same under every backend.

### Benchmark #2: nfev per fit

Residual evaluation counts across examples 01–04 (example 05 is
project-level, different file layout, skipped). Pipeline is
`define_baseline → fit_baseline(stages=2) → add_time_dependence →
fit_2d(stages=2)`.

| Example                                 | Baseline nfev | fit_2d nfev | Total   |
|-----------------------------------------|---------------|-------------|---------|
| 01 basic_fitting                        | 46            | 627         | **673** |
| 02 dependent_parameters                 | 50            | 634         | **684** |
| 03 multi_cycle                          | 40            | 191         | **231** |
| 04 par_profiles                         | 17            | 483         | **500** |

Typical fit lives in the **200–700 residual evaluations** range. This
is well above the "30 evals" pessimistic case that would have capped
the Jacobian ceiling. With analytic Jacobians via lmfit's `Dfun`:

- Finite-difference Jacobian (current): `N_free + 1` residual calls
  per LM iteration, where `N_free` is the number of free optimizer
  parameters.
- Analytic Jacobian via `jax.jacrev`: ~1 forward + ~1 Jacobian call
  per iteration (the Jacobian call itself costs ~1–3× forward
  depending on `N_free` and the graph shape).
- Net per-iteration cost ratio ≈ `(N_free + 1) / (1 + c)` where `c` is
  the Jacobian overhead factor.

For the examples above, `N_free` is in the ~4–8 range (most
parameters are fixed or expression-bound). That gives a per-iteration
ratio of roughly **1.5–3×** from Jacobian alone. Real user models with
more free parameters would see larger ratios; VARPRO amplifies further
by projecting linear parameters out of the optimizer's view entirely.

**Important caveat on these nfev numbers.** The bundled examples are
*tutorial-grade*: they have few free parameters (most are fixed or
expression-bound), tight bounds, and good initial guesses, so both
the number of LM iterations and the per-iteration FD cost are near
their floor. Real production fits typically have more free
parameters (`N_free` well above 8), looser priors, and worse initial
guesses — both factors push `nfev` higher and scale the analytic-
Jacobian win linearly with `N_free`. The 1.5–3× "Jacobian-alone"
range measured here should therefore be read as a *lower bound*, not
a typical figure, for the regime where a JAX+Jacobian port would
actually be deployed.

Combined with JAX's per-call fusion win (see revised Option B
speedups below), the realistic fit-level win for JAX-with-Jacobian
lands in the **3–8×** band for the current examples, with room to
grow for larger models.

### Benchmark #3: plan-build vs fit wall time

Measured via the `--plan-time` flag on `benchmark_gir.py`, which
monkey-patches `trspecfit.graph_ir.build_graph` and `schedule_2d`
with timers and then runs the standard `define_baseline →
fit_baseline(stages=2) → add_time_dependence → fit_2d(stages=2)`
pipeline. Plan total is the sum of `build_graph` + `schedule_2d`
elapsed time; the ratio is against the `fit_2d` wall clock.

| Example                       | build_graph | schedule_2d | Plan total | fit_2d wall | Ratio      |
|-------------------------------|-------------|-------------|------------|-------------|------------|
| 01 basic_fitting              | 0.21 ms     | 0.00 ms†    | 0.21 ms    | 8.05 s      | **0.003 %** |
| 02 dependent_parameters       | 0.24 ms     | 0.93 ms     | 1.17 ms    | 5.59 s      | **0.021 %** |
| 03 multi_cycle                | 0.28 ms     | 0.00 ms†    | 0.28 ms    | 5.46 s      | **0.005 %** |
| 04 par_profiles               | 0.15 ms     | 0.00 ms†    | 0.15 ms    | 2.09 s      | **0.007 %** |

† `schedule_2d` is only invoked for lowerable models; examples 01,
03, and 04 are non-lowerable on their default configurations and
fall back to the MCP interpreter for `fit_2d`, which is also why
their per-fit wall times are longer despite comparable nfev. The
ratios still answer the question.

Plan construction is **0.003 %–0.021 %** of fit wall time, three to
four orders of magnitude below the 10 % threshold that would have
justified opening a separate plan-builder optimization track. The
plan builder is not a bottleneck; evaluator-side work is where the
remaining speedup lives.

### What the measurements change in this document

- The **"Numba clearly wins if dispatch > 40%"** threshold in Decision
  Criteria is not met; dispatch is ~10%. Numba's leverage on this
  workload is smaller than originally projected.
- The **JAX + Jacobian fit-level multiplier**, previously flagged as
  the shakiest number, is now bracketed by measured nfev and
  reasonable Jacobian-cost assumptions. Still a range, but no longer
  "we don't know if this matters at all."
- The **free GLP rewrite** absorbed a visible slice of the
  compiled-backend headroom. Any backend now competes against an
  8.4 ms/call baseline, not 10.6 ms.
- **Grid-size sweep (benchmark #4) is deprioritized**: the
  default-size profile already answered the Numba-vs-JAX question for
  Step 5.3. It would only re-enter scope if a small-grid workflow
  becomes load-bearing.

The per-call and decision-criteria sections below have been revised
to reflect these numbers.

## Option A: Numba

### Scope

Put `@njit` on `evaluate_2d` and `eval_expr_program`. Keep everything
else — plan construction, lmfit integration, user-facing fit API, and
most of the `functions/` registry layout — unchanged.

### Concrete changes required

- **Flatten `expr_programs`.** Replace the `list[ExprProgram]` field
  with two flat arrays: an instruction buffer and a per-expression
  indptr. This is a `schedule_2d` change, not purely an evaluator
  change.
- **Replace dispatch tables with an integer ladder.** Both
  `DYNAMICS_DISPATCH` and `OP_DISPATCH` map enum values to Python
  callables; inside `nopython` mode this cannot stand. Two options:
  - Decorate every registry function upstream with `@njit` so they can
    be called through a jitted dispatch. This touches
    [functions/energy.py](../../../src/trspecfit/functions/energy.py) and
    [functions/time.py](../../../src/trspecfit/functions/time.py)
    broadly, and the parser introspection that reads positional
    parameter lists must continue to work after decoration.
  - Reimplement the formulas inline in a single `if/elif` ladder in
    the evaluator. Duplicates math but isolates the Numba surface.
  Either path is more work than it sounds; this is the largest single
  hidden cost in a Numba port.
- **Replace Python list gathers.** The `params: list[np.ndarray]`
  comprehension at [eval_2d.py:189–191](../../../src/trspecfit/eval_2d.py#L189-L191)
  and the row-by-row gathers in the dynamics loop become typed array
  gathers.
- **Handle `wofz`.** Numba cannot call `scipy.special.wofz` directly in
  nopython mode. Two practical options:
  - `numba.extending.get_cython_function_address` plus a ctypes wrapper
    around `scipy.special.cython_special.wofz`. This works but is
    fragile across SciPy versions.
  - Inline a Humlíček-W4 approximation (~20 lines). Accuracy ~1e-6 in
    the Voigt profile, which is well below the residual noise of a
    typical fit.
  The Humlíček route is the recommended path.

### Estimated work

Roughly **1.5–2 weeks** for a working port with parity tests against
the current NumPy evaluator. The registry-decoration work is the
dominant risk on the schedule; if any registry function uses a NumPy
feature Numba does not support (`np.where` with 1-argument form, some
fancy indexing idioms), it expands to a day or two per function.

### Expected speedup over current GIR

Revised downward after the measurements above: the per-call profile
showed Python dispatch is only ~10% of wall time, so Numba's primary
leverage is smaller than originally projected. Its remaining win comes
from kernel fusion of the NumPy expressions in `GLP` and similar peak
functions.

- Per-call (default-size grid): **~1.3–2×** over the current
  post-GLP-rewrite baseline. High end assumes Numba fuses the GLP
  expression into a single pass and removes the ~6% `ndarray.copy`
  cost; low end assumes fusion gains are marginal on top of what
  NumPy + the manual GLP factoring already do.
- Fit-level: similar ratio (~1.3–2×), since nfev is unchanged under
  Numba — there is no Jacobian path here.

### Strengths

- Lowest-risk incremental path.
- Keeps the NumPy-style coding style — still readable for future
  contributors.
- `parallel=True` + `prange` is a further knob if the component loop
  is long.
- No heavy new dependency; Numba is a relatively contained addition.

### Weaknesses

- The registry-decoration surface is larger than the evaluator itself.
- Does not buy analytic Jacobians, GPU acceleration, or autodiff.
- `wofz` has no clean solution; the Humlíček inline is maintained by
  us.

## Option B: JAX (full JIT)

### Scope

Convert `evaluate_2d` into a pure function of flat arrays, JIT it with
`jax.jit`, and optionally follow with `jax.jacrev` for analytic
Jacobians. lmfit can stay as the outer optimizer; the JAX-jitted
evaluator is called from the existing `residual_fun`.

### Concrete changes required

- **Everything in the Numba scope** — expr flattening, dispatch
  flattening, a `wofz` replacement (Humlíček in JAX). These are
  prerequisites for either backend.
- **Mutation → functional.** Every `traces[...] = X` becomes
  `traces = traces.at[...].set(X)` and every `+=` becomes
  `.at[...].add(...)`. Eight-ish sites across
  [eval_2d.py:145–205](../../../src/trspecfit/eval_2d.py#L145-L205).
  Mechanical but pervasive — every hot-path statement in the
  evaluator has to be rewritten.
- **Drop `.copy()` calls.** JAX arrays are immutable; copies are
  implicit where needed. The scratch-buffer pattern (`traces = ...copy()`)
  gives way to a threaded state variable carried through the
  resolution loop.
- **Remove Python exceptions from the jitted region.** The length
  check at [eval_2d.py:138](../../../src/trspecfit/eval_2d.py#L138)
  becomes a host-side check outside `jit`.
- **Convert dispatch to JAX control flow.** `lax.switch` over
  `op_kind` / `dyn_func_id`, or trace-time unroll: because `plan.n_ops`
  and the schedule are fixed for a given plan, unrolling at trace time
  is a legitimate simplification. Trade-off: unrolled trace is fast
  but must be re-traced each time the plan shape changes.
- **lmfit boundary shim.** `residual_fun` converts the lmfit parameter
  dict to a flat `jnp.array` of optimizer parameters on every call,
  then converts the JAX output back to a NumPy array for lmfit's
  downstream machinery. Per-call overhead is small but nonzero.
- **Analytic Jacobian (separate, follow-on step).** Wrap the evaluator
  with `jax.jacrev`, package as an lmfit `Dfun`, and pass it in at
  [fitlib.py:574](../../../src/trspecfit/fitlib.py#L574), which currently
  constructs `lmfit.Minimizer` without `Dfun`. Verified feasible:
  `lmfit.Minimizer.minimize(method="leastsq", ...)` accepts `Dfun`
  through its Minimizer constructor. This is additional plumbing, not
  free.

### Estimated work

Roughly **3–4 weeks** for a working port with parity tests, plus
**~1 week** to wire analytic Jacobians through lmfit. The mutation →
functional rewrite is what inflates the estimate above Numba; it
touches essentially every hot-path statement.

### Expected speedup over current GIR

Revised using the per-call profile (72% in `GLP`, ~10% dispatch, 6%
copy cost) and the measured nfev range of 200–700 per fit.

- Per-call (default-size grid): **~1.5–3×** over the
  post-GLP-rewrite baseline on CPU. XLA fusion of the GLP-style
  expressions is the main source; copy-cost elimination (mutation →
  functional, no more `.copy()` calls) adds a bit. This range already
  accounts for JAX's per-op launch overhead, which drags the low end
  on small grids.
- **With analytic Jacobian (lmfit `Dfun`): fit-level ~3–8×** on
  examples like 01–04 (~4–8 free parameters). Per-iteration cost
  ratio is roughly `(N_free + 1) / (1 + c)` where `c ≈ 2–3` is the
  `jax.jacrev` overhead factor — so the Jacobian alone contributes
  1.5–3× on top of the per-call win. Larger user models would see
  larger ratios.
- **GPU** (only if grids get very large): 10–50× per call, but
  Python-side launch overhead makes this counterproductive for
  current default-size grids.

### Strengths

- Enables analytic Jacobians (`jacrev`) — the main structural payoff
  beyond raw JIT.
- Enables GPU acceleration if/when grids grow.
- XLA fusion often beats hand-vectorized NumPy on large grids.
- Pushes the plan shape toward "array-of-arrays", which is also the
  natural shape for VARPRO (see Potential Next Steps).

### Weaknesses

- Significant rewrite of the evaluator body.
- JAX is a heavy dependency.
- Tracing and compilation happen at the first call with a given
  shape. If plans rebuild per fit
  ([trspecfit.py:2755](../../../src/trspecfit/trspecfit.py#L2755)) and
  each plan has a unique shape signature, we pay the trace cost at
  every fit. If many fits share a plan shape, we amortize.
- No native `scipy.special.wofz` under JIT — Humlíček maintained by us.

## Work vs speedup summary

Updated with measured data from benchmarks #1 and #2. All per-call
ratios are relative to the current GIR baseline *after* the GLP
fusion rewrite (~8.4 ms/call on example 02).

| Path                    | Per-call (vs current GIR) | Full fit            | Regression risk | Work (weeks) |
|-------------------------|---------------------------|---------------------|-----------------|--------------|
| Numba                   | ~1.3–2×                   | ~1.3–2× (no Dfun)   | Low             | 1.5–2        |
| JAX CPU                 | ~1.5–3×                   | ~1.5–3× (no Dfun)   | Moderate        | 3–4          |
| JAX CPU + Jacobian      | (same per-call)           | ~3–8× on current examples | Moderate        | 4–5          |
| JAX GPU (large grids)   | 10–50×                    | 10–50×              | High            | 4–5 + infra  |

The JAX+Jacobian row now rests on the measured nfev range (231–684
across examples 01–04) rather than on an unmeasured assumption. It
remains a range rather than a point estimate because `N_free` and the
Jacobian overhead factor `c` vary by model; the range brackets
plausible combinations.

## Open questions and shaky numbers

Resolved by benchmarks #1 and #2:

- ~~**JAX + Jacobian fit-level multiplier.**~~ Measured: nfev is
  231–684 across examples 01–04, `N_free` ~ 4–8, so plausible
  Jacobian win is 1.5–3× on top of the per-call ratio. See Measured
  benchmarks above.
- ~~**Python dispatch fraction.**~~ Measured: ~10% of per-call time.
  Numba's primary leverage is accordingly smaller than originally
  projected.

Also resolved:

- ~~**Plan-build fraction of fit time.**~~ Measured: 0.003–0.021 %
  across examples 01–04 (benchmark #3 above). Plan construction is
  not a bottleneck; no separate track needed.

Still open, contingent on direction:

- **Numba registry decoration scope.** How many functions across
  `functions/energy.py` and `functions/time.py` will actually survive
  `@njit` as-is? Needs a scan before committing to a Numba port.
- **Humlíček accuracy in our Voigt regime.** Standard 1e-6 accuracy
  should be fine, but worth a spot check at the damping / width range
  your fits actually use.
- **JAX trace cache behavior.** Whether per-plan shapes are stable
  across fits determines whether trace cost amortizes or recurs.
  Only material if we commit to a JAX port.

## Benchmark plan to reduce variance

Ordered roughly cheapest / most-load-bearing first. Items struck
through are complete; see "Measured benchmarks" above for results.

1. ~~**Per-call profile of the GIR-only loop.**~~ Done (2026-04-15).
   py-spy + cProfile on `bench_gir_only`, example 02. Output in
   [docs/design/archive/numba_vs_jax_benchmarks.svg](numba_vs_jax_benchmarks.svg).
   Answered: Python dispatch ~10%, NumPy math ~83% (GLP dominant).
2. ~~**`nfev` capture per fit.**~~ Done (2026-04-15). Counter wrapper
   around `fitlib.residual_fun`; run via
   `benchmark_gir.py --example 0 --nfev`. Range 231–684 across
   examples 01–04.
3. ~~**Plan-build vs fit time ratio.**~~ Done (2026-04-15).
   Monkey-patched `build_graph` + `schedule_2d` with timers; run via
   `benchmark_gir.py --example 0 --plan-time`. Range 0.003–0.021 %
   across examples 01–04 — well below the 10 % threshold.
4. ~~**Grid-size sweep.**~~ Deprioritized for Step 5.3. The
   default-size profile already answered the Numba-vs-JAX question;
   grid sweep re-enters scope only if a small-grid workflow becomes
   load-bearing.
5. **Jacobian prototype — only if committing to JAX.** Prototype
   `jax.jacrev` on a single peak function, pass the resulting `Dfun`
   to lmfit on a minimal test. Confirms the measured-nfev-based
   projection (~1.5–3× from Jacobian alone) before committing to the
   full port.

Side benchmarks, worth doing in parallel if time permits:

- **Humlíček accuracy + speed.** Head-to-head vs `scipy.special.wofz`
  at typical Voigt widths. Tells us whether this replacement is a
  cheap shim or a real liability.

## Decision criteria

The original criteria were written around thresholds (`dispatch > 40%`,
`nfev < 50`, `plan-build > 10%`) that benchmarks #1–#3 have now all
settled. Updated to reflect what the measurements actually say:

**Numba is a fit when:**

- The target is a modest, low-risk per-call win (~1.3–2×) on the
  existing evaluator, AND
- There is no near-term plan to exploit analytic Jacobians, VARPRO, or
  GPU, AND
- We prefer the smaller dependency footprint.

The measurements show dispatch is only ~10% of per-call time, so
Numba's ceiling on this workload is modest. It remains a reasonable
choice if the goal is "make what we have incrementally faster and
stop there."

**JAX is a fit when:**

- We want the fit-level win to come from analytic Jacobians (measured
  nfev 231–684 makes a 1.5–3× Jacobian contribution realistic), OR
- We foresee needing GPU acceleration at some point, OR
- VARPRO is on the roadmap — JAX's array-of-arrays shape is closer to
  what VARPRO needs.

**Do neither right now:**

- A legitimate option. The GLP / GLS / DS fusion rewrites already
  landed a cheap ~1.26× on GIR with zero backend change, and GIR is
  already ~3.8× over MCP. Starting a multi-week port is only
  justified if the fit-level targets above are actually on the
  roadmap; otherwise we can close Step 5.3 at the current
  performance and revisit when scope changes.

**Conclusion, based on the data in "Measured benchmarks":**

Benchmarks #1–#3 have collapsed the decision-relevant variance:

- Dispatch is not the bottleneck (#1, ~10%).
- Jacobians have real headroom (#2, nfev 231–684).
- Plan construction is not a bottleneck (#3, <0.05 %).

The remaining question is no longer a benchmark question — it is a
scope/product question: **do you want analytic Jacobians, VARPRO, or
GPU in the future?** If yes, JAX is the right investment. If no,
Numba is a modest incremental win and "stop here" is also valid.
Step 5.3 as framed in the execution plan is effectively closed.

## Sequencing relative to `can_lower_1d`

The original proposal considered implementing `can_lower_1d()` to
extend the GIR path to `fit_baseline` / `fit_spectrum` **before**
making the Numba-vs-JAX decision. The reasoning was to land the 1D
lowering in the current NumPy idiom, merge, then optimize.

Against that: the 1D port is valuable work but does not inform the
backend decision. It replays the same GIR infrastructure on a
different axis. Doing it before the backend choice risks building 1D
infrastructure in a shape the chosen backend then wants to change.
Cleaner sequencing:

1. ~~Run benchmarks #1, #2, and #3.~~ Done (2026-04-15).
2. Decide Numba vs JAX (or neither) from the resulting data — see
   Decision criteria above.
3. Port 2D to the chosen backend, if any.
4. Then do `can_lower_1d` on top of the winning backend, reusing the
   same plan shape and op set.

## Potential Next Steps

These items were originally listed as steps 5.4 and 5.5 in the
execution plan. They presuppose that the backend decision above has
been made.

**Analytic Jacobians (only if JAX is chosen)**

- Wrap the evaluator for `jax.jacrev`.
- Pass the Jacobian to lmfit via the `Dfun` parameter. Requires
  plumbing at
  [src/trspecfit/fitlib.py:574](../../../src/trspecfit/fitlib.py#L574),
  which currently constructs `lmfit.Minimizer` without `Dfun`.
- Measure iteration count reduction against the finite-difference
  baseline.

**Variable projection (VARPRO)**

- Identify linear parameters from the graph structure.
- Implement VARPRO separation so linear parameters are solved by least
  squares inside each nonlinear iteration rather than by the outer
  optimizer.
- Reduce optimizer dimensionality.

Both are meaningful accelerators independent of each other, and both
are orthogonal to the 1D lowering question.
