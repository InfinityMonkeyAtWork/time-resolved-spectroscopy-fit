# Benchmark GIR vs Interpreter (vs JAX)

Shared source of truth for benchmarking the compiled GIR evaluator against the
interpreter (MCP) path.

Run `benchmark_gir.py` to compare the compiled and interpreter evaluation paths
on an example fitting workflow. When the optional `[jax]` extra is installed
(`.venv/bin/pip install -e ".[jax]"`), the per-call benchmark adds a JAX
column (jitted evaluator; reports per-call time, speedup vs both paths, the
one-time XLA compile cost, and max |diff| vs GIR) and `--fit` adds a
`fit_model_jax` run (jitted residuals + analytic Jacobian on the leastsq
stage; each rep pays its own compile). Without jax both report
"skipped".

## Available examples

```bash
ls -d examples/fitting_workflows/[0-9][0-9]_*/ 2>/dev/null | \
  grep -v _fits | \
  while read -r d; do printf '  %s\n' "$(basename "$d")"; done
```

Lowerability is checked per-node by `can_lower_2d()`; there is no blanket
exclusion for convolution or subcycle dynamics — both lower when their
structural contracts are satisfied (resolved-trace time-domain convolution,
subcycle substeps compiled into schedule arrays). The examples exercise
different GIR paths:

| # | example | GIR path exercised |
|---|-----------------------------|--------------------|
| 1 | `01_basic_fitting`            | convolution (`MonoExpPosIRF` -> `*CONV` kernel) |
| 2 | `02_dependent_parameters`     | plain dynamics, no conv/subcycle/profile (default) |
| 3 | `03_multi_cycle_dynamics`     | subcycle dynamics |
| 4 | `04_parameter_profiles`       | profile models |
| — | `10_model_comparison`, `11_save_load_export` | not benchmark fixtures (model-comparison / persistence demos) |
| — | `20_multi_file_independent_fit`, `21_multi_file_shared_fit` | not currently supported by the benchmark harness |

Example 02 is the default because it is the cleanest baseline comparison
(pure dynamics, no side paths). The harness discovers example folders by
their `NN_` numeric prefix, so example numbers in `--example N` map to the
folder names above.

## Task

Parse the arguments:

- First positional integer -> `--example N` (default: `2`)
- `--fit` -> include full-fit benchmark
- `-n N` -> fit repetitions (default: `3`)
- `--par-variability` -> fit-robustness report (see below); `--starts N` sets
  the number of perturbed starts (default: `4`)

Run:

```bash
.venv/bin/python .claude/skills/benchmark/benchmark_gir.py --example <N> --calls 200 [--fit] [-n <N>]
```

Report the results to the user. Highlight the speedup ratios, the
`Max |diff|` correctness checks, and note which GIR path the example exercises
(convolution / subcycle / profile / plain). When the JAX column is present,
mention the compile cost separately — it is paid once per plan, not per call.

## Fit-count and planning-cost modes

Two additional modes report operational characteristics of the fit rather than
a head-to-head speedup:

- `--nfev` — run the standard baseline + `fit_2d` pipeline and report the total
  number of residual evaluations per stage. Useful when checking whether a
  change inflates the fit work (not just the per-call cost).
- `--plan-time` — measure `build_graph` + `schedule_2d` cost against the total
  `fit_2d` wall time. Useful for confirming that planning overhead stays
  negligible relative to the fit itself.

Both modes accept `--example 0` to run across all examples and print a summary
table at the end.

```bash
.venv/bin/python .claude/skills/benchmark/benchmark_gir.py --example <N> --nfev
.venv/bin/python .claude/skills/benchmark/benchmark_gir.py --example <N> --plan-time
```

## Parameter variability (fit robustness vs. initial guesses)

`--par-variability` runs the baseline + `fit_2d` pipeline once from the
example's own initial values (reference) and `--starts` more times (default 4)
with every free parameter's init scaled by a fixed factor ladder
(x0.6 / x0.8 / x1.25 / x1.6, clipped into bounds). No RNG — repeated
invocations give identical numbers.

The report separates the two failure signals:

- **off-optimum runs** — a start that converged to a worse redchi (secondary
  minimum). Excluded from the spread statistics and listed separately.
- **spread among best-optimum runs** — fitted-value spread across starts that
  reached the same optimum. Nonzero spread here indicates a flat objective
  direction (the parameter is not identifiable from the data) or
  init-dependent machinery: state derived once from initial parameter values
  and never rebuilt during the fit.
  Parameters above 1% relative spread are flagged `start-sensitive`.

This is a diagnostic, not a pass/fail test — it deliberately lives here rather
than in the pytest suite, where perturbed-start convergence asserts would be
either weak (tiny perturbations) or flaky (aggressive ones). Accepts
`--example 0` for an all-examples summary.

```bash
.venv/bin/python .claude/skills/benchmark/benchmark_gir.py --example <N> --par-variability
.venv/bin/python .claude/skills/benchmark/benchmark_gir.py --example 0 --par-variability --starts 6
```

## Profiling (GIR path only)

For flamegraphs of the GIR hot path, use `--profile` to run a GIR-only loop
(no interpreter path, no correctness check, no prints inside the loop) and
attach `py-spy` to the subprocess.

Prerequisite (one-time):

```bash
.venv/bin/pip install -e ".[profiling]"
```

Invocation:

```bash
.venv/bin/py-spy record --rate 500 -o docs/design/benchmarks/gir_profile.svg -- \
  .venv/bin/python .claude/skills/benchmark/benchmark_gir.py --example <N> --profile
```

`py-spy` needs permission to attach to the child process. On Linux this
requires either `sudo` or `sysctl kernel.yama.ptrace_scope=0`.
