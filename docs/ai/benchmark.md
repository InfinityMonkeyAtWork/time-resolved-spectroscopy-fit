# Benchmark GIR vs Interpreter

Shared source of truth for benchmarking the compiled GIR evaluator against the
interpreter (MCP) path.

Run `benchmark_gir.py` to compare the compiled and interpreter evaluation paths
on an example fitting workflow.

## Available examples

```bash
ls -d examples/fitting_workflows/0[0-9]_*/ 2>/dev/null | \
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
| 1 | `01_basic_fitting`          | convolution (`MonoExpPosIRF` -> `*CONV` kernel) |
| 2 | `02_dependent_parameters`   | plain dynamics, no conv/subcycle/profile (default) |
| 3 | `03_multi_cycle`            | subcycle dynamics |
| 4 | `04_par_profiles`           | profile models |
| 5 | `05_project_level_fitting`  | not currently supported by the benchmark harness |

Example 02 is the default because it is the cleanest baseline comparison
(pure dynamics, no side paths).

## Task

Parse the arguments:

- First positional integer -> `--example N` (default: `2`)
- `--fit` -> include full-fit benchmark
- `-n N` -> fit repetitions (default: `3`)

Run:

```bash
.venv/bin/python .claude/skills/benchmark/benchmark_gir.py --example <N> --calls 200 [--fit] [-n <N>]
```

Report the results to the user. Highlight the speedup ratio, the
`Max |diff|` correctness check, and note which GIR path the example exercises
(convolution / subcycle / profile / plain).

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
