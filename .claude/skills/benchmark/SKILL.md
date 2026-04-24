---
name: benchmark
description: 'Benchmark GIR compiled evaluator vs interpreter. Args: example number (default: 2), `--fit` for full fit, `-n N` for repetitions, `--nfev` for residual-evaluation counts, `--plan-time` for planning vs fit wall time.'
argument-hint: '[example_num] [--fit] [-n N] [--nfev] [--plan-time]'
disable-model-invocation: true
allowed-tools: Bash(.venv/bin/python .claude/skills/benchmark/benchmark_gir.py *)
---

Use [../../../docs/ai/benchmark.md](../../../docs/ai/benchmark.md) as the
source of truth for this skill.

Pass through the user's optional benchmark arguments unchanged:

- `[example_num] [--fit] [-n N] [--nfev] [--plan-time]`

When this wrapper and the shared doc differ, follow the shared doc.
