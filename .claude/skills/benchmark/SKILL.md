---
name: benchmark
description: 'Benchmark GIR compiled evaluator vs interpreter. Args: example number (default: 2), `--fit` for full fit, `-n N` for repetitions.'
argument-hint: '[example_num] [--fit] [-n N]'
disable-model-invocation: true
allowed-tools: Bash(.venv/bin/python .claude/skills/benchmark/benchmark_gir.py *)
---

Use [../../../docs/ai/benchmark.md](../../../docs/ai/benchmark.md) as the
source of truth for this skill.

Pass through the user's optional benchmark arguments unchanged:

- `[example_num] [--fit] [-n N]`

When this wrapper and the shared doc differ, follow the shared doc.
