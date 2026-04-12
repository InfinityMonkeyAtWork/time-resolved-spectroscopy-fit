---
name: benchmark
description: 'Benchmark GIR compiled evaluator vs interpreter. Args: example number (default: 2), `--fit` for full fit, `-n N` for repetitions.'
argument-hint: '[example_num] [--fit] [-n N]'
disable-model-invocation: true
allowed-tools: Bash(.venv/bin/python .claude/skills/benchmark/benchmark_gir.py *)
---

## Benchmark: GIR vs Interpreter

Run `benchmark_gir.py` to compare the compiled (GIR) and interpreter (MCP)
evaluation paths on an example fitting workflow.

### Available examples

!`ls -d examples/fitting_workflows/0[0-9]_*/ 2>/dev/null | grep -v _fits | while read -r d; do printf '  %s\n' "$(basename "$d")"; done`

Only examples whose dynamics are lowerable (no convolution, no subcycle)
produce a GIR vs MCP comparison. Currently example **02** is the default.

### Task

Parse the arguments from `$ARGUMENTS`:
- First positional integer -> `--example N` (default: 2)
- `--fit` -> include full-fit benchmark
- `-n N` -> fit repetitions

Run:

```
.venv/bin/python .claude/skills/benchmark/benchmark_gir.py --example <N> --calls 200 [--fit] [-n <N>]
```

Report the results to the user. Highlight the speedup ratio and note
whether the model was lowerable.
