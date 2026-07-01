---
name: check-example
description: Audit an examples/fitting_workflows notebook against the gold-standard quality bar before a merge or release.
disable-model-invocation: true
---

Use [../../../docs/ai/check-example.md](../../../docs/ai/check-example.md) as
the source of truth for this skill.

This skill takes one argument: the example to audit, as a directory path or an
`NN_` prefix under `examples/fitting_workflows/` (e.g. `04`,
`04_parameter_profiles`, or a full path). With no argument, audit every
`examples/fitting_workflows/NN_*/` directory.

When this wrapper and the shared doc differ, follow the shared doc.
