---
name: code-review
description: "Run the shared code-review checklist. Default: review diff vs main. Args: `full` (whole codebase), `diff` (default), or a file/directory/glob path."
disable-model-invocation: false
---

Use [../../../docs/ai/code-review.md](../../../docs/ai/code-review.md)
as the source of truth for this skill.

Pass through the user's optional scope argument unchanged:

- **`diff`** (default) — review only files changed relative to `main`.
- **`full`** — review all Python source under `src/` and `tests/`.
- **file / directory / glob** — restrict the review to that scope.

When this wrapper and the shared doc differ, follow the shared doc.
