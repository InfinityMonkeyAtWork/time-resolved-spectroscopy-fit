# Agent Orientation

Orientation for AI coding agents working on this repository — developing
`trspecfit` itself, not using it for analysis. If you want to *use*
`trspecfit` inside notebooks or scripts, read [llms.txt](llms.txt) instead.
If your tool does not load these files automatically, read them before
making changes.

- [CLAUDE.md](CLAUDE.md) — authoritative: behavior rules, architecture
  guardrails, code style, and testing conventions. Follow it even if you
  are not Claude.
- [TODO.md](TODO.md) — high-level project goals; the `[ACTIVE]` tag marks
  the feature currently in progress.
- [PLAN.md](PLAN.md) — live plan for active multi-step feature work (states
  when nothing is in progress).
- [docs/design/repo_architecture.md](docs/design/repo_architecture.md) —
  module map and the two-layer design (readable authoring layer vs.
  compiled hot path).
- [docs/design/supported_models.md](docs/design/supported_models.md) —
  source of truth for supported model combinations, expressions, and
  compositions.
- [docs/ai/](docs/ai/index.md) — step-by-step recipes for common repo
  tasks; consult the matching recipe before hand-rolling one.

Quick facts: run tests with `pytest -q`; Ruff is the linter and formatter;
never commit without explicit user approval of the exact message; on
renames or public-API changes, grep the entire repo — notebooks, YAML,
tests, and docs all reference the public API.
