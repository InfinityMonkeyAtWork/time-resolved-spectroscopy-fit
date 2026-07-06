# Bump Versions

Refresh the pinned dev tooling and GitHub Actions SHAs. Run every few months
or before a release. This skill does the work of Dependabot (ecosystems
`pip` + `github-actions`) manually, as one deliberate bump-everything session
instead of a stream of PRs. If the manual runs get tedious, consider migrating
to Dependabot — but note two things it will NOT handle: the `uv==` pin inside
a `run:` script line in `ci.yaml`, and config migrations that a tool bump
requires (it only surfaces those as red CI on its PRs).

## What gets bumped

1. **`[dev]` exact pins** in `pyproject.toml` (`==` versions).
2. **The `uv==` pin** in the `min-versions` job of `.github/workflows/ci.yaml`
   — it lives in a shell line, cross-referenced by comments from the `[dev]`
   section. Always bump it in the same pass.
3. **GitHub Actions SHA pins** in `.github/workflows/*.yaml`
   (`uses: owner/repo@<sha> # vX.Y.Z`).

Do NOT touch the lower bounds in `[project.dependencies]`. Those are
"supports Python >=3.12" floors, validated by the `min-versions` CI job, and
are only raised when code starts relying on a newer API — never as part of a
routine bump.

## Procedure

### 1. Dev pins (PyPI)

For each package in `[dev]` plus `uv`, look up the latest release:

```bash
curl -s https://pypi.org/pypi/<pkg>/json | python3 -c \
  "import json,sys; print(json.load(sys.stdin)['info']['version'])"
```

Update the pins in `pyproject.toml` and the `uv==` line in `ci.yaml`.

### 2. Action SHA pins (GitHub)

For each `uses:` entry, find the latest tag and resolve it to the **peeled
commit SHA** — release tags are often annotated, and `git ls-remote` shows the
tag-object SHA on the bare ref. Pin the `^{}` (peeled) SHA when one is listed;
only when a tag has no `^{}` line is it lightweight and the bare SHA already
the commit:

```bash
git ls-remote --tags https://github.com/<owner>/<repo> | tail -20  # newest tags
git ls-remote https://github.com/<owner>/<repo> \
  'refs/tags/<tag>' 'refs/tags/<tag>^{}'
```

Update to `owner/repo@<commit-sha> # <tag>` — keep the version comment
accurate, it is the only human-readable trace of what is pinned. Stay on the
same major unless release notes say the workflow inputs are unchanged.

### 3. Verify and run the suite

- Every changed pin must exist upstream: HTTP 200 from
  `https://pypi.org/pypi/<pkg>/<version>/json`, and the commit SHA resolves via
  `https://api.github.com/repos/<owner>/<repo>/commits/<sha>`.
- Reinstall and run the full check suite exactly as `ci.yaml` does: `pytest -q
  -m ""`, `ruff check .` + `ruff format --check .`, `mypy --no-incremental`,
  `pyright`, `deptry .`, `python -m build`, `twine check dist/*`.
- **Read the warnings, not just the exit codes.** Tool bumps can deprecate
  config keys that still "work" (e.g. deptry 0.25 renamed
  `pep621_dev_dependency_groups` → `optional_dependencies_dev_groups`).
  Migrate `[tool.*]` config in the same pass so warnings never accumulate.

### 4. Report

Print a table of `package/action | old | new` plus any config migrations made,
and note anything held back (e.g. a major version skipped) with the reason.
Do not commit — leave the changes for the user to review.
