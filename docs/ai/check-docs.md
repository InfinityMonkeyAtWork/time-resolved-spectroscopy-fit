# Check Docs

Shared source of truth for auditing documentation quality before a merge or
release.

Run the following documentation checks in order. Report a summary at the end
with pass/fail per check and any items that need fixing.

## 1. Sphinx build (zero warnings)

```bash
make -C docs html 2>&1 | grep -E "WARNING|ERROR"
```

If any warnings or errors appear, report them with file and line number.

## 2. Docstring style consistency

Search `src/` for Google-style docstrings (colon-terminated section headers
inside docstrings). Every docstring must use NumPy style.

Pattern:

```text
^ {4,8}(Parameters|Returns|Raises|Attributes|Args|Yields|Notes|Examples):\s*$
```

Report any matches with file and line number.

## 3. Missing docstrings on public API

Check every public (no leading `_`) function, method, and class in
`src/trspecfit/` for a triple-quoted docstring immediately after the
definition. `@overload` definitions are excluded since they conventionally
omit docstrings.

```bash
python .claude/skills/check-docs/check_missing_docstrings.py
```

Report any missing docstrings with file and line number.

## 4. Stale docstrings (signature vs Parameters mismatch)

For public functions and methods in user-facing files (`trspecfit.py`,
`mcp.py`, `functions/*.py`, `fitlib.py`, `simulator.py`) and the bridge /
compiled-layer APIs (`spectra.py`, `graph_ir.py`, `eval_1d.py`,
`eval_2d.py`), compare the function signature parameters against the
docstring Parameters section. Skip `self`, `cls`, `*args`, `**kwargs`.

Functions with no `Parameters` section are skipped silently, so minimal
internal docstrings in the compiled layer do not produce noise — only
layer-boundary APIs that document their parameters are audited.

```bash
python .claude/skills/check-docs/check_stale_docstrings.py
```

Report any mismatches (missing, extra, or renamed parameters) with file and
line number.

## 5. Broken cross-references

- Verify all toctree entries in `docs/index.rst` and `docs/api/index.rst` point
  to existing files.
- Verify all `automodule` / `autoclass` / `automethod` targets in
  `docs/api/*.rst` exist in the source.
- Verify relative links in `docs/examples/index.rst` and `docs/quickstart.md`
  point to existing files.

## 6. Exports match docs

Compare `src/trspecfit/__init__.py` exports (`__all__` / top-level imports)
against import statements shown in documentation code examples
(`README.md`, `docs/quickstart.md`, `docs/api/plot_config.rst`).
Flag anything the docs say users can import that is not actually exported.

## Summary

Print a table:

| Check | Status |
|-------|--------|
| Sphinx build | PASS / FAIL |
| Docstring style | PASS / FAIL |
| Missing docstrings | PASS / FAIL |
| Stale docstrings | PASS / FAIL |
| Cross-references | PASS / FAIL |
| Exports match docs | PASS / FAIL |
