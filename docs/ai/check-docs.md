# Check Docs

Shared source of truth for auditing documentation quality before a merge or
release.

Run the following documentation checks in order. Report a summary at the end
with pass/fail per check and any items that need fixing.

## 1. Sphinx build (zero warnings)

```bash
.venv/bin/python -m sphinx -b html -W --keep-going docs docs/_build/html
```

`-W` turns every warning into a failing exit code and `--keep-going` still
reports all of them. This build also covers broken toctree entries, missing
autodoc targets and dead MyST links in `.md` pages, so those need no separate
check. Report any warnings or errors with file and line number.

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
definition. Only module- and class-level definitions count, so local
closures inside function bodies are not treated as public API. `@overload`
definitions and `@<name>.setter` / `.deleter` accessors are excluded: the
former conventionally omit docstrings, the latter are documented on the
getter (NumPy convention).

```bash
python .claude/skills/check-docs/check_missing_docstrings.py
```

Report any missing docstrings with file and line number.

## 4. Stale docstrings (signature vs Parameters mismatch)

For every public function and method in `src/trspecfit/`, compare the
signature parameters against the docstring Parameters section. A public
class's `__init__` signature is compared against the class docstring and
against `__init__`'s own docstring, whichever carry a Parameters section;
a class without an explicit `__init__` is not compared. Skip `self`, `cls`,
`*args`, `**kwargs`. Comma-separated names (`x1, x2 : type`) and entries
without a type count as documented, as in NumPy style.

Docstrings with no `Parameters` section are skipped silently, so the minimal
docstrings of internal code do not produce noise. The check never asks for a
Parameters section; a section that exists must list every parameter.

```bash
python .claude/skills/check-docs/check_stale_docstrings.py
```

Report any mismatches (missing, extra, or renamed parameters) with file and
line number.

## 5. Broken relative links in RST pages

Sphinx does not check external-style relative links in RST
(`` `text <../../examples/.../example.ipynb>`_ ``): a dead target builds
clean. Verify every such target under `docs/**/*.rst` exists.

```bash
python .claude/skills/check-docs/check_rst_links.py
```

Report any dead targets with file and line number.

## 6. Exports match docs

Compare `src/trspecfit/__init__.py` exports (`__all__` / top-level imports)
against import statements shown in documentation code examples
(`README.md`, `docs/quickstart.md`, `docs/api/plot_config.rst`).
Flag anything the docs say users can import that is not actually exported.
The script executes every import line shown in `README.md`, `llms.txt` and
`docs/**/*.{md,rst}`:

```bash
python .claude/skills/check-docs/check_doc_imports.py
```

## Summary

Print a table:

| Check | Status |
|-------|--------|
| Sphinx build | PASS / FAIL |
| Docstring style | PASS / FAIL |
| Missing docstrings | PASS / FAIL |
| Stale docstrings | PASS / FAIL |
| RST relative links | PASS / FAIL |
| Exports match docs | PASS / FAIL |
