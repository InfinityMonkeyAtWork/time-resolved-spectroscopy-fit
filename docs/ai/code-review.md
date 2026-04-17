# Code Review Checklist

Shared source of truth for AI-assisted code review in this repo. The Claude
skill wrapper and the Codex wrapper should both point here instead of carrying
their own copies of the checklist.

## Scope

This checklist accepts one optional argument:

- **`diff`** (default) — review only files changed relative to `main`.
  Run `git diff main --name-only -- '*.py'` to get the file list, then
  apply every check below to those files only. Read the full diff with
  `git diff main` for context on what changed.
- **`full`** — review all Python source under `src/` and `tests/`.
- **file / directory / glob** — restrict the review to that scope.

**Naming exclusion**: function registry names (`GLP`, `pExpDecay`, `LinBack`,
`gaussCONV`, etc.) and their parameters (`A`, `x0`, `SD`, `xStart`, `xStop`)
use camelCase/PascalCase because `_` is the component ID delimiter. Exclude
`src/trspecfit/functions/` from naming checks, and ignore variables/parameters
that match registry parameter names elsewhere.

For each item report one of:

- **PASS** — no issues found
- **INFO** — minor observations, no action needed
- **WARN** — issues found, list them with file and line number
- **FAIL** — serious issues found, list them with file and line number

Work through the checklist in order. Use parallel agent/tool calls where items
are independent.

## 1. Bugs and correctness

Read the code in scope looking for:

- Logic errors: wrong condition, off-by-one, inverted boolean, wrong variable
- Incorrect indexing or slicing of arrays/axes
- Wrong variable reuse (e.g., mutating an input that is used later)
- Boundary mishandling (empty arrays, single-element edge cases)
- Silent type coercion that changes meaning (int vs float, list vs array)
- Regressions: if reviewing a diff, check whether changed code breaks
  existing callers, contracts, or assumptions documented in docstrings

## 2. Performance

Look for:

- Unnecessary copies of large arrays (`np.array(x)` when `x` is already an
  ndarray, `.copy()` without reason)
- Repeated expensive computations inside loops that could be hoisted
- O(n^2) or worse patterns where O(n) or O(n log n) alternatives exist
  (e.g., nested list scans, repeated `in` checks on lists instead of sets)
- Allocation inside tight loops (creating arrays/lists per iteration when
  pre-allocation or vectorization would work)
- Unvectorized element-wise loops over numpy arrays

Report as WARN only for patterns with measurable impact in typical use
(not micro-optimizations).

## 3. Bare / broad exception handling

Find `except Exception`, `except BaseException`, and bare `except:`.
Flag any that silently swallow errors (no re-raise, no logging).
Acceptable: top-level CLI entry points, describe/repr methods guarding display.

## 4. God classes / long methods

Flag any single method or function longer than 200 lines (excluding docstring
and blank lines). Flag any class with more than 30 public methods.
Report as INFO with a note on whether the size is justified.

## 5. Dead code

Search for:

- Commented-out code blocks (3+ consecutive commented lines that look like code)
- Unreachable code after `return`/`raise`/`break`/`continue`

## 6. Typing / modern Python syntax

Search for:

- `Optional[X]` or `Union[X, Y]` where `X | Y` should be used instead
  (project style, Python 3.12+)

## 7. Docstring coverage

Check all public functions, methods, and classes in `src/` for:

- Missing docstrings entirely
- Docstrings that are not NumPy-style (missing Parameters/Returns sections)

User-facing API (`trspecfit.py`, `functions/`) should have extensive docstrings
with Parameters, Returns, and Notes. Internal modules can keep method docstrings
minimal but must have module and class-level docstrings.
Report count and list the worst offenders.

## 8. Abstractions and duplication

Read `src/` modules looking for:

- Poor or missing abstractions — repeated patterns that should be extracted
  into a shared helper, base class method, or utility
- Copy-pasted logic across files or classes (similar blocks of 5+ lines)
- Overly concrete code that handles special cases inline instead of through
  polymorphism or configuration

Suggest specific refactorings where the improvement is clear. Ignore test files.
Note: repeated blocks that extract *different subsets* of fields from a config
object are not duplication — a generic helper would just move the per-site
variation elsewhere without reducing complexity.

## 9. Numpy anti-patterns

Search for:

- `np.concatenate` where `np.where` would suffice (piecewise assignment)
- `np.shape(x)` instead of `x.shape`
- `np.ones(np.shape(...))` instead of `np.ones_like(...)` / `np.full_like(...)`
- `np.arange(0, n, 1)` instead of `np.arange(n)`
- Manual loops over array elements that could be vectorized

## 10. Fragile array comparisons

Flag `==` or `!=` on floating-point arrays/values from computation.
Should use `np.isclose`, `np.allclose`, or tolerance-based comparison.
Ignore comparisons against integer values or input validation checks.

## 11. Ignored warnings

Search for `warnings.filterwarnings("ignore")` or
`warnings.simplefilter("ignore")`.
These can mask real issues. Flag unless scoped to a specific known warning.

## 12. Plotting mixed with logic

Flag functions that both compute results AND produce plots without a
`show_plot`/`debug` flag to disable the plotting side. Plotting should be
separable from computation. Also check functions that *do* have a display
flag (e.g. `show_output`, `save_img`) but contain plot blocks that bypass
it — plots should save-and-close (not display) when the flag is off.

## 13. Poor separation of concerns

Check for:

- Circular imports (A imports B imports A)
- Business logic in UI/plotting code
- I/O operations (file read/write) mixed into computation functions

## 14. Missing `__repr__` / `__str__`

Check main domain classes for a `__repr__` method. Classes that appear in
debugging output or collections should be representable. Report as INFO.

## 15. Global mutable state

Search for:

- `global` keyword usage
- Module-level mutable containers (`dict`, `list`, `set`) that are modified
  at runtime (not just defined as constants)
- Module-level variables reassigned after import time

## 16. Security

Search for:

- `eval()` or `exec()` usage
- `pickle.load()` / `pickle.loads()` on potentially untrusted data
- `yaml.load()` without `Loader=SafeLoader` (or not using `yaml.safe_load()`)
- Hardcoded file paths or credentials
- `subprocess` calls with `shell=True`
- `os.system()` calls

## 17. Missing edge-case tests

Scan test files for coverage of:

- Empty/None inputs to public API functions
- Boundary values (zero-length arrays, single-element arrays)
- Invalid YAML/config inputs
- NaN/Inf in numeric inputs

Report as INFO with suggestions for what to add.

## Summary

Print a summary table:

| # | Check | Status | Issues |
|---|-------|--------|--------|
| 1 | Bugs & correctness | ... | ... |
| 2 | Performance | ... | ... |
| 3 | Broad exceptions | ... | ... |
| 4 | God classes / long methods | ... | ... |
| 5 | Dead code | ... | ... |
| 6 | Typing / modern Python syntax | ... | ... |
| 7 | Docstring coverage | ... | ... |
| 8 | Abstractions & duplication | ... | ... |
| 9 | Numpy anti-patterns | ... | ... |
| 10 | Fragile array comparisons | ... | ... |
| 11 | Ignored warnings | ... | ... |
| 12 | Plotting mixed with logic | ... | ... |
| 13 | Separation of concerns | ... | ... |
| 14 | Missing __repr__ | ... | ... |
| 15 | Global mutable state | ... | ... |
| 16 | Security | ... | ... |
| 17 | Edge-case tests | ... | ... |

Then list any FAIL or WARN items with actionable next steps.
