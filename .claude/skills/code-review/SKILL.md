---
name: code-review
description: Run a 20-point code-review checklist on the codebase (or a specified scope). Covers correctness, style, architecture, and test quality.
disable-model-invocation: true
allowed-tools: Bash, Read, Grep, Glob, Agent
---

Run every check below against the target scope. Default scope is all Python
source under `src/` and `tests/`. If the user supplies an argument (file, directory,
or glob), restrict the review to that scope.

For each item report one of:
- **PASS** — no issues found
- **INFO** — minor observations, no action needed
- **WARN** — issues found, list them with file and line number
- **FAIL** — serious issues found, list them with file and line number

Work through the checklist in order. Use parallel Agent/tool calls where items
are independent.

---

## 1. Mutable default arguments

Search for function/method definitions whose default values are mutable
(`[]`, `{}`, `set()`). Each should use `None` with an `if arg is None` guard.

```
pattern: def \w+\(.*[=]\s*(\[\]|\{\}|set\(\))
```

## 2. Bare / broad exception handling

Find `except Exception`, `except BaseException`, and bare `except:`.
Flag any that silently swallow errors (no re-raise, no logging).
Acceptable: top-level CLI entry points, describe/repr methods guarding display.

## 3. God classes / long methods

Flag any single method or function longer than 200 lines (excluding docstring
and blank lines). Flag any class with more than 30 public methods.
Report as INFO with a note on whether the size is justified.

## 4. Magic numbers

Search for bare numeric literals (not 0, 1, -1, 2) used in logic or
computation outside of test files. Flag any that lack an explanatory comment
or named constant. Ignore array indexing, range/shape arguments, and standard
mathematical constants (pi, 2*pi, etc.).

## 5. Inconsistent naming

Check for:
- camelCase or mixedCase variable/function names (should be snake_case per PEP 8)
- ALL_CAPS names that are not module-level constants
- Boolean parameters/variables not named with is_/has_/should_/can_ prefix (INFO only)
- Bare `0`/`1` used as boolean values in function calls

## 6. Dead code

Search for:
- Commented-out code blocks (3+ consecutive commented lines that look like code)
- Unreachable code after `return`/`raise`/`break`/`continue`
- Imports that are never referenced in the file
- `Optional[X]` or `Union[X, Y]` (should be `X | Y` per project style, Python 3.12+)

## 7. DRY violations

Look for repeated code blocks (5+ similar consecutive lines appearing in
multiple places). Suggest extraction into a helper if the duplication is
non-trivial. Ignore test files for this check.

## 8. Excessive nesting

Flag any code block nested 5+ levels deep (excluding class/def). Report the
deepest nesting level per function. Mark as WARN only if nesting harms
readability (not when it is structural, e.g., nested loops over axes).

## 9. String formatting inconsistency

All string interpolation should use f-strings. Flag `str.format()`,
`%`-formatting, and `+` concatenation used for building strings.
Ignore format-spec literals for numpy/matplotlib (e.g., `"%.2f"`).

## 10. Type hint gaps

Check all public functions and methods in `src/` for:
- Missing return type annotation
- Missing parameter type annotations (exclude `self`, `cls`)
Report count and list the worst offenders.

## 11. Numpy anti-patterns

Search for:
- `np.concatenate` where `np.where` would suffice (piecewise assignment)
- `np.shape(x)` instead of `x.shape`
- `np.ones(np.shape(...))` instead of `np.ones_like(...)` / `np.full_like(...)`
- `np.arange(0, n, 1)` instead of `np.arange(n)`
- Manual loops over array elements that could be vectorized

## 12. Fragile array comparisons

Flag `==` or `!=` on floating-point arrays/values from computation.
Should use `np.isclose`, `np.allclose`, or tolerance-based comparison.
Ignore comparisons against integer values or input validation checks.

## 13. Ignored warnings

Search for `warnings.filterwarnings("ignore")` or `warnings.simplefilter("ignore")`.
These can mask real issues. Flag unless scoped to a specific known warning.

## 14. Plotting mixed with logic

Flag functions that both compute results AND produce plots without a
`show_plot`/`debug` flag to disable the plotting side. Plotting should be
separable from computation.

## 15. Poor separation of concerns

Check for:
- Circular imports (A imports B imports A)
- Business logic in UI/plotting code
- I/O operations (file read/write) mixed into computation functions

## 16. Missing `__repr__` / `__str__`

Check main domain classes for a `__repr__` method. Classes that appear in
debugging output or collections should be representable. Report as INFO.

## 17. Overuse of isinstance

Flag `isinstance(self, ...)` in base class methods (base class knowing about
subclasses). Also flag long isinstance chains (4+ branches) that could use
a dispatch dict or polymorphism. Report as INFO if count is low.

## 18. Global mutable state

Search for:
- `global` keyword usage
- Module-level mutable containers (`dict`, `list`, `set`) that are modified
  at runtime (not just defined as constants)
- Module-level variables reassigned after import time

## 19. Missing edge-case tests

Scan test files for coverage of:
- Empty/None inputs to public API functions
- Boundary values (zero-length arrays, single-element arrays)
- Invalid YAML/config inputs
- NaN/Inf in numeric inputs
Report as INFO with suggestions for what to add.

## 20. Assertions without messages

**In production code (`src/`)**: flag any bare `assert` statement — these are
stripped by `python -O` and should be replaced with explicit `raise`.

**In test code (`tests/`)**: report count of assertions without messages as
INFO only (pytest introspection makes messages optional).

---

## Summary

Print a summary table:

| # | Check | Status | Issues |
|---|-------|--------|--------|
| 1 | Mutable defaults | ... | ... |
| 2 | Broad exceptions | ... | ... |
| ... | ... | ... | ... |
| 20 | Assert messages | ... | ... |

Then list any FAIL or WARN items with actionable next steps.
