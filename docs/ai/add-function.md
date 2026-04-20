# Add Function

Shared source of truth for validating a newly added spectral, time, or profile
function and generating the standard test class. Use this after the function
implementation already exists in the appropriate source module.

This guide covers both:

- making the function available on the MCP/reference path
- checking whether additional GIR work is needed for compiled execution

## Arguments

- Format: `<function_name> <module>`
- `function_name`: the Python function name
  (e.g. `GLP`, `expFun`, `pGauss`, `lorentzCONV`)
- `module`: one of `energy`, `time`, `profile`

## Module mapping

| module | source file |
|--------|-------------|
| `energy` | `src/trspecfit/functions/energy.py` |
| `time` | `src/trspecfit/functions/time.py` |
| `profile` | `src/trspecfit/functions/profile.py` |

Test file mapping (note: CONV kernels live in `time.py` but test in a separate
file):

- Energy functions -> `tests/test_functions_energy.py`
- Time dynamics functions -> `tests/test_functions_time.py`
- Time convolution kernels (`*CONV`) -> `tests/test_functions_convolution.py`
- Profile functions -> `tests/test_functions_profile.py`

When routing to the test file, check the function name first: if it ends with
`CONV`, always use `test_functions_convolution.py` regardless of module.

## 1. Check for duplicates

Read all existing functions in the same module. Compare the new function's
formula against existing ones. Flag if another function computes an equivalent
or near-equivalent result (e.g. same shape with different parameterization).
If a duplicate is found, stop and ask the user whether to proceed or reuse the
existing function.

## 2. Validate the function

Read the source file and find the function. Verify:

- [ ] Function exists in the expected module
- [ ] Has NumPy-style docstring (Parameters, Returns, Notes if physics context)
- [ ] Empty line after docstring
- [ ] `#\n` before function definition
- [ ] No underscores in function name (enforced by guard test)
- [ ] Profile functions start with `p` prefix
- [ ] Convolution kernels end with `CONV` suffix and have a companion
      `<name>_kernel_width()` function returning an int

Note: these functions do NOT use `*` for keyword-only args because the
framework calls them via `self.fct(x, **parameters)`.

Additionally, verify the function's signature matches its registration:

- [ ] If the last parameter is `spectrum` -> it's a background function ->
      verify it is listed in `background_functions()` in
      `src/trspecfit/config/functions.py`. If missing, flag it.
- [ ] If the last parameter is NOT `spectrum` -> verify it is NOT listed
      in `background_functions()`. A mismatch means either the signature
      or the registration is wrong -> flag it.

Registration notes:

- Background functions are manually listed in
  `src/trspecfit/config/functions.py::background_functions()`.
- Convolution kernels are discovered dynamically by their `CONV` suffix in
  `config/functions.py`; they do not require a separate manual registry there.

Module-specific guidance:

- For time dynamics functions in `src/trspecfit/functions/time.py` that are
  defined as zero before `t0` and active afterward, prefer `np.where` over
  `np.concatenate` for the piecewise definition.

Finally, verify the function is discoverable by the framework:

- [ ] Function name does NOT start with `_` (the framework discovers functions
      via `dir(module)` filtering out private names in
      `config/functions.py::all_functions()`)
- [ ] No other callable with the same name exists in a different module
      (would cause ambiguity in function lookup)

Report any issues and fix them before proceeding.

## 2.5. Make GIR support the default when feasible

Adding a function to `functions/*.py` makes it available to the MCP/reference
path, but that does **not** automatically make it available to the compiled
GIR path.

Default policy:

- [ ] Try to make the new function work on GIR when it fits the current
      compiled architecture.
- [ ] If the function does not fit cleanly, fall back to MCP intentionally and
      explain why GIR support failed.
- [ ] When GIR support fails, give at least one concrete recommendation for how
      the function signature or behavior could be adjusted to make GIR support
      practical in a follow-up change.

Use MCP-only / fallback-only as the exception, not the default.

When attempting GIR support, check the relevant pieces below.

Energy functions:

- [ ] Add lowering support in `src/trspecfit/graph_ir.py` if the function needs
      a new opcode or name mapping.
- [ ] Update the compiled evaluator path if the new opcode needs explicit
      runtime dispatch or special handling.
- [ ] Verify `can_lower_1d()` / `can_lower_2d()` behavior is still correct for
      the new function.
- [ ] If lowering fails, explain whether the blocker is the function shape,
      unsupported argument pattern, or missing opcode/dispatch support.

Time dynamics functions:

- [ ] Add the function to the dynamics enum/name mapping in
      `src/trspecfit/graph_ir.py`.
- [ ] Add runtime dispatch in `src/trspecfit/eval_2d.py`
      (`DYNAMICS_DISPATCH`).
- [ ] Verify the parameter count matches between lowering and evaluator
      dispatch.
- [ ] Verify `can_lower_2d()` accepts the new function where intended.
- [ ] If lowering fails, recommend a signature that matches the compiled
      dynamics pattern (typically `func(t, par1, ..., t0, y0)` with vectorized
      NumPy math and no extra runtime context).

Convolution kernels:

- [ ] Add the kernel to the convolution enum/name mapping in
      `src/trspecfit/graph_ir.py`.
- [ ] Add runtime dispatch in `src/trspecfit/eval_2d.py`
      (`CONV_KERNEL_DISPATCH`).
- [ ] Verify kernel-width handling still works with the new kernel.
- [ ] Verify unsupported kernels still fall back cleanly to MCP.
- [ ] If lowering fails, recommend a kernel signature and behavior compatible
      with the current compiled path (pure kernel function, explicit parameter
      list, companion `<name>_kernel_width()`, no hidden state).

Profile functions:

- [ ] Add the function to the profile enum/name mapping in
      `src/trspecfit/graph_ir.py`.
- [ ] Verify the lowered 1D/2D profile evaluators can execute it without extra
      special-casing.
- [ ] Verify `can_lower_1d()` / `can_lower_2d()` behavior is still correct for
      profiled models using the new function.
- [ ] If lowering fails, recommend a vectorized signature that matches the
      current profile model (`func(x, par1, ...)` returning broadcast-friendly
      NumPy arrays).

Before finishing, explicitly state one of:

- `Function is available on MCP only; GIR falls back intentionally.`
- `Function is available on both MCP and GIR paths.`

If you report MCP-only / fallback-only, also include:

- the specific GIR blocker
- one recommended signature or behavior change that would make future GIR
  support easier

## 3. Generate the test class

Add a test class to the appropriate test file. The class name is
`Test<FunctionName>`. Follow the testing rules in `CLAUDE.md`.

Include all applicable tests from the checklist below.

### Test checklist by function type

**Energy functions** (peak-like: has `A`, `x0` parameters):

```text
test_peak_at_center         -- result at x0 equals A (analytical)
test_zero_amplitude         -- A=0 returns all zeros, no NaN
test_zero_width             -- width param -> 0 returns finite, no NaN/Inf
test_symmetry               -- symmetric about x0 (if applicable)
test_fwhm                   -- independent FWHM check (not formula reimplementation!)
test_pure_limit             -- if mixing param exists, check pure Gaussian/Lorentzian limits
```

Not all peak-like tests apply to every function. Skip tests that don't fit:

- **Asymmetric lineshapes** (e.g. DS): `A` may be a scaling factor, not the
  peak value -> skip `test_peak_at_center`. Replace `test_symmetry` with
  `test_asymmetry`. `test_fwhm` may not have a clean closed form -> skip
  unless it does. Add `test_peak_near_center` (`argmax ~= x0`) instead.

**Energy functions** (background-like: `Offset`, `Shirley`, `LinBack`):

```text
test_basic_shape            -- output has expected shape and is finite
test_analytical_value       -- check against known analytical value
test_zero_params            -- zero parameters produce expected output
```

**Time dynamics functions**:

```text
test_value_at_t0            -- check value at t=t0 (analytical)
test_asymptotic_behavior    -- check long-time limit (analytical)
test_zero_amplitude         -- A=0 returns all zeros
test_monotonicity           -- monotonically increasing/decreasing where expected
```

**Convolution kernels** (`*CONV` suffix):

```text
test_peak_at_center         -- peak is at x=0
test_peak_value_is_one      -- peak value equals 1.0
test_half_max_at_half_width -- FWHM check using independent property, NOT formula reimplementation
test_zero_for_negative_x    -- (causal kernels only: expDecayCONV)
test_zero_for_positive_x    -- (anti-causal kernels only: expRiseCONV)
test_decays_monotonically   -- monotonic decay away from peak
test_kernel_width_positive  -- companion kernel_width() > 0
```

**Profile functions**:

```text
test_value_at_zero          -- check pFunc(0, ...) analytically
test_monotonicity           -- if monotonic, verify with np.diff
test_zero_params            -- zero amplitude/slope returns zeros
```

### Critical test rule

**Never reimplement the formula** being tested as the expected value.
Use independent analytical properties instead (peak=A, FWHM=W, symmetry,
limiting cases, monotonicity).

For **pure-limit / cross-check tests** (e.g. `GLS(m=1)` vs `Lorentz`), the
parameter mapping between functions may not be 1:1 (e.g. DS `F` maps to Lorentz
`W=2F`). Verify the mapping numerically before writing the test.

## 4. Review and consolidate

After adding tests, review ALL tests in the class (both pre-existing and new)
against these quality criteria:

**Remove** tests that:

- Reimplement the formula as the expected value (e.g. computing
  `A * exp(-x**2 / (2*SD**2))` and comparing — that just mirrors the source)
- Duplicate another test's assertion (e.g. two tests both checking peak = A
  at `x0` with different parameter values — keep one with the most general
  params)
- Test the same property as another test but with a trivial parameter change
  (e.g. `test_peak_value_m0` and `test_peak_value_nonzero_m` — one
  `test_peak_at_center` with `m != 0` covers both)

**Keep** tests that verify independent analytical properties:

- Peak value, FWHM, symmetry, monotonicity, limiting cases, zero params
- Pure-limit tests that compare against *another function in the library*
  (e.g. `GLS(m=1)` vs `Lorentz`) — these are cross-checks, not
  reimplementations

**Target**: one test per independent property, no redundancy. Aim for the
checklist count (typically 4-7 tests per function).

## 5. Run tests

Run the test file to verify all new tests pass, then run the full suite:

```bash
pytest tests/<test_file>.py -q
pytest -q
```

## 6. Summary

Print a summary of what was created:

```text
Function:  <name> in <module>
Tests:     <test_file> -- class Test<Name> (N tests)
Status:    All tests passing (M total)
```
