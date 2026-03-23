---
name: add-function
description: Add a new spectral/time/profile function with tests and YAML config. Use after writing the function implementation.
argument-hint: <function_name> <module>
disable-model-invocation: true
allowed-tools: Read, Write, Edit, Bash, Grep, Glob, Agent
---

Add a new function to the trspecfit library. The user has already written the
function implementation in the appropriate module. This skill validates it,
creates the YAML test model entry, and generates the standard test class.

## Arguments

- `$ARGUMENTS` format: `<function_name> <module>`
- `function_name`: the Python function name (e.g. `GLP`, `expFun`, `pGauss`, `lorentzCONV`)
- `module`: one of `energy`, `time`, `profile`

## Module mapping

| module     | source file                          | YAML file                           |
|------------|--------------------------------------|-------------------------------------|
| `energy`   | `src/trspecfit/functions/energy.py`   | `tests/test_models_energy.yaml`     |
| `time`     | `src/trspecfit/functions/time.py`     | `tests/test_models_time.yaml`       |
| `profile`  | `src/trspecfit/functions/profile.py`  | `tests/test_models_profile.yaml`    |

Test file mapping:
- Energy functions: `tests/test_functions_energy.py`
- Time dynamics functions: `tests/test_functions_time.py`
- Time convolution kernels (`*CONV`): `tests/test_functions_convolution.py`
- Profile functions: `tests/test_functions_profile.py`

## Steps

### 1. Check for duplicates

Read all existing functions in the same module. Compare the new function's
formula against existing ones. Flag if another function computes an
equivalent or near-equivalent result (e.g. same shape with different
parameterization). If a duplicate is found, stop and ask the user whether
to proceed or reuse the existing function.

### 2. Validate the function

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

Report any issues and fix them before proceeding.

### 3. Add YAML test model entry (skip for convolution kernels)

Add an entry to the appropriate YAML file. Use the function's parameters with
reasonable test values. Follow the existing patterns in the YAML file.

**Energy functions**: create a model named `single_<lowercase_name>` with one
component using the new function.

**Time functions**: create a model with the standard IRF + the new function.

**Profile functions**: create a model named `profile_<name>`.

Look at the existing entries in the YAML file for the exact format.

### 4. Generate the test class

Add a test class to the appropriate test file. The class name is
`Test<FunctionName>`. Follow the testing rules in `CLAUDE.md`.

Include all applicable tests from the checklist below.

#### Test checklist by function type

**Energy functions** (peak-like: has A, x0 parameters):

```
test_peak_at_center        -- result at x0 equals A (analytical)
test_zero_amplitude        -- A=0 returns all zeros, no NaN
test_zero_width            -- width param -> 0 returns finite, no NaN/Inf
test_symmetry              -- symmetric about x0 (if applicable)
test_fwhm                  -- independent FWHM check (not formula reimplementation!)
test_pure_limit            -- if mixing param exists, check pure Gaussian/Lorentzian limits
```

**Energy functions** (background-like: Offset, Shirley, LinBack):

```
test_basic_shape           -- output has expected shape and is finite
test_analytical_value      -- check against known analytical value
test_zero_params           -- zero parameters produce expected output
```

**Time dynamics functions**:

```
test_value_at_t0           -- check value at t=t0 (analytical)
test_asymptotic_behavior   -- check long-time limit (analytical)
test_zero_amplitude        -- A=0 returns all zeros
test_monotonicity          -- monotonically increasing/decreasing where expected
```

**Convolution kernels** (`*CONV` suffix):

```
test_peak_at_center        -- peak is at x=0
test_peak_value_is_one     -- peak value equals 1.0
test_half_max_at_half_width -- FWHM check using independent property, NOT formula reimplementation
test_zero_for_negative_x   -- (causal kernels only: expDecayCONV)
test_zero_for_positive_x   -- (anti-causal kernels only: expRiseCONV)
test_decays_monotonically  -- monotonic decay away from peak
test_kernel_width_positive -- companion kernel_width() > 0
```

**Profile functions**:

```
test_value_at_zero         -- check pFunc(0, ...) analytically
test_monotonicity          -- if monotonic, verify with np.diff
test_zero_params           -- zero amplitude/slope returns zeros
test_shape_matches_aux     -- output shape matches aux_axis
```

#### Critical test rule

**Never reimplement the formula** being tested as the expected value.
Use independent analytical properties instead (peak=A, FWHM=W, symmetry,
limiting cases, monotonicity).

### 5. Run tests

Run the test file to verify all new tests pass, then run the full suite:

```bash
pytest tests/<test_file>.py -q
pytest -q
```

### 6. Summary

Print a summary of what was created:

```
Function:  <name> in <module>
YAML:      <yaml_file> -- model "<model_name>"
Tests:     <test_file> -- class Test<Name> (N tests)
Status:    All tests passing (M total)
```
