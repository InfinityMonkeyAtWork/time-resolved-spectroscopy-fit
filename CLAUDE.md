# General behavior

Do not make any changes until you have 95% confidence in what you need to build. Ask me follow-up questions until you reach that confidence.

# Code style

- Add an empty line after each docstring.
- Add `#\n` before function/method definitions.
- Add `#\n#\n` before class definitions.
- Python 3.12+: use `X | Y` union syntax, not `Optional[X]` or `Union[X, Y]`.
- Ruff is the formatter/linter. Line length 88.
- Numpy: prefer `np.where` over `np.concatenate` for piecewise functions;
  use `x.shape` not `np.shape(x)`.
- Function calls: use keyword arguments for all parameters except the
  primary data object, e.g. `func(data, value=10)` not `func(data, 10)`.
  Applies to all types (`bool`, `int`, `float`, `str`).
- Function signatures: use `*` to enforce keyword-only arguments for any
  parameter that isn't the primary data object (the "subject").
  E.g. `def func(data, *, threshold=0.5, normalize=True)`.
  Exception: function registry definitions in `src/trspecfit/functions/`
  keep positional parameter signatures because parsing/introspection depends
  on their ordered parameter lists.
- **Naming**: snake_case everywhere — variables, functions, methods, parameters,
  attributes. Exception: function registry names (`GLP`, `pExpDecay`, `LinBack`,
  `gaussCONV`, etc.) and their parameters (`A`, `x0`, `SD`, `xStart`, `xStop`)
  use camelCase/PascalCase because `_` is the component ID delimiter
  (`{model}_{component}_{param}`). Underscores in those names would break parsing.

# Testing

- Plain pytest classes, no `unittest.TestCase`, no `@pytest.fixture`, no `setUp`/`setup_method`.
  Shared setup lives in explicit helper methods called by each test:
  module-level axis factories (`make_energy_axis`, `make_time_axis`, `make_kernel_axis`,
  `make_aux_axis`) and class-level private builders/loaders (`_make_file_...`, `_load_..._model`,
  `_make_2d_model`, etc.). Name helpers by intent, not lifecycle.
- Test YAML files live in `tests/` (e.g. `test_models_energy.yaml`).
- Always suppress plot display in tests: pass `show_plot=False` where available,
  or `save_img=-2` for methods that use `save_img` instead (e.g. `Component.plot`,
  `Simulator.plot_comparison`).
- **Use the public API** (`Project`, `File.load_model`, `File.add_time_dependence`,
  `File.add_par_profile`, `File.set_fit_limits`, etc.) in tests — not internal
  constructors or private methods. Tests that bypass the public API can mask real
  bugs by skipping validation, axis propagation, or setup steps that users hit.
  Exception: pure-math unit tests (e.g. testing a numerical function directly) and
  tests that intentionally verify internal invariants.
- When `assert x is not None` narrows an `X | None` type so subsequent code
  can access attributes, add a `# type guard` comment. Leave it unlabeled
  when the assertion is the actual test (e.g. verifying a method populates a field).
- **Test variable naming**: when a local variable holds or derives from a
  function registry parameter or component name, keep the original casing
  (e.g. `SD = 2.0`, `c_Shirley = Component("Shirley")`, `A1 = ...`).
  Name derived variables as `{par}_{qualifier}`: `A_early`, `A_mid`,
  `x0_late`, `mean_A`, not `early_A` or `a_early`.
- Run tests: `pytest -q`

# Renaming / API changes

- On renames, grep the entire repo — notebooks, YAML, tests, and docs all reference the public API.

# Documentation

- Keep `README.md` minimal — overview and quick-start only.
- Detailed docs belong in the Sphinx/Read the Docs API reference (`docs/`).
- Treat `docs/design/model_design_rules.md` as the source of truth for currently
  supported model combinations, expression semantics, and explicitly excluded
  cases. When reviewing code or proposing features, distinguish clearly between
  real support gaps and cases the design rules intentionally do not require us
  to support.
- NumPy-style docstrings everywhere. User-facing API (`trspecfit.py`, `functions/`) gets
  extensive docstrings (Parameters, Returns, Notes with physical context). 
  Internal modules (`mcp.py`, `config/`, `fitlib.py`) keep method docstrings minimal —
  focus detail on module and class level.
