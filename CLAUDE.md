# Code style

- Add an empty line after each docstring.
- Add `#\n` before function/method definitions.
- Add `#\n#\n` before class definitions.
- Python 3.12+: use `X | Y` union syntax, not `Optional[X]` or `Union[X, Y]`.
- Ruff is the formatter/linter. Line length 88.

# Testing

- Use `setUp` method (not `@pytest.fixture`) for setup tasks shared between multiple tests.
- Test YAML files live in `tests/` (e.g. `test_models_energy.yaml`).
- Always pass `show_plot=False` and `debug=False` in test calls.
- Run tests: `pytest -q`

# Documentation

- Keep `README.md` minimal — overview and quick-start only.
- Detailed docs belong in the Sphinx/Read the Docs API reference (`docs/`).
- NumPy-style docstrings everywhere. User-facing API (`trspecfit.py`, `functions/`) gets
  extensive docstrings (Parameters, Returns, Notes with physical context). Internal modules
  (`mcp.py`, `config/`, `fitlib.py`) keep method docstrings minimal — focus detail on
  module and class level.
