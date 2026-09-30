---
orphan: true
---

# June 2026 Review — Triage

> Archived on 2026-09-30. The June 2026 review (a four-agent external review
> of the repository at about v0.9, 269 commits, dated 2026-06-10, kept outside
> the repo at `~/Documents/fable review 2026-06-10.txt`) triaged against
> `main` at v0.19.0. Every weakness is either retired with the evidence or
> carried forward to the `TODO.md` item that owns it; nothing new was
> scheduled from it. Line counts and test names are as of archival.

The rule applied (TODO item 3, 2026-09-24): internal assertions and
complicated indexing alone do not demonstrate silent numerical errors. A
finding is scheduled only with a current, reachable trigger; otherwise it is
retired, and the independent numerical checks (parameter recovery, evaluator
parity) remain the guard.

## Strengths, unchanged

The review's five strengths still describe the repo and are what the
September 2026 review asks to preserve: the roundtrip parameter-recovery
matrix (`tests/roundtrip/`), the two-layer authoring / compiled-evaluator
split with `spectra.py` as the bridge, the documentation and example
notebooks, the pre-commit and CI hygiene, and the explicit state in
`TODO.md`, `PLAN.md`, `CHANGELOG.md` and this archive.

## Weaknesses, triaged

| June finding | State at v0.19.0 | Disposition |
|---|---|---|
| `trspecfit.py` is a god module; `File` has about 46 methods | 5,666 lines; `File` has 46 public methods | Carried: TODO [14], consolidate internals as the areas change |
| `schedule_2d` is about 1,000 lines mixing lowering, profiles, expressions and scheduling | 646 lines; `_schedule_component_ops`, `_compile_profile_groups` and `_emit_profile_nodes` were split out | Retired as stated; the remainder is TODO [14] |
| Evaluators lean on `assert`, which vanishes under `python -O` | `eval_1d.py` has none; `eval_2d.py` has two, one a type guard; the scheduler in `graph_ir.py` keeps 19 contract checks | Retired under the rule: no reachable trigger shown |
| Precompute caching does intricate index arithmetic where an off-by-one yields wrong numbers silently | Interpreter / NumPy / JAX parity tests (`tests/test_gir_integration.py`, `tests/test_evaluate_2d.py`) and the recovery matrix are the independent check | Retired: parity is the guard, not a review of the indexing |
| `can_lower_2d` rejects a model without saying why | Still true: `can_lower_1d` / `can_lower_2d` / `can_lower_jax_2d` return a bare bool and a non-lowerable model falls through to the interpreter silently; only `_effective_backend` records what ran, in provenance | Carried: the reachable trigger for TODO [9], improve public validation errors |
| No coverage reporting or threshold | Measured on demand since 2026-09-30 (`pytest --cov`, 88.5% of lines over the full suite); a CI step, badge and threshold were declined because the percentage does not bear on the validity of fit results | Closed |
| No docs build in CI | Still true: only Read the Docs builds, with `fail_on_warning`, after a merge | Carried: TODO [6] |
| No test timeout; MCMC tests could hang CI | Job cap `timeout-minutes: 30` and `pytest-timeout` at 600 s per test, landed 2026-09-29 | Closed |
| Linux-only CI matrix | Accepted for now, 2026-09-30 | Closed; TODO [6] keeps the question of a reproducible dev/example environment |
| `simulator.py`, `spectra.py` and `fitlib.py` have no dedicated tests | `tests/test_fitlib.py`, `tests/test_simulator_noise.py`, `tests/test_evaluate_1d.py` / `_2d.py` and the evaluator integration harness exist; `spectra.py`, the bridge, is exercised through the parity harness | Retired |
| No stable / advanced / internal API tiers | Still open | Carried: TODO [5], already scheduled |
| `fit_model_mcp` returns `ndarray` or a list depending on a flag | Unchanged in shape, but typed with `@overload` on `plot_sum` | Retired; TODO [5] curation may still split it |
| `Model.result` is a positional list with hard-coded index meanings | `ulmfit.FitOutput` (or `None`) | Retired |
| `fit_baseline()` side-populates attributes later code assumes exist | Still sets `model_base` / `data_base`; the downstream requirement went away in v0.19.0 (`seed_source="model"`); what remains is the pre-fit surface | Carried: TODO [4], already scheduled |
| Recent removals shipped without a deprecation path | `docs/stability.md` (2026-07-12) states the pre-1.0 policy: no deprecation cycle, complete changelog, breaking changes only in minor releases | Retired |
| Dependencies fully unpinned, no lock file | Lower bounds in `pyproject.toml`, a min-versions CI job, exact-pinned dev extras and SHA-pinned actions (2026-07-01) | Retired |
| About 280 deferred pyright errors in tests | Six Optional-driven rules are off for `tests/` with a comment explaining why | Carried: TODO [6] re-checks the suppressions and their comment |
| 100% of commits from one author | 520 of 520 | Noted; not a repository action. Outreach is step 10 of the ordered plan |

The September 2026 review is triaged elsewhere: its ownership findings and
evidence are in [api_ownership_contract.md](../api_ownership_contract.md), its
work items are the numbered steps in `TODO.md`, and its behavioral-probe
method is check 21 of [code-review.md](../../ai/code-review.md).
