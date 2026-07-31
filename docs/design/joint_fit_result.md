---
orphan: true
---

# First-class joint fit results

Status: **implemented** on branch `joint-fit-result` (companion #3 remains open).

This records the semantic and public-API decisions behind the in-memory result
of `Project.fit_2d`; mechanisms belong in code and exact assertions in tests.
Schema 7 consumes this result; it does not shape it.

## What's broken

`Project.fit_2d` runs one optimization over N files, then discards everything
belonging to the optimization as a whole. The combined `FitOutput` lands on
`Project._project_fit_result` (`trspecfit.py:1507`) and the next project fit
overwrites it. A user who runs a project fit with `mc_settings=MC(use_mc=1)`
gets a joint posterior computed and thrown away.

The N per-file 2D slots survive and keep the per-file curves, residuals,
selections, and noise metadata. They cannot hold the combined parameter table,
the sharing map, joint uncertainty, whole-objective metrics, or the joint chain.

## Decisions

### One optimization, one joint result, N projections

Every successful `Project.fit_2d` produces one `JointFitResult` and one
projection slot per participating file. Together they are one bundle: the record
is built and validated before either history is appended, so a failure publishes
neither. This guarantee is deliberately limited to publication — the existing
evaluator mutates live model state during evaluation, and this branch adds no
live-state restoration transaction.

The one-file case is still a joint result: it came through the project-scoped
path.

The record holds the projection slot objects, not names or history positions.
That is what makes the bundle self-contained and schema 7's write step
mechanical.

### Shared and projected payloads are disjoint

The joint result owns the combined parameter table, the parameter map, joint
`conf_ci`, `correl`, `mcmc`, whole-objective metrics, and optimizer provenance.
Projections own one file's observed/fitted arrays, projected local parameter
values, selection, noise metadata, and per-file residual metrics.

Projections never carry `conf_ci`, `correl`, or `mcmc`. Duplicating a posterior
creates several independently mutable claims about one distribution. They are
already absent on the project path today, so nothing is being removed.

Capture follows the repository-wide
[fit-to-slot capture boundary](repo_architecture.md). No
`lmfit.MinimizerResult`, `lmfit.Parameters`, `File`, or `Model` reference
survives into the record; frames and arrays are copied at capture and accessors
hand out copies.

### The parameter map is data, not a naming convention

Combined names encode position: project-shared parameters are unprefixed,
everything else becomes `file{idx:02d}_{local}` (`trspecfit.py:1156`, `:1202`).
Decoding that convention downstream is fragile and unnecessary —
`_build_fit_params` already produces the exact relation as
`list[(combined_name, file_idx, local_name)]` (`:1129`), covering project-shared,
file-varying, and static parameters alike, and `fit_2d` already uses it to
distribute values (`:1466`).

Each projection therefore stores `Mapping[str, str]` (combined → local), total
in both directions. Readers look up; they never parse. This also keeps `fileNN_`
out of the archive contract, superseding the prefix representation in the
current schema-7 draft.

### Object model

`JointFitResult` is the semantic type in memory and after a future archive
load — schema 7 adds no parallel `SavedJointFit`. It subclasses nothing: a slot
is a file result, a joint result owns one shared optimization and references its
projections, and their metadata overlap is too thin to justify a hierarchy.

```python
@dataclass(frozen=True)
class JointFitProjection:
    parameter_map: Mapping[str, str]   # combined name -> local name
    slot: SavedFitSlot


@dataclass(frozen=True)
class JointFitResult:
    model_name: str
    projections: tuple[JointFitProjection, ...]
    params: pd.DataFrame
    metrics: Mapping[str, float]
    fit_alg: str
    fit_settings: Mapping[str, Any]
    timestamp: str
    conf_ci: pd.DataFrame | None = None
    correl: pd.DataFrame | None = None
    mcmc: MCMCResult | None = None
```

A projection carries no `file_name` of its own: `slot.file_name` is already the
captured identity, and a second copy could disagree with it. `files` derives
from the projections, which are stored in canonical file-name order; the map,
not tuple position, carries the association with optimizer parameters.
`model_name` lives on the record rather than per projection — `Project.fit_2d`
enforces one common name (`trspecfit.py:1373-1375`), and adding the field later
is trivial.

### `params` is the authoritative parameter table

Built from `FitOutput.par_fin.params` via `par_to_df(..., col_type="min")`, in
optimizer order. Per-file parameter tables are materialized projections for
file-oriented plotting and comparison.

`init_value` is the effective optimizer-entry value — after baseline-result
injection, project-sharing resolution, and expression evaluation — not the value
authored in YAML. No second initial-parameter table is stored: lmfit populates
`init_value` for varying, static, and expression parameters alike, and
`fit_wrapper` restores it after a two-stage fit, so the combined table already
carries it. When shared seeds diverge across files, `_build_fit_params` warns
and uses the first file's value; that warning is the only record of the rejected
candidates.

### Uncertainty: correlation, not covariance

The record stores `correl`; `stderr` in `params` recovers covariance as
`correl(i,j) * stderr(i) * stderr(j)`. Storing both invites disagreement.
`correl` is `None` when the optimizer produced no covariance.

`mcmc` is one widened `MCMCResult` shared by the slot and joint paths, built by
a single helper and carrying `lnsigma`. Its `__lnsigma` is a single nuisance
scale over the concatenated residual: never a per-file σ, never back-filled into
a projection, never used to calibrate metrics.

### Metric ownership

| Metric | Independent slot | Joint projection | Joint result |
|---|---|---|---|
| `chi2_raw` | file residual | file residual | whole objective |
| `chi2_red_raw` | local DoF | undefined | total data minus joint varying params |
| `chi2` | file σ | file σ | Σ projection `chi2`; `NaN` unless every σ valid |
| `chi2_red` | local DoF and σ | undefined | joint `chi2` over joint DoF |
| `r2` | file-local mean | file-local mean | undefined |
| `aic`, `bic` | file objective | undefined | whole objective |

The four projection metrics are undefined because they depend on the parameter
count, and the joint count does not decompose by file. Joint raw metrics come
from `compute_fit_metrics` over the concatenated prediction, not lmfit's
`MinimizerResult.aic`/`.bic`, so joint and per-file numbers stay comparable by
construction. A joint `r2` would depend on an arbitrary global mean across
separate measurements. `chi2` sums cleanly across heterogeneous per-file noise
scales, so no representative joint `sigma_eff` is invented. These are post-fit
diagnostics; the optimization objective stays unweighted.

In `compare_models`, structurally undefined cells are `NaN`. A column is dropped
only when every matched slot lacks the metric — one joint projection must not
suppress valid values in the other rows.

### Return value and query surface

`Project.fit_2d` returns the captured record and appends that same object to
history. It is the one fitting entry point whose result is not reachable from a
single `File`, so requiring `results.get_joint(...)` would make the caller
re-specify what they just fit. Ignoring the return value stays valid.

```python
self._joint_fit_history: list[JointFitResult] = []   # replaces _project_fit_result
```

`Project.results` snapshots both histories. `FitResults.__iter__` and `len()`
stay per-file — mixing joint records in would count one optimization N+1 times.
Joint records get `find_joint()` / `get_joint()`, with `get_joint` raising on
zero or multiple matches like `FitResults.get`. `files=` is canonicalized and
matches the full participant set.

`FitResults.plot_joint_mcmc()` renders the joint chain through the same
primitive as `plot_mcmc`. A joint projection holds no `mcmc` payload, so without
it there is no path to the joint posterior at all.

File-level accessors stay projection-oriented; joint uncertainty comes from the
joint record. The per-file `model.result` placeholders remain compatibility
views, and their lack of joint uncertainty is intentional.

## Out of scope

Projected `par_ini` (giving project-fitted files a `fit_ini` curve) and
single-evaluation splitting ship together in a later branch: enabling the first
without the second adds an interpreter pass per file. A live-state restoration
transaction is also out of scope — the interpreter joint evaluator mutates model
parameters on every objective call (`spectra.py:402-404`), which is why the
atomicity guarantee above covers publication only.

## Companion changes

Each is broken on `main` today, independent of joint results, and gets its own
commit. #1, #2, #4, and #5 landed with this branch; #3 remains open:

1. Joint projections carry meaningless `aic`/`bic`/`chi2_red*`, because the
   per-file `MinimizerResult` carries the joint `nvarys`
   (`trspecfit.py:1490-1494`, reaching `compute_fit_metrics` at `:3748`). **The
   metric table above is invalid until this lands.**
2. `get_mcmc` drops the persisted `lnsigma` (`fit_io.py:564-574`;
   `fit_results.py:721-730`) — lossy for every fit type.
3. Four call sites hand-assemble slot metadata (`trspecfit.py:3378`, `:3490`,
   `:3632`, `:3738`); the shared helper the architecture doc calls for does not
   exist yet.
4. Four call sites duplicate the `covar is not None` correlation guard
   (`:3372`, `:3484`, `:3620`, `:3732`); one shared helper should own it.
5. `compare_models` documents `chi2_red_raw` as always populated
   (`fit_results.py:1195-1198`).

## Boundary with schema 7

This branch adds no archive identity, HDF5 groups, collision or pruning
behavior, `model_structure`, or version stamps. Schema 7 later derives
`optimization_hash` from the captured record, writes one joint sidecar plus the
N projection slots, persists the parameter map, and reconstructs the same
`JointFitResult` type on read. The archive writer never consults live model
state.

Schema 6 gains nothing: it exposes joint records in-session, keeps save/export
on projection slots, and cannot reconstruct a joint record on load. That is a
temporary boundary between stacked branches, not a compatibility promise.

## Verification themes

Most assertions follow from the decisions above. Four do not:

- **Bundle integrity** — one record plus one projection per file, for two-file
  and one-file fits alike; a capture failure publishes neither history.
- **Snapshot isolation** — repeated fits append rather than overwrite, and an
  older `Project.results` does not see a later joint fit.
- **Capture is a snapshot** — mutating or replacing a live model after the fit
  does not change the record.
- **Backend equivalence** — interpreter and fused-JAX project fits produce the
  same record shape and the same projections.
