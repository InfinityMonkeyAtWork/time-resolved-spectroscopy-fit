# Active Plan — example fixes

33 real defects in `examples/fitting_workflows/`, found by running
`/check-example` over the full set. **None are fixed yet.** This branch carries
both the skill changes that found them and the fixes themselves, so the skill
can be re-run against the fixed notebooks as a regression check.

Every item below was verified against the committed files or an executed run,
not taken from an agent's summary. Where something is unverified, it says so.

Audited: **01, 02, 03, 04, 20, 21** (2026-09-12), **10, 11, 12** (2026-09-13).
Full set covered.

Clear this file per `CLAUDE.md` once the fixes land.

## Execution order (agreed 2026-09-13)

1. Rebase onto `main` — done.
2. Normalization commit — done: `scripts/normalize_notebooks.py` pins the
   kernel metadata (canonical `kernelspec`, no `language_info.version`); every
   notebook is nbformat 4.5 with cell ids (closes item 6).
3. Review the rebuilt skill (mechanics script + criteria doc); fix what falls
   out before using it as the regression gate.
4. Fix defects one commit per notebook: 01, 02, 04, 03, 20, 12, 11, 10, 21.
   Item 20 (`expFun` vocabulary across 01/04/21) is one cross-notebook commit.
   Re-verify item 15 before fixing it. After each notebook: mechanical
   pre-pass + executed audit, confirm its findings are gone.
5. Close out: one `[Unreleased]` CHANGELOG line, the two API gaps into
   `TODO.md`, clear this file (archive-or-changelog decision).

Decisions: item 4 gets a markdown note, no data regeneration. Criterion 7
requires Tips **and** Next Steps, so 10, 11 and 20 each get a Next Steps.

---

## Physics / claims wrong — highest priority

These are the ones that cost user trust, because a reader who checks them
finds the library's own example describing its own data backwards.

### 1. `01_basic_fitting` cell 16 — dynamics described backwards

Prose says `GLP_01_x0` is "rising after t = 0 and leveling off — an
exponential approach." Truth is `expFun A: 5` (positive), whose docstring
(`src/trspecfit/functions/time.py:166`) says *"Jumps to A at t0, decays toward
0"*. Committed data: peak 9.20 (t = −20) → 13.30 (t = 5) → 9.65 (t = 100).
Kick-then-relax, not rise-and-plateau.

**Fix:** one sentence.

### 2. `04_parameter_profiles` cell 4 — shift direction backwards

Prose: *"At t = 0 the peak jumps toward lower binding energy."* Measured:
argmax 98.97 → 99.42, centroid 98.59 → 99.09. It moves **higher**.

The same wrong sentence is in `data/generate_data.ipynb:151`, so the fix
touches data provenance too.

**Fix:** two sentences, two files.

### 3. `21_multi_file_shared_fit` cell 10 — truth attributed to the wrong file

Prose: ``Truth: shared `SD = 3`, `tau = 50`; per-file
`A = [5, 2, 1, 0.5, 0.2, 0.1]` (`data/models_time_truth.yaml`).``

That file contains, in full, `A: [5, True, 0, 15]` — a single value. The
per-file list is `amplitudes` in `data/generate_data.ipynb`. The citation is
correct for `SD` and `tau` and wrong for the third quantity it's attached to.

**Fix:** split the citation.

---

## Runs dirty

### 4. `03_multi_cycle_dynamics` cell 18 — unexplained `UserWarning`

The run prints the `mcp.py` knife-edge warning (*"7 time sample(s) … within a
relative tolerance of 1e-06 of a subcycle boundary"*). The word "boundary"
never appears in the notebook. `generate_data.ipynb` saves and reloads the
axes (half the recorded pitfall) but still builds `np.arange(-4, 16, 0.05)`,
landing on every boundary.

`pyproject.toml` `filterwarnings` silences this suite-wide, so a notebook run
is the only place it surfaces.

**Two forks:** a markdown note (small), or offset the axis and regenerate
`data.csv` (shifts every fitted number in the notebook). Pick one.

### 5. `20_multi_file_independent_fit` cell 7 — 5 CI warnings the sibling already fixed

`project.fit_baselines(model_name="base", stages=2)` emits five
`UserWarning: Bound reached with prob(GLP_01_m=0) = … < max(sigmas)`.

`21_multi_file_shared_fit` cell 7 already solves it:
`fit_baselines(model_name='base', stages=2, try_ci=0)` with the comment
`# Fit baselines (try_ci=0 skips confidence intervals: see 12_uncertainty_mcmc)`.

**Fix:** copy the sibling's line.

### 6. `21_multi_file_shared_fit` — structurally invalid notebook JSON

`nbformat 4`, `nbformat_minor 4`, but cell 12 alone carries `"id": "12"` —
cell ids arrived in 4.5. nbconvert's validator prints
`ERROR | Notebook JSON is invalid: Additional properties are not allowed
('id' was unexpected)`. Exit code is still 0.

Every peer is internally consistent: 01/03/04/10/11/12/20 are 4.5 with ids on
all cells; 02 is 4.4 with none. This is one hand-typed id in one cell.

**Fix:** delete the `"id"` key, or bump the file to 4.5 and let
`scripts/normalize_notebooks.py` add ids everywhere.

---

## Uncommented / misleading kwargs

### 7. `stages=2` means two different things in `20`

- Cell 7 `project.fit_baselines(..., stages=2)` — `Project.fit_baselines`
  defaults to `stages: int = 2` (`trspecfit.py:1334`). Restates a default.
- Cell 9 `f.fit_2d(..., stages=2)` — `File.fit_2d` defaults to
  `stages: int = 1` (`trspecfit.py:4556`). A deliberate choice.

Same literal, opposite meanings, uncommented in both.

### 8. `time_type='ind'` uncommented — `20` cell 5 and `21` cell 5

`define_baselines(time_start=0, time_stop=22, time_type='ind')`. Default is
`"abs"`, so `0`/`22` silently switch from time units to indices. On this axis
indices 0–22 are t = −20.0 to −9.0.

### 9. `detail=1` uncommented — `20` cell 3, `21` cell 3

`project.describe(detail=1)`; default is `detail=0` (`trspecfit.py:957`).
`1` also plots the 2D data grid.

### 10. Carried over from 01–04

`detail=` uncommented in 01 and 02; `try_ci=0` justified three different ways
in 04 and twice in 02; `stages=` uncommented in 04.

---

## YAML hygiene

### 11. `static` parameters keep stale bounds — `20` and `21`, `models_energy.yaml` `2D:` block

`m: [1E-2, static, -1, 1]`, `b: [0, static, -5, 5]`, `A: [12, static, 5, 15]`,
`x0: [8, static, 5, 15]`, `F: [1.5, static, 0.75, 2.5]`, `m: [0.3, static, 0, 1]`.

`static` → `vary=False` (`utils/lmfit.py:72`), so the bounds are dead. The
house convention is visible in `01_basic_fitting/models_energy.yaml`, where
every pinned parameter is `[value, False]` with no bounds. The values are
stale too — they're overwritten from the baseline fit before the 2D fit runs.

Also in `21/data/models_energy_truth.yaml` and `models_time_truth.yaml`:
`xStart: [0, False, 0, 20]`, `xStop: [20, False, 0, 20]`, `t0: [0, False, 0, 1]`.

### 12. Vary-level legend is in the file that doesn't use it — `20`

The `# - "project": shared across all files` legend sits in
`models_energy.yaml`, whose `2D:` block contains no `project` or `file`
values. `models_time.yaml` — which actually has `SD: [5, project, ...]` and
`A: [5, file, ...]` — has no such comment.

---

## Prose / structure (lower priority)

- **`20` has no `## Next Steps`.** Closes at `## Tips`; "Next step:" is a
  bullet inside it. Its peer `21` has a real one. (10 and 11 also lack it —
  decide whether the bar is real before fixing.)
- **`20` cell 0, 45-word sentence** with two dash pairs.
- **`20` cells 0/4/8/14 — `for file in files:` vs `for f in files:`**, one
  idiom spelled two ways.
- **`21` cell 9 dumps 44,376 characters / 423 lines** of `SavedFitSlot`,
  `optimization_hash`, `joint_ref` internals, because `project.fit_2d(...)` is
  left as the cell's trailing expression. These are criterion 15's exact
  examples of internal vocabulary. Add a `;` or assign the result.
- **`21` explains the project/file/static triple in five places** plus the
  YAML — cell 0, cell 8, cell 10, cell 11 comment, cell 12 Tips.
- **`01`/`02` duplicated dead line** `#file.model_active.visualize()`.
- **`02` cell 21** says baseline fitting gives "initial values" when
  `models_energy.yaml` pins those parameters `vary=False`.

---

## Not a defect — don't chase these

- **`20` fails to execute in place** with `ValueError: Archive at batch.fit.h5
  has schema_version '2' … writer's '7'`. That is an untracked local
  leftover from a run predating the schema-7 work. Delete
  `examples/fitting_workflows/20_multi_file_independent_fit/batch.fit.h5`
  before auditing. On a clean checkout `20` exits 0.
- **Cross-notebook repetition is correct, not a defect.** Notebooks must be
  independently runnable and understandable; a reader arrives at whichever
  notebook is closest to their problem. Do not cut an explanation because a
  sibling has it.

---

## API gaps worth a TODO (not example fixes)

- `04` cell 15 uses `file.model_active.components[1]` — the only index-based
  component access in the example set. No named accessor exists.
- `21` cell 11 hand-rolls the combined parameter table that
  `JointFitResult.params` already provides
  (`utils/fit_io.py:584`, via `project.results.get_joint(model="2D")`).
  Unverified caveat: the joint table's names are optimizer-scoped
  (`file00_GLP_01_x0_...`), so the per-file view may be deliberate — but the
  notebook never says so.


## 10 / 11 / 12 audit, 2026-09-13

### 13. `10_model_comparison` cell 8 — wrong chi-squared formula (FAIL)

Prose: ``chi2_red = chi2_red_raw / NOISE_SIGMA**2 is the sigma-calibrated value``.
The divisor is `sigma_eff**2`, not `NOISE_SIGMA**2` (`fitlib.py:142-147`:
`sigma_sq = float(sigma_eff) ** 2` then `chi2_red = chi2_red_raw / sigma_sq`),
and for a baseline fit `sigma_eff = sigma_data / sqrt(n_avg)`
(`utils/fit_io.py:144-147`). The run prints `chi2_red_raw = 0.051192`,
`chi2_red = 5.827871`; the stated formula gives 0.971. The *next bullet in the
same cell* states the sqrt(N_avg) correction, so it contradicts its neighbour.
Softer repeat in cell 39 Tips ("divide by `alt_sigma**2`") — correct only where
`sigma_eff == sigma_data`, i.e. not for the baseline rows that section produced.

### 14. `10_model_comparison` cell 10 — wrong shared-scale claim (FAIL)

Prose: ``FitResults.plot_residuals ... all models on one shared scale``.
Baseline slots route to `plot_residual_panels_1d`, which passes `sharex="col"`
and never sets `sharey` or any ylim (`utils/plot.py:1466`). Verified: the
panels render with different y-limits. Only the 2D path shares a scale
(`plot_residual_maps_2d`), which cell 27 states correctly. The 1D claim is
false in the direction that flatters the losing model.

### 15. `10_model_comparison` `models_energy.yaml` — stale prediction in a comment (FAIL)

`sbs_x0_A` block comment: "More flexible (marginally lower chi2_red on most
slices)". Measured: lower on 135 of 440 slices (31%), higher on 305. True of
the *unreduced* `chi2_raw` (lower on 437/440) — `chi2_red` divides by a DoF the
extra parameter removes. The notebook's own cell 17/20 printouts contradict it
in every visible slice. *Unverified by me — the agent re-ran sections 0-2 to
pivot 440 slices; re-check before fixing.*

### 16. `11_save_load_export` cell 8 — kwarg that does not exist (FAIL)

Prose: "The same `model=` / `fit_type=` filters that drive `compare_models`
also drive `save_fit` and `export_fit`". No spelling of `compare_models`
accepts `model=`: `FitResults.compare_models` takes `models=` (plural,
keyword-only, `fit_results.py:1940`), `File.compare_models` takes `*models`
positionally (`trspecfit.py:5131`), while `save_fit` takes `model=` singular
(`trspecfit.py:3318`). Only `fit_type=` is shared. The agent ran it: `TypeError`.

### 17. `12_uncertainty_mcmc` cell 33 — describes a warning the run never emits

Prose: "corner cannot even draw contours for the plot above and says so (*Too
few points to create valid contours*)". That string appears nowhere in the
executed outputs; the `--dump` footer lists three warnings, none of them this.
The string exists only in cell 33's own markdown.

### 18. Smaller, same round

- `11` cell 5 comment `# overwrite=True: replace any archive left by a previous
  run` — `overwrite=` is slot-scoped; `write_archive` opens `h5py.File(path,
  "a")` and appends. The notebook's own Tips says the opposite ("augments it").
- `11` cell 16 describes a `get()` error whose real text differs from the prose.
- `12` cell 27 comment says `stages=1` re-converges from 01's fitted 2D values;
  true for three parameters but `GLP_01_x0` is re-seeded from this notebook's
  own baseline (init = 9.001492, printed in cell 14).
- `12` section 4 compares MCMC widths against a Nelder stderr while section 2
  defined tier 1 as the leastsq covariance (`stages=1` runs `fit_alg_1`,
  default `'Nelder'`, `fitlib.py:824`).
- `12` cell 10/26/31 `use_mc=1` uncommented; it is tri-state, not boolean
  (`utils/lmfit.py:712`), and `use_mc=2` has a recorded trap.
- `10` cells 9, 11, 26, 28, 32 are bare-call cells with no comment, while
  section 2's equivalents all carry one.
- `2`/`10`/`20` all print truth constants and never pair them with fitted
  values in a closing section.


## 21 re-audit + remaining items, 2026-09-13

The `21` re-audit (run against the revised skill) re-found every defect the
first pass found, plus these. Items below are additional to 1-18 above.

### 19. `21_multi_file_shared_fit` cell 6 — convergence claim contradicted by the run (FAIL)

Prose: *"Since the spectral shape is identical across files (only the
time-dependent shift differs), the baseline parameters should converge to
similar values."*

`GLP_01_m` across the six files: **0.0563767, 0.2860204, 0.2074394, 0.3172321,
0.3045525, 0.4341475** — a 7.7x spread against a truth of 0.3, with file 1
essentially pure Gaussian (`GLP` docstring: "m = 0: Pure Gaussian"). The other
shape parameters do converge (`LinBack_m` 0.3993-0.4045, `GLP_01_x0`
9.9905-10.0075), so the sentence is right in general and wrong for the one
parameter a reader would check. Found independently by both `21` audits.

Compounding: cell 7 prints no parameter values at all — only "Baseline fit
complete for 6 files" plus six figures — so the reader cannot check it there
either.

### 20. `expFun` dynamics described inconsistently across the repo

Same root as items 1 and 2. `expFun` (`functions/time.py:166-190`) is
`A > 0: Jumps to A at t0, decays toward 0`. The repo calls this:
- `01` c16: "rising after t = 0 and leveling off" (wrong — item 1)
- `04` c4 / `generate_data.ipynb:151`: "jumps toward lower binding energy"
  (wrong direction — item 2)
- `21` c0/c2/c9: "exponential shift of peak position" (does not convey
  jump-and-relax)
- `20` c0: "the kicked-decay dataset" (correct, and the best phrasing)

**Fix them together and adopt `20`'s vocabulary**, or the next audit finds a
fourth wording.

### 21. `12_uncertainty_mcmc` cell 34 — a cross-check the reader cannot perform

Tips: *"the sampled noise scale should sit near the model-free sigma with a
narrow spread (about 1/sqrt(2N) of itself for N fitted points)"*. Cell 32
prints `sigma spread %` = 0.266. The fit window is 301 pixels x 481 time
slices, N = 144,781, so 100/sqrt(2N) = 0.186% — measured/predicted = 1.43.
The notebook never prints N, so the check is unusable as written. Either print
N or state the rule qualitatively.

### 22. Claims made but never demonstrated

Criterion 12 requires a try/except cell or the sentence dropped.
- `10` c4: "`compare_models` refuses to rank fits made on different fit
  windows". The refusal is real (`fit_results.py:2110-2117`) but nothing shows
  it. Peer `11` c17 does exactly this for `get()` — copy that pattern.
- `12` c33: the corner "Too few points to create valid contours" warning
  (item 17) is the same defect in reverse — described, never emitted.

### 23. API surface the notebooks skip

- `11` never names `Project.load_fits` (`trspecfit.py:869`) despite billing
  itself as "the canonical reference for the `FitResults` archive API", nor
  its trap: the returned `FitResults` is independent of `_fit_history` and
  never merges into it. `grep -rn load_fits examples/` returns nothing across
  all nine notebooks.
- `11` c8's `by=` table drops three constraints its docstring records
  (`trspecfit.py:480-487`): `chi2_red` is ranked by `|x - 1|` not minimized,
  it requires a sigma consistent across the group, and a group spanning
  multiple fit views refuses to rank.
- `11` c20: "fitted values are compared at the archive's named tolerances" —
  the names are `_PARAMS_EQUIV_RTOL` / `_PARAMS_EQUIV_ATOL`
  (`utils/fit_io.py:2502`), private constants with no public spelling.
- `11` c18: sends the reader to the `SavedFitSlot` docstring, which never
  renders — the class is not in `trspecfit.__all__` and not autodoc'd in
  `docs/api/`. Also `conf_ci` has a bare type line with no description.
- `21` c9/c11 discards the `JointFitResult` that `fit_2d` returns and rebuilds
  its `params` table by hand (see API-gaps section).

### 24. One word, two meanings

- `10`: **"archive"** means the in-session fit history (c29, c37) *and* the
  on-disk file (c35). The library keeps them apart — `Project.drop_fits`
  docstring: "Remove one fit ... from the in-session history / Never touches
  archives on disk."
- `10` c8: calls the column "the selection" while every printed table headers
  it `selection_json`, and never says its payload is in **index** units — the
  reader sees `{"base_t_ind": [0, 6], "e_lim": [40, 320]}` beside a section
  that wrote `energy_limits=[2, 16]` and `time_stop=5`, with no way to
  reconcile them.
- `21` c7 vs c9: `try_ci=0` is justified with a comment on the baseline fit and
  silently dropped on `fit_2d`, which then prints an 8-row confidence table no
  prose acknowledges. Next Steps then says "put error bars on these shared and
  per-file parameters" — they are already printed.

### 25. Results printed but never interpreted

- `10` section 2: the control runs (`wide.describe()`: `sbs_x0_A` mean 1.004615
  / std 0.081434 vs `sbs_x0_only` 1.004598 / 0.080957) and no prose reads it.
  `chi2_red` and `r2` do not separate the two models; only AIC/BIC do
  (-361980.76 vs -361545.94). The verdict arrives two sections later as a
  backward reference.
- `21` c9: prints `C(file00_..._A, ..._tau) = -0.6763` and
  `C(..._SD, ..._tau) = -0.2824` — the shared-vs-per-file correlation
  structure that is the entire point of the method — and never mentions it.
- `21` section 3 never says *why* you would share `tau`/`SD` (instrument and
  sample properties common to all fluences; the strong files then constrain
  the weak ones). The payoff is in the run: file 6 recovers
  `A = 0.10332 +/- 0.00510` only because `tau` and `SD` were pinned.
- `21` c5: the baseline window (indices 0-22 = t -20.0 to -9.0) is exactly 3
  sigma before t0 for truth `SD = 3` — a correct, non-obvious choice presented
  as "before dynamics onset", which is not the actionable rule. A naive reader
  picks index 49 (t = -0.5) and contaminates the baseline.
- `12` c27 renders a corner plot internally (`trspecfit.py:4685`) that the
  section never tells the reader is being produced; c33 then refers to "the
  plot above".

### 26. Smaller

- `12` c31 retypes `nwalkers`/`steps`/`burn` from c26 under the prose claim
  "The second chain is identical except for its start" — change c26 and the
  claim goes stale silently.
- `12` c0 roadmap enumerates the `get_*` accessors by name and omits
  `plot_mcmc` and `compare_models`, which the section also teaches.
- `11` c4 calls a recipe "one line of pandas" above a two-line snippet; c20
  calls the same recipe "the two-line pandas recipe".
- `11` c0: "prints a one-line confirmation" — the executed output is 3 lines.
- `11`: two sentences at 52 and 50 words (c18, c4).
- `10` c5 retypes the axis endpoints `[-10, 99]` that `file.time` already
  carries, as a no-op full-range restatement.
- `21` c11: `import pandas as pd` outside the cell-1 import block.

---

## Criterion-14 Tips findings — resolved 2026-09-13

Every audit that reached criterion 14 filed its findings against the notebook's
**Tips** cell: `20`->c14, `12`->c34, `10`->c39, `21`->c12. That was 4 of 4 — a
conflict between criteria, not a defect in four notebooks. Criterion 7 requires
a Tips section and criterion 14 made a fact's third statement a WARN, so a
summarising Tips section was a WARN generator by construction.

**Criterion 14 now exempts Tips from the count**: count the statements outside
Tips, land the WARN on the third of those, never on the Tips bullet. Naming a
fact in Tips is an anchor; re-explaining it there is still a finding.

Re-graded under that rule, **9 of the 11 parked findings were artifacts and are
dropped**. Two are real and move off the Tips cell:

### 27. `21_multi_file_shared_fit` — vary-level gloss explained three times before Tips

Cells 0, 8 and 10 each *explain* the project/file/static triple (not just name
it): c0 `"project": shared across all files (fitted once) / "file": independent
per file`; c8 `"project": tau, gaussCONV SD — shared dynamics across all files`;
c10 `SD and tau are "project" — fit once, shared across all files`. WARN lands
on **c10**. Cell 11's one-line comment is a legitimate anchor and c12 (Tips) is
now exempt.

### 28. `20_multi_file_independent_fit` — "not a shared fit" explained three times before Tips

Cell 0 (*"the setup is shared, the fits are not"*), cell 6 (*"It is **not** a
shared fit; only the setup is shared"*), cell 8 (*"the **shared-parameter**
path... here we want every file's tau, SD, A fit independently"*). WARN lands on
**c8**. Tips bullet 1 is now exempt.

**Dropped as criteria artifacts** (each stated only once or twice outside Tips):
`21` static-from-baseline; `10` handle-is-the-short-id and prefix-or-label;
`12` one-fit-all-three-tiers, stderr-failure-modes, acceptance-checks-mixing,
and the 50-tau warning; `20` file=-filter caveat and vary-level explanation.

**Do not gut any Tips section.** The two findings above are fixed by trimming
the third *body* statement, not the summary.


## 21 third audit, 2026-09-13 — new findings

### 29. `21` Next Steps sends the reader somewhere that cannot deliver

Cell 12: *"[`12_uncertainty_mcmc`] — put error bars on these shared and
per-file parameters."* After `project.fit_2d`, the File-level `stderr` is
`None` for every parameter — by design: *"per-file stderr/CI are absent by
design (joint covariance does not decompose cleanly per file)"*
(`trspecfit.py:1780`). `12_uncertainty_mcmc` is entirely File-level, so
following that pointer yields nothing. The error bars **do** exist via
`project.results.get_joint(model="2D")` (stderr 0.0146 SD, 0.1293 tau, 0.0077
file00 A, populated `conf_ci`) — which the notebook never names, while cell 9
already printed that CI table without comment.

### 30. `21` vary taxonomy is incomplete, and a count is wrong

- Cell 0: *"Parameters are classified by vary level in the YAML model:
  'project' / 'file' / 'static'"*. Both YAMLs also use plain `False` —
  `models_energy.yaml:26-27` (`xStart`, `xStop`) and `models_time.yaml:10`
  (`t0`). `models_energy.yaml:18` repeats the over-claim as a YAML comment,
  two lines above `xStart: [0, False]`.
- Cell 10: *"below we collect the three dynamics parameters for every file"* —
  `MonoExpPosIRF` has four (`SD`, `A`, `tau`, `t0`). `t0` appears nowhere in
  the notebook.

### 31. `21` "Global 2D Fitting" collides with the repo's own term

Section 3 is headed `## 3. Global 2D Fitting` for the multi-file joint fit.
But `get_parameters`' docstring (`trspecfit.py:4738`) says
*"- '2d': 2D global fit (from ``fit_2d``)"* — the repo's name for the
**single-file** call, which is exactly what peer `20` uses for its
**independent** path. The same notebook also calls the operation "a true joint
fit" (Tips) and "fit all files simultaneously" (cells 0, 8, 9).

### 32. `21` smaller

- Cell 9 prints an uninterpreted confidence-interval table (`try_ci` defaults
  to 1, `fitlib.py:580`) — nothing reads it.
- Cell 11 reads `get_parameters(fit_type="2d")`, whose `stderr` column is
  `None` for every row after a joint fit, and never says so.
- `### 2.2. Fit Baseline` restates its parent `## 2. Fit Baseline Spectra`.


### 33. Truth comparisons that don't cover the fit (new criterion 2 clause)

Criterion 2 now requires the comparison to cover every fitted quantity that has
committed truth, and to state a verdict. Three notebooks fail the coverage half:

- **`21`** pairs the dynamics truth (`SD`, `tau`, per-file `A`) with fitted
  values in cell 10, but the eight baseline energy parameters it also fits —
  truth committed in `data/models_energy_truth.yaml` — are never compared to
  anything. That is where the `GLP_01_m` 8x spread hides (item 19). Cell 10
  also gives no verdict: two columns, no sentence saying they agree.
- **`10`** prints its truth constants in cell 3 and fits `SD` (3.013 vs 3.0),
  `tau` (29.97 vs 30) and `A` (2.002 vs 2) — recovered to 0.4% and never
  mentioned. The closing section is about variants and labels.
- **`11`** has no truth of its own; notebook 10 generates it inside `%%capture`,
  so a reader of 11 never sees it. Cell 17 prints `GLP_01_x0 8.001277` with no
  truth beside it, and the "lossless" claim (loaded == live) is never shown.

**Fix shape:** add the missing rows, and one sentence per table saying what the
comparison shows. Where the fit reports no uncertainty (a joint fit's per-file
parameters), say the comparison is qualitative.
