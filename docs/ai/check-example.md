
# Check Example Quality

Shared source of truth for auditing an `examples/fitting_workflows/` notebook
against the gold-standard bar before a merge or release. These eleven criteria
were distilled from `01_basic_fitting` while upgrading the example set; this
doc is their permanent home.

## Scope

This check accepts one argument: **the example to audit**, given as a directory
path or an `NN_` prefix resolved under `examples/fitting_workflows/`
(e.g. `04`, `04_parameter_profiles`, or a full path). With no argument, audit
every `examples/fitting_workflows/NN_*/` directory and print one table per
example plus a roll-up.

## Grades

The **agent is the grader**; the mechanical pre-pass only supplies evidence.
For each criterion report one of:

- **PASS** — criterion met.
- **WARN** — partially met or a judgment call the author should confirm
  (e.g. narrative present but a section never says *why*).
- **FAIL** — criterion not met; list the specific gap with file/cell.
- **N/A** — criterion does not apply to this notebook's deliberate variant
  (e.g. `auto_export` on an export-topic notebook). State *why* it is N/A.

The pre-pass also prints **INFO** lines — facts it found but cannot judge
(missing `data/`, no `*_truth.yaml`, `auto_export` unset, prose-voice
candidates). These are **not defects**: resolve every INFO to PASS / N/A / FAIL
by reading the notebook. A clean example ends with **0 FAIL, 0 WARN** even when
the pre-pass emitted several INFO lines.

Run the mechanical checks first, then read the notebook and YAMLs for the
judgment checks. Work through the list in order.

## Mechanical pre-pass

```bash
# one example, or omit the argument to sweep every example:
.venv/bin/python .claude/skills/check-example/check_example_mechanics.py <example>
```

This gathers evidence for the scriptable criteria (stripped outputs, required
files, committed truth, `auto_export`, side-effect artifacts, the roadmap/TOC
numbering). It emits **PASS/WARN/FAIL** for what it can decide deterministically
(committed outputs and artifacts FAIL; the roadmap/TOC numbering PASS or WARN);
everything intent-dependent (missing `data/`, no `*_truth.yaml`, `auto_export`
unset) and
the prose-voice candidates come out as **INFO** for you to resolve. Fold its
output into criteria 2, 3, 4, 5, 9, and 10 below. It does **not** execute the notebook
(criterion 1 is the slow one — run it separately).

## 1. Runs clean end-to-end

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --stdout \
  examples/fitting_workflows/<example>/example.ipynb > /dev/null
```

PASS if it exits 0 with zero `stderr`/error outputs and zero warnings in cell
outputs. An *unavoidable* warning must carry an explaining markdown note in the
notebook; an *avoidable* one must be fixed at the source (e.g. `try_ci=0` on a
baseline fit that otherwise prints repeated `lmfit.confidence` warnings). WARN
if it runs but emits an unexplained-but-benign warning; FAIL on any error or
nonzero exit.

## 2. Truth-anchored

The closing section must let the reader verify the fit against ground truth:

- **Synthetic (default):** committed `data/*_truth.yaml` regenerable via
  `data/generate_data.ipynb`, and a closing section that quotes the truth
  values next to the fitted ones.
- **Real-data variant:** measured data kept because known physics anchors the
  fit. State the data is real, document provenance, and compare to literature
  values in the closing section.
- **Inline-synthetic variant:** generation stays inline when the tunable ground
  truth is itself the teaching device. Keep the truth constants in one labeled
  cell and print them.

FAIL if there is no truth anchor of any kind. WARN if truth exists but the
closing section never surfaces it for comparison.

## 3. Self-contained, or a signposted reuse of a sibling

Default: `data/`, the model YAMLs, and `project.yaml` live in the example dir
and the notebook stands alone. Deliberate, clearly-signposted reuse of a
sibling is equally fine — the standard is *documented handoff*, not *zero
dependency*. Recognized handoffs: a `%run` preamble that re-runs a sibling
in-kernel, or a relative-path load of a sibling's data instead of duplicating
it. FAIL only on an **undocumented** cross-notebook dependency — a path into a
sibling with no prose explaining it. If the reuse is stated up front, PASS.

## 4. No surprise side-effects

`project.yaml` sets `auto_export: False` with an explanatory comment, so the
fit calls don't spray CSV/PNG dumps. N/A when persistence/export is the
notebook's actual topic — say so.

Artifact severity: **committed** CSV/PNG/`.fit.h5` fit outputs FAIL (they
pollute the repo). **Untracked/gitignored** outputs are reported INFO, not a
failure — they are transient (left by a local run, or expected for the export
demos) as long as they are gitignored. Empty `*_fits/` directory trees from
`create_model_path` are the known eager-mkdir quirk (see TODO.md) and are
ignored entirely. `data/*.csv` inputs are never counted as artifacts.

## 5. One main message, why-driven narrative & roadmap-as-TOC

Three parts:

- **One clear main message.** The notebook has a single takeaway, stated
  plainly in the opening — e.g. 10: "`file.compare_models()` ranks candidate
  models, at any fit level"; 20: "a `Project` fits many files with shared setup
  but independent per-file fits". Every section serves that message. If you
  cannot name it in one sentence, the notebook is doing too much (see also 7,
  scope). WARN if the opening never states a single takeaway, or the message is
  diffuse across several competing points.
- **Numbered roadmap doubles as a table of contents.** The opening cell's
  roadmap uses the *same numbers* as the `## N` section headers, so step *N*
  points the reader straight at section *N* (the mechanical pre-pass checks
  this). The roadmap may be a numbered **list** or a numbered **table column**
  (an overview table whose first column is `1, 2, 3 …`). The `## N` headers
  must themselves be consecutive (a gap like `## 0,1,2,4,5` is the bug this
  catches); a leading `## 0` *setup* section (e.g. data generation or a
  preamble) is allowed and need not appear in the roadmap. WARN when the
  roadmap does not match the section numbers — either align it, or, if that
  numbered list is doing something else, bullet it so it doesn't masquerade as
  a TOC and add a real roadmap.
- **Why, not just what.** Each section explains *why* the step exists (e.g.
  01's "Why global?", 20's "Why `Project` instead of a bare loop?"), not just
  what the next call does. WARN if the prose is purely procedural ("now we call
  X") with no motivation.
- **The message is demonstrated, not just asserted.** A concrete result proves
  the claim in-notebook — a result table, a diagnostic plot, an artifact
  inspection, or a sanity check, not prose alone. (Broader than criterion 2,
  which is specifically the ground-truth comparison.) WARN if the takeaway is
  stated but never shown by a result.

## 6. Deliberate kwargs

Non-obvious arguments carry a short inline comment; the notebook does not lean
on defaults that produce unexplained output. (e.g. `try_ci=0`, `stages=2`,
`time_type='ind'`, generously-initialized convolution widths.) WARN per
unexplained non-obvious kwarg.

## 7. Scope discipline

One topic per notebook. Adjacent topics are delegated via valid relative links,
not re-taught. The notebook closes with **Tips** and **Next Steps** sections,
and the Next Steps links resolve. FAIL on a broken relative link; WARN on
scope bleed or a missing Tips/Next Steps close.

**Name the nearby trap.** If the notebook sits next to a confusable API or
workflow, it should call out the trap explicitly and link the right neighbor —
e.g. 20 says why `for f in files: f.fit_2d(...)` is the independent-fit path and
`project.fit_2d()` would be the *shared*-parameter path (→ 21). WARN when a
foreseeable "used the wrong neighbor's call" mistake is left unaddressed.

## 8. Commented YAMLs

Model YAMLs say what each block is for and point at the `functions/` source for
the available functions/parameters. Fixed parameters are normalized to
`[value, False]` (no stale bounds). WARN on an uncommented or stale YAML.

## 9. Stripped outputs

The committed `.ipynb` has zero cell outputs and null `execution_count`
(`nbstripout` stays installed). Rendered outputs are produced at docs-build
time, never committed. FAIL if any cell carries committed output.

## 10. Human prose voice

The prose reads as written by the library's authors, not generated. This is a
judgment call — read the markdown cells and quote the offending line. Flag
(WARN):

- Filler sentence fragments standing in for sentences ("Multi-file workspace,
  per-file independent fits.").
- Formulaic scaffolding: "N concrete payoffs:", "it's worth noting", "In this
  section we will", "Let's dive in".
- Hollow parallelism — every bullet an identical "term — em-dash gloss" shape,
  or "not just X, but Y" where the contrast adds nothing.
- Redundant restatement: the same idea in two consecutive sentences.
- Inflated vocabulary in a technical doc: delve, leverage, seamless, robust
  (when not a real property), crucial, comprehensive, powerful, utilize,
  showcase, realm, landscape, testament, underscores.

Do **not** flag em-dashes, contrast, or precise technical phrasing on sight —
this repo uses all three well. Flag them only when they are padding. The
mechanical pre-pass emits INFO-only candidates (a high-precision word/phrase
list); treat those as pointers to read, never as failures.

## 11. Method assumptions & failure modes

For a non-trivial method, the notebook teaches the key assumption, degeneracy,
or failure/convergence diagnostic *where it bites* — not just the happy path.
This is what separates a recipe from an example that builds judgment. Typical
traps: parameter identifiability/degeneracy (what must be fixed vs fit), what a
fit metric or residual can and cannot tell you, sampler convergence (walkers,
burn-in, autocorrelation), kernel truncation, aliasing.

**N/A for basic-mechanics notebooks** where the method carries no non-obvious
caveat — say N/A, don't invent one. WARN when a method *does* have a well-known
trap that the notebook silently skips.

## Known pitfalls (carry-over lessons)

Hard-won gotchas worth re-checking when a criterion looks borderline:

- **`%%capture` path quoting (IPython 9.x):** a bare path in `%cd -q ../dir`
  tokenizes as a malformed number and crashes. Quote it.
- **Convolution kernel sizing:** `create_t_kernel` sizes the kernel from the
  *initial* parameter value and never rebuilds it — a too-small init silently
  truncates the kernel and biases the fitted width. Initialize conv widths
  generously above the expected value.
- **Subcycle-boundary time samples:** generate synthetic multi-cycle data on
  the *reloaded* CSV axes, not the in-memory `np.arange` axes — boundary-exact
  samples flip subcycle assignment under `%.6e` rounding and bias the fit.
- **Baseline-window IRF contamination:** an IRF onset at `t0` leaks the
  convolved dynamics back into the baseline window; leave enough clean pre-t0
  spectra (extend the time axis) rather than narrowing the baseline window to
  a few points.

## Summary

Print a table:

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 1 | Runs clean end-to-end | ... | ... |
| 2 | Truth-anchored | ... | ... |
| 3 | Self-contained directory | ... | ... |
| 4 | No surprise side-effects | ... | ... |
| 5 | Main message, narrative & roadmap-as-TOC | ... | ... |
| 6 | Deliberate kwargs | ... | ... |
| 7 | Scope discipline | ... | ... |
| 8 | Commented YAMLs | ... | ... |
| 9 | Stripped outputs | ... | ... |
| 10 | Human prose voice | ... | ... |
| 11 | Method assumptions & failure modes | ... | ... |

Then list every WARN/FAIL with an actionable next step.
