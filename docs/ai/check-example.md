
# Check Example Quality

Shared source of truth for auditing an `examples/fitting_workflows/` notebook
against the gold-standard bar before a merge or release. Criteria 1–11 were
distilled from `01_basic_fitting` while upgrading the example set; 12–15 (and
the readability additions to 5, 6, 7, 10 and 11) came out of the
`10_model_comparison` review on 2026-09-02, which the original list passed
despite a wrong section reference, a peak position contradicted by the
notebook's own printout, one API spelled two ways, and the same rationale in
three places. The `11_save_load_export` review the same day added the claim
ledger and fact inventory (12, 14), practice consistency (13), and the
near-verbatim, sibling, sentence-length, formatter and YAML-comment scans.
The `12_uncertainty_mcmc` review on 2026-09-12 added the upstream-semantics
and own-target checks (12), the control rule (11), the retyped-state rule
(13), the own-API rule (7), and the `--dump` warnings footer (1), and replaced
the environment-dependent-numbers bound in 12 with "measured values become
vocabulary" after a percentage band had to be widened twice to keep up with
unseeded MCMC runs. This doc is their permanent home.

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
  (e.g. untracked export artifacts beside an export-topic notebook). State
  *why* it is N/A.

The pre-pass also prints **INFO** lines — facts it found but cannot judge
(missing `data/`, no `*_truth.yaml`, prose-voice candidates, `§`
cross-references, repeated phrases and calls, API names mentioned but never
called). These are **not defects**: resolve every INFO to PASS / N/A / FAIL
by reading the notebook. A clean example ends with **0 FAIL, 0 WARN** even when
the pre-pass emitted several INFO lines.

Run the mechanical checks first, then execute the notebook into a scratch
directory (criterion 1) and keep the executed copy, then read the notebook and
YAMLs for the judgment checks. Grade every claim about results, positions, or
winners against the *executed* outputs — the stripped source cannot show a
sentence contradicted by the notebook's own printout. Work through the list in
order.

Two artefacts are the deliverable for 12 and 14, not a by-product of skimming:
the **claim ledger** (every sentence that states a number, a count, a
behaviour, a default, a cross-reference or an API promise, each marked
verified-where or failed) and the **fact inventory** (every key fact with the
list of cells that state it). Build both while reading; the pre-pass seeds
them.

## Mechanical pre-pass

```bash
# one example, or omit the argument to sweep every example:
.venv/bin/python .claude/skills/check-example/check_example_mechanics.py <example>
```

This gathers evidence for the scriptable criteria: notebook-JSON schema,
stripped outputs, required files, committed truth, removed config keys, side-
effect artifacts, roadmap/TOC numbering, relative links, heading numbering
style, prose-voice candidates, `§` cross-references (with self-references
marked), near-duplicate API names, private-attribute access, imports outside
the import cell, repeated phrases, near-verbatim passages, repeated calls, API
names the prose mentions but the code never calls, behaviour claims (raises /
refuses / warns) with no demonstrating cell, measured-looking numbers quoted in
prose (percentages, multipliers, decimal σ distances, approximate values, point
timings), over-long sentences, over-wide code lines, style shared with same-
decade peers, and YAML comment wrapping. It emits **PASS/WARN/FAIL** for what
it can decide deterministically (notebook JSON that does not parse or disagrees
with its declared nbformat, a missing `example.ipynb`, committed outputs,
committed artifacts, removed config keys, and broken relative links FAIL; non-
consecutive or roadmap-mismatched `## N` numbering, mixed heading styles, and
private-attribute access in code WARN); everything intent-dependent comes out
as **INFO** for you to resolve. Several scans are deliberately narrowed so
their output stays worth reading: prose-voice reports PASS when it finds no
tell; prose-only names drops prose math, filenames, registry functions a YAML
selects by name, and lmfit parameter names, none of which can carry a "this
notebook calls it" claim; private-access ignores globs and filenames
(`models_*_truth.yaml` is not an attribute); measured values runs only on a
notebook with a stochastic step; near-verbatim looks only inside one notebook,
because criterion 14 does not compare notebooks; peer consistency needs two
peers before it calls a majority — with one peer it names the split, since
neither notebook is the outlier — and always reports a notebook inconsistent
with itself. Fold its output into criteria 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12,
14, and 15 below. It does **not** execute the notebook (criterion 1 is the slow
one — run it separately); its `--dump` mode reads the executed copy back for
criteria 1 and 12 and ends with a footer listing every warning or error the run
printed and whether any prose or comment mentions it.

## Criterion boundaries

A finding lands under the criterion that owns the *nature* of the defect, not
the kind of text it happens to sit in. Work these in order and stop at the
first that fits; placing findings, rather than finding them, is otherwise the
largest single overhead in an audit.

1. **Is the statement false?** → **12**, wherever it lives — prose, a code
   comment, a YAML comment, a Tips bullet. 12 owns truth, and owns it at FAIL
   severity. A comment that misstates what a kwarg does is not demoted to a
   criterion 6 WARN because it is a comment, and a YAML comment asserting a
   fit outcome that did not happen is not demoted to a criterion 8 WARN
   because it is in a YAML.
2. **Is it true, but missing where the reader needs it?** → the criterion that
   owns that content: a kwarg's reason → 6, a method's recorded trap → 11, a
   fitted-vs-truth comparison → 2, a link or a closing section → 7, an
   uninterpreted result → 5.
3. **Is it true and present, but repeated?** → **14**. A one-line comment
   echoing a keyword from the prose is the intended anchor and never counts,
   however redundant it looks.
4. **Is it true, present, once, and hard to read?** → **10**.
5. **Is it true, present, once, readable, but in the wrong place or form?**
   → the criterion that owns the form: off-topic content or a bypassed
   wrapper → **7**, a kwarg restating its default or missing its reason →
   **6**, two names or two idioms for one thing → **13**, a private
   attribute in code → **15**.

A finding none of the five claims is not a finding. One that two criteria
could claim is reported once, under the earliest that fits, with a one-clause
pointer to the other — never twice.
Report the *defect*, not the criterion's phrasing: two criteria describing one
sentence from different altitudes is one finding.

## 1. Runs clean end-to-end

```bash
# Copy the tracked examples tree out of the repo FIRST. nbconvert runs the
# kernel with cwd set to the notebook's own directory, so --output-dir
# isolates nothing on its own: fit artifacts land back in the example dir, and
# a stale one from an earlier run (a `*.fit.h5` written by another branch)
# makes the notebook fail in place — a local-state failure, not a defect.
# `git ls-files` keeps siblings (criterion 3 reuse) and drops every artifact.
# The directory carries <example> so two audits running at once cannot
# `rm -rf` each other's tree mid-execution.
rm -rf <scratch>/check-example-<example> && mkdir -p <scratch>/check-example-<example>
git ls-files -z examples/fitting_workflows \
  | tar --null -T - -cf - | tar -xf - -C <scratch>/check-example-<example>
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  <scratch>/check-example-<example>/examples/fitting_workflows/<example>/example.ipynb
# prose and trimmed outputs side by side (lmfit reports, progress bars and
# figures collapsed) — the evidence for criteria 2, 5, 11 and 12
.venv/bin/python .claude/skills/check-example/check_example_mechanics.py \
  --dump <scratch>/check-example-<example>/examples/fitting_workflows/<example>/example.ipynb
```

PASS if it exits 0 with zero `stderr`/error outputs and zero warnings in cell
outputs — and the converter's own stderr counts: an `ERROR | Notebook JSON is
invalid` arrives before any cell runs, exits 0, and never reaches the `--dump`
footer, so the pre-pass checks the committed JSON separately (`1 Notebook
schema`). An *unavoidable* warning must carry an explaining markdown note in the
notebook; an *avoidable* one must be fixed at the source (e.g. `try_ci=0` on a
baseline fit that otherwise prints repeated `lmfit.confidence` warnings). WARN
if it runs but emits an unexplained-but-benign warning; FAIL on any error or
nonzero exit.

Never grade from the stripped source alone. The executed copy is the evidence
for criteria 2, 5, 11 and 12. A working copy with outputs on only its first
few cells means the later sections have not been run since the last edit —
say so.

The `--dump` footer is this criterion's evidence: it lists each warning or
error line the run printed (stderr, stdout, and logging output alike) and
sorts it by whether a markdown cell or code comment quotes its key phrase.
A quoted one still needs the sentence to say what the warning means and why
it is expected here; an unquoted one is explained in other words somewhere
(say where) or it is the WARN above.

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

**The comparison covers what was fitted, and reaches a verdict.** Every fitted
quantity with committed truth is compared, or the notebook says why a subset is
representative — a notebook that pairs its dynamics parameters with truth while
leaving the baseline parameters it also fitted unchecked has a truth section
that looks complete and is not. And the comparison states what it shows rather
than setting two columns side by side and moving on: where the fit reports an
uncertainty, the verdict uses it (`2.988 ± 0.015` against a truth of 3 agrees;
`2.988` alone only looks close); where the fit reports none — a joint fit's
per-file parameters, a noiseless synthetic — the notebook says the comparison
is qualitative rather than implying a precision it does not have. Whether
anyone reads the table at all is criterion 5's *what is shown is read*; this is
whether the table covers the fit.

FAIL if there is no truth anchor of any kind. WARN if truth exists but the
closing section never surfaces it, surfaces only part of what was fitted, or
surfaces it without a verdict.

## 3. Self-contained, or a signposted reuse of a sibling

Default: `data/`, the model YAMLs, and `project.yaml` live in the example dir
and the notebook stands alone. Deliberate, clearly-signposted reuse of a
sibling is equally fine — the standard is *documented handoff*, not *zero
dependency*. Recognized handoffs: a `%run` preamble that re-runs a sibling
in-kernel, or a relative-path load of a sibling's data instead of duplicating
it. FAIL only on an **undocumented** cross-notebook dependency — a path into a
sibling with no prose explaining it. If the reuse is stated up front, PASS.

## 4. No surprise side-effects

Fits never write to disk (v0.14.0), so no opt-out is needed — but a stale
removed key (`auto_export:`, `path_results:`) in `project.yaml` makes
`Project()` raise at load and FAILs here. On-disk artifacts must come only
from the notebook's explicit `save_fits` / `export_fits` calls.

Artifact severity: **committed** CSV/PNG/`.fit.h5` fit outputs FAIL (they
pollute the repo). **Untracked/gitignored** outputs are reported INFO, not a
failure — they are transient (left by a local run, or expected for the export
demos) as long as they are gitignored. `data/*.csv` inputs are never counted
as artifacts.

## 5. One main message, why-driven narrative & roadmap-as-TOC

Five parts:

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
  a TOC and add a real roadmap. A roadmap row also names everything its
  section teaches — a row promising `variants()` / `diff()` for a section that
  goes on to introduce `set_label` and `drop_fits` under-sells it (WARN).
- **Why, not just what.** Each section explains *why* the step exists (e.g.
  01's "Why global?", 20's "Why `Project` instead of a bare loop?"), not just
  what the next call does. WARN if the prose is purely procedural ("now we call
  X") with no motivation.
- **The message is demonstrated, not just asserted.** A concrete result proves
  the claim in-notebook — a result table, a diagnostic plot, an artifact
  inspection, or a sanity check, not prose alone. (Broader than criterion 2,
  which is specifically the ground-truth comparison.) WARN if the takeaway is
  stated but never shown by a result.
- **What is shown is read.** The inverse defect, and the one that hides better:
  a cell prints the table, the control, the correlation matrix or the
  confidence intervals, and no prose says what the reader should take from it.
  A result nobody interprets is a hole in the argument exactly where the
  evidence was supposed to go. WARN on a printed diagnostic the notebook never
  reads — including one a default produced without being asked for.

## 6. Deliberate kwargs & honest comments

Non-obvious arguments carry a short inline comment; the notebook does not lean
on defaults that produce unexplained output. (e.g. `try_ci=0`, `stages=2`,
`time_type='ind'`, generously-initialized convolution widths.) WARN per
unexplained non-obvious kwarg.

Comments are honest and local. A comment describes the line beneath it, not a
neighbouring concept (a note on the two χ² columns sitting above a line that
only counts slices is a WARN). A code comment stays around three lines;
anything longer is prose that belongs in a markdown cell, once (see 14).
Imports live in one import cell. A notebook that inherits a kernel from a
`%run` preamble still states its *own* imports, in one cell right after the
preamble — "imported here because only this listing needs it" is not a
justification, and using a name the preamble happened to import (`trspecfit`,
`Path`) without importing it is the same defect from the other side. The
pre-pass lists cells that import. A bare call cell gets a one-line comment
saying what it shows. WARN per violation.

A kwarg passed at its default value earns its line in one of two ways: the
notebook teaches it, or its name alone tells the reader what it does and what
the value means (`thin=1`, `overwrite=False`). Any other default spelled out
reads as a deliberate choice and buries the ones that are. WARN.

## 7. Scope discipline

One topic per notebook. Adjacent topics are delegated via valid relative links,
not re-taught. The notebook closes with **Tips** and **Next Steps** sections,
and the Next Steps links resolve. FAIL on a broken relative link (the
pre-pass resolves them); WARN on scope bleed or a missing Tips/Next Steps
close. Scope bleed includes a parenthetical about a feature the notebook
cannot demonstrate — a joint multi-file exception mentioned in a single-file
notebook — delegate it with a link or cut it. Roadmap and future-work remarks
("per-pixel σ is on the roadmap") date the notebook the day they are written
and belong in `TODO.md`; cut them (WARN).

**Call the library's own API.** An example teaches trspecfit, so where
trspecfit has a method for the job the example calls that method. Reaching
past the wrapper to the package underneath (`corner.corner(mcmc.flatchain)`
where `plot_mcmc` exists, a bare `lmfit` call where a `fit_*` method does the
same) hides the API being taught and drags a dependency into the import cell
that the reader then believes they need. Going underneath is fine when the
notebook says so and the point is what the wrapper does not offer. WARN
otherwise.

**Name the nearby trap.** If the notebook sits next to a confusable API or
workflow, it should call out the trap explicitly and link the right neighbor —
e.g. 20 says why `for f in files: f.fit_2d(...)` is the independent-fit path and
`project.fit_2d()` would be the *shared*-parameter path (→ 21). WARN when a
foreseeable "used the wrong neighbor's call" mistake is left unaddressed.

## 8. Commented YAMLs

Model YAMLs say what each block is for and point at the `functions/` source for
the available functions/parameters. Fixed parameters are normalized to
`[value, False]` (no stale bounds) — and a string vary level counts as fixed
when it resolves to one, so `static` (`utils/lmfit.py`: `_vary_to_bool`) must
shed its bounds exactly as `False` does. Comments wrap cleanly — no orphan
two- or three-word lines, no lines past the formatter's width; the pre-pass
lists both. WARN on an uncommented or stale YAML. Grade the YAMLs this example
directory holds; one it reuses from a sibling is graded under that sibling. An
example that needs no YAML of its own has nothing to grade here.

## 9. Stripped outputs

The committed `.ipynb` has zero cell outputs and null `execution_count`
(`nbstripout` stays installed). Nothing renders them later either —
`docs/conf.py` sets `nbsphinx_execute = "never"`, so the published page shows
the notebook without outputs and a reader sees results only by running it.
FAIL if any cell carries committed output.

## 10. Human prose voice & readability

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

Readability defects are separate from voice and also WARN — quote the line:

- **Dangling contrasts:** "order matters here for a *different* reason" when
  no first reason was stated; "unlike above" with nothing above.
- **Colliding labels:** two bold bullet labels in one cell that differ by a
  word ("Peak shape" / "Shape") and mean different things.
- **Possessives of code identifiers:** `` `compare_models`' job `` — rephrase
  with "of".
- **Paragraph-length bullets:** three modes, each a sentence or more, is a
  table (mode / what it computes / when to use it).
- **Heading hygiene:** filler words in section titles ("Baseline Example"),
  and mixed numbering styles (`### 1.1` next to `### 1.2.`) — the pre-pass
  flags the latter.
- **Double-numbered things:** a "spectrum #4" that the code addresses as
  `slice_index = 3`. Use the index the code uses, in prose and code alike.
- **Dense option paragraphs:** a paragraph that enumerates three or more
  options with their defaults is a table.
- **Long sentences:** two dash pairs or more than about 40 words in one
  sentence — split it. The pre-pass lists candidates; it holds a list item,
  and the bold lead-in that introduces one, to 60 words instead, because a
  labeled bullet is not a sentence and its defect is the paragraph-length
  bullet above.
- **Loose antecedents:** "once you've read it" two sentences after the noun.
  Name the thing.
- **Readable code cells:** notebook code is teaching code, not library code —
  it does **not** have to read like `src/`. One argument per line, each with
  its own comment, is the clearer form here even where the formatter would
  collapse it, and `pyproject.toml` excludes `examples/` from ruff for exactly
  that reason. Do not run the formatter over a notebook, and do not grade its
  quote style against `src/`. The one mechanical constraint is width: the
  pre-pass lists code lines over 88 characters, which side-scroll in the
  rendered docs.
- **Peer consistency:** style *is* graded — against the notebook's peers, which
  are the examples whose leading digit matches (`01`-`04` are one group, `10`-
  `12` the next). A decade is written as one lesson and should read in one
  voice, so the question is never "does this match the library" but "does this
  match the notebook next to it". The pre-pass compares the dominant quote
  character, the statement each peer reaches a shared module by (`import numpy
  as np` against `from numpy import array`), and the name a peer binds the same
  call's result to (`file` against `f`). A module this notebook simply does not
  import is not a divergence. A two-notebook decade has no majority, so the
pre-pass names the split rather than an outlier; a notebook inconsistent
with *itself* is always reported.

## 11. Method assumptions & failure modes

The notebook teaches the key assumption, degeneracy, or failure/convergence
diagnostic *where it bites* — not just the happy path. This is what separates a
recipe from an example that builds judgment. Typical traps: parameter
identifiability/degeneracy (what must be fixed vs fit), what a fit metric or
residual can and cannot tell you, sampler convergence (walkers, burn-in,
autocorrelation), kernel truncation, aliasing.

**Grade by the methods the notebook calls.** N/A is something you *establish*:
list the methods it actually calls, and for each ask whether a trap is recorded
— under "Known pitfalls" below, in the function's docstring, in the prose of
a sibling notebook that calls the same method (the IRF/baseline-contamination
trap is taught in `01` cell 8 and nowhere else), or in
`docs/design/`. N/A only when that list comes back empty, and say so *with the
list*; don't invent a caveat to avoid saying it. WARN when a method has a
recorded trap the notebook silently skips.

The enumeration is the work. `01` convolves an IRF (`gaussCONV` in
`models_time.yaml`, so the baseline-window pitfall below applies), pins every
parameter but one, and passes `try_ci=0` because the minimizer needs two free
parameters for confidence intervals — a limitation of that choice, carried in a
kwarg comment. Three calls, two recorded traps, one limitation: that is what an
N/A has to survive.

**A warning in the run settles it: not N/A.** A warning is a trap firing on
this notebook's own data, so the method that raised it has one by definition.
Criterion 1's `--dump` footer already sorts every warning into "key phrase
quoted in prose" and "not quoted" — anything in the second group is a trap the
notebook hit and left unexplained, and a subcycle-boundary `UserWarning` is the
recorded pitfall below arriving in person.

A caveat is actionable: give the rule or the number ("the baseline window must
end about 3σ before t0"), not "widen it only as much as the IRF allows". WARN
on a caveat the reader cannot act on. When the notebook deliberately takes the
road a pitfall below warns against (a narrow baseline window instead of a
longer pre-t0 axis), the prose says so.

A failure mode is shown with a control. The healthy configuration and the
broken one run side by side, differing in the one input that matters, with
the diagnostic that separates them in one table — and the diagnostics that
do *not* separate them named as such (an autocorrelation warning and an
acceptance fraction that look the same for a converged and an unconverged
chain). A single broken run with prose asserting what went wrong is the
"asserted, not demonstrated" WARN of criterion 5 in method clothing.

## 12. Claims match the run and the code

Read the executed copy (criterion 1) next to the prose; every factual sentence
must survive contact with the notebook's own output and with the codebase.
Build the claim ledger first: every sentence that states a number, a count, a
behaviour (raises, refuses, is optional, defaults to), a cross-reference, or an
API promise — in this notebook *and* in the sibling a `%run` preamble pulls in,
because a promise made there ("labels work anywhere a handle prefix does")
binds this notebook's code. Then verify each entry, in this order:

- **Cross-references land in the right place.** A `§N`, "above"/"below", or
  sibling-notebook reference points at the section that actually does the
  thing ("we set σ in §1" when `set_sigma` runs in §0). The pre-pass resolves
  each `§` to its header title so you can check placement, not just existence;
  a `§` with no header in this notebook is a sibling's section (name the
  notebook) or a stale number.
- **Numbers agree with the printout.** A position, width, count, or winner
  quoted in prose matches what the notebook prints and the truth constants
  ("kicks the peak to position 10" while the cell below prints a maximum of
  9.60). Quote both.
- **Measured values become vocabulary, not numbers.** A figure the notebook
  measured on its own dataset — a percentage, a ratio, a pull, a correlation,
  a runtime — is data-, machine- or seed-specific and will differ for the
  reader; one from an unseeded step drifts between runs of the same notebook.
  Say what the reader should conclude and let the printout carry the number:
  "agrees with", "noticeably wider", "under a minute". Numbers stay when they
  are *inputs* rather than outcomes: settings the code passes (`steps=500`),
  physical or literature constants, a synthetic dataset's truth constants and
  the fitted values criterion 2 pairs with them, rules of thumb that hold for
  any data (walkers ≥ 2 × dimension, acceptance 0.2–0.5), definitions (±1σ),
  and general formulas. Any other deterministic value matching a printout is
  allowed but not preferred — if the point survives without it, drop it and
  point at the cell. The pre-pass lists candidates only for a notebook with a
  stochastic step (a deterministic fit prints the same numbers for every
  reader), plus any runtime, which is machine-specific either way.
- **Counts match what they count.** "Three fit levels" introducing a four-row
  table; "two models" over three rows.
- **The notebook does what the prose says it does.** A sentence describing
  calls the notebook never makes ("only writes via `save_fits` /
  `export_fits`" in a notebook that never saves) is a FAIL. The pre-pass lists
  API names mentioned in prose but never called; Tips and Next-Steps pointers
  are fine, statements about *this* notebook are not.
- **Predictions are borne out, at the margin they claim.** "Lower on most
  slices" for a difference in the fourth decimal is not honest; "marginally
  lower" is. YAML comments count too.
- **One spelling per API name, and it exists.** `save_fit` in Next Steps and
  `save_fits` in the opening (both exist; one notebook uses one) — the pre-pass
  flags near-duplicate identifiers.
- **Meaningless output is called meaningless.** An r² that sums to ≈ N_slices
  in sum mode is not "informational"; say it carries no information (dropping
  the column is a code change — flag it, don't make it).
- **Behaviour claims are shown, not asserted.** An error, refusal, or warning
  the prose describes appears in a `try`/`except` cell, or the warning is left
  visible and explained — otherwise drop the sentence. The pre-pass counts such
  claims and whether any cell demonstrates one.
- **Detours signal API gaps.** Code that looks something up in a DataFrame to
  feed another call (`v.loc[label == ..., "handle"].item()` → `get(handle=...)`)
  is where a documented direct path has quietly failed. Try the direct call
  the docs promise; if it fails, report an API finding instead of polishing
  the detour.
- **Method semantics come from upstream, not intuition.** A sentence saying
  what a statistic *is* (`stderr` is a marginal width, not a conditional
  one), what a warning *means* (emcee's 50τ message says τ cannot be
  estimated, not that the chain is unconverged), or what a default *does*
  (`sigma_ini=None` starts the sampled noise scale at the fit's RMS
  residual) is checked against the upstream docstring
  or source — lmfit, emcee, scipy — and the library's own. A plausible
  paraphrase that the upstream text contradicts is a FAIL: it is the one
  kind of error a reader cannot catch from the notebook alone.
- **Diagnostics hit their own targets.** A gate or band the notebook itself
  defines (χ²_red ≈ 1, acceptance 0.2–0.5, sampled σ ≈ σ_est) reads inside
  that target in the executed copy, or the miss is derived with a number. A
  miss explained away in a sentence ("lands near 1.3 rather than exactly 1
  because the noise is Poisson") is a WARN: the explanation usually covers
  an estimator or setup error, and the notebook's own printout is the
  witness.

FAIL on a wrong cross-reference, a number contradicted by the notebook's own
output, a described call that never happens, an API name that does not
exist, or a method-semantics claim the upstream documentation contradicts.
WARN on the rest.

## 13. One name, one practice per thing

**Scope: this notebook and the source it names** — its own sections, its
YAMLs, and the docstrings of the functions those YAMLs use. Not sibling
notebooks. The docstring is the anchor, which is what makes the cross-notebook
comparison unnecessary: a notebook that disagrees with a sibling about what
`expFun` does is already failing here on its own terms, because it disagrees
with `expFun`. Check each notebook against the function, never against its
neighbours.

The same object, quantity, or behaviour has one name across markdown, code
comments, printed strings, YAML comments, and the docstrings of the registry
functions the notebook uses. Physics verbs are the usual offender: §0 calls
the dynamics a "kicked decay", §3 and both YAMLs call the same `expFun` an
"exponential rise", and the function's docstring says "jumps to A, decays
toward 0" — a reader pictures three different curves. Also check metric names
(one of `chi2_red` / "σ-calibrated χ²" / "the calibrated flavour"), section
labels used in prose vs. the roadmap, and winner/challenger naming. Open the
docstrings of the functions the YAMLs use and adopt their vocabulary. WARN per
inconsistent term, listing every place it appears.

The same goes for practice and category. A kwarg is not "optional on a
single-file archive" in one cell and passed with a justification in the next —
pick one and use it throughout. And a word means what it means: fit slots in a
project's history are not "variables in scope", a per-slice metric column is
not a "flavour". WARN per contradiction.

State the library already holds is read, not retyped. A window, index range,
count, or seed that a `File`/`Project` attribute or an earlier call carries
(`file.base_t_ind`, `file.e_lim`, the `n_free` a cell just computed) is read
from there; a literal that duplicates it (`file.time < -10` re-typing the
sibling's `time_stop=-10`, a `4 free` comment beside a computed `n_free`) is
a second source of truth that drifts silently when the first one moves.
WARN per retyped value.

## 14. Say it once — explain in prose, anchor in code

**Scope: one notebook.** Never compare two notebooks here. A notebook
restating what a sibling explains is **correct**: notebooks must stand alone,
readers arrive at whichever one is closest to their problem, and repetition in
a new setting is how material is learned. Criterion 3's documented handoff is
right for *data and execution*, which the machine performs; understanding has
to be local. Do not tell an author to cut an explanation because a sibling has
it. Two notebooks saying the same thing *differently enough to disagree* is
criterion 12's question, not this one's.

Within one notebook, each rationale, caveat, and idiom is *explained* once, in
its natural home: a
design rationale (why there is no `sigma=` kwarg) in Tips; the *why* of a step
in that section's prose; a non-obvious kwarg in a one-line code comment. Code
comments then *anchor*, not re-explain: a keyword or term of art from the prose
(`# pinned m: a distinct configuration, not a re-run`) is the intended way to
tie a line to the concept the reader just met, and never counts as a repeat.
Use the prose's exact term — a synonym breaks the association (see 13). The
defect is a second explanation: sentences restating reasoning already given.
Flag (WARN, with the cells):

- A code comment that re-explains the markdown cell above it (a nine-line
  comment on σ semantics under a paragraph that just gave them). A one-line
  echo of the paragraph's keyword is the fix, not a violation.
- A sentence-length phrase in three or more cells ("canonical ≈ 1 for a good
  fit", "earn its keep") — the pre-pass lists repeated 4-grams as INFO. A
  recurring one- or two-word term of art is criterion 13 working as intended.
- The same call executed twice for the same purpose (two
  `sbs_aggregation="sum"` cells with the same comment) — the pre-pass lists
  repeated top-level calls; a deliberate repeat (a reload that demonstrates
  the reload warning) is fine.
- The same caveat under two plots, worded slightly differently.
- The same idea twice in one sentence ("creates a distinct variant …:
  differing inputs produce a new entry").

Build the fact inventory: list the notebook's key facts (σ survives load; fits
never write to disk; reference-style `select=` is project-wide) with every cell
that states each, in any wording. A fact stated in more than two places is a
WARN wherever the third statement is.

**Tips does not count toward that total.** Criterion 7 requires a Tips section,
and a summary's job is to restate — so a Tips bullet *naming* a fact the
notebook already made is the same kind of anchor a code comment is. Count the
statements outside Tips and land the WARN on the third of those, never on the
Tips bullet. The exemption covers naming, not re-explaining: a Tips bullet that
gives the reasoning again instead of the conclusion is a finding like any
other, and a fact that already reaches three cells before Tips is a WARN at
that third cell whether or not Tips repeats it.

One more place to look:

- **Near-verbatim passages.** The pre-pass lists word runs shared by two
  cells of *this* notebook — an opening bullet re-appearing as a §2 bullet, an
  opening explanation re-appearing as a code comment.

A code comment longer than about three lines that restates markdown counts
here as well as under 6.

## 15. User-facing vocabulary

Test every noun the reader meets — in the prose, and in what the cells
actually put on screen: would a reader who has only read the public API docs
recognise it? Flag:

- **Private attributes** (`Project._fit_history`) in prose, and worse in code
  (`len(project._fit_history)`) — a tutorial that reaches into private state
  teaches the reader to do the same. WARN in code, INFO in prose; the pre-pass
  finds both.
- **Internal names the notebook never has to teach:** class names the reader
  never constructs (`SavedFitSlot`), hash fields (`fit_view_sha256`),
  design-doc jargon ("materialized", "axes provider", "comparability key").
  Say what the mechanism does for the reader ("`compare_models` refuses to
  rank fits made on different windows") instead of naming it. Names the API
  hands back (`slot` from `find()`) are fine.
- **The same names arriving through a repr.** A cell whose last expression is
  a call renders the returned object, so a bare `project.fit_2d(...)` puts
  `JointFitResult`, `SavedFitSlot`, `optimization_hash`, `parameter_map` and
  raw sha256 digests on the reader's screen just as surely as a sentence
  would — and unannounced, which is worse. Judge what the cell *prints*, not
  only what the prose says; assign the result or suppress it. (Criterion 9 is
  unaffected: the committed copy is stripped, so this is invisible until the
  reader runs it.)
- **Pet words standing in for a defined term:** "flavour" for a metric
  variant, "science" for a result, "neuter" for "remove the difference".

WARN per term, listing where it appears. Vocabulary a *loaded-archive* user
needs (`handle`, `label`) is not leakage — it is the API.

## Known pitfalls (carry-over lessons)

Hard-won gotchas worth re-checking when a criterion looks borderline:

- **`%%capture` path quoting (IPython 9.x):** a bare path in `%cd -q ../dir`
  tokenizes as a malformed number and crashes. Quote it.
- **Subcycle-boundary time samples:** generate synthetic multi-cycle data on
  the *reloaded* CSV axes, not the in-memory `np.arange` axes — boundary-exact
  samples flip subcycle assignment under `%.6e` rounding and bias the fit.
- **Baseline-window IRF contamination:** an IRF onset at `t0` leaks the
  convolved dynamics back into the baseline window; leave enough clean pre-t0
  spectra (extend the time axis) rather than narrowing the baseline window to
  a few points.
- **Prose that outlived a retune:** constants get tuned after the story is
  written — "kicks the peak to position 10" stays while the IRF smearing caps
  the printed maximum at 9.6. Re-read the §0 prose against the §0 printout
  every time a constant moves.
- **Bands tuned to runs:** unseeded MCMC and other stochastic steps land
  somewhere else every execution; a percentage band in prose that had to be
  widened after a second run is the symptom, and the fix is the vocabulary
  rule in criterion 12, not a wider band.
- **Grading the stripped source:** the committed notebook has no outputs, so a
  prose claim contradicted by a result is invisible until you execute it —
  hence criterion 1 keeps the executed copy.

## Summary

Print a table:

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 1 | Runs clean end-to-end | ... | ... |
| 2 | Truth-anchored | ... | ... |
| 3 | Self-contained directory | ... | ... |
| 4 | No surprise side-effects | ... | ... |
| 5 | Main message, narrative & roadmap-as-TOC | ... | ... |
| 6 | Deliberate kwargs & honest comments | ... | ... |
| 7 | Scope discipline | ... | ... |
| 8 | Commented YAMLs | ... | ... |
| 9 | Stripped outputs | ... | ... |
| 10 | Human prose voice & readability | ... | ... |
| 11 | Method assumptions & failure modes | ... | ... |
| 12 | Claims match the run and the code | ... | ... |
| 13 | One name, one practice per thing | ... | ... |
| 14 | Say it once, anchor in code | ... | ... |
| 15 | User-facing vocabulary | ... | ... |

Then list every WARN/FAIL with an actionable next step.
