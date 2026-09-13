"""Mechanical (scriptable) checks for an examples/fitting_workflows notebook.

Covers the statically-checkable parts of docs/ai/check-example.md: stripped
outputs (9), roadmap/TOC numbering (5), required files (3), committed truth (2),
removed config keys (4), side-effect artifacts (4), and prose-voice
candidates (10).
Prints a PASS / WARN / FAIL / INFO line per check — INFO marks a fact the agent
must resolve by reading (evidence, not a verdict). The judgment criteria
(1, 6, 7, 8, 11, plus the prose/message parts of 5 and 10) are graded by reading
the notebook. Does not execute the notebook.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

EXAMPLES_ROOT = Path("examples/fitting_workflows")

# Output dirs that a fit writes into (used to tell fit CSV/PNG dumps apart from
# committed `data/` inputs). `.fit.h5` is always a fit output, dir-independent.
ARTIFACT_DIR_HINTS = ("_export", "_fits", "winner_", "fit_results")

# Criterion 10 (INFO only): high-precision prose tells that are almost always
# slop in a technical notebook. Softer words (robust, comprehensive, powerful)
# are left to the model's read to avoid false positives on legitimate use.
SLOP_TERMS = (
    "delve",
    "leverage",
    "seamless",
    "seamlessly",
    "utilize",
    "showcase",
    "realm",
    "landscape",
    "testament",
    "underscore",
    "underscores",
    "effortless",
    "effortlessly",
    "worth noting",
    "dive in",
    "let's dive",
    "in this section we will",
    "in conclusion",
)
SLOP_TERM_RE = re.compile(r"(?i)\b(" + "|".join(SLOP_TERMS) + r")\b")
# "Four concrete payoffs:", "3 key steps", etc.
SLOP_LIST_RE = re.compile(
    r"(?i)\b(\d+|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(concrete|key|main|core|simple|handy)\s+\w+"
)


#
def not_gitignored(paths: list[Path]) -> list[Path]:
    """Drop paths git ignores (e.g. the `<name>_fits/` artifact trees)."""

    if not paths:
        return paths
    try:
        ignored = subprocess.run(
            ["git", "check-ignore", "--", *(str(p) for p in paths)],
            capture_output=True,
            text=True,
        ).stdout.split("\n")
    except FileNotFoundError:
        return paths
    ignored_set = {line for line in ignored if line}
    return [p for p in paths if str(p) not in ignored_set]


#
def resolve_example(arg: str) -> Path:
    """Resolve a path or NN_ prefix to an example directory."""

    p = Path(arg)
    if p.is_dir():
        return p
    matches = [m for m in sorted(EXAMPLES_ROOT.glob(f"{arg}*")) if m.is_dir()]
    matches = not_gitignored(matches)
    if not matches:
        sys.exit(f"No example directory matches {arg!r} under {EXAMPLES_ROOT}")
    if len(matches) > 1:
        names = ", ".join(m.name for m in matches)
        sys.exit(f"{arg!r} is ambiguous: {names}")
    return matches[0]


#
def check_stripped(nb_path: Path) -> tuple[str, str]:
    """Criterion 9 — committed notebook has no outputs / execution counts."""

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    dirty = []
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            dirty.append(f"cell {i} has {len(cell['outputs'])} output(s)")
        if cell.get("execution_count") is not None:
            dirty.append(f"cell {i} has execution_count")
    if dirty:
        return "FAIL", "; ".join(dirty[:4])
    return "PASS", "no committed outputs"


#
def markdown_sources(nb: dict) -> list[str]:
    """Return each markdown cell's source as a joined string."""

    out = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        src = cell.get("source", "")
        out.append("".join(src) if isinstance(src, list) else src)
    return out


#
def split_runs(nums: list[int]) -> list[list[int]]:
    """Split a flat number list into ascending runs (reset when n <= prev)."""

    runs: list[list[int]] = []
    cur: list[int] = []
    for n in nums:
        if cur and n <= cur[-1]:
            runs.append(cur)
            cur = []
        cur.append(n)
    if cur:
        runs.append(cur)
    return runs


#
def table_number_runs(cell: str) -> list[list[int]]:
    """Return ascending integer runs from any column of markdown tables.

    Lets a numbered first column in an overview table serve as the
    roadmap/TOC.
    """

    runs: list[list[int]] = []
    lines = cell.splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].lstrip().startswith("|"):
            i += 1
            continue
        block = []
        while i < len(lines) and lines[i].lstrip().startswith("|"):
            block.append(lines[i])
            i += 1
        # block[0] header, block[1] separator, block[2:] data rows.
        rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in block[2:]]
        if not rows:
            continue
        for col in range(max(len(r) for r in rows)):
            vals = [r[col] for r in rows if col < len(r)]
            if vals and all(re.fullmatch(r"\d+", v) for v in vals):
                runs.append([int(v) for v in vals])
    return runs


#
def check_roadmap_toc(nb_path: Path) -> tuple[str, str]:
    """Criterion 5 — opening roadmap numbers match the `## N` section numbers.

    The roadmap then doubles as a table of contents. The roadmap may be a
    numbered list or a numbered table column (e.g. an overview table). The
    opening cell may carry several numbered lists (prose sub-lists); a run
    matching the section numbers counts as the roadmap. A leading `## 0` setup
    section need not appear in the roadmap.
    """

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    md = markdown_sources(nb)
    if not md:
        return "WARN", "no markdown cells"
    headers = [
        int(m.group(1))
        for cell in md
        for line in cell.splitlines()
        if (m := re.match(r"^##\s+(\d+)\.?\s+\S", line))
    ]
    if not headers:
        return "WARN", "no numbered `## N` sections"
    start = headers[0]
    if start not in (0, 1) or headers != list(range(start, start + len(headers))):
        return "FAIL", f"`## N` sections not consecutive: {headers}"
    open_nums = [
        int(m.group(1))
        for line in md[0].splitlines()
        if (m := re.match(r"^\s*(\d+)\.\s", line))
    ]
    runs = split_runs(open_nums) + table_number_runs(md[0])
    # A leading `## 0` setup section is optional in the roadmap.
    targets = [headers]
    if headers[0] == 0:
        targets.append(headers[1:])
    if any(run in targets for run in runs):
        return "PASS", f"roadmap matches `## N` sections {headers}"
    best = max(runs, key=len) if runs else []
    return (
        "WARN",
        f"roadmap {best or 'none'} != sections {headers} — align the roadmap "
        "(list or numbered table column) to the section numbers, or confirm "
        "the list isn't the roadmap",
    )


#
def scan_prose_voice(nb_path: Path) -> tuple[str, str]:
    """Criterion 10 (INFO) — high-precision AI-slop candidates to read.

    Never a failure: the model judges criterion 10 by reading the cells. This
    only surfaces near-always-slop tokens so the reader knows where to look.
    """

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    hits: list[str] = []
    for ci, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "markdown":
            continue
        src = cell.get("source", "")
        text = "".join(src) if isinstance(src, list) else src
        for line in text.splitlines():
            for m in SLOP_TERM_RE.finditer(line):
                hits.append(f"cell {ci} '{m.group(1)}'")
            if m := SLOP_LIST_RE.search(line):
                hits.append(f"cell {ci} '{m.group(0)}'")
    if not hits:
        return "INFO", "no high-precision tells (still read cells for voice)"
    shown = "; ".join(hits[:5]) + (" …" if len(hits) > 5 else "")
    return "INFO", f"{len(hits)} candidate(s): {shown}"


#
def check_required_files(ex: Path) -> tuple[str, str]:
    """Criterion 3 — data/, model YAMLs, and project.yaml all present."""

    missing = []
    if not (ex / "data").is_dir():
        missing.append("data/")
    if not (ex / "project.yaml").is_file():
        missing.append("project.yaml")
    model_yamls = [y for y in ex.glob("*.yaml") if y.name != "project.yaml"]
    if not model_yamls:
        missing.append("model *.yaml")
    if missing:
        # INFO, not WARN: presence can't decide intent. A signposted-reuse
        # notebook legitimately omits some of these (inline data generation, a
        # %run preamble, or relative-path sibling data). The agent reads the
        # notebook and resolves this to PASS / N/A / FAIL.
        return (
            "INFO",
            "missing: "
            + ", ".join(missing)
            + " — resolve: documented reuse variant (inline data, %run "
            "preamble, relative-path sibling data) → PASS/N/A, else FAIL",
        )
    return "PASS", f"data/, project.yaml, {len(model_yamls)} model YAML(s)"


#
def check_truth(ex: Path) -> tuple[str, str]:
    """Criterion 2 — committed *_truth.yaml (synthetic, default variant)."""

    truth = list((ex / "data").glob("*_truth.yaml")) if (ex / "data").is_dir() else []
    if truth:
        return "PASS", ", ".join(t.name for t in truth)
    # INFO, not WARN: absence can't decide intent — a real-data or
    # inline-generated variant is legitimate. The agent resolves it.
    return (
        "INFO",
        "no *_truth.yaml — resolve: real-data or inline-generated variant?",
    )


#
def check_removed_config_keys(ex: Path) -> tuple[str, str]:
    """Criterion 4 — project.yaml carries no removed config keys.

    Fits never write to disk since v0.14.0; a leftover ``auto_export:`` /
    ``path_results:`` key makes ``Project()`` raise at load.
    """

    pj = ex / "project.yaml"
    if not pj.is_file():
        # INFO: a %run-preamble notebook inherits a sibling's config.
        return "INFO", "no project.yaml — resolve: %run preamble inherits config?"
    text = pj.read_text(encoding="utf-8")
    m = re.search(r"^\s*(auto_export|path_results)\s*:", text, re.MULTILINE)
    if m:
        return "FAIL", f"removed key '{m.group(1)}' present — Project() will raise"
    return "PASS", "no removed config keys"


#
def find_artifacts(ex: Path) -> list[Path]:
    """Fit-output files in or beside the example dir (never `data/` inputs).

    `.fit.h5` counts anywhere (always an output); `.csv` / `.png` count only
    under an output dir so committed `data/*.csv` inputs are not flagged.
    """

    found = set(ex.rglob("*.fit.h5"))
    for pat in ("*.csv", "*.png"):
        for p in ex.rglob(pat):
            low = str(p).lower()
            if "/data/" in low.replace("\\", "/"):
                continue
            if any(h in low for h in ARTIFACT_DIR_HINTS):
                found.add(p)
    # The gitignored `<name>_fits/` tree sits beside the example dir.
    sibling = ex.parent / f"{ex.name}_fits"
    if sibling.is_dir():
        for pat in ("*.fit.h5", "*.csv", "*.png"):
            found.update(sibling.rglob(pat))
    return sorted(found)


#
def check_artifacts(ex: Path) -> tuple[str, str]:
    """Criterion 4 — committed artifacts FAIL; untracked/gitignored ones INFO.

    Committed fit outputs pollute the repo (FAIL). Untracked/gitignored outputs
    are transient (left by a local run, or expected for an export-demo notebook)
    — surfaced as INFO so they are visible rather than a silent PASS.
    """

    arts = find_artifacts(ex)
    if not arts:
        return "PASS", "no fit artifacts in tree"
    try:
        tracked_out = subprocess.run(
            ["git", "ls-files", "--", *(str(a) for a in arts)],
            capture_output=True,
            text=True,
        ).stdout
        tracked = {line for line in tracked_out.splitlines() if line}
    except FileNotFoundError:
        tracked = set()
    committed = [a for a in arts if str(a) in tracked]
    if committed:
        listed = ", ".join(str(a) for a in committed[:4])
        return "FAIL", f"committed artifacts: {listed}"
    names = ", ".join(a.name for a in arts[:5]) + (" …" if len(arts) > 5 else "")
    return (
        "INFO",
        f"{len(arts)} untracked/gitignored artifact(s): {names} — expected "
        "after a local run or for an export-demo notebook; confirm gitignored",
    )


#
def all_example_dirs() -> list[Path]:
    """Every non-gitignored `NN_*` example dir (drops the `_fits` siblings)."""

    dirs = [d for d in sorted(EXAMPLES_ROOT.glob("[0-9]*")) if d.is_dir()]
    return not_gitignored(dirs)


#
def report_example(ex: Path) -> tuple[int, int]:
    """Print the mechanical report for one example; return (fails, warns)."""

    nb = ex / "example.ipynb"
    print(f"# Mechanical checks: {ex}")
    rows: list[tuple[str, str, str]] = []
    if nb.is_file():
        rows.append(("9  Stripped outputs", *check_stripped(nb)))
        rows.append(("5  Roadmap/TOC", *check_roadmap_toc(nb)))
        rows.append(("10 Prose voice", *scan_prose_voice(nb)))
    else:
        rows.append(("9  Stripped outputs", "FAIL", "no example.ipynb"))
    rows.append(("3  Required files", *check_required_files(ex)))
    rows.append(("2  Committed truth", *check_truth(ex)))
    rows.append(("4  Removed keys", *check_removed_config_keys(ex)))
    rows.append(("4  Artifacts", *check_artifacts(ex)))
    for name, status, detail in rows:
        print(f"{status:4} | {name:22} | {detail}")
    fails = sum(1 for _, s, _ in rows if s == "FAIL")
    warns = sum(1 for _, s, _ in rows if s == "WARN")
    infos = sum(1 for _, s, _ in rows if s == "INFO")
    print(
        f"# {fails} FAIL, {warns} WARN, {infos} INFO "
        "(INFO = a fact for the agent to resolve by reading, not a defect; "
        "judgment criteria 1,6,7,8,11 and the prose/message parts of 5 need the read)"
    )
    return fails, warns


#
def main() -> None:
    """Report on one example (argv[1]) or every example (no argument)."""

    if len(sys.argv) == 1:
        examples = all_example_dirs()
        if not examples:
            sys.exit(f"no example dirs found under {EXAMPLES_ROOT}")
        totals = [0, 0]
        for i, ex in enumerate(examples):
            if i:
                print()
            f, w = report_example(ex)
            totals[0] += f
            totals[1] += w
        print(f"\n# ALL {len(examples)} examples: {totals[0]} FAIL, {totals[1]} WARN")
    elif len(sys.argv) == 2:
        report_example(resolve_example(sys.argv[1]))
    else:
        sys.exit("usage: check_example_mechanics.py [example dir or NN_ prefix]")


if __name__ == "__main__":
    main()
