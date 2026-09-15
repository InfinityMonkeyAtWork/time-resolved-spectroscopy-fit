"""Mechanical (scriptable) checks for an examples/fitting_workflows notebook.
Covers the statically-checkable parts of docs/ai/check-example.md: notebook-
JSON schema (1), stripped outputs (9), roadmap/TOC numbering (5), required
files (3), committed truth (2), removed config keys (4), side-effect artifacts
(4), relative links (7), YAML comment wrapping (8), heading numbering style,
prose-voice candidates, long sentences, over-wide code lines and style shared
with same-decade peers (10), imports outside the import cell (6), `§` cross-
references, near-duplicate API names, prose-only names, undemonstrated
behaviour claims and measured-looking numbers quoted in prose (12), repeated
phrases, near-verbatim passages and repeated calls (14), and private-attribute
access (15).
Prints a PASS / WARN / FAIL / INFO line per check — INFO marks a fact the agent
must resolve by reading (evidence, not a verdict). The judgment criteria
(1, 6, 7, 8, 11, 13, plus the prose/message parts of 5, 10, 12, 14, 15) are
graded by reading the notebook. Does not execute the notebook; ``--dump
<executed.ipynb>`` prints an executed copy cell by cell with trimmed outputs so
prose can be read against results (criterion 12).
"""

import ast
import io
import json
import re
import subprocess
import sys
import tokenize
from functools import lru_cache
from pathlib import Path

# The script lives at <repo>/.claude/skills/check-example/, so every path is
# anchored to the repo rather than to the caller's cwd.
REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_ROOT = REPO_ROOT / "examples" / "fitting_workflows"
REGISTRY_DIR = REPO_ROOT / "src" / "trspecfit" / "functions"

Cell = tuple[int, str, str]  # (index, cell_type, source)

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
    "worth knowing",
    "worth mentioning",
    "worth pointing out",
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

# Criterion 14: words that never make a 4-gram distinctive on their own.
STOPWORDS = frozenset(
    "a an and are as at be but by for from has have if in into is it its of on "
    "or so than that the then this to was we were what when which will with you "
    "your not no one two per each here there".split()
)
# Criterion 12: qualifiers whose methods belong to third-party libraries.
FOREIGN_QUALIFIERS = frozenset({"np", "pd", "plt", "scipy", "os", "sys", "Path"})

LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
SECTION_REF_RE = re.compile(r"§\s*(\d+(?:\.\d+)*)")
NUMBERED_HEADER_RE = re.compile(r"^(#{2,4})\s+(\d+(?:\.\d+)*)(\.?)\s+(\S.*)$")
BACKTICK_RE = re.compile(r"`([^`\n]+)`")
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# `name(` inside a backtick span, with an optional qualifier
CALL_MENTION_RE = re.compile(
    r"(?:([A-Za-z_][A-Za-z0-9_]*)\.)?([A-Za-z_][A-Za-z0-9_]*)\s*\("
)
# `Class.attr` inside a backtick span
CLASS_ATTR_RE = re.compile(r"\b[A-Z][A-Za-z0-9_]*\.([a-z][A-Za-z0-9_]*)\b")
# Criterion 12: a backtick span holding prose math (`x0(z) = x0 + m·z`) has no
# calls in it; an arithmetic operator or a bare `=` outside a kwarg says so.
MATH_SPAN_RE = re.compile(r"[·×^√]|\s[=+]\s|\)\s*=")
# A name carrying a file extension is a filename, not a call.
FILE_EXT_RE = re.compile(r"\.(?:yaml|yml|ipynb|py|csv|h5|md|txt)\b")
# lmfit parameter names (`expFun_01_A`, `GLP_01_x0`) — data, never a call.
LMFIT_PAR_RE = re.compile(r"\w+_\d+_\w+")
PRIVATE_QUALIFIED_RE = re.compile(r"\w+\._[a-z]\w*")
# A glob puts a non-word char before the underscore, so `models_*_truth.yaml`
# read as a private attribute; exclude glob metacharacters and any match that
# carries a file extension.
PRIVATE_BARE_RE = re.compile(
    r"(?<![\w*?])(?:\w+\.)?_[a-z]\w*(?!\w*\.(?:yaml|yml|ipynb|py|csv|h5|md|txt))"
)
IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+[A-Za-z_]", re.MULTILINE)
WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9'’]*")
# --dump collapses lmfit report bodies, timing lines, and tqdm progress bars.
DUMP_COLLAPSE_RE = re.compile(r"^(\s{4}|\[\[|Results |Time |.*\d+%\|)")
# Markdown link targets — dropped before n-gram scans so `[x](../11_x/…)`
# does not read as a repeated phrase.
LINK_TARGET_RE = re.compile(r"\]\([^)]*\)")
FENCE_RE = re.compile(r"```.*?```", re.S)
# Criterion 12: prose that promises an error / refusal / warning.
BEHAVIOR_CLAIM_RE = re.compile(
    r"\b(raises?|raised|refuses?|refused|LookupError|ValueError|TypeError|"
    r"KeyError|RuntimeError|warns?|warning)\b"
)
DEMO_RE = re.compile(
    r"^\s*(try:|except\b|with\s+(?:pytest\.raises|warnings\.catch_warnings))", re.M
)
TIMING_CONTEXT_RE = re.compile(r"(?i)\b(runtime|run time|takes|elapsed)\b")
# --dump footer (criterion 1): output lines that carry a warning or error.
# lmfit prints emcee's autocorrelation notice as plain stdout ("...with
# caution..."), so the soft signals matter as much as the class names.
SIGNAL_RE = re.compile(
    r"(?i)(\b\w*warning\b|\b\w*error\b|\bwarn\b|\bcaution\b|\bdeprecated\b)"
)
# "…/confidence.py:356: UserWarning: <msg>", "WARNING:root:<msg>", "<Name>Error: <msg>"
SIGNAL_PREFIX_RE = re.compile(
    r"^.*?(?:\b\w*(?:Warning|Error)\b\s*:\s*|\b(?:WARNING|ERROR)(?::\w+)?:\s*)"
)
# A bare call on its own line is Python echoing the source of a warning
# (`warnings.warn(msg)`), not a second warning.
BARE_CALL_RE = re.compile(r"^[\w.]+\(.*\)$")
# A number with a time unit ("~30–40 s", "2 min"); the context word above
# must appear on the same line.
TIMING_VALUE_RE = re.compile(r"\d+\s*(?:s|secs?|seconds?|ms|mins?|minutes?|h|hours?)\b")
# Criterion 12 (INFO): numbers in prose that look measured on this dataset —
# percentages, multipliers, decimal σ distances ("−1.3σ"; integer "±1σ" is a
# definition) and approximate decimals ("lands near 1.3", "≈ 0.2–0.5"; a round
# "≈ 1" is a target, not a measurement). Rules of thumb and settings match
# too; the reader sorts them, the scan just lists.
INLINE_CODE_RE = re.compile(r"`[^`]*`")
# Criterion 12: the vocabulary rule exists because an unseeded step lands
# somewhere else every run. A notebook with no such step prints the same
# numbers for the reader, so the scan stays quiet there.
STOCHASTIC_STEP_RE = re.compile(
    r"\b(use_mc|emcee|MC\(|flatchain|np\.random|default_rng|bootstrap|"
    r"simulate_noisy|noise_type)\b"
)
MEASURED_VALUE_RE = re.compile(
    r"(?:[−-]?\d+(?:\.\d+)?\s*(?:%|percent\b|×|x\b|times\b))"
    r"|(?:[−-]?\d+\.\d+\s*σ)"
    r"|(?:(?:≈|~|\babout|\broughly|\bnear|\baround|\bwithin|\blands (?:at|near)|"
    r"\bcomes out (?:at|to))\s*[−-]?\d+\.\d+)"
)


#
def repo_rel(path: Path) -> str:
    """`path` as git prints it: relative to the repo root, forward slashes."""

    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


#
def valid_json(nb_path: Path) -> bool:
    """True if the file exists and parses as JSON (a corrupt sibling is skipped)."""

    try:
        json.loads(nb_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return True


#
def not_gitignored(paths: list[Path]) -> list[Path]:
    """Drop paths git ignores (e.g. the `<name>_fits/` artifact trees)."""

    if not paths:
        return paths
    try:
        ignored = subprocess.run(
            ["git", "check-ignore", "--", *(repo_rel(p) for p in paths)],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        ).stdout.split("\n")
    except FileNotFoundError:
        return paths
    ignored_set = {line for line in ignored if line}
    return [p for p in paths if repo_rel(p) not in ignored_set]


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
def load_cells(nb_path: Path) -> list[Cell]:
    """Return (index, cell_type, source) for every cell."""

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells: list[Cell] = []
    for i, cell in enumerate(nb.get("cells", [])):
        src = cell.get("source", "")
        text = "".join(src) if isinstance(src, list) else src
        cells.append((i, cell.get("cell_type", ""), text))
    return cells


#
def split_comment(line: str) -> tuple[str, str]:
    """Split a code line into (code, comment) at the first `#` outside quotes."""

    quote = ""
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch == "#":
            return line[:i], line[i + 1 :]
    return line, ""


#
def prose_chunks(cells: list[Cell]) -> list[tuple[int, str]]:
    """Markdown cells whole (link targets dropped); code cells as comment text."""

    out: list[tuple[int, str]] = []
    for ci, ctype, text in cells:
        if ctype == "markdown":
            out.append((ci, LINK_TARGET_RE.sub("]", text)))
        elif ctype == "code":
            comments = [split_comment(line)[1].strip() for line in text.splitlines()]
            joined = " ".join(c for c in comments if c)
            if joined:
                out.append((ci, joined))
    return out


#
def check_notebook_schema(nb_path: Path) -> tuple[str, str]:
    """Criterion 1 — the committed JSON validates against its declared nbformat.

    Cell ids arrived in nbformat 4.5, so a 4.4 file carrying one (or a 4.5 file
    missing one) makes `jupyter nbconvert` print `Notebook JSON is invalid`
    before a single cell runs. The exit code stays 0 and the `--dump` footer
    reads only cell outputs, so nothing else here would notice.
    """

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    major, minor = nb.get("nbformat", 4), nb.get("nbformat_minor", 0)
    cells = nb.get("cells", [])
    if (major, minor) >= (4, 5):
        missing = [i for i, c in enumerate(cells) if "id" not in c]
        if missing:
            return "FAIL", (
                f"nbformat {major}.{minor} requires a cell id; {len(missing)} "
                f"cell(s) have none: {missing[:6]}"
            )
    else:
        carried = [i for i, c in enumerate(cells) if "id" in c]
        if carried:
            return "FAIL", (
                f"nbformat {major}.{minor} predates cell ids (4.5); "
                f"{len(carried)} cell(s) carry one: {carried[:6]} — nbconvert "
                "will call the JSON invalid"
            )
    return "PASS", f"nbformat {major}.{minor} agrees with the cells' id usage"


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
        return "WARN", f"`## N` sections not consecutive: {headers}"
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
def check_heading_style(nb_path: Path) -> tuple[str, str]:
    """Criterion 10 — numbered headings use one style per level (`1.1` vs `1.2.`)."""

    seen: dict[int, dict[str, str]] = {}
    for _, ctype, text in load_cells(nb_path):
        if ctype != "markdown":
            continue
        for line in text.splitlines():
            m = NUMBERED_HEADER_RE.match(line)
            if not m:
                continue
            style = "dot" if m.group(3) else "nodot"
            seen.setdefault(len(m.group(1)), {}).setdefault(style, line.strip())
    mixed = [
        f"'{v['nodot'][:30]}' vs '{v['dot'][:30]}'"
        for v in seen.values()
        if len(v) == 2
    ]
    if mixed:
        return "WARN", "mixed numbering styles: " + "; ".join(mixed)
    return "PASS", "numbered headings use one style per level"


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
        return "PASS", "no high-precision tells (voice still needs the read)"
    shown = "; ".join(hits[:5]) + (" …" if len(hits) > 5 else "")
    return "INFO", f"{len(hits)} candidate(s): {shown}"


#
def check_relative_links(nb_path: Path, ex: Path) -> tuple[str, str]:
    """Criterion 7 — every relative markdown link resolves from the example dir."""

    broken: list[str] = []
    n = 0
    for ci, ctype, text in load_cells(nb_path):
        if ctype != "markdown":
            continue
        for m in LINK_RE.finditer(text):
            target = m.group(1)
            if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
                continue
            path = target.split("#", 1)[0]
            if not path:
                continue
            n += 1
            if not (ex / path).exists():
                broken.append(f"cell {ci} {target}")
    if broken:
        return "FAIL", "broken: " + "; ".join(broken[:4])
    return "PASS", f"{n} relative link(s) resolve"


#
def sections_in_force(cells: list[Cell]) -> list[tuple[str | None, str | None]]:
    """For each cell, the (`## N`, `### N.M`) numbers in force at that cell."""

    top: str | None = None
    sub: str | None = None
    out: list[tuple[str | None, str | None]] = []
    for _, ctype, text in cells:
        if ctype == "markdown":
            for line in text.splitlines():
                m = NUMBERED_HEADER_RE.match(line)
                if not m:
                    continue
                if len(m.group(1)) == 2:
                    top, sub = m.group(2), None
                else:
                    sub = m.group(2)
        out.append((top, sub))
    return out


#
def scan_section_refs(nb_path: Path) -> tuple[str, str]:
    """Criterion 12 (INFO) — resolve each `§N[.M]` to its header for the read.

    A reference that sits inside the section it names is marked as a
    self-reference. A `§` with no header here may point at a sibling
    notebook's section (a %run-preamble notebook does this legitimately) or
    be a stale number.
    """

    cells = load_cells(nb_path)
    headers: dict[str, str] = {}
    for _, ctype, text in cells:
        if ctype != "markdown":
            continue
        for line in text.splitlines():
            if m := NUMBERED_HEADER_RE.match(line):
                headers.setdefault(m.group(2), m.group(4).strip())
    in_force = sections_in_force(cells)
    refs: list[str] = []
    unresolved: list[str] = []
    for ci, _, text in cells:
        for m in SECTION_REF_RE.finditer(text):
            num = m.group(1)
            if num not in headers:
                unresolved.append(f"cell {ci} §{num}")
                continue
            note = " (self-reference)" if num in in_force[ci] else ""
            refs.append(f"cell {ci} §{num} → '{headers[num][:36]}'{note}")
    if not refs and not unresolved:
        return "PASS", "no § references"
    parts: list[str] = []
    if refs:
        shown = "; ".join(refs[:6]) + (" …" if len(refs) > 6 else "")
        parts.append(
            f"{len(refs)} reference(s) — confirm each names the section that "
            f"does the thing: {shown}"
        )
    if unresolved:
        parts.append(
            "no header in this notebook for: "
            + "; ".join(unresolved[:4])
            + " — a sibling's section (say which notebook) or a stale number?"
        )
    return "INFO", " | ".join(parts)


#
def api_identifiers(cells: list[Cell]) -> dict[str, set[int]]:
    """API-looking identifiers (inner underscore, 5+ chars) → cells using them.

    Markdown counts only inside backticks; code counts everywhere.
    """

    names: dict[str, set[int]] = {}
    for ci, ctype, text in cells:
        if ctype == "markdown":
            chunks = BACKTICK_RE.findall(text)
        elif ctype == "code":
            chunks = [text]
        else:
            continue
        for chunk in chunks:
            for ident in IDENT_RE.findall(chunk):
                if len(ident) >= 5 and "_" in ident.strip("_"):
                    names.setdefault(ident, set()).add(ci)
    return names


#
def scan_near_duplicate_identifiers(nb_path: Path) -> tuple[str, str]:
    """Criterion 12 — `save_fit` next to `save_fits`: one API, two spellings?"""

    names = api_identifiers(load_cells(nb_path))
    pairs: list[str] = []
    for name in sorted(names):
        if name.endswith("s") and name[:-1] in names:
            a, b = name[:-1], name
            cells_a, cells_b = sorted(names[a])[:3], sorted(names[b])[:3]
            pairs.append(f"{a} (cells {cells_a}) / {b} (cells {cells_b})")
    if not pairs:
        return "PASS", "no near-duplicate API names"
    return "INFO", f"{len(pairs)} pair(s) — one API or two?: " + "; ".join(pairs[:4])


#
@lru_cache(maxsize=1)
def registry_function_names() -> frozenset[str]:
    """Every function defined in ``src/trspecfit/functions/``.

    A YAML picks these by name, so prose naming one is a pointer to the
    registry, never a claim that this notebook calls it.
    """

    names: set[str] = set()
    for src in REGISTRY_DIR.glob("*.py"):
        names.update(re.findall(r"^def (\w+)", src.read_text(encoding="utf-8"), re.M))
    return frozenset(names)


#
def scan_prose_only_names(nb_path: Path) -> tuple[str, str]:
    """Criterion 12 — backticked API names the prose uses but no code cell does.

    A statement that *this* notebook does something with such a name is a
    FAIL under 12; Tips / Next-Steps pointers and column names are fine.

    Four shapes are dropped before reporting, because none can carry a
    "this notebook calls it" claim: prose math (``exp(-z/tau)``), filenames
    (``generate_data.ipynb``), registry functions the YAMLs select by name
    (``pExpDecay``), and lmfit parameter names (``expFun_01_A``).
    """

    cells = load_cells(nb_path)
    code = "\n".join(text for _, ctype, text in cells if ctype == "code")
    registry = registry_function_names()
    names: dict[str, set[int]] = {}
    for ci, ctype, text in cells:
        if ctype != "markdown":
            continue
        for span in BACKTICK_RE.findall(text):
            if MATH_SPAN_RE.search(span):
                continue
            for m in CALL_MENTION_RE.finditer(span):
                if m.group(1) in FOREIGN_QUALIFIERS:
                    continue
                if span[m.end() : m.end() + 2].strip().startswith("-"):
                    continue  # `exp(-z/tau)` — a minus opens math, not a kwarg
                names.setdefault(m.group(2), set()).add(ci)
            for m in CLASS_ATTR_RE.finditer(span):
                names.setdefault(m.group(1), set()).add(ci)
            for m in IDENT_RE.finditer(span):
                ident = m.group(0)
                if not (ident[0].islower() and len(ident) >= 5):
                    continue
                if "_" not in ident.strip("_"):
                    continue
                if FILE_EXT_RE.match(span[m.end() :]):
                    continue  # `data/generate_data.ipynb`
                names.setdefault(ident, set()).add(ci)
    names = {
        name: cells_seen
        for name, cells_seen in names.items()
        if name not in registry and not LMFIT_PAR_RE.fullmatch(name)
    }
    missing = sorted(
        name for name in names if not re.search(rf"\b{re.escape(name)}\b", code)
    )
    if not missing:
        return "PASS", "every backticked API name also appears in code"
    shown = ", ".join(
        f"{name}({','.join(str(c) for c in sorted(names[name])[:3])})"
        for name in missing[:12]
    ) + (" …" if len(missing) > 12 else "")
    return (
        "INFO",
        f"{len(missing)} name(s) in prose only (name(cells)): {shown} — a claim "
        "that this notebook *does* something with one is a FAIL under 12; "
        "pointers and column names are fine",
    )


#
def scan_private_access(nb_path: Path) -> tuple[str, str]:
    """Criterion 15 — `obj._private` in code (WARN) or in prose/comments (INFO)."""

    code_hits: list[str] = []
    prose_hits: list[str] = []
    for ci, ctype, text in load_cells(nb_path):
        if ctype == "markdown":
            for span in BACKTICK_RE.findall(text):
                prose_hits += [
                    f"cell {ci} {m.group(0)}" for m in PRIVATE_BARE_RE.finditer(span)
                ]
        elif ctype == "code":
            for line in text.splitlines():
                code, comment = split_comment(line)
                code_hits += [
                    f"cell {ci} {m.group(0)}"
                    for m in PRIVATE_QUALIFIED_RE.finditer(code)
                ]
                prose_hits += [
                    f"cell {ci} {m.group(0)}" for m in PRIVATE_BARE_RE.finditer(comment)
                ]
    if code_hits:
        detail = "private attribute in code: " + "; ".join(code_hits[:3])
        if prose_hits:
            detail += f"; also named in prose ({len(prose_hits)}x)"
        return "WARN", detail
    if prose_hits:
        return "INFO", "private attribute named in prose: " + "; ".join(prose_hits[:4])
    return "PASS", "no private attributes"


#
def scan_imports(nb_path: Path) -> tuple[str, str]:
    """Criterion 6 — imports outside the first importing cell (INFO)."""

    importing = [
        ci
        for ci, ctype, text in load_cells(nb_path)
        if ctype == "code" and IMPORT_RE.search(text)
    ]
    if len(importing) <= 1:
        return "PASS", "all imports in one cell"
    return (
        "INFO",
        f"imports in cells {importing} — confirm each import after cell "
        f"{importing[0]} is deliberate (%run preamble, optional dependency)",
    )


#
def scan_repeated_phrases(
    nb_path: Path, n: int = 4, min_cells: int = 3
) -> tuple[str, str]:
    """Criterion 14 (INFO) — word 4-grams that recur in three or more cells.

    Prose = markdown cells plus code comments. Pointers only: the model decides
    whether a repeat is padding or a deliberate refrain.
    """

    grams: dict[tuple[str, ...], set[int]] = {}
    for ci, text in prose_chunks(load_cells(nb_path)):
        words = [w.lower() for w in WORD_RE.findall(text)]
        seen: set[tuple[str, ...]] = set()
        for i in range(len(words) - n + 1):
            g = tuple(words[i : i + n])
            if g in seen or all(w in STOPWORDS or len(w) <= 2 for w in g):
                continue
            seen.add(g)
            grams.setdefault(g, set()).add(ci)
    hits = sorted(
        ((g, cells) for g, cells in grams.items() if len(cells) >= min_cells),
        key=lambda item: (-len(item[1]), item[0]),
    )
    reported: list[tuple[tuple[str, ...], set[int]]] = []
    for g, cells in hits:
        # Skip a gram that merely shifts an already-reported one by a word.
        if any(len(set(g) & set(r)) >= n - 1 and cells == rc for r, rc in reported):
            continue
        reported.append((g, cells))
    if not reported:
        return "PASS", f"no phrase repeated across {min_cells}+ cells"
    shown = "; ".join(
        f"'{' '.join(g)}' (cells {sorted(cells)})" for g, cells in reported[:6]
    ) + (" …" if len(reported) > 6 else "")
    return (
        "INFO",
        f"{len(reported)} phrase(s) in {min_cells}+ cells: {shown} — a recurring "
        "term of art is fine, a recurring explanation is not",
    )


#
def top_level_calls(source: str) -> set[str]:
    """Normalized top-level call expressions in a code cell (magics dropped)."""

    lines = [ln for ln in source.splitlines() if not ln.lstrip().startswith(("%", "!"))]
    try:
        tree = ast.parse("\n".join(lines))
    except SyntaxError:
        return set()
    calls: set[str] = set()
    for node in tree.body:
        value = getattr(node, "value", None)
        if isinstance(node, ast.Expr | ast.Assign) and isinstance(value, ast.Call):
            if ast.unparse(value.func).split(".")[-1] == "print":
                continue
            calls.add(ast.unparse(value))
    return calls


#
def scan_repeated_calls(nb_path: Path) -> tuple[str, str]:
    """Criterion 14 (INFO) — the same top-level call in more than one cell."""

    seen: dict[str, list[int]] = {}
    for ci, ctype, text in load_cells(nb_path):
        if ctype != "code":
            continue
        for call in top_level_calls(text):
            seen.setdefault(call, []).append(ci)
    reps = {c: cells for c, cells in seen.items() if len(cells) >= 2}
    if not reps:
        return "PASS", "no top-level call repeated across cells"
    shown = "; ".join(
        f"{call[:60]} (cells {cells})" for call, cells in sorted(reps.items())[:5]
    ) + (" …" if len(reps) > 5 else "")
    return (
        "INFO",
        f"{len(reps)} call(s) repeated: {shown} — confirm each repeat is "
        "deliberate (a reload demo), not the same table twice",
    )


#
def words_of(text: str) -> list[str]:
    """Lower-cased word tokens with markdown link targets dropped."""

    return [w.lower() for w in WORD_RE.findall(LINK_TARGET_RE.sub("]", text))]


#
def maximal_runs(tokens: list[str], shared: set[tuple[str, ...]], n: int) -> list[str]:
    """Longest token runs in which every consecutive n-gram is in ``shared``."""

    runs: list[str] = []
    i = 0
    while i + n <= len(tokens):
        if tuple(tokens[i : i + n]) not in shared:
            i += 1
            continue
        j = i
        while j + n <= len(tokens) and tuple(tokens[j : j + n]) in shared:
            j += 1
        runs.append(" ".join(tokens[i : j + n - 1]))
        i = j
    return runs


#
def scan_near_verbatim(nb_path: Path, n: int = 8) -> tuple[str, str]:
    """Criterion 14 (INFO) — an 8-word run that appears in two or more cells.

    Catches the paragraph re-used as a bullet elsewhere and the prose
    re-explained in a code comment. Prose = markdown plus code comments.
    """

    toks = {ci: words_of(text) for ci, text in prose_chunks(load_cells(nb_path))}
    grams: dict[tuple[str, ...], set[int]] = {}
    for ci, words in toks.items():
        for g in {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}:
            grams.setdefault(g, set()).add(ci)
    shared = {g for g, cells in grams.items() if len(cells) >= 2}
    if not shared:
        return "PASS", f"no {n}-word run appears in two cells"
    found: dict[str, set[int]] = {}
    for ci, words in toks.items():
        for run in maximal_runs(words, shared, n):
            found.setdefault(run, set()).add(ci)
    items = sorted(found.items(), key=lambda kv: (-len(kv[0]), kv[0]))
    shown = "; ".join(
        f"'{run[:70]}…' (cells {sorted(cells)})" for run, cells in items[:4]
    )
    return (
        "INFO",
        f"{len(items)} passage(s) shared by two+ cells — re-explanation or "
        f"anchor?: {shown}",
    )


#
def scan_behavior_claims(nb_path: Path) -> tuple[str, str]:
    """Criterion 12 (INFO) — errors/refusals/warnings the prose promises.

    Such a claim is demonstrated by a try/except cell (or a visible, explained
    warning) or it goes.
    """

    cells = load_cells(nb_path)
    claims: list[int] = []
    for ci, ctype, text in cells:
        if ctype != "markdown":
            continue
        if BEHAVIOR_CLAIM_RE.search(FENCE_RE.sub(" ", text)):
            claims.append(ci)
    if not claims:
        return "PASS", "no behaviour claims to demonstrate"
    demo = any(ctype == "code" and DEMO_RE.search(text) for _, ctype, text in cells)
    verdict = (
        "a try/except cell exists — confirm each claim is one it shows"
        if demo
        else "no try/except cell demonstrates any of them (a warning left "
        "visible in the run counts too: see the --dump footer)"
    )
    return "INFO", f"error/refusal/warning claimed in cell(s) {claims}; {verdict}"


#
def scan_measured_values(nb_path: Path) -> tuple[str, str]:
    """Criterion 12 (INFO) — numbers in prose that look measured on this data.

    A percentage, multiplier, decimal σ distance, approximate value or point
    timing quoted in markdown or a code comment is either a setting, a rule of
    thumb, a definition or a general formula (fine) or a value measured on this
    dataset — which differs for the reader's data and drifts between runs when
    the step is stochastic, so it should be vocabulary ("agrees with",
    "noticeably wider") with the printout carrying the number. Inline code
    spans are skipped: `steps=500` is a setting by construction.
    """

    cells = load_cells(nb_path)
    code = "\n".join(text for _, ctype, text in cells if ctype == "code")
    # A deterministic notebook prints the same numbers for the reader, so its
    # quoted values are verifiable rather than drifting — and criterion 2 asks
    # for the fitted-vs-truth ones. A runtime is machine-specific either way.
    stochastic = bool(STOCHASTIC_STEP_RE.search(code))
    hits: list[str] = []
    for ci, text in prose_chunks(cells):
        body = INLINE_CODE_RE.sub(" ", FENCE_RE.sub(" ", text))
        for line in body.splitlines():
            if line.lstrip().startswith("#"):  # markdown heading
                continue
            found = (
                [m.group(0).strip() for m in MEASURED_VALUE_RE.finditer(line)]
                if stochastic
                else []
            )
            if TIMING_CONTEXT_RE.search(line) and (m := TIMING_VALUE_RE.search(line)):
                found.append(m.group(0))
            for f in found:
                hits.append(f"cell {ci} '{f}'")
    if not hits:
        return "PASS", (
            "no measured-looking numbers in prose or comments"
            if stochastic
            else "no stochastic step and no runtime claim — quoted numbers are "
            "reproducible for the reader"
        )
    shown = "; ".join(hits[:6]) + (" …" if len(hits) > 6 else "")
    return "INFO", (
        f"{len(hits)} number(s) quoted in prose — vocabulary unless an input "
        f"(setting, physical or truth constant, rule of thumb, definition, "
        f"formula): {shown}"
    )


#
def scan_long_sentences(
    nb_path: Path, max_words: int = 40, max_bullet_words: int = 60
) -> tuple[str, str]:
    """Criterion 10 (INFO) — sentences over ``max_words`` or with two+ dashes.

    A bullet legitimately packs a labeled definition into one long line, so
    list items are held to ``max_bullet_words`` instead — which is criterion
    10's *paragraph-length bullet* defect, not its long-sentence one. Link
    targets are stripped first: a relative path is not prose.
    """

    hits: list[tuple[int, int, int, str]] = []
    for ci, ctype, text in load_cells(nb_path):
        if ctype != "markdown":
            continue
        body = LINK_TARGET_RE.sub("]", FENCE_RE.sub(" ", text))
        for line in body.splitlines():
            stripped = line.lstrip()
            if stripped.startswith(("|", "#")):
                continue
            # a list marker, or a bold lead-in label ending in ':' that
            # introduces one — neither is a sentence
            is_item = bool(re.match(r"(?:[-*+>]|\d+[.)])\s", stripped)) or (
                stripped.rstrip().endswith(":")
            )
            limit = max_bullet_words if is_item else max_words
            # `[` belongs in the lookahead: a sentence that opens with a
            # markdown link would otherwise never split from the one before it.
            for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z`*\"'(\[])", line):
                n_words = len(WORD_RE.findall(sent))
                dashes = sent.count("—")
                if n_words >= limit or (not is_item and dashes >= 2 and n_words >= 28):
                    hits.append((n_words, dashes, ci, sent.strip()[:70]))
    if not hits:
        return "PASS", (
            f"no sentence over {max_words} words ({max_bullet_words} in a list "
            "item) or with two dash pairs"
        )
    hits.sort(reverse=True)
    shown = "; ".join(
        f"cell {ci} ({w} words, {d} dashes): '{s}…'" for w, d, ci, s in hits[:4]
    )
    return "INFO", f"{len(hits)} long sentence(s): {shown}"


#
def scan_code_width(nb_path: Path, width: int = 88) -> tuple[str, str]:
    """Criterion 10 (INFO) — code lines too wide to render without scrolling.

    Notebook code is teaching code: one argument per line, annotated, is the
    point, so the formatter's idea of layout does not apply here (ruff excludes
    ``examples/`` for the same reason). Width still does — a long line
    side-scrolls in the rendered docs. Magics are skipped.
    """

    wide: list[str] = []
    for ci, ctype, text in load_cells(nb_path):
        if ctype != "code":
            continue
        for ln in text.splitlines():
            if ln.lstrip().startswith(("%", "!")):
                continue
            if len(ln.rstrip()) > width:
                wide.append(f"cell {ci} ({len(ln.rstrip())} chars): {ln.strip()[:50]}…")
    if not wide:
        return "PASS", f"no code line over {width} characters"
    shown = "; ".join(wide[:4]) + (" …" if len(wide) > 4 else "")
    return (
        "INFO",
        f"{len(wide)} code line(s) over {width} characters — they side-scroll "
        f"in the rendered docs; wrap the comment or the call: {shown}",
    )


#
# A top-level `name = callee(...)` binding; the callee may be dotted.
BINDING_RE = re.compile(r"^(\w+)\s*=\s*([\w.]+)\(", re.M)


#
def peer_examples(ex: Path) -> list[Path]:
    """Examples whose leading digit matches this one — its decade, minus itself."""

    return [
        d
        for d in all_example_dirs()
        if d.name[0] == ex.name[0]
        and d.name != ex.name
        and valid_json(d / "example.ipynb")
    ]


#
def code_string_quotes(source: str) -> tuple[int, int]:
    """(single, double) quoted string literals in one code cell.

    Tokenized, not regex-matched: an apostrophe in a comment and a quote inside
    a string both look like quoting to a regex and neither is a choice the
    author made. Magics are dropped, and a triple-quoted block is not a
    quote-style decision so it does not count.
    """

    lines = [ln for ln in source.splitlines() if not ln.lstrip().startswith(("%", "!"))]
    stream = io.StringIO("\n".join(lines) + "\n")
    fstring_start = getattr(tokenize, "FSTRING_START", -1)
    single = double = 0
    try:
        for tok in tokenize.generate_tokens(stream.readline):
            if tok.type == tokenize.STRING:
                body = tok.string.lstrip("rbuRBU")
            elif tok.type == fstring_start:
                body = tok.string.lstrip("rfRF")
            else:
                continue
            if body.startswith(("'''", '"""')):
                continue
            if body.startswith("'"):
                single += 1
            elif body.startswith('"'):
                double += 1
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass  # a partial count still says which quote the author reaches for
    return single, double


#
def quote_profile(nb_path: Path) -> tuple[int, int]:
    """(single, double) quoted literals across a notebook's code cells."""

    single = double = 0
    for _, ctype, text in load_cells(nb_path):
        if ctype != "code":
            continue
        s, d = code_string_quotes(text)
        single += s
        double += d
    return single, double


#
def notebook_code(nb_path: Path) -> str:
    """A notebook's code cells joined into one module, magics dropped."""

    return "\n".join(
        ln
        for _, ctype, text in load_cells(nb_path)
        if ctype == "code"
        for ln in text.splitlines()
        if not ln.lstrip().startswith(("%", "!"))
    )


#
def import_bindings(nb_path: Path) -> dict[str, set[str]]:
    """Top-level module a notebook imports → the statement(s) reaching it.

    Keyed on the module rather than the name it binds: a peer writing ``import
    numpy as np`` and one writing ``from numpy import array`` both reach numpy,
    two different ways, and that is the divergence worth naming. Keying on the
    bound name would file them under `np` and `array` and see nothing.
    """

    try:
        tree = ast.parse(notebook_code(nb_path))
    except SyntaxError:
        return {}
    bound: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound.setdefault(alias.name.split(".")[0], set()).add(ast.unparse(node))
        elif isinstance(node, ast.ImportFrom) and node.module:
            bound.setdefault(node.module.split(".")[0], set()).add(ast.unparse(node))
    return bound


#
def call_bindings(nb_path: Path) -> dict[str, set[str]]:
    """Callee → the names this notebook binds its result to, top level only."""

    bound: dict[str, set[str]] = {}
    for _, ctype, text in load_cells(nb_path):
        if ctype != "code":
            continue
        for name, callee in BINDING_RE.findall(text):
            bound.setdefault(callee, set()).add(name)
    return bound


#
def scan_peer_consistency(nb_path: Path, ex: Path) -> tuple[str, str]:
    """Criterion 10 (INFO) — house style this notebook does not share with its peers.

    Peers are the examples whose leading digit matches (`01`-`04` are one
    group, `10`-`12` the next): a decade is written as one lesson and should
    read in one voice. The comparison is against those peers and never against
    ``src/`` — notebook code is teaching code and is not held to library style
    (see "Readable code cells"). Three comparisons: the dominant quote
    character, an import every peer carries but this notebook lacks, and a call
    whose result a peer binds to an unrelated name.
    """

    peers = [p for p in peer_examples(ex) if (p / "example.ipynb").is_file()]
    if not peers:
        return "PASS", "no peer example shares this notebook's leading digit"
    peer_names = ", ".join(p.name for p in peers)
    notes: list[str] = []

    # The decade's own quoting convention — ruff has no say here.
    single, double = quote_profile(nb_path)
    peer_quotes = [(p, quote_profile(p / "example.ipynb")) for p in peers]
    dominant = {"'" if s > d else '"' for _, (s, d) in peer_quotes if s != d}
    mine = "'" if single > double else '"' if double > single else None
    if mine and len(dominant) == 1 and mine not in dominant:
        other = next(iter(dominant))
        if len(peers) >= 2:
            notes.append(
                f"prefers {mine} ({single} vs {double}) where every peer "
                f"prefers {other}"
            )
        else:
            # A single peer is not a majority, so name the split rather than an
            # outlier — otherwise a two-notebook decade can never be compared
            # at all, and each notebook indicts the other.
            peer, (ps, pd) = peer_quotes[0]
            notes.append(
                f"the decade has no shared quote convention — {mine} here "
                f"({single} vs {double}), {other} in {peer.name} ({ps} vs {pd})"
            )
    total = single + double
    if total >= 8 and min(single, double) / total >= 0.25:
        notes.append(f"quoting is mixed within this notebook ({single} vs {double})")

    # A module the whole decade agrees on, reached differently here. A module
    # this notebook does not import at all is not a divergence — declining to
    # import what you do not need is correct. Needs a majority, so a
    # two-notebook decade skips it.
    if len(peers) >= 2:
        peer_imports = [import_bindings(p / "example.ipynb") for p in peers]
        common = set.intersection(*(set(b) for b in peer_imports))
        ours = import_bindings(nb_path)
        for module in sorted(common):
            theirs = set().union(*(b[module] for b in peer_imports))
            if len(theirs) > 1 or module not in ours or ours[module] == theirs:
                continue
            notes.append(
                f"peers reach {module} by `{next(iter(theirs))}`; "
                f"here by `{'`, `'.join(sorted(ours[module]))}`"
            )

    # The same call, a different name for its result.
    mine_bound = call_bindings(nb_path)
    renamed: dict[str, str] = {}
    for p in peers:
        theirs = call_bindings(p / "example.ipynb")
        for callee, my_names in mine_bound.items():
            their_names = theirs.get(callee)
            if their_names and not (my_names & their_names):
                renamed[callee] = (
                    f"{callee}(...) is {'/'.join(sorted(my_names))} here and "
                    f"{'/'.join(sorted(their_names))} in {p.name}"
                )
    notes.extend(renamed[k] for k in sorted(renamed))

    if not notes:
        return "PASS", f"reads like its peers ({peer_names})"
    return "INFO", f"vs {peer_names} — " + "; ".join(notes)


#
def is_long_comment(lines: list[str], i: int) -> bool:
    """True when line ``i`` is a YAML comment of more than six words."""

    return (
        0 <= i < len(lines)
        and lines[i].lstrip().startswith("#")
        and len(WORD_RE.findall(lines[i])) > 6
    )


#
def scan_yaml_comments(ex: Path, width: int = 88) -> tuple[str, str]:
    """Criterion 8 (INFO) — YAML comment lines that run long or wrap as orphans."""

    hits: list[str] = []
    for y in sorted(ex.glob("*.yaml")):
        lines = y.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if not line.lstrip().startswith("#"):
                continue
            if len(line) > width:
                hits.append(f"{y.name}:{i + 1} {len(line)} chars")
            n_words = len(WORD_RE.findall(line))
            orphan = (
                0 < n_words <= 4
                and "---" not in line
                and is_long_comment(lines, i - 1)
                and is_long_comment(lines, i + 1)
            )
            if orphan:
                hits.append(f"{y.name}:{i + 1} orphan '{line.strip()[:30]}'")
    if not hits:
        return "PASS", "YAML comments wrap cleanly"
    return "INFO", f"{len(hits)} YAML comment line(s): " + "; ".join(hits[:5])


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
            ["git", "ls-files", "--", *(repo_rel(a) for a in arts)],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        ).stdout
        tracked = {line for line in tracked_out.splitlines() if line}
    except FileNotFoundError:
        tracked = set()
    committed = [a for a in arts if repo_rel(a) in tracked]
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
def trim_stream(text: str) -> str:
    """Collapse lmfit report bodies, timing lines, and progress bars."""

    kept: list[str] = []
    collapsed = 0
    for line in text.replace("\r", "\n").splitlines():
        if not line.strip():
            continue
        if DUMP_COLLAPSE_RE.match(line) or "it/s]" in line or "s/it]" in line:
            collapsed += 1
            continue
        kept.append(line)
    if collapsed:
        kept.append(f"[… {collapsed} report/progress line(s) collapsed]")
    return "\n".join(kept)


#
def trim_stderr(text: str) -> str:
    """Drop progress bars from a stderr stream; keep every warning line whole."""

    kept: list[str] = []
    bars = 0
    for line in text.replace("\r", "\n").splitlines():
        if not line.strip():
            continue
        if "%|" in line or "it/s]" in line or "s/it]" in line:
            bars += 1
            continue
        kept.append(line)
    if bars:
        kept.append(f"[… {bars} progress line(s) collapsed]")
    return "\n".join(kept)


#
def signal_lines(raw: str) -> list[str]:
    """Warning/error messages in one output stream, prefixes stripped."""

    found: list[str] = []
    for line in raw.replace("\r", "\n").splitlines():
        line = line.strip()
        if not line or "%|" in line or BARE_CALL_RE.match(line):
            continue
        if not SIGNAL_RE.search(line):
            continue
        found.append(SIGNAL_PREFIX_RE.sub("", line, count=1).strip() or line)
    return found


#
def mentioned(message: str, corpus: set[tuple[str, ...]], n: int) -> bool:
    """True when the message's first ``n`` significant words occur in the prose."""

    words = [w for w in words_of(message) if len(w) > 1][:n]
    return len(words) >= min(n, 3) and tuple(words) in corpus


#
def print_warning_footer(
    signals: list[tuple[int, str]], errors: list[tuple[int, str]], cells: list[Cell]
) -> None:
    """Criterion 1 evidence: each warning sorted by whether the prose mentions it."""

    print("\n===== warnings / errors in executed outputs =====")
    if not signals and not errors:
        print("none")
        return
    n = 5
    corpus: set[tuple[str, ...]] = set()
    for _, text in prose_chunks(cells):
        words = [w for w in words_of(text) if len(w) > 1]
        for k in (3, 4, 5):
            corpus.update(tuple(words[i : i + k]) for i in range(len(words) - k + 1))
    counts: dict[tuple[int, str], int] = {}
    for key in signals:
        counts[key] = counts.get(key, 0) + 1
    explained = [k for k in counts if mentioned(k[1], corpus, n)]
    unexplained = [k for k in counts if k not in explained]
    for title, keys in (
        ("key phrase quoted in prose or a comment", explained),
        (
            "key phrase not quoted — explained in other words (say where), "
            "or a criterion 1 WARN",
            unexplained,
        ),
    ):
        if keys:
            print(f"{title}:")
            for ci, msg in keys:
                times = f" (x{counts[(ci, msg)]})" if counts[(ci, msg)] > 1 else ""
                print(f"  cell {ci}: {msg[:110]}{times}")
    if errors:
        print("errors — criterion 1 FAIL:")
        for ci, msg in errors:
            print(f"  cell {ci}: {msg[:110]}")


#
def dump_executed(nb_path: Path, max_chars: int = 2500) -> None:
    """Print an executed notebook cell by cell with trimmed outputs.

    Figures and lmfit ``Parameters`` reprs collapse to one line; stderr keeps
    every warning line whole and drops only progress bars, and error outputs
    are kept whole, so criterion 1 can read them. Everything else is the
    evidence for criterion 12. A footer lists every warning/error line and
    whether the prose mentions it.
    """

    if not valid_json(nb_path):
        sys.exit(f"{nb_path}: notebook JSON does not parse")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = load_cells(nb_path)
    signals: list[tuple[int, str]] = []
    errors: list[tuple[int, str]] = []
    for ci, cell in enumerate(nb.get("cells", [])):
        src = cell.get("source", "")
        text = "".join(src) if isinstance(src, list) else src
        print(f"\n===== cell {ci} [{cell.get('cell_type', '?')}] =====")
        print(text.rstrip())
        for out in cell.get("outputs", []):
            kind = out.get("output_type")
            if kind == "stream":
                raw = "".join(out.get("text", []))
                name = out.get("name", "stdout")
                signals += [(ci, m) for m in signal_lines(raw)]
                body = trim_stderr(raw) if name == "stderr" else trim_stream(raw)
                if body:
                    print(f"--- {name} ---")
                    print(body[:max_chars])
            elif kind in ("execute_result", "display_data"):
                data = out.get("data", {})
                plain = "".join(data.get("text/plain", []))
                if "image/png" in data or plain.startswith("<Figure"):
                    print("--- [figure] ---")
                elif plain.startswith("Parameters(["):
                    print("--- [lmfit Parameters repr] ---")
                else:
                    print(f"--- {kind} ---")
                    print(plain[:max_chars])
            elif kind == "error":
                errors.append((ci, f"{out.get('ename')}: {out.get('evalue')}"))
                print(f"--- ERROR {out.get('ename')}: {out.get('evalue')} ---")
    print_warning_footer(signals, errors, cells)


#
@lru_cache(maxsize=1)
def all_example_dirs() -> list[Path]:
    """Every non-gitignored `NN_*` example dir (drops the `_fits` siblings).

    Cached: the peer-consistency scan asks once per notebook, and the listing
    shells out to `git check-ignore`. Callers must not mutate it.
    """

    dirs = [d for d in sorted(EXAMPLES_ROOT.glob("[0-9]*")) if d.is_dir()]
    return not_gitignored(dirs)


#
def report_example(ex: Path) -> tuple[int, int]:
    """Print the mechanical report for one example; return (fails, warns)."""

    nb = ex / "example.ipynb"
    print(f"# Mechanical checks: {ex}")
    rows: list[tuple[str, str, str]] = []
    if nb.is_file() and not valid_json(nb):
        rows.append(
            (
                "1  Notebook schema",
                "FAIL",
                "notebook JSON does not parse (conflict markers or a truncated save?)",
            )
        )
    elif nb.is_file():
        rows.append(("1  Notebook schema", *check_notebook_schema(nb)))
        rows.append(("9  Stripped outputs", *check_stripped(nb)))
        rows.append(("5  Roadmap/TOC", *check_roadmap_toc(nb)))
        rows.append(("7  Relative links", *check_relative_links(nb, ex)))
        rows.append(("10 Heading style", *check_heading_style(nb)))
        rows.append(("10 Prose voice", *scan_prose_voice(nb)))
        rows.append(("6  Imports", *scan_imports(nb)))
        rows.append(("12 § cross-refs", *scan_section_refs(nb)))
        rows.append(("12 Near-dup names", *scan_near_duplicate_identifiers(nb)))
        rows.append(("12 Prose-only names", *scan_prose_only_names(nb)))
        rows.append(("14 Repeated phrases", *scan_repeated_phrases(nb)))
        rows.append(("14 Repeated calls", *scan_repeated_calls(nb)))
        rows.append(("12 Behaviour claims", *scan_behavior_claims(nb)))
        rows.append(("12 Measured values", *scan_measured_values(nb)))
        rows.append(("14 Near-verbatim", *scan_near_verbatim(nb)))
        rows.append(("10 Long sentences", *scan_long_sentences(nb)))
        rows.append(("10 Code width", *scan_code_width(nb)))
        rows.append(("10 Peer consistency", *scan_peer_consistency(nb, ex)))
        rows.append(("15 Private access", *scan_private_access(nb)))
    else:
        rows.append(("9  Stripped outputs", "FAIL", "no example.ipynb"))
    rows.append(("3  Required files", *check_required_files(ex)))
    rows.append(("2  Committed truth", *check_truth(ex)))
    rows.append(("4  Removed keys", *check_removed_config_keys(ex)))
    rows.append(("4  Artifacts", *check_artifacts(ex)))
    rows.append(("8  YAML comments", *scan_yaml_comments(ex)))
    for name, status, detail in rows:
        print(f"{status:4} | {name:22} | {detail}")
    fails = sum(1 for _, s, _ in rows if s == "FAIL")
    warns = sum(1 for _, s, _ in rows if s == "WARN")
    infos = sum(1 for _, s, _ in rows if s == "INFO")
    print(
        f"# {fails} FAIL, {warns} WARN, {infos} INFO "
        "(INFO = a fact for the agent to resolve by reading, not a defect; "
        "judgment criteria 1,6,7,8,11,13 and the prose/message parts of "
        "5,10,12,14,15 need the read — 12 needs the executed copy, see --dump)"
    )
    return fails, warns


#
def main() -> None:
    """Report on one example (argv[1]), every example (no argument), or --dump."""

    usage = (
        "usage: check_example_mechanics.py [example dir or NN_ prefix]\n"
        "       check_example_mechanics.py --dump <executed.ipynb>"
    )
    args = sys.argv[1:]
    if args[:1] == ["--dump"]:
        if len(args) != 2 or not Path(args[1]).is_file():
            sys.exit(usage)
        dump_executed(Path(args[1]))
    elif not args:
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
    elif len(args) == 1:
        report_example(resolve_example(args[0]))
    else:
        sys.exit(usage)


if __name__ == "__main__":
    main()
