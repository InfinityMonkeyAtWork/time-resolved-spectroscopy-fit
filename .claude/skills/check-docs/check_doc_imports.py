"""Execute every ``trspecfit`` import line shown in the documentation.

Catches docs that promise a name the package no longer exports.
"""

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
IMPORT = re.compile(r"^\s*(from trspecfit\S* import .+|import trspecfit\b.*)$")


#
def default_files() -> list[Path]:
    """README, llms.txt and every Markdown / RST page under docs/."""

    files = [REPO / "README.md", REPO / "llms.txt"]
    pages = (REPO / "docs").rglob("*")
    files += sorted(p for p in pages if p.suffix in (".md", ".rst"))
    return [f for f in files if f.exists() and "_build" not in f.parts]


#
def check_file(path: Path) -> list[str]:
    """Return one message per import line in ``path`` that fails to execute."""

    issues: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        m = IMPORT.match(line)
        if not m:
            continue
        statement = m.group(1).strip()
        try:
            exec(statement, {})  # the docs' own import lines
        except Exception as exc:  # noqa: BLE001 — any failure is the finding
            rel = path.relative_to(REPO) if path.is_relative_to(REPO) else path
            issues.append(
                f"{rel}:{lineno} — {statement!r} -> {type(exc).__name__}: {exc}"
            )
    return issues


#
def main(argv: list[str]) -> int:
    """Scan the given files (default: README, llms.txt, docs/**/*.md|rst)."""

    files = [Path(a) for a in argv] or default_files()
    issues: list[str] = []
    for path in files:
        issues.extend(check_file(path))
    for issue in issues:
        print(issue)
    print(f"\n{len(issues)} failing import line(s) found.")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
