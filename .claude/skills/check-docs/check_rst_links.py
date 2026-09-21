"""Check that relative link targets in RST pages under docs/ exist.

Sphinx does not resolve external-style RST links (`text <path>`_), so a dead
relative target builds without a warning.
"""

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DOCS = REPO / "docs"
LINK = re.compile(r"`[^`<>]*<([^<>]+)>`_")
EXTERNAL = re.compile(r"^(https?:|mailto:|#)")


#
def check_file(path: Path) -> list[str]:
    """Return one message per relative link in ``path`` whose target is missing."""

    issues: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for target in LINK.findall(line):
            if EXTERNAL.match(target):
                continue
            if not (path.parent / target.split("#")[0]).exists():
                rel = path.relative_to(REPO) if path.is_relative_to(REPO) else path
                issues.append(f"{rel}:{lineno} — dead link target {target!r}")
    return issues


#
def main(argv: list[str]) -> int:
    """Scan the given RST files (default: every ``docs/**/*.rst``)."""

    files = [Path(a) for a in argv] or sorted(DOCS.rglob("*.rst"))
    issues: list[str] = []
    for path in files:
        issues.extend(check_file(path))
    for issue in issues:
        print(issue)
    print(f"\n{len(issues)} dead link target(s) found.")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
