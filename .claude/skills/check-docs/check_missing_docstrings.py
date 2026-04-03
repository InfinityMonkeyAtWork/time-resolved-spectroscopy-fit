"""Check public functions, methods, and classes for missing docstrings."""

import ast
import sys
from pathlib import Path

SRC_ROOT = Path("src/trspecfit")


#
def has_overload(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True if the node is decorated with @overload."""

    return any(
        (isinstance(d, ast.Name) and d.id == "overload")
        or (isinstance(d, ast.Attribute) and d.attr == "overload")
        for d in node.decorator_list
    )


#
def check_file(path: Path) -> list[tuple[Path, int, str, str]]:
    """Return list of (path, lineno, kind, name) for missing docstrings."""

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    missing: list[tuple[Path, int, str, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue
            if has_overload(node):
                continue
            if not ast.get_docstring(node):
                missing.append((path, node.lineno, "def", node.name))
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("_"):
                continue
            if not ast.get_docstring(node):
                missing.append((path, node.lineno, "class", node.name))

    return missing


#
def main() -> int:
    """Scan src/trspecfit/ for public definitions missing docstrings."""

    all_missing: list[tuple[Path, int, str, str]] = []

    for py_file in sorted(SRC_ROOT.rglob("*.py")):
        all_missing.extend(check_file(py_file))

    for path, lineno, kind, name in all_missing:
        print(f"{path}:{lineno} — {kind} {name}")

    print(f"\n{len(all_missing)} missing docstring(s) found.")
    return 1 if all_missing else 0


if __name__ == "__main__":
    sys.exit(main())
