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
def is_property_accessor(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True for ``@<name>.setter`` / ``@<name>.deleter`` definitions.

    NumPy convention documents a property on its getter only.
    """

    return any(
        isinstance(d, ast.Attribute) and d.attr in ("setter", "deleter")
        for d in node.decorator_list
    )


#
def public_definitions(body: list[ast.stmt]):
    """Yield the function and class definitions at module or class level.

    Descends into class bodies (methods, nested classes) but not into
    function bodies, so local closures never count as public API.
    """

    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node
            if isinstance(node, ast.ClassDef):
                yield from public_definitions(node.body)


#
def check_file(path: Path) -> list[tuple[Path, int, str, str]]:
    """Return list of (path, lineno, kind, name) for missing docstrings."""

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    missing: list[tuple[Path, int, str, str]] = []

    for node in public_definitions(tree.body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue
            if has_overload(node) or is_property_accessor(node):
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
