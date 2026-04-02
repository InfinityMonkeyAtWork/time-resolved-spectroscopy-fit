"""Check public function signature against NumPy-style docstring Parameters section."""

import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src" / "trspecfit"

TARGET_FILES = [
    SRC / "trspecfit.py",
    SRC / "mcp.py",
    SRC / "fitlib.py",
    SRC / "simulator.py",
    *sorted(SRC.glob("functions/*.py")),
]

SKIP_PARAMS = {"self", "cls"}


#
def parse_docstring_params(docstring: str) -> list[str]:
    """Extract parameter names from a NumPy-style Parameters section."""

    lines = docstring.split("\n")
    in_params = False
    # Detect base indentation of the Parameters section
    base_indent = 0
    params = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (
            stripped == "Parameters"
            and i + 1 < len(lines)
            and re.match(r"^\s*-{3,}\s*$", lines[i + 1])
        ):
            in_params = True
            base_indent = len(line) - len(line.lstrip())
            continue
        if in_params and re.match(r"^\s*-{3,}\s*$", stripped):
            continue
        if in_params:
            # Next section header: a non-empty line at base indent followed by dashes
            if (
                i + 1 < len(lines)
                and re.match(r"^\s*-{3,}\s*$", lines[i + 1].strip())
                and stripped
            ):
                break
            # Parameter line: valid Python identifier (letter or _) at base indent,
            # followed by " : ". Allows leading ** for **kwargs.
            indent = len(line) - len(line.lstrip()) if line.strip() else -1
            if indent == base_indent:
                m = re.match(r"^\s*(\*{0,2}[a-zA-Z_]\w*)\s*:", line)
                if m:
                    raw = m.group(1)
                    if raw.startswith("**"):
                        continue  # skip **kwargs
                    name = raw.lstrip("*")
                    params.append(name)
    return params


#
def get_sig_params(node: ast.FunctionDef) -> list[str]:
    """Extract parameter names from an AST function node, skipping special ones."""

    params = []
    for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
        name = arg.arg
        if name not in SKIP_PARAMS:
            params.append(name)
    return params


#
def collect_functions(tree: ast.Module) -> list[tuple[str, ast.FunctionDef]]:
    """Collect public functions and methods from an AST module."""

    nodes: list[tuple[str, ast.FunctionDef]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                nodes.append((node.name, node))
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not item.name.startswith("_"):
                        nodes.append((f"{node.name}.{item.name}", item))
    return nodes


#
def check_file(filepath: Path) -> list[str]:
    """Check all public functions/methods in a file for docstring mismatches."""

    source = filepath.read_text()
    tree = ast.parse(source, filename=str(filepath))
    issues = []
    rel = filepath.relative_to(REPO)

    for qual_name, func_node in collect_functions(tree):
        docstring = ast.get_docstring(func_node)
        if not docstring:
            continue
        sig_params = get_sig_params(func_node)
        doc_params = parse_docstring_params(docstring)
        if not doc_params:
            continue  # no Parameters section — skip silently

        sig_set = set(sig_params)
        doc_set = set(doc_params)
        missing = sorted(sig_set - doc_set)
        extra = sorted(doc_set - sig_set)
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"missing from docstring: {missing}")
            if extra:
                parts.append(f"extra in docstring: {extra}")
            issues.append(
                f"{rel}:{func_node.lineno} — {qual_name} — {' / '.join(parts)}"
            )
    return issues


#
def main() -> None:
    all_issues: list[str] = []
    for path in TARGET_FILES:
        if path.exists():
            all_issues.extend(check_file(path))

    for issue in all_issues:
        print(issue)

    print(f"\n{len(all_issues)} mismatch(es) found.")
    sys.exit(1 if all_issues else 0)


if __name__ == "__main__":
    main()
