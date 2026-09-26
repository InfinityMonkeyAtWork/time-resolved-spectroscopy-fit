"""Check public function signature against NumPy-style docstring Parameters section."""

import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src" / "trspecfit"

TARGET_FILES = sorted(SRC.rglob("*.py"))

SKIP_PARAMS = {"self", "cls"}

FuncNode = ast.FunctionDef | ast.AsyncFunctionDef
DocNode = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


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
            # Parameter line: one or more comma-separated Python identifiers at
            # base indent, then " : type" or nothing (NumPy style allows
            # "x1, x2 : type" and omits the colon when there is no type).
            # Starred entries (*args, **kwargs) are skipped.
            indent = len(line) - len(line.lstrip()) if line.strip() else -1
            if indent == base_indent:
                name = r"\*{0,2}[a-zA-Z_]\w*"
                m = re.match(rf"^\s*({name}(?:\s*,\s*{name})*)\s*(?::|$)", line)
                if m:
                    for raw in (n.strip() for n in m.group(1).split(",")):
                        if raw.startswith("*"):
                            continue  # *args / **kwargs are not in the signature list
                        params.append(raw)
    return params


#
def get_sig_params(node: FuncNode) -> list[str]:
    """Extract parameter names from an AST function node, skipping special ones."""

    params = []
    for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
        name = arg.arg
        if name not in SKIP_PARAMS:
            params.append(name)
    return params


#
def collect_functions(tree: ast.Module) -> list[tuple[str, FuncNode, DocNode]]:
    """Collect (name, signature node, docstring node) for public API.

    Public functions and methods document themselves. A public class's
    ``__init__`` signature is checked against both the class docstring and
    ``__init__``'s own docstring, whichever carry a Parameters section.
    """

    nodes: list[tuple[str, FuncNode, DocNode]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                nodes.append((node.name, node, node))
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            for item in node.body:
                if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if item.name == "__init__":
                    nodes.append((node.name, item, node))
                    nodes.append((f"{node.name}.__init__", item, item))
                elif not item.name.startswith("_"):
                    nodes.append((f"{node.name}.{item.name}", item, item))
    return nodes


#
def check_file(filepath: Path) -> list[str]:
    """Check all public functions/methods in a file for docstring mismatches."""

    source = filepath.read_text()
    tree = ast.parse(source, filename=str(filepath))
    issues = []
    rel = filepath.relative_to(REPO)

    for qual_name, func_node, doc_node in collect_functions(tree):
        docstring = ast.get_docstring(doc_node)
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
                f"{rel}:{doc_node.lineno} — {qual_name} — {' / '.join(parts)}"
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
