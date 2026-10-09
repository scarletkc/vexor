"""Guard authored runtime copy while leaving protocol and source data in place."""

import ast
import re
from pathlib import Path

import pytest

RUNTIME_ROOT = Path(__file__).resolve().parents[2] / "vexor"
PROSE = re.compile(r"[A-Za-z]{2,}\s+[A-Za-z]{2,}")
SQL = re.compile(
    r"\b(?:SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|PRAGMA|BEGIN|"
    r"WHERE|JOIN|ORDER BY|NOT IN|IS NULL|IS NOT NULL)\b|\bIN \("
)
# These are inputs recognized in external errors or snippets of indexed source, not authored copy.
DATA_LITERALS = {
    # Exact SQLite errors recognized for Python 3.10, which lacks sqlite_errorcode.
    ("cache.py", "database is locked"),
    ("cache.py", "database table is locked"),
    ("cache.py", "attempt to write a readonly database"),
    ("providers/gemini.py", "API key"),
    ("providers/local.py", "not supported in TextEmbedding"),
    ("providers/local.py", "already registered"),
    ("providers/retry.py", "rate limit"),
    ("providers/retry.py", "try again"),
    ("providers/retry.py", "too many requests"),
    ("providers/retry.py", "service unavailable"),
    ("services/content_extract_service.py", "# -*- coding"),
    ("services/content_extract_service.py", "# coding"),
    ("services/content_extract_service.py", "module globals"),
    ("services/content_extract_service.py", "async def "),
    ("services/js_parser.py", "module globals"),
    ("services/js_parser.py", "export class "),
    ("services/content_extract_service.py", "preamble"),
}
PRESENTATION_CALLS = {
    "print", "echo", "secho", "prompt", "confirm", "add_column", "_styled",
    "_print_step_header", "_print_option", "_prompt_choice", "_prompt_required",
    "_prompt_required_secret", "_prompt_api_key", "_note_dry_run",
}
COPY_KEYWORDS = {"help", "description", "message", "reason", "title"}


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _literal_parts(node: ast.AST) -> list[ast.Constant]:
    """Read composed text without mistaking format values or lookup keys for copy."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node]
    if isinstance(node, ast.JoinedStr):
        return [part for part in node.values if isinstance(part, ast.Constant)]
    if isinstance(node, ast.BinOp):
        right = _literal_parts(node.right) if isinstance(node.op, ast.Add) else []
        return _literal_parts(node.left) + right
    if isinstance(node, ast.IfExp):
        return _literal_parts(node.body) + _literal_parts(node.orelse)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr == "format":
            return _literal_parts(node.func.value)
    return []


def _plain_text(value: str) -> str:
    # Rich markup, punctuation, and whitespace are presentation rather than authored wording.
    return re.sub(r"\[/?[a-z][a-z ]*\]", "", value)


def _copy_violations(source: str, relative_path: str) -> list[str]:
    tree = ast.parse(source)
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    presentation_data = {
        id(keyword.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg in {"style", "header_style", "border_style", "prog"}
    }
    violations: dict[int, ast.Constant] = {}

    def inspect(expression: ast.AST) -> None:
        for part in _literal_parts(expression):
            if any(character.isalpha() for character in _plain_text(part.value)) and (
                relative_path, part.value
            ) not in DATA_LITERALS:
                violations[id(part)] = part

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Catch copy assigned to a variable before it reaches a presentation call.
            if (
                id(node) not in docstrings
                and id(node) not in presentation_data
                and PROSE.search(_plain_text(node.value))
                and not SQL.search(node.value)
                and (relative_path, node.value) not in DATA_LITERALS
            ):
                violations[id(node)] = node
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call) and node.exc.args:
            inspect(node.exc.args[0])
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if node.args and (
                name.endswith("Error")
                or name in {"Exception", "BadParameter", "InvalidToolArguments"}
            ):
                inspect(node.args[0])
            if name in PRESENTATION_CALLS:
                # Option keys and step numbers are data; their names/descriptions are copy.
                args = (
                    node.args[1:] if name in {"_print_option", "_print_step_header"} else node.args
                )
                for arg in args:
                    inspect(arg)
            for keyword in node.keywords:
                if keyword.arg in COPY_KEYWORDS or (
                    name == "DoctorCheckResult" and keyword.arg in {"name", "detail"}
                ):
                    inspect(keyword.value)
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value in {"description", "message"}:
                    inspect(value)
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if node.value is not None and any(
                isinstance(target, ast.Name) and target.id in COPY_KEYWORDS
                for target in targets
            ):
                inspect(node.value)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                if (
                    isinstance(decorator, ast.Call)
                    and _call_name(decorator.func) == "command"
                    and not any(keyword.arg == "help" for keyword in decorator.keywords)
                    and ast.get_docstring(node)
                ):
                    violations[id(node.body[0].value)] = node.body[0].value
    return [
        f"{relative_path}:{part.lineno}: {part.value!r}"
        for part in sorted(violations.values(), key=lambda part: part.lineno)
    ]


def test_runtime_copy_is_centralized() -> None:
    violations = []
    for path in sorted(RUNTIME_ROOT.rglob("*.py")):
        if path == RUNTIME_ROOT / "text.py":
            continue
        violations.extend(_copy_violations(
            path.read_text(encoding="utf-8"), path.relative_to(RUNTIME_ROOT).as_posix()
        ))
    assert not violations, "Move authored copy to vexor/text.py:\n" + "\n".join(violations)


@pytest.mark.parametrize("source", [
    'raise ValueError("Invalid")',
    'error = ValueError("Invalid")\nraise error',
    'raise ValueError("不支持该维度")',
    'raise ValueError(f"Dimension {value} is not supported")',
    'raise ValueError("Bad value: {}".format(value))',
    'message = "A newly hard-coded error"\nraise ValueError(message)',
    'message = "Oops"\nconsole.print(message)',
    'console.print(_styled("Failed", Styles.ERROR))',
    'typer.Option(None, help="Choose a model")',
    'typer.prompt("Name")',
    'DoctorCheckResult(name="Config", passed=True, message=message)',
    'schema = {"description": "Directory to search"}',
    'Messages.MCP_INVALID_ARGUMENTS.format(reason="Invalid arguments")',
    '@app.command()\ndef command():\n    """Run the command."""',
])
def test_copy_guard_rejects_inline_and_indirect_copy(source: str) -> None:
    assert _copy_violations(source, "new_module.py")


@pytest.mark.parametrize("source", [
    'raise ValueError(Messages.ERROR_EMPTY_QUERY)',
    'raise ValueError(Messages.ERROR_MODE_INVALID.format(value=mode, allowed=", ".join(modes)))',
    'raise ValueError(str(exc))',
    'console.print(f"[bold]{Messages.APP_HELP}[/bold]")',
    'console.print(content, markup=False)',
    'query = "SELECT name FROM sqlite_master WHERE type = ?"',
    'schema = {"type": "string", "required": ["query"]}',
    'def helper():\n    """Explain the internal helper."""',
    '@app.command(help=Messages.HELP_SEARCH)\ndef search():\n    """Run the semantic search."""',
])
def test_copy_guard_allows_centralized_copy_and_data(source: str) -> None:
    assert not _copy_violations(source, "new_module.py")
