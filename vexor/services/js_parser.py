"""Tree-sitter based parser for JavaScript and TypeScript files."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .content_extract_service import (
    CodeChunk,
    DOC_COMMENT_MAX_CHARS,
    DOC_COMMENT_MAX_LINES,
    FULL_CHAR_LIMIT,
)

if TYPE_CHECKING:
    from tree_sitter import Node

# Supported JS/TS extensions
JS_EXTENSIONS = frozenset({".js", ".jsx", ".mjs", ".cjs"})
TS_EXTENSIONS = frozenset({".ts", ".tsx", ".mts", ".cts"})
JSTS_EXTENSIONS = JS_EXTENSIONS | TS_EXTENSIONS


def _get_parser_for_suffix(suffix: str):
    """Return a configured tree-sitter parser for the given file suffix."""
    try:
        from tree_sitter import Language, Parser
        import tree_sitter_javascript as ts_js
        import tree_sitter_typescript as ts_ts
    except ImportError:
        return None

    suffix_lower = suffix.lower()
    if suffix_lower in JS_EXTENSIONS:
        lang = Language(ts_js.language())
    elif suffix_lower in {".tsx"}:
        lang = Language(ts_ts.language_tsx())
    elif suffix_lower in TS_EXTENSIONS:
        lang = Language(ts_ts.language_typescript())
    else:
        return None

    parser = Parser(lang)
    return parser


def _read_source(path: Path, char_limit: int) -> bytes | None:
    """Read file content as bytes for tree-sitter."""
    try:
        content = path.read_bytes()
        if len(content) > char_limit:
            content = content[:char_limit]
        return content
    except (OSError, IOError):
        return None


def _node_text(node: "Node", source: bytes) -> str:
    """Extract text content from a tree-sitter node."""
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _get_function_name(node: "Node", source: bytes) -> str | None:
    """Extract function name from a function declaration node."""
    for child in node.children:
        if child.type == "identifier":
            return _node_text(child, source)
    return None


def _get_class_name(node: "Node", source: bytes) -> str | None:
    """Extract class name from a class declaration node."""
    for child in node.children:
        if child.type == "type_identifier" or child.type == "identifier":
            return _node_text(child, source)
    return None


def _get_method_name(node: "Node", source: bytes) -> str | None:
    """Extract method name from a method definition node."""
    for child in node.children:
        if child.type == "property_identifier":
            return _node_text(child, source)
    return None


def _get_variable_declarator_name(node: "Node", source: bytes) -> str | None:
    """Extract variable name from a variable declarator with arrow function."""
    for child in node.children:
        if child.type == "identifier":
            return _node_text(child, source)
    return None


def _is_arrow_function_variable(node: "Node") -> bool:
    """Check if a variable_declarator contains an arrow function."""
    for child in node.children:
        if child.type == "arrow_function":
            return True
    return False


def _get_first_line(text: str) -> str:
    """Get the first non-empty line of text."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return text[:80] if text else ""


def _trim_doc_comment(lines: list[str]) -> str | None:
    if not lines:
        return None
    trimmed = lines
    if len(trimmed) > DOC_COMMENT_MAX_LINES:
        trimmed = trimmed[:DOC_COMMENT_MAX_LINES]
    text = "\n".join(line.rstrip("\n") for line in trimmed).strip()
    if not text:
        return None
    if len(text) > DOC_COMMENT_MAX_CHARS:
        text = text[:DOC_COMMENT_MAX_CHARS].rstrip()
    return text or None


def _collect_line_comment_block(lines: list[str], start_line: int) -> tuple[int, str] | None:
    idx = start_line - 2
    if idx < 0:
        return None
    if not lines[idx].strip():
        return None
    while idx >= 0 and lines[idx].strip().startswith("//"):
        idx -= 1
    start_idx = idx + 1
    if start_idx >= start_line - 1:
        return None
    comment_text = _trim_doc_comment(lines[start_idx:start_line - 1])
    if not comment_text:
        return None
    return start_idx + 1, comment_text


def _collect_block_comment(lines: list[str], start_line: int) -> tuple[int, str] | None:
    idx = start_line - 2
    if idx < 0:
        return None
    line = lines[idx].strip()
    if not line or "*/" not in line:
        return None
    start_idx = idx
    while start_idx >= 0:
        if "/*" in lines[start_idx]:
            break
        start_idx -= 1
    if start_idx < 0:
        return None
    if not lines[start_idx].lstrip().startswith("/*"):
        return None
    comment_text = _trim_doc_comment(lines[start_idx:start_line - 1])
    if not comment_text:
        return None
    return start_idx + 1, comment_text


def _extract_doc_comment(lines: list[str], start_line: int) -> tuple[int, str] | None:
    if start_line <= 1:
        return None
    line_comment = _collect_line_comment_block(lines, start_line)
    if line_comment is not None:
        return line_comment
    return _collect_block_comment(lines, start_line)


def _collect_method_names(class_body: "Node", source: bytes) -> list[str]:
    """Collect all method names from a class body."""
    method_names = []
    for child in class_body.children:
        if child.type == "method_definition":
            name = _get_method_name(child, source)
            if name:
                method_names.append(name)
    return method_names


def extract_js_chunks(
    path: Path,
    *,
    char_limit: int = FULL_CHAR_LIMIT,
) -> list[CodeChunk]:
    """Extract AST-aware code chunks from JavaScript/TypeScript files.

    Identifies:
    - Function declarations (function foo() {})
    - Arrow functions assigned to variables (const foo = () => {})
    - Class declarations (class Foo {})
    - Methods within classes
    - Module-level code (imports, exports, other statements)
    """
    suffix = path.suffix.lower()
    if suffix not in JSTS_EXTENSIONS:
        return []

    parser = _get_parser_for_suffix(suffix)
    if parser is None:
        return []

    source = _read_source(path, char_limit)
    if not source:
        return []

    try:
        tree = parser.parse(source)
    except Exception:
        return []

    root = tree.root_node
    if root is None:
        return []

    source_str = source.decode("utf-8", errors="replace")
    lines = source_str.splitlines(keepends=True)
    max_line = len(lines)

    def to_line_number(byte_offset: int) -> int:
        """Convert byte offset to 1-based line number."""
        text_before = source[:byte_offset].decode("utf-8", errors="replace")
        return text_before.count("\n") + 1

    def slice_lines(start: int, end: int) -> str:
        """Extract text for line range (1-based, inclusive)."""
        if not max_line or start < 1:
            return ""
        start = max(1, min(start, max_line))
        end = max(start, min(end, max_line))
        return "".join(lines[start - 1:end]).strip()

    chunks: list[CodeChunk] = []
    symbols: list[tuple[int, int, str, str, str, str]] = []  # (start, end, kind, name, display, text)

    def _with_doc_comment(start_line: int, raw_text: str) -> tuple[int, str, str | None]:
        doc_comment = _extract_doc_comment(lines, start_line)
        if not doc_comment:
            return start_line, raw_text, None
        comment_start, comment_text = doc_comment
        combined = f"{comment_text}\n{raw_text}"
        return comment_start, combined, comment_text

    def process_node(node: "Node", class_name: str | None = None) -> None:
        """Process a single AST node and extract symbols."""
        node_type = node.type
        start_line = to_line_number(node.start_byte)
        end_line = to_line_number(node.end_byte)
        raw_text = _node_text(node, source)

        # Function declaration
        if node_type == "function_declaration":
            name = _get_function_name(node, source) or "anonymous"
            display = _get_first_line(raw_text)
            comment_start, text, _ = _with_doc_comment(start_line, raw_text)
            symbols.append((comment_start, end_line, "function", name, display, text))
            return

        # Arrow function in variable declaration (lexical_declaration or variable_declaration)
        if node_type in ("lexical_declaration", "variable_declaration"):
            for child in node.children:
                if child.type == "variable_declarator" and _is_arrow_function_variable(child):
                    name = _get_variable_declarator_name(child, source) or "anonymous"
                    display = _get_first_line(raw_text)
                    comment_start, text, _ = _with_doc_comment(start_line, raw_text)
                    symbols.append((comment_start, end_line, "function", name, display, text))
                    return
            return

        # Class declaration
        if node_type == "class_declaration":
            name = _get_class_name(node, source) or "AnonymousClass"
            display = f"class {name}"

            # Find class body and methods
            class_body = None
            for child in node.children:
                if child.type == "class_body":
                    class_body = child
                    break

            method_names = []
            if class_body:
                method_names = _collect_method_names(class_body, source)

            # Build class chunk text (class header + methods list)
            comment_start, _, doc_text = _with_doc_comment(start_line, raw_text)
            class_text_parts = []
            if doc_text:
                class_text_parts.append(doc_text)
            class_text_parts.append(_get_first_line(raw_text))
            if method_names:
                class_text_parts.append("Methods: " + ", ".join(method_names))
            class_text = "\n".join(class_text_parts)
            symbols.append((comment_start, end_line, "class", name, display, class_text))

            # Process methods
            if class_body:
                for child in class_body.children:
                    if child.type == "method_definition":
                        method_name = _get_method_name(child, source)
                        if method_name:
                            method_start = to_line_number(child.start_byte)
                            method_end = to_line_number(child.end_byte)
                            method_text = _node_text(child, source)
                            method_display = f"{name}.{method_name}"
                            comment_start, method_text, _ = _with_doc_comment(method_start, method_text)
                            symbols.append((
                                comment_start,
                                method_end,
                                "method",
                                f"{name}.{method_name}",
                                method_display,
                                method_text,
                            ))
            return

        # Export statement - unwrap and process inner declaration
        if node_type == "export_statement":
            for child in node.children:
                if child.type in (
                    "function_declaration",
                    "class_declaration",
                    "lexical_declaration",
                    "variable_declaration",
                ):
                    # Use parent's line range for exported symbols
                    inner_raw = _node_text(node, source)
                    inner_start = start_line
                    inner_end = end_line

                    if child.type == "function_declaration":
                        fname = _get_function_name(child, source) or "anonymous"
                        display = _get_first_line(inner_raw)
                        comment_start, inner_text, _ = _with_doc_comment(inner_start, inner_raw)
                        symbols.append((comment_start, inner_end, "function", fname, display, inner_text))
                    elif child.type == "class_declaration":
                        cname = _get_class_name(child, source) or "AnonymousClass"
                        display = f"export class {cname}"

                        class_body = None
                        for cc in child.children:
                            if cc.type == "class_body":
                                class_body = cc
                                break

                        method_names = _collect_method_names(class_body, source) if class_body else []
                        comment_start, _, doc_text = _with_doc_comment(inner_start, inner_raw)
                        class_text_parts = []
                        if doc_text:
                            class_text_parts.append(doc_text)
                        class_text_parts.append(_get_first_line(inner_raw))
                        if method_names:
                            class_text_parts.append("Methods: " + ", ".join(method_names))

                        symbols.append((comment_start, inner_end, "class", cname, display, "\n".join(class_text_parts)))

                        if class_body:
                            for mc in class_body.children:
                                if mc.type == "method_definition":
                                    mname = _get_method_name(mc, source)
                                    if mname:
                                        mstart = to_line_number(mc.start_byte)
                                        mend = to_line_number(mc.end_byte)
                                        mtext = _node_text(mc, source)
                                        comment_start, mtext, _ = _with_doc_comment(mstart, mtext)
                                        symbols.append((comment_start, mend, "method", f"{cname}.{mname}", f"{cname}.{mname}", mtext))
                    elif child.type in ("lexical_declaration", "variable_declaration"):
                        for vc in child.children:
                            if vc.type == "variable_declarator" and _is_arrow_function_variable(vc):
                                vname = _get_variable_declarator_name(vc, source) or "anonymous"
                                display = _get_first_line(inner_raw)
                                comment_start, inner_text, _ = _with_doc_comment(inner_start, inner_raw)
                                symbols.append((comment_start, inner_end, "function", vname, display, inner_text))
                    return
            return

    # Process top-level nodes
    for child in root.children:
        process_node(child)

    # Sort symbols by start line
    symbols.sort(key=lambda s: s[0])

    # Build module globals chunks for gaps between symbols
    def add_module_chunk(start: int, end: int, *, prelude: bool) -> None:
        text = slice_lines(start, end)
        if not text.strip():
            return
        chunks.append(
            CodeChunk(
                kind="module",
                name="module" if prelude else "module_globals",
                display="module" if prelude else "module globals",
                text=text,
                start_line=start,
                end_line=end,
            )
        )

    if not symbols:
        # No symbols found, treat entire file as module
        add_module_chunk(1, max_line, prelude=True)
        return chunks

    cursor = 1
    seen_symbol = False

    for start, end, kind, name, display, text in symbols:
        # Add module chunk for gap before this symbol
        if cursor < start:
            add_module_chunk(cursor, start - 1, prelude=not seen_symbol)

        chunks.append(
            CodeChunk(
                kind=kind,
                name=name,
                display=display,
                text=text,
                start_line=start,
                end_line=end,
            )
        )
        cursor = end + 1
        seen_symbol = True

    # Add trailing module chunk
    if cursor <= max_line:
        add_module_chunk(cursor, max_line, prelude=False)

    return chunks
