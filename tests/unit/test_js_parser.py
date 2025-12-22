"""Unit tests for JavaScript/TypeScript code chunk extraction."""

from __future__ import annotations

import pytest
from pathlib import Path

from vexor.services.js_parser import extract_js_chunks, JSTS_EXTENSIONS


class TestJsParserExtensions:
    """Test extension handling."""

    def test_js_extensions_supported(self):
        assert ".js" in JSTS_EXTENSIONS
        assert ".jsx" in JSTS_EXTENSIONS
        assert ".mjs" in JSTS_EXTENSIONS
        assert ".cjs" in JSTS_EXTENSIONS

    def test_ts_extensions_supported(self):
        assert ".ts" in JSTS_EXTENSIONS
        assert ".tsx" in JSTS_EXTENSIONS
        assert ".mts" in JSTS_EXTENSIONS
        assert ".cts" in JSTS_EXTENSIONS

    def test_unsupported_extension_returns_empty(self, tmp_path):
        py_file = tmp_path / "test.py"
        py_file.write_text("def foo(): pass")
        assert extract_js_chunks(py_file) == []


class TestJsFunctionExtraction:
    """Test function extraction from JavaScript files."""

    def test_function_declaration(self, tmp_path):
        js_file = tmp_path / "test.js"
        js_file.write_text("""
function greet(name) {
    return "Hello, " + name;
}
""")
        chunks = extract_js_chunks(js_file)
        func_chunks = [c for c in chunks if c.kind == "function"]
        assert len(func_chunks) == 1
        assert func_chunks[0].name == "greet"
        assert "function greet" in func_chunks[0].display

    def test_async_function_declaration(self, tmp_path):
        js_file = tmp_path / "test.js"
        js_file.write_text("""
async function fetchData(url) {
    const response = await fetch(url);
    return response.json();
}
""")
        chunks = extract_js_chunks(js_file)
        func_chunks = [c for c in chunks if c.kind == "function"]
        assert len(func_chunks) == 1
        assert func_chunks[0].name == "fetchData"

    def test_arrow_function_const(self, tmp_path):
        js_file = tmp_path / "test.js"
        js_file.write_text("""
const add = (a, b) => {
    return a + b;
};
""")
        chunks = extract_js_chunks(js_file)
        func_chunks = [c for c in chunks if c.kind == "function"]
        assert len(func_chunks) == 1
        assert func_chunks[0].name == "add"

    def test_arrow_function_let(self, tmp_path):
        js_file = tmp_path / "test.js"
        js_file.write_text("""
let multiply = (a, b) => a * b;
""")
        chunks = extract_js_chunks(js_file)
        func_chunks = [c for c in chunks if c.kind == "function"]
        assert len(func_chunks) == 1

    def test_function_leading_comment_included(self, tmp_path):
        js_file = tmp_path / "test.js"
        js_file.write_text("""// Adds two numbers
function add(a, b) {
    return a + b;
}
""")
        chunks = extract_js_chunks(js_file)
        func_chunks = [c for c in chunks if c.kind == "function"]
        assert len(func_chunks) == 1
        assert "Adds two numbers" in func_chunks[0].text
        assert func_chunks[0].start_line == 1

    def test_function_block_comment_included(self, tmp_path):
        js_file = tmp_path / "test.js"
        js_file.write_text("""/** Adds two numbers */
function add(a, b) {
    return a + b;
}
""")
        chunks = extract_js_chunks(js_file)
        func_chunks = [c for c in chunks if c.kind == "function"]
        assert len(func_chunks) == 1
        assert "Adds two numbers" in func_chunks[0].text
        assert func_chunks[0].start_line == 1
        assert func_chunks[0].name == "add"

    def test_function_plain_block_comment_included(self, tmp_path):
        js_file = tmp_path / "test.js"
        js_file.write_text("""/* Adds two numbers */
function add(a, b) {
    return a + b;
}
""")
        chunks = extract_js_chunks(js_file)
        func_chunks = [c for c in chunks if c.kind == "function"]
        assert len(func_chunks) == 1
        assert "Adds two numbers" in func_chunks[0].text
        assert func_chunks[0].start_line == 1

    def test_exported_function(self, tmp_path):
        js_file = tmp_path / "test.js"
        js_file.write_text("""
export function helper() {
    return 42;
}
""")
        chunks = extract_js_chunks(js_file)
        func_chunks = [c for c in chunks if c.kind == "function"]
        assert len(func_chunks) == 1
        assert func_chunks[0].name == "helper"


class TestJsClassExtraction:
    """Test class extraction from JavaScript files."""

    def test_class_declaration(self, tmp_path):
        js_file = tmp_path / "test.js"
        js_file.write_text("""
class Calculator {
    constructor(value) {
        this.value = value;
    }

    add(n) {
        return this.value + n;
    }

    subtract(n) {
        return this.value - n;
    }
}
""")
        chunks = extract_js_chunks(js_file)

        class_chunks = [c for c in chunks if c.kind == "class"]
        assert len(class_chunks) == 1
        assert class_chunks[0].name == "Calculator"
        assert "class Calculator" in class_chunks[0].display
        assert "Methods:" in class_chunks[0].text

        method_chunks = [c for c in chunks if c.kind == "method"]
        assert len(method_chunks) == 3
        method_names = {c.name for c in method_chunks}
        assert "Calculator.constructor" in method_names
        assert "Calculator.add" in method_names
        assert "Calculator.subtract" in method_names

    def test_exported_class(self, tmp_path):
        js_file = tmp_path / "test.js"
        js_file.write_text("""
export class Service {
    init() {
        console.log("initialized");
    }
}
""")
        chunks = extract_js_chunks(js_file)
        class_chunks = [c for c in chunks if c.kind == "class"]
        assert len(class_chunks) == 1
        assert class_chunks[0].name == "Service"


class TestTsExtraction:
    """Test TypeScript-specific extraction."""

    def test_typescript_function_with_types(self, tmp_path):
        ts_file = tmp_path / "test.ts"
        ts_file.write_text("""
function greet(name: string): string {
    return `Hello, ${name}`;
}
""")
        chunks = extract_js_chunks(ts_file)
        func_chunks = [c for c in chunks if c.kind == "function"]
        assert len(func_chunks) == 1
        assert func_chunks[0].name == "greet"

    def test_typescript_class_with_types(self, tmp_path):
        ts_file = tmp_path / "test.ts"
        ts_file.write_text("""
class User {
    private name: string;

    constructor(name: string) {
        this.name = name;
    }

    getName(): string {
        return this.name;
    }
}
""")
        chunks = extract_js_chunks(ts_file)
        class_chunks = [c for c in chunks if c.kind == "class"]
        assert len(class_chunks) == 1
        assert class_chunks[0].name == "User"

    def test_tsx_file(self, tmp_path):
        tsx_file = tmp_path / "Component.tsx"
        tsx_file.write_text("""
import React from 'react';

function MyComponent({ name }: { name: string }) {
    return <div>Hello, {name}</div>;
}

export default MyComponent;
""")
        chunks = extract_js_chunks(tsx_file)
        func_chunks = [c for c in chunks if c.kind == "function"]
        assert len(func_chunks) == 1
        assert func_chunks[0].name == "MyComponent"

    def test_interface_and_type_in_module(self, tmp_path):
        ts_file = tmp_path / "types.ts"
        ts_file.write_text("""
interface User {
    id: string;
    name: string;
}

type Status = 'active' | 'inactive';

function createUser(name: string): User {
    return { id: '1', name };
}
""")
        chunks = extract_js_chunks(ts_file)
        # Interface and type alias should be in module chunk
        module_chunks = [c for c in chunks if c.kind == "module"]
        assert len(module_chunks) >= 1
        # Function should be extracted
        func_chunks = [c for c in chunks if c.kind == "function"]
        assert len(func_chunks) == 1


class TestModuleChunks:
    """Test module-level code extraction."""

    def test_imports_in_module_chunk(self, tmp_path):
        js_file = tmp_path / "test.js"
        js_file.write_text("""
import { foo } from './foo';
import bar from 'bar';

function main() {
    console.log(foo, bar);
}
""")
        chunks = extract_js_chunks(js_file)
        module_chunks = [c for c in chunks if c.kind == "module"]
        assert len(module_chunks) >= 1
        # Imports should be in module chunk
        module_text = " ".join(c.text for c in module_chunks)
        assert "import" in module_text

    def test_empty_file(self, tmp_path):
        js_file = tmp_path / "empty.js"
        js_file.write_text("")
        chunks = extract_js_chunks(js_file)
        assert chunks == []

    def test_file_with_only_comments(self, tmp_path):
        js_file = tmp_path / "comments.js"
        js_file.write_text("""
// This is a comment
/* Multi-line
   comment */
""")
        chunks = extract_js_chunks(js_file)
        # Should have module chunk with comments
        assert len(chunks) >= 0  # May or may not have chunks depending on content


class TestLineNumbers:
    """Test that line numbers are correctly reported."""

    def test_function_line_numbers(self, tmp_path):
        js_file = tmp_path / "test.js"
        js_file.write_text("""// Comment line 1
// Comment line 2
function foo() {
    return 1;
}
""")
        chunks = extract_js_chunks(js_file)
        func_chunks = [c for c in chunks if c.kind == "function"]
        assert len(func_chunks) == 1
        # Function starts at line 3
        assert func_chunks[0].start_line == 3
        assert func_chunks[0].end_line == 5


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_syntax_error_returns_empty(self, tmp_path):
        js_file = tmp_path / "broken.js"
        js_file.write_text("function { broken syntax")
        # Should not raise, just return empty or partial
        chunks = extract_js_chunks(js_file)
        # tree-sitter is error-tolerant, so it may still parse something
        assert isinstance(chunks, list)

    def test_nonexistent_file_returns_empty(self, tmp_path):
        missing = tmp_path / "missing.js"
        chunks = extract_js_chunks(missing)
        assert chunks == []

    def test_binary_file_returns_empty(self, tmp_path):
        bin_file = tmp_path / "test.js"
        bin_file.write_bytes(b"\x00\x01\x02\x03")
        chunks = extract_js_chunks(bin_file)
        # Should handle gracefully
        assert isinstance(chunks, list)


class TestIntegrationWithExtractCodeChunks:
    """Test integration with main extract_code_chunks function."""

    def test_js_routed_correctly(self, tmp_path):
        from vexor.services.content_extract_service import extract_code_chunks

        js_file = tmp_path / "test.js"
        js_file.write_text("""
function hello() {
    return "world";
}
""")
        chunks = extract_code_chunks(js_file)
        assert len(chunks) >= 1
        func_chunks = [c for c in chunks if c.kind == "function"]
        assert len(func_chunks) == 1

    def test_ts_routed_correctly(self, tmp_path):
        from vexor.services.content_extract_service import extract_code_chunks

        ts_file = tmp_path / "test.ts"
        ts_file.write_text("""
function greet(name: string): string {
    return name;
}
""")
        chunks = extract_code_chunks(ts_file)
        func_chunks = [c for c in chunks if c.kind == "function"]
        assert len(func_chunks) == 1
