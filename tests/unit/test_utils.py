from pathlib import Path
import pytest

from vexor.utils import collect_files, format_path, resolve_directory, normalize_extensions


def test_collect_files_ignores_hidden(tmp_path):
    visible = tmp_path / "visible.txt"
    visible.write_text("data")
    hidden_file = tmp_path / ".hidden.txt"
    hidden_file.write_text("secret")
    hidden_dir = tmp_path / ".hidden"
    hidden_dir.mkdir()
    (hidden_dir / "nested.txt").write_text("nested")

    files = collect_files(tmp_path)

    assert files == [visible]


def test_collect_files_includes_hidden(tmp_path):
    visible = tmp_path / "visible.txt"
    visible.write_text("data")
    hidden_file = tmp_path / ".hidden.txt"
    hidden_file.write_text("secret")

    files = collect_files(tmp_path, include_hidden=True)

    assert set(files) == {visible, hidden_file}


def test_collect_files_non_recursive(tmp_path):
    top = tmp_path / "top.txt"
    top.write_text("data")
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    nested_file = nested_dir / "child.txt"
    nested_file.write_text("child")

    files = collect_files(tmp_path, recursive=False)

    assert files == [top]


def test_collect_files_filters_extensions(tmp_path):
    py_file = tmp_path / "demo.PY"
    py_file.write_text("py")
    md_file = tmp_path / "readme.md"
    md_file.write_text("md")
    other = tmp_path / "plain.txt"
    other.write_text("txt")

    filtered = collect_files(tmp_path, extensions=normalize_extensions([".py", ".MD"]))

    assert [path.name for path in filtered] == ["demo.PY", "readme.md"]


def test_resolve_directory_invalid(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_directory(tmp_path / "missing")


def test_format_path_relative(tmp_path):
    root = tmp_path
    file_path = tmp_path / "folder" / "file.txt"
    file_path.parent.mkdir()
    file_path.write_text("data")

    human = format_path(file_path, base=root)

    assert human.startswith(".")
    assert "file.txt" in human


def test_normalize_extensions_deduplicates_and_sorts():
    result = normalize_extensions(["py", ".MD", ".py", "", ".md"])  # mixed case + blanks

    assert result == (".md", ".py")


def test_collect_files_respects_gitignore(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / ".git" / "info").mkdir(parents=True)
    (root / ".git" / "info" / "exclude").write_text("exclude_me.txt\n", encoding="utf-8")
    (root / ".gitignore").write_text("ignored/\n*.log\n", encoding="utf-8")

    (root / "keep.txt").write_text("keep", encoding="utf-8")
    (root / "notes.log").write_text("log", encoding="utf-8")
    (root / "exclude_me.txt").write_text("exclude", encoding="utf-8")

    (root / "ignored").mkdir()
    (root / "ignored" / "ignored_only.txt").write_text("ignored", encoding="utf-8")

    (root / "sub").mkdir()
    (root / "sub" / "keep2.txt").write_text("keep2", encoding="utf-8")
    (root / "sub" / "sub_keep.txt").write_text("sub keep", encoding="utf-8")
    (root / "sub" / ".gitignore").write_text("sub_ignored.txt\n", encoding="utf-8")
    (root / "sub" / "sub_ignored.txt").write_text("sub ignored", encoding="utf-8")
    (root / "sub" / "ignored").mkdir()
    (root / "sub" / "ignored" / "ignored_nested.txt").write_text("nested ignored", encoding="utf-8")

    files = collect_files(root)

    rel_paths = [path.relative_to(root).as_posix() for path in files]
    assert rel_paths == ["keep.txt", "sub/keep2.txt", "sub/sub_keep.txt"]


def test_collect_files_can_disable_gitignore(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / ".git" / "info").mkdir(parents=True)
    (root / ".git" / "info" / "exclude").write_text("exclude_me.txt\n", encoding="utf-8")
    (root / ".gitignore").write_text("ignored/\n*.log\n", encoding="utf-8")

    paths = [
        root / "keep.txt",
        root / "notes.log",
        root / "exclude_me.txt",
        root / "ignored" / "ignored_only.txt",
        root / "sub" / "keep2.txt",
        root / "sub" / "sub_keep.txt",
        root / "sub" / "sub_ignored.txt",
        root / "sub" / "ignored" / "ignored_nested.txt",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("data", encoding="utf-8")
    (root / "sub" / ".gitignore").write_text("sub_ignored.txt\n", encoding="utf-8")

    files = collect_files(root, respect_gitignore=False)

    rel_paths = [path.relative_to(root).as_posix() for path in files]
    assert rel_paths == sorted([path.relative_to(root).as_posix() for path in paths])
