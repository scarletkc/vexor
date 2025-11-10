from pathlib import Path
import pytest

from vexor.utils import collect_files, format_path, resolve_directory


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
