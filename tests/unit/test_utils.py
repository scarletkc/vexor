from __future__ import annotations

from pathlib import Path

import pytest

import vexor.utils as utils


def test_resolve_directory_validates(tmp_path):
    with pytest.raises(FileNotFoundError):
        utils.resolve_directory(tmp_path / "missing")

    file_path = tmp_path / "file.txt"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        utils.resolve_directory(file_path)

    assert utils.resolve_directory(tmp_path) == tmp_path.resolve()


def test_normalize_extensions():
    assert utils.normalize_extensions(None) == ()
    assert utils.normalize_extensions([]) == ()
    assert utils.normalize_extensions(["py", ".py", " .MD ", ".", "", None]) == (".md", ".py")
    assert utils.normalize_extensions([".py,.md"]) == (".md", ".py")
    assert utils.normalize_extensions([".py .md"]) == (".md", ".py")
    assert utils.normalize_extensions(["py, md", " .PY "]) == (".md", ".py")
    assert utils.normalize_extensions([".", " ", None]) == ()


def test_normalize_exclude_patterns():
    assert utils.normalize_exclude_patterns(None) == ()
    assert utils.normalize_exclude_patterns([]) == ()
    assert utils.normalize_exclude_patterns(["tests/**", " tests/** "]) == ("tests/**",)
    assert utils.normalize_exclude_patterns([".js"]) == ("**/*.js",)
    assert utils.normalize_exclude_patterns([".js,.md"]) == ("**/*.js", "**/*.md")


def test_scope_gitignore_line_variants():
    assert utils._scope_gitignore_line("", "base") is None
    assert utils._scope_gitignore_line("# comment", "base") is None
    assert utils._scope_gitignore_line(r"\#not-comment", "base") == r"base/**/\#not-comment"
    assert utils._scope_gitignore_line("foo", "") == "foo"

    assert utils._scope_gitignore_line("/foo.txt", "src") == "src/foo.txt"
    assert utils._scope_gitignore_line("foo.txt", "src") == "src/**/foo.txt"
    assert utils._scope_gitignore_line("nested/foo.txt", "src") == "src/nested/foo.txt"
    assert utils._scope_gitignore_line("build/", "src") == "src/**/build/"
    assert utils._scope_gitignore_line("!keep.txt", "src") == "!src/**/keep.txt"


def test_resolve_git_dir(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()

    git_dir = root / ".git"
    git_dir.mkdir()
    assert utils._resolve_git_dir(root) == git_dir

    # Gitdir indirection via file.
    shutil_root = tmp_path / "repo2"
    shutil_root.mkdir()
    (shutil_root / ".git").write_text("gitdir: .git/modules/x\n", encoding="utf-8")
    expected = (shutil_root / ".git" / "modules" / "x").resolve()
    assert utils._resolve_git_dir(shutil_root) == expected

    (shutil_root / ".git").write_text("not a gitdir", encoding="utf-8")
    assert utils._resolve_git_dir(shutil_root) is None

    (shutil_root / ".git").write_text("gitdir:", encoding="utf-8")
    assert utils._resolve_git_dir(shutil_root) is None

    def raise_oserror(*_args, **_kwargs):
        raise OSError("boom")

    monkeypatch.setattr(Path, "read_text", lambda *_a, **_k: raise_oserror())
    assert utils._resolve_git_dir(shutil_root) is None


def test_read_gitignore_lines_returns_empty_on_error(tmp_path, monkeypatch):
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("ignored.txt\n", encoding="utf-8")

    def boom(*_args, **_kwargs):
        raise OSError("nope")

    monkeypatch.setattr(Path, "read_text", boom)
    assert utils._read_gitignore_lines(gitignore) == []


def test_build_gitignore_base_spec_outside_root_returns_default(tmp_path):
    ignore_root = tmp_path / "root"
    ignore_root.mkdir()
    scan_root = tmp_path / "other"
    scan_root.mkdir()

    spec, ignored = utils._build_gitignore_base_spec(ignore_root, scan_root)
    assert ignored is False
    assert utils._is_ignored(spec, "anything.txt", is_dir=False) is False


def test_build_gitignore_base_spec_marks_ignored_ancestor(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".gitignore").write_text("ignored_dir/\n", encoding="utf-8")

    ignored_dir = root / "ignored_dir"
    ignored_dir.mkdir()
    scan_root = ignored_dir / "inner"
    scan_root.mkdir()

    spec, ignored = utils._build_gitignore_base_spec(root, scan_root)
    assert ignored is True
    assert utils._is_ignored(spec, "ignored_dir", is_dir=True) is True


def test_collect_files_respects_gitignore_and_exclude(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".git").mkdir()
    (root / ".git" / "info").mkdir(parents=True)
    (root / ".git" / "info" / "exclude").write_text("excluded.txt\n", encoding="utf-8")

    (root / ".gitignore").write_text(
        "ignored.txt\nsub/sub_ignored.txt\nignored_dir/\n",
        encoding="utf-8",
    )
    (root / "kept.txt").write_text("ok", encoding="utf-8")
    (root / "ignored.txt").write_text("no", encoding="utf-8")
    (root / "excluded.txt").write_text("no", encoding="utf-8")
    (root / ".hidden.txt").write_text("hidden", encoding="utf-8")
    sub = root / "sub"
    sub.mkdir()
    (sub / "sub_ignored.txt").write_text("no", encoding="utf-8")
    (sub / "sub_kept.py").write_text("print('x')\n", encoding="utf-8")
    ignored_dir = root / "ignored_dir"
    ignored_dir.mkdir()
    (ignored_dir / "never_seen.txt").write_text("no", encoding="utf-8")

    files = utils.collect_files(root, include_hidden=False, recursive=True, respect_gitignore=True)
    names = [path.name for path in files]
    assert "kept.txt" in names
    assert "sub_kept.py" in names
    assert "ignored.txt" not in names
    assert "excluded.txt" not in names
    assert ".hidden.txt" not in names

    # include hidden still respects ignore rules.
    files_hidden = utils.collect_files(root, include_hidden=True, recursive=True, respect_gitignore=True)
    names_hidden = [path.name for path in files_hidden]
    assert ".hidden.txt" in names_hidden
    assert "ignored.txt" not in names_hidden

    # no respect gitignore includes ignored/excluded.
    files_all = utils.collect_files(root, include_hidden=True, recursive=True, respect_gitignore=False)
    names_all = [path.name for path in files_all]
    assert "ignored.txt" in names_all
    assert "excluded.txt" in names_all

    # extension filtering intersects.
    files_py = utils.collect_files(root, include_hidden=True, recursive=True, respect_gitignore=False, extensions=[".py"])
    assert [path.name for path in files_py] == ["sub_kept.py"]

    # non-recursive.
    top_only = utils.collect_files(root, include_hidden=True, recursive=False, respect_gitignore=False)
    assert all(path.parent == root for path in top_only)


def test_format_path_and_ensure_positive(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    path = base / "a.txt"
    path.write_text("x", encoding="utf-8")
    assert utils.format_path(path, base) == "./a.txt"

    other = tmp_path / "other"
    other.mkdir()
    assert utils.format_path(path, other) == str(path)
    assert utils.format_path(path) == str(path)

    assert utils.ensure_positive(1, "x") == 1
    with pytest.raises(ValueError):
        utils.ensure_positive(0, "x")


def test_collect_files_non_recursive_respects_gitignore_and_filters(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".git").write_text("not a gitdir", encoding="utf-8")
    (root / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")

    (root / "kept.py").write_text("print('ok')\n", encoding="utf-8")
    (root / "ignored.txt").write_text("no", encoding="utf-8")
    (root / ".hidden.txt").write_text("hidden", encoding="utf-8")

    files = utils.collect_files(root, include_hidden=False, recursive=False, respect_gitignore=True)
    assert [path.name for path in files] == ["kept.py"]

    py_only = utils.collect_files(
        root,
        include_hidden=True,
        recursive=False,
        respect_gitignore=True,
        extensions=[".py"],
    )
    assert [path.name for path in py_only] == ["kept.py"]


def test_collect_files_exclude_patterns(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "keep.py").write_text("ok", encoding="utf-8")
    (root / "skip.js").write_text("no", encoding="utf-8")
    sub = root / "tests"
    sub.mkdir()
    (sub / "test_keep.py").write_text("no", encoding="utf-8")

    files = utils.collect_files(
        root,
        include_hidden=True,
        recursive=True,
        respect_gitignore=False,
        exclude_patterns=["tests/**", ".js"],
    )
    names = [path.name for path in files]
    assert "keep.py" in names
    assert "skip.js" not in names
    assert "test_keep.py" not in names
