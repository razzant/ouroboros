"""Read extents bind the actual source bytes and the delivered text projection."""

import hashlib
import os

import pytest

from ouroboros.tool_access import resource_root_path
from ouroboros.tools.core_file_tools import _read_file, delivered_source_prefix
from ouroboros.tools.registry import ToolContext, ToolRegistry


def test_raw_revision_and_subline_range_survive_newline_normalization(tmp_path):
    raw = "first\r\nlong αβγ second\rthird\n".encode("utf-8")
    path = tmp_path / "source.md"
    path.write_bytes(raw)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path / "data")
    result = _read_file(ctx, "source.md", root="system_repo", start_line=2, max_lines=1, start_char=7)
    view = ctx.last_read_view
    normalized = raw.decode().replace("\r\n", "\n").replace("\r", "\n")
    assert view["source_revision"] == hashlib.sha256(raw).hexdigest()
    assert view["source_bytes"] == len(raw)
    assert view["complete_sha256"] == hashlib.sha256(normalized.encode()).hexdigest()
    assert view["range_basis"] == "unicode_text_universal_newlines"
    body = delivered_source_prefix(view, result, len(result))
    assert body == normalized[view["source_start_char"]:view["source_end_char"]]
    assert body == "γ second\n"
    # A source rewrite cannot retroactively change the previous read's revision.
    path.write_bytes(b"changed\n")
    assert view["source_revision"] == hashlib.sha256(raw).hexdigest()
    _read_file(ctx, "source.md", root="system_repo")
    assert ctx.last_read_view["source_revision"] == hashlib.sha256(b"changed\n").hexdigest()


def test_missing_read_does_not_inherit_a_source_revision(tmp_path):
    (tmp_path / "present.md").write_text("known source\n")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path / "data")
    _read_file(ctx, "present.md", root="system_repo")
    assert ctx.last_read_view["source_revision"]
    _read_file(ctx, "absent.md", root="system_repo")
    assert ctx.last_read_view is None


def _registry(tmp_path):
    """A real registry over one fixture tree; each call starts a new task context."""
    for name in ("repo", "data"):
        (tmp_path / name).mkdir(exist_ok=True)
    ctx = ToolContext(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data", task_id="reread-task")
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    return registry


@pytest.mark.parametrize(("root", "window"), [
    pytest.param("active_workspace", {}, id='active_workspace-window0'),
    pytest.param("active_workspace", {"max_chars": 4}, id='active_workspace-window4'),
    pytest.param("system_repo", {"start_line": 2}, id='system_repo-window1'),
    pytest.param("runtime_data", {"max_lines": 1}, id='runtime_data-window2'),
    pytest.param("task_drive", {"start_char": 3}, id='task_drive-window3'),
])
def test_a_repeat_returns_what_a_fresh_context_reads_on_every_branch_and_window(tmp_path, root, window):
    """Read history is not a read input. On both read_file branches and for each slice
    selector, the same view read again in one task, after another window, is the typed
    result a fresh task context returns: the same text and the same read evidence."""
    registry = _registry(tmp_path)
    base = resource_root_path(registry._ctx, root)
    base.mkdir(parents=True, exist_ok=True)
    (base / "repeat.txt").write_text("alpha line\nbeta line\ngamma line\n", encoding="utf-8")
    args = {"root": root, "path": "repeat.txt", **window}
    first = registry.execute_result("read_file", args)
    assert first.status == "ok" and first.meta["read_view"]["body_chars"] > 0, first.text
    other = registry.execute_result("read_file", {"root": root, "path": "repeat.txt", "start_line": 3})
    assert other.status == "ok" and other.text != first.text, other.text
    repeat = registry.execute_result("read_file", args)
    fresh = _registry(tmp_path).execute_result("read_file", args)
    assert repeat.text == first.text == fresh.text, repeat.text
    assert repeat == fresh


@pytest.mark.parametrize("root", ["system_repo", "task_drive"])
def test_same_size_rewrite_with_restored_mtime_returns_the_new_bytes(tmp_path, root):
    """Replaced bytes with the old size and mtime_ns are a new source: the read returns
    them and their revision exactly as a fresh task context does."""
    registry = _registry(tmp_path)
    base = resource_root_path(registry._ctx, root)
    base.mkdir(parents=True, exist_ok=True)
    path = base / "probe.txt"
    path.write_bytes(b"AAAAA\n")
    before = path.stat()
    args = {"root": root, "path": "probe.txt"}
    assert "AAAAA" in registry.execute_result("read_file", args).text
    path.write_bytes(b"BBBBB\n")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = path.stat()
    assert (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns)
    second = registry.execute_result("read_file", args)
    assert "BBBBB" in second.text and "AAAAA" not in second.text, second.text
    assert second.meta["read_view"]["source_revision"] == hashlib.sha256(b"BBBBB\n").hexdigest()
    assert second == _registry(tmp_path).execute_result("read_file", args)
