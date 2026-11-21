"""Deferral 5: data-plane shrink guard (block accidental >30% truncation unless force).

Covers the unit (_check_data_shrink_guard) and the wired tool paths (write_file overwrite,
edit_text) on a runtime_data file: a shrinking overwrite/edit is blocked, force=true
bypasses, append is exempt, and a normal (non-shrinking) write is unaffected.
"""

from __future__ import annotations

from ouroboros.tools.core import _check_data_shrink_guard, _edit_text, _write_file
from ouroboros.tools.registry import ToolContext
from tests._workspace_git import git_repository


def _ctx(tmp_path):
    repo = git_repository(tmp_path / "repo", unborn=True)
    drive = tmp_path / "data"
    drive.mkdir()
    return ToolContext(repo_dir=repo, drive_root=drive)


def test_check_data_shrink_guard_unit(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("x" * 100, encoding="utf-8")
    assert _check_data_shrink_guard(f, "x" * 50) is not None   # 50% -> block
    assert _check_data_shrink_guard(f, "x" * 90) is None       # 90% -> allow
    assert _check_data_shrink_guard(f, "x" * 10, force=True) is None  # force bypass
    assert _check_data_shrink_guard(tmp_path / "missing.txt", "x") is None  # fresh create


def test_write_file_overwrite_shrink_blocked_force_bypass(tmp_path):
    ctx = _ctx(tmp_path)
    assert _write_file(ctx, path="notes.txt", content="A" * 200, root="runtime_data").startswith("OK")
    blocked = _write_file(ctx, path="notes.txt", content="A" * 20, root="runtime_data")
    assert "WRITE_BLOCKED" in blocked and (ctx.drive_root / "notes.txt").read_text() == "A" * 200
    forced = _write_file(ctx, path="notes.txt", content="A" * 20, root="runtime_data", force=True)
    assert forced.startswith("OK") and (ctx.drive_root / "notes.txt").read_text() == "A" * 20


def test_write_file_append_is_exempt(tmp_path):
    ctx = _ctx(tmp_path)
    _write_file(ctx, path="log.txt", content="B" * 200, root="runtime_data")
    # append never shrinks; must be allowed regardless of size
    res = _write_file(ctx, path="log.txt", content="b", mode="append", root="runtime_data")
    assert res.startswith("OK")


def test_write_file_normal_growth_unaffected(tmp_path):
    ctx = _ctx(tmp_path)
    _write_file(ctx, path="n.txt", content="C" * 100, root="runtime_data")
    res = _write_file(ctx, path="n.txt", content="C" * 100 + "more", root="runtime_data")
    assert res.startswith("OK")


def test_edit_text_shrink_blocked_force_bypass(tmp_path):
    ctx = _ctx(tmp_path)
    _write_file(ctx, path="e.txt", content="KEEP " + ("Z" * 200), root="runtime_data")
    blocked = _edit_text(ctx, path="e.txt", old_str="Z" * 200, new_str="z", root="runtime_data")
    assert "WRITE_BLOCKED" in blocked
    forced = _edit_text(ctx, path="e.txt", old_str="Z" * 200, new_str="z", root="runtime_data", force=True)
    assert forced.startswith("OK")
