"""Read extents bind the actual source bytes and the delivered text projection."""

import hashlib

from ouroboros.tools.core_file_tools import _read_file, delivered_source_prefix
from ouroboros.tools.registry import ToolContext


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
