"""Native read receipts carry exact delivered ranges without inventing coverage."""

import hashlib
import json

import pytest

from ouroboros.review_execution import ReviewAssignment, ReviewRouteKind
from ouroboros.review_native_episode import NativeToolRoundReviewExecutor
from ouroboros.review_substrate import ReviewRequest, ReviewSlot
from ouroboros.tools.core_file_tools import _read_file, delivered_source_prefix
from ouroboros.tools.registry import ToolContext


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _executor(root):
    request = ReviewRequest(surface="multi_model_review", goal="read evidence", session_root=str(root))
    slot = ReviewSlot(slot_id="reader", model="openai/fake-reviewer", route=ReviewRouteKind.API_CHAT)
    return NativeToolRoundReviewExecutor(ReviewAssignment(request=request, slot=slot, call_id="extent"))


def _read(root, raw, **kwargs):
    (root / "source.md").write_bytes(raw)
    ctx = ToolContext(repo_dir=root, drive_root=root / "data")
    full = _read_file(ctx, "source.md", root="system_repo", **kwargs)
    executor = _executor(root)
    executor._inspection_ctx = ctx
    return executor, ctx, full


def test_partial_long_line_receipts_cover_only_the_delivered_normalized_source(tmp_path):
    raw = ("before\r\n" + "αβγ" * 1000 + "\r\nafter\r").encode()
    normalized = raw.decode().replace("\r\n", "\n").replace("\r", "\n")
    executor, ctx, full = _read(tmp_path, raw, start_line=2, max_lines=1, start_char=7)
    view = dict(ctx.last_read_view)
    first = executor._read_extent(full, view["body_start"] + 137)
    assert first["source_revision"] == _sha(raw)
    assert first["source_bytes"] == len(raw)
    assert first["complete_sha256"] == _sha(normalized.encode())
    assert first["complete_chars"] == len(normalized)
    assert first["range_basis"] == "unicode_text_universal_newlines"
    assert first["source_masked"] is False
    assert (first["source_start_char"], first["source_end_char"]) == (14, 144)
    assert first["text_chars"] == 130
    assert first["text_sha256"] == _sha(normalized[14:144].encode())
    assert first["end_line"] < first["start_line"]  # neither chunk delivered a complete line
    second_full = _read_file(ctx, "source.md", root="system_repo", start_line=2, max_lines=1, start_char=137)
    second = executor._read_extent(second_full, ctx.last_read_view["body_start"] + 137)
    assert second["source_start_char"] == first["source_end_char"]
    assert second["source_end_char"] == 274
    assert second["source_revision"] == first["source_revision"]
    assert second["text_sha256"] == _sha(normalized[144:274].encode())
    assert second["end_line"] < second["start_line"]


def test_exact_extent_excludes_headers_trailing_notes_and_unshown_body(tmp_path):
    executor, ctx, full = _read(tmp_path, "α first\nsecond\n".encode())
    view = ctx.last_read_view
    for shown in (0, view["body_start"] - 1, view["body_start"]):
        extent = executor._read_extent(full, shown)
        assert extent["source_start_char"] == extent["source_end_char"] == 0
        assert extent["text_chars"] == 0 and extent["text_sha256"] == _sha(b"")
        assert extent["eof"] is False
    with_note = full + "\nHOST NOTE: unrelated trailing text"
    extent = executor._read_extent(with_note, len(with_note))
    assert extent["source_end_char"] == extent["complete_chars"] == len("α first\nsecond\n")
    assert extent["text_sha256"] == _sha("α first\nsecond\n".encode())
    assert extent["text_chars"] < len(with_note)


def test_extent_keeps_revision_from_the_open_that_delivered_the_body(tmp_path):
    executor, ctx, full = _read(tmp_path, b"original\r\n")
    (tmp_path / "source.md").write_bytes(b"replacement\n")
    first = executor._read_extent(full, len(full))
    assert first["source_revision"] == _sha(b"original\r\n")
    assert first["text_sha256"] == _sha(b"original\n")
    new_full = _read_file(ctx, "source.md", root="system_repo")
    second = executor._read_extent(new_full, len(new_full))
    assert second["source_revision"] == _sha(b"replacement\n")
    assert second["source_revision"] != first["source_revision"]


@pytest.mark.parametrize("field", [
    "source_revision", "source_bytes", "complete_sha256", "complete_chars",
    "source_start_char", "source_end_char", "body_chars", "source_masked", "range_basis",
])
def test_missing_source_fact_retains_line_evidence_without_exact_proof(tmp_path, field):
    executor, ctx, full = _read(tmp_path, b"original\n")
    ctx.last_read_view.pop(field)
    extent = executor._read_extent(full, len(full))
    assert extent["start_line"] == extent["end_line"] == 1
    assert extent["eof"] is True
    assert not {"source_revision", "source_start_char", "source_end_char", "text_sha256"} & extent.keys()


@pytest.mark.parametrize("changed", [
    {"source_revision": "wrong"}, {"complete_sha256": "z" * 64},
    {"source_bytes": "9"}, {"complete_chars": -1}, {"source_start_char": True},
    {"source_start_char": 10}, {"source_end_char": 100}, {"body_chars": 100},
    {"body_start": True}, {"range_basis": "raw_bytes"}, {"source_masked": True},
])
def test_invalid_source_fact_never_mints_exact_range(tmp_path, changed):
    executor, ctx, full = _read(tmp_path, b"original\n")
    ctx.last_read_view.update(changed)
    extent = executor._read_extent(full, len(full))
    assert not {"source_revision", "source_start_char", "source_end_char", "text_sha256"} & extent.keys()


def test_readonly_reader_proves_original_source_with_private_key_fixture(tmp_path):
    raw = b"safe preface\n-----BEGIN PRIVATE KEY-----\nmade-up-key-body\n-----END PRIVATE KEY-----\n"
    (tmp_path / "source.md").write_bytes(raw)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path / "data",
                      task_constraint={"mode": "local_readonly_subagent"})
    full = _read_file(ctx, "source.md", root="system_repo")
    assert ctx.last_read_view["source_masked"] is False
    assert delivered_source_prefix(ctx.last_read_view, full, len(full)) == raw.decode()
    executor = _executor(tmp_path)
    executor._inspection_ctx = ctx
    extent = executor._read_extent(full, len(full))
    assert extent["source_start_char"] == 0 and extent["source_end_char"] == len(raw)
    assert extent["text_sha256"] == _sha(raw)


def test_result_fitting_receipt_matches_exact_text_returned_to_the_reviewer(tmp_path):
    raw = ('α\\"\t' * 2000 + "\n").encode()
    (tmp_path / "source.md").write_bytes(raw)
    executor = _executor(tmp_path)
    registry, _schemas = executor._inspection_registry(str(tmp_path), tmp_path / "data")
    message = executor._execute_inspection_call(registry, {
        "id": "read-1", "function": {"name": "read_file", "arguments": json.dumps({"path": "source.md"})},
    }, {}, round_idx=1, room=1500)
    receipt = executor._tool_receipts[0]
    body_start = executor._inspection_ctx.last_read_view["body_start"]
    shown = message["content"].index("\n⚠️ RESULT TRUNCATED:")
    delivered = delivered_source_prefix(executor._inspection_ctx.last_read_view, message["content"], shown)
    assert 0 < receipt["text_chars"] < len(raw.decode())
    assert receipt["source_end_char"] == receipt["text_chars"]
    assert receipt["text_sha256"] == _sha(delivered.encode())
    assert delivered == raw.decode()[:receipt["text_chars"]]
    assert message["content"][shown:].startswith("\n⚠️ RESULT TRUNCATED:")
    assert len(json.dumps(message, ensure_ascii=False)) + 2 <= 1500


def test_exact_source_identity_keeps_the_complete_opened_path(tmp_path):
    relative = "/".join(["subdirectory" * 5] * 6 + ["source.md"])
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b"source\n")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path / "data")
    full = _read_file(ctx, relative, root="system_repo")
    executor = _executor(tmp_path)
    executor._inspection_ctx = ctx
    extent = executor._read_extent(full, len(full))
    assert len(relative) > 300
    assert extent["opened_path"] == relative
    assert extent["source_revision"] == _sha(b"source\n")
