"""Recover tool evidence through its verified redacted observability source.

``artifacts.materialize_tool_result_source`` is the one reader: a partial result reads
its verified actor source, then a matching redacted projection (re-persisted as a new
exact source), else it keeps the primary-source gap. Commit-review evidence and the
result review read tool results through it.
"""

from __future__ import annotations

import copy
import gzip
import json
import pathlib
import zlib
from types import SimpleNamespace

import pytest

from ouroboros import artifacts, observability
from ouroboros.loop_tool_execution import process_tool_results
from ouroboros.tools.core import _read_file
from ouroboros.tools.registry import ToolContext
from tests._tool_result_delivery_shared import measured_fit

pytestmark = pytest.mark.serial

TOOL = "ext_receipt_probe"
CALL_ID = "call-receipt"
TAIL = "DECISIVE_TAIL: operation committed, verification failed"
FULL = "output\n" + "x" * 20_000 + "\n" + TAIL


def _record(tmp_path, *, full=FULL, remove_primary=True, keep_raw=False):
    repo = tmp_path / "repo"
    repo.mkdir()
    ctx = ToolContext(repo_dir=repo, drive_root=tmp_path, task_id="source-recovery")
    ref = observability.persist_call(
        tmp_path, task_id=ctx.task_id, call_id="observability-receipt", call_type="tool",
        payload={"tool": TOOL, "tool_call_id": CALL_ID, "result": full}, keep_raw=keep_raw,
    )
    messages, trace = [], {"tool_calls": []}
    process_tool_results(
        [{"fn_name": TOOL, "tool_call_id": CALL_ID, "result": full,
          "is_error": True, "tool_args": {}, "args_for_log": {}, "trace_ref": ref,
          "result_meta": {"status": "error", "execution_status": "failed"}}],
        messages, trace, emit_progress=lambda _text, *, incident=None: None,
        tools=SimpleNamespace(_ctx=ctx),
        # A measured 4,000-token frame: the 20K-char result is projected head+tail.
        fit_candidate=measured_fit(window=4_000, reserve=1_000),
    )
    row = trace["tool_calls"][0]
    assert row["result_partial"] is True
    source = artifacts.task_artifact_dir_path(tmp_path, ctx.task_id) / row["result_source_ref"]["path"]
    if remove_primary:
        source.unlink()
    return ctx, row


def _materialize(ctx, row):
    return artifacts.materialize_tool_result_source(ctx.drive_root, ctx.task_id, row)


def _read(ctx, ref, *, start_char=0):
    args = dict(ref["read"]["arguments"])
    args["start_char"] = start_char
    return _read_file(ctx, **args)


def test_missing_primary_recovers_full_result_from_the_redacted_projection(tmp_path):
    ctx, row = _record(tmp_path)
    original = copy.deepcopy(row)
    result, complete, meta = _materialize(ctx, row)
    assert result == FULL and complete is True
    assert TAIL in _read(ctx, meta["source_ref"], start_char=19_000)
    assert row == original
    assert artifacts.collect_task_artifact_records(tmp_path, ctx.task_id) == []


@pytest.mark.parametrize("stage", ["write", "readback"])
def test_publication_failure_keeps_the_full_result_without_a_source(tmp_path, monkeypatch, stage):
    ctx, row = _record(tmp_path)
    store = artifacts.store_actor_source_bytes
    read = artifacts.read_actor_source_bytes
    if stage == "write":
        def fail_write(*args, **kwargs):
            if kwargs.get("source_id") == CALL_ID:
                raise OSError("source write unavailable")
            return store(*args, **kwargs)
        monkeypatch.setattr(artifacts, "store_actor_source_bytes", fail_write)
    else:
        def fail_readback(drive, task, ref):
            if CALL_ID in str(ref.get("path", "")):
                raise OSError("source readback unavailable")
            return read(drive, task, ref)
        monkeypatch.setattr(artifacts, "read_actor_source_bytes", fail_readback)
    result, complete, meta = _materialize(ctx, row)
    assert result == FULL and complete is True
    assert meta["source_ref"] == {}


def test_primary_source_wins_without_projection_reads_or_republication(tmp_path, monkeypatch):
    ctx, row = _record(tmp_path, remove_primary=False)
    row["trace_ref"] = {"redacted_projection_ref": {"invalid": True}}
    reads, writes = [], []
    monkeypatch.setattr(observability, "read_blob_ref", lambda *a, **k: reads.append(True))
    monkeypatch.setattr(artifacts, "persist_exact_text_source", lambda *a, **k: writes.append(True))
    result, complete, meta = _materialize(ctx, row)
    assert result == FULL and complete is True
    assert meta == {}, "the row keeps its own primary source ref"
    assert reads == writes == []


@pytest.mark.parametrize("damage", [
    "absent_trace", "invalid_trace", "missing_blob", "wrong_size", "wrong_digest",
    "wrong_kind", "wrong_encoding", "truncated_gzip", "invalid_deflate", "non_object",
    "wrong_tool", "wrong_call", "missing_result", "null_result", "object_result",
    "missing_identity", "empty_identity",
])
def test_unusable_projection_preserves_original_refusal(tmp_path, damage):
    ctx, row = _record(tmp_path)
    ref = row["trace_ref"]["redacted_projection_ref"]
    payload = {"tool": TOOL, "tool_call_id": CALL_ID, "result": FULL}
    if damage == "absent_trace":
        row.pop("trace_ref")
    elif damage == "invalid_trace":
        row["trace_ref"] = []
    elif damage == "missing_blob":
        pathlib.Path(ref["path"]).unlink()
    elif damage == "wrong_size":
        ref["size"] += 1
    elif damage == "wrong_digest":
        ref["sha256"] = "0" * 64
    elif damage == "wrong_kind":
        ref["kind"] = "text"
    elif damage == "wrong_encoding":
        ref["encoding"] = "identity"
    elif damage in ("truncated_gzip", "invalid_deflate"):
        raw = gzip.compress(b"{}")[:-3] if damage == "truncated_gzip" else (
            b"\x1f\x8b\x08\x00" + b"\x00" * 6 + b"\x07" + b"\x00" * 8
        )
        pathlib.Path(ref["path"]).write_bytes(raw)
        with pytest.raises(EOFError if damage == "truncated_gzip" else zlib.error):
            observability.read_blob_ref(tmp_path, ref)
    elif damage == "missing_identity":
        row.pop("tool_call_id")
        payload.pop("tool_call_id")
    elif damage == "empty_identity":
        row["tool_call_id"] = payload["tool_call_id"] = ""
    elif damage == "wrong_tool":
        payload["tool"] = "another_tool"
    elif damage == "wrong_call":
        payload["tool_call_id"] = "another_call"
    elif damage == "missing_result":
        payload.pop("result")
    elif damage == "null_result":
        payload["result"] = None
    elif damage == "object_result":
        payload["result"] = {}
    elif damage == "non_object":
        payload = [payload]
    if damage in {"wrong_tool", "wrong_call", "missing_result", "null_result", "object_result",
                  "non_object", "missing_identity", "empty_identity"}:
        row["trace_ref"]["redacted_projection_ref"] = observability.write_blob(tmp_path, payload)
    result, complete, issue = artifacts.materialize_tool_result_source(tmp_path, ctx.task_id, row)
    assert result == row["result"] and complete is False
    assert issue["status"] == "source_unavailable"
    assert issue["source_ref"] == row["result_source_ref"]
    assert "FileNotFoundError" in issue["reason"]


def test_both_materializers_refuse_a_projection_from_another_call(tmp_path):
    """One shared identity check guards argument and result recovery alike."""
    ctx, row = _record(tmp_path)
    row["args"] = {"path": "<TRUNCATED:4096 chars>"}
    row["trace_ref"]["redacted_projection_ref"] = observability.write_blob(
        tmp_path, {"tool": TOOL, "tool_call_id": "another-call",
                   "args": {"path": "/recovered"}, "result": FULL},
    )
    with pytest.raises(ValueError, match="does not match the tool call"):
        artifacts._matching_projection(tmp_path, row)
    args, args_complete, args_issue = artifacts.materialize_tool_args_source(tmp_path, row)
    assert args == row["args"] and args_complete is False
    assert args_issue["status"] == "source_unavailable"
    assert args_issue["reason"].startswith("argument_source_unavailable: ValueError")
    result, complete, gap = artifacts.materialize_tool_result_source(tmp_path, ctx.task_id, row)
    assert result == row["result"] and complete is False
    assert gap["status"] == "source_unavailable"


def test_recovery_reads_redacted_projection_even_when_raw_payload_is_retained(tmp_path, monkeypatch):
    secret = "sk-" + "secret" * 8
    full = FULL + "\n" + json.dumps({"api_key": secret})
    ctx, row = _record(tmp_path, full=full, keep_raw=True)
    manifest = json.loads(pathlib.Path(row["trace_ref"]["manifest_ref"]["path"]).read_text(encoding="utf-8"))
    assert secret in observability.read_blob_ref(tmp_path, manifest["full_payload_ref"])["result"]
    calls, reader = [], observability.read_blob_ref
    def track_read(drive, ref, **kwargs):
        calls.append(ref)
        return reader(drive, ref, **kwargs)
    monkeypatch.setattr(observability, "read_blob_ref", track_read)
    result, complete, meta = _materialize(ctx, row)
    assert calls == [row["trace_ref"]["redacted_projection_ref"]]
    assert complete is True and secret not in result
    assert TAIL in result and "***REDACTED***" in result
    assert secret not in _read(ctx, meta["source_ref"], start_char=19_000)


def test_empty_string_projection_is_a_complete_tool_result(tmp_path):
    ctx, row = _record(tmp_path)
    row["trace_ref"]["redacted_projection_ref"] = observability.write_blob(
        tmp_path, {"tool": TOOL, "tool_call_id": CALL_ID, "result": ""},
    )
    result, complete, _meta = _materialize(ctx, row)
    assert result == "" and complete is True


def test_legacy_envelope_recovers_with_matching_projection(tmp_path):
    ctx, row = _record(tmp_path)
    # A historical row: head cut at the retired per-name cap, legacy marker, no source.
    row["result"] = FULL[:15_000] + f"\n... (truncated from {len(FULL)} chars, limit=15000)"
    row.pop("result_partial")
    row.pop("result_source_ref")
    result, complete, meta = _materialize(ctx, row)
    assert result == FULL and complete is True
    assert TAIL in _read(ctx, meta["source_ref"], start_char=19_000)


def test_explicit_complete_row_does_not_read_trace_or_republish(tmp_path, monkeypatch):
    ctx, row = _record(tmp_path)
    row["result"] = "0123456789\n... (truncated from 20 chars, limit=10)"
    row["result_partial"] = False
    calls = []
    monkeypatch.setattr(observability, "read_blob_ref", lambda *a, **k: calls.append(True))
    monkeypatch.setattr(artifacts, "persist_exact_text_source", lambda *a, **k: calls.append(True))
    result, complete, meta = _materialize(ctx, row)
    assert result == row["result"] and complete is True
    assert meta == {}, "no source ref is minted for a complete row"
    assert calls == []
