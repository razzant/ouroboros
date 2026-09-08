"""Recover tool evidence through its verified redacted observability source."""

from __future__ import annotations

import copy
import gzip
import json
import pathlib
import zlib
from types import SimpleNamespace

import pytest

from ouroboros import artifacts, observability
from ouroboros.loop_tool_execution import _truncate_tool_result, process_tool_results
from ouroboros.review_evidence import build_task_acceptance_evidence
from ouroboros.review_evidence_sections import _accept_enforce_budget
from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request
from ouroboros.tools.core import _read_file
from ouroboros.tools.registry import ToolContext

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
    )
    row = trace["tool_calls"][0]
    assert row["result_partial"] is True
    source = artifacts.task_artifact_dir_path(tmp_path, ctx.task_id) / row["result_source_ref"]["path"]
    if remove_primary:
        source.unlink()
    return ctx, row


def _evidence(ctx, row, *, budget=100_000):
    return build_task_acceptance_evidence(
        ctx, llm_trace={"tool_calls": [row]}, drive_root=ctx.drive_root,
        task_id=ctx.task_id, budget_chars=budget,
    )


def _read(ctx, ref, *, start_char=0):
    args = dict(ref["read"]["arguments"])
    args["start_char"] = start_char
    return _read_file(ctx, **args)


class _Reviewer:
    def __init__(self):
        self.calls = 0

    def chat(self, **_kwargs):
        self.calls += 1
        return {"content": json.dumps({
            "verdict": "DEGRADED", "findings": [], "summary": "evidence inspected",
        })}, {}


def _dispatch(ctx, packet):
    llm = _Reviewer()
    result = run_review_request(
        ReviewRequest(surface="task_acceptance", goal="inspect the tool evidence",
                      subject="candidate", evidence=packet, task_id=ctx.task_id,
                      policy={"min_successful_slots": 1}),
        slots=[ReviewSlot(slot_id="reviewer", model="review-model")],
        drive_root=ctx.drive_root, llm=llm,
    )
    return llm.calls, result


def test_missing_primary_recovers_full_result_and_dispatches_packet_reviewer(tmp_path):
    ctx, row = _record(tmp_path)
    original = copy.deepcopy(row)
    packet = _evidence(ctx, row)
    projected = packet["tool_trajectory"][0]
    assert projected["result"] == FULL
    assert projected["result_complete"] is True
    assert projected["status"] == "error" and projected["is_error"] is True
    assert TAIL in _read(ctx, projected["result_source_ref"], start_char=19_000)
    assert "__unresolved_partial_artifacts__" not in packet
    calls, _ = _dispatch(ctx, packet)
    assert calls == 1
    assert row == original
    assert artifacts.collect_task_artifact_records(tmp_path, ctx.task_id) == []


def test_recovered_result_remains_readable_after_packet_recap(tmp_path):
    ctx, row = _record(tmp_path)
    packet = _evidence(ctx, row, budget=6_000)
    assert "__immutable_core_overflow__" not in packet
    projected = packet["tool_trajectory"][0]
    assert projected["result_complete"] is False
    assert TAIL not in projected["result"]
    assert TAIL in _read(ctx, projected["result_source_ref"], start_char=19_000)
    assert {x["status"] for x in packet["__unresolved_partial_artifacts__"]} == {
        "not_materialized_for_reviewer",
    }
    assert _dispatch(ctx, packet)[0] == 1


@pytest.mark.parametrize("stage", ["write", "readback"])
def test_publication_failure_keeps_full_inline_but_refuses_after_recap(tmp_path, monkeypatch, stage):
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
    full_packet = _evidence(ctx, row)
    projected = full_packet["tool_trajectory"][0]
    assert projected["result"] == FULL and projected["result_complete"] is True
    assert projected["result_source_ref"] == {}
    assert "__unresolved_partial_artifacts__" not in full_packet
    assert _dispatch(ctx, full_packet)[0] == 1
    packet = _evidence(ctx, row, budget=4_000)
    assert "__immutable_core_overflow__" not in packet
    assert packet["tool_trajectory_source_ref"]
    assert packet["tool_trajectory"][0]["result_source_ref"] == {}
    assert all(x["status"] == "source_unavailable" and x["source_ref"] == {}
               for x in packet["__unresolved_partial_artifacts__"])
    calls, result = _dispatch(ctx, packet)
    assert calls == 0 and result.actors[0]["status"] == "not_dispatched"


def test_explicit_missing_source_cannot_borrow_original_partial_corpus(tmp_path):
    ctx, row = _record(tmp_path)
    corpus = artifacts.persist_tool_trajectory_source(tmp_path, ctx.task_id, [row])
    packet = _accept_enforce_budget({
        "tool_trajectory": [{"tool": TOOL, "result": FULL,
                             "result_complete": True, "result_source_ref": {}}],
        "tool_trajectory_source_ref": corpus,
    }, budget=4_000)
    assert packet["tool_trajectory"][0]["result_complete"] is False
    assert packet["__unresolved_partial_artifacts__"][0]["status"] == "source_unavailable"
    assert _dispatch(ctx, packet)[0] == 0


def test_primary_source_wins_without_projection_reads_or_republication(tmp_path, monkeypatch):
    ctx, row = _record(tmp_path, remove_primary=False)
    row["trace_ref"] = {"redacted_projection_ref": {"invalid": True}}
    reads, writes = [], []
    monkeypatch.setattr(observability, "read_blob_ref", lambda *a, **k: reads.append(True))
    monkeypatch.setattr(artifacts, "persist_exact_text_source", lambda *a, **k: writes.append(True))
    packet = _evidence(ctx, row)
    assert packet["tool_trajectory"][0]["result"] == FULL
    assert packet["tool_trajectory"][0]["result_source_ref"] == row["result_source_ref"]
    assert reads == writes == []


@pytest.mark.parametrize("damage", [
    "absent_trace", "invalid_trace", "missing_blob", "wrong_size", "wrong_digest",
    "wrong_kind", "wrong_encoding", "truncated_gzip", "invalid_deflate", "non_object",
    "wrong_tool", "wrong_call", "missing_result", "null_result", "object_result",
    "missing_identity", "empty_identity", "nonstring_identity",
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
    elif damage == "nonstring_identity":
        row["tool_call_id"] = payload["tool_call_id"] = 7
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
                  "non_object", "missing_identity", "empty_identity", "nonstring_identity"}:
        row["trace_ref"]["redacted_projection_ref"] = observability.write_blob(tmp_path, payload)
    result, complete, issue = artifacts.materialize_tool_result_source(tmp_path, ctx.task_id, row)
    assert result == row["result"] and complete is False
    assert issue["status"] == "source_unavailable"
    assert issue["source_ref"] == row["result_source_ref"]
    assert "FileNotFoundError" in issue["reason"]


def test_recovery_reads_redacted_projection_even_when_raw_payload_is_retained(tmp_path, monkeypatch):
    secret = "sk-" + "secret" * 8
    full = FULL + "\n" + json.dumps({"api_key": secret})
    ctx, row = _record(tmp_path, full=full, keep_raw=True)
    manifest = json.loads(pathlib.Path(row["trace_ref"]["manifest_ref"]["path"]).read_text())
    assert secret in observability.read_blob_ref(tmp_path, manifest["full_payload_ref"])["result"]
    calls, reader = [], observability.read_blob_ref
    def track_read(drive, ref, **kwargs):
        calls.append(ref)
        return reader(drive, ref, **kwargs)
    monkeypatch.setattr(observability, "read_blob_ref", track_read)
    packet = _evidence(ctx, row)
    assert calls == [row["trace_ref"]["redacted_projection_ref"]]
    assert secret not in json.dumps(packet)
    projected = packet["tool_trajectory"][0]
    assert TAIL in projected["result"] and "***REDACTED***" in projected["result"]
    assert secret not in _read(ctx, projected["result_source_ref"], start_char=19_000)


def test_empty_string_projection_is_a_complete_tool_result(tmp_path):
    ctx, row = _record(tmp_path)
    row["trace_ref"]["redacted_projection_ref"] = observability.write_blob(
        tmp_path, {"tool": TOOL, "tool_call_id": CALL_ID, "result": ""},
    )
    packet = _evidence(ctx, row)
    assert packet["tool_trajectory"][0]["result"] == ""
    assert packet["tool_trajectory"][0]["result_complete"] is True
    assert "__unresolved_partial_artifacts__" not in packet


def test_legacy_envelope_recovers_with_matching_projection(tmp_path):
    ctx, row = _record(tmp_path)
    row["result"] = _truncate_tool_result(FULL, TOOL)
    row.pop("result_partial")
    row.pop("result_source_ref")
    packet = _evidence(ctx, row)
    projected = packet["tool_trajectory"][0]
    assert projected["result"] == FULL and projected["result_complete"] is True
    assert TAIL in _read(ctx, projected["result_source_ref"], start_char=19_000)
    assert "__unresolved_partial_artifacts__" not in packet


def test_explicit_complete_row_does_not_read_trace_or_republish(tmp_path, monkeypatch):
    ctx, row = _record(tmp_path)
    row["result"] = "0123456789\n... (truncated from 20 chars, limit=10)"
    row["result_partial"] = False
    calls = []
    monkeypatch.setattr(observability, "read_blob_ref", lambda *a, **k: calls.append(True))
    monkeypatch.setattr(artifacts, "persist_exact_text_source", lambda *a, **k: calls.append(True))
    packet = _evidence(ctx, row)
    assert packet["tool_trajectory"][0]["result"] == row["result"]
    assert "result_complete" not in packet["tool_trajectory"][0]
    assert calls == []


@pytest.mark.parametrize("publication_fails", [False, True])
def test_whole_row_shedding_retains_effective_source_or_unavailability(
    tmp_path, monkeypatch, publication_fails,
):
    secret = "sk-" + "secret" * 8
    ctx, row = _record(tmp_path, full=FULL + "\n" + json.dumps({"api_key": secret}))
    if publication_fails:
        store = artifacts.store_actor_source_bytes
        def fail_individual(*args, **kwargs):
            if kwargs.get("source_id") == CALL_ID:
                raise OSError("individual source write unavailable")
            return store(*args, **kwargs)
        monkeypatch.setattr(artifacts, "store_actor_source_bytes", fail_individual)
    calls = [row] + [{"tool": "later_tool", "result": "later evidence"} for _ in range(20)]
    packet = build_task_acceptance_evidence(
        ctx, llm_trace={"tool_calls": calls}, drive_root=ctx.drive_root,
        task_id=ctx.task_id, budget_chars=8_000,
    )
    assert "__immutable_core_overflow__" not in packet
    assert len(packet["tool_trajectory"]) == 20
    assert all(item["tool"] == "later_tool" for item in packet["tool_trajectory"])
    omitted = [item for item in packet["__unresolved_partial_artifacts__"] if item["tool"] == TOOL]
    assert len(omitted) == 1
    if publication_fails:
        assert omitted[0]["status"] == "source_unavailable" and omitted[0]["source_ref"] == {}
        assert _dispatch(ctx, packet)[0] == 0
    else:
        ref = omitted[0]["source_ref"]
        assert omitted[0]["status"] == "not_materialized_for_reviewer"
        assert ref["path"] != row["result_source_ref"]["path"]
        text = _read(ctx, ref, start_char=19_000)
        assert TAIL in text and secret not in text
        assert _dispatch(ctx, packet)[0] == 1


def test_shed_source_failure_retains_one_original_gap(tmp_path):
    ctx, row = _record(tmp_path)
    row["trace_ref"] = {}
    calls = [row] + [{"tool": "later_tool", "result": "later evidence"} for _ in range(20)]
    packet = build_task_acceptance_evidence(
        ctx, llm_trace={"tool_calls": calls}, drive_root=ctx.drive_root,
        task_id=ctx.task_id, budget_chars=8_000,
    )
    assert len(packet["tool_trajectory"]) == 20
    assert all(item["tool"] == "later_tool" for item in packet["tool_trajectory"])
    gaps = [item for item in packet["__unresolved_partial_artifacts__"] if item["tool"] == TOOL]
    assert len(gaps) == 1
    assert gaps[0]["status"] == "source_unavailable"
    assert gaps[0]["source_ref"] == row["result_source_ref"]
    assert "FileNotFoundError" in gaps[0]["reason"]
