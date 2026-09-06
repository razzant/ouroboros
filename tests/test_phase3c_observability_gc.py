"""Continuity Phase 3C: child forensic refs survive headless-drive GC."""

from __future__ import annotations

import gzip
import json
import pathlib
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from ouroboros.headless import (
    copy_child_task_result,
    prepare_task_drive,
    prune_headless_task_drives,
    remove_subagent_task_drive,
)
from ouroboros.observability import persist_call, read_blob_ref, write_blob
from ouroboros.task_results import STATUS_COMPLETED, load_task_result, write_task_result


def _child(tmp_path: pathlib.Path, task_id: str) -> tuple[pathlib.Path, pathlib.Path]:
    parent = tmp_path / "data"
    parent.mkdir()
    child = prepare_task_drive(parent, task_id, "empty")
    assert child is not None
    return parent, child


def _future_now() -> float:
    return 4_000_000_000.0


def _manifest(ref: dict) -> dict:
    return json.loads(pathlib.Path(ref["path"]).read_text(encoding="utf-8"))


def _source_ref_from_visible_result(text: str) -> dict:
    prefix = "FULL_RESULT_SOURCE_JSON="
    line = next(line for line in text.splitlines() if line.startswith(prefix))
    return json.loads(line[len(prefix):])


def test_copyback_promotes_trace_manifest_and_blobs_before_headless_gc(tmp_path):
    task_id = "phase3c-trace"
    parent, child = _child(tmp_path, task_id)
    request = persist_call(
        child,
        task_id=task_id,
        call_id="llm_request",
        call_type="llm_request",
        payload={"prompt": "exact prompt", "reasoning": "exact reasoning"},
    )
    response = persist_call(
        child,
        task_id=task_id,
        call_id="llm_response",
        call_type="llm_response",
        payload={"response": "exact response"},
    )
    tool = persist_call(
        child,
        task_id=task_id,
        call_id="tool_call",
        call_type="tool_call",
        payload={"tool": "read_file", "result": "exact tool result"},
    )
    write_task_result(
        child,
        task_id,
        STATUS_COMPLETED,
        result="done",
        artifact_status="ready",
        trace_refs={
            "llm_call_refs": [{
                "request_ref": request["manifest_ref"],
                "response_ref": response["manifest_ref"],
            }],
            "tool_call_refs": [{
                "manifest_ref": tool["manifest_ref"],
                "redacted_projection_ref": tool["redacted_projection_ref"],
            }],
        },
    )

    copied = copy_child_task_result(parent, {"id": task_id, "drive_root": str(child)})

    assert copied is not None
    assert copied["child_ref_promotion"]["status"] == "complete"
    promoted_manifest_ref = copied["trace_refs"]["llm_call_refs"][0]["request_ref"]
    assert pathlib.Path(promoted_manifest_ref["path"]).is_relative_to(parent / "observability")
    promoted_manifest = _manifest(promoted_manifest_ref)
    assert read_blob_ref(parent, promoted_manifest["full_payload_ref"]) == {
        "prompt": "exact prompt",
        "reasoning": "exact reasoning",
    }
    promoted_response_ref = copied["trace_refs"]["llm_call_refs"][0]["response_ref"]
    assert read_blob_ref(
        parent, _manifest(promoted_response_ref)["full_payload_ref"]
    ) == {"response": "exact response"}
    promoted_tool_ref = copied["trace_refs"]["tool_call_refs"][0]["manifest_ref"]
    assert read_blob_ref(parent, _manifest(promoted_tool_ref)["full_payload_ref"])[
        "result"
    ] == "exact tool result"

    report = prune_headless_task_drives(parent, retention_days=7, now=_future_now())
    assert report["pruned"][0]["task_id"] == task_id
    assert not child.exists()
    assert read_blob_ref(parent, promoted_manifest["full_payload_ref"])["prompt"] == "exact prompt"
    assert read_blob_ref(parent, _manifest(promoted_response_ref)["full_payload_ref"])[
        "response"
    ] == "exact response"
    assert read_blob_ref(parent, _manifest(promoted_tool_ref)["full_payload_ref"])[
        "tool"
    ] == "read_file"


def test_pipeline_loop_outcome_trace_refs_are_rebased_and_readable_after_gc(tmp_path):
    from ouroboros.agent_task_pipeline import _store_task_result
    from ouroboros.outcomes import derive_loop_outcome

    task_id = "phase3c-loop-outcome"
    parent, child = _child(tmp_path, task_id)
    repo = tmp_path / "repo"
    repo.mkdir()
    request = persist_call(
        child,
        task_id=task_id,
        call_id="pipeline-request",
        call_type="llm_request",
        payload={"messages": [{"role": "user", "content": "exact pipeline prompt"}]},
    )
    response = persist_call(
        child,
        task_id=task_id,
        call_id="pipeline-response",
        call_type="llm_response",
        payload={"message": {"role": "assistant", "content": "exact answer"}},
    )
    tool = persist_call(
        child,
        task_id=task_id,
        call_id="pipeline-tool",
        call_type="tool_call",
        payload={"tool": "read_file", "result": "pipeline tool result"},
    )
    usage = {
        "execution_id": "phase3c-execution",
        "rounds": 1,
        "llm_call_refs": [{
            "llm_call_id": "phase3c-llm",
            "request_ref": request["manifest_ref"],
            "response_ref": response["manifest_ref"],
        }],
    }
    trace = {
        "tool_calls": [{
            "tool": "read_file",
            "tool_call_id": "pipeline-tool-call",
            "result": "pipeline tool result",
            "is_error": False,
            "trace_ref": tool,
        }],
        "reasoning_notes": [],
    }
    outcome = derive_loop_outcome("FINAL ANSWER: exact answer", usage, trace)
    _store_task_result(
        SimpleNamespace(drive_root=child, repo_dir=repo),
        {"id": task_id, "type": "task", "text": "pipeline task"},
        "FINAL ANSWER: exact answer",
        usage,
        trace,
        review_evidence={},
        loop_outcome=outcome,
    )

    copied = copy_child_task_result(parent, {"id": task_id, "drive_root": str(child)})

    assert copied is not None
    nested_refs = copied["loop_outcome"]["trace_refs"]
    nested_request = nested_refs["llm_call_refs"][0]["request_ref"]
    nested_tool = nested_refs["tool_call_refs"][0]["manifest_ref"]
    assert pathlib.Path(nested_request["path"]).is_relative_to(parent / "observability")
    assert pathlib.Path(nested_tool["path"]).is_relative_to(parent / "observability")
    prune_headless_task_drives(parent, retention_days=0, now=_future_now())
    assert not child.exists()
    assert read_blob_ref(parent, _manifest(nested_request)["full_payload_ref"])[
        "messages"
    ][0]["content"] == "exact pipeline prompt"
    assert read_blob_ref(parent, _manifest(nested_tool)["full_payload_ref"])[
        "result"
    ] == "pipeline tool result"


def test_real_truncated_tool_source_envelope_remains_actor_readable_after_gc(tmp_path):
    from ouroboros.agent_task_pipeline import _store_task_result
    from ouroboros.loop_tool_execution import process_tool_results
    from ouroboros.outcomes import derive_loop_outcome
    from ouroboros.tools.core import _read_file
    from ouroboros.tools.registry import ToolContext

    task_id = "phase3c-real-source"
    parent, child = _child(tmp_path, task_id)
    repo = tmp_path / "repo-real-source"
    repo.mkdir()
    ctx = ToolContext(repo_dir=repo, drive_root=child, task_id=task_id)
    messages: list[dict] = []
    trace = {"tool_calls": [], "reasoning_notes": []}
    exact_tail = "\nDECISIVE_SUFFIX=FAILED_AFTER_ONE_SHOT"
    full_result = "one-shot output\n" + ("x" * 100_000) + exact_tail
    process_tool_results(
        [{
            "fn_name": "run_command",
            "tool_call_id": "one-shot-call",
            "result": full_result,
            "is_error": False,
            "tool_args": {"cmd": "non-idempotent-operation"},
            "args_for_log": {"cmd": "non-idempotent-operation"},
            "trace_ref": {},
            "result_meta": {"status": "ok"},
        }],
        messages,
        trace,
        emit_progress=lambda _message, *, incident=None: None,
        tools=SimpleNamespace(_ctx=ctx),
    )
    produced_ref = _source_ref_from_visible_result(messages[0]["content"])
    request = persist_call(
        child,
        task_id=task_id,
        call_id="source-envelope-request",
        call_type="llm_request",
        payload={"messages": messages},
    )
    usage = {
        "execution_id": "source-envelope-execution",
        "rounds": 1,
        "llm_call_refs": [{
            "llm_call_id": "source-envelope-llm",
            "request_ref": request["manifest_ref"],
        }],
    }
    outcome = derive_loop_outcome("FINAL ANSWER: inspected", usage, trace)
    _store_task_result(
        SimpleNamespace(drive_root=child, repo_dir=repo),
        {"id": task_id, "type": "task", "text": "one-shot"},
        "FINAL ANSWER: inspected",
        usage,
        trace,
        review_evidence={},
        loop_outcome=outcome,
    )

    copied = copy_child_task_result(parent, {"id": task_id, "drive_root": str(child)})

    assert copied is not None
    request_ref = copied["loop_outcome"]["trace_refs"]["llm_call_refs"][0][
        "request_ref"
    ]
    prune_headless_task_drives(parent, retention_days=0, now=_future_now())
    payload = read_blob_ref(parent, _manifest(request_ref)["full_payload_ref"])
    promoted_ref = _source_ref_from_visible_result(payload["messages"][0]["content"])
    assert promoted_ref == produced_ref
    read_args = dict(promoted_ref["read"]["arguments"])
    read_args["start_char"] = 95_000
    canonical_ctx = ToolContext(repo_dir=repo, drive_root=parent, task_id=task_id)
    assert exact_tail in _read_file(canonical_ctx, **read_args)


@pytest.mark.parametrize("mismatch", ["tool", "root", "path"])
def test_task_source_read_contract_mismatch_is_typed_unavailable(tmp_path, mismatch):
    from ouroboros.artifacts import store_actor_source_bytes

    task_id = f"phase3c-source-contract-{mismatch}"
    parent, child = _child(tmp_path, task_id)
    ref = store_actor_source_bytes(
        child,
        task_id,
        category="tool_results",
        source_id="contract",
        data=b"exact source",
        extension="txt",
    )
    malformed = json.loads(json.dumps(ref))
    if mismatch == "tool":
        malformed["read"]["tool"] = "run_command"
    elif mismatch == "root":
        malformed["read"]["arguments"]["root"] = "runtime_data"
    else:
        malformed["read"]["arguments"]["path"] = (
            "source_handles/tool_results/other.txt"
        )
    write_task_result(
        child,
        task_id,
        STATUS_COMPLETED,
        result="done",
        artifact_status="ready",
        review_evidence={"exact_source_ref": malformed},
    )

    copied = copy_child_task_result(parent, {"id": task_id, "drive_root": str(child)})

    assert copied is not None
    gap = copied["review_evidence"]["exact_source_ref"]
    assert gap["availability"] == "unavailable"
    assert gap["reason"] == "invalid_ref"
    assert not (
        parent / "task_results" / "artifacts" / task_id / pathlib.Path(ref["path"])
    ).exists()
    assert prune_headless_task_drives(
        parent, retention_days=0, now=_future_now()
    )["pruned"]


def test_copyback_promotes_service_full_log_refs_in_durable_evidence_and_tool_payload(tmp_path):
    task_id = "phase3c-service"
    parent, child = _child(tmp_path, task_id)
    log_ref = write_blob(child, "READY\nfull service log\n", kind="txt")
    tool_trace = persist_call(
        child,
        task_id=task_id,
        call_id="tool_service_logs",
        call_type="tool_call",
        payload={
            "tool": "service_logs",
            "result": json.dumps({"tail": "READY", "full_log_ref": log_ref}),
        },
    )
    write_task_result(
        child,
        task_id,
        STATUS_COMPLETED,
        result="done",
        artifact_status="ready",
        trace_refs={"tool_call_refs": [{"manifest_ref": tool_trace["manifest_ref"]}]},
        verification_ledger={
            "entries": [{
                "kind": "runtime_event",
                "services": [{"log_finalization": {"full_log_ref": log_ref}}],
            }],
        },
    )

    copied = copy_child_task_result(parent, {"id": task_id, "drive_root": str(child)})

    assert copied is not None
    evidence_ref = copied["verification_ledger"]["entries"][0]["services"][0][
        "log_finalization"
    ]["full_log_ref"]
    assert read_blob_ref(parent, evidence_ref, expected_kind="txt") == "READY\nfull service log\n"
    tool_manifest = _manifest(copied["trace_refs"]["tool_call_refs"][0]["manifest_ref"])
    tool_payload = read_blob_ref(parent, tool_manifest["full_payload_ref"])
    nested_ref = json.loads(tool_payload["result"])["full_log_ref"]
    assert read_blob_ref(parent, nested_ref, expected_kind="txt") == "READY\nfull service log\n"

    prune_headless_task_drives(parent, retention_days=0, now=_future_now())
    assert not child.exists()
    assert read_blob_ref(parent, evidence_ref, expected_kind="txt").endswith("service log\n")
    assert read_blob_ref(parent, nested_ref, expected_kind="txt").startswith("READY")


def test_interrupted_live_ref_promotion_blocks_gc_until_idempotent_retry(
    tmp_path, monkeypatch,
):
    import ouroboros.observability as observability

    task_id = "phase3c-interrupted"
    parent, child = _child(tmp_path, task_id)
    trace = persist_call(
        child,
        task_id=task_id,
        call_id="tool_call",
        call_type="tool_call",
        payload={"result": "must survive"},
    )
    write_task_result(
        child,
        task_id,
        STATUS_COMPLETED,
        result="done",
        artifact_status="ready",
        trace_refs={"tool_call_refs": [{"manifest_ref": trace["manifest_ref"]}]},
    )
    real = observability.promote_call_manifest_ref
    monkeypatch.setattr(
        observability,
        "promote_call_manifest_ref",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("interrupted copy")),
    )

    copied = copy_child_task_result(parent, {"id": task_id, "drive_root": str(child)})

    assert copied is not None
    assert copied["child_ref_promotion"]["status"] == "incomplete"
    assert copied["child_ref_promotion"]["pending_refs"]
    assert remove_subagent_task_drive(parent, task_id) is False
    report = prune_headless_task_drives(parent, retention_days=0, now=_future_now())
    assert report["pruned"] == []
    assert report["skipped"][0]["reason"] == "child_refs_unpromoted"
    assert child.exists()

    monkeypatch.setattr(observability, "promote_call_manifest_ref", real)
    retried = copy_child_task_result(parent, {"id": task_id, "drive_root": str(child)})
    assert retried is not None
    assert retried["child_ref_promotion"]["status"] == "complete"
    assert retried["child_ref_promotion"]["pending_refs"] == []
    assert prune_headless_task_drives(parent, retention_days=0, now=_future_now())["pruned"]


def test_digest_mismatch_becomes_typed_unavailable_and_does_not_pin_drive(tmp_path):
    task_id = "phase3c-digest"
    parent, child = _child(tmp_path, task_id)
    ref = write_blob(child, {"result": "original"})
    with gzip.open(ref["path"], "wb") as handle:
        handle.write(b'{"result":"tampered"}')
    write_task_result(
        child,
        task_id,
        STATUS_COMPLETED,
        result="done",
        artifact_status="ready",
        trace_refs={"tool_call_refs": [{"redacted_projection_ref": ref}]},
    )

    copied = copy_child_task_result(parent, {"id": task_id, "drive_root": str(child)})

    assert copied is not None
    unavailable = copied["trace_refs"]["tool_call_refs"][0]["redacted_projection_ref"]
    assert unavailable["availability"] == "unavailable"
    assert unavailable["reason"] == "digest_mismatch"
    assert "path" not in unavailable
    assert copied["child_ref_promotion"]["unavailable_refs"]
    assert copied["child_ref_promotion"]["pending_refs"] == []
    assert prune_headless_task_drives(parent, retention_days=0, now=_future_now())["pruned"]


def test_concurrent_copyback_is_idempotent_and_copies_only_referenced_source_handle(
    tmp_path,
):
    from ouroboros.artifacts import store_actor_source_bytes

    task_id = "phase3c-source-handles"
    parent, child = _child(tmp_path, task_id)
    source_bytes = b"actor promised source"
    source_ref = store_actor_source_bytes(
        child,
        task_id,
        category="tool_results",
        source_id="tool",
        data=source_bytes,
        extension="txt",
    )
    source = child / "task_results" / "artifacts" / task_id / source_ref["path"]
    unreferenced_source_bytes = b"unreferenced source handle"
    unreferenced_source_ref = store_actor_source_bytes(
        child,
        task_id,
        category="tool_results",
        source_id="unused",
        data=unreferenced_source_bytes,
        extension="txt",
    )
    unreferenced_source = (
        child
        / "task_results"
        / "artifacts"
        / task_id
        / unreferenced_source_ref["path"]
    )
    unrelated = child / "task_results" / "artifacts" / task_id / "unrelated.txt"
    unrelated.write_text("must not copy", encoding="utf-8")
    unreferenced_blob = write_blob(child, {"unreferenced": True})
    trace = persist_call(
        child,
        task_id=task_id,
        call_id="duplicate-tool-call",
        call_type="tool_call",
        payload={"result": "copy once by content identity"},
    )
    write_task_result(
        child,
        task_id,
        STATUS_COMPLETED,
        result="done",
        artifact_status="ready",
        trace_refs={"tool_call_refs": [{"manifest_ref": trace["manifest_ref"]}]},
        review_evidence={"exact_source_ref": source_ref},
    )

    task = {"id": task_id, "drive_root": str(child)}
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(
            pool.map(
                lambda _ordinal: copy_child_task_result(parent, task),
                range(2),
            )
        )

    assert first is not None and second is not None
    assert first["child_ref_promotion"] == second["child_ref_promotion"]
    assert first["trace_refs"] == second["trace_refs"]
    assert first["review_evidence"]["exact_source_ref"] == source_ref
    copied_source = parent / "task_results" / "artifacts" / task_id / source.relative_to(
        child / "task_results" / "artifacts" / task_id
    )
    assert copied_source.read_text(encoding="utf-8") == "actor promised source"
    copied_unreferenced_source = (
        parent
        / "task_results"
        / "artifacts"
        / task_id
        / unreferenced_source.relative_to(
            child / "task_results" / "artifacts" / task_id
        )
    )
    assert not copied_unreferenced_source.exists()
    assert not (parent / "task_results" / "artifacts" / task_id / "unrelated.txt").exists()
    assert not (parent / "observability" / "blobs" / pathlib.Path(unreferenced_blob["path"]).name).exists()


def test_copyback_source_handle_promotion_survives_a_lost_write_race(tmp_path, monkeypatch):
    """The Windows interleaving of the concurrent copy-back (CI 33579445704): two
    copiers both miss the destination handle, the winner's identical bytes land,
    and the loser's own os.replace over that destination is refused as a sharing
    violation (CPython opens files without FILE_SHARE_DELETE). The promotion's
    postcondition is the VERIFIED handle at the destination, not authorship of the
    write, so the loser must publish the same complete custody projection — not an
    incomplete promotion with a pending ref."""
    from ouroboros import artifacts as artifacts_module
    from ouroboros.artifacts import store_actor_source_bytes

    task_id = "phase3c-lost-write-race"
    parent, child = _child(tmp_path, task_id)
    source_ref = store_actor_source_bytes(
        child, task_id, category="tool_results", source_id="tool",
        data=b"actor promised source", extension="txt",
    )
    write_task_result(
        child, task_id, STATUS_COMPLETED, result="done", artifact_status="ready",
        review_evidence={"exact_source_ref": source_ref},
    )
    real_store = artifacts_module.store_actor_source_bytes

    def _losing_store(*args, **kwargs):
        real_store(*args, **kwargs)  # the concurrent winner's identical copy lands
        raise PermissionError("[WinError 32] the file is in use by another process")

    monkeypatch.setattr(artifacts_module, "store_actor_source_bytes", _losing_store)

    copied = copy_child_task_result(parent, {"id": task_id, "drive_root": str(child)})

    promotion = copied["child_ref_promotion"]
    assert promotion["status"] == "complete"
    assert promotion["promoted_source_handle_count"] == 1
    assert promotion["pending_refs"] == []
    assert copied["review_evidence"]["exact_source_ref"] == source_ref
    promoted = (
        parent / "task_results" / "artifacts" / task_id / source_ref["path"]
    )
    assert promoted.read_bytes() == b"actor promised source"


def test_store_actor_source_bytes_does_not_rewrite_an_identical_handle(tmp_path, monkeypatch):
    """Content-addressed write-once: the digest is in the name, so re-storing the
    same bytes must not replace the file at all — that replace is the operation a
    concurrent copier can lose (and on Windows must lose, when a reader holds the
    destination open)."""
    from ouroboros import artifacts as artifacts_module
    from ouroboros.artifacts import store_actor_source_bytes

    task_id = "phase3c-write-once"
    drive = tmp_path / "data"
    drive.mkdir()
    first = store_actor_source_bytes(
        drive, task_id, category="tool_results", source_id="tool",
        data=b"exact bytes", extension="txt",
    )
    target = drive / "task_results" / "artifacts" / task_id / first["path"]
    before = target.stat()
    writes = []
    real_write = artifacts_module.write_bytes_atomic
    monkeypatch.setattr(
        artifacts_module, "write_bytes_atomic",
        lambda path, content, **kw: (writes.append(str(path)), real_write(path, content, **kw))[1],
    )

    again = store_actor_source_bytes(
        drive, task_id, category="tool_results", source_id="tool",
        data=b"exact bytes", extension="txt",
    )

    assert again == first
    assert writes == []
    after = target.stat()
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    # Different bytes are a DIFFERENT handle and are still written.
    other = store_actor_source_bytes(
        drive, task_id, category="tool_results", source_id="tool",
        data=b"other bytes", extension="txt",
    )
    assert other["path"] != first["path"] and writes


def test_legacy_missing_child_ref_is_typed_gap_without_permanent_retention(tmp_path):
    task_id = "phase3c-legacy-gap"
    parent, child = _child(tmp_path, task_id)
    missing = {
        "sha256": "b" * 64,
        "path": str(child / "observability" / "blobs" / (("b" * 64) + ".json.gz")),
        "kind": "json",
        "encoding": "gzip",
        "size": 12,
        "compressed_size": 20,
    }
    write_task_result(
        child,
        task_id,
        STATUS_COMPLETED,
        result="legacy",
        artifact_status="ready",
        trace_refs={"tool_call_refs": [{"redacted_projection_ref": missing}]},
    )

    copied = copy_child_task_result(parent, {"id": task_id, "drive_root": str(child)})

    assert copied is not None
    gap = copied["trace_refs"]["tool_call_refs"][0]["redacted_projection_ref"]
    assert gap["availability"] == "unavailable"
    assert gap["reason"] == "source_missing"
    assert copied["child_ref_promotion"]["status"] == "complete"
    assert prune_headless_task_drives(parent, retention_days=0, now=_future_now())["pruned"]


def test_startup_sweep_retries_only_pending_refs_then_prunes_without_manual_copyback(
    tmp_path, monkeypatch,
):
    import ouroboros.observability as observability
    import ouroboros.server_maintenance as server_maintenance

    task_id = "phase3c-startup-retry"
    parent, child = _child(tmp_path, task_id)
    first_trace = persist_call(
        child,
        task_id=task_id,
        call_id="startup-retry-first",
        call_type="tool_call",
        payload={"result": "already promoted before interruption"},
    )
    pending_trace = persist_call(
        child,
        task_id=task_id,
        call_id="startup-retry-pending",
        call_type="tool_call",
        payload={"result": "survive startup retry"},
    )
    write_task_result(
        child,
        task_id,
        STATUS_COMPLETED,
        result="stale child result",
        artifact_status="ready",
        artifacts=[{"kind": "stale_child_artifact", "path": "child-only"}],
        root_phase_checkpoint={"post_task_synthesis": "pending_once"},
        trace_refs={
            "tool_call_refs": [
                {"manifest_ref": first_trace["manifest_ref"]},
                {"manifest_ref": pending_trace["manifest_ref"]},
            ]
        },
    )
    real = observability.promote_call_manifest_ref

    def _interrupt_pending(*args, **kwargs):
        ref = args[2] if len(args) > 2 else kwargs.get("ref") or {}
        if ref.get("call_id") == "startup-retry-pending":
            raise OSError("first copy interrupted")
        return real(*args, **kwargs)

    monkeypatch.setattr(
        observability,
        "promote_call_manifest_ref",
        _interrupt_pending,
    )
    copied = copy_child_task_result(parent, {"id": task_id, "drive_root": str(child)})
    assert copied is not None
    assert copied["child_ref_promotion"]["status"] == "incomplete"

    canonical_artifact = {
        "kind": "canonical_newer_artifact",
        "path": str(parent / "canonical-newer.txt"),
    }
    write_task_result(
        parent,
        task_id,
        STATUS_COMPLETED,
        result="canonical newer result",
        artifact_status="ready_with_changes",
        artifact_bundle={"status": "ready_with_changes", "artifacts": [canonical_artifact]},
        artifacts=[canonical_artifact],
        artifact_finalized_at="2000-01-01T00:00:00+00:00",
        root_phase_checkpoint={
            "post_task_synthesis": "completed",
            "canonical_newer": True,
        },
    )
    monkeypatch.setattr(observability, "promote_call_manifest_ref", real)
    # The sweep reads its drive root from its owner module (v7 server split).
    monkeypatch.setattr(server_maintenance, "DATA_DIR", parent)
    monkeypatch.setenv("OUROBOROS_GC_RETENTION_DAYS", "1")

    server_maintenance._startup_prune_sweeps()

    settled = load_task_result(parent, task_id) or {}
    assert settled["child_ref_promotion"]["status"] == "complete"
    assert settled["result"] == "canonical newer result"
    assert settled["artifact_status"] == "ready_with_changes"
    assert settled["artifacts"] == [canonical_artifact]
    assert settled["artifact_bundle"] == {
        "status": "ready_with_changes",
        "artifacts": [canonical_artifact],
    }
    assert settled["root_phase_checkpoint"] == {
        "post_task_synthesis": "completed",
        "canonical_newer": True,
    }
    assert not child.exists()
    promoted_first = settled["trace_refs"]["tool_call_refs"][0]["manifest_ref"]
    promoted_pending = settled["trace_refs"]["tool_call_refs"][1]["manifest_ref"]
    assert read_blob_ref(parent, _manifest(promoted_first)["full_payload_ref"])[
        "result"
    ] == "already promoted before interruption"
    assert read_blob_ref(parent, _manifest(promoted_pending)["full_payload_ref"])[
        "result"
    ] == "survive startup retry"


def test_startup_prune_retries_missing_pending_source_into_typed_gap(
    tmp_path, monkeypatch,
):
    import ouroboros.observability as observability

    task_id = "phase3c-startup-missing"
    parent, child = _child(tmp_path, task_id)
    trace = persist_call(
        child,
        task_id=task_id,
        call_id="startup-missing",
        call_type="tool_call",
        payload={"result": "lost before retry"},
    )
    write_task_result(
        child,
        task_id,
        STATUS_COMPLETED,
        result="done",
        artifact_status="ready",
        trace_refs={"tool_call_refs": [{"manifest_ref": trace["manifest_ref"]}]},
    )
    real = observability.promote_call_manifest_ref
    monkeypatch.setattr(
        observability,
        "promote_call_manifest_ref",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("interrupted")),
    )
    copied = copy_child_task_result(parent, {"id": task_id, "drive_root": str(child)})
    assert copied is not None
    assert copied["child_ref_promotion"]["status"] == "incomplete"
    pathlib.Path(trace["manifest_ref"]["path"]).unlink()
    monkeypatch.setattr(observability, "promote_call_manifest_ref", real)

    report = prune_headless_task_drives(
        parent, retention_days=0, now=_future_now()
    )

    assert report["promotion_retry"]["completed"] == [task_id]
    assert report["pruned"][0]["task_id"] == task_id
    settled = load_task_result(parent, task_id) or {}
    gap = settled["trace_refs"]["tool_call_refs"][0]["manifest_ref"]
    assert gap["availability"] == "unavailable"
    assert gap["reason"] == "source_missing"
    assert settled["child_ref_promotion"]["status"] == "complete"


def test_periodic_maintenance_invokes_pending_ref_promotion_sweep(
    tmp_path, monkeypatch,
):
    import ouroboros.observability as observability
    import ouroboros.server_maintenance as server_maintenance
    import supervisor.task_lifecycle as task_lifecycle
    import supervisor.terminal_delivery as terminal_delivery

    calls: list[pathlib.Path] = []
    # The cadence state and drive root live in the maintenance owner (v7 server split).
    monkeypatch.setattr(server_maintenance, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server_maintenance.time, "time", lambda: 10_000.0)
    monkeypatch.setattr(server_maintenance, "_LAST_CANCEL_INTENT_SWEEP", [0.0])
    monkeypatch.setattr(task_lifecycle, "sweep_cancel_intents", lambda: {})
    monkeypatch.setattr(terminal_delivery, "replay_pending_deliveries", lambda _root: None)
    monkeypatch.setattr(
        observability,
        "retry_pending_child_ref_promotions",
        lambda root: calls.append(pathlib.Path(root)) or {},
        raising=False,
    )

    server_maintenance._periodic_supervisor_maintenance([10_000.0], [10_000.0])

    assert calls == [tmp_path]
