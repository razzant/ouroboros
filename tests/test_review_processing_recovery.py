"""Retained 88a70130 writer -> current generic collection, without sends.

The old slot and identity functions are exact retained source, not keys rebuilt
by the collector under test. The prompt and completed-result writer methods
ReviewCoordinator._run_slot / _persist_producer_outcome are unchanged from that
base; only the transport is fake. Delegated worker loss leaves a real prompt and
START_REQUESTED/STARTED custody, then collection may only GET the existing run.
"""
from __future__ import annotations

import copy
import dataclasses
import json
from types import SimpleNamespace

import pytest

from ouroboros import review_custody, review_dispatch, review_operation, review_substrate
from ouroboros.observability import read_call_payload
from ouroboros.review_records import ReviewActorRecord, ReviewRequest, ReviewRouteKind, ReviewSlot
from tests.fixtures import legacy_review_processing as legacy
from tests.test_review_operation_collection import (
    SURFACE, TASK, _ctx, _observe, _started, forbid_effects,  # noqa: F401 - pytest fixture
)

TEXT = json.dumps({"verdict": "FAIL", "summary": "Retained paid criticism", "findings": [
    {"item": "exact-source", "verdict": "FAIL", "severity": "critical",
     "reason": "Retained evidence " * 1000 + "EOF-PAID-CRITICISM"}]})
OPERATION = "processing-recovery"


class WorkerLost(BaseException):
    """The delegated worker died after the durable start, before outcome retention."""


def _record(root, monkeypatch, *, writer, delivery):
    request = ReviewRequest(surface=SURFACE, task_id=TASK, goal="Review the answer",
                            subject="Original answer", retry_key="processing-wave", task_attempt=1)
    slot_type = legacy.ReviewSlot if writer == "legacy" else ReviewSlot
    slot = slot_type(slot_id="s", model="codex" if delivery == "session" else "fake/model",
                     route=ReviewRouteKind.AGENT_SESSION if delivery == "session" else ReviewRouteKind.API_CHAT,
                     session_target="codex" if delivery == "session" else "", processing_preference="",
                     subagent_id="original-critic")
    sends = []

    def execute(*_args, **_kwargs):
        sends.append(delivery)
        if delivery == "session":
            _started(root, operation_id=OPERATION, run_id="existing-run")
            raise WorkerLost()
        return SimpleNamespace(message={"content": TEXT}, usage={}, raw_text=TEXT)

    with monkeypatch.context() as patch:
        if writer == "legacy":
            patch.setattr(review_custody, "_attempt_key", legacy._attempt_key)
            patch.setattr(review_dispatch, "review_reconciliation_identity", legacy.review_reconciliation_identity)
        request.reconciliation_identity = review_dispatch.review_reconciliation_identity(
            request, [slot], root_task_id=TASK)
        binding = review_dispatch.review_operation_binding(request, slot, OPERATION)
        executor = SimpleNamespace(prompt_payload=lambda: {}, prompt_chars=lambda: 1,
                                   restore_custody=lambda _state: None,
                                   set_pending_invocation_checkpoint=lambda _callback: None)
        patch.setattr(review_substrate, "_review_route_executor", lambda *_args, **_kw: executor)
        patch.setattr(review_substrate, "_execute_slot_attempt", execute)
        producer = review_substrate.ReviewCoordinator(drive_root=root, usage_ctx=_ctx(root), llm=object())
        if delivery == "session":
            with pytest.raises(WorkerLost):
                producer._run_slot(request, slot, operation_id=OPERATION)
        else:
            actor = producer._run_slot(request, slot, operation_id=OPERATION)
            assert actor.raw_text == TEXT and actor.operation_state == "settled"

    manifest, prompt, refs = read_call_payload(root, task_id=TASK, call_id=f"{OPERATION}_prompt")
    assert prompt["slot"]["processing_preference"] == ""
    assert manifest["review_operation_binding"] == binding
    current_slot = ReviewSlot(**prompt["slot"])
    assert (review_dispatch.review_operation_binding(request, current_slot, OPERATION) == binding) == (writer == "current")
    token = f"inv-{OPERATION}" if delivery == "session" else ""
    row = dataclasses.asdict(ReviewActorRecord(
        slot_id=slot.slot_id, model=slot.model, status="error", operation_id=OPERATION,
        operation_state="in_flight", late_result_pending=True, prompt_ref=refs,
        recovery_binding={**binding, "pending_invocation_id": token},
        usage={"pending_invocation_id": token} if token else {}))
    return {"authority": "host_root", "panel_id": "original-panel", "request": dataclasses.asdict(request),
            "slot_roster": [dataclasses.asdict(slot)], "actors": [row], "aggregate_signal": "DEGRADED"}, sends


def _collect(root, run):
    request = run["request"]
    return review_operation.collect_recorded_review_actor(
        root, task_id=TASK, surface=request["surface"], row=run["actors"][0],
        retry_key=request["retry_key"], root_task_id=TASK,
        review_contract=request["reconciliation_identity"]["review_contract"])


@pytest.mark.parametrize("delivery", ["api", "session"])
@pytest.mark.parametrize("writer", ["legacy", "current"])
def test_retained_writer_collects_its_paid_criticism_without_resending(
        tmp_path, monkeypatch, forbid_effects, delivery, writer):  # noqa: F811
    run, sends = _record(tmp_path, monkeypatch, writer=writer, delivery=delivery)
    original = copy.deepcopy(run)
    before = read_call_payload(tmp_path, task_id=TASK, call_id=f"{OPERATION}_prompt")[:2]
    reads = _observe(monkeypatch, {"summary": {"state": "succeeded"},
                                   "primaryOutput": {"text": TEXT, "truncated": False}})
    actor = _collect(tmp_path, run)
    assert actor["operation_state"] == "late_settled", actor
    assert "EOF-PAID-CRITICISM" in actor["raw_text"] and len(actor["raw_text"]) >= len(TEXT) - 100
    assert actor["recovery_binding"] == original["actors"][0]["recovery_binding"]
    assert actor["response_ref"]["manifest_ref"]
    assert read_call_payload(tmp_path, task_id=TASK, call_id=f"{OPERATION}_prompt")[:2] == before
    assert sends == [delivery]
    assert reads == ([("handshake",), ("get_run", "existing-run"), ("close",)] if delivery == "session" else [])


@pytest.mark.parametrize("delivery", ["api", "session"])
@pytest.mark.parametrize("axis", ["roster_hash", "root_task_id", "request_key", "pending_invocation_id"])
def test_legacy_interpretation_rejects_foreign_full_bindings(
        tmp_path, monkeypatch, forbid_effects, delivery, axis):  # noqa: F811
    run, sends = _record(tmp_path, monkeypatch, writer="legacy", delivery=delivery)
    row = run["actors"][0]
    row["recovery_binding"][axis] = "foreign-value"
    reads = _observe(monkeypatch, {"summary": {"state": "succeeded"},
                                   "primaryOutput": {"text": TEXT, "truncated": False}})
    actor = _collect(tmp_path, run)
    assert actor["operation_state"] == "custody_lost" and actor["late_result_pending"]
    assert not actor["raw_text"] and not actor.get("semantic_verdict")
    assert actor["recovery_binding"] == row["recovery_binding"] and actor["prompt_ref"] == row["prompt_ref"]
    assert sends == [delivery] and reads == []


@pytest.mark.parametrize("writer", ["legacy", "current"])
@pytest.mark.parametrize("owner", ["local", "controller", "delegated"])
def test_checkpoint_session_preserves_in_flight_deferral(
        tmp_path, monkeypatch, forbid_effects, writer, owner):  # noqa: F811
    run, sends = _record(tmp_path, monkeypatch, writer=writer, delivery="session")
    row = run["actors"][0]
    original_key = row.pop("recovery_binding")["request_key"]
    request = ReviewRequest(**run["request"])
    slot = ReviewSlot(**run["slot_roster"][0])
    reads = _observe(monkeypatch, {"summary": {"state": "running"}})
    controller = None
    if owner == "local":
        monkeypatch.setitem(review_custody._ACTIVE, original_key, SimpleNamespace(
            operation_id=OPERATION, actor=None, event=SimpleNamespace(is_set=lambda: False)))
    elif owner == "controller":
        controller = {"pid": 1}
        monkeypatch.setattr(review_operation, "controller_state", lambda _identity: "alive")
    actor, fact = review_operation._collect_pending_row(tmp_path, request, slot, row, _ctx(tmp_path), controller)
    assert fact == review_operation.DEFERRED and actor.late_result_pending
    assert not actor.raw_text and not actor.recovery_binding
    assert sends == ["session"]
    assert reads == ([("handshake",), ("get_run", "existing-run"), ("close",)] if owner == "delegated" else [])
