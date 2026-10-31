"""A completed paid review remains usable after its waiter/context disappears."""
import copy
import json
import threading
from types import SimpleNamespace

import pytest

from ouroboros import review_custody as custody
from ouroboros.observability import call_manifest_path, read_call_payload
from ouroboros.review_records import ReviewRequest, ReviewSlot
from ouroboros.review_substrate import ReviewCoordinator


def context(root, **changes):
    return SimpleNamespace(drive_root=root, task_id="late-cas", task_attempt=1,
                           task_metadata={"root_task_id": "root-cas"},
                           pending_events=[], event_queue=None, **changes)


def _request(**changes):
    values = dict(surface="multi_model_review", task_id="late-cas", goal="review",
                  retry_key="commit_review:subject-sha:cycle-1", call_type="review",
                  policy={"output_contract": "JSON findings array"})
    return ReviewRequest(**{**values, **changes})


@pytest.fixture
def completed_late(tmp_path, monkeypatch, request):
    release, settled = threading.Event(), threading.Event()
    calls = []
    text = json.dumps([{"item": "full source", "verdict": getattr(request, "param", "PASS"), "severity": "critical",
                        "reason": "complete source " * 8000 + "EOF-FINAL-EVIDENCE"}])

    class Slow:
        def chat(self, **kwargs):
            calls.append(kwargs)
            assert release.wait(10)
            return {"content": text}, {"prompt_tokens": 2, "completion_tokens": 1}

    original_settle = custody._settle_review_attempt
    def settle(*args, **kwargs):
        try:
            return original_settle(*args, **kwargs)
        finally:
            settled.set()
    monkeypatch.setattr(custody, "_settle_review_attempt", settle)
    monkeypatch.setattr(custody, "_logical_timeout", lambda *_: 0.03)
    slot = ReviewSlot("row-a", "fake/model", timeout_sec=0.03)
    ctx = context(tmp_path)
    first = ReviewCoordinator(llm=Slow(), drive_root=tmp_path, usage_ctx=ctx).run(_request(), [slot])
    row = first.actors[0]
    assert row["operation_state"] == "in_flight"
    release.set()
    assert settled.wait(10)
    assert custody._attempt_key(_request(), slot) not in custody._ACTIVE
    return tmp_path, slot, row, calls, text


def recover(parts, *, req=None, slot=None, ctx=None, row=None):
    root, original_slot, original_row, calls, _ = parts
    slot, row = slot or original_slot, row or original_row
    ctx = ctx or context(root)
    ctx._review_frozen_rows = {"multi_model_review": {slot.slot_id: copy.deepcopy(row)}}
    class Forbidden:
        def chat(self, **kwargs):
            calls.append(kwargs)
            raise AssertionError("reconciliation bought another physical review")
    return ReviewCoordinator(llm=Forbidden(), drive_root=root, usage_ctx=ctx).run(
        req or _request(reconcile_only=True), [slot])


def test_fresh_context_reaggregates_full_late_source_without_second_attempt(completed_late):
    root, slot, row, calls, text = completed_late
    result = recover(completed_late)
    actor = result.actors[0]
    assert result.aggregate_signal == "PASS"
    assert actor["operation_id"] == row["operation_id"]
    assert actor["operation_state"] == "late_settled"
    assert actor["raw_text"] == text and "EOF-FINAL-EVIDENCE" in actor["raw_text"]
    assert actor["response_ref"]["manifest_ref"]
    assert len(calls) == 1
    manifest, payload, _ = read_call_payload(root, task_id="late-cas", call_id=row["operation_id"]+"_response")
    assert manifest["producer_complete"] is True
    assert payload["producer_outcome"]["operation_state"] == "settled"


@pytest.mark.parametrize("axis", ["task", "root", "slot", "route", "operation", "subject", "contract", "roster", "epoch", "invocation"])
def test_mismatched_identity_never_borrows_a_late_pass(completed_late, axis):
    root, slot, row, calls, _ = completed_late
    req, ctx, row = _request(reconcile_only=True), context(root), copy.deepcopy(row)
    if axis == "task": req.task_id = "different-task"
    if axis == "root": ctx.task_metadata = {"root_task_id": "different-root"}
    if axis == "slot": slot = ReviewSlot("different-row", slot.model)
    if axis == "route": slot = ReviewSlot(slot.slot_id, slot.model, session_profile="different-profile")
    if axis == "operation": row["operation_id"] = "different-operation"
    if axis == "subject": req.retry_key = "commit_review:different-subject:cycle-1"
    if axis == "contract": req.policy = {"output_contract": "different output contract"}
    if axis == "roster": req.reconciliation_identity = {"roster_hash": "different-roster"}
    if axis == "epoch": req.reconciliation_identity = {"epoch": "different-epoch"}
    if axis == "invocation": row["pending_invocation_id"] = "different-invocation"
    result = recover(completed_late, req=req, slot=slot, ctx=ctx, row=row)
    assert result.aggregate_signal != "PASS"
    assert result.actors[0]["late_result_pending"]
    assert len(calls) == 1


@pytest.mark.parametrize("defect", ["partial", "missing", "tampered"])
def test_incomplete_or_unreadable_cas_keeps_custody_without_redispatch(completed_late, defect):
    root, _, row, calls, _ = completed_late
    path = call_manifest_path(root, "late-cas", row["operation_id"]+"_response")
    manifest = json.loads(path.read_text())
    if defect == "missing": path.unlink()
    elif defect == "partial":
        manifest.pop("producer_complete")
        path.write_text(json.dumps(manifest))
    else:
        from pathlib import Path
        Path(manifest["full_payload_ref"]["path"]).write_bytes(b"broken-gzip")
    result = recover(completed_late)
    assert result.aggregate_signal != "PASS"
    assert len(calls) == 1


@pytest.mark.parametrize("completed_late", ["FAIL"], indirect=True)
def test_late_fail_keeps_its_full_source_and_veto(completed_late):
    result = recover(completed_late)
    assert result.aggregate_signal == "FAIL"
    assert result.actors[0]["raw_text"] == completed_late[4]
    assert result.actors[0]["operation_state"] == "late_settled"
    assert len(completed_late[3]) == 1


# The roster_hash the pre-#1609 producer (d00559d7f, before ReviewSlot.session_access
# existed) recorded for these rosters, computed by running that producer's own code.
PRE_FIELD_ROSTER_HASH = {
    "api": "a8c84df55e7932facf7da821a9133ecaf5fedb8a5f273a20c043dfa81a3ef82b",
    "api+session": "e1b45dd5faf476416405b5802d32bd4b1a491e27ea90a50f9bfb6108378be269",
}


def _roster_hash(*slots):
    from ouroboros.review_dispatch import review_reconciliation_identity

    return review_reconciliation_identity(_request(), list(slots), root_task_id="root", contract="c")["roster_hash"]


def test_read_only_seats_keep_the_roster_identity_recorded_before_seats_carried_rights():
    from ouroboros.review_records import ReviewRouteKind

    api = ReviewSlot("row-a", "fake/model", timeout_sec=0.03)
    session = dict(route=ReviewRouteKind.AGENT_SESSION, session_target="codex=gpt-5", session_profile="p1")
    assert _roster_hash(api) == PRE_FIELD_ROSTER_HASH["api"]
    for access in ("", "readonly"):
        assert _roster_hash(api, ReviewSlot("row-s", "codex=gpt-5", session_access=access, **session)) \
            == PRE_FIELD_ROSTER_HASH["api+session"], access
    writable = {_roster_hash(api, ReviewSlot("row-s", "codex=gpt-5", session_access=access, **session))
                for access in ("workspace_write", "full")}
    assert len(writable) == 2 and PRE_FIELD_ROSTER_HASH["api+session"] not in writable  # rights are identity


@pytest.fixture
def pre_field_late(monkeypatch, request):
    """``completed_late`` produced as before seats carried rights: the real producer
    serializes every ReviewSlot without the ``session_access`` field."""
    import dataclasses

    real = dataclasses.asdict

    def pre_field(obj, *args, **kwargs):
        values = real(obj, *args, **kwargs)
        if isinstance(obj, ReviewSlot):
            values.pop("session_access")
        return values

    with monkeypatch.context() as patch:
        patch.setattr(dataclasses, "asdict", pre_field)
        return request.getfixturevalue("completed_late")


def test_a_paid_result_recorded_before_the_upgrade_is_collected_after_it(pre_field_late):
    root, _slot, row, calls, text = pre_field_late
    _manifest, payload, _ = read_call_payload(root, task_id="late-cas", call_id=row["operation_id"] + "_response")
    assert payload["producer_outcome"]["recovery_binding"]["roster_hash"] == PRE_FIELD_ROSTER_HASH["api"]
    result = recover(pre_field_late)
    assert result.aggregate_signal == "PASS"
    assert result.actors[0]["operation_state"] == "late_settled" and result.actors[0]["raw_text"] == text
    assert len(calls) == 1  # collected, never bought again
