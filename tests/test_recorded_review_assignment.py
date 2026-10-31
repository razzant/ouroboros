"""Recorded assignments and producer CAS remain authoritative after configuration changes."""

from __future__ import annotations

import copy
import dataclasses

import pytest

from ouroboros import review_operation
from ouroboros.observability import call_manifest_path, persist_call, read_call_payload
from ouroboros.review_dispatch import review_operation_binding, review_reconciliation_identity
from ouroboros.review_records import ReviewRequest, ReviewSlot
from tests.test_review_late_cas_recovery import completed_late  # noqa: F401 - pytest fixture
from tests.test_review_operation_collection import (
    PASS, TASK, _ctx, _observe, _session_run, _started, forbid_effects,  # noqa: F401
)


def _inputs(row):
    binding = row["recovery_binding"]
    return {"task_id": binding["task_id"], "surface": binding["surface"],
            "row": row, "retry_key": binding["epoch"], "root_task_id": binding["root_task_id"],
            "review_contract": binding["review_contract"]}


def _gap(actor, row):
    assert actor["operation_state"] == "custody_lost" and actor["late_result_pending"]
    assert actor["operation_id"] == row["operation_id"]
    assert actor["recovery_binding"] == row["recovery_binding"]
    assert actor["prompt_ref"] == row.get("prompt_ref", {})
    assert not actor["raw_text"] and not actor.get("semantic_verdict")
    if row.get("pending_invocation_id"):
        assert actor["usage"]["pending_invocation_id"] == row["pending_invocation_id"]


def _rewrite(root, row, suffix, manifest, payload):
    """Write a structurally valid but foreign source through the actual CAS API."""
    persist_call(
        root, task_id="late-cas", call_id=f'{row["operation_id"]}_{suffix}',
        call_type=f"review_{suffix}", payload=payload, keep_raw=True,
        manifest={key: manifest[key] for key in ("producer_complete", "review_operation_binding")
                  if key in manifest})


@pytest.fixture
def paid(completed_late, monkeypatch):  # noqa: F811 - imported fixture
    def forbidden(*_args, **_kwargs):
        pytest.fail("source collection attempted a new review or gateway connection")

    monkeypatch.setattr("ouroboros.review_substrate.ReviewCoordinator.run", forbidden)
    monkeypatch.setattr("ouroboros.claudexor_daemon.read_owned_gateway", forbidden)
    monkeypatch.setattr("ouroboros.claudexor_daemon.ensure_owned_gateway", forbidden)
    return completed_late


def test_consumer_recovers_the_full_real_paid_producer_from_its_original_assignment(paid):
    root, original_slot, row, calls, text = paid
    request, slot = review_operation.recorded_review_assignment(root, **_inputs(row))
    assert slot == original_slot and (slot.session_access or "readonly") == "readonly"
    assert review_operation_binding(request, slot, row["operation_id"]) == row["recovery_binding"]
    actor = review_operation.collect_recorded_review_actor(root, **_inputs(row))
    assert actor["raw_text"] == text and actor["raw_text"].endswith('EOF-FINAL-EVIDENCE"}]')
    assert actor["operation_state"] == "late_settled"
    assert actor["recovery_binding"] == {**row["recovery_binding"], "pending_invocation_id": ""}
    assert actor["response_ref"]["manifest_ref"] and len(calls) == 1


@pytest.mark.parametrize("physical", ["released", "settled"])
def test_collection_preserves_failure_and_no_send_facts_beside_late_lifecycle(paid, physical):
    root, _slot, row, calls, _text = paid
    manifest, payload, _refs = read_call_payload(
        root, task_id="late-cas", call_id=row["operation_id"] + "_response")
    payload["producer_outcome"].update(status="error", error="Original typed refusal",
        failure_code="reviewer_unavailable", operation_state="not_dispatched" if physical == "released" else "settled")
    payload["usage"] = {"physical_attempt_state": physical}
    payload["message"] = {"content": ""}
    _rewrite(root, row, "response", manifest, payload)
    actor = review_operation.collect_recorded_review_actor(root, **_inputs(row))
    assert actor["status"] == "error" and actor["failure_code"] == "reviewer_unavailable"
    assert actor["usage"]["physical_attempt_state"] == physical and not actor["raw_text"]
    assert actor["operation_state"] == ("not_dispatched" if physical == "released" else "late_settled")
    assert len(calls) == 1


@pytest.fixture
def historical_session(tmp_path):
    run = _session_run()
    request = ReviewRequest(**run["request"])
    slot = ReviewSlot(**run["slot_roster"][0])
    request.reconciliation_identity = review_reconciliation_identity(
        request, [slot], root_task_id=TASK, contract="recorded-output-contract")
    row = run["actors"][0]
    row["pending_invocation_id"] = f'inv-{row["operation_id"]}'
    binding = review_operation_binding(request, slot, row["operation_id"])
    row["recovery_binding"] = {**binding, "pending_invocation_id": row["pending_invocation_id"]}
    historical_slot = dataclasses.asdict(slot)
    historical_slot.pop("session_access")
    persist_call(
        tmp_path, task_id=TASK, call_id=f'{row["operation_id"]}_prompt', call_type="review_prompt",
        payload={"request": dataclasses.asdict(request), "slot": historical_slot}, keep_raw=True,
        manifest={"review_operation_binding": binding})
    return tmp_path, row, request, slot


def test_absent_historical_session_rights_reconstruct_readonly_with_exact_original_identity(historical_session):
    root, row, original_request, original_slot = historical_session
    current = dataclasses.replace(original_slot, session_access="full", session_profile="new-profile")
    assert review_operation_binding(original_request, current, row["operation_id"]) != {
        key: value for key, value in row["recovery_binding"].items() if key != "pending_invocation_id"}
    _manifest, source, _refs = read_call_payload(root, task_id=TASK, call_id=f'{row["operation_id"]}_prompt')
    assert "session_access" not in source["slot"]
    request, slot = review_operation.recorded_review_assignment(root, **_inputs(row))
    assert request == original_request and slot == original_slot
    assert (slot.session_access or "readonly") == "readonly" and slot.session_profile != current.session_profile
    assert review_operation_binding(request, slot, row["operation_id"]) == {
        key: value for key, value in row["recovery_binding"].items() if key != "pending_invocation_id"}


@pytest.mark.parametrize("defect", ["missing_prompt", "missing_request", "missing_slot", "missing_binding"])
def test_missing_original_source_is_a_gap_without_losing_the_original_binding(paid, defect):
    root, _slot, original, calls, _text = paid
    row = copy.deepcopy(original)
    row["pending_invocation_id"] = "still-owed-token"
    row["recovery_binding"]["pending_invocation_id"] = "still-owed-token"
    manifest, payload, _ = read_call_payload(root, task_id="late-cas", call_id=f'{row["operation_id"]}_prompt')
    if defect == "missing_prompt":
        call_manifest_path(root, "late-cas", f'{row["operation_id"]}_prompt').unlink()
    else:
        if defect == "missing_binding":
            manifest.pop("review_operation_binding")
        else:
            payload.pop("request" if defect == "missing_request" else "slot")
        _rewrite(root, row, "prompt", manifest, payload)
    _gap(review_operation.collect_recorded_review_actor(root, **_inputs(row)), row)
    assert len(calls) == 1


@pytest.mark.parametrize("axis", ["task_id", "root_task_id", "surface", "slot_id", "operation_id",
                                 "task_attempt", "subject_hash", "epoch", "roster_hash", "review_contract"])
@pytest.mark.parametrize("source", ["prompt", "response"])
def test_foreign_manifest_and_payload_never_supply_a_paid_answer(paid, source, axis):
    root, _slot, row, calls, _text = paid
    manifest, payload, _ = read_call_payload(root, task_id="late-cas", call_id=f'{row["operation_id"]}_{source}')
    if source == "response":
        # Even a matching pair of producer payload + manifest is not this wave.
        payload["producer_outcome"]["recovery_binding"][axis] = "foreign-value"
        manifest["review_operation_binding"] = copy.deepcopy(payload["producer_outcome"]["recovery_binding"])
    else:
        # Recompute a legitimate prompt binding where possible; the caller's
        # recorded owner must still reject this different prompt assignment.
        if axis == "slot_id":
            payload["slot"]["slot_id"] = "foreign-value"
        elif axis in {"task_id", "surface"}:
            payload["request"][axis] = "foreign-value"
        elif axis == "epoch":
            payload["request"]["retry_key"] = "foreign-value"
            payload["request"]["reconciliation_identity"][axis] = "foreign-value"
        elif axis != "operation_id":
            payload["request"]["reconciliation_identity"][axis] = "foreign-value"
        manifest["review_operation_binding"] = review_operation_binding(
            ReviewRequest(**payload["request"]), ReviewSlot(**payload["slot"]),
            "foreign-value" if axis == "operation_id" else row["operation_id"])
    _rewrite(root, row, source, manifest, payload)
    _gap(review_operation.collect_recorded_review_actor(root, **_inputs(row)), row)
    assert len(calls) == 1


def test_pending_token_in_frozen_binding_must_match_the_real_completed_producer(paid):
    root, _slot, original, calls, _text = paid
    row = copy.deepcopy(original)
    row["recovery_binding"]["pending_invocation_id"] = "foreign-producer-token"
    _gap(review_operation.collect_recorded_review_actor(root, **_inputs(row)), row)
    assert len(calls) == 1


@pytest.mark.parametrize("foreign_token", [False, True])
def test_original_session_assignment_attaches_only_with_its_exact_invocation_token(
        historical_session, monkeypatch, forbid_effects, foreign_token):  # noqa: F811
    root, original, _request, _slot = historical_session
    row = copy.deepcopy(original)
    _started(root, run_id="run-existing")
    calls = _observe(monkeypatch, {"summary": {"state": "succeeded"},
                                   "primaryOutput": {"text": PASS, "truncated": False}})
    if foreign_token:
        row["recovery_binding"]["pending_invocation_id"] = "foreign-attach-token"
    actor = review_operation.collect_recorded_review_actor(root, usage_ctx=_ctx(root), **_inputs(row))
    if foreign_token:
        _gap(actor, row)
        assert calls == [], "a foreign token never reaches even an attach-only GET"
    else:
        assert actor["operation_state"] == "late_settled" and actor["raw_text"] == PASS
        assert actor["recovery_binding"] == row["recovery_binding"]
        assert actor["usage"]["invocation_id"] == row["pending_invocation_id"]
        assert calls == [("handshake",), ("get_run", "run-existing"), ("close",)]


# ``_attempt_key`` of the ``completed_late`` request and slot as ouroboros 88a70130d computed it,
# when ReviewSlot.processing_preference defaulted to "" and an empty value stayed out of the key.
LEGACY_EMPTY_PROCESSING_KEY = "87dee94e5f8986727a2bfee91c5a213574d170c71f5e89c6a656b90dd6cf6b0f"


def _with_prompt_processing(root, row, value, binding=None):
    manifest, payload, _ = read_call_payload(root, task_id="late-cas", call_id=f'{row["operation_id"]}_prompt')
    payload["slot"]["processing_preference"] = value
    _rewrite(root, row, "prompt", {**manifest, **({"review_operation_binding": binding} if binding else {})}, payload)


def _bound(row, binding):
    """The frozen row a writer of the captured key would have recorded."""
    return {**row, "recovery_binding": dict(binding)}


def test_a_paid_review_recorded_with_the_old_empty_processing_collects_its_original_answer(paid):
    """Before processing became optional, a seat's ``""`` was unspecified: the executors sent ``None`` and
    the request key left it out. That record's paid answer is collected as recorded, at zero new sends."""
    root, original_slot, row, calls, text = paid
    assert row["recovery_binding"]["request_key"] == LEGACY_EMPTY_PROCESSING_KEY
    _with_prompt_processing(root, row, "")  # the bytes the old ``asdict(slot)`` wrote beside that key
    request, slot = review_operation.recorded_review_assignment(root, **_inputs(row))
    assert slot == original_slot and slot.processing_preference is None
    assert review_operation_binding(request, slot, row["operation_id"]) == row["recovery_binding"]
    captured = dataclasses.replace(slot, processing_preference="")
    assert review_operation_binding(request, captured, row["operation_id"]) != row["recovery_binding"]
    actor = review_operation.collect_recorded_review_actor(root, **_inputs(row))
    assert actor["raw_text"] == text and actor["operation_state"] == "late_settled"
    assert actor["recovery_binding"] == {**row["recovery_binding"], "pending_invocation_id": ""}
    assert len(calls) == 1


def test_a_captured_empty_processing_operation_keeps_its_own_identity(paid):
    """Today's captured ``""`` is a different operation: its own binding collects it as ``""``."""
    root, original_slot, row, calls, text = paid
    request, _ = review_operation.recorded_review_assignment(root, **_inputs(row))
    binding = review_operation_binding(
        request, dataclasses.replace(original_slot, processing_preference=""), row["operation_id"])
    assert binding["request_key"] != LEGACY_EMPTY_PROCESSING_KEY
    _with_prompt_processing(root, row, "", binding)
    manifest, payload, _ = read_call_payload(root, task_id="late-cas", call_id=row["operation_id"] + "_response")
    payload["producer_outcome"]["recovery_binding"] = {**binding, "pending_invocation_id": ""}
    _rewrite(root, row, "response", {**manifest, "review_operation_binding": payload["producer_outcome"]["recovery_binding"]},
             payload)
    captured = _bound(row, binding)
    _request, slot = review_operation.recorded_review_assignment(root, **_inputs(captured))
    assert slot.processing_preference == ""
    actor = review_operation.collect_recorded_review_actor(root, **_inputs(captured))
    assert actor["raw_text"] == text and actor["operation_state"] == "late_settled"
    _gap(review_operation.collect_recorded_review_actor(root, **_inputs(row)), row)
    assert len(calls) == 1


@pytest.mark.parametrize("defect", ["named_preference", "captured_binding", "foreign_key"])
def test_the_old_empty_processing_reading_needs_its_exact_legacy_binding(paid, defect):
    """Only an empty value bound to exactly the legacy key reads as unspecified; nothing else is regenerated."""
    root, original_slot, row, calls, _text = paid
    request, _ = review_operation.recorded_review_assignment(root, **_inputs(row))
    if defect == "named_preference":
        _with_prompt_processing(root, row, "fast")
    elif defect == "captured_binding":
        _with_prompt_processing(root, row, "", review_operation_binding(
            request, dataclasses.replace(original_slot, processing_preference=""), row["operation_id"]))
    else:
        _with_prompt_processing(root, row, "", {**row["recovery_binding"], "request_key": "0" * 64})
    with pytest.raises(ValueError, match="original prompt operation binding mismatch|recorded wave binding mismatch"):
        review_operation.recorded_review_assignment(root, **_inputs(row))
    _gap(review_operation.collect_recorded_review_actor(root, **_inputs(row)), row)
    assert len(calls) == 1
