"""Contract-aware producer outcomes shared by review gates, records and cards.

The response CAS remains the durable owner. This module finalizes its interpretation
once, before publication; legacy collection uses the same saved request and parser.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from typing import Any

from ouroboros.triad_review import (
    default_output_contract, parse_review_findings, parse_seat_answer, request_output_shape,
)

log = logging.getLogger(__name__)


def contract_parts(request: Any, slot: Any) -> list[str]:
    """Declared questions win; older requests retain their surface and delivery contract."""
    from ouroboros.review_ledger import seat_parts

    policy = request.policy or {}
    if policy.get("parts"):
        return list(policy["parts"])
    if request.surface == "multi_model_review":
        return list(seat_parts(slot))
    return {"scope_review": ["coupling"], "skill_review": ["skill"], "result_review": ["result"],
            "advisory_review": ["change"]}.get(request.surface, [])


def finalize_seat_outcome(request: Any, slot: Any, actor: Any) -> dict:
    """Finalize one declared contract, and supply every historical actor projection field."""
    if actor.outcome:
        return actor.outcome
    from ouroboros.review_projection import _review_actor_projection

    parts = contract_parts(request, slot)
    policy, usage = request.policy or {}, actor.usage or {}
    shape = policy.get("output_shape") or ("two_part" if set(parts) == {"change", "coupling"}
                                           else "array" if parts else request_output_shape(request))
    contract = {"parts": parts, "shape": shape,
                "output_contract": policy.get("output_contract") or default_output_contract(shape),
                "severity_policy": policy.get("severity_policy") or {}}
    contract_id = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    actor.model = str(usage.get("resolved_model", actor.model) or "")
    actor.actor_role = str(slot.role_hint or f"{request.surface} reviewer")
    actor.review_wave_id = str(actor.review_wave_id or request.resolved_wave_id or "")
    answers, items, findings, manifest, read_coverage = {}, [], [], {}, ""
    if parts:
        from ouroboros.review_ledger import row_verdict

        row = {**asdict(actor), "text": actor.raw_text,
               "verdict": "ERROR" if actor.status in {"error", "not_dispatched"} else "",
               "native_read_coverage": usage.get("native_read_coverage"),
               "read_provenance": usage.get("read_provenance") or usage.get("host_file_read_attestation")}
        record = parse_seat_answer(row, parts, severity_policy=policy.get("severity_policy"))
        status, answers, items = record.status, record.answers, record.parsed_items
        read_coverage, manifest = record.coverage, record.context_manifest
        findings = [finding for answer in answers.values() for finding in answer.get("findings", [])]
        usable = [answer for answer in answers.values() if answer.get("status") == "responded"]
        verdict = row_verdict(record.to_dict())
        actor.signal = verdict if verdict in {"PASS", "FAIL"} else "DEGRADED"
        actor.parsed = {"verdict": actor.signal, "findings": findings, "answers": answers} if usable else None
        actor.reason = next((str(item.get("reason") or "") for item in items if item.get("reason")), actor.reason)
        actor.parse_status = "valid" if usable else "awaiting" if status == "pending" else "malformed"
    else:
        # Historical object-shaped assessments and prose organs retain their own contract.
        actor.parsed, findings, actor.signal = parse_review_findings(actor.raw_text)
        status = "responded" if actor.signal in {"PASS", "FAIL"} else "parse_failure"
        if shape in {"report", "plan_reply"} and actor.raw_text.strip():
            actor.parse_status, status = "valid", "responded"
        if actor.status in {"error", "not_dispatched"} and actor.signal not in {"PASS", "FAIL"}:
            status = actor.status
        if actor.late_result_pending and actor.operation_state != "custody_lost" and not actor.raw_text.strip():
            status = "pending"
        findings = [{**item, "slot_id": actor.slot_id, "model": actor.model} for item in findings]
    truth = _review_actor_projection(actor, request.surface)
    for key in ("model", "transport_status", "parse_status", "semantic_verdict", "provider", "coverage", "reason"):
        setattr(actor, key, truth[key])
    if parts and usable:
        actor.semantic_verdict = verdict
    if status == "pending":
        actor.transport_status = actor.parse_status = "awaiting"
        actor.semantic_verdict = ""
    if actor.transport_status == "interrupted":
        status, actor.parse_status, actor.semantic_verdict = "interrupted", "unavailable", ""
    participates = bool(usable) if parts else actor.signal in {"PASS", "FAIL"}
    actor.quorum_contribution = status == "responded" and participates and not policy.get("additional")
    actor.enforcement_impact = ({"FAIL": "veto", "PASS": "supports_pass"}.get(actor.signal, "abstains")
                                if actor.quorum_contribution else "abstains")
    suffix = "error" if actor.status == "error" else "response"
    actor.outcome = {
        "outcome_id": f"{actor.operation_id}:{actor.slot_id}", "operation_id": actor.operation_id,
        "slot_id": actor.slot_id, "contract_id": contract_id, "parts": parts,
        "transport": actor.transport_status, "parse": actor.parse_status, "status": status,
        "verdict": actor.semantic_verdict,
        "answers": answers, "findings": findings, "parsed_items": items,
        "read_coverage": read_coverage, "context_manifest": manifest,
        "additional": bool(policy.get("additional")),
        "source_ref": {"task_id": request.task_id or "review", "call_id": f"{actor.operation_id}_{suffix}"},
        "usage_ref": {"ledger_attempt_ids": list(usage.get("ledger_attempt_ids") or []),
                      "delegated_run_id": str(usage.get("delegated_run_id") or "")},
    }
    return actor.outcome


def answer_received(usage_ctx: Any, request: Any, slot: Any) -> None:
    """The transport has returned bytes; interpretation has not completed yet."""
    from ouroboros.review_execution_projection import review_actor_progress_text

    emit = getattr(usage_ctx, "emit_progress_fn", None)
    if callable(emit):
        try:
            emit(review_actor_progress_text(request.surface, "answer_received", slot))
        except Exception:
            log.debug("review answer progress failed", exc_info=True)


def publish_seat_execution(request: Any, slot: Any, actor: Any, usage_ctx: Any, *, fallback_root: Any = None) -> None:
    """A seat publishes its actual execution when it settles, even after its drain left."""
    from ouroboros.reviewer_slot_config import record_reviewer_slot_executions

    from ouroboros.subagent_metric_events import review_execution
    try:
        review_execution(request, slot, actor, usage_ctx, fallback_root=fallback_root)
    except Exception:
        log.debug("review measurement unavailable", exc_info=True)
    try:
        record_reviewer_slot_executions(request.surface, [actor], {slot.slot_id: slot}, keep_on=usage_ctx)
    except Exception:
        log.debug("reviewer-slot last-execution write failed", exc_info=True)


def aggregate_outcomes(request: Any, slots: list, actors: list) -> dict:
    """Reduce finalized answers; no raw response is parsed by a panel consumer."""
    from ouroboros.review_projection import awaiting_panel_reason

    counted = [a for a in actors if not a.outcome.get("additional")]
    findings = [f for a in actors for f in a.outcome.get("findings", [])]
    awaiting = [a.slot_id for a in counted if a.outcome["status"] == "pending"]
    reasons = [f"{a.slot_id}:{a.error if a.status in {'error', 'not_dispatched'} else a.status}"
               for a in counted if a.status != "ok" and a.slot_id not in awaiting]
    reasons += [f"{a.slot_id}:degraded" for a in counted if a.signal == "DEGRADED" and a.slot_id not in awaiting]
    # This transport panel may contain only one part of the wave. Preserve its
    # caller's participation policy; the gate reduces the complete assigned roster.
    required = max(1, int((request.policy or {}).get("min_successful_slots") or 1))
    errors = any(a.status != "ok" for a in counted)
    signal = ("FAIL" if any(a.signal == "FAIL" for a in counted) else "PASS"
              if sum(a.signal == "PASS" for a in counted) >= required
              and not (errors and (request.policy or {}).get("fail_closed_on_errors")) else "DEGRADED")
    if awaiting:
        reasons.insert(0, awaiting_panel_reason(awaiting, len(counted), signal))
    elif signal == "DEGRADED" and not reasons:
        reasons.append(f"quorum_not_met: pass_count={sum(a.signal == 'PASS' for a in counted)} < min_successful={required}")
    return {"parsed_findings": findings, "aggregate_signal": signal, "degraded": signal == "DEGRADED",
            "degraded_reasons": reasons, "single_reviewer_no_diversity": len(slots) == 1}


def retain_collected_outcome(root, request, slot, actor, *, transcript=""):
    """Cold observation uses the saved contract and the same durable producer receipt."""
    from ouroboros.observability import persist_call
    from ouroboros.triad_review import emit_review_run_errors

    finalize_seat_outcome(request, slot, actor)
    emit_review_run_errors(None, request, [actor], drive_root=root)
    suffix = "error" if actor.status == "error" else "response"
    try:
        actor.response_ref = persist_call(
            root, task_id=request.task_id or "review", call_id=f"{actor.operation_id}_{suffix}",
            call_type=f"{request.surface}_review_collected",
            payload={"message": {"content": actor.raw_text, "session_transcript": transcript}, "usage": actor.usage,
                     "error": actor.error, "producer_outcome": {k: v for k, v in asdict(actor).items()
                         if k not in {"raw_text", "usage", "prompt_ref", "response_ref"}}},
            manifest={"surface": request.surface, "slot_id": slot.slot_id, "collection": "attach_only_observation",
                      "producer_complete": True, "review_operation_binding": actor.recovery_binding})
    except Exception:
        log.warning("Collected review outcome could not be retained: %s", actor.operation_id, exc_info=True)
        return None  # Remain deferred: collection never advertises undurable readiness.
    return actor


def ensure_producer_prompt(root, request, slot, actor):
    """An exception outside slot preparation still retains its exact assignment."""
    from ouroboros.observability import persist_call, read_call_payload

    call_id = f"{actor.operation_id}_prompt"
    try:
        return read_call_payload(root, task_id=request.task_id or "review", call_id=call_id)[2]
    except FileNotFoundError:
        return persist_call(root, task_id=request.task_id or "review", call_id=call_id,
                            call_type=f"{request.surface}_review_prompt",
                            payload={"request": asdict(request), "slot": asdict(slot)},
                            manifest={"surface": request.surface, "slot_id": slot.slot_id,
                                      "review_operation_binding": actor.recovery_binding})
