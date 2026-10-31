"""The intention-ledger record of one plan ask — a narrow adapter over ``review_ledger``.

One review-ledger record per ask (``surface="intention"``): the seats the mind named (their
requested and effective route, model, effort and profile, the model the substrate observed,
their receipts and capability deltas), the retained delivery of each seat and its raw reply,
costs and custody state. It is written at dispatch (``write_intention_record``) and revised
as each seat settles (``settle_ledger_seat``); its verdict is always ``NOT_PERFORMED`` with
the reason ``intention_no_verdict_question`` — an intention is weighed by the author, never
voted on, so this adapter never asks the change/coupling reducer for a verdict. The plan
record (``plan_record``) keeps the typed projection the gate reads; the exact texts live here.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional

from ouroboros.tools.plan_review_runtime import slot_is_session, slot_retrieves
from ouroboros.utils import utc_now_iso
from ouroboros.reviewer_slot_config import review_cost_facts

if TYPE_CHECKING:  # the ask's own types; the adapter reads their fields and imports nothing from the tool at runtime
    from ouroboros.tools.plan_ask import Participant, PreparedAsk

log = logging.getLogger(__name__)

INTENTION_SURFACE = "intention"


def _ledger_seat(participant: Participant, facts: Mapping[str, Any]) -> Dict[str, Any]:
    slot = participant.slot
    route = getattr(slot, "route", "api_chat")
    requested = {"route": str(getattr(route, "value", route) or ""), "model": str(slot.model), "effort": str(slot.effort or ""),
                 "profile": str(getattr(slot, "session_profile", "") or ""),
                 "delivery": "retrieving" if slot_retrieves(slot) else "packet",
                 "session_target": str(getattr(slot, "session_target", "") or ""),
                 "processing_preference": str(getattr(slot, "processing_preference", "") or ""),
                 "subagent_id": str(getattr(slot, "subagent_id", "") or "")}
    status = {"answered": "responded", "unparseable": "responded", "failed": "error", "pending": "pending",
              "not_dispatched": "not_dispatched"}.get(str(facts.get("status") or "pending"), "pending")
    effective = {**{k: requested[k] for k in ("route", "model", "effort", "profile", "delivery")}, "verdict_method": "", "source": "requested"}
    if facts.get("applied_profile"):
        effective["profile_id"] = str(facts["applied_profile"])
    return {
        "seat_id": participant.seat_id, "subagent_id": requested["subagent_id"], "parts": [], "additional": not participant.owner_marked,
        "host_added": participant.host_added,
        "requested": requested, "effective": effective,
        "observed_model": _observed_model(slot, facts, status),
        "status": status, "operation_state": str(facts.get("operation_state") or ("settled" if status == "responded" else status)),
        "answers": {}, "parts_answered": [], "coverage": "not_asked", "capability_delta": list(facts.get("capability_delta") or []),
        "receipts": _seat_receipts(facts), "usd": facts.get("cost"),
        **review_cost_facts(facts.get("cost"), route=requested["route"], model=requested["model"]),
        "critical_count": 0, "raw_text": "", "brief_sha": "", "source_refs": [],
        "intention": {"objections": [o.get("id") for o in (facts.get("reply") or {}).get("objections") or []],
                      "withdrawn": [w.get("id") for w in (facts.get("reply") or {}).get("withdrawn") or []],
                      "parse_error": str(facts.get("parse_error") or ""), "error": str(facts.get("error") or "")},
    }


def _observed_model(slot: Any, facts: Mapping[str, Any], status: str) -> str:
    """What the seat ran as, from the substrate's own telemetry: an api row's sent model IS its
    applied one; a session discloses only the model its engine resolved — never the requested
    one dressed up as observed (``UNKNOWN`` until the receipt names it)."""
    from ouroboros.review_ledger import UNKNOWN

    if status not in ("responded", "error"):
        return UNKNOWN
    if slot_is_session(slot):
        return str(facts.get("resolved_model") or "") or UNKNOWN
    return str(facts.get("model") or "") or UNKNOWN


def _seat_receipts(facts: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: facts[k] for k in ("profile_continuity_receipt", "auth_route_receipt", "room_read_coverage", "host_file_read_attestation")
            if facts.get(k)}


def _intention_verdict(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """No verdict question is asked of an intention: never PASS, never FAIL."""
    from ouroboros.review_ledger import VERDICT_NOT_PERFORMED

    answered = sum(1 for r in rows if r.get("status") == "responded")
    return {"aggregate": VERDICT_NOT_PERFORMED, "reason": "intention_no_verdict_question",
            "quorum": {"required": 0, "responded": answered, "assigned": sum(1 for r in rows if not r.get("additional"))},
            "per_row": {str(r.get("seat_id")): VERDICT_NOT_PERFORMED for r in rows}, "per_question": {}}


def write_intention_record(ctx: Any, prepared: PreparedAsk, facts_by_seat: Mapping[str, Mapping[str, Any]], *,
                           record_id: str, state_root: pathlib.Path, task_id: str,
                           revision: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """One review-ledger record per ask (``surface=intention``): seats, routes, the retained
    delivery of each seat, costs and custody. Returns ``{seat_id: delivery_ref}``. A reading of
    the request before any plan (``revision=None``) records ``plan_revision: None``."""
    from ouroboros import review_ledger as ledger
    from ouroboros.config import get_review_enforcement, get_runtime_mode
    from ouroboros.plan_record import ask_subject
    from ouroboros.tools.review_helpers import review_enforcement_blocks

    revision = revision or {}
    rows, delivery_refs = [], {}
    for participant in prepared.participants:
        sid = participant.seat_id
        seat = _ledger_seat(participant, facts_by_seat.get(sid) or {"status": "pending"})
        if sid in prepared.delivery["slot_messages"] or sid in prepared.delivery["slot_session_tasks"]:
            text = (json.dumps(prepared.delivery["slot_messages"][sid], ensure_ascii=False, default=str)
                    if sid in prepared.delivery["slot_messages"] else str(prepared.delivery["slot_session_tasks"][sid]))
            ref = ledger.retain_text_source(state_root, task_id, record_id=record_id, seat_id=sid, role="prompt",
                                            part=INTENTION_SURFACE, text=text)
            seat["source_refs"].append(ref)
            delivery_refs[sid] = ref
        rows.append(seat)
    pending = ledger.counted_pending(rows)
    panel = {**ledger.panel_facts(rows), **prepared.panel, "not_reached": ledger.panel_not_reached(rows)}
    record = ledger.ReviewLedgerRecord(
        record_id=record_id, state=ledger.STATE_PENDING if pending else ledger.STATE_SETTLED, ts=utc_now_iso(), task_id=task_id,
        root_task_id=str((getattr(ctx, "task_metadata", {}) or {}).get("root_task_id") or task_id),
        review_wave_id=prepared.retry_key, surface=INTENTION_SURFACE,
        subject={"root_kind": "system_repo" if pathlib.Path(prepared.active_root) == pathlib.Path(prepared.system_root) else "active_workspace",
                 "root": str(prepared.active_root), "kind": INTENTION_SURFACE, "base": "", "head": "", "tree_sha": "", "diff_sha": "",
                 "checkout": "", "candidate_branch": "", "governance_root": str(prepared.system_root),
                 "plan_revision": revision.get("n"), "plan_ask": prepared.ask_id, "body": prepared.body},
        brief={"goal": str(revision.get("goal") or ""), "scope": ask_subject(revision.get("n"), plan=True), "parts": [],
               "author_questions": [prepared.message] if prepared.message else [],
               "checklist": {"layer": "body" if prepared.body else "core", "body_fact": "true" if prepared.body else "false",
                             "how": "plan_is_body", "checklist_hash": "", "rules_source": {"path": "docs/CHECKLISTS.md", "sha": ""}},
               "plan_ask": prepared.ask_id, "plan_revision": revision.get("n")},
        enforcement=str(get_review_enforcement() or ""), mode=str(get_runtime_mode() or ""),
        enforcement_blocks=review_enforcement_blocks(prepared.enforcement), panel=panel, rows=rows,
        verdict=_intention_verdict(rows), tests={"policy": "NOT_RUN", "result": ledger.UNKNOWN},
        cost={"usd": round(sum(float(r["usd"]) for r in rows if isinstance(r.get("usd"), (int, float))), 6),
              "unknown": any(not isinstance(r.get("usd"), (int, float)) for r in rows)},
        fingerprints={"review_contract": "", "binding": str(revision.get("content_sha256") or ""), "reuse_key": "",
                      "retry_key": prepared.retry_key})
    try:
        ledger.write_record(state_root, record)
    except Exception:
        log.warning("intention ledger record %s could not be written", record_id, exc_info=True)
    return delivery_refs


def settle_ledger_seat(state_root: pathlib.Path, task_id: str, record_id: str, seat_id: str, facts: Mapping[str, Any],
                       raw_text: str = "") -> None:
    """Revise one seat of the ask's ledger record with its settlement; retain the raw reply."""
    from ouroboros import review_ledger as ledger

    reply_ref = (ledger.retain_text_source(state_root, task_id, record_id=record_id, seat_id=seat_id, role="response",
                                           part=INTENTION_SURFACE, text=raw_text) if raw_text else None)

    def _mutate(payload: Dict[str, Any]) -> Dict[str, Any]:
        rows = [dict(r) for r in payload.get("rows") or []]
        for seat in rows:
            if str(seat.get("seat_id") or "") != seat_id:
                continue
            status = {"answered": "responded", "unparseable": "responded", "failed": "error", "pending": "pending",
                      "not_dispatched": "not_dispatched"}.get(str(facts.get("status") or "pending"), "pending")
            seat.update(status=status, operation_state=str(facts.get("operation_state") or seat.get("operation_state") or ""),
                        usd=facts.get("cost", seat.get("usd")))
            seat.update(review_cost_facts(seat.get("usd"), route=seat["requested"]["route"], model=seat["requested"]["model"]))
            session = str(seat.get("requested", {}).get("route") or "") == "agent_session"
            observed = (str(facts.get("resolved_model") or "") if session else str(facts.get("model") or "")) if status in ("responded", "error") else ""
            if observed:
                seat["observed_model"] = observed
            if facts.get("capability_delta"):
                seat["capability_delta"] = list(facts["capability_delta"])
            receipts = _seat_receipts(facts)
            if receipts:
                seat["receipts"] = {**dict(seat.get("receipts") or {}), **receipts}
            if facts.get("applied_profile"):
                seat["effective"] = {**dict(seat.get("effective") or {}), "profile_id": str(facts["applied_profile"])}
            seat["intention"] = {"objections": [o.get("id") for o in (facts.get("reply") or {}).get("objections") or []],
                                 "withdrawn": [w.get("id") for w in (facts.get("reply") or {}).get("withdrawn") or []],
                                 "parse_error": str(facts.get("parse_error") or ""), "error": str(facts.get("error") or "")}
            if reply_ref is not None:
                seat["source_refs"] = [*(seat.get("source_refs") or []), reply_ref]
            for role in ("prompt_ref", "response_ref"):
                if isinstance(facts.get(role), dict) and facts[role]:
                    seat["source_refs"] = [*(seat.get("source_refs") or []),
                                           {"role": f"observability_{role[:-4]}", "part": INTENTION_SURFACE, "seat_id": seat_id,
                                            "status": "observability", "ref": facts[role]}]
        payload["rows"] = rows
        payload["verdict"] = {**dict(payload.get("verdict") or {}), **_intention_verdict(rows)}
        payload["state"] = ledger.STATE_PENDING if ledger.counted_pending(rows) else ledger.STATE_SETTLED
        payload["panel"]["not_reached"] = ledger.panel_not_reached(rows)
        known = [float(r["usd"]) for r in rows if isinstance(r.get("usd"), (int, float))]
        payload["cost"] = {"usd": round(sum(known), 6), "unknown": len(known) < len(rows)}
        return payload

    try:
        ledger.revise_record(state_root, record_id, _mutate)
    except Exception:
        log.warning("intention ledger record %s could not be revised for %s", record_id, seat_id, exc_info=True)
    return reply_ref
