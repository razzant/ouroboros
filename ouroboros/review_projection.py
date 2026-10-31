"""Panel identity and the compact, redacted projection of a review run.

Owns the outward-facing view of a completed panel: transport-failure
classification, redaction of model-authored reason text, the per-actor and
per-panel projections that task results and the UI consume (review-ledger
records included), the enforcement impact label, the panel-identity hash over
the actor rows, and the tolerant readers of task-acceptance records written
before the result review replaced that machine. Extracted from
ouroboros/review_substrate.py (v7 D06 split, re-cut on the v7next tip);
review_substrate.py re-exports every name.
"""

from __future__ import annotations

import hashlib
import json
import copy
import logging
import math
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, TYPE_CHECKING

from ouroboros.review_records import review_slot_awaiting, review_slot_unresolved

# A slot released at the dispatch barrier has no transport or parse event:
# its projection is a gap, never a transport failure or a malformed answer.
AWAITING_PROJECTION = "awaiting"
_AWAITING_REASON = "No answer recorded: the host returned at the dispatch barrier before this reviewer answered."
# The plan review's outcome CLASS at delivery, stamped by owner_hurry.plan_gate_decision and
# carried on outcome_axes.execution.plan_review; the earlier engine's classes (unanswered,
# none_answered, answered_open) survive only as the cause sentences of finished tasks.
PLAN_REVIEW_OBJECTIONS_OPEN = "objections_open"  # objections stood open at delivery

if TYPE_CHECKING:  # annotation-only names; lazy under future annotations, never imported at runtime
    from ouroboros.review_records import ReviewActorRecord, ReviewRequest


def _sub():
    """The parent review-substrate module, read at call time.

    The substrate members stay monkeypatch-addressable at their historical
    ``ouroboros.review_substrate`` bindings (tests rebind them there), so this
    leaf resolves every such cross-reference through the module at each call
    instead of freezing whatever object a from-import saw at import time.
    """
    from ouroboros import review_substrate

    return review_substrate


def _transport_error_status(error: Any, *, failure_phase: str = "") -> str:
    """Classify transport failures without depending on a non-empty message."""
    if getattr(error, "review_run_state", "") in {"cancelled", "interrupted"}:
        return "interrupted"
    if failure_phase == "authority":
        return "authority_error"
    error_type = type(error).__name__ if isinstance(error, BaseException) else ""
    error_text = str(error or "")
    if (
        isinstance(error, TimeoutError)
        or "timeout" in error_type.casefold()
        or "timeout" in error_text.casefold()
        or "timed out" in error_text.casefold()
    ):
        return "timeout"
    return "provider_transport_error"


def _public_review_reason(value: Any) -> str:
    """Redact model-controlled reason text before publishing it in full.

    v6.70.0 honesty change (owner decision): reviewer rationale is a cognitive
    artifact (BIBLE P1 — multi-model review outputs must not fall back to
    generic transport truncation). The former 500/800-char caps destroyed the
    only owner-reachable copy of the reasoning (task_results carried the same
    truncated projection and the full observability blobs were unreferenced),
    so the projection now publishes the COMPLETE redacted text; secrets are
    still masked by redact_projection."""
    text = str(value or "")
    if not text:
        return ""
    return str(_sub().redact_projection(text).value)


def _actor_reason(row: Dict[str, Any], reason: str) -> str:
    """Keep the reported failure beside the reason the existing card displays."""
    cause = str(row.get("reported_cause") or "")
    if cause and cause not in reason:
        reason = f"{reason}\nReported cause: {cause}".strip()
    return _public_review_reason(reason)


def awaiting_panel_reason(slot_ids: List[str], configured: int, aggregate: str) -> str:
    """The one host sentence for the slots of a panel released at the dispatch barrier."""
    return (f"awaiting {len(slot_ids)} of {configured} reviewer slot(s): {', '.join(slot_ids)}"
            + (" — no verdict" if aggregate == "DEGRADED" else ""))


def _review_actor_projection(actor: Any, surface: str) -> Dict[str, Any]:
    row = actor if isinstance(actor, dict) else asdict(actor)
    parsed = row.get("parsed") if isinstance(row.get("parsed"), (dict, list)) else None
    # The one decision point: a slot is awaited only while it carries no answer. A row that
    # carries one is judged by its answer, whatever its custody state says.
    outcome = row.get("outcome") or {}
    awaiting = (review_slot_awaiting(row) or outcome.get("status") == "pending") and parsed is None and not str(row.get("raw_text") or "").strip()
    usage = row.get("usage") if isinstance(row.get("usage"), dict) else {}
    explicit_parse = str(row.get("parse_status") or "")
    if explicit_parse == AWAITING_PROJECTION:
        explicit_parse = ""  # derived state: true only while the row is awaiting, recomputed below
    semantic = str(row.get("semantic_verdict") or "").upper()
    if not semantic and isinstance(parsed, dict):
        semantic = str(parsed.get("verdict") or parsed.get("status") or "").upper()
    if not semantic:
        semantic = str(row.get("signal") or "").upper()
    valid = (
        explicit_parse != "malformed"
        and parsed is not None
        and semantic in {"PASS", "FAIL", "DEGRADED"}
    )
    error = str(row.get("error") or "")
    transport = str(row.get("transport_status") or "")
    if awaiting:
        transport = AWAITING_PROJECTION  # the typed state outranks a word stored by an earlier projection
    elif not transport or transport == AWAITING_PROJECTION:
        not_dispatched = (
            str(row.get("status") or "") == "not_dispatched"
            or str(row.get("operation_state") or "") == "not_dispatched"
        )
        transport = (
            "not_dispatched" if not_dispatched
            else ("success" if str(row.get("status") or "") in {"ok", "empty"}
                  else _transport_error_status(error, failure_phase=str(usage.get("review_failure_phase") or "")))
        )
    criteria = parsed.get("criteria_used") if isinstance(parsed, dict) else []
    criteria = criteria if isinstance(criteria, list) else []
    if isinstance(parsed, dict):
        parsed_findings = parsed.get("findings")
    elif isinstance(parsed, list):
        parsed_findings = parsed
    else:
        parsed_findings = []
    parsed_findings = (
        [item for item in parsed_findings if isinstance(item, dict)]
        if isinstance(parsed_findings, list)
        else []
    )
    reason = str(row.get("reason") or "")
    if not reason and isinstance(parsed, dict):
        reason = str(parsed.get("summary") or parsed.get("reason") or "")
    if not reason and isinstance(parsed, list):
        for item in parsed_findings:
            reason = str(
                item.get("summary")
                or item.get("reason")
                or item.get("evidence")
                or item.get("item")
                or item.get("recommendation")
                or ""
            )
            if reason:
                break
    reason = reason or error or ("Reviewer response was malformed or absent." if not valid and explicit_parse != "valid" else "")
    if awaiting:
        reason = (reason or _AWAITING_REASON) if outcome else _AWAITING_REASON
    model = str(usage.get("resolved_model", row.get("model")) or "")
    provider = str(usage.get("provider") or row.get("provider") or "")
    if not provider:
        provider = _sub().provider_for_model(model) if model else "unknown"
    outcome_tier = (
        str(parsed.get("outcome_tier") or "").strip().lower()
        if isinstance(parsed, dict)
        else ""
    )
    if outcome_tier not in {
        _sub().OUTCOME_TIER_SOLVED, _sub().OUTCOME_TIER_BEST_EFFORT, _sub().OUTCOME_TIER_BLOCKED,
    }:
        outcome_tier = ""
    dialogue_vote = (
        str(parsed.get("dialogue_status") or "").strip().lower()
        if isinstance(parsed, dict)
        else ""
    )
    if dialogue_vote not in _sub().DIALOGUE_STATUS_VALUES:
        dialogue_vote = ""
    projection = {
        "slot_id": str(row.get("slot_id") or ""), "model": model, "provider": provider,
        "actor_role": str(row.get("actor_role") or f"{surface} reviewer"),
        "transport_status": transport,
        "parse_status": AWAITING_PROJECTION if awaiting else (explicit_parse or ("valid" if valid else "malformed")),
        "semantic_verdict": semantic if valid else "",
        "outcome_tier": outcome_tier if valid else "",
        "dialogue_status": dialogue_vote if valid else "",
        "coverage": {
            "criteria_total": len(criteria),
            "findings": len(parsed_findings),
        },
        "quorum_contribution": bool(row.get("quorum_contribution")),
        "reason": _actor_reason(row, reason),
        "enforcement_impact": str(row.get("enforcement_impact") or "abstains"),
        # Preserve the physical identity when the logical actor times out.
        "operation_id": str(row.get("operation_id") or ""),
        "operation_state": str(row.get("operation_state") or "settled"),
        "late_result_pending": bool(row.get("late_result_pending")),
        "executions": _sub().review_executions_from_actor_usage([row]),
        # Flat, redacted pointer to the private full response artifact.
        "response_ref": _response_ref_projection(row.get("response_ref")),
    }
    if outcome:
        projection.update(transport_status=outcome["transport"], parse_status=outcome["parse"],
                          semantic_verdict=outcome["verdict"])
    # Since when this row has been waiting, published only where the host wrote
    # a real instant on a row the two wait predicates still call unanswered. An
    # absent key is a hole, never a back-filled or inferred moment.
    if awaiting or review_slot_unresolved(row):
        since = str(row.get("awaiting_since") or "").strip()
        try:
            datetime.fromisoformat(since)
        except ValueError:
            pass
        else:
            projection["awaiting_since"] = since
    # Structured rows ride only where a parsed response exists: an absent
    # `findings` key is a hole, never the claim "zero findings reported".
    if parsed is not None:
        projection.update(_sub().disclosed_list_projection(
            parsed_findings,
            key="findings",
            limit=_sub().MAX_PROJECTED_ACTOR_FINDINGS,
            item=_sub().projected_finding_row,
        ))
    return projection


def _response_ref_projection(ref: Any) -> Dict[str, str]:
    if not isinstance(ref, dict):
        return {}
    out: Dict[str, str] = {}
    if ref.get("call_id"):
        out["call_id"] = str(ref["call_id"])
    projection_ref = ref.get("redacted_projection_ref")
    if isinstance(projection_ref, dict) and projection_ref.get("sha256"):
        out["sha256"] = str(projection_ref["sha256"])
    elif ref.get("sha256"):
        out["sha256"] = str(ref["sha256"])
    manifest_ref = ref.get("manifest_ref")
    if isinstance(manifest_ref, dict) and manifest_ref.get("sha256"):
        out["manifest_sha256"] = str(manifest_ref["sha256"])
    return out


def _review_enforcement_impact(run: Dict[str, Any]) -> str:
    if str(run.get("enforcement_impact") or ""):
        return str(run["enforcement_impact"])
    request = run.get("request") if isinstance(run.get("request"), dict) else {}
    hardness = str((request.get("policy") or {}).get("hardness") or "")
    signal = str(run.get("aggregate_signal") or "").upper()
    if str(run.get("authority") or "") == "agent_advisory" or hardness == _sub().HARDNESS_ADVISORY_VISIBLE:
        return "advisory"
    if signal == "PASS":
        return "allows_completion"
    return "blocks_completion" if signal == "FAIL" and hardness == _sub().HARDNESS_HARD_GATE else "degrades_completion"


def _review_panel_id(request: ReviewRequest, actors: List[ReviewActorRecord]) -> str:
    seed = {
        "surface": request.surface,
        "task_id": request.task_id,
        "actors": [
            [actor.slot_id, actor.model, actor.response_ref]
            for actor in actors
        ],
    }
    digest = hashlib.sha256(
        json.dumps(seed, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return f"panel_{digest[:16]}"


def _panel_transport(statuses: List[str]) -> str:
    """One panel's transport word over the words of its collected actors."""
    if "authority_error" in statuses and all(word in {"authority_error", "not_dispatched"} for word in statuses):
        return "authority_error"  # a local refusal plus withheld rows has no provider failure
    for word in ("success", "not_dispatched", "timeout", "authority_error"):
        if statuses and all(status == word for status in statuses):
            return word
    return "partial" if "success" in statuses else "provider_transport_error"


# The newest review-ledger records of one task a task projection carries; the rest
# stay in the ledger, and the projection says whether any exist.
LEDGER_RECORD_PANELS = 5
# A ledger seat's ``status`` once a readable answer arrived, as the parse word.
_LEDGER_PARSE = {"responded": "valid", "partial": "partial"}


def _ledger_seat_executions(seat: Dict[str, Any]) -> List[Dict[str, str]]:
    """What actually ran for one seat, when the reviewer row's own last execution proved it
    (``review_ledger._apply_execution``); a requested route alone proves nothing."""
    from ouroboros.review_execution_projection import normalize_review_executions

    effective = seat.get("effective") if isinstance(seat.get("effective"), dict) else {}
    if effective.get("source") != "reviewer_slot_last_execution":
        return []
    route, model = str(effective.get("route") or ""), str(effective.get("model") or "")
    harness = route.split(":", 1)[1] if route.startswith("agent_session:") else ""
    kind = "harness" if harness else ("native" if effective.get("delivery") == "retrieving" else "api")
    return normalize_review_executions([{"kind": kind, "harness_id": harness, "model": model}])


def _ledger_seat_actor(seat: Dict[str, Any], record: Dict[str, Any]) -> Dict[str, Any]:
    """One ledger seat as a card reviewer: the parts it was asked, each part's answer
    with its findings COUNTED (the bodies ride once, bounded, in ``findings``), and the
    observed model, with requested identity kept separately when execution is unknown."""
    outcome = seat.get("outcome") or {}
    status, surface = str(outcome.get("status") or seat.get("status") or ""), str(record.get("surface") or "review")
    requested = seat.get("requested") if isinstance(seat.get("requested"), dict) else {}
    observed = str(seat.get("observed_model") or "")
    model = observed if observed not in ("", "unknown") else ""
    answers = outcome.get("answers") or (seat.get("answers") if isinstance(seat.get("answers"), dict) else {})
    parts = [str(part) for part in seat.get("parts") or [] if part in answers] or list(answers)
    findings: List[Dict[str, Any]] = []
    counted: Dict[str, Dict[str, Any]] = {}
    for part in parts:
        answer = answers[part] if isinstance(answers[part], dict) else {}
        listed = [item for item in answer.get("findings") or [] if isinstance(item, dict)]
        findings += listed
        counted[part] = {"status": str(answer.get("status") or ""), "verdict": str(answer.get("verdict") or ""),
                         "findings": len(listed), "critical": int(answer.get("critical") or 0),
                         "coverage": str(answer.get("coverage") or "")}
    contributes = status in _LEDGER_PARSE and not seat.get("additional")
    ref = next((item.get("ref") for item in seat.get("source_refs") or []
                if isinstance(item, dict) and item.get("role") == "observability_response"), None)
    actor = {
        "slot_id": str(seat.get("seat_id") or ""), "model": model,
        "requested_model": str(requested.get("model") or ""), "observed_model": model,
        "provider": _sub().provider_for_model(model) if model else "unknown",
        "actor_role": f"{surface} {'additional ' if seat.get('additional') else ''}reviewer",
        "transport_status": str(outcome.get("transport") or seat.get("transport_status") or ("success" if status in _LEDGER_PARSE else status or "unknown")),
        "parse_status": outcome.get("parse") or _LEDGER_PARSE.get(status, "none"),
        "semantic_verdict": str(outcome.get("verdict") or "") if outcome else str(((record.get("verdict") or {}).get("per_row") or {}).get(seat.get("seat_id")) or ""),
        "quorum_contribution": contributes,
        "enforcement_impact": str(record.get("enforcement") or "unknown") if contributes else "abstains",
        "operation_state": str(seat.get("operation_state") or ""),
        "parts": parts, "answers": counted, "response_ref": _response_ref_projection(ref),
        "reason": _actor_reason(seat, str(seat.get("raw_text") or seat.get("failure_code") or "")
                                if status not in _LEDGER_PARSE else ""),
        "executions": _ledger_seat_executions(seat),
    }
    actor.update({key: _sub().redact_projection(seat[key]).value for key in _sub().TYPED_FAILURE_FACT_KEYS if seat.get(key) not in (None, "")})
    # The browser gets a selector, never a filesystem path. The source door
    # re-reads the owning record/seat and verifies the exact saved source.
    actor["sources"] = [
        {"role": str(source.get("role") or ""), "part": str(source.get("part") or ""),
         "status": str(source.get("status") or ""),
         "sha256": str((source.get("ref") or {}).get("sha256") or ""),
         "size": (source.get("ref") or {}).get("size"), "chars": source.get("chars")}
        for source in seat.get("source_refs") or []
        if isinstance(source, dict) and source.get("role") in {"response", "failure"}
        and isinstance(source.get("ref"), dict)
    ]
    actor.update(_sub().disclosed_list_projection(
        findings, key="findings", limit=_sub().MAX_PROJECTED_ACTOR_FINDINGS, item=_sub().projected_finding_row))
    return actor


# The review pool's facts on a record's ``panel`` block (who was ticked when the wave was
# composed, how strict review was, a widened request, the seats never reached); a record
# written before them carries neither ``pool`` nor ``strictness`` and projects as before.
_POOL_FACT_KEYS = ("pool", "strictness", "widened_from", "not_reached")


def _seat_name(item: Any) -> str:
    return item if isinstance(item, str) else str(item.get("seat_id") or "") if isinstance(item, dict) else ""


def _pool_facts(recorded: Dict[str, Any]) -> Dict[str, Any]:
    """The pool keys read defensively: a malformed value is dropped, never guessed, and a
    not-reached cause is redacted like every other reason the card prints."""
    out: Dict[str, Any] = {}
    if isinstance(recorded.get("pool"), list):
        out["pool"] = list(recorded["pool"])
    if isinstance(recorded.get("strictness"), str) and recorded["strictness"].strip():
        out["strictness"] = recorded["strictness"].strip()
    if isinstance(recorded.get("widened_from"), list):
        out["widened_from"] = [name for name in map(_seat_name, recorded["widened_from"]) if name]
    if isinstance(recorded.get("not_reached"), list):
        out["not_reached"] = [
            {"seat_id": _seat_name(item),
             "cause": _public_review_reason(item.get("cause") if isinstance(item, dict) else "")}
            for item in recorded["not_reached"] if _seat_name(item)]
    return out


def _seat_cost(seat: Dict[str, Any]) -> Dict[str, Any]:
    """One seat's cost on the card: a measured ``usd`` amount, a ``subscription`` seat or ``unknown``.
    The seat's ``cost_kind`` (``measured`` | ``subscription`` | ``unknown``) and ``cost_usd`` decide;
    a seat recorded without them shows the money it reported (``usd``)."""
    kind = str(seat.get("cost_kind") or "")
    if kind == "subscription":
        return {"kind": "subscription", "usd": None}
    value = seat.get("cost_usd") if kind or "cost_usd" in seat else seat.get("usd")
    if kind != "unknown" and isinstance(value, (int, float)) and not isinstance(value, bool) \
            and math.isfinite(value) and value >= 0:
        return {"kind": "usd", "usd": float(value)}
    return {"kind": "unknown", "usd": None}


def _wave_cost(costs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The wave total over its priced seats, and how many seats each kind of cost has."""
    priced = [cost["usd"] for cost in costs if cost["kind"] == "usd"]
    return {"usd": round(sum(priced), 6) if priced else None,
            "seats": {kind: sum(1 for cost in costs if cost["kind"] == kind) for kind in ("usd", "subscription", "unknown")}}


def _doctrine_panel(record: Dict[str, Any]) -> Dict[str, Any]:
    """Pass the author-check facts through. A record without them stays unchanged."""
    decision = record.get("author_decision")
    if not isinstance(decision, dict):
        return {}
    from ouroboros.review_doctrine import EVIDENCE_KIND, FACT_KIND, LANDING_KEY, public_card_decision

    if not any(isinstance(decision.get(key), dict) for key in (FACT_KIND, EVIDENCE_KIND, LANDING_KEY)):
        return {}
    public = public_card_decision(decision)
    return {"author_decision": public} if public else {}


_LANDINGS_SHOWN = 16  # a panel names the newest landed commits; its record (``source_ref``) keeps every one


def _landed_panel(record: Dict[str, Any]) -> Dict[str, Any]:
    """The commits that landed under this record, each against its reviewed tree (``commit_landings``):
    the newest ones, with the count of older ones only the record holds (``commit_landings_omitted``)."""
    from ouroboros.review_doctrine import public_commit_landings

    landed = public_commit_landings(record)
    shown = landed[-_LANDINGS_SHOWN:]
    omitted = {"commit_landings_omitted": len(landed) - len(shown)} if len(landed) > len(shown) else {}
    return {"commit_landings": shown, **omitted} if landed else {}


def ledger_record_panel(record: Dict[str, Any]) -> Dict[str, Any]:
    """One review-ledger record (``review_ledger.build_wave_record``) as a card panel:
    the record's ``panel`` facts (who sat and why) and every seat's answer per brief
    part. The aggregate is the record's own; the record file stays the full source.
    A record carrying the pool's facts also gives each actor ``additional`` (an
    adviser, never counted) and ``cost``, and the panel its wave ``cost``."""
    from ouroboros.review_ledger import LEDGER_SUBDIR

    verdict = record.get("verdict") if isinstance(record.get("verdict"), dict) else {}
    quorum = verdict.get("quorum") if isinstance(verdict.get("quorum"), dict) else {}
    seats = [seat for seat in record.get("rows") or [] if isinstance(seat, dict)]
    actors = [_ledger_seat_actor(seat, record) for seat in seats]
    parsed = {actor["parse_status"] for actor in actors}
    facts = dict(record.get("panel") or {}) if isinstance(record.get("panel"), dict) else {}
    if facts.get("reason"):
        facts["reason"] = _public_review_reason(facts["reason"])
    recorded = {key: facts.pop(key) for key in _POOL_FACT_KEYS if key in facts}
    facts.update(_pool_facts(recorded))
    # A preflight seats one named row, ticked or not, and owes no reason: never a pool panel.
    pooled = str(record.get("surface") or "") != "preflight" and ("pool" in facts or "strictness" in facts)
    costs = [_seat_cost(seat) for seat in seats] if pooled else []
    for actor, seat, cost in zip(actors, seats, costs):
        actor.update(additional=bool(seat.get("additional")), cost=cost)
    aggregate, record_id = str(verdict.get("aggregate") or "UNKNOWN").upper(), str(record.get("record_id") or "")
    return {
        "panel_id": record_id, "record_id": record_id, "ts": str(record.get("ts") or ""),
        "surface": str(record.get("surface") or "review"), "authority": "review_ledger",
        "record_state": str(record.get("state") or ""),
        "revision": record.get("revision"), "review_wave_id": str(record.get("review_wave_id") or ""),
        "subject": _sub().redact_projection(record.get("subject") or {}).value,
        "goal": _public_review_reason((record.get("brief") or {}).get("goal")),
        "aggregate_signal": aggregate,
        "transport_status": _panel_transport([actor["transport_status"] for actor in actors]),
        "parse_status": "valid" if parsed == {"valid"} else ("none" if parsed <= {"none"} else "partial"),
        "coverage": dict(verdict.get("per_question") or {}),
        "quorum": {"required": int(quorum.get("required") or 0), "contributed": int(quorum.get("responded") or 0),
                   "configured": int(quorum.get("assigned") or 0)},
        "reason": "" if aggregate == "PASS" else str(verdict.get("reason") or ""),
        "enforcement_impact": str(record.get("enforcement") or "unknown"),
        "panel_facts": facts, "actors": actors, "superseded": False,
        "source_ref": {"kind": "review_ledger_record", "path": f"state/{LEDGER_SUBDIR}/{record_id}.json"},
        **({"cost": _wave_cost(costs)} if pooled else {}),
        **_doctrine_panel(record),
        **_landed_panel(record),
    }


def task_ledger_records(task: Dict[str, Any], drive_root: Any) -> tuple:
    """This task's own newest review-ledger records, oldest first, each read whole (an
    index row lacks the panel and the answers a card prints), and whether older ones
    exist: ``0``, ``"1+"`` or ``"unknown"`` (an archived segment may hold some), as
    ``## Review records`` says it. The ledger lives on the canonical data root."""
    task_id, root = str(task.get("id") or ""), task.get("budget_drive_root") or drive_root
    if not task_id or not root:
        return [], 0
    try:
        from ouroboros import review_ledger as ledger

        rows, next_offset = ledger.task_record_page(root, task_id, limit=LEDGER_RECORD_PANELS, hot_only=True)
        records = _read_record_rows(root, rows)
        more = "1+" if next_offset is not None else ("unknown" if ledger.archived_segments_exist(root) else 0)
    except Exception:
        logging.getLogger(__name__).debug("Review ledger records unavailable for task %s", task_id, exc_info=True)
        return [review_record_gap("", "index_unavailable")], "unknown"
    return [record for record in reversed(records) if record], more


def _read_record_rows(root: Any, rows: list) -> list:
    from ouroboros import review_ledger as ledger

    records = []
    for row in rows:
        record_id = str(row.get("record_id") or "")
        try:
            record = ledger.load_record(root, record_id)
            records.append(record if record is not None else review_record_gap(record_id, "missing"))
        except Exception:
            records.append(review_record_gap(record_id, "unreadable"))
    return records


def task_review_history(root: Any, task_id: str, *, offset: int, limit: int) -> dict:
    """Explicit archived history read; local gaps never hide the neighboring records."""
    from ouroboros import review_ledger as ledger

    rows, next_offset = ledger.task_record_page(root, task_id, offset=offset, limit=limit)
    projection = compact_review_projection([], records=list(reversed(_read_record_rows(root, rows))))
    projection["review_records_omitted"] = "1+" if next_offset is not None else 0
    return {"task_id": task_id, "review_projection": projection, "next_offset": next_offset}


def review_record_gap(record_id: str, reason: str) -> Dict[str, Any]:
    """One missing unit of reading, separate from older records outside the view."""
    return {"record_id": record_id, "review_gap": reason}


_OWNER_SHORTENED = "… (shortened; the complete text is in the review record)"


def owner_bounded_text(text: str, limit: int, *, in_record: bool) -> str:
    """The SSOT display bound (``truncate_review_artifact``: same limit, same
    anti-waste floor) with its model-facing marker said in words for a person.
    The marker is replaced only where the SSOT itself appended it — no reviewer
    text is pattern-matched — and the words point at the review record that
    holds the complete text only when ``in_record`` says it holds it."""
    from ouroboros.utils import truncate_review_artifact

    full = str(text or "")
    bounded = truncate_review_artifact(full, limit=limit)
    marker = f"\n⚠️ OMISSION NOTE: truncated at {limit} chars; original length {len(full)}"
    if bounded == full or not bounded.endswith(marker):
        return bounded
    return bounded[: -len(marker)].rstrip() + (_OWNER_SHORTENED if in_record else "… (shortened)")


def compact_review_projection(review_runs: Any, records: Any = ()) -> Dict[str, Any]:
    """Project existing audit runs without copying raw prompts or responses, then the
    given review-ledger ``records`` (:func:`ledger_record_panel`), in their order."""
    panels: List[Dict[str, Any]] = []
    for index, raw_run in enumerate(review_runs or []):
        if not isinstance(raw_run, dict) or raw_run.get("surface") == "result":
            continue  # a result run binds a ledger record, which projects as its own panel below
        request = raw_run.get("request") if isinstance(raw_run.get("request"), dict) else {}
        surface = str(request.get("surface") or "review")
        actors = [_review_actor_projection(actor, surface) for actor in (raw_run.get("actors") or []) if isinstance(actor, dict)]
        policy = request.get("policy") if isinstance(request.get("policy"), dict) else {}
        min_successful = max(1, int(policy.get("min_successful_slots") or 1))
        contributing = sum(1 for actor in actors if actor["quorum_contribution"])
        awaited = [actor["slot_id"] for actor in actors if actor["transport_status"] == AWAITING_PROJECTION]
        collected = [actor for actor in actors if actor["transport_status"] != AWAITING_PROJECTION]
        transport = _panel_transport([actor["transport_status"] for actor in collected])
        parse = "valid" if collected and all(a["parse_status"] == "valid" for a in collected) else "malformed"
        reasons = raw_run.get("degraded_reasons") if isinstance(raw_run.get("degraded_reasons"), list) else []
        reasons = [str(item) for item in reasons]
        aggregate = str(raw_run.get("aggregate_signal") or "UNKNOWN").upper()
        if awaited:
            # The panel words follow the typed rows, so a run-level word recorded for an
            # awaited slot cannot outlive it. Awaited slots speak for the panel only when
            # every collected slot is clean; a real failure beside a wait keeps its own word.
            transport = AWAITING_PROJECTION if (not collected or transport == "success") else transport
            parse = AWAITING_PROJECTION if (not collected or parse == "valid") else parse
            note = awaiting_panel_reason(awaited, len(actors), aggregate)
            reason = "; ".join([note] + [
                item for item in reasons if item != note and item.split(":", 1)[0] not in awaited
            ])
        else:
            transport = str(raw_run.get("transport_status") or transport)
            parse = str(raw_run.get("parse_status") or parse)
            # v6.74.0 (A6): the fallback reason is the structured panel_reason
            # reducer — it names the real blocker (tier + finding / degraded
            # causes) instead of an opaque aggregate label. An explicitly
            # recorded reason still wins.
            reason = str(raw_run.get("reason") or "; ".join(reasons) or _sub().panel_reason(raw_run))
        panel: Dict[str, Any] = {
            "panel_id": str(raw_run.get("panel_id") or f"panel_{index + 1}"),
            "surface": surface,
            "authority": str(raw_run.get("authority") or "unspecified"),
            "aggregate_signal": aggregate,
            "transport_status": transport,
            "parse_status": parse,
            "coverage": {
                "actors_configured": len(actors),
                "transport_success": sum(1 for actor in actors if actor["transport_status"] == "success"),
                "parse_valid": sum(1 for actor in actors if actor["parse_status"] == "valid"),
                "quorum_contributing": contributing,
            },
            "quorum": {"required": min_successful, "contributed": contributing, "configured": len(actors)},
            "reason": _public_review_reason(reason),
            "enforcement_impact": _review_enforcement_impact(raw_run),
            "actors": actors,
            "superseded": bool(raw_run.get("superseded_by_revision")),
        }
        if raw_run.get("single_reviewer_no_diversity"):
            panel["single_reviewer_no_diversity"] = True
        if isinstance(raw_run.get("dialogue"), dict):
            panel["dialogue"] = raw_run.get("dialogue")
        if surface == "task_acceptance":
            # A task-acceptance run written before the result review replaced that machine.
            panel["applied_source_status"] = str(raw_run.get("applied_source_status") or "unavailable")
            for key in ("task_attempt", "panel_index", "publication_revision", "applied_source_ref"):
                if key in raw_run:
                    panel[key] = copy.deepcopy(raw_run[key])
            # A panel that settled after its task ended carries the host's own
            # sentence about it; the card and the owner's row print those bytes
            # verbatim, so the projection copies them under the same disclosed
            # bound every other owner-facing review artifact uses, a cut said in
            # words. The raw run keeps every byte, so a STORED applied record is
            # named as the home of the rest (many reviewers can pass the bound).
            if isinstance(raw_run.get("late_settlement"), dict):
                late = copy.deepcopy(raw_run["late_settlement"])
                late["note"] = owner_bounded_text(str(late.get("note") or ""), 2000,
                                                  in_record=panel["applied_source_status"] == "available")
                panel["late_settlement"] = late
        for key in (
            "candidate_hash", "evidence_revision", "fence_hash", "binding_hash",
        ):
            if raw_run.get(key) not in (None, ""):
                panel[key] = str(raw_run.get(key))
        panels.append(panel)
    gaps = []
    for record in records or ():
        if not isinstance(record, dict):
            continue
        if record.get("review_gap"):
            gaps.append(record)
            continue
        try:
            panels.append(ledger_record_panel(record))
        except Exception:
            # Valid JSON can still contain a malformed quorum/seat. It must
            # not erase panels already collected from this or another domain.
            gaps.append(review_record_gap(str(record.get("record_id") or ""), "projection_unreadable"))
    return {"panels": panels, **({"gaps": gaps} if gaps else {})}


def _actor_pending(actor: Dict[str, Any]) -> bool:
    return (actor.get("transport_status") == AWAITING_PROJECTION
            or actor.get("operation_state") in {"pending_dispatch", "in_flight"})


# The origin an old decision names for the host's own local evidence-assembly failure.
_LOCAL_PREPARATION_ORIGIN = "local_acceptance_preparation"


def acceptance_decision_projection(acceptance_decision: Dict[str, Any], subject_hash: str = "") -> Dict[str, Any]:
    """The tolerant reader of a task-acceptance decision written before the result
    review replaced that machine; nothing writes a new one."""
    out = {
        "status": str(acceptance_decision.get("status") or ""),
        # v6.78.0: the typed reason carries the distinction the collapsed status no
        # longer spells out (no-quorum vs FAIL-without-capsule vs obligations open
        # vs capsule spent vs deadline skip). Historical records have no reason.
        "reason": str(acceptance_decision.get("reason") or ""),
        "source": str(acceptance_decision.get("source") or ""),
        "rationale": str(acceptance_decision.get("rationale") or "")[:500],
        "agent_disposition": str(acceptance_decision.get("agent_disposition") or ""),
        "agent_rationale": str(acceptance_decision.get("agent_rationale") or "")[:500],
    }
    if acceptance_decision.get("enforcement"):
        out["enforcement"] = str(acceptance_decision["enforcement"])
    if acceptance_decision.get("origin"):
        out["origin"] = str(acceptance_decision["origin"])
    if acceptance_decision.get("author_action"):
        out["author_action"] = acceptance_decision["author_action"]
    if isinstance(acceptance_decision.get("review_capacity"), dict):
        out["review_capacity"] = dict(acceptance_decision["review_capacity"])
    if acceptance_decision.get("reason") in {"author_finish", "author_stop"} or acceptance_decision.get("author_action") == "stop":
        from ouroboros.review_records import validate_author_disposition

        if acceptance_decision.get("origin") == _LOCAL_PREPARATION_ORIGIN:
            incident = acceptance_decision.get("acceptance_incident") or {}
            subject_hash = f"{incident.get('incident_id')}:attempt-{incident.get('attempts')}"
        record = validate_author_disposition(acceptance_decision.get("author_disposition"), subject_hash=subject_hash)
        if isinstance(record, dict):
            out["author_disposition"] = dict(record)
        else:
            out["author_disposition"] = str(record or "")
        out["author_rationale"] = str(acceptance_decision.get("author_rationale") or "")[:500]
        out["reviewer_signal"] = str(acceptance_decision.get("reviewer_signal") or "")
    # v6.54.4: dissent + obligations transparency (blocking review policy).
    if acceptance_decision.get("dissent_noted"):
        out["dissent_noted"] = True
    if acceptance_decision.get("open_obligations"):
        out["open_obligations"] = [str(x) for x in acceptance_decision.get("open_obligations") or []][:10]
    # The host's own local acceptance-preparation failure: stable incident
    # identity and the REAL host attempt count, so the card can state the fact
    # without a reviewer ever having run (#1224).
    if isinstance(acceptance_decision.get("acceptance_incident"), dict) and acceptance_decision["acceptance_incident"]:
        from ouroboros.observability import redact_projection

        out["acceptance_incident"] = redact_projection(
            dict(acceptance_decision["acceptance_incident"])).value
    return out


def _stamp_order(item: Any) -> tuple:
    """``(task_attempt, publication_revision)`` of one stamped row or snapshot;
    ``(0, 0)`` for one that carries no publication stamp."""
    if not isinstance(item, dict) or type(item.get("publication_revision")) is not int:
        return (0, 0)
    return (item.get("task_attempt") if type(item.get("task_attempt")) is int else 0, item["publication_revision"])


def _publication_order(projection: Dict[str, Any]) -> tuple:
    """The newest host publication a snapshot carries: its own stamp (every
    checkpoint carries one, a panel-less incident snapshot included) or, for an
    older snapshot without one, the newest of its panels. ``(0, 0)`` is a
    legacy snapshot with no ordering at all."""
    rows = projection.get("panels") if isinstance(projection.get("panels"), list) else []
    return max([_stamp_order(projection), *(_stamp_order(row) for row in rows)])


def merge_review_projection(previous: Any, incoming: Any) -> Any:
    """Keep newer host publication facts when a delayed task snapshot arrives.

    This is read-side custody, never review authority. Attempt identity comes
    from the task; publication_revision only orders snapshots of the SAME
    panel. Supersession cannot be reversed by a stale or replayed projection.
    The local preparation incident follows the same ordering: a delayed
    snapshot or replica can neither erase a newer warning nor resurrect an
    incident that a newer publication resolved, whichever order the two arrive in.
    Publication stamps, late-notice receipts and the incident come only from
    task-acceptance checkpoints stored before the result review replaced that
    machine; nothing writes them now, and stored ones still merge this way.
    """
    if not isinstance(previous, dict) or not isinstance(incoming, dict):
        return incoming
    # A stale child snapshot cannot revoke the outbox's accepted duty. This
    # receipt union is independent of reviewer authority and panel ordering,
    # including legacy unstamped projections; one delivery id never migrates
    # to another panel through a merge.
    notices = {}
    for source in (previous, incoming):
        for delivery_id, receipt in (source.get("late_notice_receipts") or {}).items():
            if not isinstance(receipt, dict):
                continue
            prior = notices.get(delivery_id)
            def rank_notice(value):
                revision = value.get("publication_revision")
                return (value.get("custody") == "terminal_outbox", revision if type(revision) is int else 0)
            if prior and (prior.get("panel_id") != receipt.get("panel_id") or rank_notice(prior) >= rank_notice(receipt)):
                continue
            notices[delivery_id] = copy.deepcopy(receipt)
    if notices:
        incoming = {**incoming, "late_notice_receipts": notices}
    old_rows, new_rows = previous.get("panels"), incoming.get("panels")
    if not isinstance(old_rows, list) or not isinstance(new_rows, list):
        return incoming
    previous_order, incoming_order = _publication_order(previous), _publication_order(incoming)
    if previous_order == incoming_order == (0, 0):
        return incoming  # unchanged legacy merge semantics
    def rank(value: Dict[str, Any]) -> tuple:
        return (bool(value.get("superseded")),
                value.get("publication_revision") if type(value.get("publication_revision")) is int else 0)

    merged: Dict[tuple, Dict[str, Any]] = {}
    for index, row in enumerate(old_rows + new_rows):
        if not isinstance(row, dict):
            continue
        key = (str(row.get("surface") or ""), str(row.get("task_attempt") or ""),
               str(row.get("panel_id") or f"legacy:{index}"), row.get("panel_index"))
        prior = merged.get(key)
        if prior is None or rank(row) > rank(prior):
            merged[key] = copy.deepcopy(row)
    rows = list(merged.values())
    rows.sort(key=lambda row: (
        row.get("task_attempt") if type(row.get("task_attempt")) is int else 0,
        row.get("panel_index") if type(row.get("panel_index")) is int else 0,
    ))
    merged = {**previous, **incoming, "panels": rows}
    # The local preparation incident is a fact of the NEWEST publication, not
    # read-side custody: that snapshot's presence or absence of it stands, and
    # a newer snapshot that carries none says the trace holds none.
    newest = previous if incoming_order < previous_order else incoming
    if "acceptance_incident" in newest:
        merged["acceptance_incident"] = copy.deepcopy(newest["acceptance_incident"])
    else:
        merged.pop("acceptance_incident", None)
    # The merged snapshot keeps the newer of the two stamps it was built from,
    # and only while that stamp still names its newest publication: when a
    # panel is newer than both stamps, the next checkpoint stamps again.
    stamped = max((previous, incoming), key=_stamp_order)
    for key in ("publication_revision", "task_attempt"):
        if _stamp_order(stamped) >= max(previous_order, incoming_order) and type(stamped.get(key)) is int:
            merged[key] = stamped[key]
        else:
            merged.pop(key, None)
    return merged
