"""Panel identity and the compact, redacted projection of a review run.

Owns the outward-facing view of a completed panel: full applied acceptance
source persistence through the existing artifact/task-result owners, transport-failure
classification, redaction of model-authored reason text, the per-actor and
per-panel projections that task results and the UI consume, the enforcement
impact label, and the panel-identity hash over the actor rows. Extracted from
ouroboros/review_substrate.py (v7 D06 split, re-cut on the v7next tip);
review_substrate.py re-exports every name.
"""

from __future__ import annotations

import hashlib
import json
import copy
import logging
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, TYPE_CHECKING

from ouroboros.review_records import review_slot_awaiting, review_slot_unresolved

# A slot released at the dispatch barrier has no transport or parse event:
# its projection is a gap, never a transport failure or a malformed answer.
AWAITING_PROJECTION = "awaiting"
_AWAITING_REASON = "No answer recorded: the host returned at the dispatch barrier before this reviewer answered."
# The closed vocabulary of a plan review's outcome CLASS at delivery, beside the
# awaited case above: stamped by owner_hurry.force_plan_decision from the wave's
# slot census and carried on outcome_axes.execution.plan_review. Nothing branches
# on it except the two cause renderers (project_dialogue / log_events).
PLAN_REVIEW_UNANSWERED = "unanswered"      # some answered; some failed, were refused at $0 or are unresolved
PLAN_REVIEW_NONE_ANSWERED = "none_answered"  # nobody answered and nobody is merely awaited
PLAN_REVIEW_ANSWERED_OPEN = "answered_open"  # everybody answered; the verdict was not closed

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
    awaiting = review_slot_awaiting(row) and parsed is None and not str(row.get("raw_text") or "").strip()
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
    reason = reason or error or ("Reviewer response was malformed or absent." if not valid else "")
    if awaiting:
        reason = _AWAITING_REASON
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


def build_review_binding(
    *,
    candidate: str,
    evidence: Dict[str, Any],
    fence_token_or_state: Any,
) -> Dict[str, Any]:
    """Build the exact host-panel identity without introducing another ledger."""
    from ouroboros.review_evidence import task_acceptance_evidence_revision

    candidate_hash = hashlib.sha256(str(candidate or "").encode("utf-8")).hexdigest()
    evidence_revision = task_acceptance_evidence_revision(evidence)
    fence_value = (
        json.dumps(fence_token_or_state, sort_keys=True, separators=(",", ":"), default=str)
        if isinstance(fence_token_or_state, (dict, list, tuple))
        else str(fence_token_or_state or "direct_context")
    )
    fence_hash = hashlib.sha256(fence_value.encode("utf-8")).hexdigest()
    binding_payload = {
        "candidate_hash": candidate_hash,
        "evidence_revision": evidence_revision,
        "fence_hash": fence_hash,
    }
    binding_hash = _sub().review_binding_hash(**binding_payload)
    return {
        **binding_payload,
        "binding_hash": binding_hash,
        "panel_id": f"panel_{binding_hash[:16]}",
    }


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


def _ledger_seat_actor(seat: Dict[str, Any], record: Dict[str, Any]) -> Dict[str, Any]:
    """One ledger seat as a card reviewer: the parts it was asked, each part's answer
    with its findings COUNTED (the bodies ride once, bounded, in ``findings``), and the
    observed model, else the requested one (``panel_facts`` counts the unobserved)."""
    status, surface = str(seat.get("status") or ""), str(record.get("surface") or "review")
    requested = seat.get("requested") if isinstance(seat.get("requested"), dict) else {}
    observed = str(seat.get("observed_model") or "")
    model = observed if observed not in ("", "unknown") else str(requested.get("model") or "")
    answers = seat.get("answers") if isinstance(seat.get("answers"), dict) else {}
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
        "provider": _sub().provider_for_model(model) if model else "unknown",
        "actor_role": f"{surface} {'additional ' if seat.get('additional') else ''}reviewer",
        "transport_status": str(seat.get("transport_status") or ("success" if status in _LEDGER_PARSE else status or "unknown")),
        "parse_status": _LEDGER_PARSE.get(status, "none"),
        "semantic_verdict": str(((record.get("verdict") or {}).get("per_row") or {}).get(seat.get("seat_id")) or ""),
        "quorum_contribution": contributes,
        "enforcement_impact": str(record.get("enforcement") or "unknown") if contributes else "abstains",
        "operation_state": str(seat.get("operation_state") or ""),
        "parts": parts, "answers": counted, "response_ref": _response_ref_projection(ref),
        "reason": _actor_reason(seat, str(seat.get("raw_text") or seat.get("failure_code") or "")
                                if status not in _LEDGER_PARSE else ""),
    }
    actor.update({key: seat[key] for key in _sub().TYPED_FAILURE_FACT_KEYS if seat.get(key) not in (None, "")})
    actor.update(_sub().disclosed_list_projection(
        findings, key="findings", limit=_sub().MAX_PROJECTED_ACTOR_FINDINGS, item=_sub().projected_finding_row))
    return actor


def ledger_record_panel(record: Dict[str, Any]) -> Dict[str, Any]:
    """One review-ledger record (``review_ledger.build_wave_record``) as a card panel:
    the record's ``panel`` facts (who sat and why) and every seat's answer per brief
    part. The aggregate is the record's own; the record file stays the full source."""
    from ouroboros.review_ledger import LEDGER_SUBDIR

    verdict = record.get("verdict") if isinstance(record.get("verdict"), dict) else {}
    quorum = verdict.get("quorum") if isinstance(verdict.get("quorum"), dict) else {}
    actors = [_ledger_seat_actor(seat, record) for seat in record.get("rows") or [] if isinstance(seat, dict)]
    parsed = {actor["parse_status"] for actor in actors}
    facts = dict(record.get("panel") or {}) if isinstance(record.get("panel"), dict) else {}
    if facts.get("reason"):
        facts["reason"] = _public_review_reason(facts["reason"])
    aggregate, record_id = str(verdict.get("aggregate") or "UNKNOWN").upper(), str(record.get("record_id") or "")
    return {
        "panel_id": record_id, "record_id": record_id, "ts": str(record.get("ts") or ""),
        "surface": str(record.get("surface") or "review"), "authority": "review_ledger",
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

        rows = [row for row in ledger.recent_records(root, task_id=task_id, limit=32, hot_only=True)
                if row.get("task_id") == task_id]
        records = [ledger.load_record(root, str(row.get("record_id") or "")) for row in rows[:LEDGER_RECORD_PANELS]]
        more = "1+" if len(rows) > LEDGER_RECORD_PANELS else ("unknown" if ledger.archived_segments_exist(root) else 0)
    except Exception:
        logging.getLogger(__name__).debug("Review ledger records unavailable for task %s", task_id, exc_info=True)
        return [], "unknown"
    return [record for record in reversed(records) if record], more


def compact_review_projection(review_runs: Any, records: Any = ()) -> Dict[str, Any]:
    """Project existing audit runs without copying raw prompts or responses, then the
    given review-ledger ``records`` (:func:`ledger_record_panel`), in their order."""
    panels: List[Dict[str, Any]] = []
    for index, raw_run in enumerate(review_runs or []):
        if not isinstance(raw_run, dict):
            continue
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
                from ouroboros.acceptance_settlement import owner_bounded_text

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
    panels.extend(ledger_record_panel(record) for record in records or () if isinstance(record, dict))
    return {"panels": panels}


def _applied_source_unchanged(run: Dict[str, Any], raw: bytes) -> bool:
    """Does this panel's stored applied source already hold exactly ``raw``?

    The write-once source handle names its bytes by digest, so equality here is
    proof that nothing in the panel's record moved since its last publication.
    A panel whose stored source is unavailable cannot prove that and is
    published again (one more attempt at the store, never a new identity).
    """
    ref = run.get("applied_source_ref")
    return isinstance(ref, dict) and str(ref.get("sha256") or "") == hashlib.sha256(raw).hexdigest()


def _actor_pending(actor: Dict[str, Any]) -> bool:
    return (actor.get("transport_status") == AWAITING_PROJECTION
            or actor.get("operation_state") in {"pending_dispatch", "in_flight"})


def _producer_fact(actor: Dict[str, Any]) -> tuple:
    """What one settled producer said, by identity and stored response, never by prose."""
    ref = actor.get("response_ref") if isinstance(actor.get("response_ref"), dict) else {}
    return (actor.get("operation_state"), actor.get("transport_status"), actor.get("parse_status"),
            actor.get("semantic_verdict"), ref.get("sha256") or ref.get("call_id") or "")


def _panel_merge(stored: Dict[str, Any], incoming: Dict[str, Any]) -> str:
    """``advance``, ``same``, ``stored`` or ``conflict`` for one panel, compared slot by slot.

    A slot advances (stored pending, incoming settled), regresses (the reverse)
    or must carry the same producer facts. A snapshot that only advances wins;
    one that only regresses is stale; a different operation, roster, subject
    binding or settled fact — or a snapshot that advances one slot while
    regressing another — is a conflict: neither projection can speak for both,
    so the stored panel stays and the publisher is told.
    """
    for key in ("binding_hash", "candidate_hash"):
        if stored.get(key) and incoming.get(key) and stored[key] != incoming[key]:
            return "conflict"

    def by_slot(panel: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        return {str(actor.get("slot_id") or ""): actor for actor in panel.get("actors") or [] if isinstance(actor, dict)}

    before, after = by_slot(stored), by_slot(incoming)
    if set(before) != set(after):
        return "conflict"
    advances = regressions = 0
    for slot_id, old in before.items():
        new = after[slot_id]
        if old.get("operation_id") and new.get("operation_id") and old["operation_id"] != new["operation_id"]:
            return "conflict"
        old_pending, new_pending = _actor_pending(old), _actor_pending(new)
        if old_pending and not new_pending:
            advances += 1
        elif new_pending and not old_pending:
            regressions += 1
        elif not old_pending and _producer_fact(old) != _producer_fact(new):
            return "conflict"
    if advances and regressions:
        return "conflict"
    return "stored" if regressions else "advance" if advances else "same"


def _keep_newer_producer_facts(existing: Any, incoming: Dict[str, Any]) -> tuple:
    """Merge one incoming snapshot into the CANONICAL stored panels, slot by slot.

    Another process (a late collector, the maintenance pass) may have published
    a settled verdict or a late fact for the same panel. A stale snapshot keeps
    the stored panel; a conflicting one is rejected rather than overwriting a
    settled verdict; the first published late settlement keeps its bytes and
    ``settled_at``; supersession only grows. Only a genuine advance is ordered
    after the stored panel (its revision is raised past it); an unchanged
    snapshot keeps its own revision, so a delayed older one never wins the
    read-side merge. Returns ``(projection, rejected)``.
    """
    if not isinstance(existing, dict):
        return incoming, []

    def key(row: Dict[str, Any]) -> tuple:
        return (str(row.get("surface") or ""), str(row.get("task_attempt") or ""), str(row.get("panel_id") or ""),
                row.get("panel_index"))

    stored = {key(row): row for row in existing.get("panels") or [] if isinstance(row, dict) and row.get("panel_id")}
    panels, rejected, stamp = [], [], incoming.get("publication_revision") or 0
    for row in incoming.get("panels") or []:
        prior = stored.get(key(row)) if isinstance(row, dict) and row.get("panel_id") else None
        if isinstance(prior, dict):
            superseded = bool(prior.get("superseded") or row.get("superseded"))
            decision = _panel_merge(prior, row)
            if decision == "conflict":
                rejected.append(str(row.get("panel_id") or ""))
            if decision in {"stored", "conflict"}:
                # The stored panel stands (the read-side union keeps it); this
                # snapshot only adds a supersession it knows of.
                if superseded and not prior.get("superseded"):
                    panels.append({**copy.deepcopy(prior), "superseded": True})
                continue
            row = {**row, "superseded": superseded}
            if isinstance(prior.get("late_settlement"), dict):
                row["late_settlement"] = copy.deepcopy(prior["late_settlement"])
            if decision == "advance" and type(prior.get("publication_revision")) is int:
                row["publication_revision"] = max(int(row.get("publication_revision") or 0),
                                                  prior["publication_revision"] + 1)
                stamp = max(stamp, row["publication_revision"])
        panels.append(row)
    return {**incoming, "panels": panels, "publication_revision": stamp}, rejected


def publish_acceptance_checkpoint(
    ctx: Any, llm_trace: Dict[str, Any], *, task_id: str = "",
    drive_root: Any = None, chat_id: Any = None, partial_trace: bool = False,
) -> Dict[str, Any]:
    """Save the complete applied host record before publishing its read model.

    This does not grant review authority or alter task lifecycle. Source bytes
    use the existing immutable artifact store; publication_revision only orders
    snapshots of a panel. Under the result lock each panel is compared slot by
    slot with the CANONICAL stored one (``_keep_newer_producer_facts``): only a
    genuine producer advance is ordered past it, so neither a delayed older
    snapshot nor a publication from another process (a late collector, the
    maintenance pass) can overwrite a newer verdict. ``partial_trace`` (a trace
    rebuilt from the published source) keeps the stored incident.

    Returns what the canonical record now holds, read back from the locked
    write itself: ``status`` (``published``, ``unavailable`` or ``skipped``),
    the allocated ``revision``, the panel ids ``rejected`` as conflicting with
    stored producer facts, and the stored ``projection``. A caller that must not
    claim success unless its panel landed checks the returned projection.
    """
    from pathlib import Path

    from ouroboros.artifacts import store_actor_source_bytes
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.plan_review_references import _emit_review_reference

    runs = [run for run in (llm_trace.get("review_runs") or [])
            if isinstance(run, dict) and run.get("authority") == "host_root"
            and isinstance(run.get("request"), dict)
            and (run.get("request") or {}).get("surface") == "task_acceptance"]
    task_id = str(task_id or getattr(ctx, "task_id", "") or "")
    meta = getattr(ctx, "task_metadata", {})
    meta = meta if isinstance(meta, dict) else {}
    root = meta.get("budget_drive_root") or getattr(ctx, "budget_drive_root", None) or drive_root or getattr(ctx, "drive_root", None)
    # A LOCAL preparation failure produces no panel at all, and that absence is
    # exactly what the owner could not see (#1224). The incident rides the same
    # projection the panels do, so live delivery, history and a reconnect read
    # one fact with one identity.
    from ouroboros.acceptance_preparation import TASK_ONLY_ORIGINS, current_incident, incident_projection

    # The incident's diagnostic detail is host text, so it is redacted exactly
    # like every other published section before it leaves for an owner surface.
    incident = _sub().redact_projection(incident_projection(current_incident(llm_trace))).value
    if (not runs and not incident) or not task_id or not root:
        return {"status": "skipped", "revision": 0, "rejected": [], "projection": {}}
    revision = int(llm_trace.get("_acceptance_publication_revision") or 0) + 1
    llm_trace["_acceptance_publication_revision"] = revision
    snapshots = copy.deepcopy(llm_trace.get("review_runs") or [])
    # A task-only decision (the host's own local preparation or processing
    # failure) is no panel's applied reviewer decision: it rewrites no prior
    # panel's custody and grants a never-published LIVE producer none. A panel
    # whose OWN record moved since its last publication — a pending producer
    # that settled, a late-settlement note — is a genuine producer update and is
    # published under the next revision, and so is a settled (or custody-lost)
    # record that was never published at all: a missing publication stamp alone
    # never suppresses a real verdict. An unchanged record stays bound to the
    # revision and stored source it already has (its digest proves it unchanged).
    from ouroboros.loop_acceptance_review import acceptance_run_pending

    task_only = (llm_trace.get("acceptance_decision") or {}).get("origin") in TASK_ONLY_ORIGINS
    for index, run in enumerate(snapshots):
        if not isinstance(run, dict) or run.get("authority") != "host_root" or not isinstance(run.get("request"), dict) or run["request"].get("surface") != "task_acceptance":
            continue
        if task_only and "publication_revision" not in run and acceptance_run_pending(run):
            continue
        run.setdefault("panel_id", f"panel_{index + 1}")
        run.setdefault("panel_index", index)
        source = {key: value for key, value in run.items()
                  if key not in {"applied_source_ref", "applied_source_status", "publication_revision"}}
        raw = json.dumps(_sub().redact_projection(source).value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        if task_only and _applied_source_unchanged(run, raw):
            continue
        run.pop("applied_source_ref", None)
        run["applied_source_status"] = "unavailable"
        try:
            run["applied_source_ref"] = store_actor_source_bytes(
                Path(root), task_id, category="context_checkpoints",
                source_id="acceptance", data=raw, extension="json",
            )
            run["applied_source_status"] = "available"
        except (OSError, ValueError, TimeoutError):
            logging.getLogger(__name__).warning("Applied acceptance source unavailable", exc_info=True)
        run["publication_revision"] = revision
        # A later snapshot may have completed during slow artifact I/O. It owns
        # the in-memory projection too; the durable merge applies the same rule.
        if llm_trace.get("_acceptance_publication_revision") == revision:
            current = llm_trace["review_runs"][index]
            for key in ("panel_id", "panel_index", "task_attempt", "publication_revision", "applied_source_ref", "applied_source_status"):
                if key in run:
                    current[key] = copy.deepcopy(run[key])
            if "applied_source_ref" not in run:
                current.pop("applied_source_ref", None)
    projection = compact_review_projection(snapshots)
    # The snapshot carries its own ordering stamp, so the read-side merge can
    # order the incident's presence and resolution — including a snapshot with
    # no panel at all — exactly as it orders each panel's publications.
    projection["publication_revision"] = revision
    attempt = getattr(ctx, "task_attempt", None)
    if type(attempt) is int:
        projection["task_attempt"] = attempt
    if incident:
        projection["acceptance_incident"] = incident
    published: Dict[str, Any] = {}

    def project(current: Dict[str, Any], fields: Dict[str, Any]) -> Dict[str, Any]:
        existing = current.get("review_projection") if isinstance(current.get("review_projection"), dict) else None
        incoming = copy.deepcopy(projection)
        if partial_trace and isinstance(existing, dict) and "acceptance_incident" in existing:
            incoming["acceptance_incident"] = copy.deepcopy(existing["acceptance_incident"])
        incoming, rejected = _keep_newer_producer_facts(existing, incoming)
        published.update(revision=incoming["publication_revision"], rejected=rejected,
                         bumped={str(row.get("panel_id") or ""): row["publication_revision"]
                                 for row in incoming.get("panels") or []
                                 if isinstance(row, dict) and type(row.get("publication_revision")) is int
                                 and row["publication_revision"] > revision})
        return {"review_projection": merge_review_projection(existing, incoming),
                "status": current.get("status") or fields["status"]}

    try:
        result = write_task_result(root, task_id, "running", strict_existing_dict=True, _field_projector=project)
    except (OSError, ValueError, TimeoutError) as exc:
        logging.getLogger(__name__).warning("Applied acceptance projection unavailable", exc_info=True)
        return {"status": "unavailable", "revision": 0, "rejected": [], "projection": {},
                "error": f"{type(exc).__name__}: {exc}"}
    allocated = published.get("revision", revision)
    if published.get("rejected"):
        logging.getLogger(__name__).warning("Acceptance panels %s conflict with stored producer facts; kept stored",
                                            published["rejected"])
    for run in llm_trace.get("review_runs") or []:
        # An advance ordered past another process's stored panel carries that order on.
        if isinstance(run, dict) and str(run.get("panel_id") or "") in (published.get("bumped") or {}) \
                and run.get("publication_revision") == revision:
            run["publication_revision"] = published["bumped"][str(run["panel_id"])]
    llm_trace["_acceptance_publication_revision"] = max(
        int(llm_trace.get("_acceptance_publication_revision") or 0), allocated)
    state = result.get("review_projection") if isinstance(result.get("review_projection"), dict) else {}
    try:
        _emit_review_reference(ctx, task_id, state, surface="task_acceptance", state_root=Path(root), chat_id=chat_id)
    except (OSError, ValueError, TimeoutError):
        logging.getLogger(__name__).warning("Acceptance review reference unavailable", exc_info=True)
    return {"status": "published" if "revision" in published else "unavailable", "revision": allocated,
            "rejected": list(published.get("rejected") or []), "projection": copy.deepcopy(state)}


def acceptance_decision_projection(acceptance_decision: Dict[str, Any], subject_hash: str = "") -> Dict[str, Any]:
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

        from ouroboros.acceptance_preparation import LOCAL_PREPARATION_ORIGIN

        if acceptance_decision.get("origin") == LOCAL_PREPARATION_ORIGIN:
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
