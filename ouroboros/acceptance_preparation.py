"""The LOCAL, pre-binding stage of one host acceptance panel.

Assembling the acceptance packet can fail before any reviewer exists. That
failure used to be identified by the exception's own text and an EMPTY binding
hash, so two exception types over the same unchanged material read as two
incidents and the host/author pair looped without a single new reviewer.

What identifies a preparation attempt is the MATERIAL it would assemble: the
SEMANTIC criteria (Main's explicit current criteria, else the task contract),
canonical receipt/artifact/child material, and a bounded content
identity of the working-tree SOURCE the builder reads. Deliberately NOT the
owner transcript (a status question is a new owner message, not new material;
its digest is kept beside the record as an acknowledgement), the candidate
prose, the tool count, the owner-message generation or the error type. So a
repeated final, a rephrasing or a status question is the same incident and buys
no rebuild; changed criteria or a repaired source file is a new one. Material
that cannot be read is explicitly unknown; loss of availability preserves the
last known identity and its counters. Only newly proven material reopens it.

A new attempt otherwise needs an explicit, one-use, source-bound retry intent
(material change, repaired evidence or an owner retry) through the existing
``task_acceptance_review`` contract: re-delivery is idempotent, a spent
declaration stays spent.

This module owns no ledger, timer or lifecycle: the incident lives on the
existing llm_trace, and the paid-review identities (`binding_hash`,
`paid_identity`, `panel_id`) are untouched — a preparation incident is a
DIFFERENT identity from the money the tree spends.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import time
from hashlib import sha256
from typing import Any, Dict, List, Tuple

log = logging.getLogger("ouroboros.loop")

TRACE_KEY = "acceptance_preparation"

# The typed stages one host acceptance pass moves through. Only `preparation`
# is genuinely PRE-DISPATCH: solely there may the host state that no new
# reviewer was dispatched. `reconcile` touches runs that were already
# dispatched, so its failures never claim that sentence.
STAGE_PREPARATION = "preparation"
STAGE_RECONCILE = "reconcile"
STAGE_DISPATCH = "dispatch"
STAGE_APPLICATION = "application"

STATUS_OPEN = "open"
STATUS_FAILED = "failed"
STATUS_RESOLVED = "resolved"

UNKNOWN_SOURCE_IDENTITY = "unknown"
LOCAL_PREPARATION_ORIGIN = "local_acceptance_preparation"
HOST_PROCESSING_ORIGIN = "host_acceptance_processing"
TASK_ONLY_ORIGINS = frozenset({LOCAL_PREPARATION_ORIGIN, HOST_PROCESSING_ORIGIN})

# The substantive grounds for ONE more attempt over the same material. Semantics
# are Main's to state; the host never greps the owner's words for them.
RETRY_BASES = frozenset({"material_change", "repair_evidence", "owner_retry"})

_HISTORY_MAX = 10


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def _semantic_criteria(tool_ctx: Any) -> Any:
    """Main's explicit current criteria, else the task contract.

    The owner-directive LIST is deliberately not here: a status question is a
    new owner message, not a new requirement (owner decision 2). A real change
    reaches this identity through ``acceptance_subject.effective_criteria`` or
    an explicit ``material_change`` retry.
    """
    explicit = getattr(tool_ctx, "_delivery_effective_criteria", None)
    if explicit is not None:
        return explicit
    from ouroboros.review_evidence_sections import _accept_task_contract

    return {"task_contract": _accept_task_contract(tool_ctx)}


def _repo_of(tool_ctx: Any) -> Any:
    """The same active-repository resolution ``collect_turn_diff`` uses."""
    getter = getattr(tool_ctx, "active_repo_dir", None)
    repo = getter() if callable(getter) else getter
    return repo or getattr(tool_ctx, "repo_dir", None)


def _canonical_set(rows: Any) -> List[str]:
    """Content identities, independent of delivery order and multiplicity."""
    return sorted({json.dumps(row, ensure_ascii=False, sort_keys=True,
                              separators=(",", ":"), default=str) for row in rows})


def _artifact_material(root: Any, task_id: str) -> List[Dict[str, Any]]:
    """Hash actual task outputs, excluding host source-handle bookkeeping.

    A partial inventory, unreadable file or exceeded bound is unknown; neither
    a preview nor size/mtime is evidence of content equality.
    """
    from ouroboros.artifacts import _ARTIFACT_MANIFEST, _SOURCE_HANDLES_SUBDIR
    from ouroboros.repo_diff_capture import (
        SOURCE_IDENTITY_MAX_BYTES, SOURCE_IDENTITY_MAX_FILES, SOURCE_IDENTITY_TIMEOUT_SEC,
        _path_identity,
    )
    from ouroboros.task_results import validate_task_id
    from ouroboros.outcome_receipt_store import verification_receipts_path
    from ouroboros.utils import jsonl_append_lock_path

    if not root or not task_id:
        return []
    base = pathlib.Path(root) / "task_results" / "artifacts" / validate_task_id(task_id)
    receipt_lock = jsonl_append_lock_path(verification_receipts_path(root, task_id)).name
    try:
        base.stat()
    except FileNotFoundError:
        return []
    budget = SOURCE_IDENTITY_MAX_BYTES
    deadline = time.monotonic() + SOURCE_IDENTITY_TIMEOUT_SEC
    rows: List[Dict[str, Any]] = []

    def unreadable(exc):
        raise exc

    for parent, dirs, files in os.walk(base, onerror=unreadable, followlinks=False):
        if pathlib.Path(parent) == base:
            dirs[:] = [d for d in dirs if d != _SOURCE_HANDLES_SUBDIR]
        if any((pathlib.Path(parent) / d).is_symlink() for d in dirs):
            raise OSError("artifact directory identity unavailable")
        for name in sorted(files):
            rel = str((pathlib.Path(parent) / name).relative_to(base))
            if rel in {_ARTIFACT_MANIFEST, _ARTIFACT_MANIFEST + ".lock",
                       "verification_receipts.jsonl", receipt_lock}:
                continue
            row, used = _path_identity(base, rel, budget, hash_content=time.monotonic() <= deadline)
            if row.get("state") != "hashed" or len(rows) >= SOURCE_IDENTITY_MAX_FILES:
                raise OSError("artifact material identity incomplete")
            rows.append(row)
            budget -= used
    return sorted(rows, key=lambda row: row["path"])


def preparation_material(tool_ctx: Any, llm_trace: Dict[str, Any]) -> Dict[str, Any]:
    """Preparation content only. The paid delivery fingerprint is unchanged.

    Tool history, selected indices and repeated writes never name this incident.
    Receipts retain semantic content (including order of differing results),
    while adjacent duplicate deliveries and transport timestamps do not.
    """
    from ouroboros.outcomes import read_context_verification_receipts
    from ouroboros.tools.join_ledger import _child_result_sha256

    meta = getattr(tool_ctx, "task_metadata", {}) or {}
    task_id = str(getattr(tool_ctx, "task_id", "") or "")
    root_task_id = str(meta.get("root_task_id") or task_id)
    root = (meta.get("budget_drive_root") or getattr(tool_ctx, "budget_drive_root", None)
            or getattr(tool_ctx, "drive_root", None))
    gaps: set[str] = set()
    receipts = read_context_verification_receipts(tool_ctx, task_id, fallback_root=root,
                                                gap_reasons=gaps) if task_id else []
    if gaps:
        raise OSError("verification receipt material unavailable")
    semantic_receipts = []
    for receipt in receipts:
        row = {key: value for key, value in receipt.items()
               if key not in {"ts", "recorded_at", "tool_call_id", "round_id", "execution_id"}}
        if not semantic_receipts or semantic_receipts[-1] != row:
            semantic_receipts.append(row)
    children = _loop()._load_direct_child_results(pathlib.Path(root), task_id, root_task_id) if root and task_id else []
    return {
        "verification_receipts": semantic_receipts,
        "artifacts": _artifact_material(getattr(tool_ctx, "drive_root", None) or root, task_id),
        "children": _canonical_set({
            "task_id": str(child.get("task_id") or child.get("id") or ""),
            "status": str(child.get("status") or ""), "sha256": _child_result_sha256(child),
            "disposition": _loop()._child_disposition_state(child),
        } for child in children),
        "service_finalization": _canonical_set(_loop()._service_finalization_evidence(llm_trace)),
    }


def preparation_source_identity(tool_ctx: Any, llm_trace: Dict[str, Any]) -> Dict[str, Any]:
    """Identity of the MATERIAL this preparation would assemble.

    Semantic criteria, canonical receipt/artifact/child material,
    and the bounded working-tree source identity — nothing derived from the
    owner transcript, the candidate's prose, the tool count, the owner
    generation or the failure that is about to happen. Never raises.
    """
    unknown: List[str] = []
    criteria: Any = None
    try:
        criteria = _json_safe(_semantic_criteria(tool_ctx))
    except Exception:
        unknown.append("effective_criteria")
    try:
        material = preparation_material(tool_ctx, llm_trace)
    except Exception:
        material = ""
        unknown.append("material_evidence")
    try:
        from ouroboros.repo_diff_capture import repo_source_identity

        repo = _repo_of(tool_ctx)
        source = repo_source_identity(repo) if repo else "no_repository"
    except Exception:
        source = ""
        unknown.append("repository_source")
    try:
        from ouroboros.loop_messages import owner_source_sha256

        owner = str(owner_source_sha256(tool_ctx) or "")
    except Exception:
        owner = ""
    if unknown:
        # Explicitly unknown, and the SAME unknown every round: an unreadable
        # source is one incident, not a fresh one per attempt.
        return {"identity": UNKNOWN_SOURCE_IDENTITY, "known": False,
                "unknown_parts": sorted(unknown), "owner_source_sha256": owner}
    payload = json.dumps([criteria, material, source], ensure_ascii=False,
                         sort_keys=True, separators=(",", ":"), default=str)
    return {"identity": sha256(payload.encode("utf-8")).hexdigest(), "known": True,
            "unknown_parts": [], "owner_source_sha256": owner}


def incident_id_for(identity: str) -> str:
    """A preparation incident's own stable id — never a binding or paid identity."""
    return f"acceptance-preparation:{str(identity or UNKNOWN_SOURCE_IDENTITY)[:32]}"


def current_incident(llm_trace: Dict[str, Any]) -> Dict[str, Any]:
    """The incident record on the trace, or {} when none was ever opened."""
    record = llm_trace.get(TRACE_KEY) if isinstance(llm_trace, dict) else None
    return record if isinstance(record, dict) else {}


def _history_row(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "incident_id": str(record.get("incident_id") or ""),
        "source_identity": str(record.get("source_identity") or ""),
        "attempts": int(record.get("attempts") or 0),
        "status": str(record.get("status") or ""),
        "failure_kind": str(record.get("failure_kind") or ""),
        "stage": str(record.get("stage") or ""),
        "source_known": bool(record.get("source_known")),
    }


def begin_preparation(llm_trace: Dict[str, Any], tool_ctx: Any) -> Dict[str, Any]:
    """Register this round's preparation identity BEFORE the fallible builder.

    Called with a builder that is about to raise, this is what still yields a
    non-empty incident identity; called with one that succeeds, it is what a
    later failure is compared against. Never raises.
    """
    source = preparation_source_identity(tool_ctx, llm_trace)
    record = current_incident(llm_trace)
    # Unknown is missing evidence, never evidence of a material change. Retain
    # the last known identity through an outage, including spent retry grants.
    identity = (source["identity"] if source["known"] or not record
                else str(record.get("source_identity") or UNKNOWN_SOURCE_IDENTITY))
    if record and str(record.get("source_identity") or "") != identity:
        # Different material: the previous incident is history, and the new
        # identity gets its own first attempt (owner decision 2, 22.09 23:33).
        history = [row for row in (record.get("history") or []) if isinstance(row, dict)]
        history.append(_history_row(record))
        record = {"history": history[-_HISTORY_MAX:]}
    elif not record:
        record = {"history": []}
    record.update({
        "incident_id": incident_id_for(identity),
        "source_identity": identity,
        "source_known": bool(source["known"]),
        "unknown_parts": list(source["unknown_parts"]),
        # The owner corpus the host last read: an acknowledgement kept BESIDE
        # the identity, never part of it (a status question changes only this).
        "owner_source_sha256": str(source.get("owner_source_sha256") or ""),
        "status": str(record.get("status") or STATUS_OPEN),
    })
    record.setdefault("attempts", 0)
    record.setdefault("exposed_attempt", 0)
    record.setdefault("retry_keys", [])
    llm_trace[TRACE_KEY] = record
    return record


def record_preparation_failure(
    record: Dict[str, Any], exc: BaseException, *, stage: str = STAGE_PREPARATION,
) -> Dict[str, Any]:
    """Count one REAL host attempt and keep the failure's diagnostics.

    ``failure_kind``/``failure_detail`` are diagnostics, never identity: an
    alternating exception type over unchanged material stays one incident.
    """
    record["status"] = STATUS_FAILED
    record["stage"] = str(stage or STAGE_PREPARATION)
    record["attempts"] = int(record.get("attempts") or 0) + 1
    record["failure_kind"] = type(exc).__name__
    from ouroboros.observability import redact_projection

    record["failure_detail"] = str(redact_projection(str(exc)).value)[:500]
    return record


def record_preparation_success(record: Dict[str, Any]) -> Dict[str, Any]:
    """A successful preparation closes the current incident; its failures stay
    in history whether the record was still failed or reopened by a retry."""
    if not record:
        return record
    if int(record.get("attempts") or 0) > 0:
        history = [row for row in (record.get("history") or []) if isinstance(row, dict)]
        history.append(_history_row(record))
        record["history"] = history[-_HISTORY_MAX:]
    record["status"] = STATUS_RESOLVED
    record.pop("failure_kind", None)
    record.pop("failure_detail", None)
    return record


def preparation_exposed(record: Dict[str, Any]) -> bool:
    """Did the author actually RECEIVE this incident's CURRENT attempt?

    Queuing a message is not delivery: ``acceptance_settlement`` marks this only
    when the carrying Main request received a response, and only for the
    attempt that request carried — feedback about attempt 1 exposes nothing
    about attempt 2.
    """
    return bool(record.get("feedback_delivered")) and int(record.get("exposed_attempt") or 0) == int(
        record.get("attempts") or 0) > 0


def preparation_blocked(record: Dict[str, Any]) -> bool:
    """An exposed, unresolved failure over unchanged material: no rebuild.

    Re-running the same broken builder would only re-deliver the same failure —
    the loop owner decision 2 forbids. A new attempt needs new material (a new
    identity, so a different record) or an explicit one-use retry.
    """
    return bool(record) and str(record.get("status") or "") == STATUS_FAILED and preparation_exposed(record)


def preparation_delivery_choice(tool_ctx: Any, llm_trace: Dict[str, Any]) -> bool:
    """An informed root author may deliver the retained work with this incident.

    This validates a task-only choice, not evidence freshness or review approval.
    Owner acknowledgement, material identity and the exact retained answer still
    bind it. Child/off lanes and ordinary nominations get no fingerprint escape.
    """
    from ouroboros.loop_acceptance_review import _resolve_ctx_lineage
    from ouroboros.loop_messages import owner_source_sha256
    from ouroboros.review_records import validate_author_disposition

    record = current_incident(llm_trace)
    if not preparation_blocked(record) or _loop().get_task_review_mode() not in {"auto", "required"}:
        return False
    if not _resolve_ctx_lineage(tool_ctx)["is_root_task"]:
        return False
    if _loop()._task_acceptance_owner_generation_changed(tool_ctx):
        return False
    candidate = getattr(tool_ctx, "_delivery_candidate", None)
    if candidate is None:
        return False
    decision = llm_trace.get("acceptance_decision") or {}
    intent = decision.get("agent_finish_intent") or {}
    if intent:
        valid = (
            intent.get("incident_id") == record.get("incident_id")
            and intent.get("preparation_identity") == record.get("source_identity")
            and intent.get("incident_attempt") == record.get("attempts")
            # The act (finish|stop, checked below) is a stance by itself: an
            # action-only nomination keeps its empty disposition (C4).
            and str(decision.get("agent_disposition") or "") in {"", "accepted", "rejected", "partial", "deferred"}
            and bool(str(decision.get("agent_rationale") or "").strip())
        )
        choice = intent
    else:
        author = validate_author_disposition(decision.get("author_disposition"),
            subject_hash=f"{record.get('incident_id')}:attempt-{record.get('attempts')}")
        valid = (decision.get("origin") == LOCAL_PREPARATION_ORIGIN
                 and decision.get("status") == "finalized_unaccepted" and bool(author))
        choice = decision.get("preparation_delivery_choice") or {}
    action = choice.get("author_action")
    if not valid or action not in {"finish", "stop"} or (action == "finish" and _review_enforcement_blocks()):
        return False
    if (choice.get("candidate_sha256") != candidate.content_sha256
            or choice.get("owner_source_sha256") != owner_source_sha256(tool_ctx)):
        return False
    # The same rule as begin_preparation: unknown is missing evidence, never
    # evidence of a material change, so a transient read failure cannot turn an
    # informed stop/finish into "different material". Only proven other material does.
    source = preparation_source_identity(tool_ctx, llm_trace)
    return (not source["known"]) or source["identity"] == record.get("source_identity")


def resolve_retry_source(tool_ctx: Any, intent: Dict[str, Any]) -> Dict[str, str]:
    """Resolve an existing source; Main judges its meaning, never the host.

    A receipt index is only a locator. Its exact host row is the identity, so
    moving the row or changing rationale/basis cannot buy another attempt.
    """
    from ouroboros.loop_messages import owner_source_sha256
    from ouroboros.outcomes import read_context_verification_receipts

    owner = intent.get("owner_source_sha256")
    index = intent.get("verification_receipt_index")
    if owner and index is None and intent.get("basis") != "repair_evidence":
        current = owner_source_sha256(tool_ctx)
        if owner == current:
            return {"kind": "owner_source", "sha256": current}
    elif not owner and type(index) is int and index >= 0 and intent.get("basis") != "owner_retry":
        gaps: set[str] = set()
        receipts = read_context_verification_receipts(
            tool_ctx, str(getattr(tool_ctx, "task_id", "") or ""), gap_reasons=gaps)
        if (not gaps and index < len(receipts) and receipts[index].get("status") in {"pass", "fail", "observed"}
                and receipts[index].get("contract_kind") not in {"declared", "delegation_zero_run"}):
            row = json.dumps(receipts[index], ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), default=str)
            return {"kind": "verification_receipt", "sha256": sha256(row.encode("utf-8")).hexdigest()}
    raise ValueError("acceptance_retry needs the current owner_source_sha256 for owner_retry, "
                     "a current verification_receipt_index for repair_evidence, or either for material_change; "
                     "supply exactly one source, not a new rationale for a spent source")


def retry_intent_key(intent: Dict[str, Any], source: Dict[str, str]) -> str:
    """One incident/source grant, independent of narration, basis and locator."""
    payload = json.dumps([
        str(intent.get("incident_id") or ""), source["kind"], source["sha256"],
    ], ensure_ascii=False, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def record_retry_intent(llm_trace: Dict[str, Any], intent: Any, tool_ctx: Any = None) -> Dict[str, Any]:
    """Accept ONE source-bound retry declaration for the open incident.

    Bound: the declaration must name the current failed incident after its
    actual exposure; no queued future retry or resolved incident can grant one.
    Idempotent: a duplicate delivery of the current declaration grants nothing
    further, and a declaration that was ALREADY spent on this incident (A, B,
    A) is spent for good — the record keeps every key it ever granted. An
    ordinary nomination, a status question or new prose reaches none of this.
    """
    if not isinstance(llm_trace, dict) or not isinstance(intent, dict):
        return {}
    record = current_incident(llm_trace)
    if not preparation_blocked(record):
        return {}
    basis = str(intent.get("basis") or "").strip().lower()
    incident_id = str(intent.get("incident_id") or "").strip()
    if basis not in RETRY_BASES or incident_id != str(record.get("incident_id") or ""):
        return {}
    if tool_ctx is None:
        return {}  # No host authority to resolve a model-supplied source.
    try:
        source = resolve_retry_source(tool_ctx, {**intent, "basis": basis})
    except Exception:
        return {}
    if intent.get("source") is not None and intent["source"] != source:
        return {}  # The tool's resolved source changed before the result was consumed.
    key = retry_intent_key({"incident_id": incident_id}, source)
    existing = record.get("retry") if isinstance(record.get("retry"), dict) else {}
    if str(existing.get("key") or "") == key:
        return existing  # idempotent duplicate: still exactly one granted attempt
    keys = [str(k) for k in (record.get("retry_keys") or []) if isinstance(k, str)]
    if key in keys:
        return {}  # this exact declaration already bought its one attempt
    record["retry_keys"] = [*keys, key]  # task-bounded; eviction would reauthorize an old retry
    record["retry"] = {
        "key": key, "basis": basis, "incident_id": incident_id, "source": source,
        "rationale": " ".join(str(intent.get("rationale") or "").split())[:500],
        "granted_for_attempt": int(record.get("attempts") or 0) + 1,
        "consumed": False,
    }
    (llm_trace.get("acceptance_decision") or {}).pop("agent_finish_intent", None)
    return record["retry"]


def consume_retry(record: Dict[str, Any]) -> bool:
    """Spend a granted retry: ONE declaration buys exactly one new attempt."""
    retry = record.get("retry") if isinstance(record.get("retry"), dict) else {}
    if not retry or retry.get("consumed"):
        return False
    retry["consumed"] = True
    record["status"] = STATUS_OPEN
    return True


def incident_projection(record: Dict[str, Any]) -> Dict[str, Any]:
    """The owner/UI-facing facts of one incident: identity, count, stage, state.

    Carried by the EXISTING acceptance decision and review projections; it adds
    no lifecycle of its own and never carries the failure's raw source. A record
    that never counted a real failed attempt (the ordinary successful baseline
    preparation) projects NOTHING — there is no incident to show.
    """
    if not record:
        return {}
    attempts = int(record.get("attempts") or 0)
    if attempts <= 0:
        # Changed material starts a new counter. If its first assembly succeeds,
        # publish resolution of the previous failed incident under ITS id/count:
        # dropping it would leave the old warning cached on live/reconnected cards.
        history = record.get("history") or []
        previous = history[-1] if history and isinstance(history[-1], dict) else {}
        if (record.get("status") == STATUS_RESOLVED and previous.get("status") == STATUS_FAILED
                and int(previous.get("attempts") or 0) > 0):
            return incident_projection({**previous, "status": STATUS_RESOLVED})
        return {}
    out = {
        "incident_id": str(record.get("incident_id") or ""),
        "status": str(record.get("status") or ""),
        "stage": str(record.get("stage") or STAGE_PREPARATION),
        # REAL host attempts for this material; a replayed or skipped round
        # never increments it, so the owner-visible count stays truthful.
        "attempts": attempts,
        "source_known": bool(record.get("source_known")),
        "feedback_delivered": preparation_exposed(record),
    }
    if record.get("unknown_parts"):
        out["unknown_parts"] = [str(part) for part in record["unknown_parts"]][:5]
    if record.get("failure_kind"):
        out["failure_kind"] = str(record["failure_kind"])
    if record.get("failure_detail"):
        out["failure_detail"] = str(record["failure_detail"])[:500]
    retry = record.get("retry") if isinstance(record.get("retry"), dict) else {}
    if retry:
        out["retry"] = {"basis": str(retry.get("basis") or ""), "consumed": bool(retry.get("consumed"))}
    if record.get("history"):
        out["prior_incidents"] = len([row for row in record["history"] if isinstance(row, dict)])
    return out


def _attempt_words(record: Dict[str, Any]) -> str:
    attempts = int(record.get("attempts") or 0)
    return f"{attempts} failed host attempt{'s' if attempts != 1 else ''}"


def preparation_feedback_text(record: Dict[str, Any]) -> str:
    """The exact sentence the author receives once per real attempt."""
    identity = "could not be identified from its source" if not record.get("source_known") else (
        f"identity {str(record.get('source_identity') or '')[:12]}")
    return (
        f"Acceptance evidence could not be assembled locally: "
        f"{record.get('failure_kind') or 'error'}: {str(record.get('failure_detail') or '')[:300]}. "
        f"This is host attempt {int(record.get('attempts') or 0)} for this material ({identity}); "
        f"incident {record.get('incident_id')}. This preparation attempt dispatched no new reviewer "
        "and spent no paid review identity; earlier reviewer records, verdicts and costs are retained "
        "unchanged. Repeating the same request over the same material will not rebuild it. "
        "You may finish explicitly with a stated caveat (Advisory) or stop with the work unfinished "
        "(Blocking), or declare an explicit one-use retry through task_acceptance_review's "
        "acceptance_retry when the material changed, the cause was repaired, or the owner asked for one."
    )


def incident_cause_clauses(decision: Dict[str, Any], reason: str, phrases: Dict[str, str]) -> List[str]:
    """Both simultaneous facts of a locally failed acceptance, in the terminal record.

    The host's own evidence assembly failed AND some rail ended the task. Either
    one alone used to replace the other, which is how a record with no new
    reviewer read as "the requested rework never happened" (#1224). The clause
    speaks only for THIS preparation attempt: an earlier real reviewer FAIL or
    rework keeps its own sentence beside it (the primary cause is untouched).
    Present only when the decision carries the typed incident, so every
    historical record reads exactly as it did before; a resolved incident
    states nothing. ``phrases`` is the caller's own cause vocabulary — the
    record's prose SSOT stays with the surface that prints it.
    """
    from ouroboros.outcomes import REASON_FINAL_MESSAGE

    incident = decision.get("acceptance_incident")
    if not isinstance(incident, dict) or not incident or str(incident.get("status") or "") == "resolved":
        return []
    rail = phrases.get(reason, "") if reason not in {"", REASON_FINAL_MESSAGE} else ""
    return [phrases["acceptance_preparation_failed"], rail]


# ── the host behaviour over that state ────────────────────────────────────────
#
# The identity and the record above are facts; what follows is what the host may
# DO with them — disclose the failure once per real attempt, honour an informed
# author decision, and end honestly when neither new material nor an explicit
# retry opened another attempt. None of it records a reviewer, touches a paid
# run, or claims a verdict that does not exist. The owner sees the incident
# through the EXISTING carriers — the review projection the checkpoint
# publishes and the acceptance decision — never through a card mechanism of
# its own; the host progress lines below are ordinary host notes.


def _loop():
    from ouroboros import loop

    return loop


def _review_enforcement_blocks() -> bool:
    from ouroboros.tools.review_helpers import review_enforcement_blocks

    return review_enforcement_blocks(_loop().get_review_enforcement())


def offer_preparation_feedback(ctx: Any, record: Dict[str, Any]) -> bool:
    """Deliver the incident to its author ONCE per real host attempt.

    The repeat key is the incident's identity and attempt count, never the
    exception's text. The carried feedback names the incident AND the attempt,
    so ``expose_acceptance_feedback`` marks exactly that attempt delivered once
    an answering Main request returned.
    """
    from ouroboros import task_pacing
    from ouroboros.outcomes import ACCEPTANCE_REVISION_REQUESTED, REASON_ACCEPTANCE_PREPARATION_FAILED

    previous = ctx.llm_trace.get("acceptance_review_outcome") or {}
    incident_id = str(record.get("incident_id") or "")
    attempts = int(record.get("attempts") or 0)
    if (previous.get("incident_id") == incident_id
            and int(previous.get("incident_attempt") or 0) >= attempts):
        return False
    snapshot = task_pacing.build_budget_snapshot(ctx.tools._ctx, profile=ctx.budget_profile)
    if not task_pacing.improvement_pass_allowed(snapshot, ctx.passes_done, ctx.budget_profile)[0]:
        return False
    binding = str((ctx.review_binding or {}).get("binding_hash") or "")
    ctx.llm_trace["acceptance_review_outcome"] = {
        "binding_hash": binding, "reason": str(record.get("failure_kind") or ""),
        "incident_id": incident_id, "incident_attempt": attempts,
    }
    ctx.messages.append({"role": "user", "content": preparation_feedback_text(record), "review_feedback": [{
        "task_id": ctx.task_id, "outcome_binding_hash": binding,
        "outcome_incident_id": incident_id, "outcome_incident_attempt": attempts}]})
    _loop()._set_acceptance_decision(ctx.llm_trace, {
        "origin": LOCAL_PREPARATION_ORIGIN,
        "status": ACCEPTANCE_REVISION_REQUESTED,
        "reason": REASON_ACCEPTANCE_PREPARATION_FAILED, "source": "task_acceptance_review",
        "acceptance_incident": incident_projection(record)})
    _loop()._end_task_acceptance_fence(ctx.tools._ctx, outcome="revision")
    return True


def record_local_preparation_failure(ctx: Any, exc: BaseException) -> bool:
    """A LOCAL, pre-dispatch assembly failure: the host's own, not a reviewer's.

    No synthetic host_root critic is recorded here — inventing a DEGRADED panel
    for a failure that never reached a reviewer is exactly the fabricated-review
    fact #1224 is about. Prior real, pending and unknown paid runs keep their
    verdicts, costs, identities and custody untouched; the incident lives on the
    existing acceptance trace beside them.
    """
    record = current_incident(ctx.llm_trace) or begin_preparation(ctx.llm_trace, ctx.tools._ctx)
    record_preparation_failure(record, exc, stage=str(getattr(ctx, "stage", "") or STAGE_PREPARATION))
    ctx.emit_progress(
        f"Acceptance evidence could not be assembled locally ({record.get('failure_kind') or 'error'}); "
        f"host attempt {int(record.get('attempts') or 0)} for this material. No new reviewer was "
        "dispatched for this attempt; the work itself is retained.")
    from ouroboros.tools.review_helpers import review_enforcement_blocks

    if not review_enforcement_blocks("blocking"):
        # Cyber's submitted final remains a delivery decision, not a new forced
        # author turn. Record the local gap without inventing a critic panel.
        return finish_exposed_preparation(ctx, record)
    if offer_preparation_feedback(ctx, record):
        return True
    return finish_exposed_preparation(ctx, record)


def finish_exposed_preparation_author(ctx: Any, record: Dict[str, Any]) -> bool:
    """Honor an INFORMED author decision over an exposed preparation failure.

    Owner decision 1 (22.09 23:29): after one meaningful reaction, Advisory may
    finish with a stated caveat and Blocking may stop with the work unfinished,
    without waiting for the owner. The stance binds to the incident's SOURCE
    identity and the ATTEMPT it was informed about — never to a reviewer
    binding it never had, a tool count, the broken builder or a fingerprint.
    The reviewer-bound finish rules (`_finish_advisory_author`) are unchanged.
    """
    from ouroboros.outcomes import ACCEPTANCE_FINALIZED_UNACCEPTED
    from ouroboros.review_records import build_author_disposition

    if str(record.get("status") or "") != STATUS_FAILED or not preparation_exposed(record):
        return False
    stance = ctx.llm_trace.get("acceptance_decision") or {}
    intent = stance.get("agent_finish_intent") or {}
    action = str(intent.get("author_action") or "finish")
    disposition = str(stance.get("agent_disposition") or "")
    if (not intent or disposition not in {"", "accepted", "rejected", "partial", "deferred"}
            or action not in {"finish", "stop"} or not str(stance.get("agent_rationale") or "").strip()):
        return False  # an empty disposition is fine: the explicit act is the stance (C4)
    if (str(intent.get("preparation_identity") or "") != str(record.get("source_identity") or "")
            or intent.get("incident_id") != record.get("incident_id")
            or int(intent.get("incident_attempt") or 0) != int(record.get("attempts") or 0)):
        return False
    if _loop()._task_acceptance_owner_generation_changed(ctx.tools._ctx):
        return False
    blocks = _review_enforcement_blocks()
    if action != "stop" and blocks:
        # Blocking grants no advancement without a verdict: the honest exit is
        # an unfinished stop, which the author states explicitly.
        return False
    attempts = int(record.get("attempts") or 0)
    author = build_author_disposition(
        disposition=disposition, rationale=str(stance.get("agent_rationale") or ""),
        # Bound to the incident and its attempt: no reviewed pack exists, and
        # no fallible subject computation stands between the author and "stop".
        subject_hash=f"{record.get('incident_id')}:attempt-{attempts}",
        reviewer_signal="",  # no reviewer ran for this attempt; claiming one would fabricate the fact
        enforcement="blocking" if blocks else "advisory",
        source="author_informed_acceptance_preparation_failure",
        action=action,  # the record carries the act itself; no stance is invented for it
    )
    ended = _loop()._end_task_acceptance_fence(ctx.tools._ctx, outcome="terminal")
    if ended.status == "refused":
        _loop()._supersede_task_acceptance_for_owner_followup(ctx.tools._ctx, ctx.llm_trace)
        return True
    ctx.tools._ctx._task_acceptance_reviewed = True
    if action == "stop":
        # Like the reviewer-bound stop, this one binds no subject: an earlier panel's
        # must not reopen review over changed material on a later delivery pass; only
        # the author's next decision or owner input does (TZ-2 C4).
        ctx.tools._ctx._task_acceptance_reviewed_subject = ""
    ctx.tools._ctx._task_acceptance_pending = ""
    _loop()._mark_root_acceptance_checkpoint(
        ctx.tools._ctx, ctx.llm_trace, status="preparation_failed", pass_index=ctx.passes_done,
    )
    ctx.llm_trace.setdefault("review_decision", {}).update({
        "author_finish": action == "finish", "admission_released": bool(ended),
        "acceptance_preparation_failed": True,
    })
    _loop()._set_acceptance_decision(ctx.llm_trace, {
        "origin": LOCAL_PREPARATION_ORIGIN,
        "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
        "reason": "author_stop" if action == "stop" else "author_finish",
        "author_action": action, "source": "task_acceptance_review",
        "preparation_delivery_choice": {key: intent.get(key) for key in (
            "candidate_sha256", "owner_source_sha256", "author_action")},
        "author_disposition": author, "reviewer_signal": "",
        "acceptance_incident": incident_projection(record),
        "rationale": (
            "The host could not assemble the acceptance evidence locally; the author was told, "
            "reacted, and "
            + ("stopped with the work unfinished. No reviewer approved anything."
               if action == "stop" else
               "finished the available result explicitly. This is not a reviewer PASS.")
            + " This preparation attempt dispatched no new reviewer; earlier reviewer records are retained."
        ),
    })
    ctx.emit_progress(
        "Author stopped with work unfinished after a local acceptance-evidence failure; "
        "no review approval was granted."
        if action == "stop" else
        "Author finished explicitly after a local acceptance-evidence failure; "
        "the caveat is retained and no reviewer signed this off.")
    return True


def finish_exposed_preparation(ctx: Any, record: Dict[str, Any]) -> bool:
    """End honestly on an exposed, unresolved local preparation failure.

    The available work is delivered with the cause stated; the same broken
    builder is not run again for the same material, and no reviewer rework is
    claimed. A substantive material/criteria change, or one explicit source-bound
    retry, is what opens the next attempt.
    """
    from ouroboros.loop_acceptance_review import _end_acceptance_terminal
    from ouroboros.outcomes import (
        ACCEPTANCE_FINALIZED_UNACCEPTED, REASON_ACCEPTANCE_PREPARATION_FAILED,
    )

    if finish_exposed_preparation_author(ctx, record):
        return not bool(getattr(ctx.tools._ctx, "_task_acceptance_reviewed", False))
    _end_acceptance_terminal(ctx, "preparation_failed")
    _loop()._set_acceptance_decision(ctx.llm_trace, {
        "origin": LOCAL_PREPARATION_ORIGIN,
        "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
        "reason": REASON_ACCEPTANCE_PREPARATION_FAILED,
        "enforcement": "blocking" if _review_enforcement_blocks() else "advisory",
        "source": "task_acceptance_review",
        "acceptance_incident": incident_projection(record),
        "rationale": (
            f"Acceptance evidence could not be assembled locally "
            f"({record.get('failure_kind') or 'error'}) after "
            f"{int(record.get('attempts') or 0)} host attempt(s) on this material. "
            "The current work is retained; this preparation attempt dispatched no new reviewer "
            "and none is claimed. Earlier reviewer records, verdicts and costs are unchanged."
        ),
    })
    ctx.emit_progress(
        "Acceptance evidence could not be assembled locally; the current work is retained "
        "without a new reviewer verdict.")
    return False


def open_local_preparation(ctx: Any) -> Tuple[Dict[str, Any], Any]:
    """Establish this round's material identity BEFORE anything fallible.

    Returns ``(record, handled)``: ``handled`` is the panel's own return value
    when the informed author already decided over an exposed failure, else None.
    """
    ctx.stage = STAGE_PREPARATION
    record = begin_preparation(ctx.llm_trace, ctx.tools._ctx)
    if finish_exposed_preparation_author(ctx, record):
        return record, not bool(getattr(ctx.tools._ctx, "_task_acceptance_reviewed", False))
    return record, None


def refuse_repeat_preparation(ctx: Any, record: Dict[str, Any]):
    """Refuse to rebuild an exposed failure over unchanged material.

    Re-running the same broken builder would only re-deliver the same failure.
    One explicit source-bound retry, or genuinely new material, opens the next
    attempt; a repeated final, a rephrasing or a status question does not.
    """
    if preparation_blocked(record) and not consume_retry(record):
        return finish_exposed_preparation(ctx, record)
    return None


def close_local_preparation(ctx: Any, record: Dict[str, Any]) -> None:
    """The packet and its binding exist: this preparation succeeded.

    The incident closes and its history stays. Binding alone is not dispatch;
    admission and request preparation may still fail before any reviewer. A record
    that never failed closes silently — there was no incident to resolve.
    """
    record_preparation_success(record)
    resolved = incident_projection(record)
    if resolved:
        ctx.emit_progress(
            f"Acceptance evidence assembled after {_attempt_words(resolved)}; "
            "the earlier failure stays in this task's history.")
