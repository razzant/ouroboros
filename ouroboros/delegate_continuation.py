"""Continuation of a settled delegated run: ``delegate_start``'s ``continue_from=<run_id>``.

A delegated run stops for many reasons — its wall-clock cap, a vendor
subscription limit, a crash of the harness or of this host, a cancel, a
question its route could not ask mid-run — and the work it did is worth
finishing. This module is the ONE gate such a start passes, the ONE author of
the facts that bind the new run to its predecessor and the owner of the
same-tree hand-over. HOW the new run remembers the old one belongs to the
engine (``continueFrom``: the same native session, a session moved to another
account, or an evidence packet); the host decides only WHETHER this task may
continue that run and WHAT the continuing child is told.

Four floor invariants, read from durable custody (each refusal is typed and a
definite no-run):

1. own task line — the run is this task's, or this task is its host-confirmed
   automatic retry successor (``delegate_shared.retry_result_status``), or a
   root the owner's Continue created from the run's tree root (the recorded
   ``continued_by`` binding, ``owner_continue.recorded_continuation``); room or
   folder equality grants nothing, and a review panel's run is never a task's;
2. settled — the run has a settled terminal and no successor already continues
   it (one writer at a time);
3. no pending-ambiguous apply — an apply intent without a disposition means
   the tree MAY already carry its patch;
4. authority does not widen — native access is not wider than the run's
   recorded access, and a writing run writes for the target the old run held.

Everything else is a FACT, never a refusal: the cause (any settled cause), an
unread result, the patch disposition, another actor, route or configuration, a
changed or partially verified work order. The facts reach the parent in the
started payload and the continuing child in its PROMPT
(``continuation_prompt``) — never in ``instructions``, which a resumed vendor
session may keep from its first request.

Same tree: while the predecessor's private execution snapshot is undisposed,
the successor runs IN it (same path, same baseline, no new snapshot) and its
own capture (``capture_id``) is the one cumulative patch. Its STARTED row
supersedes the predecessor's capture (``delegate_custody._supersede``); the
claim re-checks the predecessor under the snapshot's one disposition lock, the
lock every apply/reject takes, so a hand-over and a disposition never
interleave, and a superseded predecessor can be neither applied nor rejected.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple

from ouroboros import delegate_custody as custody

log = logging.getLogger(__name__)

# The engine's start-schema key: present in ``GET /v2/agent-capabilities``
# ``runControlKeys`` exactly when the serving engine can continue a run.
ENGINE_CONTINUE_KEY = "continueFrom"
CARRIER_PREFERENCES = ("auto", "packet")

REFUSAL_SOURCE_UNKNOWN = "continuation_source_unknown"
REFUSAL_SOURCE_NOT_OWNED = "continuation_source_not_owned"
REFUSAL_SOURCE_NOT_TERMINAL = "continuation_source_not_terminal"
REFUSAL_SUPERSEDED = "continuation_superseded"
REFUSAL_APPLY_AMBIGUOUS = "continuation_apply_ambiguous"
REFUSAL_AUTHORITY_WIDENED = "continuation_authority_widened"
REFUSAL_TARGET_MISMATCH = "continuation_target_mismatch"
REFUSAL_SNAPSHOT_MISSING = "continuation_snapshot_missing"
REFUSAL_SNAPSHOT_RELEASED = "continuation_snapshot_released"
REFUSAL_ENGINE_UNSUPPORTED = "continuation_engine_unsupported"

_ACCESS_RANK = {"readonly": 0, "workspace_write": 1, "full": 2}
# Advice for the continuing child: one sentence per fact that no longer refuses.
_ADVICE = {
    "result_unread": ("Your previous final result was never read to its end by your supervisor: put "
                      "everything it must know into your final answer of this run."),
    "executor_changed": ("This run may use a different actor, route or configuration than the previous "
                         "one: trust what the workspace and recorded results show over remembered details."),
    "work_order_changed": ("Your supervisor's configured assignment differs from the one the previous run "
                           "was bound to: where they differ, the current assignment and this prompt govern."),
    "work_order_partial": ("The previous run's assignment was delivered only partially and its source ranges "
                           "are not fully verified: ask for any part you need instead of guessing it."),
}


class AdoptedSnapshot(NamedTuple):
    """The predecessor's undisposed execution snapshot, reused as this run's root."""

    snapshot_id: str
    baseline_sha: str
    path: str
    adopted_from: str


class ContinuationStart(NamedTuple):
    """Everything one continuation start contributes beyond an ordinary start."""

    facts: Dict[str, Any] = {}       # parent-facing: the started payload's ``continuation``
    prompt: str = ""                 # wire prompt: host facts + the caller's continuation text
    request: Dict[str, Any] = {}     # wire keys: ``continueFrom`` (+ ``continueCarrier``)
    custody: Dict[str, Any] = {}     # START_REQUESTED/STARTED: lineage + same-tree capture identity
    snapshot: Optional[AdoptedSnapshot] = None


NO_CONTINUATION = ContinuationStart()
CUSTODY_KEYS = ("continuation_of", "capture_id", "snapshot_task_id")


def _task_line(ctx: Any, drive: Any, state: Dict[str, Any], entry: Any) -> str:
    """How this task holds the run: ``own``, ``retry_successor``, ``owner_continue`` or ""."""
    reader = str(getattr(ctx, "task_id", "") or "")
    if not reader or not entry.task_id or entry.review_owned:
        return ""
    if reader == entry.task_id:
        return "own"
    from ouroboros.delegate_shared import retry_result_status

    status, _entry, predecessor = retry_result_status(ctx, drive, entry.run_id, state=state)
    if status == custody.OWNED and predecessor:
        return "retry_successor"
    from ouroboros.owner_continue import recorded_continuation

    return "owner_continue" if recorded_continuation(drive, entry.root_task_id or entry.task_id, reader) else ""


def still_continuable(drive: Any, entry: Any, live_task_ids=None) -> bool:
    """Retain a registration until its owner line and every root's Continue offer end.

    Each root of the recorded Continue chain (the original root and every
    successor it admitted) can be offered Continue after a technical break, so
    the offer is read at every settled member the walk reaches, not only at the
    run's own root; child tasks never carry an offer.

    The sweep supplies its authoritative live/reserved census. Addressed result
    reads validate retry and Continue edges; missing/changed evidence keeps the
    registration. This observation grants no continuation or execution authority.
    """
    if entry.review_owned or not entry.run_id:
        return False
    if live_task_ids is None or not entry.task_id:
        return True
    from types import SimpleNamespace
    from ouroboros.owner_continue import continuation_offer, recorded_continuation
    from ouroboros.task_status import SETTLED_STATUSES, load_effective_task_result
    from supervisor import queue
    from supervisor.queue_transitions import _live_retry_target_locked
    from supervisor.task_ownership import TaskOwnershipRead, prepare_retry_chain

    def resolved(row: Any) -> bool:
        # A timeout-retried task is rewritten `interrupted` naming its retry
        # (task_reaper); its line continues in that retry, followed below.
        return bool(row) and (row.get('status') in SETTLED_STATUSES or (
            row.get('status') == 'interrupted' and bool(row.get('superseded_by'))
            and row.get('superseded_by') == row.get('retry_task_id')))

    live = set(live_task_ids)
    reads = TaskOwnershipRead(drive)
    root = entry.root_task_id or entry.task_id
    pending, seen = [entry.task_id, root], set()
    try:
        local_queue = queue.INITIALIZED and Path(queue.DRIVE_ROOT).resolve() == Path(drive).resolve()
        if local_queue:
            with queue._queue_lock:
                live.update(str(row.get('id') or '') for row in queue.PENDING)
                live.update(queue.ADMISSION_RESERVATIONS)
        while pending:
            task_id = pending.pop()
            if task_id in seen:
                continue
            seen.add(task_id)
            if task_id in live:
                return True
            current = reads.load(task_id)
            if not resolved(current):
                return True
            if local_queue and queue.task_has_live_ownership(task_id, ownership=reads):
                return True
            prepare_retry_chain(queue, task_id, reads.load)
            # The supplied census also covers reserved/recoverable tasks absent
            # from the process's queue maps. Their durable rows bind retry edges.
            census = SimpleNamespace(PENDING=[dict(row, id=tid) for tid, row in reads.rows.items()
                                             if tid in live], RUNNING={},
                                     QUEUE_MAX_RETRIES=queue.QUEUE_MAX_RETRIES)
            with queue._queue_lock:
                leaf, _ = _live_retry_target_locked(census, task_id, results=reads)
            if leaf in live or any(not resolved(row) for row in reads.rows.values()):
                return True
            if leaf != task_id:
                pending.append(leaf)
            claim = current.get('continued_by')
            if claim:
                successor = claim.get('successor_task_id') if isinstance(claim, dict) else ''
                if (not successor or successor in seen
                        or not recorded_continuation(drive, task_id, successor)):
                    return True
                pending.append(successor)
            elif continuation_offer(load_effective_task_result(drive, task_id, materialize_artifacts=False)
                                    or current, task_id)['eligible']:
                return True  # the offer exactly as the task card reads it (a retry leaf's end)
        return not reads.unchanged(reads.rows)
    except Exception:
        log.debug('Registration continuation authority unavailable for %s', entry.run_id, exc_info=True)
        return True


def bind_continuation(ctx: Any, drive: Any, run_id: str, *, actor: Dict[str, Any], route: Any,
                      authority: Any, target_root: str,
                      canonical_work_order_fingerprint: str = "") -> Tuple[Dict[str, Any], str, str]:
    """``(facts, refusal_code, detail)``: the four floors, from durable custody only.

    Nothing here reads the engine, so a daemon that is gone cannot turn an
    unknown run into an admitted one; ``facts`` is complete only when
    ``refusal_code`` is empty.
    """
    rid = str(run_id or "").strip()
    if custody.custody_log_unreadable(drive):
        return {}, REFUSAL_SOURCE_UNKNOWN, (
            f"The custody log cannot be read, so run {rid!r} cannot be proven this task's or settled.")
    state = custody.replay(drive)
    entry = state.get(rid)
    if entry is None:
        return {}, REFUSAL_SOURCE_UNKNOWN, (
            f"No durable custody record names run {rid!r} on this drive; a continuation binds to a "
            "run this task can prove it holds.")
    line = _task_line(ctx, drive, state, entry)
    if not line:
        return {}, REFUSAL_SOURCE_NOT_OWNED, (
            f"Run {rid} belongs to task {entry.task_id or 'unknown'}"
            + (" (a review panel's run)" if entry.review_owned else "")
            + ", and this task is neither its owner, its confirmed retry successor, nor a root the "
            "owner's Continue created from its task tree.")
    if not entry.settled or not entry.terminal_state:
        return {}, REFUSAL_SOURCE_NOT_TERMINAL, (
            f"Run {rid} has no settled terminal on its custody rows (state {entry.terminal_state or 'unsettled'!r}): "
            "it may still be live. Wait on it with delegate_wait or cancel it and verify the receipt; "
            "never start a second writer over a run that may still be running.")
    if entry.superseded_by:
        return {}, REFUSAL_SUPERSEDED, (
            f"Run {rid} was already continued by run {entry.superseded_by}, which holds its snapshot and "
            "work now; continue from that run instead.")
    if entry.patch_apply_pending:
        return {}, REFUSAL_APPLY_AMBIGUOUS, (
            f"Run {rid} has a pending apply intent with no disposition: the tree MAY already carry its patch. "
            "Resolve it through integrate_delegated_patch(acknowledge_ambiguous=true) before continuing.")
    this_access = str(getattr(authority, "access", "") or "")
    if (entry.access not in _ACCESS_RANK or this_access not in _ACCESS_RANK
            or _ACCESS_RANK[this_access] > _ACCESS_RANK[entry.access]):
        return {}, REFUSAL_AUTHORITY_WIDENED, (
            f"Run {rid} ran with access {entry.access or 'unrecorded'!r}; this start derives "
            f"{this_access or 'unrecorded'!r}. A continuation may keep or lower native access, never widen it.")
    if this_access != "readonly" and (not entry.target_root or entry.target_root != str(target_root or "")):
        return {}, REFUSAL_TARGET_MISMATCH, (
            f"Run {rid} wrote for {entry.target_root or 'an unrecorded target'}; this task's present target is "
            f"{target_root or 'unrecorded'}. A writing continuation writes for the target the prior run held.")
    ref = entry.resource_ref if isinstance(entry.resource_ref, dict) else {}
    needs_disposition = bool(entry.snapshot_id or (
        ref.get("workspace_kind") == "directory" and ref.get("strategy") == "copy"))
    advice: List[str] = []
    if custody.settled_output_unread(entry):
        advice.append("result_unread")
    if (not entry.selected_subagent_id or entry.selected_subagent_id != str(actor.get("selected_subagent_id") or "")
            or entry.route_id != str(getattr(route, "route_id", "") or "")
            or entry.config_fingerprint != str(actor.get("config_fingerprint") or "")):
        advice.append("executor_changed")
    canonical = str(canonical_work_order_fingerprint or "")
    if canonical and canonical != entry.work_order_fingerprint:
        advice.append("work_order_changed")
    if str(custody.work_order_source_verification(entry).get("status") or "") == "cannot_verify":
        advice.append("work_order_partial")
    facts: Dict[str, Any] = {
        "continuation_of": rid,
        "cause": entry.terminal_reason or entry.terminal_state,
        "prior_terminal_state": entry.terminal_state,
        "prior_terminal_reason": entry.terminal_reason,
        "prior_owner_task_id": entry.task_id,
        "task_line": line,
        "prior_patch_disposition": entry.patch_disposed or ("undisposed" if needs_disposition else "not_applicable"),
        "prior_target_root": entry.target_root,
        "prior_access": entry.access,
        "prior_baseline_sha": entry.baseline_sha,
        "prior_output": custody.output_disposition(entry),
        "prior_actor": entry.selected_subagent_id,
        "prior_route": entry.route_id,
        "prior_work_order_fingerprint": entry.work_order_fingerprint,
        "advice": advice,
    }
    return facts, "", ""


def _adopted_snapshot(entry: Any, authority: Any) -> Tuple[Optional[AdoptedSnapshot], str, str]:
    """The predecessor's undisposed snapshot a WRITING continuation runs in, or a refusal."""
    from ouroboros.configured_subagents import SESSION_ACCESS_PROFILES

    if (str(getattr(authority, "access", "") or "") not in SESSION_ACCESS_PROFILES
            or not entry.snapshot_id or entry.patch_disposed):
        return None, "", ""
    try:
        from ouroboros.subagent_worktrees import find_execution_snapshot

        record = find_execution_snapshot(entry.snapshot_id) or {}
    except Exception as exc:
        record = {"error": f"{type(exc).__name__}: {exc}"}
    if (not record.get("path") or not entry.execution_root or not entry.baseline_sha
            or not Path(entry.execution_root).exists()):
        return None, REFUSAL_SNAPSHOT_MISSING, (
            f"Run {entry.run_id}'s private snapshot {entry.snapshot_id} is gone or unreadable "
            f"({record.get('error') or 'no registry entry or directory'}), so its work cannot be continued in "
            "the same tree. Apply or reject its captured patch with integrate_delegated_patch, then continue "
            "again: the next continuation starts from a fresh snapshot of the target.")
    return AdoptedSnapshot(entry.snapshot_id, entry.baseline_sha, entry.execution_root, entry.run_id), "", ""


def continuation_prompt(facts: Dict[str, Any], caller_text: str) -> str:
    """The continuing child's prompt: carrier-neutral host facts, then the caller's text.

    The engine puts its own notice (which carrier, what stopped) in front; this
    block states only what the host knows about the tree and the old result.
    """
    rid, workspace = facts.get("continuation_of"), str(facts.get("workspace") or "")
    disposition = str(facts.get("prior_patch_disposition") or "")
    if workspace == "same_snapshot":
        tree = ("You continue in the SAME private snapshot the previous run worked in (same path, same "
                "baseline): everything it changed there is still there, and one cumulative patch of the whole "
                "work is captured when you finish.")
    elif disposition == "applied":
        tree = ("Its captured changes were APPLIED to the target, so the tree you start from contains them: "
                "do not redo or re-apply that work.")
    elif disposition == "rejected":
        tree = "Its captured changes were REJECTED: the tree you start from does NOT contain them."
    elif disposition == "undisposed":
        tree = ("Its captured changes are NOT in the tree you start from: they wait for your supervisor's "
                "decision, so do not assume them.")
    elif str(facts.get("prior_access") or "readonly") == "readonly":
        tree = "It was read-only: it changed no files."
    else:
        tree = ("It wrote DIRECTLY into the target: the tree you start from already contains whatever it "
                "changed. Do not redo or re-apply that work.")
    reason = facts.get("prior_terminal_reason")
    lines = [
        f"HOST FACTS FOR THIS CONTINUATION OF RUN {rid}: that run settled {facts.get('prior_terminal_state')}"
        + (f" (reason {reason})" if reason else "") + f" and is not running. {tree}",
        "Check the workspace as it is NOW before repeating a step; a step that was cut off may or may not "
        "have taken effect.",
        *(_ADVICE[code] for code in facts.get("advice") or () if code in _ADVICE),
    ]
    text = "\n".join(lines)
    note = str(caller_text or "").strip()
    return f"{text}\n\n{note}" if note else text


def engine_continues(gateway: Any) -> bool:
    """Does the serving engine accept ``continueFrom`` (its run-start schema lists it)?"""
    keys = (gateway.agent_capabilities() or {}).get("runControlKeys")
    return isinstance(keys, list) and ENGINE_CONTINUE_KEY in keys


def start_binding(ctx: Any, drive: Any, token: str, *, gateway: Any, actor: Dict[str, Any], route: Any,
                  authority: Any, target_root: str, invocation_id: str, text: str, source_binding: Dict[str, Any],
                  coordination_context: str = "", carrier: Any = None,
                  canonical_work_order_fingerprint: str = "") -> Tuple[ContinuationStart, Optional[Any]]:
    """``(continuation, refusal)`` for ONE start: the floors, the engine bit, the tree.

    Decided before any snapshot or registration exists; a refusal is a definite
    no-run recorded as a start-blocked evidence row. The caller's continuation
    text is the configured session's coordination note, else the start prompt —
    never the canonical work order, which the old session (or the engine's
    evidence packet) already holds.
    """
    from ouroboros.delegate_evidence import record_start_blocked
    from ouroboros.delegate_shared import _fail

    facts, code, detail = bind_continuation(
        ctx, drive, token, actor=actor, route=route, authority=authority, target_root=target_root,
        canonical_work_order_fingerprint=canonical_work_order_fingerprint)
    adopted = None
    if not code:
        adopted, code, detail = _adopted_snapshot(custody.replay(drive)[token], authority)
    if not code and not engine_continues(gateway):
        code, detail = REFUSAL_ENGINE_UNSUPPORTED, (
            f"The serving Claudexor engine {getattr(gateway, 'engine_version', '') or '(unknown version)'} does "
            "not accept continueFrom, so the stopped run cannot be continued here. Start a plain new run whose "
            "prompt carries the remaining work, or continue after the engine is updated.")
    if code:
        record_start_blocked(ctx, str(getattr(ctx, "task_id", "") or ""), code)
        return NO_CONTINUATION, _fail("delegate_start", code, detail, continue_from=token, definitely_unrun=True)
    entry = custody.replay(drive)[token]
    preference = str(carrier or "")
    facts.update(workspace="same_snapshot" if adopted else "fresh", carrier_preference=preference or "auto")
    record = {"continuation_of": token}
    if adopted:
        from ouroboros.delegate_source_coverage import inherit_source_obligations

        inherit_source_obligations(entry, source_binding)
        record.update(capture_id=invocation_id, snapshot_task_id=entry.snapshot_task_id or entry.task_id)
    caller = coordination_context if bool(actor.get("compiled_work_order")) else text
    return ContinuationStart(
        facts=facts, prompt=continuation_prompt(facts, caller),
        request={"continueFrom": token, **({"continueCarrier": preference} if preference else {})},
        custody=record, snapshot=adopted), None


def replayed_custody(record: Dict[str, Any]) -> ContinuationStart:
    """A retry replays its recorded lineage; its body already carries the wire keys."""
    return ContinuationStart(custody={key: str(record.get(key) or "") for key in CUSTODY_KEYS})


def retry_matches_caller_text(record: Dict[str, Any], text: str) -> bool:
    """A continuation's recorded prompt is host facts plus the caller's text, so its
    retry names the caller's own text, whose digest the start recorded as its brief."""
    from hashlib import sha256

    return bool(record.get("continuation_of")) and str(record.get("work_order_fingerprint") or "") == sha256(
        str(text).encode("utf-8")).hexdigest()


def _handover_refusal(drive: Any, row: Dict[str, Any]) -> Dict[str, Any]:
    """Re-check the predecessor at the claim, under the snapshot's disposition lock."""
    rid, invocation = str(row.get("continuation_of") or ""), str(row.get("invocation_id") or "")
    prior = custody.replay(drive).get(rid)
    if prior is None or not prior.settled:
        return {"reason": REFUSAL_SOURCE_NOT_TERMINAL, "detail": f"Run {rid} is no longer a settled run."}
    if prior.superseded_by:
        return {"reason": REFUSAL_SUPERSEDED, "head": prior.superseded_by,
                "detail": f"Run {rid} was continued by run {prior.superseded_by} meanwhile; continue from it."}
    if prior.patch_disposed or prior.snapshot_id != str(row.get("snapshot_id") or ""):
        return {"reason": REFUSAL_SNAPSHOT_RELEASED, "detail": (
            f"Run {rid}'s snapshot was {prior.patch_disposed or 'rebound'} meanwhile; continue again "
            "(a fresh snapshot of the target is used once its patch is disposed).")}
    if prior.patch_apply_pending:
        return {"reason": REFUSAL_APPLY_AMBIGUOUS, "detail": f"Run {rid} has a pending apply intent."}
    rival = next((str(item.get("invocation_id") or "") for item in custody.pending_invocations(drive)
                  if str(item.get("continuation_of") or "") == rid and item.get("capture_id")
                  and str(item.get("invocation_id") or "") != invocation), "")
    if rival:
        return {"reason": REFUSAL_SUPERSEDED, "pending_invocation_id": rival,
                "detail": f"Another continuation of run {rid} ({rival}) is already pending."}
    return {}


@contextmanager
def snapshot_handover(drive: Any, row: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Hold the snapshot's ONE disposition lock while the hand-over is re-checked
    and its START_REQUESTED row written; yields the refusal ({} = proceed)."""
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock

    prior = custody.replay(drive).get(str(row.get("continuation_of") or ""))
    if prior is None:
        yield {"reason": REFUSAL_SOURCE_UNKNOWN, "detail": "The continued run has no custody record."}
        return
    lock_path = custody.disposition_lock_path(drive, prior)
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = acquire_exclusive_file_lock(lock_path, timeout_sec=20.0, owner_aware_stale=True)
    except Exception as exc:
        fd, busy = None, f" ({type(exc).__name__}: {exc})"
    else:
        busy = ""
    if fd is None:
        yield {"reason": "continuation_handover_busy",
               "detail": f"The continued run's snapshot lock is held or unavailable{busy}; retry after it completes."}
        return
    try:
        yield _handover_refusal(drive, row)
    finally:
        release_exclusive_file_lock(lock_path, fd)


def disposition_refusal(drive: Any, entry: Any) -> str:
    """Why a run's captured patch may not be applied or rejected because of a hand-over."""
    if entry is None:
        return ""
    if entry.superseded_by:
        return (f"⚠️ INTEGRATE_DELEGATED_SUPERSEDED: run {entry.run_id} was continued by run "
                f"{entry.superseded_by} in the same snapshot; its work is part of that run's cumulative "
                "capture. Dispose that run instead; nothing was changed.")
    pending = next((str(item.get("invocation_id") or "") for item in custody.pending_invocations(drive)
                    if str(item.get("continuation_of") or "") == entry.run_id and item.get("capture_id")), "")
    if pending:
        return (f"⚠️ INTEGRATE_DELEGATED_CONTINUATION_PENDING: a continuation of run {entry.run_id} "
                f"(invocation {pending}) may be running in its snapshot. Wait for that start to settle "
                "(or retry it with retry_of) before disposing; nothing was changed.")
    return ""


def terminal_continuity(run_id: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    """The engine's continuation facts for a terminal payload: its ``resumable``
    block as sent, and one line per continued try. An older engine reports
    neither, and the payload then carries neither."""
    facts: Dict[str, Any] = {}
    if isinstance(summary.get("resumable"), dict):
        facts["resumable"] = summary["resumable"]
    lines = [_receipt_line(run_id, row) for row in summary.get("continuity") or () if isinstance(row, dict)]
    if lines:
        facts["continuity"] = lines
    return facts


def _receipt_line(run_id: str, row: Dict[str, Any]) -> str:
    """One continued try: carrier, accounts, memory, attested model, and every non-default flag."""
    source = row.get("from") if isinstance(row.get("from"), dict) else {}
    target = row.get("to") if isinstance(row.get("to"), dict) else {}
    parts = [f"try {row.get('tryIndex')} ({row.get('attemptId') or '?'}): {row.get('carrier') or '?'} "
             f"after {row.get('cause') or '?'}"
             + (f" of run {source.get('runId')}" if source.get("runId") and source.get("runId") != run_id else ""),
             f"profile {source.get('profileId') or '?'} -> {target.get('profileId') or '?'}",
             f"memory {row.get('memory') or 'unknown'}",
             f"model {row.get('observedModel') or 'unattested'}"]
    if row.get("modelMismatch"):
        parts.append("MODEL MISMATCH")
    if row.get("workspace") == "different_root":
        parts.append("different root")
    if row.get("identityCheck") not in (None, "matched_before_effects", "not_applicable"):
        parts.append(f"identity {row.get('identityCheck')}")
    if row.get("inputDelivery") == "uncertain":
        parts.append("last input may not have been delivered")
    if row.get("instructions") == "vendor_snapshot":
        parts.append("instructions: vendor snapshot")
    if isinstance(row.get("reingestedTokens"), int):
        parts.append(f"re-read {row['reingestedTokens']} tokens")
    return "; ".join(parts)


__all__ = [
    "AdoptedSnapshot",
    "ContinuationStart",
    "NO_CONTINUATION",
    "bind_continuation",
    "continuation_prompt",
    "disposition_refusal",
    "engine_continues",
    "replayed_custody",
    "retry_matches_caller_text",
    "snapshot_handover",
    "start_binding",
    "terminal_continuity",
]
