"""Cancellation settlement PUBLICATION: typed outcomes, result fields, delivery.

Split from ``supervisor.task_lifecycle`` by the module-size gate — a code
boundary only, exactly like ``supervisor.queue_transitions``: the queue module
object (``q``) is always passed in by the caller, so ``supervisor.queue``
remains the single state authority and every mutation still runs under its
existing process lock. ``task_lifecycle`` re-imports every name here, so
callers, tests, and ``supervisor.queue``'s public re-exports keep ONE import
surface.

Owns the pieces of a cancel that happen AFTER custody decided the outcome:
the typed outcome vocabulary, the artifact-honest cancelled result fields,
the cost reconstruction adapter, the salvage adapter, the owed-before-settle
outbox registration (GR2-4), the publication of the stored terminal truth,
and the miss-lane terminalization/delivery adapter.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, List, Optional

from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)


# Typed per-id cancellation outcome (v6.82.0). A boolean cannot distinguish
# "cancelled", "it had already finished on its own" and "refused/failed" — and a
# cascade that OR-aggregates booleans reports success while a child is still live.
CANCEL_CANCELLED = "cancelled"
CANCEL_ALREADY_SETTLED = "already_settled"
CANCEL_NOT_FOUND = "not_found"
CANCEL_FAILED = "failed"
_CANCEL_TERMINALIZED = frozenset({CANCEL_CANCELLED, CANCEL_ALREADY_SETTLED, CANCEL_NOT_FOUND})


# Transport is recorded fact, never proof that an HTTP caller was the owner.
# The browser twin is ``web/modules/cancel_presentation.js``; both
# tables must name the same sources, because one stored ``cancel_origin`` is
# rendered by the task card AND by this host's durable terminal rows.
CANCEL_SOURCE_PHRASES = {
    "server_shutdown": "Server shutdown",
    "http_single": "Stopped from the app (Stop now)",
    "http_cascade": "Stopped from the app (Stop now)",
    "http_graceful": "Stopped from the app (Wrap up)",
    # ``_cancel_subtree_sweep`` mints this for every captured descendant.
    "cascade_descendant": "Stopped with the task tree it belongs to",
    # ``server_restart._stop_owned_work``: the /restart command (chat or app button).
    "owner_restart": "Stopped by the Restart command",
    # ``queue_snapshot._fence_snapshot_running_rows``: a row still RUNNING when the
    # previous server stopped (``shutdown_cancel_text`` states it on the result).
    "snapshot_restore": "The server stopped while this task was still running",
    # ``join_ledger`` cancel_task, Ouroboros's own tool; the asking run is below.
    "agent_tool": "Stopped by Ouroboros",
}
# The fixed label a producer writes as ``reason`` restates its phrase, so that
# exact label is not repeated beside it; any other reason text still shows.
CANCEL_SOURCE_LABELS = {
    "http_graceful": "owner requested finalize-then-stop",
    "owner_restart": "Owner restart",
    "snapshot_restore": "server_shutdown",
}
# A recorded cause is producer free text; bound it before it joins an owner line.
# The bound is a PREVIEW of a reason the record keeps whole, and says so.
CANCEL_REASON_MAX_CHARS = 160
CANCEL_REASON_PREVIEW_NOTE = " (preview; the full reason is kept with the task)"


def cancel_cause_clauses(
    origin: Dict[str, Any], result: Dict[str, Any], event: Dict[str, Any],
) -> List[str]:
    """The clauses a recorded ``cancel_origin`` PROVES, for one owner line.

    Written here, beside ``_intent_outcome_fields`` which records the origin, so
    the producer and the sentence share a home. The caller joins them with its
    own separator; the browser twin is the cancelled branch of
    ``log_events.js::taskReasonDetail`` and must stay word for word identical.

    ``requested_by`` says which run ASKED — a cascade stamps the SWEPT ROOT
    there — so reading it as the initiator named a task that somebody else had
    stopped (#1061). When this record's OWN lineage proves that asker is its
    parent or ancestor, the line states that relation. Only a typed
    ``request_origin`` proves an actor; ``requested_by`` alone never does.
    Ouroboros's own cancel_task stamps the run that asked, not a swept root,
    so there the relation is the asker's, and it names that same actor.
    """
    from ouroboros.utils import strip_markdown

    request_origin = origin.get("request_origin")
    actor = (
        str(request_origin.get("task_id") or "")
        if isinstance(request_origin, dict) and request_origin.get("kind") == "agent_task"
        else ""
    )
    source = str(origin.get("source") or "")
    asker = source == "agent_tool"
    asked = str(origin.get("requested_by") or "")
    record = {**event, **result}
    self_id = str(record.get("task_id") or record.get("id") or record.get("subagent_task_id") or "")
    parent = str(record.get("parent_task_id") or "")
    root = str(record.get("root_task_id") or "")
    relation = ""
    if asked and asked != self_id:
        if asked == parent:
            relation = "Requested by its parent task" if asker else "Stopped with its parent task"
        elif parent and asked == root:
            relation = "Requested by an ancestor task" if asker else "Stopped with an ancestor task"
    stated = str(origin.get("reason") or "")
    reason = ("" if " ".join(stated.split()) == CANCEL_SOURCE_LABELS.get(source)
              else " ".join(strip_markdown(stated).split()))
    if len(reason) > CANCEL_REASON_MAX_CHARS:
        reason = reason[:CANCEL_REASON_MAX_CHARS - 1].rstrip() + "\u2026" + CANCEL_REASON_PREVIEW_NOTE
    return [
        CANCEL_SOURCE_PHRASES.get(source, source),
        reason,
        "this task and its sub-tasks" if origin.get("scope") == "cascade" else "",
        relation,
        "Requested by a task" if actor and not (asker and relation and actor == asked) else "",
    ]


def _load_result_row(q: Any, task_id: str) -> Dict[str, Any]:
    """The durable result row, or ``{}`` — fail-soft."""
    try:
        from ouroboros.task_results import load_task_result

        return load_task_result(q.DRIVE_ROOT, task_id) or {}
    except Exception:
        log.debug("durable result read failed for %s", task_id, exc_info=True)
        return {}


def _is_workspace_task_record(record: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(record, dict):
        return False
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    return bool(str(record.get("workspace_root") or "").strip() or str(metadata.get("workspace_root") or "").strip())


def queued_address(task: Any, existing: Any) -> Dict[str, Any]:
    """A never-started task's queue address for its first result row; a recorded one stays."""
    chat = task.get("chat_id") if isinstance(task, dict) else None
    return {"chat_id": chat} if chat is not None and (existing or {}).get("chat_id") is None else {}


def _cancel_result_fields(
    task: Optional[Dict[str, Any]],
    *,
    existing: Optional[Dict[str, Any]] = None,
    result: str,
    artifact_capture: str = "never_started",
    **fields: Any,
) -> Dict[str, Any]:
    """Terminal-cancel result fields, artifact-honest per A4 (owner batch-1 9=A).

    Workspace artifact truth on a cancel comes from the REAL tree: the running
    kill path runs ``finalize_task_artifacts`` first (``artifact_capture=
    "attempted"``) and any terminal artifact status it recorded is PRESERVED
    here, never overwritten with a blanket "missing". Only when no capture
    evidence exists does the stamp depend on how far the task got: a task that
    never started honestly has ``missing`` artifacts, while an attempted capture
    that left nothing terminal is a capture FAILURE. Cancel-path captures of a
    shared workspace tree carry ``artifact_attribution: "shared_unproven"`` —
    the tree was not quiesced through the normal patch-capture seam, so the diff
    cannot prove which actor authored it (until the phase-C run isolation).

    A task cancelled before it ever ran has no result row yet, so its queue address
    is kept: the first terminal publication may run before ``task_done`` and would
    otherwise read no room at all (an existing address, hidden 0 included, stays).
    """
    payload: Dict[str, Any] = {**fields, "result": result, **queued_address(task, existing)}
    if not (_is_workspace_task_record(task) or _is_workspace_task_record(existing)):
        return payload
    try:
        from ouroboros.headless import ARTIFACT_STATUS_FAILED, ARTIFACT_STATUS_MISSING
        from ouroboros.outcomes import artifact_bundle_from_result, normalize_outcome_axes
        from ouroboros.task_status import ARTIFACT_TERMINAL_STATUSES

        existing_artifact_status = str(
            (existing or {}).get("artifact_status") or ""
        ).strip().lower()
        if existing_artifact_status in ARTIFACT_TERMINAL_STATUSES:
            # A4: a real capture already recorded its terminal truth (patch,
            # no-changes, or failed) — keep it; only annotate attribution.
            payload["artifact_attribution"] = "shared_unproven"
            return payload
        if str(artifact_capture or "") == "attempted":
            capture_status = ARTIFACT_STATUS_FAILED
            capture_error = "Workspace artifact capture did not complete during cancellation teardown."
        elif str(artifact_capture or "") == "owed_no_result":
            # A4: the kill caught a RUNNING workspace task before its durable row
            # existed, so the owed capture could not run safely — a capture
            # FAILURE (the tree may hold uncaptured work), never ``missing``.
            capture_status = ARTIFACT_STATUS_FAILED
            capture_error = (
                "Workspace artifact capture was owed but could not run: the task was "
                "killed before its durable result row existed."
            )
        else:
            capture_status = ARTIFACT_STATUS_MISSING
            capture_error = "Task cancelled before workspace patch finalization."
        base: Dict[str, Any] = {}
        if isinstance(existing, dict):
            base.update(existing)
        if isinstance(task, dict):
            base.update(task)
        payload["artifact_status"] = capture_status
        payload.setdefault("artifact_error", capture_error)
        payload["artifact_attribution"] = "shared_unproven"
        base.update(payload)
        base["status"] = "cancelled"
        base["artifact_status"] = capture_status
        base.pop("artifact_bundle", None)
        bundle = artifact_bundle_from_result(base)
        payload["artifact_bundle"] = bundle
        axes = normalize_outcome_axes(base)
        artifact_axis = dict(axes.get("artifacts") or {})
        artifact_axis["status"] = capture_status
        axes["artifacts"] = artifact_axis
        payload["outcome_axes"] = axes
    except Exception:
        log.debug("Failed to build cancelled artifact fields for task %s", (task or existing or {}).get("id") or (task or existing or {}).get("task_id"), exc_info=True)
    return payload


def shutdown_cancel_text(intent: Dict[str, Any]) -> str:
    """The owner sentence for a fence the snapshot restore minted, else "".

    ONE producer for every lane that settles such a fence. The cause is a
    property of the INTENT, not of the lane that happened to reach it: the task
    was running when the previous server stopped. At an ordinary boot the pool is
    empty when restore runs, so the fence settles through the miss lane below; a
    worker that outlived SIGTERM is claimed and killed instead, and the owner has
    to read the same cause either way. Every other intent returns the empty
    string, so each lane keeps its own default sentence.
    """
    row = intent if isinstance(intent, dict) else {}
    if (
        str(row.get("reason") or "") == "server_shutdown"
        and str(row.get("source") or "") == "snapshot_restore"
    ):
        return "Task cancelled: the server stopped while this task was still running."
    return ""


def stop_notice_outcome(origin: Any) -> str:
    """The cancel receipt's outcome words: a restore fence names its cause.

    The owner did not ask for this stop, so "was stopped" alone reads like one.
    The sentence is the card's own phrase for the same stored ``cancel_origin``.
    """
    return f"stopped. {CANCEL_SOURCE_PHRASES['snapshot_restore']}" if shutdown_cancel_text(origin) else "stopped"


def _miss_lane_cancel_text(intent: Dict[str, Any]) -> str:
    """The owner sentence for a cancel settled with nothing queued or running.

    That is the ordinary shape of this lane and its default says so. A restore
    fence is the one case where the same emptiness has a KNOWN cause, and saying
    "was neither queued nor running" there would contradict the boot line and
    strand the lost quiz without the cause its expiry carries.
    """
    return shutdown_cancel_text(intent) or (
        "Task cancelled (was neither queued nor running at supervisor teardown)."
    )


def _intent_outcome_fields(intent: Dict[str, Any]) -> Dict[str, Any]:
    """``parent_decision`` written only at OUTCOME (phase A): a parent-requested
    cancel stamps its decision on the SETTLED cancelled result, never at intent
    time — so a child that finished first keeps a decision-free completed record."""
    if not isinstance(intent, dict):
        return {}
    fields = {"cancel_observation": dict(intent["observation"])} if isinstance(intent.get("observation"), dict) else {}
    # Preserve the recorded cause after the active intent is removed. Actor
    # evidence is independent of requested_by, which also drives parent decisions.
    origin = {key: str(intent[key]) for key in (
        "source", "requested_by", "scope", "reason", "requested_at", "request_id",
    ) if intent.get(key)}
    request_origin = (fields.get("cancel_observation") or {}).get("request_origin")
    if isinstance(request_origin, dict):
        origin["request_origin"] = dict(request_origin)
    if origin:
        fields["cancel_origin"] = origin
    if not intent.get("requested_by"):
        return fields
    fields["parent_decision"] = "cancelled"
    if intent.get("reason"):
        fields["parent_decision_reason"] = str(intent.get("reason") or "")
    return fields


def _reconstructed_cost_fields(q: Any, task_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
    """Cost authority for a supervisor-settled cancel: the physical-attempt
    ledger (a never-dispatched task reconstructs to a CONFIRMED zero), degrading
    to an honest nullable unknown — never a fabricated final $0 (owner 10=B)."""
    try:
        return q.reconstruct_task_cost(
            str(task_id), fields=True,
            drive_root=pathlib.Path((task or {}).get("budget_drive_root") or q.DRIVE_ROOT),
        )
    except Exception:
        log.warning("Cost reconstruction failed for cancelled %s", task_id, exc_info=True)
        return {"cost_accounting_status": "unavailable", "cost_final": False,
                "cost_accounting_error": "ledger_unavailable",
                # ABI-3: honest name only — the retired alias is read-only.
                "accounted_upper_bound_usd": None}


def _salvage_cancelled_output(
    q: Any, task: Dict[str, Any], task_id: str,
) -> tuple[str, str, str]:
    """(result-suffix note, full salvaged text, preserved-copy path) — fail-soft.

    Thin adapter over the shared delivery-seam salvage (terminal_delivery owns
    the mechanics; the reaper shares the same underlying helpers)."""
    from supervisor.terminal_delivery import salvage_cancelled_output

    return salvage_cancelled_output(
        pathlib.Path(q.DRIVE_ROOT), q._task_drive_for_task(task, str(task_id)), str(task_id),
    )


def _register_owed_terminal_delivery(
    q: Any, task: Dict[str, Any], task_id: str, stored: Dict[str, Any],
    *, deliver: bool, salvage_text: str = "", salvage_path: str = "",
    unreconciled_runs: Optional[List[str]] = None,
) -> bool:
    """GR2-4: durably register the terminal answer as OWED before the settle.

    Builds the exact event the publish step will enqueue (the build is
    deterministic, so both carry the same content-hash ``delivery_id``) and
    registers it in the durable outbox. When a delivery is expected but cannot
    be routed (no lineage chat), a typed handoff row records that decision —
    the crash-order evidence must show the answer was consciously not owed,
    not silently dropped. Cascade-suppressed (``deliver=False``) and subagent
    outcomes are owed elsewhere (the root summary / the parent handoff).

    Returns whether the answer is DURABLY accounted for (GR3-4): owed in the
    outbox, already delivered/owed, legitimately not deliverable here
    (suppressed / subagent / non-delivering status), or consciously handed
    off. ``False`` means a real durability gap — the registration or the
    handoff record could not be written — and the caller must leave the
    cancel intent OPEN instead of settling over an unowed answer. Never
    raises and never gates the teardown itself.
    """
    try:
        from supervisor.terminal_delivery import (
            build_completed_result_event,
            build_unreviewed_salvage_event,
            register_pending_delivery,
        )

        from ouroboros.task_results import STATUS_CANCELLED, STATUS_COMPLETED

        if not deliver or str((task or {}).get("delegation_role") or "") == "subagent":
            return True
        stored_status = str((stored or {}).get("status") or "")
        if stored_status and stored_status != STATUS_CANCELLED:
            # The worker's own terminal row won over the Stop: the provisional
            # salvage owed before the terminal write is not this task's answer.
            from supervisor.terminal_delivery import withdraw_stop_episode

            withdraw_stop_episode(pathlib.Path(q.DRIVE_ROOT), task_id)
        event: Optional[Dict[str, Any]] = None
        if stored_status == STATUS_CANCELLED:
            event = build_unreviewed_salvage_event(
                pathlib.Path(q.DRIVE_ROOT), task, task_id,
                outcome=stop_notice_outcome((stored or {}).get("cancel_origin")),
                salvaged_text=salvage_text, preserved_path=salvage_path,
                unreconciled_runs=list(unreconciled_runs or []),
                settled_status=stored_status,
            )
        elif stored_status == STATUS_COMPLETED:
            # The disclosure rides the completed answer's custody row, and the
            # owed registration and the publish half must build the SAME rows
            # (one delivery id each) — both pass the identical list.
            event = build_completed_result_event(
                pathlib.Path(q.DRIVE_ROOT), task, task_id, stored,
                unreconciled_runs=list(unreconciled_runs or []),
            )
        else:
            # failed/rejected outcomes keep their own paths' delivery.
            return True
        if event is not None:
            return register_pending_delivery(pathlib.Path(q.DRIVE_ROOT), event)
        return bool(q.append_jsonl(
            pathlib.Path(q.DRIVE_ROOT) / "logs" / "supervisor.jsonl",
            {"ts": utc_now_iso(), "type": "terminal_delivery_handoff",
             "task_id": task_id, "settled_status": stored_status,
             "reason": "no_lineage_chat"},
        ))
    except Exception:
        log.warning("owed terminal-delivery registration failed for %s", task_id, exc_info=True)
        return False


def _settle_or_reopen_intent(
    q: Any, task_id: str, *, owed_ok: bool,
    intent: Optional[Dict[str, Any]], outcome: str, detail: str,
) -> None:
    """GR4-1, the ONE registration-failure rule for every settle site: an
    answer that could not be durably owed leaves the intent OPEN — the fenced
    claim is released for the watchdog, whose re-feed re-attempts the
    registration (loud via the typed ``terminal_delivery_unregistered`` event
    each tick) — instead of settling over an unowed answer. The teardown and
    publication themselves are never gated."""
    from supervisor.task_lifecycle import _release_intent_claim, _settle_intent

    if not owed_ok and intent and intent.get("request_id"):
        _release_intent_claim(
            q, task_id,
            error="owed terminal-delivery registration failed", intent=intent,
        )
    else:
        _settle_intent(q, task_id, outcome=outcome, detail=detail, intent=intent or None)


def _publish_cancelled_task(
    q: Any, task_id: str, task: Dict[str, Any], worker: Any,
    stored: Dict[str, Any], cost_fields: Dict[str, Any],
    *, deliver: bool = True, salvage_text: str = "", salvage_path: str = "",
    unreconciled_runs: Optional[List[str]] = None,
) -> str:
    """Publish the STORED terminal truth and reconcile the worker slot.

    The event carries whatever actually settled: if the worker wrote its own
    natural result before we killed it, the monotonic guard refused our
    cancellation — publishing nothing would leave that card unresolved until a
    reload, so the stored status is emitted instead.
    """
    from supervisor import workers

    from ouroboros.task_results import STATUS_CANCELLED
    from ouroboros.headless import terminal_task_files_ready

    if not terminal_task_files_ready(q.DRIVE_ROOT, {**stored, **task, "id": task_id}, stored):
        return CANCEL_FAILED  # Publication/cleanup cannot outrun terminal source custody.

    settled_status = str((stored or {}).get("status") or STATUS_CANCELLED)
    # The row leaves RUNNING only NOW — death confirmed, terminal result durable.
    # A natural task_done that raced us may have consumed the row already; that
    # handler also emitted the terminal event, so whoever pops the row owns the
    # emit and the card resolves exactly once.
    with q._queue_lock:
        row_owned = q.RUNNING.pop(task_id, None) is not None
    try:
        from ouroboros.tools.services import archive_task_service_logs
        archive_task_service_logs(pathlib.Path(q.DRIVE_ROOT), str(task_id), task)
    except Exception:
        log.debug("Failed to archive service logs for cancelled task %s", task_id, exc_info=True)
    # A2 delivery: the salvaged answer reaches chat through the shared durable
    # outbox seam (loud UNREVIEWED banner) — the incident's finished report sat
    # on disk 30 seconds from the owner. Non-subagent tasks only (a child's
    # result flows to its parent through the ordinary handoff); a completed
    # natural result is delivered as itself. Fail-soft, idempotent (the owed
    # row was registered BEFORE the intent settled — GR2-4 — and the shared
    # delivery_id dedupes the enqueue here).
    if deliver and str(task.get("delegation_role") or "") != "subagent":
        try:
            from supervisor.terminal_delivery import deliver_unreviewed_salvage

            if settled_status == STATUS_CANCELLED:
                deliver_unreviewed_salvage(
                    pathlib.Path(q.DRIVE_ROOT), task, task_id,
                    outcome=stop_notice_outcome((stored or {}).get("cancel_origin")),
                    salvaged_text=salvage_text, preserved_path=salvage_path,
                    unreconciled_runs=list(unreconciled_runs or []),
                    settled_status=settled_status,
                )
            elif settled_status == "completed":
                # Completion-wins: the worker died before its own final delivery
                # could be confirmed; ship the completed result through the same
                # deduped seam (a copy the worker already delivered is suppressed
                # by the shared delivery id).
                from supervisor.terminal_delivery import deliver_completed_result

                deliver_completed_result(
                    pathlib.Path(q.DRIVE_ROOT), task, task_id, stored,
                    unreconciled_runs=list(unreconciled_runs or []),
                )
        except Exception:
            log.debug("Terminal salvage delivery failed for %s", task_id, exc_info=True)
    if row_owned:
        try:
            q._emit_cancel_task_done(task, task_id, cost_fields=cost_fields, status=settled_status)
        except Exception:
            log.warning("Failed to publish terminal event for %s", task_id, exc_info=True)
    # Respawn recovery is the REAPER'S canonical step 5, not a private variant.
    # The helper serializes against shutdown with the lifecycle lock and starts
    # the child outside the queue lock; on failure the marker is cleared under
    # the lock so the crash detector can recover the slot on a later tick.
    from supervisor.task_reaper import _respawn_after_reap
    _respawn_after_reap(q, workers, worker.wid, expected_worker=worker)
    # A cancelled subagent's drive is NOT settled here: settlement copies and hashes the
    # child store, which the cancel path must not carry. The off-loop reconcile pass
    # settles it without waiting out retention (``headless.prune_headless_task_drives``).
    try:
        q.persist_queue_snapshot(reason="cancel_running")
    except Exception:
        log.debug("Snapshot after cancelling %s failed", task_id, exc_info=True)
    return CANCEL_ALREADY_SETTLED if settled_status != STATUS_CANCELLED else CANCEL_CANCELLED


def _deliver_on_miss(
    q: Any, task_id: str, row: Dict[str, Any], status: str,
    *, unreconciled_runs: Optional[List[str]] = None,
) -> bool:
    """Owner 5=A on the miss lane — the routing logic lives in the shared
    delivery seam (``deliver_miss_lane_outcome``); this adapter only supplies
    the queue-owned drive facts. Fail-soft.

    ``unreconciled_runs`` (GR5-3) threads the delegated-custody audit result
    into the miss-lane message, so the owner disclosure matches the kill path.

    Returns the seam's durable-accounting verdict (GR4-1): ``False`` means the
    owed registration failed and the caller must leave the cancel intent OPEN
    (claim released) instead of settling over an unowed answer. A truthy
    return from a monkeypatched/legacy delivery counts as accounted."""
    row = row if isinstance(row, dict) else {}
    try:
        from supervisor.terminal_delivery import deliver_miss_lane_outcome

        return deliver_miss_lane_outcome(
            pathlib.Path(q.DRIVE_ROOT), q._task_drive_for_task(row, str(task_id)),
            row, task_id, status,
            unreconciled_runs=list(unreconciled_runs or []),
        ) is not False
    except Exception:
        log.warning("Miss-lane terminal delivery failed for %s", task_id, exc_info=True)
        return False


def _custody_disclosure_fields(
    audit: Optional[Dict[str, Any]], unreconciled: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Both halves of the kill-path custody disclosure for ONE terminal write.

    The flat ``delegated_runs_unreconciled`` list (audited — a clean ``[]``
    clears a stale stored list) rides together with the full audit envelope
    (``delegate_terminal_reconciliation``, stamped with the killing caller's
    own trigger), so a reader and the boot backfill's no-churn gate see one
    coherent disclosure from one write (R2). ``audit=None`` with an explicit
    list keeps the flat-only shape for a caller that ran no envelope-bearing
    audit; both absent yields ``{}`` — no audit ran, and an unaudited clean
    claim must never be minted.
    """
    if isinstance(audit, dict):
        flat = [str(item) for item in (audit.get("unreconciled") or []) if str(item)]
        return {
            "delegated_runs_unreconciled": flat,
            "delegate_terminal_reconciliation": dict(audit),
        }
    if unreconciled is None:
        return {}
    return {"delegated_runs_unreconciled": [str(item) for item in unreconciled if str(item)]}


def _audit_delegated_runs_on_kill(
    q: Any, task_id: str, *, trigger: str = "cancel_publication",
    deliberate_terminal: str = "",
) -> Dict[str, Any]:
    """Settle this task's open DELEGATED runs after its worker is dead; disclose
    what stayed open. Returns the FULL audit mapping (R2) — ``unreconciled``
    holds the still-open ids for the delivery notes, and the whole mapping is
    the ``delegate_terminal_reconciliation`` envelope the killing caller
    persists in its own single terminal write.

    Task-scoped reconciliation (A1.9): the graceful ``release_task_runs`` lived
    inside the worker that is now dead (cancel kill, timeout reap, or an
    already-settled task whose custody is being audited — GR5-3), so its open
    Claudexor runs would keep mutating until the next 10-minute orphan sweep.
    Durable custody rows are the complete view; cheap when the task never
    delegated.

    The DISCLOSURE is unconditional on the audit, never on the outcome list
    (GR2-7): ``reconcile_task_runs`` returning rows proves an ATTEMPT was made
    per run, not that any run settled — ``unreadable``/``requested``/failed
    outcomes and transport exceptions all leave runs open. So after the
    reconcile (successful, empty, or raising) the durable custody rows are
    ALWAYS re-read, and every run still open — plus any invocation still
    pending for this task — is disclosed (typed forensic row + a field on the
    cancelled result + a delivery note), not silently logged at debug.

    An AUDIT FAILURE is typed UNKNOWN, never clean (GR3-7): when the custody
    re-read itself raises, this returns a ``delegated_run_state_unknown``
    marker instead of ``[]`` — the marker rides the same
    ``delegated_runs_unreconciled`` surface (result field, typed event,
    delivery note, ``audit_failed`` flavor), and periodic reconciliation
    remains the eventual closer. A pending-invocation audit failure surfaces
    the same way.

    ``deliberate_terminal`` is the terminal status this caller is about to write
    when the kill IS the task's own deliberate end (an owner cancellation). The
    A4 ordering audits custody before that write, so the durable result the
    inverted cancel floor reads does not exist yet; without this the owner's
    cancel would leave the paid run live until the next periodic sweep. A host
    bound (deadline, reap) passes nothing and keeps sparing the run.

    GR6-4 closes the quiet corner of the same class: a custody
    log that EXISTS but cannot be OPENED used to replay as empty (audits as
    "cleanly reconciled") because ``_iter_rows`` swallows its own ``OSError``
    — the audit now probes readability first and reports the typed
    ``delegated_run_state_unknown:custody_log_unreadable`` marker; an ABSENT
    log stays a positively-established clean state.
    """
    from supervisor.terminal_delivery import RUN_STATE_UNKNOWN_PREFIX

    try:
        from ouroboros.delegate_custody_reconcile import attach_only_gateway
        from ouroboros.delegate_terminal import terminal_reconcile_task

        # A kill boundary attaches and settles; it never starts a daemon or
        # replays a pending invocation (the admitted sweep recovers those).
        audit = terminal_reconcile_task(
            pathlib.Path(q.DRIVE_ROOT), task_id, trigger=trigger,
            deliberate_terminal=deliberate_terminal, gateway_factory=attach_only_gateway,
        )
    except Exception:
        log.warning(
            "Delegated-run open-run audit failed for cancelled %s; run state is UNKNOWN",
            task_id, exc_info=True,
        )
        audit = {
            "task_id": str(task_id or ""), "trigger": str(trigger or ""),
            "outcomes": [],
            "unreconciled": [f"{RUN_STATE_UNKNOWN_PREFIX}:audit_failed"],
            "audit_status": "failed",
        }
    still_open = [str(item) for item in (audit.get("unreconciled") or []) if str(item)]
    audit["unreconciled"] = still_open
    audit_failed = str(audit.get("audit_status") or "") != "ok"
    if not still_open:
        return audit
    log.warning(
        "Cancelled task %s left %d delegated run(s) unreconciled: %s",
        task_id, len(still_open), still_open,
    )
    try:
        q.append_jsonl(
            pathlib.Path(q.DRIVE_ROOT) / "logs" / "events.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "delegated_runs_unreconciled",
                "task_id": task_id,
                "run_ids": still_open,
                **({"flavor": "audit_failed"} if audit_failed else {}),
                "detail": (
                    "the teardown audit itself failed; whether this task holds live "
                    "delegated runs is UNKNOWN until periodic reconciliation settles them"
                    if audit_failed else
                    "cancellation custody could not settle these delegated runs "
                    "(engine unreachable); they may keep mutating until the orphan sweep"
                ),
            },
        )
    except Exception:
        log.debug("Unreconciled-run disclosure append failed for %s", task_id, exc_info=True)
    return audit


def _reconcile_delegated_runs_on_kill(q: Any, task_id: str) -> List[str]:
    """Back-compat list view of ``_audit_delegated_runs_on_kill``: the still-open
    ids for callers that only thread the delivery-note disclosure."""
    return list(_audit_delegated_runs_on_kill(q, task_id).get("unreconciled") or [])


def _stop_open_tree_runs(q: Any, task_id: str) -> None:
    """The cascade's verdict over its tree's open delegated runs, before its postcondition.

    A member killed by the sweep cancels its own runs behind the owner's verdict, but a member that
    had already settled spared every run its own end was no verdict for (a provider death, a crash),
    so a settled tree whose only live writer is such a run stays live. The owner's Stop of the tree
    is that verdict: ``stop_tree_runs`` cancels them attach-only, and the settled owners' stored
    custody disclosures are refreshed, as the kill path's fast lane does. A failure is logged and
    judged by the postcondition, which counts every run still open as live.
    """
    root = pathlib.Path(q.DRIVE_ROOT)
    try:
        from ouroboros.delegate_custody_reconcile import stop_tree_runs
        from ouroboros.delegate_terminal import refresh_terminal_reconciliation

        outcomes = stop_tree_runs(root, task_id)
        for owner in sorted({str(row.get("task_id") or "") for row in outcomes} - {""}):
            refresh_terminal_reconciliation(root, owner, trigger="kill_path_clear")
    except Exception:
        log.warning("Cascade could not stop the open delegated runs of %s", task_id, exc_info=True)


def _cascade_delivery_row_locked(q: Any, task_id: str) -> Dict[str, Any]:
    """A routing row for a cascade whose ROOT has already left the live maps.

    Caller holds the queue lock. Returns the first live descendant's row (they
    carry the lineage ``chat_id``), or ``{}`` when the subtree is empty too.
    (Moved verbatim from ``task_lifecycle.py`` at its module-size boundary.)
    """
    for task in q.PENDING:
        if isinstance(task, dict) and q._is_descendant_of(task, task_id) and task.get("chat_id") is not None:
            return dict(task)
    for meta in q.RUNNING.values():
        task = meta.get("task") if isinstance(meta, dict) else None
        if isinstance(task, dict) and q._is_descendant_of(task, task_id) and task.get("chat_id") is not None:
            return dict(task)
    return {}


def _finish_captured_chat_turn(
    q: Any, task_id: str, turn: Dict[str, Any], *, intent: Optional[Dict[str, Any]] = None,
    deliver: bool = True,
) -> str:
    """Custody's half of stopping the in-process direct-chat turn: the lane
    arms the cooperative stop and waits its short bound
    (``worker_chat_lane.stop_direct_chat_turn``); custody settles the intent
    against what the turn published, or releases the claim so the sweep
    retries — the HTTP caller then sees 202 pending, never a fabricated
    ``cancelled`` row over a turn that is still running. A turn that ENDED
    without publishing a terminal (the lane's error path writes none) takes
    the same miss finalizer a pooled task does — ``cancelled`` with the
    reconstructed cost — so a "successful" stop can never leave a ``running``
    row that no control can reach again. A turn that was already gone before
    the stop could be armed is the pooled lane's ``already_settled`` — unless
    its paid post-task synthesis is still billing on the in-process worker
    (``turn`` is then None): the claim is released with the typed "still
    live" error so the durable immediate intent stays open for the
    pipeline's per-stage gate, and the sweep settles ``already_settled``
    once the worker drops its in-flight key."""
    from supervisor.task_lifecycle import (
        CANCEL_FAILED, SETTLED_ALREADY, _release_intent_claim, _settle_intent,
    )
    from supervisor.worker_chat_lane import (
        DIRECT_TURN_STOP_GONE, DIRECT_TURN_STOP_LIVE, stop_direct_chat_turn,
    )
    from ouroboros.post_task_checkpoint import post_task_synthesis_in_flight
    from ouroboros.task_results import load_task_result
    from ouroboros.task_status import SETTLED_STATUSES

    outcome = (
        DIRECT_TURN_STOP_GONE if turn is None
        else stop_direct_chat_turn(task_id, turn, deliver=deliver)
    )
    if outcome == DIRECT_TURN_STOP_LIVE:
        _release_intent_claim(
            q, task_id, error="direct chat turn has not reached its next step yet",
            intent=intent,
        )
        return CANCEL_FAILED
    if post_task_synthesis_in_flight(q.DRIVE_ROOT, task_id):
        _release_intent_claim(
            q, task_id, error="post-task synthesis of the direct chat turn is still running",
            intent=intent,
        )
        return CANCEL_FAILED
    from supervisor.active_activity import get_direct_activity_registry

    if get_direct_activity_registry().get(task_id) is not None:
        # Closed message admission is not execution completion, also when it
        # closed after Stop captured a live turn. Preparation or async teardown
        # can still own the actor; it observes the durable Stop through task
        # control. The registry releases only after its runner exits.
        _release_intent_claim(q, task_id, error="direct activity still owns work", intent=intent)
        return CANCEL_FAILED
    stored = load_task_result(q.DRIVE_ROOT, task_id) or {}
    if str(stored.get("status") or "") not in SETTLED_STATUSES:
        return _finalize_cancel_intent_on_miss(q, task_id, intent=intent)
    _settle_intent(q, task_id, outcome=SETTLED_ALREADY,
                   detail=str(stored.get("status") or ""), intent=intent)
    return CANCEL_ALREADY_SETTLED if outcome == DIRECT_TURN_STOP_GONE else CANCEL_CANCELLED


def live_presence_row(row: Any) -> bool:
    """An unsettled row a live Presence turn owns: its runner settles it, not cancel custody."""
    from supervisor.queue_transitions import settled_status_of

    owner = row.get("execution_owner") if isinstance(row, dict) else None
    return isinstance(owner, dict) and owner.get("kind") == "presence" and not settled_status_of(row)


def _finalize_cancel_intent_on_miss(
    q: Any, task_id: str, *, intent: Optional[Dict[str, Any]] = None,
) -> str:
    """Neither queued nor running: settle an open cancel intent (or a legacy
    ``cancel_requested`` latch file) as cancelled with reconstructed cost.

    Three things this lane must NOT do. It must not invent a task: an intent for
    an id that has no durable result at all names a task that never existed, and
    fabricating a ``cancelled`` row with $0 for it would put a phantom task in the
    ledger — it settles as ``not_found`` instead. It must not take a row whose
    execution owner is a live Presence turn (no supervisor map knows it): that
    settles ``not_found`` too, the row untouched. And it must not bury a child
    that finished: when the row names a child drive, the child's own result is
    copied back BEFORE the cancelled write, so a crash of the split-drive
    copy-back window cannot cost a completed answer. Nor does a direct restore fence
    bury an answer the host already owes or delivered under cancel text and a
    "no salvageable output" notice: that attempt is finalized from the answer.
    """
    from supervisor.queue_transitions import _settled_status
    from supervisor.task_lifecycle import (
        SETTLED_ALREADY,
        _active_intent,
        _recover_stranded_reaping_slot,
        _release_intent_claim,
        _settle_intent,
    )

    from ouroboros.task_results import (
        STATUS_CANCEL_REQUESTED, STATUS_CANCELLED, load_task_result,
        write_task_result,
    )

    try:
        active = dict(intent or {})
        if not active:
            active = _active_intent(q, task_id)
        existing = load_task_result(q.DRIVE_ROOT, task_id) or {}
        legacy_latch = str(existing.get("status") or "") == STATUS_CANCEL_REQUESTED
        if not active and not legacy_latch:
            return CANCEL_NOT_FOUND
        if not existing:
            # No durable row ANYWHERE for this id: nothing was ever scheduled
            # under it (a mistyped/stale id reaching the cancel ingress). Settle
            # the intent honestly rather than minting a cancelled task.
            _settle_intent(q, task_id, outcome="not_found",
                           detail="no durable task result for this id", intent=intent)
            return CANCEL_NOT_FOUND
        if live_presence_row(existing):
            # A live Presence turn runs on its own thread outside every supervisor
            # map: nothing here can capture it, and a cancelled write would only
            # strand a turn that keeps spending. The row stays its runner's and
            # the Stop answers "not active", as it did before intent-first minting.
            _settle_intent(q, task_id, outcome="not_found",
                           detail="execution owned by a live Presence turn", intent=intent)
            return CANCEL_NOT_FOUND
        # A concurrent custody attempt may have captured this task between our
        # own capture miss and here (the pending double-settle probe). If the
        # live claim is no longer ours, it owns the settle — refuse and let it,
        # or the watchdog, finish.
        current = _active_intent(q, task_id)
        if (
            intent
            and current
            and str(current.get("request_id") or "") == str(intent.get("request_id") or "")
            and int(current.get("generation") or 0) != int(intent.get("generation") or 0)
        ):
            log.warning(
                "Cancel finalize-on-miss for %s yielded to a newer custody claim", task_id,
            )
            return CANCEL_FAILED
        # A4/completion-wins on the split-drive lane: promote the child's own
        # terminal result first when the row names a child drive.
        try:
            from ouroboros.headless import copy_child_task_result

            if str(existing.get("child_drive_root") or "").strip():
                copy_child_task_result(pathlib.Path(q.DRIVE_ROOT), {
                    "id": task_id,
                    "drive_root": str(existing.get("child_drive_root") or ""),
                    "child_drive_root": str(existing.get("child_drive_root") or ""),
                    "delegation_role": str(existing.get("delegation_role") or ""),
                })
        except Exception:
            log.debug("Finalize-on-miss child copy-back failed for %s", task_id, exc_info=True)
        # GR5-3: neither queued nor running — the worker is gone, but its
        # delegated runs may still be live; audit custody like the kill path
        # and thread the disclosure into every miss-lane delivery below.
        # Read after child copyback: only an unsettled task will receive our
        # cancelled write. An existing terminal keeps its own custody verdict.
        settled = _settled_status(q.DRIVE_ROOT, task_id)
        audit = _audit_delegated_runs_on_kill(
            q, task_id, **({} if settled else {"deliberate_terminal": STATUS_CANCELLED}),
        )
        unreconciled = list(audit.get("unreconciled") or [])
        if settled:
            _recover_stranded_reaping_slot(q, task_id, active)
            # D1b (R4): this branch performs no terminal write of its own, so
            # the same guarded stale-only refresh the fast already-settled
            # cancel lane runs clears a stale stored disclosure here — never
            # minting a row and never rewriting a current one.
            try:
                from ouroboros.delegate_terminal import refresh_terminal_reconciliation

                refresh_terminal_reconciliation(
                    pathlib.Path(q.DRIVE_ROOT), task_id, trigger="kill_path_clear",
                )
            except Exception:
                log.debug("Miss-lane custody-disclosure refresh failed for %s",
                          task_id, exc_info=True)
            owed_ok = _deliver_on_miss(
                q, task_id, load_task_result(q.DRIVE_ROOT, task_id) or existing, settled,
                unreconciled_runs=unreconciled,
            )
            _settle_or_reopen_intent(q, task_id, owed_ok=owed_ok, intent=intent,
                                     outcome=SETTLED_ALREADY, detail=settled)
            return CANCEL_ALREADY_SETTLED
        existing = load_task_result(q.DRIVE_ROOT, task_id) or existing
        cost_fields = _reconstructed_cost_fields(q, task_id, existing)
        from ouroboros.store import StoreBusy, acquisition_budget
        from ouroboros.working_checkpoint import adopt_registered_answer, registered_answer
        from supervisor.task_reaper import _deferred_reap_jobs
        registry_root = pathlib.Path(existing.get("budget_drive_root") or q.DRIVE_ROOT)
        # A direct restore fence keeps a positively held answer, like dead-worker recovery.
        # Other cancel paths (including pooled restore) keep their existing outcome.
        owner = existing.get("execution_owner")
        adopt = (isinstance(owner, dict) and owner.get("kind") == "direct"
                 and bool(shutdown_cancel_text(active))
                 and registered_answer(registry_root, task_id).get("basis") in {"owed", "exhausted", "delivered"})
        adopted: Optional[Dict[str, Any]] = None
        # Under Q the health tick hands a retained reap back to its reaper wholly before or after
        # this write: an idle one replays over it; one out for its replay keeps the slot and the write.
        try:
            with acquisition_budget(0), q._queue_lock:
                retained = [job for job in _deferred_reap_jobs if str(job.get("task_id") or "") == task_id]
                replaying = any(job.get("_replaying") for job in retained)
                if not replaying and adopt:
                    adopted = adopt_registered_answer(q.DRIVE_ROOT, existing, task_id, cause="app_stop",
                                                      attempt=int(existing.get("task_attempt") or 0),
                                                      registry_root=registry_root)
                if not replaying and not adopted:
                    stored = write_task_result(
                        q.DRIVE_ROOT, task_id, STATUS_CANCELLED,
                        **_cancel_result_fields(
                            existing, existing=existing, **cost_fields,
                            **_intent_outcome_fields(active),
                            # R2/R4: the audited list AND its envelope ride this single
                            # cancelled write — a clean audit clears a stale stored list.
                            **_custody_disclosure_fields(audit),
                            result=_miss_lane_cancel_text(active),
                        ),
                    )
        except StoreBusy:  # Q never waits on the store: the open intent returns to the watchdog
            _release_intent_claim(q, task_id, error="result store contended under the queue lock", intent=intent)
            return CANCEL_FAILED
        if replaying:  # the open intent returns to the watchdog, as for any reaper-owned slot
            _release_intent_claim(q, task_id, error="slot owned by the reaper's replay", intent=intent)
            return CANCEL_FAILED
        if adopted:  # published like dead-worker recovery's adoption: no cancel text, no "no output" notice
            if not retained:
                from supervisor import workers
                from supervisor.task_reaper import _finish_self_finalized_task
                from supervisor.worker_health import terminal_task_metadata

                _finish_self_finalized_task(q, workers, existing, task_id, str(existing.get("type") or "task"),
                                            str(adopted.get("status") or ""), adopted,
                                            terminal_task_metadata(existing.get("metadata")), unreconciled)
            _settle_intent(q, task_id, outcome=SETTLED_ALREADY, detail="registered_answer_finalized", intent=intent)
            q.persist_queue_snapshot(reason="cancel_finalize")
            return CANCEL_ALREADY_SETTLED
        stored_status = str((stored or {}).get("status") or "")
        if stored_status != STATUS_CANCELLED:
            # The monotonic guard refused: something settled it while we worked.
            owed_ok = _deliver_on_miss(q, task_id, stored or existing, stored_status,
                                       unreconciled_runs=unreconciled)
            _settle_or_reopen_intent(q, task_id, owed_ok=owed_ok, intent=intent,
                                     outcome=SETTLED_ALREADY, detail=stored_status)
            return CANCEL_ALREADY_SETTLED
        # GR2-4 ordering: the delivery seam registers the answer as OWED before
        # the intent settles — a crash between the two replays instead of losing
        # both the watchdog trigger and the answer. GR4-1: an unowed answer
        # reopens the intent; the publication below still proceeds — the
        # terminal truth is on disk.
        owed_ok = _deliver_on_miss(q, task_id, stored or existing, STATUS_CANCELLED,
                                   unreconciled_runs=unreconciled)
        _settle_or_reopen_intent(q, task_id, owed_ok=owed_ok, intent=intent,
                                 outcome="cancelled", detail="finalized on miss")
        # A reap the reaper retains (its slot, its replay) publishes this terminal once, on that replay.
        if not retained:
            q._emit_cancel_task_done(existing, task_id, cost_fields=cost_fields)
        q.persist_queue_snapshot(reason="cancel_finalize")
        return CANCEL_CANCELLED
    except Exception:
        log.debug("Cancel finalize-on-miss failed for %s", task_id, exc_info=True)
        _release_intent_claim(q, task_id, error="finalize-on-miss failed", intent=intent)
        return CANCEL_FAILED
