"""The durable queue snapshot: what a restart finds and what it may restore.

The snapshot is written under the queue lock from the live PENDING/RUNNING rows and
the acceptance fences beside them, and restored only while PENDING is empty and the
file is young enough to describe the world the supervisor is waking into.
"""

from __future__ import annotations

import datetime
import json
import logging
import pathlib
import time
from typing import Any, Optional

from ouroboros.contracts.schema_versions import SCHEMA_VERSION_KEY
from ouroboros.utils import utc_now_iso
from supervisor.task_admission import (
    restore_terminalization_retry,
    restore_terminalization_retry_rows,
    restored_handoff_unproven,
)
from supervisor.task_lifecycle import (
    _cancel_result_fields,
    restore_queue_fences,
)


def _queue():
    """The parent module, read at call time.

    The queue owns PENDING/RUNNING, the drive root, the liveness settings and
    the lock that guards them, and ``init``/``init_queue_refs`` REBIND those
    names. Reading them through the module is what keeps one binding: a
    from-import here would freeze the value this module saw at import time
    (the owner-approved D18/D33 mechanical exception).
    """
    from supervisor import queue

    return queue


log = logging.getLogger(__name__)

# ABI 7.0 (Q8=B): the durable queue snapshot names its schema on every write.
# Stamp-on-write ONLY — the restore path does not require the stamp (no
# compat branching), so an N−1 snapshot restores unchanged.
QUEUE_SNAPSHOT_SCHEMA_VERSION = 1


def _kept_service_pids() -> "set[int]":
    """PIDs of deliberately-kept (session-scope) services to spare from a worker
    tree-kill on cancel/hard-timeout. Best-effort; never raises."""
    try:
        from ouroboros.process_custody import live_kept_service_pids
        return live_kept_service_pids(pathlib.Path(_queue().DRIVE_ROOT))
    except Exception:
        return set()


def _retained_daemon_pids() -> "set[int]":
    """PIDs of the installation's live daemon roots (``daemon``-scope custody rows,
    the shared Claudexor daemon among them) that EVERY worker tree-kill spares:
    their lifetime is the installation's, not the worker's. Best-effort; never raises."""
    try:
        from ouroboros.process_custody import live_daemon_root_pids
        from ouroboros.claudexor_daemon import CUSTODY_PURPOSE
        return live_daemon_root_pids(
            pathlib.Path(_queue().DRIVE_ROOT), retained_purposes={CUSTODY_PURPOSE},
        )
    except Exception:
        return set()


def persist_queue_snapshot(reason: str = "") -> bool:
    """Persist queue snapshot for restart/recovery diagnostics.

    Snapshots PENDING/RUNNING under the queue lock: iterating the live dicts
    while HTTP handlers mutate them raised "dictionary changed size during
    iteration" in the supervisor loop (counted toward its crash limit).
    """
    with _queue()._queue_lock:
        pending_items = [dict(t) for t in _queue().PENDING]
        running_items = [
            (task_id, dict(meta) if isinstance(meta, dict) else {})
            for task_id, meta in _queue().RUNNING.items()
        ]
        acceptance_fences = [dict(row) for row in _queue().ACCEPTANCE_FENCES.values()]
        budget_root_fences = [dict(row) for row in _queue().BUDGET_ROOT_FENCES.values()]
        # Honest worker-pool counts from the ACTUAL pool (not the configured max): the live
        # pool can be smaller (a crash-storm/direct-chat fallback clears WORKERS) and a slot
        # mid-reap is popped from RUNNING but NOT assignable. Surface the real assignable-idle
        # count so the context queue digest never falsely advertises a free worker slot.
        try:
            from supervisor import workers as _workers_mod

            _ws = list(_workers_mod.WORKERS.values())
            worker_total = len(_ws)
            active_worker_count = sum(1 for w in _ws if getattr(w, "active_capacity", True))
            parked_worker_count = worker_total - active_worker_count
            worker_pool_disabled_reason = _workers_mod._worker_pool_execution_state()["disabled_reason"]
            if worker_pool_disabled_reason == "no_workers":
                worker_pool_disabled_reason = ""  # Absence is counted above, not an explicit disablement.
            reaping_count = sum(1 for _w in _ws if getattr(_w, "reaping", False))
            assignable_idle_workers = sum(
                1 for _w in _ws
                if (getattr(_w, "busy_task_id", None) is None and not getattr(_w, "reaping", False)
                    and getattr(_w, "active_capacity", True))
            )
        except Exception:
            worker_total = 0
            active_worker_count = parked_worker_count = 0
            worker_pool_disabled_reason = "unknown"
            reaping_count = 0
            assignable_idle_workers = 0
    pending_rows = []
    for t in pending_items:
        pending_rows.append({
            "id": t.get("id"), "type": t.get("type"), "priority": t.get("priority"),
            "attempt": t.get("_attempt"), "queued_at": t.get("queued_at"),
            "queue_seq": t.get("_queue_seq"),
            "task": {
                "id": t.get("id"), "type": t.get("type"), "chat_id": t.get("chat_id"),
                "text": t.get("text"), "priority": t.get("priority"),
                "depth": t.get("depth"), "description": t.get("description"),
                "objective": t.get("objective"), "title": t.get("title"),
                "expected_output": t.get("expected_output"),
                "constraints": t.get("constraints"), "role": t.get("role"),
                "context": t.get("context"), "parent_task_id": t.get("parent_task_id"),
                "root_task_id": t.get("root_task_id"), "session_id": t.get("session_id"),
                "actor_id": t.get("actor_id"), "delegation_role": t.get("delegation_role"),
                "workspace_root": t.get("workspace_root"), "workspace_mode": t.get("workspace_mode"),
                "project_id": t.get("project_id"),
                **({"_project_admission": t["_project_admission"]} if "_project_admission" in t else {}),
                "_project_admission_restore_hold": t.get("_project_admission_restore_hold"),
                **({"admitted_dispatch": t["admitted_dispatch"]} if "admitted_dispatch" in t else {}),
                **({"_project_scope_none": t["_project_scope_none"]} if "_project_scope_none" in t else {}),
                "focus": t.get("focus"),
                "allowed_resources": t.get("allowed_resources"), "deadline_at": t.get("deadline_at"),
                "task_contract": t.get("task_contract"),
                "_owner_hold": t.get("_owner_hold"),
                "_consciousness_continuation": t.get("_consciousness_continuation"),
                # Scheduling INTENT survives a restart and is all a PENDING child has;
                # `parent_model_lane` and the F9 admission fact `required_model_lane`
                # above all (R2-3). Pinned to SUBAGENT_INTENT_FIELDS by test_model_slot.
                "model_lane": t.get("model_lane"), "parent_model_lane": t.get("parent_model_lane"),
                "requested_model_lane": t.get("requested_model_lane"),
                "required_model_lane": t.get("required_model_lane"), "requested_executor": t.get("requested_executor"),
                "effective_model_lane": t.get("effective_model_lane"),
                "model": t.get("model"), "use_local_model": t.get("use_local_model"),
                "effective_executor": t.get("effective_executor"), "tool_profile": t.get("tool_profile"),
                "executor_route": t.get("executor_route"), "reasoning_effort": t.get("reasoning_effort"),
                "capability_delta": t.get("capability_delta"),
                "task_group_id": t.get("task_group_id"),
                "task_group": t.get("task_group"),
                "subagent_envelope": t.get("subagent_envelope"), "configured_subagent": t.get("configured_subagent"),
                "memory_mode": t.get("memory_mode"), "drive_root": t.get("drive_root"), "parent_cognitive_route": t.get("parent_cognitive_route"), "memory_reference": t.get("memory_reference"), "subagent_availability": t.get("subagent_availability"),
                "child_drive_root": t.get("child_drive_root"),
                "budget_drive_root": t.get("budget_drive_root"),
                "task_constraint": t.get("task_constraint"), "predecessor_authority_source": t.get("predecessor_authority_source"),
                "metadata": t.get("metadata"), "origin_message_ref": t.get("origin_message_ref"),
                "origin_message_text": t.get("origin_message_text"), "_attempt": t.get("_attempt"),
                "review_reason": t.get("review_reason"), "review_source_task_id": t.get("review_source_task_id"),
                "_budget_pause": t.get("_budget_pause"), "budget_resumed_at": t.get("budget_resumed_at"), "_terminalization_retry": t.get("_terminalization_retry"),
                "_cancel_intent_authority_hold": t.get("_cancel_intent_authority_hold"),
                "_owner_wait_resume": t.get("_owner_wait_resume"),
                "_budget_pause_resume": t.get("_budget_pause_resume"),
                # The durable non-dispatch hold (#1196) and any recorded
                # selection: a restart must not silently make a held row runnable.
                "_budget_pause_hold": t.get("_budget_pause_hold"),
                # A direct owner-chat turn parked under its exact budget pause
                # keeps its lane fact: it is resumed under the same id and its
                # frames, census kind and delivery read that fact (#1196).
                "_is_direct_chat": t.get("_is_direct_chat"),
            },
        })
    running_rows = []
    from ouroboros.model_wait import execution_elapsed_seconds
    from supervisor.task_model_wait import quota_waited_seconds
    now = time.time()
    for task_id, meta in running_items:
        task = meta.get("task") if isinstance(meta, dict) else {}
        started = float(meta.get("started_at") or 0.0) if isinstance(meta, dict) else 0.0
        hb = float(meta.get("last_heartbeat_at") or 0.0) if isinstance(meta, dict) else 0.0
        paused_sec = float(meta.get("budget_paused_sec") or 0.0) if isinstance(meta, dict) else 0.0
        running_rows.append({
            "id": task_id, "type": task.get("type"), "priority": task.get("priority"),
            "attempt": meta.get("attempt"), "worker_id": meta.get("worker_id"),
            "owner_wait": meta.get("owner_wait"), "started_at": started,
            "runtime_sec": round(max(0.0, now - started), 2) if started > 0 else 0.0,
            "quota_wait_sec": quota_waited_seconds(meta, now),
            "budget_paused_sec": paused_sec,
            "execution_sec": execution_elapsed_seconds(meta if isinstance(meta, dict) else {}, now),
            "heartbeat_lag_sec": round(max(0.0, now - hb), 2) if hb > 0 else None,
            "soft_sent": bool(meta.get("soft_sent")), "task": task,
        })
    payload = {
        SCHEMA_VERSION_KEY: QUEUE_SNAPSHOT_SCHEMA_VERSION,
        "ts": utc_now_iso(),
        "reason": reason,
        "pending_count": len(pending_items), "running_count": len(running_items),
        "reaping_count": reaping_count,
        "worker_total": worker_total,
        "active_worker_count": active_worker_count, "parked_worker_count": parked_worker_count,
        "worker_pool_disabled_reason": worker_pool_disabled_reason,
        "assignable_idle_workers": assignable_idle_workers,
        "acceptance_fences": acceptance_fences,
        "budget_root_fences": budget_root_fences,
        "pending": pending_rows, "running": running_rows,
    }
    try:
        _queue().atomic_write_text(_queue().QUEUE_SNAPSHOT_PATH, json.dumps(payload, ensure_ascii=False, indent=2))
        return True
    except Exception:
        log.warning("Failed to persist queue snapshot (reason=%s)", reason, exc_info=True)
        return False


def parse_iso_to_ts(iso_ts: str) -> Optional[float]:
    """Parse ISO timestamp to Unix time."""
    txt = str(iso_ts or "").strip()
    if not txt:
        return None
    try:
        return datetime.datetime.fromisoformat(txt.replace("Z", "+00:00")).timestamp()
    except Exception:
        log.debug("Failed to parse ISO timestamp: %s", txt, exc_info=True)
        return None


def _fence_snapshot_running_rows(rows: Any, *, restored_ids: "set[str]") -> "list[str]":
    """Fence every RUNNING row that survived the shutdown with a durable cancel intent.

    Restore is the last holder of the pre-restart running list, but it is NOT a
    terminal writer: minting the intent hands each row to the one settle owner,
    which claims it, kills a worker that outlived SIGTERM, reconciles, and only
    then writes the terminal with its own text — expiring an open quiz and
    closing the paired owner wait through the task-done seam. A second writer
    here would race that surviving worker; an intent cannot. An UNREADABLE
    cancel authority mints nothing: the unknown is disclosed, never fenced, and
    neither is a row with no durable result — custody would settle that intent as
    not_found and write nothing, leaving the boot notice naming a cancellation
    that never happens. Returns the fenced task ids.
    """
    from ouroboros.cancel_intents import has_active_intent, request_cancel
    from ouroboros.task_results import (
        _TRULY_TERMINAL_STATUSES, STATUS_CANCEL_REQUESTED, load_task_result,
    )

    fenced: list[str] = []
    unreadable: list[str] = []
    unrecorded: list[str] = []
    for row in rows if isinstance(rows, list) else []:
        task_id = str(row.get("id") or "") if isinstance(row, dict) else ""
        if not task_id or task_id in restored_ids:
            continue
        try:
            stored = load_task_result(_queue().DRIVE_ROOT, task_id, strict=True) or {}
            if not stored:
                # Admission writes the durable scheduled row (task_admission.py
                # :536-543), so even a failed RUNNING mirror leaves a readable
                # row this loop fences; reaching here means the admission record
                # is missing or was deleted, and custody cannot settle an id it
                # has no record of, so the row is logged and never fenced.
                unrecorded.append(task_id)
                continue
            status = str(stored.get("status") or "")
            if status in _TRULY_TERMINAL_STATUSES or status == STATUS_CANCEL_REQUESTED:
                continue
            if has_active_intent(_queue().DRIVE_ROOT, task_id, strict=True):
                continue  # cancellation custody already owns this row
            intent = request_cancel(
                _queue().DRIVE_ROOT, task_id,
                reason="server_shutdown", source="snapshot_restore",
            )
        except Exception:
            unreadable.append(task_id)
            log.warning("Snapshot restore left running row %s unfenced: its cancel "
                        "authority is unreadable", task_id, exc_info=True)
            continue
        if not intent.get("already_settled"):
            fenced.append(task_id)
    for kind, task_ids in (("queue_restore_running_fence_unreadable", unreadable),
                           ("queue_restore_running_row_without_result", unrecorded)):
        if task_ids:
            _queue().append_jsonl(
                _queue().DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {"ts": utc_now_iso(), "type": kind, "task_ids": task_ids},
            )
    return fenced


def _exact_pause_row(task: Any) -> bool:
    """Whether this snapshot row is an EXACT mid-run budget pause locator (#1196)."""
    pause = task.get("_budget_pause") if isinstance(task, dict) else None
    # Typed: the marker writers stamp a boolean; a truthy string or an
    # otherwise malformed marker is NOT a saved pause and keeps ordinary
    # drain/parent-interrupted settlement (Astra run-e0bb1ca3487c A1).
    return isinstance(pause, dict) and pause.get("exact_continuation") is True


def _retain_snapshot_pending(snapshot_pending: list, running_rows: list, *, stale: bool,
                             direct_caught: "list[str]" = ()) -> tuple:
    """Which snapshot rows survive the restore, and the RUNNING rows parked beside them.

    A grant that never reached a worker before this restart is not carried into
    the new generation: it returns to its exact pause and the owner re-issues
    Resume (a revocation that cannot be written leaves the row HELD). A grant
    the durable row says was CONSUMED is never re-armed: that row is stale
    carrier of work that ran on, so it is not retained at all and is fenced as
    the running work it names (its id is returned third). A RUNNING row whose
    durable pause was already complete is parked, never fenced; so is a direct
    root the stop caught (``direct_caught``) in that state: a direct turn is
    never a RUNNING row, so the roster is the only place the stop names it, and
    handing such an id to the shutdown cancel fence would cancel a saved pause
    (#1196). Its queue record is rebuilt from the durable result the pause
    wrote (chat, lane fact, origin, metadata, attempt) and parked through the
    same path a RUNNING row takes; a turn with no complete pause (no row, no
    source, another attempt) stays on the fence path. An exact budget pause is
    retained WITHOUT waking whatever the snapshot's age — a corrupt, missing or
    refused source becomes a typed, visible HOLD beside its marker, never a
    dropped or cancelled task. An owner-wait handoff needs its acknowledged
    restart transaction. Project admission holds retain custody through age and
    schedule-receipt uncertainty, without acquiring dispatch permission; a fresh
    host-unscoped row naming a possible handoff is held, never replayed.
    Returns ``(retained_rows, parked_rows, consumed_task_ids)``.
    """
    from ouroboros.budget_pause import budget_pause_restore_refusal, budget_pause_row
    from ouroboros.owner_wait import restore_owner_wait_allowed
    from ouroboros.project_admission import host_unscoped
    from ouroboros.task_results import load_task_result
    from supervisor.budget_resume import revoke_exact_budget_resume
    from supervisor.events_budget import HOLD_RESTORE_REFUSED_PREFIX, hold_restored_budget_pause
    from supervisor.schedule_occurrence import restore_allowed

    for task in snapshot_pending:
        if isinstance(task.get("_budget_pause_resume"), dict):
            revoke_exact_budget_resume(task, "restart_before_dispatch")
    direct_rows: list = []
    for task_id in direct_caught:
        task_id = str(task_id or "")
        if not task_id:
            continue
        try:
            stored = load_task_result(_queue().DRIVE_ROOT, task_id, strict=True) or {}
            pause = budget_pause_row(_queue().DRIVE_ROOT, task_id)
        except Exception:
            continue
        if not (stored and pause and pause.get("source_ref") and pause.get("is_direct_chat")):
            continue
        attempt = int(pause.get("task_attempt") or 1)
        record: dict = {
            "id": task_id, "type": "task", "chat_id": stored.get("chat_id"), "_is_direct_chat": True,
            "_attempt": attempt, "root_task_id": task_id, "depth": 0,
        }
        for key in ("origin_message_ref", "origin_message_text", "metadata", "project_id",
                    "task_contract", "title", "text", "budget_drive_root"):
            if stored.get(key) not in (None, ""):
                record[key] = stored[key]
        direct_rows.append({"task": record, "attempt": attempt})
    parked = _park_pausing_running_rows(list(running_rows) + direct_rows, snapshot_pending)
    retained = []
    consumed: list = []
    for task in list(snapshot_pending) + parked:
        if isinstance(task.get("_budget_pause_consumed"), dict):
            consumed.append(str(task.get("id") or ""))
            continue
        if _exact_pause_row(task):
            refusal = budget_pause_restore_refusal(_queue().DRIVE_ROOT, task)
            retained.append(task if not refusal else hold_restored_budget_pause(
                task, _queue().DRIVE_ROOT, reason=HOLD_RESTORE_REFUSED_PREFIX + refusal))
        elif task.get("_project_admission_restore_hold"):
            # Custody, not replay: retain through age/receipt uncertainty. The
            # terminal/cancel checks below still run; assignment skips this hold.
            retained.append(task)
        elif task.get("_owner_wait_resume"):
            if restore_owner_wait_allowed(_queue().DRIVE_ROOT, task):
                retained.append(task)
        elif (task.get("project_id") or "_project_admission" in task) and (
                stale or task.get("admitted_dispatch") != "none" or not restore_allowed(task)):
            # Age or an unavailable receipt cannot erase accepted Project work.
            # Assignment still requires positive original no-dispatch authority.
            retained.append({**task, "_project_admission_restore_hold": {
                "reason": "project_routing_fence_lookup_failed",
                "detail": ("The saved task needs original Project and no-dispatch verification."
                           if isinstance(task.get("_project_admission"), dict) else
                           "The original Project basis has unknown historical identity; automatic recovery is not authorized."),
            }})
        elif host_unscoped(task) and not stale and task.get("admitted_dispatch") != "none" and restore_allowed(task):
            retained.append({**task, "_project_admission_restore_hold": {
                "reason": "project_routing_fence_lookup_failed",
                "detail": "The task may already have reached a worker; automatic recovery is not authorized."}})
        elif not restore_allowed(task):
            continue  # a schedule-born row that may have been dispatched, or is unprovable, is never replayed
        elif not stale:
            retained.append(task)
    if consumed:
        _queue().append_jsonl(
            _queue().DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {"ts": utc_now_iso(), "type": "queue_restore_stale_consumed_grant_rows",
             "task_ids": consumed, "action": "fenced_as_running_work"},
        )
    return retained, parked, consumed


def _raise_parked_root_fences(parked_pausing: list) -> None:
    """Raise the root admission latch of every root parked at restore — AFTER the
    snapshot's fence map is restored, or the restore would erase it."""
    if not parked_pausing:
        return
    from supervisor.events_budget import _set_root_budget_pause_locked

    with _queue()._queue_lock:
        for parked in parked_pausing:
            marker = parked.get("_budget_pause") if isinstance(parked.get("_budget_pause"), dict) else {}
            if str(marker.get("scope") or "") == "root" and marker.get("root_task_id"):
                fence = _set_root_budget_pause_locked(str(marker["root_task_id"]), marker)
                marker["fence_id"] = fence["fence_id"]


def _refuse_restore_invalid_fences(snapshot_pending: list, *, budget: bool = False) -> int:
    """Malformed fences never restore ordinary work. Invalid acceptance evidence
    cancels ordinary rows; invalid budget evidence leaves them unrestored.

    A saved EXACT budget pause is the exception (#1196): it is retained as a
    typed, non-dispatchable HOLD with its ORIGINAL ``_budget_pause`` locator
    (an earlier restore hold on the same row is kept), appended to PENDING
    directly, never cancelled — the owner's next Resume re-validates it. A
    pause whose durable result is already terminal is left alone. Returns the
    number of retained rows.
    """
    from ouroboros.task_results import (
        _TRULY_TERMINAL_STATUSES, STATUS_CANCELLED, load_task_result, write_task_result,
    )
    from supervisor.events_budget import (
        HOLD_INVALID_ACCEPTANCE_FENCE_SNAPSHOT, HOLD_INVALID_BUDGET_FENCE_SNAPSHOT,
        budget_hold_fact, hold_restored_budget_pause,
    )

    cancelled: list[str] = []
    retained: list[str] = []
    skipped_terminal: list[str] = []
    for task in snapshot_pending:
        task_id = str(task.get("id") or "")
        if not task_id:
            continue
        if task.get("_project_admission_restore_hold") or task.get("project_id"):
            # Malformed independent fence evidence is not cleared by Project
            # revalidation. Keep the row under the existing owner hold as well.
            held = {**task, "_owner_hold": task.get("_owner_hold") or {
                "reason": "The saved budget or acceptance fence is invalid; dispatch is held.",
            }, "_project_admission_restore_hold": task.get("_project_admission_restore_hold") or {
                "reason": "project_routing_fence_lookup_failed", "detail": "Independent restore authority is invalid.",
            }}
            with _queue()._queue_lock:
                _append_held_pending_row(held)
            retained.append(task_id)
            continue
        if not _exact_pause_row(task):
            cancelled.append(task_id)
            continue
        try:
            stored = load_task_result(_queue().DRIVE_ROOT, task_id, strict=True) or {}
            if str(stored.get("status") or "") in _TRULY_TERMINAL_STATUSES:
                skipped_terminal.append(task_id)
                continue
        except Exception:
            log.debug("Result authority unreadable for paused row %s; retained held", task_id, exc_info=True)
        held = dict(task)
        if budget_hold_fact(held) is None:
            held = hold_restored_budget_pause(
                held, _queue().DRIVE_ROOT, reason=(HOLD_INVALID_BUDGET_FENCE_SNAPSHOT if budget
                                                  else HOLD_INVALID_ACCEPTANCE_FENCE_SNAPSHOT),
                detail="the snapshot's fence evidence was malformed; the saved pause is "
                       "retained, not cancelled, until an explicit Resume re-validates it")
        with _queue()._queue_lock:
            _append_held_pending_row(held)
        retained.append(task_id)
    _queue().append_jsonl(
        _queue().DRIVE_ROOT / "logs" / "supervisor.jsonl",
        {"ts": utc_now_iso(), "type": ("queue_restore_invalid_budget_root_fences" if budget
                                      else "queue_restore_invalid_acceptance_fences"),
         "affected_task_ids": cancelled, "action": "fail_closed_no_restore",
         "retained_budget_paused_task_ids": retained, "skipped_terminal_task_ids": skipped_terminal},
    )
    try:
        for task in ([] if budget else snapshot_pending):
            task_id = str(task.get("id") or "")
            if task_id and task_id in cancelled:
                existing = load_task_result(_queue().DRIVE_ROOT, task_id) or {}
                write_task_result(
                    _queue().DRIVE_ROOT, task_id, STATUS_CANCELLED,
                    **_cancel_result_fields(
                        task, existing=existing,
                        result="Task was not restored because its acceptance-fence snapshot was invalid."),
                )
    except Exception:
        log.warning("Failed to terminalize tasks from invalid acceptance-fence snapshot", exc_info=True)
    return len(retained)


def _append_held_pending_row(task: dict) -> None:
    """Append one proven-revivable held row to PENDING with queue-order facts (lock held)."""
    if "_queue_seq" not in task:
        _queue().QUEUE_SEQ_COUNTER_REF["value"] += 1
        task["_queue_seq"] = _queue().QUEUE_SEQ_COUNTER_REF["value"]
    task.setdefault("queued_at", utc_now_iso())
    _queue().PENDING.append(task)
    _queue().sort_pending()


def _park_pausing_running_rows(running_rows: list, snapshot_pending: list) -> list:
    """RUNNING rows whose durable budget pause is LIVE become parked PENDING rows (#1196).

    The worker wrote the ``budget_pause`` row and stored its source BEFORE the
    park event left it; a shutdown in that window leaves a RUNNING row in the
    snapshot whose exact continuation is complete on disk. Handing that row to
    the shutdown cancel fence would cancel a saved pause, so it is parked under
    its exact marker instead: ``pausing``/``paused`` rows directly (the row is
    confirmed ``paused`` with ``pause_source=restart_during_pausing``), and a
    ``resume_granted`` row whose grant was never consumed (the loop writes
    ``consumed_at`` before any new effect) after its grant is revoked — a
    revocation that cannot be written keeps the row parked under a typed hold.
    A consumed grant is ordinary running work and keeps the ordinary fence; an
    unreadable pause authority proves nothing and keeps it too. Rows already in
    the pending snapshot are never duplicated.
    """
    from ouroboros.budget_pause import (
        STATE_PAUSED, STATE_PAUSING, STATE_RESUME_GRANTED, budget_pause_row, exact_pause_marker,
        set_budget_pause,
    )
    from supervisor.events_budget import HOLD_RESTART_REVOCATION_UNWRITTEN, hold_budget_row

    pending_ids = {str(row.get("id") or "") for row in snapshot_pending if isinstance(row, dict)}
    parked: list = []
    for row in running_rows if isinstance(running_rows, list) else []:
        if not isinstance(row, dict) or not isinstance(row.get("task"), dict):
            continue
        task = dict(row["task"])
        task_id = str(task.get("id") or "")
        if not task_id or task_id in pending_ids:
            continue
        attempt = int(row.get("attempt") or task.get("_attempt") or 1)
        result_root = pathlib.Path(task.get("budget_drive_root") or _queue().DRIVE_ROOT)
        try:
            pause = budget_pause_row(result_root, task_id)
        except Exception:
            continue
        if (not pause or int(pause.get("task_attempt") or 0) != attempt or not pause.get("source_ref")
                or pause.get("state") not in {STATE_PAUSING, STATE_PAUSED, STATE_RESUME_GRANTED}):
            continue
        pause_id = str(pause.get("pause_id") or "")
        grant = pause.get("grant") if isinstance(pause.get("grant"), dict) else {}
        held_reason = ""
        if pause.get("state") == STATE_RESUME_GRANTED:
            if grant.get("consumed_at") or grant.get("revoked_at"):
                continue
            revoked = {**grant, "revoked_at": utc_now_iso(), "revoke_reason": "restart_before_consumption"}
            try:
                set_budget_pause(result_root, task_id, {**pause, "state": STATE_PAUSED, "grant": revoked},
                                 expected_pause_id=pause_id)
                pause = {**pause, "state": STATE_PAUSED, "grant": revoked}
            except Exception:
                log.warning("Restart could not revoke the unconsumed grant of %s; parked under a hold",
                            task_id, exc_info=True)
                held_reason = HOLD_RESTART_REVOCATION_UNWRITTEN
        elif pause.get("state") == STATE_PAUSING:
            try:
                set_budget_pause(result_root, task_id, {**pause, "state": STATE_PAUSED,
                                                        "paused_confirmed_at": time.time(),
                                                        "pause_source": "restart_during_pausing"},
                                 expected_pause_id=pause_id)
            except Exception:
                log.warning("Parked pausing row %s stays 'pausing' (row unwritable at restore)",
                            task_id, exc_info=True)
        task["_attempt"] = attempt
        task.pop("_budget_pause_resume", None)
        task["_budget_pause"] = exact_pause_marker(pause, default_root=str(task.get("root_task_id") or task_id))
        if held_reason:
            hold_budget_row(
                task, reason=held_reason,
                detail="restart found an unconsumed Resume grant whose revocation could not be written",
                extra={"pause_id": pause_id, "grant_id": str(grant.get("grant_id") or ""),
                       "root_task_id": str(task.get("root_task_id") or task_id)},
                result_root=result_root)
        parked.append(task)
    if parked:
        _queue().append_jsonl(
            _queue().DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {"ts": utc_now_iso(), "type": "queue_restore_parked_budget_pauses",
             "task_ids": [str(task.get("id") or "") for task in parked]},
        )
    return parked


def _descends_from(task: Any, roots: "set[str]", pending_by_id: dict) -> bool:
    """Whether this row's lineage reaches ANY of ``roots``.

    The whole ancestry is walked, not just the immediate parent: a snapshot holds
    a tree, and a grandchild of an interrupted root is as unstartable as its
    child. ``root_task_id`` answers first because a deep row names its root
    directly; the parent chain is then followed through the snapshot's own rows,
    which is every ancestor a restore can resolve without reading disk.
    """
    if not roots or not isinstance(task, dict):
        return False
    if str(task.get("root_task_id") or "") in roots:
        return True
    current = task
    seen: set[str] = set()
    while isinstance(current, dict):
        parent_id = str(current.get("parent_task_id") or "")
        if not parent_id or parent_id in seen:
            return False
        if parent_id in roots:
            return True
        seen.add(parent_id)
        current = pending_by_id.get(parent_id)
    return False


def _interrupted_ancestors(fenced_running: "list[str]", snapshot_pending: list, *, unknown: set) -> "set[str]":
    """The ids whose interruption a restored PENDING child cannot survive.

    The rows this boot just fenced, plus the ancestors an EARLIER boot already
    handed to cancellation custody: a multi-boot stop settles the root first, so
    by the time the child is read again its parent is no longer a RUNNING row —
    only an active intent or a stored ``cancelled`` result with the shutdown
    cause proves what happened to it. Queued ancestors are read alike. ``unknown``
    collects unreadable, missing or never-admitted ones: a parent wrote its
    running result before spawning a child, so absence is never permission.
    """
    from ouroboros.cancel_intents import has_active_intent
    from ouroboros.task_results import STATUS_CANCELLED, load_task_result

    interrupted = set(fenced_running)
    candidates: set[str] = set()
    for task in snapshot_pending:
        if not isinstance(task, dict) or not str(task.get("parent_task_id") or ""):
            continue
        for key in ("parent_task_id", "root_task_id"):
            ancestor = str(task.get(key) or "")
            if ancestor and ancestor not in interrupted:
                candidates.add(ancestor)
    for ancestor in candidates:
        try:
            if has_active_intent(_queue().DRIVE_ROOT, ancestor, strict=True):
                interrupted.add(ancestor)
                continue
            stored = load_task_result(_queue().DRIVE_ROOT, ancestor, strict=True)
        except Exception:
            log.warning("Snapshot restore could not read ancestor %s", ancestor, exc_info=True)
            stored = None
        if not stored or stored.get("admission_outcome") == "never_admitted":
            unknown.add(ancestor)  # never a proven interruption: a scope-verifiable child waits
            continue
        origin = stored.get("cancel_origin")
        if (
            str(stored.get("status") or "") == STATUS_CANCELLED
            and isinstance(origin, dict)
            and str(origin.get("reason") or "") == "server_shutdown"
        ):
            interrupted.add(ancestor)
    return interrupted


def _record_queue_restore(
    *, restored: int = 0, skipped_terminal: int = 0,
    cancel_authority_holds: Optional[list] = None, blocked_admission: Optional[list] = None,
    invalid_task_depth: Optional[list] = None, terminalized_running: Optional[list] = None,
    pending_parent_interrupted: Optional[list] = None, direct_roots_incomplete: bool = False,
) -> None:
    """The one durable row a restore leaves: what it revived, what it left to
    cancellation custody, which surviving RUNNING rows it fenced, and which
    PENDING children it refused to start behind an interrupted parent. A stale
    snapshot with nothing to revive still records the fences it minted, and a
    direct-root roster that could not name its live turns is disclosed as the
    gap it is."""
    if not (restored or skipped_terminal or blocked_admission or terminalized_running
            or pending_parent_interrupted or direct_roots_incomplete):
        return
    _queue().append_jsonl(
        _queue().DRIVE_ROOT / "logs" / "supervisor.jsonl",
        {
            "ts": utc_now_iso(),
            "type": "queue_restored_from_snapshot",
            "restored_pending": restored,
            "skipped_terminal": skipped_terminal,
            "cancel_authority_holds": list(cancel_authority_holds or []),
            "blocked_admission": list(blocked_admission or []),
            "invalid_task_depth": list(invalid_task_depth or []),
            "terminalized_running": list(terminalized_running or []),
            "pending_parent_interrupted": list(pending_parent_interrupted or []),
            "direct_roots_incomplete": bool(direct_roots_incomplete),
        },
    )


def restore_pending_from_snapshot(
    max_age_sec: int = 900, *, terminalized: Optional[list] = None,
) -> int:
    """Restore recent pending tasks from queue snapshot.

    Returns the PENDING count revived; ``terminalized`` separately collects
    surviving RUNNING ids fenced with cancel intents for the caller to name.
    """
    if _queue().PENDING:
        return 0
    try:
        if not _queue().QUEUE_SNAPSHOT_PATH.exists():
            return 0
        snap = json.loads(_queue().QUEUE_SNAPSHOT_PATH.read_text(encoding="utf-8"))
        if not isinstance(snap, dict):
            return 0
        ts = str(snap.get("ts") or "")
        ts_unix = _queue().parse_iso_to_ts(ts)
        # Timestamp validity and freshness gate ORDINARY rows only (#1196): a
        # readable snapshot whose stamp is missing or malformed is treated as
        # stale, so an identifiable exact pause or an acknowledged owner-wait
        # handoff is still retained under its own durable authority instead of
        # vanishing (a Resume would then answer task_not_pending over an intact checkpoint).
        if ts_unix is None:
            _queue().append_jsonl(
                _queue().DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {"ts": utc_now_iso(), "type": "queue_restore_snapshot_timestamp_invalid",
                 "snapshot_ts": ts[:64], "action": "treated_as_stale"},
            )
        stale = ts_unix is None or (time.time() - ts_unix) > max_age_sec
        from ouroboros.project_admission import hold_unreadable_result
        from ouroboros.task_results import (
            _TRULY_TERMINAL_STATUSES, STATUS_CANCEL_REQUESTED, STATUS_CANCELLED,
            load_task_result, write_task_result,
        )
        raw_fences = snap.get("acceptance_fences", [])
        raw_budget_fences = snap.get("budget_root_fences", [])
        snapshot_pending = [
            row.get("task")
            for row in (snap.get("pending") or [])
            if isinstance(row, dict) and isinstance(row.get("task"), dict)
        ]
        running_rows = snap.get("running")
        running_rows = running_rows if isinstance(running_rows, list) else []
        # The pre-restart RUNNING rows are read HERE, before the stale gate: this
        # is the last moment the list exists, and a stale snapshot is exactly the
        # case where nothing else will ever settle them. Direct-chat roots never
        # reach the queue, so the roster `queue.init` took over is the only place
        # they are named; both lists describe work the stop caught, and one fence
        # call gives them the one cancel-intent path custody settles — except a
        # direct root whose exact budget pause was complete on disk, which is
        # parked under its own id like a paused RUNNING row (#1196).
        direct_roots = dict(_queue().PRIOR_DIRECT_ROOTS)
        # An in-process supervisor revival re-runs queue init while direct turns of THIS process are
        # alive: the roster then names live work, not what a stop caught.
        from supervisor.active_activity import get_direct_activity_registry
        live_direct = {str(row.get("activity_id") or "") for row in get_direct_activity_registry().snapshot()}
        pending_ids = {str(task.get("id") or "") for task in snapshot_pending}
        direct_caught = [task_id for task_id in direct_roots.get("task_ids") or []
                         if task_id not in live_direct and task_id not in pending_ids]
        snapshot_pending, parked_pausing, consumed_rows = _retain_snapshot_pending(
            snapshot_pending, running_rows, stale=stale, direct_caught=direct_caught)
        fenced_running = _fence_snapshot_running_rows(
            running_rows
            + [{"id": task_id} for task_id in direct_caught]
            # A stale row whose grant was consumed names work that ran on: the
            # ordinary fence, never a re-armed pause (revoke_exact_budget_resume).
            + [{"id": task_id} for task_id in consumed_rows],
            restored_ids={str(task.get("id") or "") for task in snapshot_pending},
        )
        if terminalized is not None:
            terminalized.extend(fenced_running)
        if stale and not snapshot_pending:
            _record_queue_restore(terminalized_running=fenced_running,
                                  direct_roots_incomplete=direct_roots.get("incomplete", False))
            return 0
        snapshot_pending, pending_by_id, restored = restore_terminalization_retry_rows(
            snapshot_pending, pending=_queue().PENDING, running=_queue().RUNNING,
            queue_seq_counter_ref=_queue().QUEUE_SEQ_COUNTER_REF, sort_pending=_queue().sort_pending,
        )
        fenced_roots, malformed_fences, malformed_budget_fences = restore_queue_fences(raw_fences, raw_budget_fences)
        if not malformed_budget_fences:
            _raise_parked_root_fences(parked_pausing)
        if malformed_budget_fences or malformed_fences:
            restored += _refuse_restore_invalid_fences(snapshot_pending, budget=malformed_budget_fences)
            _record_queue_restore(restored=restored, terminalized_running=fenced_running,
                                  direct_roots_incomplete=direct_roots.get("incomplete", False))
            if restored > 0:
                _queue().persist_queue_snapshot(reason="queue_restored")
            return restored

        skipped_terminal, invalid_depth_restore = 0, []
        cancel_authority_holds: list[str] = []
        skipped_fenced, blocked_restore, orphan_children, lineage_unknown = [], [], [], set()
        acceptance_held: list[str] = []
        interrupted = _interrupted_ancestors(fenced_running, snapshot_pending, unknown=lineage_unknown)
        for task in snapshot_pending:
            chat_id = task.get("chat_id")
            if not task.get("id") or chat_id is None or chat_id == "":
                continue
            if _descends_from(task, fenced_roots, pending_by_id) and _exact_pause_row(task):
                # #1196: an acceptance fence over the root cancels NOTHING that was
                # paused mid-run: retained under a typed hold, the ordinary revival
                # checks below still apply, and the row is appended directly.
                from supervisor.events_budget import HOLD_ROOT_ACCEPTANCE_FENCED, hold_restored_budget_pause

                task = hold_restored_budget_pause(
                    dict(task), _queue().DRIVE_ROOT, reason=HOLD_ROOT_ACCEPTANCE_FENCED,
                    detail="the root entered acceptance review; the saved pause is retained, not cancelled")
                acceptance_held.append(str(task.get("id") or ""))
            elif _descends_from(task, fenced_roots, pending_by_id):
                task_id = str(task.get("id") or "")
                skipped_fenced.append(task_id)
                try:
                    existing = load_task_result(_queue().DRIVE_ROOT, task_id) or {}
                    stored = write_task_result(
                        _queue().DRIVE_ROOT, task_id, STATUS_CANCELLED,
                        **_cancel_result_fields(task, existing=existing,
                            result="Task was not restored after restart because its root had entered acceptance review.",
                        ),
                    )
                    if not isinstance(stored, dict) or stored.get("status") != STATUS_CANCELLED:
                        raise RuntimeError("Acceptance cancellation was not confirmed")
                except Exception:
                    log.warning("Failed to terminalize fenced snapshot task %s", task_id, exc_info=True)
                    if task.get("project_id") or task.get("_project_admission_restore_hold"):
                        task["_terminalization_retry"] = {
                            "status": STATUS_CANCELLED, "trigger": "acceptance_fence",
                            "reason": "The root entered acceptance review; cancellation is pending.",
                            "reconcile_delegate_custody": False,
                        }
                        with _queue()._queue_lock:
                            _append_held_pending_row(task)
                        restored += 1
                continue
            # AR2-10 (§8-A1): restore and the intent check share the queue lock.
            with _queue()._queue_lock:
                skip_revival = False
                cancel_authority_unreadable = False
                try:
                    existing = load_task_result(_queue().DRIVE_ROOT, str(task.get("id")), strict=True)
                    existing_status = str(existing.get("status") or "") if existing else ""
                except Exception:
                    if _exact_pause_row(task):
                        # #1196: an unreadable result authority never terminalizes a
                        # saved pause. The row stays PENDING under its ORIGINAL
                        # locator and a typed hold (the earlier restore hold, if
                        # any, is kept); the next Resume re-reads the authority.
                        from ouroboros.budget_pause import RESTORE_REFUSAL_RECORD_UNREADABLE
                        from supervisor.events_budget import (
                            HOLD_RESTORE_REFUSED_PREFIX, budget_hold_fact, hold_restored_budget_pause,
                        )

                        if budget_hold_fact(task) is None:
                            task = hold_restored_budget_pause(
                                dict(task), _queue().DRIVE_ROOT,
                                reason=HOLD_RESTORE_REFUSED_PREFIX + RESTORE_REFUSAL_RECORD_UNREADABLE)
                        acceptance_held.append(str(task.get("id") or ""))
                        log.warning("Snapshot restore retained paused row %s under a hold: its "
                                    "result authority is unreadable", task.get("id"), exc_info=True)
                    if hold_unreadable_result(task) or _exact_pause_row(task):
                        # A receipt read failure cannot select a terminal policy for conserved
                        # work: its pause, its Project hold, or accepted Project work held here.
                        _append_held_pending_row(task)
                        restored += 1
                        continue
                    # Result-authority loss already has terminal custody: once
                    # its retry can prove a writable result, the task is failed
                    # rather than replayed over an unknown exact-id lifecycle.
                    task["_terminalization_retry"] = {
                        "reason": "Pending task result authority is unreadable; dispatch is blocked.",
                        "status": "failed",
                        "trigger": "pending_result_authority",
                        "reconcile_delegate_custody": False,
                    }
                    restore_terminalization_retry(
                        task, pending=_queue().PENDING, running=_queue().RUNNING,
                        queue_seq_counter_ref=_queue().QUEUE_SEQ_COUNTER_REF,
                        sort_pending=_queue().sort_pending,
                    )
                    skipped_terminal += 1
                    log.debug("Snapshot restore result-authority check failed for %s", task.get("id"), exc_info=True)
                    continue
                else:
                    if (not _exact_pause_row(task) and not task.get("_owner_wait_resume")
                            and not task.get("_project_admission_restore_hold")
                            and restored_handoff_unproven(task, existing, ancestor_unknown=_descends_from(task, lineage_unknown, pending_by_id))):
                        task["_project_admission_restore_hold"] = {
                            "reason": "project_routing_fence_lookup_failed",
                            "detail": "Original no-dispatch or parent evidence is unavailable; automatic recovery is not authorized.",
                        }
                    # Terminal OR cancel-intent — both must not be resurrected as
                    # pending. Intent lives in the durable projection (phase A);
                    # the status check covers legacy latch files.
                    if not isinstance(task.get("_terminalization_retry"), dict) and (existing_status in _TRULY_TERMINAL_STATUSES or existing_status == STATUS_CANCEL_REQUESTED):
                        skip_revival = True
                    elif not isinstance(task.get("_terminalization_retry"), dict):
                        try:
                            from ouroboros.cancel_intents import has_active_intent

                            if has_active_intent(
                                _queue().DRIVE_ROOT, str(task.get("id")), strict=True,
                            ):
                                # Cancellation custody owns it; never revive a pending row.
                                skip_revival = True
                        except Exception:
                            # This is an UNKNOWN cancel fact, not a terminal
                            # outcome. Restore the ordinary row under a durable,
                            # non-dispatchable hold; the pre-dispatch SSOT later
                            # resolves it after both authorities are readable.
                            task = dict(task)
                            task["_cancel_intent_authority_hold"] = {
                                "reason": "Cancel-intent authority is unreadable; dispatch is blocked.",
                                "held_at": utc_now_iso(),
                            }
                            cancel_authority_unreadable = True
                            log.debug("Snapshot restore cancel-intent authority check failed for %s",
                                      task.get("id"), exc_info=True)
                if skip_revival:
                    skipped_terminal += 1
                    continue
                if not cancel_authority_unreadable and str(task.get("parent_task_id") or "") and not _exact_pause_row(task) and _descends_from(
                    task, interrupted, pending_by_id
                ):
                    # #1104: a planned shutdown already refuses to start these
                    # (`kill_workers(preserve_pending=True)`); an unplanned one
                    # left them in the snapshot, where reviving one starts a child
                    # whose parent no longer exists. Everything above has just
                    # proved this row is otherwise revivable and unowned, so it
                    # takes the SAME shutdown-custody marker the planned path
                    # writes: the boot's own kill step settles it with a
                    # ledger-reconstructed cost and publishes its task_done. Rows
                    # without a parent — roots, schedules, evolution — never match;
                    # nor does an exact mid-run pause (#1196: saved work, not unstarted).
                    task = dict(task)
                    task["_terminalization_retry"] = {
                        "reason": "Parent task was interrupted before this child started.",
                        "status": STATUS_CANCELLED,
                        "trigger": "pending_parent_interrupted",
                        "reconcile_delegate_custody": True,
                    }
                    restore_terminalization_retry(
                        task, pending=_queue().PENDING, running=_queue().RUNNING,
                        queue_seq_counter_ref=_queue().QUEUE_SEQ_COUNTER_REF,
                        sort_pending=_queue().sort_pending,
                    )
                    orphan_children.append(str(task.get("id") or ""))
                    skipped_terminal += 1
                    continue
                if not cancel_authority_unreadable and (str(task.get("id") or "") in acceptance_held or task.get("_project_admission_restore_hold")):
                    # Proven revivable and unowned above. Both existing holds
                    # retain custody; assignment revalidates Project evidence.
                    _append_held_pending_row(task)
                    restored += 1
                    continue
                admitted = _queue().enqueue_task(task, restoring_snapshot=True)
                if isinstance(admitted, dict) and admitted.get("_admission_blocked"):
                    _queue().restore_invalid_depth_admission(task, admitted, drive_root=_queue().DRIVE_ROOT, pending=_queue().PENDING, blocked=blocked_restore, terminalized=invalid_depth_restore, queue_seq_counter_ref=_queue().QUEUE_SEQ_COUNTER_REF)
                    try:
                        _queue().sort_pending()
                    except (TypeError, ValueError, OverflowError):
                        log.warning("Deferred snapshot sort failed; custody retained", exc_info=True)
                    continue
                if cancel_authority_unreadable:
                    cancel_authority_holds.append(str(task.get("id") or ""))
            restored += 1
        if skipped_fenced or acceptance_held:
            _queue().append_jsonl(
                _queue().DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {
                    "ts": utc_now_iso(),
                    "type": "queue_restore_skipped_acceptance_fence",
                    "task_ids": skipped_fenced,
                    "held_budget_paused_task_ids": acceptance_held,
                    "root_task_ids": sorted(fenced_roots),
                },
            )
        _record_queue_restore(
            restored=restored, skipped_terminal=skipped_terminal,
            cancel_authority_holds=cancel_authority_holds,
            blocked_admission=blocked_restore, invalid_task_depth=invalid_depth_restore,
            terminalized_running=fenced_running, pending_parent_interrupted=orphan_children,
            direct_roots_incomplete=direct_roots.get("incomplete", False),
        )
        from supervisor.queue_transitions import sweep_orphaned_budget_fences

        sweep_orphaned_budget_fences(
            _queue().PENDING, _queue().BUDGET_ROOT_FENCES, _queue().DRIVE_ROOT,
        )
        if restored > 0 or skipped_terminal > 0 or invalid_depth_restore or blocked_restore:
            _queue().persist_queue_snapshot(reason="queue_restored")
        return restored
    except Exception:
        log.warning("Failed to restore pending queue from snapshot", exc_info=True)
        return 0
