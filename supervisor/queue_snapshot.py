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
from typing import Optional

from ouroboros.contracts.schema_versions import SCHEMA_VERSION_KEY
from ouroboros.utils import utc_now_iso
from supervisor.task_admission import (
    restore_terminalization_retry,
    restore_terminalization_retry_rows,
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
            worker_pool_disabled_reason = str(
                getattr(_workers_mod, "_WORKER_POOL_DISABLED_REASON", "") or ""
            )
            reaping_count = sum(1 for _w in _ws if getattr(_w, "reaping", False))
            assignable_idle_workers = sum(
                1 for _w in _ws
                if getattr(_w, "busy_task_id", None) is None and not getattr(_w, "reaping", False)
            )
        except Exception:
            worker_total = 0
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
                "allowed_resources": t.get("allowed_resources"), "deadline_at": t.get("deadline_at"),
                "task_contract": t.get("task_contract"),
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
                "memory_mode": t.get("memory_mode"), "drive_root": t.get("drive_root"), "parent_cognitive_route": t.get("parent_cognitive_route"), "subagent_availability": t.get("subagent_availability"),
                "child_drive_root": t.get("child_drive_root"),
                "budget_drive_root": t.get("budget_drive_root"),
                "task_constraint": t.get("task_constraint"), "predecessor_authority_source": t.get("predecessor_authority_source"),
                "metadata": t.get("metadata"), "origin_message_ref": t.get("origin_message_ref"),
                "origin_message_text": t.get("origin_message_text"), "_attempt": t.get("_attempt"),
                "review_reason": t.get("review_reason"), "review_source_task_id": t.get("review_source_task_id"),
                "_budget_pause": t.get("_budget_pause"), "budget_resumed_at": t.get("budget_resumed_at"), "_terminalization_retry": t.get("_terminalization_retry"),
                "_cancel_intent_authority_hold": t.get("_cancel_intent_authority_hold"),
            },
        })
    running_rows = []
    now = time.time()
    for task_id, meta in running_items:
        task = meta.get("task") if isinstance(meta, dict) else {}
        started = float(meta.get("started_at") or 0.0) if isinstance(meta, dict) else 0.0
        hb = float(meta.get("last_heartbeat_at") or 0.0) if isinstance(meta, dict) else 0.0
        running_rows.append({
            "id": task_id, "type": task.get("type"), "priority": task.get("priority"),
            "attempt": meta.get("attempt"), "worker_id": meta.get("worker_id"),
            "runtime_sec": round(max(0.0, now - started), 2) if started > 0 else 0.0,
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


def restore_pending_from_snapshot(max_age_sec: int = 900) -> int:
    """Restore recent pending tasks from queue snapshot."""
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
        if ts_unix is None:
            return 0
        if (time.time() - ts_unix) > max_age_sec:
            return 0
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
        snapshot_pending, pending_by_id, restored = restore_terminalization_retry_rows(
            snapshot_pending, pending=_queue().PENDING, running=_queue().RUNNING,
            queue_seq_counter_ref=_queue().QUEUE_SEQ_COUNTER_REF, sort_pending=_queue().sort_pending,
        )
        fenced_roots, malformed_fences, malformed_budget_fences = restore_queue_fences(raw_fences, raw_budget_fences)
        if malformed_budget_fences:
            _queue().append_jsonl(
                _queue().DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {"ts": utc_now_iso(), "type": "queue_restore_invalid_budget_root_fences",
                 "action": "fail_closed_no_restore"},
            )
            return restored
        if malformed_fences:
            affected = [str(task.get("id") or "") for task in snapshot_pending if task.get("id")]
            _queue().append_jsonl(
                _queue().DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {
                    "ts": utc_now_iso(),
                    "type": "queue_restore_invalid_acceptance_fences",
                    "affected_task_ids": affected,
                    "action": "fail_closed_no_restore",
                },
            )
            try:
                for task in snapshot_pending:
                    task_id = str(task.get("id") or "")
                    if task_id:
                        existing = load_task_result(_queue().DRIVE_ROOT, task_id) or {}
                        write_task_result(
                            _queue().DRIVE_ROOT,
                            task_id,
                            STATUS_CANCELLED,
                            **_cancel_result_fields(
                                task,
                                existing=existing,
                                result="Task was not restored because its acceptance-fence snapshot was invalid.",
                            ),
                        )
            except Exception:
                log.warning("Failed to terminalize tasks from invalid acceptance-fence snapshot", exc_info=True)
            return restored

        skipped_terminal, invalid_depth_restore = 0, []
        cancel_authority_holds: list[str] = []
        skipped_fenced, blocked_restore = [], []
        for task in snapshot_pending:
            chat_id = task.get("chat_id")
            if not task.get("id") or chat_id is None or chat_id == "":
                continue
            fenced = False
            for fenced_root in fenced_roots:
                if str(task.get("root_task_id") or "") == fenced_root:
                    fenced = True
                    break
                current = task
                seen: set[str] = set()
                while isinstance(current, dict):
                    parent_id = str(current.get("parent_task_id") or "")
                    if not parent_id or parent_id in seen:
                        break
                    if parent_id == fenced_root:
                        fenced = True
                        break
                    seen.add(parent_id)
                    current = pending_by_id.get(parent_id)
                if fenced:
                    break
            if fenced:
                task_id = str(task.get("id") or "")
                skipped_fenced.append(task_id)
                try:
                    existing = load_task_result(_queue().DRIVE_ROOT, task_id) or {}
                    write_task_result(
                        _queue().DRIVE_ROOT,
                        task_id,
                        STATUS_CANCELLED,
                        **_cancel_result_fields(
                            task,
                            existing=existing,
                            result="Task was not restored after restart because its root had entered acceptance review.",
                        ),
                    )
                except Exception:
                    log.warning("Failed to terminalize fenced snapshot task %s", task_id, exc_info=True)
                continue
            # AR2-10 (§8-A1): restore and the intent check share the queue lock.
            with _queue()._queue_lock:
                skip_revival = False
                try:
                    existing = load_task_result(_queue().DRIVE_ROOT, str(task.get("id")), strict=True)
                    existing_status = str(existing.get("status") or "") if existing else ""
                except Exception:
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
                    log.debug(
                        "Snapshot restore result-authority check failed for %s",
                        task.get("id"),
                        exc_info=True,
                    )
                    continue
                else:
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
                            admitted = _queue().enqueue_task(task, restoring_snapshot=True)
                            if isinstance(admitted, dict) and admitted.get("_admission_blocked"):
                                _queue().restore_invalid_depth_admission(
                                    task, admitted, drive_root=_queue().DRIVE_ROOT,
                                    pending=_queue().PENDING, blocked=blocked_restore,
                                    terminalized=invalid_depth_restore,
                                    queue_seq_counter_ref=_queue().QUEUE_SEQ_COUNTER_REF,
                                )
                                try:
                                    _queue().sort_pending()
                                except (TypeError, ValueError, OverflowError):
                                    log.warning(
                                        "Deferred snapshot sort failed; custody retained",
                                        exc_info=True,
                                    )
                            else:
                                restored += 1
                                cancel_authority_holds.append(str(task.get("id") or ""))
                            log.debug(
                                "Snapshot restore cancel-intent authority check failed for %s",
                                task.get("id"),
                                exc_info=True,
                            )
                            continue
                if skip_revival:
                    skipped_terminal += 1
                    continue
                admitted = _queue().enqueue_task(task, restoring_snapshot=True)
                if isinstance(admitted, dict) and admitted.get("_admission_blocked"):
                    _queue().restore_invalid_depth_admission(task, admitted, drive_root=_queue().DRIVE_ROOT, pending=_queue().PENDING, blocked=blocked_restore, terminalized=invalid_depth_restore, queue_seq_counter_ref=_queue().QUEUE_SEQ_COUNTER_REF)
                    try:
                        _queue().sort_pending()
                    except (TypeError, ValueError, OverflowError):
                        log.warning("Deferred snapshot sort failed; custody retained", exc_info=True)
                    continue
            restored += 1
        if skipped_fenced:
            _queue().append_jsonl(
                _queue().DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {
                    "ts": utc_now_iso(),
                    "type": "queue_restore_skipped_acceptance_fence",
                    "task_ids": skipped_fenced,
                    "root_task_ids": sorted(fenced_roots),
                },
            )
        if restored > 0 or skipped_terminal > 0 or blocked_restore:
            _queue().append_jsonl(
                _queue().DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {
                    "ts": utc_now_iso(),
                    "type": "queue_restored_from_snapshot",
                    "restored_pending": restored,
                    "skipped_terminal": skipped_terminal,
                    "cancel_authority_holds": cancel_authority_holds,
                    "blocked_admission": blocked_restore, "invalid_task_depth": invalid_depth_restore,
                },
            )
        from supervisor.queue_transitions import sweep_orphaned_budget_fences

        sweep_orphaned_budget_fences(
            _queue().PENDING, _queue().BUDGET_ROOT_FENCES, _queue().DRIVE_ROOT,
        )
        if restored > 0 or skipped_terminal > 0 or invalid_depth_restore:
            _queue().persist_queue_snapshot(reason="queue_restored")
        return restored
    except Exception:
        log.warning("Failed to restore pending queue from snapshot", exc_info=True)
        return 0
