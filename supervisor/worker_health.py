"""Crash detection and the terminal a host-side teardown publishes.

Distinguishes a worker that died from one that is merely slow, respects the spawn
grace window so a booting pool is not read as a crash storm, and publishes the
task_done the dying task can no longer publish for itself.
"""

from __future__ import annotations

import logging
import pathlib
import time
from typing import Any, Dict, List, Optional

from ouroboros.outcomes import (
    EXECUTION_FAILED,
    EXECUTION_INFRA_FAILED,
    terminal_outcome_axes,
)
from supervisor.queue import _queue_lock


def _pool():
    """The parent module, read at call time.

    The pool owns the repo/drive roots, its size, the worker table, the shared
    PENDING/RUNNING refs and the crash clock, and ``init`` REBINDS them.
    Reading them through the module is what keeps one binding: a from-import
    here would freeze the value this module saw at import time (the
    owner-approved D18/D33 mechanical exception).
    """
    from supervisor import workers

    return workers


log = logging.getLogger(__name__)


def terminal_task_metadata(task_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Project ONLY lifecycle-relevant metadata onto a terminal task_done event.

    Terminal events reach chat logs and the UI, so arbitrary task metadata
    (workspace paths, secret-bearing fields) must not ride along. Exactly two
    consumers need fields here: the evolution campaign tally reads
    ``evolution_transaction``, and the assisted-merge watchdog / writer-gate
    release in events._handle_task_done reads ``managed_update`` (its
    authority_fingerprint) — a reaped resolver task would otherwise leave the
    update tx orphaned and the writer gate latched until restart."""
    meta = task_metadata if isinstance(task_metadata, dict) else {}
    out: Dict[str, Any] = {}
    for key in ("evolution_transaction", "managed_update"):
        value = meta.get(key)
        if isinstance(value, dict):
            out[key] = dict(value)
    return out


def _emit_task_done_terminal(
    task: Optional[Dict[str, Any]],
    task_id: str,
    status: str = "failed",
    *,
    reason_code: str = "",
    cost_fields: Optional[Dict[str, Any]] = None,
) -> bool:
    """Emit a task_done event so the UI resolves the live card when a task is
    torn down outside the normal completion path (crash storm, kill, hard
    timeout). Without this the spinner spins forever on these paths.

    ``cost_fields`` is one whole ``reconstruct_task_cost(fields=True)`` projection,
    taken opaquely (as ``queue._emit_cancel_task_done`` already takes it) rather
    than re-declared field by field. Three times a key was added to that
    projection and a hand-maintained mirror here was missed; a signature that
    names no cost field cannot be missed again. Callers with no reconstructed
    cost pass nothing and the event says so instead of reporting zeros as fact."""
    if not task_id:
        return False
    try:
        chat_id = int((task or {}).get("chat_id") or 0)
    except (TypeError, ValueError):
        chat_id = 0
    status = status or "failed"
    # Caller reason_code wins; budget_exhausted -> EXECUTION_FAILED below, not infra-failure.
    reason_code = reason_code or ("worker_terminal_failure" if status == "failed" else status)
    task_metadata = (task or {}).get("metadata")
    task_metadata = task_metadata if isinstance(task_metadata, dict) else {}
    terminal_metadata = _pool().terminal_task_metadata(task_metadata)
    try:
        # Only the four keys whose EMISSION RULE differs are read by name: the
        # accounting verdict always rides, the two disclosure flags ride only
        # when they have something to disclose, and everything else rides only
        # when the accounting is available -- so an unavailable projection never
        # publishes its `None` placeholders as if they were measurements.
        projection: Dict[str, Any] = dict(cost_fields or {})
        emitted: Dict[str, Any] = {
            "cost_accounting_status": str(projection.pop("cost_accounting_status", "") or "unavailable"),
            "cost_final": bool(projection.pop("cost_final", False)),
        }
        accounting_error = projection.pop("cost_accounting_error", "")
        if accounting_error:
            emitted["cost_accounting_error"] = accounting_error
        if projection.pop("ledger_integrity_degraded", False):
            emitted["ledger_integrity_degraded"] = True
        if emitted["cost_accounting_status"] == "available":
            # Verbatim, unenumerated: cost_final's disclosed cause (non_final_rows)
            # rides here today for free, and so will the next field added upstream.
            emitted.update(projection)
        _pool().get_event_q().put({
            "type": "task_done",
            "task_id": str(task_id),
            "task_type": str((task or {}).get("type") or ""),
            "chat_id": chat_id,
            "status": status,
            "outcome_axes": terminal_outcome_axes(
                lifecycle=status,
                execution=(EXECUTION_FAILED if reason_code == "budget_exhausted" else EXECUTION_INFRA_FAILED) if status == "failed" else status,
                reason_code=reason_code,
                review_trigger="worker_terminal",
            ),
            "reason_code": reason_code,
            **({"metadata": terminal_metadata} if terminal_metadata else {}),
            **emitted,
        })
        return True
    except Exception:
        log.warning("Failed to emit terminal task_done for %s", task_id, exc_info=True)
        return False


def ensure_workers_healthy() -> None:
    """Detect dead workers, finalize/requeue their tasks, respawn.

    Runs under the queue lock: the RUNNING pops and respawn decisions here
    raced with HTTP cancel handlers (double respawn → orphaned worker, and
    "dict changed size" crashes in concurrent iteration). RLock keeps the
    nested enqueue/respawn/persist calls re-entrant.
    """
    from supervisor import queue
    # Workers need init time after spawn.
    if (time.time() - _pool()._LAST_SPAWN_TIME) < _pool()._SPAWN_GRACE_SEC:
        return
    with _queue_lock:
        respawn_ids, disable_pool = _pool()._ensure_workers_healthy_locked(queue)
    if disable_pool:
        # Every lifecycle operation takes lifecycle -> queue lock.  Calling
        # kill_workers while still holding queue lock would invert that order
        # against a concurrent respawn and deadlock.
        _pool().kill_workers(disable_reason="worker_crash_storm")
        _pool().CRASH_TS.clear()
        return
    for wid in respawn_ids:
        try:
            _pool().respawn_worker(wid)
        except Exception:
            log.warning("Failed to respawn crashed worker %d", wid, exc_info=True)
            with _queue_lock:
                slot = _pool().WORKERS.get(wid)
                if slot is not None:
                    slot.reaping = False
    if respawn_ids:
        queue.persist_queue_snapshot(reason="worker_respawn_after_crash")


def _ensure_workers_healthy_locked(queue: Any) -> tuple[List[int], bool]:
    busy_crashes = 0
    dead_detections = 0
    crashed_tasks = []
    respawn_ids: List[int] = []
    for wid, w in list(_pool().WORKERS.items()):
        # The reaper owns marked slots through replacement; never double-respawn them.
        if getattr(w, "reaping", False):
            continue
        if not w.proc.is_alive():
            # Reserve the dead slot before the queue lock is released.
            w.reaping = True
            _pool()._reconcile_confirmed_dead_review_owner(
                int(getattr(w.proc, "pid", 0) or 0)
            )
            dead_detections += 1
            if w.busy_task_id is not None:
                busy_crashes += 1
            exitcode = w.proc.exitcode
            meta = _pool().RUNNING.get(w.busy_task_id, {}) if w.busy_task_id else {}
            task_info = meta.get("task", {}) if isinstance(meta, dict) else {}
            _pool().append_jsonl(
                _pool().DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {
                    "ts": _pool().utc_now_iso(),
                    "type": "worker_dead_detected",
                    "worker_id": wid,
                    "exitcode": exitcode,
                    "busy_task_id": w.busy_task_id,
                    "task_type": task_info.get("type") if isinstance(task_info, dict) else None,
                    "task_description": (task_info.get("description", "") or "")[:200] if isinstance(task_info, dict) else None,
                    "uptime_sec": round(time.time() - meta["started_at"]) if isinstance(meta, dict) and meta.get("started_at") else None,
                    "attempt": meta.get("attempt") if isinstance(meta, dict) else None,
                    "signal": -exitcode if isinstance(exitcode, int) and exitcode < 0 else None,
                },
            )
            if w.busy_task_id and isinstance(meta, dict) and meta.get("task"):
                crashed_tasks.append({"task_id": w.busy_task_id, "task_type": task_info.get("type") if isinstance(task_info, dict) else None})
                _pool().append_jsonl(
                    _pool().DRIVE_ROOT / "logs" / "supervisor.jsonl",
                    {
                        "ts": _pool().utc_now_iso(),
                        "type": "worker_crash_task_dump",
                        "worker_id": wid,
                        "task": meta["task"],
                        "started_at": meta.get("started_at"),
                        "last_heartbeat_at": meta.get("last_heartbeat_at"),
                        "attempt": meta.get("attempt"),
                    },
                )
            if w.busy_task_id and w.busy_task_id in _pool().RUNNING:
                meta = _pool().RUNNING.pop(w.busy_task_id) or {}
                try:
                    from ouroboros.tools.services import archive_task_service_logs
                    task_for_roots = meta.get("task") if isinstance(meta, dict) and isinstance(meta.get("task"), dict) else {}
                    archive_task_service_logs(pathlib.Path(_pool().DRIVE_ROOT), str(w.busy_task_id), task_for_roots)
                except Exception:
                    log.debug("Failed to archive service logs for task %s", w.busy_task_id, exc_info=True)
                task = meta.get("task") if isinstance(meta, dict) else None
                if isinstance(task, dict):
                    task_type = str(task.get("type") or "")
                    # Signal crashes are terminal infrastructure failures for every task type.
                    is_crash_signal = isinstance(exitcode, int) and exitcode < 0
                    crash_signal = -exitcode if is_crash_signal else None
                    chat_id = _pool().coerce_chat_identity(task.get("chat_id"), 0)
                    attempt = int(task.get("_attempt") or 1)
                    # Reconstruct cost/rounds from durable llm_usage for any
                    # abnormal-termination rollup below (worker died pre-finalize,
                    # so the event would otherwise carry zeros).
                    r_cost_fields = _pool().reconstruct_task_cost(str(w.busy_task_id), fields=True)

                    # Already terminal via inline/direct-chat path? Leave it.
                    already_done = False
                    existing_status = ""
                    try:
                        from ouroboros.task_results import load_task_result, _TRULY_TERMINAL_STATUSES
                        existing = load_task_result(_pool().DRIVE_ROOT, str(w.busy_task_id))
                        if existing and str(existing.get("status") or "") in _TRULY_TERMINAL_STATUSES:
                            already_done = True
                            existing_status = str(existing.get("status") or "")
                            log.info(
                                "Skipping requeue for task %s — already in terminal state: %s",
                                w.busy_task_id, existing.get("status"),
                            )
                    except Exception:
                        log.debug("Failed to check existing result for %s", w.busy_task_id, exc_info=True)

                    if already_done:
                        # Terminal on disk but the worker died — its normal task_done
                        # event may have been lost with it. Emit an (idempotent)
                        # terminal event so the live card resolves instead of
                        # spinning until reconnect/history reconciliation.
                        _pool()._emit_task_done_terminal(task, str(w.busy_task_id), existing_status or "completed")
                    elif is_crash_signal or attempt > _pool().QUEUE_MAX_RETRIES:
                        deep = task_type == "deep_self_review"
                        if is_crash_signal:
                            log.warning(
                                "Task %s worker crashed with signal %s — terminal (no retry)",
                                w.busy_task_id, crash_signal,
                            )
                            result_text = (
                                f"❌ {'Deep self-review ' if deep else ''}worker process crashed "
                                f"(signal {crash_signal}). This is an infrastructure/platform crash "
                                "and is not retried automatically. "
                                + (
                                    "Use /restart and then /review to retry after a clean restart."
                                    if deep else
                                    "Use /restart and try again; if it recurs it is a platform-level issue."
                                )
                            )
                            reason_code = "worker_crash_signal"
                        else:
                            log.warning(
                                "Task %s exceeded crash retry limit (%d/%d) — marking failed",
                                w.busy_task_id, attempt, _pool().QUEUE_MAX_RETRIES,
                            )
                            result_text = (
                                f"❌ Task failed after {attempt} crash(es) (exit {exitcode}). "
                                "Worker process died repeatedly — likely a platform-level issue. "
                                "Please try again or use a different approach."
                            )
                            reason_code = "worker_crash_retry_exhausted"
                        try:
                            from ouroboros.task_results import STATUS_FAILED, write_task_result
                            write_task_result(
                                _pool().DRIVE_ROOT, str(w.busy_task_id), STATUS_FAILED,
                                result=result_text,
                                reason_code=reason_code,
                                outcome_axes=terminal_outcome_axes(lifecycle=STATUS_FAILED, execution=EXECUTION_INFRA_FAILED, reason_code=reason_code, review_trigger="worker_terminal"),
                                crash_signal=crash_signal,
                                crash_exitcode=exitcode if isinstance(exitcode, int) else None,
                                **r_cost_fields,
                            )
                        except Exception:
                            log.debug("Failed to write failed status for %s", w.busy_task_id, exc_info=True)
                        # Message before task_done: otherwise the UI may close the card first.
                        try:
                            if is_crash_signal and deep:
                                user_msg = (
                                    f"❌ Deep self-review failed: worker process crashed (signal {crash_signal}). "
                                    "This is a known platform fork-safety limitation. "
                                    "Please use `/restart` and then `/review` to retry with a fresh process."
                                )
                            elif is_crash_signal:
                                user_msg = (
                                    f"❌ Task `{str(w.busy_task_id)[:8]}` failed: worker process crashed "
                                    f"(signal {crash_signal}). This is an infrastructure crash and was not retried."
                                )
                            else:
                                user_msg = (
                                    f"❌ Task `{str(w.busy_task_id)[:8]}` failed after {attempt} crash(es). "
                                    "Worker process crashed repeatedly. Please try again."
                                )
                            incident_task_id = str(w.busy_task_id or "")
                            _pool().send_with_budget(
                                chat_id,
                                user_msg,
                                is_progress=True,
                                task_id=incident_task_id,
                                progress_meta={
                                    "task_incident": reason_code,
                                    "toast_once": f"{incident_task_id}:{reason_code}:{attempt}",
                                },
                            )
                        except Exception:
                            log.debug("Failed to send failure message for %s", w.busy_task_id, exc_info=True)
                        _pool()._emit_task_done_terminal(
                            task, str(w.busy_task_id), "failed",
                            reason_code=reason_code, cost_fields=r_cost_fields,
                        )
                        from ouroboros.delegate_recovery import reconcile_unrecoverable_task
                        reconcile_unrecoverable_task(_pool().DRIVE_ROOT, str(w.busy_task_id))
                    elif task_type == "evolution" and not bool(_pool().load_state().get("evolution_mode_enabled")):
                        # Evolution was stopped: do not resurrect a dead evolution
                        # worker into another cycle (mirrors the hard-timeout gate
                        # in queue.enforce_task_timeouts).
                        try:
                            from ouroboros.task_results import STATUS_CANCELLED, write_task_result
                            write_task_result(
                                _pool().DRIVE_ROOT, str(w.busy_task_id), STATUS_CANCELLED,
                                result="Evolution worker died after the campaign was stopped; not retried.",
                                reason_code="evolution_stopped_no_retry",
                                outcome_axes=terminal_outcome_axes(lifecycle=STATUS_CANCELLED, execution="cancelled", reason_code="evolution_stopped_no_retry", review_trigger="worker_terminal"),
                                **r_cost_fields,
                            )
                        except Exception:
                            log.debug("Failed to write cancelled status for %s", w.busy_task_id, exc_info=True)
                        _pool()._emit_task_done_terminal(
                            task, str(w.busy_task_id), "cancelled",
                            cost_fields=r_cost_fields,
                        )
                        from ouroboros.delegate_recovery import reconcile_unrecoverable_task
                        reconcile_unrecoverable_task(_pool().DRIVE_ROOT, str(w.busy_task_id))
                    else:
                        task = dict(task)
                        task["_attempt"] = attempt + 1
                        from ouroboros.delegate_recovery import prepare_worker_crash_handoff
                        recovery_handoff = prepare_worker_crash_handoff(
                            _pool().DRIVE_ROOT, task, old_attempt=attempt, new_attempt=attempt + 1,
                            worker_id=wid,
                            exitcode=exitcode if isinstance(exitcode, int) else None,
                        )
                        try:
                            from ouroboros.task_results import STATUS_INTERRUPTED, write_task_result
                            write_task_result(
                                _pool().DRIVE_ROOT, str(w.busy_task_id), STATUS_INTERRUPTED,
                                result=f"Worker process died mid-task (attempt {attempt}). Retrying.",
                                **r_cost_fields,
                            )
                        except Exception:
                            log.debug("Failed to write interrupted status for %s", w.busy_task_id, exc_info=True)
                        try:
                            from ouroboros.owner_hurry import retry_reset

                            retry_reset(
                                queue._task_drive_for_task(task, str(w.busy_task_id)),
                                _pool().DRIVE_ROOT, str(w.busy_task_id),
                                reason="worker_crash_requeue",
                            )
                        except Exception:
                            log.debug("Crash-requeue retry reset failed for %s", w.busy_task_id, exc_info=True)
                        admitted = queue.enqueue_task(task, front=True)
                        admission_block = (
                            str(admitted.get("_admission_blocked") or "")
                            if isinstance(admitted, dict) else ""
                        )
                        if admission_block:
                            from ouroboros.delegate_recovery import veto_worker_retry_handoff
                            veto_worker_retry_handoff(
                                _pool().DRIVE_ROOT, str(w.busy_task_id), recovery_handoff, admission_block,
                            )
                            reason_code = "worker_crash_retry_admission_blocked"
                            try:
                                from ouroboros.task_results import STATUS_FAILED, write_task_result
                                write_task_result(
                                    _pool().DRIVE_ROOT,
                                    str(w.busy_task_id),
                                    STATUS_FAILED,
                                    result=(
                                        "Worker crashed and its retry was blocked by the active "
                                        f"{admission_block} admission fence."
                                    ),
                                    reason_code=reason_code,
                                    outcome_axes=terminal_outcome_axes(
                                        lifecycle=STATUS_FAILED,
                                        execution=EXECUTION_INFRA_FAILED,
                                        reason_code=reason_code,
                                        review_trigger="worker_terminal",
                                    ),
                                    **r_cost_fields,
                                )
                            except Exception:
                                log.debug(
                                    "Failed to terminalize admission-blocked retry for %s",
                                    w.busy_task_id,
                                    exc_info=True,
                                )
                            _pool()._emit_task_done_terminal(
                                task,
                                str(w.busy_task_id),
                                "failed",
                                reason_code=reason_code,
                                cost_fields=r_cost_fields,
                            )
            respawn_ids.append(wid)

    disable_pool = _pool()._worker_crash_storm_detected(
        busy_crashes=busy_crashes,
        dead_detections=dead_detections,
        crashed_tasks=crashed_tasks,
    )
    return respawn_ids, disable_pool
