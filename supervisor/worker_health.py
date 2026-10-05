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
from supervisor.log_addressing import resolve_project_chat
from supervisor.queue import _queue_lock
from supervisor.state import control_is


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
    """Reserve dead slots; the existing reaper owns file work and recovery."""
    from supervisor import queue
    from supervisor.task_reaper import retry_terminal_file_recoveries

    retry_terminal_file_recoveries()
    if _pool().disable_exhausted_worker_pool():
        return
    if (time.time() - _pool()._LAST_SPAWN_TIME) < _pool()._SPAWN_GRACE_SEC:
        return
    queue._ensure_reaper_started()
    with _queue_lock:
        _pool()._ensure_workers_healthy_locked(queue)


def _ensure_workers_healthy_locked(queue: Any) -> tuple[List[int], bool]:
    """Capture exact task/slot ownership before handing off any dead worker."""
    busy_crashes = 0
    crashed_tasks = []
    jobs = []
    for wid, w in list(_pool().WORKERS.items()):
        meta = _pool().RUNNING.get(w.busy_task_id) if w.busy_task_id else None
        if getattr(w, "reaping", False) or (
            isinstance(meta, dict) and isinstance(meta.get("_terminal_file_recovery"), dict)
        ):
            continue
        if w.proc.is_alive():
            continue
        w.reaping = True
        task = dict(meta.get("task") or {}) if isinstance(meta, dict) else {}
        attempt = int((meta or {}).get("attempt") or task.get("_attempt") or 1)
        exitcode = w.proc.exitcode
        busy_crashes += int(w.busy_task_id is not None)
        _pool().append_jsonl(_pool().DRIVE_ROOT / "logs" / "supervisor.jsonl", {
            "ts": _pool().utc_now_iso(), "type": "worker_dead_detected", "worker_id": wid,
            "exitcode": exitcode, "busy_task_id": w.busy_task_id,
            "task_type": task.get("type"), "task_description": str(task.get("description") or "")[:200],
            "uptime_sec": round(time.time() - meta["started_at"]) if meta and meta.get("started_at") else None,
            "attempt": (meta or {}).get("attempt"),
            "signal": -exitcode if isinstance(exitcode, int) and exitcode < 0 else None,
        })
        if task:
            crashed_tasks.append({"task_id": w.busy_task_id, "task_type": task.get("type")})
            _pool().append_jsonl(_pool().DRIVE_ROOT / "logs" / "supervisor.jsonl", {
                "ts": _pool().utc_now_iso(), "type": "worker_crash_task_dump", "worker_id": wid,
                "task": task, "started_at": meta.get("started_at"),
                "last_heartbeat_at": meta.get("last_heartbeat_at"), "attempt": meta.get("attempt"),
            })
        jobs.append({
            "kind": "confirmed_dead_worker", "worker_id": wid, "worker": w,
            "task_id": str(w.busy_task_id or ""), "meta": meta, "task": task,
            "attempt": attempt, "exitcode": exitcode, "drive_root": str(_pool().DRIVE_ROOT),
        })
    disable_pool = bool(jobs) and _pool()._worker_crash_storm_detected(
        busy_crashes=busy_crashes, dead_detections=len(jobs), crashed_tasks=crashed_tasks,
    )
    for job in jobs:
        job["skip_respawn"] = disable_pool
    if disable_pool:
        jobs.append({"kind": "worker_crash_storm", "workers": tuple(_pool().WORKERS.items()),
                     "drive_root": str(_pool().DRIVE_ROOT)})
    for job in jobs:
        try:
            queue._reap_queue.put(job)
        except Exception:
            # No handoff happened. Keep RUNNING intact for the next health tick.
            worker = job.get("worker")
            if worker is not None and _pool().WORKERS.get(job["worker_id"]) is worker:
                worker.reaping = False
            raise
    # Kept for private callers; all replacements/storm stops are queued now.
    return [], disable_pool


def _dead_job_is_current(job: dict) -> bool:
    """Called under queue lock; a slot id alone never identifies a generation."""
    w, task_id, meta = job["worker"], job["task_id"], job["meta"]
    if (str(_pool().DRIVE_ROOT) != job["drive_root"]
            or _pool().WORKERS.get(job["worker_id"]) is not w
            or str(w.busy_task_id or "") != task_id or w.proc.is_alive()):
        return False
    current = _pool().RUNNING.get(task_id) if task_id else None
    if current is not meta:
        return False
    if isinstance(meta, dict):
        task = meta.get("task") or {}
        return (not isinstance(meta.get("_terminal_file_recovery"), dict)
                and meta.get("worker_id", job["worker_id"]) == job["worker_id"]
                and int(meta.get("attempt") or task.get("_attempt") or 1) == job["attempt"]
                and int(task.get("_attempt") or 1) == int(job["task"].get("_attempt") or 1))
    return True


def recover_confirmed_dead_worker(job: dict) -> None:
    """Recover exact saved terminal bytes before applying the old crash policy."""
    from ouroboros.headless import prepare_terminal_task_files, terminal_task_files_ready
    from ouroboros.task_results import load_task_result, _TRULY_TERMINAL_STATUSES
    from supervisor import queue
    from supervisor.task_reaper import (
        _finish_self_finalized_task, _respawn_after_reap, TerminalFileRecoveryPending,
    )
    from supervisor.worker_pool_lifecycle import _WORKER_LIFECYCLE_LOCK

    with _queue_lock:
        if not _dead_job_is_current(job):
            return
    w, task_id, task = job["worker"], job["task_id"], job["task"]
    root = pathlib.Path(job["drive_root"])
    _pool()._reconcile_confirmed_dead_review_owner(int(getattr(w.proc, "pid", 0) or 0))
    if task_id and task:
        try:
            from ouroboros.tools.services import archive_task_service_logs
            archive_task_service_logs(root, task_id, task)
        except Exception:
            log.debug("Failed to archive service logs for task %s", task_id, exc_info=True)
        prepared = prepare_terminal_task_files(root, task)
        try:
            current = load_task_result(root, task_id, strict=True)
        except (OSError, ValueError, RuntimeError) as exc:
            raise TerminalFileRecoveryPending("CURRENT terminal result is unreadable") from exc
        ready = (isinstance(current, dict)
                 and str(current.get("status") or "") in _TRULY_TERMINAL_STATUSES
                 and terminal_task_files_ready(root, task, current))
        with _queue_lock:
            if not _dead_job_is_current(job):
                return
            _retire_dead_model_consumers(job)
        if ready:
            _finish_self_finalized_task(
                queue, _pool(), task, task_id, str(task.get("type") or ""),
                str(current["status"]), current, terminal_task_metadata(task.get("metadata")),
                worker_id=job["worker_id"], files_prepared_attempt=job["attempt"],
            )
        elif prepared.get("terminal_source_present") is not False:
            # Actual source may exist only on child, or its read was unknown.
            # Retain this exact job/reservation for the ordinary health retry.
            raise TerminalFileRecoveryPending("terminal files are not yet publishable")
        else:
            _recover_crashed_task_without_terminal(job, queue)
    if not job.get("skip_respawn"):
        # Hold only the lifecycle serializer around the identity check + existing
        # respawn. Its process launch deliberately runs outside queue lock.
        with _WORKER_LIFECYCLE_LOCK:
            with _queue_lock:
                owned = (_pool().WORKERS.get(job["worker_id"]) is w
                         and str(_pool().DRIVE_ROOT) == job["drive_root"])
            if owned:
                _respawn_after_reap(queue, _pool(), job["worker_id"], expected_worker=w)


def retire_confirmed_worker_consumers(worker: Any, meta: dict | None) -> None:
    """Reuse exact current pool ownership at every confirmed kill/join door.

    Call under the queue lock before removing the captured RUNNING/worker row.
    """
    if not isinstance(meta, dict) or not isinstance(meta.get("task"), dict):
        return
    task = meta["task"]
    _retire_dead_model_consumers({"worker": worker, "worker_id": worker.wid,
        "task_id": str(task.get("id") or ""), "task": task, "meta": meta,
        "attempt": int(meta.get("attempt") or task.get("_attempt") or 1),
        "drive_root": str(_pool().DRIVE_ROOT)})


def _retire_dead_model_consumers(job: dict, *, captured_timeout: bool = False) -> None:
    """Queue-locked exact death proof, never PID absence or task terminality.

    Timeout transfers ownership from RUNNING to its captured reaper job; the
    other doors retain RUNNING until retirement. Each uses its existing current-
    owner check, plus the same retained birth/Process/task/attempt proof below.
    Unreadable/legacy evidence stays held; money and other custody are untouched.
    """
    worker = job["worker"]
    birth = getattr(worker, "process_birth", "")
    if captured_timeout:
        from supervisor import queue
        from supervisor.task_reaper import _timeout_job_is_current
        current = _timeout_job_is_current(job, queue, _pool())
    else:
        current = _dead_job_is_current(job)
    if (not isinstance(birth, str) or not birth or worker.proc.exitcode is None
            or worker.proc.is_alive() or not current):
        return
    if (not isinstance(job.get("meta"), dict)
            or job["meta"].get("task", {}).get("id") != job["task_id"]
            or job["task"].get("id") != job["task_id"]):
        return
    try:
        from ouroboros.tool_custody import retire_tool_invocations
        retire_tool_invocations(
            pathlib.Path(job["task"].get("budget_drive_root") or job["drive_root"]),
            job["task_id"], str(job["task"].get("root_task_id") or job["task_id"]),
            pid=worker.proc.pid, process_birth=birth, task_attempt=job["attempt"])
    except Exception:
        log.warning("Confirmed worker death could not retire tool invocations for %s", job["task_id"], exc_info=True)
    try:
        from ouroboros.usage_accounting import _memoized_final_rows
        from ouroboros.model_wait import retire_model_consumers

        root = pathlib.Path(job["task"].get("budget_drive_root") or job["drive_root"])
        rows, integrity, _memo, _generation = _memoized_final_rows(root)
        if not integrity:
            raise ValueError("model consumer death custody unreadable")
        consumers = {row["local_answer_consumer_id"]: job["attempt"] for row in rows
                     if row.get("task_id") == job["task_id"]
                     and row.get("root_task_id") == (job["task"].get("root_task_id") or job["task_id"])
                     and row.get("local_answer_owner_pid") == worker.proc.pid
                     and row.get("local_answer_owner_birth") == birth
                     and type(row.get("local_answer_task_attempt")) is int
                     and row["local_answer_task_attempt"] == job["attempt"]
                     and isinstance(row.get("local_answer_consumer_id"), str) and row["local_answer_consumer_id"]}
        if consumers:
            retire_model_consumers(root, job["task_id"], consumers)
    except Exception:
        log.warning("Confirmed worker death could not retire model consumers for %s", job["task_id"], exc_info=True)


def _complete_exact_budget_pause_after_death(job: dict, root: pathlib.Path, task: dict,
                                             task_id: str, attempt: int) -> tuple[bool, bool]:
    """Worker death DURING durable budget pausing: finish the park, retry nothing.

    Returns ``(parked, fenced)`` from ONE read of the durable pause row:
    ``parked`` when the SAME task id was returned to its exact pause here, and
    ``fenced`` when pause/consumption evidence for THIS attempt (a live pause,
    a resumed row, a consumed grant — a ``pausing`` row without a source yet
    included) forbids the ordinary crash retry. Fail-closed: an UNREADABLE
    record cannot authorize a retry either.

    The pause row and its source were written before the worker unwound; the
    dead process took every local producer with it, so the durable rows are
    the complete view and the SAME task id returns to PENDING under its exact
    continuation (#1196). This is non-admission of the crash-retry path for
    exactly this window — a checkpointed task must never be replayed — not a
    general crash recovery: any other crash keeps its ordinary custody path.

    A worker that died holding an UNCONSUMED Resume grant belongs to the same
    window (#1196, F4): the grant was minted but nothing consumed it — the loop
    writes ``consumed_at`` before any new effect, and a refused continuation
    load (a source that went unreadable, a grant a newer writer superseded)
    kills the worker exactly here. The saved pause is not lost to a terminal
    crash: the grant this dead process can no longer consume is revoked under
    its own identity and the SAME task id is parked back on its exact pause for
    the owner to Resume again. A CONSUMED grant is never reopened — that task
    ran on, and its death takes terminal crash custody without ordinary retry.
    """
    from ouroboros.budget_pause import (
        LIVE_PAUSE_STATES, STATE_RESUME_GRANTED, STATE_RESUMED, budget_pause_row,
    )
    from supervisor.events_budget import install_exact_budget_pause

    result_root = pathlib.Path(task.get("budget_drive_root") or root)
    try:
        row = budget_pause_row(result_root, task_id)
    except Exception:
        return False, True
    state = str(row.get("state") or "") if row else ""
    grant = row.get("grant") if isinstance(row.get("grant"), dict) else {}
    fenced = bool(row and (state in LIVE_PAUSE_STATES or state == STATE_RESUMED or grant.get("consumed_at"))
                  and int(row.get("task_attempt") or 0) == int(attempt))
    if not (fenced and state in LIVE_PAUSE_STATES and row.get("source_ref")):
        return False, fenced
    source = "worker_death_during_pausing"
    with _queue_lock:
        if not _dead_job_is_current(job):
            return False, fenced
        if state == STATE_RESUME_GRANTED:
            if grant.get("consumed_at"):
                return False, fenced  # the loop consumed it and ran on: ordinary custody
            from supervisor.budget_resume import revoke_exact_budget_resume
            from supervisor.events_budget import BUDGET_HOLD_KEY
            from ouroboros.budget_pause import exact_pause_marker

            task.setdefault("_budget_pause_resume", {
                "pause_id": row.get("pause_id"), "grant_id": grant.get("grant_id"),
                "pause": exact_pause_marker(row, default_root=str(task.get("root_task_id") or task_id)),
            })
            if not revoke_exact_budget_resume(task, "worker_death_before_consumption"):
                if task.get("_budget_pause_consumed"):
                    return False, fenced  # a concurrent consumption keeps crash custody, never re-arms
                # Revocation failure is a nonterminal hold, never a crash retry
                # or terminal. Preserve exact source and grant identity even if
                # neither the result store nor the snapshot is writable.
                held = dict(task)
                held.setdefault("_budget_pause", exact_pause_marker(row, default_root=task_id))
                hold = held.get(BUDGET_HOLD_KEY)
                if not isinstance(hold, dict):
                    from supervisor.events_budget import HOLD_REVOCATION_UNWRITTEN, hold_budget_row

                    hold = hold_budget_row(held, reason=HOLD_REVOCATION_UNWRITTEN,
                                          extra={"pause_id": row.get("pause_id"), "grant_id": grant.get("grant_id")})
                _pool().RUNNING.pop(task_id, None)
                if not any(item.get("id") == task_id for item in _pool().PENDING):
                    _pool().PENDING.append(held)
                from supervisor import queue

                queue.sort_pending()
                hold["snapshot_persisted"] = False
                try:
                    hold["snapshot_persisted"] = bool(queue.persist_queue_snapshot(reason="dead_worker_budget_hold"))
                except Exception:
                    log.error("Budget hold snapshot failed for %s", task_id, exc_info=True)
                log.warning("Dead worker %s retained in nonterminal hold; snapshot persisted=%s",
                            task_id, hold["snapshot_persisted"])
                return True, fenced
            running = _pool().RUNNING.get(task_id)
            if isinstance(running, dict) and isinstance(running.get("task"), dict):
                running["task"].pop("_budget_pause_resume", None)
            source = "worker_death_before_grant_consumed"
    try:
        install_exact_budget_pause(_pool(), task_id, {"pause_id": row.get("pause_id")},
                                   source=source)
    except Exception:
        log.error("Exact budget pause of %s could not be completed after worker death; "
                  "leaving the row for the next reconciliation", task_id, exc_info=True)
        from supervisor.task_reaper import TerminalFileRecoveryPending

        raise TerminalFileRecoveryPending("exact budget pause parking remains pending")
    return True, fenced


def _recover_crashed_task_without_terminal(job: dict, queue: Any) -> None:
    """The previous crash outcome/retry policy, running on the existing reaper."""
    from supervisor.worker_owner_wait import has_owner_wait_checkpoint
    from supervisor.task_reaper import TerminalFileRecoveryPending

    w, task_id, task, meta = job["worker"], job["task_id"], job["task"], job["meta"]
    wid, exitcode = job["worker_id"], job["exitcode"]
    root = pathlib.Path(job["drive_root"])
    task_type = str(task.get("type") or "")
    # Signal crashes are terminal infrastructure failures for every task type.
    is_crash_signal = isinstance(exitcode, int) and exitcode < 0
    crash_signal = -exitcode if is_crash_signal else None
    # The crash toast is a DIRECT send - nothing re-addresses it downstream - so
    # the task's durable project binding has to win here, or a task converted
    # into a project mid-run is told about its crash in the chat it was born in.
    chat_id = _pool().coerce_chat_identity(
        resolve_project_chat(
            _pool().DRIVE_ROOT, task_id, task.get("parent_task_id"), task.get("root_task_id")
        )
        or task.get("chat_id"),
        0,
    )
    attempt = int(task.get("_attempt") or 1)
    # A `pausing` row without a checkpoint yet (death during the drain), or an
    # UNREADABLE pause record, fences the ordinary retry exactly like an
    # owner-wait checkpoint: completed work is never replayed (#1196).
    parked, budget_pausing = _complete_exact_budget_pause_after_death(job, root, task, task_id, attempt)
    if parked:
        return
    replay_unsafe = (not getattr(w, "active_capacity", True)
                     or has_owner_wait_checkpoint(meta, attempt) or budget_pausing)
    # Reconstruct cost/rounds from durable llm_usage for any
    # abnormal-termination rollup below (worker died pre-finalize,
    # so the event would otherwise carry zeros).
    r_cost_fields = _pool().reconstruct_task_cost(task_id, fields=True)
    with _queue_lock:
        if not _dead_job_is_current(job):
            return

    terminal_event = None
    if (is_crash_signal or attempt > _pool().QUEUE_MAX_RETRIES
          or replay_unsafe):
        deep = task_type == "deep_self_review"
        if budget_pausing:
            result_text = (
                "Worker process died with budget-continuation evidence. The checkpoint was "
                "incomplete, unreadable, or already consumed. Completed actions were not retried; "
                "no exact continuation is available for this attempt."
            )
            reason_code = "worker_crash_budget_pausing"
        elif replay_unsafe:
            result_text = (
                "Worker process died after an owner-wait checkpoint. The continuation "
                "source is retained; completed actions were not retried."
            )
            reason_code = "worker_crash_owner_wait"
        elif is_crash_signal:
            log.warning(
                "Task %s worker crashed with signal %s — terminal (no retry)",
                task_id, crash_signal,
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
                task_id, attempt, _pool().QUEUE_MAX_RETRIES,
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
                root, task_id, STATUS_FAILED,
                result=result_text,
                reason_code=reason_code,
                outcome_axes=terminal_outcome_axes(lifecycle=STATUS_FAILED, execution=EXECUTION_INFRA_FAILED, reason_code=reason_code, review_trigger="worker_terminal"),
                crash_signal=crash_signal,
                crash_exitcode=exitcode if isinstance(exitcode, int) else None,
                **r_cost_fields,
            )
        except Exception as exc:
            raise TerminalFileRecoveryPending("crash terminal result could not be stored") from exc
        # Message before task_done: otherwise the UI may close the card first.
        try:
            if replay_unsafe:
                user_msg = result_text
            elif is_crash_signal and deep:
                user_msg = (
                    f"❌ Deep self-review failed: worker process crashed (signal {crash_signal}). "
                    "This is a known platform fork-safety limitation. "
                    "Please use `/restart` and then `/review` to retry with a fresh process."
                )
            elif is_crash_signal:
                user_msg = (
                    f"❌ Task `{task_id[:8]}` failed: worker process crashed "
                    f"(signal {crash_signal}). This is an infrastructure crash and was not retried."
                )
            else:
                user_msg = (
                    f"❌ Task `{task_id[:8]}` failed after {attempt} crash(es). "
                    "Worker process crashed repeatedly. Please try again."
                )
            incident_task_id = str(task_id or "")
            _pool().send_with_budget(
                chat_id,
                user_msg,
                is_progress=True,
                task_id=incident_task_id,
                progress_meta={
                    "task_incident": reason_code,
                    "toast_once": f"{incident_task_id}:{reason_code}:{attempt}",
                },
                role="system", system_type="worker_failure")
        except Exception:
            log.debug("Failed to send failure message for %s", task_id, exc_info=True)
        terminal_event = ("failed", reason_code)
        from ouroboros.delegate_recovery import reconcile_unrecoverable_task
        reconcile_unrecoverable_task(root, task_id)
    elif task_type == "evolution" and not control_is(_pool().load_state(), "evolution_mode_enabled", True):
        # Evolution was stopped (or its control is unknown, #1307): do not resurrect
        # a dead evolution worker into another cycle (mirrors the hard-timeout gate
        # in queue.enforce_task_timeouts). A running task is never killed for this.
        try:
            from ouroboros.task_results import STATUS_CANCELLED, write_task_result
            write_task_result(
                root, task_id, STATUS_CANCELLED,
                result="Evolution worker died after the campaign was stopped; not retried.",
                reason_code="evolution_stopped_no_retry",
                outcome_axes=terminal_outcome_axes(lifecycle=STATUS_CANCELLED, execution="cancelled", reason_code="evolution_stopped_no_retry", review_trigger="worker_terminal"),
                **r_cost_fields,
            )
        except Exception as exc:
            raise TerminalFileRecoveryPending("stopped evolution result could not be stored") from exc
        terminal_event = ("cancelled", "")
        from ouroboros.delegate_recovery import reconcile_unrecoverable_task
        reconcile_unrecoverable_task(root, task_id)
    else:
        task = dict(task)
        task["_attempt"] = attempt + 1
        from ouroboros.delegate_recovery import prepare_worker_crash_handoff
        recovery_handoff = prepare_worker_crash_handoff(
            root, task, old_attempt=attempt, new_attempt=attempt + 1,
            worker_id=wid,
            exitcode=exitcode if isinstance(exitcode, int) else None,
        )
        try:
            from ouroboros.task_results import STATUS_INTERRUPTED, write_task_result
            write_task_result(
                root, task_id, STATUS_INTERRUPTED,
                result=f"Worker process died mid-task (attempt {attempt}). Retrying.",
                **r_cost_fields,
            )
        except Exception:
            log.debug("Failed to write interrupted status for %s", task_id, exc_info=True)
        try:
            from ouroboros.owner_hurry import retry_reset

            retry_reset(
                queue._task_drive_for_task(task, task_id),
                root, task_id,
                reason="worker_crash_requeue",
            )
        except Exception:
            log.debug("Crash-requeue retry reset failed for %s", task_id, exc_info=True)
        with queue.prepared_root_billing(task), _queue_lock:  # the ledger read happens before the lock
            if not _dead_job_is_current(job):
                return
            _pool().RUNNING.pop(task_id)
            try:
                admitted = queue.enqueue_task(task, front=True)
            except Exception:
                _pool().RUNNING[task_id] = meta
                raise
        admission_block = (
            str(admitted.get("_admission_blocked") or "")
            if isinstance(admitted, dict) else ""
        )
        if admission_block:
            from ouroboros.delegate_recovery import veto_worker_retry_handoff
            veto_worker_retry_handoff(
                root, task_id, recovery_handoff, admission_block,
            )
            reason_code = "worker_crash_retry_admission_blocked"
            try:
                from ouroboros.task_results import STATUS_FAILED, write_task_result
                write_task_result(
                    root,
                    task_id,
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
                    task_id,
                    exc_info=True,
                )
            _pool()._emit_task_done_terminal(
                task,
                task_id,
                "failed",
                reason_code=reason_code,
                cost_fields=r_cost_fields,
            )
    with _queue_lock:
        owned = _dead_job_is_current(job)
        if owned:
            _pool().RUNNING.pop(task_id)
    if terminal_event is not None and owned:
        # Host terminal publication follows withdrawal, as cancellation does.
        # Drain must not mistake it for a worker still preparing its files.
        _pool()._emit_task_done_terminal(
            task, task_id, terminal_event[0],
            reason_code=terminal_event[1], cost_fields=r_cost_fields,
        )


def _reaper():
    """Keep the pump and its facade patch points late-bound (queue imports it)."""
    from supervisor import task_reaper
    return task_reaper


def _submit_terminal_file_recovery(ctx: Any, task_id: str, meta: dict, state: dict) -> None:
    """Submit one attempt through the existing reaper; health retries a failed submission."""
    from supervisor import queue as q

    with q._queue_lock:
        if (ctx.RUNNING.get(task_id) is not meta
                or meta.get("_terminal_file_recovery") is not state
                or state["in_flight"] or state["event_sent"]):
            return
        state["in_flight"] = True
    try:
        q._ensure_reaper_started()
        q._reap_queue.put({"kind": "terminal_file_recovery", "ctx": ctx,
                           "task_id": task_id, "meta": meta, "state": state})
    except Exception:
        with q._queue_lock:
            state["in_flight"] = False
        log.warning("Terminal file recovery remains pending for %s", task_id, exc_info=True)


def enqueue_terminal_file_recovery(ctx: Any, evt: dict, task: dict) -> bool:
    """Retain a legacy/faulted completion in its existing RUNNING owner, off drain.

    The private in-memory latch is not persisted or a successful-save receipt.
    Child terminal bytes and existing pending refs remain restart authority.
    Repeated fault events only re-arm the existing health cadence, not a hot loop.
    """
    from supervisor import queue as q

    task_id = str(evt.get("task_id") or task.get("id") or "")
    with q._queue_lock:
        meta = ctx.RUNNING.get(task_id)
        if not isinstance(meta, dict):
            return False
        attempt = int(meta.get("attempt") or task.get("_attempt") or 1)
        stamp = evt.get("_files_prepared_attempt")
        if stamp is not None and stamp != attempt:
            return False
        state = meta.get("_terminal_file_recovery")
        if isinstance(state, dict):
            if state["attempt"] != attempt:
                return False
            # A returned frame reached drain but its CURRENT publication was
            # unreadable. Leave the next retry to health, not this event loop.
            if stamp is not None:
                state["event_sent"] = False
            return True
        state = {"attempt": attempt, "event": dict(evt), "task": dict(task),
                 "in_flight": False, "event_sent": False}
        meta["_terminal_file_recovery"] = state
    _reaper()._submit_terminal_file_recovery(ctx, task_id, meta, state)
    return True


def retry_terminal_file_recoveries() -> None:
    """Revisit only explicitly retained completions on the ordinary health tick."""
    from supervisor import queue as q

    with q._queue_lock:
        pending = [(tid, meta, meta.get("_terminal_file_recovery"))
                   for tid, meta in _pool().RUNNING.items()
                   if isinstance(meta, dict) and isinstance(meta.get("_terminal_file_recovery"), dict)]
    _reaper()._retry_deferred_reap_jobs()
    for task_id, meta, state in pending:
        _reaper()._submit_terminal_file_recovery(_pool(), task_id, meta, state)


def _recover_terminal_files(job: dict) -> None:
    from ouroboros.headless import prepare_terminal_task_files, terminal_task_files_ready
    from ouroboros.task_results import load_task_result
    from supervisor import queue as q
    from supervisor.worker_pool_lifecycle import _WORKER_LIFECYCLE_LOCK

    ctx, task_id, meta, state = (job[key] for key in ("ctx", "task_id", "meta", "state"))
    worker_id = meta.get("worker_id")
    with q._queue_lock:
        if (ctx.RUNNING.get(task_id) is not meta
                or meta.get("_terminal_file_recovery") is not state):
            return
        worker = ctx.WORKERS.get(worker_id)
    try:
        prepared = prepare_terminal_task_files(ctx.DRIVE_ROOT, state["task"])
        current = load_task_result(ctx.DRIVE_ROOT, task_id) or {}
        terminal = terminal_task_files_ready(ctx.DRIVE_ROOT, state["task"], current)
        source_present = prepared.get("terminal_source_present")
        if not terminal and source_present is not False:
            # Preserve the existing durable-publication requirement. A disk
            # outage is retryable file custody, never a second model attempt.
            log.warning("Cannot publish terminal files for %s: %s", task_id, prepared.get("error") or "publication incomplete")
            return
        with q._queue_lock:
            if (ctx.RUNNING.get(task_id) is not meta
                    or meta.get("_terminal_file_recovery") is not state):
                return
            event = {**state["event"], "_files_prepared_attempt": state["attempt"]}
            if event.get("worker_id") is None and worker_id is not None:
                event["worker_id"] = worker_id
            if terminal:
                event["status"] = current["status"]
            # A real invalid terminal uses the existing lifecycle-fault path
            # on drain. An I/O failure above does not fabricate such a result.
            state["event_sent"] = True
            state["terminal_source_present"] = source_present
        try:
            _pool().get_event_q().put(event)
        except Exception:
            with q._queue_lock:
                state["event_sent"] = False
            raise
        if terminal and worker is not None and not worker.proc.is_alive():
            # Only the captured dead process can be replaced; a newer pool is
            # never touched by completion of an old file job.
            with _WORKER_LIFECYCLE_LOCK:
                with q._queue_lock:
                    owned = ctx.WORKERS.get(worker_id) is worker
                if owned:
                    _reaper()._respawn_after_reap(q, _pool(), int(worker_id))
    except Exception:
        log.warning("Terminal file recovery failed for %s", task_id, exc_info=True)
    finally:
        with q._queue_lock:
            state["in_flight"] = False
