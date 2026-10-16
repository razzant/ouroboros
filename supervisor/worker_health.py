"""Crash detection and the terminal a host-side teardown publishes.

Distinguishes a worker that died from one that is merely slow, respects the spawn
grace window so a booting pool is not read as a crash storm, and publishes the
task_done the dying task can no longer publish for itself. A crash storm freezes the
captured membership, suppresses respawn until terminal sources settle, then fences
pooled admission; direct chat is spared.
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
    # Do not capture a storm membership halfway through a serialized spawn/retire.
    # Health intake retries next tick rather than waiting behind native teardown.
    from supervisor.worker_pool_lifecycle import _WORKER_LIFECYCLE_LOCK
    if not _WORKER_LIFECYCLE_LOCK.acquire(blocking=False):
        return
    try:
        with _queue_lock:
            _pool()._ensure_workers_healthy_locked(queue)
    finally:
        _WORKER_LIFECYCLE_LOCK.release()


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
        # Freeze the exact captured membership for its stop: under this latch no slot
        # starts or is forgotten (``respawn_worker``), so only a generation change,
        # whose own stop or start then owns the fence, can make the stop a no-op.
        _pool()._WORKER_POOL_DISABLED_REASON = "worker_crash_storm"
        jobs.append({"kind": "worker_crash_storm", "workers": tuple(_pool().WORKERS.items()),
                     "drive_root": str(_pool().DRIVE_ROOT)})
    for index, job in enumerate(jobs):
        try:
            queue._reap_queue.put(job)
        except Exception:
            # No handoff happened. Keep RUNNING intact for the next health tick.
            for unsent in jobs[index:]:
                worker = unsent.get("worker")
                if worker is not None and _pool().WORKERS.get(unsent["worker_id"]) is worker:
                    worker.reaping = False
            if disable_pool:
                # The latch never outlives its stop: an unsent stop waits with the reaper's retries.
                _reaper()._deferred_reap_jobs.append(jobs[-1])
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


def _job_current(job: dict, captured_timeout: bool) -> bool:
    """Under the queue lock: a crash job still owns RUNNING; a timeout decision moved it into the job."""
    if not captured_timeout:
        return _dead_job_is_current(job)
    from supervisor import queue
    from supervisor.task_reaper import _timeout_job_is_current
    return _timeout_job_is_current(job, queue, _pool())


def _pause_rehold_allowed(job: dict) -> bool:
    """Under Q, Stop beats a captured attempt's return to PENDING; unknown retains the reap.

    A cascade may already have settled while this child was outside the live
    maps. Its root fence still forbids re-holding. An open intent keeps the job
    deferred for cancellation custody, which can settle after this replay yields.
    """
    from ouroboros.cancel_intents import has_active_intent
    from ouroboros.store import acquisition_budget
    from supervisor import queue
    from supervisor.task_lifecycle import root_cancellation_fenced
    from supervisor.task_reaper import TerminalFileRecoveryPending

    task, task_id = job["task"], job["task_id"]
    root_id = str(task.get("root_task_id") or task_id)
    if root_cancellation_fenced(task, root_id):
        return False
    try:
        with acquisition_budget(0):
            active = any(has_active_intent(queue.DRIVE_ROOT, tid, strict=True)
                         for tid in dict.fromkeys((task_id, root_id)))
    except Exception as exc:
        raise TerminalFileRecoveryPending("cancellation authority is unreadable before re-hold") from exc
    if active:
        raise TerminalFileRecoveryPending("cancellation intent must settle before re-hold")
    return True


def recover_confirmed_dead_worker(job: dict) -> None:
    """Recover exact saved terminal bytes before applying the old crash policy.

    Saved terminal bytes, including the ``task_exception`` terminal a worker writes after
    ``handle_task`` failed, win over signal death; a live exact pause re-parks. Unknown or
    incomplete publication raises ``TerminalFileRecoveryPending`` and the reaper keeps the
    job. Only confirmed absence reaches crash policy: a signal is an infrastructure failure,
    an eligible crash retries within ``QUEUE_MAX_RETRIES`` keeping its cost and wait
    restrictions, and a budget continuation forbids ordinary retry.
    """
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
    if task_id:
        from ouroboros.mcp_task_sessions import settle_dead_task

        # Its death revoked every bridge call; close what the bridges left
        # behind before a retry of this task could need to open one again.
        settle_dead_task(root, task_id)
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
    if (not isinstance(birth, str) or not birth or worker.proc.exitcode is None
            or worker.proc.is_alive() or not _job_current(job, captured_timeout)):
        return
    if (not isinstance(job.get("meta"), dict)
            or job["meta"].get("task", {}).get("id") != job["task_id"]
            or job["task"].get("id") != job["task_id"]):
        return
    # One owner-ended writer: a write that fails keeps this positive death as a
    # witness the next addressed Continue/Resume discharges (#1554).
    from ouroboros.local_custody_repair import retire_ended_owner

    retire_ended_owner(pathlib.Path(job["task"].get("budget_drive_root") or job["drive_root"]),
                       job["task_id"], str(job["task"].get("root_task_id") or job["task_id"]),
                       pid=worker.proc.pid, birth=birth, attempt=job["attempt"], producer="worker_death")


def _complete_exact_budget_pause_after_death(job: dict, root: pathlib.Path, task: dict, task_id: str,
                                             attempt: int, *, captured_timeout: bool = False) -> tuple[bool, bool]:
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
    ``budget_pause.death_returns_to_pause`` names this window for the worker,
    which leaves a ``handle_task`` exception inside it to this recovery.

    ``captured_timeout``: the idle-timeout reaper's job (#1729,
    ``reaped_attempt_pause``). Its decision already moved RUNNING into the job,
    so the job's own currency seam applies and the park re-seats that captured
    row (``_park_captured_timeout``); an unreadable row, or a seed whose saved
    state cannot be read or written into its source, defers the reaper job.
    At either door an unreadable current owner fence defers the park
    (``OwnerFenceUnknown``): the retained job replays it from a fresh read.
    """
    from ouroboros.budget_pause import (
        LIVE_PAUSE_STATES, STATE_RESUME_GRANTED, STATE_RESUMED, budget_pause_row,
    )
    from supervisor.events_budget import install_exact_budget_pause
    from supervisor.task_reaper import TerminalFileRecoveryPending

    result_root = pathlib.Path(task.get("budget_drive_root") or root)
    try:
        row = budget_pause_row(result_root, task_id)
    except Exception as exc:
        if captured_timeout:  # unknown is never "no pause": the reaper retains its job
            raise TerminalFileRecoveryPending("pause authority is unreadable after the kill") from exc
        return False, True
    state = str(row.get("state") or "") if row else ""
    grant = row.get("grant") if isinstance(row.get("grant"), dict) else {}
    fenced = bool(row and (state in LIVE_PAUSE_STATES or state == STATE_RESUMED or grant.get("consumed_at"))
                  and int(row.get("task_attempt") or 0) == int(attempt))
    if fenced and state in LIVE_PAUSE_STATES and not row.get("source_ref"):
        # Died while still settling, before its source existed: the same attempt's
        # working state completes the exact pause instead of a terminal (#1543).
        from ouroboros.working_checkpoint import complete_pause_from_working
        try:
            row = complete_pause_from_working(result_root, task_id, row, attempt) or row
        except Exception as exc:
            if captured_timeout:  # unreadable or unwritable saved state is unknown, never "none saved"
                raise TerminalFileRecoveryPending("the source-less pause is not completed yet") from exc
            log.warning("Source-less pause of %s could not be completed from its working state",
                        task_id, exc_info=True)
    if not (fenced and state in LIVE_PAUSE_STATES and row.get("source_ref")):
        return False, fenced
    source = "worker_death_during_pausing"
    with _queue_lock:
        if not _job_current(job, captured_timeout):
            return False, fenced
        if captured_timeout and not _pause_rehold_allowed(job):
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
        if captured_timeout:
            if not _park_captured_timeout(job, task_id, str(row.get("pause_id") or ""), source):
                return False, fenced
        else:
            install_exact_budget_pause(_pool(), task_id, {"pause_id": row.get("pause_id")},
                                       source=source, defer_unknown_owner_fence=True)
    except Exception:
        log.error("Exact budget pause of %s could not be completed after worker death; "
                  "leaving the row for the next reconciliation", task_id, exc_info=True)
        raise TerminalFileRecoveryPending("exact budget pause parking remains pending")
    return True, fenced


def _park_captured_timeout(job: dict, task_id: str, pause_id: str, source: str) -> bool:
    """Park a reaped attempt's exact pause from its captured RUNNING row (#1729).

    ``install_exact_budget_pause`` parks from RUNNING, which the timeout decision
    already withdrew. One queue-lock section rechecks the job, seats a copy of
    its captured row, installs, and removes only that seated copy if it is still
    there: a failed or raced install leaves no transient RUNNING row, and a
    replacement execution is never touched. The parked row drops any grant
    carrier: a live pause of this attempt proves it is not the live grant
    (``restart_retention.park_saved_pause`` does the same).
    """
    from supervisor.events_budget import install_exact_budget_pause

    meta = job["meta"]
    with _queue_lock:
        if not _job_current(job, True) or not _pause_rehold_allowed(job):
            return False
        seated = {**meta, "task": {key: value for key, value in meta["task"].items()
                                   if key != "_budget_pause_resume"}}
        _pool().RUNNING[task_id] = seated
        try:
            install_exact_budget_pause(_pool(), task_id, {"pause_id": pause_id}, source=source,
                                       defer_unknown_owner_fence=True)
        finally:
            if _pool().RUNNING.get(task_id) is seated:
                _pool().RUNNING.pop(task_id, None)
    return True


def _hold_unsaved_pause(job: dict, seed: dict) -> bool:
    """An owner Pause seed whose dead attempt left no usable saved state (#1729): pause INTENT.

    ``restart_retention`` reads such a row as ``request_only``; no source is invented. The
    reaper neither retries nor terminalizes it: the SAME id becomes its next attempt, carrying
    only work already saved (``attach_recovery``), under ``HOLD_UNSAVED_PAUSE`` until an explicit
    Resume, whose existing seams recheck saved work, fences and money. The owner's root Resume
    starts it over a fresh whole-tree census (an effect of unknown outcome still holds it); a
    member's selection continues only saved work, so its detail says so. The seed row stays as it
    is. A changed seed defers; ``False`` means replacement ownership or cancellation forbids the hold.
    """
    from ouroboros.budget_pause import STATE_PAUSING, budget_pause_row
    from ouroboros.model_wait import fold_running_clock, stamp_execution_handoff
    from ouroboros.task_results import resolve_task_lineage
    from ouroboros.working_checkpoint import attach_recovery
    from supervisor import queue
    from supervisor.events_budget import HOLD_UNSAVED_PAUSE, hold_budget_row
    from supervisor.task_reaper import TerminalFileRecoveryPending

    task, task_id, attempt = job["task"], job["task_id"], int(job["attempt"])
    root = pathlib.Path(task.get("budget_drive_root") or job["drive_root"])
    try:
        row = budget_pause_row(root, task_id)
    except Exception as exc:
        raise TerminalFileRecoveryPending("pause authority is unreadable after the kill") from exc
    if (row.get("pause_id") != seed.get("pause_id") or row.get("state") != STATE_PAUSING or row.get("source_ref")
            or row.get("reason") != "owner" or int(row.get("task_attempt") or 0) != attempt):
        raise TerminalFileRecoveryPending("the dead attempt's pause changed under the reaper")
    held = {key: value for key, value in task.items() if key not in {"_budget_pause", "_budget_pause_resume"}}
    held["_attempt"] = attempt + 1
    attach_recovery(root, held, source_task_id=task_id, from_attempt=attempt, cause="idle_timeout", prior_task=task)
    fold_running_clock(held, job.get("meta"))
    stamp_execution_handoff(held, job.get("meta"))
    actor = resolve_task_lineage(task_id, **{key: task.get(key) for key in (  # the root or its retry attempt
        "metadata", "root_task_id", "parent_task_id", "delegation_role", "original_task_id",
        "timeout_retry_from")})["is_root_task"]
    with _queue_lock:
        if not _job_current(job, True) or not _pause_rehold_allowed(job):
            return False
        hold_budget_row(held, reason=HOLD_UNSAVED_PAUSE, result_root=root,
                        detail="stopped while pausing, before the pause was saved; " + (
                            "Resume continues only work already saved" if held.get("_working_recovery")
                            else "the owner's Resume starts it again from its request" if actor
                            else "nothing was saved to continue: cancel it or start a new run"),
                        extra={"root_task_id": str(task.get("root_task_id") or task_id),
                               "pause_id": str(row.get("pause_id") or ""), "task_attempt": attempt,
                               "pause_generation": int(row.get("pause_generation") or 0), "cause": "owner_pause"})
        _pool().PENDING.append(held)
        queue.sort_pending()
        try:
            queue.persist_queue_snapshot(reason="timeout_pause_intent_hold")
        except Exception:
            log.error("Pause-intent hold snapshot failed for %s", task_id, exc_info=True)
    return True


def reaped_attempt_pause(job: dict, attempt: int) -> str:
    """The idle-timeout reaper's reading of the dead attempt's CURRENT durable pause (#1729).

    ``"parked"``: a live exact pause, an unconsumed grant or a completable
    source-less seed of THIS attempt returned the SAME id to its exact pause
    (the owner's Pause stands; no retry, no terminal). ``"held"``: an owner
    seed of THIS attempt with no usable saved state kept the SAME id as pause
    intent (``_hold_unsaved_pause``); a money seed with none keeps the ordinary
    idle policy (``""``, the existing retry gates). ``"spent"``: this attempt consumed
    the very grant the dead row carries, so an eligible retry continues its
    saved work without that carrier. ``"replaced"``: another execution owns the
    id. ``""``: nothing of this attempt to keep. Unreadable authority, or saved
    state that cannot be read or written into the seed's source, raises
    ``TerminalFileRecoveryPending``: the reaper defers and rereads it all.
    """
    from ouroboros.budget_pause import LIVE_PAUSE_STATES, STATE_RESUMED, budget_pause_row
    from supervisor.task_reaper import TerminalFileRecoveryPending

    task, task_id = job["task"], job["task_id"]
    try:
        row = budget_pause_row(pathlib.Path(task.get("budget_drive_root") or job["drive_root"]), task_id)
    except Exception as exc:
        raise TerminalFileRecoveryPending("pause authority is unreadable after the kill") from exc
    if not row or int(row.get("task_attempt") or 0) != int(attempt):
        return ""
    grant = row.get("grant") if isinstance(row.get("grant"), dict) else {}
    consumed = row.get("state") == STATE_RESUMED or bool(grant.get("consumed_at"))
    if row.get("state") in LIVE_PAUSE_STATES and not consumed:
        parked, _fenced = _complete_exact_budget_pause_after_death(
            job, pathlib.Path(job["drive_root"]), task, task_id, attempt, captured_timeout=True)
        if parked:
            return "parked"
        if not row.get("source_ref") and row.get("reason") == "owner":
            if _hold_unsaved_pause(job, row):
                return "held"
            with _queue_lock:
                return "" if _job_current(job, True) else "replaced"
        if not row.get("source_ref") and row.get("reason") == "budget" and "_budget_pause_resume" in task:
            # Not a produced shape: a resumed attempt saves its working state before consuming its
            # grant (``save_resumed``), so its money seed completes from it above. A carrier beside an
            # unsaved money seed is unexplained authority: never retired, never copied; it defers.
            raise TerminalFileRecoveryPending("an unsaved money seed beside a carried Resume is unexplained")
        with _queue_lock:
            return "" if _job_current(job, True) else "replaced"
    carrier = task.get("_budget_pause_resume") if isinstance(task.get("_budget_pause_resume"), dict) else {}
    return "spent" if (consumed and carrier and carrier.get("pause_id") == row.get("pause_id")
                       and carrier.get("grant_id") == grant.get("grant_id")) else ""


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
    # An answer the host already holds (owed, delivered, or unreadable evidence) is
    # never generated again: the dead attempt is finalized from the outbox's bytes
    # and published like a saved terminal, with the answer still replayed by the
    # outbox. Only a confirmed absence of any registered answer reaches crash policy.
    from ouroboros.working_checkpoint import adopt_registered_answer

    try:
        adopted = adopt_registered_answer(
            root, task, task_id, attempt=attempt, cause="worker_crash",
            registry_root=pathlib.Path(task.get("budget_drive_root") or root),
        )
    except Exception as exc:
        raise TerminalFileRecoveryPending("registered-answer finalization could not be stored") from exc
    if adopted:
        with _queue_lock:
            if not _dead_job_is_current(job):
                return
        from supervisor.task_reaper import _finish_self_finalized_task

        _finish_self_finalized_task(
            queue, _pool(), task, task_id, task_type, str(adopted.get("status") or ""), adopted,
            terminal_task_metadata(task.get("metadata")), worker_id=wid,
        )
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
        prior_task, task = task, dict(task)
        task["_attempt"] = attempt + 1
        # The existing same-ID retry continues the saved work, not the original prompt.
        from ouroboros.working_checkpoint import attach_recovery
        attach_recovery(pathlib.Path(task.get("budget_drive_root") or root), task,
                        source_task_id=task_id, from_attempt=attempt, cause="worker_crash", prior_task=prior_task)
        from ouroboros.model_wait import fold_running_clock, stamp_execution_handoff

        fold_running_clock(task, meta)  # confirmed waits newer than the checkpoint (#939)
        stamp_execution_handoff(task, meta)
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
            # #1183: the dead attempt's own active fence must not refuse its retry
            # (an unknown pid proves no exact owner, so its fence stays).
            from supervisor.queue_transitions import clear_acceptance_fence_for_attempt
            if isinstance(getattr(w.proc, "pid", None), int):
                clear_acceptance_fence_for_attempt(task_id, task_attempt=attempt, pid=w.proc.pid)
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
