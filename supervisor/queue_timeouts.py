"""Activity-based liveness: which running task has stopped being alive.

A task is judged on its own progress AND on its subtree's — a coordinator whose
children are working is not idle — then against the idle window, its explicit
deadline and the absolute ceiling. The decision is taken under the queue lock; the
teardown it decides on is handed to the off-loop reaper.
"""

from __future__ import annotations

import datetime
import logging
import pathlib
import time
import uuid
from typing import Any, Dict
from supervisor.task_reaper import (
    resolve_grace_episode_for_spared_task as _resolve_grace_episode_for_spared_task,
)


def _queue():
    """The parent module, read at call time.

    The queue owns PENDING/RUNNING, the drive root, the liveness settings and the lock that guards them, and ``init``/``init_queue_refs`` REBIND those names. Reading them through the module is what keeps one binding: a from-import here would freeze the value this module saw at import time.
    """
    from supervisor import queue

    return queue


log = logging.getLogger(__name__)


def _task_deadline_ts(task: Dict[str, Any]) -> float:
    raw = str(task.get("deadline_at") or "").strip()
    if not raw:
        metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        raw = str(metadata.get("deadline_at") or "").strip()
    if not raw:
        contract = task.get("task_contract") if isinstance(task.get("task_contract"), dict) else {}
        raw = str(contract.get("deadline_at") or "").strip()
    if not raw:
        return 0.0
    try:
        parsed = datetime.datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=datetime.timezone.utc)
        return float(parsed.timestamp())
    except Exception:
        return 0.0


def _task_drive_for_task(task: Dict[str, Any], task_id: str) -> pathlib.Path:
    """Active drive of a running task (child drive for forked/workspace tasks,
    canonical otherwise) — where its mailbox and observability actually live.
    Resolution mirrors forward_to_worker: task fields, then the result record."""
    task = task if isinstance(task, dict) else {}
    child = str(task.get("child_drive_root") or task.get("drive_root") or "").strip()
    if not child:
        try:
            from ouroboros.task_results import load_task_result
            record = load_task_result(pathlib.Path(_queue().DRIVE_ROOT), str(task_id)) or {}
            child = str(record.get("child_drive_root") or record.get("headless_child_drive_root") or record.get("drive_root") or "").strip()
        except Exception:
            child = ""
    return pathlib.Path(child) if child else pathlib.Path(_queue().DRIVE_ROOT)


def enforce_task_timeouts() -> None:
    """Enforce soft/hard timeouts for running tasks.

    Holds the queue lock for the whole pass: RUNNING pops and worker respawn
    decisions raced with HTTP cancel handlers (double respawn → orphaned
    worker; wrong-task dequeue). The RLock keeps nested respawn/assign calls
    re-entrant.
    """
    # Avoid circular dependency during module load.
    from supervisor import workers

    if not _queue().RUNNING:
        return
    now = time.time()
    st = _queue().load_state()
    owner_chat_id = int(st.get("owner_chat_id") or 0)

    with _queue()._queue_lock:
        _enforce_task_timeouts_locked(workers, now, owner_chat_id, st)


def _is_descendant_of(task: Dict[str, Any], ancestor_id: str) -> bool:
    """True if `task` is in the subtree rooted at ancestor_id. Cheap in-memory (no I/O):
    root_task_id == ancestor_id (covers the common root-orchestrator case even when an
    INTERMEDIATE parent has already left RUNNING — a grandchild whose parent finished is
    still a descendant of the root), OR the parent_task_id chain (via RUNNING metas)
    reaches ancestor_id (covers a mid-tree ancestor while the chain is intact).
    """
    if not isinstance(task, dict) or not ancestor_id:
        return False
    if str(task.get("root_task_id") or "") == ancestor_id:
        return True
    cur = task
    hops = 0
    while isinstance(cur, dict) and hops < 25:
        pid = str(cur.get("parent_task_id") or "")
        if not pid:
            return False
        if pid == ancestor_id:
            return True
        nxt = _queue().RUNNING.get(pid)
        cur = nxt.get("task") if isinstance(nxt, dict) and isinstance(nxt.get("task"), dict) else None
        hops += 1
    return False


def _subtree_progressing(task_id: str, now: float, idle_timeout: float) -> bool:
    """True if any RUNNING descendant of task_id made real progress within idle_timeout.

    In-memory walk over RUNNING only (NO I/O — this runs under the queue lock): keeps a
    productively-waiting orchestrator alive while its children work, instead of a flat
    wall-clock kill. Descendant freshness uses last_progress_at (real progress), not the
    bare liveness heartbeat.
    """
    if not task_id:
        return False
    for tid, m in list(_queue().RUNNING.items()):
        if tid == task_id or not isinstance(m, dict):
            continue
        if not _is_descendant_of(m.get("task") if isinstance(m.get("task"), dict) else {}, task_id):
            continue
        # Real progress only (NOT the bare 30s liveness heartbeat): a child that merely
        # pings but makes no progress must not keep its ancestor alive.
        lp = float(m.get("last_progress_at") or m.get("started_at") or 0.0)
        if lp and (now - lp) < idle_timeout:
            return True
    return False


def _has_live_descendant(task_id: str) -> bool:
    """True if any LIVE (RUNNING or PENDING) task is a descendant of task_id (in-memory, no
    I/O). Used to recognise an orchestrator at kill time so it is NOT blind-retried — a
    blind retry would replay the plan and re-spawn the whole subtree (the timeout storm).
    PENDING is included: a parent can time out while its children are merely QUEUED (worker
    saturation / project lease), and those queued children are still its live subtree.
    """
    if not task_id:
        return False
    for tid, m in list(_queue().RUNNING.items()):
        if tid == task_id or not isinstance(m, dict):
            continue
        if _is_descendant_of(m.get("task") if isinstance(m.get("task"), dict) else {}, task_id):
            return True
    for t in list(_queue().PENDING):
        if not isinstance(t, dict) or str(t.get("id") or "") == task_id:
            continue
        if _is_descendant_of(t, task_id):
            return True
    return False


def _has_pending_descendant(task_id: str) -> bool:
    """True if any PENDING (queued, not yet assigned) task is a descendant of task_id. A
    parent whose children are merely WAITING for worker capacity (saturation / project lease)
    is not idle/stuck — keep it alive (bounded by the absolute ceiling) so it can integrate
    them once they run, instead of killing it and orphaning the queued subtree."""
    if not task_id:
        return False
    for t in list(_queue().PENDING):
        if not isinstance(t, dict) or str(t.get("id") or "") == task_id:
            continue
        if _is_descendant_of(t, task_id):
            return True
    return False


def _enforce_task_timeouts_locked(
    workers: Any, now: float, owner_chat_id: int, st: Dict[str, Any]
) -> None:
    # ONE typed owner-stop predicate before every generic timeout-grace consumer
    # (S3 §12.2 item 8): a task whose owner-requested finalization intent is
    # still OPEN is bypassed whole — no spare-withdraw, no spare-clock reset,
    # no second grace episode, no expiry kill, no RUNNING.pop, no reaper
    # enqueue, no retry scheduling. The hold deliberately outlives the grace
    # deadline (the expiry window): the deadline gates only the sweep's
    # arm-vs-feed-custody decision in supervisor/owner_stop.py +
    # sweep_cancel_intents; the intent stays the one owner will and custody
    # stays the only killer.
    from supervisor.owner_stop import running_owner_stop_tasks

    owner_stop_held = running_owner_stop_tasks(
        _queue().DRIVE_ROOT, grace_sec=_queue().FINALIZATION_GRACE_SEC,
    )
    for task_id, meta in list(_queue().RUNNING.items()):
        if not isinstance(meta, dict):
            continue
        if str(task_id) in owner_stop_held:
            continue
        task = meta.get("task") if isinstance(meta.get("task"), dict) else {}
        started_at = float(meta.get("started_at") or 0.0)
        if started_at <= 0:
            continue
        last_hb = float(meta.get("last_heartbeat_at") or started_at)
        runtime_sec = max(0.0, now - started_at)
        hb_lag_sec = max(0.0, now - last_hb)
        hb_stale = hb_lag_sec >= _queue().HEARTBEAT_STALE_SEC
        _wid = meta.get("worker_id")
        worker_id = int(_wid) if _wid is not None else -1
        task_type = str(task.get("type") or "")
        _att = meta.get("attempt")
        if _att is None:
            _att = task.get("_attempt")
        attempt = int(_att) if _att is not None else 1

        deadline_ts = _task_deadline_ts(task)
        deadline_reached = bool(deadline_ts and now >= deadline_ts)

        idle_timeout = max(
            float(_queue().get_task_idle_timeout_sec()),
            float(_queue().get_per_call_timeout_ceiling_sec()) + 120.0,
        )
        # deep_self_review runs a single long 1M-context LLM call with NO intermediate
        # progress events (no tool loop), so the idle timer governs it from started_at;
        # its prior ~60min tolerance is preserved so it is not idle-killed mid-call.
        if task_type == "deep_self_review":
            idle_timeout = max(idle_timeout, 3600.0)
        abs_ceiling = float(_queue().get_task_abs_ceiling_sec())
        last_progress_at = float(meta.get("last_progress_at") or started_at)
        idle_sec = max(0.0, now - last_progress_at)
        subtree_progressing = _subtree_progressing(task_id, now, idle_timeout)
        own_progress = idle_sec < idle_timeout
        # B3 external-wait lease: a held delegate_wait window over a live delegated run
        # is legitimate silence (hard-bounded by events._handle_external_wait_lease);
        # it spares ONLY this idle rail — ceiling/deadline/budget/cancel never consult it.
        lease_ts = meta.get("external_wait_lease_until")
        # Keep an orchestrator alive on own progress, a freshly progressing RUNNING
        # descendant, a QUEUED descendant (a kill would orphan the queued subtree), or a
        # live external-wait lease; only abs ceiling / explicit deadline / budget are
        # unconditional.
        progressing = (own_progress or subtree_progressing or _has_pending_descendant(task_id)
                       or (isinstance(lease_ts, (int, float)) and float(lease_ts) > now))
        ceiling_reached = runtime_sec >= abs_ceiling

        # Hard axes (deadline_at, abs ceiling) stop the task regardless of activity; the
        # idle/subtree gate only spares a still-progressing task with NO explicit deadline
        # — an explicit/caller deadline is honored promptly, while no blanket wall-clock
        # kills a productively-waiting orchestrator.
        if not ceiling_reached and not deadline_reached and progressing:
            # An outstanding episode outlives this reprieve or is withdrawn by it; the
            # rule (own progress answers the request, sparing only suspends its clock)
            # lives with the rest of the episode mechanics in task_reaper. The latch is
            # checked here so the drive resolution (which may read the result record)
            # stays off the no-episode path.
            if meta.get("finalization_requested_at") and _resolve_grace_episode_for_spared_task(
                _task_drive_for_task(task, str(task_id)), str(task_id), meta,
                chat_id=int(task.get("chat_id") or owner_chat_id or 0),
                own_progress=own_progress, now=now,
            ):
                _queue().RUNNING[task_id] = meta
            continue

        if ceiling_reached:
            terminal_reason = "absolute_ceiling"
        elif deadline_reached:
            terminal_reason = "deadline"
        else:
            terminal_reason = "idle_timeout"
        finalization_requested_at = float(meta.get("finalization_requested_at") or 0.0)
        if finalization_requested_at <= 0 and _queue().FINALIZATION_GRACE_SEC > 0:
            meta["finalization_requested_at"] = now
            meta["finalization_reason"] = terminal_reason
            # The control's msg_id IS the episode's identity: it is what the
            # symmetric withdraw revokes, so the latch and the mailbox control
            # can never name different episodes.
            meta["finalization_control_msg_id"] = _queue()._request_finalization_grace(
                _task_drive_for_task(task, str(task_id)), str(task_id), terminal_reason,
                chat_id=int(task.get("chat_id") or owner_chat_id or 0),
                stamp=int(now),
            )
            _queue().RUNNING[task_id] = meta
            continue
        if finalization_requested_at > 0 and now - finalization_requested_at < _queue().FINALIZATION_GRACE_SEC:
            continue

        # NOTE: "worker self-finalized at the idle boundary" is handled by the reaper's
        # POST-KILL terminal re-check (kill+join FIRST, then honor an on-disk terminal
        # result, idempotent task_done). No short-circuit here: freeing the slot inline
        # would let assign_tasks reuse it mid-flight and could drop the terminal event.

        # Variant A: hand the ENTIRE teardown to the background reaper so the loop tick
        # stays fast and the terminal write + retry enqueue happen only AFTER kill/join
        # (no race with a concurrently-assigned retry; a subagent retry reuses id/drive).
        # Live-RUNNING decisions (orchestrator -> no blind retry; retry id) freeze HERE.
        if task_type == "evolution":
            from supervisor.evolution_lifecycle import update_evolution_transaction
            if not update_evolution_transaction(task_id, dispatch_status="reaping"):
                log.warning("Evolution timeout teardown deferred: reaping state was not durable for %s", task_id)
                continue
        _queue().RUNNING.pop(task_id, None)
        proc_handle = None
        if worker_id in workers.WORKERS:
            w = workers.WORKERS[worker_id]
            if w.busy_task_id == task_id:
                w.busy_task_id = None
            # Mark reaping under the lock so assign_tasks and the crash detector both skip
            # this slot until the reaper installs a fresh worker.
            w.reaping = True
            proc_handle = w.proc

        # NOTE: the "no blind retry of an orchestrator with live descendants" guarantee is
        # TIMEOUT-REAPING-specific (this path). The worker-CRASH path
        # (workers._ensure_workers_healthy_locked) has its own signal-vs-attempt retry
        # semantics and is intentionally not gated here; a crashed-orchestrator storm is a
        # separate, rarer concern than the flat-wall-clock timeout storm this batch targets.
        orchestrator = _has_live_descendant(task_id)
        will_retry = (
            attempt <= _queue().QUEUE_MAX_RETRIES
            and isinstance(task, dict)
            and not deadline_reached
            and not ceiling_reached
            and not orchestrator
        )
        # A stopped evolution campaign breaks the auto-retry chain. `st` is the live state
        # loaded this tick, so this reflects the current owner decision.
        if will_retry and task_type == "evolution" and not bool(st.get("evolution_mode_enabled")):
            will_retry = False
        # An ACTIVE cancel intent (immediate policy, or a finalize intent already
        # CLAIMED by custody — open finalize intents never reach here, the hold
        # above skips them) must never spawn a retry clone: a new-uuid retry
        # escapes the intent (keyed by the old id) and CANCELLED_ROOT_FENCES,
        # restarting work the owner stopped.
        if will_retry:
            from ouroboros.cancel_intents import has_active_intent

            will_retry = not has_active_intent(_queue().DRIVE_ROOT, str(task_id))
        retry_task_id = ""
        if will_retry:
            same_id = task_type == "evolution" or str(task.get("delegation_role") or "") == "subagent"
            retry_task_id = task_id if same_id else uuid.uuid4().hex[:8]

        _queue()._ensure_reaper_started()
        _queue()._reap_queue.put({
            "worker_id": worker_id,
            "proc": proc_handle,
            "task_id": str(task_id),
            "task": task,
            "task_type": task_type,
            "terminal_reason": terminal_reason,
            "attempt": attempt,
            "owner_chat_id": owner_chat_id,
            "runtime_sec": runtime_sec,
            "hb_lag_sec": hb_lag_sec,
            "hb_stale": hb_stale,
            "deadline_reached": deadline_reached,
            "ceiling_reached": ceiling_reached,
            "orchestrator": orchestrator,
            "will_retry": will_retry,
            "retry_task_id": retry_task_id,
            "incident_toast_once": f"{task_id}:{terminal_reason}:{int(finalization_requested_at or now)}",
        })
        _queue().persist_queue_snapshot(reason="task_timeout_reap_queued")
