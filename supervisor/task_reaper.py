"""Variant A: off-loop worker reaper (extracted from supervisor/queue.py for module size).

The supervisor loop must stay responsive (<100ms ticks), so a timed-out task's heaviest
teardown — process kill + join (up to ~5s) + service-log archive + worker respawn (process
spawn) — runs on a single-owner background reaper thread instead of inline under the queue
lock. The loop only marks the worker ``reaping`` (so assign_tasks/crash-detector skip the
slot) and hands a fully-decided job here; ``supervisor.queue`` re-exports the thin names
(``_reap_queue`` / ``_ensure_reaper_started`` / ``_reap_timed_out_task``).
"""

from __future__ import annotations

import logging
import pathlib
import queue as _stdqueue
import shutil
import threading
import uuid
from typing import Any, Dict, Optional, Set

from ouroboros.outcomes import EXECUTION_INFRA_FAILED, terminal_outcome_axes
from ouroboros.utils import append_jsonl, utc_now_iso
from supervisor.events import HOST_NARRATION
from supervisor.message_bus import send_with_budget

log = logging.getLogger(__name__)

reap_queue: "_stdqueue.Queue[Dict[str, Any]]" = _stdqueue.Queue()
_reaper_thread: "Optional[threading.Thread]" = None
_reaper_start_lock = threading.Lock()

# Task ids handed to the reaper whose worker is not PROVABLY dead (queued,
# mid-kill, or wedged-held after a failed kill): the acceptance-fence
# dead-owner predicate treats membership as ALIVE; only confirmed death forgets.
_REAPING_TASK_IDS: Set[str] = set()
_REAPING_TASK_IDS_LOCK = threading.Lock()


def note_task_reaping(task_id: str) -> None:
    tid = str(task_id or "").strip()
    if tid:
        with _REAPING_TASK_IDS_LOCK:
            _REAPING_TASK_IDS.add(tid)


def task_reaping_in_progress(task_id: str) -> bool:
    with _REAPING_TASK_IDS_LOCK:
        return str(task_id or "").strip() in _REAPING_TASK_IDS


def _forget_task_reaping(task_id: str) -> None:
    with _REAPING_TASK_IDS_LOCK:
        _REAPING_TASK_IDS.discard(str(task_id or "").strip())


def reaper_loop() -> None:
    while True:
        try:
            job = reap_queue.get()
        except Exception:
            continue
        try:
            reap_timed_out_task(job)
        except Exception:
            log.error("Reaper failed for task %s", (job or {}).get("task_id"), exc_info=True)
            # Self-heal: an escape BEFORE the guarded teardown (e.g. the top-of-function
            # imports / variable extraction) must not strand the slot at reaping=True forever —
            # the crash detector skips reaping slots, so it would be unrecoverable until restart.
            # Clear reaping (the same conservative recovery step 5 uses) so a later tick reclaims
            # it; do NOT respawn here (an early escape may have left the original worker alive).
            try:
                from supervisor import workers as _w_mod
                from supervisor.queue import _queue_lock as _ql

                _wid_raw = (job or {}).get("worker_id")
                if _wid_raw is not None:
                    with _ql:
                        _w = _w_mod.WORKERS.get(int(_wid_raw))
                        if _w is not None:
                            _w.reaping = False
            except Exception:
                log.debug("Reaper: self-heal reaping-clear failed", exc_info=True)
        finally:
            try:
                reap_queue.task_done()
            except Exception:
                pass


def ensure_reaper_started() -> None:
    """Start the reaper thread, or RESTART it if it ever died — otherwise a dead reaper
    would strand every ``reaping=True`` slot forever (assign skips it, no one respawns it)."""
    global _reaper_thread
    t = _reaper_thread
    if t is not None and t.is_alive():
        return
    with _reaper_start_lock:
        t = _reaper_thread
        if t is not None and t.is_alive():
            return
        if t is not None:
            log.warning("Task reaper thread had died; restarting it.")
        _reaper_thread = threading.Thread(target=reaper_loop, name="task-reaper", daemon=True)
        _reaper_thread.start()


def request_finalization_grace(
    task_drive: pathlib.Path, task_id: str, terminal_reason: str, *, chat_id: int, stamp: int,
    control_msg_id: str = "", toast_text: str = "", control_text: str = "",
) -> str:
    """Ask a task to finalize cooperatively before the supervisor stops it.

    Both side effects of opening a grace window: the typed ``finalize_now``
    control on the task's ACTIVE drive (the one the loop drains — the child
    drive for forked/workspace tasks), which buys a tool-less final answer, and
    the owner-visible progress toast. Lives here with the rest of the
    supervisor's terminal-path mechanics so ``queue.py``'s enforce loop keeps
    only the DECISION.

    ``control_msg_id`` (S3, additive): the OWNER-STOP episode derives its
    control identity deterministically from the durable stop ``request_id``
    (§12.2 item 5) so a watchdog/restart replay appends the SAME id — the drain
    dedupes by msg_id, never a duplicate control. Absent = the generic timeout
    episode's fresh random id, byte-identical to the prior behavior.
    ``toast_text`` (additive) replaces the generic reached-terminal wording for
    an owner-requested episode.

    Returns the control's msg_id — the grace EPISODE's identity. The caller
    stores it next to the latch so ``withdraw_finalization_grace`` can retract
    exactly this episode; "" means no control is outstanding.
    """
    control_msg_id = str(control_msg_id or "") or uuid.uuid4().hex
    try:
        from ouroboros.owner_mailbox import KIND_FINALIZE_NOW, write_owner_message
        if not write_owner_message(
            # ``control_text`` (additive) lets the owner-stop episode carry its
            # typed reason plus the bounded child projection in the control
            # payload while the short ``terminal_reason`` keeps naming the
            # toast/incident; absent = the reason itself, as before.
            task_drive, str(control_text or "") or terminal_reason, task_id,
            msg_id=control_msg_id, kind=KIND_FINALIZE_NOW,
        ):
            control_msg_id = ""
    except Exception:
        control_msg_id = ""
        log.debug("Failed to write finalize_now control for %s", task_id, exc_info=True)
    try:
        from supervisor import workers as _workers_mod
        _workers_mod.get_event_q().put({
            "type": "send_message",
            "chat_id": chat_id,
            "text": str(toast_text or "") or (
                f"⏳ Task {task_id} reached {terminal_reason}. "
                "Finalize artifacts/results now; supervisor will stop the task after the grace window."
            ),
            "format": "markdown",
            "is_progress": True,
            "task_id": task_id,
            # Addressed to the task's card, authored by the supervisor: see
            # events.HOST_NARRATION. Without it this toast stamped the task's
            # last_progress_at and the next tick withdrew the episode it announced.
            HOST_NARRATION: True,
            "progress_meta": {
                "task_incident": terminal_reason,
                "toast_once": f"{task_id}:{terminal_reason}:{stamp}",
            },
            "ts": utc_now_iso(),
        })
    except Exception:
        log.debug("Failed to emit finalization grace warning for %s", task_id, exc_info=True)
    return control_msg_id


def withdraw_finalization_grace(
    task_drive: pathlib.Path, task_id: str, meta: Dict[str, Any], *, chat_id: int,
) -> bool:
    """Retract a grace window the supervisor no longer wants — the symmetric twin
    of ``request_finalization_grace``, and the ONLY correct way to end an episode.

    An episode is two things written together: the durable ``finalize_now``
    control in the task's mailbox and the ``finalization_requested_at`` latch in
    its RUNNING metadata. Retiring only the latch left the task holding a live
    kill order it would obey on its next drain — with no terminal condition
    pending and no kill coming — so this owns BOTH halves (plus the owner toast
    that promised a stop) and refuses to split them: if the control cannot be
    revoked, the latch stays and the caller retries on the next tick.

    Returns True when a whole episode was withdrawn.
    """
    if not float(meta.get("finalization_requested_at") or 0.0):
        return False
    control_msg_id = str(meta.get("finalization_control_msg_id") or "")
    if control_msg_id:
        revoked = False
        try:
            from ouroboros.owner_mailbox import revoke_owner_control
            revoked = revoke_owner_control(task_drive, task_id, control_msg_id)
        except Exception:
            log.debug("Failed to revoke finalize_now control for %s", task_id, exc_info=True)
        if not revoked:
            log.warning(
                "Could not revoke the finalize_now control for %s; keeping the grace latch "
                "so the episode stays whole (retried next tick).", task_id,
            )
            return False
    reason = str(meta.get("finalization_reason") or "the terminal condition")
    meta.pop("finalization_requested_at", None)
    meta.pop("finalization_reason", None)
    meta.pop("finalization_control_msg_id", None)
    try:
        # Plain progress, NOT a task_incident: the request's toast is styled as an
        # error in the UI, and its retraction is good news. It is emitted at most
        # once per episode, because the latch is already gone by here.
        from supervisor import workers as _workers_mod
        _workers_mod.get_event_q().put({
            "type": "send_message",
            "chat_id": chat_id,
            # Says what was DONE, not what the supervisor cannot know. The revocation
            # only reaches a reader that has not drained yet; whether this task had
            # already consumed the control is in-process state (`loop._owner_msg_seen`)
            # the supervisor cannot see, and claiming a cancelled stop after the task
            # took the order was a plain lie. Correcting the sentence, not adding an ack.
            "text": (
                f"▶️ Task {task_id} resumed work before the {reason} grace window closed; "
                "the stop request was retracted from its mailbox — if the task had "
                "already read it, it may still finalize."
            ),
            "format": "markdown",
            "is_progress": True,
            "task_id": task_id,
            HOST_NARRATION: True,
            "ts": utc_now_iso(),
        })
    except Exception:
        log.debug("Failed to emit finalization withdrawal for %s", task_id, exc_info=True)
    return True


def resolve_grace_episode_for_spared_task(
    task_drive: pathlib.Path, task_id: str, meta: Dict[str, Any], *,
    chat_id: int, own_progress: bool, now: float,
) -> bool:
    """What happens to an outstanding grace episode on a tick the task is NOT stopped.

    Two different reprieves reach this point and they are not the same answer:

    * The task made its OWN progress — that is the task answering the request, so the
      episode is withdrawn whole. A DESCENDANT's progress must not withdraw it: that
      spares the task deliberately but is not the task answering, and treating it as
      an answer re-armed the episode on every subtree flicker (one finalize_now and
      one owner toast per flicker, window never elapsing).
    * The task is merely SPARED. Sparing suspends the stop, so it suspends the window:
      the latch is the base of a clock measuring time the supervisor ACTUALLY intended
      to stop this task, not wall clock. A parent blocked in ``wait_tasks`` makes no
      own progress and drains no mailbox (drains happen at round boundaries), so a
      window that ran down while the subtree deliberately kept it alive killed it
      0.5s after its last child finished — zero usable grace, the finalize_now still
      unread, and a blind plan replay. The ask is NOT re-sent: one episode, one
      control, one toast. A withdrawal that could not be made durable lands here too
      and simply retries on the next tick.

    Returns True when the episode was withdrawn (the caller re-publishes the row).
    """
    if own_progress and withdraw_finalization_grace(
        task_drive, task_id, meta, chat_id=chat_id,
    ):
        return True
    meta["finalization_requested_at"] = now
    return False


def _kill_and_confirm_worker_dead(proc: Any, worker_id: int, task_id: str) -> bool:
    """Kill+join a timed-out worker (off the queue lock) and return True ONLY when it is PROVABLY
    dead. The Variant-A invariant gates the terminal write + retry on the original being dead, so a
    final hard kill is attempted if the first did not confirm death, and an is_alive() that raises is
    treated as still-alive (fail-closed) — the caller then refuses to enqueue a colliding retry."""
    from supervisor import queue as _q

    try:
        from ouroboros.platform_layer import kill_pid_tree

        # Spare deliberately-kept services so a timeout kill leaves verifier-facing services alive;
        # they reparent to init and the custody reaper governs them.
        _keep = _q._kept_service_pids()
        if proc is not None:
            if getattr(proc, "pid", None):
                kill_pid_tree(proc.pid, exclude_pids=_keep)
            elif proc.is_alive():
                proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive() and getattr(proc, "pid", None):
                kill_pid_tree(proc.pid, exclude_pids=_keep)
                proc.join(timeout=2)
    except Exception:
        log.warning("Reaper: failed to terminate worker %d for task %s", worker_id, task_id, exc_info=True)

    if proc is None:
        return True
    try:
        if not proc.is_alive():
            return True
    except Exception:
        return False  # cannot confirm -> fail closed (treat as still alive)
    try:
        from ouroboros.platform_layer import kill_pid_tree

        if getattr(proc, "pid", None):
            kill_pid_tree(proc.pid, exclude_pids=_q._kept_service_pids())
        proc.join(timeout=2)
        return not proc.is_alive()
    except Exception:
        log.debug("Reaper: final hard-kill of worker %d failed for %s", worker_id, task_id, exc_info=True)
        return False


def _retain_reaper_terminalization_custody(
    q: Any,
    rows: list[tuple[Dict[str, Any], str]],
    *,
    reason: str,
) -> list[str]:
    """Keep failed terminal writes in the existing non-dispatchable custody."""
    from supervisor import workers

    retained_ids: list[str] = []
    with q._queue_lock:
        for source_task, raw_task_id in rows:
            target_id = str(raw_task_id or "").strip()
            if not target_id or target_id in q.RUNNING:
                continue
            if any(
                isinstance(row, dict)
                and str(row.get("id") or "") == target_id
                and isinstance(row.get("_terminalization_retry"), dict)
                for row in q.PENDING
            ):
                retained_ids.append(target_id)
                continue
            marker = workers._retain_terminalization_retry_task(
                source_task,
                target_id,
                reason=reason,
                status="failed",
                trigger="reaper_retry_terminal_persistence",
            )
            q.PENDING.append(marker)
            retained_ids.append(target_id)
        if retained_ids:
            q.sort_pending()
    if retained_ids:
        try:
            q.persist_queue_snapshot(reason="reaper_terminalization_retry")
        except Exception:
            log.warning(
                "Reaper: failed to persist terminalization custody for %s",
                retained_ids,
                exc_info=True,
            )
    return retained_ids


def _run_retry_admission_transaction(
    q: Any,
    task: Dict[str, Any],
    retried: Dict[str, Any],
    *,
    task_id: str,
    retry_task_id: str,
    terminal_reason: str,
    recon_fields: Dict[str, Any],
    runtime_sec: float,
    unreconciled_runs: Optional[list],
    salvage_note: str,
) -> tuple[Dict[str, str], str]:
    """Publish one retry admission under the queue -> cancel lock order."""
    from ouroboros.task_results import (
        STATUS_CANCELLED,
        STATUS_FAILED,
        STATUS_INTERRUPTED,
        STATUS_SCHEDULED,
        _TRULY_TERMINAL_STATUSES,
        load_task_result,
        write_task_result,
    )

    admitted: Dict[str, Any] = {}
    admission_selected = False
    suppression: Dict[str, str] = {}
    admission_block = ""
    try:
        from ouroboros.cancel_intents import (
            _validated_retry_root_cancel_key,
            active_intents,
            cancellation_projection_lock,
        )

        # Match assignment's lock order (queue -> cancel projection). Holding
        # only the projection lock while enqueue_task takes the queue lock would
        # invert _drop_cancelled_pending and deadlock with ordinary dispatch.
        with q._queue_lock:
            with cancellation_projection_lock(q.DRIVE_ROOT):
                intents = active_intents(q.DRIVE_ROOT, strict=True)
                if not isinstance(intents, dict):
                    raise TypeError(
                        "cancel-intent authority returned a non-object projection"
                    )
                current = load_task_result(q.DRIVE_ROOT, task_id, strict=True) or {}
                current_status = str(current.get("status") or "")
                if current_status in _TRULY_TERMINAL_STATUSES:
                    suppression = {
                        "kind": "terminal_result",
                        "target": task_id,
                        "status": current_status,
                    }
                else:
                    retry_root = _validated_retry_root_cancel_key(
                        q.DRIVE_ROOT, task_id, task_hint=task,
                    )
                    cancel_target = next(
                        (
                            candidate
                            for candidate in dict.fromkeys(
                                (task_id, retry_task_id, retry_root)
                            )
                            if candidate and candidate in intents
                        ),
                        "",
                    )
                    if cancel_target:
                        # The worker was already kill+join confirmed before
                        # this final boundary.  Cancellation arrived after the
                        # timeout decision but before retry publication: the
                        # physical attempt therefore keeps its honest timeout
                        # failure, while the owner's intent wins admission of
                        # the successor.  Terminalize NOW, under the same
                        # queue->cancel lock pair, so a concurrent cascade can
                        # never settle the logical intent over a non-live row
                        # still claiming RUNNING.
                        stored_current = write_task_result(
                            q.DRIVE_ROOT,
                            task_id,
                            STATUS_FAILED,
                            strict_existing_dict=True,
                            reason_code=terminal_reason,
                            outcome_axes=terminal_outcome_axes(
                                lifecycle=STATUS_FAILED,
                                execution=EXECUTION_INFRA_FAILED,
                                reason_code=terminal_reason,
                                review_trigger="supervisor_terminal",
                            ),
                            **(
                                {"delegated_runs_unreconciled": unreconciled_runs}
                                if unreconciled_runs else {}
                            ),
                            **recon_fields,
                            result=(
                                f"Task killed by {terminal_reason} after "
                                f"{int(runtime_sec)}s. Retry suppressed because "
                                "cancellation won the admission boundary."
                                + str(salvage_note or "")
                            ),
                        )
                        stored_status = str(stored_current.get("status") or "")
                        if stored_status not in _TRULY_TERMINAL_STATUSES:
                            raise RuntimeError(
                                "cancel-suppressed physical retry did not settle"
                            )
                        suppression = {
                            "kind": "cancel_intent",
                            "target": cancel_target,
                        }
                    else:
                        admission_selected = True
                        admitted = q.enqueue_task(retried, front=True)
                if admission_selected:
                    admission_block = (
                        str(admitted.get("_admission_blocked") or "")
                        if isinstance(admitted, dict) else ""
                    )
                    if not admission_block:
                        try:
                            if retry_task_id and retry_task_id != task_id:
                                write_task_result(
                                    q.DRIVE_ROOT,
                                    retry_task_id,
                                    STATUS_SCHEDULED,
                                    strict_existing_dict=True,
                                    reason_code=f"{terminal_reason}_retry_scheduled",
                                    outcome_axes=terminal_outcome_axes(
                                        lifecycle=STATUS_SCHEDULED,
                                        execution="pending",
                                        reason_code=f"{terminal_reason}_retry_scheduled",
                                        review_trigger="supervisor_terminal",
                                    ),
                                    supersedes_task_id=task_id,
                                    original_task_id=task_id,
                                    result=f"Retry scheduled after {terminal_reason}.",
                                    parent_task_id=task.get("parent_task_id"),
                                    root_task_id=task.get("root_task_id") or task_id,
                                    delegation_role=task.get("delegation_role"),
                                    timeout_retry_from=task_id,
                                    timeout_retry_at=utc_now_iso(),
                                    description=task.get("description"),
                                    context=task.get("context"),
                                    workspace_root=task.get("workspace_root"),
                                    workspace_mode=task.get("workspace_mode"),
                                    memory_mode=task.get("memory_mode"),
                                    metadata=(
                                        task.get("metadata")
                                        if isinstance(task.get("metadata"), dict) else {}
                                    ),
                                )
                            stored_old = write_task_result(
                                q.DRIVE_ROOT,
                                task_id,
                                STATUS_INTERRUPTED,
                                strict_existing_dict=True,
                                reason_code=f"{terminal_reason}_retry",
                                outcome_axes=terminal_outcome_axes(
                                    lifecycle=STATUS_INTERRUPTED,
                                    execution=EXECUTION_INFRA_FAILED,
                                    reason_code=f"{terminal_reason}_retry",
                                    review_trigger="supervisor_terminal",
                                ),
                                superseded_by=(
                                    retry_task_id
                                    if retry_task_id and retry_task_id != task_id else ""
                                ),
                                retry_task_id=retry_task_id or task_id,
                                **(
                                    {"delegated_runs_unreconciled": unreconciled_runs}
                                    if unreconciled_runs else {}
                                ),
                                **recon_fields,
                                result=(
                                    f"Task killed by {terminal_reason} after "
                                    f"{int(runtime_sec)}s. Retrying."
                                    + str(salvage_note or "")
                                ),
                            )
                            stored_status = str(stored_old.get("status") or "")
                            if stored_status in _TRULY_TERMINAL_STATUSES:
                                suppression = {
                                    "kind": "terminal_result",
                                    "target": task_id,
                                    "status": stored_status,
                                }
                                admission_block = "terminal_result_race"
                            elif stored_status != STATUS_INTERRUPTED:
                                raise RuntimeError(
                                    "historical retry result did not become interrupted"
                                )
                        except Exception:
                            admission_block = "retry_result_persistence_failed"
                            log.error(
                                "Reaper: retry result publication failed for %s -> %s",
                                task_id,
                                retry_task_id,
                                exc_info=True,
                            )
                        if admission_block:
                            for index, row in enumerate(list(q.PENDING)):
                                if row is admitted:
                                    q.PENDING.pop(index)
                                    break
                            if suppression and retry_task_id and retry_task_id != task_id:
                                retry_terminal = write_task_result(
                                    q.DRIVE_ROOT,
                                    retry_task_id,
                                    STATUS_CANCELLED,
                                    strict_existing_dict=True,
                                    reason_code="terminal_result_retry_suppressed",
                                    outcome_axes=terminal_outcome_axes(
                                        lifecycle=STATUS_CANCELLED,
                                        execution=STATUS_CANCELLED,
                                        reason_code="terminal_result_retry_suppressed",
                                        review_trigger="supervisor_terminal",
                                    ),
                                    supersedes_task_id=task_id,
                                    original_task_id=task_id,
                                    timeout_retry_from=task_id,
                                    result=(
                                        "Retry was cancelled before admission because "
                                        "the original attempt had already settled."
                                    ),
                                )
                                suppression["retry_status"] = str(
                                    retry_terminal.get("status") or ""
                                )
    except Exception:
        admission_block = "cancel_intent_authority_unreadable"
        log.error(
            "Reaper: retry admission could not prove cancel-intent authority "
            "for %s -> %s",
            task_id,
            retry_task_id,
            exc_info=True,
        )
    return suppression, admission_block


def _enqueue_retry(
    q: Any,
    task: Dict[str, Any],
    *,
    task_id: str,
    retry_task_id: str,
    attempt: int,
    terminal_reason: str,
    recon_fields: Dict[str, Any],
    runtime_sec: float = 0.0,
    unreconciled_runs: Optional[list] = None,
    salvage_note: str = "",
) -> tuple[bool, int, str, Dict[str, str]]:
    """Atomically admit a dead task's retry or suppress it.

    The queue row and reciprocal result lineage are published only inside the
    final queue->cancel locked transition.  If cancellation/terminal truth wins
    the boundary, no successor result or historical link is created.  If retry
    admission wins, a later SINGLE cancel resolves the complete durable chain
    under the same projection lock and targets the physical leaf.
    """
    from ouroboros.task_results import (
        STATUS_FAILED,
        load_task_result,
        write_task_result,
    )

    retried = dict(task)
    retried["original_task_id"] = task_id
    retried["id"] = retry_task_id or task_id
    retried["_attempt"] = attempt + 1
    retried["timeout_retry_from"] = task_id
    retried["timeout_retry_at"] = utc_now_iso()
    if retry_task_id and retry_task_id != task_id:
        from ouroboros.artifacts import (
            handoff_task_attachments_for_retry,
            task_artifact_dir_path,
        )
        from ouroboros.owner_mailbox import cleanup_task_mailbox, copy_owner_mailbox_for_retry

        task_drive = q._task_drive_for_task(task, task_id)
        replacements, attachment_error = handoff_task_attachments_for_retry(
            task_drive, task_id, retry_task_id, retried,
        )
        mailbox_ok = not attachment_error and copy_owner_mailbox_for_retry(
            task_drive, task_id, retry_task_id, path_replacements=replacements,
        )
        if not mailbox_ok:
            failure = "attachment" if attachment_error else "mailbox"
            blocked_reason = f"{terminal_reason}_retry_{failure}_handoff_failed"
            message = (
                "Retry refused because durable task inputs could not be carried "
                "to the new physical task id."
                + str(salvage_note or "")
            )
            log.error(
                "Reaper: %s handoff failed for retry %s -> %s: %s",
                failure, task_id, retry_task_id, attachment_error,
            )
            shutil.rmtree(
                task_artifact_dir_path(task_drive, retry_task_id), ignore_errors=True,
            )
            cleanup_task_mailbox(task_drive, retry_task_id)
            outcome = terminal_outcome_axes(
                lifecycle=STATUS_FAILED,
                execution=EXECUTION_INFRA_FAILED,
                reason_code=blocked_reason,
                review_trigger="supervisor_terminal",
            )
            write_task_result(
                q.DRIVE_ROOT,
                task_id,
                STATUS_FAILED,
                reason_code=blocked_reason,
                outcome_axes=outcome,
                **recon_fields,
                result=message,
            )
            write_task_result(
                q.DRIVE_ROOT,
                retry_task_id,
                STATUS_FAILED,
                reason_code=blocked_reason,
                outcome_axes=outcome,
                supersedes_task_id=task_id,
                original_task_id=task_id,
                result=message,
            )
            return False, attempt, blocked_reason, {}
    suppression, admission_block = _run_retry_admission_transaction(
        q,
        task,
        retried,
        task_id=task_id,
        retry_task_id=retry_task_id,
        terminal_reason=terminal_reason,
        recon_fields=recon_fields,
        runtime_sec=runtime_sec,
        unreconciled_runs=unreconciled_runs,
        salvage_note=salvage_note,
    )

    if suppression:
        if retry_task_id and retry_task_id != task_id:
            cleanup_task_mailbox(q._task_drive_for_task(task, task_id), retry_task_id)
            shutil.rmtree(
                task_artifact_dir_path(
                    q._task_drive_for_task(task, task_id), retry_task_id,
                ),
                ignore_errors=True,
            )
        reason = (
            "cancel_pending_retry_suppressed"
            if suppression.get("kind") == "cancel_intent"
            else "terminal_result_retry_suppressed"
        )
        return False, attempt, reason, suppression
    if not admission_block:
        return True, attempt + 1, terminal_reason, {}
    if retry_task_id and retry_task_id != task_id:
        cleanup_task_mailbox(q._task_drive_for_task(task, task_id), retry_task_id)
        shutil.rmtree(
            task_artifact_dir_path(
                q._task_drive_for_task(task, task_id), retry_task_id,
            ),
            ignore_errors=True,
        )

    blocked_reason = f"{terminal_reason}_retry_admission_blocked"
    outcome = terminal_outcome_axes(
        lifecycle=STATUS_FAILED,
        execution=EXECUTION_INFRA_FAILED,
        reason_code=blocked_reason,
        review_trigger="supervisor_terminal",
    )
    message = (
        f"Retry blocked by the active {admission_block} admission fence."
        + str(salvage_note or "")
    )
    terminalization_rows: list[tuple[Dict[str, Any], str]] = []
    try:
        write_task_result(
            q.DRIVE_ROOT,
            task_id,
            STATUS_FAILED,
            reason_code=blocked_reason,
            outcome_axes=outcome,
            **recon_fields,
            result=message,
        )
    except Exception:
        terminalization_rows.append((task, task_id))
        log.error(
            "Reaper: terminal retry-suppression write failed for %s",
            task_id,
            exc_info=True,
        )
    if retry_task_id and retry_task_id != task_id:
        try:
            retry_exists = isinstance(
                load_task_result(q.DRIVE_ROOT, retry_task_id, strict=True), dict,
            )
        except Exception:
            retry_exists = False
        if retry_exists:
            try:
                write_task_result(
                    q.DRIVE_ROOT,
                    retry_task_id,
                    STATUS_FAILED,
                    reason_code=blocked_reason,
                    outcome_axes=outcome,
                    supersedes_task_id=task_id,
                    original_task_id=task_id,
                    result=message,
                )
            except Exception:
                terminalization_rows.append((retried, retry_task_id))
                log.error(
                    "Reaper: retry-leaf terminal suppression write failed for %s",
                    retry_task_id,
                    exc_info=True,
                )
    if terminalization_rows:
        try:
            retained = _retain_reaper_terminalization_custody(
                q,
                terminalization_rows,
                reason=message,
            )
        except Exception:
            # A second failure must not collapse this into the ordinary
            # no-retry path: that path publishes task_done even though no
            # terminal row exists.  Keep a typed suppression result so the
            # caller withholds the event; any marker appended before the
            # exception remains available to the existing assignment-time
            # terminalization retry.
            retained = []
            log.error(
                "Reaper: failed to retain terminalization custody for %s",
                [target_id for _task, target_id in terminalization_rows],
                exc_info=True,
            )
        return False, attempt, blocked_reason, {
            "kind": "terminal_persistence_failed",
            "targets": ",".join(
                retained
                or [target_id for _task, target_id in terminalization_rows]
            ),
        }
    return False, attempt, blocked_reason, {}


def _settle_retry_cancel_handoff(
    q: Any, task_id: str, retry_task_id: str, cancel_target: str,
) -> Dict[str, str]:
    """Give a cancel that won retry admission custody of every physical id.

    Retry admission was suppressed before successor publication.  Route the
    winning intent through its existing policy owner: cascade keeps the subtree
    postcondition, graceful stop gets its policy-aware hold, and an immediate
    single request uses ordinary custody.  No second intent is minted.
    """
    outcomes: Dict[str, str] = {}
    if not cancel_target:
        return outcomes
    try:
        from ouroboros.cancel_intents import (
            SCOPE_CASCADE,
            active_intent,
        )
        from supervisor.owner_stop import (
            OWNER_STOP_HOLDING,
            sweep_owner_stop_hold,
        )
        import time as _time

        intent = active_intent(q.DRIVE_ROOT, cancel_target) or {}
        if str(intent.get("scope") or "") == SCOPE_CASCADE:
            outcomes[cancel_target] = (
                q.CANCEL_CANCELLED
                if q.cancel_task_by_id(cancel_target, cascade=True)
                else q.CANCEL_FAILED
            )
        elif sweep_owner_stop_hold(
            q, cancel_target, intent, now=_time.time(),
        ):
            outcomes[cancel_target] = OWNER_STOP_HOLDING
        else:
            outcomes[cancel_target] = str(q.cancel_task_custody(cancel_target))
    except Exception:
        outcomes[cancel_target] = "failed"
        log.error(
            "Reaper: cancellation custody failed for suppressed retry id %s",
            cancel_target,
            exc_info=True,
        )
    return outcomes


def _emit_cancel_suppressed_retry_task_done(
    q: Any,
    workers_mod: Any,
    task: Dict[str, Any],
    task_id: str,
    task_type: str,
    cancel_target: str,
    handoff_outcomes: Dict[str, str],
    terminal_metadata: Any,
) -> bool:
    """Publish only the physical terminal event after cancel handoff is durable.

    The cascade owns the one owner-facing summary, so this deliberately does
    not call the self-finalized delivery helper.  The fast-path card event is
    safe only after the winning intent has left the active projection (which
    proves its answer/summary was durably owed) and the physical retry row is
    settled.
    """
    outcome = str(handoff_outcomes.get(cancel_target) or "")
    if outcome not in {q.CANCEL_CANCELLED, q.CANCEL_ALREADY_SETTLED}:
        return False
    try:
        from ouroboros.cancel_intents import active_intent
        from ouroboros.cost_projection import carry_cost_meta
        from ouroboros.task_results import load_task_result
        from ouroboros.task_status import SETTLED_STATUSES

        if active_intent(q.DRIVE_ROOT, cancel_target, strict=True) is not None:
            return False
        stored = load_task_result(q.DRIVE_ROOT, task_id, strict=True) or {}
        status = str(stored.get("status") or "")
        if status not in SETTLED_STATUSES:
            return False
        cost_fields = carry_cost_meta(stored)
        if not cost_fields:
            cost_fields = q.reconstruct_task_cost(task_id, fields=True)
        workers_mod.get_event_q().put({
            "type": "task_done",
            "task_id": task_id,
            "task_type": task_type,
            "chat_id": (
                int(task.get("chat_id") or 0)
                if isinstance(task, dict) else 0
            ),
            "status": status,
            "reason_code": str(stored.get("reason_code") or ""),
            "outcome_axes": (
                dict(stored["outcome_axes"])
                if isinstance(stored.get("outcome_axes"), dict)
                else terminal_outcome_axes(
                    lifecycle=status,
                    execution=EXECUTION_INFRA_FAILED,
                    reason_code=str(stored.get("reason_code") or status),
                    review_trigger="supervisor_terminal",
                )
            ),
            **cost_fields,
            "metadata": terminal_metadata,
        })
        return True
    except Exception:
        log.warning(
            "Reaper: failed to publish cancel-suppressed task_done for %s",
            task_id,
            exc_info=True,
        )
        return False


def _incident_chat_id(task: Any, owner_chat_id: int) -> Optional[int]:
    """C4: an incident notice belongs to the TASK'S OWN chat; the owner chat is
    only the absent-binding fallback (the same precedence queue.py already uses
    for grace episodes).

    Routed through the ONE notification normalizer, so membership decides instead
    of truthiness: chat **0 is the Skill Review panel** and a task bound there
    keeps its incident there (the old `> 0` test both re-routed it to the owner
    AND then refused to send, because `if owner_chat_id:` drops 0 as well). A
    negative (A2A/internal) chat is suppressed and falls through to the owner
    fallback; ``None`` means there is no deliverable route at all."""
    from supervisor.message_bus import notification_chat_route

    return notification_chat_route(
        task.get("chat_id") if isinstance(task, dict) else None,
        # A 0/absent owner chat is "not configured", not the panel — only an
        # explicit TASK binding routes to 0.
        owner_chat_id or None,
    )


def _stop_detail(ceiling_reached: bool, deadline_reached: bool, orchestrator: bool) -> str:
    """The one human sentence for WHY a reaped task will not be retried."""
    if ceiling_reached:
        return "Absolute ceiling reached; task stopped."
    if deadline_reached:
        return "Absolute deadline reached; task stopped."
    if orchestrator:
        return ("Idle with live children (orchestrator); stopped without a "
                "blind retry to avoid replaying the subtree.")
    return "Retry limit exhausted, task stopped."


def _hold_wedged_worker(task_id: str, task_type: str, worker_id: int, terminal_reason: str,
                        runtime_sec: float, notify_chat_id: Optional[int]) -> None:
    """Strict fail-closed handling for a worker that would not confirm dead after repeated kills:
    persist a durable STATUS_RUNNING result so the task is reconcilable on the next generation (the
    custody reaper terminalizes the orphan after a worker_boot) instead of vanishing into limbo, then
    record a `task_reaper_wedged` event + an owner /restart hint. The caller leaves the slot
    reaping=True; this writes no terminal/retry/task_done and clears no flag, so it cannot race the
    still-live worker. The STATUS_RUNNING write is rank-2 (below the cancel-intent floor), so the
    monotonic merge guard drops it if the worker self-finalized a terminal/cancel result first.
    Never raises."""
    from supervisor import queue as _q

    try:
        from ouroboros.task_results import STATUS_RUNNING, write_task_result

        write_task_result(
            _q.DRIVE_ROOT, task_id, STATUS_RUNNING,
            reason_code="reaper_wedged_worker_alive",
            result=(f"Timed-out worker for task {task_id} did not confirm dead after kill/join; slot "
                    "held reaping, task left running pending custody reap on the next generation."),
        )
    except Exception:
        log.debug("Reaper: failed to persist STATUS_RUNNING for wedged task %s", task_id, exc_info=True)
    log.error("Reaper: worker %d for task %s did NOT confirm dead after kill/join; holding the slot "
              "reaping (unavailable) and leaving the task RUNNING — no terminal/task_done/retry/respawn "
              "while the process may still be alive.", worker_id, task_id)
    try:
        append_jsonl(
            _q.DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": utc_now_iso(), "type": "task_reaper_wedged",
                "task_id": task_id, "task_type": task_type, "worker_id": worker_id,
                "terminal_reason": terminal_reason, "runtime_sec": round(runtime_sec, 2),
            },
        )
    except Exception:
        log.debug("Reaper: failed to log task_reaper_wedged for %s", task_id, exc_info=True)
    if notify_chat_id is not None:
        try:
            send_with_budget(
                notify_chat_id,
                (
                    f"⚠️ A timed-out worker (task {task_id}) did not die after repeated kills. Its slot is "
                    f"held unavailable and the task is left running to avoid racing a still-live process. "
                    f"If this persists, /restart to recover the slot."
                ),
                is_progress=True,
                task_id=task_id,
                progress_meta={
                    "task_incident": "task_reaper_wedged",
                    "toast_once": f"{task_id}:task_reaper_wedged:{worker_id}:{terminal_reason}",
                },
            )
        except Exception:
            log.debug("Reaper: failed to send wedged owner notification for %s", task_id, exc_info=True)


def _emit_reap_task_done(
    workers_mod: Any, task: Dict[str, Any], task_id: str, task_type: str,
    terminal_reason: str, recon_fields: Dict[str, Any], terminal_metadata: Any,
) -> None:
    """The non-retry reap's terminal event — emitted AFTER the salvage delivery
    (AR2-5a): the salvage registers the owner's answer as OWED in the durable
    outbox first, so a crash between the two can no longer resolve the card
    while losing the answer. task_done itself is covered by the durable
    terminal result plus boot reconciliation. Fail-soft."""
    try:
        done_chat_id = int(task.get("chat_id") or 0) if isinstance(task, dict) else 0
        workers_mod.get_event_q().put({
                "type": "task_done", "task_id": task_id, "task_type": task_type,
                "chat_id": done_chat_id, "status": "failed", "reason_code": terminal_reason,
                "outcome_axes": terminal_outcome_axes(lifecycle="failed", execution=EXECUTION_INFRA_FAILED, reason_code=terminal_reason, review_trigger="supervisor_terminal"),
                **recon_fields,
                "metadata": terminal_metadata,
        })
    except Exception:
        log.debug("Reaper: failed to emit task_done for %s", task_id, exc_info=True)


def _deliver_reap_salvage(
    _q: Any, task: Dict[str, Any], task_id: str, terminal_reason: str,
    unreconciled_runs: Optional[list] = None,
) -> None:
    """A2: a NON-RETRY reap delivers the salvaged answer through the shared
    durable outbox seam (a retryable reap deliberately delivers nothing — the
    retry will produce the real answer). Same seam, same dedupe as the cancel
    path; roots only, a child's result flows to its parent. Fail-soft.
    ``unreconciled_runs`` (GR5-2) rides the same message as the kill path's
    disclosure — a reap that left delegated runs open must say so."""
    if str(task.get("delegation_role") or "") == "subagent":
        return
    try:
        from ouroboros.observability import latest_llm_response_text, preserved_salvage_path
        from supervisor.terminal_delivery import deliver_unreviewed_salvage

        salvage_text = latest_llm_response_text(
            pathlib.Path(_q._task_drive_for_task(task, task_id)), task_id,
        )
        deliver_unreviewed_salvage(
            pathlib.Path(_q.DRIVE_ROOT), task, task_id,
            outcome=f"stopped by {terminal_reason}",
            salvaged_text=salvage_text,
            preserved_path=preserved_salvage_path(pathlib.Path(_q.DRIVE_ROOT), task_id),
            unreconciled_runs=list(unreconciled_runs or []),
        )
    except Exception:
        log.debug("Reaper: salvage delivery failed for %s", task_id, exc_info=True)


def _finish_self_finalized_task(
    _q: Any, workers_mod: Any, task: Dict[str, Any], task_id: str, task_type: str,
    self_status: str, _existing: Optional[Dict[str, Any]], terminal_metadata: Any,
    unreconciled_runs: Optional[list] = None,
) -> None:
    """Honor a worker's OWN terminal result found after the kill (never clobber
    it or enqueue a retry) and finish everything its death interrupted.

    A mirrored child result (copy_child_task_result sets artifact_status to
    'finalizing' for workspace tasks) still needs the artifact finalization the
    normal task_done path runs in _handle_task_done. The reaper already
    terminalized the task, so it is no longer in RUNNING and that path finds no
    task to finalize — complete it here. Rescue ONLY a stuck non-terminal
    artifact state: re-running finalize on an already-terminal result can
    regress it to FAILED (e.g. the workspace was cleaned up). Readonly
    subagents have no durable artifacts and are skipped (shared gate).
    """
    try:
        from ouroboros.headless import (
            ARTIFACT_STATUS_FINALIZING,
            ARTIFACT_STATUS_PENDING,
            finalize_task_artifacts,
            task_is_readonly_subagent,
        )

        _art = str((_existing or {}).get("artifact_status") or "").strip().lower()
        if _art in {ARTIFACT_STATUS_PENDING, ARTIFACT_STATUS_FINALIZING} and not task_is_readonly_subagent(task):
            finalize_task_artifacts(pathlib.Path(_q.DRIVE_ROOT), task)
    except Exception:
        log.debug("Reaper: artifact finalize for self-finalized %s failed", task_id, exc_info=True)

    # GR3-5: the worker may equally have died BEFORE its final answer was
    # delivered (or before the buffered send ever left) — an already-terminal
    # recovery that emits only task_done resolves the card while the owner's
    # answer stays on disk forever. Route the recovery through the SAME
    # owed-registration delivery seam the cancel miss lane uses:
    # owed-before-enqueued, deduped by the shared delivery_id, so a copy the
    # worker already delivered is suppressed durably. Fail-soft.
    try:
        from supervisor.terminal_delivery import deliver_miss_lane_outcome

        deliver_miss_lane_outcome(
            pathlib.Path(_q.DRIVE_ROOT),
            pathlib.Path(_q._task_drive_for_task(task, task_id)),
            # The durable row wins; the queue task row backfills routing
            # facts (chat/lineage/role) a sparse result may not carry.
            {**(task if isinstance(task, dict) else {}), **(_existing or {})},
            task_id, self_status,
            unreconciled_runs=list(unreconciled_runs or []),
        )
    except Exception:
        log.debug("Reaper: terminal delivery for self-finalized %s failed", task_id, exc_info=True)

    # The worker may have died before emitting its task_done (and the crash
    # detector now skips reaping slots): emit an idempotent task_done so the
    # UI card resolves.
    try:
        done_chat_id = int(task.get("chat_id") or 0) if isinstance(task, dict) else 0
        workers_mod.get_event_q().put({
                "type": "task_done", "task_id": task_id, "task_type": task_type,
                "chat_id": done_chat_id, "status": self_status,
                "reason_code": str((_existing or {}).get("reason_code") or ""),
                "metadata": terminal_metadata,
        })
    except Exception:
        log.debug("Reaper: failed to emit task_done for self-finalized %s", task_id, exc_info=True)


def _load_post_kill_terminal_result(
    q: Any, task: Dict[str, Any], task_id: str,
) -> tuple[str, Optional[Dict[str, Any]]]:
    """Return terminal truth that won the worker-death boundary, if any."""
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, load_task_result

    self_status = ""
    existing: Optional[Dict[str, Any]] = None
    try:
        existing = load_task_result(q.DRIVE_ROOT, task_id)
        if existing and str(existing.get("status") or "") in _TRULY_TERMINAL_STATUSES:
            self_status = str(existing.get("status") or "")
    except Exception:
        log.debug("Reaper: post-kill terminal re-check failed for %s", task_id, exc_info=True)
    # Forked/workspace/subagent tasks self-finalize on the CHILD drive and are copied back
    # only on task_done; a worker that died after writing its child result but before
    # copy-back would be missed by the parent-drive check above. Mirror the child result
    # back and honor it (no interrupted/failed clobber, no duplicate retry).
    if not self_status:
        try:
            from ouroboros.headless import copy_child_task_result

            child = copy_child_task_result(pathlib.Path(q.DRIVE_ROOT), task)
            if child and str(child.get("status") or "") in _TRULY_TERMINAL_STATUSES:
                existing = child
                self_status = str(child.get("status") or "")
        except Exception:
            log.debug("Reaper: child-drive terminal re-check failed for %s", task_id, exc_info=True)
    return self_status, existing


def _respawn_after_reap(q: Any, workers_mod: Any, worker_id: int) -> None:
    """Reopen a reaped slot, leaving crash recovery available on failure."""
    # respawn_worker owns the lifecycle race with shutdown and starts the child
    # outside _queue_lock, so a fork can never inherit the RLock from this thread.
    try:
        workers_mod.respawn_worker(worker_id)
    except Exception:
        log.warning(
            "Reaper: respawn failed for worker %d; clearing reaping for recovery",
            worker_id,
            exc_info=True,
        )
        try:
            with q._queue_lock:
                worker = workers_mod.WORKERS.get(worker_id)
                if worker is not None:
                    worker.reaping = False
        except Exception:
            pass
    try:
        q.persist_queue_snapshot(reason="worker_respawn_after_reap")
    except Exception:
        log.debug("Reaper: failed to persist queue snapshot after respawn", exc_info=True)


def reap_timed_out_task(job: Dict[str, Any]) -> None:
    """Full teardown for a timed-out task, run OFF the supervisor loop (Variant A).

    Order is load-bearing for correctness: kill+join the worker process FIRST, then gate the
    WHOLE post-kill sequence (terminal write, task_done, retry, respawn) on confirmed death.
    Because the original process is provably dead before any of them, a still-alive worker can
    never race a concurrently-assigned retry (which, for a subagent, reuses the same task
    id/drive) or have its result clobbered. If it does NOT confirm dead, the sequence is skipped
    (strict fail-closed): the slot is held ``reaping`` and a durable STATUS_RUNNING result is
    persisted via ``_hold_wedged_worker`` so the task is reconciled — not lost in limbo — on the
    next generation. A POST-KILL already-terminal re-check honors a worker that self-finalized at
    the idle boundary instead of clobbering its result or running a duplicate. The loop already
    popped RUNNING/cleared busy_task_id and marked the slot ``reaping`` under the lock; on
    confirmed death respawn_worker installs a fresh reaping=False Worker, re-opening the slot.
    """
    from supervisor import queue as _q
    from supervisor import workers as workers_mod

    worker_id = int(job.get("worker_id")) if job.get("worker_id") is not None else -1
    proc = job.get("proc")
    task_id = str(job.get("task_id") or "")
    task = job.get("task") if isinstance(job.get("task"), dict) else {}
    task_metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    terminal_metadata = workers_mod.terminal_task_metadata(task_metadata)
    task_type = str(job.get("task_type") or "")
    terminal_reason = str(job.get("terminal_reason") or "idle_timeout")
    attempt = int(job.get("attempt") or 1)
    owner_chat_id = int(job.get("owner_chat_id") or 0)
    runtime_sec = float(job.get("runtime_sec") or 0.0)
    hb_lag_sec = float(job.get("hb_lag_sec") or 0.0)
    hb_stale = bool(job.get("hb_stale"))
    deadline_reached = bool(job.get("deadline_reached"))
    ceiling_reached = bool(job.get("ceiling_reached"))
    orchestrator = bool(job.get("orchestrator"))
    will_retry = bool(job.get("will_retry"))
    retry_task_id = str(job.get("retry_task_id") or "")
    incident_toast_once = str(job.get("incident_toast_once") or f"{task_id}:{terminal_reason}:{attempt}")
    # 1. Kill first; every terminal write, retry, and respawn is gated on confirmed death.
    if not _kill_and_confirm_worker_dead(proc, worker_id, task_id):
        # Fully fail-closed: do NOTHING downstream that could race a still-live worker. Leave the
        # slot reaping=True (the loop already cleared busy_task_id; clearing reaping would let the
        # crash detector treat the live, non-busy orphan as a healthy IDLE worker and assign it a new
        # task). A durable STATUS_RUNNING result is persisted so the task is reconciled (not lost in
        # limbo) on the next generation — the custody reaper terminalizes the orphan after a
        # worker_boot. Surface it loudly so the owner can /restart if truly wedged.
        _hold_wedged_worker(task_id, task_type, worker_id, terminal_reason, runtime_sec,
                            _incident_chat_id(task, owner_chat_id))
        return

    # Confirmed dead: the task leaves the not-provably-dead registry, and the
    # dead worker can never end a queue-owned acceptance fence it opened —
    # release it NOW (matched on the fence's recorded owner id, so a reaped
    # child never drops its still-reviewing root's fence), before the retry
    # admission below, which the stale fence would otherwise terminalize.
    _forget_task_reaping(task_id)
    workers_mod._reconcile_confirmed_dead_review_owner(
        int(getattr(proc, "pid", 0) or 0)
    )

    try:
        from supervisor.queue import release_acceptance_fence_for_dead_owner

        release_acceptance_fence_for_dead_owner(task_id)
    except Exception:
        log.debug("Reaper: acceptance-fence release failed for %s", task_id, exc_info=True)

    try:
        from ouroboros.tools.services import archive_task_service_logs

        archive_task_service_logs(pathlib.Path(_q.DRIVE_ROOT), task_id, task)
    except Exception:
        log.debug("Reaper: failed to archive service logs for %s", task_id, exc_info=True)

    # GR5-2: the killed worker's graceful ``release_task_runs`` never ran, so
    # reconcile delegated-run custody NOW (the cancel kill path's seam), BEFORE
    # the retry/respawn decision; still-open runs are disclosed on the reap
    # outcome. Custody reconciliation only — the reaper mints no cancel intents.
    from supervisor.cancel_publication import _reconcile_delegated_runs_on_kill

    unreconciled = _reconcile_delegated_runs_on_kill(_q, task_id)

    from ouroboros.task_results import (
        STATUS_FAILED,
        load_task_result,
        write_task_result,
    )

    # 2. POST-KILL already-terminal re-check (the process is dead; this decision is final).
    self_status, _existing = _load_post_kill_terminal_result(_q, task, task_id)

    if self_status:
        _finish_self_finalized_task(
            _q, workers_mod, task, task_id, task_type, self_status,
            _existing, terminal_metadata, unreconciled,
        )
    else:
        # 3. Reconstruct real cost/rounds from durable llm_usage (the killed worker never
        #    finalized). A failure here cannot abort teardown before the slot is respawned.
        try:
            recon_fields = _q.reconstruct_task_cost(task_id, fields=True)
        except Exception:
            log.error("Reaper: task cost authority failed for %s", task_id, exc_info=True)
            recon_fields = {
                "cost_accounting_status": "unavailable", "cost_final": False,
                "cost_accounting_error": "ledger_unavailable",
            }

        # Salvage the last persisted assistant text so a hard kill surfaces real progress.
        salvage_note = ""
        try:
            from ouroboros.observability import salvaged_output_note
            salvage_note = salvaged_output_note(
                _q._task_drive_for_task(task, task_id), task_id,
                # Symmetric with the cancel path: the child drive holding the
                # blobs does not outlive the task, the canonical drive does.
                preserve_root=pathlib.Path(_q.DRIVE_ROOT),
            )
        except Exception:
            log.debug("Reaper: failed to salvage last LLM response for %s", task_id, exc_info=True)

        # The shared retry reset revokes attempt controls while preserving exact
        # owner text for the fresh physical attempt, and archives owner_hurry.
        try:
            from ouroboros.owner_hurry import retry_reset

            retry_reset(
                _q._task_drive_for_task(task, task_id), _q.DRIVE_ROOT, task_id,
                reason=f"reaper_{terminal_reason}",
            )
        except Exception:
            log.debug("Reaper: retry reset failed for killed task %s", task_id, exc_info=True)

        if not will_retry:
            try:
                write_task_result(
                    _q.DRIVE_ROOT, task_id,
                    STATUS_FAILED,
                    reason_code=terminal_reason,
                    outcome_axes=terminal_outcome_axes(
                        lifecycle=STATUS_FAILED,
                        execution=EXECUTION_INFRA_FAILED,
                        reason_code=terminal_reason,
                        review_trigger="supervisor_terminal",
                    ),
                    # GR5-2: the reap outcome discloses the delegated runs the
                    # custody reconcile above could not settle.
                    **({"delegated_runs_unreconciled": unreconciled} if unreconciled else {}),
                    **recon_fields,
                    result=(
                        f"Task killed by {terminal_reason} after "
                        f"{int(runtime_sec)}s.{salvage_note}"
                    ),
                )
            except Exception:
                log.error(
                    "Reaper: failed to write terminal result for %s",
                    task_id,
                    exc_info=True,
                )

        # 4. Enqueue the retry ONLY now (original is dead); a guarded enqueue failure cannot abort the reaper before respawn.
        requeued = False
        new_attempt = attempt
        retry_suppression: Dict[str, str] = {}
        cancel_handoff_outcomes: Dict[str, str] = {}
        if will_retry:
            try:
                requeued, new_attempt, terminal_reason, retry_suppression = _enqueue_retry(
                    _q,
                    task,
                    task_id=task_id,
                    retry_task_id=retry_task_id,
                    attempt=attempt,
                    terminal_reason=terminal_reason,
                    recon_fields=recon_fields,
                    runtime_sec=runtime_sec,
                    unreconciled_runs=unreconciled,
                    salvage_note=salvage_note,
                )
                will_retry = requeued
            except Exception:
                log.warning("Reaper: failed to enqueue retry for %s", task_id, exc_info=True)
        if retry_suppression.get("kind") == "cancel_intent":
            cancel_target = str(retry_suppression.get("target") or "")
            cancel_handoff_outcomes = _settle_retry_cancel_handoff(
                _q,
                task_id,
                retry_task_id,
                cancel_target,
            )
            _emit_cancel_suppressed_retry_task_done(
                _q,
                workers_mod,
                task,
                task_id,
                task_type,
                cancel_target,
                cancel_handoff_outcomes,
                terminal_metadata,
            )
        elif retry_suppression.get("kind") == "terminal_result":
            try:
                terminal_target = str(
                    retry_suppression.get("target") or task_id
                )
                terminal_row = load_task_result(
                    _q.DRIVE_ROOT, terminal_target,
                ) or {}
                _finish_self_finalized_task(
                    _q,
                    workers_mod,
                    task,
                    task_id,
                    task_type,
                    str(
                        terminal_row.get("status")
                        or retry_suppression.get("status")
                        or ""
                    ),
                    terminal_row,
                    terminal_metadata,
                    unreconciled,
                )
            except Exception:
                log.warning(
                    "Reaper: final retry-admission terminal recovery failed for %s",
                    task_id,
                    exc_info=True,
                )

        try:
            append_jsonl(
                _q.DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {
                    "ts": utc_now_iso(), "type": "task_terminal_timeout",
                    "task_id": task_id, "task_type": task_type, "reason": terminal_reason,
                    "worker_id": worker_id, "runtime_sec": round(runtime_sec, 2),
                    "heartbeat_lag_sec": round(hb_lag_sec, 2), "heartbeat_stale": hb_stale,
                    "attempt": attempt, "requeued": requeued, "new_attempt": new_attempt,
                    "max_retries": _q.QUEUE_MAX_RETRIES, "reaped_off_loop": True,
                    **({"retry_suppression": retry_suppression} if retry_suppression else {}),
                    **(
                        {"cancel_handoff_outcomes": cancel_handoff_outcomes}
                        if cancel_handoff_outcomes else {}
                    ),
                },
            )
        except Exception:
            log.debug("Reaper: failed to log task_terminal_timeout for %s", task_id, exc_info=True)

        # Guarded: a notification failure must not abort the reaper before respawn
        # (the slot would stay reaping=True). C4: the notice goes to the TASK'S chat.
        incident_chat_id = _incident_chat_id(task, owner_chat_id)
        if incident_chat_id is not None:
            try:
                if requeued:
                    send_with_budget(
                        incident_chat_id,
                        f"🛑 {terminal_reason}: task {task_id} killed after {int(runtime_sec)}s.\n"
                        f"Worker {worker_id} restarted. Task queued for retry attempt={new_attempt}.",
                        is_progress=True, task_id=task_id,
                        progress_meta={"task_incident": "task_reaper_retry", "toast_once": incident_toast_once},
                    )
                elif retry_suppression.get("kind") == "cancel_intent":
                    send_with_budget(
                        incident_chat_id,
                        f"🛑 {terminal_reason}: task {task_id} killed after {int(runtime_sec)}s.\n"
                        "Its retry was suppressed because cancellation won the "
                        "admission race; cancellation custody is settling the task.",
                        is_progress=True,
                        task_id=task_id,
                        progress_meta={
                            "task_incident": "task_reaper_cancel_suppressed_retry",
                            "toast_once": incident_toast_once,
                        },
                    )
                elif not retry_suppression:
                    stop_detail = _stop_detail(ceiling_reached, deadline_reached, orchestrator)
                    send_with_budget(
                        incident_chat_id,
                        f"🛑 {terminal_reason}: task {task_id} killed after {int(runtime_sec)}s.\n"
                        f"Worker {worker_id} restarted. {stop_detail}",
                        is_progress=True, task_id=task_id,
                        progress_meta={"task_incident": "task_reaper_stopped", "toast_once": incident_toast_once},
                    )
            except Exception:
                log.debug("Reaper: failed to send owner notification for %s", task_id, exc_info=True)

        if not requeued and not retry_suppression:
            # AR2-5a ordering: salvage first (it registers the answer as OWED), then task_done.
            _deliver_reap_salvage(_q, task, task_id, terminal_reason, unreconciled)
            _emit_reap_task_done(
                workers_mod, task, task_id, task_type, terminal_reason,
                recon_fields, terminal_metadata,
            )

    # 5. Respawn a fresh worker for the slot; on failure, clear reaping so the
    # crash detector can recover it instead of leaving a permanently held slot.
    _respawn_after_reap(_q, workers_mod, worker_id)
