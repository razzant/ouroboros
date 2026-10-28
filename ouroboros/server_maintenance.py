"""Generation startup and periodic upkeep: custody, delegated runs, cancel latches,
owed deliveries, orphaned results, held Continues and post-task synthesis. History-sized work runs
off-loop; delegated-snapshot GC fails closed on an unreadable log."""

from __future__ import annotations

import json
import logging
import os
import pathlib
import stat
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

from ouroboros.server_process import DATA_DIR, _restart_requested, _supervisor_stop, log
from ouroboros.config import BODY_MAINTENANCE_INTERVAL_SEC
from ouroboros.utils import utc_now_iso


def _installed_skill_names():
    """Disk-derived skill owners for companion reaping; unknown means KEEP.
    Reload timing and failed discovery must never mass-reap live companions."""
    try:
        from ouroboros.config import get_skills_repo_path
        from ouroboros.skill_loader import discover_skills

        names = {s.name for s in discover_skills(DATA_DIR, repo_path=get_skills_repo_path())}
        # An EMPTY result is None ("unknown"), NOT "everything uninstalled":
        # discover_skills returns [] when the skills dir is momentarily unavailable,
        # and an enforced reap over that would mass-kill live companions.
        return names or None
    except Exception:
        log.debug("Could not compute installed skill names for custody reaper", exc_info=True)
        return None


_LAST_CANCEL_INTENT_SWEEP = [0.0]
_CANCEL_INTENT_SWEEP_LOCK = threading.Lock()
_DELIVERY_SWEEP_LOCK = threading.Lock()
_DELIVERY_WAKE = threading.Event()
_QUESTION_SWEEP_LOCK = threading.Lock()
_PAUSE_NOTICE_SWEEP_LOCK = threading.Lock()
_USAGE_SWEEP_LOCK = threading.Lock()
_SYNTHESIS_SWEEP_LOCK = threading.Lock()
_OWNER_PAUSE_SWEEP_LOCK = threading.Lock()
_CONTINUATION_SWEEP_LOCK = threading.Lock()
_COLD_CUSTODY_SWEEP_LOCK = threading.Lock()
_LAST_COLD_CUSTODY_SWEEP = [0.0]
_CUSTODY_SWEEP_LOCK = threading.Lock()
_RECONCILE_SWEEP_LOCK = threading.Lock()
_SCHEDULE_SWEEP_LOCK = threading.Lock()
CUSTODY_SWEEP_SEC = 600  # the custody block's cadence (stamped at dispatch)
RECONCILE_SWEEP_SEC = 300  # the reconcile block's cadence (stamped when its pass ends, issue #1230)


def bind_body_lifespan(drive_root: Any, stop_event: Any, *, collection_ready=None):
    """Return this lifespan's idempotent post-publication starter.

    A previous lifespan keeps its physical jobs until they return. Only that
    actual completion admits a successor; neither a timeout nor a loop revival
    replaces its ownership. Waiting happens off the startup/control path.
    """
    from ouroboros.delegate_collection import start_body_runtime

    lock, handover = threading.Lock(), []

    def start():
        with lock:
            if stop_event.is_set() or handover:
                return
            runtime = start_body_runtime(drive_root, stop_event=stop_event, collection_ready=collection_ready)
            if runtime.active:
                return

            def after_close():
                runtime.close(wait=True)
                with lock:
                    handover.clear()
                    if not stop_event.is_set():
                        start_body_runtime(drive_root, stop_event=stop_event, collection_ready=collection_ready)

            thread = threading.Thread(target=after_close, name="body-lifespan-handover", daemon=True)
            handover.append(thread)
            try:
                thread.start()
            except BaseException:
                handover.clear()
                raise

    return start


def _reconcile_cold_custody(drive_root: Any, stop_event: Any = None, *, saved=()) -> set[str]:
    """Current identities only; requests remain discoverable without an event."""
    from ouroboros.sleep_custody import carrier_receipt, reconcile
    from supervisor import queue
    from supervisor.active_activity import get_direct_activity_registry

    # Direct turns do not occupy RUNNING. Their registry identity is sufficient
    # to read the canonical exact pause, including while the actor is pausing.
    direct = []
    for activity in get_direct_activity_registry().actors():
        ctx = getattr(getattr(activity.actor, "tools", None), "_ctx", None)
        root = getattr(ctx, "budget_drive_root", None) or getattr(ctx, "drive_root", None) or drive_root
        direct.append({"id": activity.activity_id, "project_id": activity.project_id,
                       "budget_drive_root": str(root)})
    return reconcile(queue, direct=[*direct, *(task for task in saved if carrier_receipt(task))],
                     stop_requested=lambda: _stop_requested(stop_event))


def _run_cold_custody(stop_event: Any, latch: Any) -> None:
    try:
        if not _stop_requested(stop_event):
            _reconcile_cold_custody(DATA_DIR, stop_event)
    except Exception:
        log.warning("Cold custody pass unavailable; existing ownership retained", exc_info=True)
    finally:
        latch.release()


@contextmanager
def orphan_reconcile_write_guard(task_id: str, *, stop_event: Any = None):
    """Fence the orphan reconciler's writes against assignment (queue -> row lock order) in an open generation:
    the settlement probe must prove absence; a process with no supervisor (no-provider boot) dispatches nothing."""
    from supervisor import queue

    if queue.INITIALIZED:
        queue.task_settlement_liveness(task_id)  # Prepare durable reads before the mutation interlock.
    with queue._queue_lock:
        yield not _stop_requested(stop_event) and (not queue.INITIALIZED or queue.task_settlement_liveness(task_id) is False)


def _stop_requested(stop_event: Any = None) -> bool:
    """Closed generation, stop or restart: no mutation or daemon start afterward.
    The per-generation token also keeps an old off-loop pass from writing."""
    return bool(_restart_requested.is_set() or _supervisor_stop.is_set()
                or (stop_event is not None and stop_event.is_set()))


def _live_task_ids() -> set:
    """Fresh in-memory owners for custody and cursor refresh, never result rows.
    Passed as a callable so consumers recheck after collecting candidates."""
    return _startup_live_task_ids(DATA_DIR)


def _run_terminal_duty(drive_root: pathlib.Path, step: str, latch: Any,
                       stop_event: Any = None) -> None:
    """Run one existing duty; unrelated recovery never waits for its completion."""
    try:
        if _stop_requested(stop_event):
            return
        if step == "cancel_intent_sweep":
            from supervisor.task_lifecycle import sweep_cancel_intents
            outcomes = sweep_cancel_intents()
            if outcomes:
                log.info("Cancel-intent watchdog settled: %s", outcomes)
        elif step == "terminal_delivery_replay":
            from ouroboros import custody_owed
            from ouroboros.dialogue_retry import drain, _requested
            from supervisor.terminal_delivery import replay_pending_deliveries
            _requested.clear()
            _DELIVERY_WAKE.clear()
            drain()
            custody_owed.drain()  # local custody facts this server process knows but could not write yet
            replay_pending_deliveries(drive_root)
        elif step == "question_delivery_replay":
            import asyncio
            from ouroboros.gateway.task_decision import recover_question_deliveries
            asyncio.run(recover_question_deliveries(drive_root))
        elif step == "pause_notice_replay":
            from ouroboros.pause_notices import reconcile_pause_notices
            reconcile_pause_notices(drive_root, stop_requested=lambda: _stop_requested(stop_event))
        elif step == "abandoned_usage_reconciliation":
            _reconcile_abandoned_usage(drive_root)
        elif step == "owner_pause_settlement":
            from supervisor.owner_pause_control import settle_requested_owner_pauses
            settle_requested_owner_pauses(stop_requested=lambda: _stop_requested(stop_event))
        elif step == "held_continuation_recheck":
            from supervisor.continuation_admission import recheck_held_continuations
            released = recheck_held_continuations(stop_requested=lambda: _stop_requested(stop_event))
            if released:
                log.info("Held Continue re-check released: %s", released)
        elif step == "post_task_synthesis_recovery":
            from ouroboros.agent_task_pipeline import recover_pending_root_post_task_synthesis
            from ouroboros.config import REPO_DIR
            from supervisor import queue, workers
            recover_pending_root_post_task_synthesis(
                drive_root, REPO_DIR, startup=False, owner_liveness=queue.task_settlement_liveness,
                claim_guard=queue.task_settlement_interlock, stop_requested=lambda: _stop_requested(stop_event),
                event_queue=workers.get_event_q(create=False))
        else:
            raise ValueError(f"unknown terminal duty: {step}")
        _step_recovered(step)
    except Exception:
        _step_failed(step)
    finally:
        latch.release()


def _run_schedule_duty(stop_event: Any, latch: Any) -> None:
    """One automatic schedule pass (``queue_schedules.check_scheduled_tasks``) off the loop:
    discovery, preparation, billing and allowance reads wait here, never under Q, and its
    Q-held steps try each acquisition once. The generation token stops it mutating."""
    try:
        from supervisor.queue_schedules import automatic_acquisitions, check_scheduled_tasks

        with automatic_acquisitions():
            check_scheduled_tasks(stop=lambda: _stop_requested(stop_event))
        _step_recovered("schedule_admission")
    except Exception as exc:
        from supervisor.schedule_occurrence import contention

        if not contention(exc):  # a held table or store: the next tick's pass retries
            _step_failed("schedule_admission")
    finally:
        latch.release()


def _reconcile_abandoned_usage(drive_root: pathlib.Path) -> None:
    """Compatibility seam for the existing terminal-maintenance duty."""
    from ouroboros.terminal_cost_reconciliation import reconcile_abandoned_usage

    reconcile_abandoned_usage(drive_root)


# Memory-only failure streaks per periodic step: WARNING with traceback on failures
# 1, 2, 4, 8, ...; DEBUG in between; one recovery INFO naming the streak.
_STEP_FAILURES: Dict[str, int] = {}


def _step_failed(step: str, message: str = "%s failed") -> None:
    streak = _STEP_FAILURES.get(step, 0) + 1
    _STEP_FAILURES[step] = streak
    level = logging.WARNING if streak & (streak - 1) == 0 else logging.DEBUG
    log.log(level, message + " (failure %d in a row)", step, streak, exc_info=True)


def _step_recovered(step: str) -> None:
    streak = _STEP_FAILURES.pop(step, 0)
    if streak:
        log.info("%s recovered after %d failure(s)", step, streak)


# Memory only: each latched off-loop pass's thread, start and stall-journal flag.
# After its own cadence from START, a held latch records host_duty_stall with the
# duty and current stack line; completion records host_duty_stall_end. Neither the
# loop deadline nor the previous completion marker measures this pass's duration.
_DUTIES: Dict[int, Dict[str, Any]] = {}


def _duty_busy(latch: Any, cadence_sec: float) -> None:
    duty = _DUTIES.get(id(latch))
    if duty is None or duty["alerted"] or time.time() - duty["since"] < cadence_sec:
        return
    duty["alerted"] = True
    from ouroboros.server_liveness import _loop_thread_stack
    from supervisor.state import append_jsonl

    flags: Dict[str, Any] = {}
    stack = _loop_thread_stack(duty["ident"], facts=flags)
    running = round(time.time() - duty["since"], 1)
    log.warning("Off-loop duty %s has run %.0fs, past its %.0fs cadence; its thread is at: %s",
                duty["name"], running, cadence_sec, stack[-1] if stack else "(stack unavailable)")
    try:
        append_jsonl(DATA_DIR / "logs" / "supervisor.jsonl", {
            "ts": utc_now_iso(), "type": "host_duty_stall", "duty": duty["name"], "running_sec": running,
            "cadence_sec": cadence_sec, **flags, **({"stack": stack} if stack else {})})
    except Exception:
        log.debug("host_duty_stall row failed", exc_info=True)
    try:
        from ouroboros.update_generation_facts import observe_duty_overrun

        observe_duty_overrun(duty["name"])  # once per alerted pass: the after-update facts count it
    except Exception:
        log.debug("generation duty observation failed", exc_info=True)


def _duty_ended(latch: Any, duty: Dict[str, Any]) -> None:
    """Close THIS pass's duty. The pass released its latch before this runs, so the next pass
    may already have registered its own entry: that one is put back, never dropped (it holds
    the latch, so nothing registers again meanwhile)."""
    current = _DUTIES.pop(id(latch), None)
    if current is not None and current is not duty:
        _DUTIES.setdefault(id(latch), current)
    if not duty["alerted"]:
        return
    from supervisor.state import append_jsonl

    try:
        append_jsonl(DATA_DIR / "logs" / "supervisor.jsonl", {
            "ts": utc_now_iso(), "type": "host_duty_stall_end", "duty": duty["name"],
            "running_sec": round(time.time() - duty["since"], 1)})
    except Exception:
        log.debug("host_duty_stall_end row failed", exc_info=True)


def _start_maintenance_thread(latch: Any, name: str, target: Any, args: tuple) -> bool:
    """Start one off-loop pass under a latch the CALLER already took without blocking
    (busy => that tick skipped, never queued); a start that fails releases it. The pass is
    watched as a duty (``_duty_busy`` / ``_duty_ended``)."""
    def run() -> None:
        duty["ident"] = thread.ident  # the pass names its own thread: no race with its end
        try:
            target(*args)
        finally:
            _duty_ended(latch, duty)

    duty: Dict[str, Any] = {"name": name, "ident": None, "since": time.time(), "alerted": False}
    _DUTIES[id(latch)] = duty
    try:
        thread = threading.Thread(target=run, name=name, daemon=True)
        thread.start()
        return True
    except Exception:
        _DUTIES.pop(id(latch), None)
        latch.release()
        log.warning("%s could not start", name, exc_info=True)
        return False


def _periodic_supervisor_maintenance(
    last_custody_reap: list, last_review_reconcile: list, *, stop_event: Any = None,
) -> None:
    """Throttled upkeep on the supervisor tick, nothing history-sized inline (INV-24).
    Each tick starts the schedule duty when its previous pass ended;
    the 20 s cadence independently starts cancel, delivery, question, notice, usage and synthesis duties,
    plus the held-Continue re-check while a pending Continue waits under its writer hold; every 600 s
    the custody block (reaping, delegated runs, the engine's route-health facts, then the working copies'
    retirers); every 300 s the reconcile
    block (zombie heal, the age-only retirers, drive settlement, then the maintenance dispatcher). The first
    two stamp here before handing off; the reconcile pass stamps when it ENDS (issue #1230). A due cadence
    whose latch is still held journals the duty's stall once, with its thread's stack (``_duty_busy``).
    ``stop_event`` is the loop's per-generation token: both long passes stop mutating
    once it closes. History-sized work is the maintenance child's (``ouroboros/maintenance.py``):
    a process's first tick, or a job scheduled since (``owed``), reserves its slot here."""
    if _stop_requested(stop_event):
        return
    now = time.time()
    from supervisor import queue
    from ouroboros import custody_owed
    from ouroboros.dialogue_retry import bind_wake

    bind_wake(_DELIVERY_WAKE.set)
    custody_owed.bind_wake(_DELIVERY_WAKE.set)
    if (now - _LAST_CANCEL_INTENT_SWEEP[0] <= BODY_MAINTENANCE_INTERVAL_SEC
            and _DELIVERY_WAKE.is_set() and _DELIVERY_SWEEP_LOCK.acquire(blocking=False)):
        _DELIVERY_WAKE.clear()
        _start_maintenance_thread(_DELIVERY_SWEEP_LOCK, "terminal_delivery_replay", _run_terminal_duty,
                                  (pathlib.Path(DATA_DIR), "terminal_delivery_replay", _DELIVERY_SWEEP_LOCK, stop_event))
    if now - _LAST_CANCEL_INTENT_SWEEP[0] > BODY_MAINTENANCE_INTERVAL_SEC:
        # One existing cadence, independent single-flight operations: its stamp
        # advances on dispatch, never on a slow duty's completion.
        _LAST_CANCEL_INTENT_SWEEP[0] = now
        from supervisor.continuation_admission import held_continuation_predecessors
        # A writer hold installs no root fence: the held row itself is the cheap
        # scheduling hint, and the off-loop pass re-reads the queue under its lock.
        held = (host_duty("held_continuation_recheck"),) if held_continuation_predecessors(queue.PENDING) else ()
        for duty in (*TWENTY_SECOND_DUTIES, *held):  # the one 20 s group of HOST_DUTIES, dispatched together
            if duty.latch.acquire(blocking=False):
                _start_maintenance_thread(duty.latch, duty.name, duty.run,
                                          (pathlib.Path(DATA_DIR), duty.name, duty.latch, stop_event))
            else:
                _duty_busy(duty.latch, BODY_MAINTENANCE_INTERVAL_SEC)
    # The host accepts a saved cold request while its actor is still RUNNING;
    # waiting for the park event would make release and acceptance wait on each other.
    from supervisor.active_activity import get_direct_activity_registry
    has_actors = bool(queue.RUNNING or queue.PENDING or get_direct_activity_registry().actors())
    from ouroboros.delegate_supervision import _TICK_SEC
    if (has_actors and now - _LAST_COLD_CUSTODY_SWEEP[0] >= _TICK_SEC
            and _COLD_CUSTODY_SWEEP_LOCK.acquire(blocking=False)):
        _LAST_COLD_CUSTODY_SWEEP[0] = now
        _start_maintenance_thread(_COLD_CUSTODY_SWEEP_LOCK, "cold-custody", _run_cold_custody,
                                  (stop_event, _COLD_CUSTODY_SWEEP_LOCK))
    # Presence is only a cheap scheduling hint. The off-loop owner re-reads
    # the exact fences; a network stop must never run on the assignment tick.
    if queue.BUDGET_ROOT_FENCES and _OWNER_PAUSE_SWEEP_LOCK.acquire(blocking=False):
        latch = _OWNER_PAUSE_SWEEP_LOCK
        _start_maintenance_thread(latch, "owner_pause_settlement", _run_terminal_duty,
                                  (pathlib.Path(DATA_DIR), "owner_pause_settlement", latch, stop_event))
    latch = _CUSTODY_SWEEP_LOCK  # a pass releases THIS object, never a later generation's
    if now - last_custody_reap[0] > CUSTODY_SWEEP_SEC:
        if latch.acquire(blocking=False):
            last_custody_reap[0] = now
            _start_maintenance_thread(latch, "custody-maintenance", _run_periodic_custody_sweep,
                                      (stop_event, latch))
        else:
            _duty_busy(latch, CUSTODY_SWEEP_SEC)
    latch = _RECONCILE_SWEEP_LOCK
    if now - last_review_reconcile[0] > RECONCILE_SWEEP_SEC:
        if not latch.acquire(blocking=False):
            _duty_busy(latch, RECONCILE_SWEEP_SEC)
        elif not _start_maintenance_thread(latch, "reconcile-maintenance", _run_periodic_reconcile_sweep,
                                           (last_review_reconcile, stop_event, latch)):
            last_review_reconcile[0] = time.time()  # a start that failed still waits out a cadence
    from ouroboros.maintenance import child as maintenance_child
    if maintenance_child.owed():  # a process's first look, or a job scheduled since: reserve the slot only
        _tick_maintenance_child()
    # Every tick, as the inline pass ran before; a pass still running (a slow preparation) skips this one.
    if _SCHEDULE_SWEEP_LOCK.acquire(blocking=False):
        _start_maintenance_thread(_SCHEDULE_SWEEP_LOCK, "schedule-admission", _run_schedule_duty,
                                  (stop_event, _SCHEDULE_SWEEP_LOCK))
    else:
        _duty_busy(_SCHEDULE_SWEEP_LOCK, BODY_MAINTENANCE_INTERVAL_SEC)


def _run_periodic_custody_sweep(stop_event: Any = None, latch: Any = None) -> None:
    """Keep slow hashing, reaping and reconciliation off the worker-ack thread (INV-B). The
    caller's nonblocking latch skips busy passes; ``finally`` releases that latch. Assignment
    stays concurrent: read candidates before live sets and recheck the generation before each
    mutation. Spawn only after releasing a failed-start latch, at most once per sweep."""
    try:
        try:
            if _stop_requested(stop_event):
                return
            from ouroboros.terminal_projection import reconcile_terminal_projections
            reconcile_terminal_projections(DATA_DIR)
        except Exception:
            log.warning("Terminal projection reconciliation deferred", exc_info=True)
        try:
            # Issue #844: release the owned-daemon start latch in ITS OWN try, ahead of the reap, so a
            # raising reap can never pin it; retry once, only when THIS sweep released a latch, on a
            # short-lived thread as warm_owned_daemon() does (DEVELOPMENT.md "Process Custody Rule").
            from ouroboros.claudexor_daemon import get_owned_daemon

            if _stop_requested(stop_event):
                return
            if get_owned_daemon().clear_start_failure_latch(cleared_by="supervisor_sweep"):
                threading.Thread(target=_retry_latched_daemon_start,
                                 name="owned-daemon-latch-retry", daemon=True).start()
        except Exception:
            log.debug("Owned daemon latch release failed", exc_info=True)
        # One guard for the steps (a failure still ends the pass), but the row names WHICH died:
        # unnamed at DEBUG, weeks of silent non-reaping looked healthy.
        step = "reap_orphaned_processes"
        try:
            from ouroboros.claudexor_daemon import CUSTODY_PURPOSE
            from ouroboros.process_custody import reap_orphaned_processes
            from ouroboros.process_scope import settle_task_scopes

            if _stop_requested(stop_event):
                return
            reap_orphaned_processes(
                DATA_DIR, running_task_ids=_live_task_ids,
                live_owner_skills=_installed_skill_names(),
                retained_purposes={CUSTODY_PURPOSE},
            )
            step = "settle_task_scopes"  # ended tasks' leftovers, on positive terminal evidence only (abrupt death)
            if _stop_requested(stop_event):
                return
            settle_task_scopes(DATA_DIR, _live_task_ids)
            # A delegated Claudexor run is an orphan under exactly the same predicate: its owning task is
            # no longer running. It has no pid, so the process reaper cannot see it, yet it still spends.
            step = "reconcile_delegated_runs"
            if _stop_requested(stop_event):
                return
            _reconcile_delegated_runs(_live_task_ids, stop_event=stop_event)
            step = "cursor_refresh_settled_terminals"
            if _stop_requested(stop_event):
                return
            _cursor_refresh_settled_terminals(_live_task_ids)
            # The engine's per-route failure shares, after the runs above settled: attach-only, one row
            # when they changed, so the Runtime vitals and deep self-review read them without a UI poll.
            step = "refresh_route_health"
            if _stop_requested(stop_event):
                return
            from ouroboros.claudexor_exit_facts import refresh_route_health

            refresh_route_health()
            for name in ("reap_orphaned_processes", "settle_task_scopes", "reconcile_delegated_runs",
                         "cursor_refresh_settled_terminals", "refresh_route_health"):
                _step_recovered(name)
        except Exception:
            _step_failed(step, "Periodic custody step %s failed")
        # The working copies' and old engine programs' retirers, after the delegated runs above settled: a
        # server that stays up for weeks retires them too (each keeps what a task, custody, review or daemon holds).
        for prune in (_prune_worktrees, _prune_delegated_snapshots, _prune_engine_runtimes):
            if not _stop_requested(stop_event):
                prune()
        if not _stop_requested(stop_event):
            _prune_review_checkouts(stop_event=stop_event)  # ended tasks' copies, a crashed process's orphans
    finally:
        (latch or _CUSTODY_SWEEP_LOCK).release()


def _retry_latched_daemon_start() -> None:
    """The one retry of a latched owned-daemon start (#844), by the sweep itself, on its own
    short-lived daemon thread and only after this sweep released the latch: one
    ``ensure_owned_gateway`` with ZERO admission and ZERO startup wait, so no loop thread ever
    holds a startup wait — the spawn happens, custody keeps the child, and ``daemon_starting``
    is the EXPECTED answer (the next ordinary caller joins or settles it). Any other typed
    refusal (a child that died at once has re-latched inside the manager) is a warning;
    nothing is raised into the loop or retried again; an opened gateway is closed at once."""
    from ouroboros.claudexor_daemon import ensure_owned_gateway
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    try:
        ensure_owned_gateway(admission_wait_sec=0, startup_wait_sec=0).close()
    except ClaudexorUnavailable as exc:
        if exc.code == "daemon_starting":
            log.info("Owned daemon retry after latch release is starting under custody: %s", exc)
        else:
            log.warning("Owned daemon retry after latch release refused (%s): %s", exc.code, exc)
    except Exception:
        log.warning("Owned daemon retry after latch release failed unexpectedly", exc_info=True)


def _reconcile_delegated_runs(running_task_ids: Any, *, stop_event: Any = None) -> None:
    """Settle or cancel delegated runs whose owning task is gone (startup + tick)."""
    try:
        from ouroboros.claudexor_daemon import ensure_owned_gateway, read_owned_gateway
        from ouroboros.delegate_custody import reconcile_orphaned_runs
        from ouroboros.delegate_recovery import recoverable_handles, continuation_handles
        from ouroboros.owner_wait import restore_owner_wait_allowed
        from supervisor.queue import PENDING, _queue_lock

        with _queue_lock:
            pending_waits = [dict(task) for task in PENDING if task.get("_owner_wait_resume")]
        continued = {str(task["id"]) for task in pending_waits
                     if restore_owner_wait_allowed(DATA_DIR, task)}

        # Zero admission wait: a daemon in its recovery-only window waits for the next sweep. Once a
        # stop, restart or panic is in flight the factory is ATTACH-ONLY and nothing is replayed
        # (``replay_pending`` is re-asked before each re-POST); the one ensure left here is the latch
        # retry (ARCHITECTURE §9). This admitted sweep is the ONLY surface replaying an ended task's start.
        outcomes = reconcile_orphaned_runs(
            DATA_DIR, running_task_ids=running_task_ids,
            gateway_factory=lambda: (read_owned_gateway() if _stop_requested(stop_event)
                                     else ensure_owned_gateway(admission_wait_sec=0)),
            recoverable_handles=recoverable_handles(DATA_DIR) | continuation_handles(DATA_DIR, continued),
            replay_pending=lambda: not _stop_requested(stop_event),
        )
        if outcomes:
            log.info("Delegated-run reconciliation handled %d orphan(s): %s", len(outcomes), outcomes)
            # A run settled here may belong to a task whose stored terminal result
            # carries an unreconciled disclosure that would lie forever (nanny-leaf
            # S1). Audit-only refresh; never cancels.
            from ouroboros.delegate_terminal import refresh_terminal_reconciliation

            for tid in {str(o.get("task_id") or "") for o in outcomes
                        if o.get("task_id") and (o.get("settled") or str(
                            o.get("action") or "") in (
                                "absent", "cancelled", "invocation_retired"))}:
                try:
                    refresh_terminal_reconciliation(DATA_DIR, tid)
                except Exception:
                    log.debug("Sweep terminal-result refresh failed for %s", tid, exc_info=True)
    except Exception:
        log.debug("Delegated-run reconciliation failed", exc_info=True)


def _startup_retired_settings_notice(settings: dict) -> None:
    """Tell the OWNER, in their chat, that retired keys in ``settings.json`` are NOT honored.

    ``config.normalize_settings_raw`` reports the loss on the module logger only, which an
    owner who never opens the Logs panel does not see. The dropped sets come from that same
    read seam (``config.retired_key_sets_seen``), the sentence is the one the log line uses
    (``settings_defaults.retired_setting_keys_notice``), and the dedupe is durable:
    ``state.json:retired_settings_notified`` keyed by the exact retired-key set, so a
    restart or a supervisor revival never repeats it. Nothing is sent — and nothing
    marked — while no owner chat is bound: the notice waits for the first boot that has
    somewhere to deliver it. Which reviewers run is the review-pool migration's report
    (``_startup_review_pool_notice``), not this one's.
    """
    try:
        from ouroboros.config import retired_key_sets_seen
        from ouroboros.settings_defaults import retired_setting_keys_notice
        from supervisor.message_bus import send_with_budget
        from supervisor.state import load_state, update_state

        state = load_state()
        owner_chat = int(state.get("owner_chat_id") or 0)
        if not owner_chat:
            return
        notified = state.get("retired_settings_notified")
        notified = notified if isinstance(notified, dict) else {}
        for dropped in retired_key_sets_seen():
            marker = ",".join(dropped)
            if marker in notified:
                continue
            send_with_budget(
                owner_chat,
                "⚙️ Settings: " + retired_setting_keys_notice(dropped),
                role="system", system_type="retired_settings_notice",
            )
            update_state(lambda st, key=marker: _mark_retired_settings_notified(st, key))
    except Exception:
        log.debug("retired settings owner notice failed", exc_info=True)


def _mark_retired_settings_notified(st: dict, marker: str) -> None:
    """Stamp ``marker`` in the durable owner-notice ledger ``state.json:retired_settings_notified``."""
    seen = st.get("retired_settings_notified")
    seen = dict(seen) if isinstance(seen, dict) else {}
    seen[marker] = utc_now_iso()
    st["retired_settings_notified"] = seen


def environment_retired_review_keys(environ: Dict[str, str] | None = None) -> tuple[str, ...]:
    """The retired review keys the PROCESS ENVIRONMENT carries with a value: the review lanes
    (``OUROBOROS_REVIEWER_SLOTS``), the per-surface review efforts / deep-review model and the
    older reviewer comma-lists. No release reads them from the environment any more — the
    environment merge (``config.load_settings``) walks ``SETTINGS_DEFAULTS``, which retired
    them, and the pool migration reads the DOCUMENT — so an operator who still exports them
    (a Docker/Linux unit, a Colab cell) configures nothing (D1-V03): the install runs its own
    pool (the catalog in the document, else the factory rows)."""
    from ouroboros.settings_defaults import RETIRED_COMMA_LIST_SETTING_KEYS, REVIEW_POOL_MIGRATED_SETTING_KEYS

    env = os.environ if environ is None else environ
    return tuple(key for key in REVIEW_POOL_MIGRATED_SETTING_KEYS + RETIRED_COMMA_LIST_SETTING_KEYS
                 if str(env.get(key) or "").strip())


def environment_review_notice(keys: tuple[str, ...]) -> str:
    """The ONE sentence (log line and owner chat alike) for review keys found in the environment."""
    plural = len(keys) != 1
    return (
        f"⚙️ Settings: the process environment sets {', '.join(keys)}, which {'are' if plural else 'is'} no longer "
        "read: the review lanes and the reviewer lists became the review pool — the rows of the subagent "
        "catalog marked “Reviewer” (OUROBOROS_SUBAGENTS, Settings → Agents). "
        f"{'Those values were' if plural else 'That value was'} not applied; the install's own pool runs. "
        "To configure the pool from the environment, set OUROBOROS_SUBAGENTS to a catalog with marked rows."
    )


def _startup_environment_review_notice() -> None:
    """Say ONCE, loudly, that review keys set in the environment are not read (D1-V03 / VD3-03):
    a WARNING on the server log at every boot, and the same sentence in the owner chat once per
    exact key set (durable: ``state.json:retired_settings_notified`` under an ``environment:``
    marker, the retired-settings notice's own ledger). Nothing is read from the environment
    into the pool here or anywhere: the fact is loud, the behaviour unchanged."""
    keys = environment_retired_review_keys()
    if not keys:
        return
    text = environment_review_notice(keys)
    log.warning(text)
    try:
        from supervisor.message_bus import send_with_budget
        from supervisor.state import load_state, update_state

        state = load_state()
        owner_chat = int(state.get("owner_chat_id") or 0)
        if not owner_chat:
            return
        marker = "environment:" + ",".join(keys)
        notified = state.get("retired_settings_notified")
        if marker in (notified if isinstance(notified, dict) else {}):
            return
        send_with_budget(owner_chat, text, role="system", system_type="retired_settings_notice")
        update_state(lambda st: _mark_retired_settings_notified(st, marker))
    except Exception:
        log.debug("environment review keys owner notice failed", exc_info=True)


REVIEW_POOL_MIGRATION_STATE_KEY = "review_pool_migrations"  # ``review_pool_receipts.STATE_KEY``
REVIEW_POOL_NOTICE_TYPE = "review_pool_migration_notice"


def review_pool_migration_records(state: dict | None = None) -> dict:
    """The durable per-document migration records (``state.json:review_pool_migrations``):
    ``input_sha256 -> {ts, snapshot, trigger, outcome, error, reported}`` — the ledger
    ``review_pool_receipts`` keeps; this is the name the ``## Review`` block reads."""
    from ouroboros.review_pool_receipts import migration_records

    return migration_records(state)


def review_pool_migration_payload(settings: dict, *, document: dict | None = None) -> dict | None:
    """The review-pool payload's ``migration`` fact for this data root (``review_pool_receipts.migration_payload`` over
    the ledger plus the snapshot files no record names yet). ``settings`` is what RUNS; ``document`` (default ``settings``)
    is the settings document the receipt is judged against — the GET handler passes the document on disk (VD3-06)."""
    from ouroboros.review_pool_receipts import migration_payload

    return migration_payload(settings if document is None else document, DATA_DIR, review_pool_migration_records(), running=settings)


def _startup_review_pool_notice(settings: dict) -> None:
    """Tell the OWNER once about every review-lane -> review-pool migration recorded for this
    data root, from the DURABLE receipts — never from this process's memory alone.

    The migration itself is pure and runs at the read seam (``review_pool_migration.apply_at_read_seam``; an old
    document in ``config.normalize_settings_raw``, a never-configured one over the effective input) in any reader;
    the process that SAVES the migrated document writes its receipts before that write
    (``review_pool_receipts.persist_write_receipts`` from the persistence prologue and the Colab
    writer): the snapshot ``state/review_migrations/<ts>-slots-to-pool.json`` and the
    ``state.json`` record. This boot step first gives receipts, by the writer's rule, only to
    the migrations deciding the document on disk now (``persist_boot_receipts``: the N-1 document the boot read before
    any save; with no file or a never-configured one, what ``settings`` runs; never the factory rows of defaults read
    before the wizard saved its own catalog), then reconciles a record for every snapshot another process left without
    one (a Colab kernel, the launcher menu, the UI before this supervisor generation), and sends ONE English
    owner-chat message (``review_pool_migration.owner_message``) per record whose ``reported`` is still unset, once an
    owner chat is bound. A migration that could not finish is reported the same way (its snapshot carries the error;
    the lane keys stay in the document for the owner's catalog save). A no-op outcome (the catalog was already a pool)
    leaves no receipt: nothing changed. Each unreported record is judged against the document AS READ
    (``review_pool_receipts.document_as_read``: the file as the seam leaves it, no environment; no file or a
    never-configured one reads as ``settings``); ``settings`` is what RUNS (the environment merged over it). A record
    whose outcome no longer decides that document (the factory rows receipted at a file-less start before the wizard
    saved its own catalog) is closed as history without a message (``review_pool_receipts.close_as_history``), never
    delivered as if the owner's catalog were the environment's. Where the environment's catalog overrides the rows the
    raw stage minted for an old document without review settings of its own (``environment_overridable_keys``), the
    message names THAT pool — loudly empty when none of its rows is marked. A never-configured document whose process
    environment still carries the retired review keys is told those keys are no longer read
    (``environment_retired_review_keys``).
    """
    try:
        from ouroboros import config, review_pool_receipts as receipts
        from ouroboros.review_pool_migration import owner_message
        from supervisor.message_bus import send_with_budget
        from supervisor.state import load_state, update_state

        receipts.persist_boot_receipts(DATA_DIR, config.SETTINGS_PATH, settings)
        state = load_state()
        owner_chat = int(state.get("owner_chat_id") or 0)
        records = receipts.reconcile_records(DATA_DIR, state, update_state)
        if not owner_chat:
            return
        document = receipts.document_as_read(config.SETTINGS_PATH, settings)
        for digest, record in sorted(records.items(), key=lambda item: str(item[1].get("ts") or "")):
            if record.get("reported"):
                continue
            snapshot = receipts.load_snapshot(DATA_DIR, record)
            if snapshot is None:
                log.warning("review pool migration snapshot missing, owner not told: %s", record.get("snapshot"))
                continue
            outcome = receipts.outcome_from_snapshot(snapshot)
            snapshot_path = str(record.get("snapshot") or "")
            if receipts.close_as_history(outcome, document, update_state, digest, record):
                continue
            in_force = receipts.environment_catalog_in_force(outcome, document, settings)
            text = (_environment_pool_message(snapshot_path, in_force) if in_force is not None else
                    owner_message(outcome, snapshot_path, environment_retired_keys=environment_retired_review_keys()))
            if not text:
                continue
            send_with_budget(owner_chat, text, role="system", system_type=REVIEW_POOL_NOTICE_TYPE)
            receipts.mark_reported(update_state, digest, record)
    except Exception:
        log.debug("review pool migration notice failed", exc_info=True)


def _environment_pool_message(snapshot_path: str, catalog_text: str) -> str:
    """The ONE owner-chat message when the catalog the environment carries, not the factory
    rows the migration prepared, is the review pool: it names what runs, and says loudly
    when no row is marked (``pool_empty``: the Main model reviews alone)."""
    from ouroboros import reviewer_slot_config as rs
    from ouroboros.review_pool_migration import ROLLBACK_SENTENCE

    where = (f"Snapshot: {snapshot_path}. {ROLLBACK_SENTENCE}" if snapshot_path
             else "No snapshot could be written, so there is no rollback source.")
    head = ("⚙️ Review pool: the subagent catalog set in the environment (OUROBOROS_SUBAGENTS) is in force. "
            "This document had no review settings of its own (no authored review lanes, no saved subagent "
            "catalog), so the factory reviewer rows were prepared for it — but a catalog the environment "
            "carries is explicit configuration and runs instead; the factory rows do not.")
    state = rs.review_pool_state(catalog_text)
    if state["state"] == "error":
        body = f"That catalog cannot be read ({state['error']}): no review runs until it is repaired."
    elif state["state"] == "empty":
        body = ("None of its rows is marked “Reviewer”, so the review pool is empty (pool_empty): the Main model "
                "reviews alone, one model with no second opinion.")
    else:
        rows = rs.review_pool_rows({"OUROBOROS_SUBAGENTS": catalog_text})
        body = (f"{len(rows)} reviewer rows, {len({row.target_id for row in rows})} distinct models: "
                + "; ".join(f"{row.slot_id} ({row.target_id})" for row in rows) + ".")
    return "\n".join([head, body, f"{where} Adjust in Settings → Agents or in the environment."])


def _prune_event(event_type: str, keys: tuple, **reports: dict) -> None:
    """Emit when a report has evidence under ``keys``. GC no-ops stay silent;
    an observation report with measured journal sizes intentionally rows at boot."""
    from supervisor.state import append_jsonl

    if any(report.get(key) for report in reports.values() for key in keys):
        append_jsonl(DATA_DIR / "logs" / "events.jsonl",
                     {"ts": utc_now_iso(), "type": event_type, **reports})


def _prune_worktrees() -> None:
    """Subagent working copies past the GC retention, on the custody pass: a plain copy goes only
    when the supervisor's settlement probe proves its task not live (unknown keeps it)."""
    try:
        from ouroboros import subagent_worktrees
        from supervisor.queue import task_settlement_liveness

        _prune_event("subagent_worktree_prune", ("removed",),
                     report=subagent_worktrees.prune_orphans(live=task_settlement_liveness))
    except Exception:
        log.debug("Subagent worktree prune failed", exc_info=True)


def _prune_engine_runtimes() -> None:
    """Engine program copies in ``state/cx`` neither pinned nor serving (owner Q6 = A), on the custody pass."""
    try:
        from ouroboros.claudexor_runtime_retention import prune_unpinned_runtimes

        _prune_event("engine_runtime_prune", ("removed", "errors"), report=prune_unpinned_runtimes())
    except Exception:
        log.debug("Engine runtime prune failed", exc_info=True)


from ouroboros.maintenance_retirement import prune_agent_media_uploads as prune_agent_media_uploads


# The server mirror of the durable startup_recovery gate; the child reads only the stored fact.
_STARTUP_RECOVERY_GAPS = [False]
# The full-text memory journals whose growth boot reports; the report never opens or rewrites them.
_OBSERVED_JOURNALS = ("memory/identity_journal.jsonl", "memory/world_journal.jsonl",
                      "memory/knowledge_history.jsonl", "memory/knowledge/patterns_history.jsonl")


def _memory_journal_sizes(root: pathlib.Path) -> Dict[str, Any]:
    """The ``memory_journal_observation`` report, from ``lstat`` alone. A missing journal is ``None``,
    not a measured zero; a non-regular entry or another ``OSError`` is ``None`` with an error."""
    sizes: Dict[str, Optional[int]] = {}
    errors: list = []
    for relative in _OBSERVED_JOURNALS:
        try:
            info = (root / relative).lstat()
            sizes[relative] = info.st_size if stat.S_ISREG(info.st_mode) else None
            if sizes[relative] is None:
                errors.append(f"{relative}: not_regular")
        except FileNotFoundError:
            sizes[relative] = None
        except OSError as exc:
            sizes[relative] = None
            errors.append(f"{relative}: {type(exc).__name__}")
    return {"errors": errors, "journal_bytes": sizes}


def _startup_prune_sweeps(*, preserve_task_sources=False, recovery_report=None):
    """Boot's housekeeping: note whether recovery left a gap, reap the fallback scripts (only before
    intake, when none can be live) and observe the memory journals' sizes once per generation. History-sized
    retirers run in the maintenance child; queue-owned copies stay on the custody pass."""
    _STARTUP_RECOVERY_GAPS[0] = bool(preserve_task_sources and (
        recovery_report is None or recovery_report.get("errors") or "*" in recovery_report.get("unresolved", [])))
    from ouroboros.maintenance_retirement import record_startup
    record_startup(DATA_DIR, gaps=_STARTUP_RECOVERY_GAPS[0])
    if not preserve_task_sources:  # the atomic-temp walk is the maintenance child's temp_sweep job
        from ouroboros.temp_sweep import sweep_stale_temp_files
        sweep_stale_temp_files(DATA_DIR, atomic_temps=False)
    try:
        _prune_event("memory_journal_observation", ("journal_bytes", "errors"),
                     report=_memory_journal_sizes(DATA_DIR))
    except Exception:
        log.debug("Memory journal observation failed", exc_info=True)


def _run_reconcile_prunes(stop_event: Any = None):
    """Request a cheap child dispatch check; history-sized retirement lives there."""
    if not _stop_requested(stop_event):
        from ouroboros.maintenance import child
        child.request()


# Memory only: the views pass of the review checkout prune runs once per process, until done.
_REVIEW_VIEWS_OWED = [True]
_PROCESS_STARTED = time.time()


def _prune_review_checkouts(*, stop_event: Any = None) -> None:
    """Review checking copies live only as long as their review (owner 1A, 2026-10-12).
    Live ownership is asked again for every thread seat; when it cannot be read, nothing is
    removed. The views pass stays owed to later passes until one completes."""
    try:
        from functools import partial

        from ouroboros.review_checking_prune import prune_review_checkouts

        live = partial(_startup_live_task_ids, DATA_DIR, include_pending=True)
        live()
        report = prune_review_checkouts(
            DATA_DIR, live, views_since=_PROCESS_STARTED if _REVIEW_VIEWS_OWED[0] else None,
            stop=lambda: _stop_requested(stop_event))
        if report.get("views_done"):
            _REVIEW_VIEWS_OWED[0] = False
        _prune_event("review_checkout_prune", ("released", "removed", "checkouts_removed", "errors", "skipped"),
                     report=report)
    except Exception:
        log.debug("Review checkout prune failed", exc_info=True)


def _cursor_refresh_settled_terminals(live_task_ids: Any = None) -> None:
    """Cursor-driven pass: runs settled OUTSIDE a generation's reconcile outcomes (terminal-boundary
    settlements, earlier generations) never reappear in the orphan sweep, so their stored evidence
    would stay stale forever. Bounded to newly appended custody rows per tick; reads the same
    live-owner source as both custody surfaces (a still-billing owner defers the heal). At BOOT it
    runs AFTER the D1a backfill (``_startup_custody_sweep``), keeping its ``boot_backfill`` attribution."""
    try:
        from ouroboros.delegate_terminal import refresh_recently_settled_terminals

        refreshed = refresh_recently_settled_terminals(DATA_DIR, live_task_ids=live_task_ids)
        if refreshed:
            log.info("Cursor refresh healed %d stale terminal result(s)", refreshed)
    except Exception:
        log.debug("Cursor terminal-refresh pass failed", exc_info=True)


def _startup_custody_sweep() -> None:
    """Both custody surfaces, swept once per generation at supervisor startup.

    Live owners can survive an in-process supervisor revival. The shared
    installation daemon and those owners keep their existing custody.
    """
    try:
        from ouroboros.claudexor_daemon import CUSTODY_PURPOSE
        from ouroboros.process_custody import reap_orphaned_processes

        reaped = reap_orphaned_processes(
            DATA_DIR, live_owner_skills=_installed_skill_names(),
            running_task_ids=_live_task_ids,
            retained_purposes={CUSTODY_PURPOSE},
        )
        if reaped:
            log.info("Process custody reaper killed %d orphaned process(es): %s", len(reaped), reaped)
    except Exception:
        log.debug("Process custody startup reap failed", exc_info=True)
    try:  # after the reaper: open local custody whose owner positively ended (never by generation alone)
        from ouroboros.local_custody_repair import reconcile_open_local_custody

        retired = reconcile_open_local_custody(DATA_DIR)
        if retired:
            from supervisor.state import append_jsonl

            log.info("Startup local custody retired %d ended owner row(s)", len(retired))
            append_jsonl(DATA_DIR / "logs" / "events.jsonl",
                         {"ts": utc_now_iso(), "type": "local_custody_retired", "phase": "startup", "retired": retired})
    except Exception:
        log.warning("Startup local custody reconcile failed; open custody stays held", exc_info=True)
    try:  # ended tasks' leftovers (abrupt death): settled on positive terminal evidence only
        from ouroboros.process_scope import settle_task_scopes
        settle_task_scopes(DATA_DIR, _live_task_ids)
    except Exception:
        log.debug("Task scope startup sweep failed", exc_info=True)
    _reconcile_delegated_runs(_live_task_ids)
    try:
        # D1a boot backfill, ONCE per generation and AFTER the orphan reconcile
        # (so this generation's settlements are already visible to the audit):
        # a run settled in a PREVIOUS generation never appears in any current
        # pass's outcomes, so the sweep-side refresh above can never reach its
        # task's stored disclosure — the backfill joins from the stored terminal
        # results instead and heals every generation-crossing stale row.
        from ouroboros.delegate_terminal import backfill_terminal_reconciliations

        refreshed = backfill_terminal_reconciliations(DATA_DIR)
        if refreshed:
            log.info("Boot custody backfill refreshed %d stored disclosure(s): %s",
                     len(refreshed), refreshed)
    except Exception:
        log.debug("Boot custody-disclosure backfill failed", exc_info=True)
    try:
        # Boot half of the durable terminal outbox: an answer that was registered
        # as owed but whose send never completed (crash between settle and send)
        # is re-enqueued exactly once — the delivered registry suppresses a copy
        # that actually landed.
        from supervisor.terminal_delivery import replay_pending_deliveries

        replay_pending_deliveries(DATA_DIR)
    except Exception:
        log.debug("Boot replay of pending terminal deliveries failed", exc_info=True)


def _prune_delegated_snapshots() -> None:
    """C1 delegated execution snapshots: GC cross-checked against custody. A snapshot
    stays while its run is open/undisposed OR a pending invocation names it. Closed
    snapshots go only once their owning task is proven terminal and not live (including
    lineage processes); a provisional snapshot can precede the start request. Custody
    is re-read before each removal. Fail-soft like every step in the custody pass.

    FAIL-CLOSED on an unreadable custody log (CR1-1): the keep-set replays the
    custody rows and ``_iter_rows`` swallows its own OSError, so an unreadable log
    would replay as "no open runs", empty the keep-set and destroy every live
    snapshot with the child's only copy of its work. GC deletes only over PROVEN
    settled && patch_disposed; an UNKNOWN custody state skips the prune, loudly. So does an
    owed obligations rebuild: a custody row that landed while its set could not be updated
    is missing from the set until the next start merges it."""
    try:
        from ouroboros import delegate_custody as _delegate_custody
        from ouroboros import subagent_worktrees as _snap_worktrees
        from supervisor.state import append_jsonl

        reason = "custody_log_unreadable" if _delegate_custody.custody_log_unreadable(DATA_DIR) else ""
        if reason:
            log.warning("Delegated snapshot prune SKIPPED (%s): open snapshots are unknowable (fail-closed)", reason)
            if not append_jsonl(DATA_DIR / "logs" / "events.jsonl", {
                "ts": utc_now_iso(),
                "type": "delegated_snapshot_prune_skipped",
                "reason": reason,
            }):
                # CR2-2: the log is unwritable too — the promised durable row
                # could not land. Escalate loudly; the skip itself already
                # protects the open snapshots, so this stays fail-soft.
                log.error(
                    "Delegated snapshot prune skip could NOT be recorded durably: "
                    "the delegated_snapshot_prune_skipped row was not written "
                    "(custody event log unwritable). Open snapshots remain "
                    "protected by the skip itself.")
            return
        snapshot_report = _snap_worktrees.prune_execution_snapshots(
            _delegate_custody.open_snapshot_ids(DATA_DIR))
        if snapshot_report.get("removed"):
            append_jsonl(DATA_DIR / "logs" / "events.jsonl", {
                "ts": utc_now_iso(),
                "type": "delegated_snapshot_prune",
                "report": snapshot_report,
            })
    except Exception:
        log.debug("Delegated execution snapshot prune failed", exc_info=True)




def _run_periodic_reconcile_sweep(marker: list, stop_event: Any = None, latch: Any = None) -> None:
    """Off-loop 300 s heal of addressed work (orphans, preparations, the project registry, review
    collection, late reviews) and the age-only retirers (``_run_reconcile_prunes``), then the
    drive-custody pass (bounded settlement under the queue lock), then the maintenance dispatcher
    (history-sized jobs run in its child, never here).
    The marker is stamped when the pass ENDS, before its latch opens; the 20 s cancel cadence
    never waits on any walk and a slow pass never restarts (issue #1230). The generation token
    is asked before every step and, by the mutation owners, again before each publication commits."""
    try:
        _periodic_zombie_reconcile(stop_event=stop_event)
        if not _stop_requested(stop_event):
            from supervisor.workspace_preparation import recover_interrupted_preparations
            recover_interrupted_preparations(DATA_DIR, stop=lambda: _stop_requested(stop_event))
        if not _stop_requested(stop_event):
            _run_reconcile_prunes(stop_event)
        if _stop_requested(stop_event):
            return
        from ouroboros.late_review_marks import replay_owed_late_reviews
        from ouroboros.review_operation import retire_dead_review_pointers_softly

        # This process's closed operations retry a failed final pointer; a dead controller's open one goes.
        retire_dead_review_pointers_softly(DATA_DIR, stop=lambda: _stop_requested(stop_event))
        replay_owed_late_reviews(DATA_DIR, stop=lambda: _stop_requested(stop_event))  # owed rows only
        if _stop_requested(stop_event):
            return
        _run_drive_custody_pass(stop_event)
        if not _stop_requested(stop_event):  # history-sized named jobs: the one maintenance child, never here
            _tick_maintenance_child()
        _step_recovered("reconcile_sweep")
    except Exception:
        _step_failed("reconcile_sweep", "Periodic %s failed")
    finally:
        marker[0] = time.time()
        (latch or _RECONCILE_SWEEP_LOCK).release()


def _tick_maintenance_child() -> None:
    """The maintenance dispatcher: reserve the one child slot (``ouroboros/maintenance.py``); its
    thread reads the job facts and spawns the child only when one is due."""
    from ouroboros.config import REPO_DIR
    from ouroboros.maintenance import child

    child.tick(pathlib.Path(DATA_DIR), pathlib.Path(REPO_DIR))


class HostDuty(NamedTuple):
    """One recurring pass the supervisor tick starts off-loop: its journal name (``host_duty_stall``),
    the single-flight latch the tick takes before starting it (``None``: the pass keeps its own slot)
    and the function that runs ONE pass on the thread the tick starts. Its cadence is the due
    condition beside its start in ``_periodic_supervisor_maintenance``, never a number of its own."""
    name: str
    latch: Optional[threading.Lock]
    run: Callable[..., Any]


# The reconciled duty inventory (ARCHITECTURE §5 "Maintenance and custody", invariant 24): every
# recurring pass ``_periodic_supervisor_maintenance`` starts, with the latch it takes there; the
# 20 s group is dispatched from this table, the others keep their own due conditions above and
# are started under the latch named here. The delegated body's tick
# (``delegate_collection.BodyRuntime``, the lifespan's own thread) and ``maintenance.JOBS`` (the
# child's) are the two other recurring passes of a server process. tests/test_cost_contract.py B7
# drives each of the three families twice over a scale root and requires the second pass to rest;
# tests/test_duty_inventory.py fails when the tick starts a pass this table does not name, or a
# row here names a pass the tick never starts.
HOST_DUTIES: Tuple[HostDuty, ...] = (
    HostDuty("cancel_intent_sweep", _CANCEL_INTENT_SWEEP_LOCK, _run_terminal_duty),
    HostDuty("terminal_delivery_replay", _DELIVERY_SWEEP_LOCK, _run_terminal_duty),
    HostDuty("question_delivery_replay", _QUESTION_SWEEP_LOCK, _run_terminal_duty),
    HostDuty("pause_notice_replay", _PAUSE_NOTICE_SWEEP_LOCK, _run_terminal_duty),
    HostDuty("abandoned_usage_reconciliation", _USAGE_SWEEP_LOCK, _run_terminal_duty),
    HostDuty("post_task_synthesis_recovery", _SYNTHESIS_SWEEP_LOCK, _run_terminal_duty),
    HostDuty("held_continuation_recheck", _CONTINUATION_SWEEP_LOCK, _run_terminal_duty),
    HostDuty("owner_pause_settlement", _OWNER_PAUSE_SWEEP_LOCK, _run_terminal_duty),
    HostDuty("cold-custody", _COLD_CUSTODY_SWEEP_LOCK, _run_cold_custody),
    HostDuty("custody-maintenance", _CUSTODY_SWEEP_LOCK, _run_periodic_custody_sweep),
    HostDuty("reconcile-maintenance", _RECONCILE_SWEEP_LOCK, _run_periodic_reconcile_sweep),
    HostDuty("maintenance-child", None, _tick_maintenance_child),
    HostDuty("schedule-admission", _SCHEDULE_SWEEP_LOCK, _run_schedule_duty),
)
TWENTY_SECOND_DUTIES = HOST_DUTIES[:6]  # the one cadence stamped by _LAST_CANCEL_INTENT_SWEEP


def host_duty(name: str) -> HostDuty:
    return next(duty for duty in HOST_DUTIES if duty.name == name)


SETTLEMENT_VERDICTS = "drive_settlement_verdicts"  # meta: each retained drive's verdict with its basis (task_custody)


def settlement_verdicts(root: pathlib.Path) -> Dict[str, Any]:
    """The retained drives' durable verdicts (``task_custody.rested_verdict``): decoded by a settlement lap
    that walks, never by a pass whose layouts rest."""
    from ouroboros import store

    with store.open_store(root, write=False) as txn:
        value = txn.meta(SETTLEMENT_VERDICTS)
    return {key: dict(item) for key, item in value.items() if isinstance(item, dict)} if isinstance(value, dict) else {}


def _run_drive_custody_pass(stop_event: Any = None) -> None:
    """The settled canonical mailboxes the loop thread's seam left (unread inputs to carry),
    then settle terminal child and direct drives past retention (a cancelled subagent's at
    once) through ``task_custody.settle_child_drive`` with the supervisor's probe and
    ownership interlock, at most ``DRIVE_SETTLEMENTS_PER_PASS`` attempts per layout and
    pass, continuing from the previous pass's cursor; crashed-settlement staging is swept
    first. Off the loop thread: each settlement may copy and hash a child store. It stays in
    the server: the queue lock and the liveness probe are this process's. The cursor and each
    layout's rest (``task_custody.layout_basis``: the layout directory, the terminal-write counter,
    the server generation, the retention setting, the release, plus the earliest retention instant
    of a drive left young) are the ``drive_settlement`` job row; the verdicts of retained drives are
    the ``drive_settlement_verdicts`` meta row, decoded only by a lap that walks. Both are written
    only when they move, so a restart neither restarts the lap nor re-hashes an unchanged retained
    drive, a layout whose inputs stand is not even listed, a task completing elsewhere costs one
    indexed read of the rows written terminal since, and a new drive its creator published
    (``task_custody.record_drive_arrival``) one look at that drive, its row forgotten here. A drive
    kept because its custody is not proven is material evidence in the event. The moved drives'
    deletion and the pending child-ref promotions are the maintenance child's jobs."""
    from ouroboros import maintenance_jobs, store
    from ouroboros.headless import (DRIVE_SETTLEMENTS_PER_PASS, HEADLESS_TASKS_DIR, TASK_DRIVES_DIR,
                                    prune_headless_task_drives, prune_task_drives)
    from ouroboros.owner_mailbox import sweep_settled_owner_mailboxes
    from ouroboros.task_custody import forget_arrivals, sweep_custody_leftovers
    from supervisor.queue import task_settlement_interlock, task_settlement_liveness

    def stop() -> bool:
        return _stop_requested(stop_event)

    root = pathlib.Path(DATA_DIR)
    sweep_custody_leftovers(root)
    # Canonical mailboxes the loop thread's seam left because unread rows carry inputs to
    # verify or copy: carried and unlinked here, off the loop thread.
    mailboxes = sweep_settled_owner_mailboxes(root, stop=stop)
    row = maintenance_jobs.job(root, "drive_settlement") or {}
    cursors = {key: str(value or "") for key, value in (row.get("cursor") or {}).items()}
    rests = {key: dict(value) for key, value in (row.get("rest") or {}).items() if isinstance(value, dict)}
    before = json.dumps([cursors, rests], sort_keys=True)
    legacy = row.get("retained")  # an older body kept the verdicts on this row: they move to their own once
    loaded: Dict[str, Any] = {"map": dict(legacy), "before": ""} if isinstance(legacy, dict) else {}

    def rested() -> Dict[str, Any]:
        if "map" not in loaded:
            loaded["map"] = settlement_verdicts(root)
            loaded["before"] = json.dumps(loaded["map"], sort_keys=True)
        return loaded["map"]

    reports, looked = {}, {}
    layouts = (("headless", prune_headless_task_drives, HEADLESS_TASKS_DIR), ("direct", prune_task_drives, TASK_DRIVES_DIR))
    for key, prune, base in layouts:
        if stop():
            return
        reports[key] = prune(root, live=task_settlement_liveness, guard=lambda: task_settlement_interlock(stop=stop),
                             stop=stop, budget=DRIVE_SETTLEMENTS_PER_PASS, after=cursors.get(key, ""), rested=rested,
                             rest=rests.setdefault(key, {}))
        cursors[key] = str(reports[key].get("cursor") or "")
        looked[base.name] = list(reports[key].get("arrivals") or ())
    moved_map = "map" in loaded and json.dumps(loaded["map"], sort_keys=True) != loaded["before"]
    moved_row = json.dumps([cursors, rests], sort_keys=True) != before or "retained" in row
    if moved_map or moved_row or any(looked.values()):
        with store.open_store(root, write=True) as txn:  # one transaction: the lap's facts and its verdicts move together
            if moved_map:
                txn.set_meta(SETTLEMENT_VERDICTS, loaded["map"])
            if "retained" in row:
                maintenance_jobs.forget_in(txn, "drive_settlement", "retained")
            maintenance_jobs.record_in(txn, "drive_settlement", cursor=cursors, rest=rests)
            for base_name, task_ids in looked.items():
                forget_arrivals(txn, base_name, task_ids)  # the published drives this lap looked at
    _prune_event("headless_task_drive_prune", ("pruned", "errors", "custody_pending", "removed"),
                 report=reports["headless"], task_drives=reports["direct"], mailboxes=mailboxes)


def _periodic_zombie_reconcile(*, stop_event: Any = None) -> None:
    """Heal zombie 'running' records on a supervisor cadence. A worker that died
    mid-review (crash / SIGKILL / manual stop) leaves ``review_job.json`` at running
    until the next boot reconcile (``GET /api/extensions`` is a passive read and
    heals nothing); the same death leaves the current task-store record at
    running. Both reconciles are liveness-gated (pid-dead / queue-empty + worker-boot
    evidence), so a live review or task is never touched. Off the loop thread, so
    ``stop_event`` (the generation token) is re-read before every step."""
    if _stop_requested(stop_event):
        return
    try:
        from ouroboros.skill_review_runner import reconcile_stale_review_jobs
        reconcile_stale_review_jobs(DATA_DIR)
    except Exception:
        log.debug("Periodic skill review-job reconcile failed", exc_info=True)
    if _stop_requested(stop_event):
        return
    try:
        from ouroboros.task_status import reconcile_orphaned_running_tasks

        expired_quizzes: list = []
        reconcile_orphaned_running_tasks(
            DATA_DIR, expired_quizzes=expired_quizzes, boot=False,
            write_guard=lambda task_id: orphan_reconcile_write_guard(task_id, stop_event=stop_event),
        )
        _publish_expired_quiz_frames(expired_quizzes)
    except Exception:
        log.debug("Periodic orphaned running-task reconcile failed", exc_info=True)
    if _stop_requested(stop_event):
        return
    try:
        from ouroboros.projects_registry import reconcile_projects
        reconcile_projects(DATA_DIR)
    except Exception:
        log.debug("Project registry reconcile failed", exc_info=True)
    if not _stop_requested(stop_event):
        _resume_interrupted_project_deletions()


def _resume_interrupted_project_deletions() -> None:
    try:
        from supervisor.task_lifecycle import resume_project_deletions

        resume_project_deletions(DATA_DIR)
    except Exception:
        log.debug("Project deletion recovery failed", exc_info=True)


def _migrate_startup_cancel_latches(drive_root: pathlib.Path) -> None:
    """Before restore/readers can quarantine an unstamped legacy cancel latch."""
    try:
        # Phase A boot migration: legacy ``cancel_requested`` status latches
        # become ordinary durable cancel intents; the supervisor watchdog then
        # drives each through custody to a real settled outcome.
        from ouroboros.startup_migrations import migrate_cancel_latches

        migrated = migrate_cancel_latches(drive_root)
        if migrated:
            log.info("Migrated %d legacy cancel latch(es) to durable intents: %s",
                     len(migrated), migrated)
    except Exception:
        log.debug("Legacy cancel-latch migration failed", exc_info=True)


def _publish_expired_quiz_frames(expired: list) -> None:
    """Tell already-rendered cards that a healed terminal expired their question:
    the same frame the task-done seam sends, from the supervisor-side healer that
    writes terminals off that seam (the packaged shell, mini app and phone have no
    reload affordance). Fail-soft: the durable projection is right without it."""
    if not expired:
        return
    try:
        from supervisor.message_bus import get_bridge

        bridge = get_bridge()
        for task_id, quiz_id in expired:
            bridge.send_quiz_state(str(quiz_id), str(task_id), "expired_terminal")
    except Exception:
        log.debug("Expired-quiz frames after orphan reconcile were not sent", exc_info=True)


def _startup_live_task_ids(drive_root: pathlib.Path, *, include_pending: bool = False) -> set[str]:
    """Existing queue/direct/post-task owners; recovery also preserves pending work."""
    from ouroboros.post_task_checkpoint import POST_TASK_SYNTHESIS_INFLIGHT, POST_TASK_SYNTHESIS_LOCK
    from supervisor import queue, workers
    from supervisor.active_activity import get_direct_activity_registry
    from ouroboros.sleep_custody import accepted_owner_ids

    with queue._queue_lock:
        ids = set(queue.RUNNING)
        if include_pending:
            ids.update(str(task.get("id") or "") for task in queue.PENDING)
        ids.update(w.busy_task_id for w in workers.WORKERS.values() if w.busy_task_id)
    ids.update(row["activity_id"] for row in get_direct_activity_registry().snapshot())
    ids.update(accepted_owner_ids(queue))
    root = str(pathlib.Path(drive_root).resolve(strict=False))
    with POST_TASK_SYNTHESIS_LOCK:
        ids.update(task_id for (path, task_id) in POST_TASK_SYNTHESIS_INFLIGHT if path == root)
    if include_pending:
        # No-provider startup has not restored PENDING. The saved continuation
        # remains owned by queue restore, never by file/post-task recovery.
        from ouroboros.task_status import _load_queue_snapshot
        snapshot = _load_queue_snapshot(pathlib.Path(drive_root))
        if snapshot.get("_snapshot_invalid"):
            raise ValueError("startup pending ownership is unreadable")
        ids.update(str(row["task"].get("id") or "") for row in snapshot.get("pending", [])
                   if isinstance(row, dict) and isinstance(row.get("task"), dict)
                   and row["task"].get("_owner_wait_resume"))
        saved = [row["task"] for row in snapshot.get("pending", [])
                 if isinstance(row, dict) and isinstance(row.get("task"), dict)]
        ids.update(accepted_owner_ids(queue, direct=saved))
    return ids - {""}


def _startup_worker_pids(drive_root: pathlib.Path) -> set[int] | None:
    """Capture prior worker ownership from the process ledger before spawn.

    These observations only defer recovery; they grant no signal authority. An
    unconfirmed survivor or unreadable record never proves safe file capture.
    """
    from ouroboros.process_custody import _read_ledger_strict
    try:
        readable, entries = _read_ledger_strict(pathlib.Path(drive_root))
        if not readable:
            return None
        pids = {int(row["pid"]) for row in entries if str(row.get("purpose") or "").startswith("worker:")}
        return pids - {0, os.getpid()}
    except (OSError, ValueError, TypeError, KeyError):
        log.warning("Startup worker ownership is unreadable; file recovery is deferred", exc_info=True)
        return None


def _recover_terminal_task_files(drive_root: pathlib.Path, protected: set[str]) -> dict:
    from ouroboros.startup_task_files import recover_terminal_task_files
    return recover_terminal_task_files(drive_root, protected)


def _run_startup_task_recovery(
    drive_root: pathlib.Path, repo_dir: pathlib.Path, *, skip_live_data: bool,
    prior_worker_pids: set[int] | None = None,
) -> dict:
    """File recovery precedes the orphan reconcile and the caller's actual prune.

    Provider boot calls this from the supervisor, after process custody; the
    no-provider lifespan calls it without spawning anything. There is no racing
    lifespan recovery once a supervisor can run. Unknown/live prior ownership
    defers the destructive work, without waiting or resurrecting a model task.
    """
    report = {"recovered": [], "unresolved": [], "protected": [], "errors": []}
    if skip_live_data:
        return report
    from ouroboros.platform_layer import pid_is_alive
    from ouroboros.store import StoreMigrating
    from ouroboros.store_switch import require_published

    try:
        require_published(drive_root)  # the store is published before any recovery write (P1)
    except StoreMigrating as exc:
        report["errors"].append("store_not_published")
        log.warning("Startup task recovery deferred: %s", exc)
        return report
    if prior_worker_pids is None or any(pid_is_alive(pid) for pid in prior_worker_pids):
        report["errors"].append("prior_worker_ownership_unconfirmed")
        log.warning("Startup file/task recovery deferred: prior worker ownership is unconfirmed")
        return report
    try:
        from ouroboros.task_status import _load_queue_snapshot
        snapshot = _load_queue_snapshot(pathlib.Path(drive_root))
        saved = [row["task"] for row in snapshot.get("pending", [])
                 if isinstance(row, dict) and isinstance(row.get("task"), dict)]
        _reconcile_cold_custody(drive_root, saved=saved)
        from supervisor.workspace_preparation import recover_interrupted_preparations
        preparations = recover_interrupted_preparations(drive_root, stop=_stop_requested)
        protected = _startup_live_task_ids(drive_root, include_pending=True)
    except Exception as exc:
        report["errors"].append(f"startup_ownership_unconfirmed: {type(exc).__name__}")
        log.warning("Startup task recovery deferred: live/pending ownership is unreadable", exc_info=True)
        return report
    intent_errors = []
    try:  # its own step: an unsettled new-work intent is no reason to skip the file recovery below
        from ouroboros.promotion_intent import reconcile_promotion_intents
        reconcile_promotion_intents(drive_root, protected=protected)
    except Exception:
        intent_errors.append("promotion_intent_reconcile_failed")
        log.warning("Startup new-work intent reconciliation deferred", exc_info=True)
    report = _recover_terminal_task_files(drive_root, protected)
    report["unresolved"].extend(preparations["retained"] + preparations["uncertain"])
    report["errors"].extend(f"preparation:{item}" for item in preparations["errors"])
    report["errors"].extend(intent_errors)
    excluded = protected | set(report["unresolved"])
    if report["errors"]:
        log.warning("Startup task file recovery has gaps: %s", report)
    if "*" in excluded:
        return report  # Directory enumeration could not identify the unsafe rows.
    try:
        from ouroboros.task_status import reconcile_orphaned_running_tasks

        expired_quizzes: list = []
        reconcile_orphaned_running_tasks(
            drive_root, exclude_task_ids=excluded, expired_quizzes=expired_quizzes,
            write_guard=orphan_reconcile_write_guard,
        )
        _publish_expired_quiz_frames(expired_quizzes)
    except Exception:
        report["errors"].append("orphan_recovery_failed")
        log.warning("Orphaned running-task reconciliation at startup failed", exc_info=True)
    try:
        from ouroboros.agent_task_pipeline import recover_pending_root_post_task_synthesis

        recover_pending_root_post_task_synthesis(drive_root, repo_dir, exclude_task_ids=excluded)
    except Exception:
        report["errors"].append("synthesis_recovery_failed")
        log.warning("Root post-task synthesis recovery at startup failed", exc_info=True)
    from ouroboros.obligations import members
    try:
        report["unresolved"].extend(facts.get("task_id", identity) for identity, facts in members(drive_root, "unknowns").items())
    except Exception:
        report["unresolved"].append("*")
        report["errors"].append("unknown_obligations_unavailable")
        log.warning("Startup repair gaps unavailable; destructive pruning remains deferred", exc_info=True)
    return report
