"""Restart-adjacent helpers the composition root calls at shutdown time.

The live-task census the restart drain consults, the teardown arguments that
finalize interrupted tasks with an honest reason, the managed-update guard on
preserving queued work, the checkout/update serialization gate, the owner's manual
Restart operation, the planned restart's engine-pin daemon stop,
and the event bus shutdown live here. The planned restart also uses this lifecycle; its composition-root bindings
join the deferred drain record to the performer that raises the exit signal.
"""

from __future__ import annotations

import os
import pathlib
import threading
import time
from typing import Any, Dict

from ouroboros.server_process import (
    DATA_DIR, _owner_restart_requested, _request_restart_exit, _restart_requested, log,
)

_RESTARTABLE_UPDATE_PHASES = frozenset({"pending_boot_smoke", "applying_replace"})
_LAST_RETURNING: set = set()  # this process's Restart: the ids its transaction returns
_LAST_RETURN_TX: dict = {"id": ""}  # ...and the one transaction that names them
_OWNER_STOP_BEGAN = threading.Event()  # ...and that its own owned stop began (the pin check keeps its stop)

# The current agent's request, before its finalization starts the existing drain.
# Q owns this record. The boundary serializes acceptance with dispatch marks and
# direct start, never with the requester's remaining work or the drain.
_AGENT_RESTART: dict = {}
_AGENT_RESTART_REPLY: dict = {}
agent_restart_boundary = threading.RLock()


def agent_restart_pending() -> bool:
    return _AGENT_RESTART.get("status") in {"awaiting", "accepted", "ready"}


def agent_restart_snapshot() -> dict:
    """Host ACK over the existing queue publication; never restored as admission authority."""
    return {key: _AGENT_RESTART[key] for key in
            ("request_id", "task_id", "task_attempt", "status", "reason", "refusal") if key in _AGENT_RESTART}


def agent_restart_holds(task: dict) -> bool:
    """Only work anchored in this generation's already-started tree may start.

    A parent id alone grants nothing. Each link must name the same root and an
    admitted parent; Pause, custody, money and continuation checks still apply.
    Caller-supplied retry/resume flags cannot manufacture an exemption.
    """
    if not agent_restart_pending():
        return False
    from ouroboros.store import acquisition_budget
    from ouroboros.task_results import load_task_result
    from supervisor import queue
    from supervisor.task_admission import (
        _exact_resume_granted,
        _owner_wait_current,
        _working_resume_granted,
        governing_continuation,
    )

    allowed = _AGENT_RESTART.get("started") or {}
    governing = governing_continuation(task)
    if governing:
        try:
            with acquisition_budget(0):
                root_path = task.get("budget_drive_root") or queue.DRIVE_ROOT
                stored = load_task_result(root_path, task.get("id"), strict=True) or {}
                continuing = (_owner_wait_current(task, stored) if governing == "owner_wait"
                              else _exact_resume_granted(task, root_path, existing=stored) if governing == "exact_pause"
                              else _working_resume_granted(task, root_path, existing=stored))
            if continuing:
                return False  # the existing launch owners still validate control, source, custody and money
        except Exception:
            return True  # unreadable continuation authority cannot make new work eligible
    root = str(task.get("root_task_id") or task.get("id") or "")
    rows = {str(row.get("id") or ""): row for row in queue.PENDING}
    rows.update({tid: meta.get("task") or {} for tid, meta in queue.RUNNING.items()})
    seen = set()
    while task and str(task.get("id") or "") not in seen:
        tid = str(task.get("id") or "")
        if allowed.get(tid) == root:
            return False
        seen.add(tid)
        parent = str(task.get("parent_task_id") or "")
        if (not parent or task.get("delegation_role") != "subagent"
                or str(task.get("root_task_id") or "") != root):
            break
        if allowed.get(parent) == root:
            return False
        task = rows.get(parent)
    return True


def release_agent_restart(request_id: str, reason: str) -> None:
    """Release this request alone; never reopen an Update/Stop/Panic admission owner."""
    from supervisor import queue

    with queue._queue_lock:
        if _AGENT_RESTART.get("request_id") == request_id and agent_restart_pending():
            _AGENT_RESTART.update(status="released", refusal=reason)
            queue.persist_queue_snapshot(reason="agent_restart_released")


def defer_evolution_restart(tx: dict, *, terminal_retry=None, task_attempt=None) -> str:
    """Bind terminal persistence or its saved decision to the exact intent's existing tick."""
    from supervisor import queue

    with queue._queue_lock:
        if (not agent_restart_pending() or not _AGENT_RESTART.get("evolution_restart")
                or tx.get("task_id") != _AGENT_RESTART.get("task_id")
                or (task_attempt is not None and task_attempt != _AGENT_RESTART.get("task_attempt"))):
            return ""
        if _AGENT_RESTART.get("status") != "ready":
            if terminal_retry is not None:
                _AGENT_RESTART.setdefault("evolution_terminal", terminal_retry)
            else:
                _AGENT_RESTART["evolution_terminal"] = None  # saved cycle, not requester status, now owns the decision
                _AGENT_RESTART["ready_event"] = {
                    **{key: _AGENT_RESTART[key] for key in ("request_id", "task_id", "task_attempt", "reason")},
                    "phase": "ready", "evolution_restart": True, "evolution_transaction": dict(tx),
                }
        return _AGENT_RESTART["request_id"]


def _requester_failed(ctx) -> bool:
    """Only a known terminal failure of the exact requester releases its hold."""
    from ouroboros.cancel_intents import has_active_intent
    from ouroboros.store import acquisition_budget
    from ouroboros.task_results import load_task_result

    if not _AGENT_RESTART.get("task_id") or _AGENT_RESTART.get("evolution_restart"):
        # Evolution owns the decision from intent intake: a durable task failure
        # can precede its terminal event. Lifecycle still decides Stop and refusal.
        return False
    try:
        with acquisition_budget(0):
            if has_active_intent(ctx.DRIVE_ROOT, _AGENT_RESTART["task_id"], strict=True):
                return True
            row = load_task_result(ctx.DRIVE_ROOT, _AGENT_RESTART.get("task_id"), strict=True) or {}
        return (int(row.get("task_attempt") or 1) == _AGENT_RESTART.get("task_attempt")
                and row.get("status") in {"failed", "cancelled", "interrupted", "timeout", "error"})
    except Exception:
        return False  # silence, unreadable state and an absent result prove no failure


def _accept_agent_restart(ctx) -> bool:
    """Try acceptance without waiting behind a mark writer on the supervisor loop.

    Awaiting intent already holds unsent independent work. A mark in flight
    finishes first; the next tick accepts. The tool receives no success before
    the matching ACK is published. No diagnostic marker participates.
    """
    from supervisor import queue
    from supervisor.active_activity import get_direct_activity_registry

    if not agent_restart_boundary.acquire(blocking=False):
        return False
    try:
        if not queue._queue_lock.acquire(blocking=False):
            return False
        try:
            if _AGENT_RESTART.get("status") != "awaiting":
                return _AGENT_RESTART.get("status") in {"accepted", "ready"}
            tid, attempt = _AGENT_RESTART["task_id"], _AGENT_RESTART["task_attempt"]
            meta = queue.RUNNING.get(tid) or {}
            direct = get_direct_activity_registry().get(tid)
            context = getattr(getattr(getattr(direct, "actor", None), "tools", None), "_ctx", None)
            valid = ((meta and int(meta.get("attempt") or 1) == attempt)
                     or (direct is not None and getattr(context, "task_id", None) == tid
                         and int(getattr(context, "task_attempt", None) or 1) == attempt))
            if queue.RETURNS_NAMED or _restart_requested.is_set() or not valid:
                _AGENT_RESTART.update(status="refused", refusal="requester_not_active_or_intake_closed")
            else:
                _AGENT_RESTART.update(status="accepted", started=_started_restart_work())
            queue.persist_queue_snapshot(reason="agent_restart_intent")
            return _AGENT_RESTART.get("status") == "accepted"
        finally:
            queue._queue_lock.release()
    finally:
        agent_restart_boundary.release()


def _started_restart_work() -> dict:
    from supervisor import queue
    from supervisor.active_activity import get_direct_activity_registry
    from supervisor.worker_assignment import _unsent

    started = {key: str((value.get("task") or {}).get("root_task_id") or key)
               for key, value in queue.RUNNING.items()}
    started.update({entry.activity_id: entry.activity_id for entry in get_direct_activity_registry().actors()
                    if entry.execution is not None and entry.execution.phase not in {"accepted", "admitted"}})
    started.update({str(row["id"]): str(row.get("root_task_id") or row["id"])
                    for row in queue.PENDING if _unsent(row) is not None})
    return started


def _perform_owner_restart(ctx: Any, reply=None) -> tuple[bool, str]:
    """Run the owner restart operation with an optional transport notice."""
    ok, restart_msg = _safe_restart_serialized(
        ctx.safe_restart,
        reason="owner_restart",
        unsynced_policy="rescue_and_reset",
    )
    if not ok:
        return False, restart_msg
    from supervisor.workspace_preparation import pause_admission, resume_admission

    # Before this Restart's own restrictions exist: a launch or Resume already writing its marks hands over
    # to the still-running pool, which the follow-up restriction below would refuse at its put.
    pause_admission()
    try:
        from ouroboros.delegate_recovery import prepare_owner_restart
        from supervisor.queue_schedules import schedule_transaction
        with schedule_transaction(DATA_DIR):
            transaction = prepare_owner_restart(DATA_DIR)
            _LAST_RETURN_TX["id"] = transaction["transaction_id"]
            _OWNER_STOP_BEGAN.set()
            try:
                from supervisor.followup_policy import record_restart
                record_restart(DATA_DIR, new=True)
            except Exception:
                log.exception("Restart episode awaits boot persistence")
    except Exception:
        resume_admission()
        log.warning("Failed to write owner restart transaction", exc_info=True)
        return False, "could not write restart state."
    # Everything reversible is behind us (checkout landed, restart
    # door durable): from here the restart always follows, and every
    # unconfirmed stop is a critical diagnostic, never a deferral.
    try:
        stopped_task_ids = _stop_owned_work(ctx)
    except Exception:
        # The stop may not veto. Continuing this generation instead would leave the durable
        # door above live under it (reviews read it as a stop, follow-ups as a
        # restart in progress): the restart follows, and the next start reconciles the rest.
        log.critical("Owner restart: stopping owned work raised; the restart proceeds and the next "
                     "generation reconciles the remainder", exc_info=True)
        stopped_task_ids = []  # unknown: the notice claims no stopped task
    except BaseException:
        from supervisor.queue import reopen_queue_returns

        _OWNER_STOP_BEGAN.clear()
        reopen_queue_returns("owner_restart")  # no exit follows here: this generation keeps admitting
        raise
    try:
        if reply is not None:
            # Say only what happened: with nothing owned the stop sentence
            # named a task that was never running; saved work returns after it.
            held = _owner_restart_held_count(ctx)
            returning = len(_LAST_RETURNING)
            notice = ("Stopping active task. New settings apply to the next message."
                      if stopped_task_ids else "New settings apply to the next message.")
            if returning:
                notice += (f" {returning} task{'' if returning == 1 else 's'} will continue from "
                           f"{'its' if returning == 1 else 'their'} saved state after the restart.")
            if held:
                notice += (f" {held} queued task{'' if held == 1 else 's'} held until you resume "
                           f"{'it' if held == 1 else 'them'}.")
            reply(notice, "")
    except Exception:
        log.warning("Failed to send owner restart stop notice; continuing restart", exc_info=True)
    _request_restart_exit(owner=True)
    return True, ""


def _owner_restart_held_count(ctx: Any) -> int:
    """How many queued rows the Restart holds for an explicit Resume."""
    from supervisor.restart_retention import restart_held

    try:
        return sum(1 for task in list(ctx.PENDING or []) if restart_held(task))
    except Exception:
        return 0


def _owned_live_task_ids(ctx: Any) -> list:
    """Every id this generation's cancel intent can address: pooled tasks,
    in-process direct/ephemeral activities and running post-task synthesis —
    minus every saved pause (``restart_retention``): a task whose checkpoint
    is already stored survives the Restart as the same paused task, and a
    cancel intent minted for it here would outrank that pause at restore."""
    from supervisor.restart_retention import census_without_saved_pauses

    from ouroboros.post_task_checkpoint import POST_TASK_SYNTHESIS_INFLIGHT, POST_TASK_SYNTHESIS_LOCK
    from supervisor.active_activity import get_direct_activity_registry

    task_ids = list(dict(ctx.RUNNING or {}))
    task_ids.extend(str(row.get("activity_id") or "") for row in get_direct_activity_registry().snapshot())
    root = str(pathlib.Path(DATA_DIR).resolve(strict=False))
    with POST_TASK_SYNTHESIS_LOCK:
        task_ids.extend(task_id for (path, task_id) in POST_TASK_SYNTHESIS_INFLIGHT if path == root)
    return census_without_saved_pauses([tid for tid in dict.fromkeys(task_ids) if tid], DATA_DIR)


def _stop_owned_work(ctx: Any) -> list:
    """The owner's manual Restart: stop what this generation owns, then let it re-exec.

    Runs AFTER the checkout gate and the durable restart door, so nothing
    here can veto: an unconfirmed step is a critical diagnostic with custody
    retained, and the next generation's startup custody sweep reconciles the
    remainder. Its caller paused admission before that door
    (``pause_admission``), so no launch is left between its marks and its worker.
    First prepare the retained-return transaction: native saved work returns only
    after its fresh acknowledgment, while prior holds stay.
    Record cancel intents for owned live ids outside that return set (saved
    pauses excluded), then ``kill_workers`` with ``retain_saved_work``,
    ``preserve_pending``, preserved owner waits and
    ``reconcile_delegate_custody=False``. Tasks actually settled record the
    typed ``owner_restart`` cause (an earlier Stop keeps its own). External
    runs gain no adoption authority from native continuation: request cancellation
    through the public owner-gone seam over the attach-only gateway, the
    attested owned-daemon stop exactly as Panic makes it, and finally the
    generation's one bounded stop of its owned processes (``stop_owned_work``),
    which the teardown and the restart watcher later join. Between the cancel
    intents and the daemon stop nothing may call ``ensure_owned_gateway`` — it
    would start a dead daemon — which is what the prepared transaction and live latch guarantee.

    Returns the owned live task ids it addressed — captured ONCE, before the
    stop makes them unreadable — so the caller can tell the owner what was
    stopped instead of claiming a task was stopped when nothing was running.
    """
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.claudexor_daemon import read_owned_gateway
    from ouroboros.delegate_custody import reconcile_orphaned_runs
    from ouroboros.owned_shutdown import begin_owned_stop

    _OWNER_STOP_BEGAN.set()  # owner Rollback/Apply Update also restart as the owner, but never get here
    begin_owned_stop(DATA_DIR, cause="owner_restart")  # the grace starts here; cause and every pending stop are recorded before its waits
    returning, owner_waits = _prepare_owner_restart_returns(ctx)
    _LAST_RETURNING.clear()
    _LAST_RETURNING.update(returning)
    # owner 2026-10-08 (quiz d2f7532b): this Restart returns saved work instead of stopping it
    stopped = [task_id for task_id in _owned_live_task_ids(ctx) if task_id not in returning]
    for task_id in stopped:
        try:
            request_cancel(DATA_DIR, task_id, reason="Owner restart", source="owner_restart",
                           requested_by="owner", requested_stop_policy="immediate",
                           allow_settled_target=True)
        except Exception:
            log.warning("Owner restart: cancel intent for %s was not recorded", task_id, exc_info=True)
    try:
        confirmed = ctx.kill_workers(
            force=True, terminal_status="cancelled",
            result_reason="Owner restart stopped this task before process restart.",
            reconcile_delegate_custody=False, stop_source="owner_restart",
            retain_saved_work=True, preserve_running_task_ids=owner_waits,
            **{"preserve_pending": True, **_managed_update_pending_kwargs()},
        )
    except Exception:
        log.critical("Owner restart: worker shutdown raised; the restart proceeds and the next "
                     "generation reconciles the remainder", exc_info=True)
    else:
        if confirmed is False:
            log.critical("Owner restart: worker shutdown is unconfirmed; the restart proceeds and "
                         "the next generation reconciles the remainder")
    try:  # attach only; no replay_pending: a Restart re-POSTs nothing, the next generation's sweep does
        reconcile_orphaned_runs(DATA_DIR, running_task_ids=set(), gateway_factory=read_owned_gateway)
    except Exception:
        log.warning("Owner restart: delegated-run cancellation did not complete; custody retained",
                    exc_info=True)
    _stop_owned_daemon("Owner restart")
    from ouroboros.owned_shutdown import stop_owned_work

    stop_owned_work(DATA_DIR)  # unconfirmed records stay stamped for the next start; never a veto
    return stopped


def _prepare_owner_restart_returns(ctx: Any) -> tuple:
    """The owner's Restart returns active work (owner 2026-10-08, quiz d2f7532b).

    One restart transaction names it before anything stops: parked owner waits
    (cold handoffs), interrupted runs and direct turns with saved working
    state, and the formerly runnable queue. The launcher's exit-42
    acknowledgement makes it fresh; without that the next boot holds it all.
    Returns ``(returning_ids, owner_wait_ids)``; a failure returns nothing.
    """
    try:
        from ouroboros.owner_wait import prepare_owner_wait_handoffs
        from supervisor.queue import name_queue_returns
        from supervisor.restart_retention import prepare_restart_returns
        from supervisor.workers import direct_chat_turns

        transaction_id = _LAST_RETURN_TX["id"]
        owner_waits = prepare_owner_wait_handoffs(DATA_DIR, ctx.RUNNING, transaction_id, include_sleep=False)
        returning = prepare_restart_returns(DATA_DIR, dict(ctx.RUNNING or {}),
                                            name_queue_returns(ctx.PENDING or [], "owner_restart"),
                                            transaction_id=transaction_id, direct=direct_chat_turns(),
                                            owner_wait_ids=owner_waits)
        _LAST_RETURN_TX["id"] = transaction_id
        return returning | owner_waits, owner_waits
    except Exception:
        log.warning("Owner restart: saved work returns were not prepared; the next boot holds them", exc_info=True)
        return set(), set()


def arm_owner_restart_transaction() -> str:
    """A direct re-exec after the owner's Restart carries ONLY the transaction this
    Restart prepared — never another door's (an aborted update's) still-active one."""
    from ouroboros.delegate_recovery import PLANNED_RESTART_TRANSACTION_ENV, arm_active_planned_restart_transaction

    os.environ.pop(PLANNED_RESTART_TRANSACTION_ENV, None)
    own = _LAST_RETURN_TX["id"]
    if own and arm_active_planned_restart_transaction(DATA_DIR) == own:
        return own
    os.environ.pop(PLANNED_RESTART_TRANSACTION_ENV, None)
    return ""


def _stop_owned_daemon(label: str) -> None:
    """The attested owned-daemon stop a restart makes; never a veto.

    An unconfirmed stop was already disclosed by ``stop_outcome`` (critical log
    plus the ``process_stop_unconfirmed`` supervisor row, custody retained); a
    raising stop gets the same row here, as Panic records it. Either way the
    restart proceeds and the next generation attaches to a still-live daemon.
    """
    from ouroboros.claudexor_daemon import CUSTODY_PURPOSE, get_owned_daemon
    from ouroboros.utils import append_jsonl, utc_now_iso

    try:
        outcome = get_owned_daemon().stop_outcome()
    except Exception as exc:
        log.critical("%s: owned Claudexor stop raised %s; custody is unconfirmed", label, type(exc).__name__)
        try:
            append_jsonl(pathlib.Path(DATA_DIR) / "logs" / "supervisor.jsonl", {
                "ts": utc_now_iso(), "type": "process_stop_unconfirmed",
                "purpose": CUSTODY_PURPOSE, "reason": f"stop raised {type(exc).__name__}",
            })
        except Exception:
            log.critical("%s: failed to record unconfirmed daemon stop", label)
    else:
        if outcome == "unconfirmed":  # stop_outcome already disclosed the remainder
            log.critical("%s: owned Claudexor stop unconfirmed; custody retained, the "
                         "restart proceeds and the next generation attaches to the live daemon", label)


def _stop_owned_daemon_for_new_pin() -> None:
    """A planned restart replaces an off-pin daemon, but never downgrades a newer one.

    Runs in the server lifespan teardown of every requested restart (planned
    self-restart, managed update or rollback; the owner's manual Restart has
    already stopped the daemon, so nothing answers unless that stop was
    unconfirmed). It compares the pin the next generation selects —
    ``load_runtime_pin`` reads the checkout that already landed, not this
    process's cached pin — with the serving engine's handshake through the
    attach-only ``read_owned_gateway``. An unprovisioned home, an unpublished or
    unreadable pin and an unreachable, foreign or stopped daemon leave the
    existing handoff untouched: the next generation attaches. Except after the
    owner's Restart, a lower release pin keeps the authenticated newer daemon:
    its data root may already require that engine's serving floor. This
    preserves live availability, not cold-start compatibility after that daemon
    later exits. Other version or build changes make the same attested stop the
    manual Restart makes, so the next generation spawns the pinned engine; the
    delegated runs that daemon served end with it, and the next generation's
    startup sweep and resumed parents close them as absent (no invented spend).
    """
    from ouroboros.claudexor_daemon import owned_daemon_provisioned, read_owned_gateway
    from ouroboros.engine_pin_policy import engine_pin_transition

    if not owned_daemon_provisioned():
        return
    try:
        pin = _next_generation_pin()
        if pin is None:
            return
        with read_owned_gateway() as gateway:
            serving = (gateway.engine_version, gateway.engine_build_sha)
    except Exception as exc:
        log.info("Planned restart keeps the owned Claudexor daemon: the engine pin comparison is "
                 "unavailable (%s)", exc)
        return
    transition = engine_pin_transition(pin, serving, owner_restart=_OWNER_STOP_BEGAN.is_set())
    if transition["action"] == "keep" and transition["direction"] == "backward":
        log.warning("Planned restart keeps newer owned Claudexor %s: checkout pin %s would downgrade "
                    "its data root; cold start still requires a compatible pin", serving[0], pin.version)
        return
    if transition["action"] != "replace":
        return
    log.warning("Planned restart stops the owned Claudexor daemon: engine %s (%s) is serving while the "
                "checkout pins %s (%s); the next generation starts the pinned engine and the runs "
                "in flight end with this one", serving[0] or "unknown", serving[1][:12] or "unknown",
                pin.version, pin.build_sha[:12])
    _stop_owned_daemon("Planned restart")


def _next_generation_pin():
    """The engine pin the NEXT generation selects.

    Ordinarily the landed checkout's own tracked pin file. When a store switch intent is
    recorded, the restart exit's seam moves the checkout to the intent's target after this
    cleanup: that target's pin is the answer (``store_switch.next_engine_pin``). When this
    restart carries a bound body adoption whose switch changes that file, the next
    generation boots the candidate commit: its pin is read from that commit
    (``body_adoption.bound_candidate_file``), through the same validating parser.
    """
    from ouroboros import body_adoption, config, store_switch
    from ouroboros.claudexor_runtime import _PIN_FILENAME, load_runtime_pin

    intent = store_switch.read_switch_intent(DATA_DIR)
    if intent is not None:
        return store_switch.next_engine_pin(config.REPO_DIR, intent["target_sha"])
    raw = body_adoption.bound_candidate_file(DATA_DIR, "ouroboros/" + _PIN_FILENAME)
    return load_runtime_pin() if raw is None else load_runtime_pin(data=raw)


def _live_running_task_ids(ctx: Any) -> list:
    """Pooled tasks with a fresh heartbeat and registered native executions.

    Heartbeat staleness belongs to the generic supervisor queue, not to the
    planning-scout wait policy.  The latter intentionally waits until terminal
    state or its shared cutoff even when a scout heartbeat is stale.
    """
    from supervisor.queue import HEARTBEAT_STALE_SEC

    now = time.time()
    live = []
    for tid, meta in dict(ctx.RUNNING or {}).items():
        if not isinstance(meta, dict):
            continue
        try:
            hb = float(meta.get("last_heartbeat_at") or 0.0)
        except (TypeError, ValueError):
            hb = 0.0
        if hb and (now - hb) < HEARTBEAT_STALE_SEC:
            live.append(str(tid))
    from supervisor import queue
    from supervisor.active_activity import get_direct_activity_registry, is_parked_direct_entry

    with queue._queue_lock:
        live.extend(str(task["id"]) for task in queue.PENDING
                    if task.get("_launch_claim") and task.get("admitted_dispatch") == "possible")

    # A direct turn parked in a durable owner wait is carried by the restart's
    # owner-wait handoff, exactly as the update fence treats it (#2191).
    return list(dict.fromkeys(live + [
        entry.activity_id for entry in get_direct_activity_registry().actors()
        if not is_parked_direct_entry(entry)
    ]))


def _managed_update_pending_kwargs() -> dict:
    """Preserve queued work while a durable tx or its pre-tx quiesce owns restart."""
    try:
        from ouroboros.delegate_recovery import has_planned_restart_handoffs

        if (
            has_planned_restart_handoffs(DATA_DIR)
            and _restart_requested.is_set()
            and not _owner_restart_requested.is_set()
        ):
            return {"preserve_pending": True}
        from supervisor.update_merge import active_update_tx

        if active_update_tx():
            return {"preserve_pending": True}
        from supervisor.workers import repo_writer_admission_closed, worker_pool_admission_state

        gate = repo_writer_admission_closed()
        disabled = str(worker_pool_admission_state().get("disabled_reason") or "")
        if gate.startswith("managed_update:") or disabled == "managed_update":
            return {"preserve_pending": True}
        return {}
    except Exception:
        return {"preserve_pending": True}


def _safe_restart_serialized(safe_restart_fn, *, reason: str, unsynced_policy: str):
    """Serialize checkout/reset with update apply; only a landed update may restart."""
    from supervisor import git_ops
    from supervisor.update_merge import (
        acquire_update_lock,
        read_update_tx_strict,
        release_update_lock,
    )

    try:
        lock_fh = acquire_update_lock()
    except RuntimeError:
        return False, "Managed update is changing the checkout; restart was deferred."
    except OSError as exc:
        # No checkout or teardown ran. Errors from safe_restart_fn remain unknown effects.
        return False, f"Managed update lock could not be acquired; restart was deferred ({exc})."
    try:
        status, tx = read_update_tx_strict()
        if status == "corrupt":
            return False, "Managed update state is unreadable; restart was deferred."
        if status == "future":
            return False, "Managed update state was recorded by a newer version; restart was deferred."
        if status == "valid" and tx.get("stash_restore"):
            return False, (
                "Local changes are still being recovered. Quit and reopen the desktop app, "
                "or restart the server process for a web deployment. Restart was deferred "
                "to preserve the current files."
            )
        if status == "absent" and not git_ops._clear_update_intent():
            return False, (
                "An update intent marker with no update transaction could not be removed; "
                "restart was deferred rather than applying an orphaned update."
            )
        if status == "valid" and str(tx.get("phase") or "") not in _RESTARTABLE_UPDATE_PHASES:
            return False, "Managed update merge is still being resolved; restart was deferred."
        return safe_restart_fn(reason=reason, unsynced_policy=unsynced_policy)
    finally:
        release_update_lock(lock_fh)


def _noticed_bootstrap(safe_restart_fn):
    """Bootstrap's ``safe_restart``: a failure after it RAN (a refused checkout step, dependency
    sync, the import test) is also told in the owner's chat as the message's headline line; the
    whole message, Git's output included, stays in the returned value, the log and supervisor.jsonl.
    A reset blocked to protect local changes is told as kept files, not as a failure; a start that
    fell back to the recovery point is told too (the working branch keeps the code that failed).
    ``_safe_restart_serialized``'s deliberate deferrals never call it and stay silent."""
    def run(**kwargs):
        ok, msg = safe_restart_fn(**kwargs)
        fell_back = ok and str(msg).startswith("OK: fell back to")
        if not ok or fell_back:
            headline = (str(msg).splitlines() or ["no detail"])[0]
            kept = headline.partition(": ")[2]  # after "Failed checkout <branch>"
            text = (f"Startup fell back to the recovery point ({headline.rpartition(' ')[2]}): the working branch "
                    "did not pass its import test and keeps that code for repair" if fell_back
                    else f"Startup kept the current files: {kept}" if kept.startswith("Reset blocked (")
                    else f"Startup bootstrap failed: {headline}")
            try:
                from supervisor.message_bus import notification_chat_route, send_with_budget
                from supervisor.state import load_state

                owner_chat = notification_chat_route(load_state().get("owner_chat_id"))
                if owner_chat is not None:
                    send_with_budget(owner_chat, text, role="system", system_type="startup_notice")
            except Exception:
                log.warning("Bootstrap failure notice was not delivered", exc_info=True)
        return ok, msg
    return run


def _begin_task_teardown(restart_requested: bool) -> tuple[str, str]:
    """The planned restart and Quit start tearing tasks down: admission pauses first, so a launch or
    Resume already writing its marks hands over to the still-running pool (``pause_admission``);
    returns ``_shutdown_task_cleanup_args``."""
    from supervisor.workspace_preparation import pause_admission

    pause_admission()
    return _shutdown_task_cleanup_args(restart_requested)


def _shutdown_task_cleanup_args(restart_requested: bool) -> tuple[str, str]:
    """Return ``(terminal_status, result_reason)`` for tasks torn down by a
    graceful server shutdown.

    A graceful shutdown — a requested restart (exit 42) or an external
    stop/restart signal (SIGTERM/SIGINT) — is not a worker crash storm, so a
    still-running task is finalized as ``cancelled`` with an honest reason
    instead of the default crash-storm text the supervisor uses for real
    worker deaths; its callers pass ``stop_source="server_shutdown"``, the
    typed cause that makes it a technical interruption the owner may Continue.
    """
    if restart_requested:
        reason = (
            "Server restarted before this task finished; the task was "
            "interrupted by the restart, not a worker crash."
        )
    else:
        reason = (
            "Server shut down (external stop/restart signal) before this task "
            "finished; the task was interrupted, not a worker crash."
        )
    return "cancelled", reason


def _shutdown_supervisor_event_bus() -> None:
    try:
        from supervisor.workers import shutdown_event_q

        shutdown_event_q()
    except Exception:
        pass


# Composition-root bindings stay explicit; the restart lifecycle lives here.
def _restart_current_process(server, host: str, port: int) -> None:
    # Every direct restart reaches this seam, including an assisted update whose
    # native waits were already moved to PENDING before its resolver ran.
    try:
        from ouroboros.delegate_recovery import PLANNED_RESTART_TRANSACTION_ENV, arm_active_planned_restart_transaction
        from ouroboros.server_restart import _RESTARTABLE_UPDATE_PHASES
        from supervisor.update_merge import read_update_tx_strict

        from ouroboros.server_control import _restart_stop_requested
        if _restart_stop_requested:
            server.os.environ.pop(PLANNED_RESTART_TRANSACTION_ENV, None)
        elif server.DATA_DIR == DATA_DIR and _OWNER_STOP_BEGAN.is_set():
            from ouroboros.server_restart import arm_owner_restart_transaction

            arm_owner_restart_transaction()  # only the work THIS owner Restart named returns
        elif getattr(server, "_planned_delegate_restart_transaction_id", ""):
            own = server._planned_delegate_restart_transaction_id
            if arm_active_planned_restart_transaction(server.DATA_DIR) != own:
                server.os.environ.pop(PLANNED_RESTART_TRANSACTION_ENV, None)
        else:
            status, tx = read_update_tx_strict()
            if status == "valid" and tx.get("phase") in _RESTARTABLE_UPDATE_PHASES:
                arm_active_planned_restart_transaction(server.DATA_DIR)
    except Exception:
        server.log.warning("Direct restart transaction could not be armed; continuation remains unconfirmed", exc_info=True)
    server._restart_current_process_impl(
        host, port, repo_dir=server.REPO_DIR, log=server.log,
        owner_initiated=server._owner_restart_requested.is_set(),
    )


def _handle_restart_in_supervisor(server, evt: Dict[str, Any], ctx: Any) -> None:
    """Handle agent restart request: drain live tasks across loop ticks, then
    graceful shutdown + exit(42). Never sleeps on the dispatch thread."""
    from supervisor import queue

    request_id = str(evt.get("request_id") or "")
    if evt.get("phase") == "intent":
        if not request_id or not evt.get("task_id") or type(evt.get("task_attempt")) is not int or evt["task_attempt"] < 1:
            return
        with queue._queue_lock:
            if _AGENT_RESTART.get("request_id") != request_id:
                if agent_restart_pending():
                    _AGENT_RESTART_REPLY.clear()
                    _AGENT_RESTART_REPLY.update(request_id=request_id, status="refused", refusal="restart_already_pending")
                    queue.persist_queue_snapshot(reason="agent_restart_refused")
                    return
                _AGENT_RESTART.clear()
                _AGENT_RESTART.update({**evt, "status": "awaiting", "started": {}})
        _accept_agent_restart(ctx)
        return
    if request_id:
        with queue._queue_lock:
            if (evt.get("phase") != "ready" or _AGENT_RESTART.get("request_id") != request_id or not agent_restart_pending()
                    or evt.get("task_id") != _AGENT_RESTART.get("task_id")
                    or evt.get("task_attempt") != _AGENT_RESTART.get("task_attempt")):
                return  # a failed/refused requester never revives its own or another intent
            if _requester_failed(ctx):
                release_agent_restart(request_id, "requester_failed")
                return
            _AGENT_RESTART["ready_event"] = dict(evt)
            if _AGENT_RESTART.get("status") == "awaiting":
                return
            if _AGENT_RESTART.get("status") == "ready":
                return  # repeated delivery never restarts the drain clock
            if "evolution_transaction" not in evt:
                _AGENT_RESTART["status"] = "ready"
    elif agent_restart_pending():
        # Evolution's supervisor-owned completion keeps its own receipt check;
        # it may finish only the intention of that exact requester.
        if str(evt.get("task_id") or "") != _AGENT_RESTART.get("task_id"):
            return
        return _handle_restart_in_supervisor(server, {**evt, "phase": "ready",
            "request_id": _AGENT_RESTART["request_id"], "task_attempt": _AGENT_RESTART["task_attempt"]}, ctx)
    else:
        # Supervisor-owned evolution/legacy completion already crossed finalization.
        with queue._queue_lock:
            _AGENT_RESTART.clear()
            _AGENT_RESTART.update(request_id=server.uuid.uuid4().hex, status="ready",
                                  reason=evt.get("reason"), started=_started_restart_work())
    if "evolution_transaction" in evt:
        from supervisor.evolution_lifecycle import request_evolution_restart

        tx = evt["evolution_transaction"]
        outcome = (request_evolution_restart(ctx.DRIVE_ROOT, tx, log=server.log, request=evt)
                   if tx.get("cycle_outcome") == "waiting_for_restart" and tx.get("restart_required")
                   and not tx.get("restart_verified") else "evolution_restart_not_required")
        if outcome not in {"requested", "unconfirmed"}:
            release_agent_restart(request_id, outcome)
        return  # preparation/delivery stays executable until this host consumes the exact ready event
    st = ctx.load_state()
    if st.get("owner_chat_id"):
        ctx.send_with_budget(
            int(st["owner_chat_id"]),
            f"♻️ Restart requested by agent: {evt.get('reason')}",
            role="system", system_type="restart_notice")
    from ouroboros.config import get_restart_drain_max_sec

    max_wait = get_restart_drain_max_sec()
    live = server._live_running_task_ids(ctx) if max_wait > 0 else []
    if live:
        # Defer: re-checked each tick by _check_pending_restart_drain so the
        # loop keeps draining events (heartbeats advance, RUNNING shrinks).
        server._pending_restart.clear()
        server._pending_restart.update({
            "request_id": _AGENT_RESTART.get("request_id"),
            "reason": str(evt.get("reason") or "agent_restart_request"),
            "deadline": server.time.time() + min(max_wait, 1800),
            "evolution_restart": bool(evt.get("evolution_restart")),
        })
        if st.get("owner_chat_id"):
            ctx.send_with_budget(
                int(st["owner_chat_id"]),
                f"⏳ Restart drain: waiting up to {max_wait}s for running task(s) "
                f"{', '.join(sorted(live))} to finish.",
                role="system", system_type="restart_notice")
        return
    server._perform_supervisor_restart(
        ctx, restart_reason=str(evt.get("reason") or "agent_restart_request"),
        evolution_restart=bool(evt.get("evolution_restart")),
    )


def _check_pending_restart_drain(server, ctx: Any) -> bool:
    """Loop-tick hook: complete a deferred restart once tasks drain or the
    deadline passes (proceeds fail-closed). Returns True while STILL draining, so
    the loop can skip starting new work that the restart would immediately chop."""
    if agent_restart_pending():
        if terminal_retry := _AGENT_RESTART.get("evolution_terminal"):
            terminal_retry()
            if not agent_restart_pending():
                return False
        if _requester_failed(ctx):
            release_agent_restart(_AGENT_RESTART["request_id"], "requester_failed")
            if server._pending_restart.get("request_id") == _AGENT_RESTART["request_id"]:
                server._pending_restart.clear()
            return False
        if _AGENT_RESTART.get("status") == "awaiting":
            _accept_agent_restart(ctx)
        if _AGENT_RESTART.get("status") == "accepted" and _AGENT_RESTART.get("ready_event"):
            _handle_restart_in_supervisor(server, _AGENT_RESTART["ready_event"], ctx)
            return True
    if not server._pending_restart:
        return False
    live = server._live_running_task_ids(ctx)
    if live and server.time.time() < float(server._pending_restart.get("deadline") or 0.0):
        return True  # keep draining — events still flow each tick
    pending = dict(server._pending_restart)
    server._pending_restart.clear()
    server._perform_supervisor_restart(
        ctx, restart_reason=str(pending.get("reason") or "agent_restart_request"),
        evolution_restart=bool(pending.get("evolution_restart")),
    )
    # Still "quiescing" this tick: _perform_supervisor_restart sets up the exit
    # (or fail-closed pauses) and returns to the loop — the process exits on the
    # next `while not _restart_requested` check. Returning True keeps the caller
    # from starting new enqueue/assign work on this final pre-exit tick.
    return True


def _perform_supervisor_restart(server,
    ctx: Any, *, restart_reason: str = "agent_restart_request",
    evolution_restart: bool = False,
) -> None:
    """Graceful shutdown + exit(42) (the post-drain tail; never sleeps)."""
    st = ctx.load_state()
    marker = server.read_json_dict(
        server.pathlib.Path(ctx.DRIVE_ROOT) / "state" / "pending_restart_verify.json"
    ) or {}
    claim = (
        marker.get("evolution_claim")
        if evolution_restart and marker.get("reason") == restart_reason
        else {}
    )
    claim = claim if isinstance(claim, dict) else {}
    if evolution_restart and not claim:
        release_agent_restart(str(_AGENT_RESTART.get("request_id") or ""), "evolution_receipt_missing")
        if st.get("owner_chat_id"):
            ctx.send_with_budget(
                int(st["owner_chat_id"]),
                "🧬 Restart cancelled: the exact evolution restart receipt is missing.",
                role="system", system_type="restart_notice")
        return
    if claim:
        from supervisor.evolution_lifecycle import check_evolution_authority

        authority = check_evolution_authority(
            str(claim.get("campaign_id") or ""),
            str(claim.get("transaction_id") or ""),
            str(claim.get("task_id") or ""),
            commit_sha=str(claim.get("commit_sha") or ""),
        )
        if not authority.get("ok"):
            release_agent_restart(str(_AGENT_RESTART.get("request_id") or ""), "evolution_authority_changed")
            if st.get("owner_chat_id"):
                ctx.send_with_budget(
                    int(st["owner_chat_id"]),
                    "🧬 Restart cancelled: evolution authority changed "
                    f"({authority.get('reason') or 'unknown'}).",
                    role="system", system_type="restart_notice")
            return
        expected_sha = str(claim.get("commit_sha") or "")
        try:
            head_proc, status_proc = (server.subprocess.run(
                ["git", *args], cwd=str(ctx.REPO_DIR), check=False, capture_output=True, text=True,
            ) for args in (("rev-parse", "HEAD"), ("status", "--porcelain")))
            head = head_proc.stdout.strip() if head_proc.returncode == 0 else ""
            clean = status_proc.returncode == 0 and not status_proc.stdout.strip()
        except Exception:
            head = ""
            clean = False
        # A candidate commit is claimed BEFORE its switch: the serving checkout is then at the adoption's base.
        serving_sha = server.body_adoption.authorized_base(ctx.DRIVE_ROOT, expected_sha, restart_reason) or expected_sha
        if not expected_sha or head != serving_sha or not clean:
            release_agent_restart(str(_AGENT_RESTART.get("request_id") or ""), "evolution_checkout_changed")
            if st.get("owner_chat_id"):
                ctx.send_with_budget(
                    int(st["owner_chat_id"]),
                    "🧬 Restart cancelled: the live checkout no longer matches "
                    "the exact reviewed evolution commit.",
                    role="system", system_type="restart_notice")
            return
    ok, msg = server._safe_restart_serialized(
        server.body_adoption.bind_restart(ctx.safe_restart, ctx.DRIVE_ROOT, restart_reason),
        reason="agent_restart_request",
        unsynced_policy="rescue_and_block",
    )
    if not ok:
        release_agent_restart(str(_AGENT_RESTART.get("request_id") or ""), "restart_refused")
        try:
            from supervisor.evolution_lifecycle import disable_evolution_projection, pause_evolution_campaign

            disable_evolution_projection()
            pause_evolution_campaign(f"agent restart blocked to protect local changes: {msg}")
        except Exception:
            server.log.debug("Failed to pause evolution after blocked agent restart", exc_info=True)
        if st.get("owner_chat_id"):
            ctx.send_with_budget(int(st["owner_chat_id"]), f"⚠️ Restart skipped: {msg}", role="system", system_type="restart_notice")
        return
    cleanup_status, cleanup_reason = server._begin_task_teardown(restart_requested=True)
    server._planned_delegate_restart_transaction_id = ""
    planned_handoffs: set[str] = set()
    restart_transaction_id = server.uuid.uuid4().hex
    try:
        from ouroboros.delegate_recovery import prepare_planned_restart_handoffs
        from ouroboros.owner_wait import prepare_owner_wait_handoffs
        from supervisor.queue import name_queue_returns
        from supervisor.restart_retention import prepare_restart_returns
        from supervisor.workers import direct_chat_turns

        owner_waits = prepare_owner_wait_handoffs(ctx.DRIVE_ROOT, ctx.RUNNING, restart_transaction_id)
        planned_handoffs = prepare_planned_restart_handoffs(
            ctx.DRIVE_ROOT, ctx.RUNNING,
            restart_transaction_id=restart_transaction_id,
            additional_task_ids=owner_waits,
        )
        # The planned restart also returns active work and the runnable queue (#1563);
        # the off-loop schedule duty admits nothing after this copy.
        prepare_restart_returns(ctx.DRIVE_ROOT, ctx.RUNNING, name_queue_returns(ctx.PENDING, "planned_restart"),
                                transaction_id=restart_transaction_id,
                                direct=direct_chat_turns(), owner_wait_ids=owner_waits)
        server._planned_delegate_restart_transaction_id = restart_transaction_id
    except Exception:
        server.log.debug("Planned self-restart delegate handoff preparation failed", exc_info=True)
    try:
        restart_kill_kwargs = {"preserve_pending": True, **server._managed_update_pending_kwargs()}
        if planned_handoffs:
            server._planned_delegate_restart_transaction_id = restart_transaction_id
            restart_kill_kwargs["preserve_pending"] = True
        ctx.kill_workers(
            force=True,
            terminal_status=cleanup_status,
            result_reason=cleanup_reason,
            stop_source="server_shutdown",
            preserve_running_task_ids=planned_handoffs,
            retain_saved_work=True,
            **restart_kill_kwargs,
        )
        try:  # a field update: a stale snapshot never erases a control written meanwhile (#1307)
            ctx.update_state(lambda live: live.__setitem__("session_id", server.uuid.uuid4().hex))
        except Exception:
            server.log.warning("Restart session id not rotated: runtime state unavailable", exc_info=True)
        ctx.persist_queue_snapshot(reason="pre_restart_exit")
    except BaseException:
        from supervisor.queue import reopen_queue_returns

        reopen_queue_returns("planned_restart")  # no exit follows: this generation keeps admitting
        raise
    server._request_restart_exit()
