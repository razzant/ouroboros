"""Restart-adjacent helpers the composition root calls at shutdown time.

The live-task census the restart drain consults, the teardown arguments that
finalize interrupted tasks with an honest reason, the managed-update guard on
preserving queued work, the checkout/update serialization gate, the owner's manual
Restart operation, the planned restart's engine-pin daemon stop,
and the event bus shutdown live here. The planned restart transaction itself stays in
``server.py``: its planned-handoff transaction id joins the deferred drain record
to the performer that raises the exit signal.
"""

from __future__ import annotations

import os
import pathlib
import time
from typing import Any

from ouroboros.server_process import (
    DATA_DIR, _owner_restart_requested, _request_restart_exit, _restart_requested, log,
)

_RESTARTABLE_UPDATE_PHASES = frozenset({"pending_boot_smoke", "applying_replace"})
_LAST_RETURNING: set = set()  # this process's Restart: the ids its transaction returns
_LAST_RETURN_TX: dict = {"id": ""}  # ...and the one transaction that names them


def _perform_owner_restart(ctx: Any, reply=None) -> tuple[bool, str]:
    """Run the owner restart operation with an optional transport notice."""
    ok, restart_msg = _safe_restart_serialized(
        ctx.safe_restart,
        reason="owner_restart",
        unsynced_policy="rescue_and_reset",
    )
    if not ok:
        return False, restart_msg
    state_dir = DATA_DIR / "state"
    owner_restart_flag = state_dir / "owner_restart_no_resume.flag"
    stable_skip_flag = state_dir / "panic_stop.flag"
    # A Panic flag still owed its durable disabled controls (#1307) is never replaced
    # or removed here: boot consumes it only after those controls are saved.
    try:
        panic_kept = stable_skip_flag.read_text(encoding="utf-8").strip() != "owner_restart_no_resume"
    except FileNotFoundError:
        panic_kept = False
    except Exception:
        panic_kept = True  # unreadable: unknown, so kept
    try:
        from supervisor.queue_schedules import schedule_transaction
        with schedule_transaction(DATA_DIR):
            state_dir.mkdir(parents=True, exist_ok=True)
            owner_restart_flag.write_text("owner_restart", encoding="utf-8")
            # Pair owner flag with panic_stop for stable-build auto-resume compatibility.
            if not panic_kept:
                stable_skip_flag.write_text("owner_restart_no_resume", encoding="utf-8")
            try:
                from supervisor.followup_policy import record_restart
                record_restart(DATA_DIR, new=True)
            except Exception:
                # The marker keeps final admission closed; boot must persist
                # the restriction before consuming it. Restart still proceeds.
                log.exception("Restart follow-up restriction awaits boot persistence")
    except Exception:
        owner_restart_flag.unlink(missing_ok=True)
        if not panic_kept:
            stable_skip_flag.unlink(missing_ok=True)
        log.warning("Failed to write owner restart no-resume flag", exc_info=True)
        return False, "could not write restart state."
    # Everything reversible is behind us (checkout landed, no-resume
    # intent durable): from here the restart always follows, and every
    # unconfirmed stop is a critical diagnostic, never a deferral.
    stopped_task_ids = _stop_owned_work(ctx)
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

    Runs AFTER the checkout gate and the durable no-resume flags, so nothing
    here can veto: an unconfirmed step is a critical diagnostic with custody
    retained, and the next generation's startup custody sweep reconciles the
    remainder. First prepare the retained-return transaction: native saved
    work returns only after its fresh acknowledgment, while prior holds stay.
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
    would start a dead daemon — which is what the two flags above guarantee.

    Returns the owned live task ids it addressed — captured ONCE, before the
    stop makes them unreadable — so the caller can tell the owner what was
    stopped instead of claiming a task was stopped when nothing was running.
    """
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.claudexor_daemon import read_owned_gateway
    from ouroboros.delegate_custody import reconcile_orphaned_runs
    from ouroboros.owned_shutdown import begin_owned_stop

    begin_owned_stop(DATA_DIR)  # the grace starts here; every pending stop is recorded before any wait
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
    try:
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
    import uuid

    try:
        from ouroboros.owner_wait import prepare_owner_wait_handoffs
        from supervisor.restart_retention import prepare_restart_returns
        from supervisor.workers import direct_chat_turns

        _LAST_RETURN_TX["id"] = ""
        transaction_id = uuid.uuid4().hex
        owner_waits = prepare_owner_wait_handoffs(DATA_DIR, _question_waits(ctx.RUNNING), transaction_id)
        returning = prepare_restart_returns(DATA_DIR, dict(ctx.RUNNING or {}), list(ctx.PENDING or []),
                                            transaction_id=transaction_id, direct=direct_chat_turns(),
                                            owner_wait_ids=owner_waits)
        _LAST_RETURN_TX["id"] = transaction_id
        return returning | owner_waits, owner_waits
    except Exception:
        log.warning("Owner restart: saved work returns were not prepared; the next boot holds them", exc_info=True)
        return set(), set()


def _question_waits(running: Any) -> dict:
    """Only question/review waits ride this Restart's transaction; a warm sleep or a warm
    owner-Pause park keeps its own retention (an exact pause only Resume releases, §9)."""
    from ouroboros.task_results import load_task_result

    chosen = {}
    for task_id, meta in dict(running or {}).items():
        try:
            wait = (load_task_result(DATA_DIR, task_id, strict=True) or {}).get("owner_wait") or {}
        except Exception:
            continue  # unreadable: the ordinary stop path keeps whatever custody it has
        if wait.get("reason") in {"owner", "review"}:
            chosen[task_id] = meta
    return chosen


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
    """A planned restart ends the owned daemon only when the landed checkout pins another engine.

    Runs in the server lifespan teardown of every requested restart (planned
    self-restart, managed update or rollback; the owner's manual Restart has
    already stopped the daemon, so nothing answers). It compares the pin the
    next generation selects — ``load_runtime_pin`` reads the checkout that
    already landed, not this process's cached pin — with the serving engine's
    handshake through the attach-only ``read_owned_gateway``. An unprovisioned
    home, an unpublished or unreadable pin and an unreachable, foreign or
    stopped daemon leave the existing handoff untouched: the next generation
    attaches. A different version or build SHA makes the same attested stop the
    manual Restart makes, so the next generation spawns the pinned engine; the
    delegated runs that daemon served end with it, and the next generation's
    startup sweep and resumed parents close them as absent (no invented spend).
    """
    from ouroboros.claudexor_daemon import owned_daemon_provisioned, read_owned_gateway

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
    if serving == (pin.version, pin.build_sha):
        return
    log.warning("Planned restart stops the owned Claudexor daemon: engine %s (%s) is serving while the "
                "checkout pins %s (%s); the next generation starts the pinned engine and the runs "
                "in flight end with this one", serving[0] or "unknown", serving[1][:12] or "unknown",
                pin.version, pin.build_sha[:12])
    _stop_owned_daemon("Planned restart")


def _next_generation_pin():
    """The engine pin the NEXT generation selects.

    Ordinarily the landed checkout's own tracked pin file. When this restart carries
    a bound body adoption whose switch changes that file, the next generation boots
    the candidate commit: its pin is read from that commit
    (``body_adoption.bound_candidate_file``), through the same validating parser.
    """
    import tempfile

    from ouroboros import body_adoption
    from ouroboros.claudexor_runtime import _PIN_FILENAME, load_runtime_pin

    raw = body_adoption.bound_candidate_file(DATA_DIR, "ouroboros/" + _PIN_FILENAME)
    if raw is None:
        return load_runtime_pin()
    with tempfile.TemporaryDirectory(prefix="ouroboros-next-pin-") as scratch:
        pin_file = pathlib.Path(scratch) / _PIN_FILENAME
        pin_file.write_bytes(raw)
        return load_runtime_pin(pin_file)


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
    from supervisor.active_activity import get_direct_activity_registry

    return list(dict.fromkeys(live + [
        row["activity_id"] for row in get_direct_activity_registry().snapshot()
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
