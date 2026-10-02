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

import pathlib
import time
from typing import Any

from ouroboros.server_process import (
    DATA_DIR, _owner_restart_requested, _request_restart_exit, _restart_requested, log,
)

_RESTARTABLE_UPDATE_PHASES = frozenset({"pending_boot_smoke", "applying_replace"})


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
        state_dir.mkdir(parents=True, exist_ok=True)
        owner_restart_flag.write_text("owner_restart", encoding="utf-8")
        if not panic_kept:
            # Pair owner flag with panic_stop for stable-build auto-resume compatibility.
            stable_skip_flag.write_text("owner_restart_no_resume", encoding="utf-8")
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
            # named a task that was never running.
            reply(
                "Stopping active task. Restart confirmed. New settings apply to the next message."
                if stopped_task_ids else "Restart confirmed. New settings apply to the next message.",
                "completed",
            )
    except Exception:
        log.warning("Failed to send owner restart stop notice; continuing restart", exc_info=True)
    _request_restart_exit(owner=True)
    return True, ""


def _owned_live_task_ids(ctx: Any) -> list:
    """Every id this generation's cancel intent can address: pooled tasks,
    in-process direct/ephemeral activities and running post-task synthesis."""
    from ouroboros.post_task_checkpoint import POST_TASK_SYNTHESIS_INFLIGHT, POST_TASK_SYNTHESIS_LOCK
    from supervisor.active_activity import get_direct_activity_registry

    task_ids = list(dict(ctx.RUNNING or {}))
    task_ids.extend(str(row.get("activity_id") or "") for row in get_direct_activity_registry().snapshot())
    root = str(pathlib.Path(DATA_DIR).resolve(strict=False))
    with POST_TASK_SYNTHESIS_LOCK:
        task_ids.extend(task_id for (path, task_id) in POST_TASK_SYNTHESIS_INFLIGHT if path == root)
    return [tid for tid in dict.fromkeys(task_ids) if tid]


def _stop_owned_work(ctx: Any) -> list:
    """The owner's manual Restart: stop what this generation owns, then let it re-exec.

    Runs AFTER the checkout gate and the durable no-resume flags, so nothing
    here can veto: an unconfirmed step is a critical diagnostic with custody
    retained, and the next generation's startup custody sweep reconciles the
    remainder (owner restart is a no-resume cause; nothing is adopted). In
    order: one durable cancel intent per owned live id, ``kill_workers`` with
    Panic's ``reconcile_delegate_custody=False``, delegated-run cancellation
    through the public owner-gone seam over the attach-only gateway, and the
    attested owned-daemon stop exactly as Panic makes it. Between the cancel
    intents and that stop nothing may call ``ensure_owned_gateway`` — it would
    start a dead daemon — which is what the two flags above guarantee.

    Returns the owned live task ids it addressed — captured ONCE, before the
    stop makes them unreadable — so the caller can tell the owner what was
    stopped instead of claiming a task was stopped when nothing was running.
    """
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.claudexor_daemon import read_owned_gateway
    from ouroboros.delegate_custody import reconcile_orphaned_runs

    stopped = _owned_live_task_ids(ctx)
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
            reconcile_delegate_custody=False, **_managed_update_pending_kwargs(),
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
    return stopped


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
    from ouroboros.claudexor_runtime import load_runtime_pin

    if not owned_daemon_provisioned():
        return
    try:
        pin = load_runtime_pin()
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
    worker deaths.
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
