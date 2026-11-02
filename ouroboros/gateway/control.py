"""Control, update, and evolution HTTP endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable, Dict

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.background import BackgroundTask

from ouroboros import get_version
from ouroboros.gateway._helpers import json_error, json_exception, request_drive_root, request_json_or, request_repo_dir
from ouroboros.gateway.ws import broadcast_ws_sync
from ouroboros.gateway import update_progress
from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)

_RECENT_VISIBLE_COMMANDS: Dict[str, float] = {}
_VISIBLE_COMMAND_DEDUPE_SEC = 5.0


def _request_restart(request: Request) -> bool:
    # Every caller here is an OWNER action through the control surface (Reset All
    # Data, Rollback, Apply Update) — never the agent's own restart tool, which
    # goes through the supervisor. Saying so lets the re-exec re-read the runtime
    # mode from settings instead of re-pinning the inherited boot baseline. The
    # bool tells the caller whether a restart callback existed to accept it.
    callback = getattr(getattr(request.app, "state", None), "request_restart", None)
    if callable(callback):
        callback(owner=True)
        return True
    return False


def _runtime_branch_defaults(request: Request) -> tuple[str, str]:
    callback = getattr(getattr(request.app, "state", None), "runtime_branch_defaults", None)
    if callable(callback):
        return callback()
    return "ouroboros", "ouroboros-stable"


def _managed_update_payload(*, fetch: bool, include_tags: bool) -> dict[str, Any]:
    from supervisor.git_ops import compute_managed_update_status, git_capture
    from supervisor.update_merge import active_update_tx

    status = compute_managed_update_status(fetch=fetch)
    # The update description rides the same payload: a FETCHING check records only
    # facts (never prose, never on the passive read), with the running source separate
    # from the mutable checkout HEAD that binds apply/preflight.
    letter = None
    try:
        from ouroboros import update_letter as _letter

        status = _letter.runtime_status(status)
        if fetch:
            _letter.refresh_after_check(status)
        letter = _letter.project_letter_for_panel(status)
    except Exception:
        log.debug("update letter projection failed", exc_info=True)
    # Additive minimal public projection of an active managed-update
    # transaction, so a re-opened panel can say "resolution in progress"
    # instead of silently reading as ordinary state (a second apply 409s).
    try:
        tx = active_update_tx()
    except Exception:
        tx = None  # Unread is not absent for the retained attempt.
    update_tx = (
        {
            "active": True,
            "phase": str(tx.get("phase") or ""),
            "task_id": str(tx.get("task_id") or ""),
            "restart_required": bool(tx.get("restart_required")),
            **({"local_work_recovery": True} if tx.get("stash_restore")
               or tx.get("gate_blocked_reason") == "rollback_restart_pending" else {}),
        }
        if tx
        else {"active": False}
    )
    latest_version = ""
    latest_sha = status.get("latest_sha") or ""
    if latest_sha:
        rc, version_text, _ = git_capture(["git", "show", f"{latest_sha}:VERSION"])
        if rc == 0:
            latest_version = version_text.strip()
    official_tags = []
    if include_tags:
        from supervisor.git_ops import list_official_update_tags

        official_tags = list_official_update_tags()
    return {
        "current_version": get_version(),
        "latest_version": latest_version,
        "official_tags": official_tags,
        "update_tx": update_tx,
        "update_progress": update_progress.snapshot(tx),
        "letter": letter,
        **status,
    }


def _acquire_repo_mutation_lock() -> tuple[Any, JSONResponse | None]:
    """Serialize owner-triggered repo/reset mutations with managed updates."""
    from supervisor.update_merge import (
        acquire_update_lock,
        active_update_tx,
        release_update_lock,
    )

    try:
        lock_fh = acquire_update_lock()
    except RuntimeError:
        return None, json_error(
            "Another update or recovery operation is already changing the checkout.",
            409,
        )
    if active_update_tx():
        release_update_lock(lock_fh)
        return None, json_error(
            "A managed update transaction is still active; finish or recover it first.",
            409,
        )
    return lock_fh, None


def _release_repo_mutation_lock(lock_fh: Any) -> None:
    from supervisor.update_merge import release_update_lock

    if lock_fh is not None:
        release_update_lock(lock_fh)


async def api_reset(request: Request) -> JSONResponse:
    """Reset all runtime data (state, memory, logs, settings) but keep repo."""
    import shutil

    data_dir = request_drive_root(request)
    lock_fh, lock_error = _acquire_repo_mutation_lock()
    if lock_error is not None:
        return lock_error
    try:
        from supervisor.message_bus import try_get_bridge

        bridge = try_get_bridge()
        if bridge is not None:
            bridge.panic.invalidate_owner()
        from ouroboros.launch_roots import durable_basis
        from supervisor.state_initialization import mark_pending, supervisor_evidence

        basis, _why = durable_basis(data_dir, evidence=supervisor_evidence)  # read before state/ goes: Reset keeps it
        deleted = []
        # Keep synchronization files until restart. Removing the directory that
        # contains the held managed-update lock would let a second updater enter.
        for subdir in ("state", "memory", "logs", "archive", "task_results", "uploads"):
            target = data_dir / subdir
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
                deleted.append(subdir)
        settings_file = data_dir / "settings.json"
        if settings_file.exists():
            settings_file.unlink()
            deleted.append("settings.json")
        # The owner's explicit fresh start (#1307): boot initializes a new state from
        # this pending witness instead of reading the wiped root as a lost one; the work
        # folders it never deletes keep their basis (#1239).
        mark_pending(data_dir, origin="owner_reset", work_root_basis=basis)
        _request_restart(request)
        return JSONResponse({"status": "ok", "deleted": deleted, "restarting": True})
    except Exception as exc:
        return json_exception(exc)
    finally:
        _release_repo_mutation_lock(lock_fh)


async def api_command(request: Request) -> JSONResponse:
    try:
        body = await request.json()
        cmd = body.get("cmd", "")
        if str(cmd).strip().lower().startswith("/panic"):
            from ouroboros.panic_source import source_evidence
            surface = body.get("client_surface")
            kind = "web_ui_http" if isinstance(surface, dict) else "other_http"
            panic_evidence = source_evidence(request, kind, client_surface=surface)

        if not isinstance(body.get("reply_to", ""), str):
            return json_error("reply_to must be an address string", 400)
        if cmd:
            from supervisor.message_bus import get_bridge, log_chat

            try:
                bridge = get_bridge()
            except AssertionError:
                bridge = None
            if str(cmd).strip().lower().startswith("/panic"):
                from ouroboros.server_control import request_panic

                if request_panic(bridge, cmd, evidence=panic_evidence,
                                 startup_owner_command=getattr(request.app.state, "startup_owner_command", None)):
                    return JSONResponse({"status": "ok"})
                return json_error("Panic emergency owner is unavailable.", 503)
            visible_text = str(body.get("visible_text") or "").strip()
            task_constraint = body.get("task_constraint") if isinstance(body.get("task_constraint"), dict) else None
            visible_task_id = str(body.get("visible_task_id") or "").strip()
            if visible_task_id:
                now = time.monotonic()
                expired = [
                    key for key, ts in _RECENT_VISIBLE_COMMANDS.items()
                    if now - ts > _VISIBLE_COMMAND_DEDUPE_SEC
                ]
                for key in expired:
                    _RECENT_VISIBLE_COMMANDS.pop(key, None)
                if visible_task_id in _RECENT_VISIBLE_COMMANDS:
                    return JSONResponse({"ok": True, "deduped": True, "task_id": visible_task_id})
            send_kwargs: dict[str, Any] = {"broadcast": False, "suppress_chat_log": bool(visible_text)}
            if task_constraint:
                send_kwargs["task_constraint"] = task_constraint
            # Owner Surface Fact: messages through this endpoint (CLI `chat send`,
            # SPA heal/repair posts) would otherwise masquerade as ordinary web
            # frames. The honest stamp names the ENDPOINT — the host cannot know
            # the true caller here (disclosed non-goal).
            send_kwargs["task_metadata"] = {"client_surface": {"channel": "api_command"}, "reply_to": body.get("reply_to", "")}
            # Publication precedes readiness. Bind the independent owner, keeping
            # this transport's metadata if the supervisor becomes ready meanwhile.
            if bridge is None or str(cmd).strip().lower() == "/restart":
                callback = getattr(request.app.state, "startup_owner_command", None)
                action = callback(cmd, send_kwargs=send_kwargs) if callable(callback) else None
                if action is not None:
                    return JSONResponse({"status": "ok"}, background=BackgroundTask(action))
            if bridge is None:
                return json_error("Complete provider setup before sending this command.", 409)
            bridge.ui_send(cmd, **send_kwargs)
            if visible_task_id:
                _RECENT_VISIBLE_COMMANDS[visible_task_id] = time.monotonic()
            if visible_text:
                # X3: no invented ids. `visible_task_id` is a caller-supplied
                # UI correlation key; when the caller has none there is no task
                # id yet (`ui_send` returns nothing — the router mints the id at
                # promotion), and the old bare "skill_repair" literal was a
                # fabricated id persisted into the durable chat log. The typed
                # truth is an empty id plus the pending marker.
                task_id = visible_task_id
                ts = utc_now_iso()
                payload = {
                    "type": "chat",
                    "role": "system",
                    "content": visible_text,
                    "ts": ts,
                    "source": "skill_repair",
                    "system_type": "skill_repair",
                    "task_id": task_id,
                }
                if not task_id:
                    payload["task_id_pending"] = True
                broadcast_ws_sync(payload)
                log_chat(
                    "system",
                    0,
                    0,
                    visible_text,
                    ts=ts,
                    source="skill_repair",
                    task_id=task_id,
                )
        return JSONResponse({"status": "ok"})
    except Exception as exc:
        return json_exception(exc, 400)


async def api_git_log(_request: Request) -> JSONResponse:
    """Return recent commits, tags, and current branch/sha."""
    try:
        from supervisor.git_ops import git_capture, list_commits, list_versions

        commits = list_commits(max_count=30)
        tags = list_versions(max_count=20)
        rc, branch, _ = git_capture(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        rc2, sha, _ = git_capture(["git", "rev-parse", "--short", "HEAD"])
        return JSONResponse({
            "commits": commits,
            "tags": tags,
            "branch": branch.strip() if rc == 0 else "unknown",
            "sha": sha.strip() if rc2 == 0 else "",
        })
    except Exception as exc:
        return json_exception(exc)


_ROLLBACK_REASONS = {"ui_rollback": "ui_rollback", "vcs_rollback": "agent_rollback_tool"}


def rollback_through_door(target: str, *, door: str, request_restart: Callable[[], bool]) -> tuple[int, dict]:
    """The one Rollback procedure (the UI button; the agent's ``vcs_rollback``, P4-S): lock,
    resolve, stop every writer (other tasks are interrupted at once), then either record the
    authority switch for the restart exit (a target without the store) or restore the
    checkout now; the restart follows either way, unless Git refused the restore's reset at its
    index lock (409 ``restore_not_applied``: no restart; the aborted-update path decides readmission).
    Returns the HTTP status and payload."""
    from supervisor.git_ops import DRIVE_ROOT, REPO_DIR, git_capture, rollback_to_version
    from supervisor.git_ops_reset import IndexLockRefusal, refusal_parts
    from ouroboros.store_switch import switch_or_reset

    reason = _ROLLBACK_REASONS[door]
    refusal = _index_lock_precondition()
    if refusal is not None:
        return refusal.status_code, json.loads(refusal.body)
    lock_fh, lock_error = _acquire_repo_mutation_lock()
    if lock_error is not None:
        return lock_error.status_code, json.loads(lock_error.body)
    try:
        rc, target_sha, error = git_capture(["git", "rev-parse", "--verify", f"{target}^{{commit}}"])
        if rc != 0:
            return 400, {"error": error or f"cannot resolve {target}"}
        blockers = _quiesce_repo_writers("manual_rollback")
        if blockers:
            fence = _fence_failure(blockers)
            return fence.status_code, json.loads(fence.body)
        try:
            ok, msg, deferred = switch_or_reset(
                DRIVE_ROOT, REPO_DIR, target_sha=target_sha.strip(), target_ref=target, reason=reason, door=door,
                reset=lambda: rollback_to_version(target_sha, reason=reason))
        except IndexLockRefusal as exc:
            if not exc.before_tree_effects:
                raise
            # Git refused the restore's only tree command at the held lock, so that reset changed nothing
            # (the keep branch and rescue made before it stay). No restart the same lock would refuse: the
            # aborted-update readmission runs, keeping the gate shut while its own guards still hold.
            _respawn_workers_after_failed_update()
            return 409, {"error": f"the restore was not applied: Git refused its reset: {refusal_parts(exc)[1]}",
                         "reason": "restore_not_applied"}
        if not ok:
            return 500, {"error": msg, "restart_required": True}
        try:
            restarting = request_restart()
        except Exception:
            log.warning("rollback restart request failed", exc_info=True)
            restarting = False
        return 200, {"status": "ok" if restarting else "restart_required", "message": msg,
                     "restarting": restarting, "switch_deferred": deferred}
    finally:
        _release_repo_mutation_lock(lock_fh)


def _git_rollback_fenced(request: Request, target: str) -> JSONResponse:
    """Run the complete restore transaction off the gateway event loop."""
    try:
        status, payload = rollback_through_door(target, door="ui_rollback", request_restart=lambda: _request_restart(request))
        return JSONResponse(payload, status_code=status)
    except Exception as exc:
        return json_exception(exc)


async def api_git_rollback(request: Request) -> JSONResponse:
    """Roll back to a specific commit or tag, then restart."""
    try:
        body = await request.json()
        target = body.get("target", "").strip()
    except Exception as exc:
        return json_exception(exc)
    if not target:
        return json_error("missing target", 400)
    return await asyncio.to_thread(_git_rollback_fenced, request, target)


async def api_git_promote(request: Request) -> JSONResponse:
    """Promote the current dev branch to the runtime's stable branch."""
    try:
        lock_fh, lock_error = _acquire_repo_mutation_lock()
        if lock_error is not None:
            return lock_error
        try:
            from supervisor.git_ops import promote_branch_exact

            branch_dev, branch_stable = _runtime_branch_defaults(request)
            ok, result = promote_branch_exact(
                branch_dev, branch_stable, push_remote=False
            )
            if not ok:
                return json_error(str(result.get("error") or "promotion failed"), 400)
            return JSONResponse({
                "status": "ok",
                "sha": result["sha"],
                "message": f"{branch_stable} updated to {result['sha'][:8]}",
            })
        finally:
            _release_repo_mutation_lock(lock_fh)
    except Exception as exc:
        return json_exception(exc)


async def api_update_status(_request: Request) -> JSONResponse:
    """Return passive managed-update status without fetching."""
    try:
        return JSONResponse(await asyncio.to_thread(
            _managed_update_payload, fetch=False, include_tags=False,
        ))
    except Exception as exc:
        return json_exception(exc)


async def api_update_check(_request: Request) -> JSONResponse:
    """Fetch the managed remote and return fresh update status."""
    try:
        payload = await asyncio.to_thread(
            _managed_update_payload,
            fetch=True,
            include_tags=True,
        )
        if payload.get("check_ok") is True:
            # Availability may have changed for every client; the retained attempt has not (#1818).
            update_progress.notify_changed()
        return JSONResponse(payload)
    except Exception as exc:
        return json_exception(exc)


def _respawn_workers_after_failed_update() -> None:
    """Revive workers when an update aborts after they were stopped (no restart follows)."""
    try:
        from supervisor.workers import ensure_worker_pool_started, open_repo_writer_admission_after_update_abort

        if open_repo_writer_admission_after_update_abort():
            ensure_worker_pool_started(allow_disabled_restart=True)
    except Exception:
        log.warning("update_apply: failed to respawn workers after aborted update", exc_info=True)


async def api_update_preflight(_request: Request) -> JSONResponse:
    """Plan the managed update as a REAL 3-way merge (P2). Does NOT touch the live
    worktree/branch/index (it fetches + merges in an isolated temp worktree), so the UI
    can present the right staged choice (auto / assisted / manual)."""
    try:
        from supervisor.update_merge import plan_managed_update_merge
        from ouroboros.update_impact import update_effects

        plan = await asyncio.to_thread(plan_managed_update_merge, fetch=True)
        effects = await asyncio.to_thread(update_effects, plan)
        return JSONResponse({"merge_plan": plan, "effects": effects, **await _preflight_description(plan)})
    except Exception as exc:
        return json_exception(exc)


async def _preflight_description(plan: dict) -> dict:
    """Prose and qualification of exactly this plan's range; unavailable means unknown, never a refusal."""
    from ouroboros.update_qualification import preflight_description

    try:
        return await asyncio.to_thread(preflight_description, plan)
    except Exception:
        log.warning("update preflight: range description unavailable", exc_info=True)
        return {"letter": None, "qualification": None}


async def api_update_description(request: Request) -> JSONResponse:
    """The owner's Write/Rewrite: ONE ordinary Main task describes the named range.

    The range must be the one the last check offered or the saved description's own; a
    request whose task has not ended answers 409 with that task. The task is admitted with
    its scheduled receipt like any host-started root, and runs on the ordinary Main route.
    """
    body = await request_json_or(request, {})
    body = body if isinstance(body, dict) else {}
    mode = str(body.get("mode") or "")
    base_sha, target_sha = (str(body.get(name) or "").strip().lower() for name in ("base_sha", "target_sha"))
    if mode not in ("write", "rewrite") or not base_sha or not target_sha:
        return json_error("mode (write or rewrite), base_sha and target_sha are required.", 400)
    try:
        from ouroboros.update_letter import RecordBusy, request_description

        try:
            outcome = await asyncio.to_thread(request_description, mode, base_sha, target_sha,
                                              admit=_admit_description_task)
        except RecordBusy:
            return json_error("The update description record is busy; nothing was requested.", 503)
        if outcome["outcome"] == "in_flight":
            return json_error("A description is already being written.", 409,
                              task_id=outcome["task_id"], task_status=outcome["status"])
        if outcome["outcome"] == "range_not_offered":
            return json_error("This range is not the offered update or the saved description's range; "
                              "check for updates first.", 409, legal_ranges=outcome["legal_ranges"])
        if outcome["outcome"] == "refused":
            return json_error(f"The writing task was not accepted ({outcome['reason']}).", 503,
                              reason=outcome["reason"])
        update_progress.notify_changed()
        return JSONResponse({"status": "queued", "task_id": outcome["task_id"], "key": outcome["key"]})
    except Exception as exc:
        return json_exception(exc)


def _admit_description_task(task_id: str, text: str, metadata: Dict[str, Any]) -> str:
    """Admit the owner's description request as an ordinary root task in the owner chat;
    returns ``""`` or the refusal reason. Nothing else is started or stopped. Typed as a root
    like every host-started one, so a host retry stays this request's own attempt. The press
    is the web panel's: while the binding is known empty (no owner message yet) the task is
    Main's, where the panel's door opens it; a bound or unreadable binding is used as read."""
    from ouroboros.contracts.chat_id_policy import WEB_UI_CHAT_ID
    from supervisor.queue import enqueue_with_admission_receipt, persist_queue_snapshot
    from supervisor.state import control_value, load_state

    st = load_state()
    unbound = control_value(st, "owner_chat_id") == (True, None)
    admitted = enqueue_with_admission_receipt({
        "id": task_id, "type": "task", "chat_id": WEB_UI_CHAT_ID if unbound else int(st.get("owner_chat_id") or 0),
        "root_task_id": task_id, "delegation_role": "root",
        "text": text, "metadata": metadata, "_require_worker_pool": True, "_require_unique_task_id": True,
    })
    if not isinstance(admitted, dict):
        return "refused"
    if admitted.get("_admission_blocked"):
        return str(admitted.get("_worker_pool_disabled_reason") or admitted["_admission_blocked"])
    persist_queue_snapshot(reason="update_description_requested")
    return ""


_KNOWN_UPDATE_PLAN_KINDS = frozenset({"clean", "conflicting"})
_UPDATE_STRATEGIES = frozenset({"auto_merge", "assisted", "manual", "replace"})


def _plan_is_clean(plan: dict) -> bool:
    """True for a complete deterministic Git plan with no semantic conflict."""
    return (
        str(plan.get("kind") or "") == "clean"
        and type(plan.get("local_dirty_count")) is int
        and plan.get("local_dirty_count") >= 0
        and bool(str(plan.get("merge_commit") or ""))
        and not plan.get("code_conflict_paths")
        and not plan.get("doc_conflict_paths")
    )


def _pins_match(plan: dict, base_sha: str, target_sha: str) -> bool:
    return bool(
        base_sha
        and target_sha
        and str(plan.get("base_sha") or "") == base_sha
        and str(plan.get("target_sha") or "") == target_sha
    )


def _index_lock_precondition() -> JSONResponse | None:
    from supervisor.git_ops_reset import observe_index_lock

    try:
        fact = observe_index_lock()
    except (OSError, RuntimeError) as exc:
        return JSONResponse({
            "error": f"Not applied: could not check Git's index lock: {exc}. No work was stopped.",
            "reason": "index_lock_check_failed", "outcome": "not_applied",
        }, status_code=409)
    if fact["exists"]:
        return JSONResponse({
            "error": f"Not applied: Git index lock exists at {fact['path']}. No work was stopped.",
            "reason": "index_lock_present", "outcome": "not_applied", "index_lock": fact,
        }, status_code=409)
    return None


def _quiesce_repo_writers(reason: str) -> list[str]:
    """Stop the repository writers with admission paused: a launch or Resume already writing its marks
    hands over to the still-running pool first, and nothing new starts until the pool is stopped
    (``workspace_preparation.pause_admission``); admission reopens for whatever follows."""
    from supervisor.workspace_preparation import pause_admission, resume_admission

    pause_admission()
    try:
        return _stop_repo_writers(reason)
    finally:
        resume_admission()


def _stop_repo_writers(reason: str) -> list[str]:
    """Close new writers, drain in-process turns, then prove the pool stopped."""
    from supervisor.git_ops import DRIVE_ROOT
    from supervisor.workers import (
        close_repo_writer_admission,
        drain_repo_writers,
        kill_workers_for_update,
        open_repo_writer_admission,
    )

    close_repo_writer_admission(f"managed_update:{reason}")
    update_progress.advance("draining_writers")
    blocked = drain_repo_writers()
    if blocked:
        open_repo_writer_admission()
        return [f"active:{label}" for label in blocked]
    from supervisor import workers as worker_state
    from supervisor.restart_retention import capture_update_returns

    try:
        with worker_state._queue_lock:
            update_returns = capture_update_returns(DRIVE_ROOT, worker_state.RUNNING, worker_state.WORKERS,
                                                   return_ids=())
    except Exception as exc:
        return [f"saved_work_handoff:{type(exc).__name__}: {exc}"]
    preserve_running_task_ids: set[str] = set()
    return_ids: set[str] = set()
    if reason != "manual_rollback":
        try:
            import uuid

            from ouroboros.delegate_recovery import prepare_planned_restart_handoffs
            from ouroboros.owner_wait import prepare_owner_wait_handoffs

            restart_transaction_id = uuid.uuid4().hex
            owner_wait_ids = prepare_owner_wait_handoffs(
                DRIVE_ROOT, worker_state.RUNNING, restart_transaction_id,
            )
            preserve_running_task_ids = prepare_planned_restart_handoffs(
                DRIVE_ROOT,
                worker_state.RUNNING,
                restart_transaction_id=restart_transaction_id,
                additional_task_ids=owner_wait_ids,
            )
            from supervisor.restart_retention import prepare_restart_returns

            # The update's restart returns active work and the runnable queue (#1563).
            return_ids = prepare_restart_returns(DRIVE_ROOT, worker_state.RUNNING, list(worker_state.PENDING),
                                                transaction_id=restart_transaction_id,
                                                direct=worker_state.direct_chat_turns(),
                                                owner_wait_ids=owner_wait_ids)
        except Exception as exc:
            open_repo_writer_admission()
            log.warning("Managed update owner-wait handoff preparation failed", exc_info=True)
            return [f"owner_wait_handoff:{type(exc).__name__}: {exc}"]
    with worker_state._queue_lock:
        for task_id, captured in update_returns["rows"].items():
            captured["returning"] = task_id in return_ids
    update_progress.advance("stopping_workers")
    survivors = kill_workers_for_update(
        result_reason="Task interrupted by an owner-requested managed update.",
        terminal_status="interrupted",
        preserve_running_task_ids=preserve_running_task_ids,
    )
    if survivors:
        return survivors
    try:
        from ouroboros.tools.services import kill_all_services

        update_progress.advance("stopping_services")
        stopped = kill_all_services(DRIVE_ROOT, wait=True, include_keep_alive=True)
    except Exception as exc:
        return [f"services:{type(exc).__name__}: {exc}"]
    failed = [
        str(item.get("service_id") or item.get("name") or "unknown")
        for item in stopped
        if isinstance(item, dict)
        and (item.get("stop_failed") or item.get("cleanup_dispatched") is False
             or item.get("state") in {"running", "cleanup_pending"}
             or item.get("lifecycle") in {"running", "cleanup_pending"})
    ]
    try:
        from ouroboros.process_custody import quiesce_custodied_services

        _custody_ok, custody_blockers = quiesce_custodied_services(DRIVE_ROOT)
    except Exception as exc:
        custody_blockers = [f"custody_ledger:{type(exc).__name__}: {exc}"]
    blockers = [f"service:{label}" for label in failed] + custody_blockers
    update_returns["quiesced"] = not blockers
    return blockers


def _fence_failure(blockers: list[str], stash_note: str = "") -> JSONResponse:
    active_only = bool(blockers) and all(str(item).startswith("active:") for item in blockers)
    restart_required = not active_only
    if active_only:
        names = ", ".join(str(item)[len("active:"):] for item in blockers)
        error = (
            f"Active work is still running or finishing review ({names}). "
            "No work was stopped and the repository was not changed; wait for it to finish, "
            "pause it, or restart Ouroboros to clear the blocker."
        )
    else:
        error = (
            "Could not prove every repository writer stopped. The repository was not "
            "changed, but runtime shutdown may be incomplete."
        )
    return JSONResponse(
        {
            "error": error,
            "reason": "update_writer_fence_blocked",
            "blockers": blockers,
            "restart_required": restart_required,
            **({"stash_note": stash_note} if stash_note else {}),
        },
        status_code=409,
    )


def _rollback_fenced_update(reason: str, error: str, **extra: Any) -> JSONResponse:
    from supervisor.update_merge import (
        active_update_tx, mark_update_tx_gate_blocked, rollback_managed_update, write_update_tx, _rollback_target_unmet,
    )

    tx = active_update_tx()
    failure = tx.get("git_failure") or {}
    unapplied = bool(failure.get("before_tree_effects")) and not _rollback_target_unmet(
        str(tx.get("pre_update_sha") or ""), str(tx.get("pre_update_branch") or ""))
    if tx:
        tx["rollback_error"] = error
        write_update_tx(tx)
    update_progress.advance("rolling_back")
    ok, message = rollback_managed_update(reason)
    if ok:
        _respawn_workers_after_failed_update()
        return JSONResponse(
            {"error": error, "rolled_back": not unapplied, "outcome": "not_applied" if unapplied else "rolled_back",
             "rollback": message, "stash_note": message, **extra},
            status_code=409,
        )
    if not active_update_tx().get("stash_restore"):
        mark_update_tx_gate_blocked(reason, message)
    return JSONResponse(
        {
            "error": error,
            "rolled_back": False,
            "rollback": message,
            "stash_note": message,
            "restart_required": True,
            **extra,
        },
        status_code=500,
    )


def _abandon_replace_refused_before_tree(tx: Dict[str, Any], exc: BaseException) -> Dict[str, Any] | None:
    """Abandon this replace as unapplied, as boot does, when Git refused its FIRST tree step for
    the held index lock; ``None`` leaves the fenced rollback in charge of anything else.

    Both facts are ``IndexLockRefusal.before_tree_effects`` (Git's own refusal on every attempt,
    no earlier tree step); the live tx must still be this attempt and HEAD its base. The rescue
    snapshot and keep branch made before the checkout stay and are reported by the caller.
    """
    import subprocess
    from supervisor.git_ops import git_capture
    from supervisor.git_ops_reset import IndexLockRefusal, refusal_parts
    from supervisor.update_merge import abandon_unapplied_replace, read_update_tx_strict, write_update_tx

    if not isinstance(exc, subprocess.CalledProcessError):
        return None
    status, live = read_update_tx_strict()
    if status != "valid" or live.get("phase") != "applying_replace" or live.get("attempt_id") != tx.get("attempt_id"):
        return None
    cmd, detail = refusal_parts(exc)
    before = isinstance(exc, IndexLockRefusal) and exc.before_tree_effects
    live["git_failure"] = {"cmd": cmd, "stderr": detail, "returncode": exc.returncode, "before_tree_effects": before}
    write_update_tx(live)
    if not before:
        return None
    rc_h, head, _error = git_capture(["git", "rev-parse", "--verify", "HEAD"])
    if rc_h != 0 or not head or head != str(live.get("pre_update_sha") or ""):
        return None
    return abandon_unapplied_replace(head, cause="index_lock_refused", attempt_id=str(live["attempt_id"]),
                                     cmd=cmd, stderr=detail)


def _restart_response(request: Request, *, strategy: str, plan: dict, keep_branch: str = "") -> JSONResponse:
    try:
        restarting = _request_restart(request)
    except Exception as exc:
        log.warning("managed update restart request failed", exc_info=True)
        restarting = False
        restart_error = f"{type(exc).__name__}: {exc}"
    else:
        restart_error = "restart callback is unavailable" if not restarting else ""
    if not restarting:
        try:
            from ouroboros.delegate_recovery import PLANNED_RESTART_TRANSACTION_ENV
            import os

            os.environ.pop(PLANNED_RESTART_TRANSACTION_ENV, None)
        except Exception:
            log.warning("failed to disarm managed update restart transaction", exc_info=True)
        return JSONResponse(
            {
                "status": "restart_required",
                "error": restart_error,
                "strategy": strategy,
                "merge_plan": plan,
                **({"keep_branch": keep_branch} if keep_branch else {}),
            }
        )
    update_progress.advance("restart_requested")
    return JSONResponse(
        {"status": "ok", "restarting": True, "strategy": strategy, "merge_plan": plan,
         **({"keep_branch": keep_branch} if keep_branch else {})}
    )


def _stash_local_work_fenced(
    *, branch: str, base_sha: str, target_sha: str, plan: dict
) -> tuple[dict | None, JSONResponse | None]:
    """Shared stash-first prologue for BOTH update lanes (owner decisions Q1=C, Q9).

    Runs AFTER the writer fence and BEFORE the authoritative replan, so the final
    plan — conflict inventory included — is computed from the exact clean tree the
    merge will run on (a plan computed over dirty content that a later stash
    removes could route to the wrong lane). Writes the durable ``stashing_local_work``
    tx BEFORE the stash mutation (boot recovery restores a crash in between) and
    fails closed on an unexplained still-dirty tree. Returns ``(tx, None)`` on
    success or ``(None, error_response)``."""
    import uuid as _uuid

    from supervisor.git_ops import git_capture
    from supervisor.update_merge import (
        clear_update_tx,
        stash_local_changes_for_update,
        write_update_tx,
    )

    attempt_id = _uuid.uuid4().hex[:12]
    tx = {
        "phase": "stashing_local_work",
        "pre_update_sha": base_sha,
        "pre_update_branch": branch,
        "base_sha": base_sha,
        "target_sha": target_sha,
        "target_ref": str(plan.get("target_ref") or ""),
        "update_channel": str(plan.get("update_channel") or ""),
        "attempt_id": attempt_id,
        "stash_sha": "",
        "local_work_carrier": "none",
        "requested_at": utc_now_iso(),
    }
    update_progress.link(tx)  # bound before this transaction can be written
    rc_ds, dirty_now, dirty_error = git_capture(["git", "status", "--porcelain"])
    if rc_ds != 0:
        _respawn_workers_after_failed_update()
        return None, JSONResponse(
            {"error": f"could not inspect local changes before the update: {dirty_error}"},
            status_code=409,
        )
    if dirty_now.strip():
        write_update_tx(tx)
        stash_status, stash_sha, stash_error = stash_local_changes_for_update(attempt_id)
        if stash_status == "ok" and not stash_sha:
            rc_rs, still_dirty, _rse = git_capture(["git", "status", "--porcelain"])
            if rc_rs != 0 or still_dirty.strip():
                stash_status, stash_error = "push_failed", (
                    "the worktree still reports local changes after an empty stash"
                )
        if stash_status == "lookup_unknown":
            # The entry EXISTS but cannot be listed: KEEP the durable
            # stashing_local_work tx — boot retries the lookup and restores;
            # clearing it here would orphan the owner's work behind an HTTP
            # error that may never be seen.
            _respawn_workers_after_failed_update()
            return None, JSONResponse(
                {"error": f"could not verify the update stash: {stash_error}",
                 "reason": "stash_lookup_unknown"},
                status_code=409,
            )
        if stash_status != "ok":
            clear_update_tx()
            _respawn_workers_after_failed_update()
            return None, JSONResponse(
                {"error": f"could not preserve local changes before the update: {stash_error}"},
                status_code=409,
            )
        tx["stash_sha"] = stash_sha
        tx["local_work_carrier"] = "stash" if stash_sha else "none"
        write_update_tx(tx)
        if stash_sha:
            # The stash commit is now the ONLY durable home of the owner's
            # uncommitted+untracked work — pin it for BOTH lanes so gc or a
            # stash drop can never lose it.
            from supervisor.update_merge import create_rescue_local_ref

            if not create_rescue_local_ref(stash_sha):
                note = _unwind_stashed_update(tx, "stash_pin_failed")
                _respawn_workers_after_failed_update()
                return None, JSONResponse(
                    {"error": "could not durably pin the local update stash",
                     **({"stash_note": note} if note else {})},
                    status_code=409,
                )
    return tx, None


def _unwind_stashed_update(tx: dict, context: str) -> str:
    """Undo the stash prologue when the update aborts before any repo mutation:
    restore the exact stash and clear only a confirmed restoration/preservation
    outcome. An incomplete result retains its write-ahead marker for boot."""
    from supervisor.update_merge import clear_update_tx, restore_stash_with_marker

    result = restore_stash_with_marker(tx, context)
    note = result.note
    if result.complete and not clear_update_tx():
        note = (note + "; " if note else "") + "the update transaction marker could not be cleared"
    return note


def _start_assisted_merge_fenced(plan: dict, tx: dict) -> JSONResponse:
    """Stage the exact planned merge and enqueue its one reviewed resolver.

    ``tx`` is the shared stash-first prologue transaction: the owner's dirty and
    untracked work already rides its stash entry (``stash_sha``), the tree is
    clean, and ``plan`` was computed from that clean tree — so
    ``plan.local_snapshot == base_sha`` and the merge needs no synthetic
    snapshot commit."""
    from supervisor.git_ops import (
        BRANCH_DEV, _collect_repo_sync_state, _create_rescue_snapshot,
    )
    from supervisor.state import budget_remaining, load_state
    from supervisor.update_merge import (
        assisted_writer_gate_reason,
        enqueue_assisted_resolution_task,
        ensure_assisted_resolver_ready,
        existing_failed_update_ref,
        materialize_assisted_merge_live,
        write_update_tx,
    )
    from supervisor.workers import close_repo_writer_admission, kill_workers_for_update

    branch = BRANCH_DEV
    base_sha = str(plan.get("base_sha") or "")
    target_sha = str(plan.get("target_sha") or "")
    local_snapshot = str(plan.get("local_snapshot") or "")
    if not local_snapshot or not target_sha:
        note = _unwind_stashed_update(tx, "assisted_admission_failed")
        _respawn_workers_after_failed_update()
        return JSONResponse(
            {"error": "could not build local snapshot / target",
             **({"stash_note": note} if note else {})},
            status_code=409,
        )
    try:
        remaining = budget_remaining(load_state() or {}, strict=True)
    except Exception:
        note = _unwind_stashed_update(tx, "assisted_admission_failed")
        _respawn_workers_after_failed_update()
        return JSONResponse(
            {"error": "Assisted update cannot start because model budget authority is unavailable.",
             **({"stash_note": note} if note else {})},
            status_code=409,
        )
    if remaining <= 0:
        note = _unwind_stashed_update(tx, "assisted_admission_failed")
        _respawn_workers_after_failed_update()
        return JSONResponse(
            {"error": "Assisted update needs model budget to review local changes; nothing was changed.",
             **({"stash_note": note} if note else {})},
            status_code=409,
        )
    # Money admission is the known-spend rule every reviewer seat's reservation
    # applies (#1487), checked just above: known spend below TOTAL_BUDGET admits
    # the update, and a review that later reaches the limit pauses on its own
    # fences. One commit-gate wave over the review pool's paid seats is still
    # priced at the review packs' own worst-case caps by the one explicit
    # estimator (``review_admission.managed_update_wave_estimate``), but as
    # DISCLOSURE, never a second, earlier refusal: a prospective wave larger
    # than the remainder used to refuse an update the owner's limit still
    # allowed. Agent-session rows ride subscriptions, not USD budget; an
    # estimator error is recorded, never a zero, and a broken import is not
    # swallowed.
    from ouroboros.tools.review_admission import managed_update_wave_estimate

    estimate_event = managed_update_wave_estimate(float(remaining))
    try:
        from supervisor.git_ops import DRIVE_ROOT as _dr
        from ouroboros.utils import append_jsonl as _aj, utc_now_iso as _n

        # One durable disclosure of the wave estimate beside the known room;
        # unknowable parts are counted, never filled in (P1).
        _aj(_dr / "logs" / "supervisor.jsonl", {"ts": _n(), **estimate_event})
    except Exception:
        log.debug("wave estimate event write failed", exc_info=True)

    _create_rescue_snapshot(
        branch, "ui_update_assisted_merge", _collect_repo_sync_state(),
    )

    st = load_state() or {}
    try:
        owner_chat_id = int(st.get("owner_chat_id") or 0)
    except (TypeError, ValueError):
        owner_chat_id = 0
    import uuid as _uuid

    task_id = "update_assisted_merge_" + _uuid.uuid4().hex[:8]
    prior_attempt_ref = existing_failed_update_ref(target_sha, not_at=base_sha)
    tx.update({
        "phase": "materializing_assisted",
        "local_snapshot": local_snapshot,
        "conflict_paths": (
            list(plan.get("code_conflict_paths") or [])
            + list(plan.get("doc_conflict_paths") or [])
        ),
        "task_id": task_id,
        # No resolver submitted yet: the first admission of this fresh id is provable.
        "resolver_submitted_id": "",
        "owner_chat_id": owner_chat_id,
        "resolution_attempts": 0,
        **({"failed_update_ref": prior_attempt_ref} if prior_attempt_ref else {}),
    })
    close_repo_writer_admission(assisted_writer_gate_reason(tx))
    if not ensure_assisted_resolver_ready(base_sha):
        blockers = kill_workers_for_update(
            result_reason="Assisted update resolver did not become ready.",
            terminal_status="interrupted",
        )
        if blockers:
            # Even with hung writers the tree itself was never touched: bring the
            # stashed work back and drop the prologue tx, or the owner is left
            # with invisible work and a commit-blocking marker until a restart.
            note = _unwind_stashed_update(tx, "assisted_resolver_fence_blocked")
            return _fence_failure([f"resolver:{item}" for item in blockers], stash_note=note)
        # Nothing touched the tree yet: bring the stashed work back and drop the
        # prologue tx so the owner simply retries.
        note = _unwind_stashed_update(tx, "assisted_resolver_not_ready")
        _respawn_workers_after_failed_update()
        return JSONResponse(
            {"error": "Assisted update could not boot its resolver before staging conflicts.",
             **({"stash_note": note} if note else {})},
            status_code=409,
        )
    # The server's own owner-control path must be resident BEFORE conflict
    # markers reach the live tree: a first function-local import after that
    # point raises SyntaxError on a conflicted module (#283). Best effort.
    from supervisor.worker_chat_lane import preload_owner_control_path

    preload_owner_control_path()
    # Final late-mutation guard: the resolver boot above can wait ~90s and the
    # writer fence stops Ouroboros, not humans — re-verify the exact planned
    # state IMMEDIATELY before the first destructive command.
    from supervisor.update_merge import destructive_apply_guard

    guard_reason = destructive_apply_guard(branch, base_sha)
    if guard_reason:
        note = _unwind_stashed_update(tx, "late_local_changes")
        _respawn_workers_after_failed_update()
        return JSONResponse(
            {"error": f"the repository changed before the merge could be staged ({guard_reason}); "
                      "nothing was applied — retry the update",
             "reason": "late_local_changes",
             **({"stash_note": note} if note else {})},
            status_code=409,
        )
    write_update_tx(tx)
    update_progress.advance("applying")
    ok, msg, m0_tree = materialize_assisted_merge_live(branch, local_snapshot, target_sha, base_sha)
    if not ok:
        return _rollback_fenced_update(
            "assisted_materialize_failed", f"could not stage the merge: {msg}"
        )
    tx["phase"] = "assisted_resolution"
    tx["m0_tree"] = m0_tree
    # Truthful work list: mechanical projection (VERSION) may have resolved a
    # path the plan still counted — the resolver's objective and the review
    # anchors read the ACTUAL live conflict inventory.
    try:
        from supervisor.update_merge import live_unmerged_paths

        actual_conflicts = live_unmerged_paths()
        # None = Git error: keep the plan's list, never claim "no conflicts".
        if actual_conflicts is not None:
            tx["conflict_paths"] = actual_conflicts
    except Exception:
        log.debug("live conflict inventory refresh failed; keeping the plan's list", exc_info=True)
    write_update_tx(tx)
    if not enqueue_assisted_resolution_task(tx):
        return _rollback_fenced_update(
            "assisted_worker_start_failed",
            "the merge was staged but its resolver could not be started or admitted",
        )
    return JSONResponse({"status": "assisted_started", "task_id": task_id, "merge_plan": plan})


def _apply_clean_merge_fenced(request: Request, plan: dict, tx: dict) -> JSONResponse:
    """Land one exact clean plan transactionally, then request restart.

    ``tx`` is the shared stash-first prologue transaction (owner decision Q1=C:
    dirty local work rides the stash, never committed history; Q9 unified both
    lanes behind one prologue). The tree is already clean and ``plan`` was
    computed from it."""
    from supervisor.git_ops import (
        BRANCH_DEV, _collect_repo_sync_state, _create_rescue_snapshot,
    )
    from supervisor.update_merge import (
        apply_managed_merge_update,
        update_restart_smoke,
        write_update_tx,
    )

    branch = BRANCH_DEV
    merge_commit = str(plan.get("merge_commit") or "")
    if not merge_commit:
        note = _unwind_stashed_update(tx, "clean_apply_admission_failed")
        _respawn_workers_after_failed_update()
        return JSONResponse(
            {"error": "clean update plan did not produce a target commit",
             **({"stash_note": note} if note else {})},
            status_code=409,
        )
    _create_rescue_snapshot(
        branch, "ui_update_apply_merge", _collect_repo_sync_state(),
    )
    from supervisor.update_merge import destructive_apply_guard

    guard_reason = destructive_apply_guard(branch, str(tx.get("pre_update_sha") or ""))
    if guard_reason:
        note = _unwind_stashed_update(tx, "late_local_changes")
        _respawn_workers_after_failed_update()
        return JSONResponse(
            {"error": f"the repository changed before the update could be applied ({guard_reason}); "
                      "nothing was applied — retry the update",
             "reason": "late_local_changes",
             **({"stash_note": note} if note else {})},
            status_code=409,
        )
    tx.update({
        "merge_commit": merge_commit,
        "phase": "pending_boot_smoke",
        "pre_restart_smoke": "pending",
        "rollback_attempted": False,
    })
    write_update_tx(tx)
    update_progress.advance("applying")
    ok, msg = apply_managed_merge_update(branch, merge_commit)
    if not ok:
        return _rollback_fenced_update(
            "merge_apply_failed", f"merge apply failed: {msg}"
        )
    update_progress.advance("checking")
    smoke = update_restart_smoke()
    if not smoke.get("ok"):
        return _rollback_fenced_update(
            "pre_restart_smoke_failed",
            "pre-restart smoke failed",
            smoke=smoke,
        )
    tx["pre_restart_smoke"] = "passed"
    write_update_tx(tx)
    return _restart_response(request, strategy="auto_merge", plan=plan)


@update_progress.observe_result
def _apply_smart_update_fenced(
    request: Request,
    *,
    expected_base_sha: str,
    expected_target_sha: str,
) -> JSONResponse:
    from supervisor.update_merge import (
        acquire_update_lock,
        active_update_tx,
        plan_managed_update_merge,
        release_update_lock,
    )

    refusal = _index_lock_precondition()
    if refusal is not None:
        return refusal
    plan = plan_managed_update_merge(fetch=True, build=False)
    kind = str(plan.get("kind") or "")
    if not plan.get("available") or kind not in _KNOWN_UPDATE_PLAN_KINDS:
        return JSONResponse(
            {"error": plan.get("error") or "no actionable managed update", "merge_plan": plan},
            status_code=409,
        )
    if not _pins_match(plan, expected_base_sha, expected_target_sha):
        return JSONResponse(
            {"error": "the update changed after preflight; check again", "reason": "release_moved", "merge_plan": plan},
            status_code=409,
        )
    try:
        lock_fh = acquire_update_lock()
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)
    try:
        if active_update_tx():
            return JSONResponse({"error": "a managed update is already in progress"}, status_code=409)
        update_progress.begin(base_sha=expected_base_sha, target_sha=expected_target_sha)
        blockers = _quiesce_repo_writers("smart")
        if blockers:
            return _fence_failure(blockers)
        # Stash-first (Q9): the owner's dirty + untracked work moves to a stash
        # BEFORE the authoritative replan, so the final plan — conflict inventory
        # and lane choice included — describes the exact clean tree the merge
        # will actually run on.
        from supervisor.git_ops import BRANCH_DEV as _branch_dev

        update_progress.advance("preparing")
        tx, stash_failure = _stash_local_work_fenced(
            branch=_branch_dev,
            base_sha=expected_base_sha,
            target_sha=expected_target_sha,
            plan=plan,
        )
        if stash_failure is not None:
            return stash_failure
        plan2 = plan_managed_update_merge(fetch=False, build=True)
        if (
            not plan2.get("available")
            or str(plan2.get("kind") or "") not in _KNOWN_UPDATE_PLAN_KINDS
            or not _pins_match(plan2, expected_base_sha, expected_target_sha)
            or str(plan2.get("target_ref") or "") != str(plan.get("target_ref") or "")
            or str(plan2.get("update_channel") or "") != str(plan.get("update_channel") or "")
        ):
            note = _unwind_stashed_update(tx, "plan_changed_after_stash")
            _respawn_workers_after_failed_update()
            return JSONResponse(
                {"error": "the update plan changed while writers were stopping; nothing was applied",
                 "reason": "release_moved", "merge_plan": plan2,
                 **({"stash_note": note} if note else {})},
                status_code=409,
            )
        # Q1=C / Q9 hard invariant: the replan must describe the CLEAN post-stash
        # tree. Late local changes (a human editing on the host between the stash
        # and this point — writers are fenced, humans are not) would otherwise
        # ride the synthetic-snapshot path into committed history or be wiped by
        # the destructive apply. Fail closed and disclose.
        if (
            int(plan2.get("local_dirty_count") or 0) != 0
            or str(plan2.get("local_snapshot") or "") != expected_base_sha
        ):
            note = _unwind_stashed_update(tx, "late_local_changes")
            _respawn_workers_after_failed_update()
            return JSONResponse(
                {"error": (
                    "local changes appeared after the update stash; nothing was applied — "
                    "retry the update"
                ), "reason": "late_local_changes",
                 **({"stash_note": note} if note else {})},
                status_code=409,
            )
        if _plan_is_clean(plan2):
            return _apply_clean_merge_fenced(request, plan2, tx)
        return _start_assisted_merge_fenced(plan2, tx)
    except Exception as exc:
        log.error("managed smart update failed after writer fence", exc_info=True)
        from supervisor.update_merge import active_update_tx as _active_tx

        pending = _active_tx()
        if pending.get("stash_restore"):
            # A restore may already have returned files even if its completion
            # record failed. Never route that uncertainty into generic rollback.
            record = pending["stash_restore"]
            return JSONResponse(
                {"error": f"managed update failed: {type(exc).__name__}: {exc}",
                 "reason": "stash_recovery_incomplete", "restart_required": True,
                 "stash_note": record.get("note") or "Local work recovery is pending; current files were left unchanged."},
                status_code=500,
            )
        if pending:
            return _rollback_fenced_update(
                "smart_update_exception",
                f"managed update failed: {type(exc).__name__}: {exc}",
            )
        _respawn_workers_after_failed_update()
        return json_exception(exc, 500)  # recorded once above, with its stack
    finally:
        release_update_lock(lock_fh)


async def _apply_smart_update(
    request: Request,
    *,
    expected_base_sha: str,
    expected_target_sha: str,
) -> JSONResponse:
    return await asyncio.to_thread(
        _apply_smart_update_fenced,
        request,
        expected_base_sha=expected_base_sha,
        expected_target_sha=expected_target_sha,
    )


@update_progress.observe_result
def _apply_replace_recovery_fenced(
    request: Request,
    *,
    expected_base_sha: str,
    expected_target_sha: str,
) -> JSONResponse:
    import uuid

    from ouroboros.store_switch import switch_or_reset
    from supervisor.git_ops import (
        BRANCH_DEV,
        DRIVE_ROOT,
        REPO_DIR,
        _write_update_intent,
        checkout_and_reset,
        prepare_managed_update,
    )
    from supervisor.update_merge import (
        acquire_update_lock,
        active_update_tx,
        plan_managed_update_merge,
        release_update_lock,
        update_restart_smoke,
        write_update_tx,
    )

    refusal = _index_lock_precondition()
    if refusal is not None:
        return refusal
    plan = plan_managed_update_merge(fetch=True, build=False)
    if (
        str(plan.get("kind") or "") not in (_KNOWN_UPDATE_PLAN_KINDS | {"current"})
        or not _pins_match(plan, expected_base_sha, expected_target_sha)
    ):
        return JSONResponse(
            {"error": "the recovery target changed after preflight", "reason": "release_moved", "merge_plan": plan},
            status_code=409,
        )
    try:
        lock_fh = acquire_update_lock()
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)
    try:
        if active_update_tx():
            return JSONResponse({"error": "a managed update is already in progress"}, status_code=409)
        update_progress.begin(base_sha=expected_base_sha, target_sha=expected_target_sha)
        blockers = _quiesce_repo_writers("replace_recovery")
        if blockers:
            return _fence_failure(blockers)
        update_progress.advance("preparing")
        plan2 = plan_managed_update_merge(fetch=False, build=False)
        if (
            str(plan2.get("kind") or "") not in (_KNOWN_UPDATE_PLAN_KINDS | {"current"})
            or not _pins_match(plan2, expected_base_sha, expected_target_sha)
            or str(plan2.get("target_ref") or "") != str(plan.get("target_ref") or "")
            or str(plan2.get("update_channel") or "") != str(plan.get("update_channel") or "")
        ):
            _respawn_workers_after_failed_update()
            return JSONResponse(
                {"error": "the recovery target changed while writers were stopping", "reason": "release_moved", "merge_plan": plan2},
                status_code=409,
            )
        ok, payload = prepare_managed_update(
            "replace",
            expected_base_sha=expected_base_sha,
            expected_target_sha=expected_target_sha,
            arm_intent=False,
        )
        if not ok:
            _respawn_workers_after_failed_update()
            return JSONResponse(payload, status_code=409)
        tx = {
            "pre_update_sha": expected_base_sha,
            "pre_update_branch": BRANCH_DEV,
            "target_sha": expected_target_sha,
            "target_ref": str(plan2.get("target_ref") or ""),
            "update_channel": str(plan2.get("update_channel") or ""),
            "merge_commit": expected_target_sha,
            "phase": "applying_replace",
            "pre_restart_smoke": "pending",
            "pre_update_dirty_count": int(plan2.get("local_dirty_count") or 0),
            "attempt_id": uuid.uuid4().hex[:12],
            "strategy": "replace",
        }
        update_progress.link(tx, keep_branch=str(payload.get("keep_branch") or ""))
        write_update_tx(tx)
        _write_update_intent(dict(payload["update_intent"]))
        try:
            update_progress.advance("applying")
            # A target without the store: the switch intent, the checkout moves at the restart exit
            # (store_switch); its smoke is the post-boot one the pending transaction already runs.
            checkout_ok, checkout_msg, deferred = switch_or_reset(
                DRIVE_ROOT, REPO_DIR, target_sha=expected_target_sha, target_ref=str(tx["target_ref"]),
                reason="ui_update_apply", door="update_replace",
                reset=lambda: checkout_and_reset(BRANCH_DEV, reason="ui_update_apply", unsynced_policy="rescue_and_reset"))
        except Exception as exc:
            from supervisor.git_ops_reset import refusal_parts, refusal_text

            abandoned = _abandon_replace_refused_before_tree(tx, exc)
            if abandoned is None:
                return _rollback_fenced_update(
                    "replace_checkout_exception", f"recovery checkout failed: {refusal_text(exc)}", **payload
                )
            if not abandoned.get("abandoned"):  # its cleanup failed: the tx holds, as at boot
                return JSONResponse({"error": abandoned.get("reason"), "rolled_back": False,
                                     "restart_required": True, **payload}, status_code=500)
            _respawn_workers_after_failed_update()
            return JSONResponse({
                "error": f"the update was not applied: Git refused its first checkout step: {refusal_parts(exc)[1]}",
                "reason": "replace_not_applied", "abandoned": True, "rolled_back": False, **payload,
            }, status_code=409)
        if not checkout_ok:
            return _rollback_fenced_update(
                "replace_checkout_failed", f"recovery checkout failed: {checkout_msg}", **payload
            )
        tx["phase"] = "pending_boot_smoke"
        write_update_tx(tx)
        if not deferred:
            update_progress.advance("checking")
            smoke = update_restart_smoke()
            if not smoke.get("ok"):
                return _rollback_fenced_update(
                    "replace_pre_restart_smoke_failed", "pre-restart smoke failed", smoke=smoke
                )
            tx["pre_restart_smoke"] = "passed"
            write_update_tx(tx)
        return _restart_response(request, strategy="replace", plan=plan2, keep_branch=str(payload.get("keep_branch") or ""))
    except Exception as exc:
        log.error("managed replace recovery failed after writer fence", exc_info=True)
        if active_update_tx():
            return _rollback_fenced_update(
                "replace_update_exception",
                f"managed recovery failed: {type(exc).__name__}: {exc}",
            )
        _respawn_workers_after_failed_update()
        return json_exception(exc, 500)  # recorded once above, with its stack
    finally:
        release_update_lock(lock_fh)


async def _apply_replace_recovery(
    request: Request,
    *,
    expected_base_sha: str,
    expected_target_sha: str,
) -> JSONResponse:
    return await asyncio.to_thread(
        _apply_replace_recovery_fenced,
        request,
        expected_base_sha=expected_base_sha,
        expected_target_sha=expected_target_sha,
    )


async def api_update_apply(request: Request) -> JSONResponse:
    """Apply an exact managed plan; replacement is an explicit recovery only."""
    body = await request_json_or(request, {}, exceptions=(Exception,))
    if not isinstance(body, dict):
        return json_error("JSON body must be an object.", 400)
    # Executable gateway ABI (ABI-3, Q7=A): UpdateApplyRequest declares
    # `strategy` REQUIRED with a closed vocabulary — the derived schema now
    # enforces the contract as written (no silent auto_merge default).
    from ouroboros.gateway.contracts import UpdateApplyRequest
    from ouroboros.gateway.schema import validate_ingress

    schema_errors = validate_ingress(body, UpdateApplyRequest)
    if schema_errors:
        return json_error(
            f"invalid request body: {schema_errors[0]}", 400,
            schema_errors=schema_errors[:8],
        )
    strategy = str(body.get("strategy") or "").strip().lower()
    if strategy not in _UPDATE_STRATEGIES:
        return json_error(f"unsupported update strategy: {strategy or 'missing'}", 400)
    expected_base_sha = str(body.get("expected_base_sha") or "").strip()
    expected_target_sha = str(body.get("expected_target_sha") or "").strip()
    if strategy == "manual":
        from supervisor.update_merge import plan_managed_update_merge

        plan = await asyncio.to_thread(plan_managed_update_merge, fetch=True)
        return JSONResponse({"status": "manual", "merge_plan": plan})
    if strategy != "manual":
        from supervisor.update_merge import active_update_tx

        if active_update_tx():
            return JSONResponse({"error": "a managed update is already in progress"}, status_code=409)
    if not expected_base_sha or not expected_target_sha:
        return json_error("fresh preflight base and target SHA are required", 400)
    if strategy == "replace":
        if body.get("confirm_recovery") is not True:
            return json_error("replace is a recovery action and requires confirm_recovery=true", 400)
        return await _apply_replace_recovery(
            request,
            expected_base_sha=expected_base_sha,
            expected_target_sha=expected_target_sha,
        )
    # auto_merge and assisted share one smart flow; the fresh plan, not the
    # caller's guess, decides whether the supervisor can fast-forward/merge or
    # Ouroboros must resolve it through reviewed assisted mode.
    return await _apply_smart_update(
        request,
        expected_base_sha=expected_base_sha,
        expected_target_sha=expected_target_sha,
    )


async def api_evolution_data(request: Request) -> JSONResponse:
    """The saved version history at once (``evolution_history``), never waiting for a measurement;
    ``force`` re-measures in the background. The refresh status says whether a newer copy is coming."""
    from ouroboros.evolution_history import history

    force_refresh = str(request.query_params.get("force") or "").strip().lower() in {"1", "true", "yes"}
    drive_root = request_drive_root(request)
    payload = await asyncio.to_thread(
        history.projection, str(request_repo_dir(request)), str(drive_root), force=force_refresh)
    # Older clients' newest absorb checkpoints: one stat while the journal is unchanged (evolution_history).
    payload["checkpoints"] = await asyncio.to_thread(history.checkpoints, str(drive_root))
    payload["generated_at"] = utc_now_iso()
    return JSONResponse(payload, headers={"Cache-Control": "no-store"})


__all__ = [
    "api_command",
    "api_evolution_data",
    "api_git_log",
    "api_git_promote",
    "api_git_rollback",
    "api_reset",
    "api_update_apply",
    "api_update_check",
    "api_update_preflight",
    "api_update_status",
]
