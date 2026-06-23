"""Control, update, and evolution HTTP endpoints."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros import get_version
from ouroboros.gateway._helpers import json_error, json_exception, request_drive_root, request_json_or, request_repo_dir
from ouroboros.gateway.ws import broadcast_ws_sync
from ouroboros.outcomes import public_task_result
from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)

_RECENT_VISIBLE_COMMANDS: Dict[str, float] = {}
_VISIBLE_COMMAND_DEDUPE_SEC = 5.0
_evo_cache: Dict[str, Any] = {}
_evo_task: asyncio.Task | None = None


def _request_restart(request: Request) -> None:
    callback = getattr(getattr(request.app, "state", None), "request_restart", None)
    if callable(callback):
        callback()


def _runtime_branch_defaults(request: Request) -> tuple[str, str]:
    callback = getattr(getattr(request.app, "state", None), "runtime_branch_defaults", None)
    if callable(callback):
        return callback()
    return "ouroboros", "ouroboros-stable"


def _managed_update_payload(*, fetch: bool, include_tags: bool) -> dict[str, Any]:
    from supervisor.git_ops import compute_managed_update_status, git_capture

    status = compute_managed_update_status(fetch=fetch)
    # P2 2F check-on-restart cache: a passive read (fetch=False) bails before resolving the
    # official ref ("official_status_requires_check"), so overlay the boot/manual-check cache
    # — the badge shows availability after a restart without re-fetching on every poll. A real
    # fetch refreshes the cache. Fresh local `dirty` still gates `safe_to_apply` downward.
    try:
        from supervisor.state import load_state, update_state

        if fetch and status.get("latest_sha"):
            from ouroboros.utils import utc_now_iso

            _snapshot = {
                "available": bool(status.get("available")),
                "safe_to_apply": bool(status.get("safe_to_apply")),
                "latest_sha": status.get("latest_sha") or "",
                "latest_short_sha": status.get("latest_short_sha") or "",
                "latest_message": status.get("latest_message") or "",
                "behind": int(status.get("behind") or 0),
                "ahead": int(status.get("ahead") or 0),
                "checked_at": utc_now_iso(),
            }
            update_state(lambda s: s.__setitem__("managed_update_cache", _snapshot))
        elif not fetch and not status.get("available"):
            cache = (load_state() or {}).get("managed_update_cache") or {}
            if cache.get("available") and cache.get("latest_sha"):
                status["available"] = True
                status["safe_to_apply"] = bool(cache.get("safe_to_apply")) and not status.get("dirty")
                status["latest_sha"] = cache.get("latest_sha") or ""
                status["latest_short_sha"] = cache.get("latest_short_sha") or ""
                status["latest_message"] = cache.get("latest_message") or ""
                status["behind"] = int(cache.get("behind") or 0)
                status["ahead"] = int(cache.get("ahead") or 0)
                status["from_cache"] = True
                status["checked_at"] = cache.get("checked_at") or ""
    except Exception:
        log.debug("managed update status cache overlay failed", exc_info=True)
    latest_version = ""
    target_ref = status.get("target_ref") or ""
    if target_ref and status.get("latest_sha"):
        rc, version_text, _ = git_capture(["git", "show", f"{target_ref}:VERSION"])
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
        **status,
    }


async def api_reset(request: Request) -> JSONResponse:
    """Reset all runtime data (state, memory, logs, settings) but keep repo."""
    import shutil

    data_dir = request_drive_root(request)
    try:
        deleted = []
        for subdir in ("state", "memory", "logs", "archive", "locks", "task_results", "uploads"):
            target = data_dir / subdir
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
                deleted.append(subdir)
        settings_file = data_dir / "settings.json"
        if settings_file.exists():
            settings_file.unlink()
            deleted.append("settings.json")
        _request_restart(request)
        return JSONResponse({"status": "ok", "deleted": deleted, "restarting": True})
    except Exception as exc:
        return json_exception(exc)


async def api_command(request: Request) -> JSONResponse:
    try:
        body = await request.json()
        cmd = body.get("cmd", "")
        if cmd:
            from supervisor.message_bus import get_bridge, log_chat

            bridge = get_bridge()
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
            bridge.ui_send(cmd, **send_kwargs)
            if visible_task_id:
                _RECENT_VISIBLE_COMMANDS[visible_task_id] = time.monotonic()
            if visible_text:
                task_id = visible_task_id or "skill_repair"
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


async def api_git_rollback(request: Request) -> JSONResponse:
    """Roll back to a specific commit or tag, then restart."""
    try:
        body = await request.json()
        target = body.get("target", "").strip()
        if not target:
            return json_error("missing target", 400)
        from supervisor.git_ops import rollback_to_version

        ok, msg = rollback_to_version(target, reason="ui_rollback")
        if not ok:
            return json_error(msg, 400)
        _request_restart(request)
        return JSONResponse({"status": "ok", "message": msg})
    except Exception as exc:
        return json_exception(exc)


async def api_git_promote(request: Request) -> JSONResponse:
    """Promote the current dev branch to the runtime's stable branch."""
    try:
        import subprocess as sp

        branch_dev, branch_stable = _runtime_branch_defaults(request)
        sp.run(
            ["git", "branch", "-f", branch_stable, branch_dev],
            cwd=str(request_repo_dir(request)),
            check=True,
            capture_output=True,
        )
        return JSONResponse({"status": "ok", "message": f"{branch_stable} updated to match {branch_dev}"})
    except Exception as exc:
        return json_exception(exc)


async def api_update_status(_request: Request) -> JSONResponse:
    """Return passive managed-update status without fetching."""
    try:
        return JSONResponse(_managed_update_payload(fetch=False, include_tags=False))
    except Exception as exc:
        return json_exception(exc)


async def api_update_check(_request: Request) -> JSONResponse:
    """Fetch the managed remote and return fresh update status."""
    try:
        return JSONResponse(_managed_update_payload(fetch=True, include_tags=True))
    except Exception as exc:
        return json_exception(exc)


def _respawn_workers_after_failed_update() -> None:
    """Revive workers when an update aborts after they were stopped (no restart follows)."""
    try:
        from supervisor.workers import spawn_workers

        spawn_workers()
    except Exception:
        log.warning("update_apply: failed to respawn workers after aborted update", exc_info=True)


async def api_update_preflight(_request: Request) -> JSONResponse:
    """Plan the managed update as a REAL 3-way merge (P2). Does NOT touch the live
    worktree/branch/index (it fetches + merges in an isolated temp worktree), so the UI
    can present the right staged choice (auto / assisted / manual)."""
    try:
        from supervisor.update_merge import plan_managed_update_merge

        return JSONResponse({"merge_plan": plan_managed_update_merge(fetch=True)})
    except Exception as exc:
        return json_exception(exc)


def _is_protected_for_managed_update(path: str) -> bool:
    """A managed-update path that must NOT be auto-resolved by the agent (BIBLE/CHECKLISTS/
    SAFETY + release/managed invariants) — routed to MANUAL (owner) instead."""
    from ouroboros.runtime_mode_policy import is_protected_runtime_path
    from supervisor.update_merge_policy import is_protected_doc

    return bool(is_protected_doc(path) or is_protected_runtime_path(path))


def _official_protected_hits(plan: dict) -> list:
    """PROTECTED paths the official update would touch (conflicting OR clean delta). Computed
    from the plan with a read-only `git diff base..target` (no mutation) so the apply path can
    route to MANUAL BEFORE stopping workers / staging anything."""
    from supervisor.git_ops import git_capture

    base = str(plan.get("base_sha") or "")
    target = str(plan.get("target_sha") or "")
    paths = set(plan.get("protected_conflict_paths") or [])
    if base and target:
        rc, delta, _e = git_capture(["git", "diff", "--name-only", base, target])
        if rc == 0:
            paths |= {p for p in delta.splitlines() if p.strip()}
    return sorted(p for p in paths if _is_protected_for_managed_update(p))


def _start_assisted_merge(plan: dict) -> JSONResponse:
    """Orchestrate the AUTOMATED assisted managed-update merge (P2/SC2). Under the FAIL-CLOSED
    update lock with workers stopped: re-plan, route official PROTECTED-path changes to MANUAL
    (never the agent), durably rescue local work, stage a REAL `git merge --no-commit` into the
    LIVE worktree (MERGE_HEAD + conflict markers) via the supervisor, and enqueue the single
    authorized resolution task. The agent resolves the markers with normal file tools and the
    UNMODIFIED commit_reviewed lands a reviewed 2-parent merge commit (Q11) — no blocked git,
    no parallel trust path. The merge state + tx marker survive a restart (resumable recovery)."""
    import uuid as _uuid

    from ouroboros.utils import utc_now_iso
    from supervisor.git_ops import BRANCH_DEV, _create_rescue_snapshot, git_capture
    from supervisor.state import load_state
    from supervisor.update_merge import (
        acquire_update_lock,
        active_update_tx,
        create_rescue_local_ref,
        enqueue_assisted_resolution_task,
        materialize_assisted_merge_live,
        plan_managed_update_merge,
        release_update_lock,
        rollback_managed_update,
        write_update_tx,
    )

    branch = BRANCH_DEV
    try:
        lock_fh = acquire_update_lock()
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)
    try:
        if active_update_tx():  # TOCTOU: re-check UNDER the lock
            return JSONResponse({"error": "a managed update is already in progress"}, status_code=409)
        try:
            from supervisor.workers import kill_workers

            kill_workers(
                result_reason="Task interrupted by an owner-requested assisted merge update.",
                terminal_status="interrupted",
            )
        except Exception:
            log.warning("assisted merge: failed to stop workers", exc_info=True)

        plan2 = plan_managed_update_merge(fetch=False, build=False)
        if not plan2.get("available"):
            _respawn_workers_after_failed_update()
            return JSONResponse({"error": "no managed update available", **plan2}, status_code=409)
        base_sha = str(plan2.get("base_sha") or "")
        target_sha = str(plan2.get("target_sha") or "")
        local_snapshot = str(plan2.get("local_snapshot") or "")
        # (Protected-path official changes were already routed to MANUAL upstream, BEFORE
        # kill_workers — see _apply_managed_merge — so no protected recheck / task loss here.)
        if not local_snapshot or not target_sha:
            _respawn_workers_after_failed_update()
            return JSONResponse({"error": "could not build local snapshot / target", **plan2}, status_code=409)

        rc_b, cur_branch, _be = git_capture(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        rc_s, status_txt, _se = git_capture(["git", "status", "--porcelain"])
        _create_rescue_snapshot(branch, "ui_update_assisted_merge", {
            "current_branch": cur_branch if rc_b == 0 else "",
            "dirty_lines": [ln for ln in status_txt.splitlines() if ln.strip()] if rc_s == 0 else [],
            "unpushed_lines": [], "warnings": [],
        })
        create_rescue_local_ref(local_snapshot)  # durable ref: local work survives any rollback/gc

        st = load_state() or {}
        try:
            owner_chat_id = int(st.get("owner_chat_id") or 0)
        except (TypeError, ValueError):
            owner_chat_id = 0
        task_id = "update_assisted_merge_" + _uuid.uuid4().hex[:8]
        tx = {
            "phase": "materializing_assisted",
            "pre_update_sha": base_sha,
            "pre_update_branch": branch,
            "base_sha": base_sha,
            "target_sha": target_sha,
            "local_snapshot": local_snapshot,
            "conflict_paths": (list(plan2.get("code_conflict_paths") or [])
                               + list(plan2.get("doc_conflict_paths") or [])),
            "task_id": task_id,
            "owner_chat_id": owner_chat_id,
            "resolution_attempts": 0,
            "requested_at": utc_now_iso(),
        }
        write_update_tx(tx)  # BEFORE destructive materialization (crash-safe recovery)
        ok, msg = materialize_assisted_merge_live(branch, local_snapshot, target_sha, base_sha)
        if not ok:
            rollback_managed_update("assisted_materialize_failed")
            _respawn_workers_after_failed_update()
            return JSONResponse({"error": f"could not stage the merge: {msg}"}, status_code=409)
        tx["phase"] = "assisted_resolution"
        write_update_tx(tx)
        enqueue_assisted_resolution_task(tx)  # enqueues the authorized task + spawns a worker
        return JSONResponse({"status": "assisted_started", "task_id": task_id, "merge_plan": plan2})
    finally:
        release_update_lock(lock_fh)


async def _apply_managed_merge(request: Request, strategy: str) -> JSONResponse:
    """Staged merge-aware update apply (P2). auto_merge lands a CLEAN 3-way merge behind a
    FAIL-CLOSED lock with a pre-restart smoke + transactional rollback (local work is
    preserved in the merge's local-snapshot parent + a rescue snapshot). assisted/
    doc_reconcile route to the agent-assisted flow; manual returns the plan without
    mutating."""
    import uuid

    from supervisor.git_ops import BRANCH_DEV, _create_rescue_snapshot, git_capture
    from supervisor.update_merge import (
        acquire_update_lock,
        apply_managed_merge_update,
        plan_managed_update_merge,
        release_update_lock,
        rollback_managed_update,
        update_restart_smoke,
        write_update_tx,
    )

    branch = BRANCH_DEV
    plan = plan_managed_update_merge(fetch=True, build=False)
    if not plan.get("available"):
        return JSONResponse({"error": "no managed update available", **plan}, status_code=409)
    kind = str(plan.get("kind") or "")

    if strategy == "manual":
        # No mutation: hand the UI the plan; recovery artifacts are created only on apply.
        return JSONResponse({"status": "manual", "merge_plan": plan})

    # Official changes to PROTECTED paths (BIBLE/CHECKLISTS/SAFETY + release invariants) route to
    # MANUAL on EVERY mutating strategy — checked on the initial plan BEFORE any kill_workers /
    # rescue / materialization, so a read-only handoff never interrupts active tasks. The official
    # delta (base..target) does not change when workers stop, so this pre-kill check is sufficient.
    protected_hits = _official_protected_hits(plan)
    if protected_hits:
        return JSONResponse(
            {"status": "manual", "reason": "protected_paths", "protected_paths": protected_hits,
             "merge_plan": plan}
        )

    local_dirty = int(plan.get("local_dirty_count") or 0)
    if (
        strategy in ("assisted", "doc_reconcile")
        or (strategy == "auto_merge" and kind != "clean")
        # P3/P9 (triad): auto_merge fast-commits the local-snapshot parent into history
        # WITHOUT the commit_reviewed gate. That is only acceptable when the local work is
        # already-reviewed COMMITTED history; UNCOMMITTED dirty/untracked content must NOT
        # land unreviewed. So a dirty working tree routes to the REVIEWED assisted task
        # (which still PRESERVES the local changes — Q2), never a silent auto-commit.
        or (strategy == "auto_merge" and local_dirty > 0)
    ):
        # Conflicts (code/doc) OR uncommitted local work: hand the merge to Ouroboros as an
        # AUTOMATED, REVIEWED task. The supervisor stages a real merge into the live worktree
        # and the agent resolves the markers; the resulting commit flows through the standard
        # triad/scope immune review (Q11) before it lands — no blocked git, no parallel trust
        # path. The owner watches progress in chat (no manual git).
        return _start_assisted_merge(plan)

    # ---- auto_merge (clean) ----
    try:
        lock_fh = acquire_update_lock()
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)
    try:
        from supervisor.update_merge import active_update_tx

        if active_update_tx():  # TOCTOU: re-check UNDER the lock before any mutation
            return JSONResponse({"error": "a managed update is already in progress"}, status_code=409)
        try:
            from supervisor.workers import kill_workers

            kill_workers(
                result_reason="Task interrupted by an owner-requested managed merge update (restart follows).",
                terminal_status="interrupted",
            )
        except Exception:
            log.warning("update_apply(merge): failed to stop workers", exc_info=True)

        # Re-plan + build AFTER stopping writers; never trust the pre-kill plan. Re-check
        # BOTH clean-merge AND a clean working tree here: a worker (or any in-process path)
        # may have dirtied/untracked files between the pre-kill plan and now, and auto_merge
        # must NEVER fast-commit uncommitted local content unreviewed (P3/P9) — a dirty
        # post-kill plan aborts back to the reviewed assisted/manual path.
        plan2 = plan_managed_update_merge(fetch=False, build=True)
        merge_commit = str(plan2.get("merge_commit") or "")
        if plan2.get("kind") != "clean" or not merge_commit or int(plan2.get("local_dirty_count") or 0) > 0:
            _respawn_workers_after_failed_update()
            return JSONResponse(
                {"error": "update is no longer a clean auto-merge after stopping workers", **plan2},
                status_code=409,
            )
        # (Protected-path official changes were already routed to MANUAL upstream, BEFORE
        # kill_workers — see the early _official_protected_hits check in _apply_managed_merge.)

        rc_b, cur_branch, _be = git_capture(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        rc_s, status_txt, _se = git_capture(["git", "status", "--porcelain"])
        _create_rescue_snapshot(
            branch,
            "ui_update_apply_merge",
            {
                "current_branch": cur_branch if rc_b == 0 else "",
                "dirty_lines": [ln for ln in status_txt.splitlines() if ln.strip()] if rc_s == 0 else [],
                "unpushed_lines": [],
                "warnings": [],
            },
        )
        write_update_tx(
            {
                "pre_update_sha": str(plan2.get("base_sha") or ""),
                "pre_update_branch": branch,
                "target_sha": str(plan2.get("target_sha") or ""),
                "merge_commit": merge_commit,
                "phase": "pending_boot_smoke",
                "rollback_attempted": False,
                "attempt_id": uuid.uuid4().hex[:12],
            }
        )

        ok, msg = apply_managed_merge_update(branch, merge_commit)
        if not ok:
            rollback_managed_update("merge_apply_failed")
            _respawn_workers_after_failed_update()
            return JSONResponse({"error": f"merge apply failed (rolled back): {msg}"}, status_code=409)

        smoke = update_restart_smoke()
        if not smoke.get("ok"):
            rollback_managed_update("pre_restart_smoke_failed")
            _respawn_workers_after_failed_update()
            return JSONResponse(
                {"error": "pre-restart smoke failed; rolled back to the prior version", "smoke": smoke},
                status_code=409,
            )

        _request_restart(request)
        return JSONResponse(
            {"status": "ok", "restarting": True, "strategy": "auto_merge", "merge_plan": plan2}
        )
    finally:
        release_update_lock(lock_fh)


async def api_update_apply(request: Request) -> JSONResponse:
    """Apply a managed update. Default is the merge-aware auto_merge (P2); auto_merge/
    assisted/doc_reconcile/manual route to the staged merge flow, while the legacy
    'replace' (advanced escape hatch) hard-resets to the remote behind a warning."""
    body = await request_json_or(request, {}, exceptions=(Exception,))
    strategy = str(body.get("strategy") or "auto_merge")
    # Reject ANY mutating apply while a managed-update tx is already in flight (the legacy
    # 'replace' path would otherwise kill_workers + hard-reset over an in-progress assisted
    # resolution). 'manual' is read-only and always allowed. The merge paths re-check the tx
    # under the lock (TOCTOU); this is the cheap front gate.
    if strategy != "manual":
        from supervisor.update_merge import active_update_tx

        if active_update_tx():
            return JSONResponse({"error": "a managed update is already in progress"}, status_code=409)
    if strategy in ("auto_merge", "assisted", "doc_reconcile", "manual"):
        return await _apply_managed_merge(request, strategy)
    try:
        from supervisor.git_ops import BRANCH_DEV, _clear_update_intent, checkout_and_reset, prepare_managed_update

        ok, payload = prepare_managed_update(strategy)
        if not ok:
            return JSONResponse(payload, status_code=409)
        # Stop workers AFTER the update is validated/prepared but BEFORE the
        # hard reset of the live repo: a self-modifying task writing between
        # the rescue snapshot and the reset would have its edits destroyed
        # unrescued. Doing this pre-validation killed all workers on a plain
        # 409 (no update available) with no restart to revive them.
        try:
            from supervisor.workers import kill_workers

            kill_workers(
                result_reason="Task interrupted by an owner-requested managed update (restart follows).",
                terminal_status="interrupted",
            )
        except Exception:
            log.warning("update_apply: failed to stop workers before reset", exc_info=True)
        try:
            checkout_ok, checkout_msg = checkout_and_reset(
                BRANCH_DEV,
                reason="ui_update_apply",
                unsynced_policy="rescue_and_reset",
            )
        except Exception as checkout_exc:
            _clear_update_intent()
            _respawn_workers_after_failed_update()
            return JSONResponse(
                {"error": f"Prepared update but checkout failed: {checkout_exc}", **payload},
                status_code=409,
            )
        if not checkout_ok:
            _clear_update_intent()
            _respawn_workers_after_failed_update()
            return JSONResponse(
                {"error": f"Prepared update but checkout failed: {checkout_msg}", **payload},
                status_code=409,
            )
        _request_restart(request)
        return JSONResponse({"status": "ok", "restarting": True, **payload})
    except Exception as exc:
        return json_exception(exc)


async def api_evolution_data(request: Request) -> JSONResponse:
    """Collect evolution metrics for each git tag."""
    from ouroboros.utils import collect_evolution_metrics

    global _evo_task
    now = time.time()
    force_refresh = str(request.query_params.get("force") or "").strip().lower() in {"1", "true", "yes"}
    if not force_refresh and _evo_cache.get("ts") and now - _evo_cache["ts"] < 60:
        return JSONResponse({
            "points": _evo_cache["points"],
            "checkpoints": _evo_cache.get("checkpoints", []),
            "generated_at": _evo_cache.get("generated_at", ""),
            "cached": True,
        })
    if _evo_task is None or _evo_task.done():
        _evo_task = asyncio.create_task(
            collect_evolution_metrics(
                str(request_repo_dir(request)),
                data_dir=str(request_drive_root(request)),
            )
        )
    data_points = await _evo_task
    try:
        from ouroboros.evolution_checkpoints import CHECKPOINTS_REL
        from ouroboros.utils import iter_jsonl_objects

        checkpoints = []
        rows = [
            row for row in iter_jsonl_objects(request_drive_root(request) / CHECKPOINTS_REL)
            # cycle_outcome rows are solve-capability digest fodder (different
            # schema: no git_sha/identity hashes); the Dashboard checkpoints
            # view renders absorb checkpoints only.
            if isinstance(row, dict) and row.get("kind") != "cycle_outcome"
        ]
        for row in rows[-100:]:
            checkpoints.append(public_task_result(row))
    except Exception:
        checkpoints = []
    _evo_cache["ts"] = time.time()
    _evo_cache["points"] = data_points
    _evo_cache["checkpoints"] = checkpoints
    _evo_cache["generated_at"] = utc_now_iso()
    return JSONResponse({
        "points": data_points,
        "checkpoints": checkpoints,
        "generated_at": _evo_cache["generated_at"],
        "cached": False,
    })


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
