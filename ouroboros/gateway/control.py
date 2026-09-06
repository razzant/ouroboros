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
    # The update letter rides the same payload: written only after a FETCHING
    # check (never on the passive read), projected against the live HEAD/target.
    letter = None
    try:
        from ouroboros import update_letter as _letter

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
        tx = {}
    update_tx = (
        {
            "active": True,
            "phase": str(tx.get("phase") or ""),
            "task_id": str(tx.get("task_id") or ""),
            "restart_required": bool(tx.get("restart_required")),
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
            # Owner Surface Fact: messages through this endpoint (CLI `chat send`,
            # SPA heal/repair posts) would otherwise masquerade as ordinary web
            # frames. The honest stamp names the ENDPOINT — the host cannot know
            # the true caller here (disclosed non-goal).
            send_kwargs["task_metadata"] = {"client_surface": {"channel": "api_command"}}
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


def _git_rollback_fenced(request: Request, target: str) -> JSONResponse:
    """Run the complete restore transaction off the gateway event loop."""
    try:
        from supervisor.git_ops import git_capture, rollback_to_version

        lock_fh, lock_error = _acquire_repo_mutation_lock()
        if lock_error is not None:
            return lock_error
        try:
            rc, target_sha, error = git_capture(
                ["git", "rev-parse", "--verify", f"{target}^{{commit}}"]
            )
            if rc != 0:
                return json_error(error or f"cannot resolve {target}", 400)
            blockers = _quiesce_repo_writers("manual_rollback")
            if blockers:
                return _fence_failure(blockers)
            ok, msg = rollback_to_version(target_sha, reason="ui_rollback")
            if not ok:
                return JSONResponse(
                    {"error": msg, "restart_required": True}, status_code=500
                )
            try:
                restarting = _request_restart(request)
            except Exception:
                log.warning("manual rollback landed but restart request failed", exc_info=True)
                restarting = False
            return JSONResponse({
                "status": "ok" if restarting else "restart_required",
                "message": msg,
                "restarting": restarting,
            })
        finally:
            _release_repo_mutation_lock(lock_fh)
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
        return JSONResponse(_managed_update_payload(fetch=False, include_tags=False))
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
        return JSONResponse(payload)
    except Exception as exc:
        return json_exception(exc)


def _respawn_workers_after_failed_update() -> None:
    """Revive workers when an update aborts after they were stopped (no restart follows)."""
    try:
        from supervisor.workers import ensure_worker_pool_started, open_repo_writer_admission

        open_repo_writer_admission()
        ensure_worker_pool_started(allow_disabled_restart=True)
    except Exception:
        log.warning("update_apply: failed to respawn workers after aborted update", exc_info=True)


async def api_update_preflight(_request: Request) -> JSONResponse:
    """Plan the managed update as a REAL 3-way merge (P2). Does NOT touch the live
    worktree/branch/index (it fetches + merges in an isolated temp worktree), so the UI
    can present the right staged choice (auto / assisted / manual)."""
    try:
        from supervisor.update_merge import plan_managed_update_merge

        plan = await asyncio.to_thread(plan_managed_update_merge, fetch=True)
        return JSONResponse({"merge_plan": plan})
    except Exception as exc:
        return json_exception(exc)


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


def _quiesce_repo_writers(reason: str) -> list[str]:
    """Close new writers, drain in-process turns, then prove the pool stopped."""
    from supervisor.git_ops import DRIVE_ROOT
    from supervisor.workers import (
        close_repo_writer_admission,
        drain_repo_writers,
        kill_workers_for_update,
        open_repo_writer_admission,
    )

    close_repo_writer_admission(f"managed_update:{reason}")
    blocked = drain_repo_writers()
    if blocked:
        open_repo_writer_admission()
        return [f"active:{label}" for label in blocked]
    survivors = kill_workers_for_update(
        result_reason="Task interrupted by an owner-requested managed update.",
        terminal_status="interrupted",
    )
    if survivors:
        return survivors
    try:
        from ouroboros.tools.services import kill_all_services

        stopped = kill_all_services(DRIVE_ROOT, wait=True, include_keep_alive=True)
    except Exception as exc:
        return [f"services:{type(exc).__name__}: {exc}"]
    failed = [
        str(item.get("service_id") or item.get("name") or "unknown")
        for item in stopped
        if isinstance(item, dict)
        and (item.get("stop_failed") or item.get("state") == "running" or item.get("lifecycle") == "running")
    ]
    try:
        from ouroboros.process_custody import quiesce_custodied_services

        _custody_ok, custody_blockers = quiesce_custodied_services(DRIVE_ROOT)
    except Exception as exc:
        custody_blockers = [f"custody_ledger:{type(exc).__name__}: {exc}"]
    return [f"service:{label}" for label in failed] + custody_blockers


def _fence_failure(blockers: list[str], stash_note: str = "") -> JSONResponse:
    restart_required = not all(str(item).startswith("active:") for item in blockers)
    return JSONResponse(
        {
            "error": (
                "Could not prove every repository writer stopped. The repository was not "
                "changed, but runtime shutdown may be incomplete."
            ),
            "reason": "update_writer_fence_blocked",
            "blockers": blockers,
            "restart_required": restart_required,
            **({"stash_note": stash_note} if stash_note else {}),
        },
        status_code=409,
    )


def _rollback_fenced_update(reason: str, error: str, **extra: Any) -> JSONResponse:
    from supervisor.update_merge import mark_update_tx_gate_blocked, rollback_managed_update

    ok, message = rollback_managed_update(reason)
    if ok:
        _respawn_workers_after_failed_update()
        return JSONResponse(
            {"error": error, "rolled_back": True, "rollback": message, **extra},
            status_code=409,
        )
    mark_update_tx_gate_blocked(reason, message)
    return JSONResponse(
        {
            "error": error,
            "rolled_back": False,
            "rollback": message,
            "restart_required": True,
            **extra,
        },
        status_code=500,
    )


def _restart_response(request: Request, *, strategy: str, plan: dict) -> JSONResponse:
    try:
        restarting = _request_restart(request)
    except Exception as exc:
        log.warning("managed update landed but restart request failed", exc_info=True)
        restarting = False
        restart_error = f"{type(exc).__name__}: {exc}"
    else:
        restart_error = "restart callback is unavailable" if not restarting else ""
    if not restarting:
        return JSONResponse(
            {
                "status": "restart_required",
                "error": restart_error,
                "strategy": strategy,
                "merge_plan": plan,
            }
        )
    return JSONResponse(
        {"status": "ok", "restarting": True, "strategy": strategy, "merge_plan": plan}
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
    restore the exact stash entry (marker-guarded — a crash between the stash
    apply and its drop must not let boot's replay wipe the already-restored
    copy) and clear the tx. Returns a disclosure note ("" when clean)."""
    from supervisor.update_merge import clear_update_tx, restore_stash_with_marker

    note = restore_stash_with_marker(tx, context)
    if not clear_update_tx():
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
    # Affordability floor: a resolution that cannot buy even ONE full triad+scope
    # review wave would mutate the live tree into a conflicted merge and then
    # stall mid-review. "One full wave" is priced HONESTLY at the review packs'
    # own worst-case caps — the shared 920K-token input SSOT per API row, the
    # triad's default output reserve, and the scope reviewer's 100K output
    # reserve — with the shared reservation math (agent-session rows ride
    # subscriptions, not USD budget); fail-open on estimator errors, mirroring
    # review_wave_admission's own contract.
    admission = {"fits": True}
    try:
        from ouroboros.reviewer_slot_config import commit_scope_rows, commit_triad_rows
        from ouroboros.tools.review_helpers import REVIEW_PROMPT_TOKEN_BUDGET
        from ouroboros.usage_accounting import review_wave_admission

        # Native-retrieving actor rows (subagent_id + api route) are priced at
        # the SAME one-pack-call convention as packet rows: their true worst
        # case is bounded by the episode's own rails (round cap x transcript
        # cap) and can exceed this estimate, but the typical episode is
        # pack-sized or smaller, and this floor is an explicitly fail-open
        # affordability heuristic — over-refusing assisted updates on a
        # theoretical ceiling would cost more than it protects.
        triad_models = [
            row.target_id for row in commit_triad_rows()
            if not row.is_session and row.target_id
        ]
        scope_models = [
            row.target_id for row in commit_scope_rows()
            if not row.is_session and row.target_id
        ]
        prompt_chars_cap = int(REVIEW_PROMPT_TOKEN_BUDGET) * 4
        estimated_total = 0.0
        any_estimate = False
        unpriced_total = 0
        for models, max_out in ((triad_models, 65_536), (scope_models, 100_000)):
            if not models:
                continue
            part = review_wave_admission(
                root_task_id="managed-update-admission",
                models=models,
                prompt_chars=prompt_chars_cap,
                max_completion_tokens=max_out,
                remaining_usd_override=float(remaining),
            )
            part_estimate = part.get("estimated_wave_usd")
            unpriced_total += int(part.get("unpriced_slots") or 0)
            if part_estimate is not None:
                estimated_total += float(part_estimate)
                any_estimate = True
            else:
                # The estimator failed open for this whole surface: every one of
                # its slots is unknown, not silently zero.
                unpriced_total += len(models)
        session_slots = sum(
            1 for row in (*commit_triad_rows(), *commit_scope_rows()) if row.is_session
        )
        if any_estimate:
            admission = {
                "fits": estimated_total <= float(remaining) + 1e-9,
                "estimated_wave_usd": round(estimated_total, 6),
                "remaining_usd": float(remaining),
                "unpriced_slots": unpriced_total,
                "session_slots": session_slots,
            }
        if admission.get("fits", True) and (unpriced_total or (session_slots and not any_estimate)):
            # An ADMITTED wave with unknowable parts must not read as a fully
            # priced estimate later (P1: represent the gap) — one durable line.
            try:
                from supervisor.git_ops import DRIVE_ROOT as _dr
                from ouroboros.utils import append_jsonl as _aj, utc_now_iso as _n

                _aj(_dr / "logs" / "supervisor.jsonl", {
                    "ts": _n(), "type": "managed_update_wave_floor_partial_unknown",
                    "estimated_wave_usd": admission.get("estimated_wave_usd"),
                    "unpriced_slots": unpriced_total, "session_slots": session_slots,
                    "remaining_usd": float(remaining),
                })
            except Exception:
                log.debug("wave-floor partial-unknown event write failed", exc_info=True)
    except Exception:
        log.debug("assisted admission wave estimate failed open", exc_info=True)
        admission = {"fits": True}
        try:
            from supervisor.git_ops import DRIVE_ROOT as _dr2
            from ouroboros.utils import append_jsonl as _aj2, utc_now_iso as _n2

            _aj2(_dr2 / "logs" / "supervisor.jsonl", {
                "ts": _n2(), "type": "managed_update_wave_floor_estimator_failed",
                "remaining_usd": float(remaining),
            })
        except Exception:
            log.debug("estimator-failure event write failed", exc_info=True)
    if not admission.get("fits", True):
        note = _unwind_stashed_update(tx, "assisted_admission_failed")
        _respawn_workers_after_failed_update()
        estimated = admission.get("estimated_wave_usd")
        try:
            from supervisor.git_ops import DRIVE_ROOT
            from ouroboros.utils import append_jsonl, utc_now_iso as _now

            append_jsonl(DRIVE_ROOT / "logs" / "supervisor.jsonl", {
                "ts": _now(), "type": "managed_update_wave_floor_refused",
                "estimated_wave_usd": estimated, "remaining_usd": admission.get("remaining_usd"),
            })
        except Exception:
            log.debug("wave-floor refusal event write failed", exc_info=True)
        return JSONResponse(
            {"error": (
                "Assisted update needs enough model budget for at least one full "
                f"review wave ({'at least ' if admission.get('unpriced_slots') else ''}≈${estimated} "
                "estimated at the review packs' worst-case caps for the configured reviewer panel"
                + (f"; {admission['unpriced_slots']} slot(s) unpriced" if admission.get("unpriced_slots") else "")
                + f", ${round(float(remaining), 2)} remaining); nothing was changed."
            ),
             "estimated_wave_usd": estimated,
             "remaining_usd": admission.get("remaining_usd"),
             "unpriced_slots": admission.get("unpriced_slots", 0),
             **({"stash_note": note} if note else {})},
            status_code=409,
        )

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
            "the merge was staged but its resolver worker could not start",
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
    ok, msg = apply_managed_merge_update(branch, merge_commit)
    if not ok:
        return _rollback_fenced_update(
            "merge_apply_failed", f"merge apply failed: {msg}"
        )
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
        blockers = _quiesce_repo_writers("smart")
        if blockers:
            return _fence_failure(blockers)
        # Stash-first (Q9): the owner's dirty + untracked work moves to a stash
        # BEFORE the authoritative replan, so the final plan — conflict inventory
        # and lane choice included — describes the exact clean tree the merge
        # will actually run on.
        from supervisor.git_ops import BRANCH_DEV as _branch_dev

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
        log.warning("managed smart update failed after writer fence", exc_info=True)
        from supervisor.update_merge import active_update_tx as _active_tx

        if _active_tx():
            return _rollback_fenced_update(
                "smart_update_exception",
                f"managed update failed: {type(exc).__name__}: {exc}",
            )
        _respawn_workers_after_failed_update()
        return json_exception(exc)
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


def _apply_replace_recovery_fenced(
    request: Request,
    *,
    expected_base_sha: str,
    expected_target_sha: str,
) -> JSONResponse:
    import uuid

    from supervisor.git_ops import (
        BRANCH_DEV,
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
        blockers = _quiesce_repo_writers("replace_recovery")
        if blockers:
            return _fence_failure(blockers)
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
        write_update_tx(tx)
        _write_update_intent(dict(payload["update_intent"]))
        try:
            checkout_ok, checkout_msg = checkout_and_reset(
                BRANCH_DEV,
                reason="ui_update_apply",
                unsynced_policy="rescue_and_reset",
            )
        except Exception as exc:
            return _rollback_fenced_update(
                "replace_checkout_exception", f"recovery checkout failed: {exc}", **payload
            )
        if not checkout_ok:
            return _rollback_fenced_update(
                "replace_checkout_failed", f"recovery checkout failed: {checkout_msg}", **payload
            )
        tx["phase"] = "pending_boot_smoke"
        write_update_tx(tx)
        smoke = update_restart_smoke()
        if not smoke.get("ok"):
            return _rollback_fenced_update(
                "replace_pre_restart_smoke_failed", "pre-restart smoke failed", smoke=smoke
            )
        tx["pre_restart_smoke"] = "passed"
        write_update_tx(tx)
        return _restart_response(request, strategy="replace", plan=plan2)
    except Exception as exc:
        log.warning("managed replace recovery failed after writer fence", exc_info=True)
        if active_update_tx():
            return _rollback_fenced_update(
                "replace_update_exception",
                f"managed recovery failed: {type(exc).__name__}: {exc}",
            )
        _respawn_workers_after_failed_update()
        return json_exception(exc)
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
