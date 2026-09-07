"""Events that change the runtime's posture rather than a single task's state.

Stable-branch promotion, the evolution and consciousness toggles, the owner's
injected message, the deep self-review request, and task cancellation - each
one an instruction about the runtime, not a report from a worker.
"""

from __future__ import annotations

import logging
from typing import Any, Dict
from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)


def _handle_deep_self_review_request(evt: Dict[str, Any], ctx: Any) -> None:
    ctx.queue_deep_self_review_task(
        reason=str(evt.get("reason") or "agent_self_review"),
        model=str(evt.get("model") or ""),
    )


def _handle_promote_to_stable(evt: Dict[str, Any], ctx: Any) -> None:
    import subprocess as sp

    from supervisor.git_ops import promote_branch_exact
    from supervisor.update_merge import (
        acquire_update_lock,
        active_update_tx,
        release_update_lock,
    )

    target = ctx.BRANCH_DEV
    evolution_claim = evt.get("evolution_claim")
    if isinstance(evolution_claim, dict):
        commit_sha = str(evolution_claim.get("commit_sha") or "").strip()
        if not commit_sha:
            authority = {"ok": False, "reason": "commit_receipt_missing"}
        else:
            from supervisor.evolution_lifecycle import check_evolution_authority

            authority = check_evolution_authority(
                campaign_id=str(evolution_claim.get("campaign_id") or ""),
                transaction_id=str(evolution_claim.get("transaction_id") or ""),
                task_id=str(evolution_claim.get("task_id") or ""),
                commit_sha=commit_sha,
            )
        if not authority.get("ok"):
            st = ctx.load_state()
            if st.get("owner_chat_id"):
                ctx.send_with_budget(
                    int(st["owner_chat_id"]),
                    "❌ Evolution promotion refused: the exact reviewed campaign claim "
                    f"is no longer valid ({authority.get('reason') or 'unknown'}).",
                )
            return
        try:
            dev_sha = sp.run(
                ["git", "rev-parse", ctx.BRANCH_DEV],
                cwd=str(ctx.REPO_DIR), capture_output=True, text=True, check=True,
            ).stdout.strip()
        except Exception:
            dev_sha = ""
        if dev_sha != commit_sha:
            st = ctx.load_state()
            if st.get("owner_chat_id"):
                ctx.send_with_budget(
                    int(st["owner_chat_id"]),
                    "❌ Evolution promotion refused: the development branch no longer "
                    "matches the reviewed commit receipt.",
                )
            return
        # Promote the exact reviewed SHA (TOCTOU-safe: the dev branch may move
        # between the check above and the ref update inside promote_branch_exact).
        target = commit_sha

    lock_fh = None
    try:
        lock_fh = acquire_update_lock()
        if active_update_tx():
            ok, result = False, {"error": "a managed update transaction is still active"}
        else:
            ok, result = promote_branch_exact(
                target, ctx.BRANCH_STABLE, push_remote=True,
                repo_dir=str(ctx.REPO_DIR),
            )
    except RuntimeError as exc:
        ok, result = False, {"error": str(exc)}
    finally:
        if lock_fh is not None:
            release_update_lock(lock_fh)
    if not ok:
        st = ctx.load_state()
        if st.get("owner_chat_id"):
            ctx.send_with_budget(
                int(st["owner_chat_id"]),
                f"❌ Failed to promote to stable: {result.get('error') or 'unknown error'}",
            )
        return

    st = ctx.load_state()
    if st.get("owner_chat_id"):
        new_sha = str(result["sha"])
        if result.get("remote_pushed"):
            remote_status = " (pushed to origin)"
        elif result.get("remote_error"):
            remote_status = f" (local only; remote push failed: {result['remote_error']})"
        else:
            remote_status = ""
        ctx.send_with_budget(
            int(st["owner_chat_id"]),
            f"✅ Promoted: {ctx.BRANCH_DEV} → {ctx.BRANCH_STABLE} ({new_sha[:8]}){remote_status}",
        )


def _handle_cancel_task(evt: Dict[str, Any], ctx: Any) -> None:
    """Drive one agent-requested cancel through custody — TYPED outcome end to end.

    Custody publishes the settled truth itself: a cancelled child's
    ``task_summary``/salvage row lands in the task's own thread through the
    terminal-delivery seam, an already-settled child keeps the result it
    already published, and the calling agent received the typed tool result.
    A second host acknowledgement here (#624) reached the global owner chat as
    an untyped assistant row with no task identity — duplicate presentation in
    Main Chat, and for a pending-drop race it misstated the causal history.
    Only the FAILED outcome still speaks: the task is live, the intent stays
    open, and the owner needs the typed incident."""
    task_id = str(evt.get("task_id") or "").strip()
    requested_task_id = str(evt.get("requested_task_id") or "").strip()
    display_task_id = requested_task_id or task_id
    st = ctx.load_state()
    owner_chat_id = st.get("owner_chat_id")
    from supervisor.queue import CANCEL_FAILED, CANCEL_NOT_FOUND, drive_cancel_intent_scope

    outcome = drive_cancel_intent_scope(task_id) if task_id else CANCEL_NOT_FOUND
    if not owner_chat_id or outcome != CANCEL_FAILED:
        return
    incident_meta = {
        "task_incident": "cancellation_fault",
        "toast_once": f"{display_task_id or 'unknown'}:cancellation_fault",
    }
    if task_id and display_task_id != task_id:
        incident_meta["cancel_physical_task_id"] = task_id
    ctx.send_with_budget(
        int(owner_chat_id),
        f"❌ cancel {display_task_id or '?'} did not settle — the task is still live; "
        "the durable cancel intent stays open and the supervisor watchdog retries (event)",
        is_progress=True,
        task_id=display_task_id,
        progress_meta=incident_meta,
    )


def _handle_toggle_evolution(evt: Dict[str, Any], ctx: Any) -> None:
    """Toggle evolution mode from LLM tool call."""
    enabled = bool(evt.get("enabled"))
    if enabled:
        from supervisor.evolution_lifecycle import evolution_block_reason, start_evolution_campaign

        block = evolution_block_reason()
        if block:
            st = ctx.load_state()
            if st.get("owner_chat_id"):
                ctx.send_with_budget(int(st["owner_chat_id"]), block)
            return
        # GR4-6: clear the durable owner-stop flag BEFORE the campaign is
        # minted. The old order (campaign first, flag cleared in a later state
        # write) left a window where the owner-stop backstop — fired by an old
        # evolution task settling — read flag=True + campaign=active and closed
        # the FRESH campaign. This clear is owner-authorized (the owner is
        # explicitly starting evolution). GR5-1: the prior value is captured in
        # the same locked write so a failed start can restore it.
        from supervisor.state import update_state as _update_state

        _prior_owner_stop = {"value": False}

        def _clear_owner_stop(live: Dict[str, Any]) -> None:
            _prior_owner_stop["value"] = bool(live.get("evolution_owner_stopped"))
            live["evolution_owner_stopped"] = False

        _update_state(_clear_owner_stop)
        try:
            if not start_evolution_campaign(str(evt.get("objective") or ""), source="agent_tool"):
                raise RuntimeError("campaign write was refused")
        except Exception:
            log.warning("Failed to start evolution campaign from agent tool", exc_info=True)
            # GR5-1: the start FAILED, so the pre-mint clear was not an
            # owner-authorized state change after all. Restore the CAPTURED
            # prior value — leaving it cleared would let the post-task
            # promotion pipeline (apply_pending_request reads the flag)
            # autonomously re-arm evolution the owner believes is off, and an
            # unconditional True would invent a stop that never happened.
            _update_state(lambda live: live.__setitem__(
                "evolution_owner_stopped", _prior_owner_stop["value"]))
            st = ctx.load_state()
            if st.get("owner_chat_id"):
                ctx.send_with_budget(
                    int(st["owner_chat_id"]),
                    "🧬 Evolution stayed OFF: campaign state could not be created.",
                )
            return
    from supervisor.state import update_state

    def _toggle_evolution(live: Dict[str, Any]) -> None:
        live["evolution_mode_enabled"] = enabled
        if enabled:
            live["evolution_consecutive_failures"] = 0
        # Owner stop is AUTHORITATIVE against the post-task pipeline (mirrors /evolve): set
        # the durable evolution_owner_stopped flag on disable, clear it on enable (this is an
        # owner-authorized clear). This is what apply_pending_request reads to refuse re-arm.
        live["evolution_owner_stopped"] = (not enabled)
        # Symmetry with the owner /evolve path: an explicit toggle must not inherit a
        # stale post-task one-shot autostop that would disable the campaign after one cycle.
        live["post_task_autostop"] = False

    st = update_state(_toggle_evolution)
    stop_lines: list = []
    stop_incomplete = False
    if not enabled:
        # Cancel live evolution work BEFORE the terminal campaign close below:
        # complete_evolution_campaign runs the per-cycle worktree cleanup, which skips
        # while a task still holds the shared worktree — so the running cycle must be gone
        # first. PENDING evolution tasks go through the SAME durable intent + typed
        # custody (GR2-13) — the old in-place prune left them with no intent, no
        # terminal result and no task_done, and intent-write failures vanished from
        # the caller's view while Evolution was still declared stopped.
        from supervisor.queue import evolution_stop_report, stop_evolution_tasks
        from ouroboros.post_task_evolution import drop_pending_request
        from supervisor import state as _evo_state

        drop_pending_request(_evo_state.DRIVE_ROOT)
        stopped = stop_evolution_tasks("disabled via agent tool")
        ctx.sort_pending()
        ctx.persist_queue_snapshot(reason="evolve_off_via_tool")
        stop_lines, stop_incomplete = evolution_stop_report(stopped)
    try:
        from supervisor.evolution_lifecycle import complete_evolution_campaign

        if not enabled:
            if stop_incomplete:
                # GR3-3: an INCOMPLETE stop leaves the campaign OPEN — the
                # durable evolution_owner_stopped flag already blocks new
                # cycles, and the settle-time owner-stop backstop below
                # (_close_campaign_after_owner_stop) closes the campaign once
                # the live task settles. Closing it now would declare a clean
                # terminal over still-live evolution work.
                log.warning(
                    "Evolution stop is incomplete; campaign left open for the "
                    "settle-time owner-stop backstop",
                )
            else:
                # Terminal close (not a resumable pause), so a later /evolve start mints fresh.
                complete_evolution_campaign("disabled via agent tool", status="stopped")
    except Exception:
        log.debug("Failed to update evolution campaign toggle state", exc_info=True)
    if st.get("owner_chat_id"):
        owner_chat = int(st["owner_chat_id"])
        for line in stop_lines:
            ctx.send_with_budget(owner_chat, line)
        if enabled:
            state_str = "ON"
        elif stop_incomplete:
            state_str = ("OFF (mode disabled) — but the stop is INCOMPLETE: see the "
                         "still-live task(s) above. The campaign stays open until "
                         "they settle. Post-task auto-evolution stays paused until "
                         "/evolve start")
        else:
            state_str = "OFF — post-task auto-evolution also paused until /evolve start"
        ctx.send_with_budget(owner_chat, f"🧬 Evolution: {state_str} (via agent tool)")


def _handle_toggle_consciousness(evt: Dict[str, Any], ctx: Any) -> None:
    """Toggle background consciousness from LLM tool call."""
    from supervisor.state import update_state
    action = str(evt.get("action") or "status")
    if action in ("start", "on"):
        result = ctx.consciousness.start()
        update_state(lambda st: st.__setitem__("bg_consciousness_enabled", True))
    elif action in ("stop", "off"):
        result = ctx.consciousness.stop()
        update_state(lambda st: st.__setitem__("bg_consciousness_enabled", False))
    else:
        status = "running" if ctx.consciousness.is_running else "stopped"
        result = f"Background consciousness: {status}"
    st = ctx.load_state()
    if st.get("owner_chat_id"):
        ctx.send_with_budget(int(st["owner_chat_id"]), f"🧠 {result}")


def _handle_owner_message_injected(evt: Dict[str, Any], ctx: Any) -> None:
    """Log owner injections so health checks can detect duplicate processing."""
    try:
        ctx.append_jsonl(ctx.DRIVE_ROOT / "logs" / "events.jsonl", {
            "ts": evt.get("ts", utc_now_iso()),
            "type": "owner_message_injected",
            "task_id": evt.get("task_id", ""),
            "text": evt.get("text", ""),
        })
    except Exception:
        log.warning("Failed to log owner_message_injected event", exc_info=True)
