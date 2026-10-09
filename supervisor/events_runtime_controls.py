"""Events that change the runtime's posture rather than a single task's state.

Stable-branch promotion, the evolution and consciousness toggles, the owner's
injected message, the deep self-review request, and task cancellation - each
one an instruction about the runtime, not a report from a worker.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict
from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)
_cancel_event_lock = threading.Lock()
_cancel_events_in_flight: set[tuple[str, str]] = set()


def _handle_deep_self_review_request(evt: Dict[str, Any], ctx: Any) -> None:
    from ouroboros.consciousness_authority import consciousness_origin_metadata

    ctx.queue_deep_self_review_task(
        reason=str(evt.get("reason") or "agent_self_review"),
        model=str(evt.get("model") or ""),
        origin=consciousness_origin_metadata(evt),
        reviewer=str(evt.get("reviewer") or ""),
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
                    role="system", system_type="promotion_notice")
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
                    role="system", system_type="promotion_notice")
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
                role="system", system_type="promotion_notice")
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
            role="system", system_type="promotion_notice")


def _handle_cancel_task(evt: Dict[str, Any], ctx: Any) -> None:
    """Dispatch the existing custody driver without holding supervisor intake."""
    task_id = str(evt.get("task_id") or "").strip()
    if not task_id:
        return
    key = (str(ctx.DRIVE_ROOT), task_id)
    with _cancel_event_lock:
        if key in _cancel_events_in_flight:
            return  # The durable intent also carries any stronger repeat request.
        _cancel_events_in_flight.add(key)

    def drive() -> None:
        try:
            _drive_cancel_task_event(evt, ctx)
        except Exception:
            log.warning("Cancel event remains with the durable watchdog for %s", task_id, exc_info=True)
        finally:
            with _cancel_event_lock:
                _cancel_events_in_flight.discard(key)

    try:
        threading.Thread(target=drive, name=f"cancel-task-{task_id}", daemon=True).start()
    except Exception:
        with _cancel_event_lock:
            _cancel_events_in_flight.discard(key)
        log.warning("Could not dispatch cancel event for %s; durable watchdog retains it", task_id, exc_info=True)


def _drive_cancel_task_event(evt: Dict[str, Any], ctx: Any) -> None:
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
        role="system", system_type="cancellation_notice")


EVOLUTION_CONTROL_KEYS = ("evolution_mode_enabled", "evolution_owner_stopped",
                          "evolution_stop_source", "post_task_autostop")


def owner_evolution_stop_controls(reason: str) -> str:
    """The owner Stop's control half (#1307): the live latch and the durable campaign
    ``stop_intent`` first, then the state flags — each attempted whatever the other
    did, none waiting on cancellation. Returns "" or a disclosure of what did not
    persist (the Stop still holds in this process)."""
    from supervisor.evolution_lifecycle import record_evolution_stop_intent
    from supervisor.state import StateUnavailable, update_state

    # A failed campaign filesystem/lock operation cannot skip the independent state
    # decision or the caller's cancellation. The latch is set before either write.
    from supervisor.evolution_lifecycle import _STOP_LATCH

    _STOP_LATCH["stopped"] = True
    missing = []
    try:
        if not record_evolution_stop_intent("owner", reason):
            missing.append("campaign stop intent")
    except Exception as exc:
        missing.append(f"campaign stop intent ({type(exc).__name__})")

    def _owner_stop(live: Dict[str, Any]) -> None:
        live["evolution_mode_enabled"] = False
        live["evolution_owner_stopped"] = True
        live.pop("evolution_stop_source", None)  # an owner stop: no agent source may un-stick it
        live["post_task_autostop"] = False

    try:
        update_state(_owner_stop, confirm=EVOLUTION_CONTROL_KEYS)
    except (StateUnavailable, OSError) as exc:
        missing.append(f"runtime state ({getattr(exc, 'reason', type(exc).__name__)})")
    return ("" if not missing else
            f" The Stop holds in this process, but {' and '.join(missing)} did not persist.")


def _enable_evolution_controls(live: Dict[str, Any]) -> None:
    """The common state projection after either authorized campaign start."""
    live.update(evolution_mode_enabled=True, evolution_consecutive_failures=0,
                evolution_owner_stopped=False, post_task_autostop=False)
    live.pop("evolution_stop_source", None)


def owner_evolution_start(objective: str, *, source: str = "owner_chat", origin: Any = None) -> str:
    """The owner's authorized start: clears the owner stop (GR4-6: BEFORE the campaign
    is minted), starts the campaign (which alone clears a recorded stop intent), then
    enables the projection. Returns "" on success or the owner-visible refusal."""
    from supervisor.evolution_lifecycle import evolution_block_reason, start_evolution_campaign
    from supervisor.state import StateUnavailable, control_value, mark_unconfirmed, update_state

    block = evolution_block_reason()
    if block:
        return block
    prior: Dict[str, Any] = {}

    def _clear_owner_stop(live: Dict[str, Any]) -> None:
        prior["known"], prior["value"] = control_value(live, "evolution_owner_stopped")
        live["evolution_owner_stopped"] = False
        live.pop("evolution_stop_source", None)

    def _restore(live: Dict[str, Any]) -> None:
        # GR5-1: a failed start restores exactly what it cleared — an unknown stays unknown.
        live["evolution_owner_stopped"] = prior.get("value")
        if not prior.get("known"):
            mark_unconfirmed(live, "evolution_owner_stopped")

    try:
        update_state(_clear_owner_stop, confirm=("evolution_owner_stopped", "evolution_stop_source"))
    except StateUnavailable as exc:
        return f"🧬 Evolution stayed OFF: runtime state is unavailable ({exc.reason}); no campaign was started."
    try:
        started = start_evolution_campaign(objective, source=source, **({"origin": origin} if origin else {}))
    except Exception:
        log.warning("Failed to start evolution campaign", exc_info=True)
        started = {}
    try:
        if not started:
            update_state(_restore)
            return "🧬 Evolution stayed OFF: campaign state could not be created."
        update_state(_enable_evolution_controls, confirm=EVOLUTION_CONTROL_KEYS)
    except StateUnavailable as exc:
        owner_evolution_stop_controls("owner Start could not persist activation")
        return f"🧬 Evolution did not turn on: runtime state became unavailable ({exc.reason})."
    return ""


def _handle_toggle_evolution(evt: Dict[str, Any], ctx: Any) -> None:
    """Toggle evolution mode from an LLM tool call (or an owner-sourced event).

    Owner decision В12: ``/evolve off`` is STICKY against the agent tool. Only an
    event with owner provenance (``source == "owner_chat"``) may clear a stop the
    OWNER placed (``/evolve off``, panic, an owner-sourced toggle — every stop without
    an ``evolution_stop_source`` of ``agent_tool``); an ``agent_tool`` enable against
    such a stop is refused with the same typed shape as the light-mode block. A stop
    the agent placed itself stays undoable by the agent, as it always was. An UNKNOWN
    stop (state unavailable or recovered) is never read as the agent's own (#1307).
    """
    from supervisor.state import StateUnavailable, control_value, update_state

    enabled = bool(evt.get("enabled"))
    owner_sourced = str(evt.get("source") or "") == "owner_chat"
    st = ctx.load_state()
    owner_chat = int(st.get("owner_chat_id") or 0)  # a notice route (display), not an authority

    def _notify(text: str) -> None:
        if owner_chat:
            ctx.send_with_budget(owner_chat, text, role="system", system_type="evolution_notice")

    if enabled:
        from supervisor.evolution_lifecycle import evolution_block_reason, evolution_stop_reason, start_evolution_campaign
        from ouroboros.consciousness_authority import consciousness_origin_metadata

        origin = consciousness_origin_metadata(evt)
        if owner_sourced:
            refusal = owner_evolution_start(str(evt.get("objective") or ""), origin=origin)
            if refusal:
                _notify(refusal)
                return
        else:
            stopped_known, stopped = control_value(st, "evolution_owner_stopped")
            source_known, stop_source = control_value(st, "evolution_stop_source")
            agent_may_clear = stopped_known and source_known and stop_source == "agent_tool"
            block = evolution_block_reason()
            if not block and (not stopped_known or (stopped and not agent_may_clear) or evolution_stop_reason()):
                block = (
                    "🧬 Evolution stayed OFF: the owner stopped evolution (/evolve off), and that stop "
                    "is sticky against toggle_evolution. Only the owner's /evolve start re-arms it; "
                    "no campaign was started." if stopped_known else
                    "🧬 Evolution stayed OFF: whether the owner stopped evolution is unknown right now "
                    "(runtime state unavailable); only the owner's /evolve start re-arms it."
                )
            if block:
                _notify(block)
                return
            try:
                if not start_evolution_campaign(str(evt.get("objective") or ""), source="agent_tool",
                                                **({"origin": origin} if origin else {})):
                    raise RuntimeError("campaign write was refused")
            except Exception:
                log.warning("Failed to start evolution campaign from agent tool", exc_info=True)
                _notify("🧬 Evolution stayed OFF: campaign state could not be created.")
                return
    persisted = ""
    if enabled and not owner_sourced:
        try:
            update_state(_enable_evolution_controls, confirm=EVOLUTION_CONTROL_KEYS)
        except StateUnavailable as exc:
            persisted = f" — not persisted: runtime state unavailable ({exc.reason})"
    elif not enabled and owner_sourced:
        persisted = owner_evolution_stop_controls("disabled via owner toggle")
    elif not enabled:
        def _agent_stop(live: Dict[str, Any]) -> None:
            # The stop remembers who placed it, and the key never outlives the stop it
            # describes: an owner stop already standing keeps the owner's (absent) source.
            owner_stop_stands = bool(live.get("evolution_owner_stopped")) and live.get("evolution_stop_source") != "agent_tool"
            live.update(evolution_mode_enabled=False, evolution_owner_stopped=True, post_task_autostop=False)
            if owner_stop_stands:
                live.pop("evolution_stop_source", None)
            else:
                live["evolution_stop_source"] = "agent_tool"

        try:
            update_state(_agent_stop, confirm=("evolution_mode_enabled", "post_task_autostop"))
        except StateUnavailable as exc:
            persisted = f" — not persisted: runtime state unavailable ({exc.reason})"
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
        from ouroboros.post_task_evolution import drop_pending_request
        from supervisor import state as _evo_state
        from supervisor.queue import evolution_stop_report, stop_evolution_tasks

        drop_pending_request(_evo_state.DRIVE_ROOT)
        stopped = stop_evolution_tasks("disabled via agent tool")
        ctx.sort_pending()
        ctx.persist_queue_snapshot(reason="evolve_off_via_tool")
        stop_lines, stop_incomplete = evolution_stop_report(stopped)
    try:
        from supervisor.evolution_lifecycle import complete_evolution_campaign

        if not enabled:
            if stop_incomplete:
                # GR3-3: an INCOMPLETE stop leaves the campaign OPEN — the durable
                # stop (state flag and/or campaign stop_intent) already blocks new
                # cycles, and the settle-time owner-stop backstop
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
    for line in stop_lines:
        _notify(line)
    if enabled:
        state_str = "ON"
    elif stop_incomplete:
        state_str = ("OFF (mode disabled) — but the stop is INCOMPLETE: see the "
                     "still-live task(s) above. The campaign stays open until "
                     "they settle. Post-task auto-evolution stays paused until "
                     "/evolve start")
    else:
        state_str = "OFF — post-task auto-evolution also paused until /evolve start"
    _notify(f"🧬 Evolution: {state_str} (via agent tool){persisted}")


def _handle_toggle_consciousness(evt: Dict[str, Any], ctx: Any) -> None:
    """Toggle background consciousness from LLM tool call."""
    action = str(evt.get("action") or "status")
    if action in ("start", "on", "stop", "off"):
        on = action in ("start", "on")
        result = ctx.consciousness.start() if on else ctx.consciousness.stop()
        result = f"{result}{persist_consciousness_choice(on)}"
    else:
        # Status is answered to its caller by the tool itself; reading it is not
        # an event the owner is told about (#1324). Only start/stop publish.
        return
    st = ctx.load_state()
    if st.get("owner_chat_id"):
        ctx.send_with_budget(int(st["owner_chat_id"]), f"🧠 {result}", role="system", system_type="consciousness_notice")


def persist_consciousness_choice(enabled: bool) -> str:
    """Persist a background-consciousness decision as a KNOWN control; "" or a
    disclosure that it holds only in this process (#1307)."""
    from supervisor.state import StateUnavailable, update_state

    try:
        update_state(lambda st: st.__setitem__("bg_consciousness_enabled", bool(enabled)),
                     confirm=("bg_consciousness_enabled",))
    except StateUnavailable as exc:
        return f" (not persisted: runtime state unavailable, {exc.reason})"
    return ""


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
