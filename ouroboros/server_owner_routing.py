"""Where one owner message goes.

Attachment staging into the addressed task's artifact store, the single
unambiguous mailbox delivery, the typed bubble-free routing receipt, and the
dispatch that hands everything else to the decision lane. The ``/evolve off``
stop transaction lives here too: it is the one owner command whose effect is a
message-time transaction rather than a process-lifecycle change.
"""

import base64
import os
import pathlib
import threading
import uuid
from typing import Any, Dict, Optional

from ouroboros.server_process import log
from ouroboros.server_routing_context import (
    _addressable_root_tasks,
    _decision_turn_metadata,
    _project_id_for_registered_chat,
    _reserved_project_for_chat,
    _scoped_task_metadata,
)


def _stage_mailbox_attachments(
    ctx: Any,
    task_id: str,
    task_metadata: Any,
    image_data: Any = None,
) -> tuple[str, list]:
    """Stage one routed turn's files into the existing task artifact store.

    Returns ``(attachment_note, staged_manifest)`` — the manifest is kept so a
    refused admission (the cancel-pending re-check inside the mailbox
    transaction) can remove exactly the files this call staged (GR2-9).
    """
    metadata = task_metadata if isinstance(task_metadata, dict) else {}
    uploads = list(metadata.get("chat_attachment_uploads") or [])
    temp_source: Optional[pathlib.Path] = None
    if image_data and not uploads:
        # Non-Web transports may carry an inline image rather than an uploaded
        # path. Materialise it only long enough for the canonical staging helper
        # to copy it into the addressed task's artifact store.
        try:
            raw = base64.b64decode(str(image_data[0] or ""), validate=True)
            if raw and len(raw) <= 50 * 1024 * 1024:
                mime = str(image_data[1] or "image/jpeg").lower()
                suffix = ".png" if "png" in mime else ".webp" if "webp" in mime else ".jpg"
                temp_source = pathlib.Path(ctx.DRIVE_ROOT) / "uploads" / f"routed-{uuid.uuid4().hex}{suffix}"
                temp_source.parent.mkdir(parents=True, exist_ok=True)
                with temp_source.open("xb") as handle:
                    handle.write(raw)
                    handle.flush()
                    os.fsync(handle.fileno())
                uploads.append({"path": str(temp_source), "label": "owner image"})
        except Exception:
            log.warning("Unable to stage routed inline image for task %s", task_id, exc_info=True)
    try:
        if not uploads:
            return "", []
        from ouroboros.artifacts import stage_task_attachments
        from ouroboros.gateway.tasks import _render_attachment_lines

        manifest = stage_task_attachments(ctx.DRIVE_ROOT, task_id, uploads)
        rendered = _render_attachment_lines(manifest)
        note = f"\n\n[ATTACHMENTS]\n{rendered}\n[END_ATTACHMENTS]" if rendered else ""
        return note, manifest
    finally:
        if temp_source is not None:
            try:
                temp_source.unlink(missing_ok=True)
            except OSError:
                log.debug("Unable to remove routed attachment staging source", exc_info=True)


def _route_project_chat_to_running_task(
    ctx: Any,
    chat_id: int,
    message: str,
    client_message_id: str = "",
    *,
    task_metadata: Any = None,
    image_data: Any = None,
) -> str:
    """Deliver a Project follow-up to the sole RUNNING/PENDING root mailbox.

    Multi-project (v6.32.0): a focused project room with exactly ONE active pooled
    task IS that task's context, so a follow-up is delivered to it as a TRANSPORT
    invariant (the loop drains the mailbox every round) — there is no routing CHOICE
    to make. But when the room has ZERO or MORE THAN ONE steerable task, picking a
    target is a JUDGMENT, and code must never make it mechanically (BIBLE P5 LLM-first,
    v6.34.0 WS1): this returns "" so the message flows to the decision turn, where the
    agent sees `current_chat.running_tasks` and chooses `steer_task` / `promote_chat_to_task`.
    Returns the delivered task id, or "" (no delivery — fall through to the decision lane).

    A chat is a project thread by REGISTRY membership, not a bare numeric range —
    large external-transport (Telegram-style) chat ids must not be misclassified and
    have their owner messages swallowed.
    """
    try:
        if not _project_id_for_registered_chat(ctx, chat_id):
            return ""
    except Exception:
        return ""
    try:
        steerable = _addressable_root_tasks(ctx, chat_id)
        # Exactly one candidate => unambiguous transport. Zero or many => a routing
        # decision the AGENT must make (P5/WS1), so do not deliver here.
        if len(steerable) != 1:
            return ""
        candidate = steerable[0]
        tid = str(candidate["task_id"])
        direct_agent = None
        direct_lock = None
        if candidate.get("direct_chat"):
            direct_agent = ctx.get_chat_agent()
            direct_lock = getattr(direct_agent, "_owner_message_admission_lock", None)
            if direct_lock is None:
                return ""
        task_obj: Dict[str, Any] = {}
        running = getattr(ctx, "RUNNING", {}).get(tid)
        if isinstance(running, dict):
            task_obj = running.get("task") if isinstance(running.get("task"), dict) else running
        if not task_obj:
            task_obj = next(
                (row for row in list(getattr(ctx, "PENDING", []) or []) if str(row.get("id") or "") == tid),
                {},
            )
        from ouroboros.owner_mailbox import write_owner_message
        from supervisor.queue import (
            ACCEPTANCE_FENCES,
            _queue_lock,
            _task_drive_for_task,
            persist_queue_snapshot,
        )

        # Active drive (child drive for forked/workspace tasks) — mirror
        # forward_to_worker / steer_task so the mailbox lands where the task
        # actually drains it, not the canonical root. A stable msg_id derived from
        # client_message_id makes this 1:1 delivery idempotent — a WebSocket retry of
        # the same message can't double-deliver (drain_owner_entries dedups by msg_id),
        # matching steer_task's contract.
        direct_lock_held = False
        queue_lock_held = False
        fence_generation_changed = False
        active_fence = None
        if direct_lock is not None:
            direct_lock.acquire()
            direct_lock_held = True
            if not (
                getattr(direct_agent, "_busy", False)
                and getattr(direct_agent, "_accepting_owner_messages", False)
                and str(getattr(direct_agent, "_current_task_id", "") or "") == tid
            ):
                direct_lock.release()
                direct_lock_held = False
                return ""
        task_drive = pathlib.Path(ctx.DRIVE_ROOT) if direct_lock_held else _task_drive_for_task(task_obj, tid)
        msg_id = f"{client_message_id}:{tid}" if client_message_id else None
        staged_manifest: list = []
        cancel_refused_in_txn = False

        def _drop_staged_inputs() -> None:
            # GR2-9: the admission was refused, so the files staged for this
            # message must not linger in the dying task's artifact store.
            if not staged_manifest:
                return
            try:
                from ouroboros.artifacts import remove_staged_attachments

                remove_staged_attachments(staged_manifest)
            except Exception:
                log.debug("staged-attachment cleanup failed for %s", tid, exc_info=True)

        try:
            # GR2-9 ordering: check cancellation BEFORE staging — the old order
            # copied the owner's files into the artifact store of a task whose
            # cancellation was already pending, then refused the message. The
            # cheap up-front check runs off the lock; the transactional
            # re-checks below still run and remove the staged inputs on refusal.
            from ouroboros.cancel_intents import cancel_pending

            if cancel_pending(ctx.DRIVE_ROOT, tid):
                log.info("Mailbox follow-up refused for %s: cancel pending (pre-staging)", tid)
                return ""
            attachment_note, staged_manifest = _stage_mailbox_attachments(
                ctx, tid, task_metadata, image_data,
            )
            if direct_lock_held:
                # AR2-6 (fable): the direct-agent lane used to skip the
                # cancel-pending admission check the queue lane makes below — a
                # direct turn whose cancellation is pending must not accept a
                # new owner message either. Same predicate, same honest
                # fall-through to the direct chat lane.
                if cancel_pending(ctx.DRIVE_ROOT, tid):
                    log.info("Mailbox follow-up refused for %s: cancel pending (direct lane)", tid)
                    _drop_staged_inputs()
                    return ""
            if not direct_lock_held:
                _queue_lock.acquire()
                queue_lock_held = True
                live_meta = getattr(ctx, "RUNNING", {}).get(tid)
                still_pending = any(
                    isinstance(row, dict) and str(row.get("id") or "") == tid
                    for row in list(getattr(ctx, "PENDING", []) or [])
                )
                if live_meta is None and not still_pending:
                    return ""
                # Phase A: a task whose cancellation is PENDING must not accept a
                # new owner message — same refusal the steer_task route makes,
                # checked inside this admission transaction. Falling through to
                # the direct lane is the honest outcome: the follow-up is
                # answered in chat instead of handed to a dying task.
                if cancel_pending(ctx.DRIVE_ROOT, tid):
                    log.info("Mailbox follow-up refused for %s: cancel pending", tid)
                    cancel_refused_in_txn = True
                    return ""
                fence_root = str(task_obj.get("root_task_id") or tid)
                active_fence = ACCEPTANCE_FENCES.get(fence_root)
                if isinstance(active_fence, dict) and str(active_fence.get("status") or "") == "sealed":
                    return ""
            if not write_owner_message(
                task_drive, f"{message}{attachment_note}", tid, msg_id=msg_id,
                client_surface=(
                    dict(task_metadata["client_surface"])
                    if isinstance(task_metadata, dict) and isinstance(task_metadata.get("client_surface"), dict)
                    else None
                ),
            ):
                return ""
            if direct_lock_held:
                direct_agent._owner_message_generation = int(
                    getattr(direct_agent, "_owner_message_generation", 0) or 0
                ) + 1
            else:
                if isinstance(active_fence, dict) and str(active_fence.get("status") or "") == "active":
                    active_fence["owner_message_generation"] = int(
                        active_fence.get("owner_message_generation") or 0
                    ) + 1
                    fence_generation_changed = True
        finally:
            if queue_lock_held:
                _queue_lock.release()
            if direct_lock_held:
                direct_lock.release()
            if cancel_refused_in_txn:
                # After the lock release: unlinking staged files is file I/O the
                # global queue lock should not wait on.
                _drop_staged_inputs()
        if fence_generation_changed:
            persist_queue_snapshot(reason="acceptance_fence_owner_message")
        return tid
    except Exception:
        log.debug("Mailbox follow-up routing failed; falling back to direct lane", exc_info=True)
    return ""


def _owner_evolution_stop(ctx: Any, chat_id: int) -> str:
    """The ``/evolve off`` stop transaction; returns the final status wording.

    Cancels live evolution work BEFORE the terminal campaign close:
    ``complete_evolution_campaign`` runs the per-cycle worktree cleanup, which
    skips while a task still holds the shared worktree — so the running cycle
    must be gone first. PENDING evolution tasks go through the SAME durable
    intent + typed custody (GR2-13); the old in-place prune left them with no
    intent, no terminal result and no ``task_done``, and a stop with still-live
    leftovers was declared clean.
    """
    stop_incomplete = False
    try:
        from supervisor.queue import evolution_stop_report, stop_evolution_tasks
        from ouroboros.post_task_evolution import drop_pending_request

        # Fast path: drop any queued post-task promotion so it cannot re-arm on
        # the next boot tick (the evolution_owner_stopped flag is the durable backstop).
        drop_pending_request(ctx.DRIVE_ROOT)
        stopped = stop_evolution_tasks("disabled via owner chat")
        ctx.sort_pending()
        ctx.persist_queue_snapshot(reason="evolve_off")
        stop_lines, stop_incomplete = evolution_stop_report(stopped)
        for line in stop_lines:
            ctx.send_with_budget(chat_id, line)
    except Exception:
        log.warning("Evolution stop transaction failed", exc_info=True)
        stop_incomplete = True
    try:
        from supervisor.evolution_lifecycle import complete_evolution_campaign

        if stop_incomplete:
            # GR3-3: an INCOMPLETE stop must not close the campaign — a terminal
            # "stopped" over still-live evolution work declares a clean ending
            # that did not happen. The campaign stays open; the durable
            # evolution_owner_stopped flag already blocks new cycles, and the
            # owner-stop backstop (supervisor/events.py, on the live task's own
            # settle) closes the campaign once nothing is live.
            log.warning(
                "Evolution stop is incomplete; campaign left open for the "
                "settle-time owner-stop backstop",
            )
        else:
            # Terminal close (not a resumable pause): /evolve start mints a FRESH
            # campaign rather than resurrecting this one.
            complete_evolution_campaign("disabled via owner chat", status="stopped")
    except Exception:
        log.warning("Failed to update evolution campaign state", exc_info=True)
    if stop_incomplete:
        return ("OFF (mode disabled) — but the stop is INCOMPLETE: see the "
                "still-live task(s) above. The campaign stays open until they "
                "settle. Post-task auto-evolution stays paused until /evolve start")
    return "OFF — post-task auto-evolution also paused until /evolve start"


def _record_routing_receipt(
    bridge: Any,
    ctx: Any,
    *,
    chat_id: int,
    client_message_id: str,
    action: str,
    target: str = "",
    status: str,
    persist: bool = True,
    options: Optional[list] = None,
) -> None:
    """Emit a typed bubble-free ack and optionally persist its presentation state."""
    if persist:
        try:
            from ouroboros.project_dialogue import append_chat_annotation

            append_chat_annotation(
                ctx.DRIVE_ROOT,
                client_message_id,
                action=action,
                target=target,
                status=status,
            )
        except Exception:
            log.debug("Routing annotation append failed", exc_info=True)
    try:
        ack = getattr(bridge, "send_routing_ack", None)
        if callable(ack):
            ack_kwargs = {
                "client_message_id": client_message_id,
                "action": action,
                "target": target,
                "status": status,
            }
            if options is not None:
                ack_kwargs["options"] = options
            ack(
                chat_id,
                **ack_kwargs,
            )
        else:
            broadcast = getattr(bridge, "broadcast", None)
            if callable(broadcast):
                payload = {
                    "type": "message_annotation",
                    "annotation_type": "routing_ack",
                    "chat_id": int(chat_id or 0),
                    "client_message_id": str(client_message_id or ""),
                    "action": action,
                    "target": target,
                    "status": status,
                    "suppress_bubble": True,
                }
                if options is not None:
                    payload["options"] = options
                broadcast(payload)
    except Exception:
        log.debug("Routing receipt broadcast failed", exc_info=True)


def _route_owner_message(bridge: Any, ctx: Any, incoming: Dict[str, Any]) -> None:
    """Route one non-command owner message through the canonical decision lane."""
    chat_id = int(incoming["chat_id"])
    text = str(incoming.get("text") or "")
    image_caption = str(incoming.get("image_caption") or "")
    client_message_id = str(incoming.get("client_message_id") or "")
    image_data = incoming.get("image_data")
    task_constraint = incoming.get("task_constraint")
    task_metadata = incoming.get("task_metadata")
    from ouroboros.contracts.task_constraint import normalize_task_constraint

    normalized_constraint = normalize_task_constraint(task_constraint)
    if normalized_constraint and normalized_constraint.mode == "skill_repair":
        # Repair is already a typed, narrowly confined task request. Sending it
        # through the conversation decision lane would combine skill_repair with
        # _ephemeral_turn: ephemeral hides the repair mutators while heal mode
        # blocks promotion. Promote it directly without weakening either policy.
        # DELIBERATE: task_metadata (incl. any client_surface fact) is dropped on
        # this branch — a repair task's objective is a fixed UI action and the
        # sending surface adds nothing to it (same treatment as force_plan here).
        from supervisor.events import _handle_promote_chat_to_task

        ctx.consciousness.inject_observation(
            f"Message from my human: {incoming.get('log_text') or ''}"
        )
        task_id = uuid.uuid4().hex[:16]
        event = {
            "type": "promote_chat_to_task",
            "task_id": task_id,
            "routing_token": uuid.uuid4().hex,
            "objective": text or image_caption,
            "chat_id": chat_id,
            "client_message_id": client_message_id,
            "task_constraint": task_constraint,
            "routed_from_main": True,
        }
        origin_ref = incoming.get("origin_message_ref")
        if isinstance(origin_ref, dict) and origin_ref:
            event["source_ref"] = origin_ref
            event["source_text"] = str(incoming.get("log_text") or "")
        else:
            event["origin_suppressed"] = True
        try:
            outcome = _handle_promote_chat_to_task(event, ctx)
        except Exception:
            log.warning("Direct skill-repair promotion failed", exc_info=True)
            outcome = {
                "status": "needs_manual_target",
                "reason": "repair_promotion_failed",
                "task_id": task_id,
            }
        outcome = outcome if isinstance(outcome, dict) else {"status": "scheduled", "task_id": task_id}
        outcome_status = str(outcome.get("status") or "needs_manual_target")
        if outcome_status == "scheduled":
            try:
                ctx.send_with_budget(
                    chat_id,
                    f"✅ Repair task {task_id} was accepted and durably scheduled.",
                )
            except Exception:
                log.debug("Repair promotion success notification failed", exc_info=True)
        else:
            reason = str(outcome.get("reason") or outcome_status)
            try:
                ctx.send_with_budget(
                    chat_id,
                    f"⚠️ Repair task was not started ({reason}). Please retry from the skill card.",
                )
            except Exception:
                log.debug("Repair promotion refusal notification failed", exc_info=True)
        return
    reserved_project = _reserved_project_for_chat(ctx, chat_id)
    project_id = (
        str(reserved_project.get("id") or "")
        if str((reserved_project or {}).get("lifecycle") or "active") == "active"
        else ""
    )
    if reserved_project and not project_id:
        _record_routing_receipt(
            bridge,
            ctx,
            chat_id=chat_id,
            client_message_id=client_message_id,
            action="project_route",
            target=str(reserved_project.get("id") or ""),
            status="project_unavailable",
        )
        return
    ctx.consciousness.inject_observation(f"Message from my human: {incoming.get('log_text') or ''}")
    task_metadata = _scoped_task_metadata(project_id, task_metadata)
    swarm_intent = bool(
        isinstance(task_metadata, dict) and task_metadata.get("force_plan")
    )
    # The turn's origin identity rides UNCONDITIONALLY (not only when the
    # decision lane runs): a bare direct turn with no projects/roots yet — the
    # first-ever project creation — must still carry it so promote/route/bind
    # receive the ref by value.
    origin_ref = incoming.get("origin_message_ref")
    if isinstance(origin_ref, dict) and origin_ref:
        task_metadata = {
            **(task_metadata or {}),
            "origin_message_ref": origin_ref,
            "origin_message_text": str(incoming.get("log_text") or ""),
        }
    else:
        # A suppressed (never-logged) message has a DESIGNED absence of origin;
        # downstream binders must not classify it as a producer bug.
        task_metadata = {**(task_metadata or {}), "origin_suppressed": True}
    # Owner Surface Fact channel fallback: a non-web ingress (telegram/skill
    # transports) carries no browser observables, but its channel IS the
    # surface fact. Host-stamped here, never overwriting a real descriptor;
    # source=="web" stays an honest absence (an old SPA sends no fact), and a
    # synthetic A2A chat (negative id) is machine traffic — no owner sent it,
    # so it must never wear an owner_client fact.
    from ouroboros.contracts.chat_id_policy import is_a2a_chat_id as _is_a2a

    _ingress_source = str(incoming.get("source") or "web")
    if (
        _ingress_source != "web"
        and not _is_a2a(chat_id)
        and not isinstance(task_metadata.get("client_surface"), dict)
    ):
        task_metadata = {**task_metadata, "client_surface": {"channel": _ingress_source}}
    if project_id and not swarm_intent:
        routed_to_task = _route_project_chat_to_running_task(
            ctx,
            chat_id,
            text or image_caption,
            client_message_id,
            task_metadata=task_metadata,
            image_data=image_data,
        )
        if routed_to_task:
            _record_routing_receipt(
                bridge,
                ctx,
                chat_id=chat_id,
                client_message_id=client_message_id,
                action="mailbox_delivery",
                target=routed_to_task,
                status="delivered",
            )
            return

    global_roots = _addressable_root_tasks(ctx, None)
    try:
        from ouroboros.projects_registry import list_projects

        has_projects = bool(list_projects(ctx.DRIVE_ROOT))
    except Exception:
        log.warning("Unable to inspect Projects for owner routing", exc_info=True)
        has_projects = True
    needs_decision_lane = swarm_intent or bool(project_id) or has_projects or bool(global_roots)
    if needs_decision_lane:
        task_metadata = _decision_turn_metadata(ctx, chat_id, client_message_id, task_metadata)
    agent = ctx.get_chat_agent()

    def _run_direct() -> None:
        try:
            ctx.handle_chat_direct(
                chat_id,
                text or image_caption,
                image_data,
                task_constraint=task_constraint,
                task_metadata=task_metadata,
            )
        finally:
            ctx.consciousness.resume()

    if needs_decision_lane or agent._busy:
        threading.Thread(
            target=ctx.handle_chat_ephemeral,
            args=(chat_id, text or image_caption, image_data),
            kwargs={"task_constraint": task_constraint, "task_metadata": task_metadata},
            daemon=True,
        ).start()
    else:
        ctx.consciousness.pause()
        threading.Thread(target=_run_direct, daemon=True).start()
