"""Owner steering delivery into a running task's mailbox.

Extracted VERBATIM from ``supervisor/events.py`` (byte-neutral module-size
boundary for the pinned events surface): the cancel-pending steering refusal
and the ``steer_task`` event handler. ``supervisor.events`` imports the handler
back and keeps the dispatch table; the routing-receipt emitter stays in
``events.py`` (many other handlers use it) and is imported lazily here so the
dependency stays one-way at import time.
"""

from __future__ import annotations

import logging
import pathlib
import uuid
from typing import Any, Dict

from ouroboros.task_runtime import supports_native_task_controls

log = logging.getLogger(__name__)


def _refuse_steering_while_cancelling(
    ctx: Any,
    evt: Dict[str, Any],
    target: str,
    chat_id: int,
    *,
    target_label: str = "",
    notify: bool = True,
) -> bool:
    """Whether a cancellation owns this task — refuse the steering write if so.

    Checked TWICE on purpose: once up front (cheap, off the lock) and once inside
    the transaction that admits the message to the mailbox. Between those two
    points the queue lock is taken and the durable liveness re-checked, which is
    exactly the window a cancel ingress lands in; a single up-front check would
    let a message reach a task the supervisor is already tearing down.
    """
    from supervisor.events import _emit_routing_receipt

    try:
        from ouroboros.cancel_intents import cancel_pending

        if not cancel_pending(ctx.DRIVE_ROOT, target):
            return False
    except Exception:
        log.debug("steer_task cancel-pending check failed", exc_info=True)
        return False
    _emit_routing_receipt(
        ctx, evt, action="steer_task", target=target, target_label=target_label,
        status="rejected",
        reason="cancel_pending",
    )
    if notify and chat_id:
        try:
            ctx.send_with_budget(
                chat_id,
                f"⚠️ Couldn't steer task {target_label or '(no longer available)'} — "
                "its cancellation is pending "
                "(the supervisor is tearing it down). Wait for the settled "
                "outcome or start a new task.",
            )
        except Exception:
            log.debug("steer_task cancel-pending notice failed", exc_info=True)
    return True


def _handle_steer_task(evt: Dict[str, Any], ctx: Any) -> None:
    """Deliver an agent-chosen steering message to an addressable owner root.

    A Project decision is restricted to that room.  A Main decision may address
    any root in the host-provided global manifest, including a Project-bound root.
    In both cases this only enforces transport invariants and writes the active
    task drive; stale targets are reported, never silently respawned.
    """
    from supervisor.events import _emit_routing_receipt

    target = str(evt.get("target_task_id") or "").strip()
    message = str(evt.get("message") or "")
    raw_chat_id = evt.get("chat_id")
    try:
        chat_id = int(evt.get("chat_id") or 0)
    except (TypeError, ValueError):
        chat_id = 0
    if not target or not message.strip():
        return
    direct_agent = None
    direct_lock = None
    direct_active = False
    try:
        from supervisor.workers import get_direct_chat_agent, direct_chat_turn

        direct_agent = get_direct_chat_agent(target)
        direct_lock = getattr(direct_agent, "_owner_message_admission_lock", None)
        if direct_lock is not None:
            with direct_lock:
                task = direct_chat_turn(target)
                direct_active = task is not None
    except Exception:
        direct_active = False
    if not direct_active:
        running = getattr(ctx, "RUNNING", None)
        meta = running.get(target) if isinstance(running, dict) else None
        task = meta.get("task") if isinstance(meta, dict) and isinstance(meta.get("task"), dict) else (
            meta if isinstance(meta, dict) else None
        )
    if not isinstance(task, dict):
        pending = getattr(ctx, "PENDING", [])
        task = next(
            (row for row in list(pending or []) if isinstance(row, dict) and str(row.get("id") or "") == target),
            None,
        )

    target_label = ""
    if isinstance(task, dict):
        from ouroboros.project_dialogue import routing_target_label

        target_label = routing_target_label(
            ctx.DRIVE_ROOT,
            "steer_task",
            target,
            task=task,
            project_id=str(task.get("project_id") or ""),
        )

    # Refuse before staging, but only after the active task identity has been
    # resolved so the durable receipt and owner notice use the same event-time
    # human label as a successful delivery.
    if _refuse_steering_while_cancelling(
        ctx, evt, target, chat_id, target_label=target_label,
    ):
        return

    def _matches_chat(t: Dict[str, Any]) -> bool:
        try:
            if evt.get("allow_global_root"):
                return True
            if int(t.get("chat_id") or 0) == chat_id:
                return True
        except (TypeError, ValueError):
            pass
        # A converted/bound task may keep its original chat_id on the live object
        # but belong to a project thread — match via the durable binding.
        try:
            from ouroboros.projects_registry import project_chat_for_task
            return int(project_chat_for_task(ctx.DRIVE_ROOT, target) or 0) == chat_id
        except Exception:
            return False

    steerable = (
        isinstance(task, dict)
        and (direct_active or not task.get("_is_direct_chat"))
        and str(task.get("delegation_role") or "") != "subagent"
        and _matches_chat(task)
    )
    if not steerable:
        # Fail visibly: the chosen task is no longer a steerable running task in
        # this chat. Tell the owner so the agent/owner can answer or spawn instead.
        client_message_id = str(evt.get("client_message_id") or "").strip()
        _emit_routing_receipt(
            ctx, evt, action="steer_task", target=target, target_label=target_label,
            status="needs_manual_target",
            reason="target_not_steerable",
        )
        if not client_message_id and chat_id:
            try:
                ctx.send_with_budget(
                    chat_id,
                    f"⚠️ Couldn't steer task {target_label or '(no longer available)'} — it isn't running "
                    "in this chat anymore "
                    "(it may have finished). I'll answer here or start a new task instead.",
                )
            except Exception:
                log.debug("steer_task stale-target notice failed", exc_info=True)
        log.info("steer_task: stale/invalid target %s for chat %s", target, chat_id)
        return
    if not supports_native_task_controls(task):
        _emit_routing_receipt(ctx, evt, action="steer_task", target=target, target_label=target_label,
                              status="rejected", reason="runtime_steering_unsupported")
        return
    # Idempotent delivery: a stable msg_id from client_message_id+target dedups
    # retries; without a client id use a unique id (avoid false dedup/collision).
    client_message_id = str(evt.get("client_message_id") or "").strip()
    msg_id = f"{client_message_id}:{target}" if client_message_id else f"{uuid.uuid4().hex}:{target}"
    direct_lock_held = False
    queue_lock_held = False
    fence_generation_changed = False
    delivered = False
    cancel_pending_refused = False
    active_fence = None
    staged_manifest: list = []
    attachment_report = ""
    try:
        from supervisor.queue import ACCEPTANCE_FENCES, _queue_lock, _task_drive_for_task
        from ouroboros.owner_mailbox import write_owner_message, KIND_OWNER_TEXT
        if direct_active and direct_lock is not None:
            direct_lock.acquire()
            direct_lock_held = True
            if not (
                getattr(direct_agent, "_busy", False)
                and getattr(direct_agent, "_accepting_owner_messages", False)
                and str(getattr(direct_agent, "_current_task_id", "") or "") == target
            ):
                _emit_routing_receipt(
                    ctx, evt, action="steer_task", target=target, target_label=target_label,
                    status="needs_manual_target",
                    reason="target_closed",
                )
                return
        drive = pathlib.Path(ctx.DRIVE_ROOT) if direct_active else _task_drive_for_task(task, target)
        attachment_note = ""
        uploads = evt.get("attachment_uploads") if isinstance(evt.get("attachment_uploads"), list) else []
        if uploads:
            from ouroboros.artifacts import attachment_manifest_projection, stage_task_attachments
            from ouroboros.gateway.tasks import _render_attachment_lines

            # Staging runs after the up-front cancel check (top of this handler)
            # but BEFORE the transactional re-check below — so the manifest is
            # kept and the re-check refusal removes the just-staged inputs
            # (GR2-9) instead of leaving orphaned files in the artifact store
            # of a task the supervisor is tearing down.
            staged_manifest = stage_task_attachments(drive, target, uploads)
            rendered = _render_attachment_lines(attachment_manifest_projection(drive, target, staged_manifest))
            if rendered:
                attachment_report = rendered
                attachment_note = f"\n\n[ATTACHMENTS]\n{rendered}\n[END_ATTACHMENTS]"
        if not direct_active:
            _queue_lock.acquire()
            queue_lock_held = True
            live_meta = ctx.RUNNING.get(target) if isinstance(ctx.RUNNING, dict) else None
            still_pending = any(
                isinstance(row, dict) and str(row.get("id") or "") == target
                for row in list(getattr(ctx, "PENDING", []) or [])
            )
            if live_meta is None and not still_pending:
                _emit_routing_receipt(
                    ctx, evt, action="steer_task", target=target, target_label=target_label,
                    status="needs_manual_target",
                    reason="target_finished",
                )
                return
            fence_root = str(task.get("root_task_id") or target)
            active_fence = ACCEPTANCE_FENCES.get(fence_root)
            if isinstance(active_fence, dict) and str(active_fence.get("status") or "") == "sealed":
                _emit_routing_receipt(
                    ctx, evt, action="steer_task", target=target, target_label=target_label,
                    status="needs_manual_target",
                    reason="acceptance_fence_sealed",
                )
                return
        # Re-check INSIDE the admission transaction: the up-front check runs
        # before the queue lock is taken, and a cancel ingress lands in exactly
        # that window. Held under the same lock as the write, so the refusal and
        # the admission cannot both win. No early return here (GR2-9): the
        # refusal falls through so the staged-input removal and the owner
        # notice run AFTER the lock is released (a chat send is not something
        # to hold the global queue lock for — and the old `return` skipped the
        # notice entirely).
        if _refuse_steering_while_cancelling(
            ctx, evt, target, chat_id, target_label=target_label, notify=False,
        ):
            cancel_pending_refused = True
        else:
            if not write_owner_message(
                drive,
                f"{message}{attachment_note}",
                target,
                msg_id=msg_id,
                kind=KIND_OWNER_TEXT,
                client_surface=(
                    dict(evt["client_surface"])
                    if isinstance(evt.get("client_surface"), dict) and evt.get("client_surface")
                    else None
                ),
                attachment_manifest=staged_manifest if uploads else None,
            ):
                raise OSError("owner mailbox append was not durable")
            if direct_active:
                direct_agent._owner_message_generation = int(
                    getattr(direct_agent, "_owner_message_generation", 0) or 0
                ) + 1
            else:
                if isinstance(active_fence, dict) and str(active_fence.get("status") or "") == "active":
                    active_fence["owner_message_generation"] = int(
                        active_fence.get("owner_message_generation") or 0
                    ) + 1
                    fence_generation_changed = True
            delivered = True
    except Exception:
        log.warning("steer_task delivery failed for task %s", target, exc_info=True)
        _emit_routing_receipt(
            ctx, evt, action="steer_task", target=target, target_label=target_label,
            status="needs_manual_target",
            reason="mailbox_write_failed",
        )
    finally:
        if queue_lock_held:
            _queue_lock.release()
        if direct_lock_held:
            direct_lock.release()
        if staged_manifest and not delivered:
            try:
                from ouroboros.artifacts import remove_staged_attachments

                remove_staged_attachments(staged_manifest)
            except Exception:
                log.debug("staged-attachment cleanup failed for %s", target, exc_info=True)
    if cancel_pending_refused:
        # GR2-9: the message was refused, so the inputs staged for it must not
        # linger in the dying task's artifact store.
        if chat_id:
            try:
                ctx.send_with_budget(
                    chat_id,
                    f"⚠️ Couldn't steer task {target_label or '(no longer available)'} — "
                    "its cancellation is pending "
                    "(the supervisor is tearing it down). Wait for the settled "
                    "outcome or start a new task.",
                )
            except Exception:
                log.debug("steer_task cancel-pending notice failed", exc_info=True)
    if delivered:
        if fence_generation_changed:
            ctx.persist_queue_snapshot(reason="acceptance_fence_owner_message")
        log.info("steer_task: delivered to task %s (chat %s) on drive %s", target, chat_id, drive)
        _emit_routing_receipt(
            ctx,
            evt,
            action="steer_task",
            target=target,
            target_label=target_label,
            status="delivered",
            detail=attachment_report,
            attachment_manifest=staged_manifest if uploads else None,
        )
        if attachment_report:
            from supervisor.message_bus import notification_chat_route

            notice_chat = notification_chat_route(raw_chat_id)
            if notice_chat is not None:
                try:
                    ctx.send_with_budget(
                        notice_chat,
                        f"📎 Attachment staging report for {target_label or 'Task'}:\n"
                        f"{attachment_report}",
                    )
                except Exception:
                    log.debug("steer_task attachment report notice failed", exc_info=True)
