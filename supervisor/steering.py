"""Steering delivery into a running task's mailbox.

Extracted VERBATIM from ``supervisor/events.py`` (byte-neutral module-size
boundary for the pinned events surface): the cancel-pending steering refusal
and the ``steer_task`` event handler. ``supervisor.events`` imports the handler
back and keeps the dispatch table; the routing-receipt emitter stays in
``events.py`` (many other handlers use it) and is imported lazily here so the
dependency stays one-way at import time.

Who speaks decides how the message travels (the event's ``issuer`` fact, minted
by the tool host-side, never by the model). An OWNER TURN steers with owner
text: ``KIND_OWNER_TEXT``, the owner-message generation bump, the room veto and
the owner acknowledgement/notice. A TASK speaking for itself never travels as
owner text: its words are written as a ``KIND_TASK_MESSAGE`` row with
``independent_task`` provenance through the same writer ``forward_to_worker``
uses, any host-listed active independent root is addressable (owner 6C, the
hidden partition included), no standalone chat message is sent, and the receipt plus one typed
Logs row carry the true author and target (owner 5=A/7A). A relay retains the
drained owner message's acknowledgement without acquiring owner authority.
An unlabelled owner refusal is a typed System row, never Ouroboros speech.
"""

from __future__ import annotations

import logging
import pathlib
import uuid
from typing import Any, Dict

from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)


def _relayed_owner_message(client_message_id: str) -> str:
    """The owner message identified by this receipt, or empty for a synthetic act.

    Only owner-text delivery copies it into the recipient's mailbox. A task
    relay may key its receipt on that message without passing on owner authority."""
    from ouroboros.project_dialogue import AGENT_RECEIPT_ID_PREFIX

    return "" if client_message_id.startswith(AGENT_RECEIPT_ID_PREFIX) else client_message_id


def _issuer(evt: Dict[str, Any]) -> Dict[str, Any]:
    """The event's issuer fact; an event from a producer that predates it is an
    owner turn (every such producer was a chat turn or the owner's picker click)."""
    issuer = evt.get("issuer")
    return dict(issuer) if isinstance(issuer, dict) and issuer.get("kind") else {"kind": "owner_turn"}


def _task_issued(evt: Dict[str, Any]) -> bool:
    return str(_issuer(evt).get("kind") or "") == "task"


def _presence_target_refused(ctx: Any, evt: Dict[str, Any], task: Dict[str, Any]) -> bool:
    """A Presence sender's live target must be independent work of its own binding.

    The sender is Presence by the host's stamp on the event or, failing that, by
    its own live queue row (a delegated descendant's inherited binding authority),
    so the fence never rests on one producer remembering to stamp the event; a
    malformed stamp or carrier narrows to nothing. Whose work the target is follows
    the read/cancel precedence: its canonical record decides, and the live row
    stands in only for a record without Presence provenance.
    """
    from ouroboros.dialogue_provenance import (
        presence_metadata_binding, presence_related_work, presence_target_record,
    )

    if "presence_binding_id" in evt:
        stamp = evt.get("presence_binding_id")
        binding = stamp.strip() if isinstance(stamp, str) else ""
    else:
        running = getattr(ctx, "RUNNING", None)
        meta = running.get(str(_issuer(evt).get("task_id") or "")) if isinstance(running, dict) else None
        row = meta.get("task") if isinstance(meta, dict) else None
        binding = presence_metadata_binding(row.get("metadata")) if isinstance(row, dict) else None
        if binding is None and isinstance(row, dict) and isinstance(row.get("task_contract"), dict):
            binding = "" if "capability_ceiling" in row["task_contract"] else None
    if binding is None:
        return False
    target = str(evt.get("target_task_id") or task.get("id") or "").strip()
    return not presence_related_work(binding, presence_target_record(ctx.DRIVE_ROOT, target, queue_row=task))


def _refuse_steering_while_cancelling(
    ctx: Any,
    evt: Dict[str, Any],
    target: str,
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
    try:
        from ouroboros.cancel_intents import cancel_pending

        if not cancel_pending(ctx.DRIVE_ROOT, target):
            return False
    except Exception:
        log.debug("steer_task cancel-pending check failed", exc_info=True)
        return False
    _steer_receipt(
        ctx, evt, target, target_label=target_label, status="rejected", reason="cancel_pending",
    )
    # A cancel-pending refusal is a live notice to the OWNER who asked; a task
    # that spoke for itself reads its typed refusal in the tool result instead.
    # An act that wears an owner message already shows its cause on that
    # message's receipt line; only an UNLABELLED owner act (a synthetic receipt
    # id) needs the standalone sentence — the same rule as the other refusals.
    notify = notify and not _relayed_owner_message(str(evt.get("client_message_id") or "").strip())
    if notify and not _task_issued(evt):
        _refusal_row(ctx, evt, target, target_label, "rejected", "cancel_pending")
    return True


def _refusal_row(
    ctx: Any, evt: Dict[str, Any], target: str, target_label: str, status: str, reason: str,
) -> None:
    """An unlabelled owner act has no chat receipt row, so publish its host cause as System."""
    from ouroboros.project_dialogue import routing_refusal_cause
    from supervisor.message_bus import notification_chat_route

    chat = notification_chat_route(evt.get("chat_id"))
    if chat is None:
        return
    try:
        ctx.send_with_budget(
            chat, f"{target_label or 'Task'} · {routing_refusal_cause('steer_task', status, reason, None)}",
            role="system", system_type="steer_not_delivered", task_id=target,
        )
    except Exception:
        log.debug("steer refusal row failed for %s", target, exc_info=True)


def _steer_receipt(
    ctx: Any, evt: Dict[str, Any], target: str, *, target_label: str, status: str,
    reason: str = "", detail: str = "", attachment_manifest: Any = None,
) -> Dict[str, Any]:
    """The durable token-bound receipt every outcome writes; a task-authored act
    additionally gets its Logs row and a chat acknowledgement only when tied to
    a real owner message; synthetic task receipts stay silent."""
    from supervisor.events import _emit_routing_receipt

    task_issued = _task_issued(evt)
    owner_message_id = _relayed_owner_message(str(evt.get("client_message_id") or "").strip())
    receipt = _emit_routing_receipt(
        ctx, evt, action="steer_task", target=target, target_label=target_label,
        status=status, reason=reason, detail=detail,
        attachment_manifest=attachment_manifest, publish=not task_issued or bool(owner_message_id),
    )
    if task_issued:
        # One typed Logs row per task-authored act: true author, target, outcome
        # (7A). The receiving task's own timeline gets ``task_message_injected``
        # when it drains the row; this row lands in the SENDER's timeline (keyed
        # on its id), so a refusal a target never saw is still visible somewhere.
        try:
            ctx.append_jsonl(ctx.DRIVE_ROOT / "logs" / "events.jsonl", {
                "ts": utc_now_iso(),
                "type": "task_message_routed",
                "task_id": str(_issuer(evt).get("task_id") or ""),
                "target_task_id": target,
                "status": "written" if status == "delivered" else "refused",
                "reason": reason,
                "routing_token": str(evt.get("routing_token") or ""),
            })
        except Exception:
            log.debug("task_message_routed row failed", exc_info=True)
    return receipt


def _owner_lane_allows(ctx: Any, task: Dict[str, Any], target: str, chat_id: int) -> bool:
    """The room veto for an OWNER turn: its own chat's roots, a root bound to its
    Project room, and -- from the Main lane, which sees the global manifest --
    every root. The lane is the host registry's answer for the issuing chat, so
    a Swarm root (no routing contract) and a picker click read the same rule."""
    try:
        if int(task.get("chat_id") or 0) == chat_id:
            return True
    except (TypeError, ValueError):
        pass
    # A converted/bound task may keep its original chat_id on the live object
    # but belong to a project thread — match via the durable binding.
    try:
        from ouroboros.projects_registry import project_chat_for_task
        from ouroboros.server_routing_context import _project_id_for_registered_chat

        if int(project_chat_for_task(ctx.DRIVE_ROOT, target) or 0) == chat_id:
            return True
        return not _project_id_for_registered_chat(ctx, chat_id)
    except Exception:
        return False


def _handle_steer_task(evt: Dict[str, Any], ctx: Any) -> None:
    """Deliver a steering message to an addressable independent root.

    An owner turn in a Project room is restricted to that room; one in Main may
    address any root in the host-provided global manifest, including a
    Project-bound root. A task speaking for itself may address any host-listed
    active independent root (owner 6C). In every case this only enforces
    transport invariants and writes the active task drive; stale targets are
    reported, never silently respawned.
    """
    target = str(evt.get("target_task_id") or "").strip()
    message = str(evt.get("message") or "")
    raw_chat_id = evt.get("chat_id")
    try:
        chat_id = int(evt.get("chat_id") or 0)
    except (TypeError, ValueError):
        chat_id = 0
    if not target or not message.strip():
        return
    task_issued = _task_issued(evt)
    issuer_task_id = str(_issuer(evt).get("task_id") or "")
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
        ctx, evt, target, target_label=target_label,
    ):
        return

    # Three different refusals in the same order the boolean used to fold them
    # into one. Which one fired is what the owner needs: a task running in its
    # own project room for another half hour is not a task that "may have
    # finished", and a receipt that says only `target_not_steerable` cannot tell
    # the two apart afterwards either. A direct turn is steerable exactly when it
    # is a live in-process actor OR a queue row — parked under its exact budget
    # pause, or resumed on a pooled worker under the SAME id (#1196): the owner's
    # follow-up reaches that actor's mailbox, never a second direct turn. A
    # direct turn that is neither is simply unknown here.
    if not isinstance(task, dict):
        refusal = "target_unknown"
    elif str(task.get("delegation_role") or "") == "subagent":
        refusal = "subagent_target"
    elif task_issued and _presence_target_refused(ctx, evt, task):
        refusal = "presence_work_not_related"
    elif not task_issued and not _owner_lane_allows(ctx, task, target, chat_id):
        refusal = "chat_mismatch"
    else:
        refusal = ""
    if refusal:
        # Fail visibly: the chosen task is not a steerable running task here.
        # The receipt carries the cause; an OWNER turn whose act wears no owner
        # message (a synthetic receipt id) also gets the chat sentence, because
        # nothing else in its chat can show the refusal. A task that spoke for
        # itself reads the typed refusal in its tool result and its Logs row.
        client_message_id = str(evt.get("client_message_id") or "").strip()
        _steer_receipt(
            ctx, evt, target, target_label=target_label, status="needs_manual_target", reason=refusal,
        )
        owner_unlabelled = not task_issued and not _relayed_owner_message(client_message_id)
        if owner_unlabelled:
            _refusal_row(ctx, evt, target, target_label, "needs_manual_target", refusal)
        log.info("steer_task: %s target %s for chat %s", refusal, target, chat_id)
        return
    # Idempotent delivery: a stable msg_id from client_message_id+target dedups
    # retries; without a client id use a unique id (avoid false dedup/collision).
    # The routing token completes the key: one owner message can produce SEVERAL
    # distinct steers (a turn relaying successive instructions under one origin
    # id), and without the token the drain deduplicated every one after the
    # first into silence (#896). The token rides the event by value and both
    # producers mint it once per steer — the picker derives it deterministically
    # from the click — so a retried emit of the SAME steer still collides. The
    # project-chat delivery (server_owner_routing._route_project_chat_to_running_task)
    # keys `{client_message_id}:{target}`; the two producers sit on mutually exclusive
    # ingress branches, so the two key forms never dedupe against each other.
    client_message_id = str(evt.get("client_message_id") or "").strip()
    routing_token = str(evt.get("routing_token") or "").strip()
    base_id = client_message_id or uuid.uuid4().hex
    msg_id = f"{base_id}:{target}:{routing_token}" if routing_token else f"{base_id}:{target}"
    relayed_owner_message_id = _relayed_owner_message(client_message_id)
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
        from ouroboros.owner_mailbox import (
            KIND_OWNER_TEXT, PROVENANCE_INDEPENDENT_TASK, write_owner_message, write_task_message,
        )
        if direct_active and direct_lock is not None:
            direct_lock.acquire()
            direct_lock_held = True
            if not (
                getattr(direct_agent, "_busy", False)
                and getattr(direct_agent, "_accepting_owner_messages", False)
                and str(getattr(direct_agent, "_current_task_id", "") or "") == target
            ):
                _steer_receipt(
                    ctx, evt, target, target_label=target_label,
                    status="needs_manual_target", reason="target_closed",
                )
                return
        drive = pathlib.Path(ctx.DRIVE_ROOT) if direct_active else _task_drive_for_task(task, target)
        attachment_note = ""
        # Tasks cannot attach files to peer messages: only an owner turn stages.
        uploads = (
            evt.get("attachment_uploads")
            if not task_issued and isinstance(evt.get("attachment_uploads"), list) else []
        )
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
                _steer_receipt(
                    ctx, evt, target, target_label=target_label,
                    status="needs_manual_target", reason="target_finished",
                )
                return
            # The worker can remain RUNNING solely for post-task work after
            # its solve loop and last mailbox drain. Never promise a delivery
            # to an actor whose own result is already terminal.
            from ouroboros.task_results import load_task_result
            from ouroboros.task_status import SETTLED_STATUSES

            if (load_task_result(drive, target) or {}).get("status") in SETTLED_STATUSES:
                _steer_receipt(ctx, evt, target, target_label=target_label,
                               status="needs_manual_target", reason="target_finished")
                return
            fence_root = str(task.get("root_task_id") or target)
            active_fence = ACCEPTANCE_FENCES.get(fence_root)
            if isinstance(active_fence, dict) and str(active_fence.get("status") or "") == "sealed":
                _steer_receipt(
                    ctx, evt, target, target_label=target_label,
                    status="needs_manual_target", reason="acceptance_fence_sealed",
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
            ctx, evt, target, target_label=target_label, notify=False,
        ):
            cancel_pending_refused = True
        elif task_issued:
            # A task's own words: a task-message row under the sender's id and
            # the independent provenance. No owner generation moves -- a peer
            # message supersedes no reviewed answer (owner 4=A).
            if not write_task_message(
                drive, message, target, source_task_id=issuer_task_id,
                provenance=PROVENANCE_INDEPENDENT_TASK, msg_id=msg_id,
                sender_origin=evt.get("sender_origin") if isinstance(evt.get("sender_origin"), dict) else None,
            ):
                raise OSError("task mailbox append was not durable")
            delivered = True
        else:
            if not write_owner_message(
                drive,
                f"{message}{attachment_note}",
                target,
                msg_id=msg_id,
                kind=KIND_OWNER_TEXT,
                client_message_id=relayed_owner_message_id,
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
        _steer_receipt(
            ctx, evt, target, target_label=target_label,
            status="needs_manual_target", reason="mailbox_write_failed",
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
        # linger in the dying task's artifact store. The notice is the owner's:
        # a task that spoke for itself has its typed refusal and Logs row.
        if not task_issued and not relayed_owner_message_id:
            _refusal_row(ctx, evt, target, target_label, "rejected", "cancel_pending")
    if delivered:
        if fence_generation_changed:
            ctx.persist_queue_snapshot(reason="acceptance_fence_owner_message")
        log.info("steer_task: delivered to task %s (chat %s) on drive %s", target, chat_id, drive)
        _steer_receipt(
            ctx, evt, target, target_label=target_label, status="delivered",
            detail=attachment_report, attachment_manifest=staged_manifest if uploads else None,
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
                        role="system", system_type="attachment_notice")
                except Exception:
                    log.debug("steer_task attachment report notice failed", exc_info=True)
