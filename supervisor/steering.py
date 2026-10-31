"""Steering delivery into a running task's mailbox.

The public tell_task tool uses the retained steer_task event and receipt rail.
The issuer remains capability provenance. The atomic relay entry carries exact
owner quotes and the teller's own contextual note, independently of its lane.
"""

from __future__ import annotations

import logging
import pathlib
import uuid
from typing import Any, Dict, Optional

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


def _notice_row_owed(evt: Dict[str, Any], carrier: str, *, paired: bool) -> bool:
    """Whether an owner act's refusal publishes its System row (``_refusal_row``).

    Always when its receipt selected that row as the carrier: no ACK carrying the cause
    was handed over (its durable row was refused, or no live ACK went out), so nothing else
    tells it, whatever message the act wears. Otherwise only for an unlabelled act (a
    synthetic receipt id) on a path that keeps the row the web shows, paired with the ACK.
    A task that spoke for itself reads its typed refusal in its tool result and Logs row.
    """
    if _task_issued(evt):
        return False
    return carrier == "system" or (
        paired and not _relayed_owner_message(str(evt.get("client_message_id") or "").strip()))


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
) -> Optional[str]:
    """Whether a cancellation owns this task — refuse the steering write if so.

    Checked TWICE on purpose: once up front (cheap, off the lock) and once inside
    the transaction that admits the message to the mailbox. Between those two
    points the queue lock is taken and the durable liveness re-checked, which is
    exactly the window a cancel ingress lands in; a single up-front check would
    let a message reach a task the supervisor is already tearing down.
    ``None`` when no cancellation owns it; otherwise the carrier the refusal's
    receipt selected for its owner notice.
    """
    try:
        from ouroboros.cancel_intents import cancel_pending

        if not cancel_pending(ctx.DRIVE_ROOT, target):
            return None
    except Exception:
        log.debug("steer_task cancel-pending check failed", exc_info=True)
        return None
    carrier = str(_steer_receipt(
        ctx, evt, target, target_label=target_label, status="rejected", reason="cancel_pending",
    ).get("routing_notice_carrier") or "")
    # A cancel-pending refusal is a live notice to the OWNER who asked; a task
    # that spoke for itself reads its typed refusal in the tool result instead.
    # An act that wears an owner message shows its cause on that message's
    # receipt line once its ACK is handed over; an UNLABELLED owner act (a
    # synthetic receipt id), or one whose ACK was not, needs the standalone
    # sentence — the same rule as the other refusals (``_notice_row_owed``).
    if notify and _notice_row_owed(evt, carrier, paired=True):
        _refusal_row(ctx, evt, target, target_label, "rejected", "cancel_pending", carrier)
    return carrier


def _refusal_row(
    ctx: Any, evt: Dict[str, Any], target: str, target_label: str, status: str, reason: str, carrier: str = "",
) -> None:
    """Publish an owner act's host cause as System: an unlabelled act has no chat receipt row,
    and any act whose ACK was not handed over has no other notice (``_notice_row_owed``).

    The row names the carrier its receipt selected: ``routing_ack`` when the ACK already
    carried the cause (a transport showing ACKs skips this copy), otherwise ``system``.
    It replies to the act's own owner message, never to the target task's origin.
    """
    from ouroboros.project_dialogue import routing_refusal_cause
    from supervisor.events_project_routing import _routing_notice_reply
    from supervisor.message_bus import notification_chat_route

    chat = notification_chat_route(evt.get("chat_id"))
    if chat is None:
        return
    try:
        ctx.send_with_budget(
            chat, f"{target_label or 'Task'} · {routing_refusal_cause('steer_task', status, reason, None)}",
            role="system", system_type="steer_not_delivered", task_id=target,
            progress_meta={"routing_notice_carrier": "routing_ack" if carrier == "routing_ack" else "system",
                           "reply_to": _routing_notice_reply(ctx, evt)},
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
    for quote in evt.get("quotes") or []:
        source = quote.get("source") or {}
        if source.get("client_message_id") and source["client_message_id"] != evt.get("client_message_id"):
            _emit_routing_receipt(ctx, {**evt, "client_message_id": source["client_message_id"],
                                       "chat_id": source["chat_id"]},
                                  action="steer_task", target=target, target_label=target_label,
                                  status=status, reason=reason, detail=detail)
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


def _handle_steer_task(evt: Dict[str, Any], ctx: Any) -> None:
    """Deliver a steering message to an addressable independent root.

    Any turn may address an independent root. Quotes keep owner authority and
    the note is always contextual, independently of the issuer lane. In every case this only enforces
    transport invariants and writes the active task drive; stale targets are
    reported, never silently respawned.
    """
    target = str(evt.get("target_task_id") or "").strip()
    message = str(evt.get("message") or "")
    try:
        chat_id = int(evt.get("chat_id") or 0)
    except (TypeError, ValueError):
        chat_id = 0
    if not target or (not message.strip() and not evt.get("quotes")):
        return
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
    evt = {**evt, "relay_target": task}
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
    ) is not None:
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
    elif _presence_target_refused(ctx, evt, task):
        refusal = "presence_work_not_related"
    else:
        refusal = ""
    if refusal:
        # Fail visibly: the chosen task is not a steerable running task here.
        # The receipt carries the cause; an OWNER turn whose act wears no owner
        # message (a synthetic receipt id), or whose ACK was not handed over,
        # also gets the chat sentence, because nothing else in its chat can show
        # the refusal. A task that spoke for itself reads the typed refusal in
        # its tool result and its Logs row.
        carrier = str(_steer_receipt(
            ctx, evt, target, target_label=target_label, status="needs_manual_target", reason=refusal,
        ).get("routing_notice_carrier") or "")
        if _notice_row_owed(evt, carrier, paired=True):
            _refusal_row(ctx, evt, target, target_label, "needs_manual_target", refusal, carrier)
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
    direct_lock_held = False
    queue_lock_held = False
    fence_generation_changed = False
    delivered = False
    # A refusal found inside the admission transaction: (status, reason, the carrier its
    # receipt selected). Nothing was written or staged. No refusal returns early (GR2-9):
    # each falls through so its owner notice runs AFTER the locks are released (a chat
    # send is not something to hold the global queue lock for — and an early `return`
    # skipped the notice entirely).
    refused = None
    active_fence = None
    relay_entry = {}

    def refuse(reason: str):
        carrier = _steer_receipt(ctx, evt, target, target_label=target_label,
                                 status="needs_manual_target", reason=reason).get("routing_notice_carrier")
        return "needs_manual_target", reason, str(carrier or "")

    try:
        from supervisor.queue import ACCEPTANCE_FENCES, _queue_lock, _task_drive_for_task
        if direct_active and direct_lock is not None:
            direct_lock.acquire()
            direct_lock_held = True
            if not (
                getattr(direct_agent, "_busy", False)
                and getattr(direct_agent, "_accepting_owner_messages", False)
                and str(getattr(direct_agent, "_current_task_id", "") or "") == target
            ):
                refused = refuse("target_closed")
        drive = pathlib.Path(ctx.DRIVE_ROOT) if direct_active else _task_drive_for_task(task, target)
        if not direct_active:
            _queue_lock.acquire()
            queue_lock_held = True
            live_meta = ctx.RUNNING.get(target) if isinstance(ctx.RUNNING, dict) else None
            still_pending = any(
                isinstance(row, dict) and str(row.get("id") or "") == target
                for row in list(getattr(ctx, "PENDING", []) or [])
            )
            # The worker can remain RUNNING solely for post-task work after
            # its solve loop and last mailbox drain. Never promise a delivery
            # to an actor whose own result is already terminal.
            from ouroboros.task_results import load_task_result
            from ouroboros.task_status import SETTLED_STATUSES

            if live_meta is None and not still_pending:
                refused = refuse("target_finished")
            elif (load_task_result(drive, target) or {}).get("status") in SETTLED_STATUSES:
                refused = refuse("target_finished")
            else:
                active_fence = ACCEPTANCE_FENCES.get(str(task.get("root_task_id") or target))
                if isinstance(active_fence, dict) and str(active_fence.get("status") or "") == "sealed":
                    refused = refuse("acceptance_fence_sealed")
        # Re-check INSIDE the admission transaction: the up-front check runs
        # before the queue lock is taken, and a cancel ingress lands in exactly
        # that window. Held under the same lock as the write, so the refusal and
        # the admission cannot both win.
        cancelling = None if refused else _refuse_steering_while_cancelling(
            ctx, evt, target, target_label=target_label, notify=False,
        )
        if cancelling is not None:
            refused = ("rejected", "cancel_pending", cancelling)
        elif refused is None:
            from ouroboros.owner_relay import write_relay
            relay_entry, newly_written = write_relay(
                drive, target, relay_id=str(evt.get("relay_id") or routing_token or msg_id),
                note=message, quotes=list(evt.get("quotes") or []), chat_id=chat_id, summary=str(evt.get("summary") or ""),
                teller_task_id=issuer_task_id, canonical_root=ctx.DRIVE_ROOT, sender_origin=evt.get("sender_origin"),
                client_surface=evt.get("client_surface"))
            delivered = True
            if not newly_written or not evt.get("quotes"):
                active_fence = None  # context alone supersedes no reviewed owner answer
            if direct_active and newly_written and evt.get("quotes"):
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
        refused = refuse("mailbox_write_failed")
    finally:
        if queue_lock_held:
            _queue_lock.release()
        if direct_lock_held:
            direct_lock.release()
    if refused is not None:
        # Only the cancel refusal keeps an unlabelled act's paired web row here; any
        # refusal whose ACK was not handed over tells its cause in this row.
        status, reason, carrier = refused
        if _notice_row_owed(evt, carrier, paired=reason == "cancel_pending"):
            _refusal_row(ctx, evt, target, target_label, status, reason, carrier)
    if delivered:
        if fence_generation_changed:
            ctx.persist_queue_snapshot(reason="acceptance_fence_owner_message")
        log.info("steer_task: delivered to task %s (chat %s) on drive %s", target, chat_id, drive)
        # The one visible forward is separate from the silent source receipt.
        from ouroboros.owner_relay import publish_relay
        publish_relay(ctx.DRIVE_ROOT, target, relay_entry, getattr(ctx, "bridge", None), task)
        _steer_receipt(
            ctx, evt, target, target_label=target_label, status="delivered",
        )
