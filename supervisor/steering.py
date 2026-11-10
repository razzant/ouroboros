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


def _issuer(evt: Dict[str, Any]) -> Dict[str, Any]:
    """The event's issuer fact; an event from a producer that predates it is an
    owner turn (every such producer was a chat turn or the owner's picker click)."""
    issuer = evt.get("issuer")
    return dict(issuer) if isinstance(issuer, dict) and issuer.get("kind") else {"kind": "owner_turn"}


def _task_issued(evt: Dict[str, Any]) -> bool:
    return str(_issuer(evt).get("kind") or "") == "task"


def _mind_note(evt: Dict[str, Any]) -> bool:
    """An act whose words are the teller's own: no owner quote and a synthetic receipt id (``tell_task`` with a note
    only, ``control_relay._tell_task``). Whatever turn speaks it, the note is not my human's words (D7: without a
    verified source the words are the teller's), so its refusal is the teller's to narrate from its typed tool
    result, as a task-issued act's is; the owner's chat shows no row and no ACK for it. An act that carries an owner
    quote or wears an owner message id (a relayed message, the routing card's click) keeps its owner notice."""
    from ouroboros.project_dialogue import AGENT_RECEIPT_ID_PREFIX

    return not evt.get("quotes") and str(evt.get("client_message_id") or "").startswith(AGENT_RECEIPT_ID_PREFIX)


def _notice_row_owed(evt: Dict[str, Any], carrier: str) -> bool:
    """Whether an owner act's refusal publishes its System row (``_refusal_row``): exactly when its receipt
    selected that row as the carrier, early or late alike. Nothing else then tells it: no ACK carrying the cause
    was handed over (its durable row was refused, or no live ACK went out), or the ACK answers no chat row (an
    unlabelled act of my human's words: a quote whose row has no client id rides a synthetic receipt id), which
    is never a carrier (``events_project_routing._routing_notice_ack_fields``). A task that spoke for itself reads
    its typed refusal in its tool result and Logs row; a turn's note without owner words (``_mind_note``) has no
    owner notice, so its receipt selects no carrier.
    """
    return carrier == "system" and not _task_issued(evt)


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
        presence_record_binding, presence_related_work, presence_target_record, presence_task_binding,
    )

    if "presence_binding_id" in evt:
        stamp = evt.get("presence_binding_id")
        binding = stamp.strip() if isinstance(stamp, str) else ""
    else:
        running = getattr(ctx, "RUNNING", None)
        meta = running.get(str(_issuer(evt).get("task_id") or "")) if isinstance(running, dict) else None
        row = meta.get("task") if isinstance(meta, dict) else None
        binding = presence_task_binding(row) if isinstance(row, dict) else None
    if binding is None:
        return False
    target = str(evt.get("target_task_id") or task.get("id") or "").strip()
    record = presence_target_record(ctx.DRIVE_ROOT, target, queue_row=task)
    if binding and task.get("_inline_presence_author") and presence_record_binding(record) == binding:
        return False  # the live author of this place's own conversation: place_guard names it to its callers
    if binding and _delivered_in_conversation(ctx, target, evt.get("sender_origin")):
        return False  # the work that leads this thread (it has a delivery here): a lineage-scoped note (owner 06.11, 4=A)
    return not presence_related_work(binding, record)


def _delivered_in_conversation(ctx: Any, task_id: str, sender_origin: Any) -> bool:
    """Whether ``task_id`` has a delivery receipt in the sender's own conversation (its run origin's provider, account,
    conversation and thread): the host's record that this work wrote there, a thread it rooted or answered, read from
    the dialogue store and never from the target's say-so."""
    from ouroboros.presence_bindings import conversation_key
    from ouroboros.presence_runner import _live_task_rows

    origin = sender_origin if isinstance(sender_origin, dict) else {}
    if not task_id or not (origin.get("provider") and origin.get("conversation_id")):
        return False
    key = conversation_key(str(origin["provider"]), str(origin.get("account_id") or ""), str(origin["conversation_id"]),
                           str(origin.get("thread_id") or ""))
    try:
        return any(row.get("type") == "presence_delivery" for row in _live_task_rows(ctx.DRIVE_ROOT, task_id, key))
    except Exception:
        return False  # an unreadable store proves no relation


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
    # receipt line once its ACK is handed over; otherwise its receipt selects the
    # standalone sentence — the same rule as the other refusals (``_notice_row_owed``).
    if notify and _notice_row_owed(evt, carrier):
        _refusal_row(ctx, evt, target, target_label, "rejected", "cancel_pending")
    return carrier


def _refusal_row(ctx: Any, evt: Dict[str, Any], target: str, target_label: str, status: str, reason: str) -> None:
    """Publish an owner act's host cause as the System row its receipt selected as the notice's
    carrier (``_notice_row_owed``): every client shows it, the phone included.

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
            progress_meta={"routing_notice_carrier": "system", "reply_to": _routing_notice_reply(ctx, evt)},
        )
    except Exception:
        log.warning("steer refusal row failed for %s", target, exc_info=True)


def _steer_receipt(
    ctx: Any, evt: Dict[str, Any], target: str, *, target_label: str, status: str,
    reason: str = "", detail: str = "", attachment_manifest: Any = None,
) -> Dict[str, Any]:
    """The durable token-bound receipt every outcome writes; a task-authored act
    additionally gets its Logs row and a chat acknowledgement only when tied to
    a real owner message; synthetic task receipts stay silent."""
    from supervisor.events import _emit_routing_receipt
    from supervisor.events_project_routing import _owner_message_receipt

    task_issued = _task_issued(evt)
    receipt = _emit_routing_receipt(
        ctx, evt, action="steer_task", target=target, target_label=target_label,
        status=status, reason=reason, detail=detail,
        attachment_manifest=attachment_manifest,
        # The owner hears of an act that carries their words or their click; a task's act and the mind's own note
        # (``_mind_note``) keep their durable receipt and tell their refusal to the teller alone, except on the
        # owner message a task's relay keys its receipt on (that passes on no owner authority).
        publish=(not task_issued and not _mind_note(evt)) or bool(_owner_message_receipt(evt.get("client_message_id"))),
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


def _presence_author(ctx: Any, target: str):
    """``(agent, lock, row)`` of this process's live inline Presence author ``target``, moved or not; else the
    narrow reason a live inline turn takes nothing here; else None (no such live turn: ``target_unknown``).

    A live author resolves even while its intake is closed: the delivery transaction then refuses it under
    the same lock (``presence_author_closed`` while its turn still runs). A live turn whose author is not in
    its loop here, a starting one included, is ``presence_author_closed`` too, one in another process has no
    endpoint here (``presence_author_elsewhere``), and one whose facts cannot be read is never called ended
    (``presence_author_unreadable``).
    """
    try:
        from ouroboros.conversation_lease import presence_author_agent, resolved_presence_turns
        from ouroboros.task_results import load_task_result
        from supervisor.cancel_publication import live_presence_row

        observed = resolved_presence_turns(ctx.DRIVE_ROOT)
        fact = next((fact for fact in observed["roots"] if fact["task_id"] == target), None)
        if fact is None:
            if observed["complete"] or not live_presence_row(load_task_result(ctx.DRIVE_ROOT, target, strict=True) or {}):
                return None
            return "presence_author_unreadable"
        if fact["state"] == "unreadable":
            return "presence_author_unreadable"
        if not fact["local"]:
            return "presence_author_elsewhere"
        agent = presence_author_agent(ctx.DRIVE_ROOT, target)
        lock = getattr(agent, "_owner_message_admission_lock", None)
        if lock is not None and fact["state"] == "running":
            with lock:
                if getattr(agent, "_busy", False) and getattr(agent, "_current_task_id", None) == target:
                    return agent, lock, {**fact["row"], "id": target, "_inline_presence_author": True}
        return "presence_author_closed"
    except Exception:
        log.debug("steer_task: inline Presence author %s unreadable", target, exc_info=True)
        return "presence_author_unreadable"


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
    presence_refusal = ""
    if not direct_active:
        presence = _presence_author(ctx, target)
        if isinstance(presence, tuple):
            direct_agent, direct_lock, task = presence
            direct_active = True
        elif presence:
            presence_refusal = presence  # used only if no queue row holds the target either
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
        refusal = presence_refusal or "target_unknown"
    elif str(task.get("delegation_role") or "") == "subagent":
        refusal = "subagent_target"
    elif _presence_target_refused(ctx, evt, task):
        refusal = "presence_work_not_related"
    else:
        refusal = ""
    if refusal:
        # Fail visibly: the chosen task is not a steerable running task here.
        # The receipt carries the cause; an owner act of my human's words that
        # wears no owner message (a quote without a client id), or whose ACK was
        # not handed over, also gets the chat sentence, because nothing else in
        # its chat can show the refusal. A task that spoke for itself, and a note
        # of the mind (``_mind_note``), read the typed refusal in the tool result.
        carrier = str(_steer_receipt(
            ctx, evt, target, target_label=target_label, status="needs_manual_target", reason=refusal,
        ).get("routing_notice_carrier") or "")
        if _notice_row_owed(evt, carrier):
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
                from ouroboros.presence_runner import presence_turn_is_live

                # A closed inline author whose turn still runs has not finished.
                refused = refuse("presence_author_closed" if task.get("_inline_presence_author")
                                 and presence_turn_is_live(target, drive_root=ctx.DRIVE_ROOT) else "target_closed")
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
        # A refusal found inside the transaction is told like an early one: by the
        # System row whenever its receipt selected it (``_notice_row_owed``).
        status, reason, carrier = refused
        if _notice_row_owed(evt, carrier):
            _refusal_row(ctx, evt, target, target_label, status, reason)
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
