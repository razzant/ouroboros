"""Where one owner message goes.

Attachment staging into the addressed task's artifact store, the single
unambiguous mailbox delivery, the typed bubble-free routing receipt, and the
dispatch that hands everything else to the decision lane. The ``/evolve off``
stop transaction lives here too: it is the one owner command whose effect is a
message-time transaction rather than a process-lifecycle change.
"""

from __future__ import annotations

import base64
import os
import pathlib
import threading
import uuid
from typing import Any, Dict, Optional

from ouroboros.server_process import log
from ouroboros.server_routing_context import (
    RootSnapshot,
    _decision_turn_metadata,
    addressable_roots,
    _project_id_for_registered_chat,
    _reserved_project_for_chat,
    _scoped_task_metadata,
)


def _stage_mailbox_attachments(
    ctx: Any,
    task_drive: pathlib.Path,
    task_id: str,
    task_metadata: Any,
    image_data: Any = None,
) -> tuple[str, list, str]:
    """Stage one routed turn's files into the existing task artifact store.

    Returns ``(attachment_note, staged_manifest, rendered_report)`` — the manifest is kept so a
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
            return "", [], ""
        from ouroboros.artifacts import attachment_manifest_projection, stage_task_attachments
        from ouroboros.gateway.tasks import _render_attachment_lines

        manifest = stage_task_attachments(task_drive, task_id, uploads)
        rendered = _render_attachment_lines(attachment_manifest_projection(task_drive, task_id, manifest))
        note = f"\n\n[ATTACHMENTS]\n{rendered}\n[END_ATTACHMENTS]" if rendered else ""
        return note, manifest, rendered
    finally:
        if temp_source is not None:
            try:
                temp_source.unlink(missing_ok=True)
            except OSError:
                log.debug("Unable to remove routed attachment staging source", exc_info=True)


def _start_order(row: Dict[str, Any]) -> tuple:
    """Heads of one thread in start order; the task id breaks an equal start."""
    try:
        started = float(row.get("started_at") or 0.0)
    except (TypeError, ValueError):
        started = 0.0
    return started, str(row.get("task_id") or "")


def _route_project_chat_to_running_task(
    ctx: Any,
    chat_id: int,
    message: str,
    client_message_id: str = "",
    *,
    task_metadata: Any = None,
    image_data: Any = None,
    thread_of: str = "",
    roots: Optional[RootSnapshot] = None,
) -> str:
    """Deliver ordinary room input to its sole live reader, under admission locks.

    Queued or paused roots remain explicit tell_task targets, never implicit
    readers. Zero or several readers starts a fresh head in this room; the
    model chooses whether to relay, move, detach, answer or start work. A late
    answer or a press on a finished author's card (``thread_of``: that author)
    reads only a head that a late answer or press of the same author started,
    in a Project or Main, the earliest of several: one author's late input is
    one thread. No other reader receives it. ``roots``: the message's one root snapshot.
    """
    try:
        if not thread_of and not _project_id_for_registered_chat(ctx, chat_id):
            return ""
    except Exception:
        return ""
    try:
        steerable = [row for row in (roots if roots is not None else addressable_roots(ctx)).rows(chat_id)
                     if row.get("room_reader") and (not thread_of or row.get("late_input_of") == thread_of)]
        if thread_of:
            steerable = sorted(steerable, key=_start_order)[:1]
        # Exactly one candidate => unambiguous transport. Zero or many => a routing
        # decision the AGENT must make (P5/WS1), so do not deliver here.
        if len(steerable) != 1:
            return ""
        return _deliver_to_reader(ctx, steerable[0], message, client_message_id,
                                  task_metadata=task_metadata, image_data=image_data)
    except Exception:
        log.debug("Mailbox follow-up routing failed; falling back to direct lane", exc_info=True)
    return ""


def _owner_turn_key(ctx: Any, row: Dict[str, Any], meta: Any, chat_id: int) -> str:
    """``owner:<chat>`` when ROW's RUNNING row is my human's own confirmed turn of this chat on the conversation lane.

    The lane's host facts (its admission barrier landed, the door's reply expected, no live call's reservation)
    and the task's origin and carrier facts (``conversation_lease.owner_conversation_key``), with no Project binding
    the turn took since; else ""."""
    from ouroboros.conversation_lease import OWNER_CONVERSATION, owner_conversation_key
    from ouroboros.projects_registry import project_chat_for_task
    from supervisor.conversation_lane import conversation_row

    if (row.get("inline_presence") or not conversation_row(meta) or not meta.get("reply_expected")
            or meta.get("_unpublished") or meta.get("_inline_bind")):
        return ""
    key = owner_conversation_key(meta.get("task"))
    if key != f"{OWNER_CONVERSATION}{int(chat_id or 0)}" or project_chat_for_task(ctx.DRIVE_ROOT, str(row["task_id"])):
        return ""
    return key


def _owner_turn_closed(row: Dict[str, Any], meta: Any, key: str) -> bool:
    """Whether ROW, my human's own turn of this chat (``key``: its ``_owner_turn_key``, ``meta``: its RUNNING row),
    has a final drain that already read last: its own fact for the attempt RUNNING names is closed
    (``conversation_lease.author_intake_closed``). It finishes what it holds and reads no more words. No fact, an
    unreadable or foreign one, or another attempt's proves nothing."""
    from ouroboros.conversation_lease import author_intake_closed
    from supervisor.queue import _task_drive_for_task

    tid = str(row["task_id"])
    return author_intake_closed(_task_drive_for_task(meta["task"], tid), tid, key,
                                attempt=int(meta.get("attempt") or 1))


def _route_main_followup(
    ctx: Any,
    chat_id: int,
    message: str,
    client_message_id: str = "",
    *,
    task_metadata: Any = None,
    image_data: Any = None,
    roots: Optional[RootSnapshot] = None,
) -> str:
    """Main's ordinary owner message reaches the one turn reading this conversation when it is my human's own.

    One author per conversation: a clarification typed while that turn still awaits its carrier or prepares must not
    start a second author acting on the same request. The candidates are the rows that ARE my human's own turn of
    this chat by the host's facts (``_owner_turn_key``: the conversation lane, the owner door's ingress, no Project
    binding, not a wake, a call, Presence or detached work) and still read (``_owner_turn_closed``); exactly one
    candidate takes the words, none or several leave them their own turn. A wake, a call or background work beside
    the turn holds no ``owner:<chat>`` conversation and is no candidate: skipping it is a cut by the row's kind, never
    a choice among readers. From its confirmed admission the turn takes them only under its durable intake for the
    attempt judged (``conversation_lease.owner_conversation_key``), which its final drain closes under the same mail
    lock, and never once its result settled: no settled-result check alone orders words against a loop in another
    process. A refusal at any step is the words' own new turn; the receipt says saved, never read. ``roots``: the
    message's one root snapshot."""
    try:
        roots = roots if roots is not None else addressable_roots(ctx)
        if roots.gap:
            return ""
        running = getattr(ctx, "RUNNING", {})
        owner_turns = [(row, meta, key) for row in roots.rows(chat_id) if row.get("room_reader")
                       for meta in (running.get(str(row["task_id"])),)
                       for key in (_owner_turn_key(ctx, row, meta, chat_id),)
                       if key and not _owner_turn_closed(row, meta, key)]
        if len(owner_turns) != 1:
            return ""
        row, meta, key = owner_turns[0]
        return _deliver_to_reader(ctx, row, message, client_message_id, task_metadata=task_metadata,
                                  image_data=image_data, owner=(key, meta))
    except Exception:
        log.debug("Main follow-up routing failed; the message starts its own turn", exc_info=True)
    return ""


def _deliver_to_reader(
    ctx: Any,
    candidate: Dict[str, Any],
    message: str,
    client_message_id: str,
    *,
    task_metadata: Any,
    image_data: Any,
    owner: tuple = ("", None),
) -> str:
    """One reader's delivery transaction: its id, or "" when any admission check refuses (nothing stays staged).

    ``owner``: Main's ``(intake key, RUNNING meta)`` of an owner turn, whose words its durable intake admits
    for that meta's attempt (``_route_main_followup``), Q held for memory only, never across a read or the mail
    lock's wait; a Project room's reader passes none."""
    owner_key, owner_meta = owner
    owner_attempt = int((owner_meta or {}).get("attempt") or 1)
    try:
        tid = str(candidate["task_id"])
        direct_agent = None
        direct_lock = None
        remote_author = False
        if candidate.get("direct_chat") or candidate.get("inline_presence"):
            from ouroboros.conversation_lease import presence_author_agent
            from supervisor.workers import get_direct_chat_agent

            # A moved inline Presence author takes owner words exactly as a direct turn does: its own lock. One its
            # conversation lane carries in another process has none here: its durable intake admits them instead.
            direct_agent = (presence_author_agent(ctx.DRIVE_ROOT, tid) if candidate.get("inline_presence")
                            else get_direct_chat_agent(tid))
            direct_lock = getattr(direct_agent, "_owner_message_admission_lock", None)
            remote_author = direct_lock is None and bool(candidate.get("inline_presence"))
            if direct_lock is None and not remote_author:
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
        from ouroboros.project_dialogue import routing_target_label

        target_label = routing_target_label(
            ctx.DRIVE_ROOT,
            "mailbox_delivery",
            tid,
            task=task_obj or candidate,
            project_id=str((task_obj or candidate).get("project_id") or ""),
            fallback="" if owner_key else "Task",
        )
        from ouroboros.conversation_lease import author_intake_accepting
        from ouroboros.owner_mailbox import mailbox_drain_ended, write_owner_message
        from supervisor.queue import (
            ACCEPTANCE_FENCES,
            BUDGET_ROOT_FENCES,
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
        task_drive = (pathlib.Path(ctx.DRIVE_ROOT) if direct_lock_held or remote_author
                      else _task_drive_for_task(task_obj, tid))
        msg_id = f"{client_message_id}:{tid}" if client_message_id else None
        staged_manifest: list = []
        attachment_report = ""
        message_written = False
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
            attachment_note, staged_manifest, attachment_report = _stage_mailbox_attachments(
                ctx, task_drive, tid, task_metadata, image_data,
            )
            if direct_lock_held or remote_author:  # an author's own checks; a carried one's intake admits below
                from ouroboros.room_reader import is_reading
                if not is_reading(task_drive, tid, task_obj or candidate, running=True):
                    _drop_staged_inputs()
                    return ""
                # AR2-6 (fable): the direct-agent lane used to skip the
                # cancel-pending admission check the queue lane makes below — a
                # direct turn whose cancellation is pending must not accept a
                # new owner message either. Same predicate, same honest
                # fall-through to the direct chat lane.
                if cancel_pending(ctx.DRIVE_ROOT, tid):
                    log.info("Mailbox follow-up refused for %s: cancel pending (direct lane)", tid)
                    _drop_staged_inputs()
                    return ""
            if owner_key:
                # Main's owner turn: Q spans only memory (the row judged, a snapshot of its fences); the file reads
                # and the mail lock's wait run outside it, and the append re-checks under that lock what can move.
                with _queue_lock:
                    live_meta = getattr(ctx, "RUNNING", {}).get(tid)
                    if live_meta is not owner_meta:  # its barrier, landed when judged, only stays landed
                        return ""  # not the row judged (a crash's readmission, an ended turn): their own turn
                    task_obj = live_meta.get("task") if isinstance(live_meta.get("task"), dict) else live_meta
                    live_fences = getattr(ctx, "BUDGET_ROOT_FENCES", None)
                    budget_fences = dict(BUDGET_ROOT_FENCES if live_fences is None else live_fences)
                    active_fence = ACCEPTANCE_FENCES.get(str(task_obj.get("root_task_id") or tid))
                    if isinstance(active_fence, dict) and str(active_fence.get("status") or "") == "sealed":
                        return ""
                from ouroboros.room_reader import is_reading
                if (not is_reading(task_drive, tid, task_obj, running=True, fences=budget_fences)
                        or cancel_pending(ctx.DRIVE_ROOT, tid)):
                    return ""
            elif not direct_lock_held and not remote_author:
                _queue_lock.acquire()
                queue_lock_held = True
                live_meta = getattr(ctx, "RUNNING", {}).get(tid)
                if isinstance(live_meta, dict):
                    task_obj = live_meta.get("task") if isinstance(live_meta.get("task"), dict) else live_meta
                from ouroboros.room_reader import is_reading
                if not is_reading(task_drive, tid, task_obj, running=live_meta is not None,
                                  fences=getattr(ctx, "BUDGET_ROOT_FENCES", None)):
                    return ""
                # A worker may remain RUNNING to finish paid post-work after its
                # answer/result settled. Its solve loop no longer drains this
                # mailbox; accepting an owner follow-up here would label it
                # delivered, then terminal cleanup would erase it unread. One
                # predicate with the quiz ingress (TZ-2 D15). A pooled root writes no
                # intake fact, so words after its final drain and before its stored
                # result still land here, kept unread on that result (PERSISTENCE).
                if mailbox_drain_ended(task_drive, tid):
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
            def owner_accepts_now() -> bool:
                # The mail lock can wait while Pause/holds change. Read the current controls here, without Q:
                # Project delivery still takes Q before mail, so the opposite order would deadlock it.
                current = getattr(ctx, "RUNNING", {}).get(tid)
                if current is not owner_meta:
                    return False
                task = current.get("task") if isinstance(current.get("task"), dict) else current
                fence = ACCEPTANCE_FENCES.get(str(task.get("root_task_id") or tid))
                return (_owner_turn_key(ctx, candidate, current, int(task.get("chat_id") or 0)) == owner_key
                        and not (isinstance(fence, dict) and fence.get("status") == "sealed")
                        and is_reading(task_drive, tid, task, running=True,
                                       fences=getattr(ctx, "BUDGET_ROOT_FENCES", None), canonical=ctx.DRIVE_ROOT)
                        and author_intake_accepting(task_drive, tid, owner_key, attempt=owner_attempt)
                        and not cancel_pending(ctx.DRIVE_ROOT, tid))

            if not write_owner_message(
                task_drive, f"{message}{attachment_note}", tid, msg_id=msg_id,
                # This delivery IS the owner's message; the id is the caller's
                # parameter, never parsed back out of ``msg_id``.
                client_message_id=client_message_id,
                client_surface=(
                    dict(task_metadata["client_surface"])
                    if isinstance(task_metadata, dict) and isinstance(task_metadata.get("client_surface"), dict)
                    else None
                ),
                attachment_manifest=staged_manifest if staged_manifest else None,
                late_answer=(
                    task_metadata.get("late_answer") if isinstance(task_metadata, dict) else None
                ),
                card_press=task_metadata.get("card_press") if isinstance(task_metadata, dict) else None,
                reply_reference=task_metadata.get("reply_reference") if isinstance(task_metadata, dict) else None,
                # Only while that author's loop still reads mail, checked under the mail lock its final drain closes
                # the intake under (``conversation_lease.accept_author_mail``): else the words start their own turn.
                # An owner turn's row must still be the one judged, its fact name the attempt judged, its result not
                # settled and no Stop pending since: a forced rail closes the fact alone, its flag stays its Stop door.
                admit=((lambda: author_intake_accepting(task_drive, tid)) if remote_author else
                       owner_accepts_now if owner_key else None),
            ):
                return ""
            message_written = True
            if direct_lock_held:
                direct_agent._owner_message_generation = int(
                    getattr(direct_agent, "_owner_message_generation", 0) or 0
                ) + 1
            else:
                if owner_key and isinstance(active_fence, dict):  # after the mail lock, never inside it (Q→mail)
                    _queue_lock.acquire()
                    queue_lock_held = True
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
            elif staged_manifest and not message_written:
                _drop_staged_inputs()
        if fence_generation_changed:
            persist_queue_snapshot(reason="acceptance_fence_owner_message")
        if isinstance(task_metadata, dict) and staged_manifest:
            task_metadata["_attachment_manifest"] = [
                dict(item) for item in staged_manifest if isinstance(item, dict)
            ]
            task_metadata["_attachment_report"] = attachment_report
        if isinstance(task_metadata, dict):
            task_metadata["_routing_target_label"] = target_label
            task_metadata["_routing_target"] = task_obj or candidate  # the receipt names this work and its room
        # A landed delivery publishes no row: the receipt the caller records
        # from this stash carries the manifest; the task's mailbox carries the block.
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
    from supervisor.events_runtime_controls import owner_evolution_stop_controls

    # The control half first (#1307): latch, durable campaign stop intent, state flags —
    # none of them waits for cancellation and none skips another's failure.
    not_persisted = owner_evolution_stop_controls("disabled via owner chat")
    stop_incomplete = False
    try:
        from supervisor.queue import evolution_stop_report, stop_evolution_tasks
        from ouroboros.post_task_evolution import drop_pending_request

        # Fast path: drop any queued post-task promotion so it cannot re-arm on
        # the next boot tick (the evolution_owner_stopped flag is the durable backstop).
        try:
            drop_pending_request(ctx.DRIVE_ROOT)
        except Exception:
            log.warning("Pending evolution request could not be dropped; cancellation still attempted", exc_info=True)
        stopped = stop_evolution_tasks("disabled via owner chat")
        ctx.sort_pending()
        ctx.persist_queue_snapshot(reason="evolve_off")
        stop_lines, stop_incomplete = evolution_stop_report(stopped)
        for line in stop_lines:
            ctx.send_with_budget(chat_id, line, role="system", system_type="evolution_notice")
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
                "settle. Post-task auto-evolution stays paused until /evolve start" + not_persisted)
    return "OFF — post-task auto-evolution also paused until /evolve start" + not_persisted


def _record_routing_receipt(
    bridge: Any,
    ctx: Any,
    *,
    chat_id: int,
    client_message_id: str,
    action: str,
    target: str = "",
    target_label: str = "",
    status: str,
    persist: bool = True,
    options: Optional[list] = None,
    reason: str = "",
    detail: str = "",
    attachment_manifest: Optional[list] = None,
    task: Optional[Dict[str, Any]] = None,
    unnamed_label: str = "Task",
) -> None:
    """Emit a typed bubble-free ack and optionally persist its presentation state.

    The fields are the supervisor receipts' own (``_routing_receipt_fields``, #968),
    read once, written then sent; ``task`` is the live work the words reached, as a
    relay names it. The send does not wait on the row: a landed delivery's authority
    is the mailbox it already wrote, so a lost annotation still reaches the owner
    live instead of leaving the message undelivered or Sending.
    """
    from ouroboros.project_dialogue import ROUTING_RECEIPT_FIELDS
    from supervisor.events_project_routing import _routing_receipt_fields

    fields = _routing_receipt_fields(
        ctx, {"relay_target": task}, action=action, target=target, target_label=target_label, status=status,
        reason=reason, options=options, attachment_manifest=attachment_manifest, unnamed_label=unnamed_label,
    )
    if persist:
        try:
            from ouroboros.project_dialogue import append_chat_annotation

            append_chat_annotation(
                ctx.DRIVE_ROOT, client_message_id, reason=reason, detail=detail,
                **{key: value for key, value in fields.items() if key in ROUTING_RECEIPT_FIELDS},
            )
        except Exception:
            log.debug("Routing annotation append failed", exc_info=True)
    try:
        ack = getattr(bridge, "send_routing_ack", None)
        if callable(ack):
            from supervisor.events_project_routing import _routing_notice_ack_fields

            # The same carrier fact the supervisor's receipt writer states (#966).
            notice = _routing_notice_ack_fields(ctx, {"chat_id": chat_id, "client_message_id": client_message_id}, fields)
            ack(chat_id, client_message_id=client_message_id, **fields, **notice)
        else:
            broadcast = getattr(bridge, "broadcast", None)
            if callable(broadcast):
                broadcast({"type": "message_annotation", "annotation_type": "routing_ack", "chat_id": int(chat_id or 0),
                           "client_message_id": str(client_message_id or ""), **fields, "suppress_bubble": True})
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
    # A process-local value minted by the authenticated call door, after
    # canonical message acceptance. JSON/API metadata cannot construct it.
    task_metadata = dict(task_metadata or {})
    realtime_turn = task_metadata.pop("_realtime_turn", None)
    if realtime_turn is not None:
        from ouroboros.live_conversation import LiveConversationTurn

        if not isinstance(realtime_turn, LiveConversationTurn):
            raise ValueError("invalid call authority")
    from ouroboros.contracts.task_constraint import normalize_task_constraint

    normalized_constraint = normalize_task_constraint(task_constraint)
    if normalized_constraint and (
        normalized_constraint.mode == "skill_repair"
        or (normalized_constraint.mode == "normal" and (normalized_constraint.skill_name or normalized_constraint.payload_root))
    ):
        # An explicit selected-skill development request already asks for a
        # managed task. Preserve its source/caller facts while ordinary promotion
        # validates the payload and records the admitted revision.
        from supervisor.events import _handle_promote_chat_to_task, _notify_host_initiated_refusal

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
            # The host issued this promote (skill card), so no model turn waits on
            # the receipt: a refusal is told to the owner by ONE typed System row
            # from the promote handler, in the chat the owner wrote in.
            "host_initiated": True,
        }
        metadata = task_metadata if isinstance(task_metadata, dict) else {}
        if isinstance(metadata.get("client_surface"), dict):
            event["client_surface"] = dict(metadata["client_surface"])
        if isinstance(metadata.get("ui_control"), dict):
            # My human pressed a control and the host wrote this request: the press rides the task as its
            # own fact (enable authority for this repair, ``skill_lifecycle_actions``), the words as the host's.
            event["ui_control"] = dict(metadata["ui_control"])
            event["objective_author"] = {"kind": "ui_control", "control": str(metadata["ui_control"].get("name") or "")}
        if isinstance(metadata.get("presence"), dict) and metadata["presence"]:
            event["presence"] = dict(metadata["presence"])
            event["task_contract"] = dict(metadata.get("task_contract") or {})
        origin_ref = incoming.get("origin_message_ref")
        if isinstance(origin_ref, dict) and origin_ref:
            event["source_ref"] = origin_ref
            event["source_text"] = str(incoming.get("log_text") or "")
        else:
            event["origin_suppressed"] = True
        try:
            _handle_promote_chat_to_task(event, ctx)
        except Exception:
            log.warning("Direct skill-repair promotion failed", exc_info=True)
            # Minted OUTSIDE the handler, so its publication boundary never saw it.
            _notify_host_initiated_refusal(ctx, event, {
                "status": "needs_manual_target",
                "reason": "repair_promotion_failed",
                "task_id": task_id,
            })
        # A landed repair publishes no row: the receipt under the owner's message and
        # the task's own card carry it. A refusal is told by the promote handler's
        # typed System row (host_initiated) plus that receipt.
        return
    try:
        reserved_project = _reserved_project_for_chat(ctx, chat_id)
    except (OSError, ValueError) as exc:
        _record_routing_receipt(
            bridge, ctx, chat_id=chat_id, client_message_id=client_message_id,
            action="project_route", status="project_unavailable",
            reason="project_routing_fence_lookup_failed", detail=str(exc))
        return
    project_id = (
        str(reserved_project.get("id") or "")
        if reserved_project.get("lifecycle") == "active"
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
            reason="project_unavailable",
        )
        return
    # An owner message never wakes consciousness (owner decision В13): Main answers it itself.
    task_metadata = _scoped_task_metadata(project_id, task_metadata)
    task_metadata = {**(task_metadata or {}), "client_message_id": client_message_id, "work_kind": "owner"}
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
    if isinstance(task_metadata.get("ui_control"), dict):
        # A press composed these words: the act stamped the turn, the words stay the host's (never initial_user).
        task_metadata["objective_author"] = {"kind": "ui_control", "control": str(task_metadata["ui_control"].get("name") or "")}
    # Owner Surface Fact channel fallback: a non-web ingress (telegram/skill
    # transports) carries no browser observables, but its channel IS the
    # surface fact, with the common ingress receipt stamp (``received_at``,
    # ``enqueue_local_message``). Host-stamped here, never overwriting a real descriptor;
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
        received = {"received_at": str(incoming["received_at"])} if incoming.get("received_at") else {}
        task_metadata = {**task_metadata, "client_surface": {"channel": _ingress_source, **received}}
    from ouroboros.card_store import late_press_ref
    from ouroboros.owner_quiz import late_answer_ref

    late, press = task_metadata.get("late_answer"), task_metadata.get("card_press")
    thread_of = (late_answer_ref(late) or late_press_ref(press) or {}).get("task_id", "")
    # ONE read of the addressable roots for this message: the room's reader, Main's
    # manifest and the decision metadata all see the same projection.
    roots = addressable_roots(ctx)
    routed_to_task, unnamed_label = "", "Task"
    # A live call's admitted turn keeps the room it was opened in; it never re-routes.
    if realtime_turn is None and ((project_id and not late) or thread_of):
        routed_to_task = _route_project_chat_to_running_task(
            ctx,
            chat_id,
            text or image_caption,
            client_message_id,
            task_metadata=task_metadata,
            image_data=image_data,
            thread_of=thread_of,
            roots=roots,
        )
        # The room's other live roots hear my human's words as context, whichever return routed them, a late answer
        # included; a press is a typed act on a card, not words: it reaches its author's thread and no other reader.
        if project_id and not press:
            from ouroboros.owner_relay import carry_room_context

            carry_room_context(ctx.DRIVE_ROOT, chat_id, client_message_id, roots.rows(chat_id), routed_to_task)
    elif realtime_turn is None and not (project_id or late or press or task_constraint):
        # Main: my human's next words reach the one turn already answering them in this chat, else their own turn.
        routed_to_task = _route_main_followup(ctx, chat_id, text or image_caption, client_message_id,
                                              task_metadata=task_metadata, image_data=image_data, roots=roots)
        unnamed_label = ""  # its turn named nothing yet: the receipt says saved, never a placeholder name
    if routed_to_task:
        from ouroboros.owner_relay import carry_row_at_receipt

        carry_row_at_receipt(ctx.DRIVE_ROOT, routed_to_task, chat_id, client_message_id)  # down its tree, now
        _record_routing_receipt(
            bridge,
            ctx,
            chat_id=chat_id,
            client_message_id=client_message_id,
            action="mailbox_delivery",
            target=routed_to_task,
            target_label=(
                str(task_metadata.get("_routing_target_label") or "")
                if isinstance(task_metadata, dict) else ""
            ),
            status="delivered",
            detail=(
                str(task_metadata.get("_attachment_report") or "")
                if isinstance(task_metadata, dict) else ""
            ),
            attachment_manifest=(
                list(task_metadata.get("_attachment_manifest") or [])
                if isinstance(task_metadata, dict) else None
            ),
            task=task_metadata.get("_routing_target") if isinstance(task_metadata, dict) else None,
            unnamed_label=unnamed_label,
        )
        return

    try:
        from ouroboros.projects_registry import list_projects

        has_projects = bool(list_projects(ctx.DRIVE_ROOT))
    except Exception:
        log.warning("Unable to inspect Projects for owner routing", exc_info=True)
        has_projects = True
    # Unknown roots, like unknown Projects, go to the decision lane, which is told so.
    needs_decision_lane = bool(project_id) or has_projects or bool(roots.rows()) or bool(roots.gap)
    if needs_decision_lane:
        task_metadata = _decision_turn_metadata(ctx, chat_id, client_message_id, task_metadata, roots)

    if realtime_turn is not None:
        # A live call keeps a thread of this process (``executor=inline_call``): its host port is process-local.
        threading.Thread(target=ctx.handle_chat_direct, args=(chat_id, text or image_caption, image_data),
                         kwargs={"task_constraint": task_constraint, "task_metadata": task_metadata,
                                 "realtime_turn": realtime_turn}, daemon=True).start()
        return
    # The conversation lane's admission: the RUNNING row and the start receipt here, the
    # turn's body in a carrier process (owner decision 1A). No thread per owner turn.
    ctx.handle_chat_direct(
        chat_id,
        text or image_caption,
        image_data,
        task_constraint=task_constraint,
        task_metadata=task_metadata,
    )
