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

from ouroboros.owner_mailbox import MailboxWriteUnconfirmed
from ouroboros.server_process import log
from ouroboros.server_routing_context import (
    RootSnapshot,
    _decision_turn_metadata,
    addressable_roots,
    _project_id_for_registered_chat,
    _reserved_project_for_chat,
    _root_results_exist,
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
        from ouroboros.chat_uploads import delivery_manifest_rows
        from ouroboros.gateway.tasks import _render_attachment_lines

        manifest = stage_task_attachments(task_drive, task_id, uploads)
        authority = attachment_manifest_projection(task_drive, task_id, manifest)
        rendered = _render_attachment_lines(authority)
        rows = authority["attachment_manifest"]
        if failures := delivery_manifest_rows(rows):
            authority = {**authority, "attachment_manifest": [row for row in rows if row not in failures]}
        staged = _render_attachment_lines(authority) if failures else rendered
        note = f"\n\n[ATTACHMENTS]\n{staged}\n[END_ATTACHMENTS]" if staged else ""
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


def _carry_project_room_context(ctx, chat_id, project_id, ingress_chat, client_message_id, roots, routed_to, *, reply=None, source_ref=None):
    """A numeric chat match is only a candidate; context belongs to the decided Project."""
    from ouroboros.owner_relay import carry_room_context
    from ouroboros.projects_registry import project_id_for_task

    drive_root = ctx.DRIVE_ROOT
    members = [root for root in roots.rows(chat_id) if
               _ordinary_room_reader(ctx, root)
               and (project_id_for_task(drive_root, root["task_id"]) or root.get("project_id")) == project_id]
    carry_room_context(drive_root, ingress_chat, client_message_id, members, routed_to, reply=reply, source_ref=source_ref)


def _ordinary_room_reader(ctx: Any, row: dict) -> bool:
    from ouroboros.extension_late_result import restricted_return

    tid = str(row["task_id"])
    running = getattr(ctx, "RUNNING", {}).get(tid) or {}
    if running:
        return not restricted_return(running.get("task") or running)
    return not any(
        restricted_return(task) for task in getattr(ctx, "PENDING", ()) if str(task.get("id") or "") == tid)


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
                     if row.get("room_reader") and (not thread_of or row.get("late_input_of") == thread_of)
                     and _ordinary_room_reader(ctx, row)]
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
    """``owner:<chat>`` when ROW's confirmed direct turn reads ordinary input of this chat.

    The lane's host facts (its admission barrier landed, no live call's reservation)
    and the task's original carrier address identify it. Ordinary room intake excludes moved turns."""
    from ouroboros.conversation_lease import OWNER_CONVERSATION, owner_conversation_key
    from ouroboros.projects_registry import project_chat_for_task
    from supervisor.conversation_lane import conversation_row

    if (row.get("inline_presence") or not conversation_row(meta)
            or meta.get("_unpublished") or meta.get("_inline_bind")):
        return ""
    task = meta.get("task") or {}
    if (task.get("admission_author") or {}).get("late_report_ref"):
        return ""  # a subsequent owner message keeps its own full conversation authority
    key = owner_conversation_key(task)
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
    note: Optional[Dict[str, Any]] = None,
) -> str:
    """Main's ordinary input reaches the one turn reading this conversation.
    An agent's note (``note``: its row's recorded author and address) reaches the same turn the same way, as an
    agent's note (``_deliver_to_reader``).

    One author per conversation: a clarification typed while that turn still awaits its carrier or prepares must not
    start a second author acting on the same request. Candidates are direct turns of this chat
    (``_owner_turn_key``: the conversation lane with owner or note admission, no Project
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
        fallback = task_metadata.get("work_reply_fallback") if isinstance(task_metadata, dict) else None
        excluded = str(fallback.get("target") or "") if isinstance(fallback, dict) else ""
        running = getattr(ctx, "RUNNING", {})
        owner_turns = [(row, meta, key) for row in roots.rows(chat_id)
                       if row.get("room_reader") and str(row.get("task_id") or "") != excluded
                       for meta in (running.get(str(row["task_id"])),)
                       for key in (_owner_turn_key(ctx, row, meta, chat_id),)
                       if key and not _owner_turn_closed(row, meta, key)]
        if len(owner_turns) != 1:
            return ""
        row, meta, key = owner_turns[0]
        return _deliver_to_reader(ctx, row, message, client_message_id, task_metadata=task_metadata,
                                  image_data=image_data, owner=(key, meta), note=note)
    except MailboxWriteUnconfirmed:
        raise
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
    note: Optional[Dict[str, Any]] = None,
) -> str:
    """One reader's delivery transaction: its id, or "" when any admission check refuses (nothing stays staged).

    ``owner``: Main's ``(intake key, RUNNING meta)`` of an owner turn, whose words its durable intake admits
    for that meta's attempt (``_route_main_followup``), Q held for memory only, never across a read or the mail
    lock's wait; a Project room's reader passes none. ``note``: an agent's note takes the same admission and lands
    as a context-only task message under its own heading (``owner_mailbox.PROVENANCE_AGENT_NOTE``): never owner
    mail, so it enters no owner corpus and moves no owner-message generation the delivery seal reads."""
    owner_key, owner_meta = owner
    owner_attempt = int((owner_meta or {}).get("attempt") or 1)
    raise_on_unconfirmed = isinstance(task_metadata, dict) and bool(task_metadata.get("work_reply_fallback"))
    message_written, unconfirmed_write = False, False
    target_label, task_obj, staged_manifest, attachment_report = "", {}, [], ""
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
                      else _task_drive_for_task(task_obj, tid, canonical=ctx.DRIVE_ROOT))
        msg_id = f"{client_message_id}:{tid}" if client_message_id else None
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
                from ouroboros.extension_late_result import restricted_return
                if restricted_return(task_obj):
                    return ""  # including an explicit follow-up in this Project: the owner gets a normal turn
                from ouroboros.room_reader import is_reading
                if not is_reading(task_drive, tid, task_obj, running=live_meta is not None,
                                  fences=getattr(ctx, "BUDGET_ROOT_FENCES", None), canonical=ctx.DRIVE_ROOT):
                    return ""
                # A worker may remain RUNNING to finish paid post-work after its
                # answer/result settled. Its solve loop no longer drains this
                # mailbox; accepting an owner follow-up here would label it
                # delivered, then terminal cleanup would erase it unread. One
                # predicate with the quiz ingress (TZ-2 D15). A pooled root now writes a
                # ``work:`` intake fact whose final close an addressed Reply reads
                # (``work_reply._state``); this ordinary room door still keys on the settled
                # result, so words after its final drain and before its stored result land
                # here, kept unread on that result (PERSISTENCE).
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

            if note:  # the same admission; a program's words never move an owner-message generation
                from ouroboros.owner_mailbox import PROVENANCE_AGENT_NOTE, write_task_message
                def note_is_current() -> bool:
                    return (author_intake_accepting(task_drive, tid) if remote_author else
                            owner_accepts_now() if owner_key else True)

                wrote = write_task_message(task_drive, message, tid, source_task_id="", provenance=PROVENANCE_AGENT_NOTE,
                                           msg_id=msg_id, agent_note=note, admit=note_is_current)
            else:
                try:
                    wrote = write_owner_message(
                        task_drive, f"{message}{attachment_note}", tid, msg_id=msg_id,
                        # This delivery IS the owner's message; the id is the caller's
                        # parameter, never parsed back out of ``msg_id``.
                        client_message_id=client_message_id,
                        origin_message_ref=task_metadata.get("origin_message_ref") if isinstance(task_metadata, dict) else None,
                        origin_suppressed=isinstance(task_metadata, dict) and task_metadata.get("origin_suppressed") is True,
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
                        work_reply_fallback=task_metadata.get("work_reply_fallback") if isinstance(task_metadata, dict) else None,
                        raise_on_unconfirmed=raise_on_unconfirmed,
                        # Only while that author's loop still reads mail, checked under the mail lock its final drain closes
                        # the intake under (``conversation_lease.accept_author_mail``): else the words start their own turn.
                        # An owner turn's row must still be the one judged, its fact name the attempt judged, its result not
                        # settled and no Stop pending since: a forced rail closes the fact alone, its flag stays its Stop door.
                        admit=((lambda: author_intake_accepting(task_drive, tid)) if remote_author else
                               owner_accepts_now if owner_key else None),
                    )
                except MailboxWriteUnconfirmed:
                    unconfirmed_write = True
                    raise
            if not wrote:
                return ""
            message_written = True
            if not note and direct_lock_held:
                direct_agent._owner_message_generation = int(
                    getattr(direct_agent, "_owner_message_generation", 0) or 0
                ) + 1
            elif not note:
                if owner_key and isinstance(active_fence, dict):  # after the mail lock, never inside it (Q→mail)
                    _queue_lock.acquire()
                    queue_lock_held = True
                from ouroboros.work_reply import owner_message_landed
                fence_generation_changed = owner_message_landed(active_fence)
        finally:
            if queue_lock_held:
                _queue_lock.release()
            if direct_lock_held:
                direct_lock.release()
            if cancel_refused_in_txn:
                # After the lock release: unlinking staged files is file I/O the
                # global queue lock should not wait on.
                _drop_staged_inputs()
            elif staged_manifest and not (message_written or unconfirmed_write):
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
    except MailboxWriteUnconfirmed:
        if isinstance(task_metadata, dict):
            task_metadata["_routing_target_label"] = target_label
            task_metadata["_routing_target"] = task_obj or candidate
            if staged_manifest:
                task_metadata["_attachment_manifest"] = [
                    dict(item) for item in staged_manifest if isinstance(item, dict)
                ]
                task_metadata["_attachment_report"] = attachment_report
        raise
    except Exception:
        log.debug("Mailbox follow-up routing raised", exc_info=True)
        if message_written:
            return tid  # bookkeeping cannot unsend a committed mailbox entry
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
    notice_transport: Optional[Dict[str, Any]] = None,
    source_ref: Optional[Dict[str, Any]] = None,
    source_snapshot: Optional[Dict[str, Any]] = None,
    origin_suppressed: bool = False,
    ingress_unkept: bool = False,
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

    event = {"chat_id": chat_id, "client_message_id": client_message_id, "relay_target": task,
             "source_ref": source_ref, "origin_message_snapshot": source_snapshot, "origin_suppressed": origin_suppressed}
    fields = _routing_receipt_fields(
        ctx, event, action=action, target=target, target_label=target_label, status=status,
        reason=reason, options=options, attachment_manifest=attachment_manifest, unnamed_label=unnamed_label,
    )
    if client_message_id and not origin_suppressed and not ingress_unkept and "source_row_address" not in fields:
        from ouroboros.chat_chain import format_address, row_address
        from ouroboros.dialogue_store import annotation_sources
        from ouroboros.project_dialogue import owner_message_ref_is_valid, resolve_owner_message_source
        from ouroboros.store import StoreError

        try:
            if source_ref is None:
                if source := annotation_sources(ctx.DRIVE_ROOT, [client_message_id]).get((chat_id, client_message_id)):
                    fields["source_row_address"] = source[1]
            elif (owner_message_ref_is_valid(source_ref) and not source_ref.get("row_sha256")
                  and (source_ref["chat_id"], source_ref["client_message_id"]) == (chat_id, client_message_id)):
                if row := resolve_owner_message_source(ctx.DRIVE_ROOT, source_ref):
                    fields["source_row_address"] = format_address(row_address(row))
        except (OSError, StoreError):
            log.debug("Legacy routing annotation source unavailable", exc_info=True)
    # Only a resolved source can paint a bubble, or the by-value ref of this very message when its ingress says its
    # best-effort write kept no row (``ingress_unkept``, ``message_ingress.record_inbound_message``): the bubble its
    # sender just sent, by client id, no address invented. A ref that only matches room and client id proves neither.
    # Legacy durable receipts may remain unplaced for history import.
    by_value = (ingress_unkept and isinstance(source_ref, dict) and not source_ref.get("row_sha256")
                and (source_ref.get("chat_id"), source_ref.get("client_message_id")) == (chat_id, client_message_id))
    paired_unplaced = bool(notice_transport) and not fields.get("source_row_address")
    if paired_unplaced:
        fields["source_row_address"] = ""  # a transport notice, not a row annotation
    live_id = client_message_id if fields.get("source_row_address") or (by_value and not paired_unplaced) else ""
    if persist and not origin_suppressed and not paired_unplaced and (source_ref is None or live_id):
        try:
            from ouroboros.project_dialogue import append_chat_annotation

            append_chat_annotation(
                ctx.DRIVE_ROOT, client_message_id, source_chat_id=chat_id, reason=reason, detail=detail,
                **{key: value for key, value in fields.items() if key in ROUTING_RECEIPT_FIELDS},
            )
        except Exception:
            log.debug("Routing annotation append failed", exc_info=True)
    try:
        ack = getattr(bridge, "send_routing_ack", None)
        if callable(ack):
            from supervisor.events_project_routing import _routing_notice_ack_fields

            # The same carrier fact the supervisor's receipt writer states (#966).
            notice_event = event if notice_transport else {**event, "client_message_id": live_id}
            notice = _routing_notice_ack_fields(ctx, notice_event, fields, notice_transport=notice_transport)
            ack(chat_id, client_message_id=live_id, **fields, **notice)
        else:
            broadcast = getattr(bridge, "broadcast", None)
            if callable(broadcast):
                broadcast({"type": "message_annotation", "annotation_type": "routing_ack", "chat_id": int(chat_id or 0),
                           "client_message_id": live_id, **fields, "suppress_bubble": True})
    except Exception:
        log.debug("Routing receipt broadcast failed", exc_info=True)


def _settle_refused(ctx: Any, chat_id: int, client_message_id: str, reason: str) -> None:
    """A refused owner message's named operation ends there, after any restart, as its notice said
    (``dialogue_store.refuse_dispatch``); best effort: the notice already went."""
    from ouroboros import dialogue_store
    from ouroboros.project_dialogue import routing_refusal_cause

    try:
        dialogue_store.refuse_dispatch(ctx.DRIVE_ROOT, chat_id, client_message_id,
                                       routing_refusal_cause("project_route", "project_unavailable", reason))
    except Exception:
        log.warning("Refused owner message %s could not be recorded as ended", client_message_id, exc_info=True)


def _agent_note(incoming: Dict[str, Any], origin_ref: Any) -> Dict[str, Any]:
    """An agent's note (its accepted row's recorded author, ``server`` reads it before any command): ``{author,
    row_address}`` and any ``reply_reference`` named ingress already resolved (never supplied to the mailbox as
    authority), else ``{}``. Such a message is routed as any Main message, but under an agent's name: it carries
    no owner-door stamp, invented or real (its own turn is an ordinary conversation turn whose host-only
    ``admission_author`` says what opened it), reaches a running reader as an agent's note, and is never carried down
    a tree as my human's words."""
    from ouroboros.chat_chain import format_address

    author = incoming.get("declared_author") or {}
    reply = (incoming.get("task_metadata") or {}).get("reply_reference")
    return {"author": author, "row_address": format_address(origin_ref) if (origin_ref or {}).get("row_sha256") else "",
            **({"reply_reference": reply} if reply else {})} if author else {}


def _deliver_late_live(ctx: Any, root: pathlib.Path, binding: dict, tid: str, text: str, ref: str,
                       note: dict) -> str:
    """One addressed append, ordered with this attempt's final drain. Ambiguous appends never move to another turn."""
    from ouroboros.conversation_lease import author_intake_accepting
    from ouroboros.extension_late_result import current_target
    from ouroboros.owner_mailbox import PROVENANCE_AGENT_NOTE, write_task_message
    from ouroboros.room_reader import is_reading
    from ouroboros.store import acquisition_budget
    from supervisor.queue_timeouts import _task_drive_for_task

    running = ctx.RUNNING.get(tid)
    if not running or running.get("_unpublished") or running.get("_inline_bind"):
        return ""
    task = running.get("task") or {}
    drive = _task_drive_for_task(task, tid, canonical=root)
    attempt = int(running.get("attempt") or 1)
    may_have_written = False

    def admits() -> bool:  # called under the SAME mailbox lock as the loop's close
        nonlocal may_have_written
        with acquisition_budget(0):
            may_have_written = (ctx.RUNNING.get(tid) is running
                                and author_intake_accepting(drive, tid, attempt=attempt)
                                and is_reading(drive, tid, task, running=True, canonical=root,
                                               fences=getattr(ctx, "BUDGET_ROOT_FENCES", None))
                                and current_target(root, binding) == tid)
        return may_have_written

    if write_task_message(drive, text, tid, source_task_id="", provenance=PROVENANCE_AGENT_NOTE,
                          msg_id=f"{ref}:{tid}", agent_note={**note, "late_report_ref": ref,
                          "late_reader_attempt": attempt}, admit=admits):
        return "handed_off"
    return "unconfirmed" if may_have_written else ""


def _late_result_handoff(ctx: Any, root: pathlib.Path, chat_id: int, ref: str, text: str, note: dict) -> str:
    from ouroboros.extension_late_result import (
        LateResultUnavailable, LateResultWaiting, binding_for_report, current_source, return_turn_metadata,
    )
    from ouroboros.store import StoreBusy, StoreMigrating, acquisition_budget

    binding = binding_for_report(root, ref)
    if binding["chat_id"] != chat_id:
        raise LateResultUnavailable("source room changed")
    note = {**note, "late_result_origin": binding["task_id"]}
    def source():
        try:
            with acquisition_budget(0):
                return current_source(root, binding)
        except (StoreBusy, StoreMigrating) as exc:
            raise LateResultWaiting("source read is temporarily unavailable") from exc

    target, rows = source()
    if target:
        if outcome := _deliver_late_live(ctx, root, binding, target, text, ref, note):
            return outcome
        target, rows = source()
        if target:
            # A still-running task may be between its closed intake and its terminal record. An exact
            # retry may use the finished path; no other reader or fresh allowance may take this letter.
            raise LateResultWaiting("source is not reading yet")
    try:
        with acquisition_budget(0):
            metadata = {"client_message_id": ref, **return_turn_metadata(root, binding, rows, settling=ctx.RUNNING)}
    except (StoreBusy, StoreMigrating) as exc:
        raise LateResultWaiting("source allowance is temporarily unavailable") from exc
    if binding["project_id"]:
        metadata["project_id"] = binding["project_id"]
    meta = ctx.handle_chat_direct(chat_id, text, None, task_metadata=metadata,
                                  admission_author={**note, "late_report_ref": ref})
    return "handed_off" if meta and not meta.get("_unpublished") else "unconfirmed"


def _route_late_result(ctx: Any, chat_id: int, ref: str, text: str, note: dict, *, claim: str) -> None:
    """Settle only this dispatcher claim; a definite deferral keeps the original named obligation."""
    from ouroboros import dialogue_store
    from ouroboros.extension_late_result import (LateResultUnavailable, LateResultWaiting, binding_for_report,
                                                project_handoff, record_handoff)
    from ouroboros.process_custody import current_custody_session_id
    from ouroboros.store import StoreBusy, StoreMigrating, acquisition_budget
    from supervisor.message_ingress import LATE_RESULT_DISPATCHER

    root = pathlib.Path(ctx.DRIVE_ROOT)
    process = current_custody_session_id()
    if not claim:
        return  # an unbound legacy event cannot take a fresh handoff
    try:
        try:
            with acquisition_budget(0):
                facts = dialogue_store.dispatch_facts(root, chat_id, ref) or {}
        except (StoreBusy, StoreMigrating):
            record_handoff(root, ref, {"claim": claim, "process": process, "dispatcher": LATE_RESULT_DISPATCHER}, "defer")
            return  # no handoff was attempted; the exact retry may project this decision
        if (facts.get("dispatcher"), facts.get("state"), facts.get("process"), facts.get("claim")) != (
                LATE_RESULT_DISPATCHER, "dispatched", process, claim):
            return
        decision = binding_for_report(root, ref).get("handoff") or {}
        if decision.get("claim") == facts["claim"]:
            project_handoff(root, chat_id, ref, decision)
            return
        if decision and decision.get("outcome") != "defer":
            return  # only a proven no-effect claim can lead to another dispatch
        try:
            outcome = _late_result_handoff(ctx, root, chat_id, ref, text, note)
        except LateResultWaiting:
            outcome = "defer"
        except LateResultUnavailable:
            outcome = "saved"
        except (OSError, ValueError, RuntimeError):
            log.warning("Unconfirmed late result handoff %s", ref, exc_info=True)
            outcome = "unconfirmed"
        project_handoff(root, chat_id, ref, record_handoff(root, ref, facts, outcome))
    except (StoreBusy, StoreMigrating):
        return  # a recorded decision rejoins through the same report, never another handoff
    except (OSError, ValueError, RuntimeError):
        log.warning("Late result %s could not record its handoff outcome", ref, exc_info=True)


def _transport_client_surface(metadata, incoming, chat_id, source):
    """Stamp the channel for a real non-web owner message without replacing its surface. A synthetic A2A chat
    (negative id) is machine traffic: no owner sent it, so it never wears an owner client fact."""
    from ouroboros.contracts.chat_id_policy import is_a2a_chat_id

    if source == "web" or is_a2a_chat_id(chat_id) or isinstance(metadata.get("client_surface"), dict):
        return metadata
    received = {"received_at": str(incoming["received_at"])} if incoming.get("received_at") else {}
    return {**metadata, "client_surface": {"channel": source, **received}}


def _route_owner_message(bridge: Any, ctx: Any, incoming: Dict[str, Any]) -> None:
    """Route one non-command owner message through the canonical decision lane."""
    chat_id = int(incoming["chat_id"])
    text = str(incoming.get("text") or "")
    image_caption = str(incoming.get("image_caption") or "")
    client_message_id = str(incoming.get("client_message_id") or "")
    origin_ref = incoming.get("origin_message_ref") or None
    origin_suppressed = origin_ref is None
    task_metadata = dict(incoming.get("task_metadata") or {})
    note = _agent_note(incoming, origin_ref)
    if note and (incoming.get("accepted_source_row") or {}).get("late_report_sha256"):
        _route_late_result(ctx, chat_id, client_message_id, text, note,
                           claim=str(task_metadata.get("_late_dispatch_claim") or ""))
        return
    unkept = incoming.get("origin_unkept") is True  # the ingress's own fact: no row was kept for these words
    source_snapshot = incoming.get("origin_message_snapshot") or {}
    image_data = incoming.get("image_data")
    task_constraint = incoming.get("task_constraint")
    # A process-local value minted by the authenticated call door, after
    # canonical message acceptance. JSON/API metadata cannot construct it.
    realtime_turn = task_metadata.pop("_realtime_turn", None)
    if realtime_turn is not None:
        from ouroboros.live_conversation import LiveConversationTurn

        if not isinstance(realtime_turn, LiveConversationTurn):
            raise ValueError("invalid call authority")
    # The owner bot's accepted Reply carries the room the host decided once (``chat_references.telegram_reply_room``):
    # a Project opens this message's own turn there, an unavailable target is refused, never sent to Main.
    reply_ref = task_metadata.get("reply_reference")
    telegram_room = (incoming.get("source") == "skill:telegram" and task_metadata.get("_host_operation") is True
                     and isinstance(reply_ref, dict) and reply_ref.get("telegram_room_chat_id", 0) != 0)
    ingress_chat = chat_id  # the owner's row, its receipts and refusals stay in the conversation he wrote in
    from ouroboros.chat_references import TURN_ADDRESS, decided_reply_address, telegram_chat_row

    physical = not telegram_room and telegram_chat_row(incoming.get("source"), chat_id, client_message_id)
    task_metadata.pop("work_reply_fallback", None)  # host-only; never supplied by ingress metadata
    if not telegram_room:  # only an exact Reply's own turn carries the address its acceptance gave it
        task_metadata.pop(TURN_ADDRESS, None)
    decided_project = {}
    if telegram_room:
        from ouroboros.project_admission import reserved_project_for_chat

        room = reply_ref["telegram_room_chat_id"]
        confirmed_transport = decided_reply_address(incoming.get("accepted_source_row") or {})
        try:
            if type(room) is int and room > 0:
                decided_project = reserved_project_for_chat(ctx.DRIVE_ROOT, room)
        except (OSError, ValueError):
            pass
        if decided_project.get("lifecycle") != "active":
            _record_routing_receipt(bridge, ctx, chat_id=ingress_chat, client_message_id=client_message_id,
                                    action="project_route", status="project_unavailable",
                                    reason="reply_target_unavailable",
                                    notice_transport=confirmed_transport, source_ref=origin_ref,
                                    source_snapshot=source_snapshot, origin_suppressed=origin_suppressed, ingress_unkept=unkept)
            _settle_refused(ctx, ingress_chat, client_message_id, "reply_target_unavailable")
            return
        if confirmed_transport and not task_metadata.get(TURN_ADDRESS):
            task_metadata[TURN_ADDRESS] = confirmed_transport
        chat_id = room
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
            "objective": text or image_caption or (incoming.get("origin_message_snapshot") or {}).get("placeholder_text", ""),
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
        if telegram_room:
            from ouroboros.chat_references import reply_transport_from_metadata

            event["project_id"] = decided_project["id"]
            if transported := reply_transport_from_metadata(ctx.DRIVE_ROOT, metadata):
                event[TURN_ADDRESS] = transported
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
        if isinstance(origin_ref, dict) and origin_ref:
            event["source_ref"] = origin_ref
            event["source_text"] = str(incoming.get("log_text") or "")
            if incoming.get("origin_message_snapshot"):
                event["origin_message_snapshot"] = incoming["origin_message_snapshot"]
            if metadata.get("chat_attachment_uploads"):
                event["attachment_uploads"] = metadata["chat_attachment_uploads"]
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
    try:  # the owner bot's own chat id is a Telegram address, never a room's: only a decided Reply leaves it
        reserved_project = decided_project if telegram_room else ({} if physical else _reserved_project_for_chat(ctx, chat_id))
    except (OSError, ValueError) as exc:
        if incoming.get("bound_external_owner") is not True:
            _record_routing_receipt(
                bridge, ctx, chat_id=ingress_chat, client_message_id=client_message_id,
                action="project_route", status="project_unavailable",
                source_ref=origin_ref, source_snapshot=source_snapshot,
                origin_suppressed=origin_suppressed, ingress_unkept=unkept,
                reason="project_routing_fence_lookup_failed", detail=str(exc))
            return
        # The bridge matched the stored owner pair at ingress; an unreadable room registry
        # cannot take away that Main conversation. A Project Reply was resolved separately above.
        reserved_project = {}
    project_id = (
        str(reserved_project.get("id") or "")
        if reserved_project.get("lifecycle") == "active"
        else ""
    )
    if reserved_project and not project_id:
        _record_routing_receipt(
            bridge,
            ctx,
            chat_id=ingress_chat,
            client_message_id=client_message_id,
            action="project_route",
            target=str(reserved_project.get("id") or ""),
            status="project_unavailable",
            reason="project_unavailable",
            source_ref=origin_ref,
            source_snapshot=source_snapshot,
            origin_suppressed=origin_suppressed, ingress_unkept=unkept,
        )
        return
    # An owner message never wakes consciousness (owner decision В13): Main answers it itself.
    task_metadata = _scoped_task_metadata(project_id, task_metadata)
    task_metadata["client_message_id"] = client_message_id
    if not note:
        task_metadata["work_kind"] = "owner"
    task_metadata.pop("origin_message_snapshot", None)
    task_metadata.pop("origin_message_ref", None)
    task_metadata.pop("origin_suppressed", None)
    # The turn's origin identity rides UNCONDITIONALLY (not only when the
    # decision lane runs): a bare direct turn with no projects/roots yet — the
    # first-ever project creation — must still carry it so promote/route/bind
    # receive the ref by value.
    if isinstance(origin_ref, dict) and origin_ref and not note:
        task_metadata = {
            **(task_metadata or {}),
            "origin_message_ref": origin_ref,
            "origin_message_text": "" if source_snapshot.get("placeholder_text") else str(incoming.get("log_text") or ""),
            **({"origin_message_snapshot": source_snapshot} if source_snapshot else {}),
        }
    elif not note:
        # A suppressed (never-logged) message has a DESIGNED absence of origin;
        # downstream binders must not classify it as a producer bug.
        task_metadata = {**(task_metadata or {}), "origin_suppressed": True}
    if isinstance(task_metadata.get("ui_control"), dict):
        # A press composed these words: the act stamped the turn, the words stay the host's (never initial_user).
        task_metadata["objective_author"] = {"kind": "ui_control", "control": str(task_metadata["ui_control"].get("name") or "")}
    from ouroboros.contracts.chat_id_policy import is_a2a_chat_id as _is_a2a

    _ingress_source = str(incoming.get("source") or "web")
    task_metadata = _transport_client_surface(task_metadata, incoming, chat_id, _ingress_source)
    from ouroboros.card_store import late_press_ref
    from ouroboros.owner_quiz import late_answer_ref

    late, press = task_metadata.get("late_answer"), task_metadata.get("card_press")
    thread_of = (late_answer_ref(late) or late_press_ref(press) or {}).get("task_id", "")
    reply = raw_reply if isinstance(raw_reply := task_metadata.get("reply_reference"), dict) else {}
    addressed = (_ingress_source == "web" and realtime_turn is None and (chat_id == 1 or bool(project_id))
                 and not (note or late or press or task_constraint or _is_a2a(chat_id))  # owner words only
                 and isinstance(origin_ref, dict) and bool(origin_ref.get("row_sha256")) and "work_task_id" in reply)
    routed_to_task, unnamed_label, delivery_status = "", "Task", "delivered"
    roots = addressable_roots(ctx)
    delivery, work_fallback = None, False
    if addressed:
        from ouroboros.work_reply import (
            carry_reply_context,
            carry_reply_to_members,
            route_work_reply,
            work_reply_fallback,
        )

        delivery = route_work_reply(ctx, reply, origin_ref, incoming.get("accepted_source_row"))
        work_fallback = delivery.outcome == "fallback"
        if delivery.outcome == "delivered":
            # The addressed work's own running helpers hear it now as passive context (its drain's catch-up writes
            # the same bytes); a physical retry's helpers live under the logical root. No other work is woken.
            carry_reply_to_members(ctx.DRIVE_ROOT, delivery)
        if work_fallback:
            task_metadata["work_reply_fallback"] = work_reply_fallback(delivery)
        else:
            routed_to_task, delivery_status = delivery.target, delivery.outcome
            task_metadata["_routing_target"] = delivery.task
    # A live call's admitted turn keeps the room it was opened in; it never re-routes. The owner bot's exact
    # Reply is answered by its own turn: no running work, here or in his chat, reads it as its input. The Project's
    # other live roots hear it as context, as any word of that room, read from the chat he wrote it in; none of them
    # is chosen to answer it, and its delivery address stays its own turn's.
    if telegram_room:
        if project_id and not press and not origin_suppressed:
            _carry_project_room_context(ctx, chat_id, project_id, ingress_chat, client_message_id, roots, "",
                                        reply=task_metadata.get("reply_reference"), source_ref=origin_ref)
    elif not addressed and realtime_turn is None and ((project_id and not late) or thread_of):
        routed_to_task = _route_project_chat_to_running_task(
            ctx, chat_id, text or image_caption, client_message_id,
            task_metadata=task_metadata, image_data=image_data, thread_of=thread_of, roots=roots)
        # The room's other live roots hear my human's words as context, whichever return routed them, a late answer
        # included; a press is a typed act on a card, not words: it reaches its author's thread and no other reader.
        if project_id and not press and not origin_suppressed:
            _carry_project_room_context(ctx, chat_id, project_id, chat_id, client_message_id, roots,
                                        routed_to_task, reply=task_metadata.get("reply_reference"), source_ref=origin_ref)
    elif (not addressed or work_fallback) and realtime_turn is None and not (project_id or late or press or task_constraint):
        # Main: my human's next words reach the one turn already answering them in this chat, else their own turn.
        try:
            routed_to_task = _route_main_followup(
                ctx, chat_id, text or image_caption, client_message_id,
                task_metadata=task_metadata, image_data=image_data, roots=roots, note=note)
        except MailboxWriteUnconfirmed as exc:
            routed_to_task, delivery_status = exc.task_id, "unconfirmed"
        unnamed_label = ""  # its turn named nothing yet: the receipt says saved, never a placeholder name
    # An unconfirmed fallback write leaves it unproven that a turn took the Reply: nobody is told the room handles it.
    if delivery is not None and not (work_fallback and delivery_status == "unconfirmed"):
        carry_reply_context(ctx, roots, ingress_chat, delivery,
                            exclude=(routed_to_task,) if work_fallback and routed_to_task else ())
    if routed_to_task:
        # A direct Reply has one recipient: the addressed work answers it. Its tree hears it as passive context carried
        # at receipt (``work_reply.carry_reply_to_members``) and reconciled by its own drain
        # (``owner_mailbox.member_carriage``), so no member is woken here; an ordinary room delivery still carries its
        # accepted row at receipt below: my human's words alone, never an agent's note (inv. 26).
        if (not origin_suppressed and not note and delivery_status == "delivered"
                and (not addressed or work_fallback)):
            from ouroboros.owner_relay import carry_row_at_receipt

            carry_row_at_receipt(ctx.DRIVE_ROOT, routed_to_task, chat_id, client_message_id,
                                 reply=task_metadata.get("reply_reference"), source_ref=origin_ref)
        _record_routing_receipt(
            bridge, ctx, chat_id=chat_id, client_message_id=client_message_id, action="mailbox_delivery",
            target=routed_to_task, target_label=str(task_metadata.get("_routing_target_label") or ""),
            status=delivery_status, reason="mailbox_write_unconfirmed" if delivery_status == "unconfirmed" else "",
            detail=str(task_metadata.get("_attachment_report") or ""),
            attachment_manifest=list(task_metadata.get("_attachment_manifest") or []),
            task=task_metadata.get("_routing_target"), unnamed_label=unnamed_label,
            source_ref=origin_ref, source_snapshot=source_snapshot,
            origin_suppressed=origin_suppressed, ingress_unkept=unkept)
        return

    try:
        from ouroboros.projects_registry import list_projects

        has_projects = bool(list_projects(ctx.DRIVE_ROOT))
    except Exception:
        log.warning("Unable to inspect Projects for owner routing", exc_info=True)
        has_projects = True
    # Unknown roots, like unknown Projects, go to the decision lane, which is told so. So does any
    # root's record: Main without Projects reads a stop and its open Continue as Main with them does.
    needs_decision_lane = (addressed or bool(project_id) or has_projects or bool(roots.rows()) or bool(roots.gap)
                           or _root_results_exist(ctx))
    if needs_decision_lane:
        task_metadata = _decision_turn_metadata(ctx, chat_id, client_message_id, task_metadata, roots,
                                                decided_project_id=project_id)

    if realtime_turn is not None:
        # A live call keeps a thread of this process (``executor=inline_call``): its host port is process-local.
        threading.Thread(target=ctx.handle_chat_direct, args=(chat_id, text or image_caption, image_data),
                         kwargs={"task_constraint": task_constraint, "task_metadata": task_metadata,
                                 "realtime_turn": realtime_turn}, daemon=True).start()
        return
    # The conversation lane's admission: the RUNNING row and the start receipt here, the
    # turn's body in a carrier process (owner decision 1A). No thread per owner turn.
    ctx.handle_chat_direct(chat_id, text or image_caption, image_data,
                           task_constraint=task_constraint, task_metadata=task_metadata,
                           **({"admission_author": note} if note else {}))
