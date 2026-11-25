"""``POST /api/decisions`` — the ONE owner decision-card answer ingress.

Owner decision 1=A: one UI component (``chat_decision.js``) and one incoming
answer contract with an idempotent ``request_id``; ``decision_id`` composes
EXISTING identities per family instead of minting a durable registry:

- ``quiz:{task_id}:{quiz_id}`` — served here (#Q-2b);
- ``routing:{client_message_id}:{routing_token}`` — the #198 picker family,
  dispatched to ``gateway/routing_decision.py``;
- ``interaction:{task_id}:{run_id}:{interaction_id}`` — RESERVED. #204 is
  served by the escalation hierarchy instead (owner decision 31): a delegated
  run's question wakes its nanny, who answers from task context via
  delegate_answer or escalates upward with the escalate verb — the owner sees
  a quiz card only when no ancestor answers, so no direct interaction card
  exists and this family stays a typed 501.

The quiz path mirrors the hurry ingress split (``gateway/task_hurry.py``):
projection write first (request-id idempotent, first answer wins), then the
typed ``KIND_QUIZ_ANSWER`` mailbox control on the task's physical drive, then
the live ``quiz_state`` broadcast.

A LATE answer — the card's task already finished, so its block is structurally
``expired_terminal`` — is ACCEPTED here (owner decision В17a=A, retiring the
earlier "a quiz dies with its author"): the projection records it with
``answered_after_terminal``, and because no mailbox will ever be drained the
answer is delivered as the owner's OWN message into the card's chat through the
named ingress ``supervisor.message_bus.accept_local_message``. The 2xx then says
``forwarded`` so the surface never claims a delivery that did not happen. A 409
is left for what is genuinely settled: an already answered card (first-wins) and
a non-root addressee.
"""

from __future__ import annotations

import asyncio
import logging
import pathlib
from typing import Any, Dict, Optional, Tuple

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros.gateway._helpers import request_drive_root, request_json_or, run_sync_to_completion
from ouroboros.task_results import resolve_task_lineage, validate_task_id

log = logging.getLogger(__name__)

_REQUEST_ID_MAX = 128
_COMMENT_MAX = 2000
_SERVED_FAMILIES = {"quiz", "routing"}
_KNOWN_FAMILIES = {"quiz", "routing", "interaction"}


def _parse_quiz_decision_id(decision_id: str) -> Tuple[str, str, str]:
    """Split ``quiz:{task_id}:{quiz_id}`` → (family, task_id, quiz_id)."""
    parts = str(decision_id or "").split(":", 2)
    family = parts[0] if parts else ""
    if family != "quiz" or len(parts) != 3 or not parts[1] or not parts[2]:
        return family, "", ""
    return family, parts[1], parts[2]


def _live_root_task(task_id: str, *, holds_mail: bool = False) -> Tuple[Optional[Dict[str, Any]], str]:
    """Snapshot queue memory under its lock, then read live-author storage outside it.

    ``task_not_live`` is NOT a hard refusal for a quiz answer — the caller
    consults the durable projection for the honest late-answer state; nor is
    ``task_settled``, a RUNNING worker whose result already settled (its
    mailbox is no longer drained, TZ-2 D15). ``task_not_reading`` is alive
    (queued, held, owner- or money-paused) but reads no question mail now; an
    unheld cold sleep still reads it here, never in room intake (D6). With
    ``holds_mail`` that living row comes back too: addressed question mail
    waits in its mailbox until it reads, and no hold is released."""
    from supervisor import queue as q

    with q._queue_lock:
        task: Optional[Dict[str, Any]] = None
        meta = q.RUNNING.get(task_id) if isinstance(q.RUNNING, dict) else None
        if isinstance(meta, dict) and isinstance(meta.get("task"), dict):
            task = dict(meta["task"])
        if task is None:
            task = next(
                (
                    dict(row) for row in list(q.PENDING or [])
                    if isinstance(row, dict) and str(row.get("id") or "") == task_id
                ),
                None,
            )
        if task is None:
            from supervisor.workers import direct_chat_turn

            task = direct_chat_turn(task_id)
    if task is None:
        from ouroboros.conversation_lease import presence_author_agent
        from ouroboros.task_results import load_task_result
        agent = presence_author_agent(q.DRIVE_ROOT, task_id)
        if agent is not None and getattr(agent, "_busy", False) and getattr(agent, "_current_task_id", "") == task_id:
            if not getattr(agent, "_accepting_owner_messages", False):
                return None, "task_settled"
            task = load_task_result(q.DRIVE_ROOT, task_id, strict=True)
        if task is None:
            return None, "task_not_live"
    lineage = resolve_task_lineage(
        task_id,
        metadata=task.get("metadata"),
        root_task_id=task.get("root_task_id"),
        parent_task_id=task.get("parent_task_id"),
        delegation_role=task.get("delegation_role"),
        original_task_id=task.get("original_task_id"),
        timeout_retry_from=task.get("timeout_retry_from"),
    )
    if not bool(lineage["is_root_task"]):
        # Decision-31 hierarchy: owner quiz cards come only from ROOT
        # tasks (a subagent escalates to its parent, never to a card).
        return None, "not_a_root_task"
    from supervisor.queue import _task_drive_for_task

    from ouroboros.owner_mailbox import mailbox_drain_ended

    if mailbox_drain_ended(_task_drive_for_task(task, task_id), task_id):
        # Still RUNNING for paid post-work, but the solve loop settled: nothing
        # drains the mailbox any more — the same fact that refuses owner mail
        # sends the answer down the late path instead of into a dead mailbox.
        return None, "task_settled"
    from ouroboros.room_reader import is_reading, unheld_cold_sleep
    from supervisor.workers import direct_chat_turn
    drive = _task_drive_for_task(task, task_id)
    from ouroboros.conversation_lease import presence_author_agent
    running = (task_id in q.RUNNING or direct_chat_turn(task_id) is not None
               or presence_author_agent(q.DRIVE_ROOT, task_id) is not None)
    if not (is_reading(drive, task_id, task, running=running) or unheld_cold_sleep(drive, task_id, task)):
        return (task if holds_mail else None), "task_not_reading"
    return task, ""


def _mail_holder(task_id: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """The living work an addressed answer waits for: reading now, or paused, held or queued."""
    return _live_root_task(task_id, holds_mail=True)


def _quiz_answer_frame(
    block: Dict[str, Any], option_index: Optional[int], comment: str, *, asker: str = "",
) -> str:
    """Host-authored structural frame around the owner's VERBATIM answer.

    Thin alias of the ONE shared builder ``owner_quiz.quiz_answer_frame``, so
    the live mailbox control here and the late-answer delivery in the drain
    render the same words; ``asker`` names whose text a pressed label is."""
    from ouroboros.owner_quiz import quiz_answer_frame

    return quiz_answer_frame(block, option_index, comment, asker=asker)


def _send_quiz_state(quiz_id: str, task_id: str, state: str, **fields: Any) -> None:
    """Best-effort live card update; the durable projection is the truth."""
    try:
        from supervisor.message_bus import get_bridge

        get_bridge().send_quiz_state(quiz_id, task_id, state, **fields)
    except Exception:
        log.debug("quiz_state broadcast failed for %s", quiz_id, exc_info=True)


def _refused(message: str, status: int, **extra: Any) -> Tuple[int, Dict[str, Any]]:
    """A typed refusal as a ``(status, payload)`` pair (the gateway error shape)."""
    payload: Dict[str, Any] = {"error": message}
    payload.update(extra)
    return status, payload


def _answer_fields(block: Dict[str, Any]) -> Dict[str, Any]:
    """The recorded winner as every receipt and card frame names it: option, words, and the files a Telegram
    Reply carried (``chat_uploads.answer_files``: no stored id), each only when recorded."""
    from ouroboros.chat_uploads import answer_files

    return {**({"answered_index": block["answered_index"]} if isinstance(block.get("answered_index"), int) else {}),
            **({"comment": str(block["comment"])} if str(block.get("comment") or "") else {}),
            **({"attachments": answer_files(block["attachments"])} if block.get("attachments") else {})}


def _answer_pending(root: pathlib.Path, task_id: str, quiz_id: str, reason: str,
                    message: str) -> Tuple[int, Dict[str, Any]]:
    """A failed effect must leave the accepted answer visible as owed."""
    from ouroboros.owner_quiz import question_delivery_projection, quiz_states

    block = quiz_states(root, task_id).get(quiz_id) or {}
    if block.get("state") != "answered":
        return _refused("The answer could not be confirmed. Retry the same answer when storage is available.",
                        503, task_id=task_id, reason_code="quiz_answer_unconfirmed",
                        **({"state": block["state"]} if block.get("state") else {}))
    fields = {"question_delivery": question_delivery_projection(block), **_answer_fields(block)}
    _send_quiz_state(quiz_id, task_id, "answered", **fields)
    return _refused(message, 503, task_id=task_id, state="answered", reason_code=reason, **fields)


def _record_quiz_answer_history(
    drive_root: pathlib.Path, task_id: str, task: Optional[Dict[str, Any]],
    block: Dict[str, Any], *, duplicate: bool,
) -> None:
    """Keep the winning answer in canonical dialogue beyond quiz/mailbox GC.

    The canonical delivery receipt avoids history scans on maintenance retries.
    A crash after append and before receipt may leave the same logical source twice;
    the existing dialogue identity projection deduplicates those physical rows.
    """
    from ouroboros.owner_quiz import mark_quiz_history_written
    from supervisor.log_addressing import address_task_event
    from supervisor.message_bus import log_chat

    source_id = f"quiz_answer:{task_id}:{block['quiz_id']}"
    if (block.get("delivery") or {}).get("history_at"):
        return
    if task is None:
        from ouroboros.task_results import load_task_result

        task = load_task_result(drive_root, task_id) or {}
    address = address_task_event({task_id: {"task": task}}, drive_root, {"task_id": task_id})
    published_chat, _reason = _late_answer_destination(drive_root, task_id, block)
    if published_chat is None:
        raise ValueError(_reason or "publication_address_unknown")
    address["chat_id"] = published_chat
    index = block.get("answered_index")
    stored = log_chat(
        "system", address.get("chat_id"), 0,
        _quiz_answer_frame(block, index if isinstance(index, int) else None, str(block.get("comment") or ""),
                           asker=task_id),  # the room's row: every reader is told whose text the label is
        ts=str(block["answered_at"]), source="owner_quiz_answer", task_id=task_id,
        client_message_id=source_id, record_type="quiz_answer", quiz=dict(block),
        message_meta=address, drive_root=drive_root, require_write=True,
    )
    mark_quiz_history_written(drive_root, task_id, str(block["quiz_id"]), source_row=stored)


def _answer_surface(drive_root: pathlib.Path, body: Dict[str, Any], source: str) -> Optional[Dict[str, str]]:
    """The Telegram bot and chat an answer came from, as the bundled bridge relays it (``surface``); its late answer's
    turn replies there (``chat_references.late_answer_surface``). None from any other source or shape."""
    from ouroboros.chat_references import TELEGRAM_SOURCE, bundled_bridge

    surface = body.get("surface")
    if source != TELEGRAM_SOURCE or not isinstance(surface, dict) or surface.get("kind") != "telegram":
        return None
    bot, chat = str(surface.get("bot_id") or ""), str(surface.get("conversation_id") or "")
    if not (bot.isdigit() and chat.lstrip("-").isdigit()) or not bundled_bridge(drive_root):
        return None
    return {"kind": "telegram", "conversation_id": chat, "bot_id": bot}


def _late_answer_destination(
    drive_root: pathlib.Path, task_id: str, block: Dict[str, Any],
) -> Tuple[Optional[int], str]:
    """The card's own chat, or the addressing its history row already uses.

    Canonical chat-id policy: synthetic A2A traffic has no owner turn to start
    (``send_quiz`` refuses those ids too), and the hidden partition is a real
    destination that no browser surface reads — an owner message there would
    never be seen. Both are skipped honestly instead of being invented into
    Main.
    """
    from ouroboros.contracts.chat_id_policy import HIDDEN_CHAT_ID, is_a2a_chat_id

    raw = block.get("publication_chat_id")
    if type(raw) is not int or block.get("publication_evidence") not in {"pre_exposure", "history"}:
        # An ask-time chat or today's Project binding cannot prove where an old
        # card was exposed. The explicit startup import may attach an exact
        # history witness; absent that, retain a routing gap and never guess.
        return None, "publication_address_unknown"
    chat_id = raw
    if is_a2a_chat_id(chat_id):
        return None, "a2a_chat"
    if chat_id == HIDDEN_CHAT_ID:
        return None, "hidden_chat"
    return chat_id, ""


def _forward_late_quiz_answer(
    drive_root: pathlib.Path, task_id: str, quiz_id: str, block: Dict[str, Any], *, source: str = "web",
) -> Tuple[bool, str]:
    """Deliver an accepted late answer as the owner's own message in that chat.

    The asking task is gone, so no mailbox will be drained: the answer becomes
    an ordinary owner turn through the NAMED ingress, whose canonical chat row
    is the acceptance receipt and whose ``client_message_id`` is the
    idempotency key (a crash after acceptance never authorizes another
    enqueue), so a retry of this request re-enters the same delivery instead of
    duplicating it.

    The canonical chat row and the owner's bubble carry the HUMAN's words only
    (``owner_quiz.late_answer_owner_text``: the verbatim comment, else the
    pressed option as ``{n}. {label}``), never the host frame. The typed
    ``late_answer`` provenance rides the message's metadata; the receiving turn
    rebuilds the FULL frame from the stored block for the model
    (``owner_quiz.late_answer_model_text``), so the model still reads the card.

    The typed late_answer metadata routes only to a live head that an earlier
    late answer to this asker started, never to another reader; otherwise a
    fresh head receives the complete frame.
    """
    chat_id, reason = _late_answer_destination(drive_root, task_id, block)
    if chat_id is None:
        return False, reason
    from ouroboros.chat_uploads import staging_specs, stored_refs
    from ouroboros.owner_quiz import late_answer_owner_text

    text, files = late_answer_owner_text(block), stored_refs(block.get("attachments"))
    if not text and not files:
        # An answered block always has a comment, a valid index or a file; a block
        # that has none cannot be spoken as the owner's message.
        return False, "answer_unreadable"
    # Snapshot current related work at delivery, not at ask time: a child may
    # have been promoted after the question. The owner's Main turn stays unrestricted.
    from ouroboros.presence_owner_contact import private_main_contact
    from ouroboros.task_results import load_task_result
    snapshot = dict(block)
    # A Reply's file rides the owner's message as its own: the row keeps the recorded refs (a file alone takes the
    # host placeholder, never words), and the turn stages those exact bytes from the store.
    metadata = {"late_answer": {"task_id": task_id, "quiz_id": quiz_id,
                                "answer_snapshot": {"task_id": task_id, "quiz": snapshot}},
                **({"chat_attachments": files, "chat_attachment_uploads": staging_specs(files, drive_root)}
                   if files else {})}
    if private_main_contact(block, task_id):
        current = load_task_result(drive_root, task_id, strict=True) or {}
        child = ((current.get("metadata") or {}).get("presence_work_ref")
                 or (current.get("presence_continuation") or {}).get("child_work_ref"))
        contact = dict(block["presence_contact"])
        if child:
            contact["child_work_ref"] = str(child)
        snapshot["presence_contact"] = contact
    return deliver_as_owner_message(
        drive_root, chat_id, text, f"quiz_late_answer:{task_id}:{quiz_id}", source=source,
        task_metadata=metadata,
    )


def deliver_as_owner_message(
    drive_root: pathlib.Path, chat_id: int, text: str, client_message_id: str, *,
    source: str = "web", task_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """Accept ``text`` as the owner's own message in ``chat_id`` through the NAMED ingress.

    The canonical chat row is the acceptance receipt and ``client_message_id`` the
    idempotency key (a crash after acceptance never authorizes another enqueue), so a
    retry re-enters the same delivery instead of duplicating it. A late quiz answer
    and a press on a living card whose author ended both arrive this way.
    """
    from supervisor import message_bus

    bridge = message_bus.get_bridge()
    try:
        row, rejoined = message_bus.accept_local_message(
            bridge, drive_root, text,
            chat_id=chat_id, user_id=1, source=str(source or "web"),
            client_message_id=client_message_id,
            **({"task_metadata": task_metadata} if task_metadata else {}),
        )
    except ValueError:
        # This id was already accepted, but under other bytes: a quiz delivery
        # accepted before the row carried only the owner's words (it carried
        # the host frame then), or a card press whose card was retitled since.
        # It WAS delivered once; a retry rejoins it instead of failing forever
        # or enqueueing a second owner turn.
        accepted = message_bus.accepted_chat_message(drive_root, chat_id, client_message_id)
        if accepted is None:
            raise
        from supervisor.message_ingress import dispatch_completed

        return (True, "") if dispatch_completed(accepted) else (False, "late_answer_delivery_unknown")
    if not rejoined:
        # The named ingress accepts and enqueues but does not echo; give the
        # owner the SAME user bubble their own typing produces (a rejoin
        # already echoed when it was first accepted).
        from ouroboros.projects_registry import stamp_project_thread
        from ouroboros.chat_references import owner_frame_fields
        from ouroboros.chat_uploads import attachment_views

        views = attachment_views(row.get("attachments"))
        echo = {
            # The accepted canonical row is the echo's text authority, exactly
            # as the owner's own typing echoes what the ingress recorded; its
            # rooms and quote are the ones a named source's frame carries.
            "type": "chat", "role": "user", "content": str(row.get("text") or text),
            "ts": str(row.get("ts") or ""), "source": str(source or "web"), "chat_id": chat_id,
            "sender_session_id": "", "client_message_id": client_message_id,
            **({"attachments": views} if views else {}),  # its files, as history replays the row
            **({"text_placeholder": True} if views and row.get("text_placeholder") is True else {}),
            **owner_frame_fields(drive_root, chat_id, row),
        }
        if row.get("ingress_accepted") is True:
            echo["ingress_accepted"] = True  # the row's own fact: live matches history replay
        try:
            stamp_project_thread(message_bus.DATA_DIR, echo)
            bridge.broadcast(echo)
        except Exception:
            log.debug("Owner message echo failed for %s", client_message_id, exc_info=True)
    else:
        from supervisor.message_ingress import dispatch_completed

        if not dispatch_completed(row):
            return False, "late_answer_delivery_unknown"
    return True, ""


def _quiz_decision_receipt(decision_id: str, outcome: dict, block: dict) -> Dict[str, Any]:
    """Both dispositions expose the winning canonical fields, never retry payload bytes."""
    from ouroboros.owner_quiz import CLOSURE_FIELDS, question_delivery_projection

    payload = {"ok": bool(outcome.get("ok")), "decision_id": decision_id,
               "state": str(block.get("state") or outcome.get("state") or ""),
               "duplicate": bool(outcome.get("duplicate")),
               **{key: block[key] for key in CLOSURE_FIELDS if key in block}, **_answer_fields(block)}
    if delivery := question_delivery_projection(block):
        payload["question_delivery"] = delivery
    return payload


async def _close_quiz_decision(root: pathlib.Path, task_id: str, quiz_id: str,
                               body: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    """Owner closure shares the canonical first-winner lock and closure recovery."""
    from types import SimpleNamespace
    from ouroboros.owner_quiz import record_closed, quiz_states
    from supervisor.events_chat_delivery import _handle_send_quiz_state
    from supervisor.message_bus import get_bridge

    # One tap closes: the owner's words are optional, and their absence records the standard sentence.
    reason = body.get("reason", "")
    if not isinstance(reason, str) or len(reason) > _COMMENT_MAX:
        return _refused("a closure reason is text of at most 2000 characters", 400, reason_code="close_reason_invalid")
    if "option_index" in body or "comment" in body:
        return _refused("closure carries a reason, without an answer", 400, reason_code="close_payload_invalid")
    try:
        task_id = validate_task_id(task_id)
        if not quiz_id:
            raise ValueError("quiz_id is required")
    except ValueError as exc:
        return _refused(str(exc), 400, reason_code="malformed_decision_id")
    request_id, decision_id = str(body["request_id"]).strip(), str(body["decision_id"]).strip()
    from ouroboros.gateway.secret_decision import cancel_secret, is_secret_question

    if is_secret_question(root, task_id, quiz_id):  # the owner's close of a secret request is its cancel
        return await run_sync_to_completion(cancel_secret, root, decision_id, task_id, quiz_id, request_id)
    try:
        _task, refusal = _live_root_task(task_id)
        if refusal == "not_a_root_task":
            return _refused("quiz closure addresses root tasks only", 409, reason_code=refusal)
        outcome = record_closed(root, task_id, quiz_id=quiz_id, reason=reason,
                                actor="owner", request_id=request_id)
        block = outcome.get("block") or {}
        if outcome.get("ok"):
            try:
                await run_sync_to_completion(_handle_send_quiz_state,
                    {"task_id": task_id, "quiz_id": quiz_id, "owner_retry": True},
                    SimpleNamespace(DRIVE_ROOT=root, bridge=get_bridge()))
            except Exception:
                log.warning("Question closure delivery remains pending for %s", quiz_id, exc_info=True)
            block = quiz_states(root, task_id).get(quiz_id) or block
        payload = _quiz_decision_receipt(decision_id, outcome, block)
        if not outcome.get("ok"):
            error = str(outcome.get("error") or "quiz_close_refused")
            return (404 if error == "quiz_not_found" else 409), {**payload, "error": error}
        _send_quiz_state(quiz_id, task_id, "closed", **{key: value for key, value in payload.items()
                         if key not in {"ok", "decision_id", "state", "duplicate"}})
        return 200, payload
    except Exception as exc:
        return _refused(str(exc), 503, reason_code="quiz_close_failed")


async def _dismiss_quiz_decision(root: pathlib.Path, family: str, task_id: str, quiz_id: str,
                                 body: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    """The owner removes an accepted late answer of unknown delivery from current questions."""
    from ouroboros.owner_quiz import dismiss_unknown_delivery, question_delivery_projection

    if family != "quiz":
        return _refused("removal addresses a quiz", 400, reason_code="dismiss_payload_invalid")
    if any(key in body for key in ("option_index", "comment", "reason")):
        return _refused("removal carries no answer, comment or reason", 400, reason_code="dismiss_payload_invalid")
    try:
        task_id = validate_task_id(task_id)
        if not quiz_id:
            raise ValueError("quiz_id is required")
    except ValueError as exc:
        return _refused(str(exc), 400, reason_code="malformed_decision_id")
    decision_id = str(body["decision_id"]).strip()
    try:
        outcome = dismiss_unknown_delivery(root, task_id, quiz_id=quiz_id, request_id=str(body["request_id"]).strip())
    except Exception as exc:
        return _refused(str(exc), 503, reason_code="quiz_dismiss_failed")
    block = outcome.get("block") or {}
    payload = _quiz_decision_receipt(decision_id, outcome, block)
    if not outcome.get("ok"):
        error = str(outcome.get("error") or "quiz_dismiss_refused")
        return (404 if error == "quiz_not_found" else 409), {**payload, "error": error}
    _send_quiz_state(quiz_id, task_id, str(block.get("state") or "answered"), **_answer_fields(block),
                     question_delivery=question_delivery_projection(block))
    return 200, payload


async def answer_decision(drive_root: pathlib.Path, body: Any, *, source: str = "web", _recovery: bool = False,
                          request: Optional[Request] = None,
                          attachments: Optional[list] = None) -> Tuple[int, Dict[str, Any]]:
    """The ONE decision-answer ingress, transport-neutral: ``(status, payload)``.

    ``POST /api/decisions`` (the browser card) and the loopback Host Service
    ``POST /chat/decision`` (a reviewed transport skill relaying the owner's
    tap or reply, e.g. Telegram — #472) both call this, so every surface gets
    the same idempotent ``request_id`` write, the same first-answer-wins race,
    and the same late-answer handling: accepted, recorded with
    ``answered_after_terminal``, and forwarded into the card's chat as the
    owner's own message (``forwarded`` says whether that happened). 404 stays
    for a card the projection no longer knows, 409 for one already answered.
    ``attachments``: measured refs the Host Service stored for the bundled bridge's Reply file (never a body field);
    they answer only a quiz, with or without words, and are recorded only with this request's own winner.
    """
    if not isinstance(body, dict):
        return _refused("request body must be a JSON object", 400)
    request_id = str(body.get("request_id") or "").strip()
    if not request_id or len(request_id) > _REQUEST_ID_MAX:
        return _refused(
            "request_id is required (a stable client-generated id, reused on retry)",
            400, reason_code="request_id_required",
        )
    decision_id = str(body.get("decision_id") or "").strip()
    if decision_id.split(":", 1)[0] == "model_wait":
        from ouroboros.gateway.task_model_wait import answer_model_wait_decision

        return await answer_model_wait_decision(drive_root, body, request=request)  # its saved choice's origin
    if decision_id.split(":", 1)[0] == "card":  # its own fields (values) and its own per-press semantics
        from ouroboros.gateway.card_decision import answer_card_decision

        return await answer_card_decision(drive_root, body, source=source)
    raw_comment = body.get("comment")
    if raw_comment is not None and not isinstance(raw_comment, str):
        return _refused("comment must be a string", 400, reason_code="comment_invalid")
    # VERBATIM: the owner's exact characters, edges included, reach the
    # projection and the frame — the only transformations are validation.
    comment = raw_comment or ""
    if len(comment) > _COMMENT_MAX:
        # VERBATIM contract: the frame signs the comment as the owner's exact
        # words — refuse instead of silently truncating them.
        return _refused(
            f"comment is {len(comment):,} characters (limit {_COMMENT_MAX:,}) — "
            "shorten it; it is delivered verbatim",
            400, reason_code="comment_too_long",
        )
    action = body.get("action", "answer")
    if action not in {"answer", "close", "dismiss"}:
        return _refused("action must be answer, close or dismiss", 400, reason_code="action_invalid")
    allowed = {"request_id", "decision_id", "option_index", "comment", "action", "reason"}
    if "surface" in body:
        from ouroboros.chat_references import TELEGRAM_SOURCE, bundled_bridge

        if source == TELEGRAM_SOURCE and bundled_bridge(drive_root):
            allowed.add("surface")
    if body.keys() - allowed:
        return _refused(
            "decision accepts only {request_id, decision_id, option_index?, comment?, action?, reason?}",
            400, reason_code="unexpected_fields",
        )
    family, task_id, quiz_id = _parse_quiz_decision_id(decision_id)
    if family not in _KNOWN_FAMILIES:
        return _refused(
            "unknown decision family (expected quiz:/routing:/interaction:)",
            400, reason_code="unknown_decision_family",
        )
    if family not in _SERVED_FAMILIES:
        # Typed, honest: the interaction family is RESERVED — #204 is served
        # by the escalation hierarchy (see the module docstring), so no direct
        # owner interaction card exists by design.
        return _refused(
            f"the {family} decision family is not served yet",
            501, reason_code="decision_family_not_served",
        )
    if attachments and (family != "quiz" or action != "answer"):
        return _refused("a file answers a quiz only", 400, reason_code="attachments_invalid")
    if action == "close":
        if family != "quiz":
            return _refused("closure addresses a quiz", 400, reason_code="close_payload_invalid")
        return await _close_quiz_decision(drive_root, task_id, quiz_id, body)
    if action == "dismiss":
        return await _dismiss_quiz_decision(drive_root, family, task_id, quiz_id, body)
    if "reason" in body:
        return _refused("reason belongs to a close action", 400, reason_code="unexpected_fields")
    raw_index = body.get("option_index")
    if raw_index is not None and (
        not isinstance(raw_index, int) or isinstance(raw_index, bool) or raw_index < 0
    ):
        return _refused(
            "option_index must be a non-negative integer",
            400, reason_code="option_index_invalid",
        )
    if raw_index is None:
        # An answer with NO option belongs to the quiz family only: the owner
        # rejected every offered option and wrote their own answer, which the
        # comment carries verbatim. The routing family has no such verb — its
        # option IS the destination — so it keeps the integer requirement.
        if family != "quiz" or not (comment.strip() or attachments):
            return _refused(
                "option_index is required (a quiz answer may instead carry a "
                "non-empty comment as the owner's own answer)",
                400, reason_code="option_index_required",
            )
    if family == "routing":
        from ouroboros.gateway.routing_decision import handle_routing_decision

        status_code, payload = await asyncio.to_thread(
            handle_routing_decision, drive_root,
            request_id=request_id, decision_id=decision_id,
            option_index=raw_index, comment=comment,
        )
        return status_code, payload
    if not task_id or not quiz_id:
        return _refused(
            "malformed quiz decision_id (expected quiz:{task_id}:{quiz_id})",
            400, reason_code="malformed_decision_id",
        )
    try:
        task_id = validate_task_id(task_id)
    except ValueError as exc:
        return _refused(str(exc), 400)
    from ouroboros.gateway.secret_decision import is_secret_question

    if is_secret_question(drive_root, task_id, quiz_id):
        # A secret request's values travel only through its protected form; neither an
        # option nor the comment may stand in for them or for the host's secret_filled.
        return _refused("this secret request is answered only through its protected form "
                        "(POST /api/decisions/secret)", 409, task_id=task_id, reason_code="secret_request_form_only")
    try:
        task, refusal = _live_root_task(task_id)
        if refusal == "not_a_root_task":
            return _refused(
                "quiz answers address root tasks only", 409,
                task_id=task_id, reason_code=refusal,
            )
        from ouroboros.owner_quiz import (
            CLOSURE_FIELDS, claim_quiz_delivery, question_delivery_projection, quiz_states, record_answered,
            record_answer_delivery_after_terminal, question_delivery_settled, record_quiz_delivery_failure,
            reconcile_terminal,
        )

        if task is None and refusal != "task_not_reading":
            # The author is gone, or settled into post-work. Normally the
            # task-done seam expires its open quizzes; a crash window or the
            # post-work window leaves one open — heal it here so the card's
            # lifecycle state is structurally truthful before the late answer
            # is recorded on it. The task-done seam announces only what IT
            # expires, so the sibling cards healed here learn it now.
            for sibling in reconcile_terminal(drive_root, task_id):
                if sibling != quiz_id:
                    _send_quiz_state(sibling, task_id, "expired_terminal")
        outcome = record_answered(
            drive_root, task_id,
            quiz_id=quiz_id, option_index=raw_index,
            request_id=request_id, comment=comment, attachments=attachments,
            # The late answer is accepted (В17a=A); with the task alive the
            # expired state would be a contradiction, so the flag is not set.
            allow_expired=task is None,
            answer_source=source, answer_surface=_answer_surface(drive_root, body, source),
        )
        block = outcome.get("block") if isinstance(outcome.get("block"), dict) else {}
        attempt_delivery = False
        pending_reason = ""
        forwarded = None
        forward_reason = ""
        late_delivery = bool(block.get("answered_after_terminal") or block.get("delivery_after_terminal"))
        if block.get("state") == "answered" and not question_delivery_settled(block):
            claimed = claim_quiz_delivery(drive_root, task_id, quiz_id, force=not _recovery)
            attempt_delivery = bool(claimed)
            block = claimed or quiz_states(drive_root, task_id).get(quiz_id) or block
            if (attempt_delivery and task is None and refusal != "task_not_reading"
                    and not (block.get("delivery") or {}).get("read_at")):
                block = record_answer_delivery_after_terminal(drive_root, task_id, quiz_id)
                late_delivery = True
        if attempt_delivery:
            destination, pending_reason = _late_answer_destination(drive_root, task_id, block)
            if destination is not None:
                try:
                    _record_quiz_answer_history(drive_root, task_id, task, block,
                                               duplicate=bool(outcome.get("duplicate") or not outcome.get("ok")))
                except Exception:
                    record_quiz_delivery_failure(drive_root, task_id, quiz_id, "quiz_history_write_failed")
                    return _answer_pending(drive_root, task_id, quiz_id, "quiz_history_write_failed",
                                           "The answer was recorded. Its history and delivery remain pending.")
            from ouroboros.question_answers import deliver_answer
            forwarded, forward_reason = await run_sync_to_completion(
                deliver_answer, drive_root, task_id, quiz_id, block, _mail_holder, _forward_late_quiz_answer)
            late_delivery = late_delivery or bool(forwarded)
            if forward_reason in {"awaiting_recipient_read", "quiz_history_write_pending"} and pending_reason:
                forward_reason = pending_reason
            if forward_reason:
                record_quiz_delivery_failure(drive_root, task_id, quiz_id, forward_reason)
            if forward_reason == "mailbox_write_failed":
                return _answer_pending(drive_root, task_id, quiz_id, forward_reason,
                                       "The answer was recorded. Delivery remains pending.")
        settled_block = quiz_states(drive_root, task_id).get(quiz_id)
        if settled_block:
            block = settled_block
        delivery_block = block.get("delivery") if isinstance(block.get("delivery"), dict) else {}
        late_delivery = late_delivery or bool(
            block.get("answered_after_terminal") or block.get("delivery_after_terminal")
        )
        if (delivery_block.get("kind") == "answer" and delivery_block.get("state") == "delivered"
                and delivery_block.get("route") == "fresh_head" and not delivery_block.get("read_at")):
            late_delivery = True
            forwarded = True
            forward_reason = ""
        elif (not forwarded and delivery_block.get("state") == "owed"
              and delivery_block.get("reason_code") == "late_answer_delivery_unknown"):
            late_delivery = late_delivery or delivery_block.get("route") == "fresh_head"
            forward_reason = "late_answer_delivery_unknown"
        delivery_view = question_delivery_projection(block)
        if not outcome.get("ok"):
            error = str(outcome.get("error") or "quiz_answer_refused")
            state = str(outcome.get("state") or "")
            if error == "quiz_not_found":
                return _refused("quiz not found", 404, task_id=task_id,
                                  reason_code=error)
            status = 409
            payload: Dict[str, Any] = {
                "ok": False, "error": error, "decision_id": decision_id,
                **({"question_delivery": delivery_view} if delivery_view else {}),
            }
            # The truthful lifecycle state settles the card client-side. Now
            # that a late answer is accepted, a refusal on a finished task
            # means the card was ALREADY answered — never a fabricated expiry.
            payload["state"] = state
            refused_block = outcome.get("block") if isinstance(outcome.get("block"), dict) else {}
            # The loser of a first-wins race settles honestly: the card learns the
            # WHOLE winner (option, words, files), never a false expiry.
            payload.update(_answer_fields(refused_block))
            payload.update({key: refused_block[key] for key in CLOSURE_FIELDS if key in refused_block})
            if error in {"option_out_of_range", "answer_empty"}:
                status = 400
            return status, payload
        _send_quiz_state(quiz_id, task_id, str(outcome.get("state") or "answered"), **_answer_fields(block),
                         question_delivery=delivery_view)
    except Exception:
        try:
            from ouroboros.owner_quiz import record_quiz_delivery_failure
            record_quiz_delivery_failure(drive_root, task_id, quiz_id, "quiz_delivery_failed")
        except Exception:
            log.warning("Could not record quiz delivery failure for %s", quiz_id, exc_info=True)
        return _answer_pending(drive_root, task_id, quiz_id, "quiz_delivery_failed",
                               "The answer was recorded. Delivery remains pending.")
    payload_ok = _quiz_decision_receipt(decision_id, outcome, block)
    if late_delivery:
        # The card outlived its author: say so, and say whether the answer
        # actually reached a chat (a machine or hidden destination has none).
        if block.get("answered_after_terminal") is True:
            payload_ok["answered_after_terminal"] = True
        if block.get("delivery_after_terminal") is True:
            payload_ok["delivery_after_terminal"] = True
        payload_ok["forwarded"] = bool(forwarded)
        if not forwarded and forward_reason:
            payload_ok["reason_code"] = forward_reason
    return 200, payload_ok


# The canonical history row that keeps a settled question's final record once its bounded projection let it go:
# row type → (its writers' sources, the state it keeps). Answers: ``_record_quiz_answer_history``; closures:
# ``events_chat_delivery._handle_send_quiz_state``. Each is found by its exact indexed ``<type>:<task>:<quiz>`` id.
_FINAL_ROWS = {"quiz_answer": ({"owner_quiz_answer"}, "answered"),
               "quiz_closed": ({"owner_question_close", "model_question_close"}, "closed")}


def _recorded_quiz(drive_root: pathlib.Path, decision_id: str) -> Optional[Dict[str, Any]]:
    """The question's record as stored now: its projection block, else, once that bounded projection let a settled
    answer or closure go (``owner_quiz._QUIZ_CAP``), the one its canonical history row keeps (``_FINAL_ROWS``), less
    the delivery that row froze when written (today's is not known there); ``{}`` when none holds it. A read that
    fails raises (unknown, never absent)."""
    from ouroboros import dialogue_store as ds
    from ouroboros.task_store import read_document

    _family, task_id, quiz_id = _parse_quiz_decision_id(decision_id)
    document = read_document(drive_root, task_id) if task_id else None
    quiz = (document.get("owner_quiz") or {}).get(quiz_id) if isinstance(document, dict) else None
    if isinstance(quiz, dict):
        return dict(quiz)
    if not task_id or not ds.present(drive_root):
        return {}
    ids = [("client_message_id", f"{kind}:{task_id}:{quiz_id}") for kind in _FINAL_ROWS]
    for entry in ds.probe_rows(drive_root, "chat", ids, limit=8):
        row, kept = entry.row, entry.row.get("quiz")
        sources, state = _FINAL_ROWS.get(row.get("type"), ((), ""))
        if (row.get("source") in sources and row.get("direction") == "system" and row.get("task_id") == task_id
                and isinstance(kept, dict) and kept.get("quiz_id") == quiz_id and kept.get("state") == state):
            return {key: value for key, value in kept.items() if key != "delivery"}
    return {}


def _absence_unproven(drive_root: pathlib.Path) -> bool:
    """Whether this indexed miss is NOT a definite absence: the canonical chat store's import coverage of the whole
    stream says the history it would hold is still coming (``dialogue_import.coverage``: a reader that found no rows
    there must say "still importing", never "no rows"). One indexed probe per state, read-only and bounded: it never
    starts the archive import, scans the store or forces a migration, so a planned-but-unimported cutover, a pending
    segment or a segment that never imported leaves the miss unproven. A root with no store and no chain files has
    nothing to import, and its absence IS proven."""
    from ouroboros import dialogue_import

    return not bool(dialogue_import.coverage(drive_root, "chat").get("imported"))


def answer_holds_uploads(drive_root: pathlib.Path, body: Dict[str, Any], refs: list) -> bool:
    """Whether this decision's recorded winner names the upload copies just stored for it (``answer_decision``
    records them with this request only), or its record cannot be read: an unknown outcome keeps them. Readable and
    naming none (another request won, this one replayed an earlier winner, or it was refused) is definite
    non-acceptance, so the caller removes them."""
    new = {ref["upload"] for ref in refs if isinstance(ref, dict) and ref.get("upload")}
    if not new:
        return False
    try:
        quiz = _recorded_quiz(drive_root, str(body.get("decision_id") or ""))
    except Exception:
        return True
    return any(isinstance(ref, dict) and ref.get("upload") in new for ref in quiz.get("attachments") or ())


def rejoin_decision(drive_root: pathlib.Path, body: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    """The bundled bridge's read of one Reply's answer (``rejoin``), before it receives the Reply's file again or
    after an unknown outcome: 200 with the receipt a replay of this request returns (``duplicate``) when this
    ``request_id`` is the question's recorded winner, also after its bounded projection let it go; 404
    ``request_not_recorded`` when it is not, its ``state`` saying which (``answered``: another request's definite
    winner, whatever its words or bytes; ``closed``: its closure; an open state: none yet; absent: no record holds
    the question), a final one with the fields a losing answer learns (the winner's option, words and files, the
    closure's reason); 503 when the record cannot be read, and also when no record holds the question while the
    canonical chat store's import is not yet complete — the miss then proves no absence (``_absence_unproven``), and
    the bridge reads it as unknown, downloading nothing. When that coverage says the import IS complete the miss is
    read once more, so a row that arrived between the two reads is the record and a miss re-read after complete
    coverage is the 404's absence; a hit pays for no extra read. The coverage read behind that miss needs the store
    too, so its own failure is the SAME typed 503, never an uncaught store failure. It writes nothing and
    delivers nothing: an owed delivery stays with recovery."""
    request_id, decision_id = str(body.get("request_id") or "").strip(), str(body.get("decision_id") or "").strip()
    if set(body) - {"request_id", "decision_id", "rejoin"} or not request_id or not _parse_quiz_decision_id(decision_id)[1]:
        return _refused("a rejoin names only its request and question", 400, reason_code="rejoin_invalid")
    try:
        quiz = _recorded_quiz(drive_root, decision_id)
        # An indexed miss is not an absence until the canonical history's coverage says it is: an unavailable
        # or still-importing store leaves the miss unproven (the same typed 503). With complete coverage the
        # miss is read once more, so a segment committing between the two reads is a hit; a miss re-read after
        # that coverage read is the absence the 404 claims. A hit pays for no extra read.
        if not quiz:
            if _absence_unproven(drive_root):
                return _refused("The answer record could not be read.", 503, reason_code="quiz_answer_unconfirmed")
            quiz = _recorded_quiz(drive_root, decision_id)
    except Exception:
        return _refused("The answer record could not be read.", 503, reason_code="quiz_answer_unconfirmed")
    if quiz.get("state") != "answered" or str(quiz.get("request_id") or "") != request_id:
        final = _quiz_decision_receipt(decision_id, {}, quiz) if quiz.get("state") in {"answered", "closed"} else {}
        return _refused("this request is not the recorded answer", 404, reason_code="request_not_recorded",
                        **({"state": quiz["state"]} if quiz.get("state") else {}),
                        **{key: value for key, value in final.items()
                           if key not in {"ok", "decision_id", "state", "duplicate"}})
    payload = _quiz_decision_receipt(decision_id, {"ok": True, "duplicate": True}, quiz)
    payload.update({key: True for key in ("answered_after_terminal", "delivery_after_terminal") if quiz.get(key) is True})
    return 200, payload


async def api_decision_answer(request: Request) -> JSONResponse:
    """POST /api/decisions — idempotent owner answer for a decision card."""
    body = await request_json_or(request, {})
    status, payload = await answer_decision(request_drive_root(request), body, request=request)
    return JSONResponse(payload, status_code=status)


async def recover_question_deliveries(drive_root: pathlib.Path, *, limit: int = 16) -> Dict[str, int]:
    """Retry addressed canonical quiz debt; no chat/result-history scan or paid work.

    The same answer ingress and publication/closure handlers own effects and receipts.
    Each claims its persisted backoff before attempting IO; restart never resets it.
    A routing or delivery ambiguity remains owed with its typed reason. A task with nothing
    due rests (``obligations.DutyRest``) on what it waits on, ``owner_quiz.question_waits``:
    its body is not read again until it is written or its persisted retry instant arrives,
    and it stays an ``owner_questions`` member for Main's current questions.
    """
    from types import SimpleNamespace

    from ouroboros import owner_quiz
    from ouroboros.obligations import DutyRest
    from ouroboros.owner_quiz import (
        claim_quiz_delivery, current_question, quiz_delivery_due, quiz_states,
        record_quiz_delivery_failure,
    )
    from ouroboros.task_results import load_task_result
    from ouroboros.utils import append_jsonl
    from supervisor.events_chat_delivery import _handle_send_quiz, _handle_send_quiz_state
    from supervisor.message_bus import get_bridge

    report = {"attempted": 0, "delivered": 0, "pending": 0}
    now = owner_quiz.utc_now_iso()
    rest = DutyRest(drive_root, "question_delivery", name="owner_questions", now=now)
    ctx = SimpleNamespace(DRIVE_ROOT=drive_root, bridge=get_bridge(), append_jsonl=append_jsonl)
    for task_id in rest.examine:
        try:
            task = load_task_result(drive_root, task_id, strict=True) or {}
        except (OSError, ValueError):
            report["pending"] += 1
            continue
        attempted = report["attempted"]
        for quiz_id, block in (task.get("owner_quiz") or {}).items():
            if report["attempted"] >= max(0, limit):
                rest.commit()
                return report
            if not quiz_delivery_due(block, now=now):
                continue
            report["attempted"] += 1
            try:
                if block.get("state") == "answered" and block.get("secret_request"):
                    from ouroboros.gateway.secret_decision import deliver_secret_outcome

                    await run_sync_to_completion(deliver_secret_outcome, drive_root, task_id, quiz_id, recovery=True)
                elif block.get("state") == "answered":
                    if not block.get("request_id"):
                        claim_quiz_delivery(drive_root, task_id, quiz_id)
                        record_quiz_delivery_failure(drive_root, task_id, quiz_id, "answer_request_unknown")
                    else:
                        await answer_decision(drive_root, {
                            "decision_id": f"quiz:{task_id}:{quiz_id}", "request_id": block["request_id"],
                            **({"option_index": block["answered_index"]} if "answered_index" in block else {}),
                            "comment": str(block.get("comment") or ""),
                        }, source=str(block.get("answer_source") or "web"), _recovery=True,
                            attachments=block.get("attachments"))  # its own recorded files: a replay, never a new set
                elif block.get("state") == "closed":
                    await run_sync_to_completion(_handle_send_quiz_state,
                        {"type": "send_quiz_state", "task_id": task_id, "quiz_id": quiz_id}, ctx)
                elif current_question(block) and (block.get("delivery") or {}).get("kind") == "publication":
                    await run_sync_to_completion(_handle_send_quiz, {
                        "type": "send_quiz", "task_id": task_id, "quiz_id": quiz_id,
                        "parent_task_id": task.get("parent_task_id", ""), "root_task_id": task.get("root_task_id", ""),
                        "chat_id": block.get("publication_chat_id", block.get("chat_id")),
                        "question": block.get("question", ""), "options": list(block.get("options") or []),
                        "stake": block.get("stake", ""), "assumption": block.get("assumption", ""),
                        "wait_for_answer": bool(block.get("wait_for_answer")),
                    }, ctx)
            except Exception:
                record_quiz_delivery_failure(drive_root, task_id, quiz_id, "quiz_delivery_failed")
                log.debug("Question recovery remains owed: %s/%s", task_id, quiz_id, exc_info=True)
            after = quiz_states(drive_root, task_id).get(quiz_id)
            report["delivered" if not after or (after.get("delivery") or {}).get("state") == "delivered" else "pending"] += 1
        if report["attempted"] == attempted:  # nothing due: rest on what it waits on
            waits, due_at = owner_quiz.question_waits(task, now=now)
            rest.wait(task_id, *waits, due_at=due_at)
    rest.commit()
    return report


__all__ = ["answer_decision", "answer_holds_uploads", "api_decision_answer", "deliver_as_owner_message",
           "recover_question_deliveries", "rejoin_decision"]
