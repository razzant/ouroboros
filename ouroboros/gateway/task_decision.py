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


def _live_root_task(task_id: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """Queue-lock read: the live task row, or a refusal reason.

    ``task_not_live`` is NOT a hard refusal for a quiz answer — the caller
    consults the durable projection for the honest late-answer state; nor is
    ``task_settled``, a RUNNING worker whose result already settled (its
    mailbox is no longer drained, TZ-2 D15)."""
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
    return task, ""


def _quiz_answer_frame(
    block: Dict[str, Any], option_index: Optional[int], comment: str,
) -> str:
    """Host-authored structural frame around the owner's VERBATIM choice.

    Thin alias of the ONE shared builder ``owner_quiz.quiz_answer_frame``, so
    the live mailbox control here and the late-answer delivery in the drain
    render the same words."""
    from ouroboros.owner_quiz import quiz_answer_frame

    return quiz_answer_frame(block, option_index, comment)


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


def _record_quiz_answer_history(
    drive_root: pathlib.Path, task_id: str, task: Optional[Dict[str, Any]],
    block: Dict[str, Any], *, duplicate: bool,
) -> None:
    """Keep the winning answer in canonical dialogue beyond quiz/mailbox GC.

    A retry reads the existing generation owner before healing a missing row.
    Concurrent retry duplicates retain one exact source identity; dialogue's
    identity projection, not another transaction or durable flag, deduplicates them.
    """
    from ouroboros.memory import Memory
    from supervisor.log_addressing import address_task_event
    from supervisor.message_bus import log_chat

    source_id = f"quiz_answer:{task_id}:{block['quiz_id']}"
    if duplicate:
        rows, _coverage = Memory(drive_root).read_chat_generations(predicate=lambda row: (
            row.get("type") == "quiz_answer" and row.get("task_id") == task_id
            and row.get("client_message_id") == source_id
        ))
        if rows:
            return
    if task is None:
        from ouroboros.task_results import load_task_result

        task = load_task_result(drive_root, task_id) or {}
    address = address_task_event({task_id: {"task": task}}, drive_root, {"task_id": task_id})
    index = block.get("answered_index")
    log_chat(
        "system", address.get("chat_id"), 0,
        _quiz_answer_frame(block, index if isinstance(index, int) else None, str(block.get("comment") or "")),
        ts=str(block["answered_at"]), source="owner_quiz_answer", task_id=task_id,
        client_message_id=source_id, record_type="quiz_answer", quiz=dict(block),
        message_meta=address, drive_root=drive_root, require_write=True,
    )


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

    raw = block.get("chat_id")
    if raw is None:
        from supervisor.log_addressing import address_task_event

        from ouroboros.task_results import load_task_result

        row = load_task_result(drive_root, task_id) or {}
        raw = address_task_event(
            {task_id: {"task": row}}, drive_root, {"task_id": task_id},
        ).get("chat_id")
    try:
        chat_id = int(raw)
    except (TypeError, ValueError):
        return None, "chat_unresolved"
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

    Disclosed property: in a PROJECT room that currently has exactly one live
    steerable root task, the ordinary routing delivers this message into THAT
    task's mailbox — the answer can therefore reach a different, live task in
    the same room. In Main it always starts the ordinary owner turn.
    """
    from supervisor import message_bus

    chat_id, reason = _late_answer_destination(drive_root, task_id, block)
    if chat_id is None:
        return False, reason
    from ouroboros.owner_quiz import late_answer_owner_text

    text = late_answer_owner_text(block)
    if not text:
        # An answered block always has a comment or a valid index; a block
        # that has neither cannot be spoken as the owner's words.
        return False, "answer_unreadable"
    client_message_id = f"quiz_late_answer:{task_id}:{quiz_id}"
    bridge = message_bus.get_bridge()
    try:
        row, rejoined = message_bus.accept_local_message(
            bridge, drive_root, text,
            chat_id=chat_id, user_id=1, source=str(source or "web"),
            client_message_id=client_message_id,
            # Provenance rides its OWN field; the real transport (the web card, or
            # the skill that relayed the owner's tap) is the message's source.
            task_metadata={"late_answer": {"task_id": task_id, "quiz_id": quiz_id}},
        )
    except ValueError:
        # This id was already accepted, but under other bytes: a delivery
        # accepted before the row carried only the owner's words (it carried
        # the host frame then). It WAS delivered once; a retry rejoins it
        # instead of failing forever or enqueueing a second owner turn.
        if message_bus.accepted_chat_message(drive_root, chat_id, client_message_id) is None:
            raise
        return True, ""
    if not rejoined:
        # The named ingress accepts and enqueues but does not echo; give the
        # owner the SAME user bubble their own typing produces (a rejoin
        # already echoed when it was first accepted).
        from ouroboros.projects_registry import stamp_project_thread

        echo = {
            # The accepted canonical row is the echo's text authority, exactly
            # as the owner's own typing echoes what the ingress recorded.
            "type": "chat", "role": "user", "content": str(row.get("text") or text),
            "ts": str(row.get("ts") or ""), "source": str(source or "web"), "chat_id": chat_id,
            "sender_session_id": "", "client_message_id": client_message_id,
        }
        if row.get("ingress_accepted") is True:
            echo["ingress_accepted"] = True  # the row's own fact: live matches history replay
        try:
            stamp_project_thread(message_bus.DATA_DIR, echo)
            bridge.broadcast(echo)
        except Exception:
            log.debug("Late quiz answer echo failed for %s", quiz_id, exc_info=True)
    return True, ""


async def answer_decision(drive_root: pathlib.Path, body: Any, *, source: str = "web") -> Tuple[int, Dict[str, Any]]:
    """The ONE decision-answer ingress, transport-neutral: ``(status, payload)``.

    ``POST /api/decisions`` (the browser card) and the loopback Host Service
    ``POST /chat/decision`` (a reviewed transport skill relaying the owner's
    tap or reply, e.g. Telegram — #472) both call this, so every surface gets
    the same idempotent ``request_id`` write, the same first-answer-wins race,
    and the same late-answer handling: accepted, recorded with
    ``answered_after_terminal``, and forwarded into the card's chat as the
    owner's own message (``forwarded`` says whether that happened). 404 stays
    for a card the projection no longer knows, 409 for one already answered.
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

        return await answer_model_wait_decision(drive_root, body)
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
    if any(key not in {"request_id", "decision_id", "option_index", "comment"} for key in body):
        return _refused(
            "decision accepts only {request_id, decision_id, option_index, comment?}",
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
        if family != "quiz" or not comment.strip():
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
    try:
        task, refusal = _live_root_task(task_id)
        if refusal == "not_a_root_task":
            return _refused(
                "quiz answers address root tasks only", 409,
                task_id=task_id, reason_code=refusal,
            )
        from ouroboros.owner_quiz import record_answered, reconcile_terminal

        if task is None:
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
            request_id=request_id, comment=comment,
            # The late answer is accepted (В17a=A); with the task alive the
            # expired state would be a contradiction, so the flag is not set.
            allow_expired=task is None,
        )
        block = outcome.get("block") if isinstance(outcome.get("block"), dict) else {}
        try:
            if block.get("state") == "answered":
                _record_quiz_answer_history(
                    drive_root, task_id, task, block,
                    duplicate=bool(outcome.get("duplicate") or not outcome.get("ok")),
                )
        except Exception:
            log.warning("Quiz answer history write failed for %s", quiz_id, exc_info=True)
            return _refused(
                "the answer was recorded but its dialogue history could not be written "
                "— retry to preserve and deliver it to the task",
                503, task_id=task_id, reason_code="quiz_history_write_failed",
            )
        if task is not None and block.get("state") == "answered":
            from supervisor.queue import _task_drive_for_task

            from ouroboros.owner_mailbox import KIND_QUIZ_ANSWER, write_owner_message

            # Every proven winning answer can heal its delivery, including a
            # competing new request after a partial write. The loser still
            # receives 409 below; the msg_id is stable per quiz, so the drain
            # dedupes a doubled line while a LOST control is healed by any
            # retry instead of being unrecoverable (the drain reads only the
            # mailbox, never the projection).
            # TOCTOU residual (disclosed): the task can settle between the
            # liveness read and this append — the control then waits for a
            # same-id retry attempt (reset_attempt_controls_for_retry revokes
            # only hurry/finalize kinds), and the model judges freshness from
            # the frame's stamps (30=A). No lock spans both writes on purpose.
            # The recorded block is the ONLY answer truth: a same-request_id
            # retry may legally carry a different payload (a 503 retry pressed
            # as an option after a free answer), and echoing that payload would
            # hand the task a choice the projection never recorded.
            answered_index = (int(block["answered_index"])
                              if isinstance(block.get("answered_index"), int)
                              else None)  # None: the owner's own free answer
            frame = _quiz_answer_frame(block, answered_index, str(block.get("comment") or ""))
            drive = _task_drive_for_task(task, task_id)
            if not write_owner_message(
                drive, frame, task_id,
                msg_id=f"quiz_answer:{quiz_id}", kind=KIND_QUIZ_ANSWER,
            ):
                # The projection already recorded the answer (the card is
                # truthful); the injection control failed — say so. A retry
                # of this request re-attempts the append.
                return _refused(
                    "the answer was recorded but the task control could not "
                    "be written — retry to deliver it to the task",
                    503, task_id=task_id, reason_code="mailbox_write_failed",
                )
        if not outcome.get("ok"):
            error = str(outcome.get("error") or "quiz_answer_refused")
            state = str(outcome.get("state") or "")
            if error == "quiz_not_found":
                return _refused("quiz not found", 404, task_id=task_id,
                                  reason_code=error)
            status = 409
            payload: Dict[str, Any] = {
                "ok": False, "error": error, "decision_id": decision_id,
            }
            # The truthful lifecycle state settles the card client-side. Now
            # that a late answer is accepted, a refusal on a finished task
            # means the card was ALREADY answered — never a fabricated expiry.
            payload["state"] = state
            refused_block = outcome.get("block") if isinstance(outcome.get("block"), dict) else {}
            if isinstance(refused_block.get("answered_index"), int):
                # The loser of a first-wins race settles honestly: the card
                # learns the WINNING option, never a false expiry.
                payload["answered_index"] = refused_block["answered_index"]
            if str(refused_block.get("comment") or ""):
                payload["comment"] = str(refused_block["comment"])
            if error in {"option_out_of_range", "answer_empty"}:
                status = 400
            return status, payload
        forwarded: Optional[bool] = None
        forward_reason = ""
        if task is None and block.get("answered_after_terminal"):
            # The persisted acceptance route decides delivery, never the task's
            # liveness NOW: an answer accepted LATE (no mailbox will ever drain)
            # is delivered as an ordinary owner message, and a duplicate request
            # re-enters this path on purpose — the named ingress deduplicates it,
            # so a retry after a failed delivery still delivers. An answer the
            # live task already received is never forwarded when its lost HTTP
            # response is retried after the task ended (that would be a second
            # owner turn).
            # The named ingress waits for the single host's ingress lock, which
            # an off-loop socket acceptance or skill delivery may hold across a
            # slow state read or chat scan: wait for it off the ASGI loop too. A
            # cancelled waiter still lets row → queue → echo settle first.
            try:
                forwarded, forward_reason = await run_sync_to_completion(
                    _forward_late_quiz_answer, drive_root, task_id, quiz_id, block, source=source,
                )
            except Exception:
                log.warning("Late quiz answer delivery failed for %s", quiz_id, exc_info=True)
                return _refused(
                    "the answer was recorded but could not be delivered to the chat "
                    "— retry to deliver it",
                    503, task_id=task_id, reason_code="late_answer_not_delivered",
                )
        _send_quiz_state(
            quiz_id, task_id, str(outcome.get("state") or "answered"),
            answered_index=block.get("answered_index"),
            comment=str(block.get("comment") or ""),
        )
    except Exception as exc:
        return 503, {"error": str(exc)}
    payload_ok: Dict[str, Any] = {
        "ok": True,
        "decision_id": decision_id,
        "state": str(outcome.get("state") or "answered"),
        "duplicate": bool(outcome.get("duplicate")),
    }
    recorded_index = (int(block["answered_index"])
                      if isinstance(block.get("answered_index"), int)
                      else None)
    if recorded_index is not None:
        # ABSENT, never fabricated: an answer with no option has no index,
        # and a 0 would settle the card on an option the owner refused.
        payload_ok["answered_index"] = recorded_index
    if str(block.get("comment") or ""):
        payload_ok["comment"] = str(block["comment"])
    if block.get("answered_after_terminal") is True:
        # The card outlived its author: say so, and say whether the answer
        # actually reached a chat (a machine or hidden destination has none).
        payload_ok["answered_after_terminal"] = True
        payload_ok["forwarded"] = bool(forwarded)
        if not forwarded and forward_reason:
            payload_ok["reason_code"] = forward_reason
    return 200, payload_ok




async def api_decision_answer(request: Request) -> JSONResponse:
    """POST /api/decisions — idempotent owner answer for a decision card."""
    body = await request_json_or(request, {})
    status, payload = await answer_decision(request_drive_root(request), body)
    return JSONResponse(payload, status_code=status)


__all__ = ["answer_decision", "api_decision_answer"]
