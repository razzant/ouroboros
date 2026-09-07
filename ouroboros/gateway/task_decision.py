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
the live ``quiz_state`` broadcast. A late answer to a settled task is an
honest 409 carrying the card's true lifecycle state — the card settles
instead of inviting retries.
"""

from __future__ import annotations

import asyncio
import logging
import pathlib
from typing import Any, Dict, Optional, Tuple

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros.gateway._helpers import request_drive_root, request_json_or
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
    consults the durable projection for the honest late-answer state."""
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
        return task, ""


def _quiz_answer_frame(
    block: Dict[str, Any], option_index: Optional[int], comment: str,
) -> str:
    """Host-authored structural frame around the owner's VERBATIM choice.

    The asked/answered timestamps ride inside so the MODEL judges freshness
    itself (owner decision 30=A — no host staleness verdict). With no
    ``option_index`` the owner took none of the offered options and wrote
    their own answer — say exactly that, so the model never reads the free
    answer as a gloss on a chosen option."""
    options = block.get("options") if isinstance(block.get("options"), list) else []
    lines = [
        f"[Owner quiz answer] quiz {block.get('quiz_id')} — asked {block.get('asked_at')}, "
        f"answered {block.get('answered_at')}.",
        f"Question was: {block.get('question')}",
    ]
    if option_index is None:
        lines.append(
            "The owner rejected all offered options and answered verbatim: "
            f"{comment}"
        )
    else:
        label = str(options[option_index]) if 0 <= option_index < len(options) else ""
        lines.append(f"The owner chose option {option_index + 1}: {label}")
        if comment:
            lines.append(f"Owner comment (verbatim): {comment}")
    if str(block.get("assumption") or ""):
        lines.append(
            f"You continued under the assumption: {block.get('assumption')} — "
            "judge yourself whether work has moved past the answered fork."
        )
    return "\n".join(lines)


def _refused(message: str, status: int, **extra: Any) -> Tuple[int, Dict[str, Any]]:
    """A typed refusal as a ``(status, payload)`` pair (the gateway error shape)."""
    payload: Dict[str, Any] = {"error": message}
    payload.update(extra)
    return status, payload


async def answer_decision(drive_root: pathlib.Path, body: Any) -> Tuple[int, Dict[str, Any]]:
    """The ONE decision-answer ingress, transport-neutral: ``(status, payload)``.

    ``POST /api/decisions`` (the browser card) and the loopback Host Service
    ``POST /chat/decision`` (a reviewed transport skill relaying the owner's
    tap or reply, e.g. Telegram — #472) both call this, so every surface gets
    the same idempotent ``request_id`` write, first-answer-wins race and the
    same honest 404/409 on a late answer.
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
            # The author is gone. Normally the task-done seam already expired
            # its open quizzes; a crash window can leave one open — heal it
            # here so a late answer gets the honest expired 409 instead of a
            # 200 recorded into a mailbox nobody will ever drain.
            reconcile_terminal(drive_root, task_id)
        outcome = record_answered(
            drive_root, task_id,
            quiz_id=quiz_id, option_index=raw_index,
            request_id=request_id, comment=comment,
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
            # The truthful lifecycle state settles the card client-side: a
            # closed quiz on a SETTLED task reads as expired, an already
            # answered one as answered.
            payload["state"] = state or ("expired_terminal" if task is None else "")
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
        block = outcome.get("block") if isinstance(outcome.get("block"), dict) else {}
        if task is not None:
            from supervisor.queue import _task_drive_for_task

            from ouroboros.owner_mailbox import KIND_QUIZ_ANSWER, write_owner_message

            # EVERY accepted request appends the control — fresh, same-id
            # retry, or a duplicate after a mailbox write failure (the hurry
            # heal semantics): the msg_id is stable per quiz, so the drain
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
        try:
            from supervisor.message_bus import get_bridge

            get_bridge().send_quiz_state(
                quiz_id, task_id, str(outcome.get("state") or "answered"),
                answered_index=block.get("answered_index"),
                comment=str(block.get("comment") or ""),
            )
        except Exception:
            log.debug("quiz_state broadcast failed for %s", quiz_id, exc_info=True)
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
    return 200, payload_ok




async def api_decision_answer(request: Request) -> JSONResponse:
    """POST /api/decisions — idempotent owner answer for a decision card."""
    body = await request_json_or(request, {})
    status, payload = await answer_decision(request_drive_root(request), body)
    return JSONResponse(payload, status_code=status)


__all__ = ["answer_decision", "api_decision_answer"]
