"""``card:{card_id}:{action_id}`` — the owner's press on a living card (``POST /api/decisions``).

A press on a host-drawn control (a card's ``action`` or ``form``) is the owner's act.
Each press is its own submission, idempotent on ``request_id``, and names the card
``revision`` it was drawn from: a press on a card that changed since (a kept copy, a
stale page) answers 409 ``card_stale`` instead of reaching whatever control now has
that id. An action declaring ``once: true`` keeps only its first press (a later one
answers 409 with the winner). The card store records the press first, then it is
delivered: into the authoring task's mailbox as a ``card_input`` frame while that task
lives, else its admitted Continue successor's (a paused, held or queued one keeps it until
it continues and reads it: ``awaiting_recipient_read``; no hold is released), else as the
owner's own message in the card's chat (``deliver_as_owner_message``, the late-quiz path)
— their words only (the pressed label and the form's values), as a reply to the card's
row, typed ``card_press`` so it follows its author's thread like a late answer: the head an
earlier late answer or press of that author started, else a fresh one, never another
task's reader. A failed delivery stays ``owed``; the same ``request_id`` retries it,
whatever the card became since.
"""

from __future__ import annotations

import json
import logging
import math
import pathlib
from typing import Any, Dict, Optional, Tuple

from ouroboros.gateway._helpers import run_sync_to_completion

log = logging.getLogger(__name__)

_FIELDS = frozenset({"request_id", "decision_id", "values", "revision"})
_MAX_FIELDS = 50
_MAX_VALUE_CHARS = 4000
_MAX_VALUES_BYTES = 64 * 1024


def card_input_frame(record: Dict[str, Any], label: str, values: Optional[Dict[str, Any]], *, author: str = "") -> str:
    """The host-authored frame its reader reads; the press is the owner's, the words are the host's.
    ``author`` names the task that showed the card when the reader is another task."""
    card = f"the card «{record.get('title') or ''}» that task {author} showed" if author else \
        f"your card «{record.get('title') or ''}»"
    text = f"[Card input] The owner pressed «{label}» on {card} (card {record['card_id']})."
    if values:
        text += "\nValues:\n" + json.dumps(values, ensure_ascii=False, indent=1)
    return text


def card_press_model_text(root: Any, ref: Any, owner_words: str) -> str:
    """What a head reads for a press on a finished author's card: the frame rebuilt from the stored press;
    an unreadable one keeps the owner's words with that author named."""
    from ouroboros.card_store import find_submission, late_press_ref, load_card

    press = late_press_ref(ref)
    if press is None:
        return owner_words
    record = load_card(root, press["card_id"]) or {}
    row = find_submission(record, press["request_id"]) or {}
    if not row:
        return f"[Card input] On a card of task {press['task_id']} that can no longer be read, the owner pressed:\n{owner_words}"
    return card_input_frame({**record, "title": row.get("title", record.get("title"))},
                            str(row.get("label") or row.get("action_id") or ""), row.get("values"), author=press["task_id"])


def owner_press_text(label: str, values: Optional[Dict[str, Any]]) -> str:
    """What the owner's own message says for a late press: the pressed label, then each value."""
    lines = [label]
    for key, value in (values or {}).items():
        shown = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        lines.append(f"{key}: {shown}")
    return "\n".join(lines)


def _checked_values(raw: Any, action: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    """``(values, "")`` or ``(None, reason)``: form fields only, JSON scalars, bounded."""
    if raw is None:
        return None, ""
    if action["kind"] != "form":
        return None, "values belong to a form action"
    if not isinstance(raw, dict) or len(raw) > _MAX_FIELDS:
        return None, f"values must be an object of at most {_MAX_FIELDS} fields"
    unknown = sorted(set(raw) - set(action.get("fields") or ()))
    if unknown:
        return None, f"values name fields the form does not have: {', '.join(map(str, unknown))}"
    for key, value in raw.items():
        if isinstance(value, str):
            if len(value) > _MAX_VALUE_CHARS:
                return None, f"value of {key!r} is longer than {_MAX_VALUE_CHARS} characters"
        elif not (value is None or isinstance(value, bool)
                  or (isinstance(value, (int, float)) and math.isfinite(value))):
            return None, f"value of {key!r} must be text, a finite number, a boolean or null"
    if len(json.dumps(raw, ensure_ascii=False).encode("utf-8")) > _MAX_VALUES_BYTES:
        return None, f"values exceed {_MAX_VALUES_BYTES} bytes"
    return dict(raw), ""


def _deliver(drive_root: pathlib.Path, record: Dict[str, Any], submission: Dict[str, Any],
             source: str) -> Tuple[str, str]:
    """``(delivery, reason)`` for one recorded press: ``delivered`` into its author's, else its admitted
    Continue successor's, mailbox (``awaiting_recipient_read`` while that one reads no mail now),
    ``forwarded`` or ``owed``; a press the owner's ingress already accepted converges there."""
    from ouroboros.contracts.chat_id_policy import HIDDEN_CHAT_ID, is_a2a_chat_id
    from ouroboros.gateway.task_decision import _mail_holder, deliver_as_owner_message
    from ouroboros.owner_mailbox import KIND_CARD_INPUT, write_owner_message
    from ouroboros.question_answers import admitted_successor, reading_recipient
    from supervisor import message_bus, queue as q
    from supervisor.workers import get_direct_chat_agent

    card_id, request_id = str(record["card_id"]), str(submission["request_id"])
    label = str(submission.get("label") or submission["action_id"])
    card = {**record, "title": submission.get("title", record.get("title"))}
    # A press goes to the author of the card it was pressed on, also when it is retried after a replace.
    task_id = str(submission.get("task_id", record.get("task_id")) or "")
    chat_id = record.get("publication_chat_id", record.get("chat_id"))
    client_id = f"card_input:{card_id}:{request_id}"
    # A press the owner's ingress already accepted keeps that one effect: its retry converges there.
    accepted = type(chat_id) is int and message_bus.accepted_chat_message(drive_root, chat_id, client_id) is not None
    # The author, else its admitted Continue successor: whichever lives holds the press, as for an answer.
    recipient, holder = reading_recipient(drive_root, task_id, _mail_holder) if task_id and not accepted else ("", None)
    attempted = set()
    while holder is not None and recipient not in attempted:
        attempted.add(recipient)
        agent = get_direct_chat_agent(recipient)
        # Room intake's admission owner: no mailbox write after the recipient's final drain closed.
        with getattr(agent, "_owner_message_admission_lock", None) or q._queue_lock:
            task, reason = _mail_holder(recipient)
            if task is not None:
                drive = pathlib.Path(drive_root) if agent is not None else q._task_drive_for_task(task, recipient)
                text = card_input_frame(card, label, submission.get("values"), author="" if recipient == task_id else task_id)
                # Paused, held or queued: the press waits for it like an answer does, and its read settles the wait.
                waits = reason == "task_not_reading"
                if write_owner_message(drive, text, recipient, msg_id=client_id, kind=KIND_CARD_INPUT, card_delivery={
                        "card_id": card_id, "request_id": request_id, "canonical_root": str(drive_root)} if waits else None):
                    return "delivered", "awaiting_recipient_read" if waits else ""
                return "owed", "mailbox_write_failed"
        # It ended before admission: follow its admitted Continue under that one's own lock, as an answer does.
        recipient, holder = reading_recipient(drive_root, admitted_successor(drive_root, recipient), _mail_holder)
    if type(chat_id) is not int or is_a2a_chat_id(chat_id) or chat_id == HIDDEN_CHAT_ID:
        return "owed", "card_chat_unanswerable"
    # The owner's bubble holds their words; replying to the card's row tells the model which card they answered.
    address = str(record.get("publication_address") or "")
    # No living holder: the typed press follows its author's thread in routing, the way a late answer does.
    metadata = {**({"reply_to": address} if address else {}),
                **({"card_press": {"task_id": task_id, "card_id": card_id, "request_id": request_id}} if task_id else {})}
    forwarded, reason = deliver_as_owner_message(drive_root, chat_id, owner_press_text(label, submission.get("values")),
                                                 client_id, source=source, task_metadata=metadata or None)
    return ("forwarded", "") if forwarded else ("owed", reason or "late_answer_delivery_unknown")


def _refused(message: str, status: int, **extra: Any) -> Tuple[int, Dict[str, Any]]:
    return status, {"error": message, **extra}


def _stale(decision_id: str, card_id: str, current: int) -> Tuple[int, Dict[str, Any]]:
    """The card changed after it was drawn: the press is refused, never moved to today's control."""
    return 409, {"ok": False, "error": "card_stale", "reason_code": "card_stale", "decision_id": decision_id,
                 "card_id": card_id, "revision": current}


async def answer_card_decision(drive_root: pathlib.Path, body: Dict[str, Any], *, source: str = "web") -> Tuple[int, Dict[str, Any]]:
    """Record and deliver one press; ``(status, DecisionResponse)``. The caller checked ``request_id``."""
    import asyncio

    from ouroboros.card_store import (CardStoreError, find_submission, load_card, mark_delivery, press_wait,
                                      record_submission)
    from ouroboros.contracts.plugin_api import ExtensionRegistrationError
    from ouroboros.extension_ui_validation import validate_card_components

    if any(key not in _FIELDS for key in body):
        return _refused("a card decision accepts only {request_id, decision_id, revision, values?}", 400,
                        reason_code="unexpected_fields")
    request_id, decision_id = str(body["request_id"]).strip(), str(body["decision_id"]).strip()
    parts = decision_id.split(":", 2)
    if len(parts) != 3 or not parts[1] or not parts[2]:
        return _refused("malformed card decision_id (expected card:{card_id}:{action_id})", 400,
                        reason_code="malformed_decision_id")
    revision = body.get("revision")
    if type(revision) is not int or revision < 1:
        return _refused("a card press names the revision of the card it was drawn from (a positive integer)", 400,
                        reason_code="revision_required")
    _family, card_id, action_id = parts
    record = load_card(drive_root, card_id)
    if record is None:
        return _refused("card not found", 404, reason_code="card_not_found")
    action: Dict[str, Any] = {"label": action_id}
    values: Optional[Dict[str, Any]] = None
    if find_submission(record, request_id) is None:  # a retry of a recorded press skips to its delivery
        if int(record.get("revision") or 1) != revision:
            return _stale(decision_id, card_id, int(record.get("revision") or 1))
        try:
            actions = validate_card_components(record.get("components"))["actions"] if record.get("kind") == "components" else {}
        except ExtensionRegistrationError:
            actions = {}
        if action_id not in actions:
            return _refused(f"card {card_id} has no action {action_id!r}", 404, reason_code="action_not_found")
        action = actions[action_id]
        values, problem = _checked_values(body.get("values"), action)
        if problem:
            return _refused(problem, 400, reason_code="values_invalid")
    try:
        outcome = await asyncio.to_thread(record_submission, drive_root, card_id, request_id=request_id,
                                          action_id=action_id, label=action["label"], values=values,
                                          once=bool(action.get("once")), revision=revision)
    except CardStoreError as exc:
        return _refused(str(exc), 404, reason_code=exc.code)
    except (OSError, TimeoutError) as exc:
        return _refused(f"the press could not be recorded: {exc}", 503, reason_code="card_write_failed")
    if "stale" in outcome:
        return _stale(decision_id, card_id, int(outcome["stale"]))
    if "winner" in outcome:
        winner = outcome["winner"]
        return 409, {"ok": False, "error": "card_action_once", "decision_id": decision_id, "reason_code": "card_action_once",
                     "winner": {"request_id": str(winner.get("request_id") or ""), "at": str(winner.get("at") or ""),
                                "delivery": str(winner.get("delivered") or "owed")}}
    submission, record = outcome["submission"], outcome["record"]
    delivery, reason = str(submission.get("delivered") or "owed"), press_wait(submission)
    if delivery == "owed":
        try:
            delivery, reason = await run_sync_to_completion(_deliver, drive_root, record, submission, source)
        except Exception:
            log.warning("Card press delivery remains owed for %s", card_id, exc_info=True)
            delivery, reason = "owed", "card_delivery_failed"
        if delivery != "owed":
            record = await asyncio.to_thread(mark_delivery, drive_root, card_id, request_id, delivery, reason) or record
            if delivery == "delivered":  # the answer says what the record says: a read that came first ended the wait
                reason = press_wait(find_submission(record, request_id) or {})
    try:
        from supervisor.card_delivery import send_card_state
        from supervisor.message_bus import get_bridge

        send_card_state(get_bridge(), record)
    except Exception:
        log.debug("card_state broadcast failed for %s", card_id, exc_info=True)
    payload = {"ok": True, "decision_id": decision_id, "request_id": request_id, "duplicate": bool(outcome["duplicate"]),
               "delivery": delivery, "card_id": card_id, "revision": int(record.get("revision") or 1)}
    if delivery == "owed":
        return 202, {**payload, "reason_code": reason}
    return 200, {**payload, "reason_code": reason} if reason else payload


__all__ = ["answer_card_decision", "card_input_frame", "card_press_model_text", "owner_press_text"]
