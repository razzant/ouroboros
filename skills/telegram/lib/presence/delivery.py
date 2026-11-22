"""Exact provider-send observations for the Presence outbox."""
from __future__ import annotations

from pathlib import Path

from .events import _reaction_values


def names_message(answer):
    """Telegram's answer to a send names the message it created: a positive integer ``message_id``.

    The one test of a confirmed message, for the outbox's decisions, its receipts and its reports; nothing else in the
    answer is checked. An answer that fails it stays in the row as stored, and confirms nothing.
    """
    message_id = answer.get("message_id") if isinstance(answer, dict) else None
    return type(message_id) is int and message_id > 0


def stored_answers(payload):
    """The answers a send row stored (text chunks in ``_sent_messages``, a file's or an album's in ``_sent_media``),
    as (confirmed messages, other answers kept as stored); ``None`` when the row stored no answer."""
    for key in ("_sent_messages", "_sent_media"):
        if key in payload:
            stored = payload[key] if isinstance(payload[key], list) else [payload[key]]
            return (
                [answer for answer in stored if names_message(answer)],
                [answer for answer in stored if not names_message(answer)],
            )
    return None


def tool_origin(ctx):
    origin = {"kind": "tool"}
    if getattr(ctx, "task_id", None):
        origin["task_id"] = str(ctx.task_id)
    metadata = getattr(ctx, "task_metadata", None) or {}
    event = (metadata.get("presence") or {}).get("event") or {}
    if event.get("source_event_id"):
        origin["source_event_id"] = str(event["source_event_id"])
    return origin


REACTION_HISTORY_UNAVAILABLE = "Held: Ouroboros records this reaction after an update."
REACTION_REPORT_UNREADABLE = "Held: Ouroboros will not record this reaction."


def _reaction_fact(values, target, state):
    """A set says what was asked for. Telegram does not say what it replaced, so nothing here does."""
    return {
        "schema_version": 1,
        "operation": "set",
        "new": values,
        "target": target,
        **({"effect": "applied"} if state in {"delivered", "accepted"} else {}),
    }


def _reaction_receipt(payload, state):
    parameters = payload.get("parameters") or {}
    return _reaction_fact(
        _reaction_values(parameters.get("reaction")),
        payload.get("_reaction_target")
        or {"message_id": str(parameters.get("message_id") or ""), "resolution": "unresolved"},
        state,
    )


def _album(payload):
    """Telegram moves a reaction on an album member to the album's first undeleted one and answers only True."""
    group = payload.get("_reaction_album")
    return {"album": {"media_group_id": group, "effective_message_id": "unknown"}} if group else {}


def reaction_receipt(payload):
    """The local provider receipt of a set Telegram confirmed."""
    return {
        "kind": "operation",
        "method": payload["method"],
        "result": payload["_operation_result"],
        "effect": "applied",
        **_album(payload),
    }


def is_reaction_report(report):
    """A frozen report of a reaction, in today's typed shape or an earlier version's list."""
    message = report.get("message")
    return isinstance(message, dict) and (
        message.get("method") == "setMessageReaction"
        or (message.get("kind") == "operation" and "reaction" in message)
    )


def typed_reaction_report(report):
    """What a Host that reads typed actions may receive, from this report's own retained facts."""
    message = report["message"]
    reaction, target = message.get("reaction"), message.get("target_message_id")
    if isinstance(reaction, dict):
        return report if reaction.get("schema_version") == 1 else None
    if not isinstance(reaction, list) or type(target) not in (int, str) or not str(target).strip():
        return None
    values = _reaction_values(reaction)
    if len(values) != len(reaction):
        return None
    fact = _reaction_fact(values, {"message_id": str(target), "resolution": "unresolved"}, report.get("state"))
    return {**report, "message": {**message, "reaction": fact}}


def delivery_report(delivery_id, payload, *, part_id, state, receipt=None, text="", fmt="", error=""):
    reporting = payload.get("_reporting") or {}
    if reporting.get("version") != 1:
        return None
    reaction = payload.get("kind") == "operation" and payload.get("method") == "setMessageReaction"
    if reaction and reporting.get("explicit_completion") != 1:
        return None
    receipt = receipt or {}
    wire = receipt.get("_delivery") or {}
    message = {"kind": payload.get("kind", "message")}
    for key in (
        "message_id",
        "date",
        "entities",
        "caption_entities",
        "photo",
        "document",
        "video",
        "audio",
        "voice",
        "animation",
        "media_group_id",
    ):
        if key in receipt:
            message[key] = receipt[key]
    if payload.get("file_path"):
        message["file_name"] = Path(payload["file_path"]).name
    if payload.get("reply_to_message_id"):
        message["reply_to_message_id"] = payload["reply_to_message_id"]
    if payload.get("kind") == "operation":
        message["method"] = str(payload.get("method") or "")
        message["target_message_id"] = (payload.get("parameters") or {}).get("message_id")
        if payload.get("method") == "editMessageText":
            message["replacement_text"] = str(payload.get("text") or "")
        if reaction:
            message["reaction"] = _reaction_receipt(payload, state)
            message.update(_album(payload))
        if payload.get("original_delivery_id"):
            message["original_delivery_id"] = str(payload["original_delivery_id"])
    if error:
        message.update(error=error, confirmed_parts=len((stored_answers(payload) or ([], []))[0]))
    return {
        "schema_version": 1,
        "delivery_id": delivery_id,
        "part_id": str(part_id),
        "state": state,
        "provider": "telegram",
        "account_id": str(reporting.get("account_id") or ""),
        "conversation_id": str(payload["chat_id"]),
        "thread_id": str(payload.get("topic_id") or ""),
        "text": "" if payload.get("kind") == "operation" else wire.get("text", text),
        "format": "" if payload.get("kind") == "operation" else wire.get("format", fmt),
        "message": message,
        "origin": reporting.get("origin") or {"kind": "automatic"},
    }
