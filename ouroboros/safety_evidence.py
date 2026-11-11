"""The Supervisor's evidence diet: what a source row costs the bundle, never what it says.

A tight bundle is answered first by removing what carries no meaning of its own (BIBLE P1): an image the checker
does not assess becomes facts (digest, media type, size, caption), a connector event loses its raw provider payload
and second copies of its text, and a thread or history becomes one who/when/what line per message. A place event's
frame is replaced by its typed records only when they re-render it byte for byte (``focus_frame_sources``), the
producer's attachment manifest and image captions included. Owner words, rules and notes are never shortened; the
per-place work lines (``work_here``, the previous turn) stay whole. Every row names who wrote it (``with_author``).
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
from typing import Any, Mapping

_ATTACHMENTS = ("[ATTACHMENTS]\n", "\n[END_ATTACHMENTS]")  # the presence producer's manifest block
# Who wrote a source row, by the label the host recorded it under: the one table every row's ``author`` comes from.
SOURCE_AUTHORS = {
    "initial_user": "the owner: the message that opened this task",
    "origin_message": "the owner: the message that opened the work this task belongs to or continues (it may be "
                      "about an earlier stage)",
    "owner_mailbox": "the owner: a later message to this work",
    "direct_incoming": "the owner: a later message to this work",
    "owner_quiz_answer": "the owner: the owner's answer to the card quoted in reply_to (the card's text was written "
                         "by Ouroboros)",
    "place_confirmed": "the owner: the owner's own message in this chat, from a confirmed owner account",
    "place_owner": "the owner: the owner's own message in this chat, from a confirmed owner account",
    "owner_corpus": "the owner",
    "principal_task_message": "the parent task (Ouroboros), not the owner",
    "relayed_peer_message": "a sibling task (Ouroboros), not the owner",
    "schedule_text": "text of a schedule; its supplier is named in the row",
    "live_call": "speech in a live call",
}
_OPENING_AUTHORS = {  # ``initial_text``: the run's first turn, by what the host knows of it
    "event": "a chat event that woke this turn: event.actor wrote message_text; the other keys are host facts",
    "brief": "this task's brief: written by its parent task (Ouroboros) with blocks added by the host; not the "
             "owner's words",
    "other": "host or Ouroboros text that opened this run; not the owner's words",
}
UNATTRIBUTED = "unattributed; not attested owner words"


def _image(part: Any) -> bool:
    return isinstance(part, Mapping) and part.get("type") in ("image_url", "image")


def image_facts(part: Mapping[str, Any], *, shown: str = "") -> dict[str, Any]:
    """An image as facts, its pixels not assessed; a caption the row's text already shows is not repeated."""
    url = str((part.get("image_url") or {}).get("url") or "") if isinstance(part.get("image_url"), Mapping) else ""
    source = part.get("source") if isinstance(part.get("source"), Mapping) else {}
    if source.get("type") == "base64":
        url = f"data:{source.get('media_type') or ''};base64,{source.get('data') or ''}"
    head, _, data = url.partition(",")
    facts: dict[str, Any] = {"type": "image", "pixels": "not assessed"}
    try:
        if head.startswith("data:") and head.endswith(";base64"):
            raw = base64.b64decode(data, validate=True)
            facts.update(sha256=hashlib.sha256(raw).hexdigest(), media_type=head[5:-len(";base64")], bytes=len(raw))
        elif url:
            facts["url"] = url
    except (binascii.Error, ValueError):
        facts["unreadable"] = "the data URL is not valid base64"
    caption = str(part.get("_caption") or "")
    return {**facts, **({"caption": caption} if caption and caption not in shown else {})}


def without_pixels(row: Any) -> Any:
    """A source row whose content carries images: each image becomes its facts; every other part stays as it is."""
    content = row.get("content") if isinstance(row, Mapping) else None
    if not isinstance(content, list) or not any(_image(part) for part in content):
        return row
    shown = "\n".join(str(part.get("text") or "") for part in content
                      if isinstance(part, Mapping) and part.get("type") == "text")
    return {**row, "content": [image_facts(part, shown=shown) if _image(part) else part for part in content]}


def _line(row: Mapping[str, Any], message_text: str, waking: set[str]) -> str:
    """One message as who, when and what; the waking message points at ``message_text`` instead of repeating it."""
    from ouroboros.presence_context import _thread_author, _thread_time

    user = row.get("user")
    identity, name = ((str(user.get("login") or user.get("id") or ""), str(user.get("name") or ""))
                      if isinstance(user, Mapping) else _thread_author(row))
    who = f"{name} ({identity})" if name and identity else name or identity or "unknown author"
    when = _thread_time(row)
    when = str(row.get("created_at") or when) if when == "an unrecorded time" else when
    text = next((row[key] for key in ("text", "body") if isinstance(row.get(key), str)), "")
    own = {str(row[key]) for key in ("message_id", "id", "ts") if row.get(key)}
    said = ("[this message: its text is message_text]" if own & waking or (text.strip() and text.strip() == message_text)
            else json.dumps(text, ensure_ascii=False))
    return f"{who} at {when}: {said}"


def lean_event(event: Mapping[str, Any], message_text: str) -> dict[str, Any]:
    """The connector's event without its raw provider payload or a second copy of the message text, a thread
    snapshot or history as lines (its other facts, an author directory included, kept)."""
    message = event.get("message") if isinstance(event.get("message"), Mapping) else {}
    raw = message.get("raw") if isinstance(message.get("raw"), Mapping) else {}
    waking = {str(value) for value in (message.get("message_id"), message.get("id"), message.get("ts"), raw.get("id"))
              if value}
    conversation = dict(event.get("conversation") or {}) if isinstance(event.get("conversation"), Mapping) else {}
    for key in ("thread_history", "history"):
        history = conversation.get(key)
        data = history.get("data") if isinstance(history, Mapping) and isinstance(history.get("data"), Mapping) else {}
        rows = data.get("messages") if data else history.get("rows") if isinstance(history, Mapping) else None
        if isinstance(rows, list):
            conversation[key] = {**{k: v for k, v in history.items() if k not in ("data", "rows")},
                                 **({"complete": data["complete"]} if "complete" in data else {}),
                                 "lines": [_line(row, message_text.strip(), waking) for row in rows if isinstance(row, Mapping)]}
    return {**event, "conversation": conversation,
            "message": {key: value for key, value in message.items()
                        if key != "raw" and not (key == "text" and value == message_text)}}


def focus_event_source(row: Any, metadata: Mapping[str, Any], notes: list) -> Any:
    """A place event's host frame as its typed records when they reproduce that row exactly; any other row keeps its
    words, its images as facts.

    Each row is compared on its own, so another turn's frame or a Continue order stays whole. A frame with the
    producer's attachment manifest and image blocks re-renders with them (the manifest, verified by that render, and
    the images as facts ride with the records). Notes and circle words as they stood at the event are kept only
    where they differ from those read now (``notes``: the bundle's entries, ``place_guard.authored_notes``)."""
    from ouroboros.presence_context import focus_frame_sources

    typed, images, manifest = None, [], ""
    content = row.get("content") if isinstance(row, dict) and row.get("source") == "initial_text" else None
    head = content[0] if isinstance(content, list) and content else None
    frame = content if isinstance(content, str) else (
        str(head.get("text") or "") if isinstance(head, Mapping) and head.get("type") == "text" else None)
    if frame is not None and (isinstance(content, str) or all(_image(part) for part in content[1:])):
        images = [] if isinstance(content, str) else list(content[1:])
        start = frame.find(_ATTACHMENTS[0])
        end = frame.find(_ATTACHMENTS[1], start + 1) if start >= 0 else -1
        manifest = frame[start:end + len(_ATTACHMENTS[1])] if end > start else ""
        observed = str((metadata.get("presence") or {}).get("observed_text") or "").strip()
        text = f"{observed}\n\n{manifest}".strip() if manifest else None
        rendered_from = text if not images else [{"type": "text", "text": text if text is not None else observed},
                                                 *[block for part in images for block in (
                                                     {"type": "text", "text": str(part.get("_caption") or "")}, part)]]
        typed = focus_frame_sources(metadata, frame, content=rendered_from)
    if typed is None:
        return without_pixels(row)
    at_event, place_id = typed.pop("notes_at_event", None) or [], (metadata.get("focus") or {}).get("place_id")
    circle_then, now = typed.pop("circle_at_event", None), [n for n in notes if n["place_id"] == place_id]

    def versions(rows):
        return sorted((str(n.get("text")), str(n.get("author")), str(n.get("at") or "")) for n in rows)
    if versions(at_event) != versions(n for n in now if "circle" not in n):
        typed["notes_at_event"] = at_event
    circle_now = next(({"name": n["circle"], "text": n["text"], "text_author": n["author"], "placed_by": n["placed_by"]}
                       for n in now if "circle" in n), None)
    if (circle_then or {}).get("text") and circle_now != {key: circle_then.get(key) for key in (
            "name", "text", "text_author", "placed_by")}:
        typed["circle_at_event"] = circle_then
    typed["event"] = lean_event(typed["event"], str(typed.get("message_text") or ""))
    return {**{key: value for key, value in row.items() if key != "content"},
            "host_frame": "prompts/FOCUS.md showed the mind these records and its allowance", **typed,
            **({"attachments": manifest} if manifest else {}),
            **({"images": [image_facts(part) for part in images]} if images else {})}


def with_author(row: Any, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """A source row with its ``author`` (``SOURCE_AUTHORS``). The run's opening text is a place event when its typed
    records ride with it (``focus_event_source``), a helper's brief when this task is a delegated child, else the
    host's or Ouroboros's text; a row under any other label, or none, is unattributed."""
    row = dict(row) if isinstance(row, Mapping) else {"content": row}
    source = row.get("source")
    if source == "initial_text":
        kind = "event" if "event" in row else "brief" if metadata.get("delegation_role") == "subagent" else "other"
        return {**row, "author": _OPENING_AUTHORS[kind]}
    return {**row, "author": SOURCE_AUTHORS.get(source, UNATTRIBUTED) if isinstance(source, str) else UNATTRIBUTED}
