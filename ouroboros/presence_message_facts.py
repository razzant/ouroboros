"""What a connector reports about one place message beyond its words, as host-labelled facts.

Read from ``event.message`` (CREATING_SKILLS "Places and focus transport ingress"): attachment records
that did not arrive (``attachments[]`` with a non-``staged`` status, or the slack/email dialect
``content_available: false``), descriptions of content without words (``non_text``), a forward's origin,
the replied message and its quoted part (alone, too), an album, and the address behind linked words.
``event.text``, the person's exact words, is never read for meaning or changed: the focus frame
(``presence_context``) renders these lines in their own labelled block, once, from the event record, so the
frame re-renders from that record. Staged files are the host attachment manifest's to show, never repeated
here. Every value a sender or connector controls is quoted or escaped, so one fact stays one line. A cut
names ``source``, the caller's exact reader of the whole record (``event_reader``; a row's address).
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Mapping

from ouroboros.archive_reader import display_name, size_words

STAGED = "staged"
HEADING = "What the connector reports about this message (host facts, not their words):"
# A connector may describe content without words in up to this many characters (a poll with its options, a
# checklist, a contact card); a longer description is cut with its count shown, never silently.
DESCRIPTION_CHARS = 4000
_KNOWN_STATUSES = frozenset({"too_large", "failed", "unsupported"})
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+/-]{0,126}")
# Characters that end or reorder a line where JSON quoting leaves them as they are.
_UNQUOTED_BREAKS = {ord(char): f"\\u{ord(char):04x}" for char in
                    "\x85\u2028\u2029\u061c\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first(record: Mapping[str, Any], *keys: str) -> str:
    return next((value.strip() for key in keys if isinstance(value := record.get(key), str) and value.strip()), "")


def _quoted(value: Any) -> str:
    """Free text a sender wrote, quoted so that it can neither end its line nor reorder it."""
    return json.dumps(str(value), ensure_ascii=False).translate(_UNQUOTED_BREAKS)


def _token(value: Any) -> str:
    """A short provider value (a MIME type, a kind, an id) as it is when it is one plain token, else quoted with
    every control, format and separator character escaped: a sender's value never forges another line."""
    text = str(value)
    return text if _TOKEN.fullmatch(text) else display_name(text)


def _ident(value: Any) -> str:
    return str(value) if type(value) is int else _token(value)


def attachment_outcome(record: Any) -> dict[str, Any] | None:
    """One ``attachments[]`` record in the shared vocabulary: ``status`` (staged, too_large, failed,
    unsupported or another word), ``name``, ``type``, ``size`` and ``reason``; None for a non-record. A record
    without ``status`` is staged unless it carries ``content_available: false``."""
    if not isinstance(record, Mapping):
        return None
    status = _first(record, "status").lower()
    if not status:
        unstaged = record.get("content_available") is False
        status = (("unsupported" if _first(record, "stage_error") == "unsupported_mime_type" else "failed")
                  if unstaged else STAGED)
    size = next((value for key in ("file_size", "size") if type(value := record.get(key)) is int and value > 0), None)
    return {"status": status, "name": _first(record, "file_name", "name", "filename"),
            "type": _first(record, "mime_type", "content_type", "mimetype") or _first(record, "kind"),
            "size": size, "reason": _first(record, "note", "reason", "stage_error")}


def unreceived_attachments(message: Any) -> list[dict[str, Any]]:
    """The records whose bytes did not arrive, each with its reason (email's ``attachment_stage_note`` when
    the record names none)."""
    message = _mapping(message)
    records = message.get("attachments") if isinstance(message.get("attachments"), list) else []
    fallback = _first(message, "attachment_stage_note")
    return [{**outcome, "reason": outcome["reason"] or fallback} for record in records
            if (outcome := attachment_outcome(record)) is not None and outcome["status"] != STAGED]


def non_text_items(message: Any) -> list[tuple[str, str]]:
    """``message.non_text`` as ``(kind, description)`` pairs; malformed items are not this fact."""
    items = _mapping(message).get("non_text")
    return [(_first(item, "kind") or "content", _first(item, "description")) for item in items
            if isinstance(item, Mapping) and _first(item, "description")] if isinstance(items, list) else []


def carries_content(message: Any) -> bool:
    """Whether the message carried something besides text and staged files: an event made only of these is
    never an empty one."""
    return bool(unreceived_attachments(message) or non_text_items(message))


def cut(message: Any) -> bool:
    """Whether these facts show part of ``message`` only (a description past ``DESCRIPTION_CHARS``)."""
    return any(len(description) > DESCRIPTION_CHARS for _kind, description in non_text_items(message))


def event_reader(task_id: Any, part: str = "message") -> str:
    """The exact reader of part of a focus turn's event as its task record keeps it (``metadata.presence.event``),
    for a view that shows less of it; ``""`` without a task."""
    return (f'query_data(op="json", kind="task_result", ids=[{json.dumps(str(task_id))}], '
            f'select="metadata.presence.event.{part}")') if task_id else ""


def _when(epoch: Any) -> str:
    try:
        return datetime.fromtimestamp(int(epoch), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError, OverflowError, OSError):
        return ""


def _account(value: Any) -> str:
    """A person or chat as the provider names it: name, @username and id, each only when given."""
    value = _mapping(value)
    name = " ".join(part for part in (_first(value, "first_name"), _first(value, "last_name")) if part) or _first(
        value, "title", "display_name", "name")
    username = _first(value, "username")
    handle = (f"@{username}" if re.fullmatch(r"\w{1,64}", username, re.ASCII) else f"@{display_name(username)}"
              ) if username else ""
    ident = f"id {_ident(value['id'])}" if value.get("id") not in (None, "") else ""
    words = " ".join(part for part in (_quoted(name) if name else "", handle) if part)
    return f"{words} ({ident})" if words and ident else words or ident


def _origin(origin: Any) -> str:
    if isinstance(origin, str):
        return _quoted(origin)
    origin = _mapping(origin)
    who = (_account(origin.get("sender_user")) or (_quoted(origin["sender_user_name"]) if _first(
        origin, "sender_user_name") else "") or _account(origin.get("sender_chat") or origin.get("chat"))
           or "another author")
    signed = f", signed {_quoted(origin['author_signature'])}" if _first(origin, "author_signature") else ""
    when = f", at {_when(origin.get('date'))}" if _when(origin.get("date")) else ""
    return who + signed + when


def reply_target(message: Any, *, source: str = "") -> dict[str, str] | None:
    """A provider reply target for a carried owner corpus: exact supplied words, independently named author.

    The quote is only a fragment when no full reply was supplied; neither it nor the reply becomes owner speech.
    ``source`` (the event record's exact reader) joins the address of words carried, so a view that folds them
    names where they stay whole; a platform id alone is no host address.
    """
    message = _mapping(message)
    reply = _mapping(message.get("reply_to_message"))
    ident = reply.get("message_id") if reply.get("message_id") not in (None, "") else message.get("reply_to_message_id")
    quote = _mapping(message.get("quote")).get("text")
    if not reply and ident in (None, "") and not isinstance(quote, str):
        return None
    author = _account(reply.get("from")) or _account(reply.get("sender_chat"))
    if reply.get("forward_origin"):
        author = _origin(reply["forward_origin"]) + (f" (forwarded by {author})" if author else "")
    words = next((reply[key] for key in ("text", "caption") if isinstance(reply.get(key), str)), None)
    if words is None and isinstance(quote, str):
        words, author = quote, (author or "author not recorded") + " (quoted excerpt)"
    address = f"platform message {_ident(ident)} in this room" if ident not in (None, "") else "replied message in this room (id not provided)"
    if source and words:
        address += f"; the words as carried: {source}"
    return {"address": address, "author": author, "text": words if words is not None else ""}


def _links(message: Mapping[str, Any], text: str) -> list[str]:
    """Addresses behind linked words (``text_link`` entities, UTF-16 offsets into the event's text)."""
    wire, lines = text.encode("utf-16-le"), []
    for key in ("entities", "caption_entities"):
        for entity in message.get(key) if isinstance(message.get(key), list) else []:
            entity = _mapping(entity)
            start, length, url = entity.get("offset"), entity.get("length"), _first(entity, "url")
            if entity.get("type") == "text_link" and url and type(start) is int and type(length) is int and start >= 0:
                words = wire[2 * start:2 * (start + length)].decode("utf-16-le", errors="replace")
                lines.append(f"Link behind {_quoted(words)}: {_quoted(url)}.")
    return lines


def _reply(message: Mapping[str, Any]) -> list[str]:
    reply = _mapping(message.get("reply_to_message"))
    ident = reply.get("message_id") if reply.get("message_id") not in (None, "") else message.get("reply_to_message_id")
    quote = _first(_mapping(message.get("quote")), "text")
    if not reply and ident in (None, ""):  # a quote alone is still an excerpt of someone's message
        return [f"Quotes from a message the event does not identify (author not recorded): {_quoted(quote)}."] if quote else []
    author = _account(reply.get("from")) or _account(reply.get("sender_chat"))
    words = _first(reply, "text", "caption")
    files = [name for record in reply.get("attachments") or [] if (name := _first(_mapping(record), "file_name"))]
    line = ("Replies to " + (f"message {_ident(ident)}" if ident not in (None, "") else "a message")
            + (f" by {author}" if author else "")
            + (f": {_quoted(words)}" if words else ", whose words the event does not carry")
            + (f"; it carried {', '.join(display_name(name) for name in files)}" if files else "")
            + (f"; that message was itself forwarded from {_origin(reply['forward_origin'])}"
               if reply.get("forward_origin") else "") + ".")
    return [line, *([f"Quotes from it: {_quoted(quote)}."] if quote else [])]


def connector_fact_lines(message: Any, *, text: str = "", sender: str = "", source: str = "") -> list[str]:
    """Every fact this module reads from ``message``, one sentence each, in a fixed order; a cut names ``source``."""
    message = _mapping(message)
    whole = f"; whole: {source}" if source else "; no reader of the whole was given"
    lines = [f"Not received: {display_name(item['name'] or 'an unnamed file')} "
             f"({_token(item['type']) if item['type'] else 'type unknown'}, {size_words(item['size'])}): "
             + (item["status"].replace("_", " ") if item["status"] in _KNOWN_STATUSES else _token(item["status"]))
             + (f", {_quoted(item['reason'])}" if item["reason"] else ", no reason given") + "."
             for item in unreceived_attachments(message)]
    lines += [f"Content without words, {_token(kind)}: {_quoted(description[:DESCRIPTION_CHARS])}"
              + (f" [truncated {len(description) - DESCRIPTION_CHARS} chars{whole}]"
                 if len(description) > DESCRIPTION_CHARS else "") + "." for kind, description in non_text_items(message)]
    if message.get("forward_origin"):
        lines.append(f"The channel's own post {_origin(message['forward_origin'])}, forwarded automatically into its "
                     "discussion group: the start of this comment thread."
                     if message.get("is_automatic_forward") is True else
                     f"Forwarded: these words were written by {_origin(message['forward_origin'])}, not by "
                     f"{_quoted(sender) if sender else 'the sender'}.")
    lines += _reply(message)
    if album := message.get("media_group_id"):
        position = message.get("media_group_position")
        lines.append(f"Part of album {_quoted(album)}" + (f" (item {position})" if type(position) is int else "")
                     + ": each item of an album arrives as its own event.")
    return lines + _links(message, str(text or ""))


def connector_facts(message: Any, *, text: str = "", sender: str = "", source: str = "") -> str:
    """The facts as one labelled block, or ``""`` when the message carries none."""
    lines = connector_fact_lines(message, text=text, sender=sender, source=source)
    return "\n".join([HEADING, *(f"- {line}" for line in lines)]) if lines else ""
