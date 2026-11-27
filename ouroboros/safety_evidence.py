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
# Who wrote a source row, by the label the host recorded it under. ``with_author`` names rows apart from this table:
# the run's opening text, a carried copy of a row addressed here, and a mail row by its trusted host-stamped kind: the
# host's notice of a closed question, a task-opening or live living-card form's submitted values, or no retained kind.
SOURCE_AUTHORS = {
    "initial_user": "the owner: the message that opened this task",
    # One corpus can keep the origin messages of several works.
    "origin_message": "the owner: a message recorded as a work's origin",
    "owner_mailbox": "the owner: a later message to this work",
    "direct_incoming": "the owner: a later message to this work",
    "owner_quiz_answer": "the owner: the owner's answer to the card quoted in reply_to (the card's text was written "
                         "by Ouroboros)",
    "owner_press": "the owner's answer by a press of an offered option on the card in reply_to, with no words of their "
                   "own (the option's label is the card's text, written by Ouroboros)",
    "owner_card_input": "the owner submitted a living card. The original submission is unavailable",
    # The place confirms the sending account at the message's source, wherever the work later reads it.
    **dict.fromkeys(("place_confirmed", "place_owner"),
                    "a message from the owner's confirmed account in the source conversation"),
    "owner_corpus": "the owner",
    "principal_task_message": "the parent task (Ouroboros), not the owner",
    "relayed_peer_message": "a sibling task (Ouroboros), not the owner",
    "schedule_text": "text of a schedule; its supplier is named in the row",
    "live_call": "speech in a live call",
}
# A carried copy replaces a label that would name this run as the opening or addressee. The transfer neither
# establishes nor rules out an original addressee; source and reply facts remain in the row.
CARRIED_WORDS = "a record of owner input carried into this run; any host framing is not the owner's wording"
_ADDRESSED_HERE = frozenset({"initial_user", "owner_mailbox", "direct_incoming"})
# A task-opening living form with submitted values is recorded under ``owner_mailbox`` (preserving its word-source
# contract) yet is not later mail; its producer stamps this distinct host kind.
OPENING_CARD_INPUT = "opening_card_input"
# A mail row keeps its writer's trusted typed kind (``mail_kind``). Plain mail (``owner_text``) reads as later owner
# prose; a live living-card form (``card_input``) and a task-opening living form (``opening_card_input``) name the
# owner's submitted field values distinctly from the card author's text and pressed label; a closure of a question
# (``quiz_closed``, ``supervisor/events_chat_delivery.py``) reads as the host's notice even when carried. A row under
# the mail label without a trusted kind (recorded before kinds were kept, or rebuilt from a field that never kept one)
# reads neutrally, never by its id, words, or card reference alone.
CLOSED_QUESTION = ("the host's notice that the owner closed the question it names without an answer: no option was "
                   "chosen and nothing was approved. The notice's words are the host's; the reason it quotes is the "
                   "owner's own words or the host's standard sentence")
OPENING_CARD_FORM = ("the owner: the owner's submitted field values on the living card in reply_to that opened this "
                     "task (the card's text and pressed label were written by Ouroboros, not the owner)")
CARD_FORM_MAIL = ("the owner: the owner's submitted field values delivered later to this work on the living card in "
                  "reply_to (the card's text and pressed label were written by Ouroboros, not the owner)")
MAIL_UNKINDED = ("input that reached this work under the mail label without a retained kind: the owner's own words or "
                 "the host's record of an act of theirs, such as closing a question; the row does not show which")
# The same closure where a row of my human's words is stamped (``owner_words.words_stamp``) in place of ``owner``;
# ``owner_relay._stamp_parts`` reads that stamp as the same row. Never " · ", the stamp's separator.
CLOSED_STAMP = "host notice: my human closed a question without an answer, not my human's words"
# Mail the host drained from a mailbox whose writer left no kind (an older entry): the row does not show whether it is my
# human's words or a host record, so its stamp claims neither. ``_stamp_parts`` reads it as ``owner`` too.
UNKINDED_STAMP = "mail of unrecorded kind: my human's words or a host record, not shown which"
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


def checked_call_evidence(ctx, messages, binding):
    """The corpus the Safety check reads and the call's physical target. A decision subject (a dict binding:
    ``control_mind.prepare_task_cost_cap``) is its own target: the checked room and billing group, and for an own
    judgment the addressed work's instructions, never mixed into the caller's owner sources. A cited owner row
    stays in the ordinary caller corpus, marked once, also beside quoted input."""
    from dataclasses import asdict, is_dataclass

    from ouroboros import owner_words

    target = asdict(binding) if is_dataclass(binding) else binding
    cited = ((binding.get("basis") or {}).get("source") or {}).get("row_sha256") if isinstance(binding, dict) else ""
    if isinstance(getattr(ctx, "_owner_directives", None), list):
        return owner_words.safety_sources(ctx, cites=cited or ""), target
    # User messages are quoted input when the caller has no retained host source; never certify them as owner words.
    quoted = [m for m in (messages or []) if m.get("role") == "user"]
    return quoted + [row for row in (owner_words.safety_sources(ctx, cites=cited) if cited else [])
                     if row.get("cited_by_this_call")], target


def call_subject(tool_name, arguments, binding):
    """What this call's readings are kept by: its arguments, and for a cost cap also the work and decision its binding
    step bound, so a reading of one decision never answers a call bound to another."""
    return [arguments, binding] if tool_name == "set_task_cost_cap" and isinstance(binding, dict) else arguments


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


def trusted_mail_kind(value: Any, *, source: Any = "owner_mailbox") -> str:
    """A host-stamped ``mail_kind`` recognized for an ``owner_mailbox`` row, else ``""``."""
    from ouroboros.owner_mailbox import KIND_CARD_INPUT, KIND_OWNER_TEXT, KIND_QUIZ_CLOSED

    return value if source == "owner_mailbox" and isinstance(value, str) and value in {
        KIND_OWNER_TEXT, KIND_CARD_INPUT, KIND_QUIZ_CLOSED, OPENING_CARD_INPUT,
    } else ""


def closure_notice(row: Any) -> bool:
    """A mail row its writer typed as the host's notice that my human closed a question (``quiz_closed``): the one
    reading of that kind for every reader that names a row's author (Safety, a stamp, a Continue's work order)."""
    from ouroboros.owner_mailbox import KIND_QUIZ_CLOSED

    return isinstance(row, Mapping) and trusted_mail_kind(row.get("mail_kind"), source=row.get("source")) == KIND_QUIZ_CLOSED


def stamp_voice(row: Any) -> str:
    """Who a stamped row of ``words_stamp`` speaks as: ``owner``, the host's closure notice (``CLOSED_STAMP``), or, for a
    row under the mail label that proves no author (no trusted writer kind and no physical chat row of its own,
    ``row_sha256``), neither (``UNKINDED_STAMP``). The row decides, not its carrier: a forwarded or continued copy
    (``ctx``, ``owner_corpus``) of such mail stays neutral. A typed kind (a card form included), a physical row and
    every other source keep ``owner``."""
    if closure_notice(row):
        return CLOSED_STAMP
    if (row.get("source") == "owner_mailbox" and not row.get("row_sha256")
            and not trusted_mail_kind(row.get("mail_kind"), source="owner_mailbox")):
        return UNKINDED_STAMP
    return "owner"


def stamp_when(row: Any) -> str:
    """The time a stamp prints: a mailbox row's ``ts`` is the mailbox's receipt, named so for the reader; the row's own
    bytes (its identity, its quote's address) keep the time as recorded."""
    when = str(row.get("ts") or "")
    return f"received {when}" if when and row.get("carrier") == "mailbox" and not when.startswith("received ") else when


def mailbox_mail_kind(entry: Any, item: Any = None, *, source: Any = None) -> str:
    """Trusted writer kind for an ``owner_mailbox`` row produced from mailbox ``entry`` and card ``item``. A drained row
    its writer recorded without a kind routes as owner text; that default is the reader's, so its ``recorded_kind``
    (``owner_mailbox.drain_owner_entries``) stands for the writer: none."""
    from ouroboros.owner_mailbox import KIND_CARD_INPUT, KIND_OWNER_TEXT

    resolved = source if source is not None else (
        item.get("source", "owner_mailbox") if isinstance(item, Mapping) else "owner_mailbox"
    )
    if resolved != "owner_mailbox":
        return ""
    raw = entry.get("recorded_kind", entry.get("kind")) if isinstance(entry, Mapping) else None
    reply = item.get("reply_to") if isinstance(item, Mapping) else None
    if raw == KIND_OWNER_TEXT and isinstance(reply, Mapping) and reply.get("card_id"):
        return KIND_CARD_INPUT
    return trusted_mail_kind(raw)


def opening_card_mail_kind(submitted: Any) -> str:
    """``OPENING_CARD_INPUT`` when ``submitted`` is a task-opening living form with submitted values, else ``""``."""
    reply = submitted.get("reply_to") if isinstance(submitted, Mapping) else None
    if (isinstance(submitted, Mapping) and submitted.get("source") == "owner_mailbox"
            and isinstance(reply, Mapping) and reply.get("card_id")):
        return OPENING_CARD_INPUT
    return ""


def verified_chat_mail_kind(row, source, submitted, *, opening=False) -> str:
    """The existing mail kind for an already verified physical owner row, not a reconstructed corpus row."""
    from ouroboros.owner_mailbox import KIND_CARD_INPUT, KIND_OWNER_TEXT

    if source != "owner_mailbox" or row.get("_owner_corpus_rebuilt") or "corpus_answer" in row:
        return ""
    if opening_card_mail_kind(submitted):
        return OPENING_CARD_INPUT if opening else KIND_CARD_INPUT
    return KIND_OWNER_TEXT


def held_mail_kind(row: Any, root: Any, *, source: Any) -> str:
    """The writer's kind a held chat row proves for its handed ``source`` label, read as the work my human addressed
    read it on delivery (``owner_relay.inject_relay``): its card against the root it was read from, and only for an
    attested owner row. ``""`` for another author's row, a corpus rebuild, a press or any label but the mail's."""
    from ouroboros.answer_facts import card_input, row_source
    from ouroboros.owner_relay import quotable_owner_row

    source_root = row.get("_owner_source_root") or root if isinstance(row, Mapping) else None
    if source_root is None or not quotable_owner_row(row, source_root):
        return ""
    living = card_input(row, source_root)
    kind = verified_chat_mail_kind(row, living.get("source") or row_source(row, "owner_mailbox", source_root), living)
    return trusted_mail_kind(kind, source=source)


def held_owner_source(key, row, source_root, origin, *, carried, cited):
    """A retained row in the ordinary Safety corpus; an exact checked citation can supply its physical input kind."""
    from ouroboros.answer_facts import card_input, card_reference_view, row_source
    from ouroboros.owner_relay import owner_row_identity, owner_text, quotable_owner_row
    from ouroboros.owner_words import _item_time, _mapping, answered_card_reply, answered_card_source, reply_target
    from ouroboros.project_dialogue import entry_matches_source_ref

    item = card_input(row, source_root)
    carried_reply = _mapping(_mapping(row.get("corpus_answer")).get("reply_to"))
    replied = (item.get("reply_to") or card_reference_view(carried_reply)
               or answered_card_reply(*answered_card_source(row, source_root))
               or row.get("_owner_reply_target") or reply_target(source_root, row.get("reply_to")))
    source = item.get("source") or row_source(row, "owner_mailbox", source_root)
    kind = (verified_chat_mail_kind(row, source, item, opening=entry_matches_source_ref(row, [origin]))
            if cited and quotable_owner_row(row, source_root) else "")
    return {"source": source, "content": item["content"] if item else owner_text(row), "msg_id": str(key),
            "row_sha256": owner_row_identity(row)["row_sha256"], **carried, **({"mail_kind": kind} if kind else {}),
            **({"ts": when} if (when := _item_time(row)) else {}), **({"reply_to": replied} if replied else {})}


def with_author(row: Any, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """A source row with its ``author`` (``SOURCE_AUTHORS``). The run's opening text is a place event when its typed
    records ride with it (``focus_event_source``), a helper's brief when this task is a delegated child, else the
    host's or Ouroboros's text. A mail row reads by its writer's trusted retained kind: the host's closure notice
    (``CLOSED_QUESTION``, even when carried), a task-opening living form's submitted values (``OPENING_CARD_FORM``),
    a live living-card form delivery's submitted values (``CARD_FORM_MAIL``), plain mail (the table), else neutrally
    (``MAIL_UNKINDED``). A carried copy replaces only a label asserting opening or delivery here (``CARRIED_WORDS``),
    without changing the row's source or reply facts. Unknown labels are unattributed."""
    from ouroboros.owner_mailbox import KIND_CARD_INPUT, KIND_OWNER_TEXT

    row = dict(row) if isinstance(row, Mapping) else {"content": row}
    if isinstance(row.get("delivered"), Mapping):  # the room and time a delivered copy verified; never its client id or digest
        row["delivered"] = {key: row["delivered"][key] for key in ("chat_id", "ts") if key in row["delivered"]}
    source = row.get("source")
    if source == "initial_text":
        kind = "event" if "event" in row else "brief" if metadata.get("delegation_role") == "subagent" else "other"
        return {**row, "author": _OPENING_AUTHORS[kind]}
    mail_kind = row.get("mail_kind")
    if closure_notice(row):
        return {**row, "author": CLOSED_QUESTION}
    if isinstance(source, str) and source in _ADDRESSED_HERE and row.get("carried"):
        return {**row, "author": CARRIED_WORDS}
    if source == "owner_mailbox":
        if mail_kind == OPENING_CARD_INPUT:
            return {**row, "author": OPENING_CARD_FORM}
        if mail_kind == KIND_CARD_INPUT:
            return {**row, "author": CARD_FORM_MAIL}
        if mail_kind != KIND_OWNER_TEXT:
            return {**row, "author": MAIL_UNKINDED}
    return {**row, "author": SOURCE_AUTHORS.get(source, UNATTRIBUTED) if isinstance(source, str) else UNATTRIBUTED}
