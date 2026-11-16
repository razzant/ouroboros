"""Kept copies: what an outgoing event carries, and whether a verified home in the app already holds it.

The outbox (lib/telegram_outbox.py) keeps an event's saved copy past a delivery that did not end confirmed only when
it is the only copy of its content. Typed facts decide, never the words: progress notes, a root's card rows, routing
receipts, subagent lifecycle cards and quiz lifecycle edits are replaceable telemetry, so a row the host types as
saved work (``terminal_receipt: retained_work``) or a completion that carries salvaged output is content even on a
card or progress row. Content has a home only when an existing reader finds now everything the bridge's renderer
sends of it: the history row at its ``row_address`` (``chat_references.resolve_history_row``) with the same words and
the same relayed quote and completion facts (``_TEXT_CONTENT``), or the same links; a photo's or video's row with its
caption, naming content-addressed chat media (``artifacts.resolve_chat_media_path``) whose bytes hash as the event's;
a document's row with its caption and file name, and its captured task file
(``gateway.files.resolve_task_file_reference``); a question's own record (``owner_quiz.quiz_states``) with the same
question, options, their details and recommendation, stake, assumption and host facts (a secret request's card says
where to enter it: ``secret_requests.transport_hint`` of the record's own request, carried unchanged). A bare
address, URL or quiz id proves nothing. Any read that fails or disagrees leaves the content without a home: kept,
never dropped on a guess; a document saved without its inline bytes then keeps its message and file reference only.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
from typing import Any, Dict, Optional, Tuple
from urllib.parse import unquote, urlsplit

from .telegram_origin import origin_label

REPLACEABLE = "replaceable"  # telemetry the next note replaces or the app keeps elsewhere: never kept past its end
ONLY = "only"  # content no verified home holds: kept after a delivery not confirmed, until the owner deletes it
# A document whose task file was its home, gone now: its message and the file's reference are kept, never the file.
REFERENCE = "reference"
HOME = "home"  # content a verified home holds: its saved copy goes at the end, once the home is checked again
_MEDIA = {"photo": "image_base64", "video": "video_base64", "document": "file_base64"}
_TELEMETRY = {"progress", "routing notice", "subagent card", "question update"}
# What the outbound renderer sends of a message (telegram_notifier._relay_text and _completion_text read the rest).
_TEXT_CONTENT = ("text", "relay", "completion_notice", "target_label", "outcome_axes")


def kind_of(topic: str, event: Dict[str, Any]) -> str:
    """The thing an event is, by its topic and typed fields."""
    if topic in {"photo", "video"}:
        return topic
    if topic in {"document", "links"}:
        return {"document": "file", "links": "links"}[topic]
    if topic in {"quiz", "quiz_state"}:
        return "question" if topic == "quiz" else "question update"
    if event.get("subagent_event"):
        return "subagent card"
    if event.get("annotation_type") == "routing_ack":
        return "routing notice"
    if event.get("terminal_receipt") == "retained_work":
        return "saved work"
    notice = event.get("completion_notice")
    if isinstance(notice, dict) and notice.get("salvage"):
        return "salvaged result"
    if event.get("is_progress") or event.get("card_row"):
        return "progress"
    if event.get("system_type") == "project_completion_summary":
        return "completion"
    return "notice" if event.get("role") == "system" else "reply"


def carries(topic: str, event: Dict[str, Any]) -> bool:
    """Whether the event holds what its handler sends (plugin.py's filters after the target chat): a medium's bytes
    or a document's file reference, a link with its label and URL, a question with its task and quiz ids. Any other
    topic is decided by its handler."""
    if topic in _MEDIA:
        ref = event.get("file_ref") if topic == "document" else None
        return bool(str(event.get(_MEDIA[topic]) or "").strip() or (isinstance(ref, dict) and ref))
    if topic == "links":
        actions = event.get("actions") if isinstance(event.get("actions"), list) else []
        return any(isinstance(action, dict) and str(action.get("label") or "").strip()
                   and str(action.get("url") or "").strip() for action in actions)
    if topic == "quiz":
        return all(str(event.get(key) or "").strip() for key in ("question", "task_id", "quiz_id"))
    return True


def classify(data_dir: Any, topic: str, event: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """``(class, home, event to save)``: a document whose captured file verifies is saved without its inline
    duplicate (its handler reads that file); every other event is saved exactly as published."""
    if kind_of(topic, event) in _TELEMETRY:
        return REPLACEABLE, {}, event
    home = home_of(data_dir, topic, event)
    if not home:
        bare = topic == "document" and not event.get("file_base64") and bool(event.get("file_ref"))
        return REFERENCE if bare else ONLY, {}, event
    if "file_ref" in home and event.get("file_base64"):
        event = {key: value for key, value in event.items() if key != "file_base64"}
    return HOME, home, event


def home_of(data_dir: Any, topic: str, event: Dict[str, Any]) -> Dict[str, Any]:
    """The verified home of the event's content, or ``{}``; read again at the delivery's end before its copy goes."""
    try:
        if topic == "quiz":
            return _quiz_home(data_dir, event)
        row = _history_row(data_dir, event.get("row_address"))
        if row is None:
            return {}
        address = {"row": str(event["row_address"])}
        if topic == "document":
            ref = _document_home(data_dir, event) if _same(row, event, ("caption", "filename")) else {}
            return {**address, "file_ref": ref} if ref else {}
        if topic in {"photo", "video"}:
            media = _media_home(data_dir, row, event.get(_MEDIA[topic])) if _same(row, event, ("caption",)) else {}
            return {**address, "media": media} if media else {}
        same = _same(row, event, ("title", "actions") if topic == "links" else _TEXT_CONTENT)
        return address if same and isinstance(event.get("title" if topic == "links" else "text"), str) else {}
    except Exception:  # an unreadable store, a malformed address or file: no proven home, so the copy stays
        return {}


def _same(row: Dict[str, Any], event: Dict[str, Any], keys) -> bool:
    """Whether the home holds every part named by ``keys`` that the event carries, equally; an empty part is
    equal to an absent one (a row leaves an empty caption out)."""
    return all(row.get(key) == event.get(key) or not (row.get(key) or event.get(key)) for key in keys)


def _history_row(data_dir: Any, address: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(address, str) or not address:
        return None
    from ouroboros import dialogue_store
    from ouroboros.chat_references import resolve_history_row

    row, _found = resolve_history_row(data_dir, address)
    return dialogue_store.whole_row(data_dir, row) if isinstance(row, dict) else None


def _media_home(data_dir: Any, row: Dict[str, Any], encoded: Any) -> Dict[str, Any]:
    """The chat media file the history row names, when its bytes are the event's own."""
    from ouroboros.artifacts import resolve_chat_media_path

    digest = hashlib.sha256(_decoded(encoded)).hexdigest()
    parts = urlsplit(str(row.get("download_url") or "")).path.split("/")
    if len(parts) != 6 or parts[:3] != ["", "api", "tasks"] or parts[4] != "artifacts":
        return {}
    task_id, name = unquote(parts[3]), unquote(parts[5])
    path = resolve_chat_media_path(data_dir, task_id, name)
    if path is None or not name.startswith(f"chat-media-{digest}.") or _file_digest(path) != digest:
        return {}
    return {"task_id": task_id, "name": name, "sha256": digest}


def _document_home(data_dir: Any, event: Dict[str, Any]) -> Dict[str, Any]:
    """The document's captured task file, when its bytes are the event's; its row is checked by the caller."""
    from ouroboros.gateway.files import resolve_task_file_reference

    ref = event.get("file_ref")
    if not isinstance(ref, dict) or not ref:
        return {}
    task_id = str(event.get("task_id") or ref.get("task_id") or "")
    resolve_task_file_reference(data_dir, task_id, ref)  # raises unless the file has its captured size and hash
    inline = event.get("file_base64")
    if inline and hashlib.sha256(_decoded(inline)).hexdigest() != ref["sha256"]:
        return {}  # the inline bytes are other content: nothing proves a home for them
    return {"task_id": task_id, "path": str(ref.get("path") or ""), "sha256": ref["sha256"]}


def _quiz_home(data_dir: Any, event: Dict[str, Any]) -> Dict[str, Any]:
    from ouroboros.owner_quiz import quiz_states
    from ouroboros.secret_requests import transport_hint

    task_id, quiz_id = str(event.get("task_id") or ""), str(event.get("quiz_id") or "")
    record = quiz_states(data_dir, task_id).get(quiz_id) if task_id and quiz_id else None
    if not isinstance(record, dict):
        return {}
    # A secret request's card is where to enter it (``LocalChatBridge.send_quiz``): only for the record's own request.
    secret = record.get("secret_request")
    if isinstance(secret, dict) or "secret_request" in event:
        question = transport_hint(secret) if isinstance(secret, dict) and event.get("secret_request") == secret else None
    else:
        question = record.get("question")
    options = [option if isinstance(option, dict) else {"label": option} for option in event.get("options") or []]
    labels = [str(option.get("label") or "") for option in options]
    details = [str(record_detail or "") for record_detail in record.get("option_details") or []]
    recommended = next((index for index, option in enumerate(options) if option.get("recommended") is True), None)
    same = (question is not None and question == event.get("question")
            and [str(label) for label in record.get("options") or []] == labels
            and (details + [""] * len(labels))[:len(labels)] == [str(option.get("detail") or "") for option in options]
            and record.get("recommended_index") == recommended
            and all(str(record.get(key) or "") == str(event.get(key) or "") for key in ("stake", "assumption", "host_facts")))
    return {"quiz": {"task_id": task_id, "quiz_id": quiz_id}} if same else {}


def _decoded(encoded: Any) -> bytes:
    try:
        return base64.b64decode(str(encoded or "").strip(), validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("the event's bytes are not valid base64") from None


def _file_digest(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def subject_of(topic: str, event: Dict[str, Any]) -> Dict[str, Any]:
    """What the Connection form names: the thing, its room and label, a file's name and type, the content's size."""
    kind = kind_of(topic, event)
    try:
        room = int(event.get("chat_id") or 0)
    except (TypeError, ValueError):
        room = 0
    subject: Dict[str, Any] = {"kind": kind, "room": room, "label": origin_label(event)}
    if topic in _MEDIA:  # the decoded size from the base64's length, without copying megabytes on the loop
        encoded = event.get(_MEDIA[topic]) if isinstance(event.get(_MEDIA[topic]), str) else ""
        subject.update(mime=str(event.get("mime") or ""), size=len(encoded) * 3 // 4 - encoded[-2:].count("="))
        if topic == "document":
            subject["name"] = str(event.get("filename") or "file")
            if isinstance(event.get("file_ref"), dict) and type(event["file_ref"].get("size")) is int:
                subject["size"] = event["file_ref"]["size"]
    else:
        subject["size"] = len(str(event.get("text") or event.get("question") or event.get("title") or "").encode())
    return subject


__all__ = ["HOME", "ONLY", "REFERENCE", "REPLACEABLE", "carries", "classify", "home_of", "kind_of", "subject_of"]
