"""History-only Presence ingress: a connector's earlier messages enter their room as history.

``POST /presence/history`` (``presence_history_import_version: 1``) carries a bounded batch
of one conversation's earlier provider messages (CREATING_SKILLS "Importing earlier
history"). Each record lands once, under the conversation's intake lock, as the canonical
arrival row its live capture would write (``presence_runner._log_dialogue``): the same
source event id, turn id, event identity and transport facts, with ``ts`` the provider's
send time and an immutable ``historical_import`` marker in the same append (import id,
authorship, Host receipt time, connector). Like every new arrival, live ones included, it
also keeps its send time and ``message.source_order`` in an immutable ``source_order`` block,
by which a room holding imports reads (``place_arrivals.READING_TIME``). Authorship is a
provider fact: a participant, or the connected platform account itself, never the owner and
never Ouroboros.

Nothing else happens. No task, attention decision, wake, unread revision, mailbox,
current-author mail, outbox or reply follows a record; place state does not matter. An
event already captured live keeps its custody (``custody: live``); an identical record
answers ``duplicate``; different facts under the same source event are an identity
conflict. A later live replay of an imported revision is heard silent before attention
(``place_arrivals.heard_replay``); a later edit is a new revision and a new live event.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)

HISTORY_IMPORT_VERSION = 1
# ``event.message.source_order`` on both doors, live and history: the provider's own order of a conversation's
# messages (Telegram: its message id). New arrivals keep it in their immutable ``source_order`` block.
SOURCE_ORDER_VERSION = 1
SOURCE_ORDER_LIMIT = 1 << 53  # an order every client reads exactly
MAX_RECORDS = 100
AUTHORSHIPS = ("participant", "platform_account")
_EVENT_FIELDS = frozenset({"source_event_id", "provider", "account_id", "conversation_id", "thread_id",
                           "conversation_key", "actor", "conversation", "message", "text"})
_RECORD_FIELDS = frozenset({"event", "authorship", "source_timestamp", "staged_files"})
# ``event.message`` keys that say what the provider message is; its send time is the record's
# ``source_timestamp`` (``message.sent_at`` must agree). Every other actor, conversation and
# message key is display enrichment, and a connector's local staging outcome of a medium is its
# own observation: the first stored values stand and a retry may differ there.
SOURCE_MESSAGE_FACTS = ("message_id", "event_kind", "revision", "edited_at", "reply_to_message_id")
_MEDIA_OBSERVATIONS = frozenset({"status", "note", "reason"})


class PresenceHistoryInvalid(ValueError):
    """A malformed request, refused before any record is written; ``field`` names the place."""

    code = "presence_history_invalid"

    def __init__(self, field: str, message: str = "") -> None:
        self.field = field
        super().__init__(message or f"invalid presence history field: {field}")


@dataclass(frozen=True)
class HistoryRecord:
    event: Any  # presence_runner.PresenceTurnEvent with the Host-derived conversation key
    authorship: str
    ts: str  # the provider send time, normalized to the live rows' UTC ISO form


@dataclass(frozen=True)
class HistoryRequest:
    import_id: str
    binding_id: str
    records: tuple[HistoryRecord, ...]


def _instant(value: Any) -> datetime | None:
    """A timezone-qualified ISO time as UTC; None when absent or unreadable (unknown, never invented),
    also when moving it to UTC leaves the calendar (year 1 east of UTC, year 9999 west of it)."""
    if not isinstance(value, str) or not value or len(value) > 64:
        return None
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
        return parsed.astimezone(timezone.utc) if parsed.utcoffset() is not None else None
    except (ValueError, OverflowError):
        return None


def source_order(message: Any) -> int | None:
    """``message.source_order`` when the connector sent one; anything but an integer from 0 to 2^53-1 is a
    malformed event (``ValueError``), never an order guessed or dropped."""
    if not isinstance(message, Mapping) or "source_order" not in message:
        return None
    order = message["source_order"]
    if type(order) is not int or not 0 <= order < SOURCE_ORDER_LIMIT:
        raise ValueError("event.message.source_order must be an integer from 0 to 2^53-1")
    return order


def source_order_block(message: Any, *, received: str, sent: str = "") -> dict | None:
    """A new arrival's immutable reading facts (``source_order``): its provider send time (history's
    ``source_timestamp``, else a readable ``message.sent_at`` no later than the Host's receipt ``received``)
    and the provider's order. None when neither is known: the row then reads at its ``ts`` with no order
    (``place_arrivals.READING_TIME``)."""
    if not sent:
        at, receipt = _instant(message.get("sent_at") if isinstance(message, Mapping) else None), _instant(received)
        sent = at.isoformat() if at is not None and (receipt is None or at <= receipt) else ""
    try:
        order = source_order(message)
    except ValueError:  # both doors refuse it; a value an event kept from before them stays unknown
        order = None
    return {"time": sent or None, "order": order} if sent or order is not None else None


def _unicode(value: Any, field: str) -> None:
    """Refuse text the store cannot write: a lone surrogate (a JSON ``\\ud800`` escape) is no character."""
    try:
        json.dumps(value, ensure_ascii=False).encode("utf-8")
    except UnicodeEncodeError:
        raise PresenceHistoryInvalid(field, f"{field} holds text that is not valid Unicode") from None


def _source_time(value: Any, field: str, received: datetime, message: Mapping[str, Any]) -> str:
    parsed = _instant(value)
    if parsed is None:
        raise PresenceHistoryInvalid(field, f"{field} must be an ISO 8601 time with a time zone")
    if parsed > received:  # a time ahead of the Host's clock would sort above live rows
        raise PresenceHistoryInvalid(field, f"{field} is later than the Host's receipt time")
    if "sent_at" in message and message["sent_at"] is not None and _instant(message["sent_at"]) != parsed:
        raise PresenceHistoryInvalid(field, f"{field} and event.message.sent_at name different times")
    return parsed.isoformat()


def _event(value: Any, field: str) -> Any:
    from ouroboros.presence_bindings import conversation_key
    from ouroboros.presence_runner import PresenceTurnEvent

    if not isinstance(value, dict) or set(value) != _EVENT_FIELDS:
        raise PresenceHistoryInvalid(field, f"{field} must carry exactly the live event fields, without facts or identity")
    _unicode(value, field)
    for key in ("source_event_id", "provider", "account_id", "conversation_id", "thread_id", "conversation_key", "text"):
        if not isinstance(value[key], str):
            raise PresenceHistoryInvalid(f"{field}.{key}", f"{field}.{key} must be a string")
    for key in ("actor", "conversation", "message"):
        if not isinstance(value[key], dict):
            raise PresenceHistoryInvalid(f"{field}.{key}", f"{field}.{key} must be an object")
    actor = value["actor"].get("platform_actor_id")
    if not isinstance(actor, str) or not actor.strip():
        raise PresenceHistoryInvalid(f"{field}.actor.platform_actor_id", "history names its sender by platform_actor_id")
    if "authenticated" in value["actor"] and type(value["actor"]["authenticated"]) is not bool:
        raise PresenceHistoryInvalid(f"{field}.actor.authenticated", "actor.authenticated must be a boolean, as live")
    if "transport_queue" in value["conversation"]:
        raise PresenceHistoryInvalid(f"{field}.conversation.transport_queue", "history carries no inbox observation")
    provider, account, conversation, thread = (value[key].strip() for key in (
        "provider", "account_id", "conversation_id", "thread_id"))
    source_event_id = value["source_event_id"].strip()
    if not (source_event_id and provider and account and conversation):
        raise PresenceHistoryInvalid(field, f"{field} is missing identity facts")
    return PresenceTurnEvent(
        source_event_id=source_event_id, provider=provider, account_id=account, conversation_id=conversation,
        thread_id=thread, conversation_key=conversation_key(provider, account, conversation, thread),
        actor=dict(value["actor"]), conversation=dict(value["conversation"]), message=dict(value["message"]),
        text=value["text"])


def parse_history_request(payload: Any, *, now: datetime | None = None) -> HistoryRequest:
    """Validate the whole batch before anything is written (CREATING_SKILLS wire contract)."""
    if (not isinstance(payload, dict) or not {"import_id", "records"} <= set(payload)
            or set(payload) - {"import_id", "records", "binding_id"}):
        raise PresenceHistoryInvalid("payload", "expected import_id, records and an optional binding_id")
    import_id = payload["import_id"]
    if not isinstance(import_id, str) or not import_id or import_id != import_id.strip() or len(import_id) > 128:
        raise PresenceHistoryInvalid("import_id", "import_id must be 1-128 characters without surrounding spaces")
    _unicode(import_id, "import_id")
    binding_id = payload.get("binding_id", "")
    if not isinstance(binding_id, str):
        raise PresenceHistoryInvalid("binding_id", "binding_id must be a string")
    raw = payload["records"]
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_RECORDS:
        raise PresenceHistoryInvalid("records", f"records must hold 1-{MAX_RECORDS} records")
    received = now or datetime.now(timezone.utc)
    records = []
    for index, value in enumerate(raw):
        field = f"records[{index}]"
        if (not isinstance(value, dict) or not {"event", "authorship", "source_timestamp"} <= set(value)
                or set(value) - _RECORD_FIELDS):
            raise PresenceHistoryInvalid(field, f"{field} must carry event, authorship and source_timestamp")
        if value.get("staged_files", []) != []:
            raise PresenceHistoryInvalid(f"{field}.staged_files", "history version 1 stages no files")
        if value["authorship"] not in AUTHORSHIPS:
            raise PresenceHistoryInvalid(f"{field}.authorship", "authorship must be participant or platform_account")
        event = _event(value["event"], f"{field}.event")
        try:
            source_order(event.message)
        except ValueError as exc:
            raise PresenceHistoryInvalid(f"{field}.event.message.source_order", str(exc)) from None
        if records and event.conversation_key != records[0].event.conversation_key:
            raise PresenceHistoryInvalid(f"{field}.event", "one request carries one conversation")
        records.append(HistoryRecord(event, value["authorship"], _source_time(
            value["source_timestamp"], f"{field}.source_timestamp", received, event.message)))
    return HistoryRequest(import_id, binding_id.strip(), tuple(records))


def _carrier(admission: Any, event: Any) -> dict:
    """The live arrival writer's task facts for a record that starts no task."""
    place = admission.place
    return {"metadata": {
        "presence": {"binding_id": admission.binding_id, "transport_skill": admission.transport_skill,
                     "behavior_skill": admission.behavior_skill, "profile_fingerprint": admission.profile_fingerprint,
                     "place_id": place.place_id,
                     "event": {key: getattr(event, key) for key in (
                         "source_event_id", "provider", "account_id", "conversation_id", "thread_id",
                         "conversation_key")} | {"actor": dict(event.actor)}},
        "focus": {"place_id": place.place_id, "conversation_key": event.conversation_key}}}


def _settled(event: Any, seq: int, row: Mapping[str, Any], *, duplicate: bool, task_id: str) -> dict:
    marker = row.get("historical_import") if isinstance(row.get("historical_import"), dict) else None
    return {"source_event_id": event.source_event_id, "ok": True, "recorded": True, "duplicate": duplicate,
            "custody": "history" if marker is not None else "live",
            "turn_ref": str(row.get("task_id") or task_id), "history_id": f"chat:{seq}",
            "ts": str(row.get("ts") or ""),
            "received_at": str(marker.get("received_at") or "") if marker is not None else str(row.get("ts") or "")}


def _refusal(event: Any, code: str, disposition: str, error: str) -> dict:
    return {"source_event_id": event.source_event_id, "ok": False, "code": code,
            "disposition": disposition, "error": error}


def source_message_facts(message: Any) -> dict:
    """The compared provider facts of one ``event.message``; an edit time compares as an instant."""
    message = message if isinstance(message, Mapping) else {}
    facts = {key: message.get(key) for key in SOURCE_MESSAGE_FACTS}
    if (edited := _instant(facts["edited_at"])) is not None:
        facts["edited_at"] = edited.isoformat()
    media = message.get("media")
    facts["media"] = ({key: value for key, value in media.items() if key not in _MEDIA_OBSERVATIONS}
                      if isinstance(media, Mapping) else media)
    return facts


def _matches(row: Mapping[str, Any], identity: str, record: HistoryRecord) -> bool:
    """The live identity, the source message facts and the send time; history also fixes authorship."""
    transport = row.get("transport") if isinstance(row.get("transport"), Mapping) else {}
    message = transport.get("message") if isinstance(transport.get("message"), Mapping) else {}
    if (row.get("presence_event_identity") != identity
            or source_message_facts(message) != source_message_facts(record.event.message)):
        return False
    marker = row.get("historical_import")
    if isinstance(marker, dict):
        return marker.get("authorship") == record.authorship and str(row.get("ts") or "") == record.ts
    # A live capture of the same revision keeps its custody and is never rewritten. Its row
    # time is its receipt, not a send time: only a send time it recorded is compared.
    sent = _instant(message.get("sent_at"))
    return sent is None or sent.isoformat() == record.ts


def _record_one(root: Path, admission: Any, import_id: str, record: HistoryRecord) -> dict:
    from ouroboros import dialogue_store
    from ouroboros.place_arrivals import _record, intake_lock
    from ouroboros.presence_runner import (
        _log_dialogue,
        _stable_numeric_id,
        presence_event_identity,
        presence_turn_task_id,
    )

    event = record.event
    task_id = presence_turn_task_id(admission.binding_id, event.source_event_id)
    identity = presence_event_identity(admission.binding_id, event)
    with intake_lock(root, event.conversation_key):
        with dialogue_store.transaction(root, write=False) as conn:
            prior = _record(conn, task_id)
        if prior is not None:
            seq, row, duplicate = int(prior["seq"]), dialogue_store.decode(prior["body"]), True
        else:
            marker = {"version": HISTORY_IMPORT_VERSION, "import_id": import_id, "authorship": record.authorship,
                      "received_at": utc_now_iso(), "transport_skill": admission.transport_skill}
            receipt = _log_dialogue(
                root, direction="in", chat_id=_stable_numeric_id("presence-conversation", event.conversation_key),
                user_id=_stable_numeric_id("presence-actor-log", str(event.actor["platform_actor_id"])),
                text=event.text, event=event, task=_carrier(admission, event), task_id=task_id,
                ts=record.ts, historical_import=marker)
            # The named-row dedup answers a same-room row of this event id under another turn.
            seq, row, duplicate = receipt.seq, receipt.row, receipt.duplicate
    if duplicate and not _matches(row, identity, record):
        return _refusal(event, "presence_event_identity_conflict", "rejected",
                        "this source event is already recorded with different facts")
    return _settled(event, seq, row, duplicate=duplicate, task_id=task_id)


def record_history(drive_root: Any, admission: Any, request: HistoryRequest) -> tuple[int, dict]:
    """Record each record in order; ``(HTTP status, body)`` per the frozen v1 contract.

    Before any write the chat archive must be imported: a row it still holds could be
    this event's live capture, and unknown is never answered "not recorded". A write that
    fails leaves its outcome unknown, so it and every later record answer ``retry``.
    """
    from ouroboros import dialogue_store
    from ouroboros.place_arrivals import _dialogue_indexes

    root = Path(drive_root)
    dialogue_store.require_imported(root, "chat")
    _dialogue_indexes(root)  # built once, before an import's rows can reach any reader (Main leaves them out by it)
    results: list[dict] = []
    for index, record in enumerate(request.records):
        try:
            results.append(_record_one(root, admission, request.import_id, record))
        except Exception:
            log.warning("Presence history record %d of import %s not confirmed", index, request.import_id,
                        exc_info=True)
            results.append(_refusal(record.event, "presence_history_unwritable", "retry",
                                    "the history write was not confirmed; resend this record unchanged"))
            results.extend(_refusal(later.event, "presence_history_unwritable", "retry",
                                    "not attempted after an unconfirmed write; resend this record unchanged")
                           for later in request.records[index + 1:])
            break
    unwritten = sum(1 for result in results if result.get("code") == "presence_history_unwritable")
    conflicts = sum(1 for result in results if result.get("code") == "presence_event_identity_conflict")
    if unwritten:
        return 503, {"ok": False, "code": "presence_history_unwritable", "disposition": "retry",
                     "error": f"{unwritten} of {len(results)} records were not confirmed; resend them unchanged",
                     "import_id": request.import_id, "results": results}
    if conflicts:
        return 409, {"ok": False, "code": "presence_event_identity_conflict", "disposition": "rejected",
                     "error": f"{conflicts} of {len(results)} records conflict with an already recorded source event",
                     "import_id": request.import_id, "results": results}
    return 200, {"ok": True, "import_id": request.import_id, "results": results}
