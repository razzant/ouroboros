"""Neutral Telegram update parsing with exact provider provenance for Presence."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import re
from typing import Any, Dict, Optional, Sequence, Tuple

from ..telegram_poll_binding import update_key

# Matches Bot API getFile download limit (20 MiB).
CLOUD_DOWNLOAD_LIMIT = 20 * 1024 * 1024
BOT_API_DOWNLOAD_LIMIT = CLOUD_DOWNLOAD_LIMIT

_PLACE_CHAT_TYPES = frozenset({"private", "group", "supergroup", "channel"})
NON_TEXT_KEY = "non_text"
_NON_TEXT_CHARS = 4000
ALBUM_POSITION_KEY = "media_group_position"

_AUDIO_SUFFIXES = {
    "audio/ogg": ".ogg", "audio/opus": ".ogg", "audio/mpeg": ".mp3", "audio/mp3": ".mp3",
    "audio/mp4": ".m4a", "audio/m4a": ".m4a", "audio/x-m4a": ".m4a", "audio/aac": ".aac",
    "audio/flac": ".flac", "audio/x-flac": ".flac", "audio/wav": ".wav", "audio/x-wav": ".wav",
    "audio/webm": ".webm",
}
_OTHER_CONTENT = ("live_photo", "rich_message", "users_shared", "chat_shared", "web_app_data")
_VIDEO_SUFFIXES = {
    "video/mp4": ".mp4", "video/quicktime": ".mov", "video/webm": ".webm",
    "video/x-matroska": ".mkv", "video/mpeg": ".mpeg", "image/gif": ".gif",
}
_OPTIONAL_ATTACHMENT_KEYS = frozenset({"duration_sec", "status", "note"})


def _megabytes(size: int) -> str:
    return (
        f"{size / 1e6:.0f} MB"
        if size >= 10_000_000
        else f"{size / 1e6:.1f} MB"
        if size >= 1_000_000
        else f"{size:,} bytes"
    )


def too_large_note(size: int = 0, limit: int = CLOUD_DOWNLOAD_LIMIT, telegram_said: str = "") -> str:
    """One factual sentence on why the file did not arrive (core contract); no advice."""
    rule = f"the cloud Bot API gives bots files only up to {limit // (1024 * 1024)} MB"
    if size > limit:
        return f"{_megabytes(size)}; {rule}"
    return f'Telegram answered "{telegram_said}"; {rule}' if telegram_said else f"Over the limit; {rule}"


def too_large_marks(size: int, limit: int = CLOUD_DOWNLOAD_LIMIT) -> Dict[str, str]:
    """Telegram never serves a bot a file above the download limit; retrying it would hold its conversation."""
    return {"status": "too_large", "note": too_large_note(size, limit)} if size > limit else {}


@dataclass(frozen=True)
class TelegramAttachment:
    kind: str
    file_id: str
    file_unique_id: str
    file_name: str
    mime_type: str
    file_size: int
    duration_sec: Optional[int] = None
    status: Optional[str] = None
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            key: value
            for key, value in asdict(self).items()
            if value is not None or key not in _OPTIONAL_ATTACHMENT_KEYS
        }


@dataclass(frozen=True)
class TelegramEvent:
    source_event_id: str
    provider: str
    account_id: str
    conversation_id: str
    thread_id: str
    conversation_key: str
    actor: Dict[str, Any]
    conversation: Dict[str, Any]
    message: Dict[str, Any]
    text: str
    facts: Optional[list[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.facts is None:
            result.pop("facts")
        return result


def parse_telegram_update(
    update: Dict[str, Any],
    *,
    bot_account_id: str,
    management_group_id: str = "",
    places: bool = False,
    bot_username: str = "",
    words: Sequence[str] = (),
    download_limit: int = CLOUD_DOWNLOAD_LIMIT,
) -> Optional[TelegramEvent]:
    """Return one factual external-message event or ``None`` for unsupported updates."""
    if not isinstance(update, dict):
        return None
    update_id = _integer(update.get("update_id"))
    event_kind = "message"
    message = update.get("message")
    if not isinstance(message, dict) and isinstance(update.get("edited_message"), dict):
        message = update["edited_message"]
        event_kind = "edited_message"
    if places and not isinstance(message, dict):
        for kind in ("channel_post", "edited_channel_post"):
            if isinstance(update.get(kind), dict):
                message, event_kind = update[kind], kind
                break
    reaction = update.get("message_reaction")
    if not isinstance(reaction, dict):
        reaction = update.get("message_reaction_count")
    if not isinstance(message, dict) and isinstance(reaction, dict):
        return _parse_reaction_update(
            update,
            reaction,
            bot_account_id=bot_account_id,
            management_group_id=management_group_id,
            places=places,
        )
    if update_id is None or not isinstance(message, dict):
        return None

    actor = (message.get("sender_chat") or message.get("from")) if places else message.get("from")
    chat = message.get("chat")
    message_id = _integer(message.get("message_id"))
    if not isinstance(actor, dict) or not isinstance(chat, dict) or message_id is None:
        return None
    if (not places and actor.get("is_bot")) or (places and str(actor.get("id")) == bot_account_id):
        return None
    if places and chat.get("type") not in _PLACE_CHAT_TYPES:
        return None

    actor_id = _integer(actor.get("id"))
    chat_id = _integer(chat.get("id"))
    if actor_id is None or chat_id is None:
        return None

    raw_thread_id = _integer(message.get("message_thread_id"))
    topic_id = raw_thread_id if message.get("is_topic_message") is True else None
    bot_id = str(bot_account_id or "unknown").strip() or "unknown"
    topic_key = str(topic_id) if topic_id is not None else "0"
    conversation_key = f"telegram:{bot_id}:{chat_id}:{topic_key}"
    text = str(message.get("text") or message.get("caption") or "")

    actor_fact = {
        "platform": "telegram",
        "platform_actor_id": str(actor_id),
        "username": str(actor.get("username") or ""),
        "first_name": str(actor.get("first_name") or ""),
        "last_name": str(actor.get("last_name") or ""),
        "language_code": str(actor.get("language_code") or ""),
    }
    if places:
        actor_fact["is_bot"] = bool(actor.get("is_bot"))
        if message.get("sender_chat"):
            actor_fact["sender_chat"] = deepcopy(message["sender_chat"])
            actor_fact["authenticated"] = False
    conversation_fact = {
        "platform": "telegram",
        "bot_account_id": bot_id,
        "chat_id": str(chat_id),
        "chat_type": str(chat.get("type") or "unknown"),
        "title": str(chat.get("title") or ""),
        "username": str(chat.get("username") or ""),
        "topic_id": topic_id,
        **({"is_forum": True} if chat.get("is_forum") is True else {}),
    }
    if str(chat_id) == str(management_group_id).strip() and chat.get("type") in {"group", "supergroup"}:
        conversation_fact["configured_room"] = "management_group"
    attachments = _attachments(message, download_limit)
    described = _non_text(message)
    message_fact = {
        "event_kind": event_kind,
        "message_id": message_id,
        "sent_at_epoch": _integer(message.get("date")),
        **_time_facts(_integer(message.get("date")), _integer(message.get("edit_date"))),
        "reply_to_message_id": _reply_message_id(message),
        "attachments": [item.to_dict() for item in attachments],
    }
    if raw_thread_id is not None and topic_id is None:
        message_fact["reply_chain_id"] = raw_thread_id
    for key in ("entities", "caption_entities", "quote", "forward_origin", "is_automatic_forward", "media_group_id"):
        if key in message:
            message_fact[key] = deepcopy(message[key])
    if described:
        message_fact[NON_TEXT_KEY] = described
    reply = message.get("reply_to_message")
    if isinstance(reply, dict):
        message_fact["reply_to_message"] = _reply_context(reply)
    if not text.strip() and not attachments and not described:
        return None
    if event_kind.startswith("edited_") and not text.strip() and not attachments and described[0]["kind"] == "location":
        return None

    event_id = update_key(update) or str(update_id)
    return TelegramEvent(
        source_event_id=f"telegram:{bot_id}:{event_id}",
        provider="telegram",
        account_id=bot_id,
        conversation_id=str(chat_id),
        thread_id=str(topic_id) if topic_id is not None else "",
        conversation_key=conversation_key,
        actor=actor_fact,
        conversation=conversation_fact,
        message=message_fact,
        text=text,
        facts=telegram_wake_facts(message, bot_id, bot_username, words) if places else None,
    )


def telegram_wake_facts(message: Dict[str, Any], bot_id: str, username: str, words: Sequence[str] = ()) -> list[str]:
    """Provider entities establish mentions; literal owner words are a separate signal."""
    facts = ["any_message"]
    if (message.get("chat") or {}).get("type") == "private":
        facts.append("direct_message")
    if not message.get("sender_chat") and (message.get("from") or {}).get("is_bot"):
        facts.append("from_bot")
    reply = message.get("reply_to_message") or {}
    if str((reply.get("from") or {}).get("id") or "") == bot_id:
        facts.append("reply_to_me")
    for text_key, entities_key in (("text", "entities"), ("caption", "caption_entities")):
        wire = str(message.get(text_key) or "").encode("utf-16-le")
        for entity in message.get(entities_key) or []:
            if entity.get("type") == "text_mention" and str((entity.get("user") or {}).get("id") or "") == bot_id:
                if "mention" not in facts:
                    facts.append("mention")
            if entity.get("type") == "mention" and username:
                start, length = entity.get("offset"), entity.get("length")
                if type(start) is int and type(length) is int and start >= 0 and length > 0:
                    token = wire[2 * start:2 * (start + length)].decode("utf-16-le", errors="replace")
                    if token.casefold() == "@" + username.lstrip("@").casefold() and "mention" not in facts:
                        facts.append("mention")
    if words:
        text = str(message.get("text") or message.get("caption") or "").casefold()
        if any(re.search(r"(?<!\w)" + re.escape(word.casefold()) + r"(?!\w)", text) for word in words if word):
            facts.append("signal.words")
    return facts


def _parse_reaction_update(
    update: Dict[str, Any],
    reaction: Dict[str, Any],
    *,
    bot_account_id: str,
    management_group_id: str = "",
    places: bool = False,
) -> Optional[TelegramEvent]:
    counted = "reactions" in reaction
    user = reaction.get("user") if isinstance(reaction.get("user"), dict) else None
    actor_chat = reaction.get("actor_chat") if isinstance(reaction.get("actor_chat"), dict) else None
    source = user or actor_chat or {}
    chat = reaction.get("chat") or {}
    message_id = _integer(reaction.get("message_id"))
    chat_id = _integer(chat.get("id")) if isinstance(chat, dict) else None
    actor_id = _integer(source.get("id"))
    update_id = _integer(update.get("update_id"))
    if update_id is None or message_id is None or chat_id is None or (actor_id is None and not counted):
        return None
    if (not places and source.get("is_bot")) or (places and str(actor_id) == bot_account_id):
        return None
    if places and chat.get("type") not in _PLACE_CHAT_TYPES:
        return None
    if counted:
        values = {
            "change": "count",
            "counts": [
                {**_reaction_values([row.get("type")])[0], "count": row["total_count"]}
                for row in (reaction["reactions"] if isinstance(reaction["reactions"], list) else [])
                if isinstance(row, dict)
                and _reaction_values([row.get("type")])
                and type(row.get("total_count")) is int
                and row["total_count"] >= 0
            ],
        }
        raw = {"reaction_counts": deepcopy(reaction["reactions"])}
    else:
        old, new = _reaction_values(reaction.get("old_reaction")), _reaction_values(reaction.get("new_reaction"))
        if sorted(map(str, old)) == sorted(map(str, new)):
            return None
        values = (
            {"change": "add", "new": new}
            if not old
            else {"change": "remove", "old": old}
            if not new
            else {"change": "replace", "old": old, "new": new}
        )
        raw = {
            "old_reaction": deepcopy(reaction.get("old_reaction") or []),
            "new_reaction": deepcopy(reaction.get("new_reaction") or []),
        }
    bot_id = str(bot_account_id or "unknown").strip() or "unknown"
    conversation_key = f"telegram:{bot_id}:{chat_id}:0"
    conversation = {
        "platform": "telegram",
        "bot_account_id": bot_id,
        "chat_id": str(chat_id),
        "chat_type": str(chat.get("type") or "unknown"),
        "title": str(chat.get("title") or ""),
        "username": str(chat.get("username") or ""),
        "topic_id": None,
        **({"is_forum": True} if chat.get("is_forum") is True else {}),
    }
    if str(chat_id) == str(management_group_id).strip() and chat.get("type") in {"group", "supergroup"}:
        conversation["configured_room"] = "management_group"
    proven = chat.get("is_forum") is not True and chat.get("type") in _PLACE_CHAT_TYPES
    sent_at_epoch = _integer(reaction.get("date"))
    event_id = update_key(update) or str(update_id)
    return TelegramEvent(
        source_event_id=f"telegram:{bot_id}:{event_id}",
        provider="telegram",
        account_id=bot_id,
        conversation_id=str(chat_id),
        thread_id="",
        conversation_key=conversation_key,
        actor=_reaction_actor(user, actor_chat, places=places),
        conversation=conversation,
        message={
            "event_kind": "message_reaction_count" if counted else "message_reaction",
            "message_id": message_id,
            "sent_at_epoch": sent_at_epoch,
            **_time_facts(sent_at_epoch),
            **raw,
            "reaction": {
                "schema_version": 1,
                **values,
                "target": {"message_id": str(message_id), "resolution": "resolved" if proven else "unresolved"},
            },
        },
        text="",
        facts=(["from_bot"] if source.get("is_bot") else []) if places else None,
    )


def _reaction_actor(
    user: Optional[Dict[str, Any]], actor_chat: Optional[Dict[str, Any]], *, places: bool
) -> Dict[str, Any]:
    if user:
        return {
            "kind": "user",
            "platform": "telegram",
            "platform_actor_id": str(user["id"]),
            "username": str(user.get("username") or ""),
            "first_name": str(user.get("first_name") or ""),
            "last_name": str(user.get("last_name") or ""),
            **({"is_bot": bool(user.get("is_bot"))} if places else {}),
        }
    if actor_chat:
        return {
            "kind": "chat",
            "platform": "telegram",
            "platform_actor_id": str(actor_chat["id"]),
            "display_name": str(actor_chat.get("title") or ""),
            "username": str(actor_chat.get("username") or ""),
            "authenticated": False,
        }
    return {"kind": "anonymous", "platform": "telegram", "authenticated": False}


def _reaction_values(raw: Any) -> list[Dict[str, str]]:
    values = []
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, dict) and isinstance(item.get("type"), str) and item["type"]:
            field = {"emoji": "emoji", "custom_emoji": "custom_emoji_id"}.get(item["type"], "")
            values.append({"type": item["type"], "value": str(item.get(field) or "")})
    return values


def _iso_time(epoch: int) -> str:
    try:
        return datetime.fromtimestamp(epoch, timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return ""


def _time_facts(sent_epoch: Optional[int], edited_epoch: Optional[int] = None) -> Dict[str, str]:
    times = (("sent_at", sent_epoch), ("edited_at", edited_epoch))
    return {key: value for key, epoch in times if epoch and (value := _iso_time(epoch))}


def _attachments(message: Dict[str, Any], limit: int = CLOUD_DOWNLOAD_LIMIT) -> Tuple[TelegramAttachment, ...]:
    result = []
    photo = _photo(message.get("photo"), limit)
    if photo is not None:
        result.append(photo)
    document = message.get("document")
    if isinstance(document, dict) and document.get("file_id") and not _file(message.get("animation")):
        result.append(
            TelegramAttachment(
                kind="document",
                file_id=str(document.get("file_id")),
                file_unique_id=str(document.get("file_unique_id") or ""),
                file_name=str(document.get("file_name") or "document"),
                mime_type=str(document.get("mime_type") or "application/octet-stream"),
                file_size=_integer(document.get("file_size")) or 0,
                **too_large_marks(_integer(document.get("file_size")) or 0, limit),
            )
        )
    for kind in ("voice", "audio"):
        item = message.get(kind)
        if isinstance(item, dict) and item.get("file_id"):
            result.append(_sound(kind, item, limit))
    for kind in ("video", "video_note", "animation"):
        if _file(message.get(kind)):
            result.append(_moving(kind, message[kind], limit))
    paid = message.get("paid_media")
    for item in (paid.get("paid_media") if isinstance(paid, dict) else None) or []:
        attachment = _paid(item, limit) if isinstance(item, dict) else None
        if attachment is not None:
            result.append(attachment)
    return tuple(result)


def _file(value: Any) -> bool:
    return isinstance(value, dict) and bool(value.get("file_id"))


def _photo(photos: Any, limit: int) -> Optional[TelegramAttachment]:
    choices = [item for item in photos if _file(item)] if isinstance(photos, list) else []
    if not choices:
        return None
    photo = choices[-1]
    size = _integer(photo.get("file_size")) or 0
    return TelegramAttachment(
        kind="photo",
        file_id=str(photo.get("file_id")),
        file_unique_id=str(photo.get("file_unique_id") or ""),
        file_name=f"photo-{photo.get('file_unique_id') or photo.get('file_id')}.jpg",
        mime_type="image/jpeg",
        file_size=size,
        **too_large_marks(size, limit),
    )


def _moving(kind: str, item: Dict[str, Any], limit: int) -> TelegramAttachment:
    unique = str(item.get("file_unique_id") or "")
    mime = str(item.get("mime_type") or "video/mp4")
    name = "" if kind == "video_note" else str(item.get("file_name") or "")
    name = name or f"{kind}-{unique or item.get('file_id')}"
    suffix = _VIDEO_SUFFIXES.get(mime.split(";", 1)[0].strip().lower(), "")
    if suffix and not name.lower().endswith(suffix):
        name += suffix
    size = _integer(item.get("file_size")) or 0
    duration = _integer(item.get("duration"))
    return TelegramAttachment(
        kind=kind,
        file_id=str(item.get("file_id")),
        file_unique_id=unique,
        file_name=name,
        mime_type=mime,
        file_size=size,
        duration_sec=duration if duration is not None and duration >= 0 else None,
        **too_large_marks(size, limit),
    )


def _paid(item: Dict[str, Any], limit: int) -> Optional[TelegramAttachment]:
    if item.get("type") == "photo":
        return _photo(item.get("photo"), limit)
    if item.get("type") == "video" and _file(item.get("video")):
        return _moving("video", item["video"], limit)
    return None


def _quoted(value: Any) -> str:
    return f'"{value}"' if str(value or "").strip() else ""


def _join(*parts: Any) -> str:
    return " ".join(str(part) for part in parts if str(part if part is not None else "").strip())


def _poll(poll: Dict[str, Any]) -> str:
    options = "; ".join(_quoted(option.get("text")) for option in poll.get("options") or [] if isinstance(option, dict))
    flags = ", ".join(
        flag
        for flag, on in (
            ("anonymous", poll.get("is_anonymous")),
            ("closed", poll.get("is_closed")),
            ("multiple answers", poll.get("allows_multiple_answers")),
        )
        if on
    )
    return _join(
        f"{poll.get('type') or 'regular'} poll",
        _quoted(poll.get("question")),
        f"options: {options}" if options else "",
        f"({flags})" if flags else "",
        f"explanation: {_quoted(poll.get('explanation'))}" if poll.get("explanation") else "",
    )


def _place(location: Any) -> str:
    location = location if isinstance(location, dict) else {}
    if location.get("latitude") is None or location.get("longitude") is None:
        return ""
    extra = [
        f"accuracy {location['horizontal_accuracy']} m" if location.get("horizontal_accuracy") else "",
        f"live for {location['live_period']} s" if location.get("live_period") else "",
    ]
    extra = [part for part in extra if part]
    return _join(
        f"latitude {location['latitude']}, longitude {location['longitude']}",
        f"({', '.join(extra)})" if extra else "",
    )


def _paid_media(paid: Dict[str, Any]) -> str:
    items = []
    for item in paid.get("paid_media") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "preview":
            size = f"{item['width']}x{item['height']}" if item.get("width") and item.get("height") else ""
            items.append(_join("preview without a file", size, f"{item['duration']} s" if item.get("duration") else ""))
        else:
            items.append(str(item.get("type") or "item"))
    stars = f"for {paid['star_count']} Telegram Stars" if paid.get("star_count") is not None else ""
    return _join("paid media", stars, f"({len(items)} items: {', '.join(items)})" if items else "")


def _non_text(message: Dict[str, Any]) -> list:
    described = []

    def add(kind: str, description: str) -> None:
        if description.strip():
            if len(description) > _NON_TEXT_CHARS:
                description = description[: _NON_TEXT_CHARS - 12] + " [shortened]"
            described.append({"kind": kind, "description": description})

    sticker = message.get("sticker")
    if isinstance(sticker, dict):
        shape = "video" if sticker.get("is_video") else "animated" if sticker.get("is_animated") else ""
        add(
            "sticker",
            _join(
                sticker.get("emoji"),
                shape,
                "sticker",
                f"from set {_quoted(sticker.get('set_name'))}" if sticker.get("set_name") else "",
            ),
        )
    if isinstance(message.get("poll"), dict):
        add("poll", _poll(message["poll"]))
    venue = message.get("venue")
    if isinstance(venue, dict):
        where = _place(venue.get("location"))
        add("venue", _join(_quoted(venue.get("title")), venue.get("address"), f"({where})" if where else ""))
    elif isinstance(message.get("location"), dict):
        add("location", _place(message["location"]))
    contact = message.get("contact")
    if isinstance(contact, dict):
        add(
            "contact",
            _join(
                contact.get("first_name"),
                contact.get("last_name"),
                contact.get("phone_number"),
                f"(Telegram user {contact['user_id']})" if contact.get("user_id") else "",
            ),
        )
    dice = message.get("dice")
    if isinstance(dice, dict):
        add("dice", _join(dice.get("emoji"), f"rolled {dice['value']}" if dice.get("value") is not None else ""))
    story = message.get("story")
    if isinstance(story, dict):
        chat = story.get("chat") if isinstance(story.get("chat"), dict) else {}
        add(
            "story",
            _join(
                "story",
                story.get("id"),
                "from",
                _quoted(chat.get("title")) or chat.get("first_name"),
                f"@{chat['username']}" if chat.get("username") else "",
                f"(chat {chat['id']})" if chat.get("id") else "",
            ),
        )
    if isinstance(message.get("paid_media"), dict):
        add("paid_media", _paid_media(message["paid_media"]))
    game = message.get("game")
    if isinstance(game, dict):
        add("game", _join("game", _quoted(game.get("title")), game.get("description")))
    invoice = message.get("invoice")
    if isinstance(invoice, dict):
        amount = invoice.get("total_amount")
        add(
            "invoice",
            _join(
                "invoice",
                _quoted(invoice.get("title")),
                invoice.get("description"),
                f"total_amount {amount} {invoice.get('currency') or ''}".strip() if amount is not None else "",
            ),
        )
    for field in ("giveaway", "giveaway_winners"):
        giveaway = message.get(field)
        if isinstance(giveaway, dict):
            add(
                "giveaway",
                _join(
                    "giveaway winners" if field == "giveaway_winners" else "giveaway",
                    f"{giveaway['winner_count']} winners" if giveaway.get("winner_count") else "",
                    f"prize: {giveaway['prize_description']}" if giveaway.get("prize_description") else "",
                ),
            )
    checklist = message.get("checklist")
    if isinstance(checklist, dict):
        tasks = "; ".join(
            _quoted(task.get("text")) for task in checklist.get("tasks") or [] if isinstance(task, dict)
        )
        add("checklist", _join("checklist", _quoted(checklist.get("title")), f"tasks: {tasks}" if tasks else ""))
    for field in _OTHER_CONTENT:
        if field in message:
            add("other", f"Telegram {field.replace('_', ' ')} (not described by this connector)")
    return described


def _sound(kind: str, item: Dict[str, Any], limit: int = CLOUD_DOWNLOAD_LIMIT) -> TelegramAttachment:
    unique = str(item.get("file_unique_id") or "")
    mime = str(item.get("mime_type") or ("audio/ogg" if kind == "voice" else "application/octet-stream"))
    suffix = _AUDIO_SUFFIXES.get(mime.split(";", 1)[0].strip().lower(), "")
    name = str(item.get("file_name") or "") if kind == "audio" else ""
    if not name:
        name = f"{kind}-{unique or item.get('file_id')}"
    if suffix and not name.lower().endswith(suffix) and not (
        suffix == ".ogg" and name.lower().endswith((".oga", ".opus"))
    ):
        name += suffix
    size = _integer(item.get("file_size")) or 0
    duration = _integer(item.get("duration"))
    return TelegramAttachment(
        kind=kind,
        file_id=str(item.get("file_id")),
        file_unique_id=unique,
        file_name=name,
        mime_type=mime,
        file_size=size,
        duration_sec=duration if duration is not None and duration >= 0 else None,
        **too_large_marks(size, limit),
    )


def _reply_message_id(message: Dict[str, Any]) -> Optional[int]:
    reply = message.get("reply_to_message")
    if not isinstance(reply, dict):
        return None
    return _integer(reply.get("message_id"))


def _reply_context(reply: Dict[str, Any]) -> Dict[str, Any]:
    fields = (
        "message_id",
        "message_thread_id",
        "date",
        "from",
        "sender_chat",
        "chat",
        "text",
        "caption",
        "entities",
        "caption_entities",
        "forward_origin",
    )
    snapshot = {key: deepcopy(reply[key]) for key in fields if key in reply}
    attachments = _attachments(reply)
    if attachments:
        snapshot["attachments"] = [item.to_dict() for item in attachments]
    described = _non_text(reply)
    if described:
        snapshot[NON_TEXT_KEY] = described
    return snapshot


def _integer(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None
