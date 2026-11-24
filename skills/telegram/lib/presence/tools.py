"""Model-selected Telegram Presence tools backed by the durable SQLite custody outbox."""

from __future__ import annotations

import functools
import json
import os
import pathlib
import re
import uuid
from typing import Any, Dict, Optional

from . import outbox_files
from .custody import CustodyStore
from .delivery import tool_origin
from .file_access import (
    authorize_upload_path,
    caller_scope,
    idempotency_fingerprint,
    open_authorized_upload,
    receipt_visible_to,
    sanitize_tool_receipt,
)
from .formatting import prepare_caption, prepare_text
from .host import PlaceBindingUnavailable
from .transport import CLOUD_PHOTO_UPLOAD_LIMIT, CLOUD_UPLOAD_LIMIT, MEDIA_METHODS

SEND_KINDS = ("message", *MEDIA_METHODS, "album")
ALBUM_KINDS = ("photo", "video", "audio", "document")
_VOICE_SUFFIXES = (".ogg", ".oga", ".opus", ".mp3", ".m4a")


class SendRefused(ValueError):
    """A typed refusal: nothing was queued, and ``code`` names the violated Bot API rule."""

    def __init__(self, code: str, message: str, **facts: Any):
        super().__init__(message)
        self.code = code
        self.facts = facts

    def answer(self) -> dict[str, Any]:
        return {"ok": False, "code": self.code, "error": str(self), **self.facts}


def _upload_limit(kind: str) -> int:
    return CLOUD_PHOTO_UPLOAD_LIMIT if kind == "photo" else CLOUD_UPLOAD_LIMIT


def _validate_upload_meta(kind: str, path: pathlib.Path, size: int) -> None:
    limit = _upload_limit(kind)
    if size > limit:
        raise SendRefused(
            "file_too_large",
            f"{path.name} is {size:,} bytes; the Bot API accepts a {kind} of at most "
            f"{limit // (1024 * 1024)} MB from a bot. Not queued.",
            size_bytes=size,
            limit_bytes=limit,
        )
    if kind == "voice" and not path.name.lower().endswith(_VOICE_SUFFIXES):
        raise SendRefused(
            "voice_format",
            "Telegram plays a voice message from OGG/Opus, MP3 or M4A; send this file as audio or document. Not queued.",
        )


def upload_path(ctx: Any, kind: str, file_path: Any, state_dir: pathlib.Path) -> str:
    try:
        fd, path, size = open_authorized_upload(ctx, file_path, state_dir)
    except PermissionError as exc:
        raise SendRefused("file_access_denied", str(exc)) from exc
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError) as exc:
        raise SendRefused("file_missing", "file_path must name an existing file") from exc
    except OSError as exc:
        raise SendRefused("file_access_denied", f"file_path could not be opened safely ({type(exc).__name__})") from exc
    try:
        _validate_upload_meta(kind, path, size)
    finally:
        os.close(fd)
    return str(path)


def open_and_freeze_file(ctx: Any, kind: str, file_path: Any, state_dir: Any) -> dict[str, Any]:
    """Zero-TOCTOU authorize, open via dir_fd + O_NOFOLLOW, validate, and freeze from the same fd."""
    limit = _upload_limit(kind)
    try:
        fd, path, size = open_authorized_upload(ctx, file_path, pathlib.Path(state_dir))
    except PermissionError as exc:
        raise SendRefused("file_access_denied", str(exc)) from exc
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError) as exc:
        raise SendRefused("file_missing", "file_path must name an existing file") from exc
    except OSError as exc:
        raise SendRefused("file_access_denied", f"file_path could not be opened safely ({type(exc).__name__})") from exc
    try:
        _validate_upload_meta(kind, path, size)
        return outbox_files.keep_fd(fd, path, state_dir, limit)
    except SendRefused:
        raise
    except outbox_files.CopyTooLarge as exc:
        raise SendRefused(
            "file_too_large",
            f"{exc}; the Bot API accepts a {kind} of at most {limit // (1024 * 1024)} MB from a bot. Not queued.",
            limit_bytes=limit,
        ) from exc
    except outbox_files.OutboxStorageFull as exc:
        raise SendRefused("outbox_storage_full", f"{exc}. Not queued.") from exc
    except OSError as exc:
        raise SendRefused(
            "file_copy_failed",
            f"{path.name} could not be copied for sending "
            f"({type(exc).__name__}: {exc.strerror or exc}). Not queued.",
        ) from exc
    finally:
        os.close(fd)


def queue_file(kind: str, source: str, state_dir: Any) -> dict[str, Any]:
    limit = _upload_limit(kind)
    try:
        return outbox_files.keep(pathlib.Path(source), state_dir, limit)
    except outbox_files.CopyTooLarge as exc:
        raise SendRefused(
            "file_too_large",
            f"{exc}; the Bot API accepts a {kind} of at most {limit // (1024 * 1024)} MB from a bot. Not queued.",
            limit_bytes=limit,
        ) from exc
    except outbox_files.OutboxStorageFull as exc:
        raise SendRefused("outbox_storage_full", f"{exc}. Not queued.") from exc
    except OSError as exc:
        raise SendRefused(
            "file_copy_failed",
            f"{pathlib.Path(source).name} could not be copied for sending "
            f"({type(exc).__name__}: {exc.strerror or exc}). Not queued.",
        ) from exc


def discard_files(records: list[dict[str, Any]], state_dir: Any) -> None:
    outbox_files.discard(records, state_dir)


def discard_unqueued(records: list[dict[str, Any]], state_dir: Any, store: Any, delivery_id: str) -> None:
    if not records:
        return
    if store is not None:
        try:
            named = {str(record["path"]) for record in outbox_files.records(store.outbox_payload(delivery_id))}
        except Exception:
            return
        if any(str(record.get("path")) in named for record in records):
            return
    outbox_files.discard(records, state_dir)


def frozen_caption(caption: str, markdown: bool) -> dict[str, str]:
    try:
        return prepare_caption(caption, markdown=markdown)
    except ValueError as exc:
        raise SendRefused("caption_too_long", str(exc)) from exc


def album_items(
    ctx: Any, items: Any, caption: str, markdown: bool, state_dir: Any = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(items, list) or not 2 <= len(items) <= 10:
        raise SendRefused("album_size", "an album has 2 to 10 items. Not queued.")
    if any(not isinstance(item, dict) for item in items):
        raise SendRefused("album_item", "each album item is {kind, file_path, caption}. Not queued.")
    kinds = [str(item.get("kind") or "") for item in items]
    other = next((kind for kind in kinds if kind not in ALBUM_KINDS), None)
    if other is not None:
        raise SendRefused(
            "album_kind",
            f"an album holds photo, video, audio or document items; send {other or 'that'} on its own. Not queued.",
        )
    if not (set(kinds) <= {"photo", "video"} or len(set(kinds)) == 1):
        raise SendRefused(
            "album_mixed",
            "Telegram groups photos with videos, audio only with audio and documents only with documents. Not queued.",
        )
    captions = [str(item.get("caption") or "") for item in items]
    if caption and any(captions):
        raise SendRefused("album_caption", "give the album one caption, or captions on its items, not both. Not queued.")
    captions[0] = captions[0] or caption
    rendered = [frozen_caption(text, markdown) for text in captions]
    kept: list[dict[str, Any]] = []
    if state_dir is not None:
        try:
            for kind, item in zip(kinds, items):
                kept.append(open_and_freeze_file(ctx, kind, item.get("file_path"), state_dir))
        except BaseException:
            discard_files(kept, state_dir)
            raise
        paths = [record["path"] for record in kept]
    else:
        paths = [upload_path(ctx, kind, item.get("file_path"), pathlib.Path(".")) for kind, item in zip(kinds, items)]
    queued = [{"kind": kind, "file_path": path, "caption": text} for kind, path, text in zip(kinds, paths, captions)]
    return queued, [{**item, "caption": frozen} for item, frozen in zip(queued, rendered)], kept


TELEGRAM_SEND_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "chat_id": {"type": "string"},
        "text": {
            "type": "string",
            "description": "Displayed message content only; standard Markdown unless markdown=false.",
        },
        "kind": {"type": "string", "enum": list(SEND_KINDS)},
        "file_path": {"type": "string"},
        "items": {
            "type": "array",
            "minItems": 2,
            "maxItems": 10,
            "description": "For kind=album: the files in order, each with its own optional caption.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "kind": {"type": "string", "enum": list(ALBUM_KINDS)},
                    "file_path": {"type": "string"},
                    "caption": {"type": "string"},
                },
                "required": ["kind", "file_path"],
            },
        },
        "caption": {
            "type": "string",
            "description": "Displayed media caption only, up to 1024 UTF-16 units after formatting.",
        },
        "markdown": {
            "type": "boolean",
            "default": True,
            "description": "Render standard Markdown; false preserves literal punctuation and XML.",
        },
        "topic_id": {"type": "string"},
        "reply_to_message_id": {"type": "string"},
        "request_id": {
            "type": "string",
            "description": "Separate stable deduplication key; reuse the same value for the same logical send.",
        },
    },
    "required": ["chat_id", "kind"],
}

TELEGRAM_TYPING_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "chat_id": {"type": "string", "description": "This turn's own Telegram chat ID, as in the event."},
        "topic_id": {
            "type": "integer",
            "description": "This turn's forum topic, as in the event; omit outside a forum topic.",
        },
    },
    "required": ["chat_id"],
}

MODERATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["delete_message", "restrict_member", "ban_member", "unban_member"],
        },
        "chat_id": {
            "type": "string",
            "description": "Exact Telegram chat ID from the event facts or an explicitly selected target.",
        },
        "message_id": {"type": "integer", "description": "Exact message ID for delete_message."},
        "topic_id": {
            "type": "integer",
            "description": "Known forum topic of the target for delete_message; kept local for place checks.",
        },
        "user_id": {"type": "integer", "description": "Exact actor ID for member operations."},
        "permissions": {
            "type": "object",
            "additionalProperties": {"type": "boolean"},
            "description": "Telegram ChatPermissions for restrict_member; true permissions can lift a restriction.",
        },
        "until_date": {
            "type": "integer",
            "description": "Optional Unix timestamp for ban/restrict expiry. Zero means indefinite.",
        },
        "revoke_messages": {
            "type": "boolean",
            "description": "Delete the banned member's messages; Telegram always does so in supergroups/channels.",
        },
        "request_id": {
            "type": "string",
            "description": "Stable caller ID; reuse for retries of the same operation.",
        },
    },
    "required": ["action", "chat_id", "request_id"],
}

TELEGRAM_OPERATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "method": {"type": "string", "enum": ["editMessageText", "setMessageReaction"]},
        "chat_id": {"type": "string", "description": "Exact numeric Telegram chat ID."},
        "message_id": {"type": "integer", "description": "Exact provider message ID."},
        "text": {"type": "string", "description": "Replacement text for editMessageText."},
        "parse_mode": {
            "type": "string",
            "enum": ["HTML", "MarkdownV2"],
            "description": "Optional text formatting for edits; omit for plain text.",
        },
        "topic_id": {
            "type": "integer",
            "description": "Known forum topic of the target. Checked against this topic's place and kept local.",
        },
        "reaction": {
            "type": "array",
            "maxItems": 1,
            "items": {"type": "object", "additionalProperties": True},
            "description": "At most one Telegram reaction object; [] removes this bot's reaction.",
        },
        "is_big": {"type": "boolean", "default": False},
        "original_delivery_id": {"type": "string", "description": "Optional prior durable send identity."},
        "request_id": {
            "type": "string",
            "description": "Stable operation identity; reuse only for the same logical mutation.",
        },
    },
    "required": ["method", "chat_id", "message_id", "request_id"],
}

TELEGRAM_REACT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "action": {
            "type": "string",
            "enum": ["set", "clear"],
            "description": "set puts this bot's one reaction on the message, replacing its earlier one; clear removes it.",
        },
        "chat_id": {"type": "string", "description": "Exact numeric Telegram chat ID."},
        "message_id": {"type": "integer", "description": "Exact provider message ID of the message reacted to."},
        "topic_id": {
            "type": "integer",
            "description": "Known forum topic of the target, as in the event; never sent to Telegram.",
        },
        "emoji": {"type": "string", "description": "For set: one emoji Telegram offers for reactions."},
        "custom_emoji_id": {"type": "string", "description": "For set, instead of emoji: a custom emoji ID."},
        "is_big": {"type": "boolean", "default": False},
        "request_id": {
            "type": "string",
            "description": "Stable operation identity; its receipt is telegram_receipt(operation=operation).",
        },
    },
    "required": ["action", "chat_id", "message_id", "request_id"],
}

TELEGRAM_RECEIPT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "request_id": {"type": "string"},
        "operation": {"type": "string", "enum": ["send", "moderate", "operation"]},
    },
    "required": ["request_id"],
}


def _turn_account(ctx: Any) -> str:
    """Return the originating Telegram bot account_id from a Presence turn's task_metadata, else ``""``."""
    metadata = getattr(ctx, "task_metadata", None) or {}
    if not isinstance(metadata, dict):
        return ""
    event = (metadata.get("presence") or {}).get("event") or {}
    if not isinstance(event, dict) or event.get("provider") != "telegram":
        return ""
    return str(event.get("account_id") or "").strip()


def _resolve_tool_account(ctx: Any, confirmed_account: Any) -> str:
    """Bind a tool call to its Presence turn's originating account, else the active confirmed bot."""
    return _turn_account(ctx) or str(confirmed_account() or "").strip()


def _positive_id(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    parsed = int(value)
    if parsed <= 0 or str(parsed) != str(value).strip():
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def make_send_tool(api: Any, confirmed_account=lambda: "", place_binder=lambda *_a, **_k: None):
    def telegram_send(
        ctx=None,
        *,
        chat_id: str = "",
        text: str = "",
        kind: str = "message",
        file_path: str = "",
        caption: str = "",
        topic_id: str = "",
        reply_to_message_id: str = "",
        request_id: str = "",
        markdown: bool = True,
        items: Optional[list] = None,
    ) -> Dict[str, Any]:
        target = str(chat_id or "").strip()
        selected_kind = str(kind or "message").strip()
        if re.fullmatch(r"-?[0-9]+", target) is None:
            return {"ok": False, "error": "chat_id must be an exact numeric Telegram id"}
        if selected_kind not in SEND_KINDS:
            return {"ok": False, "error": "kind must be " + ", ".join(SEND_KINDS)}
        if not isinstance(markdown, bool):
            return {"ok": False, "error": "markdown must be a boolean"}
        account = _resolve_tool_account(ctx, confirmed_account)
        if not account:
            return {
                "ok": False,
                "code": "bot_unconfirmed",
                "error": "Telegram bot identity is unconfirmed for the current token. Nothing was queued.",
            }
        scope = caller_scope(ctx)
        payload: Dict[str, Any] = {
            "kind": selected_kind,
            "chat_id": target,
            "markdown": markdown,
            "account_id": account,
            "_reporting": {"version": -1, "origin": tool_origin(ctx)},
            **({"_caller_scope": scope} if scope else {}),
        }
        try:
            if topic_id not in (None, ""):
                payload["topic_id"] = str(_positive_id(topic_id, "topic_id"))
            if reply_to_message_id not in (None, ""):
                payload["reply_to_message_id"] = str(_positive_id(reply_to_message_id, "reply_to_message_id"))
        except (TypeError, ValueError) as exc:
            return {"ok": False, "error": str(exc)}
        state_dir = pathlib.Path(api.get_state_dir())
        if payload.get("reply_to_message_id"):
            store_chk = CustodyStore.shared(state_dir / "custody.sqlite3")
            reply_mid = int(payload["reply_to_message_id"])
            known_reply = store_chk.reaction_target(account, target, reply_mid)
            given_topic = str(payload.get("topic_id") or "")
            if known_reply.get("retained"):
                if given_topic != str(known_reply.get("topic_id") or ""):
                    where = f"topic {known_reply['topic_id']}" if known_reply["topic_id"] else "this chat's General conversation"
                    return {
                        "ok": False,
                        "code": "forum_topic_mismatch",
                        "error": f"reply_to_message_id {reply_mid} is in {where}; pass matching topic_id. Not queued.",
                    }
            elif known_reply.get("forum") == 1 or (known_reply.get("forum") is None and target.startswith("-")):
                return {
                    "ok": False,
                    "code": "reply_target_unconfirmed",
                    "error": (
                        f"reply_to_message_id {reply_mid} in chat {target} is not retained here, so its topic cannot "
                        "be verified from a confirmed source. Not queued."
                    ),
                }
        try:
            if selected_kind == "message":
                if not str(text or "").strip():
                    return {"ok": False, "error": "text is required for a message"}
                if str(caption or "").strip() or str(file_path or "").strip() or items is not None:
                    raise SendRefused("unexpected_media_args", "kind=message accepts only text, not caption, file_path or items. Not queued.")
                payload["text"] = str(text)
                payload["_rendered_chunks"] = prepare_text(str(text), markdown=markdown)
            elif selected_kind == "album":
                if str(text or "").strip():
                    raise SendRefused("use_caption", "Pass album captions in caption or items[].caption, not text. Not queued.")
                payload["items"], payload["_rendered_items"], payload["_frozen"] = album_items(
                    ctx, items, str(caption or ""), markdown, state_dir
                )
            else:
                if str(text or "").strip():
                    raise SendRefused("use_caption", "Pass media captions in caption, not text. Not queued.")
                payload["caption"] = str(caption or "")
                payload["_rendered_caption"] = frozen_caption(payload["caption"], markdown)
                kept = open_and_freeze_file(ctx, selected_kind, file_path, state_dir)
                payload["file_path"], payload["_frozen"] = kept["path"], [kept]
        except SendRefused as refusal:
            return refusal.answer()
        try:
            binding = place_binder(account, target, str(payload.get("topic_id") or ""))
        except BaseException as exc:
            discard_files(payload.get("_frozen") or [], state_dir)
            if isinstance(exc, PlaceBindingUnavailable) or getattr(exc, "place_binding_unavailable", False):
                return {"ok": False, "code": str(getattr(exc, "code", "place_binding_unavailable")), "error": str(exc)}
            raise
        if isinstance(binding, dict):
            payload["_place_binding"] = binding
        receipt = str(request_id or uuid.uuid4().hex).strip()
        delivery_id, store = f"telegram-send:{receipt}", None
        fp = idempotency_fingerprint(payload)
        try:
            store = CustodyStore.shared(state_dir / "custody.sqlite3")
            outcome = store.enqueue_outbox_checked(delivery_id, payload, fingerprint=fp)
        except BaseException:
            discard_unqueued(payload.get("_frozen") or [], state_dir, store, delivery_id)
            raise
        if outcome == "conflict":
            discard_files(payload.get("_frozen") or [], state_dir)
            return {
                "ok": False,
                "code": "request_id_conflict",
                "error": "request_id was already used for a different send or caller. Choose a new request_id.",
            }
        if outcome != "queued":
            discard_files(payload.get("_frozen") or [], state_dir)
        return {
            "ok": True,
            "state": outcome,
            "request_id": receipt,
            "chat_id": target,
            "kind": selected_kind,
        }

    return telegram_send


def make_typing_tool(api: Any, place_binder=lambda *_a, **_k: None):
    def telegram_typing(ctx: Any = None, *, chat_id: str, topic_id: int | None = None) -> dict[str, Any]:
        metadata = getattr(ctx, "task_metadata", None) or {}
        event = (metadata.get("presence") or {}).get("event") or {} if isinstance(metadata, dict) else {}
        turn = str(getattr(ctx, "task_id", "") or "").strip()
        own_chat, own_topic = str(event.get("conversation_id") or ""), str(event.get("thread_id") or "")
        own_account = str(event.get("account_id") or "").strip()
        if event.get("provider") != "telegram" or not turn or not own_chat or not own_account:
            return {
                "ok": False,
                "code": "no_turn",
                "error": "typing shows only during a Telegram Presence turn, in its own chat. Nothing was shown.",
            }
        try:
            given_topic = "" if topic_id in (None, "") else str(_positive_id(topic_id, "topic_id"))
        except (TypeError, ValueError) as exc:
            return {"ok": False, "code": "invalid_topic_id", "error": str(exc)}
        if str(chat_id or "").strip() != own_chat or given_topic != own_topic:
            where = f"chat {own_chat}" + (f", topic {own_topic}" if own_topic else "")
            return {
                "ok": False,
                "code": "other_conversation",
                "error": f"typing shows only in this turn's {where}. Nothing was shown.",
            }
        store = CustodyStore.shared(pathlib.Path(api.get_state_dir()) / "custody.sqlite3")
        source = str(event.get("source_event_id") or "")
        if not store.turn_live(source, turn):
            return {"ok": False, "code": "turn_ended", "error": "this turn is no longer live here. Nothing was shown."}
        try:
            binding = place_binder(own_account, own_chat, own_topic)
        except Exception as exc:
            if not isinstance(exc, PlaceBindingUnavailable) and not getattr(exc, "place_binding_unavailable", False):
                raise
            return {"ok": False, "code": str(getattr(exc, "code", "place_binding_unavailable")), "error": str(exc)}
        store.request_typing(
            chat_id=own_chat, topic_id=own_topic, turn_ref=turn, source_event_id=source, account_id=own_account,
            place_binding=binding if isinstance(binding, dict) else None,
        )
        return {
            "ok": True,
            "typing": "requested",
            "chat_id": own_chat,
            **({"topic_id": int(own_topic)} if own_topic else {}),
            "note": (
                "Nothing was sent yet. The transport shows it while this turn runs and this place may act, until "
                "your next message here is delivered, the turn's end or about 2 minutes. It is not speech."
            ),
        }

    return telegram_typing


def make_moderation_tool(api: Any, confirmed_account=lambda: "", place_binder=lambda *_a, **_k: None):
    def telegram_moderate(
        ctx: Any = None,
        *,
        action: str,
        chat_id: str,
        request_id: str,
        message_id: int = 0,
        topic_id: int | None = None,
        user_id: int = 0,
        permissions: dict[str, bool] | None = None,
        until_date: int = 0,
        revoke_messages: bool = False,
    ) -> dict[str, Any]:
        store = CustodyStore.shared(pathlib.Path(api.get_state_dir()) / "custody.sqlite3")
        account = _resolve_tool_account(ctx, confirmed_account)
        if not account:
            return {
                "ok": False,
                "code": "bot_unconfirmed",
                "error": "Telegram bot identity is unconfirmed for the current token. Nothing was queued.",
            }
        local_topic_id: Any = None
        try:
            target = str(chat_id).strip()
            if re.fullmatch(r"-?[0-9]+", target) is None:
                raise ValueError("chat_id must be an exact numeric Telegram id")
            if not str(request_id).strip():
                raise ValueError("request_id is required for a retryable operation")
            local_topic_id = _positive_id(topic_id, "topic_id") if topic_id is not None else None
            parameters: dict[str, Any] = {"chat_id": target}
            if action == "delete_message":
                mid = _positive_id(message_id, "message_id")
                parameters["message_id"] = mid
                known = store.reaction_target(account, target, mid)
                local_topic_id, _ = _reaction_conversation(known, target, mid, local_topic_id)
            elif action in {"restrict_member", "ban_member", "unban_member"}:
                if local_topic_id is not None:
                    raise ValueError(
                        "topic_id is not allowed for group-wide member moderation; membership actions apply to the "
                        "whole group and must be authorized at group level"
                    )
                parameters["user_id"] = _positive_id(user_id, "user_id")
                if action == "restrict_member":
                    if (
                        not isinstance(permissions, dict)
                        or not permissions
                        or any(type(value) is not bool for value in permissions.values())
                    ):
                        raise ValueError("permissions must be a nonempty Telegram ChatPermissions object of booleans")
                    parameters.update(permissions=dict(permissions), use_independent_chat_permissions=True)
                if action in {"restrict_member", "ban_member"}:
                    if isinstance(until_date, bool) or int(until_date) < 0:
                        raise ValueError("until_date must be a nonnegative Unix timestamp")
                    parameters["until_date"] = int(until_date)
                if action == "ban_member":
                    parameters["revoke_messages"] = bool(revoke_messages)
                if action == "unban_member":
                    parameters["only_if_banned"] = True
            else:
                raise ValueError("unsupported Telegram moderation action")
        except (TypeError, ValueError) as exc:
            return {"ok": False, "error": str(exc)}
        receipt = str(request_id).strip()
        scope = caller_scope(ctx)
        payload: Dict[str, Any] = {
            "kind": "moderation",
            "chat_id": target,
            "action": action,
            "parameters": parameters,
            "account_id": account,
            "_reporting": {"version": 0, "origin": tool_origin(ctx)},
            **({"_caller_scope": scope} if scope else {}),
            **({"topic_id": str(local_topic_id)} if local_topic_id not in (None, "") else {}),
        }
        try:
            binding = place_binder(account, target, str(payload.get("topic_id") or ""))
        except Exception as exc:
            if not isinstance(exc, PlaceBindingUnavailable) and not getattr(exc, "place_binding_unavailable", False):
                raise
            return {"ok": False, "code": str(getattr(exc, "code", "place_binding_unavailable")), "error": str(exc)}
        if isinstance(binding, dict):
            payload["_place_binding"] = binding
        outcome = store.enqueue_outbox_checked(
            f"telegram-moderate:{receipt}", payload, fingerprint=idempotency_fingerprint(payload)
        )
        if outcome == "conflict":
            return {
                "ok": False,
                "code": "request_id_conflict",
                "error": "request_id was already used for a different moderation operation or caller.",
            }
        return {
            "ok": True,
            "state": outcome,
            "request_id": receipt,
            "action": action,
        }

    return telegram_moderate


def make_receipt_tool(api: Any):
    def telegram_receipt(ctx: Any = None, *, request_id: str, operation: str = "send") -> dict[str, Any]:
        if operation not in {"send", "moderate", "operation"}:
            return {"ok": False, "error": "operation must be send, moderate, or operation"}
        delivery_id = f"telegram-{operation}:{str(request_id).strip()}"
        store = CustodyStore.shared(pathlib.Path(api.get_state_dir()) / "custody.sqlite3")
        payload = store.outbox_payload(delivery_id)
        if not payload or not receipt_visible_to(ctx, payload):
            return {"state": "not_found"}
        return sanitize_tool_receipt(store.delivery_receipt(delivery_id))

    return telegram_receipt


def _reaction_values_or_refuse(reaction: Any) -> list[dict[str, Any]]:
    if not isinstance(reaction, list) or any(not isinstance(item, dict) for item in reaction):
        raise ValueError("reaction must be a list of Telegram reaction objects")
    if len(reaction) > 1:
        raise ValueError("a Telegram bot can set at most one reaction per message; [] removes it")
    for item in reaction:
        if item.get("type") == "paid":
            raise ValueError("Telegram does not let bots set paid reactions")
        if not (item.get("type") == "emoji" and item.get("emoji")) and not (
            item.get("type") == "custom_emoji" and item.get("custom_emoji_id")
        ):
            raise ValueError("a reaction is {type: emoji, emoji} or {type: custom_emoji, custom_emoji_id}")
    return [dict(item) for item in reaction]


def _reaction_conversation(
    known: dict[str, Any], chat_id: str, message_id: int, topic_id: int | None
) -> tuple[str, str]:
    given = str(topic_id) if topic_id is not None else ""
    private = not chat_id.startswith("-")
    if known.get("unconfirmed"):
        if given or private:
            return given, "unresolved"
        raise ValueError(
            f"forum_topic_unknown: this bot's identity is unconfirmed for the current token, so its "
            f"records of chat {chat_id} cannot be read and the topic of message {message_id} is unknown; "
            "pass the topic_id from the event if you have it, or retry once the transport status shows the bot. Not sent."
        )
    if known["retained"]:
        if given != known["topic_id"]:
            where = f"topic {known['topic_id']}" if known["topic_id"] else "this chat's General conversation"
            raise ValueError(
                f"forum_topic_mismatch: message {message_id} is in {where}; "
                f"pass topic_id={known['topic_id'] or 'nothing'}"
            )
        return given, "resolved"
    if known["forum"] == 0 or (known["forum"] is None and private):
        if given:
            raise ValueError("this chat has no topics; omit topic_id")
        return "", "resolved"
    what = (
        "has shown topics and message"
        if private
        else "is a forum and message"
        if known["forum"] == 1
        else "may be a forum (not observed yet) and message"
    )
    raise ValueError(
        f"forum_topic_unknown: chat {chat_id} {what} {message_id} is not retained here, so its "
        "topic cannot be verified from a confirmed source. Not sent."
    )


def make_operation_tool(api: Any, confirmed_account=lambda: "", place_binder=lambda *_a, **_k: None):
    def telegram_operation(
        ctx: Any = None,
        *,
        method: str,
        chat_id: str,
        message_id: int,
        request_id: str,
        text: str = "",
        parse_mode: str = "",
        topic_id: int | None = None,
        reaction: list[dict[str, Any]] | None = None,
        is_big: bool = False,
        original_delivery_id: str = "",
    ) -> dict[str, Any]:
        try:
            if method not in {"editMessageText", "setMessageReaction"}:
                raise ValueError("unsupported Telegram operation")
            target = str(chat_id).strip()
            if re.fullmatch(r"-?[0-9]+", target) is None:
                raise ValueError("chat_id must be an exact numeric Telegram id")
            message = _positive_id(message_id, "message_id")
            receipt = str(request_id or "").strip()
            if not receipt:
                raise ValueError("request_id is required for a retryable operation")
            parameters: dict[str, Any] = {"chat_id": target, "message_id": message}
            local_topic_id = _positive_id(topic_id, "topic_id") if topic_id is not None else None
            store = CustodyStore.shared(pathlib.Path(api.get_state_dir()) / "custody.sqlite3")
            account = _resolve_tool_account(ctx, confirmed_account)
            if not account:
                return {
                    "ok": False,
                    "code": "bot_unconfirmed",
                    "error": "Telegram bot identity is unconfirmed for the current token. Nothing was queued.",
                }
            reaction_target, album = None, ""
            known = store.reaction_target(account, target, message)
            if method == "editMessageText":
                if not str(text or "").strip():
                    raise ValueError("text is required for editMessageText")
                if not known.get("retained") or not known.get("is_self"):
                    raise ValueError("editMessageText can only edit this bot's own messages (retained records only)")
                local_topic_id, _ = _reaction_conversation(known, target, message, local_topic_id)
                parameters["text"] = str(text)
                if parse_mode:
                    parameters["parse_mode"] = str(parse_mode)
            else:
                parameters["reaction"] = _reaction_values_or_refuse(reaction)
                parameters["is_big"] = bool(is_big)
                local_topic_id, resolution = _reaction_conversation(known, target, message, local_topic_id)
                album = known.get("media_group_id", "")
                if album and (int(known.get("album_earlier") or 0) > 0 or not known.get("album_origin_known")):
                    raise ValueError(
                        "album_target_unconfirmed: Telegram applies reactions on a non-first album item to the "
                        "album's first undeleted message, and this album's first item is not confirmed here. Not sent."
                    )
                reaction_target = {"message_id": str(message), "resolution": resolution}
        except (TypeError, ValueError) as exc:
            return {"ok": False, "error": str(exc)}
        origin = tool_origin(ctx)
        scope = caller_scope(ctx)
        payload: Dict[str, Any] = {
            "kind": "operation",
            "chat_id": target,
            "method": method,
            "parameters": parameters,
            "text": str(text or ""),
            "account_id": account,
            "_reporting": {"version": -1, "origin": origin},
            **({"_caller_scope": scope} if scope else {}),
        }
        if reaction_target:
            payload["_reaction_target"] = reaction_target
        if album:
            payload["_reaction_album"] = album
        if original_delivery_id:
            payload["original_delivery_id"] = str(original_delivery_id)
        if local_topic_id not in (None, ""):
            payload["topic_id"] = str(local_topic_id)
        try:
            binding = place_binder(account, target, str(payload.get("topic_id") or ""))
        except Exception as exc:
            if not isinstance(exc, PlaceBindingUnavailable) and not getattr(exc, "place_binding_unavailable", False):
                raise
            return {"ok": False, "code": str(getattr(exc, "code", "place_binding_unavailable")), "error": str(exc)}
        if isinstance(binding, dict):
            payload["_place_binding"] = binding
        outcome = store.enqueue_outbox_checked(
            f"telegram-operation:{receipt}", payload, fingerprint=idempotency_fingerprint(payload)
        )
        if outcome == "conflict":
            return {
                "ok": False,
                "code": "request_id_conflict",
                "error": "request_id was already used for a different operation or caller. Choose a new request_id.",
            }
        return {
            "ok": True,
            "state": outcome,
            "request_id": receipt,
            "method": method,
            "chat_id": target,
            "message_id": message,
        }

    return telegram_operation


def make_react_tool(operation: Any):
    def telegram_react(
        ctx: Any = None,
        *,
        action: str,
        chat_id: str,
        message_id: int,
        request_id: str,
        topic_id: int | None = None,
        emoji: str = "",
        custom_emoji_id: str = "",
        is_big: bool = False,
    ) -> dict[str, Any]:
        emoji, custom = str(emoji or "").strip(), str(custom_emoji_id or "").strip()
        if action == "set":
            if bool(emoji) == bool(custom):
                return {"ok": False, "error": "set takes exactly one of emoji or custom_emoji_id"}
            reaction = [{"type": "emoji", "emoji": emoji} if emoji else {"type": "custom_emoji", "custom_emoji_id": custom}]
        elif action == "clear":
            if emoji or custom:
                return {"ok": False, "error": "clear removes this bot's reaction; omit emoji and custom_emoji_id"}
            reaction = []
        else:
            return {"ok": False, "error": "action must be set or clear"}
        return operation(
            ctx,
            method="setMessageReaction",
            chat_id=chat_id,
            message_id=message_id,
            request_id=request_id,
            topic_id=topic_id,
            reaction=reaction,
            is_big=is_big,
        )

    return telegram_react


def _json_tool(fn: Any) -> Any:
    """Serialize dict tool results to JSON text while preserving ``fn``'s signature for ctx inspection."""
    @functools.wraps(fn)
    def _wrapped(*args: Any, **kwargs: Any) -> str:
        result = fn(*args, **kwargs)
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)

    return _wrapped


def register_presence_tools(api: Any, runtime: Any) -> None:
    """Register the 6 Presence Telegram tools when the host API supports ``register_tool``."""
    register_tool = getattr(api, "register_tool", None)
    if not callable(register_tool):
        return
    place_binder = getattr(runtime, "bind_place_sync", lambda *_a, **_k: None)
    register_tool(
        "telegram_send",
        _json_tool(make_send_tool(api, runtime.recorded_bot, place_binder)),
        description=(
            "Durably queue a proactive Telegram send for an exact numeric chat, with optional topic and reply ids: "
            "text, or a file as photo, document, video, audio, voice (OGG/Opus, MP3 or M4A), animation (GIF or "
            "silent MP4) or album (2-10 items: photos with videos, or only audio, or only documents). A bot uploads "
            "at most 50 MB per file and 10 MB per photo; a larger file or a caption over 1024 characters is refused, "
            "never cut. A file is copied when queued, and that copy is what Telegram receives. Standard Markdown is "
            "rendered by default; set markdown=false for literal text. An event.message.attachments record with "
            "status too_large (a bot downloads at most 20 MiB) or failed keeps file_id, file_name, file_size and a "
            "factual note; you may ask the sender for a download link scoped to the same audience (Yandex.Disk "
            "cloud-api public download endpoint, Google Drive confirm token over 100 MB, Dropbox dl=1)."
        ),
        schema=TELEGRAM_SEND_SCHEMA,
    )
    register_tool(
        "telegram_moderate",
        _json_tool(make_moderation_tool(api, runtime.recorded_bot, place_binder)),
        description=(
            "Queue a Telegram delete/restrict/ban/unban operation with exact provider IDs. "
            "Reuse request_id when retrying; queued is not delivered."
        ),
        schema=MODERATION_SCHEMA,
    )
    operation = make_operation_tool(api, runtime.recorded_bot, place_binder)
    register_tool(
        "telegram_operation",
        _json_tool(operation),
        description=(
            "Queue an edit of this bot's own message, or set or clear this bot's reaction on any reachable message, "
            "through the Presence outbox. Use exact numeric chat/message IDs and a new request_id; in a forum pass "
            "the target's known topic_id."
        ),
        schema=TELEGRAM_OPERATION_SCHEMA,
    )
    register_tool(
        "telegram_react",
        _json_tool(make_react_tool(operation)),
        description=(
            "Set or clear this bot's one emoji reaction on an exact Telegram message. A reaction is a gesture, not "
            "speech or proof of finished work. Queued durably as setMessageReaction; check outcome with "
            "telegram_receipt(operation=\"operation\", request_id=...)."
        ),
        schema=TELEGRAM_REACT_SCHEMA,
    )
    register_tool(
        "telegram_typing",
        _json_tool(make_typing_tool(api, place_binder)),
        description=(
            "Ask for \"typing…\" in this turn's own Telegram chat while preparing a reply. Not speech, no receipt; "
            "ends on next delivered message, turn end, or ~2 minutes. Pass the event's topic_id in a forum topic."
        ),
        schema=TELEGRAM_TYPING_SCHEMA,
    )
    register_tool(
        "telegram_receipt",
        _json_tool(make_receipt_tool(api)),
        description="Read durable delivery state, provider receipt or error for a Telegram send/moderation/operation request.",
        schema=TELEGRAM_RECEIPT_SCHEMA,
    )
