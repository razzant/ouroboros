"""Inbound Telegram media → the host's shared chat-attachment path (#668).

A photo rides inline (``image_base64``). A document, video, audio, voice or
video-note message is downloaded (within this integration's 10 MiB download
cap), parked in this skill's own state directory and relayed through
``/chat/inject`` ``attachments``. The host copies it into the same
``data/uploads`` store the browser paperclip uses and stages it for the task
like any other chat attachment; the parked copy is removed once the host has
answered. Media the bridge could not receive (over the cap, or a failed
download) travels as a pathless fact beside the sender's own caption:
``{name, mime, size?, unavailable: "too_large"|"download_failed", limit_bytes?}``
(``chat_uploads.delivery_ref``). The host records it on the same accepted row
with no bytes and no words, and the task reads its reason. Nothing here
transcribes or interprets the file.
"""

from __future__ import annotations

import base64
import mimetypes
import pathlib
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from .telegram_api import TelegramBindingChanged, TelegramDownloadTooLarge, _MAX_TELEGRAM_DOWNLOAD_BYTES
from .telegram_i18n import Index

_FILE_KINDS = ("document", "video", "audio", "voice", "video_note")
_DEFAULT_MIME = {"voice": "audio/ogg", "video_note": "video/mp4", "video": "video/mp4",
                 "audio": "audio/mpeg", "document": "application/octet-stream"}
# Telegram's own container per kind; ``mimetypes`` differs per OS (.oga/.ogg).
_DEFAULT_EXT = {"voice": ".ogg", "video_note": ".mp4", "video": ".mp4", "audio": ".mp3", "document": ""}
_UNSUPPORTED_KINDS = ("sticker", "animation", "location", "contact", "poll", "venue", "dice", "game")

_TEXTS_EN = {
    "unsupported": "This kind of message isn't supported — send text, a photo, or a file (document, video, audio, voice).",
}
_TEXTS = Index("inbound", _TEXTS_EN, "refusals of files and message kinds the bridge cannot relay")


def _texts(lang: str) -> Mapping[str, str]:
    return _TEXTS[lang]


def _size(media: Dict[str, Any]) -> Optional[int]:
    """Telegram's announced ``file_size``; ``None`` when it announced none (0 is a true empty file)."""
    size = media.get("file_size")
    return size if type(size) is int and size >= 0 else None


def photo_media(photo: Any) -> Optional[Dict[str, Any]]:
    """The largest PhotoSize as one media item, or ``None`` without a file id."""
    file_id = str((photo or {}).get("file_id") or "").strip() if isinstance(photo, dict) else ""
    return {"kind": "photo", "file_id": file_id, "name": "photo.jpg", "mime": "image/jpeg",
            "size": _size(photo)} if file_id else None


def inbound_file(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Describe the ONE downloadable file of a message, or ``None``: ``{"kind", "file_id", "name", "mime", "size"}``."""
    for kind in _FILE_KINDS:
        media = message.get(kind)
        if not isinstance(media, dict):
            continue
        file_id = str(media.get("file_id") or "").strip()
        if not file_id:
            return None
        mime = str(media.get("mime_type") or "").strip() or _DEFAULT_MIME[kind]
        name = pathlib.PurePosixPath(str(media.get("file_name") or "").strip()).name
        if not name or name in {".", ".."}:
            ext = _DEFAULT_EXT[kind] if mime == _DEFAULT_MIME[kind] else (
                mimetypes.guess_extension(mime) or _DEFAULT_EXT[kind])
            name = f"{kind}_{file_id[-8:]}{ext}"
        return {"kind": kind, "file_id": file_id, "name": name, "mime": mime, "size": _size(media)}
    return None


def unsupported_kind(message: Dict[str, Any]) -> bool:
    return any(message.get(kind) for kind in _UNSUPPORTED_KINDS)


def unsupported_text(lang: str) -> str:
    return _texts(lang)["unsupported"]


@dataclass
class ParkedFile:
    """A downloaded file waiting in this skill's state dir for the host to copy."""

    path: pathlib.Path
    spec: Dict[str, str]

    def cleanup(self) -> None:
        try:
            self.path.unlink()
        except OSError:
            pass


def _unavailable(info: Dict[str, Any], reason: str) -> Dict[str, Any]:
    """The pathless fact of media that did not arrive: its name, type, announced size and why."""
    return {"name": info["name"], "mime": info["mime"], "unavailable": reason,
            **({"size": info["size"]} if info["size"] is not None else {}),
            **({"limit_bytes": _MAX_TELEGRAM_DOWNLOAD_BYTES} if reason == "too_large" else {})}


async def receive_media(api, client, info: Dict[str, Any], *, park_photo: bool = False,
                        ) -> tuple[Dict[str, Any], Optional[ParkedFile]]:
    """``/chat/inject`` fields for one media item: the inline photo, the parked file's spec, or the fact of why
    nothing was received. Only a failed fetch is such a fact: a changed token still ends the update, and a
    failure to park or hand over the received bytes stays a delivery error. Bytes count only when complete:
    a body shorter or longer than Telegram's announced size is a failed download; an announced 0 is a true
    empty file. ``park_photo``: a photo is parked as a file too, for ``/chat/decision``, which takes no inline image."""
    size = info["size"]
    fetch = client.download_photo if info["kind"] == "photo" else client.download_file
    try:
        if size is not None and size > _MAX_TELEGRAM_DOWNLOAD_BYTES:
            return {"attachments": [_unavailable(info, "too_large")]}, None
        received = await fetch(info["file_id"])
    except TelegramBindingChanged:
        raise
    except TelegramDownloadTooLarge:
        return {"attachments": [_unavailable(info, "too_large")]}, None
    except (RuntimeError, OSError):
        return {"attachments": [_unavailable(info, "download_failed")]}, None
    content = base64.b64decode(received[0]) if info["kind"] == "photo" else received
    if size is not None and len(content) != size:
        return {"attachments": [_unavailable(info, "download_failed")]}, None
    if info["kind"] == "photo" and not park_photo:
        return {"image_base64": received[0], "image_mime": received[1]}, None
    inbox = pathlib.Path(api.get_state_dir()) / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    # Display names are transport metadata, never native filesystem paths.
    path = inbox / uuid.uuid4().hex
    path.write_bytes(content)
    parked = ParkedFile(path, {"path": str(path), "name": str(info["name"]), "mime": str(info.get("mime") or "")})
    return {"attachments": [parked.spec]}, parked
