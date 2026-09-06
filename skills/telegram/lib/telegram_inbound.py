"""Inbound Telegram files → the host's shared chat-attachment path (#668).

A document, video, audio, voice or video-note message is downloaded (within
this integration's 10 MiB download cap), parked in this skill's own state directory
and relayed through ``/chat/inject`` ``attachments``. The host copies it into
the same ``data/uploads`` store the browser paperclip uses and stages it for
the task like any other chat attachment; the parked copy is removed once the
host has answered. Nothing here transcribes or interprets the file — the task
reads it through the ordinary attachment manifest.
"""

from __future__ import annotations

import mimetypes
import pathlib
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .telegram_api import _MAX_TELEGRAM_DOWNLOAD_BYTES

_FILE_KINDS = ("document", "video", "audio", "voice", "video_note")
_DEFAULT_MIME = {"voice": "audio/ogg", "video_note": "video/mp4", "video": "video/mp4",
                 "audio": "audio/mpeg", "document": "application/octet-stream"}
# Telegram's own container per kind; ``mimetypes`` differs per OS (.oga/.ogg).
_DEFAULT_EXT = {"voice": ".ogg", "video_note": ".mp4", "video": ".mp4", "audio": ".mp3", "document": ""}
_UNSUPPORTED_KINDS = ("sticker", "animation", "location", "contact", "poll", "venue", "dice", "game")

_TEXTS = {
    "en": {
        "too_large": "This file is {size} MiB; this integration accepts files up to 10 MiB. Send a smaller file or a link.",
        "unsupported": "This kind of message isn't supported — send text, a photo, or a file (document, video, audio, voice).",
    },
    "ru": {
        "too_large": "Файл весит {size} МиБ; эта интеграция принимает файлы не больше 10 МиБ. Пришлите файл поменьше или ссылку.",
        "unsupported": "Такой тип сообщения не поддерживается — пришлите текст, фото или файл (документ, видео, аудио, голосовое).",
    },
}


def _texts(lang: str) -> Dict[str, str]:
    return _TEXTS["ru" if lang == "ru" else "en"]


def inbound_file(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Describe the ONE downloadable file of a message, or ``None``.

    ``{"kind", "file_id", "name", "mime", "size", "refusal"}`` — ``refusal`` is
    ``"too_large"`` when the announced size exceeds the download cap, so the
    caller tells the owner instead of pretending to accept the file.
    """
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
        try:
            size = int(media.get("file_size") or 0)
        except (TypeError, ValueError):
            size = 0
        return {
            "kind": kind, "file_id": file_id, "name": name, "mime": mime, "size": size,
            "refusal": "too_large" if size > _MAX_TELEGRAM_DOWNLOAD_BYTES else "",
        }
    return None


def unsupported_kind(message: Dict[str, Any]) -> bool:
    return any(message.get(kind) for kind in _UNSUPPORTED_KINDS)


def unsupported_text(lang: str) -> str:
    return _texts(lang)["unsupported"]


def refusal_text(info: Dict[str, Any], lang: str) -> str:
    size_mb = f"{int(info.get('size') or 0) / (1024 * 1024):.1f}"
    return _texts(lang)["too_large"].format(size=size_mb)


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


async def park_inbound_file(api, client, info: Dict[str, Any]) -> ParkedFile:
    """Download ``info`` and park it under ``<state_dir>/inbox``; the spec is the
    ``/chat/inject`` ``attachments`` entry (path confined to this skill's state)."""
    content = await client.download_file(info["file_id"])
    inbox = pathlib.Path(api.get_state_dir()) / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    # Display names are transport metadata, never native filesystem paths.
    path = inbox / uuid.uuid4().hex
    path.write_bytes(content)
    return ParkedFile(path, {"path": str(path), "name": str(info["name"]), "mime": str(info.get("mime") or "")})
