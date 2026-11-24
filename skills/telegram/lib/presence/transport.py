"""Presence Bot API transport sharing the bridge's token binding, proxy and Bot API server.

Never polls ``getUpdates`` and never calls ``deleteWebhook``: the single unified poller in
``plugin.py`` owns ``getUpdates``, and webhook conflicts stay surfaced as ``telegram_conflict``.
Every request checks the bridge's ``TokenBinding`` before bytes leave the machine and honors
``TELEGRAM_PROXY``, ``TELEGRAM_API_BASE`` and ``TELEGRAM_API_FILES_DIR``.
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import pathlib
import re
import stat
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

import httpx

from ...scripts.telegram_settings import (
    validate_telegram_api_base,
    validate_telegram_files_dir,
    validate_telegram_proxy,
)
from .formatting import _telegram_html_to_plain, prepare_caption, prepare_text

CLOUD_API_SERVER = "https://api.telegram.org"
CLOUD_DOWNLOAD_LIMIT = 20 * 1024 * 1024
CLOUD_UPLOAD_LIMIT = 50 * 1024 * 1024
CLOUD_PHOTO_UPLOAD_LIMIT = 10 * 1024 * 1024
MEDIA_METHODS = {
    "photo": "sendPhoto",
    "document": "sendDocument",
    "video": "sendVideo",
    "audio": "sendAudio",
    "voice": "sendVoice",
    "animation": "sendAnimation",
}
_FILE_TOO_BIG = re.compile(r"file is too big", re.IGNORECASE)
_FILE_ID_UNREADABLE = re.compile(r"invalid file[ _]?id|wrong remote file identifier|can'?t unserialize", re.IGNORECASE)
_UNSENT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout, httpx.ProxyError)


class TelegramApiError(RuntimeError):
    def __init__(
        self, endpoint: str, description: str, error_code: int = 0, retry_after: Optional[float] = None
    ):
        super().__init__(f"Telegram {endpoint} failed ({error_code or 'transport'}): {description}")
        self.endpoint = endpoint
        self.description = description
        self.error_code = int(error_code or 0)
        self.retry_after = retry_after


class TelegramFileTooLarge(TelegramApiError):
    """Telegram will never serve this file to a bot: it is above the Bot API download limit."""


class TelegramFileUnavailable(TelegramApiError):
    """Telegram cannot read this file_id at all; no later attempt can change that answer."""


class TelegramRequestNotSent(TelegramApiError):
    """No byte of the request left this machine, so Telegram cannot have acted on it."""


def _file_refusal(exc: TelegramApiError) -> Optional[TelegramApiError]:
    if not exc.error_code:
        return None
    for pattern, kind in ((_FILE_TOO_BIG, TelegramFileTooLarge), (_FILE_ID_UNREADABLE, TelegramFileUnavailable)):
        if pattern.search(exc.description):
            return kind(exc.endpoint, exc.description, exc.error_code)
    return None


def _split_text(text: str, limit: int) -> List[str]:
    if len(text) <= limit:
        return [text]
    remaining = text
    chunks = []
    while remaining:
        cut = min(limit, len(remaining))
        if cut < len(remaining):
            boundary = max(remaining.rfind("\n", 0, cut), remaining.rfind(" ", 0, cut))
            if boundary > limit // 2:
                cut = boundary
        chunks.append(remaining[:cut])
        remaining = remaining[cut:].lstrip()
    return chunks or [""]


class PresenceBotClient:
    """Async Bot API client for Presence sends, downloads, edits, reactions, moderation and typing."""

    def __init__(
        self,
        token: str,
        *,
        api_server: str = CLOUD_API_SERVER,
        proxy: Optional[str] = None,
        trust_env: bool = False,
        files_dir: Optional[str] = None,
        binding: Any = None,
        generation: Optional[int] = None,
        is_current: Optional[Callable[[], bool]] = None,
    ):
        self._binding = binding
        self._generation = int(generation or 0)
        self._is_current = is_current
        self._token = str(token or "").strip()
        self._api_server = validate_telegram_api_base(api_server).rstrip("/")
        self._proxy = validate_telegram_proxy(proxy)
        self._trust_env = bool(trust_env)
        self._files_dir = validate_telegram_files_dir(files_dir)

    @property
    def download_limit(self) -> int:
        return CLOUD_DOWNLOAD_LIMIT

    def is_current(self) -> bool:
        if not self._token:
            return False
        if self._binding is not None and not self._binding.current(self._generation):
            return False
        return self._is_current is None or bool(self._is_current())

    def _check_binding(self, endpoint: str) -> None:
        if not self._token:
            raise TelegramRequestNotSent(endpoint, "TELEGRAM_BOT_TOKEN is not configured")
        if self._binding is not None and not self._binding.current(self._generation):
            raise TelegramRequestNotSent(endpoint, "bot token changed before send")
        if self._is_current is not None and not self._is_current():
            raise TelegramRequestNotSent(endpoint, "bot identity or server binding changed before send")

    async def get_me(self) -> Dict[str, Any]:
        return await self._post_json("getMe", {})

    async def get_file(self, file_id: str) -> Dict[str, Any]:
        result = await self._post_json("getFile", {"file_id": str(file_id)})
        if not isinstance(result, dict) or not result.get("file_path"):
            raise TelegramApiError("getFile", "response has no file_path")
        return result

    async def download_file(
        self, file_id: str, destination: pathlib.Path, *, max_bytes: Optional[int] = None
    ) -> pathlib.Path:
        try:
            metadata = await self.get_file(file_id)
        except TelegramApiError as exc:
            final = _file_refusal(exc)
            if final is not None:
                raise final from exc
            raise
        try:
            declared_size = int(metadata.get("file_size") or 0)
        except (TypeError, ValueError, OverflowError):
            declared_size = 0
        eff_limit = self.download_limit if max_bytes is None else min(self.download_limit, max(0, int(max_bytes)))
        if declared_size > eff_limit:
            raise TelegramFileTooLarge("getFile", f"file exceeds {eff_limit} bytes")
        file_path = str(metadata["file_path"])
        data = await self._download_bytes(file_path, declared_size, max_bytes=eff_limit)
        destination = pathlib.Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_bytes(data)
        temporary.replace(destination)
        return destination

    async def _download_bytes(self, file_path: str, declared_size: int, *, max_bytes: Optional[int] = None) -> bytes:
        self._check_binding("download")
        eff_limit = self.download_limit if max_bytes is None else min(self.download_limit, max(0, int(max_bytes)))
        clean_path = file_path.lstrip("/")
        url = f"{self._api_server}/file/bot{self._token}/{clean_path}"
        try:
            async with httpx.AsyncClient(
                timeout=35.0,
                trust_env=self._trust_env,
                proxy=self._proxy,
                follow_redirects=False,
            ) as client:
                self._check_binding("download")
                async with client.stream("GET", url) as response:
                    if response.status_code != 200:
                        raise TelegramApiError("download", f"HTTP {response.status_code}", response.status_code)
                    announced = _announced_length(response.headers)
                    if announced is not None and announced > eff_limit:
                        raise TelegramFileTooLarge("download", f"file exceeds {eff_limit} bytes")
                    content = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(content) + len(chunk) > eff_limit:
                            raise TelegramFileTooLarge("download", f"file exceeds {eff_limit} bytes")
                        content.extend(chunk)
                    if announced is not None and len(content) < announced:
                        raise TelegramApiError("download", "incomplete file response")
            data = bytes(content)
        except TelegramFileTooLarge:
            raise
        except TelegramRequestNotSent:
            raise
        except TelegramApiError as exc:
            if not self._files_dir:
                raise
            self._check_binding("download")
            data = await asyncio.to_thread(self._read_local_file, file_path, exc, eff_limit)
        except _UNSENT_ERRORS as exc:
            raise TelegramRequestNotSent("download", type(exc).__name__) from None
        except httpx.HTTPError as exc:
            raise TelegramApiError("download", type(exc).__name__) from None
        if declared_size > 0 and len(data) != declared_size:
            raise TelegramApiError("download", f"received {len(data)} bytes, getFile said {declared_size}")
        return data

    def _read_local_file(self, file_path: str, cause: Exception, max_bytes: Optional[int] = None) -> bytes:
        """Confined read for a self-hosted Bot API server started with ``--local``."""
        eff_limit = self.download_limit if max_bytes is None else min(self.download_limit, max(0, int(max_bytes)))
        configured = pathlib.Path(str(self._files_dir))
        store = configured.resolve()
        raw = pathlib.Path(str(file_path or ""))
        candidate = raw if raw.is_absolute() else store / raw
        if not (candidate.is_relative_to(configured) or candidate.is_relative_to(store)):
            raise cause from None
        resolved = candidate.resolve()
        if not resolved.is_relative_to(store):
            raise cause from None
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
        try:
            handle = os.fdopen(os.open(resolved, flags), "rb")
        except OSError:
            raise cause from None
        with handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise cause from None
            data = handle.read(eff_limit + 1)
        if len(data) > eff_limit:
            raise TelegramFileTooLarge("download", f"file exceeds {eff_limit} bytes") from cause
        return data

    async def send_message(
        self,
        chat_id: str,
        text: str,
        *,
        markdown: bool = True,
        topic_id: Optional[int] = None,
        reply_to_message_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        receipts = []
        for index, chunk in enumerate(prepare_text(str(text), markdown=markdown)):
            receipts.append(
                await self.send_text_chunk(
                    chat_id,
                    chunk["text"],
                    parse_mode=chunk["parse_mode"],
                    topic_id=topic_id,
                    reply_to_message_id=reply_to_message_id if index == 0 else None,
                )
            )
        return receipts

    async def send_text_chunk(
        self,
        chat_id: str,
        text: str,
        *,
        parse_mode: str = "",
        topic_id: Optional[int] = None,
        reply_to_message_id: Optional[int] = None,
        before_send: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"chat_id": str(chat_id), "text": text}
        if topic_id is not None:
            payload["message_thread_id"] = int(topic_id)
        if reply_to_message_id is not None:
            payload["reply_parameters"] = {"message_id": int(reply_to_message_id)}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        try:
            result = await self._post_json("sendMessage", payload)
        except TelegramApiError as exc:
            if parse_mode != "HTML" or exc.error_code != 400:
                raise
            if before_send is not None:
                await before_send()
            payload.pop("parse_mode", None)
            payload["text"] = _telegram_html_to_plain(text)
            result = await self._post_json("sendMessage", payload)
        if not isinstance(result, dict):
            return {}
        fmt = "html" if payload.get("parse_mode") == "HTML" else "plain"
        return {**result, "_delivery": {"text": payload["text"], "format": fmt}}

    async def send_media(
        self,
        kind: str,
        chat_id: str,
        file_path: pathlib.Path,
        *,
        prepared_caption: Dict[str, str],
        topic_id: Optional[int] = None,
        reply_to_message_id: Optional[int] = None,
        before_send: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> Dict[str, Any]:
        endpoint = MEDIA_METHODS.get(kind)
        if endpoint is None:
            raise ValueError(f"unsupported Telegram media kind: {kind}")
        fields: Dict[str, str] = {"chat_id": str(chat_id)}
        caption = prepared_caption["text"]
        parse_mode = prepared_caption["parse_mode"]
        if caption:
            fields["caption"] = caption
            if parse_mode:
                fields["parse_mode"] = parse_mode
        if topic_id is not None:
            fields["message_thread_id"] = str(int(topic_id))
        if reply_to_message_id is not None:
            fields["reply_parameters"] = json.dumps({"message_id": int(reply_to_message_id)})

        try:
            result = await self._post_multipart(endpoint, fields, [(kind, pathlib.Path(file_path))], timeout=60.0)
        except TelegramApiError as exc:
            if not caption or parse_mode != "HTML" or exc.error_code != 400:
                raise
            if before_send is not None:
                await before_send()
            fields.pop("parse_mode", None)
            fields["caption"] = _telegram_html_to_plain(caption)
            result = await self._post_multipart(endpoint, fields, [(kind, pathlib.Path(file_path))], timeout=60.0)
        if not isinstance(result, dict):
            return {}
        fmt = "html" if fields.get("parse_mode") == "HTML" else "plain"
        return {**result, "_delivery": {"text": fields.get("caption", ""), "format": fmt}}

    async def send_media_group(
        self,
        chat_id: str,
        items: Sequence[Dict[str, Any]],
        *,
        topic_id: Optional[int] = None,
        reply_to_message_id: Optional[int] = None,
        before_send: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> List[Dict[str, Any]]:
        media: List[Dict[str, Any]] = []
        files: List[Tuple[str, pathlib.Path]] = []
        for index, item in enumerate(items):
            entry: Dict[str, Any] = {"type": str(item["kind"]), "media": f"attach://file{index}"}
            caption = item.get("caption") or {}
            if caption.get("text"):
                entry["caption"] = caption["text"]
                if caption.get("parse_mode"):
                    entry["parse_mode"] = caption["parse_mode"]
            media.append(entry)
            files.append((f"file{index}", pathlib.Path(item["file_path"])))
        fields: Dict[str, str] = {"chat_id": str(chat_id)}
        if topic_id is not None:
            fields["message_thread_id"] = str(int(topic_id))
        if reply_to_message_id is not None:
            fields["reply_parameters"] = json.dumps({"message_id": int(reply_to_message_id)})

        try:
            result = await self._post_multipart(
                "sendMediaGroup", {**fields, "media": json.dumps(media, ensure_ascii=False)}, files, timeout=120.0
            )
        except TelegramApiError as exc:
            if exc.error_code != 400 or not any(entry.get("parse_mode") == "HTML" for entry in media):
                raise
            if before_send is not None:
                await before_send()
            media = [
                {
                    **{key: value for key, value in entry.items() if key != "parse_mode"},
                    **({"caption": _telegram_html_to_plain(entry["caption"])} if "caption" in entry else {}),
                }
                for entry in media
            ]
            result = await self._post_multipart(
                "sendMediaGroup", {**fields, "media": json.dumps(media, ensure_ascii=False)}, files, timeout=120.0
            )
        observed = []
        for index, message in enumerate(result if isinstance(result, list) else []):
            if not isinstance(message, dict):
                observed.append({})
                continue
            entry = media[index] if index < len(media) else {"caption": message.get("caption", "")}
            fmt = "html" if entry.get("parse_mode") == "HTML" else "plain"
            observed.append({**message, "_delivery": {"text": entry.get("caption", ""), "format": fmt}})
        return observed

    async def moderate(self, action: str, parameters: Dict[str, Any]) -> bool:
        endpoint = {
            "delete_message": "deleteMessage",
            "restrict_member": "restrictChatMember",
            "ban_member": "banChatMember",
            "unban_member": "unbanChatMember",
        }.get(action)
        if endpoint is None:
            raise ValueError("unsupported Telegram moderation action")
        return bool(await self._post_json(endpoint, parameters))

    async def operation(self, method: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        if method == "editMessageText":
            body: Dict[str, Any] = {
                "chat_id": str(parameters["chat_id"]),
                "message_id": int(parameters["message_id"]),
                "text": str(parameters["text"]),
            }
            if parameters.get("parse_mode"):
                body["parse_mode"] = str(parameters["parse_mode"])
            result = await self._post_json("editMessageText", body)
            return result if isinstance(result, dict) else {"result": result}
        if method == "setMessageReaction":
            value = parameters.get("reaction")
            if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
                raise ValueError("reaction must be a list of Telegram reaction objects")
            ok = bool(
                await self._post_json(
                    "setMessageReaction",
                    {
                        "chat_id": str(parameters["chat_id"]),
                        "message_id": int(parameters["message_id"]),
                        "reaction": value,
                        "is_big": bool(parameters.get("is_big", False)),
                    },
                )
            )
            return {"ok": ok}
        raise ValueError("unsupported Telegram operation")

    async def send_chat_action(
        self, chat_id: str, action: str = "typing", *, topic_id: Optional[int] = None
    ) -> bool:
        payload: Dict[str, Any] = {"chat_id": str(chat_id), "action": str(action)}
        if topic_id is not None:
            payload["message_thread_id"] = int(topic_id)
        return bool(await self._post_json("sendChatAction", payload, timeout=10.0))

    async def _post_json(self, endpoint: str, payload: Dict[str, Any], *, timeout: float = 35.0) -> Any:
        self._check_binding(endpoint)
        url = f"{self._api_server}/bot{self._token}/{endpoint}"
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                trust_env=self._trust_env,
                proxy=self._proxy,
                follow_redirects=False,
            ) as client:
                self._check_binding(endpoint)
                response = await client.post(url, json=payload)
        except TelegramRequestNotSent:
            raise
        except _UNSENT_ERRORS as exc:
            raise TelegramRequestNotSent(endpoint, type(exc).__name__) from None
        except httpx.HTTPError as exc:
            raise TelegramApiError(endpoint, type(exc).__name__) from None
        return _parse_result(endpoint, response)

    async def _post_multipart(
        self,
        endpoint: str,
        fields: Dict[str, str],
        file_specs: Sequence[Tuple[str, pathlib.Path]],
        *,
        timeout: float = 60.0,
    ) -> Any:
        self._check_binding(endpoint)
        try:
            stream = await asyncio.to_thread(_AsyncMultipartStream.open, fields, file_specs)
        except TelegramRequestNotSent:
            raise
        except OSError as exc:
            raise TelegramRequestNotSent("upload", type(exc).__name__) from None
        url = f"{self._api_server}/bot{self._token}/{endpoint}"
        headers = {
            "Content-Type": f"multipart/form-data; boundary={stream.boundary}",
            "Content-Length": str(stream.content_length),
        }
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                trust_env=self._trust_env,
                proxy=self._proxy,
                follow_redirects=False,
            ) as client:
                self._check_binding(endpoint)
                response = await client.post(url, content=stream, headers=headers)
        except TelegramRequestNotSent:
            raise
        except _UNSENT_ERRORS as exc:
            raise TelegramRequestNotSent(endpoint, type(exc).__name__) from None
        except httpx.HTTPError as exc:
            raise TelegramApiError(endpoint, type(exc).__name__) from None
        finally:
            await stream.aclose()
        return _parse_result(endpoint, response)


_UPLOAD_STREAM_CHUNK = 64 * 1024


class _AsyncMultipartStream(httpx.AsyncByteStream):
    """Bounded-memory async multipart body streaming files in 64 KiB chunks off the event loop."""

    def __init__(self, boundary: str, field_parts: List[bytes], file_parts: List[Tuple[bytes, int, int]], tail: bytes):
        self.boundary = boundary
        self._field_parts = field_parts
        self._file_parts = file_parts  # (header_bytes, fd, expected_size)
        self._tail = tail
        self.content_length = (
            sum(len(p) for p in field_parts)
            + sum(len(h) + sz + 2 for h, _fd, sz in file_parts)
            + len(tail)
        )
        self._closed = False

    @classmethod
    def open(cls, fields: Dict[str, str], file_specs: Sequence[Tuple[str, pathlib.Path]]) -> "_AsyncMultipartStream":
        boundary = f"ouro-tg-{uuid.uuid4().hex}"
        b_bytes = boundary.encode("ascii")
        field_parts: List[bytes] = []
        for key, value in fields.items():
            safe_key = str(key).replace("\\", "_").replace('"', "_").replace("\r", "_").replace("\n", "_")
            part = (
                b"--" + b_bytes + b"\r\n"
                + f'Content-Disposition: form-data; name="{safe_key}"\r\n\r\n'.encode("utf-8")
                + str(value).encode("utf-8")
                + b"\r\n"
            )
            field_parts.append(part)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
        file_parts: List[Tuple[bytes, int, int]] = []
        try:
            for field_name, file_path in file_specs:
                path = pathlib.Path(file_path)
                fd = os.open(path, flags)
                try:
                    st = os.fstat(fd)
                    if not stat.S_ISREG(st.st_mode):
                        raise TelegramRequestNotSent("upload", "upload source is not a regular file")
                    if st.st_size > CLOUD_UPLOAD_LIMIT:
                        raise TelegramRequestNotSent("upload", "file exceeds Telegram Bot API upload limit")
                    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                    safe_name = path.name.replace("\\", "_").replace('"', "_").replace("\r", "_").replace("\n", "_")
                    safe_field = str(field_name).replace("\\", "_").replace('"', "_").replace("\r", "_").replace("\n", "_")
                    hdr = (
                        b"--" + b_bytes + b"\r\n"
                        + f'Content-Disposition: form-data; name="{safe_field}"; filename="{safe_name}"\r\n'.encode("utf-8")
                        + f"Content-Type: {mime}\r\n\r\n".encode("utf-8")
                    )
                    file_parts.append((hdr, fd, int(st.st_size)))
                except BaseException:
                    os.close(fd)
                    raise
        except BaseException:
            for _h, opened_fd, _sz in file_parts:
                try:
                    os.close(opened_fd)
                except OSError:
                    pass
            raise
        return cls(boundary, field_parts, file_parts, b"--" + b_bytes + b"--\r\n")

    async def __aiter__(self):
        try:
            for part in self._field_parts:
                yield part
            for hdr, fd, expected in self._file_parts:
                yield hdr
                remaining = expected
                while remaining > 0:
                    chunk = await asyncio.to_thread(os.read, fd, min(_UPLOAD_STREAM_CHUNK, remaining))
                    if not chunk:
                        raise OSError("frozen upload file truncated during streaming")
                    remaining -= len(chunk)
                    yield chunk
                yield b"\r\n"
            yield self._tail
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        for _h, fd, _sz in self._file_parts:
            try:
                os.close(fd)
            except OSError:
                pass


def _announced_length(headers: Any) -> Optional[int]:
    if "chunked" in str(headers.get("transfer-encoding") or "").lower():
        return None
    try:
        length = int(str(headers.get("content-length") or ""))
    except ValueError:
        return None
    return length if length >= 0 else None


def _parse_result(endpoint: str, response: httpx.Response) -> Any:
    try:
        payload = response.json()
    except ValueError:
        raise TelegramApiError(endpoint, "provider returned invalid JSON", response.status_code) from None
    if not isinstance(payload, dict) or not payload.get("ok"):
        body = payload if isinstance(payload, dict) else {}
        try:
            code = int(body.get("error_code") or response.status_code or 0)
        except (TypeError, ValueError, OverflowError):
            code = response.status_code if response.status_code >= 400 else 0
        parameters = body.get("parameters") if isinstance(body.get("parameters"), dict) else {}
        wait = parameters.get("retry_after")
        raise TelegramApiError(
            endpoint,
            str(body.get("description") or f"HTTP {response.status_code}"),
            code,
            retry_after=float(wait) if type(wait) in (int, float) and wait >= 0 else None,
        )
    return payload.get("result")
