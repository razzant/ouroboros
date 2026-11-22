"""Outbound media and links event handlers for the Telegram owner bridge."""

from __future__ import annotations

import asyncio
import base64
import pathlib
from typing import Any, Callable, Dict

from . import telegram_i18n
from .miniapp_registration import _read_status
from .telegram_api import (
    _LOCALIZED_TEXTS,
    _MAX_TELEGRAM_UPLOAD_BYTES,
    TelegramRequestRejected,
    markdown_to_telegram_html,
    replacing_request,
)
from .telegram_state import _clear_silent_msg, _data_dir
from .telegram_token import _report_failure


def is_native_audio_document(filename: str, mime: str) -> bool:
    extension = pathlib.Path(str(filename or "")).suffix.casefold()
    normalized_mime = str(mime or "").split(";", 1)[0].strip().casefold()
    return extension in {".mp3", ".m4a"} or normalized_mime in {
        "audio/mpeg",
        "audio/mp3",
        "audio/mp4",
        "audio/x-m4a",
    }


def make_photo_handler(
    api: Any,
    *,
    load_settings: Callable[[Any], Dict[str, Any]],
    target_chat: Callable[[Dict[str, Any], Dict[str, Any]], int],
    client_factory: Callable[..., Any],
):
    async def handle(event: Dict[str, Any]) -> None:
        try:
            chat_id = target_chat(load_settings(api), event)
            image_base64 = str(event.get("image_base64") or "").strip()
            if chat_id and image_base64:
                client = client_factory(api, event, chat_id)
                _clear_silent_msg(api, chat_id)
                caption = await client.media_caption(chat_id, str(event.get("caption") or ""))
                await client.send_photo(
                    chat_id,
                    image_base64,
                    caption=caption,
                    mime=str(event.get("mime") or "image/png"),
                )
        except Exception as exc:
            _report_failure(api, "photo", exc)

    return handle


def make_video_handler(
    api: Any,
    *,
    load_settings: Callable[[Any], Dict[str, Any]],
    target_chat: Callable[[Dict[str, Any], Dict[str, Any]], int],
    client_factory: Callable[..., Any],
):
    async def handle(event: Dict[str, Any]) -> None:
        try:
            chat_id = target_chat(load_settings(api), event)
            video_base64 = str(event.get("video_base64") or "").strip()
            if chat_id and video_base64:
                client = client_factory(api, event, chat_id)
                _clear_silent_msg(api, chat_id)
                caption = await client.media_caption(chat_id, str(event.get("caption") or ""))
                mime = str(event.get("mime") or "video/mp4")
                files = {"video": ("video.mp4", base64.b64decode(video_base64), mime)}
                data = {"chat_id": str(chat_id), "caption": markdown_to_telegram_html(caption), "parse_mode": "HTML"}
                await client.call("sendVideo", data=data, files=files, timeout=40)
        except Exception as exc:
            _report_failure(api, "video", exc)

    return handle


def make_document_handler(
    api: Any,
    *,
    load_settings: Callable[[Any], Dict[str, Any]],
    target_chat: Callable[[Dict[str, Any], Dict[str, Any]], int],
    client_factory: Callable[..., Any],
    read_status: Callable[[Any], Dict[str, Any]] = _read_status,
):
    async def handle(event: Dict[str, Any]) -> None:
        file_handle = None
        try:
            chat_id = target_chat(load_settings(api), event)
            file_base64 = str(event.get("file_base64") or "").strip()
            file_ref = event.get("file_ref") if isinstance(event.get("file_ref"), dict) else None
            if chat_id and (file_base64 or file_ref):
                client = client_factory(api, event, chat_id)
                _clear_silent_msg(api, chat_id)
                filename = str(event.get("filename") or "file")
                caption = str(event.get("caption") or "")
                if file_ref:
                    from ouroboros.gateway.files import resolve_task_file_reference

                    task_id = str(event.get("task_id") or file_ref.get("task_id") or "")
                    source = await asyncio.to_thread(resolve_task_file_reference, _data_dir(api), task_id, file_ref)
                    if file_ref["size"] > _MAX_TELEGRAM_UPLOAD_BYTES:
                        t = _LOCALIZED_TEXTS[telegram_i18n.language()]
                        notice = t.format("file_too_large_notice", filename=filename)
                        status = await asyncio.to_thread(read_status, api)
                        if status.get("state") == "ready" and status.get("public_url"):
                            try:
                                await client.send_message_with_inline_keyboard(
                                    chat_id,
                                    notice,
                                    [[{"text": t["btn_open_app"], "web_app": {"url": status["public_url"]}}]],
                                )
                                return
                            except TelegramRequestRejected as exc:
                                if not exc.plain_retry_safe:
                                    raise
                                with replacing_request():
                                    await client.send_message(
                                        chat_id, notice + " " + t["file_open_app_hint"], parse_mode=""
                                    )
                                return
                        await client.send_message(chat_id, notice + " " + t["file_open_app_hint"], parse_mode="")
                        return
                    file_handle = source.open("rb")
                    file_bytes = file_handle
                else:
                    file_bytes = base64.b64decode(file_base64)
                mime = str(event.get("mime") or "application/octet-stream")
                caption = await client.media_caption(chat_id, caption)
                if is_native_audio_document(filename, mime):
                    try:
                        await client.send_audio(chat_id, file_bytes, filename=filename, caption=caption, mime=mime)
                    except TelegramRequestRejected as exc:
                        if exc.status_code != 400 or exc.transient:
                            raise
                        if file_handle is not None:
                            file_handle.seek(0)
                        with replacing_request():
                            await client.send_document(chat_id, file_bytes, filename=filename, caption=caption)
                else:
                    await client.send_document(chat_id, file_bytes, filename=filename, caption=caption)
        except Exception as exc:
            _report_failure(api, "document", exc)
        finally:
            if file_handle is not None:
                file_handle.close()

    return handle


def make_links_handler(
    api: Any,
    *,
    load_settings: Callable[[Any], Dict[str, Any]],
    target_chat: Callable[[Dict[str, Any], Dict[str, Any]], int],
    client_factory: Callable[..., Any],
):
    async def handle(event: Dict[str, Any]) -> None:
        try:
            chat_id = target_chat(load_settings(api), event)
            if not chat_id:
                return
            client = client_factory(api, event, chat_id)
            title = str(event.get("title") or "").strip() or "Links"
            raw_actions = event.get("actions") if isinstance(event.get("actions"), list) else []
            actions = []
            for action in raw_actions:
                if not isinstance(action, dict):
                    continue
                label = str(action.get("label") or "").strip()
                url = str(action.get("url") or "").strip()
                if label and url:
                    actions.append((label, url))
            actions = actions[:12]
            if not actions:
                return
            _clear_silent_msg(api, chat_id)
            keyboard = [[{"text": label, "url": url}] for label, url in actions]
            try:
                await client.send_message_with_inline_keyboard(chat_id, title, keyboard)
            except TelegramRequestRejected as exc:
                if not exc.plain_retry_safe:
                    raise
                api.log("warning", "Telegram links keyboard failed; sending a plain list.")
                plain_text = "\n".join([title, *(f"{label} — {url}" for label, url in actions)])
                with replacing_request():
                    await client.send_message(chat_id, plain_text, parse_mode="")
        except Exception as exc:
            _report_failure(api, "links", exc)

    return handle
