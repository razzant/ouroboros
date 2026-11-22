"""Frozen Markdown-to-Telegram-HTML preparation backed by ``lib/telegram_api.py``."""
from __future__ import annotations

from ..telegram_api import (
    _chunk_raw_text,
    _telegram_html_to_plain,
    _u16len,
    markdown_to_telegram_chunks,
    markdown_to_telegram_html,
)


def prepare_text(text: str, *, markdown: bool = True) -> list[dict[str, str]]:
    """Render once; callers persist these exact chunks before first delivery."""
    chunks = markdown_to_telegram_chunks(text) if markdown else _chunk_raw_text(text)
    return [{"text": chunk, "parse_mode": "HTML" if markdown else ""} for chunk in chunks]


def prepare_caption(caption: str, *, markdown: bool = True) -> dict[str, str]:
    """Keep caption contents intact or report the provider limit before upload."""
    text = markdown_to_telegram_html(caption) if markdown else caption
    visible = _telegram_html_to_plain(text) if markdown else text
    if _u16len(visible) > 1024:
        raise ValueError(
            "Telegram captions are limited to 1024 UTF-16 units; send a shorter caption "
            "and put the remaining content in a separate text message"
        )
    return {"text": text, "parse_mode": "HTML" if markdown else ""}


__all__ = [
    "_telegram_html_to_plain",
    "markdown_to_telegram_chunks",
    "markdown_to_telegram_html",
    "prepare_caption",
    "prepare_text",
]
