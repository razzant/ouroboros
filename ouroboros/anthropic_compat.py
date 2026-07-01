"""Anthropic-compatible endpoint helpers.

Shared by the handwritten Anthropic runtime, model-catalog loading, and Claude
Code env setup so compatible base URLs normalize consistently without leaking
secrets across those call sites.
"""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

OFFICIAL_ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
MINIMAX_ANTHROPIC_HOSTS = {"api.minimax.io", "api.minimaxi.com"}
MINIMAX_CLAUDE_MODEL = "MiniMax-M3[1m]"
MINIMAX_CLAUDE_AUTO_COMPACT_WINDOW = "700000"


def normalize_anthropic_base_url(base_url: str | None) -> str:
    raw = str(base_url or "").strip()
    if not raw:
        return OFFICIAL_ANTHROPIC_BASE_URL
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Anthropic Base URL must be an http(s) URL")
    if parsed.username or parsed.password:
        raise ValueError("Anthropic Base URL must not include credentials")
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1"):
        path = f"{path}/v1" if path else "/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def anthropic_messages_url(base_url: str | None) -> str:
    return f"{normalize_anthropic_base_url(base_url).rstrip('/')}/messages"


def anthropic_models_url(base_url: str | None) -> str:
    return f"{normalize_anthropic_base_url(base_url).rstrip('/')}/models"


def is_minimax_anthropic_base_url(base_url: str | None) -> bool:
    try:
        parsed = urlsplit(normalize_anthropic_base_url(base_url))
    except ValueError:
        return False
    return parsed.netloc.lower() in MINIMAX_ANTHROPIC_HOSTS


def minimax_claude_compact_window(value: str | None) -> str:
    raw = str(value or "").strip()
    if not raw:
        return MINIMAX_CLAUDE_AUTO_COMPACT_WINDOW
    try:
        parsed = int(raw)
    except ValueError:
        return MINIMAX_CLAUDE_AUTO_COMPACT_WINDOW
    if parsed <= 0:
        return MINIMAX_CLAUDE_AUTO_COMPACT_WINDOW
    return str(parsed)
