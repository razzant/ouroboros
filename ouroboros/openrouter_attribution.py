"""Canonical OpenRouter application attribution for Ouroboros traffic."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"  # the public endpoint; ``llm_routing`` re-exports it
OPENROUTER_APP_URL = "https://ouroboros-agent.ai/"
OPENROUTER_APP_TITLE = "Ouroboros"
OPENROUTER_APP_CATEGORIES = ("personal-agent", "programming-app")
OPENROUTER_APP_HEADERS = {
    "HTTP-Referer": OPENROUTER_APP_URL,
    "X-OpenRouter-Title": OPENROUTER_APP_TITLE,
    "X-OpenRouter-Categories": ",".join(OPENROUTER_APP_CATEGORIES),
}


def is_openrouter_endpoint(base_url: Any) -> bool:
    """Whether a configured base URL is OpenRouter's own public HTTPS endpoint (no userinfo, default port)."""
    try:
        parts = urlsplit(str(base_url or "").strip())
        port = parts.port
    except ValueError:
        return False
    return (parts.scheme == "https" and parts.hostname == "openrouter.ai" and port in (None, 443)
            and "@" not in parts.netloc)
