"""Wire facts of OpenCode Go's official Chat Completions endpoint.

Go documents an honest User-Agent and one stable session ID per coding-agent
conversation (https://opencode.ai/docs/console/go/#where-can-i-use-it, 2026-11-18).
This is a facet of an existing compatible API connection, not another provider.
"""

from __future__ import annotations

import hashlib
import json
import uuid

from ouroboros import __version__

# https://opencode.ai/docs/console/harnesses/#base-urls (Chat Completions)
GO_CHAT_BASE_URL = "https://opencode.ai/inference/go/openai/v1"
# https://opencode.ai/docs/console/api/models/#inference-model-lists (Go list)
GO_MODELS_URL = "https://opencode.ai/inference/go/v1/models"


def is_go_chat_route(provider: str | None, base_url: str | None) -> bool:
    """The documented base URL, with an optional trailing slash.

    Scheme and DNS host match in any ASCII letter case, as a client resolves them;
    every other character is exact, so a port, userinfo, query or other path is not Go.
    """
    if provider != "openai-compatible" or not isinstance(base_url, str) or not base_url.isascii():
        return False
    origin = len("https://opencode.ai")
    return base_url[:origin].lower() + base_url[origin:] in (GO_CHAT_BASE_URL, GO_CHAT_BASE_URL + "/")


def compatible_models_url(provider: str, base_url: str) -> str:
    """Go publishes one list outside its protocol-specific Chat base."""
    return GO_MODELS_URL if is_go_chat_route(provider, base_url) else f"{base_url.rstrip('/')}/models"


def go_client_headers(provider: str | None, base_url: str | None) -> dict[str, str]:
    """Identify this client on Go; reading metadata is not a coding conversation."""
    return {"User-Agent": f"Ouroboros/{__version__}"} if is_go_chat_route(provider, base_url) else {}


def go_chat_headers(target: dict) -> dict[str, str]:
    """Scope the caller's conversation to its model; an unbound send opens its own exchange."""
    headers = go_client_headers(target.get("provider"), target.get("base_url"))
    if not headers:
        return headers
    conversation = target.get("chat_conversation_id")
    model = target.get("usage_model") or target.get("resolved_model")
    identifier = (hashlib.sha256(json.dumps((conversation, model)).encode("ascii")).hexdigest()[:32]
                  if conversation else uuid.uuid4().hex)
    return {**headers, "x-opencode-session": f"ouroboros-{identifier}"}


def go_model_limits(item: dict) -> tuple[int, int]:
    """Advertised `limit.context`/`limit.output`, not a serving window or key entitlement.

    https://opencode.ai/docs/console/api/models/#fields
    """
    limits = item.get("limit")
    if not isinstance(limits, dict):
        return 0, 0

    def count(key: str) -> int:
        value = limits.get(key)
        return value if type(value) is int and value > 0 else 0

    return count("context"), count("output")
