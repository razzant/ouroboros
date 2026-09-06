"""Provider-safe naming and syntax rules for extension surfaces.

An extension's tools, routes, WebSocket handlers, UI tabs and settings sections
all live in one namespace derived from the skill name, so a surface can never
collide with a first-party tool or with another extension. This module owns that
encoding, its inverse, and the syntax assertions every registration passes
through.
"""

from __future__ import annotations

import hashlib
import pathlib
import re
from typing import Any, Dict

from ouroboros.contracts.plugin_api import ExtensionRegistrationError
from ouroboros.extension_ui_validation import WIDGET_FRAME_MAX_HEIGHT, WIDGET_FRAME_MIN_HEIGHT

_EXTENSION_NAME_PREFIX = "ext_"


_EXTENSION_SKILL_TOKEN_MAX = 32


_EXTENSION_SHORT_MAX = 24


_EXTENSION_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _extension_skill_token(skill_name: str) -> str:
    """Return a short ASCII token without changing skill identity."""
    text = str(skill_name or "").strip()
    safe = "".join(ch if (ch.isascii() and (ch.isalnum() or ch in "-_")) else "_" for ch in text)
    safe = re.sub(r"_+", "_", safe).strip("_-")
    raw_budget = _EXTENSION_SKILL_TOKEN_MAX - 2
    if safe and safe == text and len(safe) <= raw_budget:
        return f"r_{safe}"
    digest = hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:10]
    prefix_budget = _EXTENSION_SKILL_TOKEN_MAX - len(digest) - 3
    prefix = (safe or "skill")[:prefix_budget].strip("_-") or "skill"
    return f"h_{prefix}_{digest}"


def extension_name_prefix(skill_name: str) -> str:
    """Return the provider-safe prefix for one extension."""
    token = _extension_skill_token(skill_name)
    return f"{_EXTENSION_NAME_PREFIX}{len(token)}_{token}_"


def extension_surface_name(skill_name: str, short_name: str) -> str:
    """Return a provider-safe canonical surface name."""
    full = f"{extension_name_prefix(skill_name)}{short_name}"
    if not _EXTENSION_NAME_RE.match(full):
        raise ExtensionRegistrationError(
            f"extension surface name {full!r} must match provider tool-name limits"
        )
    return full


def parse_extension_surface_name(name: str) -> tuple[str, str] | None:
    """Return ``(encoded_skill_token, short_name)`` for extension surface names."""
    text = str(name or "").strip()
    if not _EXTENSION_NAME_RE.match(text) or not text.startswith(_EXTENSION_NAME_PREFIX):
        return None
    rest = text[len(_EXTENSION_NAME_PREFIX):]
    length_text, sep, remainder = rest.partition("_")
    if sep != "_" or not length_text.isdigit():
        return None
    token_len = int(length_text)
    if token_len < 1 or len(remainder) <= token_len or remainder[token_len] != "_":
        return None
    token = remainder[:token_len]
    short = remainder[token_len + 1:]
    return token, short


def _assert_namespace_path(path: str) -> str:
    """Return a normalised relative path for route registration or raise."""
    rel = str(path or "").strip()
    if not rel:
        raise ExtensionRegistrationError("path must be non-empty")
    if rel.startswith("/"):
        raise ExtensionRegistrationError(
            f"path must be relative, not absolute: {rel!r}"
        )
    if ".." in pathlib.PurePosixPath(rel).parts:
        raise ExtensionRegistrationError(
            f"path must not contain '..' segments: {rel!r}"
        )
    return rel


def _assert_tool_name(name: str) -> str:
    candidate = str(name or "").strip()
    if not candidate:
        raise ExtensionRegistrationError("tool name must be non-empty")
    if len(candidate) > _EXTENSION_SHORT_MAX:
        raise ExtensionRegistrationError(
            f"tool name must be <= {_EXTENSION_SHORT_MAX} characters: {candidate!r}"
        )
    if not candidate.replace("_", "").isalnum():
        raise ExtensionRegistrationError(
            f"tool name must be alnum/underscore only: {candidate!r}"
        )
    return candidate


def _widget_span_from_render(render: Dict[str, Any]) -> int:
    """Normalize optional UI-card width metadata from a render declaration."""
    raw = render.get("span", render.get("grid_span", 1))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 1
    return 2 if value >= 2 else 1


def _widget_geometry_from_render(render: Dict[str, Any]) -> Dict[str, int]:
    """Promote normalized framed geometry into the host tab descriptor."""
    geometry: Dict[str, int] = {}
    for key in ("height", "max_height"):
        value = render.get(key)
        if value is None:
            continue
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            continue
        geometry[key] = max(WIDGET_FRAME_MIN_HEIGHT, min(numeric, WIDGET_FRAME_MAX_HEIGHT))
    return geometry
