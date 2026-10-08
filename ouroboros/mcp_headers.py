"""Literal HTTP headers of configured MCP servers: validation, masks and restoration.

An HTTP/SSE server may carry an optional ``headers`` map of ``{name: value}``.
Values are sent exactly as written (no scheme is added or removed) and every
value is a credential whatever its header name: passive Settings projections
emit one placeholder per value, and only the exact placeholder of the one
saved server with the same identity and the same case-insensitive header name
restores a saved value. An empty value is kept in the configuration but is not
sent. The legacy ``auth_header``/``auth_token`` pair keeps its own meaning; a
header that the active legacy pair also sets is a configuration error, never a
silent precedence.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping

from ouroboros.secret_masking import CONFIGURED_SECRET_PLACEHOLDER

HEADER_VALUE_PLACEHOLDER = CONFIGURED_SECRET_PLACEHOLDER
HEADER_NAME_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_VALUE_RE = re.compile(r"^[\x20-\x7e]*$")  # printable ASCII: no control or non-ASCII bytes


def validate_headers(value: Any, *, legacy_header: str = "Authorization", legacy_token: str = "") -> Dict[str, str]:
    """Return the literal header map or raise ``ValueError`` naming the problem.

    Messages name a header only when its name is a valid token and never repeat
    a value, so a misplaced credential is not echoed into a status row.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("headers must be an object of header names and string values")
    seen: Dict[str, str] = {}
    for position, (name, item) in enumerate(value.items(), start=1):
        if not isinstance(name, str) or not HEADER_NAME_RE.fullmatch(name):
            raise ValueError(f"header #{position} name is not a single HTTP header token")
        if not isinstance(item, str):
            raise ValueError(f"header {name!r} value must be a string")
        if not _VALUE_RE.fullmatch(item):
            raise ValueError(f"header {name!r} value must be printable ASCII without control characters")
        if item != item.strip():
            raise ValueError(f"header {name!r} value must not start or end with whitespace")
        folded = name.casefold()
        if folded in seen:
            raise ValueError(f"header names {seen[folded]!r} and {name!r} differ only by case; keep one")
        seen[folded] = name
    legacy = str(legacy_header or "Authorization").casefold()
    if str(legacy_token or "").strip() and value.get(seen.get(legacy, "")):
        raise ValueError(
            f"header {seen[legacy]!r} is also set by the legacy auth token; clear the legacy "
            "Auth token or remove the header")
    return dict(value)


def mask_headers(value: Any) -> Any:
    """Passive projection: names stay, every nonempty or malformed value is a placeholder."""
    if isinstance(value, dict):
        return {name: "" if item == "" else HEADER_VALUE_PLACEHOLDER for name, item in value.items()}
    return value if value is None else HEADER_VALUE_PLACEHOLDER


def holds_placeholder(value: Any) -> bool:
    """Whether a projected ``headers`` value carries a placeholder only restoration may resolve."""
    if isinstance(value, dict):
        return any(item == HEADER_VALUE_PLACEHOLDER for item in value.values())
    return value == HEADER_VALUE_PLACEHOLDER


def find_saved_header(saved: Any, name: str) -> tuple[str, Any]:
    """``(status, value)`` of one saved header: found, missing, ambiguous or malformed."""
    if not isinstance(saved, dict):
        return "malformed", None
    folded = str(name or "").casefold()
    matches = [key for key in saved if isinstance(key, str) and key.casefold() == folded]
    if len(matches) != 1:
        return ("ambiguous" if matches else "missing"), None
    return "found", saved[matches[0]]


class MCPHeaderPlaceholderUnmatched(ValueError):
    """A masked header value without exactly one saved value to restore."""

    code = "MCP_HEADER_REENTER"


def restore_headers(incoming: Any, saved_server: Mapping[str, Any], *, server_id: str) -> Any:
    """Replace exact placeholders with the saved values of the same server identity.

    ``saved_server`` is the one saved server with this identity (empty when
    there is none). A placeholder under a header name that server does not
    carry exactly once is refused, so a credential never moves to another name.
    """
    saved = saved_server.get("headers")
    if incoming == HEADER_VALUE_PLACEHOLDER:
        if saved is not None and not isinstance(saved, dict):
            return saved  # the whole-map placeholder of a malformed saved value
        raise MCPHeaderPlaceholderUnmatched(
            f"MCP_HEADER_REENTER: server {server_id!r} has no saved headers to keep; enter its "
            "headers again. Nothing was saved.")
    if not isinstance(incoming, dict):
        return incoming
    out: Dict[str, Any] = {}
    for name, item in incoming.items():
        if item != HEADER_VALUE_PLACEHOLDER:
            out[name] = item
            continue
        status, value = find_saved_header(saved, name)
        if status != "found":
            reason = ("several saved headers match it case-insensitively" if status == "ambiguous"
                      else "it is not saved under that name for this server")
            label = repr(name) if isinstance(name, str) and HEADER_NAME_RE.fullmatch(name) else "(invalid name)"
            raise MCPHeaderPlaceholderUnmatched(
                f"MCP_HEADER_REENTER: the masked value of header {label} on server {server_id!r} "
                f"cannot be kept because {reason}. Enter the value again (or restore the original "
                "header name and server ID). Nothing was saved.")
        out[name] = value
    return out


def validate_changed_headers(incoming: Any, current: Any) -> None:
    """Validate newly authored header settings, not unrelated legacy damage.

    A save can round-trip a malformed saved map untouched. Editing its map,
    transport or legacy pair is the point at which the header contract applies.
    """
    from ouroboros.mcp_client import raw_server_id

    saved: Dict[str, list] = {}
    for entry in current if isinstance(current, list) else []:
        if raw_server_id(entry):
            saved.setdefault(raw_server_id(entry), []).append(entry)
    for entry in incoming if isinstance(incoming, list) else []:
        if not isinstance(entry, dict) or "headers" not in entry:
            continue
        matches = saved.get(raw_server_id(entry), [])
        old = matches[0] if len(matches) == 1 else {}
        def selection(row):
            return (row.get("headers"), str(row.get("transport") or "streamable_http").strip().lower(),
                    str(row.get("auth_header") or "Authorization").strip(), str(row.get("auth_token") or "").strip())

        if old and selection(entry) == selection(old):
            continue
        headers = validate_headers(entry["headers"],
                                   legacy_header=str(entry.get("auth_header") or "Authorization").strip(),
                                   legacy_token=entry.get("auth_token", ""))
        if str(entry.get("transport") or "streamable_http").strip().lower() == "stdio" and headers:
            raise ValueError("headers are unsupported for stdio")
