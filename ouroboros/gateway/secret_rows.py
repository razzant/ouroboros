"""Masking and save-time rehydration of secrets nested in list settings.

``GET /api/settings`` answers a row's secret with the Settings placeholder; a save
that sends the placeholder back keeps the ONE saved row with the same identity's
value. An identity claimed by several saved or incoming rows refuses before
anything is written: an equal placeholder never moves a credential between rows.
``MCP_SERVERS`` (identity = the loader's server id) and ``OUROBOROS_CONNECTIONS``
(identity = kind and name) share that rule; each owns which fields are secret.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from ouroboros.provider_models import PROVIDER_KINDS
from ouroboros.secret_masking import (
    MCP_RESPONSE_ONLY_FIELDS, looks_masked_mcp_secret, looks_masked_settings_secret, mask_mcp_url,
    mask_prefixed_secret, mask_settings_secret, rehydrate_mcp_url,
)


def rehydrate_rows(incoming: Any, current: Any, *, identity: Callable[[dict], Any],
                   prepare: Callable[[dict], dict], masked: Callable[[dict], bool],
                   restore: Callable[[dict, dict, Any], dict], ambiguous: Callable[[Any], Exception]) -> list:
    """Restore each incoming row's placeholders from the one saved row of its identity."""
    if not isinstance(incoming, list):
        return []
    saved: Dict[Any, list] = {}
    for row in current if isinstance(current, list) else []:
        if isinstance(row, dict) and identity(row):
            saved.setdefault(identity(row), []).append(row)
    claims = [identity(row) for row in incoming if isinstance(row, dict)]
    out = []
    for row in incoming:
        if not isinstance(row, dict):
            continue
        clone = prepare(row)
        key = identity(clone)
        matches = saved.get(key, []) if key else []
        if masked(clone) and (len(matches) > 1 or claims.count(key) > 1):
            raise ambiguous(key)
        out.append(restore(clone, matches[0] if len(matches) == 1 else {}, key))
    return out


# --- MCP_SERVERS ------------------------------------------------------------------

def mask_mcp_servers(servers: Any) -> list:
    from ouroboros.mcp_headers import mask_headers
    if not isinstance(servers, list):
        return []
    try:
        from ouroboros.mcp_client import canonical_server_id as _mcp_canonical_id
    except Exception:
        _mcp_canonical_id = lambda value: str(value or "").strip()  # type: ignore[assignment]
    out = []
    for entry in servers:
        if not isinstance(entry, dict):
            continue
        clone = dict(entry)
        if clone.get("id"):
            clone["id"] = _mcp_canonical_id(clone.get("id"))
        if "url" in clone:
            clone["url"] = mask_mcp_url(clone["url"])
        if "headers" in clone:
            clone["headers"] = mask_headers(clone["headers"])
        token = str(clone.get("auth_token") or "")
        if token:
            clone["auth_token"] = mask_prefixed_secret(token, visible_chars=8)
            clone["auth_configured"] = True
        else:
            clone["auth_token"] = ""
            clone["auth_configured"] = False
        out.append(clone)
    return out


class MCPSecretIdentityAmbiguous(ValueError):
    """A masked MCP secret whose server identity is claimed more than once."""

    code = "MCP_ID_AMBIGUOUS_SECRET"

    def __init__(self, server_id: str) -> None:
        super().__init__(
            f"{self.code}: several MCP servers resolve to server id {server_id!r}, so its saved "
            "credential cannot be matched to one of them. Give each server a distinct Server ID "
            "(or re-enter its token/URL) and save again; nothing was saved."
        )


def rehydrate_mcp_servers(incoming: Any, current: Any) -> list:
    """Restore masked MCP tokens/URLs/header values from the ONE saved server with the same
    identity (the loader's ``id``, ``slug`` or ``name``); several claimants raise
    ``MCPSecretIdentityAmbiguous``."""
    from ouroboros.mcp_client import canonical_server_id, raw_server_id
    from ouroboros.mcp_headers import holds_placeholder, restore_headers

    def prepare(entry: dict) -> dict:
        clone = {key: value for key, value in entry.items() if key not in MCP_RESPONSE_ONLY_FIELDS}
        if clone.get("id"):
            clone["id"] = canonical_server_id(clone.get("id"))
        return clone

    def masked(clone: dict) -> bool:
        return holds_placeholder(clone.get("headers")) or looks_masked_mcp_secret(str(clone.get("auth_token") or "")) or (
            "url" in clone and rehydrate_mcp_url(clone["url"], "") != str(clone["url"] or ""))

    def restore(clone: dict, existing: dict, server_id: Any) -> dict:
        if "url" in clone:
            clone["url"] = rehydrate_mcp_url(clone["url"], existing.get("url"))
        if looks_masked_mcp_secret(str(clone.get("auth_token") or "")):
            clone["auth_token"] = str(existing.get("auth_token") or "")
        if "headers" in clone:
            clone["headers"] = restore_headers(clone["headers"], existing, server_id=server_id)
        return clone

    return rehydrate_rows(incoming, current, identity=raw_server_id, prepare=prepare, masked=masked,
                          restore=restore, ambiguous=MCPSecretIdentityAmbiguous)


# --- OUROBOROS_CONNECTIONS -----------------------------------------------------------

def _connection_identity(row: dict) -> tuple:
    kind, name = str(row.get("kind") or "").strip(), str(row.get("name") or "").strip()
    return (kind, name) if kind and name else ()


def _secret_fields(row: dict) -> tuple:
    """``(row_field, flat_key)`` of a row's secrets; the flat key picks the placeholder class.
    A row of an unknown kind masks every field any kind treats as secret."""
    spec = PROVIDER_KINDS.get(str(row.get("kind") or "").strip())
    if spec is not None:
        return tuple((field, flat) for field, flat, secret in spec.fields if secret)
    return tuple({(field, flat) for spec in PROVIDER_KINDS.values() for field, flat, secret in spec.fields if secret})


def mask_connections(rows: Any) -> list:
    """``OUROBOROS_CONNECTIONS`` as ``GET /api/settings`` answers it: each secret as the placeholder of
    the same secret's flat setting (a GigaChat password stays a no-prefix placeholder)."""
    out = []
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict):
            clone = dict(row)
            for field, flat in _secret_fields(clone):
                if str(clone.get(field) or ""):
                    clone[field] = mask_settings_secret(flat, clone[field])
            out.append(clone)
    return out


class ConnectionSecretIdentityAmbiguous(ValueError):
    """A masked connection secret whose (kind, name) is claimed more than once."""

    code = "CONNECTION_ID_AMBIGUOUS_SECRET"

    def __init__(self, identity: Any) -> None:
        label = "@".join(identity) if identity else "a connection without a kind or name"
        super().__init__(
            f"{self.code}: several connections claim {label}, so its saved key cannot be matched "
            "to one of them. Give each connection its own name (or re-enter its key) and save again; "
            "nothing was saved.")


def rehydrate_connections(incoming: Any, current: Any) -> list:
    """Restore each row's masked secrets from the ONE saved row with the same kind and name.

    A placeholder with no saved row behind it (a new or renamed row) is no secret: it saves empty."""
    def masked(row: dict) -> bool:
        return any(looks_masked_settings_secret(flat, row.get(field)) for field, flat in _secret_fields(row)
                   if field in row)

    def restore(row: dict, existing: dict, _identity: Any) -> dict:
        for field, flat in _secret_fields(row):
            if field in row and looks_masked_settings_secret(flat, row[field]):
                row[field] = str(existing.get(field) or "")
        return row

    return rehydrate_rows(incoming, current, identity=_connection_identity, prepare=dict, masked=masked,
                          restore=restore, ambiguous=ConnectionSecretIdentityAmbiguous)
