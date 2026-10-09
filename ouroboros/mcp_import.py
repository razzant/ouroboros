"""One-way preview of a pasted ``mcpServers`` document as MCP Settings draft patches.

The common client dialect — a top-level ``mcpServers`` object whose keys name
servers — is translated here and nothing else happens: no process starts, no
address is contacted and nothing is saved. Each entry becomes an ``add`` (a new,
disabled server) or an ``update`` of the ONE draft server with the same
canonical identity (``mcp_client.canonical_server_id``) and the same transport.
A field the entry does not name stays as it is; a named ``headers`` object
replaces the whole map, so ``{}`` clears it. ``Authorization`` stays a literal
header, never the legacy ``auth_header``/``auth_token`` pair. Malformed JSON,
duplicate keys, wrong types and ambiguous identities are problems that keep an
entry out of the result rather than truncating it. The loader's field
validators judge imported fields without loading settings or connecting.

The displayed projection carries names and counts only; addresses and commands stay in patches,
``patch`` carries the pasted values back to the client draft, which the owner
then saves through the ordinary Settings writer.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from ouroboros.mcp_client import (
    canonical_server_id, raw_server_id, _validate_url,
    _validate_stdio_args, _validate_stdio_command,
)
from ouroboros.mcp_headers import validate_headers
from ouroboros.workspace_executor import validate_process_env

TRANSPORTS = {"http": "streamable_http", "streamable_http": "streamable_http", "sse": "sse", "stdio": "stdio"}
_FIELDS = {"streamable_http": ("url", "headers"), "sse": ("url", "headers"),
           "stdio": ("command", "args", "env", "cwd")}
_TYPE_KEYS = ("type", "transport")
_ENV_NOTICE = ("Environment values are saved as ordinary visible settings; keep secrets in Custom keys and "
               "select them with Environment from settings.")


def _unique_object(pairs: List[tuple]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON object key")
        out[key] = value
    return out


def _validate_patch(patch: Dict[str, Any]) -> None:
    """Reuse transport validators; never echo a URL, command or parser payload."""
    if patch["transport"] != "stdio":
        if not isinstance(patch.get("url"), str):
            raise ValueError("url must be a string")
        try:
            _validate_url(patch["url"])
        except ValueError:
            raise ValueError("url must be a valid permitted http:// or https:// address; "
                             "username/password credentials require Cyber mode") from None
        validate_headers(patch.get("headers"))
    else:
        _validate_stdio_command(patch.get("command"))
        _validate_stdio_args(patch.get("args"))
        validate_process_env(patch.get("env"))
        cwd = patch.get("cwd", "")
        if not isinstance(cwd, str) or "\x00" in cwd:
            raise ValueError("cwd must be a string without NUL")


def _translate(name: str, raw: Any) -> Dict[str, Any]:
    """One entry's display facts, problems and field patch, before draft matching."""
    entry: Dict[str, Any] = {"source_name": name, "server_id": canonical_server_id(name), "action": "",
                             "transport": "", "url": "", "command": "", "arg_count": 0, "header_names": [],
                             "env_names": [], "unsupported_keys": [], "problems": [], "warnings": [], "patch": {}}
    problems: List[str] = entry["problems"]
    if not entry["server_id"]:
        problems.append("the name has no usable server ID; use letters or digits")
    if not isinstance(raw, dict):
        problems.append("the entry must be an object")
        return entry
    known = {key for fields in _FIELDS.values() for key in fields}
    entry["unsupported_keys"] = sorted(set(raw) - known - set(_TYPE_KEYS))
    declared = {TRANSPORTS.get(raw[key]) if isinstance(raw[key], str) else None for key in _TYPE_KEYS if key in raw}
    if None in declared:
        problems.append("type must be one of: " + ", ".join(TRANSPORTS))
    elif len(declared) > 1:
        problems.append("type and transport name different transports")
    if "url" in raw and "command" in raw:
        problems.append("the entry has both url and command; keep one")
    if problems:
        return entry
    transport = declared.pop() if declared else ("stdio" if "command" in raw else "streamable_http")
    present = [key for key in raw if key in known]
    foreign = [key for key in present if key not in _FIELDS[transport]]
    if foreign:
        problems.append(f"{', '.join(foreign)} cannot be used with transport {transport}")
    required = "command" if transport == "stdio" else "url"
    if required not in raw:
        problems.append(f"transport {transport} needs {required}")
    nulls = [key for key in present if raw[key] is None]
    if nulls:
        problems.append("null is not a value for: " + ", ".join(nulls))
    patch = {"transport": transport, **{key: raw[key] for key in present}}
    if not problems:
        try:
            _validate_patch(patch)
        except ValueError as exc:
            problems.append(str(exc))
    entry["transport"] = transport
    if problems:
        return entry
    # URLs (query/path credentials included) and command strings are literal
    # configuration values too: only patches carry them, never display facts.
    entry.update(arg_count=len(patch.get("args") or []), header_names=list(patch.get("headers") or {}),
                 env_names=list(patch.get("env") or {}), patch=patch)
    if patch.get("env"):
        entry["warnings"].append(_ENV_NOTICE)
    return entry


def preview_import(text: str, draft: List[Any]) -> Dict[str, Any]:
    """Match translated entries to the current draft list; return the previewed actions."""
    try:
        document = json.loads(text, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"The text is not valid JSON (line {exc.lineno}, column {exc.colno}).", "entries": []}
    except (ValueError, RecursionError):
        return {"ok": False, "error": "The text is not valid JSON: duplicate JSON object key or excessive nesting.", "entries": []}
    if not isinstance(document, dict) or not isinstance(document.get("mcpServers"), dict):
        return {"ok": False, "error": "Expected a JSON object with a top-level mcpServers object.", "entries": []}
    entries = [_translate(name, raw) for name, raw in document["mcpServers"].items()]
    imported = [entry["server_id"] for entry in entries]
    claimants: Dict[str, List[int]] = {}
    for index, server in enumerate(draft):
        if raw_server_id(server):
            claimants.setdefault(raw_server_id(server), []).append(index)
    for entry in entries:
        server_id, problems = entry["server_id"], entry["problems"]
        targets = claimants.get(server_id, []) if server_id else []
        if server_id and imported.count(server_id) > 1:
            problems.append(f"several imported names resolve to server ID {server_id!r}; rename all but one")
        if len(targets) > 1:
            problems.append(f"several draft servers already use server ID {server_id!r}; make their IDs distinct first")
        target = draft[targets[0]] if len(targets) == 1 else None
        current = str((target or {}).get("transport") or "streamable_http").strip().lower()
        if target is not None and not problems and current != entry["transport"]:
            problems.append("the draft server uses a different transport; a transport is not changed by import — "
                            "remove that server first, then import it again")
        if problems:
            entry["patch"] = {}
            continue
        if target is None:
            entry["action"] = "add"
            entry["patch"] = {"id": server_id, "name": entry["source_name"], "enabled": False, **entry["patch"]}
            continue
        entry.update(action="update", index=targets[0])
        legacy = str(target.get("auth_header") or "Authorization").strip() or "Authorization"
        if str(target.get("auth_token") or "").strip() and any(
                name.casefold() == legacy.casefold() and value for name, value in entry["patch"].get("headers", {}).items()):
            entry["warnings"].append(
                "This server's legacy Auth token sets the same header. Clear the legacy Auth token before saving; "
                "until then the server reports a configuration error.")
    return {"ok": True, "entries": entries, "ignored_keys": sorted(set(document) - {"mcpServers"})}
