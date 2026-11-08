"""One-way preview of a pasted MCP client document as MCP Settings draft patches.

Two common client shapes are read: a top-level ``mcpServers`` object (Claude
Code, Cursor, portable VS Code files) or a top-level ``servers`` object (VS
Code workspace files); a document with both is ambiguous and refused. Entry
types ``http``, ``streamable_http``, ``streamable-http``, ``sse`` and ``stdio``
are understood. Nothing else happens here: no process starts, no address is
contacted, nothing is saved and no ``${...}`` reference is expanded — such text
stays literal, and the entry names the fields that carry it.

Each entry becomes an ``add`` (a new, disabled server) or an ``update`` of the
ONE draft server with the same canonical identity (``mcp_client.canonical_server_id``)
and the same transport. A field the entry does not name stays as it is. A named
``headers`` object is that server's complete header list: it replaces every
current header, the legacy ``auth_header``/``auth_token`` pair included (cleared
in the same patch), and ``{}`` clears them all. ``***set***`` is the Settings
marker for "keep the saved value", so a header with exactly that value is a
problem, never a literal. Malformed JSON, duplicate keys, wrong types and
ambiguous identities are problems that keep an entry out of the result rather
than truncating it. The loader's field validators judge imported fields without
loading settings or connecting.

A stdio entry may also name Ouroboros's own ``task_session`` and ``browser_policy``:
literal booleans under the loader's rule, judged on the entry alone because the draft
list carries no flags. So ``browser_policy: true`` needs ``task_session: true`` beside
it, and ``task_session: false`` clears the draft's policy in the same patch.

The draft list the caller sends carries identities, transports and header names
only (``MCPImportDraftServer``); values and addresses stay in the browser.
Display facts carry names and counts; ``patch`` carries the pasted values back
to the client draft, which the owner then saves through the ordinary Settings
writer.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from ouroboros.mcp_client import (
    canonical_server_id, raw_server_id, _validate_url,
    _validate_call_lifetime, _validate_stdio_args, _validate_stdio_command,
)
from ouroboros.mcp_headers import HEADER_VALUE_PLACEHOLDER, validate_headers
from ouroboros.workspace_executor import validate_process_env

WRAPPERS = ("mcpServers", "servers")
TRANSPORTS = {"http": "streamable_http", "streamable_http": "streamable_http", "streamable-http": "streamable_http",
              "sse": "sse", "stdio": "stdio"}
_LIFETIME_FLAGS = ("task_session", "browser_policy")
_FIELDS = {"streamable_http": ("url", "headers", *_LIFETIME_FLAGS), "sse": ("url", "headers", *_LIFETIME_FLAGS),
           "stdio": ("command", "args", "env", "cwd", *_LIFETIME_FLAGS)}
_TYPE_KEYS = ("type", "transport")
_SUBSTITUTION = re.compile(r"\$\{[^}]*\}")
_ENV_NOTICE = ("Environment values are saved as ordinary visible settings; keep secrets in Custom keys and "
               "select them with Environment from settings.")
_SHAPE_ERROR = "Expected a JSON object with a top-level mcpServers or servers object."


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
        reserved = [name for name, value in validate_headers(patch.get("headers")).items()
                    if value == HEADER_VALUE_PLACEHOLDER]
        if reserved:
            raise ValueError(f"header {', '.join(reserved)} has the value {HEADER_VALUE_PLACEHOLDER}, which Settings "
                             "reserves for keeping a saved value; paste the real value")
    else:
        _validate_stdio_command(patch.get("command"))
        _validate_stdio_args(patch.get("args"))
        validate_process_env(patch.get("env"))
        cwd = patch.get("cwd", "")
        if not isinstance(cwd, str) or "\x00" in cwd:
            raise ValueError("cwd must be a string without NUL")
    flags = {key: patch.get(key, False) for key in _LIFETIME_FLAGS}
    for key, value in flags.items():
        if not isinstance(value, bool):  # the loader's looser reading of old saved text is not for new input
            raise ValueError(f"{key} must be true or false")
    _validate_call_lifetime(patch["transport"], **flags)


def _substitution_fields(patch: Dict[str, Any]) -> List[str]:
    """Names of the fields whose values contain ``${...}``; values are never echoed."""
    found = [key for key in ("url", "command", "cwd") if _SUBSTITUTION.search(str(patch.get(key) or ""))]
    if any(_SUBSTITUTION.search(item) for item in patch.get("args") or []):
        found.append("args")
    for field, label in (("env", "environment"), ("headers", "header")):
        found += [f"{label} {name}" for name, value in (patch.get(field) or {}).items() if _SUBSTITUTION.search(value)]
    return found


def _translate(name: str, raw: Any) -> Dict[str, Any]:
    """One entry's display facts, problems and field patch, before draft matching."""
    entry: Dict[str, Any] = {"source_name": name, "server_id": canonical_server_id(name), "action": "",
                             "transport": "", "arg_count": 0, "header_names": [], "env_names": [],
                             "unsupported_keys": [], "problems": [], "warnings": [], "patch": {}}
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
    literal = _substitution_fields(patch)
    if literal:
        entry["warnings"].append(f"{', '.join(literal)}: ${{...}} is kept as written. Nothing is expanded, so the "
                                 "server receives those characters unless you replace them.")
    if patch.get("env"):
        entry["warnings"].append(_ENV_NOTICE)
    return entry


def _header_changes(entry: Dict[str, Any], target: Dict[str, Any]) -> None:
    """Disclose which current header names an update replaces, removes or keeps."""
    names = [name for name in target.get("header_names") or [] if isinstance(name, str) and name]
    legacy = target.get("legacy_header") if isinstance(target.get("legacy_header"), str) else ""
    if legacy and legacy.casefold() not in {name.casefold() for name in names}:
        names.append(legacy)
    if legacy:
        entry["legacy_header"] = legacy
    if target.get("unreadable_headers") is True:
        entry["unreadable_headers"] = True
    imported = entry["patch"].get("headers")
    if imported is None:
        entry["kept_header_names"] = names
        return
    incoming = {name.casefold() for name in imported}
    current = {name.casefold() for name in names}
    entry["header_changes"] = {
        "added": [name for name in imported if name.casefold() not in current],
        "replaced": [name for name in names if name.casefold() in incoming],
        "removed": [name for name in names if name.casefold() not in incoming],
    }
    # The file's list is the whole list: the old single token goes with it, atomically. Cleared even
    # when the draft no longer lists it: a row removed unsaved still leaves the saved mask on the server.
    entry["patch"].update(auth_header="Authorization", auth_token="")


def preview_import(text: str, draft: List[Any]) -> Dict[str, Any]:
    """Match translated entries to the current draft list; return the previewed actions."""
    try:
        document = json.loads(text, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"The text is not valid JSON (line {exc.lineno}, column {exc.colno}).", "entries": []}
    except (ValueError, RecursionError):
        return {"ok": False, "error": "The text is not valid JSON: duplicate JSON object key or excessive nesting.", "entries": []}
    wrappers = [key for key in WRAPPERS if isinstance(document, dict) and key in document]
    if len(wrappers) > 1:
        return {"ok": False, "error": "The document has both mcpServers and servers; keep one of them.", "entries": []}
    if not wrappers or not isinstance(document[wrappers[0]], dict):
        return {"ok": False, "error": _SHAPE_ERROR, "entries": []}
    wrapper = wrappers[0]
    entries = [_translate(name, raw) for name, raw in document[wrapper].items()]
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
            entry["warnings"] = []
            continue
        if target is None:
            entry["action"] = "add"
            entry["patch"] = {"id": server_id, "name": entry["source_name"], "enabled": False, **entry["patch"]}
            continue
        entry.update(action="update", index=targets[0])
        if entry["transport"] != "stdio":
            _header_changes(entry, target)
        elif entry["patch"].get("task_session") is False:
            entry["patch"].setdefault("browser_policy", False)
    return {"ok": True, "wrapper": wrapper, "entries": entries, "ignored_keys": sorted(set(document) - {wrapper})}
