"""A Project registry an older body lost: the synthetic pre-upgrade state the read-only archive is tested on (#3294).

Every file is written as bytes, never through the registry's or the bindings' writers, so no modern registry,
lock or backup ever exists for the storage layer to restore or reconcile from, before or after the loss:
``lost_registry`` leaves the old zero-byte commit witness with no ``projects.json`` and no lock (checked, not
cleaned), ``write_registry`` a registry for a case that needs one readable or malformed, ``write_bindings`` the
task bindings. Shared by the reader's unit tests and its browser scenario; it starts no writer of its own.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any, Dict, Iterable

from ouroboros.contracts.chat_id_policy import project_chat_id

TS = "2026-11-17T10:00:00Z"


def lost_registry(root: Any) -> pathlib.Path:
    """The old commit witness (zero bytes) beside no registry and no lock: a loss no exact copy can answer."""
    state = pathlib.Path(root) / "state"
    state.mkdir(parents=True, exist_ok=True)
    for name in ("projects.json", "projects.json.lock"):
        assert not (state / name).exists(), f"the synthetic loss starts before any registry writer: {name}"
    witness = state / "projects.json.committed"
    witness.write_bytes(b"")
    return witness


def project_row(pid: str, name: str, **extra: Any) -> Dict[str, Any]:
    """One registry row as the registry wrote it, in the project's own chat."""
    return {"id": pid, "name": name, "chat_id": project_chat_id(pid), "working_dir": "", "origin": "owner_ui",
            "created_at": TS, "last_active_at": TS, "lifecycle": "active", "routing_generation": 0,
            "routing_incarnation": hashlib.sha256(pid.encode("utf-8")).hexdigest()[:32], "visible_revision": 0,
            "pinned": False, "delete_error": "", **extra}


def write_registry(root: Any, rows: Iterable[Any]) -> pathlib.Path:
    """``state/projects.json`` holding exactly ``rows`` (any shape: a malformed row stays malformed)."""
    path = pathlib.Path(root) / "state" / "projects.json"
    path.write_text(json.dumps({"projects": list(rows), "_schema_version": 2}), encoding="utf-8")
    return path


def binding(task_id: str, pid: str, *, origin: str = "", origin_id: str = "", origin_ts: str = TS,
            chat_id: Any = None) -> Dict[str, Any]:
    """A task's binding row; ``origin`` the owner's Main message it names by value (chat, id, time, text hash)."""
    row: Dict[str, Any] = {"task_id": task_id, "project_id": pid,
                           "project_chat_id": project_chat_id(pid) if chat_id is None else chat_id, "bound_at": TS}
    if origin:
        row["source_ref"] = {"chat_id": 1, "client_message_id": origin_id, "ts": origin_ts,
                             "text_sha256": hashlib.sha256(origin.encode("utf-8")).hexdigest()}
        row["source_text"] = origin
    else:
        row["origin_absent"] = "system"
    return row


def write_bindings(root: Any, rows: Iterable[Dict[str, Any]]) -> pathlib.Path:
    """``state/project_task_bindings.json`` holding ``rows`` by their task ids."""
    path = pathlib.Path(root) / "state" / "project_task_bindings.json"
    path.write_text(json.dumps({"bindings": {row["task_id"]: row for row in rows}, "_schema_version": 1}),
                    encoding="utf-8")
    return path
