"""Versioned per-run wait checkpoints with task-scoped delivery facts.

Readers migrate singleton records in memory. Writers merge only fields their
view changed, under the actor decision lock, so concurrent run waits and ACKs
cannot erase each other's durable sources. Malformed authority never becomes
an empty checkpoint.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from ouroboros import delegate_custody as custody
from ouroboros.utils import atomic_write_json, utc_now_iso

TASK_FIELDS = ("unknown_provider_hold", "coordination_cursor", "mailbox_acknowledged_ids",
               "mailbox_acknowledged_attempt_key")


def normalize(data: Any) -> dict:
    if not isinstance(data, dict):
        raise ValueError("supervision checkpoint is not an object")
    schema = data.get("schema", 1)
    if schema not in (1, 2):
        raise ValueError("unknown supervision schema")
    if schema == 2:
        if not isinstance(data.get("runs"), dict) or any(
                not isinstance(value, dict) or not key for key, value in data["runs"].items()):
            raise ValueError("invalid per-run supervision checkpoint")
        return deepcopy(data)
    result = {"schema": 2, "runs": {}, **task_facts(data)}
    run_id = str(data.get("run_id") or "")
    if run_id:
        result["runs"][run_id] = {key: deepcopy(value) for key, value in data.items()
                                  if key not in (*TASK_FIELDS, "schema", "run_id")}
    return result


def task_facts(data: Any) -> dict:
    return {key: deepcopy(data[key]) for key in TASK_FIELDS if isinstance(data, dict) and key in data}


def read(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        data = {}
    return normalize(data)


def run_view(data: Any, run_id: str) -> dict:
    data = normalize(data)
    view = {"schema": 2, "run_id": str(run_id), "journal_cursor": 0,
            **deepcopy(data["runs"].get(str(run_id), {})), **task_facts(data)}
    view["_base"] = deepcopy(view)
    return view


def save_view(path: Path, drive: Any, task_id: str, view: dict) -> None:
    """Merge this view's edits, preserving siblings and unmodified task facts."""
    with custody.actor_decision_lock(drive, task_id):
        data = read(path)
        run_id = str(view.get("run_id") or "")
        target = data["runs"].setdefault(run_id, {}) if run_id else {}
        base = view.get("_base") or {}
        pending = target.get("pending_wake")
        if (view.get("pending_wake") and pending and pending != base.get("pending_wake")
                and pending != view["pending_wake"]):
            # Another observer published first. Return its exact durable wake;
            # this observer's cursor and new wake have not been delivered.
            view.clear()
            view.update(run_view(data, run_id))
            return
        mail_attempt = data.get("mailbox_acknowledged_attempt_key")
        for key in set(base) | set(view):
            if key in {"_base", "schema", "run_id", "updated_at"}:
                continue
            if key in base and key in view and base[key] == view[key]:
                continue
            dest = data if key in TASK_FIELDS else target
            if key not in view:
                if dest.get(key) == base.get(key):
                    dest.pop(key, None)
                continue
            value = deepcopy(view[key])
            if key == "journal_cursor":
                value = max(int(dest.get(key) or 0), int(value or 0))
            elif key == "interaction_acknowledged_ids" or (key == "mailbox_acknowledged_ids"
                    and mail_attempt == view.get("mailbox_acknowledged_attempt_key")):
                value = sorted(set(dest.get(key) or []) | set(value or []))
            elif key == "coordination_cursor":
                value = merge_cursor(dest.get(key) or {}, base.get(key) or {}, value)
            elif key == "pending_wake" and not value and dest.get(key) != base.get(key):
                continue  # A new wake was published after this ACK's read.
            dest[key] = value
        data["updated_at"] = utc_now_iso()
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, data)
        view["_base"] = deepcopy({key: value for key, value in view.items() if key != "_base"})
    from ouroboros.maintenance_retirement import artifact_changed
    artifact_changed(path.parent.parent.parent)


def merge_cursor(current: dict, before: dict, after: dict) -> dict:
    result = deepcopy(current)
    for key, value in after.items():
        if before.get(key) == value:
            continue
        if isinstance(value, dict):
            result[key] = merge_cursor(current.get(key) or {}, before.get(key) or {}, value)
        elif isinstance(value, list):
            result[key] = list(dict.fromkeys([*(current.get(key) or []), *value]))
        else:
            result[key] = value
    return result
