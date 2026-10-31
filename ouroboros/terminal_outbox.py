"""Durable terminal-outbox records, shared by delivery and one-time startup migrations.

Transport and chat disclosure belong to supervisor. This owner keeps delivered,
pending and retired payloads under the same registry lock; retirement proves no send.
"""
from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Optional

from ouroboros.utils import update_json_locked, utc_now_iso

_SCHEMA_VERSION = 4
_REGISTRY_CAP = 512


def _registry_path(drive_root: Any) -> pathlib.Path:
    root = pathlib.Path(drive_root) / "state"
    root.mkdir(parents=True, exist_ok=True)
    return root / "terminal_deliveries.json"


def _mapping_rows(current: Dict[str, Any], key: str, *, strict: bool = False) -> Dict[str, Any]:
    """Missing/None mappings mean empty; strict mutations preserve malformed forensic bytes."""
    value = current.get(key)
    if isinstance(value, dict):
        return dict(value)
    if strict and value is not None:
        raise ValueError(f"terminal-delivery registry '{key}' is malformed (not an object)")
    return {}


def _registry_document(current: Dict[str, Any], rows: List[str], pending: Dict[str, Any],
                       **maps: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the receipts, exhausted (replay/capacity limit) and superseded payloads not passed in ``maps``
    untouched. Their Done-card disposition retires only on a real receipt of that id."""
    document = {"schema_version": _SCHEMA_VERSION, "delivered": rows[-_REGISTRY_CAP:], "pending": pending}
    for key in ("receipts", "exhausted", "superseded"):
        value = maps[key] if key in maps else _mapping_rows(current, key, strict=True)
        if value:
            document[key] = dict(list(value.items())[-_REGISTRY_CAP:])
    return document


def _read_registry(drive_root: Any) -> Dict[str, Any]:
    from ouroboros.utils import read_json_dict
    path = pathlib.Path(drive_root) / "state" / "terminal_deliveries.json"
    data = read_json_dict(path) if path.is_file() else {}
    if data is None:
        raise ValueError("registry file is malformed or is not an object")
    return data


def _delivered_rows(current: Dict[str, Any], *, strict: bool = False) -> List[str]:
    """Read delivered ids; strict reads reject malformed containers or entries.

    Mutators refuse corruption without overwriting the file. Fail-soft readers
    keep only a valid list; corruption disclosure belongs to their caller."""
    delivered = current.get("delivered")
    if isinstance(delivered, list):
        if strict and any(not isinstance(item, str) for item in delivered):
            raise ValueError("terminal-delivery registry 'delivered' holds non-string entries")
        return [str(item) for item in delivered]
    if strict and delivered is not None:
        raise ValueError("terminal-delivery registry 'delivered' is malformed (not a list)")
    return []


def _dispatch_attempts(row: Dict[str, Any]) -> Optional[int]:
    """Prospective budget only: absence records no new entries; invalid is unknown."""
    value = row.get("dispatch_attempts", 0)
    return value if type(value) is int and value >= 0 else None


def _exhausted_row(row: Dict[str, Any], *, reason: str) -> Dict[str, Any]:
    """The registry's exhausted disposition of one owed row: its payload plus how it ended."""
    return {**row, "exhausted_at": utc_now_iso(), "reason": reason,
            "attempts": _dispatch_attempts(row) if "dispatch_attempts" in row else None,
            "attempts_basis": "recorded_replay_dispatch_entries", "preserved_path": ""}


def retire_producer(root: Any, *, delivery_prefix: str, system_type: str) -> int:
    """Retire a removed producer's unsent rows once at boot. Preserve payloads, sent facts and real receipts.
    Errors propagate so the migration cannot stamp a transition that did not commit."""
    path = pathlib.Path(root) / "state" / "terminal_deliveries.json"
    if not path.is_file():
        return 0
    retired = []

    def mutate(current):
        retired.clear()
        delivered = _delivered_rows(current, strict=True)
        pending = _mapping_rows(current, "pending", strict=True)
        exhausted = _mapping_rows(current, "exhausted", strict=True)
        for did, row in list(pending.items()):
            if (isinstance(row, dict) and did not in delivered and not row.get("emitted")
                    and (did.startswith(delivery_prefix) or row.get("system_type") == system_type)):
                exhausted[did] = _exhausted_row(pending.pop(did), reason="producer_retired")
                retired.append(did)
        return _registry_document(current, delivered, pending, exhausted=exhausted) if retired else None

    update_json_locked(path, mutate, strict_existing_dict=True)
    return len(retired)
