"""Source-addressed pending knowledge nominations in dialogue_meta.json.

The history log carries full nomination bytes. This compact index makes an
unpublished nomination visible even when its summary block becomes an era.
A later unrelated success cannot discharge an older source identity.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


KEY = "pending_knowledge_nominations"


class DialogueMetaUnreadable(ValueError):
    """Existing cursor or nomination obligations cannot be safely interpreted."""


def load_meta(path: Path) -> dict[str, Any]:
    """An absent cursor is new; an unreadable existing cursor is not empty.

    This meta file now owns durable pending obligations. A permissive JSON read
    would erase them on the next consolidation. Reject duplicate keys as well:
    the second copy of an obligation field cannot silently replace the first.
    """
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate dialogue meta key: {key}")
            result[key] = value
        return result

    try:
        with path.open("r", encoding="utf-8") as source:
            value = json.load(source, object_pairs_hook=unique_pairs)
    except FileNotFoundError:
        # A dangling link is an existing, unreadable source, not a new cursor.
        try:
            path.lstat()
        except FileNotFoundError:
            return {}
        raise DialogueMetaUnreadable("Dialogue meta exists but cannot be read") from None
    except (OSError, UnicodeError, ValueError) as exc:
        raise DialogueMetaUnreadable(f"Dialogue meta unreadable: {type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise DialogueMetaUnreadable("Dialogue meta must be a JSON object")
    _pending(value)  # Refuse corrupt obligations before the first paid correction call.
    return value


def _pending(meta: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = meta.get(KEY, [])
    if not isinstance(rows, list) or any(not isinstance(row, dict) or not isinstance(row.get("id"), str)
                                          for row in rows):
        raise DialogueMetaUnreadable("Unreadable nomination obligations; refusing to replace their bytes")
    if len({row["id"] for row in rows}) != len(rows):
        raise DialogueMetaUnreadable("Duplicate nomination obligation IDs")
    return {row["id"]: row for row in rows}


def prepare(meta: dict[str, Any], source_id: str, batches: list[tuple[Any, list[Any]]]) -> list[str]:
    """Record every proposed entry before publication; return positional IDs."""
    pending = _pending(meta)
    ids: list[str] = []
    for block_index, (_block, entries) in enumerate(batches):
        for entry_index, entry in enumerate(entries):
            identifier = f"{source_id}:{block_index}:{entry_index}"
            ids.append(identifier)
            if identifier not in pending:
                pending[identifier] = {
                    "id": identifier,
                    "scope": str(entry.get("scope") or "default") if isinstance(entry, dict) else "invalid",
                    "topic": str(entry.get("topic") or "") if isinstance(entry, dict) else "",
                    "reason": "publication_pending",
                }
    meta[KEY] = list(pending.values())
    return ids


def settle(meta: dict[str, Any], ids: list[str], outcomes: list[dict[str, Any]]) -> None:
    """Only this exact source's successful entries retire; missing outcomes stay owed.

    A failed or unobserved entry never expires. A future source-grounded
    resolution must address its ID explicitly; same-topic later writes cannot.
    """
    pending = _pending(meta)
    for index, identifier in enumerate(ids):
        outcome = outcomes[index] if index < len(outcomes) else {}
        if outcome.get("ok") is True:
            pending.pop(identifier, None)
        elif identifier in pending:
            row = pending[identifier]
            row["reason"] = str(outcome.get("reason") or "outcome_missing")
            if isinstance(outcome.get("scope"), str):
                row["scope"] = outcome["scope"]
            if isinstance(outcome.get("topic"), str):
                row["topic"] = outcome["topic"]
    if pending:
        meta[KEY] = list(pending.values())
    else:
        meta.pop(KEY, None)
