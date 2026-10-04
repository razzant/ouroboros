"""Strict reading of the old dialogue writer's frozen cursor, memory/dialogue_meta.json.

The writer is gone; its cursor and the knowledge nominations it never published
stay in that file. ``chronicle_import`` reads them once: every pending
nomination becomes a global mark the mind releases, and an unreadable cursor
becomes a ``cursor_gap``. A malformed obligation list is refused, never read
as empty, so no nomination silently disappears.
"""
from __future__ import annotations

import json
from typing import Any


KEY = "pending_knowledge_nominations"


class DialogueMetaUnreadable(ValueError):
    """Existing cursor or nomination obligations cannot be safely interpreted."""


def parse_meta(raw: bytes) -> dict[str, Any]:
    """The cursor's bytes as an object; unreadable bytes are never an empty cursor.

    This file holds the only index of the pending nominations; a permissive JSON
    read would report them as absent. Reject duplicate keys as well: the second
    copy of an obligation field cannot silently replace the first.
    """
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate dialogue meta key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=unique_pairs)
    except (UnicodeError, ValueError) as exc:
        raise DialogueMetaUnreadable(f"Dialogue meta unreadable: {type(exc).__name__}: {exc}") from exc
    if not isinstance(value, dict):
        raise DialogueMetaUnreadable("Dialogue meta must be a JSON object")
    _pending(value)  # Corrupt obligations are refused, never read as none.
    return value


def _pending(meta: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = meta.get(KEY, [])
    if not isinstance(rows, list) or any(not isinstance(row, dict) or not isinstance(row.get("id"), str)
                                          for row in rows):
        raise DialogueMetaUnreadable("Unreadable nomination obligations; refusing to replace their bytes")
    if len({row["id"] for row in rows}) != len(rows):
        raise DialogueMetaUnreadable("Duplicate nomination obligation IDs")
    return {row["id"]: row for row in rows}

