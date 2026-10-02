"""The mind's revisable remembering guidance, on the ordinary knowledge shelf.

The note is not another policy store: the existing knowledge writer, revisions,
history and index own it. Memory helpers read its complete current text at the
start of their operation, so a correction can change subsequent remembering.
"""

from __future__ import annotations

import json
from pathlib import Path

REMEMBERING_TOPIC = "remembering"


def remembering_guidance(data_root: Path) -> str:
    """Return complete authored guidance and its source, or an explicit read gap."""
    from ouroboros.knowledge import read_knowledge_note, resolve_knowledge_address

    try:
        note = read_knowledge_note(resolve_knowledge_address(data_root, REMEMBERING_TOPIC, "global"))
    except FileNotFoundError:
        return ""
    except (OSError, ValueError, UnicodeError):
        return "\n## Remembering guidance\nThe current global remembering note is unavailable; do not infer its contents.\n"
    return ("\n## Remembering guidance maintained through the shared knowledge tools\n"
            "Apply this revisable guidance to the supplied evidence. Preserve the distinction between "
            "the acting mind's words and your helper interpretation; this note is not new owner authority.\n"
            + json.dumps(note.source_ref(), ensure_ascii=False) + "\n" + note.text + "\n")


def memory_wake_changes(data_root: Path, boundary, gaps: set):
    """Source-addressed changes since an accepted wake, not an audit or importance judgment."""
    from ouroboros.chronicle_store import ChronicleStore

    empty = {"sequence": 0, "record_id": None}
    try:
        if not (data_root / "memory/chronicle/records.jsonl").exists():
            if boundary and boundary.get("sequence"):
                raise ValueError("accepted memory source is missing")
            return [], empty, None  # Empty installs acquire no memory directory or index.
        rows, current = ChronicleStore(data_root).observation_snapshot(boundary)
        window = {"lower": (boundary or {}).get("sequence", current["sequence"]),
                  "upper": current["sequence"], "last_record_id": current["record_id"], "basis": "accepted_sequence" if boundary is not None else "initial_baseline"}
        events = []
        for row in rows:
            fact = {key: row[key] for key in ("id", "sequence", "kind", "room_id", "target_id", "accepted",
                    "status", "visibility", "published_progress", "target_fits") if key in row}
            fact["author"] = row.get("author") or {"kind": "unknown"}
            fact["read"] = {"tool": "memory_read", "arguments": {"node_id": row["id"]}}
            if row.get("target_id"):
                fact["read_original"] = {"tool": "memory_read", "arguments": {"node_id": row["target_id"]}}
            events.append(("memory_change", None, "- memory change: " + json.dumps(fact, ensure_ascii=False)))
        return events, current, window
    except Exception as exc:
        gaps.add(f"memory changes unreadable: {type(exc).__name__}; accepted sequence retained")
        return [], boundary, {"basis": "unreadable", "lower": (boundary or {}).get("sequence")}
