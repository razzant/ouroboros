"""How my story shows the old memory a helper retold before the update, and how the floor shortens it.

An unfolded retold record is whole in the story, whatever its block and whoever reads it:
its id, period, room and length as a header, then its words. A record whose words the
capture did not keep is one line that says what its address holds: the room, the period,
how many of that room's chat rows it retells (where none is established, the old writer's
own message count) and the length of the retelling, then the ``memory_read`` call that
reads it. A journal gap is named with its detail. Under the physical floor (F3) a room's
records, whole ones included, are one line: the room, its whole period, their total length
and every id. A record a selected account tells in its place (``told_by``) is not among
these lines: it stands under the account (``memory_view_account``).

Only facts of the record, never a reason to read it. Nothing here reads a file.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping

INDENT = "  "


def indented(text: Any) -> str:
    """A record's words, each line indented: its own ``## …`` lines never become sections."""
    return "\n".join(INDENT + line if line else line for line in str(text or "").split("\n"))


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def retold_size(entry: Mapping[str, Any]) -> str:
    """``4 rows retold in 1234 chars``; without rows of its room, the old writer's message count if it kept one."""
    rows, said, chars = entry.get("rows"), entry.get("messages"), entry.get("chars")
    if rows:
        held = _count(rows, "row") + " retold"
    elif type(said) is int and said > 0:
        held = _count(said, "message") + " retold (the old writer's count)"
    else:
        held = ("no row of this room in that period; " if rows == 0 else "") + "retold"
    return held + (f" in {chars} chars" if chars else "")


def pointer_line(entry: Mapping[str, Any]) -> str:
    read = f"memory_read(node_id='{entry['id']}')"
    if entry.get("gap"):
        return f"- memory gap: {entry['label']}; {entry['period']}; {entry['gap']}; {read}"
    return f"- {entry['label']}; {entry['period']}; {retold_size(entry)}; {read}"


def retold_record(entry: Mapping[str, Any]) -> str:
    """A retold record as the full story shows it: whole when the capture kept its words, else its pointer line."""
    if entry.get("gap") or not entry.get("text"):
        return pointer_line(entry)
    return (f"#### {entry['id']} — {entry['period']} — {entry['label']} — {retold_size(entry)}\n"
            + indented(entry["text"]))


def pointer_rooms(story: Any) -> Dict[str, List[Dict[str, Any]]]:
    """The retold records the story shows itself, by room (a gap and a record told by an account stay out)."""
    rooms: Dict[str, List[Dict[str, Any]]] = {}
    for entry in story:
        if entry.get("kind") == "legacy" and not entry.get("gap") and not entry.get("told_by"):
            rooms.setdefault(str(entry["room_id"]), []).append(entry)
    return rooms


def room_pointer(entries: List[Dict[str, Any]]) -> str:
    """A room's retold records as one line: the room, its whole period, their length, every id (F3)."""
    spans = [entry["span"] for entry in entries if entry.get("span")]
    period = "; ".join(([f"{min(s[0] for s in spans)} → {max(s[1] for s in spans)}"] if spans else [])
                       + sorted({entry["period"] for entry in entries if not entry.get("span")}))
    chars = sum(entry.get("chars") or 0 for entry in entries)
    return (f"- {entries[0]['label']}; {period}; {_count(len(entries), 'retold record')}"
            + (f" in {chars} chars" if chars else "") + ": "
            + ", ".join(entry["id"] for entry in entries) + "; memory_read(node_id=<id>) reads each")


def retold_lines(retold: List[Dict[str, Any]], rooms: Any) -> List[str]:
    """The retold records in story order; a room the floor took is one line where its first record stood."""
    grouped, lines = pointer_rooms(retold), []
    for entry in retold:
        room = str(entry["room_id"])
        if entry.get("gap") or room not in rooms:
            lines.append(retold_record(entry))
        elif grouped[room][0] is entry:
            lines.append(room_pointer(grouped[room]))
    return lines
