"""How my story shows the old memory a helper retold before the update, and how the floor shortens it.

The helper's retelling comes first under its heading, then the records a transfer brought
(``chronicle_import``) under theirs, each naming its author and how it came, in place of
"retold" (``memory_inventory.author_byline``, ``transfer_words``).
An unfolded retold record is whole in the story, whatever its block and whoever reads it:
its id, period, room and length as a header, then its words. A record whose words the
capture did not keep is one line that says what its address holds: the room, the period,
how many of that room's chat rows it retells (where none is established, the old writer's
own message count) and the length of the retelling, then the ``memory_read`` call that
reads it. A journal gap is named with its detail. Under the physical floor (F3) a room's
records, whole ones included, are one line for the helper's retelling and one for its
transfers, each under its heading: the room, its whole period, their total length and every
id. A record a selected account tells in its place (``told_by``) is not among these lines: it
stands under the account (``memory_view_account``).

Only facts of the record, never a reason to read it. Nothing here reads a file.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping

INDENT = "  "
TRANSFER_LABEL = "Transferred memory"  # a transfer's label in the room-less ``legacy`` room, never "Unknown provenance"
HEADINGS = {False: "### Old memory retold by a helper before the update (not lived; exact sources readable by id)",
            True: "### Memory transferred from its source (author and transfer named per record; exact sources "
                  "readable by id)"}


def indented(text: Any) -> str:
    """A record's words, each line indented: its own ``## …`` lines never become sections."""
    return "\n".join(INDENT + line if line else line for line in str(text or "").split("\n"))


def retelling_status(status, refusals, part):
    """Progress of the old retelling, including the distinct response source of each refused fold."""
    if not status or status.get("folded", 0) >= status.get("total", 0):
        return []
    lines = part([f"Story status: the helper retelling is folded {status['folded']} of {status['total']} blocks; "
                  f"{status['open_records']} retold records are still open ({status['open_rows']} rows, "
                  f"{status['open_chars']} chars; future folds use helper route {status['helper_route']})."], "story_status")
    for refusal in refusals:
        answer = (f"its answer: read_file(root='runtime_data', path='{refusal['path']}')" if refusal.get("path")
                  else "its answer was not retained")
        lines += part([f"A helper could not fold: {refusal['label']}; {refusal['period']}; {refusal['kind']}; {answer}"],
                      "refusal", [refusal])
    return lines


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def retold_size(entry: Mapping[str, Any]) -> str:
    """``4 rows retold in 1234 chars``; without rows of its room, the old writer's message count if it kept one;
    a transfer's author and coming, then its length."""
    rows, said, chars = entry.get("rows"), entry.get("messages"), entry.get("chars")
    if entry.get("transfer"):
        return "; ".join([str(entry.get("provenance") or "transferred"), *([f"{chars} chars"] if chars else [])])
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
        provenance = f"; {retold_size(entry)}" if entry.get("transfer") else ""
        return f"- memory gap: {entry['label']}; {entry['period']}; {entry['gap']}{provenance}; {read}"
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


def _tellings(entries: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """A room's records by who told them, as ``HEADINGS`` part them: the helper's retelling, then the transfers."""
    return [told for moved in (False, True) if (told := [e for e in entries if bool(e.get("transfer")) is moved])]


def _room_line(entries: List[Dict[str, Any]]) -> str:
    merged = []
    for key, source, note in (("span", "source_period", ""),
                              ("stated_span", "stated_period", " (stated by its source)")):
        dated = [e for e in entries if e.get(key)]
        if dated:
            incomplete = " (incomplete)" if any((e.get(source) or {}).get("incomplete") for e in dated) else ""
            merged.append(f"{min(e[key][0] for e in dated)} → {max(e[key][1] for e in dated)}{incomplete}{note}")
    period = "; ".join(merged + sorted({entry["period"] for entry in entries
                                        if not entry.get("span") and not entry.get("stated_span")}))
    chars, moved = sum(entry.get("chars") or 0 for entry in entries), bool(entries[0].get("transfer"))
    return (f"- {TRANSFER_LABEL if moved else entries[0]['label']}; {period}; "
            f"{_count(len(entries), 'transferred record' if moved else 'retold record')}"
            + (f" in {chars} chars" if chars else "") + ": "
            + ", ".join(entry["id"] for entry in entries) + "; memory_read(node_id=<id>) reads each")


def room_pointer(entries: List[Dict[str, Any]]) -> str:
    """A room's retold records in the floor's form (F3): the helper's retelling and the room's transfers each one line,
    the room (a transfer's as ``TRANSFER_LABEL``), its whole period (its rows', then the span its transfers' sources
    state), their length, every id."""
    return "\n".join(map(_room_line, _tellings(entries)))


def retold_lines(retold: List[Dict[str, Any]], rooms: Any, *, part=None) -> List[str]:
    """The retold records in story order under their headings (``HEADINGS``: a transfer's after the helper's); in a
    room the floor took, its helper retelling and its transfers are each one line where their first record stood."""
    firsts = {told[0]["id"]: told for room, entries in pointer_rooms(retold).items() if room in rooms
              for told in _tellings(entries)}
    lines, heading = [], None
    for entry in retold:
        if entry.get("gap") or str(entry["room_id"]) not in rooms:
            line, sources = retold_record(entry), [entry]
            form = "gap" if entry.get("gap") else "whole" if entry.get("text") else "address"
        elif entry["id"] in firsts:
            sources, form = firsts[entry["id"]], "count"
            line = _room_line(sources)
        else:
            continue
        moved = bool(entry.get("transfer"))
        if moved is not heading:
            drawn = ["", HEADINGS[moved]]
            lines += part(drawn) if part else drawn
            heading = moved
        lines += part([line], "legacy", sources, form) if part else [line]
    return lines
