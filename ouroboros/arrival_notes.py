"""At my human's arrival, the host names the working-view notes in view: append only, facts only.

A drain that delivers a row of theirs into a work's transcript (their words, a card answer or press, a closure of
theirs, their words relayed or carried to this work) closes the last such row with host lines, as part of that
delivered message: for a press without words, their last own words in this work (``answer_facts.last_own_words``);
and, while the view holds notes the mind wrote through ``compact_context`` folds (capsules of actor authorship),
one line naming each by the unit id ``compact_context`` inspects and ``replace_note_ids`` takes, its write time
(``written_at`` in the capsule metadata, else "not recorded") and its first line. No earlier message changes; every
note in view precedes the arrival, so nothing is joined by time, and nothing is judged, asked or instructed.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Sequence


def _written(value: Any) -> str:
    """A capsule's ``written_at`` as ``YYYY-MM-DD HH:MM UTC``; "not recorded" when absent or unreadable."""
    try:
        stamp = datetime.datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return "not recorded"
    if stamp.tzinfo is not None:
        stamp = stamp.astimezone(datetime.timezone.utc)
    return stamp.strftime("%Y-%m-%d %H:%M UTC")


def notes_in_view(messages: Sequence[Any]) -> List[Dict[str, str]]:
    """Each note the mind authored in this view, in transcript order: ``{id, written, first_line}``."""
    from ouroboros.context_compaction import _capsule_metadata, _first_line, _unit_from_slice

    notes = []
    for index, message in enumerate(messages):
        meta = _capsule_metadata(message)[1] if isinstance(message, dict) else None
        if not meta or meta.get("authorship") != "actor":
            continue
        unit = _unit_from_slice(messages, index, index, trace_refs_by_tool_call_id={})
        if unit is not None:
            notes.append({"id": unit.unit_id, "written": _written(meta.get("written_at")),
                          "first_line": _first_line(message)})
    return notes


def notes_line(messages: Sequence[Any]) -> str:
    """The host line listing the notes in view, or "" when the view holds none."""
    notes = notes_in_view(messages)
    return ("[Host fact: your working-view notes in view now: " + "; ".join(
        f"{note['id']}, written {note['written']}, «{note['first_line']}»" for note in notes) + "]") if notes else ""


def mark_arrival(ctx: Any, messages: List[Dict[str, Any]], rows: Sequence[int], *, pressed: bool = False) -> None:
    """Append the arrival's host lines to the last row of my human's this drain delivered (``rows``: positions of the
    rows it delivered, typed at delivery and never merged); a delivery that warrants no line changes nothing."""
    from ouroboros.answer_facts import last_own_words

    lines = [line for line in ((last_own_words(ctx) if pressed and ctx is not None else ""), notes_line(messages))
             if line]
    if not lines or not rows:
        return
    row, text = messages[rows[-1]], "\n".join(lines)
    content = row.get("content")
    row["content"] = ([*content, {"type": "text", "text": "\n" + text}] if isinstance(content, list)
                      else f"{content}\n{text}")
