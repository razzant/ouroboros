"""At my human's arrival, the host names the working-view notes in view: append only, facts only.

A drain that delivers a row of theirs into a work's transcript (their words, a card answer or press, a closure of
theirs, their words relayed or carried to this work) closes the last such row with host lines, as part of that
delivered message: for a press without words, their last own words in this work (``answer_facts.last_own_words``);
and, while the view holds notes the mind wrote through ``compact_context`` folds (capsules of actor authorship, the
notes the round facts count as its own), one line naming each as a fold receipt names it
(``compact_context.view_notes``): by the unit id ``compact_context`` inspects and ``replace_note_ids`` takes, its
write time (``written_at`` in the capsule metadata, else "not recorded") and its first line. No earlier message
changes; every note in view precedes the arrival, so nothing is joined by time, and nothing is judged, asked or
instructed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence

log = logging.getLogger(__name__)


def notes_line(messages: List[Dict[str, Any]]) -> str:
    """The host line listing the notes the mind wrote in this view, or "" when the view holds none."""
    from ouroboros.tools.compact_context import view_notes

    notes = view_notes(messages, own=True)
    return ("[Host fact: your working-view notes in view now: " + "; ".join(
        f"{note['id']}, written {note['written']}, «{note['first_line']}»" for note in notes) + "]") if notes else ""


def mark_arrival(ctx: Any, messages: List[Dict[str, Any]], rows: Sequence[int], *, pressed: bool = False) -> None:
    """Append the arrival's host lines to the last row of my human's this drain delivered (``rows``: positions of the
    rows it delivered, typed at delivery and never merged); a delivery that warrants no line changes nothing. Lines the
    host cannot compute are left out and logged: the delivered row stands, as for the surface note beside it."""
    from ouroboros.answer_facts import last_own_words

    try:
        lines = [line for line in ((last_own_words(ctx) if pressed and ctx is not None else ""), notes_line(messages))
                 if line]
    except Exception:  # noqa: BLE001 - host facts beside a delivered row of my human's, never the delivery itself
        log.warning("Arrival lines not computed; the delivered row stands without them", exc_info=True)
        return
    if not lines or not rows:
        return
    row, text = messages[rows[-1]], "\n".join(lines)
    content = row.get("content")
    row["content"] = ([*content, {"type": "text", "text": "\n" + text}] if isinstance(content, list)
                      else f"{content}\n{text}")
