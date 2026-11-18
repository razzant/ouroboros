"""At my human's arrival, the host names the working-view notes in view: append only, facts only.

A drain that delivers a row of theirs into a work's transcript (their words, a card answer or press, a closure of
theirs, their words relayed or carried to this work) closes the last such row with a host line, as part of that
delivered message, while the view holds notes the mind wrote through ``compact_context`` folds (capsules of actor
authorship, the notes the round facts count as its own): one line naming each as a fold receipt names it
(``compact_context.view_notes``): by the unit id ``compact_context`` inspects and ``replace_note_ids`` takes, its
write time (``written_at`` in the capsule metadata, else "not recorded") and its first line. No earlier message
changes; every note in view precedes the arrival, so nothing is joined by time, and nothing is judged, asked or
instructed.

The notes line names the list only when it differs from the list last named in a row of my human's still among the
messages in hand: what the host already appended is read from those rows, where its line ends the row it closes, never
stored. A press without words gets its own line, their last own words in this work, on its own row as it is
delivered (``answer_facts.mark_press_arrival``). The ids are those of this arrival: a later fold renumbers the
units, so the list it changes is named again. A live call's drain delivers into a fresh list of rows
(``live_conversation.mailbox``) that holds no notes, so a call's arrivals name none.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence

log = logging.getLogger(__name__)

NOTES_LINE_HEAD = "[Host fact: your working-view notes in view now: "


def notes_line(messages: List[Dict[str, Any]]) -> str:
    """The host line listing the notes the mind wrote in this view, or "" when the view holds none."""
    from ouroboros.tools.compact_context import view_notes

    notes = view_notes(messages, own=True)
    return (NOTES_LINE_HEAD + "; ".join(
        f"{note['id']}, written {note['written']}, «{note['first_line']}»" for note in notes) + "]") if notes else ""


def _closing_line(row: Dict[str, Any]) -> str:
    """The last line of a row's text, where the host's lines close a row of my human's."""
    content = row.get("content")
    if isinstance(content, list):
        texts = [block.get("text") for block in content if isinstance(block, dict) and block.get("type") == "text"]
        content = texts[-1] if texts else ""
    return content.rsplit("\n", 1)[-1] if isinstance(content, str) else ""


def last_named(messages: List[Dict[str, Any]]) -> str:
    """The notes line that closes the latest row of my human's among ``messages`` to carry one, else ""."""
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, OWNER_WORDS_KIND

    for row in reversed(messages):
        if isinstance(row, dict) and row.get(HOST_CONTEXT_KIND_KEY) == OWNER_WORDS_KIND:
            if (line := _closing_line(row)).startswith(NOTES_LINE_HEAD):
                return line
    return ""


def mark_arrival(ctx: Any, messages: List[Dict[str, Any]], rows: Sequence[int]) -> None:
    """Append the notes line to the last row of my human's this drain delivered (``rows``: positions of the rows it
    delivered, typed at delivery and never merged); a view without notes changes nothing, and a notes list already
    named in a row still in ``messages`` is not named again. A line the host cannot compute is left out and logged: the
    delivered row stands, as for the surface note beside it. ``ctx`` is the delivering focus, kept for its callers."""
    try:
        notes = notes_line(messages)
        text = "" if notes == last_named(messages) else notes
    except Exception:  # noqa: BLE001 - a host fact beside a delivered row of my human's, never the delivery itself
        log.warning("Arrival lines not computed; the delivered row stands without them", exc_info=True)
        return
    if not text or not rows:
        return
    row = messages[rows[-1]]
    content = row.get("content")
    row["content"] = ([*content, {"type": "text", "text": "\n" + text}] if isinstance(content, list)
                      else f"{content}\n{text}")
