"""The improvement backlog: one ordinary authored global knowledge note.

The acting mind writes it through ``knowledge_write`` like any note (shelf lock,
exact ``expected_revision``, anchored edit, full old/new history) and may archive
it like any note; no host writer appends, counts, reopens, ranks or closes its
items. Older stored fields (counts, fingerprints, statuses) stay as the text the
author left. This module renders the current note for the readers that consult
it: the evolution chooser (``post_task_evolution``) and an evolution task's
context (``context.py``). No parser selects, sorts or hides its prose.
"""

from __future__ import annotations

import pathlib
from typing import Any

BACKLOG_TOPIC = "improvement-backlog"
BACKLOG_READER = f"knowledge_read(topic='{BACKLOG_TOPIC}', scope='global')"


def authored_backlog_text(drive_root: Any) -> str:
    """The current backlog note with its source facts, or its named state.

    An active note is given whole. An archived one is named by its revision and
    archive reason, its body one read away. A missing note is a plain fact; an
    unreadable one stays unavailable, never an empty backlog.
    """
    from ouroboros.knowledge import read_knowledge_note, resolve_knowledge_address

    try:
        note = read_knowledge_note(resolve_knowledge_address(pathlib.Path(drive_root), BACKLOG_TOPIC, "global"))
    except FileNotFoundError:
        return "No improvement-backlog note exists."
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return f"Improvement-backlog source unavailable ({type(exc).__name__}); it is unknown, not empty: {BACKLOG_READER}."
    if note.state == "archived":
        archive = note.metadata.get("archive") or {}
        return (f"The improvement-backlog note is archived (revision {note.revision}, at {archive.get('at', 'unknown')}; "
                f"reason: {archive.get('reason', 'unknown')}). Its body is not shown here: {BACKLOG_READER}.")
    return f"Source: {BACKLOG_READER}, revision {note.revision}, {len(note.text)} chars, complete.\n\n{note.text}"
