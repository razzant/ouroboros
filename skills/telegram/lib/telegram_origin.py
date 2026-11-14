"""Compact phone headings from the host's source-room and task presentation facts."""
from __future__ import annotations

from .telegram_i18n import Index

_TEXT = Index("origin", {"main": "Main chat", "project": "Project: {name}", "chat": "Chat",
                        "task": "Task: {name}", "worker": "Worker: {name}"},
              "where a mirrored message comes from: room, work and nested workers")


def _name(value, limit=48):
    text = " ".join(str(value or "").replace("\x00", "").split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def origin_label(event, lang=""):
    """Only host-labelled Projects count; an arbitrary positive chat ID never means Main."""
    origin = event.get("message_origin")
    if not isinstance(origin, dict):
        return ""  # an older producer's body stays unchanged; no guessed room or lineage
    texts = _TEXT[lang]
    room = origin.get("room") if isinstance(origin.get("room"), dict) else {}
    kind = room.get("kind")
    heading = (texts["main"] if kind == "main" else texts.format("project", name=_name(room.get("name")))
               if kind == "project" else texts["chat"])
    tasks = [row for row in origin.get("tasks") or [] if isinstance(row, dict)]
    # Retain the root and the nearest two workers. An ellipsis explicitly marks the hidden middle.
    shown = tasks if len(tasks) <= 3 else [tasks[0], None, *tasks[-2:]]
    path = []
    for row in shown:
        if row is None:
            path.append("…")
            continue
        name = _name(row.get("name")) or _name(row.get("id"), 8)
        if name:
            path.append(texts.format("worker" if row.get("worker") else "task", name=name))
    if origin.get("incomplete"):
        path.insert(0, "…")
    return heading + ("\n" + " › ".join(path) if path else "")
