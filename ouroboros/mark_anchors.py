"""Marks I tie to an action meet me with the result of the first such call in a task.

A mark may carry an ``anchor`` its author chose: a tool name, optionally with text the
call's arguments must contain, or a file or directory path a call names. The host
compares literally and adds the mark's own words, once per task, as a delivery note
trailing the result the mind is shown (``loop_tool_execution.process_tool_results``): the
batch is measured with it, the trace keeps the clean result beside it as the retained
producer payload (the ``tool_result`` host-note contract), and the host readers that
parse a result (acknowledgements, wake exposure, receipts) read it without the note. It
judges nothing and never refuses the call; a room mark meets only the work of the room
the task is in now. Reading marks is best effort: an unreadable chronicle shows no note.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List

log = logging.getLogger(__name__)

_CACHE_ATTR = "_anchored_marks"
_DELIVERED_ATTR = "_anchored_marks_delivered"
# Quoting characters of code and commands; an apostrophe is a letter of names (O'Brien).
_QUOTES = frozenset("\"`")


def anchor_problem(anchor: Any) -> str:
    """Empty for a well-formed anchor, otherwise what is wrong with it."""
    if not isinstance(anchor, dict) or not anchor:
        return "anchor is an object: {tool, text?} or {path}"
    keys = set(anchor)
    if "tool" in keys and keys <= {"tool", "text"}:
        ok = all(isinstance(anchor[key], str) and anchor[key].strip() for key in keys)
        return "" if ok else "anchor tool and text are non-empty strings"
    if keys == {"path"}:
        path = anchor["path"]
        if not isinstance(path, str) or not path.strip():
            return "anchor path is a non-empty string"
        if _QUOTES.intersection(path):
            return "anchor path has no double quote or backtick (a value with one reads as quoted code)"
        return ""
    return "anchor is {tool, text?} or {path}"


def describe(anchor: Dict[str, Any]) -> str:
    if "path" in anchor:
        return f"the path {anchor['path']}"
    return str(anchor.get("tool")) + (f" with {anchor['text']!r}" if anchor.get("text") else "")


def _strings(value: Any, depth: int = 0) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif depth < 4 and isinstance(value, dict):
        for item in value.values():
            yield from _strings(item, depth + 1)
    elif depth < 4 and isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item, depth + 1)


def _names_path(anchor_path: str, value: str) -> bool:
    """A call argument names the anchored file or directory, relative or absolute."""
    path = anchor_path.strip().replace("\\", "/").rstrip("/")
    path = path[2:] if path.startswith("./") else path
    text = value.replace("\\", "/")
    return bool(path) and (text == path or text.startswith(path + "/") or text.endswith("/" + path)
                           or "/" + path + "/" in text)


def _path_like(value: str) -> bool:
    """One line without a double quote or backtick: a path or an argv element, not file text or
    quoted code (spaces and apostrophes allowed)."""
    return bool(value) and "\n" not in value and "\r" not in value and not _QUOTES.intersection(value)


def matches(anchor: Dict[str, Any], tool: str, args: Any) -> bool:
    """A path anchor meets an argument that is a path naming it; a tool anchor meets that tool,
    and its text an argument value or a command's words."""
    if "path" in anchor:
        return any(_names_path(str(anchor["path"]), value) for value in _strings(args) if _path_like(value))
    if anchor.get("tool") != tool:
        return False
    text = anchor.get("text")
    return not text or str(text) in " ".join(_strings(args))


def _room_key(ctx: Any, root: Any) -> int:
    """The project room the task's lineage is bound to now (0 for none): cheap, cached per bindings."""
    from ouroboros.project_dialogue import bound_room_chat
    from ouroboros.projects_registry import RoomResolver

    task_id = str(getattr(ctx, "task_id", "") or "")
    meta = getattr(ctx, "task_metadata", {}) or {}
    return bound_room_chat(RoomResolver(root), {"task_id": task_id, "parent_task_id": meta.get("parent_task_id"),
                                                "root_task_id": meta.get("root_task_id")}) if task_id else 0


def _anchored_marks(ctx: Any) -> List[Dict[str, Any]]:
    """This task's anchored marks (its room's and the global ones), read again only when the
    room its lineage is bound to changes (a move of the task or of an ancestor)."""
    from ouroboros.chronicle_store import ChronicleStore
    from ouroboros.dialogue_evidence import own_room_chat
    from ouroboros.tool_access import canonical_data_root

    root = canonical_data_root(ctx)
    key = _room_key(ctx, root)
    cached = getattr(ctx, _CACHE_ATTR, None)
    if cached is not None and cached[0] == key:
        return cached[1]
    store = ChronicleStore(root)
    receipt = store.activation()  # read only: marks exist only on an activated chronicle
    if not isinstance(receipt, dict) or receipt.get("kind") != "activation":
        setattr(ctx, _CACHE_ATTR, (key, []))  # memory_mark resets it, like a cached list
        return []
    room = own_room_chat(ctx, root)
    rows = store.active_marks(None) if room is None else store.active_marks(str(room), include_global=True)
    marks = [mark for mark in rows if isinstance(mark.get("anchor"), dict) and not anchor_problem(mark["anchor"])
             and (room is not None or mark.get("scope") == "global")]
    setattr(ctx, _CACHE_ATTR, (key, marks))
    return marks


def anchored_note(ctx: Any, tool: str, args: Any) -> str:
    """The host note for the anchored marks this call meets for the first time in the task, or ``""``."""
    if ctx is None:
        return ""
    try:
        if tool == "memory_mark":
            setattr(ctx, _CACHE_ATTR, None)  # a new or released mark acts from the next call
            return ""
        delivered = getattr(ctx, _DELIVERED_ATTR, None)
        if delivered is None:
            delivered = set()
            setattr(ctx, _DELIVERED_ATTR, delivered)
        notes = []
        for mark in _anchored_marks(ctx):
            mark_id = str(mark.get("id") or "")
            if mark_id in delivered or not matches(mark["anchor"], tool, args):
                continue
            delivered.add(mark_id)
            notes.append(f"⚠️ Mark I tied to {describe(mark['anchor'])} ({mark_id}): "
                         f"{str(mark.get('text') or '').strip()}")
        return "\n".join(notes)
    except Exception:
        log.debug("anchored marks: no note for %s", tool, exc_info=True)
        return ""
