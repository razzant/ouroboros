"""Marks act while a task runs: those I tie to an action, and the changes other works make.

A mark may carry an ``anchor`` its author chose: a tool name, optionally with text the
call's arguments must contain, or a file or directory path a call names. The host
compares literally and adds the mark's own words, once per task, as a delivery note
trailing the result the mind is shown (``loop_tool_execution.process_tool_results``): the
batch is measured with it, the trace keeps the clean result beside it as the retained
producer payload (the ``tool_result`` host-note contract), and the host readers that
parse a result (acknowledgements, wake exposure, receipts) read it without the note. It
judges nothing and never refuses the call; a room mark meets only the work of the room
the task is in now.

A mark another task makes, releases or shows differently after my view was captured
(``MemoryViewSnapshot.chronicle_sequence``, read from the snapshot its marks came from)
reaches me at my next round, after any fold (``append_mark_changes``, called by
``loop_round_limits._run_round_compaction``): my room's and the global marks by my view's
own scope rule (``memory_view.mark_in_view``), never another room's room mark or a
host-kept late review. One typed host row states each changed mark's latest change: what
changed, who and when, the mark's own words, its quote while visibility is full, its target
and a release's or view change's reason. It is not my human's words and registers no
directive. My own change is never echoed, yet it is its mark's latest change. What was
stated is read back from the rows in view (journal sequence and mark): a latest change no
intact row states is stated when it came after my captured view or a row in view names its
mark, so again after a fold took its row. A cold resume captures the view again and reads
from the earliest change a restored row states: a change during the pause to a mark a row
stated is stated, the rest the new view shows. So nothing repeats and no mark a row stated
is left looking in force after its release. Such a change also drops the anchored-marks
cache. Each round costs a journal size check; a grown journal, one indexed read of the
records appended since the last one.
Reading marks is best effort: an unreadable chronicle shows no note and no change.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterator, List

log = logging.getLogger(__name__)

_CACHE_ATTR = "_anchored_marks"
_DELIVERED_ATTR = "_anchored_marks_delivered"
_CHANGES_ATTR = "_mark_changes_read"  # this context's journal size, read position and each in-scope mark's latest change
MARK_CHANGES_KIND = "ouroboros_mark_changes"
_CHANGES_HEADER = ("[Host facts: marks of this room and global ones that other works changed while this work runs — "
                   "journal sequence, the change, who and when, then the mark's own words]")
_STATED = re.compile(r"^- #(\d+) mark (\S+): ", re.MULTILINE)  # a change's head line; its words are indented
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


def _stated(messages: List[Dict[str, Any]]) -> Dict[int, str]:
    """``{journal sequence: mark id}`` of the changes the intact change rows in view state."""
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, host_row_intact

    return {int(sequence): mark for row in messages if isinstance(row, dict)
            and row.get(HOST_CONTEXT_KIND_KEY) == MARK_CHANGES_KIND and host_row_intact(row)
            for sequence, mark in _STATED.findall(row["content"])}


def _change_lines(change: Dict[str, Any], label: Any) -> List[str]:
    """One change: its journal sequence and mark, what changed, who and when, then the mark's own words indented."""
    from ouroboros.memory_view import _minute, mark_entry
    from ouroboros.memory_view_legacy import INDENT, indented

    entry, kind = mark_entry(change["mark"], label), change.get("kind")
    author = change.get("author") if isinstance(change.get("author"), dict) else {}
    focus = author.get("focus") if isinstance(author.get("focus"), dict) else {}
    who = ", ".join(part for part in (str(focus.get("role") or author.get("kind") or "not recorded"),
                                      f"task {author['task_id']}" if author.get("task_id") else "",
                                      f"in {label(focus['room_id'])}" if focus.get("room_id") is not None else "") if part)
    what = {"mark": "marked", "mark_release": "released"}.get(kind) or f"view changed to {change.get('visibility')}"
    lines = [f"- #{change['sequence']} mark {entry['id']}: {what} by {who} at {_minute(change.get('ts'))}; "
             f"[{entry['scope']}; {entry['room']}]" + (f", marked by {entry['by']} on {entry['date']}" if kind != "mark" else "")
             + f"; target {entry['target']}"
             + (f"; also shown after a task's first call matching {entry['anchor']}" if entry["anchor"] else "")]
    lines += [indented(entry["text"])] if entry["text"] else []
    lines += [f"{INDENT}quote: " + indented(entry["quote"]).lstrip()] if entry["quote"] else []
    lines += [f"{INDENT}reason: " + indented(change["reason"]).lstrip()] if change.get("reason") else []
    return lines


def append_mark_changes(ctx: Any, messages: List[Dict[str, Any]]) -> bool:
    """Append one host row with the latest change of each of my room's and the global marks that another task changed
    after my view was captured (or that a row in view already names) and no intact row in view states yet; True when
    it appended (the module docstring has the contract)."""
    if ctx is None:
        return False
    try:
        from ouroboros.chronicle_store import ChronicleStore, is_late_review_mark
        from ouroboros.context_budget import bound_host_row
        from ouroboros.context_view_facts import memory_snapshot
        from ouroboros.memory_view import mark_in_view, room_labeler
        from ouroboros.tool_access import canonical_data_root

        snapshot = memory_snapshot(getattr(ctx, "context_fit_plan", None))
        if snapshot is None or snapshot.chronicle_sequence is None or snapshot.spec.marks == "none":
            return False
        stated = _stated(messages)
        read = getattr(ctx, _CHANGES_ATTR, None)
        if read is None:  # a start, or a cold resume whose restored rows say where stating began
            read = {"size": -1, "after": min(stated, default=snapshot.chronicle_sequence + 1) - 1, "latest": {}}
            setattr(ctx, _CHANGES_ATTR, read)
        root = canonical_data_root(ctx)
        store = ChronicleStore(root)
        size = store.log_path.stat().st_size
        if size != read["size"]:  # the journal only grows: an unchanged size has no new record
            changes, newest = store.mark_changes(read["after"])
            task_id = str(getattr(ctx, "task_id", "") or "")
            for change in changes:
                if is_late_review_mark(change["mark"]) or not mark_in_view(change["mark"], snapshot.spec, running=True):
                    continue
                change["own"] = str((change.get("author") or {}).get("task_id") or "") == task_id
                read["latest"][str(change["mark"].get("id"))] = change  # my own change is its mark's latest, unechoed
                if not change["own"]:
                    setattr(ctx, _CACHE_ATTR, None)  # a new anchored mark acts, a released one stops, from the next call
            read.update(size=size, after=newest)
        # The captured view shows a mark as it stood at capture; a restored row names one it stated before a resume.
        missing = sorted((change for mark, change in read["latest"].items() if not change["own"]
                          and change["sequence"] not in stated
                          and (change["sequence"] > snapshot.chronicle_sequence or mark in stated.values())),
                         key=lambda change: change["sequence"])
        if not missing:
            return False
        label = room_labeler(root)
        text = "\n".join([_CHANGES_HEADER, *(line for change in missing for line in _change_lines(change, label))])
        messages.append(bound_host_row(MARK_CHANGES_KIND, text))
        return True
    except Exception:
        log.debug("mark changes: none stated this round", exc_info=True)
        return False
