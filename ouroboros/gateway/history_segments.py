"""Which rooms a closed history archive can hold, so a room read skips the rest.

Every room is a lens over one shared chat/progress chain
(``project_dialogue.room_membership``). A Project's newest messages can lie
behind any number of other rooms' archives, so reading the room's own feed
means walking past archives that hold none of its rows. A closed archive never
changes: one byte scan records the chat ids, task ids and owner-row identities
it carries, and a Project read skips every archive whose summary rules the room
out instead of parsing it. The summary is a conservative superset — any nested
``chat_id``/``task_id`` occurrence counts — so a skip never hides a room row;
an archive whose summary cannot be taken is read normally.
"""

from __future__ import annotations

import json
import os
import re
import threading
from typing import Any, Callable, Dict, NamedTuple, Optional

_CHAT_ID = re.compile(rb'"chat_id":\s*"?(-?\d+)')
_TASK_ID = re.compile(rb'"(?:task_id|parent_task_id|root_task_id)":\s*"([^"\\]+)"')
_OWNER_ROW = re.compile(rb'"direction":\s*"in"')
# A summary holds a few hundred ids; an install holds hundreds of archives per stream.
_CACHE_LIMIT = 4096
_cache: Dict[tuple, "SegmentSummary"] = {}
_cache_lock = threading.Lock()


class SegmentSummary(NamedTuple):
    chat_ids: frozenset
    task_ids: frozenset
    owner_rows: frozenset  # (chat id, text hash) of the owner's rows: every source identity shares both


def _summarize(data: bytes) -> SegmentSummary:
    from ouroboros.project_dialogue import _entry_source_identities  # lazy: gateway -> dialogue

    owner_rows: set = set()
    if _OWNER_ROW.search(data):
        for line in data.splitlines():
            if _OWNER_ROW.search(line):
                try:
                    row = json.loads(line)
                except ValueError:
                    continue  # the reader discloses the parse gap when it reads this archive
                if isinstance(row, dict):
                    owner_rows.update((key[0], key[3]) for key in _entry_source_identities(row))
    return SegmentSummary(
        chat_ids=frozenset(int(value) for value in _CHAT_ID.findall(data)),
        task_ids=frozenset(value.decode("utf-8", "replace") for value in _TASK_ID.findall(data)),
        owner_rows=frozenset(owner_rows),
    )


def segment_summary(path: Any, stat: os.stat_result) -> Optional[SegmentSummary]:
    """The cached summary of one closed archive as captured by ``stat``, or None."""
    key = (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    with _cache_lock:
        cached = _cache.get(key)
    if cached is not None:
        return cached
    try:
        with open(path, "rb") as handle:
            opened = os.fstat(handle.fileno())
            if (opened.st_dev, opened.st_ino) != (stat.st_dev, stat.st_ino):
                return None
            data = handle.read(stat.st_size)
    except OSError:
        return None
    if len(data) != stat.st_size:
        return None
    summary = _summarize(data)
    with _cache_lock:
        if len(_cache) >= _CACHE_LIMIT:
            _cache.clear()
        _cache[key] = summary
    return summary


def room_segment_lens(thread_id: int, project_chat_ids: set, source_refs: list,
                      bindings_by_task: Dict[str, int]) -> Optional[Callable[[SegmentSummary], bool]]:
    """``may_hold(summary)`` for a Project room; None (read everything) for any other room.

    Mirrors ``room_membership``'s three Project admissions: the room's own chat,
    a task bound to it (by task, parent or root id), or the owner row a binding
    names. Main and the rooms that share its lens hold rows in nearly every archive.
    """
    if thread_id not in project_chat_ids:
        return None
    from ouroboros.project_dialogue import _source_ref_identity  # lazy: gateway -> dialogue

    bound = frozenset(task for task, chat in bindings_by_task.items() if chat == thread_id)
    origins = frozenset((key[0], key[3]) for ref in source_refs if isinstance(ref, dict)
                        if (key := _source_ref_identity(ref)) is not None)

    def may_hold(summary: SegmentSummary) -> bool:
        return (thread_id in summary.chat_ids or not bound.isdisjoint(summary.task_ids)
                or not origins.isdisjoint(summary.owner_rows))

    return may_hold
