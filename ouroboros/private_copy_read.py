"""Passive reads of the private-copy records, for what must write nothing (the Free preview, the keep's
inventory, a registration question): no legacy import, no event row, no rename. What a record is stays the
owner's (``subagent_worktrees._parse_record``, ``_legacy_entries``); a retirer keeps the writing owner's
``_records``, which also imports and reports."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Literal, Optional, Tuple

from ouroboros import subagent_worktrees as sw


@dataclass(frozen=True)
class RecordsRead:
    rows: Tuple[Dict[str, Any], ...] = ()
    unreadable: FrozenSet[str] = frozenset()  # record file names that exist and could not be read
    unlistable: bool = False  # the records folder could not be listed: any folder may have a record
    legacy_unreadable: bool = False  # the documented escape: a malformed legacy file names no folder


def _legacy_rows(data_dir: Optional[Any]) -> Tuple[List[Dict[str, Any]], bool]:
    """The rows a still-present legacy registry would import, read in memory, and whether it is malformed."""
    try:
        entries = sw._legacy_entries(sw._legacy_registry_path(data_dir).read_bytes())
    except FileNotFoundError:
        return [], False
    except (OSError, ValueError):
        return [], True
    return [row for row in entries if sw._names_copy(row)], False


def read_records(data_dir: Optional[Any] = None) -> RecordsRead:
    """Every copy record as it stands, the unreadable ones named, without one write."""
    directory = sw._records_dir(data_dir)
    try:
        names = os.listdir(directory)
    except FileNotFoundError:
        names = []
    except OSError:
        return RecordsRead(unlistable=True)
    rows, unreadable = [], set()
    for name in sorted(names):
        if name.endswith(".json") and not name.startswith("."):
            try:
                row = sw._parse_record(directory / name)
            except (OSError, ValueError):
                unreadable.add(name)
                continue
            if row is not None:
                rows.append(row)
    legacy, malformed = _legacy_rows(data_dir)
    known = {sw._record_file(row["path"], data_dir).name for row in rows} | unreadable
    rows += [row for row in legacy if sw._record_file(row["path"], data_dir).name not in known]
    return RecordsRead(tuple(rows), frozenset(unreadable), False, malformed)


def _copy_owner(name: str) -> str:
    """The owner task a copies-root folder name carries (``dlg_<task>_<snapshot>``, ``body_<task>``, a task id)."""
    for prefix in ("dlg_", "dlgp_"):
        if name.startswith(prefix):
            return name[len(prefix):].rsplit("_", 1)[0]
    return name[len("body_"):] if name.startswith("body_") else name


def record_at(path: Any, data_dir: Optional[Any] = None) -> Optional[Dict[str, Any]]:
    """The record of the copy at ``path`` as it stands: its own file, else a still-present legacy row, else
    ``None``. An unreadable own file raises ``ValueError`` or ``OSError``; a malformed legacy file names nothing."""
    file = sw._record_file(path, data_dir)
    row = sw._parse_record(file)
    if row is not None:
        return row
    return next((row for row in _legacy_rows(data_dir)[0] if sw._record_file(row["path"], data_dir).name == file.name), None)


def checkout_registration(path: Any, data_dir: Optional[Any] = None) -> Literal["registered", "absent", "unknown"]:
    """Whether the copy at ``path`` has a record, by its own record file: ``unknown`` when that file
    exists but cannot be read, never ``absent``. A malformed legacy file stays the documented escape."""
    if not str(path or "").strip():
        return "absent"
    try:
        return "registered" if record_at(path, data_dir) is not None else "absent"
    except (OSError, ValueError):
        return "unknown"
