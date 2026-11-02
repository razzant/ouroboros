"""Rotation-aware JSONL reads: captured byte ranges and bounded filtered tails.

``JsonlChainSnapshot`` shares physical capture/read/segment ownership between
file-backed event history and the swarm rollup,
whose selection policies remain separate. The filtered-tail reader below serves
context and endpoints.

Moved here from ``gateway/_helpers.py`` (v6.90.x P2) so that context assembly
(``memory.py``, razzant/ouroboros#131) can use the same window-doubling
filtered tail the history, logs and routing endpoints already use, without a
core module importing the gateway layer (DEVELOPMENT §13). The gateway module
keeps thin wrappers over these functions with its own parser seam.

Semantics: the live file is read from a byte tail that starts at
``TAIL_WINDOW_START_BYTES`` and DOUBLES until the FILTERED quota is satisfied
(rows for which ``counts_toward_quota`` is true reach ``want``) or the window
covers the whole file — the degenerate case is one full read of a
rotation-bounded file, never a loop. Rotated ``archive/<prefix>_*.jsonl``
segments are then backfilled newest-first until the quota is met, bounded to
``max_archives`` files, and everything is reassembled chronologically (oldest
chosen archive -> live window). Older segments are NOT consulted: they stay
durable history for full-history readers. ALL rows of the chosen window are
returned (the quota decides where to stop, the caller filters); the optional
gap set names every parse or read failure the window met, and
``archives_bounded`` tells a caller whether the quota went unmet while older
archives were left unopened — the fact a continuity disclosure needs.
"""

from __future__ import annotations

import json
import os
import pathlib
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from ouroboros import utils
from ouroboros.utils import JsonlChainUnreadable, iter_jsonl_objects, jsonl_chain_handles

# One bounded tail per path per process, memoized on the file's identity: a context build reads the
# same 256 KB of ``logs/supervisor.jsonl`` for its health block and its body fact, so one read serves
# both; an append (a new size and mtime) or a rotation (a new inode) reads again. The memory bound is
# one entry per path (the two runtime logs the context tails).
_TAIL_LOCK = threading.Lock()
_TAILS: Dict[str, Tuple[tuple, Tuple[Dict[str, Any], ...]]] = {}


def cached_tail_objects(path: Any, tail_bytes: int) -> Tuple[Dict[str, Any], ...]:
    """The parseable dict rows of the last ``tail_bytes`` of ``path`` (``iter_jsonl_objects``), read once
    per file identity ``(st_dev, st_ino, st_size, st_mtime_ns)`` and tail size, kept per path; a missing
    file is ``()``. Callers read the rows and never mutate them."""
    path = pathlib.Path(path)
    try:
        stat = os.stat(path)
    except OSError:
        return ()
    identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, int(tail_bytes))
    key = str(path)
    with _TAIL_LOCK:
        kept = _TAILS.get(key)
        if kept is not None and kept[0] == identity:
            return kept[1]
    rows = tuple(iter_jsonl_objects(path, tail_bytes=tail_bytes))
    with _TAIL_LOCK:
        _TAILS[key] = (identity, rows)
    return rows

TAIL_WINDOW_START_BYTES = 512 * 1024
ARCHIVE_BACKFILL_MAX = 3


class JsonlChainSnapshot:
    """One captured byte horizon shared by history pages, wake observations and receipt evidence.

    Reads reopen through the rotation-aware handle owner and never cross the
    captured horizon. No descriptor survives a read. ``rows`` is the complete-row
    reader of wake observations and receipt evidence; history keeps its own
    parser. Callers own the meaning of gaps and the boundary they accept; this
    reader imposes neither a tail quota nor a timestamp policy. ``strict=False``
    captures only what could be listed and opened, so a caller using it must
    disclose that degraded capture itself.
    """

    def __init__(self, path: pathlib.Path, *, upper: Optional[int] = None, strict: bool = True):
        self.path, self.snapshot = path, {}
        with jsonl_chain_handles(path, strict=strict, start_offset=0, snapshot=self.snapshot):
            pass
        self.entries, self.ends = self.snapshot["entries"], self.snapshot["ends"]
        self.upper = self.snapshot["total"] if upper is None else upper
        if not 0 <= self.upper <= self.snapshot["total"]:
            raise JsonlChainUnreadable("source is shorter than its captured boundary")

    def _read(self, start: int, end: int) -> bytes:
        from bisect import bisect_left

        if not 0 <= start <= end <= self.upper:
            raise JsonlChainUnreadable("read is outside the captured boundary")
        parts = []
        while start < end:
            with jsonl_chain_handles(self.path, strict=True, start_offset=start, snapshot=self.snapshot) as handles:
                if not handles:
                    raise JsonlChainUnreadable("source ended before its captured boundary")
                size = min(end, self.ends[bisect_left(self.ends, start + 1)]) - start
                data = handles[0][1].read(size)
                if len(data) != size:
                    raise JsonlChainUnreadable("source read ended before its captured boundary")
            parts.append(data)
            start += size
        return b"".join(parts)

    def segment(self, index: int) -> tuple[int, int]:
        return (self.ends[index - 1] if index else 0), min(self.ends[index], self.upper)

    def rows(self, index: int, lower: int, gaps: set) -> Tuple[List[Tuple[int, Dict[str, Any]]], int]:
        """``([(offset, row)], end of the last complete line)`` of one segment from ``lower``.

        An unfinished live line stops the segment (its writer owns it; the next
        pass reads it whole); a torn archive line is named in ``gaps``.
        """
        base, end = self.segment(index)
        start = max(base, lower)
        if start >= end:
            return [], start
        data, rows, position = self._read(start, end), [], start
        for raw in data.splitlines(keepends=True):
            if not raw.endswith(b"\n"):
                if self.entries[index][2]:
                    break  # an unfinished live line: the next pass reads it whole
                gaps.add("torn_archive_line")
            position += len(raw)
            try:
                row = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                if raw.strip():
                    gaps.add("malformed_jsonl")
                continue
            if isinstance(row, dict):
                rows.append((position - len(raw), row))
        return rows, position


def archive_segments(archive_dir: pathlib.Path, archive_prefix: str, gaps: Optional[set] = None) -> list:
    """``archive/<prefix>_*.jsonl`` newest-first by name (chronological by construction).

    An explicit ``scandir`` because ``Path.glob`` swallows a ``PermissionError``
    on the directory and yields nothing, which reads like "never rotated";
    unreadable enumeration is reported in ``gaps`` instead.
    """
    prefix = f"{archive_prefix}_"
    try:
        with os.scandir(archive_dir) as entries:
            found = [
                pathlib.Path(entry.path) for entry in entries
                if entry.name.startswith(prefix) and entry.name.endswith(".jsonl") and entry.is_file()
            ]
    except FileNotFoundError:
        return []
    except OSError:
        if gaps is not None:
            gaps.add("unreadable_source")
        return []
    return sorted(found, key=lambda p: p.name, reverse=True)


# (archive/'s device and inode, prefix) -> (key, newest-first segment names): one per log in use
_LISTINGS: Dict[tuple, tuple] = {}


def remembered_listing(live: pathlib.Path) -> Callable[..., list]:
    """``archive_segments`` for LIVE's log (``list_archives``' signature), listed again only when it
    may have changed: its rotation, the one writer of its archive names, gives LIVE a new inode, and
    an entry added or removed moves ``archive/``'s mtime. A listing that met a gap, or a directory
    changed within a clock tick (``utils._RACY_NS``), is never reused. The directory is known by its
    identity, however a caller spells its path (the boot remembers it for the turns)."""
    def listing(archive_dir: pathlib.Path, archive_prefix: str, gaps: Optional[set] = None) -> list:
        started, name, held = time.time_ns(), None, None
        try:
            live_stat, directory = os.stat(live), os.stat(archive_dir)
            key = (live_stat.st_dev, live_stat.st_ino, directory.st_mtime_ns)
            name = (directory.st_dev, directory.st_ino, archive_prefix)
            held = _LISTINGS.get(name)
        except OSError:
            key = None
        if key is not None and held is not None and held[0] == key:
            return [pathlib.Path(archive_dir) / segment for segment in held[1]]
        found: set = set()
        segments = archive_segments(archive_dir, archive_prefix, found)
        if gaps is not None:
            gaps.update(found)
        if key is not None and not found and key[2] < started - utils._RACY_NS:
            _LISTINGS[name] = (key, tuple(segment.name for segment in segments))
        return segments

    return listing


# live log -> {archive: (file identity, task ids it holds, its gaps)}: the newest ARCHIVE_BACKFILL_MAX
_ARCHIVE_TASKS: Dict[str, Dict[str, tuple]] = {}


def task_archive_parser(live: pathlib.Path, task_id: str) -> Callable[..., list]:
    """``iter_jsonl_objects`` (``iter_objects``' signature) for ONE task's window of LIVE's log.

    A rotated archive is written once, so the task ids it holds are kept per file identity: an
    archive without TASK_ID's rows is not parsed again for that task, and the gaps it met are
    replayed. The live file, and an archive that holds the task, parse as ever. Only archives a
    read can consult (the newest ``ARCHIVE_BACKFILL_MAX`` of this log) are kept.
    """
    held = _ARCHIVE_TASKS.setdefault(str(live), {})

    def parse(path: pathlib.Path, *, tail_bytes: Optional[int] = None, gap_reasons: Optional[set] = None) -> list:
        path = pathlib.Path(path)
        if path == live or tail_bytes is not None:
            return list(iter_jsonl_objects(path, tail_bytes=tail_bytes, gap_reasons=gap_reasons))
        found: set = set()
        try:
            with path.open("rb") as handle:
                identity, known = utils.file_identity(os.fstat(handle.fileno())), held.get(str(path))
                if known is not None and known[0] == identity and task_id not in known[1]:
                    rows, found = [], set(known[2])
                else:
                    rows = list(iter_jsonl_objects(path, gap_reasons=found, _handle=handle))
                    held[str(path)] = (identity, frozenset(str(row.get("task_id", "")).strip() for row in rows),
                                       frozenset(found))
                    for stale in sorted(held)[:-ARCHIVE_BACKFILL_MAX]:
                        del held[stale]
        except FileNotFoundError:
            rows = []
        if gap_reasons is not None:
            gap_reasons.update(found)
        return rows

    return parse


def read_jsonl_segment_with_gaps(
    path: pathlib.Path,
    *,
    tail_bytes: Optional[int] = None,
    iter_objects: Optional[Callable[..., Iterable[Any]]] = None,
    stat: Optional[Callable[[pathlib.Path], Any]] = None,
) -> tuple[list, set[str]]:
    """Read one JSONL segment while retaining truthful parse/read-gap facts.

    ``iter_jsonl_objects`` intentionally skips malformed rows because most
    callers are best-effort telemetry readers.  History readers publish a
    completeness claim, so a skipped row must remain visible as a bounded
    read-gap fact even when the valid rows can still be rendered.
    """
    path = pathlib.Path(path)
    parse = iter_objects or iter_jsonl_objects  # module name resolved at call time (test seam)
    try:
        (stat or pathlib.Path.stat)(path)
    except FileNotFoundError:
        return [], set()
    except OSError:
        return [], {"unreadable_source"}

    gaps: set[str] = set()
    try:
        entries = list(parse(path, tail_bytes=tail_bytes, gap_reasons=gaps))
    except OSError:
        gaps.add("unreadable_source")
    return entries, gaps


def read_rotated_jsonl_entries(
    live: pathlib.Path,
    archive_dir: pathlib.Path,
    archive_prefix: str,
    want: int,
    counts_toward_quota,
    max_archives: int = ARCHIVE_BACKFILL_MAX,
    *,
    include_gaps: bool = False,
    iter_objects: Optional[Callable[..., Iterable[Any]]] = None,
    coverage: Optional[dict] = None,
    list_archives: Optional[Callable[..., list]] = None,
    stat: Optional[Callable[[pathlib.Path], Any]] = None,
) -> list | tuple[list, set[str]]:
    """Bounded, rotation-aware read of one JSONL log (module docstring).

    ``iter_objects`` is the parser seam (the gateway wrapper passes its own
    name so its tests keep governing it). ``list_archives`` is the listing seam
    with ``archive_segments``' signature: a request that reads many tasks lists
    ``archive/`` once and replays its gap to each (``replay_evidence_for_tasks``),
    and ``stat`` the metadata seam through which it stats a segment once.
    ``coverage``, when given, is filled with ``live_size``, ``live_window``
    (bytes of the live file read; both absent when the live file could not be
    read), ``archives`` (consulted count), ``archives_available`` and
    ``archives_bounded`` and the exact consulted ``archive_sources`` paths.
    """
    live = pathlib.Path(live)
    parse = iter_objects or iter_jsonl_objects  # module name resolved at call time (test seam)
    gaps: set[str] = set()
    live_readable = True
    stat = stat or pathlib.Path.stat
    try:
        size = stat(live).st_size
    except FileNotFoundError:
        size = 0  # not written yet: an empty window, not a gap
    except OSError:
        size, live_readable = 0, False  # cannot be read: disclosed as unread, never as "whole live file"
        gaps.add("unreadable_source")
    window = TAIL_WINDOW_START_BYTES
    collect = include_gaps or coverage is not None  # a coverage claim needs the gap facts too
    while True:
        if window >= size:
            if collect:
                live_entries, live_gaps = read_jsonl_segment_with_gaps(live, iter_objects=parse, stat=stat)
                gaps.update(live_gaps)
            else:
                live_entries = list(parse(live))
            window = size
            break
        if collect:
            live_entries, live_gaps = read_jsonl_segment_with_gaps(
                live, tail_bytes=window, iter_objects=parse, stat=stat)
            gaps.update(live_gaps)
        else:
            live_entries = list(parse(live, tail_bytes=window))
        if sum(1 for entry in live_entries if counts_toward_quota(entry)) >= want:
            break
        window *= 2
    collected = sum(1 for entry in live_entries if counts_toward_quota(entry))
    archives = (list_archives or archive_segments)(pathlib.Path(archive_dir), archive_prefix, gaps if collect else None)
    chosen: list = []
    archive_sources: list[str] = []
    for archive_path in archives:
        if collected >= want or len(chosen) >= max_archives:
            break
        try:
            if collect:
                archive_entries, archive_gaps = read_jsonl_segment_with_gaps(
                    archive_path, iter_objects=parse, stat=stat)
                gaps.update(archive_gaps)
            else:
                archive_entries = list(parse(archive_path))
        except Exception:
            gaps.add("unreadable_source")
            continue
        chosen.append(archive_entries)
        archive_sources.append(str(archive_path))
        collected += sum(1 for entry in archive_entries if counts_toward_quota(entry))
    ordered: list = []
    for archive_entries in reversed(chosen):  # oldest chosen archive first
        ordered.extend(archive_entries)
    ordered.extend(live_entries)
    if coverage is not None:
        coverage.update({
            "archives": len(chosen), "archive_sources": archive_sources, "archives_available": len(archives),
            "archives_bounded": collected < want and len(chosen) < len(archives),
            "matched": collected, "gaps": sorted(gaps),
        })
        if live_readable:  # an unreadable live file leaves no window facts: the line says "unread"
            coverage["live_size"], coverage["live_window"] = size, min(window, size)
    return (ordered, gaps) if include_gaps else ordered


def _kb(n: int) -> str:
    return f"{n / 1024:.0f} KB" if n < 1024 * 1024 else f"{n / (1024 * 1024):.1f} MB"


def coverage_line(coverage: dict) -> str:
    """One compact, human-and-model-readable line for a section header.

    Says whose rows they are, how many were shown of how many matched inside
    the read window, what the window was, and whether older archives were left
    unopened with the quota unmet — the facts a continuity disclosure needs.
    """
    task_id = str(coverage.get("task_id") or "")
    shown, matched = int(coverage.get("shown") or 0), int(coverage.get("matched") or 0)
    whose = f"task {task_id}" if task_id else "all tasks"
    live_size, live_window = int(coverage.get("live_size") or 0), int(coverage.get("live_window") or 0)
    unread = "live_size" not in coverage
    whole = (live_window >= live_size and not coverage.get("archives_bounded") and not unread
             and not coverage.get("older_available") and not coverage.get("gaps"))
    unit = str(coverage.get("unit") or "rows")  # tools.jsonl counts logical calls (#1316)
    if shown and matched > shown:
        rows = f"newest {shown} of {matched} matching {unit} in the window"
    elif shown and whole:
        rows = f"all {shown} matching {unit}"
    elif shown:
        rows = f"newest {shown} matching {unit} in the window"
    else:
        rows = f"no matching {unit}"
    if coverage.get("dialogue"):
        window = f"SQL coordinates [{coverage['lower']}, {coverage['upper']})"
        if coverage.get("older_available"):
            window += "; older matching rows available"
    elif unread:
        window = "unread"
    elif live_window >= live_size:
        window = "whole live file"
    else:
        window = f"live tail {_kb(live_window)} of {_kb(live_size)}"
    archives, available = int(coverage.get("archives") or 0), int(coverage.get("archives_available") or 0)
    if archives:
        window += f" + {archives} of {available} newest archives"
    rendered = str(coverage.get("rendered") or "")
    if rendered and shown:
        rows += f" ({rendered})"
    source = str(coverage.get("source") or "")
    parts = [f"{whose}: {rows}", f"window: {window}" + (f" of {source}" if source else "")]
    if coverage.get("archives_bounded"):
        parts.append("older archives not opened")
    gaps = coverage.get("gaps") or []
    if gaps:
        parts.append("gaps: " + ", ".join(str(g) for g in gaps))
    return "; ".join(parts)


__all__ = [
    "ARCHIVE_BACKFILL_MAX", "TAIL_WINDOW_START_BYTES", "archive_segments", "cached_tail_objects", "coverage_line",
    "read_jsonl_segment_with_gaps", "read_rotated_jsonl_entries", "remembered_listing", "task_archive_parser",
]
