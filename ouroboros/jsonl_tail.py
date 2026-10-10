"""Rotation-aware JSONL reads: captured byte ranges and bounded filtered tails.

``JsonlChainSnapshot`` shares physical capture/read/segment ownership between
history pagination, wake observations and the reflection's transport-receipt
evidence, whose selection policies remain separate. The filtered-tail reader
below serves context and endpoints.

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
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from ouroboros.utils import JsonlChainUnreadable, iter_jsonl_objects, jsonl_chain_handles

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


def read_jsonl_segment_with_gaps(
    path: pathlib.Path,
    *,
    tail_bytes: Optional[int] = None,
    iter_objects: Optional[Callable[..., Iterable[Any]]] = None,
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
        path.stat()
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
) -> list | tuple[list, set[str]]:
    """Bounded, rotation-aware read of one JSONL log (module docstring).

    ``iter_objects`` is the parser seam (the gateway wrapper passes its own
    name so its tests keep governing it). ``list_archives`` is the listing seam
    with ``archive_segments``' signature: a request that reads many tasks lists
    ``archive/`` once and replays its gap to each (``replay_evidence_for_tasks``).
    ``coverage``, when given, is filled with ``live_size``, ``live_window``
    (bytes of the live file read; both absent when the live file could not be
    read), ``archives`` (consulted count), ``archives_available`` and
    ``archives_bounded``.
    """
    live = pathlib.Path(live)
    parse = iter_objects or iter_jsonl_objects  # module name resolved at call time (test seam)
    gaps: set[str] = set()
    live_readable = True
    try:
        size = live.stat().st_size
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
                live_entries, live_gaps = read_jsonl_segment_with_gaps(live, iter_objects=parse)
                gaps.update(live_gaps)
            else:
                live_entries = list(parse(live))
            window = size
            break
        if collect:
            live_entries, live_gaps = read_jsonl_segment_with_gaps(
                live, tail_bytes=window, iter_objects=parse)
            gaps.update(live_gaps)
        else:
            live_entries = list(parse(live, tail_bytes=window))
        if sum(1 for entry in live_entries if counts_toward_quota(entry)) >= want:
            break
        window *= 2
    collected = sum(1 for entry in live_entries if counts_toward_quota(entry))
    archives = (list_archives or archive_segments)(pathlib.Path(archive_dir), archive_prefix, gaps if collect else None)
    chosen: list = []
    for archive_path in archives:
        if collected >= want or len(chosen) >= max_archives:
            break
        try:
            if collect:
                archive_entries, archive_gaps = read_jsonl_segment_with_gaps(
                    archive_path, iter_objects=parse)
                gaps.update(archive_gaps)
            else:
                archive_entries = list(parse(archive_path))
        except Exception:
            gaps.add("unreadable_source")
            continue
        chosen.append(archive_entries)
        collected += sum(1 for entry in archive_entries if counts_toward_quota(entry))
    ordered: list = []
    for archive_entries in reversed(chosen):  # oldest chosen archive first
        ordered.extend(archive_entries)
    ordered.extend(live_entries)
    if coverage is not None:
        coverage.update({
            "archives": len(chosen), "archives_available": len(archives),
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
    whole = live_window >= live_size and not coverage.get("archives_bounded") and not unread
    unit = str(coverage.get("unit") or "rows")  # tools.jsonl counts logical calls (#1316)
    if shown and matched > shown:
        rows = f"newest {shown} of {matched} matching {unit} in the window"
    elif shown and whole:
        rows = f"all {shown} matching {unit}"
    elif shown:
        rows = f"newest {shown} matching {unit} in the window"
    else:
        rows = f"no matching {unit}"
    if unread:
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
    "ARCHIVE_BACKFILL_MAX", "TAIL_WINDOW_START_BYTES", "archive_segments", "coverage_line",
    "read_jsonl_segment_with_gaps", "read_rotated_jsonl_entries",
]
