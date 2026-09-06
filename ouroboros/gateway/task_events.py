"""Task-event SSE: legacy sorted replay and additive physical-cursor streaming.

The existing ``gateway.tasks`` exports remain the route and injection surface.
Both transports share current result/lineage projections and all five sources;
v2 owns only read positions, never event persistence or task state.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import pathlib
import time
from typing import Any, Dict, List, Optional

from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

from ouroboros.gateway._helpers import coerce_int, request_drive_root
from ouroboros.headless import ARTIFACT_STATUS_FINALIZING, ARTIFACT_STATUS_PENDING
from ouroboros.outcomes import public_task_result
from ouroboros.task_results import load_task_result, task_results_dir, validate_task_id
from ouroboros.task_status import FINAL_STATUSES
from ouroboros.gateway.task_list_scan import raw_result_facts
from ouroboros.gateway.contracts import TaskEventsRequest
from ouroboros.gateway.schema import validate_ingress
from ouroboros.utils import jsonl_archive_segments, jsonl_chain_handles


def _tasks_namespace():
    """``ouroboros.gateway.tasks``, resolved at CALL time (late-bound seam).

    Long-standing monkeypatch pins patch this code's collaborators on the
    module it lived in before the size-gate split: test_perf_budgets patches
    ``gateway.tasks._read_live_jsonl_entries``; test_task_events_sse patches
    ``gateway.tasks.load_effective_task_result`` and
    ``gateway.tasks.read_json_dict``. ``tasks.py`` binds those names (its own
    imports plus re-exports from here), so resolving them through that
    namespace keeps every existing pin effective while unpatched runs reach
    the exact same functions. The import is deferred to call time because
    ``tasks.py`` imports this module for its re-exports (module-load cycle).
    """
    from ouroboros.gateway import tasks

    return tasks


_LOG_SOURCES = (
    ("progress", ("logs", "progress.jsonl")),
    ("chat", ("logs", "chat.jsonl")),
    ("events", ("logs", "events.jsonl")),
    ("tools", ("logs", "tools.jsonl")),
    ("supervisor", ("logs", "supervisor.jsonl")),
)


async def api_task_events(request: Request) -> StreamingResponse:
    try:
        task_id = validate_task_id(request.path_params.get("task_id"))
    except ValueError as exc:
        message = str(exc)
        async def _bad_id():
            yield _sse({"type": "error", "error": message, "seq": 1}, event_id=1)
        return StreamingResponse(_bad_id(), media_type="text/event-stream", status_code=400)
    if getattr(request, "method", "GET") == "POST":
        return await _api_task_events_v2(request, task_id)
    cursor = max(0, coerce_int(request.query_params.get("cursor"), 0))
    wait_sec = max(0, min(coerce_int(request.query_params.get("wait"), 30), 120))
    drive_root = request_drive_root(request)
    if not load_task_result(drive_root, task_id):
        async def _missing():
            yield _sse({"type": "error", "error": "task not found", "task_id": task_id, "seq": 1}, event_id=1)
        return StreamingResponse(_missing(), media_type="text/event-stream", status_code=404)

    async def _legacy_events():
        # Initial replay = one full archive-aware merge (identical to a fresh
        # iter_task_events call, so the client's cross-reconnect `cursor` keeps
        # addressing the same positions — the CLI contract, ouroboros/cli.py
        # _watch_task). The follow phase then reads only bytes APPENDED to each
        # discovered log per tick; new rows are emitted incrementally with
        # monotonic in-stream seq only while they all sort strictly after the
        # emitted tail, otherwise one full re-merge resumes emission from the
        # cursor (at-least-once across those boundaries — pre-existing property,
        # disclosed in ARCHITECTURE.md).
        nonlocal cursor
        deadline = time.time() + wait_sec
        follower = _TaskEventFollower(drive_root, task_id)
        emitted_final = False
        tail_key = None
        need_full = True
        while True:
            refreshed = False
            advanced = False
            if need_full:
                rows = await asyncio.to_thread(follower.full_merge)
                pending = [row for row in rows if int(row.get("seq") or 0) > cursor]
                if rows:
                    tail_key = _event_sort_key(rows[-1])
                need_full = False
                refreshed = True  # full_merge reloaded the result projection
            else:
                new_rows, advanced = await asyncio.to_thread(follower.poll)
                interleaved = bool(new_rows) and tail_key is not None and _event_sort_key(new_rows[0]) <= tail_key
                if interleaved or follower.filter_grew:
                    # New rows interleave with already-emitted history, or a new
                    # child id joined the lineage filter (rows matching only via
                    # subagent_task_id may sit in already-consumed bytes): ONE
                    # full re-merge, resume emission from the cursor.
                    rows = await asyncio.to_thread(follower.full_merge)
                    pending = [row for row in rows if int(row.get("seq") or 0) > cursor]
                    if rows:
                        tail_key = _event_sort_key(rows[-1])
                    refreshed = True
                else:
                    pending = []
                    for row in new_rows:
                        row["seq"] = cursor + len(pending) + 1
                        pending.append(row)
                    if pending:
                        tail_key = _event_sort_key(pending[-1])
            if follower._lineage_notice is not None:
                notice, follower._lineage_notice = follower._lineage_notice, None
                # A read diagnostic is not a legacy history row/rank.
                yield _sse({**notice, "seq": cursor}, event_id=cursor)
            for event in pending:
                cursor = int(event.get("seq") or cursor)
                if str(event.get("type") or "") == "task_result":
                    data = event.get("data") if isinstance(event.get("data"), dict) else {}
                    if str(data.get("status") or "").lower() in FINAL_STATUSES:
                        if not emitted_final:
                            # ONE materializing read at terminal emission (P2
                            # review, fix 5): the merged rows are status/cost
                            # projections, but watching a task to completion
                            # must still deliver the artifact-bearing terminal
                            # payload (and run its read-repair rebase) exactly
                            # once per stream.
                            full = await asyncio.to_thread(
                                _tasks_namespace().load_effective_task_result, drive_root, task_id
                            )
                            if full:
                                event["data"] = public_task_result(full)
                        emitted_final = True
                yield _sse(event, event_id=cursor)
            # Recompute the terminal projection only when something moved: log
            # offsets advanced, new roots joined, or the queue snapshot changed.
            if not refreshed and (advanced or follower.queue_snapshot_changed()):
                suppress_before = follower.suppress_task_done
                await asyncio.to_thread(follower.refresh_result)
                if follower.suppress_task_done != suppress_before:
                    # The task_done suppression window opened/closed: which rows
                    # exist in the merge changed, so re-merge before continuing.
                    need_full = True
                    continue
            if follower.result_is_final():
                if not emitted_final:
                    result = public_task_result(
                        _tasks_namespace().load_effective_task_result(drive_root, task_id)
                    )
                    if result:
                        final_event = {
                            "source": "task_result",
                            "line": 0,
                            "ts": str(result.get("ts") or ""),
                            "type": "task_result",
                            "task_id": task_id,
                            "data": result,
                            "seq": cursor + 1,
                        }
                        cursor = int(final_event["seq"])
                        yield _sse(final_event, event_id=cursor)
                break
            if time.time() >= deadline:
                yield ": heartbeat\n\n"
                break
            await asyncio.sleep(0.5)

    async def _stream():
        try:
            async for frame in _legacy_events():
                yield frame
        except (OSError, ValueError) as exc:
            yield _sse({"type": "error", "task_id": task_id, "seq": cursor + 1,
                        "error": str(exc), "reason": "history_unavailable"}, event_id=cursor + 1)

    return StreamingResponse(_stream(), media_type="text/event-stream")


# Live logs that the supervisor rotates into archive/<prefix>_<ts>.jsonl
# (supervisor/state.rotate_jsonl_log_if_needed). Every source served here rotates.
_ROTATED_LOG_PREFIXES = {source: source for source, _parts in _LOG_SOURCES}


def _event_sort_key(item: Dict[str, Any]) -> tuple:
    return (str(item.get("ts") or ""), str(item.get("source") or ""), int(item.get("line") or 0))


def _compact_ts_stamp(ts: str) -> str:
    """ISO-ish timestamp -> archive-stamp form (YYYYMMDDTHHMMSS), or "" if unusable."""
    stamp = ts.strip().replace("-", "").replace(":", "")
    return stamp[:15] if len(stamp) >= 15 and stamp[8:9] == "T" else ""


def _archive_stamp_predates(name: str, prefix: str, floor: str) -> bool:
    """True when ``<prefix>_<stamp>[_N].jsonl`` was rotated strictly before ``floor``."""
    stamp = name[len(prefix) + 1:].split(".", 1)[0].split("_", 1)[0]
    return len(stamp) == 15 and stamp < floor


def _read_live_jsonl_entries(path: pathlib.Path, offset: int) -> tuple[List[Dict[str, Any]], int, Optional[int]]:
    """Parse COMPLETE JSONL lines from byte ``offset``; returns (entries, new_offset, ino).

    A torn final line (a concurrent append caught mid-write) is left unconsumed so
    the next read starts exactly at its first byte — unlike a naive full read, no
    row is ever half-parsed and then skipped forever."""
    try:
        with path.open("rb") as handle:
            stat = os.fstat(handle.fileno())
            if offset:
                handle.seek(offset)
            data = handle.read()
    except OSError:
        return [], offset, None
    cut = data.rfind(b"\n")
    if cut < 0:
        return [], offset, stat.st_ino
    chunk = data[: cut + 1]
    entries: List[Dict[str, Any]] = []
    for raw in chunk.splitlines():
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    return entries, offset + len(chunk), stat.st_ino


class _TaskEventFollower:
    """Byte-offset follow state for one ``/api/tasks/{id}/events`` stream.

    ``full_merge`` performs the complete archive-aware scan (also serving as the
    public ``iter_task_events``) while rebuilding per-(root, source) chain state:
    consumed archive names, the live-file byte offset/inode, and the running
    parsed-line count that keeps (ts, source, line) ordering identical between
    incremental reads and a re-merge. ``poll`` then reads only appended bytes,
    re-discovers late-spawned child roots every tick (their logs join at offset
    0), and heals a mid-stream rotation by reading the newest archive's suffix
    beyond the old offset before continuing on the new live file at offset 0.
    All effective-result reads here are status/cost projections
    (``materialize_artifacts=False``) — the SSE loop must never copy artifacts
    or make disposition/sha claims on a 0.5s tick. The single sanctioned
    exception lives in the stream's emit loop, not here: the terminal
    ``task_result`` emission performs one materializing read (see
    ``api_task_events``)."""

    def __init__(self, drive_root: pathlib.Path, task_id: str) -> None:
        self.drive_root = pathlib.Path(drive_root)
        self.task_id = task_id
        self.task_filter_ids = {task_id}
        self.roots: List[pathlib.Path] = []
        self.logs: Dict[tuple, Dict[str, Any]] = {}
        self.result: Dict[str, Any] = {}
        self.suppress_task_done = False
        self.filter_grew = False
        self._queue_snapshot_mtime: Any = None
        self._results_dir = task_results_dir(self.drive_root, create=False)
        # Per-file proof belongs to this live follower, never a client cursor
        # or the shared current-facts memo. Failed reads may retain this proof.
        self._seen_result_names: Dict[str, tuple[str, str]] = {}
        self._lineage_failed_names: set[str] = set()
        self._lineage_notice: Optional[dict] = None
        # Only an explicit creation fact may bound the scan. Legacy ``ts`` can
        # describe finalization and must never hide earlier task history.
        raw = load_task_result(self.drive_root, task_id) or {}
        self._created_floor = _compact_ts_stamp(str(raw.get("created_at") or ""))

    def refresh_result(self) -> None:
        self.result = _tasks_namespace().load_effective_task_result(
            self.drive_root, self.task_id, materialize_artifacts=False
        )
        self.suppress_task_done = _is_workspace_result(self.result) and str(
            self.result.get("artifact_status") or ""
        ).lower() in {ARTIFACT_STATUS_PENDING, ARTIFACT_STATUS_FINALIZING}

    def result_is_final(self) -> bool:
        return str(self.result.get("status") or "").lower() in FINAL_STATUSES

    def queue_snapshot_changed(self) -> bool:
        try:
            mtime = (self.drive_root / "state" / "queue_snapshot.json").stat().st_mtime_ns
        except OSError:
            mtime = None
        changed = mtime != self._queue_snapshot_mtime
        self._queue_snapshot_mtime = mtime
        return changed

    def _discover_roots(self) -> bool:
        """Rebuild the filter from current compact facts, reusing unchanged rows.

        Results are written with lineage before enqueue. Stat invalidation also
        sees later changes to an existing row; reconnects share that same memo.
        Discovery is a read-only projection, never result/schema authority.
        """
        candidates = [self.drive_root]
        child = str(self.result.get("child_drive_root") or self.result.get("headless_child_drive_root") or "").strip()
        if child:
            candidates.append(pathlib.Path(child))
        facts, malformed = raw_result_facts(
            self._results_dir, reader=_tasks_namespace().read_json_dict,
        )
        failed = set(malformed)
        bindings = {name: binding for name, binding in self._seen_result_names.items() if name in failed}
        retained_count = len(bindings)
        for name, row in facts.items():
            if row["schema_refusal"] or row["delegation_role"] != "subagent":
                continue
            child_id = row["task_id"] or row["id"]
            if not child_id or not (row["parent_task_id"] == self.task_id or row["root_task_id"] == self.task_id):
                continue
            child_root = row["child_drive_root"] or row["headless_child_drive_root"]
            bindings[name] = (child_id, child_root)
        ids = {self.task_id, *(child_id for child_id, _root in bindings.values())}
        for _child_id, child_root in bindings.values():
            if child_root:
                candidates.append(pathlib.Path(child_root))
        self._seen_result_names = bindings
        if failed != self._lineage_failed_names:
            self._lineage_notice = ({"type": "history_gap", "source": "task_result",
                "task_id": self.task_id, "reason": "lineage_incomplete",
                "data": {"failed_result_reads": len(failed), "retained_result_bindings": retained_count,
                         "unknown_result_membership": len(failed) - retained_count}}
                if failed else None)
        self._lineage_failed_names = failed
        roots = sorted(set(candidates), key=str)
        changed = ids != self.task_filter_ids or roots != self.roots
        self.filter_grew = self.filter_grew or changed
        self.task_filter_ids, self.roots = ids, roots
        return changed

    def _log_state(self, root: pathlib.Path, source: str) -> Dict[str, Any]:
        key = (str(root), source)
        state = self.logs.get(key)
        if state is None:
            state = {"archives": [], "offset": 0, "ino": None, "lines": 0}
            self.logs[key] = state
        return state

    def _read_chain_delta(self, root: pathlib.Path, source: str, parts: tuple) -> List[Dict[str, Any]]:
        """Entries appended to one (root, source) chain since the recorded state.

        Fresh state (a late-discovered log) naturally degenerates to reading the
        whole chain: every archive is "new" and the live offset is 0."""
        state = self._log_state(root, source)
        live = root.joinpath(*parts)
        prefix = _ROTATED_LOG_PREFIXES.get(source)
        entries: List[Dict[str, Any]] = []
        if prefix:
            try:
                archive_paths = sorted(
                    (root / "archive").glob(f"{prefix}_*.jsonl"), key=lambda p: p.name
                )
            except OSError:
                archive_paths = []
            if self._created_floor:
                # An archive rotated before the watched task existed cannot
                # contain its rows (an archive's rows predate its rotation
                # stamp), so skip it: bounds the per-tick/merge archive work to
                # the task's lifetime instead of O(system age). Removes no
                # matching rows and touches no cursor positions by construction.
                archive_paths = [
                    path for path in archive_paths
                    if not _archive_stamp_predates(path.name, prefix, self._created_floor)
                ]
            known = set(state["archives"])
            new_archives = [p for p in archive_paths if p.name not in known]
            if new_archives:
                # Rotation: the previous live content now lives in the newest
                # archive(s). Read the first new archive beyond the consumed live
                # offset (or the offset stashed when the inode flip was observed
                # before the archive became visible), the rest fully, then
                # continue on the new live file from 0.
                had_stash = "rotated_offset" in state
                start = state.pop("rotated_offset", state["offset"])
                for index, path in enumerate(new_archives):
                    got, _, _ = _tasks_namespace()._read_live_jsonl_entries(path, start if index == 0 else 0)
                    entries.extend(got)
                    state["archives"].append(path.name)
                if not (had_stash and len(new_archives) == 1):
                    # No stash: offset/ino still describe the OLD live file (now
                    # the archive), so restart on the new live file from 0.
                    # With a consumed stash and exactly one new archive, the
                    # recorded offset/ino already track the NEW live file the
                    # follower partially consumed on the stash tick — resetting
                    # to 0 would re-emit those rows (P2 review, fix 2).
                    state["offset"] = 0
                    state["ino"] = None
        try:
            live_stat = live.stat()
        except OSError:
            return entries
        if (state["ino"] is not None and live_stat.st_ino != state["ino"]) or (
            live_stat.st_size < state["offset"]
        ):
            # Live file replaced/shrank but its archive is not visible yet: stash
            # the consumed offset for the archive suffix and restart on the new
            # live file. (Any resulting duplicate rows sort at-or-before the
            # emitted tail, which forces a full re-merge — the honest fallback.)
            if prefix and "rotated_offset" not in state:
                state["rotated_offset"] = state["offset"]
            state["offset"] = 0
        got, new_offset, ino = _tasks_namespace()._read_live_jsonl_entries(live, state["offset"])
        state["offset"], state["ino"] = new_offset, ino
        entries.extend(got)
        return entries

    def _entries_to_rows(
        self, root: pathlib.Path, source: str, entries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        state = self._log_state(root, source)
        rows: List[Dict[str, Any]] = []
        for entry in entries:
            state["lines"] += 1
            entry_task = str(entry.get("task_id") or "")
            entry_subagent = str(entry.get("subagent_task_id") or "")
            entry_parent = str(entry.get("parent_task_id") or "")
            entry_root = str(entry.get("root_task_id") or "")
            if (
                entry_task not in self.task_filter_ids
                and entry_subagent not in self.task_filter_ids
                and entry_parent != self.task_id
                and entry_root != self.task_id
            ):
                continue
            event = _event_from_log_entry(source, state["lines"], entry, root)
            if self.suppress_task_done and event.get("type") == "task_done":
                continue
            rows.append(event)
        return rows

    def full_merge(self) -> List[Dict[str, Any]]:
        """Full archive-aware merge; rebuilds ALL follow state from scratch."""
        self.logs = {}
        self.roots = []
        self.task_filter_ids = {self.task_id}
        # The process-local stat memo survives a connection/replay rebuild.
        # A replay rebuilds log positions, not the live follower's prior proof.
        self.refresh_result()
        self.queue_snapshot_changed()
        self._discover_roots()
        rows: List[Dict[str, Any]] = []
        for root in self.roots:
            for source, parts in _LOG_SOURCES:
                entries = self._read_chain_delta(root, source, parts)
                rows.extend(self._entries_to_rows(root, source, entries))
        if self.result:
            rows.append({
                "source": "task_result",
                "line": 0,
                "ts": str(self.result.get("ts") or ""),
                "type": "task_result",
                "task_id": self.task_id,
                "data": public_task_result(self.result),
            })
        rows.sort(key=_event_sort_key)
        for idx, row in enumerate(rows, 1):
            row["seq"] = idx
        self.filter_grew = False  # the merge above read every consumed byte anew
        return rows

    def poll(self) -> tuple[List[Dict[str, Any]], bool]:
        """One follow tick: (new rows sorted by (ts, source, line), advanced?)."""
        advanced = self._discover_roots()
        rows: List[Dict[str, Any]] = []
        for root in list(self.roots):
            for source, parts in _LOG_SOURCES:
                entries = self._read_chain_delta(root, source, parts)
                if entries:
                    advanced = True
                    rows.extend(self._entries_to_rows(root, source, entries))
        rows.sort(key=_event_sort_key)
        return rows, advanced


def _cursor_input(value: Any) -> Dict[str, Any]:
    """Validate only the versioned transport; paths never select read authority."""
    if value is None:
        return {"v": 2, "seq": 0, "view": "", "positions": {}}
    if value["v"] != 2 or value["seq"] < 0:
        raise ValueError("cursor version or sequence is invalid")
    if not value["view"]:
        raise ValueError("cursor view or positions is invalid")
    for root, sources in value["positions"].items():
        if not isinstance(root, str) or not isinstance(sources, dict):
            raise ValueError("cursor root positions must be objects")
        if any(source not in _ROTATED_LOG_PREFIXES or type(offset) is not int or offset < 0
               for source, offset in sources.items()):
            raise ValueError("cursor source position is invalid")
    return value


class _TaskEventCursorFollower(_TaskEventFollower):
    """Physical append positions over immutable archives plus the live file.

    Positions count the complete chain, independently of creation-floor skips.
    No timestamp sorting or history-sized event batches occur on this path.
    Archives must remain immutable and retained: a shorter/unreadable chain
    refuses continuation. Manual prefix removal masked by new appended bytes
    cannot be detected by this offset protocol and is outside its guarantee.
    """

    def __init__(self, drive_root: pathlib.Path, task_id: str, cursor: dict) -> None:
        super().__init__(drive_root, task_id)
        self.seq = cursor["seq"]
        self.view = cursor["view"]
        self.positions = {root: dict(sources) for root, sources in cursor["positions"].items()}

    def checkpoint(self) -> dict:
        return {"v": 2, "seq": self.seq, "view": self.view,
                "positions": {root: dict(sources) for root, sources in self.positions.items()}}

    def envelope(self, event: dict, *, identity: str = "", delivery: bool = True) -> dict:
        if delivery:
            self.seq += 1
        return {**event, "seq": self.seq, "event_id": identity, "cursor": self.checkpoint()}

    def refresh_view(self) -> Optional[dict]:
        self.refresh_result()
        self._discover_roots()
        facts = {"task_id": self.task_id, "task_ids": sorted(self.task_filter_ids),
                 "roots": [str(root) for root in self.roots],
                 "suppress_task_done": self.suppress_task_done, "creation_floor": self._created_floor}
        view = hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest()
        expected = {str(root): {source: 0 for source, _parts in _LOG_SOURCES} for root in self.roots}
        changed = bool(self.view) and view != self.view
        if not self.view or changed:
            self.positions = expected
        elif (set(self.positions) != set(expected)
              or any(set(self.positions[root]) != set(sources) for root, sources in expected.items())):
            raise ValueError("cursor positions do not cover this view")
        self.view = view
        if changed:
            return self.envelope({"type": "cursor_replay", "task_id": self.task_id,
                                  "reason": "view_changed"}, delivery=False)
        return None

    def read_events(self):
        """Close each bounded read before yielding its rows and per-row cursors.

        The buffer is at most 64 KiB plus one complete line. A single large
        JSONL record keeps its existing support; history is never one batch.
        Each source's logical EOF is pinned for this pass; later appends wait
        for the next pass so a busy source cannot starve the rest of the view.
        """
        if self._lineage_notice is not None:
            notice, self._lineage_notice = self._lineage_notice, None
            yield self.envelope(notice, delivery=False)
        for root in self.roots:
            for source, parts in _LOG_SOURCES:
                live = root.joinpath(*parts)
                position = self.positions[str(root)][source]
                if position == 0 and self._created_floor:
                    # Skip the known pre-birth prefix in one metadata pass,
                    # without opening each old archive as a separate batch.
                    for path in jsonl_archive_segments(live, strict=True):
                        if not _archive_stamp_predates(path.name, source, self._created_floor):
                            break
                        position += path.stat().st_size
                    self.positions[str(root)][source] = position
                snapshot: dict = {}
                while not snapshot or position < snapshot["total"]:
                    before = position
                    with jsonl_chain_handles(live, strict=True, start_offset=position,
                                             snapshot=snapshot) as handles:
                        end_position = snapshot["total"]
                        if not handles:
                            break
                        path, handle = handles[0]
                        remaining = os.fstat(handle.fileno()).st_size - handle.tell()
                        snapshot_cut = remaining > end_position - position
                        remaining = min(remaining, end_position - position)
                        if (path != live and self._created_floor
                                and _archive_stamp_predates(path.name, source, self._created_floor)):
                            position += remaining
                            self.positions[str(root)][source] = position
                            continue
                        data = handle.read(min(64 * 1024, remaining))
                        if data and not data.endswith(b"\n") and len(data) < remaining:
                            data += handle.readline(remaining - len(data))
                    for raw in io.BytesIO(data):
                        if not raw.endswith(b"\n") and (path == live or snapshot_cut):
                            break  # an append still owns this unfinished row
                        row_start = position
                        position += len(raw)
                        self.positions[str(root)][source] = position
                        identity = json.dumps([str(root), source, row_start], separators=(",", ":"))
                        try:
                            if not raw.endswith(b"\n"):
                                raise ValueError("incomplete_archive_line")
                            if not raw.strip():
                                continue
                            entry = json.loads(raw.decode("utf-8"))
                            if not isinstance(entry, dict):
                                raise ValueError("non_object_line")
                        except (ValueError, UnicodeDecodeError):
                            yield self.envelope({"type": "history_gap", "source": source,
                                "root": str(root), "task_id": self.task_id,
                                "reason": "invalid_archive_line" if path != live else "invalid_jsonl_line"},
                                identity=identity, delivery=False)
                            continue
                        if (str(entry.get("task_id") or "") not in self.task_filter_ids
                                and str(entry.get("subagent_task_id") or "") not in self.task_filter_ids
                                and str(entry.get("parent_task_id") or "") != self.task_id
                                and str(entry.get("root_task_id") or "") != self.task_id):
                            continue
                        event = _event_from_log_entry(source, 0, entry, root)
                        # Legacy ``line`` counts parsed rows. A resumed byte
                        # reader cannot reconstruct that count without replay.
                        event.pop("line")
                        if self.suppress_task_done and event["type"] == "task_done":
                            continue
                        yield self.envelope(event, identity=identity)
                    if position == before:
                        break


async def _api_task_events_v2(request: Request, task_id: str):
    try:
        body = await request.json()
        if errors := validate_ingress(body, TaskEventsRequest):
            return JSONResponse({"error": f"invalid request body: {errors[0]}",
                                 "schema_errors": errors[:8]}, status_code=400)
        cursor = _cursor_input(body.get("cursor"))
        wait = body.get("wait", 30)
        if type(wait) is not int or not 0 <= wait <= 120:
            raise ValueError("wait must be an integer from 0 to 120 seconds")
    except (ValueError, TypeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    drive_root = request_drive_root(request)
    if not load_task_result(drive_root, task_id):
        return JSONResponse({"error": "task not found", "task_id": task_id}, status_code=404)
    follower = _TaskEventCursorFollower(drive_root, task_id, cursor)

    async def stream():
        deadline = time.monotonic() + wait
        first = True
        try:
            while True:
                replay = await asyncio.to_thread(follower.refresh_view)
                if replay:
                    yield _sse(replay, event_id=follower.seq)
                rows = follower.read_events()
                try:
                    while True:
                        read = asyncio.create_task(asyncio.to_thread(next, rows, None))
                        try:
                            event = await asyncio.shield(read)
                        except asyncio.CancelledError:
                            # A cancelled HTTP waiter does not stop its file
                            # read thread. Settle it before closing the handles.
                            await read
                            raise
                        if event is None:
                            break
                        yield _sse(event, event_id=event["seq"])
                finally:
                    rows.close()
                terminal = follower.result_is_final()
                if first or terminal:
                    result = follower.result
                    if terminal:
                        result = await asyncio.to_thread(
                            _tasks_namespace().load_effective_task_result, drive_root, task_id,
                        )
                    event = follower.envelope({"source": "task_result", "type": "task_result",
                        "task_id": task_id, "data": public_task_result(result)})
                    # Synthetic result snapshots have no log identity. They are
                    # always consumed, including the fresh terminal materialization.
                    yield _sse(event, event_id=follower.seq)
                    first = False
                if terminal:
                    break
                if time.monotonic() >= deadline:
                    event = follower.envelope({"type": "cursor_checkpoint", "task_id": task_id}, delivery=False)
                    yield _sse(event, event_id=follower.seq)
                    break
                await asyncio.sleep(0.5)
        except (OSError, ValueError) as exc:
            event = follower.envelope({"type": "error", "task_id": task_id,
                "error": str(exc), "reason": "cursor_unavailable"}, delivery=False)
            yield _sse(event, event_id=follower.seq)

    return StreamingResponse(stream(), media_type="text/event-stream")


def iter_task_events(drive_root: pathlib.Path, task_id: str) -> List[Dict[str, Any]]:
    """Return synthesized replayable events for a task from existing logs.

    Archive-aware (v6.90.x P2): each rotated log's ``archive/<prefix>_*.jsonl``
    chain is read oldest-first before the live file, so a rotation never erases
    replay history. Also the SSE initial-replay/re-merge path."""
    return _TaskEventFollower(drive_root, task_id).full_merge()


def _event_from_log_entry(source: str, line_no: int, entry: Dict[str, Any], root: pathlib.Path) -> Dict[str, Any]:
    event_type = str(entry.get("type") or source)
    if source == "progress":
        event_type = "progress"
    elif source == "chat":
        event_type = "message"
    elif source == "tools":
        event_type = "tool_call"
    data = dict(entry)
    data = public_task_result(
        data,
        include_outcome_axes=any(key in data for key in ("status", "outcome_axes", "result_status", "loop_outcome")),
    )
    return {
        "source": source,
        "line": line_no,
        "ts": str(entry.get("ts") or ""),
        "type": event_type,
        "task_id": str(entry.get("task_id") or ""),
        "root": str(root),
        "data": data,
    }


def _is_workspace_result(result: Dict[str, Any]) -> bool:
    return bool(str(result.get("workspace_root") or "").strip() or str(result.get("workspace_mode") or "").strip())


def _sse(event: Dict[str, Any], *, event_id: int) -> str:
    payload = json.dumps(event, ensure_ascii=False)
    return f"id: {event_id}\nevent: task_event\ndata: {payload}\n\n"


__all__ = ["api_task_events", "iter_task_events"]
