"""Read-only runtime log endpoints for headless clients.

``chat`` and ``progress`` are the dialogue store's streams (``state/dialog.sqlite``): the tail is
an addressed read -- the newest rows, or one indexed probe per task identifier -- whose cost is
the page, never another room's history; ``_line`` is the row's durable position among every
stored object of its stream (``dialogue_positions.ordinals_of``), the count the chain tail kept.
While the archive import has not settled those counts the endpoint answers 503 ``migrating``,
never a guessed number. ``events``, ``tools`` and ``supervisor`` stay rotated files, read through
the bounded window-doubling tail (``_line`` is a position within that read window).
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import pathlib
import re
from bisect import bisect_right
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, TypedDict

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from ouroboros import dialogue_store as ds
from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.confined_files import physical_path, root_spellings
from ouroboros.dialogue_positions import ordinals_of
from ouroboros.gateway._helpers import (
    coerce_int,
    json_error,
    read_rotated_jsonl_entries,
    request_drive_root,
)
from ouroboros.jsonl_tail import JsonlChainSnapshot, parse_jsonl_source, read_rotated_jsonl_sources
from ouroboros.outcomes import public_task_result
from ouroboros.source_retention import retained_child_root
from ouroboros.store import StoreError, StoreMigrating
from ouroboros.task_custody import child_drive_candidates
from ouroboros.task_reads import TaskReads
from ouroboros.task_results import validate_task_id
from ouroboros.task_store import select_lineage
from ouroboros.utils import JsonlChainUnreadable

_ALLOWED_LOGS = {
    "chat": "chat.jsonl",
    "progress": "progress.jsonl",
    "events": "events.jsonl",
    "tools": "tools.jsonl",
    "supervisor": "supervisor.jsonl",
}
_DIALOGUE = frozenset(ds.STREAMS)  # the logs the dialogue store holds; the rest are rotated files
_TASK_COLUMNS = ("task_id", "parent_task_id", "root_task_id", "subagent_task_id")
# A task filter reads candidate pages until the limit is met; a tree whose probes name mostly rows
# the filter refuses (another root's children) is bounded here, never read whole.
_CANDIDATE_PAGES = 10
_COMPACT_BYTES = 16 * 1024
_RAW_PAGE_BYTES = 64 * 1024
_RAW_PAGE_MAX = 1024 * 1024


class LogRowSource(TypedDict):
    """Stored JSONL byte identity, never a public projection digest or window row number."""
    v: Literal[1]
    kind: Literal["log_row"]
    placement: str  # canonical, or server-admitted path relative to the canonical drive
    task_id: str  # the original query's task scope, empty for canonical unfiltered logs
    log: str
    start: int  # chain byte coordinate; inclusive
    length: int  # includes the original line terminator
    sha256: str


class LogRawReference(TypedDict):
    ref: str  # base64url JSON of source; selector, never authorization
    source: LogRowSource
    omitted_fields: list[str]  # at most 64 names, each at most 128 characters
    omitted_field_count: int
    public_bytes: Optional[int]  # None when reconstructed: Raw reads the receipt, the preview never does
    # reconstructed: a context view checkpoint's own at_source names structures held by its retained receipt
    representation: Literal["stored", "public_task_result", "reconstructed"]


class LogRawResponse(TypedDict):
    source: LogRowSource
    representation: Literal["stored", "public_task_result", "reconstructed"]
    sha256: str  # bytes of the served representation, not source.sha256
    total_bytes: int
    offset: int
    next_offset: int
    has_more: bool
    text: str
    # A reconstruction whose receipt could not be used serves the stored event whole without these declared fields.
    unavailable_fields: list[str]
    unavailable_reason: Literal["", "receipt_unavailable", "receipt_mismatch"]


class CompactLogEntry(TypedDict, total=False):
    _raw: LogRawReference  # absent for unchanged dialogue streams; other public fields are open


class CompactLogTailResponse(TypedDict):
    name: str
    entries: list[CompactLogEntry]
    view: Literal["compact"]
    gaps: list[str]  # damage observed in the selected read window, not all history
    skipped_rows: int  # measured nonblank unaddressable records before task/limit filtering


def replica_key(row: Any) -> Optional[tuple]:
    """A row the passport writes to the task's drive and the canonical log alike (REPLICA_RULE), by its
    natural key, so a reader of both keeps one; None for every other row and for one missing a key field."""
    from ouroboros.contracts.record_contract import ANCHOR_BY_TYPE

    anchor = ANCHOR_BY_TYPE.get(str(row.get("type") or "")) if isinstance(row, dict) else None
    key = tuple(row.get(name) for name in anchor.natural_key) if anchor is not None and anchor.plane == "both" else ()
    return (anchor.type, *key) if key and all(value not in (None, "") for value in key) else None


async def api_logs_tail(request: Request) -> JSONResponse:
    # Child discovery, archived reads and projection can all touch disk.
    return await asyncio.to_thread(_logs_tail, request)


def _logs_tail(request: Request) -> JSONResponse:
    name = str(request.path_params.get("name") or "").strip().lower()
    filename = _ALLOWED_LOGS.get(name)
    if not filename:
        return json_error(f"unknown log {name!r}", 404, allowed=sorted(_ALLOWED_LOGS))
    view = request.query_params.get("view", "full")
    if view not in {"full", "compact"}:
        return json_error("unknown log view", 400, reason_code="invalid_view")
    if view == "compact" and name not in _DIALOGUE:
        return _compact_tail(request, name)
    limit = max(1, min(coerce_int(request.query_params.get("limit"), 100), 2000))
    task_id = str(request.query_params.get("task_id") or "").strip()
    drive_root = request_drive_root(request)
    rows: List[Dict[str, Any]] = []
    try:
        placements, task_filter_ids = _scope(drive_root, task_id)
        def _matches_task_filter(entry: Any) -> bool:
            return isinstance(entry, dict) and _matches(entry, task_id, task_filter_ids)
        for placement, root in placements.items():
            if name in _DIALOGUE:
                numbered = _dialogue_tail(root, name, limit, task_filter_ids if task_id else None, _matches_task_filter)
            else:
                # A bounded live byte tail plus newest-first archive backfill until the filtered
                # quota is met; ``_line`` is a position within this read window, not the whole file.
                numbered = list(enumerate(read_rotated_jsonl_entries(
                    root / "logs" / filename, root / "archive", name, limit, _matches_task_filter), 1))
            for line_no, entry in numbered:
                if not isinstance(entry, dict) or not _matches_task_filter(entry):
                    continue
                item = public_task_result(
                    entry,
                    include_outcome_axes=any(key in entry for key in ("status", "outcome_axes", "result_status", "loop_outcome")),
                )
                # Full keeps its configured-root spelling; placement authority
                # comes from the same physical scope as compact and Raw.
                source_root = drive_root if placement == "canonical" else drive_root / placement
                item.setdefault("_source_root", str(source_root))
                item.setdefault("_line", line_no)
                rows.append(item)
    except StoreMigrating:
        return json_error("the dialogue is still importing; retry shortly", 503, reason_code="migrating")
    except StoreError as exc:
        return json_error(f"the dialogue store is unavailable: {exc}", 503, reason_code="history_source_unavailable")
    except OSError:
        return json_error("log scope is unavailable", 503, reason_code="log_source_unavailable")
    except (ValueError, RecursionError):
        return json_error("log scope cannot be resolved", 400, reason_code="invalid_log_source")
    rows.sort(key=lambda item: (str(item.get("ts") or ""), str(item.get("_source_root") or ""), int(item.get("_line") or 0)))
    seen: set = set()
    rows = [row for row in rows if (key := replica_key(row)) is None or not (key in seen or seen.add(key))]
    return JSONResponse({"name": name, "entries": rows[-limit:],
                         **({"view": "compact", "gaps": [], "skipped_rows": 0} if view == "compact" else {})})


def _dialogue_tail(root: pathlib.Path, stream: str, limit: int, task_ids: Optional[set],
                   matches: Callable[[Any], bool]) -> List[Tuple[int, Dict[str, Any]]]:
    """The newest ``limit`` rows of one dialogue stream that pass the task filter, oldest first,
    each with its ``_line``: the row's position among every stored object of the stream, read
    from the store's durable ordinals (typed ``StoreMigrating`` while the archive import has not
    settled them). Without a filter the tail is the stream's newest rows; with one it is the
    union of one indexed probe per task identifier, paged newest first until the limit."""
    if task_ids is None:
        rows = ds.newest(root, stream, limit=limit)
    else:
        keys = [(column, task) for task in sorted(task_ids) for column in _TASK_COLUMNS]
        rows, before = [], None
        for _page in range(_CANDIDATE_PAGES):
            page = ds.probe_rows(root, stream, keys, before=before, limit=limit, newest_first=True)
            rows.extend(row for row in page if matches(row.row))
            if len(rows) >= limit or len(page) < limit:
                break
            before = page[-1].seq
        rows = sorted(rows[:limit], key=lambda row: row.seq)
    ordinals = ordinals_of(root, stream, [row.seq for row in rows])
    return [(ordinals[row.seq].ordinal + 1, row.row) for row in ds.whole_rows(root, rows)]


class _CanonicalLogReads(TaskReads):
    """Use the existing retry projection without reading unadmitted child metadata."""

    def load(self, root: Any, task_id: str) -> Optional[Dict[str, Any]]:
        if pathlib.Path(root) != self.root:
            return None
        return super().load(root, task_id)


def _scope(root: pathlib.Path, task_id: str) -> tuple[dict, set]:
    """Re-discover admitted placements per request; client root/path hints confer no authority.

    Full, compact and Raw share this retry projection over canonical lineage and queue facts.
    Custody then admits their recorded execution placements, including a retry
    occupying its predecessor's drive, before any child metadata or log read.
    """
    configured, spellings = root, root_spellings(root)
    root = spellings[0]
    placements, ids = {"canonical": root}, {task_id} if task_id else set()
    if task_id:
        validate_task_id(task_id)
        reads = _CanonicalLogReads(root)
        snapshot = reads.queue_snapshot(root)
        queued = [dict(item.get("task") or {}, task_id=item.get("id") or item.get("task_id") or (item.get("task") or {}).get("id"))
                  for group in ("pending", "running") for item in snapshot.get(group) or []
                  if isinstance(item, dict) and isinstance(item.get("task"), dict)]
        def belongs(row):
            return row.get("delegation_role") == "subagent" and any(
                str(row.get(key) or "") == task_id for key in ("parent_task_id", "root_task_id"))
        queued = [row for row in queued if belongs(row)]
        children = select_lineage(root, task_id, task_id, queue_members=[row["task_id"] for row in queued])
        selected = [reads.effective(task_id)]
        for candidate in children:
            row = reads.effective(str(candidate.get("task_id") or ""))
            if belongs(row):
                selected.append(row)
                ids.add(str(row.get("task_id") or ""))
        for candidate in queued:
            row = reads.effective(str(candidate.get("task_id") or "")) or candidate
            selected.append(row)
            ids.add(str(row.get("task_id") or ""))
        ids.discard("")
        for row in selected:
            identity = str(row.get("task_id") or "")
            recorded = (reads.load(root, identity) or {}) if identity else {}
            for path in child_drive_candidates(recorded):
                child = physical_path(spellings, path)
                if retained_child_root(configured, child, identity):
                    placements[child.relative_to(root).as_posix()] = child
    return placements, ids


def _matches(row: dict, task_id: str, ids: set) -> bool:
    return not task_id or any(str(row.get(key) or "") in ids for key in ("task_id", "subagent_task_id")) \
        or any(str(row.get(key) or "") == task_id for key in ("parent_task_id", "root_task_id"))


def _public(row: dict) -> dict:
    return public_task_result(row, include_outcome_axes=any(
        key in row for key in ("status", "outcome_axes", "result_status", "loop_outcome")))


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _source_ref(source: dict) -> str:
    return base64.urlsafe_b64encode(_json_bytes(source)).decode("ascii").rstrip("=")


def _compact_row(row: dict, source: dict) -> dict:
    public = _public(row)
    encoded = _json_bytes(public)
    # Small rows are complete. Large fields are explicitly omitted whole: no
    # fabricated truncation marker can be mistaken for the original diagnostic.
    result, omitted, used = {}, [], 0
    priority = ("ts", "type", "task_id", "parent_task_id", "root_task_id", "subagent_task_id", "status", "phase")
    for key in dict.fromkeys([*priority, *public]):
        if key not in public:
            continue
        size = len(_json_bytes({key: public[key]}))
        if key == "_raw" or (len(encoded) > _COMPACT_BYTES and (size > 4096 or used + size > _COMPACT_BYTES)):
            omitted.append(key)
        else:
            result[key], used = public[key], used + size
    declared = _receipt_declarations(row)
    reconstructed = declared is not None
    if declared:  # what the stored event left behind its receipt is Raw's alone
        omitted.extend(key for key in declared[1] if key not in public and key not in omitted)
    unchanged = public == row
    result["_raw"] = {
        "ref": _source_ref(source), "source": source,
        "omitted_fields": [name[:128] for name in omitted[:64]], "omitted_field_count": len(omitted),
        "public_bytes": None if reconstructed else source["length"] if unchanged else len(encoded) + 1,
        "representation": "reconstructed" if reconstructed else "stored" if unchanged else "public_task_result",
    }
    return result


def _receipt_declarations(row: dict) -> Optional[Tuple[dict, dict]]:
    """A stored context view event's own ``receipt_source`` and ``at_source`` (loop_round_limits), when they
    name structures held by its task's retained ``candidate_materialization`` receipt; else None.

    Raw reads these from the re-read stored event, never from a reference: a reference only names the row."""
    receipt, at_source = row.get("receipt_source"), row.get("at_source")
    if row.get("checkpoint_kind") != "context_view" or not isinstance(receipt, dict) \
            or not isinstance(at_source, dict) or not at_source \
            or row.get("receipt_stage", "candidate_materialization") != "candidate_materialization":
        return None
    if not all(isinstance(key, str) and type(count) is int and count >= 0 for key, count in at_source.items()):
        return None
    task_id, digest = receipt.get("task_id"), receipt.get("sha256")
    if receipt.get("kind") != "task_source" or receipt.get("root") != "artifact_store" \
            or not str(receipt.get("path") or "").startswith("source_handles/context_checkpoints/") \
            or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) \
            or not isinstance(task_id, str) or not task_id or task_id != row.get("task_id"):
        return None
    return receipt, at_source


def _compact_tail(request: Request, name: str) -> JSONResponse:
    root = request_drive_root(request)
    task_id = str(request.query_params.get("task_id") or "").strip()
    limit = max(1, min(coerce_int(request.query_params.get("limit"), 100), 2000))
    rows, gaps, skipped_rows = [], set(), 0
    try:
        placements, ids = _scope(root, task_id)
        root = placements["canonical"]
        def matches(row):
            return _matches(row, task_id, ids)
        for placement, drive in placements.items():
            located, source_gaps, source_skipped = read_rotated_jsonl_sources(
                drive / "logs" / _ALLOWED_LOGS[name], limit, matches, anchor=root)
            gaps.update(source_gaps)
            skipped_rows += source_skipped
            for line, (start, entry, length, digest) in enumerate(located, 1):
                if not matches(entry):
                    continue
                source = {"v": 1, "kind": "log_row", "placement": placement, "task_id": task_id,
                          "log": name, "start": start, "length": length, "sha256": digest}
                rows.append((entry, source, str(drive), line))
    except StoreMigrating:
        return json_error("the dialogue is still importing; retry shortly", 503, reason_code="migrating")
    except (OSError, StoreError):
        return json_error("log source is unavailable", 503, reason_code="log_source_unavailable")
    except (ValueError, RecursionError):
        return json_error("log source cannot be projected", 400, reason_code="invalid_log_source")
    rows.sort(key=lambda row: (str(row[0].get("ts") or ""), str(row[0].get("_source_root") or row[2]),
                               int(row[0].get("_line") or row[3])))
    seen: set = set()
    rows = [row for row in rows if (key := replica_key(row[0])) is None or not (key in seen or seen.add(key))]
    entries = []
    try:
        for entry, source, drive, line in rows[-limit:]:
            item = _compact_row(entry, source)
            item.setdefault("_source_root", drive)
            item.setdefault("_line", line)
            entries.append(item)
    except (ValueError, RecursionError):
        return json_error("log source cannot be projected", 400, reason_code="invalid_log_source")
    return JSONResponse({"name": name, "entries": entries, "view": "compact",
                         "gaps": sorted(gaps), "skipped_rows": skipped_rows})


def _decode_ref(value: str, name: str) -> dict:
    if not value or len(value) > 4096 or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("invalid log source reference")
    source = json.loads(base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)))
    # A reference names one stored row and nothing else: a context view's declarations are that row's own.
    if not isinstance(source, dict) or set(source) != set(LogRowSource.__annotations__) \
            or type(source["v"]) is not int or source["v"] != 1 or source["log"] != name \
            or source["kind"] != "log_row":
        raise ValueError("invalid log source fields")
    if any(type(source[key]) is not int or source[key] < (1 if key == "length" else 0)
           for key in ("start", "length")):
        raise ValueError("invalid log source range")
    if not all(isinstance(source[key], str) for key in ("placement", "task_id", "sha256")) \
            or not re.fullmatch(r"[0-9a-f]{64}", source["sha256"]):
        raise ValueError("invalid log source identity")
    return source


class _RawRefusal(Exception):
    """One typed Raw answer: the status, message and reason a client sees."""

    def __init__(self, status: int, message: str, reason_code: str):
        super().__init__(message)
        self.status, self.message, self.reason_code = status, message, reason_code


def _stored_row(root: pathlib.Path, source: dict, placements: dict, ids: set) -> Tuple[bytes, dict]:
    """The one complete stored row a reference names, verified by its digest and scope, as written."""
    drive = placements.get(source["placement"])
    if drive is None:
        raise _RawRefusal(404, "log placement is not admitted in this task scope", "log_source_unavailable")
    snapshot = JsonlChainSnapshot(drive / "logs" / _ALLOWED_LOGS[source["log"]], anchor=root)
    start, end = source["start"], source["start"] + source["length"]
    if end > snapshot.upper:  # first: the segment lookup below reads only inside the captured chain
        raise JsonlChainUnreadable("source range is missing or truncated")
    base, boundary = snapshot.segment(bisect_right(snapshot.ends, start))
    if end > boundary:
        raise ValueError("a stored row cannot cross physical log segments")
    raw = snapshot._read(start, end)
    if hashlib.sha256(raw).hexdigest() != source["sha256"]:
        raise _RawRefusal(409, "recorded log bytes changed", "log_source_changed")
    if start > base and snapshot._read(start - 1, start) != b"\n":
        raise ValueError("not a complete stored row")
    raw.decode("utf-8")  # JSONL is UTF-8: its pages are cut and served as UTF-8 text
    entry = parse_jsonl_source(raw)
    if not _matches(entry, source["task_id"], ids):
        raise _RawRefusal(404, "row does not belong to the requested task scope", "log_source_unavailable")
    return raw, entry


def _reconstructed(root: pathlib.Path, entry: dict, receipt_source: dict, at_source: dict) -> Tuple[bytes, list, str]:
    """The stored final event with the absent structures its own ``at_source`` declares, from its task's retained
    ``candidate_materialization`` receipt, then the public projection, as for any Raw row; with the declared fields
    it lacks and why. A receipt that is missing or does not match adds nothing: the stored event is served whole."""
    absent = [key for key in at_source if key not in entry]  # a final event field is never replaced
    reason = ""
    if absent:
        try:
            saved = json.loads(read_actor_source_bytes(root, receipt_source["task_id"], receipt_source, anchor=root))
            receipt = saved.get("receipt") if isinstance(saved, dict) else None
            if not isinstance(receipt, dict) or saved.get("stage") != "candidate_materialization" \
                    or saved.get("published") is not False:
                raise ValueError("not this event's candidate receipt")
            fragments = {key: receipt.get(key) for key in absent}
            if any(not isinstance(value, (dict, list)) or len(value) != at_source[key] for key, value in fragments.items()):
                raise ValueError("a receipt fragment does not match its declaration")
            return _json_bytes(_public({**entry, **fragments})) + b"\n", [], ""
        except (OSError, StoreError):
            reason = "receipt_unavailable"
        except (ValueError, RecursionError):  # size, digest, path, JSON, stage, shape, or a value JSON cannot carry
            reason = "receipt_mismatch"
    return _json_bytes(_public(entry)) + b"\n", absent, reason


async def api_logs_raw(request: Request) -> Response:
    return await asyncio.to_thread(_logs_raw, request)


def _logs_raw(request: Request) -> Response:
    name = str(request.path_params.get("name") or "")
    if name not in _ALLOWED_LOGS or name in _DIALOGUE:
        return json_error("Raw references address file logs only", 404, reason_code="unknown_log")
    try:
        source = _decode_ref(request.query_params.get("ref", ""), name)
        offset = int(request.query_params.get("offset", "0"))
        limit = int(request.query_params.get("limit", str(_RAW_PAGE_BYTES)))
        download = request.query_params.get("download", "0")
        if offset < 0 or not 4 <= limit <= _RAW_PAGE_MAX or download not in {"0", "1"}:
            raise ValueError("invalid Raw page bounds")
        if download == "1" and ("offset" in request.query_params or "limit" in request.query_params):
            raise ValueError("download reads the whole representation")
        root = request_drive_root(request)
        placements, ids = _scope(root, source["task_id"])
        root = placements["canonical"]
    except (ValueError, binascii.Error, RecursionError):
        return json_error("invalid Raw reference or page bounds", 400, reason_code="invalid_raw_request")
    except (OSError, StoreError):
        return json_error("log scope is unavailable", 503, reason_code="log_source_unavailable")
    try:
        raw, entry = _stored_row(root, source, placements, ids)
        unavailable, reason = [], ""
        declared = _receipt_declarations(entry)
        if declared:
            representation = "reconstructed"
            payload, unavailable, reason = _reconstructed(root, entry, *declared)
        else:
            public = _public(entry)
            representation = "stored" if public == entry else "public_task_result"
            payload = raw if representation == "stored" else _json_bytes(public) + b"\n"
    except _RawRefusal as refusal:
        return json_error(refusal.message, refusal.status, reason_code=refusal.reason_code)
    except (OSError, StoreError):
        return json_error("recorded log source is missing, truncated or unreadable", 404, reason_code="log_source_unavailable")
    except (ValueError, RecursionError):
        return json_error("recorded log source is not a readable JSON row", 409, reason_code="log_source_changed")
    digest = hashlib.sha256(payload).hexdigest()
    headers = {"cache-control": "no-store", "x-content-type-options": "nosniff",
               "x-ouroboros-source-sha256": source["sha256"], "x-ouroboros-payload-sha256": digest,
               "x-ouroboros-raw-representation": representation}
    if reason:
        headers["x-ouroboros-raw-unavailable"] = f"{reason}; {len(unavailable)} declared fields absent"
    if download == "1":
        # A reconstruction is its own body: it never shares the stored row's file name.
        named = digest if representation == "reconstructed" else source["sha256"]
        headers["content-disposition"] = f'attachment; filename="{name}-{named}.jsonl"'
        return Response(payload, media_type="application/octet-stream", headers=headers)
    if offset > len(payload):
        return json_error("Raw offset is past the representation", 416, reason_code="raw_range_invalid")
    if offset < len(payload) and payload[offset] & 0xC0 == 0x80:
        return json_error("Raw offset splits a UTF-8 character", 416, reason_code="raw_range_invalid")
    end = min(len(payload), offset + limit)
    while end < len(payload) and payload[end] & 0xC0 == 0x80:
        end -= 1
    return JSONResponse({"source": source, "representation": representation, "sha256": digest,
                         "total_bytes": len(payload), "offset": offset, "next_offset": end,
                         "has_more": end < len(payload), "text": payload[offset:end].decode("utf-8"),
                         "unavailable_fields": unavailable, "unavailable_reason": reason}, headers=headers)


__all__ = ["api_logs_tail", "api_logs_raw", "LogRowSource", "LogRawReference", "LogRawResponse",
           "CompactLogEntry", "CompactLogTailResponse"]
