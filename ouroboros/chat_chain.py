"""The physical chat-log generation chain, index-free row addresses and source retention.

``logs/chat.jsonl`` rotates verbatim into ``archive/chat_<ts>.jsonl``; the
ordered archives plus the live file are the whole conversation. These readers
are shared by ``chat_history``, the generation-aware chat reader in ``memory``,
the owner-message source lens in ``project_dialogue``, reflection, the
consciousness wake and the one-time legacy import in ``chronicle_import``. Callers
use ``chat_chain.retain_memory_source`` as a module attribute, so one
substitution reaches every caller.

A chat row's address is ``{chat_id, ts, row_sha256}`` (text form
``row:<chat_id>@<ts>#<sha12>``) plus an optional ``hint`` that only speeds the
lookup. It names no path, inode or byte offset, so it survives rotation and a
copy of the data directory, and ``row_sha256`` turns a substituted row into a
typed ``row_mismatch`` instead of a silent answer. Resolution needs no index:
the hinted generation line first, then the archive rotated at or after the
row's ``ts`` and onward to the live file. Rotation stamps the archive name
from the same UTC clock that stamped the row, so that archive is the earliest
one that can hold it. Stream positions count rows exactly as the legacy
cursor does (JSON objects outside A2A, in chain order).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import itertools
import json
import pathlib
import re
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

from ouroboros.contracts.chat_id_policy import is_a2a_chat_id
from ouroboros.deadline_utils import parse_deadline_ts
from ouroboros.utils import jsonl_generation_signature as _chat_log_signature

_TEXT_ADDRESS = re.compile(r"row:(-?\d+)@(.*)#([0-9a-f]{12,64})")
_DIGEST = re.compile(r"[0-9a-f]{12,64}")
_ROTATED_AT = re.compile(r"chat_(\d{8}T\d{6})")
_HINT_FIELDS = ("task_id", "client_message_id", "type")
_LINEAGE_FIELDS = ("task_id", "parent_task_id", "root_task_id")


def retain_memory_source(context: Any, source_id: str, data: bytes, extension: str = "md") -> Dict[str, Any]:
    """Use existing immutable source storage with a reader valid after this task."""
    from ouroboros.artifacts import store_actor_source_bytes, task_artifact_dir_path
    root, task_id = pathlib.Path(context.drive_root).resolve(), str(context.task_id or "consolidation")
    ref = store_actor_source_bytes(root, task_id, category="context_checkpoints",
                                  source_id=source_id, data=data, extension=extension)
    path = task_artifact_dir_path(root, task_id, create=False) / ref["path"]
    return {**ref, "task_id": task_id, "canonical_root": str(root), "read": {"tool": "read_file",
            "arguments": {"root": "runtime_data", "path": path.relative_to(root).as_posix(), "start_line": 1}}}


def _ordered_chat_generation_paths(source_path: pathlib.Path) -> List[pathlib.Path]:
    """Return the physical chat chain, oldest archive to live."""
    archive_dir = source_path.parent.parent / "archive"
    try:
        archives = sorted(archive_dir.glob("chat_*.jsonl"), key=lambda p: p.name)
    except OSError:
        archives = []
    return [*archives, source_path]


def _resolve_generation_segments(
    meta: Dict[str, Any], source_path: pathlib.Path,
) -> Tuple[List[pathlib.Path], int, bool]:
    """Generation-aware consolidation cursor (v6.73.0).

    The cursor (``last_consolidated_offset`` + ``chat_log_signature``) points into
    ONE log generation. Rotation moves that generation to ``archive/chat_<ts>.jsonl``
    verbatim, so the stored first-line hash locates it in the ordered archive chain
    and consolidation continues over ``archives[i:] + live`` — the pre-rotation
    tail (and any number of intervening rotations) is consolidated, never dropped.
    Returns ``(ordered segments, offset into their concatenation, gap_detected)``;
    ``gap_detected`` is True only when the stored generation no longer exists
    anywhere (manual deletion/corruption — archives are never auto-pruned).
    """
    last_offset = int(meta.get("last_consolidated_offset", 0) or 0)
    stored_sig = meta.get("chat_log_signature") or {}
    stored_first = str(stored_sig.get("first_line_sha256") or "") if isinstance(stored_sig, dict) else ""
    live_sig = _chat_log_signature(source_path)
    archives = _ordered_chat_generation_paths(source_path)[:-1]
    if not stored_first:
        # Uninitialized cursor. Any archives that already exist rotated BEFORE
        # the first consolidation ever ran — they are unconsolidated by
        # definition, so the whole ordered chain is the window (offset 0).
        # A nonzero offset WITHOUT a signature is an ambiguous pre-signature
        # legacy shape: keep the historical live-only behavior for it.
        if last_offset == 0 and archives:
            return [*archives, source_path], 0, False
        return [source_path], last_offset, False
    if stored_first == str(live_sig.get("first_line_sha256") or ""):
        return [source_path], last_offset, False
    for index, archive_path in enumerate(archives):
        sig = _chat_log_signature(archive_path)
        if str(sig.get("first_line_sha256") or "") == stored_first:
            return [*archives[index:], source_path], last_offset, False
    return [source_path], 0, True


def _read_chat_entries(path: pathlib.Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    # Full project awareness (v6.32.0): the one identity's consolidated dialogue
    # (dialogue_blocks.json) is its WHOLE conversation — main + project threads —
    # because Ouroboros is one awareness/biography across direct chat, project
    # rooms, and background consciousness (BIBLE P1). Only A2A virtual-transport
    # ids are excluded (machine-to-machine traffic, not the human dialogue). This
    # MUST match memory.read_jsonl_tail_after_offset so the shared consolidation
    # offset indexes the same stream.
    entries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if not is_a2a_chat_id(entry.get("chat_id", 1)):
                entries.append(entry)
    return entries


# --- index-free row addresses ------------------------------------------------------------------

class RowAddressError(LookupError):
    """A range bound that does not resolve; ``resolution`` carries the typed status."""

    def __init__(self, resolution: Dict[str, Any]):
        super().__init__(str(resolution.get("status") or "row_missing"))
        self.resolution = resolution


def source_row_id(row: Dict[str, Any]) -> str:
    """Identity of an original raw chat row, before any room/view projection adds fields."""
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _row_chat_id(row: Dict[str, Any]) -> int:
    try:
        return int(row.get("chat_id", 1))
    except (TypeError, ValueError):
        return 1  # The legacy missing-address convention of chat_history and room evidence.


def chat_chain_paths(root: Any) -> List[pathlib.Path]:
    return _ordered_chat_generation_paths(pathlib.Path(root) / "logs" / "chat.jsonl")


def generation_signatures(root: Any) -> List[Tuple[pathlib.Path, str]]:
    """Existing generations with their first-line sha256, the ``gen`` of an address hint."""
    return [(path, str(_chat_log_signature(path).get("first_line_sha256") or ""))
            for path in chat_chain_paths(root) if path.exists()]


def row_address(row: Dict[str, Any], *, gen: Optional[str] = None, line: Optional[int] = None) -> Dict[str, Any]:
    hint = {key: str(row[key]) for key in _HINT_FIELDS if row.get(key)}
    if gen:
        hint["gen"] = gen
    if line:
        hint["line"] = int(line)
    return {"kind": "chat_row", "chat_id": _row_chat_id(row), "ts": str(row.get("ts") or ""),
            "row_sha256": source_row_id(row), "hint": hint}


def format_address(address: Dict[str, Any]) -> str:
    return f"row:{int(address['chat_id'])}@{address['ts']}#{str(address['row_sha256'])[:12]}"


def parse_address(text: str) -> Dict[str, Any]:
    match = _TEXT_ADDRESS.fullmatch(str(text or "").strip())
    if match is None:
        raise ValueError("a chat row address reads row:<chat_id>@<ts>#<first 12 hex of row_sha256>")
    return {"kind": "chat_row", "chat_id": int(match[1]), "ts": match[2], "row_sha256": match[3], "hint": {}}


def _normalized(address: Any) -> Dict[str, Any]:
    if isinstance(address, str):
        return parse_address(address)
    try:
        wanted = {"kind": "chat_row", "chat_id": int(address["chat_id"]), "ts": str(address["ts"] or ""),
                  "row_sha256": str(address["row_sha256"]).lower(), "hint": dict(address.get("hint") or {})}
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError("a chat row address needs chat_id, ts and row_sha256") from exc
    if not _DIGEST.fullmatch(wanted["row_sha256"]):
        raise ValueError("row_sha256 is 12 to 64 lowercase hex characters")
    return wanted


def _decoded(raw: bytes) -> Optional[Dict[str, Any]]:
    if not raw.strip():
        return None
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _stream_rows(sigs: List[Tuple[pathlib.Path, str]]) -> Iterator[Tuple[int, pathlib.Path, str, int, Dict[str, Any], int]]:
    """``(generation index, path, gen, physical line, row, stream position)`` for every stream row.

    The stream is the legacy cursor's (``_read_chat_entries``): JSON objects outside
    A2A, in chain order; blank and broken lines are not rows and take no position.
    """
    pos = 0
    for index, (path, gen) in enumerate(sigs):
        with path.open("rb") as handle:
            for line, raw in enumerate(handle, 1):
                row = _decoded(raw)
                if row is None or is_a2a_chat_id(row.get("chat_id", 1)):
                    continue
                yield index, path, gen, line, row, pos
                pos += 1


def _relative(root: pathlib.Path, path: pathlib.Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _found(root: pathlib.Path, path: pathlib.Path, gen: str, line: int, row: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "ok", "path": _relative(root, path), "line": line, "gen": gen,
            "address": row_address(row, gen=gen, line=line)}


def _same_row(row: Optional[Dict[str, Any]], wanted: Dict[str, Any]) -> bool:
    return (row is not None and _row_chat_id(row) == wanted["chat_id"]
            and str(row.get("ts") or "") == wanted["ts"])


def _rotated_at(path: pathlib.Path) -> Optional[_dt.datetime]:
    match = _ROTATED_AT.match(path.name)
    if match is None:
        return None  # The live file, or a name that carries no rotation time: always searched.
    return _dt.datetime.strptime(match[1], "%Y%m%dT%H%M%S").replace(tzinfo=_dt.timezone.utc)


def _search(root: pathlib.Path, sigs: List[Tuple[pathlib.Path, str]],
            wanted: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Archive rotated at or after the row's second, then forward to the live file.

    When that finds no row with this chat id and ts, the earlier archives are read too,
    nearest first: a rotation stamps the archive name before it waits for the append
    lock, so a row appended during that wait can carry a later second than its archive.
    """
    moment = parse_deadline_ts(wanted["ts"])
    first = 0
    if moment is not None:
        second = moment.replace(microsecond=0)
        first = next((i for i, (path, _gen) in enumerate(sigs)
                      if (_rotated_at(path) or second) >= second), len(sigs))
    ts = wanted["ts"]
    # A JSON writer stores a plain ISO timestamp verbatim; only lines holding it are parsed.
    needle = ts.encode("ascii") if ts and ts.isascii() and ts.isprintable() and not {'"', "\\"} & set(ts) else None
    matches: List[Tuple[pathlib.Path, str, int, Dict[str, Any]]] = []
    others: List[Dict[str, Any]] = []
    unreadable: List[Dict[str, Any]] = []

    def scan(generations: Iterable[Tuple[pathlib.Path, str]]) -> None:
        for path, gen in generations:
            try:
                with path.open("rb") as handle:
                    for line, raw in enumerate(handle, 1):
                        if needle is not None and needle not in raw:
                            continue
                        row = _decoded(raw)
                        if row is None:
                            if raw.strip():
                                unreadable.append({"path": _relative(root, path), "line": line})
                            continue
                        if not _same_row(row, wanted):
                            continue
                        if source_row_id(row).startswith(wanted["row_sha256"]):
                            matches.append((path, gen, line, row))
                        else:
                            others.append(row_address(row, gen=gen, line=line))
            except OSError as exc:
                unreadable.append({"path": _relative(root, path), "error": type(exc).__name__})
            if matches:
                return  # The generation holding the first match closes the search.

    scan(sigs[first:])
    if not matches and not others:
        scan(reversed(sigs[:first]))
    distinct = {}
    for path, gen, line, row in matches:
        distinct.setdefault(source_row_id(row), row_address(row, gen=gen, line=line))
    if len(distinct) > 1:
        return None, {"status": "row_ambiguous", "candidates": list(distinct.values())}
    if matches:
        path, gen, line, row = matches[0]
        return row, _found(root, path, gen, line, row)
    if others:
        return None, {"status": "row_mismatch", "candidates": others}
    if unreadable:
        return None, {"status": "row_unreadable", "unreadable": unreadable}
    return None, {"status": "row_missing"}


def resolve_row(root: Any, address: Any) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """``(row, {"status": ok|row_missing|row_mismatch|row_ambiguous|row_unreadable, ...})``.

    The hint only speeds the lookup: a hinted line whose chat_id, ts or hash differ
    is ignored and the search decides. A malformed address raises ``ValueError``.
    """
    root, wanted = pathlib.Path(root), _normalized(address)
    sigs = generation_signatures(root)
    gen, line = wanted["hint"].get("gen"), wanted["hint"].get("line")
    if gen and isinstance(line, int) and line > 0:
        for path, signature in sigs:
            if signature != gen:
                continue
            try:
                with path.open("rb") as handle:
                    row = _decoded(next(itertools.islice(handle, line - 1, None), b""))
            except OSError:
                continue
            if _same_row(row, wanted) and source_row_id(row).startswith(wanted["row_sha256"]):
                return row, _found(root, path, signature, line, row)
    return _search(root, sigs, wanted)


def stream_position_of(root: Any, address: Any) -> Optional[int]:
    """The row's position in the legacy cursor's stream; None when it does not resolve or is not a stream row."""
    row, found = resolve_row(root, address)
    if row is None:
        return None
    for _index, _path, gen, line, _row, pos in _stream_rows(generation_signatures(root)):
        if gen == found["gen"] and line == found["line"]:
            return pos
    return None


def _bound(root: pathlib.Path, address: Any) -> Optional[Tuple[str, int]]:
    if address is None:
        return None
    row, found = resolve_row(root, address)
    if row is None:
        raise RowAddressError(found)
    return found["gen"], found["line"]


def _rows_between(root: pathlib.Path, start: Optional[Tuple[str, int]], stop: Optional[Tuple[str, int]],
                  keep: Optional[Callable[[Dict[str, Any]], bool]] = None,
                  ) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any], int]]:
    sigs = generation_signatures(root)
    order: Dict[str, int] = {}
    for index, (_path, gen) in enumerate(sigs):
        order.setdefault(gen, index)
    keys = []
    for bound in (start, stop):
        if bound is not None and bound[0] not in order:
            raise RowAddressError({"status": "row_missing", "gen": bound[0]})
        keys.append(None if bound is None else (order[bound[0]], bound[1]))
    first, last = keys

    def rows() -> Iterator[Tuple[Dict[str, Any], Dict[str, Any], int]]:
        for index, _path, gen, line, row, pos in _stream_rows(sigs):
            if last is not None and (index, line) > last:
                return
            if (first is None or (index, line) >= first) and (keep is None or keep(row)):
                yield row_address(row, gen=gen, line=line), row, pos
    return rows()


def iter_rows(root: Any, *, from_addr: Any = None) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any], int]]:
    """``(address, row, pos)`` for every room in append order, from ``from_addr`` when given.

    One pass over the chain; an unresolved ``from_addr`` raises ``RowAddressError`` here.
    """
    root = pathlib.Path(root)
    return _rows_between(root, _bound(root, from_addr), None)


def _period_bounds(period: Any) -> Tuple[Optional[_dt.datetime], Optional[_dt.datetime]]:
    if period is None:
        return None, None
    if not isinstance(period, dict):
        raise ValueError("period is {'from': <ISO time>, 'to': <ISO time>}")
    bounds = []
    for key in ("from", "to"):
        value = period.get(key)
        parsed = parse_deadline_ts(value) if value else None
        if value and parsed is None:
            raise ValueError(f"period.{key} must be an ISO-8601 timestamp")
        bounds.append(parsed)
    return bounds[0], bounds[1]


def iter_room_rows(root: Any, room_id: Any, *, from_addr: Any = None, to_addr: Any = None,
                   task_ids: Optional[Iterable[str]] = None, period: Any = None,
                   ) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any], int]]:
    """``(address, row, pos)`` of one room: the stream filtered by canonical room membership.

    Membership is ``project_dialogue.room_membership`` with the same registry facts
    as ``dialogue_evidence.read_room_source``. ``from_addr``/``to_addr`` bound the
    stream inclusively; ``task_ids`` keeps rows whose task, parent or root task is
    listed; ``period`` keeps rows whose ``ts`` lies within its inclusive bounds (a
    row without a readable ``ts`` lies in no period). Every physical row keeps its
    own address: a redelivered message is not folded into its first copy.
    """
    from ouroboros.project_dialogue import room_membership, source_refs_for_project
    from ouroboros.projects_registry import all_task_bindings, list_reserved_projects

    root, chat = pathlib.Path(root), int(room_id)
    projects = {int(project["chat_id"]) for project in list_reserved_projects(root)}
    refs = source_refs_for_project(root, chat) if chat in projects else []
    member = room_membership(chat, projects, refs, all_task_bindings(root))
    tasks = None if task_ids is None else {str(task) for task in task_ids}
    lower, upper = _period_bounds(period)

    def keep(row: Dict[str, Any]) -> bool:
        if not member(_row_chat_id(row), row):
            return False
        if tasks is not None and not tasks & {str(row.get(field) or "") for field in _LINEAGE_FIELDS}:
            return False
        if lower is None and upper is None:
            return True
        moment = parse_deadline_ts(row.get("ts"))
        return moment is not None and (lower is None or moment >= lower) and (upper is None or moment <= upper)

    return _rows_between(root, _bound(root, from_addr), _bound(root, to_addr), keep)
