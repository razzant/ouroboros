"""Addressed dialogue reads over ``state/dialog.sqlite``.

A row keeps its content identity ``{chat_id, ts, row_sha256}``; imported rows also
keep every legacy ``(gen, line)`` hint. SQL coordinates are optional hints for new
rows. Stream positions count JSON objects outside A2A exactly as the old memory
cursor did. Room reads probe room/task/origin indexes before applying the shared
membership rule; the frozen JSONL archive belongs only to the importer.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import heapq
import json
import pathlib
import re
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Tuple

from ouroboros import dialogue_store as ds, dialogue_positions as dp
from ouroboros.contracts.chat_id_policy import is_a2a_chat_id
from ouroboros.deadline_utils import parse_deadline_ts

_TEXT_ADDRESS = re.compile(r"row:(-?\d+)@(.*)#([0-9a-f]{12,64})")
_DIGEST = re.compile(r"[0-9a-f]{12,64}")
_HINT_FIELDS = ("task_id", "client_message_id", "type")
_ADDRESS_PAGE = 500  # rows whose ordinals and legacy hints one read looks up (``dp.ordinals_of`` pages at 500)
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


class RowAddressError(LookupError):
    """A range bound that does not resolve; ``resolution`` carries the typed status."""

    def __init__(self, resolution: Dict[str, Any]):
        super().__init__(str(resolution.get("status") or "row_missing"))
        self.resolution = resolution


def source_row_id(row: Dict[str, Any]) -> str:
    """Identity of an original raw chat row, before any room/view projection adds fields."""
    from ouroboros.place_arrivals import SourceProjection
    if isinstance(row, SourceProjection):
        row = row.source_body
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _row_chat_id(row: Dict[str, Any]) -> int:
    try:
        return int(row.get("chat_id", 1))
    except (TypeError, ValueError):
        return 1  # The legacy missing-address convention of chat_history and room evidence.


def row_address(row: Dict[str, Any], *, gen: Optional[str] = None, line: Optional[int] = None,
                sha: Optional[str] = None) -> Dict[str, Any]:
    hint = {key: str(row[key]) for key in _HINT_FIELDS if row.get(key)}
    if gen:
        hint["gen"] = gen
    if line:
        hint["line"] = int(line)
    return {"kind": "chat_row", "chat_id": _row_chat_id(row), "ts": str(row.get("ts") or ""),
            "row_sha256": source_row_id(row) if sha is None else sha, "hint": hint}


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


def stored_address(root: Any, entry: ds.Row) -> Dict[str, Any]:
    """Content identity, with an imported physical hint or a new SQL coordinate."""
    return address_at(entry.row, entry.seq, ds.legacy_address(root, entry.seq) if entry.stream == "chat" else None)


def address_at(row: Dict[str, Any], seq: int, legacy: Any, *, sha: Optional[str] = None) -> Dict[str, Any]:
    """The stored address of the row at ``seq``: its legacy ``(gen, line)`` hint, else the coordinate. ``row`` may
    be the row's ``memory_inventory.row_meta`` (it keeps every field an address reads) with its known ``sha``."""
    address = row_address(row, gen=legacy[0], line=legacy[1], sha=sha) if legacy else row_address(row, sha=sha)
    if legacy is None:
        address["hint"]["seq"] = seq
    return address


def _resolved(root: Any, entry: ds.Row) -> Dict[str, Any]:
    address = stored_address(root, entry)
    hint = address["hint"]
    generation = ds.legacy_generation(root, hint["gen"]) if hint.get("gen") else None
    return {"status": "ok", "path": f"archive/{generation['name']}" if generation else "state/dialog.sqlite",
            "line": hint.get("line"), "gen": hint.get("gen"), "seq": entry.seq, "address": address}


def _same_row(row: Dict[str, Any], wanted: Dict[str, Any]) -> bool:
    return _row_chat_id(row) == wanted["chat_id"] and str(row.get("ts") or "") == wanted["ts"]


def _hint_answers(row: Dict[str, Any], wanted: Dict[str, Any]) -> bool:
    """A row a hint names answers the address when it is the same row and its content identity is the digest."""
    return _same_row(row, wanted) and source_row_id(row).startswith(wanted["row_sha256"])


def resolve_row(root: Any, address: Any, *, stream: str = "chat") -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Resolve a content identity, checking hints and then indexed chat/time candidates.

    Missing coverage raises the store's typed importing/failure outcome. A recorded
    invalid interval is unreadable history, never proof that the row is absent.
    """
    wanted = _normalized(address)
    hint = wanted["hint"]
    hinted = []
    if type(hint.get("seq")) is int:
        entry = ds.row_at(root, stream, hint["seq"])
        hinted = [entry] if entry is not None else []
    elif stream == "chat" and hint.get("gen") and type(hint.get("line")) is int:
        hinted = ds.legacy_candidates(root, hint["gen"], hint["line"])
    for entry in hinted:
        if _hint_answers(entry.row, wanted):
            return entry.row, _resolved(root, entry)
    candidates = ds.rows_by_identity(root, wanted["chat_id"], wanted["ts"], stream=stream)
    if wanted["chat_id"] == 1:
        candidates += ds.rows_by_identity(root, None, wanted["ts"], stream=stream)
    candidates.sort(key=lambda entry: entry.seq)
    matches = [entry for entry in candidates if source_row_id(entry.row).startswith(wanted["row_sha256"])]
    distinct = {source_row_id(entry.row): stored_address(root, entry) for entry in matches}
    if len(distinct) > 1:
        return None, {"status": "row_ambiguous", "candidates": list(distinct.values())}
    if matches:
        return matches[0].row, _resolved(root, matches[0])
    ds.require_imported(root, stream)
    if candidates:
        return None, {"status": "row_mismatch", "candidates": [stored_address(root, entry) for entry in candidates]}
    gaps = dp.skipped_between(root, stream)
    if gaps:
        return None, {"status": "row_unreadable", "unreadable": [
            {"path": "state/dialog.sqlite", "offset": gap.seq, "length": gap.length, "kind": gap.kind} for gap in gaps]}
    return None, {"status": "row_missing"}


def rows_by_address(root: Any, addresses: Iterable[Any], *, stream: str = "chat") -> Dict[str, Dict[str, Any]]:
    """``resolve_row``'s row for each address, keyed by its ``row_sha256`` (as given, or as normalized from
    the text form); the hinted ones are read together (by SQL coordinate, or by legacy ``(gen, line)``)
    and accepted on the same identity check, an address its hint does not answer is resolved alone, as
    ``resolve_row`` goes on to do, and the first address of a key that resolves answers for it."""
    wanted: Dict[str, list] = {}
    for address in addresses:
        norm = _normalized(address)
        key = str(address["row_sha256"]) if isinstance(address, dict) and "row_sha256" in address else norm["row_sha256"]
        wanted.setdefault(key, []).append(norm)
    norms = [norm for listed in wanted.values() for norm in listed]
    coordinates = [norm["hint"]["seq"] for norm in norms if type(norm["hint"].get("seq")) is int]
    pairs = [_legacy_hint(norm) for norm in norms if type(norm["hint"].get("seq")) is not int and stream == "chat"
             and _legacy_hint(norm)]
    at = ds.rows_at(root, stream, coordinates) if coordinates else {}
    legacy = ds.legacy_candidates_of(root, pairs) if pairs else {}
    found: Dict[str, Dict[str, Any]] = {}
    for key, listed in wanted.items():
        for norm in listed:
            seq = norm["hint"].get("seq")
            hinted = (([at[seq]] if seq in at else []) if type(seq) is int
                      else legacy.get(_legacy_hint(norm), []) if stream == "chat" else [])
            row = next((entry.row for entry in hinted if _hint_answers(entry.row, norm)), None)
            if row is None:
                row, _found = resolve_row(root, norm, stream=stream)
            if row is not None:
                found[key] = row
                break
    return found


def _legacy_hint(norm: Dict[str, Any]) -> Optional[Tuple[str, int]]:
    hint = norm["hint"]
    return (str(hint["gen"]), hint["line"]) if hint.get("gen") and type(hint.get("line")) is int else None


def stream_position_of(root: Any, address: Any) -> Optional[int]:
    row, found = resolve_row(root, address)
    if row is None:
        return None
    ordinal = dp.ordinal_at(root, "chat", found["seq"])
    return None if ordinal is None else ordinal.pos


def _bound(root: Any, address: Any) -> Optional[int]:
    if address is None:
        return None
    row, found = resolve_row(root, address)
    if row is None:
        raise RowAddressError(found)
    return found["seq"]


def position_boundary(root: Any, position: int) -> int:
    """A persisted stream ordinal as a byte boundary; the end when at/past EOF."""
    if position <= 0:
        return 0
    if not ds.stream_end(root, "chat"):
        ds.require_imported(root, "chat")
        return 0
    row = dp.row_at_position(root, position)
    return row.seq if row is not None else ds.stream_end(root, "chat")


def generation_cursor(root: Any, meta: Dict[str, Any]) -> Tuple[int, bool]:
    """Legacy consolidation cursor -> absolute stream ordinal, and missing-generation fact.

    The offset counts non-A2A objects, not bytes. A pre-signature nonzero
    cursor meant the old live file, whose identity the import ledger does not
    retain separately. That ambiguous cursor is a gap, never a guessed position.
    No pending segment can be mistaken for a missing one.
    """
    ds.require_imported(root, "chat")
    signature = meta.get("chat_log_signature") or {}
    gen = str(signature.get("first_line_sha256") or "") if isinstance(signature, dict) else ""
    offset = int(meta.get("last_consolidated_offset", 0) or 0)
    if gen == "dialogue:chat":
        return offset, False
    if gen:
        unit = ds.legacy_generation(root, gen)
        return (int(unit["pos_base"] or 0) + offset, False) if unit else (0, True)
    if not offset:
        return 0, False
    return 0, True


def _addressed(root: Any, entries: Iterable[ds.Row], keep: Optional[Callable[[ds.Row], bool]] = None,
               ) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any], int]]:
    """``(address, row, pos)`` of each row ``_paged`` keeps."""
    for entry, legacy, pos in _paged(root, entries, keep):
        yield address_at(entry.row, entry.seq, legacy), entry.row, pos


def _paged(root: Any, entries: Iterable[ds.Row], keep: Optional[Callable[[ds.Row], bool]] = None,
           ) -> Iterator[Tuple[ds.Row, Optional[Tuple[str, int]], int]]:
    """Each row ``keep`` takes (every one without it), outside A2A, with its legacy hint and stream
    position, looked up a page at a time: the ordinals and the legacy hints of a page's rows in one read
    each, never one per row (every room's open rows come through here, and Main's view reads them each
    turn). A page is ``_ADDRESS_PAGE`` source rows examined, kept or not, so a filtered reader that stops
    early reads at most a page beyond what it used. A source that fails mid-page (a busy store between
    its own pages) still answers the rows it read first, then fails, as the one-row lookups did."""
    page, examined, source = [], 0, iter(entries)
    while True:
        try:
            entry = next(source)
        except StopIteration:
            break
        except Exception:
            yield from _page(root, page)
            raise
        examined += 1
        if not is_a2a_chat_id(entry.row.get("chat_id", 1)) and (keep is None or keep(entry)):
            page.append(entry)
        if examined == _ADDRESS_PAGE:
            yield from _page(root, page)
            page, examined = [], 0
    yield from _page(root, page)


def _page(root: Any, page: list) -> Iterator[Tuple[ds.Row, Optional[Tuple[str, int]], int]]:
    if not page:
        return
    legacy = ds.legacy_addresses(root, [entry.seq for entry in page if entry.stream == "chat"])
    try:
        ordinals = dp.ordinals_of(root, "chat", [entry.seq for entry in page])
    except ds.StoreBusy:
        raise
    except ds.StoreError:
        ordinals = None  # a unit's counts are unsettled (``dp._uncovered``): the rows before it still answer, one by one
    for entry in page:
        ordinal = ordinals.get(entry.seq) if ordinals is not None else dp.ordinal_at(root, "chat", entry.seq)
        yield entry, legacy.get(entry.seq), ordinal.pos


def iter_rows(root: Any, *, from_addr: Any = None, lower: int = 0,
              upper: Optional[int] = None) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any], int]]:
    """The whole requested stream, in append order, with the persisted memory positions."""
    start = _bound(root, from_addr)
    lower = lower if start is None else start
    upper = ds.stream_end(root, "chat") if upper is None else upper
    ds.require_imported(root, "chat", lower=lower, upper=upper)
    return _addressed(root, ds.iter_rows(root, "chat", lower=lower, upper=upper))


def iter_stream(root: Any, *, lower: int, upper: int) -> Iterator[Tuple[ds.Row, Optional[Tuple[str, int]], int]]:
    """``iter_rows`` keeping each row's stored coordinate: ``(entry, legacy hint, pos)`` for ``address_at``."""
    ds.require_imported(root, "chat", lower=lower, upper=upper)
    return _paged(root, ds.iter_rows(root, "chat", lower=lower, upper=upper))


_REACH = ("WITH RECURSIVE reach(id) AS (SELECT value FROM json_each(?) UNION "
          "SELECT task_id FROM tasks JOIN reach ON parent_task_id=reach.id UNION "
          "SELECT task_id FROM tasks JOIN reach ON root_task_id=reach.id) SELECT json_group_array(id) FROM reach")


def _reached(conn: Any, tasks: Iterable[str]) -> set:
    """The tasks and every task under them (by parent or root), read in ONE step: each step releases the GIL."""
    return {str(task) for task in json.loads(conn.execute(_REACH, (json.dumps(sorted(tasks)),)).fetchone()[0] or "[]")}


def room_rows(root: Any, room_id: Any, *, stream: str = "chat", lower: int = 0,
              upper: Optional[int] = None, gaps: Optional[set] = None) -> Iterator[ds.Row]:
    """Indexed candidate union for the canonical room rule, including late task bindings. ``gaps`` collects
    what a Project's chat rows read so may lack (``room_appearances.read_gaps``) once the read starts."""
    from ouroboros.project_dialogue import room_membership, source_refs_for_project
    from ouroboros.projects_registry import list_reserved_projects, RoomResolver
    from ouroboros.store import open_store, store_path

    chat = int(room_id)
    projects = {int(project["chat_id"]) for project in list_reserved_projects(root)}
    refs = source_refs_for_project(root, chat) if chat in projects else []
    resolver = RoomResolver(root)
    member = room_membership(chat, projects, refs, resolver)
    keys = [("chat_id", chat)]
    identities = {}
    upper = ds.stream_end(root, stream) if upper is None else upper
    if chat == 1 and chat not in projects:
        keys = [("chat_id", room["chat_id"]) for room in ds.rooms(root)
                if room["chat_id"] not in projects and room["chat_id"] >= 0]
        keys.append(("chat_id", None))
    if chat in projects:
        tasks = {str(tid) for tid, binding in resolver.bindings.items()
                 if int(binding.get("project_chat_id") or 0) == chat}
        if tasks and store_path(root).exists():
            with open_store(root, write=False) as txn:
                tasks.update(_reached(txn.conn, tasks))
        keys.extend((field, tid) for tid in sorted(tasks) for field in _LINEAGE_FIELDS)
        if stream == "chat":  # rows written elsewhere this room also shows (``room_membership.appearance_keys``)
            shown = [f"room:{chat}", *(f"task:{tid}" for tid in sorted(tasks))]
            keys.extend(("appearance", key) for key in shown)
            if gaps is not None:
                from ouroboros.room_appearances import read_gaps

                gaps.update(read_gaps(root, chat, shown, lower, upper))
        for ref in refs:
            if ref.get("client_message_id"):
                keys.append(("client_message_id", ref["client_message_id"]))
            elif ref.get("ts"):
                source_chat = int(ref.get("chat_id") or 1)
                for cid in ([source_chat, None] if source_chat == 1 else [source_chat]):
                    for entry in ds.rows_by_identity(root, cid, str(ref["ts"]), stream=stream):
                        if lower <= entry.seq < upper:
                            identities[entry.seq] = entry
            else:
                # Hash-only legacy references still name a chat; membership checks their text.
                keys.append(("chat_id", int(ref.get("chat_id") or 1)))
                if int(ref.get("chat_id") or 1) == 1:
                    keys.append(("chat_id", None))
    candidates = ds.iter_probes(root, stream, keys, lower=lower, upper=upper)
    previous = None
    for entry in heapq.merge(candidates, sorted(identities.values(), key=lambda row: row.seq), key=lambda row: row.seq):
        if entry.seq == previous:
            continue
        previous = entry.seq
        if member(_row_chat_id(entry.row), entry.row):
            yield entry


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
                   lower: int = 0, upper: Optional[int] = None, gaps: Optional[set] = None,
                   ) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any], int]]:
    """One room's rows with inclusive content-address bounds and original stream positions (``gaps``: ``room_rows``)."""
    first, last = _bound(root, from_addr), _bound(root, to_addr)
    lower = lower if first is None else first
    upper = (ds.stream_end(root, "chat") if upper is None else upper) if last is None else last + 1
    ds.require_imported(root, "chat", lower=lower, upper=upper)
    tasks = None if task_ids is None else {str(task) for task in task_ids}
    after, before = _period_bounds(period)

    def selected(entry: ds.Row) -> bool:
        row = entry.row
        if tasks is not None and not tasks.intersection(str(row.get(field) or "") for field in _LINEAGE_FIELDS):
            return False
        if after is not None or before is not None:
            moment = parse_deadline_ts(row.get("ts"))
            if moment is None or (after is not None and moment < after) or (before is not None and moment > before):
                return False
        return True
    filtered = tasks is not None or after is not None or before is not None
    return _addressed(root, room_rows(root, room_id, lower=lower, upper=upper, gaps=gaps), selected if filtered else None)
