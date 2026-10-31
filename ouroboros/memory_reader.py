"""The supported read of this installation's memory for extensions (``PluginAPI.read_memory``, 2.1).

Three operations over the host's own data root, each answered by the owner of its memory family
and never by a second interpreter: the chronicle by ``ChronicleStore`` (its index, ``_interpret``,
``_source_state``, and its ``told`` projection decided by ``memory_view_account.told_map``), a page's,
part's, account's or retold record's period by ``memory_inventory.record_period`` (``_dated``), knowledge
by ``knowledge`` (address, inventory, note, links), room labels by ``memory_view.room_labeler``, and
the plain sources (identity, WORLD, scratchpad, a Project's workpad) and their append-only journals
by their files. ``memory_read`` and this reader share one composition of a record at every revision.

- ``catalog`` lists one ``area`` a page at a time: ``next`` is an opaque cursor, ``null`` once the
  area is ``complete``. The chronicle filters and limits in its index, a shelf opens only that page's
  notes, and a journal page reads at most ``SCAN_BYTES`` of complete lines (a single longer line is
  read alone, whole, as its own page). A filter the index cannot apply (``status``, a ``cited_by``
  match, a shelf ``view``) is applied to the examined page, so a page may hold fewer entries than
  ``limit`` while ``next`` still continues it. ``story`` is the memory view's own story capture,
  whole per call and sliced (``_story``). A list that can change between pages (the story, a note's
  links, a source's nested changes) binds its cursor to the fingerprint or revision it pages
  (``_bound_start``): a changed one is refused, never mixed. ``room_id`` ``""`` is the host's room.
- ``document`` reads one addressed text by code-point range: ``next`` names the following start and
  the revision the range belongs to, so a text of any size is read whole and from one revision. A
  record reads as of any of its revisions; a note, file, field or journal entry is refused
  ``revision_mismatch`` once its text changed. A ``field`` of a stored record, journal entry or note
  front matter reads that value: a string as text, anything else as its JSON.
- ``changes`` returns the chronicle records after a position; the position it returns is the last
  record returned, never a newer head. Knowledge and plain files have no change feed: compare
  their ``revision`` on a later read.

Every stored value in an answer is a bounded projection (``_project``): a long string, a long list, a
wide object or whatever exceeds the answer's budget becomes a ``ref`` carrying its size and the
``address`` (with ``field``) that reads it whole; a long string the host composed (a title, a label, a
stamp summary) is a preview whose ``derived_from`` names its source. A derived list that grows (a record's later
corrections, a source's nested changes) is ``{count, next}``, its continuation being catalog
parameters. Interpreting a record still reads its own corrections, a part's or account's period its
members or sources, and a source's nested changes walk that source's graph: those reads are
record-sized, never the chronicle's.

Reading changes no memory source and never activates, imports or creates one: without a journal the
chronicle is ``absent``. The chronicle's own derived index may still catch up under its lock (its
ordinary read path).
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sqlite3
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from ouroboros.chronicle_store import AUTHOR_KINDS, KINDS, ChronicleStore, draft_signer, is_late_review_mark
from ouroboros.contracts.plugin_api import MemoryReadError

CONTRACT = "ouroboros.memory/1"
DEFAULT_LIMIT, MAX_LIMIT = 50, 500
DEFAULT_LENGTH, MAX_LENGTH = 65_536, 1_048_576
PREVIEW_CHARS = 280
LIST_HEAD, KEYS_MAX, DEPTH_MAX = 8, 32, 5
ENTRY_BUDGET, VALUE_BUDGET = 3_000, 12_000  # projected characters per list entry / per document value
SUMMARY_INLINE = 2_000
SCAN_BYTES = 8 * 1024 * 1024
STATUSES = ("final", "draft", "candidate", "accepted", "rejected")
_EVENTS = ("correction", "decision", "selection", "mark_view", "mark_release")
# Plain sources: identity, the generated WORLD profile and the scratchpad's generated projection of its blocks.
# Relative to the reader's own root and contained (``_inside``): the owners' accessors (``Memory``, ``project_facts``)
# return resolved or ``config.DATA_DIR`` paths; the reader's tests write these sources through those owners.
_FILES = {"identity": ("memory/identity.md", "Identity", "identity", "markdown"),
          "scratchpad": ("memory/scratchpad.md", "Scratchpad", "scratchpad", "generated from memory/scratchpad_blocks.json"),
          "world": ("memory/WORLD.md", "World profile", "world", "generated profile")}
_JOURNALS = {"identity": "memory/identity_journal.jsonl", "world": "memory/world_journal.jsonl",
             "scratchpad": "memory/scratchpad_journal.jsonl", "knowledge:global": "memory/knowledge_history.jsonl"}
_ZERO = {"sequence": 0, "record_id": None}


def read_memory(data_root: Any, operation: str, parameters: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """One ``catalog``, ``document`` or ``changes`` answer over ``data_root``, or a ``MemoryReadError``."""
    if parameters is not None and not isinstance(parameters, Mapping):
        raise MemoryReadError("invalid_parameter", "parameters is an object")
    handler = _OPERATIONS.get(operation) if isinstance(operation, str) else None
    if handler is None:
        raise MemoryReadError("invalid_parameter", "operation is catalog, document or changes")
    root = pathlib.Path(data_root)
    try:
        return {"contract": CONTRACT, "operation": operation, **handler(root, dict(parameters or {}))}
    except MemoryReadError:
        raise
    except (TimeoutError, sqlite3.Error, OSError, ValueError) as exc:  # a busy lock or an unreadable store, not an answer
        raise MemoryReadError("unavailable", f"{type(exc).__name__}: {exc}") from exc


# --- parameters ------------------------------------------------------------------------------------

def _only(params: Mapping[str, Any], *allowed: str) -> None:
    unknown = sorted(set(params) - set(allowed))
    if unknown:
        raise MemoryReadError("invalid_parameter", f"unknown parameter(s) {unknown}; accepted: {sorted(allowed)}")


def _int(params: Mapping[str, Any], key: str, default: int, low: int, high: Optional[int] = None) -> int:
    value = params.get(key, default)
    if type(value) is not int or value < low or (high is not None and value > high):
        raise MemoryReadError("invalid_parameter", f"{key} is an integer from {low}" + (f" to {high}" if high else ""))
    return value


def _text(params: Mapping[str, Any], key: str, *, required: bool = False, empty: bool = False) -> Optional[str]:
    """A string parameter; ``empty``: ``""`` is a value (the host's room), never the omitted ``None``."""
    value = params.get(key)
    if value is None and not required:
        return None
    if not isinstance(value, str) or not (value or empty) or "\x00" in value:
        raise MemoryReadError("invalid_parameter", f"{key} is a {'' if empty else 'non-empty '}string")
    return value


def _choice(params: Mapping[str, Any], key: str, choices: Tuple[str, ...], default: str) -> str:
    value = params.get(key, default)
    if value not in choices:
        raise MemoryReadError("invalid_parameter", f"{key} is one of {list(choices)}")
    return value


def _subset(params: Mapping[str, Any], key: str, allowed: Any) -> Tuple[str, ...]:
    value = params.get(key) or ()
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) and item in allowed for item in value):
        raise MemoryReadError("invalid_parameter", f"{key} is a list drawn from {sorted(allowed)}")
    return tuple(value)


def _cursor(params: Mapping[str, Any], mark: str) -> Optional[str]:
    """The value of a cursor this area issued (``<mark><value>``), or None for the first page."""
    value = params.get("cursor")
    if value is None:
        return None
    if not isinstance(value, str) or not value.startswith(mark):
        raise MemoryReadError("invalid_parameter", "cursor is the `next` this area returned")
    return value[len(mark):]


def _number_cursor(params: Mapping[str, Any], mark: str) -> int:
    raw = _cursor(params, mark)
    if raw is None:
        return 0
    if not raw.isdigit():
        raise MemoryReadError("invalid_parameter", "cursor is the `next` this area returned")
    return int(raw)


def _page(entries: List[Dict[str, Any]], examined: List[Any], limit: int, key: Callable[[Any], str]) -> Dict[str, Any]:
    """``examined`` holds one row past the page when more follow: the cursor continues after the page's last."""
    more = len(examined) > limit
    return {"entries": entries, "next": key(examined[limit - 1]) if more else None, "complete": not more}


def _window_page(entries: List[Dict[str, Any]], start: int, total: int, limit: int, bound: str = "") -> Dict[str, Any]:
    """A page of a list from index ``start``: the cursor is the next index, then ``:<bound>`` for a list that can
    change between pages (``_bound_start``)."""
    end = min(start + limit, total)
    return {"entries": entries, "next": (f"i{end}:{bound}" if bound else f"i{end}") if end < total else None,
            "complete": end >= total}


def _bound_start(params: Mapping[str, Any], bound: str, code: str, message: str, **facts: Any) -> int:
    """The start index of a page of a list that can change between pages (a note's links, a source's nested changes,
    the story): its cursor names the revision or fingerprint it continues, and one issued over another is refused
    ``code``, never mixed."""
    raw = _cursor(params, "i")
    if raw is None:
        return 0
    start, _colon, issued = raw.partition(":")
    if not start.isdigit():
        raise MemoryReadError("invalid_parameter", "cursor is the `next` this area returned")
    if issued != bound:
        raise MemoryReadError(code, message, **facts)
    return int(start)


def _fingerprint(value: Any) -> str:
    """What binds a cursor over a derived list: the digest of the whole list as read, not a summary of it."""
    return hashlib.sha256(_field_text(value)[0].encode()).hexdigest()[:32]


# --- addresses -------------------------------------------------------------------------------------

def _project_id(value: str) -> str:
    from ouroboros.project_facts import explicit_project_id_ok

    if not explicit_project_id_ok(value):
        raise MemoryReadError("invalid_parameter", "a Project is named by its exact project id")
    return value


def _inside(root: pathlib.Path, relative: str) -> pathlib.Path:
    path = root / relative
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        raise MemoryReadError("not_found", f"{relative} leaves the data root") from None
    return path


def _file_path(root: pathlib.Path, source: str) -> Tuple[pathlib.Path, str, Optional[str], str]:
    """``(path, title, journal id, representation)`` of one plain source."""
    if source in _FILES:
        relative, title, journal, form = _FILES[source]
        return _inside(root, relative), title, journal, form
    if source.startswith("workpad:"):
        project = _project_id(source[len("workpad:"):])
        return _inside(root, f"projects/{project}/workpad.md"), f"Project {project} workpad", None, "markdown"
    raise MemoryReadError("invalid_parameter", "source is identity, world, scratchpad or workpad:<project id>")


def _journal_path(root: pathlib.Path, journal: str) -> pathlib.Path:
    if journal in _JOURNALS:
        return _inside(root, _JOURNALS[journal])
    if journal.startswith("knowledge:project:"):
        return _inside(root, f"projects/{_project_id(journal[len('knowledge:project:'):])}/knowledge_history.jsonl")
    if journal.startswith("project:"):
        return _inside(root, f"projects/{_project_id(journal[len('project:'):])}/journal.jsonl")
    raise MemoryReadError("invalid_parameter", "journal is identity, world, scratchpad, knowledge:<scope> or project:<project id>")


def _note_address(root: pathlib.Path, scope: Any, topic: Any = "overview"):
    from ouroboros.knowledge import resolve_knowledge_address

    if not isinstance(scope, str) or not isinstance(topic, str):
        raise MemoryReadError("invalid_parameter", "a note is addressed by scope and topic strings")
    try:
        return resolve_knowledge_address(root, topic, scope)
    except ValueError as exc:
        raise MemoryReadError("invalid_parameter", str(exc)) from None


def _address(params: Mapping[str, Any]) -> Dict[str, Any]:
    address = params.get("address")
    if not isinstance(address, Mapping) or address.get("kind") not in ("record", "note", "file", "journal", "entry"):
        raise MemoryReadError("invalid_parameter", "address is {kind: record|note|file|journal|entry, ...}")
    return dict(address)


def _record_address(ident: Any) -> Dict[str, Any]:
    return {"kind": "record", "id": ident}


def _relative(root: pathlib.Path, path: Any) -> Optional[str]:
    try:
        return pathlib.Path(path).resolve().relative_to(root.resolve()).as_posix()
    except (TypeError, ValueError):
        return None


# --- shapes ----------------------------------------------------------------------------------------

def _words(record: Mapping[str, Any], *, own: bool = False) -> str:
    """A record's words (``own``: without the corrections an interpretation adds), else a decision's reason."""
    if "text" in record:
        return str(record.get("text") if own else record.get("current_text", record.get("text")) or "")
    return str(record.get("reason") or "")


def _preview(text: str) -> Dict[str, Any]:
    return {"preview": text[:PREVIEW_CHARS], "chars": len(text)}


def _project(value: Any, address: Mapping[str, Any], field: List[Any], budget: List[int], depth: int = 0) -> Any:
    """A stored value bounded for one answer, sharing ``budget`` (projected characters left) with its siblings.

    A string past ``PREVIEW_CHARS``, a list past ``LIST_HEAD`` items, an object past ``KEYS_MAX`` keys and
    anything past ``DEPTH_MAX`` or the budget becomes ``{"ref": ...}`` with its size and the ``address``
    plus ``field`` that reads it whole (``document``); nothing is dropped without a ref.
    """
    at = {**address, "field": field}
    budget[0] -= 8
    if budget[0] <= 0 or depth > DEPTH_MAX:
        return {"ref": "value", "address": at}
    if isinstance(value, str):
        budget[0] -= min(len(value), PREVIEW_CHARS)
        return value if len(value) <= PREVIEW_CHARS else {"ref": "text", **_preview(value), "address": at}
    if isinstance(value, list):
        head = [_project(item, address, [*field, index], budget, depth + 1) for index, item in enumerate(value[:LIST_HEAD])]
        return head if len(value) <= LIST_HEAD else {"ref": "list", "count": len(value), "head": head, "address": at}
    if isinstance(value, Mapping):
        if len(value) > KEYS_MAX:
            return {"ref": "object", "keys": len(value), "address": at}
        budget[0] -= sum(len(str(key)) for key in value)
        return {str(key): _project(item, address, [*field, key], budget, depth + 1) for key, item in value.items()}
    return value if value is None or isinstance(value, (bool, int, float)) else str(value)  # a YAML date, say


def _derived_text(text: str, source: Mapping[str, Any]) -> Any:
    """A string composed by its owner (a title, a label, a stamp summary) whole when short, else its preview, with
    ``derived_from`` naming the source it was composed from: no ``ref``, since no field holds this exact text."""
    return text if len(text) <= PREVIEW_CHARS else {**_preview(text), "derived_from": dict(source)}


def _field_value(value: Any, field: Any) -> Any:
    """The value at ``field`` (keys and list indices) inside a stored value, or ``not_found``."""
    if not isinstance(field, list):
        raise MemoryReadError("invalid_parameter", "field is a list of keys and list indices")
    try:
        for key in field:
            value = value[key] if isinstance(value, list) and type(key) is int or isinstance(value, Mapping) else {}[key]
    except (KeyError, IndexError, TypeError):
        raise MemoryReadError("not_found", "no such field") from None
    return value


def _field_text(value: Any) -> Tuple[str, str]:
    """``(text, representation)`` of one stored value: a string as itself, anything else as its JSON."""
    if isinstance(value, str):
        return value, "text"
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str), "json"


def _dated(store: ChronicleStore, record: Mapping[str, Any]) -> Dict[str, Any]:
    """A page's, part's, account's or retold record's period keyed as its story entry is: ``source_period`` and
    ``period_basis`` from ``memory_inventory.record_period``, ``period`` as the view prints it; ``{}`` for any other
    kind. No legacy unit is computed, so a retold record is dated by its block's recorded period (``block``), never by
    reading the legacy history; a part reads its members and an account its sources, recursively, one ``get`` each."""
    if record.get("kind") not in ("page", "part", "account", "legacy", "gap"):
        return {}
    from ouroboros.memory_inventory import record_period
    from ouroboros.memory_view import _legacy_period, _period

    period = record_period(store, record, {})
    metadata = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
    text = (_legacy_period({"covers": record.get("covers"), "range_text": metadata.get("legacy_range_text")})
            if record.get("kind") in ("legacy", "gap") else _period(period.span, period.note()))
    return {"period": _derived_text(text, _record_address(record["id"])), "source_period": period.span,
            "period_basis": period.source}


def _record_entry(record: Mapping[str, Any], budget: Optional[List[int]] = None,
                  store: Optional[ChronicleStore] = None) -> Dict[str, Any]:
    """One chronicle record as a list shows it: its own words previewed, its acting state, its stored facts
    projected (``_project``) and a count where history grows; never a list of its history. With ``store``, a
    record the story dates carries its period (``_dated``)."""
    budget, address = [ENTRY_BUDGET] if budget is None else budget, _record_address(record["id"])
    author = record.get("author") if isinstance(record.get("author"), Mapping) else {}
    entry = {"address": address, "id": record["id"], "kind": record.get("kind"), "room_id": record.get("room_id"),
             "sequence": record.get("sequence"), "ts": record.get("ts"), "author": _project(author, address, ["author"], budget),
             "signer": draft_signer(author) if author.get("kind") in ("mind", "helper") else None,
             "status": record.get("status"), "folded_into": record.get("folded_into"), "revision": record.get("revision"),
             "corrections": len(record.get("corrections") or ()), "target_id": record.get("target_id") or None,
             **_preview(_words(record, own=True))}
    if store is not None:
        entry.update(_dated(store, record))
    for key in ("covers", "task_id", "accepted", "shown", "replaces", "visibility", "scope", "quote", "target_ref", "anchor"):
        if key in record:
            entry[key] = _project(record[key], address, [key], budget)
    if record.get("kind") == "account":
        selection = record.get("selection")
        entry["sources"] = len(record.get("sources") or ())
        entry["selection"] = selection and {
            **{key: selection.get(key) for key in ("id", "shown", "ts", "sequence")},
            **{key: _project(selection.get(key), _record_address(selection["id"]), [key], budget) for key in ("reason", "replaces")}}
    if record.get("kind") == "mark":
        view, released = record.get("view"), record.get("released")
        entry.update(late_review=is_late_review_mark(record), released=_record_entry(released, budget) if released else None,
                     view=_record_entry(view, budget) if view else None)
        if view:
            entry["visibility"] = view.get("visibility")
    return entry


def _note_entry(root: pathlib.Path, row: Mapping[str, Any]) -> Dict[str, Any]:
    """A note as a shelf lists it: the authored summary whole up to ``SUMMARY_INLINE``, else a ref to it."""
    address, summary = {"kind": "note", "scope": row["scope"], "topic": row["topic"]}, row.get("summary")
    budget = [ENTRY_BUDGET]
    if isinstance(summary, str) and len(summary) > SUMMARY_INLINE:
        summary = {"ref": "text", **_preview(summary), "address": {**address, "field": ["summary"]}}
    return {"address": address, "scope": row["scope"], "topic": row["topic"],
            "title": _derived_text(row.get("title") or row["topic"], address),
            "path": _relative(root, row.get("path")), "summary": summary or None, "state": row.get("state", "active"),
            "archive": _project(row.get("archive"), address, ["archive"], budget),
            "archive_error": row.get("archive_error") or "", "parse_error": row.get("parse_error") or "",
            "read_error": row.get("read_error") or ""}


# --- catalog ---------------------------------------------------------------------------------------

def _store(root: pathlib.Path) -> Optional[ChronicleStore]:
    """The chronicle only when its journal exists: a reader never creates one."""
    store = ChronicleStore(root)
    return store if store.log_path.exists() else None


def _empty() -> Dict[str, Any]:
    return {"entries": [], "next": None, "complete": True}


def _limit(params: Mapping[str, Any]) -> int:
    return _int(params, "limit", DEFAULT_LIMIT, 1, MAX_LIMIT)


def _rooms(root: pathlib.Path, params: Dict[str, Any]) -> Dict[str, Any]:
    _only(params, "area", "cursor", "limit")
    after, limit, store = _cursor(params, "r"), _limit(params), _store(root)
    if store is None:
        return {"chronicle": {"state": "absent", "activation": None, "position": dict(_ZERO)}, **_empty()}
    _none, position = store.observation_snapshot()
    activation = store.activation()
    from ouroboros.memory_view import room_labeler

    label, rows = room_labeler(root), store.rooms(after_room=after, limit=limit + 1)  # no cursor: room "" first
    entries = [{"address": {"kind": "room", "room_id": row["room_id"]}, **row,
                "label": label(row["room_id"]) if row["room_id"] else "Host records without a room"} for row in rows[:limit]]
    chronicle = {"state": "present", "position": position,
                 "activation": {"id": activation.get("id"), "ts": activation.get("ts")} if activation else None}
    return {"chronicle": chronicle, **_page(entries, rows, limit, lambda row: "r" + row["room_id"])}


def _records(root: pathlib.Path, params: Dict[str, Any]) -> Dict[str, Any]:
    _only(params, "area", "cursor", "limit", "room_id", "kinds", "author_kind", "status", "unfolded")
    room, kinds = _text(params, "room_id", empty=True), _subset(params, "kinds", KINDS)
    statuses = _subset(params, "status", STATUSES)
    author = params.get("author_kind")
    if author is not None and (not isinstance(author, str) or author not in AUTHOR_KINDS) or type(
            params.get("unfolded", False)) is not bool:
        raise MemoryReadError("invalid_parameter", f"author_kind is one of {sorted(AUTHOR_KINDS)}; unfolded is a boolean")
    after, limit, store = _number_cursor(params, "s"), _limit(params), _store(root)
    if store is None:
        return _empty()
    rows = store.interpreted(room, kinds=kinds, after_seq=after, limit=limit + 1, author_kind=author,
                             unfolded=params.get("unfolded", False), sources=False)
    entries = [_record_entry(record, store=store) for record in rows[:limit]
               if not statuses or record.get("status") in statuses]
    return _page(entries, rows, limit, lambda row: f"s{row['sequence']}")


def _told_by(store: ChronicleStore, ids: List[str]) -> Dict[str, Optional[str]]:
    """``record id -> account`` that tells it in the common view: ``told_map``, the story's own precedence, over
    only the acting selections that bear on each record (``ChronicleStore.telling_selections``, indexed)."""
    from ouroboros.memory_view_account import told_map

    return {ident: told_map(entries).get(ident) for ident, entries in store.telling_selections(ids).items()}


def _story_entry(entry: Mapping[str, Any]) -> Dict[str, Any]:
    """One story entry as the memory view composed it, its words and lists bounded: the record reads them whole."""
    out: Dict[str, Any] = {key: entry.get(key) for key in (
        "kind", "id", "room_id", "period", "source_period", "period_basis", "status", "signer", "told_by", "block",
        "rows", "chars", "messages", "written", "selection", "selection_sequence", "sequence") if key in entry}
    record = _record_address(entry["id"])
    out.update(address=record, told_by=entry.get("told_by"), label=_derived_text(str(entry.get("label") or ""), record),
               **_preview(str(entry.get("text") or "")))
    # The view's own compositions: a stamp summary of the stored host_stamp, a legacy unit's [start, end] minutes,
    # a gap's stored source_gap (or the view's short default). None is a stored field of that name.
    if entry.get("stamp"):
        out["stamp"] = _derived_text(str(entry["stamp"]), {**record, "field": ["host_stamp"]})
    if entry.get("span"):
        out["span"] = [str(item) for item in entry["span"]][:2]
    if entry.get("gap"):
        out["gap"] = _derived_text(str(entry["gap"]), {**record, "field": ["metadata", "source_gap"]})
    for key in ("fixes", "terminal_links", "quotes", "sources"):  # derived lists of the view: their size here
        if key in entry:
            out[f"{key}_count"] = len(entry.get(key) or ())
    if entry.get("kind") == "account" and entry.get("selection"):
        out["replaces"] = {"count": len(entry.get("replaces") or ()),
                           "address": {**_record_address(entry["selection"]), "field": ["replaces"]}}
    return out


def _story(root: pathlib.Path, params: Dict[str, Any]) -> Dict[str, Any]:
    """The common story as the memory view tells it, paged: one ``memory_view.capture_story`` per call.

    The owners decide it whole: legacy coverage by ``memory_inventory.legacy_units``/``legacy_pointers``, acting
    pages and parts and the shown accounts by ``memory_view._story_pages``/``memory_view_account.account_entries``,
    replacement by ``told_map``. One read costs one story capture of the view (every legacy unit, acting page,
    part and account); pages slice it. A cursor carries the story's fingerprint, so a story that changed
    between pages is refused rather than mixed. Before the one legacy import the view shows no story, nor this.
    """
    _only(params, "area", "cursor", "limit")
    limit, store = _limit(params), _store(root)
    if store is None or store.activation() is None:
        return {"state": "absent" if store is None else "not_activated", **_empty()}
    from ouroboros.chronicle_store import capture
    from ouroboros.memory_view import capture_story, room_labeler
    from ouroboros.store import StoreError

    try:
        with capture(root, tail="terminal"):  # the one state of the chronicle the view's story reads
            story, status, refusals = capture_story(store, root, room_labeler(root))
    except StoreError as exc:  # dialogue history still importing or unimportable: coverage unknown, never imported here
        raise MemoryReadError("unavailable", f"{type(exc).__name__}: {exc}") from None
    # The whole captured story binds a cursor (words, selections, stamps, periods, status), not a summary of it.
    fingerprint = _fingerprint([story, status, refusals])
    start = _bound_start(params, fingerprint, "position_mismatch",
                         "the story changed since that page: read it again from the start")
    entries = [_story_entry(entry) for entry in story[start:start + limit]]
    page = _window_page(entries, start, len(story), limit, fingerprint)
    return {"state": "active", "fingerprint": fingerprint, "count": len(story), "order": "story",
            "status": {key: value if value is None or isinstance(value, (bool, int, float)) else str(value)[:PREVIEW_CHARS]
                       for key, value in status.items()},  # the view's fixed story counts and helper route
            "refusals_count": len(refusals), **page}


def _marks(root: pathlib.Path, params: Dict[str, Any]) -> Dict[str, Any]:
    _only(params, "area", "cursor", "limit", "room_id", "state")
    room = _text(params, "room_id", empty=True)
    state = _choice(params, "state", ("active", "released", "all"), "active")
    after, limit, store = _number_cursor(params, "s"), _limit(params), _store(root)
    if store is None:
        return _empty()
    rows = store.interpreted(room, kinds=("mark",), after_seq=after, limit=limit + 1, sources=False)
    entries = [_record_entry(mark) for mark in rows[:limit]
               if state == "all" or (state == "active") == (mark.get("released") is None)]
    return _page(entries, rows, limit, lambda row: f"s{row['sequence']}")


def _project_dirs(root: pathlib.Path) -> Tuple[List[str], int]:
    """``(valid project ids with a store directory, how many directories are not a project id)``."""
    from ouroboros.project_facts import explicit_project_id_ok

    base = root / "projects"
    names = sorted(path.name for path in base.iterdir() if path.is_dir()) if base.is_dir() else []
    valid = [name for name in names if explicit_project_id_ok(name)]
    return valid, len(names) - len(valid)


def _project_names(root: pathlib.Path) -> Dict[str, str]:
    from ouroboros.projects_registry import list_reserved_projects

    try:
        return {str(row.get("id")): str(row.get("name") or "") for row in list_reserved_projects(root)}
    except (OSError, ValueError):
        return {}


def _shelves(root: pathlib.Path, params: Dict[str, Any]) -> Dict[str, Any]:
    _only(params, "area", "cursor", "limit")
    after, limit = _cursor(params, "p") or "", _limit(params)
    projects, skipped = _project_dirs(root)
    names = _project_names(root) if projects else {}
    rows = [{"scope": "global", "project_id": None, "name": None, "present": (root / "memory" / "knowledge").is_dir()}]
    rows += [{"scope": f"project:{pid}", "project_id": pid, "name": names.get(pid) or None,
              "present": (root / "projects" / pid / "knowledge").is_dir()} for pid in projects]
    rows = [{**row, "address": {"kind": "shelf", "scope": row["scope"]}} for row in rows if row["scope"] > after]
    return {"skipped_directories": skipped, **_page(rows[:limit], rows[:limit + 1], limit, lambda row: "p" + row["scope"])}


def _shelf(root: pathlib.Path, params: Dict[str, Any]) -> Dict[str, Any]:
    from ouroboros.knowledge import inventory_knowledge

    _only(params, "area", "cursor", "limit", "scope", "view")
    view = _choice(params, "view", ("active", "archived", "all"), "active")
    address, after, limit = _note_address(root, _text(params, "scope", required=True)), _cursor(params, "t"), _limit(params)
    rows = list(inventory_knowledge(address, navigation=True, after=after or "", limit=limit + 1))
    entries = [_note_entry(root, row) for row in rows[:limit] if view == "all" or row.get("state", "active") == view]
    return {"scope": address.scope, "view": view, "present": address.shelf.is_dir(),
            **_page(entries, rows, limit, lambda row: "t" + row["topic"])}


def _documents(root: pathlib.Path, params: Dict[str, Any]) -> Dict[str, Any]:
    """The plain sources and Project journals, in key order: the core files, then each Project's."""
    _only(params, "area", "cursor", "limit")
    after, limit = _cursor(params, "k") or "", _limit(params)
    projects, skipped = _project_dirs(root)
    keyed = [(f"0:{source}", {"kind": "file", "source": source}) for source in sorted(_FILES)]
    keyed += [(f"1:{pid}:{part}", {"kind": "file", "source": f"workpad:{pid}"} if part == "workpad"
               else {"kind": "journal", "journal": f"project:{pid}"}) for pid in projects for part in ("journal", "workpad")]
    # Paged in the key order the cursor compares: ``atlas-2``'s keys sort before ``atlas``'s ("-" < ":").
    keyed = [(key, address) for key, address in sorted(keyed, key=lambda pair: pair[0]) if key > after][:limit + 1]
    entries = []
    for key, address in keyed[:limit]:
        if address["kind"] == "file":
            path, title, journal, form = _file_path(root, address["source"])
            entry = {"title": title, "representation": form,
                     "journal": {"kind": "journal", "journal": journal} if journal else None}
        else:
            path = _journal_path(root, address["journal"])
            entry = {"title": f"Project {address['journal'][len('project:'):]} journal", "representation": "jsonl"}
        entries.append({"key": key, "address": address, "path": _relative(root, path), "exists": path.is_file(), **entry})
    return {"skipped_directories": skipped, **_page(entries, keyed, limit, lambda pair: "k" + pair[0])}


def _journal_page(root: pathlib.Path, journal: str, params: Mapping[str, Any],
                  match: Optional[Callable[[Mapping[str, Any]], bool]] = None) -> Dict[str, Any]:
    """Rows of one append-only journal from a byte offset: at most ``limit`` kept, ``SCAN_BYTES`` of lines read.

    A line that would cross the window starts the next page; a page's first line is read whole however long
    (one record-sized read), so every line is reached and none is cut. The offset advances only over
    complete lines: an unterminated last line (an append in progress) stays for a later read. An
    unparseable line is an entry marked ``unreadable``, never skipped. Each row is projected (``_project``).
    """
    path, offset, limit = _journal_path(root, journal), _number_cursor(params, "o"), _limit(params)
    if not path.is_file():
        return {"journal": journal, "exists": False, "position": "o0", **_empty()}
    size, entries, tail = path.stat().st_size, [], 0
    if offset > size:
        raise MemoryReadError("position_mismatch", "the journal is shorter than this cursor")
    with path.open("rb") as stream:
        if offset:
            stream.seek(offset - 1)
            if stream.read(1) != b"\n":
                raise MemoryReadError("position_mismatch", "the cursor is not the start of a journal line")
        position = offset
        while len(entries) < limit and position - offset < SCAN_BYTES:
            window = SCAN_BYTES - (position - offset)
            raw = stream.readline(window + 1)
            if len(raw) > window and not raw.endswith(b"\n"):  # this line crosses the window
                if position > offset:
                    break
                raw += stream.readline()
            if not raw.endswith(b"\n"):
                tail = len(raw)
                break
            start, position = position, position + len(raw)
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except ValueError:
                row = None
            at = {"kind": "entry", "journal": journal, "offset": start}
            if not isinstance(row, dict):
                entries.append({"offset": start, "unreadable": True, "bytes": len(raw), "address": {**at, "field": []}})
            elif match is None or match(row):
                entries.append({"offset": start, "address": {**at, "field": []}, "fields": _project(row, at, [], [ENTRY_BUDGET])})
    complete = position + tail >= size
    return {"journal": journal, "exists": True, "position": f"o{position}", "unterminated_tail_bytes": tail,
            "entries": entries, "next": None if complete else f"o{position}", "complete": complete}


def _history(root: pathlib.Path, params: Dict[str, Any]) -> Dict[str, Any]:
    _only(params, "area", "cursor", "limit", "address", "kinds")
    address = _address(params)
    if address["kind"] == "record":
        kinds = _subset(params, "kinds", _EVENTS) or _EVENTS
        after, limit, store = _number_cursor(params, "s"), _limit(params), _store(root)
        if store is None or store.get(str(address.get("id"))) is None:
            raise MemoryReadError("not_found", "no such chronicle record")
        events = store.targeting(str(address["id"]), kinds, after_seq=after, limit=limit + 1)
        return _page([_record_entry(event) for event in events[:limit]], events, limit, lambda row: f"s{row['sequence']}")
    if params.get("kinds") is not None:
        raise MemoryReadError("invalid_parameter", "kinds narrows a record's history")
    if address["kind"] == "note":
        note = _note_address(root, address.get("scope"), address.get("topic"))
        journal = "knowledge:" + note.scope
        return _journal_page(root, journal, params, lambda row: row.get("topic") == note.topic)
    if address["kind"] == "file":
        journal = _file_path(root, str(address.get("source")))[2]
        return _journal_page(root, journal, params) if journal else {"journal": None, **_empty()}
    if address["kind"] == "journal":
        return _journal_page(root, str(address.get("journal")), params)
    raise MemoryReadError("invalid_parameter", "history is read for a record, note, file or journal")


def _later_corrections(store: ChronicleStore, record_id: str, revision: Optional[str], later: List[str]) -> Dict[str, Any]:
    """The corrections after ``revision`` as a count and the history page that lists them (after that revision)."""
    cited = store.get(revision) if later and revision else None
    return {"count": len(later), "next": {"area": "history", "address": _record_address(record_id), "kinds": ["correction"],
                                          "cursor": f"s{cited['sequence']}"} if cited else None}


def _source_entry(store: ChronicleStore, account: Mapping[str, Any], index: int, state: Mapping[str, Any]) -> Dict[str, Any]:
    """One frozen source of an account beside its state now; its growing lists are counts with continuations."""
    nested = state.get("nested_changes") or ()
    return {"index": index, "address": _record_address(state.get("id")),
            **{key: state.get(key) for key in ("id", "kind", "room_id", "revision", "sha256", "status", "missing",
                                               "revision_known", "current_revision", "status_now", "folded_into")},
            "later_corrections": _later_corrections(store, str(state.get("id")), state.get("revision"),
                                                    list(state.get("later_corrections") or ())),
            "nested_changes": {"count": len(nested), "next": {
                "area": "relations", "address": _record_address(account["id"]), "relation": "source_changes",
                "source": index} if nested else None}}


def _relations(root: pathlib.Path, params: Dict[str, Any]) -> Dict[str, Any]:
    _only(params, "area", "cursor", "limit", "address", "relation", "source")
    address, relation, limit = _address(params), params.get("relation"), _limit(params)
    if not isinstance(relation, str):
        raise MemoryReadError("invalid_parameter", "relation is a string")
    # A part's members and an account's sources are stored, never changed: an index continues them.
    start = _number_cursor(params, "i") if relation in ("members", "sources") else 0
    if address["kind"] == "note" and relation == "links":
        from ouroboros.knowledge import knowledge_links

        note = _read_note(_note_address(root, address.get("scope"), address.get("topic")))
        links = knowledge_links(note)  # the note's text decides the list: a cursor continues that revision only
        start = _bound_start(params, note.revision, "revision_mismatch",
                             "the note changed since that page: read its links again from the start",
                             current_revision=note.revision)
        rows = [{"index": start + offset, "label": _derived_text(str(link.get("label") or ""), address),
                 "target": _derived_text(str(link.get("target") or ""), address),
                 "status": link.get("status"), "start_byte": link.get("start_byte"), "end_byte": link.get("end_byte"),
                 "path": _relative(root, link.get("path")),
                 "address": ({"kind": "note", "scope": link["address"]["scope"], "topic": link["address"]["topic"]}
                             if link.get("address") else None)} for offset, link in enumerate(links[start:start + limit])]
        return {"revision": note.revision, "count": len(links),
                **_window_page(rows, start, len(links), limit, note.revision)}
    if address["kind"] != "record" or relation not in ("members", "sources", "source_changes", "cited_by"):
        raise MemoryReadError("invalid_parameter",
                              "relation is members, sources, source_changes or cited_by of a record, or links of a note")
    store = _store(root)
    record = store.get(str(address.get("id"))) if store is not None else None
    if record is None:
        raise MemoryReadError("not_found", "no such chronicle record")
    if relation == "cited_by":  # accounts examined in publication order; a match names the frozen version cited
        accounts = store.records(kinds=("account",), after_seq=_number_cursor(params, "s"), limit=limit + 1)
        entries = [{"account": _record_entry(account, store=store),
                    "cited": _project(source, _record_address(account["id"]), ["sources", position], [ENTRY_BUDGET])}
                   for account in accounts[:limit] for position, source in enumerate(account.get("sources") or ())
                   if isinstance(source, Mapping) and source.get("id") == record["id"]]
        return _page(entries, accounts, limit, lambda row: f"s{row['sequence']}")
    if relation == "members":
        if record["kind"] != "part":
            raise MemoryReadError("invalid_parameter", "members are a part's")
        ids = [str(member) for member in ((record.get("covers") or {}).get("member_ids") or ())]
        found = [item for item in (store.get(ident) for ident in ids[start:start + limit]) if item is not None]
        read = {item["id"]: item for item in store.interpret(found, sources=False)}
        rows = [{"index": start + offset, "id": ident, "missing": ident not in read,
                 "member": _record_entry(read[ident], store=store) if ident in read else None}
                for offset, ident in enumerate(ids[start:start + limit])]
        # A part records its members' ids, never the revision of each it folded: none is invented here.
        return {"count": len(ids), "member_versions": "not_recorded", **_window_page(rows, start, len(ids), limit)}
    if record["kind"] != "account":
        raise MemoryReadError("invalid_parameter", "sources are an account's")
    refs = list(record.get("sources") or ())
    if relation == "source_changes":  # one source's nested changes: its source graph is walked, the page returned
        source = params.get("source")
        if type(source) is not int or not 0 <= source < len(refs):
            raise MemoryReadError("invalid_parameter", f"source is an index into the account's {len(refs)} sources")
        changes = store.interpret([{**record, "sources": [refs[source]]}])[0]["sources"][0].get("nested_changes") or []
        fingerprint = _fingerprint(changes)  # members' later corrections land mid-list: a cursor continues this list
        start = _bound_start(params, fingerprint, "position_mismatch",
                             "the source's nested changes changed since that page: read them again from the start")
        rows = [{"index": start + offset, **{key: change.get(key) for key in ("source_id", "kind", "room_id", "via", "revision")},
                 "event": _record_entry(change["event"])} for offset, change in enumerate(changes[start:start + limit])]
        return {"count": len(changes), "fingerprint": fingerprint, "member_versions": "not_recorded",
                **_window_page(rows, start, len(changes), limit, fingerprint)}
    # Only this page's frozen references are interpreted: the account's other sources cost nothing here.
    states = store.interpret([{**record, "sources": refs[start:start + limit]}])[0]["sources"]
    rows = [_source_entry(store, record, start + offset, state) for offset, state in enumerate(states)]
    return {"count": len(refs), **_window_page(rows, start, len(refs), limit)}


_AREAS = {"rooms": _rooms, "records": _records, "story": _story, "marks": _marks, "shelves": _shelves,
          "shelf": _shelf, "documents": _documents, "history": _history, "relations": _relations}


def _catalog(root: pathlib.Path, params: Dict[str, Any]) -> Dict[str, Any]:
    area = params.get("area")
    if not isinstance(area, str) or area not in _AREAS:
        raise MemoryReadError("invalid_parameter", f"area is one of {sorted(_AREAS)}")
    return {"area": area, **_AREAS[area](root, params)}


# --- document --------------------------------------------------------------------------------------

def _read_note(address: Any):
    from ouroboros.knowledge import read_knowledge_note

    try:
        return read_knowledge_note(address)
    except FileNotFoundError:
        raise MemoryReadError("not_found", "no such note") from None
    except UnicodeDecodeError:
        raise MemoryReadError("unavailable", "the note is not UTF-8") from None


def _bound(revision: Optional[str], current: str) -> None:
    if revision is not None and revision != current:
        raise MemoryReadError("revision_mismatch", "the source changed since that revision", current_revision=current)


def _field_document(value: Any, field: Any, revision: Optional[str], context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """One stored value read whole by ranges: a string as text, anything else as JSON; revision = sha256 of that text."""
    text, form = _field_text(_field_value(value, field))
    current = hashlib.sha256(text.encode("utf-8")).hexdigest()
    _bound(revision, current)
    return text, {**context, "revision": current, "current_revision": current, "field": field, "representation": form}


def _record_document(root: pathlib.Path, address: Mapping[str, Any], revision: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    store = _store(root)
    record = store.get(str(address.get("id"))) if store is not None else None
    if record is None:
        raise MemoryReadError("not_found", "no such chronicle record")
    stored = {key: value for key, value in record.items() if key != "sequence"}
    if "field" in address:  # a stored value of an immutable record: a page's seal refs, a part's members, ...
        return _field_document(stored, address["field"], revision, {"record_id": record["id"]})
    try:
        read = store.interpret([record], revision=revision, sources=False)[0]
    except ValueError as exc:  # a revision that is not one of this record's
        raise MemoryReadError("not_found", str(exc)) from None
    later, budget = read.get("later_corrections") or [], [VALUE_BUDGET]
    entry = _record_entry(read, budget)
    entry.pop("preview")
    from ouroboros.memory_view import room_labeler

    meta = {"record": entry, "stored": _project(stored, _record_address(record["id"]), [], budget),
            "corrections_shown": entry["corrections"], "current_revision": later[-1] if later else read["revision"],
            "later_corrections": _later_corrections(store, record["id"], read["revision"], later),
            "room_label": room_labeler(root)(record["room_id"]) if record.get("room_id") else None,
            "told_by": _told_by(store, [record["id"]])[record["id"]], **_dated(store, record)}
    return _words(read), {**meta, "revision": read["revision"]}


def _text_document(root: pathlib.Path, address: Mapping[str, Any], revision: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    if address["kind"] == "note":
        from ouroboros.knowledge import note_descriptor

        note = _read_note(_note_address(root, address.get("scope"), address.get("topic")))
        if "field" in address:  # a value of the note's front matter (its whole summary, its archive reason, ...)
            return _field_document(dict(note.metadata), address["field"], revision, {"note_revision": note.revision})
        _bound(revision, note.revision)
        descriptor = note_descriptor(note)
        return note.text, {"revision": note.revision, "bytes": len(note.raw), "type": _project(
            descriptor.get("type"), address, ["type"], [ENTRY_BUDGET]), "note": _note_entry(root, descriptor),
                           "current_revision": note.revision}
    path, title, journal, form = _file_path(root, str(address.get("source")))
    if not path.is_file():
        raise MemoryReadError("not_found", f"{title} does not exist")
    raw = path.read_bytes()
    current = hashlib.sha256(raw).hexdigest()
    _bound(revision, current)
    return raw.decode("utf-8"), {"revision": current, "current_revision": current, "bytes": len(raw), "title": title,
                                 "path": _relative(root, path), "representation": form,
                                 "journal": {"kind": "journal", "journal": journal} if journal else None}


def _entry_document(root: pathlib.Path, address: Mapping[str, Any], revision: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    """One journal line (``field`` ``[]``: as stored) or one value of it; reading the line is record-sized."""
    journal, offset, field = str(address.get("journal")), address.get("offset"), address.get("field")
    if type(offset) is not int or offset < 0 or not isinstance(field, list):
        raise MemoryReadError("invalid_parameter", "an entry is {journal, offset, field: [keys]}")
    path = _journal_path(root, journal)
    if not path.is_file():
        raise MemoryReadError("not_found", "no such journal")
    with path.open("rb") as stream:
        if offset:
            stream.seek(offset - 1)
            if stream.read(1) != b"\n":
                raise MemoryReadError("not_found", "the offset is not the start of a journal line")
        raw = stream.readline()
    if not raw.endswith(b"\n"):
        raise MemoryReadError("not_found", "no complete journal line at that offset")
    context = {"journal": journal, "offset": offset}
    if field:
        try:
            row = json.loads(raw)
        except ValueError:
            raise MemoryReadError("not_found", "the line is not a JSON row; read field [] for it as stored") from None
        at = {"kind": "entry", **context}
        return _field_document(row, field, revision, {**context, "entry": _project(row, at, [], [ENTRY_BUDGET])})
    text = raw[:-1].decode("utf-8", errors="replace")  # the line as stored; undecodable bytes are replaced
    current = hashlib.sha256(text.encode("utf-8")).hexdigest()
    _bound(revision, current)
    return text, {**context, "revision": current, "current_revision": current, "field": [], "representation": "line"}


def _document(root: pathlib.Path, params: Dict[str, Any]) -> Dict[str, Any]:
    _only(params, "address", "revision", "start", "length")
    address, revision = _address(params), _text(params, "revision")
    start, length = _int(params, "start", 0, 0), _int(params, "length", DEFAULT_LENGTH, 1, MAX_LENGTH)
    if address["kind"] == "record":
        text, meta = _record_document(root, address, revision)
    elif address["kind"] in ("note", "file"):
        text, meta = _text_document(root, address, revision)
    elif address["kind"] == "entry":
        text, meta = _entry_document(root, address, revision)
    else:
        raise MemoryReadError("invalid_parameter", "a journal is read by catalog area=history; its entries here")
    total = len(text)
    if start > total:
        raise MemoryReadError("invalid_parameter", f"start is past the text's end ({total})")
    end = min(start + length, total)
    return {"address": address, **meta, "text": text[start:end],
            "range": {"start": start, "end": end, "total": total, "unit": "codepoint"}, "complete": end == total,
            "next": None if end == total else {"start": end, "revision": meta["revision"]}}


# --- changes ---------------------------------------------------------------------------------------

def _changes(root: pathlib.Path, params: Dict[str, Any]) -> Dict[str, Any]:
    _only(params, "position", "limit")
    position, limit = params.get("position"), _limit(params)
    if position is not None and not (isinstance(position, Mapping) and type(position.get("sequence")) is int
                                     and position["sequence"] >= 0
                                     and (position.get("record_id") is None or isinstance(position.get("record_id"), str))):
        raise MemoryReadError("invalid_parameter", "position is {sequence, record_id} as a changes answer returned it")
    store = _store(root)
    if store is None:
        if position is not None and position["sequence"]:
            raise MemoryReadError("position_mismatch", "the chronicle no longer exists")
        return {"since_known": position is not None, "position": dict(_ZERO), "newest": dict(_ZERO), **_empty()}
    if position is None:  # a baseline: earlier changes are not inventoried
        _none, newest = store.observation_snapshot()
        return {"since_known": False, "position": newest, "newest": newest, **_empty()}
    try:
        rows, newest = store.observation_snapshot(dict(position), limit=limit + 1)
    except ValueError as exc:
        raise MemoryReadError("position_mismatch", str(exc)) from None
    shown = rows[:limit]
    entries = [{"address": _record_address(row["id"]), "id": row["id"], "kind": row["kind"], "room_id": row.get("room_id"),
                "target_id": row.get("target_id") or None, "sequence": row["sequence"], "ts": row.get("ts")} for row in shown]
    after = {"sequence": shown[-1]["sequence"], "record_id": shown[-1]["id"]} if shown else dict(position)
    return {"since_known": True, "position": after, "newest": newest, "entries": entries, "next": None,
            "complete": len(rows) <= limit}


_OPERATIONS = {"catalog": _catalog, "document": _document, "changes": _changes}
