"""Memory tools over the chronicle: ``chronicle_write``, ``memory_read``, ``memory_mark``.

The mind writes its own records; the host signs them (``focus_signature``) and
expands what a page covers. A page names a range of its room (``from``/``to``
row addresses), the room's open rows up to an address (``to`` alone: the rows
the view counts as open, ``memory_inventory.open_room_rows``, so what pages
already seal and what came after stay out) or a set of tasks; the host turns
that request into the exact SET of the room's rows BEFORE the publication
lock (the lock is not re-entrant
and reading the chain needs none), adds the room's notes of those tasks,
counts each row's source class, stamps every covered task with the host's own
facts and verifies the quotes the writer chose. A refusal names the current
revision or room head and the conflicting ids, never their text.

An ``account`` is the mind's own text connecting experience of several rooms,
based on exact source versions (``sources``: each ``{id, revision?}``; the host
freezes the revision read, the body's sha256 and a draft's status then); it seals
no row, folds nothing and moves no head. A ``selection`` is the mind's separate
choice of the common view: show the account (``shown``) in place of the records it
names (``replaces``), or withdraw it. ``memory_read`` of an account names who wrote
it, which versions informed it, what changed in a source since (a later correction,
a decision, a local fold, a missing record) and how to read each original;
``revision`` reads a record as of one of its revisions.

A delegated child or nanny is not the integrating mind: the host signs its
page or part as a helper's draft carrying its own focus, which acts at once until
the mind's ``decision``; its note, correction, decision, account or selection is
refused ``not_integrator`` before anything is read or written, and so is its part
over records other than legacy sections and its own drafts.

``memory_read`` answers in text, one header line per record or row, and never
JSON. Every mode bounds itself at the source to ``tool_result_limit`` and names
its continuation on the second line (``next_after_seq``, ``next: from=…`` or
``next_start``); a single record or row larger than one page is reached by its
own address. Once the chronicle is active, reading writes nothing to disk.

Each tool first makes sure the chronicle is activated: the first call on an
install imports the legacy dialogue memory once (``chronicle_import``, no model
call), and every later call finds the receipt and goes straight on. A call that
meets another importer waits for it; when the import still has not completed
the tool answers ``memory_not_activated`` and reads and writes nothing, because
without the receipt a room would look empty and old rows would lose their
lineage epoch.

``memory_mark`` places a bookmark in my own words on a chronicle record, a chat
row, a task or a retained source; an optional quote must be an exact substring
of that target.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

from ouroboros import chat_chain, memory_inventory
from ouroboros.chronicle_import import LEGACY_ROOM_ID, row_lineage
from ouroboros.chronicle_quotes import SPEAKERS
from ouroboros.chronicle_store import (DELEGATED_FOCUS_ROLES, STORY_KINDS, ChronicleStore, PublishResult, capture,
                                       draft_signer, source_time_span, verify_quotes)
from ouroboros.dialogue_provenance import memory_row_header, recorded_finality, render_row_text, row_author
from ouroboros.knowledge import focus_signature
from ouroboros.mark_anchors import anchor_problem, describe as describe_anchor
from ouroboros.memory_view_account import source_change_lines
from ouroboros.tool_capabilities import tool_result_limit
from ouroboros.tools.registry import ToolEntry
from ouroboros.tools.tool_result import ToolResult, _publish_tool_result, completed_local_read
from ouroboros.utils import utc_now_iso

_LINEAGE = ("task_id", "parent_task_id", "root_task_id")
_ORDER = "older to newer"
_ROW_ORDER = "stored order"  # chat rows by address (``_read_rows``), whatever order their room reads in

# --- shared --------------------------------------------------------------------------------------

def _root(ctx: Any) -> Path:
    from ouroboros.tool_access import canonical_data_root

    return canonical_data_root(ctx)


class _NotActivated(Exception):
    """The one-time legacy import has not completed; the tool reads and writes nothing."""

    def __init__(self, receipt: Dict[str, Any]):
        super().__init__(str(receipt.get("reason") or receipt.get("kind")))
        self.receipt = receipt


def _activated(root: Path) -> ChronicleStore:
    """The store after the one-time legacy import: a repeat call is the fast path.

    A busy import lock is waited for until the other importer finishes; a refused
    or failed import raises ``_NotActivated``, so no tool reads or writes before the import.
    """
    store = ChronicleStore(root)
    try:
        receipt = store.ensure_activated(wait=True)
    except ValueError as exc:  # the store's own refusal to open its journal, not an argument error
        raise _NotActivated({"kind": "import_failed", "reason": f"{type(exc).__name__}: {exc}"}) from exc
    if not isinstance(receipt, dict) or receipt.get("kind") != "activation":
        raise _NotActivated(receipt if isinstance(receipt, dict) else {"kind": "import_failed"})
    return store


def _existing_store(root: Path) -> Optional[ChronicleStore]:
    """The store only when its journal exists; a reader never creates the chronicle itself."""
    store = ChronicleStore(root)
    return store if store.log_path.exists() else None


def _room(ctx: Any, root: Path, room_id: Any) -> str:
    """An explicit room, else this task's own room; chat id 0 is an address, not absence.

    A room is its chat id as text (or ``legacy``, the old memory's mixed room): a name such
    as ``Main`` is refused with the repair, never read as an empty room or written into one.
    """
    if room_id is not None and str(room_id).strip() != "":
        room = str(room_id).strip()
        if room != LEGACY_ROOM_ID and not room.lstrip("-").isdigit():
            raise ValueError(f"room_id {room!r} is not a room address: a room is its chat id as text, as "
                             "memory_read and my memory view print it (or 'legacy' for the old memory)")
        return room
    from ouroboros.dialogue_evidence import own_room_chat  # D15->D06 is lazy-only

    own = own_room_chat(ctx, root)
    if own is None:
        raise ValueError("room_id is required: this task has no room address")
    return str(own)


def _author_of(row: Dict[str, Any], pos: Optional[int], lineage: Dict[str, Any]) -> Dict[str, Any]:
    return row_author(row, pos=pos, **lineage)


def _reply(ctx: Any, payload: Dict[str, Any], *, code: str = "OK", meta: Optional[Dict[str, Any]] = None) -> str:
    status = "ok" if code == "OK" else "error"
    return _publish_tool_result(ctx, ToolResult(status=status, code=code, meta=meta or {},
                                                text=json.dumps(payload, ensure_ascii=False, sort_keys=True)))


def _arg_error(ctx: Any, message: str) -> str:
    return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR",
                                                text=f"⚠️ TOOL_ARG_ERROR: {message}"))


def _failure(ctx: Any, exc: BaseException) -> str:
    return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_REPORTED_FAILURE",
                                                text=f"⚠️ TOOL_ERROR: chronicle unavailable: {type(exc).__name__}: {exc}"))


def _refused(ctx: Any, reason: str, detail: str = "", **fields: Any) -> str:
    payload = {"ok": False, "reason": reason, "detail": detail,
               **{key: value for key, value in fields.items() if value not in (None, (), [])}}
    return _reply(ctx, payload, code="TOOL_REPORTED_FAILURE")


def _not_activated(ctx: Any, exc: _NotActivated) -> str:
    receipt = exc.receipt
    detail = "; ".join(str(part) for part in (receipt.get("kind"), receipt.get("reason"), receipt.get("detail"))
                       if part)
    return _refused(ctx, "memory_not_activated",
                    f"memory is not active ({detail}): the one-time legacy memory import has not completed "
                    "or the journal cannot be opened; nothing was read or written",
                    conflict_ids=list(receipt.get("conflict_ids") or ()))


def _published(ctx: Any, result: PublishResult, **extra: Any) -> str:
    """Saved: the record's address, its revision and sequence. Refused: ids and heads, never text."""
    if not result.ok:
        return _refused(ctx, result.reason, result.detail, current_revision=result.current_revision,
                        current_head=result.current_head, conflict_ids=list(result.conflict_ids))
    record = result.record or {}
    payload = {"ok": True, "reason": result.reason, "kind": record.get("kind"), "node_id": record.get("id"),
               "room_id": record.get("room_id"), "sequence": record.get("sequence"),
               "revision": result.current_revision, "room_head": result.current_head, **extra}
    return _reply(ctx, {k: v for k, v in payload.items() if v is not None},
                  meta={"chronicle_record_id": str(record.get("id") or ""), "memory_written": True})


# --- page coverage, host stamp, quotes (also used by the Light fallback writer) -------------------

def _task_rows(root: Path, room: str, task_ids: Iterable[Any]) -> List[Tuple[Dict[str, Any], Dict[str, Any], int]]:
    """Rows of these tasks (own, parent or root lineage) plus the owner's words bound to them.

    The binding facts: a task row's ``origin_message_ref``, the project task binding's
    ``source_ref`` and the latest promote annotation naming the task. Owner messages
    no fact binds are not here; a ``from``/``to`` range covers them.
    """
    from ouroboros.project_dialogue import entry_matches_source_ref, latest_chat_annotations
    from ouroboros.projects_registry import project_task_bindings  # D15->D17 is lazy-only

    wanted = {str(task).strip() for task in task_ids if str(task or "").strip()}
    if not wanted:
        raise ValueError("covers.task_ids names at least one task")
    refs = [row["source_ref"] for task, row in project_task_bindings(root).items()
            if task in wanted and isinstance(row.get("source_ref"), dict)]
    messages = {message for message, row in latest_chat_annotations(root).items()
                if str(row.get("target") or "") in wanted}
    selected, owner = [], []
    for address, row, pos in chat_chain.iter_room_rows(root, room):
        if wanted & {str(row.get(field) or "") for field in _LINEAGE}:
            selected.append((address, row, pos))
            if isinstance(row.get("origin_message_ref"), dict):
                refs.append(row["origin_message_ref"])
        elif str(row.get("direction") or "") == "in":
            owner.append((address, row, pos))
    selected += [entry for entry in owner if entry_matches_source_ref(entry[1], refs)
                 or str(entry[1].get("client_message_id") or "") in messages]
    return sorted(selected, key=lambda entry: entry[2])


def _note_covers(root: Path, room: str, ids: Any) -> tuple:
    """A page may close selected notes even when no chat row remains open; no mark is affected."""
    if (not isinstance(ids, list) or not ids or any(not isinstance(i, str) or not i.strip() for i in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("covers.note_ids is a non-empty array of distinct note ids")
    store = _activated(root)
    notes = [store.get(ident) for ident in ids]
    if any(not note or note.get("kind") != "note" or str(note.get("room_id")) != room for note in notes):
        raise ValueError("covers.note_ids must resolve to notes of this room")
    tasks = list(dict.fromkeys(str(note["task_id"]) for note in notes if note.get("task_id")))
    from ouroboros.dialogue_positions import last_stream_position

    head = max(0, last_stream_position(root))  # an empty stream starts at position zero
    covers = {"room_id": room, "mode": "notes", "request": {"note_ids": ids},
              "rows": [f"note:{ident}" for ident in ids], "first": None, "last": None, "count": 0,
              "task_ids": tasks, "note_ids": ids, "stream_span": [head, head],
              "ts_span": source_time_span(note.get("ts") for note in notes)}
    return covers, {"rows": 0, "notes": len(notes), "by_author": {}}, []


def _expand(root: Path, room: str, lineage: Dict[str, Any], *, from_addr: Any = None, to_addr: Any = None,
            task_ids: Any = None, note_ids: Any = None) -> Tuple[Dict[str, Any], Dict[str, Any], List[Tuple[Dict[str, Any], Dict[str, Any], int]]]:
    if note_ids is not None:
        if task_ids is not None or from_addr or to_addr:
            raise ValueError("covers.note_ids is a separate coverage choice, without row bounds or task_ids")
        return _note_covers(root, room, note_ids)
    if task_ids is not None and (from_addr or to_addr):
        raise ValueError("covers is either {from, to}, {to} or {task_ids}, not both")
    if task_ids is not None:
        if not isinstance(task_ids, (list, tuple)):
            raise ValueError("covers.task_ids is a list of task ids")
        found, request, mode = _task_rows(root, room, task_ids), {"task_ids": [str(t) for t in task_ids]}, "tasks"
        # Like {to}: the task's rows still open, so a page after a sweep of the room's open rows takes the rest;
        # when none is left the full set goes on and the store's refusal names the page(s) that sealed them.
        still_open = {address["row_sha256"] for address, _meta, _pos in memory_inventory.open_room_rows(root, room)}
        found = [entry for entry in found if entry[0]["row_sha256"] in still_open] or found
    elif to_addr and not from_addr:  # "up to here": the room's open rows, as the view counts them, to the address
        open_rows = {address["row_sha256"] for address, _meta, _pos in memory_inventory.open_room_rows(root, room)}
        found = [entry for entry in chat_chain.iter_room_rows(root, room, to_addr=to_addr)
                 if entry[0]["row_sha256"] in open_rows]
        request, mode = {"to": to_addr}, "open_to"
    else:
        if not from_addr or not to_addr:
            raise ValueError("covers is {from, to} (row addresses, inclusive), {to} (the open rows up to it) or {task_ids}")
        found = list(chat_chain.iter_room_rows(root, room, from_addr=from_addr, to_addr=to_addr))
        request, mode = {"from": from_addr, "to": to_addr}, "range"
    seen, rows = set(), []
    for entry in found:  # a byte-identical redelivered row is one member of the set
        if entry[0]["row_sha256"] not in seen:
            seen.add(entry[0]["row_sha256"])
            rows.append(entry)
    if not rows:
        raise ValueError(f"covers resolve to no {'open ' if mode == 'open_to' else ''}row of room {room}")
    tasks = list(dict.fromkeys([str(row.get("task_id")) for _a, row, _p in rows if row.get("task_id")]
                               + (list(request.get("task_ids") or []))))
    store = _existing_store(root)
    notes: List[str] = []
    if store is not None:
        sealed = store.sealed_row_refs(room)
        notes = [record["id"] for record in store.records(room, kinds=["note"])
                 if str(record.get("task_id") or "") in tasks and f"note:{record['id']}" not in sealed]
    authors = Counter(_author_of(row, pos, lineage)["kind"] for _a, row, pos in rows)
    covers = {"room_id": room, "mode": mode, "request": request,
              "rows": [address["row_sha256"] for address, _r, _p in rows] + [f"note:{n}" for n in notes],
              "first": rows[0][0], "last": rows[-1][0], "count": len(rows), "task_ids": tasks, "note_ids": notes,
              "ts_span": source_time_span(row.get("ts") for _a, row, _p in rows),
              "stream_span": [rows[0][2], rows[-1][2]]}
    facts = {"rows": len(rows), "notes": len(notes), "by_author": dict(sorted(authors.items()))}
    return covers, facts, rows


def page_covers(root: Any, room_id: Any, *, from_addr: Any = None, to_addr: Any = None,
                task_ids: Any = None, note_ids: Any = None) -> Dict[str, Any]:
    """The exact row set a page request names, with its coverage facts and the rows read.

    ``{"covers", "coverage_facts", "rows": [(address, row), ...]}``. ``from_addr``/``to_addr``
    bound the room's stream inclusively; ``to_addr`` alone takes the room's open rows up to it;
    ``task_ids`` takes those tasks' rows and the owner's words bound to them. Unresolved bounds
    raise ``chat_chain.RowAddressError``.
    """
    covers, facts, rows = _expand(Path(root), str(room_id), row_lineage(Path(root)), from_addr=from_addr,
                                  to_addr=to_addr, task_ids=task_ids, note_ids=note_ids)
    return {"covers": covers, "coverage_facts": facts, "rows": [(address, row) for address, row, _p in rows]}


def host_stamp(root: Any, task_ids: Iterable[Any], *, rows: Iterable[Any] = ()) -> Dict[str, Any]:
    """The host's facts about each covered task, so a page cannot silently call a failure done.

    ``{"tasks": [...], "computed_at"}``. The facts are ``terminal_projection.stamp_facts``:
    per task, its terminal projection row, then its host facts row (both among ``rows``,
    the page's already-read ``(address, row)`` pairs), then the strict task result,
    else ``not_recorded``.
    """
    from ouroboros.terminal_projection import stamp_facts  # D15->D17 is lazy-only

    return {"tasks": list(stamp_facts(root, task_ids, rows=rows).values()), "computed_at": utc_now_iso()}


def _quote_resolver(root: Path, lineage: Dict[str, Any],
                    positions: Optional[Dict[str, int]] = None) -> Callable[[Any], Optional[Tuple[str, str]]]:
    """``address -> (rendered row text, author kind)`` for ``verify_quotes``; reads chat, takes no lock."""
    def resolve(address: Any) -> Optional[Tuple[str, str]]:
        try:
            row, found = chat_chain.resolve_row(root, address)
        except ValueError:
            return None
        if row is None:
            return None
        try:
            author = _author_of(row, None, lineage)
        except TypeError:  # a pre-epoch candidate needs its stream position
            sha = found["address"]["row_sha256"]
            pos = (positions or {}).get(sha)
            pos = pos if pos is not None else chat_chain.stream_position_of(root, found["address"])
            if pos is None:
                return None
            author = _author_of(row, pos, lineage)
        return render_row_text(row), author["kind"]
    return resolve


def check_quotes(root: Any, quotes: Any) -> Tuple[bool, Optional[int]]:
    """Each quote is an exact substring of the row at its address, by the named speaker class.

    ``(True, None)`` or ``(False, index of the first failing quote)``.
    """
    resolver = _quote_resolver(Path(root), row_lineage(Path(root)))
    for index, quote in enumerate(quotes or ()):
        if verify_quotes([quote], resolver) is not None:
            return False, index
    return True, None


# --- chronicle_write ------------------------------------------------------------------------------

def _write_page(ctx: Any, root: Path, store: ChronicleStore, author: Dict[str, Any], a: Dict[str, Any]) -> str:
    covers_arg = a["covers"]
    if not isinstance(covers_arg, dict):
        return _arg_error(ctx, "a page needs covers: {from, to} row addresses, {to}, {task_ids} or {note_ids}")
    room, lineage = _room(ctx, root, a["room_id"]), row_lineage(root)
    try:
        covers, facts, rows = _expand(root, room, lineage, from_addr=covers_arg.get("from"),
                                      to_addr=covers_arg.get("to"), task_ids=covers_arg.get("task_ids"), note_ids=covers_arg.get("note_ids"))
    except chat_chain.RowAddressError as exc:
        return _refused(ctx, "target_missing", f"a covers bound does not resolve: {exc.resolution.get('status')}")
    stamp = host_stamp(root, covers["task_ids"], rows=rows)
    resolver = _quote_resolver(root, lineage, {address["row_sha256"]: pos for address, _r, pos in rows})
    result = store.publish_page(room_id=room, text=a["text"], covers=covers, author=author,
                                quotes=a["quotes"] or (), host_stamp=stamp, metadata={"coverage_facts": facts,
                                "terminal_after_pos": max((pos for _a, _r, pos in chat_chain.iter_room_rows(root, room)), default=-1)},
                                quote_resolver=resolver, expected_sequence=a["expected_sequence"])
    if not result.ok:
        return _published(ctx, result)
    statuses = Counter(entry["status"] or "unknown" for entry in stamp["tasks"])
    return _published(ctx, result, rows=covers["count"], notes=len(covers["note_ids"]), coverage_facts=facts,
                      first=chat_chain.format_address(covers["first"]) if covers["first"] else None,
                      last=chat_chain.format_address(covers["last"]) if covers["last"] else None,
                      stamp=dict(sorted(statuses.items())))


def _own_or_legacy(record: Optional[Dict[str, Any]], author: Dict[str, Any]) -> bool:
    """Whether a delegated focus may fold this member: a legacy section or its own draft, never the mind's records.

    A missing id is left to the store's own ``target_missing`` refusal.
    """
    if record is None or record.get("kind") == "legacy":
        return True
    signer = record.get("author") if isinstance(record.get("author"), dict) else {}
    return signer.get("kind") == "helper" and isinstance(signer.get("focus"), dict) and (
        str(signer.get("task_id") or "") == str(author.get("task_id") or ""))


def _fold_effect(store: ChronicleStore, action: str, ids: Iterable[Any], text: str) -> str:
    """What a fold did, measured (the records' own words before, the new text after); never a claim about Max."""
    ids = [str(ident) for ident in ids]
    before = sum(len(str((store.get(ident) or {}).get("text") or "")) for ident in ids)
    return (f"{action} {len(ids)} record{'' if len(ids) == 1 else 's'}: {before:,} -> {len(text or ''):,} characters. "
            "Originals remain readable by address. This task's view is not refreshed; the next start re-measures "
            "its Max room and rewrites the cached story from the first changed retelling.")


def _write_part(ctx: Any, root: Path, store: ChronicleStore, author: Dict[str, Any], a: Dict[str, Any]) -> str:
    members = a["member_ids"]
    if not isinstance(members, list) or not members:
        return _arg_error(ctx, "a part needs member_ids: adjacent records of one lower level of one room")
    foreign = [str(m) for m in members if author["kind"] == "helper" and not _own_or_legacy(store.get(str(m)), author)]
    if foreign:  # the mind's pages and other writers' records are folded by the integrating mind
        return _refused(ctx, "not_integrator",
                        f"a delegated {author['focus']['role']} folds into its draft part only legacy sections and "
                        "its own drafts; folding other records is the integrating mind's to do, so put it in your "
                        "report. Nothing was written", conflict_ids=foreign)
    room = a["room_id"]
    if room is None or str(room).strip() == "":
        first = store.get(str(members[0]))  # the members' own room unless one is named
        room = first["room_id"] if first else _room(ctx, root, None)
    resolver = _quote_resolver(root, row_lineage(root))
    result = store.publish_part(room_id=room, text=a["text"], member_ids=[str(m) for m in members], author=author,
                                expected_sequence=a["expected_sequence"], quotes=a["quotes"] or (), quote_resolver=resolver)
    return _published(ctx, result, effect=_fold_effect(store, "folded", members, a["text"]) if result.ok else None)


def _write_note(ctx: Any, root: Path, store: ChronicleStore, author: Dict[str, Any], a: Dict[str, Any]) -> str:
    task = str(a["task_id"] or author.get("task_id") or "")
    return _published(ctx, store.write_note(room_id=_room(ctx, root, a["room_id"]), task_id=task, text=a["text"],
                                            author=author))


def _write_correction(ctx: Any, root: Path, store: ChronicleStore, author: Dict[str, Any], a: Dict[str, Any]) -> str:
    if not a["target_id"]:
        return _arg_error(ctx, "a correction needs target_id (a page, part, note, legacy or gap record)")
    return _published(ctx, store.correct(str(a["target_id"]), a["text"], author,
                                         expected_revision=a["expected_revision"] or None,
                                         expected_sequence=a["expected_sequence"]))


def _write_decision(ctx: Any, root: Path, store: ChronicleStore, author: Dict[str, Any], a: Dict[str, Any]) -> str:
    if not a["target_id"] or not isinstance(a["accepted"], bool):
        return _arg_error(ctx, "a decision needs target_id (a helper's draft), accepted true/false and a reason")
    return _published(ctx, store.decide(str(a["target_id"]), a["accepted"], author, str(a["reason"] or "")))


def _write_account(ctx: Any, root: Path, store: ChronicleStore, author: Dict[str, Any], a: Dict[str, Any]) -> str:
    sources = a["sources"]
    if not isinstance(sources, list) or not sources:
        return _arg_error(ctx, "an account needs sources: the records it is based on, each {id, revision?} or an id")
    result = store.publish_account(room_id=_room(ctx, root, a["room_id"]), text=a["text"], sources=sources, author=author)
    ids = [src.get("id") if isinstance(src, dict) else src for src in sources]
    return _published(ctx, result, sources=len(sources) if result.ok else None, effect=_fold_effect(
        store, "wrote an account over", ids, a["text"]) + " It tells nothing in their place until a selection shows it."
        if result.ok else None)


def _write_selection(ctx: Any, root: Path, store: ChronicleStore, author: Dict[str, Any], a: Dict[str, Any]) -> str:
    if not a["target_id"]:
        return _arg_error(ctx, "a selection needs target_id (an account), replaces (record ids, may be empty), "
                              "shown (default true) and a reason")
    replaces = a["replaces"] if a["replaces"] is not None else []
    if not isinstance(replaces, list):
        return _arg_error(ctx, "replaces is a list of record ids the account tells in its place")
    shown = True if a["shown"] is None else a["shown"]
    result = store.select_account(str(a["target_id"]), replaces=[str(r) for r in replaces], author=author,
                                  reason=str(a["reason"] or ""), shown=shown)
    effect = (_fold_effect(store, "the account now tells", replaces, str((store.get(str(a["target_id"])) or {}).get("text") or ""))
              if shown else "withdrawn: the account leaves the common view and the records it told return there. This "
              "task's view is not refreshed; the next start re-measures its Max room.")
    return _published(ctx, result, replaces=len(replaces) if result.ok else None, shown=shown if result.ok else None,
                      effect=effect if result.ok else None)


_WRITERS = {"page": _write_page, "part": _write_part, "note": _write_note,
            "correction": _write_correction, "decision": _write_decision,
            "account": _write_account, "selection": _write_selection}
# A delegated focus is not the integrating mind: it drafts pages and parts under its own
# signature (a part over legacy sections and its own drafts only, ``_write_part``), and the mind
# accepts or rejects them; notes, corrections and decisions stay the mind's.
_DRAFTING_ROLES = DELEGATED_FOCUS_ROLES
_DRAFT_KINDS = frozenset({"page", "part"})


def _signed_author(ctx: Any, kind: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """``(author, "")`` as the host signs this write, or ``(None, role)`` when this focus may not write ``kind``.

    The root, Main and consciousness write as the mind. A delegated child or nanny
    publishes a page or part as a helper's draft carrying its own focus, task and
    route, which the model never chooses.
    """
    author = focus_signature(ctx)
    role = author["focus"]["role"]
    if role not in _DRAFTING_ROLES:
        return author, ""
    if kind not in _DRAFT_KINDS:
        return None, role
    return {**author, "kind": "helper"}, ""


def _chronicle_write(ctx: Any, kind: str = "", room_id: Any = None, text: str = "", covers: Any = None,
                     member_ids: Any = None, quotes: Any = None, task_id: str = "", target_id: str = "",
                     expected_revision: Optional[str] = None, expected_sequence: Optional[int] = None,
                     accepted: Optional[bool] = None, reason: str = "", sources: Any = None, replaces: Any = None,
                     shown: Optional[bool] = None) -> str:
    writer = _WRITERS.get(str(kind or ""))
    if writer is None:
        return _arg_error(ctx, "kind is page, part, note, correction, decision, account or selection")
    author, role = _signed_author(ctx, str(kind))
    if author is None:
        return _refused(ctx, "not_integrator",
                        f"a delegated {role} publishes pages and parts only as drafts in its own name, which the "
                        f"integrating mind accepts or rejects; a {kind} is the integrating mind's to write, so put "
                        "it in your report. Nothing was written")
    args = {"room_id": room_id, "text": text, "covers": covers, "member_ids": member_ids, "quotes": quotes,
            "task_id": task_id, "target_id": target_id, "expected_revision": expected_revision,
            "expected_sequence": expected_sequence, "accepted": accepted, "reason": reason, "sources": sources,
            "replaces": replaces, "shown": shown}
    try:
        root = _root(ctx)
        return writer(ctx, root, _activated(root), author, args)
    except _NotActivated as exc:
        return _not_activated(ctx, exc)
    except ValueError as exc:
        return _arg_error(ctx, str(exc))
    except (OSError, TimeoutError) as exc:
        return _failure(ctx, exc)


# --- memory_read: rendering ----------------------------------------------------------------------

def _author_label(author: Any) -> str:
    author = author if isinstance(author, dict) else {}
    kind = str(author.get("kind") or "unknown")
    if kind == "mind":
        focus = author.get("focus") if isinstance(author.get("focus"), dict) else {}
        return f"mind ({focus.get('role') or 'focus not recorded'} {author.get('task_id') or ''})".replace(" )", ")")
    if kind == "helper" and isinstance(author.get("focus"), dict):  # a delegated child's or nanny's draft
        return f"helper ({draft_signer(author)})"
    detail = author.get("attribution") or author.get("operation") or author.get("writer") or ""
    return f"{kind} ({detail})" if detail else kind


def _span(span: Any) -> str:
    span = span if isinstance(span, dict) else {}
    if not span.get("start") and not span.get("end"):
        return "period unknown"
    return f"{span.get('start')}–{span.get('end')}" + (" (incomplete)" if span.get("incomplete") else "")


def _periods(store: ChronicleStore, root: Path, records: List[Dict[str, Any]]) -> Dict[str, memory_inventory.Period]:
    """Each listed part's or account's period as read from its members' rooms (``memory_inventory.record_period``)."""
    parts = [record for record in records if record.get("kind") in ("part", "account")]
    units = {unit.record_id: unit for unit in memory_inventory.legacy_units(store, root)} if parts else {}
    return {record["id"]: memory_inventory.record_period(store, record, units) for record in parts}


def _covers_summary(record: Dict[str, Any], period: Optional[memory_inventory.Period]) -> str:
    covers = record.get("covers") if isinstance(record.get("covers"), dict) else {}
    kind = record.get("kind")
    if kind == "page":
        notes = len(covers.get("note_ids") or [])
        return (f"covers {_span(covers.get('ts_span'))}, {covers.get('count', 0)} rows"
                + (f" + {notes} notes" if notes else ""))
    if kind == "part":  # the period its members' rows give, the same the view prints; never the recorded aggregate
        dated = _span(period.span) + period.note() if period is not None else "period unknown"
        return f"folds {len(covers.get('member_ids') or [])} records, {dated}"
    if kind == "account":  # the sources' span (not every event in it); how many changed since the account was written
        dated = _span(period.span) + period.note() if period is not None else "period unknown"
        sources = record.get("sources") or []
        changed = sum(1 for src in sources if isinstance(src, dict) and (
            src.get("missing") or src.get("later_corrections") or src.get("folded_into")
            or (src.get("status_now") and src.get("status_now") != src.get("status"))))
        return f"based on {len(sources)} sources, {dated}" + (f"; {changed} changed since" if changed else "")
    if kind == "selection":
        return (f"replaces {len(record.get('replaces') or [])}; " + ("shown in the common view" if record.get("shown")
                                                                     else "withdrawn from the common view"))
    if kind in ("legacy", "gap"):
        raw = covers.get("raw_range") if isinstance(covers.get("raw_range"), dict) else {}
        meta = record.get("metadata") or {}
        where = f"raw_range {raw['pos'][0]}–{raw['pos'][1]}" if raw.get("status") == "exact" and raw.get("pos") else (
            "raw_range unknown" if raw else "")
        label = meta.get("label") or meta.get("legacy_type") or ""
        return "; ".join(part for part in (str(label), where, str(meta.get("legacy_range_text") or "")) if part)
    if kind == "note":
        return f"task {record.get('task_id') or ''}"
    return ""


def _stamp_finality(entry: Dict[str, Any], before: str, after: str = "") -> str:
    """The finality and authority one stamp entry copied from its source, framed for its reader (a host
    facts entry without one is unknown); '' when the entry recorded neither."""
    finality = recorded_finality(entry.get("outcome_final"), entry.get("outcome_authority"),
                                 host_facts=entry.get("source") == "host_task_facts")
    return f"{before}{finality}{after}" if finality else ""


def _stamp_summary(stamp: Any) -> str:
    """``stamp: <task_id>=<status> (<finality>), …``: which covered task failed, and which status is
    not yet final, shows in the listing; the recorded status stays as it was stamped."""
    tasks = stamp.get("tasks") if isinstance(stamp, dict) else None
    if not isinstance(tasks, list) or not tasks:
        return ""
    return "stamp: " + ", ".join(f"{entry.get('task_id') or '?'}={entry.get('status') or 'unknown'}"
                                 + _stamp_finality(entry, " (", ")") for entry in tasks if isinstance(entry, dict))


def _target_label(target: Any) -> str:
    target = target if isinstance(target, dict) else {}
    kind = target.get("kind")
    if kind == "chronicle":
        return f"node {target.get('id')}"
    if kind == "chat_row":
        try:
            return chat_chain.format_address(target)
        except (KeyError, TypeError, ValueError):
            return "chat row"
    if kind == "task":
        return f"task {target.get('task_id')}"
    location = f" at {target['location']}" if target.get("location") else ""
    return f"source {target.get('path') or target.get('kind') or '?'}{location}"


def _record_header(record: Dict[str, Any], period: Optional[memory_inventory.Period] = None) -> str:
    parts = [f"{record.get('kind')} {record.get('id')}", f"room {record.get('room_id')}",
             _author_label(record.get("author"))]
    if record.get("kind") == "mark":
        parts += [f"scope {record.get('scope')}", f"target {_target_label(record.get('target_ref'))}",
                  f"visibility {record.get('visibility')}"]
        if isinstance(record.get("anchor"), dict):
            parts.append(f"meets me at {describe_anchor(record['anchor'])}")
    for part in (record.get("status"), _covers_summary(record, period), _stamp_summary(record.get("host_stamp"))):
        if part:
            parts.append(str(part))
    if record.get("target_id"):
        parts.append(f"target {record['target_id']}")
    if record.get("folded_into"):
        parts.append(f"folded into {record['folded_into']}")
    fixes = record.get("corrections") or []
    if fixes:  # the acting revision is the last correction's, signed by its own author, not the record's
        parts.append(f"revision {record.get('revision')} (corrected by {_author_label(fixes[-1].get('author'))})")
    elif record.get("revision"):
        parts.append(f"revision {record['revision']}")
    parts.append(f"seq {record.get('sequence')}")
    return "[" + "; ".join(parts) + "]"


def _listed(record: Dict[str, Any], periods: Dict[str, memory_inventory.Period]) -> str:
    """One record as listed in its room: the header, then the text that acts."""
    header = _record_header(record, periods.get(record["id"]))
    if record.get("kind") == "mark":
        quote = record.get("quote") if record.get("visibility") == "full" else None
        return header + "\n" + str(record.get("text") or "") + (f"\nquote: {quote}" if quote else "")
    return header + "\n" + str(record.get("current_text", record.get("text")) or "")


def _row_line(address: Dict[str, Any], row: Dict[str, Any], pos: int, lineage: Dict[str, Any]) -> Tuple[str, str]:
    """``(header, text)`` of one chat row: the one memory-row header and its words without JSON."""
    return memory_row_header(address, row, author=_author_of(row, pos, lineage)), render_row_text(row)


def _fit(head: str, items: List[str], exhausted: bool, continuation: Callable[[int, bool], str],
         limit: int) -> Tuple[str, int]:
    """The longest prefix of ``items`` that fits ``limit`` with ``head`` and its continuation line."""
    used = len(head) + 1
    sizes = [used]
    for item in items:
        sizes.append(sizes[-1] + 1 + len(item))
    for k in range(len(items), -1, -1):
        if sizes[k] > limit:
            continue
        cont = continuation(k, exhausted and k == len(items))
        if sizes[k] + len(cont) <= limit:
            return head + "\n" + cont + "".join("\n" + item for item in items[:k]), k
    return head + "\n" + continuation(0, False), 0


def _collect(source: Iterator[Any], render: Callable[[Any], str], limit: int) -> Tuple[List[Any], List[str], bool]:
    """Render items until the page is certainly full (and at least two are seen), or the source ends."""
    kept, texts, total = [], [], 0
    for value in source:
        kept.append(value)
        texts.append(render(value))
        total += len(texts[-1]) + 1
        if total > limit and len(kept) >= 2:
            return kept, texts, False
    return kept, texts, True


def _window(head: str, text: str, start: int, limit: int, resume: Callable[[int], str]) -> str:
    """A character window of one document; the second line names ``next_start`` or completeness."""
    total = len(text)
    start = min(max(int(start or 0), 0), total)
    widest = f"chars {total}–{total} of {total}; next_start={total} ({_ORDER}); " + resume(total)
    room = max(limit - len(head) - len(widest) - 2, 0)
    end = min(start + room, total)
    cont = (f"chars {start}–{end} of {total}; next_start={end} ({_ORDER}); " + resume(end) if end < total
            else f"chars {start}–{end} of {total}; complete")
    return head + "\n" + cont + "\n" + text[start:end]


# --- memory_read: modes --------------------------------------------------------------------------

def _active_marks(store: ChronicleStore, room: str, after_seq: int) -> List[Dict[str, Any]]:
    active = {mark["id"]: mark for mark in store.active_marks(room)}
    return [{**active[record["id"]], "sequence": record["sequence"]}
            for record in store.records(kinds=["mark"], after_seq=after_seq) if record["id"] in active]


def _read_records(root: Path, room: str, after_seq: int, limit: int) -> str:
    store = _existing_store(root)
    head = f"room {room}; head {store.room_head(room) if store is not None else 0}"
    entries: List[Dict[str, Any]] = []
    periods: Dict[str, memory_inventory.Period] = {}
    if store is not None:
        entries = sorted(store.room_records(room, after_seq=after_seq) + _active_marks(store, room, after_seq),
                         key=lambda record: record["sequence"])
        periods = _periods(store, root, entries)
    kept, texts, exhausted = _collect(iter(entries), lambda record: _listed(record, periods), limit)

    def continuation(k: int, done: bool) -> str:
        last = kept[k - 1]["sequence"] if k else after_seq
        if done:
            return f"complete: no records after seq {last} ({_ORDER})"
        return f"next_after_seq={last}: more records follow ({_ORDER})"

    text, shown = _fit(head, texts, exhausted, continuation, limit)
    if shown or not kept:
        return text
    big = kept[0]  # one record larger than a page is read through its own address
    size = len(str(big.get("current_text", big.get("text")) or ""))
    stub = (f"{_record_header(big, periods.get(big['id']))}\n(text of {size} chars does not fit this page: "
            f"memory_read(node_id={big['id']}) pages it)")
    return f"{head}\n{continuation(1, exhausted and len(kept) == 1)}\n{stub}"


def _read_rows(root: Path, room: str, bounds: Dict[str, Any], start: int, limit: int) -> str:
    """One room's chat rows by address in stored order: from/to and ``next: from=`` bound stored positions
    (``chat_chain.iter_room_rows``). A room holding imported earlier messages reads by send time in its
    view and ``chat_history``; here they stand where they were stored, and every page names the order."""
    try:
        int(room)
    except ValueError as exc:
        raise ValueError("rows are read from a chat room: room_id is its numeric chat id") from exc
    lineage = row_lineage(root)
    task_id = str(bounds.get("task_id") or "")
    gaps: set = set()  # what a Project's rows read so may lack (``chat_chain.room_rows``): never "complete" unsaid
    source = chat_chain.iter_room_rows(root, room, from_addr=bounds.get("from") or None,
                                       to_addr=bounds.get("to") or None, task_ids=[task_id] if task_id else None,
                                       period=bounds.get("period") or None, gaps=gaps)
    filters = [f"{key} {bounds[key]}" for key in ("from", "to", "task_id") if bounds.get(key)]
    if bounds.get("period"):
        filters.append(f"period {_span({'start': bounds['period'].get('from'), 'end': bounds['period'].get('to')})}")
    head = f"room {room}; rows" + ("; " + "; ".join(filters) if filters else "")
    offset = max(int(start or 0), 0)
    lines: List[Tuple[str, str]] = []

    def render(entry: Any) -> str:
        header, body = _row_line(*entry, lineage)
        if not lines and offset:
            body = f"(chars {min(offset, len(body))}–{len(body)} of {len(body)}) " + body[offset:]
        lines.append((header, body))
        return header + " " + body

    kept, texts, exhausted = _collect(source, render, limit)
    if gaps:
        from ouroboros.room_appearances import GAP_DETAILS

        head += "".join(f"; incomplete: {GAP_DETAILS[kind]}" for kind in sorted(gaps))

    def continuation(k: int, done: bool) -> str:
        if done:
            return f"complete: no further rows ({_ROW_ORDER})"
        return f"next: from={chat_chain.format_address(kept[k][0])} (same room and filters; {_ROW_ORDER})"

    text, shown = _fit(head, texts, exhausted, continuation, limit)
    if shown or not kept:
        return text
    address, row, _pos = kept[0]  # one row larger than a page: its words by character window
    header, words = lines[0][0], render_row_text(row)
    after = continuation(1, exhausted and len(kept) == 1)
    resume = f"next: from={chat_chain.format_address(address)} start={len(words)} (same room and filters; {_ROW_ORDER})"
    marker = f"(chars {len(words)}–{len(words)} of {len(words)}) "
    room_left = max(limit - len(head) - max(len(resume), len(after)) - len(header) - len(marker) - 3, 0)
    begin = min(offset, len(words))
    end = min(begin + room_left, len(words))
    cont = (f"next: from={chat_chain.format_address(address)} start={end} (same room and filters; {_ROW_ORDER})"
            if end < len(words) else after)
    return f"{head}\n{cont}\n{header} (chars {begin}–{end} of {len(words)}) {words[begin:end]}"


def _source_section(src: Dict[str, Any]) -> str:
    """One source of an account: the version it read, what changed since, and how to read the original."""
    head = f"source {src.get('kind')} {src.get('id')} (room {src.get('room_id')}): revision {src.get('revision')} used"
    if src.get("status") and src.get("status") != "final":  # a draft's state then; my own record has none to tell
        head += f", {src['status']} then"
    if src.get("missing"):
        return head + "; NOT in this chronicle: its words cannot be read here"
    facts = []
    if not src.get("revision_known"):
        facts.append(f"that revision is not in this chronicle; current revision {src.get('current_revision')}")
    if src.get("later_corrections"):
        facts.append("later corrections not in this account: " + ", ".join(src["later_corrections"]))
    if src.get("status_now") and src.get("status_now") != src.get("status"):
        facts.append(f"now {src['status_now']}")
    if src.get("folded_into"):
        facts.append(f"since folded into part {src['folded_into']}")
    read = f"memory_read(node_id={src.get('id')}, revision={src.get('revision')}) reads that version; without revision, as it acts now"
    return "\n".join(["; ".join([head, *facts, read]), *source_change_lines(src)])


def _node_document(store: ChronicleStore, record: Dict[str, Any], revision: str = "") -> Tuple[Dict[str, Any], str]:
    """The record as it acts (interpreted when it is a story record), or as of ``revision``, and its full text document."""
    room, kind = record["room_id"], record["kind"]
    read = store.interpret([record], revision=revision or None)[0]  # the store's one composition at every revision
    shown = read if revision or kind in STORY_KINDS and read.get("status") not in ("rejected", "candidate") else record
    later: List[str] = read.get("later_corrections") or []
    if kind == "mark":
        shown = next((m for m in store.active_marks(room) if m["id"] == record["id"]), None) or record
        shown = {**shown, "sequence": record["sequence"]}
    sections: List[str] = []
    covers = record.get("covers") if isinstance(record.get("covers"), dict) else {}
    if kind == "page":
        sections.append(f"covers ({covers.get('mode')}): {covers.get('count')} rows, "
                        f"{_address_text(covers.get('first'))} – {_address_text(covers.get('last'))}; "
                        f"tasks: {', '.join(covers.get('task_ids') or []) or 'none'}; "
                        f"notes: {', '.join(covers.get('note_ids') or []) or 'none'}")
    elif kind == "part":
        sections.append("members: " + ", ".join(covers.get("member_ids") or []))
    elif kind == "account":
        sections.append(f"written {str(record.get('ts') or '')[:10]} by {_author_label(record.get('author'))} over "
                        f"{len(record.get('sources') or [])} sources; nothing sealed or folded; shown in the common view "
                        "only by my selection (chronicle_write kind=selection)")
        sections += [_source_section(src) for src in (shown.get("sources") or record.get("sources") or [])]
    elif kind == "selection":
        sections.append(f"account: memory_read(node_id={record.get('target_id')}); replaces: "
                        + (", ".join(record.get("replaces") or []) or "nothing") + f"; shown={record.get('shown')}")
    elif covers.get("raw_range"):
        raw = covers["raw_range"]
        sections.append(f"raw_range {raw.get('status')}: positions {raw.get('pos')}; rows: memory_read(rows=true, "
                        f"room_id={record['room_id']}, from={_address_text(raw.get('first'))}, "
                        f"to={_address_text(raw.get('last'))})")
    for entry in (record.get("host_stamp") or {}).get("tasks") or []:
        verdict = f"; verdict: {entry['review_verdict']}" if entry.get("review_verdict") else ""
        finality = _stamp_finality(entry, "; ")
        sections.append(f"stamp {entry.get('task_id')}: status={entry.get('status')}; outcome={entry.get('outcome')}; "
                        f"phase={entry.get('outcome_phase')}{verdict}{finality}; from {entry.get('source')}; "
                        f"result: get_task_result(task_id={entry.get('task_id')})")
    for quote in record.get("quotes") or []:
        sections.append(f"quote ({quote.get('speaker')}, {quote.get('address')}): {quote.get('text')}")
    meta = record.get("metadata") or {}
    if meta.get("legacy_source_ref"):
        sections.append("retelling source: memory_read(source_ref="
                        + json.dumps(meta["legacy_source_ref"], ensure_ascii=False, sort_keys=True) + ")")
    for ref in record.get("source_refs") or []:
        sections.append("retained source: memory_read(source_ref=" + json.dumps(ref, ensure_ascii=False, sort_keys=True) + ")")
    if kind == "mark":
        target = record.get("target_ref")
        source = isinstance(target, dict) and target.get("kind") == "task_source"
        if source:  # the author's own read recipe resolves in the author's drive only
            target = {key: value for key, value in target.items() if key != "read"}
        sections.append(f"target: {json.dumps(target, ensure_ascii=False, sort_keys=True)}")
        if source:
            from ouroboros.memory_view import source_reader

            sections.append("source: " + source_reader(target, (record.get("review") or {}).get("source_read")))
        if record.get("quote"):
            sections.append(f"quote: {record['quote']}")
    if revision:
        sections.append(f"as of revision {revision}: " + ("the original alone" if revision == record["id"]
                                                          else "the original and its corrections up to that one")
                        + (f"; later corrections not shown: {', '.join(later)}" if later else "; none later"))
    sections.append("text:\n" + str(record.get("text") if record.get("text") is not None else record.get("reason") or ""))
    for other in store.targeting(record["id"], ("correction", "decision", "mark_view", "mark_release", "selection")):
        if revision and other["kind"] == "correction" and other["id"] not in (fix["id"] for fix in shown.get("corrections") or ()):
            continue  # a later correction is named above, not shown as that version
        body = other.get("text") if other["kind"] == "correction" else (
            f"accepted={other.get('accepted')}; " if other["kind"] == "decision" else
            f"shown={other.get('shown')}; replaces {len(other.get('replaces') or [])}: "
            + (", ".join(other.get("replaces") or []) or "nothing") + "; " if other["kind"] == "selection"
            else "") + str(other.get("reason") or "")
        sections.append(f"{other['kind']} {other['id']} by {_author_label(other.get('author'))} (seq {other['sequence']}):\n"
                        f"{body}")
    for account in store.records(kinds=("account",)):  # the accounts this record informs, with the frozen version read
        for src in account.get("sources") or ():
            if src.get("id") == record["id"]:
                sections.append(f"informs account {account['id']} (revision {src.get('revision')} read; seq {account['sequence']}): "
                                f"memory_read(node_id={account['id']})")
    return shown, "\n".join(sections)


def _address_text(address: Any) -> str:
    try:
        return chat_chain.format_address(address) if isinstance(address, dict) else str(address or "?")
    except (KeyError, TypeError, ValueError):
        return "?"


def _read_node(root: Path, node_id: str, start: int, limit: int, revision: str = "") -> str:
    store = _existing_store(root)
    record = store.get(node_id) if store is not None else None
    if record is None:
        raise ValueError(f"memory node {node_id} not found")
    shown, document = _node_document(store, record, revision)
    from ouroboros.memory_sealing import terminal_links
    links = terminal_links(root, shown)
    if links:
        document += "\n\nHost-linked terminal facts (outside page coverage):\n" + "\n".join(links)
    header = _record_header(shown, _periods(store, root, [shown]).get(shown["id"]))
    at = f", revision={revision}" if revision else ""
    return _window(header, document, start, limit, lambda end: f"memory_read(node_id={node_id}{at}, start={end})")


def _read_source(root: Path, ref: Any, start: int, limit: int) -> str:
    from ouroboros.artifacts import read_actor_source_bytes  # D15->D05 is lazy-only

    if not isinstance(ref, dict):
        raise ValueError("source_ref is a retained source reference object")
    raw = read_actor_source_bytes(root, str(ref.get("task_id") or ""), ref)
    location = f"; location {ref['location']}" if ref.get("location") else ""
    head = (f"[source {ref.get('path')}; task {ref.get('task_id')}; {len(raw)} bytes; "
            f"sha256 {str(ref.get('sha256') or '')[:12]}{location}]")
    return _window(head, raw.decode("utf-8", errors="replace"), start, limit,
                   lambda end: f"memory_read(source_ref=<same>, start={end})")


def _read_rooms(root: Path, store: ChronicleStore, start: int, limit: int) -> str:
    from ouroboros import memory_view as mv

    spec = mv.ViewSpec("integrator", room_id=None, live_rooms="directory")
    with capture(root):  # one state of the chronicle for every room's facts
        gaps, quiet = {}, {}
        rows, lineage = memory_inventory.open_rows_by_room(root, gaps=gaps, quiet=quiet), row_lineage(root)
        rooms = mv._live_rooms(store, root, spec, mv.room_labeler(root), rows, mv._notes_by_room(store), lineage, gaps, quiet)
    body = "\n".join(mv._directory_room(room) for room in rooms)
    return _window(f"Open rooms: {len(rooms)}; live {sum(room['live'] for room in rooms)}", body, start, limit,
                   lambda n: f"memory_read(room_id='*', start={n})")


def _read_story(root: Path, store: ChronicleStore, start: int, limit: int) -> str:
    from ouroboros import memory_view as mv

    story, _status, _refusals = mv.capture_story(store, root, mv.room_labeler(root))
    entries = [entry for entry in story if entry.get('kind') != 'legacy' and not entry.get('told_by')]
    return _window(f"My story: {len(entries)} pages, parts and accounts", '\n'.join(map(mv._page_pointer, entries)),
                   start, limit, lambda n: f'memory_read(story=true, start={n})')


@completed_local_read
def _memory_read(ctx: Any, node_id: str = "", room_id: Any = None, after_seq: int = 0, rows: bool = False,
                 task_id: str = "", period: Any = None, source_ref: Any = None, start: int = 0, revision: str = "",
                 story: bool = False, **bounds: Any) -> str:
    unknown = sorted(set(bounds) - {"from", "to"})
    if unknown:
        return _arg_error(ctx, f"unknown argument(s): {', '.join(unknown)}")
    if sum(bool(mode) for mode in (node_id, source_ref, rows, story)) > 1 or story and room_id is not None:
        return _arg_error(ctx, "choose one mode: node_id, source_ref, rows=true, story=true, or a room's records")
    if revision and not node_id:
        return _arg_error(ctx, "revision reads one record (node_id) as of that revision")
    if type(after_seq) is not int or after_seq < 0 or type(start) is not int or start < 0:
        return _arg_error(ctx, "after_seq and start are non-negative integers")
    limit = tool_result_limit("memory_read")
    try:
        root = _root(ctx)
        store = None if source_ref else _activated(root)  # a retained source is verified on read, without the chronicle
        if source_ref:
            text = _read_source(root, source_ref, start, limit)
        elif story:
            text = _read_story(root, store, start, limit)
        elif str(room_id) == "*":
            text = _read_rooms(root, store, start, limit)
        elif str(room_id) in {"late_reviews", "notes"}:
            from ouroboros.memory_view import _notes_by_room
            from ouroboros.memory_waiting import listing

            with capture(root, tail="none"):
                head, body = listing(str(room_id), store.active_marks(None) if room_id == "late_reviews" else (),
                                     _notes_by_room(store) if room_id == "notes" else {})
            text = _window(head, body, start, limit, lambda n: f"memory_read(room_id='{room_id}', start={n})")
        elif node_id:
            text = _read_node(root, str(node_id), start, limit, str(revision or ""))
        elif rows:
            text = _read_rows(root, _room(ctx, root, room_id), {**bounds, "task_id": task_id, "period": period},
                              start, limit)
        else:
            text = _read_records(root, _room(ctx, root, room_id), after_seq, limit)
    except _NotActivated as exc:
        return _not_activated(ctx, exc)
    except chat_chain.RowAddressError as exc:
        return _arg_error(ctx, f"a row bound does not resolve: {exc.resolution.get('status')}")
    except ValueError as exc:
        return _arg_error(ctx, str(exc))
    except (OSError, TimeoutError) as exc:
        return _failure(ctx, exc)
    return _publish_tool_result(ctx, ToolResult(status="ok", code="OK", text=text))


# --- memory_mark ----------------------------------------------------------------------------------

def _source_texts(root: Path, ref: Dict[str, Any]) -> List[str]:
    """A retained source as text, plus every string inside it when it is JSON."""
    from ouroboros.artifacts import read_actor_source_bytes  # D15->D05 is lazy-only

    text = read_actor_source_bytes(root, str(ref.get("task_id") or ""), ref).decode("utf-8", errors="replace")
    texts = [text]
    try:
        stack = [json.loads(text)]
    except ValueError:
        return texts
    while stack:
        value = stack.pop()
        if isinstance(value, str):
            texts.append(value)
        elif isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    return texts


def _task_final_text(root: Path, task_id: str) -> str:
    """The task's last outgoing row in chat: its final words to people."""
    final = ""
    for _address, row, _pos in chat_chain.iter_rows(root):
        if str(row.get("task_id") or "") == task_id and str(row.get("direction") or "") in ("out", "outgoing"):
            final = render_row_text(row)
    return final


def _mark_target(ctx: Any, root: Path, store: ChronicleStore, a: Dict[str, Any]) -> Tuple[Any, Optional[str], str]:
    """``(target_ref, default room or None, refusal reason or "")``; the quote is checked here or by the store."""
    quote = a["quote"]
    if a["node_id"]:
        record = store.get(str(a["node_id"]))
        if record is None:
            return None, None, "target_missing"
        return {"kind": "chronicle", "id": record["id"]}, record["room_id"], ""
    if a["address"]:
        row, found = chat_chain.resolve_row(root, a["address"])
        if row is None:
            return None, None, str(found.get("status") or "row_missing")
        if quote and quote not in render_row_text(row):
            return None, None, "quote_mismatch"
        return found["address"], None, ""
    if a["task_id"]:
        if quote and quote not in _task_final_text(root, str(a["task_id"])):
            return None, None, "quote_mismatch"
        return {"kind": "task", "task_id": str(a["task_id"])}, None, ""
    ref = a["source_ref"]
    if not isinstance(ref, dict):
        raise ValueError("source_ref is a retained source reference object")
    if quote and not any(quote in text for text in _source_texts(root, ref)):
        return None, None, "quote_mismatch"
    return ref, None, ""


def _release_marks(ctx: Any, store: ChronicleStore, author: Dict[str, Any], ids: Any, reason: str) -> str:
    if (not isinstance(ids, list) or not ids or any(not isinstance(item, str) or not item.strip() for item in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("release_ids is a non-empty array of distinct mark ids")
    released, refused = [], []
    for mark in ids:
        try:
            result = store.release_mark(mark, author, reason)
        except (OSError, TimeoutError) as exc:
            refused.append({"id": mark, "reason": "unavailable", "detail": str(exc)})
            continue
        if result.ok:
            released.append(mark)
        else:
            refused.append({"id": mark, "reason": result.reason, "detail": result.detail})
    return _reply(ctx, {"ok": not refused, "operation": "mark_release", "released_ids": released, "refused": refused,
                        "reason": "saved" if not refused else "partial" if released else refused[0]["reason"]},
                  code="TOOL_REPORTED_FAILURE" if refused else "OK",
                  meta={"memory_mark_ids": released, "memory_mark_operation": "mark_release"})


def _memory_mark(ctx: Any, text: str = "", node_id: str = "", address: str = "", task_id: str = "",
                 source_ref: Any = None, quote: Optional[str] = None, room_id: Any = None, scope: str = "room",
                 mark_id: str = "", visibility: str = "", release_ids: Any = None, reason: str = "",
                 anchor: Any = None) -> str:
    try:
        root = _root(ctx)
        store, author = _activated(root), focus_signature(ctx)
        if release_ids is not None:
            if any((mark_id, visibility, node_id, address, task_id, source_ref, text, quote, anchor)):
                return _arg_error(ctx, "release_ids takes only one reason, without a mark target or view change")
            return _release_marks(ctx, store, author, release_ids, str(reason or ""))
        if mark_id:
            result = store.set_mark_view(str(mark_id), str(visibility or ""), author, str(reason or ""))
        else:
            if sum(bool(target) for target in (node_id, address, task_id, source_ref)) != 1:
                return _arg_error(ctx, "a mark names exactly one target: node_id, address, task_id or source_ref")
            if quote is not None and (not isinstance(quote, str) or not quote):
                return _arg_error(ctx, "quote is an exact non-empty substring of the target")
            if anchor is not None and (problem := anchor_problem(anchor)):
                return _arg_error(ctx, problem)
            target, room, refusal = _mark_target(ctx, root, store, {
                "node_id": node_id, "address": address, "task_id": task_id, "source_ref": source_ref, "quote": quote})
            if refusal:
                return _refused(ctx, refusal, "the target does not resolve" if refusal != "quote_mismatch"
                                else "the quote is not an exact substring of the target")
            if room_id is not None and str(room_id).strip() != "":
                room = str(room_id).strip()
            result = store.mark(target, str(text or ""), author, room_id=room or _room(ctx, root, None),
                                scope=str(scope or "room"), quote=quote, anchor=anchor)
    except _NotActivated as exc:
        return _not_activated(ctx, exc)
    except ValueError as exc:
        return _arg_error(ctx, str(exc))
    except (OSError, TimeoutError) as exc:
        return _failure(ctx, exc)
    if not result.ok:
        return _published(ctx, result)
    record = result.record or {}
    mark = str(record.get("target_id") or record.get("id") or "")
    return _reply(ctx, {"ok": True, "reason": result.reason, "operation": record.get("kind"), "mark_id": mark,
                        "room_id": record.get("room_id"), "scope": record.get("scope"),
                        "visibility": record.get("visibility"), "sequence": record.get("sequence")},
                  meta={"memory_mark_id": mark, "memory_mark_operation": str(record.get("kind") or "")})


# --- schemas ---------------------------------------------------------------------------------------

def chronicle_tools() -> List[ToolEntry]:
    string = {"type": "string"}
    room = {"type": "string", "description": "Room address: its chat id as text (1 is Main). Default: this task's own room."}
    address = {"type": "string", "description": "Chat row address row:<chat_id>@<ts>#<sha12>, as memory_read prints it."}
    source = {"type": "object", "description": "A retained source reference exactly as a record or reader returned it."}
    quotes = {"type": "array", "description": "Optional decisive words you cite. Each must be an exact substring of the row at its address, spoken by that class; the host checks and never inserts words itself.",
              "items": {"type": "object", "additionalProperties": False, "required": ["address", "text", "speaker"],
                        "properties": {"address": address, "text": string,
                                       "speaker": {"type": "string", "enum": sorted(SPEAKERS)}}}}
    write = {
        "kind": {"type": "string", "enum": ["page", "part", "note", "correction", "decision", "account", "selection"],
                 "description": "page seals one or more closed arcs of a room; part folds adjacent records of one lower level (pages, legacy sections or parts); note is a separate record for my future self; correction stands under the record's words wherever the record is shown, signed and dated — it adds and never replaces them, so to say a record anew I fold it into a part over it; decision accepts or rejects a helper's draft; account is my own text connecting experience of several rooms, based on the exact versions I read (sources) — it seals no row, folds nothing and leaves every source free to be folded or read anew; selection chooses the common view: show an account (shown) in place of the records it names (replaces), or withdraw it."},
        "room_id": room,
        "text": {"type": "string", "description": "The record in my own words. No length limit; it is read back through memory_read pages."},
        "covers": {"type": "object", "additionalProperties": False,
                   "description": "page: {from, to} row addresses (inclusive), {to} alone (this room's still-open rows up to it), {task_ids} (those tasks' rows and bound owner words), or {note_ids} (selected notes of this room, even without open rows). The host expands the exact coverage; it proves no reading. Already sealed members are refused. Pages never release marks.",
                   "properties": {"from": address, "to": address, "task_ids": {"type": "array", "items": string},
                                  "note_ids": {"type": "array", "items": string, "minItems": 1, "uniqueItems": True}}},
        "member_ids": {"type": "array", "items": string, "description": "part: adjacent unfolded records of one kind in one room; default room is theirs."},
        "quotes": quotes,
        "task_id": {"type": "string", "description": "note: the task it belongs to (default: this task)."},
        "target_id": {"type": "string", "description": "correction: the record corrected; decision: the helper's draft page or part; selection: the account."},
        "sources": {"type": "array", "description": "account: the records it is based on, each {id, revision?} (revision: the one I read, as memory_read shows it; default the current one) or a bare id. Any page, part, note, legacy record, gap or account, of any room; citing one neither seals nor folds it. To incorporate later source corrections, write a new account citing the corrected direct source revisions, then select it; correcting or reselecting an old account leaves its source versions unchanged, and citing it retains its older nested edges.",
                    "items": {"anyOf": [string, {"type": "object", "additionalProperties": False, "required": ["id"],
                                                 "properties": {"id": string, "revision": string}}]}},
        "replaces": {"type": "array", "items": string, "description": "selection: the records of my story the account tells in its place. Their count and period stay visible, their exact IDs remain in the account's memory_read composition and selections, and each room keeps its detail. To supersede an earlier account, include its ID here or withdraw it. Empty shows the account beside everything."},
        "shown": {"type": "boolean", "description": "selection: true (default) shows the account in the common view, false withdraws it; the latest selection of an account acts."},
        "expected_revision": {"type": "string", "description": "correction: the target's current revision when it already has a correction (memory_read node_id shows it)."},
        "expected_sequence": {"type": "integer", "minimum": 0, "description": "The room head your text is based on (memory_read room's first line). Required for a part; optional for a page or correction. A newer head is refused with the current head and the newer record ids."},
        "accepted": {"type": "boolean", "description": "decision: accept (true) or reject (false) the draft."},
        "reason": {"type": "string", "description": "decision or selection: why."},
    }
    read = {
        "story": {"type": "boolean", "description": "true: list every current page, part and selected account with its id and reader; continue with start."},
        "node_id": {"type": "string", "description": "One record: as it acts now, its host stamp, coverage, original text, corrections and decisions."},
        "room_id": {**room, "description": room["description"] + " '*' lists all open rooms (label, period, counts, reader); 'late_reviews' lists unreleased host review marks; 'notes' lists unsealed mind notes. Lists name ids, rooms, dates and exact readers, oldest first."},
        "after_seq": {"type": "integer", "minimum": 0, "description": "Room records and active marks after this sequence (the previous page's next_after_seq)."},
        "rows": {"type": "boolean", "description": "true: the room's chat rows verbatim, one header line each, filtered by from/to, task_id, period, in stored order (the order from/to and next: from= follow). A room holding imported earlier messages reads by send time in its view and chat_history; here each stands where it was stored."},
        "from": address, "to": address,
        "task_id": {"type": "string", "description": "rows: only rows of this task (its own, parent or root lineage)."},
        "period": {"type": "object", "additionalProperties": False, "description": "rows: inclusive ISO-8601 bounds.",
                   "properties": {"from": string, "to": string}},
        "source_ref": source,
        "start": {"type": "integer", "minimum": 0, "description": "Character position to continue a record, source or long row from (the previous page's next_start or start)."},
        "revision": {"type": "string", "description": "node_id: read the record as of this revision (its own id, or one of its correction ids, as an account's sources name them): the original and the corrections up to it; later ones are named, not shown."},
    }
    mark = {
        "text": {"type": "string", "description": "What matters, in my own words."},
        "node_id": {"type": "string", "description": "Target: a chronicle record."},
        "address": address,
        "task_id": {"type": "string", "description": "Target: a task (a quote is checked against its final words in chat)."},
        "source_ref": source,
        "quote": {"type": "string", "description": "Optional exact substring of the target, kept beside the mark."},
        "room_id": room,
        "scope": {"type": "string", "enum": ["room", "global"]},
        "mark_id": {"type": "string", "description": "Change an active mark's visible detail with visibility and reason."},
        "visibility": {"type": "string", "enum": ["full", "meaning"]},
        "release_ids": {"type": "array", "items": string, "minItems": 1, "uniqueItems": True,
                        "description": "Release these active marks with one reason; returns released ids and each refusal. Their history stays."},
        "reason": string,
        "anchor": {"type": "object", "additionalProperties": False,
                   "description": "Optional action this mark meets me at: {tool, text?} is a tool name, optionally with text its call's arguments must contain; {path} is a file or directory path a call names. The host adds the mark's words to the result of the first such call in each task (a room mark: that room's work only); it compares literally and never refuses the call.",
                   "properties": {"tool": string, "text": string, "path": string}},
    }

    def schema(name: str, description: str, properties: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
        return {"name": name, "description": description,
                "parameters": {"type": "object", "properties": properties, "required": required,
                               "additionalProperties": False}}

    return [
        ToolEntry("chronicle_write", schema(
            "chronicle_write",
            "Write my own chronicle record: seal a page over a room's closed arcs (the host expands covers into the exact row set, stamps each covered task with its recorded outcome and checks quotes), fold adjacent records into a part, keep a note for my future self, correct a record (a signed revision under its own words wherever it is shown; to say it anew, fold it into a part), accept/reject a helper's draft, write an account connecting experience of several rooms over the exact source versions I read (sources; nothing sealed or folded), or select an account for the common view in place of the records it names (replaces; the latest selection acts, shown=false withdraws). A delegated child or nanny publishes pages and parts only as drafts signed in its own name for the integrating mind to accept or reject; its part folds only legacy sections and its own drafts (its note, correction, decision, account or selection, or a part over other records, is refused: not_integrator). Records are never rewritten. A refusal returns the current revision or room head and the conflicting ids; read them with memory_read.",
            write, ["kind"]), _chronicle_write),
        ToolEntry("memory_read", schema(
            "memory_read",
            "Read memory by address, as text: a room's records and active marks (first line: room and head), one record with its stamp and corrections (node_id; an account also names each source's version, what changed in it since, and the accounts a record informs; revision= reads a record as of one revision), a room's chat rows verbatim (rows=true with from/to, task_id or period), or a retained source (source_ref). Every page fits one tool result and its second line names the continuation (next_after_seq, next: from=…, or next_start), older to newer; chat rows in stored order. Choose the depth yourself.",
            read, []), _memory_read),
        ToolEntry("memory_mark", schema(
            "memory_mark",
            "Mark what matters in my own words on one target: a chronicle record (node_id), a chat row (address), a task (task_id) or a retained source (source_ref); room or global scope; an optional quote is verified as an exact substring of the target; an optional anchor ties the mark to an action, so its words also meet me in that action's result. Change a mark's visible detail (mark_id, visibility full/meaning, reason) or release marks (release_ids, one reason) when they no longer hold.",
            mark, []), _memory_mark),
    ]
