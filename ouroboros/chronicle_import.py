"""Import the legacy dialogue memory into the chronicle once, without a model call.

``ensure_activated(store)`` turns the old dialogue writer's ``memory/dialogue_blocks.json``,
``dialogue_meta.json`` and ``dialogue_summary.md`` into chronicle records published in
ONE transaction with the activation receipt and the ``legacy_frontier`` scan state.
Every readable source is retained byte for byte before any is interpreted, and the
files themselves are never written, so a corrupt one costs nothing but its own gap.

Each room section of a block becomes one ``legacy`` record (``legacy-b<NN>-r<room>``,
written by ``legacy_helper``: Light's retelling, not lived). An era stays one record:
its nested retelling is only addressed (``legacy_source_ref``), never unfolded. A
block's ``raw_range`` is its half-open slice of the chat stream by position (the
legacy cursor's stream: JSON rows outside A2A in chain order), ``exact`` only when the
blocks' message counts sum to the old cursor's stream position and no block is a gap;
otherwise ``unknown``. A deleted early archive would shift every position and break that
sum, so it surfaces as ``unknown`` too. An unreadable or unresolvable cursor makes the
frontier ``unknown`` at the chain end observed now, beside a ``cursor_gap`` record.
Pending knowledge nominations become global marks the mind releases. The receipt
records the first row that carries delegation lineage (``lineage_epoch``) and the
chain end, both data facts that later attribution and open-row readers use. When no
row carries lineage yet, the epoch is the chain end: nothing proves that the rows
already written came from a version that recorded a child's lineage, so their
outgoing rows are attributed by task results, never assumed to be my own words.

Once an activation exists the import is a no-op: the legacy files are not read again,
so a later change to them writes neither a record nor a copy. A busy
``.consolidation.lock`` (another importer, or the old writer of a process still
running an earlier version during an update) answers ``import_pending``; nothing
is lost and the next caller imports. Each memory tool activates first and waits
for a busy lock (``wait=True``), so it never reads or writes before the import.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set, Tuple

from ouroboros.chronicle_store import ChronicleStore, source_time_span
from ouroboros.platform_layer import file_lock_exclusive, file_lock_exclusive_nb, file_unlock
from ouroboros.utils import assert_test_data_path

IMPORT_TASK_ID = "chronicle-import"
LEGACY_AUTHOR = {"kind": "legacy_helper", "writer": "old_consolidator",
                 "attribution": "retelling by Light, not lived"}
_SOURCE_GAP_AUTHOR = {"kind": "host", "attribution": "source-gap fact"}
_COVERAGE_GAP_AUTHOR = {"kind": "host", "attribution": "coverage-gap fact"}
_SOURCES = (("blocks", "dialogue_blocks.json", "json"), ("meta", "dialogue_meta.json", "json"),
            ("flat", "dialogue_summary.md", "md"))
_FILES = {name: filename for name, filename, _extension in _SOURCES}
_GAP_MARK = "[MEMORY GAP]"
_NOMINATIONS = "pending_knowledge_nominations"
_LAST_UNPUBLISHED = "last_unpublished_nominations"
_UNKNOWN_RANGE = {"status": "unknown", "pos": None, "first": None, "last": None, "ts_span": None}
# A legacy record without typed room sections is one section of explicitly unknown provenance.
LEGACY_ROOM_ID = "legacy"
LEGACY_ROOM_LABEL = "Unknown provenance [legacy mixed record]"

Kept = Dict[int, Tuple[str, int, Dict[str, Any]]]


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def legacy_frontier(store: Any) -> Dict[str, Any]:
    """The first chat-stream position the legacy memory does not represent: where open rows begin.

    ``{"status": "exact"|"unknown", "pos", "last_covered", "chat_log_signature", "offset"}``;
    under ``unknown`` the position is the chain end observed at activation. Empty before
    activation.
    """
    return dict(store.scan_state().get("legacy_frontier") or {})


def row_lineage(root: Any) -> Dict[str, Any]:
    """``row_author`` keywords from the activation receipt: its lineage epoch (a data fact) and
    a strict task-result lookup. Empty before activation, and where the chain was empty then;
    a reader never creates the chronicle to ask.
    """
    from ouroboros.dialogue_provenance import task_lineage_lookup

    store = ChronicleStore(root)
    receipt = (store.activation() if store.log_path.exists() else None) or {}
    epoch = (receipt.get("metadata") or {}).get("lineage_epoch")
    if not isinstance(epoch, dict) or type(epoch.get("pos")) is not int:
        return {}
    return {"lineage_epoch": epoch, "lineage_lookup": task_lineage_lookup(root)}


def ensure_activated(store: Any, *, wait: bool = False) -> Dict[str, Any]:
    """The activation receipt, importing the legacy memory first when there is none (the import runs once).

    Returns the receipt, ``{"kind": "import_pending", ...}`` while the legacy lock is
    held elsewhere, or ``{"kind": "import_refused", ...}`` with the store's typed refusal.
    ``wait=True`` blocks on a busy lock until the other importer finishes, then finds
    its receipt (or imports when it left none); ``import_pending`` then means only that
    the wait itself failed. A caller that must not act before the import uses it.
    """
    active = store.activation()
    if active:
        return active
    lock_path = pathlib.Path(store.data_root) / "memory" / ".consolidation.lock"
    assert_test_data_path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        try:
            (file_lock_exclusive if wait else file_lock_exclusive_nb)(fd)
        except OSError:
            return {"kind": "import_pending", "reason": "legacy_memory_lock_busy"}
        try:
            return store.activation() or _import(store)
        finally:
            file_unlock(fd)
    finally:
        os.close(fd)


def _import(store: Any) -> Dict[str, Any]:
    from ouroboros import chat_chain

    root = pathlib.Path(store.data_root)
    context = SimpleNamespace(drive_root=root, task_id=IMPORT_TASK_ID)
    raw: Dict[str, bytes] = {}
    refs: Dict[str, Dict[str, Any]] = {}
    errors: Dict[str, str] = {}
    # Retain every readable source BEFORE interpreting any: a corrupt one cannot cost another its copy.
    for name, filename, extension in _SOURCES:
        try:
            raw[name] = (root / "memory" / filename).read_bytes()
        except FileNotFoundError:
            continue
        except OSError as exc:
            errors[name] = f"source unreadable: {type(exc).__name__}"
            continue
        refs[name] = chat_chain.retain_memory_source(context, "legacy_" + name, raw[name], extension)
    blocks = _blocks(raw, errors)
    meta = _cursor(raw, errors)
    flat = _flat(raw, errors)

    parsed: List[Tuple[int, Dict[str, Any], List[Dict[str, Any]]]] = []
    records: List[Dict[str, Any]] = []
    gapped = bool(errors.get("blocks"))
    for n, block in enumerate(blocks):
        try:
            if not isinstance(block, dict):
                raise ValueError("legacy block is not an object")
            parsed.append((n, block, room_sections(block)))
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            gapped = True
            records.append(_gap(refs, "blocks", str(exc), str(n), LEGACY_ROOM_ID))
    gapped = gapped or any(_is_gap(block, section) for _n, block, sections in parsed for section in sections)

    sizes = [block.get("message_count") if isinstance(block, dict) else None for block in blocks]
    sized = all(type(size) is int and size >= 0 for size in sizes)
    bounds = _bounds(sizes) if sized else []
    wanted = {pos for start, stop in bounds if stop > start for pos in (start, stop - 1)}
    cursor_start, offset, reason = _cursor_start(root, meta, bool(blocks), errors.get("meta"))
    stamps, kept, epoch, cursor = _survey(root, wanted, cursor_start, offset)
    total = len(stamps)
    if reason is None and cursor > total:
        reason = "the old cursor points past the chat rows that exist"

    def address(pos: int) -> Optional[Dict[str, Any]]:
        if pos not in kept:
            return None
        gen, line, row = kept[pos]
        return chat_chain.row_address(row, gen=gen, line=line)

    exact = reason is None and sized and not gapped and sum(sizes) == cursor
    ranges = [{"status": "exact", "pos": [start, stop], "first": address(start) if stop > start else None,
               "last": address(stop - 1) if stop > start else None,
               "ts_span": source_time_span(stamps[start:stop])} for start, stop in bounds] if exact else []
    seen: Set[str] = set()
    texts: Set[str] = set()
    for n, block, sections in parsed:
        block_sha = _digest(block)
        for index, section in enumerate(sections):
            record = _section(refs, n, block, block_sha, section, ranges[n] if exact else _UNKNOWN_RANGE)
            if record["id"] in seen:  # one room twice in a block keeps both sections
                record["id"] += f"-{index}"
            seen.add(record["id"])
            texts.add(section["content"])
            records.append(record)
    contents = {str(block.get("content", "")) for _n, block, _s in parsed}
    if flat.strip() and flat not in texts and flat not in contents:
        records.append({"id": "legacy-flat-" + hashlib.sha256(raw["flat"]).hexdigest()[:16], "kind": "legacy",
                        "room_id": LEGACY_ROOM_ID, "text": flat, "author": LEGACY_AUTHOR,
                        "source_refs": [refs["flat"]],
                        "covers": {"room_id": LEGACY_ROOM_ID, "raw_range": _UNKNOWN_RANGE},
                        "metadata": {"legacy_type": "flat", "label": LEGACY_ROOM_LABEL}})
    for name in ("blocks", "flat"):  # an unreadable cursor is the cursor_gap below
        if name in errors:
            records.append(_gap(refs, name, errors[name], "", LEGACY_ROOM_ID))

    chain_end = {"address": address(total - 1), "pos": total - 1} if total else {"address": None, "pos": None}
    signature = meta.get("chat_log_signature") if meta else None
    if reason is None:
        frontier = {"status": "exact", "pos": cursor, "last_covered": address(cursor - 1) if cursor else None,
                    "chat_log_signature": signature, "offset": meta.get("last_consolidated_offset", 0)}
    else:
        frontier = {"status": "unknown", "pos": total, "last_covered": chain_end["address"],
                    "chat_log_signature": signature,
                    "offset": meta.get("last_consolidated_offset") if meta else None, "reason": reason}
        records.append(_cursor_gap(refs, reason, chain_end, LEGACY_ROOM_ID))
    records.extend(_nomination_marks(refs, meta or {}, LEGACY_ROOM_ID))
    receipt = {"id": "legacy-import-" + _digest({name: ref.get("sha256") for name, ref in refs.items()})[:16],
               "kind": "activation", "room_id": "", "author": {"kind": "host", "operation": "legacy_import"},
               "metadata": {"source_refs": refs, "imported_records": len(records), "paid_calls": 0,
                            "legacy_files_unchanged": True, "lineage_epoch": epoch, "chain_end": chain_end}}
    # The records, their receipt and the frontier become visible together; ids are deterministic.
    result = store.publish([*records, receipt], scan_state={"legacy_frontier": frontier})
    if not result.ok:
        return {"kind": "import_refused", "reason": result.reason, "detail": result.detail,
                "conflict_ids": list(result.conflict_ids)}
    return store.activation()


# --- sources ---------------------------------------------------------------------------------------

def _blocks(raw: Dict[str, bytes], errors: Dict[str, str]) -> List[Any]:
    if "blocks" not in raw:
        return []
    try:
        value = json.loads(raw["blocks"])
        if not isinstance(value, list):
            raise ValueError("legacy blocks are not a list")
        return value
    except ValueError as exc:  # UnicodeDecodeError included
        errors["blocks"] = str(exc)
        return []


def _cursor(raw: Dict[str, bytes], errors: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """The old cursor; ``{}`` when the file is absent, ``None`` when it cannot be read."""
    if "meta" in errors:
        return None
    if "meta" not in raw:
        return {}
    from ouroboros.memory_nomination_receipts import parse_meta

    try:
        meta = parse_meta(raw["meta"])  # strict: duplicate keys and malformed nominations are refused
        offset = meta.get("last_consolidated_offset", 0)
        if type(offset) is not int or offset < 0:
            raise ValueError("legacy cursor offset is invalid")
        if not isinstance(meta.get("chat_log_signature", {}), dict):
            raise ValueError("legacy cursor generation is invalid")
    except ValueError as exc:
        errors["meta"] = str(exc)
        return None
    return meta


def _flat(raw: Dict[str, bytes], errors: Dict[str, str]) -> str:
    try:
        return raw.get("flat", b"").decode("utf-8")
    except UnicodeDecodeError as exc:
        errors["flat"] = str(exc)
        return ""


def room_sections(block: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Typed room sections of a stored legacy block; a block without them is one legacy mixed section.

    The old writer stamped ``rooms`` on every block it summarized room by room; an older
    block or an era keeps its complete content as one section of unknown provenance,
    never a guessed label.
    """
    rooms = block.get("rooms")
    if isinstance(rooms, list) and rooms and all(
        isinstance(room, dict) and isinstance(room.get("room_id"), str) and isinstance(room.get("content"), str)
        for room in rooms
    ):
        return [{"room_id": room["room_id"], "label": str(room.get("label") or room["room_id"]),
                 "message_count": int(room.get("message_count") or 0), "content": room["content"]}
                for room in rooms]
    return [{"room_id": LEGACY_ROOM_ID, "label": LEGACY_ROOM_LABEL,
             "message_count": int(block.get("message_count") or 0), "content": str(block.get("content") or "")}]


def _is_gap(block: Dict[str, Any], section: Dict[str, Any]) -> bool:
    """A gap by the old writer's typed facts (``gap_id``, ``type == "gap"``) or by content that
    BEGINS with its marker; a retelling that only mentions the marker is not a gap."""
    return bool(block.get("gap_id") or block.get("type") == "gap"
                or str(block.get("content") or "").lstrip().startswith(_GAP_MARK)
                or section["content"].lstrip().startswith(_GAP_MARK))


def _bounds(sizes: List[int]) -> List[Tuple[int, int]]:
    bounds, start = [], 0
    for size in sizes:
        bounds.append((start, start + size))
        start += size
    return bounds


# --- the chat stream -------------------------------------------------------------------------------

def _cursor_start(root: pathlib.Path, meta: Optional[Dict[str, Any]], has_blocks: bool,
                  error: Optional[str]) -> Tuple[Optional[pathlib.Path], int, Optional[str]]:
    """The old cursor's generation and offset, as the old writer resolved them, or why not."""
    from ouroboros.chat_chain import _resolve_generation_segments

    if meta is None:
        return None, 0, f"the old cursor is unreadable ({error})"
    if not meta and has_blocks:
        return None, 0, "the old cursor file is missing while legacy blocks exist"
    segments, offset, missing = _resolve_generation_segments(meta, root / "logs" / "chat.jsonl")
    if missing:
        return None, 0, "the old cursor's chat generation no longer exists"
    return segments[0], offset, None


def _survey(root: pathlib.Path, wanted: Set[int], start: Optional[pathlib.Path],
            offset: int) -> Tuple[List[Any], Kept, Optional[Dict[str, Any]], int]:
    """One pass over the chat stream: ``ts`` by position, the rows at ``wanted`` positions (plus
    the last row covered by the old cursor and the chain's last row), the lineage epoch (the
    first row carrying delegation lineage, else the chain end when rows exist), and the
    cursor's stream position (rows before its generation + offset).
    """
    from ouroboros.chat_chain import _stream_rows, chat_chain_paths, generation_signatures, row_address
    from ouroboros.dialogue_provenance import _LINEAGE_FIELDS  # the fields row_author reads

    order = {path: index for index, path in enumerate(chat_chain_paths(root))}
    wanted = set(wanted)
    stamps: List[Any] = []
    kept: Kept = {}
    epoch: Optional[Dict[str, Any]] = None
    cursor = -1
    previous = None
    for _index, path, gen, line, row, pos in _stream_rows(generation_signatures(root)):
        if cursor < 0 and start is not None and order[path] >= order[start]:
            cursor = pos + offset
            if offset:
                wanted.add(cursor - 1)
            elif previous is not None:
                kept[pos - 1] = previous
        here = (gen, line, row)
        if pos in wanted:
            kept[pos] = here
        if epoch is None and any(str(row.get(key) or "").strip() for key in _LINEAGE_FIELDS):
            epoch = {"pos": pos, "ts": str(row.get("ts") or ""), "address": row_address(row, gen=gen, line=line),
                     "basis": "first_lineage_row"}
        stamps.append(row.get("ts"))
        previous = here
    if previous is not None:
        kept[len(stamps) - 1] = previous
    if epoch is None and stamps:  # no row proves lineage was recorded: every row so far precedes the epoch
        epoch = {"pos": len(stamps), "ts": None, "address": None, "basis": "chain_end"}
    if cursor < 0:  # the cursor's generation holds no row yet
        cursor = len(stamps) + offset
    return stamps, kept, epoch, cursor


# --- records ---------------------------------------------------------------------------------------

def _section(refs: Dict[str, Dict[str, Any]], n: int, block: Dict[str, Any], block_sha: str,
             section: Dict[str, Any], raw_range: Dict[str, Any]) -> Dict[str, Any]:
    room = section["room_id"]
    return {"id": f"legacy-b{n:02d}-r{room}", "kind": "legacy", "room_id": room, "text": section["content"],
            "author": LEGACY_AUTHOR, "source_refs": [{**refs["blocks"], "location": str(n)}],
            "covers": {"room_id": room, "raw_range": raw_range},
            "metadata": {"legacy_type": "gap" if _is_gap(block, section) else str(block.get("type") or "summary"),
                         "legacy_block": n, "label": section["label"], "legacy_range_text": block.get("range"),
                         "legacy_message_count": block.get("message_count"),
                         "room_message_count": section["message_count"], "legacy_written_at": block.get("ts"),
                         "legacy_sha": block_sha, "legacy_source_ref": block.get("source_ref"),
                         "legacy_gap_id": block.get("gap_id") or ""}}


def _gap(refs: Dict[str, Dict[str, Any]], name: str, reason: str, location: str, room: str) -> Dict[str, Any]:
    text = (f"[MEMORY GAP] Legacy {name} cannot establish complete memory at {location or 'this source'}: "
            f"{reason}. Available evidence is retained; this gap is not a claim that the missing history "
            "was summarized.")
    ref = refs.get(name)
    return {"id": "legacy-gap-" + _digest([name, reason, (ref or {}).get("sha256"), location])[:16], "kind": "legacy",
            "room_id": room, "text": text, "author": _SOURCE_GAP_AUTHOR,
            "source_refs": [{**ref, "location": location}] if ref else [],
            "covers": {"room_id": room, "raw_range": _UNKNOWN_RANGE},
            "metadata": {"legacy_type": "gap", "coverage": "unknown", "source_gap": reason,
                         "legacy_path": f"memory/{_FILES[name]}", "location": location}}


def _cursor_gap(refs: Dict[str, Dict[str, Any]], reason: str, chain_end: Dict[str, Any], room: str) -> Dict[str, Any]:
    text = (f"[MEMORY GAP] {reason[0].upper()}{reason[1:]}, so which chat rows the legacy retelling covers is "
            "unknown, and any knowledge nominations it held are known only from its retained copy. The "
            "retelling stays readable beside this gap; rows after the chain end observed at activation are "
            "open. Earlier rows are read from logs/chat.jsonl and archive/chat_*.jsonl, never rebuilt by a model.")
    meta_ref = refs.get("meta")
    return {"id": "legacy-cursor-gap-" + _digest([(meta_ref or {}).get("sha256"), reason])[:16], "kind": "legacy",
            "room_id": room, "text": text, "author": _COVERAGE_GAP_AUTHOR,
            "source_refs": [meta_ref] if meta_ref else [],
            "covers": {"room_id": room, "raw_range": _UNKNOWN_RANGE},
            "metadata": {"legacy_type": "cursor_gap", "coverage": "unknown", "source_gap": reason,
                         "chain_end": chain_end}}


def _nomination_marks(refs: Dict[str, Dict[str, Any]], meta: Dict[str, Any], room: str) -> List[Dict[str, Any]]:
    """Each pending nomination, and an unpublished last batch, as one global mark: none is lost in the import."""
    marks = []

    def mark(key: str, location: str, text: str) -> None:
        marks.append({"id": "legacy-nomination-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16],
                      "kind": "mark", "room_id": room, "target_ref": {**refs["meta"], "location": location},
                      "text": text, "author": LEGACY_AUTHOR, "scope": "global", "quote": None, "visibility": "full"})

    for index, entry in enumerate(meta.get(_NOMINATIONS) or []):
        mark(entry["id"], f"{_NOMINATIONS}/{index}",
             f"Pending knowledge nomination the old dialogue writer never published: topic "
             f"'{entry.get('topic', '')}', scope '{entry.get('scope', '')}', reason '{entry.get('reason', '')}'. "
             f"Its full proposal is entry {entry['id'].split(':', 1)[0]} of memory/knowledge_history.jsonl; "
             "release this mark once it is handled or dropped.")
    last = meta.get(_LAST_UNPUBLISHED)
    if isinstance(last, dict) and type(last.get("failed")) is int and last["failed"] > 0:
        mark(_LAST_UNPUBLISHED, _LAST_UNPUBLISHED,
             f"The old dialogue writer's last knowledge-nomination publication left {last['failed']} of "
             f"{last.get('total')} unpublished (entry {last.get('entry_id', '')} of memory/knowledge_history.jsonl); "
             "release this mark once it is handled or dropped.")
    return marks
