"""A delegated writer's coverage follows what it was shown whole or read (``memory_reading``'s record).

``bound_delegated`` keeps, of a delegated writer's covers, the rows (each in its own room) and notes it was shown
whole or read and none of running work; the fallback writer passes the rows it gave Light whole through the same
call. ``foldable`` bounds a delegated part's members the same way, and ``restrict`` rebuilds covers over exactly
what is kept.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ouroboros.memory_reading import inside, row_keys


NOTES_LEFT = "notes not shown to this writer whole or read, or of running work"


class NotShown(ValueError):
    """A delegated writer's covers hold no row or note it was shown whole or read: ``left`` names what stayed open."""

    def __init__(self, left: Mapping[str, Any]) -> None:
        super().__init__("covers name no row or note this writer was shown whole or read")
        self.left = dict(left)


def _shown_row(record: Mapping[str, Any], room: str, entry: Tuple[Any, Mapping[str, Any], int], lane: int) -> bool:
    """A row the writer was shown whole or read: read exactly, my human's row by digest, or (a spoken row) inside a
    span its view drew whole or named by an event or thread message it was given. A task row stands in a view only as
    its task's line: it counts when read."""
    entry_room = (record.get("rooms") or {}).get(room) or {}
    if inside(entry_room.get("read") or (), entry[2]) or str(entry[0].get("row_sha256")) in (entry_room.get("shas") or ()):
        return True
    if lane != 1:
        return False
    if inside(entry_room.get("spans") or (), entry[2]):
        return True
    event, keys = row_keys(entry[1])
    return bool(event) and event in (record.get("events") or ()) or bool(set(keys) & set(record.get("messages") or ()))


def bound_delegated(root: Any, record: Optional[Mapping[str, Any]], room: Any, covers: Dict[str, Any],
                    rows: List[Tuple[Dict[str, Any], Dict[str, Any], int]], lineage: Optional[Mapping[str, Any]] = None
                    ) -> Tuple[Dict[str, Any], List[Tuple[Any, Any, int]], Dict[str, Any]]:
    """``(covers, rows, left)``: of a writer's covers, the rows (each in its own room) and notes it was shown whole or
    read (``record``; None: the fallback writer, every row it gave was whole and it seals no note), never a row of
    running work (by the row's task, or by a running task a ``task_ids`` selector named). ``left`` names how many
    stayed open, why, and their first and last address. Nothing left raises ``NotShown``."""
    from ouroboros import chat_chain
    from ouroboros.dialogue_provenance import row_class
    from ouroboros.task_store import task_facts

    from ouroboros.chronicle_store import ChronicleStore

    rooms = {sha: str(each) for each, refs in (covers.get("rooms") or {}).items() for sha in refs}
    notes = [ref for ref in covers.get("rows", ()) if str(ref).startswith("note:")]
    busy = set(task_facts(root, where="status=?", params=("running",))) if rows or notes else set()
    named = set((covers.get("request") or {}).get("task_ids") or ()) & busy
    kept, unread, live = [], [], []
    for entry in rows:
        tasks = {str(entry[1].get(key) or "") for key in ("task_id", "root_task_id")} - {""}
        if tasks & busy or not tasks and named:
            live.append(entry)
        elif record is None or _shown_row(record, rooms.get(entry[0]["row_sha256"], str(room)), entry,
                                          row_class(entry[1], pos=entry[2], **(lineage or {}))["lane"]):
            kept.append(entry)
        else:
            unread.append(entry)
    store = ChronicleStore(root) if notes else None
    note_tasks = {ref: str((store.get(ref[5:]) or {}).get("task_id") or "") for ref in notes}
    held = [ref for ref in notes if record is not None and ref[5:] in (record.get("nodes") or ())
            and note_tasks[ref] not in busy]
    left = {reason: {"count": len(items), "from": chat_chain.format_address(items[0][0]),
                     "to": chat_chain.format_address(items[-1][0])}
            for reason, items in (("not shown to this writer whole or read", unread), ("running work", live)) if items}
    if len(held) < len(notes) and record is not None:
        left[NOTES_LEFT] = [r[5:] for r in notes if r not in held]
    if not kept and not held:
        raise NotShown(left)
    return restrict(covers, kept, notes=held, note_tasks=[note_tasks[ref] for ref in held]), kept, left


def foldable(store: Any, author: Mapping[str, Any], members: Iterable[Any], record: Mapping[str, Any]
             ) -> Tuple[List[str], List[str]]:
    """``(foreign, unshown)`` part members of a delegated writer: foreign ones are neither a legacy section nor its own
    draft (the mind's and other writers' records are the integrating mind's to fold); unshown ones are legacy
    sections it was neither shown whole nor read. Its own drafts count as held. A missing id is left to the store."""
    nodes, foreign, unshown = set(record.get("nodes") or ()), [], []
    for member in (str(m) for m in members):
        found = store.get(member)
        signer = (found or {}).get("author") if isinstance((found or {}).get("author"), dict) else {}
        own = signer.get("kind") == "helper" and isinstance(signer.get("focus"), dict) and (
            str(signer.get("task_id") or "") == str(author.get("task_id") or ""))
        if found is not None and found.get("kind") != "legacy" and not own:
            foreign.append(member)
        elif found is not None and not own and member not in nodes:
            unshown.append(member)
    return foreign, unshown


def restrict(covers: Dict[str, Any], kept: List[Tuple[Dict[str, Any], Dict[str, Any], int]], *,
             notes: Sequence[str] = (), note_tasks: Sequence[str] = ()) -> Dict[str, Any]:
    """Covers over exactly ``kept`` (``(address, row, pos)``, stored order) and the named notes (with their tasks):
    the rows, their bounds, tasks, times and stream span; a conversation's room map keeps only what is kept."""
    from ouroboros.chronicle_store import source_time_span

    shas = [entry[0]["row_sha256"] for entry in kept]
    out = {**covers, "rows": shas + list(notes), "note_ids": [str(n)[5:] for n in notes], "count": len(kept),
           "first": kept[0][0] if kept else None, "last": kept[-1][0] if kept else None,
           "task_ids": list(dict.fromkeys([*(str(e[1].get("task_id")) for e in kept if e[1].get("task_id")),
                                           *(task for task in note_tasks if task)]))}
    if kept:  # a notes-only page keeps the stream and time spans its notes were named with
        out.update(ts_span=source_time_span(e[1].get("ts") for e in kept), stream_span=[kept[0][2], kept[-1][2]])
    if isinstance(covers.get("rooms"), dict):
        held = set(shas)
        out["rooms"] = {room: [ref for ref in refs if ref in held or ref in notes]
                        for room, refs in covers["rooms"].items()}
        out["rooms"] = {room: refs for room, refs in out["rooms"].items() if refs}
    return out
