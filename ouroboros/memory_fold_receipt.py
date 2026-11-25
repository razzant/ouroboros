"""What a fold returns to its writer (the gate's telling by kind at a start: ``context_view_facts.start_facts``).

A fold's receipt prints only what the committed record holds or a typed fact the store returned: the outcome, what
the record covers, its own words beside its members' or sources' own words, and where the effect is read. It does
not describe the writer's view. It reads chronicle records by id, one lookup each; it stores nothing and calls no
model. Measures only: no threshold, no verdict, no advice.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from ouroboros import memory_floor as floor, memory_view as mv

CLOSE = ("Originals remain readable by address; this task's view is unchanged by it; memory_read(story=true) shows my "
         "story as it stands now; a start whose gate prices a shortest telling shows it in its facts.")


def fold_receipt(saved: Optional[Mapping[str, Any]], build: Callable[[], List[str]]) -> str:
    """The receipt of a saved write: ``build``'s lines (every read the receipt needs happens inside it), then where the
    effect is read. A receipt that cannot be read back still answers saved, with the record's id."""
    try:
        lines = build()
    except Exception as exc:  # the write is saved: its receipt says so, never an error
        saved = saved or {}
        lines = [f"saved: {saved.get('kind') or 'record'} {saved.get('id')}; the rest of this receipt could not be read "
                 f"back ({type(exc).__name__})"]
    return "\n".join([*lines, CLOSE])


def _n(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def by_room(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{room} ({n})" for room, n in sorted(counts.items())) or "none"


def _read(store: Any, refs: Iterable[Tuple[str, Optional[str]]]) -> Dict[str, Tuple[Optional[str], Optional[int]]]:
    """Each record's room and own words in tokens (at the cited revision where one is named), read by id, one lookup
    each (never a capture of the whole index); ``(None, None)`` for a record the lookup does not find."""
    out: Dict[str, Tuple[Optional[str], Optional[int]]] = {}
    for ident, revision in refs:
        record = store.get(ident)
        out[ident] = ((str(record.get("room_id")), floor.view_tokens(
            store.interpret([record], revision=revision, sources=False)[0]["current_text"])) if record else (None, None))
    return out


def _own_size(record: Mapping[str, Any], noun: str = "", read: Optional[Mapping[str, Tuple[Any, Optional[int]]]] = None) -> str:
    """The record's own words in tokens (as corrected, where corrections changed its text), its lead sentence alone
    where its writer was shown the rule, and its members' or sources' own words ("size not read" for a failed lookup)."""
    text = str(record.get("current_text") or record.get("text") or "")
    meta = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    parts = [f"the {record.get('kind') or 'record'}'s own words: {floor.view_tokens(text):,} tokens"
             + (" (its text as corrected)" if record.get("corrections") else "")]
    if "lead_rule" in meta and text.strip():
        parts.append(f"its lead sentence alone {floor.view_tokens(mv.first_sentence(text)):,} tokens")
    if read is not None:
        sizes = [tokens for _room, tokens in read.values()]
        missing = sum(tokens is None for tokens in sizes)
        whose = _n(len(sizes), noun) + ("'s" if len(sizes) == 1 else "'")
        parts.append(f"its {whose} own words: " + (
            f"{sum(t for t in sizes if t is not None):,} tokens" if len(sizes) > missing else "size not read")
            + (f", {missing} size not read" if missing and len(sizes) > missing else ""))
    return "; ".join(parts)


def _row_words(row: Mapping[str, Any]) -> Optional[int]:
    """A read row's own words in tokens: its text, else its recorded size; None where the row holds neither."""
    if isinstance(row.get("text"), str):
        return floor.view_tokens(row["text"])
    return floor.tokens_of(int(row["text_chars"])) if isinstance(row.get("text_chars"), int) else None


def page_lines(record: Mapping[str, Any], rows: Iterable[Any]) -> List[str]:
    """A page: the rows and notes its committed covers hold, by room; the distinct works among the rows read (by
    ``root_task_id or task_id``); its own size beside its rows' own words, from the ``(address, row, position)`` the
    write read (no read of its own; a notes-only page has none)."""
    covers, room, rows = record["covers"], str(record["room_id"]), list(rows)
    room_of = {str(sha): str(r) for r, shas in (covers.get("rooms") or {}).items() for sha in shas}
    shas = [str(sha) for sha in covers["rows"] if not str(sha).startswith("note:")]
    works = {str(row.get("root_task_id") or row.get("task_id")) for _a, row, _p in rows
             if row.get("root_task_id") or row.get("task_id")}
    read = {str(a["row_sha256"]): (None, _row_words(row)) for a, row, _p in rows} if shas else None
    return [f"this page covers rows by room: {by_room(Counter(room_of.get(sha, room) for sha in shas))}; "
            f"{_n(len(covers.get('note_ids') or ()), 'note')}; {_n(len(works), 'work')} among its rows",
            _own_size(record, "row", read)]


def part_lines(store: Any, record: Mapping[str, Any], action: str = "this part folds") -> List[str]:
    """A part: its committed members by their rooms, its size beside theirs."""
    read = _read(store, [(str(m), None) for m in record["covers"]["member_ids"]])
    rooms = Counter(room or "room not read" for room, _t in read.values())
    return [f"{action} {_n(len(read), 'record')} by room: {by_room(rooms)}", _own_size(record, "member", read)]


def account_lines(store: Any, record: Mapping[str, Any]) -> List[str]:
    """An account: its frozen sources by their rooms, its size beside theirs."""
    sources = record.get("sources") or ()
    read = _read(store, [(str(src["id"]), src.get("revision")) for src in sources])
    return [f"wrote an account over {_n(len(sources), 'record')} by room: "
            f"{by_room(Counter(str(src.get('room_id')) for src in sources))}; it tells nothing in their place until a "
            "selection shows it", _own_size(record, "source", read)]


def selection_lines(store: Any, selection: Mapping[str, Any], back: Mapping[str, Any]) -> List[str]:
    """A selection, from its one typed readback ``back`` (``state`` acting with ``facts``, superseded with ``acting``,
    or failed with ``error``): no claim about what the account tells unless this selection acts; then its flag before
    and now, what the account tells and what this selection names directly (two counts), and its size beside its
    frozen sources'."""
    account = str(selection["target_id"])
    if back["state"] == "superseded":
        return [f"superseded: selection {back['acting']} acts now, not this one"]
    if back["state"] != "acting":
        return [f"saved: selection {selection['id']}; what the account tells now could not be read back ({back.get('error')})"]
    facts = back["facts"]
    earlier = []
    for each in store.targeting(account, ("selection",)):
        if each["id"] == selection["id"]:
            break
        earlier.append(each)
    before = ("shown" if earlier[-1].get("shown") else "not shown") if earlier else "none"
    head = (f"account {account}: {'no selection before this one' if before == 'none' else 'the one before had it ' + before}; "
            f"this one has it {'shown' if selection.get('shown') is not False else 'not shown'}; it tells "
            f"{_n(len(facts['tells']), 'record')} by room: {by_room(Counter(facts['rooms'].values()))}; this selection "
            f"names {len(facts['named'])} of them directly")
    if selection.get("shown") is False:
        return [head]
    record = store.interpret([store.get(account)], sources=False)[0]
    read = _read(store, [(str(src["id"]), src.get("revision")) for src in record.get("sources") or ()])
    return [head, _own_size(record, "source", read)]


def selection_receipt(store: Any, selection: Mapping[str, Any],
                      readback: Callable[[], Tuple[str, Mapping[str, Any]]]) -> Tuple[str, str]:
    """``(effect, receipt)`` of a saved selection: ``readback`` (the tool's one typed readback and its receipt text)
    and the formatting of both run inside the receipt's guard, so a saved selection never answers as an error."""
    held = {"receipt": f"Selection saved as {selection.get('id')}; its readback could not be read."}

    def build() -> List[str]:
        held["receipt"], back = readback()
        return selection_lines(store, selection, back)
    return fold_receipt(selection, build), held["receipt"]


def decision_lines(store: Any, decision: Mapping[str, Any]) -> List[str]:
    """A decision's committed outcome from its target's own record and the decisions before it in the log, which is
    what the store's rule checked: its kind, signer and prior status, its status now, and what its record released."""
    from ouroboros.chronicle_store import ChronicleStore, draft_signer

    target = store.get(str(decision["target_id"]))
    earlier = []
    for each in store.targeting(target["id"], ("decision",)):
        if each["id"] == decision["id"]:
            break
        earlier.append(each)
    prior = (("accepted" if earlier[-1].get("accepted") else "rejected") if earlier else
             ChronicleStore.undecided(target))
    kind, ident, by = target["kind"], target["id"], draft_signer(target.get("author"))
    head = f"the {kind} {ident} by {by} (it was {prior})"
    if decision.get("accepted"):
        return [f"accepted: {head} is accepted"]
    covers = target.get("covers") if isinstance(target.get("covers"), dict) else {}
    notes = len([ref for ref in covers.get("rows") or () if str(ref).startswith("note:")])
    rows = len(covers.get("rows") or ()) - notes
    members = len(covers.get("member_ids") or ())
    released = (f"its {_n(members, 'member')} {'is' if members == 1 else 'are'} released" if kind == "part" else
                f"as a candidate it sealed none of its {_n(rows, 'row')}" if prior == "candidate" else
                f"the {_n(rows, 'row')} and {_n(notes, 'note')} it sealed are released")
    status = "rejected" if prior in ("draft", "candidate") else "withdrawn"
    return [f"{status}: {head} is {status}; {released}"]
