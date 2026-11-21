"""``memory_read(members=true)``: what stands under a room or a record, by the relations the chronicle records.

Descent through my own tellings. Each relation is named as what it is, and no other is invented: a page
*seals* rows and notes, a part *folds* members, an account *cites* source versions, a selection *replaces*
records, a ``legacy`` section *retells* a positional range of its room. No common parent and no containment
tree is claimed; a room is no record, only the place its records and rows belong to.

- **A room** (``room_id``): its acting tellings in publication order (pages, parts, accounts, notes, legacy
  sections and journal gaps as ``ChronicleStore.room_records`` reads them), each record's withdrawal or pending
  candidacy named apart, then its untold rows. The untold rows are the store's one open-row computation
  (``memory_inventory.open_room_rows``: a chat row of the room after the legacy frontier that no acting page
  seals), never computed a second way, shown as the maximal consecutive runs of that set in the room's reading
  order (``dialogue_provenance.in_reading_order`` over every row of the room after the frontier), each with its
  bounds, count, period and speakers. No calendar bucket. The room's line names its situation name
  ``room:<id>``, the string a note's ``claims`` name; no pointer is built here.
- **A record** (``node_id``): what its own relation names, in its recorded order: a page's sealed rows as runs in
  each sealed room's reading order, then its notes; a part's members; an account's sources, each with the version
  it cites and what changed since (``ChronicleStore._source_state``); a selection's account and the records it
  replaces; a legacy section's range. A record with no such relation says so.

Each record line carries its kind and id, its status (a helper's draft, an accepted draft, a candidate, a
withdrawn record), its author as the store records it (``memory_inventory.record_byline``), its period, its
counts, the corrections standing under it, its fold and teller, and its own first sentence
(``text_sentences.first_sentence``, as the view prints a lead): the delivered lead where its ``metadata.lead_rule``
records the instruction, otherwise labelled ``(record opening, no lead)``; then its exact readers.

When the listed entries fit the tool's own page bound with room left, the next stored level is added whole, in
stored order, labelled as context, and so on while a whole level fits; a level that does not fit is left to its
readers, never cut to "the important" entries. The optional words filter (``dialogue_words``) drops the listed
lines that do not match, over the whole list before paging, so the page size never decides what matches.

The path is passive: it never activates or imports the chronicle (``tools/chronicle._activated`` is not called).
Before activation it reads nothing of the chronicle and names the gap: a room says "chronicle not activated",
never "no untold rows". The chronicle's state is read once for the whole answer (``chronicle_store.capture``).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from ouroboros import chat_chain, memory_inventory
from ouroboros.chronicle_import import LEGACY_ROOM_ID, legacy_frontier, row_lineage
from ouroboros.chronicle_store import ChronicleStore, capture, draft_signer
from ouroboros.dialogue_provenance import in_reading_order, rooms_read_by_time, row_author
from ouroboros.store import StoreError
from ouroboros.text_sentences import first_sentence
from ouroboros.tool_capabilities import result_window

NO_LEAD = " (record opening, no lead)"  # the view's label (``memory_view._page_label``, ``_note_pointer``)
NOT_ACTIVATED = "chronicle not activated"
RELATIONS = {"page": "seals", "part": "folds", "account": "cites", "selection": "replaces", "legacy": "retells"}
_SELECTORS = ("rows", "story", "owner_words", "source_ref", "proposed_selection", "after_seq", "revision", "task_id",
              "period", "from", "to")
_ORDER = "listed order"


def members_text(root: Path, members: bool, words: Any, start: int, limit: int, *, room_id: Any, node_id: Any,
                 others: Mapping[str, Any]) -> str:
    """``memory_read(members=true, …)``'s answer, after its own argument rules (ValueError: the tool's argument error)."""
    check_call(room_id=room_id, node_id=node_id, words=words, members=members, others=others)
    return read_members(root, room_id=room_id, node_id=node_id, start=start, limit=limit, words=words)


def check_call(*, room_id: Any, node_id: Any, words: Any, members: bool, others: Mapping[str, Any]) -> None:
    """The members call's own argument rules (ValueError): one address, no other mode, words only with members."""
    if not members:
        raise ValueError("words filters a members listing: memory_read(members=true, room_id=… or node_id=…, words=[…])")
    given = [name for name in _SELECTORS if others.get(name) not in (None, False, 0, "", [], {})]
    if given:
        raise ValueError(f"members=true is its own mode; it takes room_id or node_id, words and start, not {', '.join(given)}")
    room, node = room_id is not None and str(room_id).strip() != "", bool(str(node_id or "").strip())
    if room == node:
        raise ValueError("members=true needs one address: room_id=<chat id as text> (or 'legacy') for a room, "
                         "or node_id=<record id> for a record")
    if room and str(room_id).strip() != LEGACY_ROOM_ID and not str(room_id).strip().lstrip("-").isdigit():
        raise ValueError(f"room_id {str(room_id).strip()!r} is not a room address: members reads one room by its chat id "
                         "as text (or 'legacy')")
    if words is not None and (not isinstance(words, list) or not words or not all(isinstance(w, str) and w.strip()
                                                                                   for w in words)):
        raise ValueError("words is a non-empty list of non-empty strings, each one alternative")


def read_members(root: Path, *, room_id: Any = None, node_id: Any = None, start: int = 0, limit: int,
                 words: Optional[List[str]] = None) -> str:
    """One page of what stands under ``room_id`` or ``node_id`` (``check_call`` has passed)."""
    store = ChronicleStore(root)
    active = store.log_path.exists() and bool(store.activation())  # never ``ensure_activated``: a reader imports nothing
    room = str(room_id).strip() if room_id is not None and str(room_id).strip() != "" else None
    node = str(node_id or "").strip()
    if not active:
        head = (f"members of room:{room}" if room is not None else f"members of record {node}") + (
            f"; {NOT_ACTIVATED}: the one-time legacy import has not run, so "
            + ("this room's tellings and untold rows are unknown here, not none" if room is not None
               else "no record can be read") + "; any other memory_read or chronicle_write call activates it")
        return result_window(head, "", 0, limit, lambda _end: "", _ORDER)
    with capture(root, tail="none"):  # one state of the chronicle for every line of the answer
        lister = _Lister(root, store)
        head, entries = lister.room(room) if room is not None else lister.node(node)
        query = _query(words)
        shown = [entry for entry in entries if query is None or query(entry.line)]
        if query is not None:
            head += f"; words {words!r}: {len(shown)} of {len(entries)} listed lines match (applied before paging)"
        text = "\n".join(entry.line for entry in shown)
        reader = f"room_id='{room}'" if room is not None else f"node_id='{node}'"
        also = f", words={words!r}" if words is not None else ""

        def resume(end: int) -> str:
            return f"memory_read(members=true, {reader}{also}, start={end})"

        if query is None and shown:  # the widest continuation line ``result_window`` may print for a one-page answer
            widest = len(f"chars {limit}–{limit} of {limit}; next_start={limit} ({_ORDER}); " + resume(limit))
            text += lister.context(shown, limit - len(head) - widest - 2 - len(text))
    return result_window(head, text, start, limit, resume, _ORDER)


def _query(words: Optional[List[str]]) -> Optional[Callable[[str], bool]]:
    if words is None:
        return None
    from ouroboros.dialogue_words import match_text, parse

    parsed = parse(words)
    return lambda line: match_text(parsed, line)


class _Entry:
    """One listed line and, for a record with a relation, the record whose members stand under it."""
    __slots__ = ("line", "record")

    def __init__(self, line: str, record: Optional[Dict[str, Any]] = None):
        self.line, self.record = line, record


class _Lister:
    """The lines of one answer, read inside one capture of the chronicle."""

    def __init__(self, root: Path, store: ChronicleStore):
        self.root, self.store = root, store
        self.lineage = row_lineage(root)
        self._units: Optional[Dict[str, Any]] = None

    # --- addresses ------------------------------------------------------------------------------------

    def room(self, room: str) -> Tuple[str, List[_Entry]]:
        head = (f"members of room:{room}: its acting tellings in publication order, then its untold rows as runs in "
                "the room's reading order")
        records = self.store.room_records(room)
        withdrawn = [record for record in self.store.interpreted(room, kinds=("page", "part"))
                     if record.get("status") in ("rejected", "candidate")]
        entries = self._record_entries(records)
        if withdrawn:
            head += "; not acting, not listed: " + ", ".join(
                f"{'withdrawn' if record['status'] == 'rejected' else 'candidate'} {record['kind']} {record['id']}"
                for record in withdrawn) + " (memory_read(node_id=…) reads each)"
        if room == LEGACY_ROOM_ID:
            return head + "; the legacy room names no chat, so no row stands in it", entries
        frontier = legacy_frontier(self.store)
        if type(frontier.get("pos")) is not int:
            return head + "; untold rows unknown: the legacy frontier is not recorded", entries
        gaps: set = set()
        try:
            runs, timed = self._untold_runs(room, gaps)
        except StoreError as exc:  # an interval still importing or unreadable is unknown, never "no untold rows"
            return head + f"; untold rows unknown: the dialogue store answered {type(exc).__name__}", entries
        if gaps:
            from ouroboros.room_appearances import GAP_DETAILS  # D15->D17 is lazy-only

            head += "".join(f"; incomplete: {GAP_DETAILS.get(kind, kind)}" for kind in sorted(gaps))
        if timed:
            head += ("; this room reads by send time: rows=true between a run's bounds reads stored order and may "
                     "show other rows")
        head += f"; untold rows: {sum(len(run) for run in runs)} in {_n(len(runs), 'run')}" if runs else "; untold rows: none"
        return head, entries + [_Entry(self._run_line("untold", room, run)) for run in runs]

    def node(self, node_id: str) -> Tuple[str, List[_Entry]]:
        record = self.store.get(node_id)
        if record is None:
            raise ValueError(f"memory node {node_id} not found")
        record = self.store.interpret([record])[0]
        kind, relation = record["kind"], RELATIONS.get(record["kind"])
        head = f"members of {kind} {record['id']} (room:{record['room_id']})"
        if relation is None:
            what = ("nothing stands under a note in the chronicle's relations; a page that seals it names it"
                    if kind == "note" else f"a {kind} has no members relation")
            return f"{head}: {what}; memory_read(node_id='{record['id']}') reads it", []
        lines = getattr(self, f"_under_{kind}")(record)
        return f"{head}: it {relation} " + lines[0], lines[1]

    # --- what each relation holds ---------------------------------------------------------------------

    def _under_page(self, record: Dict[str, Any]) -> Tuple[str, List[_Entry]]:
        covers = record.get("covers") if isinstance(record.get("covers"), dict) else {}
        rooms = covers.get("rooms") if isinstance(covers.get("rooms"), dict) else {record["room_id"]: covers.get("rows") or []}
        entries: List[_Entry] = []
        notes: List[str] = []
        count = 0
        for room, refs in rooms.items():
            notes += [ref[5:] for ref in refs if str(ref).startswith("note:") and ref[5:] not in notes]
            wanted = {str(ref) for ref in refs if not str(ref).startswith("note:")}
            if not wanted:
                continue
            try:
                try:
                    rows = list(chat_chain.iter_room_rows(self.root, room, from_addr=covers.get("first"),
                                                          to_addr=covers.get("last")))
                except (chat_chain.RowAddressError, ValueError):  # a span bound that no longer resolves: the room whole
                    rows = list(chat_chain.iter_room_rows(self.root, room))
            except StoreError as exc:  # an interval still importing or unreadable: named, never "no rows"
                entries.append(_Entry(f"- sealed rows of room:{room}: {len(wanted)} rows not read here: the dialogue "
                                      f"store answered {type(exc).__name__}"))
                continue
            runs = self._runs(rows, wanted)
            count += sum(len(run) for run in runs)
            entries += [_Entry(self._run_line("sealed", room, run)) for run in runs]
        found = [note for ident in notes if (note := self.store.get(ident)) is not None]
        entries += self._record_entries(self.store.interpret(found) if found else [])
        missing = len(notes) - len(found)
        rooms_named = ", ".join(f"room:{room}" for room in rooms)
        return (f"{_n(count, 'row')} and {_n(len(notes), 'note')}: the rows as runs in each sealed room's reading order ({rooms_named}), "
                f"then the notes" + (f"; {missing} sealed notes are not in this chronicle" if missing else ""), entries)

    def _under_part(self, record: Dict[str, Any]) -> Tuple[str, List[_Entry]]:
        ids = [str(ident) for ident in (record.get("covers") or {}).get("member_ids") or ()]
        found = [member for ident in ids if (member := self.store.get(ident)) is not None]
        missing = len(ids) - len(found)
        return (f"{_n(len(ids), 'record')}, in its recorded order" + (f"; {missing} are not in this chronicle" if missing else ""),
                self._record_entries(self.store.interpret(found) if found else []))

    def _under_account(self, record: Dict[str, Any]) -> Tuple[str, List[_Entry]]:
        sources = [src for src in record.get("sources") or () if isinstance(src, Mapping)]
        found = {ident: hit for src in sources if (hit := self.store.get(ident := str(src.get("id") or ""))) is not None}
        acting = {item["id"]: item for item in (self.store.interpret(list(found.values())) if found else [])}
        entries = []
        for src in sources:
            cited = f"cites revision {src.get('revision')}" + (f" ({src['status']} then)" if src.get("status") and
                                                               src["status"] != "final" else "")
            if src.get("missing") or str(src.get("id")) not in acting:
                entries.append(_Entry(f"- source {src.get('kind')} {src.get('id')} (room:{src.get('room_id')}); {cited}; "
                                      "NOT in this chronicle: its words cannot be read here"))
                continue
            since = [f"later corrections not in this account: {', '.join(src['later_corrections'])}"
                     if src.get("later_corrections") else "",
                     "" if src.get("revision_known", True) else "that revision is not in this chronicle",
                     f"now {src['status_now']}" if src.get("status_now") and src.get("status_now") != src.get("status") else ""]
            source = acting[str(src["id"])]
            entries.append(_Entry(self._record_line(source, extra=[cited, *filter(None, since)],
                                                    reader=f"memory_read(node_id='{src['id']}', revision='{src.get('revision')}')"),
                                  source if source["kind"] in RELATIONS else None))
        return f"{_n(len(sources), 'source version')}, in its recorded order; nothing sealed or folded", entries

    def _under_selection(self, record: Dict[str, Any]) -> Tuple[str, List[_Entry]]:
        ids = [str(ident) for ident in record.get("replaces") or ()]
        found = [hit for ident in ids if (hit := self.store.get(ident)) is not None]
        account = record.get("target_id")
        shown = "shows" if record.get("shown") else "withdraws"
        return (f"{_n(len(ids), 'record')}: it {shown} account {account} (memory_read(node_id='{account}')) in place of them",
                self._record_entries(self.store.interpret(found) if found else []))

    def _under_legacy(self, record: Dict[str, Any]) -> Tuple[str, List[_Entry]]:
        raw = (record.get("covers") or {}).get("raw_range") or {}
        room = record["room_id"]
        if raw.get("status") != "exact" or not raw.get("pos"):
            return "a positional range of its room that is not established (raw_range unknown)", []
        unit = self._legacy_units().get(record["id"])
        rows = (f"{unit.rows} rows of room:{room} in it, {unit.uncovered} not sealed by a page" if unit is not None
                else "its rows not counted")
        first, last = (chat_chain.address_text(raw.get(key)) for key in ("first", "last"))
        reader = (f"memory_read(room_id='{room}', rows=true, from='{first}', to='{last}')"
                  if str(room).lstrip("-").isdigit() else "the legacy room names no chat")
        return (f"positions {raw['pos'][0]}–{raw['pos'][1]} of the chat stream", [_Entry(
            f"- range: positions {raw['pos'][0]}–{raw['pos'][1]}; {rows}; {reader}")])

    # --- lines ----------------------------------------------------------------------------------------

    def _record_entries(self, records: Iterable[Dict[str, Any]]) -> List[_Entry]:
        records = list(records)
        from ouroboros.memory_view_account import told_by

        told = told_by(self.store, [record["id"] for record in records]) if records else {}
        return [_Entry(self._record_line(record, told=told.get(record["id"]) or ""),
                       record if record["kind"] in RELATIONS else None) for record in records]

    def _record_line(self, record: Dict[str, Any], *, told: str = "", extra: Iterable[str] = (), reader: str = "") -> str:
        kind, ident = record["kind"], record["id"]
        status = record.get("status")
        facts = [f"room:{record.get('room_id')}",
                 {"draft": f"draft by {draft_signer(record.get('author'))}, not yet accepted", "accepted": "accepted draft",
                  "candidate": "candidate, not acting", "rejected": "withdrawn"}.get(str(status), ""),
                 f"by {memory_inventory.record_byline(record)}", self._period(record), _counts(record),
                 f"{_n(len(record['corrections']), 'correction')} under it (revision {record.get('revision')})"
                 if record.get("corrections") else "",
                 f"folded into part {record['folded_into']}" if record.get("folded_into") else "",
                 f"told in my story by account {told}" if told else "", _selected(record), *extra]
        meta = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
        words = first_sentence(" ".join(str(record.get("text") or "").split()))
        if meta.get("transfer") and words:  # a transfer's words are data, never a line of their own
            words = memory_inventory.quoted(words)
        opening = (f"«{words}»" if words else "(no words)") + ("" if "lead_rule" in meta else NO_LEAD)
        readers = [reader or f"memory_read(node_id='{ident}')"]
        if kind in RELATIONS:
            readers.append(f"under it: memory_read(members=true, node_id='{ident}')")
        return f"- {kind} {ident} [" + "; ".join(fact for fact in facts if fact) + f"] {opening}; " + "; ".join(readers)

    def _period(self, record: Dict[str, Any]) -> str:
        kind = record["kind"]
        if kind == "note":
            return f"written {memory_inventory.minute(record['ts'])}" if record.get("ts") else "date not recorded"
        if kind == "selection":
            return ""
        if kind == "page":
            return memory_inventory.period_text((record.get("covers") or {}).get("ts_span"))
        try:
            period = memory_inventory.record_period(self.store, record, self._legacy_units())
        except StoreError as exc:
            return f"period unknown ({type(exc).__name__})"
        return memory_inventory.period_text(period.span, period.note())

    def _legacy_units(self) -> Dict[str, Any]:
        if self._units is None:
            self._units = {unit.record_id: unit for unit in memory_inventory.legacy_units(self.store, self.root)}
        return self._units

    def _run_line(self, what: str, room: str, run: List[memory_inventory.Entry]) -> str:
        first, last = chat_chain.format_address(run[0][0]), chat_chain.format_address(run[-1][0])
        span = memory_inventory.period_text(_time_span(meta.get("ts") for _address, meta, _pos in run))
        speakers = Counter(str(row_author(meta, pos=pos, **self.lineage).get("label") or "author not recorded")
                           for _address, meta, pos in run)
        named = ", ".join(f"{label} {count}" for label, count in speakers.most_common())
        return (f"- {what} rows of room:{room}: {_n(len(run), 'row')}, {span}; speakers: {named}; from {first} to {last}; "
                f"memory_read(room_id='{room}', rows=true, from='{first}', to='{last}')")

    # --- runs -----------------------------------------------------------------------------------------

    def _untold_runs(self, room: str, gaps: set) -> Tuple[List[List[memory_inventory.Entry]], bool]:
        """The runs of the room's open rows (the one computation) among every row of the room after the frontier."""
        untold = {address["row_sha256"] for address, _meta, _pos in memory_inventory.open_room_rows(self.root, room,
                                                                                                    gaps=gaps)}
        if not untold:
            return [], False
        every = memory_inventory._room_entries(self.root, self.store, room)  # sealed rows too: what breaks a run
        timed = bool(rooms_read_by_time(self.root, (meta for _address, meta, _pos in every)))
        return self._runs(every, untold), timed

    def _runs(self, entries: List[memory_inventory.Entry], members: set) -> List[List[memory_inventory.Entry]]:
        """The maximal consecutive runs of ``members`` (row_sha256) among ``entries`` in their room's reading order."""
        ordered = in_reading_order(entries, rooms_read_by_time(self.root, (meta for _a, meta, _p in entries)),
                                   lambda entry: entry[1])
        runs: List[List[memory_inventory.Entry]] = []
        current: List[memory_inventory.Entry] = []
        for entry in [*ordered, None]:
            if entry is not None and entry[0]["row_sha256"] in members:
                current.append(entry)
                continue
            if current:
                runs.append(current)
                current = []
        return runs

    # --- context levels -------------------------------------------------------------------------------

    def context(self, listed: List[_Entry], room: int) -> str:
        """The next stored levels under the listed records, each added whole while it fits ``room``, else none of it."""
        added, level, parents = "", 2, [entry for entry in listed if entry.record is not None]
        expanded = set()  # a record reached twice (a folded page listed beside its part) is expanded once
        while parents and room > 0:
            block, below, used = [], [], 0
            for parent in parents:
                if parent.record["id"] in expanded:
                    continue
                expanded.add(parent.record["id"])
                _relation, entries = getattr(self, f"_under_{parent.record['kind']}")(parent.record)
                if not entries:
                    continue
                lines = [f"under {parent.record['kind']} {parent.record['id']}:"] + ["  " + e.line for e in entries]
                used += sum(len(line) + 1 for line in lines)
                if used > room:
                    return added
                block += lines
                below += [entry for entry in entries if entry.record is not None]
            if not block:
                return added
            text = (f"\ncontext, level {level}: what stands under the entries above, added whole because the page "
                    "bound fits it (not asked; its own readers page it)\n" + "\n".join(block))
            if len(text) > room:
                return added
            added, room, parents, level = added + text, room - len(text), below, level + 1
        return added


def _counts(record: Mapping[str, Any]) -> str:
    covers = record.get("covers") if isinstance(record.get("covers"), Mapping) else {}
    kind = record.get("kind")
    if kind == "page":
        notes = len(covers.get("note_ids") or ())
        return f"seals {_n(covers.get('count', 0), 'row')}" + (f" and {_n(notes, 'note')}" if notes else "")
    if kind == "part":
        return f"folds {_n(len(covers.get('member_ids') or ()), 'record')}"
    if kind == "account":
        return f"cites {_n(len(record.get('sources') or ()), 'source version')}"
    if kind == "selection":
        shown = "shows" if record.get("shown") else "withdraws"
        return f"{shown} account {record.get('target_id')}, replaces {len(record.get('replaces') or ())}"
    if kind == "legacy":
        raw = covers.get("raw_range") if isinstance(covers.get("raw_range"), Mapping) else {}
        return (f"retells positions {raw['pos'][0]}–{raw['pos'][1]}" if raw.get("status") == "exact" and raw.get("pos")
                else "retells a range not established")
    if kind == "gap":
        return "a torn line of the chronicle's journal"
    if kind == "note":
        return f"task {record.get('task_id') or 'not recorded'}"
    return ""


def _n(count: Any, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


def _selected(record: Mapping[str, Any]) -> str:
    """An account's acting selection, the record that shows it in my story or withdraws it; '' for other kinds."""
    if record.get("kind") != "account":
        return ""
    chosen = record.get("selection") if isinstance(record.get("selection"), Mapping) else None
    if chosen is None:
        return "never selected, so not in my story"
    return (f"{'shown' if chosen.get('shown') else 'withdrawn'} by selection {chosen.get('id')} "
            f"(memory_read(members=true, node_id='{chosen.get('id')}'))")


def _time_span(stamps: Iterable[Any]) -> Dict[str, Any]:
    from ouroboros.chronicle_store import source_time_span

    return source_time_span(stamps)
