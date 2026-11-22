"""``memory_read(members=true)``: what stands under a room or a record, by the relations the chronicle records.

Descent through my own tellings. Each relation is named as what it is, and no other is invented: a page
*seals* rows and notes, a part *folds* members, an account *cites* source versions, a selection *replaces*
records, a ``legacy`` section *retells* a positional range of its room. No common parent and no containment
tree is claimed; a room is no record, only the place its records and rows belong to.

- **A room** (``room_id``): its acting tellings in publication order: its own records as
  ``ChronicleStore.room_records`` reads them, and every page recorded in another room that holds an acting seal
  here (the index's ``page_rows``), each saying where it is recorded; then, as paged entries, its withdrawn and
  candidate pages and parts, named as what they are; then its untold rows. The untold rows are the store's one
  open-row computation (``memory_inventory.open_room_rows``: a chat row of the room after the legacy frontier that
  no acting page seals), collected from its own single read as the maximal runs of that set in the room's reading
  order, each with its bounds, count, period and speakers. No calendar bucket. The room's line names its
  situation name ``room:<id>``, the string a note's ``claims`` name; no pointer is built here.
- **A record** (``node_id``): what its own relation names, in its recorded order: a page's sealed rows as runs in
  each sealed room's reading order, recorded, found and missing counted per room, then its notes (a withdrawn page
  says it formerly sealed them); a part's members; an account's sources, each at the version it cites with what
  changed since (``ChronicleStore._source_state``); a selection's account and the records it replaces; a legacy
  section's range. A record with no such relation says so.

A run is shown in reading order, but rows=true reads stored order, so each run's readers are its maximal
stored-contiguous slices: following them returns exactly the run's rows.

Each record line carries its kind and id, its status (a helper's draft, an accepted draft, a candidate, a
withdrawn record), its author as the store records it (``memory_inventory.record_byline``), its period, its
counts, the corrections standing under it, its fold and teller, and its own first sentence
(``text_sentences.first_sentence``, as the view prints a lead): the delivered lead where its ``metadata.lead_rule``
records the instruction, otherwise labelled ``(record opening, no lead)``; then its exact readers.

When the listed entries fit the tool's own page bound with room left, the next stored level is added whole, in
stored order, labelled as context, and so on while a whole level fits; a level that does not fit is left to its
readers, never cut to "the important" entries. The optional words filter (``dialogue_words``) keeps the records
whose own text or one correction under them matches, each judged on its own, a cited source at the revision cited;
it runs over the whole list before paging, so the page size never decides what matches. Runs of rows are not
filtered, since this listing reads no row's words: each room's runs become one line naming a word search of that
room over their period.

Passive means: it never activates the chronicle and never runs the legacy import (``tools/chronicle._activated``
is not called); before activation it reads nothing of the chronicle and names the gap, never "no untold rows".
The store's own catch-up of its disposable index can still run on a read, as on every chronicle read. The
chronicle's state is read once for the whole answer (``chronicle_store.capture``), the acting seals of a room in
one more index statement.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

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
    if words is not None:  # the shared predicate's own rules (``dialogue_words.parse``) name a bad item
        from ouroboros.dialogue_words import parse

        parse(words)


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
               else "no record can be read") + "; a memory_read of a room or a record, or any chronicle_write, "
            "activates it")
        return result_window(head, "", 0, limit, lambda _end: "", _ORDER)
    with capture(root, tail="none"):  # one state of the chronicle for every line of the answer
        lister = _Lister(root, store)
        head, entries = lister.room(room) if room is not None else lister.node(node)
        shown = entries if words is None else _filtered(entries, words)
        if words is not None:
            told = [entry for entry in entries if entry.fields is not None]
            head += (f"; words: {sum(1 for entry in shown if entry.fields is not None)} of {len(told)} listed records "
                     "match in their own text or one correction (applied before paging); runs of rows are not filtered")
        text = "\n".join(entry.line for entry in shown)
        reader = f"room_id='{room}'" if room is not None else f"node_id='{node}'"
        also = f", words={words!r}" if words is not None else ""

        def resume(end: int) -> str:
            return f"memory_read(members=true, {reader}{also}, start={end})"

        if words is None and shown:  # the widest continuation line ``result_window`` may print for a one-page answer
            widest = len(f"chars {limit}–{limit} of {limit}; next_start={limit} ({_ORDER}); " + resume(limit))
            text += lister.context(shown, limit - len(head) - widest - 2 - len(text))
    return result_window(head, text, start, limit, resume, _ORDER)


def _filtered(entries: List["_Entry"], words: List[str]) -> List["_Entry"]:
    """The listed set under ``words``: a record stays when its own text or one correction under it matches, each
    field on its own (never its id, facts or readers), with an excerpt naming the field when the match is not in its
    printed opening; each room's runs of rows, whose words this listing never reads, become one search line; any
    other line stays."""
    from ouroboros.dialogue_words import excerpt, match_text, parse

    query, kept, runs = parse(words), [], {}
    for entry in entries:
        if entry.run is not None:
            if entry.run[1] not in runs:
                runs[entry.run[1]] = [entry.run[0], entry.run[1], []]
                kept.append(runs[entry.run[1]])
            runs[entry.run[1]][2].append(entry.run[2])
        elif entry.fields is None:
            kept.append(entry)
        elif hit := next(((name, text) for name, text in entry.fields if match_text(query, text)), None):
            opening = first_sentence(" ".join(str(entry.record.get("text") or "").split()))
            near = "" if hit[0] == "text" and match_text(query, opening) else excerpt(query, hit[1], 200)
            label = "excerpt" if hit[0] == "text" else f"excerpt ({hit[0]})"
            kept.append(_Entry(entry.line + (f"; {label}: «{near}»" if near else ""), entry.record, entry.fields))
    return [item if isinstance(item, _Entry) else _Entry(_search_line(words, *item)) for item in kept]


def _search_line(words: List[str], what: str, room: str, runs: List[List[memory_inventory.Entry]]) -> str:
    """One room's runs under ``words``: their count, and a word search of that room over their period (parsed times)."""
    rows = [meta for run in runs for _address, meta, _pos in run]
    span = _time_span(meta.get("ts") for meta in rows)
    dates = f", date_from='{span['start']}', date_to='{span['end']}'" if span["start"] else ""
    undated = sum(1 for meta in rows if not meta.get("ts"))
    return (f"- {what} rows of room:{room}: {_n(len(rows), 'row')} in {_n(len(runs), 'run')}, not filtered by words "
            f"here (this listing reads no row's words); a word search of room:{room} over their period, which also "
            f"returns told rows of that period: chat_history(words={words!r}, room_id='{room}'{dates})"
            + (f"; {undated} of them carry no time" if undated else "") + "; the unfiltered listing prints each run's "
            "exact readers")


class _Entry:
    """One listed line; for a record the record itself and its own fields (what ``words`` judges: ``(name, text)``,
    its text and each correction); for a run of rows ``(what, room, run)``."""
    __slots__ = ("line", "record", "fields", "run")

    def __init__(self, line: str, record: Optional[Dict[str, Any]] = None,
                 fields: Optional[List[Tuple[str, str]]] = None,
                 run: Optional[Tuple[str, str, List[memory_inventory.Entry]]] = None):
        self.line, self.record, self.fields, self.run = line, record, fields, run


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
        own = self.store.room_records(room)
        held = {record["id"] for record in own}
        elsewhere = [page for ident in self._sealing_pages(room) - held if (page := self.store.get(ident)) is not None]
        acting = [record for record in (self.store.interpret(elsewhere) if elsewhere else [])
                  if record.get("status") not in ("rejected", "candidate")]
        records = sorted(own + acting, key=lambda record: record["sequence"])
        withdrawn = [record for record in self.store.interpreted(room, kinds=("page", "part"))
                     if record.get("status") in ("rejected", "candidate")]
        entries = self._record_entries(records, here=room) + self._record_entries(withdrawn)
        if withdrawn:
            head += f"; {_n(len(withdrawn), 'withdrawn or candidate record')} of this room listed after the acting ones"
        if room == LEGACY_ROOM_ID:
            return head + "; the legacy room names no chat, so no row stands in it", entries
        frontier = legacy_frontier(self.store)
        if type(frontier.get("pos")) is not int:
            return head + "; untold rows unknown: the legacy frontier is not recorded", entries
        gaps: set = set()
        every: List[memory_inventory.Entry] = []
        try:
            runs: List[List[memory_inventory.Entry]] = []
            memory_inventory.open_room_rows(self.root, room, gaps=gaps, runs=runs, every=every)  # one read of the room
        except StoreError as exc:  # an interval still importing or unreadable is unknown, never "no untold rows"
            return head + f"; untold rows unknown: the dialogue store answered {type(exc).__name__}", entries
        if gaps:
            from ouroboros.room_appearances import GAP_DETAILS  # D15->D17 is lazy-only

            head += "".join(f"; incomplete: {GAP_DETAILS.get(kind, kind)}" for kind in sorted(gaps))
        head += (f"; untold rows: {sum(len(run) for run in runs)} in {_n(len(runs), 'run')}" if runs else
                 "; untold rows: none observed" if gaps else "; untold rows: none")
        return head, entries + [_Entry(self._run_line("untold", room, run, every), run=("untold", room, run))
                                for run in runs]

    def _sealing_pages(self, room: str) -> set:
        """The pages that hold an acting seal of this room's rows: one statement over the index's ``page_rows``."""
        with self.store._index() as db:  # the snapshot keeps seals without their page; the index keeps both
            return {ident for (ident,) in db.execute("SELECT DISTINCT page_id FROM page_rows WHERE room=?", (room,))}

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
        if record.get("status") == "rejected":  # a withdrawn page or part holds nothing now: what it once named
            relation, head = {"seals": "formerly sealed", "folds": "formerly folded"}[relation], head + ", withdrawn"
        elif record.get("status") == "candidate":
            head += ", a candidate, not acting"
        lines = getattr(self, f"_under_{kind}")(record)
        return f"{head}: it {relation} " + lines[0], lines[1]

    # --- what each relation holds ---------------------------------------------------------------------

    def _under_page(self, record: Dict[str, Any]) -> Tuple[str, List[_Entry]]:
        covers = record.get("covers") if isinstance(record.get("covers"), dict) else {}
        entries: List[_Entry] = []
        notes: List[str] = []
        counted: List[str] = []
        for room, refs in _rooms_of(record).items():
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
                counted.append(f"room:{room} {len(wanted)} recorded, not read ({type(exc).__name__})")
                entries.append(_Entry(f"- sealed rows of room:{room}: {_n(len(wanted), 'recorded row')} not read here: "
                                      f"the dialogue store answered {type(exc).__name__}"))
                continue
            runs = self._runs(rows, wanted)
            found = {entry[0]["row_sha256"] for run in runs for entry in run}
            missing = sorted(wanted - found)
            counted.append(f"room:{room} {len(wanted)} recorded, {len(found)} found, {len(missing)} missing")
            entries += [_Entry(self._run_line("sealed", room, run, rows), run=("sealed", room, run)) for run in runs]
            if missing:  # an edited row changes its content hash: the page still names the old one
                entries.append(_Entry(f"- sealed rows of room:{room} not found: {_n(len(missing), 'recorded row')} no "
                                      "longer resolve (an edited row changes its hash): " + ", ".join(
                                          ref[:12] for ref in missing)))
        found_notes = [note for ident in notes if (note := self.store.get(ident)) is not None]
        entries += self._record_entries(self.store.interpret(found_notes) if found_notes else [])
        missing = len(notes) - len(found_notes)
        return (f"rows ({'; '.join(counted) or 'none recorded'}) and {_n(len(notes), 'note')}: the rows as runs in "
                "each sealed room's reading order, then the notes"
                + (f"; {missing} sealed notes are not in this chronicle" if missing else ""), entries)

    def _under_part(self, record: Dict[str, Any]) -> Tuple[str, List[_Entry]]:
        ids = [str(ident) for ident in (record.get("covers") or {}).get("member_ids") or ()]
        return f"{_n(len(ids), 'record')}, in its recorded order", self._named(ids)

    def _named(self, ids: List[str]) -> List[_Entry]:
        """The records ``ids`` names, in that order; one that is not in this chronicle keeps its recorded id."""
        hits = [hit for hit in (self.store.get(ident) for ident in ids) if hit is not None]
        found = {item["id"]: item for item in self.store.interpret(hits)} if hits else {}
        listed = {entry.record["id"]: entry for entry in self._record_entries(list(found.values()))}
        return [listed[ident] if ident in listed else _Entry(f"- record {ident}: NOT in this chronicle; its words "
                                                            "cannot be read here") for ident in ids]

    def _under_account(self, record: Dict[str, Any]) -> Tuple[str, List[_Entry]]:
        sources = [src for src in record.get("sources") or () if isinstance(src, Mapping)]
        entries = []
        for src in sources:
            cited = f"cites revision {src.get('revision')}" + (f" ({src['status']} then)" if src.get("status") and
                                                               src["status"] != "final" else "")
            found = None if src.get("missing") else self.store.get(str(src.get("id") or ""))
            if found is None:
                entries.append(_Entry(f"- source {src.get('kind')} {src.get('id')} (room:{src.get('room_id')}); {cited}; "
                                      "NOT in this chronicle: its words cannot be read here"))
                continue
            source = self.store.interpret([found])[0]
            try:  # what words judge: the text as of the revision cited, which the printed reader opens
                then = self.store.interpret([found], revision=str(src.get("revision")))[0] if src.get(
                    "revision_known", True) and src.get("revision") else source
            except ValueError:
                then = source
            since = [f"later corrections not in this account: {', '.join(src['later_corrections'])}"
                     if src.get("later_corrections") else "",
                     "" if src.get("revision_known", True) else "that revision is not in this chronicle",
                     f"now {src['status_now']}" if src.get("status_now") and src.get("status_now") != src.get("status") else ""]
            entries.append(_Entry(self._record_line(source, extra=[cited, *filter(None, since)],
                                                    reader=f"memory_read(node_id='{src['id']}', revision='{src.get('revision')}')"),
                                  source, self._fields(then)))
        return f"{_n(len(sources), 'source version')}, in its recorded order; nothing sealed or folded", entries

    def _under_selection(self, record: Dict[str, Any]) -> Tuple[str, List[_Entry]]:
        ids = [str(ident) for ident in record.get("replaces") or ()]
        account = record.get("target_id")
        shown = "shows" if record.get("shown") else "withdraws"
        return (f"{_n(len(ids), 'record')}: it {shown} account {account} (memory_read(node_id='{account}')) in place of them",
                self._named(ids))

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

    def _record_entries(self, records: Iterable[Dict[str, Any]], *, here: str = "") -> List[_Entry]:
        records = list(records)
        from ouroboros.memory_view_account import told_by

        told = told_by(self.store, [record["id"] for record in records]) if records else {}
        return [_Entry(self._record_line(record, told=told.get(record["id"]) or "", here=here), record,
                       self._fields(record)) for record in records]

    def _fields(self, record: Mapping[str, Any]) -> List[Tuple[str, str]]:
        """What a record says, field by field: its own text, then each correction standing under it (as interpreted:
        at a cited revision, only those up to it)."""
        fixes = [fix for item in record.get("corrections") or () if (fix := self.store.get(str(item.get("id")))) is not None]
        return [("text", str(record.get("text") or "")), *((f"correction {fix['id']}", str(fix.get("text") or ""))
                                                          for fix in fixes)]

    def _record_line(self, record: Dict[str, Any], *, told: str = "", extra: Iterable[str] = (), reader: str = "",
                     here: str = "") -> str:
        kind, ident = record["kind"], record["id"]
        status = record.get("status")
        facts = [f"room:{record.get('room_id')}" if not here or str(record.get("room_id")) == here else
                 f"recorded in room:{record.get('room_id')}, seals rows of room:{here}",
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
            return "period not recorded (a choice" + (f", made {memory_inventory.minute(record['ts'])})"
                                                      if record.get("ts") else ")")
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

    def _run_line(self, what: str, room: str, run: List[memory_inventory.Entry],
                  stored: List[memory_inventory.Entry]) -> str:
        first, last = chat_chain.format_address(run[0][0]), chat_chain.format_address(run[-1][0])
        span = memory_inventory.period_text(_time_span(meta.get("ts") for _address, meta, _pos in run))
        speakers = Counter(str(row_author(meta, pos=pos, **self.lineage).get("label") or "author not recorded")
                           for _address, meta, pos in run)
        named = ", ".join(f"{label} {count}" for label, count in speakers.most_common())
        readers = "; ".join(f"memory_read(room_id='{room}', rows=true, from='{chat_chain.format_address(a)}', "
                            f"to='{chat_chain.format_address(b)}')" for a, b in _slices(run, stored))
        return (f"- {what} rows of room:{room}: {_n(len(run), 'row')}, {span}; speakers: {named}; from {first} to {last} "
                f"in reading order; " + readers)

    # --- runs -----------------------------------------------------------------------------------------

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
        def below_of(entries: List[_Entry]) -> List[_Entry]:
            return [entry for entry in entries if entry.record is not None and entry.record["kind"] in RELATIONS]

        added, level, parents = "", 2, below_of(listed)
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
                below += below_of(entries)
            if not block:
                return added
            text = (f"\ncontext, level {level}: what stands under the entries above, added whole because the page "
                    "bound fits it (not asked; its own readers page it)\n" + "\n".join(block))
            if len(text) > room:
                return added
            added, room, parents, level = added + text, room - len(text), below, level + 1
        return added


def _slices(run: List[memory_inventory.Entry], stored: List[memory_inventory.Entry]) -> List[Tuple[Any, Any]]:
    """The run's maximal stored-contiguous slices, ``(first, last)`` addresses in stored order: rows=true between
    each pair returns exactly those rows of the run, whatever the room's reading order."""
    index = {entry[2]: at for at, entry in enumerate(stored)}
    places = sorted(index[entry[2]] for entry in run if entry[2] in index)
    slices, begin = [], None
    for at, place in enumerate(places):
        if begin is None:
            begin = place
        if at + 1 == len(places) or places[at + 1] != place + 1:
            slices.append((stored[begin][0], stored[place][0]))
            begin = None
    return slices


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


def _rooms_of(page: Mapping[str, Any]) -> Dict[str, Any]:
    """The row refs a page seals, by room: a conversation page's ``covers.rooms``, any other page's own room."""
    covers = page.get("covers") if isinstance(page.get("covers"), Mapping) else {}
    rooms = covers.get("rooms")
    return dict(rooms) if isinstance(rooms, Mapping) else {str(page.get("room_id")): covers.get("rows") or []}


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
