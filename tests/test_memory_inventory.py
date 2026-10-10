"""The one inventory of open and folded memory (``ouroboros.memory_inventory``).

Open rows of a room are its chat rows after the legacy frontier outside the room's
sealed set; a legacy unit is folded when a part holds it or when it has rows and a
page seals every one of them; a period counts only when all its units are folded.
Each rule is pinned in both directions, on a fixture with three rooms besides Main
and twenty stream rows.
"""
from __future__ import annotations

import ast
import pathlib

from ouroboros import chat_chain
from ouroboros import chronicle_import as ci
from ouroboros import memory_inventory as mi
from ouroboros.chronicle_store import ChronicleStore
from tests import _memory_inventory_shared as shared

REPO = pathlib.Path(__file__).resolve().parents[1]


def _addresses(root):
    return {pos: address for address, _row, pos in chat_chain.iter_rows(root)}


def _page(root, room, first, last):
    from ouroboros.tools.chronicle import page_covers

    addresses = _addresses(root)
    covers = page_covers(root, room, from_addr=addresses[first], to_addr=addresses[last])["covers"]
    result = ChronicleStore(root).publish_page(room_id=room, text=f"page {first}-{last}", covers=covers,
                                               author=shared.MIND)
    assert result.ok, result
    return result.record


def _part(root, room, members):
    store = ChronicleStore(root)
    result = store.publish_part(room_id=room, text="a part", member_ids=members, author=shared.MIND,
                                expected_sequence=store.room_head(room))
    assert result.ok, result
    return result.record


def _units(root):
    return {unit.record_id: unit for unit in mi.legacy_units(ChronicleStore(root), root)}


def _positions(entries):
    return [pos for _address, _meta, pos in entries]


# --- open rows ------------------------------------------------------------------------------------

def test_open_rows_of_each_room_are_its_rows_after_the_frontier(tmp_path):
    rooms = shared.world(tmp_path)
    frontier = ci.legacy_frontier(ChronicleStore(tmp_path))
    assert frontier["status"] == "exact" and frontier["pos"] == 10
    by_room = mi.open_rows_by_room(tmp_path)
    for room in ("1", str(rooms["alpha"]), str(rooms["beta"]), "777", "0"):
        expected = [pos for _a, _r, pos in chat_chain.iter_room_rows(tmp_path, room) if pos >= 10]
        assert _positions(mi.open_room_rows(tmp_path, room)) == expected, room
        assert _positions(by_room.get(room, [])) == expected, room
    # Legacy rows are not open anywhere; the open rows of the rooms partition nothing below the frontier.
    assert all(pos >= 10 for entries in by_room.values() for pos in _positions(entries))
    assert _positions(by_room["1"]) == [10, 11, 12, 15, 16, 18, 19]
    assert _positions(by_room["777"]) == [15] and _positions(by_room["0"]) == [17]


def test_a_page_closes_its_rows_only_in_its_own_room(tmp_path):
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.projects_registry import bind_task_to_project

    rooms = shared.world(tmp_path)
    beta = str(rooms["beta"])
    # The owner's last Main message starts work bound to beta: the row is in Main and in beta.
    ref = build_owner_message_ref(chat_id=1, client_message_id="m2", ts="2026-09-03T00:09:00+00:00",
                                  text="and more")
    bind_task_to_project(tmp_path, "tb2", "beta", origin={"ref": ref, "text": "and more"})
    assert 19 in _positions(mi.open_room_rows(tmp_path, "1"))
    assert 19 in _positions(mi.open_room_rows(tmp_path, beta))
    _page(tmp_path, beta, 14, 19)
    assert _positions(mi.open_room_rows(tmp_path, beta)) == []
    assert _positions(mi.open_room_rows(tmp_path, "1")) == [10, 11, 12, 15, 16, 18, 19]
    _page(tmp_path, "1", 11, 12)
    assert _positions(mi.open_room_rows(tmp_path, "1")) == [10, 15, 16, 18, 19]
    assert set(mi.open_rows_by_room(tmp_path)) == {"1", str(rooms["alpha"]), "777", "0"}


def test_an_unknown_frontier_opens_only_rows_after_the_chain_end_seen_at_activation(tmp_path):
    shared.world(tmp_path, cursor=False)
    frontier = ci.legacy_frontier(ChronicleStore(tmp_path))
    assert frontier["status"] == "unknown" and frontier["pos"] == 20
    assert mi.open_rows_by_room(tmp_path) == {}
    shared.append(tmp_path / "logs" / "chat.jsonl", shared.msg("2026-09-04T00:00:00+00:00", "after", client_message_id="n"))
    assert _positions(mi.open_room_rows(tmp_path, "1")) == [20]


def test_before_activation_nothing_is_open_and_the_chronicle_is_not_created(tmp_path):
    shared.world(tmp_path, activate=False)
    assert mi.open_rows_by_room(tmp_path) == {} and mi.open_room_rows(tmp_path, "1") == []
    assert mi.rows_after_frontier(tmp_path) == []
    assert mi.legacy_units(ChronicleStore(tmp_path), tmp_path) == []
    assert not (tmp_path / "memory" / "chronicle").exists()


# --- the folded rule -------------------------------------------------------------------------------

def test_a_unit_folds_when_pages_seal_all_its_rows_and_not_before(tmp_path):
    rooms = shared.world(tmp_path)
    alpha = str(rooms["alpha"])
    unit_id = f"legacy-b00-r{alpha}"
    before = _units(tmp_path)[unit_id]
    assert (before.raw, before.rows, before.uncovered, before.folded) == ("exact", 3, 3, False)
    _page(tmp_path, alpha, 2, 3)  # partial: rows 2 and 3 of {2, 3, 5}
    partial = _units(tmp_path)
    assert (partial[unit_id].uncovered, partial[unit_id].folded) == (1, False)
    _page(tmp_path, alpha, 5, 5)
    units = _units(tmp_path)
    assert (units[unit_id].uncovered, units[unit_id].folded) == (0, True)
    # Main's unit of the same block shares row 5, but alpha's pages seal only alpha's rows.
    assert units["legacy-b00-r1"].rows == 4 and units["legacy-b00-r1"].folded is False
    assert mi.legacy_progress(list(units.values()))["folded"] == 0


def test_a_part_folds_a_unit_and_a_unit_without_rows_folds_only_through_a_part(tmp_path):
    shared.world(tmp_path)
    quiet = _units(tmp_path)["legacy-b01-r1"]  # Main had no row in block one
    assert (quiet.rows, quiet.folded) == (0, False)
    _page(tmp_path, "1", 11, 12)  # pages in the room do not fold an empty set
    assert _units(tmp_path)["legacy-b01-r1"].folded is False
    _part(tmp_path, "1", ["legacy-b01-r1"])
    assert _units(tmp_path)["legacy-b01-r1"].folded is True
    transport = _units(tmp_path)["legacy-b00-r777"]
    assert (transport.rows, transport.uncovered, transport.folded) == (1, 1, False)
    _part(tmp_path, "777", ["legacy-b00-r777"])
    transport = _units(tmp_path)["legacy-b00-r777"]
    assert (transport.uncovered, transport.folded) == (1, True)  # folded by the part, its rows still open


def test_a_later_binding_moves_legacy_rows_between_units(tmp_path):
    from ouroboros.projects_registry import bind_task_to_project

    rooms = shared.world(tmp_path)
    beta = str(rooms["beta"])
    assert _units(tmp_path)["legacy-b00-r1"].rows == 4
    bind_task_to_project(tmp_path, "t1", "beta", origin={"absent": "post_hoc_unresolved"})
    units = _units(tmp_path)  # the registry changed: the verdicts are recomputed, not reused
    assert units["legacy-b00-r1"].rows == 3  # t1's reply now belongs to beta
    assert [pos for _a, _m, pos in mi.open_room_rows(tmp_path, beta)] == [14]  # no open row of t1 after the frontier


def test_a_period_counts_only_when_every_unit_of_it_is_folded(tmp_path):
    rooms = shared.world(tmp_path)
    alpha = str(rooms["alpha"])
    assert mi.legacy_progress(list(_units(tmp_path).values())) == {"periods": 2, "folded": 0, "pending": [0, 1]}
    _page(tmp_path, alpha, 2, 5)
    _page(tmp_path, "1", 0, 5)  # Main's rows of block zero: 0, 1, 4 (the transport chat) and 5
    assert mi.legacy_progress(list(_units(tmp_path).values()))["pending"] == [0, 1]  # 777 still open
    _part(tmp_path, "777", ["legacy-b00-r777"])
    assert mi.legacy_progress(list(_units(tmp_path).values())) == {"periods": 2, "folded": 1, "pending": [1]}


def test_flat_memory_and_unknown_ranges_have_no_rows_and_fold_only_through_a_part(tmp_path):
    shared.world(tmp_path, cursor=False, flat="The flat old summary.")
    units = _units(tmp_path)
    flat = next(unit for unit in units.values() if unit.record_id.startswith("legacy-flat-"))
    assert (flat.room_id, flat.block, flat.raw, flat.rows, flat.folded) == ("legacy", None, "unknown", 0, False)
    assert flat.retelling_chars == len("The flat old summary.")
    sections = [unit for unit in units.values() if unit.block is not None]
    assert sections and all(unit.raw == "unknown" and unit.rows == 0 for unit in sections)
    assert mi.legacy_progress(list(units.values()))["periods"] == 2  # the flat file and the gap are no period
    _part(tmp_path, "legacy", [flat.record_id])
    assert _units(tmp_path)[flat.record_id].folded is True


def test_a_fallback_refusal_receipt_rides_with_its_unit(tmp_path):
    rooms = shared.world(tmp_path)
    beta_unit = f"legacy-b01-r{rooms['beta']}"
    receipt = {"input_sha256": "e" * 64, "kind": "context_overflow"}
    assert ChronicleStore(tmp_path).publish([], scan_state={"fallback_refusals": {beta_unit: receipt}}).ok
    units = _units(tmp_path)
    assert units[beta_unit].refusal == receipt
    assert all(unit.refusal is None for record_id, unit in units.items() if record_id != beta_unit)


def test_the_readers_publish_nothing(tmp_path):
    rooms = shared.world(tmp_path)
    chronicle = tmp_path / "memory" / "chronicle"
    ChronicleStore(tmp_path).scan_state()  # the index and lock exist once any reader ran
    log = (chronicle / "records.jsonl").read_bytes()
    files = sorted(path.name for path in chronicle.iterdir())
    mi.open_rows_by_room(tmp_path)
    mi.open_room_rows(tmp_path, str(rooms["alpha"]))
    mi.legacy_progress(mi.legacy_units(ChronicleStore(tmp_path), tmp_path))
    mi.oldest_open_segment(tmp_path, ChronicleStore(tmp_path), "1")
    mi.oldest_open_segment(tmp_path, ChronicleStore(tmp_path), str(rooms["alpha"]), pos_range=(0, 6))
    gaps = set()
    events, current, _window = mi.memory_changes(tmp_path, None, gaps)
    assert mi.memory_changes(tmp_path, {"sequence": 0, "record_id": None}, gaps)[0] == [] and not gaps
    assert events == [] and current["sequence"] > 0  # the import is in the journal, but no change to observe
    assert (chronicle / "records.jsonl").read_bytes() == log
    assert sorted(path.name for path in chronicle.iterdir()) == files


# --- open segments --------------------------------------------------------------------------------

def _shas(entries):
    return [address["row_sha256"] for address, *_rest in entries]


def _covered(root, segment):
    """What ``page_covers`` names for the segment's bounds: the rows one page would seal."""
    from ouroboros.tools.chronicle import page_covers

    return page_covers(root, segment.room_id, from_addr=segment.from_addr, to_addr=segment.to_addr)["rows"]


def _short(root, pos):
    """The 12-hex text address of a stream row, the form the view's floor fact carries."""
    return chat_chain.parse_address(chat_chain.format_address(_addresses(root)[pos]))


def _segment(root, room, **kwargs):
    return mi.oldest_open_segment(root, ChronicleStore(root), room, **kwargs)


def test_the_oldest_open_segment_stops_before_the_first_sealed_row_and_at_until(tmp_path):
    shared.world(tmp_path)
    whole = _segment(tmp_path, "1")
    assert _positions(whole.rows) == [10, 11, 12, 15, 16, 18, 19]  # Main's open rows, nothing sealed yet
    assert _shas(_covered(tmp_path, whole)) == _shas(whole.rows)
    assert whole.room_id == "1" and whole.head_sequence == ChronicleStore(tmp_path).room_head("1")
    bounded = _segment(tmp_path, "1", until=_short(tmp_path, 15))
    assert _positions(bounded.rows) == [10, 11, 12, 15]  # the rows after the newest addressed row stay open
    assert _shas(_covered(tmp_path, bounded)) == _shas(bounded.rows)
    _page(tmp_path, "1", 12, 12)
    stopped = _segment(tmp_path, "1", until=_short(tmp_path, 15))
    assert _positions(stopped.rows) == [10, 11]  # not across the sealed row, though until lies beyond it
    assert _shas(_covered(tmp_path, stopped)) == _shas(stopped.rows)
    assert _positions(_segment(tmp_path, "1", until=_short(tmp_path, 10)).rows) == [10]
    _page(tmp_path, "1", 10, 11)
    after = _segment(tmp_path, "1")
    assert _positions(after.rows) == [15, 16, 18, 19]  # the oldest run is now the one after the sealed rows
    assert after.head_sequence == ChronicleStore(tmp_path).room_head("1") > whole.head_sequence
    # An until before the first open row, outside the room or unreadable names no run.
    assert _segment(tmp_path, "1", until=_short(tmp_path, 11)) is None
    assert _segment(tmp_path, "1", until=_short(tmp_path, 13)) is None  # an alpha row
    assert _segment(tmp_path, "1", until={"chat_id": 1}) is None


def test_a_partly_sealed_unit_yields_only_its_oldest_unsealed_run(tmp_path):
    rooms = shared.world(tmp_path)
    alpha = str(rooms["alpha"])
    span = tuple(ChronicleStore(tmp_path).get(f"legacy-b00-r{alpha}")["covers"]["raw_range"]["pos"])
    assert span == (0, 6)
    whole = _segment(tmp_path, alpha, pos_range=span)
    assert _positions(whole.rows) == [2, 3, 5]  # alpha's rows of block zero, not the transport row between
    assert _shas(_covered(tmp_path, whole)) == _shas(whole.rows)
    _page(tmp_path, alpha, 3, 3)
    first = _segment(tmp_path, alpha, pos_range=span)
    assert _positions(first.rows) == [2] and _shas(_covered(tmp_path, first)) == _shas(first.rows)
    _page(tmp_path, alpha, 2, 2)
    second = _segment(tmp_path, alpha, pos_range=span)
    assert _positions(second.rows) == [5] and _shas(_covered(tmp_path, second)) == _shas(second.rows)
    assert _units(tmp_path)[f"legacy-b00-r{alpha}"].folded is False
    _page(tmp_path, alpha, 5, 5)
    assert _segment(tmp_path, alpha, pos_range=span) is None  # fully sealed: nothing for a writer
    assert _units(tmp_path)[f"legacy-b00-r{alpha}"].folded is True
    # The open rows after the frontier are another range of the same room.
    assert _positions(_segment(tmp_path, alpha).rows) == [13]


def test_peoples_words_inside_a_segment_stay_in_it(tmp_path):
    shared.world(tmp_path)
    segment = _segment(tmp_path, "1", pos_range=(0, 6))
    assert _positions(segment.rows) == [0, 1, 4, 5]
    covered = _covered(tmp_path, segment)
    assert _shas(covered) == _shas(segment.rows)
    # The owner, my reply, a transport correspondent, the owner again: a range of the room, not of its authors.
    assert [row["direction"] for _address, row in covered] == ["in", "out", "in", "in"]
    _page(tmp_path, "1", 1, 1)  # my reply sealed: the owner's first line alone is the oldest run
    assert _positions(_segment(tmp_path, "1", pos_range=(0, 6)).rows) == [0]


def test_a_segment_needs_an_activated_chronicle_and_creates_none(tmp_path):
    shared.world(tmp_path, activate=False)
    store = ChronicleStore(tmp_path)
    assert mi.oldest_open_segment(tmp_path, store, "1") is None
    assert mi.oldest_open_segment(tmp_path, store, "1", pos_range=(0, 6)) is None
    assert not (tmp_path / "memory" / "chronicle").exists()


# --- the shortage fact -------------------------------------------------------------------------------

def _trace(level, *, roundtrip=True):
    """A task trace with the view's floor fact, written by the view's own producer."""
    import json

    from ouroboros import memory_floor as mf
    from tests import _memory_view_synthetic as syn

    facts = mf.view_facts(syn.actor("main"), level, window_tokens=272_000, mode="max", allowances={})
    trace = {mi.VIEW_TRACE_KEY: mf.trace_facts(facts), "tool_calls": []}
    return json.loads(json.dumps(trace)) if roundtrip else trace


def test_replies_or_peoples_words_by_address_are_an_open_rows_shortage():
    from ouroboros.memory_view import FloorLevel
    from tests import _memory_view_synthetic as syn

    lane1 = syn.actor("main").room["lane1"]
    reply = max((item for item in lane1 if item["kind"] == "ouroboros"), key=lambda item: item["pos"])
    word = max((item for item in lane1 if item["kind"] != "ouroboros"), key=lambda item: item["pos"])
    for step, item in (("F2", reply), ("F7", word)):
        fact = mi.shortage_from_trace(_trace(FloorLevel(((step, (item["address"],)),))))
        assert fact is not None and fact.kind == "open_rows", step
        assert fact.room_id == "1" and fact.window_tokens == 272_000 and fact.mode == "max"
        expected = chat_chain.parse_address(item["address"])
        assert {key: fact.newest_addressed_row[key] for key in ("chat_id", "ts", "row_sha256")} == {
            key: expected[key] for key in ("chat_id", "ts", "row_sha256")}
    # Both kinds of fact: the open rows come first, they are what the view asked the mind about.
    legacy = syn.actor("main").room["legacy"][0]["id"]
    both = mi.shortage_from_trace(_trace(FloorLevel((("F4", (legacy,)), ("F2", (reply["address"],))))))
    assert both.kind == "open_rows"


def test_old_narrative_by_pointer_is_a_shortage_only_when_the_window_took_it():
    from ouroboros.memory_view import FloorLevel
    from tests import _memory_view_synthetic as syn

    snapshot = syn.actor("main")
    legacy = [item["id"] for item in snapshot.room["legacy"]][:2]
    fact = mi.shortage_from_trace(_trace(FloorLevel((("F4", tuple(legacy)),), by_budget=0)))
    assert fact.kind == "narrative" and fact.pointer_records == tuple(legacy) and fact.newest_addressed_row is None
    page = {"id": "page-old", "kind": "page"}
    fifth = mi.shortage_from_trace(_trace(FloorLevel((("F5", (page["id"],)),))))
    assert fifth.kind == "narrative" and fifth.pointer_records == ("page-old",)
    # The same steps taken by an owner-selected mode target are not a shortage of the window.
    assert mi.shortage_from_trace(_trace(FloorLevel((("F4", tuple(legacy)),), by_budget=2))) is None
    assert mi.shortage_from_trace(_trace(FloorLevel((("F5", ("page-old",)),), by_budget=1))) is None


def test_other_steps_foreign_shapes_and_no_fact_are_no_shortage():
    from ouroboros.memory_view import FloorLevel
    from tests import _memory_view_synthetic as syn

    room = syn.actor("main").room
    line = room["lane2"][0]
    only_facts = _trace(FloorLevel((("F1", (line["first"],)),)))
    assert only_facts[mi.VIEW_TRACE_KEY]["floor"]["newest_addressed_row"] is not None  # F1 names a row too
    assert mi.shortage_from_trace(only_facts) is None
    for step, ident in (("F1b", "2"), ("F3", "1000"), ("F6", room["lane1"][0]["address"])):
        assert mi.shortage_from_trace(_trace(FloorLevel(((step, (ident,)),)))) is None, step
    assert mi.shortage_from_trace(_trace(FloorLevel())) is None  # the full view
    good = _trace(FloorLevel((("F2", (room["lane1"][0]["address"],)),)))
    assert mi.shortage_from_trace(good) is not None
    for broken in (None, {}, [], {"tool_calls": []}, {mi.VIEW_TRACE_KEY: "fact"},
                   {mi.VIEW_TRACE_KEY: {**good[mi.VIEW_TRACE_KEY], "floor": None}},
                   {mi.VIEW_TRACE_KEY: {**good[mi.VIEW_TRACE_KEY], "room_id": None}},
                   {mi.VIEW_TRACE_KEY: {**good[mi.VIEW_TRACE_KEY], "floor": {
                       **good[mi.VIEW_TRACE_KEY]["floor"], "steps": [["F2", 1]]}}},
                   {mi.VIEW_TRACE_KEY: {**good[mi.VIEW_TRACE_KEY], "floor": {
                       **good[mi.VIEW_TRACE_KEY]["floor"], "steps": {"F2": True}}}},
                   {mi.VIEW_TRACE_KEY: {**good[mi.VIEW_TRACE_KEY], "floor": {
                       **good[mi.VIEW_TRACE_KEY]["floor"], "newest_addressed_row": None}}},
                   {mi.VIEW_TRACE_KEY: {**good[mi.VIEW_TRACE_KEY], "floor": {
                       **good[mi.VIEW_TRACE_KEY]["floor"], "newest_addressed_row": {"chat_id": 1}}}}):
        assert mi.shortage_from_trace(broken) is None, broken
    narrative = _trace(FloorLevel((("F4", (room["legacy"][0]["id"],)),)))
    assert mi.shortage_from_trace(narrative) is not None
    for floor in ({"pointer_records": []}, {"pointer_records": "legacy"}, {"by_budget": None}, {"by_budget": False}):
        changed = {mi.VIEW_TRACE_KEY: {**narrative[mi.VIEW_TRACE_KEY],
                                       "floor": {**narrative[mi.VIEW_TRACE_KEY]["floor"], **floor}}}
        assert mi.shortage_from_trace(changed) is None, floor


# --- row metadata ----------------------------------------------------------------------------------

def test_row_metadata_keeps_attribution_and_membership_without_the_text(tmp_path):
    from ouroboros.dialogue_provenance import row_class
    from ouroboros.project_dialogue import _text_sha256

    rows = shared.chat(tmp_path, {"alpha": 7, "beta": 8})
    epoch = {"pos": 12}

    def lookup(task_id):
        return {"is_root_task": True} if task_id == "t1" else None

    for pos, row in enumerate(rows):
        meta = mi.row_meta(row)
        assert "text" not in meta and meta["text_chars"] == len(row["text"])
        lineage = {"pos": pos, "lineage_epoch": epoch, "lineage_lookup": lookup}
        assert row_class(meta, **lineage) == row_class(row, **lineage), pos
        if row["direction"] == "in":
            assert meta["text_sha256"] == _text_sha256(row["text"]) and meta["source_keys"]
        else:
            assert "text_sha256" not in meta and "source_keys" not in meta
    # The attribution fields are load-bearing: without them a child's report reads as my own words.
    child = rows[7]
    stripped = {key: value for key, value in mi.row_meta(child).items()
                if key not in {"subagent_task_id", "delegation_role"}}
    assert row_class(stripped, pos=7)["author"]["kind"] == "ouroboros"
    assert row_class(mi.row_meta(child), pos=7)["author"]["kind"] == "child"


def test_the_inventory_imports_no_model_client_and_reads_projects_lazily():
    tree = ast.parse((REPO / "ouroboros" / "memory_inventory.py").read_text(encoding="utf-8"))
    top, nested = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            (top if node in tree.body else nested).update(names)
    imported = top | nested
    assert not any(name == "ouroboros.llm" or name.startswith(("ouroboros.llm", "ouroboros.consolidator"))
                   for name in imported), imported
    assert {"ouroboros.project_dialogue", "ouroboros.projects_registry"} <= nested
    assert not {"ouroboros.project_dialogue", "ouroboros.projects_registry"} & top
    assert mi.VIEW_TRACE_KEY == "memory_view"
