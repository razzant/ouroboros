"""The memory's reads over the shapes of the owner's install on 24.10 (tests/_memory_capture_shape.py).

Main's view held 241 live rooms and its readers looked rows, row positions, task lineages and
legacy authors up one at a time: 2,615 dialogue transactions for one view of 1,300 open rows, a
server core busy for the first quarter hour after an update. Each reader now reads a page of rows
together; each test holds both the answer (the one-at-a-time reader's) and the number of reads.
"""
from __future__ import annotations

import collections
import itertools
import sys

from ouroboros import chat_chain as cc, dialogue_store as ds, memory_view as mv
from tests import _memory_capture_shape as shape


def _counted(monkeypatch, module, name):
    """Each call of ``module.name`` from now on, by its caller's name."""
    calls, real = [], getattr(module, name)

    def counted(*args, **kwargs):
        calls.append(sys._getframe(1).f_code.co_name)
        return real(*args, **kwargs)

    monkeypatch.setattr(module, name, counted)
    return calls


def _one_at_a_time(root, addresses, **_kwargs):
    """The row reader before 24.10: ``resolve_row`` for each address."""
    found = {}
    for address in addresses:
        row, _status = cc.resolve_row(root, address)
        if row is not None:
            found.setdefault(address["row_sha256"], row)
    return found


def test_main_view_reads_its_open_rows_a_page_at_a_time_and_reads_as_before(tmp_path, monkeypatch):
    made = shape.open_rows(tmp_path, 1_300)
    opened = _counted(monkeypatch, ds, "transaction")
    batched = mv.capture_memory_view(tmp_path, made["task"], made["spec"])
    assert batched.active and len(batched.live_rooms) == shape.ROOMS - 1
    pages = -(-1_300 // cc._ADDRESS_PAGE)
    # The stream's pages, two lookups a page, the selected rows' one read, and the readers' bounds.
    assert len(opened) <= 3 * pages + 8, collections.Counter(opened)
    monkeypatch.setattr(cc, "rows_by_address", _one_at_a_time)
    assert mv.capture_memory_view(tmp_path, made["task"], made["spec"]) == batched


def test_a_view_enters_the_chronicle_index_for_its_activation_and_its_one_capture_only(tmp_path, monkeypatch):
    from ouroboros.chronicle_store import ChronicleStore

    made = shape.open_rows(tmp_path, 240, rooms=24)
    store, mind = ChronicleStore(tmp_path), {"kind": "mind", "task_id": "t1", "focus": {"role": "root", "task_id": "t1"}}
    for chat in (shape.MAIN, *range(1000, 1011)):  # a page and a note in twelve rooms: every reader has rooms to visit
        assert store.write_note(room_id=str(chat), task_id="t1", text=f"note {chat}", author=mind).ok
        assert store.publish_page(room_id=str(chat), text=f"page {chat}", author=mind,
                                  covers={"rows": [f"sealed-{chat}"], "stream_span": [chat, chat]}).ok
    entered = _counted(monkeypatch, ChronicleStore, "_index")
    view = mv.capture_memory_view(tmp_path, made["task"], made["spec"])
    assert view.active and len(view.live_rooms) == 17 and len(view.other_rooms) == 6 and view.room["facts"]["pages"] == 1
    assert entered == ["_reading", "capture"]  # the activation receipt, then one read for every reader of the view


def test_rows_by_address_answers_as_resolve_row_and_a_wrong_hint_still_finds_its_row(tmp_path):
    shape.open_rows(tmp_path, 60, rooms=3)
    addresses = [address for address, _row, _pos in cc.iter_rows(tmp_path)]
    assert cc.rows_by_address(tmp_path, addresses) == _one_at_a_time(tmp_path, addresses)
    moved = {**addresses[5], "hint": {"seq": addresses[9]["hint"]["seq"]}}  # a hint naming another row
    absent = cc.row_address({"chat_id": 1, "ts": "2026-01-01T00:00:00+00:00", "text": "never written"})
    for odd in (moved, absent):  # alone: an address keyed like another one would join it
        assert cc.rows_by_address(tmp_path, [odd]) == _one_at_a_time(tmp_path, [odd])
    assert moved["row_sha256"] in cc.rows_by_address(tmp_path, [moved])
    assert cc.rows_by_address(tmp_path, [absent]) == {}


def test_a_filtered_room_reader_that_stops_early_reads_at_most_a_page_beyond_what_it_used(tmp_path, monkeypatch):
    from tests._dialogue import import_all, seed

    seed(tmp_path, live={"chat": [{"chat_id": 1, "direction": "out", "ts": f"2026-09-01T{i // 3600:02d}:{i // 60 % 60:02d}:{i % 60:02d}+00:00",
                                   **({"task_id": "wanted", "text": f"wanted {i}"} if i < 2 else
                                      {"task_id": f"other{i % 50}", "text": f"other {i}"})} for i in range(5_002)]})
    import_all(tmp_path)  # one import, not 5,002 write transactions
    probes = _counted(monkeypatch, ds, "probe_rows")
    first = list(itertools.islice(cc.iter_room_rows(tmp_path, "1", task_ids=["wanted"]), 2))
    assert [row["text"] for _address, row, _pos in first] == ["wanted 0", "wanted 1"]
    assert len(probes) <= 3  # one page of examined rows, not pages of kept rows over the room's 5,000 others


def test_story_terminal_links_are_addressed_together(tmp_path, monkeypatch):
    store = shape.terminal_links(tmp_path, 40)
    reference = mv._story_pages(store, lambda room, *_args: str(room), {})
    opened = _counted(monkeypatch, ds, "transaction")
    story = mv._story_pages(store, lambda room, *_args: str(room), {})
    assert story == reference and sum(len(page["terminal_links"]) for page in story) == 40  # one line per page
    assert len(opened) <= 10, collections.Counter(opened)  # 160 when each terminal row was addressed alone


def test_room_membership_reads_the_rows_task_lineages_together(tmp_path, monkeypatch):
    from ouroboros import memory_inventory, projects_registry

    shape.open_rows(tmp_path, 1_300)
    reads = _counted(monkeypatch, projects_registry, "_read_lineage")
    assert len(memory_inventory.open_rows_by_room(tmp_path)) == shape.ROOMS
    assert len(reads) <= 2, reads  # one read a generation of lineage; 300 tasks read one by one before


def test_legacy_authors_are_read_together_and_answer_as_one_lookup_a_task(tmp_path, monkeypatch):
    from ouroboros import task_results, task_store

    made = shape.legacy_authorship(tmp_path, 240)
    alone = _counted(monkeypatch, task_results, "load_task_result")
    together = _counted(monkeypatch, task_store, "documents_by_ids")
    warmed = mv.capture_memory_view(tmp_path, made["task"], made["spec"])
    assert alone == [] and together == ["admitted_results"]
    monkeypatch.setattr(mv, "warm_lineage", lambda *_args: None)
    assert mv.capture_memory_view(tmp_path, made["task"], made["spec"]) == warmed
    assert len(alone) == 240  # the same authors, one task result at a time


def _legacy_units_fresh(root):
    """``legacy_units`` computed from nothing this process remembers."""
    from ouroboros import memory_inventory as mi
    from ouroboros.chronicle_store import ChronicleStore

    for name in ("_UNITS", "_SPAN_ROWS", "_ROW_SETS", "_RETELLING_CHARS"):
        getattr(mi, name, {}).pop(str(root.resolve()), None)
    return mi.legacy_units(ChronicleStore(root), root)


def test_legacy_units_read_their_spans_once_and_count_rooms_with_current_lineage(tmp_path, monkeypatch):
    from ouroboros import memory_inventory as mi
    from ouroboros.chronicle_store import ChronicleStore
    from tests import _memory_inventory_shared as shared
    from tests._store_shared import seed_result

    shared.world(tmp_path)
    store = ChronicleStore(tmp_path)
    before = mi.legacy_units(store, tmp_path)
    assert any(unit.rows for unit in before)
    read = _counted(monkeypatch, cc, "iter_rows")
    read_rooms = _counted(monkeypatch, cc, "iter_room_rows")
    counted = _counted(monkeypatch, mi, "_legacy_row_sets")
    ds.append(tmp_path, "chat", {"chat_id": 1, "direction": "in", "text": "a row of this turn"})
    assert mi.legacy_units(store, tmp_path) == before and counted == []  # a new row changes no unit: the answer stands
    # A task the old memory saw without a result is admitted under the bound root, and no row is written: its
    # row leaves Main for Alpha.
    seed_result(tmp_path, "t1", {"status": "completed", "chat_id": 1, "root_task_id": "bound", "parent_task_id": "bound"})
    after = mi.legacy_units(store, tmp_path)
    assert read == [] and read_rooms == [] and len(counted) == 1  # the spans' rows are not read again, only recounted
    assert after == _legacy_units_fresh(tmp_path) and [unit.rows for unit in after] != [unit.rows for unit in before]


def test_legacy_units_counted_while_the_lineage_store_was_busy_are_recounted_once_it_answers(tmp_path, monkeypatch):
    from ouroboros import memory_inventory as mi, projects_registry as pr
    from ouroboros.chronicle_store import ChronicleStore
    from tests import _memory_inventory_shared as shared
    from tests._store_shared import seed_result

    shared.world(tmp_path)
    seed_result(tmp_path, "t1", {"status": "completed", "chat_id": 1, "root_task_id": "bound", "parent_task_id": "bound"})
    store, fresh = ChronicleStore(tmp_path), _legacy_units_fresh(tmp_path)
    for key in [key for key in pr._TASK_LINEAGE if key[0] == str(tmp_path)]:
        del pr._TASK_LINEAGE[key]  # t1's lineage is not remembered yet, so the busy store decides its room
    pr._ROOM_ANSWERS.pop(str(pr._bindings_path(tmp_path)), None)
    with monkeypatch.context() as busy:
        attempts = []
        busy.setattr(pr, "_read_lineage", lambda *_args, **_kwargs: attempts.append(1))  # None: the store cannot answer now
        during, reused = mi.legacy_units(store, tmp_path), len(attempts)
        attempts.clear()
        mi._UNITS.pop(str(tmp_path.resolve()))
        assert mi.legacy_units(store, tmp_path) == during and len(attempts) == reused - 1  # the reuse check spent one attempt
    assert during != fresh  # t1's row counted in Main while its lineage could not be read
    assert mi.legacy_units(store, tmp_path) == fresh  # the store answers again, no row written: counted with it


def test_legacy_rows_are_the_rooms_rows_the_room_reader_returns(tmp_path):
    """The span rows filtered by the room rule are each unit's rows as the room reader returned them."""
    from ouroboros import memory_inventory as mi
    from ouroboros.chronicle_store import ChronicleStore
    from tests import _memory_inventory_shared as shared, _memory_production_shape as production

    for root, build in ((tmp_path / "shared", shared.world), (tmp_path / "production", production.production)):
        root.mkdir()
        build(root)
        store, facts = ChronicleStore(root), mi.membership_facts(root)
        ranges = {pointer["node_id"]: (str(pointer["room_id"]), span) for pointer in store.legacy_pointer_rows()
                  if (span := mi._exact_range(pointer)) is not None}
        assert ranges
        by_room = {}
        for record_id, (room, bounds) in ranges.items():
            entries = mi._room_entries(root, store, room, bounds)
            stamps = [row.get("ts") for _address, row, _pos in entries]
            by_room[record_id] = (frozenset(address["row_sha256"] for address, _row, _pos in entries),
                                  mi.source_time_span(stamps) if stamps else None, entries[0][2] if entries else -1)
        assert mi._legacy_row_sets(root, ranges, facts) == by_room


def test_terminal_link_rooms_read_the_writers_lineages_together(tmp_path, monkeypatch):
    from ouroboros import projects_registry

    store = shape.terminal_links(tmp_path, 40)
    reads = _counted(monkeypatch, projects_registry, "_read_lineage")
    mv._story_pages(store, lambda room, *_args: str(room), {})
    assert len(reads) <= 2, reads  # 40 writers' lineages, one read each before


def test_rows_by_address_takes_the_text_form_and_the_first_address_of_a_key_that_resolves(tmp_path):
    shape.open_rows(tmp_path, 12, rooms=2)
    address, row, _pos = next(cc.iter_rows(tmp_path))
    text = cc.format_address(address)  # the text form carries a 12-hex digest, its key
    assert cc.rows_by_address(tmp_path, [text]) == {cc._normalized(text)["row_sha256"]: row}
    wrong = {**address, "ts": "2026-01-01T00:00:00+00:00", "hint": {}}  # same digest, a time no row has
    assert cc.rows_by_address(tmp_path, [wrong, address]) == {address["row_sha256"]: row}


def test_warm_lineage_reads_every_task_the_author_rule_asks_about():
    from ouroboros.dialogue_provenance import row_author, warm_lineage

    asked, warmed = [], []

    def lookup(task_id):
        asked.append(task_id)
        return None

    lookup.warm = lambda ids: warmed.extend(sorted(ids))
    lineage = {"lineage_epoch": {"pos": 10}, "lineage_lookup": lookup}
    rows = [({"direction": "out", "task_id": "out"}, 1), ({"task_id": "no-direction"}, 2),
            ({"direction": "in", "task_id": "owner"}, 3), ({"direction": "system", "task_id": "host"}, 4),
            ({"direction": "out", "task_id": "later"}, 12), ({"direction": "out", "task_id": "kid", "parent_task_id": "p"}, 5)]
    warm_lineage(lineage, rows)
    for row, pos in rows:
        row_author(row, pos=pos, **lineage)
    assert sorted(set(asked)) == warmed == ["no-direction", "out"]


def test_documents_by_ids_keys_each_document_by_its_stored_row(tmp_path):
    from ouroboros import task_results, task_store
    from tests._store_shared import seed_result

    import sqlite3
    from ouroboros.store import store_path

    seed_result(tmp_path, "good", {"status": "completed", "root_task_id": "good", "is_root_task": True})
    seed_result(tmp_path, "zbad", {"status": "completed", "root_task_id": "elsewhere"})
    with sqlite3.connect(store_path(tmp_path)) as conn:  # a damaged body that names another task
        conn.execute("UPDATE task_bodies SET body=json_set(body, '$.task_id', 'good') WHERE task_id='zbad'")
    documents = task_store.documents_by_ids(tmp_path, ["good", "zbad"])
    assert documents["good"]["root_task_id"] == "good" and documents["zbad"]["root_task_id"] == "elsewhere"
    admitted = task_results.admitted_results(tmp_path, ["good", "zbad"])
    assert admitted["good"]["root_task_id"] == "good" and admitted["zbad"] is None  # a body naming another task is refused
