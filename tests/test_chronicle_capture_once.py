"""A chronicle record is parsed once per snapshot, every reader takes its top level by copy, and nothing inside changes."""
from __future__ import annotations

import pytest

from ouroboros import chronicle_store
from ouroboros.chronicle_store import ChronicleStore, capture
from ouroboros.context_timing import measured_sections, section_spans
from ouroboros import chat_chain
from ouroboros import memory_view as mv
from ouroboros.tools.chronicle import page_covers
from tests import _memory_inventory_shared as shared


MAIN = {"id": "turn0001", "chat_id": 1}


@pytest.fixture
def root(tmp_path):
    """The shared small world: rooms with open rows and two published pages."""
    rooms = shared.world(tmp_path)
    addresses = {pos: address for address, _row, pos in chat_chain.iter_rows(tmp_path)}
    store = ChronicleStore(tmp_path)
    for room, first, last in ((str(rooms["alpha"]), 13, 13), ("1", 10, 11)):
        covers = page_covers(tmp_path, room, from_addr=addresses[first], to_addr=addresses[last])["covers"]
        assert store.publish_page(room_id=room, text=f"Page {room} {first}-{last}.", covers=covers, author=shared.MIND).ok
    return tmp_path


def _ids(store):
    return [record["id"] for record in store.records(kinds=("page",))]


def test_a_capture_parses_each_record_once(root):
    """Inside one capture, however many readers ask, each stored text is parsed once; the counter says so."""
    store = ChronicleStore(root)
    with measured_sections(), capture(root):
        ids = _ids(store)
        for ident in ids * 3:
            store.get(ident)
        store.records(kinds=("page",))
        store.interpreted(kinds=("page",))
        work = section_spans()["work"]
    assert work["records_parsed"] and not work.get("records_parsed_again")


def test_without_a_capture_each_read_parses_again(root):
    """Outside a capture the index parses per read, as before: the memo belongs to one state of the chronicle."""
    store = ChronicleStore(root)
    ident = _ids(store)[0]
    first, second = store.get(ident), store.get(ident)
    assert first == second and first is not second


def test_a_reader_takes_a_copy_of_the_top_level(root):
    """A reader that changes the top level of what it read does not change the record of the next reader."""
    store = ChronicleStore(root)
    with capture(root):
        ident = _ids(store)[0]
        got = store.get(ident)
        got["text"] = "changed by a reader"
        assert store.get(ident)["text"] != "changed by a reader"
        for record in store.records(kinds=("page",)):
            record["text"] = "changed by a reader"
        assert all(record["text"] != "changed by a reader" for record in store.records(kinds=("page",)))


class _Frozen(dict):
    def _refuse(self, *_a, **_k):
        raise AssertionError("a reader changed what a parsed record holds inside")

    __setitem__ = __delitem__ = pop = popitem = clear = update = setdefault = _refuse


def _freeze(value):
    if isinstance(value, dict):
        return _Frozen({key: _freeze(item) for key, item in value.items()})
    return value


def test_a_capture_s_readers_never_change_what_a_record_holds_inside(root, monkeypatch):
    """The readers of one capture see a record whose nested parts refuse any change: the whole view of a real capture
    is drawn and nothing raises. Parsed records may be shared because nothing writes into them."""
    plain = chronicle_store._Snapshot.body

    def frozen(self, row):
        known = row[0] in self.bodies
        parsed = plain(self, row)
        if not known:
            parsed = self.bodies[row[0]] = _freeze(parsed)
        return parsed

    monkeypatch.setattr(chronicle_store._Snapshot, "body", frozen)
    snapshot = mv.capture_memory_view(root, MAIN, mv.view_spec_for_task(MAIN, root))
    assert mv.render_story(snapshot) and mv.render_room(snapshot)
