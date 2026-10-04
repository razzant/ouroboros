"""Two writers on one room: the acting mind, consciousness and the fallback helper.

Overlap is decided mechanically by the chronicle: a page whose row set meets another page's is
``already_sealed`` (a page needs no room head), a part read against a stale head is
``revision_conflict`` with the current head, and a helper's draft yields its rows once the mind
rejects it. Pinned on the memory-inventory fixture's Main room (open rows 10, 11, 12, 15, 16, 18, 19).
"""
from __future__ import annotations

import json

import pytest

from ouroboros import chat_chain
from ouroboros.chronicle_store import ChronicleStore
from tests import _memory_inventory_shared as shared
from tests.test_memory_fallback import _Light, _consciousness, _run, _shortage, light_route

MAIN = shared.MIND
WAKE = {"kind": "mind", "task_id": "wake1", "focus": {"role": "consciousness", "task_id": "wake1"}}
HELPER = {"kind": "helper", "writer": "fallback_page", "attribution": "helper draft, not lived"}


@pytest.fixture
def light(monkeypatch):
    return light_route(monkeypatch)


def _covers(root, first, last, room="1"):
    from ouroboros.tools.chronicle import page_covers

    addresses = {pos: address for address, _row, pos in chat_chain.iter_rows(root)}
    return page_covers(root, room, from_addr=addresses[first], to_addr=addresses[last])["covers"]


def _page(root, first, last, author, **kwargs):
    return ChronicleStore(root).publish_page(room_id="1", text=f"rows {first}-{last}", covers=_covers(root, first, last),
                                             author=author, **kwargs)


def test_disjoint_pages_of_main_consciousness_and_the_helper_all_stand_without_a_head(tmp_path, monkeypatch, light):
    shared.world(tmp_path)
    _consciousness(monkeypatch, False)
    first = _page(tmp_path, 10, 11, MAIN)
    second = _page(tmp_path, 12, 12, WAKE)
    assert first.ok and second.ok, (first, second)
    run = _run(tmp_path, _Light(), trace=_shortage("1", 16, tmp_path))
    assert run.outcome == "published", run
    store = ChronicleStore(tmp_path)
    draft = store.get(run.record_id)
    assert draft["author"]["kind"] == "helper" and draft["covers"]["stream_span"] == [15, 16]
    assert [record["status"] for record in store.pages_of_room("1")] == ["final", "final", "draft"]


def test_an_overlapping_page_is_already_sealed_and_names_the_page_that_holds_the_rows(tmp_path):
    shared.world(tmp_path)
    held = _page(tmp_path, 10, 11, MAIN)
    refused = _page(tmp_path, 11, 12, WAKE)
    assert held.ok and (refused.ok, refused.reason) == (False, "already_sealed")
    assert refused.conflict_ids == (held.record["id"],)
    assert _page(tmp_path, 12, 12, WAKE).ok  # the rows nobody holds stay free


def test_a_part_on_a_stale_head_is_refused_with_the_current_head_and_stands_after_a_reread(tmp_path):
    shared.world(tmp_path)
    store = ChronicleStore(tmp_path)
    pages = [_page(tmp_path, 10, 11, MAIN).record["id"], _page(tmp_path, 12, 12, MAIN).record["id"]]
    seen = store.room_head("1")
    assert store.write_note(room_id="1", task_id="wake1", text="consciousness wrote meanwhile", author=WAKE).ok
    stale = store.publish_part(room_id="1", text="both pages", member_ids=pages, author=MAIN, expected_sequence=seen)
    assert (stale.ok, stale.reason) == (False, "revision_conflict")
    assert stale.current_head == store.room_head("1") > seen and stale.conflict_ids
    fresh = store.publish_part(room_id="1", text="both pages", member_ids=pages, author=MAIN,
                               expected_sequence=stale.current_head)
    assert fresh.ok, fresh


def test_a_helper_draft_holds_its_rows_until_the_mind_rejects_it(tmp_path, monkeypatch, light):
    shared.world(tmp_path)
    _consciousness(monkeypatch, False)
    run = _run(tmp_path, _Light(json.dumps({"text": "a helper draft", "quotes": []})),
               trace=_shortage("1", 12, tmp_path))
    assert run.outcome == "published", run
    refused = _page(tmp_path, 11, 12, MAIN)
    assert (refused.reason, refused.conflict_ids) == ("already_sealed", (run.record_id,))
    store = ChronicleStore(tmp_path)
    assert store.decide(run.record_id, False, MAIN, "the mind writes this stretch itself").ok
    assert _page(tmp_path, 10, 12, MAIN).ok


def test_main_and_consciousness_alternate_on_one_room(tmp_path):
    shared.world(tmp_path)
    store = ChronicleStore(tmp_path)
    head = store.room_head("1")
    assert _page(tmp_path, 10, 11, MAIN, expected_sequence=head).ok
    late = _page(tmp_path, 12, 12, WAKE, expected_sequence=head)  # the wake read the head before Main wrote
    assert (late.reason, late.current_head) == ("revision_conflict", store.room_head("1"))
    assert _page(tmp_path, 12, 12, WAKE, expected_sequence=late.current_head).ok
    assert _page(tmp_path, 15, 16, MAIN).ok  # a page may also go without a head; the sets decide
    helper = store.publish_page(room_id="1", text="a helper draft", covers=_covers(tmp_path, 16, 18), author=HELPER)
    assert (helper.reason, len(helper.conflict_ids)) == ("already_sealed", 1)
