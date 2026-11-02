"""What a standing part or a selected account tells in place of its records, in the story and in their own room.

A standing part (the mind's, or a helper's draft not pending as a candidate) leaves its members' address and period
in the room; a shown account leaves out every record it tells, a part's members and replaced legacy included. Taking
either back returns the text; a correction is shown once.
"""
from __future__ import annotations

from ouroboros import memory_view as mv
from ouroboros.chronicle_store import ChronicleStore
from ouroboros.tools.chronicle import page_covers
from tests import _memory_inventory_shared as shared
from tests.test_memory_view_room import MAIN_TASK, _addr, _install, _view

HELPER = {"kind": "helper", "route": "light"}
NANNY = {"id": "nan9", "chat_id": 1, "delegation_role": "subagent", "root_task_id": "root1",
         "configured_subagent": {"route": {"kind": "agent_session"}}}


def _pages(root, rows):
    store = ChronicleStore(root)
    ids = []
    for first, last, text in ((0, 1, "FIRST PAGE TEXT"), (2, 3, "SECOND PAGE TEXT")):
        covers = page_covers(root, "1", from_addr=_addr(rows[first]), to_addr=_addr(rows[last]))["covers"]
        result = store.publish_page(room_id="1", text=text, covers=covers, author=shared.MIND)
        assert result.ok, result
        ids.append(result.record["id"])
    return store, ids


def _part(store, ids, author=shared.MIND, **kwargs):
    result = store.publish_part(room_id="1", text="THE PART'S RETELLING", member_ids=ids, author=author,
                                expected_sequence=store.room_head("1"), **kwargs)
    assert result.ok, result
    return result.record["id"]


def _memory(root, task=MAIN_TASK):
    snapshot, room = _view(root, task)
    return snapshot, mv.render_story(snapshot), room


def test_a_standing_part_leaves_its_members_address_and_period_in_their_own_room(tmp_path):
    _rooms, rows = _install(tmp_path)
    store, ids = _pages(tmp_path, rows)
    part = _part(store, ids)
    snapshot, story, room = _memory(tmp_path)
    assert "THE PART'S RETELLING" in story and "FIRST PAGE TEXT" not in story + room and "SECOND PAGE TEXT" not in room
    for item in snapshot.room["under_parts"]:
        assert item["standing"] is True
        assert (f"#### page {item['id']} — {item['period']} — under part {part} — {len(item['text'])} chars — "
                f"memory_read(node_id='{item['id']}')") in room
    # The floor offers no saving it cannot make: these members are already short.
    from ouroboros import memory_floor

    assert not [e for e in memory_floor.floor_elements(snapshot) if e[0] == "F4" and e[1] in ids]
    # A correction to a member stands once: beside the part in the story, never again in the room.
    assert store.correct(ids[0], "A CORRECTION AFTER THE PART", shared.MIND).ok
    _snapshot, story, room = _memory(tmp_path)
    assert (story + room).count("A CORRECTION AFTER THE PART") == 1 and "correction by me of" in story


def test_a_candidate_part_or_a_rejected_draft_keeps_the_members_text(tmp_path):
    _rooms, rows = _install(tmp_path)
    store, ids = _pages(tmp_path, rows)
    candidate = _part(store, ids, author=HELPER, metadata={"candidate": True})
    snapshot, _story, room = _memory(tmp_path)
    assert all("standing" not in item for item in snapshot.room["under_parts"]) and "FIRST PAGE TEXT" in room
    assert store.decide(candidate, False, shared.MIND, "Not this retelling").ok
    draft = _part(store, ids, author=HELPER)  # an acting draft (consciousness off) stands until rejected
    _snapshot, _story, room = _memory(tmp_path)
    assert "FIRST PAGE TEXT" not in room and f"under part {draft}" in room
    assert store.decide(draft, False, shared.MIND, "Wrong").ok
    snapshot, story, room = _memory(tmp_path)
    assert "FIRST PAGE TEXT" in story and "SECOND PAGE TEXT" in story and not snapshot.room["under_parts"]


def test_a_nanny_keeps_its_rooms_pages_whole_under_the_floor(tmp_path):
    _rooms, rows = _install(tmp_path)
    store, ids = _pages(tmp_path, rows)
    _part(store, ids)
    nanny, room = _view(tmp_path, NANNY)
    assert all("standing" not in item for item in nanny.room["under_parts"])
    assert "FIRST PAGE TEXT" in room and "THE PART'S RETELLING" in room


def test_a_shown_account_tells_a_parts_members_and_replaced_legacy_and_hiding_it_returns_them(tmp_path):
    _rooms, rows = _install(tmp_path)
    store, ids = _pages(tmp_path, rows)
    part = _part(store, ids)
    legacy = next(r["id"] for r in store.room_records("1", kinds=("legacy",)))
    account = store.publish_account(room_id="1", text="MY ACCOUNT ACROSS ROOMS", sources=[part, legacy],
                                    author=shared.MIND)
    assert account.ok, account
    for shown in (True, False):
        assert store.select_account(account.record["id"], replaces=[part, legacy], author=shared.MIND,
                                    reason="the account says it", shown=shown).ok
        snapshot, story, room = _memory(tmp_path)
        if shown:
            assert "MY ACCOUNT ACROSS ROOMS" in story and "THE PART'S RETELLING" not in story
            assert not snapshot.room["under_parts"] and not snapshot.room["legacy"]
            assert "Old Main talk." not in story + room
        else:
            assert "MY ACCOUNT ACROSS ROOMS" not in story and "THE PART'S RETELLING" in story
            assert [item["id"] for item in snapshot.room["under_parts"]] == ids and "Old Main talk." in story
        # A nanny carries no story, so no account stands in for its room's pages there.
        assert "FIRST PAGE TEXT" in _view(tmp_path, NANNY)[1]
