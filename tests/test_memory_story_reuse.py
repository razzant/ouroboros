"""Exact reuse of the story capture (cost law L13): one capture per exact source identity, byte for byte.

``memory_view.capture_story`` keeps the last story of a chronicle with what it read (journal size, legacy units,
room labels, page writers' later terminal rows, helper route) and reuses it only while every source answers as it
did. Each law compares the reused capture with a fresh full capture of the same state, byte for byte, after a
change that moves the story (so a stale reuse would differ), and counts the captures an unchanged state costs.
"""
from __future__ import annotations

import json

import pytest

from ouroboros import chat_chain, chronicle_store
from ouroboros import memory_view as mv
from ouroboros.chronicle_store import ChronicleStore
from tests import _memory_inventory_shared as shared

MAIN = {"id": "turn0001", "chat_id": 1}
KID = {"id": "kid00001", "chat_id": 1, "delegation_role": "subagent"}
WRITER = {"kind": "mind", "task_id": "w1", "focus": {"role": "root", "task_id": "w1"}}


@pytest.fixture
def captures(monkeypatch):
    """Count the full story captures (the reads a reuse spares)."""
    calls = []
    original = mv._capture_story

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(mv, "_capture_story", counted)
    mv._STORIES.clear()
    yield calls
    mv._STORIES.clear()


def _story(root):
    return mv.capture_story(ChronicleStore(root), root, mv.room_labeler(root))


def _story_bytes(root):
    """The story as captured and as told in every form the floor can choose (``render_story`` levels)."""
    story, status, refusals = _story(root)
    snapshot = mv.MemoryViewSnapshot(spec=mv.ROLE_DEFAULTS["integrator"], store_status={"state": "active"},
                                     frontier={}, story=tuple(story), legacy_blocks=status,
                                     fallback_refusals=tuple(refusals))
    pages = tuple(entry["id"] for entry in story if entry["kind"] != "legacy")
    forms = (mv.FULL_VIEW, mv.FloorLevel(addressed=(("F3", tuple(e["room_id"] for e in story)),)),
             mv.FloorLevel(addressed=(("F5", pages),)), mv.FloorLevel(addressed=(("F5l", pages),)))
    return (json.dumps([story, status, refusals], sort_keys=True, ensure_ascii=False),
            *(mv.render_story(snapshot, form) for form in forms))


def _view(root, task=MAIN):
    return mv.snapshot_json(mv.capture_memory_view(root, task, mv.view_spec_for_task(task, root)))


def _fresh(read, *args):
    held = dict(mv._STORIES)
    mv._STORIES.clear()
    try:
        return read(*args)
    finally:
        mv._STORIES.clear()
        mv._STORIES.update(held)


def _page(root, room, first, last, author=shared.MIND):
    from ouroboros.tools.chronicle import page_covers

    addresses = {pos: address for address, _row, pos in chat_chain.iter_rows(root)}
    covers = page_covers(root, room, from_addr=addresses[first], to_addr=addresses[last])["covers"]
    result = ChronicleStore(root).publish_page(room_id=room, text=f"Page {room} {first}-{last}.", covers=covers,
                                               author=author)
    assert result.ok, result
    return result.record["id"]


def _append(root, **row):
    shared.append(root / "logs" / "chat.jsonl", shared.msg("2026-09-04T00:00:00+00:00", "", **row))


def test_an_unchanged_state_reuses_one_capture_byte_for_byte(tmp_path, captures):
    """Every focus that carries a story reads one capture: the view's, a child's and a direct read alike."""
    shared.world(tmp_path)
    _page(tmp_path, "1", 10, 12, author=WRITER)
    main, kid, story = _view(tmp_path), _view(tmp_path, KID), _story_bytes(tmp_path)
    assert len(captures) == 1
    assert (main, kid, story) == (_fresh(_view, tmp_path), _fresh(_view, tmp_path, KID), _fresh(_story_bytes, tmp_path))
    assert len(captures) == 4  # the three fresh references alone captured again
    assert json.loads(main)["story"] == json.loads(kid)["story"]


def _rename(root, rooms):
    from ouroboros.projects_registry import list_reserved_projects, update_project

    project = next(p for p in list_reserved_projects(root) if p["chat_id"] == rooms["alpha"])
    assert update_project(root, project["id"], name="Alpha renamed")


@pytest.mark.parametrize("change", [
    "chat_row", "correction", "legacy_binding"])
def test_every_source_change_is_seen_and_the_reuse_equals_a_fresh_capture(tmp_path, captures, monkeypatch, change):
    """A change of each source moves the story (it differs from before), and the capture after it equals a fresh
    full capture byte for byte, story and whole view: a reuse never serves a state that no longer holds."""
    route = ["light-one"]
    monkeypatch.setattr("ouroboros.model_slots.get_light_model", lambda: route[0])
    rooms = shared.world(tmp_path)
    page = _page(tmp_path, "1", 10, 12, author=WRITER)
    alpha = _page(tmp_path, str(rooms["alpha"]), 13, 13)
    if change == "binding":  # the writer's later terminal row, then its task bound to another room
        _append(tmp_path, direction="system", type="task_summary", task_id="w1", status="completed")
    before, held = _story_bytes(tmp_path), len(captures)
    if change == "chat_row":  # a row of no page writer: no source of the story moves
        _append(tmp_path, client_message_id="later")
    elif change == "writer_terminal_row":
        _append(tmp_path, direction="system", type="task_summary", task_id="w1", status="completed")
    elif change == "seal":
        _page(tmp_path, str(rooms["beta"]), 14, 14)
    elif change == "fold":
        store = ChronicleStore(tmp_path)
        assert store.publish_part(room_id=str(rooms["alpha"]), text="A part.", member_ids=[alpha],
                                  author=shared.MIND, expected_sequence=store.room_head(str(rooms["alpha"]))).ok
    elif change == "correction":
        assert ChronicleStore(tmp_path).correct(page, "Corrected words.", shared.MIND).ok
    elif change == "selection":
        store = ChronicleStore(tmp_path)
        account = store.publish_account(room_id="1", text="My account.", sources=[page], author=shared.MIND)
        assert account.ok and store.select_account(account.record["id"], replaces=[page], author=shared.MIND,
                                                   reason="tells it").ok
    elif change == "label":
        _rename(tmp_path, rooms)
    elif change == "route":
        route[0] = "light-two"
    elif change == "binding":
        from ouroboros.projects_registry import bind_task_to_project

        bind_task_to_project(tmp_path, "w1", "beta", origin={"absent": "post_hoc_unresolved"})
    elif change == "legacy_binding":  # a row inside an old memory range changes room: only the legacy units move
        from ouroboros.projects_registry import bind_task_to_project

        bind_task_to_project(tmp_path, "t1", "beta", origin={"absent": "post_hoc_unresolved"})
    after = _story_bytes(tmp_path)
    assert (len(captures) == held) is (change == "chat_row")  # reused only while nothing the story reads moved
    assert after == _fresh(_story_bytes, tmp_path)
    assert _view(tmp_path) == _fresh(_view, tmp_path)
    assert (after == before) is (change == "chat_row")


def test_a_proposals_capture_neither_reuses_nor_keeps_the_stored_story(tmp_path, captures):
    """``capture(proposed=…)`` reads an unsaved selection at the stored journal size: its story is its own."""
    shared.world(tmp_path)
    page = _page(tmp_path, "1", 10, 12)
    store = ChronicleStore(tmp_path)
    account = store.publish_account(room_id="1", text="My account.", sources=[page], author=shared.MIND)
    assert account.ok
    stored = _story(tmp_path)
    with chronicle_store.capture(tmp_path, tail="terminal"):
        proposal = store.select_account(account.record["id"], replaces=[page], author=shared.MIND, reason="r",
                                        propose=True)
        assert proposal.ok and proposal.reason == "proposed"
        with chronicle_store.capture(tmp_path, proposed=proposal.record):
            proposed, again = _story(tmp_path), _story(tmp_path)
    assert len(captures) == 3  # each proposal capture is whole
    assert proposed == again != stored and any(entry.get("told_by") for entry in proposed[0])
    assert _story(tmp_path) == stored and len(captures) == 3  # the stored story, kept and reused


def test_a_capture_outside_one_pins_its_own_state(tmp_path, captures):
    """``memory_read``'s story and ``_read_story`` may call it outside a capture: one pinned state answers all."""
    shared.world(tmp_path)
    _page(tmp_path, "1", 10, 12, author=WRITER)
    assert ChronicleStore(tmp_path).stored_state() is None
    first = _story(tmp_path)
    assert _story(tmp_path) == first and len(captures) == 1


def test_view_receipts_keep_story_tokens_and_live_room_coverage(tmp_path, captures):
    """E8: the ``context_memory_view`` receipt of a build that reused the story (story tokens, room tokens, live
    rooms, marks) equals the receipt of a fresh build of the same state."""
    from ouroboros import memory_floor

    shared.world(tmp_path)
    _page(tmp_path, "1", 10, 12, author=WRITER)
    _story(tmp_path)  # the story a later build reuses

    def receipt():
        snapshot = mv.capture_memory_view(tmp_path, MAIN, mv.view_spec_for_task(MAIN, tmp_path))
        return memory_floor.view_receipt(snapshot, mv.render_story(snapshot), mv.render_room(snapshot), {})

    reused = receipt()
    assert len(captures) == 1
    assert reused == _fresh(receipt)
    assert reused["blocks"]["story_tokens"] > 0 and reused["blocks"]["conversations"] > 0
