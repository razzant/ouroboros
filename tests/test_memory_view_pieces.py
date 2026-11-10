"""Pieces of the memory view: each local form is made once for a snapshot, and the whole text is their join."""
from __future__ import annotations

import random

import pytest

from ouroboros import memory_floor as mf
from ouroboros import memory_view as mv
from ouroboros import memory_view_conversations as conversations
from ouroboros.context_timing import measured_sections, section_spans
from tests import _memory_view_synthetic as syn


def _snapshot() -> mv.MemoryViewSnapshot:
    lane = []
    for i in range(40):
        kind = "human" if i % 5 == 0 else "ouroboros"
        lane.append({**syn.spoken(i, kind, 200 + i), "owner_side": kind == "human", "carried": bool(i % 10)})
    room = syn.room("1", lane1=lane, lane2=12, notes=3, under=2)
    groups = [syn.conversation(i, notes=i % 3 == 0, words=4 if i % 4 else 2, word_chars=80, owner=i % 7 == 0)
              for i in range(12)]
    return syn.snapshot(room_facts=room, conversations=groups, story=syn.pages(12, 300) + syn.pointers(4, 3), mark_count=2)


def _levels(snapshot: mv.MemoryViewSnapshot, count: int = 25) -> list:
    table, rnd = mf._elements(snapshot), random.Random(11)
    pairs = [(step, ident) for step, ids in table.ids.items() for ident in ids]
    levels = [mv.FULL_VIEW]
    for _ in range(count):
        taken = set(rnd.sample(pairs, rnd.randint(1, len(pairs))))
        taken -= {("F5l", i) for i in table.ids.get("F5l", ()) if ("F5", i) in taken}
        for form, priors in mf._LIST_FORMS.items():
            if not any(step in priors for step, _ in taken):
                taken -= {(form, i) for i in table.ids.get(form, ())}
        levels.append(table.level(taken))
    return levels


def test_conversation_pieces_join_to_the_plain_text_at_every_level():
    """The pieces of ``## Other conversations`` are its text: at any level the join through a snapshot's pieces is
    exactly the plain render, so the memo cannot change what the mind reads."""
    snapshot = _snapshot()
    for level in _levels(snapshot):
        gone = dict(level.addressed)
        assert conversations.render(snapshot.conversations, gone, piece=snapshot.piece) == conversations.render(
            snapshot.conversations, gone)


def test_a_repeated_level_makes_no_new_piece_and_a_new_form_makes_only_its_own():
    """Inside a measured window a group's form is rendered once per key: drawing the same level again adds no piece,
    a level that changes one group's form makes only that group's new piece, and nothing repeats."""
    snapshot, level_gone = _snapshot(), {"F6w": []}
    with measured_sections():
        conversations.render(snapshot.conversations, level_gone, piece=snapshot.piece)
        first = section_spans()["work"]["view_pieces"]
        conversations.render(snapshot.conversations, level_gone, piece=snapshot.piece)
        assert section_spans()["work"]["view_pieces"] == first
        word = snapshot.conversations[0]["rooms"][0]["words"][0]["id"]
        conversations.render(snapshot.conversations, {"F6w": [word]}, piece=snapshot.piece)
        work = section_spans()["work"]
    assert work["view_pieces"] == first + 1 and not work.get("view_pieces_again")


def test_without_the_memo_the_counter_names_the_repeat(monkeypatch):
    """The guard's other direction: drawing the same forms through a piece that keeps nothing counts them again."""
    snapshot = _snapshot()

    def unkept(key, make):
        from ouroboros.context_timing import counted

        counted("view_pieces", key=(id(snapshot), key))
        return make(), 0

    with measured_sections():
        conversations.render(snapshot.conversations, {}, piece=unkept)
        conversations.render(snapshot.conversations, {}, piece=unkept)
        work = section_spans()["work"]
    assert work["view_pieces_again"] == work["view_pieces"] / 2


def test_pieces_never_reach_the_snapshot_s_canonical_form():
    """The memo is a non-field attribute: the snapshot's JSON and equality do not see it, a replaced snapshot has its own."""
    import dataclasses

    snapshot = _snapshot()
    before = mv.snapshot_json(snapshot)
    conversations.render(snapshot.conversations, {}, piece=snapshot.piece)
    assert mv.snapshot_json(snapshot) == before and "_pieces" in snapshot.__dict__
    assert "_pieces" not in dataclasses.replace(snapshot).__dict__


@pytest.mark.parametrize("lined, counted", [(0, False), (1, False), (2, True)])
def test_the_counted_rung_and_the_group_lines_join_like_the_plain_text(lined, counted):
    """The last rung's block and the group lines are pieces too: the join stays byte-equal to the plain render."""
    snapshot = _snapshot()
    keys = [group["key"] for group in snapshot.conversations][:lined]
    gone = {**({"F6g": keys} if keys else {}), **({"F6a": ["conversations"]} if counted else {})}
    assert conversations.render(snapshot.conversations, gone, piece=snapshot.piece) == conversations.render(
        snapshot.conversations, gone)


@pytest.mark.parametrize("note", ["", "### Physical floor\nThis view keeps \"quotes\" and a \\ backslash."])
def test_a_level_is_priced_by_its_pieces_exactly_as_its_drawn_text(note):
    """The price of a level is the sum of its pieces and the separators that join them: at any level, with or without the
    floor note, ``story_chars`` and ``room_chars`` are the JSON characters of the drawn texts. A piece whose key forgets
    what its text reads would differ here."""
    snapshot = _snapshot()
    for level in _levels(snapshot):
        assert mv.story_chars(snapshot, level) == mv.json_chars(mv.render_story(snapshot, level))
        assert mv.room_chars(snapshot, level, floor_note=note) == mv.json_chars(mv.render_room(snapshot, level, floor_note=note))


def test_an_unactivated_snapshot_is_priced_without_the_floor_note():
    """Before activation the room is one line and the note is not added: the price says the same as the text."""
    import dataclasses

    snapshot = _snapshot()
    inactive = dataclasses.replace(snapshot, store_status={"state": "importing", "reason": "not yet"})
    assert mv.room_chars(inactive, mv.FULL_VIEW, floor_note="### Physical floor") == mv.json_chars(
        mv.render_room(inactive, mv.FULL_VIEW, floor_note="### Physical floor"))
    assert "### Physical floor" not in mv.render_room(inactive, mv.FULL_VIEW, floor_note="### Physical floor")


def test_a_view_fit_is_drawn_only_when_read():
    """A frame the fit walks away from is priced, never drawn: ``ViewFit`` draws its texts on the first read."""
    snapshot = _snapshot()
    fit = mf.ViewFit(snapshot, mv.FULL_VIEW, "", {})
    with measured_sections():
        assert fit.story_chars and fit.room_chars
        assert not section_spans()["work"].get("view_renders")
        story, room = fit.story, fit.room
        assert section_spans()["work"]["view_renders"] == 2
    assert (story, room) == (mv.render_story(snapshot), mv.render_room(snapshot)) and fit.facts == {}
