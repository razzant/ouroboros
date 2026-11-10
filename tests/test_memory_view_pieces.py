"""Pieces of the memory view: each local form is priced once for a snapshot, only its price stays, and a level's price
is what its drawn text costs."""
from __future__ import annotations

import dataclasses
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


def _priced(snapshot: mv.MemoryViewSnapshot, gone, piece=None) -> list:
    """Each part of ``## Other conversations`` at ``gone`` through the snapshot's pieces, or through another ``piece``."""
    return conversations.pieces(list(snapshot.conversations), gone, piece or snapshot.piece)


def _costs_as_drawn(snapshot: mv.MemoryViewSnapshot, gone) -> bool:
    prices = _priced(snapshot, gone)
    text = syn.conversations_text(dataclasses.replace(snapshot).conversations, gone)
    return sum(prices) + 4 * (len(prices) - 1) == mv.json_chars(text)  # JSON spells the blank line in four characters


def test_conversation_pieces_price_the_drawn_text_at_every_level():
    """The price of ``## Other conversations`` by its pieces is what its text costs, at any level: a group's key that
    forgets something its form reads would answer another level's price here."""
    snapshot = _snapshot()
    for level in _levels(snapshot):
        assert _costs_as_drawn(snapshot, dict(level.addressed))


def test_a_repeated_level_makes_no_new_piece_and_a_new_form_makes_only_its_own():
    """Inside a measured window a group's form is rendered once per key: pricing the same level again adds no piece,
    a level that changes one group's form makes only that group's new piece, and nothing repeats."""
    snapshot, level_gone = _snapshot(), {"F6w": []}
    with measured_sections():
        _priced(snapshot, level_gone)
        first = section_spans()["work"]["view_pieces"]
        _priced(snapshot, level_gone)
        assert section_spans()["work"]["view_pieces"] == first
        word = snapshot.conversations[0]["rooms"][0]["words"][0]["id"]
        _priced(snapshot, {"F6w": [word]})
        work = section_spans()["work"]
    assert work["view_pieces"] == first + 1 and not work.get("view_pieces_again")


def test_without_the_memo_the_counter_names_the_repeat():
    """The guard's other direction: pricing the same forms through a piece that keeps nothing counts them again."""
    from ouroboros.context_timing import counted

    snapshot = _snapshot()

    def unkept(key, make):
        counted("view_pieces", key=(id(snapshot), key))
        return mv.json_chars(make())

    with measured_sections():
        _priced(snapshot, {}, unkept)
        _priced(snapshot, {}, unkept)
        work = section_spans()["work"]
    assert work["view_pieces_again"] == work["view_pieces"] / 2


def test_pieces_never_reach_the_snapshot_s_canonical_form():
    """The memo is a non-field attribute: the snapshot's JSON and equality do not see it, a replaced snapshot has its own."""
    snapshot = _snapshot()
    before = mv.snapshot_json(snapshot)
    _priced(snapshot, {})
    assert mv.snapshot_json(snapshot) == before and "_pieces" in snapshot.__dict__
    assert "_pieces" not in dataclasses.replace(snapshot).__dict__


def test_a_snapshot_keeps_the_price_of_a_form_and_no_text():
    """After a level is walked and delivered the snapshot holds a number per form it priced: no text of a form the
    floor priced and left (the walk prices far more forms than it delivers) outlives its price."""
    snapshot = _snapshot()
    level = mf.fit_memory_view(snapshot, {"margin": 0, "physical": 0, "budget": None})
    fit = mf.ViewFit(snapshot, level, "", {})
    assert fit.story and fit.room
    held = snapshot.__dict__["_pieces"]
    assert len(held) > 2 and all(isinstance(price, int) for price in held.values()), held


@pytest.mark.parametrize("lined, counted", [(0, False), (1, False), (2, True)])
def test_the_counted_rung_and_the_group_lines_cost_their_drawn_text(lined, counted):
    """The last rung's block and the group lines are pieces too: their prices are the drawn text's."""
    snapshot = _snapshot()
    keys = [group["key"] for group in snapshot.conversations][:lined]
    assert _costs_as_drawn(snapshot, {**({"F6g": keys} if keys else {}), **({"F6a": ["conversations"]} if counted else {})})


@pytest.mark.parametrize("note", ["", "### Physical floor\nThis view keeps \"quotes\" and a \\ backslash."])
def test_a_level_is_priced_by_its_pieces_exactly_as_its_drawn_text(note):
    """The price of a level is the sum of its pieces and the separators that join them: at any level, with or without the
    floor note, ``story_chars`` and ``room_chars`` are the JSON characters of the texts a snapshot without pieces draws.
    A piece whose key forgets a step its text reads answers another level's price here."""
    snapshot = _snapshot()
    fresh = dataclasses.replace(snapshot)
    for level in _levels(snapshot):
        assert mv.story_chars(snapshot, level) == mv.json_chars(mv.render_story(fresh, level))
        assert mv.room_chars(snapshot, level, floor_note=note) == mv.json_chars(mv.render_room(fresh, level, floor_note=note))


def test_a_level_holds_the_whole_story_only_when_its_text_is_the_whole_story():
    """Equal prices alone prove nothing for a level naming story steps: those texts decide, and the whole view needs no text."""
    snapshot = _snapshot()
    table = mf._elements(snapshot)
    story_step = next(step for step in mv.STORY_STEPS if table.ids.get(step))
    assert mv.whole_story_at(snapshot, mv.FULL_VIEW)
    assert not mv.whole_story_at(snapshot, table.level({(story_step, table.ids[story_step][0])}))
    assert mv.whole_story_at(snapshot, mv.FloorLevel(addressed=(("F5", ("no-such-page",)),)))


def test_an_unactivated_snapshot_is_priced_without_the_floor_note():
    """Before activation the room is one line and the note is not added: the price says the same as the text."""
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
