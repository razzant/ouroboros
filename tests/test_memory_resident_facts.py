"""What my own resident parts cost, as facts (CONSENSUS_v3 K9, owner 1A of 26.11.2026: marks stay whole and I tidy them
myself, so I must see their sizes). The integrating start's facts row names the marks block, the selected accounts and
this room's block of the view its fit drew, with the date and basis of that measurement; a part the view does not draw
is named, never printed as a number; the ``memory_mark`` receipt states the marks' total after the change, equal to
what the next view renders. No advice, threshold or action rides on any number."""
from __future__ import annotations

import dataclasses
import re
from types import SimpleNamespace

import pytest

from ouroboros import memory_floor as mf, memory_view as mv
from ouroboros.context_view_facts import RESIDENT_BASIS, memory_facts, resident_marks
from tests import _memory_scale_fixture as scale

_RESIDENT = re.compile(r"; resident here, measured (\d{4}-\d\d-\d\d \d\d:\d\d) UTC \((?P<basis>[^)]*)\): (?P<parts>[^;|]*)")


def _mature_plan(tmp_path, monkeypatch):
    from ouroboros import context, context_fit as cf, llm

    monkeypatch.setattr(cf, "main_output_reserve_tokens", lambda **_kw: 16_384)
    monkeypatch.setattr(llm.LLMClient, "chat", lambda *_a, **_kw: pytest.fail("no model call"))
    env, memory = scale.env_on(scale.installed(tmp_path, "mature"))
    return context.build_context_fit_plan(env, memory, {"id": "mature-turn", "type": "task", "chat_id": 1, "text": "hi",
                                                        "_is_direct_chat": True}, preferred_mode="max",
                                          tool_schemas=scale.tools(),
                                          route_resolver=scale.route_resolver(2_000_000, "provider", None))


def _pieces(snapshot, level):
    """The room block's pieces as the renderer itself draws them at ``level`` (``memory_view._room_parts``)."""
    pieces = {}
    mv._room_parts(snapshot, dict(level.addressed), lambda key, make: pieces.setdefault(key[0], make()))
    return pieces


def test_the_facts_row_names_what_my_marks_accounts_and_room_cost_on_a_mature_install(tmp_path, monkeypatch):
    plan = _mature_plan(tmp_path, monkeypatch)
    line = memory_facts(plan, plan.initial_mode)
    found = _RESIDENT.search(line)
    assert found and found["basis"] == RESIDENT_BASIS, line
    snapshot = mv.snapshot_from_json(plan.core.memory_view_json)
    level = mf.kept_level(snapshot, plan.projection(plan.initial_mode).memory_facts)
    pieces, accounts = _pieces(snapshot, level), [e for e in snapshot.story if e["kind"] == "account"]
    marks = [m for m in snapshot.marks if not m.get("late_review")]
    assert len(marks) == 158 and len(accounts) == 5
    parts = found["parts"]
    assert f"marks block {mf.tokens_of(mv.json_chars(pieces['marks'])):,} (158 marks)" in parts
    assert f"this room's block {mf.tokens_of(mv.json_chars(pieces['room'])):,}" in parts
    # Each account's own lines as the renderer's parts name them (its words, readers, told records and changes).
    from ouroboros.memory_form_facts import memory_composition

    drawn = [p for p in memory_composition(snapshot, level)["blocks"]["story"]["parts"] if p["kind"] == "account"]
    assert len(drawn) == 5
    told = sum(p["text_json_chars"] + 2 * p["line_joins"] for p in drawn)
    assert f"selected accounts {mf.tokens_of(told):,} (5)" in parts
    assert not re.search(r"\b(too|over|large|warn|should|consider|release)\b", found.group(0))  # facts, no advice
    # Measured once per captured view: a later row of the same core carries the same words and date, so it is no new
    # fact every turn and an intact row in view keeps it out of the next one.
    assert memory_facts(plan, plan.initial_mode) == line


def test_a_part_the_view_does_not_draw_is_named_and_never_a_number():
    """Both directions: an inactive memory says its parts were not measured and prints no size; a view that draws no
    story names no accounts and one that draws no room names no room block; an active view with marks prints them."""
    from ouroboros.context_view_facts import _resident_text
    from tests import _memory_view_synthetic as syn

    active = syn.snapshot(room_facts=syn.room("1"), story=syn.pages(2, 400), mark_count=3)
    shown = _resident_text(active, mv.FULL_VIEW)
    assert "marks block" in shown and "(3 marks)" in shown and "selected accounts none" in shown and "this room's block" in shown
    inactive = dataclasses.replace(active, store_status={"state": "journal_unreadable", "reason": "x"})
    told = _resident_text(inactive, mv.FULL_VIEW)
    assert "not measured" in told and not re.search(r"\d", told.split("(")[0]) and "marks block" not in told
    bare = dataclasses.replace(active, spec=dataclasses.replace(active.spec, story=False, room_page=False, room_lanes=False,
                                                                origin_words=False))
    plain = _resident_text(bare, mv.FULL_VIEW)
    assert "selected accounts" not in plain and "this room's block" not in plain and "marks block" in plain


def test_the_mark_receipt_states_the_marks_total_the_next_view_renders(tmp_path):
    """Both directions: after a mark and after a release the receipt's total is the marks block the next capture of
    this focus's view renders, its count the marks there; a focus without a captured view gets no number."""
    import json

    from ouroboros.tools.chronicle import _memory_mark
    from tests.test_common_story import _ctx, _rooms

    _world, _alpha, _beta, a1, _b1, _h1 = _rooms(tmp_path)
    task = {"id": "turn0001", "chat_id": 1}
    spec = mv.view_spec_for_task(task, tmp_path)
    ctx = _ctx(tmp_path)
    assert json.loads(_memory_mark(ctx, text="No captured view.", node_id=a1))["resident_marks"]["measured"] is False

    def marks_now():
        snapshot = mv.capture_memory_view(tmp_path, task, spec)
        ordinary = [m for m in snapshot.marks if not m.get("late_review")]
        return len(ordinary), mf.tokens_of(mv.json_chars(mv._marks_text(ordinary))) if ordinary else 0

    ctx.context_fit_plan = SimpleNamespace(core=SimpleNamespace(memory_view_json=mv.snapshot_json(
        mv.capture_memory_view(tmp_path, task, spec))))
    added = json.loads(_memory_mark(ctx, text="Keep the count before promising. " * 6, node_id=a1, scope="global"))
    assert added["ok"] and added["resident_marks"]["basis"].startswith(RESIDENT_BASIS)
    count, tokens = marks_now()
    assert (added["resident_marks"]["marks"], added["resident_marks"]["tokens"]) == (count, tokens) and count == 2
    released = json.loads(_memory_mark(ctx, release_ids=[added["mark_id"]], reason="Said once is enough."))
    count, tokens = marks_now()
    assert released["ok"] and (released["resident_marks"]["marks"], released["resident_marks"]["tokens"]) == (count, tokens)
    assert count == 1 and resident_marks(tmp_path, None)["measured"] is False


@pytest.mark.parametrize("failure", ["DatabaseError", "OperationalError", "undecodable view"])
def test_a_failed_read_after_a_saved_mark_keeps_its_receipt(tmp_path, monkeypatch, failure):
    """pr2rev1 G16: the receipt's total is read after the mark is committed, so a failed read of the store (any sqlite
    error, not only a busy lock) or of the captured view never turns the saved mark into a failed call: exactly one
    mark is added and the receipt says the size was not measured. A clean read gives the total (the other direction)."""
    import json
    import sqlite3

    from ouroboros import context_view_facts
    from ouroboros.tools.chronicle import _memory_mark
    from tests.test_common_story import _ctx, _rooms

    _world, _alpha, _beta, a1, _b1, _h1 = _rooms(tmp_path)
    task = {"id": "turn0001", "chat_id": 1}
    spec = mv.view_spec_for_task(task, tmp_path)
    ctx = _ctx(tmp_path)
    ctx.context_fit_plan = SimpleNamespace(core=SimpleNamespace(memory_view_json=mv.snapshot_json(
        mv.capture_memory_view(tmp_path, task, spec))))
    before = len(mv.capture_memory_view(tmp_path, task, spec).marks)

    def failed(*_a, **_k):
        raise getattr(sqlite3, failure, ValueError)("disk I/O error")
    with monkeypatch.context() as patch:
        patch.setattr(context_view_facts if failure == "undecodable view" else mv,
                      "memory_snapshot" if failure == "undecodable view" else "_capture_marks", failed)
        outcome = json.loads(_memory_mark(ctx, text="Receipt probe", node_id=a1, scope="global"))
    assert len(mv.capture_memory_view(tmp_path, task, spec).marks) == before + 1
    assert outcome["ok"] and outcome["mark_id"] and outcome["resident_marks"]["measured"] is False
    assert outcome["resident_marks"]["why"].startswith("the marks could not be read: ")
    clean = json.loads(_memory_mark(ctx, text="Second probe", node_id=a1, scope="global"))
    assert clean["resident_marks"]["marks"] == before + 2 and clean["resident_marks"]["tokens"] > 0
