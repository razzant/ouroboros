"""The physical floor of the memory view (``ouroboros.memory_floor``).

Without a shortage the view is whole; with one, elements become address lines strictly
down the ladder F1, F1b, F3, F5, F4, F2, F6, F7, my replies and people's words last and only
against the window minus the reply reserve. Every element the floor takes is still named by its
address with its period, and what never degrades stays. Each rule is pinned in both
directions on synthetic snapshots of capture shape (377 pointers in 78 rooms, a dozen or
twenty live rooms, lanes of real size) and once on a captured installation.
"""
from __future__ import annotations

import ast
import dataclasses
import pathlib
import re

import pytest

from ouroboros import context_budget as cb
from ouroboros import memory_floor as mf
from ouroboros import memory_view as mv
from tests import _memory_inventory_shared as shared
from tests import _memory_view_synthetic as syn

REPO = pathlib.Path(__file__).resolve().parents[1]
NONE = {"margin": None, "physical": None, "budget": None}


def _tokens(snapshot, level=mv.FULL_VIEW):
    return mf.view_tokens(mv.render_story(snapshot, level)) + mf.view_tokens(mv.render_room(snapshot, level))


def _rich(**room_overrides):
    """Every step has elements: pages, pointers, live rooms with words, this room's page and both lanes."""
    lane = [syn.spoken(0, "human", 900), syn.spoken(1, "ouroboros", 6_000), syn.spoken(2, "human", 700),
            syn.spoken(3, "ouroboros", 2_000)]
    facts = {"legacy": 4, "legacy_chars": 4_000, "under": 2, "origins": 1, "lane1": lane, "lane2": 6, "notes": 1}
    facts.update(room_overrides)
    return syn.snapshot(room_facts=syn.room("1", **facts), story=syn.pointers(40, 8) + syn.pages(4, 3_000),
                        live=[syn.live_room(i, words=2) for i in range(3)] + [syn.live_room(3 + i) for i in range(4)]
                        + [syn.live_room(7, notes=1)],
                        mark_count=3, owner_words=syn.OWNER_WORDS, live_rooms="lines_with_words")


def _window(margin, physical=None):
    return {"margin": margin, "physical": margin if physical is None else physical, "budget": None}


def test_without_a_shortage_no_step_runs_and_the_view_is_whole():
    snapshot = _rich()
    full = _tokens(snapshot)
    level = mf.fit_memory_view(snapshot, _window(full))
    assert level == mv.FULL_VIEW and level.steps == ()
    assert mv.render_room(snapshot, level) == mv.render_room(snapshot)
    assert mv.render_story(snapshot, level) == mv.render_story(snapshot)
    # One token short is a shortage: the first ladder step answers.
    assert mf.fit_memory_view(snapshot, _window(full - 1)).steps[0][0] == "F1"


def test_an_unknown_window_takes_no_step_at_any_size():
    snapshot = syn.actor("main")
    assert mf.fit_memory_view(snapshot, NONE).steps == ()
    unknown = mf.floor_allowances(window_tokens=None, output_reserve_tokens=65_536, non_memory_tokens=10**6)
    assert unknown == NONE and mf.fit_memory_view(snapshot, unknown).steps == ()
    known = mf.floor_allowances(window_tokens=128_000, output_reserve_tokens=65_536, non_memory_tokens=50_000)
    assert mf.fit_memory_view(snapshot, known).steps  # the same view on a small known window does step down


def test_the_allowances_are_the_one_budget_frame_with_and_without_the_working_margins():
    frame = cb.request_context_budget(window_tokens=872_000, output_reserve_tokens=65_536, non_memory_tokens=300_000,
                                      margin_count=cb.MEMORY_VIEW_WORKING_MARGINS)
    allowances = mf.floor_allowances(window_tokens=872_000, output_reserve_tokens=65_536, non_memory_tokens=300_000)
    assert allowances["margin"] == frame["with_margin_tokens"] == 872_000 - 65_536 - 218_000 - 300_000
    assert allowances["physical"] == frame["without_margin_tokens"] == 872_000 - 65_536 - 300_000
    assert allowances["budget"] is None
    low = mf.floor_allowances(window_tokens=1_050_000, output_reserve_tokens=65_536, non_memory_tokens=0,
                              target_tokens=cb.OWNER_LOW_TARGET_TOKENS)
    assert low["budget"] == 250_000 - 65_536 - 31_250  # one landing level under the target


def test_steps_run_strictly_down_the_ladder_element_by_element():
    snapshot = _rich()
    elements = mf.degradable_elements(snapshot)
    assert [step for step in mf.LADDER if any(e[0] == step for e in elements)] == list(mf.LADDER)
    order = [step for step, *_rest in elements]
    assert order == sorted(order, key=mf.LADDER.index)  # grouped in ladder order
    full = _tokens(snapshot)
    seen = []
    for allowance in range(full, -1, -max(1, full // 60)):
        steps = mf.fit_memory_view(snapshot, _window(allowance)).steps
        names = [name for name, _n in steps]
        assert names == list(mf.LADDER[:len(names)]), (allowance, steps)  # always a prefix of the ladder
        for name, count in steps[:-1]:  # every earlier step is complete when a later one ran
            assert count == sum(1 for element in elements if element[0] == name), (allowance, steps)
        seen.append(tuple(names))
    assert seen[0] == () and seen[-1] == tuple(mf.LADDER)


def test_my_replies_and_peoples_words_answer_only_to_the_window_minus_the_reply_reserve():
    snapshot = _rich()
    full = _tokens(snapshot)
    # No room for the working margins, but the window minus the reserve holds the whole view:
    kept = mf.fit_memory_view(snapshot, _window(0, physical=full))
    names = [name for name, _n in kept.steps]
    assert names == ["F1", "F1b", "F3", "F5", "F4"]
    room = mv.render_room(snapshot, kept)
    for word in [w for live in snapshot.live_rooms for w in live["words"]] + list(snapshot.room["lane1"]):
        assert word["line"] in room  # both sides of the conversation stay verbatim
    # A window that cannot hold them even after F1-F5: my replies (F2), then F6, then F7, last.
    taken = mf.fit_memory_view(snapshot, _window(0, physical=0))
    assert [name for name, _n in taken.steps][-3:] == ["F2", "F6", "F7"]
    room = mv.render_room(snapshot, taken)
    assert all(item["line"] not in room and item["head"] in room for item in snapshot.room["lane1"])
    # Between the two: F6 (other rooms' people) goes before F7 (this room's people).
    without_f7 = _tokens(snapshot, mv.FloorLevel(tuple(pair for pair in taken.addressed if pair[0] != "F7")))
    partial = mf.fit_memory_view(snapshot, _window(0, physical=without_f7 + 1))
    assert dict(partial.steps).get("F6") and "F7" not in dict(partial.steps)


def test_f1b_folds_quiet_live_rooms_into_one_line_and_keeps_noted_ones_and_this_room():
    live = [syn.live_room(i, notes=1 if i in (4, 9, 15) else 0) for i in range(20)]
    snapshot = syn.snapshot(room_facts=syn.room("1", lane2=3, lane1=[syn.spoken(0, "human", 200)]), live=live)
    whole = mv.render_room(snapshot)
    assert all(f"### {room['label']} — open" in whole for room in live)
    level = mf.fit_memory_view(snapshot, _window(_tokens(snapshot) - 2_000))
    assert dict(level.steps).get("F1b") == 17
    text = mv.render_room(snapshot, level)
    quiet = [room["room_id"] for room in live if not room["notes"]]
    summary = next(line for line in text.split("\n") if line.startswith("17 more open rooms without notes; "))
    assert summary.endswith("reads each: " + ", ".join(quiet))
    assert "2026-08-29 00:00 → " in summary  # the oldest open row of the folded rooms
    for room in live:
        assert (f"### {room['label']} — open" in text) == bool(room["notes"])
    assert "## This room (Main) — head 7" in text


def test_f2_turns_my_longest_reply_into_an_address_first():
    giant, short = syn.spoken(0, "ouroboros", 40_000), syn.spoken(1, "ouroboros", 300)
    snapshot = syn.snapshot(room_facts=syn.room("1", lane1=[giant, syn.spoken(2, "human", 120), short]))
    level = mf.fit_memory_view(snapshot, _window(_tokens(snapshot) - 5_000))
    assert level.steps == (("F2", 1),)
    text = mv.render_room(snapshot, level)
    assert f"{giant['head']} (my reply, 40000 chars — read by address)" in text and giant["line"] not in text
    assert short["line"] in text  # the short reply stays verbatim
    # Ties: the oldest of two equally long replies goes first.
    twins = syn.snapshot(room_facts=syn.room("1", lane1=[syn.spoken(5, "ouroboros", 9_000), syn.spoken(4, "ouroboros",
                                                                                                   9_000)]))
    first = mf.fit_memory_view(twins, _window(_tokens(twins) - 100))
    assert first.addressed == (("F2", (syn.spoken(4, "ouroboros", 9_000)["address"],)),)


def _minutes(text):
    return sorted(set(re.findall(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}", text)), key=lambda s: s.replace("T", " "))


@pytest.mark.parametrize("step", mf.LADDER)
def test_every_element_a_step_takes_is_still_named_by_address_and_the_horizon_holds(step):
    snapshot = _rich()
    ids = tuple(ident for name, ident, *_rest in mf.degradable_elements(snapshot) if name == step)
    level = mv.FloorLevel(((step, ids),))
    before = mv.render_story(snapshot) + "\n" + mv.render_room(snapshot)
    after = mv.render_story(snapshot, level) + "\n" + mv.render_room(snapshot, level)
    assert after != before and len(after) < len(before)
    for ident in ids if step != "F1" else ids[:1]:  # a record id, a room id or a row address
        assert ident in after, ident
    if step == "F1":  # the task lines are named by the range that reads them: the oldest first row to the newest last
        newest = max(snapshot.room["lane2"], key=lambda item: item["last_pos"])
        assert f"from='{ids[0]}', to='{newest['last']}')" in after
    edges = lambda text: (_minutes(text)[0].replace("T", " "), _minutes(text)[-1].replace("T", " "))  # noqa: E731
    assert edges(after) == edges(before)  # nothing leaves the horizon; only the granularity changes


def test_a_retold_room_becomes_one_line_with_its_whole_period_and_every_id():
    snapshot = syn.snapshot(story=syn.pointers(10, 2))
    level = mv.FloorLevel((("F3", ("1000",)),))
    lines = mv.render_story(snapshot, level).split("\n")
    room_line = [line for line in lines if "retold records:" in line]
    assert room_line == [f"- {syn._label(1000)}; 2026-07-01 00:00 → 2026-07-24 23:00; 5 retold records: "
                         "legacy-b00-r1000, legacy-b01-r1000, legacy-b02-r1000, legacy-b03-r1000, legacy-b04-r1000; "
                         "memory_read(node_id=<id>) reads each"]
    assert sum("r1001'" in line for line in lines) == 5  # the other room keeps a line per record
    assert lines.index(room_line[0]) < next(i for i, line in enumerate(lines) if "r1001'" in line)


def test_what_never_degrades_stays_at_the_bottom_of_the_ladder():
    snapshot = _rich()
    level = mf.fit_memory_view(snapshot, {"margin": 0, "physical": 0, "budget": 0})
    assert [name for name, _n in level.steps] == list(mf.LADDER)
    story, room = mv.render_story(snapshot, level), mv.render_room(snapshot, level)
    assert story.startswith("## My story\n") and "Story status: the helper retelling is folded 0 of 23" in story
    assert room.startswith(syn.OWNER_WORDS)  # a helper's owner words
    for mark in snapshot.marks:
        assert f"memory_mark(release_id='{mark['id']}'" in room
    for live in snapshot.live_rooms:
        for note in live["notes"]:
            assert f"note {note['id']} by" in room and f"### {live['label']} — open" in room  # a noted room stays
    assert "### Words that started this work (retention-proof)" in room and "Start this project. ooo" in room
    assert "### My notes not yet sealed\nnote note-here-0 by root" in room
    assert "## This room (Main) — head 7" in room
    assert re.search(r"^- .*; \d+ retold records( in \d+ chars)?: .*reads each$", story, re.M)  # one line per retold room
    assert re.search(r"^\d+ more open rooms without notes; ", room, re.M)


def test_the_physical_floor_block_appears_exactly_when_a_step_past_f1_ran_or_the_mode_was_lowered():
    snapshot = _rich()
    note = lambda level, **kw: mf.floor_note(level, window_tokens=500_000, mode=kw.pop("mode", "max"), **kw)  # noqa: E731
    only_facts = mv.FloorLevel((("F1", ("a",)), ("F1b", ("b",))))
    assert note(mv.FULL_VIEW) == "" and note(only_facts) == ""
    for step in mf.LADDER[2:]:
        text = note(mv.FloorLevel((("F1", ("a",)), (step, ("x", "y")))))
        assert text.startswith("### Physical floor\nThis window (500000 tokens, Max) does not hold"), step
        assert "1 task fact lines" in text and "memory_read reads each" in text
    lowered = note(mv.FULL_VIEW, mode="low", lowered_from="max")
    assert lowered == ("### Physical floor\nThis window (500000 tokens) cannot hold Max with even the shortest view "
                       "of my memory; this task started in Low.")
    meta = ("compact_context", "list_available_tools", "enable_tools")  # the schemas an owner's Nano sends
    nano = note(mv.FloorLevel((("F7", ("x",)),)), mode="nano", tool_names=meta)
    assert nano.endswith("\n(memory_read is reachable through enable_tools)")
    assert "enable_tools" not in note(mv.FloorLevel((("F7", ("x",)),)), mode="low", tool_names=meta)
    # Placed at the end of this room, or of the live rooms when the view has no room.
    level = mf.fit_memory_view(snapshot, _window(0))
    assert mv.render_room(snapshot, level, floor_note=note(level)).endswith(note(level))
    roomless = syn.snapshot(role="consciousness", story=syn.pointers(20, 4), live=[syn.live_room(0)])
    text = mv.render_room(roomless, mv.FULL_VIEW, floor_note="### Physical floor\nx")
    assert text.index("## Live rooms") < text.index("### Physical floor") and text.endswith("x")


def test_the_nano_floor_names_the_path_to_memory_read_that_its_request_sends():
    """The path line is a fact of the request's schemas, never of a role: enable_tools when the
    request sends it; list_available_tools and the parent when it sends neither that nor
    memory_read; nothing when it sends memory_read or its schemas are not known."""
    level = mv.FloorLevel((("F7", ("x",)),))

    def note(names, mode="nano"):
        return mf.floor_note(level, window_tokens=128_000, mode=mode, tool_names=names)

    unknown = note(None)
    assert unknown.startswith("### Physical floor\n") and "enable_tools" not in unknown and "parent" not in unknown
    head = unknown[:unknown.index(" Sealing a closed part")]  # the same fact without the chronicle_write sentence
    assert note(("compact_context", "list_available_tools", "enable_tools")) == (
        unknown + "\n(memory_read is reachable through enable_tools)")
    listing = note(("list_available_tools",))
    path = listing[len(head) + 1:]
    assert listing.startswith(head + "\n") and "enable_tools" not in listing
    assert path == ("(memory_read is not among this request's tools; list_available_tools shows what this task "
                    "can call, and my parent task can read any address I name to it)")
    bare = note(())  # a request that sends neither names only the parent
    assert bare == (head + "\n(memory_read is not among this request's tools; my parent task can read any "
                    "address I name to it)")
    for carried in (("memory_read", "chronicle_write"), ("memory_read", "enable_tools", "list_available_tools")):
        assert note(carried) == unknown, carried  # the request sends memory_read itself: no path to name
    assert note(("memory_read",)) == head
    for mode in ("max", "low"):  # outside Nano the request sends its permitted schemas: no path line
        assert note(("list_available_tools", "chronicle_write"), mode=mode) == note(None, mode=mode), mode


def test_the_floor_offers_sealing_only_when_its_request_can_call_chronicle_write():
    """chronicle_write is named exactly when the request sends it, or enable_tools that loads it, in
    every mode; a request without both (a child the window lowered to Nano sends list_available_tools
    alone) still reads that nothing is lost, and is never offered a tool it cannot call."""
    level = mv.FloorLevel((("F5", ("p1",)),))
    for mode in mf.MODES:
        def note(names):
            return mf.floor_note(level, window_tokens=128_000, mode=mode, tool_names=names,
                                 lowered_from=None if mode == "max" else "max")

        for names in (None, ("chronicle_write",), ("compact_context", "list_available_tools", "enable_tools")):
            assert "as a page (chronicle_write kind=page) or folding old pages (kind=part)" in note(names), (mode, names)
        for names in (("list_available_tools",), (), ("memory_read", "knowledge_read")):
            text = note(names)
            assert "Nothing is lost: memory_read reads each." in text, (mode, names)
            assert "chronicle_write" not in text and "kind=part" not in text, (mode, names)


def test_the_shortest_view_counts_the_longest_path_line():
    snapshot = _rich()
    taken = {}
    for step, ident, _whole, _short in mf.floor_elements(snapshot):
        taken.setdefault(step, []).append(ident)
    level = mv.FloorLevel(tuple((step, tuple(ids)) for step, ids in taken.items()))

    def shortest(names):
        note = mf.floor_note(level, window_tokens=128_000, mode="nano", lowered_from="max", tool_names=names)
        return mf.view_tokens(mv.render_story(snapshot, level)) + mf.view_tokens(
            mv.render_room(snapshot, level, floor_note=note))

    longest = ("chronicle_write", "list_available_tools")  # the sealing sentence and the longest path line
    totals = {names: shortest(names) for names in (None, (), ("enable_tools",), ("list_available_tools",), longest)}
    assert totals[longest] > totals[("list_available_tools",)] > totals[()]
    assert totals[longest] > totals[("enable_tools",)] > totals[None]
    assert mf.minimal_view_tokens(snapshot, window_tokens=128_000) == max(totals.values()) == totals[longest]


def test_an_owner_target_takes_only_its_steps_and_names_the_mode_budget():
    snapshot = _rich()
    level = mf.fit_memory_view(snapshot, {"margin": 10**9, "physical": 10**9, "budget": 0})
    assert [name for name, _n in level.steps] == list(mf.MODE_TARGET_STEPS) == ["F1", "F1b", "F3", "F5", "F4"]
    assert level.by_budget == sum(count for _name, count in level.steps)
    text = mf.floor_note(level, window_tokens=1_050_000, mode="low", target_tokens=250_000)
    assert text.startswith("### Physical floor\nThe Low mode budget (250000 tokens) does not hold") and "window" not in \
        text.split("\n")[1].split(" does ")[0]
    # The working margins alone take what the budget takes, never my replies; the window minus the reserve
    # takes them too (and, last, people's words).
    windowed = mf.fit_memory_view(snapshot, {"margin": 0, "physical": 10**9, "budget": None})
    assert "F2" not in dict(windowed.steps) and windowed.steps and windowed.by_budget == 0
    assert "F2" in dict(mf.fit_memory_view(snapshot, {"margin": 0, "physical": 0, "budget": None}).steps)
    both = mv.FloorLevel(windowed.addressed, by_budget=1)
    assert "This window (1050000 tokens, Low) and the Low mode budget (250000 tokens) do not hold" in mf.floor_note(
        both, window_tokens=1_050_000, mode="low", target_tokens=250_000)


def test_a_line_whose_address_is_not_shorter_is_not_degradable():
    short, long = syn.spoken(0, "human", 12), syn.spoken(1, "human", 2_000)
    snapshot = syn.snapshot(room_facts=syn.room("1", lane1=[short, long]))
    assert [ident for _step, ident, *_rest in mf.degradable_elements(snapshot)] == [long["address"]]
    level = mf.fit_memory_view(snapshot, {"margin": 0, "physical": 0, "budget": 0})
    text = mv.render_room(snapshot, level)
    assert short["line"] in text and long["line"] not in text and f"{long['head']} (words, 2000 chars" in text


def test_the_floor_runs_on_a_captured_view(tmp_path):
    shared.world(tmp_path)
    task = {"id": "turn0001", "chat_id": 1}
    spec = dataclasses.replace(mv.view_spec_for_task(task, tmp_path), live_rooms="lines_with_words")
    snapshot = mv.capture_memory_view(tmp_path, task, spec)
    # The capture carries what the address lines need.
    assert all({"head", "address", "chars", "pos"} <= set(item) for item in snapshot.room["lane1"])
    assert all({"last", "last_ts", "last_pos"} <= set(item) for item in snapshot.room["lane2"])
    assert any(live["words"] and {"head", "address"} <= set(live["words"][0]) for live in snapshot.live_rooms)
    assert next(entry for entry in snapshot.story if entry["id"] == "legacy-b00-r1")["span"] == [
        "2026-09-01 00:00", "2026-09-01 00:05"]
    elements = mf.degradable_elements(snapshot)
    assert {step for step, *_rest in elements} >= {"F1", "F3"}
    level = mf.fit_memory_view(snapshot, {"margin": 0, "physical": 0, "budget": None})
    story, room = mv.render_story(snapshot, level), mv.render_room(snapshot, level)
    for step, ident, *_rest in elements:
        assert step == "F1" or ident in story + room, (step, ident)
    lane2 = snapshot.room["lane2"]
    assert f"from='{lane2[0]['first']}', to='{max(lane2, key=lambda item: item['last_pos'])['last']}')" in room
    assert "2 retold records in 25 chars: legacy-b00-r1, legacy-b01-r1; " in story  # Main's two, one line, their length
    assert "next please" in room and "beta again" in room  # short words cost less verbatim than by address
    assert mv.snapshot_from_json(mv.snapshot_json(snapshot)) == snapshot  # the new facts survive the core's JSON


def test_an_owner_target_takes_the_whole_first_block_into_its_rooms_line_and_the_window_alone_keeps_it(tmp_path):
    """The first block an integrator reads whole is retold memory like the rest: an owner's Low or Nano target takes
    it with its room's later records into the room's one line (F3), with no step or boundary of its own; without a
    shortage it stays whole. A child's F3 element is its pointers, as before."""
    rooms = shared.world(tmp_path)
    alpha = str(rooms["alpha"])
    task = {"id": "turn0001", "chat_id": 1}
    snapshot = mv.capture_memory_view(tmp_path, task, mv.view_spec_for_task(task, tmp_path))
    f3 = {ident: (whole, short) for step, ident, whole, short in mf.floor_elements(snapshot) if step == "F3"}
    assert "\n  Alpha began." in f3[alpha][0] and "Alpha began." not in f3[alpha][1]
    assert f"legacy-b00-r{alpha}, legacy-b01-r{alpha}; memory_read(node_id=<id>) reads each" in f3[alpha][1]
    assert mf.fit_memory_view(snapshot, NONE) == mv.FULL_VIEW and "  Alpha began." in mv.render_story(snapshot)
    level = mf.fit_memory_view(snapshot, {"margin": 10**9, "physical": 10**9, "budget": 0})
    taken = set(dict(level.addressed)["F3"])
    assert {"1", alpha} <= taken and set(dict(level.steps)) <= set(mf.MODE_TARGET_STEPS)
    assert level.by_budget == sum(count for _step, count in level.steps)
    story = mv.render_story(snapshot, level)
    assert "Alpha began." not in story and "Main talk." not in story and f3[alpha][1] in story.split("\n")
    # A record whose room line is not shorter than its words stays whole (the floor's one rule for every element).
    assert "777" not in taken and "777" not in f3 and "\n  A transport line." in story
    assert "(not lived; read by id)" in mv.render_story(snapshot, mv.FloorLevel((("F3", (*taken, "777")),)))
    kid = {"id": "kid1", "chat_id": 1, "delegation_role": "subagent"}
    child = mv.capture_memory_view(tmp_path, kid, mv.view_spec_for_task(kid, tmp_path))
    pointers = {ident: whole for step, ident, whole, _short in mf.floor_elements(child) if step == "F3"}
    assert "Alpha began." not in pointers[alpha] and f"memory_read(node_id='legacy-b00-r{alpha}')" in pointers[alpha]


def _numbers(name):
    tree = ast.parse((REPO / "ouroboros" / name).read_text(encoding="utf-8"))
    budgets = {cb.OWNER_LOW_TARGET_TOKENS, cb.OWNER_NANO_TARGET_TOKENS, cb.NANO_MIN_HEADROOM_TOKENS, 65_536,
               cb.RECLAIM_LOW_WATER_DIVISOR}
    literal = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and node.value in budgets
               and type(node.value) is int]
    margins = [kw for node in ast.walk(tree) if isinstance(node, ast.Call) for kw in node.keywords
               if kw.arg == "margin_count" and isinstance(kw.value, ast.Constant)]
    return literal, margins, {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}


@pytest.mark.parametrize("name", ["memory_view.py", "memory_floor.py"])
def test_no_margin_or_target_literal_lives_in_the_view(name):
    literal, margins, attributes = _numbers(name)
    assert literal == [] and margins == [], (literal, margins)
    if name == "memory_floor.py":
        assert {"MEMORY_VIEW_WORKING_MARGINS", "MODE_TARGET_WORKING_MARGINS", "request_context_budget"} <= attributes
    probe = ast.parse("request_context_budget(margin_count=2, target_tokens=250000)")
    assert [kw for node in ast.walk(probe) if isinstance(node, ast.Call) for kw in node.keywords
            if kw.arg == "margin_count" and isinstance(kw.value, ast.Constant)]  # the check sees a literal margin
    assert any(isinstance(node, ast.Constant) and node.value == cb.OWNER_LOW_TARGET_TOKENS for node in ast.walk(probe))
