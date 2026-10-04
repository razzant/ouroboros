"""The floor by mode and the physical starting mode (``memory_floor.render_view_for_mode``, ``physical_mode``).

On synthetic actors of the sizes measured on the owner's copy (Main, a Project room, consciousness, a child
whose parent is in a Project or in Main, a nanny) every cell of eight windows by three
preferred modes starts in a mode whose request plus Nano's headroom fits the window. The
starting mode drops only when the shortest view of my memory cannot fit, judged on the
route's calibrated estimate; an unknown window keeps the preferred mode and takes no
step. An owner-selected Low or Nano target is a second boundary for its own steps
only; task-local Low and a mode the window lowered have none (only the owner's own choice
carries a target). A built plan starts in that mode, says so in the room's ``### Physical floor``
(never in the runtime context) and in one owner-visible checkpoint, and a new route re-renders
the view for its own window from the captured snapshot, reading no memory again.
Each rule is pinned in both directions.
"""
from __future__ import annotations

import pytest

from ouroboros import chat_chain
from ouroboros import context_budget as cb
from ouroboros import memory_floor as mf
from tests import _memory_view_synthetic as syn

WINDOWS = (1_050_000, 872_000, 640_000, 500_000, 400_000, 272_000, 200_000, 128_000)
RESERVE = {"max": 65_536, "low": 65_536, "nano": cb.NANO_MIN_HEADROOM_TOKENS}


def _start(name, window, preferred, *, ratio=1.0, known=True, snapshot=None):
    """The planned start of one task: its mode, estimated input, story, room and floor fact."""
    snapshot = snapshot or syn.actor(name)
    fixed = syn.FIXED[name]
    minimal = mf.minimal_view_tokens(snapshot, window_tokens=window)
    mode = mf.physical_mode(preferred, fixed, minimal, window_tokens=window, known_window=known,
                            reserve_by_mode=RESERVE, calibration_ratio=ratio)
    story, room, facts = mf.render_view_for_mode(
        snapshot, mode=mode, owner_mode=preferred, window_tokens=window, known_window=known, output_reserve=65_536,
        ratio=ratio, non_memory_tokens=fixed[mode], lowered_from=preferred if mode != preferred else None)
    return mode, fixed[mode] + mf.view_tokens(story) + mf.view_tokens(room), story, room, facts


def test_every_actor_window_and_mode_starts_in_a_request_that_fits():
    misses, modes = [], {}
    for name in syn.FIXED:
        for window in WINDOWS:
            for preferred in mf.MODES:
                mode, total, _story, _room, facts = _start(name, window, preferred)
                modes[(name, window, preferred)] = mode
                if total + cb.NANO_MIN_HEADROOM_TOKENS > window:
                    misses.append((name, window, preferred, mode, total))
                assert facts["floor"]["mode"] == mode and facts["floor"]["window_tokens"] == window
    assert misses == []
    assert len(modes) == 6 * 8 * 3
    # The window lowers Max only where it physically must: Main keeps Max from 500k up, drops to Low at
    # 400k-200k, to Nano at 128k; a child keeps Max everywhere.
    assert [modes[("main", w, "max")] for w in WINDOWS] == ["max"] * 4 + ["low"] * 3 + ["nano"]
    assert {modes[("child_project", w, "max")] for w in WINDOWS} == {"max"}
    assert all(modes[(name, w, "nano")] == "nano" for name in syn.FIXED for w in WINDOWS)


@pytest.mark.parametrize("window", [640_000, 500_000])
def test_in_the_500k_to_640k_band_max_holds_and_the_conversation_stays_verbatim(window):
    snapshot = syn.actor("main")
    mode, total, _story, room, facts = _start("main", window, "max", snapshot=snapshot)
    assert mode == "max" and total + 65_536 <= window
    steps = facts["floor"]["steps"]
    assert "F2" not in steps and "F6" not in steps and "F7" not in steps  # the window alone takes them
    for item in snapshot.room["lane1"]:  # people's words and my replies
        assert item["line"] in room
    if window == 500_000:  # the working margins still take host facts, pointers and the retold page
        assert list(steps) == ["F1", "F1b", "F3", "F4"]
        assert "### Physical floor\nThis window (500000 tokens, Max) does not hold" in room


def test_where_even_f1_to_f5_leave_too_much_peoples_words_go_last_other_rooms_first():
    snapshot = syn.snapshot(room_facts=syn.room("1", legacy=23, legacy_chars=8_400, lane2=19,
                                                lane1=[syn.spoken(0, "human", 9_000), syn.spoken(1, "ouroboros", 5_600),
                                                       syn.spoken(2, "human", 8_000)]),
                            story=syn.pointers(), live=[syn.live_room(i, words=2, word_chars=2_000) for i in range(12)],
                            mark_count=12, live_rooms="lines_with_words")
    fixed = syn.FIXED["main"]["max"]

    def steps(window):
        return mf.render_view_for_mode(snapshot, mode="max", owner_mode="max", window_tokens=window, known_window=True,
                                       output_reserve=65_536, ratio=1.0, non_memory_tokens=fixed)[2]["floor"]["steps"]

    roomy = steps(fixed + 65_536 + 40_000)  # after F1-F5 the conversation fits the window minus the reserve
    assert "F2" not in roomy and "F6" not in roomy and "F7" not in roomy
    margins = mf.fit_memory_view(snapshot, {"margin": 0, "physical": 10**9, "budget": None})
    held = mf.view_tokens(mf.mv.render_story(snapshot, margins)) + mf.view_tokens(mf.mv.render_room(snapshot, margins))
    mine = steps(fixed + 65_536 + held - 1)  # one token short after F1-F5: my longest reply goes first, words stay
    assert "F2" in mine and "F6" not in mine and "F7" not in mine
    tight = steps(fixed + 65_536 + 9_000)  # after F1-F5 and F2 the words still do not fit
    assert list(tight) == ["F1", "F3", "F4", "F2", "F6", "F7"]  # every live room shows words: no F1b
    some = steps(fixed + 65_536 + 13_000)  # other rooms' words make room before this room's
    assert "F6" in some and "F7" not in some


def test_the_window_lowers_the_mode_on_the_calibrated_estimate_not_the_raw_one():
    snapshot = syn.actor("main")
    fixed = syn.FIXED["main"]
    minimal = mf.minimal_view_tokens(snapshot, window_tokens=400_000)
    border = fixed["max"] + minimal + 65_536 - 1_000  # raw: 1 000 tokens short of fitting Max
    pick = lambda ratio: mf.physical_mode("max", fixed, minimal, window_tokens=border, known_window=True,  # noqa: E731
                                          reserve_by_mode=RESERVE, calibration_ratio=ratio)
    assert pick(1.0) == "low"
    assert pick(0.98) == "max"  # the route counts 2 % fewer tokens than the host's estimate: Max fits
    assert pick(1.05) == "low"
    roomy = fixed["max"] + minimal + 65_536 + 1_000
    assert mf.physical_mode("max", fixed, minimal, window_tokens=roomy, known_window=True, reserve_by_mode=RESERVE,
                            calibration_ratio=1.0) == "max"
    assert mf.physical_mode("max", fixed, minimal, window_tokens=roomy, known_window=True, reserve_by_mode=RESERVE,
                            calibration_ratio=1.02) == "low"


def test_an_unknown_window_keeps_the_preferred_mode_and_takes_no_step():
    for preferred in ("max", "low"):
        mode, _total, _story, room, facts = _start("main", 128_000, preferred, known=False)
        assert mode == preferred
        assert facts["floor"]["window_tokens"] is None and facts["floor"]["allowance_tokens"] is None
        if preferred == "max":
            assert facts["floor"]["steps"] == {} and "### Physical floor" not in room
    known_mode, *_rest = _start("main", 128_000, "max")
    assert known_mode == "nano"  # the same window, known, does lower the mode


def test_a_lowered_mode_is_a_visible_fact_of_the_room_and_of_the_trace():
    mode, _total, _story, room, facts = _start("main", 400_000, "max")
    assert mode == "low" and facts["floor"]["mode_switch"] == {"from": "max", "to": "low"}
    assert ("This window (400000 tokens) cannot hold Max with even the shortest view of my memory; this task "
            "started in Low.") in room
    assert facts["floor"]["target_tokens"] is None  # the window chose Low; the owner did not, so no Low budget
    kept, _total, _story, room, facts = _start("main", 872_000, "max")
    assert kept == "max" and facts["floor"]["mode_switch"] is None and "cannot hold" not in room


def test_an_owner_low_target_takes_only_its_steps_and_task_local_low_has_none():
    snapshot = syn.actor("main")
    fixed = syn.FIXED["main"]

    def view(mode, owner_mode):
        return mf.render_view_for_mode(snapshot, mode=mode, owner_mode=owner_mode, window_tokens=1_050_000,
                                       known_window=True, output_reserve=65_536, ratio=1.0, non_memory_tokens=fixed[mode])

    budget = cb.OWNER_LOW_TARGET_TOKENS - 65_536 - fixed["low"]
    assert mf.view_tokens(mf.mv.render_story(snapshot)) + mf.view_tokens(mf.mv.render_room(snapshot)) > budget
    story, room, facts = view("low", "low")
    steps = facts["floor"]["steps"]
    assert steps and set(steps) <= set(mf.MODE_TARGET_STEPS) and facts["floor"]["by_budget"] == sum(steps.values())
    assert facts["floor"]["target_tokens"] == cb.OWNER_LOW_TARGET_TOKENS
    assert facts["floor"]["budget_allowance_tokens"] == budget - 31_250  # one landing level under the target
    for item in snapshot.room["lane1"]:  # people's words and my replies: the window alone may take them
        assert item["line"] in room
    assert "The Low mode budget (250000 tokens) does not hold all of my memory verbatim" in room
    assert mf.view_tokens(story) + mf.view_tokens(room) <= budget - 31_250 + 200  # the note is not budgeted
    # The same view in Max, and in task-local Low under the owner's Max: no target, no step.
    for mode, owner in (("max", "max"), ("low", "max")):
        _story, room, facts = view(mode, owner)
        assert facts["floor"]["steps"] == {} and facts["floor"]["target_tokens"] is None, (mode, owner)
        assert "### Physical floor" not in room


def test_an_owner_nano_target_bounds_with_nanos_own_reserve_and_names_enable_tools():
    snapshot = syn.actor("project")
    def owner_nano(names):
        return mf.render_view_for_mode(
            snapshot, mode="nano", owner_mode="nano", window_tokens=1_050_000, known_window=True, output_reserve=65_536,
            ratio=1.0, non_memory_tokens=syn.FIXED["project"]["nano"], tool_names=names)

    _story, room, facts = owner_nano(("compact_context", "list_available_tools", "enable_tools"))
    assert facts["floor"]["target_tokens"] == cb.OWNER_NANO_TARGET_TOKENS
    assert set(facts["floor"]["steps"]) <= set(mf.MODE_TARGET_STEPS)
    assert room.endswith("(memory_read is reachable through enable_tools)")
    # The line follows the schemas the request sends: none known, no claim; list_available_tools alone, no enable_tools.
    assert owner_nano(None)[1] == room[:room.rindex("\n")]
    assert "enable_tools" not in owner_nano(("list_available_tools",))[1]
    lowered = mf.render_view_for_mode(  # Nano the window chose for the owner's Max: Nano's reserve, no Nano budget
        snapshot, mode="nano", owner_mode="max", window_tokens=1_050_000, known_window=True, output_reserve=65_536,
        ratio=1.0, non_memory_tokens=syn.FIXED["project"]["nano"])[2]
    assert lowered["floor"]["target_tokens"] is None and lowered["floor"]["steps"] == {}
    assert lowered["floor"]["physical_allowance_tokens"] == 1_050_000 - cb.NANO_MIN_HEADROOM_TOKENS - syn.FIXED[
        "project"]["nano"]


def test_the_floor_fact_names_the_newest_row_and_the_records_shown_by_address():
    snapshot = syn.actor("main")
    _mode, _total, _story, _room, facts = _start("main", 200_000, "max", snapshot=snapshot)
    floor = facts["floor"]
    assert {"F1", "F3", "F4"} <= set(floor["steps"]) and "F2" not in floor["steps"] and "F7" not in floor["steps"]
    lane1, lane2 = snapshot.room["lane1"], snapshot.room["lane2"]
    newest = max((item["last_pos"], item["last"]) for item in lane2)
    assert floor["newest_addressed_row"] == chat_chain.parse_address(newest[1])
    # A reply the window took (F2) is a row shown only by address, as a task line (F1) is.
    reply = max((item for item in lane1 if item["kind"] == "ouroboros"), key=lambda item: item["pos"])
    by_reply = mf.view_facts(snapshot, mf.mv.FloorLevel((("F2", (reply["address"],)),)), window_tokens=200_000,
                             mode="max", allowances={})
    assert by_reply["floor"]["newest_addressed_row"] == chat_chain.parse_address(reply["address"])
    assert floor["pointer_records"] == [item["id"] for item in snapshot.room["legacy"]][:floor["steps"]["F4"]]
    assert facts["room_id"] == "1" and facts["role"] == "integrator"
    assert facts["story_status"] == {"folded": 0, "total": 23}
    whole = mf.render_view_for_mode(snapshot, mode="max", owner_mode="max", window_tokens=1_050_000, known_window=True,
                                    output_reserve=65_536, ratio=1.0, non_memory_tokens=syn.FIXED["main"]["max"])[2]
    assert whole["floor"]["newest_addressed_row"] is None and whole["floor"]["pointer_records"] == []


# --- A built plan (``context_fit``): the starting mode, the checkpoint and a new route ---

_TASK = {"type": "task", "text": "hi", "id": "tmain", "chat_id": 1}


def _built(tmp_path, monkeypatch, *, books=80_000, ratio=None):
    """``(core, plan(window, **kw))`` on the shared installation; Max's books grown by ``books`` characters
    so that Max's fixed part exceeds Low's (the test repository's own books are smaller than its nav map)."""
    import dataclasses
    from types import SimpleNamespace

    from ouroboros import context, context_fit
    from tests._memory_view_context import world

    env, memory, _rooms = world(tmp_path)
    core = context._capture_context_core(env, memory, dict(_TASK), None, None)
    core = dataclasses.replace(core, architecture_md=core.architecture_md + "\n\n" + "word " * (books // 5))
    if ratio is not None:
        monkeypatch.setattr(context_fit, "_route_calibration_ratio", lambda *_a: ratio)

    def plan(window, *, status="asserted", preferred="max", tool_schemas=None):
        evidence = SimpleNamespace(route_fp="r", status=status, stale=status != "asserted", window_tokens=window)
        return context._build_context_fit_plan(env, core, dict(_TASK), preferred_mode=preferred, tool_schemas=tool_schemas,
                                               route_resolver=lambda *_a, **_kw: ({"model": "m", "provider": "p"}, evidence))
    return core, plan


def _needs(core, plan):
    """Each mode's fixed part + the shortest view + its reserve, read off a roomy plan's floor facts (ratio 1)."""
    from ouroboros.memory_view import snapshot_from_json

    roomy = plan(10_000_000)
    minimal = mf.minimal_view_tokens(snapshot_from_json(core.memory_view_json), window_tokens=10_000_000)
    return {mode: 10_000_000 - roomy.projection(mode).memory_facts["floor"]["physical_allowance_tokens"] + minimal
            for mode in mf.MODES}


def test_a_built_task_starts_in_the_first_mode_its_known_window_can_hold(tmp_path, monkeypatch):
    core, plan = _built(tmp_path, monkeypatch)
    need = _needs(core, plan)
    assert need["max"] > need["low"] + 1_000 > need["nano"] + 1_000
    for window, expected in ((need["max"] + 50, "max"), (need["max"] - 50, "low"), (need["low"] - 50, "nano")):
        built = plan(window)
        assert built.initial_mode == expected, (window, expected)
        assert built.preferred_mode == "max"  # the owner's mode is not changed, only where this task starts
        switch = built.projection(expected).memory_facts["floor"]["mode_switch"]
        assert switch == (None if expected == "max" else {"from": "max", "to": expected})
    # Unknown or stale evidence never lowers the mode, however small the number it carries.
    assert plan(need["nano"] - 50, status="failed").initial_mode == "max"
    # An owner-selected smaller mode is never raised by a roomy window.
    assert plan(10_000_000, preferred="low").initial_mode == "low"


def test_the_starting_mode_follows_the_routes_calibrated_estimate(tmp_path, monkeypatch):
    core, plan = _built(tmp_path, monkeypatch)
    border = _needs(core, plan)["max"] - 50  # by the host's raw estimate Max misses by 50 tokens
    assert plan(border).initial_mode == "low"
    monkeypatch.setattr("ouroboros.context_fit._route_calibration_ratio", lambda *_a: 0.98)
    assert plan(border).initial_mode == "max"  # the route counts 2 % fewer tokens: Max fits, the books stay
    monkeypatch.setattr("ouroboros.context_fit._route_calibration_ratio", lambda *_a: 1.02)
    assert plan(border + 100).initial_mode == "low"  # and 2 % more: a raw fit is not a fit


def test_the_lowered_mode_is_a_line_of_the_rooms_physical_floor_never_of_the_runtime_context(tmp_path, monkeypatch):
    from tests._memory_view_context import section

    core, plan = _built(tmp_path, monkeypatch)
    need = _needs(core, plan)
    for window, mode in ((need["max"] - 50, "low"), (need["max"] + 50, "max")):
        built = plan(window)
        block_c = built.messages_for(built.initial_mode)[0]["content"][2]["text"]
        line = (f"This window ({window} tokens) cannot hold Max with even the shortest view of my memory; "
                f"this task started in {mode.capitalize()}.")
        assert (line in section(block_c, "## This room (Main)")) == (mode == "low"), mode
        assert "### Physical floor" in section(block_c, "## This room (Main)") or mode == "max"
        assert "cannot hold Max" not in section(block_c, "## Runtime context")
        # The other projections of the same plan carry no such line: only the one the task starts in.
        for other in set(mf.MODES) - {built.initial_mode}:
            assert "cannot hold Max" not in built.projection(other).system_content_json


def test_the_loop_reports_a_lowered_start_as_one_owner_visible_checkpoint(tmp_path, monkeypatch):
    import json
    import queue

    import ouroboros.loop as loop
    from ouroboros.tools.registry import ToolRegistry

    class FakeLLM:
        def default_model(self):
            return "test-model"

    monkeypatch.setattr(loop, "call_llm_with_retry", lambda *_a, **_kw: ({"role": "assistant", "content": "done"}, 0.0))
    monkeypatch.setattr(loop, "_run_task_acceptance_review_once", lambda **_kw: False)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    core, plan = _built(tmp_path / "w", monkeypatch)
    need = _needs(core, plan)
    for name, window in (("lowered", need["max"] - 50), ("kept", need["max"] + 50)):
        root = tmp_path / name
        root.mkdir()
        registry = ToolRegistry(repo_dir=root, drive_root=root)
        registry._ctx.context_fit_plan = plan(window)
        loop.run_llm_loop(messages=[{"role": "user", "content": "hi"}], tools=registry, llm=FakeLLM(), drive_logs=root,
                          emit_progress=lambda *_a, **_kw: None, incoming_messages=queue.Queue(), task_id="t1",
                          drive_root=root)
        events = [json.loads(line) for line in (root / "events.jsonl").read_text(encoding="utf-8").splitlines()
                  if '"context_fit_physical_mode"' in line] if (root / "events.jsonl").exists() else []
        if name == "lowered":
            assert len(events) == 1 and events[0]["type"] == "task_checkpoint"
            assert events[0]["preferred_mode"] == "max" and events[0]["effective_mode"] == "low"
            assert events[0]["window_tokens"] == window and events[0]["owner_visible"] is True
        else:
            assert events == []


def test_a_new_route_rerenders_the_view_for_its_window_without_reading_memory_again(tmp_path, monkeypatch):
    import ouroboros.chat_chain
    import ouroboros.chronicle_store
    import ouroboros.memory_view

    core, plan = _built(tmp_path, monkeypatch, books=0)
    roomy = plan(10_000_000)
    need = _needs(core, plan)

    def refuse(*_a, **_kw):
        raise AssertionError("a new route must not read the chronicle or the chat again")

    monkeypatch.setattr(ouroboros.memory_view, "capture_memory_view", refuse)
    monkeypatch.setattr(ouroboros.chat_chain, "iter_rows", refuse)
    monkeypatch.setattr(ouroboros.chronicle_store.ChronicleStore, "__init__", refuse)

    def on(plan_, window):
        return plan_.reproject_for_route(window_tokens=window, known_window=True, ratio=1.0, output_reserve=65_536,
                                         tool_schemas=None)

    smaller = on(roomy, need["max"] + 2_000)  # Max still fits, but not with two working margins: the floor steps in
    steps = smaller.projection("max").memory_facts["floor"]["steps"]
    assert smaller.initial_mode == "max" and steps and set(steps) <= {"F1", "F1b", "F3"}
    assert roomy.projection("max").memory_facts["floor"]["steps"] == {}
    assert smaller.projection("max").system_content_json != roomy.projection("max").system_content_json
    assert smaller.projection("max").estimated_tokens < roomy.projection("max").estimated_tokens
    assert smaller.core_sha256 == roomy.core_sha256 and smaller.window_tokens == need["max"] + 2_000
    tiny = on(roomy, need["nano"] + 50)
    assert tiny.initial_mode == "nano" and tiny.projection("nano").memory_facts["floor"]["mode_switch"] == {
        "from": "max", "to": "nano"}
    larger = on(smaller, 10_000_000)  # back on a roomy route: every step gone, the very texts of the first plan
    assert larger.initial_mode == "max" and larger.projection("max").memory_facts["floor"]["steps"] == {}
    for mode in mf.MODES:
        assert larger.projection(mode).system_content_json == roomy.projection(mode).system_content_json


def test_a_route_switch_lowers_the_mode_names_it_and_moves_the_view_fact_to_the_new_projection(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    import ouroboros.context
    from ouroboros.loop_model_call import _rebind_context_fit_plan
    from ouroboros.memory_inventory import VIEW_TRACE_KEY
    from ouroboros.tools.registry import ToolRegistry

    core, plan = _built(tmp_path / "w", monkeypatch)
    need = _needs(core, plan)
    roomy = plan(10_000_000)
    for name, window, mode in (("smaller", need["max"] - 50, "low"), ("same", 10_000_000, "max")):
        root = tmp_path / name
        root.mkdir()
        registry = ToolRegistry(repo_dir=root, drive_root=root)
        registry._ctx._execution_trace = {VIEW_TRACE_KEY: {"floor": {"mode": "max"}}}
        evidence = SimpleNamespace(route_fp="r2", status="asserted", stale=False, window_tokens=window)
        monkeypatch.setattr(ouroboros.context, "_context_fit_route",
                            lambda *_a, **_kw: ({"model": "m2", "provider": "p"}, evidence))
        messages = roomy.messages_for("max")
        rebound, active = _rebind_context_fit_plan(roomy, registry, messages, model="m2", use_local=False,
                                                   preferred_mode="max", tool_schemas=[])
        assert active == rebound.initial_mode == mode and rebound.model == "m2"
        assert messages[0] == rebound.projection(mode).system_message()
        facts = registry._ctx.memory_view_facts
        assert facts["floor"]["mode"] == mode and registry._ctx._execution_trace[VIEW_TRACE_KEY] == facts
        logs = root / "logs" / "events.jsonl"
        kinds = [json.loads(line).get("checkpoint_kind") for line in logs.read_text(encoding="utf-8").splitlines()]
        assert ("context_fit_physical_mode" in kinds) == (mode == "low")
        assert "context_fit_route_rebound" in kinds


def test_a_nano_the_window_chose_is_measured_against_the_window_alone(monkeypatch, tmp_path):
    """The task trace's measurement of a Nano the window chose has Nano's reserve and no owner target."""
    from types import SimpleNamespace

    from ouroboros import capability_evidence
    from ouroboros.context_fit import measure_main_fit
    from ouroboros.loop_model_call import _main_context_profile

    assert _main_context_profile(SimpleNamespace(preferred_mode="nano"), "nano") == "owner_nano"
    assert _main_context_profile(SimpleNamespace(preferred_mode="max"), "nano") == "task_local_nano"
    assert _main_context_profile(SimpleNamespace(preferred_mode="low"), "nano") == "task_local_nano"
    monkeypatch.setattr(capability_evidence, "resolve_main_token_density", lambda *_a, **_kw: (1.0, "cold_estimate"))
    core, plan = _built(tmp_path, monkeypatch)
    built = plan(200_000)
    messages = built.messages_for("nano") + [{"role": "user", "content": "x" * 400_000}]  # ≈100k: over 85k, under 200k

    def measure(profile):
        return measure_main_fit(built, messages, [], drive_root=tmp_path, profile=profile, rendered_mode="nano",
                                round_id="e:round:1")

    window, owner = measure("task_local_nano"), measure("owner_nano")
    assert window.measurement.response_reserve_tokens == owner.measurement.response_reserve_tokens == cb.NANO_MIN_HEADROOM_TOKENS
    assert window.measurement.target_total_tokens is None and window.action == "send"
    assert owner.measurement.target_total_tokens == cb.OWNER_NANO_TARGET_TOKENS and owner.action == "reclaim_once"


def test_a_route_switch_keeps_the_owners_mode_and_starts_from_the_tasks_own(tmp_path, monkeypatch):
    """A rebind on the task's current mode keeps the plan's owner mode: a Low the window or an overflow chose
    gets no Low budget (it is not the owner's choice), the same window renders the same bytes, and no second checkpoint is sent."""
    import json
    from types import SimpleNamespace

    import ouroboros.context
    from ouroboros.loop_model_call import _main_context_profile, _rebind_context_fit_plan
    from ouroboros.tools.registry import ToolRegistry

    core, plan = _built(tmp_path / "w", monkeypatch)
    need = _needs(core, plan)
    # Built with the schemas the rebind sends (none), so the same window can render the same bytes.
    lowered, roomy, owner_low = (plan(need["max"] - 50, tool_schemas=[]), plan(10_000_000, tool_schemas=[]),
                                 plan(10_000_000, preferred="low", tool_schemas=[]))
    assert lowered.initial_mode == "low" and roomy.initial_mode == "max" and owner_low.initial_mode == "low"
    for name, built, window in (("lowered", lowered, need["max"] - 50), ("task_local", roomy, 10_000_000),
                                ("owner_low", owner_low, 10_000_000)):
        root = tmp_path / name
        root.mkdir()
        registry = ToolRegistry(repo_dir=root, drive_root=root)
        evidence = SimpleNamespace(route_fp="r2", status="asserted", stale=False, window_tokens=window)
        monkeypatch.setattr(ouroboros.context, "_context_fit_route",
                            lambda *_a, _e=evidence, **_kw: ({"model": "m2", "provider": "p"}, _e))
        messages = built.messages_for("low")
        rebound, active = _rebind_context_fit_plan(built, registry, messages, model="m2", use_local=False,
                                                   start_mode="low", tool_schemas=[])
        floor = rebound.projection("low").memory_facts["floor"]
        assert active == rebound.initial_mode == "low" and rebound.preferred_mode == built.preferred_mode, name
        logs = root / "logs" / "events.jsonl"
        kinds = [json.loads(line).get("checkpoint_kind") for line in logs.read_text(encoding="utf-8").splitlines()]
        assert "context_fit_physical_mode" not in kinds and "context_fit_route_rebound" in kinds
        if name == "owner_low":  # the owner's Low keeps its budget across the switch
            assert floor["target_tokens"] == cb.OWNER_LOW_TARGET_TOKENS
            assert _main_context_profile(rebound, "low") == "owner_low"
            continue
        assert floor["target_tokens"] is None and floor["by_budget"] == 0, name
        assert "mode budget" not in rebound.projection("low").system_content_json
        assert _main_context_profile(rebound, "low") == "task_local_low"
        if name == "lowered":  # the same window: the same bytes, the mode line kept
            assert rebound.low_projection.system_content_json == built.low_projection.system_content_json
            assert floor["mode_switch"] == {"from": "max", "to": "low"}
        else:  # an overflow's Low on a window that holds Max: not this window's doing
            assert floor["mode_switch"] is None and floor["steps"] == {}
