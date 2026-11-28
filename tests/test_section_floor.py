"""Host section relocation at the ordinary context preparation and reader seams."""
from __future__ import annotations

import copy
import dataclasses
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import context, context_fit as fit, memory_floor, memory_view, section_floor
from ouroboros.context_budget import start_room_levels
from ouroboros.context_sections import runtime_host_section
from ouroboros.section_floor import HostSection, capture_sections
from tests import _memory_view_synthetic as syn
from tests._cases import case_folders
from tests._reference_book_fixtures import booked, map_saving

RESERVE = 8_192


def test_host_ladder_lists_live_producers_and_keeps_only_the_actual_protected_core():
    from ouroboros.context_layout import TIER0_ALWAYS_FULL

    assert memory_floor.HOST_STEPS == ("runtime", "review_pool", "review_continuity", "installed_skills",
        "drive_state", "world", "subagents", "body_map", "return_facts_words")
    assert memory_floor.OPTIONAL_STEPS == ("design_bible",)
    assert section_floor.PROTECTED == TIER0_ALWAYS_FULL
    # The overview joins the protected core by the owner's choice (20.10.2026): never an address.
    assert {"system", "bible", "identity", "scratchpad", "memory_marks", "shared_understanding"} <= section_floor.PROTECTED
    assert "shared_understanding" not in memory_floor.HOST_STEPS


@pytest.mark.parametrize("missing", ["window", "measure", "eligible"])
def test_host_steps_do_not_measure_when_no_section_can_step(missing):
    calls = []
    args = dict(window=50_000, root="test-root", task_id="test-task", reserve=RESERVE,
                ratio=1.0, saving=123, measure=lambda addressed: calls.append(dict(addressed)) or 1000)
    sections = [HostSection("runtime", "whole " * 1000, "operative")]
    if missing == "eligible":
        sections = [HostSection("runtime", "whole " * 1000)]
    else:
        args[missing] = None
    previous = [{"section_id": "registry", "reader": "test-reader"}]
    assert memory_floor._host_steps(sections, previous, **args) == (previous, 123)
    assert calls == []


@pytest.mark.parametrize("window,saving,steps", [(20_000, 0, 0), (18_000, 0, 1), (20_000, 3000, 2)])
def test_host_steps_measure_once_then_only_after_each_new_address(monkeypatch, window, saving, steps):
    calls = []
    sections = [HostSection(name, "whole " * 1000, "operative")  # walk order: the registry outline is last
                for name in ("runtime", "installed_skills", "registry")]
    monkeypatch.setattr(memory_floor, "retain_section",
                        lambda part, **_kw: {"section_id": part.section_id, "reader": "test-reader"})

    def measure(addressed):
        calls.append(tuple(addressed))
        return 10_000 - len(addressed) * 2000

    receipts, remaining = memory_floor._host_steps(sections, [], measure=measure, window=window,
        reserve=RESERVE, ratio=1.0, saving=saving, root="test-root", task_id="test-task")
    assert len(receipts) == steps and remaining == 0
    assert len(calls) == 1 + steps
    assert calls == [tuple(part.section_id for part in sections[:n]) for n in range(steps + 1)]


@pytest.mark.parametrize("least", [11_809, 11_808], ids=["least-request-over-the-window", "least-request-fits"])
def test_a_frame_whose_least_request_overflows_takes_every_step_for_the_window_unpriced(monkeypatch, least):
    """Every section at its shorter form and no memory at all (``bare``) already past the window (physical 11,808 here):
    no price can fit, so each section steps for the window and neither the held request nor memory's shortest view is
    measured. A least request the window holds is priced as before and, here, takes the same steps."""
    monkeypatch.setattr(memory_floor, "retain_section",
                        lambda part, **_kw: {"section_id": part.section_id, "reader": "test-reader"})
    sections = [HostSection(name, "whole " * 1000, "operative") for name in ("runtime", "installed_skills", "registry")]
    priced, bare = [], []
    receipts, _ = memory_floor._host_steps(sections, [], window=20_000, reserve=RESERVE, ratio=1.0, saving=0,
        root="test-root", task_id="test-task", allowance=10_000,
        bare=lambda forms: bare.append(dict(forms)) or least,
        measure=lambda addressed: priced.append("held") or 30_000 - 2_000 * len(addressed),
        shortest=lambda addressed, physical: priced.append("shortest") or 20_000)
    assert bare == [{"runtime": "operative", "installed_skills": "operative", "registry": "operative"}]
    assert [(row["section_id"], row.get("start_room", False)) for row in receipts] == [
        ("runtime", False), ("installed_skills", False), ("registry", False)]
    assert priced == ([] if least > 11_808 else ["held", "shortest"] * 3 + ["held"])


@pytest.fixture(autouse=True)
def capacity_facts(monkeypatch):
    monkeypatch.setattr(fit, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)
    monkeypatch.setattr(fit, "main_output_reserve_tokens", lambda **_kw: RESERVE)
    monkeypatch.setattr("ouroboros.capability_evidence.resolve_main_token_density",
                        lambda *_a, **_kw: (1.0, "cold_estimate"))


def _evidence(window):
    return SimpleNamespace(route_fp="synthetic-route", status="confirmed" if window else "unprobeable",
                           stale=False, window_tokens=window, source="synthetic")


def _core(root, dynamic=(), head=(), snapshot=None):
    return fit.ContextCore(base_prompt="SYSTEM WHOLE", bible_md="BIBLE WHOLE",
        body_map="# Architecture", development_md="# Development", semi_stable_text="IDENTITY WHOLE", dynamic_text="\n\n".join(dynamic),
        dynamic_head_text="\n\n".join(head), user_content_json=json.dumps("WORDS THAT STARTED THE WORK"),
        dynamic_sections=capture_sections(dynamic, prefix="dynamic"),
        dynamic_head_sections=capture_sections(head, prefix="head"),
        section_root=str(root), section_task_id="task-1",
        memory_view_json=memory_view.snapshot_json(snapshot) if snapshot else "")


def _plan(core, window, preferred="max"):
    return fit.build_context_fit_plan(SimpleNamespace(drive_root=Path(core.section_root)), core,
        {"id": "task-1"}, preferred_mode=preferred,
        route_resolver=lambda *_a, **_kw: ({"provider": "test", "model": "synthetic"}, _evidence(window)))


def _text(plan, mode=None):
    return "\n\n".join(block["text"] for block in plan.messages_for(mode or plan.initial_mode)[0]["content"])


def _host_facts(plan, mode=None):
    return plan.projection(mode or plan.initial_mode).memory_facts.get("host_sections_addressed", [])


def _service(name="runtime", size=100_000):
    return HostSection(name, f"## {name}\n\n" + "observed details " * size,
                       f"## {name}\n\nOperative state remains.")


@pytest.mark.parametrize(("window", "preferred"), [(0, "max"), (0, "nano"), (2_000_000, "low")])
def test_unselected_projections_are_byte_identical_and_write_nothing(tmp_path, preferred, window):
    core = _core(tmp_path, [_service(size=2_000), "protected dynamic text"], [_service("world", 1_000)],
                 syn.snapshot(story=syn.pages(2, 4_000)))
    legacy = dataclasses.replace(core, dynamic_sections=(), dynamic_head_sections=())
    selected, baseline = _plan(core, window, preferred), _plan(legacy, window, preferred)
    for mode in memory_floor.MODES:
        assert selected.projection(mode).system_content_json == baseline.projection(mode).system_content_json
        assert selected.messages_for(mode) == baseline.messages_for(mode)
        assert selected.projection(mode).memory_facts == baseline.projection(mode).memory_facts
    assert not (tmp_path / "task_results/artifacts").exists()


def test_host_sections_share_the_start_target_and_working_margin(tmp_path):
    core = _core(tmp_path, [_service(size=12_000)], snapshot=syn.snapshot(story=syn.pages(2, 5_000)))
    full = _plan(core, 0, "nano")
    tokens = full.nano_projection.estimated_tokens
    assert tokens + RESERVE > 40_000  # the owner Nano target is smaller than this request
    exact = _plan(core, tokens + RESERVE, "nano")
    assert _host_facts(exact, "nano")
    assert core.dynamic_text not in _text(exact, "nano")
    assert exact.nano_projection.memory_facts["floor"]["steps"] == {}
    pressured = _plan(core, tokens + RESERVE - 1, "nano")
    assert _host_facts(pressured, "nano")


def test_protected_sections_never_shrink_even_with_an_operative_view(subtests, tmp_path):
    case_paths = case_folders(tmp_path)
    for section_id in sorted(section_floor.PROTECTED):
        tmp_path = next(case_paths)
        with subtests.test(section_id=section_id):
            protected = HostSection(section_id, f"## {section_id}\n" + "whole core " * 2_000, "forbidden replacement")
            core = _core(tmp_path, [protected, _service(size=3_000)])
            with pytest.raises(fit.MinimumViewUnfit):
                _plan(core, RESERVE + 1)
            plan = _plan(core, 2_000_000).reproject_for_route(window_tokens=RESERVE + 1, known_window=True,
                ratio=1, output_reserve=RESERVE, tool_schemas=[])
            assert protected.full_text in _text(plan)
            assert "forbidden replacement" not in _text(plan)
            assert plan.messages_for("max")[1]["content"] == "WORDS THAT STARTED THE WORK"
            assert section_id not in {row["section_id"] for row in _host_facts(plan)}
            disposition = fit.measure_main_fit(plan, plan.messages_for("max"), [],
                profile="owner_max", rendered_mode="max", round_id="r1", automatic_pass_used=True)
            assert disposition.predicted_capacity_miss
            assert disposition.action == "send"  # an estimator miss never hard-refuses


@pytest.mark.parametrize("short", ["the same tiny section", "longer than the tiny section " * 50])
def test_an_address_must_be_shorter_than_the_section(tmp_path, short):
    section = HostSection("runtime", "the same tiny section", short)
    plan = _plan(_core(tmp_path, [section]), 2_000_000).reproject_for_route(
        window_tokens=RESERVE + 1, known_window=True, ratio=1, output_reserve=RESERVE, tool_schemas=[])
    assert section.full_text in _text(plan)
    assert not _host_facts(plan)
    assert not (tmp_path / "task_results/artifacts").exists()


def test_runtime_keeps_scalars_authority_and_decisions_while_counting_observational_lists():
    contract = {"grants": ["read every source"], "obligations": ["keep the ABI"], "active_decisions": ["keep compatibility"]}
    data = {"runtime_mode": "advanced", "budget": {"remaining_usd": 10}, "git_head": "abc",
            "task": {"id": "t", "deadline_at": "tomorrow", "allowed_resources": ["repo", "data"]},
            "project_room": {"project_id": "p", "working_dir": "/work"},
            "task_contract": contract, "task_authority_origin": {"text": "Exact owner words"},
            "plan_review_authority": {"obligations": ["open finding"], "active_decisions": ["unchanged"]},
            "capabilities": {"mutative_subagent_surfaces": ["external_workspace"],
                             "delegation": {"reviewer_slots_last": [{"slot": "s", "details": "long history"}] * 30}},
            "current_chat": {"chat_id": 1, "running_tasks": [{"task_id": "other", "text": "long description"}] * 30}}
    before = copy.deepcopy(data)
    ledger = "\n\n## Task-tree coordination ledger\nActive decision: keep all grants."
    section = runtime_host_section(data, ledger)
    assert section.full_text == "## Runtime context\n\n" + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ledger
    operative = json.JSONDecoder().raw_decode(section.operative_view.split("\n\n", 1)[1])[0]
    for key in ("runtime_mode", "budget", "git_head", "task", "project_room", "task_contract",
                "task_authority_origin", "plan_review_authority"):
        assert operative[key] == data[key]
    assert operative["capabilities"]["mutative_subagent_surfaces"] == ["external_workspace"]
    assert operative["current_chat"]["running_tasks"]["count"] == 30
    assert operative["capabilities"]["delegation"]["reviewer_slots_last"]["count"] == 30
    assert section.operative_view.endswith(ledger) and data == before


def test_route_reprojection_only_narrows_and_recomputes_from_the_original_capture(tmp_path):
    core = _core(tmp_path, [_service(size=15_000)], [_service("world", 15_000)], syn.snapshot())
    initial = _plan(core, 2_000_000)
    narrowed = initial.reproject_for_route(window_tokens=50_000, known_window=True, ratio=1.0,
        output_reserve=RESERVE, tool_schemas=[], current_messages=initial.messages_for("max"))
    assert _host_facts(narrowed) and narrowed.max_projection.system_content_json != initial.max_projection.system_content_json
    assert narrowed.core is initial.core
    same = narrowed.reproject_for_route(window_tokens=2_000_000, known_window=True, ratio=1.0,
        output_reserve=RESERVE, tool_schemas=[], current_messages=narrowed.messages_for("max"))
    assert same.max_projection.system_content_json == narrowed.max_projection.system_content_json
    assert _host_facts(same) == _host_facts(narrowed)
    # Actual refusal on an unknown window may narrow memory, never newly address host sections.
    unknown = initial.reproject_for_route(window_tokens=0, known_window=False, ratio=1.0,
        output_reserve=RESERVE, tool_schemas=[], refused_tokens=10_000)
    assert not _host_facts(unknown)


def test_physical_mode_lowering_carries_host_addresses_from_the_shown_view(tmp_path, monkeypatch):
    monkeypatch.setattr(fit, "main_output_reserve_tokens", lambda **_kw: 65_536)
    core = _core(tmp_path, [_service("runtime", 12_000)], snapshot=syn.snapshot())
    initial = _plan(core, 100_000, "low")
    assert _host_facts(initial, "low") == _host_facts(initial, "nano")  # one selected view across book forms
    lowered = initial.reproject_for_route(window_tokens=60_000, known_window=True, ratio=1.0,
        output_reserve=65_536, tool_schemas=[], start_mode="low", current_messages=initial.messages_for("low"))
    assert lowered.initial_mode == "nano"
    assert _host_facts(lowered, "nano") == _host_facts(initial, "low")
    assert core.dynamic_text not in _text(lowered, "nano")


@pytest.mark.parametrize("map_parts", [False, True], ids=["map-without-parts", "map-with-parts"])
def test_a_shown_view_keeps_its_frame_while_its_own_completed_view_fits(tmp_path, map_parts):
    """A Max view already shown, on a smaller known window (no tools): its own folded view, receipt included, fits
    and stays Max. Before, a generic Nano-note price of its shortest view (465 vs its real 360 tokens) lowered such a
    view to a Nano view that itself needed more. Below its own completed view, the frame steps down."""
    snapshot = syn.snapshot(story=syn.pages(4, 8_000), room_facts=syn.room("1", lane1=[
        syn.spoken(1, "human", 2_000), syn.spoken(2, "ouroboros", 2_000)]))
    core = _core(tmp_path, snapshot=snapshot)
    if map_parts:
        core = dataclasses.replace(core, body_map="## Body map\n\n" + "### Part\nSource: `docs/a.md`\nIts opening.\n" * 40,
                                   body_map_parts="## Body map (parts and addresses)\n\n- Part: `docs/a.md`")
    roomy = _plan(core, 2_000_000)
    plain = _plan(_core(tmp_path, snapshot=snapshot), 2_000_000)
    shift = roomy.max_projection.estimated_tokens - plain.max_projection.estimated_tokens
    windows = (8_684 + shift, 8_620 + shift)
    held = {window: roomy.reproject_for_route(window_tokens=window, known_window=True, ratio=1.0,
                                              output_reserve=RESERVE, tool_schemas=[], start_mode="max")
            for window in windows}
    kept = held[windows[0]].projection(held[windows[0]].initial_mode)
    low = held[windows[1]].projection(held[windows[1]].initial_mode)
    assert held[windows[0]].initial_mode == "max" and kept.estimated_tokens + RESERVE <= windows[0]
    assert "F5" in kept.memory_facts["floor"]["steps"]  # memory folded; the frame stayed
    # Below its own completed view the frame steps down: to Low with the Map at its parts where that saves, to Nano
    # (the last physical rescue) where a Map without a shorter form leaves Low nothing to give.
    assert held[windows[1]].initial_mode == ("low" if map_parts else "nano")
    addressed = {row["section_id"] for row in low.memory_facts.get("host_sections_addressed", ())}
    assert ("body_map" in addressed) is map_parts


def test_tool_schema_cost_and_route_density_both_count_toward_physical_pressure(tmp_path, monkeypatch):
    core = _core(tmp_path, [_service(size=3_000)])
    tokens = _plan(core, 0).max_projection.estimated_tokens
    schemas = [{"type": "function", "function": {"name": "read_file", "description": "schema detail " * 3_000,
                                                "parameters": {"type": "object", "properties": {}}}}]
    levels = start_room_levels("integrator")  # this root's start leaves its role's eighths for work
    window = (tokens + RESERVE) * 8 // (8 - levels) + 10
    assert not _host_facts(_plan(core, window))
    with_tools = fit.build_context_fit_plan(SimpleNamespace(drive_root=tmp_path), core, {"id": "task-1"},
        preferred_mode="max", tool_schemas=schemas,
        route_resolver=lambda *_a, **_kw: ({"provider": "test", "model": "synthetic"}, _evidence(window)))
    assert _host_facts(with_tools)
    monkeypatch.setattr(fit, "_route_calibration_ratio", lambda *_a, **_kw: 2.0)
    assert _host_facts(_plan(core, window))


def test_new_host_savings_pay_the_refused_deficit_before_memory_steps(tmp_path):
    core = _core(tmp_path, [_service(size=8_000)], snapshot=syn.snapshot(story=syn.pages(2, 8_000)))
    initial = _plan(core, 2_000_000)
    narrowed = initial.reproject_for_route(window_tokens=30_000, known_window=True, ratio=1.0,
        output_reserve=RESERVE, tool_schemas=[], refused_tokens=1_000)
    assert _host_facts(narrowed)
    assert narrowed.max_projection.memory_facts["floor"]["steps"] == {}


def _presence_ceiling(*tool_names):
    # A ceiling already admitted work still carries (places mint none for new work).
    from ouroboros.presence_authority import PresenceCapabilityCeiling, PresenceToolGrant, presence_ceiling_payload

    ceiling = PresenceCapabilityCeiling(
        skill_name="moderator", skill_content_hash="c" * 64, profile_fingerprint="a" * 64,
        state_fingerprint="d" * 64, selection_fingerprint="b" * 64, model_slot="main", inline_max_rounds=10,
        tool_grants=tuple(PresenceToolGrant(name) for name in (tool_names or ("chat_history",))),
        resource_grants=(), digest="e" * 64)
    return {"capability_ceiling": presence_ceiling_payload(ceiling)}


def test_host_sections_stay_whole_for_an_actor_that_could_not_read_a_pointer_back(tmp_path):
    env = SimpleNamespace(drive_root=tmp_path)
    # An owner turn (no ceiling): pointers resolve. (A Presence ceiling that admits the read of a retained capture
    # through the real registry: tests/test_book_context_capture.py, "retained-reader".)
    assert context._section_root_for(env, SimpleNamespace()) == str(tmp_path)
    # A Presence ceiling without read_file, or with it and no grant that admits the artifact store's retained
    # captures, or a task whose contract disables the tool: a pointer would name a source this actor cannot
    # open, so the preparation leaves its host sections whole (an empty section root).
    assert context._section_root_for(env, SimpleNamespace(task_contract=_presence_ceiling("read_file"))) == ""
    assert context._section_root_for(env, SimpleNamespace(task_contract={"disabled_tools": ["read_file"]})) == ""
    assert context._section_root_for(env, None, {"task_contract": {"disabled_tools": ["read_file"]}}) == ""
    assert context._section_root_for(env, None, {"task_contract": {"disabled_tools": ["web_search"]}}) == str(tmp_path)
    assert context._section_root_for(env, SimpleNamespace(task_contract=_presence_ceiling())) == ""
    assert context._section_root_for(env, SimpleNamespace(task_metadata={"task_contract": _presence_ceiling()})) == ""
    # An unreadable carrier also keeps the sections whole rather than publishing a dead pointer.
    assert context._section_root_for(env, SimpleNamespace(task_contract={"capability_ceiling": {"bad": 1}})) == ""
    core = _core(tmp_path, [_service(size=12_000)], snapshot=syn.snapshot(story=syn.pages(2, 5_000)))
    whole = dataclasses.replace(core, section_root="")
    plan = _plan(whole, 128_000, "nano")
    assert _host_facts(plan, "nano") == [] and "observed details" in _text(plan, "nano")
    assert not (tmp_path / "task_results/artifacts").exists()



def test_host_sections_then_memory_forms_then_a_physical_index_outline_and_people_last(tmp_path):
    snapshot = syn.snapshot(story=syn.pages(3, 20_000), room_facts=syn.room("1", lane1=[
        syn.spoken(1, "ouroboros", 15_000), syn.spoken(2, "human", 15_000)]))
    host = _service(size=35_000)
    index = _service("knowledge_index", 5_000)
    core = _core(tmp_path, [host], [index], snapshot)
    # Host alone buys enough room for both the index and all memory.
    plan = _plan(core, 128_000)
    assert [r["section_id"] for r in _host_facts(plan)] == ["runtime"]
    assert plan.max_projection.memory_facts["floor"]["steps"] == {}
    assert memory_view.render_story(snapshot) in _text(plan)
    # Memory's own shorter forms make the room first: the window holds the whole index (P1, owner 2.11.2026).
    smaller = _plan(core, 50_000)
    assert [r["section_id"] for r in _host_facts(smaller)] == ["runtime"] and index in _text(smaller)
    steps = smaller.max_projection.memory_facts["floor"]["steps"]
    assert steps and "F7" not in steps
    # Only where those forms cannot make room does the window outline the index, still before human words.
    middle = _plan(core, 20_000)
    assert [(r["section_id"], "start_room" in r) for r in _host_facts(middle)] == [
        ("runtime", False), ("knowledge_index", False)]
    assert "F5l" in middle.max_projection.memory_facts["floor"]["steps"]
    assert "F7" not in middle.max_projection.memory_facts["floor"]["steps"]
    smallest = middle.reproject_for_route(window_tokens=RESERVE + 1, known_window=True, ratio=1,
        output_reserve=RESERVE, tool_schemas=[])
    steps = smallest.max_projection.memory_facts["floor"]["steps"]
    assert steps["F5"] == 3 and steps["F2"] == 1 and steps["F7"] == 1


def test_an_owner_target_never_outlines_a_world_map_the_window_holds(tmp_path):
    """WORLD.md shortens only for the window (memory audit 15.11, F7): while the window's own start bound holds the
    request with memory at its shortest forms, neither an owner's Low target nor a Max start's own room outlines the
    map (other host forms still pay); a window that cannot hold it outlines it in any mode, its exact source behind a
    reader, still before memory."""
    world = _service("world", 60_000)  # about 255,000 tokens: above Low's target even after the runtime steps
    core = _core(tmp_path, [_service(size=60_000)], [world], syn.snapshot(story=syn.pages(3, 20_000)))
    roomy = _plan(core, 1_000_000, "low")
    assert [r["section_id"] for r in _host_facts(roomy, "low")] == ["runtime"]  # the target's own host step
    assert world in _text(roomy, "low")
    tight = _plan(core, 200_000, "low")  # its start bound cannot hold the map beside the rest
    assert "world" in [r["section_id"] for r in _host_facts(tight, "low")] and world.operative_view in _text(tight, "low")
    plain = _plan(core, 200_000, "max")  # no target, a window that cannot hold the map: outlined before memory
    assert "world" in [r["section_id"] for r in _host_facts(plain, plain.initial_mode)]
    mature = _core(tmp_path, [_service(size=60_000)], [world], syn.snapshot(story=syn.pages(60, 60_000)))
    held = _plan(mature, 1_000_000, "low")  # whole memory far above the window's bound, its shortest forms below it
    assert "world" not in [r["section_id"] for r in _host_facts(held, "low")] and world in _text(held, "low")
    started = _plan(mature, 1_000_000, "max")  # the same memory in Max: the start's own room never outlines the map
    shown = started.initial_mode
    assert "world" not in [r["section_id"] for r in _host_facts(started, shown)] and world in _text(started, shown)


def test_the_facts_and_the_peoples_words_stand_until_only_the_window_could_shorten_them(tmp_path):
    """The facts beside my human's word and the people's newest words beside them (M5; owner 15.11 4A, 18.11 1A): both
    stand whatever a selected Low asks, while the window's start bound holds them beside memory's shortest telling and
    the books; past it the words step first to their readers and then the facts to their totals, memory folding first."""
    facts = HostSection("return_facts", "## Since my human's previous word\n" + "a fact line " * 30_000,
                        "## Since my human's previous word\nSince then, counted: 3 places.")
    words = HostSection("return_facts_words", "## Their newest words since my human's previous word\n" + "«words» " * 200_000,
                        "## Their newest words since my human's previous word\nThe newest words of 3 people stand at "
                        "the readers on their place lines above.")
    roomy = _plan(_core(tmp_path, [_service(size=60_000), facts, words], [], syn.snapshot(story=syn.pages(3, 20_000))),
                  1_000_000, "low")
    stepped = [r["section_id"] for r in _host_facts(roomy, "low")]
    assert "runtime" in stepped, "the Low frame steps what it may"
    assert "return_facts_words" not in stepped and "return_facts" not in stepped, "a selected Low shortens neither"
    assert words.full_text in _text(roomy, "low") and facts.full_text in _text(roomy, "low")
    mature = _core(tmp_path, [facts, words], [], syn.snapshot(conversations=[syn.conversation(i, words=20, word_chars=1_000)
                                                                                for i in range(150)]))
    held = _plan(mature, 1_000_000, "max")  # other conversations far above the window whole, at my telling below it
    shown = held.initial_mode
    assert not [r for r in _host_facts(held, shown) if r["section_id"].startswith("return_facts")]
    assert words.full_text in _text(held, shown) and facts.full_text in _text(held, shown)
    worded = _plan(mature, 400_000, "max")  # the words would cost my telling: they step, the facts keep their detail
    shown = worded.initial_mode
    stepped = [r["section_id"] for r in _host_facts(worded, shown)]
    assert "return_facts_words" in stepped and "return_facts" not in stepped and facts.full_text in _text(worded, shown)
    tight = _plan(mature, 100_000, "max")  # the facts would cost the books: their totals, and Max keeps whole books
    shown = tight.initial_mode
    assert shown == "max" and "return_facts" in [r["section_id"] for r in _host_facts(tight, shown)]
    assert facts.operative_view in _text(tight, shown) and facts.full_text not in _text(tight, shown)


def test_a_refit_of_held_work_keeps_the_facts_while_memory_can_still_fold_to_its_telling(tmp_path):
    """A held view re-fitted for a smaller window or a refusal (M5 council, Opus R5) prices the facts and the people's
    words by the same test as a start, its running tail at memory's telling and no new work room: memory folds first,
    then the words step, and the facts keep their detail until even that request exceeds the window."""
    facts = HostSection("return_facts", "## Since my human's previous word\n" + "a fact line " * 3_000,
                        "## Since my human's previous word\nSince then, counted: 3 places.")
    words = HostSection("return_facts_words", "## Their newest words since my human's previous word\n" + "«words» " * 20_000,
                        "## Their newest words since my human's previous word\nThe newest words of 3 people stand at "
                        "the readers on their place lines above.")
    core = _core(tmp_path, [facts, words], [], syn.snapshot(conversations=[syn.conversation(i, words=20, word_chars=1_000)
                                                                          for i in range(150)]))
    initial = _plan(core, 2_000_000)

    def refit(window):
        plan = initial.reproject_for_route(window_tokens=window, known_window=True, ratio=1.0, output_reserve=RESERVE,
                                           tool_schemas=[], refused_tokens=1_000)
        return plan, [row["section_id"] for row in _host_facts(plan)]

    folded, stepped = refit(300_000)  # memory pays the refusal: both stand
    assert "return_facts_words" not in stepped and "return_facts" not in stepped and words.full_text in _text(folded)
    worded, stepped = refit(60_000)
    assert "return_facts_words" in stepped and "return_facts" not in stepped and facts.full_text in _text(worded)
    assert "return_facts" in refit(16_000)[1]


def test_known_window_renders_every_active_note_in_outline_unknown_steps_nothing(tmp_path):
    from ouroboros.knowledge import render_knowledge_index, render_knowledge_outline
    rows = tuple({"topic": f"notes/n{i}", "title": f"Note {i}", "summary": "Authored facts. " * 100,
                  "type": "person" if i == 0 else "note"} for i in range(60))
    whole, outline = render_knowledge_index(rows), render_knowledge_outline(rows)
    index = HostSection("knowledge_index", whole, outline)
    core = _core(tmp_path, [_service(size=500)], [index], syn.snapshot())
    unknown = _plan(core, 0, "nano")
    assert whole in _text(unknown) and not _host_facts(unknown)
    assert unknown.nano_projection.memory_facts["floor"]["steps"] == {}
    fitted = _plan(core, 18_000, "nano")
    assert outline in _text(fitted) and whole not in _text(fitted)
    assert [row["section_id"] for row in _host_facts(fitted)] == ["runtime", "knowledge_index"]
    for i in range(60):
        assert f"- notes/n{i} — Note {i}\n" in _text(fitted)
    assert rows[0]["summary"] in _text(fitted)
    assert fitted.nano_projection.estimated_tokens + RESERVE <= fitted.window_tokens


def test_same_route_window_and_core_render_identical_blocks_for_every_caller(tmp_path):
    core = _core(tmp_path, [_service(size=20_000)], [_service("knowledge_index", 15_000)], syn.snapshot())
    env = SimpleNamespace(drive_root=tmp_path)
    resolver = lambda *_a, **_kw: ({"provider": "test", "model": "ordinary"}, _evidence(30_000))
    # Caller metadata never enters the ladder. Both ordinary entrypoints use this captured core.
    plans = [fit.build_context_fit_plan(env, core, task, preferred_mode="nano", route_resolver=resolver)
             for task in ({"id": "task-1", "type": "chat"},
                          {"id": "task-1", "type": "live_conversation"})]
    assert _host_facts(plans[0])
    assert plans[0].nano_projection.system_content_json == plans[1].nano_projection.system_content_json


def test_retained_host_source_is_exact_immutable_and_readable(tmp_path, monkeypatch):
    from ouroboros.artifacts import read_actor_source_bytes, task_artifact_dir_path
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    repo, root = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    section = _service(size=6_000)  # still forces the host form with the shared one-eighth start room
    plan = _plan(_core(root, [section]), 12_000)
    row = _host_facts(plan)[0]
    ref = row["source_ref"]
    assert read_actor_source_bytes(root, "task-1", ref).decode() == section.full_text
    path = task_artifact_dir_path(root, "task-1") / ref["path"]
    stamp = path.stat().st_mtime_ns
    ctx = ToolContext(repo_dir=repo, drive_root=root, task_id="task-1")
    registry = ToolRegistry(repo_dir=repo, drive_root=root)
    registry.set_context(ctx)
    result = registry.execute_result("read_file", {"root": "artifact_store", "path": ref["path"], "max_lines": 1000})
    assert result.status == "ok", result.text
    assert section.full_text.splitlines()[-1] in result.text
    again = _plan(plan.core, 12_000)
    assert _host_facts(again)[0]["source_ref"] == ref and path.stat().st_mtime_ns == stamp
    changed = dataclasses.replace(section, full_text=section.full_text + "\nA new capture.")
    later = _host_facts(_plan(_core(root, [changed]), 12_000))[0]
    assert later["source_ref"]["path"] != ref["path"]
    assert path.read_text(encoding="utf-8") == section.full_text


def test_failed_retention_keeps_the_whole_section(tmp_path, monkeypatch):
    def unavailable(*_a, **_kw):
        raise OSError("synthetic unavailable store")
    monkeypatch.setattr(memory_floor, "retain_section", unavailable)
    section = _service(size=400)
    plan = _plan(_core(tmp_path, [section]), 2_000_000).reproject_for_route(
        window_tokens=RESERVE + 1, known_window=True, ratio=1, output_reserve=RESERVE, tool_schemas=[])
    assert section.full_text in _text(plan) and not _host_facts(plan)


def test_declared_inputs_use_the_same_body_map_floor_without_loading_shared_memory(tmp_path, monkeypatch):
    from tests.test_doc_context import _make_env_and_memory

    env, memory = _make_env_and_memory(tmp_path)
    env.repo_path("docs/ARCHITECTURE.md").write_text(
        "# Architecture\n\nBody map.\n\n## Chapters\n\n- [Core](architecture/core.md)\n")
    chapter = env.repo_path("docs/architecture/core.md")
    chapter.parent.mkdir(exist_ok=True)
    chapter.write_text("# Core\n\nIts purpose.\n\n" + "\n\n".join(f"## Detail {i}\n\nFacts." for i in range(100)))
    task = {"id": "declared-child", "type": "task", "delegation_role": "subagent", "text": "Exact assignment",
            "configured_subagent": {"route": {"kind": "api_model"}},
            "task_contract": {"input_sources": "declared"}}
    monkeypatch.setattr(context, "build_memory_sections", lambda *_a, **_kw: pytest.fail("shared memory opened"))
    monkeypatch.setattr(context, "build_knowledge_sections", lambda *_a, **_kw: pytest.fail("knowledge opened"))
    core = context._capture_context_core(env, memory, task, None, None)
    assert '"story":false' in core.memory_view_json and '"room_page":false' in core.memory_view_json  # the one view
    assert core.section_root == str(env.drive_root) and core.section_task_id == task["id"]
    unknown = _plan(core, 0, "nano")
    assert not _host_facts(unknown)
    small = unknown.reproject_for_route(window_tokens=RESERVE + 1, known_window=True, ratio=1,
        output_reserve=RESERVE, tool_schemas=[], start_mode="nano")
    assert _host_facts(small)
    addressed = {row["section_id"] for row in _host_facts(small)}
    assert addressed <= {"body_map", "runtime", "review_pool", "subagents"}
    assert {'runtime', 'review_pool'} <= {p.section_id for p in (*core.dynamic_sections, *core.semi_stable_sections)}
    assert all(p.full_text in _text(small) for p in core.dynamic_sections if p.section_id not in addressed)
    assert "Exact assignment" in core.user_content_json


def test_the_shared_understanding_overview_is_never_addressed_while_steppable_sections_are(tmp_path):
    overview = HostSection("shared_understanding", "## Shared understanding\n\n" + "orientation " * 15_000,
                           "## Shared understanding\n\nIts own headings, verbatim:\n# One")
    core = _core(tmp_path, [_service("world", 15_000)], [overview], syn.snapshot())
    narrowed = _plan(core, 2_000_000).reproject_for_route(window_tokens=30_000, known_window=True, ratio=1.0,
        output_reserve=RESERVE, tool_schemas=[], current_messages=_plan(core, 2_000_000).messages_for("max"))
    addressed = {row["section_id"] for row in _host_facts(narrowed)}
    # The window pressure relocates the steppable producer, never the owner's whole orientation.
    assert "world" in addressed and "shared_understanding" not in addressed
    assert "orientation orientation" in _text(narrowed, "max")


def test_authored_memory_selection_keeps_host_addresses(tmp_path):
    snapshot = syn.snapshot(story=syn.pages(3, 8_000))
    core = _core(tmp_path, [_service(size=15_000)], snapshot=snapshot)
    plan = _plan(core, 30_000, "nano")
    receipts = _host_facts(plan)
    assert receipts
    address = next(f"{step}:{ident}" for step, ident, *_ in memory_floor.floor_elements(snapshot) if step == "F5")
    selected = memory_floor.select_view(plan, "nano", [address])
    assert _host_facts(selected) == receipts
    assert core.dynamic_text not in _text(selected)


WHOLE_MAP, MAP_PARTS = "Mechanism 149: its facts", "## Body map (parts and addresses)"


def test_an_integrating_low_start_keeps_the_body_map_while_the_window_holds_it_beside_my_telling(tmp_path):
    """#3649 (owner 17.11.2026: keep both maps whole; one Body Map since 25.11.2026). A new start of the owner's Low
    keeps the whole Map while the window's start bound holds the request with it beside my shortest telling:
    neither the owner's Low target nor the start's own room shortens it, while other host forms still pay the
    target. A delegated child's Map answers to the physical window alone too. A Max the window lowers runs in Low, temporarily, for the
    window, and there the Map stands at its parts before any memory goes below a telling."""
    talk = [syn.conversation(i, words=20, word_chars=1_000) for i in range(150)]  # whole far above Low's target
    core = booked(_core(tmp_path, [_service(size=20_000)], snapshot=syn.snapshot(conversations=talk)), tmp_path,
                  body="Its whole mechanism and why. " * 60)  # a Map Max cannot hold beside my tellings in 100,000
    owner_low = _plan(core, 1_000_000, "low")
    stepped = [row["section_id"] for row in _host_facts(owner_low, "low")]
    assert owner_low.initial_mode == "low" and "runtime" in stepped and "body_map" not in stepped, stepped
    assert WHOLE_MAP in _text(owner_low, "low") and MAP_PARTS not in _text(owner_low, "low")
    lowered = _plan(core, 100_000, "max")
    rows = {row["section_id"]: row for row in _host_facts(lowered, "low")}
    assert lowered.initial_mode == "low" and lowered.low_projection.memory_facts["floor"]["mode_switch"] == {
        "from": "max", "to": "low"}
    assert "start_room" not in rows["body_map"] and MAP_PARTS in _text(lowered, "low")
    assert WHOLE_MAP not in _text(lowered, "low")
    child = dataclasses.replace(core, delegated_focus=True, memory_view_json=memory_view.snapshot_json(
        syn.snapshot(role="child", story=syn.pages(40, 30_000))))  # its own whole story above Low's target
    child_plan = _plan(child, 1_000_000, "low")
    assert "body_map" not in {row["section_id"] for row in _host_facts(child_plan, "low")}
    assert WHOLE_MAP in _text(child_plan, "low")


def _repository_map(core):
    """``core`` with the Body Map and its parts form as ``reference_books.body_map`` renders this repository's books."""
    from ouroboros.reference_books import body_map, load_reference_book

    repo = Path(__file__).resolve().parents[1]
    view, parts = body_map(*(load_reference_book(repo, book) for book in ("architecture", "development")))
    return dataclasses.replace(core, body_map=view.text, body_map_parts=parts)


def test_a_delegated_focus_in_owner_max_carries_the_body_map_whole_or_runs_in_low_with_its_parts(tmp_path):
    """Max carries the Body Map whole in every focus, a delegated child's too. Its Max room is
    measured with the Map whole (priced as an integrating root's is); where Max does not hold, the child runs in Low
    with the Map at its parts like any focus. No window gives a Max frame with the Map at its parts."""
    child = dataclasses.replace(_repository_map(_core(tmp_path, snapshot=syn.snapshot(role="child", story=syn.pages(6, 3_000)))),
                                delegated_focus=True)
    root = dataclasses.replace(child, delegated_focus=False)
    probed = {}

    def start(window):
        plan = probed[window] = probed.get(window) or _plan(child, window, "max")
        return plan.initial_mode == "max"

    lost, held = 12_000, 40_000  # Low, Max: the boundary lies between
    assert not start(lost) and start(held)
    while held - lost > 1:
        middle = (lost + held) // 2
        lost, held = (lost, middle) if start(middle) else (middle, held)
    kept, lowered = probed[held], probed[lost]
    assert not _host_facts(kept, "max") and MAP_PARTS not in _text(kept, "max") and "## Body map\n" in _text(kept, "max")
    price = dict(kept.max_projection.memory_facts["max_room"]["parts"])["body map"]
    assert price == dict(_plan(root, held, "max").max_projection.memory_facts["max_room"]["parts"])["body map"]
    assert lowered.initial_mode == "low" and "body_map" in [row["section_id"] for row in _host_facts(lowered, "low")]
    assert MAP_PARTS in _text(lowered, "low") and "## Body map\n" not in _text(lowered, "low")
    # Both sides of the boundary, and the cost of the whole Map inside Max: one token below it the child runs in Low
    # (a Max that could carry the parts would still hold there); in that Low the parts keep every item of its memory
    # at a telling, which the same start with a Map that cannot step does not.
    assert held - lost == 1 and _below_a_telling(lowered, "low") == 0
    unstepping = _plan(dataclasses.replace(child, body_map_parts=""), lost, "max")
    assert unstepping.initial_mode == "low" and _below_a_telling(unstepping, "low") > 0
    for window, plan in probed.items():  # never Max with the Map addressed, in the start or its Max projection
        assert "body_map" not in [row["section_id"] for row in _host_facts(plan, "max")], window


def _below_a_telling(plan, mode):
    """How many memory items this projection shows below a telling (an address or less)."""
    from ouroboros import memory_forms

    snapshot = memory_view.snapshot_from_json(plan.core.memory_view_json)
    level = memory_floor.kept_level(snapshot, plan.projection(mode).memory_facts)
    return len({ident.removeprefix("bare:") for step, idents in level.addressed if not step.endswith("a")
                for ident in idents if memory_forms.stage_of(snapshot, step, ident) not in memory_forms.TELLING_STAGES})


def _mature_memory(pages=300, replies=2_400, conversations=100, role="integrator"):
    """A mature probe memory: 300 authored story pages, 2,400 replies, 100 other conversations (or as many as asked)."""
    from ouroboros.chronicle_lead import LEAD_RULE_VERSION

    story = [dict(page, lead_rule=LEAD_RULE_VERSION,
                  text=(f"Room {i} " + "recorded understanding " * 38 + f"settled point {i}. ") + syn.prose(2_400))
             for i, page in enumerate(syn.pages(pages, 2_400))]
    return syn.snapshot(role=role, story=story,
                        room_facts=syn.room(lane1=[syn.spoken(i, "ouroboros", 400) for i in range(replies)], lane2=2),
                        conversations=[syn.conversation(i, engaged=False) for i in range(conversations)])


def _map_form(plan, mode=None):
    """``whole`` or ``parts``: the Body Map as this projection carries it, by its receipt and its text alike."""
    mode = mode or plan.initial_mode
    stepped, text = "body_map" in [row["section_id"] for row in _host_facts(plan, mode)], _text(plan, mode)
    assert (MAP_PARTS in text) is stepped and ("## Body map\n" in text) is not stepped, mode
    return "parts" if stepped else "whole"


@pytest.mark.parametrize("window", [872_000, 64_000, 0], ids=["wide", "below-nano-target", "unmeasured"])
def test_the_owners_nano_target_steps_the_body_map_before_it_takes_memory_below_a_telling(tmp_path, window):
    """Under the owner's own Nano a mature memory goes below a telling on any route: by Nano's target on a wide
    window and on a route that names no window, by the window itself where it is at or under that target. In each
    the Map stands at its parts first (BIBLE P1), so fewer items go below a telling than with the Map held whole (a
    Map with no parts form) and no fewer than with no Map at all, and its receipt names what took it: the owner's
    Nano budget, or the window. A small memory that fits keeps the Map whole on the same route. The owner's own Low
    never takes memory below a telling by its target: on a window that holds both, the Map is whole."""
    mature = _repository_map(_core(tmp_path / "mature", snapshot=_mature_memory()))
    nano = _plan(mature, window, "nano")
    [row] = _host_facts(nano, "nano")
    assert nano.initial_mode == "nano" and row["section_id"] == "body_map" and _map_form(nano) == "parts"
    by_budget = window != 64_000  # there the window is the tighter bound, and the step is the window's
    assert row.get("start_room", False) == ("nano" if by_budget else False)
    assert ("omitted for the owner's Nano mode budget, not for the physical window" in _text(nano, "nano")) is by_budget
    held_whole = _plan(dataclasses.replace(mature, body_map_parts=""), window, "nano")
    no_map = _plan(dataclasses.replace(mature, body_map="", body_map_parts=""), window, "nano")
    counts = [_below_a_telling(plan, "nano") for plan in (no_map, nano, held_whole)]
    assert counts[0] <= counts[1] < counts[2], counts
    small = _repository_map(_core(tmp_path / "small", snapshot=syn.snapshot(story=syn.pages(3, 2_000))))
    roomy = _plan(small, window, "nano")
    assert not _host_facts(roomy, "nano") and _map_form(roomy) == "whole" and not _below_a_telling(roomy, "nano")
    low = _plan(mature, 872_000, "low")
    assert low.initial_mode == "low" and not _host_facts(low, "low") and "## Body map\n" in _text(low, "low")
    assert _below_a_telling(low, "low") == 0


@pytest.mark.parametrize("preferred", ["max", "low", "nano"])
def test_one_order_for_the_body_map_in_every_owner_mode_at_one_window(tmp_path, preferred):
    """The approved order (BIBLE P1), the same in my human's Max, Low and Nano at one window and one memory. Where
    the window holds my tellings and the work's room beside the whole Map, it is whole. One step narrower the Map
    takes its parts first, and that keeps every memory item at a telling; the same start with a Map that cannot step
    puts items below one. And a start whose memory has nothing a window could take below a telling keeps the Map
    whole even where the work's room is short: nothing needs its room (the owner's "keep both maps whole")."""
    core = booked(_core(tmp_path / "told", snapshot=_mature_memory(40, 60, 6)), tmp_path / "told", body="Its mechanism. " * 12)
    unstepping, saving = dataclasses.replace(core, body_map_parts=""), map_saving(core)
    assert saving > 2_000

    def told(window):  # every memory item of this start stands at a telling or better beside the whole Map
        try:
            plan = _plan(unstepping, window, preferred)
        except fit.MinimumViewUnfit:
            return False
        return _below_a_telling(plan, plan.initial_mode) == 0

    short, holds = 12_000, 80_000  # under the owner's Nano target, so the window is what presses in all three modes
    assert not told(short) and told(holds)
    while holds - short > 1:
        middle = (short + holds) // 2
        short, holds = (short, middle) if told(middle) else (middle, holds)
    whole, stepped = _plan(core, holds, preferred), _plan(core, holds - saving // 2, preferred)
    assert _map_form(whole) == "whole" and _below_a_telling(whole, whole.initial_mode) == 0
    assert _map_form(stepped) == "parts" and _below_a_telling(stepped, stepped.initial_mode) == 0
    [row] = [row for row in _host_facts(stepped) if row["section_id"] == "body_map"]
    assert "start_room" not in row  # the window's step: the work's room is the window's own bound
    kept = _plan(unstepping, holds - saving // 2, preferred)
    assert _below_a_telling(kept, kept.initial_mode) > 0  # what the Map's step spared

    bare = booked(_core(tmp_path / "bare", snapshot=syn.snapshot()), tmp_path / "bare", body="Its mechanism. " * 12)
    request = _plan(bare, 2_000_000, "low").low_projection.estimated_tokens
    tight = _plan(bare, request + RESERVE + 862, preferred)  # the request fits; the work's room is short
    assert tight.initial_mode != "max" or preferred != "max"
    assert _map_form(tight) == "whole" and not _host_facts(tight)


@pytest.mark.parametrize("preferred", ["max", "low"])
def test_an_optional_excerpt_yields_before_the_body_map_and_the_map_only_when_that_is_not_enough(tmp_path, preferred):
    """The Map's turn among the host forms comes after an optional excerpt (owner 23.11.2026, 1: optional maps, then
    the Map). Where the excerpt's own map makes the room my tellings need, the Map stays whole; where it does not,
    the Map takes its parts too, and in both no memory item goes below a telling."""
    excerpt = HostSection("design_bible", "## Design bible\n\n" + "A rule of the room. " * 1_200, "## Design bible (map)")
    core = booked(_core(tmp_path, head=[excerpt], snapshot=_mature_memory(30, 60, 6)), tmp_path, body="Its mechanism. " * 12)
    unstepping = dataclasses.replace(core, body_map_parts="", dynamic_head_sections=capture_sections(
        [excerpt.full_text], prefix="head"))
    map_step = map_saving(core)
    excerpt_step = memory_floor.view_tokens(excerpt.full_text) - memory_floor.view_tokens(excerpt.operative_view) - 100
    assert map_step > 2_000 and excerpt_step > 2_000

    def told(window):  # every memory item at a telling or better with the excerpt and the Map both whole
        try:
            plan = _plan(unstepping, window, preferred)
        except fit.MinimumViewUnfit:
            return False
        return _below_a_telling(plan, plan.initial_mode) == 0

    short, holds = 12_000, 200_000
    assert not told(short) and told(holds)
    while holds - short > 1:
        middle = (short + holds) // 2
        short, holds = (short, middle) if told(middle) else (middle, holds)
    first, both = (_plan(core, holds - gap, preferred) for gap in (excerpt_step // 2, excerpt_step + map_step // 2))
    assert [row["section_id"] for row in _host_facts(first)] == ["design_bible"] and _map_form(first) == "whole"
    assert [row["section_id"] for row in _host_facts(both)] == ["design_bible", "body_map"] and _map_form(both) == "parts"
    assert _below_a_telling(first, first.initial_mode) == _below_a_telling(both, both.initial_mode) == 0


@pytest.mark.parametrize("focus", ["root-max", "child-max", "root-low", "child-low", "root-nano"])
def test_a_measured_recount_takes_the_body_map_to_its_parts_before_memory_goes_below_a_telling(tmp_path, focus):
    """A refusal or a re-count in the middle of a work, in every focus and mode (BIBLE P1): my tellings pay first and
    the Map stays whole while they can. Where they cannot, the Map takes its parts before any item goes below a
    telling (a delegated focus's Max runs in Low for the window there, as a root's does), and only what is still
    owed after that takes memory below one. The call is the measured floor pass's own
    (``loop_round_limits._coarsen_memory_view``)."""
    role, preferred = focus.split("-")
    memory = _mature_memory(30, 60, 6, role="child" if role == "child" else "integrator")
    core = dataclasses.replace(booked(_core(tmp_path, snapshot=memory), tmp_path, body="Its mechanism. " * 12),
                               delegated_focus=role == "child")
    shown = _plan(core, 2_000_000, preferred)
    assert shown.initial_mode == preferred and _map_form(shown) == "whole"
    assert not any(shown.projection(preferred).memory_facts["floor"]["steps"].values())

    def size(level):
        return sum(memory_floor.tokens_of(chars(memory, level)) for chars in (memory_view.story_chars, memory_view.room_chars))

    tellings = size(memory_view.FULL_VIEW) - size(memory_floor.telling_level(memory))  # what my tellings can pay
    saving = map_saving(core)
    assert tellings > 2_000 and saving > 2_000

    def recount(refused):
        return shown.reproject_for_route(window_tokens=2_000_000, known_window=True, ratio=1.0, output_reserve=RESERVE,
            tool_schemas=[], start_mode=preferred, current_messages=shown.messages_for(preferred),
            refused_tokens=refused, telling_floor=True, admitted=True)

    paid = recount(tellings // 2)
    assert paid.initial_mode == preferred and _map_form(paid) == "whole" and _below_a_telling(paid, preferred) == 0
    assert any(paid.projection(preferred).memory_facts["floor"]["steps"].values())
    lowered = "low" if preferred == "max" else preferred
    stepped = recount(tellings + saving // 2)
    assert stepped.initial_mode == lowered and _map_form(stepped) == "parts" and _below_a_telling(stepped, lowered) == 0
    deeper = recount(tellings + saving + 1_000)
    assert deeper.initial_mode == lowered and _map_form(deeper) == "parts" and _below_a_telling(deeper, lowered) > 0


def test_a_low_start_the_window_cannot_hold_steps_the_body_map_to_its_parts(tmp_path):
    """Past the physical window beside reply reserve and my shortest telling the Body Map steps to its parts and
    addresses beside its retained full text, named a step for the window (no ``start_room``)."""
    core = booked(_core(tmp_path, snapshot=syn.snapshot()), tmp_path)
    request = _plan(core, 2_000_000, "low").low_projection.estimated_tokens  # the Map whole beside all of memory
    saving = map_saving(core)
    plan = _plan(core, request - saving // 2 + RESERVE, "low")  # physical shortfall, not the desired work margin
    rows, text = _host_facts(plan, "low"), _text(plan, "low")
    assert plan.initial_mode == "low" and [row["section_id"] for row in rows] == ["body_map"], rows
    assert "start_room" not in rows[0] and MAP_PARTS in text and WHOLE_MAP not in text
    assert "omitted for the physical window; retained full text: read_file(root='artifact_store'" in text
    # The parts still name every chapter of both books: its folder and extension once, then its name with its title.
    assert "`docs/architecture/<name>.md`:\ncore: Core" in text and "`docs/development/<name>.md`:\nchange: Change" in text

    # Without held heads, the cold fit prices the previously shown parts list.
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.held_notes import restored_capture
    from ouroboros.owner_wait import continuation_state
    from ouroboros.tools.compact_context import record_context_view

    ctx = SimpleNamespace(task_id=core.section_task_id, context_fit_plan=plan, active_context_mode="low")
    messages = plan.messages_for("low")
    record_context_view(ctx, messages, [])
    captured = dataclasses.replace(core, held_note_resume=restored_capture(continuation_state(ctx, messages, {}, {}, 1, [], set())))
    cold = _plan(captured, 2_000_000, "low")
    assert _host_facts(cold, "low") == rows
    [body] = rows
    source = read_actor_source_bytes(core.section_root, core.section_task_id, body["source_ref"]).decode()
    assert body["reader"] in _text(cold) and WHOLE_MAP in source and WHOLE_MAP not in _text(cold)
    # The held handle is absent in this actor's store: the cold replay retains the CURRENT Map's full text for this
    # reader at the same digest-named address, so the parts list stays — confirmed by reading it back, not by the
    # path's existence; without it the Map would show whole (its mechanisms in the request itself).
    unavailable = _plan(dataclasses.replace(captured, section_task_id="different-task"), 2_000_000, "low")
    [fresh] = _host_facts(unavailable, "low")
    assert fresh["section_id"] == body["section_id"] and not fresh.get("predicted")
    assert fresh["source_ref"] == body["source_ref"] and fresh["reader"] in _text(unavailable)
    retained = read_actor_source_bytes(core.section_root, "different-task", fresh["source_ref"]).decode()
    assert WHOLE_MAP in retained and WHOLE_MAP not in _text(unavailable)
    assert MAP_PARTS in _text(unavailable)

    # A store this actor can neither read the held handle from nor write to keeps the Map whole: the request shows
    # the Map's own text and no parts-list pointer.
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "task_results").write_text("not a directory", encoding="utf-8")
    unretainable = _plan(dataclasses.replace(captured, section_task_id="different-task", section_root=str(blocked)),
                         2_000_000, "low")
    assert not _host_facts(unretainable, "low") and MAP_PARTS not in _text(unretainable)
    assert WHOLE_MAP in _text(unretainable)

    # ``memory_read``'s counterfactual start models a fresh start of the same material: it never takes this
    # continuation's list (``counterfactual_starts``), and it writes nothing.
    fresh = _plan(core, 2_000_000, "low")
    files = sorted(path for path in Path(core.section_root).rglob("*") if path.is_file())
    pairwise = memory_floor.counterfactual_starts(cold, {"same": syn.snapshot()})
    facts = fresh.low_projection.memory_facts
    assert pairwise["measured"] and pairwise["arms"]["same"] == {
        "mode": fresh.initial_mode, "max_room": facts.get("max_room"), "steps": dict(facts["floor"]["steps"]),
        "host": [(row["section_id"], row["tokens_saved"]) for row in facts.get("host_sections_addressed", ())]}
    assert sorted(path for path in Path(core.section_root).rglob("*") if path.is_file()) == files


def test_a_required_book_yields_optional_maps_then_the_body_map_and_never_one_that_saves_nothing(monkeypatch):
    """A directly required handbook's first request takes every optional attachment's map first, then the Body Map's
    parts, one at a time (``memory_floor.yield_for_book``; owner 23.11.2026, 1); a Map whose parts list with its
    receipt saves nothing, or one with nowhere to retain its full text, stays whole."""
    monkeypatch.setattr(memory_floor, "retain_section", lambda part, **_kw: {
        "section_id": part.section_id, "reader": "read_file(root='artifact_store', path='host.md')"})
    body = HostSection("body_map", "## Body map\n\n" + "### Part\nSource: `docs/a/part.md`\nIts opening.\n" * 200,
                       "## Body map (parts and addresses)\n\n- Part: `docs/architecture/part.md`")
    design = HostSection("design_bible", "## Design bible\n\n" + "A rule of the room. " * 200, "## Design bible (map)")

    held, readerless = (SimpleNamespace(section_root=root, section_task_id="t") for root in ("r", ""))

    def ids(rows):
        return None if rows is None else [row["section_id"] for row in rows]

    first = memory_floor.yield_for_book([design, body], [], held, maps=True)
    assert ids(first) == ["design_bible"]
    second = memory_floor.yield_for_book([design, body], first, held, maps=True)
    assert ids(second) == ["design_bible", "body_map"] and second[-1]["tokens_saved"] > 0
    assert memory_floor.yield_for_book([design, body], second, held, maps=True) is None
    assert memory_floor.yield_for_book([body], [], held) is None  # not an unread Low start
    small = HostSection("body_map", "## Body map\n\n- Core: `docs/architecture/core.md`",
                        "## Body map (parts and addresses)\n\n- Core")  # shorter than its own, but not beside its receipt
    assert memory_floor.yield_for_book([small], [], held, maps=True) is None
    assert memory_floor.yield_for_book([body], [], readerless, maps=True) is None


@pytest.mark.parametrize("reader", ["retained", "source", "none"])
def test_the_maps_parts_point_at_what_their_actor_can_read(tmp_path, monkeypatch, reader):
    """Where the window takes the Map to its parts, the pointer beside them is one this actor can follow. An actor
    that reads retained captures back gets the retained full text. One that reads the shipped manuals but no
    retained capture (a Presence whose ceiling grants ``read_file`` and no artifact-store read) is pointed at the
    Map's own source in the system repository, and nothing is retained for it. One with no reader at all keeps the
    Map whole: no pointer it could follow exists. The same three for a refusal's Map rung (``lowered_maps``)."""
    from ouroboros.context_layout import body_map_section
    from ouroboros.reference_books import BODY_MAP_NO_READER, BODY_MAP_READER
    from ouroboros.section_floor import section_views

    line = BODY_MAP_NO_READER if reader == "none" else BODY_MAP_READER
    whole = "## Body map\n\n" + "What this part gives me and how I reach it. " * 400 + "\n\n" + line
    body = body_map_section(whole, "## Body map (parts and addresses)\n\n- Core: `docs/architecture/core.md`", steppable=True)
    source = "read_file(root='system_repo', path='docs/ARCHITECTURE.md')"
    assert body.reader == ("" if reader == "none" else source)
    retained = []
    monkeypatch.setattr(memory_floor, "retain_section", lambda part, **_kw: retained.append(part.section_id) or {
        "section_id": part.section_id, "source_ref": {"path": "kept.md"}, "reader": "read_file(root='artifact_store', path='kept.md')"})
    root = "store" if reader == "retained" else ""
    core = SimpleNamespace(section_root=root, section_task_id="t")
    price = lambda forms: memory_floor.view_tokens(forms.get("body_map", body.full_text)) + 1_000  # noqa: E731
    over = price({}) + RESERVE - 500  # the physical window is 500 tokens short of the request beside the whole Map
    marker = {"retained": "omitted for the physical window; retained full text: read_file(root='artifact_store', path='kept.md').]",
              "source": f"omitted for the physical window; its source: {source}.]"}
    for stepped in (memory_floor._host_steps([body], [], measure=price, window=over, reserve=RESERVE, ratio=1.0, saving=0,
                                             root=root, task_id="t")[0],
                    memory_floor.lowered_maps([body], core, [])):
        if reader == "none":
            assert stepped == [] and section_views([body], stepped) == {}
            continue
        [row] = stepped
        assert row["section_id"] == "body_map" and "start_room" not in row and row["tokens_saved"] > 0
        text = section_views([body], stepped)["body_map"]
        assert text.startswith("## Body map (parts and addresses)") and text.endswith(marker[reader])
        assert (row["source_ref"] is None, row["reader"] == source) == ((True, True) if reader == "source" else (False, False))
    assert retained == (["body_map", "body_map"] if reader == "retained" else [])
    # A window that holds the request leaves the Map whole for every one of them.
    held = memory_floor._host_steps([body], [], measure=price, window=price({}) + RESERVE, reserve=RESERVE, ratio=1.0,
                                    saving=0, root=root, task_id="t")[0]
    assert held == []


def test_an_unsent_and_a_sent_reprojection_keep_the_whole_body_map_while_the_window_holds_it(tmp_path):
    """Unsent start reprojection (e.g. first text rebind before any send) keeps the whole Body Map of a new Low start
    while tellings fit the physical frame, independent of assistant/user message roles; an already-sent view
    (unsent_start=False) keeps it whole too while the window holds my tellings beside it, and steps it only where not."""
    talk = [syn.conversation(i, words=8, word_chars=400) for i in range(40)]
    core = booked(_core(tmp_path, snapshot=syn.snapshot(conversations=talk)), tmp_path, body="Its practice. " * 20)
    roomy = _plan(core, 2_000_000, "low")
    assert not _host_facts(roomy, "low") and not any(roomy.low_projection.memory_facts["floor"]["steps"].values())
    assert WHOLE_MAP in _text(roomy, "low")

    tight = roomy.low_projection.estimated_tokens + RESERVE - 4_000
    # Applied review capsule carries role='assistant': unsent_start must not rely on message roles.
    messages = [*roomy.messages_for("low"), {"role": "assistant", "content": "Applied review capsule."}]
    args = dict(window_tokens=tight, known_window=True, ratio=1.0, output_reserve=RESERVE,
                tool_schemas=[], start_mode="low", current_messages=messages,
                admitted=True, room_if_narrowed=True)
    unsent = roomy.reproject_for_route(**args, unsent_start=True)
    assert unsent.initial_mode == "low"
    assert "body_map" not in {row["section_id"] for row in _host_facts(unsent, "low")}
    assert WHOLE_MAP in _text(unsent, "low")
    assert any(unsent.low_projection.memory_facts["floor"]["steps"].values())

    # A sent view: below Max the Map answers to the window in every frame, mid-run
    # reprojections included. It stays whole while my shortest tellings fit the window beside it: here memory folds
    # and the Map stands whole; it steps only where the window cannot hold my tellings beside it.
    sent = roomy.reproject_for_route(**args, unsent_start=False)
    assert "body_map" not in {row["section_id"] for row in _host_facts(sent, "low")}
    assert WHOLE_MAP in _text(sent, "low") and any(sent.low_projection.memory_facts["floor"]["steps"].values())
    shown = _plan(core, 400_000, "low")  # carbon's own numbers: a 300,000 window with a soft slack or a small refusal
    for soft in ({"slack_tokens": 5_000}, {"refused_tokens": 1_000}):
        rebound = shown.reproject_for_route(window_tokens=300_000, known_window=True, ratio=1.0, output_reserve=RESERVE,
                                            tool_schemas=[], start_mode="low", current_messages=shown.messages_for("low"), **soft)
        assert not _host_facts(rebound, "low") and WHOLE_MAP in _text(rebound, "low"), soft
    narrow_sent = roomy.reproject_for_route(**{**args, "window_tokens": tight - 36_000}, unsent_start=False)
    assert "body_map" in {row["section_id"] for row in _host_facts(narrow_sent, "low")}  # the window took it
    assert MAP_PARTS in _text(narrow_sent, "low") and WHOLE_MAP not in _text(narrow_sent, "low")

    bare_core = booked(_core(tmp_path / "bare", snapshot=syn.snapshot()), tmp_path / "bare")
    bare = _plan(bare_core, 2_000_000, "low")
    too_tight = bare.low_projection.estimated_tokens - map_saving(bare_core) // 2 + RESERVE
    narrow = bare.reproject_for_route(window_tokens=too_tight, known_window=True, ratio=1.0, output_reserve=RESERVE,
                                      tool_schemas=[], start_mode="low", start_work=True, unsent_start=True)
    assert "body_map" in [row["section_id"] for row in _host_facts(narrow, "low")]


def test_narrowed_by_window_reports_physical_body_map_cut_under_low_budget_without_false_positives(tmp_path, monkeypatch):
    """Selected Low with window > 250k: a passive soft-target host cut (start_room='low') is not a window narrowing
    and never re-reads the route, while a physical Body Map cut beside a required handbook (no start_room) triggers
    the horizon guard."""
    monkeypatch.setattr(fit, "_route_key", lambda _task: "synthetic-route-key")
    env = SimpleNamespace(drive_root=tmp_path)

    soft_core = booked(_core(tmp_path / "soft", [_service(size=60_000)],
                             snapshot=syn.snapshot(story=syn.pages(30, 40_000))), tmp_path / "soft")
    soft_calls = []

    def soft_resolver(_task, allow_fetch=True):
        soft_calls.append(allow_fetch)
        return {"provider": "test", "model": "synthetic"}, _evidence(1_000_000)

    soft_plan = fit.build_context_fit_plan(env, soft_core, {"id": "task-soft"}, preferred_mode="low",
                                           route_resolver=fit.RouteQuestion({"id": "task-soft"}, soft_resolver))
    assert len(soft_calls) == 1
    assert [row["section_id"] for row in _host_facts(soft_plan, "low")] == ["runtime"]
    assert _host_facts(soft_plan, "low")[0]["start_room"] == "low"

    req_core = dataclasses.replace(
        booked(_core(tmp_path / "req", snapshot=syn.snapshot()), tmp_path / "req", body="Its practice and why. " * 320),
        development_required=True)
    roomy = _plan(req_core, 1_000_000, "low")
    tight_over_target = roomy.low_projection.estimated_tokens + RESERVE - map_saving(req_core) // 2
    assert tight_over_target > 250_000
    tight_plan = _plan(req_core, tight_over_target, "low")
    assert [row["section_id"] for row in _host_facts(tight_plan, "low")] == ["body_map"]
    assert "start_room" not in _host_facts(tight_plan, "low")[0]
    assert "## Rule 149" in _text(tight_plan, "low")  # the required handbook stays whole

    req_windows = iter((tight_over_target, 1_000_000))
    req_calls = []

    def req_resolver(_task, allow_fetch=True):
        window = next(req_windows)
        req_calls.append(window)
        return {"provider": "test", "model": "synthetic"}, _evidence(window)

    recovered = fit.build_context_fit_plan(env, req_core, {"id": "task-req"}, preferred_mode="low",
                                           route_resolver=fit.RouteQuestion({"id": "task-req"}, req_resolver))
    assert req_calls == [tight_over_target, 1_000_000]
    assert not _host_facts(recovered, "low")
    assert "## Rule 149" in _text(recovered, "low") and WHOLE_MAP in _text(recovered, "low")


def test_fresh_low_keeps_the_body_map_below_desired_work_room_and_reports_the_remaining_room(tmp_path):
    """The Map survives while tellings, tools and the reply reserve fit physically; a required book's separate
    smaller-room notice survives too. Its unsendable ladder is covered in test_required_development_book."""
    base = booked(_core(tmp_path, snapshot=syn.snapshot()), tmp_path, body="Its practice and why. " * 100)
    roomy = _plan(base, 2_000_000, "low")
    req = roomy.low_projection.estimated_tokens
    # The physically sendable ordinary view keeps the Map even below its desired work room.
    ordinary_window = req + RESERVE + 862
    assert 862 < math.ceil(ordinary_window / 8)
    ordinary = _plan(base, ordinary_window, "low")
    assert ordinary.initial_mode == "low"
    assert not _host_facts(ordinary, "low")
    assert WHOLE_MAP in _text(ordinary, "low")
    assert "Start room is smaller than requested" in _text(ordinary, "low")

    # 2) Required book joined onto an ordinary view that fits its physical frame with room to spare:
    #    desired work-room shortfall (free ~862 < work_room) keeps the whole Map and emits the notice.
    required = dataclasses.replace(base, development_required=True)
    book = required.required_book_tokens()
    shortfall_window = req + book + RESERVE + 862
    assert 862 < math.ceil(shortfall_window / 8) < book + 862
    held = _plan(required, shortfall_window, "low")
    assert held.initial_mode == "low" and not _host_facts(held, "low")
    assert WHOLE_MAP in _text(held, "low") and "## Rule 149" in _text(held, "low")
    assert "Start room at planning was smaller than the" in _text(held, "low")
    assert "this task's directly required handbook takes" in _text(held, "low")
    assert "requested remain." not in _text(held, "low")


def test_fresh_low_without_memory_keeps_the_body_map_below_the_desired_work_margin(tmp_path):
    core = booked(_core(tmp_path), tmp_path)
    whole = _plan(core, 2_000_000, "low").low_projection.estimated_tokens
    window = whole + RESERVE + 862
    assert 862 < math.ceil(window / 8)
    held = _plan(core, window, "low")
    assert held.initial_mode == "low" and not _host_facts(held, "low")
    assert WHOLE_MAP in _text(held)


@pytest.mark.parametrize("memory", [False, True], ids=["no-snapshot", "with-tellings"])
def test_low_map_work_margin_exception_never_spares_unrelated_return_facts(tmp_path, memory):
    """The same request fits physically but misses desired work room: only the Body Map gets the exception.
    Return facts retain their existing start-room rule, also without a memory snapshot."""
    returned = HostSection("return_facts_words", "A carried return fact. " * 500, "Return facts at their source.")
    core = dataclasses.replace(booked(_core(tmp_path, snapshot=syn.snapshot() if memory else None), tmp_path),
                               dynamic_sections=(returned,))
    whole = _plan(core, 2_000_000, "low").low_projection.estimated_tokens
    window = whole + RESERVE + 862
    shown = _plan(core, window, "low")
    assert shown.initial_mode == "low"
    assert {row["section_id"] for row in _host_facts(shown, "low")} == {"return_facts_words"}
    assert returned.full_text not in _text(shown, "low")
    assert WHOLE_MAP in _text(shown, "low")


def test_a_cold_return_fits_as_its_fresh_start_did_with_the_maps_it_showed_whole(tmp_path):
    """A cold return's own fit is the fresh start's fit (#3649, the scratch equality): WORLD, the runtime facts and
    memory keep the forms a fresh start gives them under the owner's Low target, and the detailed maps it showed stay
    detailed: the cold signal changes no host or memory step (the chapter lists it showed: the test above)."""
    from ouroboros.held_notes import restored_capture
    from ouroboros.owner_wait import continuation_state
    from ouroboros.tools.compact_context import record_context_view

    talk = [syn.conversation(i, words=20, word_chars=1_000) for i in range(150)]  # whole far above Low's target
    core = booked(_core(tmp_path, [_service(size=20_000)], [_service("world", 3_000)],
                        snapshot=syn.snapshot(conversations=talk)), tmp_path, body="Its whole mechanism and why. " * 60)
    fresh = _plan(core, 1_000_000, "low")
    stepped = {row["section_id"] for row in _host_facts(fresh, "low")}
    assert "runtime" in stepped and not set(memory_floor.BOOK_MAPS) & stepped
    assert any(fresh.low_projection.memory_facts["floor"]["steps"].values())  # the target folded memory too
    ctx = SimpleNamespace(task_id=core.section_task_id, context_fit_plan=fresh, active_context_mode="low")
    messages = fresh.messages_for("low")
    record_context_view(ctx, messages, [])
    returned = dataclasses.replace(core, held_note_resume=restored_capture(
        continuation_state(ctx, messages, {}, {}, 1, [], set())))
    again = _plan(returned, 1_000_000, "low")
    assert _host_facts(again, "low") == _host_facts(fresh, "low") and _text(again, "low") == _text(fresh, "low")
    assert again.low_projection.memory_facts.get("floor") == fresh.low_projection.memory_facts.get("floor")
