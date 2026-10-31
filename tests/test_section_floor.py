"""Host section relocation at the ordinary context preparation and reader seams."""
from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import context, context_fit as fit, memory_floor, memory_view, section_floor
from ouroboros.context_budget import start_room_levels
from ouroboros.context_sections import runtime_host_section
from ouroboros.section_floor import HostSection, capture_sections
from tests import _memory_view_synthetic as syn

RESERVE = 8_192


def test_host_ladder_lists_live_producers_and_keeps_only_the_actual_protected_core():
    from ouroboros.context_layout import TIER0_ALWAYS_FULL

    assert memory_floor.HOST_STEPS == ("runtime", "review_pool", "review_continuity", "registry", "installed_skills", "drive_state",
        "world", "subagents",
        "architecture_map", "development_map")
    assert section_floor.PROTECTED == TIER0_ALWAYS_FULL
    # The overview joins the protected core by the owner's choice (20.10.2026): never an address.
    assert {"system", "bible", "identity", "scratchpad", "memory_marks", "shared_understanding"} <= section_floor.PROTECTED
    assert "shared_understanding" not in memory_floor.HOST_STEPS


@pytest.mark.parametrize("missing", ["window", "root", "task_id", "measure", "eligible"])
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
    sections = [HostSection(name, "whole " * 1000, "operative")
                for name in ("runtime", "registry", "installed_skills")]
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
        architecture_md="# Architecture", development_md="# Development", docs_need_development=True,
        semi_stable_text="IDENTITY WHOLE", dynamic_text="\n\n".join(dynamic),
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


@pytest.mark.parametrize("preferred", ["max", "low", "nano"])
@pytest.mark.parametrize("window", [0, 2_000_000])
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


@pytest.mark.parametrize("section_id", sorted(section_floor.PROTECTED))
def test_protected_sections_never_shrink_even_with_an_operative_view(tmp_path, section_id):
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
    # An owner turn (no ceiling) and a Presence ceiling that grants read_file: pointers resolve.
    assert context._section_root_for(env, SimpleNamespace()) == str(tmp_path)
    assert context._section_root_for(env, SimpleNamespace(task_contract=_presence_ceiling("read_file"))) == str(tmp_path)
    # A Presence ceiling without read_file: a pointer would name a source this actor cannot
    # open, so the preparation leaves its host sections whole (an empty section root).
    assert context._section_root_for(env, SimpleNamespace(task_contract=_presence_ceiling())) == ""
    assert context._section_root_for(env, SimpleNamespace(task_metadata={"task_contract": _presence_ceiling()})) == ""
    # An unreadable carrier also keeps the sections whole rather than publishing a dead pointer.
    assert context._section_root_for(env, SimpleNamespace(task_contract={"capability_ceiling": {"bad": 1}})) == ""
    core = _core(tmp_path, [_service(size=12_000)], snapshot=syn.snapshot(story=syn.pages(2, 5_000)))
    whole = dataclasses.replace(core, section_root="")
    plan = _plan(whole, 128_000, "nano")
    assert _host_facts(plan, "nano") == [] and "observed details" in _text(plan, "nano")
    assert not (tmp_path / "task_results/artifacts").exists()



def test_host_sections_then_index_precede_memory_and_people_step_last(tmp_path):
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
    # The index then gives way before any memory step.
    smaller = _plan(core, 50_000)
    assert [r["section_id"] for r in _host_facts(smaller)] == ["runtime", "knowledge_index"]
    assert smaller.max_projection.memory_facts["floor"]["steps"] == {}
    # At increasing physical pressure retold pages/replies go before human words.
    middle = _plan(core, 20_000)
    assert "F5l" in middle.max_projection.memory_facts["floor"]["steps"]
    assert "F7" not in middle.max_projection.memory_facts["floor"]["steps"]
    smallest = middle.reproject_for_route(window_tokens=RESERVE + 1, known_window=True, ratio=1,
        output_reserve=RESERVE, tool_schemas=[])
    steps = smallest.max_projection.memory_facts["floor"]["steps"]
    assert steps["F5"] == 3 and steps["F2"] == 1 and steps["F7"] == 1


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


def test_declared_inputs_use_the_same_book_map_floor_without_loading_shared_memory(tmp_path, monkeypatch):
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
    assert not core.memory_view_json
    assert core.section_root == str(env.drive_root) and core.section_task_id == task["id"]
    unknown = _plan(core, 0, "nano")
    assert not _host_facts(unknown)
    small = unknown.reproject_for_route(window_tokens=RESERVE + 1, known_window=True, ratio=1,
        output_reserve=RESERVE, tool_schemas=[], start_mode="nano")
    assert _host_facts(small)
    addressed = {row["section_id"] for row in _host_facts(small)}
    assert addressed <= {"architecture_map", "development_map", "runtime", "review_pool", "subagents"}
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
