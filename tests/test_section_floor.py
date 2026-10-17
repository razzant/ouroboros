"""Host section relocation at the ordinary context preparation and reader seams."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import context, context_fit as fit, memory_floor, memory_view, section_floor
from ouroboros.context_sections import runtime_host_section
from ouroboros.section_floor import HostSection, capture_sections
from tests import _memory_view_synthetic as syn
from tests.test_cache_optimization import _make_env_and_memory

RESERVE = 8_192


@pytest.fixture(autouse=True)
def capacity_facts(monkeypatch):
    monkeypatch.setattr(fit, "_route_calibration_ratio", lambda *_a: 1.0)
    monkeypatch.setattr(fit, "main_output_reserve_tokens", lambda **_kw: RESERVE)
    monkeypatch.setattr("ouroboros.capability_evidence.resolve_main_token_density",
                        lambda *_a: (1.0, "cold_estimate"))


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
    core = _core(tmp_path, [_service(size=15_000), "protected dynamic text"], [_service("shared_understanding", 5_000)],
                 syn.snapshot(story=syn.pages(2, 4_000)))
    legacy = dataclasses.replace(core, dynamic_sections=(), dynamic_head_sections=())
    selected, baseline = _plan(core, window, preferred), _plan(legacy, window, preferred)
    for mode in memory_floor.MODES:
        assert selected.projection(mode).system_content_json == baseline.projection(mode).system_content_json
        assert selected.messages_for(mode) == baseline.messages_for(mode)
        assert selected.projection(mode).memory_facts == baseline.projection(mode).memory_facts
    assert not (tmp_path / "state/context_views").exists()


def test_target_and_working_margin_alone_do_not_address_host_sections(tmp_path):
    core = _core(tmp_path, [_service(size=12_000)], snapshot=syn.snapshot(story=syn.pages(2, 5_000)))
    full = _plan(core, 0, "nano")
    tokens = full.nano_projection.estimated_tokens
    assert tokens + RESERVE > 40_000  # the owner Nano target is smaller than this request
    exact = _plan(core, tokens + RESERVE, "nano")
    assert not _host_facts(exact, "nano")
    assert core.dynamic_text in _text(exact, "nano")
    pressured = _plan(core, tokens + RESERVE - 1, "nano")
    assert _host_facts(pressured, "nano")


@pytest.mark.parametrize("section_id", sorted(section_floor.PROTECTED))
def test_protected_sections_never_shrink_even_with_an_operative_view(tmp_path, section_id):
    protected = HostSection(section_id, f"## {section_id}\n" + "whole core " * 2_000, "forbidden replacement")
    core = _core(tmp_path, [protected, _service(size=3_000)])
    plan = _plan(core, RESERVE + 1)
    assert protected.full_text in _text(plan)
    assert "forbidden replacement" not in _text(plan)
    assert plan.messages_for("max")[1]["content"] == "WORDS THAT STARTED THE WORK"
    assert section_id not in {row["section_id"] for row in _host_facts(plan)}
    disposition = fit.measure_main_fit(plan, plan.messages_for("max"), [], drive_root=tmp_path,
        profile="owner_max", rendered_mode="max", round_id="r1", automatic_pass_used=True)
    assert disposition.predicted_capacity_miss
    assert disposition.action == "send"  # an estimator miss never hard-refuses


@pytest.mark.parametrize("short", ["the same tiny section", "longer than the tiny section " * 50])
def test_an_address_must_be_shorter_than_the_section(tmp_path, short):
    section = HostSection("runtime", "the same tiny section", short)
    plan = _plan(_core(tmp_path, [section]), RESERVE + 1)
    assert section.full_text in _text(plan)
    assert not _host_facts(plan)
    assert not (tmp_path / "state/context_views").exists()


def test_host_sections_precede_memory_and_memory_uses_the_smaller_fixed_part(tmp_path, monkeypatch):
    snapshot = syn.snapshot(story=syn.pages(3, 20_000), room_facts=syn.room("1", lane1=[syn.spoken(1, "human", 15_000)]))
    core = _core(tmp_path, [_service(size=35_000)], snapshot=snapshot)
    original = memory_floor.mode_views
    calls = []

    def observe(*args, **kwargs):
        assert list((tmp_path / "state/context_views/task-1").glob("*/runtime.md"))
        calls.append(kwargs["fixed_tokens_by_mode"])
        return original(*args, **kwargs)

    monkeypatch.setattr(memory_floor, "mode_views", observe)
    plan = _plan(core, 128_000)
    assert calls and calls[0]["max"] < 2_000
    assert _host_facts(plan)[0]["section_id"] == "runtime"
    assert plan.max_projection.memory_facts["floor"]["steps"] == {}
    assert memory_view.render_story(snapshot) in _text(plan)
    assert plan.max_projection.estimated_tokens + RESERVE <= 128_000
    # A remaining real shortage still reaches the ordinary memory ladder.
    tiny = _plan(core, 14_000)
    assert tiny.max_projection.memory_facts["floor"]["steps"]
    assert _host_facts(tiny)


def test_pointers_are_immutable_content_addresses_and_resolve_through_read_file(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    repo, root = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    section = _service(size=300)
    plan = _plan(_core(root, [section]), RESERVE + 800)
    row = _host_facts(plan)[0]
    path = root / row["path"]
    assert path.read_text() == section.full_text
    assert row["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest() == path.parent.name
    assert row["tokens_saved"] > 0 and row["reader"] in _text(plan)
    ctx = ToolContext(repo_dir=repo, drive_root=root, task_id="task-1")
    registry = ToolRegistry(repo_dir=repo, drive_root=root)
    registry.set_context(ctx)
    result = registry.execute_result("read_file", {"root": "runtime_data", "path": row["path"], "max_lines": 1000})
    assert result.status == "ok", result.text
    assert all(line in result.text for line in section.full_text.splitlines())
    stamp = path.stat().st_mtime_ns
    assert _host_facts(_plan(plan.core, RESERVE + 800))[0]["path"] == row["path"]
    assert path.stat().st_mtime_ns == stamp
    changed = dataclasses.replace(section, full_text=section.full_text + "\nA new capture.")
    later = _host_facts(_plan(_core(root, [changed]), RESERVE + 800))[0]
    assert later["path"] != row["path"] and path.read_text() == section.full_text


@pytest.mark.parametrize("failure", ["write", "conflicting_capture"])
def test_retention_failure_keeps_full_text_without_an_unreadable_pointer(tmp_path, monkeypatch, failure):
    section = _service(size=200)
    if failure == "write":
        def unavailable(*_a):
            raise OSError("synthetic unavailable store")
        monkeypatch.setattr(section_floor, "_retain", unavailable)
    else:
        path = tmp_path / section_floor.section_address("task-1", section)
        path.parent.mkdir(parents=True)
        path.write_text("Existing bytes must never be overwritten.")
    plan = _plan(_core(tmp_path, [section]), RESERVE + 1)
    assert not _host_facts(plan) and section.full_text in _text(plan)
    if failure == "conflicting_capture":
        assert path.read_text() == "Existing bytes must never be overwritten."


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
    assert section.full_text == "## Runtime context\n\n" + json.dumps(data, ensure_ascii=False, indent=2) + ledger
    operative = json.JSONDecoder().raw_decode(section.operative_view.split("\n\n", 1)[1])[0]
    for key in ("runtime_mode", "budget", "git_head", "task", "project_room", "task_contract",
                "task_authority_origin", "plan_review_authority"):
        assert operative[key] == data[key]
    assert operative["capabilities"]["mutative_subagent_surfaces"] == ["external_workspace"]
    assert operative["current_chat"]["running_tasks"]["count"] == 30
    assert operative["capabilities"]["delegation"]["reviewer_slots_last"]["count"] == 30
    assert section.operative_view.endswith(ledger) and data == before


def test_shared_understanding_view_is_its_outline_its_summary_place_and_its_size(tmp_path):
    from ouroboros.knowledge import resolve_knowledge_address, write_knowledge_note

    env, _memory = _make_env_and_memory(tmp_path)
    address = resolve_knowledge_address(env.drive_root, "overview", "global")
    # The second paragraph is as much the authored orientation as the first: no slice of prose is a view.
    orientation = ("# Our orientation\n\nWe intend to understand the entire situation.\n\n"
                   "We commit to telling the owner before any irreversible step.\n")
    body = "".join(f"\n## 2026-10-{day:02d} — dated block\n\nDetails that belong to the shared understanding.\n"
                   f"\n### A finer point of day {day}\n\nMore.\n" for day in range(1, 11))
    write_knowledge_note(address, orientation + body)
    section = context.build_knowledge_sections(env)[0]
    assert isinstance(section, HostSection) and section.section_id == "shared_understanding"
    assert section.full_text.endswith(orientation + body)
    view = section.operative_view
    headings = ["# Our orientation"] + [line for day in range(1, 11) for line in (
        f"## 2026-10-{day:02d} — dated block", f"### A finer point of day {day}")]
    assert "\n".join(headings) in view  # its own headings, verbatim and in order
    assert "We intend" not in view and "We commit" not in view and "Details that" not in view
    assert "It has no authored summary yet (no summary field); full text: " \
           "knowledge_read(topic='overview', scope='global')." in view
    source_heading = "## Shared understanding\n\nSource: knowledge_read(topic='overview', scope='global').\n\n"
    assert section.full_text.startswith(source_heading)
    assert f"Its body is {len(section.full_text) - len(source_heading)} characters under 21 headings." in view
    # Ouroboros's own summary field is named where it already renders: the knowledge index.
    from ouroboros.knowledge import read_knowledge_note

    assert write_knowledge_note(address, "", mode="edit", expected_revision=read_knowledge_note(address).revision,
                                summary="Where we stand and what we hold to.").ok
    summarized = context.build_knowledge_sections(env)
    assert "Its authored summary is the overview entry of the knowledge index in this context" \
        in summarized[0].operative_view
    index = summarized[1]
    assert index.section_id == "knowledge_index" and index.operative_view is None
    assert "Where we stand and what we hold to." in index.full_text
    # Under physical window pressure the floor addresses it like any host service block.
    addressed, receipts = section_floor.fit_host_sections(
        [section, _service("runtime", 20_000)], root=str(tmp_path), task_id="task-1",
        measure=lambda _addressed: 10 ** 6, window_tokens=50_000, known_window=True,
        output_reserve=RESERVE, ratio=1.0)
    assert set(addressed) == {"runtime", "shared_understanding"}
    row = next(row for row in receipts if row["section_id"] == "shared_understanding")
    assert (tmp_path / row["path"]).read_text(encoding="utf-8") == section.full_text
    assert addressed["shared_understanding"].startswith(view)


def test_health_keeps_every_nonpassing_line_and_counts_passes(tmp_path, monkeypatch):
    from ouroboros import context_health
    from tests._context_shared import _make_health_env

    env = _make_health_env(tmp_path)
    checks = ["OK: passing check with full details " * 50 for _ in range(30)]
    checks += ["CRITICAL: exact failure\nsecond line", "WARNING: pending obligation", "INFO: unknown state"]
    def observations(_env, found):
        found.clear()
        found.extend(checks)
    monkeypatch.setattr(context_health, "_collect_log_analysis_checks", observations)
    section = context_health.build_health_invariants(env)
    assert section.full_text == "## Health Invariants\n\n" + "\n".join(f"- {line}" for line in checks)
    for line in checks[-3:]:
        assert line in section.operative_view
    assert "30 passing checks" in section.operative_view
    assert checks[0] not in section.operative_view


def test_preparation_with_caller_window_fits_large_sections_at_128k_and_records_receipt(tmp_path, monkeypatch):
    from ouroboros import context_health
    from ouroboros.knowledge import resolve_knowledge_address, write_knowledge_note

    env, memory = _make_env_and_memory(tmp_path)
    orientation = "# Our orientation\n\nKeep the authored orientation intact.\n"
    body = "\nEvidence and long authored background. " * 1_000
    write_knowledge_note(resolve_knowledge_address(env.drive_root, "overview", "global"), orientation + body)
    def observations(_env, found):
        found[:] = ["OK: detailed passing check " + "evidence " * 1_000 for _ in range(90)]
        found.append("CRITICAL: preserve this failing check")
    monkeypatch.setattr(context_health, "_collect_log_analysis_checks", observations)
    fetched = []
    def route(_task, *, allow_fetch):
        fetched.append(allow_fetch)
        return {"provider": "test", "model": "small-route"}, _evidence(0)
    monkeypatch.setattr(context, "_context_fit_route", route)
    monkeypatch.setattr(context, "get_context_mode", lambda: "max")
    task = {"id": "task-1", "text": "Exact work-starting words.", "type": "task", "chat_id": 1,
            "task_contract": {"obligations": ["Preserve ABI"], "grants": ["source read"]},
            "metadata": {"current_chat": {"chat_id": 1, "running_tasks": [
                {"task_id": str(i), "text": "historical detail " * 1_000} for i in range(50)]}}}
    from ouroboros.tools.registry import ToolContext
    ctx = ToolContext(repo_dir=env.repo_dir, drive_root=env.drive_root, task_id="task-1")
    messages, cap = context.build_llm_messages(env, memory, task, ctx=ctx, known_window_tokens=128_000)
    plan = ctx.context_fit_plan
    assert fetched == [False] and plan.window_tokens == 128_000 and plan.status == "confirmed"
    assert plan.initial_mode == "max" and plan.preferred_mode == "max"
    assert plan.max_projection.estimated_tokens + RESERVE <= 128_000
    text = _text(plan)
    assert "CRITICAL: preserve this failing check" in text and orientation + body in text
    assert "Preserve ABI" in text and "source read" in text and "## Task Contract Discipline" in text
    assert messages[1]["content"] == task["text"]
    facts = _host_facts(plan)
    assert {row["section_id"] for row in facts} >= {"health_invariants", "runtime"}
    assert "shared_understanding" not in {row["section_id"] for row in facts}
    assert cap["memory_view"]["host_sections_addressed"] == facts == ctx.memory_view_facts["host_sections_addressed"]
    event = [json.loads(line) for line in (env.drive_root / "logs/events.jsonl").read_text().splitlines()
             if json.loads(line).get("type") == "context_memory_view"][-1]
    assert event["host_sections_addressed"] == facts
    for row in facts:
        source = env.drive_root / row["path"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == row["sha256"]
    # The typed sections retain the exact unsent capture, not a second reading.
    assert body in plan.core.dynamic_head_text


def test_route_reprojection_only_narrows_and_recomputes_from_the_original_capture(tmp_path):
    core = _core(tmp_path, [_service(size=15_000)], [_service("shared_understanding", 15_000)], syn.snapshot())
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


def test_previously_addressed_later_sections_are_restored_before_any_new_cut(tmp_path):
    first, later = _service("health_invariants", 300), _service("shared_understanding", 20_000)
    measure = lambda overrides: sum(memory_floor.view_tokens(overrides.get(s.section_id, s.full_text)) for s in (first, later))
    kwargs = dict(root=str(tmp_path), task_id="task-1", measure=measure, window_tokens=RESERVE + 3_000,
                  known_window=True, output_reserve=RESERVE, ratio=1.0)
    addressed, facts = section_floor.fit_host_sections([first, later], kept=[{"section_id": later.section_id}], **kwargs)
    assert set(addressed) == {later.section_id}
    assert {row["section_id"] for row in facts} == {later.section_id}


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_caller_window_rejects_values_that_are_not_positive_capacities(tmp_path, value):
    core = _core(tmp_path)
    with pytest.raises(ValueError, match="known_window_tokens"):
        fit.build_context_fit_plan(SimpleNamespace(drive_root=tmp_path), core, {"id": "t"},
            preferred_mode="max", known_window_tokens=value,
            route_resolver=lambda *_a, **_kw: ({"provider": "test", "model": "small"}, _evidence(0)))


def test_physical_mode_lowering_carries_host_addresses_from_the_shown_view(tmp_path, monkeypatch):
    monkeypatch.setattr(fit, "main_output_reserve_tokens", lambda **_kw: 65_536)
    core = _core(tmp_path, [_service("health_invariants", 12_000)], snapshot=syn.snapshot())
    initial = _plan(core, 100_000, "low")
    assert _host_facts(initial, "low") and not _host_facts(initial, "nano")
    lowered = initial.reproject_for_route(window_tokens=60_000, known_window=True, ratio=1.0,
        output_reserve=65_536, tool_schemas=[], start_mode="low", current_messages=initial.messages_for("low"))
    assert lowered.initial_mode == "nano"
    assert _host_facts(lowered, "nano") == _host_facts(initial, "low")
    assert core.dynamic_text not in _text(lowered, "nano")


def test_registry_skills_review_and_labelled_drive_supply_explicit_views(tmp_path, monkeypatch):
    from ouroboros import subagent_runtime
    from ouroboros.context_runtime_facts import snapshot_labelled
    from tests._context_shared import _make_health_env

    env = _make_health_env(tmp_path)
    (tmp_path / "memory/registry.md").write_text(
        "### source-one\n- **path:** memory/source.md\n- **updated:** today\n- **gaps:** none\n")
    registry = context._build_registry_digest(env)
    assert registry.operative_view.endswith("- source-one: updated=today")
    assert "| source-one | memory/source.md | today | none |" in registry.full_text
    skill = {"name": "example", "enabled": True, "executable_review": True,
             "type": "extension", "version": "1", "review_status": "PASS",
             "description": "Long source description " * 8, "when_to_use": "Selected by the mind",
             "review_gate": {"author_accepted": True}, "live_loaded": False,
             "live_reason": "pending restart"}
    monkeypatch.setattr("ouroboros.skill_loader.summarize_skills", lambda *_a: {"skills": [skill]})
    installed = context._build_installed_skills_section(env)
    for text in ("example", "PASS", "enabled", "untrusted data", "accepted by author", "pending restart"):
        assert text in installed.operative_view
    assert "Long source description" in installed.full_text and "Long source description" not in installed.operative_view
    monkeypatch.setattr(subagent_runtime, "_recent_review_records", lambda *_a: ([{
        "record_id": "record-one", "surface": "plan", "aggregate": "FAIL", "ts": "a recorded instant"}], "unknown"))
    review = subagent_runtime.review_records_block(drive_root=tmp_path, task_id="task-1")
    assert "record-one: plan, FAIL" in review.operative_view and "unknown" in review.operative_view
    assert "a recorded instant" in review.full_text and "a recorded instant" not in review.operative_view
    drive = context._drive_state_section(env)
    labelled = snapshot_labelled(drive, "capture instant")
    assert isinstance(labelled, HostSection)
    assert labelled.full_text == snapshot_labelled(str(drive), "capture instant")
    assert "capture instant" in labelled.operative_view and '"budget_drift_alert": false' in labelled.operative_view


def test_tool_schema_cost_and_route_density_both_count_toward_physical_pressure(tmp_path, monkeypatch):
    core = _core(tmp_path, [_service(size=3_000)])
    tokens = _plan(core, 0).max_projection.estimated_tokens
    schemas = [{"type": "function", "function": {"name": "read_file", "description": "schema detail " * 3_000,
                                                "parameters": {"type": "object", "properties": {}}}}]
    window = tokens + RESERVE + 1
    assert not _host_facts(_plan(core, window))
    with_tools = fit.build_context_fit_plan(SimpleNamespace(drive_root=tmp_path), core, {"id": "task-1"},
        preferred_mode="max", tool_schemas=schemas,
        route_resolver=lambda *_a, **_kw: ({"provider": "test", "model": "synthetic"}, _evidence(window)))
    assert _host_facts(with_tools)
    monkeypatch.setattr(fit, "_route_calibration_ratio", lambda *_a: 2.0)
    assert _host_facts(_plan(core, window))


def test_new_host_savings_pay_the_refused_deficit_before_memory_steps(tmp_path):
    core = _core(tmp_path, [_service(size=8_000)], snapshot=syn.snapshot(story=syn.pages(2, 8_000)))
    initial = _plan(core, 2_000_000)
    narrowed = initial.reproject_for_route(window_tokens=30_000, known_window=True, ratio=1.0,
        output_reserve=RESERVE, tool_schemas=[], refused_tokens=1_000)
    assert _host_facts(narrowed)
    assert narrowed.max_projection.memory_facts["floor"]["steps"] == {}


def _presence_ceiling(*tool_names):
    from ouroboros.presence_authority import build_presence_capability_ceiling, presence_ceiling_payload
    from ouroboros.presence_capabilities import PresenceProfileResolution, PresenceSelection, PresenceToolTarget
    from ouroboros.presence_runtime import ResolvedPresenceRuntime

    targets = [PresenceToolTarget("builtin", name) for name in tool_names] or [PresenceToolTarget("builtin", "chat_history")]
    resolution = PresenceProfileResolution(
        active=tuple(PresenceSelection(f"{index + 1:064x}", target) for index, target in enumerate(targets)),
        missing_required=(), missing_optional=(), orphaned=(), runtime=ResolvedPresenceRuntime("main", 10, 10, False),
        profile_fingerprint="a" * 64, selection_fingerprint="b" * 64, required_selections_present=True)
    ceiling = build_presence_capability_ceiling(skill_name="moderator", skill_content_hash="c" * 64,
                                                state_fingerprint="d" * 64, resolution=resolution)
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
    assert not (tmp_path / "state/context_views").exists()


def test_a_known_route_reply_allowance_replaces_mains_reserve(tmp_path, monkeypatch):
    core = _core(tmp_path, ["protected dynamic text"])
    monkeypatch.setattr(fit, "main_output_reserve_tokens", lambda **_kw: 999)
    env = SimpleNamespace(drive_root=tmp_path)
    resolver = lambda *_a, **_kw: ({"provider": "test", "model": "synthetic"}, _evidence(128_000))
    mains = fit.build_context_fit_plan(env, core, {"id": "task-1"}, preferred_mode="max", route_resolver=resolver,
                                       known_window_tokens=128_000)
    routes = fit.build_context_fit_plan(env, core, {"id": "task-1"}, preferred_mode="max", route_resolver=resolver,
                                        known_window_tokens=128_000, known_output_reserve_tokens=32_000)
    assert mains.output_reserve_tokens == 999 and routes.output_reserve_tokens == 32_000
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive evidenced allowance"):
            fit.build_context_fit_plan(env, core, {"id": "task-1"}, preferred_mode="max", route_resolver=resolver,
                                       known_window_tokens=128_000, known_output_reserve_tokens=bad)
