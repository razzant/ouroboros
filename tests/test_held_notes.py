"""Held knowledge through the ordinary request, its fit, and its continuation."""
from __future__ import annotations

import errno
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from ouroboros import context, context_budget, context_fit, held_notes, knowledge, knowledge_versions, memory_floor, memory_view
from ouroboros.agent import Env, _restore_saved_clocks
from ouroboros.memory import Memory
from ouroboros.owner_wait import checkpoint_owner_wait, continuation_state, load_owner_wait, set_owner_wait
from ouroboros.projects_registry import create_project
from ouroboros.realtime_setup import input_from_context_plan
from ouroboros.section_floor import section_views
from ouroboros.tools.knowledge import _knowledge_read
from ouroboros.tools.registry import ToolRegistry
from ouroboros.tools.tool_context import ToolContext
from tests import _memory_view_synthetic as synthetic
from tests._live_world import host_world as host_world

HEADING = "## Held knowledge notes"
TASK = {"id": "held-note-turn", "chat_id": 1, "type": "task", "text": "Newest owner assignment."}
RESERVE = 4096


@pytest.fixture
def world(tmp_path, monkeypatch):
    repo, data = tmp_path / "repo", tmp_path / "data"
    files = {
        "prompts/SYSTEM.md": "I am Ouroboros.",
        "BIBLE.md": "The complete constitution.",
        "docs/ARCHITECTURE.md": "# Body\n\nBody orientation.\n\n## Chapters\n\n- [Runtime](architecture/runtime.md)\n",
        "docs/architecture/runtime.md": "# Runtime\n\nRuntime orientation.\n\n## Full mechanism\n\n" + "Book text. " * 20_000,
        "docs/DEVELOPMENT.md": "# Engineering\n\nEngineering orientation.\n\n## Chapters\n\n- [Change](development/change.md)\n",
        "docs/development/change.md": "# Change\n\nChange orientation.\n\n## Full procedure\n\n" + "Handbook. " * 3_500,
    }
    for path, text in files.items():
        file = repo / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text, encoding="utf-8")
    memory = Memory(drive_root=data, repo_dir=repo)
    memory.ensure_files()
    monkeypatch.setattr(context, "utc_now_iso", lambda: "2026-11-18T20:00:00Z")
    monkeypatch.setattr(context, "build_health_invariants", lambda *_a, **_kw: "")
    monkeypatch.setattr("ouroboros.context_runtime_facts.disk_view", lambda *_a, **_kw: {})
    monkeypatch.setattr(context_fit, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)
    monkeypatch.setattr(context_fit, "main_output_reserve_tokens", lambda **_kw: RESERVE)
    return Env(repo_dir=repo, drive_root=data), memory


def _write(world, topic, head, *, claims=("room:1",), scope="global", revision=None, **metadata):
    source = {"type": "note", **metadata, **({"claims": list(claims)} if claims else {})}
    address = knowledge.resolve_knowledge_address(world[0].drive_root, topic, scope)
    content = "---\n" + yaml.safe_dump(source) + "---\n" + head
    result = knowledge.write_knowledge_note(address, content, writer="turn", expected_revision=revision)
    assert result.ok, result.status
    return result.current


def _core(world, task=TASK, ctx=None):
    return context._capture_context_core(*world, task, None, ctx)


def _route(window, **_kwargs):
    evidence = SimpleNamespace(route_fp="held-note-fixture", status="asserted" if window else "unprobeable",
                               stale=False, window_tokens=window)
    return {"provider": "test", "model": "fixture"}, evidence


def _plan(world, window=0, *, mode="max", task=TASK, ctx=None, core=None, resolver=None):
    return context_fit.build_context_fit_plan(
        world[0], core if core is not None else _core(world, task, ctx), task, preferred_mode=mode,
        route_resolver=resolver or (lambda *_a, **_kw: _route(window)))


def _rebind(monkeypatch, ctx, plan, messages, window):
    from ouroboros.loop_model_call import _rebind_context_fit_plan

    monkeypatch.setattr(context, "_context_fit_route", lambda *_a, **_kw: _route(window))
    rebound, mode = _rebind_context_fit_plan(plan, SimpleNamespace(_ctx=ctx), messages, model="fixture", use_local=False,
        start_mode=plan.initial_mode, tool_schemas=[], preferred_mode=plan.preferred_mode)
    assert rebound.initial_mode == mode
    return rebound


def _blocks(plan, mode=None):
    return plan.messages_for(mode or plan.initial_mode)[0]["content"]


def _held(plan, mode=None):
    return _blocks(plan, mode)[-1]["text"].partition(HEADING)[2]


def _cuts(plan, mode=None):
    return [(row["source_ref"]["read"]["arguments"]["topic"], row["held_note_form"])
            for row in plan.projection(mode or plan.initial_mode).memory_facts.get("host_sections_addressed", ())
            if "held_note_form" in row]


def test_default_request_is_byte_identical_and_only_recorded_claims_add_a_section(world, monkeypatch):
    for number in range(20):
        _write(world, f"ordinary-{number}", "Ordinary durable understanding.", claims=())
    with monkeypatch.context() as patch:
        patch.setattr(held_notes, "turn_sections", lambda *_a, **_kw: [])
        baseline = _plan(world)
    quiet = _plan(world)
    for mode in ("max", "low", "nano"):
        assert json.dumps(quiet.messages_for(mode)) == json.dumps(baseline.messages_for(mode))
        assert not _held(quiet, mode)
    assert quiet.core_sha256 == baseline.core_sha256
    assert all(not (world[0].drive_root / "memory" / folder).exists()
               for folder in ("knowledge_note_records", "knowledge_held_versions"))

    note = _write(world, "carried", "Adopted head.", claims=())
    authored = "---\ntype: note\nclaims: ['room:1']\n---\nAdopted head."
    note.address.path.write_text(authored, encoding="utf-8")
    assert not _held(_plan(world))  # a field authored outside the note writer has no adoption
    live = knowledge.read_knowledge_note(note.address)
    adopted = knowledge.write_knowledge_note(note.address, authored, writer="turn", expected_revision=live.revision)
    assert adopted.ok and adopted.current.raw == live.raw
    assert "Adopted head." in _held(_plan(world))


def test_claimed_room_reaches_parent_under_low_and_nano_and_skips_delegated_roles(world, monkeypatch):
    project = create_project(world[0].drive_root, "nova", name="Nova")
    room = project["chat_id"]
    head = "Working Nova fact. " * 2_000
    _write(world, "nova", head, claims=(f"room:{room}",), summary="Nova's authored summary.")
    project_task = {**TASK, "project_id": project["id"], "chat_id": room}
    monkeypatch.setattr(context_budget, "OWNER_LOW_TARGET_TOKENS", 1_000)
    monkeypatch.setattr(context_budget, "OWNER_NANO_TARGET_TOKENS", 1_000)

    parent = _plan(world, 300_000, mode="low", task=project_task)
    assert parent.initial_mode == "low"
    assert parent.low_projection.memory_facts["floor"]["target_missed"]
    assert "Low's target (1,000 tokens) is not met" in _blocks(parent)[-1]["text"]
    assert head in _held(parent) and not _cuts(parent)
    nano = _plan(world, 300_000, mode="nano", task=project_task)
    assert nano.initial_mode == "nano" and head in _held(nano) and not _cuts(nano)
    assert nano.nano_projection.memory_facts["floor"]["target_missed"] is True
    assert "Nano's target (1,000 tokens) is not met" in _blocks(nano)[-1]["text"]
    monkeypatch.setattr(context_budget, "OWNER_NANO_TARGET_TOKENS", 85_000)
    fits = _plan(world, 300_000, mode="nano", task=project_task, core=nano.core)
    assert fits.initial_mode == "nano" and _held(fits) == _held(nano) and not _cuts(fits)
    assert fits.nano_projection.memory_facts["floor"]["target_missed"] is False
    assert "Nano's target (" not in _blocks(fits)[-1]["text"]
    assert not _held(_plan(world, 300_000, mode="low"))  # Main is not the claimed room

    for role, dispatch in (("child", {}), ("nanny", {"configured_subagent": {"route": {"kind": "agent_session"}}})):
        delegated = {**project_task, "delegation_role": "subagent", **dispatch}
        helper = _plan(world, 300_000, mode="low", task=delegated)
        assert helper.low_projection.memory_facts["role"] == role
        assert helper.low_projection.memory_facts["room_id"] == str(room)
        assert not _held(helper)


def test_place_matches_are_exact_and_the_note_keeps_fenced_lines_and_crlf(world, monkeypatch):
    monkeypatch.setattr("ouroboros.return_facts.turn_sections", lambda *_a, **_kw: ["## Facts at this owner turn"])
    route = {"provider": "chat", "account_id": "team", "conversation_id": "nova", "thread_id": "t/1"}
    place = {"focus": {"place_id": "team-place", "room_chat_id": 789, "transport": route}}
    claims = ("place:team-place", "thread:chat/team/nova/t%2F1", "conversation:chat/Team/nova")
    head = "# Nova\r\n\r\n\x60\x60\x60\x60md\r\n## Within backticks\r\n\x60\x60\x60\r\n## Still in code\r\n\x60\x60\x60\x60\r\n" \
           "~~~\r\n## Within tildes\r\n~~~\r\n  ## Indented line stays\r\n"
    note = _write(world, "work/nova", head + "## Archived detail\r\nOLD BODY", claims=claims, title="Nova team")
    _write(world, "team", "No second channel.", claims=("conversation:chat/team/other",))
    task = {**TASK, "metadata": place}
    plan = _plan(world, task=task)
    block = _blocks(plan)[-1]["text"]
    assert block.index("## Facts at this owner turn") < block.index(HEADING)
    held = _held(plan)
    assert held.count("### Nova team") == 1 and "memory/knowledge/work/nova.md" in held
    assert note.revision in held and held.count("Exact reader:") == 1
    assert "'place:team-place', 'thread:chat/team/nova/t%2F1'" in held
    assert "conversation:chat/Team/nova" not in held and "No second channel." not in held
    assert held.endswith(head) and "Archived detail" not in held and "OLD BODY" not in held
    assert plan.messages_for(plan.initial_mode)[-1] == {"role": "user", "content": TASK["text"]}
    assert not _held(_plan(world))


def test_overview_is_only_a_reference_and_inactive_or_malformed_notes_hold_nothing(world):
    overview = _write(world, "overview", "Shared head.\n## Past orientation\nStill shared in full.")
    archived = _write(world, "inactive", "Archived body.")
    result = knowledge.write_knowledge_note(archived.address, "", mode="archive", reason="No longer in force",
                                            expected_revision=archived.revision)
    assert result.ok
    malformed = _write(world, "malformed", "Syntactically valid a moment ago.")
    malformed.address.path.write_text("---\nclaims: [\n---\nMalformed body.", encoding="utf-8")
    project = create_project(world[0].drive_root, "private", name="Private")
    _write(world, "separate", "Project shelf body.", scope=f"project:{project['id']}")
    _write(world, "active", "Head without a heading.")

    plan = _plan(world, task={**TASK, "project_id": project["id"]},
                 core=_core(world, {**TASK, "project_id": project["id"], "memory_view": {"room_id": "1"}}))
    dynamic, held = _blocks(plan)[-1]["text"], _held(plan)
    assert dynamic.count("Shared head.\n## Past orientation\nStill shared in full.") == 1
    assert "Shared head." not in held and "Shared understanding reference." in held
    assert f"knowledge_read(topic='overview', scope='global', version='{overview.revision}')" in held
    assert "source metadata unavailable; read the original note" in dynamic
    assert all(word not in held for word in ("inactive", "malformed", "separate", "Project shelf body."))
    assert "Head without a heading." in held


def test_claims_and_lifecycle_are_checked_on_the_bytes_that_supply_the_head(world, monkeypatch):
    active = _write(world, "active", "Still claimed.")
    edited = _write(world, "edited", "Former room head.")
    archived = _write(world, "archived", "Former active head.")
    malformed = _write(world, "malformed", "Former readable head.")
    lookup = knowledge.active_claims

    def after_inventory(*args, **kwargs):
        result = lookup(*args, **kwargs)
        edited.address.path.write_bytes(edited.raw.replace(b"room:1", b"room:2"))
        archived.address.path.write_bytes(archived.raw.replace(b"type: note", (
            b"type: note\narchive: {at: '2026-11-18', reason: 'Moved into history'}")))
        malformed.address.path.write_bytes(malformed.raw.replace(b"type: note", b"type: ["))
        return result

    monkeypatch.setattr(knowledge, "active_claims", after_inventory)
    held = _held(_plan(world))
    assert active.revision in held and "Still claimed." in held
    assert "Former " not in held


def test_window_steps_heads_one_at_a_time_after_maps_and_leaves_a_head_without_a_summary(world):
    for topic in ("c", "b", "a"):
        _write(world, topic, f"Head {topic}. " * 5_000, **({"summary": f"Authored telling of {topic}."} if topic != "b" else {}))
    captured = _core(world)
    # At 120k, summarizing a head in the whole-book frame could keep Max. Map the books first.
    roomy, maps, one, two = (_plan(world, window, core=captured) for window in (150_000, 120_000, 43_000, 28_000))
    assert roomy.initial_mode == "max" and "Book text. Book text." in _blocks(roomy)[0]["text"]
    assert maps.initial_mode == one.initial_mode == two.initial_mode == "low"
    assert "Book text. Book text." not in _blocks(maps)[0]["text"] and "Full mechanism" in _blocks(maps)[0]["text"]
    assert _cuts(roomy) == _cuts(maps) == []
    assert _cuts(one) == [("a", "summary")]
    assert _cuts(two) == [("a", "summary"), ("c", "summary")]
    assert "Head b. " * 5_000 in _held(two)
    assert ("Head c. " * 5_000 in _held(one)) and ("Head c. " * 5_000 not in _held(two))
    assert ("Head a. " * 5_000 in _held(one, "max")) and not _cuts(one, "max")
    assert "Head shortened to its authored summary for the physical window." in _held(one)
    assert _blocks(one)[-1]["text"].count("Authored telling of a.") == 1
    assert "Authored telling of a." not in _held(one)  # the whole index already shows it

    messages = [*roomy.messages_for("max"), {"role": "user", "content": "Owner's words during work."}]
    def rebind(window):
        return roomy.reproject_for_route(window_tokens=window, known_window=True, ratio=1, output_reserve=RESERVE,
            tool_schemas=[], start_mode="max", current_messages=messages, admitted=True)

    still_max = rebind(150_000)
    assert still_max.initial_mode == "max" and not _cuts(still_max)
    rebound = rebind(31_000)
    assert rebound.initial_mode == "low" and _cuts(rebound) == [("a", "summary")]
    assert rebound.messages_for("low")[1:] == messages[1:]

    emergency = _plan(world, 12_000, core=captured)
    assert emergency.initial_mode == "nano"
    assert _cuts(emergency) == [("a", "summary"), ("c", "summary"), ("b", "reader")]
    assert all(_blocks(emergency)[-1]["text"].count(f"Authored telling of {topic}.") == 1 for topic in ("a", "c"))
    assert "Head b. " * 5_000 not in _held(emergency)
    assert "knowledge_read(topic='b', scope='global', version=" in _held(emergency)


def test_a_stepped_head_supplies_its_summary_only_when_the_resident_index_does_not(world):
    for topic, note_type in (("a-person", "person-contact"), ("b-topic", "note")):
        _write(world, topic, f"Head for {topic}. " * 2_200, type=note_type, summary=f"Authored summary for {topic}.")
    _write(world, "ordinary", "An unclaimed note.", claims=(), summary="Whole index only. " * 8_000)
    captured = _core(world)
    roomy, small = (_plan(world, window, mode="low", core=captured) for window in (150_000, 17_800))
    assert not _cuts(roomy) and "Whole index only." in _blocks(roomy)[-1]["text"]
    assert _cuts(small) == [("a-person", "summary"), ("b-topic", "summary")]
    assert "Whole index only." not in _blocks(small)[-1]["text"]
    assert "Authored summary for a-person." not in _held(small)  # the outline keeps it
    assert "Authored summary for b-topic." in _held(small)  # the outline drops it
    for topic in ("a-person", "b-topic"):
        assert _blocks(small)[-1]["text"].count(f"Authored summary for {topic}.") == 1


def test_admitted_low_rebind_prices_emergency_nano_before_deciding_whether_it_buys_room(world, monkeypatch):
    head = "A long head without a summary. " * 2_800
    _write(world, "head", head)
    later = {"role": "user", "content": "Later owner words during the running task."}

    def rebound(contract):
        ctx = ToolContext(repo_dir=world[0].repo_dir, drive_root=world[0].drive_root, task_id=TASK["id"],
                          task_contract=contract)
        plan = _plan(world, 130_000, mode="low", ctx=ctx)
        assert not _cuts(plan) and head in _held(plan)
        messages = [*plan.messages_for("low"), later]
        result = _rebind(monkeypatch, ctx, plan, messages, 18_000)
        assert messages[1:] == [*plan.messages_for("low")[1:], later]
        return result

    rescued = rebound({})
    assert rescued.initial_mode == "nano" and _cuts(rescued) == [("head", "reader")]
    assert rescued.projected_tokens_with_tools("nano", []) + RESERVE <= 18_000
    no_room = rebound({"disabled_tools": ["knowledge_read"]})
    assert no_room.initial_mode == "low" and not _cuts(no_room) and head in _held(no_room)
    assert no_room.projected_tokens_with_tools("low", []) + RESERVE > 18_000


@pytest.mark.parametrize("deciding_cost", ["memory", "reserve"])
def test_admitted_rebind_compares_both_full_prices(world, deciding_cost):
    note = _write(world, "head", "A held head without a shorter reader. " * 1_000)
    ctx = ToolContext(repo_dir=world[0].repo_dir, drive_root=world[0].drive_root, task_id=TASK["id"],
                      task_contract={"disabled_tools": ["knowledge_read"]})
    memory = synthetic.snapshot(mark_count=1)
    captured = replace(_core(world, ctx=ctx), section_root="", memory_view_json=memory_view.snapshot_json(memory))

    def compare(low_connections, nano_connections, reserve):
        core = replace(captured, connections=(low_connections, nano_connections))
        initial = _plan(world, 130_000, mode="low", core=core)
        messages = initial.messages_for("low")
        low_input = initial.projected_tokens_with_tools("low", [])
        window = low_input + reserve - 16
        fit = dict(window_tokens=window, known_window=True, ratio=1.0, output_reserve=reserve, start_mode="low",
                   tool_schemas=[], current_messages=messages, admitted=True)
        plan = initial.reproject_for_route(**fit)
        nano = initial.reproject_for_route(**fit, select_mode="nano")
        assert note.revision in _held(plan) and plan.messages_for(plan.initial_mode)[1:] == messages[1:]
        return plan, nano, low_input

    if deciding_cost == "memory":
        low, nano, _ = compare("Low connections. " * 4, "", RESERVE)
        assert low.initial_mode == "low"
        assert "### Physical floor" in _blocks(nano)[-1]["text"]
        assert nano.projected_tokens_with_tools("nano", []) > low.projected_tokens_with_tools("low", [])
        roomy_nano, _, _ = compare("Low connections. " * 40, "", RESERVE)
        assert roomy_nano.initial_mode == "nano"
    else:
        low, nano, _ = compare("", "Nano connections. " * 30, RESERVE)
        assert low.initial_mode == "low"
        assert nano.projected_tokens_with_tools("nano", []) > low.projected_tokens_with_tools("low", [])
        reply = context_budget.NANO_MIN_HEADROOM_TOKENS + 1_000
        trimmed_reserve, _, low_input = compare("", "Nano connections. " * 30, reply)
        assert trimmed_reserve.initial_mode == "nano"
        nano_input = trimmed_reserve.projected_tokens_with_tools("nano", [])
        assert low_input < nano_input < low_input + reply - context_budget.NANO_MIN_HEADROOM_TOKENS


def test_physical_emergency_takes_only_the_inline_summary_past_the_bound(world):
    _write(world, "a-resident", "Person's head. " * 2_000, type="person", summary="This person stays in the index.")
    summary = "Ordinary authored telling. " * 14
    _write(world, "b-inline", "Topic head. " * 2_000, summary=summary)
    later_summary = "A second ordinary telling stays inline. " * 4
    _write(world, "c-inline", "Another topic head. " * 2_000, summary=later_summary)
    _write(world, "unclaimed", "Unclaimed note.", claims=(), summary="Whole index only. " * 6_000)
    captured = _core(world)
    window = 8_700

    def fit(padding):
        statement = TASK["text"] + "x" * padding
        core = replace(captured, user_content_json=json.dumps(statement))
        return _plan(world, window, mode="nano", core=core)

    plain = fit(0)
    summaries = [("a-resident", "summary"), ("b-inline", "summary"), ("c-inline", "summary")]
    assert _cuts(plain) == summaries
    padding = 4 * (window - RESERVE - plain.projected_tokens_with_tools("nano", []))
    assert padding >= 0
    inside, outside = fit(padding), fit(padding + 4)
    assert inside.projected_tokens_with_tools("nano", []) + RESERVE == window
    assert _cuts(inside) == summaries and summary in _held(inside)
    assert _cuts(outside) == [("a-resident", "summary"), ("b-inline", "reader"), ("c-inline", "summary")]
    assert outside.projected_tokens_with_tools("nano", []) + RESERVE <= window
    assert outside.messages_for("nano")[1]["content"] == TASK["text"] + "x" * (padding + 4)
    assert "This person stays in the index." in _blocks(outside)[-1]["text"] and summary not in _held(outside)
    assert later_summary in _held(outside)
    part = next(part for part in captured.dynamic_sections if part.section_id == "held_note:b-inline")
    for plan in (inside, outside):
        receipts = plan.nano_projection.memory_facts["host_sections_addressed"]
        final = section_views((*captured.dynamic_head_sections, part), receipts)[part.section_id]
        [receipt] = [row for row in receipts if row["section_id"] == part.section_id]
        assert receipt["tokens_saved"] == memory_floor.view_tokens(part.full_text) - memory_floor.view_tokens(final)


def test_resident_person_label_yields_for_the_whole_assignment_only_at_the_emergency_floor(world):
    summary = "The person's authored summary stays in the index."
    _write(world, "person", "Held person head. " * 2_000, type="person", summary=summary)
    captured = _core(world)
    window = 8_700
    plain = _plan(world, window, mode="nano", core=captured)
    assert _cuts(plain) == [("person", "summary")]
    padding = 4 * (window - RESERVE - plain.projected_tokens_with_tools("nano", []))
    assert padding >= 0

    for extra, form in ((0, "summary"), (4, "reader")):
        assignment = TASK["text"] + "x" * (padding + extra)
        plan = _plan(world, window, mode="nano", core=replace(captured, user_content_json=json.dumps(assignment)))
        assert plan.messages_for("nano")[1]["content"] == assignment
        assert plan.projected_tokens_with_tools("nano", []) + RESERVE <= window
        assert _cuts(plan) == [("person", form)]
        assert _blocks(plan)[-1]["text"].count(summary) == 1 and summary not in _held(plan)


def test_prepared_voice_prices_its_work_room_while_a_fitting_running_rebind_keeps_the_head(world, host_world, monkeypatch):
    from ouroboros.contracts.live_conversation import LiveOptions
    from ouroboros.realtime_setup import measure_session_wire, session_candidate

    head = "The claimed head in the prepared voice frame. " * 1_050
    _write(world, "work", head, summary="Authored work summary.")
    initial = _plan(world, 130_000, mode="low")
    assert not _cuts(initial) and head in _held(initial)
    host = host_world.host
    host.plan = host.ctx.context_fit_plan = initial
    host.messages[:] = initial.messages_for("low")
    host._reselect_tools("low")
    host.model = replace(host.model, facts=replace(host.model.facts, working_reserve_tokens=RESERVE, basis="estimator"))
    factory, options = host_world.provider["session"], LiveOptions()
    _, _, before, _ = measure_session_wire(host, initial, host.messages, host.schemas, factory, options)
    window = before["fit"]["measurement"]["estimated_input_tokens"] + RESERVE + 600
    host.model = replace(host.model, facts=replace(host.model.facts, conversation_window_tokens=window))
    _, _, prepared = session_candidate(host, factory, options)
    assert prepared["work_room"]["first_send"] and prepared["work_room"]["deficit_tokens"] == 0
    assert _cuts(host.plan) == [("work", "summary")] and head not in _held(host.plan)

    ctx = ToolContext(repo_dir=world[0].repo_dir, drive_root=world[0].drive_root, task_id=TASK["id"])
    running = _rebind(monkeypatch, ctx, initial, initial.messages_for("low"), window)
    assert running.initial_mode == "low" and not _cuts(running) and _held(running) == _held(initial)


def test_only_a_physical_head_step_reasks_the_early_route_question(world, monkeypatch):
    monkeypatch.setattr(context_fit, "_context_route", lambda _task: ({"provider": "claudexor"}, {}))
    monkeypatch.setattr(context_budget, "OWNER_LOW_TARGET_TOKENS", 1_000)
    head = "Claimed words before the final window is known. " * 1_500
    _write(world, "route", head, summary="Authored route summary.")
    captured = _core(world)
    calls = []

    def run(first):
        calls.clear()

        def resolver(_task, *, allow_fetch):
            window = first if not calls else 130_000
            calls.append(window)
            return _route(window)

        question = context_fit.RouteQuestion(TASK, resolver=resolver)
        return _plan(world, mode="low", core=captured, resolver=question)

    small = _plan(world, 20_000, mode="low", core=captured)
    assert _cuts(small) == [("route", "summary")] and not small.low_projection.memory_facts["floor"]["steps"]
    rechecked = run(20_000)
    assert calls == [20_000, 130_000] and not _cuts(rechecked) and head in _held(rechecked)
    target_only = run(130_000)
    assert calls == [130_000] and not _cuts(target_only) and head in _held(target_only)
    assert target_only.low_projection.memory_facts["floor"]["target_missed"]


def test_required_handbook_joins_after_the_ordinary_map_frame_has_fit(world):
    head = "A held note. " * 5_000
    _write(world, "a", head, summary="First summary.")
    _write(world, "b", head, summary="Second summary.")
    ordinary = _core(world)
    ordinary = replace(ordinary, development_md=ordinary.development_md + "Full handbook appendix. " * 1_500)
    required = replace(ordinary, development_required=True)
    fits = _plan(world, 65_000, mode="low", core=ordinary)
    also_fits = _plan(world, 65_000, mode="low", core=required)
    assert not _cuts(fits) and _held(also_fits) == _held(fits)
    assert "Handbook. Handbook." in _blocks(also_fits)[1]["text"]
    assert _cuts(_plan(world, 25_000, mode="low", core=ordinary)) == [("a", "summary"), ("b", "summary")]
    # The ordinary view of this smaller request fits with one head; the added handbook is owed whole.
    ordinary_only = _plan(world, 33_000, mode="low", core=ordinary)
    assert _cuts(ordinary_only) == [("a", "summary")]
    with pytest.raises(context_fit.MinimumViewUnfit, match="directly required handbook") as refusal:
        _plan(world, 33_000, mode="low", core=required)
    assert refusal.value.required_book_tokens > 0


def test_a_shortest_telling_gets_its_room_before_any_emergency_memory_form(world):
    head = "A claimed fact. " * 3_500
    _write(world, "global-topic", head, summary="The authored summary.")
    captured = _core(world)
    memory = synthetic.snapshot(role="integrator", story=synthetic.pages(1, 80_000))
    captured = replace(captured, memory_view_json=memory_view.snapshot_json(memory))
    full = _plan(world, 55_000, mode="low", core=captured)
    tight = _plan(world, 38_000, mode="low", core=captured)
    assert not _cuts(full) and head in _held(full)
    assert _cuts(tight) == [("global-topic", "summary")] and head not in _held(tight)
    assert "Page 0. " + synthetic.prose(80_000) in _blocks(tight)[-2]["text"]


def test_only_an_admitted_call_for_the_complete_bound_revision_can_shorten_a_head(world):
    from ouroboros.presence_authority import (PresenceArgumentBinding, PresenceCapabilityCeiling, PresenceToolGrant,
                                             presence_ceiling_payload)

    registry = ToolRegistry(repo_dir=world[0].repo_dir, drive_root=world[0].drive_root)
    ctx = registry._ctx
    ctx.task_id = TASK["id"]
    head = "Unabridged place knowledge. " * 2_300
    note = _write(world, "place-facts", head, summary="What the place note covers.")
    other = _write(world, "other-topic", "Another note.", claims=())
    project_note = _write(world, note.address.topic, head, scope="project:nova", claims=(), summary=note.summary)
    earlier = _write(world, note.address.topic, head, revision=note.revision, summary="Earlier telling.")
    note = _write(world, note.address.topic, head, revision=earlier.revision, summary=note.summary)
    args = {"topic": note.address.topic, "scope": "global", "version": note.revision}
    sha = "a" * 64
    ceiling = PresenceCapabilityCeiling("fixture", sha, sha, sha, sha, "main", 4, (), (), sha)

    def grant(**bound):
        tool = PresenceToolGrant("knowledge_read", tuple(PresenceArgumentBinding((key,), "static", static_value=value)
                                                         for key, value in bound.items()))
        return {"capability_ceiling": presence_ceiling_payload(replace(ceiling, tool_grants=(tool,)))}

    def fit(contract, *, restriction="", result_check):
        ctx.task_contract = contract
        plan = _plan(world, 30_000, mode="low", ctx=ctx)
        actual = registry.execute_result("knowledge_read", args)
        assert result_check(actual), actual.code
        if restriction:
            assert not _cuts(plan) and head in _held(plan)
            assert "Exact reader: " not in _held(plan) and restriction in _held(plan)
        else:
            assert _cuts(plan) == [("place-facts", "summary")] and head not in _held(plan)
            assert "Exact reader: knowledge_read(" in _held(plan)

    def complete(result):
        return result.status == "ok" and result.text.endswith(note.text)

    fit({}, result_check=complete)
    fit(grant(view="note", start_char=0, end_char=0), result_check=complete)
    fit({"disabled_tools": ["knowledge_read"]}, restriction="task_contract.disabled_tools",
        result_check=lambda result: result.code == "RESOURCE_CONSTRAINT_BLOCKED")
    fit({"capability_ceiling": presence_ceiling_payload(ceiling)}, restriction="positive capability ceiling",
        result_check=lambda result: "PRESENCE_CAPABILITY_BLOCKED" in result.text)
    for bindings in ({"topic": other.address.topic, "version": other.revision},
                     {"scope": "project:nova", "version": project_note.revision},
                     {"version": earlier.revision}, {"view": "outline"}, {"start_char": 10, "end_char": 20}):
        fit(grant(**bindings), restriction="Presence argument bindings",
            result_check=lambda result: result.status == "ok" and not complete(result))


def test_cold_restore_rechecks_the_saved_reader_against_the_current_grants(world, monkeypatch):
    head = "These held words must stay whole when the saved reader is denied. " * 390
    note = _write(world, "restored-reader", head, summary="An authored summary.")
    initial = _plan(world, 16_000, mode="low")
    assert _cuts(initial) == [(note.address.topic, "summary")]
    checkpoint = {**held_notes.continuation_capture(initial, "low"), "messages": initial.messages_for("low"),
                  "tool_schemas": []}
    args = {"topic": note.address.topic, "scope": "global", "version": note.revision}

    for allowed, window in ((True, 16_000), (False, 16_000), (True, 12_000)):
        registry = ToolRegistry(repo_dir=world[0].repo_dir, drive_root=world[0].drive_root)
        ctx = registry._ctx
        ctx.task_id = TASK["id"]
        ctx.model_wait_context = None
        ctx.task_contract = {} if allowed else {"disabled_tools": ["knowledge_read"]}
        _restore_saved_clocks(ctx, TASK, checkpoint)
        fresh = _plan(world, window, mode="low", ctx=ctx)
        messages = list(checkpoint["messages"])
        sent = _rebind(monkeypatch, ctx, fresh, messages, window)
        shown = messages[0]["content"][-1]["text"]
        result = registry.execute_result("knowledge_read", args)
        if allowed:
            assert result.status == "ok" and result.text.endswith(note.text)
            assert _held(sent) == _held(initial) and "Exact reader: knowledge_read(" in shown and head not in shown
            assert _cuts(sent) == [(note.address.topic, "summary")]
        else:
            assert result.code == "RESOURCE_CONSTRAINT_BLOCKED"
            assert not _cuts(sent) and head in shown and "Exact reader: knowledge_read(" not in shown
            assert "Exact reader unavailable:" in shown and "task_contract.disabled_tools" in shown
        checkpoint = json.loads(json.dumps(continuation_state(ctx, messages, {}, {}, 1, [], set())))


def test_reader_retains_an_unjournaled_revision_before_the_note_moves(world, monkeypatch):
    note = _write(world, "Outside-Edit", "Original journaled words.")
    outside = note.raw.replace(b"Original journaled words.", b"Exact observed bytes.\r\n\r\n## History\r\nHalf the note.")
    note.address.path.write_bytes(outside)
    observed = knowledge.read_knowledge_note(note.address)
    with pytest.raises(knowledge_versions.VersionUnavailable):
        knowledge_versions.find_version(note.address, observed.revision)

    history, original = [], knowledge_versions.NoteHistory.__iter__

    def checked(self):
        history.append(self.address.topic)
        return original(self)

    monkeypatch.setattr(knowledge_versions.NoteHistory, "__iter__", checked)
    plan = _plan(world)
    assert history == [note.address.topic]
    retained = knowledge_versions._held_version_path(note.address, observed.revision)
    assert retained.read_bytes() == outside
    history.clear()
    assert _held(_plan(world)) == _held(plan) and not history
    retained.write_bytes(b"Damaged while the live source still exists.")
    assert _held(_plan(world)) == _held(plan) and retained.read_bytes() == outside and not history
    assert "Exact observed bytes." in _held(plan) and observed.revision in _held(plan)
    registry = ToolRegistry(repo_dir=world[0].repo_dir, drive_root=world[0].drive_root)
    alias = note.address.topic.lower() if note.address.path.with_name("outside-edit.md").is_file() else note.address.topic
    pin = registry.execute_result("knowledge_read", {"topic": alias, "scope": "global", "version": observed.revision})
    reader = pin.meta["knowledge_source"]["read"]
    assert pin.text.endswith(observed.text) and reader["arguments"]["topic"] == note.address.topic
    note.address.path.rename(note.address.path.with_name("moved.md"))
    ctx = ToolContext(repo_dir=world[0].repo_dir, drive_root=world[0].drive_root, task_id=TASK["id"])
    assert registry.execute_result(reader["tool"], reader["arguments"]).text.endswith(observed.text)
    assert "Topic 'Outside-Edit' not found in global." in _knowledge_read(ctx, note.address.topic, scope="global")
    wrong_address = _knowledge_read(ctx, "other", scope="global", version=observed.revision)
    assert "version_unavailable" in wrong_address
    retained.write_bytes(b"Changed retained bytes.")
    damaged = _knowledge_read(ctx, note.address.topic, scope="global", version=observed.revision)
    assert "version_unavailable" in damaged and "Changed retained bytes." not in damaged


def test_a_remembered_journal_miss_checks_only_appended_history_until_its_topic_changes(world, monkeypatch):
    note = _write(world, "outside", "Originally journaled.")
    note.address.path.write_bytes(note.raw.replace(b"Originally journaled.", b"An outside edit.\n## History\nBody."))
    observed = knowledge.read_knowledge_note(note.address)
    journal = knowledge_versions.journal_path(note.address)
    path = knowledge_versions._held_version_path(note.address, observed.revision)
    real_lines = knowledge_versions._journal_lines
    offsets = []

    def lines(source, needle, *, after=0):
        offsets.append(after)
        return real_lines(source, needle, after=after)

    monkeypatch.setattr(knowledge_versions, "_journal_lines", lines)
    assert observed.revision in _held(_plan(world)) and offsets == [0] and path.read_bytes() == observed.raw
    for other in ("other", "another"):
        end = journal.stat().st_size
        _write(world, other, "An unrelated note.", claims=())
        offsets.clear()
        assert observed.revision in _held(_plan(world)) and offsets == [end] and path.read_bytes() == observed.raw
        offsets.clear()
        assert observed.revision in _held(_plan(world)) and not offsets

    # A failed source publication still captures the exact pre-image in its authored row.
    with monkeypatch.context() as patch:
        def no_source(*_args, **_kwargs):
            raise OSError(errno.ENOSPC, "fixture source write failed")

        patch.setattr(knowledge, "write_bytes_atomic", no_source)
        result = knowledge.write_knowledge_note(note.address, "Next words.", mode="append")
    assert result.reason == "publication_incomplete" and result.current.raw == observed.raw
    offsets.clear()
    shown = _held(_plan(world))
    assert observed.revision in shown and offsets[0] > 0 and offsets[1:] == [0]
    assert path.read_bytes().startswith(b"\xff") and not path.with_suffix(".checked").exists()
    note.address.path.unlink()
    registry = ToolRegistry(repo_dir=world[0].repo_dir, drive_root=world[0].drive_root)
    assert registry.execute("knowledge_read", {"topic": note.address.topic, "scope": "global",
                                              "version": observed.revision}).endswith(observed.text)


def test_journal_proof_keeps_readers_without_copies_and_retires_an_outside_edit(world, monkeypatch):
    note = _write(world, "journaled", "Journaled source bytes.")
    retained = knowledge_versions._held_version_path(note.address, note.revision)
    history, original = [], knowledge_versions.NoteHistory.__iter__

    def checked(self):
        history.append(self.address.topic)
        return original(self)

    monkeypatch.setattr(knowledge_versions.NoteHistory, "__iter__", checked)
    assert note.revision in _held(_plan(world)) and not history and not retained.exists()
    record_path = knowledge_versions.claim_record_path(note.address.shelf, note.address.topic)
    record = json.loads(record_path.read_bytes())
    del record["journal_proof"]  # a completed claim from the previous body
    record_path.write_text(json.dumps(record), encoding="utf-8")
    old = _plan(world)
    assert history == [note.address.topic] and 0 < retained.stat().st_size <= 32
    history.clear()
    assert _held(_plan(world)) == _held(old) and not history

    outside = note.raw.replace(b"Journaled source bytes.", b"Outside edit with its entire history below the head.\n## History\nDetail.")
    note.address.path.write_bytes(outside)
    observed = knowledge.read_knowledge_note(note.address)
    body = knowledge_versions._held_version_path(note.address, observed.revision)
    assert observed.revision in _held(_plan(world)) and body.read_bytes() == outside
    _write(world, note.address.topic, "Later writer puts the outside bytes on the journal's old side.", revision=observed.revision)
    assert 0 < body.stat().st_size <= 32 and body.read_bytes() != outside
    assert not body.with_suffix(".checked").exists()

    note.address.path.unlink()
    ctx = ToolContext(repo_dir=world[0].repo_dir, drive_root=world[0].drive_root, task_id=TASK["id"])
    assert _knowledge_read(ctx, note.address.topic, scope="global", version=note.revision).endswith(note.text)
    assert _knowledge_read(ctx, note.address.topic, scope="global", version=observed.revision).endswith(observed.text)

    new = _write(world, "new", "A completed publication from the current journal.")
    new_copy = knowledge_versions._held_version_path(new.address, new.revision)
    assert new.revision in _held(_plan(world)) and not new_copy.exists()
    journal = knowledge_versions.journal_path(new.address)
    journal.rename(journal.with_suffix(".moved"))
    _write(world, "other-source", "A valid row in the replacement journal.", claims=())
    assert journal.is_file()
    assert new.revision in _held(_plan(world)) and new_copy.read_bytes() == new.raw
    new.address.path.unlink()
    assert _knowledge_read(ctx, new.address.topic, scope="global", version=new.revision).endswith(new.text)


def test_a_read_denied_journal_cannot_back_a_held_reader(world, monkeypatch):
    note = _write(world, "restricted-journal", "Words retained before the reader is published.")
    journal = knowledge_versions.journal_path(note.address)
    path = knowledge_versions._held_version_path(note.address, note.revision)
    healthy = _held(_plan(world))
    assert "Exact reader: knowledge_read(" in healthy and not path.exists()
    open_path = Path.open

    def deny_read(path, mode="r", *args, **kwargs):
        if path == journal and mode == "rb":
            raise PermissionError(errno.EACCES, "fixture journal read denied")
        return open_path(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny_read)
    assert journal.is_file() and _held(_plan(world)) == healthy and path.read_bytes() == note.raw
    note.address.path.rename(note.address.path.with_name("moved.md"))
    registry = ToolRegistry(repo_dir=world[0].repo_dir, drive_root=world[0].drive_root)
    result = registry.execute_result("knowledge_read", {"topic": note.address.topic, "scope": "global", "version": note.revision})
    assert result.status == "ok" and result.text.endswith(note.text)


def test_failed_copy_can_only_withhold_a_reader_without_verified_journal_bytes(world, monkeypatch):
    note = _write(world, "journaled", "Recoverable without a second copy.")
    outside = _write(world, "outside", "Published before an outside edit.")
    outside.address.path.write_bytes(outside.raw.replace(b"Published before an outside edit.", b"Unjournaled bytes."))
    record = knowledge_versions.claim_record_path(note.address.shelf, note.address.topic)
    state = json.loads(record.read_bytes())
    del state["journal_proof"]
    record.write_text(json.dumps(state), encoding="utf-8")
    attempted, real_write = [], knowledge_versions.utils.write_bytes_atomic

    def no_retained_write(path, data, **kwargs):
        if "knowledge_held_versions" in path.parts:
            attempted.append(path)
            raise OSError(errno.ENOSPC, "fixture disk is full")
        return real_write(path, data, **kwargs)

    monkeypatch.setattr(knowledge_versions.utils, "write_bytes_atomic", no_retained_write)
    shown = _held(_plan(world))
    assert {path.parent for path in attempted} == {
        knowledge_versions._held_version_path(item.address, item.revision).parent for item in (note, outside)}
    assert f"Exact reader: knowledge_read(topic='journaled', scope='global', version={note.revision!r})" in shown
    assert "Exact reader unavailable: This revision could not be retained and is unavailable in the journal." in shown
    assert "Unjournaled bytes." in shown
    assert not attempted[0].parent.exists()
    note.address.path.unlink()
    ctx = ToolContext(repo_dir=world[0].repo_dir, drive_root=world[0].drive_root, task_id=TASK["id"])
    assert _knowledge_read(ctx, note.address.topic, scope="global", version=note.revision).endswith(note.text)


def test_navigation_hands_a_cold_or_edited_body_to_the_head_and_reopens_a_changed_file(world, monkeypatch):
    note = _write(world, "read-once", "First revision.")
    other = _write(world, "other-room", "Another room's words. " * 1_000, claims=("room:01",))
    knowledge._NAVIGATION.pop(note.address, None)
    knowledge._NAVIGATION.pop(other.address, None)
    monkeypatch.setattr(knowledge, "time", SimpleNamespace(time_ns=lambda: note.address.path.stat().st_ctime_ns + 10**10))
    reads, read_bytes = [], Path.read_bytes

    def counted(path):
        if path == note.address.path:
            reads.append(path)
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counted)
    handoffs, turn_sections = [], held_notes.turn_sections

    def handoff(root, task, spec, navigation, *, ctx):
        handoffs.append(set(navigation.get("bodies", ())))
        return turn_sections(root, task, spec, navigation, ctx=ctx)

    monkeypatch.setattr(held_notes, "turn_sections", handoff)
    cold = _plan(world)
    assert "First revision." in _held(cold) and len(reads) == 1
    assert handoffs == [{note.address}] and "other-room" in _blocks(cold)[-1]["text"] and "other-room" not in _held(cold)
    assert note.address in knowledge._NAVIGATION
    reads.clear()
    assert "First revision." in _held(_plan(world)) and len(reads) == 1
    note.address.path.write_bytes(note.raw.replace(b"First revision.", b"Edited revision."))
    reads.clear()
    assert "Edited revision." in _held(_plan(world)) and len(reads) == 1

    assign = knowledge.active_claims
    note.address.path.write_bytes(note.raw.replace(b"First revision.", b"At navigation."))

    def change_after_navigation(*args, **kwargs):
        selected = assign(*args, **kwargs)
        note.address.path.write_bytes(note.raw.replace(b"First revision.", b"At capture."))
        return selected

    monkeypatch.setattr(knowledge, "active_claims", change_after_navigation)
    reads.clear()
    assert "At capture." in _held(_plan(world)) and len(reads) == 2


def test_a_saved_form_replays_only_at_the_same_density_and_request_cost(world, monkeypatch):
    head = "The held head before the frame changed. " * 1_200
    _write(world, "saved", head, summary="An authored summary.")
    ctx = ToolContext(repo_dir=world[0].repo_dir, drive_root=world[0].drive_root, task_id=TASK["id"])
    monkeypatch.setattr(context_fit, "_route_calibration_ratio", lambda *_a, **_kw: 1.6)
    tight = _plan(world, 12_000, mode="low", ctx=ctx)
    assert _cuts(tight) == [("saved", "summary")]
    window = 48_000
    carried = _rebind(monkeypatch, ctx, tight, tight.messages_for("low"), window)
    assert _cuts(carried) == _cuts(tight)
    checkpoint = json.loads(json.dumps(held_notes.continuation_capture(carried, "low")))
    resumed = replace(tight.core, held_note_resume=checkpoint[held_notes.CAPTURE_KEY])

    same = _plan(world, window, mode="low", core=resumed)
    recomputed = _plan(world, window, mode="low", core=tight.core)
    assert _cuts(same) == _cuts(carried) and _held(same) == _held(carried)
    assert not _cuts(recomputed) and head in _held(recomputed)

    cost_changed = replace(resumed, user_content_json=json.dumps(TASK["text"] + " More owner words." * 20))
    repriced = _plan(world, window, mode="low", core=cost_changed)
    assert not _cuts(repriced) and _held(repriced) == _held(recomputed)

    monkeypatch.setattr(context_fit, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)
    recalibrated = _plan(world, window, mode="low", core=resumed)
    assert not _cuts(recalibrated) and _held(recalibrated) == _held(recomputed)


@pytest.mark.parametrize("outlined", [False, True], ids=["whole-index", "outlined-index"])
def test_reprojection_voice_and_cold_resume_keep_held_rows_and_refresh_unrelated_addresses(world, monkeypatch, outlined):
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    head = "First version of the held head. " * 1_200
    note = _write(world, "moving", head, type="person", title="First held title", summary="First summary.")
    overview = _write(world, "overview", "Protected overview before the wait.\n\n## Still protected\nBelow the head.")
    if outlined:
        previous = _write(world, "earlier-entry", "Prior unclaimed note.", claims=(),
                          summary="A long summary only present in the full navigation. " * 1_800)
    ctx = ToolContext(repo_dir=world[0].repo_dir, drive_root=world[0].drive_root, task_id=TASK["id"])
    initial = ctx.context_fit_plan = _plan(world, 12_000, mode="low", ctx=ctx)
    assert _cuts(initial) == [("moving", "summary")] and "The summary is in the knowledge index." in _held(initial)
    assert ("Outline: every active note" in _blocks(initial)[-1]["text"]) is outlined
    ctx.active_context_mode = initial.initial_mode
    ctx.task_attempt = 1
    messages = initial.messages_for(initial.initial_mode)
    assert "current-entry" not in messages[0]["content"][-1]["text"]
    write_task_result(ctx.drive_root, ctx.task_id, STATUS_RUNNING, chat_id=TASK["chat_id"])
    checkpoint = checkpoint_owner_wait(ctx, messages, {}, {}, 1, [], set())
    set_owner_wait(ctx.drive_root, ctx.task_id, {**checkpoint, "state": "waiting"})
    voice = input_from_context_plan(initial, model="voice-fixture", selected_tools=[], output_allowance_tokens=RESERVE)
    assert HEADING + _held(initial) in json.loads(voice.messages_json)[0]["content"][-1]["text"]

    changed = _write(world, "moving", "Edited version of the held head.", revision=note.revision,
                     type="person", title="New held title", summary="New summary.")
    new_overview = _write(world, "overview", "Protected overview after the wait.", revision=overview.revision)
    if outlined:
        previous.address.path.unlink()
    new_summary = previous.summary if outlined else "A new index entry."
    _write(world, "current-entry", "An ordinary note created during the wait.", claims=(), summary=new_summary)
    fresh = _plan(world, 150_000, mode="low")
    assert changed.revision in _held(fresh) and "Edited version" in _held(fresh)
    assert new_overview.revision in _held(fresh) and "Protected overview after the wait." in _blocks(fresh)[-1]["text"]

    cold = ToolContext(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root, task_id=ctx.task_id)
    cold.model_wait_context = None
    cold.task_attempt = 1
    handoff = {"wait_id": checkpoint["wait_id"], "restart_transaction_id": "fixture-restart"}
    persisted = load_owner_wait(cold, handoff)
    _restore_saved_clocks(cold, {**TASK, "_owner_wait_resume": handoff}, persisted)
    same_frame = _plan(world, 12_000, mode="low", ctx=cold)
    later = {"role": "assistant", "content": "Later reply."}
    restored = [*persisted["messages"], later]
    resumed = _rebind(monkeypatch, cold, same_frame, restored, 12_000)
    assert restored[-1] == later and _cuts(resumed) == _cuts(initial) and _held(resumed) == _held(initial)
    block = restored[0]["content"][-1]["text"]
    assert block.count("First summary.") == 1 and "New summary." not in block
    assert "New held title" not in block and "current-entry" in block and "earlier-entry" not in block
    if outlined:
        assert "moving — First held title" in block
        [index] = [row for row in resumed.low_projection.memory_facts["host_sections_addressed"]
                   if row["section_id"] == "knowledge_index"]
        assert any(part.section_id == "knowledge_index" and "current-entry" in part.full_text
                   for part in resumed.core.dynamic_head_sections)
        assert index["source_ref"]["sha256"] != next(
            row["source_ref"]["sha256"] for row in initial.low_projection.memory_facts["host_sections_addressed"]
            if row["section_id"] == "knowledge_index")
    else:
        assert "[First held title](<moving.md>)" in block and new_summary in block
    assert overview.revision in _held(resumed) and new_overview.revision not in block
    assert block.count("Protected overview before the wait.") == 1 and "Below the head." in block
    [receipt] = [row for row in resumed.low_projection.memory_facts["host_sections_addressed"] if row.get("held_note_form")]
    reader = receipt["source_ref"]["read"]
    registry = ToolRegistry(repo_dir=cold.repo_dir, drive_root=cold.drive_root)
    assert registry.execute(reader["tool"], reader["arguments"]).endswith(note.text)

    wider = _plan(world, 150_000, mode="low", ctx=cold)
    assert not _cuts(wider) and head in _held(wider) and note.revision in _held(wider)
    assert _blocks(wider)[-1]["text"].count("Protected overview before the wait.") == 1
    _restore_saved_clocks(cold, TASK, {"messages": messages})  # saved before held notes existed
    assert not _held(_plan(world, 150_000, ctx=cold))
