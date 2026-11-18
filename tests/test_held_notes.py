"""Held knowledge through the ordinary request, its fit, and its continuation."""
from __future__ import annotations

import errno
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from ouroboros import context, context_budget, context_fit, held_notes, knowledge, knowledge_versions, memory_view
from ouroboros.agent import Env, _restore_saved_clocks
from ouroboros.memory import Memory
from ouroboros.owner_wait import checkpoint_owner_wait, load_owner_wait, set_owner_wait
from ouroboros.projects_registry import create_project
from ouroboros.realtime_setup import input_from_context_plan
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


def test_reader_retains_an_unjournaled_revision_before_the_note_moves(world, monkeypatch):
    note = _write(world, "outside-edit", "Original journaled words.")
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
    note.address.path.rename(note.address.path.with_name("moved.md"))
    ctx = ToolContext(repo_dir=world[0].repo_dir, drive_root=world[0].drive_root, task_id=TASK["id"])
    read = _knowledge_read(ctx, note.address.topic, scope="global", version=observed.revision)
    assert read.endswith(outside.decode("utf-8"))
    assert "Topic 'outside-edit' not found in global." in _knowledge_read(ctx, note.address.topic, scope="global")
    wrong_address = _knowledge_read(ctx, "other", scope="global", version=observed.revision)
    assert "version_unavailable" in wrong_address
    retained.write_bytes(b"Changed retained bytes.")
    damaged = _knowledge_read(ctx, note.address.topic, scope="global", version=observed.revision)
    assert "version_unavailable" in damaged and "Changed retained bytes." not in damaged


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
    assert new.revision in _held(_plan(world)) and new_copy.read_bytes() == new.raw


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
    knowledge._NAVIGATION.pop(note.address, None)
    monkeypatch.setattr(knowledge, "time", SimpleNamespace(time_ns=lambda: note.address.path.stat().st_ctime_ns + 10**10))
    reads, read_bytes = [], Path.read_bytes

    def counted(path):
        if path == note.address.path:
            reads.append(path)
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counted)
    assert "First revision." in _held(_plan(world)) and len(reads) == 1
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


def test_reprojection_voice_and_cold_resume_keep_the_sent_forms_and_their_companions(world, monkeypatch):
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    head = "First version of the held head. " * 1_200
    note = _write(world, "moving", head, summary="First summary.")
    overview = _write(world, "overview", "Protected overview before the wait.\n\n## Still protected\nBelow the head.")
    ctx = ToolContext(repo_dir=world[0].repo_dir, drive_root=world[0].drive_root, task_id=TASK["id"])
    initial = ctx.context_fit_plan = _plan(world, 12_000, mode="low", ctx=ctx)
    assert _cuts(initial) == [("moving", "summary")] and "The summary is in the knowledge index." in _held(initial)
    ctx.active_context_mode = initial.initial_mode
    ctx.task_attempt = 1
    messages = initial.messages_for(initial.initial_mode)
    write_task_result(ctx.drive_root, ctx.task_id, STATUS_RUNNING, chat_id=TASK["chat_id"])
    checkpoint = checkpoint_owner_wait(ctx, messages, {}, {}, 1, [], set())
    set_owner_wait(ctx.drive_root, ctx.task_id, {**checkpoint, "state": "waiting"})
    voice = input_from_context_plan(initial, model="voice-fixture", selected_tools=[], output_allowance_tokens=RESERVE)
    assert HEADING + _held(initial) in json.loads(voice.messages_json)[0]["content"][-1]["text"]

    changed = _write(world, "moving", "Edited version of the held head.", revision=note.revision, summary="New summary.")
    new_overview = _write(world, "overview", "Protected overview after the wait.", revision=overview.revision)
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
    original_companions = {part.section_id: part.full_text for part in initial.core.dynamic_head_sections}
    for name in ("knowledge_index", "shared_understanding"):
        part = next(part for part in resumed.core.dynamic_head_sections if part.section_id == name)
        assert part.full_text == original_companions[name] and part.full_text in restored[0]["content"][-1]["text"]
    block = restored[0]["content"][-1]["text"]
    assert block.count("First summary.") == 1 and "New summary." not in block
    assert overview.revision in _held(resumed) and new_overview.revision not in block
    assert block.count("Protected overview before the wait.") == 1 and "Below the head." in block

    wider = _plan(world, 150_000, mode="low", ctx=cold)
    assert not _cuts(wider) and head in _held(wider) and note.revision in _held(wider)
    assert _blocks(wider)[-1]["text"].count("Protected overview before the wait.") == 1
    _restore_saved_clocks(cold, TASK, {"messages": messages})  # saved before held notes existed
    assert not _held(_plan(world, 150_000, ctx=cold))
