"""Held knowledge through the ordinary request, its fit, and its continuation."""
from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import yaml

from ouroboros import context, context_budget, context_fit, held_notes, knowledge, knowledge_versions, memory_view
from ouroboros.agent import Env, _restore_saved_clocks
from ouroboros.memory import Memory
from ouroboros.owner_wait import continuation_state
from ouroboros.projects_registry import create_project
from ouroboros.realtime_setup import input_from_context_plan
from ouroboros.tools.knowledge import _knowledge_read
from ouroboros.tools.tool_context import ToolContext
from tests import _memory_view_synthetic as synthetic

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


def _plan(world, window=0, *, mode="max", task=TASK, ctx=None, core=None):
    evidence = SimpleNamespace(route_fp="held-note-fixture", status="asserted" if window else "unprobeable",
                               stale=False, window_tokens=window)
    return context_fit.build_context_fit_plan(
        world[0], core if core is not None else _core(world, task, ctx), task, preferred_mode=mode,
        route_resolver=lambda *_a, **_kw: ({"provider": "test", "model": "fixture"}, evidence))


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
    assert head in _held(_plan(world, 300_000, mode="nano", task=project_task))
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
    assert emergency.initial_mode == "nano" and ("b", "reader") in _cuts(emergency)
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


def test_a_focus_without_the_exact_reader_keeps_its_head_whole(world):
    from ouroboros.presence_authority import PresenceCapabilityCeiling, PresenceToolGrant, presence_ceiling_payload

    head = "Unabridged place knowledge. " * 2_300
    _write(world, "place-facts", head, summary="What the place note covers.")
    grant = PresenceToolGrant("knowledge_read")
    sha = "a" * 64
    ceiling = PresenceCapabilityCeiling("fixture", sha, sha, sha, sha, "main", 4, (grant,), (), sha)

    def plan(authority):
        ctx = ToolContext(repo_dir=world[0].repo_dir, drive_root=world[0].drive_root, task_id=TASK["id"],
                          task_contract={"capability_ceiling": presence_ceiling_payload(authority)})
        return _plan(world, 30_000, mode="low", ctx=ctx)

    allowed, blocked = plan(ceiling), plan(replace(ceiling, tool_grants=()))
    assert _cuts(allowed) == [("place-facts", "summary")] and head not in _held(allowed)
    assert not _cuts(blocked) and head in _held(blocked)


def test_reader_retains_an_unjournaled_revision_before_the_note_moves(world, monkeypatch):
    note = _write(world, "outside-edit", "Original journaled words.")
    outside = note.raw.replace(b"Original journaled words.", b"Exact observed bytes.\r\n\r\n## History\r\nHalf the note.")
    note.address.path.write_bytes(outside)
    observed = knowledge.read_knowledge_note(note.address)
    with pytest.raises(knowledge_versions.VersionUnavailable):
        knowledge_versions.find_version(note.address, observed.revision)

    def no_history(_self):
        pytest.fail("start reads only the recorded claims and the claimed source")

    with monkeypatch.context() as patch:
        patch.setattr(knowledge_versions.NoteHistory, "__iter__", no_history)
        plan = _plan(world)
    assert "Exact observed bytes." in _held(plan) and observed.revision in _held(plan)
    note.address.path.rename(note.address.path.with_name("moved.md"))
    ctx = ToolContext(repo_dir=world[0].repo_dir, drive_root=world[0].drive_root, task_id=TASK["id"])
    read = _knowledge_read(ctx, note.address.topic, scope="global", version=observed.revision)
    assert read.endswith(outside.decode("utf-8"))
    assert "Topic 'outside-edit' not found in global." in _knowledge_read(ctx, note.address.topic, scope="global")
    wrong_address = _knowledge_read(ctx, "other", scope="global", version=observed.revision)
    assert "version_unavailable" in wrong_address
    knowledge_versions._held_version_path(note.address, observed.revision).write_bytes(b"Changed retained bytes.")
    damaged = _knowledge_read(ctx, note.address.topic, scope="global", version=observed.revision)
    assert "version_unavailable" in damaged and "Changed retained bytes." not in damaged


def test_reprojection_voice_and_cold_resume_keep_the_captured_note_after_an_edit(world):
    note = _write(world, "moving", "First version of the held head. " * 1_200, summary="First summary.")
    ctx = ToolContext(repo_dir=world[0].repo_dir, drive_root=world[0].drive_root, task_id=TASK["id"])
    initial = ctx.context_fit_plan = _plan(world, 150_000, ctx=ctx)
    messages = initial.messages_for(initial.initial_mode)
    persisted = json.loads(json.dumps(continuation_state(ctx, messages, {}, {}, 1, [], set())))
    voice = input_from_context_plan(initial, model="voice-fixture", selected_tools=[], output_allowance_tokens=RESERVE)
    assert HEADING + _held(initial) in json.loads(voice.messages_json)[0]["content"][-1]["text"]

    changed = _write(world, "moving", "Edited version of the held head.", revision=note.revision, summary="New summary.")
    fresh = _plan(world, 150_000)
    assert changed.revision in _held(fresh) and "Edited version" in _held(fresh)
    projected = initial.reproject_for_route(window_tokens=150_000, known_window=True, ratio=1,
        output_reserve=RESERVE, tool_schemas=[], start_mode="max", select_mode="low",
        current_messages=[*messages, {"role": "assistant", "content": "Later reply."}])
    assert _held(projected) == _held(initial)
    assert projected.messages_for(projected.initial_mode)[-1]["content"] == "Later reply."

    cold = ToolContext(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root, task_id=ctx.task_id)
    cold.model_wait_context = None
    _restore_saved_clocks(cold, TASK, persisted)
    resumed = _plan(world, 150_000, ctx=cold)
    assert _held(resumed) == _held(initial)
    initial_small = _plan(world, 12_000, mode="low", core=initial.core)
    resumed_small = _plan(world, 12_000, mode="low", ctx=cold)
    assert _cuts(initial_small) == _cuts(resumed_small) == [("moving", "summary")]
    assert "First summary." not in _held(initial_small)  # originally in the index
    assert "First summary." in _held(resumed_small) and "New summary." not in _held(resumed_small)
    _restore_saved_clocks(cold, TASK, {"messages": messages})  # saved before held notes existed
    assert not _held(_plan(world, 150_000, ctx=cold))
