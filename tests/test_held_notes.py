"""Held knowledge through the ordinary request, its fit, and its continuation."""
from __future__ import annotations

import errno
import json
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from ouroboros import (context, context_budget, context_fit, context_view_facts, held_notes, knowledge, knowledge_versions,
                       memory_floor, memory_view)
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
QUEUE_FACT = {"note": "Queue observation unavailable; current capacity is unknown."}
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
        file.write_bytes(text.encode("utf-8"))
    memory = Memory(drive_root=data, repo_dir=repo)
    memory.ensure_files()
    monkeypatch.setattr(context, "utc_now_iso", lambda: "2026-11-18T20:00:00Z")
    monkeypatch.setattr(context, "build_health_invariants", lambda *_a, **_kw: "")
    monkeypatch.setattr("ouroboros.context_runtime_facts.disk_view", lambda *_a, **_kw: {})
    # The queue's dated observation is process-wide: another test's snapshot must not put its age into a request.
    monkeypatch.setattr(context, "_queue_context_fact", lambda _task: QUEUE_FACT)
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


def _plan(world, window=0, *, mode="max", task=TASK, ctx=None, core=None, resolver=None, tool_schemas=None):
    return context_fit.build_context_fit_plan(
        world[0], core if core is not None else _core(world, task, ctx), task, preferred_mode=mode,
        route_resolver=resolver or (lambda *_a, **_kw: _route(window)), tool_schemas=tool_schemas)


def _rebind(monkeypatch, ctx, plan, messages, window, *, tool_schemas=()):
    from ouroboros.loop_model_call import _rebind_context_fit_plan

    monkeypatch.setattr(context, "_context_fit_route", lambda *_a, **_kw: _route(window))
    rebound, mode = _rebind_context_fit_plan(plan, SimpleNamespace(_ctx=ctx), messages, model="fixture", use_local=False,
        start_mode=plan.initial_mode, tool_schemas=list(tool_schemas), preferred_mode=plan.preferred_mode)
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
        line = context_view_facts.memory_facts(quiet, mode)
        assert line == context_view_facts.memory_facts(baseline, mode) and "held note heads" not in line
    assert quiet.core_sha256 == baseline.core_sha256
    assert all(not (world[0].drive_root / "memory" / folder).exists()
               for folder in ("knowledge_note_records", "knowledge_held_versions"))

    note = _write(world, "carried", "Adopted head.", claims=())
    authored = "---\ntype: note\nclaims: ['room:1']\n---\nAdopted head."
    note.address.path.write_bytes(authored.encode("utf-8"))
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
    malformed.address.path.write_bytes(b"---\nclaims: [\n---\nMalformed body.")
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


def _index(plan, mode=None):
    """The form of the global knowledge index in this request: ``whole``, or the ``outline`` the window chose."""
    text = _blocks(plan, mode)[-1]["text"]
    return "outline" if "Outline: every active note" in text else "whole"


def _untaken_outlines(plan, mode=None):
    """The indexes and the registry of this frame that stand whole though the window could take their outline."""
    taken = {row["section_id"] for row in plan.projection(mode or plan.initial_mode).memory_facts.get("host_sections_addressed", ())}
    core = plan.core
    return {part.section_id for part in (*core.semi_stable_sections, *core.dynamic_sections, *core.dynamic_head_sections)
            if part.section_id in memory_floor.WHOLE_OR_OUTLINE_STEPS and part.section_id not in taken
            and part.operative_view is not None and len(part.operative_view) < len(part.full_text)}


def _ordered_world(world):
    """Three claimed heads (``b`` without an authored summary) beside an index that an unclaimed note makes long."""
    for topic in ("c", "b", "a"):
        _write(world, topic, f"Head {topic}. " * 5_000, **({"summary": f"Authored telling of {topic}."} if topic != "b" else {}))
    _write(world, "ordinary", "An unclaimed note.", claims=(), summary="Whole index only. " * 8_000)
    return _core(world)


def test_window_takes_maps_then_the_index_outline_then_heads_one_at_a_time_and_a_head_without_a_summary_stays(world, monkeypatch):
    captured = _ordered_world(world)
    # At 170k, summarizing two heads in the whole-book frame could keep Max. Map the books first.
    roomy, maps, outlined, one, two = (
        _plan(world, window, core=captured) for window in (200_000, 170_000, 80_000, 44_000, 30_000))
    assert roomy.initial_mode == "max" and "Book text. Book text." in _blocks(roomy)[0]["text"]
    assert all(plan.initial_mode == "low" for plan in (maps, outlined, one, two))
    assert "Book text. Book text." not in _blocks(maps)[0]["text"] and "Full mechanism" in _blocks(maps)[0]["text"]
    assert _index(roomy) == _index(maps) == "whole" and _cuts(roomy) == _cuts(maps) == []
    # One bound for the outline and the heads. At 80k the physical window would still hold the whole index beside the
    # heads; this start's own bound does not, so the outline pays first, and it is enough: every head stands whole.
    assert maps.projected_tokens_with_tools("low", []) + RESERVE <= 80_000
    assert _index(outlined) == "outline" and _cuts(outlined) == []
    assert all(f"Head {topic}. " * 5_000 in _held(outlined) for topic in "abc")
    assert "omitted for this start's room, not for the physical window" in _blocks(outlined)[-1]["text"]
    assert [_index(plan) for plan in (one, two)] == ["outline", "outline"]
    assert _cuts(one) == [("a", "summary")]
    assert _cuts(two) == [("a", "summary"), ("c", "summary")]
    assert "Head b. " * 5_000 in _held(two)
    assert ("Head c. " * 5_000 in _held(one)) and ("Head c. " * 5_000 not in _held(two))
    assert ("Head a. " * 5_000 in _held(one, "max")) and not _cuts(one, "max")
    assert "Head shortened to its authored summary for the physical window." in _held(one)
    assert _blocks(one)[-1]["text"].count("Authored telling of a.") == 1
    assert "Authored telling of a." in _held(one)  # the outline no longer shows it

    # A frame that holds no head keeps its index for the physical window alone: the same notes, never adopted.
    with monkeypatch.context() as patch:
        patch.setattr(held_notes, "turn_sections", lambda *_a, **_kw: [])
        unheld = _core(world)
    held_window = _plan(world, 120_000, core=unheld).projected_tokens_with_tools("low", []) + RESERVE  # as it stands
    short_room, physical = (_plan(world, held_window + spare, core=unheld) for spare in (2_000, -3_000))
    assert any(row.get("start_room") for row in short_room.low_projection.memory_facts["host_sections_addressed"])
    assert [_index(short_room), _index(physical)] == ["whole", "outline"] and not _held(short_room)

    messages = [*roomy.messages_for("max"), {"role": "user", "content": "Owner's words during work."}]
    def rebind(plan, window, mode="max"):
        return plan.reproject_for_route(window_tokens=window, known_window=True, ratio=1, output_reserve=RESERVE,
            tool_schemas=[], start_mode=mode, current_messages=messages, admitted=True)

    still_max = rebind(roomy, 150_000)
    assert still_max.initial_mode == "max" and not _cuts(still_max) and _index(still_max) == "whole"
    # A view already sent keeps no new work room: the outline first, then one head pays what the window still lacks.
    sent_outline, rebound = rebind(roomy, 60_000), rebind(roomy, 31_000)
    assert sent_outline.initial_mode == "low" and _index(sent_outline) == "outline" and not _cuts(sent_outline)
    assert rebound.initial_mode == "low" and _index(rebound) == "outline" and _cuts(rebound) == [("a", "summary")]
    assert rebound.messages_for("low")[1:] == messages[1:]
    # The form a head was sent in stays on its receipt: a narrower frame of the same work never shows it whole again.
    narrower = rebind(rebound, 25_000, "low")
    assert _cuts(narrower) == [("a", "summary"), ("c", "summary")] and "Head a. " * 5_000 not in _held(narrower)

    emergency = _plan(world, 12_000, core=captured)
    assert emergency.initial_mode == "nano"
    assert _cuts(emergency) == [("a", "reader"), ("c", "summary"), ("b", "reader")]
    assert "Authored telling of c." in _held(emergency) and "Authored telling of a." not in _blocks(emergency)[-1]["text"]
    assert "Head b. " * 5_000 not in _held(emergency)
    assert "knowledge_read(topic='b', scope='global', version=" in _held(emergency)


def test_a_smaller_window_never_shows_more_of_a_head_and_no_head_yields_beside_a_whole_index(world):
    """Down one descending sweep of windows, from whole books through the map frame to the emergency floor: the set
    of whole heads never grows, a whole-book frame shortens none, and wherever a head has yielded the index stands as
    its outline. (This frame sends no tool schemas, so Nano prices like the map frame it follows.)"""
    captured = _ordered_world(world)
    whole, forms = {"a", "b", "c"}, []
    for window in range(196_000, 8_000, -4_000):
        plan = _plan(world, window, core=captured)
        shown = {topic for topic in "abc" if f"Head {topic}. " * 5_000 in _held(plan)}
        assert shown <= whole, (window, sorted(shown), sorted(whole))
        assert shown == {"a", "b", "c"} - {topic for topic, _form in _cuts(plan)}
        assert not _cuts(plan) or (plan.initial_mode != "max" and _index(plan) == "outline"), window
        whole = shown
        form = (plan.initial_mode, _index(plan), tuple(_cuts(plan)))
        if plan.initial_mode != "max" and form not in forms:
            forms.append(form)
    # The map frames crossed every step in order: the whole index, its outline, each head by address, the emergency.
    assert forms == [("low", "whole", ()), ("low", "outline", ()), ("low", "outline", (("a", "summary"),)),
                     ("low", "outline", (("a", "summary"), ("c", "summary"))),
                     ("nano", "outline", (("a", "reader"), ("c", "summary"), ("b", "reader")))]


def test_no_head_yields_while_an_index_or_the_registry_can_still_take_its_outline(world):
    """One bound for the outlines and the heads. Down a sweep of windows in a project room the project's index, the
    global index and the registry each take their outline before any head steps; a head steps only once the outline
    walk has nothing left to take and the request still exceeds the window's bound."""
    env, _memory = world
    project = create_project(env.drive_root, "nova", name="Nova")
    task = {**TASK, "project_id": project["id"], "chat_id": project["chat_id"]}
    for topic in ("c", "b", "a"):
        _write(world, topic, f"Head {topic}. " * 5_000, claims=(f"room:{project['chat_id']}",),
               **({"summary": f"Authored telling of {topic}."} if topic != "b" else {}))
    _write(world, "ordinary", "An unclaimed note.", claims=(), summary="Whole index only. " * 4_000)
    _write(world, "local", "A project note.", claims=(), scope="project:nova", summary="Whole project index only. " * 4_000)
    registry = "".join(f"## Source {number}\n\n" + "Registry line. " * 600 + "\n\n" for number in range(8))
    (env.drive_root / "memory" / "registry.md").write_bytes(registry.encode("utf-8"))
    captured = _core(world, task)
    steps = set(memory_floor.WHOLE_OR_OUTLINE_STEPS)
    assert _untaken_outlines(_plan(world, 250_000, core=captured, task=task)) == steps  # all three can be taken here
    crossed = []
    for window in range(224_000, 8_000, -6_000):
        plan = _plan(world, window, core=captured, task=task)
        untaken, cuts = _untaken_outlines(plan), _cuts(plan)
        assert not cuts or not untaken, (window, cuts, sorted(untaken))
        if (state := (len(steps) - len(untaken), len(cuts))) not in crossed:
            crossed.append(state)
    # (outlines taken, heads stepped): each outline in its turn, then each head, then the emergency floor's third.
    assert crossed == [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (3, 3)]


def test_nano_prices_its_own_frame_and_carries_no_cut_from_the_map_frame_before_it(world):
    """Across the map frame and Nano a smaller window can buy room back: Nano unloads the tool schemas, so it prices
    its own request and shortens only what that request still lacks. Within one frame the order holds as ever."""
    captured = _ordered_world(world)
    tools = [{"type": "function", "function": {"name": f"fixture_tool_{number}", "description": "schema detail " * 2_000,
                                                "parameters": {"type": "object", "properties": {}}}} for number in range(4)]
    maps, nano, tighter = (_plan(world, window, core=captured, tool_schemas=tools) for window in (60_000, 42_000, 30_000))
    assert maps.initial_mode == "low" and _cuts(maps) == [("a", "summary"), ("c", "summary")]
    assert [nano.initial_mode, tighter.initial_mode] == ["nano", "nano"]
    assert _cuts(nano) == [("a", "summary")] and "Head c. " * 5_000 in _held(nano)  # it fits whole there: no stale cut
    assert _cuts(tighter) == [("a", "summary"), ("c", "summary")] and "Head b. " * 5_000 in _held(tighter)
    assert _index(nano) == _index(tighter) == "outline" and not (_untaken_outlines(nano) | _untaken_outlines(tighter))


def test_a_heads_label_is_its_title_or_an_h1_of_the_head_and_never_a_heading_below_it(world):
    _write(world, "below", "A head without a title.\n\n## History\n\nOlder facts.\n")
    _write(world, "inside", "A head with a smaller heading.\n\n### Detail\n\nStill the head.\n\n## Past\n\n# Deep\n")
    _write(world, "own", "# Nina Petrova\n\nWorks on Nova.\n\n## Contacts\n\nBelow.\n")
    _write(world, "titled", "A titled head.\n\n## History\n\nOlder facts.\n", title="The Nova room")
    _write(world, "blank", "A head whose title is blank.\n\n## History\n", title="  ")
    held = _held(_plan(world))
    labels = dict(reversed(pair) for pair in re.findall(r"^### (.+)\n\nGlobal note: memory/knowledge/(.+)\.md$", held, re.M))
    assert labels == {"below": "below", "inside": "inside", "blank": "blank", "own": "Nina Petrova", "titled": "The Nova room"}
    assert "### History" not in held and held.count("### Detail") == 1  # only the head's own line


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


def test_a_start_shaped_like_the_owners_keeps_all_six_heads_whole_while_memory_goes_below_a_telling(world, monkeypatch):
    """The owner's Nova rooms of 18-19 Nov: his Max with the books at chapter maps, my shortest telling tens of
    thousands of tokens past the start bound, memory's shorter forms inside it. Memory walks down its ladder and all
    six heads stand whole beside the whole index. A head yields only where the window cannot hold the request with
    memory at its shortest forms either, after the index outline, in address order. The fact the mind gets about what
    memory yielded to the window names the heads that stand whole."""
    heads = {"nova": "What is true of Nova now. ", "joi": "What Joi is and who runs it. ",
             "sasha": "How to work with Sasha. ", "nina": "How to work with Nina. ",
             "people-others": "The other people around Nova. ", "tools": "The instruments and the data. "}
    for topic, words in heads.items():
        _write(world, topic, words * 400, summary=f"Authored summary of {topic}.",
               **({"type": "person"} if topic in ("sasha", "nina") else {}))
    _write(world, "library", "An unclaimed note.", claims=(), summary="A long index summary. " * 4_000)
    memory = synthetic.snapshot(role="integrator", story=synthetic.pages(3, 80_000))
    memory = replace(memory, spec=replace(memory.spec, work_room_levels=2))  # a root's start keeps two eighths for work
    captured = replace(_core(world), memory_view_json=memory_view.snapshot_json(memory))
    telling = memory_floor.tokens_of(memory_view.story_chars(memory, memory_floor.telling_level(memory)))

    def facts(plan):  # the owner's Max, rendered at chapter maps
        assert plan.preferred_mode == "max" and plan.initial_mode == "low"
        assert "Book text. Book text." not in _blocks(plan)[0]["text"]
        return plan.low_projection.memory_facts

    def whole(plan):
        return [topic for topic, words in heads.items() if words * 400 in _held(plan)]

    roomy, shaped, tight = (_plan(world, window, core=captured) for window in (150_000, 100_000, 24_000))
    assert not facts(roomy)["floor"]["steps"] and not _cuts(roomy)
    request, story = shaped.projected_tokens_with_tools("low", []), facts(shaped)["blocks"]["story_tokens"]
    work_room = 100_000 // 8 * 2
    assert request - story + telling + RESERVE + work_room > 100_000 + 20_000  # my telling misses by tens of thousands
    assert request + RESERVE + work_room <= 100_000 and story < telling  # memory below a telling fits
    assert "F5l" in facts(shaped)["floor"]["steps"] and "Page 0. " + synthetic.prose(80_000) not in _blocks(shaped)[-2]["text"]
    assert not _cuts(shaped) and whole(shaped) == list(heads) and _index(shaped) == "whole"
    # Only where memory's shortest forms and the index outline still leave the window short do heads yield, by address.
    assert _cuts(tight) == [("joi", "summary"), ("nina", "summary"), ("nova", "summary")] and _index(tight) == "outline"
    assert whole(tight) == ["sasha", "people-others", "tools"] and "F5l" in facts(tight)["floor"]["steps"]

    held = sum(memory_floor.view_tokens(part.full_text) for part in captured.dynamic_sections
               if isinstance(part, held_notes.HeldNoteSection))
    cut = memory_floor.addressed_text(facts(shaped)["floor"]["steps"])
    line = context_view_facts.memory_facts(shaped, "low")
    assert f"; shown only by address: {cut}; held note heads: {held:,} tokens stand whole; marks in view 0" in line
    # Where the window took nothing from memory, or without held heads, the line is what it was.
    assert "held note heads" not in context_view_facts.memory_facts(roomy, "low")
    with monkeypatch.context() as patch:
        patch.setattr(held_notes, "turn_sections", lambda *_a, **_kw: [])
        unheld = replace(_core(world), memory_view_json=captured.memory_view_json)
    plain = _plan(world, 60_000, core=unheld)
    assert facts(plain)["floor"]["steps"] and not _held(plain)
    assert context_view_facts.memory_facts(plain, "low").endswith(
        f"; shown only by address: {memory_floor.addressed_text(facts(plain)['floor']['steps'])}; marks in view 0")

    # A work already running in whole books: where my telling no longer fits beside them the books go to their maps,
    # and that drop shortens nothing; memory then folds and the index outlines before the window takes a head.
    running = _plan(world, 400_000, core=captured)
    assert running.initial_mode == "max"
    messages = [*running.messages_for("max"), {"role": "user", "content": "Owner's words during work."}]
    def rebind(window):
        return running.reproject_for_route(window_tokens=window, known_window=True, ratio=1, output_reserve=RESERVE,
            tool_schemas=[], start_mode="max", current_messages=messages, admitted=True)

    mapped, folded, taken = rebind(130_000), rebind(30_000), rebind(20_500)
    assert mapped.initial_mode == "low" and not mapped.low_projection.memory_facts["floor"]["steps"]
    assert not _cuts(mapped) and whole(mapped) == list(heads) and _index(mapped) == "whole"
    assert folded.initial_mode == "low" and not _cuts(folded) and whole(folded) == list(heads)
    assert _index(folded) == "outline" and "F5l" in folded.low_projection.memory_facts["floor"]["steps"]
    assert _cuts(taken) == [("joi", "summary"), ("nina", "summary")] and taken.messages_for("low")[1:] == messages[1:]


def test_only_an_admitted_call_for_the_complete_bound_revision_can_shorten_a_head(world, monkeypatch):
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
        roomy = _plan(world, 130_000, mode="low", ctx=ctx)
        assert not _cuts(roomy) and head in _held(roomy)
        # The window can no longer hold the whole head: only a head with an admitted reader may step.
        plan = _rebind(monkeypatch, ctx, roomy, roomy.messages_for("low"), 18_000)
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
    # The tight window comes from the measured request, not from a literal beside the edge: half of the request that
    # holds the head whole (the head is most of it), plus the reply reserve. The summary fits there with room.
    whole = _plan(world, 60_000, mode="low", ctx=ctx)
    assert not _cuts(whole) and head in _held(whole)
    tight = _plan(world, whole.projected_tokens_with_tools("low", []) // 2 + RESERVE, mode="low", ctx=ctx)
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


def test_cold_replay_prices_the_restored_tail_and_resident_schemas(world, monkeypatch):
    from ouroboros.owner_wait import rebind_restored_route

    head = "Facts kept through a wait. " * 1_900
    _write(world, "resumed", head, summary="An authored summary.")
    ctx = ToolContext(repo_dir=world[0].repo_dir, drive_root=world[0].drive_root, task_id=TASK["id"])
    resident = [{"type": "function", "function": {"name": "read_loaded_source", "description": "source " * 120}}]
    opening = _plan(world, 48_000, mode="low", ctx=ctx)
    assert not _cuts(opening) and head in _held(opening)
    later = {"role": "user", "content": "These are the owner's later words. " * 180}
    messages = [*opening.messages_for("low"), later]
    tight = _rebind(monkeypatch, ctx, opening, messages, 12_000, tool_schemas=resident)
    assert _cuts(tight) == [("resumed", "summary")] and messages[-1] == later

    def saved():
        return json.loads(json.dumps(continuation_state(ctx, messages, {}, {}, 2, resident, set())))

    def replay(state, window):
        core = replace(opening.core, held_note_resume=held_notes.restored_capture(state))
        return _plan(world, window, mode="low", core=core, tool_schemas=[])

    def fresh(state, window):
        core = replace(opening.core, supplementary_messages_json=json.dumps(state["messages"][2:]))
        return _plan(world, window, mode="low", core=core, tool_schemas=state["tool_schemas"])

    small = saved()
    assert _cuts(replay(small, 12_000)) == _cuts(fresh(small, 12_000)) == _cuts(tight)
    carried = _rebind(monkeypatch, ctx, tight, messages, 48_000, tool_schemas=resident)
    before = saved()[held_notes.CAPTURE_KEY]["frame"][-1]
    messages[-1] = {**later, "content": later["content"] + "Another word. " * 200}
    grown = _rebind(monkeypatch, ctx, carried, messages, 48_000, tool_schemas=resident)
    state = saved()
    assert state[held_notes.CAPTURE_KEY]["frame"][-1] == before + 700
    same = replay(state, 48_000)
    assert _cuts(same) == _cuts(grown) == _cuts(tight)
    assert _held(same) == _held(grown) and same.messages_for("low")[1:] == state["messages"][1:]

    changed_tail = {**state, "messages": [*state["messages"], {"role": "user", "content": "Keep going."}]}
    changed_schemas = {**state, "tool_schemas": []}
    for changed in (changed_tail, changed_schemas):
        repriced, fitted = replay(changed, 48_000), fresh(changed, 48_000)
        assert not _cuts(repriced) and _held(repriced) == _held(fitted) and head in _held(repriced)

    monkeypatch.setattr("ouroboros.loop.get_context_mode", lambda: "low")
    monkeypatch.setattr("ouroboros.owner_wait._continuation_fit_route", lambda *_a: ({
        "model": "fixture", "use_local_model": False, "model_role": "main", "model_route": {},
        "credential_profile_id": None,
    }, ""))
    ctx._actor_loaded_tool_names = set()
    ctx.active_model, ctx.active_use_local = "fixture", False
    updated = [{"type": "function", "function": {"name": "read_loaded_source", "description": "Read source."}}]
    oversized = [{"type": "function", "function": {"name": "read_loaded_source", "description": "Frozen schema. " * 15_000}}]
    for frozen, schemas in ((resident, resident), (resident, updated), (oversized, updated)):
        restored = json.loads(json.dumps(state))
        restored["tool_schemas"] = list(frozen)
        ctx.active_context_mode = "low"
        ctx.context_fit_plan = replay(restored, 48_000)
        if frozen == oversized:
            assert ctx.context_fit_plan.initial_mode == "nano"
        registry = SimpleNamespace(_ctx=ctx, schemas=lambda: schemas, get_schema_by_name=lambda _name: schemas[0])
        rebuilt = list(restored["messages"])
        plan, mode = rebind_restored_route(registry, restored, rebuilt)
        assert mode == "low" and rebuilt[-2 if schemas == updated else -1] == messages[-1]
        assert restored["tool_schemas"] == schemas
        if schemas == updated:
            fitted = fresh({**restored, "messages": rebuilt}, 48_000)
            assert not _cuts(plan) and _held(plan) == _held(fitted) and head in _held(plan)
        else:
            assert _cuts(plan) == _cuts(grown) and _held(plan) == _held(grown)


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
