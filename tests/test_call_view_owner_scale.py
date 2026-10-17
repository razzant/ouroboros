"""The Gemini call view on a data root at the owner's measured sizes (``tests/_call_view_owner_scale.py``).

On the owner's data the call asked for the ordinary Nano projection: 183,827 o200k tokens,
above Gemini Live's 131,072-token input limit, and v1 fitted it against 65,536 (the output
maximum subtracted). The call view keeps tier zero whole, shows every host section with an
operative view in that form from the first open, and is fitted to the realtime route itself.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from ouroboros import context
from ouroboros.live_call_route import call_context_plan
from tests import _call_view_owner_scale as scale

REPO = Path(__file__).resolve().parents[1]
WINDOW, REPLY = 131_072, 8_192
FROZEN = "2026-10-16T21:00:00+00:00"


@pytest.fixture
def owner_scale(tmp_path, monkeypatch):
    from ouroboros import context_health
    from ouroboros.agent import Env
    from ouroboros.memory import Memory
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    root = tmp_path / "data"
    scale.build(root)
    scale.obligations(monkeypatch)
    scale.subagents(monkeypatch)
    monkeypatch.setattr(context_health, "_stray_server_note", lambda _env: "")  # the host's own processes
    monkeypatch.setattr(context, "utc_now_iso", lambda: FROZEN)
    env, memory = Env(repo_dir=REPO, drive_root=root), Memory(drive_root=root, repo_dir=REPO)
    registry = ToolRegistry(repo_dir=REPO, drive_root=root)
    ctx = ToolContext(repo_dir=REPO, drive_root=root, task_id=scale.CALL_TASK_ID)
    registry.set_context(ctx)
    task = scale.call_task()
    plan = call_context_plan(task, registry, root)
    messages, _cap = context.build_llm_messages(env, memory, task, ctx=ctx, tool_schemas=plan.tool_schemas,
                                                **plan.fit_kwargs)
    return env, memory, task, plan, ctx.context_fit_plan, messages


def _system_text(messages):
    return "\n".join(block["text"] for block in messages[0]["content"])


def test_the_gemini_call_view_fits_its_route_with_room_for_the_conversation(owner_scale):
    _env, _memory, _task, call, plan, messages = owner_scale
    assert (plan.window_tokens, plan.output_reserve_tokens, plan.initial_mode) == (WINDOW, REPLY, "nano")
    assert plan.call_view and plan.model == "gemini-3.8-live"
    assert plan.projection("nano").memory_facts["floor"]["window_tokens"] == WINDOW  # never 65,536
    request = plan.projected_tokens_with_tools("nano", call.tool_schemas)
    assert plan.projection("nano").calibration_ratio == 1.25  # no witness yet: the route's documented cold density
    # In the route's own units, the view, the user turn and the first tool view leave the reply
    # allowance and at least one low-water level of the window (an eighth) for the conversation itself.
    assert request + REPLY <= WINDOW - math.ceil(WINDOW / 8), request
    assert {"read_file", "start_task"} <= {s["function"]["name"] for s in call.tool_schemas}


def test_tier_zero_stays_byte_whole_and_every_service_block_is_operative(owner_scale):
    env, memory, _task, _call, plan, messages = owner_scale
    text = _system_text(messages)
    root = env.drive_root
    assert (REPO / "prompts" / "SYSTEM.md").read_text(encoding="utf-8") in text
    assert (REPO / "BIBLE.md").read_text(encoding="utf-8") in text
    assert (root / "memory" / "identity.md").read_text(encoding="utf-8") in text
    assert (root / "memory" / "scratchpad.md").read_text(encoding="utf-8") in text
    index = next(part for part in plan.core.dynamic_head_sections if part.section_id == "knowledge_index")
    assert index.full_text in text and "Where we stand: " in index.full_text  # the overview's own summary
    addressed = {row["section_id"]: row for row in plan.projection("nano").memory_facts["host_sections_addressed"]}
    assert set(addressed) >= {"health_invariants", "shared_understanding", "pattern_register", "world", "subagents",
                              "execution_reflections", "architecture_map", "development_map", "live_rooms"}
    assert not {"system", "bible", "identity", "scratchpad", "knowledge_index"} & set(addressed)
    sections = {part.section_id: part for part in (*plan.core.dynamic_sections, *plan.core.dynamic_head_sections,
                                                   *plan.core.semi_stable_sections)}
    for section_id, row in addressed.items():
        raw = (root / row["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == row["sha256"] and row["reader"] in text
        if section_id in sections:
            assert raw.decode("utf-8") == sections[section_id].full_text  # the pointer reads back the exact bytes
            assert sections[section_id].operative_view in text


def test_two_captures_give_a_byte_identical_call_view(owner_scale):
    env, memory, task, call, plan, messages = owner_scale

    def capture():
        core = context._capture_context_core(env, memory, task, None, None)
        again = context._build_context_fit_plan(env, core, task, preferred_mode=plan.preferred_mode,
                                                route_resolver=context._context_fit_route,
                                                tool_schemas=call.tool_schemas, **call.fit_kwargs)
        return again.projection(again.initial_mode).system_content_json

    first, second = capture(), capture()
    assert first == second
    assert json.loads(first)[-1]["text"].count("## Health Invariants") == 1
