"""Where the memory view sits in a request, and what each block's bytes depend on.

A built request is one system message of three text blocks and the task: governance and the
books (A, marked), identity with my story (B, marked), then knowledge, my rooms and the runtime
facts (C, unmarked). B depends only on the chronicle, identity, WORLD, the deep review and the
catalog: the same bytes for Main, a Project's root and a wake. Knowledge edits change only C;
a new page changes B and never A. Anthropic keeps four breakpoints (tools, A, B, the task seal);
Codex and direct OpenAI keep A and B as two system items and C as one notice (their cache is
read only inside the leading system group, by prefix: measured 2026-10-03); the view fact of
the first send reaches the cap info, the task context, one event and the trace.
Fixture: ``tests._memory_inventory_shared`` (three rooms besides Main, legacy memory).
"""
from __future__ import annotations

import copy
import json
import queue

import pytest

from tests._memory_view_context import blocks, section, world

MAIN = {"id": "tmain", "chat_id": 1}
WAKE = {"id": "wake", "chat_id": 1, "_is_direct_chat": True,
        "metadata": {"initiator": "consciousness", "usage_category": "consciousness"}}


def _messages(env, memory, task, **kwargs):
    from ouroboros.context import build_llm_messages

    return build_llm_messages(env=env, memory=memory, task={"type": "task", "text": "hi", **task}, **kwargs)


def test_the_request_is_three_blocks_with_my_story_in_b_and_knowledge_in_c(tmp_path):
    env, memory, _rooms = world(tmp_path)
    messages, _cap = _messages(env, memory, MAIN)
    content = messages[0]["content"]
    assert len(content) == 3 and messages[1]["role"] == "user"
    assert [("cache_control" in block) for block in content] == [True, True, False]
    a, b, c = (block["text"] for block in content)
    assert "## My story" in b and "## My story" not in a + c
    for heading in ("## Shared understanding", "## Knowledge base", "## Live rooms", "## This room (Main)"):
        assert heading in c and heading not in a + b, heading
    for gone in ("## Dialogue History", "## Recent chat", "## Project owner origins", "## Legacy Dialogue Summary"):
        assert gone not in a + b + c, gone


def test_b_is_byte_identical_for_main_another_room_and_a_wake(tmp_path):
    env, memory, rooms = world(tmp_path)
    views = {name: blocks(env, memory, task) for name, task in (
        ("main", MAIN), ("alpha", {"id": "bound", "chat_id": 1}), ("beta", {"id": "tb", "chat_id": rooms["beta"]}),
        ("wake", WAKE))}
    assert len({view[1] for view in views.values()}) == 1  # B: one story, whatever the room or role
    assert len({view[0] for view in views.values()}) == 1  # A: governance and books
    assert len({view[2] for view in views.values()}) == 4  # C: each one's own rooms


def test_a_knowledge_edit_changes_only_c_and_a_new_page_changes_b_but_never_a(tmp_path):
    from tests.test_memory_view_story import _page

    env, memory, _rooms = world(tmp_path)
    a0, b0, c0, _cap = blocks(env, memory, MAIN)
    overview = memory.drive_root / "memory" / "knowledge" / "overview.md"
    overview.write_text("# Overview\n\nWhat is true now: the view is wired.\n", encoding="utf-8")
    a1, b1, c1, _cap = blocks(env, memory, MAIN)
    assert (a1, b1) == (a0, b0) and c1 != c0 and "the view is wired" in c1

    page = _page(memory.drive_root, "1", 11, 12, text="My page about the next request.")
    a2, b2, c2, _cap = blocks(env, memory, MAIN)
    assert a2 == a0 and b2 != b1 and "My page about the next request." in b2
    assert b2.startswith(b1.split("\n## My story", 1)[0])  # identity and catalog keep their bytes
    assert f"page {page}" in b2 and "My page about the next request." not in c2

    identity = memory.drive_root / "memory" / "identity.md"
    identity.write_text("identity, revised", encoding="utf-8")
    a3, b3, _c3, _cap = blocks(env, memory, MAIN)
    assert a3 == a0 and b3 != b2 and "identity, revised" in b3


def _anthropic_payload(messages, tools):
    """The direct-Anthropic shape the finalizer reads: tools, then system blocks, then messages."""
    from ouroboros.context_fit import seal_task_transcript

    transcript = copy.deepcopy(messages)
    seal_task_transcript(transcript)
    return {"tools": copy.deepcopy(tools), "system": transcript[0]["content"],
            "messages": [{"role": "user", "content": transcript[1]["content"]}]}


@pytest.mark.parametrize("third_marked", [False, True])
def test_anthropic_keeps_four_breakpoints_with_the_task_seal(tmp_path, third_marked):
    """Three explicit markers (A, B, the task seal) and the automatic one on the schemas: exactly
    the Anthropic limit. A marker put back on block C would make five, and the seal would be dropped."""
    from ouroboros.llm import LLMClient

    env, memory, _rooms = world(tmp_path)
    messages, _cap = _messages(env, memory, MAIN)
    tools = [{"name": "memory_read", "description": "read", "input_schema": {"type": "object", "properties": {}}}]
    payload = _anthropic_payload(messages, tools)
    if third_marked:
        payload["system"][2]["cache_control"] = {"type": "ephemeral"}
    client = LLMClient(api_key="unused")
    client._normalize_payload_cache_ttl({"provider": "anthropic"}, payload)
    note = client._cache_breakpoint_tls.pending
    kept = LLMClient._payload_cache_breakpoints(payload)
    seal = [block for block in payload["messages"][0]["content"] if isinstance(block, dict)]
    assert len(kept) == 4 and "cache_control" in payload["tools"][-1]
    if third_marked:
        assert note == {"declared": 5, "kept": 4, "dropped": 1}
        assert not any("cache_control" in block for block in seal)  # the moving seal lost: the regression
    else:
        assert note is None
        assert any("cache_control" in block for block in seal)


def test_local_compaction_keeps_my_story_and_this_room_and_never_splits_a_reply(tmp_path):
    from ouroboros.llm_local import _compact_local_text

    env, memory, _rooms = world(tmp_path)
    _a, b, c, _cap = blocks(env, memory, MAIN)
    kept_b = _compact_local_text(b, "semi_stable")
    assert section(b, "## My story") in kept_b
    kept_c = _compact_local_text(c, "dynamic")
    assert section(c, "## This room (Main)") in kept_c  # the label in parentheses still matches "This room"
    assert "  ## with a heading" in kept_c and "## with a heading\n\n[Compacted" not in kept_c
    assert "[Compacted for local-model context" in section(kept_c, "## Live rooms")  # not a kept title


def test_the_view_fact_reaches_the_cap_info_the_task_context_and_one_event(tmp_path):
    from ouroboros.memory_inventory import VIEW_TRACE_KEY
    from ouroboros.tools.tool_context import ToolContext

    env, memory, _rooms = world(tmp_path)
    ctx = ToolContext(repo_dir=env.repo_dir, drive_root=memory.drive_root, task_id="tmain", task_metadata={},
                      current_chat_id=1)
    _messages_out, cap = _messages(env, memory, MAIN, ctx=ctx)
    facts = cap["memory_view"]
    assert facts["role"] == "integrator" and facts["room_id"] == "1"
    assert facts["store_status"] == {"state": "active"} and facts["legacy_frontier_status"] == "exact"
    assert facts["spec"]["live_rooms"] == "lines" and facts["blocks"]["live_rooms"] == 4
    assert facts["blocks"]["story_tokens"] > 0 and facts["blocks"]["room_tokens"] > 0
    assert facts["floor"]["mode"] == "max" and facts["floor"]["steps"] == {}
    assert ctx.memory_view_facts == {key: facts[key] for key in ("role", "room_id", "floor", "story_status")}
    assert VIEW_TRACE_KEY == "memory_view"
    events = [json.loads(line) for line in (memory.drive_root / "logs" / "events.jsonl").read_text(
        encoding="utf-8").splitlines() if '"context_memory_view"' in line]
    assert len(events) == 1 and events[0]["task_id"] == "tmain" and events[0]["blocks"] == facts["blocks"]


def test_a_declared_input_child_carries_no_view_fact(tmp_path):
    env, memory, _rooms = world(tmp_path)
    child = {"id": "kid", "chat_id": 1, "delegation_role": "subagent", "parent_task_id": "bound",
             "root_task_id": "bound", "task_contract": {"input_sources": "declared"},
             "configured_subagent": {"route": {"kind": "api_model"}}}
    _messages_out, cap = _messages(env, memory, child)
    assert "memory_view" not in cap
    assert not (memory.drive_root / "memory" / "chronicle").exists()  # no capture, no activation


@pytest.mark.parametrize("route,story", [("api_model", True), ("agent_session", False)])
def test_a_helpers_role_line_names_exactly_what_its_request_holds(tmp_path, route, story):
    """The child (with my story) and the nanny (without): each loaded item is a section of the
    request, each unloaded one is absent from it."""
    env, memory, rooms = world(tmp_path)
    helper = {"id": "kid", "chat_id": 1, "delegation_role": "subagent", "parent_task_id": "bound",
              "root_task_id": "bound", "configured_subagent": {"id": "h", "route": {"kind": route}}}
    _a, b, c, _cap = blocks(env, memory, helper)
    line = section(c, "## Working sources")
    loaded, missing = line.split("Loaded above: ", 1)[1].split(" Not loaded: ", 1)
    alpha = f"Project Alpha [chat_id={rooms['alpha']}]"
    assert f"the page of your parent's room {alpha}" in loaded and f"## This room ({alpha})" in c
    assert ("the top level of your story" in loaded) == story == ("## My story" in b)
    assert ("the top level of your story" in missing) == (not story)
    assert "knowledge (overview, index, patterns)" in missing and "## Shared understanding" not in c
    assert "raw conversations" in missing and "### Open conversation" not in c and "## Live rooms" not in c
    assert "the words of my human that caused this work" not in loaded  # no root record: no block drawn


def test_the_tool_schemas_count_in_the_floor_of_every_mode(tmp_path):
    """Schemas reach the plan: the same view leaves less room for memory with them, and Nano
    counts only its own selection of them."""
    from types import SimpleNamespace

    from ouroboros import context

    env, memory, _rooms = world(tmp_path)
    task = {"type": "task", "text": "hi", **MAIN}
    core = context._capture_context_core(env, memory, task, None, None)
    schemas = [{"type": "function", "function": {"name": f"tool_{i}", "description": "d" * 4_000,
                                                 "parameters": {"type": "object", "properties": {}}}} for i in range(30)]

    def route(*_a, **_kw):
        return ({"model": "m", "provider": "p"},
                SimpleNamespace(route_fp="r", status="asserted", stale=False, window_tokens=1_000_000))

    def plan(tools):
        return context._build_context_fit_plan(env, core, task, preferred_mode="max", route_resolver=route,
                                               tool_schemas=tools)

    bare, loaded = plan(None), plan(schemas)
    for mode in ("max", "low"):
        assert (bare.projection(mode).memory_facts["floor"]["allowance_tokens"]
                - loaded.projection(mode).memory_facts["floor"]["allowance_tokens"]) >= 30_000
    assert (bare.projection("nano").memory_facts["floor"]["physical_allowance_tokens"]
            == loaded.projection("nano").memory_facts["floor"]["physical_allowance_tokens"])  # none is a Nano schema


@pytest.mark.parametrize("actor,owner", [("child", "max"), ("child", "nano"), ("root", "max")])
def test_a_nano_floor_names_only_the_path_its_own_request_sends(tmp_path, monkeypatch, actor, owner):
    """The real path: the schemas come from the task's registry under the owner's mode, as the agent
    passes them (``initial_tool_schemas``), and the plan's Nano projection is the request. A Nano
    the window chose for the owner's Max sends a child list_available_tools alone: its floor names
    that and the parent, never enable_tools or chronicle_write. The owner's Nano sends the child enable_tools, and a
    root always: their floor says memory_read is reachable through it."""
    from types import SimpleNamespace

    from ouroboros import context
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.memory_floor import minimal_view_tokens
    from ouroboros.memory_view import snapshot_from_json
    from ouroboros.tool_policy import initial_tool_schemas, select_tool_schemas
    from ouroboros.tools.registry import ToolRegistry

    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", owner)
    env, memory, _rooms = world(tmp_path)
    task = {"type": "task", "text": "hi", **(MAIN if actor == "root" else {
        "id": "kid", "chat_id": 1, "delegation_role": "subagent", "parent_task_id": "bound", "root_task_id": "bound",
        "configured_subagent": {"id": "h", "route": {"kind": "api_model"}}})}
    (tmp_path / "registry").mkdir()
    registry = ToolRegistry(repo_dir=tmp_path / "registry", drive_root=tmp_path / "registry")
    if actor == "child":
        registry._ctx.task_constraint = TaskConstraint(mode="local_readonly_subagent")
    schemas = initial_tool_schemas(registry, context_mode=owner)
    core = context._capture_context_core(env, memory, task, None, None)

    def plan(window):
        evidence = SimpleNamespace(route_fp="r", status="asserted", stale=False, window_tokens=window)
        return context._build_context_fit_plan(env, core, task, preferred_mode=owner, tool_schemas=schemas,
                                               route_resolver=lambda *_a, **_kw: ({"model": "m", "provider": "p"}, evidence))

    roomy = plan(10_000_000)  # Nano's fixed part and reserve, read off a roomy plan; then a window only Nano fits
    window = (10_000_000 - roomy.projection("nano").memory_facts["floor"]["physical_allowance_tokens"]
              + minimal_view_tokens(snapshot_from_json(core.memory_view_json), window_tokens=10_000_000) + 50)
    built = plan(window)
    assert built.initial_mode == "nano"
    floor = section(built.messages_for("nano")[0]["content"][2]["text"], "### Physical floor")
    sent = select_tool_schemas(schemas, context_mode="nano").chosen
    assert "memory_read" not in sent and ("enable_tools" in floor) == ("enable_tools" in sent), (sent, floor)
    # Sealing is offered only to a request that can call chronicle_write (sent, or loaded by enable_tools).
    assert ("chronicle_write kind=page" in floor) == bool({"chronicle_write", "enable_tools"} & set(sent)), floor
    if (actor, owner) == ("child", "max"):
        assert sent == ("list_available_tools",)
        assert floor.rstrip().endswith("list_available_tools shows what this task can call, and my parent task can "
                                       "read any address I name to it)")
    else:
        assert "enable_tools" in sent and floor.rstrip().endswith("(memory_read is reachable through enable_tools)")


def test_the_task_trace_keeps_the_view_fact_of_its_request(tmp_path, monkeypatch):
    import ouroboros.loop as loop
    from ouroboros.memory_inventory import VIEW_TRACE_KEY
    from ouroboros.tools.registry import ToolRegistry

    class FakeLLM:
        def default_model(self):
            return "test-model"

    monkeypatch.setattr(loop, "call_llm_with_retry",
                        lambda *_a, **_kw: ({"role": "assistant", "content": "done"}, 0.0))
    monkeypatch.setattr(loop, "_run_task_acceptance_review_once", lambda **_kw: False)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    facts = {"role": "integrator", "room_id": "1", "floor": {"steps": {}}, "story_status": {"folded": 0, "total": 2}}
    for given in (facts, None):
        root = tmp_path / ("with" if given else "without")
        root.mkdir()
        registry = ToolRegistry(repo_dir=root, drive_root=root)
        if given:
            registry._ctx.memory_view_facts = given
        _result, _usage, trace = loop.run_llm_loop(
            messages=[{"role": "user", "content": "hi"}], tools=registry, llm=FakeLLM(), drive_logs=root,
            emit_progress=lambda *_a, **_kw: None, incoming_messages=queue.Queue(), task_id="t1", drive_root=root)
        assert trace.get(VIEW_TRACE_KEY) == given


def _codex_wire(messages):
    from ouroboros.llm_claudexor import _request

    target = {"source": "codex", "resolved_model": "gpt-6-sol", "usage_model": "claudexor::codex=gpt-6-sol"}
    payload = _request(target, copy.deepcopy(messages), None, {"reasoning_effort": "high", "model_role": "main"})
    return payload["messages"], target


def test_codex_keeps_identity_and_my_story_in_the_leading_system_group(tmp_path):
    """L2: system(A), system(B), one notice(C), the task. A knowledge edit changes only the
    notice; a new page changes B's item and never A's; round 2 extends round 1 byte for byte."""
    from ouroboros.llm_messages import HOST_CONTEXT_NOTICE_BEFORE_TASK
    from tests.test_memory_view_story import _page

    env, memory, _rooms = world(tmp_path)
    messages, _cap = _messages(env, memory, MAIN)
    wire, target = _codex_wire(messages)
    assert [message["role"] for message in wire] == ["system", "system", "user", "user"]
    assert target["wire_layout"] == {"system_prefix_split": True, "moved_blocks": 1}
    a, b, c = (block["text"] for block in messages[0]["content"])
    assert wire[0]["content"] == [{"type": "text", "text": a}] and wire[1]["content"] == [{"type": "text", "text": b}]
    assert "## My story" in wire[1]["content"][0]["text"]
    notice = wire[2]["content"]
    assert HOST_CONTEXT_NOTICE_BEFORE_TASK in notice and notice.endswith(c)
    assert "## My story" not in notice and "## Shared understanding" in notice

    round2 = messages + [{"role": "assistant", "content": "working"}, {"role": "user", "content": "go on"}]
    wire2, _target = _codex_wire(round2)
    assert wire2[:len(wire)] == wire

    (memory.drive_root / "memory" / "knowledge" / "overview.md").write_text(
        "# Overview\n\nWhat is true now: the cache keeps my story.\n", encoding="utf-8")
    edited, _target = _codex_wire(_messages(env, memory, MAIN)[0])
    assert edited[:2] == wire[:2] and edited[2] != wire[2]

    _page(memory.drive_root, "1", 11, 12, text="A page appended to my story.")
    paged, _target = _codex_wire(_messages(env, memory, MAIN)[0])
    assert paged[0] == wire[0] and paged[1] != wire[1]
    assert "A page appended to my story." in paged[1]["content"][0]["text"]


def test_direct_openai_and_openrouter_keep_the_two_system_items_and_one_notice(tmp_path, monkeypatch):
    from ouroboros.llm import LLMClient
    from ouroboros.llm_messages import HOST_CONTEXT_NOTICE_BEFORE_TASK

    monkeypatch.setattr(LLMClient, "_SUPPORTED_PARAMS_FETCHED", True, raising=False)
    monkeypatch.setattr("ouroboros.pricing._fetch_live_rows", lambda *_a, **_kw: {})
    env, memory, _rooms = world(tmp_path)
    messages, _cap = _messages(env, memory, MAIN)
    a, b, c = (block["text"] for block in messages[0]["content"])
    client = LLMClient(api_key="unused")
    for model, key in (("openai::gpt-6-sol", "OPENAI_API_KEY"), ("openai/gpt-6-sol", "OPENROUTER_API_KEY")):
        monkeypatch.setenv(key, "unused")
        target = client._resolve_remote_target(model)
        wire = client._build_remote_kwargs(target, copy.deepcopy(messages), "high", 512, "auto", None, None,
                                           skip_capability_fetch=True)["messages"]
        assert [message["role"] for message in wire] == ["system", "system", "user", "user"], model
        assert [block["text"] for block in wire[0]["content"]] == [a] and [block["text"] for block in wire[1]["content"]] == [b]
        assert HOST_CONTEXT_NOTICE_BEFORE_TASK in wire[2]["content"] and wire[2]["content"].endswith(c)
        assert target["wire_layout"] == {"system_prefix_split": True, "moved_blocks": 1}
