"""Where the memory view sits in a request, and what each block's bytes depend on.

A built request holds common governance (A', marked), the optional handbook (D, marked),
identity with my story (B, marked), then knowledge, my rooms and the runtime
facts (C, unmarked). B depends only on the chronicle, identity, WORLD, the deep review and the
catalog: the same bytes for Main, a Project's root and a wake. Knowledge edits change only C;
a new page changes B and never A' or D. Anthropic keeps the message seal and gives schemas
only a free slot. Codex and direct OpenAI keep all stable items and C as one notice (their cache is
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


def test_the_request_keeps_my_story_before_the_changing_tail_and_the_book_separate(tmp_path):
    env, memory, _rooms = world(tmp_path)
    messages, _cap = _messages(env, memory, MAIN)
    content = messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert all("cache_control" in block for block in content[:-1]) and "cache_control" not in content[-1]
    a, d, b, c = (block["text"] for block in content)
    assert d.startswith("## DEVELOPMENT.md\n") and "## DEVELOPMENT.md\n" not in a
    assert "## My story" in b and "## My story" not in a + d + c
    for heading in ("## Shared understanding", "## Knowledge base", "## Live rooms", "## This room (Main)"):
        assert heading in c and heading not in a + b, heading
    for gone in ("## Dialogue History", "## Recent chat", "## Project owner origins", "## Legacy Dialogue Summary"):
        assert gone not in a + b + c, gone


def test_b_is_byte_identical_for_main_another_room_and_a_wake(tmp_path):
    """One story for every integrating focus; a wake's room is Main, byte for byte the Main turn's, and its
    live rooms carry the words of people there (the Main turn's carry one line per room)."""
    env, memory, rooms = world(tmp_path)
    views = {name: blocks(env, memory, task) for name, task in (
        ("main", MAIN), ("alpha", {"id": "bound", "chat_id": 1}), ("beta", {"id": "tb", "chat_id": rooms["beta"]}),
        ("wake", WAKE))}
    assert len({view[1] for view in views.values()}) == 1  # B: one story, whatever the room or role
    assert len({view[0] for view in views.values()}) == 1  # A': common governance
    assert len({view[2] for view in views.values()}) == 4  # C: each one's own rooms
    main_c, wake_c = views["main"][2], views["wake"][2]
    assert section(wake_c, "## This room (Main)") == section(main_c, "## This room (Main)")
    assert "## This room (Main)" not in views["alpha"][2]
    for words in ("alpha again", "beta again"):  # people's words of the other live rooms, verbatim, to the wake alone
        assert words in section(wake_c, "## Live rooms") and words not in main_c, words


def test_full_roles_share_the_first_block_and_only_eligible_tasks_get_the_handbook(tmp_path):
    """A real workspace binding must not invalidate the common governance prefix."""
    env, memory, _rooms = world(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    tasks = {
        "main": ({**MAIN, "_is_direct_chat": True}, True),
        "wake": (WAKE, True),
        "system-root": ({"id": "root", "workspace": "none"}, True),
        "workspace-root": ({"id": "bound", "workspace_root": str(project)}, False),
        **{source: ({"id": source, "metadata": {"source": source}}, False)
           for source in ("api_task", "cli", "scheduled_task")},
        "presence": ({"id": "presence", "type": "presence", "source": "presence", "_presence_turn": True,
                      "context_requires_development": False, "chat_id": 777}, False),
    }
    contents = {name: _messages(env, memory, task)[0][0]["content"] for name, (task, _book) in tasks.items()}
    assert all(content[0] == contents["main"][0] for content in contents.values())
    for name, content in contents.items():
        book = tasks[name][1]
        assert "docs/DEVELOPMENT.md" in content[0]["text"], name
        handbook = [block for block in content if block["text"].startswith("## DEVELOPMENT.md\n")]
        assert len(handbook) == int(book), name
        if book:
            assert content[1] == handbook[0] and "cache_control" in handbook[0], name


@pytest.mark.parametrize("mode,child", [("low", False), ("nano", False), ("max", True), ("low", True), ("nano", True)])
def test_compact_modes_and_children_keep_both_book_maps_without_a_separate_handbook(tmp_path, monkeypatch, mode, child):
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", mode)
    env, memory, _rooms = world(tmp_path)
    task = {**MAIN, **({"delegation_role": "subagent", "parent_task_id": "bound",
                       "configured_subagent": {"id": "h", "route": {"kind": "api_model"}}} if child else {})}
    prefixes = []
    for flag in (False, True):
        messages, _cap = _messages(env, memory, {**task, "context_requires_self_body_docs": flag})
        content = messages[0]["content"]
        prefixes.append(content[0])
        assert "## ARCHITECTURE.md (navigation map)" in content[0]["text"]
        assert "## DEVELOPMENT.md (navigation map)" in content[0]["text"]
        assert not any(block["text"].startswith("## DEVELOPMENT.md\n") for block in content)
    assert prefixes[0] == prefixes[1]


@pytest.mark.parametrize("book", [False, True])
def test_route_reprojection_keeps_the_captured_handbook_and_counts_it_in_the_floor(tmp_path, monkeypatch, book):
    """D is reconstructed from the capture, never from a block index or changed files."""
    from dataclasses import replace
    from types import SimpleNamespace
    from ouroboros import context, context_fit

    env, memory, _rooms = world(tmp_path)
    entry = env.repo_path("docs/DEVELOPMENT.md")
    entry.write_text("# Development\n\nEngineering handbook.\n\n## Chapters\n\n- [Rules](development/rules.md)\n", encoding="utf-8")
    chapter = env.repo_path("docs/development/rules.md")
    chapter.parent.mkdir()
    chapter.write_text("# Rules\n\nChange rules.\n\n## Detail\n\n" + "handbook body " * 2000, encoding="utf-8")
    task = {"type": "task", "text": "hi", **MAIN, "workspace_root": "" if book else str(tmp_path / "project")}
    core = context._capture_context_core(env, memory, task, None, None)
    monkeypatch.setattr(context_fit, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)
    route = lambda *_a, **_kw: ({"model": "m", "provider": "p"}, SimpleNamespace(  # noqa: E731
        route_fp="r", status="asserted", stale=False, window_tokens=1_000_000))

    def build(capture):
        return context._build_context_fit_plan(env, capture, task, preferred_mode="max", route_resolver=route)

    plan = build(core)
    without = build(replace(core, docs_need_development=False))
    for mode in ("max", "low", "nano"):
        facts = plan.projection(mode).memory_facts["floor"]
        delta = without.projection(mode).memory_facts["floor"]["physical_allowance_tokens"] - facts["physical_allowance_tokens"]
        assert (delta > 5000) if book and mode == "max" else delta == 0
    chapter.write_text("# Rules\n\nChanged after capture.\n", encoding="utf-8")
    rebuilt = plan.reproject_for_route(window_tokens=2_000_000, known_window=True, ratio=1.0,
                                      output_reserve=plan.output_reserve_tokens, tool_schemas=None)
    for mode in ("max", "low", "nano"):
        before, after = plan.messages_for(mode)[0], rebuilt.messages_for(mode)[0]
        assert after["content"] == before["content"]
        texts = [block["text"] for block in after["content"]]
        assert sum(text.startswith("## DEVELOPMENT.md\n") for text in texts) == int(book and mode == "max")
        if book and mode == "max":
            assert texts[1] == "## DEVELOPMENT.md\n\n" + core.development_md
        assert all("Changed after capture" not in text for text in texts)
        assert (rebuilt.projection(mode).memory_facts["floor"]["physical_allowance_tokens"]
                - plan.projection(mode).memory_facts["floor"]["physical_allowance_tokens"]) == 1_000_000


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


@pytest.mark.parametrize("book", [False, True])
@pytest.mark.parametrize("long", [False, True])
@pytest.mark.parametrize("provider", ["anthropic", "openrouter"])
@pytest.mark.parametrize("mode", ["max", "low", "nano"])
def test_anthropic_preserves_the_message_seal_and_gives_schemas_only_a_free_slot(tmp_path, monkeypatch, book, long, provider, mode):
    """Exercise the actual wire builders, including a nested direct-Anthropic tool result."""
    from ouroboros.context_fit import seal_task_transcript
    from ouroboros.llm import LLMClient
    from tests.test_review_prompt_caching import _deep_markers, _openrouter_target, _send_direct_anthropic, _tools

    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", mode)
    env, memory, _rooms = world(tmp_path)
    messages, _cap = _messages(env, memory, {**MAIN, "workspace_root": str(tmp_path / "project") if not book else ""})
    seal_task_transcript(messages)
    if long:
        for i in range(6):
            messages.extend([
                {"role": "assistant", "content": "", "tool_calls": [{"id": f"call_{i}", "type": "function",
                    "function": {"name": "zeta_tool", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": f"call_{i}", "content": f"result {i}"},
            ])
        seal_task_transcript(messages, min_prefix_tokens=0)
    canonical, tools = copy.deepcopy(messages), _tools()
    client = LLMClient(api_key="unused")
    if provider == "anthropic":
        payload, usage = _send_direct_anthropic(monkeypatch, messages, tools)
        assert usage.get("prompt_cache_breakpoints_reduced") is None
    else:
        target = _openrouter_target("anthropic/claude-fable-5")
        payload = client._build_remote_kwargs(target, messages, "high", 512, "auto", None, tools, skip_capability_fetch=True)
        client._normalize_payload_cache_ttl(target, payload)
        assert client._cache_breakpoint_tls.pending is None
    kept = LLMClient._payload_cache_breakpoints(payload)
    assert len(kept) == len(_deep_markers(payload)) == 4
    has_book = book and mode == "max"
    assert ("cache_control" in payload["tools"][-1]) == (not has_book)
    assert kept[-1]["text"] == ("result 0" if long else messages[1]["content"][-1]["text"])
    assert any(block.get("text", "").startswith("## DEVELOPMENT.md\n") for block in kept) == has_book
    assert all(block["cache_control"] == {"type": "ephemeral", "ttl": "1h"} for block in kept)
    assert messages == canonical and tools == _tools()


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


@pytest.mark.parametrize("book", [False, True])
def test_local_compaction_keeps_identity_and_my_story_whichever_block_holds_them(tmp_path, monkeypatch, book):
    """The handbook moves B off the second block; a local model still gets my identity, story and room."""
    from ouroboros import llm_local
    from ouroboros.llm import LLMClient

    env, memory, _rooms = world(tmp_path)
    messages, _cap = _messages(env, memory, {**MAIN, "workspace_root": "" if book else str(tmp_path / "project")})
    a, *middle, c = (block["text"] for block in messages[0]["content"])
    b = middle[-1]
    assert [text.startswith("## DEVELOPMENT.md\n") for text in middle] == ([True, False] if book else [False])
    sizes = iter([10**9, 0])  # the request is over the local window; its compacted form fits
    monkeypatch.setattr(llm_local, "_estimate_message_chars", lambda _messages: next(sizes))
    kept_a, *kept_middle, kept_c = (block["text"] for block in
                                    LLMClient()._prepare_messages_for_local_context(messages, 8192, 512)[0]["content"])
    assert section(b, "## Identity") in kept_middle[-1] and section(b, "## My story") in kept_middle[-1]
    assert section(c, "## This room (Main)") in kept_c and section(a, "## BIBLE.md") in kept_a
    if book:  # the handbook compacts as a book, like the governance block before it
        assert kept_middle[0].startswith("## DEVELOPMENT.md\n\n[Compacted for local-model context")


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
    floor = section(built.messages_for("nano")[0]["content"][-1]["text"], "### Physical floor")
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


@pytest.mark.parametrize("book", [False, True])
def test_codex_keeps_identity_and_my_story_in_the_leading_system_group(tmp_path, book):
    """A', optional D and B remain system items; only C becomes a notice."""
    from ouroboros.llm_messages import HOST_CONTEXT_NOTICE_BEFORE_TASK
    from tests.test_memory_view_story import _page

    env, memory, _rooms = world(tmp_path)
    task = {**MAIN, "workspace_root": "" if book else str(tmp_path / "project")}
    messages, _cap = _messages(env, memory, task)
    wire, target = _codex_wire(messages)
    stable, c = messages[0]["content"][:-1], messages[0]["content"][-1]["text"]
    boundary = len(stable)
    assert [message["role"] for message in wire] == ["system"] * boundary + ["user", "user"]
    assert target["wire_layout"] == {"system_prefix_split": True, "moved_blocks": 1}
    assert [item["content"] for item in wire[:boundary]] == [[{"type": "text", "text": block["text"]}] for block in stable]
    assert "## My story" in wire[boundary - 1]["content"][0]["text"]
    assert any(item["content"][0]["text"].startswith("## DEVELOPMENT.md\n") for item in wire[:boundary]) == book
    notice = wire[boundary]["content"]
    assert HOST_CONTEXT_NOTICE_BEFORE_TASK in notice and notice.endswith(c)
    assert "## My story" not in notice and "## Shared understanding" in notice

    round2 = messages + [{"role": "assistant", "content": "working"}, {"role": "user", "content": "go on"}]
    wire2, _target = _codex_wire(round2)
    assert wire2[:len(wire)] == wire

    (memory.drive_root / "memory" / "knowledge" / "overview.md").write_text(
        "# Overview\n\nWhat is true now: the cache keeps my story.\n", encoding="utf-8")
    edited, _target = _codex_wire(_messages(env, memory, task)[0])
    assert edited[:boundary] == wire[:boundary] and edited[boundary] != wire[boundary]

    _page(memory.drive_root, "1", 11, 12, text="A page appended to my story.")
    paged, _target = _codex_wire(_messages(env, memory, task)[0])
    assert paged[:boundary - 1] == wire[:boundary - 1] and paged[boundary - 1] != wire[boundary - 1]
    assert "A page appended to my story." in paged[boundary - 1]["content"][0]["text"]


@pytest.mark.parametrize("book", [False, True])
def test_direct_openai_and_openrouter_keep_all_stable_items_and_one_notice(tmp_path, monkeypatch, book):
    from ouroboros.llm import LLMClient
    from ouroboros.llm_messages import HOST_CONTEXT_NOTICE_BEFORE_TASK

    monkeypatch.setattr(LLMClient, "_SUPPORTED_PARAMS_FETCHED", True, raising=False)
    monkeypatch.setattr("ouroboros.pricing._fetch_live_rows", lambda *_a, **_kw: {})
    env, memory, _rooms = world(tmp_path)
    messages, _cap = _messages(env, memory, {**MAIN, "workspace_root": "" if book else str(tmp_path / "project")})
    stable, c = messages[0]["content"][:-1], messages[0]["content"][-1]["text"]
    tools = [{"type": "function", "function": {"name": "probe", "parameters": {"type": "object", "properties": {}}}}]
    client = LLMClient(api_key="unused")
    for model, key in (("openai::gpt-6-sol", "OPENAI_API_KEY"), ("openai/gpt-6-sol", "OPENROUTER_API_KEY")):
        monkeypatch.setenv(key, "unused")
        target = client._resolve_remote_target(model)
        payload = client._build_remote_kwargs(target, copy.deepcopy(messages), "high", 512, "auto", None, tools,
                                             skip_capability_fetch=True)
        client._normalize_payload_cache_ttl(target, payload)
        wire, boundary = payload["messages"], len(stable)
        assert [message["role"] for message in wire] == ["system"] * boundary + ["user", "user"], model
        assert [message["content"][0]["text"] for message in wire[:boundary]] == [block["text"] for block in stable]
        assert all(("cache_control" in message["content"][0]) == (key == "OPENROUTER_API_KEY") for message in wire[:boundary])
        assert all("cache_control" not in tool for tool in payload["tools"])
        assert HOST_CONTEXT_NOTICE_BEFORE_TASK in wire[boundary]["content"] and wire[boundary]["content"].endswith(c)
        assert target["wire_layout"] == {"system_prefix_split": True, "moved_blocks": 1}


def test_the_task_input_pointer_follows_the_calibrated_nano_request_and_the_floor_counts_resident_schemas(tmp_path, monkeypatch):
    """A11. The pointer trigger compares the CALIBRATED Nano request, schemas it sends included, plus the
    reply floor with the target: a fitting calibrated input gets no pointer, and a schema-heavy request that
    only the schemas push over the target gets one. On a route switch the floor's fixed part counts the
    schemas a running task actually sends (an enable_tools addition), not a fresh Nano selection that drops them."""
    from types import SimpleNamespace

    from ouroboros import context, context_fit

    env, memory, _rooms = world(tmp_path)
    task = {"type": "task", "text": "x" * 240_000, **MAIN}  # ~60K raw tokens: under the target alone, over it with the schemas
    core = context._capture_context_core(env, memory, task, None, None)
    big = [{"type": "function", "function": {"name": f"tool_{i}", "description": "d" * 4_000,
                                             "parameters": {"type": "object", "properties": {}}}} for i in range(30)]
    meta = [{"type": "function", "function": {"name": name, "description": "m", "parameters": {"type": "object", "properties": {}}}}
            for name in ("enable_tools", "list_available_tools")]
    heavy_meta = [{"type": "function", "function": {"name": "enable_tools", "description": "d" * 120_000,  # ~30K tokens Nano DOES send
                                                    "parameters": {"type": "object", "properties": {}}}}, meta[1]]

    def plan(tools, ratio):
        monkeypatch.setattr(context_fit, "_route_calibration_ratio", lambda *_a, **_kw: ratio)
        return context._build_context_fit_plan(env, core, task, preferred_mode="nano", tool_schemas=tools,
                                               route_resolver=lambda *_a, **_kw: ({"model": "m", "provider": "p"}, SimpleNamespace(
                                                   route_fp="r", status="asserted", stale=False, window_tokens=1_000_000)))

    pointer = lambda built: built.nano_projection.user_content_json is not None  # noqa: E731
    assert not pointer(plan(meta, 1.0))  # 60K + floor < 85K
    assert not pointer(plan(meta + big, 1.0))  # schemas the Nano request does not send do not count
    assert pointer(plan(heavy_meta, 1.0))  # the 30K of schemas the Nano request sends push it over
    assert not pointer(plan(heavy_meta, 0.8))  # calibrated: (60K + 30K) x 0.8 + floor fits the target
    assert pointer(plan(meta, 1.5))  # calibrated the other way: a dense tokenizer makes the same input a pointer

    # The floor on a route switch: the resident list (a big schema enable_tools added) narrows Nano's room.
    built = plan(meta, 1.0)
    bare = built.reproject_for_route(window_tokens=1_000_000, known_window=True, ratio=1.0, output_reserve=65_536,
                                     tool_schemas=meta, start_mode="nano")
    loaded = built.reproject_for_route(window_tokens=1_000_000, known_window=True, ratio=1.0, output_reserve=65_536,
                                       tool_schemas=meta + big, start_mode="nano")
    room = lambda plan_: plan_.nano_projection.memory_facts["floor"]["allowance_tokens"]  # noqa: E731
    assert room(bare) - room(loaded) >= 30_000
    # The first request keeps the owner-mode selection: a full list is not what Nano sends there.
    assert (built.nano_projection.memory_facts["floor"]["physical_allowance_tokens"]
            == plan(meta + big, 1.0).nano_projection.memory_facts["floor"]["physical_allowance_tokens"])
