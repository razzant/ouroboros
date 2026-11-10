"""Content owed by a first request, independent of the memory view's role flags.

This small installation goes through the dialogue, knowledge, chronicle, place and
task writers, then the same build_llm_messages boundary as a real request. Keep the
expectations literal: changing a role's coverage must change this contract visibly.
The lower-level rendering cases remain in test_memory_view_room; live delivery is
covered in test_owner_relay.
"""
from __future__ import annotations

import copy
import json
import re
from types import SimpleNamespace

import pytest

from ouroboros import dialogue_store, memory_view
from ouroboros.chat_chain import format_address, row_address
from ouroboros.context import build_llm_messages
from ouroboros.contracts.task_constraint import normalize_task_constraint
from ouroboros.knowledge import resolve_knowledge_address, write_knowledge_note
from ouroboros.project_dialogue import append_canonical_task_summary, build_owner_message_ref
from ouroboros.projects_registry import bind_task_to_project, create_project
from ouroboros.task_results import write_task_result
from ouroboros.tools.chronicle import _chronicle_write, _memory_mark
from ouroboros.tools.registry import ToolContext, ToolRegistry
from tests._memory_view_context import section
from tests.test_cache_optimization import _make_env_and_memory


def _text(messages):
    return "\n\n".join(
        content if isinstance(content := message["content"], str)
        else "\n\n".join(block["text"] for block in content if block.get("type") == "text")
        for message in messages
    )


def _context(world, task):
    metadata = {**task, **task.get("metadata", {})}
    return ToolContext(
        repo_dir=world.env.repo_dir, drive_root=world.memory.drive_root,
        task_id=task["id"], current_chat_id=task["chat_id"], task_metadata=metadata,
        project_id=task.get("project_id", ""),
        task_contract=task.get("task_contract") or {},
        task_constraint=normalize_task_constraint(task.get("task_constraint")),
        is_direct_chat=bool(task.get("_is_direct_chat")),
    )


def _request(world, case, *, window=1_000_000):
    task = copy.deepcopy(world.tasks[case])
    ctx = _context(world, task)
    evidence = SimpleNamespace(route_fp="content-contract", status="asserted", stale=False, window_tokens=window)
    messages, cap = build_llm_messages(
        world.env, world.memory, task, ctx=ctx,
        route_resolver=lambda *_args, **_kwargs: ({"model": "fixture", "provider": "fixture"}, evidence),
    )
    return messages, cap, ctx


def _read(ctx, name, **arguments):
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    result = registry.execute_result(name, arguments)
    assert result.status == "ok", result.text
    return result.text


@pytest.fixture
def world(tmp_path, monkeypatch):
    from ouroboros import places
    from ouroboros.presence_admission import admit_presence_turn
    from ouroboros.presence_bindings import conversation_key
    from ouroboros.presence_runner import PresenceTurnEvent, capture_presence_event

    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "max")
    env, memory = _make_env_and_memory(tmp_path)
    root = memory.drive_root
    monkeypatch.setattr("ouroboros.config.DATA_DIR", root)
    alpha = create_project(root, "contract-alpha", name="Contract Alpha")
    beta = create_project(root, "contract-beta", name="Contract Beta")
    rows = {}

    def say(key, chat, minute, text, *, direction="in", **fields):
        row = {"chat_id": chat, "ts": f"2026-11-10T09:{minute:02d}:00+00:00",
               "direction": direction, "text": text, **fields}
        if direction == "in":
            row["client_message_id"] = key
        assert dialogue_store.append(root, "chat", row) is not None
        rows[key] = row
        return row

    say("sealed-human", 1, 0, "Yesterday I asked about the copper bridge.")
    say("sealed-reply", 1, 1, "Yesterday I checked the copper bridge.", direction="out",
        task_id="earlier-main", root_task_id="earlier-main", delegation_role="root")
    say("main-human", 1, 2, "Please remember the amber notebook.")
    say("main-reply", 1, 3, "I have the amber notebook in view.", direction="out",
        task_id="main-reply", root_task_id="main-reply", delegation_role="root")
    origin = say("assignment", alpha["chat_id"], 4, "Check the silver compass exactly as drawn.")
    say("alpha-reply", alpha["chat_id"], 5, "I will check every silver compass bearing.", direction="out",
        task_id="contract-root", root_task_id="contract-root", delegation_role="root")
    say("beta-human", beta["chat_id"], 6, "Keep the violet orchard open until I return.")
    say("beta-reply", beta["chat_id"], 7, "The violet orchard will stay open.", direction="out",
        task_id="earlier-beta", root_task_id="earlier-beta", delegation_role="root")
    say("child-report", beta["chat_id"], 8, "PRIVATE CHILD REPORT: soil samples are attached.", direction="out",
        task_id="earlier-child", parent_task_id="earlier-beta", root_task_id="earlier-beta",
        subagent_task_id="earlier-child", delegation_role="subagent")
    write_task_result(root, "earlier-beta", "completed", chat_id=beta["chat_id"],
                      root_task_id="earlier-beta", delegation_role="root",
                      result="EARLIER REPORT BODY: all orchard measurements.")
    assert append_canonical_task_summary(root, {
        "summary_id": "contract-earlier-beta", "type": "task_summary", "direction": "system",
        "summary_kind": "terminal_root_projection", "status": "completed",
        "task_id": "earlier-beta", "root_task_id": "earlier-beta", "delegation_role": "root",
        "chat_id": beta["chat_id"], "ts": "2026-11-10T09:09:00+00:00",
        "text": "Completed. Root task earlier-beta.", "wake_summary": "The orchard measurements are complete.",
    })
    ref = build_owner_message_ref(chat_id=alpha["chat_id"], client_message_id="assignment",
                                  ts=origin["ts"], text=origin["text"])
    bind_task_to_project(root, "contract-root", alpha["id"], origin={"ref": ref, "text": origin["text"]})
    tasks = {
        "root": {"id": "contract-root", "chat_id": alpha["chat_id"], "project_id": alpha["id"],
                 "root_task_id": "contract-root", "delegation_role": "root",
                 "origin_message_ref": ref, "origin_message_text": origin["text"]},
        "main": {"id": "contract-main", "chat_id": 1, "_is_direct_chat": True},
        "consciousness": {"id": "contract-wake", "chat_id": 1, "_is_direct_chat": True,
                          "metadata": {"initiator": "consciousness", "usage_category": "consciousness"}},
    }
    for name, kind in (("child", "api_model"), ("nanny", "agent_session")):
        tasks[name] = {"id": f"contract-{name}", "chat_id": alpha["chat_id"], "project_id": alpha["id"],
                       "delegation_role": "subagent", "parent_task_id": "contract-root", "root_task_id": "contract-root",
                       "task_constraint": {"mode": "local_readonly_subagent"},
                       "configured_subagent": {"id": name, "route": {"kind": kind, "target_id": "fixture"}}}
    for task in tasks.values():
        task.update(type="task", text="Inspect the assigned fixture.")
        write_task_result(root, task["id"], "running", **{key: value for key, value in task.items() if key != "id"})

    # Projects are always engaged. A place where only a guest has spoken exercises
    # the real quiet-room branch without changing any memory-view flags.
    for conversation, text in (("quiet", "QUIET GUEST WORDS: the indigo ferry arrived."),
                               ("presence", "Please inspect the white lantern.")):
        places.connect_place(root, connector="contract-bridge", provider="chat", account_id="fixture",
                             conversation_id=conversation, label=f"Contract {conversation}")
        admission = admit_presence_turn(drive_root=root, authenticated_transport_skill="contract-bridge",
                                       provider="chat", account_id="fixture", conversation_id=conversation)
        event = PresenceTurnEvent(
            f"contract-{conversation}-event", "chat", "fixture", conversation, "",
            conversation_key("chat", "fixture", conversation, ""),
            {"platform_actor_id": "guest", "display_name": "Guest"},
            {"title": f"Contract {conversation}"}, {"sent_at": "2026-11-10T09:10:00+00:00"}, text,
        )
        captured = capture_presence_event(admission, event, drive_root=root)
        if conversation == "presence":
            tasks["presence"] = captured
        else:
            quiet = captured["chat_id"]

    for topic, content in (
        ("overview", "# Our common understanding\n\nThe brass atlas connects all of my rooms.\n"),
        ("contract-navigation", "---\ntype: note\nsummary: Compare compass bearings before drawing a route.\n---\n"
         "# Navigation practice\n\nNOTE BODY: the east bearing is engraved underneath.\n"),
        ("contract-person", "---\ntype: person\nsummary: Mira prefers exact quotations with their dates.\n---\n"
         "# Mira\n\nPERSON BODY: Mira keeps a blue field journal.\n"),
    ):
        saved = write_knowledge_note(resolve_knowledge_address(root, topic, "global"), content,
                                     task_id="contract-root", writer="turn")
        assert saved.ok, saved

    result = SimpleNamespace(env=env, memory=memory, tasks=tasks, rows=rows, alpha=alpha, beta=beta, quiet=quiet)
    ctx = _context(result, tasks["root"])
    page = json.loads(_chronicle_write(ctx, kind="page", room_id="1",
                      text="I checked the copper bridge yesterday; its eastern support is sound.",
                      covers={"from": format_address(row_address(rows["sealed-human"])),
                              "to": format_address(row_address(rows["sealed-reply"]))}))
    assert page["ok"], page
    result.page = page["node_id"]
    for target, scope, text in (
        ({"node_id": result.page}, "global", "GLOBAL MARK: keep the original measurements."),
        ({"address": format_address(row_address(rows["assignment"])), "room_id": str(alpha["chat_id"])},
         "room", "ALPHA MARK: compare the compass with the drawing."),
        ({"address": format_address(row_address(rows["beta-human"])), "room_id": str(beta["chat_id"])},
         "room", "BETA MARK: leave the orchard gate unlocked."),
    ):
        marked = json.loads(_memory_mark(ctx, **target, scope=scope, text=text))
        assert marked["ok"], marked
    return result


@pytest.mark.parametrize("case,role,full,story", [
    ("root", "integrator", True, True),
    ("main", "integrator", True, True),
    ("consciousness", "consciousness", True, True),
    ("presence", "presence", True, True),
    ("child", "child", False, True),
    ("nanny", "nanny", False, False),
])
def test_first_request_content_contract(world, case, role, full, story):
    assert set(memory_view.ROLE_DEFAULTS) == {"integrator", "consciousness", "presence", "child", "nanny"}
    messages, cap, ctx = _request(world, case)
    text = _text(messages)
    assert cap["context_fit"]["initial_mode"] == "max"
    assert memory_view.view_role(world.tasks[case]) == role
    for words in (
        "GLOBAL MARK: keep the original measurements.",
        "ALPHA MARK: compare the compass with the drawing.",
        "Mira prefers exact quotations with their dates.",
        "Navigation practice", "Mira", "contract-navigation", "contract-person",
        "Check the silver compass exactly as drawn.",
    ):
        assert words in text, (case, words)
    for words in (
        "Please remember the amber notebook.", "I have the amber notebook in view.",
        "I will check every silver compass bearing.",
        "Keep the violet orchard open until I return.", "The violet orchard will stay open.",
        "BETA MARK: leave the orchard gate unlocked.",
        "The brass atlas connects all of my rooms.",
        "Compare compass bearings before drawing a route.",
        "## Other conversations", "## Shared understanding", "## Knowledge base",
        "The orchard measurements are complete.", "get_task_result(task_id='earlier-beta')",
    ):
        assert (words in text) is full, (case, words)
    for words in ("## My story", "I checked the copper bridge yesterday; its eastern support is sound."):
        assert (words in text) is story, (case, words)
    for words in (
        "QUIET GUEST WORDS: the indigo ferry arrived.",
        "PRIVATE CHILD REPORT: soil samples are attached.",
        "EARLIER REPORT BODY: all orchard measurements.",
        "NOTE BODY: the east bearing is engraved underneath.",
        "PERSON BODY: Mira keeps a blue field journal.",
        "Yesterday I asked about the copper bridge.", "Yesterday I checked the copper bridge.",
    ):
        assert words not in text, (case, words)

    # Omitted knowledge bodies have named topics and the reader beside their index.
    assert "knowledge_read(topic=..., scope='global')" in text
    assert "NOTE BODY: the east bearing is engraved underneath." in _read(ctx, "knowledge_read", topic="contract-navigation", scope="global")
    assert "PERSON BODY: Mira keeps a blue field journal." in _read(ctx, "knowledge_read", topic="contract-person", scope="global")
    beta_rows = _read(ctx, "memory_read", room_id=str(world.beta["chat_id"]), rows=True)
    assert "PRIVATE CHILD REPORT: soil samples are attached." in beta_rows
    assert "Keep the violet orchard open until I return." in beta_rows
    if full:
        other = section(text, "## Other conversations")
        assert "Keep the violet orchard open until I return." in other
        assert f"memory_read(room_id='{world.beta['chat_id']}', rows=true)" in other
        quiet_line = next(line for line in other.splitlines() if line.startswith("- ") and f"chat_id={world.quiet}]" in line)
        assert "words nobody considered" in quiet_line
        assert f"memory_read(room_id='{world.quiet}', rows=true)" in quiet_line
        assert "QUIET GUEST WORDS: the indigo ferry arrived." in _read(ctx, "memory_read", room_id=str(world.quiet), rows=True)
        assert "EARLIER REPORT BODY: all orchard measurements." in _read(ctx, "get_task_result", task_id="earlier-beta")
    else:
        assert "## Knowledge outline" in text and "## Working sources" in text
        missing = section(text, "## Working sources").split("Not loaded: ", 1)[1]
        for item in ("raw conversations", "other conversations", "the overview", "the whole knowledge index",
                     "the global scratchpad", "earlier task reports", "memory_read", "chat_history",
                     "knowledge_list", "knowledge_read", "get_task_result"):
            assert item in missing, (case, item)
        assert ("the top level of your story" in missing) is (case == "nanny")
        assert "The brass atlas connects all of my rooms." in _read(ctx, "knowledge_read", topic="overview", scope="global")
        assert "Compare compass bearings before drawing a route." in _read(ctx, "knowledge_list", scope="global")
        assert "EARLIER REPORT BODY: all orchard measurements." in _read(ctx, "get_task_result", task_id="earlier-beta")
        story_index = _read(ctx, "memory_read", story=True)
        assert f"memory_read(node_id='{world.page}')" in story_index
    if story:
        assert world.page in text
    # The page and the words it sealed remain readable, including by the nanny.
    page = _read(ctx, "memory_read", node_id=world.page)
    assert "I checked the copper bridge yesterday; its eastern support is sound." in page
    assert "Yesterday I asked about the copper bridge." in _read(ctx, "memory_read", room_id="1", rows=True)


def test_physical_window_replaces_words_with_resolvable_addresses(world):
    """A real route window forces the same captured world to give up words, never their source."""
    chat = world.beta["chat_id"]
    rows = [
        {"chat_id": chat, "ts": "2026-11-10T09:11:00+00:00", "direction": "in",
         "client_message_id": "long-owner", "text": "LONG OWNER SURVEY: " + "violet orchard bearing; " * 15_000},
        {"chat_id": chat, "ts": "2026-11-10T09:12:00+00:00", "direction": "out",
         "task_id": "earlier-beta", "root_task_id": "earlier-beta", "delegation_role": "root",
         "text": "LONG OUROBOROS SURVEY: " + "orchard measurements checked; " * 15_000},
    ]
    for row in rows:
        assert dialogue_store.append(world.memory.drive_root, "chat", row) is not None
    wide, wide_cap, _ = _request(world, "root")
    narrow, cap, ctx = _request(world, "root", window=65_537)
    whole, text = _text(wide), _text(narrow)
    assert wide_cap["context_fit"]["initial_mode"] == "max"
    assert cap["context_fit"]["preferred_mode"] == "max"
    assert cap["context_fit"]["window_tokens"] == 65_537
    assert "### Physical floor" in text
    assert "### Physical floor" not in whole
    assert "The brass atlas connects all of my rooms." in text
    assert "GLOBAL MARK: keep the original measurements." in text
    assert "Mira prefers exact quotations with their dates." in text
    for row in rows:
        assert row["text"] in whole
        assert row["text"] not in text
        assert format_address(row_address(row)) in text
    assert "read by address" in text
    # The room still accounts for the two original and two long rows.
    other = section(text, "## Other conversations")
    room_line = next(line for line in other.splitlines() if f"memory_read(room_id='{chat}', rows=true)" in line)
    assert "people 2, mine 2" in room_line
    # Use the advertised room reader and its continuations through real dispatch.
    # Every byte remains readable even though neither body fits the request.
    for row in rows:
        address = format_address(row_address(row))
        chunks, start = [], 0
        while True:
            chunk = _read(ctx, "memory_read", room_id=str(chat), rows=True, start=start,
                          **{"from": address, "to": address})
            head, continuation, body = chunk.split("\n", 2)
            assert str(chat) in head and address in body
            fragment = re.search(r"\(chars \d+–\d+ of \d+\) (.*)", body, re.DOTALL)
            assert fragment, body[:200]
            chunks.append(fragment[1])
            if continuation.startswith("complete:"):
                break
            next_start = int(re.search(r"\bstart=(\d+)", continuation)[1])
            assert next_start > start
            start = next_start
        assert "".join(chunks) == row["text"]
