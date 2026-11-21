"""Content owed by a first request, independent of the memory view's role flags.

This small installation goes through the dialogue, knowledge, chronicle, place and
task writers, then the same build_llm_messages boundary as a real request. Keep the
expectations literal: changing a role's coverage must change this contract visibly.
The lower-level rendering cases remain in test_memory_view_room; live delivery of my
human's words is covered in test_owner_relay. The running half of marks is here: a mark
another work makes, changes or releases after a focus's view was captured reaches that
focus at its next round through the real round hook (``mark_anchors.append_mark_changes``).
"""
from __future__ import annotations

import copy
import datetime
import json
import queue
import re
from types import SimpleNamespace

import pytest

from ouroboros import dialogue_store, memory_view
from ouroboros.chat_chain import format_address, row_address
from ouroboros.chronicle_store import ChronicleStore
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

    # Projects are always engaged. A recorded guest also makes a place a conversation
    # (commit 4, provider-independent rule), even before I answer there.
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
        "QUIET GUEST WORDS: the indigo ferry arrived.",
    ):
        assert (words in text) is full, (case, words)
    for words in ("## My story", "I checked the copper bridge yesterday; its eastern support is sound."):
        assert (words in text) is story, (case, words)
    for words in (
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
        captured = memory_view.snapshot_from_json(ctx.context_fit_plan.core.memory_view_json)
        guest = next(r for g in captured.conversations for r in g["rooms"] if r["room_id"] == str(world.quiet))
        assert guest["engaged"] and len(guest["words"]) == 1
        word = guest["words"][0]
        assert word["line"] in other and word["address"] in other and "Guest" in word["author"]
        assert not word["owner_side"] and f"memory_read(room_id='{world.quiet}', rows=true)" in other
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


@pytest.mark.parametrize("reply_task", ["earlier-beta", "untold-beta"])
def test_physical_window_replaces_words_with_resolvable_addresses(world, reply_task):
    """A real route window forces the same captured world to give up words, never their source."""
    chat = world.beta["chat_id"]
    rows = [
        {"chat_id": chat, "ts": "2026-11-10T09:11:00+00:00", "direction": "in",
         "client_message_id": "long-owner", "text": "LONG OWNER SURVEY: " + "violet orchard bearing; " * 15_000},
        {"chat_id": chat, "ts": "2026-11-10T09:12:00+00:00", "direction": "out",
         "task_id": reply_task, "root_task_id": reply_task, "delegation_role": "root",
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
        if row["direction"] == "in" or reply_task == "untold-beta":
            assert format_address(row_address(row)) in text
    assert "read by address" in text
    # The room still accounts for the two original and two long rows.
    other = section(text, "## Other conversations")
    room_line = next(line for line in other.splitlines() if f"memory_read(room_id='{chat}', rows=true)" in line)
    assert "people 2, mine 2" in room_line
    if reply_task == "earlier-beta":
        # Plan v7, law 1 and commit 2: this work's telling stands for its replies.
        # Its task and the room reader still identify the author and every source row.
        task_line = next(line for line in other.splitlines() if "host; task earlier-beta]" in line)
        assert "«The orchard measurements are complete.»" in task_line
        assert "get_task_result(task_id='earlier-beta')" in task_line
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


# --- the running half: a mark another work changes reaches each running focus of its scope at its next round -------

CHANGES = "ouroboros_mark_changes"
HEADER = ("[Host facts: marks of this room and global ones changed while this work runs — "
          "journal sequence, the change, who and when, then the mark's own words]")


def _beta_root(world):
    """The second room's running root: two rooms, a running root in each."""
    root = world.memory.drive_root
    task = {"id": "contract-beta-root", "chat_id": world.beta["chat_id"], "project_id": world.beta["id"],
            "root_task_id": "contract-beta-root", "delegation_role": "root", "type": "task",
            "text": "Inspect the assigned fixture."}
    bind_task_to_project(root, task["id"], world.beta["id"], origin={"absent": "post_hoc_unresolved"})
    write_task_result(root, task["id"], "running", **{key: value for key, value in task.items() if key != "id"})
    world.tasks["beta"] = task


def _round(world, ctx, messages, number, **fold):
    """One round of a running focus through the loop's own hook (after its drain, before its send)."""
    from ouroboros.loop_round_limits import _CompactionRoundContext, _run_round_compaction

    root = world.memory.drive_root
    tools = SimpleNamespace(_ctx=ctx, schemas=lambda: [])
    return _run_round_compaction(messages, _CompactionRoundContext(
        tools=tools, drive_root=root, drive_logs=root / "logs", task_id=ctx.task_id, round_idx=number,
        event_queue=None, emit_progress=lambda _text: None, **fold))


def _changes(messages):
    return [row["content"] for row in messages if row.get("_host_context_kind") == CHANGES]


def _marked(ctx, **fields):
    marked = json.loads(_memory_mark(ctx, **fields))
    assert marked["ok"], marked
    return marked


def _released(world, writer, mark_id, reason):
    assert json.loads(_memory_mark(writer, release_ids=[mark_id], reason=reason))["ok"]
    return ChronicleStore(world.memory.drive_root).records(kinds=["mark_release"])[-1]


def _at(world, sequence):
    """The UTC minute of the journal record at ``sequence``: when its change was made."""
    record = ChronicleStore(world.memory.drive_root).records(after_seq=sequence - 1, limit=1)[0]
    return datetime.datetime.fromisoformat(record["ts"]).astimezone(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")


@pytest.fixture
def quiet_roster(monkeypatch):
    """The independent-roots note is another producer of the same tail; these scenes read only the mark rows."""
    monkeypatch.setattr("ouroboros.peer_roster.maybe_append_roster_note", lambda *_args: False)


def test_a_mark_made_while_works_run_reaches_each_running_focus_of_its_scope_once(world, quiet_roster):
    """A global mark reaches every running focus; a room mark that room's root, helper and nanny, never the other
    room's root; the Main turn that made them gets no echo. Each change arrives once with its words, quote, author,
    time and scope, appended after the request already sent, whose bytes stay as they were."""
    from ouroboros.transcript_prefix import observe_send

    _beta_root(world)
    running = {case: _request(world, case) for case in ("root", "child", "nanny", "beta", "main")}
    sent = {}
    for case, (messages, _cap, ctx) in running.items():
        sent[case] = _round(world, ctx, copy.deepcopy(messages), 1)
        assert _changes(sent[case]) == [], case  # nothing changed since the capture
        assert observe_send(ctx, sent[case], round_idx=1) is None
    writer = running["main"][2]
    alpha_room, beta_room = str(world.alpha["chat_id"]), str(world.beta["chat_id"])
    alpha_address = format_address(row_address(world.rows["assignment"]))
    beta_address = format_address(row_address(world.rows["beta-human"]))
    glob = _marked(writer, node_id=world.page, scope="global", quote="eastern support",
                   text="LIVE GLOBAL: measure the bridge again before any route.")
    alpha = _marked(writer, address=alpha_address, room_id=alpha_room, text="LIVE ALPHA: read the compass twice.")
    beta = _marked(writer, address=beta_address, room_id=beta_room, text="LIVE BETA: lock the orchard gate at night.")
    lines = {
        "global": [f"- #{glob['sequence']} mark {glob['mark_id']}: marked by main, task contract-main, in Main at "
                   f"{_at(world, glob['sequence'])}; [global; Main]; target node {world.page}",
                   "  LIVE GLOBAL: measure the bridge again before any route.", "  quote: eastern support"],
        "alpha": [f"- #{alpha['sequence']} mark {alpha['mark_id']}: marked by main, task contract-main, in Main at "
                  f"{_at(world, alpha['sequence'])}; [room; Project Contract Alpha [chat_id={alpha_room}]]; "
                  f"target {alpha_address}", "  LIVE ALPHA: read the compass twice."],
        "beta": [f"- #{beta['sequence']} mark {beta['mark_id']}: marked by main, task contract-main, in Main at "
                 f"{_at(world, beta['sequence'])}; [room; Project Contract Beta [chat_id={beta_room}]]; "
                 f"target {beta_address}", "  LIVE BETA: lock the orchard gate at night."],
    }
    reach = {"root": ("global", "alpha"), "child": ("global", "alpha"), "nanny": ("global", "alpha"),
             "beta": ("global", "beta"), "main": ()}
    for case, (_messages, _cap, ctx) in running.items():
        before = copy.deepcopy(sent[case])
        after = _round(world, ctx, sent[case], 2)
        assert after[:len(before)] == before, case  # the request already sent keeps its bytes
        assert observe_send(ctx, after, round_idx=2) is None, case  # a pure append: its cached prefix holds
        expected = ["\n".join([HEADER, *(line for name in reach[case] for line in lines[name])])] if reach[case] else []
        assert _changes(after) == expected, case
        assert _changes(_round(world, ctx, after, 3)) == expected, case  # once


def test_my_own_mark_is_not_echoed_and_another_works_view_change_and_release_of_it_are_shown(world, quiet_roster):
    """My own mark is no news to me, but it is to my helper; another work's view change and release of my mark reach
    us both, and a quote hidden by the view change stays hidden in the release."""
    (root_messages, _cap, root), (child_messages, _cap, child) = _request(world, "root"), _request(world, "child")
    writer = _context(world, world.tasks["main"])
    alpha_room = str(world.alpha["chat_id"])
    alpha = f"Project Contract Alpha [chat_id={alpha_room}]"
    own = _marked(root, node_id=world.page, room_id=alpha_room, quote="eastern support",
                  text="ROOT OWN: compare every bearing twice.")
    marked_on = _at(world, own["sequence"])[:10]
    root_messages = _round(world, root, root_messages, 2)
    assert _changes(root_messages) == []  # my own mark comes back to me as no news
    child_messages = _round(world, child, child_messages, 2)  # my helper is another work
    assert _changes(child_messages) == ["\n".join([
        HEADER, f"- #{own['sequence']} mark {own['mark_id']}: marked by root, task contract-root, in {alpha} at "
                f"{_at(world, own['sequence'])}; [room; {alpha}]; target node {world.page}",
        "  ROOT OWN: compare every bearing twice.", "  quote: eastern support"])]
    assert json.loads(_memory_mark(writer, mark_id=own["mark_id"], visibility="meaning",
                                   reason="Its meaning is enough."))["ok"]
    view = ChronicleStore(world.memory.drive_root).records(kinds=["mark_view"])[-1]
    viewed = "\n".join([
        HEADER, f"- #{view['sequence']} mark {own['mark_id']}: view changed to meaning by main, task contract-main, "
                f"in Main at {_at(world, view['sequence'])}; [room; {alpha}], marked by root on {marked_on}; "
                f"target node {world.page}",
        "  ROOT OWN: compare every bearing twice.", "  reason: Its meaning is enough."])
    root_messages, child_messages = _round(world, root, root_messages, 3), _round(world, child, child_messages, 3)
    assert _changes(root_messages)[-1] == viewed and _changes(child_messages)[-1] == viewed
    release = _released(world, writer, own["mark_id"], "Main settled the bearings.")
    shown = "\n".join([
        HEADER, f"- #{release['sequence']} mark {own['mark_id']}: released by main, task contract-main, in Main at "
                f"{_at(world, release['sequence'])}; [room; {alpha}], marked by root on {marked_on}; "
                f"target node {world.page}",
        "  ROOT OWN: compare every bearing twice.", "  reason: Main settled the bearings."])
    for ctx, messages in ((root, root_messages), (child, child_messages)):
        assert _changes(_round(world, ctx, messages, 4))[-1] == shown, ctx.task_id


def test_another_rooms_words_reach_a_running_work_only_inside_a_mark_never_as_my_humans_words(world, quiet_roster):
    """The owner's objection, its other direction: my human's row in Beta reaches the running Alpha root only as the
    quote of a mark another work made, in a host-facts row; never as my human's words, never whole, never a
    directive."""
    from ouroboros.context_budget import OWNER_WORDS_KIND
    from ouroboros.loop_round_limits import _drain_incoming_messages

    messages, _cap, ctx = _request(world, "root")
    sent, root = copy.deepcopy(messages), world.memory.drive_root
    words = "BETA OWNER: close the orchard gate tonight and tell the night guard."
    row = {"chat_id": world.beta["chat_id"], "ts": "2026-11-10T09:30:00+00:00", "direction": "in", "text": words,
           "client_message_id": "beta-late"}
    assert dialogue_store.append(root, "chat", row) is not None
    _marked(_context(world, world.tasks["main"]), address=format_address(row_address(row)), scope="global",
            quote="close the orchard gate", text="LIVE GLOBAL ON BETA WORDS: the gate decision is for every room.")
    _drain_incoming_messages(messages, queue.Queue(), root, ctx.task_id, None, set(), owner_ctx=ctx)
    messages = _round(world, ctx, messages, 2)
    tail = messages[len(sent):]
    assert messages[:len(sent)] == sent
    [mark] = [row["content"] for row in tail if row.get("_host_context_kind") == CHANGES]
    assert "LIVE GLOBAL ON BETA WORDS: the gate decision is for every room." in mark
    assert "\n  quote: close the orchard gate" in mark
    shown = json.dumps(tail, ensure_ascii=False)
    assert words not in shown and "[Message from my human]" not in shown  # never whole, never as my human's words
    assert OWNER_WORDS_KIND not in {row.get("_host_context_kind") for row in messages}
    assert format_address(row_address(row)) not in (getattr(ctx, "_held_owner_messages", None) or {})
    assert not getattr(ctx, "_owner_directives", None)  # no directive registered


def test_a_fold_leaves_each_changed_marks_current_change_stated(world, quiet_roster, monkeypatch):
    """compact_context(keep_unit_ids=[]) folds the change rows with the rest of the prose; in the same round the host
    states each changed mark's latest change again: the release stays, the mark it ended does not look in force,
    and the change the release superseded is not repeated."""
    from ouroboros import loop
    from ouroboros.tools.compact_context import _compact_context, record_context_view

    monkeypatch.setattr(loop, "_emit_checkpoint_event", lambda *_args, **_kwargs: None)
    messages, _cap, ctx = _request(world, "root")
    writer = _context(world, world.tasks["main"])
    glob = _marked(writer, node_id=world.page, scope="global", text="LIVE GLOBAL: measure the bridge again.")
    messages = _round(world, ctx, messages, 2) + [{"role": "assistant", "content": "I will measure it again."}]
    release = _released(world, writer, glob["mark_id"], "The bridge is measured.")
    messages = _round(world, ctx, messages, 3) + [{"role": "assistant", "content": "The measurement is settled."}]
    assert [f"- #{glob['sequence']} " in row for row in _changes(messages)] == [True, False]
    record_context_view(ctx, messages, [])
    reply = _compact_context(ctx, working_note="I keep the compass work and its measurements.", keep_unit_ids=[])
    assert "requested" in reply, reply
    messages += [{"role": "assistant", "tool_calls": [{"id": "fold", "type": "function", "function": {
                     "name": "compact_context", "arguments": "{}"}}]},
                 {"role": "tool", "tool_call_id": "fold", "content": reply}]
    folded = _round(world, ctx, messages, 4, tool_schemas=[], fit_candidate=lambda *_args: {"accepted": True})
    assert ctx._context_view_receipt["status"] == "applied", ctx._context_view_receipt
    assert "I will measure it again." not in json.dumps(folded)  # the fold took the prose around the rows
    restated = "\n".join([
        HEADER, f"- #{release['sequence']} mark {glob['mark_id']}: released by main, task contract-main, in Main at "
                f"{_at(world, release['sequence'])}; [global; Main], marked by main on {_at(world, glob['sequence'])[:10]}; "
                f"target node {world.page}",
        "  LIVE GLOBAL: measure the bridge again.", "  reason: The bridge is measured."])
    assert _changes(folded) == [restated]
    assert _changes(_round(world, ctx, folded, 5)) == [restated]  # once


def test_after_a_pause_a_change_made_meanwhile_arrives_and_nothing_stated_repeats(world, quiet_roster):
    """A warm park keeps its process: the next round states each change made while it waited. A cold resume captures
    the view again and continues the saved transcript: the release of the mark a saved row stated arrives, the mark
    the new view already shows is not repeated in the tail, and the saved change does not come back."""
    messages, _cap, ctx = _request(world, "root")
    writer = _context(world, world.tasks["main"])
    first = _marked(writer, node_id=world.page, scope="global", text="LIVE GLOBAL: measure the bridge again.")
    messages = _round(world, ctx, messages, 2) + [{"role": "assistant", "content": "Measuring."}]
    saved = copy.deepcopy(messages)  # what the pause keeps
    release = _released(world, writer, first["mark_id"], "The bridge is measured.")
    later = _marked(writer, node_id=world.page, scope="global", text="LIVE GLOBAL LATER: report in millimetres.")
    released_line = (f"- #{release['sequence']} mark {first['mark_id']}: released by main, task contract-main, in Main "
                     f"at {_at(world, release['sequence'])}; [global; Main], marked by main on "
                     f"{_at(world, first['sequence'])[:10]}; target node {world.page}")
    later_line = (f"- #{later['sequence']} mark {later['mark_id']}: marked by main, task contract-main, in Main at "
                  f"{_at(world, later['sequence'])}; [global; Main]; target node {world.page}")
    warm = _round(world, ctx, copy.deepcopy(messages), 3)  # the same process continues
    fresh, _cap, resumed = _request(world, "root")  # a new process captures its context again
    cold = _round(world, resumed, [fresh[0], *copy.deepcopy(saved[1:])], 3)  # the system row is the new capture's
    assert "LIVE GLOBAL LATER: report in millimetres." in _text(fresh[:1])
    assert "LIVE GLOBAL LATER: report in millimetres." not in _text(saved[:1])
    for after, news in ((warm, [released_line, "  LIVE GLOBAL: measure the bridge again.",
                                "  reason: The bridge is measured.", later_line,
                                "  LIVE GLOBAL LATER: report in millimetres."]),
                        (cold, [released_line, "  LIVE GLOBAL: measure the bridge again.",
                                "  reason: The bridge is measured."])):
        assert _changes(after) == [*_changes(saved), "\n".join([HEADER, *news])]
        assert "\n".join(_changes(after)).count(f"- #{first['sequence']} ") == 1  # the saved change, never again
    assert _changes(_round(world, ctx, warm, 4)) == _changes(warm)
    assert _changes(_round(world, resumed, cold, 4)) == _changes(cold)


def test_a_host_late_review_and_a_works_own_mark_add_no_row(world, quiet_roster):
    """A late review the host keeps in this room stays the next start's, and my own mark is mine: neither adds a
    row. Another work's mark in the next round still arrives, so the round is not silent by construction."""
    from ouroboros.late_review_marks import publish_late_review
    from tests.test_late_review_marks import _record

    messages, _cap, ctx = _request(world, "root")
    root = world.memory.drive_root
    late = publish_late_review(root, _record(root, "earlier-alpha", world.alpha["chat_id"],
                                             "LATE REVIEW WORDS: the drawing is mirrored."))
    assert late.status == "saved", late
    assert any(mark["id"] == late.mark_id and mark["room_id"] == str(world.alpha["chat_id"])
               for mark in ChronicleStore(root).active_marks(None))  # in this room's scope
    _marked(ctx, node_id=world.page, scope="global", text="ROOT OWN GLOBAL: keep the drawing upright.")
    messages = _round(world, ctx, messages, 2)
    assert _changes(messages) == []
    other = _marked(_context(world, world.tasks["main"]), node_id=world.page, scope="global",
                    text="LIVE GLOBAL: measure the bridge again.")
    rows = _changes(_round(world, ctx, messages, 3))
    assert len(rows) == 1 and f"- #{other['sequence']} mark {other['mark_id']}: marked by main" in rows[0]
    assert "LATE REVIEW WORDS" not in rows[0] and "ROOT OWN GLOBAL" not in rows[0]


def _paused(ctx, messages):
    """What a pause keeps: the loop's own continuation values, through JSON (``owner_wait.continuation_state``)."""
    from ouroboros.owner_wait import continuation_state

    return json.loads(json.dumps(continuation_state(ctx, messages, {}, {}, 2, [], set())))


def _resumed(world, case, state):
    """A new process: its view captured again, the saved continuation restored onto it and the system row the new
    capture's (``owner_wait.restore_continuation_state``, then the row ``rebind_restored_route`` sends)."""
    from ouroboros.owner_wait import restore_continuation_state

    fresh, _cap, ctx = _request(world, case)
    messages = []
    restore_continuation_state(SimpleNamespace(_ctx=ctx), state, messages, {}, {}, set())
    messages[0] = fresh[0]
    return messages, ctx


def _start_mark(world, text):
    return {mark["text"]: mark["id"] for mark in ChronicleStore(world.memory.drive_root).active_marks(None)}[text]


def test_a_cold_resume_states_what_was_released_while_paused_though_no_change_row_was_saved(world, quiet_roster):
    """My start view showed a mark and nothing had changed when I paused, so no change row was saved. Meanwhile another
    work released that mark, and made and released another. My new view no longer shows either: both releases arrive
    with their reasons, as a warm park states them; a mark made meanwhile that my new view shows is not repeated."""
    messages, _cap, ctx = _request(world, "root")
    writer = _context(world, world.tasks["main"])
    glob = _start_mark(world, "GLOBAL MARK: keep the original measurements.")
    assert "GLOBAL MARK: keep the original measurements." in _text(messages[:1])
    messages = _round(world, ctx, messages, 2) + [{"role": "assistant", "content": "I keep the original measurements."}]
    assert _changes(messages) == []
    state = _paused(ctx, messages)
    release = _released(world, writer, glob, "New measurements replace the original ones.")
    brief = _marked(writer, node_id=world.page, scope="global", text="PAUSE ONLY: hold the bridge for an hour.")
    gone = _released(world, writer, brief["mark_id"], "The hour passed.")
    later = _marked(writer, node_id=world.page, scope="global", text="LIVE GLOBAL LATER: report in millimetres.")
    warm = _changes(_round(world, ctx, copy.deepcopy(messages), 3))  # the same process continues
    resumed_messages, resumed = _resumed(world, "root", state)
    assert "GLOBAL MARK: keep the original measurements." not in _text(resumed_messages[:1])
    assert "LIVE GLOBAL LATER: report in millimetres." in _text(resumed_messages[:1])
    cold = _round(world, resumed, resumed_messages, 3)
    for rows, news in ((warm, (release, gone, later)), (_changes(cold), (release, gone))):
        assert len(rows) == 1 and [f"- #{change['sequence']} " in rows[0] for change in (release, gone, later)] == [
            change in news for change in (release, gone, later)]
    [row] = _changes(cold)
    assert f"- #{release['sequence']} mark {glob}: released by main, task contract-main" in row
    assert "reason: New measurements replace the original ones." in row and "reason: The hour passed." in row
    assert _changes(_round(world, resumed, cold, 4)) == [row]  # nothing repeats


def _fold(world, ctx, messages, number):
    """compact_context(keep_unit_ids=[]) as the mind calls it, applied by the next round's own hook."""
    from ouroboros.tools.compact_context import _compact_context, record_context_view

    record_context_view(ctx, messages, [])
    reply = _compact_context(ctx, working_note="I keep the compass work and its measurements.", keep_unit_ids=[])
    assert "requested" in reply, reply
    messages = [*messages, {"role": "assistant", "tool_calls": [{"id": f"fold-{number}", "type": "function", "function": {
                    "name": "compact_context", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": f"fold-{number}", "content": reply}]
    folded = _round(world, ctx, messages, number, tool_schemas=[], fit_candidate=lambda *_args: {"accepted": True})
    assert ctx._context_view_receipt["status"] == "applied", ctx._context_view_receipt
    return folded


def _own(ctx, number, **arguments):
    """My own memory_mark call and the result it returned, as the transcript keeps them."""
    reply = _memory_mark(ctx, **arguments)
    assert json.loads(reply)["ok"], reply
    return [{"role": "assistant", "tool_calls": [{"id": f"own-{number}", "type": "function", "function": {
                "name": "memory_mark", "arguments": json.dumps(arguments)}}]},
            {"role": "tool", "tool_call_id": f"own-{number}", "content": reply}]


def test_a_fold_before_the_pause_took_the_last_change_row_and_a_release_made_meanwhile_still_arrives(
        world, quiet_roster, monkeypatch):
    """Before my pause a fold took the last change row (another work had changed a mark I made after my start, and I
    released it: nothing of it is mine to hear again). Resuming does not hang on such rows: a release made while I
    was paused of a mark my start view showed arrives once, and nothing of the folded row comes back."""
    from ouroboros import loop

    monkeypatch.setattr(loop, "_emit_checkpoint_event", lambda *_args, **_kwargs: None)
    messages, _cap, ctx = _request(world, "root")
    writer, glob = _context(world, world.tasks["main"]), _start_mark(world, "GLOBAL MARK: keep the original measurements.")
    mine = _marked(ctx, node_id=world.page, room_id=str(world.alpha["chat_id"]), text="ROOT OWN: compare every bearing twice.")
    assert json.loads(_memory_mark(writer, mark_id=mine["mark_id"], visibility="meaning", reason="Its meaning is enough."))["ok"]
    messages = _round(world, ctx, messages, 2) + [{"role": "assistant", "content": "I compare every bearing twice."}]
    assert len(_changes(messages)) == 1  # another work's change of my mark reached me
    messages = _round(world, ctx, messages + _own(ctx, 1, release_ids=[mine["mark_id"]], reason="Compared."), 3)
    folded = _fold(world, ctx, messages, 4)
    assert _changes(folded) == []  # the fold took the last change row
    state = _paused(ctx, folded)
    release = _released(world, writer, glob, "New measurements replace the original ones.")
    resumed_messages, resumed = _resumed(world, "root", state)
    cold = _round(world, resumed, resumed_messages, 5)
    [row] = _changes(cold)
    assert f"- #{release['sequence']} mark {glob}: released by main, task contract-main" in row
    assert "reason: New measurements replace the original ones." in row and "ROOT OWN" not in row
    assert _changes(_round(world, resumed, cold, 6)) == [row]  # once


def test_my_own_change_is_stated_once_the_result_that_carried_it_is_folded_away(world, quiet_roster, monkeypatch):
    """While the result of my own memory_mark is in view it says what I changed: no row. A fold that takes it would
    leave my captured block showing a mark I released, or another work's later mark gone from view though in force:
    the same round states my change once. A mark I made myself after my view was captured never comes back."""
    from ouroboros import loop

    monkeypatch.setattr(loop, "_emit_checkpoint_event", lambda *_args, **_kwargs: None)
    messages, _cap, ctx = _request(world, "root")
    writer, alpha = _context(world, world.tasks["main"]), _start_mark(world, "ALPHA MARK: compare the compass with the drawing.")
    later = _marked(writer, node_id=world.page, scope="global", quote="eastern support",
                    text="LIVE GLOBAL: measure the bridge again.")
    messages = _round(world, ctx, messages, 2)
    [stated] = _changes(messages)
    messages += [*_own(ctx, 1, release_ids=[alpha], reason="The drawing is settled."),
                 *_own(ctx, 2, mark_id=later["mark_id"], visibility="meaning", reason="Its meaning is enough."),
                 *_own(ctx, 3, node_id=world.page, scope="global", text="ROOT OWN LATER: keep the drawing upright.")]
    messages = _round(world, ctx, messages, 3)
    assert _changes(messages) == [stated]  # my results say it while they are in view
    folded = _fold(world, ctx, messages, 4)
    assert "The drawing is settled." not in json.dumps([row for row in folded if row.get("role") == "tool"])
    store = ChronicleStore(world.memory.drive_root)
    release, view = store.records(kinds=["mark_release"])[-1], store.records(kinds=["mark_view"])[-1]
    [row] = _changes(folded)
    assert f"- #{release['sequence']} mark {alpha}: released by root, task contract-root" in row
    assert "reason: The drawing is settled." in row
    assert f"- #{view['sequence']} mark {later['mark_id']}: view changed to meaning by root, task contract-root" in row
    assert "quote: eastern support" not in row and "ROOT OWN LATER" not in row
    assert _changes(_round(world, ctx, folded, 5)) == [row]  # once


def test_the_capture_boundary_rides_beside_the_view_and_moves_no_hash(world, quiet_roster):
    """Two captures that differ only by a later journal record no section shows keep the same hashed view; each
    boundary rides beside it, outside the core hash, and the earlier one still brings a later change to its focus.
    A mark the view shows changes the view and so the hash. (Two builds' hashes differ anyway: the drive-state
    section names its build time; hence one core, hashed with each boundary.)"""
    import dataclasses

    from ouroboros import context_fit

    first, _cap, early = _request(world, "nanny")
    writer = _context(world, world.tasks["main"])
    _marked(writer, node_id=world.page, room_id=str(world.beta["chat_id"]), text="BETA ONLY: not the nanny's room.")
    _second, _cap, late = _request(world, "nanny")
    before, after = early.context_fit_plan.core, late.context_fit_plan.core
    assert before.memory_view_sequence < after.memory_view_sequence  # the journal grew between them
    assert before.memory_view_json == after.memory_view_json

    def core_sha256(core):
        evidence = SimpleNamespace(route_fp="content-contract", status="asserted", stale=False, window_tokens=1_000_000)
        return context_fit.build_context_fit_plan(None, core, world.tasks["nanny"], preferred_mode="max", route_resolver=(
            lambda *_args, **_kwargs: ({"model": "fixture", "provider": "fixture"}, evidence))).core_sha256

    later = dataclasses.replace(before, memory_view_sequence=after.memory_view_sequence)
    assert core_sha256(before) == core_sha256(later)
    glob = _marked(writer, node_id=world.page, scope="global", text="LIVE GLOBAL: measure the bridge again.")
    assert f"mark {glob['mark_id']}: marked by main" in _changes(_round(world, early, first, 2))[-1]
    _third, _cap, changed = _request(world, "nanny")
    assert changed.context_fit_plan.core.memory_view_json != after.memory_view_json
    assert core_sha256(dataclasses.replace(later, memory_view_json=changed.context_fit_plan.core.memory_view_json)) != (
        core_sha256(later))
