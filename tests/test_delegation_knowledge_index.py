"""A helper that started without knowledge meets the knowledge index with its first delegation.

The index rides the result of that call once per task, rendered exactly as integrating contexts
see it, and nothing sent before it changes. A focus that started with knowledge, a call that is
not an accepted act of new delegation, and every later call get nothing; a task that never
delegates sends the same bytes as before. A narrow frame cuts the index like any result, naming
its exact source and its reader.
"""
from __future__ import annotations

import copy
import dataclasses
import json
from types import SimpleNamespace

import pytest

from ouroboros import knowledge as store
from ouroboros import memory_view as mv
from ouroboros.delegate_shared import _fail, delegate_result
from ouroboros.delegation_knowledge_index import _OPENING as OPENING
from ouroboros.loop_tool_execution import process_tool_results
from ouroboros.tools.control_scheduling import _record_scheduled_subagent
from ouroboros.tools.registry import ToolContext
from ouroboros.tools.tool_result import ToolResult
from tests._tool_result_delivery_shared import measured_fit


def _plan(spec):
    """The captured start a task's loop holds: a memory view drawn by ``spec``, or none (declared input)."""
    view = mv.snapshot_json(mv.MemoryViewSnapshot(spec=spec, store_status={"state": "active"}, frontier={})) if spec else ""
    return SimpleNamespace(core=SimpleNamespace(memory_view_json=view))


def ctx_for(root, role="child", task_id="kid00001"):
    ctx = ToolContext(repo_dir=root, drive_root=root, task_id=task_id, current_chat_id=1)
    ctx.context_fit_plan = _plan(dataclasses.replace(mv.ROLE_DEFAULTS[role], room_id="1") if role else None)
    return ctx


def row(name, text, *, call="c1", args=None, error=False):
    typed = ToolResult(status="error", code="TOOL_ARG_ERROR", text=text) if error else ToolResult(
        status="ok", code="OK", text=text)
    return {"fn_name": name, "tool_call_id": call, "result": text, "is_error": error, "tool_args": dict(args or {}),
            "args_for_log": dict(args or {}), "result_meta": {}, "tool_result": typed}


def scheduled(ctx, call="c1", tid="abcd1234"):
    """An accepted schedule_subagent: its receipt, and the record the real writer leaves on the task context."""
    _record_scheduled_subagent(ctx, {"task_ids": [tid], "requested_model_lane": "auto", "objective": "probe",
                                     "role": "probe"})
    return row("schedule_subagent", f"Subagent request queued {tid}: probe (subagent_id=h, route=api_model, "
                                    "ordinary recursive API actor)", call=call)


def started(call="c1", **args):
    """A delegate_start whose run started (the producer's own typed receipt)."""
    typed = delegate_result({"status": "started", "run_id": "run-1", "route": "claude/sonnet",
                             "note": "You are the nanny and the host."})
    return {**row("delegate_start", typed.text, call=call, args=args), "tool_result": typed}


def shelf(root, topic, summary, scope="global"):
    address = store.resolve_knowledge_address(root, topic, scope)
    assert store.write_knowledge_note(address, f"---\ntitle: {topic}\nsummary: {summary}\n---\n# {topic}\nBody.\n").ok


def deliver(ctx, rows, messages=None, fit=None):
    """The real seam: ``process_tool_results`` appends what the mind is shown; returns tool rows and trace."""
    messages = [] if messages is None else messages
    trace = {"tool_calls": []}
    process_tool_results(rows, messages, trace, lambda *_a: None, SimpleNamespace(_ctx=ctx),
                         fit_candidate=fit, tool_schemas=[] if fit else None)
    return [message for message in messages if message.get("role") == "tool"], trace


@pytest.mark.parametrize("focus", ["shared child", "declared child", "nanny coordinator"])
def test_a_helper_meets_the_index_with_its_first_accepted_delegation_once(tmp_path, monkeypatch, focus):
    from ouroboros.artifacts import read_actor_source_bytes, task_id_for_artifacts
    from ouroboros.context import knowledge_navigation_sections

    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    shelf(tmp_path, "people/owner", "How my human likes reviews read")
    shelf(tmp_path, "room-plan", "Where this room stands", scope="project:demo")
    ctx = ctx_for(tmp_path, role={"shared child": "child", "declared child": None, "nanny coordinator": "nanny"}[focus])
    ctx.project_id = "demo"
    first = started() if focus == "nanny coordinator" else scheduled(ctx)
    clean = first["result"]
    [shown], trace = deliver(ctx, [first])
    sections = knowledge_navigation_sections(tmp_path, "demo")
    assert shown["content"].startswith("\n\n".join([clean, OPENING, *sections])), shown["content"]
    assert "How my human likes reviews read" in shown["content"] and "Where this room stands" in shown["content"]
    assert "## Shared understanding" not in shown["content"] and "## Project workpad" not in shown["content"]
    # The host readers keep the producer's own receipt: the retained clean payload is the receipt alone.
    noted = trace["tool_calls"][0]
    assert read_actor_source_bytes(tmp_path, task_id_for_artifacts(ctx), noted["producer_source_ref"]) == clean.encode()
    # Once per task: the next accepted delegation, in a later batch, carries nothing.
    again = started(call="c2") if focus == "nanny coordinator" else scheduled(ctx, call="c2", tid="ef567890")
    [later], _trace = deliver(ctx, [again])
    assert later["content"] == again["result"]


def test_one_batch_of_delegations_carries_one_index(tmp_path):
    shelf(tmp_path, "people/owner", "SUMMARY_ONCE")
    ctx = ctx_for(tmp_path)
    rows = [scheduled(ctx, call=f"c{i}", tid=f"0000000{i}") for i in range(3)]
    shown, _trace = deliver(ctx, rows)
    assert [OPENING in message["content"] for message in shown] == [True, False, False]
    assert [message["content"] for message in shown[1:]] == [rows[1]["result"], rows[2]["result"]]
    assert sum(message["content"].count("SUMMARY_ONCE") for message in shown) == 1


@pytest.mark.parametrize("case", ["integrator", "consciousness", "no captured start", "a leaf that does not delegate",
                                  "retry_of", "refused schedule", "refused start", "memory_mark"])
def test_nothing_without_an_accepted_new_delegation_by_a_focus_that_started_without_knowledge(tmp_path, case):
    """A focus that started with knowledge (or whose start is unknown) never gets it, a later delegation
    included; for a helper, a call that is not an accepted act of new work leaves the first act to come."""
    shelf(tmp_path, "people/owner", "INDEX_SUMMARY")
    ctx = ctx_for(tmp_path, role=case if case in ("integrator", "consciousness") else "child")
    rows = {
        "a leaf that does not delegate": [row("read_file", "file text")],
        "retry_of": [started(retry_of="inv-1")],
        "refused schedule": [row("schedule_subagent", "⚠️ TOOL_ARG_ERROR (schedule_subagent): no such subagent_id",
                                 error=True)],
        "refused start": [{**row("delegate_start", "x", error=True),
                           "tool_result": _fail("delegate_start", "route_disabled", "The route is disabled.")}],
        "memory_mark": [row("memory_mark", '{"ok": true, "mark_id": "m1"}')],
    }.get(case) or [scheduled(ctx)]
    if case == "no captured start":
        ctx.context_fit_plan = None
    shown, _trace = deliver(ctx, [dict(item) for item in rows])  # a typed result is immutable, shared
    assert [message["content"] for message in shown] == [item["result"] for item in rows]
    assert "INDEX_SUMMARY" not in json.dumps(shown)
    [later], _trace = deliver(ctx, [scheduled(ctx, call="c9", tid="99999999")])
    starts_with_knowledge = case in ("integrator", "consciousness", "no captured start")
    assert (OPENING in later["content"] and "INDEX_SUMMARY" in later["content"]) is not starts_with_knowledge


def test_a_fresh_install_gets_a_short_honest_index(tmp_path):
    ctx = ctx_for(tmp_path)
    first = scheduled(ctx)
    [shown], _trace = deliver(ctx, [first])
    added = shown["content"].removeprefix(first["result"]).split("\nPRODUCER_RESULT_SOURCE_JSON=")[0]
    assert OPENING in added and "(empty)" in added and "Archived notes: 0." in added
    assert len(added) < 600, added


def test_the_index_is_the_integrators_own_and_nothing_sent_before_it_changes(tmp_path, monkeypatch):
    """Real requests: the child's start holds no knowledge, its first delegation shows the integrator's
    own ``## Knowledge base`` section, and every byte before that result stays as it was sent. A batch
    of a task that does not deliver the index (no delegation, or a focus that started with knowledge)
    produces exactly the messages it produced without this seam."""
    from ouroboros import delegation_knowledge_index
    from ouroboros.context import build_llm_messages
    from tests._memory_view_context import section
    from tests.test_cache_optimization import _make_env_and_memory

    env, memory = _make_env_and_memory(tmp_path)
    shelf(env.drive_root, "people/owner", "REAL_REQUEST_SUMMARY")
    lineage = {"delegation_role": "subagent", "parent_task_id": "root0001", "root_task_id": "root0001"}
    child_ctx = ToolContext(repo_dir=env.repo_dir, drive_root=env.drive_root, task_id="kid00001",
                            current_chat_id=1, task_metadata=dict(lineage))
    child_task = {"id": "kid00001", "type": "task", "text": "probe", "chat_id": 1, **lineage,
                  "configured_subagent": {"id": "h", "route": {"kind": "api_model"}}}
    sent, _cap = build_llm_messages(env=env, memory=memory, task=child_task, ctx=child_ctx)
    assert "## Knowledge base" not in json.dumps(sent) and "REAL_REQUEST_SUMMARY" not in json.dumps(sent)
    main_ctx = ToolContext(repo_dir=env.repo_dir, drive_root=env.drive_root, task_id="tmain001", current_chat_id=1)
    main, _cap = build_llm_messages(env=env, memory=memory, task={"id": "tmain001", "type": "task", "text": "hi",
                                                                   "chat_id": 1}, ctx=main_ctx)
    integrator_index = section(main[0]["content"][-1]["text"], "## Knowledge base").strip()
    assert "REAL_REQUEST_SUMMARY" in integrator_index

    call = {"id": "c1", "type": "function", "function": {"name": "schedule_subagent", "arguments": "{}"}}
    messages = [*sent, {"role": "assistant", "content": "", "tool_calls": [call]}]
    before = json.dumps(messages, sort_keys=True, default=str)
    deliver(child_ctx, [scheduled(child_ctx)], messages)
    assert json.dumps(messages[:-1], sort_keys=True, default=str) == before  # appended, nothing rewritten
    assert messages[-1]["role"] == "tool" and integrator_index in messages[-1]["content"]

    def unchanged_with_and_without_the_seam(ctx, rows, prefix):
        with_seam = copy.deepcopy(prefix)
        deliver(ctx, [dict(item) for item in rows], with_seam)
        without = copy.deepcopy(prefix)
        with monkeypatch.context() as patch:
            patch.setattr(delegation_knowledge_index, "with_knowledge_index", lambda _ctx, results: results)
            deliver(ctx, [dict(item) for item in rows], without)
        assert json.dumps(with_seam, sort_keys=True, default=str) == json.dumps(without, sort_keys=True, default=str)
        assert "REAL_REQUEST_SUMMARY" not in json.dumps(with_seam[len(prefix):])

    leaf = ToolContext(repo_dir=env.repo_dir, drive_root=env.drive_root, task_id="kid00002",
                       current_chat_id=1, task_metadata=dict(lineage))
    leaf.context_fit_plan = child_ctx.context_fit_plan
    unchanged_with_and_without_the_seam(leaf, [row("read_file", "file text")], messages)
    unchanged_with_and_without_the_seam(main_ctx, [scheduled(main_ctx)], main)


def test_a_narrow_frame_cuts_the_index_with_its_source_and_reader(tmp_path):
    from ouroboros.artifacts import read_actor_source_bytes, task_id_for_artifacts

    for number in range(60):
        shelf(tmp_path, f"topics/note-{number:02d}", f"SUMMARY_{number:02d} " + "detail " * 12)
    narrow, wide = ctx_for(tmp_path, task_id="kid00001"), ctx_for(tmp_path, task_id="kid00002")
    first = scheduled(narrow)
    [cut], trace = deliver(narrow, [first], fit=measured_fit(window=6_000, reserve=500))
    content = cut["content"]
    assert content.startswith(first["result"] + "\n\n" + OPENING), content[:400]
    assert "knowledge_list" in OPENING  # the cut names the reader that reads the index again
    assert "truncated from" in content and "FULL_RESULT_SOURCE_JSON=" in content and "SUMMARY_30" not in content
    shown = trace["tool_calls"][0]
    assert shown["result_partial"] is True and shown["result_source_status"] == "ready"
    whole = read_actor_source_bytes(tmp_path, task_id_for_artifacts(narrow), shown["result_source_ref"]).decode()
    assert whole.startswith(first["result"]) and all(f"SUMMARY_{n:02d}" in whole for n in range(60))
    [kept], trace = deliver(wide, [scheduled(wide)], fit=measured_fit(window=1_000_000, reserve=500))
    assert "truncated from" not in kept["content"] and all(f"SUMMARY_{n:02d}" in kept["content"] for n in range(60))
    assert "result_partial" not in trace["tool_calls"][0]
