"""A mark tied to an action meets the mind with that action's result, once per task.

The anchor is the mind's own choice; the host compares it literally with the call
and adds the mark's words to what the mind sees, as a trailing host note. Host
readers of the result never see it. A call it does not name, a released mark,
another room's mark and a second call in the same task get nothing.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from ouroboros.chronicle_store import ChronicleStore
from ouroboros.mark_anchors import anchor_problem, anchored_note, matches
from ouroboros.memory_view import _marks_text
from ouroboros.tools.chronicle import _chronicle_write, _memory_mark, _memory_read
from ouroboros.tools.registry import ToolContext
from ouroboros.tools.tool_result import ToolResult


def ctx_for(root: pathlib.Path, task_id: str = "root0001", chat_id: int = 1) -> ToolContext:
    return ToolContext(repo_dir=root, drive_root=root, task_id=task_id, current_chat_id=chat_id)


def mark(ctx, text, anchor, **fields):
    note = json.loads(_chronicle_write(ctx, kind="note", text="a lesson worth keeping"))
    return json.loads(_memory_mark(ctx, text=text, node_id=note["node_id"], anchor=anchor, **fields))


def test_anchors_match_tools_with_text_and_paths_literally():
    assert matches({"tool": "run_command"}, "run_command", {"cmd": ["ls"]})
    assert matches({"tool": "run_command", "text": "pytest"}, "run_command", {"cmd": ["python", "-m", "pytest", "-q"]})
    assert not matches({"tool": "run_command", "text": "pytest"}, "run_command", {"cmd": ["ls", "-la"]})
    assert not matches({"tool": "run_command"}, "read_file", {"path": "x"})
    for path in ("web/modules/chat.js", "/repo/web/modules/chat.js", "./web/modules/chat.js"):
        assert matches({"path": "web/modules/chat.js"}, "edit_text", {"path": path})
    assert matches({"path": "web/modules/"}, "write_file", {"path": "/repo/web/modules/settings.js"})
    assert not matches({"path": "web/modules/chat.js"}, "edit_text", {"path": "web/modules/chat.jsx"})
    assert not matches({"path": "web/modules"}, "edit_text", {"path": "web/modules_old/a.js"})
    # Paths with spaces or apostrophes are paths; file text and quoted code that mention a path are not calls naming it.
    for anchor, path in (("docs/", "docs/Release Notes.md"), ("web/modules/", "/work/My Project/web/modules/chat.js"),
                         ("src/config.py", r"C:\Users\John Smith\proj\src\config.py"),
                         ("report.md", "O'Brien/report.md"), ("docs/", "docs/Owner's Guide.md")):
        assert matches({"path": anchor}, "read_file", {"path": path}), (anchor, path)
    assert not matches({"path": "docs/"}, "write_file", {"path": "notes/a.md", "content": "see docs/ARCHITECTURE.md\nmore"})
    assert not matches({"path": "web/modules"}, "edit_text",
                       {"path": "tests/test_foo.py", "old_str": 'import x from "/repo/web/modules/settings.js"'})
    assert matches({"path": "docs/"}, "run_command", {"cmd": ["wc", "-l", "docs/CHECKLISTS.md"]})
    # A tool anchor's text is compared with the decoded argument values, never with key names.
    assert matches({"tool": "run_command", "text": "pytest -q"}, "run_command", {"cmd": ["python", "-m", "pytest", "-q"]})
    for text in ('--name="demo"', r"C:\work\demo", "line one\nline two"):
        assert matches({"tool": "run_command", "text": text}, "run_command", {"cmd": ["echo", text]})
    assert not matches({"tool": "write_file", "text": "content"}, "write_file", {"path": "a", "content": "b"})
    assert anchor_problem({"tool": "run_command", "text": "pytest"}) == ""
    assert anchor_problem({"path": "a/b"}) == "" and anchor_problem({"path": "notes/Owner's Guide.md"}) == ""
    # A path no call value could carry is refused when the mark is made, not left to never fire.
    for bad in ({}, {"text": "x"}, {"tool": ""}, {"tool": "a", "path": "b"}, {"path": ""}, "run_command",
                {"path": 'docs/"x".md'}, {"path": "docs/`x`.md"}):
        assert anchor_problem(bad)


def test_an_anchored_mark_meets_its_action_once_per_task(tmp_path):
    ctx = ctx_for(tmp_path)
    assert mark(ctx, "A quiet pytest tail hides failures: read the exit code.", {"tool": "run_command", "text": "pytest"})["ok"]
    assert anchored_note(ctx, "run_command", {"cmd": ["ls"]}) == ""
    first = anchored_note(ctx, "run_command", {"cmd": ["python", "-m", "pytest", "-q"]})
    assert first.startswith("⚠️ Mark I tied to run_command with 'pytest' (")
    assert "A quiet pytest tail hides failures" in first
    assert anchored_note(ctx, "run_command", {"cmd": ["python", "-m", "pytest"]}) == ""
    # A new task meets it again.
    assert "A quiet pytest tail" in anchored_note(ctx_for(tmp_path, task_id="root0002"), "run_command", {"cmd": ["pytest"]})


def test_released_and_foreign_room_marks_stay_silent_and_a_new_mark_acts_next_call(tmp_path):
    ctx = ctx_for(tmp_path)
    foreign = mark(ctx_for(tmp_path, task_id="other001", chat_id=2), "Only for room two.", {"path": "docs/"})
    glob = mark(ctx, "Global: docs are English.", {"path": "docs/"}, scope="global")
    assert foreign["ok"] and glob["ok"]
    seen = anchored_note(ctx, "read_file", {"path": "docs/ARCHITECTURE.md"})
    assert "Global: docs are English." in seen and "Only for room two." not in seen
    released = json.loads(_memory_mark(ctx, release_ids=[glob["mark_id"]], reason="no longer holds"))
    assert released["ok"], released
    later = ctx_for(tmp_path, task_id="root0003")
    assert anchored_note(later, "read_file", {"path": "docs/ARCHITECTURE.md"}) == ""
    # A mark written in this task acts from the next call.
    anchored_note(later, "memory_mark", {"text": "x"})
    mark(later, "Read the whole file before editing it.", {"tool": "edit_text"})
    anchored_note(later, "memory_mark", {"text": "x"})
    assert "Read the whole file before editing it." in anchored_note(later, "edit_text", {"path": "a.py"})


@pytest.mark.parametrize("moved", ["the task", "its parent"])
def test_a_task_moved_to_another_project_meets_that_room_s_marks(tmp_path, moved):
    """The cache follows the room the task's lineage is bound to: a move of the task or of
    an ancestor changes whose room marks act (real registry, binding owner and resolver)."""
    from ouroboros.dialogue_evidence import own_room_chat
    from ouroboros.projects_registry import bind_task_to_project, create_project

    project = create_project(tmp_path, "beta")
    main, beta = ctx_for(tmp_path, task_id="parent-task"), ctx_for(tmp_path, task_id="beta-task", chat_id=project["chat_id"])
    assert mark(main, "MAIN_ONLY_LESSON", {"tool": "edit_text"})["ok"]
    assert mark(beta, "BETA_ONLY_LESSON", {"tool": "edit_text"})["ok"]
    worker = ctx_for(tmp_path, task_id="child-task") if moved == "its parent" else ctx_for(tmp_path, task_id="mover-task")
    if moved == "its parent":
        worker.task_metadata = {"parent_task_id": main.task_id, "root_task_id": main.task_id}
    assert own_room_chat(worker, tmp_path) == 1
    assert anchored_note(worker, "read_file", {"path": "guide.md"}) == ""  # fills the cache while in Main
    bound = main.task_id if moved == "its parent" else worker.task_id
    bind_task_to_project(tmp_path, bound, "beta", origin={"absent": "post_hoc_unresolved"})
    assert own_room_chat(worker, tmp_path) == project["chat_id"]
    seen = anchored_note(worker, "edit_text", {"path": "guide.md"})
    assert "BETA_ONLY_LESSON" in seen and "MAIN_ONLY_LESSON" not in seen, seen


def test_a_malformed_anchor_is_refused_and_the_view_names_the_action(tmp_path):
    ctx = ctx_for(tmp_path)
    note = json.loads(_chronicle_write(ctx, kind="note", text="n"))
    refused = _memory_mark(ctx, text="x", node_id=note["node_id"], anchor={"tool": "run_command", "path": "a"})
    assert "TOOL_ARG_ERROR" in refused
    stored = ChronicleStore(tmp_path).publish([{"kind": "mark", "room_id": "1", "scope": "room", "text": "x",
                                                "author": {"kind": "helper"}, "anchor": {"path": ""},
                                                "target_ref": {"kind": "task", "task_id": "t"}}])
    assert stored.ok is False
    assert mark(ctx, "Check the exit code.", {"tool": "run_command", "text": "pytest"})["ok"]
    active = ChronicleStore(tmp_path).active_marks("1")
    assert active[-1]["anchor"] == {"tool": "run_command", "text": "pytest"}
    assert "meets me at run_command with 'pytest'" in _memory_read(ctx, room_id="1")
    shown = _marks_text([{"id": "m1", "scope": "room", "room": "Main", "text": "Check the exit code.", "by": "root",
                          "date": "2026-10-23", "target": "a note", "quote": "", "anchor": "run_command with 'pytest'"},
                         {"id": "m2", "scope": "global", "room": "Main", "text": "Plain.", "by": "root",
                          "date": "2026-10-23", "target": "a note", "quote": "", "anchor": ""}])
    assert "meets me at run_command with 'pytest'" in shown and shown.count("meets me at") == 1


def _probe_registry(tmp_path, monkeypatch, ctx):
    import ouroboros.safety as safety
    from ouroboros.tools.registry import ToolEntry, ToolRegistry

    monkeypatch.setattr(safety, "check_safety", lambda *_args, **_kwargs: (True, ""))
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(ctx)
    for name in ("fixture_probe", "fixture_other"):
        registry.register(ToolEntry(name, {"name": name, "description": "fixture",
                                           "parameters": {"type": "object", "properties": {}, "required": []}},
                                    lambda _ctx, _name=name: f"{_name} output"))
    (tmp_path / "logs").mkdir(exist_ok=True)
    return registry


def test_the_mind_sees_the_mark_while_the_host_reads_the_plain_result(tmp_path, monkeypatch):
    """Through the real seam: ``_execute_single_tool`` returns the producer's own text (status,
    trace and receipts read it), and ``process_tool_results`` adds the note to the message the
    mind receives, once per task, leaving a call it does not name alone."""
    from ouroboros.loop_tool_execution import _execute_single_tool, process_tool_results

    ctx = ctx_for(tmp_path)
    assert mark(ctx, "Probe output is partial: read the receipt.", {"tool": "fixture_probe"})["ok"]
    registry = _probe_registry(tmp_path, monkeypatch, ctx)
    rows = [_execute_single_tool(registry, {"id": f"c{i}", "function": {"name": name, "arguments": "{}"}},
                                 tmp_path / "logs") for i, name in enumerate(("fixture_other", "fixture_probe", "fixture_probe"))]
    assert [row["result"] for row in rows] == ["fixture_other output", "fixture_probe output", "fixture_probe output"]
    messages, trace = [], {"tool_calls": [], "reasoning_notes": []}
    process_tool_results(rows, messages, trace, lambda _text: None, registry)
    shown = [row["content"] for row in messages if row.get("role") == "tool"]
    assert shown[0] == "fixture_other output"
    assert shown[1].startswith("fixture_probe output\n\n⚠️ Mark I tied to fixture_probe (")
    assert "Probe output is partial: read the receipt." in shown[1]
    assert shown[2] == "fixture_probe output"
    assert trace["tool_calls"][1]["result"] == shown[1]  # review sees the view the mind saw
    # ... beside the clean producer payload, the host-note contract a trace reader parses.
    from ouroboros.artifacts import read_actor_source_bytes, task_id_for_artifacts

    noted = trace["tool_calls"][1]
    assert noted["host_annotations"] == [shown[1].removeprefix("fixture_probe output\n\n")]
    assert read_actor_source_bytes(tmp_path, task_id_for_artifacts(ctx), noted["producer_source_ref"]) == b"fixture_probe output"
    assert "producer_source_ref" not in trace["tool_calls"][0] and "producer_source_ref" not in trace["tool_calls"][2]


@pytest.mark.parametrize("anchored", [False, True], ids=["control", "anchored"])
def test_a_marked_extension_effect_keeps_its_receipt_for_result_review(tmp_path, monkeypatch, anchored):
    """A mark on an extension action reaches the mind, and result review still reads that call's
    own receipt: the effect counts as produced work and the Required floor still sees it."""
    from ouroboros.extension_surface_names import extension_surface_name
    from ouroboros.tools import review_result
    from tests.test_result_review_desktop_effects import invoke, trace_of

    monkeypatch.setattr("ouroboros.extension_loader.is_extension_live", lambda *_a, **_kw: True)
    ctx = ctx_for(tmp_path, task_id="effect01")
    (tmp_path / "logs").mkdir(exist_ok=True)
    tool = extension_surface_name("fixture_skill", "click")
    if anchored:
        assert mark(ctx, "CLICK_LESSON: look at the screen after a click.", {"tool": tool}, scope="global")["ok"]
    row = invoke(ctx, "fixture_skill", "click", lambda **_kw: json.dumps({"ok": True, "observed_effect": True,
                                                                         "action": "click Save"}), {"x": 1})
    assert ("CLICK_LESSON" in row["result"]) is anchored
    produced = review_result.produced_material(ctx)
    assert [json.loads(effect["receipt"])["action"] for effect in produced["effects"]] == ["click Save"], produced
    assert not [issue for issue in produced["issues"] if "effect receipt" in issue], produced
    assert ("CLICK_LESSON" in " ".join(produced["effects"][0].get("host_notes") or [])) is anchored
    assert review_result.required_material(ctx, trace_of(ctx), "Saved it.", produced=produced)


def test_the_note_is_measured_with_the_batch_it_rides_in(tmp_path):
    """What the mind is shown is what the delivery measured: the note is part of the candidate."""
    from types import SimpleNamespace

    from ouroboros.loop_tool_execution import process_tool_results
    from tests._tool_result_delivery_shared import measured_fit

    ctx = ctx_for(tmp_path)
    assert mark(ctx, "MEASURED_LESSON: read the exit code.", {"tool": "run_command"})["ok"]
    payload = "result line\n" * 40
    rows = [{"fn_name": "run_command", "tool_call_id": "c1", "result": payload, "is_error": False,
             "tool_args": {"cmd": ["probe"]}, "args_for_log": {"cmd": ["probe"]}, "result_meta": {},
             "tool_result": ToolResult(status="ok", code="OK", text=payload)}]
    fit, measured = measured_fit(window=200_000, reserve=1_000), []

    def recording(candidate, schemas):
        measured.append([row.get("content") for row in candidate if row.get("role") == "tool"])
        return fit(candidate, schemas)

    messages, trace = [], {"tool_calls": []}
    process_tool_results(rows, messages, trace, lambda *_a: None, SimpleNamespace(_ctx=ctx),
                         fit_candidate=recording, tool_schemas=[])
    assert "MEASURED_LESSON" in messages[-1]["content"]
    assert measured and measured[-1][-1] == messages[-1]["content"]


def test_an_anchored_publication_keeps_its_receipt_and_its_pass(tmp_path, monkeypatch):
    """A mark on submit_skill_to_hub never reaches the publication text the host parses: the
    receipt the publication veto reads is recorded and a real PR keeps its pass."""
    from types import SimpleNamespace

    from ouroboros import loop_tool_execution as execution
    from ouroboros.skill_publish_result import (
        SKILL_PUBLISH_TARGET_METADATA_KEY, apply_skill_publish_receipt_veto, serialize_skill_publish_result)

    snapshot, ruleset, repo = "a" * 64, "b" * 64, "OuroborosHub/OuroborosHub"
    payload = serialize_skill_publish_result(
        ok=True, status="pr_opened", reason_code="", skill="demo", snapshot_hash=snapshot,
        scanner={"engine": "betterleaks", "version": "1.8.1", "ruleset_sha256": ruleset}, completed_stage="pr_opened",
        completed_effects=[{"stage": "branch_created", "kind": "branch", "branch": "submit/demo-v1"},
                           {"stage": "commit_created", "kind": "commit", "commit_sha": "c" * 40},
                           {"stage": "pr_opened", "kind": "pull_request"}],
        receipt={"kind": "github_pull_request", "repository": repo, "url": f"https://github.com/{repo}/pull/17",
                 "number": 17, "skill": "demo", "snapshot_hash": snapshot, "ruleset_sha256": ruleset},
        expected_repository=repo)
    ctx = ctx_for(tmp_path, task_id="publish01")
    assert mark(ctx, "Check the license field before publishing.", {"tool": "submit_skill_to_hub"}, scope="global")["ok"]
    registry = SimpleNamespace(_ctx=ctx, CODE_TOOLS=frozenset(),
                               execute_result=lambda *_a: ToolResult(status="ok", code="OK", text=payload))
    monkeypatch.setattr(execution, "persist_call", lambda *a, **k: {})
    (tmp_path / "logs").mkdir(exist_ok=True)
    row = execution._execute_single_tool(registry, {"id": "c1", "function": {
        "name": "submit_skill_to_hub", "arguments": json.dumps({"skill": "demo"})}}, tmp_path / "logs", ctx.task_id)
    assert row["result"] == payload and "skill_publish_receipt" in row["result_meta"]
    task = {"type": "skill_publish", "metadata": {SKILL_PUBLISH_TARGET_METADATA_KEY: {"skill": "demo", "repository": repo}}}
    outcome = {"outcome_axes": {"execution": {"status": "ok", "reason_code": "final_message"},
                                "objective": {"status": "pass", "source": "result_review"}}}
    apply_skill_publish_receipt_veto(outcome, task, {"tool_calls": [{"tool": "submit_skill_to_hub", **row["result_meta"]}]})
    assert outcome["outcome_axes"]["objective"]["status"] == "pass"
    # The mark itself is still shown to the mind with that call.
    assert "Check the license field before publishing." in anchored_note(ctx, "submit_skill_to_hub", {"skill": "demo"})
