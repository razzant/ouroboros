"""Placement changes defaults at admission, not explicit resources or owned body work."""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_swarm_host_admission import host as host
from tests.test_project_hold_recovery import worker
from tests.test_body_candidate import body as body
from tests.body_candidate_support import make_ctx
from ouroboros.projects_registry import create_project, project_binding_for_task
from ouroboros.project_placement import append_placement_notice, move_task_to_project, placement_basis, readmit_placement
from supervisor.events_project_routing import _handle_promote_chat_to_task
from supervisor import workers

pytestmark = pytest.mark.serial
ORIGIN = {"absent": "mid_task_no_origin"}


@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("destination_has_folder", [False, True])
def test_explicit_resource_survives_real_placement_assignment(host, tmp_path, monkeypatch, pinned, destination_has_folder):
    source = tmp_path / "chosen-source"
    source.mkdir()
    destination = tmp_path / "destination"
    destination.mkdir()
    event = {"task_id": "one", "routing_token": "one-start", "objective": "Keep chosen files",
             "title": "Pinned work", "chat_id": 1}
    if pinned:
        event["workspace_root"] = str(source)
    outcome = _handle_promote_chat_to_task(event, host.ctx)
    assert outcome["status"] == "scheduled", outcome
    task = next(t for t in host.pending if t["id"] == "one")
    if pinned:
        assert task["metadata"]["resource_intent"] == {"kind": "explicit_resource", "root": str(source)}
    assert project_binding_for_task(host.root, "one") is None
    before = copy.deepcopy(task)
    create_project(host.root, "room", working_dir=str(destination) if destination_has_folder else "")
    move_task_to_project(host.root, "one", "room", origin=ORIGIN,
                        basis=placement_basis(host.root, "one", "room"), running=workers.RUNNING, pending=host.pending)
    sent = worker(host, monkeypatch)
    workers.RUNNING["occupied"] = {"task": {"id": "occupied", "project_id": "room"}}
    workers.assign_tasks()
    assert not sent and task.get("workspace_root") == before.get("workspace_root")
    workers.RUNNING.clear()
    workers.assign_tasks()
    assert [t["id"] for t in sent] == ["one"]
    expected = str(source) if pinned else str(destination) if destination_has_folder else ""
    assert sent[0].get("workspace_root", "") == expected
    assert sent[0]["task_contract"]["workspace"]["root"] == expected
    assert sent[0]["project_id"] == "room"
    assert not host.attempts


def _room(data, tmp_path):
    destination = tmp_path / "destination"
    destination.mkdir()
    create_project(data, "room", working_dir=str(destination))
    return destination


def _place(data, serving, task, monkeypatch):
    """The real move of one PENDING row and its next-boundary readmission."""
    monkeypatch.setattr("supervisor.queue.persist_queue_snapshot", lambda **kw: True)
    monkeypatch.setattr("supervisor.message_bus.get_bridge", lambda: SimpleNamespace(broadcast=lambda event: None))
    move_task_to_project(data, task["id"], "room", origin=ORIGIN, basis=placement_basis(data, task["id"], "room"),
                         running={}, pending=[task])
    assert readmit_placement(data, task, repo_dir=serving, running={}, pending=[task])


def _started(serving, data, task, monkeypatch):
    """The worker's own context builder: room lens, contract and candidate restore."""
    from ouroboros.agent import Env, OuroborosAgent

    (data / "state/state.json").write_text('{"spent_usd": 0.0}')
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))
    monkeypatch.setattr(OuroborosAgent, "_log_worker_boot_once", lambda self: None)
    monkeypatch.setattr("ouroboros.agent.build_llm_messages", lambda **_kw: ([], {}))
    agent = OuroborosAgent(Env(repo_dir=serving, drive_root=data))
    ctx, _messages, _caps = agent._prepare_task_context(copy.deepcopy(task))
    return agent, ctx


def _assert_defaults(agent, ctx, task, expected, source, monkeypatch):
    """Every default consumer selects ``expected``: file write/read, process cwd, the model's
    map and room rule, a child's captured focus and an own-body child's source."""
    from ouroboros.context import build_runtime_section
    from ouroboros.tool_access import filesystem_affordance_map
    from ouroboros.tool_access_reads import capture_parent_workspace
    from tests._shared import configure_test_subagent

    marker = f"default-{ctx.task_id}.txt"
    assert agent.tools.execute_result("write_file", {"path": marker, "content": "default\n"}).status == "ok"
    assert (expected / marker).read_text() == "default\n"
    assert "default" in agent.tools.execute_result("read_file", {"path": marker}).text
    assert str(expected) in agent.tools.execute_result("run_command", {"cmd": ["pwd"]}).text
    fs = filesystem_affordance_map(ctx)
    assert fs["root_paths"]["active_workspace"] == str(expected)
    assert ("project_room_dir" in fs) == (source == "project_room")
    assert capture_parent_workspace(ctx) == {"root": str(expected), "mode": "", "source": source}
    monkeypatch.setattr("ouroboros.config.DATA_DIR", ctx.drive_root)
    rule = build_runtime_section(agent.env, task)
    assert "Tasks promoted from this room" in rule or "active_workspace is working_dir" in rule
    assert ("active_workspace is working_dir" in rule) == (source == "project_room")
    if source != "task_drive":  # the parent's focus, or the body named by its absolute path
        from ouroboros import body_candidate

        monkeypatch.setenv("OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "true")
        selection = {} if source == "system_repo" else {"workspace_root": str(ctx.system_repo_dir)}
        scheduled = agent.tools.execute_result("schedule_subagent", {
            "subagent_id": configure_test_subagent(monkeypatch), "objective": "Body work",
            "expected_output": "Patch", "write_surface": "self_worktree", **selection})
        assert scheduled.status == "ok", scheduled.text
        assert ctx.pending_events[-1]["workspace_root"] == body_candidate.descriptor(ctx)["path"]


@pytest.mark.parametrize("kept", ["body", "no_folder"])
def test_readmitted_direct_actor_keeps_its_resource_in_every_default_consumer(body, tmp_path, monkeypatch, kept):
    """A restart dropped the actor's binding; the registry, not the queue row, holds the body."""
    from ouroboros import body_candidate
    serving, data = body
    destination = _room(data, tmp_path)
    task = {"id": "one", "type": "task", "text": "Keep authoring", "chat_id": 1, "_is_direct_chat": True,
            "metadata": {} if kept == "body" else {"resource_intent": {"kind": "explicit_none"}}}
    if kept == "body":
        candidate = Path(body_candidate.prepare(make_ctx(serving, data, "one", is_direct_chat=True))["path"])
    _place(data, serving, task, monkeypatch)
    assert task["workspace_root"] == "" and "body_candidate" not in task["metadata"]
    assert task["metadata"]["resource_intent"]["kind"] == ("system_repo" if kept == "body" else "explicit_none")
    agent, ctx = _started(serving, data, task, monkeypatch)
    assert ctx.project_id == "room" and "_project_room_dir" not in ctx.task_metadata
    if kept == "body":
        assert body_candidate.is_bound(ctx) and Path(ctx.system_repo_dir) == candidate
        _assert_defaults(agent, ctx, task, candidate, "system_repo", monkeypatch)
    else:
        _assert_defaults(agent, ctx, task, data / "task_drives/one", "task_drive", monkeypatch)
    messages = []
    assert append_placement_notice(ctx, messages)
    assert f"working folder is {ctx.active_repo_dir()}." in messages[-1]["content"]
    assert not list(destination.iterdir())


def test_ordinary_room_author_keeps_the_room_default_beside_its_candidate(body, tmp_path, monkeypatch):
    """Preparing a body in a folder room adds root=system_repo; the room stays every default."""
    from ouroboros import body_candidate
    serving, data = body
    destination = _room(data, tmp_path)
    task = {"id": "ordinary", "type": "task", "text": "Edit the room", "chat_id": 1, "_is_direct_chat": True,
            "project_id": "room", "metadata": {"project_id": "room"}}
    agent, ctx = _started(serving, data, task, monkeypatch)
    assert ctx.task_metadata["_project_room_dir"] == str(destination)
    body_write = agent.tools.execute_result("write_file", {"root": "system_repo", "path": "body.txt", "content": "b\n"})
    assert body_write.status == "ok", body_write.text
    candidate = Path(body_candidate.descriptor(ctx)["path"])
    assert body_candidate.is_bound(ctx) and (candidate / "body.txt").read_text() == "b\n"
    assert not (serving / "body.txt").exists()
    _assert_defaults(agent, ctx, task, destination, "project_room", monkeypatch)


def test_continuation_keeps_the_body_it_inherits_at_its_first_write(body, tmp_path, monkeypatch):
    """Placement before a Continue's first body write: prepare's validated predecessor decides.
    Until then (no eager transfer at start) defaults name the serving checkout, not the candidate."""
    from ouroboros import body_candidate
    from ouroboros.tool_access import filesystem_affordance_map
    from ouroboros.tool_access_reads import capture_parent_workspace
    from tests.test_body_candidate import _terminal
    serving, data = body
    candidate = Path(body_candidate.prepare(make_ctx(serving, data, "pred-1"))["path"])
    (candidate / "inherited.txt").write_text("from pred-1\n")
    _terminal(data, "pred-1", status="failed")
    destination = _room(data, tmp_path)
    source = {"kind": "task_result", "task_id": "pred-1", "tool": "get_task_result",
              "arguments": {"task_id": "pred-1", "include_authority": True}}
    task = {"id": "succ-1", "type": "task", "text": "Continue", "chat_id": 1, "root_task_id": "succ-1",
            "metadata": {}, "predecessor_authority_source": source}
    _place(data, serving, task, monkeypatch)
    assert task["workspace_root"] == "" and task["metadata"]["resource_intent"] == {"kind": "system_repo"}
    agent, ctx = _started(serving, data, task, monkeypatch)
    assert not body_candidate.is_bound(ctx) and body_candidate.find("succ-1") is None
    cwd = agent.tools.execute_result("run_command", {"cmd": ["pwd"]}).text
    assert str(serving) in cwd and str(candidate) not in cwd
    assert filesystem_affordance_map(ctx)["root_paths"]["active_workspace"] == str(serving)
    assert capture_parent_workspace(ctx) == {"root": str(serving), "mode": "", "source": "system_repo"}
    assert "NOT_FOUND" in agent.tools.execute_result("read_file", {"path": "inherited.txt"}).text
    written = agent.tools.execute_result("write_file", {"path": "continued.txt", "content": "next\n"})
    assert (candidate / "continued.txt").read_text() == "next\n", written.text
    assert body_candidate.find("succ-1")["path"] == str(candidate)
    assert "from pred-1" in agent.tools.execute_result("read_file", {"path": "inherited.txt"}).text
    assert str(candidate) in agent.tools.execute_result("run_command", {"cmd": ["pwd"]}).text
    assert not (destination / "continued.txt").exists() and not (serving / "continued.txt").exists()


def _interrupted_root(host, body, tmp_path, monkeypatch, producer, candidate):
    """A root bound to no room, ended by a restart. Main roots come from their real producers:
    owner-worded promotion, the worker's first write, the owner Restart. No producer leaves an
    owner-worded room root unbound (promotions and direct turns bind; API and scheduled roots have
    no owner words for Continue), so those are records."""
    from ouroboros import body_candidate
    from ouroboros.agent import OuroborosAgent
    from ouroboros.project_dialogue import build_owner_message_ref
    from tests.test_owner_continue import _interrupted
    from tests.test_restart_retention import _pool_events, _restart_door

    serving, data = body
    folder = tmp_path / "chosen"
    folder.mkdir()
    if producer in {"folder", "main"}:
        words = "Keep working on this"
        event = {"task_id": "pred", "routing_token": "pred-start", "objective": words, "title": "Chosen work",
                 "chat_id": 1, "source_text": words, "source_ref": build_owner_message_ref(
                     chat_id=1, client_message_id="owner-1", ts="2026-10-17T10:00:00+00:00", text=words),
                 **({"workspace_root": str(folder)} if producer == "folder" else {})}
        assert _handle_promote_chat_to_task(event, host.ctx)["status"] == "scheduled"
        [task] = host.pending
        agent = object.__new__(OuroborosAgent)
        agent.env = SimpleNamespace(drive_root=data, budget_drive_root=data)
        agent._persist_running_record(dict(task))  # the worker's first write
        host.pending.clear()
        workers.RUNNING["pred"] = {"task": dict(task), "worker_id": 0, "attempt": 1}
    elif producer == "room_default":
        create_project(data, "old", working_dir=str(folder))
        _interrupted(data, "pred", project_id="old", workspace_root=str(folder), workspace_mode="external",
                     metadata={"resource_intent": {"kind": "room_default", "project_id": "old"}})
    elif producer == "no_folder":
        _interrupted(data, "pred", project_id="scope",
                     metadata={"resource_intent": {"kind": "explicit_none", "project_id": "scope"}})
    else:  # recorded before resource intent existed (#1315)
        _interrupted(data, "pred", workspace_root=str(folder), workspace_mode="external")
    if candidate:
        body_candidate.prepare(make_ctx(serving, data, "pred"))
    if producer in {"folder", "main"}:
        monkeypatch.setattr(workers, "WORKERS", {})  # the slot is the Restart door's to kill
        _pool_events(workers, monkeypatch)
        _restart_door(data, monkeypatch, workers)
    return folder


@pytest.mark.parametrize(("producer", "candidate"), [
    ("folder", False), ("folder", True), ("main", True),
    ("no_folder", False), ("room_default", False), ("room_default", True), ("legacy", True)])
def test_placed_continue_keeps_the_resource_its_predecessor_chose(host, body, tmp_path, monkeypatch,
                                                                  producer, candidate):
    """Continue admits the successor with the predecessor's recorded choice; placing it before
    launch keeps an explicit folder or opt-out and folderless body work, moves a room default
    (candidate or not: a room author's candidate stays root=system_repo), and never relabels a
    kept folder as body work."""
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.followup import followup_resource_intent
    from supervisor.continuation_admission import admit_continuation

    data = body[1]
    folder = _interrupted_root(host, body, tmp_path, monkeypatch, producer, candidate)
    stored = load_task_result(data, "pred")
    chosen = (stored.get("metadata") or {}).get("resource_intent")
    assert (chosen or {}).get("kind") == {"folder": "explicit_resource", "main": "system_repo",
                                         "no_folder": "explicit_none", "room_default": "room_default"}.get(producer)
    assert project_binding_for_task(data, "pred") is None and stored["status"] == "cancelled"
    ack = admit_continuation("pred", action_nonce="press-0001-abcdef")
    assert ack["ok"] is True and ack["held"] is False, ack
    successor = ack["successor_task_id"]
    admitted = next(row for row in host.pending if row["id"] == successor)
    carried = [copy.deepcopy(admitted["metadata"].get("resource_intent")),
               load_task_result(data, successor)["metadata"].get("resource_intent")]
    destination = _room(data, tmp_path)
    move_task_to_project(data, successor, "room", origin=ORIGIN, basis=placement_basis(data, successor, "room"),
                         running=workers.RUNNING, pending=host.pending)
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    assert [row["id"] for row in sent] == [successor]
    expected = str({"folder": folder, "legacy": folder, "room_default": destination}.get(producer, ""))
    assert sent[0].get("workspace_root", "") == sent[0]["task_contract"]["workspace"]["root"] == expected
    assert sent[0]["project_id"] == "room"
    # Children copy the label by value; a follow-up carries it (an unlabelled one recovers from the record).
    assert sent[0]["metadata"].get("resource_intent") == chosen
    assert followup_resource_intent(SimpleNamespace(), sent[0]["metadata"], "room") == (chosen or {})
    assert carried == [chosen, chosen]  # the queue row, and the result a never-started successor hands on


def test_notice_reports_actual_boundary_folder_once_after_prior_move(host, tmp_path):
    destination = tmp_path / "destination"
    destination.mkdir()
    create_project(host.root, "room", working_dir=str(destination))
    task = {"id": "one", "metadata": {}}
    move_task_to_project(host.root, "one", "room", origin=ORIGIN,
                        basis=placement_basis(host.root, "one", "room"), running={}, pending=[task])
    ctx = SimpleNamespace(task_id="one", drive_root=host.root, budget_drive_root=host.root,
                          task_metadata={}, active_repo_dir=lambda: destination)
    messages = []
    assert append_placement_notice(ctx, messages)
    assert readmit_placement(host.root, task, repo_dir=workers.REPO_DIR, running={}, pending=[task])
    ctx.task_metadata = task["metadata"]
    assert append_placement_notice(ctx, messages)
    assert str(destination) in messages[-1]["content"]
    assert "Readmission has completed" in messages[-1]["content"]
    assert not append_placement_notice(ctx, messages)
    assert len(messages) == 2
