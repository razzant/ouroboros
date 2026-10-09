"""A missing child role stays missing through scheduling, execution and replay facts."""

from __future__ import annotations

import asyncio
import json
import queue
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros.task_results import load_task_result
from ouroboros.tools.registry import ToolContext
from tests._shared import configure_test_subagent
from tests.test_subagent_role_texts import _env


# These consumers patch process-wide routing/configuration and supervisor seams.
pytestmark = pytest.mark.serial

ROLE_CASES = [
    pytest.param({}, "", id="omitted"),
    pytest.param({"role": None}, "", id="legacy-null"),
    pytest.param({"role": ""}, "", id="empty"),
    pytest.param({"role": " \t\n "}, "", id="whitespace"),
    pytest.param({"role": "researcher"}, "researcher", id="explicit-researcher"),
    pytest.param({"role": "  interface reviewer  "}, "interface reviewer", id="explicit-freeform"),
]


def _request(tmp_path, monkeypatch, role_fields, *, depth=0):
    from ouroboros.tools.control import _schedule_task

    repo, drive, _ = _env(tmp_path)
    selected = configure_test_subagent(monkeypatch)
    ctx = ToolContext(
        repo_dir=repo, drive_root=drive, task_id="parent", task_depth=depth,
        current_chat_id=7, task_metadata={"root_task_id": "parent"},
    )
    result = _schedule_task(
        ctx, subagent_id=selected, objective="Inspect the interface",
        expected_output="Findings", memory_mode="empty", **role_fields,
    )
    return repo, drive, ctx, result


def _supervisor(drive):
    from ouroboros.utils import append_jsonl

    sent, pending = [], []
    return SimpleNamespace(
        DRIVE_ROOT=drive, PENDING=pending, RUNNING={}, WORKERS={0: SimpleNamespace(busy_task_id=None)}, sent=sent,
        load_state=lambda: {"owner_chat_id": 7}, save_state=lambda _state: None,
        enqueue_task=lambda task: pending.append(task) or task,
        persist_queue_snapshot=lambda **_kw: None, sort_pending=lambda: None,
        append_jsonl=append_jsonl, bridge=SimpleNamespace(push_log=lambda _row: None),
        send_with_budget=lambda chat_id, text, **kw: sent.append((chat_id, text, kw)),
    )


def _assert_role(record, role):
    assert record["role"] == role
    assert record["actor_id"] == f"subagent:{role}"
    assert record["delegation_role"] == "subagent"


@pytest.mark.parametrize("role_fields,role", ROLE_CASES)
def test_role_reaches_child_prompt_and_durable_lifecycle_without_inference(tmp_path, monkeypatch, role_fields, role):
    from ouroboros.agent import Env, OuroborosAgent
    from ouroboros.agent_task_pipeline import _store_task_result
    from ouroboros.headless import copy_child_task_result, prepare_terminal_task_files
    from ouroboros.subagent_work_order import compile_external_work_order
    from ouroboros import provider_models
    from supervisor import events

    repo, drive, parent, result = _request(tmp_path, monkeypatch, role_fields)
    assert len(parent.pending_events) == 1, result
    event = parent.pending_events[0]
    tid = event["task_id"]
    _assert_role(event, role)
    _assert_role(load_task_result(drive, tid), role)
    assert event["subagent_envelope"]["role"] == role

    supervisor = _supervisor(drive)
    events._handle_schedule_task(event, supervisor)
    assert len(supervisor.PENDING) == 1
    task = supervisor.PENDING[0]
    _assert_role(task, role)
    _assert_role(task["metadata"], role)
    _assert_role(load_task_result(drive, tid), role)
    role_block = f"[SUBAGENT ROLE]\n{role}\n"
    if role:
        assert role_block in task["text"]
    else:
        assert "[SUBAGENT ROLE]" not in task["text"]
        assert "researcher" not in task["text"]
        # A configured session's leaf consumes the canonical work order, while
        # its native nanny consumes task text. Neither invents an assignment role.
        work_order = compile_external_work_order(task)
        assert "[SUBAGENT ROLE]" not in work_order
        assert "researcher" not in work_order
    scheduled = supervisor.sent[-1]
    suffix = f" ({role})" if role else ""
    assert scheduled[1] == f"🗓️ Scheduled subagent {tid}{suffix}: Inspect the interface"
    assert scheduled[2]["progress_meta"]["subagent_role"] == role

    # Run the native context consumer without a model call or external session.
    monkeypatch.setattr(provider_models, "model_has_credentials", lambda _model: True)
    monkeypatch.setattr(OuroborosAgent, "_log_worker_boot_once", lambda self: None)
    child_drive = Path(task["child_drive_root"])
    supervisor.PENDING.remove(task)
    event_queue = queue.Queue()
    agent = OuroborosAgent(Env(repo_dir=repo, drive_root=child_drive), event_queue=event_queue)
    agent._current_chat_id = 7
    monkeypatch.setattr(agent.llm, "chat", lambda **_kw: pytest.fail("role propagation must not call a model"))
    child_context, messages, _caps = agent._prepare_task_context(task)
    assert child_context is not None
    assert child_context.task_metadata["role"] == role
    _assert_role(load_task_result(child_drive, tid), role)
    prompt = "\n".join(
        content if isinstance(content := message.get("content"), str) else "\n".join(
            block.get("text", "") for block in (content or []) if isinstance(block, dict)
        ) for message in messages
    )
    if role:
        assert role_block in prompt
    else:
        assert "[SUBAGENT ROLE]" not in prompt
    running = next(row for row in event_queue.queue if row.get("system_type") == "subagent_started")
    assert running["text"] == f"▶️ Subagent {tid} running{suffix}."
    assert running["progress_meta"]["subagent_role"] == role
    supervisor.RUNNING[tid] = {"task": task}
    events._handle_send_message(running, supervisor)

    _store_task_result(agent.env, task, "Findings delivered", {}, {}, loop_outcome={"reason_code": "completed"})
    completed = load_task_result(child_drive, tid)
    assert completed["status"] == "completed"
    _assert_role(completed, role)
    copied = copy_child_task_result(drive, task)
    _assert_role(copied, role)
    assert not prepare_terminal_task_files(drive, task)["error"]
    supervisor.RUNNING[tid] = {"task": task}
    events._handle_task_done({
        "type": "task_done", "task_id": tid, "worker_id": 0, "chat_id": 7,
        "_files_prepared_attempt": int(task.get("_attempt") or 1),
    }, supervisor)
    terminal = next(row for row in supervisor.sent if row[2].get("system_type") == "subagent_terminal_notice")
    assert terminal[1] == f"✅ Subagent {tid} completed{suffix}."
    assert terminal[2]["progress_meta"]["subagent_role"] == role
    _assert_role(load_task_result(drive, tid), role)

    # Publish the real producer notices through the durable message seam, then
    # rebuild history without the in-memory task. Empty role facts must survive
    # both hops instead of being reconstructed as a researcher on reload.
    from ouroboros.gateway.history import make_chat_history_endpoint
    from supervisor import message_bus

    monkeypatch.setattr(message_bus, "DATA_DIR", drive)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"session_id": "s", "owner_id": 7})
    monkeypatch.setattr(message_bus, "_BRIDGE", message_bus.LocalChatBridge())
    monkeypatch.setattr(message_bus, "publish_event", lambda *_args: None)
    for chat_id, text, kwargs in supervisor.sent:
        message_bus.send_with_budget(chat_id, text, **kwargs)
    endpoint = make_chat_history_endpoint(drive)
    response = asyncio.run(endpoint(SimpleNamespace(query_params={"chat_id": "7"})))
    assert response.status_code == 200
    replay = [row for row in json.loads(response.body)["messages"] if row.get("task_id") == tid]
    assert {row["system_type"] for row in replay} >= {
        "task_scheduled", "subagent_started", "subagent_terminal_notice",
    }
    notices = {"task_scheduled", "subagent_started", "subagent_terminal_notice"}
    for row in replay:
        if row.get("system_type") not in notices:
            continue  # Summary/evidence carriers do not claim child-role metadata.
        assert row["subagent_role"] == role
        assert row["parent_task_id"] == "parent"
        assert row["delegation_role"] == "subagent"

    # Exercise the parent's actual reflection assembler and capture only its
    # external inference boundary: this verifies the prompt it sends, not just
    # the intermediate child-evidence helper's result.
    from ouroboros import consolidator
    from ouroboros.post_task_synthesis import _run_reflection

    prompts = []

    def reflect(_llm, prompt, _label, **_kwargs):
        prompts.append(prompt)
        return "Observed child report.\nMEMORY_ACTIONS_JSON: []\nBACKLOG_CANDIDATES_JSON: []", {}

    monkeypatch.setattr(consolidator, "_call_consolidation_llm", reflect)
    parent_env = Env(repo_dir=repo, drive_root=drive)
    entry = _run_reflection(parent_env, agent.llm, {
        "id": "parent", "type": "task", "text": "Inspect the interface",
        "workspace_root": str(repo), "budget_drive_root": str(drive),
    }, {}, {}, {})
    assert entry["reflection"] == "Observed child report."
    assert len(prompts) == 1
    child_section = prompts[0].split("## Related child/subtask evidence\n\n", 1)[1]
    evidence, _end = json.JSONDecoder().raw_decode(child_section)
    for key in ("children_overview", "children"):
        assert len(evidence[key]) == 1
        assert evidence[key][0]["task_id"] == tid
        assert evidence[key][0]["role"] == role
    assert load_task_result(drive, tid)["role"] == role


@pytest.mark.parametrize("role_fields,role", ROLE_CASES)
def test_depth_refusal_persists_only_the_supplied_role(tmp_path, monkeypatch, role_fields, role):
    monkeypatch.setenv("OUROBOROS_MAX_SUBAGENT_DEPTH", "0")
    _repo, drive, parent, result = _request(tmp_path, monkeypatch, role_fields)
    assert "depth limit (0) exceeded" in result
    assert parent.pending_events == []
    records = list((drive / "task_results").glob("*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["reason_code"] == "subtask_depth_limit"
    _assert_role(record, role)


@pytest.mark.parametrize("role_fields,role", ROLE_CASES)
def test_final_only_history_recovers_role_without_rewriting_old_rows(tmp_path, role_fields, role):
    from ouroboros.gateway.history import make_chat_history_endpoint

    logs, results = tmp_path / "logs", tmp_path / "task_results"
    logs.mkdir()
    results.mkdir()
    history = logs / "chat.jsonl"
    history.write_text(json.dumps({
        "ts": "2026-08-21T10:00:01Z", "chat_id": 7, "direction": "out",
        "task_id": "historic-child", "text": "Retained child findings",
    }) + "\n", encoding="utf-8")
    result = results / "historic-child.json"
    result.write_text(json.dumps({
        "_schema_version": 1, "task_id": "historic-child", "status": "completed",
        "delegation_role": "subagent", "parent_task_id": "parent", "root_task_id": "parent",
        "actor_id": "historical-actor", "model": "openai/fixture-model", **role_fields,
    }), encoding="utf-8")
    originals = {path: path.read_bytes() for path in (history, result)}
    endpoint = make_chat_history_endpoint(tmp_path)
    response = asyncio.run(endpoint(SimpleNamespace(query_params={"chat_id": "7"})))
    assert response.status_code == 200
    row = next(item for item in json.loads(response.body)["messages"]
               if item.get("task_id") == "historic-child")
    assert row["subagent_role"] == role
    assert row["model"] == "openai/fixture-model"
    assert row["parent_task_id"] == "parent"
    assert {path: path.read_bytes() for path in originals} == originals


@pytest.mark.parametrize("role_fields,role", ROLE_CASES)
@pytest.mark.parametrize("unreadable", [False, True], ids=["invalid-depth", "unreadable-record"])
def test_supervisor_direct_rejection_does_not_invent_a_role(tmp_path, role_fields, role, unreadable):
    from supervisor import events

    supervisor = _supervisor(tmp_path)
    path = tmp_path / "task_results" / "child.json"
    if unreadable:
        path.parent.mkdir()
        path.write_bytes(b'{"status":"scheduled"')
    event = {
        "type": "schedule_subagent", "task_id": "child", "parent_task_id": "parent",
        "root_task_id": "parent", "delegation_role": "subagent", "depth": -1,
        "objective": "Inspect the interface", "expected_output": "Findings", **role_fields,
    }
    events._handle_schedule_task(event, supervisor)
    assert supervisor.PENDING == []
    assert supervisor.sent[-1][2]["progress_meta"]["subagent_role"] == role
    if unreadable:
        assert path.read_bytes() == b'{"status":"scheduled"'
    else:
        assert load_task_result(tmp_path, "child")["role"] == role
