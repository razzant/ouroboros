"""Task/config/CLI integration; all model and Copilot transport calls are mocked."""

from __future__ import annotations

import json
import queue
import time
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import task_runtime
from ouroboros.task_results import load_task_result
from ouroboros.tools.registry import ToolContext
from tests._headless_cli_shared import _init_repo_with_file, _managed_worker_pool_available  # noqa: F401
from tests.test_copilot_acp_events import text, update

pytestmark = pytest.mark.serial  # fixture creates real, isolated Git worktrees


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    _init_repo_with_file(workspace)
    repo = tmp_path / "system"
    repo.mkdir()
    (repo / "prompts").mkdir()
    (repo / "docs").mkdir()
    (repo / "BIBLE.md").write_text("# Constitution\nBIBLE_CONTEXT_SENTINEL\n", encoding="utf-8")
    (repo / "prompts" / "SYSTEM.md").write_text("You are Ouroboros.\nSYSTEM_CONTEXT_SENTINEL\n", encoding="utf-8")
    (repo / "docs" / "ARCHITECTURE.md").write_text("# Architecture\nARCHITECTURE_CONTEXT_SENTINEL\n", encoding="utf-8")
    (repo / "docs" / "DEVELOPMENT.md").write_text("# Engineering\n", encoding="utf-8")
    data = tmp_path / "data"
    data.mkdir()
    event_queue = queue.Queue()
    task = {
        "id": "acp-task", "type": "task", "chat_id": 0,
        "text": "Fix tracked.txt and report verification.", "description": "Fix tracked.txt",
        "root_task_id": "acp-task", "delegation_role": "root",
        "workspace_root": str(workspace), "workspace_mode": "external",
        "execution_backend": "copilot_acp", "copilot_permission_policy": "workspace",
        "metadata": {}, "drive_root": str(data),
    }
    ctx = ToolContext(
        repo_dir=repo, drive_root=data, branch_dev="test", workspace_root=workspace,
        workspace_mode="external", task_id=task["id"], current_chat_id=0,
        task_metadata={}, event_queue=event_queue,
    )
    ctx.task_started_at, ctx.task_attempt = time.time(), 1
    tools = SimpleNamespace(_ctx=ctx)
    state = {"starts": 0, "closed": 0, "prompts": [], "frames": [text("Verified result"), {"result": {"stopReason": "end_turn"}}]}
    monkeypatch.setattr("shutil.which", lambda value: value)
    from ouroboros.provider_models import MODEL_PROVIDER_CREDENTIAL_KEYS
    for key in MODEL_PROVIDER_CREDENTIAL_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("OUROBOROS_TASK_BACKEND", raising=False)
    monkeypatch.setattr("ouroboros.subagent_runtime.apply_task_start_settings_or_disclose", lambda *args: None)

    class Client:
        session_id = "s"
        stderr = b""
        overflow = {"stderr": False}

        def __init__(self, spec, **kwargs):
            self.spec, self.kwargs = spec, kwargs
            state["spec"] = spec

        def __enter__(self):
            state["starts"] += 1
            return self

        def __exit__(self, *args):
            state["closed"] += 1

        def initialize(self):
            return {"protocolVersion": 1, "agentCapabilities": {}, "agentInfo": {"name": "copilot", "version": "fixture"}}

        def new_session(self):
            return {"sessionId": "s", "models": {"currentModelId": "reported-model"}}

        def prompt(self, prompt):
            state["prompts"].append(prompt)
            self.kwargs["observe"]("client", {"method": "session/prompt", "params": {"text": prompt}})
            (workspace / "tracked.txt").write_text("fixed\n", encoding="utf-8")
            for frame in state["frames"]:
                self.kwargs["check_control"]()
                self.kwargs["observe"]("agent", frame)
                yield frame

    monkeypatch.setattr(task_runtime, "CopilotACPClient", Client)
    return SimpleNamespace(task=task, ctx=ctx, tools=tools, state=state, data=data, repo=repo, workspace=workspace, event_queue=event_queue)


def run(runtime, native=None):
    return task_runtime.run_task_loop(
        native or (lambda **kwargs: pytest.fail("selected ACP runtime called the native loop")),
        task=runtime.task, tools=runtime.tools,
        messages=[{"role": "system", "content": "Full constitution and identity"}, {"role": "user", "content": "Work order tail"}],
    )


def test_keyless_task_records_unknown_usage_and_replayable_protocol(runtime):
    answer, usage, trace = run(runtime)
    assert answer == "Verified result"
    assert runtime.ctx._skip_post_task_synthesis
    assert runtime.state["starts"] == runtime.state["closed"] == 1
    assert "Full constitution and identity" in runtime.state["prompts"][0]
    assert "Work order tail" in runtime.state["prompts"][0]
    assert trace["review_decision"]["trigger"] == "external_task_runtime"
    assert usage["terminal_origin"] == "model_final"
    assert "did not run native" in usage["terminal_host_notice"]
    row = load_task_result(runtime.data, "acp-task")
    assert row["runtime_execution"]["state"] == "completed"
    assert row["runtime_execution"]["reported_model"] == "reported-model"
    assert row["runtime_execution"]["native_review"] == "not_run"
    ledger = [json.loads(line) for line in (runtime.data / "state" / "usage_attempts.jsonl").read_text().splitlines()]
    assert len(ledger) == 1
    assert ledger[0]["kind"] == "external_unmetered"
    assert ledger[0]["cost_usd"] is None and ledger[0]["cost_final"] is False
    assert ledger[0]["root_task_id"] == "acp-task"
    events = [json.loads(line) for line in (runtime.data / "logs" / "events.jsonl").read_text().splitlines()]
    protocol = [row for row in events if row["type"] == "task_runtime_protocol"]
    assert len(protocol) == 3
    from ouroboros.observability import read_blob_ref
    source = read_blob_ref(runtime.data, protocol[0]["protocol_ref"]["redacted_projection_ref"])
    assert source["frame"]["params"]["text"] == runtime.state["prompts"][0]
    operations = list(runtime.event_queue.queue)
    assert [row["phase"] for row in operations] == ["started", "finished"]


def test_native_loop_contract_is_byte_for_byte_unchanged(runtime):
    runtime.task["execution_backend"] = "native"
    runtime.task.pop("copilot_permission_policy")
    seen = {}

    def native(**kwargs):
        seen.update(kwargs)
        return "native", {}, {}

    assert run(runtime, native) == ("native", {}, {})
    assert seen["tools"] is runtime.tools
    assert "task" not in seen and "native_loop" not in seen
    assert runtime.state["starts"] == 0


def test_native_runtime_does_not_require_private_context_on_loop_adapters():
    tools = object()
    seen = {}
    result = task_runtime.run_task_loop(
        lambda **kwargs: seen.update(kwargs) or ("native", {}, {}),
        task={"execution_backend": "native"}, tools=tools, messages=[],
    )
    assert result == ("native", {}, {})
    assert seen == {"tools": tools, "messages": []}


@pytest.mark.parametrize("frames, reason", [
    ([{"result": {"stopReason": "end_turn"}}], "acp_empty_answer"),
    ([text("partial"), {"result": {"stopReason": "cancelled"}}], "acp_turn_incomplete"),
    ([update("plan", entries="invalid")], "acp_protocol_error"),
    ([text("partial")], "acp_turn_incomplete"),
])
def test_broken_external_result_never_falls_back_or_claims_success(runtime, frames, reason):
    runtime.state["frames"] = frames
    answer, usage, _ = run(runtime)
    assert "did not complete" in answer
    assert usage["execution_status"] == "infra_failed" and usage["reason_code"] == reason
    assert usage["terminal_origin"] == "host_notice"
    assert runtime.state["starts"] == runtime.state["closed"] == 1
    assert load_task_result(runtime.data, "acp-task")["runtime_execution"]["state"] == "failed"


def test_dispatched_task_cannot_be_automatically_replayed(runtime):
    run(runtime)
    original = load_task_result(runtime.data, "acp-task")["runtime_execution"]
    _, usage, _ = run(runtime)
    assert usage["reason_code"] == "acp_replay_refused"
    assert runtime.state["starts"] == 1
    assert load_task_result(runtime.data, "acp-task")["runtime_execution"] == original


def test_owner_deadline_prevents_dispatch(runtime):
    runtime.task["deadline_at"] = "2000-01-01T00:00:00Z"
    _, usage, _ = run(runtime)
    assert usage["reason_code"] == "acp_deadline"
    assert runtime.state["starts"] == 0
    assert not (runtime.data / "state" / "usage_attempts.jsonl").exists()


def test_agent_uses_existing_context_and_terminal_pipeline_without_any_model_calls(runtime, monkeypatch):
    from ouroboros.agent import Env, OuroborosAgent

    monkeypatch.setattr("ouroboros.agent.run_llm_loop", lambda **kwargs: pytest.fail("native loop was called"))
    monkeypatch.setattr("ouroboros.llm.LLMClient.chat", lambda *args, **kwargs: pytest.fail("API model was called"))
    agent = OuroborosAgent(Env(runtime.repo, runtime.data), event_queue=runtime.event_queue)
    events = agent.handle_task(runtime.task)
    row = load_task_result(runtime.data, "acp-task")
    assert row["status"] == "completed"
    assert row["result"] == "Verified result"
    assert row["runtime_execution"]["state"] == "completed"
    assert row["outcome_axes"]["objective"]["status"] == "not_evaluated"
    assert row["review_status"]["status"] == "skipped"
    assert row["review_status"]["run_count"] == 0
    assert row["unknown_unmetered"] == 1
    assert row["cost_final"] is False
    assert not row.get("root_phase_checkpoint", {}).get("post_task_synthesis")
    assert any(event["type"] == "task_done" for event in events)
    for sentinel in ("BIBLE_CONTEXT_SENTINEL", "SYSTEM_CONTEXT_SENTINEL", "ARCHITECTURE_CONTEXT_SENTINEL"):
        assert sentinel in runtime.state["prompts"][0]


def test_gateway_snapshots_runtime_options_and_rejects_unenforceable_contracts(runtime, monkeypatch):
    from ouroboros.gateway import tasks

    captured = []
    monkeypatch.setattr("supervisor.queue.enqueue_task", lambda task: captured.append(task) or task)
    monkeypatch.setattr("supervisor.queue.persist_queue_snapshot", lambda **kwargs: True)
    monkeypatch.setattr("ouroboros.config.load_settings", lambda: {"OUROBOROS_TASK_BACKEND": "copilot_acp", "OUROBOROS_COPILOT_PERMISSION_POLICY": "workspace"})
    app = Starlette(routes=[Route("/api/tasks", tasks.api_tasks_create, methods=["POST"])])
    app.state.drive_root, app.state.repo_dir = runtime.data, runtime.repo
    client = TestClient(app)
    body = {"description": "Fix the file", "workspace_root": str(runtime.workspace)}
    response = client.post("/api/tasks", json=body)
    assert response.status_code == 200, response.text
    assert captured[0]["metadata"]["execution_backend"] == "copilot_acp"
    assert captured[0]["metadata"]["copilot_permission_policy"] == "workspace"
    result = load_task_result(runtime.data, response.json()["task_id"])
    assert result["metadata"]["execution_backend"] == "copilot_acp"
    for extra in ({"disabled_tools": ["run_command"]}, {"resource_policy": {"protected_artifacts": [{"paths": ["private"]}]}}, {"metadata": {"execution_backend": "copilot_acp"}}):
        assert client.post("/api/tasks", json={**body, **extra}).status_code == 400
    assert len(captured) == 1
    # Choosing Native still admits the existing restricted-resource flow.
    assert client.post("/api/tasks", json={**body, "execution_backend": "native", "allowed_resources": {"network": False}}).status_code == 200
    assert len(captured) == 2


def test_cli_backend_and_model_flags_reach_the_same_task_gateway(runtime, monkeypatch, capsys):
    from ouroboros import cli

    seen = []
    client = SimpleNamespace(request=lambda *args: seen.append(args) or {"task_id": "created"})
    monkeypatch.setattr(cli, "_client", lambda *args, **kwargs: client)
    args = cli.build_parser().parse_args([
        "run", "--detach", "--workspace", str(runtime.workspace),
        "--backend", "copilot_acp", "--copilot-model", "selected",
        "--copilot-permissions", "workspace", "Fix it",
    ])
    assert args.func(args) == 0
    assert capsys.readouterr().out.strip() == "created"
    method, path, body = seen[0]
    assert (method, path) == ("POST", "/api/tasks")
    assert body["execution_backend"] == "copilot_acp"
    assert body["copilot_model"] == "selected"
    assert body["copilot_permission_policy"] == "workspace"


def test_queue_freezes_backend_before_settings_change_and_keeps_legacy_native(runtime, monkeypatch):
    from supervisor import queue as q
    from ouroboros.copilot_acp_policy import runtime_options

    for key, value in (("PENDING", []), ("RUNNING", {}), ("ADMISSION_RESERVATIONS", {}), ("ACCEPTANCE_FENCES", {})):
        monkeypatch.setattr(q, key, value)
    monkeypatch.setattr(q, "DRIVE_ROOT", runtime.data)
    monkeypatch.setenv("OUROBOROS_TASK_BACKEND", "copilot_acp")
    task = q.enqueue_task({"id": "queued-runtime", "type": "task", "text": "work"})
    assert task["metadata"]["execution_backend"] == "copilot_acp"
    assert not task_runtime.supports_native_task_controls(task)
    assert task_runtime.supports_native_task_controls({"id": "legacy-native", "type": "task"})
    restored = q.enqueue_task({"id": "restored-native", "type": "task"}, restoring_snapshot=True)
    assert restored["execution_backend"] == "native"
    monkeypatch.setenv("OUROBOROS_TASK_BACKEND", "native")
    assert runtime_options(task)["execution_backend"] == "copilot_acp"


def test_external_worker_crash_never_enqueues_a_retry(runtime, monkeypatch):
    from supervisor import queue as q, workers
    from tests.test_worker_crash_retry import _make_worker

    worker = _make_worker(busy_task_id="acp-task", exitcode=1)
    worker.proc.pid = 99999998
    monkeypatch.setattr(workers, "DRIVE_ROOT", runtime.data)
    monkeypatch.setattr(workers, "WORKERS", {0: worker})
    monkeypatch.setattr(workers, "RUNNING", {"acp-task": {"task": runtime.task, "attempt": 1, "started_at": time.time() - 5}})
    monkeypatch.setattr(workers, "CRASH_TS", [])
    monkeypatch.setattr(workers, "_LAST_SPAWN_TIME", 0)
    monkeypatch.setattr(workers, "respawn_worker", lambda *args, **kwargs: None)
    monkeypatch.setattr(workers, "_emit_task_done_terminal", lambda *args, **kwargs: None)
    monkeypatch.setattr(workers, "send_with_budget", lambda *args, **kwargs: None)
    monkeypatch.setattr(q, "persist_queue_snapshot", lambda **kwargs: True)
    monkeypatch.setattr(q, "enqueue_task", lambda *args, **kwargs: pytest.fail("external task retried"))
    workers.ensure_workers_healthy()
    row = load_task_result(runtime.data, "acp-task")
    assert row["status"] == "failed"
    assert row["reason_code"] == "worker_crash_external_runtime"


def test_external_live_steering_and_hurry_refuse_before_mailbox_write(runtime, monkeypatch):
    from supervisor import queue as q, steering
    from ouroboros.gateway.task_hurry import api_task_hurry

    receipts = []
    monkeypatch.setattr(q, "DRIVE_ROOT", runtime.data)
    monkeypatch.setattr(q, "RUNNING", {"acp-task": {"task": runtime.task, "attempt": 1}})
    monkeypatch.setattr(q, "PENDING", [])
    monkeypatch.setattr(q, "ACCEPTANCE_FENCES", {})
    monkeypatch.setattr("supervisor.events._emit_routing_receipt", lambda *args, **kwargs: receipts.append(kwargs))
    monkeypatch.setattr("ouroboros.owner_mailbox.write_owner_message", lambda *args, **kwargs: pytest.fail("unsupported control was accepted"))
    ctx = SimpleNamespace(DRIVE_ROOT=runtime.data, RUNNING=q.RUNNING, PENDING=[])
    steering._handle_steer_task({"target_task_id": "acp-task", "message": "new instructions", "chat_id": 0}, ctx)
    assert receipts[-1]["reason"] == "runtime_steering_unsupported"
    app = Starlette(routes=[Route("/api/tasks/{task_id}/hurry", api_task_hurry, methods=["POST"])])
    app.state.drive_root = runtime.data
    response = TestClient(app).post("/api/tasks/acp-task/hurry", json={"request_id": "hurry"})
    assert response.status_code == 409
    assert response.json()["reason_code"] == "runtime_control_unsupported_use_stop_now"
