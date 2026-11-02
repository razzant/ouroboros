"""An early common-fitter overflow completes the admitted call's ordinary task lifecycle."""
import json
import queue
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from tests.test_call_task_lifecycle import direct_call as direct_call
from tests.test_context_minimum_view_unfit import FACTS, build_minimum_view


@pytest.mark.parametrize("call,shape", [(True, "memory"), (False, "memory"),
                                       (True, "fixed"), (False, "fixed"), (True, "fit")])
def test_preparation_overflow_is_typed_only_for_an_admitted_call(direct_call, tmp_path, monkeypatch, call, shape):
    from ouroboros import agent_task_pipeline as pipeline
    from ouroboros.context_fit import MinimumViewUnfit
    from ouroboros.gateway.tasks import api_task_get
    from ouroboros.realtime_setup import CONTEXT_CAPACITY_DETAIL
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.tools.tool_context import ToolContext
    from ouroboros.utils import append_jsonl
    from supervisor import worker_chat_lane, workers

    world = direct_call
    events, finished, errors, hosts, post_tasks = queue.Queue(), [], [], [], []
    world.turn.catalog_model = SimpleNamespace(facts=FACTS)
    world.turn.publish = finished.append
    monkeypatch.setattr(workers, "get_event_q", lambda: events)
    monkeypatch.setattr(worker_chat_lane, "append_jsonl", append_jsonl)
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *a, **kw: post_tasks.append(kw))

    def prepare(task):
        # The agent creates the durable running row before its ordinary context build.
        write_task_result(tmp_path, task["id"], "running", chat_id=1, _is_direct_chat=True)
        try:
            build_minimum_view(tmp_path, monkeypatch, shape, task=task)
        except MinimumViewUnfit as exc:
            errors.append(exc)
            raise
        hosts.append(task["id"])  # only a successful preparation may reach the live host
        return []

    agent = SimpleNamespace(handle_task=prepare, env=SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path),
                            memory=None, llm=None, tools=SimpleNamespace(_ctx=None))
    # Metadata alone does not authorize the call-only error mapping.
    admitted = worker_chat_lane._admit_chat_task(agent, 1, "Open the call.",
        task_metadata={"execution_transport": world.turn.request}, realtime_turn=world.turn if call else None)
    assert admitted is not None
    # Preparation has installed its ToolContext but has not produced a fit plan yet.
    agent.tools._ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=admitted["task"]["id"],
                                  current_chat_id=1, is_direct_chat=True, task_metadata=admitted["task"].get("metadata"))
    assert worker_chat_lane._execute_chat_task(admitted) is (shape == "fit")
    task_id = admitted["task"]["id"]
    assert not world.other.closed
    if shape == "fit":
        assert hosts == [task_id] and not world.hub.closed
        assert not errors and not finished and not world.notices and events.empty()
        return
    assert not hosts and len(errors) == 1
    stored = load_task_result(tmp_path, task_id)
    emitted = list(events.queue)
    rows = [json.loads(line) for line in (tmp_path / "logs" / "events.jsonl").read_text().splitlines()] if (
        tmp_path / "logs" / "events.jsonl").exists() else []
    if not call:
        assert world.notices[0][0][1] == f"⚠️ Error: MinimumViewUnfit: {errors[0]}"
        assert stored["status"] == "running"  # existing generic-path behavior; outside this fix
        assert not world.hub.closed and not finished and not emitted and not post_tasks
        assert not any(row["type"] == "call.fit_refused" for row in rows)
        return

    assert world.hub.closed and world.turn._closed
    assert world.hub._state["reason"] == "context_capacity"
    assert world.hub._state["detail"] == CONTEXT_CAPACITY_DETAIL
    assert len(finished) == 1 and finished[0]["reason"] == "context_capacity"
    assert not world.notices  # no raw-exception reply from the generic worker reporter
    answers = [row for row in emitted if row["type"] == "send_message"]
    # The ordinary outbox sends live and keeps a buffered replay with the same delivery id.
    assert answers and len({row["delivery_id"] for row in answers}) == 1
    assert all(row["text"] == "context_capacity: " + CONTEXT_CAPACITY_DETAIL for row in answers)
    done, = [row for row in emitted if row["type"] == "task_done"]
    assert done["status"] == "failed" and done["reason_code"] == "context_capacity"
    assert done["chat_id"] == 1 and done["task_id"] == task_id
    evaluation, = [row for row in rows if row["type"] == "task_eval"]
    assert evaluation["ok"] is False and evaluation["reason_code"] == "context_capacity"
    refusal, = [row for row in rows if row["type"] == "call.fit_refused"]
    assert refusal["stage"] == "task_preparation" and refusal["call_id"] == world.turn.spec.call_id
    measured = refusal["fit"]["measurement"]
    error = errors[0]
    assert measured["fixed_input_tokens"] == error.fixed_input_tokens
    assert measured["memory_view_tokens"] == error.memory_view_tokens
    assert measured["estimated_input_tokens"] == error.estimated_input_tokens
    assert measured["response_reserve_tokens"] == 10_000 and measured["capacity_total_tokens"] == 92_000
    assert refusal["deficit_tokens"] == measured["capacity_deficit_tokens"] == error.deficit_tokens > 0
    assert stored["status"] == "failed" and stored["reason_code"] == "context_capacity"
    assert stored["outcome_axes"]["execution"]["reason_code"] == "context_capacity"
    assert len(post_tasks) == 1  # same post-task policy as a refusal in the live host
    assert admitted["activity"].reply_handed_off

    app = Starlette(routes=[Route("/api/tasks/{task_id}", api_task_get)])
    app.state.drive_root = tmp_path
    with TestClient(app) as client:
        response = client.get(f"/api/tasks/{task_id}")
    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert response.json()["outcome_axes"]["execution"]["reason_code"] == "context_capacity"
