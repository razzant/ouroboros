"""An early common-fitter overflow completes the admitted call's ordinary task lifecycle."""
import json
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from tests.test_call_task_lifecycle import direct_call as direct_call
from tests.test_context_minimum_view_unfit import FACTS, build_minimum_view


@pytest.mark.parametrize("shape,input_cap", [("memory", False), ("fixed", False), ("fixed", True),
                                             ("fit", False), ("fault", False)])
def test_preparation_overflow_is_typed_for_an_admitted_call(direct_call, tmp_path, monkeypatch, shape, input_cap):
    from ouroboros import agent_task_pipeline as pipeline
    from ouroboros.context_fit import MinimumViewUnfit
    from ouroboros.gateway.tasks import api_task_get
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.tools.tool_context import ToolContext
    from ouroboros.utils import append_jsonl
    from supervisor import worker_chat_lane, workers
    from tests.conversation_carrier_support import carry, lane_rows

    world = direct_call
    events, finished, errors, hosts, post_tasks = world.bus, [], [], [], []
    world.turn.catalog_model = SimpleNamespace(facts=FACTS)
    world.turn.publish = finished.append
    monkeypatch.setattr(worker_chat_lane, "append_jsonl", append_jsonl)
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *a, **kw: post_tasks.append(kw))

    def prepare(task):
        # The agent creates the durable running row before its ordinary context build; preparation
        # has installed its ToolContext but has not produced a fit plan yet.
        write_task_result(tmp_path, task["id"], "running", chat_id=1, _is_direct_chat=True)
        agent.tools._ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=task["id"],
                                      current_chat_id=1, is_direct_chat=True, task_metadata=task.get("metadata"))
        if shape == "fault":
            raise RuntimeError("synthetic preparation failure")
        try:
            build_minimum_view(tmp_path, monkeypatch, shape, task=task, input_cap=input_cap)
        except MinimumViewUnfit as exc:
            errors.append(exc)
            raise
        hosts.append(task["id"])  # only a successful preparation may reach the live host
        return []

    agent = SimpleNamespace(handle_task=prepare, memory=None, llm=None, tools=SimpleNamespace(_ctx=None),
                            env=SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path, drive_path=lambda name: tmp_path / name))
    monkeypatch.setattr(workers, "_get_chat_agent", lambda: agent)
    # Metadata alone does not authorize the call-only error mapping: a live call keeps this thread
    # (``executor=inline_call``); a text turn is admitted to the lane and carried.
    call = shape != "fault"
    worker_chat_lane.handle_chat_direct(1, "Open the call.", task_metadata={"execution_transport": world.turn.request},
                                        realtime_turn=world.turn if call else None)
    [task_id] = list(lane_rows(workers))
    if not call:
        with pytest.raises(RuntimeError, match="synthetic preparation failure"):
            carry(workers, world.spare, agent)
    assert not world.other.closed
    if shape == "fit":
        assert hosts == [task_id] and not world.hub.closed
        assert not errors and not finished and not world.notices and events.empty()
        return
    assert not hosts and len(errors) == int(call)
    stored = load_task_result(tmp_path, task_id)
    emitted = list(events.queue)
    rows = [json.loads(line) for line in (tmp_path / "logs" / "events.jsonl").read_text().splitlines()] if (
        tmp_path / "logs" / "events.jsonl").exists() else []
    if not call:
        # The generic rail: the carrier's own exception terminal and its keyed final (a system frame the
        # supervisor delivers), no call closure, no typed fit refusal.
        assert stored["status"] == "failed" and stored["reason_code"] == "task_exception"
        assert "RuntimeError: synthetic preparation failure" in stored["result"]
        assert not world.hub.closed and not finished and not post_tasks
        [(notice_args, notice_kwargs)] = world.notices  # the keyed final, delivered as the supervisor would
        assert notice_kwargs["system_type"] == "task_error" and notice_kwargs["task_id"] == task_id
        [final] = [row for row in emitted if row["type"] == "send_message"]
        assert final["role"] == "system" and final["system_type"] == "task_error" and final["task_id"] == task_id
        assert final["progress_meta"]["task_terminal_status"] == "failed" and "RuntimeError" in final["text"]
        assert not [row for row in emitted if row["type"] not in {"send_message", "direct_turn_phase", "direct_turn_bound"}]
        assert not any(row["type"] == "call.fit_refused" for row in rows)
        return

    assert world.hub.closed and world.turn._closed
    assert world.hub._state["reason"] == "context_capacity"
    detail = world.hub._state["detail"]
    label = "input limit" if input_cap else "total window"
    assert f"{label} here is 92,000 tokens" in detail and "102,000" not in detail
    assert "Calls (beta) settings" in detail and "continue by text" in detail
    closed = None
    while (frame := world.hub.next_outgoing("owner")) is not None:
        if isinstance(frame, str) and (closed_frame := json.loads(frame)).get("state") == "closed":
            assert closed is None, "there should be one close frame"
            closed = closed_frame
    assert closed["reason"] == "context_capacity" and closed["detail"] == detail
    assert len(finished) == 1 and finished[0]["reason"] == "context_capacity"
    assert not world.notices  # no raw-exception reply from the generic worker reporter
    answers = [row for row in emitted if row["type"] == "send_message"]
    # The ordinary outbox sends live and keeps a buffered replay with the same delivery id.
    assert answers and len({row["delivery_id"] for row in answers}) == 1
    assert all(row["text"] == detail and row["role"] == "system" for row in answers)
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
    assert measured["response_reserve_tokens"] == 10_000
    assert measured["capacity_total_tokens"] == (102_000 if input_cap else 92_000)
    # The event names the binding physical witness beside the frame arithmetic: the input cap
    # that refused the start, never only the total-equivalent number the fit computed in.
    assert (measured["limit_kind"], measured["limit_tokens"]) == (("input" if input_cap else "total"), 92_000)
    assert refusal["deficit_tokens"] == measured["capacity_deficit_tokens"] == error.deficit_tokens > 0
    assert stored["status"] == "failed" and stored["reason_code"] == "context_capacity"
    assert stored["outcome_axes"]["execution"]["reason_code"] == "context_capacity"
    assert len(post_tasks) == 1  # same post-task policy as a refusal in the live host

    app = Starlette(routes=[Route("/api/tasks/{task_id}", api_task_get)])
    app.state.drive_root = tmp_path
    with TestClient(app) as client:
        response = client.get(f"/api/tasks/{task_id}")
    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert response.json()["outcome_axes"]["execution"]["reason_code"] == "context_capacity"


@pytest.mark.parametrize("degraded", [False, True], ids=["worker", "inline_degraded"])
def test_text_carriers_use_the_same_typed_terminal_without_closing_the_call(direct_call, monkeypatch, degraded):
    from ouroboros import agent as agent_module
    from ouroboros.task_carrier import EXECUTOR_INLINE_DEGRADED, EXECUTOR_WORKER, carrier_executor
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.tools.tool_context import ToolContext
    from supervisor import worker_chat_lane, workers
    from tests.conversation_carrier_support import carry, lane_rows

    world = direct_call
    monkeypatch.setattr(agent_module.OuroborosAgent, "_log_worker_boot_once", lambda self: None)
    agent = agent_module.OuroborosAgent(agent_module.Env(repo_dir=world.root, drive_root=world.root), event_queue=world.bus)
    monkeypatch.setattr(agent, "handle_task", agent._handle_task_scoped)
    monkeypatch.setattr(agent_module, "run_llm_loop", lambda **_kw: pytest.fail("a refused start cannot send"))

    seen = []

    def prepare(task, _authority_refusal):
        seen.append(carrier_executor(task))
        write_task_result(world.root, task["id"], "running", chat_id=1, _is_direct_chat=True)
        agent.tools._ctx = ToolContext(repo_dir=world.root, drive_root=world.root, task_id=task["id"],
                                      current_chat_id=1, is_direct_chat=True, task_metadata=task.get("metadata"))
        build_minimum_view(world.root, monkeypatch, "fixed", task=task, input_cap=True)
        pytest.fail("the synthetic context should not fit either carrier")

    monkeypatch.setattr(agent, "_prepare_task_context", prepare)
    monkeypatch.setattr(workers, "_get_chat_agent", lambda: agent)
    if degraded:
        workers.WORKERS.pop(world.spare.wid)  # the disarmed lane now has no available carrier

    worker_chat_lane.handle_chat_direct(1, "Continue by text.")
    [task_id] = list(lane_rows(workers))
    if not degraded:
        carry(workers, world.spare, agent)
    assert seen == [EXECUTOR_INLINE_DEGRADED if degraded else EXECUTOR_WORKER]
    assert not world.hub.closed and not world.other.closed
    emitted = list(world.bus.queue)
    done, = [row for row in emitted if row["type"] == "task_done"]
    assert done["status"] == "failed" and done["reason_code"] == "context_capacity"
    answers = [row for row in emitted if row["type"] == "send_message"]
    assert answers and len({row["delivery_id"] for row in answers}) == 1
    assert all(row["role"] == "system" and row["terminal_origin"] == "host_notice" for row in answers)
    result = load_task_result(world.root, task_id)
    assert result["reason_code"] == "context_capacity" and result["status"] == "failed"
    assert "input limit here is 92,000 tokens" in result["result"] and "102,000" not in result["result"]
    assert "send the request again" in result["result"] and "Error during processing" not in result["result"]
