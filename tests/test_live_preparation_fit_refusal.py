"""An early common-fitter overflow completes the admitted call's ordinary task lifecycle."""
import json
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from tests.test_call_task_lifecycle import direct_call as direct_call
from tests.test_context_minimum_view_unfit import FACTS, build_minimum_view


def _prepare_call(world, tmp_path, monkeypatch, call, shape, *, refine=None):
    """Admit the call (or a text turn) whose ordinary build is ``shape``; ``refine`` may set facts on the refusal."""
    from ouroboros import agent_task_pipeline as pipeline
    from ouroboros.context_fit import MinimumViewUnfit
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.tool_context import ToolContext
    from ouroboros.utils import append_jsonl
    from supervisor import worker_chat_lane, workers
    from tests.conversation_carrier_support import carry, lane_rows

    seen = SimpleNamespace(finished=[], errors=[], hosts=[], post_tasks=[])
    world.turn.catalog_model = SimpleNamespace(facts=FACTS)
    world.turn.publish = seen.finished.append
    monkeypatch.setattr(worker_chat_lane, "append_jsonl", append_jsonl)
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *a, **kw: seen.post_tasks.append(kw))

    def prepare(task):
        # The agent creates the durable running row before its ordinary context build; preparation
        # has installed its ToolContext but has not produced a fit plan yet.
        write_task_result(tmp_path, task["id"], "running", chat_id=1, _is_direct_chat=True)
        agent.tools._ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=task["id"],
                                      current_chat_id=1, is_direct_chat=True, task_metadata=task.get("metadata"))
        try:
            build_minimum_view(tmp_path, monkeypatch, shape, task=task)
        except MinimumViewUnfit as exc:
            if refine is not None:
                refine(exc)
            seen.errors.append(exc)
            raise
        seen.hosts.append(task["id"])  # only a successful preparation may reach the live host
        return []

    agent = SimpleNamespace(handle_task=prepare, memory=None, llm=None, tools=SimpleNamespace(_ctx=None),
                            env=SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path, drive_path=lambda name: tmp_path / name))
    monkeypatch.setattr(workers, "_get_chat_agent", lambda: agent)
    # Metadata alone does not authorize the call-only error mapping: a live call keeps this thread
    # (``executor=inline_call``); a text turn is admitted to the lane and carried.
    worker_chat_lane.handle_chat_direct(1, "Open the call.", task_metadata={"execution_transport": world.turn.request},
                                        realtime_turn=world.turn if call else None)
    [seen.task_id] = list(lane_rows(workers))
    if not call:
        if shape == "fit":
            carry(workers, world.spare, agent)
        else:
            with pytest.raises(MinimumViewUnfit):
                carry(workers, world.spare, agent)
    seen.emitted = list(world.bus.queue)
    seen.answers = [row for row in seen.emitted if row["type"] == "send_message"]
    log = tmp_path / "logs" / "events.jsonl"
    seen.rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
    return seen


@pytest.mark.parametrize("call,shape", [(True, "memory"), (False, "memory"),
                                       (True, "fixed"), (False, "fixed"), (True, "fit")])
def test_preparation_overflow_is_typed_only_for_an_admitted_call(direct_call, tmp_path, monkeypatch, call, shape):
    from ouroboros.gateway.tasks import api_task_get
    from ouroboros.host_notice import STEPS
    from ouroboros.task_results import load_task_result

    world = direct_call
    seen = _prepare_call(world, tmp_path, monkeypatch, call, shape)
    events, finished, errors, hosts, post_tasks = world.bus, seen.finished, seen.errors, seen.hosts, seen.post_tasks
    task_id, emitted, rows = seen.task_id, seen.emitted, seen.rows
    assert not world.other.closed
    if shape == "fit":
        assert hosts == [task_id] and not world.hub.closed
        assert not errors and not finished and not world.notices and events.empty()
        return
    assert not hosts and len(errors) == 1
    stored = load_task_result(tmp_path, task_id)
    if not call:
        # The generic rail: the carrier's own exception terminal and its keyed final (a system frame the
        # supervisor delivers), no call closure, no typed fit refusal.
        assert stored["status"] == "failed" and stored["reason_code"] == "task_exception"
        assert f"MinimumViewUnfit: {errors[0]}" in stored["result"]
        assert not world.hub.closed and not finished and not post_tasks
        [(notice_args, notice_kwargs)] = world.notices  # the keyed final, delivered as the supervisor would
        assert notice_kwargs["system_type"] == "task_error" and notice_kwargs["task_id"] == task_id
        [final] = [row for row in emitted if row["type"] == "send_message"]
        assert final["role"] == "system" and final["system_type"] == "task_error" and final["task_id"] == task_id
        assert final["progress_meta"]["task_terminal_status"] == "failed"
        assert final["text"] == "This turn stopped on an unexpected error."  # the phrase; the raw error rides Details
        assert final["progress_meta"]["notice"] == {"code": "task_exception",
                                                    "details": f"MinimumViewUnfit: {errors[0]}"}
        assert not [row for row in emitted if row["type"] not in {"send_message", "direct_turn_phase", "direct_turn_bound"}]
        assert not any(row["type"] == "call.fit_refused" for row in rows)
        return

    error = errors[0]
    assert world.hub.closed and world.turn._closed
    state = world.hub._state
    assert state["reason"] == "context_capacity" and state["action"] == "settings"
    assert len(finished) == 1 and finished[0]["reason"] == "context_capacity"
    assert not world.notices  # no raw-exception reply from the generic worker reporter
    answers = seen.answers
    # The ordinary outbox sends live and keeps a buffered replay with the same delivery id.
    assert answers and len({row["delivery_id"] for row in answers}) == 1
    for row in answers:  # one System line: the phrase, its step and the numbers behind Details
        notice = row["progress_meta"]["notice"]
        assert row["role"] == "system" and notice["code"] == "context_capacity" and notice["step"] == "open_calls"
        assert row["text"] == state["detail"] and "context_capacity" not in row["text"]
        for fact in ("92,000", "10,000", f"{error.estimated_input_tokens:,}", FACTS.source):
            assert fact in notice["details"], (fact, notice["details"])
        assert (f"fixed part {error.fixed_input_tokens:,}" in notice["details"]) is getattr(error, "memory_measured", True)
    done, = [row for row in emitted if row["type"] == "task_done"]
    assert done["status"] == "failed" and done["reason_code"] == "context_capacity"
    assert done["chat_id"] == 1 and done["task_id"] == task_id
    evaluation, = [row for row in rows if row["type"] == "task_eval"]
    assert evaluation["ok"] is False and evaluation["reason_code"] == "context_capacity"
    refusal, = [row for row in rows if row["type"] == "call.fit_refused"]
    assert refusal["stage"] == "task_preparation" and refusal["call_id"] == world.turn.spec.call_id
    measured = refusal["fit"]["measurement"]
    assert measured["fixed_input_tokens"] == error.fixed_input_tokens
    assert measured["memory_view_tokens"] == error.memory_view_tokens
    assert measured["estimated_input_tokens"] == error.estimated_input_tokens
    assert measured["response_reserve_tokens"] == 10_000 and measured["capacity_total_tokens"] == 92_000
    assert measured["memory_measured"] is getattr(error, "memory_measured", True)
    assert refusal["deficit_tokens"] == measured["capacity_deficit_tokens"] == error.deficit_tokens > 0
    assert stored["status"] == "failed" and stored["reason_code"] == "context_capacity"
    assert stored["outcome_axes"]["execution"]["reason_code"] == "context_capacity"
    assert stored["result"] == f"{state['detail']} {STEPS['open_calls']}"  # the record: the phrase and its step
    assert len(post_tasks) == 1  # same post-task policy as a refusal in the live host

    app = Starlette(routes=[Route("/api/tasks/{task_id}", api_task_get)])
    app.state.drive_root = tmp_path
    with TestClient(app) as client:
        response = client.get(f"/api/tasks/{task_id}")
    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert response.json()["outcome_axes"]["execution"]["reason_code"] == "context_capacity"


def test_a_refusal_carries_parts_and_says_when_memory_was_not_measured(direct_call, tmp_path, monkeypatch):
    """A refusal that priced no memory is a lower bound: it says "at least" and never prints memory as 0."""
    from ouroboros.realtime_setup import _NEEDS

    def unmeasured(exc):  # set on the instance: this lane must not depend on the fitter's own attribution
        exc.memory_measured = False
        exc.parts = (("identity", 100_000), ("system prompt", 1_234))

    # The engine as the call's opening row names it; the model id in Details.
    direct_call.turn.engine, direct_call.turn.native_model = "Synthetic · Voice", "synthetic-voice-1"
    seen = _prepare_call(direct_call, tmp_path, monkeypatch, True, "fixed", refine=unmeasured)
    assert seen.answers
    for row in seen.answers:
        notice = row["progress_meta"]["notice"]
        assert notice["template"] == _NEEDS[False] and notice["values"]["need"] == f"{seen.errors[0].estimated_input_tokens:,}"
        assert notice["values"]["engine"] == "Synthetic · Voice" and row["text"].count("Synthetic · Voice") == 1
        assert "synthetic-voice-1" in notice["details"] and "synthetic-voice-1" not in row["text"]
        assert "identity 100,000" in notice["details"] and "system prompt 1,234" in notice["details"]
        assert "memory 0" not in notice["details"] and "fixed part" not in notice["details"]
    refusal, = [row for row in seen.rows if row["type"] == "call.fit_refused"]
    assert refusal["fit"]["measurement"]["memory_measured"] is False


def test_end_pressed_during_preparation_stays_the_owners_end(direct_call, tmp_path, monkeypatch):
    """The owner's bye reached the hub before the build refused: the hub, the finished event, the room row and the
    task record carry the same ending as an End pressed while a fitting preparation ran (``call_ending``; the real
    driver's ending is pinned in ``tests/test_live_fit_refused.py``), and none names the capacity refusal."""
    from ouroboros.realtime_setup import call_ending
    from ouroboros.task_results import load_task_result

    world = direct_call
    world.hub.receive_control("owner", {"type": "bye"})
    words, usage = call_ending("finished", "ok", opened=False)
    assert words == "Call ended before it started."  # owner 26.11: no transcript for a call that never opened
    seen = _prepare_call(world, tmp_path, monkeypatch, True, "memory")
    assert world.hub.closed and world.hub._state["reason"] == usage["reason_code"]
    assert world.hub._state.get("detail") == words and not world.hub._state.get("action")  # every sheet: unstarted
    assert [event["reason"] for event in seen.finished] == [usage["reason_code"]]
    assert seen.answers and {row["text"] for row in seen.answers} == {words}
    stored = load_task_result(tmp_path, seen.task_id)
    # The task record classifies that usage as any finished call's (its outcome code is the record's own).
    assert stored["status"] not in ("failed", "infra_failed") and stored["outcome_axes"]["execution"]["status"] == "ok"
    assert stored["result"] == words
    named = json.dumps([world.hub._state, seen.finished, seen.answers, stored])
    assert "context_capacity" not in named
    refusal, = [row for row in seen.rows if row["type"] == "call.fit_refused"]  # the measurement is still logged
    assert refusal["ended_by_owner"] is True


def test_a_start_the_early_bound_refuses_reaches_the_call_as_its_notice(direct_call, tmp_path, monkeypatch):
    """Across the early bound and the call's refusal: the ordinary build's wrapper (``context.build_context_fit_plan``)
    refuses before the memory capture; the admitted call receives that refusal as ``MinimumViewUnfit`` and the room's
    one System line is the lower-bound notice with the bound's parts."""
    import sys

    from ouroboros import context
    from ouroboros.realtime_setup import _NEEDS
    from tests import _memory_scale_fixture as scale

    drive = scale.install(tmp_path / "drive")
    route = SimpleNamespace(route_fp="synthetic-route", status="confirmed", stale=False, window_tokens=25_000,
                            source="synthetic", route_facts={"working_reserve_tokens": 10_000, "basis": "estimator"})

    def early(_root, _monkeypatch, _shape, *, task):
        env, memory = scale.env_on(drive)
        context.build_context_fit_plan(env, memory, {**task, "_is_direct_chat": True}, preferred_mode="nano",
                                       tool_schemas=scale.tools(),
                                       route_resolver=lambda _task, *, allow_fetch: ({"provider": "synthetic"}, route))
        pytest.fail("the early bound admitted the start")

    monkeypatch.setattr(context, "_capture_memory_view", lambda *_a, **_kw: pytest.fail("the memory capture ran"))
    monkeypatch.setattr(sys.modules[__name__], "build_minimum_view", early)
    seen = _prepare_call(direct_call, tmp_path, monkeypatch, True, "early")
    error, = seen.errors
    assert error.memory_measured is False and error.memory_view_tokens == 0 and error.window_tokens == 25_000
    assert direct_call.hub._state["reason"] == "context_capacity" and seen.answers
    for row in seen.answers:
        notice = row["progress_meta"]["notice"]
        assert notice["template"] == _NEEDS[False] and notice["values"]["need"] == f"{error.estimated_input_tokens:,}"
        assert "memory 0" not in notice["details"] and "25,000" in notice["details"]
        assert all(f"{name} {size:,}" in notice["details"] for name, size in error.parts)
