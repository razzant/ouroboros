"""#939: the explicit execution budget through its real producers and consumers.

Each scenario drives the production seams (gateway admission, ``assign_tasks``,
the supervisor timeout rail, the exact-Resume grant and its consumption, the crash
retry producer, the supervisor wait projection) instead of seeding the
values a producer was supposed to deliver. Clock: ``model_wait`` is the one owner.
"""
from __future__ import annotations

import copy
import queue as stdqueue
import time
from types import SimpleNamespace

from tests.worker_pool_support import affirm_ready

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros.gateway.tasks import api_tasks_create
from ouroboros.model_wait import TaskModelWait, execution_elapsed_seconds
from tests._execution_clock_seams import agent_start as _agent_start
from tests.test_owner_wait_pool import pool  # noqa: F401 -- the real pool fixture
from tests.test_swarm_host_admission import host  # noqa: F401 -- the real admission host

pytestmark = pytest.mark.serial


def _fake_clock(monkeypatch, start=None):
    clock = [time.time() if start is None else start]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    monkeypatch.setattr(time, "monotonic", lambda: clock[0] - 1_000_000)
    return clock


def _idle_worker(workers, sent, wid=0):
    workers.WORKERS[wid] = affirm_ready(SimpleNamespace(wid=wid, busy_task_id=None, reaping=False,
                                           in_q=SimpleNamespace(put=lambda row: sent.append(copy.deepcopy(row)))))


def test_gateway_queue_wait_before_dispatch_is_not_execution(host, tmp_path, monkeypatch):  # noqa: F811
    from supervisor import queue, queue_timeouts, workers

    from tests.test_project_hold_recovery import worker

    app = Starlette(routes=[Route("/api/tasks", api_tasks_create, methods=["POST"])])
    app.state.drive_root = host.root
    app.state.repo_dir = tmp_path / "isolated-repo"
    with TestClient(app) as client:
        response = client.post("/api/tasks", json={"description": "Find the crash", "execution_timeout_sec": 120})
    assert response.status_code == 200, response.text
    task_id = response.json()["task_id"]
    sent = worker(host, monkeypatch)
    clock = _fake_clock(monkeypatch)
    clock[0] += 3600  # an hour in the gateway queue, as in the CyberGym evidence
    dispatched = clock[0]
    workers.assign_tasks()
    assert [row["id"] for row in sent] == [task_id]
    meta = workers.RUNNING[task_id]
    assert meta["started_at"] == dispatched
    assert not sent[0].get("deadline_at")  # no calendar deadline was minted from the duration

    # The worker's clock and the supervisor's agree from dispatch.
    waiter = _agent_start(sent[0], host.root, monkeypatch)  # the real worker start seam
    assert waiter.execution_window_remaining() == pytest.approx(120.0)
    clock[0] += 100
    assert waiter.executed_seconds() == execution_elapsed_seconds(meta, clock[0]) == pytest.approx(100.0)
    queue_timeouts._enforce_task_timeouts_locked(workers, clock[0], 0, {})
    assert "finalization_requested_at" not in meta, "the queued hour must not be charged"
    clock[0] += 21
    queue_timeouts._enforce_task_timeouts_locked(workers, clock[0], 0, {})
    assert meta["finalization_reason"] == "absolute_ceiling"  # the existing rail, no new reason
    # The seam stops after start-up, so its owner is closed (``task_ended``); the clock fact
    # ``control_reason`` maps to ``absolute_ceiling`` is the owner's own exhausted window.
    assert waiter.execution_window_remaining() <= 0 and waiter.executed_seconds() >= 120
    assert queue.PENDING == []


def test_exact_resume_capacity_delay_is_excluded_once_on_both_clocks(tmp_path, monkeypatch):
    from ouroboros import budget_pause, owner_wait
    from supervisor.events_budget import _handle_budget_pause
    from tests._budget_pause_exact_helpers import _install_queue, _loop_ctx, _parked

    clock = _fake_clock(monkeypatch)
    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    task, row = _parked(tmp_path, monkeypatch, task_id="resume-1", extra={"execution_timeout_sec": 3600})
    executed_before = float(row["paused_at"]) - float(row["started_at"])
    clock[0] += 1000  # paused
    assert queue.resume_budget_paused_task("resume-1")["ok"]
    granted_at = clock[0]
    assert task["_budget_pause_resume"]["sleep_exclusion_since"] == granted_at
    clock[0] += 120  # no worker capacity yet: still the Pause, not execution
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    assert [row["id"] for row in sent] == ["resume-1"]
    meta = workers.RUNNING["resume-1"]
    assert meta["sleep_parked_at"] == granted_at
    assert execution_elapsed_seconds(meta, clock[0]) == pytest.approx(executed_before)

    ctx, _ = _loop_ctx(tmp_path, "resume-1")
    ctx.budget_pause_resume = sent[0]["_budget_pause_resume"]
    ctx.task_started_at = ctx.budget_pause_resume["started_at"]
    saved = budget_pause.load_budget_pause(ctx)
    waiter = _agent_start(sent[0], tmp_path, monkeypatch)
    ctx.model_wait_context, waiter.tool_context = waiter, ctx
    monkeypatch.setattr(owner_wait, "rebind_restored_route", lambda *_a, **_k: (None, "max"))
    events = []
    ctx.event_queue = SimpleNamespace(put=events.append)
    clock[0] += 5  # worker start-up before the grant is consumed
    budget_pause.resume_paused_loop(SimpleNamespace(_ctx=ctx), saved, [], {}, {}, set(), budget_remaining_usd=5)
    consumed = next(event for event in events if event.get("phase") == "consumed")
    _handle_budget_pause(consumed, workers)
    _handle_budget_pause(consumed, workers)  # a duplicate notification folds nothing twice
    assert "sleep_parked_at" not in meta
    assert meta["budget_paused_sec"] == waiter.budget_paused_sec == pytest.approx(1125.0)
    assert waiter.sleep_started_monotonic is None
    clock[0] += 50
    assert waiter.executed_seconds() == pytest.approx(execution_elapsed_seconds(meta, clock[0]))
    assert waiter.executed_seconds() == pytest.approx(executed_before + 50)
    assert waiter.execution_window_remaining() == pytest.approx(3600 - executed_before - 50)


def _stored_progress(root):
    """The progress rows the handler stored (the dialogue store holds them, not a chain file)."""
    from tests import _dialogue as dialogue_test

    return dialogue_test.log_rows(root / "logs" / "progress.jsonl")


def test_a_reviewer_wait_never_moves_the_author_clock(tmp_path):
    """Unit level: two hand-built owners (the production pair is the next test)."""
    from supervisor.task_model_wait import handle_task_model_wait

    pushed, logged = [], []
    author_events, review_events = stdqueue.Queue(), stdqueue.Queue()
    author = TaskModelWait(task={"id": "job", "_attempt": 1}, drive_root=tmp_path, event_queue=author_events,
                           worker_slot_held=True, row_mutator=lambda _wid, transform: transform({}))
    review = TaskModelWait(task={"id": "job", "_attempt": 1, "model_wait_owner_id": "review-op"},
                           drive_root=tmp_path, event_queue=review_events, worker_slot_held=False,
                           row_mutator=lambda _wid, transform: transform({}))
    meta = {"task": {"id": "job", "_attempt": 1}, "attempt": 1, "started_at": time.time() - 100}
    ctx = SimpleNamespace(RUNNING={"job": meta}, DRIVE_ROOT=tmp_path, bridge=SimpleNamespace(push_log=pushed.append),
                          append_jsonl=lambda _path, row: logged.append(row))
    author.quota_enter("a-wait", "", now=time.monotonic() - 30)
    author.quota_leave("a-wait", "", now=time.monotonic())
    author._publish({"wait_id": "a-wait", "state": "resolved", "task_attempt": 1})
    handle_task_model_wait(author_events.get_nowait(), ctx)
    author_clock = dict(meta["model_wait_quota_clock"])
    assert author_clock["elapsed_sec"] == pytest.approx(30, abs=1)
    for _ in range(3):  # a reviewer's newer revisions and longer union
        review.quota_enter("r-wait", "", now=time.monotonic() - 600)
        review.quota_leave("r-wait", "", now=time.monotonic())
        review._publish({"wait_id": "r-wait", "state": "resolved", "task_attempt": 1})
        handle_task_model_wait(review_events.get_nowait(), ctx)
    assert meta["model_wait_quota_clock"] == author_clock
    assert meta["task"]["model_waits"]["r-wait"]["state"] == "resolved"  # still visible and custodied
    assert any(row.get("wait_id") == "r-wait" for row in _stored_progress(tmp_path)) and pushed


def test_a_review_operations_own_wait_owner_never_moves_the_author_clock(tmp_path):
    """The real pair: the author's scoped owner and the ``ReviewOperation`` it launches share the
    task id and the worker event queue; only the author's revisions move the RUNNING clock."""
    from ouroboros.model_wait import current_model_wait, task_model_wait_scope
    from ouroboros.review_operation import ReviewOperation
    from ouroboros.review_records import ReviewRequest
    from ouroboros.task_results import write_task_result
    from supervisor.task_model_wait import handle_task_model_wait

    write_task_result(tmp_path, "job", "running")
    pushed, logged, events = [], [], stdqueue.Queue()
    meta = {"task": {"id": "job", "_attempt": 1}, "attempt": 1, "started_at": time.time() - 100}
    supervisor = SimpleNamespace(RUNNING={"job": meta}, DRIVE_ROOT=tmp_path,
                                 bridge=SimpleNamespace(push_log=pushed.append),
                                 append_jsonl=lambda _path, row: logged.append(row))
    with task_model_wait_scope(task={"id": "job", "_attempt": 1}, drive_root=tmp_path, event_queue=events,
                               worker_slot_held=True) as author:
        assert current_model_wait() is author
        author.tool_context = SimpleNamespace(task_metadata={})
        operation = ReviewOperation(parent=author, request=ReviewRequest(surface="task_acceptance", goal="g",
                                                                         task_id="job"),
                                    task_id="job", result_root=tmp_path)
        review = operation.wait
        assert review.owner_id.startswith("review-operation-") and review.event_queue is events
        author.quota_enter("a-wait", "", now=time.monotonic() - 30)
        author.quota_leave("a-wait", "", now=time.monotonic())
        author._publish({"wait_id": "a-wait", "state": "resolved", "task_attempt": 1})
        handle_task_model_wait(events.get_nowait(), supervisor)
        author_clock = dict(meta["model_wait_quota_clock"])
        assert author_clock["elapsed_sec"] == pytest.approx(30, abs=1)
        for _ in range(3):  # the reviewer's newer revisions and longer union, same task id and queue
            review.quota_enter("r-wait", "", now=time.monotonic() - 600)
            review.quota_leave("r-wait", "", now=time.monotonic())
            review._publish({"wait_id": "r-wait", "state": "resolved", "task_attempt": 1,
                             "model_wait_owner_id": review.owner_id})
            handle_task_model_wait(events.get_nowait(), supervisor)
    assert meta["model_wait_quota_clock"] == author_clock
    assert meta["task"]["model_waits"]["r-wait"]["model_wait_owner_id"] == review.owner_id  # still visible
    assert any(row.get("wait_id") == "r-wait" for row in _stored_progress(tmp_path)) and pushed


def test_child_inherits_the_remaining_allowance_with_its_own_queue_excluded(tmp_path, monkeypatch):
    from ouroboros import model_wait
    from ouroboros.tools.control import _schedule_task
    from ouroboros.tools.registry import ToolContext
    from supervisor.task_dispatch import build_scheduled_task_payload
    from tests._shared import configure_test_subagent

    clock = _fake_clock(monkeypatch)
    subagent_id = configure_test_subagent(monkeypatch)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id, ctx.task_depth, ctx.current_chat_id = "parent", 0, 1
    ctx.event_queue = stdqueue.Queue()
    ctx.task_metadata = {"root_task_id": "parent", "execution_timeout_sec": 300}
    task = {"id": "parent", **ctx.task_metadata}
    with model_wait.task_model_wait_scope(task=task, drive_root=tmp_path, event_queue=None,
                                         worker_slot_held=False) as owner:
        ctx.model_wait_context = owner
        clock[0] += 100
        result = _schedule_task(ctx, subagent_id=subagent_id, objective="Inspect", expected_output="Findings")
        assert "TOOL_ARG_ERROR" not in result and "TASK_LIFETIME_EXHAUSTED" not in result
        event = ctx.event_queue.get_nowait()
        assert event["execution_timeout_sec"] == pytest.approx(200)
        child = build_scheduled_task_payload({**event, "tid": event["task_id"]})
        assert child["execution_timeout_sec"] == pytest.approx(200)
        clock[0] += 500  # the child's own queue wait
        with model_wait.task_model_wait_scope(task=child, drive_root=tmp_path, event_queue=None,
                                             worker_slot_held=False) as child_owner:
            assert child_owner.execution_window_remaining() == pytest.approx(200)
        clock[0] += 300  # the parent's allowance is spent: no child is minted
        spent = _schedule_task(ctx, subagent_id=subagent_id, objective="More", expected_output="More")
        assert "TASK_LIFETIME_EXHAUSTED" in spent and ctx.event_queue.empty()


@pytest.mark.parametrize("duration,global_limit,expected", [(60, 300, 60), (600, 300, 300), (None, None, None),
                                                            (0, None, None), (90, None, 90)])
def test_one_resolver_keeps_no_default_and_takes_the_narrower(duration, global_limit, expected):
    from ouroboros.config import resolve_task_execution_timeout_sec

    task = {"execution_timeout_sec": duration}
    assert resolve_task_execution_timeout_sec(task, global_ceiling=global_limit) == expected
    assert resolve_task_execution_timeout_sec({"execution_timeout_sec": True}, global_ceiling=None) is None
    assert resolve_task_execution_timeout_sec({"execution_timeout_sec": 10 ** 1000}, global_ceiling=None) is None


def test_runtime_fact_names_its_bound_and_never_overflows(monkeypatch):
    from ouroboros import config
    from ouroboros.context_runtime_facts import task_execution_clock_fact

    monkeypatch.setattr(config, "get_task_abs_ceiling_sec", lambda: 21600)
    fact = task_execution_clock_fact({"id": "job", "execution_timeout_sec": 600, "started_at": time.time()},
                                     SimpleNamespace(task_started_at=time.time()))
    assert "task execution budget" in fact["absolute_ceiling_at_basis"]
    fact = task_execution_clock_fact({"id": "job", "started_at": time.time()}, SimpleNamespace(task_started_at=time.time()))
    assert "global ceiling" in fact["absolute_ceiling_at_basis"]
    monkeypatch.setattr(config, "get_task_abs_ceiling_sec", lambda: None)
    huge = task_execution_clock_fact({"id": "job", "execution_timeout_sec": 1e308, "started_at": 1000},
                                     SimpleNamespace(task_started_at=1000))
    assert huge["absolute_ceiling_at"] is None and huge["started_at"] is not None
