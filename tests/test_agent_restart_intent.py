"""#737: host acceptance closes new starts before the requester finishes."""
from __future__ import annotations

import contextlib
import copy
import json
import queue as bus
import threading
import time
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _install_queue

pytestmark = pytest.mark.serial


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    import server
    from ouroboros import server_restart
    from ouroboros.task_results import write_task_result
    from ouroboros.tools import control_runtime
    from supervisor import workspace_preparation

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(workspace_preparation, "_CURRENT", [None])
    monkeypatch.setattr(server, "_pending_restart", {})
    if hasattr(server_restart, "_AGENT_RESTART"):
        monkeypatch.setattr(server_restart, "_AGENT_RESTART", {})
        monkeypatch.setattr(server_restart, "_AGENT_RESTART_REPLY", {})
    monkeypatch.setattr(server_restart, "_restart_requested", threading.Event())
    monkeypatch.setattr(control_runtime, "run_cmd", lambda *a, **k: "a" * 40)
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **k: 10.0)
    monkeypatch.setattr(queue, "prepare_root_billing", lambda task: None)
    monkeypatch.setattr(workers, "repo_writer_admission_closed", lambda: "")
    monkeypatch.setattr(workers, "repo_writer_task_allowed", lambda task: True)
    task = {"id": "requester", "root_task_id": "requester", "type": "task", "_attempt": 1}
    workers.RUNNING["requester"] = {"task": task, "attempt": 1, "last_heartbeat_at": time.time()}
    write_task_result(tmp_path, "requester", "running", task_attempt=1)
    events = bus.Queue()
    ctx = SimpleNamespace(repo_dir=tmp_path, drive_root=tmp_path, budget_drive_root=tmp_path,
                          task_id="requester", task_attempt=1, current_task_type="task",
                          task_metadata={}, pending_events=[], event_queue=events)
    host = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING=workers.RUNNING, PENDING=workers.PENDING,
                           load_state=lambda: {}, persist_queue_snapshot=queue.persist_queue_snapshot)
    return SimpleNamespace(server=server, restart=server_restart, queue=queue, workers=workers,
                           state=state, ctx=ctx, host=host, events=events, root=tmp_path)


def accept(runtime):
    replies, done, thread = request(runtime)
    event = runtime.events.get(timeout=5)
    from ouroboros.server_liveness import drain_worker_events

    runtime.events.put(event)
    drain_worker_events(runtime.events, runtime.host, [time.monotonic(), {}, 0, None],
                        on_restart=runtime.server._handle_restart_in_supervisor)
    assert done.wait(5)
    thread.join(5)
    assert replies[0].startswith("Restart requested."), replies
    return event


def enqueue(runtime, tid, **fields):
    task = runtime.queue.enqueue_with_admission_receipt({"id": tid, "type": "task", "chat_id": 0,
        "text": tid, "root_task_id": tid, **fields})
    assert not task.get("_admission_blocked"), task
    return task


def worker(runtime, monkeypatch):
    from tests.worker_pool_support import affirm_ready

    def forbid_signal(*args, **kwargs):
        pytest.fail("fixture workers must never reach physical process termination")

    monkeypatch.setattr("supervisor.worker_pool_lifecycle.kill_worker_tree", forbid_signal)
    monkeypatch.setattr(runtime.workers, "kill_worker_tree", forbid_signal)
    delivered = []
    slot = affirm_ready(SimpleNamespace(wid=0, busy_task_id=None,
        in_q=SimpleNamespace(put=lambda task: delivered.append(copy.deepcopy(task)))))
    monkeypatch.setattr(runtime.workers, "WORKERS", {0: slot})
    monkeypatch.setattr(runtime.workers, "MAX_WORKERS", 1)
    return slot, delivered


def request(runtime):
    from ouroboros.tools.control_runtime import _request_restart

    replies = []
    done = threading.Event()

    def call():
        try:
            replies.append(_request_restart(runtime.ctx, "finish first"))
        finally:
            done.set()

    thread = threading.Thread(target=call)
    thread.start()
    return replies, done, thread


def test_tool_waits_for_host_acceptance_before_reporting_success(runtime):
    replies, done, thread = request(runtime)
    try:
        event = runtime.events.get(timeout=2)
        assert event["type"] == "restart_request" and event["phase"] == "intent"
        assert not done.is_set(), "emitting an event is not the host's acceptance"
        runtime.server._handle_restart_in_supervisor(event, runtime.host)
        assert done.wait(5)
        assert replies[0].startswith("Restart requested."), replies
        assert not runtime.server._pending_restart, "requester finalization still owns the drain clock"
    finally:
        thread.join(10)
        assert not thread.is_alive()


def test_new_root_and_main_are_durable_without_model_calls_while_child_and_mail_work(runtime, monkeypatch):
    from ouroboros.owner_mailbox import mailbox_holds
    from supervisor import worker_chat_lane

    accept(runtime)
    _, delivered = worker(runtime, monkeypatch)
    new = enqueue(runtime, "new-root")
    child = enqueue(runtime, "child", parent_task_id="requester", root_task_id="requester",
                    delegation_role="subagent", depth=1)
    assert runtime.restart.agent_restart_holds(new)
    assert not runtime.restart.agent_restart_holds(child)
    for task in ({**new, "parent_task_id": "requester"},
                 {**child, "root_task_id": "unrelated"}, {**child, "parent_task_id": "invented"}):
        assert runtime.restart.agent_restart_holds(task), "a parent label does not establish existing work"

    model_calls, notices = [], []
    monkeypatch.setattr(runtime.workers, "_get_chat_agent", lambda: model_calls.append(True))
    monkeypatch.setattr(runtime.workers, "send_with_budget", lambda *a, **k: notices.append((a, k)))
    worker_chat_lane.handle_chat_direct(7, "A new Main request", task_metadata={"client_message_id": "main-input"})
    assert model_calls == []
    main = next(task for task in runtime.queue.PENDING if task.get("text") == "A new Main request")
    snapshot = json.loads(runtime.queue.QUEUE_SNAPSHOT_PATH.read_text())
    assert {new["id"], main["id"]} <= {row["id"] for row in snapshot["pending"]}
    assert "Saved for execution" in notices[-1][0][1]
    assert worker_chat_lane.handle_wake_direct(7, "new wake", {})["reason"] == "restart_pending"
    assert model_calls == []

    # Real mailbox delivery remains independent of the new-start gate.
    from ouroboros.owner_mailbox import write_owner_message
    assert write_owner_message(runtime.root, "Finish the current part", "requester", msg_id="owner-control")
    assert mailbox_holds(runtime.root, "requester", "owner-control")
    runtime.workers.assign_tasks()
    assert [task["id"] for task in delivered] == [child["id"]]
    assert {new["id"], main["id"]} <= {task["id"] for task in runtime.queue.PENDING}


@pytest.mark.parametrize("kind", ["root", "main"])
def test_completion_event_gap_keeps_pending_work_then_returns_it_once(runtime, monkeypatch, kind):
    from ouroboros import agent_task_pipeline as pipeline
    from tests.test_restart_retention import _pool_events
    from tests.test_restart_saved_work import _ack, _stop_and_boot

    accept(runtime)
    if kind == "main":
        from supervisor.worker_chat_lane import handle_chat_direct

        monkeypatch.setattr(runtime.workers, "_get_chat_agent", lambda: pytest.fail("Main must wait for Restart"))
        monkeypatch.setattr(runtime.workers, "send_with_budget", lambda *a, **k: None)
        handle_chat_direct(7, "Main after Restart", image_data=("cGhvdG8=", "image/png"))
        queued = runtime.queue.PENDING[-1]
        assert queued.get("image_source_ref") and "image_base64" not in queued
    else:
        queued = enqueue(runtime, "later")
    _, delivered = worker(runtime, monkeypatch)
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *a, **k: None)
    final_events = []
    pipeline.emit_task_results(SimpleNamespace(drive_root=runtime.root, repo_dir=runtime.root), None, None,
        final_events, {**runtime.workers.RUNNING["requester"]["task"], "chat_id": 0, "text": "A"},
        "Finished A", {"rounds": 1}, {"tool_calls": [], "reasoning_notes": []}, 0, runtime.root / "logs", ctx=runtime.ctx)
    ready = final_events[-1]
    assert ready["type"] == "restart_request" and ready["phase"] == "ready"
    assert ready["request_id"] == runtime.ctx.pending_restart_request["request_id"]
    assert [event["type"] for event in final_events].index("task_done") < len(final_events) - 1
    runtime.workers.RUNNING.pop("requester")  # the task_done half of a bounded event batch
    runtime.server._check_pending_restart_drain(runtime.host)
    runtime.workers.assign_tasks()
    assert delivered == [] and queued in runtime.queue.PENDING

    # Production performer through return preparation; physical stop/checkout are stand-ins.
    exits = []
    monkeypatch.setattr(runtime.server, "_safe_restart_serialized", lambda *a, **k: (True, ""))
    monkeypatch.setattr(runtime.server, "_begin_task_teardown", lambda **k: ("cancelled", "restart"))
    monkeypatch.setattr(runtime.server, "_managed_update_pending_kwargs", lambda: {})
    monkeypatch.setattr(runtime.server, "_request_restart_exit", lambda: exits.append(True))
    runtime.host.safe_restart = lambda **k: (True, "")
    runtime.host.update_state = lambda f: f({})
    kills = []
    runtime.host.kill_workers = lambda **k: kills.append(k)
    runtime.server._handle_restart_in_supervisor(ready, runtime.host)
    assert exits == [True] and kills[0]["preserve_pending"] is True
    transaction = runtime.server._planned_delegate_restart_transaction_id
    assert transaction, "a queue-only return needs the same acknowledged transaction"
    _ack(runtime.root, transaction)
    monkeypatch.setattr(runtime.workers, "WORKERS", {})  # no physical workers in this fixture
    _pool_events(runtime.workers, monkeypatch)
    runtime.restart._AGENT_RESTART.clear()  # a new process starts without the old admission intent
    runtime.queue.RETURNS_NAMED.clear()
    _stop_and_boot(runtime.queue, runtime.workers)
    _, returned_calls = worker(runtime, monkeypatch)
    assert [row["id"] for row in runtime.queue.PENDING] == [queued["id"]]
    runtime.workers.assign_tasks()
    assert [row["id"] for row in returned_calls] == [queued["id"]]
    if kind == "main":
        assert returned_calls[0]["image_source_ref"] == queued["image_source_ref"]
        assert returned_calls[0]["text"] == "Main after Restart"
    assert runtime.queue.PENDING == []
    assert runtime.queue.restore_pending_from_snapshot() == 0


@pytest.mark.parametrize("status", ["failed", "cancelled", "running", "completed", "missing"])
def test_only_known_requester_failure_releases_own_hold(runtime, monkeypatch, status):
    from ouroboros.task_results import write_task_result

    intent = accept(runtime)
    enqueue(runtime, "later")
    if status != "missing":
        write_task_result(runtime.root, "requester", status, task_attempt=1)
    else:
        monkeypatch.setattr("ouroboros.task_results.load_task_result", lambda *a, **k: None)
    runtime.queue.RETURNS_NAMED["door"] = "managed_update"
    runtime.server._pending_restart.update(request_id="another-door", deadline=time.time() + 60)
    # Do not execute any physical stop during this fault case.
    monkeypatch.setattr(runtime.server, "_perform_supervisor_restart", lambda *a, **k: None)
    runtime.server._check_pending_restart_drain(runtime.host)
    assert runtime.restart.agent_restart_pending() is (status not in {"failed", "cancelled"})
    assert runtime.queue.RETURNS_NAMED == {"door": "managed_update"}
    assert runtime.server._pending_restart.get("request_id") == "another-door"
    if status in {"failed", "cancelled"}:
        runtime.server._handle_restart_in_supervisor({**intent, "phase": "ready"}, runtime.host)
        assert not runtime.restart.agent_restart_pending()


def test_lost_ack_is_explicit_and_retry_does_not_emit_again(runtime, monkeypatch):
    from ouroboros.tools import control_events, control_runtime

    monkeypatch.setattr(control_events, "_PROMOTE_CONFIRM_TIMEOUT_SEC", 0)
    first = control_runtime._request_restart(runtime.ctx, "finish first")
    assert first.startswith("RESTART_UNCONFIRMED:")
    event = runtime.events.get_nowait()
    runtime.server._handle_restart_in_supervisor(event, runtime.host)
    second = control_runtime._request_restart(runtime.ctx, "finish first")
    assert second.startswith("Restart requested.")
    assert runtime.events.empty() and not runtime.ctx.pending_events
    runtime.restart.release_agent_restart("foreign-id", "refused")
    assert runtime.restart.agent_restart_pending()


@pytest.mark.parametrize("fault", ["closed", "save_failed"])
def test_late_or_unconfirmed_intake_promises_no_execution(runtime, monkeypatch, fault):
    from supervisor import worker_chat_lane

    accept(runtime)
    calls, notices = [], []
    monkeypatch.setattr(runtime.workers, "_get_chat_agent", lambda: calls.append(True))
    monkeypatch.setattr(runtime.workers, "send_with_budget", lambda *a, **k: notices.append(a[1]))
    if fault == "closed":
        runtime.queue.name_queue_returns(runtime.queue.PENDING, "planned_restart")
    else:
        monkeypatch.setattr(runtime.queue, "persist_queue_snapshot", lambda **k: False)
    result = runtime.queue.enqueue_task({"id": "late", "type": "task", "text": "late", "chat_id": 0})
    assert result.get("_admission_blocked") == ("restart_intake_closed" if fault == "closed" else "restart_admission_unconfirmed")
    worker_chat_lane.handle_chat_direct(7, "late Main")
    assert not calls and not runtime.queue.PENDING
    assert notices and "not accepted" in notices[-1]


@pytest.mark.parametrize("boundary", ["before_marks", "after_marks"])
def test_intent_at_last_dispatch_boundary_spares_committed_handoff(runtime, monkeypatch, boundary):
    from ouroboros import owner_pause
    from supervisor import worker_assignment as assignment

    candidate = enqueue(runtime, "picked")
    slot, delivered = worker(runtime, monkeypatch)
    assert assignment._choose_candidate(runtime.queue, 10, set(), set(), list(runtime.queue.RUNNING.values()), set())[0] == 0
    slot.busy_task_id = candidate["id"]
    candidate["_launch_claim"] = {"token": "claim", "wid": slot.wid}
    event = {"type": "restart_request", "phase": "intent", "request_id": "boundary-proof",
             "task_id": "requester", "task_attempt": 1, "reason": "finish first"}
    if boundary == "before_marks":
        original = owner_pause.launch_admission

        @contextlib.contextmanager
        def launch(*args, **kwargs):
            with original(*args, **kwargs):
                runtime.server._handle_restart_in_supervisor(event, runtime.host)
                yield

        monkeypatch.setattr(owner_pause, "launch_admission", launch)
    else:
        original = assignment._mirror_assigned_running_status

        def mirror(task):
            original(task)
            runtime.server._handle_restart_in_supervisor(event, runtime.host)
            assert "picked" in runtime.server._live_running_task_ids(runtime.host)

        monkeypatch.setattr(assignment, "_mirror_assigned_running_status", mirror)
    outcome = assignment._launch_attempt(runtime.queue, candidate, slot, "claim", lane=True)
    assert outcome is (boundary == "after_marks")
    assert [task["id"] for task in delivered] == ([candidate["id"]] if outcome else [])
    assert candidate["admitted_dispatch"] == ("possible" if outcome else "none")


def test_ack_cannot_overtake_a_mark_write_already_in_flight(runtime, monkeypatch):
    from supervisor import task_admission
    from supervisor import worker_assignment as assignment

    candidate = enqueue(runtime, "committing")
    slot, delivered = worker(runtime, monkeypatch)
    slot.busy_task_id = candidate["id"]
    candidate["_launch_claim"] = {"token": "claim", "wid": slot.wid}
    entered, proceed = threading.Event(), threading.Event()
    original = task_admission.commit_dispatch_marks

    def commit(*a, **k):
        result = original(*a, **k)
        entered.set()
        assert proceed.wait(10)
        return result

    monkeypatch.setattr(task_admission, "commit_dispatch_marks", commit)
    launch = threading.Thread(target=lambda: assignment._launch_attempt(runtime.queue, candidate, slot, "claim", lane=True))
    launch.start()
    assert entered.wait(10)
    replies, done, requester = request(runtime)
    try:
        event = runtime.events.get(timeout=5)
        runtime.server._handle_restart_in_supervisor(event, runtime.host)
        assert not done.is_set() and runtime.restart._AGENT_RESTART["status"] == "awaiting"
        proceed.set()
        launch.join(10)
        assert not launch.is_alive() and [row["id"] for row in delivered] == [candidate["id"]]
        runtime.server._check_pending_restart_drain(runtime.host)
        assert done.wait(5) and replies[0].startswith("Restart requested.")
        assert not runtime.server._pending_restart
    finally:
        proceed.set()
        launch.join(10)
        requester.join(10)


def test_refused_restart_releases_intent_without_reopening_other_doors(runtime, monkeypatch):
    accept(runtime)
    runtime.queue.RETURNS_NAMED["door"] = "managed_update"
    monkeypatch.setattr(runtime.server, "_safe_restart_serialized", lambda *a, **k: (False, "update in progress"))
    runtime.host.safe_restart = lambda **k: pytest.fail("checkout must not run")
    runtime.server._perform_supervisor_restart(runtime.host)
    assert not runtime.restart.agent_restart_pending()
    assert runtime.queue.RETURNS_NAMED == {"door": "managed_update"}


def test_main_preparation_is_not_permission_to_begin_after_intent(runtime, monkeypatch):
    from supervisor import worker_chat_lane as chat

    calls, notices = [], []
    agent = SimpleNamespace(handle_task=lambda task: calls.append(task))
    monkeypatch.setattr(runtime.workers, "_get_chat_agent", lambda: agent)
    monkeypatch.setattr(runtime.workers, "send_with_budget", lambda *a, **k: notices.append((a, k)))
    admitted = chat._admit_chat_task(agent, 7, "prepared Main")
    assert admitted is not None and not calls
    accept(runtime)
    assert chat._execute_chat_task(admitted) is False
    assert not calls and admitted["registry"].get(admitted["task"]["id"]) is None
    assert any(row["text"] == "prepared Main" for row in runtime.queue.PENDING)
    assert "task_terminal_status" not in notices[-1][1].get("progress_meta", {})


def test_direct_requester_must_match_its_live_attempt(runtime):
    from supervisor.active_activity import get_direct_activity_registry

    registry = get_direct_activity_registry()
    runtime.workers.RUNNING.clear()
    actor = SimpleNamespace(tools=SimpleNamespace(_ctx=runtime.ctx))
    entry = registry.register("requester", 7, actor=actor)
    entry.execution.advance("started")
    try:
        accept(runtime)
        assert not runtime.restart.agent_restart_holds({"id": "requester"})
    finally:
        registry.unregister("requester")


def test_retry_labels_grant_nothing_but_an_exact_existing_resume_keeps_its_authority(runtime, monkeypatch):
    from ouroboros import budget_pause
    from tests._budget_pause_exact_helpers import _parked

    forged = enqueue(runtime, "forged-retry", _attempt=2, timeout_retry_from="requester")
    parked, _ = _parked(runtime.root, monkeypatch, task_id="existing-paused")
    accept(runtime)
    assert runtime.restart.agent_restart_holds(forged)
    assert runtime.queue.resume_budget_paused_task(parked["id"])["ok"]
    assert not runtime.restart.agent_restart_holds(parked)
    row = budget_pause.budget_pause_row(runtime.root, parked["id"])
    budget_pause.set_budget_pause(runtime.root, parked["id"], {**row, "grant": {**row["grant"], "revoked_at": "now"}})
    assert runtime.restart.agent_restart_holds(parked), "intent grants no replacement for a revoked Resume"


def test_no_agent_intent_leaves_other_doors_with_their_existing_policies(runtime, monkeypatch):
    from supervisor import worker_chat_lane as chat

    runtime.queue.RETURNS_NAMED["door"] = "managed_update"
    admitted = enqueue(runtime, "existing-update-policy")
    assert admitted in runtime.queue.PENDING
    actor = SimpleNamespace()
    monkeypatch.setattr(runtime.workers, "_get_chat_agent", lambda: actor)
    turn = chat._admit_chat_task(actor, 7, "ordinary conversation during update")
    assert turn is not None, "Update's own conversation gate remains the admission authority"
    turn["registry"].unregister(turn["task"]["id"])


@pytest.mark.parametrize("fault", ["none", "closed", "save_failed"])
def test_task_http_consumer_reports_only_durable_admission(runtime, monkeypatch, fault):
    from starlette.testclient import TestClient

    from tests.test_ingress_chat_routing import _app

    accept(runtime)
    _, delivered = worker(runtime, monkeypatch)
    memory = runtime.root / "memory"
    memory.mkdir(exist_ok=True)
    (memory / "identity.md").write_text("fixture identity")
    monkeypatch.setattr("ouroboros.workspace_admission.bootstrap_process_path", lambda: [])
    if fault == "closed":
        runtime.queue.name_queue_returns(runtime.queue.PENDING, "planned_restart")
    elif fault == "save_failed":
        monkeypatch.setattr(runtime.queue, "persist_queue_snapshot", lambda **k: False)
    with TestClient(_app(runtime.root, runtime.root / "repo")) as client:
        result = client.post("/api/tasks", json={"description": "HTTP request B", "source": "web", "chat_id": 1})
    assert result.status_code == (200 if fault == "none" else 409), result.text
    if fault == "none":
        assert result.json()["status"] == "scheduled"
        assert any(row["id"] == result.json()["task_id"] for row in runtime.queue.PENDING)
        runtime.workers.assign_tasks()
        assert not delivered
    else:
        assert result.json()["status"] == "failed" and not runtime.queue.PENDING
        assert result.json()["admission"]["reason_code"] == (
            "restart_intake_closed" if fault == "closed" else "restart_admission_unconfirmed")


def test_unpublished_ack_and_another_attempt_never_claim_success(runtime, monkeypatch):
    from ouroboros.tools import control_events, control_runtime

    monkeypatch.setattr(control_events, "_PROMOTE_CONFIRM_TIMEOUT_SEC", 0)
    assert control_runtime._request_restart(runtime.ctx, "same").startswith("RESTART_UNCONFIRMED:")
    event = runtime.events.get_nowait()
    publish = runtime.queue.persist_queue_snapshot
    monkeypatch.setattr(runtime.queue, "persist_queue_snapshot", lambda **k: False)
    runtime.server._handle_restart_in_supervisor(event, runtime.host)
    assert runtime.restart.agent_restart_pending()
    assert control_runtime._request_restart(runtime.ctx, "same").startswith("RESTART_UNCONFIRMED:")
    assert runtime.events.empty()
    monkeypatch.setattr(runtime.queue, "persist_queue_snapshot", publish)
    publish(reason="later_host_publication")
    assert control_runtime._request_restart(runtime.ctx, "same").startswith("Restart requested.")
    runtime.server._handle_restart_in_supervisor({**event, "phase": "ready", "task_attempt": 2}, runtime.host)
    assert not runtime.server._pending_restart and runtime.restart._AGENT_RESTART["status"] == "accepted"


@pytest.mark.timeout(20)
@pytest.mark.parametrize("transport", ["live", "absent", "full", "broken"])
def test_parallel_restart_calls_emit_once_even_on_deferred_transport(runtime, monkeypatch, transport):
    from concurrent.futures import ThreadPoolExecutor

    from ouroboros.tools import control_events

    monkeypatch.setattr(control_events, "_PROMOTE_CONFIRM_TIMEOUT_SEC", 0)
    if transport == "absent":
        runtime.ctx.event_queue = None
    elif transport in {"full", "broken"}:
        def refuse(_event):
            raise bus.Full() if transport == "full" else OSError("queue closed")

        runtime.ctx.event_queue = SimpleNamespace(put_nowait=refuse)
    start = threading.Barrier(4)

    def call(_index):
        start.wait(timeout=5)
        return control_events.request_restart_intent(runtime.ctx, "finish first")

    with ThreadPoolExecutor(max_workers=4) as callers:
        replies = list(callers.map(call, range(4)))
    event = runtime.events.get_nowait() if transport == "live" else runtime.ctx.pending_events.pop()
    assert {reply["request_id"] for reply in replies} == {event["request_id"]}
    assert {reply["status"] for reply in replies} == {"unconfirmed"}
    assert runtime.events.empty() and not runtime.ctx.pending_events
    runtime.server._handle_restart_in_supervisor(event, runtime.host)
    assert control_events.request_restart_intent(runtime.ctx, "ignored duplicate")["status"] == "accepted"
    assert runtime.events.empty() and not runtime.ctx.pending_events
    assert runtime.ctx.pending_restart_request["reason"] == "finish first"


def test_accepted_restart_survives_cold_owner_pause_resume_and_normal_finish(runtime, monkeypatch):
    from ouroboros import agent_task_pipeline as pipeline
    from ouroboros import budget_pause, owner_pause, owner_wait
    from supervisor.events_budget import install_exact_budget_pause
    from supervisor.owner_pause_control import request_owner_pause
    from tests._budget_pause_exact_helpers import _loop_ctx

    original, limits = _loop_ctx(runtime.root, "requester")
    vars(original).update(vars(runtime.ctx))
    runtime.ctx = original
    intent = accept(runtime)
    later = enqueue(runtime, "after-restart")
    assert request_owner_pause("requester", request_id="owner-pause")["ok"]
    with pytest.raises(budget_pause.BudgetPauseRequested) as parked:
        budget_pause.enter_owner_pause(limits)
    budget_pause.end_dispatch_fence("requester")
    host = SimpleNamespace(DRIVE_ROOT=runtime.root, RUNNING=runtime.workers.RUNNING,
        PENDING=runtime.workers.PENDING, WORKERS=runtime.workers.WORKERS, sort_pending=lambda: None,
        persist_queue_snapshot=runtime.queue.persist_queue_snapshot, bridge=None)
    install_exact_budget_pause(host, "requester", budget_pause.exact_pause_marker(parked.value.pause)["checkpoint"])
    assert owner_pause.read_fence(runtime.root, "requester")["state"] == owner_pause.FENCE_PAUSED
    _, dispatched = worker(runtime, monkeypatch)
    runtime.server._check_pending_restart_drain(runtime.host)
    runtime.workers.assign_tasks()
    assert dispatched == [] and runtime.restart.agent_restart_pending()
    assert not runtime.server._pending_restart, "Pause does not start the drain clock"

    assert runtime.queue.resume_budget_paused_task("requester")["ok"]
    runtime.workers.assign_tasks()
    assert [task["id"] for task in dispatched] == ["requester"]
    resumed_task = dispatched[0]
    assert resumed_task["_attempt"] == 1
    assert runtime.workers.RUNNING["requester"]["started_at"] == original.task_started_at
    assert owner_pause.read_fence(runtime.root, "requester")["state"] == owner_pause.FENCE_PAUSED
    cold, resumed_limits = _loop_ctx(runtime.root, "requester")
    cold.budget_pause_resume = resumed_task["_budget_pause_resume"]
    state = budget_pause.load_budget_pause(cold)
    monkeypatch.setattr(owner_wait, "rebind_restored_route", lambda *a, **k: (None, "max"))
    messages, trace, usage = [], {}, {}
    budget_pause.resume_paused_loop(resumed_limits.tools, state, messages, trace, usage, set(),
                                    budget_remaining_usd=10)
    assert owner_pause.read_fence(runtime.root, "requester")["state"] == owner_pause.FENCE_RELEASED
    assert runtime.restart.agent_restart_holds(later)
    unknown = [row for row in messages if row.get("tool_call_id") == "call_b"]
    assert len(unknown) == 1 and "UNKNOWN" in unknown[0]["content"]
    from ouroboros.tools.control_runtime import _request_restart

    cold.pending_events = []
    assert _request_restart(cold, "do not replace the saved request").startswith("Restart requested.")
    assert not cold.pending_events and cold.pending_restart_reason == "finish first"

    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *a, **k: None)
    events = []
    pipeline.emit_task_results(SimpleNamespace(drive_root=runtime.root, repo_dir=runtime.root), None, None,
        events, {**resumed_task, "chat_id": 0, "text": "A"}, "Finished A", {"rounds": 1},
        {"tool_calls": [], "reasoning_notes": []}, 0, runtime.root / "logs", ctx=cold)
    ready = [event for event in events if event["type"] == "restart_request"]
    assert len(ready) == 1, "the restored requester must finish its accepted Restart"
    assert {key: ready[0][key] for key in ("request_id", "task_id", "task_attempt")} == {
        key: intent[key] for key in ("request_id", "task_id", "task_attempt")}
    assert events.index(ready[0]) > next(i for i, event in enumerate(events) if event["type"] == "task_done")
    assert cold.pending_restart_request == original.pending_restart_request
    runtime.workers.RUNNING.pop("requester")
    restarts = []
    monkeypatch.setattr(runtime.server, "_perform_supervisor_restart", lambda *a, **k: restarts.append(k))
    runtime.server._handle_restart_in_supervisor(ready[0], runtime.host)
    assert len(restarts) == 1 and restarts[0]["restart_reason"] == "finish first"
    assert later in runtime.queue.PENDING and runtime.restart.agent_restart_holds(later)


@pytest.mark.parametrize("case", ["another_task", "another_attempt", "legacy", "finished", "evolution"])
def test_restart_checkpoint_never_grants_a_new_restart_or_generic_evolution_finish(tmp_path, monkeypatch, case):
    from ouroboros import agent_task_pipeline as pipeline
    from ouroboros import owner_wait, working_checkpoint
    from ouroboros.tools.control_events import request_restart_intent
    from tests._budget_pause_exact_helpers import _loop_ctx

    original, limits = _loop_ctx(tmp_path, "requester")
    original.pending_events = []
    original.pending_restart_reason = "finish first"
    original.pending_restart_is_evolution = case == "evolution"
    request_restart_intent(original, "finish first", evolution=original.pending_restart_is_evolution)
    if case == "finished":
        original.pending_restart_reason = None
    working_checkpoint.save_round(limits, "post_batch")
    saved = json.loads(working_checkpoint.checkpoint_path(tmp_path, "requester", 1).read_bytes())
    if case == "legacy":
        saved.pop("restart", None)
    cold, resumed = _loop_ctx(tmp_path, "successor" if case == "another_task" else "requester",
                              attempt=2 if case == "another_attempt" else 1)
    owner_wait.restore_continuation_state(resumed.tools, saved, [], {}, {}, set())
    if case in {"finished", "evolution"}:
        assert cold.pending_restart_request == original.pending_restart_request
        assert cold.pending_restart_is_evolution is original.pending_restart_is_evolution
    else:
        assert not getattr(cold, "pending_restart_request", None)
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *a, **k: None)
    events = []
    pipeline.emit_task_results(SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path), None, None,
        events, {"id": cold.task_id, "type": "task", "chat_id": 0, "text": "A"}, "Finished A", {"rounds": 1},
        {"tool_calls": [], "reasoning_notes": []}, 0, tmp_path / "logs", ctx=cold)
    assert any(event["type"] == "task_done" for event in events)
    assert not any(event["type"] == "restart_request" for event in events)


@pytest.mark.parametrize("draining", [False, True])
def test_update_lock_open_failure_releases_only_restart_before_effects(runtime, monkeypatch, draining):
    import errno
    from pathlib import Path

    from ouroboros.server_liveness import drain_worker_events
    from ouroboros.task_results import write_task_result
    from supervisor import git_ops, worker_chat_lane

    intent = accept(runtime)
    queued = enqueue(runtime, "after-failed-restart")
    _, delivered = worker(runtime, monkeypatch)
    runtime.workers.assign_tasks()
    assert delivered == [] and runtime.restart.agent_restart_holds(queued)
    write_task_result(runtime.root, "requester", "completed", task_attempt=1)
    if not draining:
        runtime.workers.RUNNING.pop("requester")

    def unexpected_effect(*args, **kwargs):
        pytest.fail("lock acquisition failure must precede checkout, teardown and exit")

    runtime.host.safe_restart = unexpected_effect
    runtime.host.kill_workers = unexpected_effect
    monkeypatch.setattr(runtime.server, "_begin_task_teardown", unexpected_effect)
    monkeypatch.setattr(runtime.server, "_request_restart_exit", unexpected_effect)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", runtime.root)
    lock_path = runtime.root / "locks" / "managed_update.lock"
    opens, escaped = [], []
    original_open = Path.open

    def open_lock(path, *args, **kwargs):
        if path == lock_path:
            opens.append(path)
            raise OSError(errno.EMFILE, "injected lock open failure", str(path))
        return original_open(path, *args, **kwargs)

    runtime.queue.RETURNS_NAMED["another-door"] = "managed_update"
    with monkeypatch.context() as fault:
        fault.setattr(Path, "open", open_lock)
        runtime.events.put({**intent, "phase": "ready"})
        try:
            drain_worker_events(runtime.events, runtime.host, [time.monotonic(), {}, 0, None],
                                on_restart=runtime.server._handle_restart_in_supervisor)
            if draining:
                assert runtime.server._pending_restart["request_id"] == intent["request_id"]
                runtime.workers.RUNNING.pop("requester")
                runtime.server._check_pending_restart_drain(runtime.host)
        except OSError as exc:
            escaped.append(str(exc))
    assert runtime.events.empty() and opens == [lock_path]
    runtime.server._check_pending_restart_drain(runtime.host)  # the filesystem fault has cleared
    print({"escaped": escaped, "intent": runtime.restart.agent_restart_snapshot(),
           "pending_restart": runtime.server._pending_restart})
    assert not runtime.restart.agent_restart_pending(), "a consumed ready event must not leave a permanent hold"
    assert escaped == []
    receipt = json.loads(runtime.queue.QUEUE_SNAPSHOT_PATH.read_text())["restart_intent"]
    assert receipt["request_id"] == intent["request_id"] and receipt["status"] == "released"
    assert receipt["reason"] == "finish first" and receipt["refusal"] == "restart_refused"
    assert runtime.queue.RETURNS_NAMED == {"another-door": "managed_update"}
    assert not runtime.server._pending_restart
    assert not runtime.restart.agent_restart_holds(queued)
    runtime.queue.RETURNS_NAMED.clear()  # remove only the fixture's unrelated door for the dispatch proof
    runtime.workers.assign_tasks()
    assert [row["id"] for row in delivered] == [queued["id"]]
    assert queued not in runtime.queue.PENDING
    model_calls = []
    monkeypatch.setattr(runtime.workers, "_get_chat_agent", lambda: model_calls.append(True) or SimpleNamespace())
    turn = worker_chat_lane._admit_chat_task(None, 7, "Main after failed Restart")
    try:
        assert turn is not None and model_calls == [True]
    finally:
        if turn is not None:
            turn["registry"].unregister(turn["task"]["id"])


@pytest.mark.parametrize("phase", ["checkout", "teardown"])
def test_unknown_restart_effects_keep_independent_work_held(runtime, monkeypatch, phase):
    from ouroboros.task_results import write_task_result
    from supervisor import git_ops

    intent = accept(runtime)
    queued = enqueue(runtime, "after-unknown-restart")
    _, delivered = worker(runtime, monkeypatch)
    write_task_result(runtime.root, "requester", "completed", task_attempt=1)
    runtime.workers.RUNNING.pop("requester")
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", runtime.root)
    effects = []

    def checkout(**kwargs):
        effects.append("checkout")
        if phase == "checkout":
            raise OSError("checkout effects unknown")
        return True, ""

    def kill(**kwargs):
        effects.append("teardown")
        raise OSError("teardown effects unknown")

    runtime.host.safe_restart, runtime.host.kill_workers = checkout, kill
    monkeypatch.setattr(runtime.server, "_begin_task_teardown", lambda **k: ("cancelled", "restart"))
    monkeypatch.setattr(runtime.server, "_managed_update_pending_kwargs", lambda: {})
    monkeypatch.setattr(runtime.server, "_planned_delegate_restart_transaction_id", "")
    monkeypatch.setattr(runtime.server, "_request_restart_exit", lambda: pytest.fail("no physical exit in test"))
    with pytest.raises(OSError, match=f"{phase} effects unknown"):
        runtime.server._handle_restart_in_supervisor({**intent, "phase": "ready"}, runtime.host)
    assert effects == (["checkout"] if phase == "checkout" else ["checkout", "teardown"])
    assert runtime.restart.agent_restart_snapshot()["request_id"] == intent["request_id"]
    assert runtime.restart.agent_restart_pending() and runtime.restart.agent_restart_holds(queued)
    runtime.server._check_pending_restart_drain(runtime.host)
    runtime.workers.assign_tasks()
    assert delivered == [] and queued in runtime.queue.PENDING


@pytest.mark.parametrize("receipt_status", ["unconfirmed", "refused", "released", "accepted", "ready"])
def test_restart_receipt_status_reaches_registry_and_durable_trace(runtime, monkeypatch, receipt_status):
    from ouroboros import loop_tool_execution
    from ouroboros.observability import read_call_payload
    from ouroboros.tools import control_events
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools.tool_result import LegacyTextResultAdapter

    registry = ToolRegistry(repo_dir=runtime.root, drive_root=runtime.root)
    vars(registry._ctx).update(vars(runtime.ctx))
    runtime.ctx = registry._ctx
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setattr(control_events, "_PROMOTE_CONFIRM_TIMEOUT_SEC", 0)
    original_put, original_adapt = runtime.events.put_nowait, LegacyTextResultAdapter.from_text
    adapted, sent = [], []

    def put(event):
        original_put(event)
        if event.get("type") != "restart_request":
            return  # the real loop also publishes tool trace events on this bus
        sent.append(event)
        if receipt_status == "unconfirmed":
            return
        if receipt_status == "refused":
            runtime.queue.RETURNS_NAMED["fixture-door"] = "managed_update"
        runtime.server._handle_restart_in_supervisor(event, runtime.host)
        if receipt_status == "released":
            runtime.restart.release_agent_restart(event["request_id"], "fixture_pre_effect_refusal")
        elif receipt_status == "ready":
            runtime.server._handle_restart_in_supervisor({**event, "phase": "ready"}, runtime.host)

    def adapt(cls, tool_name, text):
        adapted.append((tool_name, text))
        return original_adapt(tool_name, text)

    monkeypatch.setattr(runtime.events, "put_nowait", put)
    monkeypatch.setattr(LegacyTextResultAdapter, "from_text", classmethod(adapt))
    rows = [loop_tool_execution._execute_single_tool(registry,
        {"id": f"restart-{index}", "function": {"name": "request_restart",
          "arguments": json.dumps({"reason": "finish first" if index == 0 else "ignored duplicate"})}},
        runtime.root / "logs", "requester") for index in range(2)]
    assert len(sent) == 1, rows
    request_id = sent[0]["request_id"]
    ok = receipt_status in {"accepted", "ready"}
    expected_status = "ok" if ok else "unavailable" if receipt_status == "unconfirmed" else "blocked"
    print({"receipt": receipt_status, "results": [(r["tool_result"].status, r["result_meta"]) for r in rows]})
    for row in rows:
        result = row["tool_result"]
        assert result.status == expected_status
        assert row["is_error"] is not ok
        if ok:
            assert result.text == ("Restart requested. Reason (the caller's own words): \"finish first\"."
                                   if row is rows[0] else "Restart requested. The previously acknowledged request is unchanged.")
        else:
            assert result.meta["request_id"] == request_id and result.meta["restart_status"] == receipt_status
            assert request_id in result.text
        manifest, payload, _ = read_call_payload(runtime.root, task_id="requester", call_id=row["trace_ref"]["call_id"])
        assert manifest["semantic_ok"] is ok and payload["semantic_ok"] is ok
        assert payload["result_meta"]["tool_result_status"] == expected_status
    if not ok:
        assert adapted == [], "native receipt facts must reach the loop without legacy text classification"
    settled = [json.loads(line) for line in (runtime.root / "logs" / "tools.jsonl").read_text().splitlines()]
    settled = [row for row in settled if row.get("tool_call_id") in {"restart-0", "restart-1"}]
    assert len(settled) == 2 and all(row["is_error"] is not ok for row in settled)
    if not ok:
        assert all(row["tool_result_meta"]["request_id"] == request_id for row in settled)


def evolution_completion(runtime, monkeypatch):
    """A reviewed cycle requesting Restart, then the real pipeline/terminal handler."""
    from ouroboros import agent_task_pipeline as pipeline
    from ouroboros.tools import control_runtime
    from supervisor import evolution_lifecycle as evolution
    from supervisor.events_evolution_done import _handle_evolution_task_done

    monkeypatch.setattr(evolution, "_STOP_LATCH", {"stopped": False})
    runtime.state.save_state({"owner_chat_id": 0, "evolution_mode_enabled": True, "evolution_owner_stopped": False})
    campaign = evolution.start_evolution_campaign("fixture cycle", source="owner")
    tx = evolution.begin_evolution_transaction("requester", cycle=1, campaign=campaign)
    assert evolution.record_evolution_commit(campaign["id"], tx["transaction_id"], "requester", "a" * 40)["ok"]
    runtime.ctx.current_task_type = "evolution"
    runtime.ctx.last_reviewed_commit_sha = "a" * 40
    runtime.ctx.task_metadata = {"evolution_transaction": tx}
    monkeypatch.setattr(control_runtime, "run_cmd", lambda args, **k: "" if "status" in args else "a" * 40)
    task = runtime.workers.RUNNING["requester"]["task"]
    task.update(type="evolution", metadata=runtime.ctx.task_metadata, chat_id=0, text="fixture cycle")
    intent = accept(runtime)
    monkeypatch.setattr(runtime.workers, "get_event_q", lambda: runtime.events)
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *a, **k: None)
    monkeypatch.setattr("ouroboros.evolution_checkpoints.append_evolution_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(evolution, "_cleanup_worktree_after_cycle",
                        lambda row, *a: row.update(cleanup_status="already_clean"))
    runtime.host.REPO_DIR = runtime.root

    def finish(status="completed", *, dispatch=False):
        if status == "completed":
            events = []
            pipeline.emit_task_results(SimpleNamespace(drive_root=runtime.root, repo_dir=runtime.root), None, None,
                events, task, "Finished cycle", {"rounds": 1}, {"tool_calls": [], "reasoning_notes": []},
                0, runtime.root / "logs", ctx=runtime.ctx)
            assert not any(row["type"] == "restart_request" for row in events), "evolution's authority owns readiness"
            terminal = next(row for row in events if row["type"] == "task_done")
        else:  # supervisor/reaper terminal: publish failure before consuming its event
            from ouroboros.task_results import write_task_result
            from ouroboros.outcomes import terminal_outcome_axes

            terminal = {"type": "task_done", "task_id": "requester", "task_type": "evolution", "status": status,
                "metadata": task["metadata"], "outcome_axes": terminal_outcome_axes(
                    lifecycle=status, execution="failed", reason_code="fixture_terminal_failure")}
            write_task_result(runtime.root, "requester", status, task_attempt=1, outcome_axes=terminal["outcome_axes"])
        _handle_evolution_task_done(runtime.host, evt=terminal, task_id="requester", task=task,
            task_done_event=terminal, outcome_axes=terminal["outcome_axes"], cost=None, rounds=1)
        if dispatch:
            from supervisor.events_task_done import _finish_task_done_dispatch
            from ouroboros.task_results import load_task_result

            _finish_task_done_dispatch(terminal, runtime.host, task_id="requester", worker_id=None,
                task=task, final_task_result=load_task_result(runtime.root, "requester"), task_done_event=terminal)
        else:
            runtime.workers.RUNNING.pop("requester")
        return terminal

    return intent, tx, finish


@pytest.mark.parametrize("decision", ["auto_off", "auto_off_unknown", "owner_stopped", "no_commit"])
def test_evolution_completion_without_restart_releases_exact_intent(runtime, monkeypatch, decision):
    from supervisor import evolution_lifecycle as evolution

    intent, tx, finish = evolution_completion(runtime, monkeypatch)
    queued = enqueue(runtime, "after-evolution")
    _, delivered = worker(runtime, monkeypatch)
    monkeypatch.setattr(runtime.server, "_perform_supervisor_restart", lambda *a, **k: pytest.fail("Restart refused"))
    monkeypatch.setenv("OUROBOROS_EVOLUTION_AUTO_RESTART", "false" if decision.startswith("auto_off") else "true")
    if decision == "auto_off_unknown":
        monkeypatch.setattr(evolution, "check_evolution_authority", lambda **k: {"ok": False, "reason": "state_lock_unavailable"})
    elif decision == "owner_stopped":
        runtime.state.update_state(lambda state: state.update(evolution_owner_stopped=True))
    elif decision == "no_commit":
        assert evolution.update_evolution_transaction("requester", commit_sha="", commit_receipt={})
    finish()
    runtime.server._check_pending_restart_drain(runtime.host)
    print({"decision": decision, "intent": runtime.restart.agent_restart_snapshot()})
    assert not runtime.restart.agent_restart_pending()
    receipt = json.loads(runtime.queue.QUEUE_SNAPSHOT_PATH.read_text())["restart_intent"]
    assert receipt["status"] == "released" and receipt["request_id"] == intent["request_id"]
    assert runtime.events.empty() and not runtime.server._pending_restart
    if decision.startswith("auto_off"):
        marker = json.loads((runtime.root / "state/pending_restart_verify.json").read_text())
        assert marker["evolution_claim"]["transaction_id"] == tx["transaction_id"]
        assert marker["expected_sha"] == "a" * 40 and marker["reason"] == "finish first"
    runtime.workers.assign_tasks()
    assert [row["id"] for row in delivered] == [queued["id"]]


@pytest.mark.parametrize("fault", ["authority_unknown", "stop_unknown", "marker_error", "adoption_unconfirmed"])
def test_evolution_completion_retries_unknown_decision_on_existing_restart_tick(runtime, monkeypatch, fault):
    from ouroboros import body_adoption
    from ouroboros.server_liveness import drain_worker_events
    from supervisor import evolution_lifecycle as evolution

    intent, _, finish = evolution_completion(runtime, monkeypatch)
    queued = enqueue(runtime, "after-evolution")
    _, delivered = worker(runtime, monkeypatch)
    requested = []
    monkeypatch.setenv("OUROBOROS_EVOLUTION_AUTO_RESTART", "true")
    monkeypatch.setattr(runtime.server, "_perform_supervisor_restart", lambda *a, **k: requested.append(k))
    with monkeypatch.context() as failure:
        if fault == "authority_unknown":
            failure.setattr(evolution, "check_evolution_authority", lambda **k: {"ok": False, "reason": "state_lock_unavailable"})
        elif fault == "stop_unknown":
            failure.setattr(evolution, "evolution_stop_reason", lambda *a: "panic_flag_unknown")
        elif fault == "marker_error":
            (runtime.root / "state/pending_restart_verify.json").unlink()  # the accepted intent still owns the caller's reason
            def write(*a, **k):
                raise OSError("injected evolution marker write failure")
            failure.setattr(evolution, "write_pending_restart_marker", write)
        else:
            failure.setattr(body_adoption, "authorize_for_task", lambda *a, **k: False)
        finish()
        runtime.server._check_pending_restart_drain(runtime.host)
        runtime.workers.assign_tasks()
        assert runtime.restart.agent_restart_pending() and runtime.restart.agent_restart_holds(queued)
        assert delivered == [] and runtime.events.empty() and requested == []
    runtime.server._check_pending_restart_drain(runtime.host)  # no duplicate task_done or model replay
    print({"recovered_fault": fault, "queued_restart_events": runtime.events.qsize()})
    assert not runtime.events.empty(), "a transient refusal must have an executable retry after it clears"
    event = runtime.events.get_nowait()
    assert event["request_id"] == intent["request_id"] and event["task_attempt"] == 1
    assert event["reason"] == "finish first" and event["evolution_restart"] is True
    # A successful put is not a consumed ready event. The same host-owned intent
    # remains executable if delivery was lost; duplicate delivery still performs once.
    runtime.server._check_pending_restart_drain(runtime.host)
    assert not runtime.events.empty(), "queue publication alone must not retire the ready-event retry"
    repeated = runtime.events.get_nowait()
    keys = ("request_id", "task_id", "task_attempt", "phase", "reason", "evolution_restart")
    assert {key: repeated[key] for key in keys} == {key: event[key] for key in keys}
    runtime.events.put(event)
    runtime.events.put(repeated)
    drain_worker_events(runtime.events, runtime.host, [time.monotonic(), {}, 0, None],
                        on_restart=runtime.server._handle_restart_in_supervisor)
    assert requested == [{"restart_reason": "finish first", "evolution_restart": True}]
    assert runtime.restart.agent_restart_holds(queued) and queued in runtime.queue.PENDING
