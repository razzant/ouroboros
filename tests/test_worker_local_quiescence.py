"""An idle worker answers its retirement probe only when no paid local producer is owed.

A returned task is not proof of settlement: a tool future abandoned at its logical
timeout (with its late settlement callback) and a model sender an owner Pause
abandoned keep producing paid effects after the attempt's own cleanup. The
invocation owner keeps a forgotten tail until its row's final ``settled`` event,
which follows the durable invocation settlement; the pause gate reads the same row.
"""

import json
import threading
import time
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from ouroboros import _usage_wait, budget_pause, loop_budget, model_wait, owner_pause, review_operation, tool_custody
from ouroboros import loop_tool_execution as execution
from ouroboros.task_results import load_task_result
from tests._usage_store_testing import attempt_rows_in_start_order
from tests.test_imagegen_tool import BODY, _transport, configured as configured, ctx as ctx  # noqa: F401
from tests.test_imagegen_wait_control import _setup
from tests.test_pause_model_receiver import _call, _until, scope as scope

pytestmark = pytest.mark.serial


@pytest.fixture
def registries(monkeypatch):
    """This process's producer registries, isolated from rows other tests left pinned."""
    monkeypatch.setattr(budget_pause, "_TOOL_FUTURES", {})
    monkeypatch.setattr(_usage_wait, "_ABANDONED_SENDS", set())


def test_forgotten_attempt_rows_stay_owned_through_the_late_callback_tail(registries):
    attempt = SimpleNamespace(task_id="tail-task", task_attempt=1)
    future = Future()
    owner = budget_pause.register_tool_future(attempt, "late-call", "generate_image", future)
    late = budget_pause.hold_tool_settlement(attempt, "late-call")  # the timeout path's late claim
    owner()
    budget_pause.forget_tool_scope(attempt)  # the attempt's own end
    # The owner keeps the forgotten tail: the pause gate and the probe read the same row.
    assert budget_pause.drain_local_tool_futures(attempt, timeout_sec=0)["unsettled"] == [
        {"operation_id": "late-call", "tool": "generate_image", "state": "running"}]
    assert not tool_custody.local_producers_settled(), "the receiver is still running"
    future.set_result("paid")
    assert not tool_custody.local_producers_settled(), "done is not callback-complete"
    late()
    assert tool_custody.local_producers_settled()
    assert budget_pause._TOOL_FUTURES == {}, "settled forgotten rows leave; nothing accumulates"


def test_forgotten_rows_leave_at_their_own_settlement_without_a_probe(registries):
    first, later = (SimpleNamespace(task_id="spare", task_attempt=attempt) for attempt in (1, 2))
    future = Future()
    budget_pause.register_tool_future(first, "late-call", "generate_image", future)()
    budget_pause.forget_tool_scope(first)
    budget_pause.forget_tool_scope(later)  # another attempt's end never drops this tail
    assert list(budget_pause._TOOL_FUTURES) == [budget_pause.tool_scope_key(first)]
    future.set_result("paid")  # its own settlement prunes it; no idle probe ran
    assert budget_pause._TOOL_FUTURES == {}


def test_timed_out_image_keeps_the_worker_until_its_paid_response_and_callback_settle(ctx, monkeypatch, registries):
    """Reviewer counterexample: the actual builtin, timeout and cleanup helpers, in-memory HTTP.

    The owner settles the durable invocation after the last hold, outside its lock:
    a done, unheld row is not yet settled.
    """
    entered, body, tail, in_tail = threading.Event(), threading.Event(), threading.Event(), threading.Event()
    settling, settle_go = threading.Event(), threading.Event()
    futures = []

    class HeldResponse(httpx.SyncByteStream):
        def __iter__(self):
            encoded = json.dumps({**BODY, "usage": {"cost": 0.42}}).encode()
            yield encoded[:10]
            entered.set()
            assert body.wait(20), "test failed to release the held image body"
            yield encoded[10:]

    def wire(request):
        assert request.method == "POST" and request.url.path == "/v1/images/generations"
        return httpx.Response(200, stream=HeldResponse())

    def logical_timeout(future, timeout_sec):
        futures.append(future)
        assert entered.wait(10), "builtin image send did not reach its response body"
        raise TimeoutError("deterministic logical timeout after physical image dispatch")

    emit = execution.emit_cognitive_operation_event

    def held_late_tail(*args, **kwargs):
        if kwargs.get("phase") == "finished" and futures and futures[0].done():
            in_tail.set()  # only the late settlement callback runs after its future is done
            assert tail.wait(20), "test failed to release the late callback"
        return emit(*args, **kwargs)

    settle = tool_custody.settle_invocation

    def held_settlement(outcome):
        if outcome.get("tool") == "generate_image":
            settling.set()  # the real _finish_tool_row: last hold released, its lock dropped
            assert settle_go.wait(20), "test failed to release the durable settlement"
        return settle(outcome)

    registry, task = _setup(ctx, monkeypatch)
    _transport(monkeypatch, wire)
    monkeypatch.setattr(execution, "future_result", logical_timeout)
    monkeypatch.setattr(execution, "emit_cognitive_operation_event", held_late_tail)
    monkeypatch.setattr(tool_custody, "settle_invocation", held_settlement)
    call = {"id": "live-image", "function": {"name": "generate_image", "arguments": json.dumps({
        "prompt": "fixture image", "model": "openai::fixture-image"})}}
    responses = lambda: list(ctx.drive_root.rglob("*_image_response.json"))  # noqa: E731
    try:
        with model_wait.task_model_wait_scope(task=task, drive_root=ctx.drive_root,
                                             event_queue=None, worker_slot_held=False):
            result = execution._execute_with_timeout(registry, call, ctx.drive_root / "logs", 300, ctx.task_id)
            assert result["result_meta"]["tool_result_code"] == "TOOL_TIMEOUT"
            # The ordinary loop exit runs this exact helper: no pause, no owner Stop.
            loop_budget._cleanup_loop_resources(None, loop_budget._LoopExitContext(
                tools=registry, drive_root=ctx.drive_root, task_id=ctx.task_id, event_queue=None,
                drive_logs=ctx.drive_root / "logs", accumulated_usage={}, llm_trace={}))
        # A logical exit never erases the physical tail: the pause gate still sees it.
        assert budget_pause.drain_local_tool_futures(ctx, timeout_sec=0)["drained"] is False
        [row] = [row for rows in budget_pause._TOOL_FUTURES.values() for row in rows.values()
                 if row["future"] is futures[0]]
        outcome = row["outcome"]

        def claim():
            handoffs = load_task_result(outcome["claim_root"], outcome["claim_task"]).get("launch_handoffs") or {}
            return handoffs.get(row["operation_id"])

        assert row["forgotten"] and claim()["state"] == "claimed", "the caller escaped by publishing custody"
        # The task finished, yet its receiver is alive: retirement is denied.
        assert not futures[0].done() and not responses()
        assert not tool_custody.local_producers_settled()
        body.set()
        assert in_tail.wait(20)
        assert futures[0].done() and responses(), "the paid response was received and retained"
        assert not tool_custody.local_producers_settled(), "the late callback tail still owns effects"
        tail.set()
        assert settling.wait(20)
        with budget_pause._TOOL_LOCK:
            assert budget_pause._row_settled_locked(row), "done and unheld already"
        assert not row["settled"].is_set() and claim()["state"] == "claimed"
        assert not tool_custody.local_producers_settled(), "the durable invocation settlement still runs"
        settle_go.set()
        assert _until(tool_custody.local_producers_settled)
        assert (claim() or {"state": "returned"})["state"] == "returned"
        assert budget_pause._TOOL_FUTURES == {}, "its owner pruned the settled forgotten row"
    finally:
        body.set()
        tail.set()
        settle_go.set()


def test_sender_abandoned_by_owner_pause_keeps_the_worker_until_it_settles(tmp_path, scope, registries):
    sent, release = threading.Event(), threading.Event()

    def send():
        sent.set()
        assert release.wait(10)
        return {"choices": [{"message": {"content": "late"}}]}

    outcome: dict = {}
    thread, _owner = _call(tmp_path, scope, send, outcome)
    try:
        assert sent.wait(5)
        owner_pause.install_fence(tmp_path, "root", request_id="pause")
        assert _until(lambda: "returned_at" in outcome)
        assert outcome["error"].receiver_abandoned is True
        assert not tool_custody.local_producers_settled(), "the sender still settles a paid answer"
    finally:
        release.set()
        thread.join(5)
    assert _until(tool_custody.local_producers_settled)
    assert attempt_rows_in_start_order(tmp_path)[0]["state"] == "settled"


def _wait_for(event):
    """Poll: a waiter that dies inside Event.wait would wedge the parent's set()."""
    deadline = time.monotonic() + 60
    while not event.is_set() and time.monotonic() < deadline:
        time.sleep(0.05)


def _injected_readiness_fault():
    raise RuntimeError("injected readiness-row fault")


def _probe_worker(in_q, out_q, data_root, repo_root, release, mode="clean", settle_entered=None, settle_go=None):
    """The real worker loop; its one task returns while a timed-out tool still runs."""
    from ouroboros import review_operation
    from supervisor.worker_process import worker_main

    if mode == "check_fault":  # the first check fails unexpectedly: unknown must defer, never kill the worker
        settled, calls = review_operation.local_reviews_settled, []

        def flaky():
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("injected probe-check fault")
            return settled()
        review_operation.local_reviews_settled = flaky
    if mode == "settling_owner":  # the owner's settlement after its last hold, outside its lock
        settle = tool_custody.settle_invocation

        def held(outcome):
            settle_entered.set()
            _wait_for(settle_go)
            return settle(outcome)
        tool_custody.settle_invocation = held

    class Agent:
        def handle_task(self, task):
            attempt = SimpleNamespace(task_id=task["id"], task_attempt=1)
            if mode == "late_owed":  # a late callback owes its write after the loop's wait began
                from ouroboros import custody_owed

                tries = []

                def write():
                    tries.append(1)
                    if len(tries) == 1:
                        raise OSError("the store is busy")  # the first attempt meets a busy store
                    out_q.put({"type": "owed_written", "tries": len(tries)})

                def late():
                    _wait_for(release)
                    custody_owed.owe(("late", task["id"]), write, label="late-call")
                threading.Thread(target=late, daemon=True).start()
                out_q.put({"type": "fixture_task_returned", "task_id": task["id"]})
                return []
            if mode == "settling_owner":
                from concurrent.futures import ThreadPoolExecutor

                def paid_call():
                    _wait_for(release)
                    return "paid"
                pool = ThreadPoolExecutor(1)
                future, owner = tool_custody.submit_tool_future(
                    attempt, "late-call", "generate_image", pool.submit, paid_call)
                late = budget_pause.hold_tool_settlement(attempt, "late-call")
                future.add_done_callback(lambda _future: late())  # the late callback's last act
                owner()
                budget_pause.forget_tool_scope(attempt)  # the loop's own attempt cleanup
                pool.shutdown(wait=False)
                out_q.put({"type": "fixture_task_returned", "task_id": task["id"]})
                return []
            future = Future()
            owner = budget_pause.register_tool_future(attempt, "late-call", "generate_image", future)
            late = budget_pause.hold_tool_settlement(attempt, "late-call")
            owner()
            budget_pause.forget_tool_scope(attempt)  # the loop's own attempt cleanup

            def settle():
                _wait_for(release)
                future.set_result("paid")
                late()
            threading.Thread(target=settle, daemon=True).start()
            out_q.put({"type": "fixture_task_returned", "task_id": task["id"]})
            return []

    replacements = {
        "ouroboros.agent.make_agent": lambda **kwargs: Agent(),
        "ouroboros.extension_loader.reload_all": lambda *args, **kwargs: None,
        "supervisor.worker_process._prepare_worker_task_runtime": (
            _injected_readiness_fault if mode == "readiness_fault" else lambda: None),
        "supervisor.worker_process._adopt_published_extensions": lambda *_: None,
    }
    with ExitStack() as stack:
        for name, value in replacements.items():
            stack.enter_context(patch(name, value))
        worker_main(0, in_q, out_q, repo_root, data_root)


@contextmanager
def _probe_worker_process(tmp_path, monkeypatch, mode):
    import multiprocessing

    root = tmp_path / "data"
    root.mkdir()
    for key, value in {"TOTAL_BUDGET": "100", "OUROBOROS_APP_ROOT": str(tmp_path),
                       "OUROBOROS_DATA_DIR": str(root), "OUROBOROS_SETTINGS_PATH": str(root / "settings.json")}.items():
        monkeypatch.setenv(key, value)
    context = multiprocessing.get_context("spawn")
    in_q, out_q = context.Queue(), context.Queue()
    release, settle_entered, settle_go = context.Event(), context.Event(), context.Event()
    proc = context.Process(target=_probe_worker, args=(
        in_q, out_q, str(root), str(Path(__file__).resolve().parents[1]), release, mode, settle_entered, settle_go))
    events = []

    def receive(predicate, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                events.append(out_q.get(timeout=max(0.01, deadline - time.monotonic())))
            except Exception:
                continue
            if isinstance(events[-1], dict) and predicate(events[-1]):
                return events[-1]
        return None

    proc.start()
    try:
        in_q.put({"id": "unwound", "type": "task", "chat_id": 1, "drive_root": str(root)})
        assert receive(lambda event: event.get("type") == "fixture_task_returned", 30)
        yield SimpleNamespace(proc=proc, in_q=in_q, receive=receive, release=release,
                              settle_entered=settle_entered, settle_go=settle_go)
        in_q.put(None)
        proc.join(10)
        assert proc.exitcode == 0
    finally:
        release.set()
        settle_go.set()
        if proc.is_alive():
            in_q.put(None)
            proc.join(5)
        if proc.is_alive():
            proc.terminate()
            proc.join(5)
        for channel in (in_q, out_q):
            channel.close()
            channel.join_thread()


def _quiescent(event):
    return event.get("type") == "worker_quiescent"


@pytest.mark.parametrize("fault", [False, True], ids=["clean", "check_fault"])
def test_real_worker_loop_answers_the_probe_only_after_its_late_tool_settles(tmp_path, monkeypatch, fault):
    with _probe_worker_process(tmp_path, monkeypatch, "check_fault" if fault else "clean") as worker:
        worker.in_q.put({"type": "worker_quiescence", "token": "probe-1"})
        # Several recheck cadences pass while the timed-out tool still runs: no answer.
        assert worker.receive(_quiescent, 2.0) is None
        worker.release.set()
        answer = worker.receive(_quiescent, 15)
        assert answer is not None and answer["token"] == "probe-1" and answer["pid"] == worker.proc.pid
        assert answer["process_birth"]


def test_real_worker_loop_waits_for_the_owner_settlement_after_the_last_hold(tmp_path, monkeypatch):
    """Future done → late callback released → owner settlement still running → no answer."""
    with _probe_worker_process(tmp_path, monkeypatch, "settling_owner") as worker:
        worker.in_q.put({"type": "worker_quiescence", "token": "probe-1"})
        assert worker.receive(_quiescent, 2.0) is None
        worker.release.set()
        assert worker.settle_entered.wait(15), "the late tail never reached its owner settlement"
        # Done and unheld, but _finish_tool_row has not set its event: several cadences, no answer.
        assert worker.receive(_quiescent, 2.0) is None
        worker.settle_go.set()
        answer = worker.receive(_quiescent, 15)
        assert answer is not None and answer["token"] == "probe-1" and answer["pid"] == worker.proc.pid
        assert answer["process_birth"]


def test_a_failed_readiness_row_never_unbinds_the_birth_the_probe_answer_carries(tmp_path, monkeypatch):
    """Owner 14 Oct 21:09 #1: the child measures its birth first, in its own try.

    Before, a raise in the readiness block left the birth unbound, and the first idle
    probe answer then ended the worker loop instead of answering.
    """
    from ouroboros.process_identity import process_kernel_birth

    with _probe_worker_process(tmp_path, monkeypatch, "readiness_fault") as worker:
        worker.release.set()
        worker.in_q.put({"type": "worker_quiescence", "token": "probe-1"})
        answer = worker.receive(_quiescent, 15)
        assert answer is not None and answer["token"] == "probe-1" and answer["pid"] == worker.proc.pid
        assert answer["process_birth"], "measured before the failing readiness row"
        assert answer["kernel_birth"] == (process_kernel_birth(worker.proc.pid) or ""), "so is the kernel birth"


# ---------------------------------------------------------------------------
# The named predicate (``worker_process.local_work_settled``): producers, reviews, producers.
# Opus d94's discriminating regressions: a handoff lands BETWEEN two registry reads; the
# two-read orders each acknowledge falsely in one direction, the predicate in neither.
# ---------------------------------------------------------------------------

@pytest.fixture
def handoff_registries(monkeypatch, registries):
    """This process's tool and review registries, empty; the real readers over them."""
    from ouroboros import review_custody

    monkeypatch.setattr(review_operation, "_LIVE", {})
    monkeypatch.setattr(review_custody, "_ACTIVE", {})


def _live_tool(operation_id):
    """A real registered tool future, unsettled until ``settle``."""
    attempt = SimpleNamespace(task_id="probe-task", task_attempt=1)
    future = Future()
    owner = budget_pause.register_tool_future(attempt, operation_id, "inspect", future)

    def settle():
        future.set_result("done")
        owner()
    return SimpleNamespace(settle=settle)


def _open_review(owner_id):
    with review_operation._LOCK:
        review_operation._LIVE[owner_id] = SimpleNamespace(closed=False)


def _close_review(owner_id):
    with review_operation._LOCK:
        review_operation._LIVE.pop(owner_id)


def _handoff_between_reads(monkeypatch, handoff):
    """Run ``handoff`` once, right after the first registry read of a probe check."""
    pending = [handoff]
    for module, name in ((tool_custody, "local_producers_settled"), (review_operation, "local_reviews_settled")):
        def read(real=getattr(module, name)):
            answer = real()
            if pending:
                pending.pop()()
            return answer
        monkeypatch.setattr(module, name, read)


def _probe(order):
    from supervisor import worker_process

    return {
        "reviews_then_producers": lambda: review_operation.local_reviews_settled() and tool_custody.local_producers_settled(),
        "producers_then_reviews": lambda: tool_custody.local_producers_settled() and review_operation.local_reviews_settled(),
        "named_predicate": worker_process.local_work_settled,
    }[order]


@pytest.mark.parametrize("order,acknowledges_falsely", [
    ("reviews_then_producers", True), ("producers_then_reviews", False), ("named_predicate", False)])
def test_a_tool_handing_off_to_a_review_between_reads_never_acknowledges(
        handoff_registries, monkeypatch, order, acknowledges_falsely):
    tool = _live_tool("late-tool")

    def handoff():  # the tool registers the review it starts, then its own row settles
        _open_review("review-x")
        tool.settle()

    _handoff_between_reads(monkeypatch, handoff)
    probe = _probe(order)
    answers = [probe() for _ in range(3)]  # the worker's bounded rechecks while the review lives
    assert (True in answers) is acknowledges_falsely
    _close_review("review-x")
    assert probe(), "settled: the probe answers"


@pytest.mark.parametrize("order,acknowledges_falsely", [
    ("producers_then_reviews", True), ("reviews_then_producers", False), ("named_predicate", False)])
def test_a_review_handing_off_to_an_inspection_tool_between_reads_never_acknowledges(
        handoff_registries, monkeypatch, order, acknowledges_falsely):
    _open_review("review-x")
    inspection = []

    def handoff():  # the review's inspection call registers its future, then the review closes
        inspection.append(_live_tool("inspection"))
        _close_review("review-x")

    _handoff_between_reads(monkeypatch, handoff)
    probe = _probe(order)
    answers = [probe() for _ in range(3)]
    assert (True in answers) is acknowledges_falsely
    inspection[0].settle()
    assert probe(), "settled: the probe answers"


def test_empty_and_fully_settled_registries_approve(handoff_registries):
    from supervisor.worker_process import local_work_settled

    assert local_work_settled(), "nothing owed: approve"
    tool = _live_tool("work")
    _open_review("review-x")
    assert not local_work_settled()
    tool.settle()
    _close_review("review-x")
    assert local_work_settled(), "everything settled: approve"


def test_a_fact_owed_after_the_idle_wait_began_wakes_the_last_spare_worker(tmp_path, monkeypatch):
    """No probe and no task arrive: the late owe alone wakes the blocked wait, and a busy store is retried."""
    with _probe_worker_process(tmp_path, monkeypatch, "late_owed") as worker:
        time.sleep(0.5)  # the loop is back in its untimed wait: nothing owed, no probe
        worker.release.set()
        written = worker.receive(lambda event: event.get("type") == "owed_written", 15)
        assert written is not None and written["tries"] == 2
