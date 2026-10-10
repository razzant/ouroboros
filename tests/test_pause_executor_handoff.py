"""Real executor submissions versus Pause, cancellation and nested new work."""
import asyncio
import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from ouroboros import usage_accounting as ua
from ouroboros import owner_pause
from ouroboros.llm_attempt import _deadline_checked_send, require_physical_dispatch_window
from ouroboros.task_results import write_task_result
from tests._usage_store_testing import attempt_rows_in_start_order

pytestmark = pytest.mark.serial


def setup_call(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TOTAL_BUDGET", "1000")
    write_task_result(tmp_path, "root", "running", root_task_id="root")
    return ua.UsageScope(drive_root=tmp_path, task_id="root", root_task_id="root")


def request():
    return ua.AttemptRequest(model="m", provider="test", reservation_usd=.2)


def extract(_response):
    return {}, .1, True


def test_sync_executor_accepts_before_pause_and_waits_without_launch_lock(tmp_path, monkeypatch):
    scope = setup_call(tmp_path, monkeypatch)
    accepted, release = threading.Event(), threading.Event()
    real_submit = ThreadPoolExecutor.submit
    marker = contextvars.ContextVar("test_handoff_context", default="missing")
    observed = []

    def submit(executor, fn, *args, **kwargs):
        def delayed():
            assert release.wait(5)
            return fn(*args, **kwargs)
        future = real_submit(executor, delayed)
        accepted.set()
        return future
    monkeypatch.setattr(ThreadPoolExecutor, "submit", submit)

    def send():
        require_physical_dispatch_window()  # Physical SDK entry is AFTER accepted Pause.
        observed.append(marker.get())
        with pytest.raises(Exception, match="owner_pause"):
            ua.execute_physical_attempt(request(), lambda: pytest.fail("new nested send"))
        return "paid answer"
    def call():
        marker.set("copied")
        with ua.usage_scope(scope):
            wrapped, prepare = _deadline_checked_send(send, None)
            observed.append(ua.execute_physical_attempt(request(), wrapped, before_dispatch=prepare, extractor=extract))
    thread = threading.Thread(target=call)
    thread.start()
    try:
        assert accepted.wait(5)
        fence, _ = owner_pause.install_fence(tmp_path, "root", request_id="submitted")
        assert fence["state"] == "requested" and thread.is_alive() and not observed
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive() and observed == ["copied", "paid answer"]
    assert attempt_rows_in_start_order(tmp_path)[0]["state"] == "settled"


def test_async_executor_accepts_before_pause_and_keeps_context(tmp_path, monkeypatch):
    scope = setup_call(tmp_path, monkeypatch)
    original = owner_pause.submit_model
    marker = contextvars.ContextVar("async_handoff_context", default="missing")
    def submitted(*args):
        future = original(*args)
        owner_pause.install_fence(tmp_path, "root", request_id="submitted")
        assert not future.done()
        return future
    monkeypatch.setattr(owner_pause, "submit_model", submitted)
    async def send():
        require_physical_dispatch_window()
        assert marker.get() == "copied"
        assert owner_pause.read_fence(tmp_path, "root")["state"] == "requested"
        with pytest.raises(Exception, match="owner_pause"):
            await ua.execute_physical_attempt_async(request(), lambda: pytest.fail("nested send"))
        return "answer"
    async def call():
        marker.set("copied")
        with ua.usage_scope(scope):
            wrapped, prepare = _deadline_checked_send(send, None)
            assert await ua.execute_physical_attempt_async(request(), wrapped, before_dispatch=prepare, extractor=extract) == "answer"
            assert ua.last_physical_attempt_capture().state == "settled"
    asyncio.run(call())
    assert attempt_rows_in_start_order(tmp_path)[0]["state"] == "settled"


def test_async_cancel_during_accounting_joins_and_retains_exact_answer(tmp_path, monkeypatch):
    scope = setup_call(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    answer = {"exact": "paid answer"}
    def extractor(response):
        assert response is answer
        entered.set()
        assert release.wait(5)
        return extract(response)
    async def send():
        return answer
    observed = []
    async def invoke():
        try:
            return await ua.execute_physical_attempt_async(request(), send, extractor=extractor)
        except BaseException as exc:
            observed.append((type(exc).__name__, vars(exc), repr(exc.__cause__)))
            raise
    async def call():
        with ua.usage_scope(scope):
            task = asyncio.create_task(invoke())
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done(), "consumer must stay alive through paid response custody"
            release.set()
            with pytest.raises(asyncio.CancelledError) as caught:
                await task
            # Python 3.10 may recreate cancellation across a Task boundary;
            # the actual caller owns exact custody before that boundary.
            assert isinstance(caught.value, asyncio.CancelledError)
            facts = observed[0][1]
            assert facts["response"] is answer
            assert facts["physical_attempt_capture"].state == "settled"
            assert facts["response_manifest_ref"]
    try:
        asyncio.run(call())
    finally:
        release.set()
    rows = ua.read_usage_records(tmp_path)
    assert [row["state"] for row in rows] == ["settled"]


def test_handed_sender_still_obeys_stop(tmp_path, monkeypatch):
    from ouroboros.model_wait import task_model_wait_scope
    scope = setup_call(tmp_path, monkeypatch)
    stop = []
    original = owner_pause.submit_model
    def submitted(*args):
        future = original(*args)
        stop.append(True)
        return future
    monkeypatch.setattr(owner_pause, "submit_model", submitted)
    async def send():
        require_physical_dispatch_window()
        pytest.fail("Stop must still refuse physical entry")
    async def call():
        with ua.usage_scope(scope), task_model_wait_scope(task={"id": "root", "_attempt": 1},
                drive_root=tmp_path, event_queue=None, worker_slot_held=False,
                owner_control=lambda: "cancelled" if stop else None):
            with pytest.raises(Exception, match="cancelled"):
                await ua.execute_physical_attempt_async(request(), send)
    asyncio.run(call())
    assert attempt_rows_in_start_order(tmp_path)[0]["state"] == "released"



def test_sticky_handoff_is_single_use_and_preserves_executor_affinity(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolRegistry
    setup_call(tmp_path, monkeypatch)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = registry._ctx.root_task_id = "root"
    observed = []
    def body(ctx, **_kw):
        observed.append(threading.get_ident())
        nested = registry.execute_result("knowledge_read", {"topic": "nested"})
        assert nested.meta["owner_pause_not_started"]
        return "done"
    registry.override_handler("knowledge_read", body)
    release = threading.Event()
    with ThreadPoolExecutor(max_workers=1) as executor:
        identity = executor.submit(threading.get_ident).result()
        blocker = executor.submit(lambda: release.wait(5))
        future = owner_pause.submit_tool(registry._ctx, "knowledge_read", executor.submit,
            registry.execute_result, "knowledge_read", {"topic": "started"})
        owner_pause.install_fence(tmp_path, "root", request_id="accepted")
        release.set()
        assert blocker.result()
        assert future.result().status == "ok"
    assert observed == [identity]


def test_async_cancel_before_sender_entry_releases_and_keeps_cancellation(tmp_path, monkeypatch):
    scope = setup_call(tmp_path, monkeypatch)
    original = owner_pause.submit_model
    def submit(*args):
        future = original(*args)
        asyncio.current_task().cancel()
        future.cancel()  # Positive executor fact: coroutine has not entered.
        return future
    monkeypatch.setattr(owner_pause, "submit_model", submit)
    async def send():
        pytest.fail("unentered sender must not run")
    async def call():
        with ua.usage_scope(scope):
            with pytest.raises(asyncio.CancelledError):
                await ua.execute_physical_attempt_async(request(), send)
    asyncio.run(call())
    assert attempt_rows_in_start_order(tmp_path)[0]["state"] == "released"



def test_pause_cannot_be_accepted_between_gate_and_executor_acceptance(tmp_path, monkeypatch):
    scope = setup_call(tmp_path, monkeypatch)
    submitting, finish_submit, pause_accepted, finish_send = [threading.Event() for _ in range(4)]
    original = ThreadPoolExecutor.submit
    failures = []
    def submit(executor, function, *args, **kwargs):
        submitting.set()
        assert finish_submit.wait(5)
        return original(executor, function, *args, **kwargs)
    monkeypatch.setattr(ThreadPoolExecutor, "submit", submit)
    def send():
        assert finish_send.wait(5)
        return "done"
    def call():
        try:
            with ua.usage_scope(scope):
                ua.execute_physical_attempt(request(), send, extractor=extract)
        except BaseException as exc:
            failures.append(exc)
    def pause():
        try:
            owner_pause.install_fence(tmp_path, "root", request_id="racing")
            pause_accepted.set()
        except BaseException as exc:
            failures.append(exc)
    sender = threading.Thread(target=call)
    sender.start()
    pauser = threading.Thread(target=pause)
    try:
        assert submitting.wait(5)
        pauser.start()
        assert not pause_accepted.wait(.2), "split precheck allowed Pause before executor acceptance"
        finish_submit.set()
        assert pause_accepted.wait(5), "network completion must not hold the launch lock"
        assert sender.is_alive()
    finally:
        finish_submit.set()
        finish_send.set()
        sender.join(5)
        if pauser.ident:
            pauser.join(5)
    assert not failures and not sender.is_alive() and not pauser.is_alive()
