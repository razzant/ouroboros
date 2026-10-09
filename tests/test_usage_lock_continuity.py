"""Owned pre-send contention preserves stack, controls, claims and async custody.

The money acquisition primitive is the usage store's write hold
(``usage_accounting._locked`` = ``usage_store.hold``): on the enforced tier
SQLite's own write lock, retried by the existing sliced waits. The store keeps
one current row per attempt, so an attempt's history reads as its final state.
"""
from __future__ import annotations

import asyncio
import contextlib
import errno
import json
import multiprocessing
import os
import queue
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros import platform_layer as platform
from ouroboros import usage_accounting as ua
from ouroboros import usage_ledger as ledger
from ouroboros.llm_attempt import PhysicalDispatchInterrupted
from ouroboros.model_wait import task_model_wait_scope
from ouroboros.review_dispatch import ReviewPaidStamp, bind_api_review_paid_stamp
from tests._usage_store_testing import ledger_rows, request, root as root

pytestmark = pytest.mark.serial


@contextlib.contextmanager
def held_lock(root, timeout=5):
    acquired, release = threading.Event(), threading.Event()

    def hold():
        with ua._locked(root):
            acquired.set()
            assert release.wait(timeout)

    thread = threading.Thread(target=hold)
    thread.start()
    assert acquired.wait(2)
    try:
        yield release
    finally:
        release.set()
        thread.join(3)
        assert not thread.is_alive()


@pytest.fixture(autouse=True)
def lifecycle_authority(root):
    from ouroboros.task_results import write_task_result
    # Managed model consumers require the same canonical lifecycle authority
    # as production. A missing result tests unreadable authority, not contention.
    # Prepare before each test installs its negative lock fault.
    for tid in ("dominant", "child"):
        write_task_result(root, tid, "running", root_task_id="dominant")


@contextlib.contextmanager
def owner(root, **values):
    events = queue.Queue()
    with task_model_wait_scope(task={"id": "child", **values.pop("task", {})},
                               drive_root=root, event_queue=events, worker_slot_held=False,
                               owner_control=values.pop("control", lambda: None), **values):
        yield events


@pytest.fixture
def short_acquisitions(monkeypatch):
    original = ua._locked

    @contextlib.contextmanager
    def shortened(root, *, timeout_sec=45):
        with original(root, timeout_sec=min(timeout_sec, .025)) as beat:
            yield beat

    monkeypatch.setattr(ua, "_locked", shortened)


def rows(root):
    return ledger_rows(root)


@contextlib.contextmanager
def held_name_lock(root, timeout=5):
    """The name-protocol helper itself (the store's ``name`` tier lock)."""
    acquired, release = threading.Event(), threading.Event()

    def hold():
        with ledger._locked(root):
            acquired.set()
            assert release.wait(timeout)

    thread = threading.Thread(target=hold)
    thread.start()
    assert acquired.wait(2)
    try:
        yield release
    finally:
        release.set()
        thread.join(3)
        assert not thread.is_alive()


@pytest.mark.parametrize("stage", ["reserve", "dispatch"])
@pytest.mark.parametrize("interactive", [False, True])
@pytest.mark.parametrize("scheduler_pause", [0, .25])
def test_same_chain_preparation_stamp_claim_and_one_send(root, short_acquisitions, stage, interactive, monkeypatch, scheduler_pause):
    calls, prepared, stamps = [], [], []
    with contextlib.ExitStack() as stack:
        events = stack.enter_context(owner(root, task={"_is_direct_chat": interactive, "_presence_turn": interactive}))
        stack.enter_context(ua.physical_attempt_limit(1))
        stack.enter_context(bind_api_review_paid_stamp(ReviewPaidStamp(lambda: stamps.append(1), fail_closed=True)))

        def start_hold():
            release = stack.enter_context(held_lock(root))
            original_put = events.put

            def release_on_entered(item, *args, **kwargs):
                original_put(item, *args, **kwargs)
                data = item.get("data", {})
                if (data.get("checkpoint_kind") == "usage_lock_wait"
                        and data.get("phase") == "entered"):
                    release.set()

            # Hold through a real reported contention, not a scheduler-sized timer.
            # All emitted checkpoints and one-send assertions below stay real.
            monkeypatch.setattr(events, "put", release_on_entered)
            time.sleep(scheduler_pause)  # longer than the former .18s release timer

        def before(held):
            prepared.append(held.attempt_id)
            if stage == "dispatch":
                start_hold()

        if stage == "reserve":
            start_hold()
        response = {"usage": {}}
        assert ua.execute_physical_attempt(request(root), lambda: calls.append(1) or response,
                                           before_dispatch=before) is response
        assert ua._PHYSICAL_LIMIT.get().used == 1
    assert len(calls) == len(prepared) == len(stamps) == 1
    assert [row["state"] for row in rows(root)] == ["settled"]
    assert len({row["attempt_id"] for row in rows(root)}) == 1
    waits = [item["data"] for item in list(events.queue)
             if item.get("data", {}).get("checkpoint_kind") == "usage_lock_wait"]
    assert [row["phase"] for row in waits] == ["entered", "ended"]
    from ouroboros.memory import Memory
    summary = Memory(root).summarize_progress(waits)
    assert "Waiting for accounting access" in summary
    assert "Accounting wait ended" in summary


@pytest.mark.parametrize("reason", ["cancelled", "deadline", "finalize_requested"])
def test_control_during_reserved_wait_returns_claim_but_failed_release_retains_bound(root, short_acquisitions, reason):
    stopped, prepared, stamps = threading.Event(), [], []
    with contextlib.ExitStack() as stack:
        stack.enter_context(owner(root, control=lambda: reason if stopped.is_set() else None))
        stack.enter_context(ua.physical_attempt_limit(1))
        stack.enter_context(bind_api_review_paid_stamp(ReviewPaidStamp(lambda: stamps.append(1), fail_closed=True)))

        def before(held):
            prepared.append(held)
            stack.enter_context(held_lock(root))
            timer = threading.Timer(.12, stopped.set)
            timer.start()
            stack.callback(timer.join, 2)

        with pytest.raises(PhysicalDispatchInterrupted) as error:
            ua.execute_physical_attempt(request(root), lambda: pytest.fail("provider called"), before_dispatch=before)
        assert error.value.control_reason == reason
        assert error.value.physical_attempt_capture.state == "reserved"
        assert ua._PHYSICAL_LIMIT.get().used == 0
        assert not ua._PHYSICAL_LIMIT.get().claimed_ids
        assert stamps == [1]  # the paid review stamp is NOT returned
    assert [row["state"] for row in rows(root)] == ["reserved"]
    assert ua.usage_projection(root)["reserved_usd"] == 1
    ua.release_attempt(prepared[0], "before_dispatch_failed:control")


def test_fence_closing_interrupts_dispatch_wait_before_lock_is_available(root, short_acquisitions, monkeypatch):
    from ouroboros import budget_pause
    fenced = threading.Event()
    monkeypatch.setattr(budget_pause, "dispatch_fenced", lambda task: fenced.is_set())
    with contextlib.ExitStack() as stack:
        stack.enter_context(owner(root))

        def before(held):
            stack.enter_context(held_lock(root))
            timer = threading.Timer(.12, fenced.set)
            timer.start()
            stack.callback(timer.join, 2)

        with pytest.raises(ua.DispatchFenced) as error:
            ua.execute_physical_attempt(request(root), lambda: pytest.fail("sent"), before_dispatch=before)
        assert error.value.physical_attempt_capture.state == "reserved"


def _known_spend(root, usd):
    """A settled, finally priced attempt: the KNOWN spend every limit decides on (#1487)."""
    held = ua.reserve_attempt(request(root, provider="openai", reservation_usd=usd))
    ua.mark_dispatched(held)
    ua.settle_attempt(held, {}, cost_usd=usd, cost_final=True)


def test_cap_is_resolved_again_after_pre_reservation_wait(root, short_acquisitions, monkeypatch):
    cap = [10]
    monkeypatch.setattr(ua, "_global_limit", lambda req: cap[0])
    _known_spend(root, 1.0)
    with owner(root), held_lock(root) as release:
        def reduce():
            cap[0] = .5  # the owner lowers the wallet below the known $1 while the send waits
            release.set()
        timer = threading.Timer(.12, reduce)
        timer.start()
        try:
            with pytest.raises(ua.BudgetExceeded):
                ua.execute_physical_attempt(request(root), lambda: pytest.fail("sent"))
        finally:
            timer.join(2)
    assert [row["state"] for row in rows(root)] == ["settled"]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_cap_reduction_after_reservation_refuses_send_and_returns_claim(root, monkeypatch, asynchronous):
    cap = [10]
    monkeypatch.setattr(ua, "_global_limit", lambda req: cap[0])
    _known_spend(root, 1.0)
    def before(held):
        assert rows(root)[-1]["state"] == "reserved"
        cap[0] = .5  # lowered below the known $1 between reservation and send
    def send():
        pytest.fail("provider called after cap reduction")
    async def async_send():
        send()
    with owner(root), ua.physical_attempt_limit(1):
        with pytest.raises(ua.BudgetExceeded) as error:
            if asynchronous:
                asyncio.run(ua.execute_physical_attempt_async(request(root), async_send, before_dispatch=before))
            else:
                ua.execute_physical_attempt(request(root), send, before_dispatch=before)
        assert error.value.limit_scope == "global"
        assert error.value.physical_attempt_capture.state == "released"
        assert ua._PHYSICAL_LIMIT.get().used == 0
    assert [row["state"] for row in rows(root)] == ["settled", "released"]
    assert ua.usage_projection(root)["accounted_usd"] == 1.0  # the known $1 only; the claim returned


def test_interactive_window_expiry_is_typed_and_never_quota_wait(root, short_acquisitions, monkeypatch):
    from ouroboros import config
    monkeypatch.setattr(config, "get_task_idle_timeout_sec", lambda: .08)
    with owner(root, task={"_is_direct_chat": True, "_presence_turn": True}), held_lock(root):
        with pytest.raises(PhysicalDispatchInterrupted) as error:
            ua.execute_physical_attempt(request(root), lambda: pytest.fail("sent"))
    assert error.value.control_reason == "accounting_wait_expired"
    assert rows(root) == []


@pytest.mark.parametrize("reason", ["kernel_refused", "name_tier_refused", "identity_unreadable", "permission", "unknown"])
def test_negative_acquisition_facts_are_never_cooperatively_retried(root, monkeypatch, reason):
    calls = []

    def refused(*args, outcome=None, **kwargs):
        calls.append(1)
        outcome.update(reason=reason)

    # The name tier (no kernel file locks) acquires the name-protocol lock for
    # every store access; its typed refusals reach the caller unchanged.
    monkeypatch.setattr(platform, "kernel_file_locks_enforced", lambda path: False)
    ua.usage_projection(root)  # the store exists, on the name tier
    monkeypatch.setattr(platform, "acquire_exclusive_file_lock", refused)
    with owner(root), pytest.raises(ledger.UsageLockUnavailable) as error:
        ua.execute_physical_attempt(request(root), lambda: pytest.fail("sent"))
    assert error.value.reason == reason
    assert calls == [1]


def test_real_platform_contention_and_kernel_refusal_are_distinct(root, monkeypatch):
    with held_name_lock(root), pytest.raises(ledger.UsageLockUnavailable) as error:
        with ledger._locked(root, timeout_sec=.02):
            pytest.fail("held lock acquired")
    assert error.value.reason == "contention"
    monkeypatch.setattr(platform, "kernel_file_locks_enforced", lambda path: True)

    def refused(fd):
        raise OSError(errno.EIO, "kernel refusal")

    monkeypatch.setattr(platform, "file_lock_exclusive_nb", refused)
    with pytest.raises(ledger.UsageLockUnavailable) as error:
        with ledger._locked(root, timeout_sec=.02):
            pytest.fail("refused lock acquired")
    assert error.value.reason == "kernel_refused"
    assert error.value.error_number == errno.EIO


def test_after_response_two_accounting_failures_return_exact_response_once(root, short_acquisitions):
    sent = []
    response = {"answer": "original paid answer", "usage": {"cost": .123}}
    with contextlib.ExitStack() as stack:
        stack.enter_context(owner(root))

        def send():
            sent.append(1)
            stack.enter_context(held_lock(root))
            return response

        assert ua.execute_physical_attempt(request(root), send) is response
        assert ua.last_physical_attempt_capture().state == "dispatched"
    assert sent == [1]
    assert ua.usage_projection(root)["unresolved_upper_bound_usd"] == 1
    assert [row["state"] for row in rows(root)] == ["dispatched"]


def _observe_async_accounting_wait(root, monkeypatch):
    facts = {}

    async def run():
        loop, sent = asyncio.get_running_loop(), []
        original = ua._locked
        facts["loop_thread"] = threading.get_ident()
        watchdog_fired = threading.Event()
        response = {"usage": {}}

        async def send():
            sent.append(1)
            return response

        # Only the loop callback (or the cleanup watchdog) may release this hold.
        with owner(root), ua.physical_attempt_limit(1), held_lock(root, timeout=None) as release:
            def release_on_loop():
                facts["callback_while_held"] = not release.is_set()
                facts["sends_at_callback"] = len(sent)
                release.set()

            @contextlib.contextmanager
            def observed_lock(*args, **kwargs):
                stack = contextlib.ExitStack()
                try:
                    heartbeat = stack.enter_context(original(*args, **kwargs))
                except ledger.UsageLockUnavailable as exc:
                    if exc.reason == "contention" and "contention_thread" not in facts:
                        facts["contention_thread"] = threading.get_ident()
                        loop.call_soon_threadsafe(release_on_loop)
                    raise
                with stack:
                    yield heartbeat

            def release_if_stuck():
                watchdog_fired.set()
                release.set()

            monkeypatch.setattr(ua, "_locked", observed_lock)
            timer = threading.Timer(5, release_if_stuck)
            timer.start()
            try:
                # Await in this Task: its ContextVar owns the terminal capture.
                assert await ua.execute_physical_attempt_async(request(root), send) is response
                capture = ua.last_physical_attempt_capture()
                assert capture.state == "settled"
                assert ua._PHYSICAL_LIMIT.get().used == 1
                assert ua._PHYSICAL_LIMIT.get().claimed_ids == {capture.attempt_id}
            finally:
                release.set()
                timer.cancel()
                timer.join(3)
                assert not timer.is_alive()
            facts["watchdog_fired"] = watchdog_fired.is_set()
        assert sent == [1]
        attempt_rows = rows(root)
        assert [row["state"] for row in attempt_rows] == ["settled"]
        assert [row["attempt_id"] for row in attempt_rows] == [capture.attempt_id]

    asyncio.run(run())
    return facts


def _assert_loop_ran_during_contention(facts):
    # A later watchdog expiry during settlement cannot undo observed progress.
    assert ("contention_thread" in facts and facts.get("callback_while_held")
            and facts.get("sends_at_callback") == 0), (
        f"event loop did not run during accounting contention: {facts}")


def test_async_wait_keeps_loop_responsive_and_context_claim(root, short_acquisitions, monkeypatch):
    _assert_loop_ran_during_contention(_observe_async_accounting_wait(root, monkeypatch))


def test_async_wait_witness_accepts_watchdog_after_loop_release(root, short_acquisitions, monkeypatch):
    callbacks = []
    original_timer, original_account = threading.Timer, ua._account_response

    def record_timer(interval, callback, *args, **kwargs):
        callbacks.append(callback)
        return original_timer(interval, callback, *args, **kwargs)

    def account_after_watchdog(*args):
        # Accounting follows release and send. Fire the real cleanup callback
        # here to prove the ordering without another wall-clock sleep.
        callback, = callbacks
        callback()
        return original_account(*args)

    monkeypatch.setattr(threading, "Timer", record_timer)
    monkeypatch.setattr(ua, "_account_response", account_after_watchdog)
    facts = _observe_async_accounting_wait(root, monkeypatch)
    assert facts["watchdog_fired"]
    _assert_loop_ran_during_contention(facts)


def test_async_wait_witness_rejects_inline_blocking(root, short_acquisitions, monkeypatch):
    from ouroboros import _usage_wait

    async def inline(function, *args, on_cancel=None, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(_usage_wait, "presend_off_loop", inline)
    facts = _observe_async_accounting_wait(root, monkeypatch)
    assert facts["contention_thread"] == facts["loop_thread"]
    assert facts["watchdog_fired"]
    with pytest.raises(AssertionError, match="event loop did not run during accounting contention"):
        _assert_loop_ran_during_contention(facts)


def test_async_cancellation_joins_reservation_committed_at_the_boundary(root, monkeypatch):
    reserved, finish = threading.Event(), threading.Event()
    original = ua.reserve_attempt

    def delayed(req):
        result = original(req)
        reserved.set()
        assert finish.wait(3)
        return result

    monkeypatch.setattr(ua, "reserve_attempt", delayed)

    async def run():
        async def send():
            pytest.fail("sent cancelled request")

        with owner(root):
            operation = asyncio.create_task(ua.execute_physical_attempt_async(request(root), send))
            while not reserved.is_set():
                await asyncio.sleep(.005)
            operation.cancel()
            await asyncio.sleep(.03)
            assert not operation.done()  # mutator is still owned
            operation.cancel()  # repeated cancellation still cannot abandon it
            finish.set()
            with pytest.raises(asyncio.CancelledError):
                await operation
        assert [row["state"] for row in rows(root)] == ["released"]
    asyncio.run(run())


def test_real_loop_round_two_wait_keeps_tool_and_live_leaf(root, short_acquisitions, monkeypatch):
    import ouroboros.loop as loop
    from ouroboros import delegate_custody as custody
    from ouroboros.tools.registry import ToolRegistry
    from tests.test_loop_transport_wait import _loop_kwargs

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    from ouroboros.task_results import write_task_result

    # The real owner Pause consumer requires admitted task/root authority.
    write_task_result(root, "t-wait", "running", root_task_id="t-wait")
    registry = ToolRegistry(repo_dir=root, drive_root=root)
    registry._ctx.task_id = "t-wait"
    leaf = custody.RunCustody(run_id="accounting-live-leaf", task_id="t-wait", route_id="stub", model="stub")
    assert custody.record_started(root, leaf)
    releases, tools, sends = [], [], []
    monkeypatch.setattr(custody, "release_task_runs", lambda *a: releases.append(a))
    real_execute = registry.execute_result

    def execute(name, args):
        tools.append((name, args))
        return real_execute(name, args)

    monkeypatch.setattr(registry, "execute_result", execute)
    (root / "fact.txt").write_text("one tool result")
    with contextlib.ExitStack() as stack:
        def call(_llm, messages, model, _tools, effort, retries, logs, tid, round_idx,
                 event_queue, accumulated, *a, **kwargs):
            if round_idx == 2:
                assert len(tools) == 1
                release = stack.enter_context(held_lock(root))
                timer = threading.Timer(.15, release.set)
                timer.start()
                stack.callback(timer.join, 2)
            def send():
                assert not releases  # accounting wait must never cancel supervising custody
                sends.append(round_idx)
                return {"usage": {"cost": .001}}
            # The real loop owns the operation; the local provider stub uses the
            # same physical wrapper as API/review adapters.
            ua.execute_physical_attempt(request(root, task_id="t-wait", root_task_id="t-wait"), send)
            if round_idx == 1:
                return {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "read-once", "type": "function", "function": {
                        "name": "read_file", "arguments": json.dumps({"path": "fact.txt"})}}]}, .001
            return {"role": "assistant", "content": "finished"}, .001
        monkeypatch.setattr(loop, "call_llm_with_retry", call)
        # A physical adapter has a TaskModelWait in production. The fake
        # call_llm_with_retry above bypasses that adapter's operation binder.
        with owner(root, task={"id": "t-wait"}):
            from ouroboros.model_wait import current_model_wait
            registry._ctx.model_wait_context = current_model_wait()
            current_model_wait().tool_context = registry._ctx
            text, usage, trace = loop.run_llm_loop(**_loop_kwargs(root, registry, []))
    assert text == "finished"
    assert sends == [1, 2] and len(tools) == 1
    assert usage.get("reason_code") != "task_exception"
    assert [row["state"] for row in rows(root)] == ["settled"] * 2


def test_async_unowned_maintenance_wait_is_bounded(root, short_acquisitions, monkeypatch):
    from ouroboros import _usage_wait as wait
    monkeypatch.setattr(wait, "USAGE_LOCK_TIMEOUT_SEC", .06)
    async def run():
        with held_lock(root):
            with pytest.raises(ledger.UsageLockUnavailable):
                await ua.execute_physical_attempt_async(request(root), lambda: pytest.fail("sent"))
    asyncio.run(run())
    assert not rows(root)


def test_interactive_expiry_rejoins_real_loop_no_call_terminal(root, short_acquisitions, monkeypatch):
    import ouroboros.loop as loop
    from ouroboros import config
    from ouroboros.model_wait import propagate_model_control
    from ouroboros.tools.registry import ToolRegistry
    from tests.test_loop_transport_wait import _loop_kwargs

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setattr(config, "get_task_idle_timeout_sec", lambda: .06)
    calls = []
    def call(*args, **kwargs):
        calls.append(1)
        try:
            ua.execute_physical_attempt(request(root), lambda: pytest.fail("sent"))
        except Exception as exc:
            propagate_model_control(exc)  # real call_llm_with_retry's typed boundary
            raise
    monkeypatch.setattr(loop, "call_llm_with_retry", call)
    with owner(root, task={"_is_direct_chat": True}) as events, held_lock(root):
        text, usage, trace = loop.run_llm_loop(**_loop_kwargs(root, ToolRegistry(repo_dir=root, drive_root=root), []))
    assert calls == [1] and not rows(root)
    assert usage["reason_code"] == "accounting_wait_expired"
    assert usage["execution_status"] == "infra_failed"
    assert "Accounting access" in text
    waits = [item["data"] for item in list(events.queue)
             if item.get("data", {}).get("checkpoint_kind") == "usage_lock_wait"]
    assert [item["phase"] for item in waits] == ["entered", "ended"]
    assert all(item["owner_visible"] for item in waits)
    assert waits[-1]["elapsed_sec"] >= waits[0]["elapsed_sec"]
    assert trace["forced_finalization"]["control_reason"] == "accounting_wait_expired"


def test_unreadable_named_identity_is_not_positive_contention(root, monkeypatch):
    original = platform._lock_identity
    def identity(target):
        return original(target) if isinstance(target, int) else ()
    monkeypatch.setattr(platform, "_lock_identity", identity)
    with pytest.raises(ledger.UsageLockUnavailable) as error:
        with ledger._locked(root, timeout_sec=.01):
            pytest.fail("unreadable identity acquired")
    assert error.value.reason == "identity_unreadable"


def test_last_poll_name_release_race_recontends_and_sends_once(root, monkeypatch):
    ticks, facts = iter([0., 0., 1.]), {}
    def raced_open(path, flags, *args):
        if flags & os.O_CREAT:
            raise FileExistsError(errno.EEXIST, "held")
        raise FileNotFoundError(errno.ENOENT, "released before probe")
    with monkeypatch.context() as patch:
        patch.setattr(platform, "kernel_file_locks_enforced", lambda _: True)
        patch.setattr(platform.os, "open", raced_open)
        patch.setattr(platform, "time", SimpleNamespace(
            monotonic=lambda: next(ticks), sleep=lambda _: None))
        assert platform.acquire_exclusive_file_lock(root / "race.lock", timeout_sec=.25, outcome=facts) is None
    assert facts["reason"] == "contention"
    original, waits, sends = ua._locked, [], []
    @contextlib.contextmanager
    def acquire(*args, **kwargs):
        waits.append(1)
        if len(waits) == 1:
            raise ledger.UsageLockUnavailable("observed last-poll race", reason=facts["reason"])
        with original(*args, **kwargs) as beat:
            yield beat
    monkeypatch.setattr(ua, "_locked", acquire)
    with owner(root), ua.physical_attempt_limit(1):
        ua.execute_physical_attempt(request(root), lambda: sends.append(1) or {"usage": {}})
        assert ua._PHYSICAL_LIMIT.get().used == 1
    assert sends == [1]
    assert [row["state"] for row in rows(root)] == ["settled"]


def _churn_lock(root, start, stop, ready):
    ready.put(os.getpid())
    assert start.wait(5)
    for _ in range(30):
        if stop.is_set():
            break
        with ua._locked(root):
            time.sleep(.012)
        time.sleep(.003)


def _trace_lock_os_failures(monkeypatch, lock_path):
    """Observe this fixture's exact lock calls; never classify or retry them."""
    failures, probes = [], set()
    opened, read, closed, unlinked = os.open, os.read, os.close, os.unlink
    selected = lambda path: isinstance(path, (str, os.PathLike)) and os.fspath(path) == str(lock_path)

    def observe(stage, call, *args, **kwargs):
        try:
            return call(*args, **kwargs)
        except OSError as exc:
            native = getattr(exc, "winerror", None)
            failures.append({"stage": stage, "exception": type(exc).__name__, "repr": repr(exc),
                             "errno": exc.errno, "native_error": native,
                             "native_error_source": "exception.winerror" if native is not None else "unavailable"})
            raise

    def open_lock(path, flags, *args, **kwargs):
        if not selected(path):
            return opened(path, flags, *args, **kwargs)
        stage = "create_exclusive" if flags & os.O_CREAT else "probe_open"
        fd = observe(stage, opened, path, flags, *args, **kwargs)
        if not flags & os.O_CREAT:
            probes.add(fd)
        return fd

    def read_lock(fd, *args, **kwargs):
        return observe("probe_read", read, fd, *args, **kwargs) if fd in probes else read(fd, *args, **kwargs)

    def close_lock(fd, *args, **kwargs):
        try:
            return observe("probe_close", closed, fd, *args, **kwargs) if fd in probes else closed(fd, *args, **kwargs)
        finally:
            probes.discard(fd)

    def unlink_lock(path, *args, **kwargs):
        return observe("unlink", unlinked, path, *args, **kwargs) if selected(path) else unlinked(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", open_lock)
    monkeypatch.setattr(os, "read", read_lock)
    monkeypatch.setattr(os, "close", close_lock)
    monkeypatch.setattr(os, "unlink", unlink_lock)
    return failures


@pytest.mark.parametrize("cancel", [False, True])
def test_owned_process_churn_keeps_single_send_and_controls(root, cancel, monkeypatch):
    ctx = multiprocessing.get_context("spawn")
    start, stop, ready = ctx.Event(), ctx.Event(), ctx.Queue()
    children = [ctx.Process(target=_churn_lock, args=(root, start, stop, ready)) for _ in range(3)]
    sends, controls = [], []
    def control():
        controls.append(time.monotonic())
        return "cancelled" if cancel and len(controls) >= 3 else None
    ua.usage_projection(root)  # one store before the churning processes open it
    try:
        for child in children:
            child.start()
        assert len({ready.get(timeout=10) for _ in children}) == 3
        lock_failures = _trace_lock_os_failures(monkeypatch, root / "state" / ledger.LOCK_REL.name)
        start.set()
        with owner(root, control=control), ua.physical_attempt_limit(1):
            if cancel:
                with pytest.raises(PhysicalDispatchInterrupted):
                    try:
                        ua.execute_physical_attempt(request(root), lambda: sends.append(1))
                    except ledger.UsageLockUnavailable as exc:
                        raise AssertionError(
                            "Accounting acquisition failed before cancellation: "
                            f"reason={exc.reason!r}, error_number={exc.error_number!r}, "
                            f"lock_os_failures={lock_failures!r}"
                        ) from exc
            else:
                ua.execute_physical_attempt(request(root), lambda: sends.append(1) or {"usage": {}})
                assert ua._PHYSICAL_LIMIT.get().used == 1
        assert sends == ([] if cancel else [1])
        assert len(controls) >= 3
        assert max((b - a for a, b in zip(controls, controls[1:])), default=0) < 2
        if not cancel:
            assert [row["state"] for row in rows(root)] == ["settled"]
    finally:
        stop.set()
        start.set()
        for child in children:
            child.join(10)
            if child.is_alive():
                child.terminate()
                child.join(5)
            assert child.exitcode == 0
        ready.close()
        ready.join_thread()
