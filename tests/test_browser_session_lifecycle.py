"""#3389: a builtin browser tab keeps living between tool calls, inside its old policy and lifecycle.

Default lane: the session owner's loop and bridge (``tools/browser_session.py``), and its request guard
over fake routes against real task rows, real Pause fences, a real Stop intent and a really contended
store in an isolated data root. Browser lane (``-m browser``, both engines): page requests reach the
server while no call runs, refusals keep their policy, a Pause stops new page requests and Resume
admits again, a task's end closes its current and retired sessions, and a Stop kill or a Panic of the
worker leaves neither a browser process nor a new request.
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
import os
import threading
import time
import types

import pytest

from ouroboros import browser_policy, owner_pause, store
from ouroboros.tools import browser, browser_session
from ouroboros.tools.registry import BrowserState, ToolContext
from tests._browser_fakes import close_generations, coro, install_async_api, wait_event

_PROBE = contextvars.ContextVar("browser_session_probe", default="unset")


class _Request:
    def __init__(self, url, method="GET", body=None, *, unreadable=False):
        self.url, self.method, self.resource_type = url, method, "fetch"
        self._body, self._unreadable = body, unreadable

    @property
    def post_data(self):
        if self._unreadable:  # Playwright decodes a body strictly: a binary one raises
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
        return self._body


class TargetClosedError(Exception):
    pass


class _Route:
    def __init__(self, request, *, fail=None, abort_fail=None):
        self.request, self.answers, self.tried = request, [], []
        self.fail, self.abort_fail = fail, abort_fail

    async def continue_(self):
        self.tried.append("continue")
        if self.fail is not None:
            raise self.fail
        self.answers.append("continue")

    async def abort(self):
        self.tried.append("abort")
        if self.abort_fail is not None:
            raise self.abort_fail
        self.answers.append("abort")


def _session(ctx):
    generation = BrowserState()
    owner = browser_session.owner_of(generation)
    owner.run(asyncio.sleep(0))
    return generation, owner, browser_session.request_guard(owner, ctx, generation)


def _between_calls(owner, awaitable):
    """A route event as Playwright delivers it between calls: on the session loop, in no command."""
    return contextvars.Context().run(asyncio.run_coroutine_threadsafe, awaitable, owner._loop)


@pytest.fixture
def task_ctx(tmp_path, monkeypatch):
    """A running task's own context over a real persisted row (the fence and Stop live on it)."""
    from ouroboros import config
    from ouroboros.task_results import write_task_result

    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(config, "DATA_DIR", data)
    write_task_result(data, "root", "running", root_task_id="root")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=data)
    ctx.task_id = ctx.root_task_id = "root"
    ctx.task_lifecycle_bound = True
    return ctx


# --- the owner loop and its bridge ----------------------------------------------------------

def test_an_empty_generation_starts_no_thread_and_closes_confirmed():
    owner = browser_session.BrowserOwner()
    assert owner.close() is True and owner.close() is True
    assert owner._thread is None
    with pytest.raises(browser_session.BrowserSessionClosed):
        owner.run(asyncio.sleep(0))  # refused at once; the coroutine is closed, never left unawaited


def test_a_command_sees_a_copy_of_its_callers_context_and_the_session_start_sees_none():
    owner = browser_session.BrowserOwner()
    token = _PROBE.set("caller")
    try:
        async def read_and_change():
            seen = _PROBE.get()
            _PROBE.set("changed on the loop")
            return seen, threading.get_ident()

        seen, ident = owner.run(read_and_change())
        assert seen == "caller" and ident != threading.get_ident()
        assert _PROBE.get() == "caller"  # a command's changes never flow back to its caller

        async def session_start():
            # What the session starts (Playwright's reader, every route and event callback) inherits nothing.
            reader = asyncio.get_running_loop().create_task(asyncio.sleep(0, result=_PROBE.get()))
            return _PROBE.get(), await reader

        assert owner.run(session_start(), clean=True) == ("unset", "unset")
        cell = types.SimpleNamespace(generation="old")  # the call's shared pin is one object

        async def move():
            cell.generation = "new"

        owner.run(move())
        assert cell.generation == "new"
    finally:
        _PROBE.reset(token)
        assert owner.close() is True


def test_the_bridge_refuses_its_own_loop_and_a_dead_loop_at_once():
    owner = browser_session.BrowserOwner()

    async def wait_on_itself():
        try:
            owner.run(asyncio.sleep(0))
        except RuntimeError as exc:
            run_refused = str(exc)
        try:
            owner.close()
        except RuntimeError as exc:
            return run_refused, str(exc)

    run_refused, close_refused = owner.run(wait_on_itself())
    assert "cannot wait for itself" in run_refused and "cannot wait for itself" in close_refused
    owner._loop.call_soon_threadsafe(owner._loop.stop)  # the loop dies under its owner
    owner._thread.join(5)
    started = time.monotonic()
    with pytest.raises(browser_session.BrowserSessionClosed):
        owner.run(asyncio.sleep(0))
    assert time.monotonic() - started < 1.0
    assert owner.close() is False  # nothing confirms what a dead loop held
    # The system hands an ended thread's identifier to a later thread (#3522: a close started after the loop was
    # joined): that caller is not the loop, so it hears the session closed and the first close's outcome.
    heard = {}

    def recycled():
        owner._thread._ident = threading.get_ident()  # the ended loop's number, as the system recycles it
        for name, call in (("run", lambda: owner.run(asyncio.sleep(0))), ("close", owner.close)):
            try:
                heard[name] = call()
            except BaseException as exc:  # noqa: BLE001 - the outcome is the subject
                heard[name] = exc

    caller = threading.Thread(target=recycled)
    caller.start()
    caller.join(10)
    assert isinstance(heard["run"], browser_session.BrowserSessionClosed) and heard["close"] is False, heard


def test_closing_settles_a_hung_command_and_a_concurrent_close_reports_the_first():
    owner = browser_session.BrowserOwner()
    started = threading.Event()
    outcome = {}

    async def hang():
        started.set()
        await asyncio.Event().wait()

    def command():
        try:
            owner.run(hang())
        except BaseException as exc:  # noqa: BLE001 - the settled outcome is the subject
            outcome["command"] = exc

    worker = threading.Thread(target=command)
    worker.start()
    assert started.wait(5)
    second = threading.Thread(target=lambda: outcome.setdefault("second", owner.close()))
    first = owner.close()
    second.start()
    second.join(10)
    worker.join(10)
    assert first is True and outcome["second"] is True
    assert isinstance(outcome["command"], browser_session.BrowserSessionClosed)
    assert not owner._thread.is_alive() and owner.commands == 0
    with pytest.raises(browser_session.BrowserSessionClosed):
        owner.run(asyncio.sleep(0))


def test_closing_never_moves_a_command_the_loop_already_settled(caplog):
    """The loop cancels a hung command and its caller wakes; when the close settles what is still pending before
    that caller has left the set, the command keeps the outcome it has: no future is moved twice."""
    owner = browser_session.BrowserOwner()
    started, armed, release, outcome = threading.Event(), threading.Event(), threading.Event(), {}
    lock = owner._lock

    class LeavingLate:  # the bridge's lock: the command's own thread leaves the pending set only on release
        def __enter__(self):
            if threading.current_thread().name == "command" and armed.is_set():
                assert release.wait(10)
            return lock.__enter__()

        def __exit__(self, *exc):
            return lock.__exit__(*exc)

    owner._lock = LeavingLate()

    async def hang():
        started.set()
        await asyncio.Event().wait()

    def command():
        try:
            owner.run(hang())
        except BaseException as exc:  # noqa: BLE001 - the settled outcome is the subject
            outcome["command"] = exc

    worker = threading.Thread(target=command, name="command")
    worker.start()
    assert started.wait(5)
    armed.set()
    with caplog.at_level(logging.DEBUG):
        assert owner.close() is True
        release.set()
        worker.join(10)
    assert isinstance(outcome["command"], browser_session.BrowserSessionClosed)
    assert not [record for record in caplog.records if record.levelno >= logging.ERROR], caplog.text


async def _never():
    await asyncio.Event().wait()


@pytest.mark.parametrize("context_close, driver_stop, confirmed", [
    ("slow", "stops", True),  # the driver's stop returned: its exit closed every browser it launched
    ("fails", "stops", True),
    ("closes", "hangs", False),  # a driver that did not confirm its stop: unknown, the generation stays counted
    ("closes", "fails", False),
    ("closes", "already gone", True),
    ("slow", None, False),  # without a driver every handle present must close
    ("already gone", None, True),
])
def test_a_close_is_confirmed_by_its_driver_stop(monkeypatch, context_close, driver_stop, confirmed):
    monkeypatch.setattr(browser_session, "_CLOSE_STEP_SEC", 0.2)
    generation = BrowserState()
    browser_session.owner_of(generation).run(asyncio.sleep(0))
    loop_thread, calls = generation._owner._thread, []

    def step(name, how):
        async def act():
            calls.append(name)
            if how in ("slow", "hangs"):
                await _never()
            if how == "fails":
                raise RuntimeError(f"{name}: protocol error")
            if how == "already gone":
                raise TargetClosedError("Target page, context or browser has been closed")
        return act

    generation._browser_context = types.SimpleNamespace(close=step("context", context_close))
    if driver_stop is not None:
        generation.pw_instance = types.SimpleNamespace(stop=step("driver", driver_stop))
    browser.cleanup_browser_handles(generation)
    assert calls == ["context"] + (["driver"] if driver_stop else [])  # each later step runs whatever came before
    assert getattr(generation, "_cleanup_done", False) is confirmed  # unconfirmed: it stays counted
    assert not loop_thread.is_alive() and generation._browser_context is None and generation.pw_instance is None


def test_concurrent_closes_elect_one_that_closes_what_the_generation_holds(monkeypatch):
    """A retired generation's queued close and the task's end may close it at once. The handles are read inside the
    elected close, so the caller that would hold them never loses to one holding none."""
    generation = BrowserState()
    owner = browser_session.owner_of(generation)
    owner.run(asyncio.sleep(0))
    closed = []
    generation.browser = types.SimpleNamespace(close=coro(lambda: closed.append("browser")))
    generation.pw_instance = types.SimpleNamespace(stop=coro(lambda: closed.append("driver")))
    reading, release, second_in = threading.Event(), threading.Event(), threading.Event()
    take, wait = browser._take_handles, owner._closed.wait

    def first_read_held(bs):  # the first reader holds what it read until released
        handles = take(bs)
        if not reading.is_set():
            reading.set()
            assert release.wait(10)
        return handles

    def waiting(*args):  # the second close waits for the first one's outcome
        second_in.set()
        return wait(*args)

    monkeypatch.setattr(browser, "_take_handles", first_read_held)
    monkeypatch.setattr(owner._closed, "wait", waiting)
    first = threading.Thread(target=browser.cleanup_browser_handles, args=(generation,))
    second = threading.Thread(target=lambda: (browser.cleanup_browser_handles(generation), second_in.set()))
    first.start()
    assert reading.wait(10)
    second.start()
    assert second_in.wait(10)  # the second close has finished, or waits for the first
    release.set()
    first.join(10)
    second.join(10)
    assert closed == ["browser", "driver"] and generation._cleanup_done is True
    assert not owner._thread.is_alive() and generation.browser is None and generation.pw_instance is None


@pytest.mark.parametrize("driver", ["stops", "hangs"])
def test_a_failed_setup_retires_its_generation_and_an_unconfirmed_close_counts_toward_the_bound(monkeypatch, driver):
    monkeypatch.setattr(browser_session, "_CLOSE_STEP_SEC", 0.2)
    starts, stops = [], []

    async def launch(**_kwargs):
        raise RuntimeError("launch failed")

    async def stop():
        stops.append(True)
        if driver == "hangs":
            await _never()

    playwright = types.SimpleNamespace(chromium=types.SimpleNamespace(launch=launch), devices={}, stop=stop)
    install_async_api(monkeypatch, playwright, starts)
    monkeypatch.setattr(browser, "_ensure_playwright_installed", lambda *, engine, **_kw: "/pinned/x")
    monkeypatch.setattr(browser, "_HAS_STEALTH", False)
    ctx = types.SimpleNamespace(browser_state=BrowserState(), task_constraint=None)
    failed = []
    for _attempt in range(4):
        generation = ctx.browser_state
        with pytest.raises(RuntimeError) as raised:
            browser._ensure_browser(ctx)
        failed.append((generation, str(raised.value)))
    if driver == "stops":
        # Each failed setup stopped the driver it started; its generation is retired, closed and out of the backlog.
        assert len(starts) == len(stops) == 4 and all(text == "launch failed" for _, text in failed)
        assert all(gen._retired and gen._cleanup_done and gen.pw_instance is None for gen, _ in failed)
        assert all(ctx.browser_state is not gen for gen, _ in failed) and browser._live_retired_generations(ctx) == []
    else:
        # An unconfirmed close stays counted: three are the bound, and the fourth call opens nothing.
        assert len(starts) == len(stops) == 3 and "BROWSER_BACKLOG_RETIRED_SESSIONS" in failed[3][1]
        assert [id(gen) for gen, _ in failed[:3]] == [id(gen) for gen in browser._live_retired_generations(ctx)]
    assert not any(thread.name == "browser-session" and thread.is_alive() for thread in threading.enumerate())


@pytest.mark.parametrize("start", ["finishes", "fails", "never finishes"])
def test_a_close_during_the_driver_start_stops_the_driver_that_start_spawned(monkeypatch, start):
    """A close owns the driver even when its handshake fails or never finishes; a late start cannot launch."""
    starting, closing, stops, launches, outcome = threading.Event(), threading.Event(), [], [], {}

    async def launch(**_kwargs):
        launches.append(True)

    async def stop():
        stops.append(True)

    playwright = types.SimpleNamespace(chromium=types.SimpleNamespace(launch=launch), devices={}, stop=stop)

    async def start_driver():  # the driver runs; its handshake answers only while a close waits for it
        starting.set()
        if start == "never finishes":
            await _never()
        assert await wait_event(closing, 10)
        if start == "fails":
            raise RuntimeError("Connection closed while reading from the driver")
        return playwright

    monkeypatch.setattr(browser_session, "PlaywrightDriver", lambda: types.SimpleNamespace(
        start=start_driver, stop=stop))
    real_close = browser_session.close_handles

    async def observed_close(*handles):
        closing.set()
        return await real_close(*handles)

    monkeypatch.setattr(browser_session, "close_handles", observed_close)
    monkeypatch.setattr(browser, "_ensure_playwright_installed", lambda *, engine, **_kw: "/pinned/x")
    monkeypatch.setattr(browser, "_HAS_STEALTH", False)
    ctx = types.SimpleNamespace(browser_state=BrowserState(), task_constraint=None)
    generation = ctx.browser_state

    def call():  # abandoned by its timeout while the start is in flight
        try:
            browser._ensure_browser(ctx)
        except BaseException as exc:  # noqa: BLE001 - the settled outcome is the subject
            outcome["call"] = exc

    worker = threading.Thread(target=call)
    worker.start()
    assert starting.wait(10)
    assert browser._detach_browser(ctx)[0] is generation  # the timeout's retirement
    queued = threading.Thread(target=browser.cleanup_browser_handles, args=(generation,))
    queued.start()
    browser.cleanup_browser(ctx)  # the task's end, concurrent with the queued close
    queued.join(30)
    worker.join(30)
    assert not generation._owner._thread.is_alive() and launches == []
    assert generation.pw_instance is None and generation._pw_driver is None
    assert stops == [True] and generation._cleanup_done is True
    assert browser._live_retired_generations(ctx) == []
    if start == "fails":
        # Closing may settle the command before its handshake failure is delivered.
        assert (isinstance(outcome["call"], browser_session.BrowserSessionClosed)
                or str(outcome["call"]) == "Connection closed while reading from the driver")
        return
    assert isinstance(outcome["call"], browser_session.BrowserSessionClosed)


# --- the request guard: one decision per request, between calls too ------------------------

def test_a_request_between_calls_is_admitted_by_one_fence_read_and_a_commands_by_none(task_ctx, monkeypatch):
    reads, budgets = [], []
    real = owner_pause.member_fence

    def counted(source, **kwargs):
        reads.append(source)
        budgets.append(store._WAIT_BUDGET.get())  # the off-loop read's own bound, not a copied one
        return real(source, **kwargs)

    monkeypatch.setattr(owner_pause, "member_fence", counted)
    generation, owner, guard = _session(task_ctx)
    try:
        idle = _Route(_Request("http://127.0.0.1:9/idle"))
        _between_calls(owner, guard(idle)).result(10)
        assert idle.answers == ["continue"] and reads == [task_ctx]
        from ouroboros._usage_wait import ACQUISITION_SLICE_SEC
        assert budgets == [ACQUISITION_SLICE_SEC]
        command = _Route(_Request("http://127.0.0.1:9/command"))
        owner.run(guard(command))  # reaches Python while a command of this generation runs
        assert command.answers == ["continue"] and reads == [task_ctx]
    finally:
        close_generations(generation)


def test_pause_refuses_new_requests_between_calls_while_an_admitted_one_finishes(task_ctx, monkeypatch):
    deciding, release = threading.Event(), threading.Event()
    real = browser_policy.browser_request_block_reason

    def slow(request, ctx, **kwargs):
        if request.url.endswith("/admitted"):
            deciding.set()
            assert release.wait(10)
        return real(request, ctx, **kwargs)

    monkeypatch.setattr(browser_policy, "browser_request_block_reason", slow)
    generation, owner, guard = _session(task_ctx)
    try:
        admitted = _Route(_Request("http://127.0.0.1:9/admitted"))
        pending = _between_calls(owner, guard(admitted))
        assert deciding.wait(10)  # its fence read found the tree open; its policy is still deciding
        owner_pause.install_fence(task_ctx.drive_root, "root", request_id="pause-1")
        assert owner.run(asyncio.sleep(0, result="free")) == "free"  # a slow decision never holds the loop
        refused = _Route(_Request("http://127.0.0.1:9/after-pause"))
        _between_calls(owner, guard(refused)).result(10)
        assert refused.answers == ["abort"]
        release.set()
        pending.result(10)
        assert admitted.answers == ["continue"]  # admitted before the fence closed: it finishes
        assert browser_session.take_block_notes(generation) == (
            "Between browser calls: OWNER_PAUSE_NOT_STARTED: a Pause of this task tree stopped a page "
            "request made between browser calls; it was not sent.")
        owner_pause.release_fence(task_ctx.drive_root, "root", reason="owner_resume")
        resumed = _Route(_Request("http://127.0.0.1:9/resumed"))
        _between_calls(owner, guard(resumed)).result(10)
        assert resumed.answers == ["continue"] and refused.answers == ["abort"]  # nothing replayed
    finally:
        release.set()
        close_generations(generation)


def test_a_commands_request_keeps_its_handed_operation_under_a_later_pause(task_ctx, monkeypatch):
    deciding, release = threading.Event(), threading.Event()
    real = browser_policy.browser_request_block_reason

    def slow(request, ctx, **kwargs):
        deciding.set()
        assert release.wait(10)
        return real(request, ctx, **kwargs)

    monkeypatch.setattr(browser_policy, "browser_request_block_reason", slow)
    generation, owner, guard = _session(task_ctx)
    route = _Route(_Request("http://127.0.0.1:9/of-the-command"))
    try:
        async def command_issuing_a_request():
            task = asyncio.get_running_loop().create_task(guard(route))
            await asyncio.sleep(0)  # the request reached Python while this command ran
            return task

        task = owner.run(command_issuing_a_request())  # the command ends; its request is still deciding
        assert deciding.wait(10) and owner.commands == 0
        owner_pause.install_fence(task_ctx.drive_root, "root", request_id="pause-2")
        release.set()

        async def join():
            await task

        owner.run(join())
        assert route.answers == ["continue"]  # its affiliation was read once, before the Pause
        assert browser_session.take_block_notes(generation) == ""
    finally:
        release.set()
        close_generations(generation)


def test_a_contended_store_refuses_between_call_requests_within_its_slice_and_never_a_command(task_ctx):
    from tests._store_journal_shared import exclusive_hold

    generation, owner, guard = _session(task_ctx)
    holder = exclusive_hold(store.store_path(task_ctx.drive_root))
    try:
        started = time.monotonic()
        idle = [_Route(_Request(f"http://127.0.0.1:9/idle-{index}")) for index in range(40)]
        pending = [_between_calls(owner, guard(route)) for route in idle]
        command = _Route(_Request("http://127.0.0.1:9/command"))
        owner.run(guard(command))  # its decision waits for none of the store-bound ones queued before it
        command_seconds = time.monotonic() - started
        for future in pending:
            future.result(10)
        all_seconds = time.monotonic() - started
    finally:
        holder.rollback()
        holder.close()
    try:
        print({"command_seconds": round(command_seconds, 3), "all_seconds": round(all_seconds, 3)})
        assert command.answers == ["continue"] and command_seconds < 0.5  # no fence read, no store wait
        assert all(route.answers == ["abort"] for route in idle)
        assert all_seconds < 6.0  # each read waits its slice, never the store's 10 s default
        notes = browser_session.take_block_notes(generation)
        assert notes == ("Between browser calls: BROWSER_POLICY_UNAVAILABLE: this task's Pause or Stop could not "
                         "be read (StoreBusy); a page request made between browser calls was not sent. (40 requests)")
        after = _Route(_Request("http://127.0.0.1:9/after"))
        _between_calls(owner, guard(after)).result(10)
        assert after.answers == ["continue"]  # the store answers again: nothing was cached either way
    finally:
        close_generations(generation)


def _tree(data, *members):
    """Rows of the tree "root": each member ``(task_id, parent)``, as the host writes a sub-task's row."""
    from ouroboros.task_results import write_task_result

    for task_id, parent in members:
        write_task_result(data, task_id, "running", root_task_id="root", parent_task_id=parent,
                          delegation_role="subagent")


@pytest.mark.parametrize("unreadable", ["pause", "stop", "an ancestor's stop"])
def test_unreadable_authority_refuses_a_request_between_calls(task_ctx, unreadable):
    if unreadable == "pause":
        task_ctx.task_id = task_ctx.root_task_id = "ghost"  # a bound task without a row: unknown, never open
        note = "this task's Pause could not be read"
    else:
        (task_ctx.drive_root / "state").mkdir(exist_ok=True)
        projection = "{not json"
        if unreadable == "an ancestor's stop":  # one malformed row: the intermediate card above this sub-task
            _tree(task_ctx.drive_root, ("mid", "root"), ("grandchild", "mid"))
            task_ctx.task_id, projection = "grandchild", '{"schema_version": 1, "intents": {"mid": "not a row"}}'
        (task_ctx.drive_root / "state" / "cancel_intents.json").write_text(projection, encoding="utf-8")
        note = "this task's Pause or Stop could not be read (CancelIntentProjectionCorrupt)"
    generation, owner, guard = _session(task_ctx)
    try:
        route = _Route(_Request("http://127.0.0.1:9/idle"))
        _between_calls(owner, guard(route)).result(10)
        assert route.answers == ["abort"]
        assert browser_session.take_block_notes(generation) == (
            f"Between browser calls: BROWSER_POLICY_UNAVAILABLE: {note}; a page request made between browser calls "
            "was not sent.")
    finally:
        close_generations(generation)


@pytest.mark.parametrize("member, stopped, scope, policy, refused", [
    ("root", "root", "", "immediate", True),  # the task's own immediate Stop
    ("root", "root", "", "finalize_then_cancel", False),  # Wrap up lets the task finish its work
    ("child", "root", "cascade", "immediate", True),  # its tree's cascade, before the sweep records the child's own
    ("child", "root", "cascade", "finalize_then_cancel", False),
    ("child", "root", "", "immediate", False),  # a Stop of the root alone does not stop its members
    ("retry leaf", "root", "cascade", "hardened", True),  # Stop now on a retry leaf hardens its logical root's cascade
    ("retry leaf", "root", "cascade", "finalize_then_cancel", False),
    # Stop now on a sub-task's card is a cascade over its own sub-tasks: it governs them before the sweep does.
    ("grandchild", "mid", "cascade", "immediate", True),
    ("grandchild", "mid", "cascade", "finalize_then_cancel", False),
    ("grandchild", "mid", "", "immediate", False),
    ("grandchild", "aunt", "cascade", "immediate", False),  # another branch's Stop
    ("grandchild", "below", "cascade", "immediate", False),  # a Stop under this task never governs it
    ("retry-child", "leaf", "cascade", "immediate", True),  # a cascade on a retry leaf governs what that attempt began
])
def test_an_immediate_stop_refuses_requests_between_calls_and_a_wrap_up_does_not(task_ctx, member, stopped, scope,
                                                                                   policy, refused):
    from ouroboros.cancel_intents import STOP_POLICY_FINALIZE, request_cancel
    from tests._cancel_intents_shared import _write_root_retry_pair

    data = task_ctx.drive_root
    if member in ("retry leaf", "retry-child"):
        _write_root_retry_pair(data, "root", "leaf", new_status="running")
    _tree(data, ("mid", "root"), ("grandchild", "mid"), ("aunt", "root"), ("below", "grandchild"),
          ("retry-child", "leaf"))
    task_ctx.task_id = {"retry leaf": "leaf"}.get(member, member)  # a member of the tree "root"
    generation, owner, guard = _session(task_ctx)
    try:
        if policy == "hardened":
            request_cancel(data, "root", reason="owner Stop", source="test", scope=scope,
                           requested_stop_policy=STOP_POLICY_FINALIZE)
            request_cancel(data, "leaf", reason="owner Stop now", source="test", requested_stop_policy="immediate")
        else:  # the owner's Stop as the gateway records it (_record_stop): before any sweep reaches the members
            request_cancel(data, stopped, reason="owner Stop", source="test", scope=scope, requested_stop_policy=policy)
        idle = _Route(_Request("http://127.0.0.1:9/idle"))
        _between_calls(owner, guard(idle)).result(10)
        command = _Route(_Request("http://127.0.0.1:9/command"))
        owner.run(guard(command))
        assert command.answers == ["continue"]  # an operation handed before the Stop finishes as before
        assert idle.answers == (["abort"] if refused else ["continue"])
        assert browser_session.take_block_notes(generation) == (
            "Between browser calls: OWNER_OPERATION_REFUSED: this task is stopping; a page request made between "
            "browser calls was not sent." if refused else "")
    finally:
        close_generations(generation)


def _row_reads(monkeypatch, busy=""):
    """Every task row the decision reads (``task_results.read_document``, under each loader); ``busy`` is contended."""
    from ouroboros import task_results
    from ouroboros.store import StoreBusy

    reads, real = [], task_results.read_document

    def read(root, task_id, **kwargs):
        reads.append(str(task_id))
        if str(task_id) == busy:
            raise StoreBusy("the store is busy", reason="contention")
        return real(root, task_id, **kwargs)

    monkeypatch.setattr(task_results, "read_document", read)
    return reads


@pytest.mark.parametrize("fault, error", [
    ("absent", "CancelIntentLineageIndeterminate"),  # the link between the member and the stopped card is missing
    ("unbindable", "ValueError"),  # its row does not bind to its own identity
    ("cycle", "CancelIntentLineageIndeterminate"),
    ("busy", "StoreBusy"),
    ("own row absent", "CancelIntentLineageIndeterminate"),
])
def test_an_unknown_link_of_the_parent_chain_refuses_while_a_cascade_above_may_govern(task_ctx, monkeypatch, fault,
                                                                                       error):
    """An immediate cascade is open on "mid"; whether it governs "leaf" depends on links that cannot be read. An
    unknown link is never read as "not an ancestor": the request is refused as unknown authority."""
    from ouroboros import task_store
    from ouroboros.cancel_intents import request_cancel

    data = task_ctx.drive_root
    _tree(data, ("mid", "root"), *{"absent": [("leaf", "sub")], "cycle": [("sub", "leaf"), ("leaf", "sub")],
                                   "own row absent": []}.get(fault, [("sub", "mid"), ("leaf", "sub")]))
    if fault == "unbindable":
        task_store.patch_document(data, "sub", lambda _row: {"task_id": "sub", "status": "", "root_task_id": "root"})
    task_ctx.task_id = "leaf"
    request_cancel(data, "mid", reason="owner Stop", source="test", scope="cascade", requested_stop_policy="immediate")
    _row_reads(monkeypatch, busy="sub" if fault == "busy" else "")
    generation, owner, guard = _session(task_ctx)
    try:
        route = _Route(_Request("http://127.0.0.1:9/idle"))
        _between_calls(owner, guard(route)).result(10)
        assert route.answers == ["abort"]
        assert browser_session.take_block_notes(generation) == (
            "Between browser calls: BROWSER_POLICY_UNAVAILABLE: this task's Pause or Stop could not be read "
            f"({error}); a page request made between browser calls was not sent.")
    finally:
        close_generations(generation)


@pytest.mark.parametrize("stops, reads, refused", [
    ([], [], False),  # no Stop at all: the fast path reads no row
    ([("root", "cascade", "immediate")], [], True),  # the tree root's cascade needs no row
    ([("mid", "", "immediate")], [], False),  # a single Stop and a Wrap up never govern a member: no walk
    ([("mid", "cascade", "finalize_then_cancel")], [], False),
    ([("mid", "cascade", "immediate")], ["grandchild"], True),  # its own row names the stopped card
    ([("aunt", "cascade", "immediate")], ["grandchild", "mid"], False),  # one row per hop up to the tree root
    ([("aunt", "cascade", "immediate"), ("below", "cascade", "immediate")], ["grandchild", "mid"], False),  # one walk
])
def test_the_guard_reads_the_parent_chain_only_while_an_immediate_cascade_may_govern(task_ctx, monkeypatch, stops,
                                                                                    reads, refused):
    from ouroboros.cancel_intents import request_cancel

    data = task_ctx.drive_root
    _tree(data, ("mid", "root"), ("grandchild", "mid"), ("aunt", "root"), ("below", "grandchild"))
    task_ctx.task_id = "grandchild"
    for stopped, scope, policy in stops:
        request_cancel(data, stopped, reason="owner Stop", source="test", scope=scope, requested_stop_policy=policy)
    seen = _row_reads(monkeypatch)
    generation, owner, guard = _session(task_ctx)
    try:
        route = _Route(_Request("http://127.0.0.1:9/idle"))
        _between_calls(owner, guard(route)).result(10)
        assert route.answers == (["abort"] if refused else ["continue"])
        assert seen == reads  # each a strict read of one row's parent link
    finally:
        close_generations(generation)


def test_the_owners_stop_on_a_sub_tasks_card_governs_its_sub_tasks_before_the_sweep(tmp_path, monkeypatch):
    """The app's Stop now on any card posts a cascade (task_control_menu.requestStop); on a sub-task with sub-tasks of
    its own the gateway records it on that card's task, an intermediate ancestor. Between that record and the sweep
    that records each member's own intent, a member's page request made between calls is already refused."""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros import cancel_intents
    from ouroboros.gateway.tasks import api_task_cancel
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor import queue, task_lifecycle, workers

    for name, value in (("DRIVE_ROOT", tmp_path), ("RUNNING", {}), ("persist_queue_snapshot", lambda reason="": None),
                        ("_emit_cancel_task_done", lambda *_a, **_k: None)):
        monkeypatch.setattr(queue, name, value)
    monkeypatch.setattr(workers, "WORKERS", {}, raising=False)
    monkeypatch.setattr(task_lifecycle, "CANCELLED_ROOT_FENCES", {}, raising=False)
    monkeypatch.setattr(task_lifecycle, "_ACTIVE_CASCADE_FENCES", {}, raising=False)
    write_task_result(tmp_path, "root", "running", root_task_id="root")
    _tree(tmp_path, ("mid", "root"), ("grandchild", "mid"))
    monkeypatch.setattr(queue, "PENDING", [{"id": task_id, "root_task_id": "root", "parent_task_id": parent}
                                           for task_id, parent in (("mid", "root"), ("grandchild", "mid"))])
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id, ctx.root_task_id, ctx.task_lifecycle_bound = "grandchild", "root", True
    generation, owner, guard = _session(ctx)
    drive, seen = queue.drive_cancel_intent_scope, {}

    def before_the_sweep(task_id):
        seen["intents"] = cancel_intents.active_intents(tmp_path)
        route = _Route(_Request("http://127.0.0.1:9/idle"))
        _between_calls(owner, guard(route)).result(10)
        seen["answers"] = route.answers
        return drive(task_id)

    monkeypatch.setattr(queue, "drive_cancel_intent_scope", before_the_sweep)
    app = Starlette(routes=[Route("/api/tasks/{task_id}/cancel", api_task_cancel, methods=["POST"])])
    app.state.drive_root = tmp_path
    try:
        with TestClient(app) as client:
            answer = client.post("/api/tasks/mid/cancel", json={"cascade": True, "stop_policy": "immediate"})
        assert answer.status_code in (200, 202), answer.text
        stop = seen["intents"]["mid"]
        assert stop["scope"] == "cascade" and cancel_intents.stop_policy(stop) == "immediate"
        assert set(seen["intents"]) == {"mid"}  # the sweep has not recorded the sub-task's own intent yet
        assert seen["answers"] == ["abort"]
        assert browser_session.take_block_notes(generation) == (
            "Between browser calls: OWNER_OPERATION_REFUSED: this task is stopping; a page request made between "
            "browser calls was not sent.")
        # The sweep then reached the sub-task, as before.
        assert load_task_result(tmp_path, "grandchild")["status"] == "cancelled" or cancel_intents.active_intent(
            tmp_path, "grandchild")
    finally:
        close_generations(generation)


def test_a_retired_session_starts_no_new_request_between_calls(task_ctx, monkeypatch):
    decisions = []
    real = browser_session._decide
    monkeypatch.setattr(browser_session, "_decide", lambda *args: decisions.append(args[2]) or real(*args))
    generation, owner, guard = _session(task_ctx)
    try:
        generation._retired = True
        idle = _Route(_Request("http://127.0.0.1:9/idle"))
        _between_calls(owner, guard(idle)).result(10)
        assert idle.answers == ["abort"] and decisions == []  # refused before any decision
        hung = _Route(_Request("http://127.0.0.1:9/of-the-hung-call"))
        owner.run(guard(hung))
        assert hung.answers == ["continue"] and decisions == [True]  # the hung call's own requests are decided
        generation._retired = False

        def retire_while_deciding(request, ctx, **kwargs):
            generation._retired = True
            return ""

        monkeypatch.setattr(browser_policy, "browser_request_block_reason", retire_while_deciding)
        late = _Route(_Request("http://127.0.0.1:9/late"))
        _between_calls(owner, guard(late)).result(10)
        assert late.answers == ["abort"]  # retired while it was decided: an old session starts nothing new
        notes = browser_session.take_block_notes(generation)
        assert notes == ("Between browser calls: BROWSER_SESSION_RETIRED: this browser session was retired; a page "
                         "request made after its last call was not sent. (2 requests)")
    finally:
        close_generations(generation)


def test_every_request_is_answered_and_a_failed_answer_is_kept(task_ctx):
    unknown = ("Between browser calls: BROWSER_REQUEST_OUTCOME_UNKNOWN: continuing a page request failed "
               "(RuntimeError); whether it was sent is unknown; the abort that followed ")
    generation, owner, guard = _session(task_ctx)
    try:
        failing = _Route(_Request("http://127.0.0.1:9/a"), fail=RuntimeError("protocol error"))
        _between_calls(owner, guard(failing)).result(10)
        # A failed continue is never retried: one abort follows, never left waiting; whether the request had
        # already gone out is unknown, so the note claims neither that it was sent nor that it was not.
        assert failing.tried == ["continue", "abort"] and failing.answers == ["abort"]
        closed = _Route(_Request("http://127.0.0.1:9/b"), fail=TargetClosedError("Target page has been closed"))
        _between_calls(owner, guard(closed)).result(10)
        assert closed.tried == ["continue"]  # the expected teardown race: no answer is owed to a closed target
        assert browser_session.take_block_notes(generation) == unknown + "was accepted"
        # Negative control: the abort fails too. Its failure is the note's; nothing calls the request aborted.
        both = _Route(_Request("http://127.0.0.1:9/c"), fail=RuntimeError("protocol error"),
                      abort_fail=ValueError("Route.abort: protocol error"))
        _between_calls(owner, guard(both)).result(10)
        assert both.tried == ["continue", "abort"] and both.answers == []
        assert browser_session.take_block_notes(generation) == unknown + "failed too (ValueError)"
        # A refused request is never continued: when its abort fails, its refusal stays its one note.
        refused = _Route(_Request("http://127.0.0.1:9/d"), abort_fail=ValueError("Route.abort: protocol error"))
        reason = "BROWSER_POLICY_UNAVAILABLE: a test refusal; it was not sent"
        _between_calls(owner, browser_session._answer(refused, reason, generation, between_calls=True)).result(10)
        assert refused.tried == ["abort"]
        assert browser_session.take_block_notes(generation) == "Between browser calls: " + reason
    finally:
        close_generations(generation)


def test_the_policy_reads_a_request_body_in_three_states(task_ctx):
    from ouroboros.server_process import record_service_binding

    record_service_binding(task_ctx.drive_root, "main", "127.0.0.1", 8765, pid=os.getpid())
    url = "http://127.0.0.1:8765/api/decisions"
    persist = '{"decision_id": "model_wait:1", "persist_role": true}'
    generation, owner, guard = _session(task_ctx)
    try:
        answers = {}
        for name, request in (("persist", _Request(url, "POST", persist)),
                              ("ordinary", _Request(url, "POST", '{"decision_id": "quiz:1"}')),
                              ("absent", _Request(url, "POST", None)),
                              ("unreadable", _Request(url, "POST", unreadable=True))):
            route = _Route(request)
            owner.run(guard(route))
            answers[name] = route.answers
        # An owner shape is refused at the actual endpoint; a body that exists but cannot be read is not
        # proof of an ordinary request; a genuinely absent body stays ordinary.
        assert answers == {"persist": ["abort"], "ordinary": ["continue"], "absent": ["continue"],
                           "unreadable": ["abort"]}
    finally:
        close_generations(generation)


def test_decisions_run_off_the_loop_in_an_empty_context(task_ctx, monkeypatch):
    seen = []

    def observe(request, ctx, **kwargs):
        seen.append((_PROBE.get(), store._WAIT_BUDGET.get(), threading.current_thread().name))
        return ""

    monkeypatch.setattr(browser_policy, "browser_request_block_reason", observe)
    generation, owner, guard = _session(task_ctx)
    token = _PROBE.set("the caller's")
    try:
        with store.acquisition_budget(0):
            owner.run(guard(_Route(_Request("http://127.0.0.1:9/command"))))
        (probe, budget, thread), = seen
        assert probe == "unset" and budget is None and thread.startswith("browser-decision_")
    finally:
        _PROBE.reset(token)
        close_generations(generation)


def test_refusal_notes_are_consumed_only_by_the_result_that_shows_them():
    generation = types.SimpleNamespace()
    browser_session.record_block(generation, "A: refused during the call")
    mark = browser_session.block_mark(generation)
    browser_session.record_block(generation, "B: refused between calls", between_calls=True)
    browser_session.record_block(generation, "B: refused between calls", between_calls=True)
    assert browser_session.take_block_since(generation, mark) == ""  # never replaces a call's own error
    assert browser_session.take_block_since(generation, 0) == "A: refused during the call"
    assert browser_session.take_block_notes(generation) == "Between browser calls: B: refused between calls (2 requests)"
    assert browser_session.take_block_notes(generation) == ""
    browser_session.record_block(generation, "C: after the last result", between_calls=True)
    assert browser_session.take_block_notes(generation) == "Between browser calls: C: after the last result"


# --- the real engines (browser lane) ----------------------------------------------------------

_DELAY_PAGE = b"""<!doctype html><meta charset="utf-8"><title>between calls</title><p id="state">loaded</p><script>
const out = window.__w = {events: []};
const ev = (k, v) => out.events.push({t: Math.round(performance.now()), k, v: v === undefined ? null : String(v)});
const q = new URLSearchParams(location.search), cross = q.get('cross') || '', gate = q.get('gate') || '';
// The server answers /gate once the test released it: after browse_page returned and no command ran. Everything
// below is the page's own, between calls; /settled says that each of its requests has been answered.
fetch('/gate?id=' + gate, {cache: 'no-store'}).then(() => {
  ev('delayed.start');
  let imported = false;
  setTimeout(() => { if (!imported) ev('bound.fired'); }, 5000);  // a page-time bound like a module loader's
  return Promise.allSettled([
    fetch('/late?tag=fetch', {cache: 'no-store'}).then(r => ev('fetch.done', r.status))
      .catch(e => ev('fetch.error', e)),
    import('/mod.js?v=' + Date.now()).then(m => { imported = true; ev('import.done', m.ok); })
      .catch(e => ev('import.error', e)),
    fetch('/api/owner/context-mode', {method: 'POST', headers: {'content-type': 'application/json'},
      body: '{"mode":"synthetic"}'}).then(r => ev('ownerpost.status', r.status)).catch(e => ev('ownerpost.error', e)),
    cross && fetch(cross + '/cross-ok', {cache: 'no-store'}).then(r => ev('cross.done', r.status))
      .catch(e => ev('cross.error', e))]);
}).then(() => fetch('/settled?id=' + gate, {cache: 'no-store'}));
</script>"""
_POLL_PAGE = b"""<!doctype html><meta charset="utf-8"><title>poll</title><script>
let n = 0; setInterval(() => fetch('/poll?n=' + (n++), {cache: 'no-store'}).catch(() => {}), 200);
</script>"""


class _Arrivals:
    """Two loopback listeners, one bound as this data root's ``main`` service, sharing one arrival log."""

    def __init__(self, data_root):
        import contextlib
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from ouroboros.server_entrypoint import bound_service_socket

        log, lock, gates = [], threading.Lock(), {}

        def gate(name):
            with lock:
                return gates.setdefault(name, threading.Event())

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                pass

            def _answer(self, body, kind="application/json"):
                path, _, query = self.path.partition("?")
                with lock:
                    log.append({"t": time.time(), "port": self.server.server_address[1], "method": self.command,
                                "path": path, "query": query})
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802 - stdlib callback name
                path = self.path.partition("?")[0]
                if path == "/delay.html":
                    return self._answer(_DELAY_PAGE, "text/html; charset=utf-8")
                if path == "/poll.html":
                    return self._answer(_POLL_PAGE, "text/html; charset=utf-8")
                if path == "/mod.js":
                    return self._answer(b"export const ok = 1;", "text/javascript")
                if path == "/gate":  # held until the test releases this page's gate
                    gate(self.path.partition("?id=")[2]).wait(60)
                return self._answer(b"{}")

            def do_POST(self):  # noqa: N802 - stdlib callback name
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                return self._answer(b'{"synthetic": true}')

        ThreadingHTTPServer.daemon_threads = True
        self._stack = contextlib.ExitStack()
        self.log, self._lock, self._gate, self._gates = log, lock, gate, gates
        plain = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        listener = self._stack.enter_context(bound_service_socket(data_root, "main", "127.0.0.1", 0))
        bound = ThreadingHTTPServer(("127.0.0.1", 0), Handler, bind_and_activate=False)
        bound.socket.close()
        bound.socket, bound.server_address = listener, listener.getsockname()
        bound.server_activate()
        self._servers = [plain, bound]
        self._threads = [threading.Thread(target=server.serve_forever, daemon=True) for server in self._servers]
        for thread in self._threads:
            thread.start()
        self.plain = f"http://127.0.0.1:{plain.server_address[1]}"
        self.bound = f"http://127.0.0.1:{bound.server_address[1]}"

    def since(self, moment, path=""):
        with self._lock:
            return [row for row in self.log if row["t"] >= moment and (not path or row["path"] == path)]

    def release(self, name, owner):
        """Open a page's gate, only while no command of its session runs: what it sends next is between calls."""
        assert owner.commands == 0
        self._gate(name).set()

    def settled(self, moment, seconds=30.0):
        """Wait outside Playwright until a released page reports every request it made answered."""
        deadline = time.monotonic() + seconds
        while not self.since(moment, "/settled") and time.monotonic() < deadline:
            time.sleep(0.05)
        return bool(self.since(moment, "/settled"))

    def close(self):
        with self._lock:
            gates = list(self._gates.values())
        for opened in gates:
            opened.set()
        for server in self._servers:
            server.shutdown()
            server.server_close()
        for thread in self._threads:
            thread.join(5)
        self._stack.close()


class _Lane:
    """The production stateful path: the registry, its bound submit wrapper and one sticky worker."""

    def __init__(self, data_root, ctx=None):
        from ouroboros import loop_tool_execution as execution
        from ouroboros.tools.registry import ToolRegistry

        self.execution = execution
        self.registry = ToolRegistry(repo_dir=data_root.parent, drive_root=data_root)
        if ctx is not None:
            self.registry.set_context(ctx)
        self.ctx = self.registry._ctx
        self.ctx.task_id = self.ctx.root_task_id = "root"
        self.ctx.messages = []
        self.logs = data_root / "logs"
        self.logs.mkdir(exist_ok=True)
        self.executor = execution.StatefulToolExecutor()
        self.calls = 0

    def call(self, name, args, timeout=120):
        import json

        self.calls += 1
        tc = {"id": f"call-{self.calls}", "function": {"name": name, "arguments": json.dumps(args)}}
        row = self.execution._execute_with_timeout(self.registry, tc, self.logs, timeout, "root",
                                                   stateful_executor=self.executor)
        return row["tool_result"]

    def close(self):
        try:
            self.executor.submit(browser.cleanup_browser, self.ctx).result(30)
        finally:
            self.executor.shutdown(wait=True)


@pytest.fixture
def lane(task_ctx, monkeypatch):
    import ouroboros.safety as safety

    pytest.importorskip("playwright.async_api", reason="Playwright is not installed")
    monkeypatch.setattr(safety, "check_safety", lambda *_a, **_k: (True, ""))
    server = _Arrivals(task_ctx.drive_root)
    lanes = []

    def open_lane(ctx=None):
        lanes.append(_Lane(task_ctx.drive_root, ctx))
        return lanes[-1]

    try:
        yield types.SimpleNamespace(server=server, open=open_lane, data=task_ctx.drive_root)
    finally:
        for opened in lanes:
            opened.close()
        server.close()


def _expected(engine, text):
    if "not already installed" in text or "Executable doesn't exist" in text:
        if engine in os.environ.get("OUROBOROS_EXPECT_BROWSER_ENGINES", "").split(","):
            pytest.fail(text[:500])
        pytest.skip(text[:300])


def _page_events(result):
    import json

    return {row["k"]: row for row in json.loads(result.producer_text)}


@pytest.mark.browser
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_page_requests_reach_the_server_between_calls_and_keep_their_policy(lane, engine):
    from ouroboros.contracts.task_constraint import TaskConstraint

    main = lane.open()
    opened = main.call("browse_page", {"url": lane.server.bound + "/delay.html?gate=main", "engine": engine})
    _expected(engine, opened.text)
    returned = time.time()
    lane.server.release("main", browser_session.owner_of(main.ctx.browser_state))
    # Outside Playwright: no browser call runs, the model is thinking; the page's requests are answered meanwhile.
    assert lane.server.settled(returned), lane.server.since(returned)
    during = {(row["method"], row["path"]) for row in lane.server.since(returned)}
    assert {("GET", "/late"), ("GET", "/mod.js")} <= during, during  # answered before the next call
    state = main.call("browser_action", {"action": "evaluate", "value": "JSON.stringify(window.__w.events)"})
    events = _page_events(state)
    # Its import settled inside the page's own 5 s bound, so that bound can no longer fire.
    assert {"fetch.done", "import.done", "ownerpost.error"} <= set(events) and "bound.fired" not in events, events
    assert not lane.server.since(0, "/api/owner/context-mode")  # the owner shape never left the browser
    assert state.host_annotations[0] == ("Between browser calls: BROWSER_OWNER_CONTROL_BLOCKED: this operation "
                                         "belongs to the owner")
    # A restricted actor on another origin: its request to the actual control endpoint is refused between
    # calls too, while its own delayed requests continue.
    restricted = ToolContext(repo_dir=lane.data.parent, drive_root=lane.data, task_constraint=TaskConstraint(
        mode="acting_subagent", surface="external_workspace", write_root=str(lane.data.parent)))
    child = lane.open(restricted)
    opened = child.call("browse_page", {"url": f"{lane.server.plain}/delay.html?gate=child&cross={lane.server.bound}",
                                        "engine": engine})
    returned = time.time()
    lane.server.release("child", browser_session.owner_of(child.ctx.browser_state))
    assert lane.server.settled(returned), lane.server.since(returned)
    during = {(row["method"], row["path"]) for row in lane.server.since(returned)}
    assert ("GET", "/late") in during and ("POST", "/api/owner/context-mode") in during, during
    assert ("GET", "/cross-ok") not in during
    state = child.call("browser_action", {"action": "evaluate", "value": "JSON.stringify(window.__w.events)"})
    assert "cross.error" in _page_events(state)
    assert state.host_annotations[0].startswith("Between browser calls: BROWSER_LOCAL_READONLY_BLOCKED: ")
    assert not lane.server.since(0, "/cross-ok")


@pytest.mark.browser
def test_route_callbacks_never_carry_a_calls_context_while_its_command_does(lane, monkeypatch):
    from ouroboros.model_wait import dispatch_deadline_remaining_sec

    def call_context():
        return {"pin": browser._browser_call.get(), "deadline": dispatch_deadline_remaining_sec(),
                "operation": owner_pause._TOOL_OPERATION.get(), "handed": owner_pause._HANDED_TOOL.get(),
                "call_end": owner_pause._CALL_END.get(), "store_budget": store._WAIT_BUDGET.get()}

    seen, commands = [], []
    facts, output = browser_session._request_facts, browser._page_output

    def observed_facts(request):
        seen.append((request.url, call_context()))
        return facts(request)

    async def observed_output(page, kind):
        commands.append(call_context())
        return await output(page, kind)

    monkeypatch.setattr(browser_session, "_request_facts", observed_facts)
    monkeypatch.setattr(browser, "_page_output", observed_output)
    main = lane.open()
    opened = main.call("browse_page", {"url": lane.server.plain + "/delay.html?gate=context"})
    _expected("chromium", opened.text)
    returned = time.time()
    lane.server.release("context", browser_session.owner_of(main.ctx.browser_state))
    assert lane.server.settled(returned)  # the page's own requests arrive between calls
    assert any("/delay.html" in url for url, _ in seen) and any("/late" in url for url, _ in seen)
    empty = {"pin": None, "deadline": None, "operation": None, "handed": None, "call_end": None, "store_budget": None}
    assert all(context == empty for _, context in seen), seen  # during the call and between calls alike
    (command,) = commands  # the command itself runs in a copy of its caller's context
    assert command["pin"] is not None and command["deadline"] is not None and command["handed"] is not None


@pytest.mark.browser
def test_a_pause_stops_new_page_requests_until_resume_and_nothing_is_replayed(lane):
    main = lane.open()
    opened = main.call("browse_page", {"url": lane.server.plain + "/poll.html"})
    _expected("chromium", opened.text)
    returned = time.time()
    time.sleep(1.0)
    assert len(lane.server.since(returned, "/poll")) >= 3  # the page keeps working between calls
    owner_pause.install_fence(lane.data, "root", request_id="pause-1")
    time.sleep(0.5)  # a request admitted before the fence may still be finishing
    quiet = time.time()
    time.sleep(1.5)
    assert lane.server.since(quiet, "/poll") == []  # no new effect while the tree is paused
    refused = main.call("browser_action", {"action": "evaluate", "value": "1"})
    assert refused.code == "OWNER_PAUSE_NOT_STARTED"  # tool calls stay fenced as before
    owner_pause.release_fence(lane.data, "root", reason="owner_resume")
    resumed = time.time()
    time.sleep(1.0)
    after = [int(row["query"].removeprefix("n=")) for row in lane.server.since(resumed, "/poll")]
    assert len(after) >= 3
    seen = [int(row["query"].removeprefix("n=")) for row in lane.server.since(0, "/poll")]
    assert seen == sorted(seen) and len(set(seen)) == len(seen)  # each poll once, the refused ones never again
    assert min(after) - max(n for n in seen if n < min(after)) > 3  # the Pause's polls left a gap
    note = main.call("browser_action", {"action": "evaluate", "value": "1"}).host_annotations[0]
    assert note.startswith("Between browser calls: OWNER_PAUSE_NOT_STARTED: a Pause of this task tree stopped")
    assert note.endswith(" requests)")


def _descendants(pid):
    import subprocess

    out = subprocess.run(["ps", "-axo", "pid=,ppid="], capture_output=True, text=True, check=True).stdout
    children = {}
    for line in out.splitlines():
        child, parent = map(int, line.split())
        children.setdefault(parent, []).append(child)
    found, stack = [], [pid]
    while stack:
        for child in children.get(stack.pop(), ()):
            found.append(child)
            stack.append(child)
    return found


def _alive(pids):
    alive = []
    for pid in pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        except PermissionError:
            pass
        alive.append(pid)
    return alive


def _settle(pids, seconds=15.0):
    deadline = time.monotonic() + seconds
    while _alive(pids) and time.monotonic() < deadline:
        time.sleep(0.1)
    return _alive(pids)


_POSIX_TREE = pytest.mark.skipif(os.name == "nt", reason="reads the process tree with ps; Windows owns its tree by Job")


@pytest.mark.browser
@_POSIX_TREE
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_the_end_of_a_task_closes_its_current_and_retired_sessions(lane, engine):
    from ouroboros.loop_budget import _cleanup_loop_resources, _LoopExitContext

    main = lane.open()
    failed = main.ctx.browser_state
    with pytest.raises(ValueError, match="Unknown Playwright device"):
        browser._ensure_browser(main.ctx, engine=engine, device="missing-device-for-setup-control")
    assert failed._cleanup_done and not failed._owner._thread.is_alive()
    other = "webkit" if engine == "chromium" else "chromium"
    opened = main.call("browse_page", {"url": lane.server.plain + "/poll.html", "engine": other})
    _expected(other, opened.text)
    assert f"Browser conditions: {other} " in opened.text
    replaced = main.ctx.browser_state
    opened = main.call("browse_page", {"url": lane.server.plain + "/poll.html", "engine": engine})
    _expected(engine, opened.text)
    assert replaced._cleanup_done and not replaced._owner._thread.is_alive()
    assert f"Browser conditions: {engine} " in opened.text
    retired = main.ctx.browser_state
    hung = main.call("browser_action", {"action": "evaluate", "value": "while (true) {}"}, timeout=3)
    assert hung.code == "TOOL_TIMEOUT" and main.ctx.browser_state is not retired and retired._retired
    main.call("browse_page", {"url": lane.server.plain + "/poll.html", "engine": engine})
    current = main.ctx.browser_state
    processes = _descendants(os.getpid())
    assert processes and not getattr(retired, "_cleanup_done", False)  # its hung call never settled
    _cleanup_loop_resources(main.executor, _LoopExitContext(
        main.registry, None, "", None, main.logs, {}, {}))
    deadline = time.monotonic() + 30
    while not (getattr(retired, "_cleanup_done", False) and getattr(current, "_cleanup_done", False)):
        assert time.monotonic() < deadline, "the end of the task did not close both sessions"
        time.sleep(0.1)
    assert _settle(processes) == []  # drivers and browsers of both sessions are gone
    # A closed session's decision pools are shut down without waiting, so an idle worker may still be taking its stop:
    # each gets the conftest's bounded grace for a stopped thread; one still alive after it is a leak.
    deadline = time.monotonic() + 2.0
    sessions = [thread for thread in threading.enumerate()
                if thread.name.startswith(("browser-session", "browser-decision", "browser-idle-decision"))]
    for thread in sessions:
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
    assert not [thread.name for thread in sessions if thread.is_alive()]
    ended = time.time()
    time.sleep(1.0)
    assert lane.server.since(ended, "/poll") == []


@pytest.mark.browser
@_POSIX_TREE
@pytest.mark.parametrize("handshake", ["slow", "hung", "fails", "connecting"])
def test_the_end_of_a_task_during_a_real_driver_start_stops_that_driver(lane, monkeypatch, handshake):
    """Hold/fail the handshake only AFTER this generation's real subprocess exists; the task end must
    close that exact process, even when start never returns a Playwright object. No process census."""
    from playwright.async_api import _context_manager
    from ouroboros.loop_budget import _cleanup_loop_resources, _LoopExitContext
    from ouroboros.platform_layer import process_start_time

    held, release, spawned = threading.Event(), threading.Event(), []
    if hasattr(_context_manager, "PlaywrightClient"):
        from playwright._impl.connection import Connection
        target, method = Connection, "initialize_playwright"
    else:
        from playwright._impl._connection import RootChannelOwner
        target, method = RootChannelOwner, "initialize"
    initialize = getattr(target, method)

    async def held_initialize(self):
        if threading.current_thread().name == "browser-session":
            transport = main.ctx.browser_state._pw_driver._transport
            process = transport._proc
            assert process.returncode is None
            birth = process_start_time(process.pid)
            assert birth
            spawned.append((process, birth))
            held.set()
            if handshake == "fails":
                transport.on_error_future.set_exception(RuntimeError("handshake failed after spawn"))
            assert await wait_event(release, 60)
        return await initialize(self)

    monkeypatch.setattr(target, method, held_initialize)
    if handshake == "connecting":
        # The OS child and its pipes exist, but asyncio has not yet returned its process
        # handle to Playwright. Cancelling this spawn must let asyncio close and reap it.
        from asyncio.base_subprocess import BaseSubprocessTransport
        connect_pipes = BaseSubprocessTransport._connect_pipes

        async def held_connect_pipes(transport, waiter):
            if threading.current_thread().name != "browser-session":
                return await connect_pipes(transport, waiter)
            deferred = types.SimpleNamespace(cancelled=waiter.cancelled, set_exception=waiter.set_exception,
                                             set_result=lambda value: None)
            await connect_pipes(transport, deferred)
            process = transport._proc
            birth = process_start_time(process.pid)
            assert birth and process.returncode is None
            spawned.append((process, birth))
            held.set()
            assert await wait_event(release, 60)
            if not waiter.done():
                waiter.set_result(None)

        monkeypatch.setattr(BaseSubprocessTransport, "_connect_pipes", held_connect_pipes)
    main = lane.open()
    generation = main.ctx.browser_state
    result = main.call("browse_page", {"url": lane.server.plain + "/poll.html"}, timeout=5)
    assert held.is_set() and len(spawned) == (2 if handshake == "fails" else 1), result.text
    if handshake == "fails":
        assert "handshake failed after spawn" in result.text
    else:
        assert result.code == "TOOL_TIMEOUT" and generation._retired, result.text
    opener = None
    if handshake == "slow":
        opener = threading.Thread(target=lambda: (time.sleep(1.0), release.set()))
        opener.start()
    try:
        _cleanup_loop_resources(main.executor, _LoopExitContext(main.registry, None, "", None, main.logs, {}, {}))
        browser.cleanup_browser(main.ctx)  # final task cleanup also sees the retired generation
        assert generation._cleanup_done is True
        assert not generation._owner._thread.is_alive() and not generation._owner._pending
        for process, birth in spawned:  # browse_page's existing one navigation retry also closes its failed start
            assert process.returncode is not None  # actual wait, before any test/process-container reap
            assert process_start_time(process.pid) != birth
        assert browser._live_retired_generations(main.ctx) == []
        assert lane.server.since(0, "/poll.html") == []  # no abandoned navigation was replayed
    finally:
        release.set()
        if opener is not None:
            opener.join(5)


_WORKER = r'''
import json, os, pathlib, sys, time
sys.path.insert(0, sys.argv[1])
from ouroboros.platform_layer import create_new_session
create_new_session()  # like supervisor/worker_process.worker_main: the worker leads its own group
from ouroboros.tools import browser
from ouroboros.tools.registry import ToolContext
data = pathlib.Path(sys.argv[2])
ctx = ToolContext(repo_dir=data.parent, drive_root=data)
ctx.task_id = ctx.root_task_id = "root"
ctx.task_lifecycle_bound = True
text = browser._browse_page(ctx, sys.argv[3], engine=sys.argv[4])
print(json.dumps({"pid": os.getpid(), "text": text[:300]}), flush=True)
time.sleep(600)  # between calls: the next model round never comes
'''


@pytest.mark.browser
@_POSIX_TREE
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
@pytest.mark.parametrize("path", ["stop", "panic"])
def test_a_stop_or_panic_of_the_worker_leaves_no_browser_and_no_new_request(lane, engine, path):
    import json
    import pathlib
    import subprocess
    import sys

    from ouroboros.cancel_intents import request_cancel
    from supervisor.worker_pool_lifecycle import kill_worker_tree

    copy = str(pathlib.Path(__file__).resolve().parents[1])
    worker = subprocess.Popen([sys.executable, "-I", "-c", _WORKER, copy, str(lane.data),
                               lane.server.plain + "/poll.html", engine], stdout=subprocess.PIPE, text=True)
    try:
        line = worker.stdout.readline()
        assert line, "the synthetic worker did not open its browser"
        ready = json.loads(line)
        _expected(engine, ready["text"])
        opened = time.time()
        time.sleep(1.0)
        assert len(lane.server.since(opened, "/poll")) >= 3  # its page works while the worker waits
        processes = _descendants(worker.pid)
        assert processes  # the driver and the browser of this worker
        if path == "stop":
            request_cancel(lane.data, "root", reason="owner Stop", source="test")  # recorded before the kill
            time.sleep(0.5)  # a request admitted before the Stop may still be finishing
            stopping = time.time()
            time.sleep(1.0)
            assert lane.server.since(stopping, "/poll") == []  # nothing new between the Stop and the kill
            kill_worker_tree(worker.pid, keep_services=True)  # the cancel custody's kill
        else:
            kill_worker_tree(worker.pid, panic_process=worker)  # the Panic request for this worker
        worker.wait(10)
        killed = time.time()
        assert _settle(processes) == [], "a browser process of the stopped worker survived"
        time.sleep(1.0)
        assert lane.server.since(killed, "/poll") == []
    finally:
        if worker.poll() is None:
            worker.kill()  # the exact process this test started
            worker.wait(10)
        worker.stdout.close()
