"""Task-session MCP lifecycle: concurrent first calls, an unanswered initialize, the call lock, a server that leaves.

Like ``tests/test_mcp_task_session.py``: every call goes through the real ToolRegistry, the loop's tool executor
and the real MCP SDK client against a real SDK server process (its fixture has the switches used here). The
steps are ordered by events, not by sleeping: the server's own log rows, a rendezvous inside the slot lookup
window, a spy on the call lock. Each test removes the server processes it started, even when it fails.
"""
from __future__ import annotations

import contextvars
import os
import signal
import sys
import threading
import time

import pytest

from ouroboros import mcp_task_session, owner_pause
from tests.test_batch4_producer_custody import _registry
from tests.test_invocation_ownership import claims, owner
from tests.test_mcp_task_session import (  # noqa: F401  (the autouse fixture must live in this module's namespace)
    _private_mcp,
    alive,
    body,
    call,
    eventually,
    fixture_server,
    restarted,
    rows,
    session_of,
    wire,
)

pytestmark = [
              pytest.mark.skipif(sys.platform == "win32", reason="asserts POSIX process death and signals")]


def switch(tmp_path, name):
    (tmp_path / name).write_text("", encoding="utf-8")


def served(result):
    assert result.status == "ok", result.text
    return body(result)


def generous(manager, settings):
    """A new server needs a moment to start: the next call gets a budget the earlier timeout steps did not."""
    manager.reconfigure({**settings, "MCP_TOOL_TIMEOUT_SEC": 30})


def in_thread(results, label, function, *args, **kwargs):
    """Run ``function`` as the tool executor's own worker would: on another thread, in a copy of the caller's context."""
    context = contextvars.copy_context()
    thread = threading.Thread(target=lambda: results.__setitem__(label, context.run(function, *args, **kwargs)),
                              name=f"lifecycle-{label}", daemon=True)
    thread.start()
    return thread


def held_escalation(monkeypatch):
    """The SDK's SIGTERM/SIGKILL escalation waits for this test: a shutdown stays pending by cause, not by clock.

    It is reached only when the server outlived the SDK's own stdin-close wait; ``release`` lets the real
    escalation run. Set ``release`` in ``finally`` so a failing test never strands the worker."""
    import anyio
    import mcp.client.stdio as sdk_stdio

    reached, release, real = threading.Event(), threading.Event(), sdk_stdio._terminate_process_tree

    async def held(process, *args, **kwargs):
        reached.set()
        await anyio.to_thread.run_sync(release.wait)
        return await real(process, *args, **kwargs)
    monkeypatch.setattr(sdk_stdio, "_terminate_process_tree", held)
    return reached, release


def reap(registry, log):
    """Close the holder, then ask only this fixture and its descendant to exit through their private file."""
    try:
        mcp_task_session.close_context_sessions(registry._ctx)
    finally:
        (log.parent / "teardown").touch()
        for session in list(mcp_task_session._LIVE):
            session.wait_closed(10)
        pids = {row["pid"] for row in rows(log, "start")} | {row["child"] for row in rows(log, "child")}
        assert eventually(lambda: not any(alive(pid) for pid in pids), 10), "a known fixture process did not exit"


class SpyLock:
    """A session's call lock that says when a second caller reaches it while another still holds it."""

    def __init__(self, inner):
        self.inner, self.contended = inner, threading.Event()

    def acquire(self, blocking=True, timeout=-1):
        if self.inner.locked():
            self.contended.set()
        return self.inner.acquire(blocking, timeout)

    def release(self):
        self.inner.release()


def test_two_first_calls_that_both_missed_the_slot_share_one_server(tmp_path, monkeypatch):
    registry, _, _ = _registry(tmp_path, monkeypatch)
    raw, log = fixture_server(tmp_path, task_session=True)
    wire(monkeypatch, registry, raw)
    real_init, window = mcp_task_session._Owned.__init__, threading.Barrier(2, timeout=2.0)

    def after_a_miss(self, ctx):  # only a caller that found no slot on the context gets here
        real_init(self, ctx)
        try:
            window.wait()  # until the other first caller has missed the slot too; a lookup that excludes it never does
        except threading.BrokenBarrierError:
            pass
    monkeypatch.setattr(mcp_task_session._Owned, "__init__", after_a_miss)
    results = {}
    try:
        with owner(registry):
            for thread in [in_thread(results, label, call, registry, "bump") for label in ("a", "b")]:
                thread.join(60)
        starts = rows(log, "start")
        assert len(results) == 2 and all(result.status == "ok" for result in results.values()), results
        assert len(starts) == 2, starts  # discovery, then ONE server for the task, not one per caller
        assert sorted(body(result)["count"] for result in results.values()) == [1, 2]
        mcp_task_session.close_context_sessions(registry._ctx)
        assert eventually(lambda: not any(alive(row["pid"]) for row in starts[1:]))  # nothing outlives the close
    finally:
        reap(registry, log)


def test_a_server_that_never_answers_initialize_is_shut_down_when_the_call_gives_up(tmp_path, monkeypatch):
    """The call's TOOL was never sent; its server's end is reported only once observed, and custody waits for it."""
    registry, _, _ = _registry(tmp_path, monkeypatch)
    raw, log = fixture_server(tmp_path, task_session=True)
    manager, settings = wire(monkeypatch, registry, raw, timeout=3)
    switch(tmp_path, "hang_init")  # after discovery: only the task session's process stays silent
    escalating, release = held_escalation(monkeypatch)
    try:
        with owner(registry):
            started = time.monotonic()
            first = call(registry, "bump", timeout=20)
            assert first.status == "timeout" and first.meta["mcp_call_not_sent"] is True, first
            assert time.monotonic() - started < 3 + 3 and not rows(log, "call")  # bounded; nothing was ever sent
            assert "transport have closed" not in first.text and "not confirmed yet" in first.text, first.text
            # The 3 s budget includes the server's own start: under load the process may log only after the call
            # gave up. Its shutdown is held below, so the row comes; wait for that event, not for a clock.
            assert eventually(lambda: rows(log, "hang_init")), "the session's server never started"
            [silent] = rows(log, "hang_init")
            session = session_of(registry)
            assert escalating.wait(10), "the SDK never reached its escalation"  # the shutdown is pending, held here
            assert session.wait_closed(0).startswith("unconfirmed") and alive(silent["pid"])
            pending = claims(tmp_path)  # the invocation keeps its MCP duty while that end is pending
            assert len(pending) == 1 and next(iter(pending.values()))["external_pending"], pending
            refused = call(registry, "bump", timeout=20)  # the process stays this context's: no second server
            assert refused.meta["mcp_call_not_sent"] is True and "has not confirmed its end" in refused.text
            assert len(rows(log, "hang_init")) == 1 and session_of(registry) is session
            release.set()
            assert session.wait_closed(15) == "closed" and eventually(lambda: not alive(silent["pid"]), 12)
            assert eventually(lambda: not claims(tmp_path)), claims(tmp_path)  # discharged by the confirmed end
            (tmp_path / "hang_init").unlink()
            generous(manager, settings)
            replaced = call(registry, "bump", timeout=20)  # a new server may start: nothing is left behind
            second = served(replaced)
            assert second["pid"] != silent["pid"] and second["count"] == 1
            assert restarted(replaced) == "the server did not finish starting within the call's time"
    finally:
        release.set()
        reap(registry, log)


@pytest.mark.parametrize("moment", ["before_lookup", "before_start"])
def test_a_server_disabled_while_its_first_call_prepares_leaves_no_usable_session(tmp_path, monkeypatch, moment):
    """The manager resolved an enabled server; the owner disables it before that call's session exists.

    ``before_lookup``: before the context's slot is read; ``before_start``: after the new session is
    registered, just before its worker would launch the server. Either way no usable holder remains."""
    registry, _, _ = _registry(tmp_path, monkeypatch)
    raw, log = fixture_server(tmp_path, task_session=True)
    manager, settings = wire(monkeypatch, registry, raw)
    disabled = []

    def disable():
        if not disabled:
            settings["MCP_SERVERS"] = [{**raw, "enabled": False}]
            manager.reconfigure(settings)
            disabled.append(True)
    real_lookup, real_start = mcp_task_session._session_for, mcp_task_session.TaskSession.start
    if moment == "before_lookup":
        monkeypatch.setattr(mcp_task_session, "_session_for",
                            lambda *args, **kwargs: disable() or real_lookup(*args, **kwargs))
    else:
        monkeypatch.setattr(mcp_task_session.TaskSession, "start", lambda self: disable() or real_start(self))
    try:
        with owner(registry):
            refused = call(registry, "bump")
        slot = vars(registry._ctx).get(mcp_task_session._SLOT)
        held = slot.sessions.get("fx") if slot else None
        assert disabled and manager.resolve_tool_name("mcp_fx__bump").status == "server_disabled"
        assert refused.status != "ok" and refused.meta["mcp_call_not_sent"] is True, refused
        assert held is None or not held.usable(), "a server disabled before its first launch stayed usable"
        assert len(rows(log, "start")) == 1 and not rows(log, "call")  # discovery only: nothing launched or sent
    finally:
        reap(registry, log)


def test_a_call_resolved_before_a_change_never_replaces_the_current_session(tmp_path, monkeypatch):
    registry, _, _ = _registry(tmp_path, monkeypatch)
    raw, log = fixture_server(tmp_path, task_session=True)
    manager, settings = wire(monkeypatch, registry, raw)
    changed = {**raw, "args": [*raw["args"], "--changed"]}
    real_lookup, resolved = mcp_task_session._session_for, {}

    def change_first(ctx, cfg, *rest, **kwargs):  # the call already holds the old configuration
        if "old" not in resolved:
            resolved["old"] = cfg
            settings["MCP_SERVERS"] = [changed]
            manager.reconfigure(settings)
        return real_lookup(ctx, resolved["old"] if resolved.pop("stale", False) else cfg, *rest, **kwargs)
    monkeypatch.setattr(mcp_task_session, "_session_for", change_first)
    try:
        with owner(registry):
            stale = call(registry, "bump")  # resolved before the change: refused, nothing started
            assert stale.status != "ok" and stale.meta["mcp_call_not_sent"] is True, stale
            assert len(rows(log, "start")) == 1 and not rows(log, "call")
            assert manager.refresh_server("fx")["ok"]  # the changed server's own listing
            current = served(call(registry, "bump"))
            resolved["stale"] = True  # a later call that still carries the configuration from before the change
            late = call(registry, "bump")
            assert late.status != "ok" and late.meta["mcp_call_not_sent"] is True, late
            assert session_of(registry).usable() and alive(current["pid"])  # the current session is untouched
            assert served(call(registry, "bump")) == {"pid": current["pid"], "count": 2}
    finally:
        reap(registry, log)


def test_a_reconfiguration_that_keeps_the_server_lets_its_first_call_start_the_session(tmp_path, monkeypatch):
    registry, _, _ = _registry(tmp_path, monkeypatch)
    raw, log = fixture_server(tmp_path, task_session=True)
    manager, settings = wire(monkeypatch, registry, raw)
    real_lookup = mcp_task_session._session_for

    def unrelated_change(*args, **kwargs):  # a new fingerprint; this server's session identity is unchanged
        manager.reconfigure({**settings, "MCP_TOOL_TIMEOUT_SEC": 25})
        return real_lookup(*args, **kwargs)
    monkeypatch.setattr(mcp_task_session, "_session_for", unrelated_change)
    try:
        with owner(registry):
            first = served(call(registry, "bump"))
            assert served(call(registry, "bump")) == {"pid": first["pid"], "count": 2}
            assert session_of(registry).usable() and len(rows(log, "start")) == 2
    finally:
        reap(registry, log)


def test_second_call_waits_at_the_lock_until_its_deadline_and_closing_ends_the_running_call(tmp_path, monkeypatch):
    registry, _, _ = _registry(tmp_path, monkeypatch)
    raw, log = fixture_server(tmp_path, task_session=True)
    manager, settings = wire(monkeypatch, registry, raw, timeout=20)
    results = {}
    try:
        with owner(registry):
            first = served(call(registry, "bump"))
            session = session_of(registry)
            spy = session._call_lock = SpyLock(session._call_lock)
            running = in_thread(results, "hang", call, registry, "hang", timeout=60)
            assert eventually(lambda: [row for row in rows(log, "call") if row["tool"] == "hang"])  # the server has it
            manager.reconfigure({**settings, "MCP_TOOL_TIMEOUT_SEC": 2})  # the second call's budget; same server identity
            started = time.monotonic()
            second = call(registry, "bump", timeout=15)
            assert spy.contended.is_set() and 1.5 < time.monotonic() - started < 8
            assert second.status == "error" and second.meta["mcp_call_not_sent"] is True
            assert "MCP_TASK_SESSION_UNAVAILABLE" in second.text and "still running" in second.text
            assert [row["tool"] for row in rows(log, "call")] == ["bump", "hang"]  # the second was never sent
            assert session.usable() and running.is_alive() and alive(first["pid"])  # the running call is untouched
            mcp_task_session.close_context_sessions(registry._ctx)  # the loop ends while that call runs
            running.join(30)
            ended = results["hang"]
            assert not running.is_alive() and ended.status == "error" and ended.meta["mcp_call_not_sent"] is False
            assert "outcome is unknown" in ended.text
            assert session.closure == "closed" and not alive(first["pid"])
            pending = claims(tmp_path)
            assert len(pending) == 1 and next(iter(pending.values()))["external_pending"]  # its effect stays unknown
    finally:
        reap(registry, log)


def test_pause_while_a_second_call_waits_at_the_lock_refuses_it_before_any_send(tmp_path, monkeypatch):
    registry, _, _ = _registry(tmp_path, monkeypatch)
    raw, log = fixture_server(tmp_path, task_session=True)
    wire(monkeypatch, registry, raw, timeout=30)
    results = {}
    try:
        with owner(registry):
            first = served(call(registry, "bump"))
            session = session_of(registry)
            spy = session._call_lock = SpyLock(session._call_lock)
            running = in_thread(results, "gate", call, registry, "gate", timeout=60)
            assert eventually(lambda: [row for row in rows(log, "call") if row["tool"] == "gate"])
            waiting = in_thread(results, "bump", call, registry, "bump", timeout=60)
            assert spy.contended.wait(10), "the second call never reached the lock"
            owner_pause.install_fence(tmp_path, "root", request_id="pause-at-the-lock")
            switch(tmp_path, "release")  # the running call now answers normally
            running.join(30)
            waiting.join(30)
            assert results["gate"].status == "ok" and body(results["gate"])["gate"] == "open"
            assert results["bump"].status == "blocked" and results["bump"].meta["mcp_call_not_sent"] is True
            assert [row["tool"] for row in rows(log, "call")] == ["bump", "gate"]  # the waiting call was never sent
            assert session.usable() and alive(first["pid"])  # Pause fences effects; it does not end the session
            assert not claims(tmp_path)
    finally:
        reap(registry, log)


def test_a_warm_owner_wait_keeps_the_session_and_the_next_call_uses_it(tmp_path, monkeypatch):
    from ouroboros import model_sleep, owner_wait

    registry, _, _ = _registry(tmp_path, monkeypatch)
    raw, log = fixture_server(tmp_path, task_session=True)
    wire(monkeypatch, registry, raw)
    ctx, woken = registry._ctx, []
    ctx.owner_wait_callback = lambda _ctx, checkpoint: woken.append(checkpoint["wait_id"]) or "owner_input"
    ctx.task_started_at = time.time() - 100.0
    try:
        with owner(registry) as waiter:
            ctx.model_wait_context = waiter
            first = served(call(registry, "bump"))
            assert model_sleep.request_sleep(ctx, model_sleep.selectors(ctx, wake_after_sec=1), "warm")["reason"] == "sleep_armed"
            owner_wait.wait_after_tools(ctx, [], {}, {}, 1, [], set())  # the real park, inside the loop
            assert woken and not waiter.closed and session_of(registry).usable()
            assert served(call(registry, "bump")) == {"pid": first["pid"], "count": 2}  # the same server, with its state
            assert len(rows(log, "start")) == 2
    finally:
        reap(registry, log)


def test_a_server_that_leaves_idle_or_during_a_call_is_replaced_by_the_next_call(tmp_path, monkeypatch):
    registry, _, _ = _registry(tmp_path, monkeypatch)
    raw, log = fixture_server(tmp_path, task_session=True)
    wire(monkeypatch, registry, raw)
    try:
        with owner(registry):
            first = served(call(registry, "bump"))
            idle = session_of(registry)
            os.kill(first["pid"], signal.SIGKILL)  # the server leaves while the session is idle
            assert eventually(lambda: not idle.usable(), 8) and idle.wait_closed(10) == "closed"
            replaced = call(registry, "bump")
            second = served(replaced)
            assert second["pid"] != first["pid"] and second["count"] == 1
            assert restarted(replaced) == "the server closed its connection"  # a fresh process is never silent
            running = session_of(registry)
            died = call(registry, "die", timeout=15)  # the server leaves during a call
            assert died.status == "error" and died.meta["mcp_call_not_sent"] is False, died  # outcome unknown, not resent
            assert restarted(died) is None  # the note went with the first answered call
            assert running is not idle and running.wait_closed(10) == "closed"
            replaced = call(registry, "bump")
            third = served(replaced)
            assert third["pid"] not in (first["pid"], second["pid"]) and third["count"] == 1
            assert restarted(replaced) == "the server closed its connection"
            assert [row["tool"] for row in rows(log, "call")].count("die") == 1
    finally:
        reap(registry, log)


def test_unconfirmed_shutdown_is_reported_as_such_and_starts_no_second_server(tmp_path, monkeypatch):
    registry, _, _ = _registry(tmp_path, monkeypatch)
    raw, log = fixture_server(tmp_path, task_session=True)
    manager, settings = wire(monkeypatch, registry, raw)
    # Ignores SIGTERM and lingers after stdin EOF: the SDK has to escalate to SIGKILL. It sends nothing unasked
    # (no pinger), so the gate below tests this shutdown, not a late message's (that is the farewell test's).
    switch(tmp_path, "stubborn")
    escalating, release = held_escalation(monkeypatch)  # the end stays unconfirmed until this test releases it
    try:
        with owner(registry):
            first = served(call(registry, "bump", timeout=15))  # a server's start is not the budget under test
            session = session_of(registry)
            manager.reconfigure({**settings, "MCP_TOOL_TIMEOUT_SEC": 2})  # the calls' budget; same server identity
            hung = call(registry, "hang", timeout=15)  # a timeout after the send: the holder closes the session
            assert hung.status == "timeout" and hung.meta["mcp_call_not_sent"] is False
            assert escalating.wait(10) and session.wait_closed(0).startswith("unconfirmed")  # still shutting down
            refused = call(registry, "bump", timeout=15)
            assert refused.meta["mcp_call_not_sent"] is True and "MCP_TASK_SESSION_UNAVAILABLE" in refused.text
            assert "has not confirmed its end" in refused.text and len(rows(log, "start")) == 2  # no second server
            assert "a later call may succeed" in refused.text  # still closing: not the loop-long refusal
            release.set()
            assert session.wait_closed(15) == "closed" and not alive(first["pid"])
            (tmp_path / "stubborn").unlink()
            generous(manager, settings)
            replaced = call(registry, "bump", timeout=15)
            later = served(replaced)
            assert later["pid"] != first["pid"] and later["count"] == 1
            assert restarted(replaced) == "a call did not answer in time"
    finally:
        release.set()
        reap(registry, log)


def test_a_server_message_after_the_session_ended_does_not_unconfirm_its_shutdown(tmp_path, monkeypatch):
    registry, _, _ = _registry(tmp_path, monkeypatch)
    raw, log = fixture_server(tmp_path, task_session=True)
    manager, settings = wire(monkeypatch, registry, raw)
    switch(tmp_path, "farewell")  # the server writes one last message once its stdin is closed
    try:
        with owner(registry):
            first = served(call(registry, "bump", timeout=15))  # a server's start is not the budget under test
            session = session_of(registry)
            manager.reconfigure({**settings, "MCP_TOOL_TIMEOUT_SEC": 2})  # the hang call's budget; same identity
            assert call(registry, "hang", timeout=15).status == "timeout"  # a timeout after the send closes the session
            assert session.wait_closed(15) == "closed" and not alive(first["pid"])
            generous(manager, settings)
            later = served(call(registry, "bump", timeout=15))  # so the next call is served, not refused as unconfirmed
            assert later["pid"] != first["pid"] and later["count"] == 1
    finally:
        reap(registry, log)


def test_a_server_message_during_the_shutdown_leaves_the_sdks_group_escalation_to_end_its_descendant(
        tmp_path, monkeypatch):
    """The session's own exit cannot cut the SDK's stdio shutdown short (its stream clones, ``_serve``).

    The server writes once after stdin EOF and lingers; that message reaches the SDK's reader while the shutdown
    waits for the process. It must wait there: the SDK still escalates to its process group, which also ends the
    server's descendant. A reader failing on that message would cancel the wait, skip the group escalation and
    kill only the leader, leaving the descendant alive."""
    registry, _, _ = _registry(tmp_path, monkeypatch)
    raw, log = fixture_server(tmp_path, task_session=True)
    wire(monkeypatch, registry, raw)
    for name in ("stubborn", "farewell", "child"):  # lingers after one late message; a child shares its group
        switch(tmp_path, name)
    escalating, release = held_escalation(monkeypatch)
    child = None
    try:
        with owner(registry):
            first = served(call(registry, "bump"))
            child = rows(log, "child")[-1]["child"]
            session = session_of(registry)
            mcp_task_session.close_context_sessions(registry._ctx)  # the loop's exit; the end stays held below
            assert escalating.wait(10), "a late server message cut the SDK's shutdown short"
            assert session.wait_closed(0).startswith("unconfirmed") and alive(first["pid"])
            release.set()
            assert session.wait_closed(15) == "closed"
            assert eventually(lambda: not alive(first["pid"]) and not alive(child), 12), "a group member survived"
    finally:
        release.set()
        reap(registry, log)


def test_a_session_that_ended_unconfirmed_leaves_its_server_unavailable_for_the_rest_of_the_loop(
        tmp_path, monkeypatch):
    """Bytes no UTF-8 reader accepts fail the SDK's reader on their own (not on a stream a shutdown closed): the
    worker ends without a confirmed shutdown, and nothing can confirm it later. The refusal says so, unlike the
    still-closing one, so the mind does not wait for an end that cannot come; no second process is started."""
    registry, _, _ = _registry(tmp_path, monkeypatch)
    raw, log = fixture_server(tmp_path, task_session=True)
    wire(monkeypatch, registry, raw)
    try:
        with owner(registry):
            served(call(registry, "bump"))
            session = session_of(registry)
            garbled = call(registry, "garble", timeout=15)
            assert garbled.status == "error" and garbled.meta["mcp_call_not_sent"] is False, garbled
            closure = session.wait_closed(15)
            assert closure.startswith("unconfirmed") and "still running" not in closure, closure
            refused = [call(registry, "bump") for _ in range(2)]  # the same answer each time: nothing changes it
            for result in refused:
                assert result.meta["mcp_call_not_sent"] is True and "ended without confirming" in result.text, result
                assert "for the rest of this task loop" in result.text and "may succeed" not in result.text
            assert len(rows(log, "start")) == 2  # discovery, then the one session: never a second server
    finally:
        reap(registry, log)


def test_a_moved_authors_pause_park_ends_its_server_before_its_scope_read_and_resume_starts_a_new_one(
        tmp_path, monkeypatch):
    """A moved Presence author's warm park under the owner's Pause holds until its task scope has no live member
    (``presence_continuation``, #2566); a task-session server carries that scope, so the park ends it first. It
    retires nothing: the resumed loop's next call starts a new server, as after a cold pause."""
    from types import SimpleNamespace

    from ouroboros import presence_carrier, presence_continuation
    from ouroboros.tool_custody import task_scope_blockers

    registry, _, _ = _registry(tmp_path, monkeypatch)
    raw, log = fixture_server(tmp_path, task_session=True)
    wire(monkeypatch, registry, raw)
    real, reads = presence_continuation._custody_blockers, []

    def scope_read(*args, **kwargs):  # the real read; bounded, so a regression fails here instead of holding
        reads.append((session_of(registry).usable(), real(*args, **kwargs)))
        return reads[-1][1] if len(reads) < 3 else []
    monkeypatch.setattr(presence_continuation, "_custody_blockers", scope_read)
    monkeypatch.setattr(presence_carrier, "_HOLD_POLL_SEC", 0.05)
    monkeypatch.setattr(presence_continuation, "_yield_conversation", lambda *_: False)  # this test lends nothing
    binding = SimpleNamespace(drive_root=tmp_path, task_id="root",
                              lease=SimpleNamespace(review_wait=lambda *_: "control:owner_resume"))
    try:
        with owner(registry):
            first = served(call(registry, "bump"))
            held = task_scope_blockers(tmp_path, "root")
            assert {"kind": "task_scope_process", "task_id": "root", "pid": first["pid"]} in held, held
            outcome = presence_continuation.presence_review_wait(binding, registry._ctx, {"reason": "owner_pause"}, [])
            # Timing assumption: this fixture server leaves on stdin EOF inside the park's bounded close wait
            # (MCP_TASK_SESSION_CLOSE_WAIT_SEC, 2 s), so the FIRST scope read already finds no live member. A slower
            # exit fails only this first-read check: the park itself keeps reading the scope until it is empty.
            assert outcome == "control:owner_resume" and reads[0] == (False, []), reads
            assert session_of(registry).closure == "closed" and not alive(first["pid"])
            replaced = call(registry, "bump")
            later = served(replaced)
        assert later["pid"] != first["pid"] and later["count"] == 1 and len(rows(log, "start")) == 3
        assert restarted(replaced) == "its author parked under the owner's Pause"
    finally:
        reap(registry, log)


def test_only_closed_stream_errors_leave_a_shutdown_confirmed():
    import anyio

    try:
        group = BaseExceptionGroup
    except NameError:  # Python < 3.11: the backport anyio itself raises
        from exceptiongroup import BaseExceptionGroup as group
    closed, broken = anyio.ClosedResourceError(), anyio.BrokenResourceError()
    only = mcp_task_session._closed_streams_only
    assert only(group("tg", [broken])) and only(group("tg", [closed, group("inner", [broken])]))
    assert not only(group("tg", [broken, RuntimeError("the shutdown itself failed")]))
    assert not only(broken) and not only(RuntimeError("x"))  # a bare error is not a task group's report
