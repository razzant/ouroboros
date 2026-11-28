"""The supervisor's display and diagnostic projections leave its loop: one host duty writes them.

On 27.11.2026 owner intake stopped twice for display work: 134 s inside the ``os.replace`` of
``state.json`` that recorded ANOTHER duty's overrun, and 145.6 s inside the ``os.stat`` of
``usage.sqlite`` that refreshed a root's cost badge. The loop now only marks and snapshots what is
due; the ``display-projections`` duty (``ouroboros/display_projections.py``) reads and writes. Here,
against the REAL supervisor loop (tests/test_server_shutdown.py's harness), each seam is held in turn
and the loop keeps taking owner input and stops on Stop; released, exactly the latest applicable
result lands. The budget mark's write-once, owed-until-written and retry rules, a start that fails,
a look that raises, the exit's one bounded last pass (the restart census's, before the store switch) and
a typed partial write are pinned at the duty's own seams, both ways.
"""

from __future__ import annotations

import collections
import errno
import json
import logging
import queue
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros import display_projections, server_maintenance, update_generation_facts
from ouroboros.cost_projection import LIVE_ROOT_COSTS


def _wait_until(predicate, budget: float = 10.0) -> None:
    end = time.time() + budget
    while not predicate() and time.time() < end:
        time.sleep(0.01)


def _pass_ended() -> None:
    """The duty's latch is free and its watchdog entry closed: the pass and its bookkeeping are over."""
    _wait_until(lambda: not display_projections.LATCH.locked()
                and id(display_projections.LATCH) not in server_maintenance._DUTIES)
    assert not display_projections.LATCH.locked(), "the display-projection pass never ended"


def _loop_turns_while_held(monkeypatch, tmp_path, holding, mark, install=lambda: None, turns=4):
    """The REAL ``server._run_supervisor`` for ``turns`` turns, the last a Stop. Each turn's maintenance look
    calls ``mark(ctx, turn)`` (what that turn's events marked) and dispatches the display projections as the
    production look does; every turn after the first takes owner input only once the pass the first turn
    started stands in its held seam, and records whether it still stands. Returns the loop's event context and
    those intake records (thread name, held)."""
    import server
    from tests.test_server_shutdown import _run, _supervisor_harness

    rec = _supervisor_harness(monkeypatch, tmp_path, ["ok"] * (turns - 1) + ["stop"])
    install()
    contexts, taken = [], []

    def maintenance(*_args, ctx=None, **_kwargs):
        contexts.append(ctx)
        mark(ctx, len(contexts))
        server_maintenance.dispatch_display_projections(ctx)

    def intake(_bridge, offset, _ctx, **_kwargs):
        if contexts:
            _wait_until(holding)
        taken.append((threading.current_thread().name, holding()))
        return offset

    monkeypatch.setattr(server, "_periodic_supervisor_maintenance", maintenance)
    monkeypatch.setattr(server, "_process_bridge_updates", intake)
    _run(rec, server)
    assert len(contexts) == turns and rec.stop.is_set()
    return contexts[0], taken


def _held_seam():
    held, release = threading.Event(), threading.Event()

    def hold() -> None:
        if not held.is_set():
            held.set()
            assert release.wait(10), "the test must release the held seam"

    return held, release, hold, (lambda: held.is_set() and not release.is_set())


def test_the_loop_serves_owner_input_and_stop_while_the_budget_state_replace_stands(monkeypatch, tmp_path):
    """The compatibility write's state replace stands (its first call: 134 s on 27.11): every later turn
    takes owner input and the Stop ends the loop with the pass still standing. Released, the stood write
    lands, and the marks of the turns that ran meanwhile cost ONE more write, of the latest ledger."""
    from supervisor import state as state_mod

    held, release, hold, holding = _held_seam()
    ledger, written = [0], []

    def write(_usage):  # the ledger render, then the replace of state.json and its last-good copy
        seen = ledger[0]
        hold()
        written.append((seen, threading.current_thread().name))
        return True

    def mark(ctx, turn):
        ledger[0] = turn  # this turn's llm_usage reached the ledger and marked the projection
        ctx.budget_projection_dirty = True

    ctx, taken = _loop_turns_while_held(
        monkeypatch, tmp_path, holding, mark,
        install=lambda: monkeypatch.setattr(state_mod, "update_budget_from_usage", write))
    loop = threading.current_thread().name
    try:
        assert taken[1:] == [(loop, True)] * 3, taken
        assert display_projections.LATCH.locked() and written == [], "the Stop did not wait for the pass"
        assert ctx.budget_projection_dirty is True, "the later marks wait for the next pass, untaken"
    finally:
        release.set()
        _pass_ended()
    assert written == [(1, display_projections.DUTY_NAME)]
    server_maintenance.dispatch_display_projections(ctx)  # the next turn's look
    _pass_ended()
    assert written == [(1, display_projections.DUTY_NAME), (4, display_projections.DUTY_NAME)]
    assert ctx.budget_projection_dirty is False and ctx.budget_projection_retry_at == 0.0


def test_the_loop_serves_owner_input_and_stop_while_a_root_figure_identity_stat_stands(monkeypatch, tmp_path):
    """The root figure's ledger read stands in the usage store's identity stat (145.6 s on 27.11): the loop
    goes on, the root marked meanwhile is read again by the next pass, and the figure the root's heartbeat
    carries is the latest read, for the carrier that still runs it: the stood read, which those marks
    overtook, is still newer than the none shown and lands, one read behind them, until the next pass's."""
    from ouroboros import usage_accounting
    from supervisor import workers as workers_mod

    held, release, hold, holding = _held_seam()
    ledger, reads = [0], []
    task = {"id": "root", "type": "task", "root_task_id": "root"}

    def breakdown(_root, **_kwargs):  # usage_store.read -> store.file_identity, where os.stat stood
        seen = ledger[0]
        hold()
        reads.append((seen, threading.current_thread().name))
        return {"accounted_usd": float(seen), "attempt_counts": {"settled": 1}, "priced_rows": 1,
                "tracked_nonfinal_rows": 0, "accounting_open_rows": 0}

    def mark(ctx, turn):
        ledger[0] = turn
        if turn == 1:
            LIVE_ROOT_COSTS.fields(ctx.DRIVE_ROOT, "root", task)  # the root's first heartbeat: its figure is due
        else:
            LIVE_ROOT_COSTS.mark(ctx.DRIVE_ROOT, "root")  # this turn's usage of the root's subtree

    def install():
        workers_mod.RUNNING["root"] = {"task": task}  # the harness's RUNNING map, which the loop's context holds
        monkeypatch.setattr(usage_accounting, "usage_breakdown", breakdown)

    ctx, taken = _loop_turns_while_held(monkeypatch, tmp_path, holding, mark, install=install)
    try:
        assert taken[1:] == [(threading.current_thread().name, True)] * 3, taken
        assert display_projections.LATCH.locked() and reads == []
        assert LIVE_ROOT_COSTS.fields(ctx.DRIVE_ROOT, "root", task) == {}, "nothing published before the read ends"
    finally:
        release.set()
        _pass_ended()
    figure = LIVE_ROOT_COSTS.fields(ctx.DRIVE_ROOT, "root", task)
    assert figure["accounted_upper_bound_usd_with_children"] == 1.0, "a read the later marks overtook lands"
    server_maintenance.dispatch_display_projections(ctx)  # those marks kept the root due
    _pass_ended()
    assert reads == [(1, display_projections.DUTY_NAME), (4, display_projections.DUTY_NAME)]
    assert LIVE_ROOT_COSTS.fields(ctx.DRIVE_ROOT, "root", task)["accounted_upper_bound_usd_with_children"] == 4.0


def test_the_loop_serves_owner_input_and_stop_while_the_after_update_state_replace_stands(monkeypatch, tmp_path):
    """Recording that another duty overran once wrote ``state.json`` on the loop (134 s on 27.11). Now the
    overrun is counted and captured on the loop, its rows wait in order, and the after-update publication
    stands on the duty: the loop goes on, overruns found meanwhile are counted exactly, and the next pass
    publishes the latest counts once, after writing every row in order."""
    held, release, hold, holding = _held_seam()
    published, rows = [], []

    def publish_after():  # update_progress._store -> update_state -> the replace that stood
        counts = update_generation_facts.sample()["duty_overruns"]
        hold()
        published.append(counts)

    def mark(_ctx, turn):  # the watchdog found one more duty past its cadence this turn
        duty = {"name": f"duty-{turn}", "ident": None, "since": time.time() - 61, "alerted": False}
        display_projections.journal_duty_stall(duty, 60.0, [f"frame-{turn}"], {})
        display_projections.journal_duty_stall_end(duty)

    def install():
        monkeypatch.setattr(update_generation_facts, "_duties", {})
        monkeypatch.setattr(update_generation_facts, "_publish_after", publish_after)
        monkeypatch.setattr("supervisor.state.append_jsonl", lambda _path, row: rows.append(row) or True)

    _ctx, taken = _loop_turns_while_held(monkeypatch, tmp_path, holding, mark, install=install)
    try:
        assert taken[1:] == [(threading.current_thread().name, True)] * 3, taken
        assert update_generation_facts._duties == {f"duty-{turn}": 1 for turn in range(1, 5)}, "counted on detection"
        assert published == []
    finally:
        release.set()
        _pass_ended()
    assert published == [{"duty-1": 1}]
    server_maintenance.dispatch_display_projections(None)  # the next turn's look: rows and a newer count wait
    _pass_ended()
    assert published == [{"duty-1": 1}, {f"duty-{turn}": 1 for turn in range(1, 5)}]
    assert [(row["type"], row["duty"]) for row in rows] == [
        (kind, f"duty-{turn}") for turn in range(1, 5) for kind in ("host_duty_stall", "host_duty_stall_end")]
    assert all(row["stack"] == [row["duty"].replace("duty", "frame")] for row in rows if row["type"] == "host_duty_stall")


@pytest.mark.parametrize("ending", ["restart_turn", "restart_during_intake", "stop_during_intake"])
def test_a_turn_a_stop_or_restart_ends_leaves_its_marks_to_the_exits_last_pass(monkeypatch, tmp_path, ending):
    """The turn whose drain marked the budget can end before its maintenance look dispatches: a restart seen
    right after the drain, or a restart or Stop arriving during intake (the look returns first). The loop
    starts no pass on its way out (a pass alive at the restart's store switch held its stores); the exit's
    last pass (``drain``) takes the mark from the context the loop bound, off the loop, once, and nothing
    starts after it."""
    import server
    from supervisor import state as state_mod
    from tests.test_server_shutdown import _run, _supervisor_harness

    rec = _supervisor_harness(monkeypatch, tmp_path, ["restart"] if ending == "restart_turn" else ["ok"])
    writes = []
    monkeypatch.setattr(state_mod, "update_budget_from_usage",
                        lambda usage: writes.append(threading.current_thread().name) or True)

    def drain(_queue, ctx, *_args, **_kwargs):
        ctx.budget_projection_dirty = True  # the turn drained one llm_usage event
        return False

    def intake(_bridge, offset, _ctx, **_kwargs):
        if ending != "restart_turn":
            (rec.restart if ending == "restart_during_intake" else rec.stop).set()
        return offset

    def maintenance(*_args, ctx=None, **_kwargs):  # production's look returns before its dispatch once stopping
        if not (rec.stop.is_set() or rec.restart.is_set()):
            server_maintenance.dispatch_display_projections(ctx)

    monkeypatch.setattr(server, "drain_worker_events", drain)
    monkeypatch.setattr(server, "_process_bridge_updates", intake)
    monkeypatch.setattr(server, "_periodic_supervisor_maintenance", maintenance)
    _run(rec, server)
    ctx = display_projections._CONTEXT
    assert ctx is not None and not display_projections.LATCH.locked() and writes == [], "no pass on the way out"
    assert ctx.budget_projection_dirty is True, "the turn's mark waits for the exit"
    assert display_projections.drain(time.monotonic() + 10) is True
    last = f"{display_projections.DUTY_NAME}-drain"
    assert writes == [last] and ctx.budget_projection_dirty is False
    ctx.budget_projection_dirty = True  # a mark after the last pass
    server_maintenance.dispatch_display_projections(ctx)
    assert display_projections.drain(time.monotonic() + 10) is True
    assert writes == [last] and ctx.budget_projection_dirty is True, "nothing starts after the last pass"


@pytest.mark.parametrize("first", ["census", "teardown"])
@pytest.mark.parametrize("stands", [True, False], ids=["pass-stands", "pass-ends"])
def test_the_restart_census_takes_the_last_pass_before_the_store_switch(tmp_path, monkeypatch, stands, first):
    """The restart's census (``server._emergency_process_cleanup``) comes after the lifespan teardown's last display
    pass, or beside it while that teardown is still running (Uvicorn slow to exit: ``teardown``): it waits for that
    drain, inside half the stop grace left, and reads whether a pass still runs. Where no teardown took the last pass
    (``census``) it waits for the pass in flight and takes it with what is outstanding (here the turn's newer budget
    mark). Either way before ``store_switch.switch_at_restart``: a pass that still stands at that bound is in the
    census, so the switch refuses rather than hand off stores that pass reads, or let it write ``state.json`` after
    the switch; a pass that ended lets the switch run. No pass starts after the census either way."""
    import server
    from ouroboros import server_control, store_switch

    root = tmp_path / "data"
    root.mkdir()
    grace = [None]
    monkeypatch.setattr("ouroboros.owned_shutdown.stop_deadline", lambda: grace[0])
    monkeypatch.setattr(server, "DATA_DIR", root)
    monkeypatch.setattr(server, "_restart_requested", threading.Event())
    monkeypatch.setattr(server, "begin_owned_stop", lambda *_, **__: None)
    monkeypatch.setattr(server, "stop_owned_work", lambda *_: {"state": "completed", "unconfirmed": []})
    monkeypatch.setattr("ouroboros.startup_historical_audit.stop_background_children", lambda: [])
    monkeypatch.setattr("supervisor.workers.kill_workers", lambda **_: None)
    monkeypatch.setattr("supervisor.workers.last_worker_exit_census", lambda: {"unconfirmed": []})
    monkeypatch.setattr("multiprocessing.active_children", lambda: [])
    monkeypatch.setattr("ouroboros.extension_companion.panic_kill_all", lambda: None)
    held, release, hold, _holding = _held_seam()
    writes = []

    def write(_usage):  # the in-flight pass stands here; the last pass's write goes straight through
        hold()
        writes.append(threading.current_thread().name)
        return True

    ctx = SimpleNamespace(DRIVE_ROOT=root, RUNNING={}, budget_projection_dirty=True, update_budget_from_usage=write)
    display_projections.bind(ctx)
    server_maintenance.dispatch_display_projections(ctx)  # a tick's pass, then the turn's newer mark
    assert held.wait(10)
    ctx.budget_projection_dirty = True
    releaser = threading.Timer(0.3, release.set)
    teardown = threading.Thread(target=server_control.finish_display_and_socket_lines, name="lifespan-teardown")
    if not stands:
        releaser.start()
    try:
        started = time.monotonic()
        grace[0] = started + 3.0
        if first == "teardown":  # the lifespan's end began the last pass first, on its own thread
            teardown.start()
            _wait_until(display_projections._CLOSED.is_set)
        cleanup = server._emergency_process_cleanup()
        waited = time.monotonic() - started
        assert cleanup["display_pass_alive"] is stands
        assert waited < 3.0 and (waited >= 1.4 if stands else ctx.budget_projection_dirty is False)
        monkeypatch.setattr(store_switch, "switch_needs", lambda *a: {"handles": False, "money": False, "facts": False})
        monkeypatch.setattr(store_switch, "rev_parse", lambda *a: "example-target")
        switches = []
        monkeypatch.setattr(store_switch, "switching_owner",
                            lambda *_: lambda intent: (switches.append(intent) or True, "switched"))
        store_switch.record_switch_intent(root, target_sha="example-target", target_ref="example", reason="test",
                                          door="ui_rollback")
        outcome = store_switch.switch_at_restart(root, root, cleanup)
        if stands:
            assert (outcome["status"], outcome["reason"]) == ("refused", "quiescence_unconfirmed")
            assert outcome["blockers"] == ["display_pass_alive"] and switches == []
        else:
            assert outcome["status"] == "switched" and len(switches) == 1
    finally:
        release.set()
        releaser.cancel()
        if teardown.is_alive():
            teardown.join(10)
        _pass_ended()
    duty, last = display_projections.DUTY_NAME, f"{display_projections.DUTY_NAME}-drain"
    assert writes == ([duty] if stands else [duty, last])
    assert ctx.budget_projection_dirty is stands, "a pass that stood left the newer mark to the exit's loss"
    ctx.budget_projection_dirty = True
    server_maintenance.dispatch_display_projections(ctx)
    assert not display_projections.LATCH.locked() and len(writes) == (1 if stands else 2), "nothing after the census"


class _GatedSink(logging.Handler):
    """A log sink whose writes wait for ``gate``, as a stalled disk would; the handler's own lock is left alone."""

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.entered, self.gate, self.lines = threading.Event(), threading.Event(), []

    def emit(self, record) -> None:
        self.entered.set()
        if self.gate.wait(10):
            self.lines.append(record.getMessage())


def test_each_lifespan_of_an_imported_host_takes_its_last_pass_and_hands_its_socket_lines_over(
        tmp_path, monkeypatch, request):
    """An imported ASGI host runs the app's lifespan and gets control back: no exit, no restart census follows. Each
    lifespan's teardown, right after its supervisor stopped and its terminal custody was written, takes its own last
    display pass (here the budget mark of a final turn its loop bound, which no look took) and hands the socket lines
    still queued to their sink, which is held when the teardown begins, inside half of the stop grace left; only then
    does the lifespan return. The next lifespan in the same interpreter finds the duty open with no context of the
    last one: its looks start passes and its own end takes its own last pass (here inside what is left of the
    process's one stop grace, ``owned_shutdown``)."""
    from starlette.testclient import TestClient

    import server as srv
    from ouroboros import owned_shutdown, process_logging
    from tests.test_extensions_api import _patch_lifespan_for_drive_root_test

    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    monkeypatch.setattr(srv.app.app.state, "drive_root", drive_root)
    monkeypatch.setattr(srv.app.app.state, "repo_dir", tmp_path / "repo")
    _patch_lifespan_for_drive_root_test(monkeypatch, srv, {}, request)
    monkeypatch.setattr(srv, "_boot_managed_update_tasks", lambda: None)
    monkeypatch.setattr("ouroboros.config.get_skills_repo_path", lambda: "")
    monkeypatch.setattr("supervisor.workers.kill_workers", lambda **_kw: None)
    grace_end = time.monotonic() + 60
    monkeypatch.setattr(owned_shutdown, "stop_deadline", lambda: grace_end)
    logger, sink = logging.getLogger("tests.lifespan_end.socket_lines"), _GatedSink()
    logger.addHandler(sink)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handoff = process_logging.RecordHandoff("test-lifespan-end-socket-lines")
    real_finish = process_logging.finish_record_handoffs

    def finish(deadline):  # the teardown's wait for the socket lines: the sink comes back while it waits
        threading.Timer(0.3, sink.gate.set).start()
        return real_finish(deadline)

    monkeypatch.setattr(process_logging, "finish_record_handoffs", finish)
    writes = []

    def context():  # what a lifespan's supervisor binds before its loop
        def write(_usage):
            writes.append(threading.current_thread().name)
            return True

        ctx = SimpleNamespace(DRIVE_ROOT=drive_root, RUNNING={}, budget_projection_dirty=False,
                              update_budget_from_usage=write)
        display_projections.bind(ctx)
        return ctx

    duty, last = display_projections.DUTY_NAME, f"{display_projections.DUTY_NAME}-drain"
    try:
        with TestClient(srv.app):
            first = context()
            first.budget_projection_dirty = True  # its final turn's llm_usage; no look came after it
            handoff.log(logger, logging.INFO, "first line")
            handoff.log(logger, logging.INFO, "second line")
            assert sink.entered.wait(5) and sink.lines == []
        assert writes == [last] and first.budget_projection_dirty is False, "the teardown wrote the bound mark"
        assert sink.lines == ["first line", "second line"], "the teardown waited for the queued lines"
        assert handoff.thread is None and display_projections._CLOSED.is_set()
        with TestClient(srv.app):
            assert not display_projections._CLOSED.is_set() and display_projections._CONTEXT is None
            second = context()
            second.budget_projection_dirty = True
            server_maintenance.dispatch_display_projections(second)  # a tick's look starts a pass again
            _pass_ended()
            assert writes == [last, duty] and second.budget_projection_dirty is False
            second.budget_projection_dirty = True  # its final turn
        assert writes == [last, duty, last] and second.budget_projection_dirty is False
    finally:
        sink.gate.set()
        logger.removeHandler(sink)


def test_the_last_pass_and_the_socket_lines_share_half_of_the_grace_left_and_the_census_takes_its_own(monkeypatch):
    """Each door that ends this image's serving (the lifespan teardown, ahead of the owned-work stop; the exit and the
    direct re-exec, ahead of the output relay's handover) gives its display projections' last pass and its socket lines
    ONE share of the stop grace, half of what is left, taken once: each waits until that share's end at most, so the two
    together end by it and what follows the door keeps the larger part. (Each taking half of what it found left, they
    took three quarters on a stalled disk: of 4.00 s, 0.99 s stayed for the owned stop.) The restart census, with no
    socket lines behind it, takes half of what is left at its own call."""
    from ouroboros import owned_shutdown, process_logging, server_control

    grace_end = time.monotonic() + 8.0
    monkeypatch.setattr(owned_shutdown, "stop_deadline", lambda: grace_end)
    received = []

    def half(at):  # the end of half of the grace left at ``at``, as the door computes it
        return at + (grace_end - at) / 2

    def drain(deadline):  # the last pass stands a while, as on a stalled disk
        received.append(("display projections", deadline))
        time.sleep(0.05)
        return True

    monkeypatch.setattr(display_projections, "drain", drain)
    monkeypatch.setattr(process_logging, "finish_record_handoffs",
                        lambda deadline: received.append(("socket lines", deadline)))
    started = time.monotonic()
    server_control.finish_display_and_socket_lines()
    ended = time.monotonic()
    (first, share), (then, lines) = received
    assert (first, then) == ("display projections", "socket lines") and lines == share, "one share, taken once"
    assert half(started) <= share <= half(ended), "half of the grace left, not the whole of it"
    received.clear()
    started = time.monotonic()
    assert server_control.drain_display_projections() is True  # the restart census before the store switch
    ended = time.monotonic()
    ((_name, census),) = received
    assert half(started) <= census <= half(ended)


class _HeldLogSink(logging.Handler):
    """A handler of the server log whose lock another thread holds (``hold``), as a write stalled on a full disk
    holds it: a record handled on any other thread waits until ``free``."""

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.lines, self.holding, self.free = [], threading.Event(), threading.Event()

    def emit(self, record) -> None:
        self.lines.append(record.getMessage())

    def hold(self) -> None:
        with self.lock:
            self.holding.set()
            assert self.free.wait(10), "the test must free the held log sink"


_DRAIN_WARNINGS = {
    "pass-in-flight": "The display projections' pass outlived its share of the stop grace; it stays unjoined",
    "last-pass": "The display projections' last pass outlived its share of the stop grace; its rest stays unwritten",
    "no-time": ("No time is left in the stop grace to wait for the display projections' last pass; "
                "its work stays unwritten"),
    "no-thread": "The display projections' last pass could not start; its work stays unwritten",
}


@pytest.mark.parametrize("ending", list(_DRAIN_WARNINGS))
def test_the_door_returns_within_its_share_with_the_server_log_held_and_the_drain_warning_follows(monkeypatch, ending):
    """The drain warns on the door's thread when the pass in flight or its last pass outlives the share, when no time
    is left to wait for a last pass, or when none can start. With the server log's handler held by another thread (a
    write stalled on a full disk), the door (``finish_display_and_socket_lines``) still returns within its share,
    ahead of the owned stop: the warning waits in the drain's record handoff, which the door's wait for the handoffs
    leaves at the share's end, and reaches the handler once it is free, once. (Logged on the door's thread, it waited
    for that handler past the whole grace: 1.40 s against a share of 0.6 s.)"""
    from ouroboros import owned_shutdown, server_control

    held, release, hold, _holding = _held_seam()
    ctx = SimpleNamespace(budget_projection_dirty=True, update_budget_from_usage=lambda usage: hold() or True)
    display_projections.bind(ctx)
    in_flight, jump, returned = [], [0.0], []
    if ending in ("pass-in-flight", "no-time"):  # a pass in flight stands in its budget write
        work = display_projections.take(ctx)
        assert display_projections.LATCH.acquire(blocking=False)
        in_flight.append(threading.Thread(target=display_projections.project, args=(work, display_projections.LATCH)))
        in_flight[0].start()
        assert held.wait(10)
        ctx.budget_projection_dirty = True  # the turn's newer mark, for a last pass
    if ending == "no-time":  # the drain's own clock passes its cutoff as the pass in flight ends
        monkeypatch.setattr(display_projections, "time",
                            SimpleNamespace(monotonic=lambda: time.monotonic() + jump[0], time=time.time))
    if ending == "no-thread":
        def no_thread(*_args, **_kwargs):
            raise RuntimeError("can't start new thread")

        monkeypatch.setattr(display_projections, "threading", SimpleNamespace(Thread=no_thread))

    def end_at_the_cutoff():
        jump[0] = 3600.0
        release.set()

    def teardown():
        server_control.finish_display_and_socket_lines()
        returned.append(time.monotonic())

    sink = _HeldLogSink()
    monkeypatch.setattr(display_projections.log, "disabled", False)
    display_projections.log.addHandler(sink)
    holder = threading.Thread(target=sink.hold, name="stalled-log-write")
    door, ender = threading.Thread(target=teardown, name="lifespan-teardown"), threading.Timer(0.2, end_at_the_cutoff)
    grace_end = time.monotonic() + 2.0
    monkeypatch.setattr(owned_shutdown, "stop_deadline", lambda: grace_end)
    try:
        holder.start()
        assert sink.holding.wait(10)
        share_end = time.monotonic() + (grace_end - time.monotonic()) / 2
        door.start()
        if ending == "no-time":
            ender.start()
        door.join(grace_end + 1.0 - time.monotonic())
        assert returned and returned[0] < share_end + 0.5, "the door returned within its share, the server log held"
        assert _DRAIN_WARNINGS[ending] not in sink.lines, "the warning waits for the held sink"
    finally:
        sink.free.set()
        release.set()
        ender.cancel()
        for thread in (holder, door, ender, *in_flight):
            if thread.ident is not None:
                thread.join(10)
        _pass_ended()
        _wait_until(lambda: display_projections._DRAIN_LOG.thread is None)
        display_projections.log.removeHandler(sink)
    assert sink.lines.count(_DRAIN_WARNINGS[ending]) == 1, "the warning reached the freed sink, once"


def test_a_look_that_raises_frees_the_latch_for_the_next_tick(monkeypatch, caplog):
    """``take`` reads memory only, yet a look that raised once kept the latch for the generation: no pass, no
    stall row (the watchdog had no entry), every later look skipped. The latch is free again and the next look
    starts its pass."""
    ctx = SimpleNamespace(budget_projection_dirty=True, update_budget_from_usage=lambda usage: True)
    real_take = display_projections.take

    def broken(_ctx):
        raise RuntimeError("dictionary changed size during iteration")

    monkeypatch.setattr(display_projections, "take", broken)
    busy = lambda *_a: pytest.fail("the latch was free")  # noqa: E731
    with caplog.at_level(logging.WARNING, logger=display_projections.log.name):
        display_projections.dispatch(ctx, lambda *_a: pytest.fail("a failed look started a pass"), busy)
    assert not display_projections.LATCH.locked() and "look failed" in caplog.text
    monkeypatch.setattr(display_projections, "take", real_take)
    starts = []

    def started(latch, name, target, args):
        starts.append(name)
        latch.release()
        return True

    display_projections.dispatch(ctx, started, busy)
    assert starts == [display_projections.DUTY_NAME]


def test_a_drain_of_many_usage_events_costs_one_write_and_the_mark_stays_owed_until_one_lands(monkeypatch, tmp_path):
    """N ``llm_usage`` events of one drain only mark (their rows say ``deferred``); the tick's look starts
    ONE pass, which writes once. A refused or raising write keeps the mark owed and is retried no sooner
    than ``BUDGET_PROJECTION_RETRY_SEC``; the first write that lands clears it; a clean context has nothing due."""
    from ouroboros import server_liveness
    from supervisor import events as events_mod

    (tmp_path / "logs").mkdir()
    writes = []
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={}, bridge=SimpleNamespace(push_log=lambda payload: None),
                          update_budget_from_usage=lambda usage: writes.append(threading.current_thread().name) or True)
    monkeypatch.setattr(events_mod, "dispatch_event", lambda evt, c: events_mod._handle_llm_usage(evt, c))
    bus: "queue.Queue" = queue.Queue()
    for n in range(8):
        bus.put({"type": "llm_usage", "task_id": f"t{n}", "usage": {"prompt_tokens": 1, "cost": 0.01}})
    server_liveness.drain_worker_events(bus, ctx, [time.monotonic(), {}, time.thread_time(), None],
                                        on_restart=lambda evt, c: None)
    assert writes == [] and ctx.budget_projection_dirty is True, "the drain only marks"
    server_maintenance.dispatch_display_projections(ctx)
    _pass_ended()
    assert writes == [display_projections.DUTY_NAME] and ctx.budget_projection_dirty is False
    rows = [json.loads(line) for line in (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 8 and {row["projection_update_status"] for row in rows} == {"deferred"}
    assert display_projections.take(ctx) is None, "a clean context has nothing due"

    monkeypatch.setattr(display_projections, "BUDGET_PROJECTION_RETRY_SEC", 30.0)
    clock = [1000.0]
    monkeypatch.setattr(display_projections.time, "monotonic", lambda: clock[0])
    outcomes = [False, RuntimeError("ledger unreadable"), True]

    def writer(_usage):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    ctx = SimpleNamespace(budget_projection_dirty=True, update_budget_from_usage=writer)
    display_projections.project(display_projections.take(ctx))
    assert len(outcomes) == 2 and ctx.budget_projection_dirty is True, "a refused write keeps the mark"
    assert display_projections.take(ctx) is None, "no retry inside the bounded interval"
    clock[0] += 30.0
    display_projections.project(display_projections.take(ctx))
    assert len(outcomes) == 1 and ctx.budget_projection_dirty is True, "an exception keeps the mark too"
    clock[0] += 30.0
    display_projections.project(display_projections.take(ctx))
    assert not outcomes and ctx.budget_projection_dirty is False and ctx.budget_projection_retry_at == 0.0


def test_a_pass_that_cannot_start_leaves_its_work_due_and_the_snapshot_shares_nothing_with_running(tmp_path):
    """The tick takes the due roots and the budget mark only to hand them to a pass: a start that fails puts
    both back, the next start carries them, and a look with nothing due starts nothing. What the pass gets
    is a copy of the lineage the projection reads, never the live RUNNING entry, its text or its metadata."""
    task = {"id": "root", "type": "task", "root_task_id": "root", "text": "the owner's long brief",
            "metadata": {"origin_message_text": "the owner's long brief", "root_task_id": "root"}}
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={"root": {"task": task}}, budget_projection_dirty=True,
                          update_budget_from_usage=lambda usage: True)
    LIVE_ROOT_COSTS.fields(tmp_path, "root", task)  # first sight: the root's figure is due
    starts = []

    def refused(latch, name, target, args):
        latch.release()  # the production starter releases the latch of a start that failed
        starts.append(None)
        return False

    def started(latch, name, target, args):
        starts.append((name, target, args[0]))
        latch.release()
        return True

    busy = lambda *_a: pytest.fail("the latch was free")  # noqa: E731
    display_projections.dispatch(ctx, refused, busy)
    assert starts == [None] and ctx.budget_projection_dirty is True and not display_projections.LATCH.locked()
    display_projections.dispatch(ctx, started, busy)
    (name, target, work), = starts[1:]
    assert (name, target) == (display_projections.DUTY_NAME, display_projections.project)
    assert work.budget is not None and ctx.budget_projection_dirty is False
    (root,) = work.roots
    assert (root.carrier, root.task) == ("root", {"root_task_id": "root", "metadata": {"root_task_id": "root"}})
    task["root_task_id"] = "moved"
    assert root.task["root_task_id"] == "root", "the snapshot is a copy"
    display_projections.dispatch(ctx, started, busy)
    assert len(starts) == 2, "nothing due: the look starts nothing"


def test_the_exit_takes_one_last_pass_with_the_queued_rows_and_the_bound_mark_and_nothing_after_it(monkeypatch):
    """The lifespan's end (``drain``), once: the queued rows, in order, and the bound loop context's due budget
    mark go in one last pass, off the caller's thread, inside half the stop grace left, and again for a row queued
    meanwhile (an overrun a supervisor loop that outlived its join still found); with nothing outstanding nothing
    starts. After it no look and no later drain starts a pass: a mark or an onset made then stays (a duty that ends
    then writes its own closing row)."""
    rows, writes = [], []
    monkeypatch.setattr("supervisor.state.append_jsonl", lambda _path, row: rows.append(row) or True)
    monkeypatch.setattr(update_generation_facts, "_duties", {})
    monkeypatch.setattr(update_generation_facts, "_publish_after", lambda: None)
    running = {"name": "cold-custody", "ident": None, "since": time.time() - 700, "alerted": False}

    def write(_usage):  # the last pass's budget write; meanwhile the watchdog alerts the overrunning cold-custody pass
        writes.append(threading.current_thread().name)
        display_projections.journal_duty_stall(running, 600.0, [], {})
        return True

    ctx = SimpleNamespace(budget_projection_dirty=True, update_budget_from_usage=write)
    display_projections.bind(ctx)
    duty = {"name": "custody-maintenance", "ident": None, "since": time.time() - 700, "alerted": False}
    display_projections.journal_duty_stall(duty, 600.0, [], {})
    display_projections.journal_duty_stall_end(duty)
    assert display_projections.drain(time.monotonic() + 10) is True
    assert [(row["type"], row["duty"]) for row in rows] == [
        ("host_duty_stall", "custody-maintenance"), ("host_duty_stall_end", "custody-maintenance"),
        ("host_duty_stall", "cold-custody"),
    ], "the onset queued during the last pass is written before the drain returns"
    assert writes == [f"{display_projections.DUTY_NAME}-drain"] and ctx.budget_projection_dirty is False
    later = {"name": "reconcile-maintenance", "ident": None, "since": time.time() - 400, "alerted": False}
    display_projections.journal_duty_stall(later, 300.0, [], {})
    ctx.budget_projection_dirty = True
    display_projections.dispatch(ctx, lambda *_a: pytest.fail("a look started a pass after the last one"),
                                 lambda *_a: pytest.fail("the latch was free"))
    assert display_projections.drain(time.monotonic() + 10) is True
    assert len(rows) == 3 and len(writes) == 1 and ctx.budget_projection_dirty is True

    monkeypatch.setattr(display_projections, "_CLOSED", threading.Event())  # another process's end
    monkeypatch.setattr(display_projections, "_ROWS", collections.deque())
    monkeypatch.setattr(update_generation_facts, "_pending", False)
    ctx.budget_projection_dirty = False
    started = time.monotonic()
    assert display_projections.drain(started + 10) is True and time.monotonic() - started < 1.0
    assert len(writes) == 1, "nothing outstanding: no last pass"


def test_the_exit_leaves_a_pass_that_stands_past_half_the_grace_and_waits_for_nothing_without_one(monkeypatch):
    """A pass in flight at the process's end is waited for until the drain's deadline at most, its door's share of the
    stop grace (half of what is left: here 1 s of 2); one that stands is left unjoined, its write never recorded as
    done, and the drain says so (the restart census reports it). Without a started grace, or once Panic owns the exit,
    nothing waits and no last pass runs."""
    from ouroboros import server_control

    held, release, hold, _holding = _held_seam()
    ctx = SimpleNamespace(budget_projection_dirty=True, update_budget_from_usage=lambda usage: hold() or True)
    work = display_projections.take(ctx)
    assert display_projections.LATCH.acquire(blocking=False)
    thread = threading.Thread(target=display_projections.project, args=(work, display_projections.LATCH))
    thread.start()
    try:
        assert held.wait(10)
        started = time.monotonic()
        assert display_projections.drain(started + 1.0) is False
        assert 0.9 <= time.monotonic() - started < 2.0, "until its deadline, no longer"
        assert display_projections.drain(started + 10) is False, "once: a later drain only reads, the first ended"
        assert time.monotonic() - started < 2.0
        assert not hasattr(ctx, "budget_projection_retry_at"), "the stood write's outcome is not recorded"
    finally:
        release.set()
        thread.join(10)
    assert ctx.budget_projection_retry_at == 0.0 and display_projections.drain(None) is True

    rows = []
    monkeypatch.setattr("supervisor.state.append_jsonl", lambda _path, row: rows.append(row) or True)
    monkeypatch.setattr(update_generation_facts, "_duties", {})
    duty = {"name": "custody-maintenance", "ident": None, "since": time.time() - 700, "alerted": False}
    display_projections.journal_duty_stall(duty, 600.0, [], {})
    for closing in ("no grace", "panic"):
        monkeypatch.setattr(display_projections, "_CLOSED", threading.Event())
        monkeypatch.setattr(server_control, "_restart_stop_requested", closing == "panic")
        started = time.monotonic()
        assert display_projections.drain(None if closing == "no grace" else started + 10) is True
        assert time.monotonic() - started < 1.0 and rows == [], closing


def test_no_last_pass_starts_that_the_drain_has_no_time_left_to_wait_for(monkeypatch):
    """The pass in flight ends exactly at the drain's share of the stop grace while the turn's newer mark waits: the
    drain starts no last pass it could not wait for (one would read as a pass still running, and a restart's store
    switch would refuse), leaves the mark owed and unwritten, and says no pass runs. With time left the same mark goes
    in a last pass."""
    clock = [1000.0]
    monkeypatch.setattr(display_projections.time, "monotonic", lambda: clock[0])
    held, release, hold, _holding = _held_seam()
    writes = []

    def write(_usage):
        hold()
        writes.append(threading.current_thread().name)
        return True

    def end_at_the_cutoff():  # the stood write returns as the drain's share of the grace runs out
        clock[0] = 1005.0
        release.set()

    ctx = SimpleNamespace(budget_projection_dirty=True, update_budget_from_usage=write)
    display_projections.bind(ctx)
    work = display_projections.take(ctx)
    assert display_projections.LATCH.acquire(blocking=False)
    thread = threading.Thread(target=display_projections.project, args=(work, display_projections.LATCH),
                              name="pass-in-flight")
    thread.start()
    ender = threading.Timer(0.2, end_at_the_cutoff)
    try:
        assert held.wait(10)
        ctx.budget_projection_dirty = True  # the turn's newer mark
        ender.start()
        assert display_projections.drain(1005.0) is True  # its share of the grace: to 1005
        assert writes == ["pass-in-flight"] and ctx.budget_projection_dirty is True, "no last pass started"
        assert not display_projections.LATCH.locked()
    finally:
        release.set()
        ender.cancel()
        thread.join(10)
    display_projections.reopen()  # another lifespan's end, with time left
    display_projections.bind(ctx)
    assert display_projections.drain(clock[0] + 10) is True
    assert writes == ["pass-in-flight", f"{display_projections.DUTY_NAME}-drain"]
    assert ctx.budget_projection_dirty is False


def test_a_backup_write_failure_is_typed_and_the_pair_converges_after_another_writer(tmp_path, monkeypatch, caplog):
    """The duty keeps the state pair's one synchronous writer: when the last-good copy fails after the primary
    landed, the typed ``backup_write_failed`` reaches the log and the mark stays owed; another writer of the
    pair (an owner control) writes both files meanwhile; the retry then writes the newer money into both."""
    import ouroboros.usage_accounting as accounting
    from supervisor import state

    state.init(tmp_path, total_budget_limit=0.0)
    state.save_state({})  # an initialized install: only explicit init creates state (#1307)

    def settle(task: str) -> None:
        hold = accounting.reserve_attempt(accounting.AttemptRequest(
            model="test/model", provider="test", drive_root=tmp_path, task_id=task, root_task_id="root",
            reservation_usd=1.0))
        accounting.mark_dispatched(hold)
        accounting.settle_attempt(hold, {"prompt_tokens": 1}, cost_usd=1.0, cost_final=True)

    real, failed = state.atomic_write_text, []

    def write(path, body):
        if path == state.STATE_LAST_GOOD_PATH and not failed:
            failed.append(path)
            raise OSError(errno.EIO, "backup unavailable")
        return real(path, body)

    monkeypatch.setattr(state, "atomic_write_text", write)
    settle("first")
    ctx = SimpleNamespace(budget_projection_dirty=True, update_budget_from_usage=state.update_budget_from_usage)
    with caplog.at_level(logging.ERROR, logger=display_projections.log.name):
        display_projections.project(display_projections.take(ctx))
    (record,) = [r for r in caplog.records if "Compatibility budget projection update failed" in r.getMessage()]
    error = record.exc_info[1]
    assert isinstance(error, state.StateUnavailable) and error.reason == "backup_write_failed" and error.primary_written
    assert ctx.budget_projection_dirty is True and ctx.budget_projection_retry_at > 0
    primary, backup = (json.loads(path.read_text(encoding="utf-8")) for path in (state.STATE_PATH, state.STATE_LAST_GOOD_PATH))
    assert primary["spent_usd"] == 1.0 and backup.get("spent_usd") != 1.0

    state.update_state(lambda st: st.update(owner_note="kept"))  # another writer of the pair, both files
    settle("second")
    ctx.budget_projection_retry_at = 0.0  # the retry bound has passed
    display_projections.project(display_projections.take(ctx))
    primary, backup = (json.loads(path.read_text(encoding="utf-8")) for path in (state.STATE_PATH, state.STATE_LAST_GOOD_PATH))
    assert primary == backup and primary["spent_usd"] == 2.0 and primary["owner_note"] == "kept"
    assert ctx.budget_projection_dirty is False


def test_a_pass_that_ended_while_its_stall_was_read_leaves_no_onset_without_an_end(monkeypatch):
    """The watchdog reads a duty's stack before it queues the onset; a pass that ended meanwhile has had its end
    noted, so no onset is queued and nothing is counted, while one still running gets its onset, then its end."""
    monkeypatch.setattr(update_generation_facts, "_duties", {})
    ended = {"name": "custody-maintenance", "ident": None, "since": time.time() - 700, "alerted": False}
    display_projections.journal_duty_stall_end(ended)  # the pass ended first
    display_projections.journal_duty_stall(ended, 600.0, [], {})
    assert not display_projections._ROWS and update_generation_facts._duties == {}
    running = {"name": "custody-maintenance", "ident": None, "since": time.time() - 700, "alerted": False}
    display_projections.journal_duty_stall(running, 600.0, [], {})
    display_projections.journal_duty_stall_end(running)
    assert [row["type"] for _line, row in display_projections._ROWS] == ["host_duty_stall", "host_duty_stall_end"]
    assert update_generation_facts._duties == {"custody-maintenance": 1}


def test_the_last_pass_writes_the_closing_row_an_overrunning_display_pass_queued_before_letting_go(monkeypatch):
    """A display pass the watchdog alerted as overrunning ends just before the lifespan's end: it queues its own
    ``host_duty_stall_end`` while it still holds its latch, so the last pass, which takes that latch, writes the end
    beside the onset even while the duty wrapper's close has not run yet; that later close queues no second end.
    (Queued after the latch was released, the end found the duty closed and stayed unwritten.)"""
    rows = []
    monkeypatch.setattr("supervisor.state.append_jsonl", lambda _path, row: rows.append(row) or True)
    monkeypatch.setattr(update_generation_facts, "_duties", {})
    monkeypatch.setattr(update_generation_facts, "_publish_after", lambda: None)
    held, release, hold, _holding = _held_seam()
    ctx = SimpleNamespace(DRIVE_ROOT=None, RUNNING={}, budget_projection_dirty=True,
                          update_budget_from_usage=lambda usage: hold() or True)
    closing, real_close = threading.Event(), server_maintenance._duty_ended

    def late_close(latch, duty):  # the wrapper's close, scheduled only after the pass let go of its latch
        assert closing.wait(10)
        real_close(latch, duty)

    monkeypatch.setattr(server_maintenance, "_duty_ended", late_close)
    server_maintenance.dispatch_display_projections(ctx)
    try:
        assert held.wait(10)
        server_maintenance._DUTIES[id(display_projections.LATCH)]["since"] -= 3600  # well past its cadence
        server_maintenance._duty_busy(display_projections.LATCH, 60.0)  # the watchdog alerts the overrun
        release.set()
        _wait_until(lambda: not display_projections.LATCH.locked())
        assert display_projections.drain(time.monotonic() + 10) is True
        assert [(row["type"], row["duty"]) for row in rows] == [
            ("host_duty_stall", display_projections.DUTY_NAME), ("host_duty_stall_end", display_projections.DUTY_NAME)]
    finally:
        release.set()
        closing.set()
    _pass_ended()
    assert len(rows) == 2 and not display_projections._ROWS, "the wrapper's close queued no second end"


def test_a_duty_that_ends_after_the_last_pass_writes_its_own_closing_row_and_leaves_no_stall_open(
        tmp_path, monkeypatch):
    """An alerted host duty (an overrun the watchdog journaled while the supervisor ran) whose pass ends during the
    teardown AFTER its last display pass: no pass writes again in this image, so the duty's own thread writes its
    ``host_duty_stall_end`` behind the onset, never holding the latch the restart census reads (the census reads no
    pass running while that row is written); the journal keeps onset and end, and the mind's vitals list no stall that
    ended. (Queued for a pass that never came, the end stayed in memory and the onset read as a stall with no end.)"""
    from ouroboros import context_runtime_facts, owned_shutdown, server_control
    from supervisor import state as state_mod

    journal = tmp_path / "logs" / "supervisor.jsonl"
    journal.parent.mkdir()
    monkeypatch.setattr(display_projections, "DATA_DIR", tmp_path)
    monkeypatch.setattr(update_generation_facts, "_duties", {})
    monkeypatch.setattr(update_generation_facts, "_publish_after", lambda: None)
    grace_end = time.monotonic() + 10.0
    monkeypatch.setattr(owned_shutdown, "stop_deadline", lambda: grace_end)
    held, release, hold, _holding = _held_seam()
    entered, gate, ends, threads = threading.Event(), threading.Event(), [], []
    real_append = state_mod.append_jsonl

    def append(path, row):  # the closing row's append stands a while, as on a slow disk
        ends.append((row["type"], threading.current_thread().name))
        entered.set()
        assert gate.wait(10)
        return real_append(path, row)

    def custody_pass(latch):  # production's duties release their latch last; the wrapper then closes the duty
        threads.append(threading.current_thread())
        try:
            hold()
        finally:
            latch.release()

    latch = threading.Lock()
    assert latch.acquire(blocking=False)
    assert server_maintenance._start_maintenance_thread(latch, "custody-maintenance", custody_pass, (latch,))
    try:
        assert held.wait(10)
        server_maintenance._DUTIES[id(latch)]["since"] -= 3600  # well past its cadence
        server_maintenance._duty_busy(latch, 600.0)  # the watchdog alerts the overrun while the supervisor ran
        server_control.finish_display_and_socket_lines()  # the teardown's last pass writes the onset
        monkeypatch.setattr(state_mod, "append_jsonl", append)
        release.set()  # the duty's pass ends later in the teardown
        assert entered.wait(10), "the ending duty writes its closing row itself"
        assert server_control.drain_display_projections() is True, "the census reads no pass while that row is written"
    finally:
        release.set()
        gate.set()
        for thread in threads:
            thread.join(10)
    rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    assert [(row["type"], row["duty"]) for row in rows] == [
        ("host_duty_stall", "custody-maintenance"), ("host_duty_stall_end", "custody-maintenance")]
    assert ends == [("host_duty_stall_end", "custody-maintenance")] and not display_projections._ROWS
    assert context_runtime_facts._supervisor_tail(tmp_path)["open"] == {}


def test_a_duty_that_ends_during_the_last_pass_waits_for_the_row_that_pass_is_writing_and_lands_behind_it(monkeypatch):
    """One writer of the queue at a time, from a row's pop to its append: while the last pass stands inside its append
    of an overrun's onset (a slow disk), the overrunning duty ends on its own thread and, the duty being closed, writes
    the queue too; it waits for that append, so the journal reads onset, then end. (Each writer popping and appending
    on its own, the end overtook the onset it closes.)"""
    rows, entered, gate = [], threading.Event(), threading.Event()

    def append(_path, row):  # the last pass stands inside the onset's append
        if row["type"] == "host_duty_stall":
            entered.set()
            assert gate.wait(10)
        rows.append((row["type"], threading.current_thread().name))
        return True

    monkeypatch.setattr("supervisor.state.append_jsonl", append)
    monkeypatch.setattr(update_generation_facts, "_duties", {})
    monkeypatch.setattr(update_generation_facts, "_publish_after", lambda: None)
    duty = {"name": "cold-custody", "ident": None, "since": time.time() - 700, "alerted": False}
    display_projections.journal_duty_stall(duty, 600.0, [], {})
    drained = []
    teardown = threading.Thread(target=lambda: drained.append(display_projections.drain(time.monotonic() + 10)),
                                name="lifespan-teardown")
    ending = threading.Thread(target=display_projections.journal_duty_stall_end, args=(duty,), name="cold-custody")
    teardown.start()
    try:
        assert entered.wait(10)  # the last pass popped the onset and stands inside its append
        ending.start()  # the duty's pass ends now, after the close
        _wait_until(lambda: display_projections._rows_queued() or rows)
        ending.join(0.2)
        assert ending.is_alive() and rows == [], "the ending thread waits for the append that stands"
    finally:
        gate.set()
        teardown.join(10)
        if ending.ident is not None:
            ending.join(10)
    assert [kind for kind, _thread in rows] == ["host_duty_stall", "host_duty_stall_end"]
    assert rows[0][1] == f"{display_projections.DUTY_NAME}-drain" and drained == [True]
    assert not display_projections._ROWS


def test_an_alerted_display_pass_ending_after_the_close_writes_its_rows_with_its_latch_let_go(monkeypatch):
    """The display duty's own pass, alerted as overrunning, outlives the teardown's share while its store work (here
    the budget write) stands: a census read then finds a pass running, and a store switch refuses. It ends after the
    close: it queues its closing row under its latch, as before, lets go of the latch, and only then writes the queue
    on its own thread, so a census read while that append stands finds no pass and a ready switch is not refused for
    a diagnostic; the rows land onset, then end. (Written under the latch, the append read as a pass running.)"""
    from ouroboros import owned_shutdown, server_control

    rows, entered, gate = [], threading.Event(), threading.Event()

    def append(_path, row):  # the closing row's append stands, as on a slow disk
        if row["type"] == "host_duty_stall_end":
            entered.set()
            assert gate.wait(10)
        rows.append((row["type"], row["duty"], threading.current_thread().name))
        return True

    monkeypatch.setattr("supervisor.state.append_jsonl", append)
    monkeypatch.setattr(update_generation_facts, "_duties", {})
    monkeypatch.setattr(update_generation_facts, "_publish_after", lambda: None)
    grace_end = time.monotonic() + 0.6
    monkeypatch.setattr(owned_shutdown, "stop_deadline", lambda: grace_end)
    held, release, hold, _holding = _held_seam()
    ctx = SimpleNamespace(DRIVE_ROOT=None, RUNNING={}, budget_projection_dirty=True,
                          update_budget_from_usage=lambda usage: hold() or True)
    server_maintenance.dispatch_display_projections(ctx)
    try:
        assert held.wait(10)  # the pass stands in its store work
        server_maintenance._DUTIES[id(display_projections.LATCH)]["since"] -= 3600  # well past its cadence
        server_maintenance._duty_busy(display_projections.LATCH, 60.0)  # the watchdog alerts the overrun
        server_control.finish_display_and_socket_lines()  # the teardown: the pass outlives its share
        assert display_projections._CLOSED.is_set()
        assert server_control.drain_display_projections() is False, "a pass doing store work reads as running"
        release.set()  # its store work ends after the close
        assert entered.wait(10), "the pass writes its rows itself after the close"
        assert server_control.drain_display_projections() is True, "no pass reads as running while that row is written"
    finally:
        release.set()
        gate.set()
    _pass_ended()
    duty = display_projections.DUTY_NAME
    assert rows == [("host_duty_stall", duty, duty), ("host_duty_stall_end", duty, duty)]
    assert not display_projections._ROWS
