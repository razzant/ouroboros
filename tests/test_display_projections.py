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
    carries is the latest read, for the carrier that still runs it."""
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
    assert LIVE_ROOT_COSTS.fields(ctx.DRIVE_ROOT, "root", task)["accounted_upper_bound_usd_with_children"] == 1.0
    server_maintenance.dispatch_display_projections(ctx)
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


@pytest.mark.parametrize("stands", [True, False], ids=["pass-stands", "pass-ends"])
def test_the_restart_census_takes_the_last_pass_before_the_store_switch(tmp_path, monkeypatch, stands):
    """The restart's census (``server._emergency_process_cleanup``) waits for the display pass in flight, then takes
    the last pass with what is outstanding (here the turn's newer budget mark), inside half the stop grace left and
    before ``store_switch.switch_at_restart``. A pass that still stands at that bound is in the census, so the switch
    refuses rather than hand off stores that pass reads, or let it write ``state.json`` after the switch; a pass
    that ended lets the switch run. No pass starts after the census either way."""
    import server
    from ouroboros import store_switch

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
    if not stands:
        releaser.start()
    try:
        started = time.monotonic()
        grace[0] = started + 3.0
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
        _pass_ended()
    duty, last = display_projections.DUTY_NAME, f"{display_projections.DUTY_NAME}-drain"
    assert writes == ([duty] if stands else [duty, last])
    assert ctx.budget_projection_dirty is stands, "a pass that stood left the newer mark to the exit's loss"
    ctx.budget_projection_dirty = True
    server_maintenance.dispatch_display_projections(ctx)
    assert not display_projections.LATCH.locked() and len(writes) == (1 if stands else 2), "nothing after the census"


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
    """The process's end (``drain``), once: the queued rows, in order, and the bound loop context's due budget
    mark go in one last pass, off the caller's thread, inside half the stop grace left; with nothing outstanding
    nothing starts. After it no look and no later drain starts a pass: what is marked or queued then stays."""
    rows, writes = [], []
    monkeypatch.setattr("supervisor.state.append_jsonl", lambda _path, row: rows.append(row) or True)
    monkeypatch.setattr(update_generation_facts, "_duties", {})
    monkeypatch.setattr(update_generation_facts, "_publish_after", lambda: None)
    ctx = SimpleNamespace(budget_projection_dirty=True,
                          update_budget_from_usage=lambda usage: writes.append(threading.current_thread().name) or True)
    display_projections.bind(ctx)
    duty = {"name": "custody-maintenance", "ident": None, "since": time.time() - 700, "alerted": False}
    display_projections.journal_duty_stall(duty, 600.0, [], {})
    display_projections.journal_duty_stall_end(duty)
    assert display_projections.drain(time.monotonic() + 10) is True
    assert [row["type"] for row in rows] == ["host_duty_stall", "host_duty_stall_end"]
    assert writes == [f"{display_projections.DUTY_NAME}-drain"] and ctx.budget_projection_dirty is False
    later = {"name": "reconcile-maintenance", "ident": None, "since": time.time() - 400, "alerted": False}
    display_projections.journal_duty_stall(later, 300.0, [], {})
    ctx.budget_projection_dirty = True
    display_projections.dispatch(ctx, lambda *_a: pytest.fail("a look started a pass after the last one"),
                                 lambda *_a: pytest.fail("the latch was free"))
    assert display_projections.drain(time.monotonic() + 10) is True
    assert len(rows) == 2 and len(writes) == 1 and ctx.budget_projection_dirty is True

    monkeypatch.setattr(display_projections, "_CLOSED", threading.Event())  # another process's end
    monkeypatch.setattr(display_projections, "_ROWS", collections.deque())
    monkeypatch.setattr(update_generation_facts, "_pending", False)
    ctx.budget_projection_dirty = False
    started = time.monotonic()
    assert display_projections.drain(started + 10) is True and time.monotonic() - started < 1.0
    assert len(writes) == 1, "nothing outstanding: no last pass"


def test_the_exit_leaves_a_pass_that_stands_past_half_the_grace_and_waits_for_nothing_without_one(monkeypatch):
    """A pass in flight at the process's end is waited for at most half of the stop grace left; one that stands
    is left unjoined, its write never recorded as done, and the drain says so (the restart census reports it).
    Without a started grace, or once Panic owns the exit, nothing waits and no last pass runs."""
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
        assert display_projections.drain(started + 2.0) is False
        assert 0.9 <= time.monotonic() - started < 2.0, "at most half of the grace left"
        assert display_projections.drain(started + 10) is False, "once: a later drain waits for nothing"
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
