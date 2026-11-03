"""Reconciliation of delegated runs by the supervisor generation's sweeps.

Started as the D11 slice of the reference theme split of
``tests/test_delegated_subagent_transport.py`` (v7 WIP 9f691656,
``tests/test_delegated_reconciliation.py``): the two tests here bind the
``ouroboros.server_maintenance`` owner the server composition split created, and
re-homing them is the byte-debt pressure valve — the giant shrinks, the pin
gains its family. The F2.1 delegation split then brought the rest of the
reference theme here (orphan-sweep predicate, absent-run closure, release
points) — this file now owns the full reconciliation theme.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from ouroboros.config import CLAUDEXOR_DELEGATED_MARKER_MIN_VERSION
from ouroboros.gateways import claudexor as cx

from tests._cancel_intents_shared import qenv as _qenv
from tests._store_shared import unreadable_result
from tests._delegated_transport_shared import (  # noqa: F401  (autouse fixture applies on import)
    _LiveRunStub,
    _event_types,
    _health_invariants,
    _owned_gateway_uses_each_test_transport,
    _waiting,
    _write_attempt,
)

qenv = _qenv


@pytest.fixture
def startup_owners(tmp_path, monkeypatch):
    from ouroboros import post_task_checkpoint, server_maintenance
    from supervisor import active_activity, queue, workers

    monkeypatch.setattr(server_maintenance, "DATA_DIR", tmp_path)
    # Not this fixture's duty: the real child's own thread would outlive the test (it has its own tests).
    monkeypatch.setattr(server_maintenance, "_tick_maintenance_child", lambda: None)
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(queue, "PENDING", [])
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(post_task_checkpoint, "POST_TASK_SYNTHESIS_INFLIGHT", {})
    registry = active_activity.DirectActivityRegistry()
    monkeypatch.setattr(active_activity, "_DIRECT_ACTIVITY_REGISTRY", registry)
    return queue, workers, post_task_checkpoint, registry


def test_the_startup_sweep_reconciles_delegated_runs_too(monkeypatch, startup_owners):
    """A fresh generation has no surviving queue, direct or post-task owners.

    Startup must reconcile with that actual empty set, without inheriting another
    test's queue. Liveness now arrives as the sweep's one CALLABLE source (INV-B:
    candidates first, liveness second), so the assertion reads what it produces.
    An in-process revival with live owners is covered separately.
    """
    import ouroboros.server_maintenance as sm
    import ouroboros.delegate_custody as dc
    import ouroboros.process_custody as pc

    seen = {}
    monkeypatch.setattr(pc, "reap_orphaned_processes", lambda root, **kw: [])
    monkeypatch.setattr(dc, "reconcile_orphaned_runs",
                        lambda root, **kw: seen.setdefault("live", kw.get("running_task_ids")) or [])
    monkeypatch.setattr(sm, "_installed_skill_names", lambda: None)
    sm._startup_custody_sweep()
    assert seen["live"] is sm._live_task_ids
    assert seen["live"]() == set(), "an empty live set is the point: nothing survived the restart"


def test_startup_revival_keeps_actual_current_owners(tmp_path, monkeypatch, startup_owners):
    from ouroboros import delegate_custody as dc, process_custody as pc, server_maintenance as sm

    queue, workers, post_task, registry = startup_owners
    queue.RUNNING["queue-live"] = {"task": {"id": "queue-live"}}
    workers.WORKERS[7] = SimpleNamespace(busy_task_id="worker-live")
    registry.register("native-live", 1)
    post_task.POST_TASK_SYNTHESIS_INFLIGHT[(str(tmp_path.resolve()), "post-live")] = None
    expected = {"queue-live", "worker-live", "native-live", "post-live"}
    for task_id in expected:
        dc.record_started(tmp_path, dc.RunCustody(
            run_id=f"run-{task_id}", task_id=task_id, route_id="r", model="m",
            project_id="p", project_owned=False, root_task_id=task_id, ledger_root=str(tmp_path),
        ))
    seen = {}
    transport = _LiveRunStub()
    real_reconcile = dc.reconcile_orphaned_runs
    def reconcile(root, **kwargs):
        seen["delegated"] = kwargs["running_task_ids"]
        kwargs["gateway_factory"] = lambda: transport
        outcomes = real_reconcile(root, **kwargs)
        seen["outcomes"] = outcomes
        return outcomes
    monkeypatch.setattr(dc, "reconcile_orphaned_runs", reconcile)
    monkeypatch.setattr(pc, "reap_orphaned_processes",
                        lambda root, **kw: seen.__setitem__("processes", kw["running_task_ids"]) or [])
    monkeypatch.setattr(sm, "_installed_skill_names", lambda: None)
    sm._startup_custody_sweep()
    # ONE source, evaluated by each surface after it has read its own candidates.
    assert seen["processes"] is seen["delegated"] is sm._live_task_ids
    assert seen["delegated"]() == expected
    assert seen["outcomes"] == []
    assert transport.cancels == []
    assert {row.run_id for row in dc.open_runs(tmp_path)} == {f"run-{task_id}" for task_id in expected}


def test_both_custody_surfaces_see_the_same_live_task_set(tmp_path, monkeypatch, startup_owners):
    """The periodic sweep must hand the delegated reconciler the SAME live task source
    the process reaper gets. Two copies of "is the owner still running" is exactly how
    one custody surface ends up reaping while its twin does not. The source is now the
    sweep's one CALLABLE (INV-B: each surface evaluates it after reading its own
    candidates), and the 600 s block itself rides a daemon thread beside the 20 s one."""
    import time
    import threading

    import ouroboros.server_maintenance as sm
    import ouroboros.delegate_custody as dc
    import ouroboros.process_custody as pc
    import supervisor.queue as queue
    import supervisor.task_lifecycle as lifecycle

    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    lock = threading.Lock()
    monkeypatch.setattr(sm, "_CANCEL_INTENT_SWEEP_LOCK", lock)
    monkeypatch.setattr(sm, "_LAST_CANCEL_INTENT_SWEEP", [0.0])
    entered, release = threading.Event(), threading.Event()
    threads = []
    def tracked_thread(**kwargs):
        thread = threading.Thread(**kwargs)
        threads.append(thread)
        return thread
    monkeypatch.setattr(sm, "threading", SimpleNamespace(Thread=tracked_thread))
    real_sweep = lifecycle.sweep_cancel_intents
    def delayed_sweep():
        entered.set()
        assert release.wait(5), "test must release its maintenance work"
        return real_sweep()
    monkeypatch.setattr(lifecycle, "sweep_cancel_intents", delayed_sweep)
    seen = {}
    monkeypatch.setattr(pc, "reap_orphaned_processes",
                        lambda root, **kw: seen.__setitem__("processes", kw.get("running_task_ids")) or [])
    monkeypatch.setattr(dc, "reconcile_orphaned_runs",
                        lambda root, **kw: seen.__setitem__("delegated", kw.get("running_task_ids")) or [])
    monkeypatch.setattr(sm, "_installed_skill_names", lambda: None)
    # Replace RUNNING wholesale, never append: queue globals are rebound by
    # init_queue_refs across the suite without restore (the upstream test
    # convention), so assuming the dict is empty here is cross-test fragile.
    monkeypatch.setattr(queue, "RUNNING", {"t-live": {}})
    from supervisor.active_activity import get_direct_activity_registry

    get_direct_activity_registry().register("native-live", 1)
    try:
        sm._periodic_supervisor_maintenance([0.0], [time.time()])
        assert entered.wait(2)
        assert [thread.name for thread in threads if thread.name not in {"cold-custody", "schedule-admission"}] == ["cancel_intent_sweep", "terminal_delivery_replay", "question_delivery_replay", "pause_notice_replay", "abandoned_usage_reconciliation", "post_task_synthesis_recovery", "custody-maintenance"]
        assert threads[0].is_alive(), "the 20 s sweep is still inside its delayed work"
        next(thread for thread in threads if thread.name == "custody-maintenance").join(5)
        assert seen["processes"] is seen["delegated"] is sm._live_task_ids
        assert seen["delegated"]() == {"t-live", "native-live"}, seen
    finally:
        # The actual maintenance owner finishes before monkeypatch restores its
        # root, lock and dependent functions, including when an assertion fails.
        release.set()
        for thread in threads:
            thread.join(timeout=5)
        assert all(not thread.is_alive() for thread in threads)
    assert not lock.locked(), "the real maintenance finally released its latch"
    assert not sm._CUSTODY_SWEEP_LOCK.locked(), "so did the custody block"


def test_an_orphaned_delegated_run_is_reconciled_when_its_owner_is_gone(tmp_path, monkeypatch):
    """The predicate is the one `process_custody.reap_orphaned_processes` already owns:
    the owning task is no longer in the supervisor's live set. A delegated run has no
    pid, so the process reaper cannot see it — but it is still spending quota and still
    writing to a workspace.

    The floor is INVERTED (owner B1-A): a live orphan is cancelled only behind a
    DELIBERATE owner terminal. An owner that died of the provider, one whose result
    is unreadable, and one whose result was never written all leave the run live
    (``left_live``, no cancel POST); the next sweep settles a spared run once the
    daemon reports it terminal."""
    import ouroboros.delegate_custody as dc
    from ouroboros.outcomes import infra_failed_axes
    from ouroboros.task_results import write_task_result

    live = _LiveRunStub(run_id="run-orphan")
    finished = _LiveRunStub(run_id="run-done")
    finished.get_run = lambda rid: {"lastSeq": 2, "summary": {"state": "succeeded", "spendUsd": 0.0}}

    for run_id, task in (("run-orphan", "t-gone"), ("run-done", "t-also-gone"),
                         ("run-spared", "t-provider-died"), ("run-garbled", "t-garbled"),
                         ("run-silent", "t-never-wrote")):
        dc.record_started(tmp_path, dc.RunCustody(
            run_id=run_id, task_id=task, route_id="r", model="m",
            project_id="p", project_owned=False, root_task_id=task, ledger_root=str(tmp_path)))
    dc.record_started(tmp_path, dc.RunCustody(
        run_id="run-alive", task_id="t-running", route_id="r", model="m",
        project_id="p", project_owned=False, root_task_id="t-running", ledger_root=str(tmp_path)))
    dc._CUSTODY.clear()
    # The owner verdicts: a deliberate completion, a provider death, an unreadable row.
    write_task_result(tmp_path, "t-gone", "completed", result="verdict")
    write_task_result(tmp_path, "t-provider-died", "failed", reason_code="provider_unavailable",
                      outcome_axes=infra_failed_axes("provider_unavailable"))
    (tmp_path / "task_results").mkdir(exist_ok=True)
    (tmp_path / "task_results" / "t-garbled.json").write_text("{not json", encoding="utf-8")
    terminal_ids = {"run-done"}

    class _Router(_LiveRunStub):
        def get_run(self, rid, **_kw):
            return (finished if rid in terminal_ids else live).get_run(rid)
        def cancel_run(self, rid, reason=""):
            return live.cancel_run(rid, reason)

    outcomes = dc.reconcile_orphaned_runs(tmp_path, {"t-running"}, gateway_factory=_Router)
    dc._CUSTODY.clear()
    by_run = {row["run_id"]: row for row in outcomes}
    assert set(by_run) == {"run-orphan", "run-done", "run-spared", "run-garbled", "run-silent"}, \
        "a live owner's run must be left alone"
    assert by_run["run-orphan"]["action"] == "cancelled"
    assert live.cancels == [("run-orphan", "owner_task_gone")], "only the deliberate terminal cancels"
    assert by_run["run-done"]["action"] == "settle_attempted" and by_run["run-done"]["settled"] is True
    for spared in ("run-spared", "run-garbled", "run-silent"):
        assert by_run[spared]["action"] == "left_live" and by_run[spared]["state"] == "running", spared
    # run-orphan stays open too: the stub answers "running" to the verify read, so its
    # cancel is merely REQUESTED (the pre-existing receipt vocabulary), not settled.
    assert {row.run_id for row in dc.open_runs(tmp_path)} == {
        "run-alive", "run-orphan", "run-spared", "run-garbled", "run-silent"}

    # The spared run settles itself on the sweep that finds it terminal.
    terminal_ids.add("run-spared")
    again = {row["run_id"]: row for row in dc.reconcile_orphaned_runs(tmp_path, {"t-running"}, gateway_factory=_Router)}
    dc._CUSTODY.clear()
    assert again["run-spared"]["action"] == "settle_attempted" and again["run-spared"]["settled"] is True
    assert again["run-silent"]["action"] == "left_live"
    # The requested-not-confirmed cancel is re-asked each sweep (unchanged); no spared
    # run was ever asked.
    assert live.cancels == [("run-orphan", "owner_task_gone")] * 2

    # Unknown liveness reconciles nothing: never mass-cancel on missing information.
    live.cancels.clear()
    assert dc.reconcile_orphaned_runs(tmp_path, None, gateway_factory=_Router) == []
    assert live.cancels == []
    dc._CUSTODY.clear()


def test_what_the_daemon_says_is_absent_is_closed_not_faulted_forever(tmp_path):
    """One root cause at two surfaces: a 404 is the daemon ANSWERING that the thing is not
    there, and both were read as "we could not find out".

    A run the daemon does not have was treated exactly like an unreachable daemon, so it
    was never settled, stayed in `open_runs`, and was re-faulted on EVERY pass — a
    permanent CRITICAL health invariant that no cancel or settlement could ever clear. Its
    sibling: a registration the daemon does not have kept `project_owned` true, so a
    terminal run could never finish settling and was reconciled forever."""
    import ouroboros.delegate_custody as dc

    class _NoSuchRun(_LiveRunStub):
        def get_run(self, rid, **_kw):
            raise cx.ClaudexorUnavailable("run_not_found", "no such run", status_code=404)

    class _NoSuchProject(_LiveRunStub):
        def get_run(self, rid, **_kw):
            return {"lastSeq": 2, "summary": {"state": "succeeded", "spendUsd": 0.0}}
        def remove_project(self, pid):
            raise cx.ClaudexorUnavailable("project_not_found", "no such project", status_code=404)

    dc._CUSTODY.clear()
    for run_id, task in (("run-gone", "t-gone"), ("run-owns-a-dead-project", "t-also-gone")):
        dc.record_started(tmp_path, dc.RunCustody(
            run_id=run_id, task_id=task, route_id="r", model="m", project_id="prj-ours",
            daemon_store_id="store-a",
            project_owned=run_id.endswith("project"), root_task_id=task, ledger_root=str(tmp_path)))
    dc._CUSTODY.clear()

    class _Router(_LiveRunStub):
        store_identity = "store-a"
        def get_run(self, rid, **_kw):
            return (_NoSuchProject() if rid.endswith("project") else _NoSuchRun()).get_run(rid)
        def remove_project(self, pid): _NoSuchProject().remove_project(pid)

    passes = []
    for _ in range(3):
        passes.append(dc.reconcile_orphaned_runs(tmp_path, {"t-live"}, gateway_factory=_Router))
        dc._CUSTODY.clear()
    assert [row["action"] for row in passes[0]] == ["absent", "settle_attempted"], passes[0]
    assert passes[1] == [] and passes[2] == [], "a closed run must not be reconciled again"
    assert dc.open_runs(tmp_path) == [], "neither run may stay open"
    assert dc.open_containment_faults(tmp_path) == []
    assert "DELEGATED RUN MAY STILL BE LIVE" not in _health_invariants(tmp_path)
    types = _event_types(tmp_path)
    assert "delegate_run_containment_fault" not in types, "absence is not a containment fault"
    assert "delegate_run_project_retire_failed" not in types, "absence IS discharge"
    # An absent run is CLOSED, not settled: no ledger row is invented for a run the daemon
    # cannot even describe.
    assert "delegate_run_closed_absent" in types
    rows = [json.loads(l) for l in (tmp_path / "logs" / "events.jsonl").read_text().splitlines()]
    ledgered = [r for r in rows if r.get("type") == "delegate_run_ledger_recorded"]
    assert [r["run_id"] for r in ledgered] == ["run-owns-a-dead-project"], ledgered

    # A daemon that is merely UNREACHABLE still faults: absence and ignorance stay apart.
    class _Deaf(_LiveRunStub):
        def get_run(self, rid, **_kw):
            raise cx.ClaudexorUnavailable("daemon_unreachable", "connection refused")

    dc.record_started(tmp_path, dc.RunCustody(
        run_id="run-unknown", task_id="t-gone", route_id="r", model="m", project_id="p",
        project_owned=False, root_task_id="t-gone", ledger_root=str(tmp_path)))
    dc._CUSTODY.clear()
    assert [row["action"] for row
            in dc.reconcile_orphaned_runs(tmp_path, {"t-live"}, gateway_factory=_Deaf)] == ["unreadable"]
    dc._CUSTODY.clear()
    assert [f["run_id"] for f in dc.open_containment_faults(tmp_path)] == ["run-unknown"]


def test_a_terminalizing_parent_releases_the_run_it_still_holds(tmp_path):
    """The in-process twin of reconciliation. A parent that finishes while its delegated
    run is still going used to leave it mutating until the next 10-minute sweep; the
    loop's own resource-release point now settles or cancels it like any held resource.
    A task that delegated nothing must pay nothing for this.

    Inverted floor (B1-A): the release cancels only behind the parent's DELIBERATE
    durable terminal. Without one (the ordinary loop-exit shape: the result is written
    after this point) the run is left live and disclosed as open for the next sweep."""
    import ouroboros.delegate_custody as dc
    from ouroboros.task_results import load_task_result, write_task_result

    live = _LiveRunStub(run_id="run-held")
    dc._CUSTODY.clear()
    dc.record_started(tmp_path, dc.RunCustody(
        run_id="run-held", task_id="t-parent", route_id="r",
        model="m", project_id="p", project_owned=False, ledger_root=str(tmp_path),
    ))
    assert dc.release_task_runs(tmp_path, "t-someone-else", gateway_factory=lambda: live) == []
    assert live.cancels == [], "another task's run is not this task's to release"

    unwritten = dc.release_task_runs(tmp_path, "t-parent", gateway_factory=lambda: live)
    dc._CUSTODY.clear()
    assert [row["action"] for row in unwritten] == ["left_live"]
    assert live.cancels == [], "no verdict yet: the run outlives the loop exit"
    assert load_task_result(tmp_path, "t-parent")["delegated_runs_unreconciled"] == ["run-held"]

    write_task_result(tmp_path, "t-parent", "completed", result="done on purpose")
    outcomes = dc.release_task_runs(tmp_path, "t-parent", gateway_factory=lambda: live)
    dc._CUSTODY.clear()
    assert [row["action"] for row in outcomes] == ["cancelled"]
    assert live.cancels == [("run-held", "owner_task_gone")]


def test_the_loops_own_release_point_reaches_the_delegated_reconciler(tmp_path, monkeypatch):
    """`release_task_runs` only helps if something CALLS it. The test beside this one drives
    the function directly, so it passed with the loop's wiring deleted — and the loop is
    the ordinary path: without it a terminalized parent leaves its run mutating until the
    next ten-minute sweep. The release must also read the CANONICAL root, not the child
    drive the subagent runs on, or it looks for custody where none was written."""
    from types import SimpleNamespace

    import ouroboros.delegate_custody as dc
    import ouroboros.loop as loop

    released = []
    monkeypatch.setattr(dc, "release_task_runs",
                        lambda root, task_id, **kw: released.append((str(root), task_id)) or [])
    canonical = tmp_path / "canonical"
    inner = SimpleNamespace(drive_root=tmp_path / "child",
                            task_metadata={"budget_drive_root": str(canonical)})
    loop._cleanup_loop_resources(None, loop._LoopExitContext(
        tools=SimpleNamespace(_ctx=inner), drive_root=tmp_path, task_id="t-parent",
        event_queue=None, drive_logs=tmp_path / "logs", accumulated_usage={}, llm_trace={}))
    assert released == [(str(canonical), "t-parent")], released


def test_a_breach_whose_cancel_was_never_verified_is_not_reported_as_cancelled(
    tmp_path, monkeypatch
):
    """A containment BREACH stops the run through the one verified cancel path, and the
    sentence the agent reads comes from that cancel's typed outcome.

    The ad-hoc cancel this replaced swallowed every exception into a log line and then
    said "The run was cancelled. Do not retry it" unconditionally — so a daemon that
    REFUSED the cancel, or that could not be reached to confirm it, left an overpowered
    run mutating a workspace while the agent was told it had stopped. That is exactly
    what `record_containment_fault`'s own contract forbids: an incident must surface as
    a critical health invariant, "never as a reassuring string in a tool result".
    """
    from ouroboros.gateways import claudexor as gw
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    import ouroboros.delegate_custody as dc

    run_dir = tmp_path / "run-1"
    home = tmp_path / "operator-home"
    home.mkdir()
    monkeypatch.setattr(cx, "operator_home", lambda: home)

    class _RefusingStub:
        engine_version = CLAUDEXOR_DELEGATED_MARKER_MIN_VERSION

        def handshake(self, **_kw): return {}

        def get_run(self, rid, **_kw):
            # Still RUNNING: the cancel changed nothing the daemon will confirm.
            return {"lastSeq": 7, "summary": {
                "state": "running", "effectiveAccess": "workspace_write",
                "runDir": str(run_dir),
            }}

        def cancel_run(self, rid, reason=""):
            raise ClaudexorUnavailable("control_refused", "daemon refused the cancel")

        def remove_project(self, pid): pass
        def close(self): pass

    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _RefusingStub())
    _write_attempt(run_dir, isolated=False, home_dir=str(home))

    out = _waiting(tmp_path, monkeypatch)

    assert out["status"] == "refused" and out["reason"] == "home_isolation_not_applied", out
    # The typed outcome rides out with the refusal instead of a comforting sentence.
    assert out["cancel_outcome"] == dc.CANCEL_CONTAINMENT_FAULT, out
    assert "CONTAINMENT FAULT" in out["detail"], out["detail"]
    assert "MAY STILL BE LIVE" in out["detail"], out["detail"]
    assert "The run was cancelled." not in out["detail"], out["detail"]


def test_the_admitted_sweep_names_the_ensure_transport_and_nothing_defaults(tmp_path, monkeypatch):
    """Regression (v6.89.0): the startup sweep reaps the previous generation's owned
    daemon and THEN reconciled through a bare discovery-only gateway — which always
    found the corpse it had just made, so every restart's reconciliation silently
    no-opped and open runs stayed unsettled until the next delegate_start. The
    admitted sweep (``server_maintenance._reconcile_delegated_runs``) therefore NAMES
    the ensure transport. Nothing defaults to it any more: the loop-exit release of
    issue #1772 admitted a daemon through the old ``_reconcile_each`` default, so a
    pass without an explicit factory is refused instead of starting one."""
    from ouroboros import delegate_custody as dc, server_maintenance as sm

    dc.record_started(tmp_path, dc.RunCustody(
        run_id="run-orphan", task_id="t-gone", route_id="r", model="m",
        ledger_root=str(tmp_path)))
    dc._CUSTODY.clear()

    ensured = []

    class _EnsuredGateway:
        def handshake(self): return {}
        def get_run(self, run_id, timeout_sec=None):
            return {"state": "cancelled", "summary": {"state": "cancelled"}}
        def cancel_run(self, run_id): return {}
        def close(self): pass

    def _fake_ensure(**_kw):
        ensured.append(True)
        return _EnsuredGateway()

    monkeypatch.setattr("ouroboros.claudexor_daemon.ensure_owned_gateway", _fake_ensure)
    monkeypatch.setattr("ouroboros.claudexor_daemon.read_owned_gateway",
                        lambda **_kw: pytest.fail("the admitted sweep ensures, it does not attach"))
    monkeypatch.setattr(sm, "DATA_DIR", tmp_path)
    monkeypatch.setattr(sm, "_stop_requested", lambda stop_event=None: False)
    sm._reconcile_delegated_runs(lambda: set())
    assert ensured, "the admitted sweep's gateway factory must go through ensure_owned_gateway"

    # A bare sweep call names no factory: it ATTACHES (never ensures) and replays nothing.
    dc.record_started(tmp_path, dc.RunCustody(
        run_id="run-orphan-2", task_id="t-gone-2", route_id="r", model="m", ledger_root=str(tmp_path)))
    dc._CUSTODY.clear()
    attached = []
    monkeypatch.setattr("ouroboros.claudexor_daemon.read_owned_gateway",
                        lambda **_kw: attached.append(True) or _EnsuredGateway())
    ensured.clear()
    dc.reconcile_orphaned_runs(tmp_path, set())
    assert attached and not ensured
    with pytest.raises(ValueError):
        dc._reconcile_each(tmp_path, [], None)  # the core still refuses a missing factory

    # And with NOTHING to reconcile the daemon is never started at all: the empty
    # early-return keeps the ordinary idle restart free of a daemon spawn.
    ensured.clear()
    empty = tmp_path / "empty-drive"
    empty.mkdir()
    assert dc.reconcile_orphaned_runs(empty, set(), gateway_factory=_fake_ensure,
                                      replay_pending=lambda: True) == []
    assert not ensured


def test_the_kill_boundary_states_its_own_verdict_before_it_writes_it(tmp_path):
    """An owner cancellation still cancels the paid run, at the kill boundary.

    The A4 ordering audits custody BEFORE the terminal write, so the durable
    result the inverted floor reads does not exist yet and the run would have
    survived until the next periodic sweep (up to ten minutes of paid work).
    The killing caller already knows the verdict, so it states it. A host bound
    that has no verdict passes nothing and still spares the run (owner B1-A).
    """
    import ouroboros.delegate_custody as dc

    def _started(run_id: str, task_id: str) -> None:
        dc._CUSTODY.clear()
        dc.record_started(tmp_path, dc.RunCustody(
            run_id=run_id, task_id=task_id, route_id="r", model="m",
            project_id="p", project_owned=False, root_task_id=task_id,
            ledger_root=str(tmp_path)))
        dc._CUSTODY.clear()

    _started("run-owner-cancelled", "t-owner-cancel")
    deliberate = _LiveRunStub(run_id="run-owner-cancelled")
    outcomes = dc.reconcile_task_runs(
        tmp_path, "t-owner-cancel", gateway_factory=lambda: deliberate,
        deliberate_terminal="cancelled",
    )
    assert [row["action"] for row in outcomes] == ["cancelled"]
    assert deliberate.cancels == [("run-owner-cancelled", "owner_task_gone")]

    _started("run-no-verdict", "t-no-verdict")
    spared = _LiveRunStub(run_id="run-no-verdict")
    outcomes = dc.reconcile_task_runs(
        tmp_path, "t-no-verdict", gateway_factory=lambda: spared,
    )
    assert [row["action"] for row in outcomes] == ["left_live"]
    assert spared.cancels == []
    dc._CUSTODY.clear()


def test_the_supervisor_kill_path_carries_the_cancellation_verdict(tmp_path, monkeypatch):
    """End to end over the supervisor seam the finder probed: the cancel kill
    path passes its own terminal into the audit, a reap passes nothing."""
    import types

    import ouroboros.delegate_terminal as delegate_terminal
    from supervisor.cancel_publication import _audit_delegated_runs_on_kill

    seen: list = []
    monkeypatch.setattr(
        delegate_terminal, "terminal_reconcile_task",
        lambda root, tid, **kw: seen.append(kw.get("deliberate_terminal", "")) or {
            "task_id": tid, "trigger": kw.get("trigger", ""), "outcomes": [],
            "unreconciled": [], "audit_status": "ok",
        },
    )
    q = types.SimpleNamespace(DRIVE_ROOT=str(tmp_path))

    _audit_delegated_runs_on_kill(q, "t1", deliberate_terminal="cancelled")
    _audit_delegated_runs_on_kill(q, "t1", trigger="reaper_deadline_exceeded")

    assert seen == ["cancelled", ""]


def test_the_kill_path_claims_a_verdict_only_when_it_writes_one(tmp_path, monkeypatch):
    """A settled task keeps its own durable verdict, so the kill path claims none.

    For an UNSETTLED task this path writes the cancelled terminal, so stating it
    to the audit that runs first is the truth. For a task that had ALREADY
    settled when the cancel arrived, completion wins on the write and the stored
    terminal stays whatever the task decided; claiming a cancellation there would
    cancel a healthy paid run behind an infrastructure failure, which is exactly
    the class owner answer B1-A spares.
    """
    import types

    import supervisor.queue as q
    from supervisor import task_lifecycle, workers
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result

    monkeypatch.setattr(q, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(q, "RUNNING", {}, raising=False)
    monkeypatch.setattr(workers, "WORKERS", {}, raising=False)

    class _Stop(BaseException):
        """Ends the lifecycle right after the audit call under test."""

    seen: list = []

    def _audit(_q, _task_id, **kwargs):
        seen.append(kwargs.get("deliberate_terminal", ""))
        raise _Stop

    monkeypatch.setattr(task_lifecycle, "_audit_delegated_runs_on_kill", _audit)
    monkeypatch.setattr(task_lifecycle, "_reconcile_dead_review_owner",
                        lambda *_a, **_kw: None)
    monkeypatch.setattr(task_lifecycle, "_restore_custody", lambda *_a, **_kw: None)
    worker = types.SimpleNamespace(proc=types.SimpleNamespace(
        is_alive=lambda: False, pid=0,
        join=lambda timeout=None: None, terminate=lambda: None,
    ))

    for settled in ("", STATUS_COMPLETED):
        if settled:
            write_task_result(tmp_path, "t-kill", settled, result="Completed before Stop")
        with pytest.raises(_Stop):
            task_lifecycle._finish_captured_running(
                "t-kill", worker, {"task": {"id": "t-kill"}}, intent=None, deliver=False,
                settled_status=settled,
            )

    assert seen == ["cancelled", ""]


@pytest.mark.parametrize("axes,expected,cancelled", [
    ("infra", "left_live", []),
    ("deliberate", "cancelled", [("run-settled-owner", "owner_task_gone")]),
    ("review_deliberate", "left_live", []),
])
def test_a_settled_owner_lets_its_durable_outcome_decide_the_live_run(
    tmp_path, axes, expected, cancelled,
):
    """The audit with no claimed verdict falls through to the durable predicate.

    This is what the kill path now does for an already-settled task: an
    infrastructure death spares the live run, a deliberate completion cancels it.
    The same deliberate completion is NOT a verdict about a run the review panel
    owns, so that row is spared by the same read (issue #1006).
    """
    import ouroboros.delegate_custody as dc
    from ouroboros.outcomes import infra_failed_axes
    from ouroboros.task_results import write_task_result

    dc._CUSTODY.clear()
    dc.record_started(tmp_path, dc.RunCustody(
        run_id="run-settled-owner", task_id="t-settled", route_id="r", model="m",
        project_id="p", project_owned=False, root_task_id="t-settled",
        source="review_substrate" if axes == "review_deliberate" else "delegated_subagent",
        ledger_root=str(tmp_path)))
    dc._CUSTODY.clear()
    if axes == "infra":
        write_task_result(tmp_path, "t-settled", "failed",
                          reason_code="provider_unavailable",
                          outcome_axes=infra_failed_axes("provider_unavailable"))
    else:
        write_task_result(tmp_path, "t-settled", "completed", result="verdict")

    transport = _LiveRunStub(run_id="run-settled-owner")
    outcomes = dc.reconcile_task_runs(
        tmp_path, "t-settled", gateway_factory=lambda: transport)

    assert [row["action"] for row in outcomes] == [expected]
    assert transport.cancels == cancelled
    assert outcomes[0].get("reason") == (
        "review_panel_owns_run" if axes == "review_deliberate" else None)
    dc._CUSTODY.clear()


def _panel_run(tmp_path, source="review_substrate", task_id="t-reviewed",
               run_id="run-panel"):
    """One live acceptance-review slot registered under the REVIEWED task's id —
    the exact shape of the incident behind issue #1006."""
    import ouroboros.delegate_custody as dc

    dc._CUSTODY.clear()
    dc.record_started(tmp_path, dc.RunCustody(
        run_id=run_id, task_id=task_id, route_id="codex", model="m",
        source=source, category="task_acceptance_review",
        review_slot_id="triad_286lhb", ledger_root=str(tmp_path)))
    dc._CUSTODY.clear()
    return dc


@pytest.mark.parametrize("source", ["review_substrate", "review_substrate:task_acceptance"])
@pytest.mark.parametrize("owner,expected", [
    ("completed", "left_live"),
    ("failed", "left_live"),
    ("cancelled", "cancelled"),
])
def test_a_review_panels_run_outlives_every_owner_terminal_but_cancellation(
    tmp_path, source, owner, expected,
):
    """The reviewed task's own terminal is not a verdict about its
    reviewer. Only an explicit cancellation of the task speaks for the panel too;
    a completion — the live incident — and an agent-declared failure spare it.
    Every durable ``review_substrate*`` spelling reads the same."""
    from ouroboros.task_results import write_task_result

    dc = _panel_run(tmp_path, source=source)
    write_task_result(tmp_path, "t-reviewed", owner, result="verdict")

    transport = _LiveRunStub(run_id="run-panel")
    outcomes = dc.reconcile_orphaned_runs(
        tmp_path, set(), gateway_factory=lambda: transport)

    assert [row["action"] for row in outcomes] == [expected]
    if expected == "left_live":
        assert outcomes[0]["reason"] == "review_panel_owns_run"
        assert transport.cancels == []
    else:
        assert transport.cancels == [("run-panel", "owner_task_gone")]
    dc._CUSTODY.clear()


@pytest.mark.parametrize("deliberate,expected", [
    ("cancelled", "cancelled"),
    ("completed", "left_live"),
])
def test_only_a_claimed_cancellation_speaks_for_a_review_panels_run(
    tmp_path, deliberate, expected,
):
    """The kill boundary audits custody BEFORE writing its terminal, so it states
    its verdict inline. For a panel-owned run only the cancellation it is about to
    write counts; any other claimed terminal leaves the reviewer live."""
    dc = _panel_run(tmp_path)
    transport = _LiveRunStub(run_id="run-panel")
    outcomes = dc.reconcile_task_runs(
        tmp_path, "t-reviewed", gateway_factory=lambda: transport,
        deliberate_terminal=deliberate)

    assert [row["action"] for row in outcomes] == [expected]
    assert transport.cancels == (
        [("run-panel", "owner_task_gone")] if expected == "cancelled" else [])
    dc._CUSTODY.clear()


def test_a_terminal_review_run_still_settles_through_the_sweep(tmp_path):
    """Physical custody keeps seeing every run: sparing is about the DELEGATION
    domain, not about leaving a finished reviewer unsettled and unledgered."""
    dc = _panel_run(tmp_path)

    class _Finished(_LiveRunStub):
        def get_run(self, rid, **_kw):
            return {"lastSeq": 2, "summary": {"state": "succeeded", "spendUsd": 0.0}}

    outcomes = dc.reconcile_orphaned_runs(tmp_path, set(), gateway_factory=_Finished)

    assert [row["action"] for row in outcomes] == ["settle_attempted"]
    assert outcomes[0]["settled"] is True
    assert [row.run_id for row in dc.open_runs(tmp_path)] == []
    dc._CUSTODY.clear()


def test_a_pending_review_invocation_is_left_to_its_panel_never_re_posted(tmp_path):
    """The review substrate rejoins its own pending invocation by its exact key
    (``review_session_custody``); a delegation sweep that re-POSTed it here would
    bind a second reviewer nobody on this task asked for. The periodic sweep leaves
    it alone entirely (owner Q2=A: 9 871 identical "retained" rows a day, each a
    store commit, an fsynced append and an archive scan, were the old cost); the
    kill path still names it once."""
    import ouroboros.delegate_custody as dc

    dc._CUSTODY.clear()
    assert dc.record_start_requested(
        tmp_path, run_id="", task_id="t-reviewed", invocation_id="inv-panel",
        idempotency_key="inv-panel", request={"prompt": "review packet"},
        route="codex", source="review_substrate.extraction")
    dc._CUSTODY.clear()

    for _ in range(3):
        assert dc.reconcile_orphaned_runs(
            tmp_path, set(), gateway_factory=lambda: pytest.fail("no sweep work opens a gateway"),
            replay_pending=lambda: True) == []  # the admitted sweep's shape
    assert _event_types(tmp_path).count("delegate_run_reconciled") == 0
    assert [record["invocation_id"] for record in dc.pending_invocations(tmp_path)] == ["inv-panel"]

    assert dc.reconcile_task_runs(  # exit does not admit: naming it opens no transport either
            tmp_path, "t-reviewed", gateway_factory=lambda: pytest.fail("a kill path opened a gateway")) == [{
        "invocation_id": "inv-panel", "task_id": "t-reviewed",
        "action": "invocation_retained",
        "reason": "review_panel_owns_invocation"}]
    assert _event_types(tmp_path).count("delegate_run_reconciled") == 1
    dc._CUSTODY.clear()


@pytest.mark.serial
@pytest.mark.parametrize("pending", [False, True], ids=["bound", "pending"])
@pytest.mark.parametrize("captured,case", [
    (captured, case) for captured in (False, True)
    for case in ("owner_stop", "snapshot_restore", "prior_infra", "child_infra")
] + [(True, "child_unreadable")])
def test_cancellation_uses_adopted_terminal_for_live_delegation(qenv, monkeypatch, captured, case, pending):
    """A fresh Stop requests cancellation before publication, including after restore.

    A prior provider failure, even one only found by child-result copyback, keeps
    its outcome and its healthy delegated run. Exercise the real custody/audit
    path; only the external transport and owner delivery are replaced.
    """
    from ouroboros import cancel_intents, claudexor_daemon, delegate_custody as dc
    from ouroboros.outcomes import infra_failed_axes
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor import cancel_publication

    root, task_id, run_id = qenv.drive, "owner", "run-owned"
    status_at_cancel = []
    recovered_keys = []

    class Gateway(_LiveRunStub):
        def start_run(self, request, *, idempotency_key=""):
            recovered_keys.append(idempotency_key)
            return {"runId": run_id}

        def cancel_run(self, rid, reason=""):
            status_at_cancel.append(load_task_result(root, task_id)["status"])
            return super().cancel_run(rid, reason)

    transport = Gateway(run_id=run_id)
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", lambda: transport)
    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", lambda: transport)
    monkeypatch.setattr(cancel_publication, "_deliver_on_miss", lambda *a, **kw: True)
    monkeypatch.setattr(qenv.tl, "_deliver_on_miss", lambda *a, **kw: True)
    monkeypatch.setattr(qenv.q, "_emit_cancel_task_done", lambda *a, **kw: None)
    monkeypatch.setattr(dc, "_CUSTODY", {})
    if pending:
        assert dc.record_start_requested(
            root, task_id=task_id, invocation_id="inv-owned", idempotency_key="inv-owned",
            route="r", project_id="p", project_owned=False, root_task_id=task_id,
            request={"prompt": "stored request", "primaryHarness": "r", "model": "m"},
        )
    else:
        dc.record_started(root, dc.RunCustody(
            run_id=run_id, task_id=task_id, route_id="r", model="m", project_id="p",
            project_owned=False, root_task_id=task_id, ledger_root=str(root),
        ))
    dc._CUSTODY.clear()  # Recover the actual durable custody, as after restart.

    fields = {"root_task_id": task_id, "result": "working"}
    failure = {"result": "provider died", "outcome_axes": infra_failed_axes("provider_unavailable")}
    if case in {"child_infra", "child_unreadable"}:
        child = root / "state" / "headless_tasks" / task_id / "data"
        write_task_result(child, task_id, "failed", root_task_id=task_id, **failure)
        fields.update(child_drive_root=str(child), delegation_role="subagent")
        if case == "child_unreadable":
            unreadable = unreadable_result(monkeypatch, task_id, root=child)
    if case == "prior_infra":
        fields.update(failure)
    write_task_result(root, task_id, "failed" if case == "prior_infra" else "running", **fields)
    restoring = case == "snapshot_restore"
    cancel_intents.request_cancel(
        root, task_id, source="snapshot_restore" if restoring else "owner",
        reason="server_shutdown" if restoring else "Stop",
        allow_settled_target=case == "prior_infra",
    )
    if captured:
        class WorkerProcess:
            pid = 0  # No OS process or signal: exercise real custody around this facade.
            alive = True

            def is_alive(self): return self.alive
            def join(self, timeout=None): pass
            def terminate(self): self.alive = False

        task = {"id": task_id, "chat_id": 0, "root_task_id": task_id, "drive_root": str(root)}
        if "child_drive_root" in fields:
            task.update(child_drive_root=str(child), drive_root=str(child), delegation_role="subagent")
        worker = SimpleNamespace(wid=0, busy_task_id=task_id, reaping=False, proc=WorkerProcess())
        qenv.workers.WORKERS[0] = worker
        qenv.q.RUNNING[task_id] = {"task": task, "worker_id": 0}
        monkeypatch.setattr(qenv.tl, "_reconcile_dead_review_owner", lambda *a, **kw: None)

    outcome = qenv.tl.cancel_task_custody(task_id, deliver=False)
    stored = load_task_result(root, task_id)
    if case == "child_unreadable":
        assert outcome == qenv.tl.CANCEL_FAILED and stored["status"] == "running"
        assert transport.cancels == [] and recovered_keys == []
        assert task_id in qenv.q.RUNNING and not worker.proc.is_alive()
        assert cancel_intents.cancel_pending(root, task_id) and child.is_dir()
        # The same open intent can finish after the child's store answers again;
        # no new scheduler, flag or cancellation claim is needed.
        unreadable["on"] = False
        assert qenv.tl.cancel_task_custody(task_id, deliver=False) == qenv.tl.CANCEL_ALREADY_SETTLED
        assert load_task_result(root, task_id)["status"] == "failed"
        assert transport.cancels == []
        assert recovered_keys == []  # a kill boundary never re-POSTs a pending start
        assert not cancel_intents.cancel_pending(root, task_id) and task_id not in qenv.q.RUNNING
        return
    # Exit does not admit (#1772): the kill boundary attaches and settles the runs it
    # can see; a START_REQUESTED-only invocation is never replayed here, whatever the
    # verdict. It stays durable and disclosed, and the admitted sweep recovers it
    # under the same key once the owner is gone (asserted below for the cancel cases).
    assert recovered_keys == []
    if case in {"owner_stop", "snapshot_restore"}:
        assert outcome == qenv.tl.CANCEL_CANCELLED
        assert stored["status"] == "cancelled"
        if pending:
            assert transport.cancels == []
            assert stored["delegated_runs_unreconciled"] == ["invocation:inv-owned"]
            swept = dc.reconcile_orphaned_runs(root, set(), gateway_factory=lambda: transport,
                                               replay_pending=lambda: True)
            assert recovered_keys == ["inv-owned"]
            assert [row["action"] for row in swept] == ["cancelled"]
            assert transport.cancels == [(run_id, "owner_task_gone")]
        else:
            assert transport.cancels == [(run_id, "owner_task_gone")]
            assert status_at_cancel == ["running"], "request precedes the cancelled write"
            # The transport still reports running: a request cannot claim physical death.
            assert stored["delegated_runs_unreconciled"] == [run_id]
    else:
        assert outcome == qenv.tl.CANCEL_ALREADY_SETTLED
        assert stored["status"] == "failed" and stored["result"] == "provider died"
        assert stored["outcome_axes"] == failure["outcome_axes"]
        assert transport.cancels == [] and status_at_cancel == []


def test_pending_recovery_without_owner_verdict_leaves_live_run(tmp_path, monkeypatch):
    """The periodic recovery default still spares an owner with no known outcome."""
    from ouroboros import delegate_custody as dc

    monkeypatch.setattr(dc, "_CUSTODY", {})
    assert dc.record_start_requested(
        tmp_path, task_id="owner", invocation_id="inv-unknown", route="r",
        project_id="p", project_owned=False, request={"prompt": "original work", "model": "m"},
    )
    recovered = []

    class Gateway(_LiveRunStub):
        def start_run(self, request, *, idempotency_key=""):
            recovered.append((request, idempotency_key))
            return {"runId": "run-unknown"}

    gateway = Gateway(run_id="run-unknown")
    outcomes = dc.reconcile_orphaned_runs(tmp_path, set(), gateway_factory=lambda: gateway,
                                          replay_pending=lambda: True)
    assert recovered == [({"prompt": "original work", "model": "m"}, "inv-unknown")]
    assert [row["action"] for row in outcomes] == ["left_live"] and gateway.cancels == []
