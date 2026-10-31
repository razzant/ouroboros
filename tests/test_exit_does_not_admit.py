"""Exit does not admit (issue #1772): a release, kill, stop or Restart path attaches
and settles; it never starts a daemon and never replays a pending invocation.

The leak behind the issue: ``release_task_runs`` at loop exit fell through to the
``ensure_owned_gateway`` default of ``_reconcile_each`` and re-POSTed a pending
START_REQUESTED row, so a finished task's exit downloaded, spawned and admitted an
engine. Replay is the admitted startup/periodic sweep's duty alone, with admission
re-asked right before each re-POST. Every guard here asserts both directions.
"""
from __future__ import annotations

import pytest

from tests._delegated_transport_shared import _LiveRunStub

TASK = "t-ended"
INVOCATION = "inv-pending"


def _pending(root, task_id=TASK, invocation_id=INVOCATION, **extra):
    from ouroboros import delegate_custody as dc

    dc._CUSTODY.clear()
    assert dc.record_start_requested(
        root, run_id="", task_id=task_id, root_task_id=task_id, invocation_id=invocation_id,
        idempotency_key=invocation_id, route="r", project_id="p", project_owned=False,
        request={"prompt": "stored request", "primaryHarness": "r", "model": "m"}, **extra)
    dc._CUSTODY.clear()
    return dc


def _held(root, run_id="run-held", task_id=TASK):
    from ouroboros import delegate_custody as dc

    dc._CUSTODY.clear()
    dc.record_started(root, dc.RunCustody(
        run_id=run_id, task_id=task_id, route_id="r", model="m", project_id="p",
        project_owned=False, root_task_id=task_id, ledger_root=str(root)))
    dc._CUSTODY.clear()
    return dc


class _Recorder(_LiveRunStub):
    """A serving daemon that records every start it is asked for."""

    def __init__(self, run_id="run-recovered", state="running"):
        super().__init__(run_id=run_id, state=state)
        self.started: list = []

    def start_run(self, request, *, idempotency_key=""):
        self.started.append((dict(request), idempotency_key))
        return {"runId": self.run_id}


def _pending_ids(root, task_id=TASK):
    from ouroboros import delegate_custody as dc

    return [row["invocation_id"] for row in dc.pending_invocations(root) if row["task_id"] == task_id]


@pytest.fixture
def transports(monkeypatch):
    """A live attach transport and an ensure that FAILS the test when touched."""
    from ouroboros import claudexor_daemon

    live = _Recorder()
    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", lambda **_kw: live)
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway",
                        lambda **_kw: pytest.fail("a release path admitted work through ensure_owned_gateway"))
    return live


def test_the_loop_exit_release_attaches_only_and_leaves_the_pending_row(tmp_path, transports):
    dc = _pending(tmp_path)
    _held(tmp_path)

    outcomes = dc.release_task_runs(tmp_path, TASK)  # the loop-exit call shape of loop_budget

    assert [row["action"] for row in outcomes] == ["left_live"]  # the held run was judged over attach
    assert transports.started == [], "a release never re-POSTs a pending invocation"
    assert _pending_ids(tmp_path) == [INVOCATION], "the row stays durable for the admitted sweep"


def test_the_kill_boundary_cancels_what_it_sees_and_replays_nothing(tmp_path, transports):
    from ouroboros.delegate_custody_reconcile import attach_only_gateway

    dc = _pending(tmp_path)
    _held(tmp_path)

    outcomes = dc.reconcile_task_runs(tmp_path, TASK, gateway_factory=attach_only_gateway,
                                      deliberate_terminal="cancelled")

    assert [row["action"] for row in outcomes] == ["cancelled"]
    assert transports.cancels == [("run-held", "owner_task_gone")]
    assert transports.started == []
    assert _pending_ids(tmp_path) == [INVOCATION]


def test_a_pending_row_alone_opens_no_transport_on_a_release_path(tmp_path, monkeypatch):
    from ouroboros import claudexor_daemon

    dc = _pending(tmp_path)
    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway",
                        lambda **_kw: pytest.fail("nothing to settle: no transport is opened"))
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway",
                        lambda **_kw: pytest.fail("a release path admitted work"))

    assert dc.release_task_runs(tmp_path, TASK) == []
    assert _pending_ids(tmp_path) == [INVOCATION]


def test_the_terminal_boundary_defaults_to_the_attach_only_transport(tmp_path, transports):
    from ouroboros.delegate_terminal import terminal_reconcile_task

    _held(tmp_path, run_id="run-terminal")
    result = terminal_reconcile_task(tmp_path, TASK)  # no factory named: attach only
    assert [row["action"] for row in result["outcomes"]] == ["left_live"]
    assert result["open_run_ids"] == ["run-terminal"]


def test_the_admitted_sweep_recovers_an_ended_tasks_pending_row_under_the_same_key(tmp_path):
    dc = _pending(tmp_path)
    ensured = _Recorder()

    outcomes = dc.reconcile_orphaned_runs(tmp_path, set(), gateway_factory=lambda: ensured,
                                          replay_pending=lambda: True)

    assert ensured.started == [({"prompt": "stored request", "primaryHarness": "r", "model": "m"}, INVOCATION)]
    assert [row["action"] for row in outcomes] == ["left_live"]  # recovered, then judged like any run
    assert _pending_ids(tmp_path) == []


@pytest.mark.parametrize("admission", [None, lambda: False], ids=["no-predicate", "refused"])
def test_the_sweep_without_admission_retains_the_row_and_opens_nothing(tmp_path, admission):
    dc = _pending(tmp_path)

    outcomes = dc.reconcile_orphaned_runs(
        tmp_path, set(), gateway_factory=lambda: pytest.fail("no transport for rows this pass may not replay"),
        replay_pending=admission)

    assert outcomes == [{"invocation_id": INVOCATION, "task_id": TASK,
                         "action": "invocation_retained", "reason": "replay_not_admitted"}]
    assert _pending_ids(tmp_path) == [INVOCATION]


def test_replay_admission_is_rechecked_before_each_re_post(tmp_path):
    dc = _pending(tmp_path, invocation_id="inv-first")
    _pending(tmp_path, task_id="t-other", invocation_id="inv-second")
    ensured = _Recorder()
    answers = iter([True, True, False])  # the gate, the first dispatch, a stop before the second

    outcomes = dc.reconcile_orphaned_runs(tmp_path, set(), gateway_factory=lambda: ensured,
                                          replay_pending=lambda: next(answers))

    assert [key for _request, key in ensured.started] == ["inv-first"]
    retained = [row for row in outcomes if row["action"] == "invocation_retained"]
    assert retained == [{"invocation_id": "inv-second", "task_id": "t-other",
                         "action": "invocation_retained", "reason": "replay_not_admitted"}]
    assert _pending_ids(tmp_path, "t-other") == ["inv-second"]


def test_a_live_or_reserved_owner_keeps_its_pending_row_even_when_replay_is_admitted(tmp_path):
    dc = _pending(tmp_path, task_id="t-live", invocation_id="inv-live")
    _pending(tmp_path, task_id="t-reserved", invocation_id="inv-reserved")
    ensured = _Recorder()

    outcomes = dc.reconcile_orphaned_runs(tmp_path, {"t-live"}, gateway_factory=lambda: ensured,
        recoverable_handles={("t-reserved", "", "inv-reserved")}, replay_pending=lambda: True)

    assert outcomes == [] and ensured.started == []
    assert _pending_ids(tmp_path, "t-live") == ["inv-live"]
    assert _pending_ids(tmp_path, "t-reserved") == ["inv-reserved"]


def test_an_exact_pending_reservation_does_not_protect_another_start_of_its_task(tmp_path):
    dc = _pending(tmp_path, invocation_id="inv-reserved")
    _pending(tmp_path, invocation_id="inv-independent")
    ensured = _Recorder()
    dc.reconcile_orphaned_runs(tmp_path, set(), gateway_factory=lambda: ensured,
        recoverable_handles={(TASK, "", "inv-reserved")}, replay_pending=lambda: True)
    assert [key for _request, key in ensured.started] == ["inv-independent"]
    assert _pending_ids(tmp_path) == ["inv-reserved"]


def test_the_stopping_sweep_attaches_and_replays_nothing_while_the_admitted_one_ensures(tmp_path, monkeypatch):
    from ouroboros import claudexor_daemon, server_maintenance as sm

    dc = _pending(tmp_path)
    attached, ensured = _Recorder(), _Recorder()
    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", lambda **_kw: attached)
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", lambda **_kw: ensured)
    monkeypatch.setattr(sm, "DATA_DIR", tmp_path)

    monkeypatch.setattr(sm, "_stop_requested", lambda stop_event=None: True)
    sm._reconcile_delegated_runs(lambda: set())
    assert attached.started == [] and ensured.started == []
    assert _pending_ids(tmp_path) == [INVOCATION]

    monkeypatch.setattr(sm, "_stop_requested", lambda stop_event=None: False)
    sm._reconcile_delegated_runs(lambda: set())
    assert attached.started == []
    assert [key for _request, key in ensured.started] == [INVOCATION]
    assert _pending_ids(tmp_path) == []
    dc._CUSTODY.clear()


def test_the_manual_restart_shape_attaches_and_replays_nothing(tmp_path, monkeypatch):
    """``server_restart._stop_owned_work`` calls the orphan sweep with the attach-only
    factory and no admission predicate; that exact shape starts nothing."""
    from ouroboros import claudexor_daemon

    dc = _pending(tmp_path)
    _held(tmp_path, run_id="run-restart")
    attached = _Recorder()
    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", lambda **_kw: attached)
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway",
                        lambda **_kw: pytest.fail("Restart never starts a daemon"))

    outcomes = dc.reconcile_orphaned_runs(tmp_path, running_task_ids=set(),
                                          gateway_factory=claudexor_daemon.read_owned_gateway)

    assert {row["action"] for row in outcomes} == {"invocation_retained", "left_live"}
    assert attached.started == []
    assert _pending_ids(tmp_path) == [INVOCATION]
