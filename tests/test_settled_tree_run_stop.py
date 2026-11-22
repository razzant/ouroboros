"""The owner's Stop reaches a settled tree's open delegated runs, and the work's row offers it there.

A root that ended (an answer, a provider death) can leave a delegated run working: its own end was no
verdict for that run, so custody spared it. The run is a live member of the root's work and the cascade
Stop door reaches it (``task_subtree_is_live`` counts the tree's open runs): the cascade cancels it behind
the owner's verdict, attach-only, and answers 200 once custody records the run settled, else 202 while the
cancel is unconfirmed, which the intent watchdog finishes. Ouroboros's own ``cancel_task`` reaches the same
run. A tree with nothing open still answers 404 and its work offers no Stop.
"""
from __future__ import annotations

import json
import queue as std_queue
import time
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import cancel_intents as ci
from ouroboros import delegate_custody as custody
from ouroboros import delegate_custody_reconcile as reconcile
from ouroboros.gateway import work_projection as wp
from ouroboros.gateway.tasks import api_task_cancel
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.utils import utc_now_iso
from tests._cancel_intents_shared import qenv as _qenv
from tests._shared import write_queue_document

# Requested by name as a test parameter: re-bound through a module attribute (F811 otherwise).
qenv = _qenv

ROOT, HELPER = "ended-root", "spent-helper"


class _Engine:
    """A serving daemon: the run works until a cancel it obeys; nothing may start. Called, it is its own
    attach-only transport, counting how often a Stop opened it."""

    def __init__(self, *, obeys=True):
        self.state, self.obeys, self.cancels, self.opened = "running", obeys, [], 0

    def __call__(self):
        self.opened += 1
        return self

    def handshake(self, **_kw):
        return {}

    def get_run(self, rid, **_kw):
        return {"lastSeq": 2, "summary": {"state": self.state, "spendUsd": 0.0}}

    def cancel_run(self, rid, reason=""):
        self.cancels.append((rid, reason))
        if self.obeys:
            self.state = "cancelled"
        return {"accepted": True, "status": "accepted"}

    def start_run(self, *_args, **_kwargs):
        pytest.fail("a Stop starts no delegated run")

    def remove_project(self, _project_id):
        pass

    def close(self):
        pass


def _engine(monkeypatch, **kwargs) -> _Engine:
    engine = _Engine(**kwargs)
    monkeypatch.setattr(reconcile, "attach_only_gateway", engine)
    return engine


def _started(drive, **facts):
    """A run's durable start as delegation records it: its tree, route and ledger."""
    custody._CUSTODY.clear()
    custody.record_started(drive, custody.RunCustody(**{"root_task_id": ROOT, "route_id": "codex", "model": "m",
                                                        "ledger_root": str(drive), **facts}))
    custody._CUSTODY.clear()


def _ended_tree(drive, *, run_settled=False):
    """The root failed on a provider death after its helper started a run; the helper finished, the run works."""
    write_task_result(drive, ROOT, "failed", result="the provider went away", root_task_id=ROOT, chat_id=42)
    write_task_result(drive, HELPER, "completed", result="started its run", root_task_id=ROOT,
                      parent_task_id=ROOT, chat_id=42)
    _started(drive, run_id="run-1", task_id=HELPER)
    if run_settled:
        assert custody.emit(drive, custody.SETTLED, {"run_id": "run-1", "task_id": HELPER, "state": "succeeded"})


def _stop(drive, task_id=ROOT, *, cascade=True, stop_action_id=""):
    """The owner's Stop through the real door, as the work row sends it (cascade, Stop now)."""
    app = Starlette(routes=[Route("/api/tasks/{task_id}/cancel", api_task_cancel, methods=["POST"])])
    app.state.drive_root = drive
    with TestClient(app) as client:
        return client.post(f"/api/tasks/{task_id}/cancel", json={"cascade": cascade, "stop_action_id": stop_action_id})


def _noop_incidents(drive, task_id):
    """The owner-visible "the cascade cancelled nothing" incidents recorded for this Stop."""
    path = drive / "logs" / "supervisor.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    return [row for row in rows if row.get("type") == "task_cancel_cascade_noop" and row.get("task_id") == task_id]


def _controls(drive):
    return [[(entry["action"], entry["target"]) for entry in work["controls"]]
            for work in wp.works_snapshot(drive, [], {})["works"]]


def test_the_cascade_stop_cancels_a_settled_trees_open_run(qenv, monkeypatch):
    _ended_tree(qenv.drive)
    engine = _engine(monkeypatch)
    assert _controls(qenv.drive) == [[("pause", ROOT), ("stop", ROOT)]], "the run's work offers the doors that take it"
    response = _stop(qenv.drive)
    assert (response.status_code, response.json()) == (200, {"ok": True, "task_id": ROOT, "cascade": True})
    assert engine.cancels == [("run-1", "owner_task_gone")]
    assert reconcile.open_runs(qenv.drive) == [], "custody recorded the cancelled run settled"
    assert load_task_result(qenv.drive, ROOT)["status"] == "failed", "completion wins: the root's own end stays"
    assert ci.active_intent(qenv.drive, ROOT) is None, "the cascade's postcondition settled the Stop"
    assert _controls(qenv.drive) == [], "nothing of the work lives on"


def test_an_unconfirmed_cancel_keeps_the_stop_pending_until_the_run_settles(qenv, monkeypatch):
    _ended_tree(qenv.drive)
    engine = _engine(monkeypatch, obeys=False)
    response = _stop(qenv.drive)
    assert response.status_code == 202 and response.json()["cancel_state"] == "pending", response.json()
    assert [run.run_id for run in reconcile.open_runs(qenv.drive)] == ["run-1"], "a requested cancel proves nothing"
    assert ci.active_intent(qenv.drive, ROOT)["scope"] == ci.SCOPE_CASCADE
    assert _noop_incidents(qenv.drive, ROOT) == [], "a cancel in flight is pending, not a cascade that did nothing"
    engine.state = "cancelled"  # the run obeys late; the watchdog drives the recorded cascade again
    qenv.tl.sweep_cancel_intents(now=time.time() + 3600)
    assert reconcile.open_runs(qenv.drive) == [] and ci.active_intent(qenv.drive, ROOT) is None
    assert engine.cancels[0] == ("run-1", "owner_task_gone")


def test_a_stop_of_a_settled_helper_reaches_its_own_run_and_spares_its_live_root(qenv, monkeypatch):
    """The run records its root's tree, yet a cascade Stop of the helper that started it reaches it: the helper's own
    runs are its subtree's. The live root is no descendant of the helper and keeps working."""
    write_task_result(qenv.drive, ROOT, "running", root_task_id=ROOT, chat_id=42)
    qenv.q.RUNNING[ROOT] = {"task": {"id": ROOT, "chat_id": 42, "root_task_id": ROOT}}
    write_task_result(qenv.drive, HELPER, "failed", result="the provider went away", root_task_id=ROOT,
                      parent_task_id=ROOT, chat_id=42)
    _started(qenv.drive, run_id="run-1", task_id=HELPER)
    engine = _engine(monkeypatch)
    response = _stop(qenv.drive, HELPER)
    assert response.status_code == 200, response.json()
    assert engine.cancels == [("run-1", "owner_task_gone")] and reconcile.open_runs(qenv.drive) == []
    assert ROOT in qenv.q.RUNNING and load_task_result(qenv.drive, ROOT)["status"] == "running"


def test_a_settled_tree_with_nothing_open_keeps_its_404_and_offers_no_stop(qenv, monkeypatch):
    _ended_tree(qenv.drive, run_settled=True)
    engine = _engine(monkeypatch)
    assert _controls(qenv.drive) == []
    assert _stop(qenv.drive).status_code == 404
    assert ci.active_intent(qenv.drive, ROOT) is None, "completion wins: no intent over a tree with nothing live"
    assert (engine.opened, engine.cancels) == (0, [])


def test_ouroboros_own_stop_reaches_the_same_run(qenv, monkeypatch):
    from ouroboros.tools.join_ledger import _cancel_task
    from supervisor.events_runtime_controls import _drive_cancel_task_event

    _ended_tree(qenv.drive)
    engine = _engine(monkeypatch)
    # The worker reads a fresh queue snapshot with no row of the tree: custody alone shows the run.
    write_queue_document(qenv.drive / "state" / "queue_snapshot.json",
                         {"ts": utc_now_iso(), "running": [], "pending": []})
    events = []
    monkeypatch.setattr("ouroboros.tools.control._emit_control_event", lambda _ctx, evt: events.append(evt) or "live")
    caller = SimpleNamespace(task_id="mind-task", drive_root=qenv.drive, pending_events=[],
                             event_queue=std_queue.Queue(),
                             task_metadata={"root_task_id": "mind-task", "delegation_role": "root"})
    reply = _cancel_task(caller, ROOT, reason="nobody needs this run any more")
    assert reply.startswith(f"Cancel requested: {ROOT}") and "external runs stop with it" in reply, reply
    assert ci.active_intent(qenv.drive, ROOT)["scope"] == ci.SCOPE_CASCADE
    _drive_cancel_task_event(events[0], SimpleNamespace())
    assert engine.cancels == [("run-1", "owner_task_gone")] and reconcile.open_runs(qenv.drive) == []
    assert ci.active_intent(qenv.drive, ROOT) is None


@pytest.mark.parametrize("remote", ["running", "lost_cancel_reply", "cancel_not_sent", "unreadable_run", "succeeded",
                                    "renewed_stop", "renewed_stop_wrong_store", "renewed_stop_without_read"])
def test_cascade_keeps_the_accepted_start_until_its_real_outcome(qenv, monkeypatch, remote):
    import httpx

    from ouroboros.delegate_custody_current import run
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from tests._delegated_transport_shared import Engine, _sweep
    from tests._owner_pause_ended_helpers import REQUEST, _decide, _pending

    drive = qenv.drive
    monkeypatch.setattr(custody, "_CUSTODY", {})
    write_task_result(drive, ROOT, "completed", root_task_id=ROOT)
    write_task_result(drive, HELPER, "completed", root_task_id=ROOT, parent_task_id=ROOT)
    assert _decide(drive, _pending(drive, HELPER, "lost", tree=ROOT))
    lost = remote == "lost_cancel_reply" or remote.startswith("renewed_stop")

    class ControlEngine(Engine):
        unsent = False
        unreadable = False

        def __call__(self, request):
            if remote == "cancel_not_sent" and request.url.path.endswith("/control") and not self.unsent:
                self.unsent = True
                raise httpx.ConnectError("connect refused before control delivery", request=request)
            if self.unreadable and request.method == "GET" and request.url.path.startswith("/v2/runs/"):
                raise httpx.ReadTimeout("no run detail", request=request)
            return super().__call__(request)

    engine = ControlEngine(drive)
    def attach():
        gateway = engine.gateway()
        if lost:
            send = gateway.cancel_run
            def lose_reply(rid, **kwargs):
                send(rid, **kwargs)
                raise ClaudexorUnavailable("daemon_unreachable", "cancel reply lost")
            gateway.cancel_run = lose_reply
        return gateway
    monkeypatch.setattr(reconcile, "attach_only_gateway", attach)
    assert _stop(drive, cascade=False).status_code == 404
    assert ci.active_intent(drive, ROOT) is None
    reply = _stop(drive, stop_action_id="first-press")
    assert reply.status_code == 202 and reply.json()["cancel_state"] == "pending"
    intent = ci.active_intent(drive, ROOT)
    assert intent["scope"] == ci.SCOPE_CASCADE
    _sweep(drive, engine)
    qenv.tl.sweep_cancel_intents(now=time.time() + 3600)
    assert ci.active_intent(drive, ROOT)["request_id"] == intent["request_id"]
    assert engine.lookups and not engine.creates and not engine.cancels and not _noop_incidents(drive, ROOT)

    job = engine.accept("lost", REQUEST, state="succeeded" if remote == "succeeded" else "running")
    _sweep(drive, engine, live={HELPER})
    engine.unreadable = remote == "unreadable_run"
    rid = engine.commands[job]["run"]
    qenv.tl.sweep_cancel_intents(now=time.time() + 3600)
    if remote != "succeeded":
        assert ci.active_intent(drive, ROOT)["request_id"] == intent["request_id"]
        if remote == "cancel_not_sent":
            assert not engine.cancels and run(drive, rid)._recovered_cancel == "not_issued"
        qenv.tl.sweep_cancel_intents(now=time.time() + 3600)
        assert engine.cancels == [rid]
        expected = custody.CANCEL_CONTAINMENT_FAULT if lost else custody.CANCEL_REQUESTED
        assert run(drive, rid)._recovered_cancel == expected
        assert ci.active_intent(drive, ROOT)["request_id"] == intent["request_id"]
        engine.unreadable = False
        if remote.startswith("renewed_stop") or remote == "running":
            assert _stop(drive, stop_action_id="first-press").status_code == 202
            assert engine.cancels == [rid]
            original_store = engine.store
            if remote == "renewed_stop_wrong_store":
                engine.store = "not-the-starts-store"
            engine.unreadable = remote == "renewed_stop_without_read"
            assert _stop(drive, stop_action_id="second-press").status_code == 202
            assert _stop(drive, stop_action_id="first-press").status_code == 202
            qenv.tl.sweep_cancel_intents(now=time.time() + 3600)
            assert engine.cancels == [rid] * (2 if remote == "renewed_stop" else 1)
            engine.store, engine.unreadable = original_store, False
        engine.bind_run(job, "cancelled")
        qenv.tl.sweep_cancel_intents(now=time.time() + 3600)
    assert ci.active_intent(drive, ROOT) is None and not reconcile.open_runs(drive)
    assert len(engine.cancels) == int(remote != "succeeded") + int(remote == "renewed_stop") and not engine.creates
    assert load_task_result(drive, HELPER)["status"] == "completed"


@pytest.mark.parametrize("verdict", ["single", "historical_cascade"])
def test_only_an_open_cascade_reaches_late_work_of_a_finished_child(qenv, monkeypatch, verdict):
    from tests._delegated_transport_shared import Engine, _sweep
    from tests._owner_pause_ended_helpers import REQUEST, _decide, _pending

    drive = qenv.drive
    monkeypatch.setattr(custody, "_CUSTODY", {})
    write_task_result(drive, ROOT, "scheduled", root_task_id=ROOT)
    write_task_result(drive, HELPER, "completed", root_task_id=ROOT, parent_task_id=ROOT)
    assert _decide(drive, _pending(drive, HELPER, "lost", tree=ROOT))
    engine = Engine(drive)
    monkeypatch.setattr(reconcile, "attach_only_gateway", engine.gateway)
    if verdict == "single":
        qenv.q.PENDING.append({"id": ROOT, "type": "task", "root_task_id": ROOT, "chat_id": 0})
        assert _stop(drive, cascade=False).status_code == 200
    else:
        # Historical terminal provenance alone never authorizes another cascade.
        write_task_result(drive, ROOT, "cancelled", root_task_id=ROOT,
                          cancel_origin={"request_id": "previous-stop", "scope": ci.SCOPE_CASCADE})
    assert ci.active_intent(drive, ROOT) is None
    assert load_task_result(drive, ROOT)["status"] == "cancelled"
    job = engine.accept("lost", REQUEST)
    _sweep(drive, engine)
    _sweep(drive, engine)
    qenv.tl.sweep_cancel_intents(now=time.time() + 3600)
    assert [r.run_id for r in reconcile.open_runs(drive)] == [engine.commands[job]["run"]]
    assert not engine.creates and not engine.cancels


@pytest.mark.parametrize("bound_at_stop", [False, True], ids=["pending", "bound"])
def test_stop_of_an_ended_branch_reaches_only_its_descendants(qenv, monkeypatch, bound_at_stop):
    from tests._delegated_transport_shared import Engine, _sweep
    from tests._owner_pause_ended_helpers import REQUEST, _decide, _pending

    drive = qenv.drive
    monkeypatch.setattr(custody, "_CUSTODY", {})
    write_task_result(drive, ROOT, "completed", root_task_id=ROOT)
    for member, parent in [(HELPER, ROOT), ("bridge", HELPER), ("starter", "bridge"), ("sibling", ROOT)]:
        write_task_result(drive, member, "completed", result=member, root_task_id=ROOT, parent_task_id=parent)
    for starter in ("starter", "sibling"):
        assert _decide(drive, _pending(drive, starter, starter, tree=ROOT))
    engine = Engine(drive)
    monkeypatch.setattr(reconcile, "attach_only_gateway", engine.gateway)
    sibling = engine.accept("sibling", REQUEST)
    job = engine.accept("starter", REQUEST) if bound_at_stop else None
    _sweep(drive, engine, live={"starter", "sibling"})
    assert _stop(drive, HELPER, cascade=False).status_code == 404
    response = _stop(drive, HELPER)
    assert response.status_code == 202 and response.json()["cancel_state"] == "pending"
    intent = ci.active_intent(drive, HELPER)["request_id"]

    def sweep_stop():
        qenv.tl.sweep_cancel_intents(now=time.time() + 3600)

    if job is None:
        sweep_stop()
        assert not engine.cancels and ci.active_intent(drive, HELPER)["request_id"] == intent
        job = engine.accept("starter", REQUEST)
        _sweep(drive, engine, live={"starter", "sibling"})
    rid = engine.commands[job]["run"]
    assert [r.run_id for r in reconcile.tree_open_runs(drive, HELPER)] == [rid]
    sweep_stop()
    sweep_stop()
    assert engine.cancels == [rid] and ci.active_intent(drive, HELPER)["request_id"] == intent
    engine.bind_run(job, "cancelled")
    sweep_stop()
    assert ci.active_intent(drive, HELPER) is None and _stop(drive, HELPER).status_code == 404
    assert [r.run_id for r in reconcile.open_runs(drive)] == [engine.commands[sibling]["run"]]
    assert engine.cancels == [rid] and not engine.creates
    assert load_task_result(drive, "starter")["result"] == "starter"


def test_a_crashed_cancel_handoff_lets_one_distinct_later_stop_reserve_again(qenv, monkeypatch):
    """A CANCEL_HANDOFF whose sender never recorded CANCEL_OUTCOME keeps its identity and its sender as
    an unresolved fault. While that sender's image may still run, every press stays read-only. Once it
    is positively gone and the owning store still shows the run active, ONE distinct, later owner Stop
    reserves a new handoff and reaches the engine; the same press and automatic reconciliation never do.
    Before the fix the orphaned handoff folded to a dead "unknown" and every later Stop sent nothing."""
    from ouroboros import process_custody
    from ouroboros.delegate_custody_current import run
    from tests._delegated_transport_shared import Engine, _rows, _sweep
    from tests._owner_pause_ended_helpers import REQUEST, _decide, _pending

    drive = qenv.drive
    monkeypatch.setattr(custody, "_CUSTODY", {})
    write_task_result(drive, ROOT, "completed", root_task_id=ROOT)
    write_task_result(drive, HELPER, "completed", root_task_id=ROOT, parent_task_id=ROOT)
    assert _decide(drive, _pending(drive, HELPER, "lost", tree=ROOT))
    engine = Engine(drive)
    job = engine.accept("lost", REQUEST)
    _sweep(drive, engine, live={HELPER})
    rid = engine.commands[job]["run"]
    assert run(drive, rid).resume_start_state == "running"

    class SenderExited(BaseException):
        pass

    def press(gateway, control_id, second=0):
        stop = control_id and {"control_id": control_id, "requested_at": f"2999-01-01T00:00:0{second}+00:00"}
        return custody.cancel_and_verify(drive, gateway, run(drive, rid), "owner_task_gone", owner_stop=stop or None)

    with engine.gateway() as first:
        def interrupted(*_args, **_kwargs):
            raise SenderExited()
        first.cancel_run = interrupted  # the sender dies at the physical cancel boundary
        with pytest.raises(SenderExited):
            custody.cancel_and_verify(drive, first, run(drive, rid), "owner_task_gone",
                                      owner_stop={"control_id": "first-press", "requested_at": utc_now_iso()})
    assert len(_rows(drive, custody.CANCEL_HANDOFF, run_id=rid)) == 1
    assert _rows(drive, custody.CANCEL_OUTCOME, run_id=rid) == []
    custody._CUSTODY.clear()  # the durable chain is the authority from here on

    with engine.gateway() as second:
        early = press(second, "impatient-press", 1)
        assert engine.cancels == [] and early["outcome"] == custody.CANCEL_CONTAINMENT_FAULT, \
            "the first sender's image may still deliver: a distinct press beside it only reads back"
        # The restart: this pid now runs a later image, so the recorded sender is positively gone.
        monkeypatch.setattr(process_custody, "current_custody_session_id", lambda: "image-after-restart")
        assert press(second, "first-press", 2)["outcome"] == custody.CANCEL_CONTAINMENT_FAULT
        assert press(second, "")["outcome"] == custody.CANCEL_CONTAINMENT_FAULT
        assert engine.cancels == [], "the same press and automatic reconciliation stay read-only"
        renewed = press(second, "second-press", 3)
    assert engine.cancels == [rid], "one distinct, later owner Stop reserves and cancels again"
    assert renewed["outcome"] in (custody.CANCEL_REQUESTED, custody.CANCEL_CONFIRMED), renewed
    custody._CUSTODY.clear()
    with engine.gateway() as third:
        once_more = press(third, "third-press", 4)
    assert engine.cancels == [rid], "the reserved handoff's outcome ends further presses"
    assert once_more["outcome"] == custody.CANCEL_REQUESTED
