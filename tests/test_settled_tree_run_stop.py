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


def _stop(drive, task_id=ROOT):
    """The owner's Stop through the real door, as the work row sends it (cascade, Stop now)."""
    app = Starlette(routes=[Route("/api/tasks/{task_id}/cancel", api_task_cancel, methods=["POST"])])
    app.state.drive_root = drive
    with TestClient(app) as client:
        return client.post(f"/api/tasks/{task_id}/cancel", json={"cascade": True})


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
    assert _controls(qenv.drive) == [[("stop", ROOT)]], "the run's work offers the Stop its door takes"
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
