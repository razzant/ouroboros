"""Shared real-Pause fixtures for ended-member and accepted-start recovery tests.

The engine is a stand-in; the queue, custody, Pause settlement and Resume consumers
remain real. These fixtures hold no test cases and import no collected test module.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import (
    _fast_hold, _install_queue, _loop_ctx, _mock_pause_observation,
)


def _owner_park(tmp_path, monkeypatch, task_id, runs, root="root-1"):
    from ouroboros import budget_pause

    ctx, limit_ctx = _loop_ctx(tmp_path, task_id)
    ctx.root_task_id = root
    _fast_hold(monkeypatch, budget_pause)
    _mock_pause_observation(monkeypatch, budget_pause, runs)
    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
        budget_pause.enter_owner_pause(limit_ctx)
    budget_pause.end_dispatch_fence(task_id)
    return raised.value.pause


class _Engine:
    """The daemon as a Pause's stop and its read-back see it. Nothing under a Pause may start work."""

    def __init__(self, state="running"):
        self.state, self.states, self.up, self.opened = state, {}, True, 0
        self.cancels, self.reads, self.starts, self.cancel_error, self.read_error = [], [], [], None, None
        self.may_start = False

    def __call__(self, *_a, **_k):
        self.opened += 1
        return self

    def handshake(self, **_kw):
        if not self.up:
            raise RuntimeError("gateway down")
        return {"compatible": True}

    def cancel_run(self, run_id, reason=""):
        self.cancels.append(run_id)
        if self.cancel_error:
            raise self.cancel_error
        return {"accepted": True, "status": "accepted"}

    def get_run(self, run_id, **_kw):
        self.reads.append(run_id)
        if self.read_error:
            raise self.read_error
        return {"lastSeq": 3, "summary": {"state": self.states.get(run_id, self.state), "spendUsd": 0.0}}

    def start_run(self, _request, *, idempotency_key=""):
        self.starts.append(idempotency_key)
        assert self.may_start, "a Pause starts no delegated run"
        return {"runId": "run-bound"}

    def close(self):
        return None


def _run(root, run_id="run-leaf", task_id="child-dead", **facts):
    from ouroboros import delegate_custody as dc

    assert dc.record_started(root, dc.RunCustody(run_id=run_id, task_id=task_id, root_task_id="root-1", route_id="codex",
                                                 ledger_root=str(root), **facts))


def _tail_tree(tmp_path, monkeypatch, root_status="running", owner_status="failed", tail=_run, chat_id=0):
    """A root, a member that ended and the run it left working: the incident's shape (a nanny's own death).
    ``chat_id``: the room the tree lives in (Main is 1, where a browser test reads its room work list)."""
    from ouroboros import delegate_custody as dc
    from ouroboros.gateways import claudexor as gw
    from ouroboros.task_results import write_task_result

    dc._CUSTODY.clear()
    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    write_task_result(tmp_path, "root-1", root_status, chat_id=chat_id, root_task_id="root-1", delegation_role="root")
    write_task_result(tmp_path, "child-dead", owner_status, chat_id=chat_id, root_task_id="root-1",
                      parent_task_id="root-1", delegation_role="subagent")
    if root_status == "running":
        workers.RUNNING["root-1"] = {"task": {"id": "root-1", "type": "task", "chat_id": chat_id,
                                              "root_task_id": "root-1"}, "worker_id": 0, "attempt": 1}
    tail(tmp_path)
    engine = _Engine()
    monkeypatch.setattr(gw, "ClaudexorGateway", engine)
    return queue, workers, engine


def _park_root(tmp_path, monkeypatch, queue, workers):
    from ouroboros import budget_pause
    from supervisor.events_budget import install_exact_budget_pause

    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING=workers.RUNNING, PENDING=workers.PENDING, WORKERS=workers.WORKERS,
                          sort_pending=lambda: None, persist_queue_snapshot=queue.persist_queue_snapshot, bridge=None)
    row = _owner_park(tmp_path, monkeypatch, "root-1", [])
    install_exact_budget_pause(ctx, "root-1", budget_pause.exact_pause_marker(row)["checkpoint"])


def _tick(queue):
    from supervisor.owner_pause_control import request_settlement_look, settle_requested_owner_pauses

    request_settlement_look()
    return settle_requested_owner_pauses(queue)


def _fence(root):
    from ouroboros.owner_pause import read_fence

    return read_fence(root, "root-1")


def _kept(root, member="child-dead"):
    return _fence(root)["ended_member_runs"][member]


def _snapshot(root):
    """The census rows and the works projection ``/api/state`` builds from them, through their real producers."""
    from ouroboros.gateway.state import _chat_activities_snapshot_safe
    from ouroboros.gateway.work_projection import works_snapshot

    raw, owners = {}, []
    census = _chat_activities_snapshot_safe(root, work_facts=raw, decision_owners=owners)
    return census, works_snapshot(root, owners, raw)


def _works(root):
    return _snapshot(root)[1]["works"]


def _live_tree(tmp_path, workers):
    """A RUNNING root, its RUNNING child and a never-started PENDING child."""
    from ouroboros.task_results import STATUS_RUNNING, STATUS_SCHEDULED, write_task_result

    write_task_result(tmp_path, "root-1", STATUS_RUNNING, chat_id=0)
    write_task_result(tmp_path, "child-run", STATUS_RUNNING, chat_id=0)
    write_task_result(tmp_path, "child-new", STATUS_SCHEDULED, chat_id=0)
    workers.RUNNING["root-1"] = {"task": {"id": "root-1", "type": "task", "chat_id": 0,
                                          "root_task_id": "root-1"}, "worker_id": 0, "attempt": 1}
    workers.RUNNING["child-run"] = {"task": {
        "id": "child-run", "type": "task", "chat_id": 0, "root_task_id": "root-1",
        "parent_task_id": "root-1", "delegation_role": "subagent"}, "worker_id": 1, "attempt": 1}
    workers.PENDING.append({"id": "child-new", "type": "task", "chat_id": 0, "_attempt": 1,
                            "root_task_id": "root-1", "parent_task_id": "root-1",
                            "delegation_role": "subagent"})


REASON = "I will work from the saved files while the original job may still be spending."
REQUEST = {"prompt": "Summarize the layout.", "access": "readonly", "mode": "ask",
           "primaryHarness": "some-route", "maxSeconds": 120}


def _pending(root, member, iid, tree="root-1", **fields):
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_resume import address

    assert custody.record_start_requested(
        root, task_id=member, root_task_id=tree, invocation_id=iid, idempotency_key=iid,
        request=REQUEST, project_id="", project_owned=False, route="some-route",
        daemon_store_id="store-a", store_identity_unproven=False, **fields)
    return address({"task_id": member, "root_task_id": tree, "invocation_id": iid})


def _paused(root, monkeypatch, tail):
    from supervisor.owner_pause_control import request_owner_pause

    queue, workers, engine = _tail_tree(root, monkeypatch, tail=tail)
    assert request_owner_pause("root-1", request_id="press")["ok"]
    _tick(queue)
    with monkeypatch.context() as park:
        _park_root(root, park, queue, workers)
    return queue, workers, engine


def _resume(door, ids=(), *, selected="root-1", reason=REASON):
    if door == "owner":
        from ouroboros.gateway.router import collect_routes
        from supervisor import queue
        from starlette.applications import Starlette
        from starlette.testclient import TestClient

        body = {"accept_unknown": list(ids), "reason": reason} if ids else {}
        app = Starlette(routes=collect_routes(data_dir=queue.DRIVE_ROOT))
        with TestClient(app) as client:
            response = client.post(f"/api/tasks/{selected}/resume", json=body)
            reply = response.json()
            if reply.get("acceptable_unknown"):
                assert response.status_code == 409
            return reply
    from supervisor.events_mind_controls import _apply
    return _apply({"task_id": selected, "action": "resume_task", "actor_task_id": "main",
                   "action_nonce": "resume", "reason": reason, "accept_unknown": list(ids)})


def _decide(root, start):
    from ouroboros.delegate_resume import _record
    return _record(root, start, start["task_id"], {"actor": "owner", "door": "api_task_resume"}, REASON)
