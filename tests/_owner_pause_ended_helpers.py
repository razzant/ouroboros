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
