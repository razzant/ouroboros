"""Wave 10 — S40: the owner's Pause, Resume and Stop hold one live task on its own id (#1729 / #1730).

A keyless real server runs from a byte-faithful checkout of the tree under test (S33's
``s33_candidate``: working-tree edits included, its process tree held until proven gone; a
HEAD clone only for a clean tree) with ONE real worker process. The owner acts only through the
HTTP doors the UI posts to (Pause, Resume, Stop); the queue ``put``, assignment, the exact pause,
its park and Resume, the launch gates and cancellation custody are the shipped ones.

Doubled, and only these: the model is the loopback stub, and the round the owner catches is held
by an event-gated ``ModelGate`` (never a timed race).

No retry is involved: since #1376 silence never ends an attempt, and the reaper ends only a hard
bound, without a successor, so the task keeps its one id from submission to Stop.
``after_owner_resume``: the owner Pauses, the attempt parks on its exact pause, the owner Resumes,
and the SAME id runs again, consumes its grant and continues its saved work (#1729).
``ordinary``: without that round trip. Both then Pause the RUNNING task while its model round is
held. A write before the Pause lands. The Pause names the logical root and fences it. A write the
model asks for after the acceptance never lands. One Resume reopens the tree, and Stop ends it
Cancelled with no late effect (#1730).

Not covered here, by design:
- crash recovery: the same-id crash rail is ``tests/test_worker_crash_retry.py`` and
  ``tests/test_working_recovery_relay.py``;
- the silence rail's negative control (a silent live attempt is attention, never a kill or a
  successor): ``tests/test_silence_attention.py::test_heartbeat_only_never_finalizes_or_reaps``
  and ``tests/test_acceptance_fence_attempts.py::test_live_silence_keeps_the_exact_active_fence_and_queues_no_reap``.
"""

from __future__ import annotations

import json
import time
import uuid

import pytest

from devtools.benchmarks.common.server_runner import _api_status
from tests.system_e2e.harness import (
    LANE_MOCK,
    ArtifactOracle,
    ModelGate,
    ScriptedStubModel,
    keyless_settings,
    require_lane,
    start_server,
    submit_running,
    wait_durable_result,
    wait_until,
)
from tests.system_e2e.test_presence_resilience import s33_candidate  # noqa: F401 - pytest fixture

KEEPALIVE = {"tool": "list_files", "arguments": {"path": "."}}
CONTROL = {"tool": "write_file", "arguments": {"root": "task_drive", "path": "s40-control.txt",
                                               "content": "written before the Pause"}}
SENTINEL = {"tool": "write_file", "arguments": {"root": "task_drive", "path": "s40-sentinel.txt",
                                                "content": "a task effect after the accepted Pause"}}
SETTINGS = {"OUROBOROS_MAX_WORKERS": 1}


class _PauseStub(ScriptedStubModel):
    """Keepalive rounds; the scenario queues one-shot steps and records when each was served."""

    def __init__(self) -> None:
        super().__init__(latency_sec=0.2)
        self.steps, self.served = [], []

    def _next_step(self, _body):
        if self.steps:
            step = self.steps.pop(0)
            self.served.append((time.time(), step["arguments"]["path"]))
            return step
        return dict(KEEPALIVE)


def _post(server, path: str, body: dict | None = None) -> dict:
    return _api_status(server.base_url, "POST", path, body or {}, timeout=60)


def _pause_row(data, task_id: str) -> dict:
    from ouroboros.budget_pause import budget_pause_row

    try:
        return budget_pause_row(data, task_id) or {}
    except Exception as exc:  # an unreadable row is a fact for the forensics, not a crash of the reader
        return {"unreadable": repr(exc)}


def _rows(oracle) -> list:
    snapshot = oracle.queue_snapshot()
    return [dict(row.get("task") or {}, id=row.get("id"), queue=state)
            for state in ("pending", "running") for row in snapshot.get(state) or [] if isinstance(row, dict)]


def _foreign_rows(oracle, origin: str) -> list:
    """Queue rows of any other id: a successor, a copy, a retry. The task owns exactly one id."""
    return [row for row in _rows(oracle) if row["id"] != origin]


def _found(data, name: str) -> list:
    return sorted(str(path) for path in data.rglob(name))


def _first_agent_round_since(stub, start: int) -> dict:
    with stub._lock:
        return next((body for _kind, body in stub.calls[start:] if body.get("tools")), {})


def _facts(oracle, data, origin: str, stub, **extra) -> dict:
    from ouroboros.owner_pause import read_fence

    keys = ("id", "queue", "_attempt", "root_task_id", "_budget_pause", "_budget_pause_resume",
            "_budget_pause_hold", "timeout_retry_from")
    return {
        "rows": [{key: row.get(key) for key in keys if row.get(key) is not None} for row in _rows(oracle)],
        "pause_row": {key: _pause_row(data, origin).get(key) for key in (
            "state", "reason", "pause_id", "task_attempt", "unreadable")},
        "fence": read_fence(data, origin),
        "result": {key: oracle.task_result(origin).get(key) for key in ("status", "reason_code")},
        "stub_calls": len(stub.calls), "stub_served": stub.served, **extra,
    }


@pytest.mark.integration
@pytest.mark.serial
@pytest.mark.parametrize("history", ["after_owner_resume", "ordinary"])
def test_s40_the_owner_pause_resume_and_stop_hold_one_task_on_its_own_id(s33_candidate, tmp_path_factory,  # noqa: F811
                                                                           history):
    require_lane(LANE_MOCK)
    from ouroboros.budget_pause import STATE_PAUSED, STATE_RESUMED
    from ouroboros.owner_pause import read_fence

    root = tmp_path_factory.mktemp(f"s40_{history}")
    gate = None
    with _PauseStub() as stub:
        server = start_server(s33_candidate, root, keyless_settings(stub, **SETTINGS))
        data, oracle = server.data_root, ArtifactOracle(server.data_root)
        facts = {"history": history, "tree": getattr(s33_candidate, "identity", str(s33_candidate))}
        try:
            origin = submit_running(server, "Keep listing the repository files. [S40]")
            facts["origin"] = origin
            assert wait_until(lambda: len(stub.calls) >= 3, 60), stub.kinds()
            if history == "after_owner_resume":
                paused = _post(server, f"/api/tasks/{origin}/pause", {"request_id": f"s40-{uuid.uuid4().hex}"})
                assert paused["status"] in (200, 202), paused
                # Settled: the exact row, the durable fence and the queue all say paused.
                assert wait_until(lambda: _pause_row(data, origin).get("state") == STATE_PAUSED
                                  and read_fence(data, origin).get("state") == "paused"
                                  and origin not in oracle.running_ids(), 120), _facts(oracle, data, origin, stub)
                at_pause = len(stub.calls)
                granted = _post(server, f"/api/tasks/{origin}/resume")
                assert granted["status"] == 200 and granted["body"].get("ok"), granted
                # #1729: the SAME id runs again, its loop consumes the grant, and it continues the saved work.
                assert wait_until(lambda: origin in oracle.running_ids() and _first_agent_round_since(stub, at_pause),
                                  120), _facts(oracle, data, origin, stub)
                assert wait_until(lambda: _pause_row(data, origin).get("state") == STATE_RESUMED, 60), \
                    _facts(oracle, data, origin, stub)
                body = _first_agent_round_since(stub, at_pause)
                earlier = {f"call_{seq}" for seq in range(1, at_pause + 1)}
                history_ids = {call.get("id") for message in body.get("messages") or [] if isinstance(message, dict)
                               for call in message.get("tool_calls") or []} | {
                    message.get("tool_call_id") for message in body.get("messages") or [] if isinstance(message, dict)}
                assert history_ids & earlier, "the Resume restarted instead of continuing the saved work"
                assert not _foreign_rows(oracle, origin), _facts(oracle, data, origin, stub)

            # #1730: a write before the Pause lands (the door is real), then the round is held.
            stub.steps.append(dict(CONTROL))
            facts["control"] = wait_until(lambda: _found(data, "s40-control.txt"), 60)
            assert facts["control"], _facts(oracle, data, origin, stub)
            gate = stub.gate = ModelGate(lambda body: bool(body.get("tools")), timeout=120)
            assert gate.arrived.wait(60), _facts(oracle, data, origin, stub)
            accepted = _post(server, f"/api/tasks/{origin}/pause", {"request_id": f"s40-{uuid.uuid4().hex}"})
            assert accepted["status"] in (200, 202), accepted
            facts["pause"] = accepted["body"]
            stub.steps.append(dict(SENTINEL))  # the model asks for a task effect after the acceptance
            gate.release.set()
            # Resume observes settled authority: requested -> paused during its census must refuse.
            parked = wait_until(lambda: (_pause_row(data, origin).get("state") == STATE_PAUSED
                                        and read_fence(data, origin).get("state") == "paused")
                                or _found(data, "s40-sentinel.txt"), 90)
            facts.update(after_pause=_facts(oracle, data, origin, stub))
            assert any(path == "s40-sentinel.txt" for _ts, path in stub.served), "the write was never asked for"
            assert not _found(data, "s40-sentinel.txt"), facts
            assert accepted["body"].get("root_task_id") == origin, accepted
            assert parked and origin not in oracle.running_ids(), facts
            assert read_fence(data, origin).get("state") == "paused" and not _foreign_rows(oracle, origin), facts

            # One Resume reopens the logical tree on the same id; Stop ends it.
            resumed = _post(server, f"/api/tasks/{origin}/resume")
            assert resumed["status"] == 200 and resumed["body"].get("root_task_id") == origin, resumed
            assert wait_until(lambda: read_fence(data, origin).get("state") == "released", 60), \
                _facts(oracle, data, origin, stub)
            assert wait_until(lambda: origin in oracle.running_ids(), 120), _facts(oracle, data, origin, stub)
            assert not _foreign_rows(oracle, origin), _facts(oracle, data, origin, stub)
            stopped = server.cancel_task(origin)
            assert stopped["status"] in (200, 202), stopped
            facts["terminal"] = wait_durable_result(oracle, origin, timeout=180).get("status")
            assert facts["terminal"] == "cancelled", facts
            assert not _found(data, "s40-sentinel.txt"), "no task effect of the paused attempt appeared later"
            assert not _foreign_rows(oracle, origin), _facts(oracle, data, origin, stub)
        finally:
            if gate is not None:
                gate.release.set()
            print("S40_FACTS", json.dumps(facts, default=str))
            server.stop()
