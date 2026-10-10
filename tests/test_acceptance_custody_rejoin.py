"""Incident form 05.10 on task acceptance (#1547): a panel of three, two API seats
already PASS, one delegated seat still running when the process restarts.

Acceptance keeps its paid operation on the recorded-run collection seam: a cold
collect reads the proven run attach-only, defers while it is live and never
declares the panel clean before that seat answers; the seat's late FAIL is the
panel's verdict when it lands, not a lost finding.
"""

from __future__ import annotations

import dataclasses
import json
import queue
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ouroboros import delegate_custody as custody
from ouroboros.loop_acceptance_review import acceptance_run_pending
from ouroboros.review_dispatch import collect_task_acceptance_run
from ouroboros.review_execution import ReviewRouteKind
from ouroboros.review_substrate import ReviewRequest, ReviewSlot
from ouroboros.review_verdict import task_acceptance_is_clean

TASK = "acceptance-1547"
PASS = json.dumps({"verdict": "PASS", "findings": [], "summary": "Independent review passes"})
LATE_FAIL = json.dumps({"verdict": "FAIL", "summary": "The late seat found a blocker",
                        "findings": [{"severity": "blocking", "summary": "claim_1 is unsupported"}]})


def _dead_pid() -> int:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


@pytest.fixture
def forbid_effects(monkeypatch):
    def forbidden(name):
        def fail(*_args, **_kwargs):
            pytest.fail(f"pure collection reached {name}")
        return fail

    monkeypatch.setattr("ouroboros.review_substrate.run_review_request", forbidden("the ordinary runner"))
    monkeypatch.setattr("ouroboros.llm.LLMClient.chat", forbidden("a model call"))
    monkeypatch.setattr("ouroboros.claudexor_daemon.ensure_owned_gateway", forbidden("daemon ensure/start"))


class _ReadOnlyGateway:
    def __init__(self, detail, calls):
        self.detail, self.calls = detail, calls

    def get_run(self, run_id, *, timeout_sec=None):
        self.calls.append(("get_run", run_id))
        return self.detail

    def get_run_artifact(self, run_id, path):
        return b""

    def close(self):
        pass

    def __getattr__(self, name):
        pytest.fail(f"pure collection called gateway.{name}")


def _ctx(root):
    return SimpleNamespace(task_id=TASK, task_attempt=1, drive_root=root, budget_drive_root=root,
                           task_metadata={}, pending_events=[], event_queue=queue.Queue())


def _recorded_panel():
    """The run as the first process left it: two API seats answered PASS, the
    delegated seat 'c' still dispatched, its controller (that process) gone."""
    request = ReviewRequest(surface="task_acceptance", task_id=TASK, goal="goal", subject="reviewed answer",
                            evidence={"requirement": "exact"}, retry_key="acceptance-wave-1547")
    slots = [ReviewSlot(slot_id="a", model="m/a"), ReviewSlot(slot_id="b", model="m/b"),
             ReviewSlot(slot_id="c", model="cursor", route=ReviewRouteKind.AGENT_SESSION, session_target="cursor")]
    answered = [{"slot_id": sid, "model": f"m/{sid}", "status": "ok", "raw_text": PASS,
                 "operation_id": f"op-{sid}", "operation_state": "settled"} for sid in ("a", "b")]
    live = {"slot_id": "c", "model": "cursor", "status": "error", "operation_id": "op-c",
            "operation_state": "pending_dispatch", "late_result_pending": True,
            "usage": {"review_controller": {"pid": _dead_pid(), "session": "gone"}}}
    return json.loads(json.dumps({
        "authority": "host_root", "request": dataclasses.asdict(request),
        "slot_roster": [dataclasses.asdict(slot) for slot in slots],
        "panel_id": "panel-1547", "aggregate_signal": "DEGRADED", "actors": answered + [live]}))


def _seat_c_started_durably(root):
    custody.emit(root, custody.START_REQUESTED, {
        "invocation_id": "inv-op-c", "task_id": TASK, "operation_id": "op-c", "slot_id": "c",
        "surface": "task_acceptance", "root_task_id": TASK, "max_seconds": 540,
        "request": {"prompt": "review", "harnesses": ["cursor"]}})
    custody.emit(root, custody.STARTED, {"run_id": "run-c", "invocation_id": "inv-op-c",
                                         "task_id": TASK, "surface": "task_acceptance"})


def test_a_restarted_acceptance_panel_waits_for_its_live_seat_and_keeps_its_late_fail(
        tmp_path, forbid_effects, monkeypatch):
    calls = []
    detail = {"summary": {"state": "running"}}
    monkeypatch.setattr("ouroboros.claudexor_daemon.read_owned_gateway", lambda: _ReadOnlyGateway(detail, calls))
    _seat_c_started_durably(tmp_path)
    panel = _recorded_panel()

    # Restart: the seat's worker is gone, its run is still live (≈9 min of its window left).
    held = collect_task_acceptance_run(panel, drive_root=tmp_path, usage_ctx=_ctx(tmp_path))
    assert held.collection["facts"] == {"a": "settled", "b": "settled", "c": "deferred"}
    assert acceptance_run_pending(held), "two PASS seats do not end a panel whose third seat is still running"
    assert not task_acceptance_is_clean(held), "no GREEN before the live seat answers"
    assert ("get_run", "run-c") in calls and all(call[0] == "get_run" for call in calls), \
        "the live run is read, never cancelled, re-posted or replaced"

    # A later collect of the same recorded run never extends or restarts anything.
    again = collect_task_acceptance_run(panel, drive_root=tmp_path, usage_ctx=_ctx(tmp_path))
    assert again.collection["facts"]["c"] == "deferred" and acceptance_run_pending(again)

    # The seat answers late with a blocking FAIL: accepted attach-only, it is the panel's verdict.
    detail.clear()
    detail.update({"summary": {"state": "succeeded", "outputConformance": ""},
                   "primaryOutput": {"text": LATE_FAIL, "truncated": False}})
    settled = collect_task_acceptance_run(panel, drive_root=tmp_path, usage_ctx=_ctx(tmp_path))
    seat = settled.actors[2]
    assert settled.collection["facts"]["c"] == "collected" and not acceptance_run_pending(settled)
    assert seat["operation_state"] == "late_settled" and seat["operation_id"] == "op-c"
    assert seat["usage"]["collection"] == "attach_only_observation" and seat["usage"]["delegated_run_id"] == "run-c"
    assert seat["parsed"]["verdict"] == "FAIL" and seat["raw_text"] == LATE_FAIL
    assert settled.aggregate_signal == "FAIL" and not task_acceptance_is_clean(settled)
    assert any("claim_1" in str(finding) for finding in settled.parsed_findings), "the late blocking finding is kept"


@pytest.mark.parametrize("state", ["cancelled", "failed"])
def test_a_seat_whose_run_ended_without_success_is_settled_as_its_own_error_never_green(
        tmp_path, forbid_effects, monkeypatch, state):
    detail = {"summary": {"state": state}}
    monkeypatch.setattr("ouroboros.claudexor_daemon.read_owned_gateway", lambda: _ReadOnlyGateway(detail, []))
    _seat_c_started_durably(tmp_path)
    settled = collect_task_acceptance_run(_recorded_panel(), drive_root=tmp_path, usage_ctx=_ctx(tmp_path))
    seat = settled.actors[2]
    assert seat["operation_state"] == "late_settled" and seat["status"] == "error" and state in seat["error"]
    assert not acceptance_run_pending(settled)
    assert not task_acceptance_is_clean(settled), "a seat that never answered is not a clean pass"
