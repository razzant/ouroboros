"""#1721: a held Continue and its refused Resume carry the blocker facts the census observed.

Real custody rows (``delegate_custody``), the real observer and writer census, the real Continue
admission, the real public Continue/Resume HTTP handlers and the held-row selection run here. The
census already knows whether a writer is running, a start's outcome is unknown or custody is
unreadable: those facts reach the hold, the HTTP body and (web/tests) the toasts and Activity.
"Wait for it or stop it" is advice only when every blocker is an observed running writer; nothing is
invented for an unknown outcome, alone or beside running work, and the S4 rule (independent reviews
do not hold a Continue) is unchanged.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from ouroboros import delegate_custody as dc
from tests._budget_pause_exact_helpers import _install_queue
from tests.test_continue_independent_review import PRED, _current_pool, _pending, _rows  # noqa: F401
from tests.test_owner_continue import NONCE, _interrupted

pytestmark = pytest.mark.serial


def _http(handler, task_id, body=None):
    from types import SimpleNamespace

    from starlette.requests import Request

    from supervisor import queue

    payload = json.dumps(body or {}).encode()

    async def receive():
        return {"type": "http.request", "body": payload, "more_body": False}

    request = Request({"type": "http", "method": "POST", "path": f"/api/tasks/{task_id}",
                       "path_params": {"task_id": task_id}, "headers": [(b"content-type", b"application/json")],
                       "app": SimpleNamespace(state=SimpleNamespace(drive_root=queue.DRIVE_ROOT))},
                      receive)
    response = asyncio.run(handler(request))
    return response.status_code, json.loads(response.body)


def _held(tmp_path, monkeypatch, blocker):
    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, PRED, origin={"source": "owner_restart", "reason": "owner_restart", "scope": "single"})
    _rows(tmp_path, leaf=blocker != "unbound_unknown_start", review=False)
    if blocker != "running_delegated_leaf":
        _pending(tmp_path, "inv-unbound", source=None)  # a start whose outcome is unknown: no run id
    dc._CUSTODY.clear()
    return queue, workers


UNBOUND = {"kind": "delegated_run", "run_id": "", "invocation_id": "inv-unbound", "state": "stop_unknown",
           "stop_outcome": "pending_invocation_unbound", "source": "delegated_subagent"}
RUNNING = {"kind": "delegated_run", "run_id": "run-leaf", "invocation_id": "", "state": "running",
           "stop_outcome": "", "source": "delegated_subagent"}


@pytest.mark.parametrize("blocker,facts", [
    ("unbound_unknown_start", [UNBOUND]), ("running_delegated_leaf", [RUNNING]),
    ("running_and_unknown_start", [RUNNING, UNBOUND]),
])
def test_the_held_continue_and_its_resume_refusal_carry_the_observed_facts(tmp_path, monkeypatch, blocker, facts):
    from ouroboros.work_stop import hold_fact as budget_hold_fact
    from ouroboros.gateway.task_continue import api_task_continue
    from ouroboros.gateway.tasks import api_task_resume

    queue, workers = _held(tmp_path, monkeypatch, blocker)
    status, ack = _http(api_task_continue, PRED, {"action_nonce": NONCE})
    assert status == 200 and ack["ok"] and ack["held"] is True
    successor_id = ack["successor_task_id"]
    expected = sorted(({**fact, "task_id": PRED, "review_owned": False} for fact in facts), key=json.dumps)
    assert sorted(ack["blockers"], key=json.dumps) == expected, "the Continue answer names what holds it"
    status, replay = _http(api_task_continue, PRED, {"action_nonce": NONCE})
    assert status == 200 and replay["held"] and replay["successor_task_id"] == successor_id
    assert sorted(replay["blockers"], key=json.dumps) == expected
    hold = budget_hold_fact(next(row for row in workers.PENDING if row["id"] == successor_id))
    assert hold["reason"] == "continuation_writer_unsettled" and sorted(hold["blockers"], key=json.dumps) == expected
    status, refusal = _http(api_task_resume, successor_id)
    # An existing held successor is a conflict, never Not Found, and the body keeps the facts.
    assert status == 409 and refusal["error"] == "predecessor_writers_unsettled"
    assert sorted(refusal["blockers"], key=json.dumps) == expected
    if blocker == "running_delegated_leaf":
        assert refusal["action"] == "wait_or_stop_the_previous_work"
    else:
        # Waiting for or stopping the running leaf would not settle a start whose outcome is unknown.
        assert "action" not in refusal, "no wait/stop advice while any outcome is unknown"
    assert next(row for row in workers.PENDING if row["id"] == successor_id), "the successor stays held"


def test_only_observed_running_writers_earn_the_wait_or_stop_advice():
    from supervisor.continuation_admission import only_running_writers

    assert only_running_writers([{"kind": "running_member", "task_id": "x"}])
    assert only_running_writers([{"kind": "delegated_run", "state": "running"}, {"kind": "dispatchable_member"}])
    assert only_running_writers([{"kind": "late_phase_running"}]) and only_running_writers([{"kind": "owned_process"}])
    for unknown in ({"kind": "delegated_run", "state": "stop_unknown", "stop_outcome": "pending_invocation_unbound"},
                    {"kind": "delegated_run", "state": "stop_requested"}, {"kind": "model_handoff"},
                    {"kind": "custody_unreadable"}, {"kind": "tree_census_unreadable"},
                    {"kind": "late_phase_unknown"}, {"kind": "tool_handoff"}, "not-a-row"):
        assert not only_running_writers([unknown]), unknown
        # Beside a running writer it still holds: waiting or stopping would not settle it.
        assert not only_running_writers([{"kind": "running_member", "task_id": "x"}, unknown]), unknown
    assert not only_running_writers([]) and not only_running_writers(None)


def test_independent_reviews_still_never_hold_the_continue(tmp_path, monkeypatch):
    """S4 (#1572) is unchanged: the facts ride the blockers that DO hold, nothing more."""
    from supervisor.continuation_admission import CONTINUE_PURPOSE, admit_continuation, conflicting_writers

    queue, _state, _workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, PRED, origin={"source": "owner_restart", "reason": "owner_restart", "scope": "single"})
    _rows(tmp_path, leaf=False, review=True)
    dc._CUSTODY.clear()
    ack = admit_continuation(PRED, action_nonce=NONCE)
    assert ack["ok"] and ack["held"] is False and "blockers" not in ack
    census = conflicting_writers(queue, PRED)
    assert [b["run_id"] for b in census if b["kind"] == "delegated_run"] == ["run-review"]
    assert census[0]["source"] and census[0]["state"] == "running", "the full census keeps the review, with facts"
    assert conflicting_writers(queue, PRED, purpose=CONTINUE_PURPOSE) == []


def test_a_long_census_reaches_the_refusal_whole_and_decides_its_advice():
    from supervisor.continuation_admission import writers_unsettled_refusal

    running = [{"kind": "running_member", "task_id": f"m{i}"} for i in range(25)]
    refusal = writers_unsettled_refusal(running)
    assert refusal["blockers"] == running and refusal["action"] == "wait_or_stop_the_previous_work"
    # An unknown start after many running writers is still reported, and still withholds the advice.
    refusal = writers_unsettled_refusal(running + [UNBOUND])
    assert refusal["blockers"] == running + [UNBOUND] and "action" not in refusal


PAUSED_HELD = {"running": "running_delegated_leaf", "unknown": "unbound_unknown_start",
               "running_and_unknown": "running_and_unknown_start"}


@pytest.mark.parametrize("condition", ["settled", "running", "unknown", "running_and_unknown", "point_refusal"])
def test_owner_resume_of_a_paused_held_continue_rechecks_its_predecessor(tmp_path, monkeypatch, condition):
    from ouroboros import owner_pause
    from ouroboros.work_stop import hold_fact as budget_hold_fact
    from ouroboros.gateway.task_continue import api_task_continue
    from ouroboros.gateway.task_pause import api_task_pause
    from ouroboros.gateway.tasks import api_task_resume
    from supervisor.events_budget import hold_budget_row
    from ouroboros.work_stop import continuation_refusal, current_stop

    queue, workers = _held(tmp_path, monkeypatch, PAUSED_HELD.get(condition, "running_delegated_leaf"))
    status, ack = _http(api_task_continue, PRED, {"action_nonce": NONCE})
    assert status == 200 and ack["held"]
    successor = ack["successor_task_id"]
    held = next(row for row in workers.PENDING if row["id"] == successor)
    status, pause = _http(api_task_pause, successor, {"request_id": "pause-held-continue"})
    assert status in {200, 202} and pause["ok"]
    fence = owner_pause.read_fence(tmp_path, successor)
    assert owner_pause.fence_closed(fence)
    if condition in {"settled", "point_refusal"}:
        assert dc.emit(tmp_path, dc.SETTLED, {"run_id": "run-leaf", "task_id": PRED, "route": "codex",
                                            "model": "leaf-model", "state": "completed"})
        dc._CUSTODY.clear()
    if condition == "point_refusal":
        hold_budget_row(held, reason="separate_owner_hold", result_root=tmp_path)
        # An unknown name is a point refusal, not an owner-only stop (A1).
        assert continuation_refusal(held)["reason"] == "separate_owner_hold"
    status, result = _http(api_task_resume, successor)
    print("paused Continue Resume:", condition, status, result)
    if condition in {"settled", "point_refusal"}:
        assert status == 200 and result["ok"], result
        assert result["same_generation"] and result["owner_pause_released"]
        assert held in workers.PENDING and budget_hold_fact(held) is None
        assert not owner_pause.fence_closed(owner_pause.read_fence(tmp_path, successor))
        assert successor not in queue.BUDGET_ROOT_FENCES
        status, replay = _http(api_task_continue, PRED, {"action_nonce": NONCE})
        assert status == 200 and not replay["held"] and "blockers" not in replay
    else:
        assert status != 200 and held in workers.PENDING
        assert owner_pause.read_fence(tmp_path, successor) == fence
        # Pause replaced the writer wait; a refusal keeps this current tree stop (A6).
        assert budget_hold_fact(held) is None
        assert current_stop(held, {"owner": fence})["cause"] == "pause"
        # The same reply as the unpaused held Continue: the census facts, advice only for running work.
        facts = {"running": [RUNNING], "unknown": [UNBOUND], "running_and_unknown": [RUNNING, UNBOUND]}
        expected = sorted(({**fact, "task_id": PRED, "review_owned": False} for fact in facts[condition]),
                          key=json.dumps)
        assert status == 409 and result["error"] == "predecessor_writers_unsettled", result
        assert sorted(result["blockers"], key=json.dumps) == expected
        assert result.get("action") == ("wait_or_stop_the_previous_work" if condition == "running" else None)


def test_a_census_past_twenty_rows_reaches_every_continue_consumer_whole(tmp_path, monkeypatch):
    """21 running delegated leaves and then an author model request with no result (the census lists
    model attempts after every delegated run): the Continue answer, its persisted hold, its Activity
    row and the held successor's refused Resume carry all 22 facts, and the advice stays withheld."""
    from ouroboros.work_stop import hold_fact as budget_hold_fact
    from ouroboros.gateway.task_continue import api_task_continue
    from ouroboros.gateway.tasks import api_task_resume
    from tests.test_continue_independent_review import _open_attempt

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setenv("TOTAL_BUDGET", "1000")
    _interrupted(tmp_path, PRED, origin={"source": "owner_restart", "reason": "owner_restart", "scope": "single"})
    dc._CUSTODY.clear()
    for n in range(21):
        assert dc.record_started(tmp_path, dc.RunCustody(
            run_id=f"run-{n:02d}", task_id=PRED, route_id="codex", model="leaf-model",
            snapshot_id=f"snap-{n:02d}", authority_source="active_workspace", ledger_root=str(tmp_path)))
    attempt = _open_attempt(tmp_path, source="delegated_subagent")
    dc._CUSTODY.clear()
    status, ack = _http(api_task_continue, PRED, {"action_nonce": NONCE})
    assert status == 200 and ack["held"] is True, ack
    census = ack["blockers"]
    assert [b["kind"] for b in census] == ["delegated_run"] * 21 + ["model_handoff"], census
    assert census[-1]["attempt_id"] == attempt
    successor = ack["successor_task_id"]
    assert budget_hold_fact(next(row for row in workers.PENDING if row["id"] == successor))["blockers"] == census
    events = [json.loads(line) for line in (tmp_path / "logs" / "events.jsonl").read_text().splitlines()]
    [admitted] = [e for e in events if e.get("type") == "owner_continue_admitted"]
    assert admitted["blockers"] == census
    status, replay = _http(api_task_continue, PRED, {"action_nonce": NONCE})
    assert status == 200 and replay["replay"] and replay["successor_task_id"] == successor, replay
    assert replay["blockers"] == census
    status, refusal = _http(api_task_resume, successor)
    assert status == 409 and refusal["error"] == "predecessor_writers_unsettled", refusal
    assert refusal["blockers"] == census and "action" not in refusal, refusal


def test_the_owner_pause_census_reads_its_closed_fence_only_for_queued_members(tmp_path, monkeypatch):
    """With no queue latch projected, the owner Pause's census (its caller's closed fence) holds a
    queued member, never a running one; Continue's census, which names no fence, is unchanged."""
    from supervisor.continuation_admission import CONTINUE_PURPOSE, conflicting_writers

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, PRED)
    dc._CUSTODY.clear()
    workers.PENDING.append({"id": "queued-child", "type": "task", "root_task_id": PRED, "parent_task_id": PRED})
    workers.RUNNING["running-child"] = {"task": {"id": "running-child", "root_task_id": PRED}}
    assert PRED not in queue.BUDGET_ROOT_FENCES

    def members(**kwargs):
        return sorted((b["kind"], b["task_id"]) for b in conflicting_writers(queue, PRED, **kwargs)
                      if b["kind"] in {"running_member", "dispatchable_member"})

    assert members(owner_pause_fence_id="fence-closed") == [("running_member", "running-child")]
    expected = [("dispatchable_member", "queued-child"), ("running_member", "running-child")]
    assert members() == expected and members(purpose=CONTINUE_PURPOSE) == expected
    assert PRED not in queue.BUDGET_ROOT_FENCES, "the census projects nothing into the queue"
