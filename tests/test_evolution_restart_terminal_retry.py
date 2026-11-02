"""#737: a consumed terminal retains its exact lifecycle decision through transient faults."""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from tests.test_agent_restart_intent import enqueue, evolution_completion, runtime, worker  # noqa: F401

pytestmark = pytest.mark.serial


def prepare(runtime, monkeypatch):  # noqa: F811
    from ouroboros.utils import append_jsonl
    from supervisor import events

    intent, tx, finish = evolution_completion(runtime, monkeypatch)
    _, delivered = worker(runtime, monkeypatch)
    runtime.host.WORKERS = runtime.workers.WORKERS
    runtime.host.bridge = SimpleNamespace(push_log=lambda *a: None)
    runtime.host.append_jsonl = append_jsonl
    runtime.host.send_with_budget = lambda *a, **k: pytest.fail("no outward notifications in test")
    monkeypatch.setattr(events, "_maybe_notify_provider_death", lambda *a: None)
    queued = enqueue(runtime, "independent-after-cycle")
    requested = []
    monkeypatch.setattr(runtime.server, "_perform_supervisor_restart", lambda *a, **k: requested.append(k))
    return intent, tx, finish, queued, delivered, requested


@pytest.mark.parametrize("fault", ["lock", "write", "write_refused", "write_then_error", "unreadable", "missing"])
@pytest.mark.parametrize("status", ["completed", "failed"])
def test_terminal_consumer_retries_same_decision_after_early_failure(runtime, monkeypatch, fault, status):  # noqa: F811
    from supervisor import evolution_lifecycle as evolution

    intent, tx, finish, queued, delivered, requested = prepare(runtime, monkeypatch)
    monkeypatch.setenv("OUROBOROS_EVOLUTION_AUTO_RESTART", "false")
    marker = runtime.root / "state/evolution_campaign.json"
    original = marker.read_bytes()
    with monkeypatch.context() as failure:
        if fault == "lock":
            failure.setattr(runtime.state, "acquire_file_lock", lambda *a, **k: None)
        elif fault in {"write", "write_refused", "write_then_error"}:
            write = evolution._write_evolution_campaign

            def fail_write(*a, **k):
                if fault == "write_refused":
                    return False
                if fault == "write_then_error":
                    write(*a, **k)
                raise OSError("injected terminal campaign write failure")

            failure.setattr(evolution, "_write_evolution_campaign", fail_write)
        elif fault == "unreadable":
            marker.write_text("{broken")
        else:
            marker.unlink()
        finish(status, dispatch=True)
        assert "requester" not in runtime.workers.RUNNING
        assert runtime.restart.agent_restart_pending() and runtime.restart.agent_restart_holds(queued)
        runtime.ctx.task_metadata["evolution_transaction"]["transaction_id"] = "late-mutation"
        runtime.server._check_pending_restart_drain(runtime.host)
        if fault != "write_then_error":
            assert runtime.restart.agent_restart_pending() and runtime.restart.agent_restart_holds(queued)
        else:  # readback proves the write; a replay can resolve before the injected writer recovers
            assert not runtime.restart.agent_restart_pending()
        assert requested == [] and runtime.events.empty()
    if fault in {"missing", "unreadable"}:
        marker.write_bytes(original)
    runtime.server._check_pending_restart_drain(runtime.host)
    runtime.server._check_pending_restart_drain(runtime.host)
    print({"fault": fault, "intent": runtime.restart.agent_restart_snapshot()})
    assert not runtime.restart.agent_restart_pending(), "terminal persistence needs an executable retry after RUNNING retires"
    receipt = json.loads(runtime.queue.QUEUE_SNAPSHOT_PATH.read_text())["restart_intent"]
    assert receipt["request_id"] == intent["request_id"] and receipt["refusal"] == "evolution_auto_restart_off"
    campaign = evolution._read_evolution_campaign()
    assert campaign["cycles_done"] == 1 and len(campaign["history"]) == 1
    assert campaign["history"][0]["transaction"]["transaction_id"] != "late-mutation"
    assert requested == [] and runtime.events.empty()
    runtime.workers.assign_tasks()
    assert [row["id"] for row in delivered] == [queued["id"]]


@pytest.mark.parametrize("status", ["failed", "timeout", "interrupted", "cancelled"])
@pytest.mark.parametrize("control", ["none", "owner_stop", "panic"])
def test_committed_failed_cycle_keeps_lifecycle_restart_decision(runtime, monkeypatch, status, control):  # noqa: F811
    from ouroboros.server_liveness import drain_worker_events
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor import evolution_lifecycle as evolution

    intent, tx, finish, queued, delivered, requested = prepare(runtime, monkeypatch)
    monkeypatch.setenv("OUROBOROS_EVOLUTION_AUTO_RESTART", "true")
    if control == "owner_stop":
        runtime.state.update_state(lambda state: state.update(evolution_owner_stopped=True))
    elif control == "panic":
        (runtime.root / "state/panic_stop.flag").write_text("panic")
    # The worker publishes its durable failure before task_done/file preparation.
    # Exercise a real restart tick AND assignment in that gap, before the callback exists.
    write_task_result(runtime.root, "requester", status, task_attempt=1)
    stored = load_task_result(runtime.root, "requester", strict=True)
    assert stored["status"] == status and stored["task_attempt"] == 1
    assert "evolution_terminal" not in runtime.restart._AGENT_RESTART
    runtime.server._check_pending_restart_drain(runtime.host)
    runtime.workers.assign_tasks()
    print({"gap_status": stored["status"], "control": control,
           "intent": runtime.restart.agent_restart_snapshot(), "assigned": [row["id"] for row in delivered]})
    assert delivered == [], "a durable failure cannot admit independent work before the cycle decision"
    assert runtime.restart.agent_restart_pending() and runtime.restart.agent_restart_holds(queued)
    assert runtime.restart.agent_restart_snapshot()["request_id"] == intent["request_id"]
    assert requested == [] and runtime.events.empty()
    finish(status, dispatch=True)
    stored = load_task_result(runtime.root, "requester", strict=True)
    print({"requested_status": status, "stored_status": stored.get("status"),
           "attempt": stored.get("task_attempt"), "requester_failed": runtime.restart._requester_failed(runtime.host)})
    assert stored["status"] == status and stored["task_attempt"] == 1
    runtime.server._check_pending_restart_drain(runtime.host)
    if control != "none":
        receipt = runtime.restart.agent_restart_snapshot()
        assert receipt["status"] == "released" and receipt["refusal"] == "evolution_owner_stopped"
        assert requested == [] and runtime.events.empty()
    else:
        assert not runtime.events.empty(), "a failed task does not discard its committed cycle's restart decision"
        drain_worker_events(runtime.events, runtime.host, [time.monotonic(), {}, 0, None],
                            on_restart=runtime.server._handle_restart_in_supervisor)
        assert requested == [{"restart_reason": "finish first", "evolution_restart": True}]
        assert runtime.restart.agent_restart_holds(queued)
    assert runtime.restart.agent_restart_snapshot()["request_id"] == intent["request_id"]
    assert len(evolution._read_evolution_campaign()["history"]) == 1 and delivered == []


@pytest.mark.parametrize("rejection", ["campaign_not_active", "transaction_missing", "transaction_mismatch"])
def test_known_terminal_rejection_releases_only_exact_intent(runtime, monkeypatch, rejection):  # noqa: F811
    from supervisor import evolution_lifecycle as evolution

    intent, tx, finish, queued, delivered, requested = prepare(runtime, monkeypatch)
    campaign = evolution._read_evolution_campaign()
    if rejection == "campaign_not_active":
        campaign["status"] = "completed"
    elif rejection == "transaction_missing":
        campaign.pop("active_transaction")
        runtime.workers.RUNNING["requester"]["task"]["metadata"].clear()
    else:
        campaign["active_transaction"]["transaction_id"] = "another-transaction"
    assert evolution._write_evolution_campaign(campaign)
    runtime.queue.RETURNS_NAMED["other-door"] = "managed_update"
    finish(dispatch=True)
    runtime.server._check_pending_restart_drain(runtime.host)
    receipt = runtime.restart.agent_restart_snapshot()
    print({"rejection": rejection, "intent": receipt})
    assert receipt["status"] == "released" and receipt["request_id"] == intent["request_id"]
    assert receipt["refusal"] == "evolution_" + rejection
    assert runtime.queue.RETURNS_NAMED == {"other-door": "managed_update"}
    assert not runtime.restart.agent_restart_holds(queued) and requested == [] and delivered == []
