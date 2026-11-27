"""Fresh cancellation checks use the addressed status, not an unrelated body."""
import pytest

from ouroboros import cancel_intents, store, task_store
from ouroboros.task_results import write_task_result


@pytest.mark.parametrize("status,pending", [("running", False), ("cancel_requested", True), ("completed", False)])
def test_cancel_latch_reads_no_body_and_observes_each_status_write(tmp_path, monkeypatch, status, pending):
    write_task_result(tmp_path, "task", "running", payload="x" * 100_000)
    monkeypatch.setattr(cancel_intents, "has_active_intent", lambda *a, **k: False)
    original = store.Txn.body_text
    reads = []
    def read(txn, task_id):
        reads.append(task_id)
        return original(txn, task_id)
    monkeypatch.setattr(store.Txn, "body_text", read)
    assert not cancel_intents.cancel_pending(tmp_path, "task", propagate_busy=True)
    assert reads == []
    write_task_result(tmp_path, "task", status)
    reads.clear()
    assert cancel_intents.cancel_pending(tmp_path, "task", propagate_busy=True) is pending
    assert reads == []


def test_strict_cancel_check_retains_body_identity_validation(tmp_path, monkeypatch):
    write_task_result(tmp_path, "task", "running")
    monkeypatch.setattr(cancel_intents, "has_active_intent", lambda *a, **k: False)
    monkeypatch.setattr(store.Txn, "body_text", lambda *_a: '{"task_id":"other","status":"running"}')
    with pytest.raises(ValueError, match="unreadable or invalid"):
        cancel_intents.cancel_pending(tmp_path, "task", strict=True)


def test_active_intent_remains_authoritative_without_status_read(tmp_path, monkeypatch):
    monkeypatch.setattr(cancel_intents, "active_intent", lambda *a, **k: {"source": "agent_tool", "request_id": "ci_1"})
    monkeypatch.setattr(task_store, "task_facts", lambda *a, **k: pytest.fail("intent already proves cancellation"))
    assert cancel_intents.cancel_pending(tmp_path, "task", propagate_busy=True)


def test_an_expired_park_close_is_no_stop_to_any_live_reader_until_a_stop_replaces_it(tmp_path):
    """One rule (``read_active_intents``): every live Stop reader is blind to a pending Close; any ordinary Stop
    request replaces it whole, with its own source and policy, keeping only the claim."""
    write_task_result(tmp_path, "task", "running")
    token = {"requested_by": "parent", "token": "a" * 64}
    close = cancel_intents.request_cancel(tmp_path, "task", source=cancel_intents.PARKED_CLOSE_SOURCE,
                                          requested_by="parent", parked_close=token)
    assert not cancel_intents.cancel_pending(tmp_path, "task", propagate_busy=True)
    assert cancel_intents.active_intent(tmp_path, "task") is None and not cancel_intents.active_intents(tmp_path)
    assert cancel_intents.active_intent(tmp_path, "task", include_close=True)["request_id"] == close["request_id"]
    claimed = cancel_intents.claim_intent(tmp_path, "task", owner="close_driver")
    # A shutdown Stop has no follow-up fields (the R2 gate missed it): it is still a Stop of its own.
    stop = cancel_intents.request_cancel(tmp_path, "task", source="snapshot_restore",
                                         requested_stop_policy="finalize_then_cancel")
    assert stop["already_requested"] is False and "parked_close" not in stop
    row = cancel_intents.active_intent(tmp_path, "task")
    assert (row["source"], cancel_intents.stop_policy(row)) == ("snapshot_restore", cancel_intents.STOP_POLICY_FINALIZE)
    assert not cancel_intents.conditional_park_close(row) and cancel_intents.cancel_pending(tmp_path, "task")
    assert (row["request_id"], row["generation"]) == (claimed["request_id"], claimed["generation"])
    again = cancel_intents.request_cancel(tmp_path, "task", source=cancel_intents.PARKED_CLOSE_SOURCE,
                                          requested_by="parent", parked_close=token)
    assert again["close_conflict"]  # a Close never joins (or narrows) the Stop that replaced it


@pytest.mark.parametrize("propagate", [False, True])
def test_addressed_cancel_read_keeps_contention_contract(tmp_path, monkeypatch, propagate):
    monkeypatch.setattr(cancel_intents, "has_active_intent", lambda *a, **k: False)
    def busy(*_a, **_k):
        raise store.StoreBusy("current status", reason="contention")
    monkeypatch.setattr(task_store, "task_facts", busy)
    if propagate:
        with pytest.raises(store.StoreBusy):
            cancel_intents.cancel_pending(tmp_path, "task", propagate_busy=True)
    else:
        assert not cancel_intents.cancel_pending(tmp_path, "task")
