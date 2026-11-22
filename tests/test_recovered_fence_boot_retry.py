"""A failed boot discovery of recovered-writer fences stays owed and retries on the settlement cadence.

Boot's fence discovery read (queue restore) is the only pass that hands recovered-writer trees to
the settlement sweep. A transient store failure on that one read must not discard the fences:
unknown never becomes absent, and the existing settlement cadence retries the indexed read until
it succeeds."""
import pytest

from ouroboros import store, store_hot_facts
from tests._budget_pause_exact_helpers import _install_queue


@pytest.fixture(autouse=True)
def _fresh_discovery_book():
    from supervisor import owner_pause_control

    owner_pause_control._DISCOVERY_OWED.clear()


def _closed_fence(root, task_id="root-1"):
    fence = {"state": "requested", "fence_id": "fence-boot", "root_task_id": task_id,
             "active_delegated_start": {"root_task_id": task_id, "run_id": "lost", "choice_id": "choice"}}
    with store.open_store(root, write=True) as txn:
        txn.store_task({"task_id": task_id, "status": "running", "owner_pause": fence})
    return fence


def test_failed_boot_fence_discovery_stays_owed_and_the_settlement_pass_retries_it(tmp_path, monkeypatch):
    from supervisor import owner_pause_control, queue_snapshot

    queue, _state, _workers = _install_queue(tmp_path, monkeypatch)
    _closed_fence(tmp_path)
    real = store_hot_facts.recovered_owner_pause_roots

    def failing(root):
        raise store.StoreError("simulated transient boot read failure")

    monkeypatch.setattr(store_hot_facts, "recovered_owner_pause_roots", failing)
    assert queue_snapshot.restore_pending_from_snapshot() == 0
    # The transient failure is not discarded: a discovery retry stays owed.
    assert owner_pause_control.settlement_look_owed(queue)

    attempts = []

    def counted(root):
        attempts.append(root)
        return real(root)

    monkeypatch.setattr(store_hot_facts, "recovered_owner_pause_roots", counted)
    owner_pause_control.settle_requested_owner_pauses(queue)
    assert attempts == [tmp_path]  # the settlement cadence retried the indexed discovery read
    assert not owner_pause_control.settlement_look_owed(queue)  # the owed retry cleared on success
    assert "root-1" in queue.BUDGET_ROOT_FENCES  # the retryed discovery reached the settlement sweep


def test_a_retry_that_fails_again_stays_owed_for_the_next_cadence(tmp_path, monkeypatch):
    from supervisor import owner_pause_control, queue_snapshot

    queue, _state, _workers = _install_queue(tmp_path, monkeypatch)
    _closed_fence(tmp_path)
    real = store_hot_facts.recovered_owner_pause_roots
    calls = []

    def flaky(root):
        calls.append(root)
        if len(calls) < 3:  # the boot read, then the first cadence retry, fail; the second retry succeeds
            raise store.StoreError("simulated recurring boot read failure")
        return real(root)

    monkeypatch.setattr(store_hot_facts, "recovered_owner_pause_roots", flaky)
    queue_snapshot.restore_pending_from_snapshot()  # the boot read fails and stays owed
    owner_pause_control.settle_requested_owner_pauses(queue)  # the cadence retry fails again: still owed
    assert owner_pause_control.settlement_look_owed(queue)
    owner_pause_control.settle_requested_owner_pauses(queue)  # the next cadence pass retries again
    assert not owner_pause_control.settlement_look_owed(queue)
    assert len(calls) == 3
