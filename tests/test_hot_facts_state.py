"""One request snapshot for live control/phase facts; card/question bodies belong to their reader."""
from contextlib import contextmanager
import pytest

from ouroboros import budget_pause, owner_pause, store, store_hot_facts as hot, task_store, work_stop
from ouroboros.gateway import state
from tests._budget_pause_exact_helpers import _install_queue
from tests.test_hot_facts import _old_task, _task
from tests.test_hot_facts_tree import _measure


def _seed(root, size):
    with store.open_store(root, write=True) as txn:
        _task(txn, "working")
        _task(txn, "finalizing", root_phase_checkpoint={"post_task_synthesis": "running"})
        _task(txn, "pausing", budget_pause={"state": "pausing", "task_attempt": 2})
        _task(txn, "queued")
        _task(txn, "paused", owner_pause={"state": "paused", "fence_id": "f", "finishing_reviews": ["review"]})
        _task(txn, "direct", owner_pause={"state": "requested", "fence_id": "d"},
              owner_wait={"state": "waiting", "reason": "owner_pause"})
        for index in range(size):
            _task(txn, f"foreign-{index:05}", status="completed", result="UNRELATED_BODY" * 100)


def _census(root, monkeypatch, availability=None):
    queue, _state, workers = _install_queue(root, monkeypatch)
    for tid in ("working", "finalizing", "pausing"):
        workers.RUNNING[tid] = {"task": {"id": tid, "root_task_id": tid}, "attempt": 2, "started_at": 1}
    workers.PENDING[:] = [{"id": "queued"}, {"id": "paused"}]
    queue.BUDGET_ROOT_FENCES["paused"] = {"cause": "owner_pause", "fence_id": "f", "status": "paused"}
    direct = [{"activity_id": "direct", "phase": "working", "kind": "direct_chat", "chat_id": 0}]
    # This contract measures D3/D4's typed facts. D5's card/question materializer
    # remains a separate reader, owned by WU5b; it must not hide control-body reads.
    monkeypatch.setattr(state, "_task_activity_facts", lambda *_: {})
    monkeypatch.setattr("ouroboros.projects_registry.list_reserved_projects", lambda _root: [])
    monkeypatch.setattr("ouroboros.post_task_checkpoint.post_task_model_waits", lambda _root: [])
    return lambda: state._chat_activities_snapshot_safe(
        root, direct_turns=direct, availability=availability if availability is not None else {"complete": True})


def test_census_uses_one_body_free_transaction_at_20_and_2000_rows(tmp_path, monkeypatch):
    roots = [tmp_path / "small", tmp_path / "large"]
    for root, size in zip(roots, (20, 2000)):
        _seed(root, size)
    real, measurements = store.open_store, []
    for root in roots:
        with monkeypatch.context() as patch:
            call = _census(root, patch)
            transactions = []
            active = []

            @contextmanager
            def opened(*args, **kwargs):
                from supervisor import queue
                assert not queue._queue_lock._is_owned(), "store read held the queue lock"
                transactions.append(kwargs["write"])
                with real(*args, **kwargs) as txn:
                    active.append(txn)
                    try:
                        yield txn
                    finally:
                        active.pop()

            def details(*_):
                assert not active, "display consumer held the store transaction"
                return {}

            patch.setattr(store, "open_store", opened)
            patch.setattr(state, "_task_activity_facts", details)
            patch.setattr(store.Txn, "body_text", lambda *_: pytest.fail("derived census read a body"))
            rows, cost = _measure(root, patch, lambda _: call())
            assert transactions == [False]
            phases = {row["activity_id"]: row["phase"] for row in rows}
            assert phases == {"working": "working", "finalizing": "finalizing", "pausing": "budget_pausing",
                              "queued": "queued", "paused": "budget_paused", "direct": "budget_paused"}
            assert next(row for row in rows if row["activity_id"] == "paused")["finishing_reviews"] is True
            measurements.append(cost)
    assert measurements[0] == measurements[1]
    assert measurements[0]["unrelated_bytes"] == 0


def test_snapshot_keeps_unknown_unknown_and_does_not_retry_each_member(tmp_path, monkeypatch):
    _seed(tmp_path, 0)
    availability = {"complete": True}
    call = _census(tmp_path, monkeypatch, availability)
    reads = []

    def unavailable(*args):
        reads.append(1)
        raise store.StoreBusy("injected read gap")

    with monkeypatch.context() as patch:
        patch.setattr(hot, "_read_rows", unavailable)
        rows = call()
    assert reads == [1]
    assert all(row["phase"] == "unknown" for row in rows)
    assert availability["complete"] is False
    availability["complete"] = True
    rows = call()
    assert availability["complete"] is True
    assert next(row for row in rows if row["activity_id"] == "working")["phase"] == "working"


def test_request_snapshot_is_coherent_and_never_leaks_into_the_next_read(tmp_path):
    with store.open_store(tmp_path, write=True) as txn:
        _task(txn, "root", budget_pause={"state": "pausing", "task_attempt": 1})
    with hot.snapshot([(tmp_path, "root")]):
        budget_pause.set_budget_pause(tmp_path, "root", {"state": "paused", "task_attempt": 1})
        assert budget_pause.budget_pause_row(tmp_path, "root")["state"] == "pausing"
    assert budget_pause.budget_pause_row(tmp_path, "root")["state"] == "paused"
    with pytest.raises(RuntimeError):
        with hot.snapshot([(tmp_path, "root")]):
            raise RuntimeError("consumer failed")
    budget_pause.set_budget_pause(tmp_path, "root", {"state": "resumed", "task_attempt": 1})
    assert budget_pause.budget_pause_row(tmp_path, "root")["state"] == "resumed"


@pytest.mark.parametrize("damaged_body", ["{}", "[]"])
def test_legacy_shape_error_is_unknown_until_the_authority_changes(tmp_path, monkeypatch, damaged_body):
    """Only the damaged row is unknown; its neighbours in the same store keep their phases."""
    _seed(tmp_path, 0)
    availability = {"complete": True}
    call = _census(tmp_path, monkeypatch, availability)
    with store.open_store(tmp_path, write=True) as txn:
        _old_task(txn, "working")
        txn.conn.execute("UPDATE task_bodies SET body=? WHERE task_id='working'", (damaged_body,))
    phases = {"working": "working", "finalizing": "finalizing", "pausing": "budget_pausing",
              "queued": "queued", "paused": "budget_paused", "direct": "budget_paused"}
    rows = {row["activity_id"]: row for row in call()}
    assert {tid: row["phase"] for tid, row in rows.items()} == {**phases, "working": "unknown"}
    assert availability["complete"] is False and rows["working"]["can_resume"] is False
    availability["complete"] = True
    with store.open_store(tmp_path, write=True) as txn:
        _task(txn, "working")
    assert {row["activity_id"]: row["phase"] for row in call()} == phases
    assert availability["complete"] is True


def test_phase_and_fence_publication_roll_back_with_the_body(tmp_path):
    with store.open_store(tmp_path, write=True) as txn:
        _task(txn, "root", owner_pause={"state": "requested", "fence_id": "f"},
              owner_wait={"state": "waiting", "reason": "owner_pause"})
        before = txn.load("root")
        txn.conn.execute("CREATE TEMP TRIGGER reject_phase BEFORE INSERT ON tasks "
                         "BEGIN SELECT RAISE(ABORT,'injected phase failure'); END")
    change = lambda row: {**row, "root_phase_checkpoint": {"post_task_synthesis": "paused"},
                          "owner_wait": {"state": "retained", "reason": "sleep"}}
    with pytest.raises(store.StoreError, match="phase failure"):
        task_store.patch_document(tmp_path, "root", change)
    with pytest.raises(store.StoreError, match="phase failure"):
        owner_pause.set_fence_state(tmp_path, "root", fence_id="f", state="paused")
    with store.open_store(tmp_path, write=True) as txn:
        assert txn.load("root") == before
        assert txn.row("root", "post_task_synthesis", "owner_wait_state", "owner_pause_state") == ("", "waiting", "requested")
        txn.conn.execute("DROP TRIGGER reject_phase")
    task_store.patch_document(tmp_path, "root", change)
    owner_pause.set_fence_state(tmp_path, "root", fence_id="f", state="paused")
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.row("root", "post_task_synthesis", "owner_wait_state", "owner_wait_reason", "owner_pause_state") == (
            "paused", "retained", "sleep", "paused")


def test_phase_controls_preserve_attempt_and_warm_pause_predicates(tmp_path, monkeypatch):
    _seed(tmp_path, 0)
    with hot.snapshot([(tmp_path, "pausing"), (tmp_path, "direct")]):
        monkeypatch.setattr(store.Txn, "body_text", lambda *_: pytest.fail("control read a body"))
        assert work_stop.row_pausing(tmp_path, {"_attempt": 2}, "pausing") is True
        assert work_stop.row_pausing(tmp_path, {"_attempt": 1}, "pausing") is False
        assert state._direct_activity_pause_projection(tmp_path, {"activity_id": "direct", "phase": "working"}, {})["phase"] == "budget_paused"


def test_late_owner_registry_failure_preserves_queue_census(tmp_path, monkeypatch):
    _seed(tmp_path, 0)
    call = _census(tmp_path, monkeypatch)
    monkeypatch.setattr("ouroboros.post_task_checkpoint.post_task_model_waits",
                        lambda _root: (_ for _ in ()).throw(RuntimeError("registry unavailable")))
    assert len(call()) == 6


def test_display_memo_keeps_the_signature_of_the_body_it_actually_read(tmp_path, monkeypatch):
    with store.open_store(tmp_path, write=True) as txn:
        _task(txn, "task", status="running")
    real, reads = task_store.read_document, []

    def racing_read(*args, **kwargs):
        reads.append(1)
        task_store.patch_document(tmp_path, "task", lambda row: {**row, "status": "completed"})
        return real(*args, **kwargs)

    monkeypatch.setattr(task_store, "read_document", racing_read)
    with hot.snapshot([(tmp_path, "task")]):
        assert state._task_activity_facts(tmp_path, "task")["display"]["status"] == "completed"
    with hot.snapshot([(tmp_path, "task")]):
        assert state._task_activity_facts(tmp_path, "task")["display"]["status"] == "completed"
    assert reads == [1], "a new body was cached under an older snapshot's signature"
