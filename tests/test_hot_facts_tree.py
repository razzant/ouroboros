"""Stop and retained membership select one tree, including legacy and live bridges."""
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from ouroboros import owner_pause, store, store_hot_facts as hot, task_store
from supervisor.owner_stop import _child_result_projection
from supervisor.terminal_cascade import _cascade_descendant_rows
from tests.test_hot_facts import _old_task, _task


def _seed(root, unrelated):
    root.mkdir()
    with store.open_store(root, write=True) as txn:
        _task(txn, "root", status="completed")
        _task(txn, "mid", parent_task_id="root", root_task_id="root", status="cancelled", result="middle")
        _task(txn, "leaf", parent_task_id="mid", root_task_id="root", status="failed", result="leaf answer")
        _task(txn, "bridge-leaf", parent_task_id="queue-bridge", root_task_id="root", status="completed")
        _task(txn, "outside", status="completed")
        _task(txn, "outside-leaf", parent_task_id="outside", root_task_id="elsewhere", status="completed")
        _old_task(txn, "legacy", metadata={"root_task_id": "root"}, status="completed", result="legacy answer")
        for index in range(unrelated):
            _task(txn, f"foreign-{index:05}", status="completed", result="UNRELATED_BODY" * 100)
    (root / "state/queue_snapshot.json").write_text(json.dumps({
        "pending": [{"id": "queue-bridge", "parent_task_id": "mid"},
                    {"id": "outside", "parent_task_id": "mid", "root_task_id": "root"}],
        "running": [{"id": "queue-leaf", "task": {"id": "queue-leaf", "parent_task_id": "bridge-leaf"}}],
    }), encoding="utf-8")


def _measure(root, monkeypatch, call):
    counts = {"statements": 0, "steps": 0, "unrelated_bytes": 0, "decoded_bytes": 0}
    real_connection, real_loads = store.connection, json.loads

    @contextmanager
    def connection(*args, **kwargs):
        with real_connection(*args, **kwargs) as conn:
            def trace(_sql):
                counts["statements"] += 1

            def step():
                counts["steps"] += 1
                return 0

            conn.set_trace_callback(trace)
            conn.set_progress_handler(step, 1)
            try:
                yield conn
            finally:
                conn.set_trace_callback(None)
                conn.set_progress_handler(None, 0)

    def loads(raw, *args, **kwargs):
        blob = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
        counts["decoded_bytes"] += len(blob)
        if b"UNRELATED_BODY" in blob:
            counts["unrelated_bytes"] += len(blob)
        return real_loads(raw, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(store, "connection", connection)
        patch.setattr(json, "loads", loads)
        result = call(root)
    return result, counts


@pytest.mark.parametrize("target", ["root", "mid"])
def test_stop_cost_and_result_are_equal_at_20_and_2000_settled_tasks(tmp_path, monkeypatch, target):
    roots = [tmp_path / "small", tmp_path / "large"]
    for root, size in zip(roots, (20, 2000)):
        _seed(root, size)
    wanted = {"bridge-leaf": "completed", "leaf": "failed", "queue-bridge": "scheduled", "queue-leaf": "running"}
    if target == "root":
        wanted.update(mid="cancelled", legacy="completed")
    observations = [_measure(root, monkeypatch, lambda r: _cascade_descendant_rows(r, target)) for root in roots]
    assert observations[0][0] == observations[1][0] == wanted
    assert observations[0][1] == observations[1][1]
    assert observations[0][1]["unrelated_bytes"] == 0
    for root in roots:
        assert hot.run_hot_facts(root)["state"] == "done"
    after = [_measure(root, monkeypatch, lambda r: _cascade_descendant_rows(r, target)) for root in roots]
    assert after[0][0] == after[1][0] == wanted
    assert after[0][1] == after[1][1]
    assert after[0][1]["decoded_bytes"] < observations[0][1]["decoded_bytes"]
    # A whole-table implementation must fail the meter, even inside one SELECT.
    old = [_measure(root, monkeypatch, task_store.list_documents)[1] for root in roots]
    assert old[1]["steps"] > old[0]["steps"] * 10
    assert old[1]["unrelated_bytes"] > old[0]["unrelated_bytes"] * 10


def test_tree_census_and_graceful_stop_read_only_their_selected_bodies(tmp_path, monkeypatch):
    root = tmp_path / "store"
    _seed(root, 20)
    for derived in (False, True):
        if derived:
            hot.run_hot_facts(root)
        census, counts = _measure(root, monkeypatch, lambda r: owner_pause.tree_member_results(r, "root"))
        assert set(census.members) == {"root", "mid", "leaf", "bridge-leaf", "legacy"} and census.unresolved == {}
        assert counts["unrelated_bytes"] == 0
        text, counts = _measure(root, monkeypatch, lambda r: _child_result_projection(SimpleNamespace(DRIVE_ROOT=r), "mid"))
        assert "leaf answer" in text and "legacy answer" not in text and "middle" not in text
        assert counts["unrelated_bytes"] == 0
