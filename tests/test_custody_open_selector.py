"""The open-only custody question reads the held rows that may be open, never the settled held majority.

``open_runs`` (the Stop door, the join ledger, the work projection, the roster) used to decode every
held row (an undisposed patch, an owned registration and collection debt keep a settled run held: live,
113 of 119) to answer which runs are still open. The selector narrows on the held index in SQL first.
"""
import json

from ouroboros import delegate_custody_current as current, delegate_custody_reconcile as reconcile, store


def _put(txn, rid, *, state, **facts):
    txn.put_custody(rid, {"run_id": rid, "task_id": f"task-{rid}", "root_task_id": "tree", **facts}, state=state)


def _seed(root, history=40):
    with store.open_store(root, write=True) as txn:
        _put(txn, "open", state="held")
        _put(txn, "open-false", state="held", settled=False)
        _put(txn, "owned", state="held", settled=True, project_owned=True, project_id="p")
        _put(txn, "patch", state="held", settled=True, snapshot_id="snap")
        _put(txn, "numeric", state="held", settled=1)
        _put(txn, "odd", state="held", settled="yes")  # no ordinary boolean: SQL passes it, Python decides
        _put(txn, "closed", state="closed", settled=True)
        for index in range(history):
            _put(txn, f"history-{index:03}", state="closed", settled=True, _payload="X" * 200)


def _decoding(monkeypatch):
    decoded, real = [], current._decode
    monkeypatch.setattr(current, "_decode", lambda raw: decoded.append(raw["run_id"]) or real(raw))
    return decoded


def test_the_open_selector_answers_exactly_the_open_runs(tmp_path):
    _seed(tmp_path)
    oracle = {rid for rid, run in current.held_runs(tmp_path).items() if not run.settled}
    assert oracle == {"open", "open-false"}
    assert {run.run_id for run in reconcile.open_runs(tmp_path)} == oracle
    assert {run.run_id for run in reconcile.tree_open_runs(tmp_path, "tree")} == oracle
    assert {run.run_id for run in reconcile.tree_open_runs(tmp_path, "task-open")} == {"open"}


def test_no_settled_held_row_is_decoded_for_an_open_only_question(tmp_path, monkeypatch):
    _seed(tmp_path)
    decoded = _decoding(monkeypatch)
    assert {run.run_id for run in reconcile.open_runs(tmp_path)} == {"open", "open-false"}
    assert sorted(decoded) == ["odd", "open", "open-false"], \
        "a settled held run (patch, owned registration, numeric true) is neither fetched nor decoded"


def test_the_open_read_does_not_grow_with_settled_history(tmp_path, monkeypatch):
    small, large = tmp_path / "small", tmp_path / "large"
    _seed(small, history=1)
    _seed(large, history=200)
    with store.open_store(large, write=True) as txn:  # settled runs custody still holds: the live majority
        for index in range(100):
            _put(txn, f"held-settled-{index:03}", state="held", settled=True, snapshot_id=f"s{index}")
    decoded = _decoding(monkeypatch)
    reconcile.open_runs(small)
    at_small = len(decoded)
    decoded.clear()
    reconcile.open_runs(large)
    assert len(decoded) == at_small, "settled held rows and closed history add no decode to the open question"


def test_a_shared_snapshot_is_still_filtered_without_a_read(tmp_path, monkeypatch):
    _seed(tmp_path)
    state = current.held_runs(tmp_path)
    monkeypatch.setattr(current, "runs", lambda *a, **k: (_ for _ in ()).throw(AssertionError("read")))
    assert {run.run_id for run in reconcile.open_runs(tmp_path, state=state)} == {"open", "open-false"}


def test_invalid_json_facts_pass_to_python(tmp_path):
    """Non-ordinary JSON (``machine_json`` writes NaN) is not judged by SQL: the row reaches Python."""
    with store.open_store(tmp_path, write=True) as txn:
        txn.conn.execute("INSERT INTO custody_runs(run_id,task_id,state,facts,updated_at) VALUES (?,?,?,?,?)",
                         ("nan", "task-nan", "held", json.dumps({"run_id": "nan", "task_id": "task-nan",
                                                                 "cost_usd": float("nan")}), "t"))
    assert set(current.open_held_runs(tmp_path)) == {"nan"}
    assert [run.run_id for run in reconcile.open_runs(tmp_path)] == ["nan"]
