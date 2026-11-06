"""The open-only custody question reads the held rows that may be open, never the settled held majority.

``open_runs`` (the Stop door, the join ledger, the work projection, the roster) used to decode every
held row (an undisposed patch, an owned registration and collection debt keep a settled run held: live,
113 of 119) to answer which runs are still open. The selector narrows on the held index in SQL first.
Whether a whole tree's delegated work is over (``tree_closure``) is the sibling question: a settled run with no
terminal state still holds there, so SQL drops only the definitely settled and terminal runs, never by ``state``.
"""
import json

import pytest

from ouroboros import delegate_custody as custody, delegate_custody_current as current
from ouroboros import delegate_custody_reconcile as reconcile, store
from ouroboros.delegate_custody import TERMINAL_STATES


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


# ---- the whole-tree closure question (``tree_closure``) -----------------------------------------------------------

TREE = "tree-c"


def _requested(root, task, invocation, *, tree=TREE):
    from ouroboros.task_results import write_task_result

    write_task_result(root, tree, "running", root_task_id=tree, delegation_role="root")  # the tree's lineage
    write_task_result(root, task, "running", root_task_id=tree, parent_task_id=tree, delegation_role="subagent")
    assert custody.record_start_requested(
        root, run_id="", task_id=task, invocation_id=invocation, idempotency_key=invocation, operation_id="op",
        max_seconds=0, request={"prompt": "work"}, project_id="", project_owned=False, route="codex",
        surface="change", slot_id="seat", root_task_id=tree, parent_task_id="")


def _started(root, task, invocation, run_id, *, tree=TREE):
    assert custody.record_started(root, custody.RunCustody(
        run_id=run_id, task_id=task, invocation_id=invocation, idempotency_key=invocation, route_id="codex",
        root_task_id=tree, ledger_root=str(root)), shape={"access": "readonly", "mode": "ask"})
    custody._CUSTODY.clear()


def _settled(root, task, run_id, state="succeeded"):
    assert custody.emit(root, custody.SETTLED, {"run_id": run_id, "task_id": task, "state": state})
    custody._CUSTODY.clear()


def _verdict(snapshot, runs):
    return bool(snapshot["pending"]) or any(not run.settled or run.terminal_state not in TERMINAL_STATES
                                            for run in runs.values())


def _answers(root, task):
    """``(open, own run ids)`` from the closure reader and from the full tree read (every run decoded)."""
    full, closure = current.task_snapshot(root, root_task_id=TREE), current.tree_closure(root, TREE, task)
    return ((_verdict(closure, closure["unfinished"]), sorted(closure["runs"])),
            (_verdict(full, full["runs"]), sorted(rid for rid, run in full["runs"].items() if run.task_id == task)))


def _history(root, count=4):
    for index in range(count):  # finished delegated work of other members of the tree
        _requested(root, f"member-{index}", f"inv-h{index}")
        _started(root, f"member-{index}", f"inv-h{index}", f"run-h{index}")
        _settled(root, f"member-{index}", f"run-h{index}")


def test_the_closure_reader_follows_a_run_from_request_to_settlement_as_the_full_read_does(tmp_path):
    _history(tmp_path)
    assert _answers(tmp_path, "owner") == ((False, []), (False, []))
    _requested(tmp_path, "owner", "inv-own")  # pending: no run row yet
    assert _answers(tmp_path, "owner") == ((True, []), (True, []))
    _started(tmp_path, "owner", "inv-own", "run-own")  # the request leaves in the STARTED transaction
    assert current.tree_closure(tmp_path, TREE, "owner")["pending"] == []
    assert _answers(tmp_path, "owner") == ((True, ["run-own"]), (True, ["run-own"]))
    _settled(tmp_path, "owner", "run-own")
    assert _answers(tmp_path, "owner") == ((False, ["run-own"]), (False, ["run-own"]))
    _requested(tmp_path, "member-0", "inv-late", tree="other-tree")  # another tree's work holds nothing here
    assert _answers(tmp_path, "owner") == ((False, ["run-own"]), (False, ["run-own"]))


@pytest.mark.parametrize("end", ["settled_outside_terminal_states", "closed_absent"])
def test_a_settled_run_without_a_terminal_state_holds_the_tree_as_before(tmp_path, end):
    _history(tmp_path)
    _requested(tmp_path, "member-x", "inv-x")
    _started(tmp_path, "member-x", "inv-x", "run-x")
    if end == "closed_absent":  # settled with no terminal observation at all
        assert custody.emit(tmp_path, custody.CLOSED_ABSENT, {"run_id": "run-x", "task_id": "member-x"})
        custody._CUSTODY.clear()
    else:
        _settled(tmp_path, "member-x", "run-x", state="lost")
    run = current.run(tmp_path, "run-x")
    assert run.settled and run.terminal_state not in TERMINAL_STATES
    assert _answers(tmp_path, "owner") == ((True, []), (True, []))


def test_the_settled_terminal_majority_is_neither_fetched_nor_decoded(tmp_path, monkeypatch):
    _history(tmp_path, count=12)
    for name in ("run-o1", "run-o2"):
        _requested(tmp_path, "owner", f"inv-{name}")
        _started(tmp_path, "owner", f"inv-{name}", name)
        _settled(tmp_path, "owner", name)
    _requested(tmp_path, "member-y", "inv-y")
    _started(tmp_path, "member-y", "inv-y", "run-y")  # still open
    decoded = _decoding(monkeypatch)
    snapshot = current.tree_closure(tmp_path, TREE, "owner")
    assert sorted(decoded) == ["run-o1", "run-o2", "run-y"], decoded  # the owner's own runs and the open one
    assert sorted(snapshot["unfinished"]) == ["run-y"] and _verdict(snapshot, snapshot["unfinished"])


@pytest.mark.parametrize("facts, passed", [
    ({"settled": True, "terminal_state": "succeeded"}, False),
    ({"settled": 1, "terminal_state": "failed"}, False),  # numeric one: dropped in SQL like true
    ({"settled": "yes", "terminal_state": "cancelled"}, True),  # no ordinary boolean: passes, Python decides
])
def test_sql_drops_only_runs_its_facts_definitely_settle_with_a_terminal_state(tmp_path, facts, passed):
    with store.open_store(tmp_path, write=True) as txn:  # the store's own custody row writer, as stored facts go
        txn.put_custody("odd", {"run_id": "odd", "task_id": "member-odd", "root_task_id": TREE, **facts},
                        state="closed")
    snapshot = current.tree_closure(tmp_path, TREE)
    assert ("odd" in snapshot["unfinished"]) is passed
    full = current.task_snapshot(tmp_path, root_task_id=TREE)
    assert _verdict(snapshot, snapshot["unfinished"]) == _verdict(full, full["runs"])
