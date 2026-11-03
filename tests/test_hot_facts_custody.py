"""Indexed custody selections retain the former JSON predicates and atomic emit."""
import json

import pytest

from ouroboros import delegate_custody as custody, delegate_custody_current as current, store, store_hot_facts as hot
from tests.test_hot_facts_tree import _measure


def _put(txn, rid, *, legacy=False, state="closed", **facts):
    facts = {"run_id": rid, "task_id": rid, "root_task_id": "foreign", **facts}
    if legacy:
        txn.conn.execute("INSERT INTO custody_runs(run_id,task_id,state,facts,updated_at) VALUES (?,?,?,?,?)",
                         (rid, facts["task_id"], state, json.dumps(facts), "same"))
    else:
        txn.put_custody(rid, facts, state=state)
        txn.conn.execute("UPDATE custody_runs SET updated_at='same' WHERE run_id=?", (rid,))


def _seed(root, size):
    with store.open_store(root, write=True) as txn:
        for index, fields in enumerate([
            {"root_task_id": "tree", "billing_group": {"billing_group_id": "unavailable:old", "billing_group_limit_usd": 3}},
            {"root_task_id": "tree", "billing_group": {"billing_group_id": "paid", "billing_group_limit_usd": None}},
            {"root_task_id": "", "task_id": "tree", "output_complete": True},
            {"root_task_id": None, "task_id": "tree"},
            {"state": "held", "authority_source": "skill_payload", "settled": True, "patch_disposed": ""},
            {"output_complete": True, "output_consumed": True},
            {"output_complete": "0", "output_consumed": 2},
            {"authority_source": "skill_payload", "settled": True, "patch_disposed": "rejected"},
        ]):
            _put(txn, f"run-{index}", legacy=index % 2 == 0, **fields)
        for name in hot.REQUEST_SETS:
            txn.conn.execute("INSERT INTO obligations(set_name,id,facts) VALUES (?,?,?)", (name, "invocation:old",
                             json.dumps({"request": {"task_id": name, "root_task_id": "tree"}})))
            txn.put_row(name, "invocation:new", {"request": {"task_id": name + "-new", "root_task_id": "tree"}})
        for index in range(size):
            _put(txn, f"foreign-{index:05}", _payload="UNRELATED_BODY" * 100)
            txn.put_row("custody_refused", f"foreign-{index:05}", {
                "request": {"task_id": "other", "root_task_id": "foreign", "_payload": "UNRELATED_BODY" * 100}})


def _oracle(root):
    with store.open_store(root, write=False) as txn:
        ids = {row[0] for row in txn.conn.execute(
            "SELECT DISTINCT task_id FROM custody_runs WHERE task_id<>'' AND (" + current._LEGACY_TREE + ")",
            ("tree", "tree"))}
        requests = []
        for name in hot.REQUEST_SETS:
            for key, facts in txn.set_rows(name).items():
                request = facts.get("request") if isinstance(facts.get("request"), dict) else {}
                tid = str(request.get("task_id") or "")
                if tid and str(request.get("root_task_id") or tid) == "tree":
                    ids.add(tid)
                if name == "custody_open" and key.startswith("invocation:"):
                    requests.append(request)
        carried = [json.loads(row[0]) for row in txn.conn.execute(
            "SELECT facts FROM custody_runs WHERE json_extract(facts,'$.billing_group.billing_group_id')<>'' "
            "AND (" + current._LEGACY_TREE + ") ORDER BY updated_at,run_id", ("tree", "tree"))]
        groups = [row.get("billing_group") or {} for row in carried]
        groups += [request.get("billing_group") or {} for request in requests
                   if str(request.get("root_task_id") or request.get("task_id") or "") == "tree"]
        group = next((group for group in groups if "billing_group_limit_usd" in group
                      and not str(group.get("billing_group_id")).startswith("unavailable:")), None)
        health = list(txn.custody_raw_rows("state=? OR " + current._MAYBE_UNREAD, ("held",)))
        payload = list(txn.custody_raw_rows(current._MAY_HOLD_PAYLOAD))
    return sorted(ids), group, health, payload


def _selected(root):
    return (current.tree_task_ids(root, "tree"), current.tree_billing_group(root, "tree"),
            list(current.health_runs(root)), list(current.payload_snapshot(root)["runs"]))


def test_custody_queries_match_json_oracle_before_and_after_derivation(tmp_path):
    _seed(tmp_path, 20)
    expected = _oracle(tmp_path)
    assert expected[1]["billing_group_id"] == "paid"
    assert "run-2" in expected[2] and "run-5" not in expected[2]
    assert expected[3] == ["run-4"]
    assert _selected(tmp_path) == expected
    assert hot.run_hot_facts(tmp_path)["state"] == "done"
    assert _selected(tmp_path) == expected


def test_custody_tree_and_health_cost_do_not_grow_with_closed_history(tmp_path, monkeypatch):
    roots = [tmp_path / "small", tmp_path / "large"]
    for root, size in zip(roots, (20, 2000)):
        _seed(root, size)
    # Count the custody module's imported decoder as well as ordinary json.loads.
    monkeypatch.setattr(current, "json_loads", lambda raw: json.loads(raw))
    for derive in (False, True):
        if derive:
            for root in roots:
                hot.run_hot_facts(root)
        measured = [_measure(root, monkeypatch, _selected) for root in roots]
        assert measured[0][0] == measured[1][0]
        assert measured[0][1] == measured[1][1]
        assert measured[0][1]["unrelated_bytes"] == 0
    old = [_measure(root, monkeypatch, _oracle)[1] for root in roots]
    assert old[1]["steps"] > old[0]["steps"] * 10
    assert old[1]["unrelated_bytes"] > old[0]["unrelated_bytes"] * 10


@pytest.mark.parametrize("complete,consumed,settled,disposed", [
    (None, None, None, None), (False, False, False, ""), (True, True, True, "applied"),
    ("0", "1", "1", []), ([], {}, 2, {}), (2, 0, 1, "rejected"),
    (float("nan"), False, True, "applied"),
])
def test_typed_conservative_candidates_match_sql_shapes(tmp_path, complete, consumed, settled, disposed):
    facts = {"authority_source": "skill_payload", "output_complete": complete, "output_consumed": consumed,
             "settled": settled, "patch_disposed": disposed}
    with store.open_store(tmp_path, write=True) as txn:
        _put(txn, "shape", **facts)
        row = txn.conn.execute("SELECT output_unread,payload_held,(" + current._MAYBE_UNREAD + "),(" +
                               current._MAY_HOLD_PAYLOAD + ") FROM custody_runs").fetchone()
        assert tuple(row[:2]) == tuple(row[2:])


def test_emit_commits_run_columns_and_outbox_together_or_neither(tmp_path, monkeypatch):
    monkeypatch.setattr(current, "flush_outbox", lambda _root: 0)
    assert custody.emit(tmp_path, custody.STARTED, {"run_id": "run", "task_id": "child", "root_task_id": "tree"})
    with store.open_store(tmp_path, write=True) as txn:
        before = txn.custody_row("run")
        txn.conn.execute("CREATE TEMP TRIGGER reject_projection BEFORE UPDATE OF output_unread ON custody_runs "
                         "BEGIN SELECT RAISE(ABORT,'injected custody projection failure'); END")
    event = {"run_id": "run", "task_id": "child", "staged": True, "artifact": "result.txt", "full_content": True}
    assert not custody.emit(tmp_path, custody.OUTPUT_SPILLED, event)
    with store.open_store(tmp_path, write=True) as txn:
        assert txn.custody_row("run") == before
        assert txn.conn.execute("SELECT output_unread FROM custody_runs").fetchone()[0] == 0
        assert len(txn.outbox_rows()) == 1
        txn.conn.execute("DROP TRIGGER reject_projection")
    assert custody.emit(tmp_path, custody.OUTPUT_SPILLED, event)
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.custody_row("run")["output_complete"] is True
        assert tuple(txn.conn.execute("SELECT root_task_id,output_unread FROM custody_runs").fetchone()) == ("tree", 1)
        assert len(txn.outbox_rows()) == 2
