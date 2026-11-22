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


def test_cascade_stop_cost_does_not_grow_with_other_trees_open_work(tmp_path, monkeypatch):
    from unittest.mock import patch

    from ouroboros.delegate_custody_reconcile import tree_open_runs

    monkeypatch.setattr(current, "json_loads", lambda raw: json.loads(raw))
    measurements = []
    for size in (8, 256):
        root = tmp_path / str(size)
        with store.open_store(root, write=True) as txn:
            for tid, parent in (("tree", ""), ("child", "tree"), ("leaf", "child"), ("sibling", "tree"),
                                ("bound", "tree"), ("bound-leaf", "bound")):
                txn.store_task({"task_id": tid, "root_task_id": "tree", "parent_task_id": parent, "status": "completed"})
            for index in range(size):
                txn.store_task({"task_id": f"foreign-{index}", "parent_task_id": "elsewhere",
                                "root_task_id": "elsewhere", "status": "completed", "result": "UNRELATED_BODY"})
                _put(txn, f"foreign-{index}", state="held", _payload="UNRELATED_BODY")
                txn.put_row("custody_open", f"invocation:{index}", {"resume_decisions": {"foreign": {}},
                            "request": {"task_id": "foreign", "root_task_id": "foreign", "_payload": "UNRELATED_BODY"}})
            # Legacy starts without a root are indexed under their starter, even after its queue row is gone.
            txn.put_row("custody_open", "invocation:lost", {"resume_decisions": {"tree": {}},
                        "request": {"task_id": "leaf"}})
            _put(txn, "live", state="held", task_id="bound-leaf", root_task_id="tree")
            txn.conn.execute("ANALYZE")
        with patch.object(store, "open_store", wraps=store.open_store) as reads:
            results, counts = _measure(root, monkeypatch, lambda r: (
                [current.cascade_work_open(r, tid) for tid in ("tree", "child", "sibling")],
                [row.run_id for row in tree_open_runs(r, "bound")]))
        assert reads.call_count == 4  # one snapshot per selection, across pending -> STARTED
        assert results == ([True, True, False], ["live"]) and counts["unrelated_bytes"] == 0
        measurements.append(counts)
    assert measurements[1]["steps"] <= measurements[0]["steps"] + 128


def test_activity_and_boot_read_current_custody_without_replaying_closed_history(tmp_path, monkeypatch):
    monkeypatch.setattr(current, "json_loads", lambda raw: json.loads(raw))
    counts = []
    for size in (8, 256):
        root = tmp_path / str(size)
        with store.open_store(root, write=True) as txn:
            for index in range(size):
                task = f"old-{index}"
                txn.store_task({"task_id": task, "status": "completed", "result": "UNRELATED_BODY",
                                "owner_pause": {"state": "released", "active_delegated_start": {"root_task_id": task}}})
                _put(txn, task, _payload="UNRELATED_BODY")
                txn.put_row("custody_refused", task, {"request": {"task_id": task, "note": "UNRELATED_BODY"}})
            _put(txn, "held", state="held", root_task_id="tree")
            txn.put_row("custody_open", "invocation:lost", {"resume_decisions": {"tree": {}},
                        "request": {"task_id": "starter", "root_task_id": "tree"}})
            txn.store_task({"task_id": "tree", "status": "completed", "owner_pause": {
                "state": "requested", "active_delegated_start": {"root_task_id": "tree"}}})
            txn.store_task({"task_id": "manual", "status": "scheduled", "owner_pause": {"state": "paused"}})
            txn.conn.execute("ANALYZE")
        result, measured = _measure(root, monkeypatch, lambda r: (
            current.open_work_custody(r), hot.recovered_owner_pause_roots(r)))
        ((runs, starts, unreadable), recovered) = result
        assert sorted(runs) == ["held"] and starts == {"tree": [("lost", True)]} and not unreadable
        assert recovered == ["tree"]  # no released history or independent owner Pause was selected
        assert measured["unrelated_bytes"] == 0
        counts.append(measured)
    assert counts[0]["statements"] == counts[1]["statements"]
    assert counts[1]["steps"] <= counts[0]["steps"] + 64


@pytest.mark.parametrize("lineage", ["absent", "implicit_self", "old_metadata", "damaged_body"])
def test_cascade_stop_finds_the_requests_owner_with_incomplete_task_lineage(tmp_path, lineage):
    with store.open_store(tmp_path, write=True) as txn:
        if lineage != "absent":
            txn.store_task({"task_id": "child", "status": "completed",
                            **({"metadata": {"root_task_id": "tree"}} if lineage == "old_metadata" else {})})
        if lineage in ("old_metadata", "damaged_body"):
            txn.conn.execute("UPDATE tasks SET root_task_id=NULL WHERE task_id='child'")
        if lineage == "damaged_body":
            txn.conn.execute("UPDATE task_bodies SET body='{' WHERE task_id='child'")
        request = {"task_id": "child", "root_task_id": "tree"}
        txn.put_row("custody_open", "invocation:lost", {"request": request})
    assert not current.cascade_work_open(tmp_path, "child")
    with store.open_store(tmp_path, write=True) as txn:
        # The older writer invalidates the request's tree projection.
        txn.conn.execute("UPDATE obligations SET facts=? WHERE set_name='custody_open' AND id='invocation:lost'",
                         (json.dumps({"request": request, "resume_decisions": {"tree": {}}}),))
    assert current.cascade_work_open(tmp_path, "child")
    assert current.cascade_work_open(tmp_path, "tree")
    assert not current.cascade_work_open(tmp_path, "elsewhere")


@pytest.mark.parametrize("pre_index_schema", [False, True])
@pytest.mark.parametrize("damage", ["invalid_json", "missing_request"])
def test_cascade_stop_reads_legacy_held_facts_and_propagates_unreadable_requests(tmp_path, pre_index_schema, damage):
    with store.open_store(tmp_path, write=True) as txn:
        txn.store_task({"task_id": "child", "root_task_id": "tree", "status": "completed"})
        txn.store_task({"task_id": "unrelated", "root_task_id": "unrelated", "status": "completed"})
        _put(txn, "old", legacy=True, task_id="child", root_task_id="tree", state="held", spend=float("nan"))
        if damage == "invalid_json":
            txn.conn.execute("INSERT INTO obligations(set_name,id,facts) VALUES ('custody_open','invocation:bad','{')")
        else:
            txn.put_row("custody_open", "invocation:bad", {"resume_decisions": {"tree": {}}})
        if pre_index_schema:
            txn.set_meta("schema_version", hot.VERSION - 1)
    assert current.cascade_work_open(tmp_path, "tree")
    assert current.cascade_work_open(tmp_path, "child")
    with pytest.raises(ValueError, match="Expecting|no task or tree"):
        current.cascade_work_open(tmp_path, "unrelated")


@pytest.mark.parametrize("complete,consumed,settled,disposed", [
    (None, None, None, None), (False, False, False, ""), (True, True, True, "applied"),
    ("0", "1", "1", []), ([], {}, 2, {}),
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


def _legacy_task(txn, tid, **fields):
    """An older writer's row: the body carries the control facts, the derived columns stay NULL."""
    document = {"task_id": tid, "status": "running", "_schema_version": 1, **fields}
    columns = {key: value for key, value in store.typed_columns(document).items() if key not in hot.TASK_COLUMNS}
    columns["root_task_id"] = document.get("root_task_id")
    columns["write_seq"] = (txn.row(tid, "write_seq") or 0) + 1
    txn.conn.execute("INSERT OR REPLACE INTO task_bodies VALUES (?,?)", (tid, store.encode_document(document)))
    names = ("task_id", *columns)
    txn.conn.execute(f"INSERT OR REPLACE INTO tasks ({','.join(names)}) VALUES ({','.join('?' * len(names))})",
                     (tid, *columns.values()))


def test_boot_fence_discovery_reads_an_older_writers_body_kept_fence(tmp_path):
    """A supported older writer keeps a closed fence in the task body while its derived columns
    stay NULL: boot discovery must read that body (the cold_custody_rows pattern), not infer the
    fence absent from the cleared columns while read_fence still finds it."""
    from ouroboros import owner_pause

    fence = {"state": "requested", "fence_id": "fence-legacy", "root_task_id": "legacy-tree",
             "active_delegated_start": {"root_task_id": "legacy-tree", "run_id": "run", "choice_id": "choice"}}
    with store.open_store(tmp_path, write=True) as txn:
        txn.store_task({"task_id": "modern-tree", "status": "completed", "owner_pause": {
            "state": "requested", "active_delegated_start": {"root_task_id": "modern-tree"}}})
        _legacy_task(txn, "legacy-tree", owner_pause=fence)
        txn.store_task({"task_id": "legacy-quiet", "status": "completed", "owner_pause": {
            "state": "requested"}})  # a closed fence without an active delegated start names no root
    assert owner_pause.fence_closed(owner_pause.read_fence(tmp_path, "legacy-tree"))
    assert sorted(hot.recovered_owner_pause_roots(tmp_path)) == ["legacy-tree", "modern-tree"]


def test_boot_fence_discovery_skips_a_fenced_tree_whose_body_read_fails(tmp_path):
    """An underived row whose body is damaged names no root, but it never poisons the discovery
    of the other underived rows."""
    with store.open_store(tmp_path, write=True) as txn:
        _legacy_task(txn, "broken", owner_pause={"state": "requested", "root_task_id": "broken",
                                                "active_delegated_start": {"root_task_id": "broken"}})
        _legacy_task(txn, "whole", owner_pause={"state": "paused", "root_task_id": "whole",
                                               "active_delegated_start": {"root_task_id": "whole"}})
        txn.conn.execute("UPDATE task_bodies SET body='{' WHERE task_id='broken'")
    assert hot.recovered_owner_pause_roots(tmp_path) == ["whole"]
