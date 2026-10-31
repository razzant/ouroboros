"""Children-first child lookup: the lineage closure against the retired global prefilter.

``find_child_tasks`` used to project every retry/supersession row of the store on every call
(69 bodies on the live install and growing, about fourteen calls per finalization). It now
starts from the rows and queue items naming the parent or root and closes their lineage over
``superseded_by``/``retry_task_id`` in both directions. The oracle is the retired
implementation, kept here verbatim: the FULL projections must stay equal (ids, result,
artifacts, cost, disposition, ``retry_lineage`` with its retained ``terminal_runs``), including
both reproduced counterexamples to a one-hop selection. The scaling guard pins the class:
unrelated retry history adds no body decode, child-store open or queue parse. A wide queue binds
its seeds as one JSON value (a lowered 999-variable limit refuses only the retired one-per-seed
expansion), and large ordinary history adds no lineage-statement work.
"""
from __future__ import annotations

import json
import pathlib
import sqlite3
from types import SimpleNamespace

import pytest

from ouroboros import store, task_status, task_store
from ouroboros.task_status import find_child_tasks
from tests._store_shared import seed_result

RETAINED = {"terminal_runs": [{"run_id": "paid-original-run", "staged_output": {"sha256": "retained-output"}}]}


def _old_select_lineage(root, parent, root_id, *, direct_only=False, queue_members=()):
    """The retired prefilter: every lineage row of the store, plus this parent's/root's subagents."""
    selects = ["SELECT task_id FROM tasks WHERE superseded_by IS NOT NULL",
               "SELECT task_id FROM tasks WHERE retry_task_id IS NOT NULL"]
    params = []
    if parent:
        selects.append("SELECT task_id FROM tasks WHERE parent_task_id=? AND delegation_role='subagent'")
        params.append(parent)
    if root_id and not direct_only:
        selects.append("SELECT task_id FROM tasks WHERE root_task_id=? AND delegation_role='subagent'")
        params.append(root_id)
    with store.open_store(root, write=False) as txn:
        fetched = task_store._fetch(txn, f"t.task_id IN ({' UNION '.join(selects)})", params)
    return task_store._decode(fetched)


def old_find_child_tasks(drive_root, *, parent_task_id="", root_task_id="", exclude_task_id="",
                         scope="subtree", materialize_artifacts=True):
    """The retired ``find_child_tasks`` body (base 954ef8aff), the oracle."""
    ts = task_status
    parent = str(parent_task_id or "").strip()
    root = str(root_task_id or "").strip()
    excluded = str(exclude_task_id or "").strip()
    direct_only = str(scope or "subtree").strip().lower() == "direct"
    selected = _old_select_lineage(pathlib.Path(drive_root), parent, root, direct_only=direct_only)
    events_index = ts._EventsTailIndex(pathlib.Path(drive_root))
    rows = {}
    for row in (ts.effective_task_result(pathlib.Path(drive_root), item, materialize_artifacts=materialize_artifacts,
                                         _events_index=events_index) for item in selected):
        tid = str(row.get("task_id") or "")
        if not tid or tid == excluded:
            continue
        if str(row.get("delegation_role") or "") != "subagent":
            continue
        if parent and str(row.get("parent_task_id") or "") == parent:
            rows[tid] = row
        elif not direct_only and root and str(row.get("root_task_id") or "") == root:
            rows[tid] = row
    snapshot = ts._load_queue_snapshot(pathlib.Path(drive_root))
    for group, status in (("pending", ts.STATUS_SCHEDULED), ("running", ts.STATUS_RUNNING)):
        for item in snapshot.get(group) or []:
            if not isinstance(item, dict):
                continue
            task = item.get("task") if isinstance(item.get("task"), dict) else {}
            tid = str(item.get("id") or task.get("id") or "")
            if not tid or tid == excluded:
                continue
            if str(task.get("delegation_role") or "") != "subagent":
                continue
            if parent and str(task.get("parent_task_id") or "") == parent:
                row = dict(task)
            elif not direct_only and root and str(task.get("root_task_id") or "") == root:
                row = dict(task)
            else:
                continue
            row.setdefault("task_id", tid)
            row["status"] = status
            existing = rows.get(tid, {})
            if not existing:
                existing = ts.load_effective_task_result(pathlib.Path(drive_root), tid,
                                                         materialize_artifacts=materialize_artifacts)
            if not existing:
                rows[tid] = row
                continue
            combined = dict(existing)
            for key, value in row.items():
                if key == "status":
                    combined["status"] = ts._merge_queue_status(str(existing.get("status") or ""), str(value or ""), row)
                elif not combined.get(key) and value:
                    combined[key] = value
            rows[tid] = combined
    return sorted(rows.values(), key=lambda item: (str(item.get("ts") or ""), str(item.get("task_id") or "")))


def _child(task_id, parent, status="completed", **extra):
    return {"task_id": task_id, "status": status, "delegation_role": "subagent", "parent_task_id": parent,
            "root_task_id": extra.pop("root_task_id", parent), "result": f"answer of {task_id}",
            "cost_usd": 0.25, **extra}


def _queue(drive, running=(), pending=()):
    (drive / "state").mkdir(parents=True, exist_ok=True)
    (drive / "state" / "queue_snapshot.json").write_text(json.dumps({
        "running": [{"id": tid, "task": {"id": tid, **task}} for tid, task in running],
        "pending": [{"id": tid, "task": {"id": tid, **task}} for tid, task in pending]}), encoding="utf-8")


def _child_drive_artifact(tmp_path, drive, task_id):
    from ouroboros.artifacts import collect_task_artifact_records, copy_file_to_task_artifacts
    from ouroboros.task_results import write_task_result

    child = drive / "state" / "headless_tasks" / task_id / "data"
    source = tmp_path / f"{task_id}-report.html"
    source.write_text("<h1>child</h1>", encoding="utf-8")
    copy_file_to_task_artifacts(SimpleNamespace(drive_root=child, task_id=task_id), source, kind="user_file")
    write_task_result(child, task_id, "completed", result="drive answer", artifact_status="ready",
                      artifacts=collect_task_artifact_records(child, task_id), ts="2026-01-01T00:00:02Z")
    return str(child)


@pytest.fixture
def lineage_drive(tmp_path):
    """One store holding every shape the closure must keep, beside unrelated retry history."""
    drive = tmp_path / "data"
    sub = {"delegation_role": "subagent"}
    rows = [
        # Astra 1: z-original -> m-retry -> a-child(P); one hop drops z-original's retained output.
        _child("a1-a-child", "PA1"), {"task_id": "a1-m-retry", "status": "failed", "retry_task_id": "a1-a-child"},
        {"task_id": "a1-z-original", "status": "failed", "retry_task_id": "a1-m-retry",
         "delegate_terminal_reconciliation": RETAINED},
        # Astra 2: the child's parent and role live only in the queue.
        {"task_id": "a2-a-child", "status": "running", "result": "partial", "cost_usd": 1.5},
        {"task_id": "a2-z-original", "status": "failed", "retry_task_id": "a2-a-child",
         "delegate_terminal_reconciliation": RETAINED},
        # The same with the queue-only child sorting AFTER its predecessor (the fold order matters).
        {"task_id": "lq-a-original", "status": "failed", "retry_task_id": "lq-z-child",
         "delegate_terminal_reconciliation": RETAINED},
        {"task_id": "lq-z-child", "status": "running"},
        # A typed child sorting after its predecessor: the last row wins, as it always did.
        {"task_id": "rv-a-original", "status": "failed", "superseded_by": "rv-z-child"}, _child("rv-z-child", "PRV"),
        # Cycles and a self pointer.
        _child("cy-1", "PCY", "failed", superseded_by="cy-2"), _child("cy-2", "PCY", superseded_by="cy-1"),
        _child("cy-self", "PCY", retry_task_id="cy-self"),
        # Missing, NULL and empty typed lineage.
        _child("nu-child", "PNU", retry_task_id="nu-ghost"), _child("nu-blank", "PNU", superseded_by="  "),
        _child("nu-empty-typed", "PNU"), {"task_id": "nu-noise-empty", "status": "failed", **sub},
        # Fable F13: a non-final successor with an EMPTY parent, reached only via the queue overlay.
        {"task_id": "f13-x-pred", "status": "failed", "retry_task_id": "f13-s-succ", **sub},
        {"task_id": "f13-s-succ", "status": "running", **sub},
        # A re-parented retry: kept under the new parent only.
        _child("rp-x", "PRP", "failed", retry_task_id="rp-y"), _child("rp-y", "PRP2"),
        # A subtree: a grandchild with its own retry predecessor.
        _child("st-c1", "RT", root_task_id="RT"), _child("st-g1", "st-c1", root_task_id="RT"),
        {"task_id": "st-g0-old", "status": "failed", "retry_task_id": "st-g1"},
        # A terminal workspace row whose artifacts are not final: projected as running.
        _child("dm-child", "PDM", workspace_root="/w/dm"),
        # A child the parent already decided on (disposition) and an excluded sibling.
        _child("ds-child", "PDS"), _child("ds-sibling", "PDS"),
        # Unrelated retry history, the class the lookup must never pay for again.
        *[{"task_id": f"nz-{index}", "status": "failed", **sub, "parent_task_id": "elsewhere",
           "retry_task_id": f"nz-{index + 1}", "child_drive_root": str(drive / "gone" / str(index))}
          for index in range(6)],
    ]
    for row in rows:
        seed_result(drive, row["task_id"], row)
    seed_result(drive, "dr-old", {"task_id": "dr-old", "status": "failed", "retry_task_id": "dr-child"})
    seed_result(drive, "dr-child", _child("dr-child", "PDR", child_drive_root=_child_drive_artifact(tmp_path, drive, "dr-child")))
    with store.open_store(drive, write=True) as txn:  # a legacy row may carry '' instead of NULL
        txn.conn.execute("UPDATE tasks SET superseded_by='' WHERE task_id IN ('nu-empty-typed', 'nu-noise-empty')")
    _queue(drive, running=[("a2-a-child", {**sub, "parent_task_id": "PA2"}), ("lq-z-child", {**sub, "parent_task_id": "PLQ"}),
                           ("f13-s-succ", {**sub, "parent_task_id": "PF13", "root_task_id": "PF13"})],
           pending=[("qp-new", {**sub, "parent_task_id": "PA1", "root_task_id": "PA1"})])
    from ouroboros.task_tree_ledger import tree_ledger_append
    from ouroboros.tools.join_ledger import _child_result_sha256

    [decided] = [row for row in old_find_child_tasks(drive, parent_task_id="PDS") if row["task_id"] == "ds-child"]
    assert tree_ledger_append("PDS", "decision", "integrated", task_id="PDS", allow_child_result_disposition=True,
                              data_root=drive, payload={"type": "child_result_disposition", "child_task_id": "ds-child",
                              "disposition": "integrated", "child_result_sha256": _child_result_sha256(decided)}).startswith("OK")
    return drive


LOOKUPS = [(parent, parent) for parent in ("PA1", "PA2", "PLQ", "PRV", "PCY", "PNU", "PF13", "PRP", "PRP2", "PDM", "PDS", "PDR",
                                          "RT", "absent", "elsewhere")] + [("", "RT"), ("st-c1", "RT"), ("PA1", "")]


def _both(drive, **query):
    return old_find_child_tasks(drive, **query), find_child_tasks(drive, **query)


def _lineage(rows, task_id):
    return next(row for row in rows if row["task_id"] == task_id).get("retry_lineage") or []


def test_children_first_closure_equals_the_global_prefilter(lineage_drive):
    for parent, root in LOOKUPS:
        for scope in ("direct", "subtree"):
            for materialize in (True, False):
                old, new = _both(lineage_drive, parent_task_id=parent, root_task_id=root, scope=scope,
                                 materialize_artifacts=materialize)
                assert new == old, (parent, root, scope, materialize)
    old, new = _both(lineage_drive, parent_task_id="PDS", exclude_task_id="ds-sibling")
    assert new == old and [row["task_id"] for row in new] == ["ds-child"]
    # The fixture exercises what it claims: retained receipts across two hops and through the
    # queue overlay, the child-drive artifact view, the recorded disposition, the queue-only arrival.
    for parent, child in (("PA1", "a1-a-child"), ("PA2", "a2-a-child"), ("PLQ", "lq-z-child")):
        lineage = _lineage(find_child_tasks(lineage_drive, parent_task_id=parent), child)
        assert lineage[0]["terminal_runs"] == RETAINED["terminal_runs"], parent
    assert [entry["task_id"] for entry in _lineage(find_child_tasks(lineage_drive, parent_task_id="PA1"), "a1-a-child")] \
        == ["a1-z-original", "a1-m-retry"]
    assert [entry["task_id"] for entry in _lineage(find_child_tasks(lineage_drive, parent_task_id="PF13"), "f13-s-succ")] \
        == ["f13-x-pred"]
    [drive_row] = find_child_tasks(lineage_drive, parent_task_id="PDR")
    assert any(str(item.get("path", "")).endswith("report.html") for item in drive_row["artifacts"])
    [decided] = [row for row in find_child_tasks(lineage_drive, parent_task_id="PDS") if row["task_id"] == "ds-child"]
    assert decided["child_result_disposition"] == "integrated"
    assert "qp-new" in {row["task_id"] for row in find_child_tasks(lineage_drive, parent_task_id="PA1")}
    assert [row["task_id"] for row in find_child_tasks(lineage_drive, parent_task_id="PRP")] == []


def test_a_child_arriving_after_a_lookup_joins_the_next_one(lineage_drive):
    """Post-panel arrival: no lookup carries a stale child set into the next call."""
    old, new = _both(lineage_drive, parent_task_id="PA1")
    assert new == old and "a1-late" not in {row["task_id"] for row in new}
    seed_result(lineage_drive, "a1-late-old", {"task_id": "a1-late-old", "status": "failed", "superseded_by": "a1-late",
                                               "delegate_terminal_reconciliation": RETAINED})
    seed_result(lineage_drive, "a1-late", _child("a1-late", "PA1"))
    old, new = _both(lineage_drive, parent_task_id="PA1")
    assert new == old and _lineage(new, "a1-late")[0]["task_id"] == "a1-late-old"


def _one_hop(real):
    def where(parent, root_id, *, direct_only, queue_members=()):
        kids = "SELECT task_id FROM tasks WHERE parent_task_id=?"
        return (f"t.task_id IN ({kids} UNION SELECT task_id FROM tasks WHERE superseded_by IN ({kids}) "
                f"OR retry_task_id IN ({kids}))", (parent, parent, parent))
    return where


def _store_seeds_only(real):
    return lambda parent, root_id, *, direct_only, queue_members=(): real(parent, root_id, direct_only=direct_only)


def _whole_closure(real):
    def where(parent, root_id, *, direct_only, queue_members=()):
        clause, params = real(parent, root_id, direct_only=direct_only, queue_members=queue_members)
        kin = len([value for value in (parent, "" if direct_only else root_id) if value])
        return clause.split(" AND (t.superseded_by IS NOT NULL")[0], params[:len(params) - kin]
    return where


@pytest.mark.parametrize("broken, parent", [(_one_hop, "PA1"), (_store_seeds_only, "PA2"), (_whole_closure, "PLQ")])
def test_the_oracle_catches_each_incomplete_selection(lineage_drive, monkeypatch, broken, parent):
    """Both directions: the shipped closure matches (above); one hop loses the two-hop receipt,
    store-only seeds lose the queue-only child's, and projecting the whole closure lets a
    lineage-less child overwrite its predecessor's fold."""
    old = old_find_child_tasks(lineage_drive, parent_task_id=parent, scope="direct")
    assert find_child_tasks(lineage_drive, parent_task_id=parent, scope="direct") == old
    monkeypatch.setattr(task_store, "lineage_where", broken(task_store.lineage_where))
    assert find_child_tasks(lineage_drive, parent_task_id=parent, scope="direct") != old


def _seed_history(drive, unrelated):
    """Three children of ``owner`` (one running, one with an absent child store) and UNRELATED
    retry/supersede chains whose rows carry absent child stores and unfinished workspaces."""
    _queue(drive, running=[("kid-c", {"delegation_role": "subagent", "parent_task_id": "owner", "root_task_id": "owner"})])
    with store.open_store(drive, write=True) as txn:
        for row in (_child("kid-a", "owner"), _child("kid-b", "owner", child_drive_root=str(drive / "gone" / "kid-b")),
                    _child("kid-c", "owner", "running")):
            txn.store_task({**row, "_schema_version": 1})
        for index in range(unrelated):
            pointer = {"superseded_by" if index % 2 else "retry_task_id": f"hist-{index + 1}"} if index % 5 != 4 else {}
            txn.store_task({"task_id": f"hist-{index}", "status": "failed" if index % 3 else "completed", "_schema_version": 1,
                            "delegation_role": "subagent", "parent_task_id": f"elsewhere-{index % 7}", "root_task_id": "elsewhere",
                            "workspace_root": "/w/hist", "child_drive_root": str(drive / "gone" / f"hist-{index}"),
                            "result": "unrelated answer " * 200, **pointer})


def _lookup_work(monkeypatch, drive, lookup):
    """Bodies decoded, child stores opened and queue snapshots parsed by one lookup."""
    counts = {"bodies": 0, "child_stores": 0, "queue_parses": 0}
    real_decode, real_load, real_probe, real_queue = task_store._decode, store.Txn.load, store.probe, task_status._load_queue_snapshot
    canonical = pathlib.Path(drive).resolve()

    def decode(*fetched):
        rows = real_decode(*fetched)
        counts["bodies"] += len(rows)
        return rows

    def load(self, task_id):
        document = real_load(self, task_id)
        counts["bodies"] += document is not None
        return document

    def probe(root):
        counts["child_stores"] += pathlib.Path(root).resolve() != canonical
        return real_probe(root)

    def queue(root):
        counts["queue_parses"] += 1
        return real_queue(root)

    with monkeypatch.context() as patch:
        patch.setattr(task_store, "_decode", decode)
        patch.setattr(store.Txn, "load", load)
        patch.setattr(store, "probe", probe)
        patch.setattr(task_status, "_load_queue_snapshot", queue)
        rows = lookup(drive, parent_task_id="owner", root_task_id="owner")
    return counts, rows


def test_unrelated_retry_history_adds_no_child_lookup_work(tmp_path, monkeypatch):
    """One lookup's decodes, child-store opens and queue parses are the same beside 10 and 2,000
    unrelated retry/supersede rows and stay within the kept rows plus the one queue read; the
    retired global prefilter (the negative control) grows with that history."""
    work = {}
    for unrelated in (10, 2000):
        _seed_history(tmp_path / f"n{unrelated}", unrelated)
        work[unrelated], rows = _lookup_work(monkeypatch, tmp_path / f"n{unrelated}", find_child_tasks)
        assert [row["task_id"] for row in rows] == ["kid-a", "kid-b", "kid-c"]
    assert work[10] == work[2000]
    assert all(value <= len(rows) + 1 for value in work[2000].values()), work[2000]
    _seed_history(tmp_path / "n200", 200)
    retired = {unrelated: _lookup_work(monkeypatch, tmp_path / f"n{unrelated}", old_find_child_tasks)[0] for unrelated in (10, 200)}
    assert all(retired[200][name] > retired[10][name] for name in work[10]), retired


# ---- a wide queue: the seeds bind as one value ------------------------------------------------

WIDE = 1200  # queue-only seeds, above the 999-variable limit the fault injection below sets
WIDE_LOOKUPS = [("PW", "RW", "direct"), ("PW", "RW", "subtree"), ("", "RW", "subtree")]


def _expanded_lineage_where(parent, root_id, *, direct_only, queue_members=()):
    """The negative control: the five-commit candidate's binding, one variable per queue seed."""
    kin = [(column, value) for column, value in (("parent_task_id", parent),
                                                ("root_task_id", "" if direct_only else root_id)) if value]
    if not kin:
        return "0", ()
    values = tuple(value for _column, value in kin)
    members = sorted({str(member) for member in queue_members if str(member or "").strip()})
    anchors = " OR ".join([f"{column}=?" for column, _value in kin]
                          + [f"task_id IN ({','.join('?' * len(members))})"] * bool(members))
    reach = (f"WITH RECURSIVE reach(id, successor, retry) AS (SELECT task_id, superseded_by, retry_task_id "
             f"FROM tasks WHERE {anchors} UNION SELECT tasks.task_id, tasks.superseded_by, tasks.retry_task_id "
             "FROM reach JOIN tasks ON tasks.superseded_by=reach.id OR tasks.retry_task_id=reach.id "
             "OR tasks.task_id=reach.successor OR tasks.task_id=reach.retry) SELECT id FROM reach")
    child = " OR ".join(f"t.{column}=?" for column, _value in kin)
    return (f"t.task_id IN ({reach}) AND (t.superseded_by IS NOT NULL OR t.retry_task_id IS NOT NULL "
            f"OR (t.delegation_role='subagent' AND ({child})))", (*values, *members, *values))


def _wide_queue(drive, seeds=WIDE):
    """SEEDS queued children of PW/RW whose parent and role live only in the queue; most have no row
    yet (a wide fan-out), three have bare rows: one ends a 40-hop retry chain whose oldest hop holds a
    retained receipt, one sits in a supersession cycle, one points at itself. Beside them a typed
    child and unrelated retry noise."""
    ids = [f"wq-{index:04}" for index in range(seeds)]
    hops = [f"wq-deep-{hop:02}" for hop in range(39, -1, -1)]  # oldest first; it sorts last, so its full fold wins
    rows = [{"task_id": hop, "status": "failed", "retry_task_id": after}
            for hop, after in zip(hops, hops[1:] + [ids[7]])]
    rows[0]["delegate_terminal_reconciliation"] = RETAINED
    noise = [{"task_id": f"wq-noise-{n}", "status": "failed", "retry_task_id": f"wq-noise-{n + 1}"} for n in range(5)]
    rows += [{"task_id": ids[7], "status": "running", "result": "partial", "cost_usd": 0.5},
             {"task_id": ids[9], "status": "failed", "superseded_by": "wq-cycle"},
             {"task_id": "wq-cycle", "status": "completed", "superseded_by": ids[9]},
             {"task_id": ids[11], "status": "failed", "retry_task_id": ids[11]},
             _child("wq-typed", "PW", root_task_id="RW"), *noise]
    for row in rows:
        seed_result(drive, row["task_id"], row)
    sub = {"delegation_role": "subagent", "parent_task_id": "PW", "root_task_id": "RW"}
    _queue(drive, running=[(tid, sub) for tid in ids[:seeds // 2]], pending=[(tid, sub) for tid in ids[seeds // 2:]])
    return ids


def _lineage_bindings(monkeypatch):
    """The variable count each lineage statement of a real lookup binds."""
    bound, real = [], task_store._fetch

    def fetch(txn, where="", params=(), *args, **kwargs):
        params = tuple(params)
        if "WITH RECURSIVE reach" in where:
            bound.append(len(params))
        return real(txn, where, params, *args, **kwargs)

    monkeypatch.setattr(task_store, "_fetch", fetch)
    return bound


def test_a_wide_queue_binds_its_seeds_as_one_value(tmp_path, monkeypatch):
    """The real lookup over 1,200 queue-only seeds equals the global-prefilter oracle in full, and
    its lineage statements bind exactly what a twelve-seed queue's do; the retired expansion bound
    one variable per seed."""
    wide, narrow = tmp_path / "wide", tmp_path / "narrow"
    ids = _wide_queue(wide)
    _wide_queue(narrow, seeds=12)
    bound = _lineage_bindings(monkeypatch)
    for parent, root, scope in WIDE_LOOKUPS:
        query = {"parent_task_id": parent, "root_task_id": root, "scope": scope}
        new = find_child_tasks(wide, **query)
        assert new == old_find_child_tasks(wide, **query), query
        # Every seed is returned (no cap; the superseded one reads as its successor), and the 40-hop receipt folds.
        assert {row["task_id"] for row in new} == {*ids, "wq-cycle", "wq-typed"} - {ids[9]}
        lineage = _lineage(new, ids[7])
        assert [entry["task_id"] for entry in lineage] == [f"wq-deep-{hop:02}" for hop in range(39, -1, -1)]
        assert lineage[0]["terminal_runs"] == RETAINED["terminal_runs"]
    wide_bound, bound[:] = list(bound), []
    for parent, root, scope in WIDE_LOOKUPS:
        find_child_tasks(narrow, parent_task_id=parent, root_task_id=root, scope=scope)
    assert wide_bound == bound == [3, 5, 3]
    assert len(_expanded_lineage_where("PW", "RW", direct_only=False, queue_members=ids)[1]) == WIDE + 4


@pytest.mark.skipif(not hasattr(sqlite3.Connection, "setlimit"), reason="Connection.setlimit needs Python 3.11+")
def test_a_lowered_variable_limit_refuses_only_the_retired_expansion(tmp_path, monkeypatch):
    """Fault injection, not a claim about any engine's default: every current-store connection
    allows 999 variables. The real lookup over 1,200 queue-only seeds still equals the oracle; the
    retired one-variable-per-seed expansion, the negative control, cannot run at all."""
    drive = tmp_path / "data"
    _wide_queue(drive)
    store.close_connections(store.store_path(drive))
    connect = store.connect

    def lowered(*args, **kwargs):
        conn = connect(*args, **kwargs)
        conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
        return conn

    monkeypatch.setattr(store, "connect", lowered)
    with store.open_store(drive, write=False) as txn:
        assert txn.conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER) == 999
    for parent, root, scope in WIDE_LOOKUPS:
        query = {"parent_task_id": parent, "root_task_id": root, "scope": scope}
        assert find_child_tasks(drive, **query) == old_find_child_tasks(drive, **query), query
        with monkeypatch.context() as patch:
            patch.setattr(task_store, "lineage_where", _expanded_lineage_where)
            with pytest.raises(store.StoreError, match="too many SQL variables"):
                find_child_tasks(drive, **query)


def test_large_ordinary_history_adds_no_lineage_statement_work(tmp_path, monkeypatch):
    """Ordinary rows (other parents, no lineage pointer) are never visited: the lineage statements'
    SQLite VM work is identical beside 100 and 20,000 of them, every one of the parent's 300
    children comes back, and a scan of the same table (the probe's sensitivity control) grows."""
    steps, work, real = [], {}, task_store._fetch

    def counted(conn, run):
        count = [0]

        def tick():
            count[0] += 1
            return 0

        conn.set_progress_handler(tick, 1)
        try:
            return run()
        finally:
            conn.set_progress_handler(None, 1)
            steps.append(count[0])

    def fetch(txn, where="", params=(), *args, **kwargs):
        if "WITH RECURSIVE reach" not in where:
            return real(txn, where, params, *args, **kwargs)
        return counted(txn.conn, lambda: real(txn, where, params, *args, **kwargs))

    monkeypatch.setattr(task_store, "_fetch", fetch)
    for ordinary in (100, 20000):
        drive = tmp_path / f"h{ordinary}"
        with store.open_store(drive, write=True) as txn:
            for index in range(300):
                txn.store_task({**_child(f"kid-{index:03}", "PH"), "_schema_version": 1})
            for index in range(ordinary):
                txn.store_task({"task_id": f"old-{index:05}", "status": "completed", "_schema_version": 1,
                                "delegation_role": "subagent", "parent_task_id": f"other-{index % 97}",
                                "root_task_id": "other", "result": "ordinary answer"})
        rows = find_child_tasks(drive, parent_task_id="PH", root_task_id="PH")
        assert [row["task_id"] for row in rows] == [f"kid-{index:03}" for index in range(300)]
        lookup, steps[:] = list(steps), []
        with store.open_store(drive, write=False) as txn:
            counted(txn.conn, lambda: txn.conn.execute(
                "SELECT count(*) FROM tasks WHERE coalesce(parent_task_id, '') || '' = ?", ("PH",)).fetchone())
        work[ordinary], steps[:] = (lookup, steps[0]), []
    assert work[100][0] == work[20000][0] and work[100][0]
    assert work[20000][1] > 20 * work[100][1]
