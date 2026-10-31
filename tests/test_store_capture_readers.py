"""Every reader moved onto ``store_capture`` answers exactly what it answered before (the pre-change code is
kept here as the reference and runs on the same store), and hands Python O(statements) rows, not O(rows):
the cost meter's ``rows_crossed`` stays put as the store grows while the reference's grows with it."""
from __future__ import annotations

import json
import sqlite3

import pytest

from ouroboros import owner_pause, projects_registry, store, store_task_cards, task_store, usage_store
from ouroboros._usage_rows import breakdown_read, breakdown_view
from tests._cost_meter import CostMeter, tagged
from tests._store_shared import seed_result
from tests.fixtures_usage_store import _seed_mixed_ledger, data_root  # noqa: F401 -- the usage fixture

CHAT_IDS = [5, 7.9, "12", True, False, {"x": 1.5}, None, "abc", -2.5, "", [1]]
RETENTION = ("task_id", "status", "delegation_role", "drive_retention_at", "body_present")


@pytest.fixture(autouse=True)
def fresh_connections():
    store.close_connections()  # every connection the test opens takes the meter's row factory
    yield
    store.close_connections()


def _seed(root, n: int) -> list:
    """N tasks: roots and children, project ids, tasks without ``ts``/``updated_at``, cards, handoff rows, drives,
    chat ids."""
    ids = []
    for index in range(n):
        task_id = f"t{index:03d}"
        document = {"task_id": task_id, "status": ("completed", "running", "failed")[index % 3],
                    "title": f"tâche {index} 😀 \x00 \"quoted\"", "result": "x" * (50 + index), "score": 0.1 + index,
                    "project_id": f"p{index % 2}", "chat_id": CHAT_IDS[index % len(CHAT_IDS)]}
        if index % 5:
            document["ts"] = f"2026-10-27T10:{index % 60:02d}:00+00:00"
        if index % 7:
            document["updated_at"] = f"2026-10-27T12:{index * 7 % 60:02d}:00+00:00"
        if index % 4 == 1:
            document.update(parent_task_id=f"t{index - 1:03d}", root_task_id=f"t{index - 1:03d}",
                            delegation_role="subagent")
        if index % 3 == 0:
            document.update(tool_calls=index, last_line=f"line {index}",
                            last_line_at=f"2026-10-27T11:{index % 60:02d}:00Z")
        if index % 2 == 0:
            document["launch_handoffs"] = {f"op{index}a": {"state": "claimed", "n": index},
                                           f"op{index}b": {"state": "returned", "nested": [1, 2.5]}}
        if index % 6 == 0:
            document["child_drive_root"] = f"/drives/{task_id}"
        seed_result(root, task_id, document)
        ids.append(task_id)
    return ids


# ---- the pre-change readers, verbatim in substance (the reference) --------------------------------------------

def _old_documents(root, where="", params=()):
    with store.open_store(root, write=False) as txn:
        clause = f"WHERE {where}" if where else ""
        cards, join = store_task_cards.select(txn)
        rows = txn.conn.execute(f"SELECT b.body, t.*{cards} FROM tasks AS t JOIN task_bodies AS b ON b.task_id = "
                                f"t.task_id{join} {clause} ORDER BY t.task_id", tuple(params)).fetchall()
        records = txn.handoff_records([record["task_id"] for record in rows])
    handoffs = store.decode_handoffs([record["task_id"] for record in rows], records)
    return [(str(record["task_id"]), store_task_cards.project(
        store.project_handoffs(json.loads(record["body"]), handoffs[record["task_id"]]), record)) for record in rows]


def _old_display(root, ids):
    ids = sorted(set(ids))
    with store.open_store(root, write=False) as txn:
        _columns, join = store_task_cards.select(txn)
        view = task_store._card_sql(task_store._display_sql()) if join else task_store._display_sql()
        record = txn.conn.execute(
            f"SELECT json_group_array(json_array(t.task_id, '' || {view}, t.write_seq)) FROM tasks AS t "
            f"JOIN task_bodies AS b ON b.task_id = t.task_id{join} WHERE t.task_id IN (SELECT value FROM json_each(?))",
            (json.dumps(ids),)).fetchone()
        handoff_rows = txn.handoff_records(ids)
    handoffs, held = store.decode_handoffs(ids, handoff_rows), {record["task_id"] for record in handoff_rows}
    found = json.loads(record[0] or "[]")
    return ({task_id: (text if task_id not in held else store.document_text(store.project_handoffs(
        json.loads(text), handoffs[task_id]))) for task_id, text, _seq in found},
        {task_id: int(seq or 0) for task_id, _text, seq in found})


def _old_scalar(root, sql, params=()):
    with store.open_store(root, write=False) as txn:
        return [tuple(record) for record in txn.conn.execute(sql, tuple(params)).fetchall()]


def _old_facts(root, where="", params=()):
    clause = f"WHERE {where}" if where else ""
    facts = {}
    for record in _old_scalar(root, f"SELECT {','.join(task_store.FACT_KEYS)} FROM tasks {clause} ORDER BY task_id",
                              params):
        record = dict(zip(task_store.FACT_KEYS, record))
        row = {key: str(record[key] or "") for key in task_store.FACT_KEYS if key != "presence_authority_recorded"}
        row["presence_authority_recorded"] = bool(record["presence_authority_recorded"])
        facts[row["task_id"]] = row
    return facts


def _old_lineage(root, task_ids):
    select = ("SELECT t.task_id, t.parent_task_id, t.root_task_id, t.delegation_role, "
              "json_extract(b.body, '$.chat_id') FROM tasks t LEFT JOIN task_bodies b ON b.task_id = t.task_id "
              "WHERE t.task_id IN (SELECT value FROM json_each(?))")
    facts = {}
    for record in _old_scalar(root, select, (json.dumps(sorted(task_ids)),)):
        try:
            chat = int(record[4]) if record[4] not in (None, "") else None
        except (TypeError, ValueError):
            chat = None
        facts[str(record[0])] = {"parent_task_id": str(record[1] or ""), "root_task_id": str(record[2] or ""),
                                 "delegation_role": str(record[3] or ""), "chat_id": chat}
    return facts


def _old_list_candidates(root, statuses, limit):
    wanted = sorted(set(statuses))
    bound, params = ("", (*wanted,)) if limit is None else (" LIMIT ?", (*wanted, int(limit)))
    selected = (f"SELECT task_id FROM tasks WHERE list_stable=1 AND status IN ({','.join('?' * len(wanted))}) "
                f"ORDER BY ts DESC, task_id ASC{bound}")
    return [document for _tid, document in _old_documents(
        root, f"t.list_stable IS NULL OR t.list_stable=0 OR t.task_id IN ({selected})", params)]


def _old_retention(root, ids):
    return {row[0]: dict(zip(RETENTION, row)) for row in _old_scalar(
        root, "SELECT t.task_id, t.status, t.delegation_role, t.drive_retention_at, b.task_id IS NOT NULL AS "
        "body_present FROM tasks AS t LEFT JOIN task_bodies AS b ON b.task_id=t.task_id WHERE t.task_id IN "
        f"({','.join('?' * len(ids))})", ids)}


def _old_tree_facts(root, tree_id, *, descendants=False, queue_rows=(), linked=()):
    """#2371's ``task_store.tree_facts`` verbatim but for module prefixes: every statement stepped row by row."""
    from ouroboros.store_hot_facts import VERSION

    tree_sql = task_store._TREE_SQL
    tree_id = str(tree_id)
    with store.open_store(root, write=False) as txn:
        flagged = int(txn.meta("schema_version") or VERSION) >= VERSION
        captured = txn.conn.execute(
            f"SELECT {tree_sql},b.body FROM tasks AS t LEFT JOIN task_bodies AS b ON b.task_id=t.task_id "
            "WHERE t.root_task_id IS NULL" + (" OR (t.root_task_id=? AND t.control_valid=0)" if flagged else ""),
            (tree_id,) if flagged else ()).fetchall()
        selected = {row["task_id"]: dict(row) for row in txn.conn.execute(
            f"SELECT {tree_sql} FROM tasks AS t WHERE t.root_task_id=?", (tree_id,))}
        ties = {*map(str, linked), *(row[0] for row in txn.conn.execute(
            "SELECT task_id FROM task_handoffs WHERE task_id IN (SELECT value FROM json_each(?)) AND "
            "CASE WHEN json_valid(facts) THEN json_extract(facts, '$.root_task_id') END=?",
            (json.dumps([r["task_id"] for r in captured]), tree_id)))}
        if descendants:
            members, rows, faults = task_store._validated(tree_id, selected, captured)
            queued = {}
            for row in queue_rows:
                tid = str(row.get("task_id") or row.get("id") or "")
                if tid:
                    queued.setdefault(tid, row)
            durable_queue = {row["task_id"]: dict(row) for row in txn.conn.execute(
                f"SELECT {tree_sql} FROM tasks AS t "
                "WHERE t.task_id IN (SELECT value FROM json_each(?))", (json.dumps(list(queued)),))}
            queued.update({tid: rows.get(tid, row) for tid, row in durable_queue.items()})
            anchors = {tree_id, *members}
            while True:
                reach = txn.conn.execute(
                    "WITH RECURSIVE reach(id) AS (SELECT value FROM json_each(?) UNION "
                    "SELECT t.task_id FROM tasks AS t JOIN reach ON t.parent_task_id=reach.id) "
                    f"SELECT {tree_sql} FROM tasks AS t "
                    "WHERE t.task_id IN (SELECT id FROM reach)", (json.dumps(sorted(anchors)),)).fetchall()
                members.update({row["task_id"]: rows.get(row["task_id"], dict(row)) for row in reach})
                reached = {tree_id, *members}
                while True:
                    extra = {tid for tid, row in queued.items() if tid not in reached and (
                        str(row.get("root_task_id") or "") == tree_id
                        or str(row.get("parent_task_id") or "") in reached)}
                    if not extra:
                        break
                    reached.update(extra)
                new = reached - anchors - members.keys()
                members.update({tid: row for tid, row in queued.items() if tid in reached and tid not in members})
                if not new:
                    return task_store.TreeCensus(members, task_store._tied(tree_id, members, rows, faults, ties))
                anchors.update(reached)
    members, rows, faults = task_store._validated(tree_id, selected, captured)
    return task_store.TreeCensus(members, task_store._tied(tree_id, members, rows, faults, ties))


# ---- equality --------------------------------------------------------------------------------------------------

def test_task_readers_answer_what_the_pre_change_readers_answered(tmp_path, monkeypatch):
    ids = _seed(tmp_path, 23)
    everything = _old_documents(tmp_path)
    assert task_store.list_documents(tmp_path) == [document for _tid, document in everything]
    assert task_store.list_documents(tmp_path, statuses=["running", "failed"]) == [
        document for _tid, document in _old_documents(tmp_path, "t.status IN (?,?)", ["failed", "running"])]
    assert task_store.documents_by_ids(tmp_path, ids[3:9]) == dict(
        _old_documents(tmp_path, f"t.task_id IN ({','.join('?' * 6)})", ids[3:9]))
    where, params = task_store.lineage_where(ids[0], ids[0], direct_only=False)
    assert task_store.select_lineage(tmp_path, ids[0], ids[0]) == [document for _tid, document in
                                                                  _old_documents(tmp_path, where, params)]
    assert task_store.filtered_list_candidates(tmp_path, ["completed", "failed"], 4) == _old_list_candidates(
        tmp_path, ["completed", "failed"], 4)
    versions = {}
    assert (task_store.display_documents(tmp_path, ids[::2], versions), versions) == _old_display(tmp_path, ids[::2])
    assert task_store.task_facts(tmp_path) == _old_facts(tmp_path)
    assert task_store.task_facts(tmp_path, where="task_id IN (SELECT value FROM json_each(?))",
                                 params=(json.dumps(ids[:7]),)) == _old_facts(
        tmp_path, "task_id IN (SELECT value FROM json_each(?))", (json.dumps(ids[:7]),))
    assert task_store.task_versions(tmp_path, ids) == {
        tid: (int(seq or 0), str(status or ""), bool(owes)) for tid, seq, status, owes in _old_scalar(
            tmp_path, "SELECT task_id, write_seq, status, owes_synthesis FROM tasks")}
    assert task_store.task_drive_status(tmp_path, ids) == {
        str(tid): (str(status or ""), str(drive or "")) for tid, status, drive in _old_scalar(
            tmp_path, "SELECT task_id, status, COALESCE(child_drive_root, headless_child_drive_root, drive_root) "
            "FROM tasks")}
    with store.open_store(tmp_path, write=False) as txn:
        assert task_store.task_ids(tmp_path) == txn.task_ids()
    newest = [tid for (tid,) in _old_scalar(tmp_path, f"SELECT task_id FROM tasks {task_store._NEWEST_ORDER}")]
    assert task_store.newest_ids(tmp_path) == newest and task_store.newest_ids(tmp_path, 5) == newest[:5]
    for project in ("", "p0"):
        index, scope, params = (("tasks_project_root_ts", "project_id=? AND ", (project,)) if project
                                else ("tasks_root_ts", "", ()))
        expected = [tid for (tid,) in _old_scalar(
            tmp_path, f"SELECT task_id FROM tasks INDEXED BY {index} WHERE {scope}{task_store.ROOT_RESULT_WHERE} "
            f"{task_store._NEWEST_ORDER} LIMIT 6", params)]
        assert task_store.newest_root_ids(tmp_path, 6, project_id=project) == expected
    assert task_store.terminal_writes_since(tmp_path, 0) == [tid for (tid,) in _old_scalar(
        tmp_path, "SELECT task_id FROM tasks INDEXED BY tasks_terminal_seq WHERE terminal_seq>? "
        "ORDER BY terminal_seq, task_id", (0,))] != []
    assert projects_registry._read_lineage(tmp_path, ids) == _old_lineage(tmp_path, ids)  # every chat_id shape
    inventory = task_store.inventory(tmp_path)
    assert inventory == [{"name": tid, "write_seq": int(seq or 0), "updated_at": str(updated or "")}
                         for tid, seq, updated in _old_scalar(tmp_path, "SELECT task_id, write_seq, updated_at "
                                                              "FROM tasks ORDER BY updated_at DESC, task_id DESC")]
    assert inventory[-1]["updated_at"] == "" != inventory[0]["updated_at"]  # a NULL ``updated_at`` sorts last

    with store.open_store(tmp_path, write=True) as txn:  # collection columns both ways, a row without its body
        txn.conn.execute("UPDATE tasks SET review_collection_pending = CASE WHEN rowid % 3 = 0 THEN 1 ELSE 0 END, "
                         "plan_collection_pending = CASE WHEN rowid % 2 = 0 THEN NULL ELSE 0 END")
        txn.conn.execute("DELETE FROM task_bodies WHERE task_id = ?", (ids[4],))
    monkeypatch.setattr(task_store, "_COLLECTION_BATCH", 3)
    for column, pending, after in (("review_collection_pending", 1, ""), ("plan_collection_pending", None, ids[2])):
        expected = [tid for (tid,) in _old_scalar(
            tmp_path, f"SELECT task_id FROM tasks WHERE {column} {'IS NULL' if pending is None else '=1'} "
            "AND task_id>? ORDER BY task_id LIMIT ?", (after, 3))]
        assert task_store._collection_ids(tmp_path, column, pending, after) == expected and len(expected) == 3
    asked = [*ids[2:9], "absent"]
    retention = task_store.drive_retention_facts(tmp_path, asked, stop=lambda: True)  # the selection, unclassified
    assert list(retention.items()) == list(_old_retention(tmp_path, asked).items())
    assert retention[ids[4]]["body_present"] == 0 and "absent" not in retention


def test_a_listing_pages_its_bodies_and_halves_a_page_past_sqlites_string_limit(tmp_path, monkeypatch):
    _seed(tmp_path, 9)
    expected = [document for _tid, document in _old_documents(tmp_path)]
    monkeypatch.setattr(task_store, "_PAGE", 4)
    assert task_store.list_documents(tmp_path) == expected
    real = task_store.capture

    def too_big_past_one(conn, sql, params=(), columns=()):
        if "b.body" in sql and len(json.loads(tuple(params)[0])) > 1:
            raise sqlite3.DataError("string or blob too big")
        return real(conn, sql, params, columns)

    monkeypatch.setattr(task_store, "capture", too_big_past_one)
    assert task_store.list_documents(tmp_path) == expected  # every page halved to one document, same snapshot

    def too_big_always(conn, sql, params=(), columns=()):
        if "b.body" in sql:
            raise sqlite3.DataError("string or blob too big")
        return real(conn, sql, params, columns)

    monkeypatch.setattr(task_store, "capture", too_big_always)
    with pytest.raises(store.StoreError):  # a single document past the limit is a typed failure, never a shorter list
        task_store.list_documents(tmp_path)


def test_usage_readers_answer_what_the_in_transaction_readers_answer(data_root):  # noqa: F811 -- the fixture
    _seed_mixed_ledger(data_root)
    with usage_store.read(data_root) as txn:
        for where, params in (("1", ()), (usage_store._OPEN_PREDICATE, ()), ("root_task_id=?", ("root",)),
                              ("task_id=?", ("t2",)), ("root_task_id=? OR task_id=?", ("root", "root"))):
            assert usage_store.decode_attempts(txn.attempt_rows(where, params)) == txn.attempts(where, params), where
        assert txn.open_attempts() == txn.attempts(usage_store._OPEN_PREDICATE) != []
        open_root = txn.attempts(f"{usage_store._OPEN_PREDICATE} AND root_task_id=?", ("root",))
        assert txn.open_attempts(root_task_id="root") == open_root != []
        shapes = ({}, {"include_owners": True}, {"root_task_id": "root"}, {"task_id": "t2"},
                  {"root_task_id": "root2", "task_id": "t2"}, {"root_task_id": "absent"})
        renders = [(shape, breakdown_read(txn, **shape)) for shape in shapes]
        expected = {(index, degraded): breakdown_view(txn, degraded=degraded, **shape)
                    for index, (shape, _render) in enumerate(renders) for degraded in (False, True)}
    for index, (shape, render) in enumerate(renders):  # rendered after the read closed
        for degraded in (False, True):
            assert render(degraded) == expected[(index, degraded)], shape
    assert owner_pause.tree_model_requests(data_root, "root") == open_root  # #2371's census read, captured


QUEUE = [{"task_id": "queued-1", "parent_task_id": "grandchild", "status": "requested"},
         {"id": "queued-2", "parent_task_id": "queued-1", "status": "requested"},
         {"task_id": "child", "parent_task_id": "root-1", "status": "requested"}]  # the durable row wins


def _census_store(root):
    """#2371's ``_tree`` with every damage kind, rows tied each way, a flagged pair, a grandchild, BLOB bodies
    (one spelling its own document, one torn, one whose bytes are no UTF-8) and a handoff that ties beside one
    that cannot."""
    from tests.test_hot_facts import _old_task, _task
    from tests.test_tree_census_damage import DAMAGE, _damage, _tree

    _tree(root)
    for kind in DAMAGE:
        _damage(root, f"stray-{kind}", kind)
    _damage(root, "tied-parent", "torn", parent_task_id="child")
    _damage(root, "tied-handoff", "array")
    _damage(root, "bad-handoff", "null")
    with store.open_store(root, write=True) as txn:
        _task(txn, "grandchild", parent_task_id="child", root_task_id="root-1", status="running")
        _task(txn, "odd", parent_task_id="root-1", root_task_id="root-1", status="")  # flagged, its own body
        _task(txn, "copied", root_task_id="root-1", status="")  # flagged, another task's body
        txn.conn.execute("UPDATE task_bodies SET body=? WHERE task_id='copied'",
                         (store.encode_document({**txn.load("copied"), "task_id": "someone-else"}),))
        _old_task(txn, "blob-own", parent_task_id="root-1", metadata={"root_task_id": "root-1"})
        _old_task(txn, "blob-torn")
        _old_task(txn, "blob-bad")
        txn.conn.execute("UPDATE task_bodies SET body=CAST(body AS BLOB) WHERE task_id='blob-own'")
        txn.conn.execute("UPDATE task_bodies SET body=CAST('{' AS BLOB) WHERE task_id='blob-torn'")
        txn.conn.execute("UPDATE task_bodies SET body=X'FFFE7B' WHERE task_id='blob-bad'")  # not UTF-8 text
        txn.put_handoff("tied-handoff", "launch_handoffs", "op-1",
                        {"tool": "shell", "task_id": "tied-handoff", "root_task_id": "root-1", "state": "claimed"})
        txn.conn.execute("INSERT INTO task_handoffs (task_id, op_id, kind, facts) "
                         "VALUES ('bad-handoff', 'op-1', 'launch_handoffs', '{')")


def test_the_tree_census_answers_what_its_per_row_read_answered(tmp_path):
    """Members and gaps equal #2371's per-row census, in the same order, with and without planner statistics."""
    _census_store(tmp_path)
    for analyzed in (False, True):
        if analyzed:
            with store.open_store(tmp_path, write=True) as txn:
                txn.conn.execute("ANALYZE")
        for tree in ("root-1", "other", "child", "stray-torn"):
            for descendants, queue in ((False, ()), (True, ()), (True, QUEUE)):
                for linked in ((), ("stray-torn", "stray-array")):
                    shape = (analyzed, tree, descendants, bool(queue), linked)
                    new = task_store.tree_facts(tmp_path, tree, descendants=descendants, queue_rows=queue,
                                                linked=linked)
                    old = _old_tree_facts(tmp_path, tree, descendants=descendants, queue_rows=queue, linked=linked)
                    assert list(new.members.items()) == list(old.members.items()), shape
                    assert list(new.unresolved.items()) == list(old.unresolved.items()), shape
    census = task_store.tree_facts(tmp_path, "root-1", descendants=True, queue_rows=QUEUE)  # the cases were real
    assert {"child", "grandchild", "legacy", "odd", "blob-own", "queued-1", "queued-2"} <= set(census.members)
    assert census.members["child"]["status"] == "running" and "copied" not in census.members
    assert census.unresolved["tied-parent"]["linked"] and census.unresolved["tied-handoff"]["linked"]
    assert not census.unresolved["bad-handoff"]["linked"]
    assert census.unresolved["blob-torn"] == census.unresolved["blob-bad"] == {"reason": "body_unreadable",
                                                                                 "linked": False}


def test_the_census_pages_its_underived_bodies_and_halves_a_page_past_sqlites_string_limit(tmp_path, monkeypatch):
    """Every underived body in one aggregate could pass SQLite's string limit (an install still deriving its rows):
    the census captures them by pages of ids inside its one snapshot, a page past the limit halves, and one body
    past it refuses the census typed, never a census without that row."""
    _census_store(tmp_path)
    shapes = (("root-1", False, ()), ("root-1", True, QUEUE), ("other", False, ()))
    expected = [_old_tree_facts(tmp_path, tree, descendants=descendants, queue_rows=queue)
                for tree, descendants, queue in shapes]

    def answered():
        for (tree, descendants, queue), old in zip(shapes, expected):
            new = task_store.tree_facts(tmp_path, tree, descendants=descendants, queue_rows=queue)
            assert list(new.members.items()) == list(old.members.items()), tree
            assert list(new.unresolved.items()) == list(old.unresolved.items()), tree
        return True

    monkeypatch.setattr(task_store, "_PAGE", 2)
    assert answered()
    real = task_store.capture

    def too_big_past_one(conn, sql, params=(), columns=()):  # any aggregate of more than one body is too big
        if "b.body" in sql and ("json_each" not in sql or len(json.loads(tuple(params)[0])) > 1):
            raise sqlite3.DataError("string or blob too big")
        return real(conn, sql, params, columns)

    monkeypatch.setattr(task_store, "capture", too_big_past_one)
    assert answered()  # every page halved to one body, in the same snapshot

    def too_big_always(conn, sql, params=(), columns=()):
        if "b.body" in sql:
            raise sqlite3.DataError("string or blob too big")
        return real(conn, sql, params, columns)

    monkeypatch.setattr(task_store, "capture", too_big_always)
    with pytest.raises(store.StoreError):
        task_store.tree_facts(tmp_path, "root-1")


# ---- O(statements) rows into Python -----------------------------------------------------------------------------

def _crossed(root, call) -> tuple:
    store.close_connections()
    with CostMeter(root) as meter:
        tagged("read", call)
    counts = meter.snapshot("read")["counts"]
    return counts.get("rows_crossed", 0), counts.get("statements", 0)


def test_captured_readers_hand_python_rows_by_statement_not_by_row(tmp_path):
    small, large = tmp_path / "small", tmp_path / "large"
    small_ids, large_ids = _seed(small, 4), _seed(large, 30)
    readers = {
        "list_documents": lambda root, ids: task_store.list_documents(root),
        "documents_by_ids": lambda root, ids: task_store.documents_by_ids(root, ids),
        "display_documents": lambda root, ids: task_store.display_documents(root, ids),
        "task_facts": lambda root, ids: task_store.task_facts(root),
        "task_versions": lambda root, ids: task_store.task_versions(root, ids),
        "task_drive_status": lambda root, ids: task_store.task_drive_status(root, ids),
        "task_ids": lambda root, ids: task_store.task_ids(root),
        "newest_ids": lambda root, ids: task_store.newest_ids(root),
        "terminal_writes_since": lambda root, ids: task_store.terminal_writes_since(root, 0),
        "inventory": lambda root, ids: task_store.inventory(root),
        "collection_ids": lambda root, ids: task_store._collection_ids(root, "review_collection_pending", 0, ""),
        "drive_retention_facts": lambda root, ids: task_store.drive_retention_facts(root, ids, stop=lambda: True),
        "lineage": lambda root, ids: projects_registry._read_lineage(root, ids),
        "tree_facts": lambda root, ids: task_store.tree_facts(root, "t000"),
        "tree_facts(descendants)": lambda root, ids: task_store.tree_facts(root, "t000", descendants=True),
    }
    for name, reader in readers.items():
        (few, few_statements), (many, many_statements) = (
            _crossed(small, lambda: reader(small, small_ids)), _crossed(large, lambda: reader(large, large_ids)))
        assert few == many and many_statements == few_statements and many <= many_statements, (name, few, many)
    old_few = _crossed(small, lambda: _old_documents(small))[0]
    old_many = _crossed(large, lambda: _old_documents(large))[0]
    assert old_many - old_few >= 26  # the negative control: the pre-change listing steps one row per task


def test_the_open_usage_set_crosses_once_per_statement(data_root):  # noqa: F811 -- the fixture
    _seed_mixed_ledger(data_root)

    def captured():
        with usage_store.read(data_root) as txn:
            return txn.attempt_rows(usage_store._OPEN_PREDICATE)

    def stepped():
        with usage_store.read(data_root) as txn:
            return txn.attempts("1")

    crossed, statements = _crossed(data_root, captured)
    assert crossed <= statements and usage_store.decode_attempts(captured()) != []
    assert _crossed(data_root, stepped)[0] > len(usage_store.decode_attempts(captured()))  # every row, one by one


def test_the_few_current_rows_cost_the_same_over_any_history_once_analyzed(tmp_path):
    """The held custody runs and the live tasks are a few rows among the whole closed history: once the
    maintenance child's ``sqlite_analyze`` wrote planner statistics, their reads stay on their indexes and cost the
    same SQLite VM steps over 10 and 300 closed rows of each, a new connection's schema load included. Without it
    SQLite walks every row: before 3.46 to spare sorting the held few in key order (tests/test_cost_contract.py B7 on
    the PERSIST leg's SQLite 3.45.1), on every version for the live tasks and the packer's keep rows, and from 3.46
    for the context's run lines."""
    from ouroboros import delegate_custody as custody, delegate_custody_current as current, evidence_packer, obligations
    from ouroboros import maintenance
    from tests._cost_meter import VM_STEP
    from tests.test_store import publish

    steps = {}
    for history in (10, 300):
        root = tmp_path / f"history-{history}"
        publish(root, *({"task_id": f"done-{n}", "status": "completed", "_schema_version": 1} for n in range(history)),
                *({"task_id": f"live-{n}", "status": "running", "_schema_version": 1} for n in range(3)))
        custody.emit(root, custody.STARTED, {"run_id": "run-template", "task_id": "done-0"})
        custody.emit(root, custody.SETTLED, {"run_id": "run-template", "task_id": "done-0", "state": "succeeded"})
        obligations.remove(root, "delegated_runs", "done-0")  # its closing receipt acknowledged, as the scale root does
        with store.open_store(root, write=True) as txn:
            closed = txn.custody_row("run-template")
            for n in range(history):
                run_id = f"run-closed-{n}"
                txn.put_custody(run_id, {**closed, "run_id": run_id, "task_id": f"done-{n}"}, state="closed")
        for n in range(3):
            custody.emit(root, custody.STARTED, {"run_id": f"run-live-{n}", "task_id": f"live-{n}"})
        assert maintenance.run_job(root, "sqlite_analyze")["state"] == "done"
        store.close_connections()  # the reads below open their connections under the meter
        with CostMeter(root) as meter:
            held = tagged("held", lambda: current.snapshot(root)["state"])
            live = tagged("live", lambda: obligations.members(root, "nonterminal"))
            health = tagged("health", lambda: current.health_runs(root, "done-1"))  # the context's run lines
            keep = tagged("keep", lambda: evidence_packer.keep_rows(root))  # each pack pass
        assert sorted(held) == [f"run-live-{n}" for n in range(3)], sorted(held)
        assert {f"live-{n}" for n in range(3)} <= set(live) and not set(live) & {"done-0", "done-1"}, sorted(live)
        assert sorted(health) == ["run-closed-1", *sorted(held)], sorted(health)
        assert sorted(keep) == [f"live-{n}" for n in range(3)], sorted(keep)
        tags = ("held", "live", "health", "keep")
        steps[history] = {tag: meter.snapshot(tag)["counts"].get("vm_steps", 0) for tag in tags}
    assert all(steps[300][tag] - steps[10][tag] <= VM_STEP for tag in tags), steps
