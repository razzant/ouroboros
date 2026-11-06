"""Current controls are atomic, addressed and independent of settled history."""
from __future__ import annotations

import json

import pytest

from ouroboros import budget_pause, maintenance, maintenance_jobs, owner_pause, store, store_hot_facts as hot
from ouroboros.task_results import write_task_result
from ouroboros.task_store import patch_document
from tests._cases import case_folders


def _task(txn, tid, **fields):
    txn.store_task({"task_id": tid, "status": "running", "_schema_version": 1, **fields})


def _old_task(txn, tid, **fields):
    document = {"task_id": tid, "status": "running", "_schema_version": 1, **fields}
    columns = {key: value for key, value in store.typed_columns(document).items() if key not in hot.TASK_COLUMNS}
    columns["root_task_id"] = document.get("root_task_id")
    columns["write_seq"] = (txn.row(tid, "write_seq") or 0) + 1
    txn.conn.execute("INSERT OR REPLACE INTO task_bodies VALUES (?,?)", (tid, store.encode_document(document)))
    names = ("task_id", *columns)
    txn.conn.execute(f"INSERT OR REPLACE INTO tasks ({','.join(names)}) VALUES ({','.join('?' * len(names))})",
                     (tid, *columns.values()))


def test_control_writers_publish_both_body_and_columns_or_neither(tmp_path, monkeypatch):
    write_task_result(tmp_path, "root", "running")
    fence, _ = owner_pause.install_fence(tmp_path, "root", request_id="press")
    pause = {"state": "pausing", "pause_id": "pause", "task_attempt": 3, "reason": "owner"}
    budget_pause.set_budget_pause(tmp_path, "root", pause)
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.row("root", "owner_pause_state", "owner_pause_fence_id", "budget_pause_state", "budget_pause_attempt") == (
            "requested", fence["fence_id"], "pausing", 3)
        before = txn.load("root")
    with store.open_store(tmp_path, write=True) as txn:
        txn.conn.execute("CREATE TEMP TRIGGER reject_hot BEFORE INSERT ON tasks "
                         "BEGIN SELECT RAISE(ABORT, 'injected write failure'); END")
    with pytest.raises(store.StoreError, match="injected write failure"):
        budget_pause.set_budget_pause(tmp_path, "root", {**pause, "state": "paused"})
    with store.open_store(tmp_path, write=True) as txn:
        assert txn.load("root") == before
        assert txn.row("root", "budget_pause_state") == "pausing"
        txn.conn.execute("DROP TRIGGER reject_hot")
    budget_pause.set_budget_pause(tmp_path, "root", {**pause, "state": "paused"})
    owner_pause.set_fence_state(tmp_path, "root", fence_id=fence["fence_id"], state="paused", finishing_reviews=["review"])
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.load("root")["budget_pause"]["state"] == txn.row("root", "budget_pause_state") == "paused"
    monkeypatch.setattr(store.Txn, "body_text", lambda *_: pytest.fail("derived controls read a task body"))
    assert budget_pause.budget_pause_row(tmp_path, "root") == {**pause, "state": "paused"}
    assert owner_pause.read_fence(tmp_path, "root")["finishing_reviews"] == ["review"]
    owner_pause._CACHE.clear()  # a fresh process has the same body-free answer
    assert owner_pause.read_fence(tmp_path, "root")["state"] == "paused"


def test_null_controls_fall_back_only_for_the_addressed_row(tmp_path, monkeypatch):
    with store.open_store(tmp_path, write=True) as txn:
        _old_task(txn, "legacy", budget_pause={"state": "pausing", "task_attempt": 2},
                  owner_pause={"state": "requested", "fence_id": "f"})
        _task(txn, "modern", result="unrelated body" * 1000)
    seen, real = [], store.Txn.body_text
    monkeypatch.setattr(store.Txn, "body_text", lambda self, tid: seen.append(tid) or real(self, tid))
    assert budget_pause.budget_pause_row(tmp_path, "legacy")["task_attempt"] == 2
    assert owner_pause.read_fence(tmp_path, "modern") == {}
    assert seen == ["legacy"]


def test_control_unknown_and_missing_authority_remain_distinct(tmp_path):
    assert owner_pause.read_fence(tmp_path, "absent") == {}
    with store.open_store(tmp_path, write=True) as txn:
        _task(txn, "bad", status=7, owner_pause={"state": "released"})
        _task(txn, "malformed", owner_pause=["closed"])
        _task(txn, "known")
    for tid in ("bad", "malformed"):
        with pytest.raises(ValueError):
            owner_pause.read_fence(tmp_path, tid)
    assert owner_pause.read_fence(tmp_path, "known") == {}
    with store.open_store(tmp_path, write=True) as txn:
        txn.delete_task("known")
    with pytest.raises(ValueError, match="authority_missing"):
        owner_pause.read_fence(tmp_path, "known")


def test_derivation_job_resumes_and_leaves_body_and_write_sequence_unchanged(tmp_path, monkeypatch):
    with store.open_store(tmp_path, write=True) as txn:
        for tid in ("a", "b", "c"):
            _old_task(txn, tid, metadata={"root_task_id": "tree"},
                      root_phase_checkpoint={"post_task_synthesis": "paused"})
        before = {tid: (txn.body_text(tid), txn.row(tid, "write_seq")) for tid in ("a", "b", "c")}
    progress = []
    monkeypatch.setattr(hot, "_DERIVE_BATCH", 1)  # one row per committed batch: a stop lands between rows
    first = hot.run_hot_facts(tmp_path, progress=progress.append,
                              should_stop=lambda: any(row.get("done") == 1 for row in progress))
    assert first["state"] == "pending" and first["cursor"] == {"table": "tasks", "after": "a"}
    store.close_connections(store.store_path(tmp_path))
    finished = hot.run_hot_facts(tmp_path)
    assert finished["state"] == "done" and finished["done"] == 3
    assert maintenance_jobs.job(tmp_path, hot.JOB) == finished
    with store.open_store(tmp_path, write=False) as txn:
        for tid in before:
            assert (txn.body_text(tid), txn.row(tid, "write_seq")) == before[tid]
            assert txn.row(tid, "root_task_id", "post_task_synthesis") == ("tree", "paused")
    monkeypatch.setattr(store.Txn, "body_text", lambda *_: pytest.fail("finished derivation read a body"))
    assert hot.run_hot_facts(tmp_path) == finished
    assert hot.read(tmp_path, "b")["post_task_synthesis"] == "paused"


def test_derivation_cas_cannot_overwrite_a_racing_writer(tmp_path, monkeypatch):
    with store.open_store(tmp_path, write=True) as txn:
        _old_task(txn, "task", budget_pause={"state": "paused", "task_attempt": 1})
    real, raced = hot.task_columns, []

    def derive(document):
        result = real(document)
        if not raced:
            raced.append(True)
            patch_document(tmp_path, "task", lambda row: {**row, "budget_pause": {"state": "resumed", "task_attempt": 2}})
        return result

    monkeypatch.setattr(hot, "task_columns", derive)
    assert hot.run_hot_facts(tmp_path)["state"] == "done"
    assert budget_pause.budget_pause_row(tmp_path, "task") == {"state": "resumed", "task_attempt": 2}
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.row("task", "budget_pause_state", "budget_pause_attempt") == ("resumed", 2)


def test_derivation_keeps_mismatched_task_identity_unreadable(tmp_path):
    """Another task's body derives nothing, not even its root: the job waits on it as damaged input."""
    with store.open_store(tmp_path, write=True) as txn:
        _old_task(txn, "bad")
        _old_task(txn, "good")
        document = txn.load("bad")
        document["task_id"] = "another"
        txn.conn.execute("UPDATE task_bodies SET body=? WHERE task_id='bad'", (store.encode_document(document),))
    for derive in (False, True):
        if derive:
            with pytest.raises(ValueError, match="identity_mismatch"):
                hot.run_hot_facts(tmp_path)
            assert maintenance_jobs.job(tmp_path, hot.JOB)["waiting_on"]["id"] == "bad"
            with store.open_store(tmp_path, write=False) as txn:
                assert txn.row("bad", "root_task_id", "control_valid") == (None, None)
        with pytest.raises(ValueError, match="authority is unreadable"):
            hot.read(tmp_path, "bad")
        assert hot.read(tmp_path, "good")["control_valid"] == 1


def test_old_custody_upsert_invalidates_columns_in_its_own_transaction(tmp_path):
    with store.open_store(tmp_path, write=True) as txn:
        txn.put_custody("run", {"run_id": "run", "task_id": "task", "root_task_id": "old"}, state="held")
        seq = txn.conn.execute("SELECT write_seq FROM custody_runs").fetchone()[0]
    updated = {"run_id": "run", "task_id": "task", "root_task_id": "new"}
    with store.open_store(tmp_path, write=True) as txn:
        txn.conn.execute("INSERT INTO custody_runs(run_id,facts) VALUES (?,?) "
                         "ON CONFLICT(run_id) DO UPDATE SET facts=excluded.facts", ("run", json.dumps(updated)))
        assert tuple(txn.conn.execute("SELECT root_task_id,facts_derived,write_seq FROM custody_runs").fetchone()) == (
            None, None, seq + 1)
    assert hot.run_hot_facts(tmp_path)["state"] == "done"
    with store.open_store(tmp_path, write=False) as txn:
        assert tuple(txn.conn.execute("SELECT root_task_id,facts_derived,write_seq FROM custody_runs").fetchone()) == (
            "new", 1, seq + 1)


def _schema_five(root, monkeypatch):
    """A real pre-upgrade store, not a new schema with a falsified version stamp."""
    with monkeypatch.context() as patch:
        patch.setattr(store, "_SCHEMA", store._SCHEMA.removesuffix(hot.SCHEMA))
        patch.setattr(store, "SCHEMA_VERSION", 5)
        with store.open_store(root, write=True) as txn:
            _old_task(txn, "legacy", metadata={"root_task_id": "tree"},
                      owner_pause={"state": "paused", "fence_id": "f"})
            txn.conn.execute("INSERT INTO custody_runs(run_id,task_id,facts,state) VALUES (?,?,?,?)",
                             ("old-run", "legacy", json.dumps({"run_id": "old-run", "task_id": "legacy",
                                 "root_task_id": "tree", "output_complete": True}), "closed"))
    store.close_connections(store.store_path(root))


def test_schema_gate_is_atomic_and_never_decodes_old_bodies(tmp_path, monkeypatch):
    from ouroboros import delegate_custody_current as custody
    from ouroboros.store_import import upgrade_current_schema

    _schema_five(tmp_path, monkeypatch)
    assert owner_pause.read_fence(tmp_path, "legacy")["state"] == "paused"
    assert custody.tree_task_ids(tmp_path, "tree") == ["legacy"]
    assert list(custody.health_runs(tmp_path)) == ["old-run"]
    upgrade = hot.upgrade

    def fail_after_ddl(txn):
        upgrade(txn)
        raise RuntimeError("injected schema publication failure")

    with monkeypatch.context() as patch:
        patch.setattr(hot, "upgrade", fail_after_ddl)
        with pytest.raises(RuntimeError, match="schema publication"):
            upgrade_current_schema(tmp_path)
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.meta("schema_version") == 5
        assert "owner_pause_state" not in {row["name"] for row in txn.conn.execute("PRAGMA table_info(tasks)")}
        before = txn.body_text("legacy")
    with monkeypatch.context() as patch:
        patch.setattr(store.Txn, "body_text", lambda *_: pytest.fail("schema upgrade decoded a task body"))
        upgrade_current_schema(tmp_path)
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.meta("schema_version") == store.SCHEMA_VERSION >= hot.VERSION
        assert txn.row("legacy", "owner_pause_state", "root_task_id") == (None, None)
        assert txn.body_text("legacy") == before
        assert txn.meta("job:" + hot.JOB)["state"] == "pending"
    assert hot.run_hot_facts(tmp_path)["state"] == "done"
    assert owner_pause.read_fence(tmp_path, "legacy")["state"] == "paused"
    assert custody.tree_task_ids(tmp_path, "tree") == ["legacy"]


def test_first_custody_publication_upgrades_an_old_child_store(tmp_path, monkeypatch):
    from ouroboros import delegate_custody as custody

    _schema_five(tmp_path, monkeypatch)
    assert custody.emit(tmp_path, custody.STARTED, {"run_id": "new-run", "task_id": "child", "root_task_id": "tree"})
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.meta("schema_version") == store.SCHEMA_VERSION >= hot.VERSION
        assert tuple(txn.conn.execute("SELECT root_task_id,facts_derived FROM custody_runs WHERE run_id='new-run'").fetchone()) == ("tree", 1)
        assert txn.row("legacy", "owner_pause_state") is None


def test_request_backfill_and_old_writer_invalidation_are_source_cas(tmp_path, monkeypatch):
    for name in hot.REQUEST_SETS:
        with store.open_store(tmp_path, write=True) as txn:
            txn.conn.execute("INSERT INTO obligations(set_name,id,facts) VALUES (?,?,?)",
                             (name, "invocation:old", json.dumps({"request": {"task_id": "child", "root_task_id": "old"}})))
    real, raced = hot.request_columns, []

    def derive(facts):
        values = real(facts)
        if not raced:
            raced.append(True)
            with store.open_store(tmp_path, write=True) as txn:
                txn.conn.execute("UPDATE obligations SET facts=? WHERE set_name='custody_open'",
                                 (json.dumps({"request": {"task_id": "child", "root_task_id": "new"}}),))
        return values

    monkeypatch.setattr(hot, "request_columns", derive)
    assert hot.run_hot_facts(tmp_path)["state"] == "done"
    with store.open_store(tmp_path, write=False) as txn:
        assert [(name, key) for name, key, _facts in hot.tree_requests(txn, "new", refused=True)] == [
            ("custody_open", "invocation:old")]
        assert [(name, key) for name, key, _facts in hot.tree_requests(txn, "old", refused=True)] == [
            ("custody_refused", "invocation:old")]
    with store.open_store(tmp_path, write=True) as txn:
        txn.conn.execute("UPDATE obligations SET facts=? WHERE set_name='custody_open'",
                         (json.dumps({"request": {"task_id": "child", "root_task_id": "third"}}),))
        assert txn.conn.execute("SELECT request_root_task_id FROM obligations WHERE set_name='custody_open'").fetchone()[0] is None
    assert hot.run_hot_facts(tmp_path)["state"] == "done"
    with store.open_store(tmp_path, write=False) as txn:
        assert len(hot.tree_requests(txn, "third")) == 1


def test_damaged_job_input_waits_for_its_own_change(tmp_path, monkeypatch):
    with store.open_store(tmp_path, write=True) as txn:
        _old_task(txn, "bad")
        txn.conn.execute("UPDATE task_bodies SET body=? WHERE task_id='bad'", ("{",))
    with pytest.raises(ValueError):
        hot.run_hot_facts(tmp_path)
    failed = maintenance_jobs.job(tmp_path, hot.JOB)
    assert failed["state"] == "failed" and failed["waiting_on"]["id"] == "bad"
    with monkeypatch.context() as patch:
        patch.setattr(store.Txn, "body_text", lambda *_: pytest.fail("unchanged damaged input was retried"))
        assert hot.run_hot_facts(tmp_path) == failed
    with store.open_store(tmp_path, write=True) as txn:
        _task(txn, "bad")
    assert hot.run_hot_facts(tmp_path)["state"] == "done"


def test_one_damaged_row_never_starves_the_valid_rows_after_it(tmp_path, monkeypatch):
    """The pass names the damaged row and still derives every independent row after it (across batches),
    then fails; it rests while exactly that row is unchanged, and an older writer's new row beside it is
    derived on the next pass. Damage never becomes a known fact: the damaged row stays underived."""
    with store.open_store(tmp_path, write=True) as txn:
        _old_task(txn, "a-bad")
        txn.conn.execute("UPDATE task_bodies SET body=? WHERE task_id='a-bad'", ("{",))
        for index in range(hot._DERIVE_BATCH * 2 + 3):
            _old_task(txn, f"legacy-{index:03d}", origin_message_ref={"chat_id": 7, "client_message_id": f"m{index}"})
        for run_id, facts in (("bad-run", "{"), ("old-run", json.dumps({"run_id": "old-run", "root_task_id": "tree"}))):
            txn.conn.execute("INSERT INTO custody_runs(run_id,task_id,facts,state) VALUES (?,?,?,?)",
                             (run_id, "legacy-000", facts, "closed"))  # a later family's legacy rows
        txn.conn.execute("INSERT INTO obligations(set_name,id,facts) VALUES (?,?,?)",
                         ("custody_open", "invocation:old", json.dumps({"request": {"root_task_id": "tree"}})))
    with pytest.raises(ValueError, match="task:a-bad"):
        hot.run_hot_facts(tmp_path)
    with store.open_store(tmp_path, write=False) as txn:
        assert [row[0] for row in txn.conn.execute(f"SELECT task_id FROM tasks WHERE {hot.TASK_UNDERIVED}")] == ["a-bad"]
        assert (txn.row("legacy-000", "origin"), txn.row("a-bad", "origin")) == ("7:m0", None)
        assert [tuple(row) for row in txn.conn.execute("SELECT run_id, root_task_id, facts_derived FROM custody_runs "
                                                       "ORDER BY run_id")] == [("bad-run", None, None), ("old-run", "tree", 1)]
        assert txn.conn.execute("SELECT request_root_task_id FROM obligations WHERE id='invocation:old'").fetchone()[0] == "tree"
    failed = maintenance_jobs.job(tmp_path, hot.JOB)
    assert failed["state"] == "failed" and failed["waiting_on"]["id"] == "a-bad"
    assert [(item["table"], item["id"], item["reason"]) for item in failed["damaged"]] == [
        ("tasks", "a-bad", "body_unreadable"), ("custody_runs", "bad-run", "facts_unreadable")]
    with monkeypatch.context() as patch:
        patch.setattr(store.Txn, "body_text", lambda *_: pytest.fail("unchanged damaged input was retried"))
        assert hot.run_hot_facts(tmp_path) == failed
    with store.open_store(tmp_path, write=True) as txn:
        _old_task(txn, "z-late")
    with pytest.raises(ValueError, match="task:a-bad"):
        hot.run_hot_facts(tmp_path)
    with store.open_store(tmp_path, write=False) as txn:
        assert (txn.row("z-late", "origin"), txn.row("a-bad", "origin")) == ("", None)
    with store.open_store(tmp_path, write=True) as txn:
        _task(txn, "a-bad")
        txn.put_custody("bad-run", {"run_id": "bad-run", "task_id": "legacy-000"}, state="closed")
    assert hot.run_hot_facts(tmp_path)["state"] == "done"
    assert maintenance_jobs.job(tmp_path, hot.JOB)["damaged"] is None


# ---- the dispatcher's look and the child's step over damaged input: maintenance.due -> run_job -----

# The stored bytes of each family's damaged row, distinct so a decode names the row it read.
_BAD = {"tasks": "{task", "custody_runs": "{run", "custody_open": "{open", "custody_refused": "{refused"}
_NAMED = [("tasks", "a-bad", "body_unreadable"), ("custody_runs", "a-bad", "facts_unreadable"),
          ("custody_open", "a-bad", "facts_unreadable"), ("custody_refused", "a-bad", "facts_unreadable")]


def _request(key, facts):
    return lambda txn, name: txn.conn.execute("INSERT INTO obligations(set_name,id,facts) VALUES (?,?,?)", (name, key, facts))


def _damage_every_family(root):
    with store.open_store(root, write=True) as txn:
        _old_task(txn, "a-bad")
        txn.conn.execute("UPDATE task_bodies SET body=? WHERE task_id='a-bad'", (_BAD["tasks"],))
        txn.conn.execute("INSERT INTO custody_runs(run_id,task_id,facts,state) VALUES ('a-bad','task',?,'closed')",
                         (_BAD["custody_runs"],))
        for name in hot.REQUEST_SETS:
            _request("a-bad", _BAD[name])(txn, name)


def _new_task_older_writer(txn):
    _old_task(txn, "z-new")


def _new_task_origin_only(txn):  # a row an upgraded store derived before the origin column
    _task(txn, "z-new")
    txn.conn.execute("UPDATE tasks SET origin=NULL WHERE task_id='z-new'")


def _new_task_chat_only(txn):  # a row an upgraded store derived before the chat column
    _task(txn, "z-new", chat_id=4242)
    txn.conn.execute("UPDATE tasks SET chat_id=NULL WHERE task_id='z-new'")


def _new_custody(txn):
    txn.conn.execute("INSERT INTO custody_runs(run_id,task_id,facts,state) VALUES ('z-new','task',?,'closed')",
                     (json.dumps({"run_id": "z-new", "root_task_id": "tree"}),))


_NEW_ROW = {"task_older_writer": (_new_task_older_writer, "SELECT origin IS NOT NULL AND chat_id IS NOT NULL AND "
                                                          "owner_pause_state IS NOT NULL FROM tasks WHERE task_id='z-new'"),
            "task_origin_only": (_new_task_origin_only, "SELECT origin IS NOT NULL FROM tasks WHERE task_id='z-new'"),
            "task_chat_only": (_new_task_chat_only, "SELECT chat_id='4242' FROM tasks WHERE task_id='z-new'"),
            "custody_runs": (_new_custody, "SELECT facts_derived=1 FROM custody_runs WHERE run_id='z-new'"),
            **{name: ((lambda name: lambda txn: _request("z-new", json.dumps({"request": {"root_task_id": "tree"}}))(txn, name))(name),
                      f"SELECT request_root_task_id='tree' FROM obligations WHERE set_name='{name}' AND id='z-new'")
               for name in hot.REQUEST_SETS}}


def _server_looks(root):
    return "hot_facts" in maintenance.due(root)


def _job(root):
    return maintenance_jobs.job(root, hot.JOB)


def _revision(root):
    with store.open_store(root, write=False) as txn:
        return txn.revision()


def _named(root):
    return [(item["table"], item["id"], item["reason"]) for item in _job(root)["damaged"] or ()]


@pytest.fixture
def decoded(monkeypatch):
    """Every damaged stored text a JSON decode was handed, in this process."""
    seen, real = [], json.loads

    def loads(raw, *args, **kwargs):
        if raw in _BAD.values():
            seen.append(raw)
        return real(raw, *args, **kwargs)

    monkeypatch.setattr(json, "loads", loads)
    return seen


@pytest.fixture
def named_damage(tmp_path, monkeypatch):
    """A root whose job's finished pass named one damaged row per family."""
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    _damage_every_family(tmp_path)
    assert _server_looks(tmp_path)
    with pytest.raises(ValueError, match="task:a-bad"):
        maintenance.run_job(tmp_path, "hot_facts")
    assert _job(tmp_path)["state"] == "failed" and _named(tmp_path) == _NAMED
    return tmp_path


def test_unchanged_named_damage_is_no_work_for_either_look(named_damage, decoded, monkeypatch):
    before, row = _revision(named_damage), _job(named_damage)
    decoded.clear()
    assert not _server_looks(named_damage)
    assert maintenance.run_job(named_damage, "hot_facts") is None
    assert hot.run_hot_facts(named_damage) == row
    assert decoded == [] and _revision(named_damage) == before and _job(named_damage) == row


@pytest.mark.parametrize("family", list(_NEW_ROW))
def test_a_new_underived_row_beside_named_damage_wakes_the_server_look(named_damage, family):
    with store.open_store(named_damage, write=True) as txn:
        _NEW_ROW[family][0](txn)
    assert _server_looks(named_damage), "the row waits for the child to start for another reason"


@pytest.mark.parametrize("family", list(_NEW_ROW))
def test_the_pass_a_new_row_starts_derives_it_without_reading_named_damage_again(named_damage, decoded, family):
    add, derived = _NEW_ROW[family]
    with store.open_store(named_damage, write=True) as txn:
        add(txn)
    decoded.clear()
    with pytest.raises(ValueError, match="task:a-bad"):  # the failure stays loud, with the same names
        maintenance.run_job(named_damage, "hot_facts")
    assert decoded == [], "unchanged damaged rows were decoded again"
    with store.open_store(named_damage, write=False) as txn:
        assert txn.conn.execute(derived).fetchone()[0] == 1
    assert _named(named_damage) == _NAMED and _job(named_damage)["state"] == "failed"
    assert not _server_looks(named_damage) and maintenance.run_job(named_damage, "hot_facts") is None


def _repair(txn, table, way):
    if table == "tasks":
        _task(txn, "a-bad") if way == "repaired" else txn.delete_task("a-bad")
    elif table == "custody_runs":
        if way == "repaired":
            txn.put_custody("a-bad", {"run_id": "a-bad", "task_id": "task"}, state="closed")
        else:
            txn.conn.execute("DELETE FROM custody_runs WHERE run_id='a-bad'")
    elif way == "repaired":
        txn.conn.execute("UPDATE obligations SET facts=? WHERE set_name=? AND id='a-bad'",
                         (json.dumps({"request": {"root_task_id": "tree"}}), table))
    else:
        txn.conn.execute("DELETE FROM obligations WHERE set_name=? AND id='a-bad'", (table,))


@pytest.mark.parametrize("way", ["repaired", "removed"])
def test_repaired_or_removed_damage_is_work_once_and_ends_in_done(named_damage, decoded, way):
    remaining = list(_NAMED)
    for table, _key, _reason in _NAMED:
        with store.open_store(named_damage, write=True) as txn:
            _repair(txn, table, way)
        remaining = [entry for entry in remaining if entry[0] != table]
        assert _server_looks(named_damage), table
        decoded.clear()
        if remaining:
            with pytest.raises(ValueError):
                maintenance.run_job(named_damage, "hot_facts")
            assert _named(named_damage) == remaining and decoded == []
            assert not _server_looks(named_damage)
    row = maintenance.run_job(named_damage, "hot_facts")
    assert row["state"] == "done" and decoded == []
    assert _job(named_damage)["damaged"] is None and not _job(named_damage).get("damage_only")
    assert not _server_looks(named_damage) and maintenance.run_job(named_damage, "hot_facts") is None


@pytest.mark.parametrize("facts", ["[1]", "null", "3"])
def test_custody_and_request_facts_that_are_no_object_wait_by_name_beside_later_rows(tmp_path, monkeypatch, facts):
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    with store.open_store(tmp_path, write=True) as txn:
        txn.conn.execute("INSERT INTO custody_runs(run_id,task_id,facts,state) VALUES ('a-odd','task',?,'closed')", (facts,))
        _new_custody(txn)
        for name in hot.REQUEST_SETS:
            _request("a-odd", facts)(txn, name)
            _request("z-new", json.dumps({"request": {"root_task_id": "tree"}}))(txn, name)
    with pytest.raises(ValueError, match="custody_runs row facts_not_object: a-odd"):
        maintenance.run_job(tmp_path, "hot_facts")
    assert _named(tmp_path) == [(table, "a-odd", "facts_not_object") for table in ("custody_runs", *hot.REQUEST_SETS)]
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.conn.execute(_NEW_ROW["custody_runs"][1]).fetchone()[0] == 1
        for name in hot.REQUEST_SETS:
            assert txn.conn.execute(_NEW_ROW[name][1]).fetchone()[0] == 1
    assert not _server_looks(tmp_path) and maintenance.run_job(tmp_path, "hot_facts") is None


@pytest.mark.parametrize("status", [[], {"state": "running"}])
def test_a_task_status_the_writer_refuses_waits_by_name_beside_later_rows(tmp_path, monkeypatch, status):
    """Valid JSON with its own id and an array or object status, which ``task_columns`` cannot classify:
    named damage, the rows after it derive, the look rests while it is unchanged and wakes on new input."""
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    with store.open_store(tmp_path, write=True) as txn:
        with pytest.raises(TypeError):  # writer acceptance is unchanged
            _task(txn, "refused", status=status)
        assert txn.row("refused", "task_id") is None
        _old_task(txn, "a-odd")
        txn.conn.execute("UPDATE task_bodies SET body=? WHERE task_id='a-odd'",
                         (json.dumps({"task_id": "a-odd", "status": status, "_schema_version": 1}),))
        _old_task(txn, "b-legacy")
    with pytest.raises(ValueError, match="task body status_unreadable: task:a-odd"):
        maintenance.run_job(tmp_path, "hot_facts")
    assert _named(tmp_path) == [("tasks", "a-odd", "status_unreadable")] and _job(tmp_path)["damage_only"]
    with store.open_store(tmp_path, write=False) as txn:
        assert (txn.row("b-legacy", "control_valid"), txn.row("a-odd", "control_valid")) == (1, None)
        assert hot._read_rows(txn, ["a-odd"])["a-odd"]["control_valid"] == 0  # a passive read: that row unknown
    assert not _server_looks(tmp_path) and maintenance.run_job(tmp_path, "hot_facts") is None
    with store.open_store(tmp_path, write=True) as txn:
        _new_task_older_writer(txn)
    assert _server_looks(tmp_path)
    with pytest.raises(ValueError, match="task:a-odd"):
        maintenance.run_job(tmp_path, "hot_facts")
    with store.open_store(tmp_path, write=True) as txn:
        _task(txn, "a-odd")  # its own writer repairs it
    assert _server_looks(tmp_path) and maintenance.run_job(tmp_path, "hot_facts")["state"] == "done"


def test_a_named_row_rewritten_still_damaged_is_read_once_and_named_anew(named_damage, decoded):
    with store.open_store(named_damage, write=True) as txn:
        txn.conn.execute("UPDATE tasks SET write_seq=write_seq+1 WHERE task_id='a-bad'")
    assert _server_looks(named_damage)
    decoded.clear()
    with pytest.raises(ValueError, match="task:a-bad"):
        maintenance.run_job(named_damage, "hot_facts")
    assert decoded == [_BAD["tasks"]], "only the changed row is read again"
    assert not _server_looks(named_damage)


def test_an_interrupted_pass_keeps_its_names_and_resumes_past_them(tmp_path, monkeypatch, decoded):
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(hot, "_DERIVE_BATCH", 1)
    _damage_every_family(tmp_path)
    with store.open_store(tmp_path, write=True) as txn:
        for tid in ("b", "c", "d"):
            _old_task(txn, tid)
    calls = []

    def stop_after_the_second_row():
        calls.append(1)
        return len(calls) > 4  # the checks around a-bad and b have passed: a stop lands on a committed cursor

    row = maintenance.run_job(tmp_path, "hot_facts", should_stop=stop_after_the_second_row)
    assert row["state"] == "pending" and _named(tmp_path)[:1] == _NAMED[:1]
    assert _server_looks(tmp_path), "an unfinished pass stays due"
    decoded.clear()
    with pytest.raises(ValueError, match="task:a-bad"):
        maintenance.run_job(tmp_path, "hot_facts")
    assert decoded.count(_BAD["tasks"]) == 0 and _named(tmp_path) == _NAMED
    with store.open_store(tmp_path, write=False) as txn:
        assert [row[0] for row in txn.conn.execute("SELECT task_id FROM tasks WHERE origin IS NULL")] == ["a-bad"]
    assert not _server_looks(tmp_path)


def test_a_hard_stop_mid_pass_leaves_the_pass_due_and_its_names_kept(named_damage, decoded, monkeypatch):
    class Killed(BaseException):
        """A stop no handler of the job's row loop answers."""

    with store.open_store(named_damage, write=True) as txn:
        _old_task(txn, "z-new")
    real = hot.task_columns

    with monkeypatch.context() as patch:
        patch.setattr(hot, "task_columns", lambda _document: (_ for _ in ()).throw(Killed()))
        with pytest.raises(Killed):
            maintenance.run_job(named_damage, "hot_facts")
    assert hot.task_columns is real and _job(named_damage)["state"] == "pending" and _named(named_damage) == _NAMED
    assert _server_looks(named_damage)
    decoded.clear()
    with pytest.raises(ValueError):
        maintenance.run_job(named_damage, "hot_facts")
    assert decoded == [] and not _server_looks(named_damage)


@pytest.mark.parametrize("rewrite", ["still_damaged", "valid"])
def test_a_writer_racing_the_pass_over_a_named_row_wins_and_the_next_pass_settles_it(named_damage, decoded, monkeypatch, rewrite):
    with store.open_store(named_damage, write=True) as txn:
        _old_task(txn, "z-new")
    real, raced = hot.task_columns, []

    def derive(document):
        if not raced:
            raced.append(True)
            with store.open_store(named_damage, write=True) as txn:
                if rewrite == "valid":
                    _old_task(txn, "a-bad")
                else:
                    txn.conn.execute("UPDATE tasks SET write_seq=write_seq+1 WHERE task_id='a-bad'")
        return real(document)

    monkeypatch.setattr(hot, "task_columns", derive)
    row = maintenance.run_job(named_damage, "hot_facts")
    assert row["state"] == "pending" and _server_looks(named_damage), "the changed row is not left behind"
    assert ("tasks", "a-bad", "body_unreadable") not in _named(named_damage)
    decoded.clear()
    with pytest.raises(ValueError):
        maintenance.run_job(named_damage, "hot_facts")
    assert ("tasks" in {entry[0] for entry in _named(named_damage)}) is (rewrite == "still_damaged")
    assert decoded == ([_BAD["tasks"]] if rewrite == "still_damaged" else [])
    assert not _server_looks(named_damage)


def test_a_held_store_mid_pass_is_no_failure_and_stays_due(tmp_path, monkeypatch, decoded):
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    _damage_every_family(tmp_path)
    with store.open_store(tmp_path, write=True) as txn:
        _old_task(txn, "b")
    real, held = hot.task_columns, []

    def derive(document):
        if not held:
            held.append(True)
            raise store.StoreBusy("held", reason="contention")
        return real(document)

    monkeypatch.setattr(hot, "task_columns", derive)
    with pytest.raises(store.StoreBusy):
        maintenance.run_job(tmp_path, "hot_facts")
    assert _job(tmp_path)["state"] == "pending", "a held store parked the job as failed"
    assert _server_looks(tmp_path)
    decoded.clear()
    with pytest.raises(ValueError, match="task:a-bad"):
        maintenance.run_job(tmp_path, "hot_facts")
    assert decoded.count(_BAD["tasks"]) == 0 and not _server_looks(tmp_path)


def test_a_derivation_fault_stays_an_error_spawns_nothing_and_runs_with_the_child(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    with store.open_store(tmp_path, write=True) as txn:
        for tid in ("a", "b"):
            _old_task(txn, tid)
    with monkeypatch.context() as patch:
        patch.setattr("ouroboros.task_status.stored_list_order_is_final", lambda _document: 1 / 0)
        with pytest.raises(ZeroDivisionError):
            maintenance.run_job(tmp_path, "hot_facts")
        row = _job(tmp_path)
        assert row["state"] == "failed" and row["waiting_on"]["id"] == "a" and not row.get("damage_only")
        assert not _server_looks(tmp_path), "a repeated fault would spawn the child on every look"
        with pytest.raises(ZeroDivisionError):
            maintenance.run_job(tmp_path, "hot_facts")  # the child, running for another reason, tries again
    assert maintenance.run_job(tmp_path, "hot_facts")["state"] == "done"
    assert not _server_looks(tmp_path)


def test_the_childs_report_and_row_keep_the_failure_visible_at_rest(named_damage, monkeypatch):
    monkeypatch.setattr(maintenance, "JOBS", {"hot_facts": maintenance.JOBS["hot_facts"]})
    with store.open_store(named_damage, write=True) as txn:
        _old_task(txn, "z-new")
    report = maintenance.run(named_damage)
    assert report["status"] == "failed" and report["jobs"]["hot_facts"]["state"] == "failed"
    row = _job(named_damage)
    assert row["last_error"]["class"] == "ValueError" and "task:a-bad" in row["error"]
    assert maintenance.run(named_damage) == {"status": "completed", "jobs": {}}
    assert _job(named_damage) == row, "a rest neither clears the failure nor rewrites the row"


def test_the_look_reads_no_body_and_costs_the_same_beside_any_number_of_derived_rows(tmp_path, monkeypatch):
    import sqlite3

    from tests._cost_meter import CostMeter, tagged

    reads, counts = [], {}

    def authorizer(action, table, _column, _db, _source):
        if action == sqlite3.SQLITE_READ and table == "task_bodies":
            reads.append(table)
        return sqlite3.SQLITE_OK

    for size in (2, 200):
        root = tmp_path / str(size)
        root.mkdir()
        monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
        with store.open_store(root, write=True) as txn:
            for index in range(size):
                _task(txn, f"derived-{index:04d}", result="r" * 4000)
        _damage_every_family(root)
        with pytest.raises(ValueError):
            maintenance.run_job(root, "hot_facts")
        row = _job(root)
        store.close_connections()
        with CostMeter(root) as meter:
            assert tagged("look", lambda: hot.hot_facts_due(root, row, False)) is False
        counts[size] = meter.snapshot("look")["counts"]
        with store.open_store(root, write=False) as txn:
            txn.conn.set_authorizer(authorizer)
            try:
                assert hot._settled(txn, hot._named_damage(row))
            finally:
                txn.conn.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)
    assert reads == []
    assert counts[2].get("statements") == counts[200].get("statements")
    assert counts[200].get("vm_steps", 0) <= counts[2].get("vm_steps", 0) + 200, (counts[2], counts[200])


# ---- schema readers across the origin step, in both directions -------------------------------------

_SCHEMA_NINE_UNDERIVED = (
    "owner_pause_state IS NULL OR owner_pause_fence_id IS NULL OR owner_pause_facts IS NULL OR budget_pause_state IS NULL "
    "OR budget_pause_attempt IS NULL OR budget_pause_facts IS NULL OR post_task_synthesis IS NULL OR owner_wait_state "
    "IS NULL OR owner_wait_reason IS NULL OR control_valid IS NULL OR list_stable IS NULL OR root_task_id IS NULL")


def schema_nine(root):
    """The store an older body left: no origin or chat column or index, its index predicate and version."""
    with store.open_store(root, write=True) as txn:
        for index in ("tasks_origin", "tasks_underived_origin", "tasks_underived_chat", "tasks_underived_hot"):
            txn.conn.execute(f"DROP INDEX IF EXISTS {index}")
        txn.conn.execute("ALTER TABLE tasks DROP COLUMN origin")
        txn.conn.execute("ALTER TABLE tasks DROP COLUMN chat_id")
        txn.conn.execute(f"CREATE INDEX tasks_underived_hot ON tasks(task_id) WHERE {_SCHEMA_NINE_UNDERIVED}")
        txn.conn.execute("UPDATE meta SET value='9' WHERE key='schema_version'")
    store.close_connections(store.store_path(root))


def test_a_store_before_the_origin_step_is_read_owed_and_upgraded_without_a_body_decode(tmp_path, monkeypatch):
    from ouroboros import task_store
    from ouroboros.store_import import upgrade_current_schema

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    ref = {"chat_id": 7, "client_message_id": "m1", "ts": "t", "text_sha256": "0" * 64}
    write_task_result(tmp_path, "legacy", "completed", result="kept", origin_message_ref=ref)
    write_task_result(tmp_path, "other", "completed", result="kept")
    schema_nine(tmp_path)
    assert hot.read(tmp_path, "legacy")["status"] == "completed"
    assert sorted(task_store.origin_documents(tmp_path, "7:m1")[0]) == ["legacy"]
    assert _server_looks(tmp_path), "the look must say owed: the runner upgrades first"
    with store.open_store(tmp_path, write=False) as txn:  # every reader above stayed passive
        assert txn.meta("schema_version") == 9
        assert "origin" not in {row["name"] for row in txn.conn.execute("PRAGMA table_info(tasks)")}
    with monkeypatch.context() as patch:
        patch.setattr(store.Txn, "body_text", lambda *_: pytest.fail("the schema step decoded a body"))
        upgrade_current_schema(tmp_path)
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.meta("schema_version") == store.SCHEMA_VERSION
        assert txn.row("legacy", "origin") is None and txn.meta("job:" + hot.JOB)["state"] == "pending"
    assert sorted(task_store.origin_documents(tmp_path, "7:m1")[0]) == ["legacy"]
    assert _server_looks(tmp_path)
    assert maintenance.run_job(tmp_path, "hot_facts")["state"] == "done"
    with store.open_store(tmp_path, write=False) as txn:
        assert (txn.row("legacy", "origin"), txn.row("other", "origin")) == ("7:m1", "")
    assert sorted(task_store.origin_documents(tmp_path, "7:m1")[0]) == ["legacy"]
    assert not _server_looks(tmp_path) and maintenance.run_job(tmp_path, "hot_facts") is None


def test_an_older_bodys_probes_still_answer_on_a_store_past_the_origin_step(tmp_path):
    """A rollback shares this store: the older dispatcher's and job's ``INDEXED BY`` probes name its own
    predicate, so the origin step leaves that index's definition alone (no query solution otherwise, and
    the older body's whole maintenance look would fail on every tick)."""
    assert hot.TASK_UNDERIVED == _SCHEMA_NINE_UNDERIVED
    with store.open_store(tmp_path, write=True) as txn:
        _old_task(txn, "older")
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.conn.execute(f"SELECT 1 FROM tasks INDEXED BY tasks_underived_hot WHERE {_SCHEMA_NINE_UNDERIVED} "
                                "LIMIT 1").fetchone() is not None
        assert [row[0] for row in txn.conn.execute(
            f"SELECT task_id FROM tasks INDEXED BY tasks_underived_hot WHERE ({_SCHEMA_NINE_UNDERIVED}) AND task_id>? "
            "ORDER BY task_id LIMIT ?", ("", 64))] == ["older"]


def test_derived_rows_read_no_body_and_underived_rows_still_do(tmp_path):
    """Proved by SQLite's authorizer, not by patching ``Txn.body_text``: the SQL itself joins no body."""
    import sqlite3

    with store.open_store(tmp_path, write=True) as txn:
        _task(txn, "modern", result="unrelated body" * 1000)
        _old_task(txn, "legacy", budget_pause={"state": "pausing", "task_attempt": 2})
        _old_task(txn, "torn")
        txn.conn.execute("UPDATE task_bodies SET body='{' WHERE task_id='torn'")
        txn.conn.execute("UPDATE task_bodies SET body='{' WHERE task_id='modern'")  # a corrupt body behind derived columns
    reads = []

    def authorizer(action, table, column, _db, _source):
        if action == sqlite3.SQLITE_READ and table == "task_bodies":
            reads.append(column)
        return sqlite3.SQLITE_OK

    with store.open_store(tmp_path, write=False) as txn:
        txn.conn.set_authorizer(authorizer)
        try:
            rows = hot._read_rows(txn, ["modern"])
            assert rows["modern"]["control_valid"] == 1 and rows["modern"]["budget_pause_state"] == ""
            assert reads == [], "a derived row's read touched task_bodies"
            rows = hot._read_rows(txn, ["legacy", "torn"])
        finally:  # ``set_authorizer(None)`` denies everything on Python 3.10: allow all instead (the fixture closes the handle)
            txn.conn.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)
    assert "body" in reads, "an underived row must still pay its strict body read"
    assert (rows["legacy"]["control_valid"], rows["legacy"]["budget_pause_attempt"]) == (1, 2)
    assert rows["torn"]["control_valid"] == 0  # an unparsable legacy body stays unknown, for that row only
    assert hot.read(tmp_path, "modern")["control_valid"] == 1
    with pytest.raises(ValueError, match="unreadable"):
        hot.read(tmp_path, "torn")


_HISTORY = {"version": 2, "fanout_count": 2, "requested_count": 2, "fanout_interval_sec_total": 0,
            "child_ids": ["x", "y"], "lanes_requested": []}
# Stored bodies that are not this task's document: a torn write, other JSON, another task's row.
_NOT_THIS_TASK = {"torn": "{", "array": "[]", "null": "null", "empty": "{}", "foreign": json.dumps({
    "task_id": "other", "status": "running", "swarm_fanout_rollup": _HISTORY, "swarm_fanout_basis": {"base": _HISTORY}})}


def test_a_body_that_is_not_this_task_leaves_only_its_own_row_unknown(subtests, tmp_path):
    case_paths = case_folders(tmp_path)
    for kind in list(_NOT_THIS_TASK):
        tmp_path = next(case_paths)
        with subtests.test(kind=kind):
            with store.open_store(tmp_path, write=True) as txn:
                _old_task(txn, "bad")
                _old_task(txn, "legacy", budget_pause={"state": "pausing", "task_attempt": 2})
                txn.conn.execute("UPDATE task_bodies SET body=? WHERE task_id='bad'", (_NOT_THIS_TASK[kind],))
            with store.open_store(tmp_path, write=False) as txn:
                rows = hot._read_rows(txn, ["bad", "legacy"])
            assert rows["bad"]["control_valid"] == 0
            assert (rows["legacy"]["control_valid"], rows["legacy"]["budget_pause_attempt"]) == (1, 2)
            with pytest.raises(ValueError, match="unreadable or invalid"):
                hot.read(tmp_path, "bad")


def test_a_fault_in_column_derivation_still_fails_the_read(tmp_path, monkeypatch):
    """Damage is a body that is not this task's object; a defect in the derivation itself stays an error."""
    with store.open_store(tmp_path, write=True) as txn:
        _old_task(txn, "legacy")
    monkeypatch.setattr("ouroboros.task_status.stored_list_order_is_final", lambda _document: 1 / 0)
    with store.open_store(tmp_path, write=False) as txn, pytest.raises(ZeroDivisionError):
        hot._read_rows(txn, ["legacy"])


def test_a_whole_upsert_replaces_such_a_body_with_only_its_own_basis(subtests, tmp_path):
    case_paths = case_folders(tmp_path)
    for kind in list(_NOT_THIS_TASK):
        tmp_path = next(case_paths)
        with subtests.test(kind=kind):
            base = {**_HISTORY, "fanout_count": 1, "requested_count": 1, "child_ids": ["own"]}
            with store.open_store(tmp_path, write=True) as txn:
                for tid in ("plain", "carried"):
                    _old_task(txn, tid)
                    txn.conn.execute("UPDATE task_bodies SET body=? WHERE task_id=?", (_NOT_THIS_TASK[kind], tid))
                _task(txn, "plain")
                _task(txn, "carried", swarm_fanout_rollup=base, swarm_fanout_basis={"base": base})
                plain, carried = txn.load("plain"), txn.load("carried")
            assert "swarm_fanout_rollup" not in plain and "swarm_fanout_basis" not in plain
            assert (carried["swarm_fanout_rollup"], carried["swarm_fanout_basis"]) == (base, {"base": base})
            assert hot.read(tmp_path, "plain")["control_valid"] == 1


def test_a_whole_upsert_keeps_a_valid_held_basis_and_patches_stay_strict(tmp_path):
    with store.open_store(tmp_path, write=True) as txn:
        _task(txn, "kept", swarm_fanout_rollup=_HISTORY, swarm_fanout_basis={"base": _HISTORY})
        _task(txn, "kept")
        assert (txn.load("kept")["swarm_fanout_rollup"], txn.load("kept")["swarm_fanout_basis"]) == (
            _HISTORY, {"base": _HISTORY})
        _old_task(txn, "torn")
        txn.conn.execute("UPDATE task_bodies SET body='{' WHERE task_id='torn'")
    with pytest.raises(json.JSONDecodeError):
        patch_document(tmp_path, "torn", lambda current: {**current, "note": "patched"})
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.body_text("torn") == "{"


# ---- the admitted chat: the room rule's lineage read across the chat step ----------------------------

# The lineage read before the chat column: each row's chat from its body, as the room rule converted it.
_BODY_LINEAGE = ("SELECT t.task_id, CASE json_type(b.body, '$.chat_id') WHEN 'real' THEN CAST(json_extract(b.body, "
                 "'$.chat_id') AS INTEGER) ELSE '' || json_extract(b.body, '$.chat_id') END FROM tasks t LEFT JOIN "
                 "task_bodies b ON b.task_id = t.task_id WHERE t.task_id IN (SELECT value FROM json_each(?))")


def _body_chats(root, ids):
    def converted(value):
        try:
            return int(value) if value not in (None, "") else None
        except (TypeError, ValueError):
            return None

    with store.open_store(root, write=False) as txn:
        return {tid: converted(chat) for tid, chat in txn.conn.execute(_BODY_LINEAGE, (json.dumps(sorted(ids)),))}


def _lineage_chats(root, ids):
    """The room rule's read (``projects_registry._read_lineage``), fresh: each id's admitted chat, None unanswered."""
    from ouroboros import projects_registry as registry

    for key in [key for key in registry._TASK_LINEAGE if key[0] == str(root)]:
        del registry._TASK_LINEAGE[key]
    read = registry._read_lineage(root, ids)
    return None if read is None else {tid: facts["chat_id"] for tid, facts in read.items()}


def schema_ten(root):
    """The store an older body left before the chat step: no chat column or index, version 10."""
    with store.open_store(root, write=True) as txn:
        txn.conn.execute("DROP INDEX IF EXISTS tasks_underived_chat")
        txn.conn.execute("ALTER TABLE tasks DROP COLUMN chat_id")
        txn.conn.execute("UPDATE meta SET value='10' WHERE key='schema_version'")
    store.close_connections(store.store_path(root))


# Each body's ``chat_id`` and the chat the room rule reads from it: a JSON real's integer part, text through int().
_CHATS = [({"chat_id": 5}, 5), ({"chat_id": 7.9}, 7), ({"chat_id": -2.5}, -2), ({"chat_id": "12"}, 12),
          ({"chat_id": " 13 "}, 13), ({"chat_id": "abc"}, None), ({"chat_id": ""}, None), ({"chat_id": None}, None),
          ({"chat_id": True}, 1), ({"chat_id": False}, 0), ({"chat_id": {"a": 1}}, None), ({"chat_id": [3]}, None),
          ({}, None)]


def test_the_admitted_chat_answers_from_its_column_as_the_body_read_did(tmp_path):
    with store.open_store(tmp_path, write=True) as txn:
        for index, (fields, _chat) in enumerate(_CHATS):
            _task(txn, f"d{index:02d}", **fields)  # a writer derives the column
            _old_task(txn, f"u{index:02d}", **fields)  # an older writer leaves it underived
    ids = [f"{side}{index:02d}" for side in "du" for index in range(len(_CHATS))]
    expected = {f"{side}{index:02d}": chat for side in "du" for index, (_fields, chat) in enumerate(_CHATS)}
    assert _body_chats(tmp_path, ids) == expected  # the read the column replaces
    assert _lineage_chats(tmp_path, ids) == expected
    with store.open_store(tmp_path, write=False) as txn:
        assert [txn.row(f"d{index:02d}", "chat_id") for index in range(len(_CHATS))] == [
            "" if chat is None else str(chat) for _fields, chat in _CHATS]  # "" is a known absence, never NULL
        assert all(txn.row(f"u{index:02d}", "chat_id") is None for index in range(len(_CHATS)))
    assert hot.run_hot_facts(tmp_path)["state"] == "done"
    with store.open_store(tmp_path, write=False) as txn:
        assert all(txn.row(f"u{index:02d}", "chat_id") == txn.row(f"d{index:02d}", "chat_id") for index in range(len(_CHATS)))
    assert _lineage_chats(tmp_path, ids) == expected


def test_a_derived_rows_lineage_reads_no_body_and_an_underived_row_still_does(tmp_path):
    """Every body malformed: a read that parses one (``json_type``) fails. The derived rows' lineage answers from
    their columns alone; an older writer's underived row still pays its body read, so its read fails as before."""
    with store.open_store(tmp_path, write=True) as txn:
        _task(txn, "kid", chat_id=7, parent_task_id="mid", root_task_id="top", delegation_role="subagent")
        _task(txn, "mid", chat_id="")
        _old_task(txn, "older", chat_id=8)
        txn.conn.execute("UPDATE task_bodies SET body='{'")
    from ouroboros import projects_registry as registry

    assert _lineage_chats(tmp_path, ["kid", "mid"]) == {"kid": 7, "mid": None}
    assert registry._TASK_LINEAGE[(str(tmp_path), "kid")] == {"parent_task_id": "mid", "root_task_id": "top",
                                                             "delegation_role": "subagent", "chat_id": 7}
    assert _lineage_chats(tmp_path, ["kid", "older"]) is None  # the store cannot answer the older row's chat now


def test_a_store_before_the_chat_step_reads_bodies_and_upgrades_without_a_body_decode(tmp_path, monkeypatch):
    from ouroboros.store_import import upgrade_current_schema

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    write_task_result(tmp_path, "legacy", "completed", result="kept", chat_id=4242)
    write_task_result(tmp_path, "other", "completed", result="kept")
    schema_ten(tmp_path)
    assert _lineage_chats(tmp_path, ["legacy", "other"]) == {"legacy": 4242, "other": None}
    assert _server_looks(tmp_path), "the look must say owed: the runner upgrades first"
    with store.open_store(tmp_path, write=False) as txn:  # the reads above stayed passive
        assert txn.meta("schema_version") == 10
        assert "chat_id" not in {row["name"] for row in txn.conn.execute("PRAGMA table_info(tasks)")}
    with monkeypatch.context() as patch:
        patch.setattr(store.Txn, "body_text", lambda *_: pytest.fail("the schema step decoded a body"))
        upgrade_current_schema(tmp_path)
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.meta("schema_version") == store.SCHEMA_VERSION == hot.CHAT_VERSION
        assert txn.row("legacy", "chat_id") is None and txn.meta("job:" + hot.JOB)["state"] == "pending"
    assert _lineage_chats(tmp_path, ["legacy", "other"]) == {"legacy": 4242, "other": None}  # underived: its body
    assert maintenance.run_job(tmp_path, "hot_facts")["state"] == "done"
    with store.open_store(tmp_path, write=True) as txn:
        assert (txn.row("legacy", "chat_id"), txn.row("other", "chat_id")) == ("4242", "")
        txn.conn.execute("UPDATE task_bodies SET body='{'")  # from here a body read would fail
    assert _lineage_chats(tmp_path, ["legacy", "other"]) == {"legacy": 4242, "other": None}  # the column answers
    assert not _server_looks(tmp_path) and maintenance.run_job(tmp_path, "hot_facts") is None
