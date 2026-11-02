"""HTTP filtered pages pay for their page and effective overlays, not settled bodies."""

import contextlib
import json

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import store, store_hot_facts, task_store
from ouroboros.gateway import tasks
from ouroboros.task_results import write_task_result
from tests._store_shared import seed_result


def client_for(root):
    app = Starlette(routes=[Route("/api/tasks", tasks.api_tasks_list),
                           Route("/api/tasks/{task_id}", tasks.api_task_get)])
    app.state.drive_root = root
    return TestClient(app)


@contextlib.contextmanager
def observed_bodies(root, monkeypatch):
    """Observe SQLite materializing body bytes, including JSON1 projections.

    A Python decode spy misses a SELECT json_remove(body, ...). This synthetic
    view puts the probe below either query shape, on the actual body column.
    HTTP uses a different thread/connection, so register on every connection.
    """
    reads = []
    execute = store._StoreConnection.execute

    def body(task_id, text):
        reads.append((task_id, len(text)))
        return text

    def observed(conn, *args, **kwargs):
        conn.create_function("observed_body", 2, body)
        return execute(conn, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(store._StoreConnection, "execute", observed)
        with store.open_store(root, write=True) as txn:
            txn.conn.execute("ALTER TABLE task_bodies RENAME TO fixture_bodies")
            txn.conn.execute("CREATE VIEW task_bodies AS SELECT task_id, "
                             "observed_body(task_id, body) AS body FROM fixture_bodies")
        try:
            yield reads
        finally:
            with store.open_store(root, write=True) as txn:
                txn.conn.execute("DROP VIEW task_bodies")
                txn.conn.execute("ALTER TABLE fixture_bodies RENAME TO task_bodies")


@pytest.mark.scale
def test_filtered_http_page_reads_only_selected_stable_bodies(tmp_path, monkeypatch):
    measurements = []
    for count, body_size in ((20, 2048), (200, 24000)):
        root = tmp_path / str(count)
        for index in range(count):
            seed_result(root, f"task-{index:04}", {"status": "completed", "result": "answer",
                "ts": f"2026-01-01T00:{index // 60:02}:{index % 60:02}Z", "trace_refs": ["x" * body_size],
                "future_envelope": {"nested": ["keep", {"unknown": True}]}})
        expected = [f"task-{count - 1:04}", f"task-{count - 2:04}"]
        with client_for(root) as client, observed_bodies(root, monkeypatch) as reads:
            for _ in range(2):
                reads.clear()
                response = client.get("/api/tasks?status=completed&limit=2")
                assert response.status_code == 200
                rows = response.json()["tasks"]
                assert [row["task_id"] for row in rows] == expected
                assert sorted(tid for tid, _ in reads) == sorted(expected)
                assert all(row["result"] == "answer" and row["future_envelope"] == {
                    "nested": ["keep", {"unknown": True}]} for row in rows)
                assert all("trace_refs" not in row for row in rows)
                measurements.append({"history": count, "body_size": body_size,
                                     "body_reads": len(reads), "body_chars": sum(size for _, size in reads)})
    print("FILTERED_PAGE_READS", json.dumps(measurements, sort_keys=True))


def test_workspace_terminal_pending_is_in_running_http_list(tmp_path):
    seed_result(tmp_path, "pending", {"status": "completed", "workspace_root": str(tmp_path / "workspace"),
                "artifact_status": "pending", "result": "code ready", "ts": "2026-01-01T00:00:01Z"})
    with client_for(tmp_path) as client:
        for artifact_status, expected in (("pending", "running"), ("ready", "completed"),
                                          ("finalizing", "running"), ("failed", "completed")):
            write_task_result(tmp_path, "pending", "completed", artifact_status=artifact_status)
            detail = client.get("/api/tasks/pending")
            listed = client.get(f"/api/tasks?status={expected}&limit=2")
            assert detail.status_code == listed.status_code == 200
            assert detail.json()["status"] == expected
            assert [row["task_id"] for row in listed.json()["tasks"]] == ["pending"]
            assert listed.json()["tasks"][0]["artifact_status"] == detail.json()["artifact_status"]


def test_effective_global_order_envelope_and_repeated_copyback(tmp_path):
    root, child = tmp_path / "root", tmp_path / "child"
    for tid, ts in (("tie-a", "2026-01-02"), ("tie-b", "2026-01-02"), ("older", "2026-01-01")):
        seed_result(root, tid, {"status": "completed", "ts": ts, "result": tid,
                               "new_key": {"keep": [1, 2]}, "loop_outcome": {"usage": {"total_rounds": 6}}})
    seed_result(root, "child-copy", {"status": "scheduled", "ts": "2025-01-01",
                                     "metadata": {"child_drive_root": str(child)}})
    seed_result(child, "child-copy", {"status": "completed", "ts": "2026-01-04", "result": "child answer"})
    seed_result(root, "retry", {"status": "interrupted", "ts": "2025-01-01", "retry_task_id": "older"})
    seed_result(root, "failed-parent", {"status": "failed", "ts": "2026-01-03", "child_drive_root": str(child),
                                       "result": "parent failure"})
    seed_result(child, "failed-parent", {"status": "running", "ts": "2026-01-05", "result": "stale mirror"})
    with client_for(root) as client:
        for stage in range(3):
            all_rows = client.get("/api/tasks?status=completed,failed&limit=0").json()["tasks"]
            expected = [tasks._compact_list_row(tasks.public_task_result(tasks.effective_task_result(
                root, row, materialize_artifacts=False))) for row in task_store.list_documents(root)]
            expected = sorted((row for row in expected if row["status"] in {"completed", "failed"}),
                              key=lambda row: str(row.get("ts") or ""), reverse=True)
            assert all_rows == expected
            for limit in (1, 2, 4):
                assert client.get(f"/api/tasks?status=completed,failed&limit={limit}").json()["tasks"] == expected[:limit]
            if stage == 0:
                assert all_rows[0]["result"] == "child answer"
                assert all_rows[1]["result"] == "parent failure"
                write_task_result(child, "child-copy", "completed", ts="2026-01-06", result="late child answer")
            elif stage == 1:
                # Copy-back changes the writer's selection fact and retains the response envelope.
                task_store.patch_document(root, "child-copy", lambda row: {
                    **row, "status": "completed", "ts": "2026-01-06", "metadata": {}, "result": "copied answer"})


def test_live_queue_unknown_and_cancellation_still_use_effective_reader(tmp_path):
    from ouroboros.cancel_intents import request_cancel

    seed_result(tmp_path, "live", {"status": "running", "ts": "2026-01-01", "result": "working"})
    seed_result(tmp_path, "old-latch", {"status": "cancel_requested", "ts": "2026-01-01"})
    seed_result(tmp_path, "unknown", {"status": "future_status", "ts": "2026-01-01"})
    request_cancel(tmp_path, "live", reason="fixture", source="fixture")
    with client_for(tmp_path) as client:
        rows = client.get("/api/tasks?status=running,cancel_requested,future_status&limit=0").json()["tasks"]
        indexed = {row["task_id"]: row for row in rows}
        assert indexed["live"]["status"] == "running"
        assert indexed["live"]["cancel_state"] == "pending"
        assert indexed["live"]["queue_reconciliation_warning"] == "queue snapshot missing or invalid"
        assert indexed["old-latch"]["cancel_state"] == "pending"
        assert indexed["unknown"]["status"] == "future_status"


def test_legacy_selection_is_read_only_then_maintenance_and_old_writer_rederive(tmp_path, monkeypatch):
    for index in range(20):
        seed_result(tmp_path, f"task-{index:02}", {"status": "completed", "ts": f"2026-01-{index + 1:02}"})
    with store.open_store(tmp_path, write=True) as txn:
        txn.conn.execute("UPDATE tasks SET list_stable=NULL")
    before = task_store.inventory(tmp_path)
    with client_for(tmp_path) as client, observed_bodies(tmp_path, monkeypatch) as reads:
        assert len(client.get("/api/tasks?status=completed&limit=2").json()["tasks"]) == 2
        assert len(reads) == 20
    assert task_store.inventory(tmp_path) == before
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.conn.execute("SELECT count(*) FROM tasks WHERE list_stable IS NULL").fetchone()[0] == 20
    assert store_hot_facts.run_hot_facts(tmp_path)["state"] == "done"
    assert task_store.inventory(tmp_path) == before
    with client_for(tmp_path) as client, observed_bodies(tmp_path, monkeypatch) as reads:
        client.get("/api/tasks?status=completed&limit=2")
        assert len(reads) == 2
    # A genuine old-column REPLACE must not keep the prior stable bit.
    with store.open_store(tmp_path, write=True) as txn:
        doc = txn.load("task-00")
        doc.update(workspace_root=str(tmp_path / "workspace"), artifact_status="pending")
        columns = {key: value for key, value in store.typed_columns(doc).items() if key != "list_stable"}
        columns["write_seq"] = txn.row("task-00", "write_seq") + 1
        txn.conn.execute("UPDATE task_bodies SET body=? WHERE task_id=?", (store.encode_document(doc), "task-00"))
        names = ("task_id", *columns)
        txn.conn.execute(f"INSERT OR REPLACE INTO tasks ({','.join(names)}) VALUES ({','.join('?' * len(names))})",
                         ("task-00", *columns.values()))
        assert txn.row("task-00", "list_stable") is None
    with client_for(tmp_path) as client:
        assert [row["task_id"] for row in client.get("/api/tasks?status=running&limit=2").json()["tasks"]] == ["task-00"]
    assert store_hot_facts.run_hot_facts(tmp_path)["state"] == "done"
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.row("task-00", "list_stable") == 0


def test_queue_only_and_unfiltered_keep_bounded_paths(tmp_path, monkeypatch):
    for index in range(20):
        seed_result(tmp_path, str(index), {"status": "completed", "ts": f"2026-01-{index + 1:02}"})
    with client_for(tmp_path) as client, observed_bodies(tmp_path, monkeypatch) as reads:
        assert client.get("/api/tasks?queue_only=1&status=completed").json()["tasks"] == []
        assert not reads
        rows = client.get("/api/tasks?limit=2").json()["tasks"]
        assert [row["task_id"] for row in rows] == ["19", "18"]
        assert len(reads) == 2


def test_schema_eight_get_is_read_only_and_upgrade_reads_no_bodies(tmp_path, monkeypatch):
    from ouroboros.store_import import upgrade_current_schema

    for index in range(20):
        seed_result(tmp_path, str(index), {"status": "completed", "ts": f"2026-01-{index + 1:02}"})
    # Actual old shape: no selection column/index, existing control facts intact.
    with store.open_store(tmp_path, write=True) as txn:
        txn.conn.execute("DROP INDEX tasks_list_stable")
        txn.conn.execute("DROP INDEX tasks_underived_hot")
        txn.conn.execute("ALTER TABLE tasks DROP COLUMN list_stable")
        predicate = store_hot_facts.TASK_UNDERIVED.replace(" OR list_stable IS NULL", "")
        txn.conn.execute(f"CREATE INDEX tasks_underived_hot ON tasks(task_id) WHERE {predicate}")
        txn.conn.execute("UPDATE meta SET value='8' WHERE key='schema_version'")
    before = task_store.inventory(tmp_path)
    with client_for(tmp_path) as client, observed_bodies(tmp_path, monkeypatch) as reads:
        assert len(client.get("/api/tasks?status=completed&limit=2").json()["tasks"]) == 2
        assert len(reads) == 20
        reads.clear()
        assert store_hot_facts.read(tmp_path, "0")["status"] == "completed"
        assert not reads
    assert task_store.inventory(tmp_path) == before
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.meta("schema_version") == 8
        assert "list_stable" not in {row["name"] for row in txn.conn.execute("PRAGMA table_info(tasks)")}
    with observed_bodies(tmp_path, monkeypatch) as reads:
        upgrade_current_schema(tmp_path)
        assert not reads
        assert store_hot_facts.read(tmp_path, "0")["status"] == "completed"
        assert not reads  # a pending list backfill does not invalidate control facts
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.meta("schema_version") == store.SCHEMA_VERSION
        assert txn.meta("job:" + store_hot_facts.JOB)["state"] == "pending"
    assert store_hot_facts.run_hot_facts(tmp_path)["state"] == "done"
    with client_for(tmp_path) as client, observed_bodies(tmp_path, monkeypatch) as reads:
        assert len(client.get("/api/tasks?status=completed&limit=2").json()["tasks"]) == 2
        assert len(reads) == 2
