"""Real child-process SQL boundaries, with only disposable public-API stores."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import sys
import threading
import time

import pytest

from ouroboros import query_data_sql as sql, store, store_import, usage_store
from ouroboros.tools.tool_context import ToolContext


@pytest.fixture
def ctx(tmp_path):
    store_import.ensure_current_store(tmp_path, with_children=False)
    store.update_task(tmp_path, "caller", lambda _: {"status": "completed", "result": "body only"})
    return ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="caller")


def run(ctx, query, **kwargs):
    # These cases test SQL/cleanup semantics, not completion within one second
    # on a saturated host. Explicit deadline cases override this allowance;
    # the registered-tool default is exercised by the mature-money test.
    args = dict(db="current", query=query, max_rows=100, max_bytes=65536,
                snapshot=False, timeout_sec=5.0)
    args.update(kwargs)
    return sql.run_sql(ctx, ctx.drive_root, **args)


def test_sql_child_keeps_windows_systemroot(ctx, monkeypatch):
    from ouroboros import process_custody

    # Windows enumerates os.environ keys in uppercase, including SystemRoot.
    system_root = os.environ.get("SystemRoot", str(ctx.drive_root / "Windows"))
    monkeypatch.setenv("SYSTEMROOT", system_root)
    monkeypatch.setenv("QUERY_TEST_SECRET", "fixture-only")
    original_spawn = process_custody.spawn_supervised

    def spawn(cmd, **kwargs):
        assert kwargs["env"].get("SYSTEMROOT") == system_root
        assert "QUERY_TEST_SECRET" not in kwargs["env"]
        return original_spawn(cmd, **kwargs)

    monkeypatch.setattr(process_custody, "spawn_supervised", spawn)
    assert run(ctx, "SELECT 42")["data"]["rows"] == [[42]]


@pytest.mark.parametrize("query", [
    "ATTACH DATABASE ':memory:' AS other",
    "CREATE TABLE surprise(x)",
    "DELETE FROM tasks",
    "UPDATE tasks SET status='failed'",
    "INSERT INTO tasks(task_id) VALUES ('unexpected')",
    "DROP TABLE tasks",
    "PRAGMA user_version=4",
    "PRAGMA query_only=OFF",
    "SELECT load_extension('not-an-extension')",
    "SELECT body FROM task_bodies",
])
def test_sql_denials_and_working_select(ctx, query):
    blocked = run(ctx, query)
    assert blocked["status"] == "error", blocked
    assert run(ctx, "SELECT task_id, status FROM tasks")["data"]["rows"] == [["caller", "completed"]]
    with store.open_store(ctx.drive_root, write=False) as txn:
        assert txn.load("caller")["result"] == "body only"


@pytest.mark.parametrize("db", ["current", "money"])
@pytest.mark.parametrize("snapshot", [False, True])
@pytest.mark.parametrize("query,message", [
    ("DELETE FROM meta", "not authorized"),
    ("SELECT )", 'near ")": syntax error'),
    ("SELECT 1; SELECT 2", "You can only execute one statement at a time."),
])
def test_sql_errors_are_query_failures_not_store_damage(ctx, db, snapshot, query, message):
    if db == "money":
        usage_store.migrate_from_journal(ctx.drive_root)
    failed = run(ctx, query, db=db, snapshot=snapshot)
    assert failed == {"status": "error", "code": "query_failed", "phase": "sql",
                      "exception_class": "ValueError", "message": message}
    assert run(ctx, "SELECT 42", db=db, snapshot=snapshot)["data"]["rows"] == [[42]]


def test_attach_cannot_create_a_file(ctx):
    target = ctx.drive_root / "unexpected.sqlite"
    assert run(ctx, "ATTACH DATABASE '" + str(target) + "' AS other")["status"] == "error"
    assert not target.exists()
    assert run(ctx, "SELECT 42")["data"]["rows"] == [[42]]


@pytest.mark.parametrize("layer", ["mode_ro", "query_only"])
def test_sql_write_layers_independently(ctx, monkeypatch, layer):
    original_connect = sqlite3.connect

    class ProbeConnection(sqlite3.Connection):
        def set_authorizer(self, callback):
            return super().set_authorizer(lambda *args: sqlite3.SQLITE_OK)

        def execute(self, statement, *args, **kwargs):
            # Exercise the real connection setup without changing pytest's heap limit.
            if statement.startswith("PRAGMA hard_heap_limit="):
                statement = "SELECT 1"
            elif layer == "mode_ro" and statement == "PRAGMA query_only=ON":
                statement = "PRAGMA query_only=OFF"
            return super().execute(statement, *args, **kwargs)

    def connect(database, **kwargs):
        if layer == "query_only":
            database = database.replace("mode=ro", "mode=rw")
        return original_connect(database, factory=ProbeConnection, **kwargs)

    monkeypatch.setattr(sql.sqlite3, "connect", connect)
    request = dict(db="current", query="SELECT task_id FROM tasks", max_rows=10, max_bytes=4096)
    with sql._connection(store.store_path(ctx.drive_root), store.TIER_ENFORCED, sql._Budget(1)) as conn:
        assert sql._fetch(conn, request)["rows"] == [["caller"]]
        assert conn.execute("PRAGMA query_only").fetchone() == (int(layer == "query_only"),)
        with pytest.raises(sqlite3.OperationalError, match="readonly database"):
            sql._fetch(conn, {**request, "query": "CREATE TABLE query_probe(value)"})
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='query_probe'").fetchall() == []


def test_authorizer_denies_extension_function_independently():
    conn = sqlite3.connect(":memory:")  # Writable, with query_only off.
    try:
        # A callable replacement removes SQLite's separate extension-loading refusal.
        conn.create_function("load_extension", 1, lambda name: "callable")
        assert conn.execute("SELECT load_extension('test-extension')").fetchone() == ("callable",)
        conn.set_authorizer(sql._authorizer("money"))
        assert conn.execute("SELECT length('readable')").fetchone() == (8,)
        with pytest.raises(sqlite3.DatabaseError, match="not authorized to use function: load_extension"):
            conn.execute("SELECT load_extension('test-extension')")
    finally:
        conn.close()


@pytest.mark.parametrize("pragma", ["table_info(tasks)", "table_xinfo(tasks)", "table_list",
                                    "index_list(tasks)", "index_info(tasks_parent)"])
def test_metadata_pragmas_are_readable(ctx, pragma):
    result = run(ctx, "PRAGMA " + pragma)
    assert result["status"] == "ok", result
    assert result["data"]["rows"]


def test_heap_is_limited_only_in_child(ctx):
    with sqlite3.connect(":memory:") as parent:
        before = parent.execute("PRAGMA hard_heap_limit").fetchone()
        failed = run(ctx, "SELECT zeroblob(419430400)")
        assert failed == {"status": "error", "code": "memory_limit", "phase": "sql", "exception_class": "MemoryError"}
        assert parent.execute("PRAGMA hard_heap_limit").fetchone() == before
        assert parent.execute("SELECT length(zeroblob(1000000))").fetchone() == (1000000,)
    assert run(ctx, "SELECT 3")["status"] == "ok"


@pytest.mark.parametrize("snapshot", [False, True])
def test_interrupt_byte_cap_and_row_cap(ctx, snapshot):
    started = time.monotonic()
    failed = run(ctx, "WITH RECURSIVE n(x) AS (VALUES(0) UNION ALL SELECT x+1 FROM n) SELECT sum(x) FROM n",
                 timeout_sec=0.05, snapshot=snapshot)
    assert failed["status"] == "error" and failed["code"] == "deadline", failed
    assert failed["phase"] in {"acquire", "snapshot_copy", "snapshot_check", "sql"}, failed
    assert time.monotonic() - started < 6
    limited = run(ctx, "SELECT printf('%010000d',1)", max_bytes=256)
    assert limited["data"] == {"columns": ["printf('%010000d',1)"], "rows": [], "truncated": True}
    rows = run(ctx, "SELECT 1 UNION ALL SELECT 2", max_rows=1)
    assert rows["data"]["rows"] == [[1]] and rows["data"]["truncated"]
    assert run(ctx, "SELECT 1")["data"]["truncated"] is False


def test_money_snapshot_and_direct_reads(ctx):
    usage_store.migrate_from_journal(ctx.drive_root)
    for snapshot in (True, False):
        result = run(ctx, "SELECT count(*) FROM sqlite_master", db="money", snapshot=snapshot)
        assert result["status"] == "ok" and result["data"]["rows"][0][0] > 0, result


@pytest.mark.parametrize("query,status", [("SELECT 1", "ok"), ("SELECT )", "error")])
def test_snapshot_removes_real_temporary_directory(ctx, monkeypatch, query, status):
    usage_store.migrate_from_journal(ctx.drive_root)
    original_temp, directories = sql.tempfile.TemporaryDirectory, []

    def temporary(**kwargs):
        directory = original_temp(**kwargs)
        path = Path(directory.name)
        assert path.is_dir()
        directories.append(path)
        return directory

    monkeypatch.setattr(sql.tempfile, "TemporaryDirectory", temporary)
    result = run(ctx, query, db="money", snapshot=True)
    assert result["status"] == status, result
    assert directories and all(not directory.exists() for directory in directories)


def test_snapshot_is_a_distinct_checked_database(ctx, tmp_path, monkeypatch):
    # Isolate the in-process probe too: never set SQLite's global heap in pytest.
    seen = []
    from contextlib import contextmanager

    @contextmanager
    def connection(path, tier, budget):
        conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, isolation_level=None)
        try:
            conn.set_trace_callback(lambda statement: seen.append((path, statement)))
            yield conn
        finally:
            conn.close()

    monkeypatch.setattr(sql, "_connection", connection)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    result = sql._execute(dict(root=str(ctx.drive_root), db="current", timeout_sec=1,
                               query="SELECT task_id FROM tasks", snapshot=True,
                               scratch=str(scratch), max_rows=10, max_bytes=65536))
    assert result["rows"] == [["caller"]]
    assert any("VACUUM INTO" in statement for _, statement in seen)
    assert any(statement == "PRAGMA quick_check" for _, statement in seen)
    assert next(path for path, statement in seen if statement == "SELECT task_id FROM tasks") == scratch / "snapshot.sqlite"


def test_current_store_hold_stays_within_half_the_writers_wait(ctx, tmp_path, monkeypatch):
    # Task writes wait store.STORE_WAIT_SEC on the lock a current-store read holds; a snapshot's own query holds nothing.
    from contextlib import contextmanager
    from itertools import count
    from types import SimpleNamespace

    # A clock that ticks on every read, as a fine one does: a coarse one (Windows) may not tick between two reads.
    clock = count(1000.0, 0.001)
    monkeypatch.setattr(sql, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    seen, real_source = [], sql._source

    @contextmanager
    def connection(path, tier, budget):
        seen.append((path.name, budget.remaining()))
        conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, isolation_level=None)
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def source(root, db, budget):
        seen.append(("source", budget.remaining()))
        with real_source(root, db, budget) as tier:
            assert tier == store.TIER_ENFORCED
            yield tier

    monkeypatch.setattr(sql, "_connection", connection)
    monkeypatch.setattr(sql, "_source", source)
    assert sql.CURRENT_HOLD_SEC == store.STORE_WAIT_SEC / 2
    for timeout, held in ((9, sql.CURRENT_HOLD_SEC), (1, 1)):
        seen.clear()
        scratch = tmp_path / f"scratch-{timeout}"
        scratch.mkdir()
        result = sql._execute(dict(root=str(ctx.drive_root), db="current", timeout_sec=timeout, snapshot=True,
                                   query="SELECT task_id FROM tasks", scratch=str(scratch), max_rows=10, max_bytes=65536))
        assert result["rows"] == [["caller"]]
        assert [name for name, _ in seen] == ["source", store.store_path(ctx.drive_root).name, "snapshot.sqlite"]
        assert 0 < seen[1][1] < seen[0][1] <= held
        assert 0 < seen[2][1] < timeout


def test_name_tier_uses_the_store_lock(tmp_path, monkeypatch):
    from ouroboros import platform_layer
    monkeypatch.setattr(platform_layer, "kernel_file_locks_enforced", lambda _, **_kw: False)
    store_import.ensure_current_store(tmp_path, with_children=False)
    store.update_task(tmp_path, "caller", lambda _: {"status": "completed"})
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="caller")
    with sql._source(tmp_path, "current", sql._Budget(1)) as tier:
        assert tier == store.TIER_NAME
    assert run(ctx, "SELECT task_id FROM tasks")["data"]["rows"] == [["caller"]]
    with store.name_lock(tmp_path, store.LOCK_NAME, timeout_sec=1):
        result = run(ctx, "SELECT 1", timeout_sec=0.05)
        assert result["status"] == "error" and result["code"] == "deadline" and result["phase"] == "acquire", result
    from ouroboros.process_custody import ledger_path
    ledger = ledger_path(tmp_path)
    records = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert any(record.get("purpose") == "query_data" for record in records)


def test_parent_deadline_reaps_child_and_removes_scratch(ctx, monkeypatch):
    from ouroboros import process_custody
    original_spawn, original_temp = process_custody.spawn_supervised, sql.tempfile.TemporaryDirectory
    children, directories = [], []

    def stalled(cmd, **kwargs):
        proc = original_spawn([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
        children.append(proc)
        return proc

    def temporary(**kwargs):
        directory = original_temp(**kwargs)
        directories.append(Path(directory.name))
        return directory

    monkeypatch.setattr(process_custody, "spawn_supervised", stalled)
    monkeypatch.setattr(sql.tempfile, "TemporaryDirectory", temporary)
    monkeypatch.setattr(sql, "STARTUP_SEC", 0.1)
    assert run(ctx, "SELECT 1", timeout_sec=0.001) == {"status": "error", "code": "deadline", "phase": "executor"}
    assert children[0].poll() is not None
    assert children[0].stdin is None and children[0].stdout.closed
    assert directories and all(not directory.exists() for directory in directories)


def test_request_larger_than_a_pipe_reaches_a_late_child_whole(ctx, monkeypatch):
    # CPython <= 3.12 never finishes stdin input that a timed-out communicate() left
    # unsent: a request beyond any pipe buffer must still reach a late child whole.
    from ouroboros import process_custody
    original_spawn = process_custody.spawn_supervised

    def late(cmd, **kwargs):
        script = "import time; time.sleep(0.3); from ouroboros.query_data_sql import main; main()"
        return original_spawn([sys.executable, "-c", script, *cmd[3:]], **kwargs)

    monkeypatch.setattr(process_custody, "spawn_supervised", late)
    text = "я" * 12000  # 72 KB once JSON-escaped, yet within the tool's 16384-character query limit
    result = run(ctx, f"SELECT length(value) AS n, substr(value, -2) AS tail FROM (SELECT '{text}' AS value)", timeout_sec=5)
    assert result == {"status": "ok", "data": {"columns": ["n", "tail"], "rows": [[12000, "яя"]], "truncated": False}}


def test_child_answer_is_utf8_whatever_its_stdout_code_page(tmp_path, monkeypatch):
    """A Windows pipe hands the child cp1252 stdout: its answer is UTF-8 bytes there too, and the
    parent's ``json.loads`` reads them back whole."""
    import io
    from ouroboros import process_custody

    request = tmp_path / "request.json"
    request.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["query_data_sql", "session", str(request)])
    monkeypatch.setattr(process_custody, "adopt_session_id", lambda _session: None)
    monkeypatch.setattr(process_custody, "start_parent_lifeline", lambda **_kwargs: None)
    monkeypatch.setattr(sql, "_result", lambda _request: {"status": "ok", "data": {"rows": [["яё𐍈"]]}})
    wire = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(wire, encoding="cp1252"))
    sql.main()
    assert json.loads(wire.getvalue()) == {"status": "ok", "data": {"rows": [["яё𐍈"]]}}


def test_sqlite_failure_owns_only_its_interrupt_and_an_expired_busy_wait(tmp_path):
    # Real SQLite errors: Python 3.10 has no sqlite_errorcode and no sqlite3.SQLITE_* names.
    def failure(conn, statement):
        with pytest.raises((sqlite3.Error, sqlite3.Warning)) as caught:
            conn.execute(statement)
        conn.close()
        return caught.value

    interrupting = sqlite3.connect(":memory:")
    interrupting.set_progress_handler(lambda: 1, 1)  # SQLite's own interrupt path
    holder = sqlite3.connect(tmp_path / "held.sqlite", isolation_level=None)
    holder.execute("CREATE TABLE t(x)")
    holder.execute("BEGIN EXCLUSIVE")
    (tmp_path / "damaged.sqlite").write_bytes(b"not a database" * 512)
    errors = dict(interrupt=failure(interrupting, "SELECT 1"),
                  busy=failure(sqlite3.connect(tmp_path / "held.sqlite", timeout=0), "SELECT x FROM t"),
                  syntax=failure(sqlite3.connect(":memory:"), "SELECT )"),
                  damaged=failure(sqlite3.connect(tmp_path / "damaged.sqlite"), "SELECT * FROM sqlite_master"))
    holder.close()
    live, spent, fired, quiet = sql._Budget(60), sql._Budget(0), threading.Event(), threading.Event()
    fired.set()

    def owned(budget, error, timer):
        try:
            budget.sqlite_failure(error, timer)
        except sql.QueryDeadline as deadline:
            assert deadline.__cause__ is error
            return True
        return False

    assert owned(live, errors["interrupt"], fired)  # Our timer fired: the expiry is the query's.
    assert not owned(spent, errors["interrupt"], quiet)  # Anyone else's interrupt stays a failure.
    assert owned(spent, errors["busy"], quiet) and not owned(live, errors["busy"], quiet)
    for name in ("syntax", "damaged"):  # Genuine failures survive our fired timer and expiry.
        assert not owned(spent, errors[name], fired), name


@pytest.mark.parametrize("started", [False, True])
@pytest.mark.parametrize("returncode", [None, 0, 1])
def test_late_parent_observation_keeps_exited_child_answer(tmp_path, monkeypatch, started, returncode):
    from types import SimpleNamespace

    if started:
        (tmp_path / "deadline").write_text("1", encoding="ascii")
    clock = iter([0, 10])  # The parent resumes after either supervision bound.
    monkeypatch.setattr(sql, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    reads = []
    answer = b'{"status":"error","code":"query_failed","exception_class":"UsageLedgerCorrupt"}'

    def communicate():
        reads.append(True)
        return answer, None

    proc = SimpleNamespace(args=["fixture-child"], poll=lambda: returncode, communicate=communicate)
    if returncode is None:
        with pytest.raises(sql.subprocess.TimeoutExpired):
            sql._communicate(proc, tmp_path, 1)
        assert not reads
    else:
        assert sql._communicate(proc, tmp_path, 1) == answer
        assert reads == [True]


@pytest.mark.parametrize("db,lock_name", [("money", "usage_attempts.lock"), ("packs", "packs_index.lock")])
def test_name_tier_money_and_index_reads_share_writer_locks(tmp_path, monkeypatch, db, lock_name):
    from ouroboros import platform_layer
    monkeypatch.setattr(platform_layer, "kernel_file_locks_enforced", lambda _, **_kw: False)
    store_import.ensure_current_store(tmp_path, with_children=False)
    if db == "money":
        usage_store.migrate_from_journal(tmp_path)
    else:
        from ouroboros import evidence_packs
        path = evidence_packs.packs_dir(tmp_path) / evidence_packs.INDEX_NAME
        path.parent.mkdir(parents=True)
        conn = sqlite3.connect(path)
        try:
            conn.execute("CREATE TABLE members(name TEXT)")
        finally:
            conn.close()
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="caller")
    assert run(ctx, "SELECT 1", db=db)["data"]["rows"] == [[1]]
    with store.name_lock(tmp_path, lock_name, timeout_sec=1):
        result = run(ctx, "SELECT 1", db=db, timeout_sec=0.05)
        assert result["status"] == "error" and result["code"] == "deadline" and result["phase"] == "acquire", result
    assert run(ctx, "SELECT 2", db=db, snapshot=True)["data"]["rows"] == [[2]]


@pytest.mark.parametrize("tier", [store.TIER_ENFORCED, store.TIER_NAME])
def test_pack_sql_holds_the_index_owner_lock(ctx, monkeypatch, tier):
    from ouroboros import evidence_packs as packs
    from tests.test_evidence_packs import _task

    monkeypatch.setattr(packs, "index_tier", lambda _root: tier)
    _task(ctx.drive_root, "packed", blobs=1)
    packs.write_pack(ctx.drive_root, "packed", 1, packs.loose_members(ctx.drive_root, "packed"))
    with packs.open_index(ctx.drive_root, write=True):
        result = run(ctx, "SELECT count(*) FROM members", db="packs", timeout_sec=0.05)
        assert result["status"] == "error" and result["code"] == "deadline" and result["phase"] == "acquire", result
    assert run(ctx, "SELECT count(*) FROM members", db="packs")["data"]["rows"] == [[2]]
    # Admission retains the owner's one lock through the read-only query scope.
    with sql._source(ctx.drive_root, "packs", sql._Budget(1)) as observed:
        assert observed == tier
        with pytest.raises(store.StoreBusy):
            with store.name_lock(ctx.drive_root, packs.INDEX_LOCK_NAME, timeout_sec=0.01):
                pytest.fail("index owner lock was released before SQL")
