"""#1974: registered money reads and supervised expiry, using synthetic stores only.

Controlled child boundaries make copy/check/SQLite failure modes distinguishing
without constructing a 566 MB owner ledger. The mature fixture has real money
schema/indexes and 12,000 wide attempts; it is not a reducer correctness oracle.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import sys
import time

import pytest

from ouroboros import query_data_sql as sql, store, store_import, usage_store
from ouroboros.tools.registry import ToolContext, ToolRegistry


@pytest.fixture
def money(tmp_path, request, monkeypatch):
    from ouroboros import platform_layer
    journal = getattr(request, "param", None)
    if journal == "name":
        monkeypatch.setattr(platform_layer, "kernel_file_locks_enforced", lambda *_a, **_k: False)
    elif journal == "PERSIST":
        monkeypatch.setattr(store, "_journal_mode", lambda *_a: "PERSIST")
    elif journal == "WAL" and store._journal_mode(tmp_path, store.TIER_ENFORCED) != "WAL":
        pytest.skip("this SQLite/filesystem does not support the store's WAL policy")
    usage_store.migrate_from_journal(tmp_path)
    store_import.ensure_current_store(tmp_path, with_children=False)
    store.update_task(tmp_path, "caller", lambda _: {"status": "completed"})
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="caller")
    yield ctx
    store.close_connections_under(tmp_path)
    usage_store.forget(tmp_path)


def call(ctx, query="SELECT count(*) FROM attempts", **kwargs):
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    result = registry.execute_result("query_data", dict(op="sql", db="money", query=query, **kwargs))
    value = json.loads(result.text)
    assert result.status == value["status"]
    return value


def mature(ctx):
    # A bulk test fixture, not a product write path: one owner transaction and
    # the real schema/indexes. Deliberately broad rows distinguish copying from
    # the indexed/aggregate readonly SQL the tool was actually asked to do.
    with usage_store.hold(ctx.drive_root) as txn:
        txn.conn.execute("""WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<12000)
            INSERT INTO attempts(attempt_id,revision,seq,seq_first,kind,state,task_id,root_task_id,
                model,provider,category,cost_usd,cost_final,pricing_known,extra)
            SELECT 'fixture-'||x,1,x,x,'attempt','settled','caller','root','fixture','local','task',
                '0.125',1,1,? FROM n""", (json.dumps({"fixture": "x" * 768}),))
    path = Path(ctx.drive_root) / usage_store.STORE_REL
    size = path.stat().st_size
    with usage_store.read(ctx.drive_root) as txn:
        # The independent expected value is derived from the fixture's literal
        # row count/cost, not another query through the code being checked.
        assert txn.conn.execute("SELECT count(*) FROM attempts").fetchone()[0] == 12000
    print(f"MATURE_MONEY rows=12000 bytes={size} sqlite={sqlite3.sqlite_version}")
    assert size >= 8 * 1024 * 1024


def _child_main(mode, trace):
    """Test-only instrumentation INSIDE the real supervised SQL child.

    Nothing mocks the handler, budget, authorizer, source hold or process owner.
    Only the named controlled boundary is delayed or made to fail.
    """
    trace = Path(trace)
    real_connection, real_source, real_enter = sql._connection, sql._source, sql._Budget.enter
    real_result = sql._result

    def record(**row):
        with trace.open("a", encoding="utf-8") as output:
            output.write(json.dumps(row) + "\n")

    class Probe:
        def __init__(self, conn, budget):
            self.conn, self.budget = conn, budget

        def __getattr__(self, name):
            return getattr(self.conn, name)

        def execute(self, statement, *args):
            if statement.startswith("VACUUM"):
                if mode == "stall_copy":
                    record(executing="snapshot_copy", started=self.budget.started)
                    time.sleep(20)  # Deliberately noninterruptible while the source is held.
                if mode.startswith(("late_", "tardy_")):
                    record(expires=self.budget.expires)
                    time.sleep(self.budget.remaining() + 0.05)  # Unrelated failure survives our fired timer.
                if mode in {"copy_io", "late_copy_io"}:
                    raise OSError("controlled copy I/O failure")
                if mode.endswith("copy_corrupt") or mode == "copy_unowned_interrupt":
                    error = (sqlite3.DatabaseError("controlled " + mode) if mode.endswith("copy_corrupt")
                             else sqlite3.OperationalError("controlled " + mode))
                    # Literal result codes: Python 3.10 has no sqlite3.SQLITE_CORRUPT/INTERRUPT.
                    error.sqlite_errorcode = 11 if mode.endswith("copy_corrupt") else 9
                    raise error
            cursor = self.conn.execute(statement, *args)
            if statement == "PRAGMA quick_check" and mode == "bad_check":
                cursor.close()
                class BadCheck:
                    def fetchall(self):
                        return [("controlled integrity failure",)]
                    def close(self):
                        pass
                return BadCheck()
            return cursor

    @contextmanager
    def connection(path, tier, budget):
        with real_connection(path, tier, budget) as conn:
            seen = set()

            def progress():
                phase = budget.phase
                if phase not in seen:
                    seen.add(phase)
                    record(executing=phase, deadline=budget.expires, started=budget.started)
                    if mode == "slow_" + phase:
                        time.sleep(budget.remaining() + 0.05)  # Interrupt an active real SQLite VM.
                return 0

            def statement_started(statement):
                record(path=path.name, statement=statement)
                if budget.phase == "sql" and "sql" not in seen and not mode.startswith("slow_"):
                    seen.add("sql")
                    record(executing="sql", deadline=budget.expires, started=budget.started)

            if mode.startswith("slow_"):
                conn.set_progress_handler(progress, 1)
            conn.set_trace_callback(statement_started)
            conn.create_function("stall", 0, lambda: time.sleep(20))  # noninterruptible until return
            if budget.phase == "snapshot_check" and mode == "writer_after_copy":
                record(ready="snapshot_check")
                time.sleep(0.4)
            yield Probe(conn, budget)

    @contextmanager
    def source(root, db, budget):
        if mode == "slow_acquire":
            time.sleep(budget.remaining() + 0.05)
        with real_source(root, db, budget) as tier:
            record(ready="source", tier=tier)
            yield tier

    def enter(self, phase):
        real_enter(self, phase)
        if mode == "cumulative":
            record(entered=phase)
            time.sleep(0.75)  # Each phase fits 2 s; their sum does not.

    def result(request):
        answer = real_result(request)
        if mode.startswith("tardy_"):
            # A contended host: the source is released, the typed answer is still on its way.
            time.sleep(1.0)
            record(delivered=time.monotonic())
        return answer

    sql._connection, sql._source, sql._Budget.enter, sql._result = connection, source, enter, result
    sql.main()


def instrument(ctx, monkeypatch, mode="trace"):
    from ouroboros import process_custody
    original_spawn, original_temp = process_custody.spawn_supervised, sql.tempfile.TemporaryDirectory
    children, directories = [], []
    trace = Path(ctx.drive_root) / (mode + ".jsonl")

    def spawn(cmd, **kwargs):
        script = f"from tests.test_query_data_sql_phases import _child_main; _child_main({mode!r}, {str(trace)!r})"
        proc = original_spawn([sys.executable, "-c", script, *cmd[3:]], **kwargs)
        children.append(proc)
        return proc

    def temporary(**kwargs):
        directory = original_temp(**kwargs)
        directories.append(Path(directory.name))
        return directory

    monkeypatch.setattr(process_custody, "spawn_supervised", spawn)
    monkeypatch.setattr(sql.tempfile, "TemporaryDirectory", temporary)
    return children, directories, trace


def read_trace(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def assert_reaped(children, directories):
    assert children and all(proc.poll() is not None for proc in children)
    assert all(proc.stdin is None and proc.stdout.closed for proc in children)
    assert directories and all(not path.exists() for path in directories)


def assert_writable(ctx):
    with usage_store.hold(ctx.drive_root, timeout_sec=0.5) as txn:
        txn.conn.execute("INSERT OR REPLACE INTO meta VALUES ('query-test-writer', 'true')")


def test_mature_money_default_is_direct_and_explicit_snapshot_is_equivalent(money, monkeypatch):
    mature(money)
    children, directories, trace = instrument(money, monkeypatch)
    query = "SELECT count(*), sum(CAST(cost_usd AS REAL)) FROM attempts"
    default = call(money, query)
    assert default["status"] == "ok" and default["source"]["snapshot"] is False, default
    assert default["data"]["rows"] == [[12000, 1500.0]]
    direct_trace = read_trace(trace)
    assert not any("VACUUM" in row.get("statement", "") or row.get("statement") == "PRAGMA quick_check"
                   for row in direct_trace)
    snapshot = call(money, query, snapshot=True, timeout_sec=5)
    assert snapshot["status"] == "ok" and snapshot["source"]["snapshot"] is True, snapshot
    assert snapshot["data"] == default["data"]
    rows = read_trace(trace)[len(direct_trace):]
    assert any("VACUUM INTO" in row.get("statement", "") for row in rows)
    assert any(row.get("statement") == "PRAGMA quick_check" for row in rows)
    assert next(row["path"] for row in rows if row.get("statement") == query) == "snapshot.sqlite"
    assert_reaped(children, directories)


@pytest.mark.parametrize("phase,snapshot", [("acquire", False), ("snapshot_copy", True),
                                              ("snapshot_check", True), ("sql", False), ("sql", True)])
def test_owned_phase_expiry_is_not_store_damage(money, monkeypatch, phase, snapshot):
    children, directories, _ = instrument(money, monkeypatch, "slow_" + phase)
    result = call(money, "SELECT count(*) FROM sqlite_master", snapshot=snapshot, timeout_sec=2)
    assert result["status"] == "error" and result["code"] == "deadline" and result["phase"] == phase, result
    assert result["timeout_sec"] == 2 and result["elapsed_sec"] >= 2, result
    assert_writable(money)
    assert_reaped(children, directories)


def test_copy_check_and_sql_share_one_operation_budget(money, monkeypatch):
    children, directories, trace = instrument(money, monkeypatch, "cumulative")
    result = call(money, "SELECT 1", snapshot=True, timeout_sec=2)
    # The copy released the source, so the typed expiry has the response margin, not the source bound.
    assert result["code"] == "deadline" and result["phase"] in {"snapshot_check", "sql"}, result
    assert [row["entered"] for row in read_trace(trace) if "entered" in row][:2] == ["snapshot_copy", "snapshot_check"]
    assert 2 <= result["elapsed_sec"] < 4
    assert_reaped(children, directories)


@pytest.mark.parametrize("mode,phase,exception", [
    ("copy_io", "snapshot_copy", "OSError"),
    ("copy_corrupt", "snapshot_copy", "UsageLedgerCorrupt"),
    ("late_copy_io", "snapshot_copy", "OSError"),
    ("late_copy_corrupt", "snapshot_copy", "UsageLedgerCorrupt"),
    # Corruption found just after expiry, whose answer then arrives later than FINISH_SEC.
    ("tardy_copy_corrupt", "snapshot_copy", "UsageLedgerCorrupt"),
    ("copy_unowned_interrupt", "snapshot_copy", "UsageAccountingError"),
    ("bad_check", "snapshot_check", "ValueError"),
])
def test_snapshot_failure_is_honest_and_never_falls_back(money, monkeypatch, mode, phase, exception):
    children, directories, trace = instrument(money, monkeypatch, mode)
    # Acquisition headroom on a saturated host; late modes still fail just after their own expiry.
    result = call(money, "SELECT 73491", snapshot=True, timeout_sec=4)
    assert result["code"] == "query_failed" and result["phase"] == phase, result
    assert result["exception_class"] == exception, result
    rows = read_trace(trace)
    assert not any(row.get("statement") == "SELECT 73491" for row in rows)
    if mode.startswith("tardy_"):  # Delivered after the bound for a child that still held its source.
        expires = next(row["expires"] for row in rows if "expires" in row)
        assert next(row["delivered"] for row in rows if "delivered" in row) > expires + sql.FINISH_SEC
    assert_writable(money)
    assert_reaped(children, directories)


@pytest.mark.parametrize("money", ["PERSIST", "name", "WAL"], indirect=True)
def test_parent_kills_uninterruptible_sql_at_operation_deadline(money, monkeypatch):
    children, directories, trace = instrument(money, monkeypatch)
    result = call(money, "SELECT stall()", timeout_sec=2)
    finished = time.monotonic()
    assert result["code"] == "deadline" and result["phase"] == "executor", result
    rows = read_trace(trace)
    active = next(row for row in rows if row.get("executing") == "sql")
    # Exclude measured child startup; the original unconditional +5 s SQL wait
    # fails this bound, and so would RESPONSE_SEC for a child still holding its source.
    assert sql.FINISH_SEC + 1 <= sql.RESPONSE_SEC
    assert finished - active["started"] < 2 + sql.FINISH_SEC + 1
    assert any(row.get("statement") == "SELECT stall()" for row in rows)
    assert_writable(money)
    assert_reaped(children, directories)


def test_current_snapshot_copy_uses_source_bound_not_full_operation_budget(money, monkeypatch):
    children, directories, trace = instrument(money, monkeypatch, "stall_copy")
    registry = ToolRegistry(repo_dir=money.repo_dir, drive_root=money.drive_root)
    registry.set_context(money)
    result = json.loads(registry.execute_result("query_data", {
        "op": "sql", "db": "current", "query": "SELECT 1", "snapshot": True, "timeout_sec": 9,
    }).text)
    finished = time.monotonic()
    assert result["code"] == "deadline" and result["phase"] == "executor", result
    active = next(row for row in read_trace(trace) if row.get("executing") == "snapshot_copy")
    assert finished - active["started"] < sql.CURRENT_HOLD_SEC + sql.FINISH_SEC + 1
    store.update_task(money.drive_root, "after-read", lambda _: {"status": "completed"})
    assert_reaped(children, directories)


@pytest.mark.parametrize("money", ["PERSIST", "name"], indirect=True)
def test_response_margin_follows_source_release_and_holds_nothing(money, monkeypatch):
    children, directories, trace = instrument(money, monkeypatch)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(call, money, "SELECT stall()", snapshot=True, timeout_sec=2)
        end = time.monotonic() + 10
        while not trace.exists() or not any(row.get("executing") == "sql" for row in read_trace(trace)):
            assert time.monotonic() < end and not future.done()
            time.sleep(0.01)
        active = next(row for row in read_trace(trace) if row.get("executing") == "sql")
        # Past the bound a child holding its source would get, inside the released child's margin.
        time.sleep(max(0.0, active["started"] + 2 + sql.FINISH_SEC + 0.3 - time.monotonic()))
        assert not future.done()
        writing = time.monotonic()
        assert_writable(money)  # The stuck snapshot query blocks no writer: it holds no source.
        assert time.monotonic() - writing < 1 and not future.done()
        result = future.result(timeout=10)
    finished = time.monotonic()
    assert result["code"] == "deadline" and result["phase"] == "executor", result
    assert 2 + sql.RESPONSE_SEC <= finished - active["started"] < 2 + sql.RESPONSE_SEC + 1
    assert_reaped(children, directories)


@pytest.mark.parametrize("journal", ["PERSIST", "name", "WAL"])
def test_snapshot_query_releases_source_before_check_and_writer_progresses(tmp_path, monkeypatch, journal):
    from ouroboros import platform_layer
    tier = store.TIER_NAME if journal == "name" else store.TIER_ENFORCED
    if journal == "WAL" and store._journal_mode(tmp_path, tier) != "WAL":
        pytest.skip("this SQLite/filesystem does not support the store's WAL policy")
    if journal == "PERSIST":
        monkeypatch.setattr(store, "_journal_mode", lambda *_a: "PERSIST")
    monkeypatch.setattr(platform_layer, "kernel_file_locks_enforced", lambda *_a, **_k: tier == store.TIER_ENFORCED)
    usage_store.migrate_from_journal(tmp_path)
    with usage_store.read(tmp_path) as txn:
        assert txn.conn.execute("PRAGMA journal_mode").fetchone()[0] == ("wal" if journal == "WAL" else "persist")
    store_import.ensure_current_store(tmp_path, with_children=False)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="caller")
    children, directories, trace = instrument(ctx, monkeypatch, "writer_after_copy")
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(call, ctx, "SELECT 1", snapshot=True, timeout_sec=2)
            end = time.monotonic() + 5
            while not trace.exists() or not any(row.get("ready") == "snapshot_check" for row in read_trace(trace)):
                assert time.monotonic() < end
                time.sleep(0.01)
            assert_writable(ctx)  # While the checked snapshot is still in flight.
            assert not future.done()
            result = future.result(timeout=5)
        assert result["status"] == "ok", result
        assert_reaped(children, directories)
    finally:
        store.close_connections_under(tmp_path)
        usage_store.forget(tmp_path)


def test_real_name_lock_acquisition_deadline_releases_guard(tmp_path, monkeypatch):
    from ouroboros import platform_layer
    monkeypatch.setattr(platform_layer, "kernel_file_locks_enforced", lambda *_a, **_k: False)
    usage_store.migrate_from_journal(tmp_path)
    store_import.ensure_current_store(tmp_path, with_children=False)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="caller")
    try:
        with store.name_lock(tmp_path, usage_store.LOCK_REL.name, timeout_sec=1):
            result = call(ctx, "SELECT 1", timeout_sec=0.15)
            assert result["code"] == "deadline" and result["phase"] == "acquire", result
        assert_writable(ctx)
        assert call(ctx, "SELECT 2")["data"]["rows"] == [[2]]
    finally:
        store.close_connections_under(tmp_path)
        usage_store.forget(tmp_path)


def test_money_sql_preserves_limits_and_authorizer_before_outer_select(money):
    for query in ("ATTACH DATABASE ':memory:' AS other", "DELETE FROM attempts", "PRAGMA query_only=OFF",
                  "SELECT load_extension('none')", "SELECT 1; SELECT 2"):
        result = call(money, query)
        assert result["code"] == "query_failed" and result["phase"] == "sql", result
    truncated = call(money, "SELECT printf('%010000d',1)", select="rows", max_bytes=256)
    assert truncated["data"] == []  # Outer select cannot restore bytes truncated during fetch.
    rows = call(money, "SELECT 1 UNION ALL SELECT 2", max_rows=1)
    assert rows["data"]["rows"] == [[1]] and rows["data"]["truncated"]
    limited = call(money, "SELECT zeroblob(419430400)")
    assert limited["code"] == "memory_limit" and limited["phase"] == "sql", limited
    for kwargs in ({"timeout_sec": 11}, {"max_rows": 10001}, {"max_bytes": sql.MAX_BYTES + 1}, {"db": "../other"}):
        registry = ToolRegistry(repo_dir=money.repo_dir, drive_root=money.drive_root)
        registry.set_context(money)
        result = json.loads(registry.execute_result("query_data", dict(op="sql", query="SELECT 1", **kwargs)).text)
        assert result["status"] == "error", result
    assert_writable(money)


def test_deadline_does_not_mask_corrupt_source(money):
    path = Path(money.drive_root) / usage_store.STORE_REL
    store.close_connections(path)
    usage_store.forget(money.drive_root)
    path.write_bytes(b"not a SQLite money store")
    result = call(money, "SELECT 1", timeout_sec=2)
    assert result["code"] == "query_failed" and result["phase"] == "acquire", result
    assert result["exception_class"] == "UsageLedgerCorrupt", result


def test_supported_wal_money_query_reads_uncheckpointed_receipt(money):
    path = Path(money.drive_root) / usage_store.STORE_REL
    if store._journal_mode(path, store.TIER_ENFORCED) != "WAL":
        pytest.skip("this SQLite/filesystem does not support the store's WAL policy")
    # Lifecycle owner, not the query, chooses WAL on this isolated store.
    usage_store.migrate_from_journal(money.drive_root, initialize_journal=True)
    with usage_store.hold(money.drive_root) as txn:
        assert txn.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        txn.conn.execute("PRAGMA wal_autocheckpoint=0")
        txn.conn.execute("INSERT OR REPLACE INTO meta VALUES ('wal-receipt', '73491')")
    assert Path(str(path) + "-wal").stat().st_size > 0
    for snapshot in (False, True):
        result = call(money, "SELECT value FROM meta WHERE key='wal-receipt'", snapshot=snapshot)
        assert result["status"] == "ok" and result["data"]["rows"] == [["73491"]], result


def test_money_admission_narrows_every_wait_to_remaining_operation_time(tmp_path, monkeypatch):
    from ouroboros import platform_layer
    monkeypatch.setattr(platform_layer, "kernel_file_locks_enforced", lambda *_a, **_k: False)
    usage_store.migrate_from_journal(tmp_path)
    usage_store.forget(tmp_path)  # Cold tier/meta path as well as the outer hold.
    seen, allowance = [], [1.0]
    real_connection, real_lock = store.connection, usage_store._named_lock

    def remaining():
        allowance[0] -= 0.05
        return allowance[0]

    @contextmanager
    def connection(path, tier, **kwargs):
        seen.append(("connection", kwargs["wait_sec"]))
        with real_connection(path, tier, **kwargs) as conn:
            yield conn

    @contextmanager
    def lock(*args, **kwargs):
        seen.append(("name_lock", kwargs["timeout_sec"]))
        with real_lock(*args, **kwargs):
            yield

    monkeypatch.setattr(store, "connection", connection)
    monkeypatch.setattr(usage_store, "_named_lock", lock)
    try:
        with usage_store.hold(tmp_path, write=False, migrate=False, timeout_sec=1, remaining_sec=remaining):
            pass
        assert [kind for kind, _ in seen] == ["name_lock", "connection", "name_lock", "connection"]
        waits = [wait for _, wait in seen]
        assert all(0 < wait < 1 for wait in waits)
        assert all(later < earlier for earlier, later in zip(waits, waits[1:]))
    finally:
        store.close_connections_under(tmp_path)
        usage_store.forget(tmp_path)
