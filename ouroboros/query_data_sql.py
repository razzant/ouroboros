"""Bounded, supervised SQL reads; SQLite's heap limit belongs only to the child.

Store owners supply lifecycle admission and name locks. Ordinary money queries
use that guarded mode=ro path; an explicit snapshot copies and checks the store.
One child operation budget covers acquisition, copy, check and SQL. The parent
owns the disposable directory, which also carries the whole request to the child.
It bounds startup and the final response separately, then reaps even a SQLite
function which does not yield to interrupt.
"""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time

from ouroboros import store

MAX_BYTES = 8 * 1024 * 1024
HEAP_BYTES = 134217728
PRAGMAS = {"table_info", "table_xinfo", "table_list", "index_list", "index_info"}
# Executor lifecycle allowances, never SQL phase budgets. After the operation deadline the parent
# waits FINISH_SEC while the child may still hold its source, so a stuck reader is reaped promptly;
# once the child has released it, RESPONSE_SEC lets a contended host deliver the typed answer.
STARTUP_SEC = 5.0
FINISH_SEC = 0.5
RESPONSE_SEC = 2.0


# A current-store read holds the lock its writers' COMMIT waits on for store.STORE_WAIT_SEC: the hold
# stays within half of that wait, so a reader can delay a task write but never fail it.
CURRENT_HOLD_SEC = store.STORE_WAIT_SEC / 2


class QueryDeadline(RuntimeError):
    """An expired budget owned by this query, not a store-health failure."""

    def __init__(self, budget):
        super().__init__(f"query deadline expired during {budget.phase}")
        self.phase = budget.phase
        self.timeout_sec = budget.seconds
        self.elapsed_sec = time.monotonic() - budget.started


class _Budget:
    def __init__(self, seconds):
        self.seconds, self.started = seconds, time.monotonic()
        self.expires = self.end = self.started + seconds
        self.phase = "acquire"

    def remaining(self):
        remaining = self.end - time.monotonic()
        if remaining <= 0:
            raise QueryDeadline(self)
        return remaining

    def enter(self, phase):
        self.remaining()  # An overrun belongs to the phase that just spent it.
        self.phase = phase

    def sqlite_failure(self, exc, interrupted):
        # Python 3.10 has neither sqlite_errorcode nor sqlite3.SQLITE_*: there SQLite's fixed
        # message names its interrupt (code 9); store.busy reads busy/locked either way.
        code = getattr(exc, "sqlite_errorcode", None)
        ours = interrupted.is_set() and (str(exc) == "interrupted" if code is None else (code & 255) == 9)
        if ours or store.busy(exc) and time.monotonic() >= self.end:
            raise QueryDeadline(self) from exc
        # Corruption, I/O, and unowned interrupts remain failures, even near expiry.


def database_path(root: Path, db: str) -> Path:
    from ouroboros import evidence_packs
    return {"current": store.store_path(root), "money": root / "state" / "usage.sqlite",
            "packs": evidence_packs.packs_dir(root) / evidence_packs.INDEX_NAME}[db]


@contextlib.contextmanager
def _source(root, db, budget):
    if db == "current":
        with store.open_store(root, write=False, wait_sec=budget.remaining()) as txn:
            if txn.phase() != store.PUBLISHED:
                raise store.StoreRetired("current store is not published")
            budget.remaining()
            yield store.probe(root)[1]  # the owner's header protocol; metadata also carries mount identity
    elif db == "money":
        from ouroboros import usage_store
        try:
            with usage_store.hold(root, write=False, migrate=False, timeout_sec=budget.remaining(),
                                  remaining_sec=budget.remaining):
                budget.remaining()
                yield usage_store.store_tier(root, wait_sec=budget.remaining(), migrate=False,
                                             remaining_sec=budget.remaining)
        except usage_store.UsageLockUnavailable as exc:
            if exc.reason == "contention":
                budget.remaining()
            raise
    else:
        from ouroboros import evidence_packs
        # The owner rebuilds the cache and holds its one lock through the SQL read.
        with evidence_packs.open_index(root, wait_sec=budget.remaining()):
            budget.remaining()
            yield evidence_packs.index_tier(root)
    budget.remaining()


@contextlib.contextmanager
def _connection(path, tier, budget):
    uri = path.as_uri() + "?mode=ro"
    if tier == store.TIER_NAME:
        uri += "&vfs=" + ("win32-none" if os.name == "nt" else "unix-none")
    conn = sqlite3.connect(uri, uri=True, timeout=budget.remaining(), isolation_level=None)
    interrupted = threading.Event()

    def interrupt():
        interrupted.set()
        conn.interrupt()

    timer = threading.Timer(max(0.0, budget.end - time.monotonic()), interrupt)
    try:
        budget.remaining()
        timer.start()
        conn.execute(f"PRAGMA hard_heap_limit={HEAP_BYTES}").close()
        yield conn
        budget.remaining()
    except (sqlite3.Error, sqlite3.Warning) as exc:
        # Convert our interrupt BEFORE usage_store.hold can translate SQLite errors.
        budget.sqlite_failure(exc, interrupted)
        if budget.phase == "sql":
            # User SQL must not cross admission as a store-health failure.
            raise ValueError(str(exc)) from exc
        raise
    finally:
        timer.cancel()
        if timer.ident is not None:
            timer.join()
        conn.close()


def _authorizer(db):
    def authorize(action, first, second, database, origin):
        allowed = action in {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE}
        if action == sqlite3.SQLITE_READ:
            allowed = db != "current" or first in {"tasks", "sqlite_master", "sqlite_schema"}
        elif action == sqlite3.SQLITE_FUNCTION:
            allowed = str(second).lower() != "load_extension"
        elif action == sqlite3.SQLITE_PRAGMA:
            allowed = str(first).lower() in PRAGMAS
        return sqlite3.SQLITE_OK if allowed else sqlite3.SQLITE_DENY
    return authorize


def _fetch(conn, request):
    conn.execute("PRAGMA query_only=ON").close()
    conn.set_authorizer(_authorizer(request["db"]))
    cursor = conn.cursor()
    try:
        cursor.execute(request["query"])
        result = {"columns": [col[0] for col in cursor.description or ()],
                  "rows": [], "truncated": False}
        used = len(json.dumps(result, ensure_ascii=False).encode("utf-8")) + 32
        if used > request["max_bytes"]:
            raise ValueError("byte_limit: column names exceed max_bytes")
        for row in cursor:
            values = [{"hex": value.hex()} if isinstance(value, bytes) else value for value in row]
            size = len(json.dumps(values, ensure_ascii=False).encode("utf-8")) + 2
            if len(result["rows"]) >= request["max_rows"] or used + size > request["max_bytes"]:
                result["truncated"] = True
                break
            result["rows"].append(values)
            used += size
        return result
    finally:
        cursor.close()


def _execute(request):
    root, db, scratch = Path(request["root"]), request["db"], Path(request["scratch"])
    path = database_path(root, db)  # Loading the source owners is startup, not operation time.
    if db == "money":
        from ouroboros import usage_store  # noqa: F401
    budget = _Budget(request["timeout_sec"])
    try:
        if request.get("supervised"):
            # The parent's hard backstop starts from this actual child operation,
            # not spawn time. Publish atomically: a partial numeric prefix is valid.
            marker = scratch / "deadline.tmp"
            marker.write_text(str(budget.expires), encoding="ascii")
            marker.replace(marker.with_name("deadline"))
        if db == "current":
            budget.end = min(budget.expires, budget.started + CURRENT_HOLD_SEC)
        try:
            with _source(root, db, budget) as tier:
                with _connection(path, tier, budget) as conn:
                    if not request["snapshot"]:
                        budget.enter("sql")
                        return _fetch(conn, request)
                    budget.enter("snapshot_copy")
                    target = scratch / "snapshot.sqlite"
                    # Trusted SQL only: user SQL never runs on this connection.
                    conn.execute("VACUUM INTO ?", (str(target),)).close()
        finally:
            if request.get("supervised"):
                with contextlib.suppress(OSError):  # Without it the parent keeps the source bound.
                    (scratch / "released").touch()
        budget.remaining()
        budget.end = budget.expires  # A snapshot's own query holds no source lock.
        budget.enter("snapshot_check")
        with _connection(target, store.TIER_ENFORCED, budget) as conn:
            cursor = conn.execute("PRAGMA quick_check")
            try:
                if cursor.fetchall() != [("ok",)]:
                    raise ValueError("snapshot quick_check failed")
            finally:
                cursor.close()
            budget.enter("sql")
            return _fetch(conn, request)
    except (sqlite3.Error, OSError, ValueError, RuntimeError, MemoryError) as exc:
        if isinstance(exc, store.StoreBusy) and exc.reason == "contention":
            budget.remaining()
        exc.phase = budget.phase
        raise


def _communicate(proc, scratch, seconds, *, source_seconds=None):
    """Bound startup separately; then the child's one deadline plus the margin its source state allows."""
    startup_end, operation_end = time.monotonic() + STARTUP_SEC, None
    source_seconds = seconds if source_seconds is None else min(seconds, source_seconds)
    while True:
        if operation_end is None:
            try:
                operation_end = float(Path(scratch, "deadline").read_text(encoding="ascii"))
            except (FileNotFoundError, ValueError):
                pass  # The child has not published its start (a partial write is not a fact).
        if operation_end is None:
            end = startup_end
        elif Path(scratch, "released").exists():
            end = operation_end + RESPONSE_SEC
        else:
            end = operation_end - seconds + source_seconds + FINISH_SEC
        remaining = end - time.monotonic()
        if remaining <= 0:
            if proc.poll() is not None:  # Late parent scheduling must not replace an already produced answer.
                return proc.communicate()[0]
            raise subprocess.TimeoutExpired(proc.args, seconds)
        try:
            # Output only: retrying communicate() keeps what was read on every supported Python.
            return proc.communicate(timeout=min(remaining, 0.05))[0]
        except subprocess.TimeoutExpired:
            pass


def run_sql(ctx, root, *, db, query, max_rows, max_bytes, snapshot, timeout_sec):
    """Only host-resolved sources cross the process boundary, never an SQL path."""
    from ouroboros.process_containment import CONTAINMENT_ENV_PREFIX
    from ouroboros.process_custody import current_custody_session_id, lifeline_parent_env, spawn_supervised

    repo = Path(__file__).resolve().parents[1]
    env = {key: value for key, value in os.environ.items()
           if key.upper() in {"PATH", "HOME", "USERPROFILE", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "TMPDIR",
                      "LANG", "LC_ALL", "OUROBOROS_SETTINGS_PATH"}
           or key.startswith(CONTAINMENT_ENV_PREFIX)}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(repo),
               OUROBOROS_DATA_DIR=str(root), OUROBOROS_REPO_DIR=str(repo), **lifeline_parent_env())
    with tempfile.TemporaryDirectory(prefix="query-data-") as scratch:
        request = dict(root=str(root), db=db, query=query, max_rows=max_rows, max_bytes=max_bytes,
                       snapshot=snapshot, timeout_sec=timeout_sec, scratch=scratch, supervised=True)
        # Written whole before the child exists: CPython <= 3.12 never finishes stdin input
        # that a timed-out communicate() left unsent, and a request can exceed any pipe buffer.
        request_path = Path(scratch, "request.json")
        request_path.write_text(json.dumps(request), encoding="utf-8")
        proc = spawn_supervised(
            [sys.executable, "-m", "ouroboros.query_data_sql", current_custody_session_id(), str(request_path)],
            drive_root=Path(ctx.drive_root), purpose="query_data", scope="task",
            owner_task_id=str(ctx.task_id or ""), cwd=str(repo), env=env,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            raw = _communicate(proc, scratch, timeout_sec,
                               source_seconds=CURRENT_HOLD_SEC if db == "current" else timeout_sec)
            if proc.returncode or len(raw) > max_bytes + 4096:
                return {"status": "error", "code": "executor_failed", "phase": "executor"}
            return json.loads(raw)
        except subprocess.TimeoutExpired:
            # No observation of the precise phase: never call an executor kill SQL failure.
            return {"status": "error", "code": "deadline", "phase": "executor"}
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            proc.stdout.close()


def _result(request):
    try:
        return {"status": "ok", "data": _execute(request)}
    except QueryDeadline as exc:
        return {"status": "error", "code": "deadline", "phase": exc.phase,
                "timeout_sec": exc.timeout_sec, "elapsed_sec": exc.elapsed_sec}
    except MemoryError as exc:
        return {"status": "error", "code": "memory_limit", "exception_class": "MemoryError", "phase": getattr(exc, "phase", "acquire")}
    except (sqlite3.Error, OSError, ValueError, RuntimeError) as exc:
        return {"status": "error", "code": "query_failed", "exception_class": type(exc).__name__,
                "message": str(exc), "phase": getattr(exc, "phase", "acquire")}


def main():
    from ouroboros.process_custody import adopt_session_id, start_parent_lifeline
    adopt_session_id(sys.argv[1])
    start_parent_lifeline(label="query_data")
    request = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    # UTF-8 on the wire whatever the console code page: a Windows pipe defaults to cp1252.
    sys.stdout.buffer.write((json.dumps(_result(request), ensure_ascii=False) + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
