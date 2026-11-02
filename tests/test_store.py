"""The current-facts store core (``ouroboros/store.py``): the SQLite core money imports
back, the tri-state open, the lock tiers, the phase inside the write seam, the document
round trip and the SQLite 3.24 floor the CI floor job proves on a 3.37 system engine."""
from __future__ import annotations

import contextlib
import errno
import os
import pathlib
import re
import shutil
import sqlite3

import pytest

from ouroboros import platform_layer, store, task_store, usage_store
from ouroboros.store import StoreBusy, StoreLost, StoreMigrating, StoreRetired
from ouroboros.utils import machine_json
from tests._store_shared import release_store_handles

REPO = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def release_connections():
    store.close_connections()
    yield
    store.close_connections()


def publish(root: pathlib.Path, *documents: dict, tier: str = store.TIER_ENFORCED,
            phase: str = store.PUBLISHED) -> pathlib.Path:
    """A published store with ``documents``, built as the importer builds one."""
    path = store.store_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = store.connect(path, tier, create=True)
    try:
        conn.execute("PRAGMA synchronous = FULL")
        conn.executescript(store._SCHEMA)
        conn.execute(f"PRAGMA application_id = {store._APPLICATION_ID[tier]}")
        conn.execute("BEGIN IMMEDIATE")
        txn = store.Txn(conn)
        for document in documents:
            txn.store_task(document)
        txn.set_meta("schema_version", store.SCHEMA_VERSION)
        txn.set_meta("lock_tier", tier)
        txn.set_meta("phase", phase)
        store.commit(conn)
    finally:
        conn.close()
    return path


def document(task_id: str, status: str = "completed", **fields) -> dict:
    return {"task_id": task_id, "status": status, "ts": "2026-10-10T00:00:00+00:00",
            "_schema_version": 1, **fields}


# ---- the core money imports back ----------------------------------------------

def test_money_runs_on_the_shared_core():
    assert usage_store.TIER_NAME is store.TIER_NAME and usage_store.TIER_ENFORCED is store.TIER_ENFORCED
    assert usage_store._identity is store.file_identity
    assert usage_store._process_lock is store.process_lock
    assert usage_store._fsync_directory is store.fsync_directory
    assert usage_store._busy is store.busy


def test_the_two_stores_name_their_own_lock_protocol_in_the_header(tmp_path):
    facts = publish(tmp_path)
    usage_store.migrate_from_journal(tmp_path)  # an empty money store on the same root
    money = tmp_path / usage_store.STORE_REL
    assert store.header_tier(facts, store._APPLICATION_ID) == store.TIER_ENFORCED
    assert store.header_tier(money, store._APPLICATION_ID) is None  # money's header is not ours
    assert usage_store._header_tier(money) == store.TIER_ENFORCED
    assert usage_store._header_tier(facts) is None
    assert store.header_tier(tmp_path / "missing.sqlite", store._APPLICATION_ID) is None


def test_money_errors_keep_their_vocabulary_over_the_shared_core(tmp_path):
    from ouroboros.usage_ledger import UsageAccountingError

    with pytest.raises(UsageAccountingError, match="usage store unavailable"):
        usage_store._connect(tmp_path / "state" / "absent.sqlite", store.TIER_ENFORCED)


# ---- the tri-state open ---------------------------------------------------------

def test_a_missing_store_is_migrating_while_imported_and_lost_beside_its_stamp(tmp_path):
    """A never-imported root holding facts is imported on the first public WRITE; a read of it
    answers ``StoreMigrating`` and creates nothing (the boot door owns the import); a root
    holding no facts reads empty; inside the lifespan's import window every other opener is
    told ``StoreMigrating``."""
    from ouroboros.utils import atomic_write_json

    assert store.probe(tmp_path) == ("absent", None)
    with store.open_store(tmp_path, write=False) as txn:  # nothing to import: empty, nothing created
        assert txn.load("t1") is None
    assert store.migrating(tmp_path) is False and not store.store_path(tmp_path).exists()
    atomic_write_json(tmp_path / "task_results" / "t1.json", document("t1"))
    with pytest.raises(StoreMigrating):  # a read never imports the facts a root holds
        with store.open_store(tmp_path, write=False):
            pass
    assert store.migrating(tmp_path) is True and not store.store_path(tmp_path).exists()  # the predicate opened nothing for write
    with store.importing(tmp_path):
        with pytest.raises(StoreMigrating):
            with store.open_store(tmp_path, write=True):
                pass
        assert not store.store_path(tmp_path).exists()  # mode=rw never creates
        assert store.store_ready(tmp_path) is False
    with store.open_store(tmp_path, write=True) as txn:  # first touch: the import ran, the store is published
        assert txn.load("t1")["status"] == "completed"
    assert store.probe(tmp_path)[0] == "present" and store.store_ready(tmp_path) is True
    release_store_handles(store.store_path(tmp_path))  # this test's own reads keep it open (Windows)
    store.store_path(tmp_path).unlink()
    assert store.generation_stamped(tmp_path)  # the import stamped the generation: a missing store is now LOST
    with pytest.raises(StoreLost):
        with store.open_store(tmp_path, write=True):
            pass
    assert not store.store_path(tmp_path).exists()


@pytest.mark.parametrize("code", [errno.EMFILE, errno.ENFILE])
def test_an_exhausted_descriptor_table_is_not_absence(tmp_path, monkeypatch, code):
    """The #1307 rule: the process that cannot open files must not conclude the store is
    missing (and then create or rebuild one); the same stat failing with ENOENT is absence."""
    path = store.store_path(tmp_path)
    real_stat = os.stat

    def failing(target, *args, **kwargs):
        if pathlib.Path(target) == path:
            raise OSError(code, os.strerror(code))
        return real_stat(target, *args, **kwargs)

    monkeypatch.setattr(os, "stat", failing)
    with pytest.raises(StoreBusy, match=errno.errorcode[code]):
        store.probe(tmp_path)
    with pytest.raises(StoreBusy):
        with store.open_store(tmp_path, write=True):
            pass
    monkeypatch.setattr(os, "stat", real_stat)
    assert store.probe(tmp_path) == ("absent", None)  # the quiet direction: ENOENT is absence
    assert not path.exists()


def test_a_present_foreign_file_is_lost_never_replaced(tmp_path):
    path = store.store_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not a database at all, but someone's bytes" * 4)
    with pytest.raises(StoreLost):
        store.probe(tmp_path)
    with pytest.raises(StoreLost):
        with store.open_store(tmp_path, write=True):
            pass
    assert path.read_bytes().startswith(b"not a database")


def test_a_published_but_unstamped_store_is_accepted_by_its_own_meta(tmp_path):
    publish(tmp_path, document("t1"))
    assert not store.generation_stamped(tmp_path)
    assert store.probe(tmp_path) == ("present", store.TIER_ENFORCED)
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.phase() == store.PUBLISHED
        assert txn.task_ids() == ["t1"]


# ---- lock tiers -----------------------------------------------------------------

def test_the_name_tier_opens_without_sqlite_locks_and_takes_the_name_lock_on_every_access(tmp_path, monkeypatch):
    publish(tmp_path, document("t1"), tier=store.TIER_NAME)
    assert store.probe(tmp_path) == ("present", store.TIER_NAME)
    uris, locks = [], []
    real_connect, real_acquire = sqlite3.connect, platform_layer.acquire_exclusive_file_lock

    def spy_connect(database, *args, **kwargs):
        uris.append(str(database))
        return real_connect(database, *args, **kwargs)

    def spy_acquire(path, **kwargs):
        locks.append(pathlib.Path(path).name)
        return real_acquire(path, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", spy_connect)
    monkeypatch.setattr(platform_layer, "acquire_exclusive_file_lock", spy_acquire)
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.load("t1")["task_id"] == "t1"
    store.update_task(tmp_path, "t1", lambda current: {**current, "status": "running"})
    assert locks == [store.LOCK_NAME, store.LOCK_NAME]
    assert all("vfs=unix-none" in uri or "vfs=win32-none" in uri for uri in uris)


def test_the_enforced_tier_uses_sqlite_locks_only(tmp_path, monkeypatch):
    publish(tmp_path, document("t1"))
    locks = []
    real_acquire = platform_layer.acquire_exclusive_file_lock
    monkeypatch.setattr(platform_layer, "acquire_exclusive_file_lock",
                        lambda path, **kwargs: locks.append(pathlib.Path(path).name) or real_acquire(path, **kwargs))
    with store.open_store(tmp_path, write=True) as txn:
        txn.set_meta("probe", 1)
    assert store.LOCK_NAME not in locks


def test_a_held_name_lock_is_busy_never_a_hang(tmp_path):
    with store.name_lock(tmp_path, store.LOCK_NAME, timeout_sec=0.0):
        with pytest.raises(StoreBusy):
            with store.name_lock(tmp_path, store.LOCK_NAME, timeout_sec=0.0):
                pass
    with store.name_lock(tmp_path, store.LOCK_NAME, timeout_sec=0.0):
        pass  # released: the quiet direction


def test_contention_is_busy_typed(tmp_path):
    publish(tmp_path, document("t1"))
    holder = store.connect(store.store_path(tmp_path), store.TIER_ENFORCED)
    try:
        holder.execute("BEGIN IMMEDIATE")
        with pytest.raises(StoreBusy):
            with store.open_store(tmp_path, write=True, wait_sec=0.0):
                pass
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    with store.open_store(tmp_path, write=True, wait_sec=0.0) as txn:  # free again: quiet
        txn.set_meta("probe", 2)


# ---- the phase inside the write seam ---------------------------------------------

def test_update_task_refuses_once_the_export_began_and_writes_nothing(tmp_path):
    publish(tmp_path, document("t1"))
    assert store.update_task(tmp_path, "t1", lambda current: {**current, "note": "before"})["note"] == "before"
    with store.open_store(tmp_path, write=True) as txn:
        txn.set_meta("phase", store.EXPORTING)
    with pytest.raises(StoreRetired):
        store.update_task(tmp_path, "t1", lambda current: {**current, "note": "after"})
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.load("t1")["note"] == "before"


def test_every_public_write_refuses_in_the_window_and_outside_published_while_service_mode_passes(tmp_path):
    """The one gate (B1): custody emit, obligation rows, a document delete and a handoff row
    all refuse typed while the import window is open for the root or the phase is not PUBLISHED;
    readiness is publication; the importer/exporter/seam's service mode passes both checks."""
    from ouroboros import delegate_custody_current as custody, obligations
    from ouroboros.task_store import delete_document, patch_handoff

    publish(tmp_path, document("t1"))
    event = {"type": "delegate_run_containment_fault", "task_id": "t1", "run_id": "r1", "event_id": "e1",
             "ts": "2026-10-10T00:00:00+00:00"}
    with store.importing(tmp_path):
        assert store.store_ready(tmp_path) is False and store.published(tmp_path) is True
        with pytest.raises(StoreMigrating):
            obligations.add(tmp_path, "custody_marks", "t1", {"task_id": "t1"})
        with pytest.raises(StoreMigrating):
            delete_document(tmp_path, "t1")
        with pytest.raises(StoreMigrating):
            patch_handoff(tmp_path, "t1", "model_waits", "op", lambda _row: {"state": "waiting"})
        assert custody.publish(tmp_path, event) is False
        with store.open_store(tmp_path, write=True, service=True) as txn:  # the importer's own mode
            txn.set_meta("probe", 1)
        with store.service_thread():  # the import owner's post-publication steps through public APIs
            obligations.add(tmp_path, "custody_marks", "t1", {"task_id": "t1"})
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.load("t1") is not None and txn.custody_rows() == {} and txn.get_row("custody_marks", "t1") == {"task_id": "t1"}
    assert store.store_ready(tmp_path) is True
    with store.open_store(tmp_path, write=True, service=True) as txn:
        txn.set_meta("phase", store.EXPORTING)
    assert store.store_ready(tmp_path) is False and store.published(tmp_path) is False  # the file exists, not ready
    with pytest.raises(StoreRetired):
        obligations.remove(tmp_path, "custody_marks", "t1")
    with pytest.raises(StoreRetired):
        delete_document(tmp_path, "t1")
    assert custody.publish(tmp_path, event) is False
    with store.open_store(tmp_path, write=True, service=True) as txn:
        txn.set_meta("phase", store.PUBLISHED)
    assert custody.publish(tmp_path, event) is True and delete_document(tmp_path, "t1") is True
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.load("t1") is None and set(txn.custody_rows()) == {"r1"}


@pytest.mark.parametrize("mode", ["PERSIST", "WAL"])
def test_a_straggler_after_the_retirement_is_told_retired_not_migrating(tmp_path, monkeypatch, mode):
    """The seam commits ``phase=retired`` into the store, clears the stamp and renames it aside.
    A writer that opened the file earlier reads that phase inside its own transaction before
    closing for the rename (required by the journal handoff on every platform); a writer
    that opens by path now (``update_task``, a reader) is told ``StoreRetired`` by the sibling,
    never «not published yet»."""
    from ouroboros.startup_migrations import stamp

    monkeypatch.setattr(store, "_journal_mode", lambda *_: mode)
    path = publish(tmp_path, document("t1"))
    retired = path.with_name(path.name + ".exported-test")
    with contextlib.closing(store.connect(path, store.TIER_ENFORCED)) as straggler:  # opened before the seam ran
        # Commit the phase before quiescing the old writer; open_store would also
        # attempt the handoff while that writer still holds its WAL connection.
        with contextlib.closing(store.connect(path, store.TIER_ENFORCED)) as retiring:
            with store.transaction(retiring, write=True):
                store.Txn(retiring).set_meta("phase", store.RETIRED)
        with store.transaction(straggler, write=True):
            assert store.Txn(straggler).phase() == store.RETIRED
    store.prepare_handoff(path, store.TIER_ENFORCED)
    stamp(tmp_path, **{store.GENERATION_KEY: None})
    os.replace(path, retired)
    with pytest.raises(StoreRetired):
        store.update_task(tmp_path, "t1", lambda current: {**current, "note": "late"})
    with pytest.raises(StoreRetired):
        with store.open_store(tmp_path, write=False):
            pass
    with contextlib.closing(sqlite3.connect(retired)) as conn:
        assert "late" not in conn.execute("SELECT body FROM task_bodies WHERE task_id='t1'").fetchone()[0]
    retired.unlink()  # no sibling, no stamp: a never-imported root again, imported on the first write
    with store.open_store(tmp_path, write=True) as txn:
        assert txn.load("t1") is None  # the retired store's rows are not the import's input


def test_a_first_touch_import_answers_a_nested_open_typed_instead_of_re_entering_itself(tmp_path, monkeypatch):
    """The import holds the import lock; a store open of the same root from inside it (a reader
    the importer's helpers reach) is told ``StoreMigrating`` rather than waiting on that lock.
    The custody probe of a never-imported root holding no facts answers empty, importing nothing."""
    from ouroboros import store_import
    from ouroboros.delegate_custody import custody_log_unreadable

    nested = []
    real = store_import._custody_rows

    def probing(root, records):
        with pytest.raises(StoreMigrating):
            with store.open_store(root, write=False):
                pass
        nested.append(str(root))
        return real(root, records)

    monkeypatch.setattr(store_import, "_custody_rows", probing)
    with store.open_store(tmp_path, write=True) as txn:
        assert txn.load("absent") is None
    assert nested == [str(tmp_path)]
    other = tmp_path / "other"
    other.mkdir()
    assert custody_log_unreadable(other) is False
    assert store.probe(other)[0] == "absent"
    monkeypatch.setattr(store_import, "_read_results", lambda root: (_ for _ in ()).throw(PermissionError("dir")))
    with pytest.raises(StoreMigrating):  # the import failing to read its inputs is typed too
        with store.open_store(tmp_path / "third", write=True):
            pass
    with store.open_store(tmp_path / "fourth", write=False) as txn:  # a root that does not exist: empty, nothing made
        assert txn.load("absent") is None and txn.set_rows("custody_open") == {}
    assert not (tmp_path / "fourth").exists()


def test_a_concurrent_first_touch_from_another_thread_waits_for_the_import_instead_of_failing(tmp_path, monkeypatch):
    """Only the importing thread's own nested open is typed: a second WRITER touching the same
    never-imported root waits on the import lock and then opens the published store; a reader
    meanwhile is told ``StoreMigrating`` at once."""
    import threading
    from ouroboros import store_import
    from ouroboros.utils import atomic_write_json

    root = tmp_path / "shared"
    root.mkdir()
    atomic_write_json(root / "task_results" / "t1.json", document("t1"))
    entered, release = threading.Event(), threading.Event()
    real = store_import._custody_rows

    def slow(root_, records):
        entered.set()
        assert release.wait(5)
        return real(root_, records)

    monkeypatch.setattr(store_import, "_custody_rows", slow)
    outcomes = {}

    def writer(name):
        try:
            with store.open_store(root, write=True) as txn:
                outcomes[name] = txn.load("t1")["task_id"]
        except store.StoreError as exc:
            outcomes[name] = type(exc).__name__

    first = threading.Thread(target=writer, args=("first",))
    first.start()
    assert entered.wait(5)  # the first thread is inside the import
    second = threading.Thread(target=writer, args=("second",))
    second.start()
    second.join(0.5)
    assert second.is_alive() and "second" not in outcomes  # waiting, not refused
    with pytest.raises(StoreMigrating):  # a reader does not wait for a store it may not import
        with store.open_store(root, write=False):
            pass
    release.set()
    first.join(5)
    second.join(5)
    assert outcomes == {"first": "t1", "second": "t1"}


# ---- documents, typed columns, obligation rows ------------------------------------

def test_update_task_round_trips_the_document_and_projects_the_typed_columns(tmp_path):
    publish(tmp_path, document("t1", parent_task_id="root", root_task_id="root", delegation_role="child"))
    assert store.update_task(tmp_path, "t1", lambda current: None) is None  # no write
    updated = store.update_task(tmp_path, "t1", lambda current: {**current, "status": "running", "extra": {"k": [1, 2.5]}})
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.load("t1") == updated
        assert txn.load("absent") is None
        row = txn.conn.execute("SELECT * FROM tasks WHERE task_id='t1'").fetchone()
        assert (row["status"], row["parent_task_id"], row["root_task_id"], row["delegation_role"]) == (
            "running", "root", "root", "child")
        assert row["pending_drive"] == 1 and row["owes_synthesis"] == 0
        assert txn.body_text("t1") == machine_json(updated)  # the result writer's one compact format
    store.update_task(tmp_path, "t1", lambda current: {**current, "status": "completed", "pause_notices": [{"n": 1}]},
                      pending_drive=True)
    with store.open_store(tmp_path, write=False) as txn:
        row = txn.conn.execute("SELECT * FROM tasks WHERE task_id='t1'").fetchone()
        assert (row["pending_drive"], row["has_pause_notices"]) == (1, 1)
        assert txn.row("t1", "has_pause_notices") == 1 and txn.row("t1", "owes_synthesis") == 0


def test_task_and_custody_share_the_write_transaction(tmp_path):
    publish(tmp_path, document("t1"))

    def write(n, boom=False):
        with store.open_store(tmp_path, write=True) as txn:
            updated = {**txn.load("t1"), "n": n, "boom": boom}
            txn.store_task(updated)
            rider(txn, updated)

    def rider(txn, updated):
        txn.conn.execute("INSERT INTO custody_runs (run_id, task_id, state, facts) VALUES (?, ?, ?, ?)",
                         (f"run{updated['n']}", updated["task_id"], "started", "{}"))
        if updated.get("boom"):
            raise RuntimeError("the rider refused")

    write(1)
    with pytest.raises(RuntimeError, match="the rider refused"):  # the rider's own exception, not a constraint's
        write(2, boom=True)
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.load("t1")["n"] == 1  # the refused transaction rolled back whole
        assert [r["run_id"] for r in txn.conn.execute("SELECT run_id FROM custody_runs")] == ["run1"]  # run2 with it


def test_obligation_rows_keep_their_file_order(tmp_path):
    publish(tmp_path)
    rows = {"z": {"task_id": "z"}, "a": {"task_id": "a", "closed_runs": {"r": "x"}}, "m": {}}
    with store.open_store(tmp_path, write=True) as txn:
        txn.replace_set("custody_open", rows)
        txn.replace_set("unknowns", {"task:q": {"reason": "malformed"}})
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.set_names() == ["custody_open", "unknowns"]
        assert list(txn.set_rows("custody_open").items()) == list(rows.items())


def test_encode_document_matches_the_result_writer_bytes(tmp_path):
    from ouroboros.utils import atomic_write_json

    row = document("t9", nested={"a": [1, {"b": "ü"}]}, cost_usd=0.000125)
    target = tmp_path / "t9.json"
    atomic_write_json(target, row)
    assert store.encode_document(row).encode("utf-8") == target.read_bytes()


# ---- the SQLite floor -----------------------------------------------------------------
_ABOVE_3_24 = re.compile(
    r"\bSTRICT\b|->>?|\bRETURNING\b|\bGENERATED\s+ALWAYS\b|\bjsonb?\(|\bjson_[a-z]+\(|\bIIF\(|\bVACUUM\s+INTO\b"
    r"|\bRENAME\s+COLUMN\b|\bDROP\s+COLUMN\b|\bOVER\s*\(|\bFILTER\s*\(|\bNULLS\s+(FIRST|LAST)\b", re.IGNORECASE)
STORE_MODULES = ("ouroboros/store.py", "ouroboros/store_import.py", "ouroboros/store_switch.py",
                 "ouroboros/store_task_cards.py")


def _string_literals(source: str):
    """Every string literal except docstrings (prose, not SQL)."""
    import ast

    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                docstrings.add(id(first.value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            yield node.value


def _offending_sql(source: str):
    return [s for s in _string_literals(source) if _ABOVE_3_24.search(s)]


def test_nothing_above_sqlite_3_24_in_the_store_modules():
    """A file written by the bundled engine must open on a 3.37 system engine (the CI floor
    job runs this module there); the modules' SQL uses no syntax the floor lacks."""
    sources = {rel: (REPO / rel).read_text(encoding="utf-8") for rel in STORE_MODULES}  # all three exist
    for rel, source in sources.items():
        assert not _offending_sql(source), (rel, _offending_sql(source))
        # the self-check: a newer construct inserted into THIS module's real source is caught
        assert _offending_sql(source + "\nq = 'SELECT body ->> \"$.x\" FROM t'\n"), rel
        assert _offending_sql(source + "\nq = 'CREATE TABLE t (a) STRICT'\n"), rel
    assert not _offending_sql('"""a -> b docstring"""\ndef f() -> str:\n    """c -> d"""\n')  # prose, annotations
    assert sqlite3.sqlite_version_info >= (3, 24, 0)


@pytest.mark.parametrize("mode", ["PERSIST", "WAL"])
def test_the_schema_builds_on_this_engine(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(store, "_journal_mode", lambda *_: mode)
    path = publish(tmp_path, document("t1"))
    with contextlib.closing(sqlite3.connect(path)) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert tables == {"tasks", "task_bodies", "task_cards", "obligations", "task_handoffs", "custody_runs",
                          "custody_outbox", "meta"}
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == ("wal" if mode == "WAL" else "delete")
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 0  # the version lives in meta


def test_readers_decode_bodies_after_the_read_transaction_closed(tmp_path, monkeypatch):
    """Grok M2: a listing fetches inside the store and decodes outside it, so a writer's COMMIT
    during the decode is not held behind the reader's SHARED lock (``StoreBusy`` after the
    wait). The probe commits a write from inside the first body's ``json.loads``."""
    import json as json_module
    from ouroboros.task_results import write_task_result
    from ouroboros.task_status import find_child_tasks

    for index in range(3):
        write_task_result(tmp_path, f"t{index}", "completed", parent_task_id="t0" if index else None,
                          delegation_role="subagent" if index else None)
    real_loads, fired = json_module.loads, []

    def decoding(text, *args, **kwargs):
        if '"task_id"' in text and not fired:  # the first body: a writer commits meanwhile
            fired.append(text)
            with store.open_store(tmp_path, write=True, wait_sec=0.5) as txn:
                txn.set_meta("probe", 1)
        return real_loads(text, *args, **kwargs)

    monkeypatch.setattr(json_module, "loads", decoding)
    assert [row["task_id"] for row in task_store.list_documents(tmp_path)] == ["t0", "t1", "t2"]
    assert fired
    fired.clear()
    assert sorted(row["task_id"] for row in find_child_tasks(tmp_path, parent_task_id="t0", scope="direct",
                                                             materialize_artifacts=False)) == ["t1", "t2"]
    assert fired
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.meta("probe") == 1


def _outside_the_lock(root, monkeypatch, probes):
    """Every mutator input is prepared with NO write transaction open: each prepared input first
    takes and releases the store's write lock itself, which ``StoreBusy``-fails under a mutator."""
    from ouroboros import task_custody

    def guard(name, real):
        def probe(*args, **kwargs):
            with store.open_store(root, write=True, wait_sec=0.3) as txn:
                txn.set_meta("probe", len(probes))
            probes.append(name)
            return real(*args, **kwargs)
        return probe

    for module, name in ((task_custody, "capture_unread_mail"), (task_custody, "capture_owner_mail")):
        monkeypatch.setattr(module, name, guard(name, getattr(module, name)))


def test_mutator_inputs_are_prepared_before_begin_and_reprepared_outside_when_the_row_moved(tmp_path, monkeypatch):
    """Mailbox captures run before BEGIN; a projector turning a row terminal
    rolls back empty and repeats with that input prepared outside the lock."""
    from ouroboros import store_import, task_results

    store_import.ensure_current_store(tmp_path)
    probes: list = []
    _outside_the_lock(tmp_path, monkeypatch, probes)
    # A first row needs no history input: its producer-owned fan-out starts empty.
    task_results.write_task_result(tmp_path, "t1", "running")
    assert probes == []
    assert task_results.load_task_result(tmp_path, "t1")["swarm_fanout_rollup"]["fanout_count"] == 0
    # The projector turns a non-terminal write terminal: the captures are prepared outside, once.
    probes.clear()
    stored = task_results.write_task_result(tmp_path, "t1", "running",
                                            _field_projector=lambda existing, fields: {**fields, "status": "completed"})
    assert stored["status"] == "completed" and sorted(probes) == ["capture_owner_mail", "capture_unread_mail"]


def _read_store_file_in_this_process(kind, root, path):
    import contextlib

    from ouroboros.artifacts import stream_artifact_file
    from ouroboros.gateway import files
    from ouroboros.tools import core_file_tools

    if kind == "header":
        state, tier = store.probe(root)
        assert state == "present" and tier is not None
    elif kind == "wal_flag":  # the first read of every store's hand-off at a door
        assert store.wal_file(path) in {True, False}
    elif kind == "preview":
        assert files._read_prefix(path, 4096).startswith(b"SQLite format 3")
    elif kind == "agent_read":
        with contextlib.suppress(UnicodeDecodeError):  # a database is not text; the read itself happened
            core_file_tools._read_source_text(path, None)
    elif kind == "copy":
        files._copy_path(path, root / "copied.sqlite")
        assert (root / "copied.sqlite").stat().st_size == path.stat().st_size
    elif kind == "stream":
        assert len(b"".join(store.iter_file_bytes(path))) == path.stat().st_size
    elif kind == "artifact":
        assert stream_artifact_file(path)["size"] == path.stat().st_size


@pytest.mark.serial
@pytest.mark.parametrize("kind", ["header", "wal_flag", "preview", "agent_read", "copy", "stream", "artifact"])
def test_reading_a_store_file_keeps_the_write_lock_another_thread_holds(tmp_path, kind):
    """Closing any descriptor of the store file releases every POSIX lock this process holds on
    it (sqlite.org howtocorrupt 2.2). ``probe`` read the header on every ``open_store``, and the
    Files panel, the read tools and artifact capture read whole files: each close freed an open
    transaction's lock and a second process could write the same pages."""
    import subprocess
    import sys
    import threading

    from ouroboros import store_import

    store_import.ensure_current_store(tmp_path)
    task_store.patch_document(tmp_path, "t1", lambda _current: {"task_id": "t1", "status": "running"})
    path = store.store_path(tmp_path)
    attempt = ("import sqlite3, sys\n"
               "conn = sqlite3.connect(sys.argv[1], isolation_level=None, timeout=0)\n"
               "try:\n    conn.execute('BEGIN IMMEDIATE')\n"
               "except sqlite3.OperationalError as exc:\n"
               "    print('blocked' if 'locked' in str(exc) else 'error: ' + str(exc))\n"
               "else:\n    print('acquired')\n    conn.execute('ROLLBACK')\n")

    def other_process() -> str:
        return subprocess.run([sys.executable, "-c", attempt, str(path)], capture_output=True,
                              text=True, timeout=60, check=True).stdout.strip()

    failures: list = []

    def read() -> None:
        try:
            _read_store_file_in_this_process(kind, tmp_path, path)
        except BaseException as exc:  # the assertion below names it
            failures.append(exc)

    with store.open_store(tmp_path, write=True):
        assert other_process() == "blocked"
        reader = threading.Thread(target=read)
        reader.start()
        reader.join(timeout=60)
        assert not reader.is_alive() and failures == []
        assert other_process() == "blocked"
    assert other_process() == "acquired"


def test_copying_a_held_store_file_onto_itself_raises_and_keeps_its_bytes(tmp_path):
    """The borrowed-descriptor copy keeps ``shutil.copy2``'s same-file refusal: opening the target
    for writing first would truncate the very file it reads."""
    from ouroboros import store_import

    store_import.ensure_current_store(tmp_path)
    path = store.store_path(tmp_path)
    assert store.is_sqlite_file(path)  # the borrowed branch (Windows copies with copy2)
    size = path.stat().st_size
    for destination in (path, path.parent):
        with pytest.raises(shutil.SameFileError):
            store.copy_file_bytes(path, destination)
        assert path.stat().st_size == size
    assert pathlib.Path(store.copy_file_bytes(path, tmp_path / "copy.sqlite")).stat().st_size == size


@pytest.mark.skipif(os.name == "nt", reason="Windows reads store files through ordinary handles")
def test_a_kept_reader_reads_like_a_file_and_continues_short_reads(tmp_path, monkeypatch):
    from ouroboros import store_import

    store_import.ensure_current_store(tmp_path)
    path = store.store_path(tmp_path)
    store.close_connections(path)
    data = path.read_bytes()
    with store.open_for_read(path) as reader:
        assert reader.read() == data and reader.tell() == len(data)
        assert reader.seek(-10, os.SEEK_END) == len(data) - 10 and reader.read() == data[-10:]
        with pytest.raises(ValueError):
            reader.seek(-1)
        reader.seek(5)
        monkeypatch.setattr(store.os, "pread", lambda fd, n, at, _pread=os.pread: _pread(fd, min(n, 3), at))
        assert reader.read(10) == data[5:15] and reader.tell() == 15
    assert store.held_descriptor(path) is None  # no connection open: returning the borrow closed it


@pytest.mark.skipif(os.name == "nt", reason="Windows keeps no shared descriptors")
def test_store_descriptors_live_only_as_long_as_their_connections(tmp_path):
    """One descriptor per touched store file for the process lifetime ran a live server to 2,682
    open descriptors in four hours (issue #1759); explicit pool disposal releases every
    connection and its raw descriptor, without relying on garbage collection."""
    from ouroboros import store_import

    roots = [tmp_path / f"drive{n}" for n in range(40)]
    for root in roots:
        store_import.ensure_current_store(root)
    store.close_connections()
    before = len(os.listdir("/dev/fd"))
    for root in roots:
        with store.open_store(root, write=True):
            assert store.read_file_bytes(store.store_path(root), limit=16) == b"SQLite format 3\x00"
            assert store.held_descriptor(store.store_path(root)) is not None  # shared while in use
        with store.open_store(root, write=False):
            pass
        store.close_connections(store.store_path(root))
    assert len(os.listdir("/dev/fd")) <= before  # nothing kept once the transactions ended
    assert not getattr(store, "_INODES", None)


@pytest.mark.serial
@pytest.mark.skipif(os.name == "nt", reason="POSIX record locks")
def test_a_reader_opened_before_a_transaction_keeps_its_lock_when_it_closes(tmp_path):
    """A borrow taken while nothing held the file must not be closed under a transaction that
    began meanwhile: the close would release that transaction's write lock."""
    import subprocess
    import sys

    from ouroboros import store_import

    store_import.ensure_current_store(tmp_path)
    path = store.store_path(tmp_path)
    attempt = ("import sqlite3, sys\n"
               "conn = sqlite3.connect(sys.argv[1], isolation_level=None, timeout=0)\n"
               "try:\n    conn.execute('BEGIN IMMEDIATE')\n"
               "except sqlite3.OperationalError as exc:\n"
               "    print('blocked' if 'locked' in str(exc) else 'error: ' + str(exc))\n"
               "else:\n    print('acquired')\n    conn.execute('ROLLBACK')\n")

    def other_process() -> str:
        return subprocess.run([sys.executable, "-c", attempt, str(path)], capture_output=True,
                              text=True, timeout=60, check=True).stdout.strip()

    reader = store.open_for_read(path)  # idle file: a fresh descriptor
    with store.open_store(tmp_path, write=True):
        assert other_process() == "blocked"
        assert reader.read(16) == b"SQLite format 3\x00"
        reader.close()  # returning the borrow while the transaction holds the lock
        assert other_process() == "blocked"
        assert store.held_descriptor(path) is not None  # its close waits for the connection
    assert other_process() == "acquired"  # an idle cached connection holds no transaction
    assert store.held_descriptor(path) is not None
    store.close_connections(path)
    assert store.held_descriptor(path) is None


@pytest.mark.skipif(os.name == "nt", reason="Windows keeps no shared descriptors")
def test_the_registry_survives_a_failed_close_a_finalizer_and_a_closed_reader(tmp_path):
    import threading

    from ouroboros import store_import

    store_import.ensure_current_store(tmp_path)
    path = store.store_path(tmp_path)
    store.close_connections(path)
    conn = store.connect(path, store.TIER_ENFORCED)
    key = (os.stat(path).st_dev, os.stat(path).st_ino)
    failures = []
    closer = threading.Thread(target=lambda: failures.append(_closing_raises(conn)))
    closer.start()
    closer.join(10)
    assert failures == [True] and store._INODES[key].conns == 1  # a refused close keeps it counted
    conn.close()
    assert key not in store._INODES

    reader = store.open_for_read(path)
    done = []

    def finalizer_inside_the_guard():  # a GC finalizer returning a borrow on the guard's own thread
        with store._INODE_GUARD:
            reader.close()
        done.append(True)

    worker = threading.Thread(target=finalizer_inside_the_guard, daemon=True)
    worker.start()
    worker.join(10)
    assert done == [True] and key not in store._INODES
    with pytest.raises(ValueError):
        reader.read(1)  # never through a descriptor that may now name another file


def _closing_raises(conn) -> bool:
    try:
        conn.close()  # another thread: sqlite3 refuses it
    except Exception:
        return True
    return False


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs FIFOs")
def test_a_fifo_is_never_opened_to_test_for_sqlite(tmp_path):
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    assert store.is_sqlite_file(fifo) is False  # returns at once: opening a FIFO would block
    with pytest.raises(shutil.SpecialFileError):
        store.copy_file_bytes(fifo, tmp_path / "copy")
