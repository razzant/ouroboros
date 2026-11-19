"""The evidence index's one builder never holds the index lock while it parses and keeps a competing usable
index; every access checks the cache's O(1) facts and writes it by its keys."""
import contextlib
import hashlib
import threading
import time

import pytest

from ouroboros import evidence_packs as packs, store
from tests._store_journal_shared import use_journal


@pytest.fixture(params=[(store.TIER_ENFORCED, "PERSIST"), (store.TIER_ENFORCED, "WAL"),
                        (store.TIER_NAME, "PERSIST")], ids=["persist", "wal", "name"])
def root(tmp_path, monkeypatch, request):
    tier, journal = request.param
    use_journal(monkeypatch, journal)
    monkeypatch.setattr(packs, "index_tier", lambda _root: tier)
    return tmp_path


def _publish(root, task, generation=1):
    raw = f"retained checkpoint for {task}".encode()
    sha = hashlib.sha256(raw).hexdigest()
    source = root / f"{task}.source"
    source.write_bytes(raw)
    return packs.write_pack(root, task, generation, [{
        "sha256": sha, "kind": packs.CHECKPOINT, "size": len(raw), "source": source,
        "path": (packs.CHECKPOINTS_REL / f"{task}-{sha}.md").as_posix(),
    }])


def _tasks(root):
    with packs.open_index(root) as conn:
        return {row[0] for row in conn.execute("SELECT task_id FROM packs")}


@contextlib.contextmanager
def _paused_listing(monkeypatch):
    """Freeze one completed parse; the caller may mutate authority before publication."""
    entered, resume = threading.Event(), threading.Event()
    original = packs.list_sidecars
    readings, failures, result = [], [], []

    def listing(*args, **kwargs):
        rows = list(original(*args, **kwargs))
        readings.append({path.name for path, _sidecar in rows})
        if len(readings) == 1:
            entered.set()
            assert resume.wait(10), "test did not release sidecar parsing"
        return iter(rows)

    def read(root):
        try:
            packs.ensure_index(root)
            result.append(_tasks(root))
        except BaseException as exc:
            failures.append(exc)

    monkeypatch.setattr(packs, "list_sidecars", listing)
    thread = None

    def start(root):
        nonlocal thread
        thread = threading.Thread(target=read, args=(root,))
        thread.start()
        assert entered.wait(5), "rebuild did not read its sidecars"

    try:
        yield start, resume, readings, result
    finally:
        resume.set()
        if thread is not None:
            thread.join(10)
            assert not thread.is_alive(), "rebuild did not finish"
        assert not failures


def test_a_rebuild_parses_outside_the_index_lock(root, monkeypatch):
    """Reading every sidecar is the builder's own work: while it parses, the index lock stays free for
    readers and publishers; it is taken only to publish the file the builder checked."""
    _publish(root, "first")
    _publish(root, "kept")
    with _paused_listing(monkeypatch) as (start, resume, readings, result):
        start(root)
        with store.name_lock(root, packs.INDEX_LOCK_NAME, timeout_sec=0):
            pass  # a second caller acquires at once while the builder's parse is stalled
        resume.set()
    assert result == [{"first", "kept"}] and readings == [{"first.members.json", "kept.members.json"}]
    with packs.open_index(root) as conn:
        for table in ("members", "handles"):
            assert {row[0] for row in conn.execute(f"SELECT task_id FROM {table}")} == {"first", "kept"}


def test_failed_rebuild_preserves_old_usable_index(root, monkeypatch):
    _publish(root, "old")
    packs.ensure_index(root)
    assert _tasks(root) == {"old"}
    path = packs.packs_dir(root) / packs.INDEX_NAME
    before = path.read_bytes(), store.file_identity(path)
    valid, listing = packs._index_valid, packs.list_sidecars
    checks = []

    def rebuild_once(*args, **kwargs):
        checks.append(True)
        return False if len(checks) == 1 else valid(*args, **kwargs)

    def crash(*args, **kwargs):
        listing(*args, **kwargs)
        raise KeyboardInterrupt("rebuild stopped before publication")

    with monkeypatch.context() as patch:
        patch.setattr(packs, "_index_valid", rebuild_once)
        patch.setattr(packs, "list_sidecars", crash)
        with pytest.raises(KeyboardInterrupt, match="stopped before publication"):
            packs.ensure_index(root)
    assert (path.read_bytes(), store.file_identity(path)) == before
    assert _tasks(root) == {"old"}
    assert not list(path.parent.glob(".index.sqlite.tmp.*"))


def test_concurrent_winner_and_its_later_write_survive(root, monkeypatch):
    _publish(root, "first")
    with _paused_listing(monkeypatch) as (start, resume, _readings, result):
        start(root)
        _publish(root, "late")
        packs.ensure_index(root)  # a competing builder publishes first
        with packs.open_index(root, write=True) as conn:
            conn.execute("INSERT INTO meta VALUES ('concurrent_receipt', 'kept')")
        winner = store.file_identity(packs.packs_dir(root) / packs.INDEX_NAME)
        resume.set()
    assert result == [{"first", "late"}]
    assert store.file_identity(packs.packs_dir(root) / packs.INDEX_NAME) == winner
    with packs.open_index(root) as conn:
        assert conn.execute("SELECT value FROM meta WHERE key='concurrent_receipt'").fetchone()[0] == "kept"


@pytest.mark.parametrize("tier", [store.TIER_ENFORCED, store.TIER_NAME], ids=["enforced", "name"])
def test_a_rebuild_never_publishes_beside_the_replaced_file_s_journal(tmp_path, monkeypatch, tier):
    """A writer stopped between a generation's rows and COMMIT leaves the index's hot journal; the main file then
    gone, the pass rebuilds from the sidecars, and the replaced file's journal must never roll back into the new
    one: it goes under the index lock before the new file takes the name, so both generations stay indexed and the
    blob reads by its sha alone. The ordinary rebuild and the competing winner keep their own tests above."""
    import pathlib

    from tests.test_evidence_packs import _blob

    use_journal(monkeypatch, "PERSIST")
    monkeypatch.setattr(packs, "index_tier", lambda _root: tier)
    _publish(tmp_path, "a")
    packs.ensure_index(tmp_path)
    raw = b'"the second generation"'
    ref = _blob(tmp_path, raw)
    pack, sidecar = packs.write_pack(tmp_path, "b", 1, [{key: ref[key] for key in ("sha256", "kind", "size")}])
    pathlib.Path(ref["path"]).unlink()  # only its pack holds it now
    index = packs.packs_dir(tmp_path) / packs.INDEX_NAME
    journal, hot, rows = index.with_name(index.name + "-journal"), tmp_path / "hot-journal", packs._index_rows

    def stopped(conn, *args):  # its pages reach the file (a small cache), then it stops before COMMIT
        conn.execute("PRAGMA cache_size = 1")
        rows(conn, *args)
        conn.execute("INSERT INTO meta VALUES ('padding', ?)", ("x" * 400_000,))
        hot.write_bytes(journal.read_bytes())  # the journal a crash right here leaves
        raise RuntimeError("stopped before COMMIT")

    monkeypatch.setattr(packs, "_index_rows", stopped)
    with pytest.raises(RuntimeError):
        packs.index_upsert(tmp_path, pack, sidecar)
    monkeypatch.setattr(packs, "_index_rows", rows)
    assert hot.read_bytes()[:8] == bytes.fromhex("d9d505f920a163d7")  # a hot journal: its pages reached the file
    index.unlink()
    journal.write_bytes(hot.read_bytes())
    assert packs.index_owed(tmp_path) and packs.ensure_index(tmp_path)["sidecars"] == 2
    assert _tasks(tmp_path) == {"a", "b"} and packs.blob_bytes(tmp_path, ref) == raw


def _wal_index_with_b(tmp_path, monkeypatch):
    """A WAL index of tasks a and b (b's blob packed only), and that blob's ref and bytes."""
    import pathlib

    from tests.test_evidence_packs import _blob

    use_journal(monkeypatch, "WAL")
    monkeypatch.setattr(packs, "index_tier", lambda _root: store.TIER_ENFORCED)
    _publish(tmp_path, "a")
    packs.ensure_index(tmp_path)
    raw = b'"the second generation"'
    ref = _blob(tmp_path, raw)
    pack, sidecar = packs.write_pack(tmp_path, "b", 1, [{key: ref[key] for key in ("sha256", "kind", "size")}])
    pathlib.Path(ref["path"]).unlink()  # only its pack holds it now
    index = packs.packs_dir(tmp_path) / packs.INDEX_NAME
    return index, index.with_name(index.name + "-wal"), ref, raw, (pack, sidecar)


def _killed_before_checkpoint(index, wal, write):
    """Run ``write`` in one transaction of a WAL writer that never checkpoints, then leave the main file and its
    WAL as the writer killed right after COMMIT leaves them (closing would checkpoint them together)."""
    with contextlib.closing(store.connect(index, store.TIER_ENFORCED)) as conn:
        conn.execute("PRAGMA wal_autocheckpoint = 0")
        with store.transaction(conn, write=True):
            write(conn)
        main, log = index.read_bytes(), wal.read_bytes()
    store.close_connections()
    index.write_bytes(main)
    wal.write_bytes(log)


@pytest.mark.parametrize("cut", ["rename", "shared-memory"])
def test_an_interrupted_publication_leaves_the_index_missing_never_the_old_file_without_its_wal(tmp_path, monkeypatch, cut):
    """A generation committed only in the old file's WAL (its writer killed before the checkpoint), and a rebuild the
    pass ordered on its own failure to open the file (no mark): an interrupted publication must never leave the old
    file without its WAL, which looks usable and answers that generation absent. The old file goes first, then its
    journals, then the rename: interrupted at the rename, or earlier where the old file's shared memory cannot be
    removed once its WAL is gone (another program holds it), the index is missing and owed, a read by sha is unknown
    now (never absent), and the next pass publishes every generation."""
    import pathlib

    index, wal, ref, raw, (pack, sidecar) = _wal_index_with_b(tmp_path, monkeypatch)
    _killed_before_checkpoint(index, wal, lambda conn: packs._index_rows(conn, pack, sidecar))
    valid, replace, unlink = packs._index_valid, packs.replace_atomic, pathlib.Path.unlink

    def stop(src, dst, *args, **kwargs):
        if pathlib.Path(dst) == index:
            raise KeyboardInterrupt("publication interrupted")
        return replace(src, dst, *args, **kwargs)

    def held(self, *args, **kwargs):
        if self.name == index.name + "-shm":
            raise KeyboardInterrupt("publication interrupted")
        return unlink(self, *args, **kwargs)

    with monkeypatch.context() as patch:  # only the pass's own checks fail: no reader marked anything
        patch.setattr(packs, "_index_valid", lambda *args, reader=False, **kwargs: reader and valid(*args, reader=True, **kwargs))
        if cut == "rename":
            patch.setattr(packs, "replace_atomic", stop)
        else:
            patch.setattr(pathlib.Path, "unlink", held)
        with pytest.raises(KeyboardInterrupt, match="publication interrupted"):
            packs.ensure_index(tmp_path)
    with pytest.raises(packs.EvidenceUnavailable):  # unknown now, never absent
        packs.blob_bytes(tmp_path, ref)
    assert packs.index_owed(tmp_path) and packs.ensure_index(tmp_path)["sidecars"] == 2
    assert _tasks(tmp_path) == {"a", "b"} and packs.blob_bytes(tmp_path, ref) == raw


def test_a_stale_wal_beside_a_missing_index_never_rolls_into_the_rebuilt_one(tmp_path, monkeypatch):
    """The WAL of a writer killed before its checkpoint, its main file then gone: the pass rebuilds from the
    sidecars, and that WAL never rolls into the new file: every pack and member stays, the blob reads by its sha."""
    index, wal, ref, raw, (pack, sidecar) = _wal_index_with_b(tmp_path, monkeypatch)
    packs.index_upsert(tmp_path, pack, sidecar)
    _killed_before_checkpoint(index, wal, lambda conn: [conn.execute(f"DELETE FROM {table}") for table in ("members", "packs")])
    index.unlink()
    assert packs.index_owed(tmp_path) and packs.ensure_index(tmp_path)["sidecars"] == 2
    with packs.open_index(tmp_path) as conn:
        counts = [conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] for table in ("packs", "members")]
    assert counts == [2, 2] and packs.blob_bytes(tmp_path, ref) == raw


@pytest.mark.parametrize("budget", [0.25, None, 10.0], ids=["slice", "ordinary", "larger-budget"])
def test_name_lock_wait_respects_budget_and_ordinary_wait_still_succeeds(root, monkeypatch, budget):
    _publish(root, "first")
    packs.ensure_index(root)
    assert _tasks(root) == {"first"}
    held, release = threading.Event(), threading.Event()
    original = store.name_lock
    waits, failures = [], []

    def hold():
        try:
            with original(root, packs.INDEX_LOCK_NAME, timeout_sec=0):
                held.set()
                assert release.wait(10)
        except BaseException as exc:
            failures.append(exc)

    @contextlib.contextmanager
    def observed(*args, **kwargs):
        waits.append(kwargs["timeout_sec"])
        with original(*args, **kwargs):
            yield

    thread = threading.Thread(target=hold)
    thread.start()
    timer = None
    try:
        assert held.wait(5)
        monkeypatch.setattr(store, "name_lock", observed)
        if budget != 0.25:
            timer = threading.Timer(0.4, release.set)
            timer.start()
        begun = time.monotonic()
        with store.acquisition_budget(budget) if budget is not None else contextlib.nullcontext():
            if budget == 0.25:
                with pytest.raises(store.StoreBusy):
                    _tasks(root)
                assert time.monotonic() - begun < 1.0
            else:
                assert _tasks(root) == {"first"}
                assert time.monotonic() - begun >= 0.3
        assert waits == [0.25 if budget == 0.25 else packs.INDEX_WAIT_SEC]
    finally:
        release.set()
        thread.join(5)
        if timer is not None:
            timer.cancel()
            timer.join(5)
    assert not thread.is_alive() and not failures


def test_busy_upsert_keeps_existing_index_and_retries_successfully(root):
    _publish(root, "first")
    packs.ensure_index(root)
    assert _tasks(root) == {"first"}
    late_path, late = _publish(root, "late")
    path = packs.packs_dir(root) / packs.INDEX_NAME
    before = path.read_bytes(), store.file_identity(path)
    with store.name_lock(root, packs.INDEX_LOCK_NAME, timeout_sec=0), store.acquisition_budget(0):
        with pytest.raises(store.StoreBusy):
            packs.index_upsert(root, late_path, late)
    assert (path.read_bytes(), store.file_identity(path)) == before
    packs.index_upsert(root, late_path, late)
    assert _tasks(root) == {"first", "late"}


def test_pack_publication_obeys_index_lock_and_leaves_its_generation_free(root):
    """A pack and its sidecar publish under one hold of the index lock: a lock held past the wait
    publishes neither (never a pack without its sidecar, no scratch left), so the retry takes the same
    generation; a free lock publishes the pair as before."""
    directory = packs.packs_dir(root)
    with store.name_lock(root, packs.INDEX_LOCK_NAME, timeout_sec=0), store.acquisition_budget(0):
        with pytest.raises(store.StoreBusy):
            _publish(root, "first")
    assert list(directory.iterdir()) == [] and list(packs._task_sidecar_names(root, "first")) == []
    path, sidecar = _publish(root, "first")
    assert sidecar["generation"] == 1 and sorted(p.name for p in directory.iterdir()) == ["first.members.json", "first.xz"]
    packs.verify_pack(path, sidecar)
    packs.ensure_index(root)
    assert _tasks(root) == {"first"}


def test_index_access_checks_o1_facts_never_the_whole_file(root, monkeypatch):
    """Writing the cache and reading it after another writer's commit run no whole-file check (each
    ran one ``PRAGMA quick_check`` before: the bytes this process had validated changed) and build
    nothing: the header names the tier, the meta row the schema, a damaged page its own SQLite error."""
    _publish(root, "first")
    packs.ensure_index(root)  # the packing pass's builder leaves a cache
    index = packs.packs_dir(root) / packs.INDEX_NAME
    built, statements, connect = store.file_identity(index), [], store.connect

    def traced(*args, **kwargs):
        conn = connect(*args, **kwargs)
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(store, "connect", traced)
    late_path, late = _publish(root, "late")
    packs.index_upsert(root, late_path, late)
    with contextlib.closing(store.connect(index, packs.index_tier(root))) as other:  # another process's commit
        with store.transaction(other, write=True):
            other.execute("INSERT OR REPLACE INTO swept VALUES ('elsewhere', '[]')")
    assert _tasks(root) == {"first", "late"}
    checks = [sql for sql in statements if sql.upper().startswith(("PRAGMA QUICK_CHECK", "PRAGMA INTEGRITY_CHECK"))]
    assert statements and not checks, checks
    assert store.file_identity(index) == built


def test_indexing_a_recorded_generation_again_is_keyed_and_restores_a_lost_member(root, monkeypatch):
    """A generation is immutable, so its rows go in by their keys: indexing it again scans no table (a
    delete by task and generation scanned every member, whose key leads with the sha), refreshes the
    ``verified_at`` of its ``packs`` row, brings back a member row the cache lost and adds none twice."""
    path, sidecar = _publish(root, "first")
    packs.index_upsert(root, path, {**sidecar, "verified_at": "v1"})
    index = packs.packs_dir(root) / packs.INDEX_NAME
    with contextlib.closing(store.connect(index, packs.index_tier(root))) as conn:
        with store.transaction(conn, write=True):
            conn.execute("DELETE FROM members")
    statements, connect = [], store.connect

    def traced(*args, **kwargs):
        conn = connect(*args, **kwargs)
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(store, "connect", traced)
    packs.index_upsert(root, path, {**sidecar, "verified_at": "v2"})
    packs.index_upsert(root, path, {**sidecar, "verified_at": "v2"})
    monkeypatch.setattr(store, "connect", connect)
    writes = [sql for sql in statements if sql.split()[0].upper() in {"INSERT", "DELETE", "UPDATE", "REPLACE"}]
    with contextlib.closing(store.connect(index, packs.index_tier(root))) as conn:
        plans = [row[-1] for sql in writes for row in conn.execute("EXPLAIN QUERY PLAN " + sql)]
        assert writes and not [plan for plan in plans if plan.startswith("SCAN")], plans
        assert [row[0] for row in conn.execute("SELECT verified_at FROM packs")] == ["v2"]
        assert conn.execute("SELECT count(*) FROM members").fetchone()[0] == len(sidecar["members"]) == 1


def test_valid_cache_avoids_sidecar_reads_and_an_open_never_rebuilds(root, monkeypatch):
    _publish(root, "first")
    packs.ensure_index(root)
    assert _tasks(root) == {"first"}
    monkeypatch.setattr(packs, "list_sidecars", lambda *a, **kw: pytest.fail("unexpected sidecar scan"))
    assert _tasks(root) == {"first"} and packs.ensure_index(root) is None
    with packs.open_index(root, wait_sec=0) as conn:
        assert conn.execute("SELECT count(*) FROM packs").fetchone()[0] == 1
    (packs.packs_dir(root) / packs.INDEX_NAME).unlink()
    with pytest.raises(packs.EvidenceUnavailable, match="awaits the packing pass"):
        with packs.open_index(root, wait_sec=0):
            pytest.fail("missing index was opened")
    assert not (packs.packs_dir(root) / packs.INDEX_NAME).exists() and packs.index_owed(root)
