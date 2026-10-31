"""An index rebuild never leases historical parsing, or loses a concurrent publication."""
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
        rows = original(*args, **kwargs)
        readings.append(set(kwargs.get("names", ())))
        if len(readings) == 1:
            entered.set()
            assert resume.wait(10), "test did not release sidecar parsing"
        return rows

    def read(root):
        try:
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


@pytest.mark.parametrize("change", ["add", "delete"])
def test_changed_sidecars_refresh_outside_lock_before_publication(root, monkeypatch, change):
    first_path, _ = _publish(root, "first")
    _publish(root, "kept")
    with _paused_listing(monkeypatch) as (start, resume, readings, result):
        start(root)
        # A second caller acquires immediately even while historical parsing is stalled.
        with store.name_lock(root, packs.INDEX_LOCK_NAME, timeout_sec=0):
            if change == "delete":
                (packs.packs_dir(root) / packs.sidecar_name("first", 1)).unlink()
                first_path.unlink()
        if change == "add":
            _publish(root, "late")
        resume.set()
    expected = {"first", "kept", "late"} if change == "add" else {"kept"}
    assert result == [expected] and _tasks(root) == expected
    assert readings == [{"first.members.json", "kept.members.json"},
                        {"late.members.json"} if change == "add" else set()]
    with packs.open_index(root) as conn:
        for table in ("members", "handles"):
            assert {row[0] for row in conn.execute(f"SELECT task_id FROM {table}")} == expected


def test_failed_rebuild_preserves_old_usable_index(root, monkeypatch):
    _publish(root, "old")
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
            _tasks(root)
    assert (path.read_bytes(), store.file_identity(path)) == before
    assert _tasks(root) == {"old"}
    assert not list(path.parent.glob(".index.sqlite.tmp.*"))


def test_concurrent_winner_and_its_later_write_survive(root, monkeypatch):
    _publish(root, "first")
    with _paused_listing(monkeypatch) as (start, resume, _readings, result):
        start(root)
        _publish(root, "late")
        with packs.open_index(root, write=True) as conn:
            conn.execute("INSERT INTO meta VALUES ('concurrent_receipt', 'kept')")
        winner = store.file_identity(packs.packs_dir(root) / packs.INDEX_NAME)
        resume.set()
    assert result == [{"first", "late"}]
    assert store.file_identity(packs.packs_dir(root) / packs.INDEX_NAME) == winner
    with packs.open_index(root) as conn:
        assert conn.execute("SELECT value FROM meta WHERE key='concurrent_receipt'").fetchone()[0] == "kept"


@pytest.mark.parametrize("budget", [0.25, None, 10.0], ids=["slice", "ordinary", "larger-budget"])
def test_name_lock_wait_respects_budget_and_ordinary_wait_still_succeeds(root, monkeypatch, budget):
    _publish(root, "first")
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


def test_sidecar_publication_obeys_index_lock_and_retries_a_new_generation(root):
    with store.name_lock(root, packs.INDEX_LOCK_NAME, timeout_sec=0), store.acquisition_budget(0):
        with pytest.raises(store.StoreBusy):
            _publish(root, "first")
        assert not (packs.packs_dir(root) / packs.sidecar_name("first", 1)).exists()
    path, sidecar = _publish(root, "first", generation=2)
    packs.verify_pack(path, sidecar)
    assert _tasks(root) == {"first"}


def test_valid_cache_avoids_sidecar_reads_and_advisory_open_never_rebuilds(root, monkeypatch):
    _publish(root, "first")
    assert _tasks(root) == {"first"}
    monkeypatch.setattr(packs, "list_sidecars", lambda *a, **kw: pytest.fail("unexpected sidecar scan"))
    assert _tasks(root) == {"first"}
    with packs.open_index(root, rebuild=False, wait_sec=0) as conn:
        assert conn.execute("SELECT count(*) FROM packs").fetchone()[0] == 1
    (packs.packs_dir(root) / packs.INDEX_NAME).unlink()
    with pytest.raises(packs.sqlite3.DatabaseError, match="reconstruction"):
        with packs.open_index(root, rebuild=False, wait_sec=0):
            pytest.fail("missing index was opened")
