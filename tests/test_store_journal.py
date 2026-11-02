"""WAL needs a fixed SQLite, enforced locks AND a positively identified local filesystem."""
import contextlib
import ctypes
import json
import os
import pathlib
import shutil
import sqlite3
import sys
import threading

import pytest

from ouroboros import store, store_import, store_switch, usage_store
from tests.test_store import publish


@pytest.fixture(autouse=True)
def release_connections():
    store.close_connections()
    yield
    store.close_connections()


@pytest.mark.parametrize("version,fixed", [
    ((3, 37, 2), False), ((3, 44, 5), False), ((3, 44, 6), True), ((3, 44, 9), True),
    ((3, 45, 9), False), ((3, 50, 4), False), ((3, 50, 6), False), ((3, 50, 7), True),
    ((3, 50, 8), True), ((3, 51, 2), False), ((3, 51, 3), True), ((3, 53, 1), True)])
@pytest.mark.parametrize("tier", [store.TIER_ENFORCED, store.TIER_NAME])
@pytest.mark.parametrize("filesystem,local", [
    ("apfs", True), ("ext4", True), ("fixed", True), ("nfs", False), ("cifs", False),
    ("fuse.drive", False), ("9p", False), ("unknown", False), (None, False)])
def test_wal_requires_all_three_facts(monkeypatch, version, fixed, tier, filesystem, local):
    monkeypatch.setattr(sqlite3, "sqlite_version_info", version)
    monkeypatch.setattr(store, "_filesystem_type", lambda _path: filesystem)
    assert store._journal_mode(pathlib.Path("store.sqlite"), tier) == (
        "WAL" if fixed and tier == store.TIER_ENFORCED and local else "PERSIST")


@pytest.mark.parametrize("tier,version", [(store.TIER_NAME, (3, 53, 1)), (store.TIER_ENFORCED, (3, 50, 4))])
def test_ineligible_engine_or_tier_never_probes_filesystem(monkeypatch, tier, version):
    monkeypatch.setattr(sqlite3, "sqlite_version_info", version)
    monkeypatch.setattr(store, "_filesystem_type", lambda _: pytest.fail("unnecessary filesystem probe"))
    assert store._journal_mode(pathlib.Path("store.sqlite"), tier) == "PERSIST"


def _mountinfo_path(path: pathlib.Path) -> str:
    """A mount point as the kernel writes it in mountinfo: space, tab, newline and backslash as octal
    escapes. The parser only runs on Linux, but this test runs everywhere: a Windows tmp_path carries
    backslashes and a drive anchor, so its root mount is that anchor rather than ``/``."""
    return str(path).translate({ord(" "): r"\040", ord("\t"): r"\011", ord("\n"): r"\012", ord("\\"): r"\134"})


@pytest.mark.parametrize("kind,expected", [
    ("ext4", "ext4"), ("xfs", "xfs"), ("nfs", "nfs"), ("nfs4", "nfs4"), ("cifs", "cifs"),
    ("smb3", "smb3"), ("fuse", "fuse"), ("fuseblk", "fuseblk"), ("fuse.drive", "fuse.drive"), ("9p", "9p")])
def test_linux_uses_longest_mount_including_bind_and_escaped_names(tmp_path, kind, expected):
    mount = tmp_path.resolve() / "nested mount"
    mount.mkdir()
    root, escaped = _mountinfo_path(pathlib.Path(mount.anchor)), _mountinfo_path(mount)
    table = f"1 0 8:1 / {root} rw - ext4 /dev/root rw\n2 1 0:2 /bind {escaped} rw - {kind} source rw\n"
    assert store._linux_filesystem(mount / "store.sqlite", table) == expected
    assert store._linux_filesystem(tmp_path / "elsewhere.sqlite", table) == "ext4"
    assert store._linux_filesystem(mount, "bad mountinfo") is None
    assert store._linux_filesystem(mount, "") is None


@pytest.mark.skipif(os.name == "nt", reason="overlay options separate directories with ':', which a Windows "
                    "drive path carries; this parser runs only on Linux")
@pytest.mark.parametrize("backing", ["ext4", "nfs", "fuse.drive", "missing"])
def test_overlay_requires_every_backing_directory_to_be_known_local(tmp_path, backing):
    overlay, lower, upper = [tmp_path / part for part in ("overlay", "lower", "upper")]
    for directory in (overlay, lower, upper):
        directory.mkdir()
    table = (f"1 0 8:1 / / rw - ext4 /dev/root rw\n"
             f"2 1 0:2 / {lower} rw - {backing} source rw\n"
             f"3 1 0:3 / {overlay} rw - overlay overlay rw,lowerdir={lower},upperdir={upper}\n")
    assert store._linux_filesystem(overlay / "store.sqlite", table) == ("overlay-local" if backing == "ext4" else None)
    hidden = "1 0 0:1 / / rw - overlay overlay rw,lowerdir=/host/invisible\n"
    assert store._linux_filesystem(tmp_path, hidden) is None


@pytest.mark.parametrize("platform,kind,expected", [
    ("darwin", "apfs", "apfs"), ("darwin", "hfs", "hfs"), ("darwin", "smbfs", None),
    ("freebsd14", "ufs", "ufs"), ("freebsd14", "zfs", None), ("freebsd14", "nfs", None)])
def test_bsd_statfs_reads_the_native_filesystem_name(monkeypatch, tmp_path, platform, kind, expected):
    class StatFS:
        def __call__(self, path, pointer):
            record = pointer._obj
            assert type(record).fstypename.offset == (72 if platform == "darwin" else 280)
            record.fstypename = kind.encode("ascii")
            return 0
    class LibC:
        statfs64 = StatFS()
        statfs = StatFS()
    monkeypatch.setattr(store.sys, "platform", platform)
    monkeypatch.setattr(ctypes, "CDLL", lambda *_a, **_kw: LibC())
    assert store._filesystem_type(tmp_path) == expected


@pytest.mark.parametrize("kind", [0, 1, 2, 3, 4, 5, 6])
def test_windows_only_accepts_fixed_volume_including_volume_mounts(monkeypatch, tmp_path, kind):
    class Volume:
        def __call__(self, path, root, size):
            root.value = "C:\\mounted-volume\\"
            return 1
    class Drive:
        def __call__(self, root):
            assert root == "C:\\mounted-volume\\"
            return kind
    class Kernel:
        GetVolumePathNameW = Volume()
        GetDriveTypeW = Drive()
    monkeypatch.setattr(store.sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "WinDLL", lambda *_a, **_kw: Kernel(), raising=False)
    assert store._filesystem_type(tmp_path) == ("fixed" if kind == 3 else None)


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32", "unknown"])
def test_unreadable_filesystem_is_unknown(monkeypatch, tmp_path, platform):
    def unavailable(*_args, **_kwargs):
        raise OSError("filesystem unavailable")
    monkeypatch.setattr(store.sys, "platform", platform)
    monkeypatch.setattr(pathlib.Path, "read_text", unavailable)
    monkeypatch.setattr(store, "_bsd_filesystem", unavailable)
    monkeypatch.setattr(store, "_windows_filesystem", unavailable)
    assert store._filesystem_type(tmp_path) is None


@pytest.mark.skipif(sys.platform != "darwin", reason="native macOS statfs integration")
def test_native_statfs_identifies_the_test_volume(tmp_path):
    # Synthetic data, but the real ABI: an offset or symbol error must not silently
    # disable WAL on the APFS/HFS test host.
    assert store._bsd_filesystem(tmp_path) in {"apfs", "hfs"}


@pytest.mark.parametrize("money", [False, True])
def test_wal_is_full_and_export_checkpoints_to_standalone_delete(tmp_path, monkeypatch, money):
    # The shipped interpreter may predate the fix. Override eligibility only for
    # this serial synthetic lifecycle; the production version gate above stays strict.
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 53, 1))
    monkeypatch.setattr(store, "_filesystem_type", lambda _path: "apfs")
    if money:
        usage_store.migrate_from_journal(tmp_path)
    else:
        publish(tmp_path)
    with usage_store.hold(tmp_path) if money else store.open_store(tmp_path, write=True) as txn:
        txn.set_meta("retained", 42)
        assert txn.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert txn.conn.execute("PRAGMA synchronous").fetchone()[0] == 2
        held = txn.conn
    path = tmp_path / (usage_store.STORE_REL if money else store.STORE_REL)
    assert path.with_name(path.name + "-wal").exists()
    if money:
        result = usage_store.export_journal(tmp_path)
        retired = path.with_name(result["store_retired_as"])
    else:
        with store.open_store(tmp_path, write=True) as txn:
            txn.set_meta("phase", store.EXPORTING)
        store_import.export_facts(tmp_path)
        retired = path.with_name(store_switch._retire(tmp_path))
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        held.execute("SELECT 1")
    assert all(not path.with_name(path.name + suffix).exists() for suffix in ("-wal", "-shm", "-journal"))
    with contextlib.closing(sqlite3.connect(retired)) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert conn.execute("SELECT value FROM meta WHERE key='retained'").fetchone()[0] == "42"
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


@pytest.mark.parametrize("money", [False, True])
@pytest.mark.parametrize("mode", ["PERSIST", "WAL"])
def test_sibling_build_is_standalone_before_publication(tmp_path, monkeypatch, money, mode):
    monkeypatch.setattr(store, "_journal_mode", lambda *_: mode)
    if money:
        usage_store.migrate_from_journal(tmp_path)
    else:
        store_import.ensure_current_store(tmp_path, with_children=False)
    assert list((tmp_path / "state").glob("*.import-*")) == []


@pytest.mark.parametrize("mode", ["PERSIST", "WAL"])
def test_failed_build_rolls_back_and_releases_its_temporary_journal(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(store, "_journal_mode", lambda *_: mode)
    path = tmp_path / "unpublished.sqlite"
    with pytest.raises(ValueError):
        with contextlib.closing(store.connect(path, store.TIER_ENFORCED, create=True)) as conn:
            conn.execute("CREATE TABLE pending(value)")
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("INSERT INTO pending VALUES (1)")
            raise ValueError("interrupted build")
    assert all(not path.with_name(path.name + suffix).exists() for suffix in ("-journal", "-wal", "-shm"))
    with contextlib.closing(sqlite3.connect(path)) as conn:
        assert conn.execute("SELECT * FROM pending").fetchall() == []
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == ("wal" if mode == "WAL" else "delete")


@pytest.mark.parametrize("dialogue", [False, True])
@pytest.mark.parametrize("wal", [True, False])
def test_a_door_reset_hands_every_store_off_out_of_wal_first(tmp_path, monkeypatch, wal, dialogue):
    # A store-aware older body never sets the journal mode: it must receive DELETE files, from the
    # door's own reset and from the seam alike (every release before the dialogue store needs the
    # seam), while a rollback-journal store needs no connection at all.
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 53, 1))
    monkeypatch.setattr(store, "_filesystem_type", lambda _path: "apfs" if wal else None)
    publish(tmp_path)
    usage_store.migrate_from_journal(tmp_path)
    with store.open_store(tmp_path, write=True) as txn:
        txn.set_meta("retained", 1)
    with usage_store.hold(tmp_path) as txn:
        txn.set_meta("retained", 1)
    paths = [tmp_path / store.STORE_REL, tmp_path / usage_store.STORE_REL]
    assert [store.wal_file(path) for path in paths] == [wal, wal]
    if not wal:
        monkeypatch.setattr(store, "prepare_handoff", lambda *a, **k: pytest.fail("rollback journals need no handoff"))
    monkeypatch.setattr(store_switch, "switch_needs",
                        lambda *_: {"facts": False, "money": False, "handles": False, "dialogue": dialogue})
    monkeypatch.setattr(store_switch, "rev_parse", lambda _repo, _ref: "0" * 40)
    seen = []

    def reset(*_intent):
        seen.append([store.wal_file(path) for path in paths])
        return True, "reset done"

    ok, _message, deferred = store_switch.switch_or_reset(tmp_path, tmp_path, target_sha="0" * 40, target_ref="v-old",
                                                          reason="test", door="ui_rollback", reset=reset)
    assert (ok, deferred) == (True, dialogue)
    if deferred:  # the seam exports and moves the checkout at the restart exit, the door as its owner
        assert [store.wal_file(path) for path in paths] == [wal, wal]  # nothing moved yet
        store.close_connections()
        assert store_switch.run_switch_seam(tmp_path, repo_dir=tmp_path, owner=reset)["status"] == "switched"
    assert seen == [[False, False]]
    assert all(not path.with_name(path.name + "-wal").exists() for path in paths)
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.meta("retained") == 1


@pytest.mark.parametrize("outcome", ["refused_before_tree_effects", "failed", "raised"])
def test_a_reset_that_changed_nothing_hands_the_stores_back(tmp_path, monkeypatch, outcome):
    """After a successful hand-off: a reset refused before any tree effect leaves this process running
    on its own stores, in the rollback journal until a boot chooses again; any other failure may
    already have moved the checkout, so the stores stay handed off."""
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 53, 1))
    monkeypatch.setattr(store, "_filesystem_type", lambda _path: "apfs")
    publish(tmp_path)
    with store.open_store(tmp_path, write=True) as txn:
        txn.set_meta("retained", 1)
    monkeypatch.setattr(store_switch, "switch_needs", lambda *_: {"facts": False, "money": False})
    handed = os.path.abspath(store.store_path(tmp_path))

    class Refused(Exception):
        before_tree_effects = outcome == "refused_before_tree_effects"

    def reset():
        assert handed in store._HANDED_OFF
        if outcome == "failed":
            return False, "the reset failed"
        raise Refused("git refused the reset")

    if outcome == "failed":
        assert store_switch.switch_or_reset(tmp_path, tmp_path, target_sha="0" * 40, target_ref="v-old", reason="t",
                                            door="ui_rollback", reset=reset) == (False, "the reset failed", False)
    else:
        with pytest.raises(Refused):
            store_switch.switch_or_reset(tmp_path, tmp_path, target_sha="0" * 40, target_ref="v-old", reason="t",
                                         door="ui_rollback", reset=reset)
    assert (handed in store._HANDED_OFF) is (outcome != "refused_before_tree_effects")
    store.forget_handoffs()


def test_a_store_that_cannot_hand_off_refuses_the_door_reset(tmp_path, monkeypatch):
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 53, 1))
    monkeypatch.setattr(store, "_filesystem_type", lambda _path: "apfs")
    publish(tmp_path)
    with store.open_store(tmp_path, write=True) as txn:
        txn.set_meta("retained", 1)
    store.close_connections()

    def busy(*_args, **_kwargs):
        raise store.StoreBusy("store checkpoint busy at handoff", reason="contention")

    monkeypatch.setattr(store, "prepare_handoff", busy)
    monkeypatch.setattr(store_switch, "switch_needs", lambda *_: {"facts": False, "money": False})
    handed = os.path.abspath(store.store_path(tmp_path))
    store._HANDED_OFF.add(handed)  # a store the refused door had already handed off
    ok, message, deferred = store_switch.switch_or_reset(
        tmp_path, tmp_path, target_sha="0" * 40, target_ref="v-old", reason="test", door="ui_rollback",
        reset=lambda: pytest.fail("the checkout moved past a store still in WAL"))
    assert (ok, deferred) == (False, False) and "checkpoint busy" in message
    assert handed not in store._HANDED_OFF, "a refused door hands every store back to this process"


def test_the_log_identity_moves_with_every_commit(tmp_path):
    """Read from the log's header and frame headers beside its size and time: a commit adds a commit
    frame, a rolled-back transaction's spilled frames add none (only the log's size and time move),
    and a log reused in place at the same size after a checkpoint carries a new generation's salts."""
    path = tmp_path / "log.sqlite"
    writer = sqlite3.connect(path, isolation_level=None)
    assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.execute("CREATE TABLE t(v)")
    held = sqlite3.connect(path, isolation_level=None)  # stays open: no last close removes the log
    log = path.with_name(path.name + "-wal")

    def commit(rows, connection=writer):
        connection.execute("BEGIN IMMEDIATE")
        connection.executemany("INSERT INTO t VALUES (?)", [("x" * 900,)] * rows)
        connection.execute("COMMIT")
        return store.log_identity(path)

    first = commit(20)
    assert first[1] > 1 and commit(1)[1] > first[1]  # every commit adds its frames
    committed = store.log_identity(path)
    spiller = sqlite3.connect(path, isolation_level=None)
    spiller.execute("PRAGMA cache_size = 5")  # the transaction spills frames into the log before it ends
    spiller.execute("BEGIN IMMEDIATE")
    spiller.executemany("INSERT INTO t VALUES (?)", [("y" * 3000,)] * 300)
    spiller.execute("ROLLBACK")
    spilled = store.log_identity(path)
    assert log.stat().st_size > 32 + 4096 * committed[1] and spilled[:3] == committed[:3] and spilled[3] > committed[3]
    small = commit(1)
    assert small[0] == committed[0] and small[1] > committed[1]  # over the spilled frames, one commit further
    assert writer.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()[0] == 0
    size = log.stat().st_size
    reused = commit(1)  # every frame is in the database: the log restarts in place
    assert log.stat().st_size == size and reused[0] != small[0] and reused[1] < small[1]
    for connection in (spiller, held, writer):
        connection.close()


def test_a_killed_writer_s_unseen_commit_frame_does_not_hide_the_next_commit(tmp_path):
    """A writer killed between writing its commit frame and publishing it in the index leaves a frame
    that looks committed, which no reader sees while another connection keeps the index alive; the
    next commit of the same length overwrites it at the same frame number (Opus reproduced it with
    SIGKILL). Here the orphan is a real commit frame of the same update, taken from a copy of the
    store, so the frame and the salts stay the same: the frame's checksum moves the identity even
    while the log's size and time stay, as a coarse or a held file's time may."""
    path = tmp_path / "log.sqlite"
    with contextlib.closing(sqlite3.connect(path, isolation_level=None)) as setup:
        setup.execute("CREATE TABLE t(v)")
        setup.execute("INSERT INTO t VALUES ('start')")
    writer = store.connect(path, store.TIER_ENFORCED, journal_mode="WAL")  # counted: its file reads borrow
    writer.execute("PRAGMA wal_autocheckpoint = 0")
    writer.execute("UPDATE t SET v = 'committed'")
    held = sqlite3.connect(path, isolation_level=None)  # keeps the index: no recovery adopts the orphan
    assert held.execute("SELECT v FROM t").fetchone() == ("committed",)
    log = path.with_name(path.name + "-wal")
    published = store.log_identity(path)
    copy = tmp_path / "copy.sqlite"
    copy.write_bytes(store.read_file_bytes(path))
    shutil.copyfile(log, copy.with_name(copy.name + "-wal"))
    with contextlib.closing(sqlite3.connect(copy, isolation_level=None)) as other:  # recovers the same log
        other.execute("PRAGMA wal_autocheckpoint = 0")
        other.execute("UPDATE t SET v = 'killed'")  # the killed writer's commit, one frame further
        orphan = copy.with_name(copy.name + "-wal").read_bytes()[log.stat().st_size:]
    with open(log, "r+b") as raw:  # it reached the log, never the index
        raw.seek(log.stat().st_size)
        raw.write(orphan)
    seen, size, written = store.log_identity(path), log.stat().st_size, log.stat().st_mtime_ns
    assert seen[1] == published[1] + 1 and held.execute("SELECT v FROM t").fetchone() == ("committed",)
    writer.execute("UPDATE t SET v = 'next'")  # the next commit of the same length overwrites the orphan
    os.utime(log, ns=(written, written))  # the log's time as a coarse or held file time keeps it
    after = store.log_identity(path)
    assert held.execute("SELECT v FROM t").fetchone() == ("next",) and log.stat().st_size == size
    assert after[:2] == seen[:2], "the same frame of the same generation"
    assert after != seen, "the commit frame's checksum must move the identity"
    for connection in (held, writer):
        connection.close()


def test_a_shorter_commit_over_a_killed_writer_s_longer_one_moves_only_the_log_s_time(tmp_path):
    """A killed writer's unseen commit spans four frames, and the next commit writes one: the
    unseen commit frame stays the log's last, so the frame, its checksum and the log's size stay the
    same, and only the log's time moves the identity (pinned: with the time constant, it would not
    move at all; on a volume that keeps a held file's time it stays until the next commit)."""
    path = tmp_path / "log.sqlite"
    with contextlib.closing(sqlite3.connect(path, isolation_level=None)) as setup:
        setup.execute("CREATE TABLE t(k INTEGER PRIMARY KEY, v)")
        setup.executemany("INSERT INTO t VALUES (?, ?)", [(k, "a" * 3000) for k in range(4)])  # a leaf page each
    writer = store.connect(path, store.TIER_ENFORCED, journal_mode="WAL")  # counted: its file reads borrow
    writer.execute("PRAGMA wal_autocheckpoint = 0")
    writer.execute("UPDATE t SET v = 'b' || substr(v, 2) WHERE k = 0")
    held = sqlite3.connect(path, isolation_level=None)  # keeps the index: no recovery adopts the orphan
    log = path.with_name(path.name + "-wal")
    published = store.log_identity(path)
    copy = tmp_path / "copy.sqlite"
    copy.write_bytes(store.read_file_bytes(path))
    shutil.copyfile(log, copy.with_name(copy.name + "-wal"))
    with contextlib.closing(sqlite3.connect(copy, isolation_level=None)) as other:  # recovers the same log
        other.execute("PRAGMA wal_autocheckpoint = 0")
        other.execute("UPDATE t SET v = 'k' || substr(v, 2)")  # the killed writer's commit: four leaf pages
        orphan = copy.with_name(copy.name + "-wal").read_bytes()[log.stat().st_size:]
    with open(log, "r+b") as raw:  # it reached the log, never the index
        raw.seek(log.stat().st_size)
        raw.write(orphan)
    then = log.stat().st_mtime_ns - 10 ** 10
    os.utime(log, ns=(then, then))  # written ten seconds ago, past any file time's grain
    seen = store.log_identity(path)
    rows = "SELECT group_concat(substr(v, 1, 1), '') FROM t"
    assert seen[1] - published[1] == 4 and held.execute(rows).fetchone() == ("baaa",)  # the orphan unseen
    writer.execute("UPDATE t SET v = 'n' || substr(v, 2) WHERE k = 3")  # one frame: shorter than the orphan
    moved = store.log_identity(path)
    assert held.execute(rows).fetchone() == ("baan",)
    assert [part for part, (now, before) in enumerate(zip(moved, seen)) if now != before] == [4], (seen, moved)
    for connection in (held, writer):
        connection.close()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs FIFOs")
def test_a_fifo_in_the_log_s_place_is_never_opened(tmp_path):
    """Opening a FIFO for reading waits for a writer: the log's identity looks at a regular file only,
    as ``is_sqlite_file`` does, on the history page's request thread and in recovery alike."""
    path = tmp_path / "log.sqlite"
    fifo = path.with_name(path.name + "-wal")
    os.mkfifo(fifo)
    result = []
    reader = threading.Thread(target=lambda: result.append(store.log_identity(path)), daemon=True)
    reader.start()
    reader.join(timeout=10)
    blocked = reader.is_alive()
    if blocked:  # a writer's open ends the reader's wait
        os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
        reader.join(timeout=10)
    assert not blocked and result == [None]


@pytest.mark.parametrize("mode", ["PERSIST", "WAL"])
def test_the_store_state_sees_a_vacuum_that_keeps_every_row(tmp_path, monkeypatch, mode):
    """``committed_content`` cannot show that nothing wrote: a VACUUM keeps every row. The store
    state also holds the write's trace; an ordinary read leaves the build's journal choice as it was."""
    from tests._store_journal_shared import committed_content, store_state, use_journal
    from tests.test_store import document

    use_journal(monkeypatch, mode)
    path = publish(tmp_path, document("t1"), document("t2"))  # the private build already chose MODE
    before = store_state(path)
    with store.open_store(tmp_path, write=False) as txn:  # an ordinary read keeps the file's journal
        assert txn.load("t1")["status"] == "completed"
    assert store_state(path) == before
    with contextlib.closing(store.connect(path, store.TIER_ENFORCED)) as conn:
        conn.execute("VACUUM")
    assert committed_content(path) == before.content and store_state(path) != before


@pytest.mark.parametrize("mode", ["PERSIST", "WAL"])
def test_committed_content_sees_rows_rewritten_in_another_order(tmp_path, monkeypatch, mode):
    """An obligation set and a task's handoffs read back in rowid order (``set_rows``, ``handoffs``),
    so the same rows rewritten in another order are other content: ``committed_content`` shows it as
    it shows a changed row, and sorts only the schema statements a VACUUM reorders. The rewrites go
    through a bare connection, as a store write would also move ``commit_seq``."""
    from collections import Counter

    from tests._store_journal_shared import committed_content, use_journal
    from tests.test_store import document

    use_journal(monkeypatch, mode)
    path = publish(tmp_path, document("t1"))
    with store.open_store(tmp_path, write=True) as txn:
        txn.replace_set("probe", {"a": {"n": 1}, "b": {"n": 2}})
        for op_id in ("op1", "op2"):
            txn.put_handoff("t1", "launch_handoffs", op_id, {"state": "claimed"})

    def rewrite(*statements):
        with contextlib.closing(store.connect(path, store.TIER_ENFORCED)) as conn:
            conn.execute("BEGIN IMMEDIATE")
            for sql in statements:
                conn.execute(sql)
            store.commit(conn)
        return committed_content(path)

    seen = [committed_content(path)]
    for table, first in (("obligations", "set_name='probe' AND id='a'"), ("task_handoffs", "op_id='op1'")):
        seen.append(rewrite(f"CREATE TEMP TABLE moved AS SELECT * FROM {table} WHERE {first}",  # it goes last
                            f"DELETE FROM {table} WHERE {first}", f"INSERT INTO {table} SELECT * FROM moved"))
        assert Counter(seen[-1]) == Counter(seen[-2]) and seen[-1] != seen[-2]  # the same statements, reordered
    with store.open_store(tmp_path, write=False) as txn:
        assert list(txn.set_rows("probe")) == ["b", "a"]
        assert list(txn.handoffs("t1")["launch_handoffs"]) == ["op2", "op1"]
    changed = rewrite("UPDATE obligations SET facts='{}' WHERE set_name='probe' AND id='a'")
    assert changed != seen[-1]
    with contextlib.closing(store.connect(path, store.TIER_ENFORCED)) as conn:
        conn.execute("VACUUM")
    assert committed_content(path) == changed  # a VACUUM keeps each table's row order


# ---- who chooses the journal (ouroboros/store_journal.py) --------------------------------

REPO = pathlib.Path(__file__).resolve().parents[1]
# A peer process on another interpreter, whose SQLite would choose MODE (argv[2]) for every store.
_PEER_WRITE = """
import contextlib, pathlib, sys
from ouroboros import store
store._journal_mode = lambda *_: sys.argv[2]
with contextlib.closing(store.connect(pathlib.Path(sys.argv[1]), store.TIER_ENFORCED, wait_sec=5)) as conn:
    conn.execute("BEGIN IMMEDIATE")
    conn.execute("INSERT OR REPLACE INTO meta VALUES ('peer', '2')")
    store.commit(conn)
    print(conn.execute("PRAGMA journal_mode").fetchone()[0])
"""


def _built(kind, root):
    """The store KIND as its own private build makes it, and its ordinary reader."""
    from ouroboros import delegate_custody_current, dialogue_store

    if kind == "facts":
        return publish(root), lambda: delegate_custody_current.runs(root)  # the helper of 2026-10-20
    if kind == "money":
        usage_store.migrate_from_journal(root)

        def read():
            with usage_store.read(root) as txn:
                return txn.meta("peer")
        return root / usage_store.STORE_REL, read
    return dialogue_store.create(root, tier=store.TIER_ENFORCED), lambda: dialogue_store.meta(root, "peer")


def _peer(path, mode):
    from tests.group_run import run_in_group

    return run_in_group([sys.executable, "-c", _PEER_WRITE, str(path), mode], timeout=60, cwd=str(REPO),
                        env=dict(os.environ))


@pytest.mark.parametrize("kind", ["facts", "money", "dialogue"])
def test_a_reader_whose_sqlite_would_choose_wal_keeps_a_rollback_store(tmp_path, monkeypatch, kind):
    """The incident: the app's SQLite may not share WAL, a helper on another interpreter could. The
    helper's ordinary read (``write=False`` is still ``mode=rw``), cold and cached, and a write of a peer
    process with that policy keep the rollback journal, so the app's own writer beside them never meets
    a WAL file it would have to turn back while the helper holds the log."""
    monkeypatch.setattr(store, "_journal_mode", lambda *_: "PERSIST")
    path, read = _built(kind, tmp_path)
    with contextlib.closing(store.connect(path, store.TIER_ENFORCED)) as owner:  # the app's connection
        assert owner.execute("PRAGMA journal_mode").fetchone()[0] == "persist"
        monkeypatch.setattr(store, "_journal_mode", lambda *_: "WAL")  # the helper's runtime
        read()
        read()
        assert not store.wal_file(path)
        owner.execute("BEGIN IMMEDIATE")
        owner.execute("INSERT OR REPLACE INTO meta VALUES ('owner', '1')")
        store.commit(owner)
        peer = _peer(path, "WAL")
        assert peer.returncode == 0, peer.stderr
        assert peer.stdout.strip().splitlines()[-1] == "persist"
        assert owner.execute("SELECT value FROM meta WHERE key='peer'").fetchone()[0] == "2"
        read()
        assert not store.wal_file(path)


@pytest.mark.parametrize("kind", ["facts", "money", "dialogue"])
def test_a_reader_whose_sqlite_must_not_share_wal_refuses_a_wal_store_unchanged(tmp_path, monkeypatch, kind):
    """The reverse: a WAL store this runtime's SQLite, tier or filesystem must not share is refused typed
    at an ordinary open, cold and again, in this process and in a peer: it is neither used under the
    WAL-reset race (sqlite.org/wal.html §11) nor turned back while its owner holds the log, and the
    owner's connection keeps working. Only a boot or handoff of that runtime leaves WAL."""
    monkeypatch.setattr(store, "_journal_mode", lambda *_: "WAL")
    path, read = _built(kind, tmp_path)
    store.close_connections()
    with contextlib.closing(store.connect(path, store.TIER_ENFORCED)) as owner:
        assert owner.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        owner.execute("SELECT * FROM meta").fetchall()
        monkeypatch.setattr(store, "_journal_mode", lambda *_: "PERSIST")
        for _ in range(2):
            with pytest.raises((store.StoreError, usage_store.UsageAccountingError), match="journal is WAL"):
                read()
        peer = _peer(path, "PERSIST")
        assert peer.returncode != 0 and "journal is WAL" in peer.stderr
        assert store.wal_file(path)
        owner.execute("BEGIN IMMEDIATE")
        owner.execute("INSERT OR REPLACE INTO meta VALUES ('owner', '1')")
        store.commit(owner)
        assert owner.execute("SELECT value FROM meta WHERE key='owner'").fetchone()[0] == "1"


@pytest.mark.parametrize("mode", ["PERSIST", "WAL"])
def test_a_private_build_publishes_its_choice_with_nothing_beside_it(tmp_path, monkeypatch, mode):
    """Every private build (facts, money, dialogue, the evidence index) is renamed into place in the journal
    it chose, WAL included, with no log or index and no hot journal under its temporary name. The index
    build writes on through ordinary handles, whose rollback PERSIST leaves a zeroed journal there; its own
    cleanup removes that once the database moved, as before this change."""
    from ouroboros import dialogue_store, evidence_packs

    monkeypatch.setattr(store, "_journal_mode", lambda *_: mode)
    replace, published, temporaries = os.replace, {}, []
    hot = bytes([0xD9, 0xD5, 0x05, 0xF9, 0x20, 0xA1, 0x63, 0xD7])  # a rollback journal's header magic

    def inspect(source, target):
        source = pathlib.Path(source)
        if source.name.startswith((".index.sqlite.tmp.", "current.sqlite.import-", "usage.sqlite.import-",
                                   "dialog.sqlite.import-")):
            assert [suffix for suffix in ("-wal", "-shm") if pathlib.Path(f"{source}{suffix}").exists()] == []
            journal = pathlib.Path(f"{source}-journal")
            if journal.exists():
                assert source.name.startswith(".index.sqlite.tmp.") and mode == "PERSIST"
                assert journal.read_bytes()[:8] != hot, "never a hot journal beside a published database"
            assert store.wal_file(source) is (mode == "WAL")
            published[pathlib.Path(target).name] = pathlib.Path(target)
            temporaries.append(source)
        return replace(source, target)

    monkeypatch.setattr(os, "replace", inspect)
    store_import.ensure_current_store(tmp_path, with_children=False)
    usage_store.migrate_from_journal(tmp_path)
    dialogue_store.create(tmp_path, tier=store.TIER_ENFORCED)
    with evidence_packs.open_index(tmp_path):
        pass
    monkeypatch.setattr(os, "replace", replace)
    store.close_connections()
    assert sorted(published) == ["current.sqlite", "dialog.sqlite", "index.sqlite", "usage.sqlite"]
    assert [path for path in temporaries for suffix in ("", "-journal", "-wal", "-shm")
            if pathlib.Path(f"{path}{suffix}").exists()] == []
    for target in published.values():
        assert store.wal_file(target) is (mode == "WAL")
        with contextlib.closing(sqlite3.connect(target)) as conn:
            assert conn.execute("PRAGMA quick_check").fetchall() == [("ok",)]


def _three_stores(root):
    from ouroboros import dialogue_import, dialogue_store

    store_import.ensure_current_store(root, with_children=False)
    usage_store.migrate_from_journal(root)
    dialogue_import.cutover(root)
    store.close_connections()
    return [store.store_path(root), root / usage_store.STORE_REL, dialogue_store.path(root)]


def test_only_the_boot_promotes_an_existing_store_to_wal(tmp_path, monkeypatch):
    """A rollback store stays rollback for every ordinary and lazy owner of a runtime that may use WAL
    (a first-touch import check, the supervisor's money revival, a lazy cutover); only the boot owners,
    before their producers, choose WAL for it, and every reader serves afterwards."""
    from ouroboros import dialogue_import, dialogue_store

    monkeypatch.setattr(store, "_journal_mode", lambda *_: "PERSIST")
    paths = _three_stores(tmp_path)
    monkeypatch.setattr(store, "_journal_mode", lambda *_: "WAL")
    store_import.ensure_current_store(tmp_path, with_children=False)
    usage_store.migrate_from_journal(tmp_path)
    dialogue_import.cutover(tmp_path)
    with store.open_store(tmp_path, write=False):
        pass
    store_import.ensure_current_store(tmp_path, with_children=False, body_sha="helper-stamp")
    assert [store.wal_file(path) for path in paths] == [False, False, False]
    assert store_import.ensure_current_store(tmp_path, with_children=False, body_sha="b" * 40, initialize_journal=True)["journal_selected"]
    assert usage_store.migrate_from_journal(tmp_path, initialize_journal=True)["journal_selected"]
    assert dialogue_import.cutover(tmp_path, initialize_journal=True)["journal_selected"]  # server._dialogue_cutover
    assert [store.wal_file(path) for path in paths] == [True, True, True]
    with store.open_store(tmp_path, write=True) as txn:
        txn.set_meta("after_boot", 1)
    with usage_store.read(tmp_path) as txn:
        assert txn.meta("import")["status"] == "completed"
    assert dialogue_store.meta(tmp_path, "cutover") is not None


def test_the_boot_leaves_wal_its_runtime_must_not_share_before_any_read(tmp_path, monkeypatch):
    """A WAL store met by a runtime that must not share WAL: every ordinary owner refuses it typed, and
    the boot owners leave WAL first, under the tier the file is used with, then read and serve it."""
    from ouroboros import dialogue_import, dialogue_store

    monkeypatch.setattr(store, "_journal_mode", lambda *_: "WAL")
    paths = _three_stores(tmp_path)
    monkeypatch.setattr(store, "_journal_mode", lambda *_: "PERSIST")
    with pytest.raises(store.StoreError, match="journal is WAL"):
        store_import.ensure_current_store(tmp_path, with_children=False)
    with pytest.raises(usage_store.UsageAccountingError, match="journal is WAL"):
        usage_store.migrate_from_journal(tmp_path)
    with pytest.raises(store.StoreError, match="journal is WAL"):
        dialogue_import.cutover(tmp_path)
    assert [store.wal_file(path) for path in paths] == [True, True, True]
    store_import.ensure_current_store(tmp_path, with_children=False, body_sha="b" * 40, initialize_journal=True)
    usage_store.migrate_from_journal(tmp_path, initialize_journal=True)
    dialogue_import.cutover(tmp_path, initialize_journal=True)
    assert [store.wal_file(path) for path in paths] == [False, False, False]
    assert all(not path.with_name(path.name + suffix).exists() for path in paths for suffix in ("-wal", "-shm"))
    with store.open_store(tmp_path, write=True) as txn:
        txn.set_meta("after_boot", 1)
        assert txn.conn.execute("PRAGMA journal_mode").fetchone()[0] == "persist"
    with usage_store.read(tmp_path) as txn:
        assert txn.meta("import")["status"] == "completed"
    assert dialogue_store.meta(tmp_path, "cutover") is not None


def test_a_boot_promotion_defers_for_an_early_reader_and_the_store_keeps_serving(tmp_path, monkeypatch, caplog):
    """Optional tuning: an early reader's lease at boot defers WAL with a warning; the rollback store keeps
    serving, the next boot chooses again, and a matching journal is a no-op that keeps every handle."""
    from ouroboros import store_journal

    monkeypatch.setattr(store, "_journal_mode", lambda *_: "PERSIST")
    path = publish(tmp_path)
    monkeypatch.setattr(store, "_journal_mode", lambda *_: "WAL")
    with store.open_store(tmp_path, write=False) as txn:  # an early HTTP reader holds its lease
        assert store_journal.configure_journal(path, store.TIER_ENFORCED, wait_sec=0) is False
        assert txn.phase() == store.PUBLISHED
    assert not store.wal_file(path) and "deferred" in caplog.text
    with store.open_store(tmp_path, write=True) as txn:
        txn.set_meta("after_reader", True)
    assert store_journal.configure_journal(path, store.TIER_ENFORCED) is True and store.wal_file(path)
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.meta("after_reader") is True
        held = txn.conn
    assert store_journal.configure_journal(path, store.TIER_ENFORCED) is True  # matching: nothing closes
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.conn is held


def test_a_required_wal_exit_never_defers_silently(tmp_path, monkeypatch):
    """Leaving WAL is compatibility, not tuning: while another connection holds the log the boot's
    selection raises (the boot fails visibly) and this runtime's ordinary opens refuse the file; once
    that holder is gone the same selection leaves WAL and the store serves."""
    from ouroboros import store_journal

    monkeypatch.setattr(store, "_journal_mode", lambda *_: "WAL")
    path = publish(tmp_path)
    store.close_connections()
    holder = sqlite3.connect(path, isolation_level=None)
    try:
        holder.execute("SELECT * FROM meta").fetchall()  # another runtime's WAL connection, open for its life
        monkeypatch.setattr(store, "_journal_mode", lambda *_: "PERSIST")
        with pytest.raises((store.StoreBusy, sqlite3.OperationalError)):
            store_journal.configure_journal(path, store.TIER_ENFORCED, wait_sec=0.2)
        assert store.wal_file(path)
        with pytest.raises(store.StoreError, match="journal is WAL"):
            with store.open_store(tmp_path, write=False):
                pass
    finally:
        holder.close()
    assert store_journal.configure_journal(path, store.TIER_ENFORCED) is True and not store.wal_file(path)
    with store.open_store(tmp_path, write=True) as txn:
        txn.set_meta("served", True)


@pytest.mark.parametrize("which", ["facts", "dialogue"])
@pytest.mark.parametrize("conversion", ["completes", "busy"])
def test_a_retier_leaves_wal_under_the_old_tier_before_the_header_names_the_new_one(tmp_path, monkeypatch, which,
                                                                                    conversion):
    """Enforced (WAL) -> name tier: the file leaves WAL through the enforced protocol it is used with, and
    only then does its header name the name tier, whose VFS takes no locks. A conversion that cannot
    finish publishes nothing: the old tier and its WAL file stay, and the next retier tries again."""
    from ouroboros import dialogue_store, platform_layer

    monkeypatch.setattr(store, "_journal_mode", lambda _path, tier: "WAL" if tier == store.TIER_ENFORCED else "PERSIST")
    monkeypatch.setattr(platform_layer, "kernel_file_locks_enforced", lambda path, **kw: True)
    if which == "facts":
        store_import.ensure_current_store(tmp_path, with_children=False)
        path, read_tier = store.store_path(tmp_path), lambda: store.probe(tmp_path)[1]
        identity = store_import._mount_identity(tmp_path / "state")
        monkeypatch.setattr(store_import, "_mount_identity", lambda _p: {**identity, "st_dev": identity["st_dev"] + 1})
        retier = lambda: store_import.ensure_current_store(tmp_path, with_children=False)  # noqa: E731
    else:
        path, read_tier = dialogue_store.create(tmp_path), lambda: dialogue_store.header_tier(tmp_path)
        retier = lambda: dialogue_store.retier(tmp_path)  # noqa: E731
    store.close_connections()
    assert store.wal_file(path) and read_tier() == store.TIER_ENFORCED
    monkeypatch.setattr(platform_layer, "kernel_file_locks_enforced", lambda path, **kw: False)
    seen, real = [], store.prepare_handoff

    def handoff(target, tier, **kwargs):
        seen.append((tier, read_tier(), store.wal_file(target)))
        if conversion == "busy":
            raise store.StoreBusy("store checkpoint busy at handoff", reason="contention")
        return real(target, tier, **kwargs)

    monkeypatch.setattr(store, "prepare_handoff", handoff)
    if conversion == "busy":
        with pytest.raises(store.StoreBusy):
            retier()
        assert store.wal_file(path) and read_tier() == store.TIER_ENFORCED
    else:
        retier()
        assert not store.wal_file(path) and read_tier() == store.TIER_NAME
    assert seen[0] == (store.TIER_ENFORCED, store.TIER_ENFORCED, True)


def test_a_wal_evidence_index_another_runtime_built_is_rebuilt_not_converted(tmp_path, monkeypatch):
    """The evidence index is a cache with its own private build: a WAL index that this runtime must not
    share fails validation instead of being converted, and the rebuild publishes this runtime's choice
    with the same rows (lazy child stores keep their creator's choice the same way)."""
    from ouroboros import evidence_packs as packs
    from tests.test_evidence_index_rebuild import _publish, _tasks

    monkeypatch.setattr(store, "_journal_mode", lambda *_: "WAL")
    _publish(tmp_path, "t1")
    assert _tasks(tmp_path) == {"t1"}
    index = packs.packs_dir(tmp_path) / packs.INDEX_NAME
    store.close_connections()
    assert store.wal_file(index)
    monkeypatch.setattr(store, "_journal_mode", lambda *_: "PERSIST")
    monkeypatch.setattr(packs, "_VALIDATED", {})
    assert _tasks(tmp_path) == {"t1"}
    store.close_connections()
    assert not store.wal_file(index)


def test_body_stamp_and_rebuild_cli_do_not_promote_current_store(tmp_path, monkeypatch, capsys):
    """The public repair CLI keeps the published journal even on a WAL-eligible interpreter."""
    from ouroboros import startup_migrations

    monkeypatch.setattr(store, "_journal_mode", lambda *_: "PERSIST")
    publish(tmp_path)
    monkeypatch.setattr(store, "_journal_mode", lambda *_: "WAL")
    store_import.ensure_current_store(tmp_path, body_sha="helper", with_children=False)
    monkeypatch.setattr(sys, "argv", ["startup_migrations", "--data-root", str(tmp_path)])
    startup_migrations.main()
    assert not store.wal_file(store.store_path(tmp_path))
    assert json.loads(capsys.readouterr().out)["status"] in {"unchanged", "reconciled"}
    startup_migrations.prepare_startup_state(tmp_path, initialize_journal=True)
    assert store.wal_file(store.store_path(tmp_path))


def test_boot_retries_a_transient_pooled_lease_before_required_wal_exit(tmp_path, monkeypatch):
    """A real HTTP-style pooled lease delays mandatory demotion but does not lose the boot."""
    import threading
    import time
    from ouroboros import startup_migrations, store_journal

    monkeypatch.setattr(store, "_journal_mode", lambda *_: "WAL")
    path = publish(tmp_path)
    entered, release = threading.Event(), threading.Event()
    errors = []
    def reader():
        try:
            with store.open_store(tmp_path, write=False) as txn:
                assert txn.phase() == store.PUBLISHED
                entered.set()
                assert release.wait(5)
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=reader)
    thread.start()
    assert entered.wait(5)
    monkeypatch.setattr(store, "_journal_mode", lambda *_: "PERSIST")
    waits = []
    def sleep(delay):
        waits.append(delay)
        release.set()
        thread.join(5)
        assert not thread.is_alive()
    from types import SimpleNamespace
    monkeypatch.setattr(store_journal, "time", SimpleNamespace(monotonic=time.monotonic, sleep=sleep))
    try:
        report = startup_migrations.prepare_startup_state(tmp_path, initialize_journal=True)
    finally:
        release.set()
        thread.join(5)
    assert not errors and waits and report["status"] == "unchanged"
    assert not store.wal_file(path)
    with store.open_store(tmp_path, write=True) as txn:
        txn.set_meta("boot_serves", True)


@pytest.mark.parametrize("failure", ["contention", "permission", "io"])
def test_required_demotion_retries_only_contention_with_one_budget(tmp_path, monkeypatch, failure):
    from types import SimpleNamespace
    from ouroboros import store_journal

    monkeypatch.setattr(store, "_journal_mode", lambda *_: "WAL")
    path = publish(tmp_path)
    monkeypatch.setattr(store, "_journal_mode", lambda *_: "PERSIST")
    clock, waits, calls = [0.0], [], []
    error = (store.StoreBusy("lease held", reason="contention") if failure == "contention" else
             PermissionError("refused") if failure == "permission" else sqlite3.OperationalError("disk I/O error"))
    def handoff(*args, wait_sec, **kwargs):
        calls.append(wait_sec)
        raise error
    def sleep(delay):
        waits.append(delay)
        clock[0] += delay
    monkeypatch.setattr(store, "prepare_handoff", handoff)
    monkeypatch.setattr(store_journal, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep))
    with pytest.raises(type(error)):
        store_journal.configure_journal(path, store.TIER_ENFORCED, wait_sec=0.12)
    assert store.wal_file(path)
    assert sum(waits) <= 0.120001 and all(later <= earlier for earlier, later in zip(calls, calls[1:]))
    assert bool(waits) is (failure == "contention")
    if failure != "contention":
        assert len(calls) == 1


@pytest.mark.parametrize("mode", ["WAL", "PERSIST"])
def test_boot_prepares_existing_child_stores_before_lazy_consumers(tmp_path, monkeypatch, mode):
    """Existing descendants need compatibility even after the children import job finished. The
    header pass creates no absent store, promotes no rollback child, and leaves all rows intact."""
    from ouroboros import dialogue_store, startup_migrations
    from tests._store_journal_shared import committed_content

    monkeypatch.setattr(store, "_journal_mode", lambda *_: mode)
    child = tmp_path / "task_drives" / "child"
    nested = child / "state" / "headless_tasks" / "nested" / "data"
    empty = tmp_path / "task_drives" / "empty"
    empty.mkdir(parents=True)
    paths = _three_stores(child) + _three_stores(nested)
    store_import.ensure_current_store(tmp_path)
    before = {path: committed_content(path) for path in paths}
    monkeypatch.setattr(store, "_journal_mode", lambda *_: "PERSIST" if mode == "WAL" else "WAL")
    if mode == "WAL":
        with pytest.raises(store.StoreError, match="journal is WAL"):
            dialogue_store.meta(child, "cutover")
    startup_migrations.prepare_startup_state(tmp_path, initialize_journal=True)
    assert all(not store.wal_file(path) for path in paths)
    assert all(committed_content(path) == before[path] for path in paths)
    assert not (empty / "state" / "dialog.sqlite").exists()
    for root in (child, nested):
        with store.open_store(root, write=True) as txn:
            txn.set_meta("lazy_writer", True)
        assert dialogue_store.meta(root, "cutover") is not None
        with usage_store.read(root) as txn:
            assert txn.meta("import")["status"] == "completed"


@pytest.mark.parametrize("mode", ["PERSIST", "WAL"])
def test_an_open_read_blocks_a_commit_only_under_a_rollback_journal(tmp_path, monkeypatch, mode):
    """The convoy WAL removes (ouroboros#2341): under a rollback journal an open read's SHARED lock holds a
    writer's COMMIT (here, with no busy wait, a typed StoreBusy and the write rolled back); under WAL the
    write commits beside the open read, and the read's snapshot keeps its own answer either way."""
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 53, 1) if mode == "WAL" else (3, 50, 4))
    monkeypatch.setattr(store, "_filesystem_type", lambda _path: "apfs")
    publish(tmp_path, {"task_id": "t", "status": "running"})

    def write():  # nested inside the read: its own connection
        with store.acquisition_budget(0), store.open_store(tmp_path, write=True) as writer:
            writer.conn.execute("UPDATE tasks SET status='completed' WHERE task_id='t'")

    with store.open_store(tmp_path, write=False) as reader:
        assert reader.conn.execute("PRAGMA journal_mode").fetchone()[0] == mode.lower()
        assert reader.row("t", "status") == "running"
        if mode == "WAL":
            write()
        else:
            with pytest.raises(store.StoreBusy):
                write()
        assert reader.row("t", "status") == "running"
    with store.open_store(tmp_path, write=False) as after:
        assert after.row("t", "status") == ("completed" if mode == "WAL" else "running")


def test_a_forced_journal_a_late_thread_still_leases_is_given_back_at_a_later_teardown(tmp_path, monkeypatch):
    """The teardown give-back of a forced journal leaves a store another thread still leases to a later teardown
    instead of failing the test that ends, and converts it once the lease is gone (tests/_store_journal_shared.py)."""
    from tests import _store_journal_shared as shared

    root = tmp_path / "root"
    publish(root, {"task_id": "t", "status": "completed", "_schema_version": 1})
    path = os.path.abspath(store.store_path(root))
    with store.open_store(root, write=False) as txn:
        native = txn.conn.execute("PRAGMA journal_mode").fetchone()[0]
    shared.use_journal(monkeypatch, "PERSIST" if native == "wal" else "WAL")
    leased, release = threading.Event(), threading.Event()

    def late_reader():
        with store.open_store(root, write=False):
            leased.set()
            release.wait(10)

    reader = threading.Thread(target=late_reader)
    reader.start()
    try:
        assert leased.wait(10)
        monkeypatch.undo()  # what the teardown hook sees: every monkeypatch undone
        shared.give_back_converted_journals()  # the lease is the other thread's: nothing raises here
        assert any(os.path.abspath(converted) == path for converted, _tier in shared._CONVERTED)
    finally:
        release.set()
        reader.join(10)
    shared.give_back_converted_journals()
    assert not any(os.path.abspath(converted) == path for converted, _tier in shared._CONVERTED)
    with store.open_store(root, write=False) as txn:
        assert txn.conn.execute("PRAGMA journal_mode").fetchone()[0] == native
