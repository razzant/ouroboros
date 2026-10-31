"""A SQLite WAL index (``-shm``) is never opened by the readers that apply the index rule.

Under WAL every open connection holds locks on the index. POSIX record locks (its read marks and
the dead-man switch) drop when ANY descriptor of the file closes (sqlite.org/howtocorrupt.html
§2.2): after a raw open and close of the index, another process takes the TRUNCATE checkpoint those
locks forbid and resets the log under a live read transaction, which then reads that process's
later commit (reproduced on macOS with SQLite 3.53.1). Windows byte-range locks fail a read across
them. So the rule is the file, on every OS: the regular file the volume resolves a SQLite database's
name plus ``-shm`` to is an index, whichever process holds the database and whenever this one connects
(who holds it when a read starts says nothing about when the read's descriptor closes). The store's
raw readers refuse it (the Files panel answers 403, the read tools and single-file copies refuse it);
the folder copies, captures, delegated snapshots and artifact listings built on them leave it out as
an excluded member, tracked or not, never read; a store's database is read through its borrowed
descriptor, never a clone that opens it. The ``-wal`` log carries no lock and reads as any file does,
as does a file merely named like a sidecar, a directory, or a spelling the volume keeps apart.
"""
from __future__ import annotations

import asyncio
import contextlib
import errno
import json
import os
import pathlib
import sqlite3
import stat
import subprocess
import sys

import pytest

from ouroboros import store
from tests._store_journal_shared import use_journal

pytestmark = pytest.mark.serial  # another interpreter probes the store's locks

_RESET = ("import sqlite3, sys\n"
          "conn = sqlite3.connect(sys.argv[1], isolation_level=None, timeout=0)\n"
          "print(conn.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()[0])\n")
_KINDS = ["preview", "agent_read", "copy", "download", "stream", "artifact", "artifact_copy"]


def _reset_refused(database) -> bool:
    """Whether another process is refused the log reset: true while a read transaction of this
    process holds its read mark in the index."""
    out = subprocess.run([sys.executable, "-c", _RESET, str(database)], capture_output=True, text=True,
                         timeout=60, check=True).stdout.strip()
    return out == "1"


_SHARED = ("import fcntl, os, sys\n"  # SQLite's unix SHARED range: PENDING_BYTE + 2, 510 bytes
           "fd = os.open(sys.argv[1], os.O_RDWR)\n"
           "try:\n"
           "    fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB, 510, 0x40000002)\n"
           "    print('free')\n"
           "except OSError:\n"
           "    print('held')\n")


def _shared_held(database) -> bool:
    """Whether this process's connection still holds its SHARED lock on DATABASE, which closing any
    descriptor of the file drops; Windows locks belong to their handle, so there it always holds."""
    if os.name == "nt":
        return True
    out = subprocess.run([sys.executable, "-I", "-c", _SHARED, str(database)], capture_output=True, text=True,
                         timeout=60, check=True).stdout.strip()
    return out == "held"


def _live_store(root):
    """A WAL store whose last commit is still in its log."""
    from ouroboros import store_import, task_store

    store_import.ensure_current_store(root)
    task_store.patch_document(root, "t1", lambda _current: {"task_id": "t1", "status": "running"})
    database = store.store_path(root)
    assert store.wal_file(database) and store.log_identity(database) is not None
    return database


def _read(kind, root, target):
    """One in-process reader of TARGET, as the product reads a file someone asked for."""
    from ouroboros.artifacts import copy_artifact_file, stream_artifact_file
    from ouroboros.gateway import files
    from ouroboros.tools import core_file_tools

    if kind == "preview":
        return files._read_prefix(target, 4096)
    if kind == "agent_read":
        with contextlib.suppress(UnicodeDecodeError):  # a log is not text; the read itself happened
            return core_file_tools._read_source_text(target, None)
        return None
    if kind == "copy":
        files._copy_path(target, root / "copied")
        return (root / "copied").read_bytes()
    if kind == "download":
        return files._file_response(target, filename=target.name)
    if kind == "stream":
        return store.read_file_bytes(target)
    if kind == "artifact_copy":  # on Linux a block-sharing clone would open and close the source itself
        return copy_artifact_file(target, root / "copied")
    return stream_artifact_file(target)


@pytest.mark.parametrize("kind", _KINDS)
def test_a_reader_refuses_the_index_and_the_read_transaction_keeps_its_locks(tmp_path, monkeypatch, kind):
    use_journal(monkeypatch, "WAL")
    database = _live_store(tmp_path)
    index = database.with_name(database.name + "-shm")
    with store.open_store(tmp_path, write=False):  # a read transaction: its read mark is locked
        assert _reset_refused(database)
        if kind == "download":
            response = _read(kind, tmp_path, index)
            assert response.status_code == 403 and b"WAL index" in response.body
        else:
            with pytest.raises(store.WalIndexRefused):
                _read(kind, tmp_path, index)
        assert _reset_refused(database), "a descriptor of the index was opened and closed"
    assert not (tmp_path / "copied").exists()


def test_the_files_panel_answers_403_for_the_index_and_keeps_the_locks(tmp_path, monkeypatch):
    """The preview, the download and the content view each refuse the index before opening it."""
    from starlette.applications import Starlette
    from starlette.testclient import TestClient

    from ouroboros.gateway import files

    use_journal(monkeypatch, "WAL")
    database = _live_store(tmp_path)
    monkeypatch.setenv("OUROBOROS_FILE_BROWSER_DEFAULT", str(tmp_path))
    index = database.relative_to(tmp_path).as_posix() + "-shm"
    with store.open_store(tmp_path, write=False), TestClient(Starlette(routes=files.file_browser_routes())) as client:
        for endpoint in ("read", "download", "content"):
            response = client.get(f"/api/files/{endpoint}", params={"path": index})
            assert response.status_code == 403 and "WAL index" in response.json()["error"], endpoint
        assert _reset_refused(database)


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("name", ["current.sqlite-wal", "notes-wal", "notes-shm", "current.sqlite"])
def test_the_log_and_ordinary_files_still_read_and_keep_the_locks(tmp_path, monkeypatch, kind, name):
    """The store's own log, a file merely named like a sidecar (one beside a sibling that is no
    database) and the database itself (through its borrowed descriptor) read as before."""
    from starlette.responses import FileResponse, StreamingResponse

    use_journal(monkeypatch, "WAL")
    database = _live_store(tmp_path)
    target = database.with_name(name)
    if name.startswith("notes"):
        (database.parent / "notes").write_text("a sibling that is not a database", encoding="utf-8")
        target.write_text("plain text\n", encoding="utf-8")
    with store.open_store(tmp_path, write=False):
        size = target.stat().st_size
        result = _read(kind, tmp_path, target)
        assert _reset_refused(database), "reading this file dropped the index's locks"
    if kind == "download":
        assert isinstance(result, (FileResponse, StreamingResponse)) and result.status_code == 200
    elif kind == "preview":
        assert len(result) == min(size, 4096)
    elif kind in {"copy", "stream"}:
        assert len(result) == size
    elif kind in {"artifact", "artifact_copy"}:
        assert result["size"] == size
    elif name.startswith("notes"):
        assert result == "plain text\n"


def test_the_header_probe_answers_without_opening_the_index(tmp_path, monkeypatch):
    """``is_sqlite_file`` reads a header to answer, so asked about the index it must not look."""
    use_journal(monkeypatch, "WAL")
    database = _live_store(tmp_path)
    index = database.with_name(database.name + "-shm")
    with store.open_store(tmp_path, write=False):
        assert store.is_wal_index(index) and not store.is_sqlite_file(index)
        assert _reset_refused(database)
    assert store.is_sqlite_file(database) and not store.is_wal_index(database)


def test_the_tests_tree_snapshot_passes_the_index_by_name(tmp_path, monkeypatch):
    """The tests' own "a read writes nothing" snapshot (``tree_state``) walks a folder holding a live
    store: it must leave the index unopened, or it drops the locks a read transaction holds."""
    from tests._store_journal_shared import tree_state

    use_journal(monkeypatch, "WAL")
    database = _live_store(tmp_path)
    with store.open_store(tmp_path, write=False):
        assert _reset_refused(database)
        snapshot = tree_state(tmp_path, mtimes=True)
        assert _reset_refused(database), "the snapshot opened and closed a descriptor of the index"
    assert "state/current.sqlite" in snapshot and "state/current.sqlite-shm" not in snapshot


def test_the_index_is_refused_by_any_name_that_reaches_it(tmp_path, monkeypatch):
    """A symbolic link to the index, and on a case-insensitive volume any spelling the volume folds
    onto it (``-SHM``, ``-ſhm``), open the same file: each is refused as the index is, and the read
    transaction keeps its lock. A spelling the volume keeps apart reaches no index and is none."""
    from ouroboros.gateway import files

    use_journal(monkeypatch, "WAL")
    database = _live_store(tmp_path)
    index = database.with_name(database.name + "-shm")
    spellings = [database.with_name(database.name + suffix) for suffix in ("-SHM", "-\u017fhm")]
    names = [name for name in spellings if name.exists()]  # a case-insensitive volume: the same file
    assert [store.is_wal_index(name) for name in spellings] == [name in names for name in spellings]
    link = tmp_path / "linked-index"
    with contextlib.suppress(OSError):  # Windows without the symbolic-link privilege
        link.symlink_to(index)
        names.append(link)
    with store.open_store(tmp_path, write=False):
        for name in names:
            assert store.is_wal_index(name), name
            with pytest.raises(store.WalIndexRefused):
                files._read_prefix(name, 4096)
        assert _reset_refused(database)


@pytest.mark.parametrize("ending", [b"\n", b"\r\n"], ids=["LF", "CRLF"])
def test_a_spelling_the_volume_keeps_apart_from_the_index_is_the_user_s_file(tmp_path, monkeypatch, ending):
    """``x.db-SHM`` is an index only where the volume resolves it to ``x.db-shm``'s own file: on a
    case-sensitive volume it is the user's file, read, copied and listed as one (folding its name made
    it an index by name alone, so every reader refused it and every copy left it out). A host volume that
    folds the spelling answers here as a case-sensitive one would: a name reaches a file only through a
    directory entry spelled exactly so."""
    from ouroboros import task_custody
    from ouroboros.gateway import files

    folder = tmp_path / "artifacts"
    folder.mkdir()
    with contextlib.closing(sqlite3.connect(folder / "x.db")) as conn:
        conn.execute("CREATE TABLE t(v)")
    notes = folder / "x.db-SHM"
    payload = b"the user's notes" + ending
    notes.write_bytes(payload)
    if (folder / "x.db-shm").exists():  # the host volume folds case: emulate one that does not
        same = os.path.samefile
        monkeypatch.setattr(os.path, "samefile", lambda a, b: same(a, b) and all(
            pathlib.Path(p).name in os.listdir(pathlib.Path(p).parent) for p in (a, b)))
    assert store.is_wal_index(notes) is False
    assert store.read_file_bytes(notes) == payload
    files._copy_path(folder, tmp_path / "copy")
    assert (tmp_path / "copy" / "x.db-SHM").read_bytes() == payload
    assert sorted(task_custody._listed(folder)) == ["x.db", "x.db-SHM"]


@pytest.mark.parametrize("database, folder", [("x", "x-shm"), ("current-shm", "current-shm-shm")])
def test_a_directory_named_like_an_index_is_copied_and_captured_whole(tmp_path, database, folder):
    """SQLite's index is a regular file: a directory under the index's name beside a closed database
    (``x-shm/`` beside ``x``, ``current-shm-shm/`` beside ``current-shm``) is the user's, and the Files
    folder copy and a capture keep it whole (the copy skipped its subtree and still answered ok)."""
    from ouroboros import workspace_patch_capture as capture
    from ouroboros.gateway import files

    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (["init"], ["-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                            "commit", "--allow-empty", "-m", "base"]):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    with contextlib.closing(sqlite3.connect(repo / database)) as conn:
        conn.execute("CREATE TABLE t(v)")
    (repo / folder).mkdir()
    (repo / folder / "report.txt").write_text("the user's report\n", encoding="utf-8")
    files._copy_path(repo, tmp_path / "copy")
    assert sorted(path.name for path in (tmp_path / "copy").iterdir()) == sorted([".git", database, folder])
    assert (tmp_path / "copy" / folder / "report.txt").read_text(encoding="utf-8") == "the user's report\n"
    _, manifest = capture.write_workspace_patch_artifacts(repo, tmp_path / "capture", task={})
    assert manifest["status"] == "ready_with_changes", manifest["errors"]
    assert f"{folder}/report.txt" in manifest["untracked_included"]
    assert store.is_wal_index(repo / folder) is False


def test_a_file_named_only_by_the_suffix_is_no_index(tmp_path):
    """A file named just ``-shm`` names no database beside it: it is no index, and every reader and
    copy takes it as the file it is (its empty base name raised ValueError out of these probes)."""
    from ouroboros.gateway import files

    folder = tmp_path / "odd"
    folder.mkdir()
    (folder / "-shm").write_bytes(b"\0" * 64)
    assert store.is_wal_index(folder / "-shm") is False and store.is_sqlite_file(folder / "-shm") is False
    assert store.read_file_bytes(folder / "-shm") == b"\0" * 64
    files._copy_path(folder, tmp_path / "odd-copy")
    assert (tmp_path / "odd-copy" / "-shm").read_bytes() == b"\0" * 64


_UPDATER = ("import sqlite3, sys\n"
            "conn = sqlite3.connect(sys.argv[1], isolation_level=None)\n"
            "conn.execute('PRAGMA journal_mode=WAL')\n"
            "conn.execute('PRAGMA wal_autocheckpoint=0')\n"
            "conn.execute('CREATE TABLE IF NOT EXISTS t(v)')\n"
            "conn.execute('BEGIN')\n"
            "conn.executemany('INSERT INTO t VALUES (?)', [('x' * 3000,)] * 300)\n"
            "conn.execute('COMMIT')\n"
            "print('ready', flush=True)\n"
            "for line in sys.stdin:\n"
            "    conn.execute(\"UPDATE t SET v = 'y' || substr(v, 2) WHERE rowid % 2 = 0\")\n"
            "    print('updated', flush=True)\n")
_KEEPER = ("import sqlite3, sys\n"
           "conn = sqlite3.connect(sys.argv[1], isolation_level=None)\n"
           "conn.execute('PRAGMA user_version').fetchone()\n"
           "print('ready', flush=True)\n"
           "for line in sys.stdin:\n"
           "    print('updated', flush=True)\n")
_HOLDER = ("import sqlite3, sys\n"
           "conn = sqlite3.connect(sys.argv[1], isolation_level=None)\n"
           "conn.execute('PRAGMA journal_mode=WAL')\n"
           "conn.execute('CREATE TABLE IF NOT EXISTS t(v)')\n"
           "conn.execute(\"INSERT INTO t VALUES ('dev server row')\")\n"
           "print('ready', flush=True)\n"
           "sys.stdin.read()\n")  # holds the database open until the test closes its input


@contextlib.contextmanager
def _held_elsewhere(database, script):
    """Another process holding DATABASE open (SCRIPT), until the block ends; the block's ``update()``
    asks it for one commit, where its script makes one."""
    holder = subprocess.Popen([sys.executable, "-I", "-c", script, str(database)],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    def update():
        holder.stdin.write("update\n")
        holder.stdin.flush()
        assert holder.stdout.readline().strip() == "updated"

    try:
        assert holder.stdout.readline().strip() == "ready"
        yield update
    finally:
        holder.stdin.close()
        try:
            holder.wait(timeout=60)
        except subprocess.TimeoutExpired:
            holder.kill()
            holder.wait()


@pytest.mark.parametrize("name", ["current.sqlite", "current-shm"])
def test_an_index_read_opened_before_this_process_connects_leaves_the_connection_its_locks(tmp_path, name):
    """Who holds the database when a read of its index starts says nothing about when the read's
    descriptor closes: a connection this process makes meanwhile holds the index's locks by then, and
    the close would drop them (astra's and Opus's probe). By name, the read never opens, a database
    itself named like an index (``current-shm``, its index ``current-shm-shm``) included."""
    database = tmp_path / name
    with _held_elsewhere(database, _UPDATER) as update:
        try:
            reader = store.open_for_read(database.with_name(database.name + "-shm"))  # before any connection here
        except store.WalIndexRefused:
            reader = None
        conn = store.connect(database, store.TIER_ENFORCED, journal_mode="WAL")
        try:
            conn.execute("PRAGMA cache_size = 2")  # the second count reads the pages again
            conn.execute("BEGIN")
            seen = conn.execute("SELECT count(*) FROM t WHERE substr(v, 1, 1) = 'x'").fetchone()[0]
            if reader is not None:
                reader.read(4096)
                reader.close()
            refused = _reset_refused(database)
            update()
            again = conn.execute("SELECT count(*) FROM t WHERE substr(v, 1, 1) = 'x'").fetchone()[0]
            conn.execute("ROLLBACK")
        finally:
            conn.close()
    assert refused, "closing the index read dropped the connection's locks"
    assert again == seen == 300 and reader is None


def test_a_download_built_before_this_process_connects_leaves_the_connection_its_locks(tmp_path, monkeypatch):
    """A download is decided when its response is built, but a file response opens and closes its file
    while it sends the body, after this process may have connected (a history page reading a child's
    store meanwhile): by name the index is refused at once, and the read transaction keeps its locks."""
    from ouroboros.gateway import files

    async def send(response):
        async def receive():
            return {"type": "http.disconnect"}

        async def sink(_message):
            return None

        await response({"type": "http", "method": "GET", "headers": []}, receive, sink)

    use_journal(monkeypatch, "WAL")
    database = _live_store(tmp_path)
    index = database.with_name(database.name + "-shm")
    with _held_elsewhere(database, _KEEPER):  # a child's own process keeps its store open
        store.close_connections()  # this process holds it no more (an evicted lease)
        response = files._file_response(index, filename=index.name)
        with store.open_store(tmp_path, write=False):
            assert _reset_refused(database)
            asyncio.run(send(response))
            assert _reset_refused(database), "sending the download dropped the read transaction's locks"
    assert response.status_code == 403


def test_another_process_s_index_is_left_out_of_a_workspace_capture_that_keeps_the_changes(tmp_path):
    """A delegated child's worktree may hold another process's live WAL database (a dev server, a
    killed test run): its index is left out as an excluded member before anything reads it, and the
    capture keeps the child's source change (failing the capture lost it)."""
    from ouroboros import workspace_patch_capture as capture

    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (["init"], ["-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                            "commit", "--allow-empty", "-m", "base"]):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    (repo / "fix.py").write_text("def fixed():\n    return 1\n", encoding="utf-8")
    with _held_elsewhere(repo / "app.db", _HOLDER):
        assert (repo / "app.db-shm").exists() and store.is_wal_index(repo / "app.db-shm")
        _, manifest = capture.write_workspace_patch_artifacts(repo, tmp_path / "capture", task={})
    assert manifest["status"] == "ready_with_changes", manifest["errors"]
    assert "fix.py" in manifest["untracked_included"]
    reasons = {row["path"]: row["reason"] for row in manifest["untracked_excluded"]}
    assert "WAL index" in reasons.get("app.db-shm", "")


def test_a_tracked_index_is_left_out_of_a_workspace_capture_that_keeps_the_changes(tmp_path):
    """``git add -A`` while a dev server ran commits its index too, and the server's next start writes a
    new one over it: the capture leaves it out of the patch and of the tracked paths an integration
    stages, says why, and keeps the change beside it (the index's bytes went into the patch, for an
    apply to write over the target's live index)."""
    from ouroboros import workspace_patch_capture as capture

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "fix.py").write_text("def fixed():\n    return 0\n", encoding="utf-8")
    with _held_elsewhere(repo / "app.db", _HOLDER):
        for args in (["init"], ["add", "-A"], ["-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                                               "commit", "-m", "base with the dev database"]):
            subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    (repo / "fix.py").write_text("def fixed():\n    return 1\n", encoding="utf-8")
    with _held_elsewhere(repo / "app.db", _HOLDER):  # its clean close removed the index; this start makes one
        _, manifest = capture.write_workspace_patch_artifacts(repo, tmp_path / "capture", task={})
    assert manifest["status"] == "ready_with_changes", manifest["errors"]
    assert "fix.py" in manifest["tracked_changed"] and "app.db-shm" not in manifest["tracked_changed"]
    assert "app.db-shm" not in manifest["diffstat"] and "fix.py" in manifest["diffstat"]
    reasons = {row["path"]: row["reason"] for row in manifest["tracked_excluded"]}
    assert "WAL index" in reasons.get("app.db-shm", "")


def test_a_directory_artifact_leaves_the_index_out_as_an_excluded_member(tmp_path, monkeypatch):
    from ouroboros.artifacts import copy_directory_to_task_artifacts

    use_journal(monkeypatch, "WAL")
    database = _live_store(tmp_path)
    with store.open_store(tmp_path, write=False):
        records = copy_directory_to_task_artifacts(None, database.parent, artifact_dir=tmp_path / "artifacts")
        assert _reset_refused(database)
    ledger = json.loads(open(records[0]["path"], encoding="utf-8").read())
    assert {"current.sqlite", "current.sqlite-wal"} <= {row["path"] for row in ledger["files"]}
    assert ledger["excluded"] == [{"path": "current.sqlite-shm", "reason": "SQLite WAL index"}]


@pytest.mark.parametrize("ending", [b"\n", b"\r\n"], ids=["LF", "CRLF"])
def test_a_base_linked_back_to_its_own_index_name_is_no_index_and_never_recurses(tmp_path, ending):
    """A regular ``b-shm`` and a link ``b`` naming it: probing ``b-shm`` led to ``b``, whose real name
    is ``b-shm`` again, without end (a RecursionError out of every reader: the panel answered 500, the
    capture failed, a snapshot was refused). A base whose real name ends in ``-shm`` is never probed."""
    from ouroboros import workspace_patch_capture as capture
    from ouroboros.gateway import files

    folder, repo = tmp_path / "odd", tmp_path / "repo"
    folder.mkdir()
    repo.mkdir()
    for args in (["init"], ["-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                            "commit", "--allow-empty", "-m", "base"]):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    (repo / "fix.py").write_text("def fixed():\n    return 1\n", encoding="utf-8")
    payload = b"notes" + ending
    for place in (folder, repo):
        (place / "b-shm").write_bytes(payload)
        try:
            (place / "b").symlink_to("b-shm")
        except OSError:
            pytest.skip("symbolic links need a privilege here")
    assert store.is_wal_index(folder / "b-shm") is False and store.is_wal_index(folder / "b") is False
    assert store.read_file_bytes(folder / "b-shm") == payload
    files._copy_path(folder, tmp_path / "odd-copy")
    assert sorted(path.name for path in (tmp_path / "odd-copy").iterdir()) == ["b", "b-shm"]
    assert (tmp_path / "odd-copy" / "b-shm").read_bytes() == payload
    _, manifest = capture.write_workspace_patch_artifacts(repo, tmp_path / "capture", task={})
    assert manifest["status"] == "ready_with_changes", manifest["errors"]


@pytest.mark.parametrize("ending", [b"\n", b"\r\n"], ids=["LF", "CRLF"])
def test_a_file_named_like_the_index_of_a_link_to_a_database_is_an_ordinary_file(tmp_path, ending):
    """SQLite names an index after its database's real path, so it sits beside the real file, never
    beside a link to it: a ``report-shm`` next to a link ``report -> actual.sqlite`` is the user's own
    file (actual.sqlite's index would be actual.sqlite-shm). It reads, copies and is listed for the
    drive settlement like any file (following the link made it an index and settlement dropped it)."""
    from ouroboros import task_custody
    from ouroboros.gateway import files

    folder = tmp_path / "artifacts"
    folder.mkdir()
    with contextlib.closing(sqlite3.connect(folder / "actual.sqlite")) as conn:
        conn.execute("CREATE TABLE t(v)")
    try:
        (folder / "report").symlink_to("actual.sqlite")
    except OSError:
        pytest.skip("symbolic links need a privilege here")
    payload = b"the user's report" + ending
    (folder / "report-shm").write_bytes(payload)
    assert store.is_wal_index(folder / "report-shm") is False and store.is_wal_index(folder / "report") is False
    assert store.read_file_bytes(folder / "report-shm") == payload
    files._copy_path(folder, tmp_path / "copy")
    assert (tmp_path / "copy" / "report-shm").read_bytes() == payload
    assert sorted(task_custody._listed(folder)) == ["actual.sqlite", "report-shm"]


def test_the_genesis_deliverable_manifest_lists_a_held_index_as_excluded(tmp_path):
    """A genesis project folder whose app holds a WAL database: the index is listed as excluded, as a
    directory artifact lists it, never as a read gap that makes the listing partial."""
    from ouroboros import headless

    project = tmp_path / "project"
    project.mkdir()
    (project / "app.py").write_text("print(1)\n", encoding="utf-8")
    with _held_elsewhere(project / "app.db", _HOLDER):
        manifest = headless._build_deliverable_manifest(project, "t1", "p1")
    assert (manifest["complete"], manifest["gap_count"]) == (True, 0), manifest["contents"]
    assert manifest["excluded"] == [{"rel": "app.db-shm", "reason": "SQLite WAL index"}]
    assert {"app.py", "app.db"} <= {row["rel"] for row in manifest["contents"]}


def test_a_child_s_artifact_listing_leaves_a_left_behind_index_out(tmp_path):
    """A WAL-mode SQLite artifact whose opener was killed leaves its index beside it: the child's
    listing the drive settlement copies from leaves the index out (shared memory, never read), so the
    settlement is not retained as unreadable until someone deletes it."""
    from ouroboros import task_custody

    source = tmp_path / "artifacts"
    source.mkdir()
    with contextlib.closing(sqlite3.connect(source / "x.db")) as conn:
        conn.execute("CREATE TABLE t(v)")
    (source / "x.db-shm").write_bytes(b"\0" * 32768)  # left behind by a killed opener
    (source / "report.txt").write_text("done\n", encoding="utf-8")
    assert sorted(task_custody._listed(source)) == ["report.txt", "x.db"]


def test_a_child_whose_wal_writer_was_killed_captures_copies_back_and_settles(tmp_path):
    """A child's WAL-mode SQLite writer killed mid-task leaves its log and index beside the database.
    The finalization capture lists the database and its log, never the index, so copy-back, the
    promotion retry and the drive settlement publish them and remove the drive (the held walk
    recorded the index, copy-back could not copy it, and every settlement kept the drive)."""
    from ouroboros import artifacts, headless, task_custody
    from ouroboros.observability import retry_pending_child_ref_promotions
    from ouroboros.task_results import load_task_result, write_task_result

    data, task = tmp_path / "data", "killed1"
    drive = headless.prepare_task_drive(data, task, "empty")
    folder = artifacts.task_artifact_dir_path(drive, task, create=True)
    (folder / "report.txt").write_text("done\n", encoding="utf-8")
    database = folder / "results.db"
    writer = subprocess.Popen([sys.executable, "-I", "-c", _UPDATER, str(database)],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert writer.stdout.readline().strip() == "ready"
    finally:
        writer.kill()  # killed while it holds the database: no close removes the log or the index
        writer.wait()
        writer.stdin.close()
        writer.stdout.close()
    assert store.is_wal_index(folder / "results.db-shm") and (folder / "results.db-wal").stat().st_size
    expected = ["report.txt", "results.db", "results.db-wal"]
    captured = artifacts.collect_task_artifact_records(drive, task)  # the finalization capture
    assert sorted(row["name"] for row in captured) == expected and all(row["sha256"] for row in captured)
    write_task_result(drive, task, "completed", result="done", artifacts=captured)
    write_task_result(data, task, "running", headless_child_drive_root=str(drive))
    headless.copy_child_task_result(data, {"id": task, "drive_root": str(drive)})
    retry_pending_child_ref_promotions(data)
    assert task_custody.settle_child_drive(data, task, drive, live=lambda _task: False)["status"] == "removed"
    canonical = artifacts.task_artifact_dir_path(data, task)
    rows = load_task_result(data, task)["artifacts"]
    assert sorted(row["name"] for row in rows) == expected and not drive.exists()
    assert {pathlib.Path(row["path"]).parent.resolve() for row in rows} == {canonical.resolve()}
    assert not (canonical / "results.db-shm").exists()
    with contextlib.closing(sqlite3.connect(canonical / "results.db")) as conn:  # its index rebuilt from the log
        assert conn.execute("SELECT count(*) FROM t").fetchone() == (300,)


@pytest.mark.parametrize("listing", ["capture", "strict_capture", "unmeasured", "view", "rescued"])
def test_a_task_artifact_listing_leaves_a_held_index_unopened_and_keeps_both_locks(tmp_path, monkeypatch, listing):
    """A store this process holds inside a task's artifact folder (its read transaction locks the
    index's read mark and the database's SHARED range): every listing over the held walk leaves the
    index out unread (POSIX opens no descriptor of it) and reads the database through the store's
    shared descriptor, so neither lock drops; the walk opened and closed a fresh descriptor of each,
    and a header probe by a fresh open of the database would still drop the second. The view also
    meets a row the unrepaired capture recorded for the index, and leaves it out unread too."""
    from ouroboros import artifacts, task_custody
    from ouroboros.task_finalization import rescued_files_fact

    use_journal(monkeypatch, "WAL")
    data, task = tmp_path / "data", "listing1"
    folder = artifacts.task_artifact_dir_path(data, task, create=True)
    database = _live_store(folder)
    (folder / "notes").write_text("a sibling that is not a database", encoding="utf-8")
    (folder / "notes-shm").write_text("the user's notes\n", encoding="utf-8")
    stale = {"kind": "task_artifact", "name": database.name + "-shm", "path": f"{database}-shm",
             "relpath": f"state/{database.name}-shm", "size": 32768, "sha256": "0" * 64}
    run = {"capture": lambda: artifacts.collect_task_artifact_records(data, task),
           "strict_capture": lambda: artifacts.collect_task_artifact_records(data, task, strict=True),
           "unmeasured": lambda: artifacts.collect_task_artifact_records(data, task, measure=False, strict=True),
           "view": lambda: task_custody.store_artifact_view(data, task, [stale]),
           "rescued": lambda: rescued_files_fact(task, [folder])}[listing]
    with store.open_store(folder, write=False):
        files = {path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file()}
        assert _reset_refused(database) and _shared_held(database)
        result = run()
        assert _reset_refused(database), "the listing opened and closed a descriptor of the index"
        assert _shared_held(database), "the listing closed a fresh descriptor of the database"
    index = database.relative_to(folder).as_posix() + "-shm"
    if listing == "rescued":
        assert (result["state"], result["count"]) == ("positive", len(files) - 1)
        return
    assert {row.get("relpath") or row["name"] for row in result} == files - {index}
    assert {"notes-shm", "state/current.sqlite", "state/current.sqlite-wal"} <= files
    assert listing in {"unmeasured", "view"} or all(row["sha256"] for row in result)


def _matrix(root):
    """Every name the index rule tells apart, in one task artifact folder; only links may be missing."""
    def database(path):
        with contextlib.closing(sqlite3.connect(path)) as conn:  # rollback journal: no sidecar stays
            conn.execute("CREATE TABLE t(v)")

    for name in ("x.db", "данные.db", "y.db", "z.db", "w", "actual.sqlite"):
        database(root / name)
    for name in ("x.db-shm", "данные.db-shm", "-shm", ".-shm", "..-shm", ".-SHM"):
        (root / name).write_bytes(b"\0" * 32768)  # left behind by killed openers; a lone -shm names none
    (root / "x.db-wal").write_bytes(b"a log reads as any file does")
    (root / "y.db-SHM").write_text("a spelling the volume may keep apart\n", encoding="utf-8")
    (root / "z.db-ſhm").write_text("or fold onto the index\n", encoding="utf-8")
    (root / "notes").write_text("a sibling that is not a database", encoding="utf-8")
    (root / "notes-shm").write_text("the user's notes\n", encoding="utf-8")
    (root / "w-shm").mkdir()
    (root / "w-shm" / "report.txt").write_text("a directory named like an index\n", encoding="utf-8")
    for folder, names in (("pair", ["current-shm"]), ("triple", ["current", "current-shm"])):
        (root / folder).mkdir()
        for name in names:
            database(root / folder / name)
        (root / folder / "current-shm-shm").write_bytes(b"\0" * 32768)
    (root / "report-shm").write_text("beside a link to a database\n", encoding="utf-8")
    (root / "b-shm").write_text("beside a link back to itself\n", encoding="utf-8")
    for link, target in (("report", "actual.sqlite"), ("b", "b-shm")):
        with contextlib.suppress(OSError):  # Windows without the symbolic-link privilege
            (root / link).symlink_to(target)


class _NtShaped:
    """``_Win32Files``' relative calls served by POSIX descriptors, so ``ConfinedDirectory``'s Windows
    branch runs anywhere; each file is recorded, by identity, under the access it got."""

    def __init__(self):
        self.data, self.attributes = set(), set()

    def open_root(self, path):
        return os.open(path, os.O_RDONLY | os.O_DIRECTORY)

    def open_relative(self, parent, name, *, directory, metadata=False):
        assert not metadata, "attributes are asked through identity()"
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | (os.O_DIRECTORY if directory else 0), dir_fd=parent)
        info = os.fstat(fd)
        self.data.add((info.st_dev, info.st_ino))
        return fd

    def identity(self, parent, name):
        info = os.stat(name, dir_fd=parent, follow_symlinks=False)  # an attribute-only handle reads nothing
        if not stat.S_ISREG(info.st_mode):
            raise OSError(errno.ELOOP if stat.S_ISLNK(info.st_mode) else errno.EISDIR, name)
        self.attributes.add((info.st_dev, info.st_ino))
        return info.st_dev, info.st_ino

    def to_fd(self, handle):
        return handle

    def close(self, handle):
        os.close(handle)


@contextlib.contextmanager
def _windows_branch():
    """``ConfinedDirectory``'s Windows branch: NT handles on Windows; elsewhere ``_NtShaped`` (yielded)."""
    from ouroboros import confined_files

    if confined_files.IS_WINDOWS:
        yield None
        return
    api = _NtShaped()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(confined_files, "IS_WINDOWS", True)
        patch.setattr(confined_files._Win32Files, "load", staticmethod(lambda: api))
        yield api


@pytest.mark.parametrize("measure", [True, False], ids=["measured", "unmeasured"])
@pytest.mark.parametrize("alias", [False, True], ids=["physical-root", "configured-alias"])
def test_the_held_walk_names_each_index_as_the_path_rule_does(tmp_path, measure, alias):
    """The walk asks the rule through each held directory, never by a descendant path, and answers as
    ``is_wal_index`` does for every name: a database's own index (left behind, Unicode-named, an index's
    database's, or a database itself named like an index) is left out; the database, its log, a
    sidecar-named file beside a text, a lone ``-shm``, a directory named like an index, files beside
    links and a spelling the volume keeps apart are listed; a spelling it folds is the index. The
    Windows branch gives the same answers and refuses each index having asked it by attributes only."""
    from ouroboros import artifacts, confined_files
    from tests.test_task_download_confinement import _junction_or_link

    data = tmp_path / "data"
    folder = artifacts.task_artifact_dir_path(data, "matrix1", create=True)
    _matrix(folder)
    files = {path.relative_to(folder).as_posix(): store.is_wal_index(path)
             for path in folder.rglob("*") if path.is_file() and not path.is_symlink()}
    excluded = {"x.db-shm", "данные.db-shm", "pair/current-shm-shm", "triple/current-shm"}
    assert {name for name, index in files.items() if index} - {"y.db-SHM", "z.db-ſhm"} == excluded
    assert {"y.db-SHM", "z.db-ſhm"} <= set(files) and "w-shm/report.txt" in files
    for spelling in ("y.db-SHM", "z.db-ſhm"):  # the volume decides: folded onto the index, or the user's
        assert files[spelling] == (folder / spelling).with_name(spelling[:-4] + "-shm").exists()
    if alias:
        configured = tmp_path / "configured"
        _junction_or_link(configured, data)
        data = configured
    rows = artifacts.collect_task_artifact_records(data, "matrix1", measure=measure, strict=True)
    assert {row.get("relpath") or row["name"] for row in rows} == {name for name, index in files.items() if not index}
    assert all(pathlib.Path(row["path"]).is_relative_to(artifacts.task_artifact_dir_path(data, "matrix1"))
               for row in rows)  # recorded through the root as configured
    with _windows_branch() as audit:
        for parent in {name.rpartition("/")[0] for name in files}:
            local = {name.rpartition("/")[2]: index for name, index in files.items()
                     if name.rpartition("/")[0] == parent}
            with confined_files.confined_directory(folder, parent.split("/") if parent else []) as held:
                assert {name: held._wal_index(name) for name in local} == local, parent
                for name in (name for name, index in local.items() if index):
                    with pytest.raises(store.WalIndexRefused):
                        held.open_for_read(name)
    indexes = {(info.st_dev, info.st_ino) for info in (os.stat(folder / name) for name, index in files.items()
                                                        if index)}
    assert audit is None or (indexes <= audit.attributes and not indexes & audit.data)  # never a data handle


def test_a_recorded_index_row_is_no_deliverable_in_the_task_view(tmp_path):
    """Rows recorded for an index before the walk applied the rule (a capture's measured row, an
    immutable one, a registration) name shared memory, not a deliverable: the view leaves them out
    unread, as it leaves a linked placement out, because serving or verifying their identity would
    read the index. The view rewrites nothing, and a recorded row whose file is missing still shows
    its identity; a user's ``notes-shm`` beside a text is a deliverable."""
    from ouroboros import artifacts, task_custody

    data, task = tmp_path / "data", "stale1"
    folder = artifacts.task_artifact_dir_path(data, task, create=True)
    with contextlib.closing(sqlite3.connect(folder / "x.db")) as conn:
        conn.execute("CREATE TABLE t(v)")
    (folder / "x.db-shm").write_bytes(b"\0" * 32768)  # left behind by a killed writer
    (folder / "notes").write_text("a sibling that is not a database", encoding="utf-8")
    (folder / "notes-shm").write_text("the user's notes\n", encoding="utf-8")

    def recorded(name, **fields):
        return {"kind": "task_artifact", "name": name, "path": str(folder / name), "size": 32768,
                "sha256": "0" * 64, **fields}

    registration = {"schema_version": 1, "artifacts": {"x.db-shm": recorded("x.db-shm", immutable=True)}}
    (folder / ".artifact_manifest.json").write_text(json.dumps(registration), encoding="utf-8")
    rows = [recorded("x.db-shm"), recorded("x.db-shm", immutable=True), recorded("notes-shm"), recorded("gone.txt")]
    view = {row["name"]: row for row in task_custody.store_artifact_view(data, task, rows)}
    assert sorted(view) == ["gone.txt", "notes", "notes-shm", "x.db"]
    assert view["gone.txt"]["sha256"] == "0" * 64


@pytest.mark.skipif(sys.platform != "win32", reason="NT share modes")
def test_windows_asks_the_index_by_attribute_only_handles(tmp_path):
    """NT checks no share mode for a handle without data access, and such a handle reads nothing.
    With the index held by a handle that shares nothing, a data open of it fails, yet the rule still
    names it and strict listings still leave it out: a data handle anywhere in the probe would have
    met that share mode, failing the probe and the listings."""
    import ctypes
    from ctypes import wintypes

    from ouroboros import artifacts, confined_files

    data, task = tmp_path / "data", "deny1"
    folder = artifacts.task_artifact_dir_path(data, task, create=True)
    with contextlib.closing(sqlite3.connect(folder / "x.db")) as conn:
        conn.execute("CREATE TABLE t(v)")
    (folder / "x.db-shm").write_bytes(b"\0" * 32768)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                     wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    holder = kernel32.CreateFileW(str(folder / "x.db-shm"), 0x80000000, 0, None, 3, 0, None)  # shares nothing
    assert holder not in (None, wintypes.HANDLE(-1).value), ctypes.get_last_error()
    try:
        with confined_files.confined_directory(folder) as held:
            with pytest.raises(OSError):
                held.open("x.db-shm")  # a data handle meets the holder's share mode
            assert held._wal_index("x.db-shm")
            with pytest.raises(store.WalIndexRefused):
                held.open_for_read("x.db-shm")
        for measure in (True, False):
            rows = artifacts.collect_task_artifact_records(data, task, measure=measure, strict=True)
            assert [row["name"] for row in rows] == ["x.db"], measure
    finally:
        kernel32.CloseHandle(holder)


def test_a_folder_copy_leaves_the_index_behind_and_the_copy_opens_whole(tmp_path, monkeypatch):
    from ouroboros.gateway import files

    use_journal(monkeypatch, "WAL")
    database = _live_store(tmp_path)
    copy = tmp_path / "state-copy"
    with store.open_store(tmp_path, write=False):
        files._copy_path(database.parent, copy)
        assert _reset_refused(database)
    copied = {path.name for path in copy.iterdir()}
    assert {"current.sqlite", "current.sqlite-wal"} <= copied and "current.sqlite-shm" not in copied
    with contextlib.closing(sqlite3.connect(copy / "current.sqlite")) as conn:  # its index rebuilt from the copied log
        assert conn.execute("SELECT status FROM tasks WHERE task_id='t1'").fetchone() == ("running",)


def test_search_skips_sqlite_journal_log_and_index_files_by_name(tmp_path):
    """The search fallback reads every file it walks with a plain open: SQLite's files stay out of it."""
    from ouroboros.code_search_rg import search_skip_reason

    for name in ("current.sqlite-shm", "current.sqlite-wal", "x.sqlite-journal", "x.db-shm", "x.sqlite3-wal"):
        (tmp_path / name).write_bytes(b"\0")
        assert search_skip_reason(tmp_path / name) == "excluded_name", name
    for name in ("notes-shm", "notes-wal", "notes.txt"):
        (tmp_path / name).write_text("text", encoding="utf-8")
        assert search_skip_reason(tmp_path / name) == "", name
