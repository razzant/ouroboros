"""Finished records are compressed in place: the one floor (``artifact_storage.compression_outcome``) and the
maintenance child's paced ``record_compression`` job over terminal evidence and finished reviewer seat homes.

Real APFS proofs run on macOS and skip elsewhere; the job's choices (which files, when, where it resumes)
are proven on every platform through a recording outcome.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import stat
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from ouroboros import artifact_storage, maintenance, maintenance_jobs, record_compression as job

darwin = pytest.mark.skipif(sys.platform != "darwin", reason="real APFS transparent compression")
TEXT = b"The finished record keeps every byte; only its allocation changes.\n"


def _write(path: pathlib.Path, raw: bytes = TEXT * 4096) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def _identity(path: pathlib.Path):
    raw = path.read_bytes()
    return len(raw), hashlib.sha256(raw).hexdigest()


@pytest.fixture
def data(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    return root


@pytest.fixture
def fake_darwin(monkeypatch):
    """The macOS path on any platform, with a ditto that must not run unless a test allows it."""
    calls = []

    def ditto(command, **_kwargs):
        calls.append(command)
        raise AssertionError("the floor should have kept this file before any rewrite")

    monkeypatch.setattr(artifact_storage, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(artifact_storage, "subprocess", SimpleNamespace(
        run=ditto, DEVNULL=subprocess.DEVNULL, SubprocessError=subprocess.SubprocessError))
    monkeypatch.setattr(artifact_storage, "_may_share", lambda path: False)
    return calls


# ---- the floor -------------------------------------------------------------------------------

@darwin
def test_real_apfs_keeps_size_bytes_and_sha_while_allocation_drops(tmp_path):
    path = _write(tmp_path / "record.log", TEXT * 16384)
    os.chmod(path, 0o640)
    before, identity = os.lstat(path), _identity(path)
    assert artifact_storage.compression_outcome(path) == "compressed"
    after = os.lstat(path)
    assert _identity(path) == identity and after.st_size == before.st_size
    assert after.st_flags & stat.UF_COMPRESSED and after.st_blocks < before.st_blocks // 2
    assert (after.st_mtime_ns, stat.S_IMODE(after.st_mode)) == (before.st_mtime_ns, 0o640)
    assert artifact_storage.compression_outcome(path) == "compressed_already", "a second pass rewrites nothing"
    assert sorted(item.name for item in tmp_path.iterdir()) == ["record.log"]


@darwin
def test_real_apfs_clone_is_kept_because_it_may_share_blocks(tmp_path):
    source = _write(tmp_path / "source.log")
    assert artifact_storage.clone_file(source, tmp_path / "clone.log")
    assert os.lstat(source).st_nlink == 1, "a clone is invisible to nlink"
    assert artifact_storage.compression_outcome(source) == "shared"
    assert artifact_storage.compression_outcome(_write(tmp_path / "private.log")) == "compressed"


def test_each_floor_guard_keeps_its_file_before_any_rewrite(tmp_path, fake_darwin, monkeypatch):
    small = _write(tmp_path / "small.log", b"x" * (artifact_storage.FLOOR_BYTES - 1))
    linked = _write(tmp_path / "linked.log")
    os.link(linked, tmp_path / "second-name.log")
    noise = _write(tmp_path / "noise.bin", os.urandom(4 * artifact_storage.FLOOR_BYTES))
    folder = tmp_path / "folder"
    folder.mkdir()
    assert artifact_storage.compression_outcome(small) == "small"
    assert artifact_storage.compression_outcome(linked) == "linked"
    assert artifact_storage.compression_outcome(noise) == "incompressible"
    assert artifact_storage.compression_outcome(folder) == "not_regular"
    plain = _write(tmp_path / "plain.log")
    monkeypatch.setattr(artifact_storage, "_may_share", lambda path: True)
    assert artifact_storage.compression_outcome(plain) == "shared"
    monkeypatch.setattr(artifact_storage, "_may_share", lambda path: None)
    assert artifact_storage.compression_outcome(plain) == "share_unknown", "unknown sharing is kept"
    monkeypatch.setattr(artifact_storage, "_may_share", lambda path: False)
    monkeypatch.setattr(artifact_storage.shutil, "disk_usage", lambda path: SimpleNamespace(free=1024))
    assert artifact_storage.compression_outcome(plain) == "low_space"
    assert fake_darwin == [], "no guard reached ditto"
    assert {item.name for item in tmp_path.iterdir()} == {
        "small.log", "linked.log", "second-name.log", "noise.bin", "folder", "plain.log"}


def test_ntfs_keeps_a_read_only_file_and_linux_is_unsupported(tmp_path, monkeypatch):
    info = SimpleNamespace(st_mode=stat.S_IFREG | 0o444, st_size=1 << 20, st_nlink=1, st_flags=0,
                           st_file_attributes=stat.FILE_ATTRIBUTE_READONLY)
    monkeypatch.setattr(artifact_storage, "sys", SimpleNamespace(platform="win32"))
    assert artifact_storage._floor(_write(tmp_path / "evidence.log"), info) == "read_only"
    monkeypatch.setattr(artifact_storage, "sys", SimpleNamespace(platform="linux"))
    assert artifact_storage.compression_supported() is False
    assert artifact_storage.compression_outcome(tmp_path / "evidence.log") == "unsupported"


def test_the_compression_sibling_carries_its_own_tag_inside_the_atomic_signature(tmp_path, monkeypatch):
    """One helper owns the tag: the sibling the macOS path writes is the job's own to reap, and still the
    atomic signature ``temp_sweep`` reaps where it walks."""
    from ouroboros import temp_sweep

    written = []

    def ditto(command, **_kwargs):
        written.append(pathlib.Path(command[-1]))
        raise subprocess.SubprocessError("killed mid-copy")

    monkeypatch.setattr(artifact_storage, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(artifact_storage, "subprocess", SimpleNamespace(
        run=ditto, DEVNULL=subprocess.DEVNULL, SubprocessError=subprocess.SubprocessError))
    monkeypatch.setattr(artifact_storage, "_may_share", lambda path: False)
    assert artifact_storage.compression_outcome(_write(tmp_path / "out.log")) == "failed"
    [sibling] = written
    assert sibling.parent == tmp_path and ".out.log.decmpfs.tmp." in sibling.name
    assert artifact_storage.is_compression_sibling(sibling.name) and temp_sweep.is_atomic_temp(sibling.name)
    assert not artifact_storage.is_compression_sibling(".out.log.tmp.4242.17.0a1b2c3d"), "another writer's"


def test_the_jobs_ditto_runs_at_the_utility_disk_tier_and_a_patch_writers_does_not(tmp_path, monkeypatch):
    """Only the job's rewrite is background work; a patch writer sharing the floor keeps its own priority."""
    commands = []

    def ditto(command, **_kwargs):
        commands.append(list(command))
        raise subprocess.SubprocessError("recorded only")

    monkeypatch.setattr(artifact_storage, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(artifact_storage, "subprocess", SimpleNamespace(
        run=ditto, DEVNULL=subprocess.DEVNULL, SubprocessError=subprocess.SubprocessError))
    monkeypatch.setattr(artifact_storage, "_may_share", lambda path: False)
    path = _write(tmp_path / "out.log")
    assert artifact_storage.compression_outcome(path, background=True) == "failed"
    assert artifact_storage.compression_outcome(path) == "failed"
    job_command, writer_command = commands
    assert job_command[:5] == ["/usr/sbin/taskpolicy", "-d", "utility", "/usr/bin/ditto", "--hfsCompression"]
    assert writer_command[:2] == ["/usr/bin/ditto", "--hfsCompression"]
    assert job_command[3:-1] == writer_command[:-1] == ["/usr/bin/ditto", "--hfsCompression", str(path)]


def test_a_dataless_file_is_kept_by_its_lstat_before_any_read(tmp_path, fake_darwin, monkeypatch):
    """An evicted (File Provider) file would be downloaded by the first read: the one lstat decides, so
    the job neither reads it nor spends its budget on it."""
    from ouroboros import artifacts

    path, real = _write(tmp_path / "evicted.log"), os.lstat

    class Evicted:
        # <sys/stat.h> SF_DATALESS on the Darwin stat this fakes; a Windows stat has no st_flags at all.
        st_flags = property(lambda self: getattr(real(path), "st_flags", 0) | 0x40000000)

        def __getattr__(self, name):
            return getattr(real(path), name)

    monkeypatch.setattr(os, "lstat", lambda target, *a, **k: Evicted() if pathlib.Path(target) == path else real(target, *a, **k))
    monkeypatch.setattr(artifacts, "stream_artifact_file", lambda *a, **k: pytest.fail("a dataless file was read"))
    assert artifact_storage.compression_outcome(path) == "dataless" and "dataless" in artifact_storage.STAT_SKIPS
    assert fake_darwin == [], "nor rewritten"


@darwin
def test_a_writer_during_the_ownership_question_keeps_its_bytes(data, tmp_path):
    """The ownership question reads the store and may wait, so it comes before the last look at the
    original: the test writer appending through its open stream meanwhile is seen there and wins (A-F2)."""
    from ouroboros.task_results import write_task_result
    from ouroboros.test_evidence import EVIDENCE_POINTER, FileOutput

    parent = data / "task_results" / "artifacts" / "done" / "source_handles" / "test-evidence"
    (tmp_path / EVIDENCE_POINTER).write_text(str(parent), encoding="utf-8")
    output, tail = FileOutput(tmp_path, "pytest"), b"written while the owner was asked\n"
    path = output.root / "stdout.bin"
    try:
        output.stdout.write(TEXT * 4096)
        original = path.read_bytes()
        write_task_result(data, "done", "completed")

        def owned():
            output.stdout.write(tail)
            return job._settled(data, ["done"])["done"]

        assert artifact_storage.compression_outcome(path, owned=owned) == "changed"
    finally:
        output.close()
    assert path.read_bytes() == original + tail and not os.lstat(path).st_flags & stat.UF_COMPRESSED
    assert sorted(item.name for item in path.parent.iterdir()) == ["stderr.bin", "stdout.bin"], "no sibling left"


@darwin
def test_an_append_during_compression_keeps_the_writers_bytes(tmp_path, monkeypatch):
    path = _write(tmp_path / "growing.log")
    real = subprocess.run

    def ditto_then_append(command, **kwargs):
        result = real(command, **kwargs)
        with open(path, "ab") as writer:
            writer.write(b"appended while the sibling was written\n")
        return result

    monkeypatch.setattr(artifact_storage, "subprocess", SimpleNamespace(
        run=ditto_then_append, DEVNULL=subprocess.DEVNULL, SubprocessError=subprocess.SubprocessError))
    inode = os.lstat(path).st_ino
    assert artifact_storage.compression_outcome(path) == "changed"
    assert path.read_bytes().endswith(b"appended while the sibling was written\n") and os.lstat(path).st_ino == inode
    assert sorted(item.name for item in tmp_path.iterdir()) == ["growing.log"], "the sibling is gone"


@darwin
def test_a_lost_owner_keeps_the_original(tmp_path):
    path = _write(tmp_path / "evidence.log")
    inode, asked = os.lstat(path).st_ino, []
    assert artifact_storage.compression_outcome(path, owned=lambda: asked.append(1) or False) == "not_owned"
    assert asked == [1] and os.lstat(path).st_ino == inode and not os.lstat(path).st_flags & stat.UF_COMPRESSED
    assert sorted(item.name for item in tmp_path.iterdir()) == ["evidence.log"]


# ---- the job --------------------------------------------------------------------------------

def _evidence(data, task_id, *, status="completed", live_files=True):
    from ouroboros.task_results import write_task_result

    write_task_result(data, task_id, status)
    base = data / "task_results" / "artifacts" / task_id
    files = {"evidence": _write(base / "delegated_runs" / "run-1" / "evidence" / "files" / "out.log"),
             "generation": _write(base / "delegated_runs" / "run-1" / "evidence-abc" / "summary.json"),
             "test": _write(base / "source_handles" / "test-evidence" / "run-1" / "stdout.bin")}
    if live_files:
        files.update(other=_write(base / "workspace.patch"),
                     lock=_write(base / "delegated_runs" / "run-1" / "disposition.lock"),
                     pending=_write(base / "delegated_runs" / "run-1" / ".evidence-x.pending" / "f.log"),
                     git=_write(base / "delegated_runs" / "run-1" / "evidence" / ".git" / "objects" / "pack"),
                     checkpoint=_write(base / "source_handles" / "context_checkpoints" / "c.json"))
    return files


def _seat(data, task_id, operation, *, area=False):
    from ouroboros import review_checking

    home = review_checking.seat_dir(data, task_id, operation)
    (home / "checking.json").parent.mkdir(parents=True, exist_ok=True)
    (home / "checking.json").write_text(json.dumps({"schema": 1, "task_id": task_id, "operation_id": operation}),
                                        encoding="utf-8")
    files = {"evidence": _write(home / "evidence" / "files" / "venv.bin"),
             "test": _write(home / ".test-evidence" / "run" / "out.log")}
    if area:
        files["area"] = _write(home / "area.noindex" / "big.log")
    return home, files


@pytest.fixture
def recorded(monkeypatch):
    """Every path offered to the floor, answered ``compressed`` without touching it."""
    seen = []

    def outcome(path, owned=None, background=False):
        assert background, "the job's own rewrites run at the utility disk tier"
        seen.append((pathlib.Path(path), owned))
        return "compressed"

    monkeypatch.setattr(artifact_storage, "compression_supported", lambda: True)
    monkeypatch.setattr(artifact_storage, "compression_outcome", outcome)
    monkeypatch.setattr(job, "_queue_live", lambda root: (lambda task_id: task_id == "seat-live"))
    return seen


def _run(data):
    return maintenance.run_job(data, job.JOB, should_stop=lambda: False)


def test_the_job_offers_only_finished_records_never_a_copy_or_git(data, recorded):
    done = _evidence(data, "done")
    live = _evidence(data, "live", status="running", live_files=False)
    offered_home, offered = _seat(data, "done", "op-1")
    _copy_home, with_copy = _seat(data, "done", "op-2", area=True)
    _live_home, of_live = _seat(data, "seat-live", "op-1")
    assert _run(data)["state"] == "done"
    paths = {path for path, _owned in recorded}
    assert paths == {done["evidence"], done["generation"], done["test"], offered["evidence"], offered["test"]}
    assert not paths & {*live.values(), *with_copy.values(), *of_live.values()}
    owned = {path: check for path, check in recorded}
    assert owned[done["evidence"]] is not None and owned[done["evidence"]]() is True, "the row is asked again"
    assert owned[offered["evidence"]] is None, "a home is held by its seat lock instead"
    row = maintenance_jobs.job(data, job.JOB)
    assert row["waiting"] == {"evidence": [], "seats": sorted(home.name for home in (_copy_home, _live_home))}
    assert _run(data) is None, "a second look at unchanged inputs is at rest"


def test_new_terminal_writes_make_the_job_due_and_only_their_records_are_read(data, recorded):
    from ouroboros.task_results import write_task_result

    _evidence(data, "first")
    late = _evidence(data, "late", status="running", live_files=False)
    _run(data)
    assert not job.due(data, maintenance_jobs.job(data, job.JOB), {}, False)
    recorded.clear()
    write_task_result(data, "late", "completed")
    assert job.due(data, maintenance_jobs.job(data, job.JOB), {}, False), "a terminal write arrived"
    _run(data)
    assert {path for path, _owned in recorded} == {late["evidence"], late["generation"], late["test"]}
    assert not job.due(data, maintenance_jobs.job(data, job.JOB), {}, False)


def test_the_budget_and_a_stop_leave_a_cursor_and_every_file_is_offered_once(data, recorded, monkeypatch):
    files = [*_evidence(data, "a", live_files=False).values(), *_evidence(data, "b", live_files=False).values()]
    monkeypatch.setattr(job, "FILES_PER_RUN", 2)
    outcomes = []
    while (outcome := _run(data)) is not None:
        outcomes.append(outcome["state"])
        row = maintenance_jobs.job(data, job.JOB)
        assert (row["cursor"] is None) == (outcome["state"] == "done")
    assert outcomes == ["pending", "pending", "done"]
    assert sorted(path for path, _owned in recorded) == sorted(files), "each file once, none skipped"
    recorded.clear()
    from ouroboros.task_results import write_task_result

    write_task_result(data, "c", "completed")  # a new pass; a stop after the first file keeps its place
    extra = _evidence(data, "c", live_files=False)
    calls = iter([False, True])
    assert maintenance.run_job(data, job.JOB, should_stop=lambda: next(calls, True))["state"] == "pending"
    assert maintenance_jobs.job(data, job.JOB)["owed"]["evidence"] == ["c"]
    monkeypatch.setattr(job, "FILES_PER_RUN", 100)
    assert _run(data)["state"] == "done"
    assert sorted(path for path, _owned in recorded) == sorted(extra.values())


def test_a_killed_runs_own_sibling_is_reaped_and_another_temporary_is_left(data, recorded):
    files = _evidence(data, "done", live_files=False)
    home, _ = _seat(data, "done", "op-1")
    own = [_write(artifact_storage.compression_sibling(files["evidence"])),
           _write(artifact_storage.compression_sibling(home / "evidence" / "files" / "venv.bin"))]
    foreign = _write(files["evidence"].with_name(".result.json.tmp.4242.17.0a1b2c3d"))  # another writer's
    keep = _write(files["evidence"].with_name(".notes.tmp.backup"))  # not the signature: an ordinary record
    outcome = _run(data)
    assert outcome["state"] == "done" and not any(path.exists() for path in own)
    offered = {path for path, _owned in recorded}
    assert not offered & {*own, foreign} and foreign.exists(), "another writer's temporary: neither a record nor ours"
    assert keep in offered and maintenance_jobs.job(data, job.JOB)["last"]["reaped"] == 2


def test_published_evidence_named_like_a_temporary_is_neither_offered_nor_reaped(data, recorded):
    """The real seat capture names and hashes a retained file whose name looks like an atomic temporary
    (A-F1): it is a record, kept as it is; only the job's own siblings are ever reaped."""
    from ouroboros import review_checking
    from ouroboros.task_results import write_task_result

    home = review_checking.seat_dir(data, "done", "op")
    home.mkdir(parents=True)
    (home / "checking.json").write_text(json.dumps({"schema": 1, "task_id": "done", "operation_id": "op",
                                                    "source": {"kind": "workspace"}}), encoding="utf-8")
    source = _write(home / "area.noindex" / ".review-evidence" / ".recovered.log.tmp.4242.17.0a1b2c3d")
    original = source.read_bytes()
    capture = review_checking.capture_seat_evidence(data, "done", "op")
    assert capture["status"] == "retained", capture
    [named] = capture["named_evidence"]
    retained = home / "evidence" / "files" / named["path"]
    assert retained.read_bytes() == original and not (home / "area.noindex").exists()
    write_task_result(data, "done", "completed")
    assert _run(data)["state"] == "done"
    assert retained.read_bytes() == original and retained not in {path for path, _owned in recorded}
    assert maintenance_jobs.job(data, job.JOB)["last"]["reaped"] == 0


def test_a_held_seat_lock_keeps_the_home_waiting_until_it_is_free(data, recorded):
    from ouroboros.review_checking import _seat_locked
    from ouroboros.task_results import write_task_result

    home, files = _seat(data, "done", "op-1")
    write_task_result(data, "done", "completed")
    with _seat_locked(home) as held:
        assert held
        _run(data)
    assert not recorded and maintenance_jobs.job(data, job.JOB)["waiting"]["seats"] == [home.name]
    write_task_result(data, "other", "completed")  # the next pass: the waiting home is looked at again
    _run(data)
    assert {path for path, _owned in recorded} == {files["evidence"], files["test"]}
    assert maintenance_jobs.job(data, job.JOB)["waiting"]["seats"] == []


def test_an_unreadable_listing_fails_the_run_and_marks_nothing_covered(data, recorded, monkeypatch):
    _evidence(data, "done", live_files=False)
    real, refused = os.scandir, data / "task_results" / "artifacts"

    def scandir(path="."):
        if pathlib.Path(path) == refused:
            raise PermissionError(13, "listing refused")
        return real(path)

    monkeypatch.setattr(maintenance, "JOBS", {job.JOB: maintenance.JOBS[job.JOB]})
    monkeypatch.setattr(os, "scandir", scandir)
    assert maintenance.run(data)["status"] == "failed"
    row = maintenance_jobs.job(data, job.JOB)
    assert row.get("counter") is None and row.get("state") != "done", "the backlog is still owed"
    assert row["last_error"]["class"] == "PermissionError" and not recorded
    monkeypatch.setattr(os, "scandir", real)
    assert _run(data)["state"] == "done" and len(recorded) == 3


def test_a_folder_that_cannot_be_listed_is_counted_and_the_walk_goes_on(data, recorded, monkeypatch):
    """A-F3: never taken for an empty folder, never a block on the rest; its task's next terminal write
    brings it back."""
    from ouroboros.task_results import write_task_result

    files = _evidence(data, "done", live_files=False)
    real, refused = os.scandir, files["test"].parent

    def scandir(path="."):
        if pathlib.Path(path) == refused:
            raise PermissionError(13, "listing refused")
        return real(path)

    monkeypatch.setattr(os, "scandir", scandir)
    assert _run(data)["state"] == "done"
    assert maintenance_jobs.job(data, job.JOB)["last"]["skipped"] == {"unreadable": 1}
    assert {path for path, _owned in recorded} == {files["evidence"], files["generation"]}, "the rest is walked"
    monkeypatch.setattr(os, "scandir", real)
    recorded.clear()
    write_task_result(data, "done", "completed")
    assert _run(data)["state"] == "done" and files["test"] in {path for path, _owned in recorded}


@pytest.mark.skipif(sys.platform == "win32" or getattr(os, "geteuid", lambda: 1)() == 0,
                    reason="POSIX search permission; root reads everything")
def test_a_task_folder_nobody_can_search_never_parks_the_backlog(data, recorded):
    """The presence probe that cannot look keeps the task a candidate: the walk counts it unreadable and
    the tasks after it are compressed in the same pass (R2-1)."""
    _evidence(data, "a-unreadable", live_files=False)
    later = _evidence(data, "b-readable", live_files=False)
    locked = data / "task_results" / "artifacts" / "a-unreadable"
    locked.chmod(0)
    try:
        outcome = _run(data)
        row = maintenance_jobs.job(data, job.JOB)
    finally:
        locked.chmod(0o755)
    assert outcome["state"] == "done" and "last_error" not in row
    assert {path for path, _owned in recorded} == set(later.values())
    assert row["last"]["skipped"] == {"unreadable": 1}


def test_a_task_written_terminal_again_is_walked_from_its_start(data, recorded, monkeypatch):
    """A-F4: a later pass stopped inside a task keeps its cursor only while the task is not written again."""
    from ouroboros.task_results import write_task_result

    write_task_result(data, "seed", "completed")
    assert _run(data)["state"] == "done"  # the backlog is behind: later passes read the terminal writes
    base = data / "task_results" / "artifacts" / "done" / "source_handles" / "test-evidence" / "run"
    a, z = _write(base / "a.log"), _write(base / "z.log")
    write_task_result(data, "done", "completed")
    monkeypatch.setattr(job, "FILES_PER_RUN", 1)
    assert _run(data)["state"] == "pending" and [path for path, _owned in recorded] == [a]
    added = _write(base / "0-new.log")  # late evidence sorting before the cursor
    write_task_result(data, "done", "completed", result="late evidence published")
    while _run(data) is not None:
        pass
    assert {added, z} <= {path for path, _owned in recorded}


def test_the_cursors_task_written_again_never_drops_the_tasks_still_owed(data, recorded, monkeypatch):
    """A later pass stopped at the first file of B, its cursor still in A: A written terminal again is
    walked again in its place among the writes, and B, owed, is still compressed."""
    from ouroboros.task_results import write_task_result

    write_task_result(data, "seed", "completed")
    assert _run(data)["state"] == "done"
    _evidence(data, "a", live_files=False)
    owed = _evidence(data, "b", live_files=False)
    monkeypatch.setattr(job, "FILES_PER_RUN", 3)
    assert _run(data)["state"] == "pending"
    row = maintenance_jobs.job(data, job.JOB)
    assert row["cursor"]["key"] == "a" and row["owed"]["evidence"] == ["b"]
    write_task_result(data, "a", "completed", result="written again")
    recorded.clear()
    monkeypatch.setattr(job, "FILES_PER_RUN", 100)
    while _run(data) is not None:
        pass
    assert set(owed.values()) <= {path for path, _owned in recorded}


def test_a_seat_home_changed_since_the_pass_stopped_in_it_is_walked_from_its_start(data, recorded, monkeypatch):
    """The seat-plane twin of A-F4: a later pass stopped inside a home keeps its cursor only while the home
    is unchanged since the pass that stopped there."""
    from ouroboros.task_results import write_task_result

    from ouroboros import review_checking

    write_task_result(data, "seed", "completed")
    assert _run(data)["state"] == "done"  # the backlog is behind: later passes read the homes changed since
    home = review_checking.seat_dir(data, "done", "op-1")
    home.mkdir(parents=True)
    (home / "checking.json").write_text(json.dumps({"schema": 1, "task_id": "done", "operation_id": "op-1"}),
                                        encoding="utf-8")
    (home / ".lock").touch()  # a real home's seat lock file exists from its first preparation
    first, _last = _write(home / "evidence" / "files" / "a.bin"), _write(home / "evidence" / "files" / "z.bin")
    write_task_result(data, "done", "completed")
    monkeypatch.setattr(job, "FILES_PER_RUN", 1)
    assert _run(data)["state"] == "pending" and [path for path, _owned in recorded] == [first]
    assert maintenance_jobs.job(data, job.JOB)["cursor"]["key"] == home.name
    added = _write(home / ".test-evidence" / "late" / "out.log")  # a new folder, sorting before the cursor
    monkeypatch.setattr(job, "FILES_PER_RUN", 100)
    while _run(data) is not None:
        pass
    assert added in {path for path, _owned in recorded}


def test_a_task_closing_mid_backlog_below_the_cursor_is_compressed_by_the_next_pass(data, recorded, monkeypatch):
    """O-F4: the backlog records the terminal counter it began at, so what closed meanwhile is read next."""
    _evidence(data, "m", live_files=False)
    _evidence(data, "z", live_files=False)
    monkeypatch.setattr(job, "FILES_PER_RUN", 2)
    assert _run(data)["state"] == "pending" and maintenance_jobs.job(data, job.JOB)["cursor"]["key"] == "m"
    late = _evidence(data, "b", live_files=False)  # turns terminal now; its name sorts before the cursor
    while _run(data) is not None:
        pass
    assert set(late.values()) <= {path for path, _owned in recorded}


def test_a_backlog_whose_cursor_folder_is_gone_resumes_at_the_next_name(data, recorded, monkeypatch):
    import shutil

    _evidence(data, "a", live_files=False)
    _evidence(data, "b", live_files=False)
    later = _evidence(data, "c", live_files=False)
    monkeypatch.setattr(job, "FILES_PER_RUN", 4)
    assert _run(data)["state"] == "pending" and maintenance_jobs.job(data, job.JOB)["cursor"]["key"] == "b"
    shutil.rmtree(data / "task_results" / "artifacts" / "b")
    recorded.clear()
    monkeypatch.setattr(job, "FILES_PER_RUN", 100)
    assert _run(data)["state"] == "done"
    assert {path for path, _owned in recorded} == set(later.values()), "a is not walked again"


def test_evidence_waits_until_its_task_is_proven_over(data, recorded, monkeypatch):
    """O-F1a: a terminal task whose lineage still runs keeps its files untouched and waits; the next pass
    after the process is gone compresses them."""
    from ouroboros import body_candidate
    from ouroboros.task_results import write_task_result

    files = _evidence(data, "done", live_files=False)
    lineage = [4242]  # a test process of the task, still writing its output
    monkeypatch.setattr(body_candidate, "live_lineage_processes", lambda root, task_id: list(lineage))
    assert _run(data)["state"] == "done"
    assert not recorded and maintenance_jobs.job(data, job.JOB)["waiting"]["evidence"] == ["done"]
    lineage.clear()
    write_task_result(data, "other", "completed")  # the next pass asks again
    _run(data)
    assert {path for path, _owned in recorded} == set(files.values())
    assert maintenance_jobs.job(data, job.JOB)["waiting"]["evidence"] == []


def test_totals_are_kept_per_plane_and_shown_under_their_own_place(data, recorded):
    from ouroboros import storage_removers

    _evidence(data, "done", live_files=False)
    _seat(data, "done", "op-1")
    _run(data)
    totals = maintenance_jobs.job(data, job.JOB)["totals"]
    assert {plane: total["compressed"] for plane, total in totals.items()} == {"evidence": 3, "seats": 2}
    found = storage_removers.place_removers(data, engine=False)
    shown = {place: next(fact for fact in found[place] if fact["id"] == job.JOB)["compressed"]
             for place in ("task_results", "app_state")}
    assert shown == {"task_results": 3, "app_state": 2}, "never one job-wide total under two places"


def test_a_supported_run_clears_another_platforms_unsupported(data, recorded, monkeypatch):
    from ouroboros import storage_removers

    _evidence(data, "done", live_files=False)
    monkeypatch.setattr(artifact_storage, "compression_supported", lambda: False)
    monkeypatch.setattr(job, "sys", SimpleNamespace(platform="linux"))  # the data folder came from Linux
    assert _run(data) == {"state": "done", "items": 0}
    assert maintenance_jobs.job(data, job.JOB)["unsupported"] == "linux"
    monkeypatch.setattr(artifact_storage, "compression_supported", lambda: True)
    monkeypatch.setattr(job, "sys", sys)
    assert _run(data)["items"] == 3 and "unsupported" not in maintenance_jobs.job(data, job.JOB)
    [fact] = [fact for fact in storage_removers.place_removers(data, engine=False)["task_results"] if fact["id"] == job.JOB]
    assert fact["state"] == "done" and fact["compressed"] == 3


def test_another_platforms_unsupported_clears_without_waiting_for_a_task(data, recorded, monkeypatch):
    """A folder that ran on macOS, then once on Linux, is due on macOS again at once: its Storage line is not
    "not supported" until some task finishes."""
    _evidence(data, "done", live_files=False)
    assert _run(data)["state"] == "done"
    monkeypatch.setattr(artifact_storage, "compression_supported", lambda: False)
    monkeypatch.setattr(job, "sys", SimpleNamespace(platform="linux"))
    assert _run(data) == {"state": "done", "items": 0}
    monkeypatch.setattr(artifact_storage, "compression_supported", lambda: True)
    monkeypatch.setattr(job, "sys", sys)
    assert job.due(data, maintenance_jobs.job(data, job.JOB), {}, False), "no terminal write is needed"
    assert _run(data)["state"] == "done" and "unsupported" not in maintenance_jobs.job(data, job.JOB)
    assert _run(data) is None, "and then at rest"


def test_the_maintenance_child_keeps_only_its_cpu_nice_and_the_count_child_its_process_tier(monkeypatch):
    """The maintenance child's other jobs (a capture the server waits for, packing, imports) keep the
    standard disk tier and download an evicted input; only the compression job lowers its own thread's."""
    from ouroboros import storage

    niced, taken = [], []
    monkeypatch.setattr(storage, "utility_io", lambda **kw: taken.append(kw) or (lambda: None))
    # Only the child's view of a POSIX host: a global os.name would make pathlib build PosixPaths on Windows.
    posix = SimpleNamespace(name="posix", nice=lambda increment: niced.append(increment) or 0)
    monkeypatch.setattr(maintenance, "os", posix)
    maintenance._lower_priority()
    assert niced == [10] and taken == [], "no disk policy for the whole child"
    monkeypatch.setattr(storage, "sys", SimpleNamespace(platform="darwin"))
    assert storage._lower_io_priority() == "utility" and taken == [{}], "the count child: its process, one helper"


@darwin
def test_a_compression_run_lowers_its_own_threads_disk_tier_and_puts_it_back(data, recorded, monkeypatch):
    """Read back from the kernel: utility tier and dataless files off on the job's thread during its run,
    the thread's own policies after it, the process's untouched throughout."""
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)

    def policies():  # (thread disk, thread dataless, process disk, process dataless)
        return tuple(libc.getiopolicy_np(kind, scope) for scope in (1, 0) for kind in (0, 3))

    before, during = policies(), []
    floor = artifact_storage.compression_outcome
    monkeypatch.setattr(artifact_storage, "compression_outcome",
                        lambda path, **kw: during.append(policies()) or floor(path, **kw))
    _evidence(data, "done", live_files=False)
    assert _run(data)["items"] == 3
    assert set(during) == {(4, 1, *before[2:])}, "utility tier and no download, this thread alone"
    assert policies() == before, "put back after the run"


def test_a_run_that_throws_still_puts_its_threads_disk_tier_back(data, recorded, monkeypatch):
    from ouroboros import storage

    events = []
    monkeypatch.setattr(storage, "utility_io", lambda **kw: events.append(("lowered", kw)) or (lambda: events.append("restored")))

    def broken(path, **_kw):
        events.append("floor")
        raise RuntimeError("the floor failed")

    monkeypatch.setattr(artifact_storage, "compression_outcome", broken)
    _evidence(data, "done", live_files=False)
    with pytest.raises(RuntimeError):
        _run(data)
    assert events == [("lowered", {"thread": True}), "floor", "restored"]


def test_unsupported_platforms_record_it_once_and_walk_nothing(data, monkeypatch):
    _evidence(data, "done", live_files=False)
    monkeypatch.setattr(artifact_storage, "compression_supported", lambda: False)
    monkeypatch.setattr(artifact_storage, "compression_outcome", lambda *a, **k: pytest.fail("walked"))
    assert job.JOB in maintenance.due(data)
    assert _run(data) == {"state": "done", "items": 0}
    assert maintenance_jobs.job(data, job.JOB)["unsupported"] == sys.platform
    assert job.JOB not in maintenance.due(data) and _run(data) is None


@darwin
def test_real_compression_of_terminal_evidence_and_a_finished_seat_home(data, monkeypatch):
    monkeypatch.setattr(job, "_queue_live", lambda root: (lambda task_id: False))
    done = _evidence(data, "done")
    home, seat = _seat(data, "done", "op-1")
    linked = _write(seat["evidence"].with_name("linked.bin"))
    os.link(linked, home / "outside-name.bin")
    identities = {path: _identity(path) for path in (*done.values(), *seat.values(), linked)}
    outcome = _run(data)
    assert outcome["state"] == "done" and outcome["items"] == 5
    for path, identity in identities.items():
        assert _identity(path) == identity, path
    compressed = {path for path in identities if os.lstat(path).st_flags & stat.UF_COMPRESSED}
    assert compressed == {done["evidence"], done["generation"], done["test"], seat["evidence"], seat["test"]}
    row = maintenance_jobs.job(data, job.JOB)
    assert {plane: total["compressed"] for plane, total in row["totals"].items()} == {"evidence": 3, "seats": 2}
    assert all(total["saved_bytes"] > 0 for total in row["totals"].values()) and row["last"]["skipped"] == {}
    assert _run(data) is None


def test_a_finished_calls_wire_journal_is_compressed_and_only_that_file(data, recorded):
    """The owner's 28.10 answer 5B: the call wire journal stays whole under the shared finished-record machinery."""
    from ouroboros.task_results import write_task_result

    for task_id, status in (("call-done", "completed"), ("call-live", "running")):
        write_task_result(data, task_id, status)
    done = data / "task_results" / "artifacts" / "call-done"
    journal = _write(done / job.WIRE_JOURNAL)
    deeper = _write(done / "source_handles" / job.WIRE_JOURNAL)  # the name elsewhere in the folder is not the journal
    (done / "nested" / job.WIRE_JOURNAL).mkdir(parents=True)  # nor is a folder of that name
    live = _write(data / "task_results" / "artifacts" / "call-live" / job.WIRE_JOURNAL)
    assert _run(data)["state"] == "done"
    paths = {path for path, _owned in recorded}
    assert journal in paths and not paths & {deeper, live}


# ---- retained task sources: tool results and delegated activity --------------------------------------

def _sources(data, task_id, *, status="completed"):
    from ouroboros.task_results import write_task_result

    write_task_result(data, task_id, status)
    base = data / "task_results" / "artifacts" / task_id / "source_handles"
    return {"result": _write(base / "tool_results" / f"call-{task_id}-{'a' * 64}.txt"),
            "activity": _write(base / "delegated_activity" / f"run-1-5-{'b' * 64}.jsonl")}


def _rest_row(data, **facts):
    """The row a body from before this coverage left: no ``coverage`` fact, its position earned by the old scope."""
    row = {"state": "done", "cursor": None, "owed": {"evidence": [], "seats": []}, "began": None,
           "waiting": {"evidence": [], "seats": []}, "counter": job._counter(data), "since": time.time() + 3600}
    maintenance_jobs.record(data, job.JOB, **{**row, **facts})


def test_finished_task_sources_are_offered_and_their_neighbours_are_not(data, recorded):
    """A task holding nothing but retained sources is a candidate; working material beside them stays out."""
    only = _sources(data, "only-sources")
    live = _sources(data, "live", status="running")
    base = data / "task_results" / "artifacts" / "only-sources" / "source_handles"
    kept = [_write(base / "context_checkpoints" / "c.json"), _write(base / "working" / "working_checkpoint-a1.json"),
            _write(base / "review_inputs" / "request-1" / "input.txt")]
    assert _run(data)["state"] == "done"
    offered = {path for path, _owned in recorded}
    assert offered == set(only.values()) and not offered & {*live.values(), *kept}
    assert maintenance_jobs.job(data, job.JOB)["coverage"] == job.COVERAGE


def test_a_pass_finished_under_the_old_scope_walks_the_old_terminal_backlog_once(data, recorded):
    """No terminal write will ever name these tasks again: the coverage fact alone makes the job due."""
    old = _sources(data, "old-terminal")
    _rest_row(data)
    row = maintenance_jobs.job(data, job.JOB)
    assert job.due(data, row, {}, False), "the old scope's counter is no proof for the wider one"
    assert _run(data)["state"] == "done"
    assert {path for path, _owned in recorded} == set(old.values())
    assert maintenance_jobs.job(data, job.JOB)["counter"] == job._counter(data)
    assert _run(data) is None, "and then at rest"


def test_an_old_backlog_stopped_partway_starts_again_below_its_cursor(data, recorded, monkeypatch):
    """The old pass already walked ``a`` under the old scope: its cursor at ``m`` would pass it by for good."""
    below, at, above = _sources(data, "a"), _sources(data, "m"), _sources(data, "z")
    counter = job._counter(data)
    _rest_row(data, state="pending", counter=None, since=None, began={"counter": counter, "at": time.time()},
              cursor={"plane": "evidence", "key": "m", "path": ["source_handles", "tool_results", "z"]})
    monkeypatch.setattr(job, "FILES_PER_RUN", 2)
    states = []
    while (outcome := _run(data)) is not None:
        states.append(outcome["state"])
    assert states == ["pending", "pending", "done"], "the budget still paces the widened pass"
    offered = [path for path, _owned in recorded]
    assert sorted(offered) == sorted([*below.values(), *at.values(), *above.values()]), "each file once"


def test_widened_backlog_uses_its_own_start_facts_without_an_extra_pass(data, recorded):
    _sources(data, "old-terminal")
    stale_counter = job._counter(data)
    _rest_row(data, state="pending", counter=None, since=None,
              began={"counter": stale_counter, "at": 1.0},
              cursor={"plane": "evidence", "key": "m", "path": []})
    _sources(data, "new-terminal")
    started = time.time()
    assert _run(data)["state"] == "done"
    row = maintenance_jobs.job(data, job.JOB)
    assert row["counter"] == job._counter(data)
    assert row["since"] >= started
    assert _run(data) is None, "the completed wider pass must rest without another stat-only scan"


def test_a_stop_inside_the_new_coverage_resumes_at_its_cursor_and_keeps_the_old_counter_from_counting(
        data, recorded, monkeypatch):
    """A stopped first run leaves no old counter that a later reader could take for a covered pass."""
    files = [*_sources(data, "a").values(), *_sources(data, "b").values()]
    _rest_row(data)
    calls = iter([False, True])
    assert maintenance.run_job(data, job.JOB, should_stop=lambda: next(calls, True))["state"] == "pending"
    row = maintenance_jobs.job(data, job.JOB)
    assert row["counter"] is None and row["coverage"] == job.COVERAGE and row["cursor"]["key"] == "a"
    assert job.due(data, row, {}, False)
    assert _run(data)["state"] == "done"
    assert sorted(path for path, _owned in recorded) == sorted(files), "each file once, none lost at the stop"


def test_a_seat_home_owed_by_the_old_pass_survives_the_restart(data, recorded, monkeypatch):
    """An old later pass stopped among the seat homes: ``since`` already moved, so only ``owed`` names this home."""
    from ouroboros.task_results import write_task_result

    home, files = _seat(data, "done", "op-1")
    write_task_result(data, "done", "completed")
    _sources(data, "a")
    _sources(data, "b")
    _rest_row(data, state="pending", cursor={"plane": "seats", "key": home.name, "path": []},
              owed={"evidence": [], "seats": [home.name]})
    monkeypatch.setattr(job, "FILES_PER_RUN", 1)
    assert _run(data)["state"] == "pending"
    assert maintenance_jobs.job(data, job.JOB)["owed"]["seats"] == [home.name], "a stop in the evidence plane keeps it"
    monkeypatch.setattr(job, "FILES_PER_RUN", 100)
    while _run(data) is not None:
        pass
    assert set(files.values()) <= {path for path, _owned in recorded}


def test_a_failed_listing_leaves_the_old_position_and_the_coverage_still_owed(data, recorded, monkeypatch):
    old = _sources(data, "old-terminal")
    _rest_row(data)
    counter = maintenance_jobs.job(data, job.JOB)["counter"]
    real, refused = os.scandir, data / "task_results" / "artifacts"

    def scandir(path="."):
        if pathlib.Path(path) == refused:
            raise PermissionError(13, "listing refused")
        return real(path)

    monkeypatch.setattr(maintenance, "JOBS", {job.JOB: maintenance.JOBS[job.JOB]})
    monkeypatch.setattr(os, "scandir", scandir)
    assert maintenance.run(data)["status"] == "failed"
    row = maintenance_jobs.job(data, job.JOB)
    assert "coverage" not in row and row["counter"] == counter and row["last_error"]["class"] == "PermissionError"
    assert job.due(data, row, {}, False) and not recorded, "never marked covered by a read that did not happen"
    monkeypatch.setattr(os, "scandir", real)
    assert _run(data)["state"] == "done" and {path for path, _owned in recorded} == set(old.values())


def test_the_old_backlog_is_recorded_as_owed_on_no_platform_that_cannot_compress(data, monkeypatch):
    _sources(data, "old-terminal")
    _rest_row(data, unsupported=sys.platform)
    monkeypatch.setattr(artifact_storage, "compression_supported", lambda: False)
    monkeypatch.setattr(artifact_storage, "compression_outcome", lambda *a, **k: pytest.fail("walked"))
    assert not job.due(data, maintenance_jobs.job(data, job.JOB), {}, False)
    assert _run(data) is None and "coverage" not in maintenance_jobs.job(data, job.JOB)


@darwin
def test_real_compression_keeps_what_every_reader_of_a_retained_source_reads(data, monkeypatch):
    """The path, ref, size and sha256 a reader was handed before compression still answer after it, through
    the verifying reader and the address scan delegated activity uses; a clone and a small file stay plain."""
    from ouroboros import artifacts, delegate_activity

    monkeypatch.setattr(job, "_queue_live", lambda root: (lambda task_id: False))
    from ouroboros.task_results import write_task_result

    write_task_result(data, "old-terminal", "completed")
    raw, small = TEXT * 4096, b"short\n"
    refs = {category: artifacts.store_actor_source_bytes(
        data, "old-terminal", category=category, source_id="run-1-5", data=raw, extension="jsonl")
        for category in ("tool_results", "delegated_activity")}
    tiny = artifacts.store_actor_source_bytes(
        data, "old-terminal", category="tool_results", source_id="tiny", data=small, extension="txt")
    base = data / "task_results" / "artifacts" / "old-terminal"
    source = artifacts.store_actor_source_bytes(
        data, "old-terminal", category="tool_results", source_id="cloned", data=raw + b"!", extension="txt")
    assert artifact_storage.clone_file(base / source["path"], data / "peer.txt")
    ctx = SimpleNamespace(drive_root=data)
    addressed = delegate_activity._retained_sources(ctx, "old-terminal", "run")
    _rest_row(data)
    assert _run(data)["state"] == "done"
    for ref in refs.values():
        info = os.lstat(base / ref["path"])
        assert info.st_flags & stat.UF_COMPRESSED and info.st_size == ref["size"] == len(raw)
        assert artifacts.read_actor_source_bytes(data, "old-terminal", ref) == raw
    assert delegate_activity._retained_sources(ctx, "old-terminal", "run") == addressed
    assert artifacts.read_actor_source_bytes(data, "old-terminal", tiny) == small
    assert not os.lstat(base / tiny["path"]).st_flags & stat.UF_COMPRESSED, "under the floor"
    assert not os.lstat(base / source["path"]).st_flags & stat.UF_COMPRESSED, "a shared file is never unshared"
    row = maintenance_jobs.job(data, job.JOB)
    assert row["last"]["skipped"] == {"shared": 1} and row["totals"]["evidence"]["compressed"] == 2
