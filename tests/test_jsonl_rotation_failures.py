"""Optional rotation failures preserve dispatch, locks and the retained chain."""

import errno
import hashlib
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros.utils import append_jsonl, iter_jsonl_chain_objects
from supervisor import state
from tests.test_jsonl_rotation import ROTATED_LOG_NAMES
from tests.test_server_shutdown import _run, _supervisor_harness


@pytest.fixture(autouse=True)
def _rotation_diagnostics(monkeypatch):
    monkeypatch.setattr(state, "_ROTATION_ERROR_AT", {})


def _seed(root, name="events.jsonl"):
    path = root / "logs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'{"event":"before"}\n')
    return path


def _append(path, event):
    # Default append can fall back to unlocked writes: success alone proves nothing.
    assert append_jsonl(path, {"event": event}, require_lock=True, lock_timeout_sec=0)
    # append_jsonl writes LF on every platform (a binary descriptor and a binary fallback),
    # so the oracle is byte-exact everywhere, the LF seeds included.
    return (json.dumps({"event": event}) + "\n").encode()


def _recover(root, path, expected):
    state.rotate_jsonl_log_if_needed(root, path.name, path.stem, max_bytes=1)
    archives = sorted((root / "archive").glob(f"{path.stem}_*.jsonl"))
    assert b"".join(segment.read_bytes() for segment in archives) == expected
    assert path.read_bytes() == b""
    expected += _append(path, "after recovery")
    assert b"".join(segment.read_bytes() for segment in archives) + path.read_bytes() == expected
    assert list(iter_jsonl_chain_objects(path)) == [json.loads(line) for line in expected.splitlines()]


@pytest.mark.parametrize(("name", "prefix"), ROTATED_LOG_NAMES)
@pytest.mark.parametrize("winerror", [32, 33])
def test_sharing_conflict_defers_without_losing_rows(tmp_path, monkeypatch, caplog, name, prefix, winerror):
    path = _seed(tmp_path, name)
    expected = path.read_bytes()
    replace = os.replace
    attempts = []
    error = PermissionError(errno.EACCES, "file in use")
    error.winerror = winerror

    def blocked(source, destination):
        if source == path:
            attempts.append(source)
            raise error
        return replace(source, destination)

    monkeypatch.setattr(state.os, "replace", blocked)
    # A frozen clock makes four attempts one throttle window; no sleep API is
    # supplied, so a newly added retry delay fails this test instead of waiting.
    monkeypatch.setattr(state, "time", SimpleNamespace(monotonic=lambda: 100.0))
    with caplog.at_level(logging.ERROR, logger=state.__name__):
        for tick in range(4):
            state.rotate_jsonl_log_if_needed(tmp_path, name, prefix, max_bytes=1)
            assert path.read_bytes() == expected
            expected += _append(path, f"held {tick}")
    assert len(attempts) == 4  # throttling diagnostics never throttles attempts
    assert not list((tmp_path / "archive").glob("*.jsonl"))
    records = [record for record in caplog.records if record.name == state.__name__]
    assert len(records) == 1 and records[0].levelno == logging.ERROR
    assert records[0].exc_info[1] is error
    assert str(path) in records[0].getMessage()
    monkeypatch.setattr(state.os, "replace", replace)
    _recover(tmp_path, path, expected)


@pytest.mark.parametrize("stage", [
    "sentinel_stat", "prelock_stat", "lock_path", "acquire", "locked_stat",
    "archive_stat", "mkdir", "rename", "touch", "release",
])
def test_oserror_at_each_rotation_stage_preserves_history_and_releases_lock(
        tmp_path, monkeypatch, caplog, stage):
    from ouroboros import utils

    path = _seed(tmp_path)
    expected = path.read_bytes()
    error = OSError(errno.ENOSPC, f"injected {stage}")
    original_stat = Path.stat
    original_exists = Path.exists
    acquired = []
    acquire = state.acquire_exclusive_file_lock

    def stat(target, *args, **kwargs):
        if ((stage == "prelock_stat" and target == path)
                or (stage == "locked_stat" and target == path and acquired)):
            raise error
        return original_stat(target, *args, **kwargs)

    def exists(target):
        if ((stage == "sentinel_stat" and target.name == state.ISOLATED_BENCHMARK_SENTINEL)
                or (stage == "archive_stat" and target.parent == tmp_path / "archive")):
            raise error
        # Exercise the explicit size stat, independent of Python versions whose
        # exists() hides every OSError (rather than only missing-path errors).
        if target == path:
            return True
        return original_exists(target)

    def tracked_acquire(*args, **kwargs):
        fd = acquire(*args, **kwargs)
        if fd is not None:
            acquired.append(fd)
        return fd

    def fail(*args, **kwargs):
        raise error

    release = state.release_exclusive_file_lock

    def release_then_fail(*args):
        # The helper owns cleanup. An error reported after release must not be
        # mistaken for successful rotation or cause an unsafe second release.
        release(*args)
        raise error

    with monkeypatch.context() as patch, caplog.at_level(logging.ERROR, logger=state.__name__):
        patch.setattr(Path, "stat", stat)
        patch.setattr(Path, "exists", exists)
        patch.setattr(state, "acquire_exclusive_file_lock", tracked_acquire)
        if stage == "lock_path":
            patch.setattr(utils, "jsonl_append_lock_path", fail)
        elif stage == "acquire":
            patch.setattr(state, "acquire_exclusive_file_lock", fail)
        elif stage == "mkdir":
            mkdir = Path.mkdir
            patch.setattr(Path, "mkdir", lambda target, *a, **kw:
                          fail() if target == tmp_path / "archive" else mkdir(target, *a, **kw))
        elif stage == "rename":
            patch.setattr(state.os, "replace", fail)
        elif stage == "touch":
            patch.setattr(Path, "touch", fail)
        elif stage == "release":
            patch.setattr(state, "release_exclusive_file_lock", release_then_fail)
        state.rotate_jsonl_log_if_needed(tmp_path, path.name, path.stem, max_bytes=1)

    records = [record for record in caplog.records if record.name == state.__name__]
    assert len(records) == 1 and records[0].exc_info[1] is error
    archives = sorted((tmp_path / "archive").glob("*.jsonl"))
    if stage in {"touch", "release"}:
        assert len(archives) == 1 and archives[0].read_bytes() == expected
        if stage == "touch":
            assert not path.exists()
        else:
            assert path.read_bytes() == b""
    else:
        assert not archives and path.read_bytes() == expected
    expected += _append(path, "after failure")
    _recover(tmp_path, path, expected)


def test_error_throttle_is_per_path_expires_and_resets_after_recovery(tmp_path, monkeypatch, caplog):
    paths = [_seed(tmp_path / root) for root in ("one", "two")]
    now = [100.0]
    monkeypatch.setattr(state, "time", SimpleNamespace(monotonic=lambda: now[0]))
    replace = os.replace

    def fail(source, destination):
        raise OSError(errno.EACCES, "rotation denied")

    with caplog.at_level(logging.ERROR, logger=state.__name__):
        monkeypatch.setattr(state.os, "replace", fail)
        for delta in (0, 29.9, 30):
            now[0] = 100 + delta
            for path in paths:
                state.rotate_jsonl_log_if_needed(path.parents[1], path.name, path.stem, max_bytes=1)
        assert len(caplog.records) == 4
        monkeypatch.setattr(state.os, "replace", replace)
        path = paths[0]
        state.rotate_jsonl_log_if_needed(path.parents[1], path.name, path.stem, max_bytes=1)
        _append(path, "recovered")
        monkeypatch.setattr(state.os, "replace", fail)
        state.rotate_jsonl_log_if_needed(path.parents[1], path.name, path.stem, max_bytes=1)
    assert len(caplog.records) == 5


def test_programming_failure_propagates_after_lock_release(tmp_path, monkeypatch):
    path = _seed(tmp_path)

    def fail(*args):
        raise ValueError("invalid rotation logic")

    monkeypatch.setattr(state.os, "replace", fail)
    with pytest.raises(ValueError, match="invalid rotation logic"):
        state.rotate_jsonl_log_if_needed(tmp_path, path.name, path.stem, max_bytes=1)
    _append(path, "after programming failure")


@pytest.mark.parametrize("owner", ["atomic", "cutover"])
def test_mandatory_persistence_still_propagates_oserror(tmp_path, monkeypatch, owner):
    from ouroboros import dialogue_import, utils

    path = _seed(tmp_path, "chat.jsonl")
    original = path.read_bytes()
    error = OSError(errno.ENOSPC, "mandatory persistence unavailable")

    def fail(*args):
        raise error

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError) as caught:
        if owner == "atomic":
            utils.replace_atomic(path, tmp_path / "destination")
        else:
            dialogue_import.rotate_final(tmp_path, "chat")
    assert caught.value is error and path.read_bytes() == original
    _append(path, "after mandatory failure")


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows sharing semantics")
def test_real_reader_without_delete_sharing_cannot_kill_rotation(tmp_path, caplog, record_property):
    # Function-local Win32 imports: this probe must still collect on POSIX.
    import ctypes
    from ctypes import wintypes

    repo = Path(__file__).resolve().parents[1]
    hashes = {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in (
        "supervisor/state.py", "server.py", "ouroboros/platform_layer.py", "ouroboros/utils.py",
        "tests/test_jsonl_rotation_failures.py", "tests/test_server_shutdown.py",
    )}
    record_property("source_sha256", json.dumps(hashes, sort_keys=True))
    print("ROTATION_NATIVE_SOURCE " + json.dumps(hashes, sort_keys=True))
    path = _seed(tmp_path)
    expected = path.read_bytes()
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p]
    kernel.CreateFileW.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.restype = wintypes.BOOL
    # GENERIC_READ; FILE_SHARE_READ | FILE_SHARE_WRITE, deliberately no FILE_SHARE_DELETE.
    handle = kernel.CreateFileW(str(path), 0x80000000, 3, None, 3, 0x80, None)
    assert handle not in (None, ctypes.c_void_p(-1).value), ctypes.get_last_error()
    try:
        # Falsifying physical control: no fabricated winerror, prove this handle
        # really denies rename before asking the protected production rotator.
        with pytest.raises(OSError) as raw:
            os.replace(path, tmp_path / "raw-attempt.jsonl")
        winerror = raw.value.winerror
        record_property("raw_winerror", winerror)
        print(f"ROTATION_NATIVE_RAW winerror={winerror} error={raw.value!r}")
        assert winerror in (32, 33)
        assert path.read_bytes() == expected and not (tmp_path / "raw-attempt.jsonl").exists()
        with caplog.at_level(logging.ERROR, logger=state.__name__):
            for tick in range(4):
                state.rotate_jsonl_log_if_needed(tmp_path, path.name, path.stem, max_bytes=1)
                expected += _append(path, f"while held {tick}")
                assert path.read_bytes() == expected
        assert not list((tmp_path / "archive").glob("*.jsonl"))
        records = [record for record in caplog.records if record.name == state.__name__]
        assert len(records) == 1 and records[0].exc_info[1].winerror == winerror
        record_property("protected_winerror", records[0].exc_info[1].winerror)
    finally:
        assert kernel.CloseHandle(handle)
    _recover(tmp_path, path, expected)
    record_property("held_ticks", 4)
    record_property("strict_append_and_exact_chain", "passed")
    print("ROTATION_NATIVE_RECOVERY held_ticks=4 strict_append_and_exact_chain=passed")


@pytest.mark.parametrize("name", ["events", "tools", "supervisor", "task_reflections"])
@pytest.mark.parametrize("winerror", [32, 33])
def test_rotation_oserror_keeps_real_supervisor_dispatching(
        monkeypatch, tmp_path, caplog, record_property, name, winerror):
    """Four failed rotations reach every downstream dispatch seam; recovery keeps exact bytes.

    The harness normally stubs rotation AND suppresses assignment. Restore the
    actual rotator and open the assignment branch, with no workers/providers.
    """
    import logging

    import server
    from ouroboros.utils import append_jsonl, iter_jsonl_chain_objects
    from supervisor import state, workers

    rotate = state.rotate_jsonl_log_if_needed
    rec = _supervisor_harness(monkeypatch, tmp_path, [])
    monkeypatch.setattr(state, "rotate_jsonl_log_if_needed", rotate)
    monkeypatch.setattr(state, "_ROTATION_ERROR_AT", {})
    monkeypatch.setattr(server, "_check_pending_restart_drain", lambda _ctx: False)
    monkeypatch.setattr("ouroboros.post_task_evolution.apply_pending_request", lambda *_a: None)
    path = tmp_path / "logs" / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = (json.dumps({"event": "before", "pad": "x" * 800_000}) + "\n").encode()
    path.write_bytes(expected)
    replace = os.replace
    attempts, phases = [], []
    error = PermissionError(13, "reader denies rename")
    error.winerror = winerror

    def blocked(source, destination):
        if source == path:
            attempts.append(source)
            raise error
        return replace(source, destination)

    def drained(*_a, **_kw):
        phases.append("events")
        return False

    def intake(_bridge, offset, _ctx, **_kw):
        phases.append("intake")
        return offset + 1

    def assigned():
        nonlocal expected
        assert rec.ready.is_set()
        phases.append("assign")
        row = {"event": f"tick {len(attempts)}"}
        assert append_jsonl(path, row, require_lock=True, lock_timeout_sec=0)
        expected += (json.dumps(row) + "\n").encode()
        if len(attempts) == 4:
            rec.stop.set()

    monkeypatch.setattr(state.os, "replace", blocked)
    monkeypatch.setattr(workers, "ensure_workers_healthy", lambda: phases.append("health"))
    monkeypatch.setattr(server, "drain_worker_events", drained)
    monkeypatch.setattr(server, "_process_bridge_updates", intake)
    monkeypatch.setattr(workers, "assign_tasks", assigned)
    with caplog.at_level(logging.ERROR):
        _run(rec, server)
    observation = {"log": name, "injected_winerror": winerror, "rotation_attempts": len(attempts),
                   "phases": phases, "ready": rec.ready.is_set(), "error": server._supervisor_error,
                   "alerts": len(rec.alerts), "backoffs": rec.stop.waits}
    record_property("supervisor_observation", json.dumps(observation, sort_keys=True))
    print("ROTATION_SUPERVISOR " + json.dumps(observation, sort_keys=True))
    assert len(attempts) == 4
    assert phases == ["health", "events", "intake", "assign"] * 4
    assert rec.ready.is_set() and server._supervisor_error is None
    assert not rec.alerts and not rec.stop.waits
    assert not any(record.levelno >= logging.CRITICAL for record in caplog.records)
    assert path.read_bytes() == expected
    assert not list((tmp_path / "archive").glob(f"{name}_*.jsonl"))
    monkeypatch.setattr(state.os, "replace", replace)
    rotate(tmp_path, path.name, name)
    archives = list((tmp_path / "archive").glob(f"{name}_*.jsonl"))
    assert len(archives) == 1 and archives[0].read_bytes() == expected
    assert path.read_bytes() == b""
    expected += _append(path, "recovered")
    assert archives[0].read_bytes() + path.read_bytes() == expected
    assert list(iter_jsonl_chain_objects(path)) == [json.loads(line) for line in expected.splitlines()]


@pytest.mark.parametrize("failure", ["rotation_programming", "worker_health"])
def test_nonoptional_failures_still_kill_real_supervisor(monkeypatch, tmp_path, failure):
    import server
    from supervisor import state, workers

    rotate = state.rotate_jsonl_log_if_needed
    rec = _supervisor_harness(monkeypatch, tmp_path, [])
    monkeypatch.setattr(state, "rotate_jsonl_log_if_needed", rotate)
    path = tmp_path / "logs" / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * 800_000)
    attempts = []

    def fail(*_a):
        attempts.append(failure)
        if failure == "rotation_programming":
            raise ValueError("rotation programming defect")
        raise OSError("critical worker health unavailable")

    if failure == "rotation_programming":
        monkeypatch.setattr(state.os, "replace", fail)
    else:
        monkeypatch.setattr(workers, "ensure_workers_healthy", fail)
    _run(rec, server)
    assert len(attempts) == 3
    assert not rec.ready.is_set() and "3 consecutive crashes" in server._supervisor_error
    assert rec.stop.waits == [2, 4]
    assert len(rec.alerts) == 1 and "died after repeated crashes" in rec.alerts[0][1]
