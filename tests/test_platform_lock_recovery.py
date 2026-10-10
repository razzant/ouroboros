"""Owner death, age and kernel ownership are separate lock observations."""

import os
import subprocess
import sys
import time

import pytest

from ouroboros import platform_layer as platform


@pytest.mark.parametrize("age", [0, 120], ids=["fresh", "old"])
def test_dead_owner_recovery_keeps_kernel_exclusion(tmp_path, monkeypatch, age):
    lock = tmp_path / "state.lock"
    fd = platform.acquire_exclusive_file_lock(lock, metadata="pid=424242\n")
    assert fd is not None
    try:
        os.utime(lock, (time.time() - age,) * 2)
        monkeypatch.setattr(platform, "pid_is_alive", lambda pid: False)
        assert platform.acquire_exclusive_file_lock(
            lock, timeout_sec=0.1, stale_sec=90, owner_aware_stale=True,
        ) is None
        assert lock.read_text() == "pid=424242\n"
    finally:
        platform.release_exclusive_file_lock(lock, fd)


@pytest.mark.parametrize("metadata,owner_aware", [("unknown\n", True), ("pid=424242\n", False)])
def test_unproven_or_unrequested_owner_recovery_keeps_age_grace(tmp_path, monkeypatch, metadata, owner_aware):
    lock = tmp_path / "state.lock"
    lock.write_text(metadata)
    monkeypatch.setattr(platform, "pid_is_alive", lambda pid: False)
    assert platform.acquire_exclusive_file_lock(
        lock, timeout_sec=0.1, stale_sec=90, owner_aware_stale=owner_aware,
    ) is None
    assert lock.read_text() == metadata
    os.utime(lock, (0, 0))
    fd = platform.acquire_exclusive_file_lock(
        lock, timeout_sec=1, stale_sec=90, owner_aware_stale=owner_aware,
    )
    assert fd is not None
    platform.release_exclusive_file_lock(lock, fd)


@pytest.mark.serial
def test_fresh_lock_of_reaped_process_is_recovered_within_caller_budget(tmp_path):
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(timeout=10)
    assert not platform.pid_is_alive(child.pid)
    lock = tmp_path / "usage_attempts.lock"
    lock.write_text(f"pid={child.pid}\n")
    started = time.monotonic()
    fd = platform.acquire_exclusive_file_lock(
        lock, timeout_sec=2, stale_sec=90, owner_aware_stale=True,
    )
    assert fd is not None
    try:
        assert time.monotonic() - started < 2
        assert f"pid={os.getpid()}" in lock.read_text()
    finally:
        platform.release_exclusive_file_lock(lock, fd)


@pytest.mark.parametrize("creation", [FileExistsError, PermissionError])
@pytest.mark.parametrize("windows", [True, False], ids=["windows", "posix"])
def test_a_name_that_refuses_its_probe_mid_release_is_contention_only_on_windows(
        tmp_path, monkeypatch, creation, windows):
    """Windows refuses to open a name while its holder deletes it (delete pending, or the
    deleter's handle shares no access), so a pre-send accounting wait must re-contend.
    The same refusal on POSIX stays a permission failure that fails closed."""
    lock = tmp_path / "usage_attempts.lock"
    opened = os.open

    def releasing_holder(path, flags, *args, **kwargs):
        if str(path) != str(lock):
            return opened(path, flags, *args, **kwargs)
        if flags & os.O_CREAT:
            raise creation(17 if creation is FileExistsError else 13, "held")
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(platform, "kernel_file_locks_enforced", lambda path: True)
    monkeypatch.setattr(platform, "IS_WINDOWS", windows)
    monkeypatch.setattr(os, "open", releasing_holder)
    outcome = {}
    assert platform.acquire_exclusive_file_lock(lock, timeout_sec=0.1, outcome=outcome) is None
    assert outcome == {"reason": "contention" if windows else "permission", "errno": 13}
