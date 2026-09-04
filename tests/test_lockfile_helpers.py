import os
import subprocess
import sys
import time

import pytest

from ouroboros import usage_ledger
from ouroboros.platform_layer import (
    acquire_exclusive_file_lock,
    release_exclusive_file_lock,
    unlink_lockfile,
)


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_dead_owner_lock_is_reclaimed_without_waiting_out_stale_window(tmp_path):
    lock_path = tmp_path / "usage_attempts.lock"
    lock_path.write_text(f"pid={_dead_pid()} ts={time.time()}\n", encoding="utf-8")

    started = time.monotonic()
    fd = acquire_exclusive_file_lock(
        lock_path, timeout_sec=2.0, stale_sec=90.0, reclaim_dead_owner=True,
    )
    elapsed = time.monotonic() - started

    assert fd is not None
    assert elapsed < 1.0
    assert f"pid={os.getpid()}" in lock_path.read_text(encoding="utf-8")
    release_exclusive_file_lock(lock_path, fd)


def test_live_owner_lock_is_never_reclaimed_by_dead_owner_path(tmp_path):
    lock_path = tmp_path / "usage_attempts.lock"
    # Our own pid is live: the dead-owner path must leave the lock alone, and a
    # fresh lock is younger than the stale window, so acquisition times out.
    lock_path.write_text(f"pid={os.getpid()} ts={time.time()}\n", encoding="utf-8")

    fd = acquire_exclusive_file_lock(
        lock_path, timeout_sec=0.3, stale_sec=90.0, reclaim_dead_owner=True,
    )

    assert fd is None
    assert lock_path.exists()


def test_lock_without_pid_metadata_still_waits_for_stale_window(tmp_path):
    lock_path = tmp_path / "usage_attempts.lock"
    lock_path.write_text("owned elsewhere\n", encoding="utf-8")

    fd = acquire_exclusive_file_lock(
        lock_path, timeout_sec=0.3, stale_sec=90.0, reclaim_dead_owner=True,
    )

    assert fd is None
    assert lock_path.exists()


def test_default_acquire_keeps_stale_age_contract_for_dead_owner(tmp_path):
    lock_path = tmp_path / "other.lock"
    lock_path.write_text(f"pid={_dead_pid()} ts={time.time()}\n", encoding="utf-8")

    fd = acquire_exclusive_file_lock(lock_path, timeout_sec=0.3, stale_sec=90.0)

    assert fd is None
    assert lock_path.exists()


def test_usage_ledger_lock_survives_owner_killed_mid_transaction(tmp_path):
    root = tmp_path / "data"
    (root / "state").mkdir(parents=True)
    lock_path = root / "state" / "usage_attempts.lock"
    # A worker that took the ledger lock and was then terminated by custody.
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os, sys, time; p = sys.argv[1]; "
            "open(p, 'x').write(f'pid={os.getpid()} ts={time.time()}\\n'); "
            "sys.stdout.write('held\\n'); sys.stdout.flush(); time.sleep(60)",
            str(lock_path),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout is not None
    assert holder.stdout.readline().strip() == "held"
    holder.terminate()
    holder.wait(timeout=10)
    assert lock_path.exists()

    started = time.monotonic()
    with usage_ledger._locked(root, timeout_sec=5.0):
        elapsed = time.monotonic() - started
    assert elapsed < 2.0
    assert not lock_path.exists()


def test_usage_ledger_lock_still_times_out_behind_a_live_holder(tmp_path):
    root = tmp_path / "data"
    (root / "state").mkdir(parents=True)
    lock_path = root / "state" / "usage_attempts.lock"
    lock_path.write_text(f"pid={os.getpid()} ts={time.time()}\n", encoding="utf-8")

    with pytest.raises(usage_ledger.UsageLockUnavailable):
        with usage_ledger._locked(root, timeout_sec=0.3):
            pass
    assert lock_path.exists()


def test_release_without_fd_does_not_unlink_existing_lock(tmp_path):
    lock_path = tmp_path / "state.lock"
    lock_path.write_text("owned elsewhere", encoding="utf-8")

    release_exclusive_file_lock(lock_path, None)

    assert lock_path.read_text(encoding="utf-8") == "owned elsewhere"


def test_release_with_fd_unlinks_owned_lock(tmp_path):
    lock_path = tmp_path / "state.lock"
    fd = acquire_exclusive_file_lock(lock_path, metadata="owned\n")
    assert fd is not None

    release_exclusive_file_lock(lock_path, fd)

    assert not lock_path.exists()


def test_path_only_git_lock_cleanup_remains_available(tmp_path):
    lock_path = tmp_path / "git.lock"
    fd = acquire_exclusive_file_lock(lock_path, metadata="owned\n")
    assert fd is not None
    os.close(fd)

    unlink_lockfile(lock_path)

    assert not lock_path.exists()
