"""Failure path of the supervised-spawn chokepoint: custody-write failure.

Kept out of test_process_custody.py only for the module-size band; it is the
same subject (spawn_supervised).
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

from ouroboros.platform_layer import subprocess_new_group_kwargs

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_POSIX_ONLY = pytest.mark.skipif(os.name == "nt", reason="POSIX process groups only")


_RECORD_FAILURE_SPAWNER = r"""
import os, subprocess, sys
from ouroboros import process_custody

def explode(*_a, **_k):
    raise OSError("simulated custody ledger write failure")

process_custody.record_process = explode
try:
    process_custody.spawn_supervised(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        drive_root=__import__("pathlib").Path(sys.argv[1]),
        purpose="record-failure-probe", scope="session",
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
except RuntimeError as exc:
    print("survived", type(exc.__cause__).__name__, flush=True)
else:
    print("no-error", flush=True)
"""


@_POSIX_ONLY
def test_record_failure_kills_only_the_child_never_the_spawners_group(tmp_path):
    """Failure path of the chokepoint: a custody-write failure makes
    spawn_supervised kill_process_tree(child), which SIGKILLs the child's whole
    POSIX group. With the default dedicated group the spawner survives; a child
    spawned into the spawner's own group would take the spawner (the server and
    every sibling in that group) down with it. The spawner runs as a separate
    session here so the assertion is about ITS survival, not the test runner's."""
    spawner = subprocess.run(
        [sys.executable, "-c", _RECORD_FAILURE_SPAWNER, str(tmp_path)],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        **subprocess_new_group_kwargs(),
    )
    assert spawner.returncode == 0, spawner.stderr
    assert spawner.stdout.split() == ["survived", "OSError"], (spawner.stdout, spawner.stderr)


def test_a_kept_launch_the_ownership_set_refuses_is_no_kept_service(tmp_path, monkeypatch):
    """A kept launch outlives its worker, and only the installation's ownership set hands it to the stops that end it
    then. When the set takes neither its record nor a pending copy, ``record_process`` raises after the ledger append,
    so ``spawn_supervised`` ends the child with its own handle and the start says why, instead of a keep nothing would
    honour; a task-scoped row keeps the ledger's warning. Each kept row carries its leader's kernel birth, ``""`` when
    it cannot be read (unknown, never a guess); other rows carry none. Synthetic: no process, no signal."""
    from ouroboros import owned_shutdown, process_custody, process_identity

    monkeypatch.setattr(owned_shutdown, "record_ledgered_process", lambda *a, **k: False)
    monkeypatch.setattr(process_custody, "process_fingerprint", lambda pid, cmd=None: {})
    monkeypatch.setattr(process_custody, "process_group_id", lambda pid: 0)
    monkeypatch.setattr(process_identity, "process_kernel_birth",
                        lambda pid: "darwin-start:1790000000.000001" if pid == 999_999_941 else None)
    (tmp_path / "state").mkdir()
    for pid, purpose in ((999_999_941, "service:page"), (999_999_942, "workspace_service:page")):
        with pytest.raises(OSError, match="ownership set"):
            process_custody.record_process(tmp_path, pid=pid, cmd=["x"], purpose=purpose, scope="session")
    row = process_custody.record_process(tmp_path, pid=999_999_943, cmd=["x"], purpose="service:page", scope="task",
                                         owner_task_id="t")
    assert row["scope"] == "task"
    births = {entry["pid"]: entry.get("kernel_birth") for entry in process_custody._read_ledger(tmp_path)}
    assert births == {999_999_941: "darwin-start:1790000000.000001", 999_999_942: "", 999_999_943: None}
