"""Fixtures, stubs and live-process scaffolding shared by the cancel-intent suites.

Split out of ``tests/test_cancel_intents_phase_a.py`` when that module was divided by
theme; every definition is verbatim (tip bytes), so each sibling suite keeps the exact
semantics it was written against. ``_reap_spawned_live_procs`` is autouse, so importing
it into a test module re-applies it there — every module that spawns a ``_LiveProc``
must import it. ``_write_root_retry_pair`` joined the shared set on the v7next tip: the
upstream retry-race hardening reads it from both the mint suite and the custody suite.
"""

from __future__ import annotations
import pathlib
import subprocess
import sys
import types
import pytest
from ouroboros.task_results import (
    write_task_result,
)


def _write_root_retry_pair(tmp_path, old_id: str, new_id: str, *, new_status="scheduled"):
    write_task_result(
        tmp_path,
        old_id,
        "interrupted",
        root_task_id=old_id,
        delegation_role="root",
        superseded_by=new_id,
        retry_task_id=new_id,
    )
    write_task_result(
        tmp_path,
        new_id,
        new_status,
        root_task_id=old_id,
        parent_task_id="",
        delegation_role="root",
        supersedes_task_id=old_id,
        original_task_id=old_id,
        timeout_retry_from=old_id,
    )

@pytest.fixture()
def qenv(tmp_path, monkeypatch):
    import supervisor.queue as q
    from supervisor import task_lifecycle, workers

    monkeypatch.setattr(q, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(q, "PENDING", [])
    monkeypatch.setattr(q, "RUNNING", {}, raising=False)
    monkeypatch.setattr(workers, "WORKERS", {}, raising=False)
    monkeypatch.setattr(workers, "respawn_worker", lambda wid: None, raising=False)
    monkeypatch.setattr(q, "persist_queue_snapshot", lambda reason="": None)
    monkeypatch.setattr(task_lifecycle, "CANCELLED_ROOT_FENCES", {}, raising=False)
    monkeypatch.setattr(task_lifecycle, "_ACTIVE_CASCADE_FENCES", {}, raising=False)
    return types.SimpleNamespace(q=q, tl=task_lifecycle, workers=workers, drive=tmp_path)

class _CaptureQueue:
    def __init__(self):
        self.events = []

    def put(self, evt):
        self.events.append(evt)

class _LiveProc:
    """A REAL OS process behind the worker-proc surface custody expects.

    Tests spawning these belong to the SERIAL lane (`@pytest.mark.serial`,
    tests/conftest policy: real-subprocess tests flake or crash xdist workers
    under `-n auto`) and every spawn is registered so the autouse reaper below
    terminates AND waits it even when the test fails before its own kill path
    runs — a leaked 120s sleeper must never outlive its test (GR2-10).
    """

    _SPAWNED: list = []

    def __init__(self):
        self._proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(120)"],
        )
        self.pid = self._proc.pid
        _LiveProc._SPAWNED.append(self._proc)

    def is_alive(self) -> bool:
        return self._proc.poll() is None

    def join(self, timeout=None):
        try:
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass

    def terminate(self):
        self._proc.terminate()

@pytest.fixture(autouse=True)
def _reap_spawned_live_procs():
    """Terminate AND reap (wait) every _LiveProc spawned by a test (GR2-10)."""
    yield
    while _LiveProc._SPAWNED:
        proc = _LiveProc._SPAWNED.pop()
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
            # poll() already reaped an exited child; nothing more owed.
        except Exception:
            pass

def _seed_llm_response(drive: pathlib.Path, task_id: str, text: str) -> None:
    from ouroboros import observability

    blob = observability.write_blob(drive, {"message": {"content": text}})
    observability.write_call_manifest(
        drive, task_id=task_id, call_id="llm_0001_response",
        manifest={"full_payload_ref": blob},
    )

def _live_split_drive_task(qenv, task_id: str) -> tuple[dict, pathlib.Path, _LiveProc]:
    from ouroboros.headless import HEADLESS_TASKS_DIR

    child_drive = qenv.drive / HEADLESS_TASKS_DIR / task_id / "data"
    child_drive.mkdir(parents=True)
    task = {
        "id": task_id,
        "chat_id": 5,
        "delegation_role": "subagent",
        "parent_task_id": "parent-e2e",
        "root_task_id": "parent-e2e",
        "child_drive_root": str(child_drive),
    }
    proc = _LiveProc()
    worker = types.SimpleNamespace(wid=0, proc=proc, busy_task_id=task_id, reaping=False)
    qenv.workers.WORKERS[0] = worker
    qenv.q.RUNNING[task_id] = {"task": task, "worker_id": 0}
    return task, child_drive, proc
