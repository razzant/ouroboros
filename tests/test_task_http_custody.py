"""HTTP cancellation must settle admitted task work and artifact read repair."""
from __future__ import annotations

import asyncio
import json
import pathlib
import threading
from types import SimpleNamespace

import anyio
import pytest
from starlette.requests import Request

from ouroboros import artifacts
from ouroboros.gateway import tasks
from ouroboros.task_results import load_task_result, write_task_result
from tests._headless_cli_shared import _managed_worker_pool_available  # noqa: F401


def _request(data, repo, body=None, *, cursor=0, method="GET"):
    async def receive():
        return {"type": "http.request", "body": json.dumps(body or {}).encode()}

    return Request({
        "type": "http", "method": method, "path": "/api/tasks/custody",
        "headers": [], "query_string": f"cursor={cursor}&wait=0".encode(),
        "path_params": {"task_id": "custody"},
        "app": SimpleNamespace(state=SimpleNamespace(drive_root=data, repo_dir=repo)),
    }, receive)


async def _cancel_while_held(operation, entered, release, mode, held_assertion):
    scopes = []
    cancelled = []

    async def call():
        with anyio.CancelScope() as scope:
            scopes.append(scope)
            try:
                await operation()
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

    waiter = asyncio.create_task(call())
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        if mode == "anyio":
            scopes[0].cancel()
        else:
            waiter.cancel()
        await asyncio.sleep(0)
        if mode == "asyncio":
            waiter.cancel()
        await asyncio.sleep(0.01)
        assert not waiter.done(), "HTTP cancellation abandoned its admitted worker"
        held_assertion()
    finally:
        release.set()
        try:
            await asyncio.wait_for(waiter, 5)
        except asyncio.CancelledError:
            assert mode == "asyncio"
    assert cancelled == [True]


@pytest.mark.parametrize("mode", ["asyncio", "anyio"])
@pytest.mark.parametrize("outcome", ["accepted", "atomic_rejection", "compose_failure", "snapshot_failure"])
def test_cancelled_create_settles_original_admission(tmp_path, monkeypatch, mode, outcome):
    from supervisor import queue

    data, repo = tmp_path / "data", tmp_path / "repo"
    data.mkdir()
    repo.mkdir()
    source = tmp_path / "input.txt"
    source.write_text("complete input", encoding="utf-8")
    pending, reservations, snapshots = [], {}, []
    monkeypatch.setattr(queue, "DRIVE_ROOT", data)
    monkeypatch.setattr(queue, "PENDING", pending)
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(queue, "ADMISSION_RESERVATIONS", reservations)

    def persist(reason=""):
        snapshots.append((reason, [dict(row) for row in pending]))
        return outcome != "snapshot_failure" or reason == "api_task_create_rollback"

    monkeypatch.setattr(queue, "persist_queue_snapshot", persist)
    if outcome == "compose_failure":
        def fail_composition(*args, **kwargs):
            raise RuntimeError("controlled composition failure")
        monkeypatch.setattr(tasks, "_compose_task_text", fail_composition)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = tasks.stage_initial_task_attachments
    captured = []

    def held_stage(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        try:
            result = original(*args, **kwargs)
            captured.extend(result[0])
            return result
        finally:
            finished.set()

    monkeypatch.setattr(tasks, "stage_initial_task_attachments", held_stage)
    attachments = [{"path": str(source), "label": f"input {i}"} for i in range(28)]
    if outcome == "atomic_rejection":
        attachments.append({"path": str(tmp_path / "missing.txt")})
    request = _request(data, repo, {
        "task_id": "custody", "description": "read all inputs", "memory_mode": "empty",
        "attachments": attachments, "allow_partial_attachments": False,
    }, method="POST")
    child = data / "state" / "headless_tasks" / "custody" / "data"

    def held_assertion():
        assert reservations.get("custody") and child.is_dir()
        assert not pending and not snapshots and not finished.is_set()

    asyncio.run(_cancel_while_held(
        lambda: tasks.api_tasks_create(request), entered, release, mode, held_assertion,
    ))
    assert finished.is_set() and not reservations
    row = load_task_result(data, "custody")
    if outcome == "accepted":
        assert len(pending) == 1 and pending[0]["id"] == "custody"
        assert snapshots[0][0] == "api_task_create" and len(snapshots[0][1]) == 1
        assert row["status"] == "scheduled" and child.is_dir()
        manifest = artifacts.resolve_attachment_manifest(child, "custody", pending[0])
        assert len(manifest) == 28
        assert pending[0]["attachment_manifest_ref"]
        assert all(pathlib.Path(item["abs_path"]).read_text(encoding="utf-8") == "complete input" for item in manifest)
    elif outcome == "snapshot_failure":
        # Cancellation cannot turn unknown persistence into a safe rollback.
        assert [item["id"] for item in pending] == ["custody"] and child.is_dir()
        assert row is None
        assert [reason for reason, _ in snapshots] == ["api_task_create"]
        assert all(pathlib.Path(item["abs_path"]).read_text(encoding="utf-8") == "complete input"
                   for item in captured if item.get("status") == "staged")
        assert queue.reserve_task_admission("custody", "other", drive_root=data)["reason"] == "duplicate_task_id"
    else:
        assert not pending and not child.exists()
        assert not artifacts.task_artifacts_dir(data, "custody", create=False).exists()
        staged = [item for item in captured if item.get("status") == "staged"]
        assert staged
        assert all(not pathlib.Path(item["abs_path"]).exists() for item in staged)
        assert row is None and not snapshots


@pytest.mark.parametrize(("surface", "mode"), [
    ("get", "asyncio"),
    ("legacy_initial", "anyio"),
    ("legacy_final", "anyio"),
    ("v2", "asyncio"),
])
def test_cancelled_result_read_finishes_materialization(tmp_path, monkeypatch, mode, surface):
    data = tmp_path / "data"
    data.mkdir()
    source = tmp_path / "complete.bin"
    source.write_bytes(b"complete output")
    record = artifacts.copy_file_to_task_artifacts(SimpleNamespace(drive_root=data, task_id="custody"), source)
    write_task_result(data, "custody", "completed", artifacts=[record], artifact_status="ready")
    original = tasks.load_effective_task_result
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    materialized = []

    def held_load(*args, **kwargs):
        if kwargs.get("materialize_artifacts", True) is False:
            return original(*args, **kwargs)
        entered.set()
        assert release.wait(5)
        try:
            result = original(*args, **kwargs)
            materialized.append(result)
            return result
        finally:
            finished.set()

    monkeypatch.setattr(tasks, "load_effective_task_result", held_load)
    request = _request(data, tmp_path / "repo", {"v": 2, "wait": 0},
                       cursor=100000 if surface == "legacy_final" else 0,
                       method="POST" if surface == "v2" else "GET")

    async def operation():
        if surface == "get":
            return await tasks.api_task_get(request)
        response = await tasks.api_task_events(request)
        async for _ in response.body_iterator:
            pass

    def held_assertion():
        assert not finished.is_set() and not materialized

    asyncio.run(_cancel_while_held(operation, entered, release, mode, held_assertion))
    assert finished.is_set() and len(materialized) == 1
    assert materialized[0]["status"] == "completed"
    assert materialized[0]["artifact_status"] == "ready"
    assert pathlib.Path(record["path"]).read_bytes() == b"complete output"


@pytest.mark.parametrize("mode", ["asyncio", "anyio"])
def test_cancelled_request_observes_worker_failure_before_return(mode):
    from ouroboros.gateway._helpers import run_sync_to_completion

    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def fail():
        entered.set()
        assert release.wait(5)
        finished.set()
        raise OSError("controlled disk failure")

    asyncio.run(_cancel_while_held(
        lambda: run_sync_to_completion(fail), entered, release, mode,
        lambda: None,
    ))
    assert finished.is_set()


@pytest.mark.parametrize("mode", ["asyncio", "anyio"])
def test_cancelled_v2_read_settles_before_closing_generator(tmp_path, monkeypatch, mode):
    from ouroboros.gateway.task_events import _TaskEventCursorFollower

    data = tmp_path / "data"
    data.mkdir()
    write_task_result(data, "custody", "running")
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    closed = []

    def held_rows(self):
        try:
            entered.set()
            assert release.wait(5)
            finished.set()
            yield self.envelope({"type": "progress", "data": {"text": "retained row"}})
        finally:
            closed.append(finished.is_set())

    monkeypatch.setattr(_TaskEventCursorFollower, "read_events", held_rows)
    request = _request(data, tmp_path / "repo", {"v": 2, "wait": 0}, method="POST")

    async def operation():
        response = await tasks.api_task_events(request)
        async for _ in response.body_iterator:
            pass

    asyncio.run(_cancel_while_held(operation, entered, release, mode, lambda: None))
    assert closed == [True]
