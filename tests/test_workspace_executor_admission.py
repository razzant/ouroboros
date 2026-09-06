"""What ``POST /api/tasks`` accepts as an executor reference.

Split verbatim out of ``tests/test_workspace_executor.py`` by theme. This module owns
the normalized reference it admits and every refusal around it: no external workspace,
an empty or malformed reference or mapping entry, a mapping onto the system repo or the
data drive, a mapping that does not cover the workspace, the reserved metadata aliases,
and ``network=none`` asked of the local backend.

Whole-file serial suite: it spawns real processes, so ``tests/conftest.py`` tags it
``serial`` and the parallel pass excludes it.
"""

from __future__ import annotations

import json
import asyncio
from types import SimpleNamespace



from tests._workspace_executor_shared import _init_repo


def test_api_task_metadata_accepts_normalized_executor_ref(tmp_path, monkeypatch):
    from ouroboros.gateway import tasks
    import supervisor.queue as queue
    import supervisor.workers as workers

    captured: dict[str, object] = {}

    async def fake_request_json_or(_request, _default):
        return {
            "description": "x",
            "workspace_root": str(tmp_path / "workspace"),
            "workspace_mode": "external",
            "memory_mode": "empty",
            "executor_ref": {
                "type": "local",
                "id": "local-api",
                "workspace_host_path": str(tmp_path / "workspace"),
                "workspace_backend_path": "/workspace",
            },
        }

    def fake_enqueue(task):
        captured.update(task)
        return task

    _init_repo(tmp_path / "workspace")
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(tasks, "request_json_or", fake_request_json_or)
    monkeypatch.setattr(tasks, "request_drive_root", lambda _request: tmp_path / "data")
    monkeypatch.setattr(tasks, "request_repo_dir", lambda _request: tmp_path / "repo")
    monkeypatch.setattr(queue, "enqueue_task", fake_enqueue)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda *a, **k: True)
    monkeypatch.setattr(workers, "WORKERS", {0: SimpleNamespace()})
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "")

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(supervisor_ready_event=None)))
    response = asyncio.run(tasks.api_tasks_create(request))
    body = json.loads(response.body.decode("utf-8"))

    assert body["ok"] is True
    metadata = captured["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["executor_ref"]["type"] == "local"
    assert metadata["executor_ref"]["id"] == "local-api"
    assert metadata["executor_ref"]["workspace_backend_path"] == "/workspace"
    assert metadata["executor_ref"]["path_mappings"][0]["host_path"] == str((tmp_path / "workspace").resolve(strict=False))


def test_api_task_rejects_executor_ref_without_external_workspace(tmp_path, monkeypatch):
    from ouroboros.gateway import tasks

    async def fake_request_json_or(_request, _default):
        return {
            "description": "x",
            "executor_ref": {"type": "local", "workspace_host_path": str(tmp_path), "workspace_backend_path": "/workspace"},
        }

    monkeypatch.setattr(tasks, "request_json_or", fake_request_json_or)
    monkeypatch.setattr(tasks, "request_drive_root", lambda _request: tmp_path / "data")
    monkeypatch.setattr(tasks, "request_repo_dir", lambda _request: tmp_path / "repo")
    (tmp_path / "data").mkdir()
    (tmp_path / "repo").mkdir()

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(supervisor_ready_event=None)))
    response = asyncio.run(tasks.api_tasks_create(request))
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 400
    assert "executor_ref requires an external workspace_root" in body["error"]


def test_api_task_rejects_empty_executor_ref(tmp_path, monkeypatch):
    from ouroboros.gateway import tasks

    workspace = tmp_path / "workspace"
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    _init_repo(workspace)
    repo.mkdir()
    data.mkdir()

    async def fake_request_json_or(_request, _default):
        return {
            "description": "x",
            "workspace_root": str(workspace),
            "workspace_mode": "external",
            "memory_mode": "empty",
            "executor_ref": {},
        }

    monkeypatch.setattr(tasks, "request_json_or", fake_request_json_or)
    monkeypatch.setattr(tasks, "request_drive_root", lambda _request: data)
    monkeypatch.setattr(tasks, "request_repo_dir", lambda _request: repo)

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(supervisor_ready_event=None)))
    response = asyncio.run(tasks.api_tasks_create(request))
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 400
    # ABI-3 ingress schema: the nested ExecutorRef contract fires first.
    assert "executor_ref" in body["error"]


def test_api_task_rejects_executor_ref_mapping_to_system_repo(tmp_path, monkeypatch):
    from ouroboros.gateway import tasks

    repo = tmp_path / "repo"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    _init_repo(repo)
    _init_repo(workspace)
    data.mkdir()

    async def fake_request_json_or(_request, _default):
        return {
            "description": "x",
            "workspace_root": str(workspace),
            "workspace_mode": "external",
            "memory_mode": "empty",
            "executor_ref": {"type": "local", "workspace_host_path": str(repo), "workspace_backend_path": "/workspace"},
        }

    monkeypatch.setattr(tasks, "request_json_or", fake_request_json_or)
    monkeypatch.setattr(tasks, "request_drive_root", lambda _request: data)
    monkeypatch.setattr(tasks, "request_repo_dir", lambda _request: repo)

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(supervisor_ready_event=None)))
    response = asyncio.run(tasks.api_tasks_create(request))
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 400
    assert "must not overlap the Ouroboros system repo" in body["error"]


def test_api_task_rejects_executor_ref_mapping_to_data_drive(tmp_path, monkeypatch):
    from ouroboros.gateway import tasks

    repo = tmp_path / "repo"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    _init_repo(repo)
    _init_repo(workspace)
    data.mkdir()

    async def fake_request_json_or(_request, _default):
        return {
            "description": "x",
            "workspace_root": str(workspace),
            "workspace_mode": "external",
            "memory_mode": "empty",
            "executor_ref": {"type": "local", "workspace_host_path": str(data), "workspace_backend_path": "/workspace"},
        }

    monkeypatch.setattr(tasks, "request_json_or", fake_request_json_or)
    monkeypatch.setattr(tasks, "request_drive_root", lambda _request: data)
    monkeypatch.setattr(tasks, "request_repo_dir", lambda _request: repo)

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(supervisor_ready_event=None)))
    response = asyncio.run(tasks.api_tasks_create(request))
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 400
    assert "must not overlap the Ouroboros data drive" in body["error"]


def test_api_task_rejects_executor_ref_not_covering_workspace(tmp_path, monkeypatch):
    from ouroboros.gateway import tasks

    repo = tmp_path / "repo"
    workspace = tmp_path / "workspace"
    other = tmp_path / "other"
    data = tmp_path / "data"
    _init_repo(repo)
    _init_repo(workspace)
    other.mkdir()
    data.mkdir()

    async def fake_request_json_or(_request, _default):
        return {
            "description": "x",
            "workspace_root": str(workspace),
            "workspace_mode": "external",
            "memory_mode": "empty",
            "executor_ref": {"type": "local", "workspace_host_path": str(other), "workspace_backend_path": "/workspace"},
        }

    monkeypatch.setattr(tasks, "request_json_or", fake_request_json_or)
    monkeypatch.setattr(tasks, "request_drive_root", lambda _request: data)
    monkeypatch.setattr(tasks, "request_repo_dir", lambda _request: repo)

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(supervisor_ready_event=None)))
    response = asyncio.run(tasks.api_tasks_create(request))
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 400
    assert "mappings must cover workspace_root" in body["error"]


def test_api_task_rejects_reserved_executor_metadata_aliases(tmp_path, monkeypatch):
    from ouroboros.gateway import tasks

    repo = tmp_path / "repo"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    _init_repo(repo)
    _init_repo(workspace)
    data.mkdir()

    async def fake_request_json_or(_request, _default):
        return {
            "description": "x",
            "workspace_root": str(workspace),
            "workspace_mode": "external",
            "memory_mode": "empty",
            "metadata": {"workspace_executor": {"type": "local"}},
        }

    monkeypatch.setattr(tasks, "request_json_or", fake_request_json_or)
    monkeypatch.setattr(tasks, "request_drive_root", lambda _request: data)
    monkeypatch.setattr(tasks, "request_repo_dir", lambda _request: repo)

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(supervisor_ready_event=None)))
    response = asyncio.run(tasks.api_tasks_create(request))
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 400
    assert "metadata.executor_ref/workspace_executor is reserved" in body["error"]


def test_api_task_rejects_reserved_executor_metadata_ref(tmp_path, monkeypatch):
    from ouroboros.gateway import tasks

    repo = tmp_path / "repo"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    _init_repo(repo)
    _init_repo(workspace)
    data.mkdir()

    async def fake_request_json_or(_request, _default):
        return {
            "description": "x",
            "workspace_root": str(workspace),
            "workspace_mode": "external",
            "memory_mode": "empty",
            "metadata": {"executor_ref": {"type": "local"}},
        }

    monkeypatch.setattr(tasks, "request_json_or", fake_request_json_or)
    monkeypatch.setattr(tasks, "request_drive_root", lambda _request: data)
    monkeypatch.setattr(tasks, "request_repo_dir", lambda _request: repo)

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(supervisor_ready_event=None)))
    response = asyncio.run(tasks.api_tasks_create(request))
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 400
    assert "metadata.executor_ref/workspace_executor is reserved" in body["error"]


def test_api_task_rejects_local_network_none(tmp_path, monkeypatch):
    from ouroboros.gateway import tasks

    repo = tmp_path / "repo"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    _init_repo(repo)
    _init_repo(workspace)
    data.mkdir()

    async def fake_request_json_or(_request, _default):
        return {
            "description": "x",
            "workspace_root": str(workspace),
            "workspace_mode": "external",
            "memory_mode": "empty",
            "executor_ref": {
                "type": "local",
                "network": "none",
                "workspace_host_path": str(workspace),
                "workspace_backend_path": "/workspace",
            },
        }

    monkeypatch.setattr(tasks, "request_json_or", fake_request_json_or)
    monkeypatch.setattr(tasks, "request_drive_root", lambda _request: data)
    monkeypatch.setattr(tasks, "request_repo_dir", lambda _request: repo)

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(supervisor_ready_event=None)))
    response = asyncio.run(tasks.api_tasks_create(request))
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 400
    assert "local executor_ref cannot enforce network=none" in body["error"]


def test_api_task_rejects_malformed_executor_mapping_entry(tmp_path, monkeypatch):
    from ouroboros.gateway import tasks

    repo = tmp_path / "repo"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    _init_repo(repo)
    _init_repo(workspace)
    data.mkdir()

    async def fake_request_json_or(_request, _default):
        return {
            "description": "x",
            "workspace_root": str(workspace),
            "workspace_mode": "external",
            "memory_mode": "empty",
            "executor_ref": {
                "type": "local",
                "workspace_host_path": str(workspace),
                "workspace_backend_path": "/workspace",
                "path_mappings": [{"host_path": str(tmp_path / "missing_backend")}],
            },
        }

    monkeypatch.setattr(tasks, "request_json_or", fake_request_json_or)
    monkeypatch.setattr(tasks, "request_drive_root", lambda _request: data)
    monkeypatch.setattr(tasks, "request_repo_dir", lambda _request: repo)

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(supervisor_ready_event=None)))
    response = asyncio.run(tasks.api_tasks_create(request))
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 400
    assert "path_mappings entries require host_path and backend_path" in body["error"]


def test_api_task_rejects_malformed_executor_ref(tmp_path, monkeypatch):
    from ouroboros.gateway import tasks

    workspace = tmp_path / "workspace"
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    _init_repo(workspace)
    repo.mkdir()
    data.mkdir()

    async def fake_request_json_or(_request, _default):
        return {
            "description": "x",
            "workspace_root": str(workspace),
            "workspace_mode": "external",
            "memory_mode": "empty",
            "executor_ref": {"workspace_host_path": str(workspace), "workspace_backend_path": "/workspace"},
        }

    monkeypatch.setattr(tasks, "request_json_or", fake_request_json_or)
    monkeypatch.setattr(tasks, "request_drive_root", lambda _request: data)
    monkeypatch.setattr(tasks, "request_repo_dir", lambda _request: repo)

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(supervisor_ready_event=None)))
    response = asyncio.run(tasks.api_tasks_create(request))
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 400
    assert "executor_ref.type is required" in body["error"]
