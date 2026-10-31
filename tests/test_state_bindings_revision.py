"""``/api/state`` sends the task bindings map only when the client's revision no longer stands.

An open chat page reads the state every three seconds; on a long-lived install the bindings map
is most of each answer (273 KB of 399 KB). Old clients, the Telegram mini app and phones send no
``bindings_since`` and keep receiving the whole map.
"""

from __future__ import annotations

import asyncio
import json
import types
import urllib.parse

import pytest
from starlette.requests import Request

from ouroboros import projects_registry as registry
from ouroboros import usage_store, utils


@pytest.fixture
def read_state(tmp_path, monkeypatch):
    from ouroboros.gateway import state as gateway_state
    from supervisor import queue, state, workers

    root = tmp_path / "data"
    (root / "state").mkdir(parents=True)
    (root / "logs").mkdir(parents=True)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    usage_store.migrate_from_journal(root)
    monkeypatch.setattr(utils, "_RACY_NS", 0)  # files are settled as written (the tick rule has its own case)
    monkeypatch.setattr(gateway_state, "_runtime_repo_identity", lambda _st: (None, ""))
    monkeypatch.setattr(state, "TOTAL_BUDGET_LIMIT", 0.0)
    monkeypatch.setattr(state, "load_state", lambda: {})
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(workers, "PENDING", [])
    monkeypatch.setattr(workers, "RUNNING", {})
    monkeypatch.setattr(queue, "get_evolution_status_snapshot", lambda **_kwargs: {})
    registry.bind_task_to_project(root, "t1", "alpha", origin={"absent": "system"})

    def read(since=None) -> dict:
        query = urllib.parse.urlencode({"bindings_since": since}).encode() if since is not None else b""
        request = Request({"type": "http", "method": "GET", "path": "/api/state", "headers": [], "query_string": query,
                           "app": types.SimpleNamespace(state=types.SimpleNamespace(drive_root=root, app_start=0.0))})
        response = asyncio.run(gateway_state.api_state(request))
        assert response.status_code == 200
        return json.loads(response.body)

    return root, read


def test_a_standing_revision_omits_the_map_and_a_bind_sends_it_whole(read_state):
    root, read = read_state
    first = read()
    revision = first["task_bindings_revision"]
    assert set(first["task_bindings"]) == {"t1"} and revision and first["task_bindings_unchanged"] is False

    held = read(revision)
    assert held["task_bindings"] is None and held["task_bindings_unchanged"] is True
    assert held["task_bindings_revision"] == revision
    assert read()["task_bindings"] == first["task_bindings"]  # no parameter: the whole map, always

    registry.bind_task_to_project(root, "t2", "alpha", origin={"absent": "system"})
    moved = read(revision)
    assert set(moved["task_bindings"]) == {"t1", "t2"} and moved["task_bindings_unchanged"] is False
    assert moved["task_bindings_revision"] not in ("", revision)


def test_a_renamed_project_or_another_process_moves_the_revision(read_state, monkeypatch):
    root, read = read_state
    revision = read()["task_bindings_revision"]
    registry.update_project(root, "alpha", name="Alpha renamed")
    renamed = read(revision)
    assert renamed["task_bindings"]["t1"]["project_name"] == "Alpha renamed"

    monkeypatch.setattr(registry, "_PROCESS_STAMP", "restarted")  # the next serving process: its own code
    assert read(renamed["task_bindings_revision"])["task_bindings_unchanged"] is False


def test_a_failed_read_is_never_unchanged(read_state):
    root, read = read_state
    revision = read()["task_bindings_revision"]
    registry._bindings_path(root).write_text("{torn", encoding="utf-8")

    failed = read(revision)
    assert failed["task_bindings"] == {} and failed["task_bindings_revision"] == ""
    assert failed["task_bindings_unchanged"] is False
    assert read("")["task_bindings_unchanged"] is False


def test_a_store_changed_within_a_clock_tick_has_no_revision(read_state, monkeypatch):
    root, read = read_state
    monkeypatch.setattr(utils, "_RACY_NS", 2_000_000_000)
    registry.bind_task_to_project(root, "t2", "alpha", origin={"absent": "system"})

    fresh = read()
    assert fresh["task_bindings_revision"] == "" and set(fresh["task_bindings"]) == {"t1", "t2"}
