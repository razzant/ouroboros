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


def test_the_revision_moves_exactly_when_a_readers_map_would_differ(read_state, monkeypatch):
    root, read = read_state
    revision = read()["task_bindings_revision"]
    # A Project touched by its own work rewrites the registry and no reader's map: the held map still stands.
    registry.touch_project(root, "alpha")
    touched = read(revision)
    assert touched["task_bindings"] is None and touched["task_bindings_unchanged"] is True

    registry.update_project(root, "alpha", name="Alpha renamed")  # the map names the Project: it differs now
    renamed = read(revision)
    assert renamed["task_bindings"]["t1"]["project_name"] == "Alpha renamed"
    assert renamed["task_bindings_revision"] not in ("", revision)

    monkeypatch.setattr(registry, "_PROCESS_STAMP", "restarted")  # the next serving process sends the same bytes
    assert read(renamed["task_bindings_revision"])["task_bindings_unchanged"] is True


def test_a_failed_read_is_never_unchanged(read_state):
    root, read = read_state
    revision = read()["task_bindings_revision"]
    registry._bindings_path(root).write_text("{torn", encoding="utf-8")

    failed = read(revision)
    assert failed["task_bindings"] == {} and failed["task_bindings_revision"] == ""
    assert failed["task_bindings_unchanged"] is False
    assert read("")["task_bindings_unchanged"] is False


def test_a_registry_that_vouches_for_no_room_sends_the_map_without_a_revision(read_state):
    """The map's Project names come from the registry: while it vouches for no room (lost beside its commit witness,
    or only invalid rows) the map goes without a revision, so a reader keeps the one it holds instead of the
    Project ids standing in for names. A readable registry gives back the revision it had."""
    root, read = read_state
    registry.create_project(root, "beta", name="Beta")
    registry.bind_task_to_project(root, "t3", "beta", origin={"absent": "system"})
    healthy = read()
    assert healthy["task_bindings"]["t3"]["project_name"] == "Beta" and healthy["task_bindings_revision"]
    path = root / "state" / "projects.json"
    intact = path.read_bytes()
    for unknown in ("lost beside its commit witness", "only an invalid row"):
        path.unlink(missing_ok=True)
        if unknown == "only an invalid row":
            path.write_text(json.dumps({**json.loads(intact), "projects": [{"chat_id": 7}]}), encoding="utf-8")
        body = read(healthy["task_bindings_revision"])
        assert body["task_bindings_revision"] == "" and body["task_bindings_unchanged"] is False, unknown
        assert body["project_chat_ids"] is None and body["project_listable_count"] is None, unknown
    path.write_bytes(intact)
    assert read()["task_bindings_revision"] == healthy["task_bindings_revision"]


def test_the_registry_witness_alone_moves_the_publication_stamp(read_state):
    """A registry lost beside its commit witness is an unknown room set; with the witness gone too it is a fresh
    install's empty one. The publication's stamp follows the witness, so a held answer never outlives it."""
    root, read = read_state
    (root / "state" / "projects.json").unlink()
    lost = registry.bindings_revision(root)
    assert lost and registry.bindings_revision(root) == lost and read()["project_chat_ids"] is None
    (root / "state" / "projects.json.committed").touch()  # as reconcile does each tick: whether it exists decides
    assert registry.bindings_revision(root) == lost, "a touched witness is no new production"
    (root / "state" / "projects.json.committed").unlink()
    assert registry.bindings_revision(root) not in ("", lost), "the witness's removal is a new production"
    assert read()["project_chat_ids"] == []


def test_a_store_changed_within_a_clock_tick_names_the_map_it_sent(read_state, monkeypatch):
    root, read = read_state
    revision = read()["task_bindings_revision"]
    monkeypatch.setattr(utils, "_RACY_NS", 2_000_000_000)
    registry.bind_task_to_project(root, "t2", "alpha", origin={"absent": "system"})

    # No file stamp proves these bytes yet; the map's own digest does, so the reader holds it from this answer on.
    fresh = read(revision)
    assert set(fresh["task_bindings"]) == {"t1", "t2"} and fresh["task_bindings_revision"] not in ("", revision)
    assert read(fresh["task_bindings_revision"])["task_bindings_unchanged"] is True
