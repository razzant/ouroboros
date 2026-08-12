"""HTTP contract for the thread lifecycle routes.

Three OWNER surfaces, gateway-only and deliberately not LLM-callable tools:

    POST /api/projects/{project_id}/threads
    POST /api/projects/{project_id}/threads/{thread_id}/update
    POST /api/projects/{project_id}/threads/{thread_id}/fork

Every one of them broadcasts `projects_changed` carrying the affected thread's
chat id, so an open client learns the new chat id BEFORE any live frame for it
can arrive (otherwise those frames leak into Main). Hermetic: tmp drive root,
no supervisor.
"""

from __future__ import annotations

import pathlib

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros.gateway.projects import (
    api_project_thread_create,
    api_project_thread_fork,
    api_project_thread_update,
)
from ouroboros.projects_registry import begin_project_deletion, create_project


@pytest.fixture()
def broadcasts(monkeypatch):
    seen: list = []
    import ouroboros.gateway.projects as gateway_projects

    monkeypatch.setattr(
        gateway_projects,
        "_broadcast_projects_changed",
        lambda project_id, chat_id: seen.append((project_id, chat_id)),
    )
    return seen


def _client(drive_root: pathlib.Path) -> TestClient:
    app = Starlette(routes=[
        Route("/api/projects/{project_id}/threads", api_project_thread_create, methods=["POST"]),
        Route(
            "/api/projects/{project_id}/threads/{thread_id}/update",
            api_project_thread_update, methods=["POST"],
        ),
        Route(
            "/api/projects/{project_id}/threads/{thread_id}/fork",
            api_project_thread_fork, methods=["POST"],
        ),
    ])
    app.state.drive_root = drive_root
    app.state.repo_dir = drive_root
    return TestClient(app)


def test_create_thread_returns_the_envelope_and_broadcasts_its_chat_id(tmp_path, broadcasts):
    create_project(tmp_path, "racer", name="Cyber Racer")
    client = _client(tmp_path)

    response = client.post("/api/projects/racer/threads", json={"name": "Tuning"})

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"project_id", "thread"}
    assert body["project_id"] == "racer"
    thread = body["thread"]
    assert thread["id"] >= 1 and thread["name"] == "Tuning"
    # R7: the new chat id rides projects_changed, synchronously.
    assert broadcasts == [("racer", thread["chat_id"])]


def test_create_thread_without_a_name_gets_a_neutral_default(tmp_path, broadcasts):
    create_project(tmp_path, "racer")
    client = _client(tmp_path)

    thread = client.post("/api/projects/racer/threads", json={}).json()["thread"]

    assert thread["name"] == "New thread"


def test_thread_routes_refuse_unknown_project_and_thread(tmp_path, broadcasts):
    create_project(tmp_path, "racer")
    client = _client(tmp_path)

    assert client.post("/api/projects/ghost/threads", json={}).status_code == 404
    assert client.post("/api/projects/ghost/threads/1/update", json={"name": "x"}).status_code == 404
    assert client.post("/api/projects/racer/threads/9/update", json={"name": "x"}).status_code == 404
    assert client.post("/api/projects/racer/threads/9/fork", json={}).status_code == 404
    assert broadcasts == []


def test_rename_validates_the_name(tmp_path, broadcasts):
    from ouroboros.projects_registry import THREAD_NAME_MAX

    create_project(tmp_path, "racer")
    client = _client(tmp_path)
    thread = client.post("/api/projects/racer/threads", json={}).json()["thread"]

    assert client.post(
        f"/api/projects/racer/threads/{thread['id']}/update", json={},
    ).status_code == 400
    too_long = client.post(
        f"/api/projects/racer/threads/{thread['id']}/update",
        json={"name": "x" * (THREAD_NAME_MAX + 1)},
    )
    assert too_long.status_code == 400
    ok = client.post(
        f"/api/projects/racer/threads/{thread['id']}/update", json={"name": "Tuned"},
    )
    assert ok.json()["thread"]["name"] == "Tuned"


def test_renaming_thread_zero_renames_the_project(tmp_path, broadcasts):
    """Thread 0 IS the project row, so its name has ONE home."""
    from ouroboros.projects_registry import get_project

    create_project(tmp_path, "racer", name="Cyber Racer")
    client = _client(tmp_path)

    response = client.post("/api/projects/racer/threads/0/update", json={"name": "Racer II"})

    assert response.json()["thread"]["id"] == 0
    assert get_project(tmp_path, "racer")["name"] == "Racer II"


def test_fork_copies_no_rows_and_auto_names_the_copy(tmp_path, broadcasts):
    create_project(tmp_path, "racer")
    client = _client(tmp_path)
    source = client.post("/api/projects/racer/threads", json={"name": "Tuning"}).json()["thread"]

    fork = client.post(f"/api/projects/racer/threads/{source['id']}/fork", json={}).json()["thread"]

    assert fork["name"] == "Copy of Tuning"
    assert fork["fork_of_chat_id"] == source["chat_id"]
    assert fork["fork_before_ts"]
    assert fork["chat_id"] != source["chat_id"]
    assert broadcasts[-1] == ("racer", fork["chat_id"])


def test_the_main_chat_is_not_forkable(tmp_path, broadcasts):
    """A3: the Main chat cannot be forked. It is not a project thread, so no
    route reaches it — every thread route is scoped to a project id, and the
    registry answers only for threads of that project."""
    from ouroboros.contracts.chat_id_policy import WEB_UI_CHAT_ID
    from ouroboros.projects_registry import resolve_chat_binding

    create_project(tmp_path, "racer")
    client = _client(tmp_path)

    # The Main chat belongs to no project and is no project's thread, so the
    # fork surface — which is reachable ONLY as project + thread id — has no
    # spelling that names it.
    assert resolve_chat_binding(tmp_path, WEB_UI_CHAT_ID) == {}
    assert client.post("/api/projects//threads/0/fork", json={}).status_code in (404, 405)
    # Thread #0 of a PROJECT is a legitimate fork source (it is a project
    # thread, not the Main chat).
    assert client.post("/api/projects/racer/threads/0/fork", json={}).status_code == 200


def test_threads_of_a_fenced_project_are_refused(tmp_path, broadcasts):
    create_project(tmp_path, "racer")
    begin_project_deletion(tmp_path, "racer")
    client = _client(tmp_path)

    # A fenced project is no longer "active", so it is not even addressable.
    assert client.post("/api/projects/racer/threads", json={"name": "x"}).status_code == 404


def test_a_project_that_starts_DELETING_mid_request_answers_409_not_500(tmp_path, monkeypatch):
    """T3R-17. A project on its way out refusing thread changes is a PRECONDITION
    the owner can read — the project is being deleted — not a crash.

    The routes look the project up first, so this is reachable exactly by the
    RACE the lookup cannot close: deletion starts between `get_project` and the
    registry write. `_active_project_row` raised a bare `ValueError` there, which
    reached `json_exception` as a 500 with no reason a UI could branch on — the
    same fact, rendered as "something broke". It is now the module's own typed
    lifecycle refusal, which every other thread route already answers as a 409.
    """
    import ouroboros.projects_registry as registry
    from ouroboros.gateway import projects as gateway_projects
    from ouroboros.projects_registry import begin_project_deletion, create_project, get_project

    create_project(tmp_path, "racer", name="Cyber Racer")
    alive = get_project(tmp_path, "racer")
    begin_project_deletion(tmp_path, "racer")
    # The lookup saw the project alive; the registry write finds it deleting.
    monkeypatch.setattr(registry, "get_project", lambda *_a, **_k: alive)
    app = Starlette(routes=[
        Route(
            "/api/projects/{project_id}/threads",
            gateway_projects.api_project_thread_create, methods=["POST"],
        ),
    ])
    app.state.drive_root = tmp_path

    with TestClient(app) as client:
        response = client.post("/api/projects/racer/threads", json={"name": "Side quest"})

    assert response.status_code == 409, response.text
    body = response.json()
    assert body["reason"] == "project_not_active"
    assert "deleting" in body["message"]


def test_the_typed_project_refusal_is_still_a_ValueError_for_older_callers(tmp_path):
    """It is raised where a plain `ValueError` was, so nothing that already
    caught one starts leaking an exception."""
    from ouroboros.project_threads_registry import ThreadLifecycleError
    from ouroboros.projects_registry import begin_project_deletion, create_project, create_thread

    create_project(tmp_path, "racer", name="Cyber Racer")
    begin_project_deletion(tmp_path, "racer")

    with pytest.raises(ValueError) as caught:
        create_thread(tmp_path, "racer", name="Side quest")

    assert isinstance(caught.value, ThreadLifecycleError)
    assert caught.value.reason == "project_not_active"
