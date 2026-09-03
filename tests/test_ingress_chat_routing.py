"""Ingress addressing for headless/API tasks (owner decisions 2A/3A, sprint CLI display).

A task admitted through ``POST /api/tasks`` used to default to chat 0 — the
hidden partition no browser surface reads — even when the caller scoped it to a
project the owner already has open. These pin the one ingress rule: an explicit
id (0 included) is the caller's, a REGISTERED project homes the run into its
thread, and everything else stays hidden.
"""

from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros.contracts.chat_id_policy import HIDDEN_CHAT_ID, project_chat_id
from ouroboros.gateway.tasks import api_tasks_create
from ouroboros.projects_registry import create_project
from supervisor.log_addressing import ProjectThreadConflict, ingress_chat_id


def test_a_registered_project_run_has_exactly_one_destination(tmp_path):
    row = create_project(tmp_path, "proj_reg", name="Registered")
    assert ingress_chat_id(None, tmp_path, "proj_reg") == row["chat_id"]
    assert ingress_chat_id(row["chat_id"], tmp_path, "proj_reg") == row["chat_id"]
    # Anywhere else is refused — the hidden partition included. A run addressed
    # away from its room is the shape that puts a card in Main whose project
    # holds none of its work, which is the defect this sprint exists to remove.
    for elsewhere in (1, HIDDEN_CHAT_ID, 999):
        with pytest.raises(ProjectThreadConflict):
            ingress_chat_id(elsewhere, tmp_path, "proj_reg")


def test_without_a_room_the_only_address_is_the_hidden_partition(tmp_path):
    """Owner decision 3A, made true by construction rather than by convention.

    Externally launched work lives in its project's thread or nowhere visible.
    A caller may still ASK for the hidden partition — that is what running quietly
    means — but not for a conversation of its own, because Main accepts any
    unknown positive id and the run would surface there as ordinary dialogue.
    """
    assert ingress_chat_id(None, tmp_path, "") == HIDDEN_CHAT_ID
    assert ingress_chat_id(0, tmp_path, "") == HIDDEN_CHAT_ID
    assert ingress_chat_id(None, tmp_path, "proj_never_registered") == HIDDEN_CHAT_ID
    for elsewhere in (1, 7, 222594327):
        with pytest.raises(ProjectThreadConflict):
            ingress_chat_id(elsewhere, tmp_path, "")


def test_an_inactive_project_is_left_to_the_queues_own_lifecycle_fence(tmp_path):
    """One refusal per question, at the layer that owns it.

    A deleting project keeps its reserved chat, and admission into it is refused
    by the queue's lifecycle fence with its typed reason. Answering earlier with
    a different error would change an established response shape for no gain.
    """
    from ouroboros.projects_registry import begin_project_deletion, create_project

    row = create_project(tmp_path, "proj_going", name="Going")
    begin_project_deletion(tmp_path, "proj_going")
    assert ingress_chat_id(None, tmp_path, "proj_going") == HIDDEN_CHAT_ID
    assert ingress_chat_id(row["chat_id"], tmp_path, "proj_going") == row["chat_id"]


def test_a_chat_id_that_is_not_a_whole_number_is_refused(tmp_path):
    # int(True) is 1 and int(1.9) is 1; neither is a chat the caller named.
    for bad in (True, False, 1.9, "nope"):
        with pytest.raises((TypeError, ValueError)):
            ingress_chat_id(bad, tmp_path, "")


def test_registered_project_homes_the_run_into_its_thread(tmp_path):
    row = create_project(tmp_path, "proj_reg", name="Registered")
    assert ingress_chat_id(None, tmp_path, "proj_reg") == row["chat_id"] == project_chat_id("proj_reg")


def test_unregistered_or_derived_project_stays_in_the_hidden_partition(tmp_path):
    # A workspace-derived proj_<hash> has no registry row and therefore no
    # thread; stamping its derived chat id would leak the run into Main, where
    # an unknown positive id is accepted as ordinary conversation.
    assert ingress_chat_id(None, tmp_path, "proj_deadbeef1234") == HIDDEN_CHAT_ID
    assert ingress_chat_id(None, tmp_path, "") == HIDDEN_CHAT_ID


def test_malformed_explicit_chat_id_still_raises_for_the_caller_s_400(tmp_path):
    with pytest.raises((TypeError, ValueError)):
        ingress_chat_id("not-an-int", tmp_path, "")


def _app(data, repo):
    app = Starlette(routes=[Route("/api/tasks", endpoint=api_tasks_create, methods=["POST"])])
    app.state.drive_root = data
    app.state.repo_dir = repo
    return app


@pytest.fixture()
def admission(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    data = tmp_path / "data"
    (data / "memory").mkdir(parents=True)
    (data / "memory" / "identity.md").write_text("seed identity", encoding="utf-8")
    from supervisor import workers

    monkeypatch.setattr(workers, "WORKERS", {0: SimpleNamespace()})
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "")
    captured = []
    monkeypatch.setattr("supervisor.queue.enqueue_task", lambda task: captured.append(dict(task)) or task)
    monkeypatch.setattr("supervisor.queue.persist_queue_snapshot", lambda reason="": True)
    monkeypatch.setattr("ouroboros.workspace_admission.bootstrap_process_path", lambda: [])
    return data, repo, captured


def test_api_task_for_a_registered_project_is_admitted_into_the_project_thread(admission):
    data, repo, captured = admission
    row = create_project(data, "proj_room", name="Room")
    response = TestClient(_app(data, repo)).post(
        "/api/tasks", json={"description": "audit the report", "project_id": "proj_room"},
    )
    assert response.status_code == 200, response.text
    assert captured and captured[0]["chat_id"] == row["chat_id"]
    # The durable scheduled record carries the address too, so a later reader
    # (detail endpoint, reaper, crash insurance) never has to re-derive it.
    from ouroboros.task_results import load_task_result

    stored = load_task_result(data, response.json()["task_id"]) or {}
    assert stored.get("chat_id") == row["chat_id"]


def test_api_task_without_a_project_stays_hidden_and_explicit_zero_is_honoured(admission):
    data, repo, captured = admission
    client = TestClient(_app(data, repo))
    assert client.post("/api/tasks", json={"description": "script run"}).status_code == 200
    assert captured[-1]["chat_id"] == HIDDEN_CHAT_ID
    row = create_project(data, "proj_room2", name="Room")
    assert row["chat_id"] > 0
    # …but a project-scoped run cannot be addressed anywhere but its room.
    for elsewhere in (0, 1):
        conflict = client.post(
            "/api/tasks",
            json={"description": "x", "project_id": "proj_room2", "chat_id": elsewhere},
        )
        assert conflict.status_code == 400, elsewhere
        assert "project thread" in conflict.text


def test_api_task_with_a_malformed_chat_id_is_still_a_typed_400(admission):
    data, repo, _ = admission
    response = TestClient(_app(data, repo)).post(
        "/api/tasks", json={"description": "x", "chat_id": "nope"},
    )
    assert response.status_code == 400
    assert "integers" in response.text


def test_derived_project_id_is_scoped_but_never_announced_in_main(tmp_path, monkeypatch):
    """Owner decision 3A: no room, no Main row.

    A ``--workspace`` run derives ``proj_<hash>``; that id is project-SCOPED for
    lease and memory, but it has no registry row and therefore no thread. The
    Main completion line offers "Open Project", so announcing a derived id would
    hand the owner a door into an empty duplicate of Main.

    The assertion is the GATE, not the delivery: the outbox needs a live
    supervisor event bus, which is process-global and not this test's subject.
    """
    from ouroboros import project_dialogue
    from ouroboros.projects_registry import create_project, task_presentation_snapshot

    enqueued = []
    monkeypatch.setattr(
        "supervisor.terminal_delivery.enqueue_terminal_delivery",
        lambda drive_root, event: enqueued.append(event) or True,
    )

    task = {"id": "t1", "project_id": "proj_deadbeef1234", "description": "run"}
    result = {"status": "completed", "project_id": "proj_deadbeef1234", "result": "done"}
    assert task_presentation_snapshot(tmp_path, "t1", task=task, result=result)["project_routable"] is False
    assert project_dialogue.enqueue_project_completion_summary(
        tmp_path, {}, "t1", task, result, {"status": "completed"},
    ) is False
    assert enqueued == [], "a derived project id must not reach the Main outbox at all"

    row = create_project(tmp_path, "proj_real", name="Real")
    assert row["chat_id"] > 0
    task2 = {"id": "t2", "project_id": "proj_real", "description": "run", "chat_id": row["chat_id"]}
    result2 = {"status": "completed", "project_id": "proj_real", "result": "done",
               "chat_id": row["chat_id"]}
    assert task_presentation_snapshot(tmp_path, "t2", task=task2, result=result2)["project_routable"] is True
    assert project_dialogue.enqueue_project_completion_summary(
        tmp_path, {}, "t2", task2, result2, {"status": "completed"},
    ) is True
    # Main gets exactly one row, and it points at the room that exists.
    assert len(enqueued) == 1
    assert enqueued[0]["chat_id"] == 1
    assert enqueued[0]["system_type"] == "project_completion_summary"


def test_main_is_told_a_project_finished_only_when_the_work_went_there(tmp_path, monkeypatch):
    """Main owes two lifecycle rows for a project, and only for a real one.

    A run BOUND to a project mid-flight has its rows re-homed into that room, so
    Main must be told it finished — otherwise the start row it already received
    hangs unanswered. A run merely SCOPED to an id that somebody registered later
    never entered that room, so announcing it would hand the owner the empty
    "Open Project" this sprint exists to remove.
    """
    from ouroboros import project_dialogue
    from ouroboros.projects_registry import (
        begin_project_deletion, bind_task_to_project, create_project,
    )

    enqueued = []
    monkeypatch.setattr(
        "supervisor.terminal_delivery.enqueue_terminal_delivery",
        lambda drive_root, event: enqueued.append(event) or True,
    )
    row = create_project(tmp_path, "proj_midflight", name="Mid flight")
    task = {"id": "m1", "project_id": "proj_midflight", "description": "run", "chat_id": 0}
    result = {"status": "completed", "project_id": "proj_midflight", "chat_id": 0, "result": "done"}

    # Registered late, never bound: the room holds none of this run's rows.
    assert project_dialogue.enqueue_project_completion_summary(
        tmp_path, {}, "m1", task, result, {"status": "completed"},
    ) is False
    assert enqueued == []

    # Bound mid-flight: the binding re-homes its rows, so Main is told.
    bind_task_to_project(tmp_path, "m1", "proj_midflight", origin={"absent": "system"})
    assert project_dialogue.enqueue_project_completion_summary(
        tmp_path, {}, "m1", task, result, {"status": "completed"},
    ) is True
    assert len(enqueued) == 1 and enqueued[0]["chat_id"] == 1

    # Addressed there at admission is the ordinary case.
    homed = {**task, "id": "m2", "chat_id": row["chat_id"]}
    assert project_dialogue.enqueue_project_completion_summary(
        tmp_path, {}, "m2", homed, {**result, "chat_id": row["chat_id"]}, {"status": "completed"},
    ) is True
    assert len(enqueued) == 2

    # A project on its way out has no room to open.
    begin_project_deletion(tmp_path, "proj_midflight")
    assert project_dialogue.enqueue_project_completion_summary(
        tmp_path, {}, "m3", homed, {**result, "chat_id": row["chat_id"]}, {"status": "completed"},
    ) is False
    assert len(enqueued) == 2
