"""The admitted Project address and nested waiting facts reach real gateways."""
import json
from types import SimpleNamespace

import pytest
from starlette.requests import Request

from ouroboros import projects_registry as registry
from ouroboros.gateway.tasks import _create_task_from_body, api_task_get, api_tasks_list
from ouroboros.gateway.state import _chat_activities_snapshot_safe
from ouroboros.task_results import load_task_result
from supervisor import queue, workers
from tests.test_swarm_host_admission import host as host_fixture
from tests.test_project_authority_outages import _uncertain_child
from tests._store_shared import result_signature
from tests.test_project_hold_recovery import accepted, worker

host = host_fixture
pytestmark = pytest.mark.serial


def request_for(host, task_id="", query=b""):
    return Request({"type": "http", "query_string": query, "path_params": {"task_id": task_id},
                    "app": SimpleNamespace(state=SimpleNamespace(drive_root=host.root, repo_dir=workers.REPO_DIR))})


@pytest.mark.parametrize("explicit", [False, True])
def test_project_api_uses_captured_address_without_a_second_authority_read(host, monkeypatch, explicit):
    project = registry.create_project(host.root, "target", name="Target")
    reads = []

    def unavailable(*args, **kwargs):
        reads.append(args)
        raise OSError("secondary lookup is unavailable")

    monkeypatch.setattr(registry, "get_reserved_project", unavailable)
    body = {"task_id": "addressed", "description": "Project work", "project_id": "target", "memory_mode": "empty"}
    if explicit:
        body["chat_id"] = project["chat_id"]
    response = _create_task_from_body(request_for(host), body)
    assert response.status_code == 200, response.body
    [queued] = host.pending
    stored = load_task_result(host.root, "addressed", strict=True)
    assert queued["chat_id"] == stored["chat_id"] == project["chat_id"]
    assert queued["_project_admission"]["project"]["chat_id"] == project["chat_id"]
    assert not reads and not host.attempts
    from supervisor.log_addressing import address_task_event
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    assert [task["id"] for task in sent] == ["addressed"]
    assert address_task_event(host.running, host.root, {"type": "llm_usage", "task_id": "addressed"})["chat_id"] == project["chat_id"]


@pytest.mark.parametrize("address", [0, 1, 999])
def test_project_api_retains_explicit_conflict_refusal(host, address):
    registry.create_project(host.root, "target")
    response = _create_task_from_body(request_for(host), {"task_id": "conflict", "description": "Project work",
        "project_id": "target", "chat_id": address, "memory_mode": "empty"})
    assert response.status_code == 400 and b"conflicts" in response.body
    assert not host.pending and load_task_result(host.root, "conflict") is None


@pytest.mark.parametrize("explicit", [False, True])
def test_project_api_inactive_room_keeps_the_admission_owners_typed_refusal(host, explicit):
    project = registry.create_project(host.root, "target")
    registry.begin_project_deletion(host.root, "target")
    body = {"task_id": "inactive", "description": "Project work", "project_id": "target", "memory_mode": "empty"}
    if explicit:
        body["chat_id"] = project["chat_id"]
    response = _create_task_from_body(request_for(host), body)
    assert response.status_code == 409, response.body
    assert json.loads(response.body)["admission"]["reason_code"] == "project_routing_fence"
    assert not host.pending


@pytest.mark.parametrize("availability", ["available", "disabled", "unsaved"])
@pytest.mark.parametrize("project_room", [False, True])
@pytest.mark.parametrize("producer", ["api", "promotion"])
def test_root_receipt_records_the_effective_admitted_room(host, monkeypatch, availability, project_room, producer):
    from ouroboros import config
    from ouroboros.composer_cognition import update_project_cognition
    from ouroboros.project_dialogue import build_owner_message_ref
    from supervisor.events_project_routing import _handle_promote_chat_to_task
    from tests.test_subagents_mode_messages import mode_settings

    values = mode_settings("off" if project_room else "swarm", availability)
    monkeypatch.setattr(config, "load_settings", lambda: dict(values))
    room = registry.create_project(host.root, "target") if project_room else {"id": "", "chat_id": 1}
    if project_room:
        update_project_cognition(host.root, room["id"], subagents_mode="swarm")
    if producer == "api":
        response = _create_task_from_body(request_for(host), {"task_id": "mode", "description": "Owner work",
            "project_id": room["id"], "memory_mode": "empty",
            "metadata": {"subagents_mode": "off" if availability == "available" else "swarm"}})
        assert response.status_code == 200, response.body
    else:
        result = _handle_promote_chat_to_task({"task_id": "mode", "routing_token": "mode-token",
            "objective": "Owner work", "project_id": room["id"], "chat_id": room["chat_id"],
            "source_ref": build_owner_message_ref(chat_id=room["chat_id"], client_message_id="mode-origin",
                ts="2026-10-27T00:00:00Z", text="Owner work"), "source_text": "Owner work"}, host.ctx)
        assert result["status"] == "scheduled", result
    [queued] = host.pending
    stored = load_task_result(host.root, "mode", strict=True)
    assert queued["metadata"]["subagents_mode"] == stored["metadata"]["subagents_mode"] == (
        "swarm" if availability == "available" else "off")


@pytest.mark.parametrize("accepted,later", [("swarm", "off"), ("off", "swarm")])
@pytest.mark.parametrize("restoring", [False, True])
def test_requeued_root_keeps_its_admission_fact_after_a_room_switch(host, monkeypatch, accepted, later, restoring):
    from ouroboros import config
    from tests.test_subagents_mode_messages import mode_settings

    values = mode_settings(accepted, "available")
    monkeypatch.setattr(config, "load_settings", lambda: dict(values))
    response = _create_task_from_body(request_for(host), {"task_id": "mode", "description": "Owner work"})
    assert response.status_code == 200, response.body
    admitted = host.pending.pop()
    assert admitted["metadata"]["subagents_mode"] == accepted
    values["OUROBOROS_SUBAGENTS_MODE"] = later
    requeued = queue.enqueue_task(admitted, restoring_snapshot=restoring)
    assert not requeued.get("_admission_blocked")
    assert requeued["metadata"]["subagents_mode"] == accepted


def test_queue_gateway_preserves_child_wait_and_recovery_without_root_census_inflation(host, tmp_path, monkeypatch):
    accepted(host, tmp_path, tid="sibling")
    child, original, unreadable = _uncertain_child(host, monkeypatch)
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    assert [task["id"] for task in sent] == ["sibling"]
    assert [row["activity_id"] for row in _chat_activities_snapshot_safe(host.root)] == ["sibling"]
    # Its unreadable result cannot deliver detail, but accepted queue custody can.
    assert host.run_async(api_task_get(request_for(host, "held"))).status_code in (404, 503)
    before = queue.QUEUE_SNAPSHOT_PATH.read_bytes()
    response = host.run_async(api_tasks_list(request_for(host, query=b"queue_only=1")))
    assert response.status_code == 200
    payload = json.loads(response.body)
    held = next(row["task"] for row in payload["queue"]["pending"] if row["id"] == child["id"])
    assert held["project_admission_hold"]["label"] == "Waiting for Project verification"
    assert "unreadable" in held["project_admission_hold"]["detail"]
    assert queue.QUEUE_SNAPSHOT_PATH.read_bytes() == before  # passive projection
    host.running.clear()
    workers.WORKERS[0].busy_task_id = None
    unreadable["on"] = False  # the committed receipt is readable again
    assert result_signature(host.root, "held") == original
    workers.assign_tasks()
    workers.assign_tasks()
    assert [task["id"] for task in sent] == ["sibling", "held"]
    recovered = json.loads(host.run_async(api_tasks_list(request_for(host, query=b"queue_only=1"))).body)
    row = next(row["task"] for row in recovered["queue"]["running"] if row["id"] == child["id"])
    assert row["project_admission_hold"] == {}
    assert not host.attempts
