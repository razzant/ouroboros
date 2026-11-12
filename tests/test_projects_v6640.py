"""Focused v6.64 Project lifecycle, canonical-dialogue, and sidebar guards."""

from __future__ import annotations
from tests._dialogue import advance_room_revision

import asyncio
import json
from types import SimpleNamespace

import pytest


def test_project_lifecycle_revision_binding_and_no_resurrection(tmp_path):
    from ouroboros.projects_registry import begin_project_deletion, bind_task_to_project, complete_project_deletion, create_project, get_project, get_reserved_project, list_projects, project_binding_for_task, reconcile_projects, reserved_project_chat_ids

    project = create_project(tmp_path, "alpha", name="Alpha")
    memory_dir = tmp_path / "projects" / "alpha"
    memory_dir.mkdir(parents=True)
    from ouroboros.project_dialogue import _text_sha256

    owner_text = "build the alpha project"
    first_ref = {
        "chat_id": 1,
        "client_message_id": "owner-1",
        "ts": "2026-07-13T00:00:00Z",
        "text_sha256": _text_sha256(owner_text),
    }
    bind_task_to_project(
        tmp_path, "task-1", "alpha", origin={"ref": first_ref, "text": owner_text},
    )
    # Binding and source identity are immutable even under a repeated conversion.
    bind_task_to_project(
        tmp_path,
        "task-1",
        "alpha",
        origin={"ref": {**first_ref, "client_message_id": "different"}, "text": owner_text},
    )
    binding = project_binding_for_task(tmp_path, "task-1")
    assert binding["source_ref"] == first_ref
    assert binding["source_text"] == owner_text

    assert advance_room_revision(tmp_path, project_id="alpha")["visible_revision"] == 1
    assert advance_room_revision(tmp_path, chat_id=project["chat_id"])["visible_revision"] == 2
    deleting = begin_project_deletion(tmp_path, "alpha")
    assert deleting["lifecycle"] == "deleting"
    assert deleting["routing_generation"] == 1
    assert get_project(tmp_path, "alpha") is None  # admission is already closed
    assert advance_room_revision(tmp_path, project_id="alpha") is None
    with pytest.raises(ValueError, match="cannot accept bindings"):
        bind_task_to_project(tmp_path, "task-after-fence", "alpha", origin={"absent": "system"})

    tombstone = complete_project_deletion(tmp_path, "alpha")
    assert tombstone["lifecycle"] == "tombstoned"
    assert list_projects(tmp_path) == []
    assert project["chat_id"] in reserved_project_chat_ids(tmp_path)
    assert project_binding_for_task(tmp_path, "task-1") is not None
    # The existing memory folder cannot resurrect the room at startup.
    assert reconcile_projects(tmp_path) == 0
    assert get_reserved_project(tmp_path, "alpha")["lifecycle"] == "tombstoned"
    with pytest.raises(ValueError, match="permanently reserved"):
        create_project(tmp_path, "alpha", name="Again")


def test_project_name_limit_is_enforced_before_gateway_side_effects(tmp_path):
    from ouroboros.gateway.projects import api_projects_create
    from ouroboros.projects_registry import PROJECT_NAME_MAX, create_project, get_project, update_project

    name = "x" * PROJECT_NAME_MAX
    create_project(tmp_path, "exact", name=name)
    assert update_project(tmp_path, "exact", name=name)["name"] == name
    with pytest.raises(ValueError, match="<= 80"):
        update_project(tmp_path, "exact", name=name + "x")

    async def _json():
        return {"id": "too-long", "name": name + "x"}

    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(drive_root=tmp_path)),
        json=_json,
    )
    response = asyncio.run(api_projects_create(request))
    assert response.status_code == 400
    assert get_project(tmp_path, "too-long") is None


def test_direct_routing_keeps_annotation_and_final_in_history_projection(tmp_path, monkeypatch):
    """Finalization→supervisor→chat-log keeps one durable answer beside the
    routing annotation and ordinary readable progress."""
    from ouroboros import agent_task_pipeline as pipeline
    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros.project_dialogue import append_chat_annotation
    from supervisor import message_bus
    from supervisor.events import _handle_send_message

    logs = tmp_path / "logs"
    logs.mkdir()
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        message_bus,
        "load_state",
        lambda: {"owner_id": 1, "session_id": "session-1"},
    )
    monkeypatch.setattr(message_bus, "_send_markdown", lambda *args, **kwargs: (True, ""))
    for name in (
        "_run_memory_fallback_draft",
        "_run_post_task_processing_async",
    ):
        monkeypatch.setattr(pipeline, name, lambda *args, **kwargs: None)

    message_bus.log_chat(
        "in",
        1,
        1,
        "Start the robot task",
        client_message_id="owner-route-1",
    )
    assert append_chat_annotation(
        tmp_path,
        "owner-route-1",
        action="promote_chat_to_task",
        target="robot01",
        status="scheduled",
    )

    pending_events = []
    pipeline.emit_task_results(
        env=SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path),
        memory=object(),
        llm=object(),
        pending_events=pending_events,
        task={
            "id": "decision-route-1",
            "type": "task",
            "chat_id": 1,
            "text": "Start the robot task",
            "_is_direct_chat": True,
            "_skip_post_task_synthesis": True,
        },
        text="The robot task was submitted as robot01.",
        usage={"rounds": 2, "cost": 0.01},
        llm_trace={"tool_calls": [{"tool": "promote_chat_to_task"}], "reasoning_notes": []},
        start_time=0.0,
        drive_logs=logs,
        ctx=SimpleNamespace(
            pending_restart_reason="",
            _typed_routing_action_emitted="promote_chat_to_task",
        ),
    )

    event_ctx = SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        send_with_budget=message_bus.send_with_budget,
        append_jsonl=lambda *args, **kwargs: None,
    )
    _handle_send_message({
        "type": "send_message",
        "chat_id": 1,
        "task_id": "decision-route-1",
        "text": "Submitting the robot task",
        "log_text": "Submitting the robot task",
        "format": "markdown",
        "is_progress": True,
    }, event_ctx)
    final_event = next(event for event in pending_events if event["type"] == "send_message")
    _handle_send_message(final_event, event_ctx)

    response = asyncio.run(
        make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={"chat_id": "1"}))
    )
    messages = json.loads(response.body.decode("utf-8"))["messages"]
    owner = next(message for message in messages if message.get("client_message_id") == "owner-route-1")
    assert owner["chat_annotation"] == {
        "action": "promote_chat_to_task",
        "target": "robot01",
        "status": "scheduled",
    }
    finals = [
        message for message in messages
        if message.get("text") == "The robot task was submitted as robot01."
    ]
    assert len(finals) == 1
    progress = next(message for message in messages if message.get("is_progress"))
    assert progress.get("is_progress") is True
    assert progress.get("text") or progress.get("content")
