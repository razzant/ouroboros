"""Focused v6.64 Project lifecycle, canonical-dialogue, and sidebar guards."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest


def test_project_lifecycle_revision_binding_and_no_resurrection(tmp_path):
    from ouroboros.projects_registry import (
        begin_project_deletion,
        bind_task_to_project,
        complete_project_deletion,
        create_project,
        get_project,
        get_reserved_project,
        increment_project_visible_revision,
        list_projects,
        project_binding_for_task,
        reconcile_projects,
        reserved_project_chat_ids,
    )

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

    assert increment_project_visible_revision(tmp_path, project_id="alpha")["visible_revision"] == 1
    assert increment_project_visible_revision(tmp_path, chat_id=project["chat_id"])["visible_revision"] == 2
    deleting = begin_project_deletion(tmp_path, "alpha")
    assert deleting["lifecycle"] == "deleting"
    assert deleting["routing_generation"] == 1
    assert get_project(tmp_path, "alpha") is None  # admission is already closed
    assert increment_project_visible_revision(tmp_path, project_id="alpha") is None
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


def test_chat_annotations_are_compact_and_torn_tail_tolerant(tmp_path):
    from ouroboros.project_dialogue import append_chat_annotation, latest_chat_annotations

    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "chat.jsonl").write_text(
        json.dumps({
            "ts": "2026-07-13T00:00:00Z",
            "direction": "in",
            "chat_id": 1,
            "client_message_id": "owner-1",
            "text": "continue",
        }) + "\n",
        encoding="utf-8",
    )
    assert append_chat_annotation(
        tmp_path,
        "owner-1",
        action="routed",
        target="project:alpha",
        status="pending",
    )
    assert append_chat_annotation(
        tmp_path,
        "owner-1",
        action="routed",
        target="project:alpha",
        status="delivered",
    )
    path = tmp_path / "logs" / "chat_annotations.jsonl"
    with path.open("ab") as handle:
        handle.write(b'{"type":"chat_annotation"')

    latest = latest_chat_annotations(tmp_path)
    assert latest["owner-1"]["status"] == "delivered"
    assert set(latest["owner-1"]) == {
        "ts", "type", "client_message_id", "action", "target", "status",
    }

    from ouroboros.gateway.history import make_chat_history_endpoint

    endpoint = make_chat_history_endpoint(tmp_path)
    response = asyncio.run(endpoint(SimpleNamespace(query_params={"chat_id": "1"})))
    messages = json.loads(response.body.decode("utf-8"))["messages"]
    owner = next(message for message in messages if message.get("client_message_id") == "owner-1")
    assert owner["chat_annotation"] == {
        "action": "routed",
        "target": "project:alpha",
        "status": "delivered",
    }


def test_chat_annotation_compaction_drops_rows_after_chat_retention(tmp_path):
    from ouroboros.project_dialogue import append_chat_annotation, latest_chat_annotations

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").write_text("", encoding="utf-8")
    stale = {
        "ts": "2026-07-13T00:00:00Z",
        "type": "chat_annotation",
        "client_message_id": "expired",
        "action": "routed",
        "target": "x" * 800_000,
        "status": "delivered",
    }
    (logs / "chat_annotations.jsonl").write_text(
        json.dumps(stale) + "\n", encoding="utf-8",
    )

    assert append_chat_annotation(
        tmp_path, "also-expired", action="routed", status="delivered",
    )
    assert latest_chat_annotations(tmp_path) == {}
    assert (logs / "chat_annotations.jsonl").read_text(encoding="utf-8") == ""


def test_project_sidebar_and_menu_static_contracts():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    html = (root / "web" / "index.html").read_text(encoding="utf-8")
    app = (root / "web" / "app.js").read_text(encoding="utf-8")
    menu = (root / "web" / "modules" / "project_create.js").read_text(encoding="utf-8")
    chat = (root / "web" / "modules" / "chat.js").read_text(encoding="utf-8")
    css = (root / "web" / "style.css").read_text(encoding="utf-8")

    assert 'class="nav-projects-header"' in html
    assert '<svg class="nav-chevron"' in html
    assert 'aria-label="New project"><svg' in html
    assert ".nav-projects-header" in css
    assert ".nav-projects-header .nav-row-meta:not(:empty)" in css
    projects_css = css[css.index(".nav-projects-header"):css.index("/* New Project dialog */")]
    assert "position: absolute" not in projects_css

    assert "project_seen_revision" in app
    assert "acknowledgeProjectAfterPaint" in app
    assert "inst.refreshHistory?.({ revision })" in app
    assert "paint?.painted" in app
    assert "await markProjectViewed(project.id, revision)" in app
    assert "async function markProjectViewed" in app
    assert "await fetchJson('/api/ui/preferences'" in app
    assert "item.append(btn, trailing)" in app
    assert "hideProjectFromSidebar" not in app
    assert "project_last_viewed" not in app
    assert "project_hidden" not in app

    assert "chatAnnotation: msg.chat_annotation || null" in chat
    annotation_handler = chat[
        chat.index("ws.on('message_annotation'"):
        chat.index("ws.on('log'")
    ]
    assert "updateMessageAnnotation" in annotation_handler
    assert "addMessage(" not in annotation_handler
    assert "clearTransientRoutingAnnotations();" in chat

    assert "menu.setAttribute('role', 'menu')" in menu
    assert 'role="menuitem" data-prm="rename"' in menu
    assert 'role="menuitem" class="danger" data-prm="delete"' in menu
    assert 'data-prm="hide"' not in menu
    for key in ("Escape", "ArrowDown", "ArrowUp", "Home", "End"):
        assert key in menu
    assert "window.innerWidth" in menu and "window.innerHeight" in menu
    assert "const PROJECT_NAME_MAX = 80" in menu
    assert "newName.length > maxNameLength" in menu
    assert 'maxlength="${maxNameLength}"' in menu


def test_project_main_mirror_never_creates_second_unread_static_contract():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    chat = (root / "web" / "modules" / "chat.js").read_text(encoding="utf-8")

    unread_fn = chat[
        chat.index("function incrementUnreadIfNeeded"):
        chat.index("ws.on('typing'")
    ]
    project_guard = unread_fn.index("if (isKnownProjectFrame(msg)) return;")
    increment = unread_fn.index("state.unreadCount++;")
    assert project_guard < increment
    assert "Project visible_revision is the sole unread authority" in unread_fn

    # The useful Main штаб/live-card mirror remains, but every unread call keeps
    # the original frame so the Project-origin guard can classify it.
    fanout = chat[
        chat.index("const isProjectMirrorFrame"):
        chat.index("ws.on('message_annotation'")
    ]
    assert "mirrorProject && isProjectMirrorFrame(msg)" in fanout
    assert "appendTaskSummaryToLiveCard(msg);" in fanout
    assert "updateLiveCardFromProgressMessage(msg);" in fanout
    assert "incrementUnreadIfNeeded(msg);" in fanout
    assert "incrementUnreadIfNeeded();" not in chat

    # History replay reconstructs the mirror but is not a new-delivery signal;
    # only live frames may advance Main's global unread counter.
    history = chat[
        chat.index("async function syncHistory"):
        chat.index("function cancelHistoryPaint")
    ]
    assert "appendTaskSummaryToLiveCard(msg" in history
    assert "incrementUnreadIfNeeded" not in history


def test_ephemeral_decision_progress_marker_survives_history_replay(tmp_path):
    from ouroboros.gateway.history import make_chat_history_endpoint

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "progress.jsonl").write_text(
        json.dumps({
            "ts": "2026-07-14T00:00:00Z",
            "type": "send_message",
            "direction": "out",
            "chat_id": 1,
            "task_id": "decision-1",
            "is_progress": True,
            "content": "Choosing the existing task",
            "format": "markdown",
            "ephemeral_decision": True,
        }) + "\n",
        encoding="utf-8",
    )

    endpoint = make_chat_history_endpoint(tmp_path)
    response = asyncio.run(endpoint(SimpleNamespace(query_params={"chat_id": "1"})))
    messages = json.loads(response.body.decode("utf-8"))["messages"]
    progress = next(message for message in messages if message.get("task_id") == "decision-1")
    assert progress["ephemeral_decision"] is True


def test_ephemeral_routing_keeps_annotation_and_final_in_history_projection(tmp_path, monkeypatch):
    """Finalization→supervisor→chat-log keeps one durable answer beside the
    routing annotation; progress remains marked for Web card suppression."""
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
        "_store_task_result",
        "_run_chat_consolidation",
        "_run_scratchpad_consolidation",
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
        env=SimpleNamespace(drive_root=tmp_path),
        memory=object(),
        llm=object(),
        pending_events=pending_events,
        task={
            "id": "decision-route-1",
            "type": "task",
            "chat_id": 1,
            "text": "Start the robot task",
            "_is_direct_chat": True,
            "_ephemeral_turn": True,
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
        "progress_meta": {"ephemeral_decision": True},
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
    assert progress["ephemeral_decision"] is True


def test_ephemeral_decision_web_frames_never_create_task_card_or_second_receipt():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    chat = (root / "web" / "modules" / "chat.js").read_text(encoding="utf-8")

    assert "const ephemeralDecisionTaskIds = new Set();" in chat
    register = chat[
        chat.index("function registerEphemeralDecisionFrame"):
        chat.index("function buildMessageKey")
    ]
    assert "ephemeralDecisionTaskIds.add(taskId);" in register
    assert "record.root?.remove();" in register

    card_factory = chat[
        chat.index("function createLiveCardRecord"):
        chat.index("function getLiveCardRecord")
    ]
    assert "!ephemeralDecisionTaskIds.has(normalizedGroupId)" in card_factory

    progress = chat[
        chat.index("function updateLiveCardFromProgressMessage"):
        chat.index("function updateSubagentCardFromEvent")
    ]
    logs = chat[
        chat.index("function updateLiveCardFromLogEvent"):
        chat.index("function addMessage")
    ]
    assert "if (registerEphemeralDecisionFrame(msg)) return;" in progress
    assert "if (registerEphemeralDecisionFrame(evt)) return;" in logs
    assert logs.index("showContextFitToast(evt);") < logs.index("registerEphemeralDecisionFrame(evt)")

    fanout = chat[
        chat.index("ws.on('chat'"):
        chat.index("ws.on('message_annotation'")
    ]
    # Inline ephemeral answers are not blanket-suppressed. Typed routing turns
    # retain any non-empty final answer while their progress/card stays hidden.
    assert fanout.count("if (ephemeralDecision) return;") == 1  # progress/card path only
    assert fanout.index("showTaskIncidentToast(msg);") < fanout.index("if (ephemeralDecision) return;")
    assert "addMessage(msg.content" in fanout
