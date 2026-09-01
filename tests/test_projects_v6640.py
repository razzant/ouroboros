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


def test_create_project_reports_creation_fact_without_persisting_it(tmp_path):
    """B1 contract: the returned dict carries the additive `created` key —
    True only for the call that registered the row, False on the idempotent
    replay — and the key never lands in the persisted registry entry."""
    from ouroboros.projects_registry import create_project, get_project, list_projects

    first = create_project(tmp_path, "alpha", name="Alpha")
    assert first["created"] is True
    replay = create_project(tmp_path, "alpha")
    assert replay["created"] is False
    assert replay["id"] == "alpha"
    assert replay["chat_id"] == first["chat_id"]
    assert all("created" not in row for row in list_projects(tmp_path))
    assert "created" not in (get_project(tmp_path, "alpha") or {})


def test_project_started_row_rides_outbox_pins_main_and_dedupes_durably(tmp_path, monkeypatch):
    """B1: the agent-initiated `project_started` row mirrors the completion
    mechanics — same terminal-delivery outbox, `project-start:<pid>` as the ONE
    restart-surviving dedupe, and the send handler pins chat 1 (Main) even
    though the task is already project-bound (lineage routing would otherwise
    pull the row into the Project thread)."""
    from ouroboros.project_dialogue import announce_project_started
    from ouroboros.projects_registry import bind_task_to_project, create_project
    from supervisor import events, workers

    project = create_project(tmp_path, "launch", name="Launch 🚀")
    bind_task_to_project(
        tmp_path, "root-project", project["id"], project["chat_id"],
        origin={"absent": "system"},
    )
    queued = []
    monkeypatch.setattr(
        workers, "get_event_q", lambda: SimpleNamespace(put=queued.append),
    )
    events._DELIVERED_MESSAGE_IDS.clear()
    task = {"id": "root-project", "project_id": "launch", "title": "Ship release"}

    assert announce_project_started(tmp_path, project, "root-project", task=task) is True
    assert announce_project_started(tmp_path, project, "root-project", task=task) is True
    assert len(queued) == 2  # duplicate live copies share one durable delivery id
    assert queued[0]["delivery_id"] == "project-start:launch"
    assert queued[0]["system_type"] == "project_started"
    assert queued[0]["role"] == "system"
    assert queued[0]["chat_id"] == 1
    assert queued[0]["text"] == (
        "Launch 🚀 › Ship release · Started\nWork is running in this Project."
    )
    assert queued[0]["progress_meta"] == {
        "project_id": "launch",
        "project_name": "Launch 🚀",
        "target_label": "Launch 🚀 › Ship release",
    }

    sent = []
    ctx = SimpleNamespace(
        DRIVE_ROOT=tmp_path, RUNNING={},
        send_with_budget=lambda *a, **k: sent.append((a, k)),
        append_jsonl=lambda *_a, **_k: None,
    )
    events._handle_send_message(queued[0], ctx)
    events._handle_send_message(queued[1], ctx)

    assert len(sent) == 1  # second copy suppressed by the durable delivery id
    assert sent[0][0][0] == 1  # pinned to Main despite the project binding
    assert sent[0][1]["system_type"] == "project_started"
    assert sent[0][1]["role"] == "system"
    assert sent[0][1]["progress_meta"]["target_label"] == "Launch 🚀 › Ship release"


def test_owner_api_create_does_not_announce_project_started(tmp_path, monkeypatch):
    """Owner decision 2=A: manual/HTTP project creation stays silent — no
    `project_started` row rides the terminal-delivery outbox from the owner
    API seam (only the agent-initiated workers seams announce)."""
    from ouroboros.gateway.projects import api_projects_create

    delivered = []
    monkeypatch.setattr(
        "supervisor.terminal_delivery.enqueue_terminal_delivery",
        lambda *_a, **_k: delivered.append(1) or True,
    )

    async def _json():
        return {"id": "manual-room", "name": "Manual Room"}

    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(drive_root=tmp_path)),
        json=_json,
    )
    response = asyncio.run(api_projects_create(request))
    assert response.status_code == 200
    from ouroboros.projects_registry import get_project

    assert get_project(tmp_path, "manual-room") is not None
    assert delivered == []


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
        detail="- missing: rejected (reason=source_missing, ordinal=0)",
        attachment_manifest=[{
            "ordinal": 0,
            "status": "rejected",
            "reason": "source_missing",
            "label": "missing",
        }],
    )
    path = tmp_path / "logs" / "chat_annotations.jsonl"
    with path.open("ab") as handle:
        handle.write(b'{"type":"chat_annotation"')

    latest = latest_chat_annotations(tmp_path)
    assert latest["owner-1"]["status"] == "delivered"
    assert set(latest["owner-1"]) == {
        "ts", "type", "client_message_id", "action", "target", "status",
        "detail", "attachment_manifest",
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
        "detail": "- missing: rejected (reason=source_missing, ordinal=0)",
        "attachment_manifest": [{
            "ordinal": 0,
            "status": "rejected",
            "reason": "source_missing",
            "label": "missing",
        }],
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
        chat.index("onWs('message_annotation'"):
        chat.index("onWs('log'")
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


def test_project_activity_stays_out_of_main_static_contract():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    chat = (root / "web" / "modules" / "chat.js").read_text(encoding="utf-8")

    unread_fn = chat[
        chat.index("function incrementUnreadIfNeeded"):
        chat.index("onWs('typing'")
    ]
    project_guard = unread_fn.index("if (isKnownProjectFrame(msg)) return;")
    increment = unread_fn.index("state.unreadCount++;")
    assert project_guard < increment
    assert "Project visible_revision is the sole unread authority" in unread_fn

    # Main has its own compact host-stamped completion row. Project progress,
    # logs and ordinary summaries never enter its live fan-out.
    fanout = chat[
        chat.index("const isMyThread"):
        chat.index("onWs('message_annotation'")
    ]
    assert "mirrorProject" not in fanout
    # Every Chat instance enters through the shared thread gate. Its Main arm
    # still excludes both server-stamped and already-known Project frames.
    assert "return chatThreadAccepts(msg, isMain, chatId, state.projectChatIds);" in fanout
    activity = (root / "web" / "modules" / "chat_activity.js").read_text(encoding="utf-8")
    main_gate = activity[
        activity.index("export function mainThreadAccepts"):
        activity.index("/** Main routing for the legacy LocalChatBridge")
    ]
    shared_gate = activity[
        activity.index("export function chatThreadAccepts"):
        activity.index("/**\n * Route one LocalChatBridge log envelope")
    ]
    assert "if (msg && msg.project_thread) return false;" in main_gate
    assert "projectChatIds.has(cid)" in main_gate
    assert "if (isMain) return mainThreadAccepts(msg, projectChatIds);" in shared_gate
    assert "return Number(msg?.chat_id ?? 1) === chatId;" in shared_gate
    assert "PROJECT_ROW_TYPES.has(msg.system_type)" in fanout
    assert "appendTaskSummaryToLiveCard(msg);" in fanout
    assert "updateLiveCardFromProgressMessage(msg, { grantCancelAuthority: true });" in fanout
    assert "incrementUnreadIfNeeded(msg);" in fanout
    assert "incrementUnreadIfNeeded();" not in chat

    # History replay renders the durable compact completion independently of
    # task-card reconstruction.
    history = chat[
        chat.index("async function syncHistory"):
        chat.index("function cancelHistoryPaint")
    ]
    assert "appendTaskSummaryToLiveCard(msg" in history
    assert "PROJECT_ROW_TYPES.has(msg.system_type)" in history
    assert "incrementUnreadIfNeeded" not in history
    assert "name: projectName || 'Project'" in chat
    assert "name: projectName || projectId" not in chat


def test_project_lifecycle_rows_render_design_system_action_static_contract():
    """B2 static guard (sol item 4 UI / fable 4.3): BOTH host-stamped Project
    lifecycle rows render their action through the shared design-system helper
    inside a `.system-message-actions` container; the custom pill class and its
    bare `<br>` spacing are structurally gone from chat.js AND style.css."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    chat = (root / "web" / "modules" / "chat.js").read_text(encoding="utf-8")
    style = (root / "web" / "style.css").read_text(encoding="utf-8")
    helpers = (root / "web" / "modules" / "ui_helpers.js").read_text(encoding="utf-8")

    # One shared set drives render, history replay, and live fan-out.
    assert (
        "const PROJECT_ROW_TYPES = new Set(['project_started', 'project_completion_summary']);"
        in chat
    )
    render = chat[
        chat.index("if (PROJECT_ROW_TYPES.has(systemType) && projectId) {"):
        chat.index("function updateMessageAnnotation")
    ]
    assert "createSystemMessageAction({" in render
    assert "'system-message-actions'" in render
    assert "document.createElement('br')" not in render

    # The custom pill is gone everywhere; the conversion-flow buttons moved to
    # the shared design-system role beside their `btn btn-xs btn-danger` sibling.
    assert "chat-live-project-btn" not in chat
    assert "chat-live-project-btn" not in style
    assert 'class="btn btn-xs btn-default" data-turn-into-project' in chat
    # The identity chip keeps its own role untouched.
    assert "chat-live-project-card-btn" in chat

    # Layout-only container CSS; the helper owns the one semantic button role.
    assert ".system-message-actions {" in style
    assert ".system-message-action {" in style
    assert "export function createSystemMessageAction(" in helpers
    assert "'btn btn-default btn-sm system-message-action'" in helpers


def test_chat_ws_subscriptions_flow_through_disposer_helper():
    """P3 lifecycle: every WS subscription in chat.js must go through the
    onWs helper so destroy() can release it. A bare ws.on() call would leak
    a listener past the instance lifetime; the helper's own definition is
    the single allowed occurrence of `ws.on(`."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    chat = (root / "web" / "modules" / "chat.js").read_text(encoding="utf-8")

    assert "const onWs = (event, fn) => wsDisposers.push(ws.on(event, fn));" in chat
    assert chat.count("ws.on(") == 1


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
        # buildMessageKey moved to chat_activity.js; the next stable symbol
        # after the register pair bounds the slice now.
        chat.index("function readPendingReconnectBanner")
    ]
    assert "ephemeralDecisionTaskIds.add(taskId);" in register
    assert "record.root?.remove();" in register

    card_factory = chat[
        chat.index("function createLiveCardRecord"):
        chat.index("function getLiveCardRecord")
    ]
    assert "!ephemeralDecisionTaskIds.has(normalizedGroupId)" in card_factory

    logs = chat[
        chat.index("function updateLiveCardFromLogEvent"):
        chat.index("function addMessage")
    ]
    assert chat.count("if (ephemeral !== undefined) return ephemeral;") == 3
    assert logs.index("registerEphemeralDecisionFrame(evt)") < logs.index(
        "const taskId = getLogTaskGroupId(evt)"
    )

    fanout = chat[
        chat.index("onWs('chat'"):
        chat.index("onWs('message_annotation'")
    ]
    # Inline ephemeral answers are not blanket-suppressed. Typed routing turns
    # retain any non-empty final answer while their progress/card stays hidden.
    assert "const isEphemeral = ephemeral !== undefined;" in fanout
    assert fanout.count("if (isEphemeral) return ephemeral;") == 1
    assert fanout.index("showTaskIncidentToast(msg);") < fanout.index("if (isEphemeral) return ephemeral;")
    assert "addMessage(msg.content" in fanout
