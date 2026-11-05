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
        workers, "get_event_q", lambda *, create=True: SimpleNamespace(put=queued.append),
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
    assert queued[0]["text"] == "Launch 🚀 › Ship release · Started"
    assert queued[0]["progress_meta"] == {
        "project_id": "launch",
        "project_name": "Launch 🚀",
        "task_name": "Ship release",
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



def test_chat_annotation_compaction_rewrites_the_journal_in_binary_mode(tmp_path, monkeypatch):
    """The compacted journal keeps the appends' LF bytes: Windows writes CRLF through a descriptor opened
    without ``O_BINARY``. POSIX translates nothing, so the bit is simulated as in ``test_atomic_write_v639``."""
    import os as _os

    from ouroboros import project_dialogue
    from ouroboros.project_dialogue import append_chat_annotation

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").write_text("", encoding="utf-8")
    (logs / "chat_annotations.jsonl").write_text(json.dumps({
        "ts": "2026-07-13T00:00:00Z", "type": "chat_annotation", "client_message_id": "expired",
        "action": "routed", "target": "x" * 800_000, "status": "delivered"}) + "\n", encoding="utf-8")
    real_o_binary = getattr(_os, "O_BINARY", None)
    fake_o_binary = real_o_binary if real_o_binary is not None else 1 << 26
    opened, real_open = [], _os.open

    def spy_open(path, flags, *args, **kwargs):
        opened.append((str(path), flags))
        return real_open(path, flags if real_o_binary is not None else flags & ~fake_o_binary, *args, **kwargs)

    monkeypatch.setattr(_os, "O_BINARY", fake_o_binary, raising=False)
    monkeypatch.setattr(project_dialogue.os, "open", spy_open)
    assert append_chat_annotation(tmp_path, "also-expired", action="routed", status="delivered")
    monkeypatch.undo()
    rewrites = [flags for path, flags in opened if ".chat_annotations.jsonl.tmp." in path]
    assert rewrites and all(flags & fake_o_binary for flags in rewrites)
    assert (logs / "chat_annotations.jsonl").read_bytes() == b""

def test_chat_annotation_compaction_keeps_receipts_that_address_no_message(tmp_path):
    """A steer the agent authored itself is answered by a receipt keyed on its
    routing token, and its only reader is the tool waiting on `routing_wait`.
    Chat-row membership cannot retire it: the append that crosses the threshold
    used to delete the very row it had just written, so a landed delivery came
    back as unconfirmed. Stale owner rows still go."""
    from ouroboros.project_dialogue import (
        AGENT_RECEIPT_ID_PREFIX, _COMPACT_AT_BYTES, append_chat_annotation,
        latest_chat_annotations,
    )

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").write_text(
        json.dumps({
            "direction": "in", "chat_id": 1,
            "client_message_id": "msg-1789388120127-2", "text": "publish the skills",
        }) + "\n",
        encoding="utf-8",
    )
    annotations = logs / "chat_annotations.jsonl"
    live = {
        "ts": "2026-09-14T12:37:49Z", "type": "chat_annotation",
        "client_message_id": "msg-1789388120127-2", "action": "promote_chat_to_task",
        "target": "a3e646fc62a44015", "status": "scheduled", "detail": "x" * 400,
    }
    stale = {**live, "client_message_id": "msg-expired-1"}
    with annotations.open("w", encoding="utf-8") as stream:
        while annotations.stat().st_size < _COMPACT_AT_BYTES:
            stream.write(json.dumps(live) + "\n")
            stream.write(json.dumps(stale) + "\n")
            stream.flush()
    assert annotations.stat().st_size >= _COMPACT_AT_BYTES  # the next append compacts

    receipt_id = f"{AGENT_RECEIPT_ID_PREFIX}f00dtoken"
    assert append_chat_annotation(
        tmp_path, receipt_id, action="steer_task", target="a3e646fc62a44015",
        status="delivered", routing_token="f00dtoken",
    )

    latest = latest_chat_annotations(tmp_path)
    assert annotations.stat().st_size < _COMPACT_AT_BYTES  # it really did compact
    assert latest[receipt_id]["status"] == "delivered"  # survives its own append
    assert latest[receipt_id]["routing_token"] == "f00dtoken"
    assert "msg-1789388120127-2" in latest  # the owner's message keeps its receipt
    assert "msg-expired-1" not in latest  # chat retention still drops the rest

    # Per-id dedupe bounds the synthetic rows exactly like the addressed ones.
    assert append_chat_annotation(
        tmp_path, receipt_id, action="steer_task", target="a3e646fc62a44015",
        status="needs_manual_target", routing_token="f00dtoken",
    )
    rows = [
        json.loads(line) for line in annotations.read_text(encoding="utf-8").splitlines() if line
    ]
    assert [row["status"] for row in rows if row["client_message_id"] == receipt_id] == [
        "delivered", "needs_manual_target",
    ]
    assert latest_chat_annotations(tmp_path)[receipt_id]["status"] == "needs_manual_target"


def test_synthetic_receipt_retention_is_bounded_by_the_newest_cap(tmp_path):
    """Every steer mints a fresh token, so per-id dedupe bounds nothing: without a
    cap the synthetic rows accumulate forever, the file stays permanently above the
    compaction threshold and every append rewrites the whole of it. The newest cap
    survives, which is all any live `routing_wait` (15 s of polling) can need."""
    from ouroboros.project_dialogue import (
        AGENT_RECEIPT_ID_PREFIX, _COMPACT_AT_BYTES, _RETAINED_AGENT_RECEIPTS,
        append_chat_annotation, latest_chat_annotations,
    )

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").write_text("", encoding="utf-8")
    annotations = logs / "chat_annotations.jsonl"
    seeded = 70
    with annotations.open("w", encoding="utf-8") as stream:
        for index in range(seeded):
            stream.write(json.dumps({
                "ts": f"2026-09-14T12:{index // 60:02d}:{index % 60:02d}Z",
                "type": "chat_annotation",
                "client_message_id": f"{AGENT_RECEIPT_ID_PREFIX}token{index:04d}",
                "action": "steer_task", "target": "t-target", "status": "delivered",
                "routing_token": f"token{index:04d}", "detail": "x" * 200,
            }) + "\n")
    assert annotations.stat().st_size < _COMPACT_AT_BYTES  # no compaction yet
    assert len(latest_chat_annotations(tmp_path)) == seeded

    # Cross the threshold with one oversized stale row, then append the newest receipt.
    with annotations.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "ts": "2026-07-13T00:00:00Z", "type": "chat_annotation",
            "client_message_id": "expired", "action": "routed",
            "target": "x" * _COMPACT_AT_BYTES, "status": "delivered",
        }) + "\n")
    newest = f"{AGENT_RECEIPT_ID_PREFIX}tokennewest"
    assert append_chat_annotation(
        tmp_path, newest, action="steer_task", target="t-target",
        status="delivered", routing_token="tokennewest",
    )

    latest = latest_chat_annotations(tmp_path)
    assert annotations.stat().st_size < _COMPACT_AT_BYTES  # it really did compact
    synthetic = [mid for mid in latest if mid.startswith(AGENT_RECEIPT_ID_PREFIX)]
    assert len(synthetic) <= _RETAINED_AGENT_RECEIPTS
    assert newest in synthetic          # the row its own append had to keep
    assert "expired" not in latest      # chat retention still drops addressed rows


def test_legacy_decision_progress_remains_readable_in_history(tmp_path):
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
    assert progress.get("is_progress") is True
    assert progress.get("text") or progress.get("content")


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
