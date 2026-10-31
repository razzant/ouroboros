"""A scheduled document selection reaches the child's first start through the real chain (C505).

The parent's real ``schedule_subagent`` event goes through supervisor admission, the durable
queue snapshot and its restore; the worker's own copy of that payload then drives the real
bootstrap, whose first start imports through the real gateway methods into the C505 engine
double. Only the engine and the worker process boundary (a JSON copy) are synthetic.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ouroboros import delegate_custody as custody
from ouroboros.delegate_documents import material_id, task_document_imports
from tests._delegated_transport_shared import (  # noqa: F401
    _owned_gateway_uses_each_test_transport,
    _select_transport_access,
)
from tests.test_delegate_documents import _HARNESS, BRIEF, STORE, DocumentEngine
from tests.test_scheduled_effort_intent import _admit, _restore, runtime  # noqa: F401

pytestmark = pytest.mark.serial  # Production state/queue init rebinds module globals.
_SELECTION = ("attachment_paths", "attachment_import", "document_import_intent")


def _selected(carrier):
    return {key: carrier[key] for key in _SELECTION if key in carrier}


@pytest.fixture
def chain(runtime, monkeypatch):  # noqa: F811
    """The parent task with two staged documents and an image, and an agent-session row."""
    from ouroboros.artifacts import attachment_manifest_projection, stage_task_attachments
    from ouroboros.gateways import claudexor as gateway_module
    from ouroboros.tools.registry import ToolContext
    from tests._shared import configure_test_subagent

    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(runtime.drive.parent / "snap_root"))
    _select_transport_access(monkeypatch, "readonly")
    sources = runtime.drive.parent / "owner-files"
    sources.mkdir()
    (sources / "brief.md").write_bytes(BRIEF)
    (sources / "notes.txt").write_bytes(b"plain notes\n")
    (sources / "photo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 32)
    rows = stage_task_attachments(runtime.drive, "parent", [
        {"path": str(sources / name), "label": name} for name in ("brief.md", "notes.txt", "photo.png")])
    assert [row["status"] for row in rows] == ["staged"] * 3, rows
    parent = ToolContext(repo_dir=runtime.repo, drive_root=runtime.drive, task_id="parent", task_depth=0,
                         current_chat_id=7, task_metadata={
                             "root_task_id": "parent",
                             "task_contract": attachment_manifest_projection(runtime.drive, "parent", rows)})
    parent.active_model, parent.active_effort, parent.active_use_local = "openai/parent", "high", False
    engine = DocumentEngine()
    monkeypatch.setattr(gateway_module, "ClaudexorGateway", lambda *a, **k: engine)
    sid = configure_test_subagent(monkeypatch, subagent_id="docs-session", kind="agent_session",
                                  target="some-route=weak-model")
    return runtime, parent, engine, sid


def _scheduled_event(chain, **selection):
    from ouroboros.tools.control_scheduling import _schedule_task

    _runtime, parent, _engine, sid = chain
    reply = _schedule_task(parent, subagent_id=sid, objective="Summarize the brief", expected_output="A summary",
                           memory_mode="empty", input_sources="declared", **selection)
    assert len(parent.pending_events) == 1, reply
    event = parent.pending_events[0]
    assert event["type"] == "schedule_subagent" and event["configured_subagent"]["route"]["kind"] == "agent_session"
    return event


def _worker_bootstrap(chain, monkeypatch, payload):
    """The worker's first start of the admitted child, from its own copy of the payload."""
    from ouroboros import claudexor_daemon
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.subagent_bootstrap import bootstrap_before_context
    from ouroboros.tools.registry import ToolContext

    rt, _parent, engine, _sid = chain
    assert payload["task_constraint"]["mode"] == "local_readonly_subagent"
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", lambda: engine)  # admission never touched it
    custody._CUSTODY.clear()  # a fresh worker process
    child = ToolContext(repo_dir=rt.repo, drive_root=Path(payload["child_drive_root"] or payload["drive_root"]),
                        task_constraint=TaskConstraint(mode="local_readonly_subagent"))
    child.task_id = payload["id"]
    child.task_metadata = {**payload["metadata"], **{key: payload[key] for key in (
        "root_task_id", "parent_task_id", "drive_root", "child_drive_root", "budget_drive_root") if payload.get(key)}}
    return child, bootstrap_before_context(child, payload, _HARNESS)


@pytest.mark.parametrize("mode", ["reuse", "new"])
def test_a_scheduled_selection_crosses_admission_queue_and_worker_to_the_first_start(chain, monkeypatch, mode):
    from ouroboros.task_results import load_task_result

    rt, _parent, engine, _sid = chain
    event = _scheduled_event(chain, attachment_paths=["attachments/brief.md"], attachment_import=mode)
    selection = _selected(event)
    assert selection["attachment_paths"] == ["attachments/brief.md"] and selection["attachment_import"] == mode
    assert ("document_import_intent" in selection) is (mode == "new")
    task = _admit(rt, event)
    admitted = load_task_result(rt.drive, task["id"])
    assert _selected(task) == _selected(task["metadata"]) == _selected(admitted) == selection
    payload = _restore(rt)  # the durable snapshot, restored, as the worker's own copy
    assert _selected(payload) == _selected(payload["metadata"]) == selection
    assert engine.calls == []

    child, wake = _worker_bootstrap(chain, monkeypatch, payload)
    receipt = json.loads(wake)
    assert receipt["status"] == "configured_session_started", receipt
    (_, _, declaration), *_lookups = engine.of("import")
    assert declaration["source"] == f"ouroboros-task-input:{payload['id']}/attachments/brief.md"
    assert [call[2] for call in engine.of("put")] == [BRIEF]
    (_, _, body), = engine.of("run")
    assert body.get("attachments") == [{"resourceId": "res_1"}]
    assert "SELECTED DOCUMENTS" in body["prompt"] and "- attachments/brief.md" in body["prompt"]
    assert body["prompt"] == child._configured_actor_bootstrap["canonical_work_order"]
    rows = task_document_imports(rt.drive, payload["id"])
    (material, record), = rows.items()
    assert (record["state"], record["store_id"], record.get("intent")) == (
        "ready", STORE, selection.get("document_import_intent"))
    started, = [row for row in custody._iter_rows(custody.event_log_path(rt.drive))
                if row.get("type") == custody.START_REQUESTED and row.get("task_id") == payload["id"]]
    assert [fact["path"] for fact in started["document_attachments"]] == ["attachments/brief.md"]
    copied = payload["task_contract"]["attachment_manifest"]
    assert [row["relpath"] for row in copied] == ["attachments/brief.md"] and material == material_id(copied[0])


def test_without_the_intent_registry_entries_the_selection_never_reaches_the_first_start(chain, monkeypatch):
    """Negative control: the registry is the one carrier. Without its document entries the same
    event loses the selection in admission, so the first start attaches nothing and its work order
    names no selected documents, although the producer's own requested row still holds them."""
    from ouroboros import subagents
    from ouroboros.task_results import load_task_result

    rt, _parent, engine, _sid = chain
    monkeypatch.setattr(subagents, "SUBAGENT_INTENT_FIELDS",
                        tuple(key for key in subagents.SUBAGENT_INTENT_FIELDS if key not in _SELECTION))
    event = _scheduled_event(chain, attachment_paths=["attachments/brief.md"], attachment_import="new")
    assert set(_selected(event)) == set(_SELECTION)
    requested = load_task_result(rt.drive, event["task_id"])
    task = _admit(rt, event)
    assert _selected(task) == _selected(task["metadata"]) == {}
    payload = _restore(rt)
    assert _selected(payload) == {}
    receipt = json.loads(_worker_bootstrap(chain, monkeypatch, payload)[1])
    assert receipt["status"] == "configured_session_started", receipt
    (_, _, body), = engine.of("run")
    assert "attachments" not in body and "SELECTED DOCUMENTS" not in body["prompt"]
    assert engine.of("import") == [] and task_document_imports(rt.drive, payload["id"]) == {}
    assert _selected(requested) == _selected(event)
