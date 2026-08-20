"""route_to_project + list_projects (v6.33.0 WS10 LLM-first routing)."""

from __future__ import annotations

import types

from ouroboros.projects_registry import create_project
from ouroboros.tools.control import _list_projects, _route_to_project, get_tools


def _ctx(tmp_path, events=None, *, task_metadata=None, **overrides):
    values = dict(
        pending_events=events if events is not None else [],
        event_queue=None,
        current_chat_id=1,
        drive_root=tmp_path,
        task_metadata=task_metadata or {},
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


def test_route_to_existing_project_emits_event_and_receipt(tmp_path):
    create_project(tmp_path, "racer", name="Racer")
    # The origin identity is captured at INGRESS and rides task_metadata by
    # value; the tool never re-derives it from chat-log content (v6.73.0 — the
    # message text may even be an LLM paraphrase and routing still keeps the ref).
    origin_ref = {
        "chat_id": 1,
        "client_message_id": "owner-route-1",
        "ts": "2026-07-14T12:00:00Z",
        "text_sha256": "b" * 64,
    }
    events = []
    ctx = _ctx(tmp_path, events, task_metadata={
        "client_message_id": "owner-route-1",
        "origin_message_ref": origin_ref,
        "origin_message_text": "continue the engine tuning",
    })
    out = _route_to_project(ctx, "racer", "paraphrased: keep tuning the engine", reason="follow-up")
    assert out.startswith("⚠️ ROUTE_UNCONFIRMED:")
    assert "do not retry automatically" in out.lower()
    assert len(events) == 1
    evt = events[0]
    assert evt["type"] == "promote_chat_to_task"
    assert evt["project_id"] == "racer"
    assert evt["routed_from_main"] is True
    assert "keep tuning the engine" in evt["objective"]
    assert "routing reason: follow-up" in evt["objective"]
    assert evt["chat_id"] == 1
    assert evt["task_id"]
    assert evt["routing_token"]
    assert evt["source_ref"] == origin_ref
    assert evt["source_text"] == "continue the engine tuning"
    assert ctx._typed_routing_action_emitted == "route_to_project"


def test_main_swarm_route_carries_intent_and_emits_only_once(tmp_path, monkeypatch):
    create_project(tmp_path, "racer", name="Racer")
    monkeypatch.setattr(
        "ouroboros.tools.control_events._wait_for_promotion_admission",
        lambda *_args, **_kwargs: {"status": "unconfirmed", "reason": "confirmation_timeout"},
    )
    events = []
    ctx = _ctx(
        tmp_path,
        events,
        task_metadata={
            "client_message_id": "swarm-route-1",
            "force_plan": True,
            "force_plan_source": "swarm",
        },
        is_ephemeral_turn=True,
        project_id="",
    )

    first = _route_to_project(ctx, "racer", "Audit and fix this in Racer")
    second = _route_to_project(ctx, "racer", "Audit and fix this in Racer")

    assert first == second
    assert len(events) == 1
    assert events[0]["force_plan"] is True
    assert events[0]["force_plan_source"] == "swarm"
    assert ctx._swarm_handoff_attempt["task_id"] == events[0]["task_id"]


def test_project_swarm_route_to_other_project_is_rejected_without_event(tmp_path):
    create_project(tmp_path, "beta", name="Beta")
    events = []
    ctx = _ctx(
        tmp_path,
        events,
        task_metadata={"force_plan": True, "force_plan_source": "swarm"},
        is_ephemeral_turn=True,
        project_id="alpha",
    )

    out = _route_to_project(ctx, "beta", "Audit and fix this in Beta")

    assert "SWARM_PROJECT_SCOPE_OWNED" in out
    assert events == []


def test_route_to_missing_project_emits_typed_manual_target(tmp_path):
    events = []
    metadata = {
        "client_message_id": "owner-1",
        "routing_contract": {"manual_options": [{"task_id": "task-1", "title": "Fix it"}]},
    }
    ctx = _ctx(tmp_path, events, task_metadata=metadata)
    out = _route_to_project(ctx, "ghost", "do the thing")
    assert "ROUTING_UNCONFIRMED" in out
    assert len(events) == 1
    assert events[0]["type"] == "routing_manual_target"
    assert events[0]["routing_token"]
    assert events[0]["chat_id"] == 1
    assert events[0]["client_message_id"] == "owner-1"
    assert events[0]["requested_target"] == "ghost"
    assert events[0]["reason"] == "target_not_found"
    assert events[0]["options"] == [{"task_id": "task-1", "title": "Fix it"}]
    assert ctx._typed_routing_action_emitted == "routing_manual_target"


def test_route_rejects_dirty_project_id(tmp_path):
    events = []
    out = _route_to_project(_ctx(tmp_path, events), "Bad Name!", "msg")
    assert "ROUTING_UNCONFIRMED" in out
    assert events[0]["routing_token"]
    assert events[0]["reason"] == "invalid_project_id"


def test_route_empty_target_is_the_typed_abstention_path(tmp_path):
    events = []
    metadata = {
        "client_message_id": "owner-2",
        "routing_contract": {
            "manual_options": [{"action": "new_task_in_project", "label": "New task in Project"}],
        },
    }
    out = _route_to_project(_ctx(tmp_path, events, task_metadata=metadata), "", "ambiguous follow-up")
    assert "ROUTING_UNCONFIRMED" in out
    assert events[0]["routing_token"]
    assert events[0]["reason"] == "target_unspecified"
    assert events[0]["options"][0]["label"] == "New task in Project"


def test_route_requires_message(tmp_path):
    create_project(tmp_path, "racer", name="Racer")
    events = []
    ctx = _ctx(tmp_path, events)
    out = _route_to_project(ctx, "racer", "   ")
    assert "TOOL_ARG_ERROR" in out
    assert events == []
    assert not hasattr(ctx, "_typed_routing_action_emitted")


def test_list_projects_lists_created_projects(tmp_path):
    create_project(tmp_path, "racer", name="Racer")
    create_project(tmp_path, "site", name="Marketing Site")
    out = _list_projects(_ctx(tmp_path))
    assert "racer" in out and "Racer" in out
    assert "site" in out and "Marketing Site" in out


def test_list_projects_empty(tmp_path):
    out = _list_projects(_ctx(tmp_path))
    assert "No projects yet" in out


def test_route_tool_uncertainty_contract_requires_manual_target():
    tool = next(entry for entry in get_tools() if entry.name == "route_to_project")
    description = tool.schema["description"]
    assert "needs_manual_target" in description
    assert "New task in Project" in description
    assert "answer inline and offer" not in description
