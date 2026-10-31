"""REV1 destination and promotion scope; working neighbors stay admitted."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from ouroboros import place_guard, places
from ouroboros.gateway import host_service
from ouroboros.presence_context import _own_work_section
from ouroboros.tool_effects import ToolEffect
from tests.test_place_focus import focus as focus, task_for


def test_declared_targets_preserve_account_room_and_topic(tmp_path, monkeypatch):
    active = places.Place("active", "bridge", "chat", "account", "room", "Active", thread_id="one")
    closed = replace(active, place_id="closed", thread_id="two", state="silent", label="Silent topic")
    foreign = replace(closed, place_id="foreign", account_id="other", thread_id="one")
    monkeypatch.setattr(places, "load_places", lambda _: [active, closed, foreign])
    monkeypatch.setattr(place_guard, "effect_of", lambda *_a, **_k: ToolEffect(
        "message", targets=("to", "cc", "account_id"), conversation="room", thread="topic", declared=True))
    monkeypatch.setattr(place_guard, "connector_of", lambda *_: "bridge")
    ctx = SimpleNamespace(drive_root=tmp_path, task_id="caller", task_metadata={})
    args = {"room": "room", "topic": "one", "to": "room", "account_id": "account"}
    assert not place_guard.dispatch_refusal(ctx, "send", args)
    assert "Silent topic" in place_guard.dispatch_refusal(ctx, "send", {**args, "topic": "two"})
    assert "Silent topic" in place_guard.dispatch_refusal(ctx, "send", {**args, "account_id": "other"})
    # Separate recipients remain separate checks, including a closed Cc.
    monkeypatch.setattr(places, "load_places", lambda _: [replace(active, thread_id=""),
                         replace(closed, conversation_id="closed-room", thread_id="")])
    monkeypatch.setattr(place_guard, "effect_of", lambda *_a, **_k: ToolEffect("message", targets=("to", "cc"), declared=True))
    assert not place_guard.dispatch_refusal(ctx, "send", {"to": "room"})
    assert "Silent topic" in place_guard.dispatch_refusal(ctx, "send", {"to": "room", "cc": "closed-room"})


def test_outbox_accepts_thread_and_omission_stays_room_lookup(tmp_path, monkeypatch):
    broad = places.Place("room", "bridge", "chat", "a", "r", "Room")
    topic = replace(broad, place_id="topic", thread_id="two", state="silent")
    monkeypatch.setattr(places, "load_places", lambda _: [broad, topic])
    async def authenticated(*_):
        return "bridge", {}
    monkeypatch.setattr(host_service, "_authenticated", authenticated)
    query = {"provider": "chat", "account_id": "a", "conversation_id": "r"}
    with TestClient(host_service.create_host_service_app(tmp_path)) as client:
        assert client.get("/presence/place", params=query).json()["place_id"] == "room"
        assert client.get("/presence/place", params={**query, "thread_id": "one"}).json()["state"] == "active"
        assert client.get("/presence/place", params={**query, "thread_id": "two"}).json() == {
            "place_id": "topic", "label": "Room", "state": "silent"}


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("no_folder", [False, True])
def test_promotion_preserves_requested_scope_unless_legacy_ceiling(focus, monkeypatch, legacy, no_folder):
    from ouroboros.tools import control_routing as routing
    from supervisor import worker_promotion, workers
    from ouroboros import workspace_admission as workspace
    from ouroboros.contracts.task_contract import attach_task_contract
    from ouroboros.presence_authority import presence_ceiling_payload
    from tests.test_presence_runner import _admission
    root, place, admission, event = focus
    inherited = str(root / "inherited")
    parent = task_for((root, place, replace(admission, workspace_root=inherited), event))
    parent["task_contract"]["resource_policy"] = {"unrelated": "keep"}
    if legacy:
        parent["task_contract"]["capability_ceiling"] = presence_ceiling_payload(_admission().capability_ceiling)
    events = []
    monkeypatch.setattr(routing, "_promotion_pool_disabled_from_snapshot", lambda _: "")
    monkeypatch.setattr(routing, "_emit_and_wait_for_routing", lambda _ctx, evt: (
        events.append(evt) or "confirmed", {"status": "scheduled"}))
    ctx = SimpleNamespace(task_id=parent["id"], task_metadata=parent["metadata"], task_contract=parent["task_contract"],
                          current_chat_id=parent["chat_id"], drive_root=root)
    requested = "" if no_folder else str(root / "requested")
    assert routing._start_task(ctx, "Build a report", title="Report", project_id="report-project", workspace_root=requested,
                               workspace="none" if no_folder else "",
                               source="https://example.invalid/report.git").startswith("OK: task")
    evt = events[0]
    assert (evt["project_id"], evt["workspace_root"], evt["source"]) == (
        ("", "", "") if legacy else ("report-project", requested, "https://example.invalid/report.git"))
    assert evt["presence"] == parent["metadata"]["presence"]
    task = {"id": "promoted", "text": "report", "project_id": evt["project_id"], "_project_admission": {}}
    workers._apply_presence_promotion_authority(evt, task, objective="report", expected_output="report")
    calls = []
    monkeypatch.setattr(workers, "DRIVE_ROOT", root)
    monkeypatch.setattr(workers, "REPO_DIR", root)
    monkeypatch.setattr(workspace, "resolve_room_workspace", lambda **kw: (calls.append(kw) or kw["explicit_workspace"], ""))
    monkeypatch.setattr(workspace, "bounded_workspace_preflight", lambda _: {})
    monkeypatch.setattr("ouroboros.projects_registry.update_project", lambda *_a, **_k: None)
    monkeypatch.setattr("ouroboros.headless.prepare_task_drive", lambda *_a, **_k: None)
    assert worker_promotion._admit_promoted_workspace(evt, None, task, pid=evt["project_id"], tid="promoted") is None
    assert calls[0]["explicit_workspace"] == (str(root / "inherited") if legacy else requested)
    assert calls[0]["project_id"] == ("" if legacy else "report-project")
    assert task.get("workspace_root", "") == calls[0]["explicit_workspace"]
    attach_task_contract(task)  # the final supervisor attachment must agree with admission
    assert task["task_contract"]["workspace"] == {
        "root": inherited if legacy else requested, "mode": "external" if legacy or requested else ""}
    assert task["task_contract"]["resource_policy"] == {"unrelated": "keep"}
    assert ("capability_ceiling" in task["task_contract"]) is legacy
    assert parent["task_contract"]["workspace"]["root"] == inherited
    assert task["_presence_origin"] and task["metadata"]["presence"] == parent["metadata"]["presence"]


def test_account_suggestions_are_recent_bounded_people_not_confirmation(focus):
    from ouroboros import dialogue_store
    from ouroboros.place_projection import place_view
    root, place, _, _ = focus
    assert place_view(root, place, declarations=[])["account_candidates"] == []
    for i in range(60):
        dialogue_store.append(root, "chat", {"type": "place_arrival", "place_id": place.place_id,
            "ts": f"2026-10-22T00:00:{i:02}", "sender_session_id": str(i), "sender_label": f"Person {i}",
            "transport": {"actor": {"is_bot": i == 59}}})
    dialogue_store.append(root, "chat", {"type": "place_arrival", "place_id": "other", "ts": "2099",
                                       "sender_session_id": "stranger", "sender_label": "Wrong place"})
    candidates = place_view(root, place, declarations=[])["account_candidates"]
    assert len(candidates) == 20 and candidates[0] == {"platform_actor_id": "58", "label": "Person 58"}
    assert all(row["platform_actor_id"] not in {"59", "stranger", "0"} for row in candidates)
    assert not places.get_place(root, place.place_id).owner_accounts


def test_retained_place_tool_words_keep_the_same_behavior(focus, monkeypatch):
    from ouroboros.tools.presence import get_tools
    from ouroboros.presence_authority import presence_work_refusal
    root, _, _, _ = focus
    task = task_for(focus)
    description = next(t.schema["description"] for t in get_tools() if t.name == "presence_cancel_work")
    assert "place" in description and "binding" not in description
    ctx = SimpleNamespace(task_id=task["id"], task_metadata=task["metadata"], drive_root=root)
    # Existing callable contract and typed code stay intact; only the human wording changes.
    refusal = presence_work_refusal(ctx, "absent", drive_root=root)
    assert "PRESENCE_CAPABILITY_BLOCKED" in refusal and "place" in refusal and "binding" not in refusal
    monkeypatch.setattr("ouroboros.tools.recent_tasks.recent_tasks_page", lambda *_a, **_k: {
        "tasks": [], "read_gap": {"queue_snapshot": True}})
    prose = _own_work_section(root, {"binding_id": "legacy"}, task["id"])
    assert "this place" in prose and "binding" not in prose.replace("own_binding", "")
    assert 'presence_scope="own_binding"' in prose  # retained argument ABI
    assert _own_work_section(root, {}, task["id"]) == ""


def test_legacy_presence_promotion_keeps_its_host_scope_and_records_handoff(tmp_path, monkeypatch):
    from ouroboros.tools.control import _start_task
    from tests.test_promote_chat_flow import _confirm_promote, _managed_swarm_ctx

    _confirm_promote(monkeypatch)
    presence = {"binding_id": "presence-binding"}
    ctx = _managed_swarm_ctx(tmp_path, project_id="alpha", task_metadata={"presence": presence})
    ctx.task_contract = {"capability_ceiling": {}}  # retained legacy work alone has a scope ceiling
    out = _start_task(
        ctx, "Audit and fix the issue", title="Issue audit and fix", project_id="beta", project_name="Injected Project",
        workspace_root="/tmp/foreign", workspace="none",
        source="https://example.invalid/repo.git",
    )
    assert out.startswith("OK: task")
    evt = ctx.pending_events[0]
    assert evt["presence"] == presence
    assert evt["project_id"] == evt["project_name"] == evt["workspace_root"] == evt["workspace"] == evt["source"] == ""
    assert "force_plan" not in evt
    assert ctx._swarm_handoff_attempt["status"] == "scheduled"
    assert ctx._swarm_handoff_attempt["task_id"] == evt["task_id"]
