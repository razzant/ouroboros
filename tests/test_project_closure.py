"""Project decisions are reversible; accepted new work reopens without changing routing authority."""
import ast
import json
import pathlib
import queue
from types import SimpleNamespace
import typing

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from ouroboros import dialogue_store as ds, projects_registry as registry
from ouroboros.gateway.client_contracts import ProjectUpdateRequest
from ouroboros.gateway.contracts import ProjectClosure, ProjectEntry
from ouroboros.gateway.router import collect_routes
from ouroboros.gateway.schema import validate_ingress
from ouroboros.tools.control import _close_project, _list_projects, get_tools
from supervisor import message_ingress, terminal_delivery
from tests.test_gateway_parity import _js_typedef_fields, _js_typedef_properties

CLOSED = {"at": "2026-10-01T10:00:00Z", "by": "ouroboros:one", "note": "The work is finished."}


def context(root):
    return SimpleNamespace(task_id="one", drive_root=root / "execution", budget_drive_root=root,
                           task_metadata={}, task_contract={}, event_queue=queue.Queue())


@pytest.fixture
def room(tmp_path):
    registry.create_project(tmp_path, "alpha", name="Alpha", working_dir="/example", origin="owner_ui")
    return tmp_path


def test_closed_is_optional_reversible_metadata_without_a_routing_change(room):
    before = registry.get_project(room, "alpha")
    assert "closed" not in before
    saved = registry.update_project(room, "alpha", closed=CLOSED)
    assert {k: v for k, v in saved.items() if k != "closed"} == before
    assert registry.get_project(room, "alpha", strict=True)["closed"] == CLOSED
    assert registry.list_projects(room)[0]["closed"] == CLOSED
    assert registry.projects_summary(room)[0]["closed"] == CLOSED
    assert registry.project_admission_basis("alpha", saved) == registry.project_admission_basis("alpha", before)
    assert registry.update_project(room, "alpha", name="Renamed")["closed"] == CLOSED
    reopened = registry.update_project(room, "alpha", closed=None)
    assert "closed" not in reopened and reopened["name"] == "Renamed"
    assert "closed" not in json.loads((room / "state/projects.json").read_text())["projects"][0]


@pytest.mark.parametrize("value", [True, False, "closed", {}, [], {**CLOSED, "by": ""},
                                  {**CLOSED, "at": None}, {**CLOSED, "note": 1}, {**CLOSED, "extra": "x"}])
@pytest.mark.parametrize("already_closed", [False, True])
def test_malformed_closure_refuses_the_whole_partial_update(room, value, already_closed):
    if already_closed:
        registry.update_project(room, "alpha", closed=CLOSED)
    path = room / "state/projects.json"
    before = path.read_bytes()
    with pytest.raises(ValueError, match="closed"):
        registry.update_project(room, "alpha", name="Not saved", closed=value)
    assert path.read_bytes() == before


def test_new_task_reopens_but_invalid_and_replayed_bindings_do_not(room, monkeypatch):
    registry.bind_task_to_project(room, "old", "alpha", origin={"absent": "system"})
    registry.update_project(room, "alpha", closed=CLOSED)
    registry.bind_task_to_project(room, "old", "alpha", origin={"absent": "system"})
    assert registry.get_project(room, "alpha")["closed"] == CLOSED
    with pytest.raises(ValueError):
        registry.bind_task_to_project(room, "invalid", "alpha", origin={})
    assert registry.get_project(room, "alpha")["closed"] == CLOSED
    with monkeypatch.context() as patch:
        patch.setattr(registry, "_save_bindings", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
        with pytest.raises(OSError):
            registry.bind_task_to_project(room, "failed", "alpha", origin={"absent": "system"})
    assert registry.get_project(room, "alpha")["closed"] == CLOSED
    registry.bind_task_to_project(room, "new", "alpha", origin={"absent": "system"})
    assert "closed" not in registry.get_project(room, "alpha")
    assert registry.project_id_for_task(room, "new") == "alpha"


@pytest.mark.parametrize("writer", ["accept_chat", "log_chat"])
def test_only_a_fresh_accepted_owner_message_reopens_its_room(room, writer):
    chat = registry.get_project(room, "alpha")["chat_id"]
    registry.create_project(room, "other")
    registry.update_project(room, "other", closed=CLOSED)
    write = getattr(message_ingress, writer)

    def incoming(source, identity, user=1):
        args = (chat, user, "Owner words") if writer == "accept_chat" else ("in", chat, user, "Owner words")
        return write(*args, source=source, client_message_id=identity, drive_root=room)

    registry.update_project(room, "alpha", closed=CLOSED)
    incoming("skill_repair", "skill")
    incoming("web", "other-user", user=2)
    assert registry.get_project(room, "alpha")["closed"] == CLOSED
    incoming("web", "owner")
    assert "closed" not in registry.get_project(room, "alpha")
    assert registry.get_project(room, "other")["closed"] == CLOSED
    if writer == "accept_chat":
        registry.update_project(room, "alpha", closed=CLOSED)
        assert incoming("web", "owner").duplicate
        assert registry.get_project(room, "alpha")["closed"] == CLOSED


def test_rejected_owner_append_does_not_reopen(room, monkeypatch):
    chat = registry.get_project(room, "alpha")["chat_id"]
    registry.update_project(room, "alpha", closed=CLOSED)
    monkeypatch.setattr(ds, "append", lambda *a, **kw: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(RuntimeError, match="persisted"):
        message_ingress.accept_chat(chat, 1, "Words", source="web", client_message_id="owner", drive_root=room)
    assert registry.get_project(room, "alpha")["closed"] == CLOSED


def test_mind_closes_and_reopens_with_one_durable_system_row_per_change(room):
    ctx = context(room)
    before = registry.get_project(room, "alpha")
    assert _close_project(ctx, "alpha", "Finished.").startswith("OK:")
    closed = registry.get_project(room, "alpha")
    assert closed["closed"]["by"] == "ouroboros:one" and closed["closed"]["note"] == "Finished."
    assert closed["routing_incarnation"] == before["routing_incarnation"]
    assert closed["lifecycle"] == registry.PROJECT_ACTIVE
    assert "already closed" in _close_project(ctx, "alpha", "Finished.")
    assert _close_project(ctx, "alpha", "More work.", reopen=True).startswith("OK:")
    assert "closed" not in registry.get_project(room, "alpha")
    assert "already open" in _close_project(ctx, "alpha", "More work.", reopen=True)
    events = [ctx.event_queue.get_nowait(), ctx.event_queue.get_nowait()]
    assert ctx.event_queue.empty()
    assert len(terminal_delivery.pending_deliveries(room)) == 2
    for event in events:
        assert event["role"] == "system" and event["system_type"] == "project_closure"
        assert event["chat_id"] == before["chat_id"] and not event.get("task_id")
        assert "Ouroboros (one)" in event["text"]
    assert closed["closed"]["at"] not in events[0]["text"]
    assert "Finished." in events[0]["text"] and "message from you" in events[0]["text"]
    assert "reopened" in events[1]["text"] and "More work." in events[1]["text"]


def test_owner_and_mind_share_one_idempotent_closure_transition(room, monkeypatch):
    ctx = context(room)
    _close_project(ctx, "alpha", "Finished.")
    original = registry.get_project(room, "alpha")["closed"]
    app = Starlette(routes=collect_routes(data_dir=room))
    app.state.drive_root = room
    saves, save = [], registry._save
    monkeypatch.setattr(registry, "_save", lambda *args: (saves.append(True), save(*args))[1])
    with TestClient(app) as client:
        assert client.post("/api/projects/alpha/update", json={"closed": True}).json()["project"]["closed"] == original
        assert "already closed" in _close_project(ctx, "alpha", "A later reason.")
        assert not saves
        assert "closed" not in client.post("/api/projects/alpha/update", json={"closed": False}).json()["project"]
        assert len(saves) == 1
        assert "already open" in _close_project(ctx, "alpha", "Again.", reopen=True)
        client.post("/api/projects/alpha/update", json={"closed": False})
        assert len(saves) == 1
        saved = client.post("/api/projects/alpha/update", json={"closed": True}).json()["project"]["closed"]
        assert saved["by"] == "owner"
        assert "already closed" in _close_project(ctx, "alpha", "Again.")
        assert registry.get_project(room, "alpha")["closed"] == saved and len(saves) == 2


@pytest.mark.parametrize("chat_id", [1, 0, -1, 777, None, "invalid"])
def test_nonproject_owner_messages_skip_the_registry(room, monkeypatch, chat_id):
    monkeypatch.setattr(registry, "list_reserved_projects", lambda *a, **kw: pytest.fail("no registry census"))
    message_ingress.reopen_for_owner_message(room, {"direction": "in", "source": "web", "user_id": 1, "chat_id": chat_id})


def test_mind_refusal_and_notice_failure_keep_their_actual_effects(room, monkeypatch):
    ctx = context(room)
    ctx.task_contract = {"lineage": {"delegation_role": "subagent"}}
    assert "subagents inherit" in _close_project(ctx, "alpha", "Done")
    ctx.task_contract = {}
    for pid, note, reopen in [("../alpha", "Done", False), ("alpha", "", False), ("alpha", "Done", "false")]:
        assert "TOOL_ARG_ERROR" in _close_project(ctx, pid, note, reopen)
    assert "closed" not in registry.get_project(room, "alpha") and ctx.event_queue.empty()
    monkeypatch.setattr(terminal_delivery, "enqueue_terminal_delivery_outcome", lambda *a, **kw: "unavailable")
    assert "unavailable; the project change is saved" in _close_project(ctx, "alpha", "Done")
    assert registry.get_project(room, "alpha")["closed"]["note"] == "Done"
    registry.begin_project_deletion(room, "alpha")
    assert "no longer active" in _close_project(ctx, "alpha", "Return", True)
    assert registry.get_reserved_project(room, "alpha")["closed"]["note"] == "Done"


def test_open_projects_are_listed_oldest_first_after_dialogue_activity_and_before_limit(room):
    for pid, at in [("alpha", "2026-01-01"), ("beta", "2026-01-02"), ("closed", "2025-01-01"), ("deleted", "2024-01-01")]:
        registry.create_project(room, pid, origin="owner_ui")
        registry.update_project(room, pid, last_active_at=at)
    registry.update_project(room, "closed", closed=CLOSED)
    registry.begin_project_deletion(room, "deleted")
    chat = registry.get_project(room, "alpha")["chat_id"]
    ds.append(room, "chat", {"direction": "out", "chat_id": chat, "text": "New activity", "ts": "2026-02-01"})
    assert [p["id"] for p in registry.list_projects(room, closed=False, oldest_first=True)] == ["beta", "alpha"]
    assert registry.projects_summary(room, closed=False, oldest_first=True, limit=1)[0]["id"] == "beta"
    text = _list_projects(context(room), closed=False, oldest_first=True)
    assert text.index("beta") < text.index("alpha") and "deleted" not in text
    assert "last_active_at=2026-02-01" in text and "closed=false" in text
    text = _list_projects(context(room), closed=True)
    assert '"note": "The work is finished."' in text and '"by": "ouroboros:one"' in text


@pytest.mark.parametrize("closed,oldest_first,limit", [
    (False, True, 50), (False, True, 80), (False, True, 999), (True, False, 50), (None, False, 200),
])
def test_project_reader_pages_reach_every_match_and_stop_on_the_last_page(room, closed, oldest_first, limit):
    path = room / "state/projects.json"
    data = json.loads(path.read_text())
    template = data["projects"][0]
    data["projects"] = [{**template, "id": f"room-{i:03}", "chat_id": 10000 + i,
                         "last_active_at": f"2026-10-{i // 24 + 1:02}T{i % 24:02}:00:00Z",
                         **({"closed": CLOSED} if i % 5 == 0 else {})} for i in range(300)]
    path.write_text(json.dumps(data))
    expected = [p["id"] for p in data["projects"] if closed is None or ("closed" in p) == closed]
    if not oldest_first:
        expected.reverse()
    params, seen, pages = {"limit": limit, "closed": closed, "oldest_first": oldest_first}, [], 0
    while True:
        text = _list_projects(context(room), **params)
        ids = [line[2:].split(" — ", 1)[0] for line in text.splitlines() if line.startswith("- ")]
        assert ids and len(ids) <= min(limit, 200)
        seen.extend(ids)
        pages += 1
        next_calls = [line.removeprefix("Next: ") for line in text.splitlines() if line.startswith("Next: ")]
        if len(seen) == len(expected):
            assert not next_calls  # includes a full final page, not just a short page
            break
        assert len(next_calls) == 1 and pages < 10
        call = ast.parse(next_calls[0], mode="eval").body
        assert isinstance(call, ast.Call) and call.func.id == "list_projects"
        params = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
        assert params["start"] == len(seen)
    assert pages > 1 and seen == expected and len(seen) == len(set(seen))
    assert "Next:" not in _list_projects(context(room), start=len(expected), closed=closed, oldest_first=oldest_first)
    assert [p["id"] for p in registry.projects_summary(room, limit=3)] == ["room-299", "room-298", "room-297"]
    schema = next(e.schema for e in get_tools() if e.name == "list_projects")
    assert schema["parameters"]["properties"]["start"]["default"] == 0


@pytest.mark.parametrize("start", [-1, True, 1.5, "1", None])
def test_project_reader_refuses_invalid_offsets(room, start):
    assert "TOOL_ARG_ERROR" in _list_projects(context(room), start=start)


def test_closure_tool_matches_move_access_and_publishes_its_arguments(tmp_path, monkeypatch):
    from ouroboros import consciousness_authority as ca
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tool_capabilities import CORE_TOOL_NAMES
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    assert "close_project" in CORE_TOOL_NAMES
    entries = {e.name: e for e in get_tools()}
    assert entries["close_project"].schema["parameters"]["required"] == ["project_id", "note"]
    for mode in (None, "local_readonly_subagent", "acting_subagent"):
        tools = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
        ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path,
                          task_constraint=TaskConstraint(mode=mode, allow_enable=False) if mode else None)
        tools.set_context(ctx)
        assert bool(tools.get_schema_by_name("close_project")) == bool(tools.get_schema_by_name("move_to_project"))
        assert bool(tools.get_schema_by_name("close_project")) == (mode is None)
    for level in ca.LEVELS:
        disabled = ca.disabled_tools_for(level)
        assert ("close_project" in disabled) == ("move_to_project" in disabled)


def test_owner_update_and_client_contracts_agree(room):
    app = Starlette(routes=collect_routes(data_dir=room))
    app.state.drive_root = room
    with TestClient(app) as client:
        for value in ("true", 1, 0, None, {}, []):
            assert client.post("/api/projects/alpha/update", json={"name": "Unsaved", "closed": value}).status_code == 400
        saved = client.post("/api/projects/alpha/update", json={"closed": True}).json()["project"]
        assert saved["closed"]["by"] == "owner" and saved["name"] == "Alpha"
        assert client.get("/api/projects").json()["projects"][0]["closed"] == saved["closed"]
        document = client.get("/api/contract").json()
        assert document["schemas"]["ProjectUpdateRequest"]["properties"]["closed"] == {"type": "boolean"}
        listed = document["schemas"]["ProjectListResponse"]["properties"]["projects"]["items"]
        assert set(listed["properties"]["closed"]["required"]) == {"at", "by", "note"}
        assert "closed" not in listed.get("required", [])
        assert "closed" not in client.post("/api/projects/alpha/update", json={"closed": False}).json()["project"]
    assert validate_ingress({"closed": False}, ProjectUpdateRequest) == []
    assert validate_ingress({"closed": "false"}, ProjectUpdateRequest)
    web = pathlib.Path(__file__).resolve().parents[1] / "web/modules"
    assert _js_typedef_fields((web / "api_types.js").read_text(), "ProjectClosure") == set(typing.get_type_hints(ProjectClosure))
    assert _js_typedef_fields((web / "api_types.js").read_text(), "ProjectEntry") == set(typing.get_type_hints(ProjectEntry))
    assert ("boolean=", "closed") in _js_typedef_properties((web / "client_contract_types.js").read_text(), "ProjectUpdateRequest")
