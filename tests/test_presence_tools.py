"""Chat-first owner controls for generic presence bindings and selections."""

from __future__ import annotations

import json

from ouroboros.tools.presence import get_tools
from ouroboros.tools.registry import ToolContext
from tests._store_shared import seed_result


def _configure(ctx: ToolContext, action: str, **params):
    entry = next(item for item in get_tools() if item.name == "configure_presence")
    return entry.handler(ctx, action, **params)


def _ctx(tmp_path) -> ToolContext:
    repo = tmp_path / "repo"
    repo.mkdir()
    return ToolContext(repo_dir=repo, drive_root=tmp_path)


def test_configure_place_connect_choices_and_disconnect(tmp_path, monkeypatch):
    from ouroboros import places
    monkeypatch.setattr("ouroboros.place_projection.place_declarations", lambda _root: [{
        "skill": "bridge", "id": "room", "label": "Chat", "destination": "vendor:chat", "wakes": ["mention"]}])
    ctx = _ctx(tmp_path)
    created = json.loads(_configure(ctx, "connect", connector="bridge", account_id="a"))["place"]
    saved = json.loads(_configure(ctx, "set_state", place_id=created["place_id"], state="silent"))["place"]
    assert saved["state"] == "silent"
    assert json.loads(_configure(ctx, "list"))["places"][0]["place_id"] == created["place_id"]
    assert json.loads(_configure(ctx, "disconnect", place_id=created["place_id"]))["ok"]
    assert places.load_places(tmp_path) == []


def test_retired_presence_actions_are_not_registered():
    assert not {"presence_finish", "initiate_presence"} & {tool.name for tool in get_tools()}


def test_presence_cancel_work_accepts_own_binding_work_from_any_of_its_conversations(monkeypatch, tmp_path) -> None:
    ctx = _ctx(tmp_path)
    ctx.task_metadata = {
        "presence": {
            "binding_id": "1" * 32,
            "event": {"conversation_key": "telegram:bot-1:room-1"},
        }
    }

    def row(task_id, *, binding="1" * 32, key="telegram:bot-1:room-1", **fields):
        seed_result(tmp_path, task_id, {
            "_schema_version": 1, "task_id": task_id, "status": "running",
            "metadata": {"presence": {"binding_id": binding, "event": {"conversation_key": key}}} if binding else {},
            **fields,
        })

    row("promoted-work-1", delegation_role="root", root_task_id="promoted-work-1")
    row("other-thread-work", key="telegram:bot-1:room-1:thread-9", delegation_role="root")
    row("foreign-binding-work", binding="2" * 32, delegation_role="root")
    row("presence-inline-turn")  # an inline turn is not deferred work
    row("owner-root", binding="", delegation_role="root")
    monkeypatch.setattr(
        "ouroboros.tools.join_ledger._cancel_task",
        lambda _ctx, task_id, reason="": f"cancel:{task_id}:{reason}",
    )
    entry = next(item for item in get_tools() if item.name == "presence_cancel_work")

    assert entry.handler(ctx, "promoted-work-1", "no longer needed") == (
        "cancel:promoted-work-1:no longer needed"
    )
    # Owner Q1: the same nonempty binding, a different thread or room of it.
    assert entry.handler(ctx, "other-thread-work") == "cancel:other-thread-work:"
    ctx.task_metadata["presence"]["event"]["conversation_key"] = "telegram:bot-1:other"
    assert entry.handler(ctx, "promoted-work-1") == "cancel:promoted-work-1:"
    for foreign in ("foreign-binding-work", "presence-inline-turn", "owner-root", "never-existed"):
        assert entry.handler(ctx, foreign).startswith("ERROR: PRESENCE_WORK_NOT_CORRELATED")
    ctx.task_metadata["presence"]["binding_id"] = ""  # an empty binding never compares equal
    assert entry.handler(ctx, "promoted-work-1").startswith("ERROR: PRESENCE_WORK_NOT_CORRELATED")
