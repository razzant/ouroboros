"""Competing real placement doors and deletion, ordered at the commit boundary."""

import asyncio
import copy
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
from types import SimpleNamespace

import pytest

from ouroboros import project_placement as placement
from ouroboros.projects_registry import (
    begin_project_deletion, create_project, get_project, project_binding_for_task,
)
from tests import _dialogue as dialogue_test
from tests.test_swarm_host_admission import host, rows  # noqa: F401

pytestmark = pytest.mark.serial


def move(host, entrance, pid):  # noqa: F811
    if entrance == "owner":
        from ouroboros.gateway.projects import api_project_from_task

        async def body():
            return {"task_id": "one", "project_id": pid, "name": pid.title()}

        request = SimpleNamespace(json=body, app=SimpleNamespace(state=SimpleNamespace(drive_root=host.root)))
        response = asyncio.run(api_project_from_task(request))
        return response.status_code, json.loads(response.body)
    from ouroboros.tools.control_delegation import _move_to_project
    from supervisor.events_project_routing import _handle_ensure_project_scope

    ctx = SimpleNamespace(task_id="one", task_metadata={}, task_contract={}, project_id="",
                          drive_root=host.root, budget_drive_root=host.root, pending_events=[],
                          event_queue=SimpleNamespace(put_nowait=lambda event: _handle_ensure_project_scope(event, host.ctx)))
    result = _move_to_project(ctx, project_name=pid.title(), project_id=pid)
    assert ctx.project_id == "", "placement cannot relabel the running folder/lease"
    [receipt] = [row for row in rows(host.root / "logs/chat_annotations.jsonl")
                 if row.get("action") == "ensure_project_scope"]
    return result, receipt


@pytest.mark.parametrize("winner", ["owner", "mind"])
def test_button_and_mind_capture_unbound_then_commit_one_placement(host, monkeypatch, winner):  # noqa: F811
    task = {"id": "one", "root_task_id": "one", "chat_id": 1, "workspace_root": "unchanged"}
    host.running["one"] = {"task": task}
    before = copy.deepcopy(task)
    ready, committed = Barrier(2), Event()
    real_move = placement.move_task_to_project
    captured = []

    def interleave(root, tid, pid, **kwargs):
        captured.append(copy.deepcopy(kwargs["basis"]))
        ready.wait(timeout=10)
        if pid != winner:
            assert committed.wait(10), "winning placement never returned"
        try:
            return real_move(root, tid, pid, **kwargs)
        finally:
            if pid == winner:
                committed.set()

    monkeypatch.setattr(placement, "move_task_to_project", interleave)
    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(move, host, "owner", "owner")
        mind = pool.submit(move, host, "mind", "mind")
        owner_status, owner_reply = owner.result(timeout=20)
        mind_status, mind_reply = mind.result(timeout=20)
    assert all(basis["project"] is None for basis in captured)
    assert owner_status == 200 and mind_status.startswith("OK:")
    assert owner_reply["project_id"] == mind_reply["target"] == winner
    loser_reply = mind_reply if winner == "owner" else owner_reply
    assert loser_reply["status"] == "delivered" and loser_reply["reason"] == "renamed"
    assert project_binding_for_task(host.root, "one")["project_id"] == winner
    assert get_project(host.root, "mind" if winner == "owner" else "owner") is None
    assert list(host.running) == ["one"] and not host.pending and task == before
    handoffs = [row for row in dialogue_test.log_rows(host.root / "logs/chat.jsonl") if row.get("type") == "project_handoff"]
    assert len(handoffs) <= 1 and all(row["task_id"] == "one" for row in handoffs)


@pytest.mark.parametrize("entrance", ["owner", "mind"])
@pytest.mark.parametrize("delete_first", [False, True])
def test_destination_deletion_fences_both_placement_doors(host, monkeypatch, entrance, delete_first):  # noqa: F811
    create_project(host.root, "room", name="Room")
    real_move = placement.move_task_to_project

    def interleave(root, tid, pid, **kwargs):
        if delete_first:
            begin_project_deletion(root, pid)
        result = real_move(root, tid, pid, **kwargs)
        if not delete_first:
            begin_project_deletion(root, pid)
        return result

    monkeypatch.setattr(placement, "move_task_to_project", interleave)
    status, result = move(host, entrance, "room")
    from ouroboros.project_admission import reserved_project_for_chat
    from ouroboros.contracts.chat_id_policy import project_chat_id

    assert reserved_project_for_chat(host.root, project_chat_id("room"))["lifecycle"] == "deleting"
    if delete_first:
        assert project_binding_for_task(host.root, "one") is None
        if entrance == "owner":
            assert status == 409, result
            assert result["reason_code"] == "project_routing_fence"
        else:
            assert "SCOPE_REJECTED" in status
            assert result["status"] == "rejected" and result["reason"] == "project_routing_fence"
    else:
        assert (status == 200 if entrance == "owner" else status.startswith("OK:"))
        assert project_binding_for_task(host.root, "one")["project_id"] == "room"
        pending = {"id": "one", "workspace_root": "original"}
        assert not placement.refresh_pending_placement(host.root, pending)
        assert pending == {"id": "one", "workspace_root": "original"}
    assert not host.pending and not host.running
