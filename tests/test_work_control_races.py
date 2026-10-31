"""Owner and mind controls compete at the real durable admission boundaries."""

import asyncio
import copy
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
from types import SimpleNamespace

import pytest

from ouroboros.task_results import load_task_result, write_task_result
from tests.test_owner_continue import _interrupted
from tests.test_owner_relay import context, source
from tests import _dialogue as dialogue_test
from tests.test_swarm_host_admission import host, rows  # noqa: F401

pytestmark = pytest.mark.serial


@pytest.mark.parametrize("winner", ["owner", "mind"])
def test_continue_button_and_tool_capture_then_claim_once(host, monkeypatch, winner):  # noqa: F811
    from ouroboros import owner_continue
    from ouroboros.gateway.task_continue import api_task_continue
    from ouroboros.tool_call_log import invocation_scope
    from ouroboros.tools.control_mind import _continue_task
    from supervisor.events_mind_controls import _apply

    _interrupted(host.root)
    ctx = context(host.root)
    with invocation_scope({"tool_call_id": "continue-call"}):
        assert json.loads(_continue_task(ctx, "pred-1"))["status"] == "requested"
    [event] = ctx.pending_events
    ready, claimed = Barrier(2), Event()
    real_claim = owner_continue.claim_on_predecessor

    def claim(root, predecessor, nonce, binding):
        actor = "owner" if nonce == "owner-press" else "mind"
        ready.wait(timeout=10)
        if actor != winner:
            assert claimed.wait(10)
        try:
            return real_claim(root, predecessor, nonce, binding)
        finally:
            if actor == winner:
                claimed.set()

    monkeypatch.setattr(owner_continue, "claim_on_predecessor", claim)

    async def body():
        return {"action_nonce": "owner-press"}

    def owner():
        return asyncio.run(api_task_continue(SimpleNamespace(json=body, path_params={"task_id": "pred-1"})))

    with ThreadPoolExecutor(max_workers=2) as pool:
        owner_future, mind_future = pool.submit(owner), pool.submit(_apply, event)
        response, mind = owner_future.result(timeout=30), mind_future.result(timeout=30)
    owner_result = json.loads(response.body)
    accepted = owner_result if winner == "owner" else mind
    refused = mind if winner == "owner" else owner_result
    assert accepted["ok"] and refused.get("reason_code", refused.get("error")) == "already_continued"
    assert response.status_code == (200 if winner == "owner" else 409)
    successor = accepted["successor_task_id"]
    assert refused["successor_task_id"] == successor
    assert [task["id"] for task in host.pending] == [successor] and not host.running
    stored = load_task_result(host.root, "pred-1")["continued_by"]
    assert stored["state"] == "admitted" and stored["successor_task_id"] == successor
    assert stored["by"].get("actor", "owner") == ("ouroboros" if winner == "mind" else "owner")
    if winner == "mind":
        assert stored["by"]["task_id"] == event["actor_task_id"]
        assert stored["by"]["text"] == event["reason"]
    assert "actor" not in stored
    # Re-delivery through both doors cannot replace the winning episode.
    monkeypatch.setattr(owner_continue, "claim_on_predecessor", real_claim)
    assert json.loads(owner().body)["successor_task_id"] == successor
    assert _apply(event)["successor_task_id"] == successor
    assert [task["id"] for task in host.pending] == [successor]
    assert load_task_result(host.root, "pred-1")["continued_by"]["by"] == stored["by"]


@pytest.mark.parametrize("pause_first", [False, True])
def test_retried_tell_crosses_pause_once_and_keeps_owed_mail(host, pause_first):  # noqa: F811
    from ouroboros.owner_mailbox import drain_owner_entries, mail_read_state
    from ouroboros.owner_pause import read_fence
    from ouroboros.tools.control_relay import _tell_task
    from supervisor import queue
    from supervisor.owner_pause_control import request_owner_pause
    from supervisor.steering import _handle_steer_task
    from tests.test_question_answer_routing import drain

    row, handle = source(host.root)
    ctx = context(host.root, row)
    _tell_task(ctx, "receiver", "Model context, not owner words", [handle])
    [event] = ctx.pending_events
    target = {"id": "receiver", "root_task_id": "receiver", "chat_id": 42,
              "type": "task", "_attempt": 1, "admitted_dispatch": "none"}
    host.pending.append(target)
    write_task_result(host.root, "receiver", "scheduled", root_task_id="receiver", chat_id=42)
    queue.ACCEPTANCE_FENCES["receiver"] = {"status": "active", "owner_message_generation": 2}
    if not pause_first:
        _handle_steer_task(event, host.ctx)
    pause = request_owner_pause("receiver", request_id="owner-pause")
    assert pause["ok"], pause
    before = copy.deepcopy(read_fence(host.root, "receiver"))
    for _ in range(2):
        _handle_steer_task(event, host.ctx)
    assert read_fence(host.root, "receiver") == before
    assert queue.ACCEPTANCE_FENCES["receiver"]["owner_message_generation"] == 3
    assert host.pending == [target] and not host.running
    [mail] = drain_owner_entries(host.root, "receiver")
    assert mail_read_state(host.root, "receiver", mail["msg_id"]) is not True
    relays = [r for r in dialogue_test.log_rows(host.root / "logs/chat.jsonl") if r.get("type") == "relay_entry"]
    assert len(relays) == 1 and relays[0]["relay"]["relay_id"] == event["relay_id"]
    # Only the owner's explicit Resume releases the fence; the same mail is read once.
    resumed = queue.resume_budget_paused_task("receiver")
    assert resumed["ok"], resumed
    receiver, messages = drain(host.root, "receiver")
    assert [d["content"] for d in receiver._owner_directives] == [row["text"]]
    assert "Model context" in str(messages)
    assert mail_read_state(host.root, "receiver", mail["msg_id"]) is True
    assert not drain_owner_entries(host.root, "receiver")
