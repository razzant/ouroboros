"""Live carriers and room handoff retain the ordinary restart/delivery gates."""
import json

import pytest

from tests.test_agent_restart_intent import runtime as runtime, accept
from tests.test_call_task_lifecycle import direct_call as direct_call
from tests.test_delivery_contention import sender as sender, owed
from tests.test_live_review_r9_relay import actor

pytestmark = pytest.mark.serial


def test_restart_defers_live_intent_without_constructing_an_actor(runtime, direct_call, monkeypatch):
    from ouroboros.realtime_turn import recovery_refusal
    from supervisor import worker_chat_lane

    accept(runtime)
    starts = []
    monkeypatch.setattr(runtime.workers, "_get_chat_agent", lambda: starts.append(True))
    turn = direct_call.turn
    worker_chat_lane.handle_chat_direct(1, "Open the admitted call", realtime_turn=turn,
        task_metadata={"client_message_id": "deferred-call", "execution_transport": turn.request})
    assert starts == []
    [saved] = [task for task in runtime.queue.PENDING if task.get("text") == "Open the admitted call"]
    assert saved["model"] == turn.model and saved["use_local_model"] is False
    assert saved["metadata"]["execution_transport"] == turn.request
    snapshot = json.loads(runtime.queue.QUEUE_SNAPSHOT_PATH.read_text())
    [retained] = [task for task in snapshot["pending"] if task["id"] == saved["id"]]
    assert retained["task"]["metadata"] == saved["metadata"]
    assert recovery_refusal(retained["task"], None) == "realtime_restart_required"
    assert "Saved for execution" in direct_call.notices[-1][0][1]
    assert not any("Error:" in args[1] for args, _ in direct_call.notices)


def test_replay_dispatch_refusal_never_hands_an_unsent_answer_to_a_live_turn(tmp_path, monkeypatch, sender):
    from ouroboros import owner_mailbox as mail
    from supervisor import events_chat_delivery as delivery, terminal_delivery as td

    ctx, frames, _ = sender
    actor(monkeypatch, tmp_path)
    event = owed(tmp_path, text="Not sent yet", tid="source")
    calls = []

    def refused(root, delivery_id):
        calls.append(delivery_id)
        return False

    monkeypatch.setattr(td, "record_replay_dispatch", refused)
    assert delivery._handle_send_message({**event, "_delivery_replay": True}, ctx) is False
    assert calls == [event["delivery_id"]]
    assert frames == [] and not mail.drain_owner_entries(tmp_path, "recipient")
    assert not td.already_delivered(tmp_path, event["delivery_id"])
    assert td.pending_deliveries(tmp_path)[0]["text"] == event["text"]
