"""Live carriers and room handoff retain the ordinary restart/delivery gates."""


from tests.test_agent_restart_intent import runtime as runtime, accept
from tests.test_call_task_lifecycle import direct_call as direct_call
from tests.test_delivery_contention import sender as sender, owed
from tests.test_live_review_r9_relay import actor



def test_restart_never_saves_a_live_call_turn(direct_call, runtime, monkeypatch):
    # ``direct_call`` first: ``runtime`` then installs the queue tables the restart's census reads.
    """A call utterance is dialogue: neither a requested restart nor its drain saves it as a queued root (a saved call
    is refused after the restart, ``realtime_turn.recovery_refusal``); it runs while the process lives."""
    from supervisor import worker_chat_lane
    from tests.test_call_task_lifecycle import _agent

    accept(runtime)
    runtime.restart._AGENT_RESTART["status"] = "ready"  # the physical drain: a text turn would be saved here
    seen = []
    agent = _agent(direct_call.root, lambda task=None: seen.append(task["id"]) or [])
    monkeypatch.setattr(runtime.workers, "_get_chat_agent", lambda: agent)
    turn = direct_call.turn
    worker_chat_lane.handle_chat_direct(1, "Open the admitted call", realtime_turn=turn,
        task_metadata={"client_message_id": "drain-call", "execution_transport": turn.request})
    assert [task for task in runtime.queue.PENDING if task.get("text") == "Open the admitted call"] == []
    assert len(seen) == 1 and runtime.restart._AGENT_RESTART["started"][seen[0]] == seen[0]
    assert not any("Saved for execution" in str(args[1]) for args, _ in direct_call.notices)
    assert not any("Error:" in str(args[1]) for args, _ in direct_call.notices)


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
