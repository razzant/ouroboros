"""Question answers follow the actual admitted lineage across its final drain."""

import copy

import pytest

from ouroboros.task_results import load_task_result, write_task_result
from tests.test_owner_continue import _interrupted
from tests.test_project_hold_recovery import worker
from tests.test_question_answer_routing import debt, drain
from tests.test_quiz_late_answer import _inbox, _late_bridge
from tests.test_swarm_host_admission import host  # noqa: F401

pytestmark = pytest.mark.serial


@pytest.mark.parametrize("destination,cut", [
    ("asker", "none"), ("fresh_head", "before"), ("fresh_head", "selected"),
    ("successor", "before"), ("successor", "selected"), ("successor", "appended"),
    ("held_successor", "selected"),
])
def test_answer_follows_reader_at_finish_without_creating_a_successor(host, monkeypatch, destination, cut):  # noqa: F811
    from ouroboros import question_answers as qa
    from ouroboros.gateway.task_decision import answer_decision, recover_question_deliveries
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.owner_pause import read_fence
    from ouroboros.owner_quiz import record_asked, record_publication
    from supervisor import message_bus, workers
    from supervisor.continuation_admission import admit_continuation
    from supervisor.owner_pause_control import request_owner_pause
    from supervisor.queue_transitions import reconcile_terminal_task_projections

    task = {"id": "asker", "root_task_id": "asker", "type": "task", "chat_id": 1, "_attempt": 1}
    host.running["asker"] = {"task": task}
    write_task_result(host.root, "asker", "running", root_task_id="asker", chat_id=1)
    record_asked(host.root, "asker", quiz_id="q", question="Which database?", options=["sqlite", "postgres"],
                 assumption="sqlite", chat_id=1)
    record_publication(host.root, "asker", "q", 1)
    bridge, _frames = _late_bridge(host.root, monkeypatch)
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    sent = worker(host, monkeypatch)
    successor, held = "", None

    def finish():
        nonlocal successor, held
        host.running.pop("asker")
        if destination in {"successor", "held_successor"}:
            _interrupted(host.root, "asker")
            reconcile_terminal_task_projections(host.root, "asker")
            result = admit_continuation("asker", action_nonce="already-requested")
            assert result["ok"] and not result["held"], result
            successor = result["successor_task_id"]
            if destination == "held_successor":
                assert request_owner_pause(successor, request_id="owner-hold")["ok"]
                held = copy.deepcopy(read_fence(host.root, successor))
            workers.assign_tasks()
            assert len(sent) == (0 if held else 1)
        else:
            write_task_result(host.root, "asker", "completed", result="finished")
            reconcile_terminal_task_projections(host.root, "asker")

    if cut == "before":
        finish()
    elif cut == "selected":
        real_select = qa.reading_recipient
        selected = False

        def select(*args):
            nonlocal selected
            result = real_select(*args)
            if not selected:
                selected = True
                assert result[0] == "asker"
                finish()
            return result

        monkeypatch.setattr(qa, "reading_recipient", select)

    body = {"decision_id": "quiz:asker:q", "request_id": "answer-1", "option_index": 1, "comment": "exact answer"}
    status, response = host.run_async(answer_decision(host.root, body))
    assert status == 200, response
    if cut == "appended":
        assert debt(host.root)["recipient_task_id"] == "asker"
        finish()
        host.run_async(recover_question_deliveries(host.root))
        # Recovery retry observes the ended recipient and its existing successor.
        status, response = host.run_async(answer_decision(host.root, body))
        assert status == 200, response
    recipient = ("asker" if destination == "asker" else successor
                 if destination in {"successor", "held_successor"} else "")
    assert debt(host.root)["recipient_task_id"] == recipient
    if held:
        # A held successor holds the answer unread until it continues; no head relays it.
        assert not _inbox(bridge) and debt(host.root)["reason_code"] == "awaiting_recipient_read"
        assert [entry["msg_id"] for entry in drain_owner_entries(host.root, successor)] == [debt(host.root)["msg_id"]]
    elif recipient:
        assert not _inbox(bridge)
        ctx, _messages = drain(host.root, recipient)
        assert [r["content"] for r in ctx._owner_directives] == ["exact answer"]
        assert debt(host.root)["state"] == "delivered"
    else:
        [message] = _inbox(bridge)
        assert message["task_metadata"]["late_answer"]["task_id"] == "asker"
        assert message["chat_id"] == 1 and debt(host.root)["route"] == "fresh_head"
        assert not drain_owner_entries(host.root, "asker")
    if held:
        assert read_fence(host.root, successor) == held and not sent
    assert host.run_async(answer_decision(host.root, body))[0] == 200
    assert not _inbox(bridge)
    if successor:
        assert load_task_result(host.root, "asker")["continued_by"]["successor_task_id"] == successor
        assert set(host.running) | {row["id"] for row in host.pending} == {successor}
        refused = admit_continuation("asker", action_nonce="second-press")
        assert refused["error"] == "already_continued" and refused["successor_task_id"] == successor
    else:
        assert "continued_by" not in load_task_result(host.root, "asker")


def test_owner_pause_after_delivery_keeps_the_answer_for_its_own_resume(host, monkeypatch):  # noqa: F811
    from ouroboros.gateway.task_decision import answer_decision, recover_question_deliveries
    from ouroboros.owner_mailbox import KIND_QUIZ_ANSWER, drain_owner_entries
    from ouroboros.owner_pause import release_fence
    from ouroboros.owner_quiz import record_asked, record_publication
    from supervisor import message_bus
    from supervisor.owner_pause_control import request_owner_pause

    host.running["asker"] = {"task": {"id": "asker", "root_task_id": "asker", "type": "task", "chat_id": 1,
                                      "_attempt": 1}}
    write_task_result(host.root, "asker", "running", root_task_id="asker", chat_id=1)
    record_asked(host.root, "asker", quiz_id="q", question="Which database?", options=["sqlite", "postgres"],
                 assumption="sqlite", chat_id=1)
    record_publication(host.root, "asker", "q", 1)
    bridge, _frames = _late_bridge(host.root, monkeypatch)
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    body = {"decision_id": "quiz:asker:q", "request_id": "answer-1", "option_index": 1, "comment": "exact answer"}
    assert host.run_async(answer_decision(host.root, body))[0] == 200  # delivered while it was busy
    assert request_owner_pause("asker", request_id="owner-pause")["ok"]
    assert host.run_async(answer_decision(host.root, body))[0] == 200  # the same request, retried
    host.run_async(recover_question_deliveries(host.root))
    # Pause moves the answer nowhere: no fresh head, no second copy, still owed to the paused work.
    assert not _inbox(bridge) and debt(host.root)["recipient_task_id"] == "asker"
    assert [entry["msg_id"] for entry in drain_owner_entries(host.root, "asker")
            if entry["kind"] == KIND_QUIZ_ANSWER] == ["quiz_answer:asker:q"]
    release_fence(host.root, "asker", reason="owner_resume")  # where the resumed root starts
    ctx, _messages = drain(host.root, "asker")
    assert [r["content"] for r in ctx._owner_directives] == ["exact answer"]
    assert debt(host.root)["state"] == "delivered"
    assert host.run_async(answer_decision(host.root, body))[0] == 200 and not _inbox(bridge)
    assert getattr(drain(host.root, "asker")[0], "_owner_directives", []) == []


def test_a_receipt_lost_after_append_survives_a_real_owner_pause_without_a_second_reader(host, monkeypatch):  # noqa: F811
    from ouroboros import question_answers as qa
    from ouroboros.gateway.task_decision import answer_decision
    from ouroboros.owner_pause import read_fence, fence_closed, release_fence
    from ouroboros.owner_quiz import record_asked, record_publication
    from supervisor.owner_pause_control import request_owner_pause
    task = {"id": "asker", "root_task_id": "asker", "type": "task", "chat_id": 1, "_attempt": 1}
    host.running["asker"] = {"task": task}
    write_task_result(host.root, "asker", "running", root_task_id="asker", chat_id=1)
    record_asked(host.root, "asker", quiz_id="q", question="Which database?", options=["sqlite", "postgres"],
                 assumption="sqlite", chat_id=1)
    record_publication(host.root, "asker", "q", 1)
    bridge, _ = _late_bridge(host.root, monkeypatch)
    monkeypatch.setattr("supervisor.message_bus.get_bridge", lambda: bridge)
    body = {"decision_id": "quiz:asker:q", "request_id": "r", "option_index": 1, "comment": "exact answer"}
    real = qa._update_delivery
    def lose_receipt(root, asker, quiz, changes):
        if changes.get("mailbox_at"):
            raise OSError("mailbox append saved; delivery receipt unavailable")
        return real(root, asker, quiz, changes)
    monkeypatch.setattr(qa, "_update_delivery", lose_receipt)
    assert host.run_async(answer_decision(host.root, body))[0] == 503
    monkeypatch.setattr(qa, "_update_delivery", real)
    paused = request_owner_pause("asker", request_id="owner-pause")
    assert paused["ok"], paused
    assert host.run_async(answer_decision(host.root, body))[0] == 200
    assert _inbox(bridge) == [], "the paused work keeps its saved copy; nobody else reads it"
    assert fence_closed(read_fence(host.root, "asker"))
    release_fence(host.root, "asker", reason="owner_resume")
    ctx, _ = drain(host.root, "asker")
    assert [r["content"] for r in getattr(ctx, "_owner_directives", [])] == ["exact answer"]
