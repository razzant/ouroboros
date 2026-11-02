"""Publication and closure use canonical receipts before exposing delivery success."""

from tests import _dialogue as dialogue_test
import json
from types import SimpleNamespace

import pytest

from ouroboros import owner_quiz
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.utils import append_jsonl
from supervisor import events_chat_delivery as handlers, message_bus


@pytest.fixture
def publication(tmp_path, monkeypatch):
    write_task_result(tmp_path, "t", "running", chat_id=1)
    owner_quiz.record_asked(tmp_path, "t", quiz_id="q", question="Which approach?",
                            options=["First", "Second"], assumption="Continue research", chat_id=1)
    bridge = message_bus.LocalChatBridge({})
    frames, events = [], []
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 1})
    monkeypatch.setattr(message_bus, "publish_event", lambda *args: events.append(args))
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, bridge=bridge, append_jsonl=append_jsonl)
    event = {"type": "send_quiz", "task_id": "t", "quiz_id": "q", "chat_id": 1,
             "question": "Which approach?", "options": [{"label": "First"}, {"label": "Second"}],
             "assumption": "Continue research"}
    return SimpleNamespace(root=tmp_path, ctx=ctx, event=event, frames=frames, events=events)


def _block(publication):
    return owner_quiz.quiz_states(publication.root, "t")["q"]


def _due(publication):
    row = load_task_result(publication.root, "t")
    row["owner_quiz"]["q"]["delivery"]["next_retry_at"] = "2000-01-01T00:00:00+00:00"
    write_task_result(publication.root, "t", row["status"], owner_quiz=row["owner_quiz"])


@pytest.mark.parametrize("end", ["terminal", "timeout"])
@pytest.mark.parametrize("assumption", ["", "Continue reading only"])
def test_required_publication_recovers_after_wait_end(tmp_path, monkeypatch, end, assumption):
    import asyncio
    from ouroboros import dialogue_store
    from ouroboros.gateway.task_decision import recover_question_deliveries
    from ouroboros.owner_wait import wait_after_tools
    from tests.test_native_owner_wait import native_context

    ctx = native_context(tmp_path)
    ctx.current_chat_id = 1
    origin = {"task_id": ctx.task_id, "binding_id": "source-place", "source_event_id": "external-event"}
    owner_quiz.record_asked(tmp_path, ctx.task_id, quiz_id="q1", question="May I proceed?",
        options=[], assumption=assumption, wait_for_answer=True, chat_id=1, presence_contact=origin)
    bridge, frames = message_bus.LocalChatBridge({}), []
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    monkeypatch.setattr(message_bus, "publish_event", lambda *_: None)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 1})
    host = SimpleNamespace(DRIVE_ROOT=tmp_path, bridge=bridge, append_jsonl=append_jsonl)
    event = {"type": "send_quiz", "task_id": ctx.task_id, "quiz_id": "q1", "chat_id": 1,
             "question": "May I proceed?", "options": [], "assumption": assumption, "wait_for_answer": True}

    def unavailable(*_a, **_kw):
        raise dialogue_store.StoreError("fixture publication unavailable")

    with monkeypatch.context() as scoped:
        scoped.setattr(dialogue_store, "append", unavailable)
        handlers._handle_send_quiz(event, host)
    before = owner_quiz.quiz_states(tmp_path, ctx.task_id)["q1"]
    assert before["delivery"]["state"] == "owed" and not before.get("published_at")
    assert not frames
    if end == "terminal":
        write_task_result(tmp_path, ctx.task_id, "completed")
        assert owner_quiz.reconcile_terminal(tmp_path, ctx.task_id) == ["q1"]
    else:
        ctx._owner_wait_deadline_at, ctx._owner_wait_max_minutes = "2000-01-01T00:00:00+00:00", 1
        messages = []
        wait_after_tools(ctx, messages, {}, {}, 1, [], set())
        wait = load_task_result(tmp_path, ctx.task_id)["owner_wait"]
        assert wait["state"] == "resumed" and wait["resume_reason"] == "timeout"
        if not assumption:
            assert "No assumption was recorded; no answer is not consent" in str(messages)
    ended = owner_quiz.quiz_states(tmp_path, ctx.task_id)["q1"]
    assert ended["wait_ended_at"] and not ended.get("wait_for_answer")
    assert not ended.get("published_at") and ended["assumption"] == assumption
    def due(quizzes):
        quizzes["q1"]["delivery"]["next_retry_at"] = ""
        return True
    owner_quiz._mutate_projection(tmp_path, ctx.task_id, due)
    assert asyncio.run(recover_question_deliveries(tmp_path)) == {"attempted": 1, "delivered": 1, "pending": 0}
    after = owner_quiz.quiz_states(tmp_path, ctx.task_id)["q1"]
    assert after["published_at"] and after["delivery"]["state"] == "delivered"
    assert after["assumption"] == assumption and owner_quiz.current_question(after)
    assert after["state"] == ("expired_terminal" if end == "terminal" else "open")
    [frame] = [row for row in frames if row["type"] == "quiz"]
    assert frame["wait_for_answer"] is False and frame["assumption"] == assumption
    assert frame["wait_ended_at"] == ended["wait_ended_at"]
    assert frame["presence_contact"] == origin and frame["chat_id"] == 1
    assert asyncio.run(recover_question_deliveries(tmp_path))["attempted"] == 0


def test_strict_publication_failure_keeps_timed_debt_then_exposes_delivered_fact(publication, monkeypatch):
    from ouroboros import dialogue_store as ds

    real_append = ds.append
    observed = []

    def refuse(*_args, **_kwargs):
        block = _block(publication)
        observed.append(block["delivery"]["attempts"])
        assert block["publication_chat_id"] == 1
        assert block["publication_evidence"] == "pre_exposure"
        raise ds.StoreError("injected publication write failure")

    monkeypatch.setattr(ds, "append", refuse)
    handlers._handle_send_quiz(publication.event, publication.ctx)
    block = _block(publication)
    assert observed == [1] and not publication.frames and not publication.events
    assert block["delivery"]["state"] == "owed"
    assert block["delivery"]["reason_code"] == "publication_write_failed"
    assert not block.get("published_at")
    # A duplicate event must not bypass the persisted retry delay.
    handlers._handle_send_quiz(publication.event, publication.ctx)
    assert observed == [1] and _block(publication)["delivery"] == block["delivery"]
    monkeypatch.setattr(ds, "append", real_append)
    _due(publication)
    handlers._handle_send_quiz({**publication.event, "chat_id": 73}, publication.ctx)
    assert _block(publication)["delivery"]["attempts"] == 2
    assert _block(publication)["published_at"]
    [frame] = publication.frames
    assert frame["chat_id"] == 1  # The failed write already bound the address.
    assert frame["question_delivery"] == {"kind": "publication", "state": "delivered"}
    assert publication.events[0][1]["question_delivery"] == frame["question_delivery"]
    [row] = dialogue_test.log_rows(publication.root / "logs/chat.jsonl")
    assert row["client_message_id"] == "quiz_publication:t:q"
    assert row["quiz"]["question_delivery"]["state"] == "owed"  # Snapshot before durable append.
    from ouroboros.gateway.history import _assemble_history_response
    replay = json.loads(_assemble_history_response(publication.root, 1, 10, 0))["messages"]
    assert replay[0]["quiz"]["question_delivery"] == frame["question_delivery"]
    handlers._handle_send_quiz(publication.event, publication.ctx)
    assert len(publication.frames) == 1


@pytest.mark.parametrize("chat_id, reason", [(None, "chat_unresolved"), (-5, "a2a_chat")])
def test_unrouteable_publication_claims_retry_before_return(publication, chat_id, reason):
    handlers._handle_send_quiz({**publication.event, "chat_id": chat_id}, publication.ctx)
    delivery = _block(publication)["delivery"]
    assert delivery["attempts"] == 1 and delivery["next_retry_at"]
    assert delivery["state"] == "owed" and delivery["reason_code"] == reason
    assert not publication.frames and not _block(publication).get("published_at")


def test_bridge_refusal_keeps_publication_debt_and_stale_ask_does_not_claim_answer(publication):
    publication.ctx.bridge = SimpleNamespace(send_quiz=lambda *a, **k: (False, "unavailable"))
    handlers._handle_send_quiz(publication.event, publication.ctx)
    assert _block(publication)["delivery"]["reason_code"] == "publication_send_failed"
    assert owner_quiz.record_answered(publication.root, "t", quiz_id="q", option_index=1,
                                     request_id="owner")["ok"]
    before = _block(publication)["delivery"]
    handlers._handle_send_quiz(publication.event, publication.ctx)
    assert _block(publication)["delivery"] == before
    assert before["kind"] == "answer" and before["attempts"] == 0


@pytest.mark.parametrize("address,evidence", [(True, "pre_exposure"), (1.5, "history"),
                                             ("1", "history"), (1, "timestamps")])
def test_corrupt_saved_publication_is_a_gap_for_publish_and_late_answer(publication, address, evidence):
    from ouroboros.gateway.task_decision import _late_answer_destination

    row = load_task_result(publication.root, "t")
    row["owner_quiz"]["q"].update(publication_chat_id=address, publication_evidence=evidence)
    write_task_result(publication.root, "t", row["status"], owner_quiz=row["owner_quiz"])
    handlers._handle_send_quiz(publication.event, publication.ctx)
    block = _block(publication)
    assert not publication.frames and not publication.events
    assert block["delivery"]["state"] == "owed"
    assert block["delivery"]["reason_code"] == "publication_address_unknown"
    assert _late_answer_destination(publication.root, "t", block) == (None, "publication_address_unknown")
    assert block["publication_chat_id"] == address


def test_closure_without_publication_witness_never_guesses_from_ask_chat(publication):
    # A genuinely legacy question has no canonical new-publication entitlement.
    row = load_task_result(publication.root, "t")
    row["owner_quiz"]["q"].pop("delivery")
    write_task_result(publication.root, "t", row["status"], owner_quiz=row["owner_quiz"])
    owner_quiz.record_closed(publication.root, "t", quiz_id="q", reason="Superseded", actor_task_id="t")
    handlers._handle_send_quiz_state(publication.event, publication.ctx)
    block = _block(publication)
    assert block["delivery"]["reason_code"] == "publication_address_unknown"
    assert block["delivery"]["attempts"] == 1 and not block.get("discharged_at")
    assert not publication.frames and not dialogue_test.log_rows(publication.root / "logs/chat.jsonl")


def test_closure_ack_failure_reuses_durable_history_receipt(publication, monkeypatch):
    owner_quiz.record_publication(publication.root, "t", "q", 1)
    owner_quiz.record_closed(publication.root, "t", quiz_id="q", reason="Superseded", actor_task_id="t")
    real_discharge = owner_quiz.mark_discharged
    monkeypatch.setattr(owner_quiz, "mark_discharged", lambda *_a: (_ for _ in ()).throw(OSError("ack failed")))
    with pytest.raises(OSError, match="ack failed"):
        handlers._handle_send_quiz_state(publication.event, publication.ctx)
    block = _block(publication)
    assert block["delivery"]["history_at"] and not block.get("discharged_at")
    assert not publication.frames
    first_history = dialogue_test.log_text(publication.root / "logs/chat.jsonl")
    handlers._handle_send_quiz_state(publication.event, publication.ctx)  # timer gates repeat effects
    assert _block(publication)["delivery"]["attempts"] == 1
    _due(publication)
    monkeypatch.setattr(owner_quiz, "mark_discharged", real_discharge)
    handlers._handle_send_quiz_state(publication.event, publication.ctx)
    assert dialogue_test.log_text(publication.root / "logs/chat.jsonl") == first_history
    assert _block(publication)["discharged_at"]
    [frame] = publication.frames
    assert frame["state"] == "closed" and frame["closed_reason"] == "Superseded"
    assert frame["question_delivery"] == {"kind": "closure", "state": "delivered"}
    assert publication.events[0][1]["question_delivery"] == frame["question_delivery"]
