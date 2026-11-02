"""Answers keep their addressee through holds, Continue, last drains and crashes."""
import queue
from types import SimpleNamespace

import pytest

from ouroboros.gateway import task_decision as td
from ouroboros.owner_quiz import record_asked, record_publication, quiz_states
from ouroboros.task_store import patch_document
from ouroboros.question_answers import admitted_successor
from tests.test_quiz_answer import _post
from tests.test_quiz_late_answer import _late_bridge, _inbox


def runtime(tmp_path, monkeypatch, *, running=True):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from supervisor import queue as q
    task = {"id": "asker", "chat_id": 1, "drive_root": str(tmp_path)}
    monkeypatch.setattr(q, "RUNNING", {"asker": {"task": task}} if running else {})
    monkeypatch.setattr(q, "PENDING", [])
    monkeypatch.setattr(q, "BUDGET_ROOT_FENCES", {})
    monkeypatch.setattr(q, "_task_drive_for_task", lambda task, tid: tmp_path)
    monkeypatch.setattr(td, "request_drive_root", lambda request: tmp_path)
    from ouroboros.task_results import write_task_result
    write_task_result(tmp_path, "asker", status="running" if running else "completed", chat_id=1)
    record_asked(tmp_path, "asker", quiz_id="q", question="Which database?", options=["sqlite", "postgres"],
                 assumption="sqlite", chat_id=1)
    record_publication(tmp_path, "asker", "q", 1)
    bridge, frames = _late_bridge(tmp_path, monkeypatch)
    app = Starlette(routes=[Route("/api/decisions", endpoint=td.api_decision_answer, methods=["POST"])])
    return app, bridge, q, task


def answer(app):
    return _post(app, {"decision_id": "quiz:asker:q", "request_id": "r", "option_index": 1, "comment": "exact answer"})


def debt(root):
    return quiz_states(root, "asker")["q"]["delivery"]


def drain(root, task_id):
    from ouroboros.loop_round_limits import _drain_incoming_messages
    ctx = SimpleNamespace(task_id=task_id, task_metadata={}, drive_root=root, budget_drive_root=root)
    messages = []
    _drain_incoming_messages(messages, queue.Queue(), root, task_id, None, set(), owner_ctx=ctx)
    return ctx, messages


def continue_to(root, state="admitted"):
    from ouroboros.owner_continue import binding_sha
    from ouroboros.task_results import write_task_result
    write_task_result(root, "successor", status="running", chat_id=1)
    binding = {"predecessor_task_id": "asker", "successor_task_id": "successor", "action_nonce": "nonce"}
    admission = {"binding": binding, "binding_sha256": binding_sha(binding)}
    patch_document(root, "asker", lambda row: {**row, "continued_by": {
        **admission, "state": state, "successor_task_id": "successor"}})
    patch_document(root, "successor", lambda row: {**row, "continuation_admission": admission})


def test_live_answer_is_owed_until_saved_read_and_keeps_only_owner_words(tmp_path, monkeypatch):
    app, bridge, q, task = runtime(tmp_path, monkeypatch)
    response = answer(app)
    assert response.status_code == 200, response.text
    assert debt(tmp_path)["state"] == "owed"
    assert debt(tmp_path)["recipient_task_id"] == "asker" and debt(tmp_path)["msg_id"]
    ctx, messages = drain(tmp_path, "asker")
    assert debt(tmp_path)["state"] == "delivered"
    assert "Which database?" in str(messages) and "postgres" in str(messages)
    assert [r["content"] for r in ctx._owner_directives] == ["exact answer"]
    from ouroboros.owner_relay import quote_rows
    [handle] = ctx._held_owner_messages
    [quoted], error = quote_rows(ctx, [handle])
    assert not error and quoted["text"] == "exact answer"
    assert "Which database?" in quoted["question_frame"]
    assert _inbox(bridge) == []


def test_only_admitted_successor_can_receive_an_answer(tmp_path, monkeypatch):
    app, bridge, q, task = runtime(tmp_path, monkeypatch, running=False)
    continue_to(tmp_path)
    q.RUNNING["successor"] = {"task": {**task, "id": "successor"}}
    response = answer(app)
    assert response.status_code == 200, response.text
    assert debt(tmp_path)["recipient_task_id"] == "successor"
    assert _inbox(bridge) == []
    drain(tmp_path, "successor")
    assert debt(tmp_path)["state"] == "delivered"


def test_bound_claim_and_unrelated_room_reader_cannot_take_the_answer(tmp_path, monkeypatch):
    app, bridge, q, task = runtime(tmp_path, monkeypatch, running=False)
    continue_to(tmp_path, "bound")
    q.RUNNING["successor"] = {"task": {**task, "id": "successor"}}
    assert admitted_successor(tmp_path, "asker") == ""
    response = answer(app)
    assert response.status_code == 200 and response.json()["forwarded"] is True, response.text
    [message] = _inbox(bridge)
    assert message["task_metadata"]["late_answer"]["task_id"] == "asker"
    assert message["chat_id"] == 1 and debt(tmp_path)["route"] == "fresh_head"
    assert answer(app).status_code == 200 and _inbox(bridge) == []


def test_held_asker_keeps_its_hold_and_holds_its_answer(tmp_path, monkeypatch):
    """A living asker that reads no mail now still holds its answer: no new head, the hold stays,
    and the card says the task reads it when it continues; once released it reads it once."""
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.project_dialogue import QUESTION_STATUS
    app, bridge, q, task = runtime(tmp_path, monkeypatch)
    task["_owner_hold"] = True
    response = answer(app)
    assert response.status_code == 200, response.text
    assert task["_owner_hold"] is True and _inbox(bridge) == []
    assert [entry["msg_id"] for entry in drain_owner_entries(tmp_path, "asker")] == ["quiz_answer:asker:q"]
    assert debt(tmp_path)["recipient_task_id"] == "asker"
    assert debt(tmp_path)["reason_code"] == "awaiting_recipient_read"
    assert QUESTION_STATUS["answer_saved"] == "Answer saved · the task reads it when it continues"
    task.pop("_owner_hold")
    ctx, _messages = drain(tmp_path, "asker")
    assert [r["content"] for r in ctx._owner_directives] == ["exact answer"]
    assert debt(tmp_path)["state"] == "delivered"


@pytest.mark.parametrize("held", [False, True])
def test_last_drain_race_recovers_unread_answer_once(tmp_path, monkeypatch, held):
    """The recipient ended before reading its answer (a held one was stopped): exactly one fresh head."""
    app, bridge, q, task = runtime(tmp_path, monkeypatch)
    if held:
        task["_owner_hold"] = True
    assert answer(app).status_code == 200
    assert debt(tmp_path)["state"] == "owed"
    q.RUNNING.clear()
    patch_document(tmp_path, "asker", lambda row: {**row, "status": "completed"})
    response = answer(app)
    assert response.status_code == 200 and response.json()["forwarded"], response.text
    assert len(_inbox(bridge)) == 1
    assert answer(app).status_code == 200 and not _inbox(bridge)


def test_crash_after_mailbox_append_retries_same_message_identity(tmp_path, monkeypatch):
    from ouroboros import question_answers as qa
    from ouroboros.owner_mailbox import drain_owner_entries
    app, bridge, q, task = runtime(tmp_path, monkeypatch)
    real = qa._update_delivery
    def lose_ack(root, asker, quiz, changes):
        if changes.get("mailbox_at"):
            raise OSError("crash after append")
        return real(root, asker, quiz, changes)
    monkeypatch.setattr(qa, "_update_delivery", lose_ack)
    assert answer(app).status_code == 503
    identity = debt(tmp_path)["msg_id"]
    monkeypatch.setattr(qa, "_update_delivery", real)
    assert answer(app).status_code == 200
    assert [r["msg_id"] for r in drain_owner_entries(tmp_path, "asker")] == [identity]
    drain(tmp_path, "asker")
    assert debt(tmp_path)["state"] == "delivered" and not _inbox(bridge)


def sleeping_asker(tmp_path, monkeypatch, hold=""):
    """An asker parked in its own cold sleep (the rail ``continue_in_background`` uses)."""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from ouroboros.owner_pause import install_fence
    from supervisor.events_budget import HOLD_OWNER_RESTART, hold_budget_row
    from tests._budget_pause_exact_helpers import _install_queue
    from tests.test_model_sleep import _cold_park
    q, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    _cold_park(tmp_path, monkeypatch, workers, task_id="asker", wake_at="2099-01-01T00:00:00+00:00")
    assert [row["_budget_pause"]["reason"] for row in q.PENDING] == ["sleep"]
    if hold == "restart_hold":
        hold_budget_row(q.PENDING[0], reason=HOLD_OWNER_RESTART, detail="owner Restart", result_root=tmp_path)
    elif hold == "owner_pause":
        install_fence(tmp_path, "asker", request_id="owner-pause")
    record_asked(tmp_path, "asker", quiz_id="q", question="Which database?", options=["sqlite", "postgres"],
                 assumption="sqlite", chat_id=1)
    record_publication(tmp_path, "asker", "q", 1)
    bridge, _frames = _late_bridge(tmp_path, monkeypatch)
    monkeypatch.setattr(td, "request_drive_root", lambda request: tmp_path)
    return Starlette(routes=[Route("/api/decisions", endpoint=td.api_decision_answer, methods=["POST"])]), bridge, q


def mail(root, kind):
    from ouroboros.owner_mailbox import drain_owner_entries
    return [entry["msg_id"] for entry in drain_owner_entries(root, "asker") if entry["kind"] == kind]


@pytest.mark.parametrize("hold", ["", "restart_hold", "owner_pause"])
def test_cold_sleeping_asker_holds_its_own_answer_and_wakes_only_unheld(tmp_path, monkeypatch, hold):
    from ouroboros.budget_pause import budget_pause_row
    from ouroboros.owner_pause import fence_closed, read_fence
    from supervisor.events_budget import budget_hold_fact
    from supervisor.sleep_wake import wake_ready_sleepers
    app, bridge, q = sleeping_asker(tmp_path, monkeypatch, hold)
    response = answer(app)
    assert response.status_code == 200, response.text
    woke = [outcome for outcome in wake_ready_sleepers(q) if outcome.get("ok")]
    if not hold:
        # Its own answer is its mail: one copy, no second head, and its readiness wakes it.
        assert mail(tmp_path, "quiz_answer") == ["quiz_answer:asker:q"] and _inbox(bridge) == []
        assert debt(tmp_path)["recipient_task_id"] == "asker"
        assert woke and budget_pause_row(tmp_path, "asker")["sleep_ready"]["reason"] == "owner_text"
        return
    # A held or owner-paused sleeper holds its answer unread: no fresh head, no wake, the hold stays.
    assert mail(tmp_path, "quiz_answer") == ["quiz_answer:asker:q"] and _inbox(bridge) == []
    assert debt(tmp_path)["recipient_task_id"] == "asker" and not woke
    if hold == "restart_hold":
        assert budget_hold_fact(q.PENDING[0])["reason"] == "owner_restart_hold"
    else:
        assert fence_closed(read_fence(tmp_path, "asker"))


@pytest.mark.parametrize("held", [False, True])
def test_cold_sleeping_asker_learns_the_owner_closure_and_wakes_only_unheld(tmp_path, monkeypatch, held):
    from supervisor.sleep_wake import wake_ready_sleepers
    app, _bridge, q = sleeping_asker(tmp_path, monkeypatch, "restart_hold" if held else "")
    response = _post(app, {"decision_id": "quiz:asker:q", "request_id": "close", "action": "close"})
    assert response.status_code == 200, response.text
    # A held sleeper finds the closure in its mailbox when it continues; only an unheld one wakes now.
    assert mail(tmp_path, "quiz_closed") == ["quiz_closed:q"]
    assert any(outcome.get("ok") for outcome in wake_ready_sleepers(q)) is not held


@pytest.mark.parametrize("pause", ["owner_hold", "money_pause", "queued"])
def test_retried_answer_stays_owed_to_its_unread_live_recipient(tmp_path, monkeypatch, pause):
    app, bridge, q, task = runtime(tmp_path, monkeypatch)
    assert answer(app).status_code == 200
    assert debt(tmp_path)["recipient_task_id"] == "asker" and debt(tmp_path)["mailbox_at"]
    # Paused (or parked) after the mailbox write, before its next drain.
    if pause == "queued":
        q.PENDING.append(q.RUNNING.pop("asker")["task"])
    else:
        task.update({"_owner_hold": True} if pause == "owner_hold" else {"_budget_pause": {"reason": "budget"}})
    response = answer(app)  # the same request id, retried
    assert response.status_code == 200, response.text
    assert _inbox(bridge) == [] and mail(tmp_path, "quiz_answer") == ["quiz_answer:asker:q"]
    assert debt(tmp_path)["recipient_task_id"] == "asker"
    assert debt(tmp_path)["reason_code"] == "awaiting_recipient_read"
    # Resume: the same work reads its one copy, exactly once.
    task.pop("_owner_hold", None)
    task.pop("_budget_pause", None)
    if pause == "queued":
        q.RUNNING["asker"] = {"task": q.PENDING.pop()}
    ctx, _messages = drain(tmp_path, "asker")
    assert [r["content"] for r in ctx._owner_directives] == ["exact answer"]
    assert debt(tmp_path)["state"] == "delivered"
    assert answer(app).status_code == 200 and _inbox(bridge) == []
    assert getattr(drain(tmp_path, "asker")[0], "_owner_directives", []) == []


def test_paused_recipient_keeps_its_saved_answer_and_reads_it_once(tmp_path, monkeypatch):
    app, bridge, q, task = runtime(tmp_path, monkeypatch)
    assert answer(app).status_code == 200
    task["_owner_hold"] = True
    assert answer(app).status_code == 200, "a retried answer stays with the work that holds its copy"
    assert _inbox(bridge) == [] and debt(tmp_path)["recipient_task_id"] == "asker"
    assert task["_owner_hold"] is True
    task.pop("_owner_hold")
    ctx, _ = drain(tmp_path, "asker")
    assert [r["content"] for r in ctx._owner_directives] == ["exact answer"]
    assert answer(app).status_code == 200 and _inbox(bridge) == []
    assert getattr(drain(tmp_path, "asker")[0], "_owner_directives", []) == []


def due(root):
    """The persisted backoff has passed: maintenance retries the owed answer now."""
    from ouroboros.owner_quiz import _mutate_projection

    def advance(quizzes):
        quizzes["q"]["delivery"]["next_retry_at"] = ""
        return True
    _mutate_projection(root, "asker", advance)


@pytest.mark.parametrize("retry", ["owner", "maintenance"])
@pytest.mark.parametrize("ingress", ["accepted", "accepted_before_restart", "refused"])
def test_a_late_answer_retried_after_a_continue_keeps_one_effect(tmp_path, monkeypatch, ingress, retry):
    """The late path ran, then the owner continued the asker. An answer the owner's ingress accepted, whose
    delivery mark was then lost, converges on that message: the successor gets no second copy, and after a
    restart its delivery stays unknown. One the ingress never accepted still reaches the successor."""
    import asyncio

    import supervisor.message_bus as mb
    from ouroboros import owner_quiz, process_custody
    from ouroboros.owner_mailbox import drain_owner_entries

    app, bridge, q, task = runtime(tmp_path, monkeypatch, running=False)
    accepted = ingress != "refused"
    with monkeypatch.context() as lost:
        if accepted:  # the ingress accepted it, then the delivery mark was lost
            lost.setattr(owner_quiz, "mark_discharged", lambda *a, **k: {})
        else:  # the ingress failed before anything landed
            lost.setattr(mb, "accept_chat", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("not persisted")))
        assert answer(app).status_code == (200 if accepted else 503)
    assert debt(tmp_path)["state"] == "owed" and debt(tmp_path)["route"] == "fresh_head"
    assert len(_inbox(bridge)) == int(accepted)

    if ingress == "accepted_before_restart":  # the process that handed its dispatch over has ended
        monkeypatch.setattr(process_custody, "current_custody_session_id", lambda: "next-generation")
    continue_to(tmp_path)
    q.RUNNING["successor"] = {"task": {**task, "id": "successor"}}
    if retry == "owner":
        assert answer(app).status_code == 200, "the same request id"
    else:
        due(tmp_path)
        asyncio.run(td.recover_question_deliveries(tmp_path))
    copies = [entry["msg_id"] for entry in drain_owner_entries(tmp_path, "successor", set())]
    assert _inbox(bridge) == []
    if ingress == "accepted":
        assert copies == [], "one answer, one effect: the owner's message the ingress accepted"
        assert debt(tmp_path)["state"] == "delivered"
    elif ingress == "accepted_before_restart":
        assert copies == [], "an accepted answer of unknown dispatch is never delivered again elsewhere"
        assert (debt(tmp_path)["state"], debt(tmp_path)["reason_code"]) == ("owed", "late_answer_delivery_unknown")
    else:
        assert copies == ["quiz_answer:asker:q"] and debt(tmp_path)["recipient_task_id"] == "successor"


def test_an_appended_answer_whose_receipt_failed_stays_with_its_paused_recipient(tmp_path, monkeypatch):
    from ouroboros import question_answers as qa
    app, bridge, q, task = runtime(tmp_path, monkeypatch)
    real = qa._update_delivery
    def lose_receipt(root, asker, quiz, changes):
        if changes.get("mailbox_at"):
            raise OSError("mailbox append saved; delivery receipt unavailable")
        return real(root, asker, quiz, changes)
    monkeypatch.setattr(qa, "_update_delivery", lose_receipt)
    assert answer(app).status_code == 503
    monkeypatch.setattr(qa, "_update_delivery", real)
    task["_owner_hold"] = True
    assert answer(app).status_code == 200
    assert _inbox(bridge) == [], "the copy in its mailbox is the receipt: no fresh head"
    assert debt(tmp_path)["mailbox_at"], "the missing receipt is repaired from the mailbox"
    assert task["_owner_hold"] is True
    task.pop("_owner_hold")
    ctx, _ = drain(tmp_path, "asker")
    assert [r["content"] for r in ctx._owner_directives] == ["exact answer"]
