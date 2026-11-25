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
    from ouroboros.question_answers import MAILBOX_ROUTE_FACT
    assert str(messages).count(MAILBOX_ROUTE_FACT) == 1  # at delivery the letter states its own route once
    assert [r["content"] for r in ctx._owner_directives] == ["exact answer"]
    from ouroboros.owner_relay import quote_rows
    [handle] = ctx._held_owner_messages
    [quoted], error = quote_rows(ctx, [handle])
    assert not error and quoted["text"] == "exact answer"
    assert "Which database?" in quoted["question_frame"] and MAILBOX_ROUTE_FACT not in quoted["question_frame"]
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


@pytest.mark.parametrize(("ingress", "retry"), [
    ("accepted", "owner"),
    ("accepted_before_restart", "maintenance"),
    ("refused", "owner"),
])
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


def _continuable_asker(tmp_path, monkeypatch, *, chat_id=1):
    """A live root task wired for real Continue admission and real question delivery recovery."""
    from starlette.applications import Starlette
    from starlette.routing import Route

    from tests._budget_pause_exact_helpers import _install_queue
    from tests._owner_continue_helpers import _running_predecessor

    q, _state, workers = _install_queue(tmp_path, monkeypatch)
    _running_predecessor(tmp_path, "asker", chat_id=chat_id)
    task = {"id": "asker", "chat_id": chat_id, "drive_root": str(tmp_path)}
    q.RUNNING["asker"] = {"task": task}
    monkeypatch.setattr(q, "_task_drive_for_task", lambda _task, _tid: tmp_path)
    monkeypatch.setattr(td, "request_drive_root", lambda request: tmp_path)
    record_asked(tmp_path, "asker", quiz_id="q", question="Which database?", options=["sqlite", "postgres"],
                 assumption="sqlite", chat_id=chat_id)
    record_publication(tmp_path, "asker", "q", chat_id)
    bridge, _frames = _late_bridge(tmp_path, monkeypatch)
    app = Starlette(routes=[Route("/api/decisions", endpoint=td.api_decision_answer, methods=["POST"])])
    return app, bridge, q, workers


def _interrupt_asker(tmp_path, q, *, settle_mailbox: bool = False):
    from ouroboros.owner_mailbox import _ack_path, _mailbox_path
    from ouroboros.task_custody import settle_task_mailbox
    from ouroboros.task_results import STATUS_CANCELLED, write_task_result

    q.RUNNING.clear()
    write_task_result(tmp_path, "asker", STATUS_CANCELLED, result="interrupted",
                      cancel_origin={"source": "snapshot_restore", "reason": "server_shutdown"})
    if settle_mailbox:
        assert settle_task_mailbox(tmp_path, "asker", tmp_path) is True
        assert not _mailbox_path(tmp_path, "asker").exists()
        assert not _ack_path(tmp_path, "asker").exists()


@pytest.mark.parametrize("mode", ["text", "button_press"])
@pytest.mark.parametrize("settled_mailbox", [False, True])
def test_continue_then_recovery_hands_unread_answer_once_to_successor(tmp_path, monkeypatch, mode, settled_mailbox):
    """Continue omits an unread quiz answer from the prepared opening (keeping ordinary owner mail),
    leaving question_answers.deliver_answer as the single delivery owner to hand it once to the successor."""
    import asyncio

    from ouroboros.owner_mailbox import KIND_OWNER_TEXT, write_owner_message
    from supervisor.continuation_admission import admit_continuation
    from tests._owner_continue_helpers import NONCE

    app, bridge, q, workers = _continuable_asker(tmp_path, monkeypatch)
    assert write_owner_message(tmp_path, "Also include Q3 notes", "asker", msg_id="owner-2", kind=KIND_OWNER_TEXT)
    body = {"decision_id": "quiz:asker:q", "request_id": "r", "option_index": 1}
    if mode == "text":
        body["comment"] = "exact answer"
    assert _post(app, body).status_code == 200
    assert debt(tmp_path)["state"] == "owed" and debt(tmp_path)["recipient_task_id"] == "asker"

    _interrupt_asker(tmp_path, q, settle_mailbox=settled_mailbox)

    admitted = admit_continuation("asker", action_nonce=NONCE)
    assert admitted["ok"] is True, admitted
    successor = admitted["successor_task_id"]
    task = next(row for row in workers.PENDING if row["id"] == successor)
    later = task["metadata"]["continuation"]["owner_sources"]["later"]
    assert [row["content"] for row in later] == ["Also include Q3 notes"]
    assert "exact answer" not in task["text"] and "Which database?" not in task["text"]
    assert not any(row.get("quiz_id") == "q" for row in task["metadata"]["owner_corpus"])

    due(tmp_path)
    asyncio.run(td.recover_question_deliveries(tmp_path))
    assert _inbox(bridge) == [], "an admitted successor receives the answer via its mailbox, not fresh_head"
    assert debt(tmp_path)["recipient_task_id"] == successor and debt(tmp_path)["state"] == "owed"

    ctx, messages = drain(tmp_path, successor)
    assert debt(tmp_path)["state"] == "delivered" and debt(tmp_path)["read_at"]
    assert "Which database?" in str(messages) and "postgres" in str(messages)
    if mode == "text":
        assert [(r["source"], r["content"]) for r in ctx._owner_directives] == [("owner_quiz_answer", "exact answer")]
    else:
        assert [(r["source"], r["content"]) for r in ctx._owner_directives] == [("owner_press", "2. postgres")]

    assert _post(app, body).status_code == 200 and _inbox(bridge) == []
    assert getattr(drain(tmp_path, successor)[0], "_owner_directives", []) == []


@pytest.mark.parametrize("settled_mailbox", [False, True])
@pytest.mark.parametrize("ingress", ["accepted", "accepted_before_restart"])
def test_late_then_continue_keeps_one_delivery_without_opening_copy(tmp_path, monkeypatch, settled_mailbox, ingress):
    """Once the late path took an unread answer (whether settled or accepted before restart),
    a subsequent Continue keeps ordinary owner mail and never carries the quiz answer into its opening."""
    import asyncio

    from ouroboros import owner_quiz, process_custody
    from ouroboros.owner_mailbox import KIND_OWNER_TEXT, drain_owner_entries, write_owner_message
    from supervisor.continuation_admission import admit_continuation
    from tests._owner_continue_helpers import NONCE

    app, bridge, q, workers = _continuable_asker(tmp_path, monkeypatch)
    assert write_owner_message(tmp_path, "Also include Q3 notes", "asker", msg_id="owner-2", kind=KIND_OWNER_TEXT)
    assert answer(app).status_code == 200
    _interrupt_asker(tmp_path, q, settle_mailbox=settled_mailbox)

    due(tmp_path)
    with monkeypatch.context() as lost:
        if ingress == "accepted_before_restart":
            lost.setattr(owner_quiz, "mark_discharged", lambda *a, **k: {})
        asyncio.run(td.recover_question_deliveries(tmp_path))
    assert len(_inbox(bridge)) == 1 and debt(tmp_path)["route"] == "fresh_head"
    if ingress == "accepted_before_restart":
        monkeypatch.setattr(process_custody, "current_custody_session_id", lambda: "next-generation")

    admitted = admit_continuation("asker", action_nonce=NONCE)
    assert admitted["ok"] is True, admitted
    successor = admitted["successor_task_id"]
    task = next(row for row in workers.PENDING if row["id"] == successor)
    assert [row["content"] for row in task["metadata"]["continuation"]["owner_sources"]["later"]] == [
        "Also include Q3 notes"]
    assert "exact answer" not in task["text"] and "Which database?" not in task["text"]

    due(tmp_path)
    asyncio.run(td.recover_question_deliveries(tmp_path))
    assert _inbox(bridge) == []
    assert drain_owner_entries(tmp_path, successor, set()) == []
    if ingress == "accepted_before_restart":
        assert (debt(tmp_path)["state"], debt(tmp_path)["reason_code"]) == ("owed", "late_answer_delivery_unknown")
    else:
        assert debt(tmp_path)["state"] == "delivered"


@pytest.mark.parametrize("mode", ["text", "button_press"])
def test_read_answer_survives_settlement_in_continue_opening_without_resend(tmp_path, monkeypatch, mode):
    """When the predecessor durably ACKed the letter (even if acknowledge_answer never landed) and
    mailbox settlement persisted the existing read_at receipt before unlinking files, Continue
    carries the consumed text or wordless press as history and recovery settles without resending."""
    import asyncio

    from ouroboros.owner_mailbox import acknowledge_task_messages, drain_owner_entries
    from supervisor.continuation_admission import admit_continuation
    from tests._owner_continue_helpers import NONCE

    app, bridge, q, workers = _continuable_asker(tmp_path, monkeypatch)
    body = {"decision_id": "quiz:asker:q", "request_id": "r", "option_index": 1}
    if mode == "text":
        body["comment"] = "exact answer"
    assert _post(app, body).status_code == 200
    assert acknowledge_task_messages(tmp_path, "asker", ["quiz_answer:asker:q"], wake_id="loop_delivery")
    assert not debt(tmp_path).get("read_at") and debt(tmp_path)["state"] == "owed"

    _interrupt_asker(tmp_path, q, settle_mailbox=True)
    assert debt(tmp_path).get("read_at") and debt(tmp_path)["state"] == "delivered"

    admitted = admit_continuation("asker", action_nonce=NONCE)
    assert admitted["ok"] is True, admitted
    successor = admitted["successor_task_id"]
    task = next(row for row in workers.PENDING if row["id"] == successor)
    later = task["metadata"]["continuation"]["owner_sources"]["later"]
    if mode == "text":
        assert [(row["source"], row["content"]) for row in later] == [("owner_quiz_answer", "exact answer")]
    else:
        assert [(row["source"], row["content"]) for row in later] == [("owner_press", "2. postgres")]

    due(tmp_path)
    asyncio.run(td.recover_question_deliveries(tmp_path))
    assert debt(tmp_path)["state"] == "delivered" and debt(tmp_path)["read_at"]
    assert _inbox(bridge) == [] and drain_owner_entries(tmp_path, successor, set()) == []


@pytest.mark.parametrize("evidence", [
    "legacy_unread_without_receipt",
    "incomplete_unread_empty",
    "incomplete_unread_with_row",
    "absent_unread_no_acks",
])
def test_incomplete_or_legacy_terminal_capture_stays_unknown_while_non_quiz_continue_proceeds(
        tmp_path, monkeypatch, evidence):
    """Without an exact per-letter settlement receipt (or positive read_at), neither an incomplete
    or absent unread_mailbox nor a legacy/unioned unread_mailbox row proves read or unread: delivery
    stays owed without resending or marking read, while Continue still carries ordinary non-quiz mail."""
    import asyncio

    from ouroboros.owner_mailbox import KIND_OWNER_TEXT, drain_owner_entries, write_owner_message
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import admit_continuation
    from tests._owner_continue_helpers import NONCE

    app, bridge, q, workers = _continuable_asker(tmp_path, monkeypatch)
    assert write_owner_message(tmp_path, "Also include Q3 notes", "asker", msg_id="owner-2", kind=KIND_OWNER_TEXT)
    assert answer(app).status_code == 200
    _interrupt_asker(tmp_path, q, settle_mailbox=True)

    row = load_task_result(tmp_path, "asker", strict=True)
    quiz_line = next(line for line in row["owner_mailbox"]["rows"] if "quiz_answer:asker:q" in line)

    def degrade(current):
        updated = dict(current)
        quizzes = dict(updated.get("owner_quiz") or {})
        block = dict(quizzes["q"])
        block["delivery"] = {k: v for k, v in (block.get("delivery") or {}).items() if k != "settled_unread"}
        quizzes["q"] = block
        updated["owner_quiz"] = quizzes
        if evidence == "legacy_unread_without_receipt":
            updated["unread_mailbox"] = {"schema": 1, "total": 1, "rows": [quiz_line], "read_complete": True,
                                         "captured_at": "2026-11-24T00:00:00+00:00"}
        elif evidence == "incomplete_unread_empty":
            updated["unread_mailbox"] = {"schema": 1, "total": 0, "rows": [], "read_complete": False,
                                         "captured_at": "2026-11-24T00:00:00+00:00"}
        elif evidence == "incomplete_unread_with_row":
            updated["unread_mailbox"] = {"schema": 1, "total": 1, "rows": [quiz_line], "read_complete": False,
                                         "captured_at": "2026-11-24T00:00:00+00:00"}
        else:
            updated.pop("unread_mailbox", None)
        return updated

    patch_document(tmp_path, "asker", degrade)

    admitted = admit_continuation("asker", action_nonce=NONCE)
    assert admitted["ok"] is True, admitted
    successor = admitted["successor_task_id"]
    task = next(item for item in workers.PENDING if item["id"] == successor)
    assert [r["content"] for r in task["metadata"]["continuation"]["owner_sources"]["later"]] == [
        "Also include Q3 notes"]
    assert "exact answer" not in task["text"]

    due(tmp_path)
    asyncio.run(td.recover_question_deliveries(tmp_path))
    assert (debt(tmp_path)["state"], debt(tmp_path)["reason_code"]) == ("owed", "mailbox_write_failed")
    assert not debt(tmp_path).get("read_at")
    assert _inbox(bridge) == [] and drain_owner_entries(tmp_path, successor, set()) == []


def test_previously_accepted_ordinary_letter_moves_on_from_the_reader_fresh_full_unread_capture(
        tmp_path, monkeypatch):
    """An accepted ordinary letter with a recorded reader moves on when that settled reader's own complete
    unread capture, not older than the accepted write, names the letter: a legacy cleanup left no stamp, yet the
    positive terminal row proves delivered-and-unread, so recovery re-stages to the Continue instead of owing
    forever. Stale, partial or foreign captures keep the owed/unknown boundary of the parametrized seam above."""
    import asyncio

    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import admit_continuation
    from tests._owner_continue_helpers import NONCE

    app, bridge, q, workers = _continuable_asker(tmp_path, monkeypatch)
    assert answer(app).status_code == 200
    _interrupt_asker(tmp_path, q, settle_mailbox=True)

    row = load_task_result(tmp_path, "asker", strict=True)
    quiz_line = next(line for line in row["owner_mailbox"]["rows"] if "quiz_answer:asker:q" in line)
    mailbox_at = debt(tmp_path)["mailbox_at"]

    def degrade(current):
        updated = dict(current)
        quizzes = dict(updated.get("owner_quiz") or {})
        block = dict(quizzes["q"])
        block["delivery"] = {k: v for k, v in (block.get("delivery") or {}).items() if k != "settled_unread"}
        quizzes["q"] = block
        updated["owner_quiz"] = quizzes
        updated["unread_mailbox"] = {"schema": 1, "total": 1, "rows": [quiz_line], "read_complete": True,
                                     "captured_at": mailbox_at}
        return updated

    patch_document(tmp_path, "asker", degrade)

    admitted = admit_continuation("asker", action_nonce=NONCE)
    assert admitted["ok"] is True, admitted
    successor = admitted["successor_task_id"]

    due(tmp_path)
    asyncio.run(td.recover_question_deliveries(tmp_path))

    moved = debt(tmp_path)
    assert moved["state"] == "owed" and moved["reason_code"] == "awaiting_recipient_read"
    assert moved["recipient_task_id"] == successor and not moved.get("read_at")
    assert _inbox(bridge) == []  # it moved to the Continue, not to the owner's late path
    assert len(drain_owner_entries(tmp_path, successor, set())) == 1


def test_unreadable_ordinary_quiz_receipt_omits_quiz_letter_while_continue_proceeds(tmp_path, monkeypatch):
    """An unreadable asker record for an addressed quiz_answer omits only that letter without
    producing a fatal source gap, so ordinary owner mail and Continue admission proceed."""
    from ouroboros import task_store
    from ouroboros.owner_mailbox import KIND_OWNER_TEXT, KIND_QUIZ_ANSWER, write_owner_message
    from supervisor.continuation_admission import admit_continuation
    from tests._budget_pause_exact_helpers import _install_queue
    from tests._owner_continue_helpers import NONCE, _interrupted

    _q, _state, workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, task_id="pred-1", chat_id=1)
    assert write_owner_message(tmp_path, "Also include Q3 notes", "pred-1", msg_id="owner-2", kind=KIND_OWNER_TEXT)
    card = {"state": "answered", "quiz_id": "q", "question": "Which db?", "options": ["sqlite", "postgres"],
            "answered_index": 1, "comment": "secretive choice"}
    assert write_owner_message(tmp_path, "secretive choice", "pred-1", msg_id="quiz_answer:other-asker:q",
                               kind=KIND_QUIZ_ANSWER,
                               quiz_delivery={"task_id": "other-asker", "quiz_id": "q",
                                              "canonical_root": str(tmp_path), "answer_snapshot": card})

    real_read = task_store.read_document

    def fail_asker(root, task_id, *args, **kwargs):
        if str(task_id) == "other-asker":
            raise OSError("asker store unavailable")
        return real_read(root, task_id, *args, **kwargs)

    monkeypatch.setattr(task_store, "read_document", fail_asker)

    admitted = admit_continuation("pred-1", action_nonce=NONCE)
    assert admitted["ok"] is True, admitted
    successor = admitted["successor_task_id"]
    task = next(item for item in workers.PENDING if item["id"] == successor)
    assert [r["content"] for r in task["metadata"]["continuation"]["owner_sources"]["later"]] == [
        "Also include Q3 notes"]
    assert "secretive choice" not in task["text"]


@pytest.mark.parametrize("route", ["history", "continue"])
def test_protected_secret_outcome_keeps_pre_change_delivery_and_settlement_behavior(tmp_path, monkeypatch, route):
    """Protected-question outcomes keep their exact pre-change deliver_answer and mailbox-settlement behavior: no
    ordinary settled_unread receipt and no mailbox route mutation, and once their reader ended with the outcome's
    letter unread and settled away, the earlier route moves it on once, to that reader's admitted Continue or else to
    the caller's history fallback. The block holds dummy outcome facts only (names, never a value)."""
    from ouroboros import question_answers as qa
    from ouroboros.owner_mailbox import KIND_QUIZ_ANSWER, drain_owner_entries
    from ouroboros.task_custody import settle_task_mailbox
    from ouroboros.task_results import STATUS_CANCELLED, STATUS_RUNNING, write_task_result
    from tests._budget_pause_exact_helpers import _install_queue

    q, _state, _workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(q, "_task_drive_for_task", lambda _t, _tid: tmp_path)
    write_task_result(tmp_path, "asker", STATUS_RUNNING, chat_id=1, root_task_id="asker")
    block = {
        "quiz_id": "sec-1", "state": "answered", "question": "Provide token", "options": [],
        "secret_request": {"plugin": "demo", "pluginName": "Demo", "fields": [{"id": "api_key"}]},
        "secret_outcome": {"event": "secret_filled", "pluginName": "Demo", "fields": ["api_key"]},
        "delivery": {"kind": "answer", "state": "owed", "logical_id": "quiz_answer:asker:sec-1"},
    }
    patch_document(tmp_path, "asker", lambda row: {**row, "owner_quiz": {"sec-1": block}})

    outcome = qa.deliver_answer(tmp_path, "asker", "sec-1", block,
                                lambda tid: ({"id": tid, "chat_id": 1, "drive_root": str(tmp_path)}, ""),
                                lambda *a, **k: (True, ""))
    assert outcome == (False, "awaiting_recipient_read")
    delivery = quiz_states(tmp_path, "asker")["sec-1"]["delivery"]
    assert "route" not in delivery and "settled_unread" not in delivery

    write_task_result(tmp_path, "asker", STATUS_CANCELLED, result="interrupted")
    assert settle_task_mailbox(tmp_path, "asker", tmp_path) is True
    delivery = quiz_states(tmp_path, "asker")["sec-1"]["delivery"]
    assert "settled_unread" not in delivery
    if route == "continue":
        continue_to(tmp_path)

    def reader(tid):  # the asker ended; only its admitted Continue reads now
        return ({"id": tid, "chat_id": 1, "drive_root": str(tmp_path)}, "") if tid == "successor" else (
            None, "task_not_live")

    routed = []
    retry = qa.deliver_answer(tmp_path, "asker", "sec-1", block, reader, lambda *a, **k: routed.append(a) or (True, ""))
    delivery = quiz_states(tmp_path, "asker")["sec-1"]["delivery"]
    if route == "continue":
        assert retry == (False, "awaiting_recipient_read") and routed == []
        assert (delivery["recipient_task_id"], delivery["state"]) == ("successor", "owed")
        [letter] = [e for e in drain_owner_entries(tmp_path, "successor", set()) if e.get("kind") == KIND_QUIZ_ANSWER]
        assert letter["quiz_delivery"]["answer_snapshot"]["secret_outcome"] == block["secret_outcome"]
    else:
        assert retry == (True, "") and len(routed) == 1 and delivery["state"] == "delivered"


@pytest.mark.parametrize("cleanup_point", ["before_live_ack", "after_live_ack"])
def test_live_ack_transfer_during_settlement_keeps_consumed_answer_in_continue(tmp_path, monkeypatch, cleanup_point):
    """Continue checks the vanishing live ACK ledger before the canonical receipt, so mailbox
    cleanup transferring that positive ACK to read_at right before or right after the live ACK
    check never drops the consumed answer from the successor's opening."""
    import asyncio

    from ouroboros import owner_mailbox
    from ouroboros.owner_mailbox import acknowledge_task_messages, drain_owner_entries
    from ouroboros.task_custody import settle_task_mailbox
    from supervisor.continuation_admission import admit_continuation
    from tests._owner_continue_helpers import NONCE

    app, bridge, q, workers = _continuable_asker(tmp_path, monkeypatch)
    assert answer(app).status_code == 200
    assert acknowledge_task_messages(tmp_path, "asker", ["quiz_answer:asker:q"], wake_id="loop_delivery")
    assert not debt(tmp_path).get("read_at")
    _interrupt_asker(tmp_path, q, settle_mailbox=False)

    real_read_state = owner_mailbox.mail_read_state
    settled = []

    def racing_mail_read_state(drive_root, task_id, msg_id):
        if task_id == "asker" and msg_id == "quiz_answer:asker:q" and not settled:
            settled.append(True)
            if cleanup_point == "before_live_ack":
                assert settle_task_mailbox(tmp_path, "asker", tmp_path) is True
                return real_read_state(drive_root, task_id, msg_id)
            observed = real_read_state(drive_root, task_id, msg_id)
            assert observed is True
            assert settle_task_mailbox(tmp_path, "asker", tmp_path) is True
            return observed
        return real_read_state(drive_root, task_id, msg_id)

    monkeypatch.setattr(owner_mailbox, "mail_read_state", racing_mail_read_state)

    admitted = admit_continuation("asker", action_nonce=NONCE)
    assert admitted["ok"] is True, admitted
    assert settled == [True]
    successor = admitted["successor_task_id"]
    task = next(row for row in workers.PENDING if row["id"] == successor)
    later = task["metadata"]["continuation"]["owner_sources"]["later"]
    assert [(row["source"], row["content"]) for row in later] == [("owner_quiz_answer", "exact answer")]

    due(tmp_path)
    asyncio.run(td.recover_question_deliveries(tmp_path))
    assert debt(tmp_path)["state"] == "delivered" and debt(tmp_path)["read_at"]
    assert _inbox(bridge) == [] and drain_owner_entries(tmp_path, successor, set()) == []
