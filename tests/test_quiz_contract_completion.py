"""Engineering brief: admitted membership, explicit owner close, immutable publication."""

from tests import _dialogue as dialogue_test
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import queue as queue_module
import threading

import pytest

from ouroboros import obligations, owner_quiz
from ouroboros.gateway.task_decision import answer_decision, recover_question_deliveries
from ouroboros.owner_mailbox import drain_owner_entries
from ouroboros.task_results import write_task_result
from supervisor import queue
from tests.test_quiz_history_evidence import runtime as runtime, _ask, _answer


def ask(root, qid="q"):
    return owner_quiz.record_asked(root, "task", quiz_id=qid, question="Proceed?", options=["Yes", "No"])


def fail_after_store(monkeypatch, exc):
    """The question's document and its ``owner_questions`` flag are written, then the
    transaction fails before COMMIT: the store rolls both back together."""
    from ouroboros import store
    original = store.Txn.store_task
    def store_task(self, document, **kwargs):
        original(self, document, **kwargs)
        if (document.get("owner_quiz") or {}).get("q"):
            raise exc
    monkeypatch.setattr(store.Txn, "store_task", store_task)


@pytest.mark.parametrize("existing", [False, True])
def test_admission_membership_failure_precedes_canonical_commit(tmp_path, monkeypatch, existing):
    import sqlite3
    if existing:
        assert ask(tmp_path, "old")["state"] == "open"
    before = owner_quiz.quiz_states(tmp_path, "task")
    members = set(obligations.members(tmp_path, "owner_questions"))
    fail_after_store(monkeypatch, sqlite3.OperationalError("disk I/O error"))
    assert ask(tmp_path) == {"refused": "owner_questions_unavailable"}
    assert owner_quiz.quiz_states(tmp_path, "task") == before
    assert set(obligations.members(tmp_path, "owner_questions")) == members


def test_tool_membership_failure_has_typed_refusal_no_wait_or_event(tmp_path, monkeypatch):
    import sqlite3
    from ouroboros import store
    from tests.test_quiz_answer import _tool_ctx
    from ouroboros.tools.core_artifacts import _escalate

    original = store.Txn.store_task
    def store_task(self, document, **kwargs):
        original(self, document, **kwargs)
        if document.get("owner_quiz"):
            raise sqlite3.OperationalError("disk I/O error")
    monkeypatch.setattr(store.Txn, "store_task", store_task)
    ctx = _tool_ctx(tmp_path)
    ctx.owner_wait_callback = lambda *_: None
    reply = _escalate(ctx, question="Proceed?", wait_for_answer=True)
    assert "QUIZ_MEMBERSHIP_UNAVAILABLE" in reply
    assert not ctx.pending_events and not getattr(ctx, "_owner_wait_requested", "")
    assert owner_quiz.quiz_states(tmp_path, "root-1") == {}


def test_crash_inside_the_question_transaction_leaves_neither_question_nor_membership(tmp_path, monkeypatch):
    fail_after_store(monkeypatch, OSError("crash before COMMIT"))
    with pytest.raises(OSError, match="before COMMIT"):
        ask(tmp_path)
    assert owner_quiz.quiz_states(tmp_path, "task") == {}
    assert "task" not in obligations.members(tmp_path, "owner_questions")


def close(runtime, request="close-1", reason="The proposal is no longer needed"):
    return asyncio.run(answer_decision(runtime.root, {
        "decision_id": f"quiz:{runtime.task['id']}:q", "request_id": request,
        "action": "close", "reason": reason,
    }))


def prepare(runtime):
    write_task_result(runtime.root, runtime.task["id"], "running", chat_id=1)
    _ask(runtime, "q")


def test_owner_close_delivers_typed_noanswer_and_retry_preserves_winner(runtime):
    from ouroboros.owner_wait import classify_wake
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from types import SimpleNamespace

    prepare(runtime)
    status, first = close(runtime)
    assert status == 200 and first["state"] == "closed"
    assert first["closed_actor"] == "owner" and first["closed_request_id"] == "close-1"
    assert first["closed_at"] and "answered_index" not in first and "closed_by_task_id" not in first
    [entry] = drain_owner_entries(runtime.root, runtime.task["id"], set())
    assert entry["kind"] == "quiz_closed" and "no approval was given" in entry["text"]
    assert classify_wake([entry], "q") == "question_closed"
    ctx = SimpleNamespace(task_attempt=1, budget_drive_root=runtime.root)
    messages = []
    _drain_incoming_messages(messages, queue_module.Queue(), runtime.root, runtime.task["id"], None, set(), ctx)
    assert len(messages) == 1 and "closed without answer" in messages[0]["content"]
    status, duplicate = close(runtime, reason="changed retry bytes")
    assert status == 200 and duplicate["duplicate"]
    assert duplicate["closed_reason"] == first["closed_reason"]
    assert duplicate["closed_at"] == first["closed_at"]
    status, loser = close(runtime, request="close-2", reason="later requester")
    assert status == 409 and loser["closed_request_id"] == "close-1"
    status, answer = _answer(runtime, "q", request_id="late-answer", index=0)
    assert status == 409 and answer["closed_actor"] == "owner"
    assert answer["closed_reason"] == first["closed_reason"]
    assert runtime.bridge._inbox.empty()
    rows = dialogue_test.log_rows(runtime.root / "logs/chat.jsonl")
    assert len([row for row in rows if row.get("type") == "quiz_closed"]) == 1
    assert not [row for row in rows if row.get("type") == "quiz_answer"]


@pytest.mark.parametrize("words", ["omitted", ""])
def test_owner_close_over_http_needs_no_words_and_refuses_an_invalid_reason(tmp_path, monkeypatch, words):
    """``POST /api/decisions``: one tap closes. The owner's words stay optional text of at most
    2000 characters; their absence records the standard sentence, anything else is refused."""
    from tests.test_quiz_answer import _decision_app, _post
    from tests.test_quiz_late_answer import _late_bridge

    write_task_result(tmp_path, "task", "running", chat_id=1)
    for qid in ("tapped", "typed"):
        owner_quiz.record_asked(tmp_path, "task", quiz_id=qid, question="Proceed?", options=["Yes", "No"], chat_id=1)
    _late_bridge(tmp_path, monkeypatch)  # a real bridge; both cards' Main publication is witnessed
    app = _decision_app(tmp_path, monkeypatch)
    tap = {"request_id": "tap", "decision_id": "quiz:task:tapped", "action": "close"}
    for invalid in (7, ["Withdrawn"], "x" * 2001):
        reply = _post(app, {**tap, "reason": invalid})
        assert reply.status_code == 400 and reply.json()["reason_code"] == "close_reason_invalid", reply.text
    assert owner_quiz.quiz_states(tmp_path, "task")["tapped"]["state"] == "open"
    reply = _post(app, tap if words == "omitted" else {**tap, "reason": words})
    assert reply.status_code == 200 and reply.json()["state"] == "closed", reply.text
    assert reply.json()["closed_reason"] == owner_quiz.OWNER_CLOSE_REASON == "Closed by the owner without an answer."
    assert owner_quiz.quiz_states(tmp_path, "task")["tapped"]["closed_reason"] == owner_quiz.OWNER_CLOSE_REASON
    longest = "x" * 2000
    reply = _post(app, {"request_id": "typed", "decision_id": "quiz:task:typed", "action": "close", "reason": longest})
    assert reply.status_code == 200 and reply.json()["closed_reason"] == longest, reply.text


def test_owner_closure_mailbox_failure_recovers_automatically(runtime, monkeypatch):
    from ouroboros import owner_mailbox

    prepare(runtime)
    real_write = owner_mailbox.write_owner_message
    monkeypatch.setattr(owner_mailbox, "write_owner_message", lambda *a, **kw: False)
    status, receipt = close(runtime)
    assert status == 200 and receipt["question_delivery"]["state"] == "owed"
    assert receipt["question_delivery"]["reason_code"] == "mailbox_write_failed"
    assert asyncio.run(recover_question_deliveries(runtime.root))["attempted"] == 0
    monkeypatch.setattr(owner_mailbox, "write_owner_message", real_write)
    def advance(quizzes):
        quizzes["q"]["delivery"]["next_retry_at"] = ""
        return True
    owner_quiz._mutate_projection(runtime.root, runtime.task["id"], advance)
    assert asyncio.run(recover_question_deliveries(runtime.root)) == {"attempted": 1, "delivered": 1, "pending": 0}
    assert drain_owner_entries(runtime.root, runtime.task["id"], set())[0]["kind"] == "quiz_closed"


def test_owner_close_terminal_never_starts_a_new_turn(runtime, monkeypatch):
    prepare(runtime)
    monkeypatch.setattr(queue, "RUNNING", {})
    write_task_result(runtime.root, runtime.task["id"], "completed", result="done")
    status, receipt = close(runtime)
    assert status == 200 and receipt["question_delivery"]["state"] == "delivered"
    assert runtime.bridge._inbox.empty()
    assert drain_owner_entries(runtime.root, runtime.task["id"], set()) == []


def test_answer_and_close_first_winner_under_result_lock(tmp_path):
    for i in range(8):
        qid = str(i)
        ask(tmp_path, qid)
        ready = threading.Barrier(2)
        def answer():
            ready.wait()
            return owner_quiz.record_answered(tmp_path, "task", quiz_id=qid, option_index=1, request_id="answer")
        def retire():
            ready.wait()
            return owner_quiz.record_closed(tmp_path, "task", quiz_id=qid, reason="Withdrawn", actor="owner", request_id="close")
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(answer), pool.submit(retire)]
            outcomes = [future.result() for future in futures]
        assert sum(outcome["ok"] for outcome in outcomes) == 1
        block = owner_quiz.quiz_states(tmp_path, "task")[qid]
        assert ("answered_index" in block) != ("closed_reason" in block)
        assert block["delivery"]["state"] == "owed"


def test_after_conversion_new_question_publishes_and_late_answers_in_project(runtime, monkeypatch):
    from ouroboros.projects_registry import bind_task_to_project, create_project

    project = create_project(runtime.root, "after-conversion", name="New destination")
    bind_task_to_project(runtime.root, runtime.task["id"], project["id"], origin={"absent": "system"})
    prepare(runtime)  # ask-time chat=Main is stale; actual first publication resolves binding.
    block = owner_quiz.quiz_states(runtime.root, runtime.task["id"])["q"]
    assert block["chat_id"] == 1 and block["publication_chat_id"] == project["chat_id"]
    monkeypatch.setattr(queue, "RUNNING", {})
    write_task_result(runtime.root, runtime.task["id"], "completed", result="done")
    assert _answer(runtime, "q", request_id="after-conversion", index=0)[1]["forwarded"]
    assert runtime.bridge._inbox.get_nowait()["chat_id"] == project["chat_id"]


def test_hidden_publication_is_deliberately_bound_to_main(runtime):
    write_task_result(runtime.root, runtime.task["id"], "running", chat_id=0)
    _ask(runtime, "q", chat_id=0)
    block = owner_quiz.quiz_states(runtime.root, runtime.task["id"])["q"]
    assert block["chat_id"] == 0 and block["publication_chat_id"] == 1
    assert runtime.frames[-1]["chat_id"] == 1


def test_unavailable_binding_never_falls_back_to_main(runtime):
    write_task_result(runtime.root, runtime.task["id"], "running", chat_id=1)
    owner_quiz.record_asked(runtime.root, runtime.task["id"], quiz_id="q", question="Which?", options=["A"], assumption="Research")
    path = runtime.root / "state" / "project_task_bindings.json"
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text("corrupt")
    with pytest.raises(ValueError):
        runtime.bridge.send_quiz(1, "q", "Which?", [{"label": "A"}], assumption="Research", task_id=runtime.task["id"])
    assert not runtime.frames
    assert "publication_chat_id" not in owner_quiz.quiz_states(runtime.root, runtime.task["id"])["q"]


def test_closure_has_distinct_dialogue_provenance_and_terminal_mail_custody(runtime):
    from ouroboros.dialogue_provenance import row_author, render_row_text

    prepare(runtime)
    assert close(runtime)[0] == 200
    row = next(json.loads(line) for line in dialogue_test.log_text(runtime.root / "logs/chat.jsonl").splitlines()
               if json.loads(line).get("type") == "quiz_closed")
    assert row_author(row)["kind"] == "human"
    rendered = render_row_text(row)
    assert "closed without answer" in rendered and "No approval recorded" in rendered
    assert "[answer " not in rendered
    result = write_task_result(runtime.root, runtime.task["id"], "completed", result="done")
    assert any(json.loads(line)["kind"] == "quiz_closed" for line in result["owner_mailbox"]["rows"])
