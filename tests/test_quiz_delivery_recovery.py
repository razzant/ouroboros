"""Canonical accepted quiz effects recover automatically with bounded durable retries."""
import asyncio
import base64
import queue as queue_module
import threading

import pytest
from tests._dialogue import chat_rows

from ouroboros.gateway.task_decision import recover_question_deliveries
from ouroboros.owner_quiz import (
    _mutate_projection, late_answer_model_text, late_answer_ref, quiz_states,
    record_closed,
)
from ouroboros.task_results import load_task_result, write_task_result
from supervisor import message_bus, queue
from tests.test_quiz_history_evidence import runtime as runtime, _ask, _answer, _facts


def block(runtime, qid="q"):
    return quiz_states(runtime.root, runtime.task["id"])[qid]


def due(runtime, qid="q"):
    def advance(quizzes):
        quizzes[qid]["delivery"]["next_retry_at"] = ""
        return True
    _mutate_projection(runtime.root, runtime.task["id"], advance)


def recover(runtime):
    return asyncio.run(recover_question_deliveries(runtime.root))


def prepare(runtime):
    write_task_result(runtime.root, runtime.task["id"], "running", chat_id=1)
    _ask(runtime, "q")


def test_history_failure_recovers_without_client_retry_or_history_scan(runtime, monkeypatch):
    from ouroboros.memory import Memory
    from ouroboros.owner_mailbox import drain_owner_entries, acknowledge_transcript_entry

    prepare(runtime)
    real_log = message_bus.log_chat
    def unavailable(*args, **kwargs):
        if kwargs.get("record_type") == "quiz_answer":
            raise OSError("evidence unavailable")
        return real_log(*args, **kwargs)
    monkeypatch.setattr(message_bus, "log_chat", unavailable)
    status, reply = _answer(runtime, "q", request_id="stable-owner-request", index=1, comment=" exact choice ")
    assert status == 503 and reply["state"] == "answered"
    assert reply["question_delivery"] == {
        "kind": "answer", "state": "owed", "reason_code": "quiz_history_write_failed"}
    original = block(runtime)
    assert original["delivery"]["attempts"] == 1 and not original.get("discharged_at")
    assert recover(runtime) == {"attempted": 0, "delivered": 0, "pending": 0}
    assert block(runtime)["delivery"] == original["delivery"]
    monkeypatch.setattr(message_bus, "log_chat", real_log)
    monkeypatch.setattr(Memory, "read_chat_generations", lambda *a, **kw: pytest.fail("maintenance scanned chat generations"))
    due(runtime)
    assert recover(runtime) == {"attempted": 1, "delivered": 0, "pending": 1}
    repaired = block(runtime)
    assert not repaired.get("discharged_at") and repaired["delivery"]["attempts"] == 2
    assert repaired["request_id"] == original["request_id"] == "stable-owner-request"
    assert repaired["answered_at"] == original["answered_at"]
    assert repaired["answered_index"] == 1 and repaired["comment"] == " exact choice "
    [entry] = drain_owner_entries(runtime.root, runtime.task["id"], set())
    assert entry["kind"] == "quiz_answer" and "option 2: Second" in entry["text"]
    assert acknowledge_transcript_entry(runtime.root, runtime.task["id"], entry)
    assert block(runtime)["delivery"]["state"] == "delivered"
    assert recover(runtime)["attempted"] == 0


def test_live_answer_mailbox_failure_then_terminal_recovers_through_accepted_ingress(runtime, monkeypatch):
    from ouroboros import owner_mailbox

    prepare(runtime)
    real_write = owner_mailbox.write_owner_message
    monkeypatch.setattr(owner_mailbox, "write_owner_message", lambda *a, **kw: False)
    status, reply = _answer(runtime, "q", request_id="owner-live", index=None, comment="My exact alternative")
    assert status == 503 and reply["question_delivery"]["reason_code"] == "mailbox_write_failed"
    original = block(runtime)
    assert original["delivery"]["history_at"] and not original.get("answered_after_terminal")
    original_history = chat_rows(runtime.root)
    monkeypatch.setattr(queue, "RUNNING", {})
    write_task_result(runtime.root, runtime.task["id"], "completed", result="finished before delivery")
    monkeypatch.setattr(owner_mailbox, "write_owner_message", real_write)
    due(runtime)
    assert recover(runtime) == {"attempted": 1, "delivered": 1, "pending": 0}
    repaired = block(runtime)
    assert repaired["delivery_after_terminal"] is True and not repaired.get("answered_after_terminal")
    assert repaired["answered_at"] == original["answered_at"]
    assert repaired["delivery"]["attempts"] == 2
    rows = chat_rows(runtime.root)
    assert len([row for row in rows if row.get("type") == "quiz_answer"]) == 1
    [accepted] = [row for row in rows if row.get("client_message_id") == "quiz_late_answer:task-quiz:q"]
    ref = late_answer_ref(accepted["late_answer"])
    assert ref["answer_snapshot"]["quiz"]["comment"] == "My exact alternative"
    assert ref["answer_snapshot"]["quiz"]["request_id"] == "owner-live"
    queued = runtime.bridge._inbox.get_nowait()
    assert queued["task_metadata"]["late_answer"] == ref
    # Canonical hot-set GC cannot remove the accepted message's frame authority.
    def remove(quizzes):
        del quizzes["q"]
        return True
    _mutate_projection(runtime.root, runtime.task["id"], remove)
    frame = late_answer_model_text(runtime.root, ref, accepted["text"])
    assert "Question was: Choose for q?" in frame
    assert "My exact alternative" in frame and "without choosing an offered option" in frame
    assert "could not be read" not in frame
    assert "answer_snapshot" not in late_answer_ref({**ref, "quiz_id": "other"})
    assert chat_rows(runtime.root)[:len(original_history)] == original_history


def test_legacy_late_answer_keeps_owner_words_when_card_storage_is_unavailable(tmp_path, monkeypatch):
    from ouroboros import owner_quiz

    def unavailable(*args, **kwargs):
        raise OSError("fixture card storage unavailable")

    monkeypatch.setattr(owner_quiz, "quiz_states", unavailable)
    frame = late_answer_model_text(tmp_path, {"task_id": "ended", "quiz_id": "old-question"},
                                   "Keep my exact decision: research only.")
    assert "Keep my exact decision: research only." in frame
    assert "could not be read" in frame


def test_closed_history_failure_recovers_without_owner_answer(runtime, monkeypatch):
    from ouroboros.owner_mailbox import drain_owner_entries

    prepare(runtime)
    record_closed(runtime.root, runtime.task["id"], quiz_id="q", reason="The proposed action was withdrawn", actor_task_id=runtime.task["id"])
    real_log = message_bus.log_chat
    monkeypatch.setattr(message_bus, "log_chat", lambda *a, **kw: (_ for _ in ()).throw(OSError("unavailable")))
    assert recover(runtime) == {"attempted": 1, "delivered": 0, "pending": 1}
    first = block(runtime)
    assert first["delivery"]["state"] == "owed"
    assert recover(runtime)["attempted"] == 0
    monkeypatch.setattr(message_bus, "log_chat", real_log)
    due(runtime)
    assert recover(runtime) == {"attempted": 1, "delivered": 1, "pending": 0}
    settled = block(runtime)
    assert settled["state"] == "closed" and settled["discharged_at"]
    assert "answered_at" not in settled and "answered_index" not in settled
    assert drain_owner_entries(runtime.root, runtime.task["id"], set()) == []
    rows = chat_rows(runtime.root)
    assert not [row for row in rows if row.get("type") == "quiz_answer"]
    [closed] = [row for row in rows if row.get("type") == "quiz_closed"]
    assert closed["text"] == settled["closed_reason"]
    assert runtime.frames[-1]["question_delivery"] == {"kind": "closure", "state": "delivered"}


def forget_witness(runtime):
    """A question an import could not address (ambiguous or unstable history)."""
    def forget(quizzes):
        quizzes["q"].pop("publication_chat_id")
        quizzes["q"].pop("publication_evidence")
        return True
    _mutate_projection(runtime.root, runtime.task["id"], forget)


def test_legacy_unknown_publication_stays_typed_debt_until_an_import_binds_it(runtime, monkeypatch):
    from ouroboros.owner_quiz import record_publication

    prepare(runtime)
    forget_witness(runtime)
    monkeypatch.setattr(queue, "RUNNING", {})
    status, reply = _answer(runtime, "q", request_id="old-route", index=0)
    assert status == 200 and not reply["forwarded"]
    assert reply["question_delivery"]["reason_code"] == "publication_address_unknown"
    due(runtime)  # no retry can bind an address: the debt stays current without attempts
    assert recover(runtime)["attempted"] == 0
    debt = block(runtime)["delivery"]
    assert debt["attempts"] == 1 and debt["reason_code"] == "publication_address_unknown"
    assert runtime.bridge._inbox.empty()
    record_publication(runtime.root, runtime.task["id"], "q", 1, evidence="history")
    due(runtime)
    assert recover(runtime) == {"attempted": 1, "delivered": 1, "pending": 0}
    assert runtime.bridge._inbox.get_nowait()["client_message_id"] == "quiz_late_answer:task-quiz:q"
    assert block(runtime)["discharged_at"] and block(runtime)["delivery"]["attempts"] == 2


def migrated_without_address(runtime):
    """An older body's ask (no delivery record, no publication address) whose card history names
    two chats: the one-time import keeps it a typed ``question:`` unknown, never a guessed chat."""
    from ouroboros.obligations import members
    from ouroboros.startup_migrations import migrate_question_publications
    from ouroboros import dialogue_store as ds

    def legacy(quizzes):
        for key in ("publication_chat_id", "publication_evidence", "publication_bound_at", "published_at", "delivery"):
            quizzes["q"].pop(key, None)
        return True
    _mutate_projection(runtime.root, runtime.task["id"], legacy)
    # Two imported publication witnesses disagree on the address.
    ds.append(runtime.root, "chat", {"direction": "out", "type": "quiz", "task_id": runtime.task["id"],
                                                    "chat_id": 2, "quiz": {"quiz_id": "q"}})
    assert migrate_question_publications(runtime.root) == {"bound": 0, "unknown": 1}
    assert members(runtime.root, "unknowns")["question:task-quiz:q"]["reason"] == "publication_address_ambiguous"


@pytest.mark.parametrize("author", ["running"])
def test_live_author_receives_an_answer_whose_card_address_is_unknown_once(runtime, monkeypatch, author):
    """A task alive across the upgrade asked under the older body; its migration found no
    unique publication witness. Its mailbox needs no chat address: the winning answer reaches
    the author once, and only the transcript row waits, typed, for an address."""
    from ouroboros.obligations import members
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.owner_quiz import quiz_delivery_due, record_publication

    prepare(runtime)
    migrated_without_address(runtime)
    if author == "paused_pending":
        monkeypatch.setattr(queue, "RUNNING", {})
        monkeypatch.setattr(queue, "PENDING", [dict(runtime.task)])
    status, reply = _answer(runtime, "q", request_id="live-owner", index=1, comment=" exact words ")
    assert status == 200 and reply["state"] == "answered" and reply["duplicate"] is False
    assert reply["question_delivery"] == {
        "kind": "answer", "state": "owed", "reason_code": "publication_address_unknown"}
    def mailbox():
        return [entry for entry in drain_owner_entries(runtime.root, runtime.task["id"], set())
                if entry["msg_id"] == "quiz_answer:task-quiz:q"]
    [entry] = mailbox()
    assert entry["kind"] == "quiz_answer" and "option 2: Second" in entry["text"] and " exact words " in entry["text"]
    debt = block(runtime)["delivery"]
    assert debt["mailbox_at"] and not debt.get("history_at") and not block(runtime).get("discharged_at")
    due(runtime)
    assert not quiz_delivery_due(block(runtime)) and recover(runtime)["attempted"] == 0
    assert runtime.task["id"] in members(runtime.root, "owner_questions")
    retried = _answer(runtime, "q", request_id="live-owner", index=1, comment=" exact words ")
    assert retried[0] == 200 and retried[1]["duplicate"] is True
    loser = _answer(runtime, "q", request_id="competing", index=0)
    assert loser[0] == 409 and loser[1]["answered_index"] == 1
    assert len(mailbox()) == 1 and block(runtime)["answered_index"] == 1
    from ouroboros.owner_mailbox import acknowledge_transcript_entry
    assert acknowledge_transcript_entry(runtime.root, runtime.task["id"], entry)
    # The author settles; an explicit import later binds the address from exact history: the
    # row is written once, and the answer the author already holds never becomes an owner turn.
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(queue, "PENDING", [])
    write_task_result(runtime.root, runtime.task["id"], "completed", result="finished")
    record_publication(runtime.root, runtime.task["id"], "q", 1, evidence="history")
    due(runtime)
    assert recover(runtime) == {"attempted": 1, "delivered": 1, "pending": 0}
    settled = block(runtime)
    assert settled["discharged_at"] and not settled.get("delivery_after_terminal")
    rows = chat_rows(runtime.root)
    [row] = [row for row in rows if row.get("type") == "quiz_answer"]
    assert row["chat_id"] == 1 and row["quiz"]["comment"] == " exact words "
    assert not [row for row in rows if row.get("client_message_id") == "quiz_late_answer:task-quiz:q"]
    assert runtime.bridge._inbox.empty() and mailbox() == []  # saved ACK excludes read mail


def test_owner_closure_reaches_a_live_author_while_its_row_waits_for_an_address(runtime):
    from ouroboros.gateway.task_decision import answer_decision
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.owner_quiz import record_publication

    prepare(runtime)
    migrated_without_address(runtime)
    status, reply = asyncio.run(answer_decision(runtime.root, {
        "request_id": "close-1", "decision_id": "quiz:task-quiz:q", "action": "close", "reason": "Withdrawn"}))
    assert status == 200 and reply["state"] == "closed"
    assert reply["question_delivery"] == {
        "kind": "closure", "state": "owed", "reason_code": "publication_address_unknown"}
    [entry] = drain_owner_entries(runtime.root, runtime.task["id"], set())
    assert entry["kind"] == "quiz_closed" and "no approval was given" in entry["text"]
    due(runtime)
    assert recover(runtime)["attempted"] == 0 and block(runtime)["delivery"]["mailbox_at"]
    record_publication(runtime.root, runtime.task["id"], "q", 1, evidence="history")
    due(runtime)
    assert recover(runtime) == {"attempted": 1, "delivered": 1, "pending": 0}
    assert len(drain_owner_entries(runtime.root, runtime.task["id"], set())) == 1
    rows = chat_rows(runtime.root)
    assert [row["text"] for row in rows if row.get("type") == "quiz_closed"] == ["Withdrawn"]


@pytest.mark.parametrize("door", ["rebuild", "rollback"])
def test_a_closure_owing_its_unpublished_ask_is_no_import_unknown(runtime, monkeypatch, door):
    """An ask interrupted before publication and then closed hands its publication to the closure
    (``publication_pending``), which binds the card's address itself. The rebuild or a rollback's
    import must not record it as a ``question:`` unknown: once that closure is delivered, later
    boots return early and nothing would clear the row. An older body's ask stays unknown."""
    from ouroboros import obligations, startup_migrations
    from ouroboros.owner_quiz import record_asked
    from tests.test_main_decisions import older_body_ask

    tid = runtime.task["id"]
    write_task_result(runtime.root, tid, "running", chat_id=1)
    if door == "rollback":
        startup_migrations.prepare_startup_state(runtime.root)  # this body's watermarks precede the rollback
    record_asked(runtime.root, tid, quiz_id="q", question="Still needed?", options=["Yes", "No"], chat_id=1)
    record_closed(runtime.root, tid, quiz_id="q", reason="Withdrawn", actor_task_id=tid)
    assert block(runtime)["delivery"]["publication_pending"] is True
    older_body_ask(runtime.root, tid, "older", chat_id=1)  # its card left no history row

    def questions():
        return {key for key in obligations.members(runtime.root, "unknowns") if key.startswith("question:")}
    report = startup_migrations.prepare_startup_state(runtime.root, rebuild=door == "rebuild")
    if door == "rollback":
        from ouroboros.store_import import run_legacy_reconcile
        assert report["question_publications"] == {}
        report = run_legacy_reconcile(runtime.root)
    assert report["question_publications"] == {"bound": 0, "unknown": 1}
    assert questions() == {f"question:{tid}:older"}
    assert recover(runtime) == {"attempted": 1, "delivered": 1, "pending": 0}
    settled = block(runtime)
    assert (settled["publication_chat_id"], settled["publication_evidence"]) == (1, "pre_exposure")
    assert settled["discharged_at"] and settled["delivery"]["state"] == "delivered"
    rows = chat_rows(runtime.root)
    assert [(row["chat_id"], row["text"]) for row in rows if row.get("type") == "quiz_closed"] == [(1, "Withdrawn")]
    monkeypatch.setattr(startup_migrations, "_legacy_question_publications", lambda *a: pytest.fail("repeat import"))
    assert startup_migrations.prepare_startup_state(runtime.root)["question_publications"] == {}
    assert questions() == {f"question:{tid}:older"}


def test_photo_material_serves_exact_selected_bytes_at_existing_artifact_endpoint(tmp_path):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient
    from ouroboros.artifacts import store_task_artifact_bytes
    from ouroboros.gateway import task_archive
    from ouroboros.gateway.tasks import api_task_artifact
    from ouroboros.tools.core_artifacts import _escalate
    from tests.test_quiz_answer import _tool_ctx

    photo = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aXuoAAAAASUVORK5CYII=")
    ctx = _tool_ctx(tmp_path)
    stored = store_task_artifact_bytes(tmp_path, "root-1", name="photo.png", data=photo)
    assert _escalate(ctx, question="Is this the requested image?", assumption="Continue documenting", materials=[{"name": "photo.png"}]).startswith("OK:")
    [question] = quiz_states(tmp_path, "root-1").values()
    [material] = question["materials"]
    assert material["sha256"] == stored["sha256"]
    app = Starlette(routes=[Route("/api/tasks/{task_id}/artifacts/{name}", api_task_artifact)])
    app.state.drive_root = tmp_path
    reply = TestClient(app).get(material["download_url"])
    if not task_archive.CONFINED:  # owner-approved #1297: no confined open, a typed refusal, never unconfined bytes
        assert reply.status_code == 503 and reply.json()["reason_code"] == "artifact_unavailable"
        return
    assert reply.status_code == 200 and reply.content == photo
    assert reply.headers["content-type"] == "image/png"


@pytest.mark.serial
def test_explicit_close_releases_existing_parked_worker_only_after_native_grant(tmp_path):
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.owner_quiz import record_asked
    from ouroboros.owner_wait import append_wake_notice, set_owner_wait, worker_owner_wait
    from tests.test_owner_wait import context

    ctx = context(tmp_path)
    record_asked(tmp_path, "root-1", quiz_id="q1", question="Proceed?", options=[], wait_for_answer=True)
    checkpoint = {"wait_id": "wait-close", "quiz_id": "q1"}
    set_owner_wait(tmp_path, "root-1", {**checkpoint, "state": "waiting"})
    commands, events, outcome = queue_module.Queue(), queue_module.Queue(), []
    thread = threading.Thread(target=lambda: outcome.append(worker_owner_wait(2, commands, events, ctx, checkpoint)))
    thread.start()
    try:
        park = events.get(timeout=2)
        identity = {key: park[key] for key in ("type", "task_id", "task_attempt", "wait_id")}
        commands.put({**identity, "phase": "parked"})
        record_closed(tmp_path, "root-1", quiz_id="q1", reason="The action is withdrawn", actor_task_id="root-1")
        resume = events.get(timeout=3)
        assert resume["phase"] == "resume" and resume["resume_reason"] == "question_closed"
        assert not outcome and load_task_result(tmp_path, "root-1")["owner_wait"]["state"] == "waiting"
        # Simulate the existing supervisor's capacity grant, not a self-grant.
        set_owner_wait(tmp_path, "root-1", {**checkpoint, "state": "resumed"}, checkpoint["wait_id"])
        commands.put({**identity, "phase": "resume_granted"})
        thread.join(timeout=2)
        assert outcome == ["question_closed"]
        assert load_task_result(tmp_path, "root-1")["owner_wait"]["resume_reason"] == "question_closed"
        assert drain_owner_entries(tmp_path, "root-1", set()) == []
        notices = []
        append_wake_notice(ctx, checkpoint, outcome[0], notices)
        assert "The action is withdrawn" in notices[0]["content"]
        assert "without an owner answer" in notices[0]["content"]
        assert "remains open" not in notices[0]["content"]
    finally:
        commands.put({"type": "owner_wait", "phase": "resume_granted", "task_id": "root-1", "task_attempt": 1, "wait_id": "wait-close"})
        thread.join(timeout=2)


def test_typed_owed_beats_stale_discharge_and_backoff_survives_more_than_five_attempts(runtime, monkeypatch):
    from ouroboros.owner_quiz import quiz_needs_retention
    from ouroboros.obligations import members
    from ouroboros import owner_mailbox
    import datetime

    prepare(runtime)
    monkeypatch.setattr(owner_mailbox, "write_owner_message", lambda *a, **kw: False)
    assert _answer(runtime, "q", request_id="keep", index=1)[0] == 503
    def stale_stamp(quizzes):
        quizzes["q"]["discharged_at"] = "2020-01-01T00:00:00+00:00"
        return True
    _mutate_projection(runtime.root, runtime.task["id"], stale_stamp)
    assert quiz_needs_retention(block(runtime))
    assert runtime.task["id"] in members(runtime.root, "owner_questions")
    original = block(runtime)
    logical_id = original["delivery"]["logical_id"]
    for attempt in range(2, 10):
        due(runtime)
        assert recover(runtime) == {"attempted": 1, "delivered": 0, "pending": 1}
        retained = block(runtime)
        delivery = retained["delivery"]
        assert delivery["state"] == "owed" and delivery["attempts"] == attempt
        assert delivery["logical_id"] == logical_id
        delay = datetime.datetime.fromisoformat(delivery["next_retry_at"]) - datetime.datetime.fromisoformat(delivery["last_attempt_at"])
        assert delay.total_seconds() == min(3600, 30 * 2 ** min(attempt - 1, 7))
        assert recover(runtime)["attempted"] == 0
    assert block(runtime)["answered_at"] == original["answered_at"]
    assert quiz_needs_retention({"state": "open", "delivery": {"kind": "publication", "state": "delivered"}})
    assert not quiz_needs_retention({"state": "closed", "delivery": {"kind": "closure", "state": "delivered"}})


def test_late_enqueue_failure_is_not_acknowledged_as_delivery_on_automatic_retry(runtime, monkeypatch):
    prepare(runtime)
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(runtime.bridge, "enqueue_local_message", lambda *a, **kw: (_ for _ in ()).throw(OSError("queue refused")))
    status, reply = _answer(runtime, "q", request_id="queue-error", index=1)
    assert status == 503 and reply["question_delivery"]["state"] == "owed"
    due(runtime)
    assert recover(runtime) == {"attempted": 1, "delivered": 0, "pending": 1}
    retained = block(runtime)
    assert retained["delivery"]["reason_code"] == "late_answer_delivery_unknown"
    assert not retained.get("discharged_at") and runtime.bridge._inbox.empty()
    rows = chat_rows(runtime.root)
    assert len([row for row in rows if row.get("client_message_id") == "quiz_late_answer:task-quiz:q"]) == 1
    # No retry learns more than the lost witness: automatic attempts stop, the debt stays current,
    # and the card says unknown rather than pending.
    from ouroboros.obligations import members
    from ouroboros.project_dialogue import project_question_pointer
    due(runtime)
    assert recover(runtime)["attempted"] == 0 and block(runtime)["delivery"]["attempts"] == 2
    assert runtime.task["id"] in members(runtime.root, "owner_questions")
    pointer = project_question_pointer({"task_id": runtime.task["id"], "quiz_id": "q"}, block(runtime),
                                       {"id": "p", "chat_id": 1, "name": "Project"})
    assert pointer["text"] == "Answer recorded · delivery unknown in Project"


def unknown_late_delivery(runtime, monkeypatch):
    """An accepted late answer whose dispatch witness was lost after acceptance."""
    prepare(runtime)
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(runtime.bridge, "enqueue_local_message", lambda *a, **kw: (_ for _ in ()).throw(OSError("queue refused")))
    assert _answer(runtime, "q", request_id="queue-error", index=1)[0] == 503
    due(runtime)
    recover(runtime)
    assert block(runtime)["delivery"]["reason_code"] == "late_answer_delivery_unknown"


def _decide(runtime, **body):
    from ouroboros.gateway.task_decision import answer_decision
    return asyncio.run(answer_decision(runtime.root, {"decision_id": f"quiz:{runtime.task['id']}:q", **body}))


def test_owner_removes_an_unknown_late_answer_from_current_questions_without_resend(runtime, monkeypatch):
    from ouroboros.obligations import members
    from ouroboros.project_dialogue import project_question_pointer

    unknown_late_delivery(runtime, monkeypatch)
    before, history = block(runtime), chat_rows(runtime.root)
    assert runtime.task["id"] in members(runtime.root, "owner_questions")
    assert _decide(runtime, request_id="rm-1", action="dismiss", comment="no")[1]["reason_code"] == "dismiss_payload_invalid"
    assert _decide(runtime, request_id="rm-1", action="dismiss", decision_id="quiz:task-quiz:other")[0] == 404
    status, reply = _decide(runtime, request_id="rm-1", action="dismiss")
    assert status == 200 and reply["ok"] is True and reply["duplicate"] is False and reply["state"] == "answered"
    assert reply["question_delivery"] == {"kind": "answer", "state": "owed", "dismissed": True,
                                          "reason_code": "late_answer_delivery_unknown"}
    removed = block(runtime)
    assert (removed["answered_index"], removed["request_id"]) == (before["answered_index"], before["request_id"]) == (1, "queue-error")
    assert removed["delivery"]["state"] == "owed" and "delivered_at" not in removed["delivery"]
    assert not removed.get("discharged_at") and removed["delivery"]["dismissed_request_id"] == "rm-1"
    assert runtime.task["id"] not in members(runtime.root, "owner_questions")  # leaves current questions
    assert [frame["question_delivery"] for frame in runtime.frames if frame.get("type") == "quiz_state"][-1] == reply["question_delivery"]
    # Repeated clicks, another client's stale removal and the original answer's retried request all
    # settle on the recorded fact; nothing is attempted, written to chat or resent.
    assert _decide(runtime, request_id="rm-1", action="dismiss")[1]["duplicate"] is True
    assert _decide(runtime, request_id="rm-2", action="dismiss")[1]["duplicate"] is True
    retried = _answer(runtime, "q", request_id="queue-error", index=1)
    assert retried[0] == 200 and retried[1]["duplicate"] is True and retried[1]["question_delivery"]["dismissed"] is True
    due(runtime)
    assert recover(runtime)["attempted"] == 0 and runtime.bridge._inbox.empty()
    unchanged = lambda delivery: {key: value for key, value in delivery.items() if key != "next_retry_at"}  # due() clears it
    assert chat_rows(runtime.root) == history
    assert unchanged(block(runtime)["delivery"]) == unchanged(removed["delivery"])  # no attempt, failure or settlement
    pointer = project_question_pointer({"task_id": runtime.task["id"], "quiz_id": "q"}, block(runtime),
                                       {"id": "p", "chat_id": 1, "name": "Project"})
    assert pointer["text"] == "Answer recorded · delivery unknown in Project"  # the record stays honest


def test_removal_refuses_anything_but_an_unknown_late_answer(runtime):
    prepare(runtime)
    status, reply = _decide(runtime, request_id="rm-open", action="dismiss")
    assert status == 409 and reply["error"] == "delivery_not_unknown" and reply["state"] == "open"
    assert _answer(runtime, "q", request_id="live", index=0)[0] == 200
    from ouroboros.owner_mailbox import drain_owner_entries, acknowledge_transcript_entry
    [entry] = drain_owner_entries(runtime.root, runtime.task["id"], set())
    assert acknowledge_transcript_entry(runtime.root, runtime.task["id"], entry)
    status, reply = _decide(runtime, request_id="rm-delivered", action="dismiss")
    assert status == 409 and reply["error"] == "delivery_not_unknown"
    assert reply["question_delivery"] == {"kind": "answer", "state": "delivered"} and "dismissed_at" not in block(runtime)["delivery"]


def test_removal_refuses_an_answer_whose_delivery_still_retries(runtime, monkeypatch):
    """Only an unknown dispatch is removable: removing a retrying debt would silence its delivery."""
    prepare(runtime)
    monkeypatch.setattr(queue, "RUNNING", {})
    real_accept = message_bus.accept_chat
    def write_then_raise(*args, **kwargs):
        receipt = real_accept(*args, **kwargs)
        if kwargs.get("client_message_id") == "quiz_late_answer:task-quiz:q":
            raise OSError("write returned an error after durable append")
        return receipt
    with monkeypatch.context() as failing:
        failing.setattr(message_bus, "accept_chat", write_then_raise)
        assert _answer(runtime, "q", request_id="committed", index=1)[0] == 503
    owed = block(runtime)["delivery"]
    assert owed["state"] == "owed" and owed["reason_code"] != "late_answer_delivery_unknown"
    status, reply = _decide(runtime, request_id="rm-retrying", action="dismiss")
    assert status == 409 and reply["error"] == "delivery_not_unknown"
    assert "dismissed_at" not in block(runtime)["delivery"]
    due(runtime)
    assert recover(runtime) == {"attempted": 1, "delivered": 1, "pending": 0}, "the retry still delivers"


def test_proven_undispatched_acceptance_recovers_its_saved_snapshot_automatically(runtime, monkeypatch):
    prepare(runtime)
    monkeypatch.setattr(queue, "RUNNING", {})
    real_accept = message_bus.accept_chat
    def write_then_raise(*args, **kwargs):
        receipt = real_accept(*args, **kwargs)
        if kwargs.get("client_message_id") == "quiz_late_answer:task-quiz:q":
            raise OSError("write returned an error after durable append")
        return receipt
    monkeypatch.setattr(message_bus, "accept_chat", write_then_raise)
    assert _answer(runtime, "q", request_id="committed", index=1, comment=" exact ")[0] == 503
    assert runtime.bridge._inbox.empty()
    monkeypatch.setattr(message_bus, "accept_chat", real_accept)
    due(runtime)
    assert recover(runtime) == {"attempted": 1, "delivered": 1, "pending": 0}
    [accepted] = [row for row in chat_rows(runtime.root) if row.get("client_message_id") == "quiz_late_answer:task-quiz:q"]
    queued = runtime.bridge._inbox.get_nowait()
    # The retry's attempt stamps differ; saved ingress authority wins verbatim.
    assert queued["task_metadata"]["late_answer"] == accepted["late_answer"]
    assert accepted["late_answer"]["answer_snapshot"]["quiz"]["delivery"]["attempts"] == 1
    assert block(runtime)["delivery"]["attempts"] == 2
    assert runtime.bridge._inbox.empty()


def test_question_published_in_main_then_converted_retains_publication_destination(runtime, monkeypatch):
    from ouroboros.projects_registry import bind_task_to_project, create_project

    prepare(runtime)
    assert block(runtime)["publication_chat_id"] == 1
    project = create_project(runtime.root, "converted", name="Converted source")
    bind_task_to_project(runtime.root, runtime.task["id"], project["id"], origin={"absent": "system"})
    monkeypatch.setattr(queue, "RUNNING", {})
    write_task_result(runtime.root, runtime.task["id"], "completed", project_id=project["id"], chat_id=project["chat_id"])
    status, reply = _answer(runtime, "q", request_id="after-conversion", index=1)
    assert status == 200 and reply["forwarded"]
    queued = runtime.bridge._inbox.get_nowait()
    assert queued["chat_id"] == 1
    assert block(runtime)["publication_chat_id"] == 1
    [fact] = _facts(runtime)
    assert fact["chat_id"] == 1
    assert [row for row in runtime.frames if row.get("type") == "chat" and row.get("chat_id") == 1]


@pytest.mark.parametrize("stamp", ["bad-time", "2020-01-01T00:00:00"])
def test_invalid_retry_time_repairs_to_bounded_canonical_attempt(runtime, stamp):
    from ouroboros.owner_quiz import claim_quiz_delivery, quiz_delivery_due

    prepare(runtime)
    record_closed(runtime.root, runtime.task["id"], quiz_id="q", reason="Withdrawn", actor_task_id=runtime.task["id"])
    def invalid(quizzes):
        quizzes["q"]["delivery"]["next_retry_at"] = stamp
        return True
    _mutate_projection(runtime.root, runtime.task["id"], invalid)
    assert quiz_delivery_due(block(runtime))
    claimed = claim_quiz_delivery(runtime.root, runtime.task["id"], "q")
    assert claimed["delivery"]["attempts"] == 1
    assert not quiz_delivery_due(claimed)


def test_closed_fact_outlives_delivery_gc_until_paired_wait_can_observe_it(tmp_path):
    from ouroboros.owner_quiz import mark_discharged, record_asked
    from ouroboros.owner_wait import _question_closed, set_owner_wait
    from tests.test_owner_wait import context

    ctx = context(tmp_path)
    record_asked(tmp_path, "root-1", quiz_id="q1", question="Proceed?", options=[], wait_for_answer=True)
    set_owner_wait(tmp_path, "root-1", {"quiz_id": "q1", "wait_id": "w", "state": "waiting"})
    record_closed(tmp_path, "root-1", quiz_id="q1", reason="Withdrawn", actor_task_id="root-1")
    mark_discharged(tmp_path, "root-1", "q1")
    for index in range(20):
        record_asked(tmp_path, "root-1", quiz_id=f"new-{index}", question="Follow-up", options=[])
    assert quiz_states(tmp_path, "root-1")["q1"]["closed_reason"] == "Withdrawn"
    assert _question_closed(ctx, {"quiz_id": "q1"})
