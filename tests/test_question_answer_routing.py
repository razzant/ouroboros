"""Answers keep their addressee through holds, Continue, last drains and crashes."""
import queue
from types import SimpleNamespace

import pytest

from ouroboros.gateway import task_decision as td
from ouroboros.owner_quiz import record_asked, record_publication, quiz_states
from ouroboros.task_store import patch_document
from ouroboros.question_answers import admitted_successor
from tests._question_answer_shared import _late_bridge, _post, continue_to, drain, runtime, stage_question_mail
from tests.test_quiz_late_answer import _inbox


def answer(app):
    return _post(app, {"decision_id": "quiz:asker:q", "request_id": "r", "option_index": 1, "comment": "exact answer"})


def debt(root):
    return quiz_states(root, "asker")["q"]["delivery"]


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


def test_recalled_answer_and_card_press_keep_their_content_without_settling_twice(tmp_path, monkeypatch):
    from ouroboros import card_store, owner_mailbox, question_answers
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_relay import quote_rows

    app, bridge, _q, _task = runtime(tmp_path, monkeypatch)
    assert answer(app).status_code == 200
    press_id, press = "card_input:card:press", "Pressed Save on the draft."
    assert owner_mailbox.write_owner_message(tmp_path, press, "asker", msg_id=press_id,
        kind=owner_mailbox.KIND_CARD_INPUT, card_delivery={"card_id": "card", "request_id": "press"})
    original, _ = drain(tmp_path, "asker")
    receipt = owner_mailbox._ack_path(tmp_path, "asker").read_bytes()
    saved_debt = debt(tmp_path)
    [handle] = original._held_owner_messages

    def duplicate(*_args, **_kwargs):
        pytest.fail("a restored transcript settled the original action twice")

    monkeypatch.setattr(card_store, "acknowledge_press", duplicate)
    monkeypatch.setattr(question_answers, "acknowledge_answer", duplicate)
    messages, events = [], queue.Queue()
    ctx = SimpleNamespace(task_id="asker", task_metadata={}, drive_root=tmp_path, budget_drive_root=tmp_path,
                          _restore_mailbox=True, _mailbox_content_ids=set())
    _drain_incoming_messages(messages, queue.Queue(), tmp_path, "asker", events, set(), owner_ctx=ctx)
    assert sum(str(row.get("content") or "").count(press) for row in messages) == 1
    assert [row["content"] for row in ctx._owner_directives] == ["exact answer", press]
    [quote], error = quote_rows(ctx, [handle])
    assert not error and quote["text"] == "exact answer" and "Which database?" in quote["question_frame"]
    assert not list(events.queue) and not _inbox(bridge)
    assert owner_mailbox._ack_path(tmp_path, "asker").read_bytes() == receipt and debt(tmp_path) == saved_debt


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


@pytest.mark.parametrize("written_by_current_producer", [True, False], ids=["fresh", "legacy-unassigned"])
def test_first_closure_route_distinguishes_creation_from_legacy_ack_gap(tmp_path, monkeypatch, written_by_current_producer):
    from ouroboros import owner_mailbox as mail
    from ouroboros.owner_quiz import record_closed

    app, _bridge, q, task = runtime(tmp_path, monkeypatch, running=False)
    continue_to(tmp_path)
    q.RUNNING["successor"] = {"task": {**task, "id": "successor"}}
    ack = mail._ack_path(tmp_path, "asker")
    ack.parent.mkdir(parents=True, exist_ok=True)
    ack.write_bytes(b"{historical-torn-receipt")
    if not written_by_current_producer:
        assert record_closed(tmp_path, "asker", quiz_id="q", reason="Done.", actor="owner", request_id="close")["ok"]
        def remove_creation_stamp(row):
            row["owner_quiz"]["q"]["delivery"].pop("first_route_pending", None)
            return row
        patch_document(tmp_path, "asker", remove_creation_stamp)
    assert _post(app, {"decision_id": "quiz:asker:q", "request_id": "close", "action": "close", "reason": "Done."}).status_code == 200
    assert bool(mail.drain_owner_entries(tmp_path, "successor")) is written_by_current_producer
    assert not debt(tmp_path).get("read_at")  # route admission is not consumption
    assert ack.read_bytes() == b"{historical-torn-receipt"


@pytest.mark.parametrize("holder", ["running", "paused", "held"])
def test_closure_reaches_the_admitted_reader_once_with_its_reason(tmp_path, monkeypatch, holder):
    from ouroboros.owner_mailbox import drain_owner_entries

    app, bridge, q, task = runtime(tmp_path, monkeypatch, running=False)
    continue_to(tmp_path)
    successor = {**task, "id": "successor"}
    q.RUNNING["neighbour"] = {"task": {**task, "id": "neighbour"}}
    if holder == "running":
        q.RUNNING["successor"] = {"task": successor}
    else:
        successor.update({"_owner_hold": True} if holder == "held" else {"_budget_pause": {"reason": "budget"}})
        q.PENDING.append(successor)
    reason = "The original requirement was withdrawn."
    request = {"decision_id": "quiz:asker:q", "request_id": "close", "action": "close", "reason": reason}
    for _ in range(2):
        response = _post(app, request)
        assert response.status_code == 200, response.text
    [entry] = drain_owner_entries(tmp_path, "successor")
    assert entry["msg_id"] == "quiz_closed:q" and entry["kind"] == "quiz_closed"
    assert entry["text"].count(reason) == 1 and "no option was chosen" in entry["text"]
    assert not drain_owner_entries(tmp_path, "asker") and not drain_owner_entries(tmp_path, "neighbour")
    assert not _inbox(bridge) and debt(tmp_path)["recipient_task_id"] == "successor"
    ctx, messages = drain(tmp_path, "successor")
    assert reason in str(messages) and [r["content"] for r in ctx._owner_directives] == [reason]
    assert not drain(tmp_path, "successor")[1]
    assert debt(tmp_path)["read_at"] and debt(tmp_path)["state"] == "delivered"
    assert "answered_index" not in quiz_states(tmp_path, "asker")["q"]


@pytest.mark.parametrize("retired_mailbox", [False, True], ids=["old-wake-receipt", "settled-unread-capture"])
def test_closed_card_is_published_once_when_the_first_recipient_ends_before_read(tmp_path, monkeypatch, retired_mailbox):
    from ouroboros import owner_mailbox as mail, working_checkpoint
    from tests._dialogue import chat_rows

    app, bridge, q, task = runtime(tmp_path, monkeypatch, running=False)
    frames = []
    bridge._broadcast_fn = frames.append
    continue_to(tmp_path)
    q.PENDING.append({**task, "id": "successor", "_owner_hold": True})
    reason = "The experiment is complete."
    request = {"decision_id": "quiz:asker:q", "request_id": "close", "action": "close", "reason": reason}
    assert _post(app, request).status_code == 200
    assert debt(tmp_path)["state"] == "owed" and debt(tmp_path)["mailbox_at"] and not debt(tmp_path).get("read_at")
    assert [frame["state"] for frame in frames if frame.get("type") == "quiz_state"] == ["closed"]
    mid = debt(tmp_path)["msg_id"]
    # The old wake named this letter, but the queued reader never held its words.
    assert mail.acknowledge_task_messages(tmp_path, "successor", [mid], wake_id="historical-wake")
    assert not mail.mail_read_state(tmp_path, "successor", mid, transcript_only=True)
    q.PENDING.clear()
    patch_document(tmp_path, "successor", lambda row: {**row, "status": "completed"})
    if retired_mailbox:
        assert mail.cleanup_task_mailbox(tmp_path, "successor", canonical_root=tmp_path)
        assert not mail.mailbox_holds(tmp_path, "successor", mid)
    from ouroboros.owner_continue import owner_sources
    from ouroboros.task_results import load_task_result
    predecessor = load_task_result(tmp_path, "successor", strict=True)
    if retired_mailbox:
        assert predecessor["unread_mailbox"]["read_complete"] and len(predecessor["unread_mailbox"]["rows"]) == 1
    assert not [row for row in owner_sources(tmp_path, predecessor, "successor")["later"] if row["msg_id"] == mid]
    continue_to(tmp_path, asker="successor", successor="next")
    q.RUNNING["next"] = {"task": {**task, "id": "next"}}
    for _ in range(2):
        assert _post(app, request).status_code == 200
    assert debt(tmp_path)["state"] == "owed" and debt(tmp_path)["recipient_task_id"] == "next"
    [letter] = mail.drain_owner_entries(tmp_path, "next", include_acknowledged=True)
    assert letter["msg_id"] == mid and reason in letter["text"]
    assert len(mail._mailbox_path(tmp_path, "next").read_bytes().splitlines()) == 1
    ctx, limit = stage_question_mail(tmp_path, "next")
    assert debt(tmp_path)["state"] == "owed" and not mail.mail_read_state(tmp_path, "next", mid)
    working_checkpoint.save_ready(limit)
    assert mail.mail_read_state(tmp_path, "next", mid, transcript_only=True) and debt(tmp_path)["read_at"]
    assert debt(tmp_path)["state"] == "delivered" and not drain(tmp_path, "next")[1]
    assert [row["content"] for row in ctx._owner_directives] == [reason]
    assert "answered_at" not in quiz_states(tmp_path, "asker")["q"]
    assert [row["text"] for row in chat_rows(tmp_path) if row.get("type") == "quiz_closed"] == [reason]
    assert len([frame for frame in frames if frame.get("type") == "quiz_state"]) == 1 and not _inbox(bridge)


@pytest.mark.parametrize("effect", ["unwritten", "wake", "wake-drive", "saved", "saved-before-cleanup", "saved-empty", "saved-orphan",
                                    "saved-orphan-sweep", "public-base", "public-base-read", "public-base-conflict", "public-base-duplicate",
                                    "public-base-split-duplicate", "public-base-retired", "public-base-retired-read"])
def test_closure_across_retirement_and_explicit_continue(tmp_path, monkeypatch, effect):
    import asyncio
    from ouroboros import owner_mailbox as mail, question_answers, task_custody, working_checkpoint
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import admit_continuation
    from tests._question_answer_shared import public_base_closure, scheduled_closure_reader
    from tests._question_answer_shared import NONCE, _interrupted

    q, (bridge, frames) = scheduled_closure_reader(tmp_path, monkeypatch)
    reason, mid = "Drop the optional experiment.", "quiz_closed:q"
    source = tmp_path
    if effect == "wake-drive":
        source = tmp_path / "task_drives" / "asker"
        source.mkdir(parents=True)
        q.RUNNING["asker"]["task"]["drive_root"] = str(source)
    request = {"decision_id": "quiz:asker:q", "request_id": "close", "action": "close", "reason": reason}
    if effect.startswith("public-base"):
        public_base_closure(tmp_path, reason, event_id="other" if effect.endswith("conflict") else "q")
        if effect.endswith("duplicate"):
            path = mail._mailbox_path(tmp_path, "asker")
            original_line = path.read_bytes()
            if effect == "public-base-split-duplicate":
                second = tmp_path / "task_drives" / "asker"
                path = mail._mailbox_path(second, "asker")
                path.parent.mkdir(parents=True)
            with path.open("ab") as old_writer:
                old_writer.write(original_line)
    else:
        with monkeypatch.context() as fault:
            if effect == "unwritten":
                fault.setattr(mail, "write_owner_message", lambda *a, **k: False)
            assert asyncio.run(td.answer_decision(tmp_path, request))[0] == 200
        assert debt(tmp_path)["recipient_task_id"] == "asker"
        assert bool(debt(tmp_path)["mailbox_at"]) is (effect != "unwritten")

    if effect.startswith("wake"):
        assert mail.acknowledge_task_messages(source, "asker", [mid], wake_id="historical-wake")
    events = queue.Queue()
    saved = effect.startswith("saved")
    old_read = effect in {"public-base-read", "public-base-retired-read"}
    if saved or old_read:
        ctx, limit = stage_question_mail(tmp_path, "asker", events)
        # Terminal capture can observe unread before the deferred ACK. Its grow-only union survives that ACK.
        assert reason in str(limit.messages) and not mail.mail_read_state(tmp_path, "asker", mid, transcript_only=True)
    def save_without_card_stamp():
        with monkeypatch.context() as fault:
            fault.setattr(question_answers, "acknowledge_answer", lambda *a: None)
            working_checkpoint.save_ready(limit)

    if effect == "saved-empty":
        save_without_card_stamp()
    q.RUNNING.pop("asker")
    _interrupted(tmp_path, "asker")
    result = load_task_result(tmp_path, "asker", strict=True)
    if effect == "unwritten":
        assert result["owner_mailbox"]["rows"] == [] and "unread_mailbox" not in result
    if effect == "public-base-split-duplicate":
        assert result["owner_mailbox"]["rows"] == [original_line.decode().rstrip("\n")] * 2
        assert mail.cleanup_task_mailbox(second, "asker", canonical_root=tmp_path)
    if saved or old_read:
        if effect == "saved-empty":
            assert "unread_mailbox" not in result
        else:
            assert result["unread_mailbox"]["rows"]
            save_without_card_stamp()
        assert mail.mail_read_state(tmp_path, "asker", mid, transcript_only=True)
    assert not debt(tmp_path).get("read_at")
    if effect in {"saved-orphan", "saved-orphan-sweep"}:
        # The public cleanup could stop between unlinking the captured letter and unlinking its ordinary ACK.
        mail._mailbox_path(tmp_path, "asker").unlink()
    if effect == "saved":
        with monkeypatch.context() as fault:
            fault.setattr(question_answers, "_update_delivery", lambda *a, **kw: quiz_states(tmp_path, "asker")["q"])
            assert not mail.cleanup_task_mailbox(tmp_path, "asker")
        assert mail.mail_read_state(tmp_path, "asker", mid, transcript_only=True)
    if effect == "wake-drive":
        settled = task_custody.settle_child_drive(tmp_path, "asker", source, live=lambda _task: False)
        assert settled["status"] == "removed" and not source.exists(), settled
    elif effect == "saved-orphan-sweep":
        assert mail.sweep_settled_owner_mailboxes(tmp_path) == {"removed": ["asker"], "kept": 0}
        assert debt(tmp_path)["read_at"]
    elif effect != "saved-before-cleanup":
        with monkeypatch.context() as version:
            if effect.startswith("public-base-retired"):
                # Older retirement kept the grow-only capture and discarded the exact read state.
                version.setattr(task_custody, "reconcile_closure_mail", lambda *a: True, raising=False)
            assert mail.cleanup_task_mailbox(tmp_path, "asker")
    admitted = admit_continuation("asker", action_nonce=NONCE)
    assert admitted["ok"], admitted
    if effect == "saved-before-cleanup":
        assert not debt(tmp_path).get("read_at")  # Continue can use the ordinary ACK without rewriting its source
        assert mail.cleanup_task_mailbox(tmp_path, "asker")
    assert not mail._mailbox_path(source, "asker").exists() and not mail._ack_path(source, "asker").exists()
    successor = admitted["successor_task_id"]
    assert admitted_successor(tmp_path, "asker") == successor
    [queued] = [row for row in q.PENDING if row["id"] == successor]
    carried = [row["content"] for row in queued["metadata"]["owner_corpus"] if row.get("msg_id") == mid]
    assert carried == ([reason] if saved else [])
    for _ in range(2):
        assert asyncio.run(td.answer_decision(tmp_path, request))[0] == 200

    ctx, limit = stage_question_mail(tmp_path, successor, events)
    working_checkpoint.save_ready(limit)
    if effect.endswith(("conflict", "duplicate")) or effect.startswith("public-base-retired"):
        assert not mail.mailbox_holds(tmp_path, successor, mid)
        assert reason not in str(limit.messages) and not ctx._owner_directives
        assert debt(tmp_path)["reason_code"] == "mailbox_read_unknown" and not debt(tmp_path).get("read_at")
    else:
        assert str(limit.messages).count(reason) == int(not (saved or old_read))
        assert [r["content"] for r in ctx._owner_directives] == ([reason] if effect in {"unwritten", "wake", "wake-drive"} else [])
        assert debt(tmp_path)["state"] == "delivered" and debt(tmp_path)["read_at"]
        assert not drain(tmp_path, successor)[1]
    assert not list(events.queue)
    assert len([frame for frame in frames if frame.get("type") == "quiz_state"]) == 1 and not _inbox(bridge)


@pytest.mark.parametrize("duplicate", [False, True], ids=["unique-letter", "duplicate-letter"])
def test_continue_carries_only_a_proven_closure_despite_torn_ack(tmp_path, monkeypatch, duplicate):
    import asyncio
    from ouroboros import owner_mailbox as mail, question_answers, working_checkpoint
    from supervisor.continuation_admission import admit_continuation
    from tests._question_answer_shared import NONCE, _interrupted, scheduled_closure_reader

    q, _ = scheduled_closure_reader(tmp_path, monkeypatch)
    reason, mid = "Drop the optional experiment.", "quiz_closed:q"
    request = {"decision_id": "quiz:asker:q", "request_id": "close", "action": "close", "reason": reason}
    assert asyncio.run(td.answer_decision(tmp_path, request))[0] == 200
    _, limit = stage_question_mail(tmp_path, "asker")
    assert reason in str(limit.messages)
    with monkeypatch.context() as fault:
        fault.setattr(question_answers, "_update_delivery", lambda *a, **kw: {})
        working_checkpoint.save_ready(limit)
    q.RUNNING.pop("asker")
    _interrupted(tmp_path, "asker")

    path = mail._ack_path(tmp_path, "asker")
    with path.open("ab") as stream:
        stream.write(b'{"msg_id":"other",')
    original = path.read_bytes()
    mailbox = mail._mailbox_path(tmp_path, "asker")
    if duplicate:
        letter = mailbox.read_bytes()
        with mailbox.open("ab") as stream:
            stream.write(letter)
    original_mail = mailbox.read_bytes()
    assert mail.mail_read_state(tmp_path, "asker", mid, transcript_only=True) is True
    assert mail.mail_read_state(tmp_path, "asker", "other", transcript_only=True) is None
    assert not debt(tmp_path).get("read_at")
    admitted = admit_continuation("asker", action_nonce=NONCE)
    assert admitted["ok"], admitted
    successor = admitted["successor_task_id"]
    [queued] = [row for row in q.PENDING if row["id"] == successor]
    carried = [row["content"] for row in queued["metadata"]["owner_corpus"] if row.get("msg_id") == mid]
    assert carried == ([] if duplicate else [reason])
    assert not debt(tmp_path).get("read_at")
    assert asyncio.run(td.answer_decision(tmp_path, request))[0] == 200
    assert bool(debt(tmp_path).get("read_at")) is (not duplicate)
    assert debt(tmp_path)["recipient_task_id"] == "asker"
    assert not mail.mailbox_holds(tmp_path, successor, mid)
    assert not mail.cleanup_task_mailbox(tmp_path, "asker")
    assert path.read_bytes() == original and mailbox.read_bytes() == original_mail


def test_ended_closure_reader_with_unknown_ack_does_not_repost_without_unread_capture(tmp_path, monkeypatch):
    from ouroboros import owner_mailbox as mail

    app, _bridge, q, task = runtime(tmp_path, monkeypatch, running=False)
    continue_to(tmp_path)
    q.RUNNING["successor"] = {"task": {**task, "id": "successor"}}
    request = {"decision_id": "quiz:asker:q", "request_id": "close", "action": "close", "reason": "Done."}
    assert _post(app, request).status_code == 200
    assert debt(tmp_path)["recipient_task_id"] == "successor"
    mail._ack_path(tmp_path, "successor").write_bytes(b"{torn")
    q.RUNNING.pop("successor")
    patch_document(tmp_path, "successor", lambda row: {**row, "status": "completed"})
    continue_to(tmp_path, asker="successor", successor="next")
    q.RUNNING["next"] = {"task": {**task, "id": "next"}}
    assert _post(app, request).status_code == 200
    assert not mail.drain_owner_entries(tmp_path, "next")
    assert debt(tmp_path)["recipient_task_id"] == "successor" and not debt(tmp_path).get("read_at")


@pytest.mark.parametrize("newest,expected", [("completed", "successor"), ("running", "next")])
def test_answer_follows_the_latest_live_admitted_reader(tmp_path, monkeypatch, newest, expected):
    app, bridge, q, task = runtime(tmp_path, monkeypatch, running=False)
    continue_to(tmp_path)
    continue_to(tmp_path, asker="successor", successor="next")
    patch_document(tmp_path, "next", lambda row: {**row, "status": newest})
    q.RUNNING["successor"] = {"task": {**task, "id": "successor"}}
    if newest == "running":
        q.RUNNING["next"] = {"task": {**task, "id": "next"}}
    assert answer(app).status_code == 200 and debt(tmp_path)["recipient_task_id"] == expected
    assert not _inbox(bridge)


@pytest.mark.parametrize("published", [False, True])
@pytest.mark.parametrize("binding", ["admitted", "bound", "mismatched", "cycle", "admitted_during_retirement", "ended"])
def test_closure_stays_owed_until_its_reader_line_ends(tmp_path, monkeypatch, binding, published):
    from ouroboros.owner_continue import binding_sha
    from ouroboros.owner_quiz import record_closed, record_quiz_delivery_failure, retire_closure_owed_nowhere

    _app, _bridge, _q, _task = runtime(tmp_path, monkeypatch, running=False)
    record_asked(tmp_path, "asker", quiz_id="lost-card", question="Which one?", options=["one", "two"], chat_id=1)
    assert record_closed(tmp_path, "asker", quiz_id="lost-card", actor="owner", reason="No longer needed.",
                         request_id="close-lost-card")["ok"]
    record_quiz_delivery_failure(tmp_path, "asker", "lost-card", "publication_address_unknown")
    continue_to(tmp_path, "bound" if binding == "bound" else "admitted")
    if binding == "mismatched":
        patch_document(tmp_path, "successor", lambda row: {**row, "continuation_admission": {}})
    if binding in ("ended", "cycle", "admitted_during_retirement"):
        patch_document(tmp_path, "successor", lambda row: {**row, "status": "completed"})
    if binding == "admitted_during_retirement":
        from ouroboros import question_answers
        from ouroboros.task_results import write_task_result

        ended = question_answers.reader_line_ended

        def admit_before_commit(*args, **kwargs):
            assert ended(*args, **kwargs)
            new = {"predecessor_task_id": "successor", "successor_task_id": "next", "action_nonce": "forward"}
            admission = {"binding": new, "binding_sha256": binding_sha(new)}
            write_task_result(tmp_path, "next", status="running", chat_id=1)
            patch_document(tmp_path, "next", lambda row: {**row, "continuation_admission": admission})
            patch_document(tmp_path, "successor", lambda row: {**row, "continued_by": {
                **admission, "state": "admitted", "successor_task_id": "next"}})
            return True

        monkeypatch.setattr(question_answers, "reader_line_ended", admit_before_commit)
    if binding == "cycle":
        back = {"predecessor_task_id": "successor", "successor_task_id": "asker", "action_nonce": "back"}
        admission = {"binding": back, "binding_sha256": binding_sha(back)}
        patch_document(tmp_path, "successor", lambda row: {**row, "continued_by": {
            **admission, "state": "admitted", "successor_task_id": "asker"}})
        patch_document(tmp_path, "asker", lambda row: {**row, "continuation_admission": admission})
    if published:
        from ouroboros.owner_quiz import mark_discharged, mark_quiz_history_written
        record_publication(tmp_path, "asker", "lost-card", 1)
        mark_quiz_history_written(tmp_path, "asker", "lost-card")
        settled = mark_discharged(tmp_path, "asker", "lost-card", no_reader=True)
        assert bool(settled) is (binding == "ended")
    else:
        assert retire_closure_owed_nowhere(tmp_path, "asker", "lost-card") is (binding == "ended")
    block = quiz_states(tmp_path, "asker")["lost-card"]
    assert block["closed_reason"] == "No longer needed."
    assert block["delivery"]["state"] == (("delivered" if published else "no_target") if binding == "ended" else "owed")
    assert not block["delivery"].get("read_at")


@pytest.mark.parametrize("binding", ["bound", "mismatched", "cycle"])
def test_closure_never_enters_an_unadmitted_or_cyclic_reader_mailbox(tmp_path, monkeypatch, binding):
    from ouroboros.owner_continue import binding_sha
    from ouroboros.owner_mailbox import drain_owner_entries

    app, bridge, q, task = runtime(tmp_path, monkeypatch, running=False)
    continue_to(tmp_path, "bound" if binding == "bound" else "admitted")
    if binding == "mismatched":
        patch_document(tmp_path, "successor", lambda row: {**row, "continuation_admission": {}})
    if binding == "cycle":
        back = {"predecessor_task_id": "successor", "successor_task_id": "asker", "action_nonce": "back"}
        admission = {"binding": back, "binding_sha256": binding_sha(back)}
        patch_document(tmp_path, "successor", lambda row: {**row, "continued_by": {
            **admission, "state": "admitted", "successor_task_id": "asker"}})
        patch_document(tmp_path, "asker", lambda row: {**row, "continuation_admission": admission})
    for tid in ("successor", "other"):
        q.RUNNING[tid] = {"task": {**task, "id": tid}}
    response = _post(app, {"decision_id": "quiz:asker:q", "request_id": "close", "action": "close", "reason": "Done."})
    assert response.status_code == 200, response.text
    assert debt(tmp_path)["state"] == "owed" and not debt(tmp_path).get("mailbox_at")
    assert not _inbox(bridge) and all(not drain_owner_entries(tmp_path, tid) for tid in ("asker", "other", "successor"))


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
