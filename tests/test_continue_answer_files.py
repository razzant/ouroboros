"""A Telegram Reply's file answers a card with its caption (owner's choice, 18 Nov): the owner's Continue of the
work that read that answer carries it as the same ONE recorded answer, whether the Reply had words, was a file alone,
or its file never arrived, and the successor holds its own copy of the bytes (or the reason).

Driven through the real consumers, as ``tests/test_owner_continue.py``: an interrupted root whose mailbox holds the
accepted answer's letter, ``admit_continuation``, the successor's first request and its own owner corpus, and a
second hop that keeps the same answer."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _install_queue
from tests._owner_continue_helpers import NONCE, _interrupted

_PNG = b"\x89PNG\r\n\x1a\n" + b"pixels" * 4
ASKER = "larch-desk"


def _card(refs, comment=""):
    return {"state": "answered", "quiz_id": "which", "question": "Which screen?", "options": ["Amber", "Violet"],
            "asked_at": "2040-01-01T00:00:00Z", "answered_at": "2040-01-01T00:05:00Z", "request_id": "tg:9:m:42:7",
            "attachments": refs, **({"comment": comment} if comment else {})}


def _answered_letter(tmp_path, *, comment="", late=False):
    """The reading work's mailbox holds the accepted Reply answer exactly as the host wrote it: the card frame (a live
    asker's ``quiz_answer`` letter) or the owner's message with typed ``late_answer`` provenance (a head a late answer
    started), its file staged in that work's own store (``server_owner_routing._stage_mailbox_attachments``)."""
    from ouroboros import chat_uploads
    from ouroboros.owner_mailbox import KIND_QUIZ_ANSWER, write_owner_message
    from ouroboros.owner_quiz import recorded_answer_frame
    from ouroboros.server_owner_routing import _stage_mailbox_attachments

    ref = chat_uploads.store_upload(_PNG, "photo.png", data_dir=tmp_path)[1]
    card = {**_card([ref], comment), **({"answered_after_terminal": True} if late else {})}
    reader = "pred-1" if late else ASKER
    note, manifest, _ = _stage_mailbox_attachments(
        None, tmp_path, reader, {"chat_attachment_uploads": chat_uploads.staging_specs([ref], tmp_path)})
    if late:
        assert write_owner_message(tmp_path, comment + note, reader, msg_id=f"quiz_late_answer:{ASKER}:which:{reader}",
                                   client_message_id=f"quiz_late_answer:{ASKER}:which", attachment_manifest=manifest,
                                   late_answer={"task_id": ASKER, "quiz_id": "which",
                                                "answer_snapshot": {"task_id": ASKER, "quiz": card}})
    else:
        assert write_owner_message(
            tmp_path, recorded_answer_frame(card, asker=ASKER, reader=ASKER) + note, reader, kind=KIND_QUIZ_ANSWER,
            msg_id=f"quiz_answer:{ASKER}:which", attachment_manifest=manifest,
            quiz_delivery={"task_id": ASKER, "quiz_id": "which", "answer_snapshot": card})
    return ref, reader


@pytest.mark.parametrize("flavor", ["live-asker", "read-by-asker", "late-head"])
@pytest.mark.parametrize("comment", ["", "this one, as in the photo"], ids=["file-alone", "caption"])
def test_continue_carries_the_answer_with_its_file_to_the_successor_and_a_second_hop_keeps_it(
        tmp_path, monkeypatch, comment, flavor):
    from ouroboros.loop_messages import _initialize_owner_directives
    from ouroboros.owner_mailbox import acknowledge_transcript_entry
    from supervisor.continuation_admission import admit_continuation

    _, _, workers = _install_queue(tmp_path, monkeypatch)
    ref, reader = _answered_letter(tmp_path, comment=comment, late=flavor == "late-head")
    receipt = {"owner_quiz": {"which": {**_card([ref], comment), "discharged_at": "2040-01-01T00:05:02Z", "delivery": {
        "kind": "answer", "state": "delivered", "recipient_task_id": ASKER, "msg_id": f"quiz_answer:{ASKER}:which",
        "recipient_drive": str(tmp_path), "mailbox_at": "2040-01-01T00:05:01Z", "history_at": "2040-01-01T00:05:01Z",
        "read_at": "2040-01-01T00:05:02Z"}}}} if flavor == "live-asker" else {}  # the asker's own exact read receipt
    _interrupted(tmp_path, task_id=reader, **receipt)
    if flavor == "read-by-asker":  # read before the work was interrupted: the retained letter still names answer and copy
        acknowledge_transcript_entry(tmp_path, reader, {"msg_id": f"quiz_answer:{ASKER}:which"})

    first = admit_continuation(reader, action_nonce=NONCE)

    assert first["ok"], first
    [row] = workers.PENDING
    assert "photo.png" in row["text"], "the successor's first request names the answer's file"
    assert ("with an attachment and no words" in row["text"]) is (not comment) and (comment in row["text"])
    answer, = [item for item in row["metadata"]["owner_corpus"] if item.get("quiz_id") == "which"]
    assert answer["content"] == comment and [item["sha256"] for item in answer["attachments"]] == [ref["sha256"]]
    assert answer["source_task_id"] == ASKER and answer["reply_to"]["address"] == f"quiz which of task {ASKER}"
    [staged] = [item for item in row["attachments"] if item.get("status") == "staged"]
    assert staged["sha256"] == ref["sha256"] and staged["label"] == "photo.png", "the successor holds its own copy"
    focus = SimpleNamespace(task_metadata=row["metadata"], drive_root=tmp_path)
    _initialize_owner_directives(focus, [{"role": "user", "content": row["text"]}])
    directive, = [item for item in focus._owner_directives if item.get("quiz_id") == "which"]
    assert directive["content"] == comment and [item["sha256"] for item in directive["attachments"]] == [ref["sha256"]]

    # The successor is itself interrupted: its Continue keeps the same recorded answer, its file and its card.
    assert not workers.WORKERS
    assert workers.kill_workers(terminal_status="cancelled", result_reason="server shutdown", stop_source="server_shutdown")
    second = admit_continuation(first["successor_task_id"], action_nonce=NONCE)

    assert second["ok"], second
    [again] = workers.PENDING
    kept, = [item for item in again["metadata"]["owner_corpus"] if item.get("quiz_id") == "which"]
    assert kept["content"] == comment and [item["sha256"] for item in kept["attachments"]] == [ref["sha256"]]
    assert "photo.png" in again["text"] and ("with an attachment and no words" in again["text"]) is (not comment)


def test_continue_names_a_reply_file_that_never_arrived_beside_the_caption_without_a_copy(tmp_path, monkeypatch):
    from ouroboros import chat_uploads
    from ouroboros.owner_mailbox import KIND_QUIZ_ANSWER, acknowledge_transcript_entry, write_owner_message
    from ouroboros.owner_quiz import recorded_answer_frame
    from supervisor.continuation_admission import admit_continuation

    _, _, workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, task_id=ASKER)
    fact = chat_uploads.delivery_ref({"name": "clip.mp4", "mime": "video/mp4", "size": 30_000_000,
                                      "unavailable": "too_large", "limit_bytes": 10_485_760})
    card = _card([fact], "see the clip")
    assert write_owner_message(tmp_path, recorded_answer_frame(card, asker=ASKER, reader=ASKER), ASKER,
                               kind=KIND_QUIZ_ANSWER, msg_id=f"quiz_answer:{ASKER}:which",
                               quiz_delivery={"task_id": ASKER, "quiz_id": "which", "answer_snapshot": card})
    assert acknowledge_transcript_entry(tmp_path, ASKER, {"msg_id": f"quiz_answer:{ASKER}:which"})  # the work read it

    first = admit_continuation(ASKER, action_nonce=NONCE)

    assert first["ok"], first
    [row] = workers.PENDING
    assert "see the clip" in row["text"] and "clip.mp4: not received (reason=too_large" in row["text"]
    assert not [item for item in row.get("attachments") or [] if item.get("status") == "staged"]


@pytest.mark.parametrize("case", ["unread", "read_then_settled", "unread_then_settled"])
def test_an_answer_whose_record_is_gone_rides_only_on_proof_its_own_work_read_it(tmp_path, monkeypatch, case):
    """The asker's bounded record no longer holds the question (its block evicted after delivery settled), so no
    receipt can say who read the letter. Continue carries it only on this work's own read: after its mailbox was
    settled, its custody keeps the letter as owner mail and never as unread. A letter it never read (still waiting,
    or captured unread while recovery delivered the answer elsewhere) stays out; the other owner mail still rides."""
    from ouroboros.owner_mailbox import KIND_OWNER_TEXT, acknowledge_transcript_entry, write_owner_message
    from ouroboros.task_custody import settle_task_mailbox
    from supervisor.continuation_admission import admit_continuation

    _, _, workers = _install_queue(tmp_path, monkeypatch)
    _answered_letter(tmp_path, comment="this one")
    assert write_owner_message(tmp_path, "Keep the appendix", ASKER, msg_id="owner-2", kind=KIND_OWNER_TEXT)
    if case.startswith("read"):
        assert acknowledge_transcript_entry(tmp_path, ASKER, {"msg_id": f"quiz_answer:{ASKER}:which"})
    _interrupted(tmp_path, task_id=ASKER)
    if case.endswith("settled"):
        assert settle_task_mailbox(tmp_path, ASKER, tmp_path) is True

    first = admit_continuation(ASKER, action_nonce=NONCE)

    assert first["ok"], first
    [row] = workers.PENDING
    later = sorted(item["content"] for item in row["metadata"]["continuation"]["owner_sources"]["later"])
    assert later == (["Keep the appendix", "this one"] if case.startswith("read") else ["Keep the appendix"])


@pytest.mark.parametrize("comment", ["", "this one"], ids=["file-alone", "caption"])
def test_continue_reads_the_answer_from_the_runs_own_retained_corpus_when_its_letter_is_gone(
        tmp_path, monkeypatch, comment):
    """The run recorded the answer (``loop_round_limits._record_quiz_delivery``) with its files; its mailbox letter is
    gone: the retained corpus row alone is still the one answer, words or none."""
    from ouroboros import chat_uploads
    from ouroboros.owner_words import answered_card_reply
    from supervisor.continuation_admission import admit_continuation

    _, _, workers = _install_queue(tmp_path, monkeypatch)
    ref = chat_uploads.store_upload(_PNG, "photo.png", data_dir=tmp_path)[1]
    card = _card([ref], comment)
    row = {"source": "owner_quiz_answer", "content": comment, "msg_id": f"quiz_answer:{ASKER}:which",
           "source_task_id": ASKER, "quiz_id": "which", "reply_to": answered_card_reply(card, ASKER),
           "attachments": [ref]}
    _interrupted(tmp_path, task_id=ASKER, owner_quiz={"which": card},
                 review_evidence={"task_inputs": {"owner_requirements_and_decisions": [row]}})

    first = admit_continuation(ASKER, action_nonce=NONCE)

    assert first["ok"], first
    [successor] = workers.PENDING
    kept, = [item for item in successor["metadata"]["owner_corpus"] if item.get("quiz_id") == "which"]
    assert kept["content"] == comment and [item["sha256"] for item in kept["attachments"]] == [ref["sha256"]]
    assert "photo.png" in successor["text"] and ("with an attachment and no words" in successor["text"]) is (not comment)
    [staged] = [item for item in successor.get("attachments") or [] if item.get("status") == "staged"]
    assert staged["sha256"] == ref["sha256"], "its copy comes from the card's own record of the winning file"


def test_continue_stages_no_copy_from_a_corpus_ref_the_winner_does_not_hold(tmp_path, monkeypatch):
    """A retained corpus row names a file the card's recorded answer does not: the row is no authority for bytes."""
    from ouroboros import chat_uploads
    from ouroboros.owner_words import answered_card_reply
    from supervisor.continuation_admission import admit_continuation

    _, _, workers = _install_queue(tmp_path, monkeypatch)
    recorded = chat_uploads.store_upload(_PNG, "photo.png", data_dir=tmp_path)[1]
    other = chat_uploads.store_upload(b"\x89PNG\r\n\x1a\nother", "other.png", data_dir=tmp_path)[1]
    card = _card([recorded], "this one")
    row = {"source": "owner_quiz_answer", "content": "this one", "msg_id": f"quiz_answer:{ASKER}:which",
           "source_task_id": ASKER, "quiz_id": "which", "reply_to": answered_card_reply(card, ASKER),
           "attachments": [other]}
    _interrupted(tmp_path, task_id=ASKER, owner_quiz={"which": card},
                 review_evidence={"task_inputs": {"owner_requirements_and_decisions": [row]}})

    first = admit_continuation(ASKER, action_nonce=NONCE)

    assert first["ok"], first
    [successor] = workers.PENDING
    assert not [item for item in successor.get("attachments") or [] if item.get("status") == "staged"]


def test_work_order_renders_checked_attachment_facts_once_for_each_start_answer(tmp_path):
    """A drafted start's answered cards render checked attachment facts once beside each answer in ``work_order_text``."""
    from ouroboros import chat_uploads
    from ouroboros.owner_continue import work_order_text

    ref = chat_uploads.store_upload(_PNG, "photo.png", data_dir=tmp_path)[1]
    fact = chat_uploads.delivery_ref({"name": "clip.mp4", "mime": "video/mp4", "size": 30_000_000,
                                      "unavailable": "too_large", "limit_bytes": 10_485_760})
    commented = _card([fact], "see the clip")
    wordless = {**_card([ref]), "quiz_id": "other"}
    later_ref = chat_uploads.store_upload(b"\x89PNG\r\n\x1a\nlater", "later.png", data_dir=tmp_path)[1]
    later_card = {**_card([later_ref], "later note"), "quiz_id": "later-1"}
    order = work_order_text(
        "pred-1", "server_shutdown",
        {"original": None, "schedule": None, "peer_context": [],
         "start": {"source": "drafted_start", "task_id": "pred-1", "objective": "Follow up",
                   "author": {"kind": "task", "task_id": "drafter-1"},
                   "answers": [
                       {"source": "owner_quiz_answer", "content": "see the clip",
                        "msg_id": f"quiz_answer:{ASKER}:which", "source_task_id": ASKER,
                        "quiz_id": "which", "card": commented, "attachments": [fact]},
                       {"source": "owner_quiz_answer", "content": "",
                        "msg_id": f"quiz_answer:{ASKER}:other", "source_task_id": ASKER,
                        "quiz_id": "other", "card": wordless, "attachments": [ref]},
                   ]},
         "later": [
             {"source": "owner_quiz_answer", "content": "later note",
              "msg_id": f"quiz_answer:{ASKER}:later-1", "source_task_id": ASKER,
              "quiz_id": "later-1", "card": later_card, "attachments": [later_ref]},
         ]},
    )

    assert order.count("clip.mp4: not received (reason=too_large") == 1
    assert order.count("photo.png") == 1
    assert order.count("later.png") == 1
    assert "with an attachment and no words" in order


def test_start_answer_reads_evicted_direct_answer_with_proven_identity_and_attachments(tmp_path, monkeypatch):
    """After ``owner_quiz._QUIZ_CAP`` evicts an ordinary card, direct ``start_answer`` reads its canonical
    ``quiz_answer`` history row without borrowing ``delivery``, and renders its file once in ``work_order_text``."""
    from ouroboros import chat_uploads, dialogue_store
    from ouroboros.answer_facts import answer_source
    from ouroboros.gateway.task_decision import _record_quiz_answer_history
    from ouroboros.owner_continue import _drafted_start, work_order_text
    from ouroboros.owner_quiz import (
        _QUIZ_CAP, mark_discharged, record_answered, record_asked, record_closed, record_publication,
    )
    from ouroboros.owner_relay import owner_row_identity
    from ouroboros.owner_source import start_answer
    from ouroboros.owner_words import answered_card_reply
    from ouroboros.question_answers import answer_message_id
    from ouroboros.task_results import load_task_result

    _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, task_id=ASKER)
    ref = chat_uploads.store_upload(_PNG, "photo.png", data_dir=tmp_path)[1]
    record_asked(tmp_path, ASKER, quiz_id="which", question="Which screen?", options=["Amber", "Violet"], chat_id=7)
    record_publication(tmp_path, ASKER, "which", 7)
    card = record_answered(tmp_path, ASKER, quiz_id="which", option_index=None,
                           request_id="tg:9:m:42:7", comment="use this", attachments=[ref])["block"]
    _record_quiz_answer_history(tmp_path, ASKER, None, card, duplicate=False)
    mark_discharged(tmp_path, ASKER, "which")
    for idx in range(_QUIZ_CAP):
        qid = f"q-{idx}"
        record_asked(tmp_path, ASKER, quiz_id=qid, question=f"Q{idx}?", options=["Yes"], chat_id=7)
        record_closed(tmp_path, ASKER, quiz_id=qid, reason="done", actor="owner", request_id=f"close-{idx}")
        mark_discharged(tmp_path, ASKER, qid)
    assert "which" not in load_task_result(tmp_path, ASKER)["owner_quiz"]

    stored = dialogue_store.probe_rows(
        tmp_path, "chat", [("client_message_id", answer_message_id(ASKER, "which"))], limit=2,
    )[0].row
    good_row = {
        "source": answer_source(card), "content": "use this",
        "msg_id": answer_message_id(ASKER, "which"), "source_task_id": ASKER, "quiz_id": "which",
        "reply_to": answered_card_reply(card, ASKER),
        "row_sha256": owner_row_identity(stored)["row_sha256"],
    }
    resolved = start_answer(tmp_path, "pred-1", good_row, "drafter-1")
    assert resolved is not None and [item["sha256"] for item in resolved["attachments"]] == [ref["sha256"]]

    gaps = []
    start = _drafted_start(
        tmp_path, {"objective": "Follow up"}, "pred-1",
        {"objective_author": {"kind": "task", "task_id": "drafter-1"}, "owner_corpus": [good_row]}, gaps,
    )
    assert gaps == [] and start is not None and len(start["answers"]) == 1
    order = work_order_text("pred-1", "server_shutdown",
                            {"original": None, "schedule": None, "start": start, "later": [], "peer_context": []})
    assert order.count("photo.png") == 1
    assert start_answer(tmp_path, "pred-1", {**good_row, "msg_id": "forged-late-id"}, "drafter-1") is None
    assert start_answer(tmp_path, "pred-1", {**good_row, "row_sha256": "f" * 64}, "drafter-1") is None


def test_start_answer_refuses_on_live_card_unreadable_store_or_unproved_history(tmp_path, monkeypatch):
    """Non-evicted live cards, unreadable/invalid projections, duplicate addresses, and unproved rows stay None."""
    from ouroboros import dialogue_store, task_store
    from ouroboros.owner_continue import _drafted_start
    from ouroboros.owner_quiz import record_answered, record_asked, record_closed, recorded_answer_frame
    from ouroboros.owner_source import _recorded_answered_quiz, start_answer
    from ouroboros.question_answers import answer_message_id
    from ouroboros.store import StoreError
    from supervisor.message_bus import log_chat

    _install_queue(tmp_path, monkeypatch)
    if not dialogue_store.present(tmp_path):
        dialogue_store.create(tmp_path)
    _interrupted(tmp_path, task_id=ASKER, owner_quiz={})

    record_asked(tmp_path, ASKER, quiz_id="still-live", question="Which?", options=["A"], chat_id=7)
    record_closed(tmp_path, ASKER, quiz_id="still-live", reason="done", actor="owner", request_id="close-live")
    record_asked(tmp_path, ASKER, quiz_id="live-no-row", question="Which?", options=["A"], chat_id=7)
    live_ans = record_answered(tmp_path, ASKER, quiz_id="live-no-row", option_index=None,
                               request_id="req-live", comment="use live")["block"]
    log_chat(
        "system", 7, 0, recorded_answer_frame(live_ans, asker=ASKER, reader=ASKER),
        ts=live_ans["answered_at"], source="owner_quiz_answer", task_id=ASKER,
        client_message_id=answer_message_id(ASKER, "still-live"), record_type="quiz_answer",
        quiz={**live_ans, "quiz_id": "still-live"}, message_meta={"chat_id": 7}, drive_root=tmp_path, require_write=True,
    )
    assert start_answer(tmp_path, "pred-1", {
        "source": "owner_quiz_answer", "content": "use live",
        "msg_id": answer_message_id(ASKER, "still-live"), "source_task_id": ASKER, "quiz_id": "still-live",
    }, "drafter-1") is None
    assert start_answer(tmp_path, "pred-1", {
        "source": "owner_quiz_answer", "content": "use live",
        "msg_id": answer_message_id(ASKER, "live-no-row"), "source_task_id": ASKER, "quiz_id": "live-no-row",
        "row_sha256": "a" * 64,
    }, "drafter-1") is None

    for qid, ts, src, req in (
        ("dup", "2040-01-01T00:05:01Z", "owner_quiz_answer", "req-1"),
        ("dup", "2040-01-01T00:05:02Z", "owner_quiz_answer", "req-1"),
        ("no-req", "2040-01-01T00:05:03Z", "owner_quiz_answer", ""),
        ("wrong-src", "2040-01-01T00:05:04Z", "model_question_close", "req-2"),
    ):
        card = {**_card([], "evicted answer"), "quiz_id": qid, "request_id": req}
        log_chat(
            "system", 7, 0, f"answer-{qid}", ts=ts, source=src, task_id=ASKER,
            client_message_id=answer_message_id(ASKER, qid), record_type="quiz_answer",
            quiz=card, message_meta={"chat_id": 7}, drive_root=tmp_path, require_write=True,
        )
    for qid in ("dup", "no-req", "wrong-src"):
        row = {"source": "owner_quiz_answer", "content": "evicted answer",
               "msg_id": answer_message_id(ASKER, qid), "source_task_id": ASKER, "quiz_id": qid}
        assert start_answer(tmp_path, "pred-1", row, "drafter-1") is None

    probed = []
    evicted_row = {"source": "owner_quiz_answer", "content": "use this",
                   "msg_id": answer_message_id(ASKER, "evicted-1"), "source_task_id": ASKER, "quiz_id": "evicted-1"}
    def _boom(*_args, **_kwargs):
        raise StoreError("store unavailable")

    with monkeypatch.context() as m:
        m.setattr(task_store, "read_document", lambda *_a, **_k: {"owner_quiz": "invalid"})
        m.setattr(dialogue_store, "probe_rows", lambda *_a, **_k: probed.append(True) or [])
        assert start_answer(tmp_path, "pred-1", evicted_row, "drafter-1") is None and probed == []
        m.setattr(task_store, "read_document", _boom)
        assert start_answer(tmp_path, "pred-1", evicted_row, "drafter-1") is None and probed == []

    monkeypatch.setattr(dialogue_store, "probe_rows", _boom)
    with pytest.raises(StoreError):
        _recorded_answered_quiz(tmp_path, ASKER, "evicted-1")
    gaps = []
    _drafted_start(
        tmp_path, {"objective": "Follow up"}, "pred-1",
        {"objective_author": {"kind": "task", "task_id": "drafter-1"}, "owner_corpus": [evicted_row]},
        gaps,
    )
    assert gaps == ["start_quiz_answer_unresolved"]
