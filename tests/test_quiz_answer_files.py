"""A Telegram Reply answers a question with its file (owner's choice, 18 Nov): the verbatim caption and the file are
ONE recorded answer of that request, staged for the asking work when delivered; a file that never arrived is the
host's fact beside the caption, and a file alone answers without invented words."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import pathlib

import pytest

from ouroboros import chat_uploads
from ouroboros.gateway import task_decision as td
from ouroboros.owner_quiz import (
    late_answer_owner_text,
    quiz_states,
    reconcile_terminal,
    record_answered,
    record_asked,
    record_publication,
    recorded_answer_frame,
)

_PNG = b"\x89PNG\r\n\x1a\n" + b"pixels" * 4


def _upload(root, payload=_PNG, name="photo.png"):
    return chat_uploads.store_upload(payload, name, data_dir=root)[1]


def _ask(root, task="task-1", quiz="q1"):
    record_asked(root, task, quiz_id=quiz, question="Which screen?", options=["A", "B"], assumption="A meanwhile")
    record_publication(root, task, quiz, 1)


def _live(monkeypatch, root, live=True):
    """The asker reads its mail (or is gone); every live card frame is captured."""
    import supervisor.message_bus as mb
    import supervisor.queue as q

    frames = []
    monkeypatch.setattr(mb, "get_bridge", lambda: type("Bridge", (), {
        "send_quiz_state": lambda self, quiz_id, task_id, state, **fields: frames.append({"state": state, **fields})})())
    monkeypatch.setattr(td, "_live_root_task", lambda task_id, **_: (({"id": task_id, "chat_id": 1}, "") if live
                                                                      else (None, "task_not_live")))
    monkeypatch.setattr(q, "_task_drive_for_task", lambda task, task_id: root, raising=False)
    return frames


def _answer(root, body, **kwargs):
    return asyncio.run(td.answer_decision(root, body, source="skill:telegram", **kwargs))


def test_a_file_alone_answers_without_invented_words(tmp_path):
    ref = _upload(tmp_path)
    record_asked(tmp_path, "t1", quiz_id="q1", question="Which?", options=["A"], assumption="A")
    assert record_answered(tmp_path, "t1", quiz_id="q1", option_index=None, request_id="r0")["error"] == "answer_empty"

    block = record_answered(tmp_path, "t1", quiz_id="q1", option_index=None, request_id="r1", attachments=[ref])["block"]

    assert block["attachments"] == [ref] and "comment" not in block and "answered_index" not in block
    frame = recorded_answer_frame(block, asker="t1")
    assert "The owner answered with an attachment and no words, without choosing an offered option." in frame
    assert "Verbatim:" not in frame
    assert f"[Host-retained attachments: photo.png (image/png, {len(_PNG)} bytes)]" in frame
    assert late_answer_owner_text(block) == "", "the owner's row takes the host placeholder, never words"


def test_a_caption_whose_file_never_arrived_keeps_the_words_and_the_hosts_reason(tmp_path):
    fact = chat_uploads.delivery_ref({"name": "clip.mp4", "mime": "video/mp4", "size": 30_000_000,
                                      "unavailable": "too_large", "limit_bytes": 10_485_760})
    record_asked(tmp_path, "t1", quiz_id="q1", question="Which?", options=["A"], assumption="A")

    block = record_answered(tmp_path, "t1", quiz_id="q1", option_index=None, request_id="r1",
                            comment="see the clip", attachments=[fact])["block"]

    frame = recorded_answer_frame(block, asker="t1")
    assert "Verbatim: see the clip\n[Host fact: attachment delivery; not the owner's words]" in frame
    assert ("- clip.mp4: not received (reason=too_large, mime=video/mp4, size=30000000, limit_bytes=10485760, "
            "ordinal=0)") in frame
    assert "Host-retained attachments" not in frame, "no file is named as held"
    assert chat_uploads.staging_specs([fact], tmp_path) == [{"label": "clip.mp4", "mime": "video/mp4", "size": 30_000_000,
                                                            "unavailable": "too_large", "limit_bytes": 10_485_760}]


def test_only_the_host_supplies_a_file_and_only_to_a_quiz_answer(tmp_path, monkeypatch):
    ref = _upload(tmp_path)
    _ask(tmp_path)
    _live(monkeypatch, tmp_path)
    body = {"request_id": "r", "decision_id": "quiz:task-1:q1"}

    assert _answer(tmp_path, {**body, "attachments": [ref]})[1]["reason_code"] == "unexpected_fields"
    for refused in ({"request_id": "r", "decision_id": "routing:m:t", "option_index": 0}, {**body, "action": "close"}):
        status, payload = _answer(tmp_path, refused, attachments=[ref])
        assert (status, payload["reason_code"]) == (400, "attachments_invalid")
    assert quiz_states(tmp_path, "task-1")["q1"]["state"] == "open"


def test_the_asking_work_reads_the_caption_and_its_file_staged_in_its_own_store(tmp_path, monkeypatch):
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.owner_mailbox import KIND_QUIZ_ANSWER, drain_owner_entries

    ref = _upload(tmp_path)
    _ask(tmp_path)
    frames = _live(monkeypatch, tmp_path)

    status, payload = _answer(tmp_path, {"request_id": "tg:9:m:42:7", "decision_id": "quiz:task-1:q1",
                                         "comment": "this one"}, attachments=[ref])

    assert status == 200 and payload["comment"] == "this one"
    shown = [{key: ref[key] for key in ("name", "kind", "mime", "size", "sha256")}]
    assert payload["attachments"] == shown, "a receipt names the file, never its stored id or witness"
    assert frames[-1]["attachments"] == shown and frames[-1]["comment"] == "this one"
    entry, = [e for e in drain_owner_entries(tmp_path, "task-1", set()) if e.get("kind") == KIND_QUIZ_ANSWER]
    assert "Verbatim: this one" in entry["text"] and "[Host-retained attachments: photo.png" in entry["text"]
    assert "\n\n[ATTACHMENTS]\n" in entry["text"] and "_photo.png [status=staged, ordinal=0]" in entry["text"]
    staged, = [row for row in entry["attachment_manifest"] if row.get("status") == "staged"]
    assert staged["sha256"] == ref["sha256"]
    assert (task_artifact_dir_path(tmp_path, "task-1") / staged["relpath"]).read_bytes() == _PNG


def test_the_living_asker_sees_a_reply_image_as_owner_mail_shows_one(tmp_path, monkeypatch):
    import queue as queue_mod
    from types import SimpleNamespace

    from ouroboros.loop_round_limits import _drain_incoming_messages

    _ask(tmp_path)
    _live(monkeypatch, tmp_path)
    pixel = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")
    assert _answer(tmp_path, {"request_id": "r", "decision_id": "quiz:task-1:q1", "comment": "this one"},
                   attachments=[_upload(tmp_path, pixel)])[0] == 200
    messages = [{"role": "user", "content": "Initial requirement"}]

    _drain_incoming_messages(messages, queue_mod.Queue(), tmp_path, "task-1", None, set(), owner_ctx=SimpleNamespace())

    content, = [m["content"] for m in messages[1:] if isinstance(m["content"], list)]
    assert "Verbatim: this one" in content[0]["text"]
    assert [part["type"] for part in content].count("image_url") == 1, "its pixels, not only a path"


def test_a_replay_or_another_request_never_changes_the_recorded_file(tmp_path, monkeypatch):
    from ouroboros.owner_mailbox import KIND_QUIZ_ANSWER, drain_owner_entries

    first, again = _upload(tmp_path), _upload(tmp_path)  # the same bytes, received twice
    _ask(tmp_path)
    _live(monkeypatch, tmp_path)
    body = {"request_id": "tg:9:m:42:7", "decision_id": "quiz:task-1:q1"}
    assert _answer(tmp_path, body, attachments=[first])[0] == 200
    assert td.answer_holds_uploads(tmp_path, body, [first]) is True

    status, payload = _answer(tmp_path, body, attachments=[again])  # Telegram delivered the same update again
    assert status == 200 and payload["duplicate"] is True
    assert quiz_states(tmp_path, "task-1")["q1"]["attachments"] == [first]
    assert td.answer_holds_uploads(tmp_path, body, [again]) is False, "the replay's own copy is not the winner's"

    other = {"request_id": "tg:9:m:42:8", "decision_id": "quiz:task-1:q1"}
    status, payload = _answer(tmp_path, other, attachments=[again])
    assert status == 409 and [item["sha256"] for item in payload["attachments"]] == [first["sha256"]]
    assert td.answer_holds_uploads(tmp_path, other, [again]) is False, "the same bytes from another request still lost"
    assert len([e for e in drain_owner_entries(tmp_path, "task-1", set()) if e.get("kind") == KIND_QUIZ_ANSWER]) == 1


def test_a_late_file_alone_becomes_the_owners_message_carrying_that_file(tmp_path, monkeypatch):
    ref = _upload(tmp_path)
    _ask(tmp_path)
    reconcile_terminal(tmp_path, "task-1")
    _live(monkeypatch, tmp_path, live=False)
    sent = []
    monkeypatch.setattr(td, "deliver_as_owner_message",
                        lambda root, chat_id, text, message_id, **kw: sent.append((chat_id, text, message_id, kw)) or (True, ""))

    status, payload = _answer(tmp_path, {"request_id": "tg:9:m:42:7", "decision_id": "quiz:task-1:q1"},
                              attachments=[ref])

    assert status == 200 and payload["forwarded"] is True and payload["answered_after_terminal"] is True
    (chat_id, text, message_id, kw), = sent
    assert (chat_id, text, message_id) == (1, "", "quiz_late_answer:task-1:q1"), "no words are invented for it"
    metadata = kw["task_metadata"]
    assert metadata["chat_attachments"] == [ref]
    spec, = metadata["chat_attachment_uploads"]
    assert spec["path"] == str(pathlib.Path(tmp_path) / "uploads" / ref["upload"])
    assert (spec["sha256"], spec["size"], spec["label"]) == (ref["sha256"], ref["size"], "photo.png")
    assert metadata["late_answer"]["answer_snapshot"]["quiz"]["attachments"] == [ref]


def test_a_rejoin_reads_only_this_requests_record_and_keeps_unknown_copies(tmp_path, monkeypatch):
    ref = _upload(tmp_path)
    _ask(tmp_path)
    _live(monkeypatch, tmp_path)
    body = {"request_id": "tg:9:m:42:7", "decision_id": "quiz:task-1:q1"}
    assert td.rejoin_decision(tmp_path, {**body, "rejoin": True})[1]["reason_code"] == "request_not_recorded"
    assert _answer(tmp_path, {**body, "comment": "this one"}, attachments=[ref])[0] == 200

    status, payload = td.rejoin_decision(tmp_path, {**body, "rejoin": True})
    assert status == 200 and payload["duplicate"] is True and payload["comment"] == "this one"
    assert payload["attachments"][0]["sha256"] == ref["sha256"] and "upload" not in payload["attachments"][0]
    status, payload = td.rejoin_decision(tmp_path, {"request_id": "tg:9:m:42:8", "decision_id": "quiz:task-1:q1",
                                                    "rejoin": True})
    assert (status, payload["state"]) == (404, "answered"), "equal words or bytes from another request are not it"
    assert td.rejoin_decision(tmp_path, {**body, "rejoin": True, "comment": "x"})[0] == 400

    def unreadable(*_args, **_kwargs):
        raise OSError("store busy")

    monkeypatch.setattr("ouroboros.task_store.read_document", unreadable)
    assert td.rejoin_decision(tmp_path, {**body, "rejoin": True})[0] == 503
    assert td.answer_holds_uploads(tmp_path, body, [_upload(tmp_path)]) is True, "an unknown record keeps a copy"


def test_the_files_are_copied_before_the_readers_admission_lock(tmp_path, monkeypatch):
    import threading

    import supervisor.queue as q
    from ouroboros import question_answers

    class Lock:
        def __init__(self):
            self.inner, self.held = threading.RLock(), 0

        def __enter__(self):
            self.inner.acquire()
            self.held += 1
            return self

        def __exit__(self, *_exc):
            self.held -= 1
            self.inner.release()

    ref = _upload(tmp_path)
    _ask(tmp_path)
    _live(monkeypatch, tmp_path)
    lock, copying = Lock(), []
    monkeypatch.setattr(q, "_queue_lock", lock)
    stage = question_answers._staged_answer_files
    monkeypatch.setattr(question_answers, "_staged_answer_files", lambda *args: copying.append(lock.held) or stage(*args))

    assert _answer(tmp_path, {"request_id": "r", "decision_id": "quiz:task-1:q1"}, attachments=[ref])[0] == 200
    assert copying == [0], "file I/O never holds the queue's admission lock"


def test_an_attempt_whose_reader_ended_keeps_the_copy_another_attempts_letter_names(tmp_path, monkeypatch):
    """Two deliveries of one answer stage the same bytes under one name in the reader's store. The one that wrote the
    letter reused the other's copy; the other, finding the reader gone at its admission, must not remove it."""
    from ouroboros import question_answers
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.owner_mailbox import KIND_QUIZ_ANSWER, drain_owner_entries

    ref = _upload(tmp_path)
    _ask(tmp_path)
    _live(monkeypatch, tmp_path)
    block = record_answered(tmp_path, "task-1", quiz_id="q1", option_index=None, request_id="r1", attachments=[ref])["block"]
    reads = []

    def reader(task_id):  # live for the first attempt's choice and the whole second attempt, then ended
        reads.append(task_id)
        return ({"id": task_id, "chat_id": 1}, "") if len(reads) <= 3 else (None, "task_not_live")

    stage, raced, owned = question_answers._staged_answer_files, [], set()

    def staged_then_raced(*args):
        copies = stage(*args)
        if not raced:  # simulate a distinct persisted delivery landing between this attempt's copy and admission
            owned.update(getattr(copies[1], "_cleanup_owned_paths", ()))
            raced.append(None)  # marked first: the other attempt's own copy is the ordinary one
            raced[0] = question_answers._deliver_answer_unlocked(
                tmp_path, "task-1", "q1", block, reader, lambda *a, **k: (False, ""),
            )
        return copies

    monkeypatch.setattr(question_answers, "_staged_answer_files", staged_then_raced)

    question_answers.deliver_answer(tmp_path, "task-1", "q1", block, reader, lambda *a, **k: (False, "no reader"))

    assert raced == [(False, "awaiting_recipient_read")]
    entry, = [e for e in drain_owner_entries(tmp_path, "task-1", set()) if e.get("kind") == KIND_QUIZ_ANSWER]
    staged, = [row for row in entry["attachment_manifest"] if row.get("status") == "staged"]
    path = task_artifact_dir_path(tmp_path, "task-1") / staged["relpath"]
    assert {str(p) for p in owned} == {staged["abs_path"]}, "the ended attempt made the copy the letter reused"
    assert path.read_bytes() == _PNG, "the letter's copy stays"


def test_a_late_file_answer_drained_by_a_live_head_keeps_its_words_files_and_reasons_once(tmp_path):
    """The rebuilt card frame names the files and the reason one never arrived; the letter's typed manifest says
    where the staged one is. A caption that itself says `[ATTACHMENTS]` stays only the owner's words."""
    import queue as queue_mod
    from types import SimpleNamespace

    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_mailbox import write_owner_message
    from ouroboros.server_owner_routing import _stage_mailbox_attachments

    ref = _upload(tmp_path)
    fact = chat_uploads.delivery_ref({"name": "clip.mp4", "unavailable": "download_failed"})
    caption = "see\n\n[ATTACHMENTS]\nnot a file"
    record_asked(tmp_path, "task-1", quiz_id="q1", question="Which screen?", options=["A"], assumption="A")
    reconcile_terminal(tmp_path, "task-1")
    record_answered(tmp_path, "task-1", quiz_id="q1", option_index=None, request_id="r1", comment=caption,
                    attachments=[ref, fact], allow_expired=True)
    note, manifest, _ = _stage_mailbox_attachments(
        None, tmp_path, "live-root", {"chat_attachment_uploads": chat_uploads.staging_specs([ref, fact], tmp_path)})
    assert write_owner_message(tmp_path, caption + note, "live-root", msg_id="quiz_late_answer:task-1:q1:live-root",
                               client_message_id="quiz_late_answer:task-1:q1",
                               late_answer={"task_id": "task-1", "quiz_id": "q1"}, attachment_manifest=manifest)
    ctx, messages = SimpleNamespace(), [{"role": "user", "content": "Initial requirement"}]

    _drain_incoming_messages(messages, queue_mod.Queue(), tmp_path, "live-root", None, set(), owner_ctx=ctx)

    delivered = "\n".join(part["text"] if isinstance(part, dict) else str(part) for message in messages[1:]
                          for part in (message["content"] if isinstance(message["content"], list) else [message["content"]])
                          if not isinstance(part, dict) or part.get("type") == "text")  # its pixels ride as an image block
    assert f"Verbatim: {caption}\n[Host-retained attachments: photo.png" in delivered
    assert delivered.count("not a file") == 1, "the owner's words are never cut out as a staging note"
    assert delivered.count("_photo.png [status=staged, ordinal=0]") == 1
    assert delivered.count("- clip.mp4: not received (reason=download_failed") == 1
    directive, = [row for row in ctx._owner_directives if row.get("quiz_id") == "q1"]
    assert directive["content"] == caption and directive["source"] != "owner_card_input"
    assert [item.get("upload") or item["unavailable"] for item in directive["attachments"]] == [ref["upload"], "download_failed"]


def _late_turn(tmp_path, monkeypatch, refs, *, comment="", drop=None):
    """A late answer's turn as its carrier takes it: the accepted answer, its recorded files as staging specs, the
    first step run. ``drop(upload)`` changes the store between the answer's acceptance and that step."""
    from types import SimpleNamespace

    import supervisor.workers as workers
    from supervisor import worker_chat_lane

    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    record_asked(tmp_path, "task-1", quiz_id="q1", question="Which screen?", options=["A"], assumption="A")
    reconcile_terminal(tmp_path, "task-1")
    block = record_answered(tmp_path, "task-1", quiz_id="q1", option_index=None, request_id="r1", comment=comment,
                            attachments=refs, allow_expired=True)["block"]
    for ref in refs:
        if drop is not None and ref.get("upload"):
            drop(chat_uploads.uploads_dir(tmp_path) / ref["upload"])
    metadata = {"late_answer": {"task_id": "task-1", "quiz_id": "q1", "answer_snapshot": {"task_id": "task-1", "quiz": block}},
                "chat_attachments": refs, "chat_attachment_uploads": chat_uploads.staging_specs(refs, tmp_path)}
    task = worker_chat_lane._seed_task(1, late_answer_owner_text(block), None, None, metadata)
    worker_chat_lane.prepare_conversation_task(SimpleNamespace(), task, tmp_path)
    return task


def _rejected_rows(tmp_path):
    rows = [json.loads(line) for line in (tmp_path / "logs" / "supervisor.jsonl").read_text(encoding="utf-8").splitlines()]
    return [row for row in rows if row.get("type") == "attachments_rejected"]


@pytest.mark.parametrize("change,reason", [
    (lambda path: path.unlink(), "source_missing"),
    (lambda path: path.write_bytes(b"other bytes entirely"), "copy_failed"),
], ids=["gone", "changed"])
def test_a_late_answer_whose_stored_file_cannot_be_staged_still_arrives_with_its_words_and_the_reason(
        tmp_path, monkeypatch, change, reason):
    """The canonical bytes are checked against the recorded size and digest while they are copied: a file gone or
    changed since the answer was accepted is refused, never replaced, and the turn that carries the recorded
    answer still starts, so the asking work's answer is read beside why the file is not there (never lost with it)."""
    ref = _upload(tmp_path)

    task = _late_turn(tmp_path, monkeypatch, [ref], comment="this one, as in the photo", drop=change)

    assert task["text"].startswith("[Late answer to a question asked by task task-1, which had finished]")
    assert "Verbatim: this one, as in the photo" in task["text"]
    assert f"- photo.png: rejected (reason={reason}, ordinal=0)" in task["text"]
    assert task["text"].count("[ATTACHMENTS]") == 1
    assert "attachment_images" in task and task["attachment_images"] == []
    (diagnostic,) = _rejected_rows(tmp_path)
    assert diagnostic["started"] is True and [(r["label"], r["reason"]) for r in diagnostic["rejected"]] == [("photo.png", reason)]
    assert not (tmp_path / "task_results" / "artifacts" / task["id"] / "attachments").exists(), "no copy is kept"


def test_a_late_file_alone_that_cannot_be_staged_still_reaches_its_work_with_the_reason(tmp_path, monkeypatch):
    task = _late_turn(tmp_path, monkeypatch, [_upload(tmp_path)], drop=lambda path: path.unlink())

    assert "The owner answered with an attachment and no words, without choosing an offered option." in task["text"]
    assert "- photo.png: rejected (reason=source_missing, ordinal=0)" in task["text"]
    assert "Verbatim:" not in task["text"]


# An ordinary message whose every file is refused still starts nothing: tests/test_attachment_staging.py
# (``test_initial_ui_turn_is_atomic_on_staging_rejection``) witnesses that unchanged neighbour.


def test_a_late_answer_with_one_file_staged_and_one_gone_names_each_exactly_once(tmp_path, monkeypatch):
    kept, lost = _upload(tmp_path), _upload(tmp_path, b"\x89PNG\r\n\x1a\nsecond", "second.png")

    task = _late_turn(tmp_path, monkeypatch, [kept, lost], comment="both",
                      drop=lambda path: path.unlink() if path.name.endswith("second.png") else None)

    assert task["text"].count("_photo.png [status=staged, ordinal=0]") == 1
    assert task["text"].count("- second.png: rejected (reason=source_missing, ordinal=1)") == 1
    from ouroboros.artifacts import task_artifact_dir_path

    (staged,) = [row for row in task["attachments"] if row["status"] == "staged"]
    assert staged["sha256"] == kept["sha256"], "the file the work holds is the recorded one, byte for byte"
    assert (task_artifact_dir_path(tmp_path, task["id"]) / staged["relpath"]).read_bytes() == _PNG
    (diagnostic,) = _rejected_rows(tmp_path)
    assert diagnostic["started"] is True


def test_the_living_asker_reads_its_answer_and_why_a_file_is_gone_from_the_store(tmp_path, monkeypatch):
    """The same transfer at the reader that is still alive: the file's copy fails while the answer is being delivered,
    and the letter carries the caption and the reason, the answer recorded and delivered once."""
    from ouroboros.owner_mailbox import KIND_QUIZ_ANSWER, drain_owner_entries

    ref = _upload(tmp_path)
    _ask(tmp_path)
    _live(monkeypatch, tmp_path)
    (chat_uploads.uploads_dir(tmp_path) / ref["upload"]).unlink()  # gone after the answer was accepted

    status, payload = _answer(tmp_path, {"request_id": "tg:9:m:42:7", "decision_id": "quiz:task-1:q1", "comment": "this one"},
                              attachments=[ref])

    assert status == 200 and payload["comment"] == "this one" and payload["state"] == "answered"
    entry, = [e for e in drain_owner_entries(tmp_path, "task-1", set()) if e.get("kind") == KIND_QUIZ_ANSWER]
    assert "Verbatim: this one" in entry["text"]
    assert entry["text"].count("- photo.png: rejected (reason=source_missing, ordinal=0)") == 1
    assert [(row["status"], row["reason"]) for row in entry["attachment_manifest"]] == [("rejected", "source_missing")]


def test_a_live_late_head_drains_the_caption_and_the_reason_when_the_file_is_gone(tmp_path):
    import queue as queue_mod
    from types import SimpleNamespace

    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_mailbox import write_owner_message
    from ouroboros.server_owner_routing import _stage_mailbox_attachments

    ref = _upload(tmp_path)
    record_asked(tmp_path, "task-1", quiz_id="q1", question="Which screen?", options=["A"], assumption="A")
    reconcile_terminal(tmp_path, "task-1")
    record_answered(tmp_path, "task-1", quiz_id="q1", option_index=None, request_id="r1", comment="the photo one",
                    attachments=[ref], allow_expired=True)
    (chat_uploads.uploads_dir(tmp_path) / ref["upload"]).unlink()
    note, manifest, _ = _stage_mailbox_attachments(
        None, tmp_path, "live-root", {"chat_attachment_uploads": chat_uploads.staging_specs([ref], tmp_path)})
    assert write_owner_message(tmp_path, "the photo one" + note, "live-root", msg_id="quiz_late_answer:task-1:q1:live-root",
                               client_message_id="quiz_late_answer:task-1:q1",
                               late_answer={"task_id": "task-1", "quiz_id": "q1"}, attachment_manifest=manifest)
    ctx, messages = SimpleNamespace(), [{"role": "user", "content": "Initial requirement"}]

    _drain_incoming_messages(messages, queue_mod.Queue(), tmp_path, "live-root", None, set(), owner_ctx=ctx)

    delivered = "\n".join(part["text"] if isinstance(part, dict) else str(part) for message in messages[1:]
                          for part in (message["content"] if isinstance(message["content"], list) else [message["content"]])
                          if not isinstance(part, dict) or part.get("type") == "text")
    assert "Verbatim: the photo one" in delivered
    assert delivered.count("- photo.png: rejected (reason=source_missing, ordinal=0)") == 1
    directive, = [row for row in ctx._owner_directives if row.get("quiz_id") == "q1"]
    assert directive["content"] == "the photo one" and [item["upload"] for item in directive["attachments"]] == [ref["upload"]]


def test_a_late_answer_whose_file_is_gone_and_another_never_arrived_states_each_reason_once(tmp_path, monkeypatch):
    """The transport's own fact (a download that failed) and the store's (a file gone since) are different reasons, each
    stated once: the first as the host's delivery fact in the card frame, the second in the files' own block."""
    fact = chat_uploads.delivery_ref({"name": "clip.mp4", "unavailable": "download_failed"})

    task = _late_turn(tmp_path, monkeypatch, [_upload(tmp_path), fact], comment="both", drop=lambda path: path.unlink())

    assert task["text"].count("- photo.png: rejected (reason=source_missing, ordinal=0)") == 1
    assert task["text"].count("- clip.mp4: not received (reason=download_failed, mime=application/octet-stream, ordinal=1)") == 1
    assert task["text"].count("[ATTACHMENTS]") == 2, "the host's delivery fact and the staging block, one each"
    from types import SimpleNamespace

    from ouroboros.context import _first_user_content

    first = _first_user_content(task, SimpleNamespace(drive_root=tmp_path))  # the model's whole first input
    text = "\n".join(part["text"] for part in first if part.get("type") == "text") if isinstance(first, list) else first
    assert text.count("- clip.mp4: not received (reason=download_failed") == 1, "one renderer names it"


def test_a_file_alone_is_an_answer_only_with_evidence_and_never_the_last_own_words(tmp_path):
    from ouroboros.answer_facts import last_own_words
    from ouroboros.owner_continue import valid_start
    from ouroboros.owner_relay import row_attachments
    from ouroboros.owner_source import answered_quiz_block
    from tests.test_owner_relay import context

    ref = _upload(tmp_path)
    record_asked(tmp_path, "t1", quiz_id="q1", question="Which?", options=["A"], assumption="A")
    record_answered(tmp_path, "t1", quiz_id="q1", option_index=None, request_id="r1", attachments=[ref])
    assert answered_quiz_block(tmp_path, "t1", "q1")["attachments"] == [ref], "a file alone is the recorded answer"
    record_asked(tmp_path, "t1", quiz_id="q2", question="Which?", options=["A"], assumption="A")
    record_answered(tmp_path, "t1", quiz_id="q2", option_index=None, request_id="r2", attachments=[{"name": "x"}])
    assert answered_quiz_block(tmp_path, "t1", "q2") == {}, "an entry with no upload and no reason evidences nothing"

    answer = {"source": "owner_quiz_answer", "content": "", "msg_id": "quiz_answer:t1:q1", "source_task_id": "t1",
              "quiz_id": "q1", "card": {"quiz_id": "q1"}, "attachments": [ref]}
    start = {"source": "drafted_start", "task_id": "t2", "objective": "Use it", "author": {"kind": "task"}}
    assert valid_start({**start, "answers": [answer]})
    assert not valid_start({**start, "answers": [{**answer, "attachments": [{}]}]})
    assert not valid_start({**start, "answers": [{**answer, "attachments": []}]})

    row = {"type": "quiz_answer", "direction": "system", "source": "owner_quiz_answer", "task_id": "t1",
           "quiz": answered_quiz_block(tmp_path, "t1", "q1"), "chat_id": 1, "ts": "2040-01-01T00:05:00Z"}
    assert [item["upload"] for item in row_attachments(row)] == [ref["upload"]], "the canonical answer row's files"
    assert row_attachments({**row, "source": "skill:telegram"}) == [], "only the host's own answer row"

    ctx = context(tmp_path)
    words = {"source": "direct_incoming", "content": "Keep the blue one", "client_message_id": "said",
             "chat_id": "1", "ts": "2040-01-01T01:00:00Z"}
    ctx._owner_directives = [words, {**words, "content": "", "client_message_id": "file", "attachments": [ref],
                                     "ts": "2040-01-01T02:00:00Z"}]
    assert "said" in last_own_words(ctx) and "file" not in last_own_words(ctx)


def test_the_tasks_own_retention_keeps_a_card_answers_staged_file_and_no_other_reader_inherits_it(tmp_path, monkeypatch):
    from ouroboros.owner_mailbox import owner_attachment_manifest

    _ask(tmp_path)
    _live(monkeypatch, tmp_path)
    assert _answer(tmp_path, {"request_id": "r", "decision_id": "quiz:task-1:q1"}, attachments=[_upload(tmp_path)])[0] == 200

    kept = owner_attachment_manifest(tmp_path, "task-1", answers=True)

    assert [row["sha256"] for row in kept if row.get("status") == "staged"] == [hashlib.sha256(_PNG).hexdigest()]
    assert owner_attachment_manifest(tmp_path, "task-1") == [], "a child's or delegate's inheritance is unchanged"


def test_an_attempt_whose_reader_read_another_attempts_letter_and_ended_delivers_nothing_again(tmp_path, monkeypatch):
    """Only a read settles an earlier effect: the stale attempt finds the letter read and routes no second copy."""
    from ouroboros import question_answers
    from ouroboros.owner_mailbox import KIND_QUIZ_ANSWER, drain_owner_entries

    _ask(tmp_path)
    _live(monkeypatch, tmp_path)
    block = record_answered(tmp_path, "task-1", quiz_id="q1", option_index=None, request_id="r1",
                            attachments=[_upload(tmp_path)])["block"]
    reads, raced, routed = [], [], []

    def reader(task_id):
        reads.append(task_id)
        return ({"id": task_id, "chat_id": 1}, "") if len(reads) <= 3 else (None, "task_not_live")

    stage = question_answers._staged_answer_files

    def staged_then_raced(*args):
        copies = stage(*args)
        if not raced:  # simulate a distinct persisted delivery and reader ACK landing before this attempt's admission
            raced.append(None)  # marked first: the other attempt's own copy is the ordinary one
            raced[0] = question_answers._deliver_answer_unlocked(
                tmp_path, "task-1", "q1", block, reader, lambda *a, **k: (False, ""),
            )
            entry, = [e for e in drain_owner_entries(tmp_path, "task-1", set()) if e.get("kind") == KIND_QUIZ_ANSWER]
            question_answers.acknowledge_answer(tmp_path, "task-1", entry)
        return copies

    monkeypatch.setattr(question_answers, "_staged_answer_files", staged_then_raced)

    outcome = question_answers.deliver_answer(tmp_path, "task-1", "q1", block, reader,
                                              lambda *a, **k: routed.append(a) or (True, ""))

    assert routed == [] and outcome[0] is False, "no late head or fresh head receives it a second time"
    assert quiz_states(tmp_path, "task-1")["q1"]["delivery"]["read_at"]


@pytest.mark.parametrize("ledger,files", [("read", True), ("unknown", True), ("unread", True), ("read", False)],
                         ids=["read", "unknown", "unread", "read-no-file"])
def test_an_attempt_that_lost_its_admission_resolves_the_letter_another_attempt_saved(tmp_path, monkeypatch,
                                                                                     ledger, files):
    """While this attempt copied the file off the lock, another delivered the letter and the reader ended. That
    reader's own read ledger decides before any Continue or the late path: a read settles it, nothing sent again (its
    projection receipt never landed); an unreadable ledger keeps the answer owed, sent nowhere else; only a letter its
    ended reader never read takes the existing late path, once. The same holds with no file to copy."""
    from ouroboros import question_answers
    from ouroboros.owner_mailbox import KIND_QUIZ_ANSWER, _ack_path, acknowledge_task_messages, drain_owner_entries

    _ask(tmp_path)
    _live(monkeypatch, tmp_path)
    block = record_answered(tmp_path, "task-1", quiz_id="q1", option_index=None, request_id="r1", comment="this one",
                            attachments=[_upload(tmp_path)] if files else None)["block"]
    reads, raced, routed = [], [], []

    def reader(task_id):  # live for the first attempt's choice and the whole second attempt, then ended
        reads.append(task_id)
        return ({"id": task_id, "chat_id": 1}, "") if len(reads) <= 3 else (None, "task_not_live")

    stage = question_answers._staged_answer_files

    def staged_then_raced(*args):
        copies = stage(*args)
        if not raced:  # simulate a distinct persisted delivery landing before this attempt's admission
            raced.append(None)  # marked first: the other attempt's own copy is the ordinary one
            raced[0] = question_answers._deliver_answer_unlocked(
                tmp_path, "task-1", "q1", block, reader, lambda *a, **k: (False, ""),
            )
            entry, = [e for e in drain_owner_entries(tmp_path, "task-1", set()) if e.get("kind") == KIND_QUIZ_ANSWER]
            if ledger == "read":  # the reader's own durable read; acknowledge_answer's receipt never landed
                assert acknowledge_task_messages(tmp_path, "task-1", [entry["msg_id"]], wake_id="loop_delivery")
            elif ledger == "unknown":  # the reader's read ledger can no longer be read whole
                with _ack_path(tmp_path, "task-1").open("a", encoding="utf-8") as ledger_file:
                    ledger_file.write("{torn\n")
        return copies

    monkeypatch.setattr(question_answers, "_staged_answer_files", staged_then_raced)

    outcome = question_answers.deliver_answer(tmp_path, "task-1", "q1", block, reader,
                                              lambda *a, **k: routed.append(a) or (True, ""))

    delivery = quiz_states(tmp_path, "task-1")["q1"]["delivery"]
    letters = [e for e in drain_owner_entries(tmp_path, "task-1", set(), include_acknowledged=True)
               if e.get("kind") == KIND_QUIZ_ANSWER]
    assert raced == [(False, "awaiting_recipient_read")] and len(letters) == 1, "one letter, written once"
    if ledger == "read":
        assert routed == [] and outcome == (False, "quiz_history_write_pending")
        assert delivery["read_at"] and delivery["recipient_task_id"] == "task-1"
    elif ledger == "unknown":
        assert routed == [] and outcome == (False, "mailbox_write_failed")
        assert (delivery["state"], delivery["msg_id"]) == ("owed", "quiz_answer:task-1:q1") and not delivery.get("read_at")
    else:
        assert len(routed) == 1 and outcome == (True, ""), "its ended reader never read it: the late path, once"


def test_an_attempt_that_lost_its_admission_converges_on_a_late_answer_accepted_meanwhile(tmp_path, monkeypatch):
    """While this attempt copied the file, its reader ended, that reader's Continue was admitted, and another attempt's
    late answer was accepted by the owner's ingress: that stays the answer's one effect, never a letter to the Continue."""
    import supervisor.message_bus as mb
    from ouroboros import question_answers
    from ouroboros.owner_mailbox import drain_owner_entries

    _ask(tmp_path)
    _live(monkeypatch, tmp_path)
    block = record_answered(tmp_path, "task-1", quiz_id="q1", option_index=None, request_id="r1",
                            attachments=[_upload(tmp_path)])["block"]
    accepted, routed = [], []
    monkeypatch.setattr(question_answers, "admitted_successor", lambda root, task: "task-2" if task == "task-1" else "")
    monkeypatch.setattr(mb, "accepted_chat_message", lambda root, chat, cid: {"client_message_id": cid} if accepted else None)

    def reader(task_id):
        return ({"id": task_id, "chat_id": 1}, "") if task_id == "task-2" or not accepted else (None, "task_not_live")

    stage = question_answers._staged_answer_files

    def staged_then_accepted(*args):
        if not accepted:  # meanwhile: the reader ended, its Continue started, another attempt's late answer accepted
            accepted.append(True)
            question_answers._update_delivery(tmp_path, "task-1", "q1", {
                "recipient_task_id": "", "msg_id": "quiz_late_answer:task-1:q1", "route": "fresh_head",
                "state": "owed", "reason_code": "late_answer_delivery_unknown"})
        return stage(*args)

    monkeypatch.setattr(question_answers, "_staged_answer_files", staged_then_accepted)

    outcome = question_answers.deliver_answer(tmp_path, "task-1", "q1", block, reader,
                                              lambda *a, **k: routed.append(a) or (True, ""))

    assert outcome == (True, "") and len(routed) == 1, "it converges on the accepted message, which its ingress rejoins"
    assert drain_owner_entries(tmp_path, "task-2", set()) == [], "no letter to the Continue admitted since"


def test_a_rejoin_after_the_projection_let_the_question_go_keeps_its_closure_or_its_other_winner(tmp_path, monkeypatch):
    """A settled question can leave the bounded projection; its canonical history row still says what became of it. A
    closure stays closed with its reason, and another channel's winner comes with the fields a losing answer learns,
    so the bridge settles its card with nothing received or sent again. The delivery that row froze when it was
    written is not today's and is never shown as such."""
    from ouroboros.owner_mailbox import KIND_QUIZ_ANSWER, acknowledge_transcript_entry, drain_owner_entries
    from ouroboros.owner_quiz import _mutate_projection

    _ask(tmp_path)
    _ask(tmp_path, quiz="q2")
    _live(monkeypatch, tmp_path)
    assert asyncio.run(td.answer_decision(tmp_path, {"request_id": "web-1", "decision_id": "quiz:task-1:q1",
                                                     "option_index": 1, "comment": "B, as discussed"}))[0] == 200
    assert asyncio.run(td.answer_decision(tmp_path, {"request_id": "close-1", "decision_id": "quiz:task-1:q2",
                                                     "action": "close", "reason": "No longer needed."}))[0] == 200
    entry, = [e for e in drain_owner_entries(tmp_path, "task-1", set()) if e.get("kind") == KIND_QUIZ_ANSWER]
    assert acknowledge_transcript_entry(tmp_path, "task-1", entry)  # the asker read the answer
    assert [quiz_states(tmp_path, "task-1")[q]["delivery"]["state"] for q in ("q1", "q2")] == ["delivered"] * 2
    _mutate_projection(tmp_path, "task-1", lambda quizzes: [quizzes.pop(q) for q in ("q1", "q2")])  # as the cap does

    def rejoin(request_id, quiz):
        return td.rejoin_decision(tmp_path, {"request_id": request_id, "decision_id": f"quiz:task-1:{quiz}",
                                             "rejoin": True})

    status, closed = rejoin("tg:9:m:42:8", "q2")
    assert (status, closed["reason_code"], closed["state"]) == (404, "request_not_recorded", "closed")
    assert (closed["closed_reason"], closed["closed_actor"], closed["closed_request_id"]) == (
        "No longer needed.", "owner", "close-1")
    status, other = rejoin("tg:9:m:42:8", "q1")
    assert (status, other["state"], other["answered_index"], other["comment"]) == (404, "answered", 1, "B, as discussed")
    status, own = rejoin("web-1", "q1")
    assert (status, own["duplicate"], own["comment"]) == (200, True, "B, as discussed")
    assert not any("question_delivery" in payload for payload in (closed, other, own)), "the row's frozen 'owed'"
    assert quiz_states(tmp_path, "task-1") == {}, "a rejoin writes nothing back"


@pytest.mark.parametrize("case", ["read-then-cleaned", "unread-then-cleaned", "read-as-it-ended", "unread-held"])
def test_an_ended_readers_letter_moves_on_only_when_proven_unread(tmp_path, monkeypatch, case):
    """Settlement routinely removes an ended work's mailbox with its acknowledgements (task_custody.settle_task_mailbox):
    their absence proves no non-read, so the answer stays owed with that recipient and goes nowhere else. Its end is
    observed before the letter is read back, so a read that landed by then settles it; only a letter still held and
    unread takes the existing late path."""
    from ouroboros import question_answers
    from ouroboros.owner_mailbox import (
        KIND_QUIZ_ANSWER, _ack_path, _mailbox_path, acknowledge_task_messages, drain_owner_entries,
    )

    _ask(tmp_path)
    _live(monkeypatch, tmp_path)
    block = record_answered(tmp_path, "task-1", quiz_id="q1", option_index=None, request_id="r1", comment="this one")["block"]
    ended, routed = [], []

    def read_letter():  # the reader's own durable read; acknowledge_answer's receipt never landed
        entry, = [e for e in drain_owner_entries(tmp_path, "task-1", set()) if e.get("kind") == KIND_QUIZ_ANSWER]
        assert acknowledge_task_messages(tmp_path, "task-1", [entry["msg_id"]], wake_id="loop_delivery")

    def reader(task_id):
        if not ended:
            return {"id": task_id, "chat_id": 1}, ""
        if case == "read-as-it-ended" and len(ended) == 1:
            ended.append(True)
            read_letter()  # it read the letter, then ended, just before this observation
        return None, "task_not_live"

    first = question_answers.deliver_answer(tmp_path, "task-1", "q1", block, reader, lambda *a, **k: (True, ""))
    assert first == (False, "awaiting_recipient_read")
    if case == "read-then-cleaned":
        read_letter()
    ended.append(True)
    if case.endswith("cleaned"):
        _mailbox_path(tmp_path, "task-1").unlink()
        _ack_path(tmp_path, "task-1").unlink(missing_ok=True)

    outcome = question_answers.deliver_answer(tmp_path, "task-1", "q1", block, reader,
                                              lambda *a, **k: routed.append(a) or (True, ""))

    delivery = quiz_states(tmp_path, "task-1")["q1"]["delivery"]
    if case == "unread-held":
        assert len(routed) == 1 and outcome == (True, ""), "proven unread after its end: the existing late path, once"
    elif case == "read-as-it-ended":
        assert routed == [] and outcome == (False, "quiz_history_write_pending") and delivery["read_at"]
    else:
        assert routed == [] and outcome == (False, "mailbox_write_failed"), "no acknowledgement left proves no non-read"
        assert (delivery["recipient_task_id"], delivery["msg_id"], delivery["state"]) == (
            "task-1", "quiz_answer:task-1:q1", "owed")


def _unconfirmed(root, files, store=None, *, admitted=False):
    """An accepted answer whose earlier attempt named its reader, letter id and store but never recorded the write
    receipt (``mailbox_at``): a harmless synthetic confirmation that did not complete. ``files``: the host's fact of a
    file that never arrived rides with the words. ``admitted``: the asker's lifecycle row precedes its ask, as a real
    asker's does; without it the asker's record holds only the question, which strict admission refuses."""
    from ouroboros import question_answers
    from ouroboros.task_results import write_task_result

    if admitted:
        write_task_result(root, "task-1", "running", chat_id=1)
    _ask(root)
    fact = chat_uploads.delivery_ref({"name": "clip.mp4", "unavailable": "download_failed"})
    block = record_answered(root, "task-1", quiz_id="q1", option_index=None, request_id="r1", comment="this one",
                            attachments=[fact] if files else None)["block"]
    question_answers._update_delivery(root, "task-1", "q1", {
        "recipient_task_id": "task-1", "msg_id": "quiz_answer:task-1:q1", "recipient_drive": str(store or root),
        "state": "owed", "mailbox_at": "", "reason_code": "awaiting_recipient_read"})
    return block


@pytest.mark.parametrize("case", ["file-fact", "words-only", "file-fact-read-as-it-ended"])
def test_an_unconfirmed_write_to_an_ended_reader_stays_owed_only_for_a_file_answer(tmp_path, monkeypatch, case):
    """The admitted reader stopped reading before its record settled, so its mailbox and acknowledgements (never cleaned
    before settlement) are still the authority: no receipt, no letter held, no read. Words, buttons and protected input
    keep their earlier route; an answer with a file (or the host's fact of one never received) stays owed and goes
    nowhere else, unless the ledger, read again after its end, shows the read."""
    from ouroboros import question_answers
    from ouroboros.owner_mailbox import acknowledge_task_messages

    block = _unconfirmed(tmp_path, files=case != "words-only", admitted=True)
    _live(monkeypatch, tmp_path)
    routed, observed = [], []

    def reader(task_id):  # ended; the last case read the letter just before this very observation
        if case.endswith("ended") and not observed:
            assert acknowledge_task_messages(tmp_path, task_id, ["quiz_answer:task-1:q1"], wake_id="loop_delivery")
        observed.append(task_id)
        return None, "task_not_live"

    outcome = question_answers.deliver_answer(tmp_path, "task-1", "q1", block, reader,
                                              lambda *a, **k: routed.append(a) or (True, ""))

    delivery = quiz_states(tmp_path, "task-1")["q1"]["delivery"]
    if case == "file-fact":
        assert routed == [] and outcome == (False, "mailbox_write_failed")
        assert (delivery["recipient_task_id"], delivery["msg_id"], delivery["state"]) == (
            "task-1", "quiz_answer:task-1:q1", "owed")
    elif case == "words-only":
        assert len(routed) == 1 and outcome == (True, ""), "the earlier route of words, buttons and protected input"
    else:
        assert routed == [] and outcome == (False, "quiz_history_write_pending") and delivery["read_at"]


@pytest.mark.parametrize("case", ["same-store", "other-store", "vanished"])
def test_an_unconfirmed_file_letter_is_repeated_only_at_its_own_address(tmp_path, monkeypatch, case):
    """While its reader lives, the earlier write is repeated at that reader, with that letter id, in that store. Where
    the store changed, or the reader vanished before any was chosen, nothing is copied, written or sent to another
    head: the earlier outcome is read afresh and the answer stays owed."""
    from ouroboros import question_answers
    from ouroboros.owner_mailbox import KIND_QUIZ_ANSWER, drain_owner_entries

    earlier_store = tmp_path / "earlier-store"
    block = _unconfirmed(tmp_path, True, store=earlier_store if case == "other-store" else tmp_path)
    _live(monkeypatch, tmp_path)
    seen, routed = [], []

    def reader(task_id):  # alive when the earlier write is read back; "vanished" is gone before a reader is chosen
        seen.append(task_id)
        return (None, "task_not_live") if case == "vanished" and len(seen) > 1 else ({"id": task_id, "chat_id": 1}, "")

    outcome = question_answers.deliver_answer(tmp_path, "task-1", "q1", block, reader,
                                              lambda *a, **k: routed.append(a) or (True, ""))

    delivery = quiz_states(tmp_path, "task-1")["q1"]["delivery"]
    letters = [e for e in drain_owner_entries(tmp_path, "task-1", set()) if e.get("kind") == KIND_QUIZ_ANSWER]
    assert routed == [], "no late or fresh head receives a second copy"
    if case == "same-store":
        assert outcome == (False, "awaiting_recipient_read") and delivery["mailbox_at"]
        assert [e["msg_id"] for e in letters] == ["quiz_answer:task-1:q1"], "the same letter id, once"
    else:
        assert outcome == (False, "mailbox_write_failed") and letters == []
        assert (delivery["recipient_drive"], delivery["state"]) == (
            str(earlier_store if case == "other-store" else tmp_path), "owed")


def test_a_rejoin_miss_is_not_an_absence_while_the_history_import_is_incomplete(tmp_path):
    """The boundary BEFORE any SQL store exists, kept separate from the cutover's ledger branch: a root holding the
    frozen chat chain and no store yet cannot prove an indexed miss to be an absence, because
    ``dialogue_import.coverage`` answers it from ``dialogue_store.holds_dialogue_files`` (``imported`` False), so the
    bridge is told the record could not be read — it downloads nothing and re-answers nothing — while the same
    question against a fresh root that holds no chain file IS a proven absence. Nothing here opens or creates a
    store first: the rejoin read is this root's first touch, and a read never imports."""
    from ouroboros import dialogue_store as ds

    body = {"request_id": "tg:9:m:42:7", "decision_id": "quiz:task-1:q1", "rejoin": True}
    legacy = tmp_path / "logs" / "chat.jsonl"
    legacy.parent.mkdir()
    legacy.write_text('{"type": "chat", "text": "older history"}\n', encoding="utf-8")

    status, payload = td.rejoin_decision(tmp_path, body)
    assert (status, payload["reason_code"]) == (503, "quiz_answer_unconfirmed")
    assert payload["error"] == "The answer record could not be read."
    assert not ds.present(tmp_path), "answered from the chain file, not from a store this read created"

    no_chain = tmp_path / "no-chain-yet"
    no_chain.mkdir()
    status, payload = td.rejoin_decision(no_chain, body)
    assert (status, payload["reason_code"]) == (404, "request_not_recorded")


def test_a_rejoin_miss_re_read_after_the_import_lands_is_the_record_not_an_absence(tmp_path, monkeypatch):
    """An indexed miss taken while an import is still owed is not judged on its own: when the coverage read says
    the import landed, the miss is read once more and the row that arrived meanwhile is the record — the replay's
    200, not a 404 absence. The stub runs the ordinary import; a hit pays for no extra read."""
    from ouroboros import dialogue_import
    from tests._dialogue import seed

    _ask(tmp_path, quiz="q0")  # the document holds q0; the row that arrives below is q1
    seed(tmp_path, archives={"chat_20261120T000000.jsonl": [
        {"ts": "2026-11-20T00:00:00+00:00", "direction": "system", "type": "quiz_answer",
         "source": "owner_quiz_answer", "chat_id": 1, "task_id": "task-1",
         "client_message_id": "quiz_answer:task-1:q1", "text": "The owner answered.",
         "quiz": {"quiz_id": "q1", "state": "answered", "request_id": "tg:9:m:42:7",
                  "answered_index": 0, "comment": "First"}}]})
    dialogue_import.cutover(tmp_path)  # the store and its plan: this segment stays owed
    body = {"request_id": "tg:9:m:42:7", "decision_id": "quiz:task-1:q1", "rejoin": True}
    assert td.rejoin_decision(tmp_path, body)[1]["reason_code"] == "quiz_answer_unconfirmed", \
        "the owed import leaves the miss unproven"

    real_coverage = dialogue_import.coverage

    def coverage_with_the_import_landing(root, stream, seq=None):
        dialogue_import.run_dialogue_import(root)  # the ordinary job: the owed segment lands here
        return real_coverage(root, stream, seq)

    monkeypatch.setattr(dialogue_import, "coverage", coverage_with_the_import_landing)
    status, payload = td.rejoin_decision(tmp_path, body)
    assert (status, payload["duplicate"], payload["state"]) == (200, True, "answered"), \
        "the row that arrived meanwhile is the record, never an absence"
    assert payload["answered_index"] == 0


def test_a_rejoin_miss_after_a_complete_import_is_a_definite_absence(tmp_path):
    """With the canonical history's import complete, an indexed miss is the question's absence: the bridge gets
    the old 404 ``request_not_recorded``, never a 503 — an unconfirmed read is only for a miss the coverage
    cannot settle."""
    from ouroboros import dialogue_import
    from tests._dialogue import seed

    _ask(tmp_path, quiz="q0")
    seed(tmp_path, archives={"chat_20261120T000000.jsonl": [
        {"ts": "2026-11-20T00:00:00+00:00", "direction": "system", "type": "chat",
         "chat_id": 1, "text": "older words"}]})
    dialogue_import.cutover(tmp_path)
    dialogue_import.run_dialogue_import(tmp_path)
    assert dialogue_import.coverage(tmp_path, "chat")["imported"] is True
    status, payload = td.rejoin_decision(
        tmp_path, {"request_id": "tg:9:m:42:7", "decision_id": "quiz:task-1:q1", "rejoin": True})
    assert (status, payload["reason_code"]) == (404, "request_not_recorded")
    assert "state" not in payload


@pytest.mark.parametrize("order", [
    "continue_then_recovery",
    "late_then_continue",
    "late_unsettled_then_continue",
])
def test_file_answer_handoff_orders_keep_one_delivery_and_no_leaked_attachments(tmp_path, monkeypatch, order):
    """A benign file answer settles through one delivery owner across both Continue-then-recovery and
    late-then-Continue orderings (even after mailbox settlement and even when the late route is still owed),
    never leaking the suppressed answer's attachment into Continue's opening or successor artifact store."""
    from ouroboros import owner_quiz, process_custody
    from ouroboros.owner_mailbox import (
        KIND_OWNER_TEXT, KIND_QUIZ_ANSWER, acknowledge_transcript_entry, drain_owner_entries, write_owner_message,
    )
    from ouroboros.task_custody import settle_task_mailbox
    from ouroboros.task_results import STATUS_CANCELLED, write_task_result
    from supervisor.continuation_admission import admit_continuation
    from tests._budget_pause_exact_helpers import _install_queue
    from tests._owner_continue_helpers import NONCE, _running_predecessor
    from tests.test_question_answer_routing import due
    from tests.test_quiz_late_answer import _inbox, _late_bridge

    q, _state, workers = _install_queue(tmp_path, monkeypatch)
    _running_predecessor(tmp_path, "asker", chat_id=1)
    q.RUNNING["asker"] = {"task": {"id": "asker", "chat_id": 1, "drive_root": str(tmp_path)}}
    monkeypatch.setattr(q, "_task_drive_for_task", lambda _task, _tid: tmp_path)
    _ask(tmp_path, task="asker", quiz="q")
    bridge, _frames = _late_bridge(tmp_path, monkeypatch)
    ref = _upload(tmp_path)

    assert write_owner_message(tmp_path, "Keep the appendix", "asker", msg_id="owner-2", kind=KIND_OWNER_TEXT)
    status, _body = _answer(tmp_path, {"request_id": "tg:1:m:9:1", "decision_id": "quiz:asker:q",
                                       "comment": "see photo"}, attachments=[ref])
    assert status == 200
    assert quiz_states(tmp_path, "asker")["q"]["delivery"]["recipient_task_id"] == "asker"

    q.RUNNING.clear()
    write_task_result(tmp_path, "asker", STATUS_CANCELLED, result="interrupted",
                      cancel_origin={"source": "snapshot_restore", "reason": "server_shutdown"})
    assert settle_task_mailbox(tmp_path, "asker", tmp_path) is True

    if order == "continue_then_recovery":
        admitted = admit_continuation("asker", action_nonce=NONCE)
        assert admitted["ok"] is True, admitted
        successor = admitted["successor_task_id"]
        task = next(row for row in workers.PENDING if row["id"] == successor)
        assert [row["content"] for row in task["metadata"]["continuation"]["owner_sources"]["later"]] == [
            "Keep the appendix"]
        assert "attachment_manifest" not in task
        assert not (tmp_path / "task_results" / "artifacts" / successor).exists()

        due(tmp_path)
        asyncio.run(td.recover_question_deliveries(tmp_path))
        assert _inbox(bridge) == []
        delivery = quiz_states(tmp_path, "asker")["q"]["delivery"]
        assert delivery["recipient_task_id"] == successor and delivery["state"] == "owed"
        from ouroboros.artifacts import task_artifact_dir_path

        [letter] = [e for e in drain_owner_entries(tmp_path, successor, set()) if e.get("kind") == KIND_QUIZ_ANSWER]
        [staged] = [row for row in letter.get("attachment_manifest") or [] if row.get("status") == "staged"]
        assert staged["sha256"] == ref["sha256"]
        assert (task_artifact_dir_path(tmp_path, successor) / staged["relpath"]).read_bytes() == _PNG
        assert acknowledge_transcript_entry(tmp_path, successor, letter)
        assert quiz_states(tmp_path, "asker")["q"]["delivery"]["state"] == "delivered"
        return

    due(tmp_path)
    with monkeypatch.context() as lost:
        if order == "late_unsettled_then_continue":
            lost.setattr(owner_quiz, "mark_discharged", lambda *a, **k: {})
        asyncio.run(td.recover_question_deliveries(tmp_path))
    [late] = _inbox(bridge)
    assert late["task_metadata"]["chat_attachments"] == [ref]
    if order == "late_unsettled_then_continue":
        monkeypatch.setattr(process_custody, "current_custody_session_id", lambda: "next-generation")

    admitted = admit_continuation("asker", action_nonce=NONCE)
    assert admitted["ok"] is True, admitted
    successor = admitted["successor_task_id"]
    task = next(row for row in workers.PENDING if row["id"] == successor)
    assert [row["content"] for row in task["metadata"]["continuation"]["owner_sources"]["later"]] == [
        "Keep the appendix"]
    assert "attachment_manifest" not in task
    assert not (tmp_path / "task_results" / "artifacts" / successor).exists()

    due(tmp_path)
    asyncio.run(td.recover_question_deliveries(tmp_path))
    assert _inbox(bridge) == []
    assert drain_owner_entries(tmp_path, successor, set()) == []


@pytest.mark.parametrize("kind", ["file_only", "unavailable_fact"])
def test_consumed_file_only_and_unavailable_fact_answers_carry_through_continue_after_settlement(
        tmp_path, monkeypatch, kind):
    """A consumed file-only quiz answer (or unavailable-file delivery fact alone) whose positive ACK
    is persisted by mailbox settlement rides into Continue's opening with its staged file or delivery
    fact intact, and recovery never resends it."""
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.owner_mailbox import acknowledge_task_messages, drain_owner_entries
    from ouroboros.task_custody import settle_task_mailbox
    from ouroboros.task_results import STATUS_CANCELLED, write_task_result
    from supervisor.continuation_admission import admit_continuation
    from tests._budget_pause_exact_helpers import _install_queue
    from tests._owner_continue_helpers import NONCE, _running_predecessor
    from tests.test_question_answer_routing import due
    from tests.test_quiz_late_answer import _inbox, _late_bridge

    q, _state, workers = _install_queue(tmp_path, monkeypatch)
    _running_predecessor(tmp_path, "asker", chat_id=1)
    q.RUNNING["asker"] = {"task": {"id": "asker", "chat_id": 1, "drive_root": str(tmp_path)}}
    monkeypatch.setattr(q, "_task_drive_for_task", lambda _task, _tid: tmp_path)
    _ask(tmp_path, task="asker", quiz="q")
    bridge, _frames = _late_bridge(tmp_path, monkeypatch)
    ref = (_upload(tmp_path) if kind == "file_only"
           else chat_uploads.delivery_ref({"name": "clip.mp4", "unavailable": "download_failed"}))

    status, _body = _answer(tmp_path, {"request_id": "tg:1:m:9:2", "decision_id": "quiz:asker:q"}, attachments=[ref])
    assert status == 200
    assert acknowledge_task_messages(tmp_path, "asker", ["quiz_answer:asker:q"], wake_id="loop_delivery")
    assert not quiz_states(tmp_path, "asker")["q"]["delivery"].get("read_at")

    q.RUNNING.clear()
    write_task_result(tmp_path, "asker", STATUS_CANCELLED, result="interrupted",
                      cancel_origin={"source": "snapshot_restore", "reason": "server_shutdown"})
    assert settle_task_mailbox(tmp_path, "asker", tmp_path) is True
    assert quiz_states(tmp_path, "asker")["q"]["delivery"]["state"] == "delivered"

    admitted = admit_continuation("asker", action_nonce=NONCE)
    assert admitted["ok"] is True, admitted
    successor = admitted["successor_task_id"]
    task = next(row for row in workers.PENDING if row["id"] == successor)
    [carried] = task["metadata"]["continuation"]["owner_sources"]["later"]
    assert carried["quiz_id"] == "q" and carried["content"] == ""
    assert "The owner answered with an attachment and no words" in task["text"]
    if kind == "file_only":
        [staged] = [row for row in task.get("attachment_manifest") or [] if row.get("status") == "staged"]
        assert staged["sha256"] == ref["sha256"]
        assert (task_artifact_dir_path(tmp_path, successor) / staged["relpath"]).read_bytes() == _PNG
    else:
        assert [item.get("unavailable") for item in carried.get("attachments") or []] == ["download_failed"]
        assert not [row for row in task.get("attachment_manifest") or [] if row.get("status") == "staged"]

    due(tmp_path)
    asyncio.run(td.recover_question_deliveries(tmp_path))
    assert _inbox(bridge) == [] and drain_owner_entries(tmp_path, successor, set()) == []


def test_unreadable_quiz_receipt_omits_quiz_file_attachment_while_continue_proceeds(tmp_path, monkeypatch):
    """When an addressed file-bearing quiz_answer's asker record cannot be read, Continue omits both
    the letter and its staged attachment without failing admission for other valid owner mail."""
    from ouroboros import task_store
    from ouroboros.owner_mailbox import KIND_OWNER_TEXT, KIND_QUIZ_ANSWER, write_owner_message
    from ouroboros.server_owner_routing import _stage_mailbox_attachments
    from supervisor.continuation_admission import admit_continuation
    from tests._budget_pause_exact_helpers import _install_queue
    from tests._owner_continue_helpers import NONCE, _interrupted

    _q, _state, workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, task_id="pred-1", chat_id=1)
    assert write_owner_message(tmp_path, "Keep the appendix", "pred-1", msg_id="owner-2", kind=KIND_OWNER_TEXT)
    ref = _upload(tmp_path)
    note, manifest, _ = _stage_mailbox_attachments(
        None, tmp_path, "pred-1", {"chat_attachment_uploads": chat_uploads.staging_specs([ref], tmp_path)})
    card = {"state": "answered", "quiz_id": "q", "question": "Which screen?", "options": ["A"], "attachments": [ref]}
    assert write_owner_message(tmp_path, "see photo" + note, "pred-1", msg_id="quiz_answer:other-asker:q",
                               kind=KIND_QUIZ_ANSWER, attachment_manifest=manifest,
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
    task = next(row for row in workers.PENDING if row["id"] == successor)
    assert [row["content"] for row in task["metadata"]["continuation"]["owner_sources"]["later"]] == [
        "Keep the appendix"]
    assert "attachment_manifest" not in task
    assert not (tmp_path / "task_results" / "artifacts" / successor).exists()


@pytest.mark.parametrize("case", [
    "complete_non_write",
    "settled_unread_receipt",
    "settled_read_receipt",
    "legacy_unread_without_receipt",
    "incomplete_owner",
    "incomplete_unread",
])
def test_unconfirmed_file_write_uses_exact_settlement_receipts_and_holds_on_incomplete(tmp_path, monkeypatch, case):
    """After an unconfirmed file write whose reader settled, complete exact non-write or an exact
    settled_unread receipt moves on once, a positive ACK receipt persisted before cleanup settles
    without resend, and legacy/incomplete captures stay owed."""
    from ouroboros import question_answers
    from ouroboros.owner_mailbox import KIND_QUIZ_ANSWER, acknowledge_task_messages, write_owner_message
    from ouroboros.owner_quiz import quiz_delivery_due
    from ouroboros.task_custody import settle_task_mailbox
    from ouroboros.task_results import STATUS_CANCELLED, write_task_result
    from ouroboros.task_store import patch_document

    block = _unconfirmed(tmp_path, files=True)
    _live(monkeypatch, tmp_path)
    letter_line = json.dumps({"msg_id": "quiz_answer:task-1:q1", "kind": "quiz_answer", "text": "this one"})
    if case in {"settled_unread_receipt", "settled_read_receipt"}:
        assert write_owner_message(tmp_path, "this one", "task-1", msg_id="quiz_answer:task-1:q1",
                                   kind=KIND_QUIZ_ANSWER,
                                   quiz_delivery={"task_id": "task-1", "quiz_id": "q1",
                                                  "canonical_root": str(tmp_path), "answer_snapshot": dict(block)})
        if case == "settled_read_receipt":
            assert acknowledge_task_messages(tmp_path, "task-1", ["quiz_answer:task-1:q1"], wake_id="loop_delivery")
        write_task_result(tmp_path, "task-1", STATUS_CANCELLED, result="interrupted")
        question_answers._update_delivery(tmp_path, "task-1", "q1", {"next_retry_at": "2099-01-01T00:00:00+00:00"})
        assert settle_task_mailbox(tmp_path, "task-1", tmp_path) is True
        # A proven unread letter is due for recovery at once, not after the backoff its reader's life ran up.
        assert quiz_delivery_due(quiz_states(tmp_path, "task-1")["q1"]) is (case == "settled_unread_receipt")
    else:
        write_task_result(tmp_path, "task-1", STATUS_CANCELLED, result="interrupted")

        def seed_custody(current):
            updated = dict(current)
            if case == "complete_non_write":  # a capture from before stores were recorded: its silence names no store
                updated["owner_mailbox"] = {"schema": 1, "total": 0, "rows": [], "read_complete": True,
                                            "captured_at": "2026-11-24T00:00:00+00:00"}
                updated.pop("unread_mailbox", None)
            elif case == "legacy_unread_without_receipt":
                updated["owner_mailbox"] = {"schema": 1, "total": 1, "rows": [letter_line], "read_complete": True,
                                            "captured_at": "2026-11-24T00:00:00+00:00"}
                updated["unread_mailbox"] = {"schema": 1, "total": 1, "rows": [letter_line], "read_complete": True,
                                             "captured_at": "2026-11-24T00:00:00+00:00"}
            elif case == "incomplete_owner":
                updated["owner_mailbox"] = {"schema": 1, "total": 0, "rows": [], "read_complete": False,
                                            "captured_at": "2026-11-24T00:00:00+00:00"}
                updated.pop("unread_mailbox", None)
            else:  # incomplete_unread: the owner capture alone would prove non-write; the unread one is partial
                updated["owner_mailbox"] = {"schema": 1, "total": 0, "rows": [], "read_complete": True,
                                            "captured_at": "2026-11-24T00:00:00+00:00", "stores": ["canonical"]}
                updated["unread_mailbox"] = {"schema": 1, "total": 0, "rows": [], "read_complete": False,
                                             "captured_at": "2026-11-24T00:00:00+00:00"}
            return updated

        patch_document(tmp_path, "task-1", seed_custody)

    routed = []
    outcome = question_answers.deliver_answer(
        tmp_path, "task-1", "q1", block, lambda _tid: (None, "task_not_live"),
        lambda *a, **k: routed.append(a) or (True, ""),
    )
    delivery = quiz_states(tmp_path, "task-1")["q1"]["delivery"]
    if case == "settled_unread_receipt":
        assert outcome == (True, "") and len(routed) == 1
    elif case == "settled_read_receipt":
        assert outcome == (False, "quiz_history_write_pending") and routed == [] and delivery["read_at"]
    else:
        assert outcome == (False, "mailbox_write_failed") and routed == []
        assert delivery["state"] == "owed" and not delivery.get("read_at")


@pytest.mark.parametrize("case", ["retired", "never_read"])
def test_a_store_the_capture_read_keeps_its_non_write_proof_after_its_drive_retires(tmp_path, monkeypatch, case):
    """The owner capture names each store it read whole, and the non-write proof reads those, never the folders found
    now: an unconfirmed file write to the reader's own child drive moves on once after the real settlement retired that
    drive, and stays owed where the capture never read the drive that exists now."""
    from ouroboros import headless, question_answers, task_custody
    from ouroboros.task_results import STATUS_CANCELLED, load_task_result, write_task_result

    drive = tmp_path / "state" / "headless_tasks" / "task-1" / "data"
    if case == "retired":
        assert headless.prepare_task_drive(tmp_path, "task-1", "empty") == drive
    block = _unconfirmed(tmp_path, True, store=drive)
    _live(monkeypatch, tmp_path)
    write_task_result(tmp_path, "task-1", STATUS_CANCELLED, result="interrupted")  # the real terminal capture
    if case == "retired":
        settled = task_custody.settle_child_drive(tmp_path, "task-1", drive, live=lambda _task_id: False)
        assert settled["status"] == "removed" and not drive.exists(), settled
    else:
        headless.prepare_task_drive(tmp_path, "task-1", "empty")
    assert load_task_result(tmp_path, "task-1")["owner_mailbox"]["stores"] == (
        ["canonical", "headless"] if case == "retired" else ["canonical"])

    routed = []
    outcome = question_answers.deliver_answer(tmp_path, "task-1", "q1", block, lambda _tid: (None, "task_not_live"),
                                              lambda *a, **k: routed.append(a) or (True, ""))

    if case == "retired":
        assert outcome == (True, "") and len(routed) == 1
    else:
        assert outcome == (False, "mailbox_write_failed") and routed == []
        assert quiz_states(tmp_path, "task-1")["q1"]["delivery"]["state"] == "owed"


@pytest.mark.parametrize("case", [
    "settled_missing_captures",
    "settled_complete_non_write",
    "protected_settled_missing_captures",
    "unadmitted_record",
])
def test_unconfirmed_non_file_write_treats_settled_missing_captures_as_unknown(tmp_path, monkeypatch, case):
    """An actual settled result with missing mailbox captures does not prove non-write for an
    unconfirmed ordinary text/button answer, nor does a reader's record that strict admission refuses,
    while complete non-write moves on and protected outcomes keep their pre-change legacy behavior."""
    from ouroboros import question_answers
    from ouroboros.task_results import STATUS_CANCELLED, write_task_result
    from ouroboros.task_store import patch_document

    block = _unconfirmed(tmp_path, files=False)
    _live(monkeypatch, tmp_path)
    if case != "unadmitted_record":  # unadmitted: the record holds only the question, no lifecycle row
        write_task_result(tmp_path, "task-1", STATUS_CANCELLED, result="interrupted")

    def adjust(current):
        updated = dict(current)
        if case != "settled_complete_non_write":
            updated.pop("owner_mailbox", None)
            updated.pop("unread_mailbox", None)
        if case == "protected_settled_missing_captures":
            quizzes = dict(updated.get("owner_quiz") or {})
            quizzes["q1"] = {
                **quizzes["q1"],
                "secret_request": {"plugin": "demo", "pluginName": "Demo", "fields": [{"id": "api_key"}]},
                "secret_outcome": {"event": "secret_filled", "pluginName": "Demo", "fields": ["api_key"]},
            }
            updated["owner_quiz"] = quizzes
        return updated

    patch_document(tmp_path, "task-1", adjust)
    routed = []
    outcome = question_answers.deliver_answer(
        tmp_path, "task-1", "q1", block, lambda _tid: (None, "task_not_live"),
        lambda *a, **k: routed.append(a) or (True, ""),
    )
    delivery = quiz_states(tmp_path, "task-1")["q1"]["delivery"]
    if case in {"settled_missing_captures", "unadmitted_record"}:
        assert outcome == (False, "mailbox_write_failed") and routed == []
        assert delivery["state"] == "owed" and not delivery.get("read_at")
    else:
        assert outcome == (True, "") and len(routed) == 1


def test_ordinary_delivery_guard_serializes_same_question_and_lets_other_quiz_proceed(tmp_path, monkeypatch):
    """A per-question in-flight owner guards ordinary delivery: a concurrent duplicate on the same
    question neither re-stages nor double-delivers, while a different quiz proceeds without waiting,
    and releasing the guard lets subsequent checks run normally."""
    import threading

    from ouroboros import question_answers
    from ouroboros.owner_mailbox import KIND_QUIZ_ANSWER, drain_owner_entries

    ref1, ref2 = _upload(tmp_path), _upload(tmp_path, b"\x89PNG\r\n\x1a\nsecond", "second.png")
    _ask(tmp_path, quiz="q1")
    _ask(tmp_path, quiz="q2")
    _live(monkeypatch, tmp_path)
    b1 = record_answered(tmp_path, "task-1", quiz_id="q1", option_index=None, request_id="r1",
                         attachments=[ref1])["block"]
    b2 = record_answered(tmp_path, "task-1", quiz_id="q2", option_index=None, request_id="r2",
                         attachments=[ref2])["block"]

    entered, release = threading.Event(), threading.Event()
    stage_calls = {"q1": 0, "q2": 0}
    real_stage = question_answers._staged_answer_files

    def gated_stage(root, drive, recipient, blk):
        qid = str(blk.get("quiz_id") or "")
        stage_calls[qid] = stage_calls.get(qid, 0) + 1
        copies = real_stage(root, drive, recipient, blk)
        if qid == "q1":
            entered.set()
            release.wait(timeout=5)
        return copies

    monkeypatch.setattr(question_answers, "_staged_answer_files", gated_stage)
    reader = lambda task_id: ({"id": task_id, "chat_id": 1}, "")
    routed, owner_outcome = [], []
    fallback = lambda *a, **k: routed.append(a) or (True, "")

    worker = threading.Thread(
        target=lambda: owner_outcome.append(
            question_answers.deliver_answer(tmp_path, "task-1", "q1", b1, reader, fallback)
        ),
        daemon=True,
    )
    worker.start()
    try:
        assert entered.wait(timeout=5)
        dup = question_answers.deliver_answer(tmp_path, "task-1", "q1", b1, reader, fallback)
        assert dup == (False, "")
        assert stage_calls["q1"] == 1
        assert not quiz_states(tmp_path, "task-1")["q1"]["delivery"].get("reason_code")

        other = question_answers.deliver_answer(tmp_path, "task-1", "q2", b2, reader, fallback)
        assert other == (False, "awaiting_recipient_read")
        assert stage_calls["q2"] == 1
        assert [e["msg_id"] for e in drain_owner_entries(tmp_path, "task-1", set())
                if e.get("kind") == KIND_QUIZ_ANSWER] == ["quiz_answer:task-1:q2"]
    finally:
        release.set()
        worker.join(timeout=5)

    assert not worker.is_alive()
    assert owner_outcome == [(False, "awaiting_recipient_read")]
    assert routed == [] and stage_calls == {"q1": 1, "q2": 1}
    letters = [e["msg_id"] for e in drain_owner_entries(tmp_path, "task-1", set())
               if e.get("kind") == KIND_QUIZ_ANSWER]
    assert sorted(letters) == ["quiz_answer:task-1:q1", "quiz_answer:task-1:q2"]
    assert question_answers.deliver_answer(tmp_path, "task-1", "q1", b1, reader, fallback) == (
        False, "awaiting_recipient_read",
    )


@pytest.mark.parametrize("case", ["existing_cause", "delivery_unknown", "late_delivered_before_readback"])
def test_inflight_follower_preserves_existing_cause_and_actual_delivery_unknown_reason(tmp_path, monkeypatch, case):
    """Through the public gateway ingress, a concurrent follower hitting the in-flight guard never
    overwrites an existing delivery cause with an in-flight marker, still surfaces an actual
    ``late_answer_delivery_unknown`` reason on a late-answer retry, and agrees with canonical
    readback when the late ingress route delivered after the follower's snapshot."""
    import threading

    from ouroboros import question_answers
    from ouroboros.owner_mailbox import KIND_QUIZ_ANSWER, drain_owner_entries

    _ask(tmp_path)
    entered, release = threading.Event(), threading.Event()
    owner_outcome: list[tuple[bool, str]] = []

    if case == "existing_cause":
        ref = _upload(tmp_path)
        _live(monkeypatch, tmp_path, live=True)
        block = record_answered(tmp_path, "task-1", quiz_id="q1", option_index=None, request_id="r1",
                                comment="this one", attachments=[ref])["block"]
        question_answers._update_delivery(tmp_path, "task-1", "q1", {
            "recipient_task_id": "task-1", "msg_id": "quiz_answer:task-1:q1",
            "recipient_drive": str(tmp_path), "state": "owed", "mailbox_at": "",
            "reason_code": "mailbox_write_failed",
        })
        stage_calls, real_stage = [0], question_answers._staged_answer_files

        def paused_stage(*args):
            stage_calls[0] += 1
            copies = real_stage(*args)
            entered.set()
            release.wait(timeout=5)
            return copies

        monkeypatch.setattr(question_answers, "_staged_answer_files", paused_stage)
        worker = threading.Thread(
            target=lambda: owner_outcome.append(
                question_answers.deliver_answer(tmp_path, "task-1", "q1", block, td._mail_holder,
                                                lambda *a, **k: (False, "unexpected"))
            ),
            daemon=True,
        )
        worker.start()
        try:
            assert entered.wait(timeout=5)
            status, payload = _answer(
                tmp_path, {"request_id": "r1", "decision_id": "quiz:task-1:q1", "comment": "this one"},
                attachments=[ref],
            )
            assert status == 200 and payload["duplicate"] is True
            assert payload["question_delivery"] == {
                "kind": "answer", "state": "owed", "reason_code": "mailbox_write_failed",
            }
            assert quiz_states(tmp_path, "task-1")["q1"]["delivery"]["reason_code"] == "mailbox_write_failed"
            assert stage_calls == [1]
        finally:
            release.set()
            worker.join(timeout=5)

        assert not worker.is_alive()
        assert owner_outcome == [(False, "awaiting_recipient_read")]
        assert quiz_states(tmp_path, "task-1")["q1"]["delivery"]["mailbox_at"]
        entries = [e for e in drain_owner_entries(tmp_path, "task-1", set()) if e.get("kind") == KIND_QUIZ_ANSWER]
        assert [e["msg_id"] for e in entries] == ["quiz_answer:task-1:q1"]
        return

    reconcile_terminal(tmp_path, "task-1")
    _live(monkeypatch, tmp_path, live=False)
    block = record_answered(tmp_path, "task-1", quiz_id="q1", option_index=None, request_id="r1",
                            comment="this one", allow_expired=True)["block"]
    if case == "delivery_unknown":
        question_answers._update_delivery(tmp_path, "task-1", "q1", {
            "recipient_task_id": "", "msg_id": "quiz_late_answer:task-1:q1", "route": "fresh_head",
            "state": "owed", "reason_code": "late_answer_delivery_unknown",
        })
    fallback_calls, follower_forwarded, follower_snapshot = [0], [], []
    monkeypatch.setattr(td, "deliver_as_owner_message",
                        lambda *a, **k: follower_forwarded.append(a) or (True, ""))

    def paused_fallback(*_args, **_kwargs):
        fallback_calls[0] += 1
        entered.set()
        release.wait(timeout=5)
        return (True, "") if case == "late_delivered_before_readback" else (False, "late_answer_delivery_unknown")

    worker = threading.Thread(
        target=lambda: owner_outcome.append(
            question_answers.deliver_answer(tmp_path, "task-1", "q1", block,
                                            lambda _tid: (None, "task_not_live"), paused_fallback)
        ),
        daemon=True,
    )
    if case == "late_delivered_before_readback":
        real_follower = question_answers._inflight_follower_outcome

        def follower_then_owner_finishes(*args):
            outcome = real_follower(*args)
            follower_snapshot.append(outcome)
            release.set()
            worker.join(timeout=5)
            return outcome

        monkeypatch.setattr(question_answers, "_inflight_follower_outcome", follower_then_owner_finishes)
    worker.start()
    try:
        assert entered.wait(timeout=5)
        status, payload = _answer(
            tmp_path, {"request_id": "r1", "decision_id": "quiz:task-1:q1", "comment": "this one"},
        )
        assert status == 200 and payload["duplicate"] is True
        if case == "late_delivered_before_readback":
            assert follower_snapshot == [(False, "")]
            assert payload["forwarded"] is True and "reason_code" not in payload
            assert payload["question_delivery"] == {"kind": "answer", "state": "delivered"}
        else:
            assert payload["forwarded"] is False and payload["reason_code"] == "late_answer_delivery_unknown"
            assert payload["question_delivery"] == {
                "kind": "answer", "state": "owed", "reason_code": "late_answer_delivery_unknown",
            }
            assert quiz_states(tmp_path, "task-1")["q1"]["delivery"]["reason_code"] == "late_answer_delivery_unknown"
        assert fallback_calls == [1] and follower_forwarded == []
    finally:
        release.set()
        worker.join(timeout=5)

    assert not worker.is_alive()
    if case == "late_delivered_before_readback":
        assert owner_outcome == [(True, "")]
        assert quiz_states(tmp_path, "task-1")["q1"]["delivery"]["state"] == "delivered"
    else:
        assert owner_outcome == [(False, "late_answer_delivery_unknown")]
        assert quiz_states(tmp_path, "task-1")["q1"]["delivery"]["reason_code"] == "late_answer_delivery_unknown"
