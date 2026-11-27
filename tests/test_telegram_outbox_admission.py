"""What the outbox admits (skills/telegram/lib/telegram_outbox.py): its pre-save filters, question facts and queue head.

An event its handler sends nowhere (no target chat, nothing to send, a question update with no card yet) is dropped
before anything is saved, so a full queue, a failing journal or a missing bot token never turns it into a delivery
issue; the filter says exactly what each handler decides. A question card created after its question settled reads
the host's own record (owner_quiz, the durable source the host writes before it publishes a fact); a record that
cannot be read leaves the facts this load kept. A secret request's card is canonical only as its own record's hint.
A question saved behind a waiting message is decided after it is saved, and the queue's head never runs while its
own decision waits. The host's producers (supervisor.message_bus, ouroboros.owner_quiz, ouroboros.secret_requests)
write and publish; the shipped bridge runs against the loopback Bot API origin of tests/test_telegram_outbox.py.
No Telegram, owner token or network.
"""
from __future__ import annotations

import asyncio
import base64

import pytest

from tests.test_telegram_outbox import (
    _BOT,
    _Bridge,
    _closed_base,
    _failing_journal,
    _not_delivered,
    _Origin,
    _run,
    _until,
)
from tests.test_telegram_outbox_copies import _blob, _bridge, _Producer
from tests.test_telegram_proxy import _TOKEN

_MEANWHILE = "I keep sqlite meanwhile."
_REF = {"kind": "task_artifact", "root": "artifact_store", "task_id": "t1", "path": "notes.txt", "size": 3,
        "sha256": "0" * 64}
_EMPTY = [  # each carries nothing its handler sends
    ("chat.links", {"title": "Links", "actions": [{"label": " ", "url": "https://example.org"}, {"label": "Docs"}, "x"]}),
    ("chat.links", {"title": "Links", "actions": "https://example.org"}),
    ("chat.quiz", {"task_id": "t1", "quiz_id": "q1", "question": "  ", "options": [{"label": "Yes"}]}),
    ("chat.quiz", {"task_id": "t1", "question": "Which db?", "options": [{"label": "sqlite"}]}),
    ("chat.photo", {"image_base64": " ", "caption": "chart"}),
    ("chat.video", {"caption": "clip"}),
    ("chat.document", {"filename": "notes.txt", "file_ref": {}}),
]
_ELIGIBLE = [
    ("chat.links", {"title": "", "actions": [{"label": "Docs", "url": "https://example.org/docs"}]}),
    ("chat.quiz", {"task_id": "t1", "quiz_id": "q1", "question": "Which db?", "options": []}),
    ("chat.photo", {"image_base64": base64.b64encode(b"PNG").decode(), "caption": "chart"}),
    ("chat.video", {"video_base64": base64.b64encode(b"MP4").decode()}),
    ("chat.document", {"filename": "notes.txt", "file_ref": _REF}),
]


@pytest.mark.parametrize("pressure", ["full", "journal", "no-token"])
def test_an_event_that_carries_nothing_to_send_is_never_saved_counted_or_refused(tmp_path, monkeypatch, pressure):
    """An empty link list or question, or media without bytes, is dropped before a full queue, a failing journal or
    a missing token (it would wait a day) can report it as not sent, since its handler's own filters send nothing;
    each control that carries something meets the same pressure as any delivery."""
    async def scenario():
        bridge = _Bridge(tmp_path / "state", _closed_base(), **({"token": ""} if pressure == "no-token" else {}))
        bridge.outbox._load()
        if pressure != "no-token":
            await bridge.publish("chat.outbound", {"text": "A head that waits.", "transport": {}})  # ConnectError
        if pressure == "full":
            monkeypatch.setattr(bridge.module("telegram_outbox"), "MAX_ACTIVE", 1)
        if pressure == "journal":
            _failing_journal(bridge)
        before = bridge.files()
        for topic, event in _EMPTY:
            await bridge.publish(topic, {**event, "transport": {}})
        assert bridge.files() == before and bridge.outbox.summary()["not_sent"] == 0
        assert len(bridge.records()) == (0 if pressure == "no-token" else 1)
        for topic, event in _ELIGIBLE:
            await bridge.publish(topic, {**event, "transport": {}})
        records = bridge.records()
        if pressure == "no-token":  # each waits for a token, as every message does
            assert [r["state"] for r in records] == ["pending"] * len(_ELIGIBLE)
        else:
            assert bridge.outbox.summary()["not_sent"] == len(_ELIGIBLE)
            reason = "full" if pressure == "full" else "journal PermissionError"
            assert [r.get("reason") for r in records[1:]] == ([reason] * len(_ELIGIBLE) if pressure == "full" else [])
            assert _not_delivered(bridge).count("Not sent:") == len(_ELIGIBLE)
    _run(scenario)


@pytest.mark.parametrize("topic,event", [*_EMPTY, *_ELIGIBLE,
                                         ("chat.document", {"file_base64": base64.b64encode(b"doc").decode()}),
                                         ("chat.quiz", {"task_id": "t1", "quiz_id": "q2", "question": "Key?",
                                                        "secret_request": {"plugin_name": "x", "fields": []}})])
def test_the_filter_before_saving_says_what_each_handler_decides(tmp_path, topic, event):
    """The pre-save filter (``copies.carries``) against each shipped handler's own decision run on the same event
    with a target chat: a disagreement would drop what would be sent, or save what never is."""
    async def scenario():
        bridge = _Bridge(tmp_path / "state", _closed_base())
        name = topic.split(".", 1)[1]
        decided = await bridge.outbox._decide(bridge.outbox._bodies[name], {**event, "transport": {}})
        assert decided is bridge.module("telegram_copies").carries(name, event)
        assert bridge.files() == []
    _run(scenario)


def test_media_the_mirror_mode_leaves_out_is_never_refused_by_a_full_queue(tmp_path, monkeypatch):
    """A web photo the bridge never sends (Telegram's own chats only) is not refused as content that did not fit:
    nothing of it is saved, counted or listed. The same photo from the phone's chat is refused."""
    data = tmp_path / "data"

    async def scenario():
        web = _Producer(monkeypatch, data).photo(b"BM" + _blob(20_000, 93), mime="image/bmp")
        phone = {"kind": "telegram", "conversation_id": 42}
        bridge = _bridge(data, _closed_base(), TELEGRAM_MIRROR_MODE="telegram_only")
        await bridge.publish("chat.outbound", {"text": "From the phone, waits.", "transport": phone})
        monkeypatch.setattr(bridge.module("telegram_outbox"), "MAX_ACTIVE", 1)
        await bridge.publish(*web)
        assert len(bridge.records()) == 1 and bridge.outbox.summary()["not_sent"] == 0
        assert _not_delivered(bridge) == "No recent delivery issues."
        await bridge.publish("chat.photo", {**web[1], "transport": phone})
        assert bridge.records()[-1]["reason"] == "full" and bridge.outbox.summary()["not_sent"] == 1
    _run(scenario)


def test_a_quiz_fact_without_a_card_waits_in_no_queue_for_no_token_nor_a_day(tmp_path):
    """With no bot token, a lifecycle fact for a question that has no Telegram card is not journaled (a day later it
    would be listed as a question update not sent): it never meets the journal, whether the queue is free or waits,
    and the card created later in this load still takes it."""
    async def scenario():
        bridge = _Bridge(tmp_path / "state", _closed_base(), token="")
        fact = {"task_id": "task-1", "quiz_id": "q1", "state": "answered", "answered_index": 1}
        await bridge.publish("chat.quiz_state", fact)  # the queue is free
        assert bridge.records() == [] and bridge.files() == []
        await bridge.publish("chat.outbound", {"text": "Waits for a token.", "transport": {}})
        await bridge.publish("chat.quiz_state", {**fact, "question_delivery": {"kind": "answer", "state": "delivered"}})
        assert [r["topic"] for r in bridge.records()] == ["outbound"] and len(bridge.files()) == 2  # the queue waits
        bridge.now += 24 * 3600 + 5
        await bridge.drain()  # still no token: the reply ends unsent; the fact left nothing to end
        assert [(r["topic"], r["reason"]) for r in bridge.records()] == [("outbound", "expired")]
        assert "Question update" not in _not_delivered(bridge)
        bridge.api.granted["TELEGRAM_BOT_TOKEN"] = _TOKEN
        assert bridge.plugin.binding_for(bridge.api).read_now(bridge.api)
        async with _Origin() as origin:
            bridge.serve(origin.base)
            await bridge.publish("chat.quiz", {"task_id": "task-1", "quiz_id": "q1", "question": "Which db?",
                                               "options": [{"label": "sqlite"}, {"label": "postgres"}], "transport": {}})
        assert origin.methods() == ["sendMessage", "editMessageText"]
        assert origin.sent("editMessageText")[0]["text"].endswith("\nAnswered: 2. postgres")  # delivered: no line
        await bridge.stop()
    _run(scenario)


@pytest.mark.parametrize("case", ["answered-reload", "closed-reload", "missed-delivery", "unreadable-record"])
def test_a_card_sent_after_its_question_settled_shows_the_hosts_record(tmp_path, monkeypatch, case):
    """The bridge keeps no durable copy of early facts of its own beside the host's record of the question (one would
    be blind to what never reached this load). The card waits (Telegram unreachable) while the owner answers or closes the
    question in the app; the host records it, then publishes the fact, which this load keeps for the card. Sent at
    last, after a reload, or after a later fact never reached this load, the card reads the host's record, the
    newest, and settles on it, whatever stale fact this load kept; a record that cannot be read leaves the fact kept
    here, never an open card."""
    from ouroboros import owner_quiz, task_store

    data = tmp_path / "data"

    async def scenario():
        produce = _Producer(monkeypatch, data)
        owner_quiz.record_asked(data, "work-1", quiz_id="q1", question="Which db?", options=["sqlite", "postgres"],
                                assumption=_MEANWHILE, chat_id=1)
        ok, why = produce.bus.send_quiz(1, "q1", "Which db?", [{"label": "sqlite"}, {"label": "postgres"}],
                                        assumption=_MEANWHILE, task_id="work-1")
        assert ok, why
        asked = produce._one("chat.quiz")
        bridge = _bridge(data, _closed_base())
        await bridge.publish(*asked)  # the card waits
        if case == "closed-reload":
            assert owner_quiz.record_closed(data, "work-1", quiz_id="q1", reason="Plan changed.", actor="owner",
                                            request_id="close-1")["ok"]
            produce.bus.send_quiz_state("q1", "work-1", "closed", closed_reason="Plan changed.")
        else:
            assert owner_quiz.record_answered(data, "work-1", quiz_id="q1", option_index=1, request_id="web-1")
            block = owner_quiz.quiz_states(data, "work-1")["q1"]
            produce.bus.send_quiz_state("q1", "work-1", "answered", answered_index=1,
                                        question_delivery=owner_quiz.question_delivery_projection(block))
        fact = produce._one("chat.quiz_state")
        await bridge.publish(*fact)  # no card yet: kept for it
        if case == "missed-delivery":  # the answer reaches its task; that fact never reaches this load
            assert owner_quiz.mark_discharged(data, "work-1", "q1")
        if case.endswith("reload"):
            await bridge.stop()
            bridge = _bridge(data, _closed_base())  # a reload: what that load kept in memory is gone
            other = "answered" if case == "closed-reload" else "closed"
            await bridge.publish("chat.quiz_state", {**fact[1], "state": other, "answered_index": 0,
                                                     "closed_reason": "A stale fact."})
        if case == "unreadable-record":
            def unreadable(*_args, **_kwargs):
                raise OSError("the task store cannot be read")

            monkeypatch.setattr(task_store, "read_document", unreadable)
        async with _Origin() as origin:
            bridge.serve(origin.base)
            await bridge.drain()
        assert origin.methods() == ["sendMessage", "editMessageText"]
        edit = origin.sent("editMessageText")[0]["text"]
        if case == "closed-reload":
            assert edit.endswith("\nQuestion closed.\nPlan changed.")
        elif case == "missed-delivery":
            assert edit.endswith("\nAnswered: 2. postgres")  # the delivery the host recorded since: no stale line
        else:  # the host's record, or the fact kept here when that record cannot be read
            assert edit.endswith("\nAnswered: 2. postgres\nAnswer recorded · delivery pending")
        assert bridge.records() == []
        await bridge.stop()
    _run(scenario)


def test_the_queues_head_never_runs_while_its_own_decision_waits_for_its_card(tmp_path):
    """Pins ``<=`` in ``Outbox._next`` (``<`` passed every other test of the queue): a question saved behind
    a waiting progress note is decided after it is saved, and that decision waits on its card's lock (an edit of the
    card in flight). The note ahead is delivered meanwhile, so the question is the queue's due head; it still does
    not run while its decision waits: the queue's lock stays free and its handler's first real step, ending the
    Silent progress run, has not happened. Once decided, it runs once and ends that run."""
    async def scenario():
        bridge = _Bridge(tmp_path / "state", _closed_base(), TELEGRAM_SILENT_MODE="on", TELEGRAM_MIRROR_PROGRESS="on")
        await bridge.publish("chat.outbound", {"text": "Step one.", "is_progress": True, "task_id": "t1",
                                               "transport": {}})
        quiz, state = bridge.module("telegram_quiz"), bridge.module("telegram_state")
        lock = quiz.card_lock(quiz.mint_token("task-1", "q1"))
        await lock.acquire()
        deciding = asyncio.ensure_future(bridge.publish("chat.quiz", {
            "task_id": "task-1", "quiz_id": "q1", "question": "Keep the cache?", "options": [], "transport": {}}))
        await _until(lambda: len(bridge.records()) == 2)  # saved behind the note; its decision now waits
        async with _Origin() as origin:
            bridge.serve(origin.base)
            bridge.start()
            bridge.now += 60
            bridge.outbox.wake()
            await _until(lambda: bridge.outbox.summary()["delivered"] == 1)
            tracked = state._get_silent_msg(bridge.api, _BOT, 42)
            assert tracked == 501  # the note is Silent mode's progress run
            await asyncio.sleep(0.3)  # the sender meets the due head
            assert origin.methods() == ["sendMessage"] and not deciding.done()
            assert not bridge.outbox._fifo().locked() and [r["state"] for r in bridge.records()] == ["pending"]
            assert state._get_silent_msg(bridge.api, _BOT, 42) == tracked
            lock.release()
            await deciding
            await bridge.drain()
        assert origin.methods() == ["sendMessage", "sendMessage"] and "Keep the cache?" in origin.sent()[1]["text"]
        assert state._get_silent_msg(bridge.api, _BOT, 42) == 0 and bridge.records() == []
        await bridge.stop()
    _run(scenario)


def test_a_secret_requests_card_is_canonical_only_as_its_own_records_hint(tmp_path, monkeypatch):
    """``showSecretHitl`` records the host-authored question, and ``LocalChatBridge.send_quiz`` publishes in its
    place where to enter the secret (``transport_hint`` of the record's own request). That card has its home in
    the record whatever else is kept; another request, the bare question, a card that names no request, or an
    ordinary question that claims one, does not."""
    from ouroboros import config as cfg
    from ouroboros import owner_quiz
    from ouroboros import secret_requests as sr
    from tests.test_secret_hitl import _write_skill

    data = tmp_path / "data"
    monkeypatch.setattr(cfg, "get_skills_repo_path", lambda: "")
    _write_skill(data)

    async def scenario():
        produce = _Producer(monkeypatch, data)
        request = sr.resolve_secret_request(data, "fakeplug", ["botToken"])
        owner_quiz.record_asked(data, "work-1", quiz_id="qs", question=sr.question_text(request), options=[],
                                wait_for_answer=True, chat_id=1, secret_request=request)
        ok, why = produce.bus.send_quiz(1, "qs", sr.question_text(request), [], task_id="work-1", wait_for_answer=True)
        assert ok, why
        secret = produce._one("chat.quiz")[1]
        assert secret["question"] == sr.transport_hint(request) and secret["secret_request"] == request
        owner_quiz.record_asked(data, "work-1", quiz_id="q1", question="Which db?", options=["sqlite", "postgres"],
                                assumption=_MEANWHILE, chat_id=1)
        ok, why = produce.bus.send_quiz(1, "q1", "Which db?", [{"label": "sqlite"}, {"label": "postgres"}],
                                        assumption=_MEANWHILE, task_id="work-1")
        assert ok, why
        ordinary = produce._one("chat.quiz")[1]
        cases = {
            "secret card": secret,
            "another request": {**secret, "secret_request": {**request, "plugin_name": "otherplug"}},
            "the bare question": {**secret, "question": sr.question_text(request)},
            "no request named": {key: value for key, value in secret.items() if key != "secret_request"},
            "ordinary card": ordinary,
            "ordinary, reworded": {**ordinary, "question": "Which database?"},
            "ordinary claiming a request": {**ordinary, "secret_request": request, "question": secret["question"]},
        }
        copies = _bridge(data, _closed_base()).module("telegram_copies")
        assert {name: copies.classify(data, "quiz", event)[0] for name, event in cases.items()} == {
            "secret card": "home", "another request": "only", "the bare question": "only", "no request named": "only",
            "ordinary card": "home", "ordinary, reworded": "only", "ordinary claiming a request": "only"}
        async with _Origin(["403"] * 3) as origin:
            bridge = _bridge(data, origin.base)
            await bridge.publish(*produce.photo(b"BM" + _blob(20_000, 94), mime="image/bmp"))  # a copy kept first
            await bridge.publish("chat.quiz", secret)
            await bridge.publish("chat.quiz", cases["another request"])
        assert [(r["topic"], r["copy"], bool(r.get("held"))) for r in bridge.records()] == [
            ("photo", "only", True), ("quiz", "home", False), ("quiz", "only", True)]
        assert "Enter it in the protected form" in origin.sent()[0]["text"]
    _run(scenario)
