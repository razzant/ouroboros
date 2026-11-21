"""A host notice: one phrase, its next step and its details (DESIGN principle 7, ``ouroboros/host_notice.py``).

The producer writes the phrase as the row's text and the typed notice beside it; the row keeps the notice through
persistence and history; readers are sent one line, the phrase and its step in the install's language; the mind reads
the step and the raw details beside the phrase, so a raw error it read in an older row's words stays in its view.
"""

from __future__ import annotations

import asyncio
import json
import typing
from types import SimpleNamespace

import pytest

from ouroboros import host_notice
from ouroboros.dialogue_provenance import dialogue_text
from ouroboros.gateway.history import make_chat_history_endpoint
from ouroboros.gateway.history_contracts import ChatHistoryMessage
from ouroboros.gateway.schema import validate_ingress
from supervisor import message_bus
from tests.test_telegram_i18n import _seed

RAW = "PhysicalAttemptPreparationFailed: StoreBusy: current store lock unavailable: /Users/x/packs_publication.lock"


def test_a_notice_carries_a_known_step_and_its_raw_details_and_refuses_words_nobody_wrote():
    assert host_notice.notice("task_exception", details=RAW) == {"code": "task_exception", "details": RAW}
    assert host_notice.notice("budget_exhausted", step="raise_total_budget", details="  ") == {
        "code": "budget_exhausted", "step": "raise_total_budget"}
    with pytest.raises(ValueError):
        host_notice.notice("x", step="press_the_magic_button")
    assert host_notice.stored({"notice": {"code": "c", "details": "d", "line": "the reader's line"}}) == {
        "code": "c", "details": "d"}, "a row persists no reader's line"
    assert host_notice.notice_of({"notice": {"details": "no code"}}) == {}


def test_the_raw_error_the_mind_read_in_an_older_row_stays_in_its_view():
    """BIBLE P1: the old row said the exception in its words; the notice row says a phrase, and the mind reads the
    same exception beside it (``dialogue_text``, the one reader of history, search and memory rows)."""
    old = {"direction": "system", "role": "system", "text": f"⚠️ Error during processing: {RAW}"}
    notice = {"direction": "system", "role": "system", "text": "This task stopped on an unexpected error.",
              **host_notice.fields("task_exception", details=RAW)}
    assert RAW in dialogue_text(old) and RAW in dialogue_text(notice)
    assert dialogue_text(notice) == f"This task stopped on an unexpected error.\n[Details: {RAW}]"
    stepped = {"text": "The total budget is spent, so this turn did not start.",
               **host_notice.fields("budget_exhausted", step="raise_total_budget")}
    assert dialogue_text(stepped).endswith(f"\n[Step: {host_notice.STEPS['raise_total_budget']}]")
    from ouroboros.memory import Memory

    line = Memory._format_chat_line({**notice, "ts": "2026-11-21T12:00:00Z"}, compact=True)
    assert line.startswith("📋 12:00 [system] This task stopped") and RAW in line
    assert RAW in Memory.summarize_progress(SimpleNamespace(), [{**notice, "ts": "2026-11-21T12:00:00Z"}])


def test_the_one_line_joins_the_phrase_and_its_step_each_translated_first(tmp_path, monkeypatch):
    phrase = "The total budget is spent, so this turn did not start."
    row = {"text": phrase, "role": "system", **host_notice.fields("budget_exhausted", step="raise_total_budget")}
    assert host_notice.notice_line(row, "") == f"{phrase} {host_notice.STEPS['raise_total_budget']}"
    _seed(tmp_path, "ru", {phrase: {"text": "Бюджет исчерпан."},
                           "code:notice.step.raise_total_budget": {"text": "Поднимите его."}})
    assert host_notice.notice_line(row, "ru", drive_root=tmp_path) == "Бюджет исчерпан. Поднимите его."
    # A part the memory lacks stays English; the other part is still translated.
    _seed(tmp_path, "de", {"code:notice.step.raise_total_budget": {"text": "Erhöhen."}})
    assert host_notice.notice_line(row, "de", drive_root=tmp_path) == f"{phrase} Erhöhen."
    # The technical pause is translated by its text, as Telegram did before notices; other rows keep their words.
    pause = {"text": "Paused for its authority.", "role": "system", "system_type": "task_pause_notice",
             "pause_reason": "authority"}
    _seed(tmp_path, "ru", {pause["text"]: {"text": "Пауза."}, "Some host words.": {"text": "Нет."}})
    assert host_notice.notice_line(pause, "ru", drive_root=tmp_path) == "Пауза."
    assert host_notice.notice_line({**pause, "pause_reason": "owner"}, "ru", drive_root=tmp_path) == pause["text"]
    assert host_notice.notice_line({"text": "Some host words.", "role": "system"}, "ru", drive_root=tmp_path) == "Some host words."


@pytest.fixture
def chat(tmp_path, monkeypatch):
    """The real bridge over a disposable data root; frames are what a socket would receive."""
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "progress.jsonl").touch()
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    events: list = []
    monkeypatch.setattr(message_bus, "publish_event", lambda _kind, event: events.append(event))
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 7})
    bridge = message_bus.LocalChatBridge({})
    frames: list = []
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    return SimpleNamespace(frames=frames, events=events, root=tmp_path)


def test_a_notice_row_keeps_its_notice_and_every_reader_is_sent_its_line(chat):
    phrase = "This turn stopped on an unexpected error."
    message_bus.send_with_budget(1, phrase, task_id="t1", role="system", system_type="task_error",
                                 progress_meta=host_notice.fields("task_exception", details=RAW))
    message_bus.send_with_budget(1, "No conversation carrier could start, so this turn continues inside the server "
                                    "process.", is_progress=True, task_id="t1", role="system", system_type="worker_failure",
                                 progress_meta=host_notice.fields("conversation_degraded_executor", details="boot"))
    live = [frame for frame in chat.frames if frame.get("notice")]
    assert [frame["notice"]["line"] for frame in live] == [phrase, live[1]["content"]]
    assert live[0]["notice"]["details"] == RAW and live[0]["content"] == phrase, "the raw error is not its words"
    assert [event["notice"]["line"] for event in chat.events if event.get("notice")] == [phrase, live[1]["content"]]
    response = asyncio.run(make_chat_history_endpoint(chat.root)(SimpleNamespace(query_params={})))
    rows = [row for row in json.loads(response.body.decode("utf-8"))["messages"] if row.get("notice")]
    assert [(row["text"], row["notice"]) for row in rows] == [(frame["content"], frame["notice"]) for frame in live]
    for row in rows:
        assert not validate_ingress(row, ChatHistoryMessage), validate_ingress(row, ChatHistoryMessage)
    assert "notice" in typing.get_type_hints(ChatHistoryMessage)


def test_the_persisted_row_keeps_the_notice_without_any_readers_line(tmp_path):
    from supervisor.message_ingress import _chat_record

    _root, record = _chat_record("system", 1, 7, "A phrase.", drive_root=tmp_path, message_meta={
        "notice": {"code": "c", "step": "restart", "details": "raw", "line": "A phrase. Restart Ouroboros to recover."}})
    assert record["notice"] == {"code": "c", "step": "restart", "details": "raw"}
    _root, plain = _chat_record("system", 1, 7, "A phrase.", drive_root=tmp_path, message_meta={})
    assert "notice" not in plain

