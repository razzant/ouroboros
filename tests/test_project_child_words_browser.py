"""Each stored child-speech occurrence stays one card line through live delivery and replay."""
from tests._dialogue import advance_room_revision

import json
import os
from pathlib import Path

import pytest

from ouroboros import dialogue_store
from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data
from tests.ui_chat_viewport_smoke import _SETTLE_RESTORE_FRAMES, _emit_ws_frame
from tests.test_project_unread_read_band_browser import PANEL, _open_room, _seed_room, _wait_for

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

# The words a child's card shows, one entry per line it draws, opened down to that card.
_CHILD_WORDS = """(messages, taskId) => {
    const card = messages.querySelector(`.chat-live-card[data-task-id="${taskId}"]`);
    if (!card) return null;
    for (let node = card; node; node = node.parentElement?.closest('.chat-live-card')) {
        if (node.dataset.expanded !== '1') node.querySelector('[data-live-disclosure]').click();
    }
    card.scrollIntoView({block: 'center'});
    return [...card.querySelectorAll('.chat-live-line')].filter(line => line.closest('.chat-live-card') === card)
        .map(line => ({text: line.querySelector('.chat-live-line-body')?.innerText.trim() || '',
                       repeat: line.querySelector('.chat-live-line-repeat:not([hidden])')?.innerText.trim() || ''}))
        .filter(line => line.text.startsWith('CHILD_WORDS'));
}"""


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_childs_words_said_again_are_another_line_of_its_card_live_and_after_a_reload(direct_server_with_data,
                                                                                         engine):
    """A child's own words are lines of its card, one each time it spoke, the same words said again
    included: as the bridge delivers them, once a read adopts them, past its final, and on reload."""
    from playwright.sync_api import sync_playwright

    from ouroboros.task_results import write_task_result

    data = direct_server_with_data["data_dir"]
    at = lambda minute: f"2026-09-28T10:{minute:02d}:00Z"  # noqa: E731
    lineage = {"delegation_role": "subagent", "parent_task_id": "root-words", "root_task_id": "root-words",
               "subagent_role": "researcher"}
    meta = {**lineage, "subagent_task_id": "kid-words", "narration": False}
    rows = [{"ts": at(minute), "direction": "out", "text": f"Reply {minute}"} for minute in range(3)]
    rows.append({"ts": at(5), "direction": "out", "text": "ROOT_ANSWER: the scout is on it.", "task_id": "root-words"})
    chat_id, _ = _seed_room(data, "words-room", "Words room", rows)
    write_task_result(data, "kid-words", "running", **lineage)
    progress = lambda ts, task_id, text, **fields: {  # noqa: E731
        "ts": ts, "type": "send_message", "task_id": task_id, "is_progress": True, "direction": "out",
        "chat_id": chat_id, "text": text, "content": text, "format": "markdown", "role": "assistant",
        "system_type": "", **fields}
    for row in (progress(at(4), "root-words", "Reading the brief.", narration=True),
                progress(at(6), "kid-words", "Listing the repository root.", **meta)):
        dialogue_store.append(data, "progress", row)
    said = ["CHILD_WORDS_1: halfway through.", "CHILD_WORDS_2: found the README.",
            "CHILD_WORDS_2: found the README.", "CHILD_WORDS_1: halfway through."]
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", data.parent))
    evidence.mkdir(parents=True, exist_ok=True)
    acks = []
    with sync_playwright() as pw:
        browser, page, row = _open_room(pw, engine, direct_server_with_data["url"], "words-room", acks)
        try:
            messages = page.locator(PANEL)
            messages.locator('.chat-live-card[data-task-id="kid-words"]').wait_for(state="attached")
            page.evaluate(_SETTLE_RESTORE_FRAMES)

            def speak(index, text, **fields):
                """message_bus.send_with_budget: the stored row, then the bridge's frame of it."""
                ts = f"2026-09-28T10:20:{index:02d}.{123456 + index:06d}+00:00"
                dialogue_store.append(data, "chat", {"ts": ts, "direction": "out", "chat_id": chat_id, "text": text,
                                                     "format": "markdown", "task_id": "kid-words", **fields, **meta})
                frame = {"type": "chat", "role": "assistant", "content": text, "markdown": True, "is_progress": False,
                         "ts": ts, "task_id": "kid-words", "chat_id": chat_id, "project_thread": True, **meta,
                         **({"system_type": fields["type"]} if "type" in fields else {})}
                _emit_ws_frame(page, frame)
                return frame

            words = lambda: messages.evaluate(_CHILD_WORDS, "kid-words")  # noqa: E731
            frames = [speak(index, text, type="proactive_message") for index, text in enumerate(said)]
            assert _wait_for(page, lambda: len(words() or []) >= len(said)), ("each time it spoke is a line", words())
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            live = words()
            page.screenshot(path=str(evidence / f"read-band-{engine}-child-words-live.png"))
            assert [line["text"] for line in live] == said and not any(line["repeat"] for line in live), live

            _emit_ws_frame(page, frames[-1])
            page.wait_for_timeout(300)
            assert words() == live, "the same frame delivered again is no new words"
            speak(len(said), "CHILD_FINAL: root listed.")
            write_task_result(data, "kid-words", "completed", **lineage)
            card = messages.locator('.chat-live-card[data-task-id="kid-words"]')
            assert _wait_for(page, lambda: card.evaluate("n => n.dataset.finished") == "1"), "its final concludes it"
            assert words() == live, "its final keeps every line it spoke"

            # A counted reply moves the room's revision: the read that follows adopts the live lines.
            dialogue_store.append(data, "chat", {"ts": at(30), "direction": "out", "chat_id": chat_id,
                                                 "text": "Reply after the child"})
            advance_room_revision(data, chat_id=chat_id)
            _emit_ws_frame(page, {"type": "projects_changed"})
            messages.locator(".chat-bubble").filter(has_text="Reply after the child").wait_for(state="attached")
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            page.wait_for_timeout(300)
            adopted = words()
            adopted_ids = messages.evaluate("""m => [...m.querySelectorAll(
                '.chat-live-card[data-task-id="kid-words"] .chat-live-line[data-history-id]')].length""")
            assert adopted == live and adopted_ids >= len(said), ("each stored row adopts its own line", adopted)

            page.reload(wait_until="domcontentloaded")
            page.wait_for_function("() => window.__testSockets?.some(s => s.readyState === 1)")
            row.evaluate("el => el.click()")
            messages.locator('.chat-live-card[data-task-id="kid-words"]').wait_for(state="attached")
            messages.locator(".chat-bubble").filter(has_text="Reply after the child").wait_for(state="attached")
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            reloaded = words()
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            concluded = card.evaluate("n => n.dataset.finished")
            page.screenshot(path=str(evidence / f"read-band-{engine}-child-words-reload.png"))
            (evidence / f"read-band-{engine}-child-words.json").write_text(
                json.dumps({"live": live, "adopted": adopted, "reloaded": reloaded}, indent=2), encoding="utf-8")
            assert reloaded == live, ("a reload draws the lines drawn live", live, reloaded)
            assert concluded == "1", "the stored final concludes the child beside its words"
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
@pytest.mark.parametrize("ordering", ["history-first", "live-history-redelivery"])
def test_running_child_occurrences_survive_history_live_ordering(direct_server_with_data, monkeypatch,
                                                                engine, ordering):
    """Store through the bridge producer, read the actual endpoint, then deliver its held frames."""
    from playwright.sync_api import sync_playwright

    from ouroboros.task_results import write_task_result
    from supervisor import message_bus

    data = direct_server_with_data["data_dir"]
    chat_id, _ = _seed_room(data, "occurrence-room", "Occurrence room", [
        {"ts": "2026-10-11T12:00:00+00:00", "direction": "out", "text": "ROOT_ANSWER: the scout is on it.",
         "task_id": "root-occurrence"}])
    lineage = {"delegation_role": "subagent", "parent_task_id": "root-occurrence",
               "root_task_id": "root-occurrence", "subagent_role": "researcher"}
    meta = {**lineage, "subagent_task_id": "kid-occurrence", "narration": False}
    write_task_result(data, "kid-occurrence", "running", **lineage)
    frames = []
    bridge = message_bus.LocalChatBridge({})
    bridge._broadcast_fn = frames.append  # The WebSocket scheduler may deliver after history reads storage.
    monkeypatch.setattr(message_bus, "DATA_DIR", data)
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 7, "session_id": "occurrence"})
    monkeypatch.setattr(message_bus, "publish_event", lambda *_a, **_k: None)

    def produce(text, second, **fields):
        message_bus.send_with_budget(chat_id, text, task_id="kid-occurrence", fmt="markdown",
                                     progress_meta=meta, ts=f"2026-10-11T12:00:{second:02d}.123456+00:00", **fields)
        return frames[-1]

    produce("Listing the repository root.", 1, is_progress=True)
    said = ["CHILD_WORDS_1: halfway through.", "CHILD_WORDS_2: found the README.",
            "CHILD_WORDS_2: found the README.", "CHILD_WORDS_1: halfway through."]
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", data.parent))
    evidence.mkdir(parents=True, exist_ok=True)
    prefix = f"occurrence-{engine}-{ordering}"
    observed = {"ordering": ordering, "engine": engine, "console_errors": [], "checkpoints": {}}
    with sync_playwright() as pw:
        browser, page, _ = _open_room(pw, engine, direct_server_with_data["url"], "occurrence-room", [])
        page.context.tracing.start(screenshots=True, snapshots=True)
        page.on("pageerror", lambda error: observed["console_errors"].append(str(error)))
        try:
            messages = page.locator(PANEL)
            card = messages.locator('.chat-live-card[data-task-id="kid-occurrence"]')
            card.wait_for(state="attached")
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            words = lambda: messages.evaluate(_CHILD_WORDS, "kid-occurrence")  # noqa: E731

            def snapshot(label):
                observed["checkpoints"][label] = {
                    "words": words(), "finished": card.get_attribute("data-finished"),
                    "source_ids": card.locator('.chat-live-line[data-history-id]').evaluate_all(
                        "lines => lines.map(line => line.dataset.historyId)"),
                }
                page.evaluate(_SETTLE_RESTORE_FRAMES)
                page.screenshot(path=str(evidence / f"{prefix}-{label}.png"))
                (evidence / f"{prefix}.json").write_text(json.dumps(observed, indent=2), encoding="utf-8")
                return observed["checkpoints"][label]

            def read_stored(label, second):
                # A standalone arrival makes the ordinary Project refresh read the stored child rows.
                message_bus.send_with_budget(chat_id, label, ts=f"2026-10-11T12:00:{second:02d}+00:00")
                _emit_ws_frame(page, {"type": "projects_changed"})
                messages.locator('.chat-bubble').filter(has_text=label).wait_for(state="attached")
                page.evaluate(_SETTLE_RESTORE_FRAMES)

            speech = [produce(text, index + 2, system_type="proactive_message") for index, text in enumerate(said)]
            if ordering == "live-history-redelivery":
                for frame in speech:
                    _emit_ws_frame(page, frame)
                assert _wait_for(page, lambda: len(words() or []) == len(said)), words()
            read_stored("Read while the child is running", 10)
            response = page.request.get(f'{direct_server_with_data["url"]}/api/chat/history?chat_id={chat_id}')
            assert response.ok
            observed["running_history"] = response.json()
            observed["live_frames"] = speech
            source_ids = [row["history_id"] for row in observed["running_history"]["messages"]
                          if row.get("system_type") == "proactive_message"]
            assert len(set(source_ids)) == len(said)
            adopted = snapshot("adopted")
            assert adopted["finished"] == "0"
            assert [line["text"] for line in adopted["words"]] == said
            assert set(source_ids) <= set(adopted["source_ids"])
            for _ in range(2):
                for frame in speech:
                    _emit_ws_frame(page, frame)
            page.wait_for_timeout(300)
            delivered = snapshot("redelivered")
            assert delivered == adopted, ("one line per occurrence after delayed delivery or redelivery", observed)
            assert not any(line["repeat"] for line in delivered["words"])

            _emit_ws_frame(page, produce("CHILD_FINAL: root listed.", 20))
            write_task_result(data, "kid-occurrence", "completed", **lineage)
            assert _wait_for(page, lambda: card.get_attribute("data-finished") == "1")
            assert card.locator('.chat-live-line-body').filter(has_text="CHILD_FINAL").count() == 1
            read_stored("Read after the child concluded", 30)
            for frame in speech:
                _emit_ws_frame(page, frame)
            page.wait_for_timeout(300)
            concluded = snapshot("concluded")
            assert concluded["words"] == adopted["words"] and concluded["finished"] == "1"
            assert card.locator('.chat-live-line-body').filter(has_text="CHILD_FINAL").count() == 1
            assert messages.locator('.chat-bubble').filter(has_text="CHILD_WORDS").count() == 0
            assert observed["console_errors"] == []
        finally:
            (evidence / f"{prefix}.json").write_text(json.dumps(observed, indent=2), encoding="utf-8")
            page.context.tracing.stop(path=str(evidence / f"{prefix}-trace.zip"))
            browser.close()
