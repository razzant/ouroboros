"""Notes through the real server, its scheduler tick and the browser (ui_browser lane).

A running supervisor shows a due note in Main as one System row whose host signature
stands on its own line above the mind's words; reopening replays it from history
without a banner; Activity names a waiting note by its words; and with notifications
on, a note that fires while the page is open rings once in the existing messages
category, titled by its author. Two reconnects retain that live row and its text
selection while history gives it a durable identity, without ringing again.
No model is called for any of it.
"""
from __future__ import annotations

from tests import _dialogue as dialogue_test

import json
import os
import re
import time
from pathlib import Path

import pytest

from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401 - pytest fixture
from tests.ui_chat_viewport_smoke import wait_drawer_closed
from tests.test_reconnect_feed_browser import HISTORY_LANDED

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

# Midday UTC instants keep "Dec 31" / "Jan 1" stable in any server zone from UTC-11 to UTC+11.
DUE = "Call mother about Sunday."
LATER = "Water the plants."
LIVE = "The kettle is on."
BUBBLE = '.chat-bubble[data-system-type="reminder"]'
# Narrow captures wait for the drawer to slide out and every CSS transition to end.
SETTLED = """() => Promise.all(document.getAnimations()
    .filter(animation => typeof CSSTransition === 'function' && animation instanceof CSSTransition)
    .map(animation => animation.finished.catch(() => null)))
    .then(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))))"""
PREFS = {"enabled": True, "needs_answer": True, "task_done": True, "important": True,
         "main_reply": False, "sound": False, "show_text": True}


def _note(root: Path, schedule_id: str, text: str, run_at: str) -> None:
    from supervisor.queue import upsert_scheduled_task

    record = {"id": schedule_id, "name": "Reminder of task t-ui", "description": text, "kind": "notify",
              "source": "task_followup", "enabled": True, "timezone": "",
              "trigger": {"type": "once", "run_at": run_at},
              "notification": {"text": text, "set_at": "1999-12-31T12:00:00+00:00"}}
    upsert_scheduled_task(record, drive_root=root, host_followup={
        "followup_origin": {"task_id": "t-ui", "root_task_id": "t-ui"},
        "followup_relation": {"kind": "independent", "declared_by": "t-ui", "revision": "r"}})


def _wait_for_row(request, url: str, text: str) -> None:
    """The tick runs on the supervisor's own cadence; wait for its durable row, bounded."""
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        messages = request.get(url + "/api/chat/history").json().get("messages") or []
        if any(m.get("system_type") == "reminder" and m.get("text", "").endswith(text) for m in messages):
            return
        time.sleep(0.5)
    raise AssertionError(f"no reminder row for {text!r} within 90 s")


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_notes_reach_main_activity_and_the_live_banner(direct_server_with_data, engine):  # noqa: F811 - pytest fixture
    from playwright.sync_api import sync_playwright

    server = direct_server_with_data
    root, url = server["data_dir"], server["url"]
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR") or root / "ui-evidence") / f"owner-notes-{engine}"
    evidence.mkdir(parents=True, exist_ok=True)
    server["stop_server"]()
    _note(root, "note-due", DUE, "2000-01-01T12:00:00+00:00")  # due for decades: the first tick shows it
    _note(root, "note-later", LATER, "2999-01-01T09:00:00+00:00")
    server["start_server"]()

    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 860})
            _wait_for_row(page.request, url, DUE)  # shown before the page exists: history only
            page.add_init_script(f"localStorage.setItem('ouroboros.notifications', {json.dumps(json.dumps(PREFS))})")
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            page.locator('[data-nav-page="chat"]').click()
            page.wait_for_selector('#chat-messages[data-history-hydrated="true"]', state="attached", timeout=30_000)
            bubble = page.locator(f"#chat-messages {BUBBLE}").filter(has_text=DUE)
            bubble.wait_for(state="visible", timeout=30_000)
            assert bubble.locator(".sender").inner_text().strip() == "System"
            message = bubble.locator(".message")
            signature, words = message.inner_text().split("\n", 1)
            assert signature.startswith("Reminder · Ouroboros · written Dec 31 ") and " · for Jan 1 " in signature
            assert " · delivered " in signature and signature.endswith(")"), signature
            assert words.strip() == DUE
            assert message.evaluate("node => getComputedStyle(node).whiteSpace") == "pre-wrap"
            assert page.locator("#toast-stack .toast").count() == 0, "history replay never rings"
            page.screenshot(path=str(evidence / "main-history-desktop.png"))
            page.set_viewport_size({"width": 390, "height": 844})
            page.evaluate(SETTLED)
            box = bubble.bounding_box()
            assert box and box["x"] >= 0 and box["x"] + box["width"] <= 390
            page.screenshot(path=str(evidence / "main-history-narrow.png"))
            page.set_viewport_size({"width": 1280, "height": 860})

            # A note due while the page is open arrives live: one row, one banner in the
            # existing messages category, titled by its author, the words below the signature.
            _note(root, "note-live", LIVE, "2000-01-02T00:00:00+00:00")
            page.locator(f"#chat-messages {BUBBLE}").filter(has_text=LIVE).wait_for(state="visible", timeout=90_000)
            toast = page.locator("#toast-stack .toast").filter(has_text="Reminder from Ouroboros")
            toast.wait_for(state="visible", timeout=10_000)
            assert toast.inner_text().strip().endswith(LIVE)
            assert page.locator("#toast-stack .toast").filter(has_text="Reminder").count() == 1
            page.screenshot(path=str(evidence / "main-live-banner.png"))

            live = page.locator(f"#chat-messages {BUBBLE}").filter(has_text=LIVE)
            live.evaluate("""(node, words) => {
                window.__liveReminder = node;
                const body = node.querySelector('.message').firstChild;
                const start = body.textContent.lastIndexOf(words);
                const range = document.createRange();
                range.setStart(body, start); range.setEnd(body, start + words.length);
                const selection = getSelection();
                selection.removeAllRanges(); selection.addRange(range);
                window.__reminderSelection = { anchorNode: selection.anchorNode,
                    anchorOffset: selection.anchorOffset, focusNode: selection.focusNode,
                    focusOffset: selection.focusOffset };
                window.__extraReminderBanners = 0;
                window.__reminderBannerObserver = new MutationObserver(records => {
                    for (const record of records) for (const added of record.addedNodes) {
                        if (added.nodeType === 1 && added.matches('.toast')
                            && added.textContent.includes('Reminder from Ouroboros')) {
                            window.__extraReminderBanners += 1;
                        }
                    }
                });
                window.__reminderBannerObserver.observe(document.querySelector('#toast-stack'), { childList: true });
            }""", LIVE)
            assert page.evaluate("getSelection().toString()") == LIVE
            # Live delivery may precede its durable append; this scenario tests adoption of a saved row.
            _wait_for_row(page.request, url, LIVE)
            for attempt in (1, 2):
                page.wait_for_function("() => window.__ouroWs.ws?.readyState === WebSocket.OPEN"
                                       " && Boolean(window.__ouroWs._lastSha)", timeout=30_000)
                page.evaluate(HISTORY_LANDED)
                with page.expect_response(lambda response: "/api/chat/history" in response.url
                                          and response.request.method == "GET", timeout=30_000) as history:
                    page.evaluate("window.__ouroWs.ws.close()")
                assert history.value.status == 200
                rows = history.value.json()["messages"]
                row = next(row for row in rows if row.get("system_type") == "reminder"
                           and row.get("text", "").endswith(LIVE))
                page.wait_for_function("() => window.historyLanded > 0", timeout=30_000)
                assert page.locator('[data-system-type="reconnect"]').count() == 0
                page.evaluate(SETTLED)
                page.screenshot(path=str(evidence / f"main-live-reconnect-{attempt}.png"))
                assert live.count() == 1, "reconnect history must adopt the live reminder"
                assert live.evaluate("node => node === window.__liveReminder")
                assert live.get_attribute("data-history-id") == row["history_id"]
                assert live.get_attribute("data-history-source") == row["history_position"]["source"]
                assert live.get_attribute("data-history-offset") == str(row["history_position"]["offset"])
                assert page.evaluate("""() => {
                    const current = getSelection(), saved = window.__reminderSelection;
                    return ['anchorNode', 'anchorOffset', 'focusNode', 'focusOffset']
                        .every(key => current[key] === saved[key]);
                }""")
                assert page.evaluate("getSelection().toString()") == LIVE
                assert page.evaluate("window.__extraReminderBanners") == 0, "history replay must not ring again"
            page.evaluate("window.__reminderBannerObserver.disconnect()")

            page.click('[data-nav-page="dashboard"]')
            page.click('[data-dashboard-tab="activity"]')
            schedules = page.locator('[data-activity-section="schedules"]')
            waiting = schedules.locator(".activity-row").filter(has_text=LATER)
            waiting.wait_for(state="visible", timeout=30_000)
            assert waiting.locator(".activity-name").inner_text() == LATER
            assert waiting.locator(".activity-sub[data-i18n-authored]").inner_text() == LATER
            timing = waiting.locator(".activity-sub").filter(has_text=re.compile(r"^reminder · one-shot · at/after "))
            assert timing.count() == 1
            assert timing.locator("time").get_attribute("datetime") == "2999-01-01T09:00:00Z"
            schedules.scroll_into_view_if_needed()
            page.screenshot(path=str(evidence / "activity-desktop.png"), full_page=True)
            page.set_viewport_size({"width": 390, "height": 844})
            wait_drawer_closed(page)
            page.evaluate(SETTLED)
            assert waiting.locator(".activity-sub").evaluate_all(
                "nodes => nodes.every(node => node.scrollWidth <= node.clientWidth + 1)")
            page.screenshot(path=str(evidence / "activity-narrow.png"), full_page=True)
        finally:
            browser.close()
    rows = dialogue_test.log_rows(root / "logs" / "chat.jsonl")
    assert [r["text"].split("\n", 1)[1] for r in rows if r.get("type") == "reminder"] == [DUE, LIVE]
