"""A real core restarts into another version under a kept page (#2249), Chromium.

``direct_server_with_data`` serves a copy of this checkout. While it is stopped, the test gives that copy the next
VERSION through the candidate overlay (``CandidateCheckout.rewrite``), so the restarted process reports another
``runtime_version`` over the same checkout SHA. The core's own page keeps itself on the equal SHA (ws.js ``decide``),
with its draft, caret, focus and reading place, and both version labels must follow the restarted core. The
controlled-transport cases and their helpers: ``tests/test_version_labels_browser.py``.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from tests.candidate_checkout import VERSION_PATH
from tests.test_reconnect_feed_browser import HISTORY_LANDED
from tests.test_ui_smoke_playwright import _store_rows
from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data
from tests.test_version_labels_browser import CARET, DRAFT, FACTS, HEALTH_READS, MAIN, PIN, settle_labels
from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
_ROWS_SHOWN = "(n) => document.querySelectorAll('#chat-messages [data-history-id]').length >= n"
_NEWEST_SOCKET_OPEN = "(n) => window.__testSockets.length >= n && window.__testSockets.at(-1).readyState === WebSocket.OPEN"


def _rows(count):
    text = " The reading notes continue with ordinary context about the cited pages." * 4
    return [{"ts": f"2026-10-25T09:{index:02d}:00+00:00", "direction": "in" if index % 2 == 0 else "out",
             "chat_id": 1, "format": "markdown",
             "text": (f"Question {index}: what changed in section {index}?" if index % 2 == 0
                      else f"Answer {index}." + text)} for index in range(count)]


def _next_version(text):
    """The next patch release with this copy's own build marker, so the restarted process still proves its origin."""
    return re.sub(r"^(\d+\.\d+\.)(\d+)", lambda match: f"{match[1]}{int(match[2]) + 1}", text, count=1)


@pytest.mark.parametrize("before, after", [
    ("7.8.1+candidate.x", "7.8.2+candidate.x"),
    ("7.0.0-rc.9+candidate.x", "7.0.1-rc.9+candidate.x"),
    ("7.9.0rc2+candidate.x", "7.9.1rc2+candidate.x"),
    ("7.9.0.post1+candidate.x", "7.9.1.post1+candidate.x"),
])
def test_version_change_preserves_release_suffix_and_origin(before, after):
    assert _next_version(before) == after


def test_a_kept_page_follows_the_restarted_core_to_its_new_version(direct_server_with_data):
    from playwright.sync_api import sync_playwright

    server = direct_server_with_data
    old = server["candidate_version"]
    new = _next_version(old)
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", server["data_dir"].parent)) / "version-labels-restart"
    evidence.mkdir(parents=True, exist_ok=True)
    _store_rows(server["data_dir"], "chat", _rows(24))
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1100, "height": 722})
            page.add_init_script(f"({_CAPTURE_TEST_SOCKET})();({HEALTH_READS})()")
            page.goto(server["url"], wait_until="domcontentloaded")
            page.wait_for_function(_ROWS_SHOWN, arg=24, timeout=30_000)
            page.wait_for_function(_NEWEST_SOCKET_OPEN, arg=1, timeout=30_000)
            settle_labels(page)
            assert page.evaluate("() => document.querySelector('meta[name=\"ouroboros-ui-owner\"]')?.content") == "core"
            page.fill("#chat-input", DRAFT)
            pinned = page.evaluate(PIN, [MAIN, CARET])
            assert pinned["below"] > 200, pinned
            page.wait_for_timeout(300)
            before = page.evaluate(FACTS, MAIN)
            assert (before["nav"], before["about"]) == (f"v{old}", f"app {old} | rt {old}"), before
            page.screenshot(path=str(evidence / "restart-before.png"), animations="disabled")

            page.evaluate(HISTORY_LANDED)
            server["stop_server"]()
            page.wait_for_function("() => document.querySelector('#chat-status')?.textContent.includes('Reconnecting')")
            server["candidate"].rewrite(VERSION_PATH, f"{new}\n".encode("utf-8"))
            server["start_server"]()  # verifies the copy, then that /api/health reports the new version
            page.wait_for_function(_NEWEST_SOCKET_OPEN, arg=2, timeout=60_000)
            page.wait_for_function("() => window.historyLanded > 0", timeout=30_000)
            settle_labels(page)
            page.wait_for_timeout(300)
            after = page.evaluate(FACTS, MAIN)
            page.screenshot(path=str(evidence / "restart-after.png"), animations="disabled")
            page.locator('[data-nav-page="settings"]').click()
            page.locator('[data-settings-tab="about"]').click()
            page.locator("#about-version").wait_for(state="visible")
            page.screenshot(path=str(evidence / "restart-about.png"), animations="disabled")
            # Control: a page opened after the restart reads the new version at load, with or without the repair.
            fresh = browser.new_page(viewport={"width": 1100, "height": 722})
            fresh.add_init_script(f"({HEALTH_READS})()")
            fresh.goto(server["url"], wait_until="domcontentloaded")
            fresh.wait_for_function(_ROWS_SHOWN, arg=24, timeout=30_000)
            settle_labels(fresh)
            cold = fresh.evaluate(FACTS, MAIN)
            (evidence / "restart-facts.json").write_text(json.dumps(
                {"old": old, "new": new, "before": before, "after": after, "cold": cold}, indent=2), encoding="utf-8")

            assert after["keptDocument"] and "_ouro_reason" not in after["url"], after
            assert after["draft"] == DRAFT and after["caret"] == [CARET, CARET] and after["focused"], after
            assert after["sameNodes"] and after["rowKept"] and after["drift"] <= 6, after
            assert (cold["nav"], cold["about"]) == (f"v{new}", f"app {new} | rt {new}"), cold
            assert (after["nav"], after["about"]) == (f"v{new}", f"app {new} | rt {new}"), after
        finally:
            browser.close()
