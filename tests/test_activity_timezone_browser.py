"""Owner-facing schedule times use one zone per row, each time keeping the exact UTC instant (#1320).

A real browser in a non-UTC zone (Asia/Tokyo) against the real server: the chat frame
it sends carries that IANA zone. Activity says a stored UTC instant in the schedule's own
zone, named when it is not the viewer's, else in Tokyo time; every time keeps the exact
UTC instant as its datetime and title. The screenshot is evidence for inspection.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data

_CAPTURE_FRAMES = """() => {
    window.__sentFrames = [];
    const nativeSend = WebSocket.prototype.send;
    WebSocket.prototype.send = function (data) {
        try { window.__sentFrames.push(JSON.parse(data)); } catch (_error) { /* not JSON */ }
        return nativeSend.call(this, data);
    };
}"""


@pytest.mark.ui_browser
def test_activity_schedules_render_in_the_viewer_zone_with_utc_beside(direct_server_with_data):
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", direct_server_with_data["data_dir"].parent))
    evidence.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            context = browser.new_context(timezone_id="Asia/Tokyo", locale="en-GB",
                                          viewport={"width": 1440, "height": 900})
            page = context.new_page()
            page.add_init_script(f"({_CAPTURE_FRAMES})()")
            for body in (
                {"id": "tz-once", "name": "Release reminder", "trigger": {"type": "once",
                 "run_at": "2027-01-15T09:00:00Z"}, "task": {"type": "task", "text": "remind"}},
                {"id": "tz-cron", "name": "Morning digest", "timezone": "Europe/Moscow",
                 "trigger": {"type": "cron", "expr": "0 9 * * *"}, "task": {"type": "task", "text": "digest"}},
            ):
                response = page.request.post(url + "/api/schedules", data=json.dumps(body),
                                             headers={"Content-Type": "application/json"})
                assert response.ok, response.text()
            page.goto(url, wait_until="domcontentloaded")
            page.fill("#chat-input", "Respond with exactly OK")
            page.click("#chat-send")
            page.wait_for_function("() => window.__sentFrames.some(f => f.type === 'chat')")
            frame = page.evaluate("() => window.__sentFrames.find(f => f.type === 'chat')")
            assert frame["client_surface"]["timezone"] == "Asia/Tokyo"
            page.click('[data-nav-page="dashboard"]')
            page.click('[data-dashboard-tab="activity"]')
            section = page.locator('[data-activity-section="schedules"]')
            section.locator("time").first.wait_for(state="visible")
            text = section.inner_text()
            once = section.locator('time[datetime="2027-01-15T09:00:00Z"]').first
            assert once.get_attribute("title") == "2027-01-15T09:00:00Z"
            # 09:00 UTC is 18:00 in Tokyo, the viewer's own zone, so the row names none.
            assert "Task · Once on 15 Jan at 18:00 · active" in text, text
            # A cron row says its cadence and next run in its own zone, named; 09:00 in Moscow is 06:00 UTC. The cron
            # code and the zone are the row's Details.
            assert re.search(r"Task · Daily at 09:00 \(Europe/Moscow\) · next \d{1,2} \w{3} at 09:00 · active", text), text
            digest = section.locator(".activity-schedule", has_text="Morning digest")
            following = digest.locator(".activity-sub time").first
            assert re.fullmatch(r"\d{4}-\d\d-\d\dT06:00:00Z", following.get_attribute("datetime") or ""), text
            assert following.get_attribute("title") == following.get_attribute("datetime")
            digest.locator("details.activity-details > summary").click()
            facts = digest.locator(".activity-facts").inner_text()
            assert "cron 0 9 * * *" in facts and "time zone Europe/Moscow" in facts, facts
            shot = evidence / "activity-timezone-tokyo.png"
            section.scroll_into_view_if_needed()
            page.screenshot(path=str(shot), full_page=True)
            print(f"ACTIVITY_TIMEZONE_SCREENSHOT {shot}")
        finally:
            browser.close()
