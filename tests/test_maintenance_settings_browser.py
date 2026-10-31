"""The real Settings counter describes its inspected tail, including zero matches."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest_plugins = ("tests.test_ui_smoke_playwright",)
pytestmark = [pytest.mark.serial, pytest.mark.ui_browser]


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_inspected_safety_event_window_zero_and_nonzero(direct_server_with_data, engine):
    from playwright.sync_api import expect, sync_playwright

    now = datetime.now(timezone.utc)
    sample = {"count": 0}
    requests = []
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page = browser.new_page(viewport={"width": 1100, "height": 850})

            def tail(route):
                requests.append(route.request.url)
                entries = [{"type": "ordinary_event", "ts": now.isoformat()} for _ in range(1997)]
                entries.extend({"type": "safety_mode_skip", "ts": now.isoformat()}
                               for _ in range(sample["count"]))
                entries.extend([
                    {"type": "safety_mode_skip", "ts": (now - timedelta(days=2)).isoformat()},
                    {"type": "safety_mode_skip", "ts": "not-a-date"},
                ])
                route.fulfill(json={"entries": entries})

            page.route("**/api/logs/events?limit=2000", tail)
            for count in (0, 1):
                sample["count"] = count
                page.goto(direct_server_with_data["url"], wait_until="domcontentloaded")
                page.wait_for_selector("#page-chat", timeout=30_000)
                page.click('[data-nav-page="settings"]')
                expect(page.locator("#btn-save-settings")).to_be_enabled(timeout=30_000)
                page.click('[data-settings-tab="behavior"]')
                counter = page.locator("#s-safety-skip-counter")
                expected = ("1 safety check(s) waved through" if count else
                            "No waved-through safety checks found")
                expect(counter).to_have_text(expected + " among inspected recent events dated within 24h.")
                counter.scroll_into_view_if_needed()
                assert counter.is_visible()
                assert all("limit=2000" in url for url in requests)
                evidence = os.environ.get("OUROBOROS_UI_EVIDENCE_OUT")
                if evidence:
                    out = Path(evidence)
                    out.mkdir(parents=True, exist_ok=True)
                    page.screenshot(path=str(out / f"safety-window-{engine}-{count}.png"))
            assert len(requests) >= 2
        finally:
            browser.close()
