"""Full applied review download through the live Chat Reviews consumer."""

import hashlib
import json
import os
from pathlib import Path

import pytest

pytest_plugins = ("tests.test_ui_smoke_playwright",)


@pytest.mark.ui_browser
@pytest.mark.serial
def test_applied_review_download_updates_before_task_terminal(direct_server_with_data):
    from playwright.sync_api import sync_playwright

    from ouroboros import loop, review_projection
    from ouroboros.task_results import load_task_result, write_task_result
    from tests.test_acceptance_publication import _context, _run
    from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET, _emit_ws_frame

    root = direct_server_with_data["data_dir"]
    ctx = _context(root)
    trace = {"review_runs": [{**_run(), "task_attempt": ctx.task_attempt}]}
    legacy_panel = review_projection.compact_review_projection(trace["review_runs"])["panels"][0]
    legacy_panel["task_attempt"] = ctx.task_attempt
    legacy_panel["panel_index"] = 0
    write_task_result(root, ctx.task_id, "running", result="Work is still in progress.",
                      review_projection={"panels": [legacy_panel]})
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(root.parent)))
    evidence.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1000}, accept_downloads=True)
            page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
            page.goto(direct_server_with_data["url"], wait_until="domcontentloaded")
            page.wait_for_function("() => window.__testSockets?.[0]?.readyState === 1")
            _emit_ws_frame(page, {"type": "chat", "role": "system", "system_type": "review_reference",
                                 "surface": "task_acceptance", "task_id": ctx.task_id,
                                 "chat_id": 1, "state_revision": "a" * 64})
            card = page.locator(f'.chat-live-card[data-task-id="{ctx.task_id}"]')
            card.wait_for(state="visible")
            card.locator("[data-live-summary-button]").click()
            card.locator("[data-review-section-toggle]").click()
            card.locator("[data-review-group-toggle]").click()
            card.locator("[data-review-attempt-toggle]").click()
            detail = card.locator("[data-review-attempt-detail]")
            assert "Full applied review unavailable." in detail.inner_text()

            loop._set_acceptance_decision(trace, {"status": "accepted", "reason": "clean_pass"})
            review_projection.publish_acceptance_checkpoint(ctx, trace)
            stored = load_task_result(root, ctx.task_id)
            assert stored["status"] == "running"
            assert "artifacts" not in stored
            ref = stored["review_projection"]["panels"][0]["applied_source_ref"]
            # Route the actual producer's typed invalidation through Chat's
            # existing socket decoder, hydration and keyed DOM reconciliation.
            event = ctx.event_queue.get_nowait()["data"]
            _emit_ws_frame(page, {**event, "type": "chat", "role": "system",
                                 "system_type": "review_reference"})
            link = detail.get_by_role("link", name="Download full applied review")
            link.wait_for(state="visible")
            assert card.locator("[data-review-attempt-toggle]").get_attribute("aria-expanded") == "true"
            assert "Full applied review unavailable." not in detail.inner_text()
            href = link.get_attribute("href")
            response = page.request.get(direct_server_with_data["url"] + href)
            assert response.status == 200
            raw = response.body()
            assert hashlib.sha256(raw).hexdigest() == ref["sha256"]
            full = json.loads(raw)
            assert len(full["actors"][0]["parsed"]["findings"]) == 80
            assert full["applied_decision"]["status"] == "accepted"
            assert load_task_result(root, ctx.task_id)["status"] == "running"
            link.scroll_into_view_if_needed()
            card.screenshot(path=str(evidence / "applied-review-link.png"))
            with page.expect_download() as downloaded:
                link.click()
            download = downloaded.value
            assert download.failure() is None
            download.save_as(str(evidence / "downloaded-applied-review.json"))
            assert hashlib.sha256((evidence / "downloaded-applied-review.json").read_bytes()).hexdigest() == ref["sha256"]
        finally:
            browser.close()
