from __future__ import annotations

import pytest

from ouroboros import dialogue_store
from tests._store_shared import seed_result

pytest_plugins = ("tests.test_ui_smoke_playwright",)


@pytest.mark.ui_browser
def test_ui_smoke_final_only_child_stays_inside_child_card(direct_server_with_data):
    """A child final remains routed when every child progress row is absent."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    # The running server reads its own dialogue store (#1982), never log files written after it started.
    dialogue_store.append(data_dir, "progress", {
        "ts": "2026-08-21T10:00:00Z", "chat_id": 1, "task_id": "parent-final-only",
        "content": "Parent is working", "is_progress": True,
    })
    sentinel = "FINAL_ONLY_CHILD_SENTINEL"
    dialogue_store.append(data_dir, "chat", {
        "ts": "2026-08-21T10:00:01Z", "chat_id": 1, "direction": "out",
        "task_id": "child-final-only", "text": sentinel, "format": "markdown",
    })
    results = data_dir / "task_results"
    results.mkdir(parents=True, exist_ok=True)
    seed_result(data_dir, "child-final-only", {
        "_schema_version": 1,
        "task_id": "child-final-only", "status": "completed",
        "delegation_role": "subagent", "parent_task_id": "parent-final-only",
        "root_task_id": "parent-final-only", "role": "legacy-reviewer",
        "model": "anthropic/claude-opus-5",
    })

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                child = page.locator('.chat-live-card.subagent[data-task-id="child-final-only"]')
                child.wait_for(state="visible", timeout=30_000)
                assert child.get_attribute("data-parent-task-id") == "parent-final-only"
                assert child.get_attribute("data-finished") == "1"
                # A child is never a convertible unit: only its root offers conversion.
                assert child.locator("[data-turn-into-project]").count() == 0
                # The final text lives INSIDE the child card: in its (collapsed-hidden)
                # activity node, and on screen once the child is expanded.
                assert sentinel in child.locator(":scope > [data-live-summary-button] [data-live-activity]").text_content()
                child.locator(":scope > [data-live-summary-button] [data-live-disclosure]").click()
                assert sentinel in child.inner_text()
                assert child.locator("xpath=ancestor::*[@data-subagents-for='parent-final-only']").count() == 1
                assert page.locator(".chat-bubble", has_text=sentinel).count() == 0
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise
