from __future__ import annotations

import json

import pytest

pytest_plugins = ("tests.test_ui_smoke_playwright",)


@pytest.mark.ui_browser
def test_ui_smoke_final_only_child_stays_inside_child_card(direct_server_with_data):
    """A child final remains routed when every child progress row is absent."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    logs = data_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "progress.jsonl").write_text(json.dumps({
        "ts": "2026-08-21T10:00:00Z", "chat_id": 1, "task_id": "parent-final-only",
        "content": "Parent is working", "is_progress": True,
    }) + "\n", encoding="utf-8")
    sentinel = "FINAL_ONLY_CHILD_SENTINEL"
    (logs / "chat.jsonl").write_text(json.dumps({
        "ts": "2026-08-21T10:00:01Z", "chat_id": 1, "direction": "out",
        "task_id": "child-final-only", "text": sentinel, "format": "markdown",
    }) + "\n", encoding="utf-8")
    results = data_dir / "task_results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "child-final-only.json").write_text(json.dumps({
        "_schema_version": 1,
        "task_id": "child-final-only", "status": "completed",
        "delegation_role": "subagent", "parent_task_id": "parent-final-only",
        "root_task_id": "parent-final-only", "role": "legacy-reviewer",
        "model": "anthropic/claude-opus-5",
    }), encoding="utf-8")

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
                # The final text lives INSIDE the child card: in its (collapsed-hidden)
                # activity node, and on screen once the child is expanded.
                assert sentinel in child.locator(":scope > [data-live-summary-button] [data-live-activity]").text_content()
                child.locator(":scope > [data-live-summary-button]").click()
                assert sentinel in child.inner_text()
                assert child.locator("xpath=ancestor::*[@data-subagents-for='parent-final-only']").count() == 1
                assert page.locator(".chat-bubble", has_text=sentinel).count() == 0
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise
