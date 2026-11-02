"""Activity keeps a withdrawn schedule visible and restorable, its accepted work in flight included."""
from __future__ import annotations

import datetime
import json
import os

import pytest

from ouroboros.test_evidence import diagnostic_directory

from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data


pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


def test_schedule_delete_withdraws_rows_that_stay_visible_and_restorable(direct_server_with_data):
    from playwright.sync_api import expect, sync_playwright

    from ouroboros.task_results import write_task_result

    fixture = direct_server_with_data
    root, url = fixture["data_dir"], fixture["url"]
    fixture["stop_server"]()
    now = datetime.datetime.now(datetime.timezone.utc)
    due = (now - datetime.timedelta(minutes=5)).isoformat()
    token, task_id, schedule_id = "held-occurrence", "aaaabbbb", "accepted-ui"
    task = {"id": task_id, "type": "task", "text": "Inspect the release results", "chat_id": 1,
            "metadata": {"schedule_occurrence": {"schedule_id": schedule_id, "token": token}}}
    write_task_result(root, task_id, "scheduled", chat_id=1, schedule_admission={
        "schedule_id": schedule_id, "token": token, "dispatch": "none", "task": task, "due_at": due,
    })
    # An accepted occurrence waiting to be republished after a resource hold.
    # Seed only while stopped, so the fixture's scheduler cannot race the write.
    row = {"id": schedule_id, "name": "Accepted continuation", "enabled": True,
           "trigger": {"type": "cron", "expr": "0 9 * * *"}, "timezone": "UTC",
           "next_run_at": (now + datetime.timedelta(days=1)).isoformat(), "task": task,
           "occurrence": {"token": token, "task_id": task_id, "phase": "admitted", "due_at": due},
           "hold": {"reason": "workspace_unusable", "detail": "Waiting for the working folder",
                    "since": now.isoformat(), "retry_after": (now + datetime.timedelta(hours=1)).isoformat()}}
    (root / "state" / "scheduled_tasks.json").write_text(
        json.dumps({"schema_version": 1, "tasks": [row]}), encoding="utf-8")
    fixture["start_server"]()
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=os.environ.get("OUROBOROS_TEST_CHROME") or None)
        try:
            for name, viewport in (("desktop", {"width": 1440, "height": 900}),
                                   ("mobile", {"width": 390, "height": 844})):
                page = browser.new_page(viewport=viewport)
                try:
                    clean_id = "disabled-" + name
                    created = page.request.post(url + "/api/schedules", data=json.dumps({
                        "id": clean_id, "name": "Disabled reminder", "enabled": False,
                        "trigger": {"type": "once", "run_at": (now + datetime.timedelta(days=1)).isoformat()},
                        "task": {"type": "task", "text": "Check later"},
                    }), headers={"Content-Type": "application/json"})
                    assert created.ok, created.text()
                    page.goto(url, wait_until="domcontentloaded")
                    if name == "mobile":
                        page.click('[data-mobile-nav-toggle]')
                    page.click('[data-nav-page="dashboard"]')
                    page.click('[data-dashboard-tab="activity"]')
                    section = page.locator('[data-activity-section="schedules"]')
                    retained = section.locator(f'[data-act="schedule-delete"][data-id="{schedule_id}"]')
                    if name == "desktop":
                        retained.click()
                        with page.expect_response(lambda response: response.url.endswith(f"/api/schedules/{schedule_id}/action")
                                                  and response.request.method == "POST") as changed:
                            page.get_by_role("dialog").get_by_role("button", name="Delete", exact=True).click()
                        result = changed.value.json()
                        assert result["status"] == "withdrawn" and result["schedule"] is not None, result
                        toast = page.locator(".toast").filter(has_text="Schedule withdrawn.").last
                        expect(toast).to_have_css("opacity", "1")
                    # Known work in flight keeps the withdrawn row in the standing list.
                    expect(retained).to_be_visible()
                    expect(section).to_contain_text("deletion pending · work in flight")
                    shot = diagnostic_directory(root.parent, "schedule-delete") / f"schedule-delete-{name}.png"
                    page.screenshot(path=str(shot), full_page=True, animations="disabled")
                    print(f"SCHEDULE_DELETE_SCREENSHOT {shot}")
                    removable = section.locator(f'[data-act="schedule-delete"][data-id="{clean_id}"]')
                    removable.click()
                    with page.expect_response(lambda response: response.url.endswith(f"/api/schedules/{clean_id}/action")
                                              and response.request.method == "POST") as removed:
                        page.get_by_role("dialog").get_by_role("button", name="Delete", exact=True).click()
                    result = removed.value.json()
                    assert result["status"] == "withdrawn" and result["schedule"] is not None, result
                    # Without work in flight the withdrawn row moves to history and can come back.
                    expect(removable).to_be_hidden()
                    section.locator("[data-activity-history] > summary").click()
                    expect(section.locator("[data-activity-history]")).to_contain_text("withdrawn · restorable")
                    restore = section.locator(f'[data-act="schedule-toggle"][data-id="{clean_id}"][data-action="restore"]')
                    with page.expect_response(lambda response: response.url.endswith(f"/api/schedules/{clean_id}/action")
                                              and response.request.method == "POST") as restored:
                        restore.click()
                    assert restored.value.json()["status"] == "updated", restored.value.json()
                    expect(section.locator(f'[data-act="schedule-toggle"][data-id="{clean_id}"][data-action="disable"]')).to_be_visible()
                finally:
                    page.close()
        finally:
            browser.close()
