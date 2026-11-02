"""Exercise TZ-1 owner history/readiness presentation in the real SPA."""
from __future__ import annotations

import json

import pytest

from tests.test_subscription_setup_browser import subscription_ui as subscription_ui, capture

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


def test_saved_input_and_starting_survive_reload(subscription_ui):
    ui = subscription_ui
    page = ui["page"]
    rows = [{
        "role": "user", "text": "TZ-1 saved message", "ts": "2026-09-25T03:00:00Z",
        "client_message_id": "tz1-saved-message", "ingress_accepted": True,
    }]
    page.route("**/api/state*", lambda route: route.fulfill(
        content_type="application/json", body=json.dumps({
            "supervisor_ready": False, "supervisor_error": "still starting",
            "active_chat_activities": [], "active_chat_activities_complete": True,
            "projects": [],
        }),
    ))
    page.route("**/api/chat/history*", lambda route: route.fulfill(
        content_type="application/json", body=json.dumps({"messages": rows, "progress": []}),
    ))
    page.goto(ui["url"])
    saved = page.locator('.chat-bubble.user[data-client-message-id="tz1-saved-message"] [data-ingress-saved]')
    saved.wait_for()
    assert saved.inner_text() == "Input saved"
    assert page.get_by_text("Starting…", exact=True).count() >= 1
    capture(page, "tz1-saved-and-starting")
    ui["errors"].watch(page)  # the reload may cut off the page's own reads (tests/ui_console_errors.py)
    page.reload()
    saved.wait_for()
    assert saved.count() == 1
    assert page.get_by_text("Starting…", exact=True).count() >= 1
    assert not ui["errors"].unexpected(), ui["errors"].report()


def test_failed_supervisor_init_never_paints_online(subscription_ui):
    """The host's failure rail publishes `supervisor_ready: false` beside the
    error; the real SPA header says Starting… and never Online over it."""
    ui = subscription_ui
    page = ui["page"]
    page.route("**/api/state*", lambda route: route.fulfill(
        content_type="application/json", body=json.dumps({
            "supervisor_ready": False, "supervisor_error": "Supervisor init failed: boot dependency refused",
            "active_chat_activities": [], "active_chat_activities_complete": True, "projects": [],
        }),
    ))
    page.route("**/api/chat/history*", lambda route: route.fulfill(
        content_type="application/json", body=json.dumps({"messages": [], "progress": []}),
    ))
    page.goto(ui["url"])
    badge = page.locator("#chat-status")
    badge.wait_for()
    page.wait_for_function("() => document.querySelector('#chat-status')?.textContent === 'Starting…'")
    assert "online" not in (badge.get_attribute("class") or "").split()
    assert page.get_by_text("Online", exact=True).count() == 0
    capture(page, "tz1-failed-init-starting")
    assert not ui["errors"], ui["errors"]


def test_recorded_folder_download_from_real_gateway_in_chat(subscription_ui, tmp_path):
    """The served SPA reads real task detail and ZIP bytes from the gateway,
    rather than an authored HTML fragment or a fake detail response."""
    import io
    import zipfile

    from tests.test_task_file_serving import TASK, _client, _split_child
    from ouroboros import headless

    data, child, _store = _split_child(tmp_path)
    headless.copy_child_task_result(data, {"id": TASK, "drive_root": str(child)})
    client = _client(data)
    subscription_ui["backend"]["task_gateway"] = client
    page = subscription_ui["page"]
    url_prefix = f"/api/tasks/{TASK}"
    seen = []

    def gateway(route):
        from urllib.parse import urlparse
        path = urlparse(route.request.url).path
        suffix = route.request.url.split(path, 1)[1]
        response = client.get(path + suffix)
        seen.append((path, response.status_code))
        route.fulfill(status=response.status_code,
                      headers={"content-type": response.headers.get("content-type", "application/json"),
                               "content-disposition": response.headers.get("content-disposition", "")},
                      body=response.content)

    page.route(f"**{url_prefix}*", gateway)
    page.goto(subscription_ui["url"])
    # The event is only an activation fixture. The browser's task-detail read,
    # archive availability and downloaded bytes all come from the real gateway.
    page.evaluate("row => window.__ouroWs.emit('chat', row)", {
        "chat_id": 1, "task_id": TASK, "role": "assistant", "is_progress": True,
        "content": "Building reports", "ts": "2026-09-25T07:00:00Z",
    })
    page.evaluate("row => window.__ouroWs.emit('chat', row)", {
        "chat_id": 1, "task_id": TASK, "role": "system", "system_type": "task_summary",
        "task_terminal_status": "completed", "content": "Done", "ts": "2026-09-25T07:01:00Z",
    })
    card = page.locator(f'#chat-messages .chat-live-card[data-task-id="{TASK}"]')
    card.wait_for()
    assert card.locator('[data-live-phase]').inner_text() == "Done"
    card.locator('[data-live-disclosure]').click()
    page.wait_for_selector(f'#chat-messages .chat-live-card[data-task-id="{TASK}"] [data-result-files]')
    folder = card.locator('[data-result-files] a[href*="?archive=reports"]')
    assert folder.count() == 1 and "reports/" in folder.inner_text()
    with page.expect_download() as item:
        folder.click()
    assert item.value.failure() is None, (item.value.failure(), seen, subscription_ui["errors"])
    payload = item.value.path().read_bytes()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert archive.namelist() == ["reports/a/summary.txt", "reports/b/summary.txt"]
        assert archive.read("reports/a/summary.txt") == b"alpha"
    assert (url_prefix, 200) in seen, seen  # real detail, not a fixture JSON object
    capture(page, "tz1-real-folder-download")
    assert not subscription_ui["errors"], subscription_ui["errors"]
