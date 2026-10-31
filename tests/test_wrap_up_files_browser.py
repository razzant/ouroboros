"""Wrap up's saved material through the real SPA Files row and artifact gateway.

The bridge, terminal receipt, history, detail and download handlers are real; all
work and model outputs are synthetic. No server runtime, provider or controls run.
"""
from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from tests import test_subscription_setup_browser as setup
from tests.test_human_rows_browser import ROOM, _host, _push, _serve_history
from tests.test_sidebar_activity_browser import FRAMES, activity, mount_sidebar

subscription_ui = setup.subscription_ui
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_saved_work_download_and_receipt_survive_real_chat_reload(subscription_ui, tmp_path, monkeypatch, theme):
    from playwright.sync_api import expect

    from ouroboros.artifacts import collect_task_artifact_records, store_task_artifact_bytes
    from ouroboros.task_finalization import prepare_terminal_send_event
    from ouroboros.task_results import write_task_result
    from supervisor import events_chat_delivery, message_bus
    from tests.test_owner_stop_retained_work import _capture, _child, _client, _retained_long, _wrap_up

    ui, page = subscription_ui, subscription_ui["page"]
    out = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR") or tmp_path)
    out.mkdir(parents=True, exist_ok=True)
    bridge, frames = _host(tmp_path, monkeypatch)
    monkeypatch.setattr(events_chat_delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))
    client = _client(tmp_path)
    ui["backend"]["task_gateway"] = client  # downloads use the fixture's HTTP server and real handler
    _serve_history(page, tmp_path)

    def task_response(route):
        url = urlsplit(route.request.url)
        if "/artifacts/" in url.path:
            route.continue_()  # the browser really downloads the gateway's response
            return
        response = client.get(url.path + ("?" + url.query if url.query else ""))
        route.fulfill(status=response.status_code, headers=dict(response.headers), body=response.content)

    page.route("**/api/tasks/**", task_response)
    mounted = mount_sidebar(ui, 1100, theme, True, ROOM, [activity("parent1")])
    page.set_viewport_size({"width": 1100, "height": 750})
    sender = SimpleNamespace(DRIVE_ROOT=tmp_path, task_registry={}, RUNNING={},
                             send_with_budget=message_bus.send_with_budget, append_jsonl=lambda *_a: None,
                             load_state=lambda: {"owner_chat_id": 1})
    write_task_result(tmp_path, "parent1", "running", chat_id=1, result="working")
    _child(tmp_path, result="A saved helper answer.", terminal_origin="model_final")
    tail = "WRAP-UP-BROWSER-TAIL: all saved words are available."
    _retained_long(tmp_path, whole="saved line\n" * 110_000 + tail)
    _intent, stop = _wrap_up(tmp_path)
    _capture(tmp_path, "parent1", stop)
    for index in range(12):
        store_task_artifact_bytes(tmp_path, "parent1", f"report-{index}.txt", f"Report {index}".encode())
    artifacts = collect_task_artifact_records(tmp_path, "parent1")
    message_bus.send_with_budget(1, "Collecting the saved work.", is_progress=True,
                                 task_id="parent1", narration=True)
    _push(mounted, frames)
    write_task_result(tmp_path, "parent1", "failed", chat_id=1, result="Stopped with saved work.",
                      artifacts=artifacts, terminal_origin="model_final")
    event = prepare_terminal_send_event(
        tmp_path, {"id": "parent1", "chat_id": 1}, "Stopped with saved work.",
        {"terminal_origin": "model_final"},
        {"type": "send_message", "task_id": "parent1", "chat_id": 1,
         "text": "Stopped with saved work.", "progress_meta": {"task_terminal_status": "failed"}}, presence=False)
    events_chat_delivery._handle_send_message(event, sender)
    _push(mounted, frames, 1)
    mounted["refresh"](active_chat_activities=[])
    page.get_by_text("Stopped with saved work.", exact=True).wait_for()

    def check(phase):
        page.wait_for_selector('.chat-live-card[data-task-id="parent1"]', state="attached")
        for card in page.locator(".chat-live-card").all():
            disclosure = card.locator(":scope > [data-live-summary-button] [data-live-disclosure]")
            if card.get_attribute("data-expanded") != "1" and disclosure.is_visible():
                disclosure.click()
        files = page.locator("[data-result-files]").first
        if files.get_attribute("data-expanded") != "1":
            files.locator(".chat-live-line-head").click()
        receipt_line = page.locator('.chat-live-line').filter(
            has=page.get_by_text("Work saved at Wrap up:", exact=True)).first
        expect(receipt_line).to_be_visible()
        if receipt_line.locator(".chat-live-line-toggle").count() and receipt_line.get_attribute("data-expanded") != "1":
            receipt_line.locator(".chat-live-line-toggle").click()
        bullets = receipt_line.locator(".chat-live-line-body .md-li")
        assert bullets.count() == 4, "child, run and the two guidance lines are distinct bullets"
        assert "Download “Work saved at Wrap up”" not in bullets.nth(1).inner_text()
        assert bullets.nth(2).inner_text().startswith("• Download “Work saved at Wrap up”")
        link = files.get_by_role("link", name="Work saved at Wrap up", exact=True)
        expect(link).to_be_visible()
        expect(files.locator(".chat-live-line-time")).to_have_text("12 files")
        assert "+4 more files" in files.inner_text(), "ordinary eight-row cap stays, the saved-work door does not vanish"
        page.evaluate(FRAMES)
        files.scroll_into_view_if_needed()
        page.screenshot(path=str(out / f"wrap-up-files-{theme}-{phase}.png"), animations="disabled")
        with page.expect_download() as download:
            link.click()
        destination = out / f"wrap-up-{theme}-{phase}.md"
        download.value.save_as(str(destination))
        assert download.value.failure() is None
        data = destination.read_bytes()
        assert data.endswith(("> " + tail + "\n").encode())
        assert data.count(b"> saved line\n") == 110_000
        assert b"Its answer, in its own words:" in data and b"A saved helper answer." in data
        assert b"source_handles" not in data and str(tmp_path).encode() not in data
        assert files.locator("a").first.get_attribute("download") == "work-saved-at-wrap-up-parent1.md"
        return data

    first = check("live")
    ui["errors"].watch(page)  # the reload may cut off the page's own reads (tests/ui_console_errors.py)
    page.reload(wait_until="domcontentloaded")
    page.wait_for_function("() => window.__ouroWs?.ws?.readyState === WebSocket.OPEN")
    second = check("replay")
    assert first == second, "replay opens the same saved sources, not a truncated preview"
    assert not ui["errors"].unexpected(), ui["errors"].report()
    (out / f"wrap-up-{theme}-checks.json").write_text(json.dumps({
        "bytes": len(first), "lines": 110_000, "tail": tail, "replay_equal": first == second,
        "theme": theme, "viewport": "1100x750"}), encoding="utf-8")
