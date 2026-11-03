"""Activity census rows use real direct actors and the shared task controls."""
from __future__ import annotations

import json
import os
import time
import threading
from pathlib import Path

import pytest

from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data
from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET


@pytest.mark.ui_browser
@pytest.mark.parametrize("action", ["hurry", "finalize", "stop_now"])
def test_activity_direct_turn_reaches_shared_control_endpoint(direct_server_with_data, monkeypatch, action):
    from playwright.sync_api import sync_playwright
    from tests import fixtures_mock_llm

    monkeypatch.setattr(fixtures_mock_llm, "HOLD_SECONDS", 90)
    fixtures_mock_llm.HOLD_RELEASE.clear()
    entered = threading.Event()

    def held_completion(handler):
        payload = json.loads(handler.rfile.read(int(handler.headers.get("Content-Length", 0))))
        entered.set()
        fixtures_mock_llm.HOLD_RELEASE.wait(90)
        streaming = payload.get("stream") is True
        answer = {"id": "held-completion", "choices": [{"index": 0, "finish_reason": "stop",
            "delta" if streaming else "message": {"role": "assistant", "content": "OK"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        content = ("data: " + json.dumps(answer) + "\n\ndata: [DONE]\n\n" if streaming else json.dumps(answer)).encode()
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream" if streaming else "application/json")
        handler.send_header("Content-Length", str(len(content)))
        handler.end_headers()
        handler.wfile.write(content)

    monkeypatch.setattr(fixtures_mock_llm._Handler, "do_POST", held_completion)
    url = direct_server_with_data["url"]
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", direct_server_with_data["data_dir"].parent))
    evidence.mkdir(parents=True, exist_ok=True)
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            try:
                page = browser.new_page(viewport={"width": 1440, "height": 900})
                page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
                page.goto(url, wait_until="domcontentloaded")
                page.wait_for_function("() => window.__testSockets?.some(s => s.readyState === 1)")
                page.fill("#chat-input", "Respond with exactly OK")
                page.click("#chat-send")
                deadline = time.monotonic() + 30
                actor = None
                while time.monotonic() < deadline and (actor is None or not entered.is_set()):
                    state = page.request.get(url + "/api/state").json()
                    actor = next((a for a in state.get("active_chat_activities", []) if a["kind"] == "direct_chat"), None)
                    if actor is None or not entered.is_set():
                        page.wait_for_timeout(100)
                assert actor and entered.is_set(), state
                task_id = actor["activity_id"]
                queue = page.request.get(url + "/api/tasks?queue_only=1").json()["queue"]
                # ARCHITECTURE "Conversation lane": an owner turn is admitted by routing with its RUNNING row
                # published first, never PENDING, so the queue names it once, as a direct turn of the lane.
                rows = {part: [q for q in queue[part] if (q.get("id") or q.get("task", {}).get("id")) == task_id]
                        for part in ("running", "pending")}
                assert len(rows["running"]) == 1 and not rows["pending"], (rows, queue)
                owner = rows["running"][0]["task"]["_execution_owner"]
                assert (owner["kind"], owner["lane"]) == ("direct", "conversation"), owner
                page.click('[data-nav-page="dashboard"]')
                page.click('[data-dashboard-tab="activity"]')
                button = page.locator(f'[data-activity-section="queue"] [data-id="{task_id}"]')
                button.wait_for(state="visible")
                assert button.count() == 1
                row = button.locator("xpath=../..")
                text = row.inner_text()
                # The queue names this turn (the census adds only identities the queue lacks): by the owner's own
                # request, as a live row of the lane, never queued.
                assert "Respond with exactly OK" in text and "Queued" not in text, text
                assert "Nothing running" not in page.locator('[data-activity-section="queue"]').inner_text()
                button.click()
                menu = page.locator('body > .task-control-menu')
                menu.wait_for(state="visible")
                # A direct turn is a ROOT: it offers the owner's whole-tree Pause (Batch4 5A) too.
                assert menu.locator('[data-task-control]').all_text_contents() == [
                    "Wrap up", "Hurry up", "Pause", "Stop now"]
                page.screenshot(path=str(evidence / f"activity-{action}.png"), full_page=True)
                endpoint = "hurry" if action == "hurry" else "cancel"
                # Direct actors stop cooperatively. Let the held fake provider
                # complete after cancellation is submitted, so the real endpoint
                # can observe settlement instead of an artificial never-returning call.
                release = None
                if action == "stop_now":
                    def release_after_request(request):
                        nonlocal release
                        if request.method == "POST" and request.url.endswith(f"/{task_id}/cancel"):
                            release = threading.Timer(0.3, fixtures_mock_llm.HOLD_RELEASE.set)
                            release.start()
                    page.on("request", release_after_request)
                with page.expect_response(lambda r: f"/api/tasks/{task_id}/{endpoint}" in r.url and r.request.method == "POST") as response:
                    menu.locator(f'[data-task-control="{action}"]').click()
                if release is not None:
                    release.join(timeout=1)
                receipt = response.value
                body = receipt.json()
                assert receipt.status in (200, 202), body
                assert body.get("ok") is True, body
                request = receipt.request.post_data_json
                if action == "hurry":
                    assert request["request_id"], request
                else:
                    assert request.get("stop_policy", "immediate") == ("immediate" if action == "stop_now" else "finalize_then_cancel")
                (evidence / f"activity-{action}.json").write_text(json.dumps({
                    "actor": actor, "queue": queue, "status": receipt.status, "request": request, "response": body,
                }, indent=2), encoding="utf-8")
                fixtures_mock_llm.HOLD_RELEASE.set()
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    state = page.request.get(url + "/api/state").json()
                    if state.get("active_chat_activities_complete") is True and not any(
                            a["activity_id"] == task_id for a in state["active_chat_activities"]):
                        break
                    page.wait_for_timeout(100)
                else:
                    raise AssertionError(state)
            finally:
                browser.close()
    finally:
        fixtures_mock_llm.HOLD_RELEASE.set()
