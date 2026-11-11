"""Skill keys commit locally while a running task keeps its start snapshot; synthetic credentials only."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest
from tests.settings_browser_helpers import open_settings, settings_page, commit_row, press_editor_save

pytest_plugins = ("tests.test_ui_smoke_playwright",)

SYNTHETIC_OPENROUTER = "sk-or-synthetic-ui-fixture-" + "0123456789abcdef" * 2
SYNTHETIC_SKILL_TOKEN = "synthetic-weather-token-" + "0123456789abcdef" * 2
CUSTOM_KEY = "UI_FIXTURE_CUSTOM_KEY"


def _held_completion(entered):
    from tests import fixtures_mock_llm

    def answer(handler):
        payload = json.loads(handler.rfile.read(int(handler.headers.get("Content-Length", 0))))
        entered.set()
        fixtures_mock_llm.HOLD_RELEASE.wait(120)
        streaming = payload.get("stream") is True
        message = {"id": "held-completion", "choices": [{"index": 0, "finish_reason": "stop",
                   "delta" if streaming else "message": {"role": "assistant", "content": "OK"}}],
                   "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        body = ("data: " + json.dumps(message) + "\n\ndata: [DONE]\n\n" if streaming else json.dumps(message)).encode()
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream" if streaming else "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    return answer


@pytest.mark.ui_browser
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_skill_keys_save_locally_without_echo_or_changing_running_work(direct_server_with_data, monkeypatch, engine):
    from playwright.sync_api import expect, sync_playwright

    from tests import fixtures_mock_llm
    from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET

    server = direct_server_with_data
    url = server["url"]
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(server["data_dir"].parent)))
    evidence.mkdir(parents=True, exist_ok=True)
    listing = {"skills": [{"name": "weather", "grants": {"requested_keys": ["WEATHER_TOKEN"]}}],
               "live": {"settings_sections": []}}
    entered = threading.Event()
    fixtures_mock_llm.HOLD_RELEASE.clear()
    monkeypatch.setattr(fixtures_mock_llm._Handler, "do_POST", _held_completion(entered))
    saves, stored = [], lambda: json.loads((server["data_dir"] / "settings.json").read_text(encoding="utf-8"))
    try:
        with sync_playwright() as pw:
            browser = getattr(pw, engine).launch()
            try:
                page = browser.new_page(viewport={"width": 1280, "height": 900})
                page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
                page.on("request", lambda request: saves.append(request.post_data_json)
                        if request.method == "POST" and request.url.endswith("/api/settings") else None)
                # Only the requested-key listing is advertised; every save is the real gateway's.
                page.route("**/api/extensions", lambda route: route.fulfill(json=listing))
                page.goto(url, wait_until="domcontentloaded")
                page.wait_for_function("() => window.__testSockets?.some(s => s.readyState === 1)")
                listing_page = open_settings(page, "accounts")
                text = commit_row(page, "GITHUB_REPO", "owner/skill-key-proof")
                assert text.startswith("Saved")
                assert saves[-1] == {"GITHUB_REPO": "owner/skill-key-proof"}

                key_page = open_settings(page, "accounts/key/openrouter")
                key_page.locator("#s-openrouter").fill(SYNTHETIC_OPENROUTER)
                assert "Saved" in press_editor_save(page)
                assert saves[-1] == {"OPENROUTER_API_KEY": SYNTHETIC_OPENROUTER}
                assert stored()["OPENROUTER_API_KEY"] == SYNTHETIC_OPENROUTER

                new_key = open_settings(page, "accounts/key/custom")
                new_key.locator("[data-custom-secret-key]").fill(CUSTOM_KEY)
                new_key.locator("[data-custom-secret-value]").fill("synthetic-custom-value")
                new_key.locator("[data-editor-save]").click()
                saved_row = settings_page(page, "accounts").locator(f'[data-setting-row="{CUSTOM_KEY}"]')
                expect(saved_row).to_be_visible(timeout=20_000)
                assert stored()[CUSTOM_KEY] == "synthetic-custom-value"
                saved_row.locator("[data-secret-open]").click()
                saved_row.locator("[data-secret-remove]").click()
                page.locator("[data-confirm-cancel]").click()
                assert stored()[CUSTOM_KEY] == "synthetic-custom-value"
                saved_row.locator("[data-secret-remove]").click()
                page.locator("[data-confirm-ok]").click()
                expect(saved_row).to_have_count(0)
                assert saves[-1] == {CUSTOM_KEY: ""}
                assert not stored().get(CUSTOM_KEY)

                # A skill's key saved while a direct turn genuinely runs: the rule once,
                # beside the gateway's running-task warning.
                page.click('[data-nav-page="chat"]')
                page.fill("#chat-input", "Respond with exactly OK")
                page.click("#chat-send")
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline and not (entered.is_set() and any(
                        activity.get("kind") == "direct_chat" for activity in
                        page.request.get(url + "/api/state").json().get("active_chat_activities", []))):
                    page.wait_for_timeout(100)
                assert entered.is_set()
                listing_page = open_settings(page, "accounts")
                row = listing_page.locator('[data-setting-row="WEATHER_TOKEN"]')
                row.locator("[data-secret-open]").click()
                text = commit_row(page, "WEATHER_TOKEN", SYNTHETIC_SKILL_TOKEN)
                assert "Saved" in text and "applies to new tasks" in text
                assert "was" not in text and "Undo" not in text and SYNTHETIC_SKILL_TOKEN not in text
                expect(row.locator("[data-setting-control]")).to_have_value("")
                assert stored()["WEATHER_TOKEN"] == SYNTHETIC_SKILL_TOKEN
                assert saves[-1] == {"WEATHER_TOKEN": SYNTHETIC_SKILL_TOKEN}
                page.screenshot(path=str(evidence / f"skill-key-running-task-{engine}.png"), animations="disabled")
            finally:
                browser.close()
    finally:
        fixtures_mock_llm.HOLD_RELEASE.set()
