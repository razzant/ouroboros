"""A Settings save states the skill-key rule once, whatever keys it carried (#1122).

A running task keeps its start snapshot; a skill reads the keys it is granted on its
next read. The save status says so in its next-task sentence, never per key and never
twice beside the running-task warning. Every value typed here is a synthetic fixture.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest

pytest_plugins = ("tests.test_ui_smoke_playwright",)

SKILL_RULE = "skills pick up granted keys on their next read"
NEXT_TASK = f"Settings saved. Changes take effect on the next task; {SKILL_RULE}"
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


@pytest.mark.serial
@pytest.mark.ui_browser
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_settings_save_states_the_skill_key_rule_once(direct_server_with_data, monkeypatch, engine):
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
                status = page.locator("#settings-status")

                def open_settings(tab):
                    page.click('[data-nav-page="settings"]')
                    expect(page.locator("#btn-save-settings")).to_be_enabled(timeout=30_000)
                    expect(status).to_have_text("Settings refreshed", timeout=30_000)
                    page.click(f'[data-settings-tab="{tab}"]')

                def save(expected, shot):
                    page.click("#btn-save-settings")
                    expect(status).to_contain_text(expected, timeout=20_000)
                    text = status.inner_text()
                    page.screenshot(path=str(evidence / f"settings-save-{shot}-{engine}.png"))
                    return text

                # An immediate-effect key: no next-task sentence and no skill rule.
                open_settings("advanced")
                page.locator("#s-gh-repo").fill("owner/skill-key-proof")
                text = save("Settings saved. Changes took effect immediately", "immediate-key")
                assert SKILL_RULE not in text and "next task" not in text, text

                # A provider key is read live (route_choice, #2308): it took effect at once, with no skill rule.
                page.click('[data-settings-tab="providers"]')
                page.locator("#s-openrouter").fill(SYNTHETIC_OPENROUTER)
                text = save("Settings saved. Changes took effect immediately", "common-key")
                assert SKILL_RULE not in text and "An agent task is running" not in text, text
                assert saves[-1]["OPENROUTER_API_KEY"] == SYNTHETIC_OPENROUTER
                assert stored()["OPENROUTER_API_KEY"] == SYNTHETIC_OPENROUTER

                # A saved custom key, then its removal: a real change, never "No changes detected".
                page.click('[data-settings-tab="secrets"]')
                page.click("#btn-add-custom-secret")
                row = page.locator("[data-custom-secret-row]").last
                row.locator("[data-custom-secret-key]").fill(CUSTOM_KEY)
                row.locator("[data-custom-secret-value]").fill("synthetic-custom-value")
                save(NEXT_TASK, "custom-key-added")
                assert stored()[CUSTOM_KEY] == "synthetic-custom-value"
                saved_row = page.locator("[data-custom-secret-row]").filter(
                    has=page.locator(f'[data-custom-secret-key][value="{CUSTOM_KEY}"]'))
                expect(saved_row).to_have_count(1, timeout=20_000)
                saved_row.locator("[data-custom-secret-remove]").click()
                text = save(NEXT_TASK, "custom-key-removed")
                assert "No changes detected" not in text and saves[-1].get(CUSTOM_KEY) == ""
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
                open_settings("secrets")
                field = page.locator('#skill-requested-secrets [data-secret-setting="WEATHER_TOKEN"]')
                expect(field).to_have_count(1, timeout=20_000)
                field.fill(SYNTHETIC_SKILL_TOKEN)
                text = save("keeps the reviewers, subagents, context and effort it started with", "skill-key-running-task")
                assert text.startswith(NEXT_TASK) and text.count(SKILL_RULE) == 1, text
                assert "The saved changes apply from the next task" not in text, text
                assert stored()["WEATHER_TOKEN"] == SYNTHETIC_SKILL_TOKEN
            finally:
                browser.close()
    finally:
        fixtures_mock_llm.HOLD_RELEASE.set()
