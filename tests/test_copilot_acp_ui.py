"""Opt-in rendered Settings and task-history proof using the existing UI stand."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess

import pytest

from tests.test_ui_smoke_playwright import direct_server_with_data as _direct_server_with_data
from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET, _emit_ws_frame

direct_server_with_data = _direct_server_with_data
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


def test_copilot_settings_save_and_task_activity_replay(direct_server_with_data, tmp_path):
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright
    from ouroboros.task_results import write_task_result
    from ouroboros.utils import append_jsonl, utc_now_iso
    from ouroboros.contracts.chat_id_policy import WEB_UI_CHAT_ID

    url, data = direct_server_with_data["url"], direct_server_with_data["data_dir"]
    evidence = pathlib.Path(os.environ.get("OUROBOROS_BROWSER_EVIDENCE_OUT") or tmp_path)
    evidence.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page_errors = []
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
            page.on("pageerror", lambda error: page_errors.append(str(error)))

            def ready():
                page.wait_for_function(
                    "() => window.__testSockets?.some(socket => socket.readyState === WebSocket.OPEN)",
                    timeout=30_000,
                )

            def settings_ready():
                page.wait_for_function(
                    "() => document.getElementById('btn-save-settings')?.disabled === false"
                    " && !document.getElementById('settings-status')?.textContent.startsWith('Loading')",
                    timeout=30_000,
                )

            # Optional contributor before/after evidence: render the exact base
            # frontend through the same real consumer, never alter its DOM.
            base = os.environ.get("OUROBOROS_UI_BASE_REF", "")
            routes = []
            if base:
                repo = pathlib.Path(__file__).resolve().parents[1]
                for name in ("settings_ui.js", "settings.js", "log_events.js"):
                    source = subprocess.run(
                        ["git", "show", f"{base}:web/modules/{name}"], cwd=repo,
                        capture_output=True, text=True, check=True,
                    ).stdout
                    pattern = f"**/static/modules/{name}"
                    page.route(pattern, lambda route, _request, source=source: route.fulfill(body=source, content_type="text/javascript"))
                    routes.append(pattern)
                page.goto(url, wait_until="domcontentloaded")
                ready()
                page.click('[data-nav-page="settings"]')
                page.click('[data-settings-tab="agents"]')
                settings_ready()
                page.locator("#available-subagents-editor").wait_for(state="visible")
                assert page.locator("#s-task-backend").count() == 0
                page.screenshot(path=str(evidence / "copilot-settings-before.png"))
                for pattern in routes:
                    page.unroute(pattern)
            page.goto(url, wait_until="domcontentloaded")
            ready()
            page.click('[data-nav-page="settings"]')
            page.click('[data-settings-tab="agents"]')
            settings_ready()
            page.locator("#s-task-backend").wait_for(state="visible")
            page.select_option("#s-task-backend", "copilot_acp")
            page.select_option("#s-copilot-permissions", "workspace")
            with page.expect_response(
                lambda response: response.request.method == "POST" and response.url.endswith("/api/settings"),
                timeout=30_000,
            ) as saved:
                page.click("#btn-save-settings")
            assert saved.value.status == 200, saved.value.text()
            settings = page.request.get(f"{url}/api/settings").json()
            assert settings["OUROBOROS_TASK_BACKEND"] == "copilot_acp"
            assert settings["OUROBOROS_COPILOT_PERMISSION_POLICY"] == "workspace"
            page.reload(wait_until="domcontentloaded")
            ready()
            page.click('[data-nav-page="settings"]')
            page.click('[data-settings-tab="agents"]')
            settings_ready()
            assert page.locator("#s-task-backend").input_value() == "copilot_acp"
            assert page.locator("#s-copilot-permissions").input_value() == "workspace"
            page.screenshot(path=str(evidence / "copilot-settings-after.png"))

            # Production-shaped retained rows, not a claimed live model run.
            # The real CLI transport is separately smoke-tested; this checks
            # that its new event vocabulary survives the real history reader.
            task_id = "copilot-ui-fixture"
            common = {
                "ts": utc_now_iso(), "task_id": task_id, "chat_id": WEB_UI_CHAT_ID,
                "execution_backend": "copilot_acp", "execution_id": "ui-fixture",
            }
            rows = [
                {**common, "type": "task_runtime_update", "sequence": 1, "acp_update_type": "plan", "text": "in_progress: Inspect the regression"},
                {**common, "type": "task_runtime_update", "sequence": 2, "acp_update_type": "permission_resolved", "text": "Copilot permission allowed once"},
                {**common, "type": "tool_call_started", "sequence": 3, "tool_call_id": "fixture-edit", "tool": "Copilot edit", "args": {"path": "tracked.txt"}},
                {**common, "type": "task_runtime_update", "sequence": 4, "acp_update_type": "diff", "text": "tracked.txt\n--- before\nold\n+++ after\nfixed"},
                {**common, "type": "tool_call_finished", "sequence": 5, "tool_call_id": "fixture-edit", "tool": "Copilot edit", "is_error": False, "result_preview": "File updated"},
            ]
            write_task_result(data, task_id, "running", chat_id=WEB_UI_CHAT_ID, title="Copilot ACP fixture")
            page.click('[data-nav-page="chat"]')
            for row in rows:
                _emit_ws_frame(page, {"type": "log", "data": row})
            card = page.locator(f'.chat-live-card[data-task-id="{task_id}"]')
            card.wait_for(state="visible")
            card.locator(":scope > [data-live-summary-button]").click()
            assert "Copilot ACP" in card.inner_text()
            assert "tracked.txt" in card.inner_text()
            page.screenshot(path=str(evidence / "copilot-task-live.png"))
            for row in rows:
                log_name = "tools.jsonl" if row["type"].startswith("tool_call_") else "progress.jsonl"
                append_jsonl(data / "logs" / log_name, row)
            write_task_result(
                data, task_id, "completed", chat_id=WEB_UI_CHAT_ID,
                title="Copilot ACP fixture", execution_backend="copilot_acp",
                result="Fixture change verified.", unknown_unmetered=1, cost_final=False,
                outcome_axes={"execution": {"status": "ok"}, "review": {"status": "skipped"}, "objective": {"status": "not_evaluated"}},
            )
            done = {**common, "type": "task_done", "status": "completed"}
            append_jsonl(data / "logs" / "events.jsonl", done)
            _emit_ws_frame(page, {"type": "log", "data": done})
            page.reload(wait_until="domcontentloaded")
            ready()
            card.wait_for(state="visible", timeout=30_000)
            card.locator(":scope > [data-live-summary-button]").click()
            assert "Copilot ACP" in card.inner_text()
            assert "tracked.txt" in card.inner_text()
            page.screenshot(path=str(evidence / "copilot-task-replay.png"))
            (evidence / "copilot-rendered-flow.json").write_text(json.dumps({
                "settings_saved": True, "settings_reloaded": True,
                "task_live": True, "task_history_replayed": True,
                "task_data_source": "production-shaped fixture, not a live model execution",
                "viewport": {"width": 1440, "height": 1000}, "baseline": base or None,
            }, indent=2) + "\n", encoding="utf-8")
        finally:
            if page_errors:
                print("UI_PAGE_ERRORS " + json.dumps(page_errors))
            browser.close()
