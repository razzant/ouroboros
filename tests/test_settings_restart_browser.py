"""Production Settings against an isolated server and a tiny local-model HTTP child."""

from ouroboros.local_model import RUNTIME_PROBE
import os
import json
import pathlib
import sys
import textwrap

import pytest

from tests import test_ui_smoke_playwright as smoke
from tests.settings_browser_helpers import open_entry, open_settings, settings_page
from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET, _SETTLE_RESTORE_FRAMES
from tests.ui_console_errors import PageErrors

pytest_plugins = ("tests.test_ui_smoke_playwright",)


@pytest.fixture
def settings_server(request, tmp_path, monkeypatch):
    if os.name == "nt":
        pytest.skip("POSIX test launcher; production browser behavior is shared")
    bootstrap = tmp_path / "bootstrap.py"
    fake_model = tmp_path / "fake_model.py"
    fake_model.write_text(textwrap.dedent('''\
        import http.server, json, sys
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps({"data": [{"id": "fixture", "context_window": 9999}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *args): pass
        http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
        '''), encoding="utf-8")
    bootstrap.write_text(textwrap.dedent(f'''\
        import pathlib, runpy, subprocess, sys
        sys.path.insert(0, str(pathlib.Path.cwd()))
        from ouroboros.local_model import LocalModelManager
        LocalModelManager.download_model = lambda self, source, filename: str(pathlib.Path({str(tmp_path)!r}) / filename)
        original_popen, original_run = subprocess.Popen, subprocess.run
        class Popen(original_popen):
            def __init__(self, command, **kwargs):
                if isinstance(command, list) and "ouroboros.local_model_server" in command:
                    command = [sys.executable, {str(fake_model)!r}, str(command[command.index("--port") + 1])]
                super().__init__(command, **kwargs)
        def run(command, **kwargs):
            if isinstance(command, list) and command[-1] == {RUNTIME_PROBE!r}:
                return subprocess.CompletedProcess(command, 0, "", "")
            return original_run(command, **kwargs)
        subprocess.Popen, subprocess.run = Popen, run
        sys.argv = sys.argv[1:]
        runpy.run_path(str(pathlib.Path.cwd() / 'server.py'), run_name="__main__")
        '''), encoding="utf-8")
    launcher = tmp_path / "python-fixture"
    launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{bootstrap}" "$@"\n', encoding="utf-8")
    launcher.chmod(0o700)
    monkeypatch.setattr(smoke, "_fixture_python", lambda: str(launcher))
    return request.getfixturevalue("direct_server_with_data")


RESTART_ROW = '#page-settings [data-settings-card] [data-needs="restart"]'
FOLDER = "OUROBOROS_SKILLS_REPO_PATH"


@pytest.mark.ui_browser
@pytest.mark.parametrize("engine,width", [("chromium", 1400), ("webkit", 390)])
def test_pending_survives_reconnect_draft_and_restart_request(settings_server, engine, width):
    """Saved settings that wait for a restart are one line of Ouroboros's card on the Settings home (settings_home.js),
    read from the server each time the home is shown: it survives a new page, sending the command does not end it, a
    typed value survives the reconnect, and it leaves when the saved value is the running one again."""
    from playwright.sync_api import expect, sync_playwright

    folder = settings_server["data_dir"] / "more-skills"
    folder.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch(headless=True)
        page = browser.new_page(viewport={"width": width, "height": 1000})
        errors = PageErrors(page, engine)
        page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
        page.add_init_script("""const originalSend = WebSocket.prototype.send;
            WebSocket.prototype.send = function(data) {
                if (JSON.parse(data).cmd === '/restart') { window.restartSent = true; return; }
                return originalSend.call(this, data);
            };""")
        try:
            def show(path="", row=""):
                page.wait_for_function("() => Boolean(window.__ouroSettings)", timeout=30_000)
                assert page.evaluate("([path, row]) => window.__ouroSettings.open(path, { row })", [path, row]) is not False
                page.evaluate(_SETTLE_RESTORE_FRAMES)

            def open_home():
                page.goto(settings_server["url"], wait_until="domcontentloaded")
                show("")
                expect(page.locator("#page-settings [data-settings-card]")).to_be_visible(timeout=30_000)
                # The home has read the settings document once its rules row answers.
                expect(page.locator('#page-settings [data-settings-open="rules"] [data-settings-answer]')).not_to_be_empty(timeout=30_000)

            def save_folder(value):
                show("skill-sources", FOLDER)
                field = page.locator(f"#setting-{FOLDER}")
                expect(field).to_be_visible(timeout=30_000)
                field.fill(value)
                field.evaluate("node => node.blur()")
                row = page.locator(f'#page-settings [data-setting-row="{FOLDER}"]')
                expect(row.locator("[data-setting-result]")).to_contain_text("Saved", timeout=30_000)
                return row

            def reconnect():
                # The same body reconnects, which keeps this document only once the page remembered the served
                # SHA: dropped before its first read after open, ws.js reloads it as unknown (#2432).
                page.wait_for_function("() => Boolean(window.__ouroWs?._lastSha)", timeout=30_000)
                page.evaluate("window.__testSockets[0].close()")
                page.wait_for_function("window.__testSockets.some(socket => socket.readyState === WebSocket.OPEN)")

            open_home()
            page.wait_for_function("""async (key) => {
                const data = await (await fetch('/api/settings')).json();
                return !data._meta.restart_state.unknown_keys.includes(key);
            }""", arg=FOLDER, timeout=30_000)
            expect(page.locator(RESTART_ROW)).to_have_count(0)
            row = save_folder(str(folder))
            # The server's own answer: its effect map says when the change applies, its receipt carries what the row
            # replaced, so the line offers Undo, and its restart state says what still runs.
            expect(row.locator("[data-setting-result]")).to_contain_text("Saved · applies after restart")
            expect(row.locator("[data-setting-undo]")).to_be_visible()
            expect(row.locator("[data-setting-meta]")).to_contain_text(f"after restart: {folder}")
            show("")
            expect(page.locator(RESTART_ROW)).to_be_visible(timeout=30_000)
            expect(page.locator(RESTART_ROW + " .ui-row-title")).to_have_text("Restart to apply: Skills folder")
            open_home()  # a completely new page, no local latch
            expect(page.locator(RESTART_ROW)).to_be_visible(timeout=30_000)
            # Restart asks in the shared confirmation and sends the command; the line stays until the server says so.
            page.locator(RESTART_ROW + " [data-need-action]").click()
            page.locator("[data-confirm-ok]").click()
            page.wait_for_function("window.restartSent === true")
            expect(page.locator(RESTART_ROW)).to_be_visible()
            # A value being typed survives the reconnect's re-read; the line is still there.
            show("skill-sources", "OUROBOROS_CLAWHUB_REGISTRY_URL")
            typed = page.locator("#setting-OUROBOROS_CLAWHUB_REGISTRY_URL")
            expect(typed).to_be_visible(timeout=30_000)
            stored = typed.input_value()
            typed.fill("https://clawhub.ai/api/v2")
            reconnect()
            expect(typed).to_have_value("https://clawhub.ai/api/v2")
            typed.fill(stored)  # as it was: leaving the field commits nothing
            show("")
            expect(page.locator(RESTART_ROW)).to_be_visible()
            evidence = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(settings_server["data_dir"].parent)))
            evidence.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(evidence / f"settings-card-restart-{engine}.png"))
            # The saved value is the running one again: nothing waits, and the line has left.
            row = save_folder("")
            show("")
            expect(page.locator(RESTART_ROW)).to_have_count(0, timeout=30_000)
            # A second change of the same row, then its Undo: the old value goes back through the server's per-key
            # condition (the receipt's token), and with the running value saved again nothing waits.
            row = save_folder(str(folder))
            row.locator("[data-setting-undo]").click()
            expect(page.locator(f"#setting-{FOLDER}")).to_have_value("", timeout=30_000)
            expect(row.locator("[data-setting-result]")).to_contain_text("Saved", timeout=30_000)
            show("")
            expect(page.locator(RESTART_ROW)).to_have_count(0, timeout=30_000)

            # An old launcher that does not say what controls the next start: the card says the address may need one.
            def old_launcher_metadata(route):
                response = route.fetch()
                data = response.json()
                data["_meta"]["restart_state"]["restart_source_unknown_keys"] = ["OUROBOROS_SERVER_HOST"]
                route.fulfill(response=response, json=data)

            page.route("**/api/settings", old_launcher_metadata)
            open_home()
            expect(page.locator(RESTART_ROW + " .ui-row-title")).to_have_text("The saved server address may need a restart", timeout=30_000)
            page.screenshot(path=str(evidence / f"settings-card-host-source-unknown-{engine}.png"))
            page.unroute_all(behavior="wait")
            assert errors.unexpected() == [], errors
        except Exception:
            evidence = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(settings_server["data_dir"].parent)))
            evidence.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(evidence / f"settings-card-failure-{engine}.png"))
            print(page.locator("body").inner_text()[-4000:])
            raise
        finally:
            browser.close()


@pytest.mark.ui_browser
@pytest.mark.parametrize("engine,width", [("chromium", 1400), ("webkit", 390)])
def test_a_saved_local_model_setting_waits_for_the_local_models_own_stop_then_start(settings_server, engine, width, free_tcp_port):
    """Settings › Models › Local model against the real server and a tiny local-model child: a setting saved while
    the model runs says it applies when the local model is restarted, the page and Ouroboros's card say the saved
    settings wait for Stop, then Start, no restart of Ouroboros is offered, and Stop, then Start applies them."""
    from playwright.sync_api import expect, sync_playwright

    from tests.settings_browser_helpers import HOME, commit_row, open_settings

    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch(headless=True)
        page = browser.new_page(viewport={"width": width, "height": 1000})
        errors = PageErrors(page, engine)
        try:
            page.goto(settings_server["url"], wait_until="domcontentloaded")
            level = open_settings(page, "models/local", timeout=30_000)
            level.locator('[data-setting-row="LOCAL_MODEL_SOURCE"]').wait_for(state="visible")
            assert commit_row(page, "LOCAL_MODEL_SOURCE", "fixture/model", timeout=30_000).startswith("Saved")
            assert commit_row(page, "LOCAL_MODEL_FILENAME", "fixture.gguf", timeout=30_000).startswith("Saved")
            level.locator("details[data-settings-fold] > summary").click()
            assert commit_row(page, "LOCAL_MODEL_PORT", free_tcp_port, timeout=30_000).startswith("Saved")
            level.locator("#btn-local-start").click()
            status = level.locator("#local-model-status")
            expect(status).to_contain_text("Ready", timeout=30_000)
            pending = level.locator("[data-local-model-pending]")
            expect(pending).to_be_hidden()
            # A change while it runs: the row says when it applies, in the local model's own terms.
            line = commit_row(page, "LOCAL_MODEL_CONTEXT_LENGTH", 8192, timeout=30_000)
            assert line.startswith("Saved · applies when the local model is restarted"), line
            assert "Ouroboros" not in line and "after restart" not in line
            expect(pending).to_contain_text("Stop, then Start", timeout=30_000)
            assert "Restart" not in level.inner_text()
            evidence = pathlib.Path(os.environ.get('OUROBOROS_UI_EVIDENCE_DIR', str(settings_server['data_dir'].parent)))
            evidence.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(evidence / f'local-model-pending-{engine}.png'))
            # Ouroboros's card names the local model's own action, and no restart of Ouroboros waits.
            open_settings(page, "")
            card = page.locator(f"{HOME} [data-settings-card]")
            expect(card.locator('[data-needs="local-model"]')).to_contain_text("Stop, then Start", timeout=30_000)
            assert card.locator('[data-needs="restart"]').count() == 0
            level = open_settings(page, "models/local")
            level.locator("#btn-local-stop").click()
            expect(level.locator("#local-model-status")).to_contain_text("Offline", timeout=30_000)
            level.locator("#btn-local-start").click()
            expect(level.locator("#local-model-status")).to_contain_text("Ready", timeout=30_000)
            expect(level.locator("[data-local-model-pending]")).to_be_hidden(timeout=30_000)
            page.screenshot(path=str(evidence / f'local-model-applied-{engine}.png'))
            assert errors.unexpected() == [], errors
        except Exception:
            evidence = pathlib.Path(os.environ.get('OUROBOROS_UI_EVIDENCE_DIR', str(settings_server['data_dir'].parent)))
            evidence.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(evidence / f'local-model-failure-{engine}.png'))
            print(page.locator('body').inner_text()[-4000:])
            raise
        finally:
            browser.close()


# The scene below was the last part of the first test: the Agents list (a delegation's history), driven through the
# tab frame Settings had. It belongs to the Agents page's lane (E1) and stands here until it is rewritten on that page.


@pytest.mark.ui_browser
@pytest.mark.parametrize("engine,width", [("chromium", 1400), ("webkit", 390)])
def test_agents_history_keeps_the_execution_identity_when_access_changes(settings_server, engine, width):
    from playwright.sync_api import expect, sync_playwright

    evidence = pathlib.Path(os.environ.get('OUROBOROS_UI_EVIDENCE_DIR', str(settings_server['data_dir'].parent)))
    evidence.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch(headless=True)
        page = browser.new_page(viewport={"width": width, "height": 1000})
        errors = PageErrors(page, engine)
        try:
            page.route('**/api/model-catalog*', lambda route: route.fulfill(json={'items': [], 'model_sources': [], 'errors': []}))
            page.goto(settings_server['url'], wait_until='domcontentloaded')
            # The same actual receipt consumer on the owner-facing Agents card.
            actor = {"subagent_id": "fixture-helper", "recommended_use": "Fixture helper",
                     "route": {"kind": "api_model", "target_id": "openai-compatible::mock-model"},
                     "effort": "high", "processing_preference": "standard"}
            session_actor = {"subagent_id": "fixture-session", "recommended_use": "Fixture session",
                "route": {"kind": "agent_session", "target_id": "codex=fixture-model"},
                "access": "full", "effort": "high", "processing_preference": "standard"}
            from ouroboros.subagent_history import record_last_delegation, execution_identity
            record_last_delegation(route="api_model", requested_model=actor["route"]["target_id"],
                applied_model="", run_id="browser-failure", selected_subagent_id="fixture-helper",
                drive_root=settings_server['data_dir'], occurred_at="2026-09-18T12:00:00Z",
                outcome="failed", failure_code="quota_exhausted", identity=execution_identity(actor))
            record_last_delegation(route="codex", requested_model="fixture-model", applied_model="fixture-model",
                run_id="browser-session", selected_subagent_id="fixture-session", drive_root=settings_server['data_dir'],
                occurred_at="2026-09-18T12:00:01Z", outcome="succeeded", identity=execution_identity(session_actor))
            # Neither fixture row is a reviewer: the owner confirms the empty review pool.
            response = page.request.post(settings_server['url'] + '/api/settings', data={
                "OUROBOROS_SUBAGENTS": json.dumps({"enabled": True, "items": [actor, session_actor]}),
                "allow_empty_review_pool": True})
            assert response.ok, response.text()
            open_settings(page, "agents")
            open_entry(page, "fixture-helper")
            # A past failure is history behind Details & history, never a standing caption.
            page.locator('[data-subagent-details] > summary').first.click()
            meta = page.locator('[data-subagent-last-task] dd').first
            expect(meta).to_contain_text('failed (quota_exhausted)', timeout=30_000)
            expect(meta).to_contain_text('2026-09-18T12:00:00Z')
            assert meta.evaluate("node => getComputedStyle(node).whiteSpace") == 'normal'
            assert meta.evaluate("node => node.scrollWidth <= node.clientWidth")
            meta.scroll_into_view_if_needed()
            page.screenshot(path=str(evidence / f'subagent-history-{engine}.png'))
            open_settings(page, 'agents')
            session_card = open_entry(page, 'fixture-session').locator('[data-subagent-row]')
            expect(session_card.locator('[data-subagent-field="access"]')).to_have_value('full')
            session_card.locator('[data-subagent-details] > summary').click()
            expect(session_card.locator('[data-subagent-last-task] dd')).to_contain_text('codex session')
            expect(session_card.locator('[data-subagent-last-task] dd')).not_to_contain_text('settings')
            session_card.screenshot(path=str(evidence / f'subagent-access-history-{engine}.png'))
            session_card.locator('[data-subagent-field="access"]').select_option('workspace_write')
            expect(session_card.locator('[data-subagent-last-task] dd')).to_contain_text('Earlier settings · ')
            # Editing a catalog that marks no reviewer asks the owner to confirm it again.
            page.locator('[data-editor-save]').click()
            page.locator('[data-confirm-ok]').click()
            expect(settings_page(page, 'agents/fixture-session').locator('[data-editor-result]')).to_contain_text('Saved')
            page.reload(wait_until='domcontentloaded')
            open_settings(page, 'agents')
            open_entry(page, 'fixture-session')
            expect(page.locator('[data-subagent-field="access"]')).to_have_value('workspace_write')
            assert errors.unexpected() == [], errors
        finally:
            browser.close()
