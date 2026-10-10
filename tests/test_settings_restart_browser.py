"""Production Settings against an isolated server and a tiny local-model HTTP child."""

import os
import json
import pathlib
import sys
import textwrap

import pytest

from tests import test_ui_smoke_playwright as smoke
from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET, _SETTLE_RESTORE_FRAMES

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
            if isinstance(command, list) and command[-1] == "import llama_cpp":
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


@pytest.mark.serial
@pytest.mark.ui_browser
@pytest.mark.parametrize("engine,width", [("chromium", 1400), ("webkit", 390)])
def test_pending_survives_reconnect_draft_and_restart_request(settings_server, engine, width, free_tcp_port):
    from playwright.sync_api import expect, sync_playwright

    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch(headless=True)
        page = browser.new_page(viewport={"width": width, "height": 1000})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
        page.add_init_script("""const originalSend = WebSocket.prototype.send;
            WebSocket.prototype.send = function(data) {
                if (JSON.parse(data).cmd === '/restart') { window.restartSent = true; return; }
                return originalSend.call(this, data);
            };""")
        try:
            def open_settings():
                page.goto(settings_server["url"], wait_until="domcontentloaded")
                if width < 980:
                    page.locator('#page-chat [data-mobile-nav-toggle]').click()
                page.locator('[data-nav-page="settings"]').click()
                expect(page.locator('#btn-save-settings')).to_be_enabled(timeout=30_000)
                page.evaluate(_SETTLE_RESTORE_FRAMES)

            def fill_value(selector, value):
                page.locator(selector).evaluate("(node, value) => {node.value=value; node.dispatchEvent(new Event('input',{bubbles:true}));}", str(value))

            def save():
                page.locator('#btn-save-settings').click()
                page.wait_for_function("!document.querySelector('#btn-save-settings').disabled || document.querySelector('[data-confirm-ok]')")
                if page.locator('[data-confirm-ok]').is_visible():
                    page.locator('[data-confirm-ok]').click()
                expect(page.locator('#btn-save-settings')).to_be_enabled(timeout=30_000)
                expect(page.locator('#settings-status')).to_contain_text('saved', timeout=30_000)

            open_settings()
            # Worker application is published only after its real pool starts.
            page.wait_for_function("""async () => {
                const data = await (await fetch('/api/settings')).json();
                return !data._meta.restart_state.unknown_keys.includes('OUROBOROS_MAX_WORKERS');
            }""", timeout=30_000)
            fill_value('#s-workers', 2)
            save()
            expect(page.locator('#btn-restart-now')).to_be_visible()
            open_settings()  # a completely new page, no local latch
            expect(page.locator('#btn-restart-now')).to_be_visible()
            page.locator('#btn-restart-now').click()
            page.locator('[data-confirm-ok]').click()
            page.wait_for_function('window.restartSent === true')
            expect(page.locator('#btn-restart-now')).to_be_visible()
            fill_value('#s-workers', 3)  # preserve a draft over reconnect metadata
            page.evaluate("window.__testSockets[0].close()")
            page.wait_for_function("window.__testSockets.some(socket => socket.readyState === WebSocket.OPEN)")
            expect(page.locator('#s-workers')).to_have_value('3')
            expect(page.locator('#btn-restart-now')).to_be_visible()
            fill_value('#s-workers', 1)
            save()
            expect(page.locator('#btn-restart-now')).to_be_hidden()

            # The real direct-server fixture has an explicit loopback env host.
            fill_value('#s-server-host', '0.0.0.0')
            save()
            expect(page.locator('#settings-restart-status')).to_contain_text('launch configuration overrides')
            expect(page.locator('#settings-status')).to_have_text('Settings saved')
            expect(page.locator('#btn-restart-now')).to_be_hidden()
            evidence = pathlib.Path(os.environ.get('OUROBOROS_UI_EVIDENCE_DIR', str(settings_server['data_dir'].parent)))
            evidence.mkdir(parents=True, exist_ok=True)
            page.locator('#settings-restart-status').scroll_into_view_if_needed()
            page.screenshot(path=str(evidence / f'settings-host-override-{engine}.png'))

            # Exercise the same consumer with an old launcher's missing source;
            # producer/PID-record matching is covered by test_settings_host_source.
            def old_launcher_metadata(route):
                response = route.fetch()
                data = response.json()
                state = data['_meta']['restart_state']
                state['restart_source_unknown_keys'] = ['OUROBOROS_SERVER_HOST']
                state['summary'] = ('Saved server host differs from the running listener. This launcher did not report '
                                    'whether a launch override controls the next start; Restart may apply the saved host.')
                route.fulfill(response=response, json=data)

            page.route('**/api/settings', old_launcher_metadata)
            open_settings()
            expect(page.locator('#settings-restart-status')).to_contain_text('Restart may apply')
            expect(page.locator('#btn-restart-now')).to_be_visible()
            fill_value('#s-workers', 3)
            page.evaluate("window.__testSockets[0].close()")
            page.wait_for_function("window.__testSockets.some(socket => socket.readyState === WebSocket.OPEN)")
            expect(page.locator('#s-workers')).to_have_value('3')
            expect(page.locator('#btn-restart-now')).to_be_visible()
            page.locator('#settings-restart-status').scroll_into_view_if_needed()
            page.screenshot(path=str(evidence / f'settings-host-source-unknown-{engine}.png'))
            # Reconnect refresh may still be inside fetch/fulfill: drain it
            # before removing interception, rather than continuing its route twice.
            page.unroute_all(behavior='wait')
            fill_value('#s-workers', 1)
            fill_value('#s-server-host', '127.0.0.1')
            save()
            expect(page.locator('#btn-restart-now')).to_be_hidden()

            fill_value('#s-local-source', 'fixture/model')
            fill_value('#s-local-filename', 'fixture.gguf')
            fill_value('#s-local-port', free_tcp_port)
            fill_value('#s-local-ctx', 0)
            save()
            page.locator('[data-settings-tab="advanced"]').click()
            page.locator('#btn-local-start').click()
            expect(page.locator('#local-model-status')).to_contain_text('Ready', timeout=30_000)
            fill_value('#s-local-ctx', 8192)
            save()
            expect(page.locator('#settings-restart-status')).to_contain_text('Stop, then Start')
            expect(page.locator('#btn-restart-now')).to_be_hidden()
            evidence = pathlib.Path(os.environ.get('OUROBOROS_UI_EVIDENCE_DIR', str(settings_server['data_dir'].parent)))
            evidence.mkdir(parents=True, exist_ok=True)
            page.locator('#settings-restart-status').scroll_into_view_if_needed()
            page.screenshot(path=str(evidence / f'settings-applied-pending-{engine}.png'))
            page.locator('#btn-local-stop').click()
            expect(page.locator('#local-model-status')).to_contain_text('Offline', timeout=30_000)
            page.locator('#btn-local-start').click()
            expect(page.locator('#local-model-status')).to_contain_text('Ready', timeout=30_000)
            expect(page.locator('#settings-restart-status')).to_be_hidden(timeout=30_000)
            page.screenshot(path=str(evidence / f'settings-applied-ready-{engine}.png'))
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
            open_settings()
            page.locator('[data-settings-tab="agents"]').click()
            # A past failure is history behind Details & history, never a standing caption.
            page.locator('[data-subagent-details] > summary').first.click()
            meta = page.locator('[data-subagent-last-task] dd').first
            expect(meta).to_contain_text('failed (quota_exhausted)', timeout=30_000)
            expect(meta).to_contain_text('2026-09-18T12:00:00Z')
            assert meta.evaluate("node => getComputedStyle(node).whiteSpace") == 'normal'
            assert meta.evaluate("node => node.scrollWidth <= node.clientWidth")
            meta.scroll_into_view_if_needed()
            page.screenshot(path=str(evidence / f'subagent-history-{engine}.png'))
            session_card = page.locator('[data-subagent-row]').nth(1)
            expect(session_card.locator('[data-subagent-field="access"]')).to_have_value('full')
            session_card.locator('[data-subagent-details] > summary').click()
            expect(session_card.locator('[data-subagent-last-task] dd')).to_contain_text('codex session')
            expect(session_card.locator('[data-subagent-last-task] dd')).not_to_contain_text('settings')
            session_card.screenshot(path=str(evidence / f'subagent-access-history-{engine}.png'))
            session_card.locator('[data-subagent-field="access"]').select_option('workspace_write')
            expect(session_card.locator('[data-subagent-last-task] dd')).to_contain_text('Earlier settings · ')
            # Editing a catalog that marks no reviewer asks the owner to confirm it again.
            page.locator('[data-review-pool-allow-empty]').check()
            save()
            open_settings()
            page.locator('[data-settings-tab="agents"]').click()
            expect(page.locator('[data-subagent-field="access"]')).to_have_value('workspace_write')
            assert errors == []
        except Exception:
            evidence = pathlib.Path(os.environ.get('OUROBOROS_UI_EVIDENCE_DIR', str(settings_server['data_dir'].parent)))
            evidence.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(evidence / f'settings-failure-{engine}.png'))
            print(page.locator('body').inner_text()[-4000:])
            raise
        finally:
            browser.close()
