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
from tests.settings_section_helpers import show_card
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
        import http.server, json, os, pathlib, sys
        updated = pathlib.Path(__file__).with_suffix(".updated")
        if sys.argv[1] == "--upgrade":
            updated.touch()
            sys.exit(0)
        if pathlib.Path(sys.argv[2]).name == "new-format.gguf" and not updated.exists():
            sys.exit("Fixture model header could not be loaded")
        from ouroboros.local_model_server import LOCAL_LAUNCH_ENV, LOCAL_LAUNCH_HEADER
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps({"data": [{"id": "fixture", "context_window": 9999}]}).encode()
                self.send_response(200)
                self.send_header(LOCAL_LAUNCH_HEADER, os.environ[LOCAL_LAUNCH_ENV])
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *args): pass
        http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
        '''), encoding="utf-8")
    bootstrap.write_text(textwrap.dedent(f'''\
        import pathlib, runpy, subprocess, sys, time
        sys.path.insert(0, str(pathlib.Path.cwd()))
        import huggingface_hub
        from ouroboros import local_model
        model_dir = pathlib.Path({str(tmp_path)!r})
        def download(*, filename, tqdm_class, **kwargs):
            if filename == "held-transfer.gguf":
                with tqdm_class(total=4, disable=False) as progress:
                    progress.update(1)
                    deadline = time.monotonic() + 90
                    while not (model_dir / "release-transfer").exists():
                        if time.monotonic() >= deadline:
                            raise TimeoutError("fixture download was never released")
                        time.sleep(0.05)
                    progress.update(3)
            return str(model_dir / filename)
        huggingface_hub.list_repo_files = lambda repo: []
        huggingface_hub.hf_hub_download = download
        local_model._get_install_command = lambda: [sys.executable, {str(fake_model)!r}, "--upgrade"]
        original_popen, original_run = subprocess.Popen, subprocess.run
        class Popen(original_popen):
            def __init__(self, command, **kwargs):
                if isinstance(command, list) and "ouroboros.local_model_server" in command:
                    command = [sys.executable, {str(fake_model)!r}, command[command.index("--port") + 1],
                               command[command.index("--model") + 1]]
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
                if path:
                    assert page.evaluate("([path, row]) => window.__ouroSettings.open(path, { row })", [path, row]) is not False
                else:  # Ouroboros's card: on the list in one column, beside a page the pane its row opens
                    show_card(page, timeout=30_000)
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

    from tests.settings_browser_helpers import commit_row, open_settings

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
            card = show_card(page, timeout=30_000)
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


@pytest.mark.ui_browser
@pytest.mark.parametrize("engine,width", [("chromium", 1400), ("webkit", 390)])
def test_stop_during_download_starts_an_available_file_before_the_transfer_returns(settings_server, engine, width, free_tcp_port, tmp_path):
    from playwright.sync_api import expect, sync_playwright
    from tests.settings_browser_helpers import commit_row

    available = tmp_path / "available.gguf"
    available.touch()
    release = tmp_path / "release-transfer"
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": width, "height": 900})
            page.goto(settings_server["url"], wait_until="domcontentloaded")
            level = open_settings(page, "models/local", timeout=30_000)
            commit_row(page, "LOCAL_MODEL_SOURCE", "fixture/repo")
            commit_row(page, "LOCAL_MODEL_FILENAME", "held-transfer.gguf")
            level.locator("details[data-settings-fold] > summary").click()
            commit_row(page, "LOCAL_MODEL_PORT", free_tcp_port)
            start = level.get_by_role("button", name="Start", exact=True)
            stop = level.get_by_role("button", name="Stop", exact=True)
            status = level.locator("#local-model-status")
            start.click()
            expect(status).to_contain_text("Downloading · 25%", timeout=30_000)
            expect(start).to_be_disabled()
            stop.click()
            expect(status).to_contain_text("Offline", timeout=30_000)
            commit_row(page, "LOCAL_MODEL_SOURCE", str(available))
            start.click()
            expect(status).to_contain_text("Ready", timeout=30_000)
            with page.expect_response(lambda res: res.url.endswith("/api/local-model/start")
                                      and res.request.post_data_json["source"] == "fixture/repo", timeout=30_000) as old:
                release.touch()
            assert old.value.status == 409
            current = page.request.get(f"{settings_server['url']}/api/local-model/status").json()
            assert (current["status"], current["model_path"]) == ("ready", str(available))
            expect(level.locator("#local-model-action-status")).to_have_text("")
            stop.click()
            expect(status).to_contain_text("Offline", timeout=30_000)
        finally:
            release.touch()
            browser.close()


@pytest.mark.ui_browser
@pytest.mark.parametrize("engine,width,stop_before_reply", [
    pytest.param("chromium", 1400, False, id="chromium-1400"),
    pytest.param("webkit", 390, False, id="webkit-390"),
    pytest.param("chromium", 1400, True, id="chromium-1400-stop-before-reply"),
    pytest.param("webkit", 390, True, id="webkit-390-stop-before-reply"),
])
def test_local_model_update_after_startup_exit_requires_a_click(settings_server, engine, width, free_tcp_port, stop_before_reply):
    from playwright.sync_api import expect, sync_playwright
    from tests.settings_browser_helpers import commit_row

    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": width, "height": 900})
            posts = []
            page.on("request", lambda req: posts.append(req.url.rsplit("/", 1)[-1])
                    if req.method == "POST" and "/api/local-model/" in req.url else None)
            page.route("**/api/local-model/status", lambda route: route.fulfill(json={
                "status": "error", "runtime_status": "ok", "error": "Fixture model header could not be loaded"}))
            page.goto(settings_server["url"], wait_until="domcontentloaded")
            level = open_settings(page, "models/local", timeout=30_000)
            install = level.locator("#btn-local-install-runtime")
            status = level.locator("#local-model-status")
            expect(status).to_contain_text("Fixture model header could not be loaded", timeout=30_000)
            expect(install).to_be_hidden()
            page.unroute("**/api/local-model/status")
            commit_row(page, "LOCAL_MODEL_SOURCE", "fixture/model")
            commit_row(page, "LOCAL_MODEL_FILENAME", "new-format.gguf")
            level.locator("details[data-settings-fold] > summary").click()
            commit_row(page, "LOCAL_MODEL_PORT", free_tcp_port)
            level.locator("#btn-local-start").click()
            expect(status).to_contain_text("Fixture model header could not be loaded", timeout=30_000)
            expect(install).to_be_visible(timeout=30_000)
            expect(install).to_have_text("Update the local runtime")
            expect(status).to_contain_text("A newer runtime may support this model.")
            assert posts == ["start"]
            stop = level.locator("#btn-local-stop")
            expected_posts = ["start", "install-runtime"]
            if stop_before_reply:
                def next_polled_status():
                    return page.wait_for_event("response", predicate=lambda res: res.url.endswith("/api/local-model/status")).json()

                # Hold the browser's POST before admission, then forward it and keep its real reply until after Stop.
                held = []
                page.route("**/api/local-model/install-runtime", lambda route: held.append(route))
                with page.expect_request("**/api/local-model/install-runtime"):
                    install.click()
                assert next_polled_status()["runtime_status"] == "ok"
                expect(stop).to_be_disabled()
                route, = held
                reply = route.fetch()
                receipt = reply.json()
                assert reply.ok and receipt["status"] == "installing" and receipt["install_id"]
                expect(status).to_contain_text("runtime installed", timeout=30_000)
                expect(stop).to_be_enabled()
                with page.expect_response("**/api/local-model/stop") as stopped:
                    stop.click()
                assert stopped.value.json() == {"status": "stopped"}
                expect(status).to_contain_text("Offline", timeout=30_000)
                with page.expect_response("**/api/local-model/install-runtime") as delivered:
                    route.fulfill(response=reply)
                assert delivered.value.json() == receipt
                assert next_polled_status()["status"] == "offline"
                expect(level.locator("#local-model-action-status")).to_have_text("Automatic start cancelled.")
                expected_posts.append("stop")
                assert posts == expected_posts
                level.locator("#btn-local-start").click()
            else:
                evidence = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(settings_server["data_dir"].parent)))
                evidence.mkdir(parents=True, exist_ok=True)
                install.scroll_into_view_if_needed()
                page.screenshot(path=str(evidence / f"local-model-update-{engine}.png"))
                install.click()
            expect(status).to_contain_text("Ready", timeout=30_000)
            expect(install).to_be_hidden()
            assert posts == [*expected_posts, "start"]
            stop.click()
            expect(status).to_contain_text("Offline", timeout=30_000)
        finally:
            browser.close()


@pytest.mark.ui_browser
@pytest.mark.parametrize("engine,width", [("chromium", 1400), ("webkit", 390)])
def test_a_stop_from_another_client_ends_the_start_an_install_offered_even_across_a_later_install(
        settings_server, engine, width, free_tcp_port):
    """The page queued the Start its Update offered, with the ``install_id`` that install's reply gave it. Another client
    then Stops and admits a fresh install; when the page's next status read finds ``install_ok`` the server refuses the
    queued Start (409), the install's own result stays, and the page's explicit Start works."""
    import time

    from playwright.sync_api import expect, sync_playwright

    from tests.settings_browser_helpers import commit_row

    url = settings_server["url"]
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": width, "height": 900})
            page.goto(url, wait_until="domcontentloaded")
            level = open_settings(page, "models/local", timeout=30_000)
            install, status = level.locator("#btn-local-install-runtime"), level.locator("#local-model-status")
            commit_row(page, "LOCAL_MODEL_SOURCE", "fixture/model")
            commit_row(page, "LOCAL_MODEL_FILENAME", "new-format.gguf")
            level.locator("details[data-settings-fold] > summary").click()
            commit_row(page, "LOCAL_MODEL_PORT", free_tcp_port)
            level.locator("#btn-local-start").click()
            expect(install).to_have_text("Update the local runtime", timeout=30_000)
            held, gate = [], {"open": False}  # the page's status reads wait while the other client acts
            page.route("**/api/local-model/status", lambda route: route.continue_() if gate["open"] else held.append(route))
            with page.expect_response("**/api/local-model/install-runtime") as admitted:
                install.click()
            first = admitted.value.json()["install_id"]
            # The reply is handled (its Start queued) once the action line clears.
            expect(level.locator("#local-model-action-status")).to_have_text("", timeout=30_000)

            def installed():
                deadline = time.monotonic() + 30
                while page.request.get(f"{url}/api/local-model/status").json()["runtime_status"] != "install_ok":
                    assert time.monotonic() < deadline, "the fixture's runtime update never finished"
                    page.wait_for_timeout(100)

            installed()
            assert page.request.post(f"{url}/api/local-model/stop").json() == {"status": "stopped"}
            second = page.request.post(f"{url}/api/local-model/install-runtime").json()
            assert second["status"] == "installing" and second["install_id"] != first
            installed()
            gate["open"] = True
            with page.expect_response("**/api/local-model/start") as attempt:
                for route in held:
                    route.continue_()
            assert attempt.value.request.post_data_json["install_id"] == first
            assert attempt.value.status == 409 and "cancelled" in attempt.value.json()["error"]
            expect(level.locator("#local-model-action-status")).to_contain_text("Not started", timeout=30_000)
            expect(status).to_contain_text("Offline", timeout=30_000)
            assert page.request.get(f"{url}/api/local-model/status").json()["runtime_status"] == "install_ok"
            level.locator("#btn-local-start").click()
            expect(status).to_contain_text("Ready", timeout=30_000)
            level.locator("#btn-local-stop").click()
            expect(status).to_contain_text("Offline", timeout=30_000)
        finally:
            browser.close()


@pytest.mark.ui_browser
@pytest.mark.parametrize("engine,width", [("chromium", 1400), ("webkit", 390)])
def test_managed_deployment_never_offers_the_runtime_install(settings_server, engine, width, free_tcp_port):
    """A managed deployment's administrator owns the program's interpreter (POST /api/local-model/install-runtime
    refuses managed_by_deployment), so the local model page says who installs the runtime and never offers Install or
    Update on any path: the note shows, a startup exit keeps its error and the hedged newer-runtime fact, and no
    install-runtime request leaves the page."""
    from playwright.sync_api import expect, sync_playwright

    from tests.settings_browser_helpers import commit_row

    def managed_settings(route):
        response = route.fetch()
        if route.request.method == "POST":
            route.fulfill(response=response)
            return
        data = response.json()
        data.setdefault("_meta", {}).setdefault("policy_state", {})["deployment"] = {
            "source": "file", "version": 1, "digest": "fixture", "access_max": None, "min_coverage": None,
            "checker": None, "allowed_exits": [], "organization_rules": False, "extensions": None,
        }
        route.fulfill(response=response, json=data)

    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": width, "height": 900})
            posts = []
            page.on("request", lambda req: posts.append(req.url.rsplit("/", 1)[-1])
                    if req.method == "POST" and "/api/local-model/" in req.url else None)
            page.route("**/api/settings", managed_settings)
            page.goto(settings_server["url"], wait_until="domcontentloaded")
            level = open_settings(page, "models/local", timeout=30_000)
            install = level.locator("#btn-local-install-runtime")
            status = level.locator("#local-model-status")
            note = level.locator("[data-local-model-managed]")
            expect(note).to_be_visible(timeout=30_000)
            expect(note).to_contain_text("Your administrator installs and updates the local runtime")
            expect(install).to_be_hidden()
            commit_row(page, "LOCAL_MODEL_SOURCE", "fixture/model")
            commit_row(page, "LOCAL_MODEL_FILENAME", "new-format.gguf")
            level.locator("details[data-settings-fold] > summary").click()
            commit_row(page, "LOCAL_MODEL_PORT", free_tcp_port)
            level.locator("#btn-local-start").click()
            # The owned model process exits during startup: the error and the hedged runtime fact stay; the
            # update control a click would only see refused never appears.
            expect(status).to_contain_text("Fixture model header could not be loaded", timeout=30_000)
            expect(status).to_contain_text("A newer runtime may support this model.")
            expect(note).to_be_visible()
            expect(install).to_be_hidden()
            assert posts == ["start"], posts
            evidence = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(settings_server["data_dir"].parent)))
            evidence.mkdir(parents=True, exist_ok=True)
            note.scroll_into_view_if_needed()
            page.screenshot(path=str(evidence / f"local-model-managed-{engine}.png"))
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
