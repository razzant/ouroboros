"""Throwaway hosted Mac case: the real pywebview Cocoa backend, with DOM clicks and no OS input."""
import copy
import json
import os
import sys
import threading
import time
from wsgiref.simple_server import make_server

from tests import test_account_resources_browser as accounts

resources_ui = accounts.resources_ui
subscription_ui = accounts.subscription_ui


def test_system_wk_more(resources_ui, record_property, tmp_path):
    assert sys.platform == "darwin" and os.environ.get("RUNNER_ENVIRONMENT") == "github-hosted"
    import bottle
    import webview
    from WebKit import WKWebView

    ui = resources_ui
    status = ui["fixture"]["status"]
    status["resources"][15]["resets"]["value"] = []
    accounts.budgets(ui, 0)
    initial = copy.deepcopy(status)
    accounts.budgets(ui, 12)
    status["quota"][15]["constraints"][0]["used_ratio"] = 0.61
    enlarged = accounts.setup_browser.passive_status(status)
    grow = threading.Event()
    delivered = threading.Event()
    app = bottle.Bottle()

    @app.get("/")
    def page():
        result = bottle.static_file("index.html", root=str(accounts.setup_browser.WEB))
        result.set_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                          "style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'")
        return result

    @app.get("/static/<asset:path>")
    def asset(asset):
        return bottle.static_file(asset, root=str(accounts.setup_browser.WEB))

    @app.get("/api/<path:path>")
    def api(path):
        if path == "claudexor/status":
            if bottle.request.query.view == "quota":
                if grow.is_set():
                    delivered.set()
                    return enlarged
                return accounts.setup_browser.passive_status(initial)
            return initial
        if path == "onboarding":
            return bottle.HTTPResponse(status=204)
        return {"settings": ui["settings"], "model-catalog": ui["fixture"]["catalog"],
                "state": {"supervisor_ready": True, "active_chat_activities": [], "projects": []},
                "health": {"ok": True, "version": "fixture"}, "projects": {"projects": []},
                "chat/history": {"messages": [], "progress": []}}.get(path, {})

    samples = {}
    errors = []
    with make_server("127.0.0.1", 0, app) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        window = webview.create_window("Accounts More QA", f"http://127.0.0.1:{server.server_port}/#settings",
                                       width=1100, height=750, text_select=True)
        js = window.evaluate_js

        def wait(expression):
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if js(expression):
                    return
                time.sleep(.1)
            raise AssertionError(f"timed out waiting for {expression}")

        def measure(stage, expanded):
            sample = samples[stage] = js("""(() => {
                const level = qaLevel(), button = qaMore(), r = button.getBoundingClientRect();
                const x = r.x + r.width / 2, y = r.y + r.height / 2;
                const rowY = button.closest('.harness-account-row').getBoundingClientRect().y;
                return {x, y, rowY, top: level.scrollTop, height: level.scrollHeight, viewport: level.clientHeight,
                    hit: button.contains(document.elementFromPoint(x, y)), expanded: button.getAttribute('aria-expanded'),
                    width: innerWidth, windowHeight: innerHeight, dpr: devicePixelRatio};
            })()""")
            assert sample["hit"] and sample["expanded"] == expanded, samples
            before = samples["before"]
            assert max(abs(sample[k] - before[k]) for k in ("x", "y", "rowY", "top")) <= 2, samples
            return sample

        def observe():
            try:
                from webview.platforms.cocoa import BrowserView
                wait("Boolean(window.__ouroSettings)")
                assert isinstance(BrowserView.instances[window.uid].webview, WKWebView)
                js("window.__ouroSettings.open('accounts/claude', {map:true}); true")
                js(f"window.qaLevel = () => document.querySelector('{accounts.PROVIDER}'); "
                   f"window.qaMore = () => document.querySelector('{accounts.row(15)} [data-resources]'); true")
                wait("Boolean(qaMore()) && !qaLevel().hasAttribute('data-settings-arrive') "
                     "&& qaLevel().querySelectorAll('.harness-account-row').length === 20")
                # The sole intentional scroll places account 15's More near the bottom of the viewport.
                js("""{const r = qaMore().getBoundingClientRect(), level = qaLevel();
                    level.scrollTop += r.y + r.height / 2 - Math.min(innerHeight - 48, level.getBoundingClientRect().bottom - 24);}
                    true""")
                time.sleep(.2)
                before = measure("before", "false")
                assert before["top"] > before["viewport"]
                js("qaMore().click()")
                wait("qaMore().getAttribute('aria-expanded') === 'true'")
                time.sleep(.15)
                opened = measure("opened", "true")
                assert opened["height"] > before["height"], samples
                time.sleep(1)
                measure("one_second", "true")
                grow.set()
                wait("qaMore().closest('.harness-account-row').innerText.includes('Workspace budget 11') "
                     "&& qaMore().closest('.harness-account-row').innerText.includes('61% used')")
                assert delivered.is_set()
                time.sleep(.15)
                changed = measure("after_poll", "true")
                assert changed["height"] - opened["height"] > opened["viewport"], samples
                js("qaMore().click()")
                wait("!qaMore().closest('.harness-account-row').querySelector('.account-resource-panel')")
                time.sleep(1)
                measure("closed", "false")
            except Exception as exc:
                errors.append(exc)
            finally:
                window.destroy()

        try:
            webview.start(observe, gui="cocoa", private_mode=True, storage_path=str(tmp_path))
        finally:
            server.shutdown()
            record_property("system_wk_dom_click_samples", json.dumps(samples))
    if errors:
        raise errors[0]
    assert "closed" in samples
