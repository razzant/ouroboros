"""The main-interface choice, end to end: Settings saves it, a restart applies it,
and the core's own built-in page follows it exactly once.

The real ``server.py`` serves its built-in page with the per-response ownership marker
and reports the applied choice as ``served_ui`` on ``/api/state``. With Chromium and
WebKit: an untouched Settings save never authors the choice; an unusable directory
keeps the built-in page (no reload, the reason shown beside the field); a directory
reloads the marked page once into the directory's own page, which then never reloads
for the core; clearing the choice there leaves that page in place until the core's
address is reopened; ``none`` reloads the marked page once into the core's "no main
interface" answer. Requires the core's selection, marker and ``served_ui`` producers.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import time
import urllib.request

import pytest

from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data
from tests.ui_console_errors import PageErrors

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
CUSTOM_TITLE = "Ouroboros custom interface"


def _save_choice_over_api(url: str, value: str) -> None:
    """The choice saved the way a separately hosted interface or script saves it."""
    request = urllib.request.Request(url + "/api/settings", data=json.dumps({"OUROBOROS_UI_DIR": value}).encode(),
                                     method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - loopback test server
        assert response.status == 200


@pytest.mark.parametrize("browser_name", ["chromium", "webkit"])
def test_interface_choice_applies_at_restart_and_the_core_page_follows_it(direct_server_with_data, browser_name):
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import expect, sync_playwright

    core = direct_server_with_data
    url = core["url"]
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", core["data_dir"].parent)) / "interface-choice"
    evidence.mkdir(parents=True, exist_ok=True)
    custom = core["data_dir"].parent / "custom-ui"
    shutil.copytree(core["repo_dir"] / "web", custom)
    index = custom / "index.html"
    index.write_text(index.read_text(encoding="utf-8").replace("<title>Ouroboros</title>", f"<title>{CUSTOM_TITLE}</title>", 1),
                     encoding="utf-8")
    with sync_playwright() as playwright:
        browser = getattr(playwright, browser_name).launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1400, "height": 1000})
            saves = []
            # Each top-level document records the address it loaded with, before
            # chat.js tidies the reload bookkeeping out with history.replaceState.
            page.add_init_script("""(() => { if (window.top !== window) return;
                const loads = JSON.parse(sessionStorage.getItem('__loads') || '[]');
                sessionStorage.setItem('__loads', JSON.stringify([...loads, location.href])); })()""")
            page.on("request", lambda request: request.method == "POST" and request.url.endswith("/api/settings")
                    and saves.append(json.loads(request.post_data or "{}")))
            errors = PageErrors(page, browser_name)
            marker = page.locator('meta[name="ouroboros-ui-owner"][content="core"]')
            connected = "() => window.__ouroWs?.isConnected?.() === true"

            def loads():
                return page.evaluate("() => JSON.parse(sessionStorage.getItem('__loads') || '[]')")

            def open_advanced():
                page.locator('[data-nav-page="settings"]').click()
                expect(page.locator("#btn-save-settings")).to_be_enabled(timeout=30_000)
                page.locator('[data-settings-tab="advanced"]').click()
                page.locator("#s-ui-dir").scroll_into_view_if_needed()

            def fill(selector, value):
                page.locator(selector).evaluate(
                    "(node, value) => {node.value=value; node.dispatchEvent(new Event('input',{bubbles:true}));}", value)

            def save():
                page.locator("#btn-save-settings").click()
                expect(page.locator("#settings-status")).to_contain_text("saved", timeout=30_000)

            def restart():
                core["restart_server"]()
                page.wait_for_function(connected, timeout=60_000)

            page.goto(url)
            page.wait_for_function(connected, timeout=30_000)
            assert marker.count() == 1, "the core marks its own built-in page"
            open_advanced()
            expect(page.locator("#s-ui-dir")).to_have_value("")
            expect(page.locator("#s-ui-dir-applied")).to_have_text("Serving now: the built-in interface.")

            # An unrelated save never authors the choice.
            fill("#s-gh-repo", "owner/interface-choice")
            save()
            assert "OUROBOROS_UI_DIR" not in saves[-1]

            # An unusable directory: saved, applied at restart as built-in with its reason; no reload.
            fill("#s-ui-dir", "/nonexistent/ouroboros-interface")
            save()
            assert saves[-1]["OUROBOROS_UI_DIR"] == "/nonexistent/ouroboros-interface"
            expect(page.locator("#settings-restart-status")).to_contain_text("Restart Ouroboros to apply")
            seen = len(loads())
            restart()
            applied = page.locator("#s-ui-dir-applied")
            expect(applied).to_have_attribute("data-tone", "warn", timeout=30_000)
            expect(applied).to_contain_text("Serving now: the built-in interface.")
            expect(applied).to_contain_text("/nonexistent/ouroboros-interface")
            time.sleep(3)
            assert len(loads()) == seen and marker.count() == 1, "an unusable directory is no reason to reload"
            page.locator("#s-ui-dir").scroll_into_view_if_needed()
            page.screenshot(path=str(evidence / f"{browser_name}-ui-dir-refused.png"))

            # A directory: the marked page reloads ONCE into the directory's own page.
            fill("#s-ui-dir", str(custom))
            save()
            restart_started = len(loads())
            core["restart_server"]()
            page.wait_for_function(f"() => document.title === {json.dumps(CUSTOM_TITLE)}", timeout=60_000)
            page.wait_for_function(connected, timeout=60_000)
            time.sleep(5)
            reloads = [a for a in loads()[restart_started:] if "_ouro_reason=ui-source" in a]
            assert len(reloads) == 1 and len(loads()) == restart_started + 1, loads()[restart_started:]
            assert marker.count() == 0, "a directory's page is its own, unmarked"
            open_advanced()
            expect(page.locator("#s-ui-dir-applied")).to_have_text("Serving now: the interface directory.")
            expect(page.locator("#s-ui-dir")).to_have_value(str(custom))

            # Cleared from the directory's page: saved blank, applied at restart; this page
            # owns its refresh and stays until the core's address is reopened.
            fill("#s-ui-dir", "")
            save()
            assert saves[-1]["OUROBOROS_UI_DIR"] == ""
            seen = len(loads())
            restart()
            time.sleep(5)
            assert len(loads()) == seen and page.title() == CUSTOM_TITLE
            expect(page.locator("#s-ui-dir-applied")).to_have_text("Serving now: the built-in interface.", timeout=30_000)
            page.goto(url)
            page.wait_for_function(connected, timeout=30_000)
            assert marker.count() == 1 and page.title() == "Ouroboros"

            # none: the marked page reloads once into the core's own answer.
            open_advanced()
            fill("#s-ui-dir", "none")
            save()
            restart_started = len(loads())
            core["restart_server"]()
            page.get_by_text("No main interface is served here").wait_for(timeout=60_000)
            assert [a for a in loads()[restart_started:] if "_ouro_reason=ui-source" in a], loads()[restart_started:]
            page.screenshot(path=str(evidence / f"{browser_name}-ui-none.png"))

            # The way back without a page: save blank elsewhere, restart, reopen.
            _save_choice_over_api(url, "")
            core["restart_server"]()
            page.goto(url)
            page.wait_for_function(connected, timeout=30_000)
            assert marker.count() == 1
            assert errors.unexpected() == [], errors  # WebKit's report of a navigation-cancelled fetch aside
        finally:
            browser.close()
