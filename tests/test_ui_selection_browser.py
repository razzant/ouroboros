"""The main-interface choice, end to end: its row in Settings saves it (This computer › Advanced, Interface
folder), a restart applies it, and the core's own built-in page follows it exactly once.

The real ``server.py`` serves its built-in page with the per-response ownership marker
and reports the applied choice as ``served_ui`` on ``/api/state``. With Chromium and
WebKit: saving another row never authors the choice; an unusable directory
keeps the built-in page (no reload, the reason shown in the row as a warning); a directory
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

from tests.settings_browser_helpers import commit_row, open_settings, setting_row
from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data
from tests.ui_console_errors import PageErrors

pytestmark = [pytest.mark.ui_browser]
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
                open_settings(page, "computer/advanced", timeout=30_000)
                field("OUROBOROS_UI_DIR").scroll_into_view_if_needed()

            def field(row_id):
                return setting_row(page, row_id).locator("[data-setting-control]")

            def served():
                return setting_row(page, "OUROBOROS_UI_DIR").locator("[data-served-ui]")

            def save(row_id, value):
                """Type a row's value and leave the field, which is its save; the row's result line is the answer."""
                return commit_row(page, row_id, value, timeout=30_000)

            def meta(row_id):
                return setting_row(page, row_id).locator("[data-setting-meta]")

            def restart():
                core["restart_server"]()
                page.wait_for_function(connected, timeout=60_000)

            page.goto(url)
            page.wait_for_function(connected, timeout=30_000)
            assert marker.count() == 1, "the core marks its own built-in page"
            open_advanced()
            expect(field("OUROBOROS_UI_DIR")).to_have_value("")
            expect(served()).to_have_text("Serving now: the built-in interface.")

            # Another row's save never authors the choice: a row writes its own key and nothing else.
            # The server's receipt and timing, word for word: what the value replaced (an empty one in the row's own word).
            assert save("OUROBOROS_EXTRA_CA_BUNDLE", "/nonexistent/extra-ca.pem") == "Saved · applies to new tasks · was none · Undo"
            assert saves[-1] == {"OUROBOROS_EXTRA_CA_BUNDLE": "/nonexistent/extra-ca.pem"}
            assert save("OUROBOROS_EXTRA_CA_BUNDLE", "") == "Saved · applies to new tasks · was /nonexistent/extra-ca.pem · Undo"
            assert "OUROBOROS_UI_DIR" not in saves[-1]

            # An unusable directory: saved, applied at restart as built-in with its reason; no reload.
            assert save("OUROBOROS_UI_DIR", "/nonexistent/ouroboros-interface") == "Saved · applies after restart · Undo"
            # What still runs beside what was saved, from the server's `restart_state.running`.
            expect(meta("OUROBOROS_UI_DIR")).to_have_text(
                "Running: the built-in interface · after restart: /nonexistent/ouroboros-interface", timeout=30_000)
            assert saves[-1] == {"OUROBOROS_UI_DIR": "/nonexistent/ouroboros-interface"}
            seen = len(loads())
            restart()
            open_advanced()  # the page reads what is served whenever it is shown
            expect(served()).to_have_attribute("data-tone", "warn", timeout=30_000)
            expect(served()).to_contain_text("Serving now: the built-in interface.")
            expect(served()).to_contain_text("/nonexistent/ouroboros-interface")
            time.sleep(3)
            assert len(loads()) == seen and marker.count() == 1, "an unusable directory is no reason to reload"
            field("OUROBOROS_UI_DIR").scroll_into_view_if_needed()
            page.screenshot(path=str(evidence / f"{browser_name}-ui-dir-refused.png"))

            # A directory: the marked page reloads ONCE into the directory's own page.
            assert save("OUROBOROS_UI_DIR", str(custom)) == "Saved · applies after restart · Undo"
            restart_started = len(loads())
            core["restart_server"]()
            page.wait_for_function(f"() => document.title === {json.dumps(CUSTOM_TITLE)}", timeout=60_000)
            page.wait_for_function(connected, timeout=60_000)
            time.sleep(5)
            reloads = [a for a in loads()[restart_started:] if "_ouro_reason=ui-source" in a]
            assert len(reloads) == 1 and len(loads()) == restart_started + 1, loads()[restart_started:]
            assert marker.count() == 0, "a directory's page is its own, unmarked"
            open_advanced()
            expect(served()).to_have_text("Serving now: the interface directory.")
            expect(field("OUROBOROS_UI_DIR")).to_have_value(str(custom))

            # Cleared from the directory's page: saved blank, applied at restart; this page
            # owns its refresh and stays until the core's address is reopened.
            assert save("OUROBOROS_UI_DIR", "") == "Saved · applies after restart · Undo"
            expect(meta("OUROBOROS_UI_DIR")).to_have_text(f"Running: {custom} · after restart: the built-in interface", timeout=30_000)
            assert saves[-1] == {"OUROBOROS_UI_DIR": ""}
            seen = len(loads())
            restart()
            time.sleep(5)
            assert len(loads()) == seen and page.title() == CUSTOM_TITLE
            open_advanced()
            expect(served()).to_have_text("Serving now: the built-in interface.", timeout=30_000)
            page.goto(url)
            page.wait_for_function(connected, timeout=30_000)
            assert marker.count() == 1 and page.title() == "Ouroboros"

            # none: the marked page reloads once into the core's own answer.
            open_advanced()
            assert save("OUROBOROS_UI_DIR", "none") == "Saved · applies after restart · Undo"
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
