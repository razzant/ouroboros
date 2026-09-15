"""Skills read/recovery through production controllers, with no running backend."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlsplit

import pytest


REPO = Path(__file__).resolve().parents[1]
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


@pytest.fixture
def skills_browser():
    if os.environ.get("OUROBOROS_RUN_UI_SMOKE") != "1":
        pytest.skip("set OUROBOROS_RUN_UI_SMOKE=1 to run browser UI smoke")
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        yield playwright


def _open_skills(page):
    """Mount the real page; stub only HTTP answers, never controller/renderer code."""
    installed = {
        "name": "weather", "source": "external", "description": "Local forecast",
        "version": "1.0", "enabled": False, "review_status": "pending",
        "required_keys": [], "missing_keys": [], "missing_permissions": [],
    }
    catalog = {"slug": "weather", "sanitized_name": "weather",
               "display_name": "Weather", "latest_version": "1.0",
               "summary": "Local forecast", "official": True}
    state = {
        "extensions": [], "extensions_error": "", "installed_error": "",
        "catalog_error": "", "hub_error": "", "requests": [],
        "hold_paths": set(), "pending": [],
        "state_response": {"skills_repo_configured": False, "github_token_configured": True},
        "queue_response": {"events": [], "active": None},
    }

    def route(request_route):
        request = request_route.request
        path = urlsplit(request.url).path
        if path.startswith("/api/"):
            state["requests"].append((request.method, path))
            if path in state["hold_paths"]:
                state["pending"].append((path, request_route))
                return
        if path == "/":
            sheets = "".join(
                f'<link rel="stylesheet" href="/static/{name}">'
                for name in ("ui.css", "style.css") if (REPO / "web" / name).exists()
            )
            request_route.fulfill(content_type="text/html", body=(
                f'<!doctype html><html><head>{sheets}</head><body><main id="content"></main>'
                '<script type="module">'
                'import {initSkills} from "/static/modules/skills.js";'
                'window.skillsView=initSkills({}); document.querySelector("#page-skills").classList.add("active");'
                'window.dispatchEvent(new CustomEvent("ouro:page-shown",{detail:{page:"skills"}}));'
                '</script></body></html>'
            ))
            return
        if path.startswith("/static/"):
            source = REPO / "web" / path.removeprefix("/static/")
            if source.is_file() and source.resolve().is_relative_to((REPO / "web").resolve()):
                mime = "text/css" if source.suffix == ".css" else "text/javascript"
                request_route.fulfill(content_type=mime, body=source.read_bytes())
            else:
                request_route.fulfill(status=404, body="Missing fixture asset")
            return
        errors = {
            "/api/extensions": state["extensions_error"],
            "/api/marketplace/clawhub/installed": state["installed_error"],
            "/api/marketplace/clawhub/search": state["catalog_error"],
            "/api/marketplace/ouroboroshub/catalog": state["hub_error"],
        }
        if errors.get(path):
            request_route.fulfill(status=503, content_type="application/json",
                                  body=json.dumps({"error": errors[path]}))
            return
        responses = {
            "/api/state": state["state_response"],
            "/api/extensions": {"skills": state["extensions"], "live": {}},
            "/api/skills/lifecycle-queue": state["queue_response"],
            "/api/marketplace/clawhub/installed": {"skills": []},
            "/api/marketplace/clawhub/search": {"results": [catalog], "next_cursor": ""},
            "/api/marketplace/ouroboroshub/catalog": {"results": [catalog]},
        }
        if path not in responses:
            request_route.fulfill(status=404, content_type="application/json",
                                  body=json.dumps({"error": f"Unexpected fixture request: {path}"}))
            return
        request_route.fulfill(content_type="application/json", body=json.dumps(responses[path]))

    page.route("**/*", route)
    return state, installed


def _release_read(state, path, payload, status=200):
    index = next(index for index, (pending_path, _) in enumerate(state["pending"]) if pending_path == path)
    _, request_route = state["pending"].pop(index)
    request_route.fulfill(status=status, content_type="application/json", body=json.dumps(payload))


def _close_pending_browser(browser, state):
    try:
        for _, request_route in state['pending']:
            request_route.abort()
        state['pending'].clear()
    finally:
        browser.close()


def _refresh(page):
    refresh = page.locator("#skills-refresh")
    refresh.click()
    page.wait_for_function("() => !document.querySelector('#skills-refresh').disabled")


def _capture(page, browser_name, width, state):
    if location := os.environ.get("OUROBOROS_UI_EVIDENCE_OUT"):
        target = Path(location)
        target.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(target / f"skills-{browser_name}-{width}-{state}.png"))


@pytest.mark.parametrize("browser_name,width", [("chromium", 1440), ("webkit", 390)])
def test_presence_workspace_save_and_runtime_reset_keep_owner_folder(skills_browser, browser_name, width):
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    try:
        page = browser.new_page(viewport={"width": width, "height": 900})
        state, installed = _open_skills(page)
        runtime = {
            "defaults": {"model_slot": "main", "inline_max_rounds": 10},
            "overrides": {}, "workspace_root": "/work/current",
            "state_fingerprint": "a" * 64,
        }
        state["extensions"] = [{**installed, "enabled": True, "review_status": "clean",
            "review_gate": {"executable_review": True}, "presence_runtime": runtime}]
        writes = []

        def save(route):
            body = route.request.post_data_json
            assert body["expected_state_fingerprint"] == runtime["state_fingerprint"]
            writes.append(body)
            runtime["overrides"] = body["runtime_overrides"]
            if "workspace_root" in body:
                runtime["workspace_root"] = body["workspace_root"]
            runtime["state_fingerprint"] = str(len(writes)) * 64
            route.fulfill(content_type="application/json", body=json.dumps({
                "ok": True, "skill": "weather", "presence_runtime": runtime,
            }))

        page.route("**/api/owner/skills/weather/presence-runtime", save)
        page.goto("http://skills.test/", wait_until="networkidle")
        card = page.locator('.skills-card[data-skill="weather"]')
        card.locator('.skills-details > summary').click()
        form = card.locator('[data-presence-runtime-form]')
        folder = form.locator('[name="workspace_root"]')
        assert folder.input_value() == "/work/current"
        selected = "/work/shared documents/quarterly reports"
        folder.fill(selected)
        form.locator('[name="model_slot"]').select_option("light")
        _capture(page, browser_name, width, "presence-workspace-edit")
        assert folder.bounding_box()["width"] > 200
        form.locator('button[type="submit"]').click()
        page.wait_for_function("document.querySelector('[data-presence-runtime-form]').dataset.stateFingerprint === '1'.repeat(64)")
        assert writes[0]["workspace_root"] == selected
        assert writes[0]["runtime_overrides"]["model_slot"] == "light"
        if not form.is_visible():
            card.locator('.skills-details > summary').click()
        assert folder.input_value() == selected
        form.locator('[data-presence-runtime-reset]').click()
        page.wait_for_function("document.querySelector('[data-presence-runtime-form]').dataset.stateFingerprint === '2'.repeat(64)")
        assert "workspace_root" not in writes[1]
        assert writes[1]["runtime_overrides"] == {"model_slot": None, "inline_max_rounds": None}
        if not form.is_visible():
            card.locator('.skills-details > summary').click()
        assert folder.input_value() == selected
        _capture(page, browser_name, width, "presence-workspace-saved")
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    finally:
        browser.close()


@pytest.mark.parametrize("browser_name", ["chromium", "webkit"])
@pytest.mark.parametrize("width", [390, 1440])
def test_skills_failure_recovery_and_current_catalog_refresh(skills_browser, browser_name, width):
    """Failed reads stay distinct from empty, keep prior content and recover in place."""
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    try:
        page = browser.new_page(viewport={"width": width, "height": 844})
        state, installed = _open_skills(page)
        state["extensions_error"] = "Installed list unavailable"
        page.goto("http://skills.test/", wait_until="networkidle")
        assert "Installed list unavailable" in page.locator("#skills-status").inner_text()
        assert page.locator("#skills-empty").is_hidden()
        assert page.locator("#skills-list [data-skill]").count() == 0
        _capture(page, browser_name, width, "failed-first-read")

        state["extensions_error"] = ""
        state["extensions"] = [installed]
        _refresh(page)
        assert "Local forecast" in page.locator("#skills-list").inner_text()
        page.evaluate("window.retainedSkill = document.querySelector('#skills-list').firstElementChild")
        state["extensions_error"] = "Installed list unavailable"
        _refresh(page)
        assert "Installed list unavailable" in page.locator("#skills-status").inner_text()
        assert "Local forecast" in page.locator("#skills-list").inner_text()
        assert page.evaluate("window.retainedSkill === document.querySelector('#skills-list').firstElementChild")
        _capture(page, browser_name, width, "stale-retained")

        state["extensions_error"] = ""
        state["extensions"] = []
        _refresh(page)
        assert page.locator("#skills-empty").is_visible()
        _capture(page, browser_name, width, "confirmed-empty")
        state["installed_error"] = "Installed lookup unavailable"
        page.locator('[data-tab="marketplace"]').click()
        page.wait_for_function("() => !document.querySelector('#skills-refresh').disabled")
        assert "could not be read" in page.locator("#mp-status").inner_text()
        assert page.locator('#mp-results button[data-mp-action="install"]:enabled').count() == 0
        _capture(page, browser_name, width, "installed-unavailable")

        state["installed_error"] = ""
        before = len(state["requests"])
        _refresh(page)
        requests = state["requests"][before:]
        assert ("GET", "/api/marketplace/clawhub/search") in requests
        assert page.locator('#mp-results button[data-mp-action="install"]:enabled').count() == 1
        _capture(page, browser_name, width, "catalog-recovered")
        state["catalog_error"] = "Catalog temporarily unavailable"
        _refresh(page)
        assert page.locator("#mp-status").inner_text().count("Catalog temporarily unavailable") == 1
        assert "Catalog temporarily unavailable" not in page.locator("#mp-results").inner_text()
        assert "Weather" in page.locator("#mp-results").inner_text()
        _capture(page, browser_name, width, "catalog-failed-retained")

        page.locator('[data-tab="ouroboroshub"]').click()
        page.wait_for_function("() => !document.querySelector('#skills-refresh').disabled")
        before = len(state["requests"])
        _refresh(page)
        assert ("GET", "/api/marketplace/ouroboroshub/catalog") in state["requests"][before:]
        state["hub_error"] = "Hub temporarily unavailable"
        _refresh(page)
        assert page.locator('#skills-pane-ouroboroshub').inner_text().count("Hub temporarily unavailable") == 1
        _capture(page, browser_name, width, "hub-failed-retained")
        assert all(method == "GET" for method, _ in state["requests"])
    finally:
        browser.close()


@pytest.mark.parametrize("browser_name", ["chromium", "webkit"])
@pytest.mark.parametrize("held", ["state", "queue", "both"])
def test_skills_primary_paints_while_optional_requests_remain_pending(skills_browser, browser_name, held):
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    try:
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        state, installed = _open_skills(page)
        state["extensions"] = [installed]
        state["hold_paths"] = {path for kind, path in [("state", "/api/state"), ("queue", "/api/skills/lifecycle-queue")]
                               if held in (kind, "both")}
        page.goto("http://skills.test/", wait_until="domcontentloaded")
        page.locator('.skills-card[data-skill="weather"]').wait_for()
        page.wait_for_function("!document.querySelector('#skills-refresh').disabled")
        assert len(state["pending"]) == len(state["hold_paths"])
        assert "loading" in page.locator("#skills-status").inner_text()
        assert "Configure GITHUB_TOKEN" not in page.locator("#skills-list").inner_text()
        _capture(page, browser_name, 1280, f"optional-{held}-pending-primary-visible")
    finally:
        _close_pending_browser(browser, state)


@pytest.mark.parametrize("browser_name", ["chromium", "webkit"])
def test_skills_late_enrichment_preserves_presence_and_portalled_menu(skills_browser, browser_name):
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    try:
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        state, installed = _open_skills(page)
        state["extensions"] = [{**installed, "enabled": True, "review_status": "clean",
            "review_stale": False, "review_gate": {"executable_review": True},
            "grants": {"all_granted": True}, "presence_runtime": {
                "defaults": {"model_slot": "light", "inline_max_rounds": 10},
                "overrides": {}, "state_fingerprint": "a" * 64}}]
        state["hold_paths"] = {"/api/state", "/api/skills/lifecycle-queue"}
        page.goto("http://skills.test/", wait_until="domcontentloaded")
        card = page.locator('.skills-card[data-skill="weather"]')
        card.wait_for()
        card.locator('.skills-details > summary').click()
        field = card.locator('[name="inline_max_rounds"]')
        field.fill("27")
        page.evaluate("""() => {
            window.keptForm = document.querySelector('[data-presence-runtime-form]');
            window.keptField = keptForm.elements.inline_max_rounds;
            keptField.focus();
            keptField.dispatchEvent(new CompositionEvent('compositionstart', {bubbles:true, data:'2'}));
        }""")
        running = {"id": "disable-weather", "target": "weather", "kind": "disable", "status": "running"}
        ghost = {"id": "install-ghost", "target": "ghost", "kind": "install", "status": "failed",
                 "source": "clawhub", "error": "Ghost install failed"}
        _release_read(state, "/api/skills/lifecycle-queue", {"active": running, "events": [ghost]})
        page.wait_for_function("document.querySelector('.skills-card[data-skill=weather] .skills-status-chip').textContent.includes('Disabling')")
        assert page.evaluate("keptField === document.activeElement && keptField === document.querySelector('[name=inline_max_rounds]') && keptForm === document.querySelector('[data-presence-runtime-form]')")
        assert field.input_value() == "27"
        assert card.locator('.skills-details').get_attribute('open') is not None
        assert card.locator('.skills-toggle').is_disabled()
        assert not card.locator('.skills-toggle').is_checked()
        ghost_card = page.locator('.skills-card[data-skill="ghost"]')
        assert ghost_card.inner_text().count("Ghost install failed") == 1
        assert ghost_card.locator('button[data-skill-action="retry_install"]').is_enabled()
        page.evaluate("keptField.dispatchEvent(new CompositionEvent('compositionend', {bubbles:true, data:'2'}))")
        trigger = card.locator('[data-skill-menu-trigger]')
        trigger.click()
        page.evaluate("window.keptMenu=document.querySelector('body > .skills-card-menu-dialog');window.keptMenuFocus=document.activeElement")
        publish = page.locator('body > .skills-card-menu-dialog .skills-submit-hub')
        assert publish.get_attribute('data-submit-disabled') == 'true'
        assert "status unavailable" in publish.get_attribute('title')
        _release_read(state, "/api/state", {"github_token_configured": True})
        page.wait_for_function("document.querySelector('body > .skills-card-menu-dialog .skills-submit-hub').dataset.submitDisabled === 'false'")
        assert page.evaluate("keptMenu === document.querySelector('body > .skills-card-menu-dialog') && keptMenuFocus === document.activeElement && keptField === document.querySelector('[name=inline_max_rounds]')")
        assert field.input_value() == '27'
        page.keyboard.press('Escape')
        state["hold_paths"].remove('/api/state')
        for status in ('failed', 'done'):
            _refresh(page)
            card.locator('.skills-details > summary').click()
            field.fill('31')
            page.evaluate("window.keptField=document.querySelector('[name=inline_max_rounds]')")
            _release_read(state, '/api/skills/lifecycle-queue', {"events": [{**running, "status": status, "error": "Transport refused the disable request"}]})
            page.wait_for_function("!document.querySelector('#skills-status').textContent")
            assert page.evaluate("keptField === document.querySelector('[name=inline_max_rounds]')")
            assert field.input_value() == '31'
            assert card.locator('.skills-toggle').is_checked()
            assert card.locator('.skills-toggle').is_enabled()
            assert ghost_card.count() == 0
            assert card.inner_text().count('Transport refused the disable request') == (1 if status == 'failed' else 0)
        state['hold_paths'].add('/api/state')
        _refresh(page)
        card.locator('.skills-details > summary').click()
        field.fill('32')
        page.evaluate('window.keptField=document.querySelector("[name=inline_max_rounds]")')
        trigger.click()
        page.evaluate('window.keptMenu=document.querySelector("body > .skills-card-menu-dialog")')
        _release_read(state, '/api/state', {'error': 'Settings offline'}, 503)
        _release_read(state, '/api/skills/lifecycle-queue', {'error': 'Queue offline'}, 503)
        page.wait_for_function("document.querySelector('#skills-status').textContent.includes('Lifecycle progress could not be refreshed')")
        assert 'Skill settings could not be refreshed' in page.locator('#skills-status').inner_text()
        assert page.evaluate('keptMenu === document.querySelector("body > .skills-card-menu-dialog") && keptField === document.querySelector("[name=inline_max_rounds]")')
        assert field.input_value() == '32'
        page.keyboard.press('Escape')
        _capture(page, browser_name, 1280, 'late-enrichment-preserved-form')
        _refresh(page)
        before = page.locator('#skills-list').inner_html()
        page.evaluate('skillsView.destroy()')
        _release_read(state, '/api/state', {'github_token_configured': False})
        _release_read(state, '/api/skills/lifecycle-queue', {'events': [ghost]})
        assert page.locator('#skills-list').inner_html() == before
    finally:
        _close_pending_browser(browser, state)


@pytest.mark.parametrize("browser_name", ["chromium", "webkit"])
def test_skills_late_enrichment_keeps_local_toggle_and_publish_pending(skills_browser, browser_name):
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    try:
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        state, installed = _open_skills(page)
        state['extensions'] = [{**installed, 'enabled': True, 'review_status': 'clean',
            'review_gate': {'executable_review': True}, 'grants': {'all_granted': True},
            'submit_hub': {'visible': True, 'task_start_allowed': True}}]
        state['hold_paths'] = {'/api/state', '/api/skills/lifecycle-queue',
            '/api/skills/weather/toggle', '/api/skills/weather/publish-preflight'}
        page.goto('http://skills.test/', wait_until='domcontentloaded')
        card = page.locator('.skills-card[data-skill="weather"]')
        card.wait_for()
        card.locator('.skills-toggle').focus()
        card.locator('.skills-toggle').press('Space')
        page.wait_for_function("document.querySelector('.skills-toggle').dataset.skillPending === 'true'")
        page.evaluate('window.keptToggle=document.querySelector(".skills-toggle")')
        _release_read(state, '/api/skills/lifecycle-queue', {'events': []})
        page.wait_for_function("!document.querySelector('#skills-status').textContent.includes('Lifecycle progress loading')")
        assert page.evaluate('keptToggle === document.querySelector(".skills-toggle") && keptToggle.disabled && !keptToggle.checked')
        card.locator('[data-skill-menu-trigger]').click()
        page.evaluate('window.keptPublish=document.querySelector("body > .skills-card-menu-dialog .skills-submit-hub")')
        page.locator('body > .skills-card-menu-dialog .skills-submit-hub').click()
        assert page.evaluate('keptPublish.disabled')
        _release_read(state, '/api/state', {'github_token_configured': False})
        page.wait_for_function("!document.querySelector('#skills-status').textContent")
        assert page.evaluate('keptPublish === document.querySelector(".skills-submit-hub") && keptPublish.disabled && keptToggle.disabled && !keptToggle.checked')
        _release_read(state, '/api/skills/weather/publish-preflight', {'ok': True, 'skill': 'weather',
            'repository': 'fixture/hub', 'state': 'ready', 'task_start_allowed': True})
        page.locator('.confirm-dialog').wait_for()
        page.keyboard.press('Escape')
        page.locator('.confirm-dialog').wait_for(state='detached')
        assert page.evaluate('!keptPublish.disabled')
        _release_read(state, '/api/skills/weather/toggle', {'ok': True})
        assert state['requests'].count(('POST', '/api/skills/weather/toggle')) == 1
        assert state['requests'].count(('POST', '/api/skills/weather/publish-preflight')) == 1
        assert ('POST', '/api/tasks') not in state['requests']
    finally:
        _close_pending_browser(browser, state)


@pytest.mark.parametrize("browser_name", ["chromium", "webkit"])
def test_skills_menu_opened_during_primary_read_closes_only_at_replacement(skills_browser, browser_name):
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    try:
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        state, installed = _open_skills(page)
        state['extensions'] = [installed]
        page.goto('http://skills.test/', wait_until='networkidle')
        state['hold_paths'] = {'/api/extensions'}
        for failed in (True, False):
            page.locator('#skills-refresh').click()
            page.locator('[data-skill-menu-trigger]').click()
            menu = page.locator('body > .skills-card-menu-dialog[open]')
            menu.wait_for()
            if failed:
                _release_read(state, '/api/extensions', {'error': 'Primary offline'}, 503)
            else:
                _release_read(state, '/api/extensions', {'skills': state['extensions'], 'live': {}})
            page.wait_for_function("!document.querySelector('#skills-refresh').disabled")
            assert menu.count() == (1 if failed else 0)
            assert page.locator('body > .skills-card-menu-dialog').count() == (1 if failed else 0)
    finally:
        _close_pending_browser(browser, state)


@pytest.mark.parametrize("browser_name", ["chromium", "webkit"])
def test_skills_late_virtual_failure_keeps_retry_install_callable(skills_browser, browser_name):
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    try:
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        state, installed = _open_skills(page)
        state['extensions'] = [installed]
        ghost = {'id': 'ghost-install', 'target': 'ghost', 'kind': 'install', 'status': 'running', 'source': 'clawhub'}
        state['queue_response'] = {'events': [ghost]}
        page.goto('http://skills.test/', wait_until='networkidle')
        state['hold_paths'] = {'/api/skills/lifecycle-queue', '/api/marketplace/clawhub/install'}
        _refresh(page)
        page.evaluate('window.keptRealCard=document.querySelector(".skills-card[data-skill=weather]")')
        _release_read(state, '/api/skills/lifecycle-queue', {'events': [{**ghost, 'status': 'failed', 'error': 'Registry refused the request'}]})
        retry = page.locator('.skills-card[data-skill=ghost] button[data-skill-action=retry_install]')
        retry.wait_for()
        assert retry.is_enabled()
        assert page.evaluate('keptRealCard === document.querySelector(".skills-card[data-skill=weather]")')
        with page.expect_request('**/api/marketplace/clawhub/install') as request:
            retry.click()
        assert request.value.post_data_json == {'slug': 'ghost', 'overwrite': True, 'auto_review': True}
        _release_read(state, '/api/marketplace/clawhub/install', {'ok': True})
        assert state['requests'].count(('POST', '/api/marketplace/clawhub/install')) == 1
    finally:
        _close_pending_browser(browser, state)


@pytest.mark.parametrize("browser_name", ["chromium", "webkit"])
@pytest.mark.parametrize("width", [320, 641, 981])
def test_skills_tabs_and_bottom_edge_menu(skills_browser, browser_name, width):
    """Keyboard tabs and menu actions remain reachable in a short viewport."""
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    try:
        page = browser.new_page(viewport={"width": width, "height": 460})
        state, installed = _open_skills(page)
        state["extensions"] = [
            {**installed, "name": f"weather-{index}",
             "content_hash": "a" * 64,
             "payload_root": f"skills/external/weather-{index}"}
            for index in range(6)
        ]
        page.goto("http://skills.test/", wait_until="networkidle")
        selected = page.locator('[data-tab="installed"]')
        selected.focus()
        selected.press("ArrowRight")
        assert page.locator('[data-tab="marketplace"]').get_attribute("aria-selected") == "true"
        assert page.locator('#skills-pane-marketplace').is_visible()
        page.keyboard.press("Home")
        assert selected.get_attribute("aria-selected") == "true"
        page.wait_for_function("() => !document.querySelector('#skills-refresh').disabled")

        trigger = page.locator('.skills-card[data-skill="weather-5"] [data-skill-menu-trigger]')
        trigger.scroll_into_view_if_needed()
        trigger.focus()
        trigger.press("Enter")
        menu = page.locator('.skills-card-menu-dialog[open]')
        menu.wait_for(state="visible")
        assert page.locator(':focus').get_attribute("role") == "menuitem"
        page.keyboard.press("End")
        assert page.locator(':focus').inner_text() == "Delete"
        box = menu.bounding_box()
        assert box is not None
        assert 0 <= box['x'] and box['x'] + box['width'] <= width + 1
        assert 0 <= box['y'] and box['y'] + box['height'] <= 461
        assert page.locator(':focus').evaluate("""element => {
            const box = element.getBoundingClientRect();
            return element.contains(document.elementFromPoint(box.x + box.width / 2, box.y + box.height / 2));
        }""")
        _capture(page, browser_name, width, "bottom-edge-menu")
        page.keyboard.press("Escape")
        assert page.locator('.skills-card-menu-dialog[open]').count() == 0
        assert trigger.evaluate("element => element === document.activeElement")

        trigger.press("Enter")
        page.keyboard.press("End")
        page.keyboard.press("Enter")
        dialog = page.locator(".confirm-dialog")
        dialog.wait_for(state="visible")
        assert "Delete" in dialog.inner_text()
        assert page.locator('.skills-card-menu-dialog[open]').count() == 0
        page.keyboard.press("Escape")
        dialog.wait_for(state="detached")
        assert trigger.evaluate("element => element === document.activeElement")
        assert all(method == "GET" for method, _ in state["requests"])
    finally:
        _close_pending_browser(browser, state)
