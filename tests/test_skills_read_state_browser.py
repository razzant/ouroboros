"""Skills read/recovery through production controllers, with no running backend.

The Installed view paints from the local listing alone (sources and catalogs never hold it),
keeps a labelled previous list on a failed read, patches cards in place so open forms, menus
and disclosures survive late answers, and Browse keeps an earlier catalog when a refresh fails.
"""

from __future__ import annotations

import json

import pytest

from tests._skills_page_harness import (
    URL,
    capture,
    fixture,
    no_overflow,
    open_skills,
    refresh,
    skills_browser as _skills_browser,
)

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
skills_browser = _skills_browser

WEATHER = {
    "name": "weather", "source": "external", "description": "Local forecast", "version": "1.0",
    "enabled": False, "review_status": "pending", "payload_root": "skills/external/weather",
    "content_hash": "a" * 64, "grants": {}, "permissions": [], "removal": "delete",
}
CONFIRMED_SOURCES = fixture("skill_sources.json")
for _row in CONFIRMED_SOURCES["sources"]:
    _row["confirmed"] = True


@pytest.mark.parametrize("browser_name,width", [("chromium", 1440), ("webkit", 390)])
def test_retired_presence_runtime_has_no_form(skills_browser, browser_name, width):
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    try:
        page = browser.new_page(viewport={"width": width, "height": 900})
        open_skills(page, extensions=[{**WEATHER, "presence_runtime": {"workspace_root": "/old"}}])
        page.goto(URL, wait_until="networkidle")
        assert page.locator('[data-presence-runtime-form]').count() == 0
    finally:
        browser.close()


@pytest.mark.parametrize("browser_name", ["chromium", "webkit"])
@pytest.mark.parametrize("width", [390, 1440])
def test_skills_failure_recovery_and_catalog_refresh(skills_browser, browser_name, width):
    """Failed reads stay distinct from empty, keep prior content and recover in place."""
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    try:
        page = browser.new_page(viewport={"width": width, "height": 844})
        skills = open_skills(page, sources=CONFIRMED_SOURCES)
        skills.fail("/api/extensions", "Installed list unavailable")
        page.goto(URL, wait_until="networkidle")
        assert "Installed list unavailable" in page.locator("#skills-status").inner_text()
        assert page.locator("#skills-empty").is_hidden()
        assert page.locator("#skills-list [data-skill]").count() == 0
        capture(page, browser_name, width, "failed-first-read")

        skills.set_skills([WEATHER])
        refresh(page)
        assert "Local forecast" in page.locator("#skills-list").inner_text()
        page.evaluate("window.retainedSkill = document.querySelector('#skills-list').firstElementChild")
        skills.fail("/api/extensions", "Installed list unavailable")
        refresh(page)
        assert "Showing the previous list" in page.locator("#skills-status").inner_text()
        assert "Local forecast" in page.locator("#skills-list").inner_text()
        assert page.evaluate("window.retainedSkill === document.querySelector('#skills-list').firstElementChild")
        capture(page, browser_name, width, "stale-retained")

        skills.set_skills([])
        refresh(page)
        assert page.locator("#skills-empty").is_visible()
        capture(page, browser_name, width, "confirmed-empty")

        # Browse: a catalog that cannot be read is said; a later failed refresh keeps the earlier list.
        skills.fail("/api/skill-catalog", "Catalog temporarily unavailable")
        page.locator('.skills-tab[data-tab="browse"]').click()
        status = page.locator("[data-browse-status]")
        page.wait_for_function("document.querySelector('[data-browse-status]').textContent.includes('Couldn’t read the catalogs')")
        assert page.locator("[data-browse-board] .marketplace-card").count() == 0
        skills.answers["/api/skill-catalog"] = fixture("skill_catalog.json")
        refresh(page)
        page.locator('[data-browse-board] .marketplace-card[data-item="ouroboros/duckduckgo"]').wait_for()
        skills.fail("/api/skill-catalog", "Catalog temporarily unavailable")
        refresh(page)
        assert status.inner_text().count("Catalog temporarily unavailable") == 1
        assert "Showing the earlier list" in status.inner_text()
        assert page.locator('[data-browse-board] .marketplace-card[data-item="ouroboros/duckduckgo"]').count() == 1
        capture(page, browser_name, width, "catalog-failed-retained")

        # The registry is asked only on an explicit search; its failure is said once, local matches stay.
        skills.answers["/api/skill-catalog/search"] = {"items": [], "registry_error": "ClawHub: rate limited"}
        before = len(skills.requests)
        search = page.locator("[data-browse-search]")
        search.fill("weather")
        search.press("Enter")
        page.wait_for_function("document.querySelector('[data-browse-status]').textContent.includes('couldn’t search ClawHub: rate limited')")
        assert ("GET", "/api/skill-catalog/search?q=weather") in skills.requests[before:]
        assert status.inner_text().count("rate limited") == 1
        assert page.locator('[data-browse-board] .marketplace-card[data-item="ouroboros/weather"]').count() == 1
        capture(page, browser_name, width, "registry-failed")
        assert all(method == "GET" for method, _ in skills.requests)
        assert no_overflow(page)
    finally:
        browser.close()


@pytest.mark.parametrize("browser_name", ["chromium", "webkit"])
@pytest.mark.parametrize("held", ["state", "queue", "both"])
def test_skills_primary_paints_while_optional_requests_remain_pending(skills_browser, browser_name, held):
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        skills = open_skills(page, extensions=[WEATHER])
        skills.hold = {path for kind, path in [("state", "/api/state"), ("queue", "/api/skills/lifecycle-queue")]
                       if held in (kind, "both")}
        page.goto(URL, wait_until="domcontentloaded")
        page.locator('.skills-card[data-skill="weather"]').wait_for()
        page.wait_for_function("!document.querySelector('#skills-refresh').disabled")
        assert len(skills.pending) == len(skills.hold)
        assert page.locator("#skills-status").inner_text() == "", "a pending optional read is not a failure"
        assert "GITHUB_TOKEN" not in page.locator("#skills-list").inner_text()
        capture(page, browser_name, 1280, f"optional-{held}-pending-primary-visible")
    finally:
        if skills:
            skills.close(browser)


@pytest.mark.parametrize("browser_name", ["chromium", "webkit"])
def test_skills_late_enrichment_preserves_input_and_portalled_menu(skills_browser, browser_name):
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        row = {**WEATHER, "enabled": True, "review_status": "clean", "review_stale": False,
               "review_gate": {"executable_review": True}, "grants": {"all_granted": True},
               "presence_runtime": {"defaults": {"model_slot": "light", "inline_max_rounds": 10},
                                    "overrides": {}, "state_fingerprint": "a" * 64}}
        skills = open_skills(page, extensions=[row])
        skills.hold = {"/api/state", "/api/skills/lifecycle-queue"}
        page.goto(URL, wait_until="domcontentloaded")
        card = page.locator('.skills-card[data-skill="weather"]')
        card.wait_for()
        card.locator('.skills-details > summary').click()
        card.locator('.skills-details').evaluate("node => node.insertAdjacentHTML('beforeend', '<form data-local-test-form><input name=inline_max_rounds></form>')")
        field = card.locator('[name="inline_max_rounds"]')
        field.fill("27")
        page.evaluate("""() => {
            window.keptForm = document.querySelector('[data-local-test-form]');
            window.keptField = keptForm.elements.inline_max_rounds;
            keptField.focus();
            keptField.dispatchEvent(new CompositionEvent('compositionstart', {bubbles:true, data:'2'}));
        }""")
        running = {"id": "disable-weather", "target": "weather", "kind": "disable", "status": "running"}
        ghost = {"id": "install-ghost", "target": "ghost", "kind": "install", "status": "failed",
                 "source": "clawhub", "error": "Ghost install failed"}
        skills.release("/api/skills/lifecycle-queue", {"active": running, "events": [ghost]})
        page.wait_for_function("document.querySelector('.skills-card[data-skill=weather] .skills-card-status').textContent.includes('Turning off')")
        assert page.evaluate("keptField === document.activeElement && keptField === document.querySelector('[name=inline_max_rounds]') && keptForm === document.querySelector('[data-local-test-form]')")
        assert field.input_value() == "27"
        assert card.locator('.skills-details').get_attribute('open') is not None
        assert card.locator('.skills-toggle').is_disabled()
        assert not card.locator('.skills-toggle').is_checked()
        ghost_card = page.locator('.skills-card[data-skill="ghost"]')
        assert ghost_card.inner_text().count("Ghost install failed") == 1
        assert "Nothing was installed." in ghost_card.inner_text()
        assert ghost_card.locator('button[data-skill-action="find_in_browse"]').is_enabled()
        page.evaluate("keptField.dispatchEvent(new CompositionEvent('compositionend', {bubbles:true, data:'2'}))")
        trigger = card.locator('[data-skill-menu-trigger]')
        trigger.click()
        page.evaluate("window.keptMenu=document.querySelector('body > .skills-card-menu-dialog:not([data-leaving])');window.keptMenuFocus=document.activeElement")
        publish = page.locator('body > .skills-card-menu-dialog:not([data-leaving]) .skills-submit-hub')
        assert publish.get_attribute('data-submit-disabled') == 'true'
        assert "status unavailable" in publish.get_attribute('data-submit-reason')
        skills.release("/api/state", {"github_token_configured": True})
        page.wait_for_function("document.querySelector('body > .skills-card-menu-dialog:not([data-leaving]) .skills-submit-hub').dataset.submitDisabled === 'false'")
        assert page.evaluate("keptMenu === document.querySelector('body > .skills-card-menu-dialog:not([data-leaving])') && keptMenuFocus === document.activeElement && keptField === document.querySelector('[name=inline_max_rounds]')")
        assert field.input_value() == '27'
        page.keyboard.press('Escape')
        skills.hold.discard('/api/state')
        for status in ('failed', 'done'):
            refresh_button = page.locator('#skills-refresh')
            refresh_button.click()
            skills.release('/api/skills/lifecycle-queue', {"events": [{**running, "status": status, "error": "Transport refused the disable request"}]})
            page.wait_for_function("!document.querySelector('#skills-refresh').disabled")
            page.wait_for_function("document.querySelector('.skills-card[data-skill=weather] .skills-toggle').checked")
            assert page.evaluate("keptField === document.querySelector('[name=inline_max_rounds]')")
            assert field.input_value() == '27', 'the open form survives every re-read'
            assert card.locator('.skills-toggle').is_enabled()
            assert ghost_card.count() == 0
            assert card.inner_text().count('Transport refused the disable request') == (1 if status == 'failed' else 0)
        capture(page, browser_name, 1280, 'late-enrichment-preserved-form')
        skills.hold = {'/api/state', '/api/skills/lifecycle-queue'}
        page.locator('#skills-refresh').click()
        page.wait_for_function("!document.querySelector('#skills-refresh').disabled")
        before = page.locator('#skills-list').inner_html()
        page.evaluate('skillsView.destroy()')
        skills.release('/api/state', {'github_token_configured': False})
        skills.release('/api/skills/lifecycle-queue', {'events': [ghost]})
        page.wait_for_timeout(200)
        assert page.locator('#skills-list').inner_html() == before, 'a destroyed view ignores late answers'
    finally:
        if skills:
            skills.close(browser)


@pytest.mark.parametrize("browser_name", ["chromium", "webkit"])
def test_details_follow_the_listing_and_keep_what_the_owner_opened(skills_browser, browser_name):
    """A re-read updates Details' facts in place (a forgotten publication record goes); an opened
    findings list and the Presence form keep their identity until the review itself changes."""
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    try:
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        findings = [{"item": "exec", "verdict": "warn", "reason": "Spawns a shell"}]
        row = {**WEATHER, "enabled": True, "review_status": "warnings", "review_gate": {"executable_review": True},
               "review_findings": findings, "published": {
                   "version": "1.0", "pr_number": 7, "pr_url": "https://github.com/hub/project/pull/7",
                   "content_hash": "a" * 64, "repository": "hub/project"},
               "presence_runtime": {"defaults": {"model_slot": "main", "inline_max_rounds": 10},
                                    "overrides": {}, "state_fingerprint": "a" * 64}}
        skills = open_skills(page, extensions=[row])
        page.goto(URL, wait_until="networkidle")
        card = page.locator('.skills-card[data-skill="weather"]')
        card.locator('.skills-details > summary').click()
        publication = card.locator('.skills-detail-row', has_text="Publication")
        assert "Submitted v1.0 · PR #7" in publication.inner_text()
        card.locator('.skills-review-findings > summary').click()
        card.locator('.skills-review-findings li').first.wait_for()
        card.locator('.skills-details').evaluate("node => node.insertAdjacentHTML('beforeend', '<form data-local-test-form><input name=inline_max_rounds></form>')")
        card.locator('[name="inline_max_rounds"]').fill("27")
        page.evaluate("""() => {
            window.keptForm = document.querySelector('[data-local-test-form]');
            window.keptFindings = document.querySelector('.skills-review-findings');
        }""")

        skills.set_skills([{**row, "published": None}])
        refresh(page)
        assert publication.count() == 0, "the forgotten record is gone without a reload"
        assert card.locator('.skills-details').get_attribute('open') is not None
        assert page.evaluate("keptForm === document.querySelector('[data-local-test-form]')")
        assert card.locator('[name="inline_max_rounds"]').input_value() == "27"
        assert page.evaluate("keptFindings === document.querySelector('.skills-review-findings') && keptFindings.open")
        assert card.locator('.skills-review-findings li').count() == 1

        # A new review of these bytes replaces the block: its findings are the new run's.
        skills.set_skills([{**row, "published": None, "review_findings": findings + [
            {"item": "net", "verdict": "info", "reason": "Opens sockets"}]}])
        refresh(page)
        assert card.locator('.skills-review-findings > summary').inner_text() == "2 review findings"
        assert card.locator('.skills-review-findings li').count() == 0, "the new block is built when opened"
        assert page.evaluate("keptForm === document.querySelector('[data-local-test-form]')")
        capture(page, browser_name, 1280, "details-follow-listing")
    finally:
        browser.close()


@pytest.mark.parametrize("browser_name", ["chromium", "webkit"])
def test_installed_never_waits_for_sources_or_catalogs(skills_browser, browser_name):
    """Installed paints from the local listing; a source read in flight blocks nothing, and the
    Official mark explains its promise once the source it names is known."""
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        listing = fixture("extensions_v4.json")["skills"]
        skills = open_skills(page, extensions=listing, sources=CONFIRMED_SOURCES, catalog=fixture("skill_catalog.json"))
        skills.hold = {"/api/skill-sources", "/api/skill-catalog"}
        page.goto(URL, wait_until="domcontentloaded")
        card = page.locator('.skills-card[data-skill="weather"]')
        card.wait_for()
        assert page.locator('#skills-list > .skills-card').count() == len(listing)
        assert ("GET", "/api/skill-catalog") not in skills.requests, "Installed never reads a catalog"
        assert any(path == "/api/skill-sources" for _, path in skills.requests)
        card.locator('[data-official-mark]').click()
        assert page.locator('.skills-mark-popup:not([data-leaving])').count() == 0, 'the promise names its source: it waits for it'
        skills.release("/api/skill-sources")
        popup = page.locator('.skills-mark-popup:not([data-leaving])')
        popup.wait_for()
        assert "the maintainer of Ouroboros vouches for this exact version" in popup.inner_text()
        assert "github.com/razzant/OuroborosHub" in popup.inner_text()
        page.keyboard.press("Escape")
        assert popup.count() == 0
        capture(page, browser_name, 1280, "installed-without-sources")
    finally:
        if skills:
            skills.close(browser)


@pytest.mark.parametrize("browser_name", ["chromium", "webkit"])
def test_skills_late_enrichment_keeps_local_toggle_and_publish_pending(skills_browser, browser_name):
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={'width': 1280, 'height': 900})
        skills = open_skills(page, extensions=[{**WEATHER, 'enabled': True, 'review_status': 'clean',
            'review_gate': {'executable_review': True}, 'grants': {'all_granted': True},
            'submit_hub': {'visible': True, 'task_start_allowed': True}}])
        skills.hold = {'/api/state', '/api/skills/lifecycle-queue', '/api/skills/weather/toggle',
                       '/api/skills/weather/publish-preflight'}
        page.goto(URL, wait_until='domcontentloaded')
        card = page.locator('.skills-card[data-skill="weather"]')
        card.wait_for()
        card.locator('.skills-toggle').focus()
        card.locator('.skills-toggle').press('Space')
        page.wait_for_function("document.querySelector('.skills-toggle').dataset.skillPending === 'true'")
        page.evaluate('window.keptToggle=document.querySelector(".skills-toggle")')
        skills.release('/api/skills/lifecycle-queue', {'events': []})
        assert page.evaluate('keptToggle === document.querySelector(".skills-toggle") && keptToggle.disabled && !keptToggle.checked')
        card.locator('[data-skill-menu-trigger]').click()
        page.evaluate('window.keptPublish=document.querySelector("body > .skills-card-menu-dialog:not([data-leaving]) .skills-submit-hub")')
        page.locator('body > .skills-card-menu-dialog:not([data-leaving]) .skills-submit-hub').click()
        page.wait_for_function('keptPublish.disabled')
        skills.release('/api/state', {'github_token_configured': False})
        page.wait_for_timeout(100)
        assert page.evaluate('keptPublish === document.querySelector(".skills-submit-hub") && keptPublish.disabled && keptToggle.disabled && !keptToggle.checked')
        # With no connected destination, Publish still offers My skills; keeping the
        # default must send the same empty preflight body as before.
        destination_dialog = page.locator('.skills-dialog')
        assert destination_dialog.locator('[data-create-my-skills]').count() == 1
        assert skills.posted('/api/skill-sources/my-skills') == []
        destination_dialog.locator('[data-dialog-ok]').click()
        skills.release('/api/skills/weather/publish-preflight', {'ok': True, 'skill': 'weather',
            'repository': 'fixture/hub', 'state': 'ready', 'task_start_allowed': True})
        page.locator('.confirm-dialog').wait_for()
        page.keyboard.press('Escape')
        page.locator('.confirm-dialog').wait_for(state='detached')
        page.wait_for_function('!keptPublish.disabled')
        skills.release('/api/skills/weather/toggle', {'ok': True})
        assert skills.requests.count(('POST', '/api/skills/weather/toggle')) == 1
        assert skills.requests.count(('POST', '/api/skills/weather/publish-preflight')) == 1
        assert skills.posted('/api/skills/weather/publish-preflight') == [{}]
        assert ('POST', '/api/tasks') not in skills.requests
    finally:
        if skills:
            skills.close(browser)


@pytest.mark.parametrize("browser_name", ["chromium", "webkit"])
def test_a_menu_opened_during_a_reread_survives_it(skills_browser, browser_name):
    """Cards are patched in place: a re-read, failed or successful, never closes an open menu."""
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={'width': 1280, 'height': 900})
        skills = open_skills(page, extensions=[WEATHER])
        page.goto(URL, wait_until='networkidle')
        skills.hold = {'/api/extensions'}
        for failed in (True, False):
            page.locator('#skills-refresh').click()
            if page.locator('body > .skills-card-menu-dialog:not([data-leaving])[open]').count() == 0:
                page.locator('[data-skill-menu-trigger]').click()
            menu = page.locator('body > .skills-card-menu-dialog:not([data-leaving])[open]')
            menu.wait_for()
            page.evaluate("window.keptMenu = document.querySelector('body > .skills-card-menu-dialog:not([data-leaving])[open]')")
            if failed:
                skills.release('/api/extensions', {'error': 'Primary offline'}, 503)
            else:
                skills.release('/api/extensions', {'skills': [WEATHER], 'live': {}})
            page.wait_for_function("!document.querySelector('#skills-refresh').disabled")
            assert page.evaluate("keptMenu === document.querySelector('body > .skills-card-menu-dialog:not([data-leaving])[open]')")
        page.keyboard.press('Escape')
        assert page.locator('body > .skills-card-menu-dialog:not([data-leaving])[open]').count() == 0
    finally:
        if skills:
            skills.close(browser)


@pytest.mark.parametrize("browser_name", ["chromium", "webkit"])
def test_a_failed_install_row_says_nothing_was_installed_and_finds_the_skill_in_browse(skills_browser, browser_name):
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={'width': 1280, 'height': 900})
        skills = open_skills(page, extensions=[WEATHER], sources=CONFIRMED_SOURCES, catalog=fixture("skill_catalog.json"))
        ghost = {'id': 'ghost-install', 'target': 'ghost', 'kind': 'install', 'status': 'running', 'source': 'clawhub'}
        skills.answers['/api/skills/lifecycle-queue'] = {'events': [ghost]}
        skills.answers['/api/skill-catalog/search'] = {'items': []}
        page.goto(URL, wait_until='networkidle')
        assert 'Installing…' in page.locator('.skills-card[data-skill=ghost] .skills-card-status').inner_text()
        skills.hold = {'/api/skills/lifecycle-queue'}
        refresh(page)
        page.evaluate('window.keptRealCard=document.querySelector(".skills-card[data-skill=weather]")')
        skills.release('/api/skills/lifecycle-queue', {'events': [{**ghost, 'status': 'failed', 'error': 'Registry refused the request'}]})
        find = page.locator('.skills-card[data-skill=ghost] button[data-skill-action=find_in_browse]')
        find.wait_for()
        ghost_card = page.locator('.skills-card[data-skill=ghost]')
        assert ghost_card.inner_text().count('Registry refused the request') == 1
        assert 'Nothing was installed.' in ghost_card.inner_text()
        assert page.evaluate('keptRealCard === document.querySelector(".skills-card[data-skill=weather]")')
        find.click()
        page.locator('#skills-pane-browse').wait_for(state='visible')
        assert page.locator('[data-browse-search]').input_value() == 'ghost'
        page.wait_for_function("document.querySelector('[data-browse-status]').textContent.includes('ClawHub answered')")
        assert ('GET', '/api/skill-catalog/search?q=ghost') in skills.requests
        assert all(method == 'GET' for method, _ in skills.requests)
    finally:
        if skills:
            skills.close(browser)


@pytest.mark.parametrize("browser_name", ["chromium", "webkit"])
@pytest.mark.parametrize("width", [320, 641, 981])
def test_skills_tabs_and_bottom_edge_menu(skills_browser, browser_name, width):
    """Keyboard tabs and menu actions remain reachable in a short viewport."""
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": width, "height": 460})
        skills = open_skills(page, extensions=[
            {**WEATHER, "name": f"weather-{index}", "payload_root": f"skills/external/weather-{index}"}
            for index in range(6)
        ], sources=CONFIRMED_SOURCES)
        page.goto(URL, wait_until="networkidle")
        selected = page.locator('[data-tab="installed"]')
        selected.focus()
        selected.press("ArrowRight")
        assert page.locator('[data-tab="browse"]').get_attribute("aria-selected") == "true"
        assert page.locator('#skills-pane-browse').is_visible()
        assert page.locator('#skills-pane-installed').is_hidden()
        page.keyboard.press("Home")
        assert selected.get_attribute("aria-selected") == "true"
        page.wait_for_function("() => !document.querySelector('#skills-refresh').disabled")

        trigger = page.locator('.skills-card[data-skill="weather-5"] [data-skill-menu-trigger]')
        trigger.scroll_into_view_if_needed()
        trigger.focus()
        trigger.press("Enter")
        menu = page.locator('.skills-card-menu-dialog:not([data-leaving])[open]')
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
        capture(page, browser_name, width, "bottom-edge-menu")
        page.keyboard.press("Escape")
        assert page.locator('.skills-card-menu-dialog:not([data-leaving])[open]').count() == 0
        assert trigger.evaluate("element => element === document.activeElement")

        trigger.press("Enter")
        page.keyboard.press("End")
        page.keyboard.press("Enter")
        dialog = page.locator(".confirm-dialog")
        dialog.wait_for(state="visible")
        assert "Delete" in dialog.inner_text()
        assert page.locator('.skills-card-menu-dialog:not([data-leaving])[open]').count() == 0
        page.keyboard.press("Escape")
        dialog.wait_for(state="detached")
        assert trigger.evaluate("element => element === document.activeElement")
        assert all(method == "GET" for method, _ in skills.requests)
        assert no_overflow(page)
    finally:
        if skills:
            skills.close(browser)


@pytest.mark.parametrize("browser_name", ["chromium", "webkit"])
def test_skills_review_reads_the_list_once_and_findings_render_on_open(skills_browser, browser_name):
    """Review paints its state from the list in memory and reads the list once after the
    request; a card's findings block is collapsed, summary only, until it is opened."""
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        findings = [
            {"item": "exec", "verdict": "warn", "reason": "Spawns a shell"},
            {"item": "net", "verdict": "info", "reason": "Opens sockets"},
        ]
        skills = open_skills(page, extensions=[{**WEATHER, "review_status": "warnings",
                                                "review_gate": {"executable_review": False}, "review_findings": findings}])
        page.goto(URL, wait_until="networkidle")
        card = page.locator('.skills-card[data-skill="weather"]')
        card.locator('.skills-details > summary').click()
        block = card.locator('.skills-review-findings')
        assert block.locator('summary').inner_text() == "2 review findings"
        assert block.locator('li').count() == 0
        assert "Spawns a shell" not in page.locator('#skills-list').inner_html()
        capture(page, browser_name, 1280, "findings-collapsed")
        block.locator('summary').click()
        block.locator('li').first.wait_for()
        assert block.locator('li').all_inner_texts() == ["warn exec: Spawns a shell", "info net: Opens sockets"]
        capture(page, browser_name, 1280, "findings-opened")
        block.locator('summary').click()
        block.locator('summary').click()
        assert block.locator('li').count() == 2, "closing and reopening keeps the one built list"

        # Review from the card's tray: the state needs no list read; the one read follows the request.
        skills.hold = {"/api/skills/weather/review"}
        before = skills.requests.count(("GET", "/api/extensions"))
        card.locator('.skills-tray button[data-skill-action="review"]').click()
        dialog = page.locator(".confirm-dialog")
        dialog.wait_for(state="visible")
        dialog.locator("[data-confirm-ok]").click()
        page.wait_for_function("document.querySelector('.skills-card[data-skill=weather]').dataset.reviewing === '1'")
        assert "Reviewing…" in card.locator('.skills-card-status').inner_text()
        assert len(skills.pending) == 1, "the review request is in flight"
        assert skills.requests.count(("GET", "/api/extensions")) == before, "no list read to paint the state"
        assert block.locator('li').count() == 2, "the local repaint keeps the opened findings"
        capture(page, browser_name, 1280, "review-local")
        skills.release("/api/skills/weather/review", {"status": "clean", "findings": []})
        page.wait_for_function("!document.querySelector('.skills-card[data-skill=weather]').dataset.reviewing")
        skills.wait_for_count(("GET", "/api/extensions"), before + 1)
        page.wait_for_timeout(200)
        assert skills.requests.count(("GET", "/api/extensions")) == before + 1, "one list read after the action"
        assert skills.requests.count(("POST", "/api/skills/weather/review")) == 1
    finally:
        if skills:
            skills.close(browser)


def _failed_review_projection(tmp_path, monkeypatch, deps_reason):
    """Real current failures, plus retained pre-ledger lifecycle rows read by /api/extensions."""
    from types import SimpleNamespace

    from ouroboros.skill_review import SkillReviewOutcome
    from ouroboros.skill_review_runner import run_skill_review_lifecycle_blocking, skill_review_ui_projection

    drive_root, skills_root = tmp_path / "drive", tmp_path / "skills"
    (skills_root / "weather").mkdir(parents=True)
    drive_root.mkdir()
    (skills_root / "weather" / "SKILL.md").write_text(
        "---\nname: weather\ndescription: Local forecast\nversion: 1.0\ntype: instruction\n---\nbody\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("supervisor.message_bus.send_with_budget", lambda *a, **k: None)
    monkeypatch.setattr("ouroboros.skill_dependencies.ensure_skill_dependencies",
                        lambda *_a, **_k: ("failed", deps_reason))
    monkeypatch.setattr("ouroboros.skill_review_runner._reconcile_extension_payload",
                        lambda *_a, **_k: {"action": None, "reason": None, "process": "", "server_reconcile": ""})

    def review(impl):
        return run_skill_review_lifecycle_blocking(
            SimpleNamespace(drive_root=drive_root, repo_dir=tmp_path, messages=[]), "weather",
            source="skills", review_impl=impl, repo_path=str(skills_root),
        )

    def clean(_ctx, name):
        from ouroboros.skill_loader import find_skill, save_review_state, SkillReviewState
        loaded = find_skill(drive_root, name, repo_path=str(skills_root))
        save_review_state(drive_root, name, SkillReviewState(status="clean", content_hash=loaded.content_hash))
        return SkillReviewOutcome(skill_name=name, status="pass", content_hash=loaded.content_hash,
                                  findings=[{"item": "manifest_schema", "verdict": "PASS"}])

    def broken(_ctx, _name):
        raise RuntimeError("reviewer transport closed")

    assert review(clean)["status"] == "clean"
    dependency_failure = skill_review_ui_projection(drive_root, "weather")
    first = dependency_failure["current"]
    assert first["review_status"] == "clean" and first["lifecycle_status"] == "failed"
    assert first["terminal_reason"] == deps_reason
    with pytest.raises(RuntimeError, match="reviewer transport closed"):
        review(broken)
    second = skill_review_ui_projection(drive_root, "weather")["current"]
    # Since #2123 only a real review wave writes the shared ledger. These stubs do
    # not dispatch one. Seed the frozen legacy format explicitly to exercise its
    # supported historical reader alongside the actual current failed job.
    from ouroboros.skill_review_history import review_history_path
    path = review_history_path(drive_root, "weather")
    path.write_text("".join(json.dumps({**row, "job_status": row["status"],
        "status": row["review_status"] or row["status"]}) + "\n" for row in (first, second)), encoding="utf-8")
    before = path.read_bytes()
    projection = skill_review_ui_projection(drive_root, "weather")
    assert path.read_bytes() == before, "reading legacy failures must not rewrite them"
    return dependency_failure, projection


@pytest.mark.parametrize("browser_name,width", [("chromium", 1440), ("webkit", 390)])
def test_failed_review_runs_read_as_failures_with_their_recorded_reason(
        skills_browser, browser_name, width, tmp_path, monkeypatch):
    """PR #782: a failed run is not an open `pending`, and a clean verdict whose dependency
    install failed keeps both facts; a long recorded reason stays bounded on a narrow card and
    the full reason opens by touch or keyboard (no hover title)."""
    dependency_failure, projection = _failed_review_projection(tmp_path, monkeypatch, "pip install failed: " + "e" * 3000)
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    try:
        page = browser.new_page(viewport={"width": width, "height": 900})
        skills = open_skills(page, extensions=[{**WEATHER, "skill_review": dependency_failure}])
        page.goto(URL, wait_until="networkidle")
        card = page.locator('.skills-card[data-skill="weather"]')
        card.locator('.skills-details > summary').click()
        current = card.locator(".skills-review-current").inner_text()
        assert current.endswith(" · clean · lifecycle failed"), current
        reason = card.locator(".skills-review-current + .skills-review-reason")
        assert reason.inner_text().startswith("Reason: pip install failed: eee")
        assert reason.bounding_box()["height"] < 200
        skills.set_skills([{**WEATHER, "skill_review": projection}])
        refresh(page)
        current = card.locator(".skills-review-current").inner_text()
        assert current.endswith(" · review verdict unavailable · lifecycle failed"), current
        assert "pending" not in current
        assert card.locator(".skills-review-current + .skills-review-reason").inner_text() == (
            "Reason: RuntimeError: reviewer transport closed")
        card.locator(".skills-review-history > summary").click()
        rows = card.locator(".skills-review-history li")
        assert rows.count() == 2
        first, second = rows.all_inner_texts()
        assert first.splitlines()[0].endswith(" · clean · lifecycle failed · skills"), first
        assert first.splitlines()[1].startswith("Reason: pip install failed: eee")
        assert second.splitlines()[0].endswith(" · review verdict unavailable · lifecycle failed · skills")
        reason = rows.first.locator("div.skills-review-reason")
        assert reason.get_attribute("title") is None
        assert reason.bounding_box()["height"] < 200, "the bounded reason stays a few lines"
        full = rows.first.locator("details.skills-review-reason")
        full.locator("summary").click()
        assert full.inner_text().rstrip().endswith("e" * 100)
        capture(page, browser_name, width, "review-run-failed")
        assert no_overflow(page)
    finally:
        browser.close()
