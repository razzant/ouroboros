"""Health and lifecycle transitions through the real Skills controller.

HTTP replies are controlled facts, not a live installation. Automatic recovery needs no
Refresh or page reload, and same-kind updates retain the owner's working controls.
"""
from __future__ import annotations

import pytest

from tests._skills_page_harness import (
    URL, capture, no_overflow, open_skills, refresh, skills_browser as _skills_browser,
)

pytestmark = pytest.mark.ui_browser
skills_browser = _skills_browser


def _installed(name):
    return {
        "name": name, "source": "external", "description": f"Installed {name}",
        "version": "1.0", "type": "extension", "enabled": False,
        "review_status": "clean", "review_stale": False,
        "review_gate": {"executable_review": True},
        "content_hash": "a" * 64, "payload_root": f"skills/external/{name}",
        "grants": {"all_granted": True}, "permissions": [], "removal": "delete",
    }


def _queue(kind="install", status="running"):
    return {"events": [{"id": "calendar-operation", "target": "calendar",
                        "kind": kind, "status": status, "message": "Installing calendar",
                        "source": "skills"}],  # the queue's `source` is the caller, never where the package came from
            "active": None}


def _calendar_health(error="Daemon stopped", companion_reason="companion_restart_exhausted:daemon", **facts):
    ready = {**_installed("calendar"), "enabled": True, "live_loaded": True, "desired_live": True,
             "dispatch_live": True, "live_reason": "ready", "load_error": None, **facts}
    return ready, {**ready, "live_reason": companion_reason, "load_error": error}


def _automatic_read(page, skills, change):
    request = ("GET", "/api/extensions")
    count = skills.requests.count(request)
    change()
    skills.wait_for_count(request, count + 1, timeout_ms=7000)
    # A matching reply, not just elapsed time or a request, must reach the consumer.
    page.wait_for_function("!document.querySelector('#skills-refresh').disabled")


def _track_listing_bodies(page):
    page.clock.pause_at(page.evaluate("new Date(Date.now() + 60_000).toISOString()"))
    page.evaluate("""() => {
        window.listingReads = [];
        window.holdNextListing = false;
        const fetch = window.fetch.bind(window);
        window.fetch = async (url, init) => {
            if (url !== '/api/extensions') return fetch(url, init);
            const hold = window.holdNextListing;
            window.holdNextListing = false;
            const read = {signal: init.signal, sentAt: Date.now()};
            window.listingReads.push(read);
            const response = await fetch(url, init);
            if (!hold) return response;
            // HTTP answered. Its JSON read remains pending even after an abort, so a late body can be tested too.
            return {ok: response.ok, status: response.status, json: () => new Promise(resolve => { read.releaseBody = resolve; })};
        };
    }""")


def _hold_listing_body(page):
    page.evaluate("() => { window.holdNextListing = true; skillSocketEvent('extension_health_changed'); }")
    page.clock.run_for(350)
    page.wait_for_function("Boolean(listingReads.at(-1)?.releaseBody)", polling=20, timeout=5000)


def test_open_skills_card_recovers_on_companion_health_hint(skills_browser):
    from playwright.sync_api import expect

    browser = skills_browser.chromium.launch(headless=True)
    try:
        page = browser.new_page(viewport={"width": 1100, "height": 750})
        page.clock.install()
        ready, broken = _calendar_health("Daemon exhausted its restarts",
            companion_reason="companion_restart_exhausted:alpha,beta,gamma,delta,epsilon", published={
            "version": "1.0", "content_hash": "a" * 64, "pr_number": 7, "repository": "example/skills",
        })
        skills = open_skills(page, extensions=[broken])
        page.goto(URL, wait_until="networkidle")
        card = page.locator('.skills-card[data-skill="calendar"]')
        chip = card.locator('.skills-card-status')
        expect(chip).to_have_text("Background process stopped")
        card.locator('.skills-details > summary').click()
        expect(card.locator('[data-skill-detail-facts]')).to_contain_text(
            'background process stopped: alpha, beta, gamma, delta, epsilon')
        page.evaluate("""async () => {
            const { applyPayload } = await import('/static/modules/i18n.js');
            applyPayload({language: 'fr', entries: {'Forget this record': {text: 'Oublier cette trace'}}});
        }""")
        forget = card.locator('[data-skill-action="forget_publication"]')
        expect(forget).to_have_text('Oublier cette trace')
        forget.evaluate("button => { window.keptForget = button; button.focus(); }")
        page.evaluate("window.calendarCard = document.querySelector('.skills-card[data-skill=calendar]')")
        capture(page, "chromium", 1100, "companion-failed")

        listing = ("GET", "/api/extensions")
        before = skills.requests.count(listing)
        page.clock.run_for(6500)
        assert skills.requests.count(listing) == before, "an idle broken card does not poll the listing"
        partially_recovered = {**broken, "live_reason": "companion_restart_exhausted:beta,epsilon",
                               "load_error": "companion 'beta' exhausted its restart budget; companion 'epsilon' exhausted its restart budget"}
        skills.set_skills([partially_recovered])
        page.evaluate("() => { skillSocketEvent('extension_health_changed'); skillSocketEvent('extension_health_changed'); }")
        page.clock.fast_forward(350)
        expect(chip).to_have_text("Background process stopped")
        assert skills.requests.count(listing) == before + 1, "the two hints share one read"
        assert page.evaluate("calendarCard === document.querySelector('.skills-card[data-skill=calendar]')")
        assert card.locator('.skills-details').get_attribute('open') is not None
        facts = card.locator('[data-skill-detail-facts]')
        expect(facts).to_contain_text('background process stopped: beta, epsilon')
        for recovered_name in ('alpha', 'gamma', 'delta'):
            expect(facts).not_to_contain_text(recovered_name)
        expect(forget).to_be_focused()
        assert forget.evaluate("button => button === keptForget")
        capture(page, "chromium", 1100, "companion-partially-recovered")

        skills.set_skills([ready])
        page.evaluate("skillSocketEvent('extension_health_changed')")
        page.clock.fast_forward(350)
        expect(chip).not_to_have_text("Background process stopped")
        expect(card.locator('[data-skill-trays]')).not_to_contain_text("Daemon exhausted its restarts")
        assert skills.requests.count(listing) == before + 2, "partial and complete recovery share one read each"
        expect(facts).not_to_contain_text('background process stopped')
        expect(facts).to_contain_text('Loaded')
        expect(forget).to_be_focused()
        assert forget.evaluate("button => button === keptForget")
        capture(page, "chromium", 1100, "companion-recovered")

        # A frame missed with no connection is not replayed; the next socket open re-reads the fact.
        skills.set_skills([broken])
        page.evaluate("skillSocketEvent('open')")
        page.clock.fast_forward(350)
        expect(chip).to_have_text("Background process stopped")
        before = skills.requests.count(listing)
        skills.set_skills([ready])
        page.evaluate("""() => {
            Object.defineProperty(document, 'hidden', {value: true, configurable: true});
            document.dispatchEvent(new Event('visibilitychange'));
            skillSocketEvent('extension_health_changed');
        }""")
        page.clock.fast_forward(350)
        assert skills.requests.count(listing) == before, "a hidden document does not read for the hint"
        page.evaluate("""() => {
            delete document.hidden;
            document.dispatchEvent(new Event('visibilitychange'));
        }""")
        page.clock.fast_forward(350)
        expect(chip).not_to_have_text("Background process stopped")
        assert skills.requests.count(listing) == before + 1

        # Both an optional queue update and the later disappearance of Running leave this record alone.
        queue = '/api/skills/lifecycle-queue'
        skills.answers[queue] = _queue(kind="disable")
        page.evaluate("skillSocketEvent('extension_lifecycle')")
        page.clock.fast_forward(350)
        expect(card.locator('.skills-toggle')).to_be_disabled()
        expect(forget).to_be_focused()
        skills.set_skills([{**ready, "enabled": False, "desired_live": False, "live_loaded": False, "dispatch_live": False}])
        skills.answers[queue] = _queue(kind="disable", status="succeeded")
        page.evaluate("skillSocketEvent('extension_lifecycle')")
        page.clock.fast_forward(350)
        expect(card.locator('.skills-toggle')).to_be_enabled()
        expect(card.locator('[data-skill-detail-facts]')).not_to_contain_text('Running')
        expect(forget).to_be_focused()
        assert forget.evaluate("button => button === keptForget")
        assert not skills.posted('/api/skills/calendar/publication/clear')
    finally:
        browser.close()


def test_companion_hint_retries_a_failed_listing_without_resetting_the_card(skills_browser):
    from playwright.sync_api import expect

    browser = skills_browser.chromium.launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": 390, "height": 844})
        page.clock.install()
        ready, broken = _calendar_health()
        unknown = {**ready, **_installed("reports"), "enabled": True, "live_reason": "health_unavailable"}
        sibling = {**_installed("weather"), "enabled": True, "live_reason": "load_error", "load_error": "Weather cannot start"}
        skills = open_skills(page, extensions=[broken, sibling])
        page.goto(URL, wait_until="networkidle")
        card = page.locator('.skills-card[data-skill="calendar"]')
        chip = card.locator('.skills-card-status')
        status = page.locator('#skills-status')
        expect(chip).to_have_text("Background process stopped")
        summary = card.locator('.skills-details > summary')
        summary.click()
        summary.focus()
        page.evaluate("window.keptCalendar = document.querySelector('.skills-card[data-skill=calendar]')")
        page.clock.pause_at(page.evaluate("new Date(Date.now() + 60_000).toISOString()"))

        listing = ("GET", "/api/extensions")
        before = skills.requests.count(listing)
        optional_paths = ("/api/state", "/api/skills/lifecycle-queue")
        skills.hold.update(optional_paths)
        skills.fail(listing[1], "Temporary listing outage")
        page.evaluate("skillSocketEvent('extension_health_changed')")
        page.clock.fast_forward(350)
        expect(status).to_contain_text("Showing the previous list")
        skills.hold.clear()
        for path in optional_paths:
            with page.expect_request_finished(lambda request: request.url.split('?')[0] == URL.rstrip('/') + path):
                skills.release(path)
        page.clock.run_for(1000)
        expect(status).to_contain_text("Temporary listing outage")
        expect(chip).to_have_text("Background process stopped")
        assert skills.requests.count(listing) == before + 1, "no immediate retry loop on failure"

        skills.set_skills([ready, unknown, sibling])
        skills.fail(optional_paths[1], "Queue unavailable")
        page.clock.fast_forward(15_000)
        expect(chip).to_have_text("")
        expect(status).to_contain_text("Lifecycle progress could not be refreshed")
        expect(status).not_to_contain_text("Showing the previous list")
        expect(summary).to_be_focused()
        assert card.locator('.skills-details').get_attribute('open') is not None
        assert page.evaluate("keptCalendar === document.querySelector('.skills-card[data-skill=calendar]')")
        reports = page.locator('.skills-card[data-skill="reports"]')
        expect(reports.locator('.skills-card-status [data-tone="neutral"]')).to_have_text("Background status unavailable")
        expect(reports.locator('.skills-toggle')).to_be_checked()
        expect(reports.locator('[data-skill-action="repair"]')).to_have_count(0)
        reports.locator('.skills-details > summary').click()
        expect(reports.locator('[data-skill-detail-facts]')).to_contain_text('Loaded · Background status unavailable')
        expect(page.locator('.skills-card[data-skill="weather"]')).to_contain_text("Weather cannot start")
        assert no_overflow(page)
        page.clock.fast_forward(31_000)
        assert skills.requests.count(listing) == before + 2, "after success an idle page stops polling"
        page.clock.resume()
        capture(page, "chromium", 390, "companion-health-unavailable")
    finally:
        if skills:
            skills.close(browser)
        else:
            browser.close()


def test_repair_keeps_focus_when_a_companion_warning_leaves(skills_browser):
    from playwright.sync_api import expect

    browser = skills_browser.chromium.launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": 1100, "height": 750})
        page.clock.install()
        failed = {**_installed("alpha"), "load_error": "Alpha failed to import",
                  "retained_companions": {"beta": "stop requested"}}
        other = {**_installed("other"), "load_error": "Other failed to import"}
        skills = open_skills(page, extensions=[failed, other])
        page.goto(URL, wait_until="networkidle")
        card = page.locator('.skills-card[data-skill="alpha"]')
        repair = card.locator('[data-skill-trays] [data-skill-action="repair"]')
        other_repair = page.locator('.skills-card[data-skill="other"] [data-skill-trays] [data-skill-action="repair"]')
        expect(card).to_contain_text("Waiting for beta to exit.")
        repair.focus()
        page.evaluate("window.alphaCard = document.querySelector('.skills-card[data-skill=alpha]')")
        other_repair.evaluate("button => { window.keptOtherRepair = button; }")

        recovered = {**failed, "retained_companions": {}}
        skills.set_skills([recovered, other])
        page.evaluate("skillSocketEvent('extension_health_changed')")
        page.clock.fast_forward(350)
        expect(card).not_to_contain_text("Waiting for beta to exit.")
        # The leaving warning re-renders the tray, so the tray's nodes are new: what
        # must survive is the owner's place — the same repair action focused in the
        # same card — not the identity of the former node.
        expect(repair).to_be_focused()
        assert page.evaluate("""() => {
            const active = document.activeElement;
            return active instanceof HTMLElement && active.dataset.skill === 'alpha'
                && active.dataset.skillAction === 'repair'
                && window.alphaCard.querySelector('[data-skill-trays]').contains(active);
        }""")
        assert page.evaluate("alphaCard === document.querySelector('.skills-card[data-skill=alpha]')")
        expect(other_repair).not_to_be_focused()
        # A tray that did not change keeps its own controls as they were.
        assert other_repair.evaluate("button => button === window.keptOtherRepair")

        healthy = {**recovered, "load_error": None}
        skills.set_skills([healthy, other])
        page.evaluate("skillSocketEvent('extension_health_changed')")
        page.clock.fast_forward(350)
        expect(card.locator('[data-skill-trays] [data-skill-action="repair"]')).to_have_count(0)
        # A removed action takes the focus with it: no other card or action is focused instead.
        expect(other_repair).not_to_be_focused()
        assert not page.evaluate("document.activeElement?.matches('[data-skill-action=\"repair\"]')")
        assert page.evaluate("document.activeElement === document.body")
        assert page.evaluate("alphaCard === document.querySelector('.skills-card[data-skill=alpha]')")
        assert other_repair.evaluate("button => button === window.keptOtherRepair")
    finally:
        if skills:
            skills.close(browser)
        else:
            browser.close()


def test_held_module_repaint_does_not_focus_another_start_button(skills_browser):
    from playwright.sync_api import expect

    browser = skills_browser.chromium.launch(headless=True)
    try:
        page = browser.new_page()
        skills = open_skills(page, extensions=[_installed('calendar')])
        skills.answers['/api/ui/contributions'] = {'contributions': [
            {'key': f'calendar:{key}', 'skill': 'calendar', 'label': key,
             'revision': 'r1', 'runtime': 'client', 'provides': 'conversation',
             'render': {'kind': 'host', 'url': '/static/tests/fixtures/skills_v4/host_module_demo.js'}}
            for key in ('first', 'second')
        ]}
        page.goto(URL + '?host_modules=1&hold=all', wait_until='networkidle')
        first = page.locator('[data-skill-trays] [data-host-module-key="calendar:first"]')
        second = page.locator('[data-skill-trays] [data-host-module-key="calendar:second"]')
        expect(first).to_be_visible()
        second.focus()
        expect(second).to_be_focused()
        # A runtime fact can change while a Skills control has keyboard focus.
        assert page.evaluate("hostRuntime.startHeldModule('calendar:second')")
        expect(second).to_have_count(0)
        expect(first).to_be_visible()
        expect(first).not_to_be_focused()
        page.keyboard.press('Enter')
        expect(first).to_be_visible()
        assert page.evaluate("hostRuntime.pageModuleFacts().held.map(item => item.key)") == ['calendar:first']
    finally:
        browser.close()


def test_companion_hint_burst_paints_in_flight_read_then_one_trailing_read(skills_browser):
    from playwright.sync_api import expect

    browser = skills_browser.chromium.launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": 1100, "height": 750})
        page.clock.install()
        ready, broken = _calendar_health()
        skills = open_skills(page, extensions=[broken])
        page.goto(URL, wait_until="networkidle")
        card = page.locator('.skills-card[data-skill="calendar"]')
        chip = card.locator('.skills-card-status')
        expect(chip).to_have_text("Background process stopped")
        summary = card.locator('.skills-details > summary')
        summary.click()
        summary.focus()
        page.clock.pause_at(page.evaluate("new Date(Date.now() + 60_000).toISOString()"))
        page.evaluate("""() => {
            window.keptCalendar = document.querySelector('.skills-card[data-skill=calendar]');
            window.listingReads = 0;
            const fetch = window.fetch.bind(window);
            window.fetch = (url, init) => {
                if (url === '/api/extensions') window.listingReads++;
                return fetch(url, init);
            };
        }""")

        listing = "/api/extensions"
        skills.hold.add(listing)
        hint = "skillSocketEvent('extension_health_changed')"
        for _ in range(4):
            page.evaluate(hint)
            page.clock.run_for(100)
        assert page.evaluate("listingReads") == 1, "the first read starts before hints go quiet"

        for _ in range(3):
            page.evaluate(hint)
            page.clock.run_for(350)
        assert page.evaluate("listingReads") == 1, "hints received during the held read do not start more GETs"
        skills.release(listing, {"skills": [{**ready, "live_reason": "health_unavailable"}], "live": {}})
        expect(chip).to_have_text("Background status unavailable")

        page.clock.run_for(350)
        assert page.evaluate("listingReads") == 2, "later hints share a single trailing read"
        skills.release(listing, {"skills": [ready], "live": {}})
        expect(chip).to_have_text("")
        expect(summary).to_be_focused()
        assert card.locator('.skills-details').get_attribute('open') is not None
        assert page.evaluate("keptCalendar === document.querySelector('.skills-card[data-skill=calendar]')")
        page.clock.run_for(31_000)
        assert page.evaluate("listingReads") == 2, "once the hint is covered, no timer polls an idle card"
    finally:
        if skills:
            skills.close(browser)
        else:
            browser.close()


def test_reconnect_recovers_from_a_stalled_listing_body(skills_browser):
    from playwright.sync_api import expect

    browser = skills_browser.chromium.launch(headless=True)
    try:
        page = browser.new_page()
        page.clock.install()
        ready, broken = _calendar_health()
        skills = open_skills(page, extensions=[broken])
        page.goto(URL, wait_until="networkidle")
        chip = page.locator('.skills-card[data-skill="calendar"] .skills-card-status')
        status = page.locator('#skills-status')
        expect(chip).to_have_text("Background process stopped")
        lifetime = page.evaluate("import('/static/modules/api_client.js').then(m => m.STATE_READ_LIFETIME_MS)")
        _track_listing_bodies(page)
        _hold_listing_body(page)

        skills.set_skills([ready])
        page.evaluate("() => { for (const type of ['open', 'extension_health_changed', 'extension_lifecycle']) skillSocketEvent(type); }")
        page.clock.run_for(350)
        assert page.evaluate("listingReads.length === 1 && !listingReads[0].signal.aborted")
        elapsed = page.evaluate("Date.now() - listingReads[0].sentAt")
        page.clock.run_for(lifetime - elapsed)
        expect(status).to_contain_text("the request took too long", timeout=3000)
        expect(status).to_contain_text("Showing the previous list")
        expect(chip).to_have_text("Background process stopped")
        assert page.evaluate("listingReads.length === 1 && listingReads[0].signal.aborted")

        page.clock.run_for(350)
        expect(chip).to_have_text("")
        expect(status).to_have_text("")
        page.evaluate("skill => listingReads[0].releaseBody({skills: [skill], live: {}})", broken)
        page.clock.fast_forward(2 * lifetime)
        expect(chip).to_have_text("")
        assert page.evaluate("listingReads.length") == 2, "one newer read; no idle retry or poll once it answers"
    finally:
        browser.close()


def test_refresh_and_destroy_cancel_a_stalled_listing_body(skills_browser):
    from playwright.sync_api import expect

    browser = skills_browser.chromium.launch(headless=True)
    try:
        page = browser.new_page()
        page.clock.install()
        ready, broken = _calendar_health()
        skills = open_skills(page, extensions=[broken])
        page.goto(URL, wait_until="networkidle")
        chip = page.locator('.skills-card[data-skill="calendar"] .skills-card-status')
        expect(chip).to_have_text("Background process stopped")
        _track_listing_bodies(page)
        _hold_listing_body(page)

        skills.set_skills([ready])
        page.locator('#skills-refresh').press('Enter')
        expect(chip).to_have_text("")
        assert page.evaluate("listingReads.length === 2 && listingReads[0].signal.aborted")
        page.evaluate("skill => listingReads[0].releaseBody({skills: [skill], live: {}})", broken)

        _hold_listing_body(page)
        expect(chip).to_have_text("")
        page.evaluate("skillsView.destroy()")
        assert page.evaluate("listingReads.at(-1).signal.aborted")
        page.evaluate("skill => { listingReads.at(-1).releaseBody({skills: [skill], live: {}}); skillSocketEvent('open'); }", broken)
        page.clock.fast_forward(120_000)
        expect(chip).to_have_text("")
        assert page.evaluate("listingReads.length") == 3
    finally:
        browser.close()


def test_collision_cards_keep_only_their_own_retained_process_across_reads(skills_browser):
    from playwright.sync_api import expect

    browser = skills_browser.chromium.launch(headless=True)
    try:
        page = browser.new_page(viewport={"width": 1100, "height": 750})
        collision = {**_installed("calendar"), "identity_collision": True, "source": "user_repo", "payload_root": "",
                     "type": "instruction", "load_error": "Skill name collision", "removal": ""}
        first = {**collision, "collision_row_id": "a" * 64, "payload_root": "skills/external/calendar"}
        second = {**collision, "collision_row_id": "b" * 64, "payload_root": "skills/clawhub/calendar"}
        retained = {**first, "retained_companions": {"daemon": "stop requested"}}
        ordinary = _installed("weather")
        skills = open_skills(page, extensions=[retained, second, ordinary])
        page.goto(URL, wait_until="networkidle")
        cards = page.locator('.skills-card[data-skill="calendar"]')
        held_card = page.locator(f'.skills-card[data-skill-row="collision:{first["collision_row_id"]}"]')
        sibling = page.locator(f'.skills-card[data-skill-row="collision:{second["collision_row_id"]}"]')
        page.evaluate("window.weatherCard = document.querySelector('.skills-card[data-skill=weather]')")

        for current in ([retained, second, ordinary], [second, retained, ordinary], [first, second, ordinary]):
            skills.set_skills(current)
            refresh(page)
            expect(cards).to_have_count(2)
            expect(sibling).not_to_contain_text("Waiting for daemon to exit.")
            if retained in current:
                expect(held_card).to_contain_text("Waiting for daemon to exit.")
            else:
                expect(held_card).not_to_contain_text("Waiting for daemon to exit.")
        assert page.evaluate("weatherCard === document.querySelector('.skills-card[data-skill=weather]')")
        for card, row, menu in ((held_card, first, False), (held_card, first, True), (sibling, second, True)):
            if menu:
                card.locator('[data-skill-menu-trigger]').click()
                page.locator('body > .skills-card-menu-dialog [data-skill-action="open_detail"]').click()
            else:
                card.locator('.skills-card-name').click()
            sheet = page.locator('.skills-detail-sheet')
            sheet.locator('.skills-sheet-details > summary').click()
            expect(sheet.locator('.skills-sheet-facts')).to_contain_text(row['payload_root'])
            sheet.locator('[data-sheet-close]').click()
    finally:
        browser.close()


@pytest.mark.parametrize("browser_name", ["chromium", "webkit"])
def test_virtual_install_becomes_operable_without_reload(skills_browser, browser_name):
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": 1100, "height": 750})
        weather = {**_installed("weather"), "presence_runtime": {
            "defaults": {"model_slot": "main", "inline_max_rounds": 10},
            "overrides": {}, "state_fingerprint": "b" * 64,
        }}
        listing = [weather]
        queue = _queue()
        skills = open_skills(page)
        skills.answers["/api/extensions"] = lambda _request: {"skills": listing, "live": {}}
        skills.answers["/api/skills/lifecycle-queue"] = lambda _request: queue
        page.goto(URL, wait_until="networkidle")
        card = page.locator('.skills-card[data-skill="calendar"]')
        card.wait_for()
        assert card.locator('.skills-toggle').count() == 0
        assert card.locator('[data-skill-menu-trigger]').count() == 0
        assert card.locator('.skills-details').count() == 0
        weather_card = page.locator('.skills-card[data-skill="weather"]')
        weather_card.locator('.skills-details > summary').click()
        weather_card.locator('.skills-details').evaluate("node => node.insertAdjacentHTML('beforeend', '<input name=inline_max_rounds>')")
        field = weather_card.locator('[name="inline_max_rounds"]')
        field.fill("27")
        page.evaluate("""() => {
            window.keptCard = document.querySelector('.skills-card[data-skill=weather]');
            window.keptField = keptCard.querySelector('[name=inline_max_rounds]');
            window.keptCalendar = document.querySelector('.skills-card[data-skill=calendar]');
            keptField.focus();
        }""")
        capture(page, browser_name, 1100, "virtual-install")

        calendar = _installed("calendar")
        _automatic_read(page, skills, lambda: listing.append(calendar))
        page.wait_for_function("document.querySelector('.skills-card[data-skill=calendar]').textContent.includes('Installed calendar')")
        capture(page, browser_name, 1100, "installed-still-pending")
        assert card.locator('.skills-toggle').count() == 1
        assert card.locator('[data-skill-menu-trigger]').count() == 1
        assert card.locator('.skills-details').count() == 1
        assert card.locator('.skills-toggle').is_disabled(), "real pending work is not a virtual card"
        assert page.evaluate("keptCard === document.querySelector('.skills-card[data-skill=weather]') && keptField === document.activeElement")
        assert field.input_value() == "27"
        assert page.evaluate("keptCalendar !== document.querySelector('.skills-card[data-skill=calendar]')")
        page.evaluate("window.realCalendar = document.querySelector('.skills-card[data-skill=calendar]')")

        _automatic_read(page, skills, lambda: queue.update(_queue(status="succeeded")))
        page.wait_for_function("!document.querySelector('.skills-card[data-skill=calendar] .skills-toggle').disabled")
        assert page.evaluate("realCalendar === document.querySelector('.skills-card[data-skill=calendar]')")
        assert field.input_value() == "27"
        card.locator('.skills-details > summary').click()
        assert card.locator('.skills-details').get_attribute('open') is not None
        card.locator('[data-skill-menu-trigger]').click()
        page.locator('body > .skills-card-menu-dialog').wait_for()
        page.keyboard.press('Escape')

        def toggle(request):
            calendar["enabled"] = request.post_data_json["enabled"]
            return {"ok": True}

        skills.answers["/api/skills/calendar/toggle"] = toggle
        card.locator('label.ui-switch').click()
        skills.wait_for_count(('POST', '/api/skills/calendar/toggle'), 1)
        page.wait_for_function("document.querySelector('.skills-card[data-skill=calendar] .skills-toggle').checked && !document.querySelector('.skills-card[data-skill=calendar] .skills-toggle').disabled")
        assert skills.posted('/api/skills/calendar/toggle') == [{"enabled": True}]
        assert field.input_value() == "27"
        assert no_overflow(page)
        capture(page, browser_name, 1100, "installed-controls-operable")
    finally:
        if skills:
            skills.close(browser)
        else:
            browser.close()


@pytest.mark.parametrize("browser_name", ["chromium", "webkit"])
@pytest.mark.parametrize("filtered", [False, True])
def test_placeholder_focus_follows_only_a_visible_replacement(skills_browser, browser_name, filtered):
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": 390, "height": 844})
        listing = [_installed("weather")]
        skills = open_skills(page)
        skills.answers['/api/extensions'] = lambda _request: {"skills": listing, "live": {}}
        skills.answers['/api/skills/lifecycle-queue'] = _queue()
        page.goto(URL, wait_until="networkidle")
        card = page.locator('.skills-card[data-skill="calendar"]')
        card.wait_for()
        if filtered:
            page.locator('[data-installed-filter="off"]').click()
        card.locator('.skills-card-name').focus()
        before = page.locator('.skills-scroll').evaluate('(node) => node.scrollTop')
        page.evaluate("window.oldCalendar = document.querySelector('.skills-card[data-skill=calendar]')")
        _automatic_read(page, skills, lambda: listing.append({**_installed("calendar"), "enabled": filtered}))
        page.wait_for_function("document.querySelector('.skills-card[data-skill=calendar]') !== oldCalendar")
        assert card.locator('.skills-toggle').count() == 1
        assert card.is_hidden() == filtered
        assert page.evaluate("document.activeElement === document.querySelector('.skills-card[data-skill=calendar] .skills-card-name')") == (not filtered)
        assert abs(page.locator('.skills-scroll').evaluate('(node) => node.scrollTop') - before) <= 1
        assert no_overflow(page)
        capture(page, browser_name, 390, f"replacement-focus-filtered-{filtered}")
    finally:
        if skills:
            skills.close(browser)
        else:
            browser.close()


@pytest.mark.parametrize("browser_name", ["chromium", "webkit"])
@pytest.mark.parametrize("menu_owner", ["calendar", "weather"])
def test_real_to_virtual_closes_only_its_own_portalled_menu(skills_browser, browser_name, menu_owner):
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": 1100, "height": 750})
        listing = [_installed("calendar"), _installed("weather")]
        queue = _queue(kind="uninstall")
        skills = open_skills(page)
        skills.answers['/api/extensions'] = lambda _request: {"skills": listing, "live": {}}
        skills.answers['/api/skills/lifecycle-queue'] = lambda _request: queue
        page.goto(URL, wait_until="networkidle")
        card = page.locator('.skills-card[data-skill="calendar"]')
        card.locator('.skills-toggle').wait_for()
        page.locator(f'.skills-card[data-skill="{menu_owner}"] [data-skill-menu-trigger]').click()
        popover = page.locator('body > .skills-card-menu-dialog')
        popover.wait_for()
        # WebKit pointer activation need not focus the button: explicitly use the
        # keyboard focus state whose ownership must survive the next remote read.
        popover.locator('button:not([disabled])').first.focus()
        page.evaluate("""() => {
            window.oldCalendar = document.querySelector('.skills-card[data-skill=calendar]');
            window.keptMenu = document.querySelector('body > .skills-card-menu-dialog');
            window.keptMenuFocus = document.activeElement;
            window.cardOrder = [...document.querySelector('#skills-list').children].map(n => n.dataset.skill);
        }""")
        _automatic_read(page, skills, lambda: listing.pop(0))
        page.wait_for_function("document.querySelector('.skills-card[data-skill=calendar]') !== oldCalendar")
        assert card.locator('.skills-toggle').count() == 0
        assert card.locator('[data-skill-menu-trigger]').count() == 0
        assert card.locator('.skills-details').count() == 0
        assert page.evaluate("JSON.stringify(cardOrder) === JSON.stringify([...document.querySelector('#skills-list').children].map(n => n.dataset.skill))")
        if menu_owner == 'calendar':
            assert popover.count() == 0
            assert page.evaluate("!keptMenu.isConnected && document.activeElement === document.querySelector('.skills-card[data-skill=calendar] .skills-card-name')")
        else:
            assert page.evaluate("keptMenu === document.querySelector('body > .skills-card-menu-dialog') && keptMenuFocus === document.activeElement")
            page.keyboard.press('Escape')
        _automatic_read(page, skills, lambda: queue.update(_queue(kind="uninstall", status="succeeded")))
        card.wait_for(state="detached")
        assert page.locator('body > .skills-card-menu-dialog').count() == 0
        capture(page, browser_name, 1100, f"removed-menu-{menu_owner}")
    finally:
        if skills:
            skills.close(browser)
        else:
            browser.close()


QUEUE = "/api/skills/lifecycle-queue"
LISTING = ("GET", "/api/extensions")
NEEDS = '[data-installed-filter="needs"] .skills-category-count'


def _failed(target, kind="enable", error="Port 8080 is busy", **extra):
    return {"id": f"{target}-{kind}-failed", "target": target, "kind": kind, "status": "failed",
            "error": error, "message": "", "source": "skills", **extra}


def _pause(page):
    page.clock.pause_at(page.evaluate("new Date(Date.now() + 60_000).toISOString()"))
    page.wait_for_timeout(300)  # a read the jump itself made due has answered before anything is counted


def test_failed_enable_leaves_a_ready_on_card_only_after_a_fresh_listing(skills_browser):
    """The owner's flow: a failed Enable, then Refresh. No socket event is involved."""
    from playwright.sync_api import expect

    browser = skills_browser.chromium.launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": 390, "height": 844})
        ready, broken = _calendar_health("Port 8080 is busy", companion_reason="load_error")
        weather = {**_installed("weather"), "enabled": True, "live_loaded": True, "live_reason": "ready"}
        off = {**_installed("notes"), "enabled": False}
        events = [_failed("calendar"), _failed("weather", kind="update", error="Update refused"),
                  _failed("notes", error="Notes could not start")]
        skills = open_skills(page, extensions=[broken, weather, off])
        skills.answers[QUEUE] = {"events": events, "active": None}
        page.goto(URL, wait_until="networkidle")
        card = page.locator('.skills-card[data-skill="calendar"]')
        chip = card.locator('.skills-card-status')
        needs = page.locator(NEEDS)
        expect(chip).to_have_text("Couldn’t start")
        expect(needs).to_have_text("3")
        capture(page, "chromium", 390, "failed-enable-before")

        # An unreadable listing is no evidence of recovery: the earlier facts and warnings are kept.
        skills.set_skills([ready, weather, off])
        skills.fail(LISTING[1], "Listing outage")
        refresh(page)
        expect(page.locator('#skills-status')).to_contain_text("Showing the previous list")
        expect(chip).to_have_text("Couldn’t start")
        expect(needs).to_have_text("3")

        # A fresh listing that lists it still unhealthy keeps the failure; one that lists it ready ends only Enable's.
        skills.answers[LISTING[1]] = {"skills": [broken, weather, off], "live": {}}
        refresh(page)
        expect(chip).to_have_text("Couldn’t start")
        skills.answers[LISTING[1]] = {"skills": [ready, weather, off], "live": {}}
        refresh(page)
        expect(chip).to_have_text("")
        expect(card.locator('.skills-load-error')).to_have_count(0)
        expect(card.locator('.skills-toggle')).to_be_checked()
        expect(needs).to_have_text("2")
        expect(page.locator('.skills-card[data-skill="weather"] .skills-card-status')).to_have_text("Update failed")
        expect(page.locator('.skills-card[data-skill="notes"] .skills-card-status')).to_have_text("Enable failed")
        expect(page.locator('.skills-card[data-skill="notes"] .skills-load-error')).to_have_text("Notes could not start")
        assert page.evaluate("fetch('/api/skills/lifecycle-queue').then(r => r.json()).then(q => q.events.some(e => e.target === 'calendar' && e.status === 'failed'))"), \
            "the failed job stays in the queue's history"
        assert not skills.posted('/api/skills/calendar/toggle')
        capture(page, "chromium", 390, "failed-enable-after-fresh-ready")
    finally:
        if skills:
            skills.close(browser)
        else:
            browser.close()


def test_failure_over_an_older_listing_is_answered_by_one_timer_read(skills_browser):
    """The race: a failure reaches the page with a listing that was requested before the failure was seen.

    Socket hints only start the reads that arrange each race. Every clearance below is the card's own confirming
    timer reading the listing again; the settlement with no hint at all is the next case.
    """
    from playwright.sync_api import expect

    browser = skills_browser.chromium.launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": 390, "height": 844})
        page.clock.install()
        ready, broken = _calendar_health("Port 8080 is busy", companion_reason="load_error")
        skills = open_skills(page, extensions=[ready])
        page.goto(URL, wait_until="networkidle")
        card = page.locator('.skills-card[data-skill="calendar"]')
        chip = card.locator('.skills-card-status')
        expect(chip).to_have_text("")
        _pause(page)

        # The listing is held while the failure arrives: its healthy booleans were read before the failure was seen,
        # so they cannot say whether the failure is still true. The card shows it, and reads once more.
        before = skills.requests.count(LISTING)
        skills.hold.add(LISTING[1])
        skills.answers[QUEUE] = {"events": [_failed("calendar")], "active": None}
        page.evaluate("skillSocketEvent('extension_lifecycle')")
        page.clock.run_for(350)
        skills.release(LISTING[1], {"skills": [ready], "live": {}})
        skills.hold.clear()
        expect(chip).to_have_text("Enable failed")
        expect(card.locator('.skills-load-error')).to_have_text("Port 8080 is busy")
        expect(page.locator(NEEDS)).to_have_text("1")
        assert skills.requests.count(LISTING) == before + 1
        capture(page, "chromium", 390, "failure-over-older-listing")

        # The newer read finds it still unhealthy: the failure stays, and nothing polls it in a loop.
        skills.set_skills([broken])
        page.clock.run_for(3000)
        expect(chip).to_have_text("Couldn’t start")
        assert skills.requests.count(LISTING) == before + 2
        page.clock.run_for(60_000)
        assert skills.requests.count(LISTING) == before + 2, "an unhealthy row is not polled for recovery"

        # The other order: the listing lands first and the queue, held, brings a second failure after it.
        skills.set_skills([ready])
        skills.hold.add(QUEUE)
        skills.answers[QUEUE] = {"events": [_failed("calendar"), _failed("calendar", error="Port 8081 is busy", id="second-failure")],
                                 "active": None}
        count = skills.requests.count(LISTING)
        page.evaluate("skillSocketEvent('extension_lifecycle')")
        page.clock.run_for(350)
        skills.wait_for_count(LISTING, count + 1)
        skills.wait_for_request(QUEUE)
        page.wait_for_timeout(200)
        expect(card.locator('.skills-load-error')).to_have_count(0)  # the first failure is answered; the second has not arrived
        skills.release(QUEUE)
        skills.hold.clear()
        expect(chip).to_have_text("Enable failed")
        expect(card.locator('.skills-load-error')).to_have_text("Port 8081 is busy")
        count = skills.requests.count(LISTING)
        page.clock.run_for(3000)
        skills.wait_for_count(LISTING, count + 1)
        expect(chip).to_have_text("")
        assert skills.requests.count(LISTING) == count + 1
        page.clock.run_for(60_000)
        assert skills.requests.count(LISTING) == count + 1, "an answered failure schedules no more reads"

        # A newest running job is never replaced by an older answered failure.
        skills.answers[QUEUE] = {"events": [_failed("calendar"), {"id": "retry", "target": "calendar", "kind": "enable",
                                                                  "status": "running", "message": "Turning on calendar"}], "active": None}
        page.evaluate("skillSocketEvent('extension_lifecycle')")
        page.clock.run_for(350)
        expect(chip).to_contain_text("Turning on calendar")
        expect(card.locator('.skills-toggle')).to_be_disabled()
    finally:
        if skills:
            skills.close(browser)
        else:
            browser.close()


def test_an_unreadable_queue_keeps_the_marks_a_newer_listing_answers_with(skills_browser):
    """A failed optional queue read neither forgets what the page saw nor shows a failure it never saw."""
    from playwright.sync_api import expect

    browser = skills_browser.chromium.launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": 390, "height": 844})
        page.clock.install()
        ready, _ = _calendar_health()
        skills = open_skills(page, extensions=[ready])
        page.goto(URL, wait_until="networkidle")
        card = page.locator('.skills-card[data-skill="calendar"]')
        chip = card.locator('.skills-card-status')
        _pause(page)

        # The failure is on screen, with its confirming read still waiting on the timer.
        skills.answers[QUEUE] = {"events": [_failed("calendar")], "active": None}
        page.evaluate("skillSocketEvent('extension_lifecycle')")
        page.clock.run_for(350)
        expect(chip).to_have_text("Enable failed")

        # A read whose queue cannot be read keeps the queue and the marks it had; its listing began after the failure
        # was seen, so it answers it.
        skills.fail(QUEUE, "Queue unavailable")
        page.evaluate("skillSocketEvent('extension_health_changed')")
        page.clock.run_for(350)
        expect(page.locator('#skills-status')).to_contain_text("Lifecycle progress could not be refreshed")
        expect(chip).to_have_text("")
        expect(card.locator('.skills-load-error')).to_have_count(0)
        capture(page, "chromium", 390, "queue-unreadable-listing-answers")

        # The failure is answered for good: nothing is left to poll for.
        before = skills.requests.count(LISTING)
        page.clock.run_for(60_000)
        assert skills.requests.count(LISTING) == before, "nothing polls an answered failure"
        expect(chip).to_have_text("")
    finally:
        if skills:
            skills.close(browser)
        else:
            browser.close()


def test_failed_enable_stays_without_affirmative_extension_readiness_or_job_identity(skills_browser):
    from playwright.sync_api import expect

    browser = skills_browser.chromium.launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": 390, "height": 1100})
        page.clock.install()
        ready, _ = _calendar_health()
        def on(name, **facts):
            return {**_installed(name), "enabled": True, **facts}

        rows = [
            ready,  # the control: it is answered, so the others stay for the rule's reason alone
            on("script_tool", type="script", live_reason="not_extension"),
            on("quiet"),  # an extension the host says nothing of
            on("odd", live_loaded=True, live_reason="something_new"),
            on("anon", live_loaded=True, live_reason="ready"),
        ]
        events = [_failed(row["name"]) for row in rows if row["name"] != "anon"]
        events.append({key: value for key, value in _failed("anon").items() if key != "id"})  # no identity to prove an answer by
        skills = open_skills(page, extensions=rows)
        skills.answers[QUEUE] = {"events": events, "active": None}
        page.goto(URL, wait_until="networkidle")
        chips = {row["name"]: page.locator(f'.skills-card[data-skill="{row["name"]}"] .skills-card-status') for row in rows}
        for name in ("script_tool", "quiet", "odd", "anon"):
            expect(chips[name]).to_have_text("Enable failed")
            expect(page.locator(f'.skills-card[data-skill="{name}"] .skills-load-error')).to_have_text("Port 8080 is busy")

        # The control's confirming read, which the jump to a paused clock runs once, is the only one the page makes
        # on its own, and it answers the control alone: the others show a failure nothing can prove answered.
        _pause(page)
        expect(chips["calendar"]).to_have_text("")
        for name in ("script_tool", "quiet", "odd", "anon"):
            expect(chips[name]).to_have_text("Enable failed")
        expect(page.locator(NEEDS)).to_have_text("4")
        before = skills.requests.count(LISTING)
        assert before == 2, "the first read and the control's single confirming read"
        page.clock.run_for(60_000)
        assert skills.requests.count(LISTING) == before, "no row is polled for a recovery nothing can prove"

        # Further fresh listings change none of it.
        for _ in range(2):
            page.locator("#skills-refresh").click()
            skills.wait_for_count(LISTING, before + 1)
            before += 1
            expect(page.locator("#skills-refresh")).to_be_enabled()
        for name in ("script_tool", "quiet", "odd", "anon"):
            expect(chips[name]).to_have_text("Enable failed")
        expect(chips["calendar"]).to_have_text("")
        capture(page, "chromium", 390, "failed-enable-kept-without-proof")
    finally:
        if skills:
            skills.close(browser)
        else:
            browser.close()


def test_failed_enable_and_retained_exit_clear_on_timers_with_no_socket_hint(skills_browser):
    """The settlement a card makes by itself: no hint reaches the page at any point, and nothing here sends one."""
    from playwright.sync_api import expect

    browser = skills_browser.chromium.launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": 390, "height": 844})
        page.clock.install()
        ready, _ = _calendar_health()
        exiting = {**ready, "retained_companions": {"daemon": "stop requested"}}
        skills = open_skills(page, extensions=[exiting])
        skills.answers[QUEUE] = {"events": [_failed("calendar")], "active": None}
        page.goto(URL, wait_until="networkidle")
        page.evaluate("""() => {
            window.hints = 0;
            const send = window.skillSocketEvent;
            window.skillSocketEvent = (type) => { window.hints += 1; send(type); };
        }""")
        card = page.locator('.skills-card[data-skill="calendar"]')
        chip = card.locator('.skills-card-status')
        expect(chip).to_have_text("Enable failed")
        expect(card).to_contain_text("Waiting for daemon to exit.")
        expect(page.locator(NEEDS)).to_have_text("1")
        _pause(page)
        capture(page, "chromium", 390, "retained-exit-failed-enable")

        # Between its reads the card only waits; at the cadence it reads, and a process still exiting keeps both facts.
        before = skills.requests.count(LISTING)
        page.clock.run_for(12_000)
        assert skills.requests.count(LISTING) == before, "the listing is not read between cadences"
        page.clock.run_for(3_000)
        skills.wait_for_count(LISTING, before + 1)
        expect(card).to_contain_text("Waiting for daemon to exit.")
        expect(chip).to_have_text("Enable failed")

        # The process exits. The card's own timer reads the listing and settles both facts.
        skills.set_skills([ready])
        count = skills.requests.count(LISTING)
        page.clock.run_for(15_000)
        skills.wait_for_count(LISTING, count + 1)
        expect(card).not_to_contain_text("Waiting for daemon to exit.")
        expect(chip).to_have_text("")
        expect(card.locator('.skills-load-error')).to_have_count(0)
        expect(page.locator('[data-installed-filter="needs"]')).to_have_count(0)
        settled = skills.requests.count(LISTING)
        page.clock.run_for(60_000)
        assert skills.requests.count(LISTING) == settled, "a settled page schedules nothing"
        assert page.evaluate("hints") == 0, "no socket hint was part of this"
        capture(page, "chromium", 390, "retained-exit-settled")
    finally:
        if skills:
            skills.close(browser)
        else:
            browser.close()


def test_a_resolved_failed_enable_stays_resolved_when_another_companion_faults_later(skills_browser):
    """The owner's sequence in the real page: Enable fails, a fresh listing shows the skill ready, then a second
    companion stops. The card shows that companion's fault, never the Enable error a listing already retired; a
    later, different failed Enable still shows, and the first job stays in the queue's history throughout."""
    from playwright.sync_api import expect

    browser = skills_browser.chromium.launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": 390, "height": 844})
        ready, broken = _calendar_health("Port 8080 is busy", companion_reason="load_error")
        beta_stopped = {**ready, "live_reason": "companion_restart_exhausted:beta",
                        "load_error": "companion 'beta' exhausted its restart budget"}
        skills = open_skills(page, extensions=[broken])
        skills.answers[QUEUE] = {"events": [_failed("calendar")], "active": None}
        page.goto(URL, wait_until="networkidle")
        card = page.locator('.skills-card[data-skill="calendar"]')
        chip = card.locator('.skills-card-status')
        error = card.locator('.skills-load-error')
        expect(chip).to_have_text("Couldn’t start")
        expect(error).to_have_text("Port 8080 is busy")

        # A fresh ready listing retires the failed Enable (the card also reads it once more by itself).
        skills.set_skills([ready])
        refresh(page)
        expect(chip).to_have_text("")
        expect(error).to_have_count(0)

        # Beta then stops: its own fault is shown, and the retired Enable error is not brought back with it.
        skills.set_skills([beta_stopped])
        refresh(page)
        expect(chip).to_have_text("Background process stopped")
        expect(card).to_contain_text("companion 'beta' exhausted its restart budget")
        expect(card).not_to_contain_text("Port 8080 is busy")
        expect(card.locator('.skills-toggle')).to_be_checked()
        refresh(page)
        expect(chip).to_have_text("Background process stopped")
        expect(card).not_to_contain_text("Port 8080 is busy")

        # The same holds for a pending exit of the other companion.
        skills.set_skills([{**ready, "retained_companions": {"beta": "stop requested"}}])
        refresh(page)
        expect(card).to_contain_text("Waiting for beta to exit.")
        expect(card).not_to_contain_text("Port 8080 is busy")
        expect(chip).not_to_have_text("Enable failed")
        expect(chip).not_to_have_text("Couldn’t start")

        # A different Enable that fails afterwards is a new job: it shows, and the first stays in the history.
        second = _failed("calendar", error="Port 9090 is busy", id="calendar-enable-failed-2")
        skills.answers[QUEUE] = {"events": [_failed("calendar"), second], "active": None}
        skills.set_skills([beta_stopped])
        refresh(page)
        expect(card).to_contain_text("Port 9090 is busy")
        expect(card).not_to_contain_text("Port 8080 is busy")
        expect(chip).not_to_have_text("")
        assert page.evaluate("fetch('/api/skills/lifecycle-queue').then(r => r.json())"
                             ".then(q => q.events.filter(e => e.target === 'calendar' && e.status === 'failed').length)") == 2, \
            "the resolved job stays in the queue's history"
        assert not skills.posted('/api/skills/calendar/toggle')
        capture(page, "chromium", 390, "failed-enable-resolved-then-second-companion")
    finally:
        if skills:
            skills.close(browser)
        else:
            browser.close()


ANSWERED_KEY = "ouroboros.skills.answeredFailures"
HISTORY = ("fetch('/api/skills/lifecycle-queue').then(r => r.json())"
           ".then(q => q.events.filter(e => e.target === 'calendar' && e.status === 'failed').map(e => e.id))")


def _two_calendar_directories():
    """A second directory named calendar: the host lists every occupant of the name as its own colliding row, none live."""
    collision = {**_installed("calendar"), "identity_collision": True, "source": "user_repo", "type": "instruction",
                 "load_error": "Skill name collision", "removal": "", "live_reason": "load_error",
                 "manifest_file": ""}  # the producer's placeholder: these folders' real declarations were not read
    return ({**collision, "collision_row_id": "a" * 64, "payload_root": "skills/external/calendar"},
            {**collision, "collision_row_id": "b" * 64, "payload_root": "skills/clawhub/calendar"})


def _kept(page):
    """What this tab keeps of the answered jobs (parsed), or None."""
    return page.evaluate("key => JSON.parse(sessionStorage.getItem(key) || 'null')", ANSWERED_KEY)


def _wait_kept(page, expected):
    page.wait_for_function("([key, expected]) => JSON.stringify(JSON.parse(sessionStorage.getItem(key) || 'null')) === JSON.stringify(expected)",
                           arg=[ANSWERED_KEY, expected])


def test_a_resolved_failed_enable_is_not_moved_onto_a_row_by_a_second_directory_of_its_name(skills_browser):
    """The same job, once answered, is settled by its queue id before the join by name looks for a row: a second
    directory taking the name, and the order of the two, must not bring its words back on whichever row is last. A
    different failed Enable still shows, once, as an attempt of its own beside the two directories and on neither, in
    either order; the answered job stays in the queue's history."""
    from playwright.sync_api import expect

    browser = skills_browser.chromium.launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": 390, "height": 844})
        ready, broken = _calendar_health("Port 8080 is busy", companion_reason="load_error")
        first, second = _two_calendar_directories()
        skills = open_skills(page, extensions=[broken])
        skills.answers[QUEUE] = {"events": [_failed("calendar")], "active": None}
        page.goto(URL, wait_until="networkidle")
        listing = page.locator("#skills-list")
        cards = page.locator('.skills-card[data-skill="calendar"]')
        expect(listing).to_contain_text("Port 8080 is busy")

        skills.set_skills([ready])
        refresh(page)
        expect(listing).not_to_contain_text("Port 8080 is busy")

        # A second directory takes the name: both occupants are listed as colliding rows, and the answered job belongs
        # to neither, whichever the listing puts last.
        for rows in ([first, second], [second, first], [first, second]):
            skills.set_skills(rows)
            refresh(page)
            expect(cards).to_have_count(2)
            expect(listing).not_to_contain_text("Port 8080 is busy")
            expect(listing).not_to_contain_text("Enable failed")

        # Each colliding folder opens by its exact identity and states no access: its placeholder manifest declares nothing.
        skills.set_skills([first, second])
        refresh(page)
        sheet = page.locator(".skills-detail-sheet")
        for row in (first, second):
            card = page.locator(f'.skills-card[data-skill-row="collision:{row["collision_row_id"]}"]')
            expect(card).not_to_contain_text("Can use")
            card.locator(".skills-card-name").click()
            sheet.locator(".skills-sheet-details > summary").click()
            expect(sheet.locator(".skills-sheet-facts")).to_contain_text(row["payload_root"])
            expect(sheet).not_to_contain_text("What it can use")
            expect(sheet).not_to_contain_text("Nothing beyond its own folder")
            expect(sheet).not_to_contain_text("Your keys")
            sheet.locator("[data-sheet-close]").click()
            expect(sheet).to_have_count(0)

        # The directories go again and the skill is ready: still answered.
        skills.set_skills([ready])
        refresh(page)
        expect(cards).to_have_count(1)
        expect(listing).not_to_contain_text("Port 8080 is busy")

        # A different Enable that fails afterwards is a new job. It names no directory, so it shows once as its own
        # attempt beside the two, whichever order the listing gives them: neither directory carries it.
        skills.answers[QUEUE] = {"events": [_failed("calendar"), _failed("calendar", error="Port 9090 is busy", id="calendar-enable-failed-2")],
                                 "active": None}
        directories = [page.locator(f'.skills-card[data-skill-row="collision:{row["collision_row_id"]}"]') for row in (first, second)]
        attempt = page.locator('.skills-card[data-skill="calendar"][data-lifecycle-virtual]')
        for rows in ([first, second], [second, first], [first, second]):
            skills.set_skills(rows)
            refresh(page)
            expect(cards).to_have_count(3)
            for directory in directories:
                expect(directory).to_have_count(1)
                expect(directory).not_to_contain_text("Port 9090 is busy")
                expect(directory).not_to_contain_text("Enable failed")
            expect(attempt).to_have_count(1)
            expect(attempt).to_contain_text("Enable failed")
            expect(attempt).to_contain_text("Port 9090 is busy")
            expect(attempt).to_contain_text("this is not one of them")
            expect(attempt.locator(".skills-toggle, [data-skill-menu-trigger], .skills-details")).to_have_count(0)
            expect(listing).not_to_contain_text("Port 8080 is busy")
        assert page.evaluate(HISTORY) == ["calendar-enable-failed", "calendar-enable-failed-2"], "the answered job stays in the history"
        assert not skills.posts, "nothing was enabled again"
        capture(page, "chromium", 390, "failed-enable-resolved-then-name-collision")
    finally:
        if skills:
            skills.close(browser)
        else:
            browser.close()


def test_a_name_only_attempt_read_beside_two_directories_stays_off_the_one_left_when_the_other_is_renamed(skills_browser):
    """The supported sequence with both real directories. A (external) and B (clawhub) carry the name `calendar`, and an Enable
    job that names only the skill is read as failed beside them: an attempt of its own, on neither. A is renamed away, so B is
    listed alone, still Off. The attempt stays beside B in words that no longer say two folders exist, while B keeps its own
    state, its own Details and its place in the page across the next repaint and a reload of the same tab. B turning ready and
    On answers nothing: the attempt leaves only with a newer job of its name. An earlier failed Enable that a fresh listing
    answered stays retired, and the queue's history stays the queue's. An active Enable and its shadowed queued successor
    both keep their ambiguity after one directory goes, through unreadable queue replies, reload and eventual failure."""
    from playwright.sync_api import expect

    browser = skills_browser.chromium.launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": 390, "height": 844})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        ready, broken = _calendar_health("Port 8080 is busy", companion_reason="load_error")
        external, clawhub = _two_calendar_directories()  # A and B
        # B as the host lists it once A is renamed away: the same folder, alone under the name, with no collision marks.
        survivor = {**_installed("calendar"), "payload_root": clawhub["payload_root"]}
        survivor_on = {**ready, "payload_root": clawhub["payload_root"]}
        answered = _failed("calendar")
        attempt_job = _failed("calendar", error="Port 9090 is busy", id="calendar-enable-failed-2")
        skills = open_skills(page, extensions=[broken])
        skills.answers[QUEUE] = {"events": [answered], "active": None}
        page.goto(URL, wait_until="networkidle")
        listing = page.locator("#skills-list")
        cards = page.locator('.skills-card[data-skill="calendar"]')
        attempt = page.locator('.skills-card[data-skill="calendar"][data-lifecycle-virtual]')
        folder = page.locator('.skills-card[data-skill="calendar"]:not([data-lifecycle-virtual])')
        expect(listing).to_contain_text("Port 8080 is busy")

        # An earlier failed Enable, answered by a fresh listing, is retired for good.
        skills.set_skills([ready])
        refresh(page)
        expect(listing).not_to_contain_text("Port 8080 is busy")
        _wait_kept(page, ["calendar-enable-failed"])

        # A later Enable is read as failed beside the two directories: it names neither, so it is a row of its own.
        skills.answers[QUEUE] = {"events": [answered, attempt_job], "active": None}
        skills.set_skills([external, clawhub])
        refresh(page)
        expect(attempt).to_contain_text("Port 9090 is busy")
        expect(attempt).to_contain_text("this is not one of them")
        expect(cards).to_have_count(3)
        expect(attempt.locator('.skills-card-name')).to_have_text('calendar')
        expect(attempt.locator('.skills-card-name')).not_to_have_css('cursor', 'pointer')
        attempt.locator('.skills-card-name').click()
        expect(page.locator('.skills-detail-sheet')).to_have_count(0)
        for row in (external, clawhub):
            expect(page.locator(f'.skills-card[data-skill-row="collision:{row["collision_row_id"]}"]')).not_to_contain_text("Port 9090 is busy")

        def attempt_stands_beside_the_one_left(*, on=False):
            expect(cards).to_have_count(2)
            expect(attempt).to_have_count(1)
            expect(attempt).to_contain_text("Enable failed")
            expect(attempt).to_contain_text("Port 9090 is busy")
            expect(attempt).to_contain_text("seen while more than one installed folder shared the name")
            expect(attempt).not_to_contain_text("More than one installed folder uses it")
            expect(attempt).not_to_contain_text("Nothing was installed")
            # A placeholder: no switch, menu or Details, nothing that operates on the folder of its name.
            expect(attempt.locator(".skills-toggle, [data-skill-menu-trigger], .skills-details")).to_have_count(0)
            expect(folder).to_have_count(1)
            if on:
                expect(folder.locator(".skills-toggle")).to_be_checked()
            else:
                expect(folder.locator(".skills-toggle")).not_to_be_checked()
            expect(folder).not_to_contain_text("Enable failed")
            expect(folder).not_to_contain_text("Port 9090 is busy")
            expect(folder.locator(".skills-load-error")).to_have_count(0)
            expect(listing).not_to_contain_text("Port 8080 is busy")

        # A is renamed away: B is listed alone, still Off, and carries none of the job.
        skills.set_skills([survivor])
        refresh(page)
        attempt_stands_beside_the_one_left()
        capture(page, "chromium", 390, "attempt-beside-the-one-directory-left")

        # B's own Details open B's folder, not the attempt's.
        expect(folder.locator('.skills-card-name')).to_have_css('cursor', 'pointer')
        folder.locator(".skills-card-name").click()
        sheet = page.locator(".skills-detail-sheet")
        sheet.locator(".skills-sheet-details > summary").click()
        expect(sheet.locator(".skills-sheet-facts")).to_contain_text(clawhub["payload_root"])
        expect(sheet.locator(".skills-sheet-facts")).not_to_contain_text(external["payload_root"])
        sheet.locator("[data-sheet-close]").click()
        expect(sheet).to_be_hidden()

        # The same tab, reloaded while the queue still lists the job: what the page read is kept, so the attempt does not move.
        page.reload(wait_until="networkidle")
        attempt_stands_beside_the_one_left()

        # B turns ready and On. The next ordinary repaint (a socket hint, so nothing takes the focus) patches the same cards in
        # place and keeps the focus on B's name; no listed row can answer the attempt, so it stays.
        folder.locator(".skills-card-name").focus()
        page.evaluate("""() => { window.cardsBefore = [...document.querySelectorAll('#skills-list > .skills-card[data-skill="calendar"]')]; }""")
        skills.set_skills([survivor_on])
        _automatic_read(page, skills, lambda: page.evaluate("skillSocketEvent('extension_lifecycle')"))
        expect(folder.locator(".skills-toggle")).to_be_checked()
        attempt_stands_beside_the_one_left(on=True)
        assert page.evaluate("""() => {
            const now = [...document.querySelectorAll('#skills-list > .skills-card[data-skill="calendar"]')];
            const kept = cardsBefore.find((card) => !card.hasAttribute('data-lifecycle-virtual'));
            return now.length === cardsBefore.length && now.every((card, at) => card === cardsBefore[at])
                && document.activeElement === kept.querySelector('.skills-card-name');
        }"""), "the same cards stay, and the focus stays on B's name"

        # B goes too: the attempt keeps its reading, its words and its own card, and no folder is left to take it.
        skills.set_skills([])
        refresh(page)
        expect(folder).to_have_count(0)
        expect(cards).to_have_count(1)
        expect(attempt).to_contain_text("seen while more than one installed folder shared the name")
        assert page.evaluate("""() => cardsBefore.find((card) => card.hasAttribute('data-lifecycle-virtual'))
            === document.querySelector('.skills-card[data-skill="calendar"][data-lifecycle-virtual]')"""), "the attempt's card stays"

        # A newer Enable that fails is an ordinary job of the one directory (Off again): the older attempt is no longer the
        # newest job of the name and is gone, and nothing of the retired job comes back.
        third = _failed("calendar", error="Port 7070 is busy", id="calendar-enable-failed-3")
        skills.answers[QUEUE] = {"events": [answered, attempt_job, third], "active": None}
        skills.set_skills([survivor])
        refresh(page)
        expect(folder.locator(".skills-load-error")).to_have_text("Port 7070 is busy")
        expect(attempt).to_have_count(0)
        expect(cards).to_have_count(1)
        for words in ("Port 9090 is busy", "Port 8080 is busy"):
            expect(listing).not_to_contain_text(words)

        # The newest status speaks for the name: a settled job leaves nothing of any older failure.
        settled = {"id": "calendar-enable-4", "target": "calendar", "kind": "enable", "status": "succeeded", "error": "", "message": ""}
        skills.answers[QUEUE] = {"events": [answered, attempt_job, third, settled], "active": None}
        refresh(page)
        expect(folder.locator(".skills-load-error")).to_have_count(0)
        expect(cards).to_have_count(1)
        for words in ("Port 7070 is busy", "Port 9090 is busy", "Port 8080 is busy", "Enable failed"):
            expect(listing).not_to_contain_text(words)
        assert page.evaluate(HISTORY) == ["calendar-enable-failed", "calendar-enable-failed-2", "calendar-enable-failed-3"], \
            "every failed job stays in the queue's history"

        # The active attempt wins the display over the queued one. Both were read beside both directories.
        live = {"id": "calendar-enable-live", "target": "calendar", "kind": "enable",
                "status": "running", "error": "", "message": "Turning on calendar"}
        queued = {**live, "id": "calendar-enable-queued", "status": "queued", "message": "Next Enable queued"}
        history = [answered, attempt_job, third, settled, live, queued]
        skills.answers[QUEUE] = {"events": history, "active": live}
        skills.set_skills([external, clawhub])
        refresh(page)
        expect(cards).to_have_count(3)
        expect(attempt).to_contain_text("this is not one of them")
        expect(attempt).not_to_contain_text("Port 9090 is busy")  # the newest job of the name speaks
        expect(attempt).to_contain_text("Turning on")
        expect(attempt).not_to_contain_text("Next Enable queued")
        page.wait_for_function("['calendar-enable-live', 'calendar-enable-queued'].every(id => JSON.parse("
                               "sessionStorage.getItem('ouroboros.skills.nameLevelFailures') || '[]').includes(id))")
        for row in (external, clawhub):
            directory = page.locator(f'.skills-card[data-skill-row="collision:{row["collision_row_id"]}"]')
            expect(directory).not_to_contain_text("Turning on")

        skills.set_skills([survivor])
        refresh(page)
        expect(cards).to_have_count(2)
        expect(attempt).to_have_count(1)
        expect(attempt).to_contain_text("Turning on")
        expect(attempt).to_contain_text("seen while more than one installed folder shared the name")
        expect(attempt).not_to_contain_text("Enable failed")
        expect(folder.locator(".skills-toggle")).to_be_enabled()
        expect(folder.locator(".skills-toggle")).not_to_be_checked()
        expect(folder).not_to_contain_text("Turning on")
        expect(folder).not_to_contain_text("Enable failed")

        # The active job finishes and the queued attempt becomes visible, without inheriting the surviving folder.
        history[-2] = {**live, "status": "succeeded"}
        skills.answers[QUEUE] = {"events": history, "active": None}
        refresh(page)
        expect(cards).to_have_count(2)
        expect(attempt).to_have_attribute("data-skill-row", "attempt:calendar")
        expect(attempt).to_contain_text("Turning on")
        expect(attempt).to_contain_text("seen while more than one installed folder shared the name")
        expect(attempt).not_to_contain_text("Enable failed")
        expect(folder.locator(".skills-toggle")).to_be_enabled()
        expect(folder.locator(".skills-toggle")).not_to_be_checked()

        # Neither a null queue body nor an unreadable queue on reload prunes the queued mark.
        for answer in (lambda _request: None, (503, {"error": "Queue unavailable"})):
            skills.answers[QUEUE] = answer
            page.reload(wait_until="networkidle")
            expect(page.locator("#skills-status")).to_contain_text("Lifecycle progress could not be refreshed")
            assert page.evaluate("JSON.parse(sessionStorage.getItem('ouroboros.skills.nameLevelFailures')).includes('calendar-enable-queued')")

        # The same tab reads the queued job now running; the remembered ambiguity keeps it off the survivor.
        running = {**queued, "status": "running", "message": "Turning on calendar"}
        skills.answers[QUEUE] = {"events": [*history[:-1], running], "active": running}
        page.reload(wait_until="networkidle")
        expect(attempt).to_have_count(1)
        expect(attempt).to_contain_text("seen while more than one installed folder shared the name")
        expect(folder.locator(".skills-toggle")).to_be_enabled()
        expect(folder).not_to_contain_text("Turning on")

        # The running job fails: still its own attempt with its own error; the survivor carries none of it.
        history[-1] = {**running, "status": "failed", "error": "Port 9091 is busy"}
        skills.answers[QUEUE] = {"events": history, "active": None}
        refresh(page)
        expect(attempt).to_contain_text("Enable failed")
        expect(attempt).to_contain_text("Port 9091 is busy")
        expect(attempt).to_contain_text("seen while more than one installed folder shared the name")
        expect(folder.locator(".skills-toggle")).to_be_enabled()
        expect(folder.locator(".skills-toggle")).not_to_be_checked()
        expect(folder).not_to_contain_text("Port 9091 is busy")
        expect(folder).not_to_contain_text("Enable failed")
        expect(folder.locator(".skills-load-error")).to_have_count(0)
        assert page.evaluate(HISTORY) == ["calendar-enable-failed", "calendar-enable-failed-2",
                                          "calendar-enable-failed-3", "calendar-enable-queued"], \
            "the failed running job joins the queue's history"
        capture(page, "chromium", 390, "running-attempt-beside-the-one-directory-left")
        assert not skills.posts, "nothing was enabled"
        assert errors == []
    finally:
        if skills:
            skills.close(browser)
        else:
            browser.close()


def test_a_resolved_failed_enable_stays_resolved_when_the_tab_is_reloaded(skills_browser):
    """The owner's second sequence: Enable fails, a fresh listing shows the skill ready, a companion stops, and the same
    tab is reloaded while the queue still lists the job. The page forgot the answer; the tab kept it. Another skill's
    failure that no listing answered, and a later Enable that fails, still show."""
    from playwright.sync_api import expect

    browser = skills_browser.chromium.launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": 390, "height": 1100})
        ready, broken = _calendar_health("Port 8080 is busy", companion_reason="load_error")
        beta_stopped = {**ready, "live_reason": "companion_restart_exhausted:beta",
                        "load_error": "companion 'beta' exhausted its restart budget"}
        off = {**_installed("notes"), "enabled": False}
        # An older failed update of calendar sits behind the newest job: only the newest job of a name speaks, and no
        # listing can answer an update.
        events = [_failed("calendar", kind="update", error="Older update failure"), _failed("calendar"),
                  _failed("notes", error="Notes could not start")]
        skills = open_skills(page, extensions=[broken, off])
        skills.answers[QUEUE] = {"events": events, "active": None}
        page.goto(URL, wait_until="networkidle")
        card = page.locator('.skills-card[data-skill="calendar"]')
        chip = card.locator('.skills-card-status')
        notes = page.locator('.skills-card[data-skill="notes"]')
        expect(card.locator('.skills-load-error')).to_have_text("Port 8080 is busy")

        skills.set_skills([ready, off])
        refresh(page)
        expect(chip).to_have_text("")
        skills.set_skills([beta_stopped, off])
        refresh(page)
        expect(chip).to_have_text("Background process stopped")
        expect(card).not_to_contain_text("Port 8080 is busy")

        page.reload(wait_until="networkidle")
        for _ in range(2):  # as the page opens, and after the owner's Refresh
            expect(chip).to_have_text("Background process stopped")
            expect(card).to_contain_text("companion 'beta' exhausted its restart budget")
            expect(card).not_to_contain_text("Port 8080 is busy")
            expect(card).not_to_contain_text("Older update failure")  # the answered newest job does not expose it
            expect(notes.locator('.skills-card-status')).to_have_text("Enable failed")
            expect(notes.locator('.skills-load-error')).to_have_text("Notes could not start")
            refresh(page)
        expect(chip).to_have_text("Background process stopped")
        expect(card).not_to_contain_text("Port 8080 is busy")

        # A later Enable that fails is another job: it shows, and the earlier ones stay in the history.
        skills.answers[QUEUE] = {"events": [*events, _failed("calendar", error="Port 9090 is busy", id="calendar-enable-failed-2")],
                                 "active": None}
        refresh(page)
        expect(card).to_contain_text("Port 9090 is busy")
        expect(card).not_to_contain_text("Port 8080 is busy")
        expect(card).not_to_contain_text("Older update failure")
        assert page.evaluate(HISTORY) == ["calendar-update-failed", "calendar-enable-failed", "calendar-enable-failed-2"]
        assert not skills.posts, "no Enable was pressed: nothing was posted"
        capture(page, "chromium", 390, "failed-enable-resolved-then-reload")

        # No answer recorded, no suppression: a tab that never saw the listing answer shows both, ready or not.
        page.evaluate("key => sessionStorage.removeItem(key)", ANSWERED_KEY)
        skills.answers[QUEUE] = {"events": events, "active": None}
        page.reload(wait_until="networkidle")
        expect(card).to_contain_text("Port 8080 is busy")
        expect(card).to_contain_text("companion 'beta' exhausted its restart budget")
    finally:
        if skills:
            skills.close(browser)
        else:
            browser.close()


def test_the_answered_marks_of_a_tab_leave_only_with_their_jobs_in_an_accepted_queue_read(skills_browser):
    """What the tab keeps is the answered job ids and nothing else. A queue that cannot be read takes none away, before
    or after a reload; an accepted read that no longer lists a job as failed takes it; an id the queue never listed
    hides no failure and goes at the first accepted read."""
    from playwright.sync_api import expect

    browser = skills_browser.chromium.launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": 390, "height": 844})
        ready, broken = _calendar_health("Port 8080 is busy", companion_reason="load_error")
        beta_stopped = {**ready, "live_reason": "companion_restart_exhausted:beta",
                        "load_error": "companion 'beta' exhausted its restart budget"}
        failed = {"events": [_failed("calendar")], "active": None}
        skills = open_skills(page, extensions=[broken])
        skills.answers[QUEUE] = failed
        page.goto(URL, wait_until="networkidle")
        card = page.locator('.skills-card[data-skill="calendar"]')
        chip = card.locator('.skills-card-status')
        status = page.locator("#skills-status")
        expect(card.locator('.skills-load-error')).to_have_text("Port 8080 is busy")
        assert _kept(page) is None, "nothing is kept before a listing has answered"

        skills.set_skills([ready])
        refresh(page)
        expect(chip).to_have_text("")
        _wait_kept(page, ["calendar-enable-failed"])

        # An unreadable queue takes nothing away, and neither does a reload while it stays unreadable.
        skills.fail(QUEUE, "Queue unavailable")
        refresh(page)
        expect(status).to_contain_text("Lifecycle progress could not be refreshed")
        assert _kept(page) == ["calendar-enable-failed"]
        page.reload(wait_until="networkidle")
        expect(status).to_contain_text("Lifecycle progress could not be refreshed")
        assert _kept(page) == ["calendar-enable-failed"]

        # The queue comes back listing the job: the answer still holds, though the row is no longer ready.
        skills.set_skills([beta_stopped])
        skills.answers[QUEUE] = failed
        refresh(page)
        expect(status).to_have_text("")  # the accepted queue read has been applied
        expect(chip).to_have_text("Background process stopped")
        expect(card).not_to_contain_text("Port 8080 is busy")
        assert _kept(page) == ["calendar-enable-failed"]

        # The queue no longer lists the job as failed (it settled, or left the bounded queue): the accepted read takes the mark.
        skills.answers[QUEUE] = {"events": [], "active": None}
        refresh(page)
        _wait_kept(page, None)

        # An id the queue never listed (and entries that are no ids) hides nothing, and goes at the first accepted read.
        page.evaluate("key => sessionStorage.setItem(key, JSON.stringify(['ghost-job', 7, '', null]))", ANSWERED_KEY)
        skills.set_skills([broken])
        skills.answers[QUEUE] = {"events": [_failed("calendar", id="a-new-job")], "active": None}
        page.reload(wait_until="networkidle")
        expect(card.locator('.skills-load-error')).to_have_text("Port 8080 is busy")
        _wait_kept(page, None)
        assert not skills.posts
    finally:
        if skills:
            skills.close(browser)
        else:
            browser.close()


def test_a_session_store_that_refuses_leaves_the_answer_to_this_page_alone(skills_browser):
    """Best effort: with this tab's storage blocked the page still answers, shows and settles as it did, and a reload
    forgets the answer; nothing else keeps it."""
    from playwright.sync_api import expect

    browser = skills_browser.chromium.launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": 390, "height": 844})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.add_init_script("Object.defineProperty(window, 'sessionStorage', "
                             "{configurable: true, get() { throw new DOMException('blocked', 'SecurityError'); }});")
        ready, broken = _calendar_health("Port 8080 is busy", companion_reason="load_error")
        beta_stopped = {**ready, "live_reason": "companion_restart_exhausted:beta",
                        "load_error": "companion 'beta' exhausted its restart budget"}
        skills = open_skills(page, extensions=[broken])
        skills.answers[QUEUE] = {"events": [_failed("calendar")], "active": None}
        page.goto(URL, wait_until="networkidle")
        card = page.locator('.skills-card[data-skill="calendar"]')
        chip = card.locator('.skills-card-status')
        expect(card.locator('.skills-load-error')).to_have_text("Port 8080 is busy")
        skills.set_skills([ready])
        refresh(page)
        expect(chip).to_have_text("")
        skills.set_skills([beta_stopped])
        refresh(page)
        expect(chip).to_have_text("Background process stopped")
        expect(card).not_to_contain_text("Port 8080 is busy")
        page.reload(wait_until="networkidle")
        expect(card).to_contain_text("Port 8080 is busy")
        assert errors == []
    finally:
        if skills:
            skills.close(browser)
        else:
            browser.close()


def test_a_pending_install_opens_a_sourced_sheet_by_keyboard_and_invents_nothing(skills_browser):
    """A pending install keeps its keyboard-operable title and a sheet that claims no permissions or keys; an installed
    neighbour keeps its exact row and honest Details."""
    from playwright.sync_api import expect

    browser = skills_browser.chromium.launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": 390, "height": 844})
        # A real folder whose manifest was not read: the host's placeholder permissions are no declaration.
        unread = {**_installed("folder"), "manifest_file": "", "load_error": "manifest parse error: bad", "type": "instruction"}
        skills = open_skills(page, extensions=[_installed("weather"), unread])
        skills.answers[QUEUE] = _queue()  # a pending install of a name no row lists yet
        page.goto(URL, wait_until="networkidle")
        sheet = page.locator(".skills-detail-sheet")
        title = page.locator('.skills-card[data-skill="calendar"][data-lifecycle-virtual] button.skills-card-name')
        expect(title).to_have_count(1)
        # The queue event's `source` is its caller: no origin, no made-here or built-in claim, in the card or the sheet.
        expect(page.locator('.skills-card[data-skill="calendar"] .skills-card-who')).not_to_contain_text("Made here")
        expect(page.locator('.skills-card[data-skill="calendar"]')).not_to_contain_text("Built in")
        title.focus()
        page.keyboard.press("Enter")
        expect(sheet).to_have_count(1)
        expect(sheet).not_to_contain_text("Your keys")
        expect(sheet).not_to_contain_text("Nothing beyond its own folder")
        expect(sheet.locator("h3", has_text="What it can use")).to_have_count(0)
        expect(sheet.locator("h3", has_text="Review")).to_have_count(0)  # a queue job has no installed review
        expect(sheet).not_to_contain_text("Made here")
        page.keyboard.press("Escape")
        expect(sheet).to_have_count(0)
        expect(title).to_be_focused()

        page.locator('.skills-card[data-skill="weather"] .skills-card-name').click()
        expect(sheet).to_contain_text("Nothing beyond its own folder")
        expect(sheet).to_contain_text("Your keys: none")
        sheet.locator("[data-sheet-close]").click()
        expect(sheet).to_have_count(0)

        # The unread folder opens by its exact identity but claims neither empty access nor a verified manifest's rows.
        card = page.locator('.skills-card[data-skill="folder"]')
        expect(card).not_to_contain_text("Can use")
        expect(page.locator('.skills-card[data-skill="weather"]')).to_contain_text("Can use")
        card.locator(".skills-card-name").click()
        expect(sheet).to_contain_text("skills/external/folder")
        expect(sheet).not_to_contain_text("Nothing beyond its own folder")
        expect(sheet).not_to_contain_text("Your keys")
        sheet.locator("[data-sheet-close]").click()
        expect(sheet).to_have_count(0)
        assert not skills.posts, "nothing was enabled or installed"
    finally:
        if skills:
            skills.close(browser)
        else:
            browser.close()


@pytest.mark.parametrize("browser_name", ["chromium", "webkit"])
@pytest.mark.parametrize("kind", ["enable", "install"])
def test_a_pending_install_replaced_by_a_failed_enable_with_no_row_opens_no_details(skills_browser, browser_name, kind):
    """A pending install becomes a failed job (Enable or the install itself) while /api/extensions lists nothing: the
    focused title turns passive but keeps keyboard focus, no sheet opens, and the card keeps its own error and Find in Browse."""
    from playwright.sync_api import expect

    browser = getattr(skills_browser, browser_name).launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": 390, "height": 844})
        queue = _queue()
        skills = open_skills(page, extensions=[])
        skills.answers[QUEUE] = lambda _request: queue
        page.goto(URL, wait_until="networkidle")
        sheet = page.locator(".skills-detail-sheet")
        pending = page.locator('.skills-card[data-skill="calendar"]')
        title = pending.locator("button.skills-card-name")
        expect(title).to_have_count(1)
        title.focus()
        expect(title).to_be_focused()

        # No pointer click: an automatic read replaces the card while the keyboard focus is on the title. The install's
        # own failure is the same job (same id) as the running one; a failed Enable is a different job.
        failed = ({**queue["events"][0], "status": "failed", "error": "Skill “calendar” isn’t installed"} if kind == "install"
                  else _failed("calendar", kind=kind, error="Skill “calendar” isn’t installed"))
        _automatic_read(page, skills, lambda: queue.update({"events": [failed], "active": None}))
        expect(pending).to_have_count(1)
        expect(pending.locator("button.skills-card-name")).to_have_count(0)
        expect(pending.locator("span.skills-card-name")).to_have_text("calendar")
        # Focus follows the replacement, yet the passive text is no Tab stop and does nothing when pressed.
        expect(pending.locator("span.skills-card-name")).to_have_attribute("tabindex", "-1")
        expect(pending.locator("span.skills-card-name")).to_be_focused()
        page.keyboard.press("Enter")
        expect(sheet).to_have_count(0)
        expect(pending.locator(".skills-card-status")).to_have_text(f"{kind.capitalize()} failed")
        expect(pending.locator(".skills-card-who")).not_to_contain_text("Made here")
        assert page.evaluate(HISTORY) == [failed["id"]], "the failed job stays in the queue's history"
        expect(pending).to_contain_text("Skill “calendar” isn’t installed")
        expect(pending.get_by_role("button", name="Find in Browse")).to_have_count(1)
        expect(pending.locator(".skills-details, .skills-toggle, [data-skill-menu-trigger]")).to_have_count(0)
        capture(page, browser_name, 390, f"failed-{kind}-no-row-passive-title")
        pending.locator(".skills-card-name").click()
        expect(sheet).to_have_count(0)
        assert not skills.posts, "nothing was enabled or installed"
    finally:
        if skills:
            skills.close(browser)
        else:
            browser.close()


def _received(page, skills, change):
    """A change the page is told of by the lifecycle hint, as the socket sends it. Returns once the read it starts has
    begun; the test then waits for the DOM that read must bring before it looks at anything else."""
    _automatic_read(page, skills, lambda: (change(), page.evaluate("skillSocketEvent('extension_lifecycle')")))


@pytest.mark.parametrize("browser_name", ["chromium", "webkit"])
def test_an_open_pending_install_preview_follows_its_exact_queue_job(skills_browser, browser_name):
    """The preview opened with Enter from a pending install's title stays while that same job is still pending, even
    as its own words and another skill's job change. It closes on the accepted read that shows another job of the name
    or this one failed; focus then goes to the title the card has (passive text after a failure), never to the body."""
    from playwright.sync_api import expect

    browser = getattr(skills_browser, browser_name).launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": 390, "height": 844})
        queue = _queue(status="queued")
        job = queue["events"][0]
        skills = open_skills(page, extensions=[_installed("weather")])
        skills.answers[QUEUE] = lambda _request: queue
        page.goto(URL, wait_until="networkidle")
        sheet = page.locator(".skills-detail-sheet")
        weather = page.locator('.skills-card[data-skill="weather"]')
        pending = page.locator('.skills-card[data-skill="calendar"]')
        title = pending.locator("button.skills-card-name")

        def preview(words):
            title.focus()
            page.keyboard.press("Enter")
            expect(sheet).to_have_count(1)
            expect(sheet).to_contain_text(words)

        # The same job goes from queued to running with new words while another skill's update fails: this install is
        # what it was. The cards show both changes (the read has been accepted), and the very same preview is still
        # open with the control that held the focus.
        preview("Installing calendar")
        sheet.locator("[data-sheet-close]").focus()
        page.evaluate("() => { window.kept = document.querySelector('.skills-detail-sheet'); }")
        running = {**job, "status": "running", "message": "Installing calendar: copying files"}
        _received(page, skills, lambda: queue.update(
            {"events": [_failed("weather", kind="update", error="Update refused"), running], "active": running}))
        expect(weather.locator(".skills-card-status")).to_have_text("Update failed", timeout=7000)
        expect(pending.locator(".skills-card-purpose")).to_have_text("Installing calendar: copying files", timeout=7000)
        assert page.evaluate("""() => kept.open && kept === document.querySelector('.skills-detail-sheet')
            && document.activeElement === kept.querySelector('[data-sheet-close]')"""), "the same preview, focus where it was"
        page.keyboard.press("Escape")
        expect(sheet).to_have_count(0)
        expect(title).to_be_focused()

        # The first attempt ends and another takes the name: that preview described the old job, so it closes on the
        # read that shows the new one, and the focus is on the title the card still has.
        preview("Installing calendar: copying files")
        cancelled = {**job, "status": "cancelled", "message": ""}
        again = {**job, "id": "calendar-operation-2", "status": "running", "message": "Installing calendar again"}
        _received(page, skills, lambda: queue.update({"events": [cancelled, again], "active": again}))
        expect(pending.locator(".skills-card-purpose")).to_have_text("Installing calendar again", timeout=7000)
        expect(sheet).to_have_count(0)
        expect(title).to_be_focused()

        # That job fails: the title becomes passive text (no Tab stop), and that is where the focus goes.
        preview("Installing calendar again")
        failed = {**again, "status": "failed", "message": "", "error": "Skill “calendar” isn’t installed"}
        _received(page, skills, lambda: queue.update({"events": [cancelled, failed], "active": None}))
        expect(pending.locator(".skills-card-status")).to_have_text("Install failed", timeout=7000)
        expect(sheet).to_have_count(0)
        passive = pending.locator("span.skills-card-name")
        expect(pending.locator("button.skills-card-name")).to_have_count(0)
        expect(passive).to_have_attribute("tabindex", "-1")
        expect(passive).to_be_focused()
        page.keyboard.press("Enter")
        expect(sheet).to_have_count(0)
        assert not skills.posts, "nothing was enabled or installed"
    finally:
        if skills:
            skills.close(browser)
        else:
            browser.close()


@pytest.mark.parametrize("browser_name", ["chromium", "webkit"])
@pytest.mark.parametrize("scenario", ["removed", "namesake", "filtered"])
def test_a_retired_pending_preview_gives_the_focus_to_no_other_card(skills_browser, browser_name, scenario):
    """A read that arrives while a pending install's preview is open takes its job out of the queue, gives its name to
    two folders (the job becomes an attempt beside them), or fails it under a filter that hides its title. The preview
    closes, and no card of the page, hidden or of another identity, takes the focus."""
    from playwright.sync_api import expect

    browser = getattr(skills_browser, browser_name).launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": 390, "height": 844})
        queue = _queue()
        job = queue["events"][0]
        skills = open_skills(page, extensions=[])
        skills.answers[QUEUE] = lambda _request: queue
        page.goto(URL, wait_until="networkidle")
        sheet = page.locator(".skills-detail-sheet")
        cards = page.locator('.skills-card[data-skill="calendar"]')
        if scenario == "filtered":
            # The card is found by its running words only (“Installing calendar”); the failure's words do not match.
            page.locator("#skills-installed-search").fill("installing")
            expect(cards).not_to_have_attribute("hidden", "")
        cards.locator("button.skills-card-name").focus()
        page.keyboard.press("Enter")
        expect(sheet).to_have_count(1)

        first, second = _two_calendar_directories()
        failed = {**job, "status": "failed", "message": "", "error": "Skill “calendar” isn’t installed"}
        change, shown = {
            "removed": (lambda: queue.update({"events": [], "active": None}),
                        lambda: expect(cards).to_have_count(0, timeout=7000)),
            "namesake": (lambda: skills.set_skills([first, second]),
                         lambda: expect(cards).to_have_count(3, timeout=7000)),  # two folders and the attempt beside them
            "filtered": (lambda: queue.update({"events": [failed], "active": None}),
                         lambda: expect(cards).to_have_attribute("hidden", "", timeout=7000)),
        }[scenario]
        _received(page, skills, change)
        shown()
        expect(sheet).to_have_count(0)
        assert page.evaluate("document.activeElement === document.body"), "no other card took the focus"
        assert not skills.posts, "nothing was enabled or installed"
    finally:
        if skills:
            skills.close(browser)
        else:
            browser.close()
