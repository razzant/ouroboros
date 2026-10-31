"""Structural lifecycle transitions through the real Skills controller and its own poll.

HTTP replies are controlled facts, not a live installation. No Refresh or page reload
repairs the view; normal same-kind updates must retain the owner's working controls.
"""
from __future__ import annotations

import pytest

from tests._skills_page_harness import (
    URL, capture, no_overflow, open_skills, skills_browser as _skills_browser,
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
                        "kind": kind, "status": status, "message": "Installing calendar"}],
            "active": None}


def _automatic_read(page, skills, change):
    request = ("GET", "/api/extensions")
    count = skills.requests.count(request)
    change()
    skills.wait_for_count(request, count + 1, timeout_ms=7000)
    # A matching reply, not just elapsed time or a request, must reach the consumer.
    page.wait_for_function("!document.querySelector('#skills-refresh').disabled")


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
        card.locator('label.skills-switch').click()
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
