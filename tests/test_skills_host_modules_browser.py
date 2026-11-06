"""Real Skills consumers: closing an announcement preserves its facts and recovery.

HTTP is routed by the harness; Panic here never reaches the running application.
"""
from __future__ import annotations

import pytest

from tests._skills_page_harness import URL, capture, no_overflow, open_skills, skills_browser as _skills_browser

pytestmark = pytest.mark.ui_browser
skills_browser = _skills_browser

DEMO_SKILL = {
    "name": "demo", "source": "external", "description": "Demo page module", "version": "1.0",
    "type": "extension", "enabled": True, "review_status": "clean", "review_stale": False,
    "review_gate": {"executable_review": True}, "content_hash": "a" * 64,
    "payload_root": "skills/external/demo", "grants": {"all_granted": True}, "permissions": [],
    "removal": "delete",
}
CONTRIBUTIONS = {
    "contributions": [{
        "key": "demo:host", "skill": "demo", "skill_label": "Demo", "label": "Demo module",
        "revision": "r1", "runtime": "client", "provides": "conversation",
        "render": {"kind": "host", "url": "/static/tests/fixtures/skills_v4/host_module_demo.js"},
    }],
    "offers": {}, "preferences": {}, "voice": {}, "icons": [], "problems": {},
}
CATALOG = {"items": [{"id": "fixture/demo", "name": "demo", "source_id": "fixture", "source_name": "Fixture",
                      "description": "Demo page module", "version": "1.0", "installed": True}],
           "categories": [], "sources": [], "starter": {}}


@pytest.mark.parametrize("browser_name", ["chromium", "webkit"])
@pytest.mark.parametrize("theme,width", [("dark", 1180), ("light", 390), ("light", 320), ("dark", 320)])
@pytest.mark.parametrize("mode", ["host_fail", "hold", "panic_hold"])
def test_skills_is_real_owner_surface_for_page_module_fault_and_hold(skills_browser, browser_name, theme, width, mode):
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": width, "height": 844}, has_touch=width == 390)
        page.add_init_script(f"document.addEventListener('DOMContentLoaded', () => document.documentElement.dataset.theme = '{theme}');")
        query = "?host_modules=1&" + ("host_fail=1" if mode == "host_fail" else "hold=all")
        skills = open_skills(page, extensions=[DEMO_SKILL], catalog=CATALOG)
        skills.answers["/api/ui/contributions"] = CONTRIBUTIONS
        page.goto(URL + query, wait_until="networkidle")
        card = page.locator('.skills-card[data-skill="demo"]')
        state = "Off in this window" if mode == "host_fail" else "Held in this window"
        page.wait_for_function("state => document.querySelector('.skills-card[data-skill=\"demo\"]').textContent.includes(state)", arg=state)
        browse = page.locator('.marketplace-card[data-item="fixture/demo"]')
        page.wait_for_function("getComputedStyle(document.querySelector('.toast')).opacity === '1'")
        if mode != "host_fail":
            assert page.locator('.toast-action').evaluate("el => { const range = document.createRange(); range.selectNodeContents(el); return range.getClientRects().length === 1; }"), "the recovery label must not be squeezed across lines"
        capture(page, browser_name, width, f"{theme}-{mode}-notice")

        toast = page.locator('.toast').first
        close = toast.locator('.toast-dismiss')
        close.wait_for()
        assert close.get_attribute("aria-label") == "Dismiss notice"
        close.focus()
        page.keyboard.press("Enter")
        toast.wait_for(state="detached")
        assert state in card.inner_text(), "dismissal hides the announcement, not the fact"
        needs = page.locator('[data-installed-filter="needs"]')
        assert needs.locator('.skills-category-count').inner_text() == "1"
        needs.click()
        assert card.is_visible(), "the fault/hold is included in the actual filter"
        if mode == "panic_hold":
            # Exercise the actual fact event into the mounted controller, not a prebuilt renderer model.
            page.evaluate("window.dispatchEvent(new CustomEvent('ouro:panic'))")
            page.wait_for_function("!document.querySelector('[data-skill-action=\"start_host_module\"]')")
            assert "Reload this page to restore it" in card.inner_text()
            assert page.locator('.toast-action').count() == 0
            assert page.evaluate("hostRuntime.pageModuleFacts().held.length") == 1
            assert page.evaluate("hostRuntime.startHeldModule('demo:host')") is False
            # The real Panic producer without a transport must admit uncertainty, stay visible and close by keyboard.
            page.evaluate("async () => { const {panicWithoutSocket} = await import('/static/modules/panic.js'); await panicWithoutSocket({}, null); }")
            panic_notice = page.locator('.toast:has-text("Panic was not confirmed")')
            panic_notice.wait_for()
            page.wait_for_function("getComputedStyle(document.querySelector('.toast')).opacity === '1'")
            capture(page, browser_name, width, f"{theme}-{mode}-unconfirmed")
            panic_notice.locator('.toast-dismiss').focus()
            page.keyboard.press("Enter")
            panic_notice.wait_for(state="detached")
            assert "Reload this page to restore it" in card.inner_text()
            page.get_by_role("tab", name="Browse", exact=True).click()
            browse.wait_for()
            assert "needs you" in browse.inner_text()
            page.get_by_role("tab", name="Installed", exact=True).click()
        else:
            # Close the overlay before using the tab it covers on a phone.
            page.get_by_role("tab", name="Browse", exact=True).click()
            browse.wait_for()
            assert "needs you" in browse.inner_text()
            page.get_by_role("tab", name="Installed", exact=True).click()
            if mode == "host_fail":
                assert "Its page module stopped in this window" in card.inner_text()
                assert card.locator('[data-skill-action="start_host_module"]').count() == 0
            else:
                start = card.locator('[data-skill-action="start_host_module"]')
                start.click()
                page.wait_for_function("hostRuntime.mounted().includes('demo:host')")
                page.wait_for_function("!document.querySelector('[data-skill-action=\"start_host_module\"]')")
                page.wait_for_function("document.querySelectorAll('#toast-stack .toast').length === 0")
                # The Needs you filter removes the recovered card until the owner selects All.
                assert not card.is_visible()
                page.locator('[data-installed-filter="all"]').click()
                assert card.is_visible()
                assert "Held in this window" not in card.inner_text()
                assert card.locator('.skills-toggle').is_checked()
                page.get_by_role("tab", name="Browse", exact=True).click()
                assert "On" in browse.inner_text()
                assert "needs you" not in browse.inner_text()
                page.get_by_role("tab", name="Installed", exact=True).click()

        assert no_overflow(page)
        assert not skills.posts, "local dismiss/recovery and offline Panic send no backend mutation"
        capture(page, browser_name, width, f"{theme}-{mode}-result")
    finally:
        if skills:
            skills.close(browser)
        else:
            browser.close()
