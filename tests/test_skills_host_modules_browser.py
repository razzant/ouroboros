"""Real Skills consumer checks for page-module faults, holds and recovery."""
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


@pytest.mark.parametrize("browser_name", ["chromium", "webkit"])
@pytest.mark.parametrize("mode", ["host_fail", "hold"])
def test_skills_is_real_owner_surface_for_page_module_fault_and_hold(skills_browser, browser_name, mode):
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": 390, "height": 844})
        query = "?host_modules=1"
        if mode == "host_fail":
            query += "&host_fail=1"
        else:
            query += "&hold=all"
        skills = open_skills(page, extensions=[DEMO_SKILL])
        skills.answers["/api/ui/contributions"] = CONTRIBUTIONS
        page.goto(URL + query, wait_until="networkidle")
        card = page.locator('.skills-card[data-skill="demo"]')
        card.wait_for(state="attached")

        if mode == "host_fail":
            page.wait_for_function("document.querySelector('.skills-card[data-skill=\"demo\"]').textContent.includes('Off in this window')")
            assert "Its page module stopped in this window" in card.inner_text()
            toast = page.locator('.toast:has-text("Demo module")')
            toast.locator('.toast-dismiss').wait_for()
            toast.locator('.toast-dismiss').click()
            assert "Its page module stopped in this window" in card.inner_text()
            assert card.locator('.toast-dismiss').count() == 0
        else:
            page.wait_for_function("document.querySelector('.skills-card[data-skill=\"demo\"]').textContent.includes('Held in this window')")
            assert "Held back:" in card.inner_text()
            card.locator('[data-skill-action="start_host_module"]').wait_for()
            toast = page.locator('.toast:has-text("Demo did not start")')
            toast.locator('.toast-dismiss').wait_for()
            toast.locator('.toast-dismiss').click()
            assert "Held back:" in card.inner_text()
            start = card.locator('[data-skill-action="start_host_module"]')
            start.click()
            page.wait_for_function('!document.querySelector(\'[data-skill-action="start_host_module"]\')')
            assert "Held in this window" not in card.inner_text()

        assert no_overflow(page)
        capture(page, browser_name, 390, f"host-module-{mode}")
    finally:
        if skills:
            skills.close(browser)
        else:
            browser.close()
