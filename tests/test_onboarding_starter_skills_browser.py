"""The first-run wizard's Starter skills row through the production wizard.

Reuses the wizard's static-server/API-boundary fixture: the starter list is the gateway's
`GET /api/onboarding/starter-skills` answer from web/tests/fixtures/skills_v4, and the one atomic
completion carries the owner's ticks and hub trust choices. Nothing installs here.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_subscription_setup_browser import capture, subscription_ui as subscription_ui

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
STARTER = json.loads((Path(__file__).resolve().parents[1] / "web/tests/fixtures/skills_v4/onboarding_starter_skills.json")
                     .read_text(encoding="utf-8"))


def _summary(ui, *, answer=STARTER, status=200, width=1360):
    page = ui["page"]
    page.set_viewport_size({"width": width, "height": 900})
    page.route("**/api/onboarding/starter-skills", lambda route: route.fulfill(
        status=status, content_type="application/json", body=json.dumps(answer)))
    page.goto(ui["url"] + "/onboarding")
    page.wait_for_selector("#quick-start-btn:not([hidden])")
    page.click("#quick-start-btn")
    page.wait_for_selector(".summary-card")
    return page


def _completion(ui):
    return [payload for path, payload in ui["posts"] if path == "/api/onboarding/complete"]


@pytest.mark.parametrize("width", [1360, 390])
def test_starter_row_opens_into_the_categorized_list_and_the_completion_carries_the_choices(subscription_ui, width):
    ui = subscription_ui
    page = _summary(ui, width=width)
    row = page.locator(".skills-starter-row")
    row.wait_for()
    page.wait_for_function("document.querySelector('.skills-starter-row').textContent.includes('ticked')")
    assert "Starter skills" in row.inner_text()
    assert "10 · 6 ticked · Change ›" in row.inner_text()
    assert ("Ticked: telegram, pdf_tools, word_tools, duckduckgo, memory-atlas, anti-slop. 4 more you can tick."
            in row.inner_text())
    capture(page, f"starter-row-collapsed-{width}")

    page.locator("[data-starter-toggle]").click()
    panel = page.locator("[data-starter-panel]")
    panel.wait_for()
    heads = [text.split("\n")[0].strip() for text in panel.locator(".skills-starter-group h4").all_inner_texts()]
    assert heads[0].startswith("Built in — comes with the app"), heads
    assert [head.rstrip("0123456789").strip() for head in heads] == [
        "Built in — comes with the app", "Messaging", "Documents", "Search & research", "About Ouroboros", "Writing & review"]
    assert panel.locator(".skills-starter-card.active").count() == 6
    assert "They install without review — you trust Ouroboros. Skills that need a key wait for it." in panel.inner_text()
    assert "Needs: OPENROUTER_API_KEY" in panel.inner_text()
    scroll = panel.locator("[data-starter-scroll]")
    assert scroll.evaluate("el => getComputedStyle(el).overflowY") == "auto"
    trust = panel.locator(".skills-starter-trust")
    assert trust.locator('[data-starter-trust="ouroboros"][data-trust="trusted"]').get_attribute("aria-pressed") == "true"
    assert trust.locator('[data-starter-trust="ouroboros-community"][data-trust="review"]').get_attribute("aria-pressed") == "true"
    capture(page, f"starter-row-open-{width}")

    page.locator('[data-starter-item="ouroboros/pdf_tools"]').uncheck()
    page.locator('[data-starter-item="ouroboros/perplexity"]').check()
    assert "10 · 6 ticked" in row.inner_text()
    trust.locator('[data-starter-trust="ouroboros-community"][data-trust="trusted"]').click()
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    page.click("#next-btn")
    for _ in range(200):
        if _completion(ui):
            break
        page.wait_for_timeout(25)
    payload = _completion(ui)[0]
    assert payload["starter_skills"] == ["builtin/telegram", "ouroboros/word_tools", "ouroboros/duckduckgo",
                                         "ouroboros/perplexity", "ouroboros/memory-atlas", "ouroboros/anti-slop"]
    assert payload["skill_source_trust"] == {"ouroboros": "trusted", "ouroboros-community": "trusted"}


def test_untick_all_installs_nothing_and_an_unreadable_list_keeps_the_host_default(subscription_ui):
    ui = subscription_ui
    page = _summary(ui)
    page.locator("[data-starter-toggle]").click()
    page.locator('[data-starter-all="0"]').click()
    page.locator("[data-starter-toggle]").click()
    assert "Nothing is ticked: no skills install now." in page.locator(".skills-starter-row").inner_text()
    page.click("#next-btn")
    for _ in range(200):
        if _completion(ui):
            break
        page.wait_for_timeout(25)
    assert _completion(ui)[0]["starter_skills"] == []

    ui["posts"].clear()
    page = ui["new_page"]()
    ui["page"] = page
    page = _summary(ui, answer={"error": "starter list unavailable"}, status=503)
    page.wait_for_function("document.querySelector('.skills-starter-row').textContent.includes('The default set')")
    assert "Ouroboros installs the recommended skills after it starts" in page.locator(".skills-starter-row").inner_text()
    capture(page, "starter-row-unreadable")
    page.click("#next-btn")
    for _ in range(200):
        if _completion(ui):
            break
        page.wait_for_timeout(25)
    payload = _completion(ui)[0]
    assert "starter_skills" not in payload and "skill_source_trust" not in payload, "the host's default list stays"


def _fresh_offer():
    import copy
    answer = copy.deepcopy(STARTER)
    answer['groups'] = [{**group, 'items': [item for item in group['items'] if item.get('builtin')]}
                        for group in answer['groups']]
    answer['groups'] = [group for group in answer['groups'] if group['items']]
    for source in answer['sources']:
        if source['kind'] == 'hub':
            source['state'] = 'reading'
    return answer


@pytest.fixture(params=['chromium', 'webkit'])
def starter_ui(request, monkeypatch):
    # Select inside the isolated test process: the safe launcher strips this
    # optional harness setting from the caller's environment.
    monkeypatch.setenv('OUROBOROS_UI_BROWSER_ENGINE', request.param)
    ui = request.getfixturevalue('subscription_ui')
    ui['engine'] = request.param
    return ui


@pytest.mark.parametrize('width,theme', [(width, theme) for width in (1280, 390) for theme in ('light', 'dark')])
def test_fresh_root_offer_arrives_after_summary_preserving_owner_ticks(starter_ui, width, theme):
    ui = starter_ui
    page = ui['page']
    page.set_viewport_size({'width': width, 'height': 900})
    page.add_init_script(f"localStorage.setItem('ouroboros.theme', '{theme}')")
    pending = []
    reads = []

    def offer(route):
        reads.append(route.request.url)
        if len(reads) == 1:
            route.fulfill(content_type='application/json', body=json.dumps(_fresh_offer()))
        else:
            pending.append(route)

    page.route('**/api/onboarding/starter-skills', offer)
    page.goto(ui['url'] + '/onboarding')
    page.locator('#quick-start-btn:not([hidden])').click()
    row = page.locator('.skills-starter-row')
    row.wait_for()
    assert 'Reading the Ouroboros hub…' in row.inner_text()
    assert 'default starter list once it can read the hub' in row.inner_text()
    page.locator('[data-starter-toggle]').click()
    page.locator('[data-starter-item="builtin/telegram"]').uncheck()
    page.locator('[data-starter-panel]').scroll_into_view_if_needed()
    engine = ui['engine']
    capture(page, f'starter-{engine}-{width}-{theme}-reading')
    for _ in range(200):
        if pending:
            break
        page.wait_for_timeout(25)
    assert pending
    pending.pop(0).fulfill(content_type='application/json', body=json.dumps(STARTER))
    page.locator('[data-starter-item="ouroboros/duckduckgo"]').wait_for()
    assert not page.locator('[data-starter-item="builtin/telegram"]').is_checked()
    assert page.locator('[data-starter-item="ouroboros/duckduckgo"]').is_checked()
    assert 'Reading the Ouroboros hub' not in row.inner_text()
    builtin = page.locator('.skills-starter-card', has=page.locator('[data-starter-item="builtin/telegram"]'))
    assert builtin.locator('[data-mark="official"]').count() == 1
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    assert page.locator('[data-starter-panel]').evaluate('el => el.scrollWidth <= el.clientWidth + 1')
    capture(page, f'starter-{engine}-{width}-{theme}-ready')
    page.locator('#next-btn').click()
    for _ in range(200):
        if _completion(ui):
            break
        page.wait_for_timeout(25)
    assert _completion(ui)[0]['starter_skills'] == ['ouroboros/pdf_tools', 'ouroboros/word_tools',
                                                   'ouroboros/duckduckgo', 'ouroboros/memory-atlas', 'ouroboros/anti-slop']


@pytest.mark.parametrize('changed', [False, True])
def test_finish_while_hub_is_reading_uses_defaults_only_if_ticks_untouched(starter_ui, changed):
    ui = starter_ui
    page = _summary(ui, answer=_fresh_offer(), width=390)
    if changed:
        page.locator('[data-starter-toggle]').click()
        page.locator('[data-starter-all="0"]').click()
    else:
        assert 'default starter list once it can read the hub' in page.locator('.skills-starter-row').inner_text()
    page.locator('#next-btn').click()
    for _ in range(200):
        if _completion(ui):
            break
        page.wait_for_timeout(25)
    payload = _completion(ui)[0]
    if changed:
        assert payload['starter_skills'] == []
    else:
        assert 'starter_skills' not in payload
    assert payload['skill_source_trust'] == {'ouroboros': 'trusted', 'ouroboros-community': 'review'}


def test_summary_still_polls_after_five_minutes_in_earlier_setup(starter_ui):
    ui = starter_ui
    page = ui['page']
    page.set_default_timeout(60_000)
    page.set_viewport_size({'width': 390, 'height': 900})
    page.add_init_script("""
        const actualNow = Date.now;
        window.setupElapsed = 0;
        Date.now = () => actualNow() + window.setupElapsed;
    """)
    reads = []

    def offer(route):
        reads.append(route.request.url)
        route.fulfill(content_type='application/json', body=json.dumps(_fresh_offer() if len(reads) == 1 else STARTER))

    page.route('**/api/onboarding/starter-skills', offer)
    page.goto(ui['url'] + '/onboarding', wait_until='networkidle', timeout=120_000)
    page.locator('#quick-start-btn:not([hidden])').wait_for()
    assert len(reads) == 1
    page.evaluate('window.setupElapsed = 300_000')
    page.locator('#quick-start-btn').click()
    row = page.locator('.skills-starter-row')
    row.wait_for()
    assert 'Reading the Ouroboros hub' in row.inner_text()
    page.wait_for_function("document.querySelector('.skills-starter-row').textContent.includes('10 · 6 ticked')")
    assert len(reads) == 2
    assert 'Reading the Ouroboros hub' not in row.inner_text()
    page.locator('[data-starter-toggle]').click()
    assert page.locator('[data-starter-item="ouroboros/duckduckgo"]').is_checked()
    page.locator('[data-starter-panel]').scroll_into_view_if_needed()
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    capture(page, f"starter-{ui['engine']}-390-r2-delayed-summary")
