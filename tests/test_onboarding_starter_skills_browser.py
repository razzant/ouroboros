"""The first-run wizard's Skills step through the production wizard (owner 2026-11-21).

Reuses the wizard's static-server/API-boundary fixture: the starter list is the gateway's
`GET /api/onboarding/starter-skills` answer from web/tests/fixtures/skills_v4, and the one atomic
completion carries the owner's ticks and hub trust choices. Nothing installs here.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests._wizard_walk import step_order
from tests.test_subscription_setup_browser import capture, subscription_ui as subscription_ui

pytestmark = [pytest.mark.ui_browser]
STARTER = json.loads((Path(__file__).resolve().parents[1] / "web/tests/fixtures/skills_v4/onboarding_starter_skills.json")
                     .read_text(encoding="utf-8"))
STEP = '[data-starter-skills="step"]'
ROW = ".skills-starter-row"


def _skills_step(ui, *, answer=STARTER, status=200, width=1360, route=True):
    """Review & start, past Consciousness where offered, onto the Skills step."""
    page = ui["page"]
    page.set_viewport_size({"width": width, "height": 900})
    if route:
        page.route("**/api/onboarding/starter-skills", lambda route: route.fulfill(
            status=status, content_type="application/json", body=json.dumps(answer)))
    page.goto(ui["url"] + "/onboarding")
    page.wait_for_selector("#quick-start-btn:not([hidden])")
    assert "skills" in step_order(page), "a fresh install's setup offers the step"
    page.click("#quick-start-btn")
    if "consciousness" in step_order(page):
        page.wait_for_selector("[data-consciousness-step]")
        page.click("#next-btn")
    page.wait_for_selector(STEP)
    return page


def _completion(ui):
    for _ in range(200):
        posted = [payload for path, payload in ui["posts"] if path == "/api/onboarding/complete"]
        if posted:
            return posted
        ui["page"].wait_for_timeout(25)
    return []


def _card(page, item_id):
    return page.locator(".skills-starter-card", has=page.locator(f'[data-starter-item="{item_id}"]'))


@pytest.mark.parametrize("width", [1360, 390])
def test_the_step_lists_the_offer_and_the_completion_carries_the_choices(subscription_ui, width):
    ui = subscription_ui
    page = _skills_step(ui, width=width)
    page.wait_for_selector(".skills-starter-card")
    assert page.locator(".step-title").inner_text() == "Skills"
    assert "Ouroboros starts with the ticked skills; clear any you do not want." in page.locator(".step-header").inner_text()
    assert page.locator(".skills-starter-count").inner_text() == "6 of 10 ticked"
    heads = [text.split("\n")[0].strip() for text in page.locator(".skills-starter-group h3").all_inner_texts()]
    assert [head.rstrip("0123456789").strip() for head in heads] == [
        "Built in — comes with the app", "Messaging", "Documents", "Search & research", "About Ouroboros", "Writing & review"]
    assert page.locator(".skills-starter-card input:checked").count() == 6
    # A ticked card keeps the neutral body (the box says it), the same in both themes.
    page.mouse.move(0, 0)  # no card under the pointer: a hover deepens one
    assert len(set(page.locator(".skills-starter-card").evaluate_all(
        "cards => cards.map((card) => getComputedStyle(card).backgroundColor)"))) == 1
    # Inside a group what works at once leads; what needs a key comes last (here Telegram after Computer control).
    assert page.locator('.skills-starter-group').first.locator('[data-starter-item]').evaluate_all(
        "boxes => boxes.map((box) => box.dataset.starterItem)") == ["builtin/unix_computer_use", "builtin/telegram"]
    telegram = _card(page, "builtin/telegram").inner_text()
    assert "Telegram" in telegram and "Chat with Ouroboros from your phone in Telegram" in telegram
    assert "Needs a Telegram bot token" in telegram and "Built in" not in telegram and "Official" not in telegram
    needs = _card(page, "builtin/telegram").locator(".skills-starter-needs")
    pitch = _card(page, "builtin/telegram").locator(".skills-starter-pitch")
    assert needs.evaluate("el => getComputedStyle(el).color") != pitch.evaluate("el => getComputedStyle(el).color")
    assert page.locator('[data-mark="official"], .skills-chip').count() == 0, "where a skill comes from is said once"
    assert page.get_by_text("Official skills from the Ouroboros hub: its maintainer vouches for each release.").count() == 1
    # Without the catalog's words the card keeps the skill's name and its author's text; a key shows in words.
    assert "unix_computer_use" in _card(page, "builtin/unix_computer_use").inner_text()
    assert "Uses your OpenRouter API Key" in _card(page, "ouroboros/perplexity").inner_text()
    assert _card(page, "ouroboros/keenable").locator(".skills-starter-needs").count() == 0
    trust = page.locator(".skills-starter-trust")
    assert trust.inner_text().startswith(
        "Official skills install without a review; skills from the community are reviewed first.")
    assert trust.locator("[data-starter-trust]").count() == 0, "how they install is one plain line until opened"
    assert "Without review" not in page.locator(STEP).inner_text() and "Review each" not in page.locator(STEP).inner_text()
    capture(page, f"skills-step-{width}")

    page.locator('[data-starter-item="ouroboros/pdf_tools"]').uncheck()
    page.locator('[data-starter-item="ouroboros/perplexity"]').check()
    assert page.locator(".skills-starter-count").inner_text() == "6 of 10 ticked"
    trust.locator("[data-starter-trust-toggle]").click()
    assert trust.locator('[data-starter-trust="ouroboros"][data-trust="trusted"]').get_attribute("aria-pressed") == "true"
    trust.locator('[data-starter-trust="ouroboros-community"][data-trust="trusted"]').click()
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    capture(page, f"skills-step-trust-{width}")
    page.click("#next-btn")  # Review & start's Continue goes to Summary
    row = page.locator(ROW)
    row.wait_for()
    assert row.inner_text().replace("\n", " ") == "Skills 6 will install · Change"
    page.click("#next-btn")
    payload = _completion(ui)[0]
    assert payload["starter_skills"] == ["builtin/telegram", "ouroboros/word_tools", "ouroboros/duckduckgo",
                                         "ouroboros/perplexity", "ouroboros/memory-atlas", "ouroboros/anti-slop"]
    assert payload["skill_source_trust"] == {"ouroboros": "trusted", "ouroboros-community": "trusted"}


def test_summary_change_returns_to_the_step_with_the_ticks_kept(subscription_ui):
    ui = subscription_ui
    page = _skills_step(ui)
    page.locator('[data-starter-all="0"]').click()
    page.click("#next-btn")
    page.locator(ROW).wait_for()
    assert "None will install" in page.locator(ROW).inner_text()
    page.locator("[data-starter-change]").click()
    page.wait_for_selector(".skills-starter-card")
    assert page.locator(".step-title").inner_text() == "Skills"
    assert page.locator(".skills-starter-card input:checked").count() == 0
    page.locator('[data-starter-item="ouroboros/duckduckgo"]').check()
    page.click("#next-btn")
    page.locator(ROW).wait_for()
    assert "1 will install" in page.locator(ROW).inner_text()
    page.click("#next-btn")
    assert _completion(ui)[0]["starter_skills"] == ["ouroboros/duckduckgo"]


def test_untick_all_installs_nothing_and_an_unreadable_list_keeps_the_host_default(subscription_ui):
    ui = subscription_ui
    page = _skills_step(ui)
    page.locator('[data-starter-all="0"]').click()
    assert page.locator(".skills-starter-count").inner_text() == "0 of 10 ticked"
    page.click("#next-btn")
    page.locator(ROW).wait_for()
    page.click("#next-btn")
    assert _completion(ui)[0]["starter_skills"] == []

    ui["posts"].clear()
    page = ui["new_page"]()
    ui["page"] = page
    page = _skills_step(ui, answer={"error": "starter list unavailable"}, status=503)
    page.wait_for_function("document.querySelector('[data-starter-skills=\"step\"]').textContent.includes('couldn’t be read')")
    assert "Ouroboros installs the recommended skills after it starts" in page.locator(STEP).inner_text()
    capture(page, "skills-step-unreadable")
    page.click("#next-btn")
    assert "The recommended set" in page.locator(ROW).inner_text()
    page.click("#next-btn")
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
def test_the_hub_arrives_on_the_step_keeping_ticks_focus_and_place(starter_ui, width, theme):
    """The hub's skills join the list while the person is on it: their ticks, the focused box and the card being read
    stay where they were, and the newcomers join without a second entrance (DESIGN principle 6)."""
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
    page = _skills_step(ui, width=width, route=False)
    step = page.locator(STEP)
    assert 'Reading the Ouroboros hub…' in step.inner_text()
    assert 'default starter list once it can read the hub' in step.inner_text()
    page.locator('[data-starter-item="builtin/telegram"]').uncheck()
    box = page.locator('[data-starter-item="builtin/unix_computer_use"]')
    box.focus()
    page.evaluate("window.scrollTo(0, 60)")
    anchor = page.evaluate("() => document.querySelector('[data-starter-card=\"builtin/unix_computer_use\"]').getBoundingClientRect().top")
    engine = ui['engine']
    capture(page, f'skills-{engine}-{width}-{theme}-reading')
    for _ in range(200):
        if pending:
            break
        page.wait_for_timeout(25)
    assert pending
    pending.pop(0).fulfill(content_type='application/json', body=json.dumps(STARTER))
    page.locator('[data-starter-item="ouroboros/duckduckgo"]').wait_for()
    assert not page.locator('[data-starter-item="builtin/telegram"]').is_checked()
    assert page.locator('[data-starter-item="ouroboros/duckduckgo"]').is_checked()
    assert 'Reading the Ouroboros hub' not in step.inner_text()
    assert page.evaluate("document.activeElement?.dataset.starterItem") == 'builtin/unix_computer_use', 'focus stays'
    top = page.evaluate("() => document.querySelector('[data-starter-card=\"builtin/unix_computer_use\"]').getBoundingClientRect().top")
    assert abs(top - anchor) <= 1, (anchor, top)
    assert page.locator('.skills-starter-list[data-arrive]').count() == 0, 'the hub brings no second entrance'
    assert page.get_by_text('Official skills from the Ouroboros hub: its maintainer vouches for each release.').count() == 1
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    capture(page, f'skills-{engine}-{width}-{theme}-ready')
    page.click('#next-btn')
    page.locator(ROW).wait_for()
    page.click('#next-btn')
    assert _completion(ui)[0]['starter_skills'] == ['ouroboros/pdf_tools', 'ouroboros/word_tools',
                                                   'ouroboros/duckduckgo', 'ouroboros/memory-atlas', 'ouroboros/anti-slop']


@pytest.mark.parametrize('changed', [False, True])
def test_finish_while_hub_is_reading_uses_defaults_only_if_ticks_untouched(starter_ui, changed):
    ui = starter_ui
    page = _skills_step(ui, answer=_fresh_offer(), width=390)
    if changed:
        page.locator('[data-starter-all="0"]').click()
    else:
        assert 'default starter list once it can read the hub' in page.locator(STEP).inner_text()
    page.click('#next-btn')
    page.locator(ROW).wait_for()
    if not changed:
        assert 'default starter list once it can read the hub' in page.locator(ROW).inner_text()
    page.click('#next-btn')
    payload = _completion(ui)[0]
    if changed:
        assert payload['starter_skills'] == []
    else:
        assert 'starter_skills' not in payload
    assert payload['skill_source_trust'] == {'ouroboros': 'trusted', 'ouroboros-community': 'review'}


def test_the_step_still_polls_after_five_minutes_in_earlier_setup(starter_ui):
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
    if 'consciousness' in step_order(page):
        page.wait_for_selector('[data-consciousness-step]')
        page.click('#next-btn')
    step = page.locator(STEP)
    step.wait_for()
    assert 'Reading the Ouroboros hub' in step.inner_text()
    page.wait_for_function("document.querySelector('.skills-starter-count')?.textContent === '6 of 10 ticked'")
    assert len(reads) == 2
    assert 'Reading the Ouroboros hub' not in step.inner_text()
    assert page.locator('[data-starter-item="ouroboros/duckduckgo"]').is_checked()
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    capture(page, f"skills-{ui['engine']}-390-r2-delayed-step")


ARRIVAL = """() => [...document.querySelectorAll('.skills-starter-card')].slice(0, 14).map((el) => {
    const style = getComputedStyle(el);
    return {name: style.animationName, delay: parseFloat(style.animationDelay) * (style.animationDelay.endsWith('ms') ? 1 : 1000),
            duration: parseFloat(style.animationDuration) * (style.animationDuration.endsWith('ms') ? 1 : 1000)};
})"""


@pytest.mark.parametrize('motion', ['no-preference', 'reduce'])
def test_the_cards_arrive_once_per_entry_and_not_under_reduced_motion(starter_ui, motion):
    """Entering the step plays one short staggered entrance (each card a surface that comes on the sheet curve, the
    whole under a second); its boxes answer at once; a repaint never plays it again, leaving and coming back does.
    Under reduced motion the list stands as it ends."""
    ui = starter_ui
    ui['page'].emulate_media(reduced_motion=motion)
    page = _skills_step(ui)
    page.wait_for_selector('.skills-starter-card')
    page.locator('[data-starter-item="ouroboros/keenable"]').check()  # during the entrance: usable at once
    assert page.locator('[data-starter-item="ouroboros/keenable"]').is_checked()
    arrival = page.evaluate(ARRIVAL)
    if motion == 'reduce':
        assert {card['name'] for card in arrival} == {'none'}, arrival
    else:
        assert {card['name'] for card in arrival} == {'skills-starter-arrive'}, arrival
        delays = [card['delay'] for card in arrival]
        assert delays == sorted(delays) and delays[0] == 0 and delays[1] > 0, delays
        assert max(card['delay'] + card['duration'] for card in arrival) < 1000, arrival
    page.locator('[data-starter-all="1"]').click()  # a repaint: the entrance is not played again
    assert page.locator('.skills-starter-list[data-arrive]').count() == 0
    assert {card['name'] for card in page.evaluate(ARRIVAL)} == {'none'}
    page.click('#next-btn')
    page.locator(ROW).wait_for()
    page.locator('[data-starter-change]').click()  # entering again plays it again (unless motion is reduced)
    page.wait_for_selector('.skills-starter-card')
    assert page.locator('.skills-starter-list[data-arrive]').count() == 1
    assert {card['name'] for card in page.evaluate(ARRIVAL)} == ({'none'} if motion == 'reduce' else {'skills-starter-arrive'})
