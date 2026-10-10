"""The open Available-subagents card in a REAL render (docs/DESIGN.md §6): the
Reviewer box keeps its place and node under pointer and keyboard, drafts and open
disclosures survive repaints, and history waits behind Details & history while
current exceptions stay visible. Settings and the first-run wizard consume the one
editor; controlled API responses, no runtime, no model calls."""
from __future__ import annotations

import json

import pytest
from tests import test_subscription_role_routes_browser as roles

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
subscription_ui = roles.subscription_ui
role_ui = roles.role_ui

# The owner's own words, kept byte for byte through repaint and save.
DESCRIPTION = ("openrouter grok-4.7 — быстрая, креативная, дешёвая модель. Классно подходит для "
               "брейнштурма, набрасывания идей, прожарки. Но иногда галлюцинирует.\n"
               "Поэтому часто смотри через её призму.")
LONG_MODEL = "gpt-test-with-a-deliberately-long-model-identifier-1m-context-preview"
HISTORY_WORDS = ("Last review", "Last task run", "record rev_", "Earlier settings", "Stored as",
                 "uses a session seat", "can reach outside the working folder", "In the review pool")
VIEWPORTS = {"laptop": (1095, 858), "phone": (390, 844), "wide": (1600, 1000)}


def _roster(ui):
    roles.configure_mixed(ui)
    roles.api_lane(ui, "openai")
    rows = [
        {"subagent_id": "grok", "recommended_use": DESCRIPTION,
         "route": {"kind": "agent_session", "target_id": "cursor=grok-4.7-xhigh-fast"}},
        {"subagent_id": "pinned", "recommended_use": "Reviews with a pinned account.", "review_eligible": True,
         "route": {"kind": "agent_session", "target_id": "codex=gpt-test", "credential_profile_id": "ghost"}},
        {"subagent_id": "packet", "recommended_use": "Direct API reviewer.", "review_eligible": True,
         "delivery": "packet", "route": {"kind": "api_model", "target_id": "openai::gpt-api"}},
        {"subagent_id": "parked", "recommended_use": "Marked, then switched off.", "review_eligible": True,
         "enabled": False, "effort": "low", "route": {"kind": "agent_session", "target_id": "codex=gpt-test"}},
        {"subagent_id": "long", "recommended_use": "Subscription model with a long name.",
         "route": {"kind": "api_model", "target_id": f"claudexor::opaque-source={LONG_MODEL}",
                   "credential_profile_id": "personal"}},
    ]
    ui["settings"].update(OUROBOROS_SUBAGENTS={"enabled": True, "items": rows})
    # The owner's case: the Cursor engine is not listed now, which the head word already says.
    status = ui["fixture"]["status"]
    status["harnesses"] = [harness for harness in status["harnesses"] if harness["id"] != "cursor"]
    ui["backend"]["review_pool"] = {
        "pool": [{"subagent_id": "pinned", "last_execution": {
            "effective": {"route": "agent_session:codex", "model": "gpt-test", "credential_profile_id": "personal"},
            "review_record_id": "rev_42"}}],
        "excluded": [], "last_executions": {}, "config_error": "", "migration": None, "row_costs": {},
    }
    # A failed task run under other settings: history in Details, never a current status.
    ui["fixture"]["status"]["subagent_last_delegation"] = {"latest_by_subagent": {"grok": {
        "selected_subagent_id": "grok", "route": "cursor", "requested_model": "grok-4.7",
        "applied_model": "Grok 4.7 256K", "outcome": "failed", "failure_code": "quota_exhausted",
        "ts": "2026-10-09T11:17:48.609Z",
        "identity": {"kind": "agent_session", "target_id": "cursor=grok-4.7", "credential_profile_id": "",
                     "access": "full", "effort": "", "processing_preference": ""}}}}
    return rows


def _open(ui, size, scheme="light"):
    page = ui["page"]
    page.set_viewport_size({"width": size[0], "height": size[1]})
    page.emulate_media(color_scheme=scheme)
    page = roles.open_agents(ui)
    page.wait_for_function("() => document.querySelector('[data-subagent-last-review]:not([hidden])')")
    page.evaluate("""async () => {
        await document.fonts.ready;
        for (let i = 0; i < 8; i++) await new Promise(requestAnimationFrame);
    }""")
    return page


def _box(locator):
    box = locator.bounding_box()
    return round(box["x"], 1), round(box["y"], 1)


def _visible_card_text(card):
    """What the card shows with its disclosures closed: no history, no standing captions."""
    return card.evaluate("""card => [...card.querySelectorAll('*')]
        .filter(n => n.childNodes.length && [...n.childNodes].some(c => c.nodeType === 3 && c.textContent.trim())
            && n.checkVisibility() && !n.closest('details:not([open]) > :not(summary)'))
        .map(n => [...n.childNodes].filter(c => c.nodeType === 3).map(c => c.textContent.trim()).join(' '))
        .join(' | ')""")


@pytest.mark.parametrize("viewport,scheme", [("laptop", "light"), ("laptop", "dark"), ("phone", "light"), ("wide", "dark")])
def test_reviewer_keeps_its_place_and_node_under_pointer_and_keyboard(role_ui, viewport, scheme):
    ui = role_ui
    _roster(ui)
    page = _open(ui, VIEWPORTS[viewport], scheme)
    card = page.locator("[data-subagent-row]").first
    box = card.locator('[data-subagent-field="review_eligible"]')
    text = card.locator('[data-subagent-field="recommended_use"]')
    box.evaluate("e => { window.reviewerNode = e; }")
    text.evaluate("e => { window.descriptionNode = e; e.setSelectionRange(5, 9); }")
    intent = page.locator("[data-subagents-intent]")
    assert intent.inner_text() == "Saved"
    card.evaluate("e => e.scrollIntoView({block: 'start'})")
    roles.capture(page, f"quiet-{viewport}-{scheme}-before")
    start, card_top = _box(box), _box(card)

    # Pointer path: WebKit need not focus a clicked checkbox, so geometry is the claim here.
    box.click()
    assert box.is_checked() and _box(box) == start and _box(card) == card_top
    assert intent.inner_text() == ""
    dirty = page.locator("#settings-unsaved-indicator")
    assert dirty.is_visible() and dirty.get_attribute("aria-hidden") != "true"
    assert "Unsaved changes" in dirty.aria_snapshot(), "the one dirty indication remains accessible"
    roles.capture(page, f"quiet-{viewport}-{scheme}-marked")
    box.click()
    assert not box.is_checked() and _box(box) == start and _box(card) == card_top

    # Keyboard path: focus stays on the very same node through on, off and back on.
    box.focus()
    for expected in (True, False, True):
        page.keyboard.press("Space")
        assert box.is_checked() is expected
        assert page.evaluate("document.activeElement === window.reviewerNode")
        assert _box(box) == start and _box(card) == card_top
    assert box.evaluate("e => e === window.reviewerNode"), "the mark never rebuilds the card"
    assert text.evaluate("e => e === window.descriptionNode && e.value") == DESCRIPTION
    assert text.evaluate("e => [e.selectionStart, e.selectionEnd]") == [5, 9]
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")

    with page.expect_response("**/api/settings"):
        page.locator("#btn-save-settings").click()
    saved = roles.saved_catalog(ui)["items"]
    assert saved[0]["review_eligible"] is True and saved[0]["recommended_use"] == DESCRIPTION
    assert [row.get("review_eligible") for row in saved[1:]] == [True, True, True, None]
    assert saved[1]["route"]["credential_profile_id"] == "ghost" and saved[3]["enabled"] is False
    assert saved[3]["effort"] == "low" and saved[4]["route"]["target_id"].endswith(LONG_MODEL)


def test_current_exceptions_stay_visible_and_history_waits_in_details(role_ui):
    ui = role_ui
    _roster(ui)
    page = _open(ui, VIEWPORTS["laptop"])
    cards = page.locator("[data-subagent-row]")
    grok, pinned, packet, parked, long = (cards.nth(i) for i in range(5))
    for card in (grok, pinned, packet, parked, long):
        visible = _visible_card_text(card)
        assert not any(word in visible for word in HISTORY_WORDS), visible
        assert "Saved ·" not in visible and "Draft ·" not in visible
    # A specific refusal is said; a bare repeat of the head word is not.
    assert pinned.locator("[data-subagent-status-reason]").inner_text() == (
        "codex · pinned account ghost currently unavailable")
    assert grok.locator("[data-subagent-status]").inner_text() == "Unavailable"
    assert grok.locator("[data-subagent-status-reason]").is_hidden()
    assert parked.locator("[data-subagent-review-exception]").inner_text() == (
        "Switched off, so not in the review pool.")
    # Every open field stays open and labelled; the marked API row shows delivery and its price.
    labels = grok.locator(".available-subagent-route > .available-subagent-field").evaluate_all(
        "els => els.map(e => e.firstChild.textContent.trim())")
    assert labels == ["Source", "Model", "Account", "Reasoning effort", "Access"]
    assert packet.locator('[data-subagent-field="delivery"]').input_value() == "packet"
    assert packet.locator("[data-subagent-delivery-cost]").inner_text() == "cost unknown"
    assert pinned.locator('[data-subagent-field="effort"] option[value=""]').inner_text() == "Default (reviews at high)"
    assert long.locator('[data-subagent-field="model"]').input_value() == LONG_MODEL
    roles.capture(page, "quiet-exceptions-closed")

    # Details: the two histories are different events, each named.
    grok.locator("[data-subagent-details] > summary").click()
    pinned.locator("[data-subagent-details] > summary").click()
    task = grok.locator("[data-subagent-last-task] dd").inner_text()
    assert task.startswith("Earlier settings · cursor session · Grok 4.7 256K") and "failed (quota_exhausted)" in task
    assert grok.locator("[data-subagent-last-review]").is_hidden()
    assert pinned.locator("[data-subagent-last-review] dd").inner_text() == (
        "codex session · gpt-test · account personal · record rev_42")
    assert grok.locator("[id$='-access-help']").inner_text().startswith("Full system access (the default)")
    assert packet.locator("[data-subagent-stored] dd").text_content() == "openai::gpt-api"
    grok.scroll_into_view_if_needed()
    roles.capture(page, "quiet-details-open")


def test_open_disclosures_drafts_and_focus_survive_repaint_refresh_and_reload(role_ui):
    ui = role_ui
    _roster(ui)
    page = _open(ui, VIEWPORTS["laptop"])
    cards = page.locator("[data-subagent-row]")
    grok, long = cards.nth(0), cards.nth(4)
    grok.locator("[data-subagent-details] > summary").click()
    long.locator("[data-processing-details] > summary").click()
    opened = """() => [...document.querySelectorAll('[data-subagent-row]')].map(card =>
        [card.querySelector('[data-subagent-details]').open, card.querySelector('[data-processing-details]').open])"""
    expected = [[True, False], [False, False], [False, False], [False, False], [False, True]]
    assert page.evaluate(opened) == expected

    grok.locator('[data-subagent-field="review_eligible"]').check()
    assert page.evaluate(opened) == expected
    # A structural edit (a new account pin) rebuilds the cards; disclosures and the focused control come back.
    long.locator('[data-subagent-field="account"]').focus()
    long.locator('[data-subagent-field="account"]').select_option("work")
    page.wait_for_function("() => document.activeElement?.dataset?.subagentField === 'account'")
    assert page.evaluate(opened) == expected
    # A status refresh patches in place.
    page.evaluate("async () => (await import('/static/modules/subagents_settings.js')).reloadSubagentsSection()")
    assert page.evaluate(opened) == expected
    # A focused summary keeps focus through a rebuild too.
    grok.locator("[data-subagent-details] > summary").focus()
    long.locator('[data-subagent-field="account"]').evaluate("e => { e.value = 'personal'; e.dispatchEvent(new Event('change')); }")
    assert page.evaluate("document.activeElement?.parentElement?.hasAttribute('data-subagent-details')")
    with page.expect_response("**/api/settings"):
        page.locator("#btn-save-settings").click()
    page.wait_for_function("() => document.querySelector('[data-subagents-intent]')?.textContent === 'Saved'")
    assert page.evaluate(opened) == expected
    roles.capture(page, "quiet-disclosures-after-save")


def test_delegation_switch_names_what_it_stops(role_ui):
    ui = role_ui
    _roster(ui)
    page = _open(ui, VIEWPORTS["laptop"])
    toggle = page.locator("[data-subagents-enabled]")
    assert page.locator(".available-subagents-toolbar label.local-toggle").inner_text().strip() == "Delegation"
    stays = page.locator("[data-review-pool-stays]")
    assert stays.is_hidden()
    before = _box(toggle)
    toggle.uncheck()
    # Said when it matters, below the switch, which stays where the owner clicked it.
    assert stays.inner_text() == "Delegation is off; rows marked Reviewer still review."
    assert _box(toggle) == before
    roles.capture(page, "quiet-delegation-off")


@pytest.mark.parametrize("long_labels", [False, True])
@pytest.mark.parametrize("consumer,width,tracks", [("settings", 700, 1), ("settings", 740, 2), ("wizard", 700, 3)])
def test_narrow_consumers_use_card_columns_and_one_dirty_indicator(role_ui, long_labels, consumer, width, tracks, record_property):
    ui = role_ui
    rows = _roster(ui)
    if consumer == "settings":
        page = _open(ui, (width, 900))
    else:
        ui["fixture"]["preview"]["available_subagents"] = {"enabled": True, "items": rows}
        page = ui["page"]
        page.set_viewport_size({"width": width, "height": 900})
        page.goto(ui["url"] + "/onboarding")
        page.wait_for_selector("#quick-start-btn:not([hidden])")
        page.click("#next-btn")
        page.locator("details:has(#onboarding-available-subagents) > summary").click()
        page.locator("[data-subagent-row]").first.wait_for()
        page.evaluate("""async () => {
            await document.fonts.ready;
            for (let i = 0; i < 8; i++) await new Promise(requestAnimationFrame);
        }""")
    card = page.locator("[data-subagent-row]").first
    if long_labels:
        card.locator(".available-subagent-route > .available-subagent-field").evaluate_all("""labels => {
            const words = ['Modellquelle auswählen', 'Sprachmodell und Kontextfenster',
                'Verbindung und Benutzerkonto', 'Intensität der Schlussfolgerung', 'Zugriffsberechtigungen'];
            labels.forEach((label, i) => { label.firstChild.textContent = words[i] + ' '; });
        }""")
        card.locator(".available-subagent-reviewer").evaluate("""label => {
            [...label.childNodes].filter(n => n.nodeType === 3 && n.textContent.trim())
                .forEach(n => { n.textContent = ' Als Gutachter verwenden '; });
        }""")
    card.evaluate("e => e.scrollIntoView({block: 'start'})")
    shape = card.evaluate("""card => {
        const route = card.querySelector('.available-subagent-route');
        const rem = parseFloat(getComputedStyle(document.documentElement).fontSize);
        const style = getComputedStyle(card);
        const width = card.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
        const bounds = route.getBoundingClientRect();
        return {width, expected: width <= 22 * rem ? 1 : width <= 34 * rem ? 2 : 3,
            tracks: getComputedStyle(route).gridTemplateColumns.split(' ').length,
            overflow: [...route.querySelectorAll('select, label')].some(e =>
                e.getBoundingClientRect().right > bounds.right + 1),
            pageOverflow: document.documentElement.scrollWidth > innerWidth};
    }""")
    record_property("card_columns", json.dumps(shape))
    # At 740px Settings has room for two columns despite the old <=760px override.
    # At 700px its sidebar leaves one, while the wizard's wider card holds three.
    assert shape["expected"] == tracks, shape
    assert shape["tracks"] == shape["expected"], shape
    assert not shape["overflow"] and not shape["pageOverflow"], shape
    box = card.locator('[data-subagent-field="review_eligible"]')
    box.focus()
    start, card_top = _box(box), _box(card)
    page.keyboard.press("Space")
    assert box.is_checked() and _box(box) == start and _box(card) == card_top
    assert box.evaluate("e => e === document.activeElement")
    assert page.locator("[data-subagents-intent]").inner_text() == ("" if consumer == "settings" else "Unsaved changes")
    if consumer == "settings":
        assert page.locator("#settings-unsaved-indicator").is_visible()
    assert card.locator('[data-subagent-field="recommended_use"]').input_value() == DESCRIPTION
    roles.capture(page, f"quiet-{width}-{consumer}-{'long-labels' if long_labels else 'english'}")


def test_wizard_consumer_keeps_the_reviewer_box_in_place(role_ui):
    ui = role_ui
    page = ui["page"]
    page.set_viewport_size({"width": 1095, "height": 858})
    page.goto(ui["url"] + "/onboarding")
    page.wait_for_selector("#quick-start-btn:not([hidden])")
    page.click("#next-btn")
    page.locator("details:has(#onboarding-available-subagents) > summary").click()
    card = page.locator("#onboarding-available-subagents [data-subagent-row]").first
    box = card.locator('[data-subagent-field="review_eligible"]')
    box.wait_for()
    intent = page.locator("#onboarding-available-subagents [data-subagents-intent]")
    assert intent.inner_text() == "Generated draft"
    box.evaluate("e => { window.reviewerNode = e; }")
    card.scroll_into_view_if_needed()
    start = _box(box)
    box.click()
    assert _box(box) == start and intent.inner_text() == "Unsaved changes"
    box.focus()
    page.keyboard.press("Space")
    assert _box(box) == start and page.evaluate("document.activeElement === window.reviewerNode")
    assert card.locator("[data-subagent-details] > summary").inner_text() == "Details & history"
    roles.capture(page, "quiet-wizard-1095")
