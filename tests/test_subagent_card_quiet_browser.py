"""The agent card's head and form in a REAL render (docs/DESIGN.md §6). Settings draws the form on an agent's
page: the Reviewer switch writes at once and keeps its place under pointer and keyboard, the form's unsaved text
and open disclosures survive every repaint, and history waits behind Details & history while current exceptions
stay visible. The first-run wizard draws head and form in one card, as before. Controlled API responses, no
runtime, no model calls."""
from __future__ import annotations

import json

import pytest
from tests import test_subscription_role_routes_browser as roles
from tests.settings_browser_helpers import open_entry, open_settings, press_editor_save, settings_page

pytestmark = [pytest.mark.ui_browser]
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


def _open(ui, size, scheme="light", agent="grok"):
    """An agent's page (or the list, with `agent=None`), settled: fonts in, the pool facts read."""
    page = ui["page"]
    page.set_viewport_size({"width": size[0], "height": size[1]})
    page.emulate_media(color_scheme=scheme)
    page = roles.open_agents(ui)
    if agent:
        _agent(page, agent)
    page.evaluate("""async () => {
        await document.fonts.ready;
        for (let i = 0; i < 8; i++) await new Promise(requestAnimationFrame);
    }""")
    return page


def _agent(page, agent_id):
    """Open an agent's page from the list and return its form once the review-pool facts have landed."""
    open_settings(page, "agents")
    form = open_entry(page, agent_id).locator("[data-subagent-row]")
    page.wait_for_function("""path => { const level = [...document.querySelectorAll('#page-settings [data-settings-page]')]
        .find((el) => el.dataset.settingsPage === path);
        return level && /\\S/.test(level.querySelector('[data-subagent-review-facts]')?.textContent || ''); }""",
                           arg=f"agents/{agent_id}")
    return form


def _settled(page, field, checked, focused=False):
    """A switch of the open agent's page shows `checked` and its write is over; `focused`: it holds the focus it
    had when it was pressed from the keyboard (WebKit need not focus a control pressed by the pointer)."""
    page.wait_for_function("""([field, checked, focused]) => {
        const level = [...document.querySelectorAll('#page-settings [data-settings-page]')].filter((el) => !el.hidden).at(-1);
        const input = level?.querySelector(`[data-subagent-field="${field}"]`);
        return input && input.checked === checked && input.getAttribute('aria-busy') !== 'true' && !input.disabled
            && (!focused || document.activeElement === input); }""", arg=[field, checked, focused])


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
def test_reviewer_keeps_its_place_under_pointer_and_keyboard_and_the_form_keeps_its_draft(role_ui, viewport, scheme):
    ui = role_ui
    _roster(ui)
    page = _open(ui, VIEWPORTS[viewport], scheme)
    form = settings_page(page, "agents/grok").locator("[data-subagent-row]")
    box = form.locator('[data-subagent-field="review_eligible"]')
    switch = form.locator('.agent-use:has([data-subagent-field="review_eligible"]) .ui-switch')
    text = form.locator('[data-subagent-field="recommended_use"]')
    draft = DESCRIPTION + " И ещё одно несохранённое слово."
    text.fill(draft)
    text.evaluate("e => e.setSelectionRange(5, 9)")
    roles.capture(page, f"quiet-{viewport}-{scheme}-before")
    start, card_top = _box(switch), _box(form)

    # Pointer path: the mark is one write of its entry; the switch stays where it was pressed.
    switch.click()
    _settled(page, "review_eligible", True)
    assert _box(switch) == start and _box(form) == card_top
    assert len(roles.roster_writes(ui)) == 1 and roles.saved_catalog(ui)["items"][0]["review_eligible"] is True
    assert roles.saved_catalog(ui)["items"][0]["recommended_use"] == DESCRIPTION, "the switch never publishes the form"
    roles.capture(page, f"quiet-{viewport}-{scheme}-marked")
    switch.click()
    _settled(page, "review_eligible", False)
    assert _box(switch) == start and _box(form) == card_top

    # Keyboard path: each press writes, and the focus comes back to the switch every time.
    box.focus()
    for expected in (True, False, True):
        page.keyboard.press("Space")
        _settled(page, "review_eligible", expected, focused=True)
        assert _box(switch) == start and _box(form) == card_top
    assert text.input_value() == draft, "the unsaved text survives every repaint"
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    assert settings_page(page, "agents/grok").evaluate("el => el.scrollWidth <= el.clientWidth + 1")

    # The page's own Save publishes the text, into its own entry, with the marks the switches stored. (An edit
    # ends the switch's result line, so the Save's own result is seen to arrive.)
    text.fill(draft)
    assert press_editor_save(page).startswith("Saved")
    saved = roles.saved_catalog(ui)["items"]
    assert saved[0]["review_eligible"] is True and saved[0]["recommended_use"] == draft
    assert [row.get("review_eligible") for row in saved[1:]] == [True, True, True, None]
    assert saved[1]["route"]["credential_profile_id"] == "ghost" and saved[3]["enabled"] is False
    assert saved[3]["effort"] == "low" and saved[4]["route"]["target_id"].endswith(LONG_MODEL)


def test_current_exceptions_stay_visible_and_history_waits_in_details(role_ui):
    ui = role_ui
    _roster(ui)
    page = _open(ui, VIEWPORTS["laptop"], agent=None)
    level = settings_page(page, "agents")
    # The list: a head says the one word of a current exception; nothing of an agent's history.
    assert level.locator('[data-entry-id="grok"] .agent-chip[data-tone="warn"]').inner_text() == "Unavailable"
    assert level.locator('[data-entry-id="pinned"] .agent-chip[data-tone="warn"]').inner_text() == "Unavailable"
    assert level.locator('[data-entry-id="packet"] .agent-chip[data-tone="warn"]').count() == 0
    assert not any(word in level.inner_text() for word in HISTORY_WORDS)
    forms = {}
    for agent in ("grok", "pinned", "packet", "parked", "long"):
        forms[agent] = _agent(page, agent)
        visible = _visible_card_text(forms[agent])
        assert not any(word in visible for word in HISTORY_WORDS), visible
        assert "Saved ·" not in visible and "Draft ·" not in visible
        # An agent's page says the whole status sentence, in view: no hover is needed for it.
        status = forms[agent].locator("[data-subagent-status]")
        assert status.get_attribute("title") is None and status.inner_text().strip()
        if agent == "grok":
            assert status.inner_text() == "cursor · currently unavailable"
            labels = forms[agent].locator(".available-subagent-route > .available-subagent-field").evaluate_all(
                "els => els.map(e => e.firstChild.textContent.trim())")
            assert labels == ["Source", "Model", "Account", "Reasoning effort", "Access"]
            roles.capture(page, "quiet-exceptions-closed")
            # Details: the two histories are different events, each named.
            forms[agent].locator("[data-subagent-details] > summary").click()
            task = forms[agent].locator("[data-subagent-last-task] dd").inner_text()
            assert task.startswith("Earlier settings · cursor session · Grok 4.7 256K") and "failed (quota_exhausted)" in task
            assert forms[agent].locator("[data-subagent-last-review]").is_hidden()
            assert forms[agent].locator("[id$='-access-help']").inner_text().startswith("Full system access (the default)")
            roles.capture(page, "quiet-details-open")
        elif agent == "pinned":
            assert status.inner_text() == "codex · pinned account ghost currently unavailable"
            # A marked Auto row reviews at the top of the owner's chat range (#1693; ARCHITECTURE §3 "Available subagents").
            assert forms[agent].locator('[data-subagent-field="effort"] option[value=""]').inner_text() == (
                "Auto (reviews at the top of the chat range)")
            forms[agent].locator("[data-subagent-details] > summary").click()
            assert forms[agent].locator("[data-subagent-last-review] dd").inner_text() == (
                "codex session · gpt-test · account personal · record rev_42")
        elif agent == "packet":
            assert forms[agent].locator('[data-subagent-field="delivery"]').input_value() == "packet"
            assert forms[agent].locator("[data-subagent-delivery-cost]").inner_text() == "cost unknown"
            assert forms[agent].locator("[data-subagent-stored] dd").text_content() == "openai::gpt-api"
        elif agent == "parked":
            assert forms[agent].locator("[data-subagent-review-exception]").inner_text() == (
                "Switched off, so not in the review pool.")
        else:
            assert forms[agent].locator('[data-subagent-field="model"]').input_value() == LONG_MODEL


def test_open_disclosures_drafts_and_focus_survive_repaint_refresh_and_save(role_ui):
    ui = role_ui
    _roster(ui)
    page = _open(ui, VIEWPORTS["laptop"], agent="long")
    form = settings_page(page, "agents/long").locator("[data-subagent-row]")
    form.locator("[data-subagent-details] > summary").click()
    form.locator("[data-processing-details] > summary").click()
    opened = """() => { const form = [...document.querySelectorAll('#page-settings [data-settings-page]')]
        .filter((el) => !el.hidden).at(-1).querySelector('[data-subagent-row]');
        return [form.querySelector('[data-subagent-details]').open, form.querySelector('[data-processing-details]').open]; }"""
    assert page.evaluate(opened) == [True, True]
    form.locator('[data-subagent-field="recommended_use"]').fill("Unsaved words.")

    # A switch's write repaints the page from the server: the disclosures and the unsaved text come back.
    form.locator('.agent-use:has([data-subagent-field="review_eligible"]) .ui-switch').click()
    _settled(page, "review_eligible", True)
    assert page.evaluate(opened) == [True, True]
    assert form.locator('[data-subagent-field="recommended_use"]').input_value() == "Unsaved words."
    # A structural edit (a new account pin) rebuilds the form; the focused control comes back.
    form.locator('[data-subagent-field="account"]').focus()
    form.locator('[data-subagent-field="account"]').select_option("work")
    page.wait_for_function("() => document.activeElement?.dataset?.subagentField === 'account'")
    assert page.evaluate(opened) == [True, True]
    # A status refresh patches in place.
    page.evaluate("async () => (await import('/static/modules/claudexor_status_store.js')).claudexorStatus.refresh({ includeModels: true })")
    assert page.evaluate(opened) == [True, True]
    # A focused summary keeps focus through a rebuild too.
    form.locator("[data-subagent-details] > summary").focus()
    form.locator('[data-subagent-field="account"]').evaluate("e => { e.value = 'personal'; e.dispatchEvent(new Event('change')); }")
    assert page.evaluate("document.activeElement?.parentElement?.hasAttribute('data-subagent-details')")
    assert press_editor_save(page).startswith("Saved")
    assert page.evaluate(opened) == [True, True]
    assert roles.saved_catalog(ui)["items"][4]["recommended_use"] == "Unsaved words."
    roles.capture(page, "quiet-disclosures-after-save")


def test_the_rosters_switch_names_what_it_stops(role_ui):
    ui = role_ui
    _roster(ui)
    page = _open(ui, VIEWPORTS["laptop"], agent=None)
    level = settings_page(page, "agents")
    row = level.locator('[data-setting-row="OUROBOROS_SUBAGENTS"]')
    toggle = row.locator(".ui-switch")
    assert row.locator(".ui-row-title").inner_text().strip() == "Hand work to helpers"
    assert row.locator("[data-setting-meta]").inner_text() == "Ouroboros decides when. Off stops it in every room."
    stays = level.locator("[data-review-pool-stays]")
    assert stays.is_hidden()
    before = _box(toggle)
    toggle.click()
    page.wait_for_function("() => document.querySelector('#page-settings [data-review-pool-stays]')?.hidden === false")
    # Said when it matters, in the list below the switch, which stays where the owner pressed it.
    assert stays.inner_text() == "Helpers are off; agents marked Reviewer still review."
    assert _box(toggle) == before and not row.locator("[data-setting-control]").is_checked()
    assert row.locator("[data-setting-result]").inner_text().strip().startswith("Saved")
    roles.capture(page, "quiet-delegation-off")


@pytest.mark.parametrize("long_labels", [False, True])
@pytest.mark.parametrize("consumer,width,tracks", [("settings", 360, 1), ("settings", 720, 1), ("settings", 1360, 3), ("wizard", 700, 3)])
def test_narrow_consumers_lay_the_form_out_by_their_own_width(role_ui, long_labels, consumer, width, tracks, record_property):
    ui = role_ui
    rows = _roster(ui)
    if consumer == "settings":
        page = _open(ui, (width, 900))
        card = settings_page(page, "agents/grok").locator("[data-subagent-row]")
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
        if consumer == "wizard":
            card.locator(".available-subagent-reviewer").evaluate("""label => {
                [...label.childNodes].filter(n => n.nodeType === 3 && n.textContent.trim())
                    .forEach(n => { n.textContent = ' Als Gutachter verwenden '; });
            }""")
    card.evaluate("e => e.scrollIntoView({block: 'start'})")
    shape = card.evaluate("""(card, consumer) => {
        const route = card.querySelector('.available-subagent-route');
        const rem = parseFloat(getComputedStyle(document.documentElement).fontSize);
        const style = getComputedStyle(card);
        const width = card.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
        const bounds = route.getBoundingClientRect();
        const expected = consumer === 'settings' ? (width <= 34 * rem ? 1 : 3)
            : width <= 22 * rem ? 1 : width <= 34 * rem ? 2 : 3;
        return {width, expected,
            tracks: getComputedStyle(route).gridTemplateColumns.split(' ').length,
            overflow: [...route.querySelectorAll('select, label')].some(e =>
                e.getBoundingClientRect().right > bounds.right + 1),
            pageOverflow: document.documentElement.scrollWidth > innerWidth};
    }""", consumer)
    record_property("card_columns", json.dumps(shape))
    # The form reads its own container: Settings keeps the full Auto wording on one field per line below 34rem,
    # including a 720px window beside its sidebar. The setup card keeps its own one/two/three-column grid.
    assert shape["expected"] == tracks, shape
    assert shape["tracks"] == shape["expected"], shape
    assert not shape["overflow"] and not shape["pageOverflow"], shape
    box = card.locator('[data-subagent-field="review_eligible"]')
    place = card.locator('.agent-use:has([data-subagent-field="review_eligible"]) .ui-switch') if consumer == "settings" else box
    box.focus()
    start, card_top = _box(place), _box(card)
    page.keyboard.press("Space")
    if consumer == "settings":
        _settled(page, "review_eligible", True, focused=True)  # one write of the entry; the focus returns to the switch
    assert box.is_checked() and _box(place) == start and _box(card) == card_top
    assert box.evaluate("e => e === document.activeElement")
    if consumer == "wizard":
        assert page.locator("[data-subagents-intent]").inner_text() == "Unsaved changes"
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


def test_the_list_says_automatic_for_the_reads_own_catalog_and_nothing_for_a_saved_one(subscription_ui, tmp_path,
                                                                                          monkeypatch):
    """The real page over a real Settings read: a never-configured install's factory catalog is said to be automatic
    (it follows Main and access) and that the first change saves it; the same catalog saved, equal to the factory's
    row for row, says nothing. The line is the read's fact, never inferred from the rows."""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros import config as cfg
    from ouroboros import review_pool_migration
    from ouroboros.gateway import settings as settings_mod

    ui = subscription_ui
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(review_pool_migration, "_MIGRATIONS_SEEN", {})
    for key in cfg.SETTINGS_DEFAULTS:
        monkeypatch.delenv(key, raising=False)
    app = Starlette(routes=[Route("/api/settings", endpoint=settings_mod.api_settings_get, methods=["GET"])])
    app.state.drive_root = app.state.repo_dir = tmp_path

    def agents_over(document):
        cfg.SETTINGS_PATH.write_text(json.dumps(document), encoding="utf-8")
        ui["settings"].clear()
        ui["settings"].update(TestClient(app).get("/api/settings").json())
        page = ui["new_page"]()
        page.goto(ui["url"] + "/#settings")
        level = open_settings(page, "agents")
        level.locator("[data-entry-id]").first.wait_for(state="visible")
        return page, level.locator("[data-agents-automatic]")

    page, automatic = agents_over({"ANTHROPIC_API_KEY": "sk-ant-present-key"})
    assert ui["settings"]["_meta"]["available_subagents"]["automatic"] is True
    assert automatic.inner_text() == ("Automatic: this list follows your Main model and accounts. "
                                      "Your first change here saves it as yours.")
    assert not [body for path, body in ui["posts"] if path in {"/api/settings", "/api/owner/safety-model"}]
    roles.capture(page, "intent-automatic")
    factory = ui["settings"]["OUROBOROS_SUBAGENTS"]
    page, automatic = agents_over({"ANTHROPIC_API_KEY": "sk-ant-present-key", "OUROBOROS_SUBAGENTS": factory})
    assert ui["settings"]["_meta"]["available_subagents"]["automatic"] is False and automatic.count() == 0
    roles.capture(page, "intent-saved-factory-equal")
