"""The Dashboard tabs' own settings: Consciousness, Evolution, Updates and Storage each carry a card of rows that save
themselves (settings_controls.js), drawn once outside every region the tab repaints (dashboard.js mountTabSettings).

The real SPA in headless Chromium over the shared fixture's endpoints, the settings endpoint answered here: on each
tab a field the owner is in keeps its text and caret through the tab's own periodic refresh and a settings read, and
leaving it shows the row's result line; the Updates channel keeps its focused choice through a status read; the
Evolution choice "Every N tasks" posts both keys in one body; the Consciousness allowance door opens its row in Costs
(app.js openDashboardTab passes the row on). A tab opened from Settings shows Back first, and Escape, Back itself or
the system's Back return to Settings at the scroll position it left, in a window (1087 x 740) and at the 320 x 640
floor, where the menu stays and no tab scrolls sideways.

INTEGRATED: the save's `changes` receipt and `_meta.effects` (lane A) are answered here until lane A lands; the result
lines' "was ..." and effect words need them. A tab is opened from Settings the way openDashboardTab does it (show the
Dashboard, then activateTab with `back`), since Settings' own doors come with lanes C1 and C2; Back opens the Settings
page until lane C1's openSettings(path) lands.
"""
from __future__ import annotations

import json
from urllib.parse import urlparse

import pytest

from tests.test_subscription_setup_browser import subscription_ui as subscription_ui

pytestmark = [pytest.mark.ui_browser]
MAX_TASKS, ALLOWANCE = "OUROBOROS_CONSCIOUSNESS_MAX_TASKS", "OUROBOROS_CONSCIOUSNESS_DAILY_USD"
ON, CADENCE, BUDGET = "OUROBOROS_POST_TASK_EVOLUTION", "OUROBOROS_POST_TASK_EVOLUTION_CADENCE", "OUROBOROS_POST_TASK_EVOLUTION_BUDGET_USD"
CHANNEL, RETENTION = "OUROBOROS_UPDATE_CHANNEL", "OUROBOROS_GC_RETENTION_DAYS"
STORED = {"OUROBOROS_CONSCIOUSNESS_AUTONOMY": "act", "OUROBOROS_BG_WAKEUP_MIN": 300, "OUROBOROS_BG_WAKEUP_MAX": 7200, MAX_TASKS: 2,
          "OUROBOROS_PRESENCE_MAX_ACTIVE": 3, ALLOWANCE: 20, ON: False, CADENCE: "llm", BUDGET: 5,
          "OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE": "", CHANNEL: "stable", RETENTION: 7}
# INTEGRATED (lane A): when each key's change applies, as GET /api/settings `_meta.effects` says it.
EFFECTS = {MAX_TASKS: "next_wake", "OUROBOROS_BG_WAKEUP_MIN": "next_wake", "OUROBOROS_BG_WAKEUP_MAX": "next_wake",
           ON: "next_task", CADENCE: "next_task", BUDGET: "next_task", CHANNEL: "now", RETENTION: "now"}
VOLUME = {"free_bytes": 170e9, "total_bytes": 4e12, "observed_at": "2026-11-11T10:00:00Z"}
# A count that runs: Storage reads itself every second while it does.
COUNTING = {"started_at": "2026-11-11T10:59:00Z", "place": "engine", "current": "projects", "places_done": 1,
            "places_total": 12, "bytes": 1e9, "files": 10, "done": []}
NOW = "window.dispatchEvent(new CustomEvent('ouro:settings-updated', { detail: { source: 'settings' } }))"


@pytest.fixture
def ui(request, monkeypatch):
    monkeypatch.setenv("OUROBOROS_UI_BROWSER_ENGINE", "chromium")
    return request.getfixturevalue("subscription_ui")


def _nav(page, place, width):
    if width < 720:
        page.locator(".page.active [data-mobile-nav-toggle]:visible").click()
    page.click(f'[data-nav-page="{place}"]')
    page.wait_for_selector(f"#page-{place}.active")


def _open(ui, *, tab=None, width=1087, height=740):
    """The app with the stored settings above; a save stores its body and answers the receipt (INTEGRATED, lane A)."""
    ui["settings"].update(STORED)
    ui["settings"]["_meta"] = {**ui["settings"]["_meta"], "effects": EFFECTS}
    page = ui["new_page"](viewport={"width": width, "height": height})
    saves = []

    def settings(route):
        if route.request.method != "POST":
            route.fallback()
            return
        body = route.request.post_data_json
        saves.append(body)
        changes = {key: {"old": ui["settings"].get(key), "new": value, "token": f"{key}@{len(saves)}"} for key, value in body.items()}
        ui["settings"].update(body)
        route.fulfill(content_type="application/json", body=json.dumps({"status": "saved", "changes": changes}))

    page.route("**/api/settings", settings)
    page.route("**/api/storage", lambda route: route.fulfill(content_type="application/json", body=json.dumps(
        {"count": None, "counting": COUNTING, "error": None, "volume": VOLUME, "retention_days": 7})))
    page.clock.install()
    page.goto(ui["url"] + "/")
    page.wait_for_selector("#page-chat")
    if tab:
        _nav(page, "dashboard", width)
        page.click(f'[data-dashboard-tab="{tab}"]')
    return page, saves


def _reads(path, method="GET"):
    return lambda request: urlparse(request.url).path == path and request.method == method


def _sideways(page):
    """What scrolls sideways: the page, or a settings card or row wider than its box."""
    return page.evaluate("""() => [document.documentElement.scrollWidth > innerWidth ? 'page' : '',
        ...[...document.querySelectorAll('.page.active .dashboard-settings, .page.active [data-setting-row]')]
            .filter((el) => el.getClientRects().length && el.scrollWidth > el.clientWidth + 1)
            .map((el) => el.dataset.settingRow || el.getAttribute('aria-label') || el.className)].filter(Boolean)""")


FIELDS = [
    # The tab, its field, the read its own refresh makes and that refresh's period, the line leaving the field shows.
    ("consciousness", MAX_TASKS, "/api/consciousness", 21_000, "Saved · applies from the next wake · was 2 · Undo"),
    ("evolution", BUDGET, "/api/state", 21_000, "Saved · applies to new tasks · was 5 · Undo"),
    ("storage", RETENTION, "/api/storage", 1_500, "Saved · was 7 · Undo"),
]


@pytest.mark.parametrize("tab,key,read,period,line", FIELDS, ids=[field[0] for field in FIELDS])
def test_a_field_keeps_its_text_and_caret_through_the_tab_refresh_and_leaving_it_says_what_happened(ui, tab, key, read, period, line):
    page, saves = _open(ui, tab=tab)
    field = f"setting-{key}"
    page.wait_for_function("([id, value]) => document.getElementById(id)?.value === value", arg=[field, str(STORED[key])])
    page.fill(f"#{field}", "12")
    page.keyboard.press("ArrowLeft")
    page.evaluate("(id) => { document.getElementById(id).typedHere = true; }", field)
    with page.expect_request(_reads(read)):
        page.clock.fast_forward(period)
    with page.expect_request(_reads("/api/settings")):
        page.evaluate(NOW)
    page.wait_for_timeout(300)
    caret = page.evaluate("""(id) => { const el = document.getElementById(id);
        return [document.activeElement === el, el.typedHere === true, el.value, el.selectionStart, el.selectionEnd]; }""", field)
    assert caret == [True, True, "12", 1, 1], "the same field keeps the focus, the text and the caret"
    assert saves == [], "typing saves nothing"
    page.keyboard.press("Tab")
    page.wait_for_function("([id, words]) => document.getElementById(id)?.textContent === words", arg=[f"{field}-result", line])
    assert saves == [{key: 12}], saves
    assert _sideways(page) == []


def test_the_update_channel_keeps_its_focused_choice_through_a_status_read_and_saves_when_pressed(ui):
    page, saves = _open(ui, tab="updates")
    row = f'#page-dashboard [data-setting-row="{CHANNEL}"]'
    page.wait_for_selector(f'{row} [data-setting-value="stable"][aria-pressed="true"]')
    assert page.locator("#page-dashboard [data-open-settings-advanced]").count() == 0, "no link to Settings for the channel"
    page.focus(f'{row} [data-setting-value="qa"]')
    page.evaluate("() => { document.activeElement.typedHere = true; }")
    with page.expect_request(_reads("/api/settings")):
        page.evaluate("window.dispatchEvent(new CustomEvent('ouro:dashboard-subtab-shown', { detail: { tab: 'updates' } }))")
    page.wait_for_timeout(300)
    assert page.evaluate("() => document.activeElement.typedHere === true && document.activeElement.dataset.settingValue") == "qa"
    page.keyboard.press("Enter")
    page.wait_for_selector(f'{row} [data-setting-value="qa"][aria-pressed="true"]')
    assert saves == [{CHANNEL: "qa"}], saves
    assert page.locator(f'{row} [data-setting-value="stable"]').get_attribute("aria-pressed") == "false"


def test_every_n_tasks_shows_its_count_and_saves_both_keys_in_one_body(ui):
    page, saves = _open(ui, tab="evolution")
    card = '#page-dashboard [aria-label="Cycles"]'
    result = f"setting-{ON}-result"
    page.wait_for_selector(f'{card} [data-evo-when="off"][aria-pressed="true"]')
    assert page.locator(f"{card} [data-evo-n]").is_hidden(), "no count while it is off"
    page.click(f'{card} [data-evo-when="every_n"]')
    page.wait_for_selector(f'{card} [data-evo-when="every_n"][aria-pressed="true"]')
    assert saves == [{ON: True, CADENCE: "every_n:3"}], saves
    page.wait_for_function("(id) => document.getElementById(id)?.textContent === 'Saved · applies to new tasks · was Off'", arg=result)
    page.fill(f"{card} [data-evo-n]", "5")
    assert len(saves) == 1, "typing the count saves nothing"
    page.keyboard.press("Tab")
    page.wait_for_function("(id) => document.getElementById(id)?.textContent === 'Saved · applies to new tasks · was Every 3 tasks'", arg=result)
    assert saves[-1] == {ON: True, CADENCE: "every_n:5"}, saves
    page.fill(f"{card} [data-evo-n]", "0")
    page.keyboard.press("Tab")
    page.wait_for_function("(id) => document.getElementById(id)?.textContent === 'Not saved. Enter a whole number of tasks, 1 or more.'", arg=result)
    assert len(saves) == 2, "a count of 0 is never sent"
    page.click(f'{card} [data-evo-when="off"]')
    page.wait_for_selector(f'{card} [data-evo-when="off"][aria-pressed="true"]')
    assert saves[-1] == {ON: False, CADENCE: "llm"}, saves
    assert page.locator(f"{card} [data-evo-n]").is_hidden()


def test_the_allowance_door_opens_its_row_in_costs(ui):
    page, _ = _open(ui, tab="consciousness")
    door = '#page-dashboard [data-setting-row="consciousness-allowance"]'
    page.wait_for_function("(s) => document.querySelector(s)?.textContent.includes('$20.00')", arg=door)
    page.click(door)
    page.wait_for_selector('#dashboard-panel-costs.active')
    page.wait_for_function("(id) => document.activeElement?.id === id", arg=f"setting-{ALLOWANCE}")
    assert page.locator("#dashboard-panel-costs > .dashboard-back").count() == 0, "a door inside the Dashboard leaves no Back"


@pytest.mark.parametrize("width,height", [(1087, 740), (320, 640)])
def test_a_tab_opened_from_settings_leads_back_to_settings_where_it_was(ui, width, height):
    page, _ = _open(ui, width=width, height=height)
    _nav(page, "settings", width)
    scroller = "#page-settings .settings-scroll"
    page.wait_for_selector(scroller)
    top = page.evaluate("(s) => { const el = document.querySelector(s); el.scrollTop = 240; return el.scrollTop; }", scroller)
    assert top > 0, "Settings scrolls at this size"
    for leave in ("Escape", "Back", "system Back"):
        _nav(page, "dashboard", width)
        page.evaluate("document.getElementById('page-dashboard').activateDashboardTab('consciousness', { back: '' })")
        back = page.locator("#dashboard-panel-consciousness > .dashboard-back")
        back.wait_for()
        assert page.evaluate("() => document.getElementById('dashboard-panel-consciousness').firstElementChild.className") == "dashboard-back"
        assert back.inner_text().strip() == "Settings"
        assert page.evaluate("""() => { const back = document.querySelector('#dashboard-panel-consciousness > .dashboard-back');
            return back.getBoundingClientRect().bottom <= document.querySelector('#dashboard-panel-consciousness > .dashboard-body').getBoundingClientRect().top + 1; }"""), \
            "Back sits above the tab's scroller"
        if width < 720:
            assert page.locator("#page-dashboard [data-mobile-nav-toggle]").is_visible(), "the menu stays"
        assert _sideways(page) == [], leave
        if leave == "Escape":
            page.keyboard.press("Escape")
        elif leave == "Back":
            back.locator("button").click()
        else:
            page.go_back()
        page.wait_for_selector("#page-settings.active")
        assert back.count() == 0, leave
        assert page.evaluate("(s) => document.querySelector(s).scrollTop", scroller) == top, f"{leave} returns where Settings was"


def test_no_tab_scrolls_sideways_at_the_floor(ui):
    page, _ = _open(ui, tab="consciousness", width=320, height=640)
    for tab in ("consciousness", "costs", "updates", "storage", "evolution"):
        page.click(f'[data-dashboard-tab="{tab}"]')
        page.wait_for_selector(f"#dashboard-panel-{tab}.active")
        page.wait_for_timeout(200)
        assert _sideways(page) == [], tab
