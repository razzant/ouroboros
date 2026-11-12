"""The Dashboard tabs' own settings: Consciousness, Evolution, Updates and Storage each carry cards of rows that save
themselves (settings_controls.js), drawn once outside every region the tab repaints (dashboard.js mountTabSettings),
and Costs carries the subscriptions block from the passive quota read.

The real SPA in headless Chromium and WebKit over the shared fixture's endpoints, the settings endpoint answered here: on each
tab a field the owner is in keeps its text and caret through the tab's own periodic refresh and a settings read, and
leaving it shows the row's result line; the Updates channel keeps its focused choice through a status read; the
Evolution choice "Every N tasks" posts both keys in one body and says itself in its own sentence; the doors of
Consciousness and Costs lead where they say and Settings leads back. Every one of the five tabs opened from the
Settings home shows Back first, and Escape, Back itself or the system's Back return to Settings at the scroll position
it was left at, in a window (1087 x 740) and at the 320 x 640 floor, where the menu stays and no tab scrolls sideways.

The settings endpoint is answered here the way the server answers it: `_meta.effects` on a read, the receipt
(`changes`: old, new, token) on a save. Undo's precondition and its refusal are the node test's
(web/tests/dashboard_settings.test.js).
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import pytest

from tests.settings_browser_helpers import HOME, open_settings
from tests.test_subscription_setup_browser import subscription_ui as subscription_ui

pytestmark = [pytest.mark.ui_browser]
MAX_TASKS, ALLOWANCE = "OUROBOROS_CONSCIOUSNESS_MAX_TASKS", "OUROBOROS_CONSCIOUSNESS_DAILY_USD"
ON, CADENCE, BUDGET = "OUROBOROS_POST_TASK_EVOLUTION", "OUROBOROS_POST_TASK_EVOLUTION_CADENCE", "OUROBOROS_POST_TASK_EVOLUTION_BUDGET_USD"
CHANNEL, RETENTION = "OUROBOROS_UPDATE_CHANNEL", "OUROBOROS_GC_RETENTION_DAYS"
STORED = {"OUROBOROS_CONSCIOUSNESS_AUTONOMY": "act", "OUROBOROS_BG_WAKEUP_MIN": 300, "OUROBOROS_BG_WAKEUP_MAX": 7200, MAX_TASKS: 2,
          "OUROBOROS_PRESENCE_MAX_ACTIVE": 3, ALLOWANCE: 20, ON: False, CADENCE: "llm", BUDGET: 5,
          "OUROBOROS_CONSCIOUSNESS_SAFETY_MODE": "light",
          "OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE": "", CHANNEL: "stable", RETENTION: 7}
# When each key's change applies, as GET /api/settings `_meta.effects` says it.
EFFECTS = {MAX_TASKS: "next_wake", "OUROBOROS_BG_WAKEUP_MIN": "next_wake", "OUROBOROS_BG_WAKEUP_MAX": "next_wake",
           ON: "next_task", CADENCE: "next_task", BUDGET: "next_task", CHANNEL: "now", RETENTION: "now"}
VOLUME = {"free_bytes": 170e9, "total_bytes": 4e12, "observed_at": "2026-11-11T10:00:00Z"}
# A count that runs: Storage reads itself every second while it does.
COUNTING = {"started_at": "2026-11-11T10:59:00Z", "place": "engine", "current": "projects", "places_done": 1,
            "places_total": 12, "bytes": 1e9, "files": 10, "done": []}
NOW = "window.dispatchEvent(new CustomEvent('ouro:settings-updated', { detail: { source: 'settings' } }))"
TABS = ("consciousness", "costs", "updates", "storage", "evolution")


@pytest.fixture(params=["chromium", "webkit"])
def ui(request, monkeypatch):
    monkeypatch.setenv("OUROBOROS_UI_BROWSER_ENGINE", request.param)
    return request.getfixturevalue("subscription_ui")


def _nav(page, place, width):
    if width < 720:
        page.locator(".page.active [data-mobile-nav-toggle]:visible").click()
    page.click(f'[data-nav-page="{place}"]')
    page.wait_for_selector(f"#page-{place}.active")


def _quota():
    """The passive quota view: two Codex accounts in rotation (one at its limit), one out of it."""
    ahead = lambda hours: (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()
    profile = lambda name, on=True: {"profile": {"harness_id": "codex", "profile_id": name, "display_name": name, "enabled": on},
                                     "status": {}, "identity": {}}
    window = lambda name, used, hours: {"subject": {"harness": "codex", "subject_id": name}, "freshness": "fresh",
                                        "constraints": [{"id": "w", "used_ratio": used, "resets_at": ahead(hours)}]}
    return {"view": "quota", "reads": {"catalog": "not_read", "accounts": "ok", "quota": "ok"}, "read_errors": {}, "quota_absences": [],
            "profiles": {"profiles": [profile("work"), profile("home"), profile("old", False)]},
            "quota": [window("work", 0.2, 2), window("home", 1, 5)]}


def _open(ui, *, tab=None, width=1087, height=740):
    """The app with the stored settings above; a save stores its body and answers the receipt."""
    ui["settings"].update(STORED)
    ui["settings"]["_meta"] = {**ui["settings"]["_meta"], "effects": EFFECTS}
    page = ui["new_page"](viewport={"width": width, "height": height})
    saves, quota_reads = [], []

    def settings(route):
        if route.request.method != "POST":
            route.fallback()
            return
        body = route.request.post_data_json
        saves.append(body)
        changes = {key: {"old": ui["settings"].get(key), "new": value, "token": f"{key}@{len(saves)}"} for key, value in body.items()}
        ui["settings"].update(body)
        route.fulfill(content_type="application/json", body=json.dumps({"status": "saved", "changes": changes}))

    def quota(route):
        quota_reads.append(route.request.url)
        route.fulfill(content_type="application/json", body=json.dumps(_quota()))

    page.route("**/api/settings", settings)
    page.route("**/api/storage", lambda route: route.fulfill(content_type="application/json", body=json.dumps(
        {"count": None, "counting": COUNTING, "error": None, "volume": VOLUME, "retention_days": 7})))
    page.route("**/api/claudexor/status?view=quota", quota)
    page.clock.install()
    page.goto(ui["url"] + "/")
    page.wait_for_selector("#page-chat")
    if tab:
        _nav(page, "dashboard", width)
        page.click(f'[data-dashboard-tab="{tab}"]')
    page.quota_reads = quota_reads
    return page, saves


def _scale_word(page, value):
    """The Safety scale's own word for a stored level, from the one list the page uses (consciousness_words.js)."""
    return page.evaluate("""async (value) => { const app = [...document.scripts].find((script) => /app\\.js/.test(script.src)).src;
        const words = await import(new URL('modules/consciousness_words.js', app).href);
        return words.SAFETY_LEVELS.find((level) => level.value === value).label; }""", value)


def _serve(page, prefix):
    """Settings opens a level only for a path some page registered; any other lands on its home with a search
    (settings.js `go`). Where the lane that owns `prefix` has not registered its page yet, a stand-in does, so a door
    and its way back are checked here without waiting for that lane."""
    page.evaluate("""async (prefix) => { const app = [...document.scripts].find((script) => /app\\.js/.test(script.src)).src;
        const pages = await import(new URL('modules/settings_pages.js', app).href);
        if (!pages.settingsPageFor(prefix)) pages.registerSettingsPage(prefix, { title: prefix, mount() {} }); }""", prefix)


def _reads(path, method="GET"):
    return lambda request: urlparse(request.url).path == path and request.method == method


def _answers(path, method="GET"):
    return lambda response: urlparse(response.url).path == path and response.request.method == method


def _applied(page, answer):
    """The page has read `answer` whole and painted what it does with it: a request merely sent proves no repaint."""
    answer.value.finished()
    page.evaluate("() => new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done)))")


def _sideways(page):
    """What scrolls sideways: the page, or a settings card or row wider than its box."""
    return page.evaluate("""() => [document.documentElement.scrollWidth > innerWidth ? 'page' : '',
        ...[...document.querySelectorAll('.page.active .dashboard-body, .page.active .dashboard-settings, .page.active [data-setting-row]')]
            .filter((el) => el.getClientRects().length && el.scrollWidth > el.clientWidth + 1)
            .map((el) => el.dataset.settingRow || el.getAttribute('aria-label') || el.id || el.className)].filter(Boolean)""")


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
    with page.expect_response(_answers(read)) as refreshed:
        page.clock.fast_forward(period)
    _applied(page, refreshed)
    with page.expect_response(_answers("/api/settings")) as reread:
        page.evaluate(NOW)
    _applied(page, reread)
    caret = page.evaluate("""(id) => { const el = document.getElementById(id);
        return [document.activeElement === el, el.typedHere === true, el.value, el.selectionStart, el.selectionEnd]; }""", field)
    assert caret == [True, True, "12", 1, 1], "the same field keeps the focus, the text and the caret"
    assert saves == [], "typing saves nothing"
    page.keyboard.press("Tab")
    page.wait_for_function("([id, words]) => document.getElementById(id)?.textContent === words", arg=[f"{field}-result", line])
    assert saves == [{key: 12}], saves
    assert _sideways(page) == []


def test_the_update_channel_is_three_choices_and_keeps_its_focused_one_through_a_status_read(ui):
    page, saves = _open(ui, tab="updates")
    row = f'#page-dashboard [data-setting-row="{CHANNEL}"]'
    page.wait_for_selector(f'{row} [data-setting-value="stable"][aria-pressed="true"]')
    assert page.locator("#page-dashboard [data-open-settings-advanced]").count() == 0, "no link to Settings for the channel"
    assert page.locator(f"{row} .ui-row-title").inner_text() == "Which updates"
    choices = page.locator(f"{row} .dashboard-choice")
    assert [" ".join(text.split()) for text in choices.all_inner_texts()] == [
        "Stable Released versions. The main branch.", "QA Release candidates. The ouroboros-stable branch.",
        "Development Every change as it lands. The ouroboros branch."]
    boxes = [choice.bounding_box() for choice in choices.all()]
    assert all(a["y"] + a["height"] <= b["y"] and a["x"] == b["x"] and a["width"] == b["width"] for a, b in zip(boxes, boxes[1:])), \
        "the choices are cards of one width, one under another"
    page.focus(f'{row} [data-setting-value="qa"]')
    page.evaluate("() => { document.activeElement.typedHere = true; }")
    with page.expect_response(_answers("/api/settings")) as reread:
        page.evaluate("window.dispatchEvent(new CustomEvent('ouro:dashboard-subtab-shown', { detail: { tab: 'updates' } }))")
    _applied(page, reread)
    assert page.evaluate("() => document.activeElement.typedHere === true && document.activeElement.dataset.settingValue") == "qa"
    page.keyboard.press("Enter")
    page.wait_for_selector(f'{row} [data-setting-value="qa"][aria-pressed="true"]')
    assert saves == [{CHANNEL: "qa"}], saves
    assert page.locator(f'{row} [data-setting-value="stable"]').get_attribute("aria-pressed") == "false"
    assert page.locator(f"#setting-{CHANNEL}-result").inner_text() == "", "a choice that acts now is its own confirmation"


def test_every_n_tasks_shows_its_count_and_saves_both_keys_in_one_body(ui):
    page, saves = _open(ui, tab="evolution")
    card = "#page-dashboard [data-evo-settings]"
    result = f"setting-{ON}-result"
    caption = f'{card} [data-setting-row="{ON}"] [data-row-caption]'
    page.wait_for_selector(f'{card} [data-evo-when="off"][aria-pressed="true"]')
    assert page.locator(caption).inner_text() == "It starts no cycle by itself."
    assert page.locator(f"{card} [data-evo-n]").is_hidden(), "no count while it is off"
    page.click(f'{card} [data-evo-when="every_n"]')
    page.wait_for_selector(f'{card} [data-evo-when="every_n"][aria-pressed="true"]')
    assert saves == [{ON: True, CADENCE: "every_n:3"}], saves
    page.wait_for_function("(id) => document.getElementById(id)?.textContent === 'Saved · applies to new tasks · was Off · Undo'", arg=result)
    page.wait_for_function("(s) => document.querySelector(s)?.textContent.startsWith('After every 3 tasks, chats included,')", arg=caption)
    page.fill(f"{card} [data-evo-n]", "5")
    assert len(saves) == 1, "typing the count saves nothing"
    page.keyboard.press("Tab")
    page.wait_for_function("(id) => document.getElementById(id)?.textContent === 'Saved · applies to new tasks · was Every 3 tasks · Undo'", arg=result)
    assert saves[-1] == {ON: True, CADENCE: "every_n:5"}, saves
    page.fill(f"{card} [data-evo-n]", "0")
    page.keyboard.press("Tab")
    page.wait_for_function("(id) => document.getElementById(id)?.textContent === 'Not saved. Enter a whole number of tasks, 1 or more.'", arg=result)
    assert len(saves) == 2, "a count of 0 is never sent"
    page.click(f'{card} [data-evo-when="off"]')
    page.wait_for_selector(f'{card} [data-evo-when="off"][aria-pressed="true"]')
    assert saves[-1] == {ON: False, CADENCE: "llm"}, saves
    assert page.locator(f"{card} [data-evo-n]").is_hidden()
    # The standing direction is a card of its own: its field spans the card.
    field = page.locator("#setting-OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE")
    solo = page.locator(f"{card} > section[data-solo]")
    assert solo.locator(".ui-row-title").inner_text() == "What to work on"
    assert field.bounding_box()["width"] > solo.bounding_box()["width"] * 0.8


def test_the_doors_of_consciousness_lead_where_they_say_and_settings_leads_back(ui):
    page, _ = _open(ui, tab="consciousness")
    row = lambda name: f'#page-dashboard [data-setting-row="{name}"]'
    # The allowance door carries the name its limit has on Costs (a rolling 24 hours, never "a day") and the sum alone.
    page.wait_for_function("(s) => document.querySelector(s)?.querySelector('[data-row-value]')?.textContent === '$20'", arg=row("consciousness-allowance"))
    assert page.locator(row("consciousness-allowance") + " .ui-row-title").inner_text() == "Consciousness per 24 hours"
    assert page.locator("#page-dashboard [data-cx-summary] :is([data-cx-act='settings'], [data-cx-act='costs'], [data-cx-act='circles'])").count() == 0
    assert page.locator(row("OUROBOROS_CONSCIOUSNESS_AUTONOMY") + " [data-row-caption]").inner_text().startswith("Also acts on its own")
    assert page.locator(row("OUROBOROS_BG_WAKEUP") + " [data-row-caption]").inner_text() == "It picks a moment between 5 minutes and 2 hours."
    assert page.locator(row("OUROBOROS_BG_WAKEUP")).evaluate(
        "(el) => [...el.querySelectorAll('.settings-row-field > *')].map((n) => n.tagName === 'INPUT' ? n.value : n.textContent)") == ["300", "to", "7200", "s"]
    # The model door opens its role in Models, whose Back names this tab and returns to it.
    _serve(page, "models")
    _serve(page, "permissions")
    page.click(row("consciousness-model"))
    back = page.locator('#page-settings [data-settings-page="models/role/consciousness"] [data-settings-back]')
    back.wait_for()
    assert back.inner_text().strip() == "Consciousness"
    back.click()
    page.wait_for_selector("#dashboard-panel-consciousness.active")
    # The four doors of "Around it" stand in the picture's order, each with its value before its chevron.
    around = page.locator('#page-dashboard [data-cx-settings] > section[aria-label="Around it"] [data-setting-door]')
    assert [" ".join(text.split()) for text in around.evaluate_all(
        "(doors) => doors.map((door) => [door.querySelector('.ui-row-title').textContent, door.querySelector('[data-row-value]')?.textContent ?? ''].join(' | '))")] == [
        "Thinks with | Main model", f"Its actions are checked | {_scale_word(page, 'light')}", "Where it lives |", "Improving itself | Off"]
    # The Safety door opens its row's page in Settings, whose Back names this tab and returns to it.
    page.click(row("consciousness-safety"))
    back = page.locator('#page-settings [data-settings-page="permissions"] [data-settings-back]')
    back.wait_for()
    assert back.inner_text().strip() == "Consciousness"
    back.click()
    page.wait_for_selector("#dashboard-panel-consciousness.active")
    # The evolution door opens its row on the Evolution tab, and the allowance door its row in Costs: a door inside
    # the Dashboard leaves no Back.
    page.click(row("consciousness-evolution"))
    page.wait_for_selector("#dashboard-panel-evolution.active")
    page.wait_for_function("() => document.activeElement?.dataset.evoWhen === 'off'")
    assert page.locator("#dashboard-panel-evolution > .dashboard-back").count() == 0
    page.click('[data-dashboard-tab="consciousness"]')
    page.click(row("consciousness-allowance"))
    page.wait_for_selector("#dashboard-panel-costs.active")
    page.wait_for_function("(id) => document.activeElement?.id === id", arg=f"setting-{ALLOWANCE}")
    assert page.locator("#dashboard-panel-costs > .dashboard-back").count() == 0


def test_costs_shows_subscriptions_from_the_passive_view_alone_and_its_doors_open_accounts(ui):
    page, _ = _open(ui, tab="costs")
    door = page.locator('#page-costs .costs-subscriptions [data-costs-account="accounts/codex"]')
    door.wait_for()
    assert " ".join(door.inner_text().split()) == "Codex 2 in rotation · 1 at the limit · next reset in 5h · 1 out of rotation 1 of 2 have room ›"
    assert door.locator(".dashboard-settings-glyph svg").count() == 1
    assert page.quota_reads and all(urlparse(url).query == "view=quota" for url in page.quota_reads)
    assert not [url for url in ui["reads"] if "/api/claudexor/" in url and "view=quota" not in url], "no other account read from this tab"
    assert not [path for path, _ in ui["posts"] if "/api/claudexor/" in path], "nothing is woken or refreshed"
    _serve(page, "accounts")
    door.click()
    back = page.locator('#page-settings [data-settings-page="accounts/codex"] [data-settings-back]')
    back.wait_for()
    assert back.inner_text().strip() == "Costs"
    back.click()
    page.wait_for_selector("#dashboard-panel-costs.active")


@pytest.mark.parametrize("width,height", [(1087, 740), (320, 640)])
def test_a_tab_opened_from_settings_leads_back_to_settings_where_it_was(ui, width, height):
    page, _ = _open(ui, width=width, height=height)
    open_settings(page)
    assert page.evaluate("(s) => { const el = document.querySelector(s); return el.scrollHeight > el.clientHeight + 100; }", HOME), \
        "the home scrolls at this size"

    def leave_home(tab):
        """Press the home row of a tab where it stands (no scroll of the test's own) and say where the home was."""
        row = page.locator(f'{HOME} [data-settings-open="dashboard:{tab}"]')
        row.evaluate("(el) => el.scrollIntoView({ block: 'center' })")
        page.wait_for_timeout(120)
        top = page.evaluate("(s) => document.querySelector(s).scrollTop", HOME)
        row.evaluate("(el) => el.click()")
        page.wait_for_selector(f"#dashboard-panel-{tab}.active")
        return top

    def check_back(tab, leave):
        back = page.locator(f"#dashboard-panel-{tab} > .dashboard-back")
        back.wait_for()
        facts = page.evaluate("""(tab) => { const panel = document.getElementById(`dashboard-panel-${tab}`);
            const back = panel.firstElementChild, body = panel.querySelector(':scope > .dashboard-body');
            return [back.classList.contains('dashboard-back'), back.getBoundingClientRect().bottom <= body.getBoundingClientRect().top + 1]; }""", tab)
        assert facts == [True, True], "Back is the first row, above the tab's scroller"
        assert back.inner_text().strip() == "Settings"
        if width < 720:
            assert page.locator("#page-dashboard [data-mobile-nav-toggle]").is_visible(), "the menu stays"
        assert _sideways(page) == [], (tab, leave)
        if leave == "Escape":
            page.keyboard.press("Escape")
        elif leave == "Back":
            back.locator("button").click()
        else:
            page.go_back()
        page.wait_for_selector("#page-settings.active")
        assert back.count() == 0, leave

    tops = {}
    for tab in TABS:
        tops[tab] = leave_home(tab)
        check_back(tab, "Escape")
        assert page.evaluate("(s) => document.querySelector(s).scrollTop", HOME) == tops[tab], f"Escape from {tab} returns where Settings was"
    assert tops["storage"] > 0 and tops["updates"] > 0, f"the home was scrolled when its last rows were pressed: {tops}"
    for leave in ("Back", "system Back"):
        top = leave_home("storage")
        check_back("storage", leave)
        assert top > 0
        assert page.evaluate("(s) => document.querySelector(s).scrollTop", HOME) == top, f"{leave} returns where Settings was"
    # A tab reached any other way has no Back.
    _nav(page, "dashboard", width)
    assert page.locator("#page-dashboard .dashboard-back").count() == 0


def test_no_tab_scrolls_sideways_at_the_floor_and_a_three_way_choice_keeps_one_line(ui):
    page, _ = _open(ui, tab="consciousness", width=320, height=640)
    for tab in TABS:
        page.click(f'[data-dashboard-tab="{tab}"]')
        page.wait_for_selector(f"#dashboard-panel-{tab}.active")
        page.wait_for_timeout(200)
        assert _sideways(page) == [], tab
    page.click('[data-dashboard-tab="consciousness"]')
    tops = page.locator('#page-dashboard [data-setting-row="OUROBOROS_CONSCIOUSNESS_AUTONOMY"] [data-setting-value]').evaluate_all(
        "(buttons) => buttons.map((button) => Math.round(button.getBoundingClientRect().top))")
    assert len(tops) == 3 and len(set(tops)) == 1, f"Advises, Acts and Evolves stand on one line at 320 px: {tops}"
    # A number and a door keep their place at the row's end; a range goes under its words. "At the row's end" is
    # beside the words: right of the name and no lower than the last line under it. (The control is centred against
    # the name and its line of meaning, so beside three such lines a 34 px number starts a pixel under the name's own
    # line: its top against the name's bottom said nothing about where it stands.)
    ends = page.evaluate("""() => Object.fromEntries(['OUROBOROS_CONSCIOUSNESS_MAX_TASKS', 'consciousness-allowance', 'OUROBOROS_BG_WAKEUP'].map((id) => {
        const row = document.querySelector(`#page-dashboard [data-setting-row="${id}"]`);
        const box = (selector) => row.querySelector(selector).getBoundingClientRect();
        const [title, meta, actions] = [box('.ui-row-title'), box('.ui-row-meta'), box('.ui-row-actions')];
        return [id, actions.left >= title.right - 1 && actions.top < Math.max(title.bottom, meta.bottom)]; }))""")
    assert ends == {"OUROBOROS_CONSCIOUSNESS_MAX_TASKS": True, "consciousness-allowance": True, "OUROBOROS_BG_WAKEUP": False}, ends
