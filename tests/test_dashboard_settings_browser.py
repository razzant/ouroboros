"""The Dashboard tabs' own settings: Consciousness, Evolution, Updates and Storage each carry cards of rows that save
themselves (settings_controls.js), drawn once outside every region the tab repaints (dashboard.js mountTabSettings),
and Costs carries the subscriptions block from the passive quota read.

The real SPA in headless Chromium and WebKit over the shared fixture's endpoints, the settings endpoint answered here: on each
tab a field the owner is in keeps its text and caret through the tab's own periodic refresh and a settings read, and
leaving it shows the row's result line; the Updates channel keeps its focused choice through a status read; the
Evolution choice "Every N tasks" posts both keys in one body and says itself in its own sentence; the doors of
Consciousness and Costs lead where they say. The named shell capsule, Escape and the system's Back return each tab to Settings at every width, with the list's scroll preserved. At the 320 x 640 floor
each page's own Back reaches the list, and every tab and three-way choice fits. In two panes the open page's pane
is the same flex column, its panel the same box, whether the page stands beside the list on its own or was chosen.

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
from tests.test_segmented_layout_browser import _MEASURE as STRIP_MEASURE
from tests.test_subscription_setup_browser import subscription_ui as subscription_ui
from tests.ui_chat_viewport_smoke import wait_drawer_closed

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


def _open(ui, *, tab=None, width=1087, height=740, consciousness=None):
    """The app with the stored settings above; a save stores its body and answers the receipt. ``consciousness`` is
    what ``GET /api/consciousness`` answers where a test reads its facts."""
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
    if consciousness is not None:
        page.route("**/api/consciousness", lambda route: route.fulfill(
            content_type="application/json", body=json.dumps(consciousness)) if route.request.method == "GET" else route.fallback())
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
    # The Model door says the route the server publishes (an empty slot follows Main), never a guess from the slot.
    page, _ = _open(ui, tab="consciousness", consciousness={
        "model_route": {"value": "claudexor::codex=gpt-6-astra", "layer": "main", "local": False}})
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
    page.locator('#page-settings [data-settings-page="models/role/consciousness"]').wait_for(state="visible")
    back = page.locator('#page-settings > .app-page-header [data-shell-back]')
    back.wait_for()
    assert back.get_attribute("aria-label") == "Back to Dashboard › Consciousness" and back.get_attribute("title") == "Back to Dashboard › Consciousness"
    parent = page.locator('#page-settings [data-shell-level]')
    assert parent.count() == 0, 'the jump is the only back control until spent'
    assert page.locator('#page-settings [data-settings-lead-name]').inner_text() == "Consciousness model"
    back.click()
    page.wait_for_selector("#dashboard-panel-consciousness.active")
    model_row = page.locator(row("consciousness-model"))
    focus = model_row.evaluate("""el => ({focused: el === document.activeElement, ring: getComputedStyle(el).outlineStyle,
        input: el.dataset.returnFocus, active: document.activeElement?.outerHTML.slice(0,500)})""")
    assert focus['focused'] and focus['ring'] == 'none', focus
    model_row.click()
    page.locator('[data-settings-page="models/role/consciousness"]:visible').wait_for()
    page.keyboard.press("Escape")
    page.wait_for_selector("#dashboard-panel-consciousness.active")
    assert model_row.evaluate("""el => el === document.activeElement && el.dataset.returnFocus === 'keyboard'
        && getComputedStyle(el).outlineStyle === 'solid'""")
    # Spending a jump through the map keeps the child named and its own parent reachable.
    page.click(row("consciousness-model"))
    page.locator('[data-settings-page="models/role/consciousness"]:visible').wait_for()
    page.evaluate("document.querySelector('[data-nav-page=widgets]').click()")
    page.wait_for_selector('#page-widgets.active')
    page.evaluate("document.querySelector('[data-nav-page=settings]').click()")
    page.wait_for_selector('#page-settings.active')
    assert back.get_attribute("aria-label") == back.get_attribute("title") == "Back to Models" and parent.count() == 0
    back.click()
    page.locator('[data-settings-page="models"]:visible').wait_for()
    page.evaluate("document.querySelector('[data-nav-page=dashboard]').click()")
    page.wait_for_selector('#page-dashboard.active')
    # The four doors of "Around it" stand in the picture's order, each with its value before its chevron.
    around = page.locator('#page-dashboard [data-cx-settings] > section[aria-label="Around it"] [data-setting-door]')
    assert [" ".join(text.split()) for text in around.evaluate_all(
        "(doors) => doors.map((door) => [door.querySelector('.ui-row-title').textContent, door.querySelector('[data-row-value]')?.textContent ?? ''].join(' | '))")] == [
        "Model | gpt-6-astra (Main model)", f"Its actions are checked | {_scale_word(page, 'light')}", "Where it lives |", "Improving itself | Off"]
    # The Safety door opens its row's page in Settings, whose Back names this tab and returns to it.
    page.click(row("consciousness-safety"))
    page.locator('#page-settings [data-settings-page="permissions"]').wait_for(state="visible")
    back = page.locator('#page-settings > .app-page-header [data-shell-back]')
    back.wait_for()
    assert back.get_attribute("aria-label") == "Back to Dashboard › Consciousness" and back.get_attribute("title") == "Back to Dashboard › Consciousness"
    back.click()
    page.wait_for_selector("#dashboard-panel-consciousness.active")
    # The evolution door opens its row on the Evolution tab, and the allowance door its row in Costs: a door inside
    # the Dashboard leaves its named return in the same slot.
    page.click(row("consciousness-evolution"))
    page.wait_for_selector("#dashboard-panel-evolution.active")
    page.wait_for_function("() => document.activeElement?.dataset.evoWhen === 'off'")
    assert page.locator("#page-dashboard [data-shell-back]").get_attribute("aria-label") == "Back to Dashboard › Consciousness"
    page.click('[data-dashboard-tab="consciousness"]')
    page.click(row("consciousness-allowance"))
    page.wait_for_selector("#dashboard-panel-costs.active")
    page.wait_for_function("(id) => document.activeElement?.id === id", arg=f"setting-{ALLOWANCE}")
    assert page.locator("#page-dashboard [data-shell-back]").get_attribute("aria-label") == "Back to Dashboard › Consciousness"


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
    page.locator('#page-settings [data-settings-page="accounts/codex"]').wait_for(state="visible")
    back = page.locator('#page-settings > .app-page-header [data-shell-back]')
    back.wait_for()
    assert back.get_attribute("aria-label") == "Back to Dashboard › Costs" and back.get_attribute("title") == "Back to Dashboard › Costs"
    back.click()
    page.wait_for_selector("#dashboard-panel-costs.active")


def test_leaving_costs_by_one_columns_back_ends_its_pending_read_and_opening_it_again_reads_anew(ui):
    """On a phone Back from Costs shows the Dashboard's list and no tab or page event says Costs was left. The quota read
    Costs has out is released then, as when another tab is chosen, and opening Costs again reads anew."""
    page, _ = _open(ui, width=390, height=844)
    held = []
    page.route("**/api/claudexor/status?view=quota", lambda route: held.append(route))  # a read that stays out
    page.evaluate("""() => { const real = window.fetch; window.__quotaSignals = [];
        window.fetch = function (input, init) {
            if (String((input && input.url) || input).includes('view=quota')) window.__quotaSignals.push(init && init.signal);
            return real.apply(this, arguments); }; }""")
    read = lambda: page.evaluate("window.__quotaSignals.length")
    released = lambda at: page.evaluate("(at) => window.__quotaSignals[at].aborted", at)

    def until(check, what):
        for _ in range(100):
            if check():
                return
            page.wait_for_timeout(50)
        raise AssertionError(what)

    base = read()
    _nav(page, "dashboard", 390)
    page.click('[data-dashboard-tab="costs"]')
    page.wait_for_selector("#dashboard-panel-costs.active")
    until(lambda: read() == base + 1, "opening Costs reads once")
    assert not released(base)
    page.locator("#page-dashboard [data-shell-back]").click()
    assert page.locator("#page-dashboard").get_attribute("data-section-view") == "auto", "the list is back"
    until(lambda: released(base), "Costs is hidden under the list, so its read is released")
    page.click('[data-dashboard-tab="costs"]')
    # The page's fetch and the route's callback are two reports of one request: answer only once both have come, so the
    # callback cannot lag the read count and leave `held[-1]` the request that was released.
    until(lambda: read() == base + 2 and len(held) == read(), "opening Costs again reads anew and its request reaches the route")
    assert not released(base + 1)
    held[-1].fulfill(content_type="application/json", body=json.dumps(_quota()))
    page.locator('#page-costs .costs-subscriptions [data-costs-account="accounts/codex"]').wait_for()


# What Costs asks for, by the address and the way its page's fetch is called (`window.__reads` is filled where each is
# made: a read starts inside the handler that shows the page, so nothing is left to wait for before counting it). The
# totals, the 24 hours and the quota are Costs' alone; the limits and the allowance are told from the neighbours' by
# the signal only Costs gives them.
_READ = {
    "totals": "(url) => url.pathname === '/api/cost-breakdown' && url.search === ''",
    "period": "(url) => url.pathname === '/api/cost-breakdown' && url.search === '?period=24h'",
    "quota": "(url) => url.pathname === '/api/claudexor/status' && url.search === '?view=quota'",
    "limits": "(url, read) => url.pathname === '/api/settings' && read.method === 'GET' && read.signal !== null",
    "allowance": "(url, read) => url.pathname === '/api/consciousness' && read.signal !== null",
}


def _record_reads(page):
    page.evaluate("""() => { const real = window.fetch; window.__reads = [];
        window.fetch = function (input, init) {
            const url = String((input && input.url) || input);
            window.__reads.push({ url, method: (init && init.method) || 'GET', signal: (init && init.signal) || null });
            return real.apply(this, arguments); }; }""")


def _asked(page, name):
    return page.evaluate(f"() => window.__reads.filter((read) => ({_READ[name]})(new URL(read.url, location.href), read)).length")


def _ended(page, name):
    return page.evaluate(f"""() => window.__reads.filter((read) => ({_READ[name]})(new URL(read.url, location.href), read))
        .map((read) => read.signal.aborted)""")


def test_one_columns_arrow_keys_select_costs_without_any_of_its_reads_and_the_press_that_opens_it_reads_each_once(ui):
    """On a phone the Dashboard's list hides every page: its arrow keys move the selection onto Costs and open nothing. The
    sheets say the page has no box, so none of its reads (the totals, the 24 hours, the quota) starts; the press that opens
    the page starts each once."""
    from playwright.sync_api import expect

    page, _ = _open(ui, width=390, height=844)
    _record_reads(page)
    _nav(page, "dashboard", 390)
    # The drawer is still leaving under the place it handed over, and the room stays inert until its close motion ends
    # (app.js): a focus now lands nowhere. Wait for the state the shell settles into (the drawer closed, the list live),
    # then say where the keys start: the list's Logs row holds the focus before the first arrow.
    wait_drawer_closed(page)
    page.wait_for_function("() => { const row = document.querySelector('[data-dashboard-tab=\"logs\"]'); return Boolean(row) && !row.closest('[inert]'); }")
    page.focus('[data-dashboard-tab="logs"]')
    assert page.evaluate("() => document.activeElement === document.querySelector('[data-dashboard-tab=\"logs\"]')"), \
        "the arrow keys start from the list's Logs row"
    for _ in range(3):
        page.keyboard.press("ArrowDown")
    page.wait_for_function("() => document.querySelector('#dashboard-tab-costs').getAttribute('aria-selected') === 'true'")
    assert page.locator("#page-dashboard").get_attribute("data-section-view") == "auto", "the list is still the page"
    expect(page.locator("#page-costs")).not_to_be_visible()
    assert [_asked(page, name) for name in ("totals", "period", "quota")] == [0, 0, 0], "a page that is not drawn reads nothing"
    page.click('[data-dashboard-tab="costs"]')
    expect(page.locator("#page-costs")).to_be_visible()
    for name in ("totals", "period", "quota"):
        page.wait_for_function(f"() => window.__reads.filter((read) => ({_READ[name]})(new URL(read.url, location.href), read)).length >= 1")
    assert [_asked(page, name) for name in ("totals", "period", "quota")] == [1, 1, 1], "the press that opens the page reads each once"


def test_costs_back_ends_every_read_it_has_out_not_the_quota_alone_and_opening_it_again_reads_anew(ui):
    """The companion of the quota case above for the whole page: Back from Costs on a phone ends the totals, the 24 hours,
    the quota, the limits and the allowance it has out in one stroke. They are held, so that an answer could still come;
    none can, for each transport is ended. Opening Costs again reads each anew with a transport that is not."""
    page, _ = _open(ui, width=390, height=844)
    held = []
    for pattern in ("**/api/cost-breakdown*", "**/api/claudexor/status?view=quota", "**/api/consciousness"):
        page.route(pattern, lambda route: held.append(route))
    page.route("**/api/settings", lambda route: held.append(route) if route.request.method == "GET" else route.fallback())
    _record_reads(page)
    names = ("totals", "period", "quota", "limits", "allowance")
    _nav(page, "dashboard", 390)
    page.click('[data-dashboard-tab="costs"]')
    page.wait_for_selector("#dashboard-panel-costs.active")
    for name in names:
        page.wait_for_function(f"() => window.__reads.filter((read) => ({_READ[name]})(new URL(read.url, location.href), read)).length >= 1")
    assert [_ended(page, name)[-1] for name in names] == [False] * 5, "opened, the page has all five out, each with a transport to end"
    page.locator("#page-dashboard [data-shell-back]").click()
    assert page.locator("#page-dashboard").get_attribute("data-section-view") == "auto", "the list is back"
    for name in names:
        page.wait_for_function(f"() => window.__reads.filter((read) => ({_READ[name]})(new URL(read.url, location.href), read)).at(-1).signal.aborted")
    first = {name: _asked(page, name) for name in names}
    page.click('[data-dashboard-tab="costs"]')
    for name in names:
        page.wait_for_function(f"(was) => window.__reads.filter((read) => ({_READ[name]})(new URL(read.url, location.href), read)).length === was + 1", arg=first[name])
    assert [_ended(page, name)[-1] for name in names] == [False] * 5, "opened again, each reads anew and none of the new transports is ended"


@pytest.mark.parametrize("width,height", [(1087, 740), (320, 640)])
def test_a_tab_opened_from_settings_leads_back_to_settings_where_it_was(ui, width, height):
    page, _ = _open(ui, width=width, height=height)
    open_settings(page)
    assert page.evaluate("(s) => { const el = document.querySelector(s); return el.scrollHeight > el.clientHeight + 100; }", HOME), \
        "the home scrolls at this size"

    # Two panes: whether the home row naming the page open beside it is in the list's view (None in one column).
    named_shown = """(s) => { const box = document.querySelector(s), r = box.getBoundingClientRect();
        const row = box.closest('[data-section-view="page"]') && box.querySelector('[aria-current="page"]');
        const seen = (n) => { const b = n.getBoundingClientRect(); return b.top >= r.top - 1 && b.bottom <= r.bottom + 1; };
        return row ? seen(row) : null; }"""

    def leave_home(tab):
        """Press the home row of a tab where it stands (no scroll of the test's own) and say where the home was."""
        row = page.locator(f'{HOME} [data-settings-open="dashboard:{tab}"]')
        row.evaluate("(el) => el.scrollIntoView({ block: 'center' })")
        page.wait_for_timeout(120)
        top = page.evaluate("(s) => document.querySelector(s).scrollTop", HOME)
        named[tab] = page.evaluate(named_shown, HOME)
        row.evaluate("(el) => el.click()")
        page.wait_for_selector(f"#dashboard-panel-{tab}.active")
        return top

    def returned_where_it_was(tab, top, leave):
        """The home is where it was. A row naming the open page beside it that was out of view comes in when it can
        share the view with the pressed row (owner's critic, 16.11: a return does not leave the open page unnamed);
        the pressed row keeps its view either way."""
        now = page.evaluate("(s) => document.querySelector(s).scrollTop", HOME)
        assert page.evaluate("""(s) => { const r = document.querySelector(s).getBoundingClientRect(),
            b = document.activeElement.getBoundingClientRect(); return b.top >= r.top - 1 && b.bottom <= r.bottom + 1; }""", HOME), (tab, leave)
        if named[tab] is False and now != top:
            assert page.evaluate(named_shown, HOME) is True, f"{leave} from {tab} moved the home only to name the open page"
            revealed.add(tab)
        else:
            assert now == top, f"{leave} from {tab} returns where Settings was"

    def check_back(tab, leave):
        back = page.locator("#page-dashboard > .app-page-header [data-shell-back]")
        assert page.locator(f"#dashboard-panel-{tab} > .dashboard-back").count() == 0
        assert _sideways(page) == [], (tab, leave)
        back.wait_for(state="visible")
        assert back.get_attribute("aria-label").startswith("Back to Settings")
        assert back.get_attribute("title") == back.get_attribute("aria-label")
        assert page.locator("#page-dashboard [data-shell-back]").count() == 1
        assert page.evaluate("() => Boolean(window.__ouroLayers.returnPage())")
        if leave == "Escape":
            page.keyboard.press("Escape")
        elif leave == "Back":
            back.click()
        else:
            page.go_back()
        page.wait_for_selector("#page-settings.active")
        assert back.count() == 0, leave
        focus = page.evaluate("""() => ({row: document.activeElement?.dataset.settingsOpen,
            ring: getComputedStyle(document.activeElement).outlineStyle,
            input: document.activeElement?.dataset.returnFocus, html: document.activeElement?.outerHTML.slice(0, 500)})""")
        assert focus['row'] == f'dashboard:{tab}', focus
        assert focus['input'] == ('keyboard' if leave == 'Escape' else 'pointer'), focus
        assert (focus['ring'] != 'none') == (leave == 'Escape'), focus

    tops, named, revealed = {}, {}, set()
    for tab in TABS:
        tops[tab] = leave_home(tab)
        check_back(tab, "Escape")
        returned_where_it_was(tab, tops[tab], "Escape")
    assert tops["storage"] > 0 and tops["updates"] > 0, f"the home was scrolled when its last rows were pressed: {tops}"
    assert (width < 640) == all(value is None for value in named.values()), named
    assert (width < 640) == (not revealed), f"two panes bring the naming row in for a nearby pressed row: {revealed}, {named}"
    for leave in ("Back", "system Back"):
        top = leave_home("storage")
        check_back("storage", leave)
        assert top > 0
        returned_where_it_was("storage", top, leave)
    # The primary map clears returns and opens the section list.
    _nav(page, "dashboard", width)
    assert page.locator("#page-dashboard [data-shell-back]").count() == 0


# The open page's pane and its shown panel: how the pane lays the panel out, and the panel's box.
PANE = """() => { const place = document.querySelector('#page-dashboard');
    const pane = place.querySelector('.dashboard-panels'), panel = pane.querySelector('.dashboard-panel.active');
    const style = getComputedStyle(pane), r = panel.getBoundingClientRect();
    return { view: place.dataset.sectionView, tab: panel.dataset.dashboardPanel, display: style.display,
        direction: style.flexDirection, panel: [r.left, r.top, r.width, r.height] }; }"""


def test_a_wide_dashboard_lays_out_its_page_alike_whether_it_stands_beside_the_list_on_its_own_or_was_chosen(ui):
    # Two panes show an automatic view's pane through ui.css's shared rule, which draws a pane as a grid; the
    # Dashboard's pane stays the flex column its panels fill (style.css), so the shown panel keeps one box in both views.
    page, _ = _open(ui, width=1087, height=740)
    _nav(page, "dashboard", 1087)
    page.wait_for_selector("#page-dashboard .dashboard-panel.active")
    settle = "() => new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done)))"
    page.evaluate(settle)
    automatic = page.evaluate(PANE)
    page.click(f'[data-dashboard-tab="{automatic["tab"]}"]')  # the same row: now a chosen page
    page.wait_for_function("() => document.querySelector('#page-dashboard').dataset.sectionView === 'page'")
    page.evaluate(settle)
    chosen = page.evaluate(PANE)
    assert (automatic["view"], chosen["view"], chosen["tab"]) == ("auto", "page", automatic["tab"]), (automatic, chosen)
    assert [automatic["display"], automatic["direction"]] == [chosen["display"], chosen["direction"]] == ["flex", "column"], \
        (automatic, chosen)
    assert automatic["panel"] == chosen["panel"], (automatic, chosen)


def test_no_tab_scrolls_sideways_at_the_floor_and_a_three_way_choice_keeps_one_line(ui):
    page, _ = _open(ui, tab="consciousness", width=320, height=640)
    for tab in TABS:
        if not page.locator(f'[data-dashboard-tab="{tab}"]').is_visible():
            page.locator('#page-dashboard [data-shell-back]').click()
        page.click(f'[data-dashboard-tab="{tab}"]')
        page.wait_for_selector(f"#dashboard-panel-{tab}.active")
        page.wait_for_timeout(200)
        assert _sideways(page) == [], tab
    page.locator('#page-dashboard [data-shell-back]').click()
    page.click('[data-dashboard-tab="consciousness"]')
    # DESIGN:432–437: one line when its words fit; otherwise the strip wraps inside its row, never sideways.
    strip = page.evaluate(STRIP_MEASURE, ['OUROBOROS_CONSCIOUSNESS_AUTONOMY',
        '#page-dashboard [data-setting-row="OUROBOROS_CONSCIOUSNESS_AUTONOMY"] .ui-segment-group'])
    assert [button['label'] for button in strip['buttons']] == ['Advises', 'Acts', 'Evolves']
    if strip['needed'] <= strip['row']['right'] - strip['row']['left'] + 0.5:
        assert len({round(button['top']) for button in strip['buttons']}) == 1, strip
    assert all(button['lines'] == 1 and button['left'] >= strip['left']
               and button['right'] <= strip['right'] + 0.5 for button in strip['buttons']), strip
    assert not strip['spills'] and strip['left'] >= strip['row']['left'] - 0.5 and strip['right'] <= strip['row']['right'] + 0.5, strip
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


@pytest.mark.parametrize("width,height", [(1087, 740), (390, 844)])
def test_a_door_shares_its_row_with_a_long_value_and_keeps_its_chevron_beside_it(ui, width, height):
    page, _ = _open(ui, width=width, height=height)
    route = {"value": "anthropic/claude-opus-5.5", "layer": "main", "local": False}
    page.route("**/api/consciousness*", lambda served: served.fulfill(
        content_type="application/json", body=json.dumps({"model_route": route})))
    _nav(page, "dashboard", width)
    page.click('[data-dashboard-tab="consciousness"]')
    model_row = '#page-dashboard [data-setting-row="consciousness-model"]'
    page.wait_for_function("(s) => document.querySelector(s)?.querySelector('[data-row-value]')?.textContent === '$20'",
                           arg='#page-dashboard [data-setting-row="consciousness-allowance"]')

    def measure(expected):
        page.wait_for_function("([s, words]) => document.querySelector(s)?.querySelector('[data-row-value]')?.textContent === words",
                               arg=[model_row, expected])
        return page.evaluate("""() => {
            const ids = ['consciousness-allowance', 'consciousness-model', 'consciousness-safety', 'consciousness-places', 'consciousness-evolution'];
            const lines = (el) => el ? Math.round(el.getBoundingClientRect().height / parseFloat(getComputedStyle(el).lineHeight)) : 0;
            return Object.fromEntries(ids.map((id) => {
                const row = document.querySelector(`#page-dashboard [data-setting-row="${id}"]`);
                const r = row.getBoundingClientRect();
                const [title, meta, actions, value, chevron] = ['.ui-row-title', '.ui-row-meta', '.ui-row-actions', '[data-row-value]', '.settings-row-chevron']
                    .map((s) => row.querySelector(s));
                const [t, m, a, c] = [title, meta, actions, chevron].map((el) => el.getBoundingClientRect());
                const v = value && value.getClientRects().length ? value.getBoundingClientRect() : null;
                return [id, {
                    rowWidth: r.width,
                    titleLines: lines(title),
                    valueWidth: v ? v.width : 0,
                    valueLines: v ? lines(value) : 0,
                    valueAlign: value ? getComputedStyle(value).textAlign : null,
                    actionsBeside: a.left >= t.right - 1 && a.top < Math.max(t.bottom, m.bottom),
                    chevronBeside: !v || (c.left >= v.right - 1 && c.top < v.bottom - 0.5),
                    chevronEndGap: r.right - c.right,
                }];
            }));
        }""")

    medium = measure("anthropic/claude-opus-5.5 (Main model)")
    assert all(d["actionsBeside"] and d["chevronBeside"] and d["chevronEndGap"] <= 16 for d in medium.values()), medium
    assert medium["consciousness-model"]["titleLines"] == 1
    assert medium["consciousness-model"]["valueLines"] <= (1 if width >= 1000 else 2), medium["consciousness-model"]

    # A known provider kind is stripped from the shown route; an unknown head is one OpenRouter id shown whole (#3943).
    route["value"] = "openrouter::meta-llama/llama-4-maverick-17b-128e-instruct"
    with page.expect_response(_answers("/api/consciousness")) as refreshed:
        page.clock.fast_forward(21_000)
    _applied(page, refreshed)
    long_route = measure("meta-llama/llama-4-maverick-17b-128e-instruct (Main model)")
    assert all(d["actionsBeside"] and d["chevronBeside"] and d["chevronEndGap"] <= 16 for d in long_route.values()), long_route
    model = long_route["consciousness-model"]
    assert model["titleLines"] == 1 and model["valueAlign"] == "end", model
    assert model["valueWidth"] > model["rowWidth"] * 0.55 and model["valueLines"] <= 3, model
    assert _sideways(page) == []
