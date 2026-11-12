"""The Costs limits save themselves: each is a setting row (settings_controls.js; DESIGN §4 "Status and chips").

The real SPA and the shared HTTP client in Chromium and WebKit, the Dashboard's Costs tab, with the settings endpoint
answered per scenario: typing a limit sends nothing, leaving its field (Tab or Enter) posts exactly that key, and the
row's result line, in its own meta line beside the field, says what happened. No timer ends it (the browser clock moves
past any acknowledgement time). A refusal keeps the typed text and says the server's sentence; a value below the setup
contract's floor is refused in the row and never sent. Undo re-posts the old value with its token. A refresh of the tab
writes a fresh value only into a field the owner is not in. Light and dark, phone and window, and the 320 x 640 floor
without sideways scrolling.

LANE A: the save's `changes` receipt and `_meta.effects` are what the server answers once lane A lands; until then the
route below answers them. The assertions that need them ("was 200", Undo and its token) are marked LANE A; today's
server answers without a receipt, which a row says as "Saved" (checked too).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.test_subscription_setup_browser import subscription_ui as subscription_ui

pytestmark = [pytest.mark.ui_browser]
TOTAL, PER_TASK, ALLOWANCE = "TOTAL_BUDGET", "OUROBOROS_PER_TASK_COST_USD", "OUROBOROS_CONSCIOUSNESS_DAILY_USD"
ROUNDS, LIFETIME = "OUROBOROS_MAX_ROUNDS", "OUROBOROS_TASK_ABS_CEILING_SEC"
# LANE A: when each key's change applies, as GET /api/settings `_meta.effects` says it.
EFFECTS = {TOTAL: "now", PER_TASK: "next_task", ALLOWANCE: "next_wake", ROUNDS: "next_task", LIFETIME: "next_task"}


@pytest.fixture(params=["chromium", "webkit"])
def costs_ui(request, monkeypatch):
    monkeypatch.setenv("OUROBOROS_UI_BROWSER_ENGINE", request.param)
    ui = request.getfixturevalue("subscription_ui")
    ui["engine"] = request.param
    return ui


def _open_costs(ui, *, width, theme, answers, height=844):
    """The Costs tab with a settings endpoint that answers each save from ``answers`` in order; a saved value is
    stored, so a later read shows it."""
    # This fixture's route serves a sparse settings dict; the real endpoint includes effective defaults.
    ui["settings"].update({TOTAL: 200, PER_TASK: 50, ALLOWANCE: 20, ROUNDS: "unlimited", LIFETIME: 3600})
    ui["settings"]["_meta"] = {**ui["settings"]["_meta"], "effects": EFFECTS}
    page = ui["new_page"](viewport={"width": width, "height": height}, color_scheme=theme)
    saves = []

    def settings(route):
        request = route.request
        if request.method != "POST":
            route.fallback()
            return
        body = request.post_data_json
        saves.append(body)
        status, answer = answers.pop(0)
        if status < 400:
            ui["settings"].update({key: value for key, value in body.items() if key != "if_unchanged"})
        route.fulfill(status=status, content_type="application/json", body=json.dumps(answer))

    page.route("**/api/settings", settings)
    page.add_init_script(f"localStorage.setItem('ouroboros.theme', {json.dumps(theme)})")
    page.clock.install()
    page.goto(ui["url"] + "/")
    page.wait_for_selector("#page-chat")
    if width < 720:
        page.locator(".page.active [data-mobile-nav-toggle]:visible").click()
    page.click('[data-nav-page="dashboard"]')
    page.click('[data-dashboard-tab="costs"]')
    page.wait_for_function(f"document.getElementById('setting-{TOTAL}').value === '200'")
    page.wait_for_function("document.getElementById('costs-budget-help').textContent.includes('confirmed and estimated prices count')")
    return page, saves


def _line(page, key):
    """A row's result: its words (Undo included, the raw-text disclosure left out) and its tone."""
    return page.evaluate("""(key) => { const el = document.getElementById(`setting-${key}-result`);
        const said = [...el.childNodes].filter((node) => !node.classList?.contains('ui-result-details'))
            .map((node) => node.textContent).join('');
        return [said, el.dataset.tone || '']; }""", key)


def _says(page, key, words, tone):
    page.wait_for_function("([key, words]) => document.getElementById(`setting-${key}-result`).textContent.startsWith(words)",
                           arg=[key, words])
    assert _line(page, key) == [words, tone]


def _type(page, key, text, *, leave="Tab"):
    page.fill(f"#setting-{key}", text)
    page.press(f"#setting-{key}", leave)


def _beside_field(page, key):
    """The result is the row's own: in its meta line, in the row that holds the field, never under the card."""
    return page.evaluate("""(key) => { const result = document.getElementById(`setting-${key}-result`);
        const row = result.closest('[data-setting-row]');
        return row.dataset.settingRow === key && row.contains(document.getElementById(`setting-${key}`))
            && result.parentElement.classList.contains('ui-row-meta') && result.getBoundingClientRect().height > 0; }""", key)


def _fits(page):
    return page.evaluate("""() => { const card = document.querySelector('.costs-budget');
        const box = card.getBoundingClientRect();
        const inside = [...card.querySelectorAll('[data-setting-row], [data-setting-control], [data-setting-no-limit], [data-setting-result]')]
            .filter((el) => el.getClientRects().length)
            .every((el) => { const r = el.getBoundingClientRect(); return r.left >= box.left - 1 && r.right <= box.right + 1; });
        return { page: document.documentElement.scrollWidth <= innerWidth, card: card.scrollWidth <= card.clientWidth + 1, inside }; }""")


def _shot(page, ui, name):
    root = os.environ.get("OUROBOROS_UI_EVIDENCE_DIR")
    if root:
        Path(root).mkdir(parents=True, exist_ok=True)
        page.locator(".costs-budget").screenshot(path=str(Path(root) / f"costs-{name}-{ui['engine']}.png"))


@pytest.mark.parametrize("theme,width", [("dark", 1100), ("light", 390)])
def test_leaving_a_limit_saves_it_and_its_row_says_what_happened(costs_ui, theme, width):
    answers = [(200, {"status": "saved", "changes": {TOTAL: {"old": 200, "new": 250, "token": "b@1"}}}),  # LANE A
               (200, {"status": "saved", "changes": {TOTAL: {"old": 250, "new": 200, "token": "b@2"}}}),  # LANE A
               (200, {"status": "saved", "next_task_changed": True}),  # today's server: no receipt
               (400, {"error": "OUROBOROS_MAX_ROUNDS must be a positive integer or 'unlimited'.", "saved": False})]
    page, saves = _open_costs(costs_ui, width=width, theme=theme, answers=answers)
    tag = f"{width}-{theme}"
    page.fill(f"#setting-{TOTAL}", "250")
    assert saves == [], "typing saves nothing"
    page.press(f"#setting-{TOTAL}", "Tab")
    _says(page, TOTAL, "Saved · was 200 · Undo", "ok")  # LANE A: the receipt's old value and token
    assert saves == [{TOTAL: 250}], saves
    assert _beside_field(page, TOTAL), "the result is in the row's own meta line"
    page.clock.fast_forward(10_000)
    assert _line(page, TOTAL) == ["Saved · was 200 · Undo", "ok"], "no timer ends a result"
    _shot(page, costs_ui, f"saved-{tag}")
    # Undo re-posts the old value with the installed value's token (LANE A).
    page.click(f"#setting-{TOTAL}-result [data-setting-undo]")
    _says(page, TOTAL, "Saved · was 250 · Undo", "ok")
    assert saves[-1] == {TOTAL: 200, "if_unchanged": {TOTAL: "b@1"}}, saves
    assert page.input_value(f"#setting-{TOTAL}") == "200"
    # Without a receipt (today's server) a save says only "Saved"; Enter leaves the field as Tab does.
    _type(page, PER_TASK, "51", leave="Enter")
    _says(page, PER_TASK, "Saved", "ok")
    assert saves[-1] == {PER_TASK: 51}, saves
    # Below the setup contract's floor: refused in the row, never sent, the typed text kept.
    _type(page, TOTAL, "0")
    _says(page, TOTAL, "Not saved. Enter a number, 0.01 or more.", "danger")
    assert len(saves) == 3, saves
    assert page.input_value(f"#setting-{TOTAL}") == "0"
    # The server refuses a bound it does not take: its own sentence, and the text stays.
    page.locator(".costs-budget summary", has_text="More limits").click()
    _type(page, ROUNDS, "0")
    _says(page, ROUNDS, "Not saved. OUROBOROS_MAX_ROUNDS must be a positive integer or 'unlimited'.", "danger")
    assert saves[-1] == {ROUNDS: 0}, saves
    assert page.input_value(f"#setting-{ROUNDS}") == "0"
    page.clock.fast_forward(10_000)
    assert _line(page, ROUNDS)[1] == "danger", "a refusal stays"
    _shot(page, costs_ui, f"refused-{tag}")
    # An edit of the row is a new subject: its last result goes.
    page.fill(f"#setting-{TOTAL}", "20")
    assert _line(page, TOTAL) == ["", ""]
    assert _fits(page) == {"page": True, "card": True, "inside": True}
    assert len(saves) == 4 and not answers


def test_a_refresh_never_rewrites_the_field_the_owner_is_in(costs_ui):
    page, saves = _open_costs(costs_ui, width=1100, theme="light", answers=[(200, {"status": "saved"})])
    refresh = "window.dispatchEvent(new CustomEvent('ouro:settings-updated', { detail: { source: 'settings' } }))"
    page.fill(f"#setting-{PER_TASK}", "")
    page.keyboard.type("75")
    costs_ui["settings"].update({TOTAL: 400, PER_TASK: 60, ALLOWANCE: 30})
    page.evaluate(refresh)  # what Settings' own save says to the other pages
    page.wait_for_function(f"document.getElementById('setting-{TOTAL}').value === '400'")
    assert page.input_value(f"#setting-{PER_TASK}") == "75", "the field the owner is typing in keeps its text"
    assert page.evaluate("document.activeElement.id") == f"setting-{PER_TASK}"
    assert page.input_value(f"#setting-{ALLOWANCE}") == "30"
    page.press(f"#setting-{PER_TASK}", "Tab")
    _says(page, PER_TASK, "Saved", "ok")
    assert saves == [{PER_TASK: 75}], saves
    # A field the owner is in but has not typed into is left alone too; once left, it takes the fresh value.
    page.focus(f"#setting-{ALLOWANCE}")
    costs_ui["settings"].update({ALLOWANCE: 35, TOTAL: 410})
    page.evaluate(refresh)
    page.wait_for_function(f"document.getElementById('setting-{TOTAL}').value === '410'")
    assert page.input_value(f"#setting-{ALLOWANCE}") == "30"
    page.focus(f"#setting-{TOTAL}")
    page.evaluate(refresh)
    page.wait_for_function(f"document.getElementById('setting-{ALLOWANCE}').value === '35'")
    assert saves == [{PER_TASK: 75}], "nothing was sent by a refresh"


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_the_limits_card_fits_a_320_by_640_screen(costs_ui, theme):
    answers = [(400, {"error": "OUROBOROS_TASK_ABS_CEILING_SEC must be a positive integer or 'unlimited'.", "saved": False})]
    page, saves = _open_costs(costs_ui, width=320, height=640, theme=theme, answers=answers)
    page.locator(".costs-budget summary", has_text="More limits").click()
    _type(page, LIFETIME, "0")
    _says(page, LIFETIME, "Not saved. OUROBOROS_TASK_ABS_CEILING_SEC must be a positive integer or 'unlimited'.", "danger")
    _type(page, TOTAL, "abc")
    _says(page, TOTAL, "Not saved. Enter a number, 0.01 or more.", "danger")
    assert saves == [{LIFETIME: 0}], saves
    assert _fits(page) == {"page": True, "card": True, "inside": True}
    _shot(page, costs_ui, f"320-{theme}")
