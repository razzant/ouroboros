"""The Costs budget save is feedback about an action just taken (DESIGN §4 "Status and chips").

The real SPA and the shared HTTP client in Chromium and WebKit, the Dashboard's Costs tab, with the settings endpoint
answered per scenario and time moved by the browser clock: a plain acknowledgement ends by itself after its time, while
a restart still required (by this save or an earlier one), a warning and an error stay until the next save replaces
them, and an older acknowledgement never erases what a newer save says or its tone (a validation message included).
Light and dark, phone and window.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.test_subscription_setup_browser import subscription_ui as subscription_ui

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
ACK_MS = 4000


@pytest.fixture(params=["chromium", "webkit"])
def costs_ui(request, monkeypatch):
    monkeypatch.setenv("OUROBOROS_UI_BROWSER_ENGINE", request.param)
    ui = request.getfixturevalue("subscription_ui")
    ui["engine"] = request.param
    return ui


def _open_costs(ui, *, width, theme, answers):
    """The Costs tab with a settings endpoint that answers each save from ``answers`` in order."""
    page = ui["new_page"](viewport={"width": width, "height": 844}, color_scheme=theme)
    saves = []

    def settings(route):
        request = route.request
        if request.method != "POST":
            route.fallback()
            return
        saves.append(request.post_data_json)
        status, body = answers.pop(0)
        route.fulfill(status=status, content_type="application/json", body=json.dumps(body))

    page.route("**/api/settings", settings)
    page.add_init_script(f"localStorage.setItem('ouroboros.theme', {json.dumps(theme)})")
    page.clock.install()
    page.goto(ui["url"] + "/")
    page.wait_for_selector("#page-chat")
    if width < 720:
        page.locator(".page.active [data-mobile-nav-toggle]:visible").click()
    page.click('[data-nav-page="dashboard"]')
    page.click('[data-dashboard-tab="costs"]')
    page.wait_for_selector("#btn-save-budget", state="visible")
    return page, saves


def _words(page):
    return page.evaluate("""() => { const el = document.getElementById('budget-save-status');
        return [el.textContent, el.dataset.tone || '']; }""")


def _save(page, *, budget="200", per_task="50"):
    page.fill("#s-budget", budget)
    page.fill("#s-per-task-cost", per_task)
    page.click("#btn-save-budget")
    page.wait_for_function("!document.getElementById('btn-save-budget').disabled")


def _shot(page, ui, name):
    root = os.environ.get("OUROBOROS_UI_EVIDENCE_DIR")
    if root:
        Path(root).mkdir(parents=True, exist_ok=True)
        page.locator(".costs-budget-card").screenshot(path=str(Path(root) / f"costs-{name}-{ui['engine']}.png"))


@pytest.mark.parametrize("theme,width", [("dark", 1100), ("light", 390)])
def test_plain_acknowledgement_ends_while_restart_warning_and_error_stay(costs_ui, theme, width):
    answers = [(200, {"ok": True, "saved": True, "immediate_changed": True}),
               (200, {"ok": True, "saved": True, "restart_required": True}),
               (200, {"ok": True, "saved": True, "no_changes": True, "restart_required": True}),
               (200, {"ok": True, "saved": True, "immediate_changed": True, "warnings": ["Limit is below known spend."]}),
               (500, {"error": "Settings file is read-only."})]
    page, saves = _open_costs(costs_ui, width=width, theme=theme, answers=answers)
    tag = f"{width}-{theme}"
    # A plain acknowledgement of what the owner just did: there, then gone after its time, with its tone.
    _save(page)
    assert _words(page) == ["Saved. Took effect immediately.", "ok"]
    page.clock.fast_forward(ACK_MS - 200)
    assert _words(page)[0] == "Saved. Took effect immediately.", "still there before its time"
    page.clock.fast_forward(400)
    assert _words(page) == ["", ""], "it ended by itself"
    # A restart the save requires is a standing condition, not an acknowledgement: no time ends it.
    _save(page)
    assert _words(page) == ["Saved. Restart required.", "warn"]
    _shot(page, costs_ui, f"restart-required-{tag}")
    page.clock.fast_forward(ACK_MS + 500)  # past any acknowledgement time, short of the socket watchdog's 45 s
    assert _words(page) == ["Saved. Restart required.", "warn"]
    # The same values again change nothing, and the restart is still owed: the line says so, and stays.
    _save(page)
    page.clock.fast_forward(ACK_MS + 500)  # past any acknowledgement time, short of the socket watchdog's 45 s
    assert _words(page) == ["No changes. Restart still required.", "warn"]
    _shot(page, costs_ui, f"no-changes-restart-{tag}")
    # A warning stays too, and the next save replaces it.
    _save(page)
    page.clock.fast_forward(ACK_MS + 500)  # past any acknowledgement time, short of the socket watchdog's 45 s
    assert _words(page) == ["Saved. Took effect immediately. Warnings: Limit is below known spend.", "warn"]
    # An error is never an acknowledgement.
    _save(page)
    page.clock.fast_forward(ACK_MS + 500)  # past any acknowledgement time, short of the socket watchdog's 45 s
    assert _words(page) == ["Error: Settings file is read-only.", "error"]
    _shot(page, costs_ui, f"error-{tag}")
    assert len(saves) == 5 and all(save == {"TOTAL_BUDGET": 200, "OUROBOROS_PER_TASK_COST_USD": 50} for save in saves), saves
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    assert page.evaluate("""() => { const c = document.querySelector('.costs-budget-card');
        return c.scrollWidth <= c.clientWidth + 1; }""")


def test_an_older_acknowledgement_never_erases_a_newer_save_or_a_validation_message(costs_ui):
    answers = [(200, {"ok": True, "saved": True, "immediate_changed": True}),
               (503, {"error": "Offline."}),
               (200, {"ok": True, "saved": True, "next_task_changed": True})]
    page, saves = _open_costs(costs_ui, width=1100, theme="dark", answers=answers)
    _save(page)
    page.clock.fast_forward(ACK_MS - 1000)  # the first acknowledgement is one second from ending
    _save(page)                              # a failing save inside that second
    page.clock.fast_forward(ACK_MS)          # past the first timer's deadline: the error is still the answer
    assert _words(page) == ["Error: Offline.", "error"]
    # An invalid value after an acknowledgement is answered at once and is not erased by that acknowledgement's timer.
    _save(page)
    assert _words(page) == ["Saved. Applies on the next task.", "ok"]
    page.clock.fast_forward(ACK_MS - 500)
    page.fill("#s-budget", "0")
    page.click("#btn-save-budget")
    page.clock.fast_forward(ACK_MS)
    assert _words(page) == ["Budget values must be at least 0.01.", "error"], "it outlives the old timer, never in its tone"
    assert len(saves) == 3, "an invalid value is never sent"
