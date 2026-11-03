"""The hierarchy of the secondary pages, measured on the real SPA the way the owner sees it (finding #9).

Chromium and WebKit, light and dark, a phone and a window, over fixture answers on the shapes the modules read:

* Costs names the known spend first and keeps what leaves it incomplete in the same card; the composition, the open
  hold and the technical facts step down in size and come after it, then the limit, then the tables. An unavailable
  ledger says so and invents no zero.
* Activity rows name their work whole (no ellipsis), in the primary ink above their quieter facts, say how long a task
  has run in words, and never put a control over the words.
* Storage's saved groups keep a group's name its room on a narrow card: the long size that carries a qualifier goes
  under the name.
* The Widgets card head keeps the menu and the move handle visible at rest, quiet, the same box and side by side on
  every card, and a framed card's lifecycle controls never push the pair below them.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlparse

import pytest

from tests.test_subscription_setup_browser import subscription_ui as subscription_ui

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


@pytest.fixture(params=["chromium", "webkit"])
def ui(request, monkeypatch):
    monkeypatch.setenv("OUROBOROS_UI_BROWSER_ENGINE", request.param)
    ui = request.getfixturevalue("subscription_ui")
    ui["engine"] = request.param
    return ui


def _json(body, status=200, *, path=None):
    """A route handler answering one path (any path when None) with ``body``; other requests fall through."""
    def handler(route):
        if path is not None and urlparse(route.request.url).path != path:
            route.fallback()
            return
        route.fulfill(status=status, content_type="application/json", body=json.dumps(body))
    return handler


def _open(ui, *, width, theme, place, tab=None, routes=()):
    page = ui["new_page"](viewport={"width": width, "height": 844}, color_scheme=theme)
    for pattern, handler in routes:
        page.route(pattern, handler)
    page.add_init_script(f"localStorage.setItem('ouroboros.theme', {json.dumps(theme)})")
    page.goto(ui["url"] + "/")
    page.wait_for_selector("#page-chat")
    if width < 720:
        page.locator(".page.active [data-mobile-nav-toggle]:visible").click()
    page.click(f'[data-nav-page="{place}"]')
    if tab:
        page.click(f'[data-dashboard-tab="{tab}"]')
    return page


def _shot(page, ui, name, width, theme):
    root = os.environ.get("OUROBOROS_UI_EVIDENCE_DIR")
    if root:
        Path(root).mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(Path(root) / f"{name}-{width}-{theme}-{ui['engine']}.png"))


# ---------------------------------------------------------------------------------------------------------- Costs
def _bucket(cost, calls, **extra):
    return {"cost": cost, "calls": calls, **extra}


def _ledger(**accounting):
    base = {"available": True, "settled_usd": 63.42, "confirmed_usd": 61.1, "estimated_usd": 2.32, "reserved_usd": 0.0,
            "unresolved_upper_bound_usd": 0.0, "unknown_unmetered": 0, "limit_usd": 200, "cost_final": True}
    return {"total_calls": 214, "total_cost": 63.42, "accounting": {**base, **accounting},
            "by_model": {"anthropic::claude-opus-5-5": _bucket(38.2, 71), "openai::gpt-5.6-sol": _bucket(15.9, 48)},
            "by_api_key": {"sk-or-…a91f": _bucket(41.6, 97)}, "by_model_category": {"task": _bucket(51.0, 150)},
            "by_task_category": {"task": _bucket(49.1, 139)}}


COST_CASES = [
    # width, theme, ledger, status, lead, [cost final, unresolved, unknown], reserved
    (390, "light", _ledger(confirmed_usd=58.1, estimated_usd=5.32, reserved_usd=4.3, unresolved_upper_bound_usd=1.25,
                           unknown_unmetered=3, cost_final=False, non_final_rows=2),
     200, "$63.42 / $200.00", ["Pending (2 open)", "$1.25", "3"], "$4.30"),
    (1100, "dark", _ledger(), 200, "$63.42 / $200.00", ["Yes", "$0.00", "0"], "$0.00"),
    (390, "dark", {"error": "Spend records could not be read"}, 500, "Unavailable", ["Unavailable", "—", "—"], "—"),
]


@pytest.mark.parametrize("width,theme,ledger,status,lead,facts,reserved", COST_CASES,
                         ids=["open-phone-light", "closed-window-dark", "unavailable-phone-dark"])
def test_costs_names_the_known_spend_with_what_is_still_open_before_anything_else(
        ui, width, theme, ledger, status, lead, facts, reserved):
    page = _open(ui, width=width, theme=theme, place="dashboard", tab="costs",
                 routes=[("**/api/cost-breakdown", _json(ledger, status))])
    page.wait_for_function("document.querySelector('#cost-known-limit').textContent !== 'Loading…'")
    seen = page.evaluate("""() => {
        const el = (sel) => document.querySelector(sel);
        const top = (sel) => el(sel).getBoundingClientRect().top;
        const size = (sel) => parseFloat(getComputedStyle(el(sel)).fontSize);
        const text = (sel) => el(sel).textContent.trim();
        const card = el('.costs-lead');
        const scroll = el('#page-costs .costs-scroll');
        return {
            text: [text('#cost-known-limit'), ...['#cost-final', '#cost-unresolved', '#cost-unknown'].map(text)],
            reserved: text('#cost-reserved'),
            inLead: ['#cost-known-limit', '#cost-final', '#cost-unresolved', '#cost-unknown'].map((s) => card.contains(el(s))),
            below: ['#cost-confirmed', '#cost-estimated', '#cost-reserved', '#cost-calls', '#cost-top-model', '#s-budget']
                .map((s) => !card.contains(el(s))),
            lead: top('#cost-known-limit'),
            open: ['#cost-final', '#cost-unresolved', '#cost-unknown'].map(top),
            breakdown: ['#cost-confirmed', '#cost-estimated', '#cost-reserved'].map(top),
            budget: top('#s-budget'),
            tables: top('#cost-by-model'),
            sizes: [size('#cost-known-limit'), size('#cost-final'), size('#cost-confirmed'), size('#cost-calls')],
            leadBottom: card.getBoundingClientRect().bottom,
            viewport: innerHeight,
            leadText: card.textContent,
            sideways: scroll.scrollWidth > scroll.clientWidth + 1,
        };
    }""")
    assert seen["text"] == [lead, *facts], seen["text"]
    assert seen["reserved"] == reserved
    # The known spend and everything that leaves it incomplete stand in one card; the rest comes after it.
    assert all(seen["inLead"]) and all(seen["below"]), seen
    assert seen["lead"] < min(seen["open"]) and max(seen["open"]) < min(seen["breakdown"]), seen
    assert max(seen["breakdown"]) < seen["budget"] < seen["tables"], seen
    # Each tier steps down: the known spend, the open facts and the composition, then the technical count.
    lead_size, fact_size, composition_size, technical_size = seen["sizes"]
    assert lead_size > composition_size > fact_size == technical_size, seen["sizes"]
    # On the first screen of a phone: the number and what is open about it, without a scroll.
    assert seen["leadBottom"] < seen["viewport"], seen
    if status != 200:
        assert "$" not in seen["leadText"], "an unavailable ledger invents no zero"
    assert not seen["sideways"], "the page never scrolls sideways"
    _shot(page, ui, "costs", width, theme)


# ------------------------------------------------------------------------------------------------------- Activity
LONG_TITLES = ["Install Ouroboros on the second Mac through Homebrew and explain what the working folders mean",
               "Telegram completion notifications after 7.7: find why only root tasks send a final message"]


def _queue(titles, runtime=1700):
    row = lambda i, title, **extra: {"id": f"t-{i}", "type": "task", **extra, "task": {"id": f"t-{i}", "title": title}}
    return {"queue": {"running": [row(i, title, runtime_sec=runtime) for i, title in enumerate(titles)], "pending": []}}


STATE = {"supervisor_ready": True, "active_chat_activities": [], "projects": [], "bg_consciousness_enabled": True,
         "bg_consciousness_state": {"enabled": True, "status": "sleeping", "detail": "Sleeping until 18:40.", "level": "act",
                                    "next_wake_at": "2026-10-27T15:40:00Z", "last_wake_at": "2026-10-27T14:40:00Z",
                                    "last_wake_outcome": "done", "spent_24h_usd": 1.84, "daily_usd": 5.0,
                                    "tasks_running": 0, "max_tasks": 2, "unknown_unmetered": 0}}


@pytest.mark.parametrize("width,theme", [(390, "light"), (1100, "dark")], ids=["phone-light", "window-dark"])
def test_activity_rows_keep_each_name_whole_and_say_how_long_in_words(ui, width, theme):
    page = _open(ui, width=width, theme=theme, place="dashboard", tab="activity", routes=[
        ("**/api/tasks*", _json(_queue(LONG_TITLES), path="/api/tasks")),
        ("**/api/state", _json(STATE, path="/api/state")),
        ("**/api/schedules", _json({"tasks": []}, path="/api/schedules"))])
    page.wait_for_selector('[data-activity-section="queue"] .activity-row')
    page.wait_for_selector('[data-activity-section="background"] .activity-row')
    rows = page.evaluate("""() => [...document.querySelectorAll('.activity-row')].map((row) => {
        const name = row.querySelector('.activity-name'), sub = row.querySelector('.activity-sub');
        const control = row.querySelector('[data-act]').getBoundingClientRect();
        const box = row.getBoundingClientRect();
        const range = document.createRange();
        range.selectNodeContents(row.querySelector('.activity-row-main'));
        const lines = [...range.getClientRects()].filter((r) => r.width > 0);
        const style = (el) => getComputedStyle(el);
        return {
            name: name.textContent, sub: sub.textContent,
            clipped: [name, sub].some((el) => el.scrollWidth > el.clientWidth + 1 || style(el).textOverflow === 'ellipsis'),
            wraps: [name, sub].every((el) => style(el).whiteSpace === 'normal'),
            nameSize: parseFloat(style(name).fontSize), subSize: parseFloat(style(sub).fontSize),
            nameInk: style(name).color, subInk: style(sub).color,
            controlInside: control.left >= box.left - 0.5 && control.right <= box.right + 0.5 && control.width > 0,
            overControl: lines.some((r) => r.right > control.left && r.left < control.right
                && r.bottom > control.top && r.top < control.bottom),
        };
    })""")
    queue = [row for row in rows if row["name"] in LONG_TITLES]
    assert [row["name"] for row in queue] == LONG_TITLES, "every name is there whole"
    for row in rows:
        assert not row["clipped"], row
        assert row["wraps"], row
        assert row["controlInside"] and not row["overControl"], row
        assert row["nameSize"] > row["subSize"] and row["nameInk"] != row["subInk"], row
    # 1700 s is said as 28 minutes, never as a raw number of seconds.
    assert all(row["sub"].endswith(" · 28 min") for row in queue), [row["sub"] for row in queue]
    assert not any("1700s" in row["sub"] for row in rows)
    # The Consciousness row keeps its last fact (it was cut by an ellipsis).
    background = next(row for row in rows if row["name"] == "Consciousness")
    assert "tasks 0/2" in background["sub"], background
    assert not page.evaluate("""() => { const e = document.querySelector('.activity-scroll');
        return e.scrollWidth > e.clientWidth + 1; }""")
    _shot(page, ui, "activity", width, theme)


# -------------------------------------------------------------------------------------------------------- Storage
GB = 1024 ** 3


def _storage():
    volume = {"free_bytes": 90 * GB, "total_bytes": 531 * GB, "observed_at": "2026-10-27T15:12:00Z"}
    place = lambda pid, group, gb: {"id": pid, "group": group, "roots": [pid], "present": True, "same_volume": True,
                                    "bytes": int(gb * GB), "logical_bytes": int(gb * GB), "files": 100, "unreadable": 0,
                                    "parts": [], "rest": {"count": 0, "bytes": 0}}
    places = [place("model_calls", "records", 3.3), place("work_folders", "temporary", 2.2)]
    count = {"started_at": "2026-10-27T13:20:00Z", "finished_at": "2026-10-27T13:22:00Z", "duration_seconds": 96,
             "io_priority": "low", "basis": "allocated", "bytes": sum(p["bytes"] for p in places),
             "logical_bytes": sum(p["bytes"] for p in places), "files": 200, "unreadable": 0, "volume": volume,
             "places": places, "growth_bytes": 0, "growth_since": "2026-10-24T15:20:00Z"}
    group = lambda gid, size, **extra: {"id": gid, "count": 9, "inside_wait": 0, "in_use": 0, "unreadable": 0, "bytes": size,
                                        "bytes_complete": True, "items": [], **extra}
    return {"count": count, "counting": None, "error": None, "volume": volume, "retention_days": 7, "free": {
        "state": "ready", "plan": "plan-7", "started_at": "2026-10-27T15:10:00Z", "finished_at": "2026-10-27T15:11:00Z",
        "done": 3, "total": 3, "run": None, "error": None, "groups": [
            group("review_checkouts", 2 * GB), group("lost_copies", int(1.3 * GB)),
            group("review_evidence", int(3.7 * GB), bytes_complete=False, unreadable=2)],
        "pool": {"count": 0, "bytes": 0, "bytes_complete": True, "kinds": {}, "excluded": []}}}


@pytest.mark.parametrize("width,theme", [(390, "light"), (1100, "dark")], ids=["phone-light", "window-dark"])
def test_temporary_space_keeps_a_groups_name_its_room_on_a_narrow_card(ui, width, theme):
    page = _open(ui, width=width, theme=theme, place="dashboard", tab="storage",
                 routes=[("**/api/storage", _json(_storage(), path="/api/storage"))])
    # The saved groups stand behind "Not included": read-only rows first, the checklist on purpose.
    page.locator('[data-free-disclosure="left"] > summary').click()
    page.locator("[data-free-saved]").click()
    page.wait_for_selector("[data-free-group]")
    groups = page.evaluate("""() => [...document.querySelectorAll('.storage-free-group')].map((group) => {
        const label = group.querySelector('.storage-free-label'), size = group.querySelector('.storage-free-size');
        const l = label.getBoundingClientRect(), s = size.getBoundingClientRect();
        const range = document.createRange();
        range.selectNodeContents(label);
        return {name: label.textContent, qualified: /could not be read/.test(size.textContent), width: l.width,
                lines: new Set([...range.getClientRects()].map((r) => Math.round(r.top))).size,
                sizeBelowName: s.top >= l.bottom - 1, sameLine: Math.abs(s.top - l.top) < 12};
    })""")
    evidence = next(group for group in groups if group["qualified"])
    assert evidence["name"] == "Reviewers’ saved files of finished tasks"
    assert evidence["lines"] <= 2, evidence
    if width < 520:
        # The name keeps the card's width; the size that carries its qualifier stands under it.
        assert evidence["width"] > 200 and evidence["sizeBelowName"], evidence
        # A plain size stays on the name's line, where sizes are scanned.
        assert all(group["sameLine"] for group in groups if not group["qualified"]), groups
    else:
        assert evidence["sameLine"], evidence
    _shot(page, ui, "storage-tray", width, theme)


# -------------------------------------------------------------------------------------------------------- Widgets
def _declarative(skill, title, text):
    return {"skill": skill, "tab_id": "main", "key": f"{skill}:main", "title": title, "span": 1,
            "render": {"kind": "declarative", "schema_version": 1, "components": [{"type": "markdown", "text": text}]}}


def _framed(skill, title):
    return {"skill": skill, "tab_id": "main", "key": f"{skill}:main", "title": title, "span": 1,
            "render": {"kind": "module", "entry": "main.js", "entry_type": "classic", "start": "manual", "height": 160}}


WIDGET_TABS = [
    _declarative("weather", "Weather", "**7 °C** now, light rain after 18:00."),
    _declarative("release_health", "Release health: production pipeline, error budget and rollback readiness",
                 "All checks are passing."),
    _declarative("infra_spend", "Quarterly infrastructure spending and reliability overview for the platform",
                 "Within budget."),
    _framed("parked", "Nightly export"),
]


@pytest.mark.parametrize("width,theme", [(390, "light"), (1100, "dark")], ids=["phone-light", "window-dark"])
def test_widget_heads_keep_the_menu_and_the_move_handle_as_one_visible_quiet_pair(ui, width, theme):
    page = _open(ui, width=width, theme=theme, place="widgets",
                 routes=[("**/api/widgets", _json({"ui_tabs": WIDGET_TABS}, path="/api/widgets"))])
    page.wait_for_selector(".widgets-card [data-widget-power]")
    # A fault line under a narrow framed card: the lifecycle controls may drop, the pair never does.
    page.evaluate("""() => { const s = document.querySelector('[data-widget-status]'); s.hidden = false;
        s.dataset.tone = 'error'; s.textContent = 'Widget script error: Script error.'; }""")
    page.wait_for_timeout(300)
    cards = page.evaluate("""() => [...document.querySelectorAll('.widgets-card')].map((card) => {
        const rect = (el) => el.getBoundingClientRect();
        const menu = card.querySelector('[data-widget-menu-trigger]'), drag = card.querySelector('[data-widget-reorder-handle]');
        const power = card.querySelector('[data-widget-power]');
        const [m, d, c] = [rect(menu), rect(drag), rect(card)];
        // The title's own line boxes, never one under the pair (a wrapping title once ran its ink under the ⋮).
        const words = document.createRange();
        words.selectNodeContents(card.querySelector('.widgets-card-title strong'));
        const under = [...words.getClientRects()].filter((line) => [m, d].some((r) =>
            Math.min(line.right, r.right) - Math.max(line.left, r.left) > 0.5
            && Math.min(line.bottom, r.bottom) - Math.max(line.top, r.top) > 0.5)).length;
        const shown = (el, r) => r.width > 0 && r.height > 0 && getComputedStyle(el).opacity === '1'
            && getComputedStyle(el).visibility === 'visible' && getComputedStyle(el).display !== 'none';
        const quiet = (el) => getComputedStyle(el).borderTopWidth === '0px'
            && getComputedStyle(el).backgroundColor === 'rgba(0, 0, 0, 0)';
        return {title: card.querySelector('.widgets-card-title strong').textContent,
                visible: shown(menu, m) && shown(drag, d), quiet: quiet(menu) && quiet(drag),
                sameBox: Math.abs(m.width - d.width) <= 1 && Math.abs(m.height - d.height) <= 1,
                sideBySide: Math.abs(m.top - d.top) <= 1 && m.right <= d.left + 1 && d.left - m.right <= 12,
                inHead: d.top - c.top < 40 && c.right - d.right >= 8,
                pairFirst: power ? d.top <= rect(power).top + 1 : true, titleUnderPair: under};
    })""")
    assert [card["title"] for card in cards] == [tab["title"] for tab in WIDGET_TABS]
    for card in cards:
        assert card["visible"] and card["quiet"] and card["sameBox"] and card["sideBySide"], card
        assert card["inHead"] and card["pairFirst"] and card["titleUnderPair"] == 0, card
    _shot(page, ui, "widgets", width, theme)
