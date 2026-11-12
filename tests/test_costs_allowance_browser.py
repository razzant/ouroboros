"""The consciousness allowance in Costs › Budget, through the real candidate server (owner, 10 Nov 2026).

Costs is the allowance's one editor, a setting row that saves when its field is left: a typed 0 and decimals save
and read back after a reload, the posted body is that one limit, and an unrelated row written elsewhere keeps it: a
row posts its own key and nothing else. Settings has no control for it, only the home's door to Costs.
Under the field stands the clock's own 24-hour window, from spend seeded through the ledger itself: what it leaves
out, a reached limit with its conditional room time in the viewer's named zone. A wake held at the allowance it started
with, the Off switch and an unreadable ledger are said apart; the held state's snapshot is the consciousness
producers' own (``hold_cause``, ``describe_consciousness_state``) over the server's answer, and the Consciousness tab
and the chip's Shade say it and open Costs. Desktop and phone, light and dark, Chromium and WebKit; the frames go to
``OUROBOROS_UI_EVIDENCE_DIR`` when it is set.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import pytest

pytest_plugins = ("tests.test_ui_smoke_playwright",)
pytestmark = [pytest.mark.ui_browser]

HOUR = 3600.0
KEY = "OUROBOROS_CONSCIOUSNESS_DAILY_USD"
FIELD = f"#setting-{KEY}"  # the allowance's setting row (settings_controls.js settingsRow)
UNRELATED = "OUROBOROS_GC_RETENTION_DAYS"  # another tab's row: how long finished work is kept (Storage › Cleanup)
ZONE = "Europe/Moscow"  # the viewer's zone: the room time must name it
COMBOS = [(1440, 900, "light"), (1440, 900, "dark"), (390, 844, "light"), (390, 844, "dark")]


def _evidence(name: str) -> Path | None:
    root = os.environ.get("OUROBOROS_UI_EVIDENCE_DIR")
    if not root:
        return None
    path = Path(root) / "costs-allowance"
    path.mkdir(parents=True, exist_ok=True)
    return path / name


def _settings(data_dir: Path) -> dict:
    return json.loads((data_dir / "settings.json").read_text(encoding="utf-8"))


def _page(browser, engine, *, width=1440, height=900, theme="light"):
    """A page in the viewer's zone and in English (the page writes a date in the browser's locale, and WebKit takes
    the host's unless told: "12 нояб." where ``_room_words`` says "Nov 12"); its uncaught errors in ``page.errors``
    (``tests/ui_console_errors.PageErrors``: only WebKit's own-origin fetch a navigation cancelled is set aside, #1468)."""
    from tests.ui_console_errors import PageErrors

    context = browser.new_context(viewport={"width": width, "height": height}, color_scheme=theme, locale="en-US",
                                  timezone_id=ZONE, has_touch=width < 720, device_scale_factor=2)
    page = context.new_page()
    page.add_init_script(f"localStorage.setItem('ouroboros.theme', {json.dumps(theme)})")
    page.errors = PageErrors(page, engine)
    return page


def _nav(page, target: str) -> None:
    """A page of the shell from its navigation, through the phone's drawer where the bar is narrow."""
    toggle = page.locator(".page.active [data-mobile-nav-toggle]:visible")
    if toggle.count():
        toggle.first.click()
    page.click(f'[data-nav-page="{target}"]')


def _costs(page, url: str, *, load: bool = True) -> None:
    from playwright.sync_api import expect

    if load:
        page.goto(url + "/", wait_until="domcontentloaded")
        page.wait_for_selector("#page-chat", timeout=30_000)
    _nav(page, "dashboard")
    page.click('[data-dashboard-tab="costs"]')
    expect(page.locator(FIELD)).to_be_visible(timeout=30_000)
    page.wait_for_function("document.getElementById('costs-allowance-spend').textContent.trim() !== 'Reading…'"
                           f" && document.getElementById('setting-{KEY}').value !== ''", timeout=30_000)


def _saving(page):
    """The settings write a row sends when its field is left, awaited to its answer."""
    return page.expect_response(lambda response: response.request.method == "POST"
                                and response.url.endswith("/api/settings"), timeout=30_000)


def _write_unrelated(page, data_dir: Path, posts: list, days: int) -> dict:
    """Type and leave another tab's row (Storage's retention): its save carries that key alone. Returns what is stored."""
    from playwright.sync_api import expect

    page.click('[data-dashboard-tab="storage"]')
    field = page.locator(f"#setting-{UNRELATED}")
    expect(field).to_be_visible(timeout=30_000)
    page.wait_for_function(f"document.getElementById('setting-{UNRELATED}').value !== ''", timeout=30_000)
    with _saving(page):
        field.fill(str(days))
        field.press("Tab")
    expect(page.locator(f"#setting-{UNRELATED}-result")).to_contain_text("Saved", timeout=30_000)
    assert posts[-1] == {UNRELATED: days}, f"a row's save carries its own key and never the allowance: {posts[-1]}"
    stored = _settings(data_dir)
    assert int(stored[UNRELATED]) == days, stored
    return stored


def _switch(page, url: str, on: bool) -> None:
    """Consciousness on or off through the Consciousness tab's own Start/Stop (this isolated server's switch)."""
    from playwright.sync_api import expect

    _costs(page, url)
    page.locator("[data-costs-consciousness]").click()
    summary = page.locator("[data-cx-summary]")
    expect(summary).to_contain_text("Consciousness · o", timeout=30_000)
    toggle = page.locator('[data-cx-act="toggle"]')
    if toggle.inner_text() == ("Start" if on else "Stop"):
        toggle.click()
    expect(summary).to_contain_text(f"Consciousness · {'on' if on else 'off'}", timeout=30_000)


def _fits(page) -> None:
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), "the page scrolls sideways"
    assert page.evaluate("""() => { const c = document.querySelector('.costs-budget');
        return c.scrollWidth <= c.clientWidth + 1; }"""), "the Budget card overflows"
    # The shell gate's hit test also covers the populated states: a held wake adds its own sentence link.
    from tests.test_design_system_gate_browser import MEASURE

    reach = page.evaluate(MEASURE, '.costs-budget')
    assert not any(reach.values()), reach


def _shoot(page, name: str) -> None:
    card = _evidence(f"{name}-card.png")
    if card is None:
        return
    page.locator(".costs-budget").scroll_into_view_if_needed()
    page.locator(".costs-budget").screenshot(path=str(card))
    page.screenshot(path=str(_evidence(f"{name}-screen.png")))


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_the_allowance_is_edited_in_costs_alone_and_survives_an_unrelated_row_written_elsewhere(direct_server_with_data, engine):
    from playwright.sync_api import expect, sync_playwright

    url, data_dir = direct_server_with_data["url"], direct_server_with_data["data_dir"]
    assert KEY not in _settings(data_dir), "starts at the default"
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page = _page(browser, engine)
            posts = []
            page.on("request", lambda request: posts.append(request.post_data_json)
                    if request.method == "POST" and request.url.endswith("/api/settings") else None)
            _switch(page, url, True)  # a fresh install starts it Off, which this card says first
            _costs(page, url)
            allowance = page.locator(FIELD)
            expect(allowance).to_have_value("20")
            expect(page.locator(f"label[for=setting-{KEY}]")).to_have_text("Consciousness per 24 hours")
            # The keyboard reaches it after the per-task cap, as the card reads.
            page.focus("#setting-OUROBOROS_PER_TASK_COST_USD")
            page.keyboard.press("Tab")
            assert page.evaluate("document.activeElement.id") == f"setting-{KEY}"
            with _saving(page):
                allowance.fill("0")
                allowance.press("Tab")  # leaving the field saves it
            status = page.locator(f"#setting-{KEY}-result")
            expect(status).to_contain_text("Saved", timeout=30_000)
            # LANE A: with the server's receipt the line also says when it applies and "was 20", with Undo.
            said = status.inner_text()
            assert said == "Saved" or (said.startswith("Saved · ") and "was 20" in said), said
            assert posts[-1] == {KEY: 0}, posts
            assert float(_settings(data_dir)[KEY]) == 0
            expect(page.locator("#costs-allowance-state")).to_contain_text(
                "At $0 consciousness may not spend: new wake-ups, replies in places and its own tasks do not start.")
            _shoot(page, f"{engine}-zero-saved")
            _costs(page, url)  # a full reload reads the saved 0 back
            expect(allowance).to_have_value("0")
            # A saved 0 is a value: an unrelated row written elsewhere keeps it (the page-wide Save of the old Settings
            # once wrote the field's '20' back). A row posts its own key and nothing else.
            stored = _write_unrelated(page, data_dir, posts, 9)
            assert float(stored[KEY]) == 0, stored
            _costs(page, url)
            expect(allowance).to_have_value("0")
            allowance.fill("12.75")
            with _saving(page):  # leaving the field for the Consciousness door saves it first
                page.locator("[data-costs-consciousness]").click()
            _costs(page, url, load=False)
            expect(allowance).to_have_value("12.75")
            assert posts[-1] == {KEY: 12.75}, posts
            assert float(_settings(data_dir)[KEY]) == 12.75
            _costs(page, url)
            expect(allowance).to_have_value("12.75")
            expect(page.locator("#costs-allowance-spend")).to_contain_text("of $12.75 in the last 24 hours")
            # Another unrelated row keeps the decimal too.
            stored = _write_unrelated(page, data_dir, posts, 11)
            assert float(stored[KEY]) == 12.75, stored
            # It is edited in Costs alone: the page holds one control for it, Settings none, and the home's Costs row is
            # the door to it, which leads back.
            assert page.locator(f'[data-setting-row="{KEY}"]').count() == 1
            _nav(page, "settings")
            home = page.locator("#page-settings [data-settings-home]")
            expect(home).to_be_visible(timeout=30_000)
            assert page.locator(f'#page-settings [data-setting-row="{KEY}"], #page-settings #setting-{KEY}').count() == 0
            door = home.locator('[data-settings-open="dashboard:costs"]')
            door.scroll_into_view_if_needed()
            if (shot := _evidence(f"{engine}-settings-door.png")) is not None:
                door.screenshot(path=str(shot))
            door.click()
            expect(page.locator(FIELD)).to_be_visible(timeout=30_000)
            expect(allowance).to_have_value("12.75")
            expect(page.locator("#dashboard-panel-costs > .dashboard-back")).to_have_text("Settings")
            assert page.errors.unexpected() == [], page.errors.report()
        finally:
            browser.close()


def _seed_spend(data_dir: Path, now: float) -> float:
    """$12.40 known and one call without a price over two own wakes, through the ledger's own writers; returns the
    instant the older wake's spend leaves the 24 hours."""
    from unittest.mock import patch

    from ouroboros import store, usage_accounting as ua, usage_store
    from tests.fixtures_usage_store import _settle

    def at(ts):
        return patch.object(usage_store, "utc_now_iso", lambda: dt.datetime.fromtimestamp(ts, dt.timezone.utc).isoformat())

    try:
        with at(now - 20 * HOUR):
            _settle(data_dir, cost=8.40, cost_final=True, task_id="wake-a", root_task_id="wake-a", category="consciousness")
        with at(now - 3 * HOUR):
            _settle(data_dir, cost=4.00, cost_final=True, task_id="wake-b", root_task_id="wake-b", category="consciousness")
        with at(now - HOUR):
            ua.record_unmetered_external_dispatch("ext-b", drive_root=data_dir, model="ext-model", task_id="wake-b",
                                                  root_task_id="wake-b", category="consciousness")
    finally:
        usage_store.forget(data_dir)
        store.close_connections_under(data_dir)
    return now - 20 * HOUR + 24 * HOUR


def _room_words(after: float, now: float) -> str:
    """What the viewer in ``ZONE`` reads for that instant: the whole minute by which room is back, its zone named."""
    from zoneinfo import ZoneInfo

    zone = ZoneInfo(ZONE)
    at = dt.datetime.fromtimestamp((int(after // 60) * 60) + 60, zone)
    date = "" if at.date() == dt.datetime.fromtimestamp(now, zone).date() else f"{at:%b} {at.day}, "
    return f"{date}{at:%H:%M} GMT+3"


def _held(bg: dict) -> dict:
    """The server's snapshot with its own wake held at the allowance it started with, as the producers say it."""
    from ouroboros.consciousness_view import describe_consciousness_state, hold_cause
    from ouroboros.consciousness_wake import wake_task_metadata

    pause = {"pause_id": "p", "state": "paused", "reason": "budget", "rail": "graceful_ceiling", "scope": "root",
             "cost_ceiling": {"state": "active", "ceiling_usd": 2.99, "root_cap_usd": 50.0, "planning_margin_usd": None,
                              "basis": "producer_allowance"}}
    hold = hold_cause(wake_task_metadata("act", "heartbeat", root_cost_ceiling_usd=2.99,
                                         root_cost_ceiling_source="allowance"), pause)
    assert hold == {"cause": "allowance", "ceiling_usd": 2.99}
    return describe_consciousness_state({**bg, "live_wake_task_id": "wake-held", "live_wake_held": True,
                                         "live_wake_hold": hold}, True)


def _unreadable(bg: dict) -> dict:
    from ouroboros.consciousness_view import describe_consciousness_state

    return describe_consciousness_state({**bg, "spent_24h_usd": None, "unknown_unmetered": 0, "allowance_status":
                                         "allowance_unknown", "allowance_room_after": "", "allowance_open_rows": None,
                                         "last_wake_outcome": "skipped:allowance_unknown",
                                         "last_error": "UsageLockUnavailable: busy"}, True)


def _overlay(page, change) -> None:
    """Answer /api/consciousness and /api/state with the server's own answer under ``change`` (``None``: as served)."""
    # These are this page's only routes. Let an in-flight fetch/fulfill finish before replacing the state.
    page.unroute_all(behavior="wait")
    if change is None:
        return

    def consciousness(route):
        response = route.fetch()
        body = response.json()
        route.fulfill(response=response, json={**body, **change({k: v for k, v in body.items() if k != "recent_wakes"})})

    def state(route):
        response = route.fetch()
        body = response.json()
        if isinstance(body.get("bg_consciousness_state"), dict):
            body["bg_consciousness_state"] = change(body["bg_consciousness_state"])
        route.fulfill(response=response, json=body)

    page.route("**/api/consciousness*", consciousness)
    page.route("**/api/state*", state)


@pytest.mark.parametrize("engine", ["webkit", "chromium"])
def test_the_allowance_says_its_spend_hold_and_room_in_each_state(direct_server_with_data, engine):
    """The facts under the field in every state the owner can meet, each looked at on a desktop and a phone in both
    themes; a held wake is said on Costs, the Consciousness tab and the Shade, each with its doors."""
    import time

    from playwright.sync_api import expect, sync_playwright

    server = direct_server_with_data
    url, data_dir = server["url"], server["data_dir"]
    server["stop_server"]()
    now = time.time()
    leaves = _seed_spend(data_dir, now)
    server["start_server"]()
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            pages = {combo: _page(browser, engine, width=combo[0], height=combo[1], theme=combo[2]) for combo in COMBOS}
            _switch(pages[COMBOS[0]], url, True)

            def each(state, check):
                for (width, _height, theme), page in pages.items():
                    _costs(page, url)
                    check(page)
                    _fits(page)
                    _shoot(page, f"{engine}-{width}-{theme}-{state}")

            def spend_says(text):
                return lambda page: expect(page.locator("#costs-allowance-spend")).to_have_text(text)

            # 1. Room left: the server's own window, the unpriced call said, no condition line; last, the model its
            # wake-ups think with (this server's Main, the slot left empty), in every state but Off.
            normal = "≥ $12.40 of $20.00 in the last 24 hours · 1 without a price · Consciousness ›"
            model = "Thinks with mock-model (Main model)."
            each("normal", lambda page: (spend_says(normal)(page),
                                         expect(page.locator(".costs-allowance-line")).to_have_text([model])))
            # 2. The limit reached: what waits, and when room returns if nothing more is charged, in the viewer's zone.
            page = pages[COMBOS[0]]
            with _saving(page):
                page.locator(FIELD).fill("10")
                page.locator(FIELD).press("Enter")  # Enter commits the field as leaving it does
            expect(page.locator(f"#setting-{KEY}-result")).to_contain_text("Saved", timeout=30_000)
            served = page.evaluate("fetch('/api/consciousness').then(r => r.json())")
            assert served["allowance_status"] == "exhausted" and served["daily_usd"] == 10, served
            assert abs(dt.datetime.fromisoformat(served["allowance_room_after"]).timestamp() - leaves) < 1, served
            reached = ("Consciousness limit reached: new wake-ups, replies in places and its own tasks wait. If its charges stay "
                       f"as they are, there is room again by {_room_words(leaves, now)}.")
            each("reached", lambda page: (spend_says("≥ $12.40 of $10.00 in the last 24 hours · 1 without a price · "
                                                     "Consciousness ›")(page),
                                          expect(page.locator(".costs-allowance-line")).to_have_text([reached, model])))
            # 3. Its wake held at the allowance it started with: two facts, the window and the wake, with its card's door.
            for page in pages.values():
                _overlay(page, _held)
            held = ("The consciousness wake-up is paused at the allowance it had left when it started ($2.99). Resume keeps "
                    "that limit, so it pauses again while its known spend is at it; room returning or a higher limit does "
                    "not raise it. Stop now on its card ends it and frees the clock for its next check. Open in Main ↗")
            each("held", lambda page: expect(page.locator(".costs-allowance-line")).to_have_text([reached, held, model]))
            page = pages[COMBOS[0]]
            opened = page.evaluate_handle("""() => { const seen = [];
                window.addEventListener('ouro:open-task', (event) => seen.push(event.detail)); return seen; }""")
            page.locator(".costs-allowance-line [data-open-task]").click()
            assert opened.json_value() == [{"task_id": "wake-held", "chat_id": 1}]
            # ... said on the Consciousness tab with the owner's mark and its allowance door, which opens that row in
            # Costs, and in the Shade.
            for (width, _height, theme), page in pages.items():
                _costs(page, url)
                page.locator("[data-costs-consciousness]").click()
                headline = page.locator(".consciousness-headline")
                expect(headline).to_contain_text("Wake-up wake-held stopped at the allowance it had left when it "
                                                 "started ($2.99).", timeout=30_000)
                assert headline.locator(".status-mark").get_attribute("data-hue") == "owner"
                expect(page.locator(".consciousness-headline-note")).to_have_text(
                    "Resume keeps that limit, so it pauses again while its known spend is at it, and a higher allowance "
                    "does not raise it. Stop now on its card ends it and frees the clock for its next check.")
                if (shot := _evidence(f"{engine}-{width}-{theme}-held-consciousness.png")) is not None:
                    page.locator("[data-cx-summary]").screenshot(path=str(shot))
                page.locator('#page-dashboard [data-setting-row="consciousness-allowance"]').click()
                expect(page.locator(FIELD)).to_be_focused(timeout=30_000)
                _nav(page, "chat")
                chip = page.locator("#page-chat .cx-chip")
                expect(chip).to_be_visible(timeout=30_000)
                expect(chip.locator(".cx-chip-mark")).to_have_attribute("data-hue", "owner", timeout=30_000)
                chip.click()
                shade = page.locator("#page-chat .cx-shade")
                expect(shade.locator(".cx-state")).to_contain_text("stopped at the allowance it had left")
                expect(shade.locator(".cx-state-note")).to_contain_text("Resume keeps that limit, so it pauses again")
                costs = shade.locator(".cx-door", has_text="Costs ›")
                expect(costs).to_be_visible()
                page.wait_for_function("""() => [...document.querySelectorAll('#page-chat .cx-shade')].every((shade) =>
                    shade.getAnimations({ subtree: true }).every((animation) => animation.playState !== 'running'))""")
                page.wait_for_timeout(250)  # the material's own fade-in, whatever drives it, settles before the frame
                box = shade.bounding_box()
                assert box["x"] >= 0 and box["x"] + box["width"] <= width + 1, box
                if (shot := _evidence(f"{engine}-{width}-{theme}-held-shade.png")) is not None:
                    page.screenshot(path=str(shot))
                costs.click()
                expect(page.locator(FIELD)).to_be_visible(timeout=30_000)
            # 4. An unreadable ledger: unknown, never $0, and no time.
            for page in pages.values():
                _overlay(page, _unreadable)
            each("unknown", lambda page: (
                spend_says("What it spent in the last 24 hours is unknown. · Consciousness ›")(page),
                expect(page.locator(".costs-allowance-line")).to_have_text(
                    ["The usage ledger could not be read, so whether consciousness has room is unknown.", model])))
            # 4b. A long model id with no break point of its own (no space, hyphen or slash: only the line's overflow-wrap
            # breaks it) stays inside the card, a phone's included.
            long_id = "qwen3_coder_480b_a35b_instruct_" * 3 + "fp8"
            for page in pages.values():
                _overlay(page, lambda bg: {**bg, "model_route": {"value": long_id, "layer": "consciousness", "local": False}})
            each("long-model", lambda page: expect(page.locator(".costs-allowance-line").last).to_have_text(
                f"Thinks with {long_id}."))
            # 5. Off, through the tab's own switch: nothing counts here until it is started again.
            for page in pages.values():
                _overlay(page, None)
            _switch(pages[COMBOS[0]], url, False)
            each("off", lambda page: expect(page.locator(".costs-allowance-line")).to_have_text(
                ["Consciousness is off: it starts nothing that counts here until it is started again."]))
            for page in pages.values():
                assert page.errors.unexpected() == [], page.errors.report()
        finally:
            browser.close()
