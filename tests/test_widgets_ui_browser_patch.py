"""Widgets keyed-patch and reconnect browser smoke (cycle-A fix round), on
chromium and webkit. A CHANGED card whose module frame is running (its skill's
``revision`` moved) is stopped in order first: the old card stands, marked and
"Stopping…", while its async dispose hook flushes through the bridge, and only
then is it removed and the fresh card mounted — never two frames for one key,
never a frame torn down in the same turn as its dispose message. The same
reconcile runs on every WebSocket (re)connect (``ws.on('open')``) and removes /
re-adds cards accordingly while untouched cards keep their nodes and frames.
Two quick launch-policy changes are written one after the other (never two
read-modify-writes of the stored map in flight), and closing a policy menu with
Escape hands focus back to its trigger on both engines. Kept apart from
``test_widgets_ui_browser_lifecycle.py`` so that file stays under the
size-ratchet band; the fixtures are shared by import."""

from __future__ import annotations

import json
import time

import pytest

from tests.test_ui_smoke_playwright import direct_server_with_data as _direct_server_with_data
from tests.test_widgets_ui_browser_lifecycle import _click_nav, _write_lifecycle_widget_extension

direct_server_with_data = _direct_server_with_data

# Host-side probe: every host fetch to an extension route is logged with the
# number of widget frames attached and whether the frame the test marked as
# `window.__oldFrame` is still connected at that moment. A dispose hook's
# bridged flush goes through the parent's `fetch`, so this observes from the
# host's side whether the old frame was still alive when its hook ran.
_PATCH_PROBE_SCRIPT = r"""
(() => {
    const log = [];
    window.__hostFetchLog = log;
    const original = window.fetch.bind(window);
    window.fetch = (input, init) => {
        const url = typeof input === 'string' ? input : String(input && input.url || input);
        if (url.includes('/api/extensions/')) {
            log.push({
                url,
                frames: document.querySelectorAll('#widgets-list iframe').length,
                oldFrameConnected: window.__oldFrame ? window.__oldFrame.isConnected : null,
            });
        }
        return original(input, init);
    };
})();
"""


@pytest.mark.ui_browser
@pytest.mark.parametrize("browser_name", ("chromium", "webkit"))
def test_ui_smoke_widget_changed_card_and_reconnect_reconcile(direct_server_with_data, browser_name):
    """(1) A revision bump of a RUNNING auto card, reconciled through the ws
    `open` trigger: the old card is marked `data-widget-removed`, both cards
    stand side by side while the hook's flush POST and its answer reach the
    host with the old frame still connected, then the old card goes and exactly
    one fresh frame remains; the hang and manual cards (unchanged entries) keep
    their nodes and frames. (2) The `open` reconcile removes a vanished card and
    re-adds it when it is back, touching no running frame. (3) Two quick policy
    changes hold at most one stored-map read in flight and both land. (4) Escape
    closes the menu and focuses its ⋮ trigger."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    skill = _write_lifecycle_widget_extension(data_dir)
    page_errors: list[str] = []

    def card(tab_id: str) -> str:
        return f'[data-widget-key="{skill}:{tab_id}"]'

    def frame_count(page, tab_id: str) -> int:
        return page.locator(f"{card(tab_id)} iframe").count()

    def wait_frame(page, tab_id: str, present: bool, timeout: int = 10_000) -> None:
        page.wait_for_function(
            "([selector, present]) => (document.querySelector(`${selector} iframe`) !== null) === present",
            arg=[card(tab_id), present],
            timeout=timeout,
        )

    def toggle(page, enabled: bool) -> int:
        return page.evaluate(
            """async ([skill, enabled]) => (await fetch(`/api/skills/${encodeURIComponent(skill)}/toggle`, {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({enabled}),
            })).status""",
            [skill, enabled],
        )

    def emit_open(page) -> None:
        # The loopback debug hook app.js exposes; the same `open` the client
        # emits on every (re)connect, which the Widgets page reconciles on.
        page.evaluate("() => { window.__ouroWs.emit('open', {previouslyConnected: true}); }")

    def hang_frame_kept(page) -> bool:
        return page.evaluate("(selector) => document.querySelector(`${selector} iframe`)?.__ouroHangFrame === true", card("hang"))

    def routed_list(route, mutate) -> None:
        data = route.fetch().json()
        mutate(data)
        route.fulfill(status=200, content_type="application/json", body=json.dumps(data))

    try:
        with sync_playwright() as pw:
            browser = getattr(pw, browser_name).launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.add_init_script(_PATCH_PROBE_SCRIPT)
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                assert toggle(page, True) == 200
                page.click('[data-nav-page="widgets"]')
                for tab_id in ("manual", "auto", "hang", "gauge"):
                    page.locator(card(tab_id)).wait_for(state="visible", timeout=30_000)
                wait_frame(page, "auto", True)
                wait_frame(page, "hang", True)
                page.wait_for_function(
                    "(selector) => document.querySelector(`${selector} [data-widget-power]`)?.textContent === 'Stop'",
                    arg=card("auto"),
                    timeout=10_000,
                )
                page.frame_locator(f"{card('auto')} iframe").locator("#root").wait_for(state="visible", timeout=10_000)

                # Mark the running auto frame and its card (the opaque-origin document
                # is unreadable from here, so identity is an expando on the nodes), the
                # hang frame (an unchanged entry), and watch the auto key's cards.
                page.evaluate(
                    """(sel) => {
                        const cardNode = document.querySelector(sel.auto);
                        const frame = cardNode.querySelector('iframe');
                        frame.__ouroOldFrame = true;
                        window.__oldFrame = frame;
                        cardNode.__ouroOldCard = true;
                        document.querySelector(`${sel.hang} iframe`).__ouroHangFrame = true;
                        window.__patchObs = {removedMarked: false, oldStoppingText: '', maxAutoCards: 1, maxAutoFrames: 1};
                        new MutationObserver(() => {
                            if (cardNode.hasAttribute('data-widget-removed')) {
                                window.__patchObs.removedMarked = true;
                                window.__patchObs.oldStoppingText = cardNode.querySelector('[data-widget-status]')?.textContent || '';
                            }
                            const cards = document.querySelectorAll(sel.auto);
                            window.__patchObs.maxAutoCards = Math.max(window.__patchObs.maxAutoCards, cards.length);
                            window.__patchObs.maxAutoFrames = Math.max(
                                window.__patchObs.maxAutoFrames, document.querySelectorAll(`${sel.auto} iframe`).length);
                        }).observe(document.getElementById('widgets-list'), {
                            subtree: true, childList: true, attributes: true, attributeFilter: ['data-widget-removed'],
                        });
                    }""",
                    {"auto": card("auto"), "hang": card("hang")},
                )
                page.evaluate("window.__hostFetchLog.length = 0")

                # (1) The auto card's skill revision moved; the reconnect reconcile
                # fetches the list, patches by key and lets the old frame flush first.
                page.route("**/api/widgets", lambda route: routed_list(route, lambda data: [
                    tab.__setitem__("revision", "f" * 64)
                    for tab in data.get("ui_tabs", []) if tab.get("key") == f"{skill}:auto"
                ]))
                emit_open(page)
                page.wait_for_function(
                    """(selector) => {
                        const cards = document.querySelectorAll(selector);
                        if (cards.length !== 1 || cards[0].__ouroOldCard === true) return false;
                        const frames = cards[0].querySelectorAll('iframe');
                        return frames.length === 1 && frames[0].__ouroOldFrame !== true;
                    }""",
                    arg=card("auto"),
                    timeout=15_000,
                )
                page.frame_locator(f"{card('auto')} iframe").locator("#root").wait_for(state="visible", timeout=10_000)
                page.wait_for_function(
                    "(selector) => document.querySelector(`${selector} [data-widget-status]`)?.textContent === 'Running'",
                    arg=card("auto"),
                    timeout=10_000,
                )
                page.unroute("**/api/widgets")
                observed = page.evaluate("window.__patchObs")
                assert observed["removedMarked"] is True, observed
                assert observed["oldStoppingText"] == "Stopping…", observed
                assert observed["maxAutoCards"] == 2, observed
                assert observed["maxAutoFrames"] == 1, observed
                fetch_log = page.evaluate("window.__hostFetchLog")
                flush = [row for row in fetch_log if row["url"].endswith(f"/api/extensions/{skill}/flush")]
                answered = [row for row in fetch_log if f"/api/extensions/{skill}/ping?flushed=200" in row["url"]]
                assert len(flush) == 1, fetch_log
                assert flush[0]["oldFrameConnected"] is True, fetch_log
                assert len(answered) == 1, fetch_log
                assert answered[0]["oldFrameConnected"] is True, fetch_log
                assert page.evaluate("window.__oldFrame.isConnected") is False
                assert page.evaluate("(selector) => document.querySelectorAll(selector).length", card("auto")) == 1
                assert frame_count(page, "auto") == 1
                assert page.locator("#widgets-list [data-widget-removed]").count() == 0
                assert hang_frame_kept(page), "an unchanged entry must keep its frame across the patch"
                assert frame_count(page, "manual") == 0

                # (2) The reconnect reconcile alone (no lifecycle event) removes a card
                # that left the list and re-adds it once it is back; running frames of
                # other cards are not touched by either pass.
                page.route("**/api/widgets", lambda route: routed_list(route, lambda data: data.__setitem__(
                    "ui_tabs", [tab for tab in data.get("ui_tabs", []) if tab.get("key") != f"{skill}:gauge"],
                )))
                emit_open(page)
                page.locator(card("gauge")).wait_for(state="detached", timeout=10_000)
                page.unroute("**/api/widgets")
                emit_open(page)
                page.locator(card("gauge")).wait_for(state="visible", timeout=10_000)
                page.wait_for_timeout(300)
                assert hang_frame_kept(page)
                assert frame_count(page, "auto") == 1
                assert page.locator("#widgets-list iframe").count() == 2

                # (3) Two quick launch-policy changes: the stored-map reads are held
                # here, so a second read in flight would be visible. With the writes
                # chained there is exactly one at a time, and both choices land.
                held: list = []

                def hold_reads(route):
                    if route.request.method == "GET":
                        held.append(route)
                    else:
                        route.continue_()

                def wait_held(count: int) -> None:
                    deadline = time.monotonic() + 5
                    while len(held) < count and time.monotonic() < deadline:
                        page.wait_for_timeout(50)
                    assert len(held) == count, len(held)

                page.route("**/api/ui/preferences", hold_reads)
                page.locator(f"{card('manual')} [data-widget-menu-trigger]").click()
                page.locator(f"{card('manual')} [data-widget-start-mode=\"auto\"]").click()
                page.locator(f"{card('hang')} [data-widget-menu-trigger]").click()
                page.locator(f"{card('hang')} [data-widget-start-mode=\"manual\"]").click()
                wait_held(1)
                page.wait_for_timeout(400)
                assert len(held) == 1, "the second change must wait for the first write"
                held.pop().continue_()
                wait_held(1)
                held.pop().continue_()
                # Both writes are on their way; stop holding reads before the
                # verification below issues its own GET.
                page.unroute("**/api/ui/preferences")
                page.wait_for_function(
                    """async ([manualKey, hangKey]) => {
                        const prefs = await (await fetch('/api/ui/preferences')).json();
                        const modes = prefs.widget_start_mode || {};
                        return modes[manualKey] === 'auto' && modes[hangKey] === 'manual' && Object.keys(modes).length === 2;
                    }""",
                    arg=[f"{skill}:manual", f"{skill}:hang"],
                    timeout=10_000,
                )
                wait_frame(page, "manual", True)
                assert page.locator(f"{card('manual')} [data-widget-start-mode=\"auto\"]").get_attribute("aria-checked") == "true"
                assert page.locator(f"{card('hang')} [data-widget-start-mode=\"manual\"]").get_attribute("aria-checked") == "true"
                assert hang_frame_kept(page), "Manual changes nothing until Stop"

                # (4) Escape closes the menu and returns focus to the ⋮ trigger.
                page.locator(f"{card('hang')} [data-widget-menu-trigger]").click()
                page.locator(f"{card('hang')} [data-widget-start-mode=\"manual\"]").wait_for(state="visible", timeout=5_000)
                assert page.evaluate("() => document.activeElement?.hasAttribute('data-widget-start-mode')")
                page.keyboard.press("Escape")
                page.locator(f"{card('hang')} [data-widget-start-mode=\"manual\"]").wait_for(state="hidden", timeout=5_000)
                assert page.evaluate(
                    "(selector) => document.activeElement === document.querySelector(`${selector} [data-widget-menu-trigger]`)",
                    card("hang"),
                )
                assert page.locator(f"{card('hang')} [data-widget-menu-trigger]").get_attribute("aria-expanded") == "false"

                _click_nav(page, "dashboard")
                page.wait_for_function("() => document.querySelectorAll('#widgets-list iframe').length === 0", timeout=5_000)
                assert page_errors == [], page_errors
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.mark.ui_browser
@pytest.mark.parametrize("browser_name", ("chromium", "webkit"))
def test_ui_smoke_widget_start_never_joins_a_stale_mount(direct_server_with_data, browser_name):
    """Cycle-B pre-fix set, CA-19 / A3-10 (F1) and CA-20 (F2): a start request
    never joins a mount that bails as stale. (a) Leave → return → leave → return
    inside the one-second acknowledgement window of the never-acking `hang`
    card (the four navigations are asserted to land inside it): the third
    entry's start used to join the second entry's mount, which bailed on its
    stale page generation, and the card stayed a "Stopped" facade until the
    next reconcile; it must end with one NEW frame, Running. (b) Owner Start on
    a stopped `auto` card with its module source held, then a revision bump
    through the reconnect reconcile: the fresh card (the old one is replaced at
    once — no frame to stop in order yet) used to join the old card's mount,
    which bailed on its detached node; it must end with one frame, Running.
    (c) The same bump on a `manual` card's held Start: the fresh card used to
    keep an EMPTY body (`settleStopped` bailed on the mount in flight); it must
    show its facade with Start."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    skill = _write_lifecycle_widget_extension(direct_server_with_data["data_dir"])
    page_errors: list[str] = []
    revision_overrides: dict[str, str] = {}
    module_hold: dict = {"pattern": "", "held": []}

    def card(tab_id: str) -> str:
        return f'[data-widget-key="{skill}:{tab_id}"]'

    def wait_active(page, active: bool) -> None:
        page.wait_for_function(
            "(active) => document.getElementById('page-widgets').classList.contains('active') === active",
            arg=active,
            timeout=5_000,
        )

    def wait_power(page, tab_id: str, text: str, timeout: int = 10_000) -> None:
        page.wait_for_function(
            """([selector, text]) => {
                const power = document.querySelector(`${selector} [data-widget-power]`);
                return power?.textContent === text && !power.disabled;
            }""",
            arg=[card(tab_id), text],
            timeout=timeout,
        )

    def wait_fresh_frame(page, tab_id: str, timeout: int = 15_000) -> None:
        # Exactly one frame, and not the one marked before the scenario.
        page.wait_for_function(
            """(selector) => {
                const frames = document.querySelectorAll(`${selector} iframe`);
                return frames.length === 1 && frames[0].__ouroOldFrame !== true;
            }""",
            arg=card(tab_id),
            timeout=timeout,
        )

    def state(page, tab_id: str) -> dict:
        return page.locator(card(tab_id)).first.evaluate(
            """node => ({
                power: node.querySelector('[data-widget-power]')?.textContent,
                status: node.querySelector('[data-widget-status]')?.textContent,
                statusHidden: node.querySelector('[data-widget-status]')?.hidden,
                iframes: node.querySelectorAll('iframe').length,
                facade: node.querySelectorAll('[data-widget-facade]').length,
                height: node.getBoundingClientRect().height,
            })"""
        )

    def toggle(page, enabled: bool) -> int:
        return page.evaluate(
            """async ([skill, enabled]) => (await fetch(`/api/skills/${encodeURIComponent(skill)}/toggle`, {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({enabled}),
            })).status""",
            [skill, enabled],
        )

    def emit_open(page) -> None:
        page.evaluate("() => { window.__ouroWs.emit('open', {previouslyConnected: true}); }")

    def widgets_handler(route) -> None:
        data = route.fetch().json()
        for tab in data.get("ui_tabs", []):
            if tab.get("key") in revision_overrides:
                tab["revision"] = revision_overrides[tab["key"]]
        route.fulfill(status=200, content_type="application/json", body=json.dumps(data))

    def module_handler(route) -> None:
        if module_hold["pattern"] and module_hold["pattern"] in route.request.url:
            module_hold["held"].append(route)
        else:
            route.continue_()

    def wait_held(page, count: int) -> None:
        deadline = time.monotonic() + 5
        while len(module_hold["held"]) < count and time.monotonic() < deadline:
            page.wait_for_timeout(50)
        assert len(module_hold["held"]) == count, len(module_hold["held"])

    def release_held() -> None:
        module_hold["pattern"] = ""
        while module_hold["held"]:
            module_hold["held"].pop().continue_()

    def start_held(page, tab_id: str) -> None:
        # Owner Start with the module source held: the mount is in flight.
        module_hold["pattern"] = "/module/widget.js"
        page.locator(f"{card(tab_id)} [data-widget-power]").click()
        wait_held(page, 1)
        starting = state(page, tab_id)
        assert starting["status"] == "Starting…" and starting["iframes"] == 0, starting

    def bump_revision(page, tab_id: str, marker: str) -> None:
        # The entry changed while its mount is in flight (no disposer, nothing to
        # stop in order): the reconcile replaces the card node synchronously.
        page.evaluate("(selector) => { document.querySelector(selector).__ouroOldCard = true; }", card(tab_id))
        revision_overrides[f"{skill}:{tab_id}"] = marker * 64
        emit_open(page)
        page.wait_for_function(
            """(selector) => {
                const cards = document.querySelectorAll(selector);
                return cards.length === 1 && cards[0].__ouroOldCard !== true;
            }""",
            arg=card(tab_id),
            timeout=10_000,
        )

    try:
        with sync_playwright() as pw:
            browser = getattr(pw, browser_name).launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            page.route("**/api/widgets", widgets_handler)
            page.route(f"**/api/extensions/{skill}/module/*", module_handler)
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                assert toggle(page, True) == 200
                # The hang card first in the key order: an entry's sync reaches its
                # start right after the list fetch, before any other mount.
                assert page.evaluate(
                    """async (order) => (await fetch('/api/ui/preferences', {method: 'POST',
                        headers: {'Content-Type': 'application/json'}, body: JSON.stringify({widget_order: order})})).status""",
                    [f"{skill}:hang", f"{skill}:auto", f"{skill}:manual", f"{skill}:gauge"],
                ) == 200
                page.click('[data-nav-page="widgets"]')
                for tab_id in ("manual", "auto", "hang", "gauge"):
                    page.locator(card(tab_id)).wait_for(state="visible", timeout=30_000)
                wait_power(page, "hang", "Stop")
                wait_power(page, "auto", "Stop")

                # (a) Four navigations inside the hang card's acknowledgement window.
                page.evaluate("(selector) => { document.querySelector(`${selector} iframe`).__ouroOldFrame = true; }", card("hang"))
                started = page.evaluate("performance.now()")
                for _ in range(2):
                    _click_nav(page, "dashboard")
                    wait_active(page, False)
                    page.wait_for_timeout(60)
                    _click_nav(page, "widgets")
                    wait_active(page, True)
                    page.wait_for_timeout(60)
                span = page.evaluate("performance.now()") - started
                assert span < 1000, f"the four navigations must land inside the 1 s ack window ({span:.0f} ms)"
                wait_fresh_frame(page, "hang")
                wait_power(page, "hang", "Stop", timeout=15_000)
                hang = state(page, "hang")
                assert hang["iframes"] == 1 and hang["facade"] == 0 and hang["status"] == "Running", hang
                wait_power(page, "auto", "Stop")
                assert page.locator("#widgets-list [data-widget-removed]").count() == 0

                # (b) Owner Start of a stopped auto card, source held, then its revision moves.
                page.locator(f"{card('auto')} [data-widget-power]").click()  # owner Stop
                wait_power(page, "auto", "Start")
                assert state(page, "auto")["iframes"] == 0
                start_held(page, "auto")
                bump_revision(page, "auto", "b")
                release_held()
                wait_fresh_frame(page, "auto")
                wait_power(page, "auto", "Stop", timeout=15_000)
                auto = state(page, "auto")
                assert auto["iframes"] == 1 and auto["facade"] == 0 and auto["status"] == "Running", auto
                assert page.locator(card("auto")).count() == 1
                assert page.locator("#widgets-list iframe").count() == 2  # auto + hang
                assert page.locator("#widgets-list [data-widget-removed]").count() == 0

                # (c) The same on a manual card: the fresh card ends with its facade.
                start_held(page, "manual")
                bump_revision(page, "manual", "c")
                release_held()
                page.locator(f"{card('manual')} [data-widget-facade]").wait_for(state="visible", timeout=15_000)
                wait_power(page, "manual", "Start", timeout=15_000)
                manual = state(page, "manual")
                assert manual["iframes"] == 0 and manual["facade"] == 1 and manual["statusHidden"] is True, manual
                assert manual["height"] > 300, manual  # the facade at the declared height, not an empty 83 px body
                assert page.locator(card("manual")).count() == 1
                assert page.locator("#widgets-list iframe").count() == 2

                _click_nav(page, "dashboard")
                page.wait_for_function("() => document.querySelectorAll('#widgets-list iframe').length === 0", timeout=5_000)
                assert page_errors == [], page_errors
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.mark.ui_browser
@pytest.mark.parametrize("browser_name", ("chromium", "webkit"))
def test_ui_smoke_widget_last_card_removal_evicts_session_state(direct_server_with_data, browser_name):
    """Final-gate finding WL-02: a card that disappears must leave nothing of
    itself behind, on BOTH removal paths. The keyed patch already evicted the
    declarative session state and the owner's page-session Stop; the rebuild
    branch — the one the patch never sees, when the LAST card leaves — did not.
    Stop the ``auto`` card, disable the only skill so the list goes empty, then
    re-enable it: the card must come back RUNNING, because the Stop that
    suppressed it belonged to a card that no longer exists. Before the fix it
    returned as a suppressed facade and only a window reload cleared it."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    skill = _write_lifecycle_widget_extension(direct_server_with_data["data_dir"])
    page_errors: list[str] = []

    def card(tab_id: str) -> str:
        return f'[data-widget-key="{skill}:{tab_id}"]'

    def toggle(page, enabled: bool) -> int:
        return page.evaluate(
            """async ([skill, enabled]) => (await fetch(`/api/skills/${encodeURIComponent(skill)}/toggle`, {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({enabled}),
            })).status""",
            [skill, enabled],
        )

    def wait_power(page, tab_id: str, text: str, timeout: int = 15_000) -> None:
        page.wait_for_function(
            """([selector, text]) => {
                const power = document.querySelector(`${selector} [data-widget-power]`);
                return power?.textContent === text && !power.disabled;
            }""",
            arg=[card(tab_id), text],
            timeout=timeout,
        )

    try:
        with sync_playwright() as pw:
            browser = getattr(pw, browser_name).launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                assert toggle(page, True) == 200
                page.click('[data-nav-page="widgets"]')
                page.locator(card("auto")).wait_for(state="visible", timeout=30_000)
                wait_power(page, "auto", "Stop")

                # The owner stops the auto card: the suppression is page-session
                # memory, keyed by this card.
                page.locator(f'{card("auto")} [data-widget-power]').click()
                wait_power(page, "auto", "Start")
                page.locator(f'{card("auto")} [data-widget-facade]').wait_for(state="visible", timeout=10_000)

                # The only skill goes away: every card leaves, which is the rebuild
                # branch rather than the keyed patch.
                assert toggle(page, False) == 200
                page.wait_for_function(
                    '(prefix) => document.querySelectorAll(`[data-widget-key^="${prefix}"]`).length === 0',
                    arg=f"{skill}:",
                    timeout=20_000,
                )
                assert page.locator("#widgets-list iframe").count() == 0

                # The same keys come back. The Stop belonged to a card that no
                # longer exists, so the auto card starts again on its own.
                assert toggle(page, True) == 200
                page.locator(card("auto")).wait_for(state="visible", timeout=20_000)
                wait_power(page, "auto", "Stop")
                page.wait_for_function(
                    '(selector) => document.querySelectorAll(`${selector} iframe`).length === 1',
                    arg=card("auto"),
                    timeout=15_000,
                )
                assert page.locator(f'{card("auto")} [data-widget-facade]').count() == 0
                assert page_errors == [], page_errors
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise
