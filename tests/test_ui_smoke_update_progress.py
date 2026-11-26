"""The actual Updates consumer stays informative through a pending apply."""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET, _emit_ws_frame
from tests.ui_console_errors import PageErrors

pytest_plugins = ("tests.test_ui_smoke_playwright",)


@pytest.mark.ui_browser
@pytest.mark.parametrize("width", [1280, 390])
def test_update_progress_pending_reopen_and_failure(direct_server, tmp_path, width):
    from playwright.sync_api import expect, sync_playwright

    plan = {
        "available": True, "kind": "clean", "local_dirty_count": 0,
        "base_sha": "a" * 40, "target_sha": "b" * 40,
        "code_conflict_paths": [], "doc_conflict_paths": [], "hot_code_paths": [],
    }
    status = {
        "managed": True, "check_ok": True, "available": True, "safe_to_apply": True,
        "current_version": "7.0.0", "latest_version": "7.0.0",
        "current_short_sha": "aaaaaaaa", "latest_short_sha": "bbbbbbbb",
        "update_tx": {"active": False}, "warnings": [],
    }
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(tmp_path)))
    evidence.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": width, "height": 900})
        pending_apply = []

        def switch_to(tab):
            # A phone's Dashboard shows its list or one of its pages (docs/DESIGN.md "One spine of depth"): the page's
            # Back brings the list, whose row opens the next page. A desk keeps the list beside the page.
            if width < 980:
                page.click("#page-dashboard [data-shell-back]")
            page.click(f'[data-dashboard-tab="{tab}"]')

        try:
            page.add_init_script(f"({_CAPTURE_TEST_SOCKET})();")
            def source_install_state(route):
                response = route.fetch()
                # This test owns pending-request recovery, not served-SHA
                # reload policy (covered by the WS reconnect unit tests). Use the
                # supported unversioned-source state during forced reconnect.
                route.fulfill(response=response, json={**response.json(), "sha": ""})

            page.route("**/api/state", source_install_state)
            page.route("**/api/update/status**", lambda route: route.fulfill(json=status))
            page.route("**/api/update/preflight", lambda route: route.fulfill(json={"merge_plan": plan}))
            page.route("**/api/update/apply", lambda route: pending_apply.append(route))
            page.goto(direct_server, wait_until="domcontentloaded")
            page.wait_for_selector("#page-chat")
            page.wait_for_function("window.__testSockets?.some(s => s.readyState === 1)")
            if width < 980:
                page.click("[data-mobile-nav-toggle]")
            page.click('[data-nav-page="dashboard"]')
            page.click('[data-dashboard-tab="updates"]')
            page.click("#btn-update-primary")
            page.click("[data-confirm-ok]")
            expect(page.locator("#updates-summary")).to_have_text("Applying the update…")
            page.screenshot(path=str(evidence / f"updates-{width}-before.png"), full_page=True)
            # Reconnect while the first apply is pending but the server has not
            # begun its observable executor yet. It must not offer another Apply.
            with page.expect_response("**/api/update/status"):
                page.evaluate("window.__testSockets[0].close()")
                page.wait_for_function("window.__testSockets.length > 1 && window.__testSockets.at(-1).readyState === 1")
            page.evaluate("() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")
            expect(page.locator("#updates-summary")).to_have_text("Applying the update…")
            expect(page.locator("#btn-update-primary")).to_be_disabled()
            page.evaluate("window.__testSockets = [window.__testSockets.at(-1)]")
            switch_to("logs")
            switch_to("updates")
            expect(page.locator("#updates-summary")).to_have_text("Applying the update…")
            expect(page.locator("#btn-update-primary")).to_be_disabled()
            status["update_progress"] = {
                "operation_id": "pending-apply", "generation": "server-a",
                "stage": "stopping_workers", "active": True,
                "stage_started_at": datetime.now(timezone.utc).isoformat(),
            }
            _emit_ws_frame(page, {"type": "update_progress_changed"})
            expect(page.locator("#updates-summary")).to_have_text("Stopping worker processes…")
            expect(page.locator("#updates-attempt")).to_be_hidden()
            expect(page.locator("#btn-update-primary")).to_be_disabled()
            page.screenshot(path=str(evidence / f"updates-{width}-after.png"), full_page=True)

            switch_to("logs")
            switch_to("updates")
            expect(page.locator("#updates-summary")).to_have_text("Stopping worker processes…")
            status["update_progress"].update(stage="checking")
            _emit_ws_frame(page, {"type": "update_progress_changed"})
            expect(page.locator("#updates-summary")).to_have_text("Checking the update and dependencies…")

            # A disconnected apply client cannot erase server-owned progress.
            assert pending_apply, "the original apply stayed pending while progress changed"
            for route in pending_apply:
                route.abort()
            pending_apply.clear()
            page.reload(wait_until="domcontentloaded")
            page.wait_for_function("window.__testSockets?.some(s => s.readyState === 1)")
            if width < 980:
                page.click("[data-mobile-nav-toggle]")
            page.click('[data-nav-page="dashboard"]')
            page.click('[data-dashboard-tab="updates"]')
            expect(page.locator("#updates-summary")).to_have_text("Checking the update and dependencies…")
            status["update_progress"].update(active=False, result="failed", restart_required=True, error="A service could not be stopped.")
            _emit_ws_frame(page, {"type": "update_progress_changed"})
            expect(page.locator("#updates-summary")).to_have_text("The update did not complete.")
            expect(page.locator("#btn-update-primary")).to_have_text("Restart now")
            page.evaluate("() => Promise.all(document.getAnimations().filter(a => a.effect?.getTiming().iterations !== Infinity).map(a => a.finished.catch(() => {})))")
            page.screenshot(path=str(evidence / f"updates-{width}-failure.png"), full_page=True)
            assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
        finally:
            for route in pending_apply:
                route.abort()
            browser.close()


@pytest.mark.ui_browser
def test_successful_recheck_keeps_release_tags_and_the_last_attempt(direct_server):
    """A check refreshes availability: it neither clears the retained attempt nor loses its tags. Up to date, the
    headline's check is Check and describe, one request whose answer carries that check's status; a status frame
    during it neither ends its Checking… nor erases the releases its answer brings."""
    from playwright.sync_api import expect, sync_playwright

    status = {
        "managed": True, "check_ok": True, "available": False, "safe_to_apply": False,
        "current_version": "7.0.0", "current_short_sha": "aaaaaaaa",
        "update_tx": {"active": False}, "official_tags": [],
        "update_progress": {"operation_id": "failed-attempt", "current_generation": True, "active": False,
                            "result": "failed", "outcome": "not_applied", "reason": "release_moved",
                            "work_stop": "none", "error": "The target changed."},
    }
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        pending_check = []
        try:
            page.add_init_script(f"({_CAPTURE_TEST_SOCKET})();")
            page.route("**/api/update/status**", lambda route: route.fulfill(json=status))
            page.route("**/api/update/check", lambda route: pending_check.append(route))
            page.route("**/api/update/description", lambda route: pending_check.append(route))
            page.goto(direct_server, wait_until="domcontentloaded")
            page.wait_for_selector("#page-chat")
            page.wait_for_function("window.__testSockets?.some(s => s.readyState === 1)")
            page.click('[data-nav-page="dashboard"]')
            page.click('[data-dashboard-tab="updates"]')
            expect(page.locator("#updates-attempt-headline")).to_have_text("No update was applied.")
            expect(page.locator("#btn-update-primary")).to_have_text("Check and describe")
            page.click("#btn-update-primary")
            expect(page.locator("#btn-update-primary")).to_have_text("Checking…")
            expect(page.locator("#btn-update-primary")).to_be_disabled()
            # A concurrent invalidation during the check; its passive read carries no tags.
            _emit_ws_frame(page, {"type": "update_progress_changed"})
            expect(page.locator("#btn-update-primary")).to_have_text("Checking…")
            assert len(pending_check) == 1 and json.loads(pending_check[0].request.post_data) == {"mode": "latest"}
            pending_check.pop().fulfill(json={"status": "no_update", "checked_key": None,
                                              "update": {**status, "official_tags": [{"tag": "v7.0.0-test", "sha": "a" * 40}]}})
            expect(page.locator("#updates-official-tags")).to_contain_text("v7.0.0-test")
            expect(page.locator("#btn-update-primary")).to_have_text("Check and describe")
            expect(page.locator("#updates-attempt-headline")).to_have_text("No update was applied.")
            expect(page.locator("#updates-attempt-text")).to_contain_text("The update changed after it was prepared.")
            expect(page.locator("#updates-attempt-text")).not_to_contain_text("stopped")
        finally:
            for route in pending_check:
                route.abort()
            browser.close()


def _sidebar(page, width):
    """The sidebar the indicator lives in: always there on a desk, the open drawer on a phone."""
    if width < 720 and not page.evaluate("document.body.classList.contains('nav-drawer-open')"):
        back = page.locator('.page.active [data-shell-back]:visible')
        if back.count():
            back.click()
        page.locator(".page.active [data-mobile-nav-toggle]:visible").click()
        page.wait_for_function("document.getElementById('content').inert === true")


def _settled(page):
    """Every finite animation and transition has ended: colours are read where they rest, never mid-transition."""
    page.evaluate("() => Promise.all(document.getAnimations().filter(a => a.effect?.getTiming().iterations !== Infinity)"
                  ".map(a => a.finished.catch(() => {})))")


def _reload(page, width):
    page.reload(wait_until="domcontentloaded")
    page.wait_for_function("window.__testSockets?.some(s => s.readyState === 1)")
    _sidebar(page, width)


_NOTICE = """() => {
    const box = (el) => { const r = el.getBoundingClientRect(); return { x: r.x, y: r.y, r: r.right, b: r.bottom, w: r.width, h: r.height }; };
    const pill = document.getElementById('update-pill');
    const close = document.querySelector('.update-notice-close');
    const sidebar = document.getElementById('primary-sidebar').getBoundingClientRect();
    const token = (name, property) => { const p = close.appendChild(document.createElement('i')); p.style[property] = `var(${name})`;
        const v = getComputedStyle(p)[property]; p.remove(); return v; };
    const c = getComputedStyle(close);
    // A control's scroll size counts its touch reach: under a coarse pointer the shared hit area (ui.css .ui-hit-area,
    // .btn::after) is an absolute pseudo-element out to 44 px, so a 28 px capsule reads scrollWidth 35 over clientWidth
    // 26 in WebKit and Chromium alike with nothing drawn cut. What the capsule draws is read with that reach set aside;
    // the reach itself is read where it can really be cut, against the clip of the column that scrolls around it.
    const spills = (el) => el.scrollWidth > el.clientWidth + 1 || el.scrollHeight > el.clientHeight + 1;
    const haloed = spills(pill);
    const bare = document.head.appendChild(document.createElement('style'));
    bare.textContent = '#update-pill::after, .update-notice-close::after { content: none !important; }';
    const clipped = spills(pill) || spills(close);
    bare.remove();
    const side = document.getElementById('primary-sidebar'), sr = side.getBoundingClientRect();
    const scroller = { x: sr.left + side.clientLeft, r: sr.left + side.clientLeft + side.clientWidth,
                       y: sr.top + side.clientTop, b: sr.top + side.clientTop + side.clientHeight };
    const reach = (el) => { const r = el.getBoundingClientRect(), a = getComputedStyle(el, '::after');
        const ring = getComputedStyle(el), out = parseFloat(ring.outlineOffset) + parseFloat(ring.outlineWidth);
        const drawn = { x: r.left - out, r: r.right + out, y: r.top - out, b: r.bottom + out };
        if (a.content === 'none' || a.content === 'normal') return drawn;
        const x = r.left + el.clientLeft + parseFloat(a.left), y = r.top + el.clientTop + parseFloat(a.top);
        const hit = { x, r: x + parseFloat(a.width), y, b: y + parseFloat(a.height) };
        return { x: Math.min(drawn.x, hit.x), r: Math.max(drawn.r, hit.r), y: Math.min(drawn.y, hit.y), b: Math.max(drawn.b, hit.b) }; };
    const within = (q) => q.x >= scroller.x - 0.5 && q.r <= scroller.r + 0.5 && q.y >= scroller.y - 0.5 && q.b <= scroller.b + 0.5;
    const reaches = { pill: reach(pill), close: reach(close) };
    return { pill: box(pill), close: box(close), sidebar: { x: sidebar.x, r: sidebar.right }, scroller, reaches, haloed,
             reachInside: within(reaches.pill) && within(reaches.close),
             clipped, overflow: document.documentElement.scrollWidth - innerWidth,
             active: document.activeElement?.id || String(document.activeElement?.className || ''),
             closeColor: c.color, pillColor: getComputedStyle(pill).color,
             // The close is the shared neutral capsule (ui.css .btn.btn-default), never a copy of its paint.
             closePart: close.classList.contains('btn') && close.classList.contains('btn-default'),
             closePaint: [c.backgroundColor === token('--surface-capsule', 'backgroundColor'),
                          c.boxShadow === token('--elev-rest', 'boxShadow'), c.borderTopLeftRadius] };
}"""

# The footer's capsule glyph (its words are its name, read but not drawn): which state it says, its
# geometry inside the capsule, its ink and its picture.
_GLYPH = """() => { const pill = document.getElementById('update-pill'), g = pill.querySelector('.update-pill-glyph');
    const svg = g.querySelector('svg'), s = svg.getBoundingClientRect(), p = pill.getBoundingClientRect();
    const w = pill.querySelector('.update-pill-words').getBoundingClientRect();
    return { says: g.dataset.says, picture: svg.innerHTML, color: getComputedStyle(g).color, w: s.width, h: s.height,
             inside: s.left >= p.left - 0.5 && s.right <= p.right + 0.5 && s.top >= p.top - 0.5 && s.bottom <= p.bottom + 0.5,
             centred: Math.abs((s.left + s.right - p.left - p.right) / 2) <= 0.5 && Math.abs((s.top + s.bottom - p.top - p.bottom) / 2) <= 0.5,
             words: w.width * w.height }; }"""


_GLYPH_SIDE = 16      # the footer capsule's glyph square in CSS px
_GLYPH_SLACK = 0.05   # an engine's rectangle is float-rounded: Chromium reports 15.999998092651367 for 16


def _drawn(page, says):
    """The capsule draws the glyph of `says`, whole and centred in it, its words unseen. Only the capsule's own
    transitions are waited on (bounded): a page behind it, as the Updates panel in WebKit, may hold one that never ends."""
    page.wait_for_function("() => (document.getElementById('update-pill')?.getAnimations({subtree: true}) || [])"
                           ".every((a) => a.playState !== 'running')")
    glyph = page.evaluate(_GLYPH)
    # The side is 16 px within a subpixel tolerance that still catches a real change of size (15 or 17 px) or a
    # glyph that no longer squares up; the state and the rest of the geometry keep their exact checks.
    assert glyph["says"] == says, glyph
    assert glyph["w"] == pytest.approx(_GLYPH_SIDE, abs=_GLYPH_SLACK), glyph
    assert glyph["h"] == pytest.approx(_GLYPH_SIDE, abs=_GLYPH_SLACK), glyph
    assert abs(glyph["w"] - glyph["h"]) <= _GLYPH_SLACK, glyph
    assert glyph["inside"] and glyph["centred"] and glyph["words"] <= 1, glyph
    return glyph


# Where the focus stands, read against the sidebar the page composed (pages.js renderNavRail, index.html's foot; owner
# 24.11 «Рельс снаружи», 25.11): the version row is the column's own foot under the Projects list, and Main's card,
# which follows the rail, is what the retired "last [data-nav-page]" fallback picked.
_FOCUS = """() => { const a = document.activeElement, row = document.querySelector('#primary-sidebar > .sidebar-foot-meta'),
    rail = document.querySelector('#primary-sidebar .nav-rail');
    const places = [...document.querySelectorAll('#primary-sidebar [data-nav-page]')];
    return { id: a?.id || '', page: a?.dataset?.navPage || '', inFoot: Boolean(a?.closest?.('.sidebar-foot-meta')),
             inRail: Boolean(a?.closest?.('.nav-rail')),
             beside: Boolean(row) && a === [...rail.querySelectorAll('[data-nav-page]')].pop(),
             last: a === places.pop() }; }"""
# The fallback after a focused footer control disappears: the rail's Settings, never Main's card.
_BESIDE = {"id": "", "page": "settings", "inFoot": False, "inRail": True, "beside": True, "last": False}


def _rested(page, selector):
    """The finite transitions under `selector` have ended, or two seconds passed. Never the whole document's: a page
    behind the rail (the Dashboard, once opened) may hold one that never ends, and `_settled` would then wait forever."""
    page.evaluate("""(selector) => { const root = document.querySelector(selector);
        const finite = (root?.getAnimations({subtree: true}) || []).filter((a) => a.effect?.getTiming().iterations !== Infinity);
        return Promise.race([Promise.all(finite.map((a) => a.finished.catch(() => {}))),
                             new Promise((resolve) => setTimeout(resolve, 2000))]); }""", selector)


# The close's focus ring against its capsule: the close stands beside the capsule on the version's own line (the
# column's foot, owner 25.11), and the ring's outer edge stays clear of the capsule it stands beside.
_RING = """() => { const c = document.querySelector('.update-notice-close'), s = getComputedStyle(c);
    const p = document.getElementById('update-pill').getBoundingClientRect();
    const r = c.getBoundingClientRect(), out = parseFloat(s.outlineOffset) + parseFloat(s.outlineWidth);
    return { offset: s.outlineOffset, style: s.outlineStyle, clear: r.left - out >= p.right - 0.5 }; }"""


@pytest.mark.ui_browser
@pytest.mark.parametrize("width", [1100, 390])
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_update_indicator_failure_closes_on_its_notice_and_success_stays_in_the_panel(direct_server, tmp_path, width, engine):
    """Owners 3A and 1B (#1818): a running stage and a failed or unconfirmed outcome stay on the existing capsule, the
    last until the close beside it hides it (this browser, this attempt); a confirmed success never wears it and never
    masks a newer version, its record being the Dashboard panel's alone; a needed restart keeps its words and draws its
    own glyph, never the stage's (the foot shows the capsule's glyph alone). A reviewed resolution at work is a stage;
    stopped short (its gate blocked, its local work awaiting recovery) it draws the bell in the stage's neutral ink and
    opens a panel that names the recovery; its merge committed, it is the check's stage until its own check passed, then
    that same needed restart, which the panel offers.
    None of them closes. A row control that goes away while it holds the focus hands it to the rail's Settings,
    never to Main's card. Every press only navigates: nothing prepares, applies, checks or restarts."""
    from playwright.sync_api import expect, sync_playwright

    stamp = lambda minutes=0: (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat()
    resolver = {"operation_id": "attempt-r", "generation": "server-a", "current_generation": False, "active": False,
                "stage": "preparing", "result": "assisted_started", "outcome": "", "reason": "", "error": "",
                "blockers": [], "tx_attempt_id": "tx-r", "started_at": stamp()}
    tx = lambda phase, **extra: {"active": True, "phase": phase, "task_id": "", "restart_required": False,
                                 "attempt_id": "tx-r", **extra}
    tab = "Alt+Tab" if engine == "webkit" else "Tab"  # WebKit tabs to buttons with Option held
    refusal = {"operation_id": "attempt-a", "current_generation": True, "active": False, "result": "failed",
               "outcome": "not_applied", "outcome_at": datetime.now(timezone.utc).isoformat(),
               "stage": "draining_writers", "reason": "update_writer_fence_blocked",
               "blockers": ["active:Main conversation"], "work_stop": "none", "concluded_by": "executor", "restart_required": False,
               "error": "Could not prove every repository writer stopped.", "base_sha": "a" * 40, "target_sha": "b" * 40}
    status = {"managed": True, "check_ok": True, "available": True, "safe_to_apply": True, "local_changes": "absent",
              "current_version": "7.0.0", "latest_version": "7.1.0",
              "current_short_sha": "aaaaaaaa", "latest_short_sha": "bbbbbbbb",
              "update_tx": {"active": False}, "warnings": [],
              "update_progress": {**refusal, "active": True, "outcome": "", "result": "",
                                  "stage": "stopping_workers", "work_stop": "invoked"}}
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(tmp_path)))
    evidence.mkdir(parents=True, exist_ok=True)
    actions = []
    theme = "dark" if width == 1100 else "light"

    def open_page(browser):
        page = browser.new_page(viewport={"width": width, "height": 750}, color_scheme=theme, has_touch=width < 980)
        page.add_init_script(f"({_CAPTURE_TEST_SOCKET})();")
        errors = PageErrors(page, engine)
        page.on("request", lambda request: actions.append(request.url) if request.method == "POST" and any(
            path in request.url for path in ("/api/update/", "/api/command", "/api/git/")) else None)
        page.route("**/api/update/status**", lambda route: route.fulfill(json=status))
        page.goto(direct_server, wait_until="domcontentloaded")
        page.wait_for_selector("#page-chat")
        page.wait_for_function("window.__testSockets?.some(s => s.readyState === 1)")
        return page, errors

    def conclude(page, **fields):
        status["update_progress"] = {**refusal, **fields}
        _emit_ws_frame(page, {"type": "update_progress_changed"})

    def show(page, event="update_progress_changed", **fields):
        status.update(fields)
        _emit_ws_frame(page, {"type": event})

    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch(headless=True)
        try:
            page, errors = open_page(browser)
            _sidebar(page, width)
            pill, close = page.locator("#update-pill"), page.locator(".update-notice-close")
            check, settings = page.locator("#nav-version-check"), page.locator('.nav-rail [data-nav-page="settings"]')
            foot = page.locator("#primary-sidebar .sidebar-foot-meta")
            # A stage names itself and has no close: it ends by itself, with the attempt.
            expect(pill).to_have_text("Stopping work…")
            expect(close).to_be_hidden()
            stage = _drawn(page, "stage")
            page.locator("#primary-sidebar .sidebar-foot-meta").screenshot(path=str(evidence / f"update-foot-{engine}-{width}-stage.png"))
            # A failure stays, with its own close beside the capsule on the version's one line, in the capsule's ink.
            conclude(page)
            expect(pill).to_have_text("Update not applied")
            expect(pill).to_have_attribute("data-tone", "error")
            expect(close).to_be_visible()
            _settled(page)
            failure = _drawn(page, "outcome")
            assert failure["color"] == page.evaluate(_NOTICE)["pillColor"], ("the warning in the failure's ink", failure)
            notice = page.evaluate(_NOTICE)
            middle = lambda box: box["y"] + box["h"] / 2
            assert notice["close"]["x"] >= notice["pill"]["r"] - 0.5, ("the close stands beside the capsule", notice)
            assert abs(middle(notice["close"]) - middle(notice["pill"])) <= 0.5, ("on its line", notice)
            assert notice["close"]["w"] >= 24 and notice["close"]["h"] >= 24, ("a real pointer target", notice)
            assert notice["pill"]["r"] <= notice["sidebar"]["r"] and not notice["clipped"] and notice["overflow"] <= 0, notice
            # Each one's touch reach and focus ring stay inside the column's clip: the scroller would cut them there.
            assert notice["reachInside"], ("a finger's reach and the ring are never cut by the column", notice)
            assert notice["closeColor"] == notice["pillColor"], ("the close takes the capsule's ink", notice)
            assert notice["closePart"] and notice["closePaint"] == [True, True, "999px"], ("the shared capsule", notice)
            # Under a coarse pointer the drawn 24 px circle gains the shared 44 px hit area (`.ui-hit-area`).
            coarse = page.evaluate("matchMedia('(pointer: coarse)').matches")
            if coarse:
                hit = page.evaluate("""() => { const c = document.querySelector('.update-notice-close'), s = getComputedStyle(c, '::after');
                    const p = document.getElementById('update-pill'), ps = getComputedStyle(p, '::after');
                    return [parseFloat(s.width), parseFloat(s.height),
                            c.getBoundingClientRect().left + parseFloat(s.left)
                            - (p.getBoundingClientRect().right - parseFloat(ps.right))]; }""")
                assert hit[0] >= 44 and hit[1] >= 44, ("a touch target of at least 44 px", hit)
                assert hit[2] >= -0.5, ("its reach stops short of the capsule's beside it", hit)
                # The capsule's own scroll size counts that reach (the old reading); set aside, nothing it draws is cut.
                assert notice["haloed"] and not notice["clipped"], notice
                # A reach the column would cut is caught (falsified right here).
                wide = page.add_style_tag(content=".update-notice-close::after { inset: -30px !important; }")
                assert page.evaluate(_NOTICE)["reachInside"] is False, "a reach past the column's clip is caught"
                wide.evaluate("el => el.remove()")
            # A capsule narrower than its glyph is caught as cut (falsified right here).
            narrow = page.add_style_tag(content=".update-notice > .update-pill { width: 12px !important; }")
            assert page.evaluate(_NOTICE)["clipped"] is True, "a glyph cut by its capsule is caught"
            narrow.evaluate("el => el.remove()")
            page.locator("#primary-sidebar .sidebar-foot-meta").screenshot(path=str(evidence / f"update-foot-{engine}-{width}-failure.png"))
            # The capsule still goes to the one panel; the panel keeps the saved result and has no remote hide.
            pill.click()
            expect(page.locator("#dashboard-panel-updates")).to_be_visible()
            expect(page.locator("#updates-attempt-headline")).to_have_text("No update was applied.")
            expect(page.locator("#updates-attempt-text")).to_have_text(
                "Work was still finishing: Main conversation. This attempt stopped no running work. Try again when it finishes.")
            expect(page.locator("#btn-update-primary")).to_have_text("Update to 7.1.0")
            page.locator("#updates-attempt-details > summary").click()
            expect(page.locator("#updates-attempt-facts")).to_contain_text("update_writer_fence_blocked")
            assert page.locator("#updates-attempt-dismiss").count() == 0, "the hide lives on the notice, never on another page"
            _sidebar(page, width)
            # Keyboard: Tab from the capsule reaches the close, Enter hides this notice and focus lands on what is left.
            pill.focus()
            page.keyboard.press(tab)
            assert "update-notice-close" in page.evaluate(_NOTICE)["active"].split(), "the close follows the capsule in tab order"
            # The one focus ring (DESIGN §3 "Focus") around the close, clear of the capsule beside it, which a ring drawn
            # farther out crosses (falsified right here).
            ring = page.evaluate(_RING)
            assert ring == {"offset": "2px", "style": "solid", "clear": True}, ring
            outer = page.add_style_tag(content=".update-notice-close:focus-visible { outline-offset: 30px !important; }")
            assert page.evaluate(_RING)["clear"] is False, "a ring farther out would cross the capsule"
            outer.evaluate("el => el.remove()")
            box = page.locator(".update-notice").bounding_box()
            page.screenshot(path=str(evidence / f"update-foot-{engine}-{width}-focus.png"), clip={
                "x": max(0, box["x"] - 12), "y": max(0, box["y"] - 12), "width": box["width"] + 24, "height": box["height"] + 24})
            page.keyboard.press("Enter")
            expect(pill).to_have_text("Update")
            expect(close).to_be_hidden()
            assert page.evaluate(_NOTICE)["active"] == "update-pill", "focus never falls to the page"
            expect(page.locator("#updates-attempt-headline")).to_have_text("No update was applied.")
            # Reload: this browser remembers the hide for that attempt.
            _reload(page, width)
            expect(pill).to_have_text("Update")
            expect(close).to_be_hidden()
            # The next attempt is a new notice, closable; an unconfirmed outcome stays beside a newer version, and
            # pressing the close with a pointer hides it the same way.
            conclude(page, operation_id="attempt-b", outcome="unknown", reason="no_completion_record", work_stop="", blockers=[])
            expect(pill).to_have_text("Update outcome unconfirmed")
            expect(pill).to_have_attribute("data-tone", "neutral")
            expect(close).to_be_visible()
            box = pill.bounding_box()
            assert box and box["x"] >= 0 and box["x"] + box["width"] <= width, box
            _settled(page)
            page.screenshot(path=str(evidence / f"update-indicator-{engine}-{width}-unconfirmed.png"))
            # Another browser has no such memory: the same record shows its notice there.
            other = browser.new_context(viewport={"width": width, "height": 750}, color_scheme=theme)
            try:
                second = other.new_page()
                second.add_init_script(f"({_CAPTURE_TEST_SOCKET})();")
                second.route("**/api/update/status**", lambda route: route.fulfill(json=status))
                second.goto(direct_server, wait_until="domcontentloaded")
                second.wait_for_function("window.__testSockets?.some(s => s.readyState === 1)")
                expect(second.locator("#update-pill")).to_have_text("Update outcome unconfirmed")
            finally:
                other.close()
            close.click()
            expect(pill).to_have_text("Update")
            expect(close).to_be_hidden()
            # A confirmed success wears nothing: no capsule, no toast, no timer; a newer version still shows.
            conclude(page, operation_id="attempt-c", outcome="applied", result="applied", concluded_by="boot", blockers=[],
                     work_stop="", reason="")
            expect(pill).to_have_text("Update")
            expect(pill).not_to_have_attribute("data-tone", "ok")
            expect(close).to_be_hidden()
            status["available"] = False
            _emit_ws_frame(page, {"type": "update_status_ready"})
            expect(pill).to_be_hidden()
            page.wait_for_timeout(300)
            assert page.locator(".toast").count() == 0 and page.locator("#update-pill:visible").count() == 0
            _settled(page)
            page.screenshot(path=str(evidence / f"update-indicator-{engine}-{width}-success-silent.png"))
            _reload(page, width)
            expect(pill).to_be_hidden()
            # An accepted channel save hides the focused check; its status read shows it again without stealing
            # focus. An unmanaged install turns the check into plain text.
            expect(check).to_be_visible()
            check.focus()
            assert page.evaluate(_FOCUS)["id"] == "nav-version-check", page.evaluate(_FOCUS)
            hidden = page.evaluate("""() => { window.dispatchEvent(new CustomEvent('ouro:settings-updated',
                { detail: { reason: 'setting saved', source: 'settings-row', row: 'OUROBOROS_UPDATE_CHANNEL',
                            page: 'dashboard:updates', keys: ['OUROBOROS_UPDATE_CHANNEL'] } }));
                return document.getElementById('nav-version-check').hidden; }""")
            assert hidden is True and page.evaluate(_FOCUS) == _BESIDE, page.evaluate(_FOCUS)
            expect(check).to_be_visible()
            assert page.evaluate(_FOCUS) == _BESIDE, page.evaluate(_FOCUS)
            check.focus()
            show(page, event="update_status_ready", managed=False)
            expect(check).to_be_hidden()
            assert page.evaluate(_FOCUS) == _BESIDE, page.evaluate(_FOCUS)
            show(page, event="update_status_ready", managed=True)
            expect(check).to_be_visible()
            # Its record is the Dashboard panel's, untouched by any of this.
            page.click('[data-nav-page="dashboard"]')
            page.click('[data-dashboard-tab="updates"]')
            expect(page.locator("#updates-attempt-headline")).to_have_text("Update completed.")
            expect(page.locator("#updates-attempt-text")).to_contain_text("The restarted server confirmed this update.")
            page.screenshot(path=str(evidence / f"update-panel-{engine}-{width}-success-record.png"), full_page=True)
            # A needed restart keeps its words and has no close: it is a stage until the restart ends it.
            _sidebar(page, width)
            conclude(page, operation_id="attempt-d", active=False, outcome="", result="restart_required", restart_required=True,
                     stage="stopping_services", blockers=["service:mcp"], work_stop="invoked")
            expect(pill).to_have_text("Restart needed")
            expect(close).to_be_hidden()
            # The capsule's glyph must distinguish a needed step from work still in progress.
            restart = _drawn(page, "restart")
            assert restart["picture"] != stage["picture"] and restart["color"] != stage["color"], (restart, stage)
            page.locator("#primary-sidebar .sidebar-foot-meta").screenshot(path=str(evidence / f"update-foot-{engine}-{width}-restart.png"))
            # A failed retention write remains visible in the preserved Dashboard record, including after restart is needed.
            if width < 720:
                page.keyboard.press("Escape")
                page.wait_for_function("document.getElementById('content').inert === false")
                # The section's Back led to its list before navigation; its selected row reopens the record.
                page.click('[data-dashboard-tab="updates"]')
            status['update_progress']['retention_unconfirmed'] = True
            _emit_ws_frame(page, {'type': 'update_progress_changed'})
            expect(page.locator('#btn-update-primary')).to_have_text('Restart now')
            expect(page.locator('#updates-attempt-text')).to_be_visible()
            expect(page.locator('#updates-attempt-text')).to_contain_text('could not be saved')
            conclude(page, operation_id='attempt-e', outcome='unknown', reason='no_completion_record',
                     work_stop='', blockers=[], retention_unconfirmed=True)
            expect(page.locator('#updates-attempt-headline')).to_have_text('The last update outcome is unconfirmed.')
            expect(page.locator('#updates-attempt-text')).to_contain_text('could not be saved')
            assert page.locator('#dashboard-panel-updates').evaluate('el => el.scrollWidth <= el.clientWidth')

            # A reviewed resolution at work is a stage; its gate blocked, it stops short under the bell in the stage's
            # neutral ink, with no close and the version plain.
            _sidebar(page, width)
            show(page, update_progress=resolver, update_tx=tx("assisted_resolution"))
            expect(pill).to_have_text("Resolving update…")
            expect(close).to_be_hidden()
            show(page, update_tx=tx("gate_blocked"))
            expect(pill).to_have_text("Update not concluded")
            expect(pill).to_have_attribute("data-tone", "neutral")
            blocked = _drawn(page, "unresolved")
            assert blocked["picture"] != stage["picture"] and blocked["color"] == stage["color"], (blocked, stage)
            expect(close).to_be_hidden()
            expect(check).to_be_hidden()
            foot.screenshot(path=str(evidence / f"update-foot-{engine}-{width}-unresolved.png"))
            # Enter on the focused capsule opens the recovery panel without offering Restart or Apply.
            pill.focus()
            assert page.evaluate(_FOCUS)["id"] == "update-pill", page.evaluate(_FOCUS)
            page.keyboard.press("Enter")
            expect(page.locator("#updates-summary")).to_have_text("An update attempt stopped mid-flight and needs recovery.")
            expect(page.locator("#updates-hint")).to_contain_text("Quit and reopen the desktop app")
            expect(page.locator("#btn-update-primary")).to_have_text("Check again")
            # On a phone the drawer closes and the room hands over to the Dashboard first: the headline is measured on
            # screen once that motion is over.
            page.wait_for_function("() => !document.body.classList.contains('nav-drawer-open')"
                                   " && document.getElementById('content').inert === false"
                                   " && document.querySelector('.page.active')?.id === 'page-dashboard'")
            _rested(page, "#content")
            headline = page.locator("#updates-summary").bounding_box()
            assert headline and headline["x"] >= 0 and headline["x"] + headline["width"] <= width + 0.5, headline
            assert 0 <= headline["y"] and headline["y"] + headline["height"] <= 750, headline
            # Its merge committed (pending_boot_smoke) and its own check runs: the check's stage, drawn as the first
            # stage was, no restart and no close. Once that check passed it is the needed restart, drawn as attempt-d's
            # was, still with no close; its press opens the panel's Restart now, which says the state once.
            _sidebar(page, width)
            show(page, update_tx=tx("pending_boot_smoke", pre_restart_smoke="pending"))
            expect(pill).to_have_text("Checking update…")
            checking = _drawn(page, "stage")
            assert (checking["picture"], checking["color"]) == (stage["picture"], stage["color"]), checking
            expect(close).to_be_hidden()
            show(page, update_tx=tx("pending_boot_smoke", pre_restart_smoke="passed"))
            expect(pill).to_have_text("Restart needed")
            ready = _drawn(page, "restart")
            assert (ready["picture"], ready["color"]) == (restart["picture"], restart["color"]), ("one look", ready, restart)
            page.wait_for_timeout(300)
            expect(close).to_be_hidden()
            foot.screenshot(path=str(evidence / f"update-foot-{engine}-{width}-assisted-restart.png"))
            pill.click()
            expect(page.locator("#updates-summary")).to_have_text("An update is waiting for a restart to finish.")
            expect(page.locator("#btn-update-primary")).to_have_text("Restart now")
            expect(page.locator("#updates-attempt-headline")).to_be_hidden()
            _rested(page, "#content")
            page.screenshot(path=str(evidence / f"update-panel-{engine}-{width}-assisted-restart.png"), full_page=True)
            # Its local work awaiting recovery defers that restart: the bell again, no close.
            _sidebar(page, width)
            show(page, update_tx=tx("pending_boot_smoke", pre_restart_smoke="passed", local_work_recovery=True))
            expect(pill).to_have_text("Update not concluded")
            _drawn(page, "unresolved")
            expect(close).to_be_hidden()

            # A notice closed with nothing left in the row (an install that offers no check), and a capsule that held the
            # focus and goes with no check to take it: the focus lands on the rail's Settings, never on Main's card.
            show(page, event="update_status_ready", managed=False, update_tx={"active": False},
                 update_progress={**refusal, "operation_id": "attempt-u", "outcome_at": stamp(2)})
            expect(pill).to_have_text("Update not applied")
            expect(close).to_be_visible()
            pill.focus()
            page.keyboard.press(tab)
            assert "update-notice-close" in page.evaluate("document.activeElement.className").split()
            page.keyboard.press("Enter")
            expect(pill).to_be_hidden()
            assert page.evaluate(_FOCUS) == _BESIDE, page.evaluate(_FOCUS)
            show(page, update_progress={"operation_id": "attempt-s", "current_generation": True, "active": True,
                                        "stage": "applying", "started_at": stamp(3)})
            expect(pill).to_have_text("Applying update…")
            pill.focus()
            assert page.evaluate(_FOCUS)["id"] == "update-pill", page.evaluate(_FOCUS)
            show(page, update_progress={"operation_id": "attempt-s", "current_generation": True, "active": False,
                                        "result": "applied", "outcome": "applied", "concluded_by": "boot",
                                        "started_at": stamp(3), "outcome_at": stamp(4)})
            expect(pill).to_be_hidden()
            assert page.evaluate(_FOCUS) == _BESIDE, page.evaluate(_FOCUS)
            _rested(page, "#primary-sidebar .nav-rail")
            box = settings.bounding_box()
            page.screenshot(path=str(evidence / f"update-foot-{engine}-{width}-focus-beside.png"), clip={
                "x": max(0, box["x"] - 16), "y": max(0, box["y"] - 16), "width": box["width"] + 32, "height": box["height"] + 120})
            assert actions == [], actions
            assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
            assert errors.unexpected() == [], errors
        finally:
            browser.close()


@pytest.mark.ui_browser
@pytest.mark.parametrize('failed_stage', ['preflight', 'apply'])
def test_definite_failed_request_releases_local_phase(direct_server, failed_stage):
    from playwright.sync_api import sync_playwright, expect
    status = {'managed': True, 'check_ok': True, 'available': True, 'safe_to_apply': True,
              'current_version': '7.0.0', 'latest_version': '7.0.0',
              'current_short_sha': 'aaaaaaaa', 'latest_short_sha': 'bbbbbbbb',
              'update_tx': {'active': False}, 'update_progress': {}, 'warnings': []}
    plan = {'available': True, 'kind': 'clean', 'local_dirty_count': 0,
            'base_sha': 'a' * 40, 'target_sha': 'b' * 40,
            'code_conflict_paths': [], 'doc_conflict_paths': [], 'hot_code_paths': []}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1280, 'height': 900})
        try:
            page.route('**/api/update/status**', lambda route: route.fulfill(json=status))
            page.route('**/api/update/preflight', lambda route: route.fulfill(
                status=409 if failed_stage == 'preflight' else 200,
                json={'error': 'network check failed'} if failed_stage == 'preflight' else {'merge_plan': plan}))
            page.route('**/api/update/apply', lambda route: route.fulfill(
                status=409, json={'error': 'the update changed after preflight; check again', 'reason': 'release_moved'}))
            page.goto(direct_server, wait_until='domcontentloaded')
            page.wait_for_selector('#page-chat')
            page.click('[data-nav-page="dashboard"]')
            page.click('[data-dashboard-tab="updates"]')
            expect(page.locator('#btn-update-primary')).to_be_enabled()
            if failed_stage == 'preflight':
                with page.expect_response('**/api/update/status'):
                    page.click('#btn-update-primary')
            else:
                page.click('#btn-update-primary')
                with page.expect_response('**/api/update/status'):
                    page.click('[data-confirm-ok]')
            page.evaluate('() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))')
            expect(page.locator('#btn-update-primary')).to_be_enabled(timeout=2000)
        finally:
            browser.close()
