"""The actual Updates consumer stays informative through a pending apply."""

import os
from datetime import datetime, timezone
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
            page.click('[data-dashboard-tab="logs"]')
            page.click('[data-dashboard-tab="updates"]')
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

            page.click('[data-dashboard-tab="logs"]')
            page.click('[data-dashboard-tab="updates"]')
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
    """A check refreshes availability: it neither clears the retained attempt nor loses its tags."""
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
            page.goto(direct_server, wait_until="domcontentloaded")
            page.wait_for_selector("#page-chat")
            page.wait_for_function("window.__testSockets?.some(s => s.readyState === 1)")
            page.click('[data-nav-page="dashboard"]')
            page.click('[data-dashboard-tab="updates"]')
            expect(page.locator("#updates-attempt-headline")).to_have_text("No update was applied.")
            expect(page.locator("#btn-update-primary")).to_have_text("Check again")
            page.click("#btn-update-primary")
            expect(page.locator("#updates-summary")).to_have_text("Checking the official channel…")
            expect(page.locator("#btn-update-primary")).to_be_disabled()
            # A concurrent invalidation during the check; its passive read carries no tags.
            _emit_ws_frame(page, {"type": "update_progress_changed"})
            assert len(pending_check) == 1
            pending_check.pop().fulfill(json={**status, "official_tags": [{"tag": "v7.0.0-test", "sha": "a" * 40}]})
            expect(page.locator("#updates-official-tags")).to_contain_text("v7.0.0-test")
            expect(page.locator("#updates-attempt-headline")).to_have_text("No update was applied.")
            expect(page.locator("#updates-attempt-text")).to_contain_text("The update changed after it was prepared.")
            expect(page.locator("#updates-attempt-text")).not_to_contain_text("stopped")
        finally:
            for route in pending_check:
                route.abort()
            browser.close()


def _sidebar(page, width):
    """The sidebar the indicator lives in: always there on a desk, the open drawer on a phone."""
    if width < 980 and not page.evaluate("document.body.classList.contains('nav-drawer-open')"):
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
    return { pill: box(pill), close: box(close), sidebar: { x: sidebar.x, r: sidebar.right },
             clipped: pill.scrollWidth > pill.clientWidth + 1, overflow: document.documentElement.scrollWidth - innerWidth,
             active: document.activeElement?.id || String(document.activeElement?.className || ''),
             closeColor: getComputedStyle(close).color, pillColor: getComputedStyle(pill).color };
}"""

# The close's focus ring against its capsule: the ring's outer edge must stay inside the capsule and clear of its words.
_RING = """() => { const c = document.querySelector('.update-notice-close'), s = getComputedStyle(c);
    const pill = document.getElementById('update-pill'), p = pill.getBoundingClientRect();
    const r = c.getBoundingClientRect(), out = parseFloat(s.outlineOffset) + parseFloat(s.outlineWidth);
    const words = p.right - parseFloat(getComputedStyle(pill).paddingRight);
    return { offset: s.outlineOffset, style: s.outlineStyle, inCapsule: r.left - out >= words
        && r.right + out <= p.right + 0.5 && r.top - out >= p.top - 0.5 && r.bottom + out <= p.bottom + 0.5 }; }"""


@pytest.mark.ui_browser
@pytest.mark.parametrize("width", [1100, 390])
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_update_indicator_failure_closes_on_its_notice_and_success_stays_in_the_panel(direct_server, tmp_path, width, engine):
    """Owners 3A and 1B (#1818): a running stage and a failed or unconfirmed outcome stay on the existing capsule, the
    last until the close beside it hides it (this browser, this attempt); a confirmed success never wears it and never
    masks a newer version, its record being the Dashboard panel's alone; a needed restart keeps its words."""
    from playwright.sync_api import expect, sync_playwright

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

    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch(headless=True)
        try:
            page, errors = open_page(browser)
            _sidebar(page, width)
            pill, close = page.locator("#update-pill"), page.locator(".update-notice-close")
            # A stage names itself and has no close: it ends by itself, with the attempt.
            expect(pill).to_have_text("Stopping work…")
            expect(close).to_be_hidden()
            # A failure stays, with its own close inside the capsule's end, in the capsule's ink.
            conclude(page)
            expect(pill).to_have_text("Update not applied")
            expect(pill).to_have_attribute("data-tone", "error")
            expect(close).to_be_visible()
            _settled(page)
            notice = page.evaluate(_NOTICE)
            assert notice["close"]["x"] >= notice["pill"]["x"] and notice["close"]["r"] <= notice["pill"]["r"] + 0.5, notice
            assert notice["close"]["y"] >= notice["pill"]["y"] - 0.5 and notice["close"]["b"] <= notice["pill"]["b"] + 0.5, notice
            assert notice["close"]["w"] >= 24 and notice["close"]["h"] >= 24, ("a real pointer target", notice)
            assert notice["pill"]["r"] <= notice["sidebar"]["r"] and not notice["clipped"] and notice["overflow"] <= 0, notice
            assert notice["closeColor"] == notice["pillColor"], ("the close takes the capsule's ink", notice)
            # Under a coarse pointer the drawn 24 px circle gains the shared 44 px hit area (`.ui-hit-area`).
            coarse = page.evaluate("matchMedia('(pointer: coarse)').matches")
            if coarse:
                hit = page.evaluate("""() => { const s = getComputedStyle(document.querySelector('.update-notice-close'), '::after');
                    return [parseFloat(s.width), parseFloat(s.height)]; }""")
                assert hit[0] >= 44 and hit[1] >= 44, ("a touch target of at least 44 px", hit)
            page.locator("#primary-sidebar .sidebar-foot").screenshot(path=str(evidence / f"update-foot-{engine}-{width}-failure.png"))
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
            page.keyboard.press("Alt+Tab" if engine == "webkit" else "Tab")  # WebKit tabs to buttons with Option held
            assert page.evaluate(_NOTICE)["active"] == "update-notice-close ui-hit-area", "the close follows the capsule in tab order"
            # The one focus ring in its flush variant (DESIGN §3 "Focus"): it stays inside the capsule's end, clear of the
            # capsule's edge and of its words, which a ring drawn outside the close crosses (falsified right here).
            ring = page.evaluate(_RING)
            assert ring == {"offset": "-2px", "style": "solid", "inCapsule": True}, ring
            outer = page.add_style_tag(content=".update-notice-close:focus-visible { outline-offset: 2px !important; }")
            assert page.evaluate(_RING)["inCapsule"] is False, "an outer ring would cross the capsule and its words"
            outer.evaluate("el => el.remove()")
            box = page.locator(".update-notice").bounding_box()
            page.screenshot(path=str(evidence / f"update-foot-{engine}-{width}-focus.png"), clip={
                "x": max(0, box["x"] - 12), "y": max(0, box["y"] - 12), "width": box["width"] + 24, "height": box["height"] + 24})
            page.keyboard.press("Enter")
            expect(pill).to_have_text("Update available")
            expect(close).to_be_hidden()
            assert page.evaluate(_NOTICE)["active"] == "update-pill", "focus never falls to the page"
            expect(page.locator("#updates-attempt-headline")).to_have_text("No update was applied.")
            # Reload: this browser remembers the hide for that attempt.
            _reload(page, width)
            expect(pill).to_have_text("Update available")
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
            expect(pill).to_have_text("Update available")
            expect(close).to_be_hidden()
            # A confirmed success wears nothing: no capsule, no toast, no timer; a newer version still shows.
            conclude(page, operation_id="attempt-c", outcome="applied", result="applied", concluded_by="boot", blockers=[],
                     work_stop="", reason="")
            expect(pill).to_have_text("Update available")
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
            # A failed retention write remains visible in the preserved Dashboard record, including after restart is needed.
            if width < 980:
                page.keyboard.press("Escape")
                page.wait_for_function("document.getElementById('content').inert === false")
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
            assert actions == [], actions
            assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
            assert errors.unexpected() == [], errors
        finally:
            browser.close()


@pytest.mark.serial
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
