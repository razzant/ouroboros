"""One state read that never answers no longer owns the page's state (#2343): the real SPA on Chromium and WebKit.

The fixture holds the page's first ``/api/state`` read and never answers it. Boot waits on that read before it opens
its socket, so the browser clock jumps past the read's lifetime instead of waiting a minute. The page then paints the
failed read as Unavailable, opens its socket and reads the server's real answers. A later read is held the same way
and a socket open retires it: the page reads past it at once and paints nothing unavailable. A Project the fixture
adds to those later answers shows that the page-wide read (not ws.js's own probe) applied while the older read still
hung. Screenshots record the held, the recovered and the read-past views. The header Restart reads the census before
it asks: with every state read held, its confirmation still opens at the read's lifetime, and its send is a recorder.
"""
import json
import re

import pytest

from tests._shared import producer_evidence_dir
from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data
from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET, _TEST_SOCKET_OPEN, _emit_ws_frame, state_reads

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

LIFETIME_MS = 60_000  # web/modules/api_client.js STATE_READ_LIFETIME_MS
_NO_FIGURE = re.compile(r"^(Loading…|Unavailable)$")
_READ_PAST = {"id": "read-past-retired", "name": "Read past a retired read", "chat_id": 991, "lifecycle": "active"}
# The Restart confirmation's send goes to a recorder; a real socket frame naming /restart is dropped and flagged.
# Nothing this file does restarts the QA server (or any installation).
_FAKE_RESTART_SEND = """() => {
    window.__restartSent = [];
    window.__restartLeak = false;
    window.__ouroWs.send = (frame, options) => {
        window.__restartSent.push({ frame, options });
        return { status: 'sent', clientMessageId: '' };
    };
    for (const socket of window.__testSockets) {
        const send = socket.send.bind(socket);
        socket.send = (data) => {
            if (String(data).includes('/restart')) { window.__restartLeak = true; return; }
            send(data);
        };
    }
}"""


def _wait(page, condition, what, tries=250):
    for _ in range(tries):
        if condition():
            return
        page.wait_for_timeout(20)
    raise AssertionError(what)


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_state_read_that_never_answers_releases_the_page(direct_server_with_data, engine):
    from playwright.sync_api import expect, sync_playwright

    server = direct_server_with_data
    evidence = producer_evidence_dir(server["data_dir"].parent, "state-read-recovery",
                                     env="OUROBOROS_BROWSER_EVIDENCE_OUT") / engine
    evidence.mkdir(parents=True, exist_ok=True)
    held, failed, finished, errors, stages = [], [], [], [], []
    mode = {"hold": True, "patch": False}

    def state_read(route):
        if mode["hold"]:
            mode["hold"] = False
            held.append(route)  # never answered by this fixture
            return
        if not mode["patch"]:
            route.continue_()
            return
        response = route.fetch()
        body = response.json()
        route.fulfill(response=response, json={
            **body, "projects": [*(body.get("projects") or []), _READ_PAST],
            "project_chat_ids": [*(body.get("project_chat_ids") or []), _READ_PAST["chat_id"]],
        })

    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        page = browser.new_page(viewport={"width": 1100, "height": 900})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("requestfailed", lambda request: failed.append(request.url) if state_reads(request.url) else None)
        page.on("requestfinished", lambda request: finished.append(request.url) if state_reads(request.url) else None)
        page.add_init_script(f"({_CAPTURE_TEST_SOCKET})();")
        page.route(state_reads, state_read)
        page.clock.install()
        try:
            page.goto(server["url"], wait_until="domcontentloaded")
            page.locator("#chat-input").wait_for(state="visible")
            budget = page.locator("#chat-budget-text")
            _wait(page, lambda: held, "the page's first state read reached the fixture")
            page.wait_for_timeout(1500)
            assert page.evaluate("window.__testSockets.length") == 0, "boot waits on its first state read"
            expect(budget).to_have_text("Loading…")
            stages.append("first read held: no socket yet, the budget still loading")
            page.screenshot(path=str(evidence / "held-first-read.png"))

            page.clock.fast_forward(LIFETIME_MS + 1000)
            page.wait_for_function(_TEST_SOCKET_OPEN, timeout=30_000)
            expect(budget).not_to_have_text(_NO_FIGURE, timeout=30_000)
            assert finished, "a newer state read was answered"
            stages.append("past the lifetime: the held read failed, the socket opened, a real answer painted the budget")
            page.screenshot(path=str(evidence / "recovered.png"))

            mode["hold"] = True
            _emit_ws_frame(page, {"type": "projects_changed"})  # app.js forces a page-wide read
            _wait(page, lambda: len(held) == 2, "a later page-wide read reached the fixture and is held")
            known = budget.inner_text()
            mode["patch"] = True
            page.evaluate("window.__testSockets.find(s => s.readyState === WebSocket.OPEN).dispatchEvent(new Event('open'))")
            expect(page.locator(f'[data-project-id="{_READ_PAST["id"]}"]')).to_have_count(1, timeout=10_000)
            expect(budget).to_have_text(known)
            stages.append("a socket open retired the held read: the page read past it and painted nothing unavailable")
            page.screenshot(path=str(evidence / "read-past-retired.png"))
            assert not errors, errors
        finally:
            page.screenshot(path=str(evidence / "last-view.png"))
            (evidence / "receipt.json").write_text(json.dumps({
                "engine": engine, "stages": stages, "page_errors": errors, "held_state_reads": len(held),
                "failed_state_reads": failed, "finished_state_reads": len(finished),
                "candidate_identity": server["candidate_identity"],
                "boundary": "Real SPA and server; this fixture holds two state reads and adds one Project to later answers",
            }, indent=2), encoding="utf-8")
            for route in held:
                try:
                    route.abort()
                except Exception:  # the page already cancelled it: nothing left to release
                    pass
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_the_restart_confirmation_opens_although_its_census_read_never_answers(direct_server_with_data, engine):
    """The header Restart reads the live census before it asks. With every state read held, the one confirmation
    opens at the read's lifetime and says the pause status is unknown; confirming reaches only a recorder."""
    from playwright.sync_api import expect, sync_playwright

    server = direct_server_with_data
    evidence = producer_evidence_dir(server["data_dir"].parent, "state-read-recovery",
                                     env="OUROBOROS_BROWSER_EVIDENCE_OUT") / f"restart-{engine}"
    evidence.mkdir(parents=True, exist_ok=True)
    held, errors, stages = [], [], []
    mode = {"hold": False}

    def state_read(route):
        if mode["hold"]:
            held.append(route)  # never answered by this fixture
            return
        route.continue_()

    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        page = browser.new_page(viewport={"width": 1100, "height": 900})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.add_init_script(f"({_CAPTURE_TEST_SOCKET})();")
        page.route(state_reads, state_read)
        page.clock.install()
        sent = None
        try:
            page.goto(server["url"], wait_until="domcontentloaded")
            page.wait_for_function(_TEST_SOCKET_OPEN, timeout=30_000)
            page.evaluate(_FAKE_RESTART_SEND)
            mode["hold"] = True
            page.locator('[data-chat-command="restart"]').click()
            _wait(page, lambda: held, "a state read reached the fixture and is held")
            dialog = page.locator(".confirm-dialog")
            page.wait_for_timeout(1500)
            expect(dialog).to_have_count(0)
            stages.append("census read held: the confirmation waits on it")
            page.screenshot(path=str(evidence / "restart-census-held.png"))

            page.clock.fast_forward(LIFETIME_MS + 1000)
            dialog.wait_for(state="visible", timeout=30_000)
            assert dialog.locator("#confirm-dialog-title").inner_text() == "Restart agent"
            expect(dialog.locator(".marketplace-modal-body")).to_contain_text("Pause status could not be read")
            stages.append("past the lifetime: the one confirmation opened, its pause status unknown")
            page.screenshot(path=str(evidence / "restart-confirmation-unknown.png"))

            dialog.locator("[data-confirm-ok]").click()
            page.wait_for_function("window.__restartSent.length === 1", timeout=10_000)
            sent = page.evaluate("window.__restartSent")
            assert sent == [{"frame": {"type": "command", "cmd": "/restart"}, "options": {"queue": False}}]
            assert page.evaluate("window.__restartLeak") is False, "a /restart frame tried to leave the page"
            stages.append("confirmed: one /restart frame reached the recorder, queue:false; none left the page")
            assert not errors, errors
        finally:
            page.screenshot(path=str(evidence / "restart-last-view.png"))
            (evidence / "receipt.json").write_text(json.dumps({
                "engine": engine, "stages": stages, "page_errors": errors, "held_state_reads": len(held),
                "recorded_sends": sent, "candidate_identity": server["candidate_identity"],
                "boundary": "Real SPA and server; every state read held after load; the send is a recorder",
            }, indent=2), encoding="utf-8")
            for route in held:
                try:
                    route.abort()
                except Exception:  # the page already cancelled it: nothing left to release
                    pass
            browser.close()
