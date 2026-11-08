"""A reconnect adds nothing to the feed in the real SPA, in Chromium and WebKit, over
the controlled transport of ``tests/test_room_work_browser.py``.

Main and an open Project room lose their socket twice while each room's saved history moves on. Docked, Main shows
beside the Project and both reconcile on reconnect; on a phone the Project covers Main, which stays quiet and reconciles
on its return; a reconnect whose
history read fails; and the first open of pages that a reload reason brought back. Room status, the failed read and
the rehydrated rows are what the owner sees; only a reload onto changed code (``_ouro_reason=sha-change``) adds its
one ephemeral ``Restart complete`` row.
"""
from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import pytest

from tests import test_room_work_browser as rw

room_ui = rw.room_ui
subscription_ui = rw.subscription_ui
pytestmark = [pytest.mark.ui_browser]


class Transport(rw.Transport):
    """The room transport answering whole history pages, rows stamped as the server stamps them: a page without its
    boundary or coverage reads as a failed or gapped load, the notes this file must see appear and go."""

    def _history(self, route):
        chat_id = int(parse_qs(urlparse(route.request.url).query).get("chat_id", ["1"])[0])
        source = f"chat:room-{chat_id}"
        messages = [{**row, "history_id": f"{source}:{offset}", "history_position": {"source": source, "offset": offset}}
                    for offset, row in enumerate(self.history.get(chat_id, []))]
        route.fulfill(json={"messages": messages, "progress": [], "has_more": False, "next_cursor": None,
                            "page_cursor": f"page:{chat_id}", "window": {"complete": True, "truncated_by": []},
                            "coverage": {"v": 1, "view": source, "upper": {"chat": len(messages), "progress": 0},
                                         "spans": {"chat": {"from": 0, "to": len(messages), "chain": source, "gaps": []},
                                                   "progress": None}}})


NOTICE = '[data-system-type="reconnect"]'
FAILED = "Some saved history could not be loaded."
MAIN, PROJECT = "#chat-messages", f"#pchat-{rw.PROJECT['id']}-messages"
# The rows one feed shows, in order: the typing indicator and the history control are chrome.
ROWS = """(feed) => [...document.querySelector(feed).children]
    .filter((node) => node.classList.contains('chat-bubble') && !node.classList.contains('typing-bubble'))
    .map((node) => ({text: (node.querySelector('.message')?.textContent || '').trim(),
                     type: node.dataset.systemType || '', ephemeral: node.dataset.ephemeral === '1'}))"""
# Observe receipt of each history body, then allow two paint frames. This is a synchronization aid;
# the callers' DOM/state assertions, not this counter, judge the resulting rendering.
HISTORY_LANDED = """() => {
    window.historyLanded = 0;
    const nativeFetch = window.fetch.bind(window);
    window.fetch = async (...args) => {
        const response = await nativeFetch(...args);
        if (new URL(String(args[0]), location.href).pathname === '/api/chat/history') {
            const nativeJson = response.json.bind(response);
            response.json = async () => {
                const body = await nativeJson();
                requestAnimationFrame(() => requestAnimationFrame(() => { window.historyLanded += 1; }));
                return body;
            };
        }
        return response;
    };
}"""


def rows(page, feed):
    return page.evaluate(ROWS, feed)


def settle(page):
    """Two frames and a pause: a row appended after the history read had every chance to land."""
    rw.settle_frames(page)
    page.wait_for_timeout(300)


def history_reads(transport):
    return sum(1 for method, path in transport.requests if method == "GET" and path == "/api/chat/history")


@pytest.mark.parametrize("layout", ["docked", "stack"])
def test_successive_reconnects_rehydrate_main_and_project_without_a_row(room_ui, layout):
    history = {1: rw.reading(3), 42: rw.reading(3, 0, "Project")}
    transport = Transport(room_ui["page"], history=history, projects=[rw.PROJECT])
    viewport = {"width": 1100, "height": 722} if layout == "docked" else {"width": 390, "height": 780}
    page = rw.open_app(room_ui, transport, viewport=viewport)
    assert page.evaluate("document.documentElement.dataset.layout") == layout
    page.wait_for_function("() => document.querySelectorAll('#chat-messages > .chat-bubble').length >= 6")
    rw.open_project(page, rw.PROJECT)
    for feed in (MAIN, PROJECT):
        page.wait_for_function("(feed) => document.querySelectorAll(`${feed} > .chat-bubble`).length >= 6", arg=feed)
    settle(page)
    assert page.locator(NOTICE).count() == 0

    for n in (1, 2):
        count = transport.disconnect()
        # Saved while the socket is down, so only the reconnect's history read can bring them.
        main_row = rw.assistant(f"Main reply saved while the socket was down ({n}).", 40 + n)
        project_row = rw.assistant(f"Project reply saved while the socket was down ({n}).", 40 + n)
        history[1].append(main_row)
        history[42].append(project_row)
        reads = history_reads(transport)
        page.wait_for_function("() => document.querySelector('#project-panel .status-badge')?.textContent.includes('Reconnecting')")
        rw.capture(page, f"reconnect-feed-{n}-disconnected")
        transport.await_reconnect(count)
        page.locator(PROJECT).get_by_text(project_row["text"]).wait_for()
        if layout == "docked":  # Main shows beside the room: live, it reconciles on the reconnect too
            page.locator(MAIN).get_by_text(main_row["text"]).wait_for()
        else:  # covered whole, Main is away: no hidden replay; it reconciles when it shows again
            settle(page)
            assert page.locator(MAIN).get_by_text(main_row["text"]).count() == 0, 'covered Main does no hidden replay'
            page.locator('#project-panel-close').evaluate("el => el.click()")
            page.locator(MAIN).get_by_text(main_row["text"]).wait_for()
        settle(page)
        rw.capture(page, f"reconnect-feed-{n}-{layout}-reconnected")
        assert history_reads(transport) >= reads + 2, "each room read its history again: on the reconnect, or on its return"
        assert page.locator("#reconnect-overlay").count() == 0, "the shell does not cover the conversation"
        assert page.get_by_text(FAILED).count() == 0
        assert page.locator(NOTICE).count() == 0 and page.get_by_text("Reconnected").count() == 0, \
            "a reconnect is not a transcript event"
        for feed, chat_id in ((MAIN, 1), (PROJECT, 42)):
            shown = rows(page, feed)
            assert len(shown) == len(history[chat_id]), (feed, shown)
            assert not any(row["ephemeral"] or row["type"] for row in shown), (feed, shown)
            assert shown[-1]["text"] == history[chat_id][-1]["text"], (feed, shown[-1])
        if layout == "stack":
            rw.open_project(page, rw.PROJECT)


def test_a_reconnect_whose_history_read_fails_shows_the_failure_and_no_success_row(room_ui):
    history = {1: rw.reading(3)}
    transport = Transport(room_ui["page"], history=history)
    page = rw.open_app(room_ui, transport, viewport={"width": 1100, "height": 722})
    page.wait_for_function("(feed) => document.querySelectorAll(`${feed} > .chat-bubble`).length >= 6", arg=MAIN)
    settle(page)
    assert page.get_by_text(FAILED).count() == 0, "the failure below is the outage's, not the fixture's"

    def outage(route):
        route.fulfill(status=503, json={"error": "fixture history outage"})

    page.route("**/api/chat/history*", outage)
    transport.await_reconnect(transport.disconnect())
    page.get_by_text(FAILED).wait_for()
    retry = page.locator(f"{MAIN} .chat-load-older-btn", has_text="Retry loading messages")
    retry.wait_for()
    settle(page)
    rw.capture(page, "reconnect-feed-history-failed")
    assert page.locator(NOTICE).count() == 0, "a failed resync never reads as a successful reconnect"
    assert len(rows(page, MAIN)) == 6, "the reader keeps the transcript they had"

    page.unroute("**/api/chat/history*", outage)
    history[1].append(rw.assistant("Reply read once the history answers again.", 45))
    retry.click()
    page.locator(MAIN).get_by_text("Reply read once the history answers again.").wait_for()
    page.get_by_text(FAILED).wait_for(state="hidden")
    settle(page)
    assert page.locator(NOTICE).count() == 0
    assert [row["text"] for row in rows(page, MAIN)][-1] == "Reply read once the history answers again."


def test_first_open_after_a_reload_says_only_restart_complete_and_only_once(room_ui):
    newest = "The newest saved reply."
    history = {1: [*rw.reading(2), rw.assistant(newest, 50)]}
    transport = Transport(room_ui["page"], history=history)
    page = rw.open_app(room_ui, transport, viewport={"width": 1100, "height": 722})
    page.locator(MAIN).get_by_text(newest).wait_for()
    settle(page)
    assert page.locator(NOTICE).count() == 0, "a first open has nothing to announce"

    def reopen(reason):
        page.goto(f"{room_ui['url']}/?_ouro_refresh=1&_ouro_reason={reason}")
        page.locator(MAIN).get_by_text(newest).wait_for()
        # The first open clears the reload's parameters right after any notice it shows.
        page.wait_for_function("() => !/[?&]_ouro_(reason|refresh)=/.test(location.search)")
        settle(page)

    room_ui["errors"].watch(page)  # each reopen may cut off the page's own reads (tests/ui_console_errors.py)
    for reason in ("socket-disconnect", "sha-unknown", "ui-source"):
        reopen(reason)
        rw.capture(page, f"reconnect-feed-reload-{reason}")
        assert page.locator(NOTICE).count() == 0, reason

    reopen("sha-change")
    rw.capture(page, "reconnect-feed-restart-complete")
    notice = page.locator(f"{MAIN} {NOTICE}")
    assert notice.count() == 1 and notice.locator(".message").inner_text() == "Restart complete"
    shown = rows(page, MAIN)
    assert shown[-1] == {"text": "Restart complete", "type": "reconnect", "ephemeral": True}, shown[-1]
    assert [row["text"] for row in shown[:-1]][-1] == newest, "the notice follows the history it came back to"
    # Await the history response before checking that another open has not appended a notice.
    page.evaluate(HISTORY_LANDED)
    transport.await_reconnect(transport.disconnect())
    page.wait_for_function("() => window.historyLanded > 0")
    settle(page)
    assert rows(page, MAIN) == shown, "a later reconnect adds nothing and keeps what the reload said"


@pytest.mark.parametrize("away", ["settings", "hidden"])
def test_main_keeps_the_reload_notice_until_its_first_visible_return(room_ui, away):
    """The socket can open before Main hears it: on another page or in a hidden window.

    Headless browsers do not expose native tab hiding; only document visibility is substituted. The real
    app, room suspension, socket and history consumers still handle the entire return.
    """
    page = room_ui["page"]
    transport = Transport(page, history={1: rw.reading(2), 42: rw.reading(2, 0, "Project")}, projects=[rw.PROJECT])
    if away == "hidden":
        page.add_init_script("""(() => {
            let hidden = true;
            Object.defineProperty(Document.prototype, 'hidden', {configurable:true, get:() => hidden});
            Object.defineProperty(Document.prototype, 'visibilityState',
                {configurable:true, get:() => hidden ? 'hidden' : 'visible'});
            window.showTestDocument = () => {hidden=false; document.dispatchEvent(new Event('visibilitychange'));};
        })()""")
    page.goto(f"{room_ui['url']}/?_ouro_refresh=1&_ouro_reason=sha-change#{'settings' if away == 'settings' else 'chat'}")
    page.wait_for_selector("#page-chat .rw-trigger", state="attached")
    for _ in range(100):
        if transport.sockets:
            break
        page.wait_for_timeout(50)
    assert transport.sockets, "the page never opened its socket"
    settle(page)
    assert page.locator(NOTICE).count() == 0, "Main is away: no hidden rendering"
    if away == "hidden":
        page.evaluate("window.showTestDocument()")
    else:
        page.locator('[data-nav-page="chat"]').first.click()
    page.locator(f"{MAIN} {NOTICE}").wait_for()
    settle(page)
    shown = rows(page, MAIN)
    assert shown[-1] == {"text": "Restart complete", "type": "reconnect", "ephemeral": True}
    assert sum(row["type"] == "reconnect" for row in shown) == 1
    rw.capture(page, f"restart-complete-main-return-{away}")
    # A second return and a genuine reconnect do not duplicate the owed notice.
    page.locator('[data-nav-page="settings"]').first.click()
    page.locator('[data-nav-page="chat"]').first.click()
    page.evaluate(HISTORY_LANDED)
    transport.await_reconnect(transport.disconnect())
    page.wait_for_function("() => window.historyLanded > 0")
    settle(page)
    assert rows(page, MAIN) == shown
    rw.open_project(page, rw.PROJECT)
    settle(page)
    assert page.locator(f"{PROJECT} {NOTICE}").count() == 0, "late Project does not inherit Main's notice"


def test_late_project_does_not_repeat_the_reload_notice_on_reconnect(room_ui):
    history = {1: rw.reading(2), 42: rw.reading(2, 0, "Project")}
    transport = Transport(room_ui["page"], history=history, projects=[rw.PROJECT])
    page = rw.open_app(room_ui, transport, viewport={"width": 1100, "height": 722})
    page.locator(MAIN).get_by_text("Main answer 1.", exact=False).wait_for()
    # The URL can still carry a reload reason while Main's initial hydration is held.
    # Recreate that condition before the real late-mounted Project reads the URL.
    page.evaluate("history.replaceState({}, '', '?_ouro_reason=sha-change&_ouro_refresh=1')")
    rw.open_project(page, rw.PROJECT)
    page.locator(PROJECT).get_by_text("Project answer 1.", exact=False).wait_for()
    page.evaluate(HISTORY_LANDED)
    transport.await_reconnect(transport.disconnect())
    page.wait_for_function("() => window.historyLanded >= 1")
    settle(page)
    rw.capture(page, "late-project-reconnect-no-restart")
    assert page.locator(NOTICE).count() == 0, "an ordinary reconnect must not replay a captured reload reason"
