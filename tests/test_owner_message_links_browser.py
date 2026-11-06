"""The owner's own messages link their explicit http(s) addresses in the real SPA, in Chromium and WebKit.

The served page is the real `web/` tree; the test answers only the HTTP API and the socket. One owner message comes
from Main's saved history, one from the composer (the local echo, then the host's own echo), one live from another
session and one from a Project room's history. Each keeps every character the owner wrote (Markdown, HTML and code
marks stay text), links only its explicit http(s) addresses, and Copy gives back the words exactly. Opening a link is
observed two ways, kept apart: an ordinary browser opens a new page, which the test answers itself (nothing leaves the
machine); a simulated desktop bridge (`window.pywebview.api.open_external_url`, a recorder rather than the real
launcher) receives the address through the product's own link interceptor. Screenshots go to
`OUROBOROS_UI_EVIDENCE_DIR` when it is set.
"""
from __future__ import annotations

import json
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import parse_qs, urldefrag, urlparse

import pytest

from tests.ui_clipboard_sink import INSTALL, recorded, set_sink
from tests.ui_console_errors import PageErrors

pytestmark = [pytest.mark.ui_browser]
WEB = Path(__file__).resolve().parents[1] / "web"
TS = "2026-11-06T09:00:0{}Z"
PROJECT = {"id": "links-proj", "name": "Source review", "chat_id": 42, "lifecycle": "active", "visible_revision": 0}

LONG_URL = "https://example.com/" + "/".join(f"segment-{n:02d}-with-a-long-name" for n in range(8)) + "?q=1&r=2"
CYRILLIC = "https://пример.рф/путь?q=1&lang=ru#раздел"
CYRILLIC_HREF = ("https://xn--e1afmkfd.xn--p1ai/%D0%BF%D1%83%D1%82%D1%8C?q=1&lang=ru"
                 "#%D1%80%D0%B0%D0%B7%D0%B4%D0%B5%D0%BB")
SAVED = "\n".join([
    "Sources for the brief, please check each one:",
    f"«{CYRILLIC}» — the summary.",
    'See https://en.wikipedia.org/wiki/Foo_(bar)). Also "https://example.com/report?id=7&view=full".',
    "Literal: `https://example.com/code` **bold** [x](https://example.com/md) <b>not bold</b>",
    '<a href="javascript:window.__hacked=1">https://example.com/visible</a><img src=x onerror="window.__hacked=2">',
    "Not links: javascript:alert(1) www.example.com mailto:owner@example.com",
    f"Long: {LONG_URL}",
])
SAVED_LINKS = [
    [CYRILLIC, CYRILLIC_HREF],
    ["https://en.wikipedia.org/wiki/Foo_(bar)", "https://en.wikipedia.org/wiki/Foo_(bar)"],
    ["https://example.com/report?id=7&view=full", "https://example.com/report?id=7&view=full"],
    ["https://example.com/code", "https://example.com/code"],
    ["https://example.com/md", "https://example.com/md"],
    ["https://example.com/visible", "https://example.com/visible"],
    [LONG_URL, LONG_URL],
]
SENT = "New one: https://example.org/new?a=1&b=2_, thanks!"
REMOTE = "From the phone: (https://example.net/phone)"
IN_PROJECT = "Project source: https://example.com/project/plan#step-2."

# What a message shows: its links, its words, any other element inside it, and whether it scrolls sideways.
MESSAGE = """(bubble) => {
    const message = bubble.querySelector(':scope > .message');
    return {
        links: [...message.querySelectorAll('a')].map((a) => [a.textContent, a.href, a.className, a.target, a.rel]),
        text: message.textContent,
        others: [...message.querySelectorAll('*')].filter((el) => el.tagName !== 'A').map((el) => el.tagName),
        overflow: message.scrollWidth - message.clientWidth,
        pageOverflow: document.scrollingElement.scrollWidth - innerWidth,
    };
}"""
FOCUS_RING = """(link) => {
    const style = getComputedStyle(link);
    const probe = link.appendChild(document.createElement('i'));
    probe.style.color = 'var(--focus-accent-border)';
    const ring = getComputedStyle(probe).color;
    probe.remove();
    return {focused: document.activeElement === link, visible: link.matches(':focus-visible'),
            outline: [style.outlineStyle, style.outlineWidth, style.outlineColor], ring};
}"""


def history_page(rows):
    coverage = {"v": 1, "view": "fixture", "upper": {"chat": len(rows), "progress": 0}, "spans": {
        "chat": {"from": 0, "to": len(rows), "chain": "fixture", "gaps": []},
        "progress": {"from": 0, "to": 0, "chain": "empty", "gaps": []}}}
    return {"messages": rows, "progress": [], "has_more": False, "page_cursor": "fixture-page", "coverage": coverage}


def owner_row(text, index, chat_id=1):
    return {"role": "user", "text": text, "ts": TS.format(index), "source": "web", "chat_id": chat_id,
            "client_message_id": f"owner-{chat_id}-{index}", "history_id": f"chat:{chat_id}:{index}"}


HISTORY = {1: [owner_row(SAVED, 1)], PROJECT["chat_id"]: [owner_row(IN_PROJECT, 2, PROJECT["chat_id"])]}


def capture(page, name):
    root = os.environ.get("OUROBOROS_UI_EVIDENCE_DIR")
    if root:
        Path(root).mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(Path(root) / f"{name}.png"), animations="disabled")


@pytest.fixture(params=["chromium", "webkit"])
def links_ui(request):
    if os.environ.get("OUROBOROS_RUN_UI_SMOKE") != "1":
        pytest.skip("Set OUROBOROS_RUN_UI_SMOKE=1 to run the browser flow")
    playwright = pytest.importorskip("playwright.sync_api")
    engine = request.param
    opened: list[str] = []   # external pages a new window asked for, answered here
    foreign: list[str] = []  # anything else that tried to leave the page
    frames: list[dict] = []  # what the page sent over its socket
    sockets: list = []

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            self.path = self.path.removeprefix("/static")
            super().do_GET()

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(WEB)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"127.0.0.1:{server.server_port}"

    def outside(route):
        request = route.request
        if urlparse(request.url).netloc == origin:
            route.fallback()
        elif request.is_navigation_request():  # the app page itself staying put is asserted where links open
            opened.append(request.url)
            route.fulfill(content_type="text/html", body="<title>Opened</title>")
        else:
            foreign.append(request.url)
            route.abort()

    def api(route):
        url = urlparse(route.request.url)
        if url.path == "/api/onboarding":
            route.fulfill(status=204)
            return
        if url.path == "/api/chat/history":
            chat_id = int(parse_qs(url.query).get("chat_id", ["1"])[0])
            route.fulfill(json=history_page(HISTORY.get(chat_id, [])))
            return
        body = {
            "/api/state": {"supervisor_ready": True, "active_chat_activities": [], "projects": [PROJECT],
                           "project_chat_ids": [PROJECT["chat_id"]]},
            "/api/projects": {"projects": [PROJECT]},
            "/api/health": {"ok": True, "version": "fixture"},
        }.get(url.path, {})
        route.fulfill(json=body)

    def socket(ws):
        sockets.append(ws)

        def received(message):
            frame = json.loads(message)
            frames.append(frame)
            if frame.get("type") == "chat":  # the host's own echo of a saved owner message
                ws.send(json.dumps({"type": "chat", "role": "user", "content": frame["content"], "source": "web",
                                    "client_message_id": frame.get("client_message_id", ""),
                                    "sender_session_id": frame.get("sender_session_id", ""), "ts": TS.format(5)}))

        ws.on_message(received)
        ws.send(json.dumps({"type": "heartbeat"}))

    try:
        with playwright.sync_playwright() as pw:
            try:
                browser = getattr(pw, engine).launch(headless=True)
            except playwright.Error as exc:
                if "Executable doesn't exist" in str(exc):
                    pytest.skip(f"{engine} is not installed: {exc}")
                raise
            try:
                context = browser.new_context(viewport={"width": 1280, "height": 900})
                page = context.new_page()
                page.add_init_script(INSTALL)
                errors = PageErrors(page, engine)
                context.route("**/*", outside)
                context.route("**/api/**", api)
                context.route_web_socket("**/ws", socket)
                yield {"page": page, "url": f"http://{origin}", "engine": engine, "opened": opened,
                       "foreign": foreign, "frames": frames, "sockets": sockets}
                assert not errors.unexpected(), errors
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def open_main(ui, theme, size):
    page = ui["page"]
    page.set_viewport_size(size)
    page.add_init_script(f"localStorage.setItem('ouroboros.theme', {json.dumps(theme)})")
    page.goto(ui["url"] + "/")
    page.wait_for_selector("#chat-input", state="attached")
    page.locator("#chat-messages .chat-bubble.user", has_text="Sources for the brief").wait_for()
    page.wait_for_function("() => document.documentElement.dataset.theme")
    return page


def owner_bubble(page, words, root="#chat-messages"):
    bubble = page.locator(f"{root} .chat-bubble.user", has_text=words)
    bubble.first.wait_for()
    return bubble


def assert_message(page, bubble, words, links, where):
    shown = bubble.evaluate(MESSAGE)
    assert shown["text"] == words, (where, shown["text"])
    assert shown["links"] == [[text, href, "md-link", "_blank", "noopener noreferrer"] for text, href in links], (
        where, shown["links"])
    assert shown["others"] == [], (where, shown["others"])
    assert shown["overflow"] <= 1 and shown["pageOverflow"] <= 1, (where, shown)
    return shown


def open_in_new_page(page, ui, act, href, where):
    before = page.url
    with page.expect_popup() as popup:
        act()
    opened = popup.value
    opened.wait_for_load_state()
    # The page lands on the whole address; its request carries no fragment.
    assert opened.url == href and ui["opened"][-1] == urldefrag(href).url, (where, opened.url, ui["opened"])
    opened.close()
    assert page.url == before, (where, "the app itself never navigates away")


@pytest.mark.parametrize("theme,size", [
    ("dark", {"width": 1280, "height": 900}), ("light", {"width": 1280, "height": 900}),
    ("light", {"width": 390, "height": 844}), ("dark", {"width": 390, "height": 844}),
], ids=["desktop-dark", "desktop-light", "phone-light", "phone-dark"])
def test_owner_messages_link_addresses_keep_their_words_and_open_outside(links_ui, theme, size):
    ui, engine = links_ui, links_ui["engine"]
    tag = f"{engine}-{'phone' if size['width'] < 600 else 'desktop'}-{theme}"
    page = open_main(ui, theme, size)

    # Saved history: every character as written, only the explicit addresses linked; nothing it holds ran.
    saved = owner_bubble(page, "Sources for the brief")
    assert_message(page, saved, SAVED, SAVED_LINKS, f"{tag} history")
    assert page.evaluate("() => window.__hacked") is None

    # Copy gives back the owner's words exactly, not the links' destinations.
    set_sink(page, "ok", True, reset=True)
    saved.locator(".chat-message-copy").focus()
    page.keyboard.press("Enter")
    page.wait_for_function("() => window.__sink.writes.length === 1")
    assert recorded(page)["writes"] == [SAVED], tag

    # The keyboard reaches a link with the one focus ring and opens it; a pointer opens another.
    links = saved.locator(".message a")
    links.nth(2).focus()
    page.keyboard.press("Alt+Shift+Tab" if engine == "webkit" else "Shift+Tab")  # Safari's Tab skips links
    ring = links.nth(1).evaluate(FOCUS_RING)
    assert ring["focused"] and ring["visible"], (tag, ring)
    assert ring["outline"] == ["solid", "2px", ring["ring"]], (tag, ring)
    capture(page, f"owner-links-focus-{tag}")
    open_in_new_page(page, ui, lambda: page.keyboard.press("Enter"), SAVED_LINKS[1][1], f"{tag} keyboard")
    open_in_new_page(page, ui, lambda: links.nth(0).click(), CYRILLIC_HREF, f"{tag} pointer")

    # A new message from the composer links at once and stays one bubble after the host's own echo.
    page.locator("#chat-input").fill(SENT)
    page.locator("#chat-input").press("Enter")
    page.wait_for_function("() => document.querySelectorAll('#chat-messages .chat-bubble.user').length === 2")
    sent_frame = next(frame for frame in ui["frames"] if frame.get("type") == "chat")
    assert sent_frame["content"] == SENT, sent_frame
    sent = owner_bubble(page, "New one:")
    assert sent.count() == 1
    assert_message(page, sent, SENT, [["https://example.org/new?a=1&b=2_", "https://example.org/new?a=1&b=2_"]],
                   f"{tag} sent")

    # A message the owner sent from another session arrives live the same way.
    ui["sockets"][-1].send(json.dumps({"type": "chat", "role": "user", "content": REMOTE, "source": "telegram",
                                       "client_message_id": "phone-1", "sender_session_id": "phone",
                                       "ts": TS.format(6)}))
    remote = owner_bubble(page, "From the phone:")
    assert_message(page, remote, REMOTE, [["https://example.net/phone", "https://example.net/phone"]], f"{tag} live")
    saved.scroll_into_view_if_needed()
    capture(page, f"owner-links-main-{tag}")

    # A Project room's saved owner message links the same way and opens outside.
    page.evaluate("p => window.dispatchEvent(new CustomEvent('ouro:open-project', {detail: {project: p}}))", PROJECT)
    page.wait_for_function("() => !document.getElementById('project-panel').hidden")
    in_project = owner_bubble(page, "Project source:", "#project-panel")
    assert_message(page, in_project, IN_PROJECT,
                   [["https://example.com/project/plan#step-2", "https://example.com/project/plan#step-2"]],
                   f"{tag} project")
    page.wait_for_timeout(400)  # the panel's arrival motion settles before the shot
    capture(page, f"owner-links-project-{tag}")
    open_in_new_page(page, ui, lambda: in_project.locator(".message a").click(),
                     "https://example.com/project/plan#step-2", f"{tag} project")
    assert ui["foreign"] == [], ui["foreign"]


def test_owner_message_link_opens_through_the_simulated_desktop_bridge(links_ui):
    """The desktop app's interceptor hands the address to the host bridge; no window opens in the page."""
    ui, engine = links_ui, links_ui["engine"]
    page = ui["page"]
    page.add_init_script("""(() => {
        window.__bridged = [];
        window.pywebview = {api: {open_external_url: async (url) => { window.__bridged.push(url); return {ok: true}; }}};
    })()""")
    page = open_main(ui, "light", {"width": 1280, "height": 900})
    links = owner_bubble(page, "Sources for the brief").locator(".message a")
    popups: list = []
    page.on("popup", lambda popup: popups.append(popup))
    links.nth(0).click()
    links.nth(2).focus()
    page.keyboard.press("Enter")
    page.wait_for_function("() => window.__bridged.length === 2")
    page.wait_for_timeout(300)
    assert page.evaluate("() => window.__bridged") == [CYRILLIC_HREF, SAVED_LINKS[2][1]], engine
    assert popups == [] and ui["opened"] == [] and ui["foreign"] == [], (engine, ui["opened"], ui["foreign"])
