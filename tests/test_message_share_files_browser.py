"""Share's File format over the static SPA in a real engine (no runtime), then the exact saved file on its own.

One saved answer becomes one page of its own rendering (a table, a formula as MathML, a diagram's SVG, a chart's
picture with its data), read as the room draws it (the same computed type, rhythm and ink), never with the owner's
question its reply quote shows in the chat, and made only once the room has drawn its diagram and chart even when File
is pressed before. A component card becomes one page of every tab as a labelled section with its picture and chart, its
controls disabled and nothing the owner typed or pressed; an app card one page whose first screen is the card's words
where scripts do not run, and the working app where they do: two guests are 70 €, three are 105 €. The app takes the
first screen only once its frame proves it runs: a viewer that refuses the frame's scripts, an app whose script the
policy refuses and an app that throws keep the words with the reason, and no frame is left. The sheet's preview is the
saved file byte for byte. Each saved file is opened from disk with scripts on and off, at 1100 px and 390 px, and
fetches nothing. A live skill widget is no file; a sheet closed while its card is read paints nothing late.
Run under OUROBOROS_UI_BROWSER_ENGINE=chromium and =webkit."""
from __future__ import annotations

import base64
import json
import os
import re
import struct
import zlib
from pathlib import Path
from urllib.parse import urlparse

import pytest

from tests import test_subscription_setup_browser as setup_browser
from tests.test_message_share_browser import FAKES, status, wait_until, ways

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser]

GUIDE = "https://example.org/river-guide"
QUESTION = "Plan a day by the river for two, with the budget."
QUESTION_ROW = "row:1@2026-11-10T09:00:00Z#aaaaaaaaaaaa"
NEIGHBOUR = "A neighbouring answer that stays in the chat."
RICH = ("## Route budget\n\n- Park first, at ten\n- Then the museum\n\n"
        "| Stop | Time | Euro |\n|---|---|---|\n| Park | 10:00 | 0 |\n| Museum | 11:00 | 24 |\n"
        "| Cafe | 13:00 | 36 |\n\nThe sum is \\(24 + 36 + 10 = 70\\) euros.\n\n"
        "```mermaid\ngraph LR\n  Park --> Museum --> Cafe\n```\n\n"
        '```chart\n{"type": "bar", "data": {"labels": ["Museum", "Cafe", "Tram"], '
        '"datasets": [{"label": "Euro", "data": [24, 36, 10]}]}}\n```\n\n'
        f"More at [the guide]({GUIDE}) and [the plan](/projects/secret-room).")
# An answer charting points (a scatter without labels): its file keeps every point in the chart's data table.
POINTS = ("Points along the river walk.\n\n"
          '```chart\n{"type": "scatter", "data": {"datasets": [{"label": "Walk", '
          '"data": [{"x": 1, "y": 3}, {"x": 2.5, "y": 7}]}]}}\n```')
TRIP_WORDS = "Trip plan: park, museum and cafe. 70 € for two."
RIVER_WORDS = "Day by the river: park at 10:00, museum at 11:00, cafe at 13:00. For two with the tram: 70 €."
SECRETS = ("card-trip", "card-river", "card-broken", "card-blocked", "task-secret-17", "req-secret-1", "row:1@", "h-c1",
           "SECRET-GUEST")


def png(width=48, height=24):
    """A small real PNG (a warm gradient), so a picture that leaves is a decoded image, not a placeholder."""
    rows = b"".join(b"\x00" + b"".join(bytes((200, 60 + 4 * x, 90 + 3 * y)) for x in range(width)) for y in range(height))
    chunk = lambda kind, data: struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


PICTURE = "data:image/png;base64," + base64.b64encode(png()).decode()
TRIP = {
    "card_id": "card-trip", "kind": "components", "title": "Trip plan", "fallback_text": TRIP_WORDS, "chat_id": 1,
    "task_id": "task-secret-17", "revision": 3, "pinned": False, "submissions_dropped": 0,
    "submissions_summary": [{"action_id": "confirm", "request_id": "req-secret-1", "delivery": "delivered",
                             "reason_code": "", "ts": "2026-11-10T10:00:00Z"}],
    "components": [{"type": "tabs", "id": "plan", "tabs": [
        {"label": "Overview", "components": [
            {"type": "metric", "label": "Total", "value": 70, "unit": "€"},
            {"type": "image", "path": "picture", "alt": "Houses by the river", "label": "The river"}]},
        {"label": "Budget", "components": [
            {"type": "chart", "id": "budget", "label": "Budget", "chart_type": "bar", "unit": "€", "path": "budget"}]},
        {"label": "Book", "components": [
            {"type": "form", "id": "book", "title": "Book a table", "submit": {"id": "book"},
             "fields": [{"name": "guests", "label": "Guests", "type": "text"}]},
            {"type": "action", "id": "confirm", "label": "Confirm", "submit": {"id": "confirm", "once": True}}]}]}],
    "data": {"picture": PICTURE,
             "budget": {"labels": ["Museum", "Cafe", "Tram"], "datasets": [{"label": "Euro", "data": [24, 36, 10]}]}},
}
RIVER_APP = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Day by the river</title>
<style>
:root{--ink:var(--ouro-text,#25252b);--tray:var(--ouro-tray,#f6f2f4)}
body{margin:0;color:var(--ink);font-family:var(--ouro-font,system-ui,sans-serif)}main{padding:16px}
.row{display:flex;justify-content:space-between;align-items:center;gap:12px;margin:0 0 12px}
button{min-width:40px;height:40px;border-radius:20px;border:0;background:var(--tray);color:var(--ink);font:inherit}
img{display:block;width:100%;height:96px;object-fit:cover;border-radius:12px}
</style></head>
<body><main>
<h1>Day by the river</h1>
<img alt="Coloured houses by a blue river" src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 96 32'%3E%3Crect width='96' height='32' fill='%23f5dce3'/%3E%3Cpath d='M0 20Q48 14 96 20V32H0Z' fill='%239bc9dd'/%3E%3C/svg%3E">
<div class="row"><span>Guests</span><span><button type="button" id="minus" aria-label="Fewer guests">−</button>
<output id="guests">2</output> <button type="button" id="plus" aria-label="More guests">+</button></span></div>
<label><input type="checkbox" id="tram" checked> Add the tram, 5 € each</label>
<p class="row">For everyone <strong><output id="total">70 €</output></strong></p>
<button type="button" id="send">Send to Ouroboros</button>
</main><script>
(() => {
  const saved = window.OuroborosCard && window.OuroborosCard.state.value;
  let count = Number(saved && saved.guests) || 2;
  const byId = (id) => document.getElementById(id);
  const render = () => {
    const total = count * 12 + count * 18 + (byId('tram').checked ? count * 5 : 0);
    byId('guests').textContent = count;
    byId('total').textContent = total + ' €';
    byId('minus').disabled = count <= 1; byId('plus').disabled = count >= 6;
  };
  const keep = () => { try { window.OuroborosCard.state.save({ guests: count }); } catch (error) {} };
  byId('minus').addEventListener('click', () => { count = Math.max(1, count - 1); render(); keep(); });
  byId('plus').addEventListener('click', () => { count = Math.min(6, count + 1); render(); keep(); });
  byId('tram').addEventListener('change', render);
  byId('send').addEventListener('click', () => window.OuroborosCard.submit({ guests: count }, 'Guests: ' + count));
  render();
})();
</script></body></html>
"""
BROKEN_APP = "<p>Half drawn</p><script>throw new Error('boom at start')</script>"
# An app whose own script the policy refuses (it would come from an address): its inline bridge still runs and measures.
BLOCKED_APP = '<p>Waiting for its script</p><script src="https://example.org/river-app.js"></script>'
RIVER = {"card_id": "card-river", "kind": "app", "title": "Day by the river", "fallback_text": RIVER_WORDS, "chat_id": 1,
         "task_id": "task-secret-17", "revision": 1, "pinned": False, "submissions_summary": [], "submissions_dropped": 0,
         "app": {"sha256": "a" * 64, "url": "/api/cards/card-river/app"}}
BROKEN = {**RIVER, "card_id": "card-broken", "title": "Broken app", "fallback_text": "The broken app's own words.",
          "app": {"sha256": "b" * 64, "url": "/api/cards/card-broken/app"}}
BLOCKED = {**RIVER, "card_id": "card-blocked", "title": "Blocked app", "fallback_text": "The blocked app's own words.",
           "app": {"sha256": "c" * 64, "url": "/api/cards/card-blocked/app"}}
SKILL = {"card_id": "card-skill", "kind": "widget", "title": "Weather", "fallback_text": "Weather: 14 °C, light rain.",
         "chat_id": 1, "revision": 1, "pinned": False, "submissions_summary": [], "submissions_dropped": 0,
         "widget": {"skill": "weather", "tab_id": "now"}}
BODIES = {body["card_id"]: body for body in (TRIP, RIVER, BROKEN, BLOCKED, SKILL)}
APPS = {"card-river": RIVER_APP, "card-broken": BROKEN_APP, "card-blocked": BLOCKED_APP}


def card_row(body, n):
    return {"role": "assistant", "msg_type": "card", "card": {key: body[key] for key in ("card_id", "kind", "title", "revision")},
            "fallback_text": body["fallback_text"], "text": body["fallback_text"], "chat_id": 1,
            "ts": f"2026-11-10T10:0{n}:00Z", "history_id": f"h-c{n}", "row_address": f"row:1@2026-11-10T10:0{n}:00Z#{n:012d}"}


def history(cards):
    """The owner's question, a neighbouring answer, the rich answer quoting the question (shown, since a row lies
    between them), an answer charting points, then the cards (the broken apps only where their failure is the point)."""
    return {"messages": [
        {"role": "user", "text": QUESTION, "chat_id": 1, "source": "web", "ts": "2026-11-10T09:00:00Z", "history_id": "h-0",
         "row_address": QUESTION_ROW},
        {"role": "assistant", "text": NEIGHBOUR, "chat_id": 1, "ts": "2026-11-10T09:00:10Z", "history_id": "h-2",
         "row_address": "row:1@2026-11-10T09:00:10Z#cccccccccccc"},
        {"role": "assistant", "text": RICH, "chat_id": 1, "ts": "2026-11-10T09:00:20Z", "history_id": "h-1",
         "row_address": "row:1@2026-11-10T09:00:20Z#bbbbbbbbbbbb",
         "reply_to": {"address": QUESTION_ROW, "chat_id": 1, "role": "user", "kind": "message", "author_label": "You",
                      "snippet": QUESTION, "ts": "2026-11-10T09:00:00Z"}},
        {"role": "assistant", "text": POINTS, "chat_id": 1, "ts": "2026-11-10T09:00:30Z", "history_id": "h-3",
         "row_address": "row:1@2026-11-10T09:00:30Z#dddddddddddd"},
        *(card_row(body, index + 1) for index, body in enumerate(cards)),
    ], "progress": []}


def evidence(name, data: bytes | str):
    root = os.environ.get("OUROBOROS_UI_EVIDENCE_DIR")
    if root:
        Path(root).mkdir(parents=True, exist_ok=True)
        target = Path(root) / name
        target.write_bytes(data if isinstance(data, bytes) else data.encode())


def open_app(ui, *, width, scheme, sockets=None, bodies=None, cards=(TRIP, RIVER, SKILL), before=None):
    page = ui["page"]
    origin = urlparse(ui["url"]).netloc
    outside: list[str] = []
    served = bodies if bodies is not None else BODIES
    rows = history(cards)

    def only_this_origin(route):
        url = urlparse(route.request.url)
        if url.scheme in {"blob", "data", "about"} or url.netloc == origin:
            route.fallback()
        else:
            outside.append(route.request.url)
            route.abort()

    def card(route):
        path = urlparse(route.request.url).path.split("/api/cards/", 1)[1]
        card_id, _, tail = path.partition("/")
        if tail == "app" and card_id in APPS:
            route.fulfill(content_type="text/plain; charset=utf-8", body=APPS[card_id])
        elif card_id in served and not tail:
            route.fulfill(content_type="application/json", body=json.dumps(served[card_id]))
        else:
            route.fulfill(status=404, content_type="application/json", body=json.dumps({"reason_code": "card_not_found"}))
    page.add_init_script(FAKES)
    page.route("**/*", only_this_origin)
    page.route("**/api/chat/history*", lambda route: route.fulfill(content_type="application/json", body=json.dumps(rows)))
    page.route("**/api/cards/**", card)
    if sockets is not None:
        page.route_web_socket("**/ws", lambda ws: (sockets.append(ws), ws.send(json.dumps({"type": "heartbeat"}))))
    if before:
        before(page)
    page.set_viewport_size({"width": width, "height": 750 if width > 620 else 844})
    page.emulate_media(color_scheme=scheme)
    page.goto(ui["url"] + "/")
    page.wait_for_selector("#chat-input", state="attached")
    return page, outside


def bubble_of(page, card_id=None, text=None):
    if card_id:
        return page.locator(f".chat-bubble[data-card-id='{card_id}']")
    return page.locator(".chat-bubble.assistant", has_text=text).first


def open_share(page, bubble):
    bubble.scroll_into_view_if_needed()
    bubble.locator("[data-message-action='share']").click()
    sheet = page.locator(".message-share[open]")
    sheet.wait_for()
    return sheet


def ready_file(page, sheet):
    """The sheet's File as it is shown, and the same file as Download saves it."""
    frame = sheet.locator(".message-share-file")
    frame.wait_for(timeout=15_000)
    shown = frame.evaluate("node => node.srcdoc")
    with page.expect_download() as download:
        sheet.locator("[data-share-action='download']").click()
    path = download.value.path()
    saved = Path(path).read_bytes()
    assert saved.decode("utf-8") == shown, "the preview is the saved file itself"
    wait_until(lambda: status(page) == "Download started.", page)
    return download.value.suggested_filename, saved.decode("utf-8")


# The host's own attributes; the bubble the file keeps around an answer says only that its Markdown is enhanced.
HOST_ATTRIBUTES = re.compile(r"\sdata-(?:widget|card|row|history|message|task|code-copy|mermaid|processed|md-|scroll|fold|i18n"
                             r"|chat(?!-markdown-enhanced=\"true\"))")


def assert_private(html, *, origin):
    for secret in SECRETS + (QUESTION, NEIGHBOUR, origin, "/api/", "/static/", "sessionStorage", "submissions"):
        assert secret not in html, f"the file carries {secret!r}"
    assert not HOST_ATTRIBUTES.search(html), HOST_ATTRIBUTES.search(html)
    assert "Content-Security-Policy" in html and "default-src 'none'" in html


def open_file(ui, html, name, *, width, scripts=True, scheme="light"):
    """The saved bytes from disk in a fresh page: what a recipient's browser opens. Every request is recorded."""
    root = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR") or ui["tmp"])
    root.mkdir(parents=True, exist_ok=True)
    target = root / name
    target.write_text(html, encoding="utf-8")
    page = ui["new_page"](viewport={"width": width, "height": 750 if width > 620 else 844}, java_script_enabled=scripts,
                          color_scheme=scheme)
    seen = {"requests": [], "responses": [], "failed": []}
    page.on("request", lambda request: seen["requests"].append(request.url))
    page.on("response", lambda response: seen["responses"].append(response.url))
    page.on("requestfailed", lambda request: seen["failed"].append((request.url, request.failure or "")))
    page.goto(target.as_uri())
    page.wait_for_load_state("load")
    return page, seen, target.as_uri()


def assert_fetched_nothing(seen, uri):
    """Nothing but the file itself was fetched: whatever else the page asked for, its policy refused before sending
    (Chromium still reports such a request, failed as blocked; nothing answers it)."""
    others = [url for url in seen["requests"] if url != uri]
    answered = [url for url in seen["responses"] if url != uri]
    refused = {url for url, failure in seen["failed"] if re.search(r"csp|blocked|content.security|refused", failure, re.I)}
    assert not answered and all(url in refused for url in others), seen


def first_screen(page):
    return page.evaluate("""() => ({
        overflow: document.documentElement.scrollWidth > window.innerWidth + 1,
        frames: document.querySelectorAll('iframe').length,
        text: document.body.innerText,
        title: document.querySelector('.ouro-share-title')?.getBoundingClientRect().top ?? -1,
        words: document.querySelector('.chat-card-fallback')?.getBoundingClientRect().bottom ?? -1,
        height: window.innerHeight,
    })""")


STYLE_PROPERTIES = ["font-family", "font-size", "font-weight", "line-height", "letter-spacing", "color", "white-space",
                    "margin-top", "margin-bottom", "padding-top", "padding-left"]
# The computed styles of `picks` (name, selector) inside the element `root` matches under `scope` (a node, else the page).
READ_STYLES = """(scope, [root, picks, properties]) => {
    const base = (scope || document).querySelector(root);
    return Object.fromEntries(picks.map(([name, selector]) => {
        const node = base && base.querySelector(selector);
        const style = node ? getComputedStyle(node) : null;
        return [name, style ? Object.fromEntries(properties.map((property) => [property, style.getPropertyValue(property)])) : null];
    }));
}"""
ANSWER_PICKS = [("heading", ":scope > h2"), ("paragraph", ":scope > p"), ("list item", ":scope li"),
                ("table cell", ":scope td"), ("link", ":scope a")]
CARD_PICKS = [("card title", ".chat-card-title"), ("metric label", ".widget-metric > span"),
              ("metric value", ".widget-metric > strong"), ("made by", ".chat-card-made")]


@pytest.fixture
def ui(subscription_ui, tmp_path):
    return {**subscription_ui, "tmp": tmp_path}


@pytest.mark.parametrize("width,scheme", [(1100, "light"), (390, "dark")])
def test_an_answer_leaves_as_one_page_of_its_own_rendering(ui, width, scheme):
    page, outside = open_app(ui, width=width, scheme=scheme)
    origin = urlparse(ui["url"]).netloc
    answer = bubble_of(page, text="Route budget")
    answer.locator(".md-mermaid svg").wait_for()
    answer.locator(".md-chart canvas").wait_for()
    answer.locator(".katex math").first.wait_for(state="attached")
    # The chat shows the owner's question this answer replies to, as a quote in its bubble (one line, cut on a word),
    # outside its message.
    quote = answer.locator(":scope > .chat-quote")
    assert quote.is_visible() and QUESTION[:22] in quote.inner_text(), quote.inner_text()
    sheet = open_share(page, answer)
    assert sheet.locator("[data-share-format]").evaluate_all("nodes => nodes.map((node) => node.dataset.shareFormat)") \
        == ["text", "card", "file"]
    assert sheet.locator("[data-share-format='text']").get_attribute("aria-pressed") == "true", "an answer opens as Text"
    sheet.locator("[data-share-format='file']").click()
    name, html = ready_file(page, sheet)
    assert name == "Route budget.html", "an answer's file is named by its title"
    assert ways(page) == ["download", "share"] and sheet.locator("[data-share-action='choose']").count() == 0
    assert sheet.locator(".message-share-count").inner_text().startswith("Whole answer · ")
    assert sheet.locator(".message-share-file-note").count() == 0, "everything was drawn: nothing leaves as its source"
    setup_browser.capture(page, f"share-file-answer-sheet-{width}-{scheme}")
    sheet.locator("[data-share-action='share']").click()
    wait_until(lambda: len(page.evaluate("window.__shared")) == 1, page)
    [file] = page.evaluate("window.__shared[0].files")
    assert file == {"name": name, "type": "text/html", "size": len(html.encode())}, file
    evidence(f"answer-{width}-{scheme}.html", html)
    assert_private(html, origin=origin)  # the quoted question and its row address included
    assert QUESTION[:22] not in html and "chat-quote" not in html, "the reply quote stays in the chat"
    assert "<script" not in html, "an answer's page has no script"
    assert "<math" in html and "katex-html" not in html, "a formula leaves as MathML"
    assert "<svg" in html and "Museum" in html and "graph LR" not in html, "the diagram leaves as its SVG"
    assert 'class="ouro-share-chart"' in html and "data:image/png;base64," in html, "the chart leaves as its picture"
    assert "widget-chart-data" in html and ">36<" in html, "and its data beside it"
    assert GUIDE in html and "secret-room" not in html, "a web address stays; the host's own route does not"
    assert "data-code-copy" not in html and "<button" not in html
    page.keyboard.press("Escape")
    sheet.wait_for(state="detached")
    for scripts in (True, False):
        file_page, seen, uri = open_file(ui, html, f"answer-{width}-{scheme}-{'js' if scripts else 'nojs'}.html",
                                             width=width, scripts=scripts, scheme=scheme)
        screen = first_screen(file_page)
        assert not screen["overflow"], screen
        assert "Route budget" in screen["text"] and "Museum" in screen["text"] and "Ouroboros" in screen["text"]
        assert file_page.locator("table").first.is_visible() and file_page.locator("svg").first.is_visible()
        assert file_page.locator("math").first.is_visible()
        chart = file_page.locator("img.ouro-share-chart")
        assert chart.evaluate("node => node.complete && node.naturalWidth > 0") and chart.is_visible()
        assert_fetched_nothing(seen, uri)
        setup_browser.capture(file_page, f"share-file-answer-{'js' if scripts else 'nojs'}-{width}-{scheme}")
        file_page.close()
    assert not outside, outside


@pytest.mark.parametrize("width,scheme", [(1100, "light"), (390, "dark")])
def test_a_shared_answer_and_card_read_as_the_room_draws_them(ui, width, scheme):
    """The same elements, in the room and in the saved file at the same width and theme, compute the same type, rhythm
    and ink: the file keeps the room's ancestor contract instead of a second stylesheet."""
    page, outside = open_app(ui, width=width, scheme=scheme)
    answer = bubble_of(page, text="Route budget")
    answer.locator(".md-mermaid svg").wait_for()
    answer.locator(".md-chart canvas").wait_for()
    card = bubble_of(page, "card-trip")
    card.locator(".widget-metric").wait_for()
    found = {}
    for kind, bubble, root, picks in (("answer", answer, ":scope > .message", ANSWER_PICKS),
                                      ("card", card, ".chat-card", CARD_PICKS)):
        room = bubble.evaluate(READ_STYLES, [root, picks, STYLE_PROPERTIES])
        sheet = open_share(page, bubble)
        if kind == "answer":
            sheet.locator("[data-share-format='file']").click()
        _name, html = ready_file(page, sheet)
        page.keyboard.press("Escape")
        sheet.wait_for(state="detached")
        file_page, _seen, _uri = open_file(ui, html, f"parity-{kind}-{width}-{scheme}.html", width=width, scheme=scheme)
        shown = file_page.locator("body").evaluate(READ_STYLES, [".ouro-share-content" if kind == "answer" else ".chat-card",
                                                                 picks, STYLE_PROPERTIES])
        file_page.close()
        found[kind] = {"room": room, "file": shown}
        for name, _selector in picks:
            assert room[name] and shown[name], (kind, name, room[name], shown[name])
            differ = {prop: (room[name][prop], shown[name][prop]) for prop in STYLE_PROPERTIES
                      if room[name][prop] != shown[name][prop]}
            assert not differ, f"{kind} {name}: room vs file {differ}"
    evidence(f"parity-{width}-{scheme}.json", json.dumps(found, indent=1, ensure_ascii=False))
    assert not outside, outside


def test_file_pressed_while_the_room_still_draws_waits_for_its_drawing(ui):
    """File pressed before the room's diagram and chart are drawn: the sheet says it is making the file, and the file
    carries the SVG and the chart's picture once they are drawn, never the diagram's source."""
    held: list = []
    page, outside = open_app(ui, width=1100, scheme="light",
                             before=lambda page: page.route("**/mermaid.min.js", lambda route: held.append(route)))
    answer = bubble_of(page, text="Route budget")
    answer.locator(".md-mermaid").wait_for()
    wait_until(lambda: held, page)
    assert answer.locator(".md-mermaid svg").count() == 0, "the diagram is still being drawn"
    sheet = open_share(page, answer)
    sheet.locator("[data-share-format='file']").click()
    making = sheet.locator(".document-reader-status", has_text="Making the file…")
    making.wait_for()
    page.wait_for_timeout(700)
    assert making.is_visible() and sheet.locator(".message-share-file").count() == 0, "no file before the drawing is done"
    held.pop().fallback()
    name, html = ready_file(page, sheet)
    assert answer.locator(".md-mermaid svg").count() == 1
    assert "<svg" in html and "graph LR" not in html, "the drawn diagram leaves, not its source"
    assert 'class="ouro-share-chart"' in html and "widget-chart-data" in html
    assert sheet.locator(".message-share-file-note").count() == 0
    evidence("answer-drawn-late-1100-light.html", html)
    setup_browser.capture(page, "share-file-answer-drawn-late-1100-light")
    assert not outside, outside


def test_a_chart_of_points_leaves_with_every_point_in_its_data(ui):
    """A scatter chart without labels leaves as its picture beside one data row per point: its series, x and y."""
    page, outside = open_app(ui, width=1100, scheme="light")
    origin = urlparse(ui["url"]).netloc
    answer = bubble_of(page, text="Points along the river walk")
    answer.locator(".md-chart canvas").wait_for()
    sheet = open_share(page, answer)
    sheet.locator("[data-share-format='file']").click()
    _name, html = ready_file(page, sheet)
    evidence("answer-points-1100-light.html", html)
    assert_private(html, origin=origin)
    page.keyboard.press("Escape")
    sheet.wait_for(state="detached")
    file_page, seen, uri = open_file(ui, html, "answer-points-1100-light-js.html", width=1100)
    assert file_page.locator("img.ouro-share-chart").evaluate("node => node.complete && node.naturalWidth > 0")
    rows = file_page.locator(".widget-chart-data tbody tr").evaluate_all(
        "rows => rows.map((row) => Array.from(row.children, (cell) => cell.textContent))")
    assert rows == [["Walk", "1", "3"], ["Walk", "2.5", "7"]], rows
    assert_fetched_nothing(seen, uri)
    file_page.close()
    assert not outside, outside


@pytest.mark.parametrize("width,scheme", [(1100, "dark"), (390, "light")])
def test_a_component_card_leaves_whole_with_its_tabs_picture_and_chart_and_nothing_pressed(ui, width, scheme):
    page, outside = open_app(ui, width=width, scheme=scheme)
    origin = urlparse(ui["url"]).netloc
    card = bubble_of(page, "card-trip")
    card.locator(".widget-tabs").wait_for()
    # The owner types into the card and presses it; its once control already counted a press (the card's record locks
    # it). None of it leaves.
    card.locator("[data-widget-tab-idx='2']").click()
    assert card.locator("[data-widget-action]").is_disabled()
    card.locator("input[name='guests']").fill("SECRET-GUEST-VALUE")
    card.locator("form[data-widget-form] button[type='submit']").click()
    card.locator("form[data-widget-form] .widget-status", has_text="Sent").wait_for()
    assert card.locator("input[name='guests']").input_value() == "SECRET-GUEST-VALUE"
    sheet = open_share(page, card)
    assert sheet.locator("[data-share-format]").evaluate_all("nodes => nodes.map((node) => node.dataset.shareFormat)") \
        == ["text", "file"]
    assert sheet.locator("[data-share-format='file']").get_attribute("aria-pressed") == "true", "a card opens as its File"
    name, html = ready_file(page, sheet)
    assert name == "Trip plan.html"
    assert sheet.locator(".message-share-count").inner_text() == "The whole card, every tab and picture"
    setup_browser.capture(page, f"share-file-card-sheet-{width}-{scheme}")
    evidence(f"components-{width}-{scheme}.html", html)
    assert_private(html, origin=origin)
    assert "<script" not in html and "Sent" not in html, "no press of the owner leaves"
    assert PICTURE in html, "the card's own picture leaves inline"
    # Text is the card's own words, exactly.
    sheet.locator("[data-share-format='text']").click()
    assert sheet.locator(".message-share-text").inner_text() == TRIP_WORDS
    page.keyboard.press("Escape")
    sheet.wait_for(state="detached")
    for scripts in (True, False):
        file_page, seen, uri = open_file(ui, html, f"components-{width}-{scheme}-{'js' if scripts else 'nojs'}.html",
                                             width=width, scripts=scripts, scheme=scheme)
        screen = first_screen(file_page)
        assert not screen["overflow"], screen
        sections = file_page.locator(".widget-group > h4").all_inner_texts()
        assert sections == ["Overview", "Budget", "Book"], sections
        assert file_page.locator(".widget-tab-btn").count() == 0, "every tab is a section, none hidden behind a button"
        assert file_page.locator(".widget-media img").evaluate("node => node.complete && node.naturalWidth === 48")
        chart = file_page.locator("img.ouro-share-chart")
        assert chart.evaluate("node => node.complete && node.naturalWidth > 0") and chart.is_visible()
        file_page.locator(".widget-chart-data summary").click()
        assert file_page.locator(".widget-chart-data td").all_inner_texts() == ["24", "36", "10"]
        controls = file_page.locator("input, button, select, textarea")
        assert controls.count() >= 3 and controls.evaluate_all("nodes => nodes.every((node) => node.disabled)")
        assert file_page.locator("form").count() == 0
        assert file_page.locator("input[name='guests']").input_value() == ""
        assert "Its buttons and fields work only in Ouroboros." in screen["text"]
        assert_fetched_nothing(seen, uri)
        setup_browser.capture(file_page, f"share-file-card-{'js' if scripts else 'nojs'}-{width}-{scheme}")
        file_page.close()
    assert not outside, outside


@pytest.mark.parametrize("width,scheme", [(1100, "light"), (390, "dark")])
def test_an_app_card_leaves_working_where_scripts_run_and_as_its_words_where_they_do_not(ui, width, scheme):
    page, outside = open_app(ui, width=width, scheme=scheme)
    origin = urlparse(ui["url"]).netloc
    river = bubble_of(page, "card-river")
    river.scroll_into_view_if_needed()
    live = river.frame_locator("iframe")
    live.locator("#plus").click()
    live.locator("#total", has_text="105 €").wait_for()  # the chat keeps this state for this card in this tab
    sheet = open_share(page, river)
    name, html = ready_file(page, sheet)
    assert name == "Day by the river.html"
    assert sheet.locator(".message-share-count").inner_text() == "The whole app, its code included"
    assert "phone’s file preview" in sheet.locator(".message-share-file-note").inner_text()
    preview = sheet.frame_locator(".message-share-file").frame_locator("iframe")
    preview.locator("#total", has_text="70 €").wait_for()  # the file starts from no saved state
    setup_browser.capture(page, f"share-file-app-sheet-{width}-{scheme}")
    evidence(f"app-{width}-{scheme}.html", html)
    assert_private(html, origin=origin)
    assert "Day by the river" in html and "byId('plus')" in html, "the whole app leaves, its code included"
    first_markup = html.split("<body", 1)[1].split("<script", 1)[0]
    assert "<iframe" not in first_markup and RIVER_WORDS in first_markup, "the static page is the words, never a frame"
    page.keyboard.press("Escape")
    sheet.wait_for(state="detached")
    # Where scripts run: the app takes the first screen once its frame proves it runs, counts, and sends nothing.
    file_page, seen, uri = open_file(ui, html, f"app-{width}-{scheme}-js.html", width=width, scheme=scheme)
    file_page.locator(".ouro-share-app:not(.is-starting) iframe").wait_for()
    assert file_page.locator("#ouro-share-first").is_hidden()
    app = file_page.frame_locator(".ouro-share-frame")
    app.locator("#total", has_text="70 €").wait_for()
    app.locator("#plus").click()
    app.locator("#total", has_text="105 €").wait_for()
    app.locator("#send").click()
    file_page.locator("#ouro-share-status", has_text="This copy sends nothing.").wait_for()
    box = file_page.locator(".ouro-share-frame").bounding_box()
    assert box["height"] >= 200 and box["y"] < file_page.viewport_size["height"], box
    assert not first_screen(file_page)["overflow"]
    assert_fetched_nothing(seen, uri)
    setup_browser.capture(file_page, f"share-file-app-js-{width}-{scheme}")
    file_page.close()
    # Where scripts do not run (a phone's file preview): the title and the card's words, at the top, no frame.
    file_page, seen, uri = open_file(ui, html, f"app-{width}-{scheme}-nojs.html", width=width, scripts=False, scheme=scheme)
    screen = first_screen(file_page)
    assert screen["frames"] == 0 and not screen["overflow"], screen
    assert RIVER_WORDS in screen["text"] and "Its app opens in a browser with JavaScript." in screen["text"]
    assert 0 <= screen["title"] < screen["words"] <= screen["height"], screen
    assert_fetched_nothing(seen, uri)
    setup_browser.capture(file_page, f"share-file-app-nojs-{width}-{scheme}")
    file_page.close()
    # A viewer that runs the file's own script but refuses scripts in the frame (the file as saved, its frame's sandbox
    # emptied, and a short time to start): the frame proves nothing, so the words stay, and no frame is left at the end.
    refused = html.replace('"sandbox":"allow-scripts"', '"sandbox":""').replace('"start":10000', '"start":800')
    assert refused.count('"sandbox":""') == 1 and refused.count('"start":800') == 1
    file_page, seen, uri = open_file(ui, refused, f"app-{width}-{scheme}-frame-refused.html", width=width, scheme=scheme)
    file_page.wait_for_timeout(350)
    assert file_page.locator("#ouro-share-first").is_visible() and RIVER_WORDS in first_screen(file_page)["text"]
    assert file_page.locator(".ouro-share-app:not(.is-starting)").count() == 0, "nothing unproven takes the first screen"
    file_page.locator("#ouro-share-note", has_text="The app did not start here in time.").wait_for()
    screen = first_screen(file_page)
    assert screen["frames"] == 0 and RIVER_WORDS in screen["text"] and not screen["overflow"], screen
    assert_fetched_nothing(seen, uri)
    setup_browser.capture(file_page, f"share-file-app-frame-refused-{width}-{scheme}")
    file_page.close()
    assert not outside, outside


def test_a_broken_app_keeps_its_words_a_skill_widget_is_no_file_and_a_closed_sheet_paints_nothing(ui):
    try:
        broken_app_skill_widget_and_closed_sheet(ui)
    finally:
        # The deliberately broken app throws inside its own frames (the chat's, the sheet's preview, the opened file):
        # that error, and only it, is expected here.
        ui["errors"][:] = [entry for entry in ui["errors"] if "boom at start" not in str(entry)]


def broken_app_skill_widget_and_closed_sheet(ui):
    sockets: list = []
    bodies = {key: dict(value) for key, value in BODIES.items()}
    page, outside = open_app(ui, width=1100, scheme="light", sockets=sockets, bodies=bodies,
                             cards=(TRIP, BROKEN, BLOCKED, SKILL))
    origin = urlparse(ui["url"]).netloc
    # A broken app and an app whose script the policy refuses: each file opens on its words with the reason, and no frame.
    for card_id, words, reason in (
            ("card-broken", "The broken app's own words.",
             # Chromium names the error; WebKit mutes an opaque frame's error to "Script error." (as on the chat's card).
             r"The app did not start here\. App error: (?:.*boom at start|Script error\.)"),
            ("card-blocked", "The blocked app's own words.",
             r"The app did not start here\. Blocked by the card’s policy: script-src(?:-elem)?")):
        sheet = open_share(page, bubble_of(page, card_id))
        _name, html = ready_file(page, sheet)
        page.keyboard.press("Escape")
        sheet.wait_for(state="detached")
        assert_private(html, origin=origin)
        evidence(f"{card_id}.html", html)
        file_page, seen, uri = open_file(ui, html, f"{card_id}-js.html", width=390)
        file_page.locator("#ouro-share-note", has_text="The app did not start here.").wait_for()
        said = file_page.locator("#ouro-share-note").inner_text()
        assert re.fullmatch(reason, said), said
        screen = first_screen(file_page)
        assert screen["frames"] == 0 and words in screen["text"], screen
        assert_fetched_nothing(seen, uri)
        setup_browser.capture(file_page, f"share-file-{card_id}-js-390")
        file_page.close()
    # A skill's live widget: its words as Text, and the File says why there is none.
    skill = bubble_of(page, "card-skill")
    sheet = open_share(page, skill)
    note = sheet.locator(".document-reader-status.is-error")
    note.wait_for()
    assert note.inner_text().startswith("This card is a skill’s live widget"), note.inner_text()
    assert ways(page) == []
    sheet.locator("[data-share-format='text']").click()
    assert sheet.locator(".message-share-text").inner_text() == SKILL["fallback_text"]
    page.keyboard.press("Escape")
    sheet.wait_for(state="detached")
    # A sheet closed while its card is read: the late body paints nothing and leaves no stage behind.
    held: list = []
    page.route("**/api/cards/card-trip", lambda route: held.append(route))
    bodies["card-trip"] = {**TRIP, "revision": 4}
    sockets[-1].send(json.dumps({"type": "card_state", "card": {"card_id": "card-trip", "kind": "components",
                                                                "title": "Trip plan", "revision": 4},
                                 "submissions_summary": [], "submissions_dropped": 0, "ts": "2026-11-10T10:05:00Z"}))
    wait_until(lambda: held, page)
    sheet = open_share(page, bubble_of(page, "card-trip"))
    sheet.locator(".document-reader-status", has_text="Making the file…").wait_for()
    sheet.locator("[data-share-action='close']").click()
    sheet.wait_for(state="detached")
    page.unroute("**/api/cards/card-trip")
    held.pop().fulfill(content_type="application/json", body=json.dumps(bodies["card-trip"]))
    page.wait_for_timeout(500)
    assert page.locator(".message-share").count() == 0 and page.locator(".message-share-stage").count() == 0
    assert not outside, outside


def test_in_telegram_a_card_opens_on_its_words_and_says_a_file_cannot_leave_there(ui):
    page, _outside = open_app(ui, width=390, scheme="dark")
    card = bubble_of(page, "card-trip")
    card.locator(".widget-tabs").wait_for()
    page.evaluate("window.__installTelegram()")
    # A Telegram client's web view whose share sheet takes no file (as most do).
    page.evaluate("Object.defineProperty(navigator, 'canShare', { configurable: true, value: (data) => !data?.files })")
    sheet = open_share(page, card)
    assert sheet.locator("[data-share-format='text']").get_attribute("aria-pressed") == "true"
    assert sheet.locator(".message-share-text").inner_text() == TRIP_WORDS
    assert sheet.locator("[data-share-action='share']").get_attribute("title") == "Share in Telegram"
    sheet.locator("[data-share-format='file']").click()
    sheet.locator(".message-share-file").wait_for()
    wait_until(lambda: status(page).startswith("Telegram cannot take a file from here."), page)
    assert ways(page) == [], "no Download inside Telegram, and no share sheet takes the file"
    setup_browser.capture(page, "share-file-telegram-390-dark")
