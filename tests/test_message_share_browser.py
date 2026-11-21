"""Share over the static SPA in a real engine (no runtime): a saved Ouroboros answer's Share opens the sheet on exactly
what would leave; Copy, the system share sheet and Download pass the shown Text or the Card's own PNG on; Choose
excerpt takes whole parts or the exact words selected inside the preview; a table refuses the Card with its reason;
Escape and Panic close it. In a synthetic Telegram client (Bot API 8.0 shape, not a real Telegram app) the bot is
asked for exactly the shown words, the picker's outcome is said as Telegram tells it, and a room leaving the screen
while the message is prepared closes the sheet so the late answer opens no picker. 1100 px and 390 px, both themes."""
from __future__ import annotations

import base64
import contextlib
import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import pytest

from tests import test_subscription_setup_browser as setup_browser

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser]

PREPARE = "/api/extensions/telegram/miniapp/prepare-share"
GUIDE = "https://example.org/route?day=1&language=en"
ANSWER = ("Stay by the river: the hills are behind you.\n\n## The route\n\n"
          "1. **Praça do Comércio, 9:30.** Start in the square.\n2. **Ribeira das Naus, 9:45.** Turn right.\n\n"
          f"```sh\n  npm test\n```\n\nMore at [the guide]({GUIDE}).")
TEXT = ("Stay by the river: the hills are behind you.\n\nThe route\n\n"
        "1. Praça do Comércio, 9:30. Start in the square.\n2. Ribeira das Naus, 9:45. Turn right.\n\n"
        f"```sh\n  npm test\n```\n\nMore at the guide ({GUIDE}).")
TABLE = "| Stop | Time |\n|---|---|\n| Belém | 11:00 |"
DEEP = '\n'.join('  ' * level + '- parent' for level in range(20)) + '\n\n' \
    + '\n'.join(' ' * 40 + line for line in ('```text', 'deep payload', '```'))
HISTORY = {"messages": [
    {"role": "user", "text": "Which way along the river?", "chat_id": 1, "source": "web", "ts": "2026-11-09T09:00:00Z",
     "history_id": "h-0", "row_address": "row:1@2026-11-09T09:00:00Z#aaaaaaaaaaaa"},
    {"role": "assistant", "text": ANSWER, "chat_id": 1, "ts": "2026-11-09T09:00:20Z", "history_id": "h-1",
     "row_address": "row:1@2026-11-09T09:00:20Z#bbbbbbbbbbbb"},
    {"role": "assistant", "text": TABLE, "chat_id": 1, "ts": "2026-11-09T09:01:00Z", "history_id": "h-2",
     "row_address": "row:1@2026-11-09T09:01:00Z#cccccccccccc"},
    {"role": "assistant", "text": DEEP, "chat_id": 1, "ts": "2026-11-09T09:02:00Z", "history_id": "h-3",
     "row_address": "row:1@2026-11-09T09:02:00Z#dddddddddddd"},
], "progress": []}

FAKES = """(() => {
    window.__copied = [];
    window.__images = [];
    window.__shared = [];
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: {
        writeText: async (text) => { window.__copied.push(text); },
        write: async (items) => { window.__images.push(items.length); },
    } });
    Object.defineProperty(navigator, 'canShare', { configurable: true,
        value: (data) => Boolean(data && (typeof data.text === 'string' || (data.files || []).length)) });
    Object.defineProperty(navigator, 'share', { configurable: true, value: async (data) => {
        window.__shared.push({ text: data.text ?? null,
            files: (data.files || []).map((file) => ({ name: file.name, type: file.type, size: file.size })) });
    } });
    // A Telegram client's surface as Bot API 8.0 documents it: a synthetic stand-in, not a Telegram app.
    window.__installTelegram = () => {
        const handlers = {};
        const fire = (name, arg) => (handlers[name] || []).slice().forEach((handler) => handler(arg));
        window.__tg = { shared: [], outcome: 'sent' };
        window.Telegram = { WebApp: {
            initData: 'signed-by-telegram', version: '8.0',
            isVersionAtLeast: (version) => Number(version) <= 8,
            onEvent: (name, handler) => { (handlers[name] ||= []).push(handler); },
            offEvent: (name, handler) => { handlers[name] = (handlers[name] || []).filter((item) => item !== handler); },
            shareMessage: (id, callback) => {
                window.__tg.shared.push(id);
                const outcome = window.__tg.outcome;
                setTimeout(() => {
                    if (outcome === 'sent') { callback?.(true); fire('shareMessageSent'); }
                    else { callback?.(false); fire('shareMessageFailed', { error: outcome }); }
                }, 20);
            },
        } };
    };
})();"""


def open_app(ui, *, width, scheme):
    page = ui["page"]
    origin = urlparse(ui["url"]).netloc
    outside: list[str] = []

    def only_this_origin(route):
        url = urlparse(route.request.url)
        # The page's own bytes (WebKit routes a blob: URL too) and its own origin; anything else would leave.
        if url.scheme in {"blob", "data"} or url.netloc == origin:
            route.fallback()
        else:
            outside.append(route.request.url)
            route.abort()
    page.add_init_script(FAKES)
    page.route("**/*", only_this_origin)
    page.route("**/api/chat/history*", lambda route: route.fulfill(content_type="application/json", body=json.dumps(HISTORY)))
    page.set_viewport_size({"width": width, "height": 750 if width > 620 else 844})
    page.emulate_media(color_scheme=scheme)
    page.goto(ui["url"] + "/")
    page.wait_for_selector("#chat-input", state="attached")
    return page, outside


def wait_until(predicate, page, timeout=10):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        page.wait_for_timeout(50)


def sheet_text(page):
    return page.locator(".message-share .message-share-text").evaluate("node => node.textContent")


def ways(page):
    return page.locator(".message-share .message-share-actions [data-share-action]").evaluate_all(
        "nodes => nodes.map((node) => node.dataset.shareAction)")


def status(page):
    return page.locator(".message-share .message-share-status").evaluate("node => node.hidden ? '' : node.textContent")


def save_card(page, name):
    page.wait_for_function("""() => {
        const image = document.querySelector('.message-share-card');
        return image && image.complete && image.naturalWidth > 0;
    }""")
    data = page.evaluate("""async () => {
        const image = document.querySelector('.message-share-card');
        const bytes = new Uint8Array(await (await fetch(image.src)).arrayBuffer());
        let binary = '';
        bytes.forEach((byte) => { binary += String.fromCharCode(byte); });
        return { width: image.naturalWidth, height: image.naturalHeight, shown: image.getAttribute('width'),
            box: image.getBoundingClientRect().width, png: btoa(binary) };
    }""")
    raw = base64.b64decode(data["png"])
    root = os.environ.get("OUROBOROS_UI_EVIDENCE_DIR")
    if root:
        Path(root).mkdir(parents=True, exist_ok=True)
        (Path(root) / f"{name}.png").write_bytes(raw)
    return data, raw


@pytest.mark.parametrize("width,scheme", [(1100, "light"), (390, "dark")])
def test_share_shows_exactly_what_leaves_and_passes_it_on_by_the_ways_this_device_has(subscription_ui, width, scheme):
    page, outside = open_app(subscription_ui, width=width, scheme=scheme)
    answer = page.locator(".chat-bubble.assistant").first
    share = answer.locator("[data-message-action='share']")
    share.wait_for()
    controls = answer.locator(".chat-message-actions > *").evaluate_all(
        "nodes => nodes.map(node => node.dataset.messageAction || node.className.split(' ')[0])")
    assert controls == ["chat-message-reply", "share", "chat-message-copy"], controls
    assert share.get_attribute("aria-label") == "Share"
    assert page.locator(".chat-bubble.user [data-message-action='share']").count() == 0, "the owner's own words stay"
    # Share → the whole answer as Text: its words, the link's whole address, the code's own spaces; nothing else.
    share.click()
    sheet = page.locator(".message-share[open]")
    sheet.wait_for()
    assert sheet.locator("[data-share-format='text']").get_attribute("aria-pressed") == "true"
    assert sheet_text(page) == TEXT
    assert "Which way" not in sheet.inner_text(), "the question stays in the chat"
    assert sheet.locator(".message-share-body").evaluate(
        "node => node.classList.contains('is-quiet-focus') && getComputedStyle(node).outlineStyle") == "none", \
        "opened by a pointer: no ring until a key"
    assert ways(page) == ["copy", "share"]
    box = sheet.bounding_box()
    assert box["x"] >= 0 and box["x"] + box["width"] <= width + 0.5, box
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    setup_browser.capture(page, f"share-text-{width}-{scheme}")
    sheet.locator("[data-share-action='copy']").click()
    wait_until(lambda: status(page) == "Copied.", page)
    assert page.evaluate("window.__copied") == [TEXT]
    sheet.locator("[data-share-action='share']").click()
    wait_until(lambda: status(page) == "Passed to the share sheet.", page)
    assert page.evaluate("window.__shared") == [{"text": TEXT, "files": []}]
    # Choose excerpt: whole parts by a tap; Use excerpt passes just those on.
    sheet.locator("[data-share-action='choose']").click()
    parts = sheet.locator(".message-share-part")
    parts.first.wait_for()
    assert parts.count() == 6 and parts.first.get_attribute("role") == "checkbox"
    parts.nth(1).click()
    parts.nth(2).click()
    assert parts.nth(2).get_attribute("aria-checked") == "true"
    count = sheet.locator(".message-share-count").inner_text()
    assert count.endswith(" words") and " of " in count, count
    setup_browser.capture(page, f"share-choose-{width}-{scheme}")
    sheet.locator("[data-share-action='use']").click()
    assert sheet_text(page) == "The route\n\n1. Praça do Comércio, 9:30. Start in the square."
    assert sheet.locator(".message-share-count").inner_text().startswith("Excerpt · ")
    # Exact words selected inside the preview survive the press on Use excerpt (the shell clears a page selection
    # under such a press).
    sheet.locator("[data-share-action='choose']").click()
    parts.first.wait_for()
    page.evaluate("""() => {
        const span = [...document.querySelectorAll('.message-share-part [data-share-run]')]
            .find((node) => node.textContent.includes('Start in the square'));
        const text = span.firstChild;
        const start = text.data.indexOf('Start');
        const range = document.createRange();
        range.setStart(text, start);
        range.setEnd(text, start + 'Start in the square'.length);
        getSelection().removeAllRanges();
        getSelection().addRange(range);
    }""")
    page.wait_for_function("document.querySelector('.message-share-count').textContent.startsWith('4 of ')")
    sheet.locator("[data-share-action='use']").click()
    assert sheet_text(page) == "1. Start in the square"
    # Card: the PNG itself, at full size where it fits; Download and the share sheet take that file.
    sheet.locator("[data-share-format='card']").click()
    sheet.locator(".message-share-card").wait_for()
    card, raw = save_card(page, f"share-card-{width}-{scheme}")
    assert raw[:8] == b"\x89PNG\r\n\x1a\n" and card["width"] == 1320 and card["shown"] == "440", card
    assert card["box"] <= width, card
    assert {"download", "share"} <= set(ways(page)), ways(page)
    setup_browser.capture(page, f"share-card-sheet-{width}-{scheme}")
    if width <= 620:
        tops = sheet.locator(".message-share-actions .btn").evaluate_all(
            "nodes => nodes.map((node) => Math.round(node.getBoundingClientRect().top))")
        assert len(set(tops)) == 1, f"the ways out keep one line together: {tops}"
    with page.expect_download() as download:
        sheet.locator("[data-share-action='download']").click()
    assert download.value.suggested_filename == "ouroboros-excerpt.png"
    wait_until(lambda: status(page) == "Download started.", page)
    sheet.locator("[data-share-action='share']").click()
    wait_until(lambda: len(page.evaluate("window.__shared")) == 2, page)
    [file] = page.evaluate("window.__shared[1].files")
    assert file["name"] == "ouroboros-excerpt.png" and file["type"] == "image/png" and file["size"] == len(raw), file
    # Escape closes and the press's own control has the focus again.
    page.keyboard.press("Escape")
    sheet.wait_for(state="detached")
    assert page.evaluate("document.activeElement?.dataset?.messageAction") == "share"
    # A table: Text carries it, the Card says why it cannot; the sheet is as tall as what it shows.
    table = page.locator(".chat-bubble.assistant").nth(1)
    table.locator("[data-message-action='share']").click()
    sheet.wait_for()
    assert sheet_text(page) == "Stop | Time\nBelém | 11:00"
    if width > 620:
        assert sheet.bounding_box()["height"] < 400, sheet.bounding_box()
    sheet.locator("[data-share-format='card']").click()
    note = sheet.locator(".document-reader-status.is-error")
    note.wait_for()
    assert note.inner_text() == "A card cannot show a table. Share it as Text, or choose an excerpt without it."
    assert ways(page) == []
    page.keyboard.press("Escape")
    sheet.wait_for(state="detached")
    # A leaf chosen from a deep Markdown list fits by height but not width. Refuse before creating a clipped PNG.
    page.locator(".chat-bubble.assistant").nth(2).locator("[data-message-action='share']").click()
    sheet.wait_for()
    sheet.locator("[data-share-action='choose']").click()
    sheet.locator('.message-share-part.is-code').click()
    sheet.locator("[data-share-action='use']").click()
    exact = sheet_text(page)
    assert 'deep payload' in exact
    sheet.locator("[data-share-format='card']").click()
    note = sheet.locator('.document-reader-status.is-error')
    note.wait_for()
    assert note.inner_text() == 'This layout is too wide for one readable card. Share it as Text, or choose a simpler excerpt.'
    assert sheet.locator('.message-share-card').count() == 0
    sheet.locator("[data-share-format='text']").click()
    assert sheet_text(page) == exact, 'the whole chosen Text remains available after refusal'
    # Panic closes it.
    page.evaluate("window.dispatchEvent(new CustomEvent('ouro:panic'))")
    sheet.wait_for(state="detached")
    assert not outside, f"nothing was fetched from outside the page: {outside}"


@pytest.mark.parametrize("width,scheme", [(390, "light"), (1100, "dark")])
def test_in_telegram_the_bot_prepares_the_shown_words_and_a_room_leaving_cancels_the_late_picker(subscription_ui, width, scheme):
    page, _outside = open_app(subscription_ui, width=width, scheme=scheme)
    answer = page.locator(".chat-bubble.assistant").first
    answer.locator("[data-message-action='share']").wait_for()
    page.evaluate("window.__installTelegram()")
    posted: list[dict] = []
    held: list = []

    def prepare(route):
        posted.append(route.request.post_data_json)
        if len(posted) == 1:
            held.append(route)  # the first preparation waits; the room leaves meanwhile
            return
        route.fulfill(content_type="application/json", body=json.dumps({"id": f"prepared-{len(posted)}",
                                                                        "expiration_date": 1_900_000_000}))
    page.route(f"**{PREPARE}", prepare)
    answer.locator("[data-message-action='share']").click()
    sheet = page.locator(".message-share[open]")
    sheet.wait_for()
    share = sheet.locator("[data-share-action='share']")
    assert share.get_attribute("title") == "Share in Telegram"
    if width < 620:
        # Telegram's bottom safe area as host_telegram.js mirrors it: everything in the sheet's foot stays above it.
        page.evaluate("document.documentElement.style.setProperty('--tg-safe-area-inset-bottom', '34px')")
        page.wait_for_timeout(50)
        edge = sheet.locator(".message-share-foot").evaluate(
            "node => node.getBoundingClientRect().bottom - parseFloat(getComputedStyle(node).paddingBottom)")
        assert edge <= page.viewport_size["height"] - 34, edge
    share.click()
    wait_until(lambda: held, page)
    assert status(page) == "Preparing the message in Telegram…"
    assert sheet.locator("[data-share-format='card']").is_disabled() and sheet.locator("[data-share-action='choose']").is_disabled(), \
        "the preview holds still until Telegram has exactly these words"
    # The room leaves the screen (another page): the sheet closes with what the room opened.
    page.evaluate("document.querySelector('[data-nav-page=\"dashboard\"]').click()")
    sheet.wait_for(state="detached")
    late = held.pop()
    with contextlib.suppress(Exception):  # closing aborted the page's request: then there is nothing to answer
        late.fulfill(content_type="application/json", body=json.dumps({"id": "prepared-late", "expiration_date": 1}))
    page.wait_for_timeout(400)
    assert page.evaluate("window.__tg.shared") == [], "a late preparation opens no picker"
    assert page.locator(".message-share").count() == 0
    page.evaluate("document.querySelector('[data-nav-page=\"chat\"]').click()")
    answer.locator("[data-message-action='share']").wait_for()
    # Each press: one prepared message, one picker; the owner closing the picker is quiet, Telegram's reasons are said.
    answer.locator("[data-message-action='share']").click()
    sheet.wait_for()
    for outcome, said in (("USER_DECLINED", ""), ("MESSAGE_EXPIRED", "That prepared message expired. Press Share… again."),
                          ("MESSAGE_SEND_FAILED", "Telegram could not send it."), ("sent", "Sent in Telegram.")):
        shared = len(page.evaluate("window.__tg.shared"))
        page.evaluate(f"window.__tg.outcome = {json.dumps(outcome)}")
        sheet.locator("[data-share-action='share']").click()
        wait_until(lambda: len(page.evaluate("window.__tg.shared")) == shared + 1, page)
        wait_until(lambda: status(page) == said, page)
        page.wait_for_timeout(100)
        assert status(page) == said, outcome
    assert posted == [{"text": TEXT}] * 5, "the bot is asked for exactly the shown words, once per press"
    assert page.evaluate("window.__tg.shared") == ["prepared-2", "prepared-3", "prepared-4", "prepared-5"]
    # A picture in Telegram: no file download there.
    sheet.locator("[data-share-format='card']").click()
    sheet.locator(".message-share-card").wait_for()
    assert "download" not in ways(page), ways(page)
    setup_browser.capture(page, f"share-telegram-card-{width}-{scheme}")
    page.keyboard.press("Escape")
    sheet.wait_for(state="detached")
