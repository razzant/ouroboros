"""Living cards in real engines (docs/DESIGN.md "Living cards"; CLIENT_CONTRACT §5 "Living cards").

On a real server in Chromium and WebKit, a seeded Main chat holds one card of each kind: a component card, a
reviewed skill's declarative widget shown inline, and an app. At 1280 and 360 px, light and dark, each card shows
its title, its body and "Ouroboros made this" inside the column with no horizontal page scroll. A press on the
component card reaches /api/decisions as the card's decision (answered by a stub here: the server side is covered by
the living-cards suites) and says Sent; the app runs in an opaque frame without network, and its submit lands in the
owner's draft without sending; Keep on Widgets puts the card on the Widgets page, where Remove takes it off again.
"""
from __future__ import annotations

import json

import pytest

from tests.test_effort_range_browser import _evidence_dir, _launch, _open_chat
from tests.test_ui_smoke_playwright import (
    _write_phase3_widget_smoke_extension,
    direct_server_with_data as direct_server_with_data,
)

pytestmark = [pytest.mark.ui_browser]

APP_HTML = """<!doctype html><html><head><style>
body { margin: 0; padding: 8px; color: var(--ouro-text); font-family: var(--ouro-font); }
button { min-height: 36px; }
</style></head><body>
<p id="seen">Loan calculator</p><button id="choose" type="button">Choose Pro</button>
<button id="leave" type="button" onclick="location.href = 'about:blank'">Leave</button>
<script>
document.getElementById('seen').textContent += ' · kept ' + JSON.stringify(OuroborosCard.state.value);
document.body.dataset.theme = OuroborosCard.theme;
document.getElementById('choose').addEventListener('click', () => {
    OuroborosCard.state.save({ chose: 'pro' });
    OuroborosCard.submit({ plan: 'pro' }, 'Pro plan');
});
fetch('https://example.com/').then(() => { document.body.dataset.network = 'reached'; },
    () => { document.body.dataset.network = 'blocked'; });
</script></body></html>"""

# A one-pixel PNG given the way ``show_card`` stores an image it read from a file: inline in the component's own ``src``.
PIXEL = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/"
         "iZk9HQAAAABJRU5ErkJggg==")

COMPONENTS = [
    {"type": "markdown", "text": "Two plans fit what you described."},
    {"type": "image", "src": PIXEL, "alt": "Both plans side by side"},
    {"type": "table", "path": "plans", "columns": [{"label": "Plan", "path": "name"}, {"label": "Price", "path": "price"}]},
    {"type": "action", "id": "pick", "label": "Choose Pro", "submit": {"id": "pick_pro", "once": True}},
]

TOGGLE = """async (skill) => {
    const response = await fetch(`/api/skills/${encodeURIComponent(skill)}/toggle`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ enabled: true }) });
    return { status: response.status, body: await response.json() };
}"""

FACTS = """() => {
    const cards = [...document.querySelectorAll('#page-chat .chat-card-bubble')].map((bubble) => {
        const r = bubble.getBoundingClientRect();
        return { id: bubble.dataset.cardId, left: r.left, right: r.right, width: r.width,
            title: bubble.querySelector('.chat-card-title')?.textContent || '',
            made: bubble.querySelector('.chat-card-made')?.textContent || '',
            body: bubble.querySelector('[data-card-body]')?.innerText || '',
            fallback: Boolean(bubble.querySelector('[data-card-body] > .chat-card-fallback')) };
    });
    const column = document.querySelector('#page-chat .chat-messages, #page-chat #chat-messages')?.getBoundingClientRect();
    return { cards, overflow: document.documentElement.scrollWidth - window.innerWidth, view: window.innerWidth,
        column: column ? { left: column.left, right: column.right } : null };
}"""


def _seed(data_dir) -> dict:
    from ouroboros import dialogue_store
    from ouroboros.card_store import store_app_bytes, write_card

    widget_skill = _write_phase3_widget_smoke_extension(data_dir)
    components = write_card(data_dir, chat_id=1, task_id="", title="Pick a plan", fallback_text="Basic 9 · Pro 19",
                            kind="components", body={"components": COMPONENTS,
                                                     "data": {"plans": [{"name": "Basic", "price": 9}, {"name": "Pro", "price": 19}]}})
    widget = write_card(data_dir, chat_id=1, task_id="", title="Cache health", fallback_text="Cache hit 87.5 %",
                        kind="widget", body={"widget": {"skill": widget_skill, "tab_id": "main"}})
    app = write_card(data_dir, chat_id=1, task_id="", title="Loan", fallback_text="A loan calculator",
                     kind="app", body={"app": store_app_bytes(data_dir, APP_HTML.encode("utf-8"))})
    append = lambda row: dialogue_store.append(data_dir, "chat", row)  # noqa: E731
    row = lambda minute, **fields: {"ts": f"2026-10-17T10:{minute:02d}:00+00:00", "direction": "out", "chat_id": 1,
                                    "user_id": 0, "task_id": "", **fields}
    for minute, record in ((0, components), (1, widget)):
        append(row(minute, text=record["fallback_text"], type="card", card={
            "card_id": record["card_id"], "kind": record["kind"], "title": record["title"], "revision": 1}))
    # Enough conversation between them that the app is far from view when the top of the chat is read.
    for minute in range(2, 26):
        append(row(minute, text=f"Note {minute}: a line of an ordinary answer.\n\nAnd its second paragraph."))
    append(row(26, text=app["fallback_text"], type="card", card={
        "card_id": app["card_id"], "kind": "app", "title": app["title"], "revision": 1}))
    return {"components": components["card_id"], "widget": widget["card_id"], "app": app["card_id"], "skill": widget_skill}


def _wait_cards(page, ids) -> None:
    page.wait_for_selector(f'.chat-card[data-card-id="{ids["components"]}"] [data-widget-action]', timeout=30_000)
    page.wait_for_function("id => document.querySelector(`.chat-card[data-card-id=\"${id}\"]`)?.innerText.includes('Operations overview')",
                           arg=ids["widget"], timeout=30_000)
    page.wait_for_selector(f'.chat-card[data-card-id="{ids["app"]}"] iframe.chat-card-frame', timeout=30_000)


def _assert_layout(facts: dict, where: str, ids: dict) -> None:
    assert [card["id"] for card in facts["cards"]] == [ids["components"], ids["widget"], ids["app"]], (where, facts)
    assert [card["title"] for card in facts["cards"]] == ["Pick a plan", "Cache health", "Loan"], (where, facts)
    assert all(card["made"] == "Ouroboros made this" for card in facts["cards"]), (where, facts)
    assert not any(card["fallback"] for card in facts["cards"]), (where, "every body drew", facts)
    assert "Basic" in facts["cards"][0]["body"] and "Pro" in facts["cards"][0]["body"], (where, facts)
    assert facts["overflow"] <= 0, (where, facts)
    for card in facts["cards"]:
        assert card["left"] >= -0.5 and card["right"] <= facts["view"] + 0.5, (where, card)


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_living_cards_render_press_submit_and_keep(direct_server_with_data, engine, tmp_path):  # noqa: F811
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    server = direct_server_with_data
    url = server["url"]
    ids = _seed(server["data_dir"])
    evidence = _evidence_dir(tmp_path)
    with sync_playwright() as pw:
        browser, page = _launch(pw, engine, viewport={"width": 1280, "height": 900})
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            toggled = page.evaluate(TOGGLE, ids["skill"])
            assert toggled["status"] == 200 and toggled["body"].get("enabled") is True, toggled
            page.close()
            for theme, width in (("light", 1280), ("dark", 1280), ("light", 360), ("dark", 360)):
                where = f"{engine} {theme} {width}px"
                page = browser.new_page(viewport={"width": width, "height": 900})
                try:
                    _open_chat(page, url, width=width, height=900, theme=theme)
                    _wait_cards(page, ids)
                    _assert_layout(page.evaluate(FACTS), where, ids)
                    drawn = page.locator(f'.chat-card[data-card-id="{ids["components"]}"] .widget-media img')
                    drawn.evaluate("img => img.decode()")
                    assert drawn.evaluate("img => [img.naturalWidth, img.alt]") == [1, "Both plans side by side"], \
                        (where, "the card's own inline image is drawn, not named as missing")
                    frame = page.locator(f'.chat-card[data-card-id="{ids["app"]}"] iframe.chat-card-frame')
                    assert frame.get_attribute("sandbox") == "allow-scripts", where
                    inner = page.frame_locator(f'.chat-card[data-card-id="{ids["app"]}"] iframe.chat-card-frame')
                    inner.locator('body[data-network="blocked"]').wait_for(state="attached", timeout=10_000)
                    assert inner.locator("body").get_attribute("data-theme") == theme, where
                    page.screenshot(path=str(evidence / f"living-cards-{engine}-{theme}-{width}.png"), full_page=False,
                                    animations="disabled")
                    feed = page.evaluate("""(id) => { const r = document.querySelector(`#page-chat .chat-card-bubble[data-card-id="${id}"]`)
                        ?.getBoundingClientRect(); const dock = document.querySelector('#page-chat .chat-input-wrap')?.getBoundingClientRect();
                        return { bottom: r?.bottom, dock: dock?.top }; }""", ids["app"])
                    assert feed["bottom"] <= feed["dock"] + 0.5, (where, "the newest card stays above the composer", feed)
                    # A skill's widget in a message is capped at 70 % of the window until Expand.
                    widget_card = page.locator(f'.chat-card[data-card-id="{ids["widget"]}"]')
                    capped = widget_card.locator(".chat-card-widget").bounding_box()["height"]
                    assert capped <= 900 * 0.7 + 1, (where, capped)
                    assert widget_card.locator(".chat-card-expand").is_visible(), where
                    for kind in ("components", "widget"):
                        page.locator(f'.chat-card-bubble[data-card-id="{ids[kind]}"]').screenshot(
                            path=str(evidence / f"living-cards-{engine}-{theme}-{width}-{kind}.png"), animations="disabled")
                    if (theme, width) != ("light", 1280):
                        continue
                    widget_card.locator(".chat-card-expand").click()
                    assert widget_card.locator(".chat-card-widget").bounding_box()["height"] > capped + 50, "Expand lifts the cap"
                    assert widget_card.locator(".chat-card-expand").inner_text() == "Collapse"
                    widget_card.locator(".chat-card-expand").click()

                    # A press is the owner's: it goes to the card's decision and says Sent.
                    pressed = []

                    def answer(route, request):
                        pressed.append(json.loads(request.post_data or "{}"))
                        if type(pressed[-1].get("revision")) is not int or pressed[-1]["revision"] < 1:  # as the ingress
                            route.fulfill(status=400, content_type="application/json", body=json.dumps({
                                "error": "revision required", "reason_code": "revision_required"}))
                            return
                        route.fulfill(status=200, content_type="application/json", body=json.dumps({
                            "ok": True, "decision_id": pressed[-1].get("decision_id"), "request_id": pressed[-1].get("request_id"),
                            "duplicate": False, "delivery": "delivered", "card_id": ids["components"], "revision": 1}))

                    page.route("**/api/decisions", answer)
                    card = page.locator(f'.chat-card[data-card-id="{ids["components"]}"]')
                    card.locator("[data-widget-action]").click()
                    card.locator("[data-widget-feedback]", has_text="Sent ✓").wait_for(timeout=10_000)
                    assert len(pressed) == 1, pressed
                    assert set(pressed[0]) == {"request_id", "decision_id", "revision"} and pressed[0]["revision"] == 1, pressed
                    assert pressed[0]["decision_id"] == f"card:{ids['components']}:pick_pro", pressed
                    assert card.locator("[data-widget-action]").is_disabled(), "a once control locks"

                    # Far from view the app is a face with Open; back in view it runs again.
                    app_card = page.locator(f'.chat-card[data-card-id="{ids["app"]}"]')
                    app_card.locator("[data-card-open]").wait_for(timeout=10_000)
                    assert app_card.locator("iframe").count() == 0
                    # The reader brings it back (the page's own scroll; Playwright's WebKit helper does not reach into a far
                    # row the browser skips: chat_row_sleep.js).
                    app_card.evaluate("card => card.scrollIntoView({ block: 'center' })")
                    app_card.locator("iframe.chat-card-frame").wait_for(timeout=10_000)

                    # The app's submit fills the owner's draft; nothing is sent.
                    page.fill("#chat-input", "")
                    inner.locator("#choose").click()
                    page.wait_for_function("() => document.querySelector('#chat-input').value.includes('From «Loan»')", timeout=10_000)
                    assert page.input_value("#chat-input") == 'From «Loan»: Pro plan {"plan":"pro"}', page.input_value("#chat-input")
                    assert page.locator("#page-chat .chat-bubble.user").count() == 0, "an app never sends"
                    # What the app kept is handed back when it mounts again.
                    card.scroll_into_view_if_needed()
                    app_card.locator("[data-card-open]").wait_for(timeout=10_000)
                    app_card.scroll_into_view_if_needed()
                    inner.locator("#seen", has_text='kept {"chose":"pro"}').wait_for(timeout=10_000)
                    # An app that navigates its frame away is stopped at once.
                    inner.locator("#leave").click()
                    app_card.locator(".chat-card-status", has_text="The app tried to open a page and was stopped.").wait_for(timeout=10_000)
                    assert app_card.locator("iframe").count() == 0

                    # Keep on Widgets: the card lands on the Widgets page; Remove takes it off.
                    card.locator("[data-card-pin]").click()
                    card.locator("[data-card-actions]", has_text="Kept on Widgets").wait_for(timeout=10_000)
                    page.click('[data-nav-page="widgets"]')
                    kept = page.locator(f'[data-widget-key="_cards:{ids["components"]}"]')
                    kept.wait_for(state="visible", timeout=30_000)
                    assert "Basic" in kept.inner_text() and kept.locator(".widgets-card-source").inner_text() == "kept from the chat"
                    kept.locator("[data-widget-action]").click()
                    kept.locator("[data-widget-feedback]", has_text="Sent ✓").wait_for(timeout=10_000)
                    assert pressed[-1]["decision_id"] == f"card:{ids['components']}:pick_pro" and pressed[-1]["revision"] == 1, \
                        "a kept card presses the revision it was kept at"
                    page.screenshot(path=str(evidence / f"living-cards-{engine}-widgets.png"), animations="disabled")
                    kept.locator("[data-widget-menu-trigger]").click()
                    page.locator(".skills-card-menu-dialog:not([data-leaving]) [data-widget-unpin]").click()
                    kept.wait_for(state="detached", timeout=30_000)
                finally:
                    page.close()
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_chat_repaint_keeps_place_across_face_switch_and_temp_rows(direct_server_with_data, engine):
    """DESIGN 'A repaint keeps the reading place': face switches on living cards and temporary/progress
    rows collapsing or disappearing above and below the viewport must never cancel or reverse a reader's
    scroll, must keep the reference row within 2 px when stationary, must keep bottom-follow when at the
    bottom, and must keep offloading invisible cards."""
    from playwright.sync_api import sync_playwright
    from ouroboros import dialogue_store
    from ouroboros.card_store import store_app_bytes, write_card
    from tests.ui_chat_viewport_smoke import (
        _CAPTURE_TEST_SOCKET,
        _OBSERVE_STATE_READS,
        _SETTLE_RESTORE_FRAMES,
        _SETTLE_TWO_FRAMES,
        _emit_ws_frame,
        _wait_socket_open_quiescent,
        state_reads,
    )

    data_dir = direct_server_with_data["data_dir"]
    url = direct_server_with_data["url"]

    app_small_html = """<!doctype html><html><body style="margin:0;padding:8px">
    <div style="height:80px">Interactive pilot calculator</div>
    </body></html>"""
    long_fallback = "\n\n".join(
        f"Section {i}: Editorial summary of the channel pilot draft and metrics line {i} with realistic wrapping."
        for i in range(6)
    )
    app_card = write_card(
        data_dir,
        chat_id=1,
        task_id="",
        title="Editorial Pilot App",
        fallback_text=long_fallback,
        kind="app",
        body={"app": store_app_bytes(data_dir, app_small_html.encode("utf-8"))},
    )

    for idx in range(48):
        dialogue_store.append(
            data_dir,
            "chat",
            {
                "ts": f"2026-11-18T08:{idx:02d}:00+00:00",
                "direction": "out" if idx % 2 else "in",
                "chat_id": 1,
                "text": f"History message {idx:02d}: " + ("discussion text about channel layout and cards. " * 6),
            },
        )

    dialogue_store.append(
        data_dir,
        "chat",
        {
            "ts": "2026-11-18T09:30:00+00:00",
            "direction": "out",
            "chat_id": 1,
            "type": "card",
            "text": app_card["fallback_text"],
            "card": {
                "card_id": app_card["card_id"],
                "kind": "app",
                "title": app_card["title"],
                "revision": 1,
            },
        },
    )

    for idx in range(6):
        dialogue_store.append(
            data_dir,
            "chat",
            {
                "ts": f"2026-11-18T09:{35 + idx:02d}:00+00:00",
                "direction": "out" if idx % 2 == 0 else "in",
                "chat_id": 1,
                "text": f"Recent message {idx:02d} at 09:{35 + idx:02d}: "
                + ("paragraph of recent conversation below the interactive card. " * 2),
            },
        )

    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1100, "height": 750})
            try:
                page.add_init_script("() => localStorage.setItem('ouro_theme', 'light')")
                page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
                page.add_init_script(f"({_OBSERVE_STATE_READS})()")
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                _wait_socket_open_quiescent(page)
                page.wait_for_function("() => document.querySelector('#chat-messages')?.dataset.historyHydrated === 'true'")
                page.evaluate(_SETTLE_RESTORE_FRAMES)

                def incomplete_census(route):
                    resp = route.fetch()
                    body = resp.json()
                    body["active_chat_activities_complete"] = False
                    route.fulfill(response=resp, json=body)

                page.route(state_reads, incomplete_census)

                # 1. Face switch during wheel gestures: invisible card offloads at bottom, mounts on upward wheel
                #    without cancelling or reversing the scroll (neither single ticks nor multi-step burst).
                page.mouse.move(550, 400)
                for delta, steps in ((-300, 1), (-450, 1), (-300, 5)):
                    page.evaluate("""() => {
                        const feed = document.querySelector('#chat-messages');
                        feed.scrollTop = feed.scrollHeight;
                        feed.dispatchEvent(new Event('scroll'));
                    }""")
                    page.wait_for_function(
                        """(cardId) => {
                            const card = document.querySelector(`.chat-card-bubble[data-card-id="${cardId}"]`);
                            return Boolean(card?.querySelector('.chat-card-face')) && !card.querySelector('iframe');
                        }""",
                        arg=app_card["card_id"],
                        timeout=10_000,
                    )
                    page.evaluate(_SETTLE_RESTORE_FRAMES)
                    before = page.evaluate("""(cardId) => {
                        const feed = document.querySelector('#chat-messages');
                        const card = document.querySelector(`.chat-card-bubble[data-card-id="${cardId}"]`);
                        const nextMsg = card.nextElementSibling;
                        return {
                            nextMsgTop: nextMsg.getBoundingClientRect().top - feed.getBoundingClientRect().top,
                        };
                    }""", app_card["card_id"])
                    step_delta = delta // steps
                    for _ in range(steps):
                        page.mouse.wheel(0, step_delta)
                        if steps > 1:
                            page.wait_for_timeout(16)
                    page.wait_for_function(
                        """(cardId) => Boolean(document.querySelector(`.chat-card-bubble[data-card-id="${cardId}"] iframe`))""",
                        arg=app_card["card_id"],
                        timeout=10_000,
                    )
                    page.evaluate(_SETTLE_TWO_FRAMES)
                    after = page.evaluate("""(cardId) => {
                        const feed = document.querySelector('#chat-messages');
                        const card = document.querySelector(`.chat-card-bubble[data-card-id="${cardId}"]`);
                        const nextMsg = card.nextElementSibling;
                        return {
                            hasIframe: Boolean(card.querySelector('iframe')),
                            nextMsgTop: nextMsg.getBoundingClientRect().top - feed.getBoundingClientRect().top,
                        };
                    }""", app_card["card_id"])
                    expected_next_top = before["nextMsgTop"] - delta
                    shift = abs(after["nextMsgTop"] - expected_next_top)
                    assert after["hasIframe"], f"{engine}: card should stay mounted as live frame after wheel {delta}"
                    assert shift <= 2.0, (
                        f"{engine}: wheel {delta} (steps={steps}) shifted visible message by {shift:.2f}px "
                        f"(expected {expected_next_top:.2f}, got {after['nextMsgTop']:.2f})"
                    )

                # 1b. Smooth-animated keyboard scroll (PageUp) across the offloaded card: exercises WebKit's
                #     multi-frame scroll animation on both macOS and Linux (unlike macOS wheel, which jumps in 1 frame).
                wait_at_rest = """() => new Promise(done => {
                    const feed = document.querySelector('#chat-messages');
                    let last = feed.scrollTop, still = 0;
                    const tick = () => {
                        still = feed.scrollTop === last ? still + 1 : 0;
                        last = feed.scrollTop;
                        if (still >= 8) done(last);
                        else requestAnimationFrame(tick);
                    };
                    requestAnimationFrame(tick);
                })"""
                page.evaluate("""() => {
                    const feed = document.querySelector('#chat-messages');
                    feed.scrollTop = 4000;
                    feed.dispatchEvent(new Event('scroll'));
                }""")
                page.evaluate(_SETTLE_RESTORE_FRAMES)
                page.mouse.click(550, 400)
                plain_before = page.evaluate("() => document.querySelector('#chat-messages').scrollTop")
                page.keyboard.press("PageUp")
                page.wait_for_function("b => document.querySelector('#chat-messages').scrollTop !== b", arg=plain_before, timeout=10_000)
                plain_page_up = plain_before - page.evaluate(wait_at_rest)
                page.evaluate("""() => {
                    const feed = document.querySelector('#chat-messages');
                    feed.scrollTop = feed.scrollHeight;
                    feed.dispatchEvent(new Event('scroll'));
                }""")
                page.wait_for_function(
                    """(cardId) => {
                        const card = document.querySelector(`.chat-card-bubble[data-card-id="${cardId}"]`);
                        return Boolean(card?.querySelector('.chat-card-face')) && !card.querySelector('iframe');
                    }""",
                    arg=app_card["card_id"],
                    timeout=10_000,
                )
                page.evaluate(_SETTLE_RESTORE_FRAMES)
                before_page_up = page.evaluate("""(cardId) => {
                    const feed = document.querySelector('#chat-messages');
                    const card = document.querySelector(`.chat-card-bubble[data-card-id="${cardId}"]`);
                    return card.nextElementSibling.getBoundingClientRect().top - feed.getBoundingClientRect().top;
                }""", app_card["card_id"])
                page.locator("#chat-messages .chat-bubble:not(.typing-bubble) .message").last.click()
                page.keyboard.press("PageUp")
                page.wait_for_function(
                    """(cardId) => Boolean(document.querySelector(`.chat-card-bubble[data-card-id="${cardId}"] iframe`))""",
                    arg=app_card["card_id"],
                    timeout=10_000,
                )
                page.evaluate(wait_at_rest)
                after_page_up = page.evaluate("""(cardId) => {
                    const feed = document.querySelector('#chat-messages');
                    const card = document.querySelector(`.chat-card-bubble[data-card-id="${cardId}"]`);
                    return {
                        hasIframe: Boolean(card.querySelector('iframe')),
                        nextMsgTop: card.nextElementSibling.getBoundingClientRect().top - feed.getBoundingClientRect().top,
                    };
                }""", app_card["card_id"])
                expected_page_up_top = before_page_up + plain_page_up
                page_up_shift = abs(after_page_up["nextMsgTop"] - expected_page_up_top)
                assert after_page_up["hasIframe"], f"{engine}: card should stay mounted as live frame after PageUp"
                assert page_up_shift <= 2.0, (
                    f"{engine}: PageUp shifted visible message by {page_up_shift:.2f}px "
                    f"(expected {expected_page_up_top:.2f}, got {after_page_up['nextMsgTop']:.2f})"
                )

                # 2. Temporary / progress rows collapsing or disappearing above and below a reference row in history.
                ref_info = page.evaluate("""() => {
                    const feed = document.querySelector('#chat-messages');
                    const bubbles = [...feed.querySelectorAll(':scope > .chat-bubble[data-history-id]')];
                    return {
                        refId: bubbles[25].dataset.historyId,
                        prevTs: bubbles[24].dataset.ts,
                    };
                }""")
                above_ts = "2026-11-18T08:24:30+00:00"
                below_ts = "2026-11-18T09:59:00+00:00"

                def spawn_transient(task_id: str, ts: str):
                    _emit_ws_frame(page, {
                        "type": "chat",
                        "role": "assistant",
                        "chat_id": 1,
                        "task_id": task_id,
                        "is_progress": True,
                        "cancelable": True,
                        "rounds": 1,
                        "tool_calls": 1,
                        "completion_tool_calls": 1,
                        "content": "",
                        "ts": ts,
                    })

                def conclude_transient(task_id: str, ts: str):
                    _emit_ws_frame(page, {
                        "type": "chat",
                        "role": "system",
                        "system_type": "task_summary",
                        "chat_id": 1,
                        "task_id": task_id,
                        "status": "completed",
                        "task_terminal_status": "completed",
                        "content": "Done",
                        "ts": ts,
                    })

                # 2a: Transient card at top of viewport (+10 px) + transient card below viewport vanish.
                spawn_transient("temp-top-1", above_ts)
                spawn_transient("temp-bot-1", below_ts)
                page.evaluate(_SETTLE_TWO_FRAMES)
                ref_before_2a = page.evaluate("""(refId) => {
                    const feed = document.querySelector('#chat-messages');
                    const fRect = feed.getBoundingClientRect();
                    const tempAbove = feed.querySelector('.chat-live-card[data-task-id="temp-top-1"]');
                    const refNode = feed.querySelector(`[data-history-id="${refId}"]`);
                    feed.dispatchEvent(new WheelEvent('wheel', { deltaY: -10, bubbles: true }));
                    feed.scrollTop += (tempAbove.getBoundingClientRect().top - fRect.top) - 10;
                    feed.dispatchEvent(new Event('scroll'));
                    return refNode.getBoundingClientRect().top - fRect.top;
                }""", ref_info["refId"])
                page.evaluate(_SETTLE_TWO_FRAMES)
                conclude_transient("temp-top-1", above_ts)
                conclude_transient("temp-bot-1", below_ts)
                page.evaluate(_SETTLE_TWO_FRAMES)
                after_2a = page.evaluate("""(refId) => {
                    const feed = document.querySelector('#chat-messages');
                    const refNode = feed.querySelector(`[data-history-id="${refId}"]`);
                    return {
                        goneAbove: !feed.querySelector('.chat-live-card[data-task-id="temp-top-1"]'),
                        goneBelow: !feed.querySelector('.chat-live-card[data-task-id="temp-bot-1"]'),
                        refTop: refNode.getBoundingClientRect().top - feed.getBoundingClientRect().top,
                    };
                }""", ref_info["refId"])
                assert after_2a["goneAbove"] and after_2a["goneBelow"]
                assert abs(after_2a["refTop"] - ref_before_2a) <= 2.0, (
                    f"{engine}: top transient removal drifted reference row by {after_2a['refTop'] - ref_before_2a:.2f}px"
                )

                # 2b: Transient card straddling top (-25 px, colliding dataset.ts) + transient card below vanish.
                spawn_transient("temp-straddle-1", above_ts)
                spawn_transient("temp-straddle-bot-1", below_ts)
                page.evaluate(_SETTLE_TWO_FRAMES)
                ref_before_2b = page.evaluate("""({ refId, prevTs }) => {
                    const feed = document.querySelector('#chat-messages');
                    const fRect = feed.getBoundingClientRect();
                    const tempAbove = feed.querySelector('.chat-live-card[data-task-id="temp-straddle-1"]');
                    tempAbove.dataset.ts = prevTs;
                    const refNode = feed.querySelector(`[data-history-id="${refId}"]`);
                    feed.scrollTop += (tempAbove.getBoundingClientRect().top - fRect.top) + 25;
                    feed.dispatchEvent(new Event('scroll'));
                    return refNode.getBoundingClientRect().top - fRect.top;
                }""", ref_info)
                page.evaluate(_SETTLE_TWO_FRAMES)
                conclude_transient("temp-straddle-1", above_ts)
                conclude_transient("temp-straddle-bot-1", below_ts)
                page.evaluate(_SETTLE_TWO_FRAMES)
                ref_after_2b = page.evaluate("""(refId) => {
                    const feed = document.querySelector('#chat-messages');
                    const refNode = feed.querySelector(`[data-history-id="${refId}"]`);
                    return refNode.getBoundingClientRect().top - feed.getBoundingClientRect().top;
                }""", ref_info["refId"])
                assert abs(ref_after_2b - ref_before_2b) <= 2.0, (
                    f"{engine}: straddling transient removal drifted reference row by {ref_after_2b - ref_before_2b:.2f}px"
                )

                # 2c: Expanded progress card straddling top collapses into a Project portal while transient card below vanishes.
                for step_i in range(6):
                    _emit_ws_frame(page, {
                        "type": "chat",
                        "role": "assistant",
                        "chat_id": 1,
                        "task_id": "portal-collapse-1",
                        "is_progress": True,
                        "cancelable": True,
                        "content": f"Progress step {step_i}: verifying timeline anchor invariants across engines.",
                        "ts": above_ts,
                    })
                spawn_transient("temp-portal-bot-1", below_ts)
                page.evaluate(_SETTLE_TWO_FRAMES)
                ref_before_2c = page.evaluate("""(refId) => {
                    const feed = document.querySelector('#chat-messages');
                    const card = feed.querySelector('.chat-live-card[data-task-id="portal-collapse-1"]');
                    card.querySelector('[data-live-disclosure]').click();
                    const fRect = feed.getBoundingClientRect();
                    const refNode = feed.querySelector(`[data-history-id="${refId}"]`);
                    feed.scrollTop += (refNode.getBoundingClientRect().top - fRect.top) - 180;
                    feed.dispatchEvent(new Event('scroll'));
                    return refNode.getBoundingClientRect().top - fRect.top;
                }""", ref_info["refId"])
                page.evaluate(_SETTLE_TWO_FRAMES)
                _emit_ws_frame(page, {
                    "type": "chat",
                    "role": "system",
                    "system_type": "project_handoff",
                    "chat_id": 1,
                    "task_id": "portal-collapse-1",
                    "project_id": "proj-1",
                    "project_name": "Anchor Invariants",
                    "content": "Moved work to Anchor Invariants",
                    "ts": below_ts,
                })
                conclude_transient("temp-portal-bot-1", below_ts)
                page.evaluate(_SETTLE_TWO_FRAMES)
                after_2c = page.evaluate("""(refId) => {
                    const feed = document.querySelector('#chat-messages');
                    const card = feed.querySelector('.chat-live-card[data-task-id="portal-collapse-1"]');
                    const refNode = feed.querySelector(`[data-history-id="${refId}"]`);
                    return {
                        expanded: card?.dataset.expanded,
                        refTop: refNode.getBoundingClientRect().top - feed.getBoundingClientRect().top,
                    };
                }""", ref_info["refId"])
                assert after_2c["expanded"] == "0"
                assert abs(after_2c["refTop"] - ref_before_2c) <= 2.0, (
                    f"{engine}: collapsing top card drifted reference row by {after_2c['refTop'] - ref_before_2c:.2f}px"
                )

                # 2d: Transient cards wholly above and wholly below the viewport vanish.
                spawn_transient("temp-far-above-1", "2026-11-18T08:20:30+00:00")
                spawn_transient("temp-far-below-1", below_ts)
                page.evaluate(_SETTLE_TWO_FRAMES)
                ref_before_2d = page.evaluate("""(refId) => {
                    const feed = document.querySelector('#chat-messages');
                    const fRect = feed.getBoundingClientRect();
                    const refNode = feed.querySelector(`[data-history-id="${refId}"]`);
                    feed.scrollTop += (refNode.getBoundingClientRect().top - fRect.top) - 80;
                    feed.dispatchEvent(new Event('scroll'));
                    return refNode.getBoundingClientRect().top - fRect.top;
                }""", ref_info["refId"])
                page.evaluate(_SETTLE_TWO_FRAMES)
                conclude_transient("temp-far-above-1", "2026-11-18T08:20:30+00:00")
                conclude_transient("temp-far-below-1", below_ts)
                page.evaluate(_SETTLE_TWO_FRAMES)
                ref_after_2d = page.evaluate("""(refId) => {
                    const feed = document.querySelector('#chat-messages');
                    const refNode = feed.querySelector(`[data-history-id="${refId}"]`);
                    return refNode.getBoundingClientRect().top - feed.getBoundingClientRect().top;
                }""", ref_info["refId"])
                assert abs(ref_after_2d - ref_before_2d) <= 2.0, (
                    f"{engine}: wholly above/below transient removal drifted reference row by {ref_after_2d - ref_before_2d:.2f}px"
                )

                # 3. Reader at the bottom keeps following the bottom when new rows arrive (and far cards stay offloaded).
                height_before_new = page.evaluate("""() => {
                    const feed = document.querySelector('#chat-messages');
                    feed.dispatchEvent(new WheelEvent('wheel', { deltaY: 600, bubbles: true }));
                    feed.scrollTop = feed.scrollHeight;
                    feed.dispatchEvent(new Event('scroll'));
                    return feed.scrollHeight;
                }""")
                page.evaluate(_SETTLE_RESTORE_FRAMES)
                for reply_i in range(3):
                    _emit_ws_frame(page, {
                        "type": "chat",
                        "role": "assistant",
                        "chat_id": 1,
                        "content": f"Live bottom follow check row {reply_i}: " + ("new text arriving at the live edge. " * 4),
                        "ts": f"2026-11-18T10:0{reply_i}:00+00:00",
                    })
                page.evaluate(_SETTLE_RESTORE_FRAMES)
                bottom_state = page.evaluate("""(cardId) => {
                    const feed = document.querySelector('#chat-messages');
                    const card = document.querySelector(`.chat-card-bubble[data-card-id="${cardId}"]`);
                    const newest = [...feed.querySelectorAll(':scope > .chat-bubble:not(.typing-bubble)')].at(-1);
                    return {
                        scrollHeight: feed.scrollHeight,
                        gapFromBottom: feed.scrollHeight - feed.scrollTop - feed.clientHeight,
                        cardOffloaded: Boolean(card?.querySelector('.chat-card-face')) && !card.querySelector('iframe'),
                        lastText: newest?.textContent || '',
                    };
                }""", app_card["card_id"])
                assert bottom_state["scrollHeight"] > height_before_new + 150
                assert bottom_state["gapFromBottom"] <= 2.0, f"{engine}: expected bottom follow, gap={bottom_state['gapFromBottom']}"
                assert bottom_state["cardOffloaded"], f"{engine}: far card should offload back to static face at bottom"
                assert "Live bottom follow check row 2" in bottom_state["lastText"]
            finally:
                page.close()
        finally:
            browser.close()
