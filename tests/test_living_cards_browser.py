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
from tests.test_ui_smoke_playwright import _write_phase3_widget_smoke_extension, direct_server_with_data  # noqa: F401

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

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

COMPONENTS = [
    {"type": "markdown", "text": "Two plans fit what you described."},
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
