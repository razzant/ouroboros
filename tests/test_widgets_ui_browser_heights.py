"""Widgets board heights on a real server, in Chromium and WebKit (docs/DESIGN.md "Widgets board").

The mixed board: a card's width and its content's height are two independent owner choices
(`ui_preferences.widget_size {w, h}`), natural height by default, and a card that has more than its height shows a
line with Show all and Back that never touch the saved size. One fixture skill holds every kind of card: a short and
a long self-sizing module, one capped by its author, a fixed module, a route frame with and without the bridge, a
long and a short declarative card, a card kept from the chat, and a module whose scrollbar takes layout width.

Every check reads the real consumers: the frames' own documents (overflow, scroll, the reporter's messages), the
server's preferences, and a route counter that each widget calls when it mounts and when it is disposed, so a resize,
a move, Show all and Back can be shown to restart nothing; leaving and returning, a reload and a revision change show
what does. Screenshots go to OUROBOROS_UI_EVIDENCE_DIR.
"""
from __future__ import annotations

import json
import os
import pathlib
import textwrap
import time

import pytest

from tests.test_ui_smoke_playwright import direct_server_with_data as _direct_server_with_data

direct_server_with_data = _direct_server_with_data
pytestmark = [pytest.mark.ui_browser]

SKILL = "board_heights_smoke"
GAP_PX = 14

# A widget tells the route counter when it mounts and when its dispose hook runs; the bridge carries the requests.
_TELL = """
const tell = (kind) => window.OuroborosWidget.fetch(
    '/api/extensions/{skill}/hit?tab={tab}&kind=' + kind, {{ method: 'POST' }}).catch(() => {{}});
tell('mount');
window.__ouroWidgetOnDispose?.(() => tell('dispose'));
// Test observation only: how many board viewport messages and size reports this document saw.
window.__viewportMessages = 0;
window.__ouroWidgetOnViewport?.(() => {{ window.__viewportMessages += 1; }});
window.__reports = 0;
const report = window.__ouroWidgetReport;
if (report) window.__ouroWidgetReport = (message) => {{ window.__reports += 1; report(message); }};
"""

_LONG = """(() => {{
    const root = document.getElementById('root');
    root.innerHTML = '<label>Note <input id="field"></label><ol id="rows" style="margin:8px 0"></ol>'
        + '<button id="more" type="button">More</button> <button id="less" type="button">Less</button>';
    const rows = document.getElementById('rows');
    const add = (count) => {{
        for (let i = 0; i < count; i += 1) {{
            const row = document.createElement('li');
            row.style.height = '20px';
            row.textContent = 'Row ' + (rows.children.length + 1);
            rows.appendChild(row);
        }}
    }};
    add({rows});
    document.getElementById('more').addEventListener('click', () => add(20));
    document.getElementById('less').addEventListener('click', () => {{
        for (let i = 0; i < 20 && rows.lastChild; i += 1) rows.lastChild.remove();
    }});
    {tell}
}})();
"""

_SHORT = """(() => {{
    document.getElementById('root').innerHTML = '<p id="line">One short line of status.</p>';
    {tell}
}})();
"""

_FIXED = """(() => {{
    const root = document.getElementById('root');
    root.style.height = '1000px';
    root.style.background = 'linear-gradient(180deg, #1b5e3a, #0d2a3a)';
    root.textContent = 'Fixed game area';
    {tell}
}})();
"""

# Its own scrollbar takes layout width (a styled scrollbar is classic in both engines), and its text wraps by width.
_SCROLLBAR = """(() => {{
    const style = document.createElement('style');
    style.textContent = '::-webkit-scrollbar{{width:14px}}::-webkit-scrollbar-thumb{{background:#888}}';
    document.head.appendChild(style);
    const root = document.getElementById('root');
    root.innerHTML = Array.from({{ length: 24 }}, (_, i) =>
        '<p>Paragraph ' + (i + 1) + ': words that wrap differently when a scrollbar takes part of the width.</p>').join('');
    {tell}
}})();
"""

_ROUTE_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Route page</title>
<style>body{{font:14px system-ui;margin:12px}} li{{height:24px}}</style>{bridge}</head>
<body><h3 id="title">{title}</h3><ol>{rows}</ol>{script}</body></html>"""


def _write_skill(data_dir: pathlib.Path) -> str:
    from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_review_state

    skill_dir = data_dir / "skills" / "external" / SKILL
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(textwrap.dedent(f"""\
        ---
        name: {SKILL}
        description: Isolated Widgets board heights fixture.
        version: 0.1.0
        type: extension
        entry: plugin.py
        permissions: ["route", "widget"]
        ---
        # Widgets board heights fixture
        """), encoding="utf-8")
    rows = "".join(f"<li>Route row {i + 1}</li>" for i in range(40))
    plain = _ROUTE_PAGE.format(bridge="", title="Route page", rows=rows, script="")
    bridged = _ROUTE_PAGE.format(
        bridge='<script src="/api/widget-bridge.js"></script>', title="Bridged page", rows=rows,
        script=("<script>window.OuroborosWidget.ready.then(() => window.OuroborosWidget.fetch("
                f"'/api/extensions/{SKILL}/hit?tab=bridged&kind=mount', {{method: 'POST'}}));"
                "window.__ouroWidgetOnDispose?.(() => window.OuroborosWidget.fetch("
                f"'/api/extensions/{SKILL}/hit?tab=bridged&kind=dispose', {{method: 'POST'}}));</script>"))
    (skill_dir / "plugin.py").write_text(textwrap.dedent(f"""\
        from starlette.responses import HTMLResponse

        _COUNTS = {{}}
        _ROWS = [{{"issue": f"#{{1300 + i}}", "title": f"Board fixture row {{i}}", "state": "open" if i % 3 else "closed"}}
                 for i in range(30)]
        _PLAIN = {plain!r}
        _BRIDGED = {bridged!r}


        async def hit(request):
            key = request.query_params.get("tab", "") + ":" + request.query_params.get("kind", "")
            _COUNTS[key] = _COUNTS.get(key, 0) + 1
            return {{"ok": True}}


        async def counts(_request):
            return dict(_COUNTS)


        async def rows(_request):
            return {{"rows": _ROWS}}


        async def page(_request):
            return HTMLResponse(_PLAIN)


        async def bridged(_request):
            return HTMLResponse(_BRIDGED)


        def register(api):
            api.register_route("hit", hit, methods=("POST",))
            api.register_route("counts", counts)
            api.register_route("rows", rows)
            api.register_route("page", page)
            api.register_route("bridged", bridged)
            api.register_ui_tab("short", "Short status", render={{"kind": "module", "entry": "short.js", "start": "auto"}})
            api.register_ui_tab("long", "Long list", render={{"kind": "module", "entry": "long.js", "start": "auto"}})
            api.register_ui_tab("capped", "Capped list", render={{
                "kind": "module", "entry": "capped.js", "start": "auto", "max_height": 400}})
            api.register_ui_tab("fixed", "Fixed game", render={{
                "kind": "module", "entry": "fixed.js", "height": 720, "start": "auto", "span": 2}})
            api.register_ui_tab("route", "Route page", render={{"kind": "iframe", "route": "page", "start": "auto"}})
            api.register_ui_tab("bridged", "Bridged page", render={{"kind": "iframe", "route": "bridged", "start": "auto"}})
            api.register_ui_tab("scrollbar", "Wide scrollbar", render={{"kind": "module", "entry": "scrollbar.js", "start": "auto"}})
            api.register_ui_tab("table", "Issues", render={{
                "kind": "declarative", "schema_version": 1, "components": [
                    {{"type": "poll", "label": "Load", "route": "rows", "auto_start": True, "max_ticks": 1}},
                    {{"type": "table", "path": "rows", "columns": [
                        {{"label": "Issue", "path": "issue"}}, {{"label": "Title", "path": "title"}},
                        {{"label": "State", "path": "state"}},
                    ]}},
                ]}})
            api.register_ui_tab("note", "Note", render={{
                "kind": "declarative", "schema_version": 1,
                "components": [{{"type": "callout", "text": "A short note."}}]}})
        """), encoding="utf-8")
    for tab, body in (("short", _SHORT), ("long", _LONG), ("capped", _LONG), ("fixed", _FIXED),
                      ("scrollbar", _SCROLLBAR)):
        tell = _TELL.format(skill=SKILL, tab=tab)
        (skill_dir / f"{tab}.js").write_text(body.format(tell=tell, rows=60), encoding="utf-8")
    content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
    save_review_state(data_dir, SKILL, SkillReviewState(status="pass", content_hash=content_hash))
    return SKILL


FRAMED = ("short", "long", "capped", "fixed", "route", "bridged", "scrollbar")


def card(tab: str) -> str:
    return f'[data-widget-key="{SKILL}:{tab}"]'


# --- page facts -------------------------------------------------------------------------------------------------------

FACTS = """(skill) => {
    const list = document.getElementById('widgets-list');
    const out = { layout: list.dataset.widgetLayout || '', listWidth: list.getBoundingClientRect().width, cards: {} };
    for (const node of list.querySelectorAll('[data-widget-key]:not([data-widget-removed])')) {
        const tab = node.dataset.widgetKey.slice(skill.length + 1);
        const box = node.getBoundingClientRect();
        const frame = node.querySelector('iframe');
        const body = node.querySelector('[data-widget-mount]');
        const more = node.querySelector('[data-widget-more]');
        out.cards[tab] = {
            x: box.x - list.getBoundingClientRect().x, y: box.y - list.getBoundingClientRect().y,
            width: box.width, height: box.height,
            frame: frame ? frame.getBoundingClientRect().height : null,
            bodyScroll: body ? body.scrollHeight - body.clientHeight : 0,
            bounded: body ? body.hasAttribute('data-widget-bounded') : false,
            more: more && !more.hidden ? more.textContent.trim() : '',
            status: node.querySelector('[data-widget-status]:not([hidden])')?.textContent?.trim() || '',
        };
    }
    return out;
}"""

CHILD = """() => ({
    overflow: document.documentElement.style.getPropertyValue('overflow-y'),
    scrollHeight: document.documentElement.scrollHeight, innerHeight: window.innerHeight,
    clientWidth: document.documentElement.clientWidth, innerWidth: window.innerWidth,
    scrollY: window.scrollY, viewport: window.__viewportMessages ?? null, reports: window.__reports ?? null,
    mark: window.__boardMark ?? null, field: document.getElementById('field')?.value ?? null,
})"""


def facts(page) -> dict:
    return page.evaluate(FACTS, SKILL)


def child(page, tab: str) -> dict:
    return page.locator(f"{card(tab)} iframe").element_handle().content_frame().evaluate(CHILD)


def saved(page) -> dict:
    return page.evaluate("async () => (await (await fetch('/api/ui/preferences')).json())")


def counts(page) -> dict:
    return page.evaluate(f"async () => (await (await fetch('/api/extensions/{SKILL}/counts')).json())")


def wait_until(page, predicate: str, arg=None, timeout: int = 8_000) -> None:
    """Polls `predicate` until it is truthy (`wait_for_function` takes a promise for an answer and never waits for an
    async predicate)."""
    deadline = time.monotonic() + timeout / 1000
    while not page.evaluate(predicate, arg):
        assert time.monotonic() < deadline, f"timed out waiting for: {predicate[:100]}"
        page.wait_for_timeout(50)


def wait_saved(page, tab: str, size) -> None:
    key = f"{SKILL}:{tab}"
    deadline = time.monotonic() + 8
    while True:
        stored = saved(page).get("widget_size", {}).get(key)
        if stored == size or (size is not None and stored and all(stored.get(k) == v for k, v in size.items())):
            return
        assert time.monotonic() < deadline, f"{key}: saved {stored!r}, wanted {size!r}"
        page.wait_for_timeout(50)


def wait_frame_height(page, tab: str, height: float, slack: float = 1.5) -> None:
    page.wait_for_function(
        "([selector, height, slack]) => Math.abs(document.querySelector(`${selector} iframe`)?.getBoundingClientRect().height - height) <= slack",
        arg=[card(tab), height, slack], timeout=8_000,
    )


def mark_frames(page) -> None:
    for tab in FRAMED:
        frame = page.locator(f"{card(tab)} iframe")
        frame.evaluate("frame => { frame.__boardMark = true; }")
        frame.element_handle().content_frame().evaluate("() => { window.__boardMark = 'same-window'; }")


def frames_kept(page, tabs=FRAMED) -> bool:
    for tab in tabs:
        frame = page.locator(f"{card(tab)} iframe")
        if frame.count() != 1 or not frame.evaluate("frame => frame.__boardMark === true"):
            return False
        if frame.element_handle().content_frame().evaluate("() => window.__boardMark ?? null") != "same-window":
            return False
    return True


def choose(page, tab: str, attribute: str, value: str) -> None:
    page.locator(f"{card(tab)} [data-widget-menu-trigger]").click()
    page.locator(f'body > .skills-card-menu-dialog[open] [{attribute}="{value}"]').click()


def drag(page, selector: str, dx: float, dy: float, *, at: str = "center", steps: int = 10) -> None:
    """Press on the handle (brought into view; a tall right edge is taken near its top), move, release."""
    handle = page.locator(selector)
    handle.scroll_into_view_if_needed()
    box = handle.bounding_box()
    x = box["x"] + (box["width"] / 2 if at == "center" else box["width"] - 2)
    y = box["y"] + (min(box["height"] / 2, 40) if at != "center" else box["height"] / 2)
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x + dx, y + dy, steps=steps)
    page.mouse.up()


def open_widgets(page, *, click: bool = True) -> None:
    """Show Widgets (by its nav row unless the page was opened at `#widgets`) and wait until every card runs."""
    if click:
        page.click('[data-nav-page="widgets"]')
    for tab in FRAMED:
        page.locator(f"{card(tab)} iframe").wait_for(state="attached", timeout=30_000)
    page.locator(f"{card('table')} table tbody tr").nth(29).wait_for(state="attached", timeout=15_000)
    for tab in FRAMED:
        page.wait_for_function(
            "(selector) => document.querySelector(selector)?.querySelector('[data-widget-status]')?.textContent.trim() === 'Running'",
            arg=card(tab), timeout=15_000)
    # The self-sizing frames have reported.
    for tab in ("short", "long", "capped", "scrollbar"):
        page.wait_for_function("(selector) => document.querySelector(`${selector} iframe`)?.dataset.widgetContentHeight",
                               arg=card(tab), timeout=15_000)


def evidence_dir(data_dir: pathlib.Path) -> pathlib.Path:
    target = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(data_dir.parent))) / "widgets-heights"
    target.mkdir(parents=True, exist_ok=True)
    return target


def launch(pw, engine: str, **context):
    from playwright.sync_api import Error as PlaywrightError

    try:
        browser = getattr(pw, engine).launch(headless=True)
    except PlaywrightError as exc:
        pytest.skip(f"{engine} is not installed: {exc}")
    return browser, browser.new_context(**context)


def enable(page) -> None:
    status = page.evaluate(
        """async (skill) => (await fetch(`/api/skills/${encodeURIComponent(skill)}/toggle`, {
            method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({enabled: true}),
        })).status""", SKILL)
    assert status == 200


# --- the board's heights ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("engine", ("chromium", "webkit"))
def test_board_heights_width_and_show_all_restart_nothing(direct_server_with_data, engine):
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    _write_skill(data_dir)
    shots = evidence_dir(data_dir)
    errors: list[str] = []

    with sync_playwright() as pw:
        browser, context = launch(pw, engine, viewport={"width": 1440, "height": 1000})
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            enable(page)
            open_widgets(page)
            board = facts(page)
            cards = board["cards"]

            # Natural heights: a short frame is as tall as its document (no 320 px of blank), a long one shows all of
            # it, the author's ceiling caps its frame and its document scrolls, fixed frames keep their height.
            short = child(page, "short")
            assert cards["short"]["frame"] < 120, cards["short"]
            assert short["scrollHeight"] <= short["innerHeight"] + 1, short
            long_child = child(page, "long")
            natural = cards["long"]["frame"]
            assert natural > 1000 and long_child["overflow"] == "hidden", (natural, long_child)
            assert long_child["scrollHeight"] <= long_child["innerHeight"] + 1, long_child
            assert abs(cards["capped"]["frame"] - 400) <= 1 and child(page, "capped")["overflow"] == "", cards["capped"]
            capped_child = child(page, "capped")
            assert capped_child["scrollHeight"] > capped_child["innerHeight"] + 100, capped_child
            assert abs(cards["fixed"]["frame"] - 720) <= 1 and abs(cards["route"]["frame"] - 320) <= 1
            assert not any(item["more"] for item in cards.values()), "nothing shows a Show all line yet"
            assert not any(item["bodyScroll"] > 1 for tab, item in cards.items() if tab not in FRAMED), "no host scroller"
            page.screenshot(path=str(shots / f"{engine}-natural.png"), full_page=True)

            mark_frames(page)
            field = page.frame_locator(f"{card('long')} iframe").locator("#field")
            field.fill("kept text")
            width_before = cards["long"]["width"]
            before_counts = counts(page)
            assert all(before_counts.get(f"{tab}:mount") == 1 for tab in ("short", "long", "capped", "fixed", "scrollbar")), before_counts

            # The bottom edge sets the content's height; the width stays the author's.
            drag(page, f"{card('long')} [data-widget-height-handle]", 0, -(natural - 420))
            stored = saved(page)["widget_size"][f"{SKILL}:long"]
            assert stored["w"] == 0 and 380 <= stored["h"] <= 460, stored
            h = stored["h"]
            wait_frame_height(page, "long", h)
            bounded = child(page, "long")
            assert bounded["overflow"] == "" and bounded["scrollHeight"] > bounded["innerHeight"] + 100, bounded
            assert bounded["viewport"] >= 1, "the board told the frame its new ceiling"
            board = facts(page)
            assert abs(board["cards"]["long"]["width"] - width_before) <= 1, "a height never pins a width"
            assert board["cards"]["long"]["more"].startswith("Show all"), board["cards"]["long"]
            hidden = int(board["cards"]["long"]["more"].split("·")[1].split("px")[0].replace(",", "").strip())
            assert abs(hidden - (natural - h)) <= 4, (hidden, natural, h)
            assert board["cards"]["long"]["bodyScroll"] <= 1, "the frame is the one scroller"
            page.locator(f"{card('long')} iframe").element_handle().content_frame().evaluate("() => window.scrollTo(0, 300)")
            assert child(page, "long")["scrollY"] >= 299
            page.screenshot(path=str(shots / f"{engine}-long-bounded.png"))

            # Show all lifts the height in this window only; Back returns to it. Neither writes a size.
            writes_before = saved(page)["widget_size"]
            page.locator(f"{card('long')} [data-widget-expand]").click()
            wait_frame_height(page, "long", natural, slack=4)
            assert child(page, "long")["overflow"] == "hidden"
            assert facts(page)["cards"]["long"]["more"] == f"Back to {h} px"
            page.screenshot(path=str(shots / f"{engine}-long-show-all.png"))
            page.locator(f"{card('long')} [data-widget-expand]").click()
            wait_frame_height(page, "long", h)
            assert child(page, "long")["overflow"] == ""
            assert saved(page)["widget_size"] == writes_before

            # The author's ceiling: Show all promises no more than it; the line counts only what the owner hides.
            choose(page, "capped", "data-widget-height", "360")
            wait_saved(page, "capped", {"w": 0, "h": 360})
            wait_frame_height(page, "capped", 360)
            assert facts(page)["cards"]["capped"]["more"] == "Show all · 40 px more"
            page.locator(f"{card('capped')} [data-widget-expand]").click()
            wait_frame_height(page, "capped", 400)
            assert child(page, "capped")["overflow"] == "", "at the author's ceiling the frame scrolls"
            page.locator(f"{card('capped')} [data-widget-expand]").click()
            wait_frame_height(page, "capped", 360)

            # A fixed frame takes the height as its viewport: its keys step it, Show all returns the declared height.
            page.locator(f"{card('fixed')} [data-widget-height-handle]").focus()
            page.keyboard.press("ArrowUp")
            page.keyboard.press("ArrowUp")
            wait_saved(page, "fixed", {"w": 0, "h": 640})
            wait_frame_height(page, "fixed", 640)
            assert page.locator("[data-widget-arrange-status]").text_content() == "Height: 640 px"
            assert facts(page)["cards"]["fixed"]["more"] == "Show all"
            page.locator(f"{card('fixed')} [data-widget-expand]").click()
            wait_frame_height(page, "fixed", 720)
            page.locator(f"{card('fixed')} [data-widget-expand]").click()
            wait_frame_height(page, "fixed", 640)

            # A route frame grows past its declared height: a bigger working area, nothing hidden to show.
            choose(page, "route", "data-widget-height", "560")
            wait_frame_height(page, "route", 560)
            assert facts(page)["cards"]["route"]["more"] == ""

            # A host-drawn card scrolls in its body under a height; Show all and Back as for a frame.
            table_natural = facts(page)["cards"]["table"]["height"]
            choose(page, "table", "data-widget-height", "360")
            wait_saved(page, "table", {"w": 0, "h": 360})
            page.wait_for_function("(selector) => document.querySelector(selector).querySelector('[data-widget-mount]').hasAttribute('data-widget-bounded')",
                                   arg=card("table"), timeout=5_000)
            table = facts(page)["cards"]["table"]
            assert table["bodyScroll"] > 100 and table["more"].startswith("Show all"), table
            page.locator(f"{card('table')} [data-widget-expand]").click()
            page.wait_for_function("(selector) => !document.querySelector(selector).querySelector('[data-widget-mount]').hasAttribute('data-widget-bounded')",
                                   arg=card("table"), timeout=5_000)
            expanded = facts(page)["cards"]["table"]
            assert expanded["bodyScroll"] <= 1 and expanded["more"] == "Back to 360 px", expanded
            assert expanded["height"] >= table_natural, (expanded, table_natural)
            page.locator(f"{card('table')} [data-widget-expand]").click()

            # The right edge sets the width and keeps the height; Natural height keeps the width; Reset clears both.
            column = facts(page)["cards"]["short"]["width"]
            page.evaluate("(selector) => document.querySelector(selector).scrollIntoView({block: 'start'})", card("long"))
            drag(page, f"{card('long')} [data-widget-resize-handle]", column + GAP_PX, 0, at="edge")
            # A second column for this card can change the board's column count, so the step nearest the edge is the
            # masonry's to say; what the owner asked for is a wider card, and its height stays.
            wait_until(page, "async (key) => (((await (await fetch('/api/ui/preferences')).json()).widget_size || {})[key]?.w || 0) >= 2",
                       arg=f"{SKILL}:long")
            widened = saved(page)["widget_size"][f"{SKILL}:long"]
            assert widened["h"] == h, widened
            choose(page, "long", "data-widget-height", "0")
            wait_saved(page, "long", {"w": widened["w"], "h": 0})
            page.wait_for_function("(selector) => document.querySelector(`${selector} iframe`).getBoundingClientRect().height > 1000",
                                   arg=card("long"), timeout=8_000)
            choose(page, "long", "data-widget-size", "reset")
            wait_saved(page, "long", None)

            # Nothing above restarted a widget or moved a frame.
            assert frames_kept(page), "a frame was replaced"
            assert child(page, "long")["field"] == "kept text"
            after = counts(page)
            for tab in ("short", "long", "capped", "fixed", "scrollbar"):
                assert after.get(f"{tab}:mount") == 1 and not after.get(f"{tab}:dispose"), (tab, after)
            page.screenshot(path=str(shots / f"{engine}-after-sizing.png"), full_page=True)
            assert errors == [], errors
        finally:
            context.close()
            browser.close()


# --- a scrollbar that takes layout width -------------------------------------------------------------------------------

@pytest.mark.parametrize("engine", ("chromium", "webkit"))
def test_a_scrollbar_that_takes_width_settles_inside_a_bounded_frame(direct_server_with_data, engine):
    """Released at the owner's ceiling, a module's own scrollbar takes 14 px of its width and its text wraps longer:
    the frame keeps the owner's height and the reports settle instead of chasing each other, and Show all hides the
    scrollbar again. Playwright starts headless Chromium with its scrollbars hidden; this case starts it without that
    switch, so the scrollbar takes layout width as it does for an owner whose system shows scrollbars."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    _write_skill(data_dir)
    shots = evidence_dir(data_dir)
    errors: list[str] = []

    with sync_playwright() as pw:
        launch_options = {"headless": True}
        if engine == "chromium":
            launch_options["ignore_default_args"] = ["--hide-scrollbars"]
        try:
            browser = getattr(pw, engine).launch(**launch_options)
        except PlaywrightError as exc:
            pytest.skip(f"{engine} is not installed: {exc}")
        context = browser.new_context(viewport={"width": 1440, "height": 1000})
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            enable(page)
            open_widgets(page)
            natural = child(page, "scrollbar")
            assert natural["overflow"] == "hidden" and natural["innerWidth"] == natural["clientWidth"], natural
            choose(page, "scrollbar", "data-widget-height", "360")
            wait_frame_height(page, "scrollbar", 360)
            page.wait_for_timeout(400)
            settled = child(page, "scrollbar")
            assert settled["overflow"] == "" and settled["scrollHeight"] > settled["innerHeight"] + 100, settled
            assert settled["innerWidth"] - settled["clientWidth"] in (14, 15), f"no layout scrollbar: {settled}"
            page.wait_for_timeout(1200)
            later = child(page, "scrollbar")
            assert later["reports"] - settled["reports"] <= 1, (settled, later)
            assert abs(facts(page)["cards"]["scrollbar"]["frame"] - 360) <= 1
            page.screenshot(path=str(shots / f"{engine}-layout-scrollbar.png"))
            page.locator(f"{card('scrollbar')} [data-widget-expand]").click()
            page.wait_for_function("(selector) => document.querySelector(`${selector} iframe`).getBoundingClientRect().height > 400",
                                   arg=card("scrollbar"), timeout=8_000)
            page.wait_for_timeout(400)
            shown = child(page, "scrollbar")
            assert shown["overflow"] == "hidden" and shown["innerWidth"] == shown["clientWidth"], shown
            assert shown["scrollHeight"] <= shown["innerHeight"] + 1, shown
            assert errors == [], errors
        finally:
            context.close()
            browser.close()


# --- moving, leaving, returning, reloading ----------------------------------------------------------------------------

ORDER = """async () => ((await (await fetch('/api/ui/preferences')).json()).widget_order || [])"""


def stored_order(page) -> list:
    return [key.split(":", 1)[1] for key in page.evaluate(ORDER) if key.startswith(f"{SKILL}:")]


def visual_order(page) -> list:
    """The cards where the masonry puts them (its target positions, not a glide's passing frame): top to bottom,
    then left to right. Waits for a drop's glide to end first."""
    page.wait_for_function("() => !document.getElementById('widgets-list').hasAttribute('data-widget-arranging')",
                           timeout=5_000)
    placed = page.evaluate(
        """(skill) => [...document.querySelectorAll('#widgets-list [data-widget-key]:not([data-widget-removed])')]
            .map((node) => [node.dataset.widgetKey.slice(skill.length + 1),
                parseFloat(node.style.getPropertyValue('--masonry-y')) || 0,
                parseFloat(node.style.getPropertyValue('--masonry-x')) || 0])""", SKILL)
    return [tab for tab, _, _ in sorted(placed, key=lambda item: (round(item[1]), round(item[2])))]


def touch_drag(page, selector: str, dx: float, dy: float, steps: int = 12) -> None:
    """A real touch gesture (Chromium's CDP touch input): press, move in steps, lift."""
    session = page.context.new_cdp_session(page)
    box = page.locator(selector).bounding_box()
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    session.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{"x": x, "y": y}]})
    for step in range(1, steps + 1):
        session.send("Input.dispatchTouchEvent", {"type": "touchMove", "touchPoints": [
            {"x": x + dx * step / steps, "y": y + dy * step / steps}]})
        page.wait_for_timeout(16)
    session.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
    session.detach()


@pytest.mark.parametrize("engine", ("chromium", "webkit"))
def test_board_carry_keys_menu_and_returning_keep_every_frame(direct_server_with_data, engine):
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    _write_skill(data_dir)
    shots = evidence_dir(data_dir)
    errors: list[str] = []

    with sync_playwright() as pw:
        browser, context = launch(pw, engine, viewport={"width": 1440, "height": 1000}, has_touch=engine == "chromium")
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            enable(page)
            open_widgets(page)
            mark_frames(page)
            start = visual_order(page)
            # The second card of the first row, carried onto the first: both in view at the top of the board.
            mover, first_card = start[1], start[0]

            def stored_at(index: int) -> None:
                wait_until(page,
                    "async ([key, index]) => ((await (await fetch('/api/ui/preferences')).json()).widget_order || []).at(index) === key",
                    arg=[f"{SKILL}:{mover}", index], timeout=8_000)

            def carry_to_first(press_only: bool = False):
                page.evaluate("() => document.querySelector('.widgets-scroll').scrollTo(0, 0)")
                grip_box = page.locator(f"{card(mover)} [data-widget-reorder-handle]").bounding_box()
                head = page.locator(f"{card(visual_order(page)[0])} .widgets-card-head").bounding_box()
                page.mouse.move(grip_box["x"] + grip_box["width"] / 2, grip_box["y"] + grip_box["height"] / 2)
                page.mouse.down()
                page.mouse.move(head["x"] + 40, head["y"] + 20, steps=14)
                if not press_only:
                    page.mouse.up()

            # Carry: the card follows the pointer, the board outlines where it lands; Escape puts it back.
            carry_to_first(press_only=True)
            page.wait_for_function("() => document.getElementById('widgets-list').dataset.widgetMoving === 'true'", timeout=5_000)
            assert page.locator(f"{card(mover)}[data-widget-lifted]").count() == 1
            page.wait_for_timeout(300)
            page.screenshot(path=str(shots / f"{engine}-carry.png"))
            page.keyboard.press("Escape")
            page.mouse.up()
            page.wait_for_function("() => !document.getElementById('widgets-list').dataset.widgetMoving", timeout=5_000)
            assert stored_order(page) == [] and visual_order(page) == start, "Escape put it back and stored nothing"

            # Release puts it there, stored.
            carry_to_first()
            stored_at(0)
            page.wait_for_timeout(400)
            assert visual_order(page)[0] == mover
            assert "place 1 of" in page.locator("[data-widget-arrange-status]").text_content()

            # The grip's keys and the menu step it; the menu offers only the moves its place allows.
            page.locator(f"{card(mover)} [data-widget-reorder-handle]").focus()
            page.keyboard.press("ArrowRight")
            stored_at(1)
            choose(page, mover, "data-widget-move", "end")
            stored_at(-1)
            page.locator(f"{card(mover)} [data-widget-menu-trigger]").click()
            menu = page.locator("body > .skills-card-menu-dialog[open]")
            assert menu.locator('[data-widget-move="later"]').is_hidden() and menu.locator('[data-widget-move="end"]').is_hidden()
            assert menu.locator('[data-widget-move="earlier"]').is_visible()
            page.screenshot(path=str(shots / f"{engine}-menu.png"))
            page.keyboard.press("Escape")
            choose(page, mover, "data-widget-move", "start")
            stored_at(0)
            assert first_card != mover

            # A finger carries it by the grip (Chromium's touch input; WebKit headless offers no touch gesture here).
            if engine == "chromium":
                page.evaluate("() => document.querySelector('.widgets-scroll').scrollTo(0, 0)")
                page.wait_for_timeout(300)
                second = visual_order(page)[1]
                grip_box = page.locator(f"{card(mover)} [data-widget-reorder-handle]").bounding_box()
                head = page.locator(f"{card(second)} .widgets-card-head").bounding_box()
                touch_drag(page, f"{card(mover)} [data-widget-reorder-handle]",
                           head["x"] + head["width"] - 60 - (grip_box["x"] + grip_box["width"] / 2),
                           head["y"] + 20 - (grip_box["y"] + grip_box["height"] / 2))
                stored_at(1)
            assert frames_kept(page), "moving replaced a frame"
            moved = counts(page)
            assert all(moved.get(f"{tab}:mount") == 1 and not moved.get(f"{tab}:dispose")
                       for tab in ("short", "long", "capped", "fixed", "scrollbar")), moved

            # A height survives leaving and returning: the frame comes back as a new document at that height at
            # once, its document scrolls from its first report, and the board does not move meanwhile.
            choose(page, "long", "data-widget-height", "360")
            wait_saved(page, "long", {"w": 0, "h": 360})
            wait_frame_height(page, "long", 360)
            page.click('[data-nav-page="chat"]')
            wait_until(page, f"async () => ((await (await fetch('/api/extensions/{SKILL}/counts')).json())['long:dispose'] || 0) >= 1",
                                   timeout=10_000)
            page.evaluate(
                """(selector) => {
                    const card = document.querySelector(selector);
                    window.__returnHeights = [];
                    window.__returnFrames = [];
                    new ResizeObserver(() => window.__returnHeights.push(Math.round(card.getBoundingClientRect().height))).observe(card);
                    new MutationObserver(() => {
                        const frame = card.querySelector('iframe');
                        if (frame && !frame.__seen) {
                            frame.__seen = true;
                            window.__returnFrames.push(frame.style.getPropertyValue('--widget-frame-height'));
                        }
                    }).observe(card, { childList: true, subtree: true });
                }""", card("long"))
            page.click('[data-nav-page="widgets"]')
            wait_until(page, f"async () => ((await (await fetch('/api/extensions/{SKILL}/counts')).json())['long:mount'] || 0) >= 2",
                                   timeout=15_000)
            page.wait_for_function("(selector) => document.querySelector(`${selector} iframe`)?.dataset.widgetContentHeight",
                                   arg=card("long"), timeout=15_000)
            returned = page.evaluate("() => ({ heights: window.__returnHeights, frames: window.__returnFrames })")
            assert returned["frames"][:1] == ["360px"], returned
            # 0 is the hidden page the observer was installed on; once shown the card keeps one height.
            shown = [height for height in returned["heights"] if height > 0]
            assert len(set(shown)) <= 1, f"the card moved while it came back: {returned}"
            remount = child(page, "long")
            assert remount["overflow"] == "" and remount["viewport"] == 0, "scrolls from its first report, unasked"
            assert remount["scrollHeight"] > remount["innerHeight"] + 100

            # A reload reads the sizes back; Show all is not among them.
            page.locator(f"{card('long')} [data-widget-expand]").click()
            page.reload(wait_until="domcontentloaded", timeout=30_000)
            open_widgets(page)
            wait_frame_height(page, "long", 360)
            assert facts(page)["cards"]["long"]["more"].startswith("Show all")

            # A narrow window is one column: heights still apply, the right edge is gone.
            page.set_viewport_size({"width": 600, "height": 900})
            page.wait_for_function("() => document.getElementById('widgets-list').dataset.widgetLayout === 'stack'", timeout=5_000)
            wait_frame_height(page, "long", 360)
            assert page.locator(f"{card('long')} [data-widget-resize-handle]").is_hidden()
            if page.evaluate("() => document.documentElement.dataset.pointer") == "coarse":
                # Under a finger the bottom edge is the grip of the one card its menu's Resize chose, and of no other.
                assert page.locator(f"{card('long')} [data-widget-height-handle]").is_hidden()
                page.locator(f"{card('long')} [data-widget-menu-trigger]").click()
                page.locator("body > .skills-card-menu-dialog[open] [data-widget-adjust]").click()
            assert page.locator(f"{card('long')} [data-widget-height-handle]").is_visible()
            page.keyboard.press("Escape")
            assert page.evaluate("() => document.querySelectorAll('[data-widget-adjusting]').length") == 0
            page.set_viewport_size({"width": 1440, "height": 1000})
            page.wait_for_function("() => document.getElementById('widgets-list').dataset.widgetLayout === 'columns'", timeout=5_000)
            assert errors == [], errors
        finally:
            context.close()
            browser.close()


# --- a press keeps its pointer over the card's own frame --------------------------------------------------------------

HEARD = """() => { window.__heard = 0;
    for (const type of ['pointermove', 'pointerup', 'mouseup']) document.addEventListener(type, () => { window.__heard += 1; }, true); }"""


@pytest.mark.parametrize("engine", ("chromium", "webkit"))
def test_a_press_on_a_head_or_grip_ends_where_it_is_released_even_over_its_own_frame(direct_server_with_data, engine):
    """A mouse press on a card's head or grip that travels into the card's own frame lifts the card there, and its
    release over the frame ends the press: nothing stays lifted, the board's order is untouched and nothing is saved by
    the pointer's later moves or clicks; the frame hears none of it. A hand's drag lifts wherever it starts, from a move
    that comes a frame after the press (`settled`) or from moves that start at once, every frame (`hand`, its first ones
    may still reach the frame). A jump into the frame within the press's own painted frame (`instant`) may reach the
    frame whole in an engine that routes the pointer by what it last painted (Chromium's out-of-process frames), and the
    board then ends the press cleanly: nothing lifted after it, nothing saved. Paired: the same press carried down over
    its frame and onto another card's place moves it there."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    _write_skill(data_dir)
    shots = evidence_dir(data_dir)
    report = {}
    errors: list[str] = []

    def lifted() -> int:
        return page.locator("[data-widget-lifted]").count()

    def board_state() -> dict:
        return page.evaluate("""() => { const list = document.getElementById('widgets-list');
            return { lifted: list.querySelectorAll('[data-widget-lifted]').length, moving: list.dataset.widgetMoving || '',
                     arranging: list.hasAttribute('data-widget-arranging') }; }""")

    with sync_playwright() as pw:
        browser, context = launch(pw, engine, viewport={"width": 1440, "height": 1000})
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            enable(page)
            open_widgets(page)
            mark_frames(page)
            start = visual_order(page)
            inner = page.locator(f"{card('long')} iframe").element_handle().content_frame()
            for label, (grip, from_bottom, into_frame, steps, pace) in {
                "head_small_nudge": (False, 2, 8, 2, "settled"), "head_flick": (False, 20, 30, 1, "settled"),
                "grip_down": (True, None, 16, 4, "settled"), "head_hand_speed": (False, 2, 30, 6, "hand"),
                "head_flick_instant": (False, 20, 30, 1, "instant"),
            }.items():
                page.evaluate("() => document.querySelector('.widgets-scroll').scrollTo(0, 0)")
                page.locator(card("long")).scroll_into_view_if_needed()
                page.wait_for_timeout(200)
                inner.evaluate(HEARD)
                head = page.locator(f"{card('long')} .widgets-card-head").bounding_box()
                frame = page.locator(f"{card('long')} iframe").bounding_box()
                if grip:
                    box = page.locator(f"{card('long')} [data-widget-reorder-handle]").bounding_box()
                    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
                else:
                    x, y = head["x"] + 60, head["y"] + head["height"] - from_bottom
                page.mouse.move(x, y)
                page.mouse.down()
                target = frame["y"] + into_frame
                if pace == "hand":
                    for step in range(1, steps + 1):
                        page.mouse.move(x, y + (target - y) * step / steps)
                        page.wait_for_timeout(16)
                else:
                    if pace == "settled":
                        page.wait_for_timeout(50)
                    page.mouse.move(x, target, steps=steps)
                during = lifted()
                page.mouse.up()
                heard = inner.evaluate("() => window.__heard")
                # The place's outline goes with the next layout, an animation frame after the release.
                page.wait_for_timeout(150)
                released = board_state()
                page.wait_for_timeout(250)
                placed = {"stored": stored_order(page), "shown": visual_order(page)}
                # The pointer, its button up, goes on over the board, over another card's head, and clicks there.
                other = page.locator(f"{card(start[0] if start[0] != 'long' else start[1])} .widgets-card-head").bounding_box()
                page.mouse.move(other["x"] + 60, other["y"] + 10, steps=4)
                page.mouse.move(other["x"] + 90, other["y"] + 14, steps=4)
                page.wait_for_timeout(300)
                later = board_state()
                page.mouse.click(other["x"] + 90, other["y"] + 14)
                page.wait_for_timeout(400)
                report[label] = {"during": during, "heard": heard, "released": released, "later": later, "placed": placed,
                                 "stored": stored_order(page), "shown": visual_order(page)}
                if pace != "instant" or engine == "webkit":
                    assert during == 1, ("the press lifts the card over its own frame", report[label])
                if pace == "settled" or engine == "webkit":
                    assert heard == 0, ("the frame hears nothing of the press", report[label])
                assert released["lifted"] == 0 and not released["moving"], ("released over the frame", report[label])
                assert later["lifted"] == 0 and not later["moving"], ("nothing carried after the release", report[label])
                assert report[label]["stored"] == placed["stored"] and report[label]["shown"] == placed["shown"], (
                    "nothing changed after the release", report[label])
                if not grip:
                    assert placed["stored"] == [] and placed["shown"] == start, ("a short travel moves nothing", report[label])
            page.screenshot(path=str(shots / f"{engine}-press-over-own-frame.png"))

            # Paired: carried down over its own frame and on to another card's place in the first row, it lands there and
            # is stored.
            page.evaluate("() => document.querySelector('.widgets-scroll').scrollTo(0, 0)")
            page.wait_for_timeout(200)
            shown = visual_order(page)
            index = 0 if shown[0] != "long" else 1
            box = page.locator(f"{card('long')} [data-widget-reorder-handle]").bounding_box()
            frame = page.locator(f"{card('long')} iframe").bounding_box()
            target = page.locator(f"{card(shown[index])} .widgets-card-head").bounding_box()
            page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            page.mouse.down()
            page.mouse.move(box["x"] + box["width"] / 2, frame["y"] + 120, steps=6)
            page.mouse.move(target["x"] + 40, target["y"] + 20, steps=14)
            page.mouse.up()
            wait_until(page,
                "async ([key, index]) => ((await (await fetch('/api/ui/preferences')).json()).widget_order || [])[index] === key",
                arg=[f"{SKILL}:long", index], timeout=8_000)
            assert board_state()["lifted"] == 0
            assert frames_kept(page), "moving replaced a frame"
            assert errors == [], errors
        finally:
            (shots / f"{engine}-press-over-own-frame.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            context.close()
            browser.close()


# --- a slow start holds back only its own card ------------------------------------------------------------------------

HOLDS = """(key) => { try { return Boolean(JSON.parse(localStorage.getItem('ouroboros.holds') || '{}').entries?.[key]); }
    catch (e) { return false; } }"""


@pytest.mark.parametrize("engine", ("chromium", "webkit"))
def test_a_held_lease_holds_back_only_its_card_and_the_ledger_waits_for_the_frame(direct_server_with_data, engine):
    """The route frame's lease is held at the network: every other card starts and is usable meanwhile, the held
    card records no start in the ledger (no code of it runs yet), leaving Widgets abandons it, and returning starts
    it once. A revision change replaces a running card in order and its height carries over; a preferences reply
    that was read before a height was written never undoes it."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    _write_skill(data_dir)
    errors: list[str] = []
    held: list = []
    gate = {"hold": True}
    revisions: dict[str, str] = {}

    def leases(route):
        body = route.request.post_data_json or {}
        if gate["hold"] and body.get("tab_id") == "route" and "token" not in body:
            held.append(route)
        else:
            route.continue_()

    def widgets_list(route):
        data = route.fetch().json()
        for tab in data.get("ui_tabs", []):
            if tab.get("key") in revisions:
                tab["revision"] = revisions[tab["key"]]
        route.fulfill(status=200, content_type="application/json", body=json.dumps(data))

    with sync_playwright() as pw:
        browser, context = launch(pw, engine, viewport={"width": 1440, "height": 1000})
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            enable(page)
            page.route("**/api/widgets/asset-leases", leases)
            page.click('[data-nav-page="widgets"]')
            for tab in ("short", "long", "capped", "scrollbar", "bridged"):
                page.wait_for_function(
                    "(selector) => document.querySelector(selector)?.querySelector('[data-widget-status]')?.textContent.trim() === 'Running'",
                    arg=card(tab), timeout=20_000)
            deadline = time.monotonic() + 10
            while not held and time.monotonic() < deadline:
                page.wait_for_timeout(50)
            assert len(held) == 1, "the route frame's lease reached the network and is held"
            route_card = page.locator(card("route"))
            assert route_card.locator("iframe").count() == 0
            assert route_card.locator("[data-widget-status]").text_content().strip() == "Starting…"
            assert not page.evaluate(HOLDS, f"widget:{SKILL}:route"), "no start is recorded before the frame exists"
            # A neighbour is usable now, not after the held card.
            page.frame_locator(f"{card('long')} iframe").locator("#field").fill("usable")
            assert child(page, "long")["field"] == "usable"

            # Leaving abandons the held start; returning starts it once.
            page.click('[data-nav-page="chat"]')
            page.wait_for_timeout(300)
            gate["hold"] = False
            for route in held:
                try:
                    route.continue_()
                except Exception:  # noqa: BLE001 - the page already gave up the request
                    pass
            page.click('[data-nav-page="widgets"]')
            page.wait_for_function(
                "(selector) => document.querySelector(selector)?.querySelector('[data-widget-status]')?.textContent.trim() === 'Running'",
                arg=card("route"), timeout=20_000)
            assert route_card.locator("iframe").count() == 1
            page.wait_for_function(f"(key) => !({HOLDS})(key)", arg=f"widget:{SKILL}:route", timeout=10_000)

            # A revision change: the running card stops in order and starts again, at its height.
            choose(page, "long", "data-widget-height", "360")
            wait_saved(page, "long", {"w": 0, "h": 360})
            before = counts(page)
            page.route("**/api/widgets", widgets_list)
            revisions[f"{SKILL}:long"] = "e" * 64
            page.evaluate("() => { window.__ouroWs.emit('open', {previouslyConnected: true}); }")
            wait_until(page, f"async () => ((await (await fetch('/api/extensions/{SKILL}/counts')).json())['long:mount'] || 0) > {before.get('long:mount', 0)}",
                                   timeout=15_000)
            after = counts(page)
            assert after.get("long:dispose", 0) == before.get("long:dispose", 0) + 1, (before, after)
            wait_frame_height(page, "long", 360)

            # A preferences reply read before a height was written arrives after it: the height stays.
            stale: list = []

            def hold_preferences(route):
                if route.request.method == "GET" and not stale:
                    stale.append((route, route.fetch()))
                else:
                    route.continue_()

            page.route("**/api/ui/preferences", hold_preferences)
            page.evaluate("() => { window.__ouroWs.emit('open', {previouslyConnected: true}); }")
            deadline = time.monotonic() + 10
            while not stale and time.monotonic() < deadline:
                page.wait_for_timeout(50)
            assert stale, "the list read's preferences request was held"
            choose(page, "short", "data-widget-height", "560")
            wait_saved(page, "short", {"w": 0, "h": 560})
            stale_route, stale_response = stale[0]
            stale_route.fulfill(response=stale_response)
            page.wait_for_timeout(800)
            assert page.evaluate("(selector) => document.querySelector(selector).querySelector('[data-widget-height=\"560\"]') !== null",
                                 card("short"))
            page.locator(f"{card('short')} [data-widget-menu-trigger]").click()
            assert page.locator('body > .skills-card-menu-dialog[open] [data-widget-height="560"]').get_attribute("aria-checked") == "true"
            page.keyboard.press("Escape")
            assert errors == [], errors
        finally:
            context.close()
            browser.close()


# --- a frame in a message keeps its own size policy ---------------------------------------------------------------------

@pytest.mark.parametrize("engine", ("chromium", "webkit"))
def test_a_widget_in_a_message_is_still_cut_until_expand_and_hears_nothing_from_the_board(direct_server_with_data, engine):
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    from ouroboros import dialogue_store
    from ouroboros.card_store import pin_card, write_card

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    _write_skill(data_dir)
    shots = evidence_dir(data_dir)
    record = write_card(data_dir, chat_id=1, task_id="", title="Long list", fallback_text="A long list",
                        kind="widget", body={"widget": {"skill": SKILL, "tab_id": "long"}})
    dialogue_store.append(data_dir, "chat", {
        "ts": "2026-11-08T10:00:00+00:00", "direction": "out", "chat_id": 1, "user_id": 0, "task_id": "",
        "text": record["fallback_text"], "type": "card",
        "card": {"card_id": record["card_id"], "kind": "widget", "title": record["title"], "revision": 1}})
    kept = write_card(data_dir, chat_id=1, task_id="", title="Kept table", fallback_text="Thirty rows",
                      kind="components", body={"components": [{"type": "table", "path": "rows", "columns": [
                          {"label": "Row", "path": "n"}]}], "data": {"rows": [{"n": i} for i in range(30)]}})
    pin_card(data_dir, kept["card_id"])
    errors: list[str] = []

    with sync_playwright() as pw:
        browser, context = launch(pw, engine, viewport={"width": 1280, "height": 900})
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            enable(page)
            # The board gives the same widget a height first: nothing of that reaches the message.
            open_widgets(page)
            choose(page, "long", "data-widget-height", "360")
            wait_frame_height(page, "long", 360)
            # The kept card is host-drawn: a height makes its body the one scroller.
            kept_key = page.locator(".widgets-card:has(.widgets-card-source)").get_attribute("data-widget-key")
            assert kept_key.endswith(f':{kept["card_id"]}'), kept_key
            page.locator(f'[data-widget-key="{kept_key}"] [data-widget-menu-trigger]').click()
            page.locator('body > .skills-card-menu-dialog[open] [data-widget-height="360"]').click()
            page.wait_for_function("(key) => document.querySelector(`[data-widget-key=\"${key}\"] [data-widget-mount]`)?.hasAttribute('data-widget-bounded')",
                                   arg=kept_key, timeout=5_000)
            assert page.evaluate("(key) => { const b = document.querySelector(`[data-widget-key=\"${key}\"] [data-widget-mount]`); return b.scrollHeight > b.clientHeight; }",
                                 kept_key)

            page.click('[data-nav-page="chat"]')
            frame = page.locator(f'.chat-card[data-card-id="{record["card_id"]}"] iframe')
            frame.wait_for(state="attached", timeout=30_000)
            inline = frame.element_handle().content_frame()
            inline.wait_for_function("() => window.__reports >= 1", timeout=15_000)
            cap = int(900 * 0.7)
            page.wait_for_function("([selector, cap]) => Math.abs(document.querySelector(selector).getBoundingClientRect().height - cap) <= 1.5",
                                   arg=[f'.chat-card[data-card-id="{record["card_id"]}"] iframe', cap], timeout=10_000)
            facts_inline = inline.evaluate(CHILD)
            assert facts_inline["overflow"] == "hidden", "a frame in a message is cut until Expand, not scrolled"
            assert facts_inline["viewport"] == 0, "no board message reaches a frame in a message"
            page.screenshot(path=str(shots / f"{engine}-inline-cut.png"))
            toggle = page.locator(f'.chat-card[data-card-id="{record["card_id"]}"] .chat-card-expand')
            toggle.click()
            page.wait_for_function("([selector, cap]) => document.querySelector(selector).getBoundingClientRect().height > cap + 200",
                                   arg=[f'.chat-card[data-card-id="{record["card_id"]}"] iframe', cap], timeout=10_000)
            assert inline.evaluate(CHILD)["viewport"] == 0
            assert errors == [], errors
        finally:
            context.close()
            browser.close()


# --- documents that follow their frame, and whole documents --------------------------------------------------------
# A self-sizing board frame is exactly as tall as its document. A document that follows its own viewport (a 100% root,
# a 100vh root, a box sized from innerHeight) has no height of its own: with nothing outside it, it rests at the frame
# it has; with margins outside it, it grows by them at every report until a ceiling stops it (the author's max_height,
# or the owner's height, where it scrolls and nothing more is reported; Show all and Back move between them). A body
# made as tall as the viewport by its own min-height holds only its content. An ordinary document counts whole,
# margins included, also when its content grows to exactly the frame's height, and also when Show all lets margins
# that had collapsed through the body back in.

_FOLLOW_CSS = {
    "fill": "html,body{height:100%;margin:0} #root{height:100%;box-sizing:border-box;padding:8px}",
    "vh": "body{margin:0} #root{min-height:100vh;box-sizing:border-box;padding:8px}",
    "vh_margins": "#root{min-height:100vh}",
    "vh_capped": "#root{min-height:100vh}",
    "reset": "body{min-height:100vh;margin:0} #root{padding:8px}",
}
_FOLLOW_STYLE = """(() => {{
    const style = document.createElement('style');
    style.textContent = {css!r};
    document.head.appendChild(style);
    document.getElementById('root').innerHTML = '<p>{text}</p>';
}})();
"""
_CANVAS = """(() => {
    document.body.style.margin = '0';
    const box = document.createElement('div');
    box.style.background = '#234';
    document.getElementById('root').appendChild(box);
    const fit = () => { box.style.height = window.innerHeight + 'px'; };
    window.addEventListener('resize', fit);
    fit();
})();
"""
_TAIL = '<button id="tail" style="position:absolute;bottom:0;height:20px">Bottom control</button>'
# Grows to exactly its frame's height when the test asks (8 px default body margins).
_GROWS = ("const root = document.getElementById('root');"
          "root.style.cssText = 'height:200px;position:relative';"
          f"root.innerHTML = 'Grows {_TAIL}';"
          "window.growToFrame = () => { root.style.height = window.innerHeight + 'px'; };\n")
# Margins that collapse through the body while its overflow is released (body margin 20px 8px 0).
_PARAGRAPH = ("document.body.style.margin = '20px 8px 0';"
              "document.getElementById('root').innerHTML = '<p id=\"para\" style=\"height:210px;margin:16px 0;"
              f"position:relative\">Paragraph {_TAIL}</p>';\n")
# The second review's document: an authored 500 px body, a 20 px root, a control placed at the body's bottom.
_ANCHORED = ("document.head.insertAdjacentHTML('beforeend', '<style>body{height:500px;position:relative;margin:0}"
             "#root{height:20px}</style>');"
             f"document.getElementById('root').innerHTML = 'Start {_TAIL}';\n")
# Four documents laid out against their own window: the default body margins (or a body holding 900 px of its own) leave
# their bottom control below every frame they are given, by a flex footer or an absolutely placed control.
_BOTTOM_CONTROL = '<button id="tail" type="button" style="height:20px">Bottom control</button>'
_PLACED_CONTROL = '<button id="tail" type="button" style="position:absolute;bottom:0;height:20px">Bottom control</button>'
_TAIL_COUNTS = "document.getElementById('tail').addEventListener('click', () => { window.__tailClicks = (window.__tailClicks || 0) + 1; });\n"
_FOLLOWERS = {
    "flex_bottom": ("document.head.insertAdjacentHTML('beforeend', '<style>body{min-height:100vh;display:flex;"
                    "flex-direction:column}footer{margin-top:auto}</style>');\n"
                    "document.getElementById('root').textContent = 'Content';\n"
                    f"document.body.insertAdjacentHTML('beforeend', '<footer>{_BOTTOM_CONTROL}</footer>');\n" + _TAIL_COUNTS),
    "abs_bottom": ("document.head.insertAdjacentHTML('beforeend', '<style>body{min-height:100vh;position:relative}</style>');\n"
                   f"document.getElementById('root').innerHTML = 'Content {_PLACED_CONTROL}';\n" + _TAIL_COUNTS),
    "root_bottom": ("document.head.insertAdjacentHTML('beforeend', '<style>#root{min-height:100vh;position:relative}</style>');\n"
                    f"document.getElementById('root').innerHTML = '<p>Content</p>{_PLACED_CONTROL}';\n" + _TAIL_COUNTS),
    "mixed900": ("document.head.insertAdjacentHTML('beforeend', '<style>body{min-height:100vh;display:flex;"
                 "flex-direction:column}footer{margin-top:auto}</style>');\n"
                 "document.getElementById('root').innerHTML = '<div style=\"height:900px;background:#234\">Tall</div>';\n"
                 f"document.body.insertAdjacentHTML('beforeend', '<footer>{_BOTTOM_CONTROL}</footer>');\n" + _TAIL_COUNTS),
    # Three whose extra grows with the frame, never the same twice: a 110vh root; a 100vh body with 5vh padding above
    # and below its content box; a 105vh root between the default 8 px margins. Each control sits at its box's bottom.
    "prop_root": ("document.head.insertAdjacentHTML('beforeend', '<style>body{margin:0}#root{height:110vh;"
                  "position:relative}</style>');\n"
                  f"document.getElementById('root').innerHTML = '<p>Content</p>{_PLACED_CONTROL}';\n" + _TAIL_COUNTS),
    "prop_body": ("document.head.insertAdjacentHTML('beforeend', '<style>body{margin:0;min-height:100vh;padding:5vh 0;"
                  "box-sizing:content-box;position:relative}</style>');\n"
                  f"document.getElementById('root').innerHTML = '<p>Content</p>{_PLACED_CONTROL}';\n" + _TAIL_COUNTS),
    "prop_mixed": ("document.head.insertAdjacentHTML('beforeend', '<style>#root{min-height:105vh;position:relative}"
                   "</style>');\n"
                   f"document.getElementById('root').innerHTML = '<p>Content</p>{_PLACED_CONTROL}';\n" + _TAIL_COUNTS),
}
# Two documents sized as a fraction of their frame: a 60vh root, and a list capped at 70vh above its button. Each smaller
# frame makes them smaller until their content stops it (docs/CREATING_SKILLS.md: they have no height of their own).
_FRACTIONS = {
    "frac_root": ("document.head.insertAdjacentHTML('beforeend', '<style>#root{height:60vh;background:#234;color:#fff}"
                  "</style>');\ndocument.getElementById('root').textContent = 'Chart at 60vh';\n"),
    "frac_list": ("document.head.insertAdjacentHTML('beforeend', '<style>#box{max-height:70vh;overflow:auto}</style>');\n"
                  "document.getElementById('root').innerHTML = '<div id=\"box\">' + Array.from({ length: 80 }, (_, i) =>"
                  f" '<div>List row ' + i + '</div>').join('') + '</div>{_BOTTOM_CONTROL}';\n" + _TAIL_COUNTS),
}
_FOLLOW_TABS = ("fill", "vh", "vh_margins", "vh_capped", "canvas", "reset", "grows", "paragraph", "anchored", *_FOLLOWERS,
                *_FRACTIONS)
# Followers with nothing outside them; the two margined 100vh roots carry the author's 600 px ceiling.
FOLLOWS_VIEWPORT = ("fill", "vh", "canvas")
FOLLOW_CEILING = 600
# The four documents above stop growing just past the loading height (the tall one just past its 900 px); the three
# whose extra grows with the frame stop there too, two echoes past it (at most about 1.15² × 320 px).
SMALL_FOLLOWERS = ("flex_bottom", "abs_bottom", "root_bottom")
GROWING_FOLLOWERS = ("prop_root", "prop_body", "prop_mixed")


def _write_followers(data_dir: pathlib.Path) -> str:
    from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_review_state

    name = "board_follow_probe"
    skill_dir = data_dir / "skills" / "external" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(textwrap.dedent(f"""\
        ---
        name: {name}
        description: Self-sizing modules that follow their viewport, and documents whose margins matter.
        version: 0.1.0
        type: extension
        entry: plugin.py
        permissions: ["widget"]
        ---
        # Follow probe
        """), encoding="utf-8")
    (skill_dir / "plugin.py").write_text(textwrap.dedent(f"""\
        def register(api):
            for tab in {_FOLLOW_TABS!r}:
                ceiling = {{"max_height": {FOLLOW_CEILING}}} if tab.startswith("vh_") else {{}}
                api.register_ui_tab(tab, tab, render={{"kind": "module", "entry": f"{{tab}}.js", "start": "auto", **ceiling}})
        """), encoding="utf-8")
    for tab, css in _FOLLOW_CSS.items():
        (skill_dir / f"{tab}.js").write_text(_FOLLOW_STYLE.format(css=css, text=f"{tab}: a short line."), encoding="utf-8")
    (skill_dir / "canvas.js").write_text(_CANVAS, encoding="utf-8")
    (skill_dir / "grows.js").write_text(_GROWS, encoding="utf-8")
    (skill_dir / "paragraph.js").write_text(_PARAGRAPH, encoding="utf-8")
    (skill_dir / "anchored.js").write_text(_ANCHORED, encoding="utf-8")
    for tab, source in {**_FOLLOWERS, **_FRACTIONS}.items():
        (skill_dir / f"{tab}.js").write_text(source, encoding="utf-8")
    save_review_state(data_dir, name, SkillReviewState(status="pass", content_hash=compute_content_hash(skill_dir, manifest_entry="plugin.py")))
    return name


WHOLE = """() => {
    const tail = document.getElementById('tail');
    return { inner: window.innerHeight, scroll: document.documentElement.scrollHeight,
        hidden: document.documentElement.style.getPropertyValue('overflow-y') === 'hidden',
        tail: tail ? tail.getBoundingClientRect().bottom : null };
}"""


def _whole(facts: dict) -> bool:
    """The document fits its frame: its bottom control inside, nothing hidden below the frame."""
    return (facts["tail"] is None or facts["tail"] <= facts["inner"] + 0.5) and not (
        facts["hidden"] and facts["scroll"] > facts["inner"] + 1)


def _settle_frames(page, probe: str, tabs, rounds: int = 3, wait_ms: int = 500) -> dict:
    """The tabs' frame heights once none of them moved for `rounds` looks `wait_ms` apart (or as they are after 12).
    An engine pauses animation frames in a frame out of view, so each card is first brought into view for a moment."""
    def heights() -> dict:
        return page.evaluate("""([skill, tabs]) => Object.fromEntries(tabs.map((tab) => [tab,
            Math.round(document.querySelector(`[data-widget-key="${skill}:${tab}"] iframe`)?.getBoundingClientRect().height ?? -1)]))""",
                             [probe, list(tabs)])
    for tab in tabs:
        page.locator(f'[data-widget-key="{probe}:{tab}"]').scroll_into_view_if_needed()
        page.wait_for_timeout(250)
    last, same = None, 0
    for _ in range(12):
        current = heights()
        same = same + 1 if current == last else 0
        last = current
        if same >= rounds:
            break
        page.wait_for_timeout(wait_ms)
    return last


def _reachable(page, frame_selector: str, inner_frame) -> dict:
    """A real wheel over the frame, then a real click on the control at the bottom of its document."""
    frame = page.locator(frame_selector)
    frame.scroll_into_view_if_needed()
    box = frame.bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + min(box["height"] / 2, 300))
    page.mouse.wheel(0, 5000)
    page.wait_for_timeout(500)
    tail = page.frame_locator(frame_selector).locator("#tail")
    facts = inner_frame.evaluate("""() => { const tail = document.getElementById('tail').getBoundingClientRect();
        return { y: window.scrollY, inner: window.innerHeight, tail: tail.bottom, top: tail.top,
                 hidden: document.documentElement.style.getPropertyValue('overflow-y') === 'hidden' }; }""")
    tail.click(timeout=5_000)
    page.wait_for_timeout(150)
    facts["clicks"] = inner_frame.evaluate("() => window.__tailClicks || 0")
    return facts


@pytest.mark.parametrize("engine", ("chromium", "webkit"))
def test_documents_that_follow_their_frame_stop_where_they_scroll_and_whole_documents_keep_their_margins(direct_server_with_data, engine):
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    probe = _write_followers(data_dir)
    shots = evidence_dir(data_dir)
    report = {}

    def frame_heights() -> dict:
        return page.evaluate("""(skill) => Object.fromEntries([...document.querySelectorAll(`[data-widget-key^="${skill}:"]`)]
            .map((node) => [node.dataset.widgetKey.slice(skill.length + 1),
                Math.round(node.querySelector('iframe')?.getBoundingClientRect().height ?? -1)]))""", probe)

    def inner(tab: str):
        return page.locator(f'[data-widget-key="{probe}:{tab}"] iframe').element_handle().content_frame()

    with sync_playwright() as pw:
        browser, context = launch(pw, engine, viewport={"width": 1440, "height": 1000})
        page = context.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            assert page.evaluate("""async (skill) => (await fetch(`/api/skills/${encodeURIComponent(skill)}/toggle`, {
                method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({enabled: true}),
            })).status""", probe) == 200
            # The owner's 240 px on one margined 100vh root and on the paragraph.
            page.evaluate("""async (skill) => fetch('/api/ui/preferences', {method: 'POST',
                headers: {'Content-Type': 'application/json'}, body: JSON.stringify({widget_size: {
                    [skill + ':vh_capped']: {w: 0, h: 240}, [skill + ':paragraph']: {w: 0, h: 240}}})})""", probe)
            page.click('[data-nav-page="widgets"]')
            for tab in _FOLLOW_TABS:
                page.wait_for_function("(selector) => document.querySelector(`${selector} iframe`)?.dataset.widgetContentHeight",
                                       arg=f'[data-widget-key="{probe}:{tab}"]', timeout=30_000)
            report["settled"] = _settle_frames(page, probe, _FOLLOW_TABS)
            page.wait_for_timeout(1500)
            report["later"] = frame_heights()
            page.screenshot(path=str(shots / f"{engine}-follow-probe.png"))
            # A follower with nothing outside it has no height of its own: its first report comes once its entry has
            # run, at the 320 px loading height, and it rests there (never shrunk by the host's empty page, never
            # climbing). The body made tall by its own min-height is as tall as what it holds.
            for tab in FOLLOWS_VIEWPORT:
                assert report["settled"][tab] == report["later"][tab] == 320, (tab, report)
            assert report["later"]["reset"] < 120, report
            # The control at the bottom of an authored 500 px body is where that document ends, as it is drawn: the
            # frame is 500 px and the control inside it (the body's own height is not content; what it places is).
            report["anchored"] = {"frame": report["later"]["anchored"], "child": inner("anchored").evaluate(WHOLE),
                                  "style": inner("anchored").evaluate("() => document.body.getAttribute('style')")}
            assert report["anchored"]["frame"] == 500 and _whole(report["anchored"]["child"]), report["anchored"]

            # A follower with space outside it stops growing a step or two past where it began and scrolls there: no
            # drift, no growth to a ceiling, its document released, nothing more reported.
            def stopped(tab: str, key: str) -> dict:
                selector = f'[data-widget-key="{probe}:{tab}"]'
                content = page.evaluate("(s) => document.querySelector(`${s} iframe`).dataset.widgetContentHeight", selector)
                frames = frame_heights()[tab]
                page.wait_for_timeout(800)
                child_facts = inner(tab).evaluate(WHOLE)
                report[key] = {"frames": frames, "frames_later": frame_heights()[tab], "child": child_facts, "content": content,
                               "content_later": page.evaluate("(s) => document.querySelector(`${s} iframe`).dataset.widgetContentHeight", selector)}
                assert report[key]["frames"] == report[key]["frames_later"], ("the frame stands still", report[key])
                assert not child_facts["hidden"], ("its document scrolls", report[key])
                assert report[key]["content"] == report[key]["content_later"], ("nothing more is reported", report[key])
                return report[key]

            margined = stopped("vh_margins", "free_at_stop")
            assert 320 < margined["frames"] <= 420, margined
            for tab in (*SMALL_FOLLOWERS, *GROWING_FOLLOWERS):
                facts = stopped(tab, tab)
                assert report["settled"][tab] == report["later"][tab] == facts["frames"], (tab, facts)
                assert 320 < facts["frames"] <= (420 if tab in SMALL_FOLLOWERS else 480), (tab, facts)
            tall = stopped("mixed900", "mixed900")
            assert 900 <= tall["frames"] <= 1100, tall
            # Each one's bottom control is reached by a real wheel and takes a real click there.
            for tab in (*SMALL_FOLLOWERS, *GROWING_FOLLOWERS, "mixed900"):
                facts = _reachable(page, f'[data-widget-key="{probe}:{tab}"] iframe', inner(tab))
                report[f"{tab}_reach"] = facts
                assert facts["y"] > 0 and facts["tail"] <= facts["inner"] + 0.5, (tab, facts)
                assert facts["clicks"] == 1, (tab, facts)
            # A document sized as a fraction of its frame is followed down until its content stops it, and stays there,
            # whole: the list scrolls in its own box and the button under it is in view and works.
            for tab in _FRACTIONS:
                assert 0 < report["settled"][tab] == report["later"][tab] < 320, (tab, report)
                report[tab] = inner(tab).evaluate(WHOLE)
                assert _whole(report[tab]), (tab, report[tab])
            assert inner("frac_list").evaluate("() => { const box = document.getElementById('box');"
                                                " return box.scrollHeight > box.clientHeight + 100; }")
            page.frame_locator(f'[data-widget-key="{probe}:frac_list"] iframe').locator("#tail").click(timeout=5_000)
            assert inner("frac_list").evaluate("() => window.__tailClicks || 0") == 1
            page.screenshot(path=str(shots / f"{engine}-follow-probe-stopped.png"), full_page=True)

            # The owner's 240 px is the frame's height, stopped or not; Show all lets it grow only as far as its own stop
            # (a follower has no more to show), Back returns to 240, each with no creep.
            def owned_shows(frames) -> None:
                assert frame_heights()["vh_capped"] == frames, report
            owned_shows(240)
            stopped("vh_capped", "capped")
            capped = f'[data-widget-key="{probe}:vh_capped"]'
            page.locator(f"{capped} [data-widget-expand]").click()
            page.wait_for_timeout(1500)
            shown = stopped("vh_capped", "capped_show_all")
            assert 240 < shown["frames"] <= 360, shown
            page.locator(f"{capped} [data-widget-expand]").click()
            page.wait_for_timeout(1000)
            owned_shows(240)
            stopped("vh_capped", "capped_back")
            page.locator(f"{capped} [data-widget-expand]").click()
            page.wait_for_timeout(1500)
            again = stopped("vh_capped", "capped_show_all_again")
            # Each Show all starts over from the same report and stops a step or two later. Which of the frame's passing
            # heights the document happens to be laid out at can add a step (the reporter before this change does the
            # same); it does not keep growing.
            assert 240 < again["frames"] <= 360 and abs(again["frames"] - shown["frames"]) <= 40, (again, shown)
            page.locator(f"{capped} [data-widget-expand]").click()
            page.wait_for_timeout(600)

            # Content grown to exactly the frame's height: the margins below it still count. The card is brought into
            # view first: an engine may pause drawing (and so observers and animation frames) in a frame out of view.
            grows = page.locator(f'[data-widget-key="{probe}:grows"]')
            grows.scroll_into_view_if_needed()
            report["grows_box"] = grows.bounding_box()
            inner("grows").evaluate("() => window.growToFrame()")
            page.wait_for_timeout(800)
            report["grows"] = inner("grows").evaluate(WHOLE)
            assert _whole(report["grows"]), report["grows"]

            # The owner's 240 px releases the paragraph; Show all lets its collapsed margins back in, whole.
            selector = f'[data-widget-key="{probe}:paragraph"]'
            page.locator(f"{selector} [data-widget-expand]").click()
            page.wait_for_timeout(800)
            report["show_all"] = inner("paragraph").evaluate(WHOLE)
            assert _whole(report["show_all"]), report["show_all"]
            # Paired: shorter and taller again while shown, and Back to the owner's height.
            inner("paragraph").evaluate("() => { document.getElementById('para').style.height = '150px'; }")
            page.wait_for_timeout(600)
            report["shrunk"] = inner("paragraph").evaluate(WHOLE)
            assert _whole(report["shrunk"]), report["shrunk"]
            inner("paragraph").evaluate("() => { document.getElementById('para').style.height = '210px'; }")
            page.wait_for_timeout(600)
            report["grown"] = inner("paragraph").evaluate(WHOLE)
            assert _whole(report["grown"]), report["grown"]
            page.locator(f"{selector} [data-widget-expand]").click()
            page.wait_for_timeout(600)
            assert abs(page.locator(f"{selector} iframe").bounding_box()["height"] - 240) <= 1.5
            page.screenshot(path=str(shots / f"{engine}-follow-probe-paragraph.png"))

            # Natural height on a follower the owner sized: the fit starts over and stops where it stopped before.
            choose_key = f'[data-widget-key="{probe}:abs_bottom"]'
            before = frame_heights()["abs_bottom"]
            page.locator(f"{choose_key} [data-widget-menu-trigger]").click()
            page.locator('body > .skills-card-menu-dialog[open] [data-widget-height="360"]').click()
            page.wait_for_function("([s, h]) => Math.abs(document.querySelector(`${s} iframe`).getBoundingClientRect().height - h) <= 1",
                                   arg=[choose_key, 360], timeout=8_000)
            page.locator(f"{choose_key} [data-widget-menu-trigger]").click()
            page.locator('body > .skills-card-menu-dialog[open] [data-widget-height="0"]').click()
            page.wait_for_timeout(1500)
            report["natural_again"] = frame_heights()["abs_bottom"]
            assert report["natural_again"] == before, report
            assert not inner("abs_bottom").evaluate(WHOLE)["hidden"], report
        finally:
            (shots / f"{engine}-follow-probe.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            context.close()
            browser.close()


# --- the owner's height is exact; ordinary growth is followed to its end ----------------------------------------------

_GROWTH_ROWS = ("const add = () => { const d = document.createElement('div'); d.style.height = '20px'; d.textContent = 'row';"
                " document.getElementById('root').appendChild(d); };\n")
_GROWTH = {
    # Rows arriving every animation frame and every 120 ms, a height that eases over half a second, content that shrinks.
    "stream_frames": _GROWTH_ROWS + "let n = 0; const tick = () => { add(); if (++n < 40) requestAnimationFrame(tick); }; tick();\n",
    "stream_interval": _GROWTH_ROWS + "let n = 0; add(); const t = setInterval(() => { add(); if (++n >= 12) clearInterval(t); }, 120);\n",
    "transition": ("const r = document.getElementById('root'); r.style.cssText = 'height:100px;transition:height .5s ease;background:#234';\n"
                   "setTimeout(() => { r.style.height = '500px'; }, 300);\n"),
    "shrinks": _GROWTH_ROWS + "for (let i = 0; i < 30; i += 1) add();\nwindow.shrink = () => { const r = document.getElementById('root'); while (r.children.length > 10) r.lastChild.remove(); };\n",
    # A self-sizing card's image that arrives after its entry: nothing is drawn until it loads (root 0 px), then its 400 px.
    "late_image": ("document.body.style.margin = '0';\nconst root = document.getElementById('root');\n"
                   "root.innerHTML = '<img id=\"late\" alt=\"\" style=\"display:block\">';\n"
                   "const svg = '<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"64\" height=\"400\"><rect width=\"64\" height=\"400\" fill=\"teal\"/></svg>';\n"
                   "setTimeout(() => { document.getElementById('late').src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg); }, 400);\n"),
    # The earlier counterexample: content 240 px grows to 360 px in the same step as Show all lifts an owner height.
    "coincide": ("document.body.style.margin = '0';\nconst r = document.getElementById('root');\n"
                 "r.style.cssText = 'height:240px;position:relative';\n"
                 f"r.innerHTML = '{_PLACED_CONTROL}';\n"
                 "window.__ouroWidgetOnViewport?.((v) => { if (v.ceiling > 1000) { r.style.height = '360px'; } });\n"),
    # A 100 px root and a box placed in it that grows later, at once or by a transition: the root keeps its 100 px.
    "placed_grows": ("document.body.style.margin = '0';\nconst r = document.getElementById('root');\n"
                     "r.style.cssText = 'height:100px;position:relative';\n"
                     f"r.innerHTML = '<div id=\"panel\" style=\"position:absolute;top:0;left:0;right:0;height:100px\">Panel{_PLACED_CONTROL}</div>';\n"
                     "window.grow = () => { document.getElementById('panel').style.height = '240px'; };\n"),
    "placed_eases": ("document.body.style.margin = '0';\nconst r = document.getElementById('root');\n"
                     "r.style.cssText = 'height:100px;position:relative';\n"
                     f"r.innerHTML = '<div id=\"panel\" style=\"position:absolute;top:0;left:0;right:0;height:100px;"
                     f"transition:height .4s ease\">Panel{_PLACED_CONTROL}</div>';\n"
                     "window.grow = () => { document.getElementById('panel').style.height = '240px'; };\n"),
    # A control placed in an authored 500 px body, moved later: no extent of the document changes, only what it holds.
    "placed_moves": ("document.head.insertAdjacentHTML('beforeend', '<style>body{height:500px;position:relative;margin:0}"
                     "#root{height:20px}</style>');\n"
                     "document.getElementById('root').innerHTML = 'Start <button id=\"tail\" type=\"button\" "
                     "style=\"position:absolute;top:80px;height:20px\">Move</button>';\n"
                     "window.place = (top) => { document.getElementById('tail').style.top = `${top}px`; };\n"),
    # A menu that :hover or :focus-within shows below a 30 px root: only the pointer moves, the document stays as it was.
    "hover_menu": ("document.head.insertAdjacentHTML('beforeend', '<style>body{margin:0}#root{height:30px;position:relative}"
                   "#menu{display:none;position:absolute;top:30px;height:200px;width:200px;background:#9cf}"
                   "#root:hover #menu,#root:focus-within #menu{display:block}</style>');\n"
                   "document.getElementById('root').innerHTML = '<button id=\"trigger\" type=\"button\">Menu</button>"
                   "<div id=\"menu\"><button id=\"last\" type=\"button\" style=\"margin-top:170px\">Last action</button></div>';\n"
                   "document.getElementById('last').addEventListener('click', () => { window.__lastClicks = (window.__lastClicks || 0) + 1; });\n"),
    # 180 px of content that grows by 200 px when asked (window.more).
    "late_growth": ("document.body.style.margin = '0';\n"
                    "document.getElementById('root').innerHTML = '<div id=\"base\" style=\"height:180px\">Base</div>';\n"
                    "window.more = () => { const d = document.createElement('div'); d.style.height = '200px';"
                    " d.textContent = 'More'; document.getElementById('root').appendChild(d); };\n"),
}


def _write_growth(data_dir: pathlib.Path) -> str:
    from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_review_state

    name = "board_growth_probe"
    skill_dir = data_dir / "skills" / "external" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(textwrap.dedent(f"""\
        ---
        name: {name}
        description: Self-sizing modules that grow, stream, animate and shrink.
        version: 0.1.0
        type: extension
        entry: plugin.py
        permissions: ["widget"]
        ---
        # Growth probe
        """), encoding="utf-8")
    (skill_dir / "plugin.py").write_text(textwrap.dedent(f"""\
        def register(api):
            for tab in {tuple(_GROWTH)!r}:
                api.register_ui_tab(tab, tab, render={{"kind": "module", "entry": f"{{tab}}.js", "start": "auto"}})
        """), encoding="utf-8")
    for tab, source in _GROWTH.items():
        (skill_dir / f"{tab}.js").write_text(source, encoding="utf-8")
    save_review_state(data_dir, name, SkillReviewState(status="pass", content_hash=compute_content_hash(skill_dir, manifest_entry="plugin.py")))
    return name


# An animation that moves nothing beside a follower: a badge in a corner fading in over 8 s; its twin has no badge.
_BADGE = ("document.head.insertAdjacentHTML('beforeend', '<style>@keyframes badgeIn{from{opacity:.4}to{opacity:1}}"
          "#badge{position:fixed;top:8px;right:8px;width:16px;height:16px;background:teal;"
          "animation:badgeIn 8s linear 1}</style>');\n"
          "document.body.insertAdjacentHTML('beforeend', '<div id=\"badge\"></div>');\n")


def _write_modules(data_dir: pathlib.Path, name: str, sources: dict) -> str:
    from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_review_state

    skill_dir = data_dir / "skills" / "external" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(textwrap.dedent(f"""\
        ---
        name: {name}
        description: Self-sizing modules for one board probe.
        version: 0.1.0
        type: extension
        entry: plugin.py
        permissions: ["widget"]
        ---
        # Board probe
        """), encoding="utf-8")
    (skill_dir / "plugin.py").write_text(textwrap.dedent(f"""\
        def register(api):
            for tab in {tuple(sources)!r}:
                api.register_ui_tab(tab, tab, render={{"kind": "module", "entry": f"{{tab}}.js", "start": "auto"}})
        """), encoding="utf-8")
    for tab, source in sources.items():
        (skill_dir / f"{tab}.js").write_text(source, encoding="utf-8")
    save_review_state(data_dir, name, SkillReviewState(status="pass", content_hash=compute_content_hash(skill_dir, manifest_entry="plugin.py")))
    return name


@pytest.mark.parametrize("engine", ("chromium", "webkit"))
def test_an_animation_that_moves_nothing_does_not_keep_a_follower_growing(direct_server_with_data, engine):
    """A 110vh follower beside a badge's fade stops where its twin without the badge stops and stands still while the
    fade runs; looked at once more when it has ended, it takes one more height at most and stays there."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    follower = _FOLLOWERS["prop_root"]
    probe = _write_modules(data_dir, "board_badge_probe", {"badge": follower + _BADGE, "plain": follower})
    shots = evidence_dir(data_dir)
    report = {}

    def heights() -> dict:
        return page.evaluate("""(skill) => Object.fromEntries(['badge', 'plain'].map((tab) => [tab, Math.round(
            document.querySelector(`[data-widget-key="${skill}:${tab}"] iframe`)?.getBoundingClientRect().height ?? -1)]))""",
                             probe)

    def inner(tab: str):
        return page.locator(f'[data-widget-key="{probe}:{tab}"] iframe').element_handle().content_frame()

    with sync_playwright() as pw:
        browser, context = launch(pw, engine, viewport={"width": 1440, "height": 1000})
        page = context.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            assert page.evaluate("""async (skill) => (await fetch(`/api/skills/${encodeURIComponent(skill)}/toggle`, {
                method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({enabled: true}),
            })).status""", probe) == 200
            page.click('[data-nav-page="widgets"]')
            for tab in ("badge", "plain"):
                page.wait_for_function("(s) => document.querySelector(`${s} iframe`)?.dataset.widgetContentHeight",
                                       arg=f'[data-widget-key="{probe}:{tab}"]', timeout=30_000)
            # While the fade runs: stopped where the twin stops, and still.
            page.wait_for_timeout(1200)
            during = []
            for _ in range(5):
                during.append(heights())
                page.wait_for_timeout(250)
            report["during"] = during
            report["running"] = inner("badge").evaluate("() => document.getAnimations().length")
            assert report["running"] == 1, report
            assert len({item["badge"] for item in during}) == 1, report
            assert abs(during[-1]["badge"] - during[-1]["plain"]) <= 2, report
            # Once it has ended: at most one more height, then still, its bottom control reached by a wheel and a click.
            inner("badge").wait_for_function("() => document.getAnimations().length === 0", timeout=15_000)
            report["after"] = _settle_frames(page, probe, ["badge", "plain"], rounds=3)
            page.wait_for_timeout(1500)
            report["later"] = heights()
            assert report["after"] == report["later"], report
            assert during[-1]["badge"] <= report["later"]["badge"] <= round(during[-1]["badge"] * 1.15) + 2, report
            assert report["later"]["plain"] == during[-1]["plain"], report
            report["reach"] = _reachable(page, f'[data-widget-key="{probe}:badge"] iframe', inner("badge"))
            reach = report["reach"]
            assert reach["y"] > 0 and reach["tail"] <= reach["inner"] + 0.5 and reach["clicks"] == 1, report
            assert errors == [], errors
        finally:
            (shots / f"{engine}-badge-follower.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            context.close()
            browser.close()


@pytest.mark.parametrize("engine", ("chromium", "webkit"))
def test_the_owners_height_is_exact_and_ordinary_growth_is_followed_to_its_end(direct_server_with_data, engine):
    """Streaming, a height transition, a shrink and a late image reach their real height with no stop and no drift; the
    owner's height is exact in both directions on any self-sizing card, Natural height and Reset measure anew."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    _write_skill(data_dir)
    probe = _write_growth(data_dir)
    shots = evidence_dir(data_dir)
    report = {}

    def growth(tab: str) -> str:
        return f'[data-widget-key="{probe}:{tab}"]'

    def inner(tab: str):
        return page.locator(f"{growth(tab)} iframe").element_handle().content_frame()

    def frame_of(selector: str) -> float:
        return page.locator(f"{selector} iframe").bounding_box()["height"]

    with sync_playwright() as pw:
        browser, context = launch(pw, engine, viewport={"width": 1440, "height": 1000})
        page = context.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            enable(page)
            assert page.evaluate("""async (skill) => (await fetch(`/api/skills/${encodeURIComponent(skill)}/toggle`, {
                method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({enabled: true}),
            })).status""", probe) == 200
            page.evaluate("""async (key) => fetch('/api/ui/preferences', {method: 'POST',
                headers: {'Content-Type': 'application/json'}, body: JSON.stringify({widget_size: {[key]: {w: 0, h: 120}}})})""",
                          f"{probe}:coincide")
            open_widgets(page)
            for tab in _GROWTH:
                page.locator(f"{growth(tab)} iframe").wait_for(state="attached", timeout=30_000)
            report["settled"] = _settle_frames(page, probe, _GROWTH, rounds=4)
            settled = report["settled"]
            # Rows and the transition end at their real height, whole and hidden; nothing was stopped by the growth.
            for tab, want in (("stream_frames", 800 + 16), ("stream_interval", 260 + 16), ("transition", 516)):
                whole = inner(tab).evaluate(WHOLE)
                report[tab] = whole
                assert abs(settled[tab] - want) <= 1 and _whole(whole) and whole["hidden"], (tab, settled, whole)
            assert abs(settled["late_image"] - 400) <= 1, (settled, "the image is measured when it arrives")
            # A normal shrink follows immediately, and growth after it again.
            assert abs(settled["shrinks"] - (600 + 16)) <= 1, settled
            inner("shrinks").evaluate("() => window.shrink()")
            page.wait_for_timeout(600)
            report["shrunk"] = _settle_frames(page, probe, ["shrinks"], rounds=2)
            assert abs(report["shrunk"]["shrinks"] - (200 + 16)) <= 1, report
            assert _whole(inner("shrinks").evaluate(WHOLE))
            # A box placed in the root that grows later, at once or eased, takes the frame with it to its 240 px, whole:
            # the root keeps its 100 px, so only the change in the document says so.
            for tab in ("placed_grows", "placed_eases"):
                assert abs(settled[tab] - 100) <= 1, (tab, settled)
                inner(tab).evaluate("() => window.grow()")
            page.wait_for_timeout(1000)
            report["placed"] = _settle_frames(page, probe, ["placed_grows", "placed_eases"], rounds=2)
            for tab in ("placed_grows", "placed_eases"):
                report[f"{tab}_whole"] = inner(tab).evaluate(WHOLE)
                assert abs(report["placed"][tab] - 240) <= 1 and _whole(report[f"{tab}_whole"]), (tab, report)
            # A control placed in an authored body moves down, then up: the body, the root and the document keep their
            # heights, and the frame follows the control's bottom both ways.
            assert abs(settled["placed_moves"] - 100) <= 1, settled
            for top, want in ((180, 200), (30, 50)):
                inner("placed_moves").evaluate("(top) => window.place(top)", top)
                moved = _settle_frames(page, probe, ["placed_moves"], rounds=2)["placed_moves"]
                tail = inner("placed_moves").evaluate(
                    "() => [document.getElementById('tail').getBoundingClientRect().bottom, innerHeight]")
                report[f"placed_moves_{top}"] = {"frame": moved, "tail": tail}
                assert abs(moved - want) <= 1 and tail[0] <= tail[1] + 0.5, report
            # A menu that :hover shows below the root takes the frame with it while the pointer is there, and its last
            # action takes a real click; the pointer gone, the frame is the root's again.
            assert abs(settled["hover_menu"] - 30) <= 1, settled
            menu_frame = page.locator(f"{growth('hover_menu')} iframe")
            # In the middle of the view and once the board stands still, so the menu opens under a pointer that stays on it.
            page.locator(growth("hover_menu")).evaluate("(card) => card.scrollIntoView({block: 'center'})")
            page.mouse.move(5, 5)
            page.wait_for_timeout(800)
            box = menu_frame.bounding_box()
            page.mouse.move(box["x"] + 20, box["y"] + 15)
            page.wait_for_function("(s) => Math.abs(document.querySelector(s).getBoundingClientRect().height - 230) <= 1",
                                   arg=f"{growth('hover_menu')} iframe", timeout=8_000)
            last = inner("hover_menu").evaluate(
                "() => [document.getElementById('last').getBoundingClientRect().bottom, innerHeight]")
            report["hover_menu"] = {"shown": frame_of(growth("hover_menu")), "last": last}
            assert last[0] <= last[1] + 0.5, report
            page.frame_locator(f"{growth('hover_menu')} iframe").locator("#last").click(timeout=5_000)
            assert inner("hover_menu").evaluate("() => window.__lastClicks || 0") == 1, report
            # The pointer gone and the focus out of it (a click may have focused the action), the menu closes.
            page.mouse.move(5, 5)
            inner("hover_menu").evaluate("() => document.activeElement?.blur()")
            page.wait_for_timeout(600)
            report["hover_menu"]["left"] = frame_of(growth("hover_menu"))
            assert abs(report["hover_menu"]["left"] - 30) <= 1, report

            # The old counterexample: 240 px under the owner's 120, grown to 360 px in the same step as Show all.
            assert abs(frame_of(growth("coincide")) - 120) <= 1, report
            page.locator(f"{growth('coincide')} [data-widget-expand]").click()
            page.wait_for_timeout(1200)
            report["coincide"] = {"frame": frame_of(growth("coincide")), "child": inner("coincide").evaluate(WHOLE)}
            assert abs(report["coincide"]["frame"] - 360) <= 1 and _whole(report["coincide"]["child"]), report["coincide"]
            page.locator(f"{growth('coincide')} [data-widget-expand]").click()
            page.wait_for_timeout(600)
            assert abs(frame_of(growth("coincide")) - 120) <= 1, report

            # The owner's height is the frame's height, shorter or taller than its content, and persists alone.
            long_natural = facts(page)["cards"]["long"]["frame"]
            short_natural = facts(page)["cards"]["short"]["frame"]
            mark_frames(page)
            field = page.frame_locator(f"{card('long')} iframe").locator("#field")
            field.fill("kept text")
            # A short card grows past its content; the room below it is the owner's and it says nothing about Show all.
            choose(page, "short", "data-widget-height", "560")
            wait_saved(page, "short", {"w": 0, "h": 560})
            wait_frame_height(page, "short", 560)
            short_owned = facts(page)["cards"]["short"]
            assert short_owned["more"] == "" and abs(short_owned["frame"] - 560) <= 1, short_owned
            assert child(page, "short")["scrollHeight"] <= child(page, "short")["innerHeight"] + 1
            # A long card shrinks below it, and a drag grows it past its natural height too.
            drag(page, f"{card('long')} [data-widget-height-handle]", 0, -(long_natural - 480))
            stored = saved(page)["widget_size"][f"{SKILL}:long"]
            assert 440 <= stored["h"] <= 520 and stored["w"] == 0, stored
            wait_frame_height(page, "long", stored["h"])
            assert facts(page)["cards"]["long"]["more"].startswith("Show all")
            box = page.locator(f"{card('long')} iframe").bounding_box()
            drag(page, f"{card('long')} [data-widget-height-handle]", 0, stored["h"] + 300 - box["height"])
            page.wait_for_timeout(300)
            enlarged = saved(page)["widget_size"][f"{SKILL}:long"]
            assert enlarged["h"] > stored["h"] + 200, enlarged
            wait_frame_height(page, "long", enlarged["h"])
            # A drag ending within the catch of its natural height is natural again.
            box = page.locator(f"{card('long')} iframe").bounding_box()
            drag(page, f"{card('long')} [data-widget-height-handle]", 0, long_natural - box["height"])
            page.wait_for_timeout(300)
            assert saved(page)["widget_size"].get(f"{SKILL}:long", {}).get("h", 0) == 0, saved(page)["widget_size"]
            page.wait_for_function("([s, h]) => Math.abs(document.querySelector(`${s} iframe`).getBoundingClientRect().height - h) <= 4",
                                   arg=[card("long"), long_natural], timeout=8_000)

            # A height and a width are separate choices: each keeps the other, Reset clears both and measures anew.
            choose(page, "short", "data-widget-size", "2")
            wait_saved(page, "short", {"w": 2, "h": 560})
            wait_frame_height(page, "short", 560)
            choose(page, "short", "data-widget-height", "0")
            wait_saved(page, "short", {"w": 2, "h": 0})
            wait_frame_height(page, "short", short_natural, slack=2)
            choose(page, "short", "data-widget-height", "360")
            wait_saved(page, "short", {"w": 2, "h": 360})
            choose(page, "short", "data-widget-size", "reset")
            wait_saved(page, "short", None)
            wait_frame_height(page, "short", short_natural, slack=2)
            assert frames_kept(page, ("short", "long")), "a frame was replaced"
            assert child(page, "long")["field"] == "kept text"
            page.screenshot(path=str(shots / f"{engine}-exact-heights.png"), full_page=True)
            assert errors == [], errors
        finally:
            (shots / f"{engine}-exact-heights.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            context.close()
            browser.close()


# --- a report on its way when the fit starts over -----------------------------------------------------------------------
# The page's side of every frame channel, held and delivered on cue: what a busy page does with a size report its frame
# sent a moment before the owner pressed Natural height.
PORT_HOLD = """(() => {
    if (window.top !== window) return;
    const hold = window.__portHold = { holding: false, queue: [] };
    const own = Object.getOwnPropertyDescriptor(MessagePort.prototype, 'onmessage');
    Object.defineProperty(MessagePort.prototype, 'onmessage', {
        configurable: true,
        get() { return own.get.call(this); },
        set(handler) {
            if (typeof handler !== 'function') { own.set.call(this, handler); return; }
            own.set.call(this, (event) => {
                if (hold.holding && event.data?.type === 'ouro-widget-resize') { hold.queue.push(() => handler(event)); return; }
                return handler(event);
            });
        },
    });
})();"""


@pytest.mark.parametrize("engine", ("chromium", "webkit"))
def test_a_report_on_its_way_when_the_fit_starts_over_is_not_taken_for_it(direct_server_with_data, engine):
    """Natural height starts a card's fit over at the loading height. A real report its frame sent from the owner's
    height a moment before, delivered only after the press, is not that fit: the frame measures anew at the loading
    height, settles on its content, and follows what grows afterwards, whole."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    probe = _write_growth(data_dir)
    key = f"{probe}:late_growth"
    selector = f'[data-widget-key="{key}"]'
    shots = evidence_dir(data_dir)
    report = {}

    def frame_is(height: float) -> None:
        page.wait_for_function("([s, h]) => Math.abs(document.querySelector(`${s} iframe`).getBoundingClientRect().height - h) <= 1.5",
                               arg=[selector, height], timeout=8_000)

    with sync_playwright() as pw:
        browser, context = launch(pw, engine, viewport={"width": 1440, "height": 1000})
        page = context.new_page()
        page.add_init_script(PORT_HOLD)
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            assert page.evaluate("""async (skill) => (await fetch(`/api/skills/${encodeURIComponent(skill)}/toggle`, {
                method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({enabled: true}),
            })).status""", probe) == 200
            page.evaluate("""async (key) => fetch('/api/ui/preferences', {method: 'POST',
                headers: {'Content-Type': 'application/json'}, body: JSON.stringify({widget_size: {[key]: {w: 0, h: 480}}})})""", key)
            page.click('[data-nav-page="widgets"]')
            page.wait_for_function("(s) => document.querySelector(`${s} iframe`)?.dataset.widgetContentHeight", arg=selector,
                                   timeout=30_000)
            page.locator(selector).scroll_into_view_if_needed()
            frame_is(480)
            inner = page.locator(f"{selector} iframe").element_handle().content_frame()
            # The page holds what its frames post; the frame reports a 1 px change measured at the owner's 480 px.
            page.evaluate("() => { window.__portHold.holding = true; }")
            inner.evaluate("() => { document.getElementById('base').style.height = '181px'; }")
            page.wait_for_function("() => window.__portHold.queue.length > 0", timeout=5_000)
            page.locator(f"{selector} [data-widget-menu-trigger]").click()
            # Natural height, and then, in the same turn, the report that was on its way.
            report["held"] = page.evaluate("""() => {
                document.querySelector('body > .skills-card-menu-dialog[open] [data-widget-height="0"]').click();
                const hold = window.__portHold;
                hold.holding = false;
                const queued = hold.queue.splice(0);
                queued.forEach((deliver) => deliver());
                return queued.length;
            }""")
            assert report["held"] >= 1, report
            frame_is(181)
            report["natural"] = inner.evaluate(WHOLE)
            assert _whole(report["natural"]), report
            # Content that grows afterwards is followed, whole.
            inner.evaluate("() => window.more()")
            frame_is(381)
            page.wait_for_timeout(300)
            report["grown"] = {"frame": page.locator(f"{selector} iframe").bounding_box()["height"], "child": inner.evaluate(WHOLE)}
            assert _whole(report["grown"]["child"]) and report["grown"]["child"]["inner"] >= 380, report
            assert page.evaluate("async (key) => ((await (await fetch('/api/ui/preferences')).json()).widget_size || {})[key] ?? null",
                                 key) is None
            page.screenshot(path=str(shots / f"{engine}-report-on-its-way.png"))
            assert errors == [], errors
        finally:
            (shots / f"{engine}-report-on-its-way.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            context.close()
            browser.close()


# --- a small drag keeps an exact height where the content has no height of its own -------------------------------------

BODY = """(key) => { const body = document.querySelector(`[data-widget-key="${key}"] [data-widget-mount]`);
    return { client: body.clientHeight, scroll: body.scrollHeight, bounded: body.hasAttribute('data-widget-bounded'),
             frame: body.querySelector('iframe')?.getBoundingClientRect().height ?? null }; }"""


@pytest.mark.parametrize("engine", ("chromium", "webkit"))
def test_a_small_drag_keeps_the_owners_exact_height_where_the_content_has_no_height_of_its_own(direct_server_with_data, engine):
    """A bounded host-drawn card taller than what it holds (a declarative note, a card kept from the chat), a frame whose
    document fills its window and two that ask for their margins beyond any window (a margined 100vh root, a control at
    the bottom of a 100vh body) have no height of their own to catch: a drag of 12 px up or down sets that exact height,
    saved, and a reload keeps it. Paired: a bounded card whose content is taller, and a frame whose report outlived a
    change of its height, still catch their content's own height and go natural there."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    from ouroboros.card_store import pin_card, write_card

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    _write_skill(data_dir)
    probe = _write_followers(data_dir)
    kept = write_card(data_dir, chat_id=1, task_id="", title="Kept note", fallback_text="A short kept note",
                      kind="components", body={"components": [{"type": "callout", "text": "A short kept note."}]})
    pin_card(data_dir, kept["card_id"])
    shots = evidence_dir(data_dir)
    report = {}

    def stored(key: str):
        return page.evaluate("async (key) => ((await (await fetch('/api/ui/preferences')).json()).widget_size || {})[key] ?? null", key)

    def wait_stored(key: str, h: int) -> None:
        wait_until(page, """async ([key, h]) => (((await (await fetch('/api/ui/preferences')).json()).widget_size || {})[key]
            || { h: 0 }).h === h""", arg=[key, h], timeout=8_000)

    def shown(key: str) -> float:
        facts = page.evaluate(BODY, key)
        return facts["frame"] if facts["frame"] is not None else facts["client"]

    def ready() -> None:
        page.locator(f"{card('note')} .widget-callout").wait_for(state="attached", timeout=30_000)
        page.wait_for_function("() => document.querySelector('.widgets-card:has(.widgets-card-source) [data-widget-mount]')"
                               "?.textContent.includes('A short kept note')", timeout=30_000)
        for selector in (*(f'[data-widget-key="{probe}:{tab}"]' for tab in ("fill", "vh_margins", "abs_bottom")), card("long")):
            page.wait_for_function("(s) => document.querySelector(`${s} iframe`)?.dataset.widgetContentHeight",
                                   arg=selector, timeout=30_000)
        page.locator(f"{card('table')} table tbody tr").nth(29).wait_for(state="attached", timeout=15_000)
        page.wait_for_timeout(800)

    with sync_playwright() as pw:
        browser, context = launch(pw, engine, viewport={"width": 1440, "height": 1000})
        page = context.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            enable(page)
            assert page.evaluate("""async (skill) => (await fetch(`/api/skills/${encodeURIComponent(skill)}/toggle`, {
                method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({enabled: true}),
            })).status""", probe) == 200
            page.click('[data-nav-page="widgets"]')
            ready()
            kept_key = page.locator(".widgets-card:has(.widgets-card-source)").get_attribute("data-widget-key")
            keys = {"note": f"{SKILL}:note", "kept": kept_key, "fill": f"{probe}:fill", "margined": f"{probe}:vh_margins",
                    "bottom": f"{probe}:abs_bottom"}
            page.evaluate("""async (sizes) => fetch('/api/ui/preferences', {method: 'POST',
                headers: {'Content-Type': 'application/json'}, body: JSON.stringify({widget_size: sizes})})""",
                          {**{key: {"w": 0, "h": 500} for key in keys.values()}, f"{SKILL}:table": {"w": 0, "h": 300},
                           f"{SKILL}:long": {"w": 0, "h": 900}})
            page.reload(wait_until="domcontentloaded", timeout=30_000)
            page.click('[data-nav-page="widgets"]')
            ready()
            for name, key in keys.items():
                assert abs(shown(key) - 500) <= 1.5, (name, page.evaluate(BODY, key))
            # 12 px up from 500: 488 exactly; reloaded, still 488; 12 px down from there: 504 (the 8 px grid), reloaded.
            for name, key in keys.items():
                selector = f'[data-widget-key="{key}"] [data-widget-height-handle]'
                drag(page, selector, 0, -12, steps=8)
                wait_stored(key, 488)
                report[f"{name}_up"] = {"stored": stored(key), "facts": page.evaluate(BODY, key)}
                assert abs(shown(key) - 488) <= 1.5, report[f"{name}_up"]
            page.reload(wait_until="domcontentloaded", timeout=30_000)
            page.click('[data-nav-page="widgets"]')
            ready()
            for name, key in keys.items():
                report[f"{name}_reloaded"] = page.evaluate(BODY, key)
                assert abs(shown(key) - 488) <= 1.5, (name, report[f"{name}_reloaded"])
                drag(page, f'[data-widget-key="{key}"] [data-widget-height-handle]', 0, 12, steps=8)
                wait_stored(key, 504)
                assert abs(shown(key) - 504) <= 1.5, (name, page.evaluate(BODY, key))
            page.reload(wait_until="domcontentloaded", timeout=30_000)
            page.click('[data-nav-page="widgets"]')
            ready()
            for name, key in keys.items():
                report[f"{name}_reloaded_down"] = page.evaluate(BODY, key)
                assert abs(shown(key) - 504) <= 1.5, (name, report[f"{name}_reloaded_down"])
            page.screenshot(path=str(shots / f"{engine}-small-drags-exact.png"), full_page=True)
            # Paired: the table holds more than its 300 px; dragged to within a few pixels of all its rows, it is
            # natural again: its height goes and it shows every row. (A tall window, so the whole drag stays in it.)
            page.set_viewport_size({"width": 1440, "height": 1800})
            page.locator(card("table")).evaluate("(node) => node.scrollIntoView({ block: 'start' })")
            page.wait_for_timeout(400)
            table = page.evaluate(BODY, f"{SKILL}:table")
            report["table_before"] = table
            assert table["bounded"] and table["scroll"] > table["client"] + 100, table
            drag(page, f"{card('table')} [data-widget-height-handle]", 0, table["scroll"] - table["client"] - 6, steps=16)
            wait_saved(page, "table", None)
            report["table_after"] = page.evaluate(BODY, f"{SKILL}:table")
            assert not report["table_after"]["bounded"], report["table_after"]
            # And a frame: a long document's report stays what it was when its frame takes another height, so a drag that
            # first moves it, then ends a few pixels from all its rows, is natural again. A follower reports anew at every
            # height, so its report never comes to that.
            handle = f"{card('long')} [data-widget-height-handle]"
            page.locator(card("long")).evaluate("(node) => node.scrollIntoView({ block: 'start' })")
            page.wait_for_timeout(300)
            natural = int(page.evaluate("(s) => document.querySelector(`${s} iframe`).dataset.widgetContentHeight", card("long")))
            report["long_natural"] = natural
            drag(page, handle, 0, 40, steps=8)
            wait_saved(page, "long", {"w": 0, "h": 944})
            wait_frame_height(page, "long", 944)
            drag(page, handle, 0, natural - 944 - 6, steps=16)
            wait_saved(page, "long", None)
            wait_frame_height(page, "long", natural, slack=2)
            assert errors == [], errors
        finally:
            (shots / f"{engine}-small-drags-exact.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            context.close()
            browser.close()


# --- returning to the board: no drift, however it left --------------------------------------------------------------

@pytest.mark.parametrize("engine", ("chromium", "webkit"))
def test_returning_to_the_board_leaves_every_height_where_it_was(direct_server_with_data, engine):
    """A follower that stopped, an ordinary card, an owner height and a kept-running card come back exactly as they left,
    however many times: Widgets left and shown again, Stop and Start, a revision change."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    probe = _write_followers(data_dir)
    tabs = ("abs_bottom", "mixed900", "anchored", "vh_capped")
    keep_running = f"{probe}:root_bottom"
    revisions: dict[str, str] = {}

    def widgets_list(route):
        data = route.fetch().json()
        for tab in data.get("ui_tabs", []):
            if tab.get("key") in revisions:
                tab["revision"] = revisions[tab["key"]]
        route.fulfill(status=200, content_type="application/json", body=json.dumps(data))

    def key_of(tab: str) -> str:
        return f"{probe}:{tab}"

    def selector(tab: str) -> str:
        return f'[data-widget-key="{key_of(tab)}"]'

    def heights() -> dict:
        return page.evaluate("""([skill, tabs]) => Object.fromEntries(tabs.map((tab) => [tab,
            Math.round(document.querySelector(`[data-widget-key="${skill}:${tab}"] iframe`)?.getBoundingClientRect().height ?? -1)]))""",
                             [probe, [*tabs, "root_bottom"]])

    def inner(tab: str):
        return page.locator(f"{selector(tab)} iframe").element_handle().content_frame()

    report = {}
    shots = evidence_dir(data_dir)
    with sync_playwright() as pw:
        browser, context = launch(pw, engine, viewport={"width": 1440, "height": 1000})
        page = context.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            assert page.evaluate("""async (skill) => (await fetch(`/api/skills/${encodeURIComponent(skill)}/toggle`, {
                method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({enabled: true}),
            })).status""", probe) == 200
            page.evaluate("""async ([capped, keep]) => fetch('/api/ui/preferences', {method: 'POST',
                headers: {'Content-Type': 'application/json'}, body: JSON.stringify({
                    widget_size: {[capped]: {w: 0, h: 240}}, widget_start_mode: {[keep]: 'retain'}})})""",
                          [key_of("vh_capped"), keep_running])
            page.click('[data-nav-page="widgets"]')
            for tab in tabs:
                page.wait_for_function("(s) => document.querySelector(`${s} iframe`)?.dataset.widgetContentHeight",
                                       arg=selector(tab), timeout=30_000)
            page.wait_for_function("(s) => document.querySelector(`${s} iframe`)?.dataset.widgetContentHeight",
                                   arg=selector("root_bottom"), timeout=30_000)
            first = _settle_frames(page, probe, [*tabs, "root_bottom"], rounds=4)
            report["first"] = first
            for tab in ("abs_bottom", "root_bottom"):
                assert 320 < first[tab] <= 420, (tab, first)
            assert first["vh_capped"] == 240 and first["anchored"] == 500, first
            inner("root_bottom").evaluate("() => { window.__boardMark = 'same-window'; }")

            def come_back() -> dict:
                for tab in tabs:
                    page.locator(f"{selector(tab)} iframe").wait_for(state="attached", timeout=15_000)
                    page.wait_for_function("(s) => document.querySelector(s).querySelector('[data-widget-status]').textContent.trim() === 'Running'",
                                           arg=selector(tab), timeout=15_000)
                page.wait_for_timeout(1200)
                return heights()

            # Leave and return, three times: the same heights, and the kept card keeps its document.
            report["returns"] = []
            for _ in range(3):
                page.click('[data-nav-page="chat"]')
                page.wait_for_timeout(400)
                page.click('[data-nav-page="widgets"]')
                back = come_back()
                report["returns"].append(back)
                assert back == first, (back, first, report)
                assert inner("root_bottom").evaluate("() => window.__boardMark") == "same-window"
            # Stop and Start: the same height for the follower.
            page.locator(f"{selector('abs_bottom')} [data-widget-menu-trigger]").click()
            page.locator('body > .skills-card-menu-dialog[open] [data-widget-stop]').click()
            page.wait_for_function("(s) => !document.querySelector(`${s} iframe`)", arg=selector("abs_bottom"), timeout=8_000)
            page.locator(f"{selector('abs_bottom')} [data-widget-power]").click()
            page.wait_for_function("(s) => !!document.querySelector(`${s} iframe`) && document.querySelector(s).querySelector('[data-widget-status]').textContent.trim() === 'Running'",
                                   arg=selector("abs_bottom"), timeout=15_000)
            page.wait_for_timeout(1200)
            restarted = heights()
            report["restarted"] = restarted
            assert restarted["abs_bottom"] == first["abs_bottom"], report
            # A revision change is a new widget: it fits from the loading height (no remembered stop carries over) and
            # stops at the same height again.
            page.route("**/api/widgets", widgets_list)
            revisions[key_of("abs_bottom")] = "e" * 64
            page.evaluate("() => { window.__ouroWs.emit('open', {previouslyConnected: true}); }")
            page.wait_for_function("(s) => !!document.querySelector(`${s} iframe`) && document.querySelector(s).querySelector('[data-widget-status]').textContent.trim() === 'Running'",
                                   arg=selector("abs_bottom"), timeout=15_000)
            page.wait_for_timeout(1500)
            report["new_revision"] = heights()
            assert report["new_revision"]["abs_bottom"] == first["abs_bottom"], report
            page.screenshot(path=str(shots / f"{engine}-returns.png"), full_page=True)
            assert errors == [], errors
        finally:
            (shots / f"{engine}-returns.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            context.close()
            browser.close()


@pytest.mark.parametrize("engine", ("chromium", "webkit"))
def test_a_stopped_follower_settles_with_a_scrollbar_that_takes_width(direct_server_with_data, engine):
    """At the stop the document's scrollbar appears and takes layout width: the frame and the reports stay put, and the
    bottom control is reachable with the scrollbar in place."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    probe = _write_followers(data_dir)
    selector = f'[data-widget-key="{probe}:abs_bottom"]'
    with sync_playwright() as pw:
        launch_options = {"headless": True}
        if engine == "chromium":
            launch_options["ignore_default_args"] = ["--hide-scrollbars"]
        try:
            browser = getattr(pw, engine).launch(**launch_options)
        except PlaywrightError as exc:
            pytest.skip(f"{engine} is not installed: {exc}")
        context = browser.new_context(viewport={"width": 1440, "height": 1000})
        page = context.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            assert page.evaluate("""async (skill) => (await fetch(`/api/skills/${encodeURIComponent(skill)}/toggle`, {
                method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({enabled: true}),
            })).status""", probe) == 200
            page.click('[data-nav-page="widgets"]')
            page.wait_for_function("(s) => document.querySelector(`${s} iframe`)?.dataset.widgetContentHeight", arg=selector, timeout=30_000)
            inner = page.locator(f"{selector} iframe").element_handle().content_frame()
            inner.evaluate("""() => { const style = document.createElement('style');
                style.textContent = '::-webkit-scrollbar{width:14px}::-webkit-scrollbar-thumb{background:#888}';
                document.head.appendChild(style); }""")
            first = _settle_frames(page, probe, ["abs_bottom"], rounds=4)
            page.wait_for_timeout(1500)
            later = _settle_frames(page, probe, ["abs_bottom"], rounds=2)
            facts = inner.evaluate("""() => ({ inner: window.innerWidth, client: document.documentElement.clientWidth,
                hidden: document.documentElement.style.getPropertyValue('overflow-y') === 'hidden',
                scroll: document.documentElement.scrollHeight, y: window.scrollY, viewport: window.innerHeight })""")
            assert first == later and 320 < later["abs_bottom"] <= 420, (first, later)
            assert not facts["hidden"] and facts["scroll"] > facts["viewport"], facts
            assert facts["inner"] - facts["client"] in (14, 15), f"no layout scrollbar: {facts}"
            reached = _reachable(page, f"{selector} iframe", inner)
            assert reached["y"] > 0 and reached["tail"] <= reached["inner"] + 0.5 and reached["clicks"] == 1, reached
        finally:
            context.close()
            browser.close()


# --- the module's entry: where a board frame starts measuring ---------------------------------------------------------
# A board frame keeps its 320 px loading height until the module's entry has run (right after a classic entry's
# script, after a module entry has evaluated, its top-level await included) and something is drawn: a classic entry
# that has drawn nothing yet keeps the loading height, and so does one that never draws or that threw. What a widget
# draws later is measured when it appears; until its entry has run or its page has loaded, the document scrolls in its
# loading height. Each waiting widget is released by the test (`window.finish`), not by a timer.

_ENTRY_ROWS = ("const rows = (count) => '<ol style=\"margin:0\">' + Array.from({ length: count }, (_, i) =>"
               " '<li style=\"height:20px\">Row ' + (i + 1) + '</li>').join('') + '</ol>';\n")
_ENTRY_FOLLOWER = ("document.body.style.margin = '0';\nconst box = document.createElement('div');\n"
                   "box.style.background = '#234';\ndocument.getElementById('root').replaceChildren(box);\n"
                   "const fit = () => { box.style.height = window.innerHeight + 'px'; };\n"
                   "window.addEventListener('resize', fit);\nfit();\n")
_ENTRY_WAIT = ("await new Promise((resolve) => {\n    window.finish = resolve;\n"
               "    document.getElementById('root').dataset.waiting = '1';\n});\n")
_ENTRY_TAIL = "'<button id=\"tail\" type=\"button\">Bottom control</button>'"
_ENTRY_SOURCES = {
    "empty": "// Draws nothing.\n",
    "css_only": ("const style = document.createElement('style');\n"
                 "style.textContent = '#root{height:180px;background:#2a4}';\ndocument.head.appendChild(style);\n"),
    "short": "document.getElementById('root').innerHTML = '<p>One short line.</p>';\n",
    "throws": "throw new Error('classic entry failed on purpose');\n",
    "late": ("(async () => {\n" + _ENTRY_WAIT + _ENTRY_ROWS
             + "document.getElementById('root').innerHTML = rows(20);\n})();\n"),
    "late_follow": "(async () => {\n" + _ENTRY_WAIT + _ENTRY_FOLLOWER + "})();\n",
    "sync_follow": _ENTRY_FOLLOWER + "(async () => {\n" + _ENTRY_WAIT + "box.textContent = 'Data arrived';\n})();\n",
    "esm_follow": _ENTRY_WAIT + _ENTRY_FOLLOWER,
    "esm_wait": (_ENTRY_ROWS + f"document.getElementById('root').innerHTML = rows(30) + {_ENTRY_TAIL};\n" + _ENTRY_WAIT
                 + "document.getElementById('root').insertAdjacentHTML('beforeend', '<p>Done</p>');\n"),
    "esm_throws": (_ENTRY_ROWS + f"document.getElementById('root').innerHTML = rows(25) + {_ENTRY_TAIL};\n"
                   "throw new Error('module entry failed on purpose');\n"),
}
_ENTRY_MODULES = ("esm_follow", "esm_wait", "esm_throws")


def _write_entries(data_dir: pathlib.Path) -> str:
    from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_review_state

    name = "board_entry_probe"
    skill_dir = data_dir / "skills" / "external" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(textwrap.dedent(f"""\
        ---
        name: {name}
        description: Self-sizing modules whose entries are empty, short, failing, late or waiting.
        version: 0.1.0
        type: extension
        entry: plugin.py
        permissions: ["widget"]
        ---
        # Entry probe
        """), encoding="utf-8")
    (skill_dir / "plugin.py").write_text(textwrap.dedent(f"""\
        def register(api):
            for tab in {tuple(_ENTRY_SOURCES)!r}:
                kind = {{"entry_type": "module"}} if tab in {_ENTRY_MODULES!r} else {{}}
                render = {{"kind": "module", "entry": f"{{tab}}.js", "start": "auto", **kind}}
                api.register_ui_tab(tab, tab, render=render)
        """), encoding="utf-8")
    for tab, source in _ENTRY_SOURCES.items():
        (skill_dir / f"{tab}.js").write_text(source, encoding="utf-8")
    content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
    save_review_state(data_dir, name, SkillReviewState(status="pass", content_hash=content_hash))
    return name


@pytest.mark.parametrize("engine", ("chromium", "webkit"))
def test_a_board_frame_measures_from_its_entry_and_keeps_its_loading_height_until_then(direct_server_with_data, engine):
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    probe = _write_entries(data_dir)
    shots = evidence_dir(data_dir)
    report = {}

    def selector(tab: str) -> str:
        return f'[data-widget-key="{probe}:{tab}"]'

    def heights() -> dict:
        return page.evaluate("""(skill) => Object.fromEntries(
            [...document.querySelectorAll(`[data-widget-key^="${skill}:"]`)].map((node) => [
                node.dataset.widgetKey.slice(skill.length + 1), {
                frame: Math.round(node.querySelector('iframe')?.getBoundingClientRect().height ?? -1),
                reported: node.querySelector('iframe')?.dataset.widgetContentHeight ?? null,
                fault: node.querySelector('[data-widget-status]')?.dataset.widgetFault ?? null}]))""", probe)

    def inner(tab: str):
        return page.locator(f"{selector(tab)} iframe").element_handle().content_frame()

    with sync_playwright() as pw:
        browser, context = launch(pw, engine, viewport={"width": 1600, "height": 2000})
        page = context.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            assert page.evaluate("""async (skill) => (await fetch(`/api/skills/${encodeURIComponent(skill)}/toggle`, {
                method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({enabled: true}),
            })).status""", probe) == 200
            page.click('[data-nav-page="widgets"]')
            for tab in _ENTRY_SOURCES:
                page.locator(f"{selector(tab)} iframe").wait_for(state="attached", timeout=30_000)
            for tab in ("css_only", "short", "sync_follow"):
                page.wait_for_function("(s) => document.querySelector(`${s} iframe`)?.dataset.widgetContentHeight",
                                       arg=selector(tab), timeout=30_000)
            page.wait_for_function("(s) => document.querySelector(s)?.querySelector('[data-widget-status]')?.dataset.widgetFault",
                                   arg=selector("throws"), timeout=30_000)
            for tab in ("late", "late_follow", "sync_follow", "esm_follow", "esm_wait"):
                inner(tab).wait_for_selector('#root[data-waiting="1"]', state="attached", timeout=30_000)
            page.wait_for_function(
                "(s) => document.querySelector(s)?.querySelector('[data-widget-status]')?.dataset.widgetFault",
                arg=selector("esm_throws"), timeout=30_000)
            page.wait_for_timeout(1200)
            report["entered"] = heights()
            page.screenshot(path=str(shots / f"{engine}-entry-probe-entered.png"))
            entered = report["entered"]
            # Measured from the entry, however short, once something is drawn; nothing drawn keeps the loading height (a
            # widget that truly draws nothing stays there), CSS alone sizes it.
            assert entered["empty"]["frame"] == 320 and entered["empty"]["reported"] is None, entered["empty"]
            assert entered["css_only"]["frame"] == 196, entered["css_only"]
            assert entered["short"]["frame"] < 120, entered["short"]
            assert _whole(inner("short").evaluate(WHOLE)), entered["short"]
            # A classic entry that throws has still run, and drew nothing: the loading height, its fault on the card. The
            # fault's text is the engine's: WebKit gives a script of an opaque-origin or about:srcdoc document, as this
            # inline entry is, only "Script error."; the module entry below, fetched with CORS from its lease, keeps its
            # message there.
            assert entered["throws"]["frame"] == 320, entered["throws"]
            classic_fault = "Widget script error: Script error." if engine == "webkit" else "failed on purpose"
            assert classic_fault in (entered["throws"]["fault"] or ""), entered["throws"]
            # A classic entry that awaits has returned at its first await and drew nothing yet: the loading height stands.
            assert entered["late"]["frame"] == entered["late_follow"]["frame"] == 320, entered
            assert entered["late"]["reported"] is None and entered["late_follow"]["reported"] is None, entered
            # Drawn before it returned, a layout that follows its frame rests at the loading height.
            assert entered["sync_follow"]["frame"] == 320, entered["sync_follow"]
            # A module entry still in its top-level await, or one that failed, never marks its end, but its page has
            # loaded: what it drew by then is measured, nothing (the loading height stands) or its rows, whole.
            assert entered["esm_follow"]["frame"] == 320 and entered["esm_follow"]["reported"] is None, entered["esm_follow"]
            for tab in ("esm_wait", "esm_throws"):
                report[f"{tab}_whole"] = inner(tab).evaluate(WHOLE)
                assert entered[tab]["frame"] > 320 and entered[tab]["reported"], (tab, entered[tab])
                assert _whole(report[f"{tab}_whole"]) and report[f"{tab}_whole"]["hidden"], (tab, report)
            assert "failed on purpose" in (entered["esm_throws"]["fault"] or ""), entered["esm_throws"]

            for tab in ("late", "late_follow", "sync_follow", "esm_follow", "esm_wait"):
                inner(tab).evaluate("() => window.finish()")
            page.wait_for_timeout(1500)
            report["finished"] = heights()
            page.wait_for_timeout(1200)
            report["later"] = heights()
            page.screenshot(path=str(shots / f"{engine}-entry-probe-finished.png"))
            finished, later = report["finished"], report["later"]
            # Drawn later, content is measured when it appears: 20 rows of 20 px between 8 px margins.
            assert finished["late"]["frame"] == later["late"]["frame"] == 416, (finished["late"], later["late"])
            assert _whole(inner("late").evaluate(WHOLE))
            # A layout that follows its frame, drawn after a classic entry returned, finds the loading height and rests
            # there like the same layout drawn before returning or after a module's top-level await.
            for tab in ("late_follow", "sync_follow", "esm_follow"):
                assert finished[tab]["frame"] == later[tab]["frame"] == 320, (tab, finished[tab], later[tab])
            assert later["esm_follow"]["reported"] == "320", later["esm_follow"]
            # A module entry whose await ended is measured whole, its scrollbar hidden again.
            report["esm_wait_done"] = inner("esm_wait").evaluate(WHOLE)
            assert later["esm_wait"]["frame"] > 320 and _whole(report["esm_wait_done"]), (later["esm_wait"], report)
            assert report["esm_wait_done"]["hidden"], report["esm_wait_done"]
            assert later["esm_throws"]["frame"] == entered["esm_throws"]["frame"], (later["esm_throws"], entered["esm_throws"])
        finally:
            (shots / f"{engine}-entry-probe.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            context.close()
            browser.close()


# --- a bounded host-drawn card across reflow, and the bottom edge over Show all --------------------------------------

_REFLOW = "A useful status sentence that wraps when its card narrows. " * 12


def _write_reflow(data_dir: pathlib.Path) -> str:
    from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_review_state

    name = "board_reflow_probe"
    skill_dir = data_dir / "skills" / "external" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(textwrap.dedent(f"""\
        ---
        name: {name}
        description: One declarative card whose text wraps with its width.
        version: 0.1.0
        type: extension
        entry: plugin.py
        permissions: ["widget"]
        ---
        # Reflow probe
        """), encoding="utf-8")
    (skill_dir / "plugin.py").write_text(textwrap.dedent(f"""\
        def register(api):
            api.register_ui_tab("note", "Note", render={{"kind": "declarative", "schema_version": 1,
                "components": [{{"type": "callout", "text": {_REFLOW!r}}}]}})
            api.register_ui_tab("other", "Other", render={{"kind": "declarative", "schema_version": 1,
                "components": [{{"type": "callout", "text": "A second card."}}]}})
        """), encoding="utf-8")
    save_review_state(data_dir, name, SkillReviewState(status="pass", content_hash=compute_content_hash(skill_dir, manifest_entry="plugin.py")))
    return name


@pytest.mark.parametrize("engine", ("chromium", "webkit"))
def test_a_bounded_card_refits_on_reflow_and_a_drop_on_its_height_ends_show_all(direct_server_with_data, engine):
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    probe = _write_reflow(data_dir)
    key = f"{probe}:note"
    selector = f'[data-widget-key="{key}"]'
    shots = evidence_dir(data_dir)
    report = {}
    more = """(selector) => { const line = document.querySelector(`${selector} [data-widget-more]`);
        const body = document.querySelector(`${selector} [data-widget-mount]`);
        return { shown: !line.hidden, text: line.textContent.trim(), client: body.clientHeight, scroll: body.scrollHeight }; }"""
    stored = "async (key) => ((await (await fetch('/api/ui/preferences')).json()).widget_size || {})[key] ?? null"

    with sync_playwright() as pw:
        browser, context = launch(pw, engine, viewport={"width": 1920, "height": 1080})
        page = context.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            assert page.evaluate("""async (skill) => (await fetch(`/api/skills/${encodeURIComponent(skill)}/toggle`, {
                method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({enabled: true}),
            })).status""", probe) == 200
            page.evaluate("""async ([key, other]) => fetch('/api/ui/preferences', {method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({widget_size: {[key]: {w: 12, h: 160}, [other]: {w: 0, h: 160}}})})""", [key, f"{probe}:other"])
            page.click('[data-nav-page="widgets"]')
            page.locator(f"{selector} .widget-callout").wait_for(state="attached", timeout=30_000)
            page.wait_for_timeout(800)
            report["wide"] = page.evaluate(more, selector)
            assert report["wide"]["scroll"] <= report["wide"]["client"] + 1 and not report["wide"]["shown"], report
            # Only the window narrows: the text wraps past the owner's 160 px and the card says so.
            page.set_viewport_size({"width": 600, "height": 900})
            page.wait_for_timeout(1000)
            report["narrow"] = page.evaluate(more, selector)
            assert report["narrow"]["scroll"] > report["narrow"]["client"] + 1, report
            assert report["narrow"]["shown"] and report["narrow"]["text"].startswith("Show all"), report
            page.screenshot(path=str(shots / f"{engine}-reflow-narrow.png"))

            # Show all, then a press on the bottom edge that hardly moves: still shown, nothing saved.
            page.locator(f"{selector} [data-widget-expand]").click()
            page.wait_for_timeout(400)
            assert page.evaluate(more, selector)["text"].startswith("Back to 160"), report
            drag(page, f"{selector} [data-widget-height-handle]", 0, 3, steps=3)
            page.wait_for_timeout(400)
            report["noop"] = page.evaluate(more, selector)
            assert report["noop"]["text"].startswith("Back to 160"), report
            assert page.evaluate(stored, key) == {"w": 12, "h": 160}
            # A drag back onto the owner's height: Show all ends there, the saved height unchanged.
            natural = report["noop"]["client"]
            drag(page, f"{selector} [data-widget-height-handle]", 0, 160 - natural, steps=12)
            page.wait_for_timeout(500)
            report["dropped"] = page.evaluate(more, selector)
            assert report["dropped"]["text"].startswith("Show all"), report
            assert abs(report["dropped"]["client"] - 160) <= 1, report
            assert page.evaluate(stored, key) == {"w": 12, "h": 160}

            # Wide again: the text fits under 160 px and the line goes.
            page.set_viewport_size({"width": 1920, "height": 1080})
            page.wait_for_timeout(1000)
            report["wide_again"] = page.evaluate(more, selector)
            assert not report["wide_again"]["shown"], report

            # A disclosure opened under the owner's height changes no node or text: the card hears its toggle and says
            # what is out of view; closed again, the line goes.
            other = f'[data-widget-key="{probe}:other"]'
            page.evaluate("""(selector) => document.querySelector(`${selector} [data-widget-mount]`).insertAdjacentHTML('beforeend',
                '<details data-probe-details><summary>More</summary><div style="height:400px">Detail</div></details>')""", other)
            page.wait_for_timeout(500)
            report["details_closed"] = page.evaluate(more, other)
            assert not report["details_closed"]["shown"], report
            summary = page.locator(f"{other} [data-probe-details] summary")
            summary.click()
            page.wait_for_timeout(500)
            report["details_open"] = page.evaluate(more, other)
            assert report["details_open"]["shown"] and report["details_open"]["text"].startswith("Show all"), report
            summary.click()
            page.wait_for_timeout(500)
            assert not page.evaluate(more, other)["shown"], report
        finally:
            (shots / f"{engine}-reflow.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            context.close()
            browser.close()


# --- media that learns its size after the card is drawn ---------------------------------------------------------------
# An image's bytes arrive after its card was drawn under an owner height: no node changes, the image's box does, and the
# card's line says what is out of view. A failed image changes its box too.

_TALL_SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="1200"><rect width="64" height="1200" fill="teal"/></svg>'


def _write_media(data_dir: pathlib.Path) -> str:
    from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_review_state

    name = "board_media_probe"
    skill_dir = data_dir / "skills" / "external" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(textwrap.dedent(f"""\
        ---
        name: {name}
        description: A declarative card whose image arrives late.
        version: 0.1.0
        type: extension
        entry: plugin.py
        permissions: ["widget"]
        ---
        # Media probe
        """), encoding="utf-8")
    (skill_dir / "plugin.py").write_text(textwrap.dedent("""\
        def register(api):
            api.register_ui_tab("late", "Late image", render={"kind": "declarative", "schema_version": 1,
                "components": [{"type": "image", "route": "picture", "alt": "A tall picture"}]})
        """), encoding="utf-8")
    save_review_state(data_dir, name, SkillReviewState(status="pass", content_hash=compute_content_hash(skill_dir, manifest_entry="plugin.py")))
    return name


@pytest.mark.parametrize("engine", ("chromium", "webkit"))
def test_an_image_that_arrives_late_under_an_owner_height_brings_show_all(direct_server_with_data, engine):
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    probe = _write_media(data_dir)
    key = f"{probe}:late"
    selector = f'[data-widget-key="{key}"]'
    shots = evidence_dir(data_dir)
    report = {}
    line = """(selector) => { const card = document.querySelector(selector); const body = card.querySelector('[data-widget-mount]');
        const more = card.querySelector('[data-widget-more]');
        return { shown: !more.hidden, text: more.textContent.trim(), client: body.clientHeight, scroll: body.scrollHeight }; }"""
    with sync_playwright() as pw:
        browser, context = launch(pw, engine, viewport={"width": 1100, "height": 750})
        page = context.new_page()
        held = []
        try:
            page.route(f"**/api/extensions/{probe}/picture*", lambda route: held.append(route))
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            assert page.evaluate("""async (skill) => (await fetch(`/api/skills/${encodeURIComponent(skill)}/toggle`, {
                method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({enabled: true}),
            })).status""", probe) == 200
            page.evaluate("""async (key) => fetch('/api/ui/preferences', {method: 'POST',
                headers: {'Content-Type': 'application/json'}, body: JSON.stringify({widget_size: {[key]: {w: 0, h: 360}}})})""", key)
            page.click('[data-nav-page="widgets"]')
            page.locator(f"{selector} img").wait_for(state="attached", timeout=30_000)
            deadline = time.time() + 15
            while not held and time.time() < deadline:
                page.wait_for_timeout(100)
            assert held, "the image request is held"
            page.wait_for_timeout(400)
            report["before"] = page.evaluate(line, selector)
            assert not report["before"]["shown"], report
            held.pop().fulfill(status=200, content_type="image/svg+xml", body=_TALL_SVG)
            page.wait_for_function("(selector) => document.querySelector(`${selector} img`).naturalHeight === 1200",
                                   arg=selector, timeout=15_000)
            page.wait_for_timeout(600)
            report["after"] = page.evaluate(line, selector)
            after = report["after"]
            assert after["shown"] and after["scroll"] > after["client"] + 800, report
            assert after["text"] == f"Show all · {after['scroll'] - after['client']:,} px more", report
            page.screenshot(path=str(shots / f"{engine}-late-image.png"))
        finally:
            (shots / f"{engine}-late-image.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            context.close()
            browser.close()


# --- a finger: the grip in the card's own foot, the gaps scroll ----------------------------------------------------------
# Under a coarse pointer every card keeps a foot the touch minimum tall, inside the card, and the bottom edge's button
# is the grip at its right end: it covers nothing the widget draws, nor the gap or the next card. A swipe that starts in
# the gap scrolls the board and saves nothing; a tap on the widget's own bottom-right control is the widget's; a tap on
# Show all is Show all; a swipe on the grip sets a custom height (and, where widths apply, sideways the width); a tap,
# or a drag the browser cancels, saves nothing; the grip's keys still step the height.

def _touch(page, points, *, end: str = "touchEnd") -> None:
    """A real touch gesture through Chromium's CDP touch input: start at points[0], move through the rest, end."""
    session = page.context.new_cdp_session(page)
    x, y = points[0]
    session.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{"x": x, "y": y}]})
    for x, y in points[1:]:
        session.send("Input.dispatchTouchEvent", {"type": "touchMove", "touchPoints": [{"x": x, "y": y}]})
        page.wait_for_timeout(16)
    session.send("Input.dispatchTouchEvent", {"type": end, "touchPoints": []})
    session.detach()
    page.wait_for_timeout(500)


def _line(x: float, y: float, dy: float, steps: int = 12, dx: float = 0) -> list:
    return [(x + dx * step / steps, y + dy * step / steps) for step in range(steps + 1)]


# A module that draws a 160 px box with its own button at the bottom right; the button counts the taps it gets.
_CORNER = """(() => {
    document.body.style.margin = '0';
    window.__cornerTaps = 0;
    const root = document.getElementById('root');
    root.style.cssText = 'height:160px;display:flex;align-items:flex-end;justify-content:flex-end;box-sizing:border-box';
    root.innerHTML = '<button id="corner" type="button" style="height:32px">Corner</button>';
    document.getElementById('corner').addEventListener('click', () => { window.__cornerTaps += 1; });
})();
"""


def _write_touch_cards(data_dir: pathlib.Path) -> str:
    """Six short declarative cards (a phone board long enough to scroll) and a module whose control is in its corner."""
    from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_review_state

    name = "board_touch_probe"
    skill_dir = data_dir / "skills" / "external" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(textwrap.dedent(f"""\
        ---
        name: {name}
        description: Six short cards for touch scrolling over the Widgets board, and one module with a corner control.
        version: 0.1.0
        type: extension
        entry: plugin.py
        permissions: ["widget"]
        ---
        # Touch probe
        """), encoding="utf-8")
    (skill_dir / "plugin.py").write_text(textwrap.dedent("""\
        def register(api):
            api.register_ui_tab("corner", "Corner", render={"kind": "module", "entry": "corner.js", "start": "auto"})
            for i in range(6):
                api.register_ui_tab(f"c{i}", f"Card {i}", render={"kind": "declarative", "schema_version": 1,
                    "components": [{"type": "callout", "text": f"Card {i} line {j}: a short note."} for j in range(4)]})
        """), encoding="utf-8")
    (skill_dir / "corner.js").write_text(_CORNER, encoding="utf-8")
    save_review_state(data_dir, name, SkillReviewState(status="pass", content_hash=compute_content_hash(skill_dir, manifest_entry="plugin.py")))
    return name


HIT = """([x, y]) => { const t = document.elementFromPoint(x, y);
    return t?.closest('[data-widget-height-handle]') ? 'grip' : t?.closest('[data-widget-expand]') ? 'show-all'
        : t?.closest('[data-widget-adjust-done]') ? 'done' : t?.closest('iframe') ? 'frame'
        : t?.closest('[data-widget-key]') ? 'card' : 'gap'; }"""
ADJUSTING = "() => [...document.querySelectorAll('[data-widget-adjusting]')].map((node) => node.dataset.widgetKey)"
FOCUSED = """() => { const a = document.activeElement; return a?.matches?.('[data-widget-height-handle]') ? 'grip'
    : a?.matches?.('[data-widget-menu-trigger]') ? 'trigger' : (a?.tagName || '').toLowerCase();
    }"""
FOCUSED_CARD = "() => document.activeElement?.closest?.('[data-widget-key]')?.dataset.widgetKey || ''"
SCROLL = "() => document.querySelector('.widgets-scroll').scrollTop"
SIZES = "async () => (await (await fetch('/api/ui/preferences')).json()).widget_size || {}"
# Every card's foot: how far its border box reaches below the last thing it draws (its body, or its Show all line), and
# what a finger meets just inside its bottom right corner.
FEET = """() => [...document.querySelectorAll('#widgets-list [data-widget-key]')].map((node) => {
    const box = node.getBoundingClientRect();
    const more = node.querySelector('[data-widget-more]');
    const last = (more && !more.hidden ? more : node.querySelector('[data-widget-mount]')).getBoundingClientRect();
    const shown = (selector) => getComputedStyle(node.querySelector(selector)).display !== 'none';
    const t = document.elementFromPoint(box.right - 22, box.bottom - 22);
    return { key: node.dataset.widgetKey, foot: box.bottom - last.bottom, grip: shown('[data-widget-height-handle]'),
        done: shown('.widgets-card-adjust'), corner: t?.closest('[data-widget-height-handle]') ? 'grip' : 'content' };
})"""


def _tap(page, selector: str) -> None:
    """One finger tap at the middle of `selector`, brought into view if it is not (the page's touchscreen: Chromium and
    WebKit alike)."""
    page.locator(selector).scroll_into_view_if_needed()
    box = page.locator(selector).bounding_box()
    page.touchscreen.tap(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.wait_for_timeout(300)


def _resize(page, selector: str) -> None:
    """By finger: the card's ⋮ menu, then its Resize."""
    _tap(page, f"{selector} [data-widget-menu-trigger]")
    item = "body > .skills-card-menu-dialog[open] [data-widget-adjust]"
    page.locator(item).wait_for(state="visible", timeout=5_000)
    _tap(page, item)


def _in_view(page, selector: str) -> dict:
    box = page.locator(selector).bounding_box()
    view = page.evaluate("() => { const r = document.querySelector('.widgets-scroll').getBoundingClientRect(); return [r.top, r.bottom]; }")
    return {"box": box, "view": view, "inside": bool(box) and box["y"] >= view[0] - 0.5 and box["y"] + box["height"] <= view[1] + 0.5}


def test_a_finger_scrolls_from_the_gap_and_sizes_only_the_card_it_chose_to_resize(direct_server_with_data):
    """Native touch through Chromium's CDP (DESIGN "Widgets board": under a finger only the card picked by its menu's
    Resize has a foot). No other card reserves a strip or catches a finger at its corner; the chosen card's foot sits
    below its Show all line; it survives scrolling and repeated gestures; Done, a tap outside and Escape put it away;
    entering and leaving write nothing; a frame keeps its document throughout."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    probe = _write_touch_cards(data_dir)
    shots = evidence_dir(data_dir)
    report = {}
    key = lambda tab: f"{probe}:{tab}"  # noqa: E731
    sel = lambda tab: f'[data-widget-key="{probe}:{tab}"]'  # noqa: E731
    stored = {key("c2"): {"w": 0, "h": 120}, key("c4"): {"w": 0, "h": 2700}}
    with sync_playwright() as pw:
        browser, context = launch(pw, "chromium", viewport={"width": 390, "height": 844}, has_touch=True, is_mobile=True)
        page = context.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            assert page.evaluate("""async (skill) => (await fetch(`/api/skills/${encodeURIComponent(skill)}/toggle`, {
                method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({enabled: true}),
            })).status""", probe) == 200
            # One card under an owner height below its content (its Show all line shows), one far taller than its own.
            page.evaluate("""async (sizes) => fetch('/api/ui/preferences', {method: 'POST',
                headers: {'Content-Type': 'application/json'}, body: JSON.stringify({widget_size: sizes})})""", stored)
            page.goto(f"{url}#widgets", wait_until="domcontentloaded", timeout=30_000)
            page.reload(wait_until="domcontentloaded", timeout=30_000)
            for i in range(6):
                page.locator(f"{sel(f'c{i}')} .widget-callout").nth(3).wait_for(state="attached", timeout=30_000)
            corner = sel("corner")
            page.wait_for_function("(selector) => document.querySelector(`${selector} iframe`)?.dataset.widgetContentHeight",
                                   arg=corner, timeout=30_000)
            page.wait_for_timeout(500)
            assert page.evaluate("() => document.documentElement.dataset.pointer") == "coarse"
            assert page.evaluate("() => document.getElementById('widgets-list').dataset.widgetLayout") == "stack"
            assert page.evaluate("() => { const s = document.querySelector('.widgets-scroll'); return s.scrollHeight - s.clientHeight; }") > 400
            note = sel("c1")
            page.evaluate("() => { document.querySelector('.widgets-scroll').scrollTop = 0; }")
            page.wait_for_timeout(300)

            # No card is adjusted: none reserves a strip, none shows Done or a grip, none catches a finger at its corner.
            feet = page.evaluate(FEET)
            report["feet"] = feet
            assert page.evaluate(ADJUSTING) == [], report
            assert all(not f["grip"] and not f["done"] and f["corner"] == "content" and f["foot"] <= 1.5 for f in feet), report
            for selector in ("[data-widget-corner-handle]", "[data-widget-resize-handle]", "[data-widget-height-handle]"):
                assert page.locator(f"{note} {selector}").is_hidden(), selector
            page.screenshot(path=str(shots / "chromium-touch-board.png"))

            box = page.locator(note).bounding_box()
            gap = (box["x"] + box["width"] * 0.25, box["y"] + box["height"] + 5)
            report["hit_gap"] = page.evaluate(HIT, list(gap))
            assert report["hit_gap"] == "gap", report
            # A swipe from the gap scrolls the board and saves nothing.
            before = page.evaluate(SCROLL)
            _touch(page, _line(*gap, -200))
            report["gap_swipe"] = {"before": before, "after": page.evaluate(SCROLL), "sizes": page.evaluate(SIZES)}
            assert report["gap_swipe"]["after"] > before + 50 and report["gap_swipe"]["sizes"] == stored, report

            # The widget's own control at its bottom right is the widget's: a tap reaches it, nothing is resized.
            page.locator(corner).scroll_into_view_if_needed()
            page.wait_for_timeout(300)
            frame = page.locator(f"{corner} iframe").element_handle().content_frame()
            frame.evaluate("() => { window.__boardMark = 'same-window'; }")
            control = frame.locator("#corner").bounding_box()
            corner_height = page.locator(corner).bounding_box()["height"]
            tap = (control["x"] + control["width"] - 6, control["y"] + control["height"] - 6)
            report["corner"] = {"control": control, "hit": page.evaluate(HIT, list(tap))}
            assert report["corner"]["hit"] == "frame", report
            _touch(page, [tap])
            report["corner"]["taps"] = frame.evaluate("() => window.__cornerTaps")
            assert report["corner"]["taps"] == 1 and abs(page.locator(corner).bounding_box()["height"] - corner_height) <= 1, report

            # Show all is Show all, its whole row its own: a tap lifts the height in this window only, Back returns it.
            bounded = sel("c2")
            page.locator(bounded).scroll_into_view_if_needed()
            page.wait_for_timeout(300)
            expand = page.locator(f"{bounded} [data-widget-expand]").bounding_box()
            ex, ey = expand["x"] + expand["width"] / 2, expand["y"] + expand["height"] / 2
            report["show_all"] = {"button": expand, "hit": [page.evaluate(HIT, [x, y]) for x, y in
                                  ((ex, ey), (expand["x"] + 2, ey), (expand["x"] + expand["width"] - 2, ey), (ex, ey - 21), (ex, ey + 21))]}
            assert report["show_all"]["hit"] == ["show-all"] * 5, report
            _touch(page, [(ex, ey)])
            assert page.locator(f"{bounded} [data-widget-expand]").inner_text().startswith("Back to 120")
            page.locator(f"{bounded} [data-widget-expand]").scroll_into_view_if_needed()
            _tap(page, f"{bounded} [data-widget-expand]")
            assert page.locator(f"{bounded} [data-widget-expand]").inner_text().startswith("Show all")
            assert page.evaluate(SIZES) == stored, "Show all is never saved"

            # Resize from the card's menu: that card alone gets its foot, Done at the left and the 44 px grip at the
            # right, below everything it draws; its grip takes the focus. Choosing it writes nothing.
            page.evaluate("() => { document.querySelector('.widgets-scroll').scrollTop = 0; }")
            page.wait_for_timeout(300)
            _resize(page, note)
            box = page.locator(note).bounding_box()
            grip = page.locator(f"{note} [data-widget-height-handle]").bounding_box()
            body = page.locator(f"{note} [data-widget-mount]").bounding_box()
            done = page.locator(f"{note} [data-widget-adjust-done]").bounding_box()
            report["adjust"] = {"card": box, "grip": grip, "body": body, "done": done, "adjusting": page.evaluate(ADJUSTING),
                                "focus": page.evaluate(FOCUSED), "sizes": page.evaluate(SIZES), "feet": page.evaluate(FEET)}
            assert report["adjust"]["adjusting"] == [key("c1")] and report["adjust"]["focus"] == "grip", report
            assert round(grip["width"]) == 44 and round(grip["height"]) == 44, report
            assert grip["x"] >= box["x"] and grip["x"] + grip["width"] <= box["x"] + box["width"], report
            assert grip["y"] >= body["y"] + body["height"] - 0.5 and grip["y"] + grip["height"] <= box["y"] + box["height"] + 0.5, report
            assert done["x"] + done["width"] <= grip["x"] and done["y"] >= body["y"] + body["height"] - 0.5, report
            assert [f["key"] for f in report["adjust"]["feet"] if f["grip"] or f["done"] or f["foot"] > 1.5] == [key("c1")], report
            assert report["adjust"]["sizes"] == stored, report
            page.screenshot(path=str(shots / "chromium-touch-resize.png"))

            # A swipe from the gap still scrolls the board, and the card stays chosen.
            gap = (box["x"] + box["width"] * 0.25, box["y"] + box["height"] + 5)
            assert page.evaluate(HIT, list(gap)) == "gap"
            before = page.evaluate(SCROLL)
            _touch(page, _line(*gap, -150))
            report["adjust_scroll"] = {"before": before, "after": page.evaluate(SCROLL), "adjusting": page.evaluate(ADJUSTING)}
            assert report["adjust_scroll"]["after"] > before + 50 and report["adjust_scroll"]["adjusting"] == [key("c1")], report
            page.evaluate("() => { document.querySelector('.widgets-scroll').scrollTop = 0; }")
            page.wait_for_timeout(300)
            grip = page.locator(f"{note} [data-widget-height-handle]").bounding_box()
            centre = (grip["x"] + grip["width"] / 2, grip["y"] + grip["height"] / 2)
            height = page.locator(note).bounding_box()["height"]
            # A tap on the grip, and a drag on it that the browser cancels, save nothing and leave the card as it was.
            _touch(page, [centre, (centre[0], centre[1] + 2)])
            _touch(page, _line(*centre, -60, steps=6), end="touchCancel")
            report["tap_and_cancel"] = {"sizes": page.evaluate(SIZES), "height": page.locator(note).bounding_box()["height"]}
            assert report["tap_and_cancel"]["sizes"] == stored and abs(report["tap_and_cancel"]["height"] - height) <= 1, report
            # A swipe on the grip sets a custom height (widths do not apply on a phone, so a sideways part does nothing);
            # the card stays chosen, and a second swipe sets another.
            _touch(page, _line(*centre, -80, dx=-40))
            first = page.evaluate(SIZES).get(key("c1"))
            assert first and first["w"] == 0 and first["h"] >= 120, first
            assert abs(page.locator(f"{note} [data-widget-mount]").evaluate("(b) => b.clientHeight") - first["h"]) <= 1
            assert page.evaluate(ADJUSTING) == [key("c1")]
            grip = page.locator(f"{note} [data-widget-height-handle]").bounding_box()
            centre = (grip["x"] + grip["width"] / 2, grip["y"] + grip["height"] / 2)
            # Past the content's own height (near it the grip catches the natural height again).
            _touch(page, _line(*centre, 160))
            second = page.evaluate(SIZES).get(key("c1"))
            report["grip_swipes"] = [first, second]
            assert second and second["h"] > first["h"] + 120, report
            # The grip's keys still step the height: Home is the least, End the content's own again.
            page.locator(f"{note} [data-widget-height-handle]").focus()
            page.keyboard.press("Home")
            page.wait_for_timeout(300)
            report["keys_home"] = page.evaluate(SIZES).get(key("c1"))
            page.keyboard.press("End")
            page.wait_for_timeout(300)
            report["keys_end"] = page.evaluate(SIZES).get(key("c1"))
            assert report["keys_home"] == {"w": 0, "h": 120} and not (report["keys_end"] or {}).get("h"), report
            sizes_now = page.evaluate(SIZES)

            # A tap on its own content keeps it; Done puts it away and gives the focus back to its menu button without
            # moving the board.
            _tap(page, f"{note} .widget-callout >> nth=0")
            assert page.evaluate(ADJUSTING) == [key("c1")]
            page.evaluate("() => { document.querySelector('.widgets-scroll').scrollTop = 40; }")
            page.wait_for_timeout(200)
            before = page.evaluate(SCROLL)
            _tap(page, f"{note} [data-widget-adjust-done]")
            report["done"] = {"adjusting": page.evaluate(ADJUSTING), "focus": page.evaluate(FOCUSED),
                              "focus_card": page.evaluate(FOCUSED_CARD), "scroll": [before, page.evaluate(SCROLL)]}
            assert report["done"]["adjusting"] == [] and report["done"]["focus"] == "trigger", report
            assert report["done"]["focus_card"] == key("c1") and abs(report["done"]["scroll"][1] - before) <= 1, report
            assert all(not f["grip"] and f["foot"] <= 1.5 for f in page.evaluate(FEET)), "the foot is gone"

            # A tap on another card puts it away and moves no focus there.
            page.evaluate("() => { document.querySelector('.widgets-scroll').scrollTop = 0; }")
            page.wait_for_timeout(200)
            _resize(page, note)
            _tap(page, f"{sel('c3')} .widget-callout >> nth=0")
            report["outside"] = {"adjusting": page.evaluate(ADJUSTING), "focus": page.evaluate(FOCUSED)}
            assert report["outside"]["adjusting"] == [] and report["outside"]["focus"] != "trigger", report

            # Under Show all: the foot is its own row below the line; Show all and Back keep the card chosen; Escape
            # (a keyboard on the tablet) puts it away and returns to the menu button.
            page.locator(bounded).scroll_into_view_if_needed()
            page.wait_for_timeout(300)
            _resize(page, bounded)
            more = page.locator(f"{bounded} [data-widget-more]").bounding_box()
            grip = page.locator(f"{bounded} [data-widget-height-handle]").bounding_box()
            report["show_all_foot"] = {"more": more, "grip": grip, "adjusting": page.evaluate(ADJUSTING)}
            assert report["show_all_foot"]["adjusting"] == [key("c2")], report
            assert grip["y"] >= more["y"] + more["height"] - 0.5, report
            page.locator(bounded).screenshot(path=str(shots / "chromium-touch-resize-show-all.png"))
            _tap(page, f"{bounded} [data-widget-expand]")
            assert page.locator(f"{bounded} [data-widget-expand]").inner_text().startswith("Back to 120")
            page.locator(f"{bounded} [data-widget-expand]").scroll_into_view_if_needed()
            _tap(page, f"{bounded} [data-widget-expand]")
            assert page.locator(f"{bounded} [data-widget-expand]").inner_text().startswith("Show all")
            assert page.evaluate(ADJUSTING) == [key("c2")]
            page.locator(f"{bounded} [data-widget-height-handle]").focus()
            page.keyboard.press("Escape")
            page.wait_for_timeout(200)
            report["escape"] = {"adjusting": page.evaluate(ADJUSTING), "focus": page.evaluate(FOCUSED), "card": page.evaluate(FOCUSED_CARD)}
            assert report["escape"] == {"adjusting": [], "focus": "trigger", "card": key("c2")}, report

            # A card taller than the screen: Resize from its menu at its top brings its grip into view.
            tall = sel("c4")
            page.evaluate("(selector) => document.querySelector(selector).scrollIntoView({block: 'start'})", tall)
            page.wait_for_timeout(300)
            _resize(page, tall)
            report["tall"] = _in_view(page, f"{tall} [data-widget-height-handle]")
            report["tall"]["focus"] = page.evaluate(FOCUSED)
            report["tall"]["card"] = page.locator(tall).bounding_box()
            assert report["tall"]["card"]["height"] > 2700 and report["tall"]["inside"] and report["tall"]["focus"] == "grip", report
            page.screenshot(path=str(shots / "chromium-touch-resize-tall.png"))
            _tap(page, f"{tall} [data-widget-adjust-done]")

            # The framed card: chosen, resized and put away, its document is the one it had.
            page.locator(corner).scroll_into_view_if_needed()
            page.wait_for_timeout(300)
            _resize(page, corner)
            grip = page.locator(f"{corner} [data-widget-height-handle]").bounding_box()
            _touch(page, _line(grip["x"] + 22, grip["y"] + 22, 60))
            _tap(page, f"{corner} [data-widget-adjust-done]")
            report["frame_kept"] = {"mark": frame.evaluate("() => window.__boardMark ?? null"),
                                    "frames": page.locator(f"{corner} iframe").count(), "sizes": page.evaluate(SIZES)}
            assert report["frame_kept"]["mark"] == "same-window" and report["frame_kept"]["frames"] == 1, report
            assert report["frame_kept"]["sizes"].get(key("corner"), {}).get("h", 0) > 0, report
            # Best effort, not a promise: a tap inside another card's frame is heard only as the window's blur.
            _resize(page, note)
            frame_box = page.locator(f"{corner} iframe").bounding_box()
            page.locator(corner).scroll_into_view_if_needed()
            frame_box = page.locator(f"{corner} iframe").bounding_box()
            page.touchscreen.tap(frame_box["x"] + 20, frame_box["y"] + 20)
            page.wait_for_timeout(300)
            report["frame_tap_ends"] = page.evaluate(ADJUSTING) == []
            for k, v in sizes_now.items():
                if k != key("corner"):
                    assert page.evaluate(SIZES).get(k) == v, ("entering and leaving wrote nothing", k)
        finally:
            (shots / "chromium-touch-grip.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            context.close()

        # A two-column tablet under a finger: no card has a foot until Resize, then a sideways swipe on its grip sets
        # the width too.
        context = browser.new_context(viewport={"width": 1024, "height": 768}, has_touch=True)
        page = context.new_page()
        tablet = {}
        try:
            page.goto(f"{url}#widgets", wait_until="domcontentloaded", timeout=30_000)
            for i in range(6):
                page.locator(f"{sel(f'c{i}')} .widget-callout").nth(3).wait_for(state="attached", timeout=30_000)
            page.wait_for_timeout(500)
            assert page.evaluate("() => document.getElementById('widgets-list').dataset.widgetLayout") == "columns"
            target = sel("c3")
            page.locator(target).scroll_into_view_if_needed()
            page.wait_for_timeout(300)
            tablet["feet"] = page.evaluate(FEET)
            assert all(not f["grip"] and f["foot"] <= 1.5 for f in tablet["feet"]), tablet
            _resize(page, target)
            box = page.locator(target).bounding_box()
            grip = page.locator(f"{target} [data-widget-height-handle]").bounding_box()
            tablet["card"], tablet["grip"] = box, grip
            assert round(grip["width"]) == 44 and round(grip["height"]) == 44, tablet
            assert grip["x"] + grip["width"] <= box["x"] + box["width"] and grip["y"] + grip["height"] <= box["y"] + box["height"] + 0.5, tablet
            assert page.locator(f"{target} [data-widget-resize-handle]").is_hidden(), "no thin edge for a finger"
            centre = (grip["x"] + grip["width"] / 2, grip["y"] + grip["height"] / 2)
            _touch(page, _line(*centre, -60, dx=420))
            tablet["sizes"] = page.evaluate(SIZES)
            saved_size = tablet["sizes"].get(key("c3"))
            assert saved_size and saved_size["w"] > 0 and saved_size["h"] > 0, tablet
            page.screenshot(path=str(shots / "chromium-touch-grip-tablet.png"))
        finally:
            (shots / "chromium-touch-grip-tablet.json").write_text(json.dumps(tablet, indent=2), encoding="utf-8")
            context.close()
            browser.close()


@pytest.mark.parametrize("engine", ("chromium", "webkit"))
def test_resize_from_the_menu_is_a_finger_mode_a_mouse_never_sees(direct_server_with_data, engine):
    """In both engines: under a touch screen the menu's Resize gives one card the foot (Done, the grip, focus on the
    grip), Done and a tap outside put it away, nothing is written for either; with a mouse the menu offers no Resize,
    no card has a foot and the thin edges are where they were."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    probe = _write_touch_cards(data_dir)
    shots = evidence_dir(data_dir)
    report = {}
    key = lambda tab: f"{probe}:{tab}"  # noqa: E731
    sel = lambda tab: f'[data-widget-key="{probe}:{tab}"]'  # noqa: E731
    stored = {key("c2"): {"w": 0, "h": 120}}

    def ready(page):
        for i in range(6):
            page.locator(f"{sel(f'c{i}')} .widget-callout").nth(3).wait_for(state="attached", timeout=30_000)
        page.wait_for_timeout(500)

    with sync_playwright() as pw:
        browser, context = launch(pw, engine, viewport={"width": 390, "height": 844}, has_touch=True)
        page = context.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            assert page.evaluate("""async (skill) => (await fetch(`/api/skills/${encodeURIComponent(skill)}/toggle`, {
                method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({enabled: true}),
            })).status""", probe) == 200
            page.evaluate("""async (sizes) => fetch('/api/ui/preferences', {method: 'POST',
                headers: {'Content-Type': 'application/json'}, body: JSON.stringify({widget_size: sizes})})""", stored)
            page.goto(f"{url}#widgets", wait_until="domcontentloaded", timeout=30_000)
            page.reload(wait_until="domcontentloaded", timeout=30_000)
            ready(page)
            report["pointer"] = page.evaluate("() => document.documentElement.dataset.pointer")
            if report["pointer"] != "coarse":
                pytest.skip(f"{engine} with a touch screen does not report a coarse primary pointer here")
            feet = page.evaluate(FEET)
            report["feet"] = feet
            assert all(not f["grip"] and not f["done"] and f["foot"] <= 1.5 and f["corner"] == "content" for f in feet), report
            note = sel("c1")
            _resize(page, note)
            box = page.locator(note).bounding_box()
            grip = page.locator(f"{note} [data-widget-height-handle]").bounding_box()
            report["adjust"] = {"adjusting": page.evaluate(ADJUSTING), "focus": page.evaluate(FOCUSED), "card": box, "grip": grip,
                                "in_view": _in_view(page, f"{note} [data-widget-height-handle]")["inside"], "sizes": page.evaluate(SIZES)}
            assert report["adjust"]["adjusting"] == [key("c1")] and report["adjust"]["focus"] == "grip", report
            assert round(grip["width"]) == 44 and round(grip["height"]) == 44 and report["adjust"]["in_view"], report
            assert report["adjust"]["sizes"] == stored, report
            page.screenshot(path=str(shots / f"{engine}-touch-resize.png"))
            # The grip answers a pointer's drag (a mouse here: the engine's own touch drag is CDP-only).
            drag(page, f"{note} [data-widget-height-handle]", 0, -60)
            page.wait_for_timeout(300)
            report["dragged"] = page.evaluate(SIZES).get(key("c1"))
            assert report["dragged"] and report["dragged"]["h"] >= 120, report
            _tap(page, f"{note} [data-widget-adjust-done]")
            report["done"] = {"adjusting": page.evaluate(ADJUSTING), "focus": page.evaluate(FOCUSED), "card": page.evaluate(FOCUSED_CARD)}
            assert report["done"] == {"adjusting": [], "focus": "trigger", "card": key("c1")}, report
            after_done = page.evaluate(SIZES)
            # A tap in the gap below it (no control there) puts it away.
            _resize(page, note)
            box = page.locator(note).bounding_box()
            gap = (box["x"] + box["width"] * 0.25, box["y"] + box["height"] + 5)
            report["gap"] = {"hit": page.evaluate(HIT, list(gap))}
            page.touchscreen.tap(*gap)
            page.wait_for_timeout(300)
            report["gap"]["adjusting"] = page.evaluate(ADJUSTING)
            assert report["gap"] == {"hit": "gap", "adjusting": []}, report
            # A tap on a neighbouring card's button is that button's, and it puts the foot away too: the foot is not
            # taken off before the click, so nothing moves under the finger.
            bounded = sel("c2")
            _resize(page, note)
            _tap(page, f"{bounded} [data-widget-expand]")
            report["neighbour"] = {"adjusting": page.evaluate(ADJUSTING),
                                   "button": page.locator(f"{bounded} [data-widget-expand]").inner_text()}
            assert report["neighbour"]["adjusting"] == [] and report["neighbour"]["button"].startswith("Back to 120"), report
            _tap(page, f"{bounded} [data-widget-expand]")
            # Under Show all the foot is its own row below the line.
            page.locator(bounded).scroll_into_view_if_needed()
            page.wait_for_timeout(300)
            _resize(page, bounded)
            more = page.locator(f"{bounded} [data-widget-more]").bounding_box()
            grip = page.locator(f"{bounded} [data-widget-height-handle]").bounding_box()
            report["show_all_foot"] = {"more": more, "grip": grip}
            assert grip["y"] >= more["y"] + more["height"] - 0.5, report
            page.locator(bounded).screenshot(path=str(shots / f"{engine}-touch-resize-show-all.png"))
            _tap(page, f"{sel('c4')} .widget-callout >> nth=0")
            assert page.evaluate(ADJUSTING) == [], "a tap outside"
            assert page.evaluate(SIZES) == after_done, "choosing and putting away write nothing"
        finally:
            (shots / f"{engine}-touch-resize.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            context.close()

        context = browser.new_context(viewport={"width": 1100, "height": 750})
        page = context.new_page()
        desk = {}
        try:
            page.goto(f"{url}#widgets", wait_until="domcontentloaded", timeout=30_000)
            ready(page)
            assert page.evaluate("() => document.documentElement.dataset.pointer") == "fine"
            desk["feet"] = page.evaluate(FEET)
            assert all(not f["done"] and f["foot"] <= 1.5 for f in desk["feet"]), desk
            note = sel("c1")
            edge = page.locator(f"{note} [data-widget-height-handle]").bounding_box()
            desk["edge"] = edge
            assert round(edge["height"]) == 12, desk
            page.locator(f"{note} [data-widget-menu-trigger]").click()
            page.locator("body > .skills-card-menu-dialog[open]").wait_for(state="visible")
            desk["resize_offered"] = page.locator("body > .skills-card-menu-dialog[open] [data-widget-adjust]").is_visible()
            assert desk["resize_offered"] is False, desk
            page.keyboard.press("Escape")
            # The bottom edge still drags a height with a mouse.
            drag(page, f"{note} [data-widget-height-handle]", 0, 80)
            page.wait_for_timeout(300)
            desk["dragged"] = page.evaluate(SIZES).get(key("c1"))
            assert desk["dragged"] and desk["dragged"]["h"] > 0, desk
            assert page.evaluate(ADJUSTING) == []
        finally:
            (shots / f"{engine}-desk-no-resize.json").write_text(json.dumps(desk, indent=2), encoding="utf-8")
            context.close()
            browser.close()


# --- what lies around a narrow widget ---------------------------------------------------------------------------------
# A frame inherits the page's colour scheme. A browser lays an opaque canvas of a document's own scheme under a
# document whose scheme differs from its frame's: on the dark theme a transparent, light-scheme page (a calculator,
# or any unstyled page) sits on white. The board's frame paints no fill of its own and the host sets no colour scheme
# (`appearance` is the author's intent only): a module that paints dark declares its own scheme.

_CALC = """(() => {{
    const style = document.createElement('style');
    style.textContent = ':root{{--panel:#fffaf5;--ink:#3b2f3a}}:root[data-theme=dark]{{--panel:#2a2330;--ink:#f3e8f0}}'
        + 'html,body{{margin:0;background:transparent;color:var(--ink);font:14px system-ui}}'
        + '.wrap{{max-width:340px;height:300px;margin:4px auto 8px;padding:12px;background:var(--panel);border-radius:24px}}'
        + {extra!r};
    document.head.appendChild(style);
    document.getElementById('root').innerHTML = '<div class="wrap">Calculator</div>';
    window.OuroborosWidget.onTheme((theme) => {{ document.documentElement.dataset.theme = theme; }});
}})();
"""
_BARE = "document.getElementById('root').innerHTML = '<p>Plain text in the browser defaults.</p>'.repeat(12);\n"
_PROBE_TABS = ("plain", "scheme", "host", "fixed", "bare")


def _write_calculators(data_dir: pathlib.Path) -> str:
    from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_review_state

    name = "board_canvas_probe"
    skill_dir = data_dir / "skills" / "external" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(textwrap.dedent(f"""\
        ---
        name: {name}
        description: Narrow transparent modules with and without a colour scheme, and an unstyled control.
        version: 0.1.0
        type: extension
        entry: plugin.py
        permissions: ["widget"]
        ---
        # Canvas probe
        """), encoding="utf-8")
    (skill_dir / "plugin.py").write_text(textwrap.dedent("""\
        def register(api):
            for tab, title, extra in (("plain", "Without a scheme", {}), ("scheme", "With its scheme", {}),
                                      ("host", "Appearance host", {"appearance": "host"}),
                                      ("fixed", "Appearance fixed", {"appearance": "fixed"}),
                                      ("bare", "Unstyled control", {})):
                api.register_ui_tab(tab, title, render={"kind": "module", "entry": f"{tab}.js", "start": "auto",
                                                         "span": 2, **extra})
        """), encoding="utf-8")
    for tab in ("plain", "host", "fixed"):
        (skill_dir / f"{tab}.js").write_text(_CALC.format(extra=""), encoding="utf-8")
    (skill_dir / "scheme.js").write_text(_CALC.format(extra=":root[data-theme=dark]{color-scheme:dark}"), encoding="utf-8")
    (skill_dir / "bare.js").write_text(_BARE, encoding="utf-8")
    save_review_state(data_dir, name, SkillReviewState(status="pass", content_hash=compute_content_hash(skill_dir, manifest_entry="plugin.py")))
    return name


@pytest.mark.parametrize("engine", ("chromium", "webkit"))
def test_the_board_frame_adds_no_fill_around_a_narrow_widget(direct_server_with_data, engine):
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from io import BytesIO

    from PIL import Image
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    probe = _write_calculators(data_dir)
    shots = evidence_dir(data_dir)
    findings = {}

    def sample(page, tab: str) -> dict:
        selector = f'[data-widget-key="{probe}:{tab}"]'
        page.evaluate("(selector) => document.querySelector(selector).scrollIntoView({block: 'center'})", selector)
        page.wait_for_timeout(150)
        frame = page.locator(f"{selector} iframe").bounding_box()
        head = page.locator(f"{selector} .widgets-card-head").bounding_box()
        around = page.screenshot(clip={"x": frame["x"] + frame["width"] - 10, "y": frame["y"] + 120, "width": 4, "height": 4})
        card = page.screenshot(clip={"x": head["x"] + head["width"] / 2, "y": head["y"] + 2, "width": 4, "height": 4})
        return {"around": Image.open(BytesIO(around)).convert("RGB").getpixel((2, 2)),
                "card": Image.open(BytesIO(card)).convert("RGB").getpixel((2, 2))}

    def near(a, b) -> bool:
        return max(abs(x - y) for x, y in zip(a, b)) <= 12

    with sync_playwright() as pw:
        browser, context = launch(pw, engine, viewport={"width": 1440, "height": 900})
        page = context.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            page.evaluate("""async (skill) => (await fetch(`/api/skills/${encodeURIComponent(skill)}/toggle`, {
                method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({enabled: true}),
            })).status""", probe)
            page.goto(f"{url}#widgets", wait_until="domcontentloaded", timeout=30_000)
            for theme in ("dark", "light"):
                page.evaluate("(theme) => window.ouroTheme?.set?.(theme)", theme)
                page.reload(wait_until="domcontentloaded", timeout=30_000)
                for tab in _PROBE_TABS:
                    page.wait_for_function(
                        "(selector) => document.querySelector(`${selector} iframe`)?.dataset.widgetContentHeight",
                        arg=f'[data-widget-key="{probe}:{tab}"]', timeout=20_000)
                page.wait_for_timeout(500)
                # The product: no fill on the frame, no colour scheme set by the host on any frame.
                frames = page.evaluate("""(skill) => [...document.querySelectorAll(`[data-widget-key^="${skill}:"] iframe`)]
                    .map((frame) => ({ fill: getComputedStyle(frame).backgroundColor, scheme: frame.style.colorScheme,
                        inherited: getComputedStyle(frame).colorScheme }))""", probe)
                assert all(item["fill"] in ("rgba(0, 0, 0, 0)", "transparent") for item in frames), frames
                assert all(item["scheme"] == "" for item in frames), "the host sets no colour scheme on a frame"
                findings[theme] = {"frames": frames, "scheme": sample(page, "scheme")}
                page.screenshot(path=str(shots / f"{engine}-canvas-{theme}.png"))
                # A module that declares the scheme it paints sits on the card's own surface in either theme.
                scheme = findings[theme]["scheme"]
                assert near(scheme["around"], scheme["card"]), (theme, findings[theme])
        finally:
            (shots / f"{engine}-canvas-findings.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")
            context.close()
            browser.close()


# --- the board at three sizes ----------------------------------------------------------------------------------------
# Two arrangements written through the preferences API: one with most cards full width and one mixed (small cards
# beside each other, the long ones bounded). Neither is anyone's real preference: the stand is isolated.

# --- a framed card whose module could not load says why, inside the card's inset -------------------------------------

@pytest.mark.parametrize("engine", ("chromium", "webkit"))
def test_a_framed_card_whose_module_failed_to_load_says_so_inside_the_card_inset(direct_server_with_data, engine):
    """A framed card's body is the frame, with no inset of its own; when its module fails to load, the words that say so
    take the host-drawn body's inset: aligned with the title, clear of the card's edges."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    _write_skill(data_dir)
    shots = evidence_dir(data_dir)
    geometry = {}
    with sync_playwright() as pw:
        browser, context = launch(pw, engine, viewport={"width": 1440, "height": 1000})
        page = context.new_page()
        try:
            page.route("**/module/short.js*", lambda route: route.fulfill(status=500, body="module unavailable"))
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            enable(page)
            page.click('[data-nav-page="widgets"]')
            page.locator(f"{card('short')} .skills-load-error").wait_for(state="visible", timeout=30_000)
            page.locator(card("short")).scroll_into_view_if_needed()
            page.wait_for_timeout(300)
            geometry = page.evaluate("""(selector) => {
                const box = (node) => node.getBoundingClientRect();
                const card = box(document.querySelector(selector));
                const title = box(document.querySelector(`${selector} .widgets-card-title`));
                const error = document.querySelector(`${selector} .skills-load-error`);
                return { left: box(error).left - card.left, title: title.left - card.left, right: card.right - box(error).right,
                         bottom: card.bottom - box(error).bottom, text: error.textContent.trim() };
            }""", card("short"))
            page.locator(card("short")).screenshot(path=str(shots / f"{engine}-module-failed.png"))
            assert "module load failed" in geometry["text"], geometry
            assert abs(geometry["left"] - geometry["title"]) <= 1 and geometry["left"] >= 12, geometry
            assert geometry["right"] >= 12 and geometry["bottom"] >= 12, geometry
        finally:
            (shots / f"{engine}-module-failed.json").write_text(json.dumps(geometry, indent=2), encoding="utf-8")
            context.close()
            browser.close()


LAYOUTS = {
    "wide-cards": {
        "widget_order": ["capped", "long", "fixed", "table", "short", "route", "bridged", "scrollbar", "note"],
        "widget_size": {tab: {"w": 12, "h": 0} for tab in ("capped", "long", "fixed", "table", "scrollbar")},
    },
    "mixed": {
        "widget_order": ["capped", "short", "note", "long", "fixed", "table", "route", "bridged", "scrollbar"],
        "widget_size": {"long": {"w": 0, "h": 360}, "table": {"w": 2, "h": 420}, "fixed": {"w": 2, "h": 480},
                        "route": {"w": 1, "h": 360}, "scrollbar": {"w": 0, "h": 300}},
    },
}

GEOMETRY = """() => {
    const list = document.getElementById('widgets-list');
    const cards = [...list.querySelectorAll('[data-widget-key]:not([data-widget-removed])')].map((node) => {
        const box = node.getBoundingClientRect();
        const body = node.querySelector('[data-widget-mount]');
        const title = node.querySelector('.widgets-card-title strong');
        const tools = node.querySelector('.widgets-card-tools').getBoundingClientRect();
        return { key: node.dataset.widgetKey, left: box.left, top: box.top, right: box.right, bottom: box.bottom,
            framed: Boolean(node.querySelector('iframe')),
            bodyScrolls: body.scrollHeight > body.clientHeight + 1,
            bounded: body.hasAttribute('data-widget-bounded'),
            titleFits: title.scrollWidth <= title.clientWidth + 1,
            toolsInside: tools.right <= box.right + 0.5 && tools.left >= box.left - 0.5 };
    });
    const scroller = document.querySelector('.widgets-scroll');
    return { cards, overflowX: document.scrollingElement.scrollWidth - window.innerWidth,
        scrollerOverflowX: scroller.scrollWidth - scroller.clientWidth, layout: list.dataset.widgetLayout };
}"""


@pytest.mark.parametrize("engine", ("chromium", "webkit"))
def test_board_views_at_three_sizes_in_both_themes(direct_server_with_data, engine):
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    _write_skill(data_dir)
    shots = evidence_dir(data_dir)
    report = {}

    with sync_playwright() as pw:
        browser, context = launch(pw, engine, viewport={"width": 1100, "height": 750})
        page = context.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            enable(page)
            # Opened at #widgets and reloaded: a phone's sidebar is a drawer, so no nav row is pressed.
            page.goto(f"{url}#widgets", wait_until="domcontentloaded", timeout=30_000)
            for name, layout in LAYOUTS.items():
                body = {"widget_order": [f"{SKILL}:{tab}" for tab in layout["widget_order"]],
                        "widget_size": {f"{SKILL}:{tab}": size for tab, size in layout["widget_size"].items()}}
                body["widget_size"] = {**{f"{SKILL}:{tab}": None for tab in ("short", "long", "capped", "fixed", "route",
                                                                              "bridged", "scrollbar", "table", "note")},
                                       **body["widget_size"]}
                written = page.evaluate("""async (body) => (await fetch('/api/ui/preferences', {method: 'POST',
                    headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)})).status""", body)
                assert written == 200
                for theme in ("dark", "light"):
                    for width, height in ((1100, 750), (1920, 1080), (390, 844)):
                        page.set_viewport_size({"width": width, "height": height})
                        page.evaluate("(theme) => window.ouroTheme?.set?.(theme)", theme)
                        page.reload(wait_until="domcontentloaded", timeout=30_000)
                        open_widgets(page, click=False)
                        page.wait_for_timeout(400)
                        geometry = page.evaluate(GEOMETRY)
                        label = f"{engine}-{name}-{width}x{height}-{theme}"
                        report[label] = geometry
                        page.screenshot(path=str(shots / f"{label}.png"))
                        page.screenshot(path=str(shots / f"{label}-full.png"), full_page=True)
                        assert geometry["overflowX"] <= 0 and geometry["scrollerOverflowX"] <= 0, (label, geometry)
                        boxes = geometry["cards"]
                        for i, a in enumerate(boxes):
                            for b in boxes[i + 1:]:
                                overlap = min(a["right"], b["right"]) - max(a["left"], b["left"]) > 1 and \
                                    min(a["bottom"], b["bottom"]) - max(a["top"], b["top"]) > 1
                                assert not overlap, (label, a["key"], b["key"])
                            assert not (a["framed"] and a["bodyScrolls"]), ("a framed card's body scrolls", label, a)
                            assert a["bodyScrolls"] == a["bounded"] or not a["bodyScrolls"], ("an unbounded body scrolls", label, a)
                            assert a["toolsInside"], ("the head's tools leave the card", label, a)
                        assert geometry["layout"] == ("stack" if width == 390 else "columns"), (label, geometry["layout"])
        finally:
            (shots / f"{engine}-views.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
            context.close()
            browser.close()
