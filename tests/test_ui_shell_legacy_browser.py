"""A retained desktop DOM must not make the next shell stylesheet squeeze Main or trap the sidebar's wheel.

The former shell had an in-flow #divider before #content. Drive the real CSS
through Chromium against that old document shape and the current one; no server
or stored chat data is needed for this geometry regression.

It also wrapped the sidebar's rows in .sidebar-scroll. The real SPA with 30 Projects
runs in the literal v7.7.0 document (tests/fixtures/nminus1/index_v7.7.0.html, blob
113445a8 of v7.7.0:web/index.html), in the current zones inside that wrapper and in a
fresh document, with browser wheel input in WebKit and Chromium.
"""

from pathlib import Path

import pytest

from tests import test_design_system_gate_browser as gate
from tests.test_subscription_setup_browser import subscription_ui  # noqa: F401

gate_ui, seed_page, settle = gate.gate_ui, gate.seed_page, gate.settle
WEB = Path(__file__).resolve().parents[1] / "web"
V7_7_0 = Path(__file__).resolve().parent / "fixtures" / "nminus1" / "index_v7.7.0.html"


@pytest.mark.ui_browser
def test_retained_divider_cannot_take_the_main_grid_track():
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    ui = (WEB / "ui.css").read_text(encoding="utf-8")
    composition = (WEB / "style.css").read_text(encoding="utf-8")
    for legacy in (False, True):
        shell = f"""<!doctype html><html data-layout="docked"><head>
            <style>{ui}</style><style>{composition}</style></head><body>
            <div id="app"><nav id="primary-sidebar">Main Chat</nav>
            {'<div id="divider"></div>' if legacy else ''}
            <main id="content">Main transcript</main>
            <div id="project-panel-backdrop" class="project-panel-backdrop" hidden></div>
            <aside id="project-panel" class="project-panel" hidden></aside></div>
            {'<div id="reconnect-overlay"><div class="spinner"></div>Reconnecting to Ouroboros...</div>' if legacy else ''}
            </body></html>"""
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            try:
                page = browser.new_page(viewport={"width": 1536, "height": 816})
                page.set_content(shell)
                measured = page.evaluate("""() => {
                    const sidebar = document.querySelector('#primary-sidebar').getBoundingClientRect();
                    const main = document.querySelector('#content').getBoundingClientRect();
                    const divider = document.querySelector('#divider');
                    const veil = document.querySelector('#reconnect-overlay');
                    return {left: main.left, right: main.right, width: main.width,
                        sidebarRight: sidebar.right,
                        dividerDisplay: divider && getComputedStyle(divider).display,
                        veilDisplay: veil && getComputedStyle(veil).display,
                        rootOverflow: document.documentElement.scrollHeight - innerHeight,
                        tracks: getComputedStyle(document.querySelector('#app')).gridTemplateColumns};
                }""")
                assert abs(measured["left"] - measured["sidebarRight"]) <= 1, measured
                assert measured["width"] >= 1000 and measured["right"] <= 1537, measured
                assert len(measured["tracks"].split()) == 2, measured
                assert measured["rootOverflow"] <= 0, ("nothing below the shell scrolls the page", measured)
                if legacy:
                    assert measured["dividerDisplay"] == "none", measured
                    assert measured["veilDisplay"] == "none", ("the old reconnect veil is never shown", measured)

                page.evaluate("""() => {
                    for (const id of ['project-panel-backdrop', 'project-panel']) {
                        const el = document.getElementById(id);
                        el.hidden = false;
                        el.classList.add('open');
                    }
                }""")
                panel = page.evaluate("""() => {
                    const main = document.querySelector('#content').getBoundingClientRect();
                    const panel = document.querySelector('#project-panel').getBoundingClientRect();
                    return {mainLeft: main.left, mainWidth: main.width,
                        panelLeft: panel.left, panelRight: panel.right};
                }""")
                assert panel["mainLeft"] == measured["left"] and panel["mainWidth"] == measured["width"], panel
                assert panel["panelLeft"] >= measured["sidebarRight"] and panel["panelRight"] <= 1537, panel

                page.evaluate("document.documentElement.dataset.layout = 'stack'")
                page.set_viewport_size({"width": 414, "height": 700})
                narrow = page.evaluate("""() => {
                    const main = document.querySelector('#content').getBoundingClientRect();
                    return {left: main.left, right: main.right, width: main.width};
                }""")
                assert narrow["left"] == 0 and 400 <= narrow["width"] <= 414, narrow
                layer = page.locator('#project-panel').bounding_box()
                assert layer and abs(layer['x']) <= 1 and layer['width'] >= 400, layer
            finally:
                browser.close()


LAST = "#nav-projects-list > .nav-project-item:not([hidden]) .nav-project-row"
PLACES = ('[data-nav-page="chat"]', '[data-nav-page="files"]', '[data-nav-page="settings"]')
# The wheel's owner is whichever of these moves.
OFFSETS = """() => ({sidebar: document.getElementById('primary-sidebar').scrollTop,
    list: document.getElementById('nav-projects-list').scrollTop, page: document.scrollingElement.scrollTop})"""
# Reached: inside the viewport and every clipping box, and the topmost element at its visible centre.
REACH = """selector => {
    const e = [...document.querySelectorAll(selector)].pop(), r = e.getBoundingClientRect();
    let top = Math.max(r.top, 0), bottom = Math.min(r.bottom, innerHeight);
    let left = Math.max(r.left, 0), right = Math.min(r.right, innerWidth);
    for (let p = e.parentElement; p; p = p.parentElement) {
        const s = getComputedStyle(p);
        if (s.display === 'contents' || (s.overflowX === 'visible' && s.overflowY === 'visible')) continue;
        const c = p.getBoundingClientRect();
        top = Math.max(top, c.top); bottom = Math.min(bottom, c.bottom);
        left = Math.max(left, c.left); right = Math.min(right, c.right);
    }
    if (bottom - top < 1 || right - left < 1) return {hit: false, whole: false};
    const at = document.elementFromPoint((left + right) / 2, (top + bottom) / 2);
    return {hit: Boolean(at) && e.contains(at), whole: bottom - top >= r.height - 0.5};
}"""
POINTER = """() => {
    const list = document.getElementById('nav-projects-list'), r = list.getBoundingClientRect();
    const x = r.left + r.width / 2, y = (Math.max(r.top, 0) + Math.min(r.bottom, innerHeight)) / 2;
    return {x, y, over: list.contains(document.elementFromPoint(x, y))};
}"""
MAIN = """() => {
    const main = document.getElementById('content').getBoundingClientRect();
    const side = document.getElementById('primary-sidebar').getBoundingClientRect();
    return {gap: main.left - side.right, right: main.right,
            tracks: getComputedStyle(document.getElementById('app')).gridTemplateColumns.split(' ').length};
}"""


def _document(shape):
    html = (WEB / "index.html").read_text(encoding="utf-8")
    if shape == "v7.7.0":
        return V7_7_0.read_text(encoding="utf-8")
    if shape == "wrapped":  # the current zones inside the old wrapper
        start = html.index(">", html.index('<nav id="primary-sidebar"')) + 1
        end = html.index("</nav>", start)
        return html[:start] + '<div class="sidebar-scroll">' + html[start:end] + "</div>" + html[end:]
    return html


def _rest(page, before):
    """A wheel step lands asynchronously: wait for it to start, or not, then for three still frames."""
    for _ in range(25):
        settle(page)
        now = page.evaluate(OFFSETS)
        if now != before:
            break
    else:
        return before
    still = 0
    for _ in range(120):
        settle(page)
        later = page.evaluate(OFFSETS)
        still, now = (still + 1 if later == now else 0), later
        if still == 3:
            break
    return now


def _wheel(page, dy):
    """Wheel steps over the pointer until one moves nothing; offsets and reach after every step."""
    steps = [{"offsets": page.evaluate(OFFSETS)}]
    for _ in range(20):
        page.mouse.wheel(0, dy)
        offsets = _rest(page, steps[-1]["offsets"])
        steps.append({"offsets": offsets, "last": page.evaluate(REACH, LAST),
                      "settings": page.evaluate(REACH, PLACES[-1])})
        if offsets == steps[-2]["offsets"]:
            return steps
    raise AssertionError(f"the wheel never came to rest: {steps}")


@pytest.mark.ui_browser
@pytest.mark.serial
@pytest.mark.parametrize("width,height,owner", [(1280, 928, "list"), (1280, 700, "list"), (1280, 500, "sidebar"),
                                                (640, 800, "list")], ids=["928", "700", "short-500", "drawer-640"])
@pytest.mark.parametrize("shape", ["v7.7.0", "wrapped", "fresh"])
def test_retained_sidebar_wheel_scrolls_projects_between_main_and_the_tools(gate_ui, shape, width, height, owner):
    """The Projects list is the one scroller between Main and the tools (DESIGN "Shell"), in a retained document too:
    the wheel over it moves the list alone, Main and the tools stay reachable, the last Project comes wholly into view
    and the wheel brings the list back. A short window keeps the whole-column fallback (no sticky Main in the old
    markup). Without `#primary-sidebar > .sidebar-scroll { display: contents; }` Chromium moves nothing there."""
    page = seed_page(gate_ui, width, height, touch=False)
    document = _document(shape)
    page.route(gate_ui["url"] + "/", lambda route: route.fulfill(content_type="text/html", body=document))
    try:
        page.goto(gate_ui["url"] + "/")
        page.wait_for_selector("#nav-projects-list [data-project-id]")
        assert page.locator("#primary-sidebar > .sidebar-scroll").count() == (shape != "fresh")
        if width < 720:
            page.locator("[data-mobile-nav-toggle]:visible").first.click()
            page.wait_for_function("() => document.getElementById('primary-sidebar').getBoundingClientRect().left === 0")
        else:
            main = page.evaluate(MAIN)
            assert abs(main["gap"]) <= 1 and main["right"] <= width + 1 and main["tracks"] == 2, main
        settle(page)
        before = {place: page.evaluate(REACH, place)["hit"] for place in PLACES}
        pointer = page.evaluate(POINTER)
        assert pointer["over"], pointer
        page.mouse.move(pointer["x"], pointer["y"])
        down = _wheel(page, 400)
        start, end = down[0]["offsets"], down[-1]["offsets"]
        moved = [key for key in start if end[key] != start[key]]
        assert moved == [owner], f"the wheel over the Projects moved {moved}, not the {owner}: {down}"
        assert any(step["last"]["hit"] and step["last"]["whole"] for step in down[1:]), down
        assert down[-1]["settings"]["hit"], down
        if owner == "list":
            assert all(before.values()), f"Main and the tools before any wheel: {before}"
            assert all(page.evaluate(REACH, place)["hit"] for place in PLACES), "Main and the tools after the wheel"
        up = _wheel(page, -1200)
        assert up[-1]["offsets"] == {"sidebar": 0, "list": 0, "page": 0}, up
        assert page.evaluate(REACH, PLACES[0])["hit"], up
    finally:
        page.close()
