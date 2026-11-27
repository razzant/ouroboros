"""A retained desktop DOM must not make the next shell stylesheet squeeze Main or trap the sidebar's wheel.

The former shell had an in-flow #divider before #content. Drive the real CSS
through Chromium against that old document shape and the current one; no server
or stored chat data is needed for this geometry regression.

It also wrapped the sidebar's rows in .sidebar-scroll. The real SPA with 30 Projects
runs in the literal v7.7.0 document (tests/fixtures/nminus1/index_v7.7.0.html, blob
113445a8 of v7.7.0:web/index.html), in the current zones inside that wrapper, in the
document of the foot's two cards (v7.12.0: tests/fixtures/nminus1/index_v7.12.0.html, blob
27bf94b8 of its web/index.html, the drawer's × in its head), in the one before them (v7.7.1 to
v7.11: the version row already in .sidebar-foot) and in a fresh document, with browser wheel
input in WebKit and Chromium; every one composes the same rail (owner 24.11).
"""

from pathlib import Path

import pytest

from tests import test_design_system_gate_browser as gate
from tests.test_subscription_setup_browser import subscription_ui  # noqa: F401

gate_ui, seed_page, settle = gate.gate_ui, gate.seed_page, gate.settle
WEB = Path(__file__).resolve().parents[1] / "web"
V7_7_0 = Path(__file__).resolve().parent / "fixtures" / "nminus1" / "index_v7.7.0.html"
V7_12_0 = Path(__file__).resolve().parent / "fixtures" / "nminus1" / "index_v7.12.0.html"


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
# The middle of the list's visible rows: inside the viewport and, in a short window, under the pinned head (the rail stands
# beside the list, never over it).
POINTER = """() => {
    const list = document.getElementById('nav-projects-list'), r = list.getBoundingClientRect();
    const head = document.querySelector('#primary-sidebar .sidebar-head');
    const pinned = (e) => e && getComputedStyle(e).position === 'sticky';
    const top = Math.max(r.top, 0, pinned(head) ? head.getBoundingClientRect().bottom : 0);
    const bottom = Math.min(r.bottom, innerHeight);
    const x = r.left + r.width / 2, y = (top + bottom) / 2;
    return {x, y, room: bottom - top, over: list.contains(document.elementFromPoint(x, y))};
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
    if shape in ("v7.12.0", "previous"):  # the foot of the two cards, its × in the drawer's head; before it (v7.7.1 to
        old = V7_12_0.read_text(encoding="utf-8")  # v7.11) the foot already held the version row
        if shape == "v7.12.0":
            return old
        foot = '<div class="sidebar-foot">'
        assert old.count(foot) == 1
        return old.replace(foot, foot + '\n<div class="sidebar-foot-meta">\n<span id="nav-version"></span>\n</div>')
    return html


# The version's anchors: how many #nav-version and version rows, and where the one version stands.
ANCHORS = """() => ({versions: document.querySelectorAll('#nav-version').length,
    rows: document.querySelectorAll('.sidebar-foot-meta').length,
    inRail: Boolean(document.querySelector('.nav-rail-system > .sidebar-foot-meta #nav-version')),
    inColumn: Boolean(document.querySelector('#primary-sidebar > .sidebar-foot-meta #nav-version'))})"""


def _anchors(shape):
    """One version in one row in every supported document, standing in the column's own foot under the Projects list
    (owner 25.11): the rail also mounts into a retained v7.7.0 document and adopts its lone #nav-version into that
    foot, so no shape keeps a second, plain version beside it."""
    return {"versions": 1, "rows": 1, "inRail": False, "inColumn": True}


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
@pytest.mark.parametrize("width,height,owner", [(1280, 928, "list"), (1280, 700, "list"), (1280, 540, "list"),
                                                (1280, 480, "sidebar"), (640, 800, "list")],
                         ids=["928", "700", "540", "short-480", "drawer-640"])
@pytest.mark.parametrize("shape", ["v7.7.0", "wrapped", "previous", "v7.12.0", "fresh"])
def test_retained_sidebar_wheel_scrolls_projects_between_main_and_the_tools(gate_ui, shape, width, height, owner):
    """The Projects list is the one scroller between Main and the tools (DESIGN "Shell"), in a retained document too:
    the wheel over it moves the list alone, Main and the tools stay reachable, the last Project comes wholly into view
    and the wheel brings the list back. A 540 px desktop window keeps the docked zones; a short window (480) keeps the
    whole-column fallback (no sticky Main in the old markup). Without `#primary-sidebar > .sidebar-scroll { display:
    contents; }` Chromium moves nothing there."""
    page = seed_page(gate_ui, width, height, touch=False)
    document = _document(shape)
    page.route(gate_ui["url"] + "/", lambda route: route.fulfill(content_type="text/html", body=document))
    try:
        page.goto(gate_ui["url"] + "/")
        page.wait_for_selector("#nav-projects-list [data-project-id]")
        assert page.locator("#primary-sidebar > .sidebar-scroll").count() == (shape in ("v7.7.0", "wrapped"))
        assert page.evaluate(ANCHORS) == _anchors(shape), page.evaluate(ANCHORS)
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


def _serve(body):
    return lambda route: route.fulfill(content_type="text/html", body=body)


RAIL_BOX = """() => { const rail = document.querySelector('.nav-rail');
    return {height: rail.getBoundingClientRect().height, width: rail.getBoundingClientRect().width,
            version: document.getElementById('nav-version').textContent,
            destinations: [...rail.querySelectorAll('[data-nav-page]')].map((row) => row.dataset.navPage)}; }"""

# The version is the update check on a checkable installation: once the status read says the install offers the
# check, the version stands inside the visible check control — not plain text beside it.
ACTIONABLE = """() => { const check = document.getElementById('nav-version-check'), v = document.getElementById('nav-version');
    return Boolean(check && !check.hidden && check.contains(v)); }"""

# The rail's one place per destination, in registry order (pages.js): the three sections, then the two system rows;
# Extensions opens through Settings (owner 24.11).
DESTINATIONS = ["circles", "files", "widgets", "dashboard", "settings"]

# The same set on the entire sidebar, not only the rail: the rail (with the drawer's × first) leads in DOM order ahead
# of Main's row and the Projects, with no duplicate destination left standing in the old rows (v7.7.0's Utilities
# group), no foot or drawer head left behind, one bound #nav-drawer-close capping the rail in every shape (reused in
# newer retained HTML, synthesized in v7.7.0), and the old shapes keeping their .sidebar-scroll wrapper.
SIDEBAR_BOX = """() => {
    const close = document.querySelector('#primary-sidebar .nav-rail > #nav-drawer-close:first-child');
    const main = document.querySelector('#primary-sidebar [data-nav-page="chat"]');
    const projects = document.getElementById('nav-projects');
    const follows = (a, b) => Boolean(a && b && (a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING));
    return {sidebar: [...document.querySelectorAll('#primary-sidebar [data-nav-page]')].map((row) => row.dataset.navPage),
        utilities: document.querySelectorAll('#primary-sidebar .nav-utilities').length,
        leftovers: document.querySelectorAll('#primary-sidebar :is(.sidebar-foot, .drawer-head, .nav-card)').length,
        close: Boolean(close),
        closeCount: document.querySelectorAll('#nav-drawer-close').length,
        closeFirst: follows(close, main) && follows(close, projects),
        closeIcon: Boolean(close?.querySelector('svg')),
        wrapper: document.querySelectorAll('#primary-sidebar > .sidebar-scroll').length,
        // Find a project and + lead the one head, then Main: the same order for reading and Tab in every shape.
        head: [...(document.querySelector('#primary-sidebar .sidebar-head')?.children || [])].map((el) => el.classList[0]).slice(0, 2),
        heads: document.querySelectorAll('#primary-sidebar .sidebar-head').length,
        findFirst: follows(document.querySelector('#primary-sidebar .nav-find-input'), main),
        headTop: (() => { const f = document.querySelector('#primary-sidebar .nav-projects-find')?.getBoundingClientRect(),
            m = main.getBoundingClientRect(); return f && f.bottom <= m.top; })()};
}"""


@pytest.mark.ui_browser
@pytest.mark.parametrize("width,height", [(1280, 928), (390, 844)], ids=["docked", "drawer"])
def test_a_retained_document_gets_the_same_rail_as_a_fresh_one(gate_ui, width, height):
    """An index.html kept across an update holds the foot the rail replaces (v7.12.0, its × in the drawer's head; before
    it the version row already in the foot), and the v7.7.0 document holds a lone #nav-version beside its old rows: the
    rail leads #primary-sidebar in DOM order and adopts the version, so every shape's rail has the fresh document's
    destinations, size and one written version, and the version is the check — not a second, empty version row. One
    #nav-drawer-close caps the rail first in DOM order (reused on newer retained shapes, synthesized on v7.7.0) and
    actually closes the phone drawer via app.js's existing binding; v7.7.0's own Utilities group leaves once the rail
    stands, and the wrapped shape keeps its .sidebar-scroll wrapper."""
    page = seed_page(gate_ui, width, height, touch=width < 720)
    page.route("**/api/health*", lambda route: route.fulfill(json={"ok": True, "version": "7.12.0", "runtime_version": "7.12.0"}))
    page.route("**/api/update/status*", lambda route: route.fulfill(json={"managed": True, "available": False}))
    feet = {}
    try:
        for shape in ("fresh", "previous", "v7.12.0", "v7.7.0", "wrapped"):
            document = _document(shape)
            page.route(gate_ui["url"] + "/", _serve(document))
            page.goto(gate_ui["url"] + "/")
            page.wait_for_selector("#nav-projects-list [data-project-id]")
            page.wait_for_function("document.getElementById('nav-version').textContent === 'v7.12.0'")
            page.wait_for_function(ACTIONABLE)
            if width < 720:
                page.locator("[data-mobile-nav-toggle]:visible").first.click()
                page.wait_for_function("() => document.getElementById('primary-sidebar').getBoundingClientRect().left === 0")
            settle(page)
            measured = {**page.evaluate(RAIL_BOX), **page.evaluate(ANCHORS), **page.evaluate(SIDEBAR_BOX)}
            if width < 720:
                page.locator("#nav-drawer-close").click()
                page.wait_for_function(
                    "() => !document.getElementById('primary-sidebar').classList.contains('open')"
                    " && window.__ouroLayers.size() === 0"
                )
                measured["closed"] = True
            feet[shape] = measured
            page.unroute(gate_ui["url"] + "/")
    finally:
        page.close()
    fresh = feet["fresh"]
    for shape, rail in feet.items():
        assert rail["versions"] == 1 and rail["rows"] == 1 and rail["inColumn"] and not rail["inRail"], feet
        assert rail["destinations"] == DESTINATIONS and len(set(rail["destinations"])) == 5, feet
        assert rail["version"] == "v7.12.0", feet
        assert rail["sidebar"] == DESTINATIONS + ["chat"], ("one set of destinations on the entire sidebar, rail first", feet)
        assert rail["utilities"] == 0 and rail["leftovers"] == 0 and rail["wrapper"] == (shape in ("v7.7.0", "wrapped")), feet
        assert rail["close"] and rail["closeCount"] == 1 and rail["closeFirst"] and rail["closeIcon"], (
            "one bound #nav-drawer-close caps the rail first in DOM order in every shape", feet)
        assert rail["heads"] == 1 and rail["head"] == ["nav-projects-header", "nav-row"], ("the find row leads the head", shape, feet)
        assert rail["findFirst"] and rail["headTop"], ("the field reads and stands before Main", shape, feet)
        if width < 720:
            assert rail.get("closed") is True, ("clicking #nav-drawer-close closes the drawer", shape, feet)
    for shape in ("previous", "v7.12.0", "v7.7.0", "wrapped"):
        assert abs(feet[shape]["height"] - fresh["height"]) <= 0.5 and feet[shape]["width"] == fresh["width"], ("the same rail", feet)


# The macOS window's measured band, as the shell writes it (launcher_window_chrome.py): 66 px bar, lights to 79, centred at 25.5.
INTEGRATED_BAND = """() => { const r = document.documentElement; r.dataset.windowChrome = 'integrated';
    r.style.setProperty('--window-chrome-top', '66px'); r.style.setProperty('--window-chrome-start', '79px');
    r.style.setProperty('--window-chrome-mid', '25.5px'); }"""
FIND_BOX = """() => { const f = document.querySelector('#primary-sidebar .nav-projects-find').getBoundingClientRect();
    return { left: f.left, middle: (f.top + f.bottom) / 2 }; }"""


@pytest.mark.ui_browser
@pytest.mark.parametrize("shape", ["fresh", "wrapped", "v7.7.0"])
def test_a_retained_documents_find_row_stands_on_the_lights_line_in_the_mac_window(gate_ui, shape):
    """In the macOS window Find a project stands on the traffic lights' line past them in every shape, the zones in the
    old .sidebar-scroll wrapper and v7.7.0's made head included (style.css reaches the head through the wrapper)."""
    page = seed_page(gate_ui, 1230, 838, touch=False)
    page.route(gate_ui["url"] + "/", _serve(_document(shape)))
    try:
        page.goto(gate_ui["url"] + "/")
        page.wait_for_selector("#nav-projects-list [data-project-id]")
        page.evaluate(INTEGRATED_BAND)
        settle(page)
        find = page.evaluate(FIND_BOX)
        assert find["middle"] == pytest.approx(25.5, abs=0.6) and find["left"] >= 79 + 12 - 0.5, (shape, find)
    finally:
        page.close()
