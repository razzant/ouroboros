"""The integrated macOS title bar through the real SPA on the selected Playwright engine.

The launcher's document script (``launcher_window_chrome.DOCUMENT_SCRIPT``) runs here beside a
stand-in message port, and the test plays the shell's measured state into it, so what is checked is
the real page's side of the contract: it declares once with its painted palette, reserves nothing
until the state arrives (and nothing without a shell), keeps its controls clear of the traffic-light
area from the drawer layout through the 800-981 px window widths (the desktop window's minimum is 800)
and in the full-width project layout, turns empty band presses into drags while
its header buttons keep theirs and the document reader's name stays selectable text, gives no band
press to a transcript row scrolled under either room's header, and lets go
when the state goes. The numbers are a sample of a measured state; AppKit's real geometry and
gestures are the native qualification's.
"""
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from ouroboros.launcher_window_chrome import DOCUMENT_SCRIPT, HANDLER, INTEGRATED
from tests.test_subscription_setup_browser import capture
from tests.test_subscription_setup_browser import subscription_ui as _subscription_ui

subscription_ui = _subscription_ui  # re-exported pytest fixture (requested by name below)

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

TOP, START = 52, 73  # a unified bar and its traffic lights, as one native run measured them
LIGHTS = {"left": 19, "right": START, "top": 18, "bottom": 34}
PORT = """Object.defineProperty(window, 'webkit', {configurable: true, value: {messageHandlers: {%s: {
    postMessage: (message) => (window.__shellMessages = window.__shellMessages || []).push(message)}}}});""" % HANDLER


def shell_page(ui, width, height=722):
    page = ui["new_page"](viewport={"width": width, "height": height})
    page.add_init_script(PORT)
    page.add_init_script(DOCUMENT_SCRIPT)
    page.goto(ui["url"])
    page.wait_for_selector("#chat-input")
    return page


def apply(page, top=TOP, start=START):
    # Like the native shell, answer the document's latest declaration (including page-cache returns).
    page.evaluate("""([top, start]) => {
        const doc = window.__shellMessages.filter(message => message.type === 'declare').at(-1).doc;
        window.__ouroWindowChrome({doc, top, start});
    }""", [top, start])


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_shell_fixture_replies_to_the_restored_document(request, monkeypatch, engine):
    monkeypatch.setenv("OUROBOROS_UI_BROWSER_ENGINE", engine)
    ui = request.getfixturevalue("subscription_ui")
    page = shell_page(ui, 1100)
    page.evaluate("window.dispatchEvent(new PageTransitionEvent('pageshow', {persisted: true}))")
    assert page.evaluate("window.__shellMessages.filter(m => m.type === 'declare').at(-1).doc !== performance.timeOrigin")
    apply(page)
    assert page.evaluate("document.documentElement.dataset.windowChrome") == INTEGRATED
    apply(page, top=0, start=0)
    assert page.evaluate("document.documentElement.dataset.windowChrome || null") is None


def box(page, selector):
    return page.locator(selector).first.bounding_box()


def clear_of_lights(rect):
    return (rect["x"] >= LIGHTS["right"] or rect["x"] + rect["width"] <= LIGHTS["left"]
            or rect["y"] >= LIGHTS["bottom"] or rect["y"] + rect["height"] <= LIGHTS["top"])


# Every visible control the owner can press, with where it is; hidden and zero-size nodes are not on screen.
CONTROLS = """() => [...document.querySelectorAll('a[href], button, input, select, textarea, summary, [role="button"], '
    + '[role="tab"], [tabindex]:not([tabindex="-1"])')].map((el) => [el, el.getBoundingClientRect()])
    .filter(([el, r]) => r.width > 0 && r.height > 0 && getComputedStyle(el).visibility !== 'hidden')
    .map(([el, r]) => ({ name: el.id || el.getAttribute('aria-label') || el.textContent.trim().slice(0, 24) || el.tagName,
                         x: r.x, y: r.y, width: r.width, height: r.height }))"""
LAYOUT = """() => {
    const sidebar = document.getElementById('primary-sidebar');
    const first = document.querySelector('[data-nav-page="chat"]').getBoundingClientRect();
    const toggle = document.querySelector('#page-chat [data-mobile-nav-toggle]');  // a room's ☰: its composer's first control
    const handle = document.querySelector('.resize-handle');
    // The sidebar's top inset lives on its pinned head (#1998: three zones), the band adds to it.
    return { sidebar_position: getComputedStyle(sidebar).position,
             sidebar_padding_top: getComputedStyle(sidebar.querySelector('.sidebar-head')).paddingTop,
             sidebar_width: sidebar.getBoundingClientRect().width, first_row: { x: first.x, y: first.y, width: first.width, height: first.height },
             drawer_toggle: toggle ? getComputedStyle(toggle).display : 'absent',
             resize_handle: handle ? getComputedStyle(handle).display : 'absent' };
}"""
# The app's own way into a Project room (a Project reference raises it).
OPEN_PANEL = """() => window.dispatchEvent(new CustomEvent('ouro:open-project', {
    detail: { project: { id: 'window-chrome', name: 'A project', chat_id: 101, lifecycle: 'active' } } }))"""
# The real document reader on a delivered copy held inline, as a live chat file card hands it over.
OPEN_READER = """async (name) => {
    const { createDocumentReader } = await import('/static/modules/document_reader.js');
    const text = 'Delivered notes\\n';
    window.__reader = createDocumentReader({ actions: { open() {}, download() {} } });
    window.__reader.open({ filename: name, meta: `TXT · ${text.length} B`, kind: 'text', size: text.length,
                           canOpen: true, reader: { base64: btoa(text) } });
}"""


def record(name, facts):
    root = os.environ.get("OUROBOROS_UI_EVIDENCE_DIR")
    if root:
        Path(root).mkdir(parents=True, exist_ok=True)
        (Path(root) / f"{name}.json").write_text(json.dumps(facts, indent=1), encoding="utf-8")


def drags(page):
    return [m["type"] for m in page.evaluate("window.__shellMessages || []") if m["type"] in ("drag", "double-click")]


def seed_history(root, rows=40, chat_id=1):
    """Ouroboros rows that each carry a link (and, like every row, a reply button): more than the column holds."""
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "logs" / "chat.jsonl").write_text("".join(json.dumps({
        "ts": f"2026-10-17T09:00:{index:02d}Z", "chat_id": chat_id, "direction": "out",
        "text": f"Note {index}: [the delivered report {index}](https://example.com/report-{index}) and a few words after it.",
    }) + "\n" for index in range(rows)), encoding="utf-8")


def serve_history(page, root):
    """Main's history from the real endpoint over ``root``; Playwright runs its routes in its own loop."""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway.history import make_chat_history_endpoint

    client = TestClient(Starlette(routes=[Route("/api/chat/history", make_chat_history_endpoint(root))]))

    def answer(route):
        url = urlsplit(route.request.url)
        result = client.get(f"{url.path}?{url.query}")
        route.fulfill(status=result.status_code, content_type="application/json", body=result.content)

    page.route("**/api/chat/history**", answer)


GUTTER = 6  # the global ::-webkit-scrollbar width: the band leaves the feed its scrollbar
# Every click a transcript control receives; a link's navigation is held back, so the page stays.
TRANSCRIPT_CLICKS = """() => { window.__transcriptClicks = []; document.addEventListener('click', (event) => {
    const control = event.target instanceof Element ? event.target.closest('a[href], button, summary, [role="button"]') : null;
    if (!control || !control.closest('#chat-messages')) return;
    window.__transcriptClicks.push(control.getAttribute('href') || control.className);
    if (control.matches('a[href]')) event.preventDefault();
}, true); }"""
# Band points over Main's column, short of the scrollbar gutter, where a press would reach the transcript.
BAND_REACH = """([top, gutter]) => {
    const feed = document.getElementById('chat-messages');
    const r = feed.getBoundingClientRect();
    const reached = [];
    for (let y = 0.5; y < top; y += 1)
        for (let x = r.left + 0.5; x < r.right - gutter; x += 4)
            if (feed.contains(document.elementFromPoint(x, y))) reached.push([x, y]);
    return reached;
}"""
# The last link above ``y``, scrolled back until its middle lies there, under the header.
UNDER_BAND = """(y) => {
    const feed = document.getElementById('chat-messages');
    const middle = (a) => { const r = a.getClientRects()[0]; return r.top + r.height / 2; };
    const link = [...feed.querySelectorAll('a[href^="https://example.com/"]')].filter((a) => middle(a) < y).pop();
    feed.scrollTop += middle(link) - y;
    return link.getAttribute('href');
}"""
# A link wholly below the header and its fade and above the composer: the transcript's own, below the band.
BELOW_BAND = """() => {
    const below = document.querySelector('.chat-page-header').getBoundingClientRect().bottom + 16;
    const above = document.getElementById('chat-input-area').getBoundingClientRect().top;
    const link = [...document.querySelectorAll('#chat-messages a[href^="https://example.com/"]')]
        .find((a) => { const r = a.getBoundingClientRect(); return r.top >= below && r.bottom <= above; });
    return link.getAttribute('href');
}"""
POINT = """(href) => { const r = document.querySelector(`#chat-messages a[href="${href}"]`).getClientRects()[0];
    return { x: r.left + Math.min(r.width / 2, 40), y: r.top + r.height / 2 }; }"""
# Each visible control of Main's header (a closed menu's items keep boxes, off screen), and whether a press at its
# middle is its own.
HEADER_CONTROLS = """() => [...document.querySelectorAll('.chat-page-header :is(a[href], button, summary, [role="button"])')]
    .filter((el) => !el.closest('details:not([open]) > :not(summary)'))
    .map((el) => [el, el.getBoundingClientRect()]).filter(([, r]) => r.width > 0 && r.height > 0)
    .map(([el, r]) => { const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
        return { name: el.dataset.chatCommand || el.id || el.className, y: r.top, own: Boolean(hit && el.contains(hit)) }; })"""
GUTTER_FACTS = """(top) => { const feed = document.getElementById('chat-messages'); const r = feed.getBoundingClientRect();
    return { scrollbar: feed.offsetWidth - feed.clientWidth, edge: document.elementFromPoint(r.right - 1, top / 2) === feed }; }"""
CATCHER = "getComputedStyle(document.querySelector('.chat-page-header'), '::after').content"


def test_the_spa_declares_and_reserves_only_the_measured_band(subscription_ui):
    ui = subscription_ui
    page = shell_page(ui, 1100)
    declared = page.evaluate("window.__shellMessages")
    assert [m["type"] for m in declared] == ["declare"]
    assert declared[0]["chrome"] == INTEGRATED
    assert (declared[0]["choice"], declared[0]["theme"]) == (
        page.locator("html").get_attribute("data-theme-choice"), page.locator("html").get_attribute("data-theme"))
    sidebar_top = page.evaluate("getComputedStyle(document.querySelector('#primary-sidebar .sidebar-head')).paddingTop")
    assert sidebar_top == "16px" and page.locator("html").get_attribute("data-window-chrome") is None

    apply(page)
    assert page.locator("html").get_attribute("data-window-chrome") == INTEGRATED
    assert box(page, '[data-nav-page="chat"]')["y"] >= TOP, "the first sidebar row starts below the band"
    restart = box(page, '[data-chat-command="restart"]')
    assert restart["y"] < TOP and clear_of_lights(restart), "the header stays level with the lights, beside them"
    capture(page, "window-chrome-integrated-wide")

    page.mouse.click(150, TOP / 2)  # the sidebar's empty band
    page.mouse.click(restart["x"] + restart["width"] / 2, restart["y"] + restart["height"] / 2)
    page.mouse.click(150, TOP + 200)  # below the band: the page's own press
    page.mouse.dblclick(150, TOP / 2)
    assert drags(page) == ["drag", "drag", "double-click"]

    apply(page, top=0, start=0)  # full screen or the ordinary bar
    assert page.locator("html").get_attribute("data-window-chrome") is None
    assert page.evaluate("getComputedStyle(document.querySelector('#primary-sidebar .sidebar-head')).paddingTop") == "16px"
    page.mouse.click(150, TOP / 2)
    assert drags(page) == ["drag", "drag", "double-click"]


def test_narrow_layouts_start_past_the_traffic_lights(subscription_ui):
    ui = subscription_ui
    page = shell_page(ui, 600)  # the drawer layout
    apply(page)
    # A room's ☰ is its composer's first control, far below the band; the header's first item, the title, starts
    # past the lights.
    toggle = box(page, "#page-chat .chat-toolbar-row .mobile-nav-toggle")
    assert toggle and toggle["y"] >= TOP
    title = box(page, ".chat-page-header .app-page-title")
    assert title and clear_of_lights(title) and title["x"] >= START
    capture(page, "window-chrome-integrated-drawer")
    bar = page.evaluate("getComputedStyle(document.querySelector('.project-panel-bar')).paddingLeft")
    assert float(bar.removesuffix("px")) >= START, "a full-width project panel starts past the lights too"

    wide = shell_page(ui, 1360)
    apply(wide)
    bar = wide.evaluate("getComputedStyle(document.querySelector('.project-panel-bar')).paddingLeft")
    assert bar == "14px", "a panel docked right keeps its own inset"


def test_a_browser_without_the_shell_reserves_nothing(subscription_ui):
    page = subscription_ui["new_page"](viewport={"width": 1100, "height": 722})
    page.goto(subscription_ui["url"])
    page.wait_for_selector("#chat-input")
    assert page.locator("html").get_attribute("data-window-chrome") is None
    assert page.evaluate("getComputedStyle(document.querySelector('#primary-sidebar .sidebar-head')).paddingTop") == "16px"
    assert page.evaluate("window.__shellMessages") is None


def test_onboarding_keeps_a_draggable_band_and_the_frame_inside_the_window(subscription_ui):
    """The real overlay owns its frame geometry; the frame's content need not know the native shell."""
    page = shell_page(subscription_ui, 900, 600)
    page.route("**/api/onboarding", lambda route: route.fulfill(status=200, body="setup required"))
    page.route("**/onboarding", lambda route: route.fulfill(
        status=200, content_type="text/html", body='<button id="first">Appearance</button>'))
    page.evaluate("async () => { const m = await import('/static/modules/onboarding_overlay.js'); "
                  "window.__disposeSetup = await m.initOnboardingOverlay(); }")
    page.wait_for_selector(".onboarding-frame")
    apply(page)
    frame = box(page, ".onboarding-frame")
    assert frame["y"] == TOP
    assert frame["y"] + frame["height"] <= 600
    assert box(page, ".onboarding-overlay")["y"] == 0, "setup still blocks the entire underlying UI"
    page.mouse.click(150, TOP / 2)
    assert drags(page) == ["drag"], "the reserved strip is the main document, not the iframe"
    page.frame_locator(".onboarding-frame").locator("#first").focus()
    assert page.frame_locator(".onboarding-frame").locator("#first").evaluate("el => el === document.activeElement")
    apply(page, top=0, start=0)
    assert box(page, ".onboarding-frame")["y"] == 0, "full screen and ordinary chrome need no reserve"
    page.evaluate("window.__disposeSetup()")


def test_the_document_readers_name_in_the_band_stays_selectable_text(subscription_ui):
    """In the smallest window the reader's name lies in the band. It is the document's text, selectable
    as everywhere in the reader (DESIGN "Document reading"), so a press on it selects; the head's empty
    padding beside it is still the title bar's."""
    page = shell_page(subscription_ui, 800, 500)
    apply(page)
    name = "delivery-notes-for-the-owner.txt"
    page.evaluate(OPEN_READER, name)
    page.wait_for_selector("dialog.document-reader[open] .document-reader-source")
    title, head = box(page, ".document-reader-title"), box(page, ".document-reader-head")
    y = title["y"] + title["height"] / 2
    assert y < TOP and head["y"] + 4 < TOP, "the name and the head's top edge lie in the band at this window size"
    assert page.locator("html").get_attribute("data-window-chrome") == INTEGRATED

    page.mouse.click(head["x"] + 8, head["y"] + 4)  # the head's own padding, an empty part of the band
    assert drags(page) == ["drag"], "the band is the title bar over the reader too"
    page.mouse.move(title["x"] + 1, y)
    page.mouse.down()
    page.mouse.move(title["x"] + 90, y, steps=5)
    page.mouse.up()
    selected = page.evaluate("String(getSelection())")
    assert selected and selected in name, f"pressing the name selects it, as without the shell; posted {drags(page)}"
    page.mouse.dblclick(title["x"] + 20, y)
    selected_word = page.evaluate("String(getSelection())")
    assert selected_word and selected_word in name
    assert drags(page) == ["drag"], "the name is neither a window drag nor the title-bar double-click"
    assert page.locator(".document-reader-title").evaluate("el => getComputedStyle(el).cursor") == "text"
    capture(page, "window-chrome-800x500-document-reader")
    record("window-chrome-800x500-document-reader", {"title": title, "head": head, "selected": selected})
    page.evaluate("window.__reader.destroy()")


@pytest.mark.parametrize("width", [800, 900, 980, 981])
def test_from_the_minimum_window_width_every_top_control_clears_the_lights(subscription_ui, width):
    """The window is never narrower than 800 pt: up to 980 px the sidebar stays a column (the drawer is a
    640 px layout; 980 px only hides its resize handle), and at 900 px and below a project panel covers the
    whole window. In each, nothing the owner can press sits under the traffic lights, and the baseline
    without a shell shows why the sidebar reserve exists."""
    ui = subscription_ui
    plain = ui["new_page"](viewport={"width": width, "height": 600})
    plain.goto(ui["url"])
    plain.wait_for_selector("#chat-input")
    baseline = plain.evaluate(LAYOUT)
    assert baseline["sidebar_position"] != "fixed" and baseline["drawer_toggle"] == "none", "no drawer at this width"
    assert baseline["sidebar_padding_top"] == "16px"
    assert not clear_of_lights(baseline["first_row"]), "without the reserve the first row would sit under the lights"

    page = shell_page(ui, width, 600)
    apply(page)
    integrated = page.evaluate(LAYOUT)
    assert integrated["sidebar_position"] == baseline["sidebar_position"] and integrated["drawer_toggle"] == "none"
    assert integrated["first_row"]["y"] >= TOP
    blocked = [c for c in page.evaluate(CONTROLS) if not clear_of_lights(c)]
    assert blocked == []
    page.evaluate(OPEN_PANEL)
    page.wait_for_selector("#project-panel.open:not([hidden]) .chat-panel-statusbar")
    page.wait_for_timeout(400)  # the panel's open transition
    panel = page.evaluate("() => ({ left: document.getElementById('project-panel').getBoundingClientRect().x })")
    with_panel = [c for c in page.evaluate(CONTROLS) if not clear_of_lights(c)]
    assert with_panel == []
    assert clear_of_lights(box(page, "#project-panel-title")), "the panel's title starts past the lights too"
    capture(page, f"window-chrome-{width}-project-panel")
    record(f"window-chrome-width-{width}", {"baseline": baseline, "integrated": integrated, "panel_left": panel["left"],
                                             "controls_in_band": [c for c in page.evaluate(CONTROLS) if c["y"] < TOP]})


@pytest.mark.parametrize("width", [800, 1100, 1440])
@pytest.mark.parametrize("room", ["main", "project"])
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_band_presses_over_a_scrolled_transcript_are_the_title_bars(request, monkeypatch, tmp_path, width, room, engine):
    """A room opens at its newest message, so older rows lie under its header, which paints over them but lets a
    press between its groups through. In the band such a press must still be the title bar's: it drags and a
    double-click is the window's, and the link or button hidden under the header gets neither. The header's
    controls, the transcript below the band and the feed's scrollbar keep their presses; without the band
    (full screen, the ordinary bar) the header is as in a browser again."""
    monkeypatch.setenv("OUROBOROS_UI_BROWSER_ENGINE", engine)
    ui = request.getfixturevalue("subscription_ui")
    seed_history(tmp_path, chat_id=101 if room == "project" else 1)
    feed = "#project-panel .chat-messages" if room == "project" else "#chat-messages"
    header = ".project-panel-bar" if room == "project" else ".chat-page-header"

    def script(source):
        if room == "main":
            return source
        return (source.replace("document.getElementById('chat-messages')", "document.querySelector('#project-panel .chat-messages')")
                .replace("document.getElementById('chat-input-area')", "document.querySelector('#project-panel .chat-input-area')")
                .replace("#chat-messages", "#project-panel .chat-messages").replace(".chat-page-header", ".project-panel-bar"))

    name = f"window-chrome-{width}-{room}-{engine}"
    page = ui["new_page"](viewport={"width": width, "height": 722})
    serve_history(page, tmp_path)
    page.add_init_script(PORT)
    page.add_init_script(DOCUMENT_SCRIPT)
    page.goto(ui["url"])
    if room == "project":
        page.wait_for_selector("#chat-input")
        page.evaluate(OPEN_PANEL)
        page.wait_for_selector("#project-panel.open .rw-trigger")
        page.wait_for_timeout(400)
    page.wait_for_selector(f'{feed} a[href="https://example.com/report-39"]')
    page.wait_for_function(script("() => { const f = document.getElementById('chat-messages');"
                                  " return f.scrollTop > 0 && f.scrollHeight - f.clientHeight - f.scrollTop < 2; }"))
    page.evaluate(script(TRANSCRIPT_CLICKS))
    apply(page)
    facts = {"reached": page.evaluate(script(BAND_REACH), [TOP, GUTTER]), "catcher": page.evaluate(script(CATCHER))}
    hidden = page.evaluate(script(UNDER_BAND), 2)  # the band's top strip, above every header group
    capture(page, f"{name}-row-under-the-band")
    point = page.evaluate(script(POINT), hidden)
    page.mouse.click(point["x"], point["y"])
    facts["press"] = {"drags": drags(page), "clicks": page.evaluate("window.__transcriptClicks")}
    page.mouse.dblclick(point["x"], point["y"])
    facts["double"] = {"drags": drags(page), "clicks": page.evaluate("window.__transcriptClicks")}
    facts["header"] = page.evaluate(script(HEADER_CONTROLS))
    # Panic stands in every room's bar and takes a press there (a trial: nothing is stopped).
    page.locator(f'{header} [data-chat-command="panic"]').click(trial=True)
    if room == "project":
        page.locator("#project-panel-close").click(trial=True)
    facts["gutter"] = page.evaluate(script(GUTTER_FACTS), TOP)
    below = page.evaluate(script(BELOW_BAND))
    target = page.evaluate(script(POINT), below)
    page.mouse.click(target["x"], target["y"])
    facts["below"] = {"drags": drags(page), "clicks": page.evaluate("window.__transcriptClicks")}
    if room == "project":
        # Falsifier: removing only the new hit shield must expose a transcript target in the band.
        unshield = page.add_style_tag(content='.project-panel-bar::after { content: none !important; }')
        facts["unshielded"] = page.evaluate(script(BAND_REACH), [TOP, GUTTER])
        unshield.evaluate("el => el.remove()")
        assert facts["unshielded"], "the regression scene must contain transcript targets under the Project band"
    apply(page, top=0, start=0)
    facts["released"] = page.evaluate(script(CATCHER))
    record(f"{name}-band-over-transcript", {"hidden": hidden, "point": point, "below": below, **facts})

    reached = facts["reached"]
    assert not reached, f"{len(reached)} band points reach the transcript under the header, e.g. {reached[:4]}"
    assert facts["press"] == {"drags": ["drag"], "clicks": []}, f"a band press over the hidden {hidden}"
    assert facts["double"] == {"drags": ["drag", "drag", "double-click"], "clicks": []}
    assert all(control["own"] for control in facts["header"]), facts["header"]
    assert facts["gutter"]["edge"] and facts["gutter"]["scrollbar"] <= GUTTER, facts["gutter"]
    assert facts["below"] == {"drags": ["drag", "drag", "double-click"], "clicks": [below]}
    assert facts["catcher"] not in ("none", "normal") and facts["released"] in ("none", "normal")
