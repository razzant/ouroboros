"""Actual sidebar + Project gateway/storage; controlled non-project census, no owner state.

The existing shell server supplies assets and non-project APIs. Project list/update
requests exercise the real gateway and registry in a private TestClient drive.
Pin and Unpin are the first actions of a row's menu (its ⋯ under a mouse, a long press on touch) and the row's own
glyph at the end of line 1 for a hovering pointer or the keyboard; pinned rows lead the one keyed list with a quiet
resting pin glyph beside their shifted time, and one step of air parts them from the rest, with no caption or column.
Find a project and Show more reach every room beyond the feed; a reference opens its own room by its id.
"""
import json
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from ouroboros.gateway.router import collect_routes
from ouroboros import projects_registry as registry, room_messages
from tests.test_subscription_setup_browser import subscription_ui  # noqa: F401
from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401
from tests._shared import producer_evidence_dir
from tests._work_projection_shared import with_work_fields

pytestmark = pytest.mark.ui_browser

LONG = "Old project with a long descriptive title"
# Visible rows in order: their gap to the row above, whether they lead or follow the pins, their full names.
GROUPS = """() => {
    const items = [...document.querySelectorAll('#nav-projects-list > .nav-project-item:not([hidden])')];
    return items.map((item, i) => {
        const row = item.querySelector('.nav-project-row'), label = row.querySelector('.nav-row-label');
        return {id: item.dataset.navProjectId, pinned: item.hasAttribute('data-pinned'),
                after: item.hasAttribute('data-after-pins'), height: row.getBoundingClientRect().height,
                gap: i ? item.getBoundingClientRect().top - items[i - 1].getBoundingClientRect().bottom : null,
                title: row.title, name: row.getAttribute('aria-label'),
                solo: row.hasAttribute('data-solo'),
                ellipsized: label.scrollWidth > label.clientWidth,
                labelWidth: label.clientWidth, labelScroll: label.scrollWidth, labelText: label.textContent};
    });
}"""


@pytest.fixture(params=[("chromium", 1280, "dark"), ("webkit", 390, "light")],
                ids=["chromium-1280-dark", "webkit-390-light-touch"])
def pins_ui(request, monkeypatch, tmp_path):
    engine, width, theme = request.param
    monkeypatch.setenv("OUROBOROS_UI_BROWSER_ENGINE", engine)
    ui = request.getfixturevalue("subscription_ui")
    root = tmp_path / "pin-data"
    for pid, name in (("older", LONG), ("newer", "Recent project")):
        registry.create_project(root, pid, name=name, origin="owner_ui")
    registry.update_project(root, "older", last_active_at="2026-01-01T00:00:00Z")
    app = Starlette(routes=collect_routes(data_dir=root))
    app.state.drive_root = root
    control = {"fail": False, "hold": False, "held": [], "hold_writes": False, "held_writes": [],
               "writes": [], "unread": set(), "activities": [], "message_times": {},
               "fail_lookup": False, "hold_lookups": False,
               "held_lookups": [], "fail_list": False, "hold_lists": False, "held_lists": [],
               "shared_cursors": False, "null_project_count": False, "sent": [], "sockets": [],
               # A page made while these are set boots with that language payload, Main's revision and its read cursor.
               "i18n": None, "main_revision": None, "main_seen": None}
    pages = []
    with TestClient(app) as client:
        def forward(route, path, body):
            response = client.post(path, json=body)
            route.fulfill(status=response.status_code, content_type="application/json", body=response.text)

        def forward_get(route):
            url = urlparse(route.request.url)
            response = client.get(url.path + (f"?{url.query}" if url.query else ""))
            route.fulfill(status=response.status_code, content_type="application/json", body=response.text)

        def respond(route, language=None):
            path = urlparse(route.request.url).path
            if path == "/api/state":
                rows = registry.projects_summary(root, include_pinned=True, by_message=True)  # as gateway/state.py
                for project in rows:
                    if project['id'] in control['unread']:
                        project['visible_revision'] = 2
                    if project['id'] in control['message_times']:
                        project['last_message_at'] = control['message_times'][project['id']]
                # The sidebar's live marks read the server's projection of the census rows (its rooms).
                state = with_work_fields({"sha": "pin-fixture", "supervisor_ready": True,
                                          "active_chat_activities_complete": True,
                                          "active_chat_activities": control['activities'], "projects": rows,
                                          "project_messages_revision": room_messages.project_revision(root, registry.list_sidebar_projects(root)),
                                          "project_listable_count": None if control["null_project_count"]
                                          else len(registry.list_sidebar_projects(root)),
                                          # Every reserved room, as gateway/state.py answers: rooms beyond the feed too.
                                          "project_chat_ids": sorted(registry.reserved_project_chat_ids(root))})
                if control["main_revision"] is not None:
                    state["main_visible_revision"] = control["main_revision"]
                if control["hold"]:
                    control["held"].append((route, json.dumps(state)))
                    return
                route.fulfill(content_type="application/json", body=json.dumps(state))
            elif path.startswith("/api/projects/") and route.request.method == "POST":
                body = route.request.post_data_json
                control["writes"].append(body)
                if control["fail"]:
                    route.fulfill(status=503, content_type="application/json", body='{"error":"Synthetic write outage"}')
                elif control["hold_writes"]:
                    control["held_writes"].append((route, path, body))
                else:
                    forward(route, path, body)
            elif path == "/api/projects":
                query = parse_qs(urlparse(route.request.url).query)
                if "id" in query and control["fail_lookup"]:
                    route.fulfill(status=503, content_type="application/json", body='{"error":"Synthetic lookup outage"}')
                elif "id" in query and control["hold_lookups"]:
                    control["held_lookups"].append(route)
                elif query.get("all") == ["1"] and control["fail_list"]:
                    route.fulfill(status=503, content_type="application/json", body='{"error":"Synthetic list outage"}')
                elif query.get("all") == ["1"] and control["hold_lists"]:
                    response = client.get('/api/projects?all=1')
                    control["held_lists"].append((route, response.text))
                else:
                    forward_get(route)
            elif path in ("/api/ui/i18n", "/api/ui/i18n/missing") and language is not None:
                route.fulfill(json=language if path == "/api/ui/i18n" else {"ok": True})
            elif path == "/api/ui/preferences" and control["main_seen"] is not None and route.request.method == "GET":
                route.fulfill(json={"main_seen_revision": control["main_seen"]})
            elif path == "/api/ui/preferences" and control["shared_cursors"]:
                # The real shared read cursors, for a case where another client reads a room.
                if route.request.method == "POST":
                    forward(route, path, route.request.post_data_json)
                else:
                    forward_get(route)
            else:
                route.fallback()

        def socket(ws):
            # A quiet socket that keeps what the page sends (a composer's frame names its room).
            control["sockets"].append(ws)
            ws.on_message(lambda message: control["sent"].append(message))
            ws.send(json.dumps({"type": "heartbeat"}))

        def new_page(height=900, now=None, **options):
            """A page whose clock starts at ``now`` when given, before its first paint reads the time."""
            page = ui["new_page"](viewport={"width": width, "height": height}, has_touch=width < 720, **options)
            pages.append(page)
            page.add_init_script(f"localStorage.setItem('ouroboros.theme', {json.dumps(theme)})")
            # The language a page boots with stays its own: a later page's language never reaches it on a re-read.
            language = control["i18n"]
            page.route("**/api/**", lambda route: respond(route, language))
            page.route_web_socket("**/ws", socket)
            page.clock.install(**({"time": now} if now else {}))
            page.goto(ui["url"])
            page.locator('.nav-project-item').first.wait_for()
            open_drawer(page, width)
            return page

        page = new_page()
        try:
            yield {**ui, "page": page, "make_page": new_page, "root": root, "control": control, "forward": forward,
                   "forward_get": forward_get, "width": width, "engine": engine, "theme": theme, "client": client,
                   "tag": f"{engine}-{width}-{theme}"}
        finally:
            # Stop route callbacks and polling before their TestClient drive closes.
            for page in pages:
                page.close()


def open_drawer(page, width):
    if width < 720:
        page.locator('#page-chat [data-mobile-nav-toggle]').click()
        page.locator('#primary-sidebar.open').wait_for()


def expect():
    return pytest.importorskip("playwright.sync_api").expect


def row(page, pid):
    return page.locator(f'.nav-project-item[data-nav-project-id="{pid}"]')


def menu(page):
    return page.locator('body > .project-row-menu:not([data-leaving])')


def order(page):
    return page.locator('#nav-projects-list > *').evaluate_all('els => els.map(el => el.dataset.navProjectId)')


def refresh(page):
    page.clock.fast_forward(21_000)


def until(page, predicate, rounds=200):
    """Route callbacks run while Playwright is called; round-trip until one has recorded what the test needs."""
    for _ in range(rounds):
        if predicate():
            return
        page.evaluate("0")
    raise AssertionError("the route callback never recorded the request")


def coarse(page):
    return page.evaluate("matchMedia('(pointer: coarse)').matches")


def long_press(page, target):
    """A finger held still on ``target`` past the long press (row_menu_gesture.js), then lifted. The next press acts at
    once (gesture_tail.js): nothing waits for the lift."""
    box = target.bounding_box()
    finger = {"pointerType": "touch", "isPrimary": True, "pointerId": 41,
              "clientX": box["x"] + box["width"] / 2, "clientY": box["y"] + box["height"] / 2}
    target.dispatch_event("pointerdown", finger)
    page.clock.run_for(500)
    target.dispatch_event("pointerup", finger)


def open_menu(page, pid):
    """Open a row's menu as its device does: its ⋯ under a mouse, a long press on the row on touch, where the ⋯ takes
    no room. Returns the control the menu gives focus back to."""
    if coarse(page):
        long_press(page, row(page, pid).locator('.nav-project-row'))
        expect()(menu(page)).to_be_visible()
        settle_menu(page)
        return row(page, pid).locator('.nav-project-row')
    row(page, pid).locator('.nav-project-kebab').click()
    expect()(menu(page)).to_be_visible()
    settle_menu(page)
    return row(page, pid).locator('.nav-project-kebab')


def settle_menu(page):
    """A menu grows in from where it was opened (ui.css `ui-popup-in`): its box is read once it has."""
    menu(page).evaluate("m => Promise.all(m.getAnimations().map((a) => a.finished.catch(() => null)))")


def choose(page, pid, label):
    """Open a row's menu and press its first action, which must read ``label``."""
    open_menu(page, pid)
    item = menu(page).locator('[data-prm="pin"]')
    expect()(item).to_have_text(label)
    item.click()
    expect()(menu(page)).to_have_count(0)


def assert_one_boundary(groups):
    """Pins lead; the first row shown after them keeps one step of air (8px + the list's 1px gap), nothing else."""
    pinned = [g["pinned"] for g in groups]
    assert pinned == sorted(pinned, reverse=True), groups
    for index, group in enumerate(groups):
        boundary = index > 0 and groups[index - 1]["pinned"] and not group["pinned"]
        assert group["after"] is boundary, (group, groups)
        if index:
            assert group["gap"] == pytest.approx(9 if boundary else 1, abs=0.5), (group, groups)


# Each row's glyphs outside its Pin and ⋯ (none), and whether its Pin is drawn or a target; a Pin at rest has no
# pointer on its row and no keyboard on itself. On touch, a focused or open ⋯ owns the row's end and hides the Pin.
REST = """async () => {
    const { SHELL_ICONS } = await import('/static/modules/page_icons.js');
    const paths = (node) => [...node.querySelectorAll('path')].map((p) => p.getAttribute('d'));
    const parse = (html) => { const d = document.createElement('div'); d.innerHTML = html; return paths(d); };
    const pinPaths = parse(SHELL_ICONS.pin), unpinPaths = parse(SHELL_ICONS.unpin);
    const same = (a, b) => a.length > 0 && a.length === b.length && a.every((v, i) => v === b[i]);
    return [...document.querySelectorAll('#nav-projects-list > .nav-project-item')].map((item) => {
        const pin = item.querySelector(':scope > .nav-project-pin'), style = pin && getComputedStyle(pin);
        const kebab = item.querySelector(':scope > .nav-project-kebab');
        const shown = pin ? [...pin.querySelectorAll(':scope > svg')].filter((svg) => getComputedStyle(svg).display !== 'none') : [];
        const shownPaths = shown.length === 1 ? paths(shown[0]) : [];
        return {id: item.dataset.navProjectId, pinned: item.hasAttribute('data-pinned'),
                deleting: item.classList.contains('is-deleting'),
                resting: !item.matches(':hover') && !pin?.matches(':focus-visible'),
                kebabHidden: matchMedia('(pointer: coarse)').matches && Boolean(kebab?.matches('[data-focused], [aria-expanded="true"]')),
                stray: [...item.querySelectorAll('svg')].filter((svg) => !svg.closest('.nav-project-pin, .nav-project-kebab')).length,
                drawn: Boolean(style) && style.opacity !== '0' && style.visibility !== 'hidden',
                target: Boolean(style) && style.pointerEvents !== 'none',
                restGlyph: same(shownPaths, pinPaths),
                actGlyph: same(shownPaths, unpinPaths)};
    });
}"""


def assert_flat_and_unlabelled(page):
    """One flat keyed list: every child a Project row, the word Pinned only in names, and at rest only pinned rows
    draw their quiet pin glyph (hit-testable on a primary fine pointer; mark-only on coarse, never on unpinned rows)."""
    children = page.locator('#nav-projects-list > *').evaluate_all(
        'els => els.map(el => [el.className, el.dataset.navProjectId || ""])')
    assert children and all(cls.startswith('nav-project-item') and pid for cls, pid in children), children
    assert len({pid for _, pid in children}) == len(children)
    assert "Pinned" not in page.locator('#primary-sidebar').inner_text()
    fine = page.evaluate("matchMedia('(hover: hover) and (pointer: fine)').matches")
    for item in page.evaluate(REST):
        assert item["stray"] == 0, item
        if item["resting"]:
            if item["pinned"] and not item["deleting"] and not item["kebabHidden"]:
                assert item["drawn"] and item["restGlyph"] and not item["actGlyph"], item
                assert item["target"] is fine, item
            else:
                assert not item["drawn"] and not item["target"], item


def test_menu_pin_reorders_keyed_rows_without_opening_or_reading_across_reload_and_windows(pins_ui, tmp_path):
    ui, page = pins_ui, pins_ui["page"]
    evidence = producer_evidence_dir(tmp_path, "project-pin", env="OUROBOROS_BROWSER_EVIDENCE_OUT")
    # No pins: the ordinary list, no boundary; ⋯ offers Pin before Rename and Delete.
    assert order(page) == ["newer", "older"]
    assert page.locator('#nav-projects-list [data-pinned], #nav-projects-list [data-after-pins]').count() == 0
    page.evaluate("window.savedKebab = document.querySelector('[data-nav-project-id=older] .nav-project-kebab')")
    row(page, "older").locator('.nav-project-kebab').focus()
    page.keyboard.press("Enter")
    expect()(menu(page)).to_be_visible()
    assert menu(page).locator('[role=menuitem]').all_inner_texts() == ["Pin", "Rename…", "Working folder…", "Delete project…"]
    assert page.evaluate("document.activeElement.dataset.prm") == "pin"
    page.screenshot(path=str(evidence / f"menu-pin-{ui['tag']}.png"), animations="disabled")
    page.keyboard.press("Enter")
    expect()(menu(page)).to_have_count(0)
    expect()(row(page, "older")).to_have_attribute("data-pinned", "")
    assert order(page) == ["older", "newer"]
    # The same keyed node moved and kept the keyboard.
    assert page.evaluate("savedKebab === document.activeElement && savedKebab.isConnected")
    assert registry.get_project(ui["root"], "older")["pinned"] is True
    assert ui["control"]["writes"] == [{"pinned": True}]
    # Pinning neither opens nor reads the Project.
    assert page.locator('.nav-project-row.active').count() == 0
    assert page.locator('#project-panel').is_hidden()
    assert not [body for path, body in ui["posts"]
                if path == "/api/ui/preferences" and "project_seen_revision" in (body or {})]
    groups = page.evaluate(GROUPS)
    assert_one_boundary(groups)
    assert [g["id"] for g in groups] == ["older", "newer"]
    assert groups[0]["title"] == groups[0]["name"] == f"{LONG} · Pinned"
    assert groups[1]["title"] == "Recent project"
    assert_flat_and_unlabelled(page)
    page.locator('#primary-sidebar').screenshot(path=str(evidence / f"expanded-{ui['tag']}.png"), animations="disabled")
    # Reload reads the pin from the registry.
    ui["errors"].watch(page)  # the reload may cut off the page's own reads (tests/ui_console_errors.py)
    page.reload()
    page.locator('.nav-project-item').first.wait_for()
    open_drawer(page, ui["width"])
    expect()(row(page, "older")).to_have_attribute("data-pinned", "")
    assert order(page) == ["older", "newer"]
    # Another window unpins; this one follows at its next state read, and the last pin takes its air with it.
    second = ui["make_page"]()
    expect()(row(second, "older")).to_have_attribute("data-pinned", "")
    choose(second, "older", "Unpin")
    expect()(row(second, "older")).not_to_have_attribute("data-pinned", "")
    refresh(page)
    expect()(row(page, "older")).not_to_have_attribute("data-pinned", "")
    assert order(page) == ["newer", "older"]
    assert page.locator('#nav-projects-list [data-after-pins]').count() == 0
    assert row(page, "older").locator('.nav-project-row').get_attribute("title") == LONG
    # Rename remains a partial update and does not pin the room again.
    response = ui["client"].post('/api/projects/older/update', json={"name": "Renamed"})
    assert response.status_code == 200 and response.json()["project"]["pinned"] is False
    refresh(page)
    expect()(row(page, "older").locator('.nav-row-label')).to_have_text("Renamed")


def test_pointer_pin_of_first_visible_row_preserves_the_reading_place(pins_ui, tmp_path):
    """A saved pin mutates app.js's row; the last painted ordering key must stay independent of it."""
    ui, page, root = pins_ui, pins_ui["page"], pins_ui["root"]
    for index in range(35):
        registry.create_project(root, f"passage-{index:02}", name=f"Passage project {index:02}", origin="owner_ui")
    refresh(page)
    expect()(page.locator('.nav-project-item')).to_have_count(37)
    target, neighbor = row(page, "passage-15"), row(page, "passage-14")
    target.evaluate("el => el.scrollIntoView({block: 'start'})")
    first = page.evaluate("""() => {
        const list = document.getElementById('nav-projects-list'), top = list.getBoundingClientRect().top;
        return [...list.children].find(el => el.dataset.navProjectId
            && el.getBoundingClientRect().bottom > top).dataset.navProjectId;
    }""")
    assert first == "passage-15"
    before = neighbor.bounding_box()["y"]
    evidence = producer_evidence_dir(tmp_path, "pin-reading-place", env="OUROBOROS_BROWSER_EVIDENCE_OUT")
    page.locator('#primary-sidebar').screenshot(path=str(evidence / f'before-{ui["tag"]}.png'))
    open_menu(page, "passage-15")
    menu(page).locator('[data-prm="pin"]').click()
    expect()(target).to_have_attribute("data-pinned", "")
    settle(page)
    page.locator('#primary-sidebar').screenshot(path=str(evidence / f'after-{ui["tag"]}.png'))
    assert neighbor.bounding_box()["y"] == pytest.approx(before, abs=1)
    no_read_acknowledged(ui)


def test_collapsed_list_keeps_pins_and_unpins_in_place(pins_ui, tmp_path):
    ui, page = pins_ui, pins_ui["page"]
    evidence = producer_evidence_dir(tmp_path, "project-pin", env="OUROBOROS_BROWSER_EVIDENCE_OUT")
    choose(page, "older", "Pin")
    expect()(row(page, "older")).to_have_attribute("data-pinned", "")
    # The approved list never collapses: the pin remains in the one feed beside the recent row.
    expect()(row(page, "older")).to_be_visible()
    expect()(row(page, "newer")).to_be_visible()
    assert page.locator('#nav-projects-toggle, #nav-projects-show-all').count() == 0
    assert page.locator('#nav-projects-list [data-after-pins]').count() == 1
    page.screenshot(path=str(evidence / f"pinned-{ui['tag']}.png"), animations="disabled")
    # Unpin in place: the row remains in the same list and merely returns after the other row.
    choose(page, "older", "Unpin")
    expect()(row(page, "older")).not_to_have_attribute("data-pinned", "")
    assert registry.get_project(ui["root"], "older")["pinned"] is False
    expect()(row(page, "older")).to_be_visible()
    assert order(page) == ["newer", "older"]
    assert page.locator('#nav-projects-list [data-pinned], #nav-projects-list [data-after-pins]').count() == 0


def test_failed_and_pending_saves_are_honest_and_late_reads_cannot_undo_them(pins_ui):
    ui, page, control = pins_ui, pins_ui["page"], pins_ui["control"]
    control["fail"] = True
    choose(page, "older", "Pin")
    expect()(page.locator('#toast-stack')).to_contain_text("Could not save project pin: Synthetic write outage")
    expect()(row(page, "older")).not_to_have_attribute("data-pinned", "")
    assert registry.get_project(ui["root"], "older")["pinned"] is False
    assert len(control["writes"]) == 1
    control["fail"] = False
    # A save in flight reads as such in ⋯ and cannot be pressed twice; Rename and Delete stay available.
    control["hold_writes"] = True
    choose(page, "older", "Pin")
    until(page, lambda: len(control["held_writes"]) == 1)
    expect()(row(page, "older")).not_to_have_attribute("data-pinned", "")  # never shown before it is saved
    open_menu(page, "older")
    pending = menu(page).locator('[data-prm="pin"]')
    expect()(pending).to_have_text("Pinning…")
    expect()(pending).to_have_attribute("aria-disabled", "true")
    assert page.evaluate("document.activeElement.dataset.prm") == "rename"
    pending.dispatch_event("click")
    expect()(menu(page)).to_be_visible()
    assert len(control["writes"]) == 2
    assert menu(page).locator('[aria-disabled="true"]').count() == 1
    page.keyboard.press("Escape")
    expect()(menu(page)).to_have_count(0)
    # A state read requested before the confirmation cannot undo it.
    control["hold"] = True
    with page.expect_request(lambda request: urlparse(request.url).path == '/api/state'):
        refresh(page)
    until(page, lambda: bool(control["held"]))
    for route, path, body in control["held_writes"]:
        ui["forward"](route, path, body)
    control["held_writes"].clear()
    control["hold_writes"] = False
    expect()(row(page, "older")).to_have_attribute("data-pinned", "")
    control["hold"] = False
    for route, body in control["held"]:
        route.fulfill(content_type="application/json", body=body)
    control["held"].clear()
    with page.expect_response(lambda response: urlparse(response.url).path == '/api/state'):
        refresh(page)
    expect()(row(page, "older")).to_have_attribute("data-pinned", "")
    assert order(page) == ["older", "newer"]
    assert registry.get_project(ui["root"], "older")["pinned"] is True
    choose(page, "older", "Unpin")
    expect()(row(page, "older")).not_to_have_attribute("data-pinned", "")
    assert control["writes"][-1] == {"pinned": False}


@pytest.mark.parametrize("succeeded", [True, False], ids=["saved", "refused"])
def test_menu_opened_during_save_retires_pending_state_at_settlement(pins_ui, succeeded):
    ui, page, control = pins_ui, pins_ui["page"], pins_ui["control"]
    control["hold_writes"] = True
    choose(page, "older", "Pin")
    until(page, lambda: len(control["held_writes"]) == 1)
    glyph = row(page, "older").locator('.nav-project-pin')
    expect()(glyph).to_have_attribute("title", "Pinning…")
    expect()(glyph).to_have_attribute("aria-label", f"Pinning… {LONG}")
    expect()(glyph).to_have_attribute("aria-disabled", "true")
    glyph.dispatch_event("click")  # both controls share the existing pending-write guard
    assert len(control["writes"]) == 1
    anchor = open_menu(page, "older")
    expect()(menu(page).locator('[data-prm="pin"]')).to_have_text("Pinning…")
    assert page.evaluate("document.activeElement.dataset.prm") == "rename"
    route, path, body = control["held_writes"].pop()
    control["hold_writes"] = False
    if succeeded:
        ui["forward"](route, path, body)
    else:
        route.fulfill(status=503, content_type="application/json", body='{"error":"Synthetic write outage"}')
        expect()(page.locator('#toast-stack')).to_contain_text("Synthetic write outage")
    # Leave the pending menu open through completion: no stale disabled action survives.
    expect()(menu(page)).to_have_count(0)
    assert anchor.evaluate("el => el === document.activeElement")
    open_menu(page, "older")
    action = menu(page).locator('[data-prm="pin"]')
    expect()(action).to_have_text("Unpin" if succeeded else "Pin")
    assert action.get_attribute("aria-disabled") is None
    expect()(glyph).to_have_attribute("title", "Unpin" if succeeded else "Pin")
    assert glyph.get_attribute("aria-disabled") is None
    assert len(control["writes"]) == 1
    assert registry.get_project(ui["root"], "older")["pinned"] is succeeded


def test_pins_keep_unread_live_rows_menu_anchor_other_windows_and_deleting(pins_ui):
    ui, page, control = pins_ui, pins_ui["page"], pins_ui["control"]
    control["unread"].add("newer")
    control["activities"] = [{"activity_id": "work", "project_id": "newer",
                              "chat_id": registry.get_project(ui["root"], "newer")["chat_id"],
                              "kind": "managed_task", "phase": "working"}]
    refresh(page)
    newer = row(page, "newer").locator('.nav-project-row')
    expect()(newer).to_have_attribute("data-unread", "")
    # Its one mark: the work moves, tinted crimson for the new messages.
    expect()(newer.locator('.status-mark[data-shape="moving"][data-hue="unread"]')).to_be_visible()
    choose(page, "older", "Pin")
    expect()(row(page, "older")).to_have_attribute("data-pinned", "")
    assert order(page) == ["older", "newer"]
    expect()(newer).to_have_attribute("data-unread", "")
    assert newer.get_attribute("aria-label") == "Recent project · Working · new messages"
    # The row's own mark carries work and unread; the approved list has no aggregate header fact.
    expect()(row(page, "newer")).to_be_visible()
    assert page.locator('#nav-projects-count, #nav-projects-toggle, .nav-projects-facts').count() == 0
    # Another window unpins while this one's menu is open on that row: the menu stays with its row, and its label
    # still names what it does, so pressing it never flips the pin back. A menu opened from the ⋯ follows its moved
    # anchor; one a finger opened stays where the finger was.
    anchor = open_menu(page, "older")
    expect()(menu(page).locator('[data-prm="pin"]')).to_have_text("Unpin")
    page.evaluate("window.savedMenuRow = document.querySelector('[data-nav-project-id=older]')")
    original_anchor, original_popup = anchor.bounding_box(), menu(page).bounding_box()
    ui["client"].post('/api/projects/older/update', json={"name": "Changed elsewhere", "pinned": False})
    refresh(page)
    expect()(row(page, "older").locator('.nav-row-label')).to_have_text("Changed elsewhere")
    assert order(page) == ["newer", "older"]
    assert page.evaluate("savedMenuRow.isConnected && savedMenuRow === document.querySelector('[data-nav-project-id=older]')")
    expect()(menu(page)).to_be_visible()
    moved, popup = anchor.bounding_box(), menu(page).bounding_box()
    assert moved['y'] != original_anchor['y']  # a real reorder, not a same-position repaint
    if coarse(page):
        assert popup == original_popup
    else:
        assert abs(popup['y'] - (moved['y'] + moved['height'] + 4)) <= 1
    with page.expect_response(lambda response: response.url.endswith('/api/projects/older/update')):
        menu(page).locator('[data-prm="pin"]').click()
    assert control["writes"][-1] == {"pinned": False}
    assert registry.get_project(ui["root"], "older")["pinned"] is False
    expect()(row(page, "older")).not_to_have_attribute("data-pinned", "")
    control['unread'].clear()
    refresh(page)
    expect()(newer).not_to_have_attribute("data-unread", "")
    expect()(newer.locator('.status-mark')).to_have_attribute("data-hue", "ink")
    # A deleting row keeps its one trailing fact and no ⋯, so nothing can pin it.
    registry.update_project(ui["root"], "older", pinned=True)
    registry.begin_project_deletion(ui["root"], "older")
    refresh(page)
    expect()(row(page, "older").locator('.nav-project-row')).to_be_disabled()
    expect()(row(page, "older").locator('.nav-project-deleting-status')).to_have_text("Deleting…")
    assert row(page, "older").locator('.nav-project-kebab').count() == 0
    assert row(page, "older").locator('.nav-project-pin').count() == 0
    assert row(page, "older").locator('.nav-project-time').evaluate("el => getComputedStyle(el).marginRight") == "0px"
    title = row(page, "older").locator('.nav-project-row').get_attribute("title")
    assert title.startswith("Changed elsewhere · Pinned — Deleting…"), title


def test_many_pins_order_within_groups_long_names_and_short_windows(pins_ui, tmp_path):
    ui, page, root, client = pins_ui, pins_ui["page"], pins_ui["root"], pins_ui["client"]
    evidence = producer_evidence_dir(tmp_path, "project-pin", env="OUROBOROS_BROWSER_EVIDENCE_OUT")
    names = {f"p{i}": f"Fictional research room {i} with a deliberately long descriptive name" for i in range(10)}
    for i, (pid, name) in enumerate(names.items()):
        registry.create_project(root, pid, name=name, origin="owner_ui")
        registry.update_project(root, pid, last_active_at=f"2026-03-{i + 1:02}T00:00:00Z")
    for pid in ("p1", "p4", "p7"):
        assert client.post(f'/api/projects/{pid}/update', json={"pinned": True}).json()["project"]["pinned"] is True
    ui["control"]["unread"].update({"p1", "p8"})
    refresh(page)
    expect()(row(page, "p7")).to_have_attribute("data-pinned", "")
    # Pins lead; recency orders within each group, unread only marks a row (owner 2026-11-07).
    expected = ["p7", "p4", "p1", "newer", "p9", "p8", "p6", "p5", "p3", "p2", "p0", "older"]
    assert order(page) == expected
    groups = page.evaluate(GROUPS)
    assert [g["id"] for g in groups] == expected
    assert_one_boundary(groups)
    assert_flat_and_unlabelled(page)
    by_id = {g["id"]: g for g in groups}
    assert by_id["p1"]["title"] == f"{names['p1']} · Pinned · new messages"
    # A row with no activity line spends both lines on its name; long names wrap/clamp instead of entering the time or mark rail.
    assert by_id["p1"]["solo"] and by_id["p1"]["labelScroll"] <= by_id["p1"]["labelWidth"], by_id["p1"]
    assert by_id["p8"]["title"] == f"{names['p8']} · new messages"
    expect()(row(page, "p1").locator('.status-mark[data-shape="dot"][data-hue="unread"]')).to_be_visible()
    assert page.locator('#primary-sidebar').evaluate("e => e.scrollWidth <= e.clientWidth")
    page.locator('#primary-sidebar').screenshot(path=str(evidence / f"many-{ui['tag']}.png"), animations="disabled")
    # A short window keeps the boundary in the whole column's scroll (480 px: a desk's zones fit from 500); the approved
    # two-line rows keep their touch/desk heights.
    page.set_viewport_size({"width": ui["width"], "height": 480})
    page.wait_for_function("() => getComputedStyle(document.getElementById('nav-projects-list'))"
                           ".overflowY === 'visible'")
    short = page.evaluate(GROUPS)
    assert_one_boundary(short)
    assert {g["height"] for g in short} == {52 if ui["width"] < 720 else 48}
    assert page.locator('#primary-sidebar').evaluate(
        "e => e.scrollHeight > e.clientHeight && e.scrollWidth <= e.clientWidth")
    page.screenshot(path=str(evidence / f"short-{ui['tag']}.png"), animations="disabled")


def test_menu_repositions_after_a_hidden_pin_is_revealed(pins_ui, tmp_path):
    ui, page, control = pins_ui, pins_ui["page"], pins_ui["control"]
    choose(page, "newer", "Pin")
    # The approved feed has pins and fifty recent rooms, with no collapse. An older room outside that feed
    # is revealed by another window's pin, while the first pin's menu stays open.
    for index in range(50):
        registry.create_project(ui["root"], f"recent-{index:02}", name=f"Recent room {index:02}", origin="owner_ui")
    refresh(page)
    expect()(page.locator('.nav-project-item')).to_have_count(51)
    expect()(row(page, "older")).to_be_hidden()
    before = shown_ids(page)
    assert before[0] == "newer"
    anchor = open_menu(page, "newer")
    original_anchor, original_popup = anchor.bounding_box(), menu(page).bounding_box()
    # Another window pins the older room, which has spoken since (this fixture's store is not dated yet: activity
    # orders it): it is revealed above this row.
    registry.update_project(ui["root"], "older", last_active_at="2099-01-01T00:00:00Z")
    control["unread"].add("older")
    response = ui["client"].post('/api/projects/older/update', json={"pinned": True})
    assert response.status_code == 200
    refresh(page)
    expect()(row(page, "older")).to_be_visible()
    assert shown_ids(page) == ["older", *before]
    expect()(page.locator('.nav-project-item')).to_have_count(52)
    expect()(menu(page)).to_be_visible()
    moved, popup = anchor.bounding_box(), menu(page).bounding_box()
    assert moved['y'] != original_anchor['y']
    if coarse(page):  # a finger opened it: it stays where the finger was
        assert popup == original_popup
    else:
        assert abs(popup['y'] - (moved['y'] + moved['height'] + 4)) <= 1
    assert_one_boundary(page.evaluate(GROUPS))
    evidence = producer_evidence_dir(tmp_path, "pin-menu", env="OUROBOROS_BROWSER_EVIDENCE_OUT")
    page.screenshot(path=str(evidence / f"collapsed-menu-{ui['tag']}.png"), animations="disabled")


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_pin_real_server_updates_other_window_and_survives_reload(direct_server_with_data, engine, tmp_path):  # noqa: F811
    """No API/WS interception: real isolated server, public routes, two independent clients."""
    from playwright.sync_api import sync_playwright
    from tests._dialogue import advance_room_revision

    server = direct_server_with_data
    root, url = server["data_dir"], server["url"]
    registry.create_project(root, "pin-old", name="Archive research project")
    registry.create_project(root, "pin-new", name="Recent unread project")
    registry.update_project(root, "pin-old", last_active_at="2026-01-01T00:00:00Z")
    revision = advance_room_revision(root, project_id="pin-new")["visible_revision"]
    evidence = producer_evidence_dir(tmp_path, "pin-real-server", env="OUROBOROS_BROWSER_EVIDENCE_OUT")
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            first = browser.new_page(viewport={"width": 1172, "height": 875})
            second = browser.new_page(viewport={"width": 1172, "height": 875})
            for page in (first, second):
                page.goto(url, wait_until="domcontentloaded")
                expect()(row(page, "pin-new").locator('.nav-project-row')).to_have_attribute("data-unread", "", timeout=30000)
            choose(first, "pin-old", "Pin")
            expect()(row(first, "pin-old")).to_have_attribute("data-pinned", "")
            expect()(row(second, "pin-old")).to_have_attribute("data-pinned", "", timeout=30000)
            for page in (first, second):
                assert order(page) == ["pin-old", "pin-new"]
                expect()(row(page, "pin-new").locator('.status-mark[data-hue="unread"]')).to_be_visible()
                assert page.locator('#project-panel').is_hidden()
                assert_flat_and_unlabelled(page)
                assert_one_boundary(page.evaluate(GROUPS))
            first.reload()
            expect()(row(first, "pin-old")).to_have_attribute("data-pinned", "", timeout=30000)
            first.screenshot(path=str(evidence / f"real-server-{engine}.png"))
            choose(second, "pin-old", "Unpin")
            expect()(row(first, "pin-old")).not_to_have_attribute("data-pinned", "", timeout=30000)
            assert order(first) == ["pin-new", "pin-old"]
            assert registry.get_project(root, "pin-old")["pinned"] is False
            assert next(p for p in registry.projects_summary(root) if p["id"] == "pin-new")["visible_revision"] == revision
        finally:
            browser.close()


# 900: the zones, the list its own scroller; 500: a desk's zones still, a 390 phone's drawer too short for Main, a row
# and its foot together (both go with the list); 480: a desk's whole column scrolling under its two pinned bars.
@pytest.mark.parametrize("height", [900, 500, 480])
@pytest.mark.parametrize("keyboard", [True, False], ids=["keyboard", "pointer"])
def test_local_pin_reveal_respects_input_and_visible_scroll_region(pins_ui, height, keyboard):
    ui, page, root = pins_ui, pins_ui["page"], pins_ui["root"]
    page.set_viewport_size({"width": ui["width"], "height": height})
    for i in range(35):
        registry.create_project(root, f"scroll-{i}", name=f"Research {i}", origin="owner_ui")
    refresh(page)
    expect()(page.locator('.nav-project-item')).to_have_count(37)
    kebab = row(page, "older").locator('.nav-project-kebab')
    row(page, "older").scroll_into_view_if_needed()
    scroll = "() => ['primary-sidebar', 'nav-projects-list'].map(id => document.getElementById(id).scrollTop)"
    if keyboard:
        kebab.focus()
        page.keyboard.press("Enter")
    elif coarse(page):
        long_press(page, row(page, "older").locator('.nav-project-row'))
    else:
        kebab.click()
    expect()(menu(page).locator('[data-prm="pin"]')).to_have_text("Pin")
    before = page.evaluate(scroll)
    if keyboard:
        page.keyboard.press("Enter")
    elif coarse(page):
        menu(page).locator('[data-prm="pin"]').tap()
    else:
        menu(page).locator('[data-prm="pin"]').click()
    expect()(row(page, "older")).to_have_attribute("data-pinned", "")
    if keyboard:
        assert kebab.evaluate("el => el === document.activeElement")
        assert kebab.evaluate("""el => {
            const r = el.getBoundingClientRect();
            return r.top >= 0 && r.bottom <= innerHeight
                && el.contains(document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2));
        }"""), "keyboard focus must remain visible and unobscured by the sticky head"
    else:
        # Native anchoring can shift by the moved row plus the new spacing boundary,
        # but must not follow that row hundreds of pixels to its new position.
        allowance = row(page, "older").evaluate("el => el.getBoundingClientRect().height + parseFloat(getComputedStyle(el).getPropertyValue('--space-2')) + 3")
        assert max(abs(a - b) for a, b in zip(before, page.evaluate(scroll))) <= allowance, "pointer pin must not jump the list"
    expect()(page.locator('#toast-stack').filter(has_text="Could not save project pin")).to_have_count(0)


# A row's own Pin as it stands: drawn, a target, what it does and says, its ink beside the row's, real SVG path
# identity against SHELL_ICONS, and the boxes of the Pin, its rendered glyph, its ⋯, the name, the time and the mark.
PIN = """async (pid) => {
    const { SHELL_ICONS } = await import('/static/modules/page_icons.js');
    const paths = (node) => [...node.querySelectorAll('path')].map((p) => p.getAttribute('d'));
    const parse = (html) => { const d = document.createElement('div'); d.innerHTML = html; return paths(d); };
    const pinPaths = parse(SHELL_ICONS.pin), unpinPaths = parse(SHELL_ICONS.unpin);
    const same = (a, b) => a.length > 0 && a.length === b.length && a.every((v, i) => v === b[i]);
    const item = document.querySelector(`.nav-project-item[data-nav-project-id="${pid}"]`);
    const pin = item.querySelector(':scope > .nav-project-pin'), row = item.querySelector('.nav-project-row');
    const kebab = item.querySelector(':scope > .nav-project-kebab');
    const box = (el) => { const r = el.getBoundingClientRect(); return {left: r.left, right: r.right, top: r.top, bottom: r.bottom}; };
    const style = getComputedStyle(pin), kebabStyle = kebab && getComputedStyle(kebab);
    const svgs = [...pin.querySelectorAll(':scope > svg')];
    const visible = svgs.filter((svg) => getComputedStyle(svg).display !== 'none');
    const shownSvg = visible.length === 1 ? visible[0] : null;
    const shownPaths = shownSvg ? paths(shownSvg) : [];
    const slot = row.querySelector('.nav-mark-slot'), mark = row.querySelector('.status-mark');
    const markStyle = mark && getComputedStyle(mark);
    return {drawn: style.opacity === '1' && style.visibility !== 'hidden', target: style.pointerEvents !== 'none',
            action: pin.dataset.action, name: pin.getAttribute('aria-label'), title: pin.title, ink: style.color,
            ring: style.outlineStyle, rowInk: getComputedStyle(row).color, rowFill: getComputedStyle(row).backgroundColor,
            focused: pin === document.activeElement, keyboard: pin.matches(':focus-visible'),
            first: item.firstElementChild === pin, item: box(item), pin: box(pin),
            glyph: shownSvg ? box(shownSvg) : null, kebab: kebab ? box(kebab) : null,
            kebabDrawn: Boolean(kebabStyle) && kebabStyle.opacity === '1' && kebabStyle.visibility !== 'hidden',
            ariaHidden: svgs.length === (pin.dataset.action === 'unpin' ? 2 : 1)
                && svgs.every((svg) => svg.getAttribute('aria-hidden') === 'true'),
            restGlyph: same(shownPaths, pinPaths), actGlyph: same(shownPaths, unpinPaths),
            label: box(row.querySelector('.nav-row-label')), time: box(row.querySelector('.nav-project-time')),
            line: box(row.querySelector('.nav-project-line1')),
            timeDrawn: getComputedStyle(row.querySelector('.nav-project-time')).visibility !== 'hidden',
            markDrawn: Boolean(markStyle) && markStyle.display !== 'none' && markStyle.visibility !== 'hidden',
            slot: slot ? box(slot) : null};
}"""
# Records what the next click in the document lands on (`window.__landed`): a row's Pin, its row or something else,
# with that row's id.
CLICK_TARGET = """() => { window.__landed = null; document.addEventListener('click', (event) => {
    const hit = event.target.closest('.nav-project-pin, .nav-project-row');
    window.__landed = hit ? [hit.classList.contains('nav-project-pin') ? 'pin' : 'row',
        hit.closest('[data-nav-project-id]').dataset.navProjectId] : ['other', ''];
}, {capture: true, once: true}); }"""


def middle(box):
    return (box["left"] + box["right"]) / 2, (box["top"] + box["bottom"]) / 2


def width_of(box):
    return box["right"] - box["left"]


def end_inset(state, key, rtl=False):
    return (state[key]["left"] - state["item"]["left"]) if rtl else (state["item"]["right"] - state[key]["right"])


def wait_row_paint(page, pid):
    """Flush style and finish finite row/control transitions (skipping infinite status-mark bounce)."""
    row(page, pid).evaluate("""item => {
        void item.offsetWidth;
        return Promise.all(item.getAnimations({subtree: true})
            .filter((a) => a.playState === 'running' && Number.isFinite(a.effect?.getTiming?.().iterations))
            .map((a) => a.finished.catch(() => null)));
    }""")


def keyboard_focus(page, target):
    """Switch input modality to keyboard, then focus ``target``, so `:focus-visible` holds on Chromium and WebKit (whose
    Tab need not stop on buttons); callers assert what landed."""
    page.locator('.nav-find-input').focus()
    page.keyboard.press("ArrowRight")
    target.evaluate("el => el.focus()")


def focus_pin_via_keyboard(page, pid):
    """Focus ``pid``'s Pin as the keyboard does."""
    glyph = row(page, pid).locator('.nav-project-pin')
    keyboard_focus(page, glyph)
    state = page.evaluate(PIN, pid)
    assert state["focused"] and state["keyboard"], state
    return glyph, state


def cdp_touch(cdp, x, y, hold_ms=0, page=None):
    cdp.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{"x": x, "y": y, "id": 1}]})
    if hold_ms and page is not None:
        page.clock.run_for(hold_ms)
    cdp.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})


def test_row_pin_glyph_serves_a_hovering_pointer_and_the_keyboard_never_a_finger(pins_ui, tmp_path):
    """Unpinned rows reveal Pin only on hover or keyboard focus; pinned rows keep a quiet pin glyph at rest beside
    their 20px-shifted time, swap to Unpin only on the Pin button's own hover or keyboard focus, coexist with status
    marks, keep their [W-56, W-30] Pin box clear of ⋯ (a pinned time's last letters stay the row's: the RTL test
    below), and keep touch taps opening the row."""
    ui, page, control = pins_ui, pins_ui["page"], pins_ui["control"]
    evidence = producer_evidence_dir(tmp_path, "pin-glyph", env="OUROBOROS_BROWSER_EVIDENCE_OUT")
    pin = lambda pid: page.evaluate(PIN, pid)  # noqa: E731
    landed = lambda: page.evaluate("window.__landed")  # noqa: E731
    control["message_times"] = {"newer": "2026-03-10T11:20:00Z", "older": "2026-03-10T10:15:00Z"}
    refresh(page)
    settle(page)  # a phone's drawer has slid in: boxes are read where they stay
    rest, newer_rest = pin("older"), pin("newer")
    assert rest["first"] and rest["action"] == "pin" and rest["ariaHidden"] and not rest["drawn"] and not rest["target"], rest
    assert (rest["name"], rest["title"]) == (f"Pin {LONG}", "Pin")
    assert rest["label"]["right"] == pytest.approx(rest["time"]["left"] - 8, abs=0.01), rest
    assert width_of(rest["time"]) > 0 and rest["timeDrawn"], rest
    assert rest["pin"]["left"] < rest["time"]["right"] <= rest["pin"]["right"], rest
    assert rest["pin"]["right"] == pytest.approx(rest["time"]["right"] + (5 if coarse(page) else 3), abs=0.5), rest
    assert rest["pin"]["top"] == pytest.approx(rest["item"]["top"] + 3, abs=0.5), rest
    assert rest["pin"]["bottom"] - rest["pin"]["top"] == pytest.approx(24, abs=0.5), rest
    assert width_of(rest["pin"]) == pytest.approx(44 if coarse(page) else 26, abs=0.5), rest
    if not coarse(page):
        assert end_inset(rest, "pin") == pytest.approx(30, abs=0.5), rest
        assert rest["item"]["right"] - rest["pin"]["left"] == pytest.approx(56, abs=0.5), rest
        assert rest["pin"]["right"] <= rest["kebab"]["left"] + 0.01, rest
    assert_flat_and_unlabelled(page)
    if coarse(page):
        assert page.evaluate("matchMedia('(hover: none)').matches")
        # Unpinned coarse row keeps its centred 44px keyboard target; blur hides it and tapping opens the row.
        _, focused = focus_pin_via_keyboard(page, "older")
        assert focused["drawn"] and focused["target"] and focused["ring"] == "solid" and focused["restGlyph"], focused
        assert middle(focused["glyph"])[0] == pytest.approx(middle(focused["pin"])[0], abs=0.5), focused
        page.evaluate("document.activeElement.blur()")
        resting = pin("older")
        assert not resting["drawn"]
        x, y = middle(resting["pin"])
        assert page.evaluate("([x, y]) => document.elementFromPoint(x, y).closest('.nav-project-row')?.dataset.projectId",
                             [x, y]) == "older"
        page.locator('#primary-sidebar').screenshot(path=str(evidence / f"touch-rest-{ui['tag']}.png"), animations="disabled")
        page.evaluate(CLICK_TARGET)
        page.touchscreen.tap(x, y)
        assert landed() == ["row", "older"]
        settle(page)
        expect()(row(page, "older").locator('.nav-project-row')).to_have_attribute("aria-current", "page")
        expect()(page.locator('#project-panel')).to_be_visible()
        assert control["writes"] == [] and registry.get_project(ui["root"], "older")["pinned"] is False
        # Close and pin older with live status: same-frame geometry proves 20px time/label shift and aligned rail.
        page.locator('#project-panel-close').click()
        settle(page)
        expect()(page.locator('#project-panel')).to_be_hidden()
        open_drawer(page, ui["width"])
        assert ui["client"].post("/api/projects/older/update", json={"pinned": True}).status_code == 200
        control["unread"].add("older")
        control["activities"] = [{"activity_id": "work-pin", "project_id": "older",
                                  "chat_id": registry.get_project(ui["root"], "older")["chat_id"],
                                  "kind": "managed_task", "phase": "working"}]
        refresh(page)
        expect()(row(page, "older")).to_have_attribute("data-pinned", "")
        settle(page)
        pinned_rest, newer_after = pin("older"), pin("newer")
        assert pinned_rest["drawn"] and pinned_rest["restGlyph"] and not pinned_rest["actGlyph"] and not pinned_rest["target"], pinned_rest
        assert pinned_rest["ariaHidden"] and pinned_rest["timeDrawn"] and pinned_rest["markDrawn"], pinned_rest
        assert pinned_rest["slot"]["left"] == pytest.approx(newer_after["slot"]["left"], abs=0.5), (pinned_rest, newer_after)
        assert end_inset(pinned_rest, "time") - end_inset(newer_after, "time") == pytest.approx(20, abs=0.5), (pinned_rest, newer_after)
        assert width_of(rest["label"]) - width_of(pinned_rest["label"]) == pytest.approx(20, abs=0.5), (rest, pinned_rest)
        assert width_of(newer_rest["label"]) == pytest.approx(width_of(newer_after["label"]), abs=0.5), (newer_rest, newer_after)
        assert pinned_rest["glyph"]["left"] - pinned_rest["time"]["right"] == pytest.approx(4, abs=0.5), pinned_rest
        assert not newer_after["drawn"] and not newer_after["target"], newer_after
        # Even after keyboard focus on a pinned row's Pin, tapping the glyph still passes through and opens the row.
        _, pinned_kb = focus_pin_via_keyboard(page, "older")
        assert pinned_kb["drawn"] and pinned_kb["actGlyph"] and not pinned_kb["restGlyph"], pinned_kb
        assert pinned_kb["ring"] == "solid" and not pinned_kb["timeDrawn"] and not pinned_kb["markDrawn"] and not pinned_kb["target"], pinned_kb
        gx, gy = middle(pinned_kb["glyph"])
        assert page.evaluate("([x, y]) => document.elementFromPoint(x, y).closest('.nav-project-row')?.dataset.projectId",
                             [gx, gy]) == "older"
        page.evaluate(CLICK_TARGET)
        page.touchscreen.tap(gx, gy)
        assert landed() == ["row", "older"]
        settle(page)
        expect()(row(page, "older").locator('.nav-project-row')).to_have_attribute("aria-current", "page")
        assert control["writes"] == [] and registry.get_project(ui["root"], "older")["pinned"] is True
        # Reopen drawer over active pinned row: chosen ink at rest, hidden when ⋯ owns the end, long press offers Unpin.
        page.locator('#project-panel [data-mobile-nav-toggle]').click()
        page.locator('#primary-sidebar.open').wait_for()
        settle(page)
        chosen = pin("older")
        assert chosen["drawn"] and chosen["restGlyph"] and not chosen["actGlyph"] and chosen["ink"] == chosen["rowInk"], chosen
        kebab = row(page, "older").locator('.nav-project-kebab')
        kebab.focus()
        assert not pin("older")["drawn"]
        page.keyboard.press("Enter")
        expect()(menu(page)).to_be_visible()
        assert not pin("older")["drawn"]
        page.keyboard.press("Escape")
        expect()(menu(page)).to_have_count(0)
        page.evaluate("document.activeElement.blur()")
        assert pin("older")["drawn"]
        long_press(page, row(page, "older").locator('.nav-project-row'))
        expect()(menu(page)).to_be_visible()
        expect()(menu(page).locator('[data-prm="pin"]')).to_have_text("Unpin")
        page.keyboard.press("Escape")
        expect()(menu(page)).to_have_count(0)
        return
    assert page.evaluate("matchMedia('(hover: hover) and (pointer: fine)').matches")
    older = row(page, "older").locator('.nav-project-row')
    # Hovering an unpinned row draws Pin over its hidden time; moving onto Pin takes primary ink and click pins.
    page.mouse.move(*middle(rest["label"]))
    wait_row_paint(page, "older")
    hovered = pin("older")
    assert hovered["drawn"] and hovered["target"] and hovered["restGlyph"] and hovered["label"] == rest["label"], hovered
    assert not hovered["timeDrawn"] and hovered["time"] == rest["time"], hovered
    assert hovered["item"]["right"] - middle(hovered["glyph"])[0] == pytest.approx(42, abs=0.5), hovered
    assert hovered["ink"] != hovered["rowInk"] and not pin("newer")["drawn"], hovered
    page.locator('#primary-sidebar').screenshot(path=str(evidence / f"hover-pin-{ui['tag']}.png"), animations="disabled")
    page.mouse.move(*middle(hovered["pin"]))
    wait_row_paint(page, "older")
    on = pin("older")
    assert on["rowFill"] == hovered["rowFill"] and on["ink"] == on["rowInk"], on
    page.mouse.down()
    page.mouse.up()
    expect()(row(page, "older")).to_have_attribute("data-pinned", "")
    assert order(page) == ["older", "newer"] and control["writes"] == [{"pinned": True}]
    assert page.locator('.nav-project-row.active').count() == 0 and page.locator('#project-panel').is_hidden()
    assert no_read_acknowledged(ui)
    # Pinned row at rest without blur: non-visible focus stays on Pin while ⋯ hides, mark/time return, glyph stays at W-42.
    page.mouse.move(ui["width"] - 40, 300)
    control["unread"].add("older")
    control["activities"] = [{"activity_id": "work-pin", "project_id": "older",
                              "chat_id": registry.get_project(ui["root"], "older")["chat_id"],
                              "kind": "managed_task", "phase": "working"}]
    refresh(page)
    settle(page)
    left, newer_after = pin("older"), pin("newer")
    assert left["focused"] and not left["keyboard"] and left["drawn"] and left["restGlyph"] and not left["actGlyph"] and left["target"], left
    assert not newer_after["drawn"] and not newer_after["target"], newer_after
    assert left["ariaHidden"] and left["timeDrawn"] and left["markDrawn"] and not left["kebabDrawn"], left
    assert left["slot"]["left"] == pytest.approx(newer_after["slot"]["left"], abs=0.5), (left, newer_after)
    assert left["glyph"]["left"] == pytest.approx(hovered["glyph"]["left"], abs=0.5), (left, hovered)
    assert left["glyph"]["right"] == pytest.approx(hovered["glyph"]["right"], abs=0.5), (left, hovered)
    assert left["pin"]["right"] <= left["kebab"]["left"] + 0.01, left
    assert end_inset(left, "time") - end_inset(newer_after, "time") == pytest.approx(20, abs=0.5), (left, newer_after)
    assert width_of(rest["label"]) - width_of(left["label"]) == pytest.approx(20, abs=0.5), (rest, left)
    assert width_of(newer_rest["label"]) == pytest.approx(width_of(newer_after["label"]), abs=0.5), (newer_rest, newer_after)
    assert left["glyph"]["left"] - left["time"]["right"] == pytest.approx(3, abs=0.5), left
    assert_flat_and_unlabelled(page)
    # 1) Hover pinned row label: resting pin and time stay visible while ⋯ opens, with zero shift.
    page.mouse.move(*middle(left["label"]))
    wait_row_paint(page, "older")
    row_hover = pin("older")
    assert row_hover["drawn"] and row_hover["target"] and row_hover["kebabDrawn"], row_hover
    assert row_hover["restGlyph"] and not row_hover["actGlyph"] and row_hover["ink"] != row_hover["rowInk"], row_hover
    assert row_hover["timeDrawn"] and not row_hover["markDrawn"] and row_hover["action"] == "unpin", row_hover
    assert (row_hover["name"], row_hover["title"]) == (f"Unpin {LONG}", "Unpin")
    for key in ("item", "label", "time", "glyph", "pin"):
        assert row_hover[key] == left[key], (key, left[key], row_hover[key])
    # 2) Hover Pin button itself: swaps in place to slashed Unpin in primary ink, keeping time and row hover fill.
    page.mouse.move(*middle(row_hover["pin"]))
    wait_row_paint(page, "older")
    btn_hover = pin("older")
    assert btn_hover["drawn"] and btn_hover["target"] and btn_hover["actGlyph"] and not btn_hover["restGlyph"], btn_hover
    assert btn_hover["ink"] == btn_hover["rowInk"] and btn_hover["rowFill"] == row_hover["rowFill"], btn_hover
    assert btn_hover["timeDrawn"] and not btn_hover["markDrawn"], btn_hover
    for key in ("item", "label", "time", "glyph", "pin"):
        assert btn_hover[key] == left[key], (key, left[key], btn_hover[key])
    page.locator('#primary-sidebar').screenshot(path=str(evidence / f"hover-unpin-{ui['tag']}.png"), animations="disabled")
    # Double-click unpin guard: the second press lands on the moved neighbour's Pin and pins or opens nothing.
    control["hold_writes"] = True
    page.mouse.down(click_count=1)
    page.mouse.up(click_count=1)
    until(page, lambda: len(control["held_writes"]) == 1)
    control["hold_writes"] = False
    ui["forward"](*control["held_writes"].pop())
    expect()(row(page, "older")).not_to_have_attribute("data-pinned", "")
    assert order(page) == ["newer", "older"]
    settle(page)
    assert pin("newer")["drawn"]  # the hover followed the row that moved under the pointer
    page.evaluate(CLICK_TARGET)
    page.mouse.down(click_count=2)
    page.mouse.up(click_count=2)
    assert landed() == ["pin", "newer"]  # the press the guard exists for
    settle(page)
    assert control["writes"] == [{"pinned": True}, {"pinned": False}], control["writes"]
    assert registry.get_project(ui["root"], "newer")["pinned"] is False
    assert page.locator('.nav-project-row.active').count() == 0 and page.locator('#project-panel').is_hidden()
    # Moving pointer off after mouse click restores resting unpinned state without blur (focus on newer's Pin). Newer
    # gets a mark of its own (its new messages' dot), so the keyboard stops below have one to hide.
    page.mouse.move(ui["width"] - 40, 300)
    control["unread"].add("newer")
    refresh(page)
    expect()(row(page, "newer").locator('.nav-project-row')).to_have_attribute("data-unread", "")
    settle(page)
    unpinned_rest, newer_left = pin("older"), pin("newer")
    assert not unpinned_rest["drawn"] and not unpinned_rest["kebabDrawn"] and unpinned_rest["timeDrawn"] and unpinned_rest["markDrawn"], unpinned_rest
    assert newer_left["focused"] and not newer_left["keyboard"] and not newer_left["drawn"] and not newer_left["kebabDrawn"], newer_left
    assert newer_left["markDrawn"], newer_left
    # 3) Keyboard traversal Pin -> row -> ⋯ -> Escape: Tab -> Enter pins, each keyboard stop keeps ⋯ and hides mark.
    page.locator('.nav-find-input').focus()
    page.keyboard.press("Tab")
    assert page.evaluate("document.activeElement.id") == "nav-projects-add"
    page.keyboard.press("Tab")
    keyed = pin("newer")
    assert keyed["focused"] and keyed["keyboard"] and keyed["drawn"] and keyed["ring"] == "solid" and keyed["kebabDrawn"], keyed
    page.keyboard.press("Enter")
    expect()(row(page, "newer")).to_have_attribute("data-pinned", "")
    assert control["writes"][-1] == {"pinned": True} and len(control["writes"]) == 3
    kept = pin("newer")
    assert kept["focused"] and kept["keyboard"] and kept["drawn"] and kept["actGlyph"] and not kept["restGlyph"], kept
    assert kept["kebabDrawn"] and not kept["timeDrawn"] and not kept["markDrawn"] and kept["target"] and kept["action"] == "unpin", kept
    page.locator('#primary-sidebar').screenshot(path=str(evidence / f"keyboard-{ui['tag']}.png"), animations="disabled")
    page.keyboard.press("Tab")  # on to the name: resting pin & time return, ⋯ stays drawn over hidden mark
    after_tab = pin("newer")
    assert page.evaluate("document.activeElement.dataset.projectId") == "newer"
    assert after_tab["drawn"] and after_tab["restGlyph"] and not after_tab["actGlyph"] and after_tab["target"], after_tab
    assert after_tab["timeDrawn"] and after_tab["kebabDrawn"] and not after_tab["markDrawn"], after_tab
    page.keyboard.press("Tab")  # on to ⋯: open menu -> Escape keeps keyboard focus and hidden mark
    assert page.evaluate("document.activeElement.classList.contains('nav-project-kebab')")
    assert pin("newer")["kebabDrawn"] and not pin("newer")["markDrawn"]
    page.keyboard.press("Enter")
    expect()(menu(page)).to_be_visible()
    page.keyboard.press("Escape")
    expect()(menu(page)).to_have_count(0)
    assert pin("newer")["kebabDrawn"] and not pin("newer")["markDrawn"]
    page.keyboard.press("Tab")  # the keyboard leaves the row: its mark returns, the next row's gives way
    left_row, next_row = pin("newer"), pin("older")
    assert left_row["markDrawn"] and not left_row["kebabDrawn"] and next_row["keyboard"] and not next_row["markDrawn"], (left_row, next_row)
    page.keyboard.press("Shift+Tab")
    # Hybrid touch guard on an unpinned row: tap begins on the row, so it opens even if hover drew Pin under the click.
    page.mouse.move(ui["width"] - 40, 300)
    x, y = middle(pin("older")["pin"])
    assert not pin("older")["drawn"]
    cdp = page.context.new_cdp_session(page)
    try:
        page.evaluate(CLICK_TARGET)
        cdp_touch(cdp, x, y)
        assert landed() == ["pin", "older"]  # the click the tap brought, once its hover drew the Pin
        settle(page)
        expect()(older).to_have_attribute("aria-current", "page")
        assert len(control["writes"]) == 3 and registry.get_project(ui["root"], "older")["pinned"] is False
        # Over the chosen unpinned row the Pin wears the row's ink, and the row keeps its chosen fill under it.
        page.mouse.move(*middle(pin("older")["label"]))
        wait_row_paint(page, "older")
        chosen = pin("older")
        page.mouse.move(*middle(chosen["pin"]))
        wait_row_paint(page, "older")
        chosen_on = pin("older")
        assert chosen["drawn"] and chosen["ink"] == chosen["rowInk"], chosen
        assert chosen_on["ink"] == chosen["rowInk"] and chosen_on["rowFill"] == chosen["rowFill"], chosen_on
        # Hybrid touch on a keyboard-focused pinned row's glyph still opens it without unpinning; chosen pinned ink holds.
        page.mouse.move(ui["width"] - 40, 300)
        _, focused_pin = focus_pin_via_keyboard(page, "newer")
        assert focused_pin["actGlyph"] and focused_pin["target"], focused_pin
        page.evaluate(CLICK_TARGET)
        cdp_touch(cdp, *middle(focused_pin["glyph"]))
        assert landed()[1] == "newer"
        settle(page)
        expect()(row(page, "newer").locator('.nav-project-row')).to_have_attribute("aria-current", "page")
        assert len(control["writes"]) == 3 and registry.get_project(ui["root"], "newer")["pinned"] is True
    finally:
        cdp.detach()
    page.mouse.move(ui["width"] - 40, 300)
    page.evaluate("document.activeElement?.blur()")
    wait_row_paint(page, "newer")
    chosen_pin = pin("newer")
    assert chosen_pin["drawn"] and chosen_pin["restGlyph"] and not chosen_pin["actGlyph"] and chosen_pin["ink"] == chosen_pin["rowInk"], chosen_pin
    page.mouse.move(*middle(chosen_pin["pin"]))
    wait_row_paint(page, "newer")
    chosen_pin_on = pin("newer")
    assert chosen_pin_on["drawn"] and chosen_pin_on["actGlyph"] and not chosen_pin_on["restGlyph"], chosen_pin_on
    assert chosen_pin_on["ink"] == chosen_pin["rowInk"] and chosen_pin_on["rowFill"] == chosen_pin["rowFill"], chosen_pin_on
    page.locator('#primary-sidebar').screenshot(path=str(evidence / f"chosen-{ui['tag']}.png"), animations="disabled")
    page.evaluate("document.documentElement.dataset.theme = 'light'")  # the other theme's frame, for the record
    page.mouse.move(*middle(pin("newer")["label"]))
    wait_row_paint(page, "newer")
    page.locator('#primary-sidebar').screenshot(path=str(evidence / f"hover-light-{ui['tag']}.png"), animations="disabled")


def test_focused_pin_accepts_semantic_activation_without_a_pointer_press(pins_ui):
    """A semantic detail-1 click on the visibly focused button activates that button, not its neighbouring row.
    This fixes the event contract; it is not a native screen-reader qualification."""
    ui, page, control = pins_ui, pins_ui["page"], pins_ui["control"]
    settle(page)
    glyph, focused = focus_pin_via_keyboard(page, "older")
    assert focused["drawn"], focused
    glyph.dispatch_event("click", {"detail": 1})
    expect()(row(page, "older")).to_have_attribute("data-pinned", "")
    assert control["writes"] == [{"pinned": True}]
    assert page.locator('#project-panel').is_hidden() and no_read_acknowledged(ui)
    page.keyboard.press("Space")
    expect()(row(page, "older")).not_to_have_attribute("data-pinned", "")
    assert control["writes"] == [{"pinned": True}, {"pinned": False}]
    assert page.locator('#project-panel').is_hidden() and no_read_acknowledged(ui)


def test_hovered_hybrid_touch_and_stale_menu_gestures(pins_ui):
    """A finger opens a pinned row at rest or when mouse hover exposed Pin; long press on the pin spot opens the
    menu with Unpin without writing/opening on lift; right-click/long-press + Escape never poison semantic detail=1."""
    ui, page, control = pins_ui, pins_ui["page"], pins_ui["control"]
    pin = lambda pid: page.evaluate(PIN, pid)  # noqa: E731
    assert ui["client"].post("/api/projects/older/update", json={"pinned": True}).status_code == 200
    refresh(page)
    expect()(row(page, "older")).to_have_attribute("data-pinned", "")
    settle(page)
    if coarse(page):
        gx, gy = middle(pin("older")["glyph"])
        assert page.evaluate("([x, y]) => document.elementFromPoint(x, y).closest('.nav-project-row')?.dataset.projectId",
                             [gx, gy]) == "older"
        finger = {"pointerType": "touch", "isPrimary": True, "pointerId": 47, "clientX": gx, "clientY": gy}
        older_row = row(page, "older").locator('.nav-project-row')
        older_row.dispatch_event("pointerdown", finger)
        page.clock.run_for(500)
        expect()(menu(page).locator('[data-prm="pin"]')).to_have_text("Unpin")
        older_row.dispatch_event("pointerup", finger)
        settle(page)
        expect()(menu(page)).to_be_visible()
        assert page.locator('#project-panel').is_hidden() and control["writes"] == []
        page.keyboard.press("Escape")
        expect()(menu(page)).to_have_count(0)
        glyph, _ = focus_pin_via_keyboard(page, "older")
        glyph.dispatch_event("click", {"detail": 1})
        expect()(row(page, "older")).not_to_have_attribute("data-pinned", "")
        assert control["writes"] == [{"pinned": False}] and page.locator('#project-panel').is_hidden()
        return
    cdp = page.context.new_cdp_session(page)
    try:
        for hover_part in (None, "label", "pin"):
            if hover_part is None:
                page.mouse.move(ui["width"] - 40, 300)
            else:
                page.mouse.move(*middle(pin("older")[hover_part]))
            wait_row_paint(page, "older")
            hovered = pin("older")
            assert hovered["target"], (hover_part, hovered)
            page.evaluate(CLICK_TARGET)
            cdp_touch(cdp, *middle(hovered["pin"]))
            assert page.evaluate("window.__landed") == ["pin", "older"], hover_part
            settle(page)
            expect()(row(page, "older").locator('.nav-project-row')).to_have_attribute("aria-current", "page")
            assert control["writes"] == [] and registry.get_project(ui["root"], "older")["pinned"] is True
            page.locator('#project-panel-close').click()
            settle(page)
            expect()(page.locator('#project-panel')).to_be_hidden()
        page.mouse.move(*middle(pin("older")["pin"]))
        wait_row_paint(page, "older")
        gx, gy = middle(pin("older")["pin"])
        cdp_touch(cdp, gx, gy, hold_ms=500, page=page)
        settle(page)
        expect()(menu(page).locator('[data-prm="pin"]')).to_have_text("Unpin")
        assert page.locator('#project-panel').is_hidden() and control["writes"] == []
        page.keyboard.press("Escape")
        expect()(menu(page)).to_have_count(0)
    finally:
        cdp.detach()
    # Long-press -> Escape and right-click -> Escape must not leave stale press state for a no-pointer detail=1 click.
    page.mouse.move(ui["width"] - 40, 300)
    glyph, _ = focus_pin_via_keyboard(page, "older")
    glyph.dispatch_event("click", {"detail": 1})
    expect()(row(page, "older")).not_to_have_attribute("data-pinned", "")
    assert control["writes"] == [{"pinned": False}] and page.locator('#project-panel').is_hidden()
    page.mouse.click(*middle(pin("older")["label"]), button="right")
    expect()(menu(page)).to_be_visible()
    page.keyboard.press("Escape")
    expect()(menu(page)).to_have_count(0)
    page.mouse.move(ui["width"] - 40, 300)
    glyph, _ = focus_pin_via_keyboard(page, "older")
    glyph.dispatch_event("click", {"detail": 1})
    expect()(row(page, "older")).to_have_attribute("data-pinned", "")
    assert control["writes"] == [{"pinned": False}, {"pinned": True}] and page.locator('#project-panel').is_hidden()


# A right-to-left install: its language profile says `rtl`, which i18n.js applies to <html dir>. Answered from the
# page's first read, no later English answer can take the direction back.
RTL = {"language": "ar", "english": False, "revision": 1, "profile": {"direction": "rtl"}, "entries": {}}
# One frame of the mark column: Main's slot, mark, count and the words of its label beside each Project row's slot and
# mark, so both are measured in the same layout.
COLUMN = """() => {
    const box = (el) => { if (!el) return null; const r = el.getBoundingClientRect();
        return {left: r.left, right: r.right, top: r.top, bottom: r.bottom}; };
    const shown = (el) => Boolean(el) && !el.hidden && getComputedStyle(el).visibility !== 'hidden';
    const card = document.querySelector('.nav-row-main'), count = card.querySelector('.unread-badge');
    const mark = card.querySelector('.nav-mark'), label = card.querySelector('.nav-ouro-line1 > .nav-row-label');
    const words = document.createRange();
    words.selectNodeContents(label);
    const rows = {};
    for (const item of document.querySelectorAll('#nav-projects-list > .nav-project-item')) {
        const own = item.querySelector('.nav-project-row .status-mark');
        rows[item.dataset.navProjectId] = {item: box(item), slot: box(item.querySelector('.nav-mark-slot')),
                                           mark: shown(own) ? box(own) : null, shape: own?.dataset.shape || ''};
    }
    return {dir: document.documentElement.dir, lang: document.documentElement.lang, rows,
            main: {card: box(card), slot: box(card.querySelector('.nav-mark-slot')), shape: mark.dataset.shape,
                   mark: shown(mark) ? box(mark) : null, count: box(count), countText: count?.textContent || '',
                   label: box(label), words: box(words)}};
}"""


def column_frame(ui, rtl, evidence):
    """A fresh page in one direction with Main's count shown (a Project open over Main, three of Main's messages
    unread): Main's mark and both rows' marks measured in one frame, which is saved beside its picture."""
    control = ui["control"]
    control.update(i18n=RTL if rtl else None, main_revision=0, main_seen=0)
    # Pago Pago puts this morning-UTC stamp on the local day before; the rows share it, so their times are one width.
    # The page's clock starts on a Monday before its first paint, so a stamp three days before reads as its short
    # weekday (projects_nav.js whenOf, en-US in every language) whatever day the test runs, from that first paint.
    page = ui["make_page"](timezone_id="Pacific/Pago_Pago", now="2026-11-23T12:00:00Z")
    page.wait_for_function("dir => document.documentElement.dir === dir", arg="rtl" if rtl else "")
    first = page.evaluate("""() => ({now: new Date().toISOString().slice(0, 10), times: Object.fromEntries(
        [...document.querySelectorAll('#nav-projects-list > .nav-project-item')].map((item) =>
            [item.dataset.navProjectId, item.querySelector('.nav-project-time')?.textContent ?? '']))})""")
    assert first["now"] == "2026-11-23", first
    assert (first["times"]["friday"], first["times"]["older"], first["times"]["newer"]) == ("Fri", "9 Mar", "9 Mar"), first
    with page.expect_response(lambda response: urlparse(response.url).path == '/api/state'):
        refresh(page)  # Main's cursor is read and its revision seen: nothing of Main's is unread yet
    opened = row(page, "opened").locator('.nav-project-row')
    if coarse(page):
        opened.tap()
        settle(page)
        page.locator('#project-panel [data-mobile-nav-toggle]').click()
        page.locator('#primary-sidebar.open').wait_for()
    else:
        opened.click()
        page.mouse.move(ui["width"] / 2, 300)  # over the room, so no row draws its hover controls in the frame
    expect()(opened).to_have_attribute("aria-current", "page")
    control["main_revision"] = 3
    refresh(page)
    expect()(page.locator('.nav-row-main .unread-badge')).to_be_visible()
    settle(page)
    frame = page.evaluate(COLUMN)
    name = f"column-{'rtl' if rtl else 'ltr'}-{ui['tag']}"
    page.locator('#primary-sidebar').screenshot(path=str(evidence / f"{name}.png"), animations="disabled")
    (evidence / f"{name}.json").write_text(json.dumps(frame, ensure_ascii=False, indent=1))
    return page, frame


def assert_one_column(frame, rtl):
    """Main's mark stands in the Project rows' column: its slot over theirs, its moving dots where a working row's
    stand, its count one gap and one step before it, and the words of its label after Ouroboros's name."""
    main, rows = frame["main"], frame["rows"]
    assert frame["dir"] == ("rtl" if rtl else ""), frame
    assert main["mark"] and rows["older"]["mark"] and rows["newer"]["mark"] and main["countText"], frame
    assert (main["shape"], rows["older"]["shape"], rows["newer"]["shape"]) == ("moving", "moving", "dot"), frame
    for pid in ("older", "newer"):
        for side in ("left", "right"):
            assert main["slot"][side] == pytest.approx(rows[pid]["slot"][side], abs=0.5), (pid, side, frame)
    for side in ("left", "right"):
        assert main["mark"][side] == pytest.approx(rows["older"]["mark"][side], abs=0.5), (side, frame)
    gap = main["count"]["left"] - main["slot"]["right"] if rtl else main["slot"]["left"] - main["count"]["right"]
    assert gap == pytest.approx(12, abs=0.5), frame  # the line's 8px gap and the count's 4px step
    start = main["label"]["right"] - main["words"]["right"] if rtl else main["words"]["left"] - main["label"]["left"]
    assert start == pytest.approx(0, abs=1), frame


# A row's name beside its own Pin: the name's box, the letters it shows (its text clipped to that box), the shown glyph,
# the Pin's box and whether the keyboard's ring is drawn around it, the time's own text width and the room it keeps
# (its box and end margin).
NAME_BY_PIN = """(pid) => {
    const item = document.querySelector(`.nav-project-item[data-nav-project-id="${pid}"]`);
    const label = item.querySelector('.nav-row-label'), time = item.querySelector('.nav-project-time');
    const pin = item.querySelector(':scope > .nav-project-pin'), style = getComputedStyle(pin), words = document.createRange();
    const glyph = [...pin.querySelectorAll(':scope > svg')].find((svg) => getComputedStyle(svg).display !== 'none');
    const box = label.getBoundingClientRect(), g = glyph.getBoundingClientRect(), p = pin.getBoundingClientRect();
    words.selectNodeContents(label);
    const ink = [...words.getClientRects()].filter((r) => r.bottom > box.top && r.top < box.bottom)
        .map((r) => [Math.max(r.left, box.left), Math.min(r.right, box.right)]).filter(([l, r]) => r > l);
    const own = document.createRange();
    own.selectNodeContents(time);
    return {label: [box.left, box.right], glyph: [g.left, g.right], pin: [p.left, p.right],
            ink: [Math.min(...ink.map((i) => i[0])), Math.max(...ink.map((i) => i[1]))], time: time.textContent,
            natural: own.getBoundingClientRect().width,
            room: time.getBoundingClientRect().width + parseFloat(getComputedStyle(time).marginInlineEnd),
            drawn: style.opacity === '1' && style.visibility !== 'hidden', ring: style.outlineStyle !== 'none'};
}"""


def assert_drawn_pin_clears_the_name(page, pid, rtl, evidence, tag):
    """At rest a row's time keeps its own width (a pinned row's 20px for its resting glyph beside it), so no name loses
    width. While the Pin is drawn (a hovering pointer, the keyboard) the Pin stays put and the name, whose box clips
    every letter, ends where the one reserve leaves it: as at rest when the time already kept the drawn Pin clear, else
    the row's 1px border short of the Pin's box, which holds the glyph and the keyboard's ring."""
    rest, pinned = page.evaluate(NAME_BY_PIN, pid), row(page, pid).get_attribute("data-pinned") is not None
    assert rest["room"] == pytest.approx(rest["natural"] + (20 if pinned else 0), abs=0.5) and rest["drawn"] is pinned, rest
    for way in ("keyboard",) if coarse(page) else ("hover", "keyboard"):
        if way == "keyboard":
            focus_pin_via_keyboard(page, pid)
        else:
            row(page, pid).locator('.nav-row-label').hover()
            wait_row_paint(page, pid)
        drawn = page.evaluate(NAME_BY_PIN, pid)
        name = f"pin-room-{'rtl' if rtl else 'ltr'}-{pid}-{way}-{tag}"
        page.locator('#primary-sidebar').screenshot(path=str(evidence / f"{name}.png"), animations="disabled")
        (evidence / f"{name}.json").write_text(json.dumps({"rest": rest, "drawn": drawn}, ensure_ascii=False, indent=1))
        box = drawn["pin"]
        end = max(rest["label"][0], box[1] + 1) if rtl else min(rest["label"][1], box[0] - 1)
        assert drawn["drawn"] and box == rest["pin"], (pid, way, rest, drawn)
        assert drawn["label"][0 if rtl else 1] == pytest.approx(end, abs=0.5), (pid, way, end, rest, drawn)
        if way == "keyboard":
            page.evaluate("document.activeElement.blur()")
        else:
            page.mouse.move(page.viewport_size["width"] / 2, 300)
            wait_row_paint(page, pid)


def assert_time_edge_opens_row(page, pid, rtl, ui, evidence):
    """A pinned row's time ends inside its Pin's box (the box keeps the glyph's place and the button's size), yet every
    letter of the time is the row's: its last pixel, at rest and with the pointer on the row, hits the row, the glyph
    still hits the Pin under a mouse, and a press on that last pixel opens the room and leaves it pinned."""
    hit = """([x, y]) => { const el = document.elementFromPoint(x, y)?.closest('.nav-project-pin, .nav-project-row');
        return el ? el.className.split(' ').find((c) => c === 'nav-project-pin' || c === 'nav-project-row') : ''; }"""
    if not coarse(page):
        page.mouse.move(page.viewport_size["width"] / 2, 300)
        wait_row_paint(page, pid)
    state = page.evaluate(PIN, pid)
    x = (state["time"]["left"] + 0.5) if rtl else (state["time"]["right"] - 0.5)
    y = middle(state["time"])[1]
    assert page.evaluate("document.documentElement.dir") == ("rtl" if rtl else ""), state
    assert state["drawn"] and state["timeDrawn"] and state["pin"]["left"] < x < state["pin"]["right"], state
    facts = {"time": page.locator(f'.nav-project-item[data-nav-project-id="{pid}"] .nav-project-time').text_content(),
             "point": [x, y], "pin": state["pin"], "timeBox": state["time"], "glyph": state["glyph"],
             "rest": page.evaluate(hit, [x, y])}
    if not coarse(page):
        page.mouse.move(*middle(state["label"]))
        wait_row_paint(page, pid)
        facts["rowHover"] = page.evaluate(hit, [x, y])
        facts["glyphHit"] = page.evaluate(hit, list(middle(state["glyph"])))
        facts["timeDrawnOnHover"] = page.evaluate(PIN, pid)["timeDrawn"]
        assert facts["glyphHit"] == "nav-project-pin" and facts["timeDrawnOnHover"], facts
        assert facts["rowHover"] == "nav-project-row", facts
    assert facts["rest"] == "nav-project-row", facts
    name = f"time-edge-{'rtl' if rtl else 'ltr'}-{pid}-{ui['tag']}"
    page.locator('#primary-sidebar').screenshot(path=str(evidence / f"{name}.png"), animations="disabled")
    writes = len(ui["control"]["writes"])
    page.evaluate(CLICK_TARGET)
    if coarse(page):
        page.touchscreen.tap(x, y)
    else:
        page.mouse.move(x, y)
        page.mouse.down()
        page.mouse.up()
    facts["landed"] = page.evaluate("window.__landed")
    settle(page)
    facts["opened"] = row(page, pid).locator('.nav-project-row').get_attribute("aria-current") == "page"
    facts["pinned"] = registry.get_project(ui["root"], pid)["pinned"]
    facts["writes"] = ui["control"]["writes"][writes:]
    (evidence / f"{name}.json").write_text(json.dumps(facts, ensure_ascii=False, indent=1))
    assert facts["landed"] == ["row", pid] and facts["opened"], facts
    assert facts["pinned"] is True and facts["writes"] == [], facts


def test_pinned_row_layout_in_rtl(pins_ui, tmp_path):
    """Right to left (the install's language profile, i18n.js), Main's mark and count keep the Project rows' column,
    mirrored from the left-to-right frame, and a pinned row's glyph, shifted time, ⋯ and mark anchor to the inline
    end without colliding with long Arabic names, on desktop and phone. With no time, a short weekday or a date, and
    long names in either direction, a drawn Pin stays clear of every letter in both directions, and at rest no row
    loses width to it."""
    ui, control, root = pins_ui, pins_ui["control"], pins_ui["root"]
    evidence = producer_evidence_dir(tmp_path, "pin-rtl", env="OUROBOROS_BROWSER_EVIDENCE_OUT")
    registry.create_project(root, "opened", name="Opened project with a long Latin working title and عنوان عربي",
                            origin="owner_ui")
    registry.create_project(root, "quiet", name="مشروع هادئ بلا رسائل بعد وعنوانه العربي طويل with a Latin tail",
                            origin="owner_ui")
    registry.create_project(root, "quietpin", name="مشروع مثبت بلا رسائل بعد وعنوانه العربي طويل with a Latin tail",
                            origin="owner_ui")
    assert ui["client"].post("/api/projects/quietpin/update", json={"pinned": True}).status_code == 200
    registry.create_project(root, "friday", name="Friday room with a long Latin working title and عنوان عربي",
                            origin="owner_ui")
    assert ui["client"].post("/api/projects/older/update", json={
        "name": "مشروع بحث طويل جداً لاختبار اتجاه النص العربي في القائمة الجانبية", "pinned": True}).status_code == 200
    assert ui["client"].post("/api/projects/newer/update", json={
        "name": "مشروع حديث طويل لاختبار عدم تداخل التاريخ مع العنوان"}).status_code == 200
    control["message_times"] = {"newer": "2026-03-10T10:15:00Z", "older": "2026-03-10T10:15:00Z",
                                "friday": "2026-11-20T12:00:00Z"}  # Fri, beside the regular date and no time
    control["unread"].update({"older", "newer"})
    control["activities"] = [{"activity_id": "work-rtl", "project_id": "older",
                              "chat_id": registry.get_project(root, "older")["chat_id"],
                              "kind": "managed_task", "phase": "working"},
                             {"activity_id": "main", "project_id": "", "chat_id": 1,
                              "kind": "direct_chat", "phase": "thinking"}]
    ltr_page, ltr = column_frame(ui, False, evidence)
    page, rtl = column_frame(ui, True, evidence)
    assert_one_column(ltr, rtl=False)
    assert_one_column(rtl, rtl=True)
    # The column mirrors: Main's slot ends as far from the card's start-side edge in RTL as from its end in LTR.
    assert rtl["main"]["slot"]["left"] - rtl["main"]["card"]["left"] == pytest.approx(
        ltr["main"]["card"]["right"] - ltr["main"]["slot"]["right"], abs=0.5), (ltr, rtl)
    for at_rtl, at_page in ((False, ltr_page), (True, page)):
        for pid in ("opened", "quiet", "quietpin", "friday", "newer", "older"):
            assert_drawn_pin_clears_the_name(at_page, pid, at_rtl, evidence, ui["tag"])
    pin = lambda pid: page.evaluate(PIN, pid)  # noqa: E731
    pinned, unpinned = pin("older"), pin("newer")
    assert pinned["drawn"] and pinned["restGlyph"] and not pinned["actGlyph"] and pinned["target"] is (not coarse(page)), pinned
    assert pinned["timeDrawn"] and pinned["markDrawn"] and unpinned["markDrawn"], (pinned, unpinned)
    assert not unpinned["drawn"] and not unpinned["target"], unpinned
    assert end_inset(pinned, "time", rtl=True) - end_inset(unpinned, "time", rtl=True) == pytest.approx(20, abs=0.5), (pinned, unpinned)
    assert width_of(unpinned["label"]) - width_of(pinned["label"]) == pytest.approx(20, abs=0.5), (pinned, unpinned)
    assert pinned["time"]["left"] - pinned["glyph"]["right"] == pytest.approx(4 if coarse(page) else 3, abs=0.5), pinned
    assert pinned["label"]["left"] == pytest.approx(pinned["time"]["right"] + 8, abs=0.5), pinned
    assert pinned["label"]["left"] > pinned["glyph"]["right"] and pinned["label"]["left"] > pinned["pin"]["right"], pinned
    if coarse(page):
        assert end_inset(pinned, "pin", rtl=True) == pytest.approx(0, abs=0.5), pinned
        assert width_of(pinned["pin"]) == pytest.approx(44, abs=0.5), pinned
        assert end_inset(pinned, "glyph", rtl=True) == pytest.approx(5, abs=0.5), pinned
        for at_rtl, at_page in ((False, ltr_page), (True, page)):
            assert_time_edge_opens_row(at_page, "older", at_rtl, ui, evidence)
        return
    assert end_inset(pinned, "kebab", rtl=True) == pytest.approx(2, abs=0.5), pinned
    assert pinned["kebab"]["right"] <= pinned["pin"]["left"] + 0.01, pinned
    assert end_inset(pinned, "pin", rtl=True) == pytest.approx(30, abs=0.5), pinned
    assert width_of(pinned["pin"]) == pytest.approx(26, abs=0.5), pinned
    assert middle(pinned["glyph"])[0] - pinned["item"]["left"] == pytest.approx(42, abs=0.5), pinned
    page.mouse.move(*middle(pinned["pin"]))
    wait_row_paint(page, "older")
    hovered = pin("older")
    assert hovered["drawn"] and hovered["target"] and hovered["actGlyph"] and not hovered["restGlyph"], hovered
    assert hovered["glyph"] == pinned["glyph"] and hovered["time"] == pinned["time"], (pinned, hovered)
    for at_rtl, at_page in ((False, ltr_page), (True, page)):
        assert_time_edge_opens_row(at_page, "older", at_rtl, ui, evidence)


@pytest.mark.parametrize("pins_ui", [("webkit", 1280, "light")], ids=["webkit-1280-light-mouse"], indirect=True)
def test_keyboard_in_a_row_hides_its_mark_until_it_leaves_with_a_webkit_mouse(pins_ui, tmp_path):
    """WebKit with a mouse (hover, a fine pointer): a row's mark gives way at each keyboard stop in its row (Pin, name,
    ⋯ and back from its menu) and while hovered, and returns once the keyboard or the pointer leaves; a pinned row's
    mark stands beside its resting pin, and a press on its time's last letter opens the room. Chromium's mouse walks
    the same stops in the glyph test above."""
    ui, page, control = pins_ui, pins_ui["page"], pins_ui["control"]
    pin = lambda pid: page.evaluate(PIN, pid)  # noqa: E731
    assert ui["client"].post("/api/projects/older/update", json={"pinned": True}).status_code == 200
    control["message_times"] = {"older": "2026-03-10T10:15:00Z"}
    control["unread"].update({"older", "newer"})
    refresh(page)
    expect()(row(page, "older")).to_have_attribute("data-pinned", "")
    expect()(row(page, "newer").locator('.nav-project-row')).to_have_attribute("data-unread", "")
    settle(page)
    assert page.evaluate("matchMedia('(hover: hover) and (pointer: fine)').matches")
    rest = pin("older"), pin("newer")
    assert all(state["markDrawn"] and not state["kebabDrawn"] for state in rest), rest
    assert rest[0]["drawn"] and rest[0]["restGlyph"] and rest[0]["timeDrawn"], rest
    newer = row(page, "newer")
    for stop in ('.nav-project-pin', '.nav-project-row', '.nav-project-kebab'):
        keyboard_focus(page, newer.locator(stop))
        assert newer.locator(stop).evaluate("el => el === document.activeElement && el.matches(':focus-visible')"), stop
        here, there = pin("newer"), pin("older")
        assert here["kebabDrawn"] and not here["markDrawn"] and there["markDrawn"], (stop, here, there)
    page.keyboard.press("Enter")
    expect()(menu(page)).to_be_visible()
    page.keyboard.press("Escape")
    expect()(menu(page)).to_have_count(0)
    assert newer.locator('.nav-project-kebab').evaluate("el => el === document.activeElement && el.matches(':focus-visible')")
    assert pin("newer")["kebabDrawn"] and not pin("newer")["markDrawn"]
    keyboard_focus(page, row(page, "older").locator('.nav-project-pin'))  # the keyboard leaves newer for the pinned row
    left, pinned_kb = pin("newer"), pin("older")
    assert left["markDrawn"] and not left["kebabDrawn"], left
    assert pinned_kb["actGlyph"] and not pinned_kb["markDrawn"] and not pinned_kb["timeDrawn"], pinned_kb
    page.evaluate("document.activeElement.blur()")
    page.mouse.move(*middle(pin("newer")["label"]))
    wait_row_paint(page, "newer")
    hovered = pin("newer")
    assert hovered["kebabDrawn"] and hovered["drawn"] and not hovered["markDrawn"] and pin("older")["markDrawn"], hovered
    page.mouse.move(ui["width"] - 40, 300)
    wait_row_paint(page, "newer")
    assert pin("newer")["markDrawn"] and not pin("newer")["kebabDrawn"] and pin("older")["markDrawn"]
    assert_time_edge_opens_row(page, "older", False, ui, producer_evidence_dir(tmp_path, "pin-webkit-mouse",
                                                                                env="OUROBOROS_BROWSER_EVIDENCE_OUT"))


# ---- Rooms beyond the feed: Find a project, Show more and a reference's own room --------------------------------

TOPICS = ("garden irrigation", "reading list", "bicycle repairs", "language practice", "kitchen shelves",
          "photo archive", "travel notes", "household budget", "chess openings", "piano practice")


def seed_archive(root, count=498):
    """``count`` invented rooms older than the fixture's two, written once beside them: one pinned, one closed, one
    deleting, one tombstoned and one with a Unicode name. Returns every listable id, newest first."""
    path = root / "state/projects.json"
    data = json.loads(path.read_text())
    template = next(row for row in data["projects"] if row["id"] == "older")
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    for i in range(count):
        data["projects"].append({**template, "id": f"archive-{i:03}", "chat_id": 40000 + i, "pinned": i == 100,
                                 "name": f"Archive {i:03} · {TOPICS[i % len(TOPICS)]}",
                                 "last_active_at": (start + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ")})
    rows = {row["id"]: row for row in data["projects"]}
    rows["archive-010"]["closed"] = {"at": "2026-02-01T00:00:00Z", "by": "owner", "note": "Closed by owner."}
    rows["archive-003"]["lifecycle"] = "deleting"
    rows["archive-004"]["lifecycle"] = "tombstoned"
    rows["archive-001"]["name"] = "Ünïcode Café — Плановые Работы"
    path.write_text(json.dumps(data))
    return [row["id"] for row in registry.projects_summary(root, limit=None)]


def shown_ids(page):
    return [pid for pid in order(page) if pid]


def reads_of(reads, path, **query):
    """The requests to ``path`` whose query names every given parameter with that value."""
    return [url for url in reads if urlparse(url).path == path
            and all(parse_qs(urlparse(url).query).get(key) == [value] for key, value in query.items())]


def no_read_acknowledged(ui):
    return not [body for path, body in ui["posts"] if path == "/api/ui/preferences" and "project_seen_revision" in (body or {})]


def click_reference(page, project):
    """A Project reference as project_reference.js builds it ({id, name} only), pressed by a key."""
    page.evaluate("""async (project) => {
        const { projectReference } = await import('/static/modules/project_reference.js');
        const button = projectReference({ id: project.id, name: project.name });
        document.body.append(button);
        button.click();
        button.remove();
    }""", {"id": project["id"], "name": project["name"]})


def settle(page):
    """Let the shell's motion and its frame callbacks run under the installed clock."""
    page.clock.run_for(1000)
    page.evaluate("""() => Promise.all(
        document.getAnimations()
            .filter((a) => a.playState === 'running' && Number.isFinite(a.effect?.getTiming?.().iterations))
            .map((a) => a.finished.catch(() => null))
    )""")
    page.evaluate("new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
