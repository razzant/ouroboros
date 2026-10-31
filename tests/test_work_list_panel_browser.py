"""The room's work list is one floating sheet of its own material (DESIGN "Room work status", principle 2).

The real SPA (room_work.js, room_work.css, ui.css) in Chromium and WebKit over the controlled transport of
``tests/test_room_work_browser.py``: Main and a Project, light and dark, at a window and at a phone's width, with the
awkward names and actions the product really shows. Four oracles; the geometry and material ones are falsified in-test
(a translucent sheet and a wrong edge must fail them) and the rest by the recorded source mutations:

* geometry: the sheet hangs under the capsules, from the status capsule's left edge to the bar's last control, a gap
  below the capsule, and follows the bar when it resizes;
* material: its own opaque sheet, float shadow and sheet radius, the same whatever its ARIA role, and no pixel of what
  lies under it shows through (a differential screenshot over deliberately loud content), nor through a row's menu;
* containment: names clamp, targets stay inside the sheet and reachable, nothing overflows sideways;
* behaviour: Escape, an outside press and leaving close it and return focus; reduced motion draws no entrance or fold;
  Main's history note waits behind the open list.

Screenshots and the facts read in the page go to ``OUROBOROS_UI_EVIDENCE_DIR`` when it is set.
"""
from __future__ import annotations

import io
import json
import time
from urllib.parse import urlparse

import pytest
from PIL import Image, ImageChops

from tests.test_room_work_browser import (
    LONGER_NAME, LONG_NAME, PROJECT, Transport, activity, capture, evidence, facts, open_project, progress, reading,
    room_ui, settle_frames, settle_unfold, subscription_ui, trigger, wait_facts,
)

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
__all__ = ["room_ui", "subscription_ui"]

# One read of the open sheet, its bar and the capsules it hangs from, in CSS pixels.
_SHEET = """(scope) => {
    const root = scope === 'main' ? document.querySelector('#page-chat') : document.querySelector('#project-panel');
    const bar = scope === 'main' ? root.querySelector('.chat-page-header') : root.querySelector('.project-panel-bar');
    const sheet = root.querySelector('.rw-shade:not(.cx-shade)');  // the list, not the consciousness chip's Shade
    const capsule = bar.querySelector('.app-page-title-row');
    const box = (el) => { const r = el.getBoundingClientRect(); return { x: r.x, y: r.y, r: r.right, b: r.bottom, w: r.width, h: r.height }; };
    const shown = (el) => el.getClientRects().length > 0 && getComputedStyle(el).visibility !== 'hidden';
    const controls = [...bar.querySelectorAll('button, summary')].filter((el) => shown(el) && !capsule.contains(el)
        && !el.closest('.rw-shade-anchor'));
    const style = getComputedStyle(sheet);
    const probe = (css) => { const el = document.body.appendChild(document.createElement('i'));
        el.style.cssText = css; const s = getComputedStyle(el);
        const out = { color: s.backgroundColor, image: s.backgroundImage, shadow: s.boxShadow, radius: s.borderTopLeftRadius };
        el.remove(); return out; };
    const rows = [...sheet.querySelectorAll('.rw-row')].map((row) => ({ id: row.dataset.id, ...box(row),
        name: (() => { const n = row.querySelector('.rw-name'); return { ...box(n), clipped: n.scrollHeight > n.clientHeight + 1,
            wide: n.scrollWidth > n.clientWidth + 1 }; })(),
        controls: [...row.querySelectorAll('.rw-pill, .rw-more')].filter(shown).map((el) => ({ cls: el.className, ...box(el) })) }));
    const alpha = (c) => { const m = c.match(/rgba?\\(([^)]+)\\)/); const p = m ? m[1].split(/[ ,/]+/).filter(Boolean) : [];
        return p.length > 3 ? parseFloat(p[3]) : 1; };
    return {
        bar: box(bar), capsule: box(capsule), sheet: box(sheet), end: Math.max(...controls.map((el) => el.getBoundingClientRect().right)),
        rows, viewport: { w: innerWidth, h: innerHeight },
        overflowX: sheet.scrollWidth - sheet.clientWidth, pageOverflow: document.documentElement.scrollWidth - innerWidth,
        material: { color: style.backgroundColor, image: style.backgroundImage, shadow: style.boxShadow,
            radii: [style.borderTopLeftRadius, style.borderTopRightRadius, style.borderBottomRightRadius, style.borderBottomLeftRadius],
            alpha: alpha(style.backgroundColor), backdrop: style.backdropFilter || style.webkitBackdropFilter || 'none' },
        expected: { sheet: probe('background: var(--sheet-float); box-shadow: var(--elev-float); border-radius: var(--radius-sheet)') },
        reachable: [...sheet.querySelectorAll('.rw-main, .rw-more, .rw-pill')].filter(shown).every((el) => {
            const r = el.getBoundingClientRect(); const hit = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2);
            return Boolean(hit) && (hit === el || el.contains(hit)); }),
    };
}"""

# Loud content under the sheet: every bubble and the feed itself in colours no neutral sheet can be mistaken for.
_LOUD = """() => {
    const style = document.createElement('style'); style.id = 'loud-under-the-list';
    style.textContent = `.chat-bubble, .chat-bubble * { background: #ff00ff !important; color: #00ff00 !important; }
        .chat-messages, .chat-messages * { color: #00ff00 !important; }
        .chat-messages { background: #ffff00 !important; }`;
    document.head.appendChild(style);
}"""
_CALM = "() => document.getElementById('loud-under-the-list')?.remove()"


def _transport(page):
    history = {1: [*reading(14), progress("rw-live", "Comparing the cited pages.", 30, suggested_name=LONGER_NAME),
                   progress("rw-pausing", "Collecting appendix evidence.", 31, suggested_name="Collect appendix evidence"),
                   *reading(6, 32, "Later")],
               42: [*reading(12, 0, "Project"), progress("p-live", "Reading the archive.", 30,
                                                          suggested_name=LONG_NAME)]}
    return Transport(page, history=history, projects=[PROJECT], details={
        "rw-far": {"task_id": "rw-far", "status": "running", "suggested_name": "Index the archived notes"}},
        activities=[activity("rw-live"), activity("rw-pausing", "budget_paused", pause_cause="owner"),
                    activity("rw-far", "queued"),
                    activity("p-live", chat_id=42, project_id=PROJECT["id"]),
                    activity("p-hold", "queued", chat_id=42, project_id=PROJECT["id"],
                             project_admission_hold={"label": "Waiting for Project access"})])


def _open(ui, *, width, height, theme, room, **page_options):
    """The app in its own page (its window, theme and motion preference are the test's), the room's list still closed."""
    page = ui["new_page"](viewport={"width": width, "height": height}, color_scheme=theme, **page_options)
    transport = _transport(page)
    origin = urlparse(ui["url"]).netloc
    page.route("**/*", lambda route: route.fallback() if urlparse(route.request.url).netloc == origin else route.abort())
    page.add_init_script(f"localStorage.setItem('ouroboros.theme', {json.dumps(theme)})")
    page.goto(ui["url"] + "/")
    page.wait_for_selector("#page-chat .rw-trigger", state="attached")
    deadline = time.monotonic() + 15
    while not transport.sockets and time.monotonic() < deadline:
        page.wait_for_timeout(50)
    assert transport.sockets, "the page never opened its socket"
    transport.refresh()
    if room == "project":
        open_project(page, PROJECT)
        page.wait_for_function("getComputedStyle(document.querySelector('#project-panel')).opacity === '1'")
    else:
        page.locator("#chat-messages").evaluate("el => { el.scrollTop = Math.round(el.scrollHeight * 0.5); }")
    wait_facts(page, room, lambda f: f["tone"] == "live", f"the {room} live control")
    page.wait_for_timeout(250)
    return page, transport


def _open_list(page, room):
    trigger(page, room).click()
    wait_facts(page, room, lambda f: len(f["rows"]) == (3 if room == "main" else 2), f"the {room} rows")
    settle_unfold(page, room)
    return page.evaluate(_SHEET, room)


def assert_hangs_under_the_capsules(facts, *, tag):
    sheet, capsule = facts["sheet"], facts["capsule"]
    assert abs(sheet["x"] - capsule["x"]) <= 1, ("the sheet starts at the status capsule's left edge", tag, sheet, capsule)
    assert abs(sheet["r"] - facts["end"]) <= 1, ("the sheet ends at the bar's last control", tag, sheet, facts["end"])
    assert abs((sheet["y"] - capsule["b"]) - 8) <= 1, ("a gap below the capsule", tag, sheet, capsule)
    bar = facts["bar"]
    assert sheet["x"] >= bar["x"] - 0.5 and sheet["r"] <= bar["x"] + bar["w"] + 0.5, ("inside the bar's span", tag)
    assert sheet["x"] >= 0 and sheet["r"] <= facts["viewport"]["w"] and sheet["b"] <= facts["viewport"]["h"], tag
    assert facts["overflowX"] <= 0 and facts["pageOverflow"] <= 0, ("nothing overflows sideways", tag, facts)


def assert_own_material(facts, *, tag):
    material, expected = facts["material"], facts["expected"]["sheet"]
    assert material["alpha"] == 1, ("an opaque colour layer: no bubble shows through", tag, material)
    assert material["image"] == expected["image"] and material["color"] == expected["color"], (tag, material, expected)
    assert material["shadow"] == expected["shadow"] != "none", ("the float shadow", tag, material)
    assert set(material["radii"]) == {expected["radius"]} and expected["radius"] == "24px", ("the sheet radius", tag, material)


def _pixels(page, clip):
    return Image.open(io.BytesIO(page.screenshot(clip=clip, animations="disabled"))).convert("RGB")


def _band(facts):
    """A strip of the sheet's own bottom padding: no word, mark or control is ever drawn there."""
    sheet = facts["sheet"]
    return {"x": sheet["x"] + 40, "y": sheet["b"] - 9, "width": max(8, sheet["w"] - 80), "height": 5}


def assert_nothing_shows_through(page, facts):
    """The strip looks the same over calm content and over loud content: the sheet is opaque in pixels, not by a flag."""
    page.evaluate(_CALM)
    settle_frames(page)
    calm = _pixels(page, _band(facts))
    page.evaluate(_LOUD)
    settle_frames(page)
    try:
        loud = _pixels(page, _band(facts))
    finally:
        page.evaluate(_CALM)
    difference = max(high for _, high in ImageChops.difference(calm, loud).getextrema())
    assert difference <= 1, ("content under the list shows through it", difference)


@pytest.mark.parametrize("room", ["main", "project"])
@pytest.mark.parametrize("theme", ["dark", "light"])
@pytest.mark.parametrize("width,height", [(1100, 722), (390, 844)])
def test_the_list_is_its_own_opaque_sheet_under_the_capsules(room_ui, width, height, theme, room):
    page, _transport = _open(room_ui, width=width, height=height, theme=theme, room=room)
    tag = f"{room}-{width}x{height}-{theme}-{room_ui['engine']}"
    facts = _open_list(page, room)
    capture(page, f"sheet-{tag}")
    evidence(f"sheet-{tag}", facts)
    assert_hangs_under_the_capsules(facts, tag=tag)
    assert_own_material(facts, tag=tag)
    assert facts["reachable"], ("every row target is the element hit at its centre", tag)
    assert_nothing_shows_through(page, facts)
    # The material is the surface's own, never its role's: the same sheet whatever ARIA role it declares.
    sheet = page.locator(f"{'#page-chat' if room == 'main' else '#project-panel'} .rw-shade:not(.cx-shade)")
    for role in ("menu", "listbox", "group"):
        sheet.evaluate("(el, role) => el.setAttribute('role', role)", role)
        assert page.evaluate(_SHEET, room)["material"] == facts["material"], ("role chooses no paint", role, tag)
    sheet.evaluate("el => el.setAttribute('role', 'dialog')")
    # Falsify the oracles on this very consumer: a translucent float sheet and a wrong edge must both fail them.
    page.add_style_tag(content=".rw-shade { background: rgba(32, 32, 34, 0.92) !important; }")
    settle_frames(page)
    translucent = page.evaluate(_SHEET, room)
    with pytest.raises(AssertionError, match="opaque colour layer"):
        assert_own_material(translucent, tag=tag)
    with pytest.raises(AssertionError, match="shows through"):
        assert_nothing_shows_through(page, translucent)
    page.add_style_tag(content=".rw-shade-anchor { --rw-left: 0px !important; }")
    settle_frames(page)
    with pytest.raises(AssertionError, match="left edge"):
        assert_hangs_under_the_capsules(page.evaluate(_SHEET, room), tag=tag)


# Under a menu lies the list itself as well as the room: both turn loud, so only the menu's own backing is read.
_LOUD_UNDER_MENU = """() => {
    const style = document.createElement('style'); style.id = 'loud-under-the-list';
    style.textContent = `.chat-bubble, .chat-bubble *, .rw-shade, .rw-shade * { background: #ff00ff !important; color: #00ff00 !important; }
        .chat-messages, .chat-messages * { color: #00ff00 !important; }
        .chat-messages { background: #ffff00 !important; }`;
    document.head.appendChild(style);
}"""


def assert_menu_holds_its_words(page, clip):
    """The menu's inside, its words included, reads the same over calm and over loud content."""
    page.evaluate(_CALM)
    settle_frames(page)
    calm = _pixels(page, clip)
    page.evaluate(_LOUD_UNDER_MENU)
    settle_frames(page)
    try:
        loud = _pixels(page, clip)
    finally:
        page.evaluate(_CALM)
    difference = max(high for _, high in ImageChops.difference(calm, loud).getextrema())
    assert difference <= 1, ("content under the menu shows through it", difference)
    return difference


@pytest.mark.parametrize("room", ["main", "project"])
@pytest.mark.parametrize("theme", ["dark", "light"])
def test_a_rows_menu_is_backed_like_every_popup(room_ui, theme, room):
    """A row's own menu (task_control_menu.js, no ``.ui-popup``) floats over the list and the room; its words keep the
    popups' opaque backing, and the oracle fails on the translucent float paint the menu wore before."""
    page, _transport = _open(room_ui, width=1100, height=722, theme=theme, room=room)
    tag = f"{room}-{theme}-{room_ui['engine']}"
    _open_list(page, room)
    root = "#page-chat" if room == "main" else "#project-panel"
    page.locator(f"{root} .rw-row[data-id='{'rw-live' if room == 'main' else 'p-live'}'] .rw-more").click()
    menu = page.locator("body > .task-control-menu")
    menu.wait_for(state="visible")
    page.mouse.move(0, 0)
    page.wait_for_function("() => !document.querySelector('body > .task-control-menu').getAnimations()"
                           ".some((a) => a.playState === 'running')")
    settle_frames(page)
    box = menu.bounding_box()
    clip = {"x": box["x"] + 8, "y": box["y"] + 8, "width": box["width"] - 16, "height": box["height"] - 16}
    page.evaluate(_LOUD_UNDER_MENU)
    settle_frames(page)
    capture(page, f"row-menu-loud-{tag}")
    page.evaluate(_CALM)
    measured = {"clip": clip, "difference": assert_menu_holds_its_words(page, clip),
                "paint": menu.evaluate("el => { const s = getComputedStyle(el); return [s.backgroundColor, s.backgroundImage]; }")}
    evidence(f"row-menu-{tag}", measured)
    page.add_style_tag(content=".task-control-menu { background: rgba(32, 32, 34, 0.92) !important; }")
    settle_frames(page)
    with pytest.raises(AssertionError, match="shows through"):
        assert_menu_holds_its_words(page, clip)


@pytest.mark.parametrize("room", ["main", "project"])
@pytest.mark.parametrize("width,height", [(1100, 722), (390, 844)])
def test_awkward_names_and_actions_stay_inside_the_sheet(room_ui, width, height, room):
    page, _transport = _open(room_ui, width=width, height=height, theme="dark", room=room)
    facts = _open_list(page, room)
    sheet = facts["sheet"]
    assert facts["rows"], facts
    for row in facts["rows"]:
        assert row["x"] >= sheet["x"] - 0.5 and row["r"] <= sheet["r"] + 0.5, ("a row stays in the sheet", row, sheet)
        # The name wraps in the language the product is used in (Cyrillic, German compounds) and clamps at two lines.
        assert not row["name"]["wide"], ("a long name wraps, it never widens its row", row)
        for control in row["controls"]:
            assert control["x"] >= row["x"] - 0.5 and control["r"] <= row["r"] + 0.5, ("an action stays in its row", control, row)
            assert control["r"] <= sheet["r"] + 0.5 and control["x"] >= sheet["x"] - 0.5, (control, sheet)
    if room == "project" or width == 390:
        assert any(row["name"]["clipped"] for row in facts["rows"]), "the longest real name is clamped, not shown whole"
    assert facts["reachable"] and facts["overflowX"] <= 0 and facts["pageOverflow"] <= 0, facts


@pytest.mark.parametrize("room", ["main", "project"])
def test_the_open_list_follows_its_capsules_when_the_bar_resizes(room_ui, room):
    page, _transport = _open(room_ui, width=1100, height=722, theme="light", room=room)
    wide = _open_list(page, room)
    assert_hangs_under_the_capsules(wide, tag=f"{room}-wide")
    page.set_viewport_size({"width": 820, "height": 722})
    settle_frames(page)
    page.wait_for_timeout(200)
    narrow = page.evaluate(_SHEET, room)
    assert_hangs_under_the_capsules(narrow, tag=f"{room}-narrow")
    assert narrow["sheet"]["w"] < wide["sheet"]["w"], ("the sheet narrowed with its bar", wide["sheet"], narrow["sheet"])
    page.set_viewport_size({"width": 1100, "height": 722})
    settle_frames(page)
    page.wait_for_timeout(200)
    assert_hangs_under_the_capsules(page.evaluate(_SHEET, room), tag=f"{room}-wide-again")


@pytest.mark.parametrize("room", ["main", "project"])
@pytest.mark.parametrize("reduced", [True, False])
def test_escape_outside_press_and_reduced_motion(room_ui, room, reduced):
    page, transport = _open(room_ui, width=1100, height=722, theme="dark", room=room,
                            reduced_motion="reduce" if reduced else "no-preference")
    root = "#page-chat" if room == "main" else "#project-panel"
    trigger(page, room).click()
    wait_facts(page, room, lambda f: not f["surfaceHidden"], "open")
    entrance = page.locator(f"{root} .rw-shade:not(.cx-shade)").evaluate("el => getComputedStyle(el).animationName")
    assert entrance == ("none" if reduced else "rw-unfold"), ("the entrance follows the motion preference", entrance)
    settle_unfold(page, room)
    page.keyboard.press("Escape")
    wait_facts(page, room, lambda f: f["surfaceHidden"], "closed", timeout=2 if reduced else 20)
    assert facts(page, room)["active"]["isTrigger"], "Escape returns focus to the control that opened the list"
    # An outside press closes it too, and the press is not swallowed by anything it lands on.
    trigger(page, room).click()
    wait_facts(page, room, lambda f: not f["surfaceHidden"], "reopened")
    settle_unfold(page, room)
    page.mouse.click(700 if room == "main" else 900, 520)
    wait_facts(page, room, lambda f: f["surfaceHidden"], "an outside press closed it")
    assert transport.posts == [], "closing a list posts nothing: no decision, no task control"


def test_main_history_note_waits_behind_the_open_list(room_ui):
    page, _transport = _open(room_ui, width=1100, height=722, theme="dark", room="main")
    note = page.locator("#page-chat .chat-page-note > *").first
    note.wait_for(state="visible")
    assert note.is_visible()
    _open_list(page, "main")
    visibility = page.locator("#page-chat .chat-page-note").evaluate("el => getComputedStyle(el).visibility")
    assert visibility == "hidden", "the note hangs where the open list lies, so it waits behind it"
    page.keyboard.press("Escape")
    wait_facts(page, "main", lambda f: f["surfaceHidden"], "closed")
    assert note.is_visible(), "and comes back when the list closes"
    evidence(f"note-{room_ui['engine']}", {"visibility": visibility})
