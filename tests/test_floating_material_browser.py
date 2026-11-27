"""A popup's material is its own, whatever its ARIA role (DESIGN principle 2).

The real Skills page in Chromium and WebKit: the Official mark's explanation (a ``role="dialog"`` popup, which once had
no background and no shadow because the popup material was selected by ``role="menu"``) is compared with a real
menu of the same page, in both themes at a window and at a phone's width. The oracle is falsified in-test so it can fail.
On a short screen the same explanation scrolls only its body, from the Close it opens on back to its first words.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pytest
from PIL import Image, ImageChops

from tests._skills_page_harness import URL, fixture, open_skills, skills_browser as _skills_browser

pytestmark = [pytest.mark.ui_browser]
skills_browser = _skills_browser

CONFIRMED_SOURCES = fixture("skill_sources.json")
for _row in CONFIRMED_SOURCES["sources"]:
    _row["confirmed"] = True

_MATERIAL = """(el) => { const s = getComputedStyle(el); const r = el.getBoundingClientRect();
    return { color: s.backgroundColor, image: s.backgroundImage, shadow: s.boxShadow, radius: s.borderTopLeftRadius,
             border: `${s.borderTopWidth} ${s.borderTopStyle} ${s.borderTopColor}`, ink: s.color,
             backdrop: s.backdropFilter || s.webkitBackdropFilter || 'none', role: el.getAttribute('role'),
             box: { x: r.x, y: r.y, r: r.right, b: r.bottom }, vw: innerWidth, vh: innerHeight }; }"""


# Finite animations are over (the Skills page keeps looping status marks alive for as long as it is open).
_SETTLED = """() => [...document.getAnimations()].every(a => a.effect?.getTiming().iterations === Infinity || a.playState !== 'running')"""


def _paint(facts: dict) -> dict:
    return {key: facts[key] for key in ("color", "image", "shadow", "radius", "border", "ink", "backdrop")}


def assert_floats(facts: dict, tag: str):
    assert facts["color"] not in {"rgba(0, 0, 0, 0)", "transparent"}, ("a floating popup is never transparent", tag, facts)
    assert facts["shadow"] != "none", ("it floats on the float shadow", tag, facts)
    assert facts["radius"] == "18px", ("the object radius", tag, facts)


@pytest.mark.parametrize("browser_name", ["chromium", "webkit"])
@pytest.mark.parametrize("theme", ["dark", "light"])
@pytest.mark.parametrize("width", [1280, 390])
def test_official_explanation_wears_the_same_material_as_a_menu(skills_browser, browser_name, theme, width):
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    skills = None
    tag = f"{browser_name}-{width}-{theme}"
    try:
        page = browser.new_page(viewport={"width": width, "height": 900}, has_touch=width == 390)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        listing = fixture("extensions_v4.json")["skills"]
        skills = open_skills(page, extensions=listing, sources=CONFIRMED_SOURCES, catalog=fixture("skill_catalog.json"))
        page.goto(URL, wait_until="networkidle")
        page.evaluate("theme => document.documentElement.dataset.theme = theme", theme)
        card = page.locator('.skills-card[data-skill="weather"]')
        card.wait_for()
        # The Official explanation: a dialog-role popup, opened beside its mark.
        card.locator('[data-official-mark]').click()
        explanation = page.locator('.skills-mark-popup:not([data-leaving])')
        explanation.wait_for()
        page.wait_for_function(_SETTLED)
        official = explanation.evaluate(_MATERIAL)
        assert official["role"] == "dialog"
        assert_floats(official, tag)
        out = os.environ.get("OUROBOROS_UI_EVIDENCE_DIR")
        if out:
            Path(out).mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(Path(out) / f"official-{tag}.png"))
        assert official["box"]["x"] >= 0 and official["box"]["r"] <= official["vw"], ("inside the window", tag, official)
        # Compare actual pixels over changing content, including behind the popup's words.
        # Computed alpha and equality with a menu alone cannot prove readable separation.
        bounds = explanation.bounding_box()
        clip = {"x": bounds["x"] + 20, "y": bounds["y"] + 20,
                "width": bounds["width"] - 40, "height": bounds["height"] - 40}
        calm = Image.open(io.BytesIO(page.screenshot(clip=clip))).convert("RGB")
        loud_style = page.add_style_tag(content=".skills-card, .skills-card * { background: #ff00ff !important; color: #00ff00 !important; }")
        page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
        loud = Image.open(io.BytesIO(page.screenshot(clip=clip))).convert("RGB")
        loud_style.evaluate("el => el.remove()")
        difference = max(high for _, high in ImageChops.difference(calm, loud).getextrema())
        assert difference <= 1, ("content shows through the popup's words", tag, difference)
        page.keyboard.press("Escape")
        explanation.wait_for(state="detached")
        # A real menu of the same page: its material is the one the explanation must wear.
        trigger = page.locator('.skills-card [data-skill-menu-trigger]').first
        trigger.click()
        menu = page.locator('.ui-popup[role="menu"]:not([data-leaving])').first
        menu.wait_for()
        page.wait_for_function(_SETTLED)
        real_menu = menu.evaluate(_MATERIAL)
        assert_floats(real_menu, tag)
        assert _paint(official) == _paint(real_menu), ("one popup material for every role", tag, official, real_menu)
        # Role chooses layout, never paint: every role on a bare popup resolves to the same material.
        probe = page.evaluate("""(roles) => roles.map((role) => { const el = document.body.appendChild(document.createElement('div'));
            el.className = 'ui-popup'; if (role) el.setAttribute('role', role); el.textContent = 'probe'; el.style.visibility = 'hidden';
            const s = getComputedStyle(el);
            const out = { role, color: s.backgroundColor, image: s.backgroundImage, shadow: s.boxShadow, radius: s.borderTopLeftRadius,
                          border: `${s.borderTopWidth} ${s.borderTopStyle} ${s.borderTopColor}`, ink: s.color,
                          backdrop: s.backdropFilter || s.webkitBackdropFilter || 'none' };
            el.remove(); return out; })""", ["", "menu", "dialog", "listbox", "tooltip"])
        for found in probe:
            assert _paint(found) == _paint(real_menu), ("a role selects no paint", found["role"], tag, found, real_menu)
        # The listbox of the model chooser is the same popup: its own rule adds layout only.
        listbox = page.evaluate("""() => { const el = document.body.appendChild(document.createElement('div'));
            el.className = 'ui-popup ui-model-options'; el.setAttribute('role', 'listbox'); const s = getComputedStyle(el);
            const out = { color: s.backgroundColor, image: s.backgroundImage, shadow: s.boxShadow, radius: s.borderTopLeftRadius,
                          border: `${s.borderTopWidth} ${s.borderTopStyle} ${s.borderTopColor}`, ink: s.color,
                          backdrop: s.backdropFilter || s.webkitBackdropFilter || 'none' };
            el.remove(); return out; }""")
        assert _paint(listbox) == _paint(real_menu), ("the model list wears it too", tag, listbox)
        # Falsify the oracle on this consumer: a popup whose paint a role-keyed rule took away must fail it.
        page.add_style_tag(content='.ui-popup:not([role="menu"]) { background: transparent !important; box-shadow: none !important; }')
        stripped = page.evaluate("""() => { const el = document.body.appendChild(document.createElement('div')); el.className = 'ui-popup';
            el.setAttribute('role', 'dialog'); const s = getComputedStyle(el);
            const out = { color: s.backgroundColor, shadow: s.boxShadow, radius: s.borderTopLeftRadius }; el.remove(); return out; }""")
        with pytest.raises(AssertionError, match="never transparent"):
            assert_floats(stripped, tag)
        assert errors == [], errors
        json_out = os.environ.get("OUROBOROS_UI_EVIDENCE_DIR")
        if json_out:
            (Path(json_out) / f"official-{tag}.json").write_text(
                json.dumps({"official": official, "menu": real_menu, "probes": probe}, indent=1), encoding="utf-8")
    finally:
        if skills:
            skills.close(browser)
        else:
            browser.close()


# Only a popup's body scrolls (DESIGN principle 9: fade bodies, not shells); its shell clips. A body the shell does not
# bound grows past it, and the words it hides cannot be scrolled to.
_REACH = """(popup) => { const body = popup.querySelector(':scope > .ui-popup-body'), s = getComputedStyle(popup);
    const box = popup.getBoundingClientRect(), close = popup.querySelector('[data-popup-close]');
    const inner = { top: box.top + parseFloat(s.borderTopWidth) + parseFloat(s.paddingTop),
                    bottom: box.bottom - parseFloat(s.borderBottomWidth) - parseFloat(s.paddingBottom) };
    return { ...inner, boxTop: box.top, boxBottom: box.bottom, vh: innerHeight, shellScrolled: popup.scrollTop,
             bodyBottom: body ? body.getBoundingClientRect().bottom : null, sideways: body ? body.scrollWidth - body.clientWidth : null,
             bodyScrolled: body ? body.scrollTop : null, range: body ? body.scrollHeight - body.clientHeight : null,
             wordsTop: popup.querySelector('p').getBoundingClientRect().top,
             closeTop: close.getBoundingClientRect().top, closeBottom: close.getBoundingClientRect().bottom,
             closeFocused: document.activeElement === close }; }"""


@pytest.mark.parametrize("browser_name", ["chromium", "webkit"])
@pytest.mark.parametrize("width,height", [(667, 320), (390, 244)])
def test_official_explanation_on_a_short_screen_scrolls_its_words_to_close(skills_browser, browser_name, width, height):
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    skills = None
    tag = f"{browser_name}-{width}x{height}"
    out = os.environ.get("OUROBOROS_UI_EVIDENCE_DIR")

    def shot(page, state):
        if out:
            Path(out).mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(Path(out) / f"official-short-{tag}-{state}.png"))

    try:
        page = browser.new_page(viewport={"width": width, "height": height})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        listing = fixture("extensions_v4.json")["skills"]
        skills = open_skills(page, extensions=listing, sources=CONFIRMED_SOURCES, catalog=fixture("skill_catalog.json"))
        page.goto(URL, wait_until="networkidle")
        mark = page.locator('.skills-card[data-skill="weather"] [data-official-mark]')
        mark.wait_for()
        # The mark in the middle of its list: the explanation has the half of the window it opens into.
        mark.evaluate("el => el.scrollIntoView({block: 'center'})")
        mark.click()
        explanation = page.locator('.skills-mark-popup:not([data-leaving])')
        explanation.wait_for()
        page.wait_for_function(_SETTLED)
        opened = explanation.evaluate(_REACH)
        shot(page, "opened")
        assert opened["closeFocused"], ("it opens on Close, its one item", tag, opened)
        assert opened["boxTop"] >= 0 and opened["boxBottom"] <= opened["vh"], ("inside the window", tag, opened)
        assert opened["shellScrolled"] == 0, ("the shell never scrolls: what it moved out cannot come back", tag, opened)
        assert opened["bodyBottom"] is not None and opened["bodyBottom"] <= opened["bottom"] + 0.5, (
            "the shell bounds its body", tag, opened)
        assert opened["range"] > 0, ("the words want more room than the short window gives: the body scrolls", tag, opened)
        assert opened["sideways"] <= 0, ("its rows wrap inside its own scrollbar (WebKit's is 6px): no sideways scroll", tag, opened)
        # From the focus it opened with, back to the first words and through to Close, by the wheel over it.
        box = explanation.bounding_box()
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        page.mouse.wheel(0, -2000)
        page.wait_for_function("el => el.querySelector(':scope > .ui-popup-body').scrollTop === 0", arg=explanation.element_handle())
        top = explanation.evaluate(_REACH)
        shot(page, "top")
        assert top["wordsTop"] >= top["top"] - 0.5, ("the first words show", tag, top)
        page.mouse.wheel(0, 2000)
        page.wait_for_function("""el => { const body = el.querySelector(':scope > .ui-popup-body');
            return body.scrollTop >= body.scrollHeight - body.clientHeight - 1; }""", arg=explanation.element_handle())
        end = explanation.evaluate(_REACH)
        assert end["closeTop"] >= end["top"] - 0.5 and end["closeBottom"] <= end["bottom"] + 0.5, ("Close shows whole", tag, end)
        explanation.locator("[data-popup-close]").click()
        explanation.wait_for(state="detached")
        # The shell lays out its body only while it shows: a hidden popup and a closed dialog holding one stay hidden.
        shown = page.evaluate("""() => ['hidden', 'closed', 'open'].map((kind) => {
            const el = document.body.appendChild(document.createElement(kind === 'hidden' ? 'div' : 'dialog'));
            el.className = 'ui-popup'; el.hidden = kind === 'hidden'; el.innerHTML = '<div class="ui-popup-body">probe</div>';
            if (kind === 'open') el.show();
            const out = getComputedStyle(el).display; el.remove(); return out; })""")
        assert shown == ["none", "none", "flex"], (tag, shown)
        assert errors == [], errors
        if out:
            (Path(out) / f"official-short-{tag}.json").write_text(
                json.dumps({"opened": opened, "top": top, "end": end, "shown": shown}, indent=1), encoding="utf-8")
    finally:
        if skills:
            skills.close(browser)
        else:
            browser.close()


# The author kit (ouroboros/server_web.py read_author_kit_assets) is ui.css with no host script: a popup an author
# draws keeps its rows directly in the shell, and the shell scrolls them; a shell holding a body clips, and the body
# scrolls. Both in the kit's own stylesheet bytes, by the wheel a person uses.
_KIT_PAGE = """<!doctype html><html><head><meta charset="utf-8"><style>{css}</style></head><body style="margin:0">
<div class="ui-popup" role="menu" aria-label="Author menu" data-direct
     style="--ui-popup-top:16px;--ui-popup-left:16px;--ui-popup-max-height:88px;--ui-popup-max-width:280px">{rows}</div>
<div class="ui-popup" role="menu" aria-label="Wrapped menu" data-wrapped
     style="--ui-popup-top:16px;--ui-popup-left:320px;--ui-popup-max-height:88px;--ui-popup-max-width:280px">
<div class="ui-popup-body">{rows}</div></div></body></html>"""
_KIT_SCROLL = """(el) => { const body = el.querySelector(':scope > .ui-popup-body'); const s = getComputedStyle(el);
    return { shell: el.scrollTop, body: body ? body.scrollTop : null, height: el.getBoundingClientRect().height,
             overflow: s.overflowY, color: s.backgroundColor, shadow: s.boxShadow, radius: s.borderTopLeftRadius,
             range: (body || el).scrollHeight - (body || el).clientHeight }; }"""


@pytest.mark.parametrize("browser_name", ["chromium", "webkit"])
def test_author_kit_popup_scrolls_its_direct_rows_and_a_wrapped_one_only_its_body(skills_browser, browser_name, tmp_path):
    from ouroboros.server_web import read_author_kit_assets

    css = read_author_kit_assets(Path(__file__).resolve().parents[1])["css"]
    rows = "".join(f'<button type="button" role="menuitem">Author row {n}</button>' for n in range(1, 13))
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    seen = {}
    try:
        page = browser.new_page(viewport={"width": 640, "height": 320})
        page.set_content(_KIT_PAGE.format(css=css, rows=rows))
        page.wait_for_function(_SETTLED)
        for name in ("direct", "wrapped"):
            popup = page.locator(f"[data-{name}]")
            seen[f"{name}-opened"] = before = popup.evaluate(_KIT_SCROLL)
            assert before["height"] <= 88.5 and before["range"] > 100, (name, browser_name, before)
            box = popup.bounding_box()
            page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            page.mouse.wheel(0, 200)
            page.wait_for_function(
                "el => (el.querySelector(':scope > .ui-popup-body') || el).scrollTop > 0", arg=popup.element_handle())
            seen[f"{name}-wheeled"] = after = popup.evaluate(_KIT_SCROLL)
            assert after["color"] not in {"rgba(0, 0, 0, 0)", "transparent"} and after["shadow"] != "none", (name, after)
            assert after["radius"] == "18px", (name, after)
        direct, wrapped = seen["direct-wheeled"], seen["wrapped-wheeled"]
        assert direct["overflow"] == "auto" and direct["shell"] > 0, ("the author's rows scroll in their shell", seen)
        assert wrapped["overflow"] == "hidden" and wrapped["shell"] == 0 and wrapped["body"] > 0, (
            "a shell holding a body clips and its body scrolls", seen)
        out = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(tmp_path)))
        out.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(out / f"author-kit-popups-{browser_name}.png"))
        (out / f"author-kit-popups-{browser_name}.json").write_text(json.dumps(seen, indent=1), encoding="utf-8")
    finally:
        browser.close()
