"""A popup's material is its own, whatever its ARIA role (DESIGN principle 2).

The real Skills page in Chromium and WebKit: the Official mark's explanation (a ``role="dialog"`` popup, which once had
no background and no shadow because the popup material was selected by ``role="menu"``) is compared with a real
menu of the same page, in both themes at a window and at a phone's width. The oracle is falsified in-test so it can fail.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pytest
from PIL import Image, ImageChops

from tests._skills_page_harness import URL, fixture, open_skills, skills_browser as _skills_browser

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
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
