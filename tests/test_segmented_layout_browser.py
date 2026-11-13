"""Current segmented choices: fit, chosen paint and keyboard persistence on their owning pages.

Settings rows save themselves. Controls now drawn as selects or choice cards are covered by their page tests;
this file measures every surviving strip, including the Agents fold, in both themes and engines.
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest

from tests.settings_browser_helpers import open_settings, setting_row
from tests.test_design_system_gate_browser import MEASURE as GATE_MEASURE
from tests.test_evolution_browser import history as evolution_history

pytest_plugins = ("tests.test_ui_smoke_playwright",)

ROOT = pathlib.Path(__file__).resolve().parents[1]
# Every group the renderer draws in Settings, with its number of choices.
GROUPS = {
    'OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS': 3,
    'OUROBOROS_TASK_REVIEW_MODE': 3, 'OUROBOROS_REVIEW_MAX_CYCLES': 5,
    'OUROBOROS_PROMPT_CACHE_TTL': 3, 'OUROBOROS_SAFETY_MODE': 3,
    'OUROBOROS_CONSCIOUSNESS_SAFETY_MODE': 3, 'OUROBOROS_SERVER_HOST': 2,
    'OUROBOROS_CONSCIOUSNESS_AUTONOMY': 3,
}
# The renderer's strip on a place sheet, then the two built by hand: (selector, choices).
OTHERS = {"place-state": ("[data-place-state]", 3), "evolution-measures": (".evo-measures", 5),
          "skills-trust": (".skills-trust-segments", 2)}
STRIPS = {**{target: (f'[data-setting-row="{target}"] .ui-segment-group', count) for target, count in GROUPS.items()}, **OTHERS}
DRAWN = set(GROUPS) | {"place-state"}
DESK, PHONE = {"width": 1280, "height": 860}, {"width": 320, "height": 640}
THEMES = ("dark", "light")
LONG = ("OUROBOROS_TASK_REVIEW_MODE", "Nach jeder Aufgabe, wann immer das Sprachmodell es für richtig hält")

_MEASURE = """([name, selector]) => {
  const group = [...document.querySelectorAll(selector)].find((node) => node.getClientRects().length);
  if (!group) return null;
  const px = (value) => parseFloat(value) || 0;
  const box = (rect) => ({left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom});
  // The row a strip lives in: its parent's content box. Both tagged: the gate's measure takes the row as its root,
  // and the screenshot frames the strip in it.
  const row = group.parentElement, rowStyle = getComputedStyle(row), rowBox = row.getBoundingClientRect();
  group.dataset.strip = name;
  row.dataset.stripRow = name;
  const style = getComputedStyle(group);
  const buttons = [...group.querySelectorAll('.ui-segment')].map((button) => {
    const rect = button.getBoundingClientRect(), look = getComputedStyle(button);
    const range = document.createRange();
    range.selectNodeContents(button);
    const text = range.getBoundingClientRect();
    return {...box(rect), width: rect.width, label: button.textContent.trim(), text: box(text),
            lines: new Set([...range.getClientRects()].map((line) => Math.round(line.top))).size,
            // The width of its word on one line, with the option's own padding and border.
            word: text.width + px(look.paddingLeft) + px(look.paddingRight) + px(look.borderLeftWidth) + px(look.borderRightWidth),
            touch: px(look.getPropertyValue('--touch-target-min')),
            chosen: button.matches('.active, [aria-checked="true"], [aria-pressed="true"]'),
            fill: look.backgroundColor, overflows: button.scrollWidth > button.clientWidth + 1};
  });
  const doc = document.scrollingElement;
  return {name, ...box(group.getBoundingClientRect()), track: style.backgroundColor, overflowX: style.overflowX,
          spills: group.scrollWidth > group.clientWidth + 1, buttons,
          row: {left: rowBox.left + px(rowStyle.borderLeftWidth) + px(rowStyle.paddingLeft),
                right: rowBox.right - px(rowStyle.borderRightWidth) - px(rowStyle.paddingRight)},
          titled: [group, ...group.querySelectorAll('*')].filter((node) => node.hasAttribute('title')).length,
          inline: group.hasAttribute('style'), page: doc.scrollWidth - doc.clientWidth};
}"""
# Transitions are off on the measured page, so a theme's colours are final at once: WebKit never ends a transition it
# skipped (web/modules/motion.js), and the app treats `transition: none` as a motion that ends on the next frame.
_STILL = "*, *::before, *::after { transition: none !important; }"
_SETTLE = """async (theme) => {
  document.documentElement.dataset.theme = theme;
  await new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done)));
}"""
# The strip in the middle of its scroller, with the words above it in its row: a tall section is cut, never the strip.
_FRAME = """(name) => {
  const group = document.querySelector(`[data-strip="${name}"]`);
  group.scrollIntoView({block: 'center', inline: 'nearest'});
  const strip = group.getBoundingClientRect(), row = group.parentElement.getBoundingClientRect();
  const top = Math.max(0, Math.max(row.top, strip.top - 160) - 8);
  const bottom = Math.min(innerHeight, Math.min(row.bottom, strip.bottom + 80) + 8);
  const left = Math.max(0, row.left - 8), right = Math.min(innerWidth, row.right + 8);
  return {x: left, y: top, width: right - left, height: bottom - top};
}"""
_PLACE = """async (raw) => {
  const [{openPlaceSheet}, {normalizePlacesResponse}] = await Promise.all([
    import('/static/modules/place_sheet.js'), import('/static/modules/consciousness_api.js')]);
  const view = normalizePlacesResponse(raw);
  const never = () => new Promise(() => {});
  window.placeSheet = openPlaceSheet(view.places[0], {circles: view.circles || [], readList: async () => view,
    readStatus: never, readModels: never, readRooms: async () => ({rooms: []})});
}"""
_TRUST = """async () => {
  const {openFirstStartDialog} = await import('/static/modules/skill_dialogs.js');
  openFirstStartDialog([
    {id: 'official', name: 'Ouroboros', trust: 'trusted', about: '12 skills marked Official by the maintainer'},
    {id: 'community', name: 'Ouroboros community', trust: 'review', about: '48 skills shared by other people'}]);
}"""


def _go(page, place: str) -> None:
    toggle = page.locator("[data-mobile-nav-toggle]:visible")
    if toggle.count():
        toggle.first.click()
    page.locator(f'[data-nav-page="{place}"]').click()


def _open(page, url: str) -> None:
    page.goto(url, wait_until="domcontentloaded")
    page.wait_for_selector("#page-chat", timeout=30_000)
    _go(page, "settings")
    page.wait_for_selector(".settings-shell", timeout=20_000)


def _transparent(colour: str) -> bool:
    return colour in ("transparent", "rgba(0, 0, 0, 0)")


def _check(strip: dict, count: int, *, desk: bool) -> list[str]:
    name, buttons, problems = strip["name"], strip["buttons"], []
    if len(buttons) != count:
        problems.append(f"{len(buttons)} options, expected {count}")
    if strip["titled"] or strip["inline"]:
        problems.append(f"a hover title ({strip['titled']}) or an inline style ({strip['inline']}) on the strip")
    if strip["page"] > 1:
        problems.append(f"the page scrolls sideways by {strip['page']:.1f}px")
    if strip["left"] < strip["row"]["left"] - 0.5 or strip["right"] > strip["row"]["right"] + 0.5:
        problems.append(f"leaves its row: {strip['left']:.1f}..{strip['right']:.1f} in {strip['row']}")
    if strip["spills"] or strip["overflowX"] in ("auto", "scroll"):
        problems.append(f"scrolls or spills sideways (overflow-x {strip['overflowX']})")
    for button in buttons:
        text, label = button["text"], button["label"]
        if button["width"] < button["touch"] - 0.5:
            problems.append(f"{label!r} is {button['width']:.1f}px wide, under the {button['touch']:.0f}px touch minimum")
        if (text["left"] < button["left"] - 0.5 or text["right"] > button["right"] + 0.5
                or text["top"] < button["top"] - 0.5 or text["bottom"] > button["bottom"] + 0.5 or button["overflows"]):
            problems.append(f"{label!r} leaves its option: text {text}, option {button}")
        if button["left"] < strip["left"] - 0.5 or button["right"] > strip["right"] + 0.5:
            problems.append(f"{label!r} sticks out of the track")
    chosen = [button for button in buttons if button["chosen"]]
    if len(chosen) != 1:
        problems.append(f"{len(chosen)} chosen options")
    else:
        # What an option shows: its own fill, or the track's through a clear one.
        shown = [strip["track"] if _transparent(button["fill"]) else button["fill"] for button in buttons if not button["chosen"]]
        if _transparent(chosen[0]["fill"]) or chosen[0]["fill"] in shown:
            problems.append(f"the chosen option ({chosen[0]['fill']}) looks like another ({shown})")
    if desk and name in DRAWN:
        if len({round(button["top"]) for button in buttons}) != 1:
            problems.append("wraps on the desk")
        for button in buttons:
            if abs(button["width"] - max(button["touch"], button["word"])) > 1:
                problems.append(f"{button['label']!r} is {button['width']:.1f}px wide, its word {button['word']:.1f}px")
    return [f"{name}: {problem}" for problem in problems]


@pytest.mark.ui_browser
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_every_strip_fits_its_row_and_shows_its_choice(direct_server_with_data, engine):
    from playwright.sync_api import sync_playwright

    fixture = direct_server_with_data
    evidence = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(fixture["data_dir"].parent)))
    evidence.mkdir(parents=True, exist_ok=True)
    places = json.loads((ROOT / "web/tests/fixtures/consciousness_contract.json").read_text(encoding="utf-8"))["places"]
    problems = []
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch(headless=True)
        try:
            for viewport in (DESK, PHONE):
                desk = viewport is DESK
                page = browser.new_page(viewport=viewport, has_touch=not desk)
                page.route("**/api/evolution-data*", lambda route: route.fulfill(json=evolution_history()))
                seen, wrapped = set(), set()

                def shoot(name: str, file: str) -> None:
                    page.screenshot(path=str(evidence / file), clip=page.evaluate(_FRAME, name), animations="disabled")

                def measure(name: str) -> None:
                    selector, count = STRIPS[name]
                    for theme in THEMES:
                        page.evaluate(_SETTLE, theme)
                        strip = page.evaluate(_MEASURE, [name, selector])
                        if strip is None:
                            return
                        seen.add(name)
                        width = viewport["width"]
                        problems.extend(f"{engine} {width} {theme} {line}" for line in _check(strip, count, desk=desk))
                        if len({round(button["top"]) for button in strip["buttons"]}) > 1:
                            wrapped.add(name)
                        shoot(name, f"strip-{engine}-{width}-{theme}-{name}.png")
                        if not desk:
                            gate = page.evaluate(GATE_MEASURE, f'[data-strip-row="{name}"]')
                            if any(gate.values()):
                                problems.append(f"{engine} {width} {theme} {name}: the design gate measures {gate}")

                _open(page, fixture["url"])
                page.add_style_tag(content=_STILL)
                for path, fold in [('agents', 'OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS'),
                                   ('permissions', 'OUROBOROS_REVIEW_MAX_CYCLES'),
                                   ('models', 'OUROBOROS_PROMPT_CACHE_TTL'), ('computer', '')]:
                    open_settings(page, path, fold)
                    for name in GROUPS:
                        measure(name)
                    target, label = LONG
                    if not desk and page.locator(STRIPS[target][0]).is_visible():
                        # A translated label longer than the line breaks inside its option, and the strip still fits.
                        page.evaluate("""([selector, label]) => {
                            const option = document.querySelector(`${selector} .ui-segment:last-child`);
                            option.textContent = label;
                        }""", [STRIPS[target][0], label])
                        page.evaluate(_SETTLE, "light")
                        strip = page.evaluate(_MEASURE, [target, STRIPS[target][0]])
                        problems.extend(f"{engine} 320 long label {line}" for line in _check(strip, 3, desk=False))
                        if strip["buttons"][-1]["lines"] < 2:
                            problems.append(f"{engine} 320: the long label stayed on one line: {strip['buttons'][-1]}")
                        shoot(target, f"strip-{engine}-320-light-long-label.png")

                page.evaluate(_PLACE, places)
                page.locator("[data-place-state]").wait_for(state="visible")
                measure("place-state")
                page.locator("[data-place-close]").click()

                page.evaluate(_TRUST)
                page.locator(".skills-trust-segments").first.wait_for(state="visible")
                measure("skills-trust")
                page.locator(".skills-dialog [data-dialog-cancel]").first.click()

                _go(page, "dashboard")
                page.locator('[data-dashboard-tab="consciousness"]').click()
                page.locator(STRIPS['OUROBOROS_CONSCIOUSNESS_AUTONOMY'][0]).wait_for(state='visible')
                measure('OUROBOROS_CONSCIOUSNESS_AUTONOMY')
                page.locator('[data-dashboard-tab="evolution"]').click()
                page.locator(".evo-measures").wait_for(state="visible", timeout=30_000)
                measure("evolution-measures")

                if seen != set(STRIPS):
                    problems.append(f"{engine} {viewport['width']}: never found {sorted(set(STRIPS) - seen)}")
                # The negative control: the desk keeps every strip on one line; at 320 px one does wrap, so the wrap
                # path is what the checks above measured there.
                if desk and wrapped:
                    problems.append(f"{engine} desk: {sorted(wrapped)} wrap")
                if not desk and not wrapped & DRAWN:
                    problems.append(f"{engine} 320: no drawn strip wraps, so the wrap was never measured")
                page.close()
        finally:
            browser.close()
    assert not problems, "\n".join(problems)


@pytest.mark.ui_browser
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_keyboard_choice_focus_ring_and_the_chosen_value_survive_save(direct_server_with_data, engine):
    from playwright.sync_api import expect, sync_playwright

    fixture = direct_server_with_data
    settings_path = fixture["data_dir"] / "settings.json"
    group = '[data-setting-row="OUROBOROS_TASK_REVIEW_MODE"] .ui-segment-group'
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 900})

            def open_behavior():
                _open(page, fixture["url"])
                open_settings(page, "permissions")

            def press(value: str, key: str) -> None:
                page.keyboard.press("Shift")  # keyboard modality: script focus then counts as keyboard focus
                page.locator(f'{group} [data-setting-value="{value}"]').focus()
                with page.expect_response('**/api/settings'):
                    page.keyboard.press(key)
                expect(setting_row(page, 'OUROBOROS_TASK_REVIEW_MODE').locator('[data-setting-result]')).to_contain_text('Saved')

            def save() -> dict:
                return json.loads(settings_path.read_text(encoding="utf-8"))

            open_behavior()
            press("required", "Enter")
            assert page.locator(f'{group} [data-setting-value="required"]').get_attribute("aria-pressed") == "true"
            ring = page.evaluate("""(selector) => {
                const focused = document.activeElement;
                const probe = document.createElement('span');
                probe.style.color = 'var(--focus-accent-border)';
                document.body.append(probe);
                const token = getComputedStyle(probe).color;
                probe.remove();
                const style = getComputedStyle(focused);
                return {visible: focused.matches(':focus-visible'), value: focused.dataset.settingValue, token,
                        outline: [style.outlineStyle, style.outlineWidth, style.outlineOffset, style.outlineColor]};
            }""", group)
            assert ring["visible"] and ring["value"] == "required", ring
            assert ring["outline"] == ["solid", "2px", "2px", ring["token"]], ring
            # Hover paints the button; it never draws the focus ring.
            hovered = page.locator(f'{group} [data-setting-value="off"]')
            hovered.hover()
            assert hovered.evaluate("(node) => getComputedStyle(node).outlineStyle") == "none"
            assert save()["OUROBOROS_TASK_REVIEW_MODE"] == "required"

            open_behavior()
            # A strip claims no choice until its value is read: wait for the read, never for a time.
            expect(page.locator(f'{group} [data-setting-value="required"]')).to_have_attribute("aria-pressed", "true")
            press("off", " ")  # Space chooses too
            assert page.locator(f'{group} [data-setting-value="off"]').get_attribute("aria-pressed") == "true"
            assert save()["OUROBOROS_TASK_REVIEW_MODE"] == "off"

            open_behavior()
            chosen = page.locator(f'{group} [data-setting-value="off"]')
            expect(chosen).to_have_attribute("aria-pressed", "true")
            assert "Off" in chosen.inner_text()
            assert page.locator(f'{group} [data-setting-value="required"]').get_attribute("aria-pressed") == "false"
        finally:
            browser.close()
