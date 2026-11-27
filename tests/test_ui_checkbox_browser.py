"""The check box and the radio (`.ui-checkbox`, `.ui-radio`, ui.css): what a finger selects and what forced colours show.

On a real page in both engines, with the product's own sheets and the markup its builders write (setup's starter skills,
New project's folder choices, a form's inline fields, a share part's own box):
- under a touch pointer each label holding a box or radio is at least the touch minimum tall, and a tap on either side of
  the boundary between two rows changes exactly the field of the row it lands in, never its neighbour;
- a box that stands alone (a share part's button) has the touch minimum allocated around it;
- under forced colours, in both forced palettes and both app themes, a drawn mark's edge shows against the surface under
  it when unchecked, and its tick or bar shows against its fill when ticked or mixed, also inside a dark assistant bubble.
"""
from __future__ import annotations

import io

import pytest
from PIL import Image

from tests import test_subscription_setup_browser as setup_browser

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser]

ROWS = """() => {
    const host = document.createElement('div');
    host.style.cssText = 'position:fixed;inset:0;z-index:9999;overflow:auto;padding:16px;background:var(--bg-primary)';
    const skill = (id, title) => `<label class="skills-starter-skill" data-row="${id}"><input type="checkbox" class="ui-checkbox"`
        + ` data-field="${id}" checked><span class="skills-starter-words"><strong>${title}</strong>`
        + ` <span class="skills-starter-pitch">Read and create documents of this kind</span></span></label>`;
    const choice = (value, label) => `<label data-row="${value}"><input class="ui-radio" type="radio" name="np-source"`
        + ` data-field="${value}" value="${value}"${value === 'attach' ? ' checked' : ''}> ${label}</label>`;
    const field = (id, label) => `<label class="ui-field ui-field-inline" data-row="${id}"><input class="ui-checkbox"`
        + ` type="checkbox" data-field="${id}"> <span>${label}</span></label>`;
    host.innerHTML = `<section class="skills-starter-group">${skill('pdf', 'PDF files')}${skill('word', 'Word documents')}</section>`
        + `<div class="new-project-source-choices" style="max-width:300px">${choice('fileless', 'No folder')}`
        + `${choice('genesis', 'New folder')}${choice('attach', 'Existing folder…')}${choice('clone', 'Clone from Git…')}</div>`
        + `<div class="ui-field">${field('digest', 'Send a weekly digest')}${field('cache', 'Keep the cache')}</div>`
        + '<p style="margin:24px 0 0">before</p><button class="ui-checkbox message-share-check" role="checkbox"'
        + ' aria-checked="false" data-field="share"></button><p style="margin:0">after</p>';
    document.body.append(host);
    // The share part's own box toggles as message_share.js's does.
    const share = host.querySelector('[data-field="share"]');
    share.addEventListener('click', () => share.setAttribute('aria-checked', String(share.getAttribute('aria-checked') !== 'true')));
    return true;
}"""
STATE = """() => Object.fromEntries([...document.querySelectorAll('[data-field]')].map((node) =>
    [node.dataset.field, node.matches('button') ? node.getAttribute('aria-checked') === 'true' : node.checked]))"""
BOX = """(name) => { const box = document.querySelector(`[data-field="${name}"]`);
    const row = document.querySelector(`[data-row="${name}"]`) || box;
    const r = row.getBoundingClientRect(), b = box.getBoundingClientRect();
    return {top: r.top, bottom: r.bottom, height: r.height, x: b.x + b.width / 2, y: b.y + b.height / 2}; }"""


def touch_page(ui, engine):
    options = {"viewport": {"width": 390, "height": 844}, "has_touch": True}
    if engine == "chromium":
        options["is_mobile"] = True
    page = ui["new_page"](**options)
    page.goto(ui["url"] + "/")
    page.wait_for_selector("#chat-input", state="attached")
    page.add_style_tag(content=(setup_browser.WEB / "onboarding.css").read_text(encoding="utf-8"))
    return page


def flips(page, x, y):
    before = page.evaluate(STATE)
    page.touchscreen.tap(x, y)
    after = page.evaluate(STATE)
    return sorted(name for name in after if after[name] != before[name])


@pytest.fixture(params=["chromium", "webkit"])
def engine_ui(request, monkeypatch):
    monkeypatch.setenv("OUROBOROS_UI_BROWSER_ENGINE", request.param)
    ui = request.getfixturevalue("subscription_ui")
    ui["engine"] = request.param
    return ui


def test_a_tap_changes_the_row_it_lands_in_and_a_lone_box_has_its_target(engine_ui):
    page = touch_page(engine_ui, engine_ui["engine"])
    assert page.evaluate("matchMedia('(pointer: coarse)').matches"), "the touch context is a coarse pointer"
    page.evaluate(ROWS)
    minimum = page.evaluate("parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--touch-target-min'))")
    for upper, lower in (("pdf", "word"), ("digest", "cache")):
        a, b = page.evaluate(BOX, upper), page.evaluate(BOX, lower)
        assert a["height"] >= minimum and b["height"] >= minimum, (upper, lower, a, b)
        assert flips(page, a["x"], a["bottom"] - 2) == [upper], (upper, "a tap just inside a row's bottom is its own")
        assert flips(page, b["x"], b["top"] + 2) == [lower], (lower, "a tap just inside the next row's top is that row's")
    # Folder choices wrap onto two lines: a tap inside No folder's own circle, at its bottom edge, picks No folder.
    fileless = page.evaluate(BOX, "fileless")
    assert fileless["height"] >= minimum, fileless
    page.touchscreen.tap(fileless["x"], fileless["y"] + 7)
    assert page.evaluate(STATE)["fileless"], "the circle's own row, not the choice under it"
    # The lone box: the touch minimum around its glyph is its own (allocated, so the words beside it are not covered).
    reach = page.evaluate("""() => { const box = document.querySelector('[data-field="share"]'), r = box.getBoundingClientRect();
        const x = r.x + r.width / 2, y = r.y + r.height / 2;
        const run = (dx, dy) => Array.from({length: 80}, (_, i) => document.elementFromPoint(x + dx * (i - 40), y + dy * (i - 40)) === box)
            .filter(Boolean).length;
        return {tall: run(0, 1), wide: run(1, 0), glyph: [r.width, r.height]}; }""")
    assert reach["glyph"] == [16, 16], reach
    assert reach["tall"] >= minimum - 1 and reach["wide"] >= minimum - 1, reach
    assert flips(page, page.evaluate(BOX, "share")["x"], page.evaluate(BOX, "share")["y"] - minimum / 2 + 2) == ["share"]
    setup_browser.capture(page, f"checkbox-touch-{engine_ui['engine']}")


MARKS = """() => {
    const host = document.createElement('div');
    host.id = 'checkbox-marks';
    host.style.cssText = 'position:fixed;left:0;top:0;z-index:9999;display:flex;gap:24px;padding:24px;background:var(--bg-primary)';
    const marks = '<button class="ui-checkbox" role="checkbox" aria-checked="false" data-mark="button-off"></button>'
        + '<span class="ui-checkbox md-checkbox" aria-hidden="true" data-mark="span-off"></span>'
        + '<button class="ui-checkbox" role="checkbox" aria-checked="true" data-mark="button-on"></button>'
        + '<span class="ui-checkbox md-checkbox is-checked" aria-hidden="true" data-mark="span-on"></span>'
        + '<button class="ui-checkbox" role="checkbox" aria-checked="mixed" data-mark="button-mixed"></button>';
    host.innerHTML = marks + '<div class="chat-bubble assistant" style="padding:12px"><span class="ui-checkbox md-checkbox is-checked"'
        + ' aria-hidden="true" data-mark="voice-on"></span></div>';
    document.body.append(host);
    return [...host.querySelectorAll('[data-mark]')].map((node) => {
        const r = node.getBoundingClientRect();
        return [node.dataset.mark, r.x, r.y, r.width, r.height];
    });
}"""


def luminance(rgb):
    def channel(value):
        value /= 255
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
    return 0.2126 * channel(rgb[0]) + 0.7152 * channel(rgb[1]) + 0.0722 * channel(rgb[2])


def contrast(a, b):
    high, low = sorted((luminance(a), luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_forced_colours_show_every_drawn_mark_and_its_state(engine_ui, theme):
    page = engine_ui["new_page"](viewport={"width": 600, "height": 160}, device_scale_factor=3)
    page.goto(engine_ui["url"] + "/")
    page.wait_for_selector("#chat-input", state="attached")
    page.evaluate("theme => { document.documentElement.dataset.theme = theme; }", theme)
    for scheme in ("light", "dark"):
        page.emulate_media(forced_colors="active", color_scheme=scheme)
        boxes = page.evaluate(MARKS)
        shot = Image.open(io.BytesIO(page.screenshot())).convert("RGB")
        for name, x, y, w, h in boxes:
            left, right, mid = round(x * 3), round((x + w) * 3), round((y + h / 2) * 3)
            row = [shot.getpixel((px, mid)) for px in range(left, right)]
            if name.endswith("-off"):
                surface = shot.getpixel((left - 6, mid))
                best = max(contrast(pixel, surface) for pixel in row)
                assert best >= 3, (theme, scheme, name, "unchecked edge on its surface", surface, best)
            else:
                fill = shot.getpixel((left + round(w * 3 * 0.2), mid))
                inner = [shot.getpixel((px, py)) for px in range(left + 6, right - 6) for py in range(mid - 9, mid + 9)]
                best = max(contrast(pixel, fill) for pixel in inner)
                assert best >= 3, (theme, scheme, name, "tick or bar on its fill", fill, best)
        # The marks wear the user's palette, not the app's: no app rule (a voice's edge, the mixed fill) outranks it.
        paint = page.evaluate("""() => {
            const system = (name) => { const probe = document.createElement('i'); probe.style.color = name; document.body.append(probe);
                const value = getComputedStyle(probe).color; probe.remove(); return value; };
            const [text, highlight] = [system('CanvasText'), system('Highlight')];
            return [...document.querySelectorAll('#checkbox-marks [data-mark]')].map((node) => {
                const look = getComputedStyle(node), off = node.dataset.mark.endsWith('-off');
                return [node.dataset.mark, look.borderTopColor === (off ? text : highlight),
                    off || look.backgroundColor === highlight];
            });
        }""")
        assert all(edge and fill for _name, edge, fill in paint), (theme, scheme, paint)
        page.evaluate("() => document.getElementById('checkbox-marks').remove()")
    setup_browser.capture(page, f"checkbox-forced-{engine_ui['engine']}-{theme}")
