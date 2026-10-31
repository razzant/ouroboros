"""Main/Project message-field height in the real UI: native content sizing and the measured fallback.

The field grows with its text, or an empty field's placeholder, to the stylesheet's one cap,
then scrolls, and shrinks again; a field whose content fits shows no scrollbar and no cut
glyph (DESIGN "message field"). Each engine runs twice: with the mode `chat_input_sizing.js`
selects (WebKit 26 measures, because its content sizing cuts a wrapped placeholder), and with
`field-sizing` denied, a genuine fallback because the stylesheet keys native sizing on that
mode (checked through the computed style). CSS zoom and device scale factor are proxies for
display scaling, not a native Windows or WebView2 result.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pytest

from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data
from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET, _emit_ws_frame, _SETTLE_RESTORE_FRAMES

_DENY_FIELD_SIZING = """() => {
    const supports = CSS.supports.bind(CSS);
    CSS.supports = (...args) => !String(args[0]).includes('field-sizing') && supports(...args);
}"""
_RECORD_ERRORS = """() => {
    window.__sizingErrors = [];
    window.addEventListener('error', event => window.__sizingErrors.push(String(event.message || event)));
}"""
_GEOMETRY = r"""el => {
    const style = getComputedStyle(el);
    const top = el.scrollTop;
    el.scrollTop = 1e6;
    const maxScroll = el.scrollTop;
    el.scrollTop = top;
    const rect = el.getBoundingClientRect();
    return {
        rectHeight: rect.height, rectWidth: rect.width, clientHeight: el.clientHeight, clientWidth: el.clientWidth,
        scrollHeight: el.scrollHeight, scrollTop: el.scrollTop, maxScroll,
        line: parseFloat(style.lineHeight), pad: parseFloat(style.paddingTop) + parseFloat(style.paddingBottom),
        maxHeight: style.maxHeight, sizing: el.dataset.sizing || '', fieldSizing: style.fieldSizing || '',
        overflowY: style.overflowY, inlineHeight: el.style.height,
        measuredHeight: el.style.getPropertyValue('--chat-input-height'),
        selection: [el.selectionStart, el.selectionEnd], focused: document.activeElement === el,
        hardLines: el.value.split('\n').length, length: el.value.length,
        zoom: getComputedStyle(document.documentElement).zoom, dpr: window.devicePixelRatio,
        viewport: window.innerWidth,
    };
}"""
# Counts the field's own attribute writes (style, data-sizing) so convergence is observable.
_WATCH_WRITES = """el => {
    window.__sizingWrites = 0;
    window.__sizingWatch?.disconnect();
    window.__sizingWatch = new MutationObserver(records => { window.__sizingWrites += records.length; });
    window.__sizingWatch.observe(el, { attributes: true, attributeFilter: ['style', 'data-sizing'] });
}"""
_PARAGRAPH = ("Paragraph changes wrapping on narrower space and keeps every word readable. " * 2).strip()
# A translation that wraps at a phone width, its last line carrying descenders.
_LONG_PLACEHOLDER = ("Напишите сообщение для Ouroboros; Shift+Enter переносит строку, Enter отправляет сразу, "
                     "файлы добавляйте скрепкой рядом")
# A face the app serves, half again as wide as any interface face: the late-font text
# rewraps whatever this platform's own face is.
_LATE_FONT = "/static/vendor/katex/fonts/KaTeX_Typewriter-Regular.woff2"
_LATE_FACE = "{ display: 'swap', sizeAdjust: '150%' }"
# The longest prefix that fits one row of the field in its own face, and its width in
# another face: the near-wrap text is derived from this platform's fonts, not assumed.
_NEAR_WRAP = """(el, other) => {
    const style = getComputedStyle(el);
    const room = el.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
    const span = document.createElement('span');
    span.style.cssText = `position:absolute;visibility:hidden;white-space:pre;font-size:${style.fontSize}`;
    document.body.append(span);
    const width = (text, family) => {
        span.style.fontFamily = family;
        span.textContent = text;
        return span.getBoundingClientRect().width;
    };
    let text = '';
    for (const ch of 'A message that fits one row in this face and wraps once a wider face loads late, '.repeat(3)) {
        if (width(text + ch, style.fontFamily) > room - 12) break;
        text += ch;
    }
    const result = { text: text.trim(), room, own: width(text, style.fontFamily), other: width(text, other) };
    span.remove();
    return result;
}"""


def _assert_fits(g, mode, case, rows=None):
    """Below the cap: the box holds its whole text, so nothing scrolls."""
    if rows is not None:
        assert g["clientHeight"] >= rows * g["line"] + g["pad"] - 1, (case, g)
    if mode == "content":
        assert g["maxScroll"] == 0, (case, g)
    else:
        # A rounded integer scrollHeight may leave the box under a pixel short of its text;
        # that residual is hidden, never a scrollbar.
        assert g["scrollHeight"] - g["clientHeight"] <= 1 and g["maxScroll"] < 1, (case, g)
    assert g["clientHeight"] < 140, (case, g)
    _assert_mode(g, mode, case, "fit")


def _assert_capped(g, mode, case, *, at_end=True):
    """At the one CSS cap the text scrolls, and a caret at the end keeps the last line in view."""
    assert abs(g["clientHeight"] - 140) <= 1 and g["overflowY"] == "auto" and g["maxScroll"] > 0, (case, g)
    if at_end:
        assert g["maxScroll"] - g["scrollTop"] <= 1, (case, g)
    _assert_mode(g, mode, case, "capped")


def _assert_mode(g, mode, case, state):
    assert g["inlineHeight"] == "", (case, g)  # neither mode writes the height property itself
    if mode == "content":
        assert g["sizing"] == "content" and g["fieldSizing"] == "content" and g["measuredHeight"] == "", (case, g)
    else:
        assert g["sizing"] == state and g["fieldSizing"] != "content", (case, g)
        assert g["overflowY"] == ("hidden" if state == "fit" else "auto"), (case, g)
        # The measured height lives in its one custom property; at the cap it is the CSS cap itself.
        if state == "capped":
            assert g["measuredHeight"] == g["maxHeight"], (case, g)
        else:
            assert g["measuredHeight"].endswith("px"), (case, g)


class _Run:
    """One engine run: `variant` is how the mode came about (`selected` or `denied`)."""
    def __init__(self, page, engine, mode, evidence, records, variant):
        self.page, self.engine, self.mode, self.evidence, self.records = page, engine, mode, evidence, records
        self.label = f"{variant}-{mode}"

    def settle(self):
        self.page.evaluate(_SETTLE_RESTORE_FRAMES)

    def geometry(self, field, case):
        self.settle()
        g = field.evaluate(_GEOMETRY)
        self.records.append({"engine": self.engine, "mode": self.mode, "run": self.label, "case": case, **g})
        return g

    def shoot(self, row, case):
        row.screenshot(path=str(self.evidence / f"composer-sizing-{self.engine}-{self.label}-{case}.png"))

    def replace(self, field, text):
        field.focus()
        self.page.keyboard.press("ControlOrMeta+A")
        self.page.keyboard.press("Backspace")
        if text:
            self.page.keyboard.insert_text(text)


def _typing(run, field, row):
    page, mode = run.page, run.mode
    field.click()
    g = run.geometry(field, "empty")
    _assert_fits(g, mode, "empty", rows=1)
    assert g["clientHeight"] < 2 * g["line"] + g["pad"], g
    run.shoot(row, "empty")
    page.keyboard.type("A short message")
    _assert_fits(run.geometry(field, "one-line"), mode, "one-line", rows=1)
    run.shoot(row, "one-line")
    # Typed hard lines, each Shift+Enter leaving the caret on a new empty last line.
    run.replace(field, "")
    for number, word in enumerate(("one", "two", "three", "four", "five"), start=1):
        page.keyboard.type(word)
        _assert_fits(run.geometry(field, f"lines-{number}"), mode, f"lines-{number}", rows=number)
        if number == 5:
            run.shoot(row, "five-lines")  # five rows: past the former 120px script cap
        page.keyboard.press("Shift+Enter")
        if number < 5:
            _assert_fits(run.geometry(field, f"trailing-{number}"), mode, f"trailing-{number}", rows=number + 1)
    # The trailing sixth line, where the caret is, reaches the cap.
    _assert_capped(run.geometry(field, "trailing-5"), mode, "trailing-5")
    for word in ("six", "seven", "eight"):
        page.keyboard.type(word)
        page.keyboard.press("Shift+Enter")
    page.keyboard.type("nine")
    _assert_capped(run.geometry(field, "capped"), mode, "capped")
    run.shoot(row, "capped")
    # Typing at the start keeps the start in view rather than snapping to the end.
    field.evaluate("el => { el.setSelectionRange(0, 0); el.scrollTop = 0; }")
    page.keyboard.type("Z")
    g = run.geometry(field, "capped-edit-start")
    _assert_capped(g, mode, "capped-edit-start", at_end=False)
    assert g["scrollTop"] <= g["line"] and g["selection"] == [1, 1] and g["focused"], g
    # Deleting and clearing shrink it again.
    field.evaluate("el => el.setSelectionRange(el.value.length, el.value.length)")
    for _ in range(len("\nnine\neight\nseven\nsix")):
        page.keyboard.press("Backspace")
    _assert_fits(run.geometry(field, "deleted-to-five"), mode, "deleted-to-five", rows=5)
    run.replace(field, "")
    _assert_fits(run.geometry(field, "cleared"), mode, "cleared", rows=1)
    # Pasted text grows it; past the cap the end of the pasted text is in view.
    page.keyboard.insert_text(_PARAGRAPH)
    _assert_fits(run.geometry(field, "pasted-paragraph"), mode, "pasted-paragraph")
    page.keyboard.insert_text(" " + _PARAGRAPH * 8)
    _assert_capped(run.geometry(field, "pasted-long"), mode, "pasted-long")


def _layout(run, field, row):
    page, mode = run.page, run.mode
    run.replace(field, _PARAGRAPH)
    field.evaluate("el => el.blur()")
    wide = run.geometry(field, "paragraph-wide")
    _assert_fits(wide, mode, "paragraph-wide")
    # A narrower column, with no typing, rewraps the same text.
    page.set_viewport_size({"width": 620, "height": 800})
    narrow = run.geometry(field, "paragraph-narrow")
    _assert_fits(narrow, mode, "paragraph-narrow")
    assert narrow["clientWidth"] < wide["clientWidth"] and narrow["clientHeight"] > wide["clientHeight"], (wide, narrow)
    run.shoot(row, "paragraph-narrow")
    page.set_viewport_size({"width": 1240, "height": 800})
    rewide = run.geometry(field, "paragraph-rewide")
    _assert_fits(rewide, mode, "paragraph-rewide")
    assert abs(rewide["clientHeight"] - wide["clientHeight"]) <= 1, (wide, rewide)
    # The busy Send label narrows the field while the composer keeps its width.
    page.locator("#chat-send").evaluate("el => { el.textContent = 'Uploading files'; }")
    busy = run.geometry(field, "busy-label")
    assert busy["clientWidth"] < wide["clientWidth"], (wide, busy)
    _assert_fits(busy, mode, "busy-label")
    page.locator("#chat-send").evaluate("el => { el.textContent = 'Send'; }")
    _assert_fits(run.geometry(field, "busy-label-restored"), mode, "busy-label-restored")
    # Observer writes converge: a width change settles and then stays quiet.
    field.evaluate(_WATCH_WRITES)
    page.set_viewport_size({"width": 900, "height": 800})
    run.settle()
    first = page.evaluate("() => window.__sizingWrites")
    run.settle()
    run.settle()
    quiet = page.evaluate("() => window.__sizingWrites")
    run.records.append({"engine": run.engine, "mode": mode, "case": "observer-writes", "afterResize": first, "later": quiet})
    assert quiet == first, (first, quiet)
    if mode == "content":
        assert first == 0, first  # the engine sizes the field; nothing is written
    else:
        assert 0 < first <= 8, first  # one or two measurements, each a few attribute writes
    page.evaluate("() => window.__sizingWatch.disconnect()")
    # A selection inside capped text, with focus, survives a width change in place.
    run.replace(field, "\n".join(f"Line {n} of a long draft" for n in range(1, 11)))
    field.evaluate("el => { el.setSelectionRange(4, 9); el.scrollTop = 0; }")
    page.set_viewport_size({"width": 1100, "height": 800})
    g = run.geometry(field, "capped-selection-resize")
    _assert_capped(g, mode, "capped-selection-resize", at_end=False)
    assert g["selection"] == [4, 9] and g["focused"] and g["scrollTop"] == 0, g
    page.set_viewport_size({"width": 1240, "height": 800})
    # A sub-pixel narrowing that rewraps while clientWidth stays the same whole pixel, as a
    # window resized by one device pixel at 125% does: two widths a fifth of a pixel apart
    # straddle a text's wrap point, found by bisecting the field's own width.
    run.replace(field, "")
    found = field.evaluate("""(el, texts) => {
        const tall = (text, width) => {
            el.style.width = `${width}px`;
            el.value = text;
            const height = el.scrollHeight;
            el.value = '';
            return height > 1.5 * parseFloat(getComputedStyle(el).lineHeight) + 10;
        };
        el.style.flex = 'none';
        const full = el.getBoundingClientRect().width;
        for (const text of texts) {
            let [fits, wrapped] = [full, 120];
            for (let step = 0; step < 24; step++) {
                const middle = (fits + wrapped) / 2;
                if (tall(text, middle)) wrapped = middle; else fits = middle;
            }
            const widths = [fits + 0.1, fits - 0.1];
            const client = widths.map(width => { el.style.width = `${width}px`; return el.clientWidth; });
            if (client[0] === client[1]) return { text, widths };
        }
        return null;
    }""", [" ".join(_PARAGRAPH.split()[:words]) for words in range(6, 11)])
    assert found, "no text's wrap point lies inside one whole clientWidth"
    field.evaluate("(el, width) => { el.style.width = `${width}px`; }", found["widths"][0])
    run.settle()
    run.replace(field, found["text"])
    field.evaluate("el => el.blur()")
    wide = run.geometry(field, "subpixel-wide")
    _assert_fits(wide, mode, "subpixel-wide", rows=1)
    field.evaluate("(el, width) => { el.style.width = `${width}px`; }", found["widths"][1])
    narrow = run.geometry(field, "subpixel-narrow")
    assert narrow["clientWidth"] == wide["clientWidth"], (wide, narrow)
    _assert_fits(narrow, mode, "subpixel-narrow", rows=2)
    field.evaluate("el => { el.style.removeProperty('width'); el.style.removeProperty('flex'); }")
    run.replace(field, "")


def _keyboard(run, field):
    page, mode = run.page, run.mode
    run.replace(field, "")
    page.keyboard.type("first line")
    page.keyboard.press("Shift+Enter")
    page.keyboard.type("second line")
    page.keyboard.press("Shift+Enter")
    page.keyboard.type("third line")
    _assert_fits(run.geometry(field, "before-send"), mode, "before-send", rows=3)
    page.keyboard.press("Enter")
    page.wait_for_function("() => document.getElementById('chat-input').value === ''")
    _assert_fits(run.geometry(field, "sent"), mode, "sent", rows=1)
    page.keyboard.press("ArrowUp")
    g = run.geometry(field, "history-recall")
    assert field.input_value() == "first line\nsecond line\nthird line"
    _assert_fits(g, mode, "history-recall", rows=3)
    assert g["focused"], g
    # An input method's Enter commits a candidate: it neither sends nor resizes.
    before = run.geometry(field, "composition-before")
    field.evaluate("""el => {
        el.dispatchEvent(new CompositionEvent('compositionstart', { bubbles: true, data: '' }));
        el.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', keyCode: 229, isComposing: true,
            bubbles: true, cancelable: true }));
        el.dispatchEvent(new CompositionEvent('compositionend', { bubbles: true, data: '' }));
    }""")
    after = run.geometry(field, "composition-after")
    assert field.input_value() == "first line\nsecond line\nthird line"
    assert after["clientHeight"] == before["clientHeight"], (before, after)


def _transcript(run, field):
    """Growth keeps the newest message clear of the composer and an earlier reading place still."""
    page, mode = run.page, run.mode
    messages = page.locator("#chat-messages")
    _emit_ws_frame(page, {"type": "chat", "role": "assistant", "chat_id": 1,
                          "content": "Sizing transcript.\n\n" + "\n\n".join(f"Paragraph {n}" for n in range(60)),
                          "ts": "2026-10-11T10:00:00Z"})
    run.replace(field, "")
    messages.evaluate("el => { el.scrollTop = el.scrollHeight; }")
    run.settle()
    # One task: the input event and the measurement after it, with no frame in between.
    same_task = messages.evaluate("""(messages, text) => {
        const field = document.getElementById('chat-input');
        const area = document.getElementById('chat-input-area');
        const startTop = area.getBoundingClientRect().top;
        field.value = text;
        field.dispatchEvent(new Event('input', { bubbles: true }));
        const last = [...messages.children].filter(el => el.getBoundingClientRect().height > 0).at(-1);
        return { startTop, composerTop: area.getBoundingClientRect().top, lastBottom: last.getBoundingClientRect().bottom,
                 lastText: last.textContent.slice(0, 40),
                 belowBottom: messages.scrollHeight - messages.scrollTop - messages.clientHeight,
                 reserve: getComputedStyle(messages).getPropertyValue('--chat-input-reserve') };
    }""", "a\nb\nc\nd\ne")
    run.records.append({"engine": run.engine, "mode": mode, "case": "stuck-growth-same-task", **same_task})
    assert same_task["composerTop"] < same_task["startTop"] - 80, same_task  # four more rows, same task
    assert same_task["lastBottom"] <= same_task["composerTop"] + 1 and same_task["belowBottom"] <= 1, same_task
    run.settle()
    # Away from the bottom, growth leaves the message being read where it was.
    messages.evaluate("el => { el.scrollTop = el.scrollHeight / 3; }")
    run.settle()
    anchor = """messages => {
        const view = messages.getBoundingClientRect();
        const bubble = [...messages.querySelectorAll('p')]
            .find(node => node.getBoundingClientRect().top >= view.top + 80);
        if (!window.__sizingAnchor) window.__sizingAnchor = bubble;
        return window.__sizingAnchor.getBoundingClientRect().top;
    }"""
    page.evaluate("() => { window.__sizingAnchor = null; }")
    before = messages.evaluate(anchor)
    field.evaluate("el => { el.value = 'one'; el.dispatchEvent(new Event('input', { bubbles: true })); }")
    run.settle()
    field.evaluate("el => { el.value = 'a\\nb\\nc\\nd\\ne\\nf\\ng'; el.dispatchEvent(new Event('input', { bubbles: true })); }")
    run.settle()
    after = messages.evaluate(anchor)
    run.records.append({"engine": run.engine, "mode": mode, "case": "reading-anchor", "before": before, "after": after})
    assert abs(after - before) <= 1, (before, after)
    run.replace(field, "")


def _scaling(run, field, row, late_text):
    """Fractional CSS zoom, a display scale and a late font face, with the text left as it is."""
    page, mode = run.page, run.mode
    for text, rows in (("x", 1), ("a\nb\nc\nd\ne", 5), (_PARAGRAPH, None)):
        run.replace(field, text)
        field.evaluate("el => el.blur()")
        for zoom in ("1.25", "1.5", ""):
            page.evaluate("z => { document.documentElement.style.zoom = z; }", zoom)
            case = f"zoom-{zoom or '1'}-{len(text)}"
            _assert_fits(run.geometry(field, case), mode, case, rows=rows)
            if rows == 1 and zoom:
                run.shoot(row, f"zoom-{zoom}")
    if run.engine == "chromium":
        # A display scale change at the same CSS width (a window moved to another monitor).
        cdp = page.context.new_cdp_session(page)
        for scale in (1.25, 1.5, 1):
            cdp.send("Emulation.setDeviceMetricsOverride",
                     {"width": 1240, "height": 800, "deviceScaleFactor": scale, "mobile": False})
            _assert_fits(run.geometry(field, f"device-scale-{scale}"), mode, f"device-scale-{scale}")
        cdp.send("Emulation.clearDeviceMetricsOverride")
        cdp.detach()
    # A late font face rewraps the same text at the same width, with no edit: the fallback
    # re-measures when the document's font loading period settles.
    run.replace(field, late_text)
    field.evaluate("el => el.blur()")
    before = run.geometry(field, "font-before")
    _assert_fits(before, mode, "font-before", rows=1)
    status = field.evaluate(f"""async el => {{
        const face = new FontFace('SizingLate', 'url({_LATE_FONT}?sizing-late)', {_LATE_FACE});
        document.fonts.add(face);
        const style = document.createElement('style');
        style.id = 'sizing-late-font';
        style.textContent = `#chat-input {{ font-family: SizingLate, ${{getComputedStyle(el).fontFamily}}; }}`;
        document.head.append(style);
        await face.load();
        return face.status;
    }}""")
    after = run.geometry(field, "font-loaded")
    assert status == "loaded" and after["scrollHeight"] > before["scrollHeight"] + after["line"] / 2, (status, before, after)
    _assert_fits(after, mode, "font-loaded", rows=2)
    page.evaluate("() => document.getElementById('sizing-late-font').remove()")
    run.replace(field, "")


def _ink_rows(png):
    """Rows of a screenshot holding glyph ink: pixels far from the top-left background."""
    from PIL import Image

    image = Image.open(io.BytesIO(png)).convert("RGB")
    pixels, (width, height) = image.load(), image.size
    background = pixels[1, 1]
    return [any(max(abs(a - b) for a, b in zip(pixels[x, y], background)) > 48 for x in range(2, width - 2))
            for y in range(height)]


def _cut_ink(run, field, text):
    """Glyph rows of `text` set as the value that lie below the box holding it as placeholder."""
    def shot():
        box = field.bounding_box()
        return run.page.screenshot(clip=box), round(box["height"])
    placeholder, height = shot()
    run.replace(field, text)
    field.evaluate("el => el.blur()")
    run.settle()
    value, _ = shot()
    run.replace(field, "")
    field.evaluate("el => el.blur()")
    shown, held = _ink_rows(placeholder), _ink_rows(value)
    return {"placeholderInkRows": sum(shown), "valueInkRows": sum(held),
            "cutRows": sum(held[height:]), "placeholderBox": height}


def _fitting_placeholder(field):
    """Choose a multirow prefix using an uncapped text probe, not the sizer under test.

    The same translation is five rows on macOS but six on Linux. Keep a full row
    of headroom for this below-cap case; the separate long case must still scroll.
    """
    return field.evaluate("""(el, source) => {
        const style = getComputedStyle(el);
        const probe = document.createElement('textarea');
        // No id, sizing flag, listener or application height rule reaches this probe.
        for (const name of ['font', 'letter-spacing', 'padding', 'border', 'box-sizing',
                            'white-space', 'word-break', 'overflow-wrap', 'tab-size']) {
            probe.style.setProperty(name, style.getPropertyValue(name));
        }
        Object.assign(probe.style, {position: 'absolute', visibility: 'hidden',
            width: style.width, height: '1px', minHeight: '0', maxHeight: 'none',
            fieldSizing: 'fixed', overflow: 'hidden'});
        document.body.append(probe);
        const line = parseFloat(style.lineHeight);
        const pad = parseFloat(style.paddingTop) + parseFloat(style.paddingBottom);
        const cap = parseFloat(style.maxHeight);
        let text = '', natural = 0;
        try {
            for (const word of source.split(' ')) {
                probe.value = text ? `${text} ${word}` : word;
                if (probe.scrollHeight > cap - line) break;
                text = probe.value;
                natural = probe.scrollHeight;
            }
            return {text, natural, line, pad, cap};
        } finally { probe.remove(); }
    }""", _LONG_PLACEHOLDER)


def _placeholder(run, field, row):
    """An empty field holds its placeholder like text: a long one wraps and grows without a
    cut glyph or a scrollbar, and past the cap it scrolls. A placeholder rewritten in place
    (the language's dictionary arriving) is held with no edit."""
    page, mode = run.page, run.mode
    run.replace(field, "")
    field.evaluate("el => el.blur()")
    page.set_viewport_size({"width": 390, "height": 800})
    _assert_fits(run.geometry(field, "placeholder-390"), mode, "placeholder-390", rows=1)
    run.shoot(row, "placeholder-390")
    original = field.get_attribute("placeholder")
    fitting = _fitting_placeholder(field)
    assert fitting["text"] and 3 * fitting["line"] + fitting["pad"] - 1 <= fitting["natural"] <= fitting["cap"] - fitting["line"], fitting
    run.records.append({"engine": run.engine, "run": run.label, "case": "placeholder-text-oracle", **fitting})
    field.evaluate("(el, text) => { el.placeholder = text; }", fitting["text"])
    g = run.geometry(field, "long-placeholder-390")
    run.shoot(row, "long-placeholder-390")
    _assert_fits(g, mode, "long-placeholder-390")
    assert g["clientHeight"] > 3 * g["line"] and g["scrollTop"] == 0, g
    ink = _cut_ink(run, field, fitting["text"])
    run.records.append({"engine": run.engine, "run": run.label, "case": "long-placeholder-ink", **ink})
    assert ink["placeholderInkRows"] > 3 * g["line"] / 2 and ink["cutRows"] == 0, ink
    # A placeholder longer than the cap scrolls like text: the owner can reach its end.
    field.evaluate("(el, text) => { el.placeholder = text; }", " ".join([_LONG_PLACEHOLDER] * 4))
    g = run.geometry(field, "over-cap-placeholder-390")
    _assert_capped(g, mode, "over-cap-placeholder-390", at_end=False)
    box = field.bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.wheel(0, 2000)
    g = run.geometry(field, "over-cap-placeholder-wheel")
    run.shoot(row, "over-cap-placeholder-wheel")
    assert g["scrollTop"] > 0, g
    field.evaluate("(el, text) => { el.placeholder = text; }", original)
    _assert_fits(run.geometry(field, "placeholder-restored"), mode, "placeholder-restored", rows=1)
    page.set_viewport_size({"width": 1240, "height": 800})


def _project(run, data_project):
    """A Project draft is restored while its panel is still hidden and sized once shown."""
    page, mode = run.page, run.mode
    nav = page.locator(f'.nav-project-row[data-project-id="{data_project}"]')
    panel_field = page.locator("#project-panel textarea.chat-input")
    panel_row = page.locator("#project-panel .chat-text-row")
    nav.evaluate("el => el.click()")
    panel_field.wait_for(state="visible")
    run.settle()
    panel_field.click()
    page.keyboard.type("Draft line one")
    page.keyboard.press("Shift+Enter")
    page.keyboard.type("Draft line two")
    page.keyboard.press("Shift+Enter")
    page.keyboard.type("Draft line three")
    _assert_fits(run.geometry(panel_field, "project-typed"), mode, "project-typed", rows=3)
    # Close (a pressed row only opens, #1998): the instance is destroyed, its draft kept.
    close = page.locator("#project-panel-close")
    close.evaluate("el => el.click()")
    page.wait_for_function("() => !document.querySelector('#project-panel textarea.chat-input')")
    page.wait_for_function("() => document.getElementById('project-panel').hidden")
    nav.evaluate("el => el.click()")
    panel_field.wait_for(state="visible")
    g = run.geometry(panel_field, "project-reopened")
    assert panel_field.input_value() == "Draft line one\nDraft line two\nDraft line three"
    _assert_fits(g, mode, "project-reopened", rows=3)
    run.shoot(panel_row, "project-reopened")
    # A destroyed instance writes nothing afterwards.
    panel_field.evaluate("el => { window.__closedField = el; }")
    panel_field.evaluate(_WATCH_WRITES)
    close.evaluate("el => el.click()")
    page.wait_for_function("() => !window.__closedField.isConnected")
    page.set_viewport_size({"width": 900, "height": 800})
    run.settle()
    page.set_viewport_size({"width": 1240, "height": 800})
    run.settle()
    writes = page.evaluate("() => { window.__sizingWatch.disconnect(); return window.__sizingWrites; }")
    assert writes == 0, writes


def _late_text(run, field):
    """One row in the field's own face, two in the late face: measured in both here."""
    near = field.evaluate(f"""async el => {{
        const face = new FontFace('SizingCalibrate', 'url({_LATE_FONT})', {_LATE_FACE});
        document.fonts.add(face);
        await face.load();
        return ({_NEAR_WRAP})(el, 'SizingCalibrate');
    }}""")
    run.records.append({"engine": run.engine, "run": run.label, "case": "late-font-near-wrap", **near})
    assert near["own"] <= near["room"] - 12 and 2 * near["room"] > near["other"] >= near["room"] + 12, near
    return near["text"]


def _early_font(browser, url, run, variant, text):
    """A font loading period already running when the field attaches: its end re-measures.
    The face stays on the wire until the near-wrap text is in the field."""
    context = browser.new_context(viewport={"width": 1240, "height": 800})
    held = []
    context.route(f"**{_LATE_FONT}?sizing-early", lambda route: held.append(route))
    page = context.new_page()
    if variant == "denied":
        page.add_init_script(f"({_DENY_FIELD_SIZING})()")
    page.add_init_script(f"""(() => {{
        const face = new FontFace('SizingEarly', 'url({_LATE_FONT}?sizing-early)', {_LATE_FACE});
        document.fonts.add(face);
        face.load().catch(() => {{}});
        addEventListener('DOMContentLoaded', () => {{
            const style = document.createElement('style');
            style.textContent = `#chat-input {{ font-family: SizingEarly, ${{getComputedStyle(document.body).fontFamily}}; }}`;
            document.head.append(style);
        }});
    }})()""")
    page.goto(url, wait_until="domcontentloaded")
    field = page.locator("#chat-input")
    field.wait_for(state="visible")
    run = _Run(page, run.engine, run.mode, run.evidence, run.records, variant)
    run.replace(field, text)
    field.evaluate("el => el.blur()")
    before = run.geometry(field, "early-font-before")
    assert held and page.evaluate("() => document.fonts.status") == "loading", held
    _assert_fits(before, run.mode, "early-font-before", rows=1)
    for route in held:
        route.continue_()
    page.wait_for_function("() => document.fonts.status === 'loaded'")
    _assert_fits(run.geometry(field, "early-font-loaded"), run.mode, "early-font-loaded", rows=2)
    run.shoot(page.locator("#chat-input-area .chat-text-row"), "early-font-loaded")
    context.close()


@pytest.mark.ui_browser
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_message_field_sizing(direct_server_with_data, engine):
    from playwright.sync_api import sync_playwright
    from ouroboros.projects_registry import create_project

    data = direct_server_with_data["data_dir"]
    url = direct_server_with_data["url"]
    project = create_project(data, "sizing-room", name="Sizing room")
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", data.parent))
    evidence.mkdir(parents=True, exist_ok=True)
    records = []
    with sync_playwright() as pw:
        # The scrollbar under test must be drawn: headless Chromium hides scrollbars by default.
        options = {"ignore_default_args": ["--hide-scrollbars"]} if engine == "chromium" else {}
        browser = getattr(pw, engine).launch(**options)
        try:
            # `selected`: the mode the helper picks in this engine; `denied`: no field-sizing.
            for variant in ("selected", "denied"):
                deny = variant == "denied"
                context = browser.new_context(viewport={"width": 1240, "height": 800})
                page = context.new_page()
                page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
                page.add_init_script(f"({_RECORD_ERRORS})()")
                if deny:
                    page.add_init_script(f"({_DENY_FIELD_SIZING})()")
                page.goto(url, wait_until="domcontentloaded")
                page.wait_for_function("() => window.__testSockets?.some(s => s.readyState === 1)")
                field = page.locator("#chat-input")
                field.wait_for(state="visible")
                selected = field.evaluate("el => el.dataset.sizing")
                records.append({"engine": engine, "run": variant, "case": "engine", "sizing": selected,
                                "fieldSizingSupported": page.evaluate("() => CSS.supports('field-sizing', 'content')"),
                                "userAgent": page.evaluate("() => navigator.userAgent")})
                assert selected in ({"fit"} if deny else {"content", "fit"}), selected
                run = _Run(page, engine, "content" if selected == "content" else "measured", evidence, records, variant)
                row = page.locator("#chat-input-area .chat-text-row")
                _typing(run, field, row)
                _layout(run, field, row)
                _keyboard(run, field)
                _transcript(run, field)
                late_text = _late_text(run, field)
                _scaling(run, field, row, late_text)
                _placeholder(run, field, row)
                _project(run, project["id"])
                errors = page.evaluate("() => window.__sizingErrors")
                assert not [e for e in errors if "ResizeObserver" in e], errors
                context.close()
                _early_font(browser, url, run, variant, late_text)
                # Device scale factor: another proxy for a scaled display.
                context = browser.new_context(viewport={"width": 1240, "height": 800}, device_scale_factor=1.5)
                page = context.new_page()
                if deny:
                    page.add_init_script(f"({_DENY_FIELD_SIZING})()")
                page.goto(url, wait_until="domcontentloaded")
                field = page.locator("#chat-input")
                field.wait_for(state="visible")
                run = _Run(page, engine, run.mode, evidence, records, variant)
                for case, text, rows in (("dpr-1.5-one", "x", 1), ("dpr-1.5-five", "a\nb\nc\nd\ne", 5)):
                    run.replace(field, text)
                    _assert_fits(run.geometry(field, case), run.mode, case, rows=rows)
                run.shoot(page.locator("#chat-input-area .chat-text-row"), "dpr-1.5")
                context.close()
        finally:
            (evidence / f"composer-sizing-{engine}.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
            browser.close()
