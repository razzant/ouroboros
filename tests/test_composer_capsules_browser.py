"""The composer's two owner capsules in the real composer (DESIGN "Composer owner controls").

On a real server in Chromium and WebKit: `Context Max` opens inline into exactly three modes
(a choice posts the owner endpoint at once, a refusal says why and keeps the saved mode, a reload
reads it back; Enter opens onto the chosen mode, arrows choose a neighbour, Esc returns to the
head); one capsule is open at a time — one press on the other capsule switches on a desktop, and on
a phone, where an open strip stands alone, the capsule that opens next is never left hidden by the
first one's layout. Effort's max bracket keeps clear of the capsule's rounded end — measured
against the rounded contour at the bracket's corners — beside the others, alone, at the tightest
padding, at a scrolling strip's end and during a drag, while Context's pill-shaped thumb stays
concentric. Closed, the capsules keep their names while the line holds them. Reduced motion
removes the transitions; a resize refits an open strip. On no animation frame of an opening or
a closing does the row wrap; a translated Context strip that scrolls on a 320 px phone shows the
chosen mode whole on opening, after Home and End, and when a resize makes the open strip scroll,
while a later relayout keeps where the owner scrolled it. Right to left the strips mirror: they
scroll leftwards, keep the moved mode or bracket whole, fade where levels hide, and a press lands
on the level under it. A closed row whose long translated values still wrap without the names
cuts the longest values to one width (the whole word stays in the button's name and the open strip),
and no frame of an opening or a closing wraps there, left to right or right to left.
"""
from __future__ import annotations

import json

import pytest

from tests.test_effort_range_browser import _evidence_dir, _launch, _open_chat, _painted, _shot
from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401 - pytest fixture import
from tests.ui_chat_viewport_smoke import set_viewport

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

CONTEXT = "#page-chat .chat-context-mode"
EFFORT = "#page-chat .chat-effort-range"
ROW = """() => {
    const row = document.querySelector('#page-chat .chat-toolbar-row');
    const rect = (node) => node.getBoundingClientRect();
    const shown = (node) => Boolean(node) && getComputedStyle(node).display !== 'none';
    const ctx = row.querySelector('.chat-context-mode');
    const eff = row.querySelector('.chat-effort-range');
    const kids = [...row.children].filter(shown);
    const tops = kids.map((kid) => rect(kid).top);
    const segs = [...ctx.querySelectorAll('.chat-seg')];
    const active = document.activeElement;
    return {
        ctx: {
            open: ctx.dataset.open, mode: ctx.dataset.contextMode, fit: ctx.dataset.fit || '', shown: shown(ctx),
            head: ctx.querySelector('.chat-capsule-head').textContent, value: ctx.querySelector('.chat-capsule-value').textContent,
            checked: segs.filter((seg) => seg.getAttribute('aria-checked') === 'true').map((seg) => seg.dataset.mode),
            tabbable: segs.filter((seg) => seg.tabIndex === 0).map((seg) => seg.dataset.mode),
            focused: active?.dataset?.mode || (active === ctx.querySelector('.chat-capsule-head') ? 'head' : ''),
            right: rect(ctx).right, disabled: ctx.dataset.disabled || '',
        },
        eff: { open: eff.dataset.open, fit: eff.dataset.fit || '', shown: shown(eff), right: rect(eff).right, head: eff.querySelector('.chat-capsule-head').textContent },
        subShown: shown(row.querySelector('.chat-subagents-mode')),  // the third capsule, closed
        menuShown: shown(row.querySelector('.composer-menu')),  // a room's ☰ (the stack layout shows it)
        namesShown: [...row.querySelectorAll('.chat-capsule-name')].map(shown),
        solo: row.dataset.ownerSolo || '', compact: row.dataset.ownerCompact || '',
        oneLine: Math.max(...tops) - Math.min(...tops) < 2, rowHeight: rect(row).height,
        rowRight: rect(row).right - parseFloat(getComputedStyle(row).paddingRight),
        docScroll: document.documentElement.scrollWidth > innerWidth + 1,
    };
}"""
# A mark's clearance from the capsule's inner contour: the straight gutter at the end, the
# rounded contour's distance at the mark's top and bottom corners, and its vertical insets.
CLEARANCE = """sel => {
    const el = document.querySelector(sel);
    const style = getComputedStyle(el);
    const shell = el.getBoundingClientRect();
    const border = parseFloat(style.borderRightWidth) || 0;
    const top = shell.top + border, bottom = shell.bottom - border, right = shell.right - border;
    const r = (bottom - top) / 2, cy = top + r, cx = right - r;
    const contour = (y) => cx + Math.sqrt(Math.max(0, r * r - (y - cy) ** 2));
    const mark = (node) => {
        if (!node) return null;
        const m = node.getBoundingClientRect();
        return { gutter: right - m.right, corners: Math.min(contour(m.top), contour(m.bottom)) - m.right, top: m.top - top, bottom: bottom - m.bottom };
    };
    return {
        cap: mark(el.querySelector('.chat-effort-max')), thumb: mark(el.querySelector('.chat-context-thumb')),
        fit: el.dataset.fit || '', scroll: el.querySelector('[data-scroll]')?.dataset.scroll || '',
    };
}"""
# Every frame of an opening and then a closing, read in requestAnimationFrame: where the row's
# shown controls stand (a spread of their tops means a second line) and the capsule's width.
FRAMES = """async ({ sel, frames }) => {
    const row = document.querySelector('#page-chat .chat-toolbar-row');
    const el = document.querySelector(sel);
    const head = el.querySelector('.chat-capsule-head');
    const shown = (node) => getComputedStyle(node).display !== 'none';
    // The chosen mode or the recommended level, whole inside the visible strip.
    const whole = () => {
        const word = el.querySelector('.chat-seg[aria-checked="true"], .chat-effort-seg[data-rec="true"]').getBoundingClientRect();
        const s = el.querySelector('[data-scroll]').getBoundingClientRect();
        const b = el.querySelector('.chat-capsule-body').getBoundingClientRect();
        return word.width > 0 && word.left >= Math.max(s.left, b.left) - 0.5 && word.right <= Math.min(s.right, b.right) + 0.5;
    };
    const sample = () => {
        const tops = [...row.children].filter(shown).map((kid) => kid.getBoundingClientRect().top);
        return { spread: Math.max(...tops) - Math.min(...tops), height: row.getBoundingClientRect().height, width: el.getBoundingClientRect().width,
                 whole: whole(), compact: row.dataset.ownerCompact || '' };
    };
    const frame = () => new Promise((resolve) => requestAnimationFrame(resolve));
    // Every frame a press can paint, however few this host draws (a loaded WebKit draws two or three per
    // transition): the row's transitions are held from the press on, stepped through their whole timeline, then
    // finished, and the frames after that are sampled as they come.
    const held = () => {
        const moving = document.getAnimations().filter((animation) => 'transitionProperty' in animation
            && row.contains(animation.effect?.target));
        moving.forEach((animation) => animation.pause());
        return moving;
    };
    const run = async () => {
        held();
        await frame();  // the press's own frame work runs; what it starts is held too
        const moving = held();
        const end = Math.max(0, ...moving.map((animation) => animation.effect.getComputedTiming().endTime));
        const out = [];
        for (let i = 0; i < frames; i += 1) {
            moving.forEach((animation) => { animation.currentTime = end * i / (frames - 1); });
            // The step is read once the product has refit to it: its ResizeObserver hands the fit to the next
            // frame (composer_owner_controls.js observeLayout), so that frame runs before the read.
            for (let wait = 0; wait < 3; wait += 1) await frame();
            out.push(sample());
        }
        moving.forEach((animation) => animation.finish());
        for (let i = 0; i < 4; i += 1) { await frame(); out.push(sample()); }
        return out;
    };
    const press = () => head.dispatchEvent(new MouseEvent('click', { bubbles: true, detail: 1 }));
    const closed = sample();
    press();
    const fit = el.dataset.fit || '';
    const opening = await run();
    press();
    const closing = await run();
    return { closed, fit, opening, closing, open: el.dataset.open };
}"""
# The capsule once the closing settled: its width, a value that shows whole, nothing held.
SETTLED = """sel => {
    const el = document.querySelector(sel);
    const value = el.querySelector('.chat-capsule-value');
    return { width: el.getBoundingClientRect().width, clipped: value.scrollWidth > value.clientWidth + 1,
             held: el.style.getPropertyValue('--capsule-value-width') + el.style.getPropertyValue('--capsule-body-width') };
}"""
# A translated install (the house idiom: the gateway's /api/ui/i18n answer, stubbed): the overlay
# rewrites the modes in place, and on a 320 px phone their longer words scroll inside the strip.
TRANSLATED = {"language": "ru", "english": False, "revision": 1, "entries": {
    english: {"text": text, "provenance": "imported"}
    for english, text in (("Context", "Контекст"), ("Nano", "Компактный"), ("Low", "Сокращённый"), ("Max", "Максимальный"))}}
# A mode's word against the strip that shows it: whole (inside the strip's box) and where the strip fades.
SHOWN = """mode => {
    const el = document.querySelector('#page-chat .chat-context-mode');
    const strip = el.querySelector('.chat-context-strip');
    const seg = el.querySelector(`.chat-seg[data-mode="${mode}"]`);
    const s = strip.getBoundingClientRect();
    const m = seg.getBoundingClientRect();
    const word = seg.querySelector('.chat-seg-text').getBoundingClientRect();
    return {
        fit: el.dataset.fit || '', scroll: strip.dataset.scroll || '', scrollLeft: strip.scrollLeft, word: seg.textContent,
        strip: [s.left, s.right], seg: [m.left, m.right], wordBox: [word.left, word.right],
        // offsetLeft/clientWidth/scrollWidth round to CSS pixels: an end can clip <1 px of
        // button padding while every glyph stays visible. Never excuse clipped text.
        whole: m.left >= s.left - 1 && m.right <= s.right + 1 && word.left >= s.left && word.right <= s.right,
        focused: document.activeElement?.dataset?.mode || '', docScroll: document.documentElement.scrollWidth > innerWidth + 1,
        fade: getComputedStyle(strip).maskImage || getComputedStyle(strip).webkitMaskImage || '',
    };
}"""
# A right-to-left install (its language profile says `rtl`; the words are the test's own, longer
# than the English): both strips scroll on a 320 px phone.
RTL = {"language": "ar", "english": False, "revision": 1, "profile": {"direction": "rtl"}, "entries": {
    english: {"text": text, "provenance": "imported"}
    for english, text in (("Context", "السياق"), ("Nano", "السياق المصغّر"), ("Low", "السياق المنخفض"), ("Max", "السياق الكامل"))}}
# Effort's marks against its strip and its shell: boxes as [left, right], whole = inside the strip.
MARKS = """() => {
    const el = document.querySelector('#page-chat .chat-effort-range');
    const strip = el.querySelector('.chat-effort-strip');
    const s = strip.getBoundingClientRect();
    const box = (sel) => { const r = el.querySelector(sel).getBoundingClientRect(); return [r.left, r.right]; };
    const whole = (sel) => { const [l, r] = box(sel); return l >= s.left - 0.5 && r <= s.right + 0.5; };
    const shell = el.getBoundingClientRect();
    const border = parseFloat(getComputedStyle(el).borderLeftWidth) || 0;
    return {
        fit: el.dataset.fit || '', scroll: strip.dataset.scroll || '', dir: getComputedStyle(strip).direction,
        fade: getComputedStyle(strip).maskImage || getComputedStyle(strip).webkitMaskImage || '',
        band: box('.chat-effort-band'), pill: box('.chat-effort-rec'), seg: box('.chat-effort-seg[data-rec="true"]'),
        min: box('.chat-effort-min'), max: box('.chat-effort-max'), pillWhole: whole('.chat-effort-rec'),
        maxWhole: whole('.chat-effort-max'), ultraWhole: whole('[data-effort-seg="ultra"]'),
        none: box('[data-effort-seg="none"]'), shellLeft: shell.left + border,
        inset: el.querySelector('.chat-effort-max').getBoundingClientRect().top - shell.top - border,
        focused: document.activeElement?.dataset?.handle || '',
    };
}"""
# Long translated values on a 320 px phone, left to right and right to left (the test's own words).
LONG_VALUES = {
    "ltr": ({"language": "ru", "english": False, "revision": 1, "entries": {
        english: {"text": text, "provenance": "imported"}
        for english, text in (("Context", "Контекст"), ("Nano", "Компактный"), ("Low", "Сокращённый"), ("Max", "Максимальный"),
                              ("Effort", "Усилие"), ("code:effort.tier.medium", "Средний"), ("Swarm", "Рой"))}},
            "Максимальный", "Средний"),
    "rtl": ({"language": "ar", "english": False, "revision": 1, "profile": {"direction": "rtl"}, "entries": {
        english: {"text": text, "provenance": "imported"}
        for english, text in (("Context", "السياق"), ("Nano", "السياق المصغّر"), ("Low", "السياق المنخفض"), ("Max", "السياق الكامل"),
                              ("Effort", "الجهد"), ("code:effort.tier.medium", "متوسط الجهد"), ("Swarm", "السرب"))}},
            "السياق الكامل", "متوسط الجهد"),
}
# The closed capsules' values against their shells, and every shown control against the row's content box.
CAPPED = """() => {
    const row = document.querySelector('#page-chat .chat-toolbar-row');
    const rs = getComputedStyle(row);
    const r = row.getBoundingClientRect();
    const box = (node) => { const b = node.getBoundingClientRect(); return [b.left, b.right]; };
    const content = [r.left + parseFloat(rs.paddingLeft), r.right - parseFloat(rs.paddingRight)];
    const kids = [...row.children].filter((kid) => getComputedStyle(kid).display !== 'none' && kid.getBoundingClientRect().width > 0);
    return {
        contained: kids.every((kid) => { const [l, rr] = box(kid); return l >= content[0] - 1 && rr <= content[1] + 1; }),
        capsules: [...row.querySelectorAll('.chat-capsule')].map((el) => {
            const value = el.querySelector('.chat-capsule-value');
            const head = el.querySelector('.chat-capsule-head');
            const border = parseFloat(getComputedStyle(el).borderLeftWidth) || 0;
            const [l, rr] = box(el);
            return { word: value.textContent, cut: value.scrollWidth > value.clientWidth + 1, value: box(value),
                     shell: [l + border, rr - border], label: head.getAttribute('aria-label') || '', title: head.title };
        }),
    };
}"""


def _row(page) -> dict:
    return page.evaluate(ROW)


def _until(page, predicate: str, timeout: int = 5_000) -> None:
    page.wait_for_function(f"() => {{ const r = ({ROW})(); return {predicate}; }}", timeout=timeout)


def _wait_for_transitions(page, selector: str) -> None:
    # A busy renderer can reach the wall-clock deadline before the opening's final frame.
    page.wait_for_function("""sel => document.querySelector(sel).getAnimations({subtree: true})
        .every(animation => !animation.pending && animation.playState !== 'running')""", arg=selector, timeout=5_000)


def _morph_settled(page) -> None:
    # A capsule's morph ends with its glass: the controls that stepped aside leave the line then.
    page.wait_for_function("""() => { const row = document.querySelector('#page-chat .chat-toolbar-row');
        return !('ownerLeaving' in row.dataset) && row.getAnimations({subtree: true})
            .filter(animation => animation.effect?.getComputedTiming().endTime !== Infinity)
            .every(animation => !animation.pending && animation.playState !== 'running'); }""", timeout=10_000)


def _record(tmp_path, name: str, facts) -> None:
    (_evidence_dir(tmp_path) / f"{name}.json").write_text(json.dumps(facts, indent=1), encoding="utf-8")


def _between(frames, ends) -> int:
    """The frames whose capsule stands clearly between its closed and its open width."""
    return sum(min(ends) + 2 < frame["width"] < max(ends) - 2 for frame in frames)


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_context_capsule_chooses_one_of_three_modes_and_reads_back(direct_server_with_data, engine, tmp_path):  # noqa: F811
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    with sync_playwright() as pw:
        browser, page = _launch(pw, engine, viewport={"width": 1440, "height": 900})
        try:
            _open_chat(page, url, width=1440, theme="light")
            closed = _row(page)
            assert closed["ctx"]["head"] == "ContextMax" and closed["ctx"]["checked"] == ["max"], closed
            assert closed["ctx"]["tabbable"] == [] and closed["oneLine"] and closed["namesShown"] == [True, True, True], closed
            _shot(page, tmp_path, f"capsules-closed-{engine}-light")
            posts = []
            page.on("request", lambda r: posts.append(json.loads(r.post_data)) if r.url.endswith("/api/owner/context-mode") else None)
            page.click(f"{CONTEXT} .chat-capsule-head")
            _until(page, "r.ctx.open === 'true'")
            page.wait_for_timeout(450)
            opened = _row(page)
            assert opened["ctx"]["fit"] == "inline" and opened["eff"]["shown"] and opened["subShown"], opened
            assert opened["oneLine"] and opened["ctx"]["right"] <= opened["rowRight"] + 1 and not opened["docScroll"], opened
            assert posts == [], "opening saves nothing"
            _shot(page, tmp_path, f"capsules-context-open-{engine}-light")
            # A press on a mode posts it at once; the capsule shows it once the server took it.
            with page.expect_response(lambda r: r.url.endswith("/api/owner/context-mode"), timeout=30_000) as waited:
                page.click(f'{CONTEXT} .chat-seg[data-mode="low"]')
            assert waited.value.status == 200 and posts[-1] == {"mode": "low"}, posts
            _until(page, "r.ctx.mode === 'low' && r.ctx.disabled === 'false'")
            assert _row(page)["ctx"]["checked"] == ["low"]
            # Keyboard: Esc returns to the head, Enter opens onto the chosen mode, an arrow chooses.
            page.keyboard.press("Escape")
            _until(page, "r.ctx.open === 'false' && r.ctx.focused === 'head'")
            assert _row(page)["ctx"]["head"] == "ContextLow"
            page.keyboard.press("Enter")
            _until(page, "r.ctx.open === 'true' && r.ctx.focused === 'low'")
            with page.expect_response(lambda r: r.url.endswith("/api/owner/context-mode"), timeout=30_000):
                page.keyboard.press("ArrowLeft")
            _until(page, "r.ctx.mode === 'nano' && r.ctx.focused === 'nano' && r.ctx.tabbable.join() === 'nano'")
            _shot(page, tmp_path, f"capsules-context-nano-{engine}-light")
            # A refusal: the server's sentence, the saved mode stays.
            page.route("**/api/owner/context-mode", lambda route: route.fulfill(
                status=409, content_type="application/json",
                body=json.dumps({"error": "Context mode refused by the test."})))
            page.click(f'{CONTEXT} .chat-seg[data-mode="max"]')
            page.wait_for_selector(".toast", timeout=10_000)
            assert "Context mode refused by the test." in page.locator(".toast").first.inner_text()
            _until(page, "r.ctx.mode === 'nano' && r.ctx.disabled === 'false' && r.ctx.checked.join() === 'nano'")
            page.unroute("**/api/owner/context-mode")
            # The one global value: a reload reads Nano back from /api/state, in Dark chosen as Settings does.
            page.evaluate("() => window.ouroTheme.set('dark')")
            page.reload(wait_until="domcontentloaded")
            page.wait_for_function(f"() => document.querySelector('{EFFORT}')?.dataset.known === 'true'", timeout=30_000)
            _until(page, "r.ctx.mode === 'nano' && r.ctx.head === 'ContextNano'", timeout=30_000)
            assert _painted(page) == ["dark", "dark"], "the dark shots show the dark theme"
            _shot(page, tmp_path, f"capsules-closed-nano-{engine}-dark")
            page.click(f"{CONTEXT} .chat-capsule-head")
            _until(page, "r.ctx.open === 'true'")
            page.wait_for_timeout(450)
            _shot(page, tmp_path, f"capsules-context-open-{engine}-dark")
        finally:
            browser.close()


@pytest.mark.parametrize("engine,width,touch", [("chromium", 1440, False), ("webkit", 1440, False), ("chromium", 390, True),
                                                ("webkit", 390, True), ("chromium", 440, False)])
def test_one_capsule_open_at_a_time_and_a_narrow_switch_keeps_both_reachable(direct_server_with_data, engine, width, touch, tmp_path):  # noqa: F811
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    context = {"viewport": {"width": width, "height": 860}}
    if touch:
        context.update(is_mobile=True, has_touch=True)
    with sync_playwright() as pw:
        browser, page = _launch(pw, engine, **context)
        try:
            _open_chat(page, url, width=width, height=860)
            press = page.tap if touch else page.click
            closed = _row(page)
            assert closed["oneLine"] and not closed["docScroll"], closed
            assert closed["menuShown"] is (width < 1000), ("a phone's room keeps ☰ first in the row", closed)
            seen = {"closed": closed}
            for first, second in (("effort", "context"), ("context", "effort")):
                heads = {"effort": f"{EFFORT} .chat-capsule-head", "context": f"{CONTEXT} .chat-capsule-head"}
                short = {"effort": "eff", "context": "ctx"}
                press(heads[first])
                _until(page, f"r.{short[first]}.open === 'true'")
                page.wait_for_timeout(450)
                _morph_settled(page)  # a slow WebKit runner is still mid-morph after the 450 ms
                one = _row(page)
                if one[short[first]]["fit"] == "solo":
                    # Alone on the line: the room's ☰, the pills and the other capsule step aside; an
                    # outside press gives the line back before the other capsule can be reached.
                    assert one["solo"] == first and not one[short[second]]["shown"] and not one["subShown"], one
                    assert not one["menuShown"], one
                    press("#chat-messages")
                    _until(page, f"r.{short[first]}.open === 'false' && r.solo === '' && r.{short[second]}.shown && r.subShown"
                                 f" && r.menuShown === {str(closed['menuShown']).lower()}")
                else:
                    assert width > 1000 and one["solo"] == "" and one[short[second]]["shown"], one
                # The second capsule opens with one press (on a desktop the first one closes at once),
                # visible and contained: never hidden by the first one's layout.
                press(heads[second])
                _until(page, f"r.{short[second]}.open === 'true' && r.{short[first]}.open === 'false'")
                page.wait_for_timeout(450)
                _morph_settled(page)
                two = _row(page)
                assert two[short[second]]["shown"] and two["solo"] in ("", second), two
                assert two["oneLine"] and two[short[second]]["right"] <= two["rowRight"] + 1 and not two["docScroll"], two
                assert abs(two["rowHeight"] - closed["rowHeight"]) <= 1, (two, closed)
                _shot(page, tmp_path, f"capsules-switch-{first}-to-{second}-{engine}-{width}")
                seen[f"{first}->{second}"] = [one, two]
                page.keyboard.press("Escape")
                _until(page, "r.ctx.open === 'false' && r.eff.open === 'false'")
                back = _row(page)
                assert back["ctx"]["shown"] and back["eff"]["shown"] and back["subShown"] and back["solo"] == "", back
                assert back["menuShown"] is closed["menuShown"] and back["oneLine"], back
            _record(tmp_path, f"capsules-switch-{engine}-{width}", seen)
        finally:
            browser.close()


@pytest.mark.parametrize("engine,width", [("chromium", 1440), ("webkit", 1440), ("chromium", 390), ("webkit", 390),
                                          ("chromium", 360), ("chromium", 320)])
def test_the_max_bracket_keeps_clear_of_the_rounded_end(direct_server_with_data, engine, width, tmp_path):  # noqa: F811
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    phone = width < 1000
    context = {"viewport": {"width": width, "height": 860}, "device_scale_factor": 2}
    if phone:
        context.update(is_mobile=True, has_touch=True)
    with sync_playwright() as pw:
        browser, page = _launch(pw, engine, **context)
        try:
            _open_chat(page, url, width=width, height=860)
            measured = {}
            # Context's thumb at Max, the last mode: a pill concentric with the rounded end.
            (page.tap if phone else page.click)(f"{CONTEXT} .chat-capsule-head")
            _until(page, "r.ctx.open === 'true'")
            _wait_for_transitions(page, CONTEXT)
            thumb = page.evaluate(CLEARANCE, CONTEXT)["thumb"]
            assert abs(thumb["gutter"] - thumb["top"]) <= 1 and abs(thumb["top"] - thumb["bottom"]) <= 1, thumb
            measured["context-thumb-max"] = thumb
            page.keyboard.press("Escape")
            # Effort's max bracket at Ultra, the last level.
            (page.tap if phone else page.click)(f"{EFFORT} .chat-capsule-head")
            _until(page, "r.eff.open === 'true'")
            page.wait_for_timeout(450)
            page.focus(f"{EFFORT} .chat-effort-max")
            with page.expect_response(lambda r: r.url.endswith("/api/owner/effort-range"), timeout=30_000):
                page.keyboard.press("End")
            page.wait_for_timeout(300)
            page.evaluate(f"() => {{ const s = document.querySelector('{EFFORT} .chat-effort-strip'); s.scrollLeft = s.scrollWidth; }}")
            page.wait_for_timeout(200)
            at_end = page.evaluate(CLEARANCE, EFFORT)
            measured["effort-max-ultra"] = at_end
            # Fonts decide when a phone needs internal scrolling; the contour must clear in
            # either measured mode. Wide desktop and the dedicated scroll cases cover both ends.
            assert at_end["fit"] in (("inline",) if not phone else ("solo", "overflow")), at_end
            if at_end["fit"] == "overflow":
                assert at_end["scroll"] == "end", at_end
            cap = at_end["cap"]
            # The square bracket clears the rounded end: its corners stand off the contour, and the
            # straight gutter is at least the band inset it keeps from the top and the bottom.
            assert cap["corners"] >= 2, at_end
            assert cap["gutter"] >= cap["top"] - 0.5 and abs(cap["top"] - cap["bottom"]) <= 1, at_end
            _shot(page, tmp_path, f"capsules-effort-ultra-end-{engine}-{width}")
            if not phone:
                # A drag: back to X-High, then the pointer carries the bracket to Ultra; mid-drag
                # (the pointer still down) the bracket already keeps its clearance.
                page.keyboard.press("ArrowLeft")
                page.wait_for_timeout(300)
                box = page.locator(f"{EFFORT} .chat-effort-max").bounding_box()
                ultra = page.locator(f'{EFFORT} [data-effort-seg="ultra"]').bounding_box()
                page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
                page.mouse.down()
                page.mouse.move(ultra["x"] + ultra["width"] - 2, ultra["y"] + ultra["height"] / 2, steps=8)
                page.wait_for_timeout(100)
                dragging = page.evaluate(CLEARANCE, EFFORT)
                measured["effort-max-mid-drag"] = dragging
                with page.expect_response(lambda r: r.url.endswith("/api/owner/effort-range"), timeout=30_000):
                    page.mouse.up()
                assert dragging["cap"]["corners"] >= 2, dragging
            _record(tmp_path, f"capsules-clearance-{engine}-{width}", measured)
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_closed_names_reduced_motion_and_a_resize_that_refits(direct_server_with_data, engine, tmp_path):  # noqa: F811
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    with sync_playwright() as pw:
        browser, page = _launch(pw, engine, viewport={"width": 1440, "height": 900}, reduced_motion="reduce")
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            _open_chat(page, url, width=1440)
            durations = page.evaluate(f"""() => [...document.querySelectorAll('{EFFORT} .chat-capsule-body, {EFFORT} .chat-capsule-value, {CONTEXT} .chat-context-thumb')]
                .map((node) => getComputedStyle(node).transitionDuration)""")
            assert all(set(d.replace(" ", "").split(",")) == {"0s"} for d in durations), durations
            page.click(f"{EFFORT} .chat-capsule-head")
            _until(page, "r.eff.open === 'true' && r.eff.fit === 'inline'")
            # The pointer leaves the row: a pointer resting on a capsule is hover, which opens a closed one (WebKit
            # moves it over again as the row reflows); this flow is a press, two resizes and Escape.
            page.mouse.move(2, 2)
            facts = {"wide": _row(page)}
            # The window narrows to a phone's width with the strip open: it refits alone, one line.
            set_viewport(page, 390, 900)
            _until(page, "['solo', 'overflow'].includes(r.eff.fit) && r.solo === 'effort'")
            facts["narrow"] = _row(page)
            assert facts["narrow"]["oneLine"] and not facts["narrow"]["docScroll"], facts
            assert facts["narrow"]["eff"]["right"] <= facts["narrow"]["rowRight"] + 1, facts
            set_viewport(page, 1440, 900)
            _until(page, "r.eff.fit === 'inline' && r.solo === '' && r.ctx.shown && r.subShown")
            page.keyboard.press("Escape")
            # Probe both sides of the actual named row's width. System fonts differ between
            # Linux and macOS; 390/340 are not product breakpoints (DESIGN).
            _until(page, "r.eff.open === 'false' && r.namesShown.every(Boolean) && r.oneLine")
            page.wait_for_function("""sel => !document.querySelector(sel).style.getPropertyValue('--capsule-body-width')""", arg=EFFORT)
            named_width = page.evaluate("""() => {
                const row = document.querySelector('#page-chat .chat-toolbar-row'), css = getComputedStyle(row);
                const kids = [...row.children].filter(n => getComputedStyle(n).display !== 'none');
                return Math.ceil(kids.reduce((sum, n) => sum + n.getBoundingClientRect().width, 0)
                    + parseFloat(css.columnGap) * (kids.length - 1) + parseFloat(css.paddingLeft) + parseFloat(css.paddingRight));
            }""")
            for index, (width, names) in enumerate(((named_width + 4, True), (named_width - 4, False), (named_width + 4, True))):
                page.locator('#page-chat .chat-toolbar-row').evaluate('(row, width) => row.style.width = width + "px"', width)
                _until(page, f"r.namesShown.every(n => n === {str(names).lower()}) && r.oneLine")
                facts[f"closed-{index}-{width}"] = _row(page)
                _shot(page, tmp_path, f"capsules-closed-names-{engine}-{index}-{width}")
            assert not errors, errors
            _record(tmp_path, f"capsules-names-motion-resize-{engine}", facts)
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_opening_and_closing_never_wrap_the_row_on_any_frame(direct_server_with_data, engine, tmp_path):  # noqa: F811
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    with sync_playwright() as pw:
        browser, page = _launch(pw, engine, viewport={"width": 1240, "height": 900})
        try:
            _open_chat(page, url, width=1240)
            runs = {}
            # The value gives way while the strip grows, and comes back while it collapses: across
            # the widths where a strip opens beside the others, the capsule never stands wider than
            # closed or open (all the fit measures), so no frame takes a second line.
            for width in range(700, 1250, 50):
                page.set_viewport_size({"width": width, "height": 900})
                _until(page, "r.oneLine && r.ctx.open === 'false' && r.eff.open === 'false'")
                page.wait_for_timeout(250)
                for name, selector in (("context", CONTEXT), ("effort", EFFORT)):
                    run = page.evaluate(FRAMES, {"sel": selector, "frames": 40})
                    page.wait_for_timeout(500)
                    run["settled"] = page.evaluate(SETTLED, selector)
                    runs[f"{name}-{width}"] = run
            _record(tmp_path, f"capsules-frames-{engine}", runs)
            wrapped = {key: [round(frame["height"]) for frame in run["opening"] + run["closing"] if frame["spread"] >= 2]
                       for key, run in runs.items()}
            assert not any(wrapped.values()), {key: heights for key, heights in wrapped.items() if heights}
            for key, run in runs.items():
                ends = (run["closed"]["width"], run["opening"][-1]["width"])
                peak = max(frame["width"] for frame in run["opening"] + run["closing"])
                assert peak <= max(ends) + 1, (key, peak, ends)
                assert run["open"] == "false" and not run["settled"]["clipped"] and run["settled"]["held"] == "", (key, run["settled"])
                assert abs(run["settled"]["width"] - run["closed"]["width"]) <= 0.5, (key, run["settled"], run["closed"])
                if run["fit"] == "inline":  # it still animates: frames between the two widths, both ways
                    assert _between(run["opening"], ends) >= 3 and _between(run["closing"], ends) >= 3, key
            inline = [key for key, run in runs.items() if run["fit"] == "inline"]
            assert len(inline) >= 10, f"the strips must animate beside the others somewhere: {inline}"
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_translated_context_strip_that_scrolls_keeps_the_chosen_mode_in_view(direct_server_with_data, engine, tmp_path):  # noqa: F811
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    with sync_playwright() as pw:
        browser, page = _launch(pw, engine, viewport={"width": 320, "height": 860}, is_mobile=True, has_touch=True)
        try:
            page.route("**/api/ui/i18n", lambda route: route.fulfill(json=TRANSLATED))
            page.route("**/api/ui/i18n/missing", lambda route: route.fulfill(json={"ok": True}))
            _open_chat(page, url, width=320, height=860)
            page.wait_for_function(f"""() => document.querySelector('{CONTEXT} .chat-seg[data-mode="max"]')?.textContent === 'Максимальный'""",
                                   timeout=30_000)
            seen = {}
            # Opening: the chosen mode, Max, is the strip's last word; the strip shows it whole.
            page.tap(f"{CONTEXT} .chat-capsule-head")
            _until(page, "r.ctx.open === 'true'")
            page.wait_for_timeout(450)
            seen["tap"] = page.evaluate(SHOWN, "max")
            _shot(page, tmp_path, f"capsules-translated-context-open-{engine}-320")
            assert seen["tap"]["fit"] == "overflow" and seen["tap"]["word"] == "Максимальный", seen
            assert seen["tap"]["whole"] and seen["tap"]["scroll"] == "end" and not seen["tap"]["docScroll"], seen
            page.keyboard.press("Escape")
            _until(page, "r.ctx.open === 'false' && r.ctx.focused === 'head'")
            # The keyboard: Enter opens onto Max; Home chooses Nano at the start, End Max at the end.
            page.keyboard.press("Enter")
            _until(page, "r.ctx.open === 'true' && r.ctx.focused === 'max'")
            page.wait_for_timeout(100)
            seen["Enter"] = page.evaluate(SHOWN, "max")
            assert seen["Enter"]["whole"] and seen["Enter"]["scroll"] == "end", seen
            for key, mode, edge in (("Home", "nano", "start"), ("End", "max", "end")):
                with page.expect_response(lambda r: r.url.endswith("/api/owner/context-mode"), timeout=30_000):
                    page.keyboard.press(key)
                _until(page, f"r.ctx.mode === '{mode}' && r.ctx.disabled === 'false' && r.ctx.focused === '{mode}'")
                page.wait_for_timeout(100)
                seen[key] = page.evaluate(SHOWN, mode)
                _shot(page, tmp_path, f"capsules-translated-context-{key.lower()}-{engine}-320")
                assert seen[key]["whole"] and seen[key]["scroll"] == edge and not seen[key]["docScroll"], seen
            _record(tmp_path, f"capsules-translated-context-{engine}-320", seen)
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_resize_that_makes_the_open_context_strip_scroll_shows_the_chosen_mode_once(direct_server_with_data, engine, tmp_path):  # noqa: F811
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    with sync_playwright() as pw:
        browser, page = _launch(pw, engine, viewport={"width": 600, "height": 860}, is_mobile=True, has_touch=True)
        try:
            page.route("**/api/ui/i18n", lambda route: route.fulfill(json=TRANSLATED))
            page.route("**/api/ui/i18n/missing", lambda route: route.fulfill(json={"ok": True}))
            _open_chat(page, url, width=600, height=860)
            page.wait_for_function(f"""() => document.querySelector('{CONTEXT} .chat-seg[data-mode="max"]')?.textContent === 'Максимальный'""",
                                   timeout=30_000)
            seen = {}
            # Open from the keyboard onto Max while the strip still fits; then the phone narrows to 320 px.
            page.focus(f"{CONTEXT} .chat-capsule-head")
            page.keyboard.press("Enter")
            _until(page, "r.ctx.open === 'true' && r.ctx.focused === 'max'")
            page.wait_for_timeout(100)
            seen["600"] = page.evaluate(SHOWN, "max")
            page.set_viewport_size({"width": 320, "height": 860})
            _until(page, "r.ctx.fit === 'overflow'")
            page.wait_for_timeout(200)
            seen["320"] = page.evaluate(SHOWN, "max")
            _shot(page, tmp_path, f"capsules-translated-context-resized-{engine}-320")
            # The owner swipes back to the first mode; a later relayout that keeps the strip
            # scrolling (the phone's toolbar takes some height) leaves it where the owner put it.
            page.evaluate(f"() => {{ document.querySelector('{CONTEXT} .chat-context-strip').scrollLeft = 0; }}")
            page.wait_for_function(f"() => document.querySelector('{CONTEXT} .chat-context-strip').dataset.scroll === 'start'")
            page.set_viewport_size({"width": 320, "height": 700})
            page.wait_for_function("() => innerHeight === 700")
            page.wait_for_timeout(200)
            seen["swiped"] = page.evaluate(SHOWN, "nano")
            _shot(page, tmp_path, f"capsules-translated-context-swiped-{engine}-320")
            _record(tmp_path, f"capsules-translated-context-resized-{engine}-320", seen)
            assert seen["600"]["fit"] != "overflow" and seen["600"]["whole"], seen
            assert seen["320"]["fit"] == "overflow" and seen["320"]["focused"] == "max", seen
            assert seen["320"]["whole"] and seen["320"]["scroll"] == "end" and not seen["320"]["docScroll"], seen
            assert seen["swiped"]["fit"] == "overflow" and seen["swiped"]["whole"] and seen["swiped"]["scroll"] == "start", seen
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_resize_reveals_the_focused_effort_bracket_without_changing_opening_target(direct_server_with_data, engine, tmp_path):  # noqa: F811
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser, page = _launch(pw, engine, viewport={"width": 390, "height": 860}, is_mobile=True, has_touch=True)
        try:
            _open_chat(page, direct_server_with_data["url"], width=390, height=860)
            page.tap(f"{EFFORT} .chat-capsule-head")
            _until(page, "r.eff.open === 'true' && r.eff.fit === 'solo'")
            page.focus(f"{EFFORT} .chat-effort-max")
            with page.expect_response(lambda r: r.url.endswith("/api/owner/effort-range")):
                page.keyboard.press("End")
            page.set_viewport_size({"width": 320, "height": 860})
            _until(page, "r.eff.fit === 'overflow'")
            page.wait_for_timeout(250)
            seen = page.evaluate(f"""() => {{
                const el = document.querySelector('{EFFORT}');
                const strip = el.querySelector('.chat-effort-strip');
                const cap = el.querySelector('.chat-effort-max');
                const s = strip.getBoundingClientRect(), c = cap.getBoundingClientRect();
                return {{ focused: document.activeElement === cap, left: c.left, right: c.right,
                    stripLeft: s.left, stripRight: s.right, scroll: strip.dataset.scroll }};
            }}""")
            assert seen["focused"] and seen["left"] >= seen["stripLeft"] and seen["right"] <= seen["stripRight"] + 0.5, seen
            _record(tmp_path, f"capsules-focused-bracket-resize-{engine}", seen)
            _shot(page, tmp_path, f"capsules-focused-bracket-resize-{engine}")
            page.keyboard.press("Escape")
            page.keyboard.press("Enter")
            page.wait_for_function(f"() => document.activeElement === document.querySelector('{EFFORT} .chat-effort-rec')")
        finally:
            browser.close()


def _install(page, payload) -> None:
    page.route("**/api/ui/i18n", lambda route: route.fulfill(json=payload))
    page.route("**/api/ui/i18n/missing", lambda route: route.fulfill(json={"ok": True}))


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_right_to_left_strips_that_scroll_keep_the_moved_mode_and_bracket_in_view(direct_server_with_data, engine, tmp_path):  # noqa: F811
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    with sync_playwright() as pw:
        browser, page = _launch(pw, engine, viewport={"width": 320, "height": 860}, is_mobile=True, has_touch=True)
        seen = {}
        try:
            _install(page, RTL)
            _open_chat(page, url, width=320, height=860)
            page.wait_for_function(f"""() => document.documentElement.dir === 'rtl'
                && document.querySelector('{CONTEXT} .chat-seg[data-mode="max"]')?.textContent === 'السياق الكامل'""", timeout=30_000)
            # Context: the chosen mode, Max, is the strip's last word, on its left; opening scrolls
            # leftwards (a negative scrollLeft) until it is whole, and the fade is on the right,
            # where Nano hides.
            page.tap(f"{CONTEXT} .chat-capsule-head")
            _until(page, "r.ctx.open === 'true'")
            page.wait_for_timeout(450)
            seen["tap"] = page.evaluate(SHOWN, "max")
            _shot(page, tmp_path, f"capsules-rtl-context-open-{engine}-320")
            assert seen["tap"]["fit"] == "overflow" and seen["tap"]["scrollLeft"] < 0, seen
            assert seen["tap"]["whole"] and seen["tap"]["scroll"] == "end" and "to left" in seen["tap"]["fade"], seen
            assert not seen["tap"]["docScroll"], seen
            page.keyboard.press("Escape")
            _until(page, "r.ctx.open === 'false' && r.ctx.focused === 'head'")
            page.keyboard.press("Enter")
            _until(page, "r.ctx.open === 'true' && r.ctx.focused === 'max'")
            page.wait_for_timeout(100)
            seen["Enter"] = page.evaluate(SHOWN, "max")
            assert seen["Enter"]["whole"] and seen["Enter"]["scroll"] == "end", seen
            row = page.locator('#page-chat .chat-toolbar-row')
            row_width = row.evaluate('node => node.getBoundingClientRect().width')
            # Fractional columns expose the browser's integer scroll coordinates even with the
            # same font. Keep the text whole at both ends; a mode left offscreen must fail it.
            for fraction in (0, 0.2, 0.4, 0.6, 0.8):
                row.evaluate('(node, width) => node.style.width = width + "px"', row_width - fraction)
                for key, mode, edge in (("Home", "nano", "start"), ("End", "max", "end")):
                    with page.expect_response(lambda r: r.url.endswith("/api/owner/context-mode"), timeout=30_000):
                        page.keyboard.press(key)
                    _until(page, f"r.ctx.mode === '{mode}' && r.ctx.disabled === 'false' && r.ctx.focused === '{mode}'")
                    page.wait_for_timeout(100)
                    sample = f"{fraction}-{key}"
                    seen[sample] = page.evaluate(SHOWN, mode)
                    _shot(page, tmp_path, f"capsules-rtl-context-{sample.lower()}-{engine}-320")
                    assert seen[sample]["whole"] and seen[sample]["scroll"] == edge and not seen[sample]["docScroll"], seen
                    if key == "Home":
                        assert not page.evaluate(SHOWN, "max")["whole"], "the offscreen word must not count as whole"
            row.evaluate("node => node.style.removeProperty('width')")
            # The owner swipes back to the start (scrollLeft 0, Nano on the right); a relayout that
            # keeps the strip scrolling leaves it there.
            page.evaluate(f"() => {{ document.querySelector('{CONTEXT} .chat-context-strip').scrollLeft = 0; }}")
            page.wait_for_function(f"() => document.querySelector('{CONTEXT} .chat-context-strip').dataset.scroll === 'start'")
            page.set_viewport_size({"width": 320, "height": 700})
            page.wait_for_function("() => innerHeight === 700")
            page.wait_for_timeout(200)
            seen["swiped"] = page.evaluate(SHOWN, "nano")
            assert seen["swiped"]["whole"] and seen["swiped"]["scroll"] == "start" and "to left" in seen["swiped"]["fade"], seen
            page.keyboard.press("Escape")
            _until(page, "r.ctx.open === 'false'")
            # Effort: the scale runs leftwards from the head. The pill covers its level and the
            # brackets face the range from either side, the min bracket on the right.
            page.tap(f"{EFFORT} .chat-capsule-head")
            _until(page, "r.eff.open === 'true'")
            page.wait_for_timeout(450)
            seen["effort"] = opened = page.evaluate(MARKS)
            _shot(page, tmp_path, f"capsules-rtl-effort-open-{engine}-320")
            assert opened["fit"] == "overflow" and opened["dir"] == "rtl" and opened["pillWhole"], seen
            assert abs(opened["pill"][0] - opened["seg"][0]) <= 0.6 and abs(opened["pill"][1] - opened["seg"][1]) <= 0.6, seen
            assert abs(opened["band"][1] - opened["min"][1]) <= 0.6 and abs(opened["band"][0] - opened["max"][0]) <= 0.6, seen
            assert opened["max"][1] < opened["pill"][0] < opened["pill"][1] < opened["min"][0], seen
            # End carries the max bracket to Ultra, the last level on the far left: whole, clear of the
            # rounded end by at least its vertical inset, the fade on the right.
            page.focus(f"{EFFORT} .chat-effort-max")
            with page.expect_response(lambda r: r.url.endswith("/api/owner/effort-range"), timeout=30_000):
                page.keyboard.press("End")
            page.wait_for_timeout(300)
            seen["effort-End"] = at_end = page.evaluate(MARKS)
            _shot(page, tmp_path, f"capsules-rtl-effort-ultra-{engine}-320")
            assert at_end["maxWhole"] and at_end["ultraWhole"] and at_end["scroll"] == "end" and "to left" in at_end["fade"], seen
            assert at_end["max"][0] - at_end["shellLeft"] >= at_end["inset"] - 0.5, seen
            # A press lands on the level under it: back at the start, a press on None (outside the
            # helper range) sets primary there without moving the min bracket.
            page.evaluate(f"() => {{ document.querySelector('{EFFORT} .chat-effort-strip').scrollLeft = 0; }}")
            page.wait_for_function(f"() => document.querySelector('{EFFORT} .chat-effort-strip').dataset.scroll === 'start'")
            none = page.evaluate(MARKS)["none"]
            box = page.locator(f'{EFFORT} [data-effort-seg="none"]').bounding_box()
            with page.expect_response(lambda r: r.url.endswith("/api/owner/effort-range"), timeout=30_000) as saved:
                page.mouse.click(none[0] + (none[1] - none[0]) * 0.75, box["y"] + box["height"] / 2)
            assert json.loads(saved.value.request.post_data)["recommended"] == "none", saved.value.request.post_data
            assert json.loads(saved.value.request.post_data)["min"] == "low", saved.value.request.post_data
            page.wait_for_timeout(300)
            seen["effort-press"] = pressed = page.evaluate(MARKS)
            # The scale's outer edge is two whole-pixel offsets (as Ultra's is left to right): within a pixel.
            assert abs(pressed["pill"][1] - pressed["none"][1]) <= 1 and abs(pressed["min"][1] - pressed["band"][1]) <= 0.6, seen
            _shot(page, tmp_path, f"capsules-rtl-effort-press-{engine}-320")
        finally:
            _record(tmp_path, f"capsules-rtl-{engine}-320", seen)
            browser.close()


@pytest.mark.parametrize("engine,direction", [("chromium", "ltr"), ("webkit", "ltr"), ("chromium", "rtl"), ("webkit", "rtl")])
def test_long_closed_values_on_a_320px_phone_give_way_on_one_line(direct_server_with_data, engine, direction, tmp_path):  # noqa: F811
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    payload, context_word, effort_word = LONG_VALUES[direction]
    url = direct_server_with_data["url"]
    with sync_playwright() as pw:
        browser, page = _launch(pw, engine, viewport={"width": 320, "height": 860}, is_mobile=True, has_touch=True)
        seen = {}
        try:
            _install(page, payload)
            _open_chat(page, url, width=320, height=860)
            page.wait_for_function(f"""() => document.querySelector('{CONTEXT} .chat-capsule-value')?.textContent === '{context_word}'
                && document.querySelector('{EFFORT} .chat-capsule-value')?.textContent === '{effort_word}'""", timeout=30_000)
            # Closed: without the names the values still take two lines, so the longest give way to
            # one measured width; the row keeps one line and every control stays inside it.
            _until(page, "r.compact !== ''")  # the settled, closed row decided its fit
            page.wait_for_timeout(200)
            seen["closed"] = closed = _row(page)
            seen["capped"] = capped = page.evaluate(CAPPED)
            _shot(page, tmp_path, f"capsules-long-values-closed-{engine}-{direction}-320")
            assert closed["oneLine"] and closed["rowHeight"] < 40 and capped["contained"] and not closed["docScroll"], seen
            assert closed["compact"] == "values" and closed["namesShown"] == [False, False, False], seen
            assert closed["subShown"] and closed["ctx"]["shown"] and closed["eff"]["shown"], seen
            subagents, context, effort = capped["capsules"]
            assert context["word"] == context_word and context["cut"], seen
            for capsule in (subagents, context, effort):  # a value, cut or whole, keeps clear of both ends of its shell
                assert capsule["value"][0] >= capsule["shell"][0] + 1 and capsule["value"][1] <= capsule["shell"][1] - 1, seen
            # The whole word stays the button's: its accessible name and the open strip (no hover-only title).
            assert context_word in page.locator(f"{CONTEXT} .chat-capsule-head").aria_snapshot(), seen
            assert effort_word in effort["label"] and not effort["title"], seen
            page.tap(f"{CONTEXT} .chat-capsule-head")
            _until(page, "r.ctx.open === 'true'")
            page.wait_for_timeout(450)
            seen["open"] = page.evaluate(SHOWN, "max")
            assert seen["open"]["word"] == context_word and seen["open"]["whole"] and not seen["open"]["docScroll"], seen
            page.keyboard.press("Escape")
            _until(page, "r.ctx.open === 'false' && r.compact === 'values' && r.oneLine")
            # A wider line takes the cut back: the values show whole again.
            page.set_viewport_size({"width": 600, "height": 860})
            _until(page, "r.compact !== 'values' && r.oneLine")
            seen["wide"] = page.evaluate(CAPPED)
            assert not any(capsule["cut"] for capsule in seen["wide"]["capsules"]), seen
        finally:
            _record(tmp_path, f"capsules-long-values-{engine}-{direction}-320", seen)
            browser.close()


@pytest.mark.parametrize("engine,direction", [("chromium", "ltr"), ("webkit", "ltr"), ("chromium", "rtl"), ("webkit", "rtl")])
def test_long_values_open_and_close_without_a_wrapping_frame(direct_server_with_data, engine, direction, tmp_path):  # noqa: F811
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    payload, context_word, effort_word = LONG_VALUES[direction]
    url = direct_server_with_data["url"]
    with sync_playwright() as pw:
        browser, page = _launch(pw, engine, viewport={"width": 320, "height": 900})
        runs = {}
        try:
            _install(page, payload)
            _open_chat(page, url, width=320)
            page.wait_for_function(f"""() => document.querySelector('{CONTEXT} .chat-capsule-value')?.textContent === '{context_word}'
                && document.querySelector('{EFFORT} .chat-capsule-value')?.textContent === '{effort_word}'""", timeout=30_000)
            # From the narrowest phone, where the closed values are cut, through names that give way
            # and strips that stand alone, to strips that open beside the others: every frame.
            for width in (320, 360, 400, 480, 560, 640, 720, 800):
                page.set_viewport_size({"width": width, "height": 900})
                _until(page, f"innerWidth === {width} && r.ctx.open === 'false' && r.eff.open === 'false'")
                page.wait_for_timeout(250)  # the row refits on the resize (a wrapped row fails below, frame by frame)
                for name, selector in (("context", CONTEXT), ("effort", EFFORT)):
                    before = page.evaluate(CAPPED)
                    run = page.evaluate(FRAMES, {"sel": selector, "frames": 40})
                    page.wait_for_timeout(500)
                    run["settled"] = page.evaluate(SETTLED, selector)
                    run["values"] = [[capsule["cut"] for capsule in before["capsules"]],
                                     [capsule["cut"] for capsule in page.evaluate(CAPPED)["capsules"]]]
                    runs[f"{name}-{width}"] = run
        finally:
            _record(tmp_path, f"capsules-long-values-frames-{engine}-{direction}", runs)
            browser.close()
    wrapped = {key: [round(frame["height"]) for frame in [run["closed"]] + run["opening"] + run["closing"] if frame["spread"] >= 2]
               for key, run in runs.items()}
    assert not any(wrapped.values()), {key: heights for key, heights in wrapped.items() if heights}
    # The narrowest phone is the case under test: there the closed values are cut.
    assert runs["context-320"]["opening"][0]["compact"] == "values" and any(runs["context-320"]["values"][0]), runs["context-320"]
    for key, run in runs.items():
        ends = (run["closed"]["width"], run["opening"][-1]["width"])
        peak = max(frame["width"] for frame in run["opening"] + run["closing"])
        assert peak <= max(ends) + 1, (key, peak, ends)
        # Open, the chosen word stands whole in the strip; closed again, the capsule and its cut
        # are what they were, nothing held.
        assert run["opening"][-1]["whole"], (key, run["opening"][-1])
        assert run["open"] == "false" and run["settled"]["held"] == "", (key, run["settled"])
        assert abs(run["settled"]["width"] - run["closed"]["width"]) <= 0.5 and run["values"][0] == run["values"][1], (key, run)
        if run["fit"] == "inline":  # it still animates: frames between the two widths, both ways
            assert _between(run["opening"], ends) >= 3 and _between(run["closing"], ends) >= 3, key


# The phone's alone strip, frame by frame (owner 19.10, 2A; capsule_morph.js): every animation of the press, paused
# and read at fixed times — the head's place, the line's height, the controls that step aside — then finished.
MORPH_FRAMES = """async ({ sel, times }) => {
    const row = document.querySelector('#page-chat .chat-toolbar-row');
    const el = document.querySelector(sel);
    const head = el.querySelector('.chat-capsule-head');
    const rect = (node) => { const r = node.getBoundingClientRect(); return { left: r.left, right: r.right, top: r.top, width: r.width }; };
    const shownKids = () => [...row.children].filter((kid) => kid !== el && getComputedStyle(kid).display !== 'none');
    const before = { capsule: rect(el), head: rect(head), kids: shownKids().map(rect), height: row.getBoundingClientRect().height };
    head.dispatchEvent(new MouseEvent('click', { bubbles: true, detail: 1 }));
    const animations = [el, ...row.children].flatMap((node) => node.getAnimations({ subtree: true }))
        .filter((animation, i, all) => all.indexOf(animation) === i);
    animations.forEach((animation) => animation.pause());
    const frames = [];
    for (const t of times) {
        animations.forEach((animation) => { animation.currentTime = t; });
        await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
        const leaving = [...row.querySelectorAll(':scope > [data-capsule-leaving]')];
        frames.push({ t, height: row.getBoundingClientRect().height, head: rect(head).left, morph: el.dataset.morph || '',
            surface: Boolean(el.querySelector(':scope > .chat-capsule-surface')),
            leaving: leaving.map((kid) => ({ left: rect(kid).left, opacity: Number(getComputedStyle(kid).opacity) })),
            back: shownKids().map((kid) => Number(getComputedStyle(kid).opacity)) });
    }
    animations.forEach((animation) => { try { animation.finish(); } catch (e) { animation.cancel(); } });
    await new Promise((resolve) => setTimeout(resolve, 60));
    const after = { morph: el.dataset.morph || '', surfaces: el.querySelectorAll('.chat-capsule-surface').length,
        leaving: row.querySelectorAll('[data-capsule-leaving]').length, ownerLeaving: row.hasAttribute('data-owner-leaving'),
        fit: el.dataset.fit || '', open: el.dataset.open, solo: row.dataset.ownerSolo || '', head: rect(head).left,
        kids: shownKids().length, height: row.getBoundingClientRect().height };
    return { before, count: animations.length, frames, after };
}"""


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_strip_that_opens_alone_grows_from_its_capsule_and_folds_back(direct_server_with_data, engine, tmp_path):  # noqa: F811
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    runs = {}
    with sync_playwright() as pw:
        browser, page = _launch(pw, engine, viewport={"width": 390, "height": 860}, is_mobile=True, has_touch=True)
        try:
            _open_chat(page, url, width=390, height=860, theme="light")
            page.wait_for_timeout(300)
            for name, selector in (("effort", EFFORT), ("context", CONTEXT)):
                runs[f"{name}-open"] = page.evaluate(MORPH_FRAMES, {"sel": selector, "times": [0, 60, 150, 300]})
                _shot(page, tmp_path, f"capsules-morph-{name}-open-{engine}")
                runs[f"{name}-close"] = page.evaluate(MORPH_FRAMES, {"sel": selector, "times": [0, 80, 220]})
            page.emulate_media(reduced_motion="reduce")
            runs["reduced"] = page.evaluate(MORPH_FRAMES, {"sel": EFFORT, "times": [0]})
            page.keyboard.press("Escape")
        finally:
            _record(tmp_path, f"capsules-morph-{engine}", runs)
            browser.close()
    for name in ("effort", "context"):
        opening, closing = runs[f"{name}-open"], runs[f"{name}-close"]
        assert opening["after"]["fit"] in ("solo", "overflow") and opening["after"]["solo"] == name, (name, opening["after"])
        first, last = opening["frames"][0], opening["frames"][-1]
        # The glass is its own layer while it moves; the head starts where the closed capsule's head stood.
        assert first["morph"] == "open" and first["surface"], (name, first)
        assert abs(first["head"] - opening["before"]["head"]["left"]) <= 1, (name, first, opening["before"])
        assert abs(last["head"] - opening["after"]["head"]) <= 1, (name, last, opening["after"])
        # The controls that step aside stand where they stood and fade; the line never takes a second line.
        assert len(first["leaving"]) == len(opening["before"]["kids"]) >= 2, (name, first)
        assert all(abs(kid["left"] - box["left"]) <= 1 for kid, box in zip(first["leaving"], opening["before"]["kids"])), (name, first)
        assert all(kid["opacity"] > 0.9 for kid in first["leaving"]), (name, first)
        assert all(kid["opacity"] < 0.05 for kid in opening["frames"][2]["leaving"]), (name, opening["frames"][2])
        assert all(abs(frame["height"] - opening["before"]["height"]) <= 1 for frame in opening["frames"]), (name, opening["frames"])
        # Nothing it added outlives it.
        assert opening["after"]["surfaces"] == 0 and opening["after"]["leaving"] == 0 and not opening["after"]["ownerLeaving"], name
        assert opening["after"]["morph"] == "" and opening["after"]["kids"] == 0, (name, opening["after"])
        # Closing runs back: the head rides home, the controls fade in, and the line is whole again.
        c0 = closing["frames"][0]
        assert c0["morph"] == "close" and c0["surface"] and abs(c0["head"] - closing["before"]["head"]["left"]) <= 1, (name, c0)
        assert c0["back"] and all(opacity < 0.1 for opacity in c0["back"]), (name, c0)
        assert closing["after"]["open"] == "false" and closing["after"]["solo"] == "", (name, closing["after"])
        assert closing["after"]["kids"] == len(opening["before"]["kids"]) and closing["after"]["surfaces"] == 0, (name, closing["after"])
        assert abs(closing["after"]["head"] - opening["before"]["head"]["left"]) <= 1, (name, closing["after"], opening["before"])
        assert all(abs(frame["height"] - opening["before"]["height"]) <= 1 for frame in closing["frames"]), (name, closing["frames"])
    reduced = runs["reduced"]
    assert reduced["frames"][0]["morph"] == "" and not reduced["frames"][0]["surface"], ("reduced motion: at once", reduced)
