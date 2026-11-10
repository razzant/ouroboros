"""The design skills' browser-side files in real engines: the comparison page, measure.js, freeze.js.

Each behaviour is checked in both directions where it can be: the page swaps variants in place and
keeps the compared element still; measure.js reports each planted defect on `defects.html` and stays
quiet on `clean.html`; a frozen page renders alone from an empty folder with nothing fetched, and keeps
a var() shorthand CSSOM cannot serialize by copying its sheet's text, yet names it in `missing` whether
copied or lost, since equal serialized rules do not prove that text still holds the live value.
WebKit is the engine of the desktop app's window, Chromium the bundled one; a missing engine skips
unless OUROBOROS_EXPECT_BROWSER_ENGINES is set.
"""
from __future__ import annotations

import io
import os
import threading
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tests.test_design_skill_assets import FREEZE, MEASURE, compare_data, fill_compare

pytestmark = [pytest.mark.browser]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "design_skills"
ENGINES = ["webkit", "chromium"]
LONG_NOTE = ("Two lines instead of four: who asks and from which project, then the state. The answer field takes "
             "the composer's shape, Send sits inside it, and the service line moves down beside the time, so the "
             "question reads first and the controls stay where the eye expects them after a long answer.")


@contextmanager
def browser(engine):
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import Error, sync_playwright

    from ouroboros.tools.browser import _set_playwright_browsers_path_if_bundled

    _set_playwright_browsers_path_if_bundled()
    with sync_playwright() as pw:
        try:
            instance = getattr(pw, engine).launch(headless=True)
        except Error as exc:
            if os.environ.get("OUROBOROS_EXPECT_BROWSER_ENGINES"):
                pytest.fail(f"{engine}: {exc}")
            pytest.skip(f"{engine} is not installed: {str(exc).splitlines()[0]}")
        try:
            yield instance
        finally:
            instance.close()


def open_page(instance, url, width=900, height=800):
    """A page that records every request it makes; the caller checks that nothing left the disk."""
    page = instance.new_page(viewport={"width": width, "height": height})
    page.requests = []
    page.on("request", lambda request: page.requests.append(request.url))
    page.goto(url)
    return page


def assert_nothing_fetched(page):
    assert all(url.startswith(("file:", "about:", "data:")) for url in page.requests), page.requests


def rgba(locator):
    """An element's pixels as exact RGBA bytes with their size: any changed pixel differs."""
    from PIL import Image

    image = Image.open(io.BytesIO(locator.screenshot())).convert("RGBA")
    return image.size, image.tobytes()


def _frame(variant: str, scene: str, spacer: int) -> str:
    return f"""<!doctype html><html><head><style>
 body {{ margin: 0; font: 16px/1.5 system-ui, sans-serif; background: #fff; color: #1c2230; }}
 #list {{ height: 520px; overflow: auto; }}
 .subject {{ height: 150px; margin: 0 24px; padding: 12px; background: #e8edf7; }}
</style></head><body><div id="list" data-freeze-scroll="200,0"><div style="height:{spacer}px">{scene} above</div>
<div class="subject">{variant} in {scene}</div><div style="height:1400px">below</div></div></body></html>"""


def sample_page(tmp_path: Path, ids=("now", "a", "b")) -> Path:
    """A two-scene, three-variant comparison page copied alone into an empty folder."""
    variants = list(zip(ids, ("Now", "A", "B"), ("Today's screen.", LONG_NOTE, "A tray with the controls.")))
    scenes = [("draft", "Draft"), ("long", "Long question")]
    spacers = dict(zip(ids, (300, 380, 450)))
    data = {
        "title": "Which header reads more calmly?", "lang": "en", "viewport": {"width": 600, "height": 560},
        "variants": [{"id": v, "label": label, "note": note, "recommended": v == ids[1]} for v, label, note in variants],
        "scenes": [{"id": s, "label": label, "note": f"{label} scene"} for s, label in scenes],
        "frames": {f"{v}/{s}": _frame(v, s, spacers[v] + (40 if s == "long" else 0)) for v, *_ in variants for s, _l in scenes},
        "subject": ".subject", "scrollers": ["#list"],
        "questions": [{"text": "Which header?", "context": "", "options": [{"key": "A", "text": "Two lines", "recommended": True},
                                                                         {"key": "B", "text": "A tray"}]},
                      {"text": "Keep the Clear button?", "options": [{"key": "A", "text": "Remove it", "recommended": True}]}],
        "source": "Fixture frames.",
    }
    folder = tmp_path / "-".join(("alone", *ids))
    folder.mkdir()
    page = folder / "compare.html"
    page.write_text(fill_compare(data), encoding="utf-8")
    assert compare_data(page.read_text(encoding="utf-8"))["subject"] == ".subject"
    return page


def wait_shown(page, variant, scene, side=False):
    page.wait_for_function("""([v, s, side]) => {
        const shown = [...document.querySelectorAll('#stage iframe:not([data-off])')].map(f => f.dataset.key);
        const pressed = document.querySelector(`[data-variant="${v}"]`).getAttribute('aria-pressed');
        return side ? shown.length > 1 && shown.every(k => k.endsWith('/' + s))
                    : shown.length === 1 && shown[0] === `${v}/${s}` && pressed === 'true';
    }""", arg=[variant, scene, side])
    page.wait_for_timeout(150)


def subject_top(page) -> float:
    return page.evaluate("""() => {
        const f = document.querySelector('#stage iframe:not([data-off])');
        return f.getBoundingClientRect().top + f.contentDocument.querySelector('.subject').getBoundingClientRect().top;
    }""")


@pytest.mark.parametrize("engine", ENGINES)
def test_compare_page_swaps_in_place_with_the_subject_anchored(engine, tmp_path):
    path = sample_page(tmp_path)
    with browser(engine) as instance:
        page = open_page(instance, path.as_uri())
        wait_shown(page, "now", "draft")
        page.screenshot()
        before, stage_top = subject_top(page), page.evaluate("document.getElementById('stage').getBoundingClientRect().top")
        for key, variant in (("2", "a"), ("ArrowRight", "b"), ("ArrowLeft", "a"), ("1", "now")):
            page.keyboard.press(key)
            wait_shown(page, variant, "draft")
            assert abs(subject_top(page) - before) <= 2, (key, variant)
            assert page.evaluate("document.getElementById('stage').getBoundingClientRect().top") == pytest.approx(stage_top, abs=0.5)
        note = page.locator('[data-note="a"]')
        assert note.evaluate("p => getComputedStyle(p).textOverflow") != "ellipsis"
        assert note.evaluate("p => p.getBoundingClientRect().height > 1.6 * parseFloat(getComputedStyle(p).lineHeight)"), "a long caption wraps"
        assert page.evaluate("document.getElementById('scale').textContent") == "", "900 px shows a 600 px frame at 1:1"

        page.keyboard.press("s")
        wait_shown(page, "now", "draft", side=True)
        bands = page.evaluate("""() => [...document.querySelectorAll('#stage iframe:not([data-off])')].map(f => ({
            clip: f.style.clipPath, transform: f.style.transform }))""")
        assert len(bands) == 3 and all(band["clip"].startswith("inset(") and "scale(1)" in band["transform"] for band in bands)
        assert page.locator("#stage .cap").count() == 3
        page.keyboard.press("2")
        wait_shown(page, "a", "draft")

        page.locator('[data-scene="long"]').click()
        wait_shown(page, "a", "long")
        assert page.locator('[data-scene="long"]').get_attribute("aria-pressed") == "true"
        assert page.evaluate("location.hash") == "#scene=long&v=a"
        assert_nothing_fetched(page)


@pytest.mark.parametrize("engine", ENGINES)
def test_compare_page_questions_offer_none_and_compose_the_answer(engine, tmp_path):
    path = sample_page(tmp_path)
    with browser(engine) as instance:
        page = open_page(instance, path.as_uri())
        wait_shown(page, "now", "draft")
        first, second = page.locator(".q").nth(0), page.locator(".q").nth(1)
        assert first.locator(".opt").count() == 3 and "None of these" in first.locator(".opt").last.inner_text()
        assert first.locator(".opt.recommended").count() == 1
        first.locator(".opt").first.click()
        second.locator(".opt").last.click()
        assert "1A 2 none" in page.locator(".answer").inner_text()
        if engine == "chromium":
            page.keyboard.press("Tab")
            assert page.evaluate("getComputedStyle(document.activeElement).outlineStyle") != "none", "focus is visible"
        assert_nothing_fetched(page)


@pytest.mark.parametrize("engine", ENGINES)
def test_measure_reports_planted_defects_and_stays_quiet_on_the_clean_page(engine):
    script = MEASURE.read_text(encoding="utf-8")
    with browser(engine) as instance:
        reports = {}
        for name in ("defects", "clean"):
            page = open_page(instance, (FIXTURES / f"{name}.html").as_uri(), width=800, height=600)
            page.screenshot()                     # measure what is painted
            reports[name] = page.evaluate(script, {})
            assert_nothing_fetched(page)
        found = reports["defects"]["findings"]
        kinds = {item["kind"] for item in found}
        assert {"text-clipped-by-ancestor", "container-overflow", "pinned-over-content", "text-covered"} <= kinds, found
        clipped = [item for item in found if item["kind"] == "text-clipped-by-ancestor"]
        assert any(item["value"]["axis"] == "x" and "tab" in item["where"] for item in clipped), "the cut tab"
        assert any(item["value"]["axis"] == "y" and "#card" in item["where"] for item in clipped), "the cut paragraph"
        overflow = {item["value"]["axis"]: item for item in found if item["kind"] == "container-overflow"}
        assert overflow["x"]["value"]["mode"] == "auto" and "tab" in overflow["x"]["value"]["culprit"]
        assert overflow["y"]["value"]["mode"] == "hidden"
        assert any("#bar" in item["where"] and "#tail" in item["where"] for item in found if item["kind"] == "pinned-over-content")
        assert any("#label" in item["where"] and "#badge" in item["where"] for item in found if item["kind"] == "text-covered")
        for kind, where in (("text-clipped", "#self-cut"), ("text-truncated", "#ellipsis"), ("controls-touching", "button.first"),
                            ("target-small", "#icon"), ("text-tiny", "#tiny"), ("contrast-low", "#faint"), ("contrast-low", "#pale"),
                            ("contrast-unknown", "#washed"), ("image-broken", "#broken"), ("edges-near-miss", "#nudge"),
                            ("controls-overlap", "button.over"), ("page-overflow", "#wide")):
            assert any(item["kind"] == kind and where in item["where"] for item in found), (kind, found)
        assert any(item["kind"] == "font-fallback" and item["value"] == "Ghost Face" for item in found), found
        assert all(set(item) == {"kind", "message", "where", "value"} for item in found)
        assert reports["defects"]["facts"]["leftEdges"] and reports["defects"]["facts"]["counts"]
        assert reports["clean"]["findings"] == [], reports["clean"]["findings"]


@pytest.mark.parametrize("engine", ENGINES)
def test_measure_finds_the_subject_inside_the_comparison_frames(engine, tmp_path):
    path = sample_page(tmp_path)
    with browser(engine) as instance:
        page = open_page(instance, path.as_uri())
        wait_shown(page, "now", "draft")
        page.screenshot()
        facts = page.evaluate(MEASURE.read_text(encoding="utf-8"), {"subject": ".subject"})["facts"]
        assert facts["subject"] and facts["subject"]["h"] == pytest.approx(174, abs=1)   # 150 + padding
        assert abs(facts["subject"]["y"] - subject_top(page)) <= 1


@contextmanager
def served(folder: Path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(folder)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)


# pad.css, the page's <style> and an authored inline style each set a var() shorthand beside one of its longhands (#2576).
PADDINGS = "['#pad', '#chip', '#inline'].map((selector) => getComputedStyle(document.querySelector(selector)).padding)"
LIVE_PADDINGS = ["6px 13px 6px 2px", "3px 1px 3px 11px", "5px 5px 5px 1px"]
LOST = "(a var() shorthand CSSOM cannot serialize)"
COPIED = "(a var() shorthand CSSOM cannot serialize, copied from its sheet's text: compare it with the live page)"
CHIP = "padding-top, padding-bottom, padding-left of .chip in inline CSS"   # padding-right overrides the shorthand


@pytest.mark.parametrize("engine", ENGINES)
def test_freeze_output_renders_alone_with_state_kept_and_scripts_gone(engine, tmp_path):
    with served(FIXTURES / "freeze_site") as base, browser(engine) as instance:
        live = open_page(instance, base + "index.html", width=800, height=600)
        live.wait_for_selector("#made")
        live.fill("#name", "Ada")
        live.evaluate("document.getElementById('list').scrollTop = 120")
        live.screenshot()
        box = live.locator("#made").bounding_box()
        assert live.evaluate(PADDINGS) == LIVE_PADDINGS
        padded = rgba(live.locator("#pad"))
        frozen = live.evaluate(FREEZE.read_text(encoding="utf-8"))
        assert set(frozen["missing"]) == {f"padding-top, padding-right, padding-bottom of .pad in {base}pad.css {COPIED}",
                                          f"{CHIP} {COPIED}"}, "copied, never called current; the inline style is exact"
        assert live.evaluate("document.querySelectorAll('script').length") == 1, "the live page is untouched"
        assert live.evaluate("document.getElementById('list').scrollTop") == 120

        folder = tmp_path / "frozen"
        folder.mkdir()
        (folder / "frame.html").write_text(frozen["html"], encoding="utf-8")
        page = open_page(instance, (folder / "frame.html").as_uri(), width=800, height=600)
        assert page.evaluate("document.querySelectorAll('script').length") == 0
        assert page.locator("#made").inner_text() == "Rendered by a script"
        assert page.locator("#made").evaluate("e => getComputedStyle(e).color") == "rgb(43, 93, 52)"
        assert page.locator("#name").input_value() == "Ada"
        assert page.locator("#go").get_attribute("onclick") is None
        assert page.locator("#list").get_attribute("data-freeze-scroll") == "120,0"
        assert page.locator("#pic").get_attribute("src").startswith("data:image/svg+xml")
        assert page.evaluate("document.getElementById('pic').naturalWidth") == 40
        hero = page.locator("#hero").evaluate("e => getComputedStyle(e).backgroundImage")
        mark = page.locator("#mark").evaluate("e => getComputedStyle(e).backgroundImage")
        assert "data:image/svg+xml" in hero and "data:image/svg+xml" in mark
        assert mark != hero and "marks/" not in frozen["html"], "dot.svg is never replaced inside marks/dot.svg"
        frozen_box = page.locator("#made").bounding_box()
        assert abs(frozen_box["x"] - box["x"]) <= 1 and abs(frozen_box["y"] - box["y"]) <= 1
        assert page.evaluate(PADDINGS) == LIVE_PADDINGS, "a var() shorthand beside its longhand survives"
        assert "data:image/svg+xml" in page.locator("#pad").evaluate("e => getComputedStyle(e).backgroundImage")
        assert "pic.svg" not in frozen["html"] and "nowhere.svg" not in frozen["html"]
        assert rgba(page.locator("#pad")) == padded, "the frozen element is pixel-identical to the live one"
        assert_nothing_fetched(page)


def reopen_frame(instance, frozen, folder: Path):
    """The frame opened the way a frame is delivered: alone, from an empty folder."""
    folder.mkdir()
    (folder / "frame.html").write_text(frozen["html"], encoding="utf-8")
    return open_page(instance, (folder / "frame.html").as_uri(), width=800, height=600)


@pytest.mark.parametrize("engine", ENGINES)
def test_freeze_names_each_var_shorthand_a_changed_or_textless_source_loses(engine, tmp_path):
    """After insertRule the linked sheet's own text is stale; a constructed sheet and a script-written inline style
    have none. Rules stay (the inserted one too) and each emptied declaration is named as lost; the unchanged <style>
    is copied and named as such."""
    with served(FIXTURES / "freeze_site") as base, browser(engine) as instance:
        live = open_page(instance, base + "index.html", width=800, height=600)
        live.wait_for_selector("#made")
        live.evaluate("""() => {
            const pad = [...document.styleSheets].find((sheet) => sheet.href?.endsWith('/pad.css'));
            pad.insertRule('#made { color: rgb(1, 2, 3); }', pad.cssRules.length);
            const adopted = new CSSStyleSheet();
            adopted.replaceSync('.chip { --chip-none: ; margin: var(--chip-gap, 4px); margin-left: 0px; }');
            document.adoptedStyleSheets = [...document.adoptedStyleSheets, adopted];
            const mark = document.getElementById('mark');
            mark.style.padding = 'var(--chip-gap, 4px)';
            mark.style.paddingLeft = '1px';
        }""")
        live.screenshot()
        shape = "[...document.styleSheets, ...document.adoptedStyleSheets].map((sheet) => sheet.cssRules.length)"
        before = live.evaluate(shape)
        frozen = live.evaluate(FREEZE.read_text(encoding="utf-8"))
        assert live.evaluate(shape) == before, "the live sheets are untouched"
        assert set(frozen["missing"]) == {f"margin-top, margin-right, margin-bottom of .chip in inline CSS {LOST}",
                                          f"padding-top, padding-right, padding-bottom of .pad in {base}pad.css {LOST}",
                                          f"padding-top, padding-right, padding-bottom of the inline style of div#mark {LOST}",
                                          f"{CHIP} {COPIED}"}

        page = reopen_frame(instance, frozen, tmp_path / "frozen")
        padding = "e => getComputedStyle(e).padding"
        assert page.locator("#made").evaluate("e => getComputedStyle(e).color") == "rgb(1, 2, 3)", "the CSSOM edit stays"
        assert page.evaluate(PADDINGS)[1:] == LIVE_PADDINGS[1:], "the copied <style> and the authored inline style hold"
        assert page.locator("#pad").evaluate(padding) == "0px 0px 0px 2px", "named because really lost"
        assert page.locator("#mark").evaluate(padding) == "0px 0px 0px 1px", "named because really lost"
        assert "data:image/svg+xml" in page.locator("#pad").evaluate("e => getComputedStyle(e).backgroundImage")
        assert_nothing_fetched(page)


@pytest.mark.parametrize("engine", ENGINES)
def test_freeze_copies_a_sheets_own_text_only_when_the_frame_can_carry_it(engine, tmp_path):
    """An @import in the text (applied or since deleted), a URL only the rules spell as url() and a page in quirks
    mode keep the rules and name their losses; the sheet behind the @import is copied and nothing is fetched."""
    lossy = "padding: var(--x, 4px); padding-left: 1px;"
    site = tmp_path / "site"
    site.mkdir()
    for name, text in {
        "dot.svg": '<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8"><rect width="8" height="8"/></svg>',
        "child.css": f"@media (min-width: 1px) {{ .c {{ {lossy} }} }}",
        "parent.css": f"@import url(child.css);\n.p {{ {lossy} }}",
        "dropped.css": f"@import url(child.css);\n.d {{ {lossy} }}",
        "image.css": f'.i {{ {lossy} height: 8px; background-image: image-set("dot.svg" 1x); }}',
        "index.html": "<!doctype html>" + "".join(f'<link rel="stylesheet" href="{sheet}.css">' for sheet in ("parent", "dropped", "image"))
                      + "".join(f'<div class="{name}" id="{name}">{name}</div>' for name in "cpdi"),
        "quirks.html": f'<style>.q {{ {lossy} width: 10; }}</style><div class="q" id="q">q</div>',
    }.items():
        (site / name).write_text(text, encoding="utf-8")
    script = FREEZE.read_text(encoding="utf-8")
    padding, width = "e => getComputedStyle(e).padding", "e => getComputedStyle(e).width"
    lost = "padding-top, padding-right, padding-bottom of"
    with served(site) as base, browser(engine) as instance:
        live = open_page(instance, base + "index.html", width=800, height=600)
        live.screenshot()
        live.evaluate("[...document.styleSheets].find((sheet) => sheet.href.endsWith('/dropped.css')).deleteRule(0)")
        frozen = live.evaluate(script)
        assert set(frozen["missing"]) == {f"{lost} .p in {base}parent.css {LOST}", f"{lost} .d in {base}dropped.css {LOST}",
                                          f"{lost} .i in {base}image.css {LOST}", f"{lost} .c in {base}child.css {COPIED}"}
        assert "@import" not in frozen["html"]
        page = reopen_frame(instance, frozen, tmp_path / "frozen")
        assert page.locator("#c").evaluate(padding) == "4px 4px 4px 1px", "the imported sheet's own text"
        assert page.locator("#p").evaluate(padding) == "0px 0px 0px 1px", "named because really lost"
        assert "data:image/svg+xml" in page.locator("#i").evaluate("e => getComputedStyle(e).backgroundImage")
        assert_nothing_fetched(page)

        live = open_page(instance, base + "quirks.html", width=800, height=600)
        assert live.evaluate("document.compatMode") == "BackCompat"
        assert live.locator("#q").evaluate(width) == "10px"
        frozen = live.evaluate(script)
        assert frozen["missing"] == [f"{lost} .q in inline CSS {LOST}"]
        page = reopen_frame(instance, frozen, tmp_path / "frozen-quirks")
        assert page.evaluate("document.compatMode") == "CSS1Compat"
        assert page.locator("#q").evaluate(width) == "10px", "what quirks mode read stays in a standards frame"
        assert_nothing_fetched(page)


@pytest.mark.parametrize("engine", ENGINES)
def test_freeze_names_a_copied_var_shorthand_whether_or_not_its_text_is_current(engine, tmp_path):
    """Serialized rules omit a var() shorthand's tokens beside its longhand, so they cannot confirm the copied text:
    after a CSSOM edit swapping only those tokens (in a <style> and in a linked sheet) and a linked file rebuilt in the
    same shape, `missing` names the same declarations as before, and the frame keeps the text it could read. A var()
    shorthand CSSOM writes back whole stays exact and unnamed; a page with only such rules freezes clean."""
    lossy = "padding: var(--first); padding-left: 1px;"
    whole = ".k { padding: var(--first); } .k2 { padding: var(--first) !important; padding-left: 1px; }"

    def divs(*names):
        return "".join(f'<div class="{name}" id="{name}">{name}</div>' for name in names)

    site = tmp_path / "site"
    site.mkdir()
    for name, text in {
        "linked.css": f".l {{ {lossy} }}",
        "rebuilt.css": f".r {{ {lossy} }}",
        "index.html": '<!doctype html><link rel="stylesheet" href="linked.css"><link rel="stylesheet" href="rebuilt.css">'
                      f'<style>:root {{ --first: 4px; --second: 19px; }} .s {{ {lossy} }} {whole}</style>'
                      f'<main id="m">{divs("s", "l", "r", "k", "k2")}</main>',
        "whole.html": f"<!doctype html><style>:root {{ --first: 4px; }} {whole}</style>{divs('k', 'k2')}",
    }.items():
        (site / name).write_text(text, encoding="utf-8")
    script = FREEZE.read_text(encoding="utf-8")
    paddings = "[...document.querySelectorAll('div')].map((e) => getComputedStyle(e).padding)"
    loaded, named = ["4px 4px 4px 1px"] * 3 + ["4px"] * 2, "padding-top, padding-right, padding-bottom of"
    with served(site) as base, browser(engine) as instance:
        live = open_page(instance, base + "index.html", width=800, height=600)
        live.screenshot()
        assert live.evaluate(paddings) == loaded
        unchanged = live.evaluate(script)
        assert set(unchanged["missing"]) == {f"{named} .s in inline CSS {COPIED}", f"{named} .l in {base}linked.css {COPIED}",
                                             f"{named} .r in {base}rebuilt.css {COPIED}"}
        page = reopen_frame(instance, unchanged, tmp_path / "unchanged")
        assert page.evaluate(paddings) == loaded, "nothing changed, so the copied text holds the live value"
        assert rgba(page.locator("#m")) == rgba(live.locator("#m"))
        assert_nothing_fetched(page)

        live.evaluate("""() => {
            const rule = (selector) => [...document.styleSheets].flatMap((sheet) => [...sheet.cssRules])
                .find((each) => each.selectorText === selector);
            rule('.s').style.cssText = 'padding: var(--second); padding-left: 1px;';
            rule('.l').style.setProperty('padding', 'var(--second)');
            rule('.l').style.setProperty('padding-left', '1px');
        }""")
        (site / "rebuilt.css").write_text(f".r {{ {lossy.replace('--first', '--second')} }}", encoding="utf-8")
        assert live.evaluate(paddings) == ["19px 19px 19px 1px"] * 2 + loaded[2:]
        changed = live.evaluate(script)
        assert set(changed["missing"]) == set(unchanged["missing"]), "the page cannot tell the change, so freeze names both"
        page = reopen_frame(instance, changed, tmp_path / "changed")
        frame = page.evaluate(paddings)
        assert frame[:2] == loaded[:2], "the text's tokens, not the edited ones: why each is named"
        assert frame[2] in (loaded[2], "19px 19px 19px 1px"), "the loaded or the rebuilt file, named either way"
        assert frame[3:] == loaded[3:], "a var() shorthand CSSOM writes back whole stays exact"
        assert_nothing_fetched(page)

        live = open_page(instance, base + "whole.html", width=800, height=600)
        clean = live.evaluate(script)
        assert clean["missing"] == [], "nothing CSSOM cannot serialize: a clean capture"
        page = reopen_frame(instance, clean, tmp_path / "whole")
        assert page.evaluate(paddings) == ["4px", "4px"]
        assert_nothing_fetched(page)


@pytest.mark.parametrize("engine", ENGINES)
def test_freeze_keeps_stylesheet_text_out_of_html(engine, tmp_path):
    """CSS text is not HTML: raw fallback tails and literal CSS strings must not become document nodes.
    Exercise copied linked CSS, dynamic/adopted sheets, and cloned styles without sheets, including in a template;
    the actual CSS strings remain visible as content."""
    site = tmp_path / "site"
    site.mkdir()
    literal = "</StYle><b id=injected-string>literal</b>"
    adopted = "</STYLE><b id=injected-adopted>adopted</b>"
    dynamic = "</style><b id=injected-dynamic>dynamic</b>"
    css = ('.target { padding: var(--pad, 6px); padding-left: 1px; } '
           f'.literal::before {{ content: "{literal}"; }}\n'
           '</style><section id=injected-tail>inert</section>'
           '<script type="application/json" id="injected-script">{}</script><style>')
    (site / "linked.css").write_text(css, encoding="utf-8")
    (site / "index.html").write_text(
        '<!doctype html><link rel="stylesheet" href="linked.css">'
        '<div class="target">Target</div><div class="literal"></div>'
        '<div class="adopted"></div><div class="dynamic"></div>', encoding="utf-8")
    content = "[...document.querySelectorAll('.literal, .adopted, .dynamic')].map(e => getComputedStyle(e, '::before').content)"
    with served(site) as base, browser(engine) as instance:
        live = open_page(instance, base + "index.html", width=800, height=600)
        live.evaluate("""([adopted, dynamic]) => {
            const sheet = new CSSStyleSheet();
            sheet.replaceSync('.adopted::before { content: ' + JSON.stringify(adopted) + '; }');
            document.adoptedStyleSheets = [sheet];
            const style = document.createElement('style');
            style.textContent = '.dynamic::before { content: ' + JSON.stringify(dynamic) + '; }';
            document.head.append(style);
            const inactive = document.createElement('style');
            inactive.type = 'text/plain';
            inactive.textContent = '</style><b id="injected-inactive">inert</b><style type="text/plain">';
            document.head.append(inactive);
            const template = document.createElement('template');
            template.id = 'kept-template';
            const nested = document.createElement('style');
            nested.textContent = '</style></template><b id="injected-template">inert</b><template><style>';
            template.content.append(nested);
            document.body.append(template);
            if (inactive.sheet || nested.sheet) throw new Error('the control must have no stylesheet');
        }""", [adopted, dynamic])
        live.screenshot()
        expected = live.evaluate(content)
        assert all(value in expected[index] for index, value in enumerate((literal, adopted, dynamic)))
        assert live.locator('[id^="injected-"]').count() == 0
        assert live.locator('.target').evaluate("e => getComputedStyle(e).padding") == "6px 6px 6px 1px"
        frozen = live.evaluate(FREEZE.read_text(encoding="utf-8"))
        assert any('.target' in item and COPIED in item for item in frozen["missing"])
        page = reopen_frame(instance, frozen, tmp_path / "frozen")
        assert page.locator('[id^="injected-"]').count() == 0, "stylesheet text introduced HTML after script removal"
        assert page.locator('script').count() == 0
        assert page.locator('#kept-template').count() == 1
        assert page.evaluate(content) == expected, "escaping the HTML boundary must preserve CSS strings"
        assert page.locator('.target').evaluate("e => getComputedStyle(e).padding") == "6px 6px 6px 1px"
        assert_nothing_fetched(page)
        host = open_page(instance, "about:blank", width=800, height=600)
        host.evaluate("html => { const frame = document.createElement('iframe'); frame.srcdoc = html; document.body.append(frame); }", frozen["html"])
        frame = host.frame_locator('iframe')
        from playwright.sync_api import expect
        expect(frame.locator('.target')).to_have_text('Target')
        expect(frame.locator('[id^="injected-"]')).to_have_count(0)
        expect(frame.locator('script')).to_have_count(0)
        assert_nothing_fetched(host)


@pytest.mark.parametrize("engine", ENGINES)
def test_clean_view_hides_every_author_word_and_keeps_the_scene(engine, tmp_path):
    """C hides the title, names, notes, recommendation and questions; the frames, the switch and the scroll stay."""
    path = sample_page(tmp_path)
    with browser(engine) as instance:
        page = open_page(instance, path.as_uri())
        wait_shown(page, "now", "draft")
        authored = ["#notes", "#scene-note", "#more", '[data-variant="a"] .lab', '[data-variant="a"] .rec']
        visible = lambda selector: page.locator(selector).first.is_visible()
        assert all(visible(selector) for selector in authored), "the normal view shows them"
        title = page.evaluate("document.getElementById('title').textContent")
        page.keyboard.press("c")
        wait_shown(page, "now", "draft")
        assert not any(visible(selector) for selector in authored), "the clean view hides every author word"
        assert page.locator('[data-variant="a"] .num').inner_text() == "2"
        assert page.evaluate("document.getElementById('title').textContent") != title
        assert page.evaluate("location.hash") == "#scene=draft&v=1&clean=1", "the link names the variant by number"
        names = page.evaluate("[...document.querySelectorAll('#stage iframe')].map(f => f.title)")
        assert names and all(name.isdigit() for name in names), names
        page.keyboard.press("2")
        wait_shown(page, "a", "draft")
        assert page.evaluate("location.hash") == "#scene=draft&v=2&clean=1"
        page.keyboard.press("s")
        wait_shown(page, "a", "draft", side=True)
        captions = page.locator("#stage .cap").all_inner_texts()
        assert captions == ["1", "2", "3"], "side by side names the variants by number only"
        page.keyboard.press("c")
        page.wait_for_timeout(150)
        assert visible("#more") and page.evaluate("location.hash").find("clean") < 0, "C brings the author's words back"
        assert_nothing_fetched(page)
        shared = open_page(instance, path.as_uri() + "#scene=draft&v=2&clean=1")
        wait_shown(shared, "a", "draft")
        assert shared.evaluate("document.body.classList.contains('clean')"), "a clean link opens clean, on its variant"
        numeric = sample_page(tmp_path, ids=("now", "1", "2"))  # ids that read as numbers: the link's kind decides
        wait_shown(open_page(instance, numeric.as_uri() + "#scene=draft&v=2&clean=1"), "1", "draft")
        wait_shown(open_page(instance, numeric.as_uri() + "#scene=draft&v=2"), "2", "draft")
