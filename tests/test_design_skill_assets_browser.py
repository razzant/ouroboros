"""The design skills' browser-side files in real engines: the comparison page, measure.js, freeze.js.

Each behaviour is checked in both directions where it can be: the page swaps variants in place and
keeps the compared element still; measure.js reports each planted defect on `defects.html` and stays
quiet on `clean.html`; a frozen page renders alone from an empty folder with nothing fetched.
WebKit is the engine of the desktop app's window, Chromium the bundled one; a missing engine skips
unless OUROBOROS_EXPECT_BROWSER_ENGINES is set.
"""
from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tests.test_design_skill_assets import FREEZE, MEASURE, compare_data, fill_compare

pytestmark = [pytest.mark.browser, pytest.mark.serial]
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


@pytest.mark.parametrize("engine", ENGINES)
def test_freeze_output_renders_alone_with_state_kept_and_scripts_gone(engine, tmp_path):
    with served(FIXTURES / "freeze_site") as base, browser(engine) as instance:
        live = open_page(instance, base + "index.html", width=800, height=600)
        live.wait_for_selector("#made")
        live.fill("#name", "Ada")
        live.evaluate("document.getElementById('list').scrollTop = 120")
        live.screenshot()
        box = live.locator("#made").bounding_box()
        frozen = live.evaluate(FREEZE.read_text(encoding="utf-8"))
        assert frozen["missing"] == []
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
        assert_nothing_fetched(page)


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
