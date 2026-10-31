"""Before-paint fitting in the real SPA, including the first mount without state.

A read-only row observer is installed by a DOM observer after the actual controls mount.
Its initial/resize deliveries occur after the product's docking-box callback in the same
render update. This is deliberately not a read in the resize event or its first rAF:
those precede ResizeObserver and would reject a fit that is ready before paint.
For a text change inside rAF, the following frame samples the preceding painted state.
The ordinary capsule suite independently covers opening/closing, gestures and RTL.
"""
from __future__ import annotations

import pytest

from tests.test_composer_capsules_browser import LONG_VALUES, _record
from tests.test_effort_range_browser import _launch, _open_chat, _shot
from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

PROBE = """() => {
    const seen = new WeakSet(), rows = [], observers = [];
    window.__composerPaint = { samples: [], errors: [], phase: 'mount' };
    const probe = window.__composerPaint;
    window.addEventListener('error', e => probe.errors.push(String(e.message)));
    probe.sample = row => {
        const r = row.getBoundingClientRect();
        if (!(r.width > 0 && r.height > 0)) return;
        const kids = [...row.children].filter(n => n.getBoundingClientRect().width > 0 && getComputedStyle(n).display !== 'none');
        if (!kids.length) return;
        const boxes = kids.map(n => n.getBoundingClientRect()), css = getComputedStyle(row);
        probe.samples.push({
            phase: probe.phase, room: row.closest('#page-chat, .chat-instance-panel')?.id,
            width: r.width, height: r.height,
            oneLine: Math.max(...boxes.map(b => b.top)) - Math.min(...boxes.map(b => b.top)) < 2,
            contained: boxes.every(b => b.left >= r.left + parseFloat(css.paddingLeft) - 1 && b.right <= r.right - parseFloat(css.paddingRight) + 1),
            compact: row.dataset.ownerCompact || '', solo: row.dataset.ownerSolo || '',
            known: row.querySelector('.chat-effort-range')?.dataset.known,
            names: [...row.querySelectorAll('.chat-capsule-name')].map(n => n.getBoundingClientRect().width > 0),
        });
    };
    probe.sampleAll = () => rows.filter(r => r.isConnected).forEach(probe.sample);
    const mount = new MutationObserver(() => {
        for (const row of document.querySelectorAll('.chat-toolbar-row')) {
            if (seen.has(row) || row.querySelectorAll('.chat-capsule').length !== 3) continue;  // Subagents, Context, Effort
            seen.add(row); rows.push(row);
            const observer = new ResizeObserver(() => probe.sample(row));
            observer.observe(row); observers.push(observer);
        }
    });
    mount.observe(document, { childList: true, subtree: true });
    probe.stop = () => { mount.disconnect(); observers.forEach(o => o.disconnect()); };
}"""


def _frames(page, count=4):
    page.evaluate("""count => new Promise(resolve => {
        const step = () => { if (--count <= 0) resolve(); else requestAnimationFrame(step); };
        requestAnimationFrame(step);
    })""", count)


def _phase(page, name):
    _frames(page)
    page.evaluate("name => { __composerPaint.phase = name; __composerPaint.samples = []; }", name)


def _assert_phase(page, room, *, names=None):
    _frames(page)
    page.evaluate("__composerPaint.sampleAll()")
    facts = page.evaluate("__composerPaint.samples")
    rows = [row for row in facts if row["room"] == room]
    assert rows, (room, facts)
    assert all(row["oneLine"] and row["contained"] for row in rows), rows
    if names is not None:
        assert all(row["names"] == [names] * 3 for row in rows), rows
    assert page.evaluate("__composerPaint.errors") == []
    return rows


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
@pytest.mark.parametrize("width", [340, 360])
def test_initial_visible_mount_fits_before_state(direct_server_with_data, engine, width, tmp_path):  # noqa: F811
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser, page = _launch(pw, engine, viewport={"width": width, "height": 900}, reduced_motion="reduce")
        try:
            page.add_init_script(f"({PROBE})()")
            # An unavailable state is supported: layout must not depend on the polling answer.
            page.route("**/api/state", lambda route: route.fulfill(status=503, json={"error": "test state unavailable"}))
            page.goto(direct_server_with_data["url"], wait_until="domcontentloaded")
            page.wait_for_function("__composerPaint.samples.some(s => s.room === 'page-chat')")
            samples = _assert_phase(page, "page-chat")
            _record(tmp_path, f"first-mount-{engine}-{width}", samples)
            _shot(page, tmp_path, f"first-mount-{engine}-{width}")
            assert samples[0]["known"] == "false", "the control must be observed before state arrives"
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
@pytest.mark.parametrize("motion", ["reduce", "no-preference"])
def test_width_show_and_translation_do_not_wait_a_painted_frame(direct_server_with_data, engine, motion, tmp_path):  # noqa: F811
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser, page = _launch(pw, engine, viewport={"width": 1440, "height": 900}, reduced_motion=motion)
        facts = {}
        try:
            page.add_init_script(f"({PROBE})()")
            # The chosen dictionary is also the gateway's answer: a reconnect must not restore
            # the fixture's English default over a manually applied translated snapshot.
            language = {"payload": {"language": "en", "english": True, "entries": {}, "revision": 0}}
            page.route("**/api/ui/i18n", lambda route: route.fulfill(json=language["payload"]))
            page.route("**/api/ui/i18n/missing", lambda route: route.fulfill(json={"ok": True}))
            _open_chat(page, direct_server_with_data["url"], width=1440)
            for width in (360, 600, 340, 1440):
                _phase(page, f"viewport-{width}")
                page.set_viewport_size({"width": width, "height": 900})
                facts[f"viewport-{width}"] = _assert_phase(page, "page-chat")
            # Main's real docking page is hidden on navigation. Change its allocated width while hidden,
            # then show it within rAF: the next frame must already show the fitted closed controls.
            page.evaluate("document.querySelector('#page-chat').classList.remove('active')")
            _frames(page)
            page.set_viewport_size({"width": 340, "height": 900})
            _phase(page, "show-narrow")
            page.evaluate("""() => new Promise(resolve => requestAnimationFrame(() => {
                document.querySelector('#page-chat').classList.add('active'); requestAnimationFrame(resolve);
            }))""")
            facts["show-narrow"] = _assert_phase(page, "page-chat")
            _phase(page, "translated")
            language["payload"] = LONG_VALUES["ltr"][0]
            page.evaluate("""async payload => {
                const {setLanguage} = await import('/static/modules/i18n.js');
                await new Promise(resolve => requestAnimationFrame(async () => {
                    await setLanguage(payload.language, payload);
                    requestAnimationFrame(() => { __composerPaint.sampleAll(); resolve(); });
                }));
            }""", LONG_VALUES["ltr"][0])
            facts["translated"] = _assert_phase(page, "page-chat")
            _shot(page, tmp_path, f"first-frame-translated-{engine}-{motion}")
            assert page.locator('#page-chat .chat-context-mode .chat-capsule-value').inner_text() == LONG_VALUES["ltr"][1]
        finally:
            _record(tmp_path, f"first-frame-paths-{engine}-{motion}", facts)
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_restored_project_width_and_real_splitter_fit_before_paint(direct_server_with_data, engine, tmp_path):  # noqa: F811
    from playwright.sync_api import sync_playwright
    from ouroboros.projects_registry import create_project

    create_project(direct_server_with_data["data_dir"], "first-frame", name="First frame")
    with sync_playwright() as pw:
        browser, page = _launch(pw, engine, viewport={"width": 1440, "height": 900}, reduced_motion="reduce")
        facts = {}
        room = "panel-pchat-first-frame"
        try:
            response = page.request.post(direct_server_with_data["url"] + "/api/ui/preferences", data={"project_panel_width": 340})
            assert response.status == 200
            page.add_init_script(f"({PROBE})()")
            _open_chat(page, direct_server_with_data["url"], width=1440)
            _phase(page, "restored-project")
            page.click('.nav-project-row[data-project-id="first-frame"]')
            page.wait_for_selector(f"#{room} .chat-effort-range")
            facts["restored-project"] = _assert_phase(page, room)
            assert abs(page.locator('#project-panel').bounding_box()["width"] - 340) < 1
            for target in (760, 340):
                _phase(page, f"splitter-{target}")
                handle = page.locator('#project-panel > .resize-handle-left').bounding_box()
                current = page.locator('#project-panel').bounding_box()["width"]
                x, y = handle["x"] + handle["width"] / 2, handle["y"] + handle["height"] / 2
                page.mouse.move(x, y)
                page.mouse.down()
                page.mouse.move(x + current - target, y)
                page.mouse.up()
                facts[f"splitter-{target}"] = _assert_phase(page, room)
            _shot(page, tmp_path, f"first-frame-project-{engine}")
        finally:
            _record(tmp_path, f"first-frame-project-{engine}", facts)
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
@pytest.mark.parametrize("kind", ["context", "effort"])
def test_value_change_preserves_owner_selection_motion(direct_server_with_data, engine, kind, tmp_path):  # noqa: F811
    """A final position alone passes when the text observer cancels the owner's transition."""
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    with sync_playwright() as pw:
        browser, page = _launch(pw, engine, viewport={"width": 1440, "height": 900}, reduced_motion="no-preference")
        facts = {}
        try:
            assert page.request.post(url + "/api/owner/context-mode", data={"mode": "max"}).status == 200
            assert page.request.post(url + "/api/owner/effort-range", data={"min": "low", "recommended": "high", "max": "max"}).status == 200
            _open_chat(page, url, width=1440)
            selector = f"#page-chat .chat-{'context-mode' if kind == 'context' else 'effort-range'}"
            page.locator(selector + " .chat-capsule-head").click()
            page.wait_for_function("sel => { const el = document.querySelector(sel); return el.dataset.open === 'true' && !el.getAnimations({subtree: true}).length; }", arg=selector)
            _frames(page)
            page.evaluate("""({selector, kind}) => {
                const el = document.querySelector(selector), value = el.querySelector('.chat-capsule-value');
                const mark = el.querySelector(kind === 'context' ? '.chat-context-thumb' : '.chat-effort-rec');
                const sample = () => ({left: parseFloat(getComputedStyle(mark).left), animations: mark.getAnimations().length});
                const probe = window.__selectionMotion = {start: sample(), samples: [], done: false};
                let started = false;
                const observer = new MutationObserver(() => {
                    if (started) return;
                    started = true;
                    // Registered after the product observer; sample after its whole microtask turn.
                    queueMicrotask(() => {
                        probe.samples.push(sample());
                        const tick = () => {
                            probe.samples.push(sample());
                            if (probe.samples.length < 24) requestAnimationFrame(tick);
                            else { probe.done = true; observer.disconnect(); }
                        };
                        requestAnimationFrame(tick);
                    });
                });
                observer.observe(value, {childList: true, characterData: true, subtree: true});
            }""", {"selector": selector, "kind": kind})
            if kind == "context":
                page.locator(selector + ' .chat-seg[data-mode="nano"]').click()
            else:
                page.locator(selector + " .chat-effort-rec").focus()
                page.keyboard.press("ArrowLeft")
            page.wait_for_function("window.__selectionMotion?.done")
            facts = page.evaluate("__selectionMotion")
            assert facts["start"]["animations"] == 0, facts
            assert abs(facts["start"]["left"] - facts["samples"][-1]["left"]) > 1, facts
            assert any(s["animations"] > 0 and abs(s["left"] - facts["samples"][-1]["left"]) > 0.5 for s in facts["samples"]), facts
        finally:
            _record(tmp_path, f"owner-selection-motion-{kind}-{engine}", facts)
            browser.close()
