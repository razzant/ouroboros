"""Composer refitting stays free of ResizeObserver error events in both browser engines.

The owner-controls observer used to mutate its observed row synchronously, which also
resized the ancestor composer. A width sweep and container-only resize exercise that
feedback path. Asynchronous frame-adjacent samples check bounded convergence; they are
not a proof of every displayed frame or native WKWebView behaviour.
"""
from __future__ import annotations

import json

import pytest

from tests.test_effort_range_browser import _evidence_dir, _launch, _open_chat
from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401 - pytest fixture import

pytestmark = [pytest.mark.ui_browser]

ROW_SEL = "#page-chat .chat-toolbar-row"
# Window `error` events, recorded before any page script runs. `pageerror` is not a reliable carrier of this
# event in every engine, and the test is about the event the owner's webview would see.
WINDOW_ERRORS = "() => { window.__windowErrors = []; addEventListener('error', (e) => window.__windowErrors.push(String(e.message))); }"
# Frame-adjacent height samples since the last reset, not a capture of every paint.
END_OF_FRAME = f"""() => {{
    window.__eof = [];
    const tick = () => requestAnimationFrame(() => setTimeout(() => {{
        window.__eof.push(Math.round(document.querySelector('{ROW_SEL}').getBoundingClientRect().height)); tick();
    }}, 0));
    tick();
}}"""
STATE = f"""() => {{ const row = document.querySelector('{ROW_SEL}'); const r = row.getBoundingClientRect();
    return {{ height: Math.round(r.height), width: Math.round(r.width), compact: row.dataset.ownerCompact || '' }}; }}"""
RESIZE_LOOP = "ResizeObserver loop"


def _eof(page):
    samples = page.evaluate("() => window.__eof.splice(0)")
    return samples


def _loop_errors(page):
    return [m for m in page.evaluate("() => window.__windowErrors") if RESIZE_LOOP in m]


def _launch_with_recorders(pw, engine, url, width, tmp_path):
    browser, page = _launch(pw, engine, viewport={"width": width, "height": 900})
    page.add_init_script(f"({WINDOW_ERRORS})()")
    page_errors = []
    page.on('pageerror', lambda error: page_errors.append(str(error)))
    _open_chat(page, url, width=width)
    page.evaluate(END_OF_FRAME)
    page.wait_for_timeout(300)
    return browser, page, page_errors


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_window_narrowing_refits_without_a_resize_observer_error(direct_server_with_data, engine, tmp_path):  # noqa: F811
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    with sync_playwright() as pw:
        browser, page, page_errors = _launch_with_recorders(pw, engine, url, 1240, tmp_path)
        facts = {}
        try:
            one_line = page.evaluate(STATE)["height"]
            facts = {"one_line": one_line, "steps": {}, "sequence": []}
            # Narrow through the width where the closed row first wraps, and back out; the loop of widths is the
            # original strict sequence (1240 .. 320) refined around the threshold.
            for width in (1240, 430, 400, 390, 380, 370, 360, 350, 340, 320, 300, 340, 390, 430, 1240):
                _eof(page)
                page.set_viewport_size({"width": width, "height": 900})
                page.wait_for_timeout(150)
                samples = _eof(page)
                step = {"state": page.evaluate(STATE), "frame_adjacent_heights": samples}
                facts["steps"][str(width)] = step
                facts["sequence"].append({"width": width, **step})
                wrapped = [h for h in samples if h > one_line * 1.5]
                assert not wrapped, f"{engine} {width}px: a wrapped row was sampled ({wrapped}); the resize fit must be observable promptly"
                assert page.evaluate(STATE)["height"] <= one_line * 1.5, (width, facts["steps"][str(width)])
            crossed = [w for w, s in facts["steps"].items() if s["state"]["compact"]]
            assert crossed, f"the sweep never made the row compact, so it proved nothing: {facts['steps']}"
            errors = _loop_errors(page)
            facts["loop_errors"] = errors
            (_evidence_dir(tmp_path) / f"composer-resize-loop-window-{engine}.json").write_text(json.dumps(facts, indent=1), encoding="utf-8")
            assert not errors, errors
            assert not page.evaluate('() => window.__windowErrors'), facts
            assert not page_errors, page_errors
        finally:
            facts['loop_errors'] = _loop_errors(page)
            facts['window_errors'] = page.evaluate('() => window.__windowErrors')
            facts['page_errors'] = page_errors
            (_evidence_dir(tmp_path) / f'composer-window-{engine}.json').write_text(json.dumps(facts, indent=1), encoding='utf-8')
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_container_only_narrowing_refits_the_row_without_a_resize_observer_loop_error(direct_server_with_data, engine, tmp_path):  # noqa: F811
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    with sync_playwright() as pw:
        # The native window's minimum width; the sidebar's drag handle then narrows only the chat column.
        browser, page, page_errors = _launch_with_recorders(pw, engine, url, 800, tmp_path)
        facts = {}
        try:
            one_line = page.evaluate(STATE)["height"]
            facts = {"one_line": one_line, "steps": {}, "sequence": []}
            for sidebar in (280, 340, 400, 460, 520, 560, 460, 340, 280):
                _eof(page)
                page.evaluate("w => document.documentElement.style.setProperty('--sidebar-width', w + 'px')", sidebar)
                page.wait_for_timeout(150)
                samples = _eof(page)
                state = page.evaluate(STATE)
                step = {"state": state, "frame_adjacent_heights": samples}
                facts["steps"][str(sidebar)] = step
                facts["sequence"].append({"sidebar": sidebar, **step})
                # No window resize announced this, so the refit may land one frame later, never more.
                assert sum(h > one_line * 1.5 for h in samples) <= 1, (sidebar, samples)
                assert state["height"] <= one_line * 1.5, (sidebar, state)
            assert facts["steps"]["560"]["state"]["compact"], facts["steps"]["560"]
            assert not facts["steps"]["280"]["state"]["compact"], "the names come back when the column widens again"
            errors = _loop_errors(page)
            facts["loop_errors"] = errors
            (_evidence_dir(tmp_path) / f"composer-resize-loop-container-{engine}.json").write_text(json.dumps(facts, indent=1), encoding="utf-8")
            assert not errors, errors
            assert not page.evaluate('() => window.__windowErrors'), facts
            assert not page_errors, page_errors
        finally:
            facts['loop_errors'] = _loop_errors(page)
            facts['window_errors'] = page.evaluate('() => window.__windowErrors')
            facts['page_errors'] = page_errors
            (_evidence_dir(tmp_path) / f'composer-container-{engine}.json').write_text(json.dumps(facts, indent=1), encoding='utf-8')
            browser.close()
