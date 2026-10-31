"""Main/Project composer clearance and wizard checkbox geometry in real UI flows.

Playwright's WebKit did not reproduce the system WKWebView defect behind the transcript's spacer; that engine has its
own optional check on macOS (devtools/macos_wkwebview/README.md).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ouroboros import dialogue_store
from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data
from tests.test_subscription_setup_browser import subscription_ui as subscription_ui
from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET, _emit_ws_frame, _SETTLE_RESTORE_FRAMES
from tests.ui_motion_samples import SAMPLE_TRANSITION


_MEASURE = """messages => {
    messages.scrollTop = messages.scrollHeight;
    const composer = messages.parentElement.querySelector('#chat-input-area, .chat-input-area');
    const last = [...messages.children].filter(el => el.getBoundingClientRect().height > 0).at(-1);
    const end = getComputedStyle(messages, '::after');
    return {height: messages.clientHeight, scrollHeight: messages.scrollHeight,
        scrollTop: messages.scrollTop, padding: getComputedStyle(messages).paddingBottom,
        spacer: end.content, basis: end.flexBasis, shrink: end.flexShrink,
        lastBottom: last?.getBoundingClientRect().bottom ?? null,
        composerTop: composer.getBoundingClientRect().top,
        reserve: getComputedStyle(messages).getPropertyValue('--chat-input-reserve')};
}"""


@pytest.mark.ui_browser
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_main_and_project_composer_clearance(direct_server_with_data, engine):
    from playwright.sync_api import sync_playwright
    from ouroboros.projects_registry import create_project

    data = direct_server_with_data["data_dir"]
    project = create_project(data, "composer-room", name="Composer room")
    dialogue_store.append(data, "chat", {
        "ts": "2026-09-17T10:00:00Z", "direction": "out", "chat_id": project["chat_id"],
        "text": "Saved Project answer.\n" + "\n".join(f"History line {i}" for i in range(12)),
        "format": "markdown",
    })
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", data.parent))
    evidence.mkdir(parents=True, exist_ok=True)
    observations = []
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page = browser.new_page(viewport={"width": 1187, "height": 734})
            page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
            page.goto(direct_server_with_data["url"], wait_until="domcontentloaded")
            page.wait_for_function("() => window.__testSockets?.some(s => s.readyState === 1)")
            for width in (1187, 390):
                page.set_viewport_size({"width": width, "height": 734})
                for surface in ("main", "project"):
                    if surface == "project":
                        page.locator('.nav-project-row[data-project-id="composer-room"]').evaluate("el => el.click()")
                        messages = page.locator('#project-panel .chat-messages')
                        messages.locator('.chat-bubble').filter(has_text="Saved Project answer").wait_for(state="visible")
                    else:
                        page.locator('[data-nav-page="chat"]').evaluate("el => el.click()")
                        messages = page.locator('#chat-messages')
                    page.evaluate(_SETTLE_RESTORE_FRAMES)
                    for text in ("", "one\ntwo\nthree\nfour\nfive"):
                        composer_input = messages.locator('xpath=..').locator('textarea')
                        composer_input.fill(text)
                        page.evaluate(_SETTLE_RESTORE_FRAMES)
                        metrics = messages.evaluate(_MEASURE)
                        observations.append({"width": width, "surface": surface, "multiline": bool(text), **metrics})
                        assert metrics["padding"] == "0px" and metrics["shrink"] == "0", metrics
                        assert metrics["lastBottom"] is None or metrics["lastBottom"] <= metrics["composerTop"] + 1, metrics
                    chat_id = project["chat_id"] if surface == "project" else 1
                    _emit_ws_frame(page, {"type": "chat", "role": "assistant", "chat_id": chat_id,
                        "content": "Growing answer.\n" + "\n".join(f"New line {i}" for i in range(50)),
                        "ts": "2026-09-18T10:00:00Z"})
                    page.evaluate(_SETTLE_RESTORE_FRAMES)
                    metrics = messages.evaluate(_MEASURE)
                    assert metrics["lastBottom"] <= metrics["composerTop"] + 1, metrics
                    page.screenshot(path=str(evidence / f"composer-{engine}-{surface}-{width}.png"), full_page=True)
                    if width == 390:
                        page.evaluate("() => {document.documentElement.style.setProperty('--vvh','500px'); document.body.classList.add('keyboard-open'); document.documentElement.classList.add('keyboard-open');}")
                        metrics = messages.evaluate(_MEASURE)
                        assert metrics["spacer"] == "none", metrics
                        assert metrics["lastBottom"] <= metrics["composerTop"] + 1, metrics
                        page.evaluate("() => {document.body.classList.remove('keyboard-open'); document.documentElement.classList.remove('keyboard-open');}")
            (evidence / f"composer-{engine}.json").write_text(json.dumps(observations, indent=2), encoding="utf-8")
        finally:
            browser.close()


_FIRST_FRAME_BACK = """async () => {
    document.querySelector('[data-nav-page="chat"]').click();
    await new Promise((resolve) => requestAnimationFrame(resolve));
    const messages = document.getElementById('chat-messages');
    const last = [...messages.querySelectorAll('.chat-bubble')].at(-1);
    return {lastBottom: last.getBoundingClientRect().bottom,
            composerTop: document.getElementById('chat-input-area').getBoundingClientRect().top,
            reserve: getComputedStyle(messages).getPropertyValue('--chat-input-reserve')};
}"""


@pytest.mark.ui_browser
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_main_behind_a_page_keeps_its_reserve_and_comes_back_above_the_composer(direct_server_with_data, engine):
    """Main hidden behind a page measured its composer as 0 and wrote the 92 px minimum reserve; back on Main its
    newest message was laid out 80 px low, under the composer, until the next measure (WebKit). Hidden, it keeps the
    reserve it was laid out with, so the first frame back already clears the composer."""
    from playwright.sync_api import sync_playwright

    data = direct_server_with_data["data_dir"]
    dialogue_store.append(data, "chat", {
        "ts": "2026-09-17T10:00:00Z", "direction": "out", "chat_id": 1,
        "text": "Main answer.\n" + "\n".join(f"Main line {i}" for i in range(40)), "format": "markdown",
    })
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page = browser.new_page(viewport={"width": 390, "height": 734})
            page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
            page.goto(direct_server_with_data["url"], wait_until="domcontentloaded")
            page.wait_for_function("() => window.__testSockets?.some(s => s.readyState === 1)")
            messages = page.locator('#chat-messages')
            messages.locator('.chat-bubble').filter(has_text="Main answer").wait_for(state="visible")
            page.locator('#chat-input').fill("one\ntwo\nthree\nfour")
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            shown = messages.evaluate("m => getComputedStyle(m).getPropertyValue('--chat-input-reserve')")
            page.locator('[data-nav-page="settings"]').evaluate("el => el.click()")
            page.locator('#page-settings').wait_for(state="visible")
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            hidden = messages.evaluate("m => getComputedStyle(m).getPropertyValue('--chat-input-reserve')")
            back = page.evaluate(_FIRST_FRAME_BACK)
        finally:
            browser.close()
    assert int(shown.strip().removesuffix("px")) > 92, shown
    assert hidden == shown, ("hidden behind a page, Main kept its reserve", shown, hidden)
    assert back["lastBottom"] <= back["composerTop"] + 1, ("the first frame back clears the composer", back)


# Where ☰ stands along the real transition: Reply on the newest answer, or the strip's ×.
_MENU_FRAMES = """async (action) => {
    const menu = document.querySelector('#page-chat .chat-toolbar-row [data-mobile-nav-toggle]');
    const strip = () => document.querySelector('#page-chat .chat-quote--composer');
    const sample = """ + SAMPLE_TRANSITION + """;
    return sample({root: document.querySelector('#chat-input-area'), target: strip, property: 'max-height',
        ready: () => strip()?.dataset.motion === (action === 'stage' ? 'open' : 'shut') && !strip().hasAttribute('data-hold'),
        act: () => { if (action === 'stage') [...document.querySelectorAll('#chat-messages .chat-message-reply')].at(-1).click();
                     else strip().querySelector('.chat-quote-cancel').click(); },
        measure: () => menu.getBoundingClientRect().top});
}"""


@pytest.mark.ui_browser
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
@pytest.mark.parametrize("reduce", [False, True], ids=["motion", "reduced"])
def test_a_reply_strip_opens_and_closes_like_the_capsule_rows_fold(direct_server_with_data, engine, reduce):
    """Owner 20.10 (A): staging or cancelling a reply moved the capsule row and its ☰ about 60 px in one frame. The strip
    opens and closes like the row's fold now, so ☰ travels through intermediate positions; reduced motion takes one
    step. Seek the browser's own transitions: a loaded runner can miss every frame of a 280 ms animation."""
    from playwright.sync_api import sync_playwright

    data = direct_server_with_data["data_dir"]
    dialogue_store.append(data, "chat", {
        "ts": "2026-09-17T10:00:00Z", "direction": "out", "chat_id": 1,
        "text": "An answer worth replying to.", "format": "markdown",
    })
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page = browser.new_page(viewport={"width": 390, "height": 734},
                                    reduced_motion="reduce" if reduce else "no-preference")
            page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
            page.goto(direct_server_with_data["url"], wait_until="domcontentloaded")
            page.wait_for_function("() => window.__testSockets?.some(s => s.readyState === 1)")
            page.locator('#chat-messages .chat-bubble').filter(has_text="An answer worth replying to").wait_for(state="visible")
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            moves = {"opening": page.evaluate(_MENU_FRAMES, "stage")}
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            moves["closing"] = page.evaluate(_MENU_FRAMES, "cancel")
        finally:
            browser.close()
    for name, motion in moves.items():
        tops = motion['values']
        steps = [round(b - a, 2) for a, b in zip(tops, tops[1:])]
        travel = abs(tops[-1] - tops[0])
        moving = [step for step in steps if abs(step) > 0.5]
        assert travel > 30, (name, tops)
        if reduce:
            assert motion['duration'] == 0 and len(moving) <= 1, (name, motion)
        else:
            assert motion['duration'] > 0, (name, motion)
            assert len(moving) >= 3 and max(abs(step) for step in steps) < travel * 0.6, (name, steps)


@pytest.mark.ui_browser
@pytest.mark.serial
def test_wizard_enabled_checkbox_stays_inline(subscription_ui):
    """The real wizard keeps its already-fixed checkbox width at phone scale."""
    from tests.test_ui_smoke_agents_panel import _wizard_step_until, _WIZARD_ON_AGENTS_JS
    from tests.test_subscription_setup_browser import capture

    page = subscription_ui["page"]
    page.goto(subscription_ui["url"] + "/onboarding")
    _wizard_step_until(page, _WIZARD_ON_AGENTS_JS, forward=True)
    label = page.locator('#onboarding-available-subagents .available-subagents-toolbar .local-toggle')
    for width in (1440, 390):
        page.set_viewport_size({"width": width, "height": 900})
        label.scroll_into_view_if_needed()
        geometry = label.evaluate("""label => {
            const input = label.querySelector('input');
            const box = input.getBoundingClientRect();
            const text = [...label.childNodes].find(n => n.nodeType === Node.TEXT_NODE && n.textContent.trim());
            const range = new Range(); range.selectNodeContents(text);
            const word = range.getBoundingClientRect();
            return {checkboxWidth:box.width, height:label.getBoundingClientRect().height,
                textTop:word.top, textBottom:word.bottom, checkboxTop:box.top, checkboxBottom:box.bottom};
        }""")
        assert geometry["checkboxWidth"] == 14, geometry
        assert geometry["height"] <= 28, geometry
        assert geometry["textBottom"] <= geometry["checkboxBottom"] + 6, geometry
        capture(page, f"wizard-inline-checkbox-{width}")
