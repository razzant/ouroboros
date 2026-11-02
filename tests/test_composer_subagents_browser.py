"""The Subagents capsule in the real composer's row (DESIGN "Composer owner controls"; owner 11.10, again 27.10).

On a real server in Chromium and WebKit, at 1280 and 390, light and dark: the capsule opens in the row as
`Subagents | Off Auto Swarm`, beside the other capsules on a desk and alone on a phone (the others step aside and the
strip takes the line); while it is open the one hint stands above it (what the chosen value means and how a change
applies; DESIGN "Hints"), and a choice saves at once while the meaning follows it. With Settings › Agents off it says
Off with a lock, Auto and Swarm cannot be chosen, the hint says why with Open Settings, and Open Settings closes the
strip and opens Settings › Agents. The room's subagents facts are the test's on every state read. On a phone a value
keeps its look and a finger's target: a tap a few px above or below it chooses it, on Context's and Effort's strips
too. On a 320 px phone the keyboard walks it: Enter opens onto the chosen value, an arrow chooses, Escape gives the
focus back, Tab goes from the strip to Open Settings and an outside press closes it.
"""
from __future__ import annotations

import json
import re

import pytest

from tests.test_effort_range_browser import _evidence_dir, _launch, _open_chat, _shot
from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401 - pytest fixture import

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

SUBAGENTS = "#page-chat .chat-subagents-mode"
CONTEXT = "#page-chat .chat-context-mode"
EFFORT = "#page-chat .chat-effort-range"
RULE = "Applies to new steps; running work finishes."
# What each value means, read from the capsule's own module: the open hint says exactly these words.
MEANINGS = "async () => (await import('/static/modules/composer_subagents.js')).SUBAGENT_MEANINGS"
# The page's state read: the first `/api/state` and every `?bindings_since=` read after it (chat_activity.js).
STATE_READS = re.compile(r"/api/state(\?|$)")
# The capsule, the capsules shown on its line, its values and its hint.
FACTS = """() => {
    const row = document.querySelector('#page-chat .chat-toolbar-row');
    const el = row.querySelector('.chat-subagents-mode');
    const hint = el.querySelector('.ui-hint');
    const rect = (node) => { const r = node.getBoundingClientRect(); return { left: r.left, right: r.right, top: r.top, bottom: r.bottom }; };
    const shown = (node) => Boolean(node) && node.getClientRects().length > 0 && getComputedStyle(node).visibility !== 'hidden';
    const kind = (node) => ['context', 'effort', 'menu'].find((k) => node.matches(`.chat-${k}-mode, .chat-${k}-range, .composer-${k}`)) || '';
    const text = (part) => (shown(hint?.querySelector(`.ui-hint-${part}`)) ? hint.querySelector(`.ui-hint-${part}`).textContent : '');
    const segs = [...el.querySelectorAll('.chat-seg')];
    return {
        open: el.dataset.open, fit: el.dataset.fit || '', locked: el.dataset.locked, mode: el.dataset.mode,
        lock: shown(el.querySelector('.chat-subagents-lock')), capsule: rect(el),
        beside: [...row.children].filter((kid) => kid !== el && shown(kid) && kind(kid)).map(kind),
        values: segs.map((seg) => [seg.dataset.mode, seg.getAttribute('aria-checked'), seg.getAttribute('aria-disabled') || '']),
        inCapsule: segs.every((seg) => shown(seg) && seg.getBoundingClientRect().top >= el.getBoundingClientRect().top - 1
            && seg.getBoundingClientRect().bottom <= el.getBoundingClientRect().bottom + 1),
        hint: shown(hint) ? rect(hint) : null, field: rect(document.getElementById('chat-input')),
        word: text('word'), meaning: text('text'), rule: text('rule'), settings: text('action') === 'Open Settings',
        docScroll: document.documentElement.scrollWidth > innerWidth + 1,
    };
}"""


def _hold(page, facts: dict) -> dict:
    """Every state read answers the room's subagents facts the test holds; a save lands in them (routed once: the test
    changes them in place)."""
    page.route(STATE_READS, lambda route: route.fulfill(json={**route.fetch().json(), **facts}))

    def save(route):
        facts["subagents_mode"] = route.request.post_data_json["mode"]
        route.fulfill(json={"ok": True, "project_id": "", "subagents_mode": facts["subagents_mode"]})
    page.route("**/api/owner/subagents-mode", save)
    return facts


def _settle(page) -> None:
    # The opening's own motion (capsule_morph.js, the strip's transitions) ends before anything is measured.
    page.wait_for_function("""() => { const row = document.querySelector('#page-chat .chat-toolbar-row');
        return !('ownerLeaving' in row.dataset) && row.getAnimations({subtree: true})
            .filter((a) => a.effect?.getComputedTiming().endTime !== Infinity)
            .every((a) => !a.pending && a.playState !== 'running'); }""", timeout=10_000)
    page.wait_for_timeout(150)


# A target's centre and its drawn top and bottom.
SPOT = "(node) => { const r = node.getBoundingClientRect(); return { x: r.left + r.width / 2, top: r.top, bottom: r.bottom }; }"
# Where Effort's brackets stand and which level is the primary.
BRACKETS = """(el) => ({ caps: [...el.querySelectorAll('.chat-effort-cap')].map((cap) => {
    const r = cap.getBoundingClientRect(); return [r.left, r.right]; }),
    rec: el.querySelector('.chat-effort-seg[data-rec="true"]')?.dataset.effortSeg || '' })"""


def _open(page, press) -> dict:
    closed = page.locator(SUBAGENTS).evaluate("(el) => el.getBoundingClientRect().top")
    press(f"{SUBAGENTS} .chat-capsule-head")
    page.wait_for_function("sel => document.querySelector(sel)?.dataset.open === 'true'", arg=SUBAGENTS)
    _settle(page)
    return _since(closed, page.evaluate(FACTS))


def _since(closed: float, facts: dict) -> dict:
    """How far the capsule rose since it stood closed (the hint never joins the row's flow, so it never rises)."""
    return {**facts, "closed": closed, "rose": round(closed - facts["capsule"]["top"], 1)}


def _assert_in_the_row(facts: dict, *, phone: bool) -> None:
    """Open in the row: its values inside the capsule on the composer's line, beside the others on a desk and alone on
    a phone; the hint above the capsule, inside the window; nothing scrolls sideways."""
    assert facts["open"] == "true" and facts["inCapsule"], facts
    if phone:
        assert facts["beside"] == [], ("on a phone the strip takes the line: the others step aside", facts)
    else:
        assert facts["fit"] == "inline" and {"context", "effort"} <= set(facts["beside"]), ("beside the others", facts)
    hint = facts["hint"]
    assert hint and hint["bottom"] <= facts["capsule"]["top"] + 1 and hint["top"] >= 0, \
        ("the hint stands above the capsule, inside the window", facts)
    assert abs(facts["rose"]) <= 1, ("opening moved the capsule: the hint never joins the row's flow", facts)
    assert not facts["docScroll"], facts


@pytest.mark.parametrize("engine,width", [("chromium", 1280), ("webkit", 1280), ("chromium", 390), ("webkit", 390)])
def test_subagents_opens_in_the_row_with_the_hint_above_it(
        direct_server_with_data, engine, width, tmp_path):  # noqa: F811
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    phone = width < 720
    context = {"viewport": {"width": width, "height": 860}}
    if phone:
        context.update(is_mobile=True, has_touch=True)
    seen = {}
    with sync_playwright() as pw:
        browser, page = _launch(pw, engine, **context)
        press = page.tap if phone else page.click
        try:
            held = _hold(page, {})
            for theme in ("light", "dark"):
                held.update(subagents_mode="auto", subagents_availability="available")
                _open_chat(page, url, width=width, height=860, theme=theme)
                facts = seen[f"{theme}-auto"] = opened = _open(page, press)
                _shot(page, tmp_path, f"subagents-inline-auto-{engine}-{width}-{theme}")
                _assert_in_the_row(facts, phone=phone)
                meanings = page.evaluate(MEANINGS)
                assert facts["word"] == "Auto" and facts["meaning"] == meanings["auto"], (facts, meanings)
                assert facts["rule"] == RULE, facts
                assert not facts["settings"] and not facts["lock"], facts
                # A choice saves at once; the strip stays open and the meaning follows it.
                with page.expect_response(lambda r: r.url.endswith("/api/owner/subagents-mode"), timeout=30_000):
                    press(f'{SUBAGENTS} .chat-seg[data-mode="swarm"]')
                page.wait_for_function("sel => document.querySelector(sel).dataset.mode === 'swarm'", arg=SUBAGENTS)
                _settle(page)
                facts = seen[f"{theme}-swarm"] = _since(opened["closed"], page.evaluate(FACTS))
                _shot(page, tmp_path, f"subagents-inline-swarm-{engine}-{width}-{theme}")
                _assert_in_the_row(facts, phone=phone)
                assert facts["meaning"] == meanings["swarm"] and facts["rule"] == RULE, (facts, meanings)
                assert [value[1] for value in facts["values"]] == ["false", "false", "true"], facts
                page.keyboard.press("Escape")
                page.wait_for_function("sel => document.querySelector(sel).dataset.open === 'false'", arg=SUBAGENTS)
            # Settings › Agents turns subagents off: Off with a lock, only Off can be chosen, the hint says why with
            # Open Settings; Open Settings closes the strip and goes to Settings › Agents.
            for theme in ("light", "dark"):
                held.update(subagents_mode="swarm", subagents_availability="disabled")
                _open_chat(page, url, width=width, height=860, theme=theme)
                page.wait_for_function("sel => document.querySelector(sel)?.dataset.locked === 'true'", arg=SUBAGENTS,
                                       timeout=30_000)
                assert page.locator(f"{SUBAGENTS} .chat-capsule-value").inner_text() == "Off"
                facts = seen[f"{theme}-settings-off"] = _open(page, press)
                _shot(page, tmp_path, f"subagents-inline-settings-off-{engine}-{width}-{theme}")
                _assert_in_the_row(facts, phone=phone)
                assert facts["locked"] == "true" and facts["lock"], facts
                assert facts["values"] == [["off", "true", ""], ["auto", "false", "true"],
                                           ["swarm", "false", "true"]], facts
                assert not facts["word"] and not facts["rule"], ("the reason stands in the meaning's place", facts)
                assert facts["meaning"] == "Subagents are turned off in Settings › Agents." and facts["settings"], facts
                # A press on a value Settings rules out (aria-disabled, so the press is forced) chooses nothing.
                locked = page.locator(f'{SUBAGENTS} .chat-seg[data-mode="swarm"]')
                (locked.tap if phone else locked.click)(force=True)
                page.wait_for_timeout(300)
                assert page.locator(SUBAGENTS).get_attribute("data-mode") == "off"
            press(f"{SUBAGENTS} .ui-hint-action")
            page.wait_for_selector('.settings-panel.active[data-settings-panel="agents"]', state="visible",
                                   timeout=20_000)
            assert page.locator(SUBAGENTS).get_attribute("data-open") == "false"
        finally:
            (_evidence_dir(tmp_path) / f"subagents-inline-{engine}-{width}.json").write_text(
                json.dumps(seen, indent=1), encoding="utf-8")
            page.unroute_all(behavior="ignoreErrors")  # a state read still in flight ends with the browser
            browser.close()


@pytest.mark.parametrize("engine,width", [("chromium", 390), ("webkit", 390), ("chromium", 320), ("webkit", 320)])
def test_a_tap_just_outside_a_value_chooses_it_on_a_phone(direct_server_with_data, engine, width, tmp_path):  # noqa: F811
    """Under a finger every target is at least the touch minimum (DESIGN "Touch and containment"): the open strip's
    values keep their look and reach above and below the line, so a tap 4 px under or over a value, outside what is
    drawn, chooses it and the strip stays open; Subagents' Auto and Swarm, Context's mode, and Effort's level where its
    strip does not scroll (a strip that scrolls clips that reach). WebKit hit-tests a touch at its exact point, so
    without the reach the tap lands beside the capsule there and closes it."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    seen = {}
    with sync_playwright() as pw:
        browser, page = _launch(pw, engine, viewport={"width": width, "height": 860}, is_mobile=True, has_touch=True)

        def tap_off(target: str, edge: str, dy: int, endpoint: str) -> dict:
            spot = page.locator(target).evaluate(SPOT)
            with page.expect_response(lambda r: r.url.endswith(endpoint), timeout=30_000):
                page.touchscreen.tap(spot["x"], spot[edge] + dy)
            return spot

        def open_capsule(sel: str) -> None:
            page.tap(f"{sel} .chat-capsule-head")
            page.wait_for_function("sel => document.querySelector(sel)?.dataset.open === 'true'", arg=sel)
            _settle(page)

        def close_capsule(sel: str) -> None:
            page.keyboard.press("Escape")
            page.wait_for_function("sel => document.querySelector(sel).dataset.open === 'false'", arg=sel)
            _settle(page)

        try:
            _hold(page, {"subagents_mode": "off", "subagents_availability": "available"})
            _open_chat(page, url, width=width, height=860, theme="light")
            open_capsule(SUBAGENTS)
            for mode, edge, dy in (("auto", "bottom", 4), ("swarm", "top", -4)):
                spot = tap_off(f'{SUBAGENTS} .chat-seg[data-mode="{mode}"]', edge, dy, "/api/owner/subagents-mode")
                page.wait_for_function("([sel, mode]) => document.querySelector(sel).dataset.mode === mode",
                                       arg=[SUBAGENTS, mode])
                facts = seen[f"subagents-{mode}"] = {**page.evaluate(FACTS), "spot": spot}
                assert facts["open"] == "true", (f"{dy:+d} px off {mode}: chosen, and the strip stays open", facts)
            _shot(page, tmp_path, f"subagents-edge-tap-{engine}-{width}")
            close_capsule(SUBAGENTS)
            # Context's strip is the same mode strip.
            open_capsule(CONTEXT)
            mode = "low" if page.locator(CONTEXT).get_attribute("data-context-mode") != "low" else "max"
            spot = tap_off(f'{CONTEXT} .chat-seg[data-mode="{mode}"]', "bottom", 4, "/api/owner/context-mode")
            page.wait_for_function("([sel, mode]) => document.querySelector(sel).dataset.contextMode === mode",
                                   arg=[CONTEXT, mode])
            seen["context"] = {"mode": mode, "spot": spot, "open": page.locator(CONTEXT).get_attribute("data-open")}
            assert seen["context"]["open"] == "true", seen["context"]
            close_capsule(CONTEXT)
            # Effort: a level tapped 4 px under it becomes the primary, clear of the brackets' own reach.
            open_capsule(EFFORT)
            fit = page.locator(EFFORT).get_attribute("data-fit") or ""
            seen["effort"] = {"fit": fit}
            if fit != "overflow":
                level = f'{EFFORT} .chat-effort-seg[data-effort-seg="xhigh"]'
                before = page.locator(EFFORT).evaluate(BRACKETS)
                x = page.locator(level).evaluate(SPOT)["x"]
                clear = all(x < left - 16 or x > right + 16 for left, right in before["caps"])
                assert before["rec"] != "xhigh" and clear, before
                spot = tap_off(level, "bottom", 4, "/api/owner/effort-range")
                page.wait_for_function("""sel => document.querySelector(sel)
                    .querySelector('.chat-effort-seg[data-rec="true"]')?.dataset.effortSeg === 'xhigh'""", arg=EFFORT)
                seen["effort"].update(spot=spot, open=page.locator(EFFORT).get_attribute("data-open"))
                assert seen["effort"]["open"] == "true", seen["effort"]
        finally:
            (_evidence_dir(tmp_path) / f"subagents-edge-tap-{engine}-{width}.json").write_text(
                json.dumps(seen, indent=1), encoding="utf-8")
            page.unroute_all(behavior="ignoreErrors")
            browser.close()


# What holds the keyboard's focus: a value's mode, the head, the hint's Open Settings or another element.
FOCUSED = """() => { const a = document.activeElement;
    return a?.dataset?.mode || (a?.matches?.('.chat-subagents-head') ? 'head' : a?.matches?.('.ui-hint-action') ? 'settings'
        : (a?.tagName || '').toLowerCase()); }"""


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_the_keyboard_walks_subagents_on_a_320px_phone(direct_server_with_data, engine, tmp_path):  # noqa: F811
    """A 320 px phone with a keyboard: Enter on the head opens the strip onto the chosen value, an arrow chooses the next
    one and saves it, Escape closes it and gives the focus back to the head; a tap opens it and a tap outside closes it.
    With Settings › Agents off, Enter opens onto Off and Tab goes from the strip straight to the hint's Open Settings,
    whose Enter opens Settings › Agents. The capsule never moves while it opens."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    head = f"{SUBAGENTS} .chat-capsule-head"
    seen = {}
    with sync_playwright() as pw:
        browser, page = _launch(pw, engine, viewport={"width": 320, "height": 760}, is_mobile=True, has_touch=True)

        def is_open(state: str) -> None:
            page.wait_for_function("([sel, state]) => document.querySelector(sel)?.dataset.open === state",
                                   arg=[SUBAGENTS, state], timeout=10_000)
            _settle(page)

        try:
            held = _hold(page, {"subagents_mode": "auto", "subagents_availability": "available"})
            _open_chat(page, url, width=320, height=760, theme="dark")
            closed = page.locator(SUBAGENTS).evaluate("(el) => el.getBoundingClientRect().top")
            page.focus(head)
            page.keyboard.press("Enter")
            is_open("true")
            facts = seen["enter"] = {**page.evaluate(FACTS), "focused": page.evaluate(FOCUSED)}
            assert facts["focused"] == "auto" and facts["beside"] == [], ("onto the chosen value, alone on the line", facts)
            assert abs(closed - facts["capsule"]["top"]) <= 1 and facts["hint"]["bottom"] <= facts["capsule"]["top"] + 1, facts
            with page.expect_response(lambda r: r.url.endswith("/api/owner/subagents-mode"), timeout=30_000):
                page.keyboard.press("ArrowRight")
            page.wait_for_function("sel => document.querySelector(sel).dataset.mode === 'swarm'", arg=SUBAGENTS)
            seen["arrow"] = page.evaluate(FOCUSED)
            assert seen["arrow"] == "swarm", seen
            page.keyboard.press("Escape")
            is_open("false")
            seen["escape"] = page.evaluate(FOCUSED)
            assert seen["escape"] == "head", seen
            page.tap(head)
            is_open("true")
            page.tap("#chat-messages")
            is_open("false")
            # Settings › Agents off: the strip, then Open Settings, one Tab apart.
            held.update(subagents_availability="disabled")
            _open_chat(page, url, width=320, height=760, theme="dark")
            page.wait_for_function("sel => document.querySelector(sel)?.dataset.locked === 'true'", arg=SUBAGENTS,
                                   timeout=30_000)
            page.focus(head)
            page.keyboard.press("Enter")
            is_open("true")
            seen["locked-enter"] = page.evaluate(FOCUSED)
            assert seen["locked-enter"] == "off", seen
            # Safari's Tab reaches only fields unless "Press Tab to highlight each item" is on; Option+Tab reaches
            # every control, and WebKit here keeps that default.
            page.keyboard.press("Alt+Tab" if engine == "webkit" else "Tab")
            seen["locked-tab"] = page.evaluate(FOCUSED)
            _shot(page, tmp_path, f"subagents-keyboard-settings-{engine}-320")
            assert seen["locked-tab"] == "settings", ("Tab goes from the strip to Open Settings", seen)
            page.keyboard.press("Enter")
            page.wait_for_selector('.settings-panel.active[data-settings-panel="agents"]', state="visible",
                                   timeout=20_000)
        finally:
            (_evidence_dir(tmp_path) / f"subagents-keyboard-{engine}-320.json").write_text(
                json.dumps(seen, indent=1), encoding="utf-8")
            page.unroute_all(behavior="ignoreErrors")
            browser.close()
