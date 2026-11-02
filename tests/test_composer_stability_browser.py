"""Nothing the composer opens moves the composer (docs/DESIGN.md "Hints"; bible principle 6: a state change never moves
a neighbour, resizes a bar or changes a height).

In Chromium and WebKit, on a desk and on a 390 px phone, every control in the composer that opens something (each
visible `[aria-expanded]`; ☰ opens the navigation, not the composer) is pressed in turn. Opening it, choosing another
value in it and closing it leave the field, the row and the pressed control where they stood, so nothing jumps under
the pointer or the finger. The composer is anchored to the bottom, so a line added under the row would lift the row:
what a control says about itself rides the one hint over the page instead. The room's Subagents are once available
and once off in Settings, since each state says its own words.
"""
from __future__ import annotations

import json

import pytest

from tests.test_composer_subagents_browser import _hold, _settle
from tests.test_effort_range_browser import _evidence_dir, _launch, _open_chat
from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401 - pytest fixture import

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

OPENERS = "#page-chat .chat-input-wrap [aria-expanded]:not([data-mobile-nav-toggle])"
# The tops of the field, the row and every control that opens something (null where it stepped aside), and its name.
WHERE = """(selector) => {
    const wrap = document.querySelector('#page-chat .chat-input-wrap');
    const top = (node) => (node && node.getClientRects().length ? Math.round(node.getBoundingClientRect().top * 10) / 10 : null);
    const name = (node) => node.id || ['subagents', 'context', 'effort'].find((k) => node.closest(`.chat-${k}-mode, .chat-${k}-range`))
        || node.getAttribute('aria-label') || node.className;
    return { field: top(wrap.querySelector('.chat-field')), row: top(wrap.querySelector('.chat-toolbar-row')),
        openers: [...document.querySelectorAll(selector)].map((node) => [name(node), top(node), node.getAttribute('aria-expanded')]) };
}"""


def _still(rest: dict, now: dict, i: int, step: str) -> None:
    moved = {part: (rest[part], now[part]) for part in ("field", "row") if abs((now[part] or 0) - (rest[part] or 0)) > 1}
    name, before, _ = rest["openers"][i]
    after = now["openers"][i][1]
    if after is not None and before is not None and abs(after - before) > 1:
        moved[name] = (before, after)
    assert not moved, (f"{step}: {name} moved the composer", moved)


@pytest.mark.parametrize("engine,width", [("chromium", 1280), ("webkit", 1280), ("chromium", 390), ("webkit", 390)])
def test_nothing_the_composer_opens_moves_the_composer(direct_server_with_data, engine, width, tmp_path):  # noqa: F811
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

        def press(locator) -> None:
            (locator.tap if phone else locator.click)()

        def expanded(i: int, state: str) -> None:
            page.wait_for_function("([sel, i, state]) => document.querySelectorAll(sel)[i]?.getAttribute('aria-expanded') === state",
                                   arg=[OPENERS, i, state], timeout=10_000)
            _settle(page)

        try:
            held = _hold(page, {"subagents_mode": "auto"})
            for availability in ("available", "disabled"):
                held.update(subagents_availability=availability)
                _open_chat(page, url, width=width, height=860, theme="dark")
                page.wait_for_function("() => document.querySelector('#page-chat .chat-subagents-mode')?.dataset.known === 'true'",
                                       timeout=30_000)
                _settle(page)
                rest = page.evaluate(WHERE, OPENERS)
                names = [name for name, _, _ in rest["openers"]]
                assert {"subagents", "context", "effort", "chat-plus"} <= set(names), rest
                for i, name in enumerate(names):
                    opener = page.locator(OPENERS).nth(i)
                    if not opener.is_visible():
                        continue
                    press(opener)
                    expanded(i, "true")
                    steps = {"open": page.evaluate(WHERE, OPENERS)}
                    _still(rest, steps["open"], i, f"{availability}, opening {name}")
                    other = page.locator(f'#page-chat .chat-{name}-mode .chat-seg[aria-checked="false"]:not([data-disabled="true"])')
                    if name in ("subagents", "context") and other.count():
                        with page.expect_response(lambda r: "/api/owner/" in r.url, timeout=30_000):
                            press(other.first)
                        _settle(page)
                        steps["choose"] = page.evaluate(WHERE, OPENERS)
                        _still(rest, steps["choose"], i, f"{availability}, choosing in {name}")
                    page.keyboard.press("Escape")
                    expanded(i, "false")
                    steps["closed"] = page.evaluate(WHERE, OPENERS)
                    _still(rest, steps["closed"], i, f"{availability}, closing {name}")
                    seen[f"{availability}-{name}"] = {"rest": rest, **steps}
        finally:
            (_evidence_dir(tmp_path) / f"composer-stability-{engine}-{width}.json").write_text(
                json.dumps(seen, indent=1), encoding="utf-8")
            page.unroute_all(behavior="ignoreErrors")
            browser.close()
