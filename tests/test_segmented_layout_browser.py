"""Every Settings segmented control on the real page: equal columns derived from its choices.

The one renderer (`page_header.renderSegmentedField`) feeds 13 call sites and 15
groups. On a real server in Chromium and WebKit, at a desktop and a phone width:
up to four choices fill their row; the seven- and eight-step effort scales keep four
equal columns with a partial last row whose buttons are not stretched; the five
one-glyph cycle values stay one row; a narrow card stacks two equal columns; no
label leaves its button. Then the keyboard: Enter and Space choose, focus wears the
one DESIGN ring (not the hover paint), the draft turns dirty, and Same as Task / Chat
— the empty value — survives Save and reload where `high` stood before.
"""

from __future__ import annotations

import json
import math
import os
import pathlib

import pytest

pytest_plugins = ("tests.test_ui_smoke_playwright",)

# Every group the renderer draws, with its number of choices.
GROUPS = {
    "s-allow-mutative-subagents": 3, "s-effort-task": 7, "s-effort-evolution": 7,
    "s-effort-consciousness": 8, "s-review-enforcement": 2,
    "s-task-review-mode": 3, "s-review-max-cycles": 5, "s-image-input-mode": 4, "s-context-mode": 3,
    "s-prompt-cache-ttl": 3, "s-safety-mode": 3, "s-update-channel": 3, "s-runtime-mode": 4,
    "s-post-task-evolution-mode": 3, "s-consciousness-autonomy": 3,
}
COMPACT = {"s-review-max-cycles"}

_MEASURE = """() => [...document.querySelectorAll('[data-effort-group]')]
  .filter((group) => group.getClientRects().length).map((group) => {
    const box = group.getBoundingClientRect();
    const buttons = [...group.querySelectorAll('[data-effort-value]')].map((button) => {
      const rect = button.getBoundingClientRect();
      const range = document.createRange();
      range.selectNodeContents(button);
      const text = range.getBoundingClientRect();
      return {left: rect.left, right: rect.right, top: Math.round(rect.top), width: rect.width,
              textLeft: text.left, textRight: text.right};
    });
    return {target: group.dataset.effortTarget, left: box.left, right: box.right, width: box.width, buttons};
  })"""


def _open(page, url: str, width: int) -> None:
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(url, wait_until="domcontentloaded")
    page.wait_for_selector("#page-chat", timeout=30_000)
    toggle = page.locator("[data-mobile-nav-toggle]:visible")
    if toggle.count():
        toggle.first.click()
    page.locator('[data-nav-page="settings"]').click()
    page.wait_for_selector(".settings-shell", timeout=20_000)


def _groups(page) -> dict:
    found = {}
    for tab in ("agents", "behavior"):
        page.locator(f'[data-settings-tab="{tab}"]').click()
        page.evaluate("() => new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done)))")
        found.update({group["target"]: group for group in page.evaluate(_MEASURE)})
    return found


def _check_layout(groups: dict, *, narrow: bool) -> list[str]:
    problems = []
    assert set(groups) == set(GROUPS), sorted(set(GROUPS) ^ set(groups))
    for target, group in groups.items():
        count = GROUPS[target]
        buttons = group["buttons"]
        columns = count if target in COMPACT else min(count, 2 if narrow else 4)
        rows = [top for top in sorted({button["top"] for button in buttons})]
        widths = [button["width"] for button in buttons]
        first_row = [button for button in buttons if button["top"] == rows[0]]
        if len(buttons) != count:
            problems.append(f"{target}: {len(buttons)} buttons, expected {count}")
        if len(rows) != math.ceil(count / columns) or len(first_row) != columns:
            problems.append(f"{target}: rows {len(rows)} x {len(first_row)}, expected {columns} columns")
        if max(widths) - min(widths) > 1:  # equal columns in every row, the partial last row included
            problems.append(f"{target}: unequal widths {widths}")
        if abs(first_row[0]["left"] - group["left"]) > 1.5 or abs(group["right"] - first_row[-1]["right"]) > 1.5:
            problems.append(f"{target}: the first row does not fill the card ({group['right'] - first_row[-1]['right']:.1f}px gap)")
        for button in buttons:
            if button["textLeft"] < button["left"] - 0.5 or button["textRight"] > button["right"] + 0.5:
                problems.append(f"{target}: a label leaves its button ({button})")
    return problems


@pytest.mark.serial
@pytest.mark.ui_browser
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_every_segmented_group_lays_out_from_its_own_choices(direct_server_with_data, engine):
    from playwright.sync_api import sync_playwright

    fixture = direct_server_with_data
    evidence = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(fixture["data_dir"].parent)))
    evidence.mkdir(parents=True, exist_ok=True)
    problems = []
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch(headless=True)
        try:
            page = browser.new_page()
            for width, narrow in ((1280, False), (390, True)):
                _open(page, fixture["url"], width)
                groups = _groups(page)
                problems += [f"{engine} {width}: {problem}" for problem in _check_layout(groups, narrow=narrow)]
                page.screenshot(path=str(evidence / f"segmented-{engine}-{width}-behavior.png"), full_page=True)
        finally:
            browser.close()
    assert not problems, "\n".join(problems)


@pytest.mark.serial
@pytest.mark.ui_browser
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_keyboard_choice_focus_ring_and_the_empty_inherit_value_survive_save(direct_server_with_data, engine):
    from playwright.sync_api import expect, sync_playwright

    fixture = direct_server_with_data
    settings_path = fixture["data_dir"] / "settings.json"
    group = '[data-effort-target="s-effort-consciousness"]'
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch(headless=True)
        try:
            page = browser.new_page()

            def open_behavior():
                _open(page, fixture["url"], 1280)
                page.locator('[data-settings-tab="behavior"]').click()
                expect(page.locator("#btn-save-settings")).to_be_enabled(timeout=30_000)

            def press(value: str, key: str) -> None:
                page.keyboard.press("Shift")  # keyboard modality: script focus then counts as keyboard focus
                page.locator(f'{group} [data-effort-value="{value}"]').focus()
                page.keyboard.press(key)

            def save() -> dict:
                page.locator("#btn-save-settings").click()
                expect(page.locator("#settings-status")).to_contain_text("Settings saved", timeout=30_000)
                return json.loads(settings_path.read_text(encoding="utf-8"))

            open_behavior()
            press("high", "Enter")
            assert page.locator("#s-effort-consciousness").input_value() == "high"
            expect(page.locator("#settings-unsaved-indicator")).to_have_class(__import__("re").compile("is-visible"))
            ring = page.evaluate("""(selector) => {
                const focused = document.activeElement;
                const probe = document.createElement('span');
                probe.style.color = 'var(--focus-accent-border)';
                document.body.append(probe);
                const token = getComputedStyle(probe).color;
                probe.remove();
                const style = getComputedStyle(focused);
                return {visible: focused.matches(':focus-visible'), value: focused.dataset.effortValue, token,
                        outline: [style.outlineStyle, style.outlineWidth, style.outlineOffset, style.outlineColor]};
            }""", group)
            assert ring["visible"] and ring["value"] == "high", ring
            assert ring["outline"] == ["solid", "2px", "2px", ring["token"]], ring
            # Hover paints the button; it never draws the focus ring.
            hovered = page.locator(f'{group} [data-effort-value="low"]')
            hovered.hover()
            assert hovered.evaluate("(node) => getComputedStyle(node).outlineStyle") == "none"
            assert save()["OUROBOROS_EFFORT_CONSCIOUSNESS"] == "high"

            open_behavior()
            assert page.locator(f'{group} [data-effort-value="high"]').get_attribute("aria-pressed") == "true"
            press("", " ")  # Space on Same as Task / Chat
            assert page.locator("#s-effort-consciousness").input_value() == ""
            assert save()["OUROBOROS_EFFORT_CONSCIOUSNESS"] == ""

            open_behavior()
            inherit = page.locator(f'{group} [data-effort-value=""]')
            assert inherit.get_attribute("aria-pressed") == "true" and "Same as Task" in inherit.inner_text()
            assert page.locator(f'{group} [data-effort-value="high"]').get_attribute("aria-pressed") == "false"
        finally:
            browser.close()
