"""Widgets grid browser smoke: the owner's fixed card place and size.

On chromium and webkit, with real module frames: a frame whose content grows
changes neither its card's size nor any card's cell; the move handle (keys and
a pointer drag) and the corner grip (keys) change cells and sizes that are
written to ``ui_preferences.widget_layout``; a window reload restores every
card at the same cell with the same pixel size; a narrow window stacks the
cards in ``widget_order`` at their saved heights and a wide one restores the
grid. Through all of it every card keeps its DOM node and every frame its
window (a moved or re-inserted <iframe> would reload). Kept apart from the
lifecycle / geometry suites so none of them grows past the size-ratchet band."""

from __future__ import annotations

import os
import pathlib
import textwrap

import pytest

from tests.test_ui_smoke_playwright import direct_server_with_data as _direct_server_with_data

direct_server_with_data = _direct_server_with_data

ROW_PX, GAP_PX = 40, 14


def _write_grid_widget_extension(data_dir: pathlib.Path) -> str:
    """Install two auto-started module frames (one auto-height with a Grow
    button, one fixed at 360px) and one declarative card."""
    from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_review_state

    name = "grid_widget_smoke"
    skill_dir = data_dir / "skills" / "external" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        textwrap.dedent(
            f"""\
            ---
            name: {name}
            description: Isolated widget grid arrangement fixture.
            version: 0.1.0
            type: extension
            entry: plugin.py
            permissions: ["widget"]
            ---
            # Widget grid fixture
            """
        ),
        encoding="utf-8",
    )
    (skill_dir / "plugin.py").write_text(
        textwrap.dedent(
            """\
            def register(api):
                api.register_ui_tab("grow", "Grow", render={"kind": "module", "entry": "grow.js", "start": "auto"})
                api.register_ui_tab("fixed", "Fixed", render={"kind": "module", "entry": "fixed.js", "height": 360, "start": "auto"})
                api.register_ui_tab(
                    "notes",
                    "Notes",
                    render={
                        "kind": "declarative",
                        "schema_version": 1,
                        "components": [{"type": "markdown", "text": "Grid notes"}],
                    },
                )
            """
        ),
        encoding="utf-8",
    )
    (skill_dir / "grow.js").write_text(
        textwrap.dedent(
            """\
            (() => {
                const root = document.getElementById('root');
                root.innerHTML = '<button id="grow" type="button">Grow</button><div id="rows"></div>';
                document.getElementById('grow').addEventListener('click', () => {
                    const rows = document.getElementById('rows');
                    for (let i = 0; i < 30; i += 1) {
                        const row = document.createElement('div');
                        row.style.height = '40px';
                        row.textContent = `Row ${i + 1}`;
                        rows.appendChild(row);
                    }
                });
            })();
            """
        ),
        encoding="utf-8",
    )
    (skill_dir / "fixed.js").write_text(
        "(() => { document.getElementById('root').textContent = 'Fixed'; })();\n",
        encoding="utf-8",
    )
    content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
    save_review_state(data_dir, name, SkillReviewState(status="pass", content_hash=content_hash))
    return name


@pytest.mark.ui_browser
@pytest.mark.parametrize("browser_name", ("chromium", "webkit"))
def test_ui_smoke_widget_grid_keeps_owner_place_and_size(direct_server_with_data, browser_name):
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    skill = _write_grid_widget_extension(data_dir)
    keys = [f"{skill}:{tab}" for tab in ("fixed", "grow", "notes")]
    framed = [f"{skill}:{tab}" for tab in ("fixed", "grow")]
    evidence_dir = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(data_dir.parent)))
    evidence_dir.mkdir(parents=True, exist_ok=True)

    def card(key: str) -> str:
        return f'[data-widget-key="{key}"]'

    def cells(page) -> dict:
        return page.evaluate(
            """(keys) => Object.fromEntries(keys.map((key) => {
                const node = document.querySelector(`[data-widget-key="${key}"]`);
                const read = (name) => Number(node.style.getPropertyValue(name));
                return [key, {x: read('--widget-col') - 1, y: read('--widget-row') - 1, w: read('--widget-w'), h: read('--widget-h')}];
            }))""",
            keys,
        )

    def rect(page, key: str) -> dict:
        return page.locator(card(key)).evaluate(
            "node => { const box = node.getBoundingClientRect(); return {x: box.x, y: box.y, width: box.width, height: box.height}; }"
        )

    def saved_layout(page) -> dict:
        return page.evaluate("async () => (await (await fetch('/api/ui/preferences')).json()).widget_layout || {}")

    def wait_saved(page, key: str, slot: dict) -> None:
        page.wait_for_function(
            """async ([key, slot]) => {
                const saved = ((await (await fetch('/api/ui/preferences')).json()).widget_layout || {})[key];
                return !!saved && ['x', 'y', 'w', 'h'].every((name) => saved[name] === slot[name]);
            }""",
            arg=[key, slot],
            timeout=5_000,
        )

    def mark_frames(page) -> None:
        for key in framed:
            page.locator(f"{card(key)} iframe").evaluate("frame => { frame.__gridMark = true; }")
            page.locator(f"{card(key)} iframe").element_handle().content_frame().evaluate("() => { window.__gridMark = 'same-window'; }")

    def frames_kept(page) -> bool:
        for key in framed:
            frame = page.locator(f"{card(key)} iframe")
            if frame.count() != 1 or not frame.evaluate("frame => frame.__gridMark === true"):
                return False
            if frame.element_handle().content_frame().evaluate("() => window.__gridMark ?? null") != "same-window":
                return False
        return True

    def node_order(page) -> list:
        return page.evaluate("() => [...document.querySelectorAll('#widgets-list [data-widget-key]')].map((node) => node.dataset.widgetKey)")

    def open_widgets(page) -> None:
        page.click('[data-nav-page="widgets"]')
        for key in framed:
            page.locator(f"{card(key)} iframe").wait_for(state="attached", timeout=30_000)
        page.wait_for_function(
            "(key) => document.querySelector(`[data-widget-key=\"${key}\"]`)?.style.getPropertyValue('--widget-col') !== ''",
            arg=keys[-1],
            timeout=10_000,
        )

    def press(page, key: str, handle: str, *presses: str) -> None:
        page.locator(f"{card(key)} [data-widget-{handle}-handle]").focus()
        for name in presses:
            page.keyboard.press(name)

    try:
        with sync_playwright() as pw:
            browser = getattr(pw, browser_name).launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                toggled = page.evaluate(
                    """async (skill) => (await fetch(`/api/skills/${encodeURIComponent(skill)}/toggle`, {
                        method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({enabled: true}),
                    })).status""",
                    skill,
                )
                assert toggled == 200
                open_widgets(page)
                assert page.evaluate("document.getElementById('widgets-list').dataset.widgetLayout") == "grid"
                mark_frames(page)
                dom_before = node_order(page)
                first = cells(page)
                for key, slot in first.items():
                    wait_saved(page, key, slot)
                    box = rect(page, key)
                    assert abs(box["height"] - (slot["h"] * (ROW_PX + GAP_PX) - GAP_PX)) <= 1, (key, slot, box)

                # Content growth: the frame grows inside its card, the card and every cell stay.
                grow = framed[1]
                card_before = rect(page, grow)
                frame_before = page.locator(f"{card(grow)} iframe").evaluate("frame => frame.getBoundingClientRect().height")
                page.frame_locator(f"{card(grow)} iframe").locator("#grow").click()
                page.wait_for_function(
                    "([selector, before]) => document.querySelector(`${selector} iframe`).getBoundingClientRect().height > before + 300",
                    arg=[card(grow), frame_before],
                    timeout=10_000,
                )
                page.wait_for_timeout(300)
                assert rect(page, grow) == card_before
                assert cells(page) == first
                assert all(saved_layout(page).get(key) == slot for key, slot in first.items()), "first display pins every card; growth does not change it"

                # Keyboard move and resize of the fixed card: one cell per key, saved, frames kept.
                fixed = framed[0]
                start = first[fixed]
                press(page, fixed, "move", "ArrowDown", "ArrowDown", "ArrowRight")
                moved = {**start, "x": min(start["x"] + 1, 12 - start["w"]), "y": start["y"] + 2}
                wait_saved(page, fixed, moved)
                width_before = rect(page, fixed)["width"]
                press(page, fixed, "resize", "ArrowRight", "ArrowDown")
                resized = {**moved, "w": min(moved["w"] + 1, 12 - moved["x"]), "h": moved["h"] + 1}
                wait_saved(page, fixed, resized)
                assert cells(page)[fixed] == resized
                assert rect(page, fixed)["width"] > width_before
                assert frames_kept(page)

                # Pointer drag of the grow card by its move handle: two columns right, one row
                # down (clear of the scroller's edge, where the drag would scroll it too).
                slot = cells(page)[grow]
                page.locator(f"{card(grow)} [data-widget-move-handle]").scroll_into_view_if_needed()
                handle = page.locator(f"{card(grow)} [data-widget-move-handle]").bounding_box()
                list_width = page.evaluate("document.getElementById('widgets-list').clientWidth")
                pitch = (list_width + GAP_PX) / 12
                x, y = handle["x"] + handle["width"] / 2, handle["y"] + handle["height"] / 2
                page.mouse.move(x, y)
                page.mouse.down()
                page.mouse.move(x + 2 * pitch, y + ROW_PX + GAP_PX, steps=8)
                page.mouse.up()
                dragged = {**slot, "x": min(slot["x"] + 2, 12 - slot["w"]), "y": slot["y"] + 1}
                wait_saved(page, grow, dragged)
                arranged = cells(page)
                assert arranged[grow] == dragged
                assert frames_kept(page)
                assert node_order(page) == dom_before, "an arrangement never moves a card node"
                pixels = {key: rect(page, key) for key in keys}
                page.screenshot(path=str(evidence_dir / f"widget-grid-{browser_name}.png"), full_page=True)

                # Revisit after a window reload: same cells, same pixel sizes.
                page.reload(wait_until="domcontentloaded", timeout=30_000)
                open_widgets(page)
                assert cells(page) == arranged
                assert {key: slot for key, slot in saved_layout(page).items() if key in keys} == arranged
                for key in keys:
                    after = rect(page, key)
                    assert abs(after["width"] - pixels[key]["width"]) <= 1, (key, after, pixels[key])
                    assert abs(after["height"] - pixels[key]["height"]) <= 1, (key, after, pixels[key])

                # Narrow: one stacked column in widget_order at the saved heights; the frames stay.
                mark_frames(page)
                page.set_viewport_size({"width": 600, "height": 900})
                page.wait_for_function("document.getElementById('widgets-list').dataset.widgetLayout === 'stack'", timeout=5_000)
                order = page.evaluate("async () => (await (await fetch('/api/ui/preferences')).json()).widget_order")
                tops = [rect(page, key)["y"] for key in order if key in keys]
                assert tops == sorted(tops), (order, tops)
                for key in keys:
                    assert abs(rect(page, key)["height"] - (arranged[key]["h"] * (ROW_PX + GAP_PX) - GAP_PX)) <= 1
                assert frames_kept(page)
                page.screenshot(path=str(evidence_dir / f"widget-grid-stack-{browser_name}.png"), full_page=True)

                page.set_viewport_size({"width": 1440, "height": 1000})
                page.wait_for_function("document.getElementById('widgets-list').dataset.widgetLayout === 'grid'", timeout=5_000)
                assert cells(page) == arranged
                assert frames_kept(page)
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise
