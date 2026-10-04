"""Widgets board browser smoke: today's masonry with the owner's column spans.

On chromium and webkit, against a real server with real module frames and real
declarative cards of very different heights (a 720px game-like frame, a short
metric, a long route-backed table, an auto-height module): untouched, the
masonry packs the cards in the owner's order at their authors' spans (two
columns for `span: 2`, one otherwise); the card menu, the right-edge drag and
its arrow keys set a card's span — 1, 2, 3 columns or Full width, each named in
the live region — stored in ``ui_preferences.widget_size`` (Reset size deletes
one) that a window reload restores; a list too narrow for two columns is a
stack in the same order whose menu says widths apply when the list is wide.
Through all of it every card keeps its DOM node and every frame its window (a
moved or re-inserted <iframe> would reload). Screenshots of the board, the open
menu and the stack go to the evidence directory (docs/DESIGN.md "Widgets
board")."""

from __future__ import annotations

import os
import pathlib
import textwrap

import pytest

from tests.test_ui_smoke_playwright import direct_server_with_data as _direct_server_with_data

direct_server_with_data = _direct_server_with_data

GAP_PX = 14


def _write_board_widget_extension(data_dir: pathlib.Path) -> str:
    """Install five cards: a 720px game-like module frame (span 2), a short
    metric, a long route-backed table, an auto-height module with a Grow
    button and a short note."""
    from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_review_state

    name = "board_widget_smoke"
    skill_dir = data_dir / "skills" / "external" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        textwrap.dedent(
            f"""\
            ---
            name: {name}
            description: Isolated widgets board fixture.
            version: 0.1.0
            type: extension
            entry: plugin.py
            permissions: ["route", "widget"]
            ---
            # Widgets board fixture
            """
        ),
        encoding="utf-8",
    )
    (skill_dir / "plugin.py").write_text(
        textwrap.dedent(
            """\
            _ROWS = [
                {"issue": f"#{1300 + i}", "title": f"Board fixture row {i}", "state": "open" if i % 3 else "closed"}
                for i in range(30)
            ]


            async def rows(_request):
                return {"rows": _ROWS}


            def register(api):
                api.register_route("rows", handler=rows, methods=("GET",))
                api.register_ui_tab("game", "Game", render={
                    "kind": "module", "entry": "game.js", "height": 720, "start": "auto", "span": 2,
                })
                api.register_ui_tab("gauge", "Gauge", render={
                    "kind": "declarative", "schema_version": 1,
                    "components": [{"type": "metric", "label": "Used", "value": "62", "unit": "%"}],
                })
                api.register_ui_tab("grow", "Grow", render={"kind": "module", "entry": "grow.js", "start": "auto"})
                api.register_ui_tab("issues", "Issues", render={
                    "kind": "declarative", "schema_version": 1, "components": [
                        {"type": "poll", "label": "Load", "route": "rows", "auto_start": True, "max_ticks": 1},
                        {"type": "table", "path": "rows", "columns": [
                            {"label": "Issue", "path": "issue"}, {"label": "Title", "path": "title"},
                            {"label": "State", "path": "state"},
                        ]},
                    ],
                })
                api.register_ui_tab("notes", "Notes", render={
                    "kind": "declarative", "schema_version": 1,
                    "components": [{"type": "callout", "text": "A short note."}],
                })
            """
        ),
        encoding="utf-8",
    )
    (skill_dir / "game.js").write_text(
        "(() => { const root = document.getElementById('root'); root.style.height = '700px';"
        " root.style.background = 'linear-gradient(135deg, #1b5e3a, #0d2a3a)'; root.textContent = 'Game'; })();\n",
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
                    for (let i = 0; i < 20; i += 1) {
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
    content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
    save_review_state(data_dir, name, SkillReviewState(status="pass", content_hash=content_hash))
    return name


@pytest.mark.ui_browser
@pytest.mark.parametrize("browser_name", ("chromium", "webkit"))
def test_ui_smoke_widget_board_masonry_and_owner_widths(direct_server_with_data, browser_name):
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    skill = _write_board_widget_extension(data_dir)
    key = {tab: f"{skill}:{tab}" for tab in ("game", "gauge", "grow", "issues", "notes")}
    framed = [key["game"], key["grow"]]
    evidence_dir = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(data_dir.parent)))
    evidence_dir.mkdir(parents=True, exist_ok=True)

    def card(tab: str) -> str:
        return f'[data-widget-key="{key[tab]}"]'

    def rects(page) -> dict:
        # Relative to the list, so a scroll (a click into a frame scrolls it into
        # view) never reads as a card moving.
        return page.evaluate(
            """(keys) => {
                const list = document.getElementById('widgets-list').getBoundingClientRect();
                return Object.fromEntries(Object.entries(keys).map(([tab, key]) => {
                    const box = document.querySelector(`[data-widget-key="${key}"]`).getBoundingClientRect();
                    return [tab, {x: box.x - list.x, y: box.y - list.y, width: box.width, height: box.height}];
                }));
            }""",
            key,
        )

    def list_width(page) -> float:
        return page.evaluate("document.getElementById('widgets-list').getBoundingClientRect().width")

    def layout(page) -> str:
        return page.evaluate("document.getElementById('widgets-list').dataset.widgetLayout || ''")

    def width_fixed(page) -> list:
        # The cards no step can widen or narrow; their edge handle is hidden.
        return page.evaluate(
            "() => [...document.querySelectorAll('#widgets-list .widgets-card[data-widget-width-fixed]')].map((node) => node.dataset.widgetKey)"
        )

    def status(page) -> str:
        return page.locator("[data-widget-arrange-status]").text_content()

    def saved_sizes(page) -> dict:
        return page.evaluate("async () => (await (await fetch('/api/ui/preferences')).json()).widget_size || {}")

    def wait_saved(page, tab: str, size) -> None:
        page.wait_for_function(
            """async ([key, size]) => {
                const saved = ((await (await fetch('/api/ui/preferences')).json()).widget_size || {})[key];
                return size === null ? saved === undefined : (!!saved && saved.w === size.w && saved.h === 0);
            }""",
            arg=[key[tab], size],
            timeout=5_000,
        )

    def wait_width(page, tab: str, width: float) -> None:
        page.wait_for_function(
            "([selector, width]) => Math.abs(document.querySelector(selector).getBoundingClientRect().width - width) <= 1.5",
            arg=[card(tab), width],
            timeout=5_000,
        )

    def mark_frames(page) -> None:
        for frame_key in framed:
            frame = page.locator(f'[data-widget-key="{frame_key}"] iframe')
            frame.evaluate("frame => { frame.__boardMark = true; }")
            frame.element_handle().content_frame().evaluate("() => { window.__boardMark = 'same-window'; }")

    def frames_kept(page) -> bool:
        for frame_key in framed:
            frame = page.locator(f'[data-widget-key="{frame_key}"] iframe')
            if frame.count() != 1 or not frame.evaluate("frame => frame.__boardMark === true"):
                return False
            if frame.element_handle().content_frame().evaluate("() => window.__boardMark ?? null") != "same-window":
                return False
        return True

    def node_order(page) -> list:
        return page.evaluate("() => [...document.querySelectorAll('#widgets-list [data-widget-key]')].map((node) => node.dataset.widgetKey)")

    def open_widgets(page) -> None:
        page.click('[data-nav-page="widgets"]')
        for frame_key in framed:
            page.locator(f'[data-widget-key="{frame_key}"] iframe').wait_for(state="attached", timeout=30_000)
        page.locator(f"{card('issues')} table tbody tr").nth(29).wait_for(state="attached", timeout=15_000)

    def choose(page, tab: str, item: str) -> None:
        page.locator(f"{card(tab)} [data-widget-menu-trigger]").click()
        page.locator(f'body > .skills-card-menu-dialog[open] [data-widget-size="{item}"]').click()

    def same(a: float, b: float) -> bool:
        return abs(a - b) <= 1.5

    try:
        with sync_playwright() as pw:
            browser = getattr(pw, browser_name).launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 800})
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
                page.wait_for_function("document.getElementById('widgets-list').dataset.widgetLayout === 'columns'", timeout=5_000)
                mark_frames(page)
                dom_before = node_order(page)

                # Untouched: today's masonry at the authors' spans. Six asked-for columns on a
                # list with room for three: the game spans two, every other card one.
                box = rects(page)
                column = box["gauge"]["width"]
                assert same(column, (list_width(page) + GAP_PX) / 3 - GAP_PX)
                assert same(box["game"]["width"], 2 * column + GAP_PX)
                for tab in ("grow", "issues", "notes"):
                    assert same(box[tab]["width"], column), tab
                # Cards keep their own content height; the masonry stacks them by it.
                assert box["issues"]["height"] > box["notes"]["height"] + 400
                assert box["gauge"]["height"] < box["game"]["height"] - 400
                assert page.evaluate("document.getElementById('widgets-list').style.getPropertyValue('--masonry-h')")
                assert width_fixed(page) == [], "on a board of five cards a step can change every card"
                page.screenshot(path=str(evidence_dir / f"widget-board-{browser_name}.png"), full_page=True)

                # Content growth relayouts the masonry; the grown frame keeps its window.
                frame_before = page.locator(f"{card('grow')} iframe").evaluate("frame => frame.getBoundingClientRect().height")
                page.frame_locator(f"{card('grow')} iframe").locator("#grow").click()
                page.wait_for_function(
                    "([selector, before]) => document.querySelector(`${selector} iframe`).getBoundingClientRect().height > before + 300",
                    arg=[card("grow"), frame_before],
                    timeout=10_000,
                )
                assert rects(page)["grow"]["height"] > box["grow"]["height"] + 300
                assert frames_kept(page)

                # The card menu: one column is checked; Full width spans every column.
                page.locator(f"{card('gauge')} [data-widget-menu-trigger]").click()
                menu = page.locator("body > .skills-card-menu-dialog[open]")
                menu.wait_for()
                assert menu.locator('[data-widget-size="1"]').get_attribute("aria-checked") == "true"
                assert menu.locator("[data-widget-size-note]").is_hidden()
                page.screenshot(path=str(evidence_dir / f"widget-board-menu-{browser_name}.png"))
                menu.locator('[data-widget-size="12"]').click()
                wait_saved(page, "gauge", {"w": 12})
                wait_width(page, "gauge", list_width(page))
                assert status(page) == "Width: full width", "a menu choice is named like a key or a drag"

                # The right edge drags in the board's column pitch: one column wider is two.
                page.evaluate("(selector) => document.querySelector(selector).scrollIntoView({block: 'start'})", card("issues"))
                handle = page.locator(f"{card('issues')} [data-widget-resize-handle]").bounding_box()
                x, y = handle["x"] + handle["width"] / 2, handle["y"] + 60
                page.mouse.move(x, y)
                page.mouse.down()
                page.mouse.move(x + column + GAP_PX, y, steps=8)
                page.mouse.up()
                wait_saved(page, "issues", {"w": 2})
                wait_width(page, "issues", 2 * column + GAP_PX)
                assert status(page) == "Width: 2 columns"

                # Its arrow keys step 1 -> 2 -> 3 -> Full width, Home back to one column.
                page.locator(f"{card('notes')} [data-widget-resize-handle]").focus()
                page.keyboard.press("ArrowRight")
                wait_saved(page, "notes", {"w": 2})
                assert status(page) == "Width: 2 columns"
                page.keyboard.press("ArrowRight")
                wait_saved(page, "notes", {"w": 3})
                page.keyboard.press("End")
                wait_saved(page, "notes", {"w": 12})
                assert status(page) == "Width: full width"
                page.keyboard.press("Home")
                wait_saved(page, "notes", {"w": 1})
                assert status(page) == "Width: 1 column"

                # Reset size returns the gauge to its author's one column.
                choose(page, "gauge", "reset")
                wait_saved(page, "gauge", None)
                wait_width(page, "gauge", column)
                assert frames_kept(page)
                assert node_order(page) == dom_before, "arranging never moves a card node"
                pixels = rects(page)

                # A window reload reads the same widths back from the server.
                page.reload(wait_until="domcontentloaded", timeout=30_000)
                open_widgets(page)
                assert saved_sizes(page) == {key["issues"]: {"w": 2, "h": 0}, key["notes"]: {"w": 1, "h": 0}}
                wait_width(page, "issues", pixels["issues"]["width"])
                for tab, item in rects(page).items():
                    assert same(item["width"], pixels[tab]["width"]), tab

                # Narrow: a stack in the same order, no edge handle, the menu says widths
                # apply when the list is wide; a width chosen there is stored, not shown.
                mark_frames(page)
                page.set_viewport_size({"width": 600, "height": 900})
                page.wait_for_function("document.getElementById('widgets-list').dataset.widgetLayout === 'stack'", timeout=5_000)
                stacked = rects(page)
                tops = [stacked[tab]["y"] for tab in ("game", "gauge", "grow", "issues", "notes")]
                assert tops == sorted(tops), tops
                assert all(same(item["width"], list_width(page)) for item in stacked.values())
                assert page.locator(f"{card('notes')} [data-widget-resize-handle]").is_hidden()
                assert sorted(width_fixed(page)) == sorted(key.values()), "in a stack no step changes a card"
                page.locator(f"{card('gauge')} [data-widget-menu-trigger]").click()
                page.locator("body > .skills-card-menu-dialog[open] [data-widget-size-note]").wait_for(state="visible")
                page.screenshot(path=str(evidence_dir / f"widget-board-stack-menu-{browser_name}.png"))
                page.locator('body > .skills-card-menu-dialog[open] [data-widget-size="2"]').click()
                wait_saved(page, "gauge", {"w": 2})
                assert status(page) == "Width: 2 columns", "on a narrow list the menu is the only path"
                assert same(rects(page)["gauge"]["width"], list_width(page))
                assert frames_kept(page)

                page.set_viewport_size({"width": 1280, "height": 800})
                page.wait_for_function("document.getElementById('widgets-list').dataset.widgetLayout === 'columns'", timeout=5_000)
                wide = rects(page)
                assert same(wide["gauge"]["width"], 2 * wide["notes"]["width"] + GAP_PX)
                assert frames_kept(page)
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise
