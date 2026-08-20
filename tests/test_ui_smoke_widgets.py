"""Declarative widgets and the settings they round-trip through.

Split verbatim out of ``tests/test_ui_smoke_playwright.py`` by theme. This module owns
the phase-3 declarative widget extension the smoke installs and drives, the owner
context-mode and scope-review acknowledgement surfaces, and the subagent-depth setting
that must survive a round trip through the UI.

Every test here launches a real browser and is marked ``ui_browser``, so the default
local run deselects the whole module.
"""

from __future__ import annotations

import json
import os
import pathlib
import textwrap
import urllib.request

import pytest


from tests._ui_smoke_shared import direct_server_with_data as _direct_server_with_data

# Fixtures are requested by name as test parameters, so they are re-bound through a
# module attribute: a direct import of a name that reappears as a parameter is an F811
# redefinition under the CI ruff gate.
direct_server_with_data = _direct_server_with_data


def _write_phase3_widget_smoke_extension(data_dir: pathlib.Path) -> str:
    """Install an exact-hash reviewed extension for the real Widgets flow."""
    from ouroboros.skill_loader import (
        SkillReviewState,
        compute_content_hash,
        save_review_state,
    )

    name = "phase3_widget_smoke"
    skill_dir = data_dir / "skills" / "external" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        textwrap.dedent(
            f"""\
            ---
            name: {name}
            description: Isolated declarative widget visual smoke.
            version: 0.1.0
            type: extension
            entry: plugin.py
            permissions: ["route", "widget", "ws_handler"]
            ---
            # Phase 3 widget smoke
            """
        ),
        encoding="utf-8",
    )
    (skill_dir / "plugin.py").write_text(
        textwrap.dedent(
            """\
            import asyncio


            _STATE = {
                "metric": 87.5,
                "message": "The nested form completed successfully.",
                "rows": [
                    {"label": "Safe reference", "url": "https://example.com/report", "status": "ready"},
                    {"label": "Unsafe reference", "url": "javascript:alert(1)", "status": "blocked"},
                ],
                "chart": {
                    "labels": ["Warm", "Gap", "Hot"],
                    "datasets": [{"label": "Hit rate", "data": [74, None, 91]}],
                },
                "long_json": {"token": "x" * 600},
                "cards": [
                    {"id": "card-1", "label": "Inspect visual evidence", "column": "todo"},
                    {"id": "card-2", "label": "Ship reviewed flow", "column": "done"},
                ],
            }


            def _snapshot():
                return {
                    **_STATE,
                    "rows": [dict(row) for row in _STATE["rows"]],
                    "chart": {
                        "labels": list(_STATE["chart"]["labels"]),
                        "datasets": [
                            {"label": item["label"], "data": list(item["data"])}
                            for item in _STATE["chart"]["datasets"]
                        ],
                    },
                    "cards": [dict(card) for card in _STATE["cards"]],
                }


            async def submit(request):
                body = await request.json()
                await asyncio.sleep(0.25)
                _STATE["message"] = f"Submitted {body.get('query') or 'request'} safely."
                return _snapshot()


            async def move(request):
                body = await request.json()
                await asyncio.sleep(0.15)
                for card in _STATE["cards"]:
                    if card["id"] == body.get("card_id"):
                        card["column"] = body.get("column_id")
                return _snapshot()


            async def save(request):
                body = await request.json()
                await asyncio.sleep(0.2)
                return {"message": f"Saved {body.get('mode') or 'safe'} mode."}


            async def tick(request):
                await asyncio.sleep(0.4)
                data = _STATE["chart"]["datasets"][0]["data"]
                data[0] = (data[0] or 0) + 1
                return _snapshot()


            def register(api):
                async def emit_live(_request):
                    api.send_ws_message("live", {
                        "count": "42.25",
                        "progress": 67,
                        "label": "streaming",
                        "state": "healthy",
                        "unknown": {"nested": True},
                        "nonfinite": "1e999",
                        "image_src": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
                        "gallery_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
                        "file_src": "/api/extensions/phase3_widget_smoke/live-file?name=report",
                    })
                    return {"sent": True}

                api.register_route("submit", submit, methods=("POST",))
                api.register_route("move", move, methods=("POST",))
                api.register_route("save", save, methods=("POST",))
                api.register_route("tick", tick, methods=("POST",))
                api.register_route("emit-live", emit_live, methods=("POST",))
                api.register_ui_tab(
                    "main",
                    "Phase 3 design system",
                    render={
                        "kind": "declarative",
                        "schema_version": 1,
                        "components": [
                            {
                                "type": "group",
                                "id": "operations",
                                "title": "Operations overview",
                                "description": "Host-owned composition with stable nested identity.",
                                "layout": "grid",
                                "columns": 2,
                                "components": [
                                    {"type": "metric", "id": "hit-rate", "label": "Cache hit", "path": "metric", "unit": "%", "precision": 1, "tone": "success"},
                                    {"type": "callout", "id": "result-callout", "path": "message", "tone": "warning"},
                                    {
                                        "type": "tabs",
                                        "id": "flows",
                                        "tabs": [
                                            {
                                                "label": "Submit",
                                                "components": [
                                                    {
                                                        "type": "form",
                                                        "id": "query-form",
                                                        "title": "Nested request",
                                                        "route": "submit",
                                                        "method": "POST",
                                                        "columns": 2,
                                                        "submit_label": "Run request",
                                                        "busy_label": "Running…",
                                                        "fields": [
                                                            {"name": "query", "label": "Query", "placeholder": "Ada", "help": "Rendered and escaped by the host.", "span": 2, "required": True},
                                                            {"name": "limit", "label": "Limit", "type": "number", "min": 1, "max": 10, "step": 1, "default": 3},
                                                            {"name": "secret", "label": "Ephemeral secret", "type": "password", "placeholder": "not persisted"},
                                                        ],
                                                    }
                                                ],
                                            },
                                            {
                                                "label": "Data",
                                                "components": [
                                                    {
                                                        "type": "table",
                                                        "id": "result-table",
                                                        "path": "rows",
                                                        "columns": [
                                                            {"label": "Reference", "path": "label", "presentation": "link", "href_path": "url"},
                                                            {"label": "Status", "path": "status", "presentation": "status"},
                                                        ],
                                                    },
                                                    {"type": "chart", "id": "gap-chart", "path": "chart", "chart_type": "line", "unit": "%", "aria_label": "Cache hit rate with an intentional gap"},
                                    {"type": "status", "id": "poll-status", "loading": "Loading data"},
                                    {"type": "poll", "id": "chart-poll", "route": "tick", "method": "POST", "interval_ms": 1000, "max_ticks": 3, "label": "Refresh chart"},
                                                ],
                                            },
                                        ],
                                    },
                                ],
                            },
                            {
                                "type": "subscription",
                                "id": "live-subscription",
                                "event": "live",
                                "target": "live",
                                "render": [
                                    {
                                        "type": "group",
                                        "id": "live-group",
                                        "title": "Live telemetry",
                                        "layout": "grid",
                                        "columns": 2,
                                        "components": [
                                            {"type": "metric", "id": "live-direct-count", "label": "Direct count", "path": "count", "precision": 1},
                                            {
                                                "type": "tabs",
                                                "id": "live-tabs",
                                                "tabs": [
                                                    {
                                                        "label": "Stream",
                                                        "components": [
                                                            {"type": "metric", "id": "live-tab-count", "label": "Tab count", "path": "count", "precision": 1},
                                                            {"type": "metric", "id": "live-state", "label": "State", "path": "state"},
                                                            {"type": "metric", "id": "live-unknown", "label": "Unknown", "path": "unknown"},
                                                            {"type": "metric", "id": "live-nonfinite", "label": "Non-finite", "path": "nonfinite"},
                                                            {"type": "progress", "id": "live-progress", "path": "progress", "label_key": "label"},
                                                        ],
                                                    },
                                                    {
                                                        "label": "Media",
                                                        "components": [
                                                            {"type": "image", "id": "live-image", "path": "image_src", "label": "Live image", "alt": "Live image"},
                                                            {"type": "file", "id": "live-file", "path": "file_src", "label": "Live file", "filename": "live-report.txt"},
                                                            {
                                                                "type": "gallery",
                                                                "id": "live-gallery",
                                                                "items": [
                                                                    {"type": "image", "path": "gallery_image", "label": "Gallery image", "alt": "Gallery image"}
                                                                ],
                                                            },
                                                        ],
                                                    },
                                                ],
                                            },
                                        ],
                                    }
                                ],
                            },
                            {
                                "type": "markdown",
                                "id": "notes",
                                "text": "### Notes\\n\\n- first bullet with an unbroken token abcdefghijklmnopqrstuvwxyz0123456789abcdefghijklmnopqrstuvwxyz0123456789\\n- second bullet\\n\\n1. ordered item one\\n2. ordered item two",
                            },
                            {"type": "json", "id": "long-json", "path": "long_json", "label": "Long JSON"},
                            {
                                "type": "kanban",
                                "id": "delivery-board",
                                "path": "cards",
                                "columns": [
                                    {"id": "todo", "label": "To do"},
                                    {"id": "done", "label": "Done"},
                                ],
                                "on_move": {"route": "move", "method": "POST"},
                            },
                        ],
                    },
                )
                api.register_settings_section(
                    "config",
                    "Phase 3 settings",
                    schema={
                        "components": [
                            {
                                "type": "form",
                                "id": "settings-form",
                                "route": "save",
                                "method": "POST",
                                "submit_label": "Save mode",
                                "busy_label": "Saving mode…",
                                "fields": [
                                    {"name": "mode", "label": "Mode", "type": "select", "default": "safe", "options": [{"label": "Safe", "value": "safe"}, {"label": "Fast", "value": "fast"}]},
                                    {"name": "token", "label": "Temporary token", "type": "password", "placeholder": "not persisted"},
                                ],
                            }
                        ]
                    },
                )
            """
        ),
        encoding="utf-8",
    )
    content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
    save_review_state(
        data_dir,
        name,
        SkillReviewState(status="pass", content_hash=content_hash),
    )
    return name

@pytest.mark.ui_browser
def test_ui_smoke_phase3_declarative_widgets_and_settings(direct_server_with_data):
    """Exercise the real reviewed-extension consumer flow for schema v1."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    skill = _write_phase3_widget_smoke_extension(data_dir)
    evidence_dir = pathlib.Path(
        os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(data_dir.parent))
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                toggled = page.evaluate(
                    """async (skill) => {
                        const response = await fetch(`/api/skills/${encodeURIComponent(skill)}/toggle`, {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({enabled: true}),
                        });
                        return {status: response.status, body: await response.json()};
                    }""",
                    skill,
                )
                assert toggled["status"] == 200, toggled
                assert toggled["body"].get("enabled") is True, toggled

                page.click('[data-nav-page="widgets"]')
                card = page.locator(f'[data-widget-key="{skill}:main"]')
                card.wait_for(state="visible", timeout=30_000)
                assert "Operations overview" in card.inner_text()
                cache_metric = card.locator('.widget-metric').filter(has_text="Cache hit")
                assert cache_metric.locator('strong').inner_text() == "—"

                form = card.locator('[data-widget-form="id:query-form"]')
                form.locator('input[name="query"]').fill("Ada")
                form.locator('input[name="limit"]').fill("4")
                form.locator('input[name="secret"]').fill("ephemeral")
                submit = form.locator('button[type="submit"]')
                submit.click()
                page.wait_for_function(
                    "() => document.querySelector('[data-widget-form=\"id:query-form\"] button')?.disabled === true"
                )
                assert submit.inner_text() == "Running…"
                page.wait_for_function(
                    "() => document.querySelector('[data-widget-form=\"id:query-form\"] button')?.disabled === false",
                    timeout=10_000,
                )
                assert "Submitted Ada safely." in card.inner_text()
                assert "87.5 %" in cache_metric.locator('strong').inner_text()
                metric = cache_metric
                callout = card.locator('.widget-callout')
                assert metric.get_attribute("data-tone") == "ok"
                assert callout.get_attribute("data-tone") == "warn"
                assert metric.evaluate("element => getComputedStyle(element).borderLeftColor") == "rgb(52, 211, 153)"
                assert callout.evaluate("element => getComputedStyle(element).borderLeftColor") == "rgb(251, 191, 36)"

                emitted = page.evaluate(
                    """async (skill) => {
                        const response = await fetch(`/api/extensions/${encodeURIComponent(skill)}/emit-live`, {method: 'POST'});
                        return {status: response.status, body: await response.json()};
                    }""",
                    skill,
                )
                assert emitted == {"status": 200, "body": {"sent": True}}
                page.wait_for_function(
                    """() => {
                        const live = document.querySelector('.widget-subscription-render');
                        if (!live) return false;
                        const metrics = [...live.querySelectorAll('.widget-metric')];
                        const value = (label) => metrics.find((item) => item.querySelector('span')?.textContent === label)?.querySelector('strong')?.textContent.trim();
                        return value('Direct count') === '42.3'
                            && value('Tab count') === '42.3'
                            && value('State') === 'healthy'
                            && value('Unknown') === '—'
                            && value('Non-finite') === '—'
                            && live.querySelector('.widget-progress span')?.textContent.includes('67% · streaming');
                    }""",
                    timeout=10_000,
                )
                live = card.locator('.widget-subscription-render')
                live.get_by_role("button", name="Media").click()
                live_image = live.locator('img[alt="Live image"]')
                gallery_image = live.locator('img[alt="Gallery image"]')
                live_image.wait_for(state="visible", timeout=5_000)
                gallery_image.wait_for(state="visible", timeout=5_000)
                assert live_image.get_attribute("src").startswith("data:image/png;base64,")
                assert gallery_image.get_attribute("src").startswith("data:image/png;base64,")
                assert live.get_by_role("button", name="Live file").get_attribute("data-widget-download-url") == "/api/extensions/phase3_widget_smoke/live-file?name=report"
                live.get_by_role("button", name="Stream").click()

                card.get_by_role("button", name="Data").click()
                chart = card.locator('[data-widget-chart-key="id:gap-chart"]')
                chart.wait_for(state="visible", timeout=5_000)
                canvas_box = chart.bounding_box()
                assert canvas_box and 250 <= canvas_box["height"] <= 370, canvas_box
                chart_config = json.loads(chart.get_attribute("data-widget-chart-config"))
                assert chart_config["data"]["datasets"][0]["data"] == [74, None, 91]
                assert chart_config["data"]["datasets"][0]["spanGaps"] is False
                assert chart_config["options"]["spanGaps"] is False
                assert chart.get_attribute("aria-label") == "Cache hit rate with an intentional gap"
                # Poll refetch adopts the live canvas and preserves stale content.
                chart.evaluate("el => { el.__adoptMarker = 42; }")
                first_point = chart_config["data"]["datasets"][0]["data"][0]
                card.get_by_role("button", name="Refresh chart").click()
                page.wait_for_function(
                    """() => document.querySelector('.widget-status')?.dataset.state === 'refreshing'""",
                    timeout=5_000,
                )
                assert card.locator('.widget-status').inner_text() == "Loading data"  # declared loading label reused
                assert card.locator('canvas[data-widget-chart-key="id:gap-chart"]').count() == 1  # content kept during refetch
                # The fixture declares max_ticks=3. Wait for its final value so
                # no scheduled renderAll can detach the geometry probes below.
                page.wait_for_function(
                    """(prev) => {
                        const el = document.querySelector('canvas[data-widget-chart-key="id:gap-chart"]');
                        if (!el) return false;
                        const cfg = JSON.parse(el.dataset.widgetChartConfig || '{}');
                        return cfg.data?.datasets?.[0]?.data?.[0] >= prev + 3;
                    }""",
                    arg=first_point,
                    timeout=10_000,
                )
                assert chart.evaluate("el => el.__adoptMarker") == 42  # SAME canvas node — adopted, not recreated
                page.wait_for_function(
                    """() => document.querySelector('.widget-status')?.dataset.state === 'success'""",
                    timeout=10_000,
                )

                table = card.locator('.widget-chart-data table')
                table_text = table.text_content() or ""
                assert "Gap" in table_text
                assert "—" in table_text
                unsafe_row = card.locator('.widget-table tbody tr').filter(has_text="Unsafe reference")
                assert unsafe_row.locator('a').count() == 0

                page.evaluate("window.Chart = undefined")
                card.get_by_role("button", name="Submit").click()
                card.get_by_role("button", name="Data").click()
                fallback = card.locator('.widget-chart-fallback')
                fallback.wait_for(state="visible", timeout=5_000)
                assert fallback.locator('canvas').count() == 0
                assert fallback.locator('details[open] table').count() == 1

                move = card.locator('[data-widget-kanban-card="card-1"] [data-widget-kanban-move]')
                move.select_option("done")
                page.wait_for_function(
                    """() => Boolean(
                        document.querySelector('[data-widget-kanban-col="done"] [data-widget-kanban-card="card-1"]')
                        || document.querySelector('.widget-kanban .widget-status[data-state="error"]')
                    )""",
                    timeout=10_000,
                )
                assert card.locator(
                    '[data-widget-kanban-col="done"] [data-widget-kanban-card="card-1"]'
                ).count() == 1, card.inner_text()
                long_json = card.locator('.widget-json').filter(has_text="Long JSON")
                long_json.locator('summary').click()
                json_pre = long_json.locator('pre')
                json_pre.wait_for(state="visible", timeout=5_000)
                assert json_pre.evaluate(
                    "element => getComputedStyle(element).maxHeight"
                ) == "360px"

                list_box = page.locator('#widgets-list').bounding_box()
                card_box = card.bounding_box()
                operations_group = card.locator('.widget-group').filter(has_text="Operations overview")
                group_box = operations_group.bounding_box()
                tabs_box = operations_group.locator('.widget-tabs').bounding_box()
                kanban_columns = card.locator('.widget-kanban-col')
                todo_box = kanban_columns.nth(0).bounding_box()
                done_box = kanban_columns.nth(1).bounding_box()
                assert list_box and card_box and group_box and tabs_box
                assert todo_box and done_box
                assert card_box["width"] >= list_box["width"] * 0.9
                markdown_block = card.locator('.widget-markdown.ui-rich-content')
                assert markdown_block.locator('li').count() >= 4
                first_li_box = markdown_block.locator('li').first.bounding_box()
                assert first_li_box and card_box
                assert first_li_box["x"] >= card_box["x"]
                assert first_li_box["x"] + first_li_box["width"] <= card_box["x"] + card_box["width"] + 1
                assert tabs_box["width"] >= group_box["width"] * 0.9
                assert abs(todo_box["y"] - done_box["y"]) < 2
                assert done_box["x"] > todo_box["x"] + todo_box["width"]
                page.screenshot(
                    path=str(evidence_dir / "phase3-widgets-desktop.png"),
                    full_page=True,
                )

                page.set_viewport_size({"width": 430, "height": 932})
                page.wait_for_timeout(100)
                assert page.evaluate(
                    "document.documentElement.scrollWidth <= document.documentElement.clientWidth"
                )
                narrow_card = card.bounding_box()
                narrow_columns = card.locator('.widget-kanban-col')
                narrow_todo = narrow_columns.nth(0).bounding_box()
                narrow_done = narrow_columns.nth(1).bounding_box()
                assert narrow_card and narrow_todo and narrow_done
                assert narrow_done["y"] > narrow_todo["y"] + narrow_todo["height"]
                empty_column = card.locator('.widget-kanban-col.is-empty')
                assert empty_column.count() == 1
                empty_box = empty_column.bounding_box()
                assert empty_box and empty_box["height"] < 56
                json_geometry = long_json.evaluate(
                    """element => {
                        const pre = element.querySelector('pre');
                        const card = element.closest('[data-widget-key]');
                        return {
                            cardClient: card.clientWidth,
                            cardScroll: card.scrollWidth,
                            jsonClient: element.clientWidth,
                            jsonScroll: element.scrollWidth,
                            preClient: pre.clientWidth,
                            preScroll: pre.scrollWidth,
                        };
                    }"""
                )
                assert json_geometry["cardScroll"] <= json_geometry["cardClient"]
                assert json_geometry["jsonScroll"] <= json_geometry["jsonClient"]
                assert json_geometry["preScroll"] <= json_geometry["preClient"]
                card.screenshot(
                    path=str(evidence_dir / "phase3-widgets-narrow.png"),
                )
                page.locator('.widgets-scroll').evaluate(
                    "element => { element.scrollTop = element.scrollHeight; }"
                )
                page.wait_for_timeout(100)
                page.screenshot(
                    path=str(evidence_dir / "phase3-widgets-narrow-kanban.png"),
                )

                page.set_viewport_size({"width": 1440, "height": 1000})
                page.wait_for_timeout(100)

                page.click('[data-nav-page="settings"]')
                page.locator('[data-settings-tab="advanced"]').click()
                section = page.locator('.settings-extension-section').filter(
                    has_text="Phase 3 settings"
                )
                section.wait_for(state="visible", timeout=30_000)
                settings_form = section.locator('[data-extension-settings-form]')
                settings_form.locator('select[name="mode"]').select_option("fast")
                settings_form.locator('input[name="token"]').fill("discard-me")
                save = settings_form.locator('button[type="submit"]')
                save.click()
                page.wait_for_function(
                    "() => [...document.querySelectorAll('[data-extension-settings-form] button')].some((button) => button.disabled && button.textContent === 'Saving mode…')"
                )
                section.locator('[data-extension-settings-status]').filter(
                    has_text="Saved fast mode."
                ).wait_for(state="visible", timeout=10_000)
                assert save.is_enabled()
                page.screenshot(
                    path=str(evidence_dir / "phase3-settings-desktop.png"),
                    full_page=True,
                )
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise

@pytest.mark.ui_browser
def test_ui_smoke_v679_subagent_depth_zero_round_trips_through_settings(direct_server_with_data):
    """v6.79.0: the owner can actually reach, save, and re-read a Subagent Depth of 0.

    The structural fix (``_bounded_positive_int_setting`` honouring a configured 0) is
    unreachable if the visible control refuses the value or the load path rewrites it back to
    the fallback. This drives the real consumer flow — Settings -> Advanced -> type -> save ->
    full page reload — and pins the three neighbouring states so the fix cannot silently break
    them: 0 (no delegation), a normal positive value, and empty (falls back, does not persist
    an invalid value). Screenshots are written for vision inspection; a saved screenshot is
    not verification on its own (docs/DEVELOPMENT.md "Browser/mobile verification").
    """
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    settings_path = data_dir / "settings.json"
    evidence_dir = pathlib.Path(
        os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(data_dir.parent))
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)

    def saved_depth():
        return json.loads(settings_path.read_text(encoding="utf-8")).get(
            "OUROBOROS_MAX_SUBAGENT_DEPTH", "<absent>"
        )

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1400, "height": 1000})
            try:
                def open_settings_advanced():
                    """Reload the whole app the way an owner would, then reopen the field."""
                    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                    page.wait_for_selector('[data-nav-page="settings"]', timeout=30_000)
                    page.click('[data-nav-page="settings"]')
                    page.wait_for_selector("#s-subagent-depth", state="attached", timeout=30_000)
                    # D-10: subagent depth bounds the AGENTS, so it moved out of
                    # Advanced -> Runtime Limits into Agents -> Delegation.
                    page.click('[data-settings-tab="agents"]')
                    depth = page.locator("#s-subagent-depth")
                    depth.wait_for(state="visible", timeout=30_000)
                    # Saving is blocked until the settings load succeeds; waiting on the real
                    # enablement avoids racing the first fetch.
                    page.wait_for_function(
                        "() => document.querySelector('#btn-save-settings')?.disabled === false",
                        timeout=30_000,
                    )
                    depth.scroll_into_view_if_needed()
                    return depth

                def type_depth(value):
                    page.fill("#s-subagent-depth", value)
                    page.dispatch_event("#s-subagent-depth", "input")
                    page.dispatch_event("#s-subagent-depth", "change")

                def save_and_wait():
                    page.click("#btn-save-settings")
                    page.wait_for_function(
                        "() => (document.querySelector('#settings-status')?.textContent || '')"
                        ".includes('Settings saved')",
                        timeout=30_000,
                    )

                depth = open_settings_advanced()
                # The control must admit 0 at all: a min of 1 would make it unreachable.
                assert depth.get_attribute("min") == "0"
                assert depth.get_attribute("max") == "10"
                assert depth.is_enabled()
                assert depth.input_value() == "2"  # unset -> visible fallback
                page.screenshot(path=str(evidence_dir / "v679-depth-01-initial-unset.png"))

                # 0 is a valid value for the control, not a validation error.
                type_depth("0")
                assert page.evaluate(
                    "() => document.querySelector('#s-subagent-depth').validity.valid"
                ) is True
                assert page.evaluate(
                    "() => document.querySelector('#s-subagent-depth').validationMessage"
                ) == ""
                page.screenshot(path=str(evidence_dir / "v679-depth-02-typed-zero.png"))

                save_and_wait()
                assert page.locator("#s-subagent-depth").input_value() == "0"
                assert saved_depth() == 0
                page.screenshot(path=str(evidence_dir / "v679-depth-03-saved-zero.png"))

                # The round trip is the point: a reload must not rewrite 0 back to 2.
                open_settings_advanced()
                assert page.locator("#s-subagent-depth").input_value() == "0"
                assert saved_depth() == 0
                page.screenshot(path=str(evidence_dir / "v679-depth-04-reload-zero.png"))

                # Neighbouring state: an ordinary positive value still round-trips.
                type_depth("3")
                save_and_wait()
                assert saved_depth() == 3
                open_settings_advanced()
                assert page.locator("#s-subagent-depth").input_value() == "3"
                page.screenshot(path=str(evidence_dir / "v679-depth-05-reload-three.png"))

                # Neighbouring state: empty is not a value — it falls back to 2 rather than
                # persisting an unparsable setting.
                type_depth("")
                page.screenshot(path=str(evidence_dir / "v679-depth-06-empty-typed.png"))
                save_and_wait()
                assert saved_depth() == 2
                open_settings_advanced()
                assert page.locator("#s-subagent-depth").input_value() == "2"
                page.screenshot(path=str(evidence_dir / "v679-depth-07-reload-after-empty.png"))
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise

@pytest.mark.ui_browser
def test_ui_owner_context_mode_and_scope_review_ack(direct_server_with_data):
    """Owner context intent and scope-review ack, driven in a real browser.

    Two claimed-complete owner flows that source-string tests cannot certify:

    1. OWNER MAX. Switching an explicit Low to Max succeeds without a Main-route
       context-window confirmation; the frozen compatibility field remains false.
    2. SCOPE-REVIEW CAPABILITY ACK. Saving a scope-review slot whose route has no >=1M evidence
       must raise the owner confirm and, on accept, persist a route-scoped capability ack and say
       so in the settings status line.
    """
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    settings_path = data_dir / "settings.json"
    evidence_dir = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(data_dir.parent)))
    evidence_dir.mkdir(parents=True, exist_ok=True)

    # Boot into explicit owner Low with the one-window false provenance tombstone.
    seeded = json.loads(settings_path.read_text(encoding="utf-8"))
    seeded["OUROBOROS_CONTEXT_MODE"] = "low"
    seeded["OUROBOROS_CONTEXT_MODE_AUTO_LOW"] = "false"
    seeded["OUROBOROS_SCOPE_REVIEW_MODELS"] = seeded["OUROBOROS_MODEL"]
    settings_path.write_text(json.dumps(seeded), encoding="utf-8")
    direct_server_with_data["restart_server"]()

    with urllib.request.urlopen(f"{url}/api/state", timeout=5) as resp:  # noqa: S310 - local test server
        boot_state = json.loads(resp.read().decode("utf-8"))
    assert boot_state["context_mode"] == "low"
    assert boot_state["context_mode_auto_low"] is False

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            dialogs: list[str] = []
            page.on("dialog", lambda dialog: (dialogs.append(dialog.message), dialog.accept()))
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=60_000)
                toggle = page.locator("#chat-context-mode")
                toggle.wait_for(state="visible", timeout=60_000)
                assert toggle.get_attribute("data-context-mode") == "low"
                page.screenshot(path=str(evidence_dir / "context-mode-low-before.png"))

                toggle.locator('.chat-seg[data-mode="max"]').click()
                page.wait_for_function(
                    "() => document.querySelector('#chat-context-mode')?.dataset.contextMode === 'max'",
                    timeout=30_000,
                )
                assert page.locator(".confirm-dialog:not([hidden])").count() == 0
                page.screenshot(path=str(evidence_dir / "context-mode-max-after.png"))
                persisted = json.loads(settings_path.read_text(encoding="utf-8"))
                assert persisted["OUROBOROS_CONTEXT_MODE"] == "max"
                assert persisted["OUROBOROS_CONTEXT_MODE_AUTO_LOW"] == "false"
                with urllib.request.urlopen(f"{url}/api/state", timeout=5) as resp:  # noqa: S310
                    after = json.loads(resp.read().decode("utf-8"))
                assert after["context_mode"] == "max"
                assert after["context_mode_auto_low"] is False

                # 2. Scope-review capability notice -> owner confirm -> route-scoped ack.
                # 6.2: the scope route is a review-lane row — pick the API-model
                # route in the grouped combobox and type the id. D-10 moved the
                # lanes out of Models into their own Agents tab.
                page.click('[data-nav-page="settings"]')
                page.wait_for_selector("#s-context-mode", state="attached", timeout=30_000)
                page.locator('[data-settings-tab="agents"]').click()
                page.wait_for_selector("#reviewer-slots-section", timeout=30_000)
                scope_route = page.locator(
                    '#reviewer-scope-rows .reviewer-slot-row [data-slot-route]'
                ).first
                scope_route.wait_for(state="visible", timeout=30_000)
                scope_route.select_option("api")
                custom_input = page.locator(
                    '#reviewer-scope-rows .reviewer-slot-row [data-slot-custom-api]'
                ).first
                custom_input.wait_for(state="visible", timeout=30_000)
                custom_input.fill("openai-compatible::scope-reviewer-x")
                page.locator("#btn-save-settings").click()
                # The capability ack is an in-app dialog since the native-dialog
                # class ban (tests/test_web_dialogs_static.py); Playwright's
                # page.on("dialog") hook only fires for window.alert/confirm/prompt.
                ack_dialog = page.locator(".confirm-dialog")
                ack_dialog.wait_for(state="visible", timeout=60_000)
                ack_text = ack_dialog.inner_text()
                page.screenshot(path=str(evidence_dir / "v6800-scope-review-ack.png"), full_page=True)
                ack_dialog.locator("[data-confirm-ok]").last.click()
                page.wait_for_function(
                    "() => (document.querySelector('#settings-status')?.textContent || '')"
                    ".includes('scope-review route')",
                    timeout=60_000,
                )

                assert "1,000,000-token context window" in ack_text
                assert "openai-compatible::scope-reviewer-x" in ack_text, "the ack must name the exact route"
                status_text = page.locator("#settings-status").inner_text()
                assert "Confirmed the required context window for 1 scope-review route(s)." in status_text
                evidence = json.loads((data_dir / "state" / "capability_evidence.json").read_text(encoding="utf-8"))
                acked = [
                    entry for entry in (evidence.get("acks") or evidence.get("probes") or {}).values()
                    if str(entry.get("model") or "") == "openai-compatible::scope-reviewer-x"
                ]
                assert acked, "no route-scoped capability evidence was stored for the acked reviewer"
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise
