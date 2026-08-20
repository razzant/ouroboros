"""The chat surface: chronology, scrolling and the composer.

Split verbatim out of ``tests/test_ui_smoke_playwright.py`` by theme. This module owns
the chronology a reconnect may not reorder, the collapsed activity line, the desktop
scroll behaviour, the composer chips on desktop and mobile, and the mobile keyboard
state that must never hide an open drawer.

Every test here launches a real browser and is marked ``ui_browser``, so the default
local run deselects the whole module.
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest


from tests._ui_smoke_shared import direct_server as _direct_server
from tests._ui_smoke_shared import direct_server_with_data as _direct_server_with_data

# Fixtures are requested by name as test parameters, so they are re-bound through a
# module attribute: a direct import of a name that reappears as a parameter is an F811
# redefinition under the CI ruff gate.
direct_server = _direct_server
direct_server_with_data = _direct_server_with_data


@pytest.mark.ui_browser
@pytest.mark.parametrize("browser_engine", ["chromium", "webkit"])
def test_ui_smoke_collapsed_activity_line_named_vs_unnamed(
    direct_server_with_data,
    browser_engine,
):
    """Collapsed root summaries stay compact without destroying full activity."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    logs_dir = data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "chat.jsonl").write_text("", encoding="utf-8")
    unique_tail = "UNIQUE_FULL_ACTIVITY_TAIL"
    long_activity = (
        "Analyzing the dataset and comparing every source. " * 18
        + "https://example.com/" + "unbroken-segment-" * 18 + unique_tail
    )
    (logs_dir / "progress.jsonl").write_text(
        json.dumps({
            "ts": "2026-07-29T10:00:00+00:00",
            "chat_id": 1,
            "task_id": "named-act",
            "content": long_activity,
        }) + "\n" + json.dumps({
            "ts": "2026-07-29T10:00:01+00:00",
            "chat_id": 1,
            "task_id": "unnamed-act",
            "content": "Doing things without a name",
        }) + "\n",
        encoding="utf-8",
    )
    task_results = data_dir / "task_results"
    task_results.mkdir(parents=True, exist_ok=True)
    (task_results / "named-act.json").write_text(json.dumps({
        "task_id": "named-act",
        "status": "completed",
        "suggested_name": "Data Analysis",
        "cost_usd": 0.42,
        "cost_accounting_status": "available",
        "cost_final": True,
    }) + "\n", encoding="utf-8")

    try:
        with sync_playwright() as pw:
            browser_type = getattr(pw, browser_engine)
            try:
                browser = browser_type.launch(headless=True)
            except PlaywrightError as exc:
                if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
                    pytest.skip(f"Playwright {browser_engine} browser is not installed: {exc}")
                raise
            try:
                for width, height, mobile in [(1440, 1000, False), (390, 844, True)]:
                    context = browser.new_context(
                        viewport={"width": width, "height": height},
                        is_mobile=mobile,
                        has_touch=mobile,
                    )
                    page = context.new_page()
                    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                    named = page.locator('.chat-live-card[data-task-id="named-act"]')
                    named.wait_for(state="attached", timeout=30_000)
                    unnamed = page.locator('.chat-live-card[data-task-id="unnamed-act"]')
                    unnamed.wait_for(state="attached", timeout=30_000)
                    page.wait_for_function(
                        "() => document.querySelector('.chat-live-card[data-task-id=\"named-act\"]"
                        " [data-live-title]')?.textContent === 'Data Analysis'",
                        timeout=30_000,
                    )
                    assert named.get_attribute("data-expanded") == "0"
                    named_activity = named.locator('[data-live-activity]')
                    activity_text = named_activity.text_content().strip()
                    assert activity_text
                    assert len(activity_text) <= 240
                    assert activity_text.endswith(("…", "..."))
                    assert unique_tail not in activity_text
                    assert named_activity.get_attribute("title") is None
                    geometry = named.evaluate(
                        """card => {
                            const facts = selector => {
                                const el = card.querySelector(selector);
                                const style = getComputedStyle(el);
                                const lineHeight = parseFloat(style.lineHeight);
                                const rect = el.getBoundingClientRect();
                                return { lines: rect.height / lineHeight, width: rect.width };
                            };
                            return {
                                title: facts('[data-live-title]'),
                                activity: facts('[data-live-activity]'),
                                clientWidth: card.clientWidth,
                                scrollWidth: card.scrollWidth,
                            };
                        }"""
                    )
                    assert geometry["title"]["lines"] <= 2.2, geometry
                    assert geometry["activity"]["lines"] <= 2.2, geometry
                    assert geometry["scrollWidth"] <= geometry["clientWidth"] + 1, geometry
                    assert "cost=$0.42" in named.locator('[data-live-meta]').inner_text()

                    unnamed_activity = unnamed.locator('[data-live-activity]')
                    assert "Doing things without a name" in unnamed.locator('[data-live-title]').text_content()
                    assert unnamed_activity.text_content().strip() == ""
                    assert not unnamed_activity.is_visible()

                    named.locator(':scope > [data-live-summary-button]').click()
                    line_toggle = named.locator(':scope > [data-live-timeline] .chat-live-line-toggle').first
                    line_toggle.wait_for(state="visible", timeout=5_000)
                    line_toggle.click()
                    page.wait_for_function(
                        "tail => document.querySelector('.chat-live-card[data-task-id=\"named-act\"]')"
                        ".innerText.includes(tail)",
                        arg=unique_tail,
                    )
                    page.screenshot(
                        path=str(data_dir.parent / f"compact-activity-{browser_engine}-{width}.png"),
                        full_page=True,
                    )
                    context.close()
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise

@pytest.mark.ui_browser
def test_ui_smoke_chat_chronology_reconnect_and_plain_answer_marker(direct_server_with_data):
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    logs_dir = data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir = pathlib.Path(
        os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(data_dir.parent))
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    anchor_summary = {
        "ts": "2025-07-18T10:00:03+00:00",
        "direction": "system",
        "type": "task_summary",
        "system_type": "task_summary",
        "task_id": "chronology-anchor",
        "chat_id": 1,
        "text": "Mounted task card whose earliest event will be backfilled.",
        "tool_calls": 1,
        "rounds": 2,
        "outcome_axes": {
            "lifecycle": {"status": "completed"},
            "execution": {"status": "ok"},
            "objective": {"status": "pass"},
            "review": {"status": "pass"},
            "artifacts": {"status": "ready"},
        },
    }
    t3 = {
        "ts": "2025-07-18T10:00:03.200000+00:00",
        "direction": "out",
        "chat_id": 1,
        "text": "Third historical message.\n" + "\n".join(
            f"Scrollable historical detail {index}." for index in range(80)
        ),
        "format": "markdown",
    }
    (logs_dir / "chat.jsonl").write_text(
        json.dumps(anchor_summary) + "\n" + json.dumps(t3) + "\n",
        encoding="utf-8",
    )

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1280, "height": 800})
            page = context.new_page()
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                third = page.locator(".chat-bubble", has_text="Third historical message.").first
                third.wait_for(state="attached", timeout=30_000)
                assert third.is_visible()
                mounted_anchor = page.locator(
                    '.chat-live-card[data-task-id="chronology-anchor"]'
                )
                mounted_anchor.wait_for(state="attached", timeout=30_000)
                assert mounted_anchor.is_visible()

                t1 = {
                    "ts": "2025-07-18T10:00:01+00:00",
                    "direction": "out",
                    "chat_id": 1,
                    "text": "First historical message.\nFINAL ANSWER: 41",
                    "format": "markdown",
                }
                t2 = {
                    "ts": "2025-07-18T10:00:02+00:00",
                    "direction": "system",
                    "type": "notice",
                    "chat_id": 1,
                    "text": "Second historical system message.\nFINAL ANSWER: 42",
                    "format": "markdown",
                }
                disconnected_summary = {
                    "ts": "2025-07-18T10:00:02.500000+00:00",
                    "direction": "system",
                    "type": "task_summary",
                    "system_type": "task_summary",
                    "task_id": "chronology-disconnected",
                    "chat_id": 1,
                    "text": "Disconnected summary-only card.",
                    "tool_calls": 1,
                    "rounds": 2,
                    "outcome_axes": {
                        "lifecycle": {"status": "completed"},
                        "execution": {"status": "ok"},
                        "objective": {"status": "pass"},
                        "review": {"status": "pass"},
                        "artifacts": {"status": "ready"},
                    },
                }
                t4 = {
                    "ts": "2025-07-18T10:00:04+00:00",
                    "direction": "out",
                    "chat_id": 1,
                    "text": "Fourth new message below the reading anchor.",
                    "format": "markdown",
                }
                (logs_dir / "chat.jsonl").write_text(
                    "".join(
                        json.dumps(row) + "\n"
                        for row in (anchor_summary, t3, t1, t2, disconnected_summary, t4)
                    ),
                    encoding="utf-8",
                )
                (logs_dir / "progress.jsonl").write_text(
                    json.dumps({
                        "ts": "2025-07-18T10:00:01.500000+00:00",
                        "chat_id": 1,
                        "task_id": "chronology-progress-only",
                        "content": "Progress-only terminal card.",
                    }) + "\n" + json.dumps({
                        "ts": "2025-07-18T10:00:01.750000+00:00",
                        "chat_id": 1,
                        "task_id": "chronology-anchor",
                        "content": "Earlier progress backfilled for the mounted anchor card.",
                    }) + "\n",
                    encoding="utf-8",
                )
                task_results = data_dir / "task_results"
                task_results.mkdir(parents=True, exist_ok=True)
                (task_results / "chronology-progress-only.json").write_text(json.dumps({
                    "task_id": "chronology-progress-only",
                    "status": "completed",
                    "outcome_axes": {
                        "lifecycle": {"status": "completed"},
                        "execution": {"status": "ok"},
                        "objective": {"status": "best_effort"},
                        "review": {"status": "degraded"},
                        "artifacts": {"status": "ready"},
                    },
                }) + "\n", encoding="utf-8")

                scroll_before = page.evaluate(
                    """() => {
                        const messages = document.querySelector('#chat-messages');
                        const anchor = messages.querySelector(
                            '.chat-live-card[data-task-id="chronology-anchor"]'
                        );
                        messages.scrollTop = Math.max(1, anchor.offsetTop - 40);
                        return {
                            top: messages.scrollTop,
                            height: messages.scrollHeight,
                            remaining: messages.scrollHeight - messages.scrollTop - messages.clientHeight,
                            anchorTop: anchor?.getBoundingClientRect().top,
                        };
                    }"""
                )
                assert scroll_before["top"] > 0
                assert scroll_before["remaining"] > 160
                direct_server_with_data["restart_server"]()
                page.wait_for_function(
                    "() => [...document.querySelectorAll('.chat-bubble.system')]"
                    ".some((node) => node.textContent.includes('Reconnected'))",
                    timeout=20_000,
                )
                page.wait_for_selector(
                    '.chat-live-card[data-task-id="chronology-progress-only"][data-finished="1"]',
                    timeout=30_000,
                )
                page.wait_for_selector(
                    '.chat-live-card[data-task-id="chronology-disconnected"][data-finished="1"]',
                    timeout=30_000,
                )
                state = page.evaluate(
                    """() => [...document.querySelector('#chat-messages').children]
                        .filter((node) => !node.classList.contains('typing-bubble')
                            && !node.textContent.includes('Reconnected'))
                        .map((node) => ({
                            text: node.textContent,
                            ts: node.dataset.ts || '',
                            card: node.classList.contains('chat-live-card'),
                            taskId: node.dataset.taskId || '',
                        }))"""
                )
                assert [item["card"] for item in state] == [
                    False, True, True, False, True, False, False,
                ]
                assert "First historical message." in state[0]["text"]
                assert "Progress-only terminal card." in state[1]["text"]
                assert state[2]["taskId"] == "chronology-anchor"
                assert "Earlier progress backfilled" in state[2]["text"]
                assert "Second historical system message." in state[3]["text"]
                assert state[4]["taskId"] == "chronology-disconnected"
                assert "Third historical message." in state[5]["text"]
                assert "Fourth new message below the reading anchor." in state[6]["text"]
                assert all(item["ts"].isdigit() for item in state)
                assert page.locator(".final-answer-chip").count() == 0
                assert "FINAL ANSWER: 41" in page.locator("#chat-messages").inner_text()
                assert "FINAL ANSWER: 42" in page.locator("#chat-messages").inner_text()
                assert "2025" in page.locator(
                    '.chat-live-card[data-task-id="chronology-progress-only"]'
                ).inner_text()
                scroll_after = page.evaluate(
                    """() => {
                        const messages = document.querySelector('#chat-messages');
                        const anchor = messages.querySelector(
                            '.chat-live-card[data-task-id="chronology-anchor"]'
                        );
                        return {
                            top: messages.scrollTop,
                            height: messages.scrollHeight,
                            anchorTop: anchor?.getBoundingClientRect().top,
                        };
                    }"""
                )
                assert abs(scroll_after["anchorTop"] - scroll_before["anchorTop"]) <= 6
                page.locator("#chat-messages").evaluate("(messages) => { messages.scrollTop = 0; }")
                page.screenshot(
                    path=str(evidence_dir / "phase3-chat-chronology-desktop.png"),
                    full_page=True,
                )

                page.set_viewport_size({"width": 390, "height": 844})
                page.keyboard.press("Escape")
                page.wait_for_selector("#primary-sidebar:not(.open)", timeout=5_000)
                backdrop = page.locator(".nav-drawer-backdrop")
                backdrop.wait_for(state="attached", timeout=5_000)
                assert backdrop.is_hidden()
                page.wait_for_timeout(250)
                page.locator("#chat-messages").evaluate("(messages) => { messages.scrollTop = 0; }")
                narrow_top_geometry = page.evaluate(
                    """() => {
                        const header = document.querySelector('.chat-page-header');
                        const first = document.querySelector('#chat-messages > :not(.typing-bubble)');
                        return {
                            headerBottom: header?.getBoundingClientRect().bottom,
                            firstTop: first?.getBoundingClientRect().top,
                        };
                    }"""
                )
                assert narrow_top_geometry["firstTop"] >= narrow_top_geometry["headerBottom"] - 2
                page.screenshot(
                    path=str(evidence_dir / "phase3-chat-chronology-narrow.png"),
                    full_page=True,
                )

                page.goto(f"{url}/?_ouro_reason=sha-change", wait_until="domcontentloaded", timeout=30_000)
                page.get_by_text("Restart complete").wait_for(state="visible", timeout=30_000)
                first = page.locator(".chat-bubble", has_text="First historical message.").first
                first.wait_for(state="attached", timeout=30_000)
                assert first.is_visible()
                assert page.locator(".final-answer-chip").count() == 0
                page.screenshot(
                    path=str(evidence_dir / "phase3-chat-chronology-reload.png"),
                    full_page=True,
                )
            finally:
                context.close()
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise

@pytest.mark.ui_browser
def test_ui_smoke_desktop_composer_chips_above_input_send_inside(direct_server):
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            try:
                page.goto(direct_server, wait_until="domcontentloaded", timeout=30_000)
                page.wait_for_selector("#chat-input", timeout=30_000)
                metrics = page.evaluate(
                    """() => {
                        const rect = (selector) => {
                            const el = document.querySelector(selector);
                            const r = el.getBoundingClientRect();
                            return { left: r.left, right: r.right, top: r.top, bottom: r.bottom, width: r.width, height: r.height };
                        };
                        return {
                            input: rect('#chat-input'),
                            toolbar: rect('.chat-toolbar-row'),
                            send: rect('.chat-send-group'),
                            sendButton: rect('.chat-send-inline'),
                            swarm: rect('.chat-swarm'),
                            contextMode: rect('.chat-context-mode'),
                        };
                    }"""
                )
                # v6.32.0 composer redesign (owner: "чипы правильнее НАД полем ввода"):
                # the chips row (Swarm + Low/Max) sits ABOVE the text input...
                assert metrics["toolbar"]["bottom"] <= metrics["input"]["top"] + 4, metrics
                assert metrics["swarm"]["bottom"] <= metrics["input"]["top"] + 4, metrics
                assert metrics["contextMode"]["bottom"] <= metrics["input"]["top"] + 4, metrics
                # ...the two chips share that row (aligned tops)...
                assert abs(metrics["swarm"]["top"] - metrics["contextMode"]["top"]) <= 2, metrics
                # ...and the Send button stays INSIDE the input's vertical band (same text row).
                assert metrics["send"]["top"] >= metrics["input"]["top"] - 4, metrics
                assert metrics["send"]["bottom"] <= metrics["input"]["bottom"] + 4, metrics
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise

@pytest.mark.ui_browser
def test_ui_smoke_mobile_composer_toolbar_does_not_overlap_input(direct_server):
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
            try:
                page.goto(direct_server, wait_until="domcontentloaded", timeout=30_000)
                page.wait_for_selector("#chat-input", timeout=30_000)
                metrics = page.evaluate(
                    """() => {
                        const rect = (selector) => {
                            const el = document.querySelector(selector);
                            const r = el.getBoundingClientRect();
                            return { left: r.left, right: r.right, top: r.top, bottom: r.bottom, width: r.width, height: r.height };
                        };
                        const inputStyle = getComputedStyle(document.querySelector('#chat-input'));
                        return {
                            input: rect('#chat-input'),
                            toolbar: rect('.chat-toolbar-row'),
                            pills: rect('.chat-composer-pills'),
                            send: rect('.chat-send-group'),
                            sendButton: rect('.chat-send-inline'),
                            swarm: rect('.chat-swarm'),
                            contextMode: rect('.chat-context-mode'),
                            paddingRight: inputStyle.paddingRight,
                        };
                    }"""
                )
                # Mobile (390px): chips ride ABOVE the input row, while the input
                # shares its row with the attach button (left) and the Send button
                # (right). The usable input width is therefore naturally below the
                # old desktop-era 300px target; assert it stays usable (>= half the
                # viewport) and never runs under the Send button.
                assert metrics["input"]["width"] >= 190, metrics
                assert metrics["input"]["right"] <= metrics["send"]["left"] + 2, metrics
                assert metrics["toolbar"]["bottom"] <= metrics["input"]["top"] + 1, metrics
                assert metrics["send"]["top"] >= metrics["input"]["top"] - 1, metrics
                assert metrics["send"]["bottom"] <= metrics["input"]["bottom"] + 1, metrics
                assert abs(metrics["swarm"]["height"] - metrics["sendButton"]["height"]) <= 1, metrics
                assert abs(metrics["contextMode"]["height"] - metrics["sendButton"]["height"]) <= 1, metrics
                assert metrics["paddingRight"] != "256px", metrics
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise

def _install_controlled_visual_viewport(page, initial_height: int) -> None:
    """Install a deterministic viewport-height signal before application JS.

    This exercises Ouroboros's viewport/focus state machine, not a native OS
    keyboard. The assertions below separately inspect the rendered drawer.
    """
    page.add_init_script(
        f"""(() => {{
            let height = {int(initial_height)};
            const viewport = new EventTarget();
            Object.defineProperty(viewport, 'height', {{ get: () => height }});
            Object.defineProperty(window, 'visualViewport', {{
                configurable: true,
                value: viewport,
            }});
            window.__setTestVisualViewportHeight = (nextHeight) => {{
                height = Number(nextHeight);
                viewport.dispatchEvent(new Event('resize'));
            }};
        }})()"""
    )

def _mobile_keyboard_drawer_assertions(page, url: str, screenshot_path: pathlib.Path) -> None:
    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    page.wait_for_selector("#chat-input", timeout=30_000)

    # A transient Telegram/WebView viewport shrink with no focused editable must
    # never claim that the software keyboard is open.
    page.evaluate("() => window.__setTestVisualViewportHeight(500)")
    page.wait_for_timeout(50)
    assert not page.locator("body").evaluate("el => el.classList.contains('keyboard-open')")

    # Restore the stable app viewport, then prove the same shrink is recognized
    # while the composer really owns focus.
    page.evaluate("() => window.__setTestVisualViewportHeight(844)")
    page.wait_for_timeout(50)
    page.focus("#chat-input")
    page.evaluate("() => window.__setTestVisualViewportHeight(500)")
    page.wait_for_function("() => document.body.classList.contains('keyboard-open')", timeout=5_000)

    toggle = page.locator("#page-chat [data-mobile-nav-toggle]")
    toggle.click()
    page.wait_for_function(
        "() => document.body.classList.contains('nav-drawer-open')"
        " && !document.body.classList.contains('keyboard-open')"
        " && document.activeElement?.id !== 'chat-input'",
        timeout=5_000,
    )
    # Wait for the drawer to actually arrive instead of sleeping past the 180ms
    # transform transition: a fixed 220ms budget left ~40ms of margin and lost
    # that race on the Linux WebKit runner, which then measured the drawer at its
    # closed position (-105% => left -336) and failed. A drawer that never opens
    # still fails here, now naming the cause instead of a stale geometry read.
    # The predicate is byte-for-byte the one asserted below, so the wait can
    # never pass on a value the assertion would reject (a rounded variant let
    # left=-1.017 through and failed one line later).
    page.wait_for_function(
        "() => document.querySelector('#primary-sidebar')"
        ".getBoundingClientRect().left >= -1",
        timeout=5_000,
    )

    state = page.evaluate(
        """() => {
            const sidebar = document.querySelector('#primary-sidebar');
            const backdrop = document.querySelector('#nav-drawer-backdrop');
            const toggle = document.querySelector('#page-chat [data-mobile-nav-toggle]');
            const rect = sidebar.getBoundingClientRect();
            return {
                bodyOpen: document.body.classList.contains('nav-drawer-open'),
                sidebarOpen: sidebar.classList.contains('open'),
                sidebarDisplay: getComputedStyle(sidebar).display,
                sidebarVisibility: getComputedStyle(sidebar).visibility,
                sidebarRect: {left: rect.left, right: rect.right, width: rect.width, height: rect.height},
                backdropHidden: backdrop.hidden,
                backdropDisplay: getComputedStyle(backdrop).display,
                ariaExpanded: toggle.getAttribute('aria-expanded'),
                activeId: document.activeElement?.id || '',
                keyboardBody: document.body.classList.contains('keyboard-open'),
                keyboardRoot: document.documentElement.classList.contains('keyboard-open'),
            };
        }"""
    )
    assert state["bodyOpen"] and state["sidebarOpen"], state
    assert state["ariaExpanded"] == "true", state
    assert not state["backdropHidden"] and state["backdropDisplay"] != "none", state
    assert state["sidebarDisplay"] != "none" and state["sidebarVisibility"] != "hidden", state
    assert state["sidebarRect"]["width"] > 200 and state["sidebarRect"]["height"] > 400, state
    assert state["sidebarRect"]["left"] >= -1 and state["sidebarRect"]["right"] > 0, state
    assert state["activeId"] != "chat-input", state
    assert not state["keyboardBody"] and not state["keyboardRoot"], state

    # The now-visible drawer must still own a vertically scrollable content
    # surface even though the keyboard touch lock was active one frame earlier.
    scroll = page.evaluate(
        """() => {
            const scroller = document.querySelector('#primary-sidebar .sidebar-scroll');
            for (let i = 0; i < 60; i += 1) {
                const row = document.createElement('button');
                row.className = 'nav-row';
                row.type = 'button';
                row.textContent = `Drawer overflow probe ${i}`;
                scroller.appendChild(row);
            }
            scroller.scrollTop = scroller.scrollHeight;
            return {
                scrollTop: scroller.scrollTop,
                scrollHeight: scroller.scrollHeight,
                clientHeight: scroller.clientHeight,
                overflowY: getComputedStyle(scroller).overflowY,
            };
        }"""
    )
    assert scroll["scrollHeight"] > scroll["clientHeight"], scroll
    assert scroll["scrollTop"] > 0, scroll
    assert scroll["overflowY"] in {"auto", "scroll"}, scroll
    page.screenshot(path=str(screenshot_path), full_page=True)

    # Exercise the real backdrop click in the visible strip to the right of the
    # 320px drawer, then require all state/ARIA projections to close together.
    page.locator("#nav-drawer-backdrop").click(position={"x": 380, "y": 400})
    page.wait_for_function(
        "() => !document.body.classList.contains('nav-drawer-open')"
        " && !document.querySelector('#primary-sidebar').classList.contains('open')"
        " && document.querySelector('#nav-drawer-backdrop').hidden"
        " && document.querySelector('#page-chat [data-mobile-nav-toggle]').getAttribute('aria-expanded') === 'false'",
        timeout=5_000,
    )

@pytest.mark.ui_browser
def test_ui_smoke_mobile_keyboard_state_cannot_hide_open_drawer_chromium(direct_server_with_data):
    """Controlled visualViewport state plus real Chromium drawer geometry."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
            try:
                _install_controlled_visual_viewport(page, 844)
                _mobile_keyboard_drawer_assertions(
                    page,
                    direct_server_with_data["url"],
                    direct_server_with_data["data_dir"].parent / "mobile-keyboard-drawer-chromium.png",
                )
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise

@pytest.mark.ui_browser
def test_ui_smoke_mobile_keyboard_state_cannot_hide_open_drawer_webkit(direct_server_with_data):
    """Same controlled state-machine check in WebKit with an iPhone profile."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as pw:
            iphone = pw.devices.get("iPhone 13")
            if not iphone:
                pytest.skip("Playwright iPhone 13 device descriptor unavailable")
            try:
                browser = pw.webkit.launch(headless=True)
            except PlaywrightError as exc:
                if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
                    pytest.skip(f"Playwright WebKit browser is not installed: {exc}")
                raise
            context = browser.new_context(**iphone)
            page = context.new_page()
            try:
                # The controller's threshold is driven by our deterministic app
                # viewport; the iPhone descriptor still owns rendering/input.
                _install_controlled_visual_viewport(page, 844)
                _mobile_keyboard_drawer_assertions(
                    page,
                    direct_server_with_data["url"],
                    direct_server_with_data["data_dir"].parent / "mobile-keyboard-drawer-webkit.png",
                )
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise

@pytest.mark.ui_browser
def test_ui_smoke_direct_mode_chat_scrolls_on_desktop(direct_server):
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    def scroll_metrics(page):
        return page.evaluate(
            """() => {
                const messages = document.querySelector('#chat-messages');
                if (!messages) return null;
                messages.scrollTop = 0;
                const top = messages.scrollTop;
                messages.scrollTop = messages.scrollHeight;
                const bottom = messages.scrollTop;
                return {
                    clientHeight: messages.clientHeight,
                    scrollHeight: messages.scrollHeight,
                    top,
                    bottom,
                    overflowY: getComputedStyle(messages).overflowY,
                    runtimeVvh: document.getElementById('runtime-vvh')?.textContent || '',
                    bodyHeight: Math.round(document.body.getBoundingClientRect().height),
                    windowHeight: window.innerHeight,
                };
            }"""
        )

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            try:
                page.goto(direct_server, wait_until="domcontentloaded", timeout=30_000)
                page.get_by_role("button", name="Chat").click()
                page.wait_for_selector("#chat-messages", timeout=30_000)
                # Wait for the initial history rebuild to finish before injecting
                # synthetic rows; otherwise that authoritative rebuild may erase
                # the probe immediately after insertion on slower startup paths.
                page.wait_for_selector("#chat-messages .chat-bubble.assistant", timeout=30_000)
                # A viewport change can re-render the chat from the (empty) real
                # history and drop injected probe nodes, so injection is a helper
                # re-run before every measurement instead of a one-shot setup.
                inject_probe_bubbles = """() => {
                    const messages = document.querySelector('#chat-messages');
                    messages.replaceChildren();
                    for (let i = 0; i < 48; i += 1) {
                        const bubble = document.createElement('div');
                        bubble.className = 'chat-bubble assistant';
                        bubble.textContent = `Desktop scroll probe ${i} `.repeat(16);
                        bubble.style.minHeight = '48px';
                        messages.appendChild(bubble);
                    }
                }"""
                page.evaluate(inject_probe_bubbles)

                metrics = scroll_metrics(page)
                assert metrics is not None
                assert metrics["overflowY"] in {"auto", "scroll"}
                assert metrics["scrollHeight"] > metrics["clientHeight"] + 100
                assert metrics["bottom"] > metrics["top"] + 100
                assert "--vvh:100dvh" in metrics["runtimeVvh"]
                assert abs(metrics["bodyHeight"] - metrics["windowHeight"]) <= 2

                page.set_viewport_size({"width": 1280, "height": 400})
                page.wait_for_timeout(100)
                page.set_viewport_size({"width": 1280, "height": 800})
                page.wait_for_timeout(100)
                page.evaluate(inject_probe_bubbles)

                metrics_after_resize = scroll_metrics(page)
                assert metrics_after_resize is not None
                assert metrics_after_resize["scrollHeight"] > metrics_after_resize["clientHeight"] + 100
                assert metrics_after_resize["bottom"] > metrics_after_resize["top"] + 100
                assert "--vvh:100dvh" in metrics_after_resize["runtimeVvh"]
                assert abs(metrics_after_resize["bodyHeight"] - metrics_after_resize["windowHeight"]) <= 2
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise
