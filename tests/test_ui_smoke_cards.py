"""Live task cards: the geometry a mutation may not disturb.

Split verbatim out of ``tests/test_ui_smoke_playwright.py`` by theme. This module owns
the viewport a live card mutation must preserve, the nesting of subagent child cards,
the geometry cards keep at depth and inside the project panel, and the height a finished
card keeps when its transcript overflows.

Every test here launches a real browser and is marked ``ui_browser``, so the default
local run deselects the whole module.
"""

from __future__ import annotations

import json

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
def test_ui_smoke_live_card_mutations_preserve_viewport(
    direct_server_with_data,
    browser_engine,
):
    """Live card growth follows bottom or preserves the visible descendant."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    logs_dir = data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "chat.jsonl").write_text("", encoding="utf-8")
    (logs_dir / "progress.jsonl").write_text("", encoding="utf-8")

    capture_socket = """() => {
        const NativeWebSocket = window.WebSocket;
        window.__testSockets = [];
        window.WebSocket = class TestWebSocket extends NativeWebSocket {
            constructor(...args) {
                super(...args);
                window.__testSockets.push(this);
            }
        };
    }"""
    settle = "() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))"

    def emit(page, frame):
        page.evaluate(
            """frame => {
                const socket = window.__testSockets?.[0];
                if (!socket) throw new Error('test socket not captured');
                socket.dispatchEvent(new MessageEvent('message', {
                    data: JSON.stringify(frame),
                }));
            }""",
            frame,
        )
        page.evaluate(settle)

    def card_top(page, task_id):
        return page.locator(f'.chat-live-card[data-task-id="{task_id}"]').evaluate(
            "card => card.getBoundingClientRect().top"
        )

    def put_at_viewport_top(page, selector):
        return page.evaluate(
            """selector => {
                const messages = document.querySelector('#chat-messages');
                const anchor = document.querySelector(selector);
                const before = messages.scrollTop;
                messages.scrollTop += anchor.getBoundingClientRect().top
                    - messages.getBoundingClientRect().top;
                messages.dispatchEvent(new Event('scroll'));
                return {
                    moved: Math.abs(messages.scrollTop - before),
                    remaining: messages.scrollHeight - messages.scrollTop - messages.clientHeight,
                };
            }""",
            selector,
        )

    try:
        with sync_playwright() as pw:
            browser_type = getattr(pw, browser_engine)
            try:
                browser = browser_type.launch(headless=True)
            except PlaywrightError as exc:
                if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
                    pytest.skip(f"Playwright {browser_engine} browser is not installed: {exc}")
                raise
            page = browser.new_page(viewport={"width": 1280, "height": 760})
            try:
                page.add_init_script(f"({capture_socket})()")
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                page.wait_for_function(
                    "() => window.__testSockets?.some(socket => socket.readyState === WebSocket.OPEN)",
                    timeout=30_000,
                )
                page.wait_for_function(
                    "() => document.querySelector('#chat-messages')?.innerText.includes('Ouroboros has awakened')",
                    timeout=30_000,
                )

                emit(page, {
                    "type": "chat", "role": "assistant", "is_progress": True,
                    "chat_id": 1, "task_id": "vp-parent",
                    "content": "Parent begins", "ts": "2026-08-03T10:00:00+00:00",
                })
                for idx in range(1, 5):
                    emit(page, {
                        "type": "chat", "role": "assistant", "is_progress": True,
                        "chat_id": 1, "task_id": f"vp-child-{idx}",
                        "delegation_role": "subagent", "subagent_event": "scheduled",
                        "subagent_task_id": f"vp-child-{idx}", "parent_task_id": "vp-parent",
                        "root_task_id": "vp-parent", "subagent_role": f"reader-{idx}",
                        "content": f"Child {idx} scheduled",
                        "ts": f"2026-08-03T10:00:0{idx}+00:00",
                    })
                for idx in range(1, 11):
                    emit(page, {
                        "type": "chat", "role": "assistant", "is_progress": True,
                        "chat_id": 1, "task_id": f"vp-follow-{idx}",
                        "content": (f"Following task {idx} " * 14),
                        "ts": f"2026-08-03T10:01:{idx:02d}+00:00",
                    })
                page.wait_for_selector('.chat-live-card[data-task-id="vp-follow-10"]', timeout=30_000)

                # Pinned readers continue following the newest content.
                before_bottom = page.evaluate(
                    """() => {
                        const m = document.querySelector('#chat-messages');
                        m.scrollTop = m.scrollHeight;
                        m.dispatchEvent(new Event('scroll'));
                        return m.scrollHeight;
                    }"""
                )
                page.evaluate(settle)
                emit(page, {
                    "type": "chat", "role": "assistant", "is_progress": True,
                    "chat_id": 1, "task_id": "vp-bottom-child",
                    "delegation_role": "subagent", "subagent_event": "scheduled",
                    "subagent_task_id": "vp-bottom-child", "parent_task_id": "vp-follow-10",
                    "root_task_id": "vp-follow-10", "subagent_role": "bottom-reader",
                    "content": "A newly mounted child at the bottom",
                    "ts": "2026-08-03T10:02:00+00:00",
                })
                bottom = page.evaluate(
                    """() => {
                        const m = document.querySelector('#chat-messages');
                        return { height: m.scrollHeight, remaining: m.scrollHeight - m.scrollTop - m.clientHeight };
                    }"""
                )
                assert bottom["height"] > before_bottom + 30, bottom
                assert bottom["remaining"] <= 6, bottom

                # The reader is inside a child. Naming the parent and growing its
                # visible timeline above that child must keep the child stationary.
                parent = page.locator('.chat-live-card[data-task-id="vp-parent"]')
                parent.locator(':scope > [data-live-summary-button]').click()
                child_selector = '.chat-live-card[data-task-id="vp-child-2"] > [data-live-summary-button]'
                mid = put_at_viewport_top(page, child_selector)
                page.evaluate(settle)
                assert mid["remaining"] > 160, mid
                child_before = page.locator(child_selector).evaluate("el => el.getBoundingClientRect().top")
                parent_height_before = parent.evaluate("card => card.getBoundingClientRect().height")
                emit(page, {
                    "type": "task_named", "task_id": "vp-parent",
                    "suggested_name": "A deliberately long generated project name " * 12,
                })
                child_after_name = page.locator(child_selector).evaluate("el => el.getBoundingClientRect().top")
                parent_height_named = parent.evaluate("card => card.getBoundingClientRect().height")
                assert parent_height_named > parent_height_before + 20
                assert abs(child_after_name - child_before) <= 6

                for idx in range(8):
                    emit(page, {
                        "type": "chat", "role": "assistant", "is_progress": True,
                        "chat_id": 1, "task_id": "vp-parent",
                        "content": (f"Visible parent timeline update {idx} " * 10),
                        "ts": f"2026-08-03T10:03:{idx:02d}+00:00",
                    })
                child_after_growth = page.locator(child_selector).evaluate("el => el.getBoundingClientRect().top")
                parent_height_grown = parent.evaluate("card => card.getBoundingClientRect().height")
                assert parent_height_grown > parent_height_named + 100
                assert abs(child_after_growth - child_before) <= 6

                # A child mounted in an earlier card must not move the next
                # top-level card the reader is looking at.
                anchor_id = "vp-follow-1"
                anchor_selector = f'.chat-live-card[data-task-id="{anchor_id}"]'
                mid = put_at_viewport_top(page, anchor_selector)
                page.evaluate(settle)
                assert mid["remaining"] > 160, mid
                anchor_before = card_top(page, anchor_id)
                parent_before_mount = parent.evaluate("card => card.getBoundingClientRect().height")
                emit(page, {
                    "type": "chat", "role": "assistant", "is_progress": True,
                    "chat_id": 1, "task_id": "vp-late-child",
                    "delegation_role": "subagent", "subagent_event": "scheduled",
                    "subagent_task_id": "vp-late-child", "parent_task_id": "vp-parent",
                    "root_task_id": "vp-parent", "subagent_role": "late-reader",
                    "content": "Late child mounted above the reader",
                    "ts": "2026-08-03T10:04:00+00:00",
                })
                assert parent.evaluate("card => card.getBoundingClientRect().height") > parent_before_mount + 30
                assert abs(card_top(page, anchor_id) - anchor_before) <= 6

                # Terminal auto-collapse is another large height change above the
                # same reader anchor.
                parent_before_finish = parent.evaluate("card => card.getBoundingClientRect().height")
                emit(page, {
                    "type": "chat", "role": "system", "system_type": "task_summary",
                    "chat_id": 1, "task_id": "vp-parent", "content": "Parent completed",
                    "ts": "2026-08-03T10:05:00+00:00",
                    "outcome_axes": {
                        "lifecycle": {"status": "completed"}, "execution": {"status": "ok"},
                        "objective": {"status": "pass"}, "review": {"status": "pass"},
                        "artifacts": {"status": "ready"},
                    },
                })
                assert parent.get_attribute("data-expanded") == "0"
                assert parent.evaluate("card => card.getBoundingClientRect().height") < parent_before_finish - 100
                assert abs(card_top(page, anchor_id) - anchor_before) <= 6

                # Review evidence still auto-expands a child, under the same
                # viewport contract and without changing the ordinary policy.
                review_child = page.locator('.chat-live-card[data-task-id="vp-late-child"]')
                review_before = review_child.evaluate("card => card.getBoundingClientRect().height")
                emit(page, {
                    "type": "chat", "role": "assistant", "is_progress": True,
                    "chat_id": 1, "task_id": "vp-late-child",
                    "delegation_role": "subagent", "subagent_event": "completed",
                    "subagent_task_id": "vp-late-child", "parent_task_id": "vp-parent",
                    "root_task_id": "vp-parent", "subagent_role": "late-reader",
                    "result": "Review-bearing result " * 16,
                    "status": "completed", "ts": "2026-08-03T10:06:00+00:00",
                    "review_projection": {"panels": [{
                        "panel_id": "viewport-review", "surface": "task_acceptance",
                        "authority": "host_root", "aggregate_signal": "PASS",
                        "transport_status": "success", "parse_status": "valid",
                        "quorum": {"required": 1, "contributed": 1, "configured": 1},
                        "enforcement_impact": "supports_pass", "reason": "viewport evidence",
                        "actors": [],
                    }]},
                })
                assert review_child.get_attribute("data-expanded") == "1"
                assert review_child.evaluate("card => card.getBoundingClientRect().height") > review_before + 20
                assert abs(card_top(page, anchor_id) - anchor_before) <= 6
                page.screenshot(
                    path=str(data_dir.parent / f"live-card-viewport-{browser_engine}.png"),
                    full_page=True,
                )
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise

@pytest.mark.ui_browser
def test_ui_smoke_direct_mode_nests_subagent_child_cards(direct_server_with_data):
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    logs_dir = data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    child_review_projection = {
        "panels": [{
            "panel_id": "panel_child_review",
            "surface": "task_acceptance",
            "authority": "host_root",
            "aggregate_signal": "DEGRADED",
            "transport_status": "success",
            "parse_status": "valid",
            "quorum": {"required": 1, "contributed": 0, "configured": 1},
            "enforcement_impact": "degrades_completion",
            "reason": "Child evidence was incomplete.",
            "actors": [{
                "slot_id": "child_actor",
                "actor_role": "task acceptance",
                "provider": "openrouter",
                "model": "anthropic/claude-fable-5",
                "transport_status": "success",
                "parse_status": "valid",
                "semantic_verdict": "DEGRADED",
                "quorum_contribution": False,
                "enforcement_impact": "abstains",
                "reason": "Missing child visual evidence.",
            }],
        }],
    }
    child_outcome_axes = {
        "lifecycle": {"status": "completed"},
        "execution": {"status": "ok"},
        "objective": {"status": "best_effort"},
        "review": {"status": "degraded"},
        "artifacts": {"status": "ready"},
    }
    child_activity_tail = "UNIQUE_CHILD_ACTIVITY_TAIL"
    child_activity_early = "UNIQUE_CHILD_ACTIVITY_EARLY"
    child_activity = (
        child_activity_early + " Searching evidence across repositories.\n"
        + "Comparing every source and preserving the complete routed narration. " * 14
        + "\nhttps://example.com/" + "child-evidence-segment-" * 14 + child_activity_tail
    )
    rows = [
        {
            "ts": "2026-05-25T10:00:00+00:00",
            "chat_id": 1,
            "task_id": "parent1",
            "content": "Parent task started",
            "is_progress": True,
        },
        {
            "ts": "2026-05-25T10:00:01+00:00",
            "chat_id": 1,
            "task_id": "child1",
            "content": "Scheduled subagent child1",
            "is_progress": True,
            "delegation_role": "subagent",
            "subagent_event": "scheduled",
            "subagent_task_id": "child1",
            "parent_task_id": "parent1",
            "root_task_id": "parent1",
            "subagent_role": "researcher",
        },
        {
            # Real rejection carrier shape: task_id addresses the parent chat,
            # while subagent_task_id is the child presentation identity.
            "ts": "2026-05-25T10:00:01.500000+00:00",
            "chat_id": 1,
            "task_id": "parent1",
            "content": "Rejected child should stay a child",
            "is_progress": True,
            "delegation_role": "subagent",
            "subagent_event": "rejected",
            "subagent_task_id": "rejected1",
            "parent_task_id": "parent1",
            "root_task_id": "parent1",
            "subagent_role": "rejected-reader",
            "status": "rejected_duplicate",
            "error": "Active-subagent cap rejected this child",
        },
        {
            "ts": "2026-05-25T10:00:02+00:00",
            "chat_id": 1,
            "task_id": "child1",
            "content": "Subagent child1 running",
            "is_progress": True,
            "delegation_role": "subagent",
            "subagent_event": "running",
            "subagent_task_id": "child1",
            "parent_task_id": "parent1",
            "root_task_id": "parent1",
            "subagent_role": "researcher",
            "status": "running",
        },
        {
            "ts": "2026-05-25T10:00:02.500000+00:00",
            "chat_id": 1,
            "task_id": "child1",
            "content": child_activity,
            "is_progress": True,
            "delegation_role": "subagent",
            "subagent_event": "progress",
            "subagent_task_id": "child1",
            "parent_task_id": "parent1",
            "root_task_id": "parent1",
            "subagent_role": "researcher",
            "status": "running",
        },
        {
            "ts": "2026-05-25T10:00:03+00:00",
            "chat_id": 1,
            "task_id": "child1",
            "content": "Subagent child1 completed",
            "is_progress": True,
            "delegation_role": "subagent",
            "subagent_event": "completed",
            "subagent_task_id": "child1",
            "parent_task_id": "parent1",
            "root_task_id": "parent1",
            "subagent_role": "researcher",
            "status": "completed",
            "cost_usd": 0.125,
            "result": "Child result with evidence table\n| source | verdict |\n| A | pass |",
            "trace_summary": "searched sources\ncompared output",
            "outcome_axes": child_outcome_axes,
            "reason_code": "acceptance_degraded",
            "review_projection": child_review_projection,
        },
        {
            "ts": "2026-05-25T10:00:03.100000+00:00",
            "chat_id": 1,
            "task_id": "grandchild1",
            "content": "Scheduled nested subagent grandchild1",
            "is_progress": True,
            "delegation_role": "subagent",
            "subagent_event": "scheduled",
            "subagent_task_id": "grandchild1",
            "parent_task_id": "child1",
            "root_task_id": "parent1",
            "subagent_role": "evidence-mapper",
        },
        {
            "ts": "2026-05-25T10:00:03.200000+00:00",
            "chat_id": 1,
            "task_id": "grandchild1",
            "content": "Nested subagent grandchild1 completed",
            "is_progress": True,
            "delegation_role": "subagent",
            "subagent_event": "completed",
            "subagent_task_id": "grandchild1",
            "parent_task_id": "child1",
            "root_task_id": "parent1",
            "subagent_role": "evidence-mapper",
            "status": "completed",
            "result": "Nested evidence result",
        },
    ]
    (logs_dir / "progress.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    (logs_dir / "chat.jsonl").write_text(
        json.dumps({
            "ts": "2026-05-25T10:00:03.500000+00:00",
            "chat_id": 1,
            "direction": "out",
            "task_id": "child1",
            "text": "Final child answer should stay inside the child card.",
            "format": "markdown",
        }) + "\n",
        encoding="utf-8",
    )

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                page.wait_for_selector(".chat-live-card", state="attached", timeout=30_000)
                assert page.locator(".chat-live-card").first.is_visible()
                # Subagents render as always-visible child cards nested under
                # the parent card. Child completion must not finish the parent.
                page.wait_for_function("() => document.querySelectorAll('.chat-live-card').length === 4", timeout=30_000)
                page.wait_for_function(
                    "() => { const p = document.querySelector('.chat-live-card:not(.subagent)');"
                    " const c = document.querySelector('.chat-live-card.subagent[data-parent-task-id=\"parent1\"]');"
                    " const g = document.querySelector('.chat-live-card.subagent[data-parent-task-id=\"child1\"]');"
                    " return !!p && !!c && c.closest('.chat-subagents') && c.parentElement.closest('.chat-live-card') === p"
                    " && !!g && g.closest('.chat-subagents') && g.parentElement.closest('.chat-live-card') === c"
                    " && /researcher \\(child1\\)/.test(c.innerText)"
                    " && /evidence-mapper \\(grandchi/.test(g.innerText); }",
                    timeout=30_000,
                )
                parent = page.locator(".chat-live-card:not(.subagent)").first
                child = page.locator('.chat-live-card.subagent[data-task-id="child1"]')
                rejected_child = page.locator('.chat-live-card.subagent[data-task-id="rejected1"]')
                grandchild = page.locator('.chat-live-card.subagent[data-parent-task-id="child1"]').first
                parent_ts = int(parent.get_attribute("data-ts"))
                child_ts = int(child.get_attribute("data-ts"))
                grandchild_ts = int(grandchild.get_attribute("data-ts"))
                assert parent_ts < child_ts < grandchild_ts
                parent_count = parent.locator(':scope > [data-live-summary-button] [data-live-count]').first
                child_count = child.locator(':scope > [data-live-summary-button] [data-live-count]').first
                parent_text = parent.inner_text()
                child_text = child.inner_text()
                assert "Parent task started" in parent_text
                assert "2 children" in parent_count.inner_text()
                assert "researcher (child1)" in child_text
                assert "1 child" in child_count.inner_text()
                assert "child=child1" not in child_text
                assert "role=researcher" not in child_text
                assert "panel_child_review" in child_text
                assert "claude-fable-5" in child_text
                assert "verdict=DEGRADED" in child_text
                assert "evidence-mapper (grandchi" in grandchild.inner_text()
                assert child.get_attribute("data-task-id") == "child1"
                assert page.locator(
                    '.chat-live-card[data-task-id="parent1"] > .chat-subagents > '
                    '.chat-live-card.subagent[data-task-id="child1"]'
                ).count() == 1
                assert page.locator(
                    '.chat-live-card.subagent[data-task-id="child1"] > .chat-subagents > '
                    '.chat-live-card.subagent[data-task-id="grandchild1"]'
                ).count() == 1
                assert page.locator("#chat-messages > .chat-live-card.subagent").count() == 0
                assert parent.get_attribute("data-finished") == "0"
                assert rejected_child.get_attribute("data-finished") == "1"
                assert rejected_child.locator(
                    ":scope > [data-live-summary-button] [data-live-phase]"
                ).get_attribute("data-phase") == "warn"
                assert child.get_attribute("data-finished") == "1"
                assert child.locator(":scope > [data-live-summary-button] [data-live-phase]").first.get_attribute("data-phase") == "warn"
                assert child.get_attribute("data-subagent-role") == "researcher"
                child_activity_el = child.locator(":scope > [data-live-summary-button] [data-live-activity]")
                assert len(child_activity_el.text_content().strip()) <= 240
                assert child_activity_tail not in child_activity_el.text_content()
                assert child_activity_el.get_attribute("title") is None
                assert grandchild.get_attribute("data-finished") == "1"
                assert grandchild.get_attribute("data-subagent-role") == "evidence-mapper"
                assert page.locator(".chat-bubble.progress").count() == 0
                assert page.locator(".chat-bubble").filter(
                    has_text="Final child answer should stay inside the child card."
                ).count() == 0

                # Review actor/model details are disclosed immediately even on a
                # nested child; ordinary nested cards remain collapsed.
                assert child.get_attribute("data-expanded") == "1"
                assert grandchild.get_attribute("data-expanded") == "0"
                child_summary = child.locator(":scope > [data-live-summary-button]").first
                progress_line = child.locator(".chat-live-line", has_text="Searching evidence").first
                progress_toggle = progress_line.locator(".chat-live-line-toggle")
                progress_toggle.wait_for(state="visible", timeout=5_000)
                progress_toggle.click()
                assert child_activity_early in progress_line.inner_text()
                assert child_activity_tail in progress_line.inner_text()
                review_line = child.locator(".chat-live-line", has_text="panel_child_review").first
                review_toggle = review_line.locator(".chat-live-line-toggle")
                review_toggle.wait_for(state="visible", timeout=5_000)
                review_toggle.click()
                expanded_text = child.inner_text(timeout=5_000)
                assert "Final child answer should stay inside the child card." in expanded_text
                assert "Child result with evidence table" in expanded_text
                assert "| source | verdict |" in expanded_text
                assert "searched sources" in expanded_text
                assert "compared output" in expanded_text
                assert "done" in expanded_text.lower()
                assert "Scheduled subagent child1" not in expanded_text
                assert child_summary.get_attribute("aria-expanded") == "true"
                assert child.locator("[data-live-timeline]").first.get_attribute("id")
                assert review_toggle.get_attribute("aria-controls")

                page.reload(wait_until="domcontentloaded", timeout=30_000)
                page.wait_for_function("() => document.querySelectorAll('.chat-live-card').length === 4", timeout=30_000)
                page.wait_for_function(
                    "() => { const p = document.querySelector('.chat-live-card:not(.subagent)');"
                    " const c = document.querySelector('.chat-live-card.subagent[data-parent-task-id=\"parent1\"]');"
                    " const g = document.querySelector('.chat-live-card.subagent[data-parent-task-id=\"child1\"]');"
                    " return !!p && !!c && c.closest('.chat-subagents') && c.parentElement.closest('.chat-live-card') === p"
                    " && !!g && g.closest('.chat-subagents') && g.parentElement.closest('.chat-live-card') === c; }",
                    timeout=30_000,
                )
                replay_parent = page.locator(".chat-live-card:not(.subagent)").first
                replay_child = page.locator('.chat-live-card.subagent[data-task-id="child1"]')
                replay_rejected = page.locator('.chat-live-card.subagent[data-task-id="rejected1"]')
                replay_grandchild = page.locator('.chat-live-card.subagent[data-parent-task-id="child1"]').first
                assert replay_parent.get_attribute("data-finished") == "0"
                assert replay_rejected.get_attribute("data-finished") == "1"
                assert replay_rejected.locator(
                    ":scope > [data-live-summary-button] [data-live-phase]"
                ).get_attribute("data-phase") == "warn"
                assert replay_child.get_attribute("data-finished") == "1"
                assert replay_child.locator(":scope > [data-live-summary-button] [data-live-phase]").first.get_attribute("data-phase") == "warn"
                assert replay_grandchild.get_attribute("data-finished") == "1"
                assert replay_child.get_attribute("data-expanded") == "1"
                assert replay_grandchild.get_attribute("data-expanded") == "0"
                assert "researcher (child1)" in replay_child.inner_text()
                assert "child=child1" not in replay_child.inner_text()
                assert "role=researcher" not in replay_child.inner_text()
                assert "Final child answer should stay inside the child card." in replay_child.inner_text()
                replay_progress = replay_child.locator(".chat-live-line", has_text="Searching evidence").first
                replay_progress.locator(".chat-live-line-toggle").click()
                assert child_activity_early in replay_progress.inner_text()
                assert child_activity_tail in replay_progress.inner_text()
                page.wait_for_timeout(900)  # cover the routine background history sync
                assert replay_child.locator('.chat-live-line-repeat:not([hidden])').count() == 0
                page.screenshot(path=str(data_dir.parent / "review-truth-child-reconnect.png"), full_page=True)
                assert page.locator(".chat-bubble.progress").count() == 0
                assert page.locator(".chat-bubble", has_text="Final child answer should stay inside the child card.").count() == 0

                page.evaluate(
                    """async () => {
                        const resp = await fetch('/api/ui/preferences', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ nested_subagents_expanded: true }),
                        });
                        if (!resp.ok) throw new Error(await resp.text());
                    }"""
                )
                page.reload(wait_until="domcontentloaded", timeout=30_000)
                page.wait_for_function("() => document.querySelectorAll('.chat-live-card').length === 4", timeout=30_000)
                const_pref_check = (
                    "() => {"
                    " const c = document.querySelector('.chat-live-card.subagent[data-parent-task-id=\"parent1\"]');"
                    " const g = document.querySelector('.chat-live-card.subagent[data-parent-task-id=\"child1\"]');"
                    " return !!c && !!g && c.dataset.expanded === '1' && g.dataset.expanded === '1';"
                    " }"
                )
                page.wait_for_function(const_pref_check, timeout=30_000)
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise

@pytest.mark.ui_browser
def test_ui_smoke_finished_cards_keep_height_when_transcript_overflows(direct_server):
    """Regression: live cards / skill_review bubbles use overflow:hidden, which
    gives them an automatic flex min-height of 0. When the transcript column
    overflows they must NOT be shrunk to a 1px strip — the list scrolls instead.
    (rc.1 removed the inline min-height that previously masked this collapse.)"""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 600})
            try:
                page.goto(direct_server, wait_until="domcontentloaded", timeout=30_000)
                page.wait_for_selector("#chat-messages", timeout=30_000)
                result = page.evaluate(
                    """() => {
                        const messages = document.querySelector('#chat-messages');
                        messages.replaceChildren();
                        // Overflow the column with collapsed, overflow:hidden cards.
                        for (let i = 0; i < 24; i += 1) {
                            const card = document.createElement('div');
                            card.className = 'chat-live-card';
                            card.dataset.finished = '1';
                            card.dataset.expanded = '0';
                            const btn = document.createElement('div');
                            btn.className = 'chat-live-summary-button';
                            btn.style.minHeight = '48px';
                            btn.textContent = `Finished card ${i}`;
                            card.appendChild(btn);
                            messages.appendChild(card);
                        }
                        const heights = [...messages.querySelectorAll('.chat-live-card')]
                            .map((el) => Math.round(el.getBoundingClientRect().height));
                        return {
                            heights,
                            scrollHeight: messages.scrollHeight,
                            clientHeight: messages.clientHeight,
                        };
                    }"""
                )
                assert result["heights"], "no cards rendered"
                # Without flex-shrink:0 the overflow:hidden cards collapse to ~1px.
                assert min(result["heights"]) >= 40, result
                # The column should scroll rather than absorb the overflow.
                assert result["scrollHeight"] > result["clientHeight"] + 100, result
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise

@pytest.mark.ui_browser
@pytest.mark.parametrize("browser_engine", ["chromium", "webkit"])
def test_ui_smoke_live_cards_keep_usable_geometry_at_depth_and_in_project_panel(
    direct_server_with_data,
    browser_engine,
):
    """Rendered regression for the one-letter-wide nested-card failure.

    A real replayed task tree reaches the configured hard depth of ten. The
    narrow checks use geometry instead of CSS declarations, then reload to
    cover replay. Launcher-default Main and a narrow Project panel prove the
    card-local container responds to its actual consumer width rather than the
    viewport.
    """
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    from ouroboros.projects_registry import create_project

    data_dir = direct_server_with_data["data_dir"]
    url = direct_server_with_data["url"]
    project = create_project(data_dir, "layout-project", name="Layout Project")
    logs_dir = data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    rows = [{
        "ts": "2026-05-25T10:00:00+00:00",
        "chat_id": 1,
        "task_id": "layout-root",
        "content": "Root task started",
        "suggested_name": "Deep nested layout regression",
        "is_progress": True,
    }]
    long_url = "https://example.com/" + "nested-segment-without-breaks-" * 18 + "final"
    parent_id = "layout-root"
    for depth in range(1, 11):
        child_id = f"layout-child-{depth:02d}"
        rows.append({
            "ts": f"2026-05-25T10:00:{depth:02d}+00:00",
            "chat_id": 1,
            "task_id": child_id,
            "content": f"Depth {depth} subagent completed",
            "is_progress": True,
            "delegation_role": "subagent",
            "subagent_event": "completed",
            "subagent_task_id": child_id,
            "parent_task_id": parent_id,
            "root_task_id": "layout-root",
            "subagent_role": "pty-tests",
            "model": "google/gemini-3.6-flash",
            "status": "completed",
            "result": f"Depth {depth} complete. Full evidence: {long_url}",
        })
        parent_id = child_id
    panel_row = {
        "ts": "2026-05-25T10:01:00+00:00",
        "chat_id": project["chat_id"],
        "task_id": "panel-root",
        "content": "Inspecting a narrow Project panel with a long unbroken reference " + long_url,
        "suggested_name": "Narrow Project panel keeps a usable title column",
        "is_progress": True,
    }
    (logs_dir / "progress.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    mobile_geometry = """() => {
        const messages = document.querySelector('#page-chat #chat-messages');
        const messagesStyle = getComputedStyle(messages);
        const usableMessageWidth = messages.clientWidth
            - parseFloat(messagesStyle.paddingInlineStart || messagesStyle.paddingLeft || '0')
            - parseFloat(messagesStyle.paddingInlineEnd || messagesStyle.paddingRight || '0');
        const cards = [...messages.querySelectorAll('.chat-live-card')];
        const root = messages.querySelector(':scope > .chat-live-card[data-task-id="layout-root"]');
        const deepest = messages.querySelector('.chat-live-card[data-task-id="layout-child-10"]');
        const cardFacts = cards.map((card) => {
            const title = card.querySelector(':scope > .chat-live-summary-button [data-live-title]');
            const activity = card.querySelector(':scope > .chat-live-summary-button [data-live-activity]');
            const style = getComputedStyle(title);
            const lineHeight = parseFloat(style.lineHeight);
            const titleRect = title.getBoundingClientRect();
            const activityStyle = getComputedStyle(activity);
            const activityLineHeight = parseFloat(activityStyle.lineHeight);
            const activityRect = activity.getBoundingClientRect();
            return {
                id: card.dataset.taskId,
                clientWidth: card.clientWidth,
                scrollWidth: card.scrollWidth,
                titleWidth: titleRect.width,
                titleHeight: titleRect.height,
                titleLines: lineHeight > 0 ? titleRect.height / lineHeight : 99,
                activityLines: activityLineHeight > 0 ? activityRect.height / activityLineHeight : 99,
                activityTitle: activity.getAttribute('title'),
            };
        });
        const main = root.querySelector(':scope > .chat-live-summary-button .chat-live-summary-main').getBoundingClientRect();
        const side = root.querySelector(':scope > .chat-live-summary-button .chat-live-summary-side').getBoundingClientRect();
        return {
            messageWidth: usableMessageWidth,
            rootWidth: root.getBoundingClientRect().width,
            deepestWidth: deepest.getBoundingClientRect().width,
            rootMainBottom: main.bottom,
            rootSideTop: side.top,
            cardFacts,
        };
    }"""

    def assert_mobile_geometry(page):
        page.wait_for_selector(
            '#page-chat .chat-live-card[data-task-id="layout-root"]',
            state="attached",
            timeout=30_000,
        )
        page.wait_for_timeout(500)
        rendered_ids = page.locator("#page-chat .chat-live-card").evaluate_all(
            "cards => cards.map(card => card.dataset.taskId)"
        )
        assert len(rendered_ids) == 11, rendered_ids
        facts = page.evaluate(mobile_geometry)
        assert facts["rootWidth"] >= facts["messageWidth"] * 0.95, facts
        assert facts["rootWidth"] - facts["deepestWidth"] <= 40, facts
        assert facts["rootSideTop"] >= facts["rootMainBottom"] - 1, facts
        assert all(card["scrollWidth"] <= card["clientWidth"] + 1 for card in facts["cardFacts"]), facts
        assert min(card["titleWidth"] for card in facts["cardFacts"]) >= 160, facts
        assert max(card["titleLines"] for card in facts["cardFacts"]) <= 2.2, facts
        assert max(card["activityLines"] for card in facts["cardFacts"]) <= 2.2, facts
        assert all(card["activityTitle"] is None for card in facts["cardFacts"]), facts
        deepest = page.locator('.chat-live-card[data-task-id="layout-child-10"]')
        assert "pty-tests · gemini-3.6-flash" in deepest.inner_text()
        deepest.locator(":scope > [data-live-summary-button]").click()
        line_toggle = deepest.locator(":scope > [data-live-timeline] .chat-live-line-toggle").last
        line_toggle.wait_for(state="visible", timeout=5_000)
        line_toggle.click()
        expanded = deepest.evaluate(
            """card => {
                const line = card.querySelector(':scope > [data-live-timeline] .chat-live-line');
                const title = line.querySelector('.chat-live-line-title');
                return {
                    cardClient: card.clientWidth,
                    cardScroll: card.scrollWidth,
                    lineClient: line.clientWidth,
                    lineScroll: line.scrollWidth,
                    titleWidth: title.getBoundingClientRect().width,
                    text: line.innerText,
                };
            }"""
        )
        assert expanded["cardScroll"] <= expanded["cardClient"] + 1, expanded
        assert expanded["lineScroll"] <= expanded["lineClient"] + 1, expanded
        assert expanded["titleWidth"] >= 150, expanded
        assert long_url in expanded["text"], expanded

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
                mobile_context = browser.new_context(
                    viewport={"width": 390, "height": 844},
                    is_mobile=True,
                    has_touch=True,
                )
                mobile = mobile_context.new_page()
                mobile.goto(url, wait_until="domcontentloaded", timeout=30_000)
                assert_mobile_geometry(mobile)
                mobile.screenshot(
                    path=str(data_dir.parent / f"live-card-depth-10-{browser_engine}.png"),
                    full_page=True,
                )
                mobile.reload(wait_until="domcontentloaded", timeout=30_000)
                assert_mobile_geometry(mobile)
                mobile_context.close()

                wide = browser.new_page(viewport={"width": 1100, "height": 750})
                wide.goto(url, wait_until="domcontentloaded", timeout=30_000)
                wide.wait_for_selector(
                    '#page-chat .chat-live-card[data-task-id="layout-root"]',
                    state="attached",
                    timeout=30_000,
                )
                wide.wait_for_timeout(500)
                rendered_ids = wide.locator("#page-chat .chat-live-card").evaluate_all(
                    "cards => cards.map(card => card.dataset.taskId)"
                )
                assert len(rendered_ids) == 11, rendered_ids
                wide_facts = wide.evaluate(
                    """() => {
                        const ids = ['layout-root', 'layout-child-01', 'layout-child-02'];
                        return ids.map((id) => {
                            const card = document.querySelector(`#page-chat .chat-live-card[data-task-id="${id}"]`);
                            const summary = card.querySelector(':scope > .chat-live-summary-button .chat-live-summary');
                            const main = summary.querySelector('.chat-live-summary-main').getBoundingClientRect();
                            const side = summary.querySelector('.chat-live-summary-side').getBoundingClientRect();
                            const rect = card.getBoundingClientRect();
                            return {
                                id,
                                left: rect.left,
                                width: rect.width,
                                wrap: getComputedStyle(summary).flexWrap,
                                mainTop: main.top,
                                mainBottom: main.bottom,
                                sideTop: side.top,
                                sideBottom: side.bottom,
                                client: card.clientWidth,
                                scroll: card.scrollWidth,
                            };
                        });
                    }"""
                )
                assert [card["wrap"] for card in wide_facts] == ["nowrap", "nowrap", "wrap"], wide_facts
                assert wide_facts[1]["left"] - wide_facts[0]["left"] >= 30, wide_facts
                assert wide_facts[2]["left"] - wide_facts[1]["left"] >= 30, wide_facts
                assert all(card["scroll"] <= card["client"] + 1 for card in wide_facts), wide_facts
                for card in wide_facts[:2]:
                    assert min(card["mainBottom"], card["sideBottom"]) \
                        > max(card["mainTop"], card["sideTop"]), wide_facts

                with (logs_dir / "progress.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(panel_row) + "\n")
                wide.evaluate(
                    "() => document.documentElement.style.setProperty('--project-panel-width', '440px')"
                )
                project_row = wide.locator('[data-project-id="layout-project"]')
                project_row.wait_for(state="visible", timeout=30_000)
                project_row.click()
                panel_card = wide.locator(
                    '.chat-instance-panel .chat-live-card[data-task-id="panel-root"]'
                )
                panel_card.wait_for(state="visible", timeout=30_000)
                panel_facts = panel_card.evaluate(
                    """card => {
                        const panel = card.closest('.chat-instance-panel');
                        const summary = card.querySelector(':scope > .chat-live-summary-button .chat-live-summary');
                        const main = summary.querySelector('.chat-live-summary-main').getBoundingClientRect();
                        const side = summary.querySelector('.chat-live-summary-side').getBoundingClientRect();
                        const title = summary.querySelector('[data-live-title]').getBoundingClientRect();
                        return {
                            panelWidth: panel.getBoundingClientRect().width,
                            cardWidth: card.getBoundingClientRect().width,
                            cardClient: card.clientWidth,
                            cardScroll: card.scrollWidth,
                            titleWidth: title.width,
                            mainBottom: main.bottom,
                            sideTop: side.top,
                        };
                    }"""
                )
                assert panel_facts["panelWidth"] <= 560, panel_facts
                assert panel_facts["cardWidth"] >= panel_facts["panelWidth"] * 0.9, panel_facts
                assert panel_facts["cardScroll"] <= panel_facts["cardClient"] + 1, panel_facts
                assert panel_facts["titleWidth"] >= 180, panel_facts
                assert panel_facts["sideTop"] >= panel_facts["mainBottom"] - 1, panel_facts
                wide.screenshot(
                    path=str(data_dir.parent / f"live-card-project-panel-{browser_engine}.png"),
                    full_page=True,
                )
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise
