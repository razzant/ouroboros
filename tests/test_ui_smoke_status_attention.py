from __future__ import annotations

import json

import pytest

from tests.ui_chat_viewport_smoke import _OBSERVE_STATE_READS, _wait_state_reads_quiescent

pytest_plugins = ("tests.test_ui_smoke_playwright",)


@pytest.mark.ui_browser
@pytest.mark.parametrize(
    ("width", "height"),
    [(1280, 800), (390, 844)],
    ids=["desktop", "narrow"],
)
def test_task_status_stays_factual_in_main_and_project_chat(
    direct_server_with_data,
    width,
    height,
):
    """Task outcomes stay local and factual in both real Chat consumers."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    from ouroboros.projects_registry import create_project

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    project = create_project(data_dir, "status-attention", name="Status Attention")
    project_chat_id = int(project["chat_id"])
    logs_dir = data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "chat.jsonl").write_text("", encoding="utf-8")
    (logs_dir / "progress.jsonl").write_text("", encoding="utf-8")

    capture_socket = """() => {
        const NativeWebSocket = window.WebSocket;
        window.__statusAttentionTestSockets = [];
        window.WebSocket = class TestWebSocket extends NativeWebSocket {
            constructor(...args) {
                super(...args);
                window.__statusAttentionTestSockets.push(this);
            }
        };
    }"""

    def emit(page, frame):
        page.evaluate(
            """frame => {
                const socket = window.__statusAttentionTestSockets
                    ?.find(candidate => candidate.readyState === WebSocket.OPEN);
                if (!socket) throw new Error('test socket is not open');
                socket.dispatchEvent(new MessageEvent('message', {
                    data: JSON.stringify(frame),
                }));
            }""",
            frame,
        )
        page.evaluate(
            "() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))"
        )

    def emit_progress(page, chat_id, task_id, content):
        emit(page, {
            "type": "chat",
            "role": "assistant",
            "is_progress": True,
            "chat_id": chat_id,
            "task_id": task_id,
            "content": content,
            "ts": "2026-08-24T12:00:00+00:00",
        })

    def emit_task_done(page, chat_id, task_id, status):
        lifecycle = "cancelled" if status == "cancelled" else "completed"
        execution = "failed" if status == "failed" else "ok"
        objective = "fail" if status == "failed" else "pass"
        emit(page, {
            "type": "log",
            "chat_id": chat_id,
            "data": {
                "type": "task_done",
                "task_id": task_id,
                "status": status,
                "reason_code": "provider_route_failed" if status == "failed" else "",
                "outcome_axes": {
                    "lifecycle": {"status": lifecycle},
                    "execution": {"status": execution},
                    "objective": {"status": objective},
                },
                "ts": "2026-08-24T12:00:09+00:00",
            },
        })

    def direct_card(scope, task_id):
        return scope.locator(
            f'.chat-live-card:not(.subagent)[data-task-id="{task_id}"]'
        )

    def phase_node(card):
        return card.locator(
            ":scope > [data-live-summary-button] [data-live-phase]"
        )

    def phase_text(card):
        return phase_node(card).inner_text()

    def assert_phase_accessibility(card, kind, text):
        phase = phase_node(card)
        assert phase.get_attribute("role") == "status"
        assert phase.get_attribute("aria-live") == "polite"
        assert phase.get_attribute("aria-atomic") == "true"
        assert phase.get_attribute("aria-label") == f"{kind} status: {text}"

    def card_visual_state(card):
        return card.evaluate(
            """node => {
                const style = getComputedStyle(node);
                return {
                    className: node.className,
                    backgroundColor: style.backgroundColor,
                    borderColor: style.borderColor,
                    boxShadow: style.boxShadow,
                };
            }"""
        )

    def wait_for_phase(page, scope_selector, task_id, text):
        page.wait_for_function(
            """({scopeSelector, taskId, text}) => {
                const scope = document.querySelector(scopeSelector);
                const card = scope?.querySelector(
                    `.chat-live-card[data-task-id="${taskId}"]`
                );
                return card?.querySelector(':scope > [data-live-summary-button] [data-live-phase]')
                    ?.textContent === text;
            }""",
            arg={"scopeSelector": scope_selector, "taskId": task_id, "text": text},
            timeout=30_000,
        )

    def assert_card_geometry(page, scope_selector, task_ids):
        geometry = page.evaluate(
            """({scopeSelector, taskIds}) => {
                const scope = document.querySelector(scopeSelector);
                const scopeRect = scope.getBoundingClientRect();
                const cards = taskIds.map(taskId => {
                    const card = scope.querySelector(
                        `.chat-live-card[data-task-id="${taskId}"]`
                    );
                    const summary = card.querySelector(':scope > [data-live-summary-button]');
                    const phase = summary.querySelector('[data-live-phase]');
                    const title = summary.querySelector('[data-live-title]');
                    const cardRect = card.getBoundingClientRect();
                    const summaryRect = summary.getBoundingClientRect();
                    return {
                        taskId,
                        cardLeft: cardRect.left,
                        cardRight: cardRect.right,
                        summaryLeft: summaryRect.left,
                        summaryRight: summaryRect.right,
                        cardClient: card.clientWidth,
                        cardScroll: card.scrollWidth,
                        summaryClient: summary.clientWidth,
                        summaryScroll: summary.scrollWidth,
                        phaseClient: phase.clientWidth,
                        phaseScroll: phase.scrollWidth,
                        titleClient: title.clientWidth,
                        titleScroll: title.scrollWidth,
                    };
                });
                return {
                    viewportWidth: window.innerWidth,
                    documentClient: document.documentElement.clientWidth,
                    documentScroll: document.documentElement.scrollWidth,
                    scopeLeft: scopeRect.left,
                    scopeRight: scopeRect.right,
                    scopeClient: scope.clientWidth,
                    scopeScroll: scope.scrollWidth,
                    cards,
                };
            }""",
            {"scopeSelector": scope_selector, "taskIds": task_ids},
        )
        assert geometry["documentScroll"] <= geometry["documentClient"] + 1, geometry
        assert geometry["scopeScroll"] <= geometry["scopeClient"] + 1, geometry
        for card in geometry["cards"]:
            assert card["cardLeft"] >= geometry["scopeLeft"] - 1, geometry
            assert card["cardRight"] <= geometry["scopeRight"] + 1, geometry
            assert card["cardLeft"] >= -1, geometry
            assert card["cardRight"] <= geometry["viewportWidth"] + 1, geometry
            assert card["summaryLeft"] >= card["cardLeft"] - 1, geometry
            assert card["summaryRight"] <= card["cardRight"] + 1, geometry
            assert card["cardScroll"] <= card["cardClient"] + 1, geometry
            assert card["summaryScroll"] <= card["summaryClient"] + 1, geometry
            assert card["phaseScroll"] <= card["phaseClient"] + 1, geometry
            assert card["titleScroll"] <= card["titleClient"] + 1, geometry

    def run_thread_flow(
        page,
        scope,
        scope_selector,
        status_selector,
        chat_id,
        prefix,
    ):
        continuing_root = f"{prefix}-continuing"
        child_id = f"{prefix}-child"
        failed_root = f"{prefix}-failed"

        emit_progress(page, chat_id, continuing_root, "Parent continues the requested work")
        emit(page, {
            "type": "chat",
            "role": "assistant",
            "is_progress": True,
            "chat_id": chat_id,
            "task_id": child_id,
            "delegation_role": "subagent",
            "subagent_event": "scheduled",
            "subagent_task_id": child_id,
            "parent_task_id": continuing_root,
            "root_task_id": continuing_root,
            "subagent_role": "route-checker",
            "content": "Child checks the delegated route",
            "status": "scheduled",
            "ts": "2026-08-24T12:00:01+00:00",
        })

        parent = direct_card(scope, continuing_root)
        child = scope.locator(
            f'.chat-live-card.subagent[data-task-id="{child_id}"]'
        )
        parent.wait_for(state="visible", timeout=30_000)
        child.wait_for(state="visible", timeout=30_000)
        assert phase_text(parent) == "Working"
        assert phase_text(child) == "Working"
        assert_phase_accessibility(parent, "Task", "Working")
        assert_phase_accessibility(child, "Subagent", "Working")
        parent_visual_before_child_failure = card_visual_state(parent)

        emit(page, {
            "type": "chat",
            "role": "assistant",
            "is_progress": True,
            "chat_id": chat_id,
            "task_id": child_id,
            "delegation_role": "subagent",
            "subagent_event": "failed",
            "subagent_task_id": child_id,
            "parent_task_id": continuing_root,
            "root_task_id": continuing_root,
            "subagent_role": "route-checker",
            "content": "Child route finished unsuccessfully",
            "status": "failed",
            "reason_code": "delegated_route_unavailable",
            "error": "The selected tool route was unavailable; the parent can continue.",
            "ts": "2026-08-24T12:00:02+00:00",
        })

        assert child.evaluate(
            "card => card.parentElement?.closest('.chat-live-card')?.dataset.taskId"
        ) == continuing_root
        assert parent.get_attribute("data-finished") == "0"
        assert child.get_attribute("data-finished") == "1"
        assert phase_text(parent) == "Working"
        assert phase_text(child) == "Failed"
        assert_phase_accessibility(parent, "Task", "Working")
        assert_phase_accessibility(child, "Subagent", "Failed")
        assert phase_node(parent).get_attribute("data-phase") == "working"
        assert phase_node(child).get_attribute("data-phase") == "error"
        assert card_visual_state(parent) == parent_visual_before_child_failure
        assert "error" not in child.get_attribute("class").split()

        page.wait_for_function(
            "selector => document.querySelector(selector)?.textContent === 'Working...'",
            arg=f"{scope_selector} {status_selector}",
            timeout=30_000,
        )
        assert "Attention" not in scope.locator(status_selector).inner_text()
        assert page.locator("#toast-stack .toast").count() == 0
        assert page.locator('[data-nav-page="chat"] .unread-badge').count() == 0

        child_summary = child.locator(":scope > [data-live-summary-button]")
        assert child.get_attribute("data-expanded") == "0"
        child_summary.focus()
        child_summary.press("Enter")
        assert child.get_attribute("data-expanded") == "1"
        child_line_toggle = child.locator(
            ":scope > [data-live-timeline] [data-live-line-toggle]"
        ).last
        child_line_toggle.focus()
        child_line_toggle.press("Enter")
        # The card says the cause in the owner's words; the machine code stays on
        # the record half, never behind a `Reason:` label (PR #970).
        child_timeline = child.locator(":scope > [data-live-timeline]").inner_text()
        assert "The selected tool route was unavailable; the parent can continue." in child_timeline
        assert "Reason:" not in child_timeline
        assert "delegated_route_unavailable" not in child_summary.inner_text()
        child_line_toggle.press("Space")
        child_summary.press("Space")
        assert child.get_attribute("data-expanded") == "0"

        emit_task_done(page, chat_id, continuing_root, "completed")
        wait_for_phase(page, scope_selector, continuing_root, "Done")
        assert parent.get_attribute("data-finished") == "1"
        assert phase_text(parent) == "Done"
        assert phase_text(child) == "Failed"
        assert_phase_accessibility(parent, "Task", "Done")
        assert_phase_accessibility(child, "Subagent", "Failed")
        page.wait_for_function(
            "selector => document.querySelector(selector)?.textContent === 'Online'",
            arg=f"{scope_selector} {status_selector}",
            timeout=30_000,
        )

        emit_progress(page, chat_id, failed_root, "A separate root starts")
        failed = direct_card(scope, failed_root)
        failed.wait_for(state="visible", timeout=30_000)
        assert phase_text(failed) == "Working"
        assert_phase_accessibility(failed, "Task", "Working")
        emit_task_done(page, chat_id, failed_root, "failed")
        wait_for_phase(page, scope_selector, failed_root, "Failed")
        assert failed.get_attribute("data-finished") == "1"
        assert phase_text(failed) == "Failed"
        assert_phase_accessibility(failed, "Task", "Failed")
        # This synthetic code has no producer phrase. Unknown reasons remain
        # visible verbatim; hiding them would discard the only available cause.
        assert "provider_route_failed" in failed.locator(
            ":scope > [data-live-summary-button] [data-live-activity]"
        ).inner_text()
        assert "provider_route_failed" not in failed.locator(
            ":scope > [data-live-summary-button] [data-live-title]"
        ).inner_text()

        status = scope.locator(status_selector)
        page.wait_for_function(
            "selector => document.querySelector(selector)?.textContent === 'Online'",
            arg=f"{scope_selector} {status_selector}",
            timeout=30_000,
        )
        assert status.inner_text() == "Online"
        assert "Attention" not in status.inner_text()
        scope_text = " ".join(scope.locator(".chat-live-card").all_inner_texts())
        assert "Issue" not in scope_text
        assert "Notice" not in scope_text
        assert "Attention" not in " ".join(
            scope.locator("[data-live-phase]").all_inner_texts()
        )
        assert page.locator("#toast-stack .toast").count() == 0
        assert page.locator('[data-nav-page="chat"] .unread-badge').count() == 0
        for card in (child, failed):
            actions = " ".join(card.locator("button").all_inner_texts())
            for manufactured in ("Retry", "Resume", "Repair", "Reconnect", "Grant access"):
                assert manufactured not in actions

        task_ids = [continuing_root, child_id, failed_root]
        assert_card_geometry(page, scope_selector, task_ids)

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": width, "height": height})
            try:
                page.add_init_script(f"({capture_socket})()")
                page.add_init_script(f"({_OBSERVE_STATE_READS})()")
                # The initial rebuildAll replay wipes and rebuilds the feed
                # from durable history (chat.js syncHistory). Frames emitted
                # on the test socket exist nowhere durable, so one emitted
                # before that replay applies is destroyed with the wipe. Gate
                # the flow on the hydration fetch itself (body finished), then
                # settle one page task: the apply is synchronous after
                # resp.json(), so the next macrotask observes it.
                with page.expect_response(
                    lambda response: "/api/chat/history" in response.url,
                    timeout=30_000,
                ) as history_hydration:
                    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                history_hydration.value.finished()
                page.wait_for_function(
                    "() => window.__statusAttentionTestSockets"
                    "?.some(socket => socket.readyState === WebSocket.OPEN)",
                    timeout=30_000,
                )
                page.evaluate(
                    "() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))"
                )
                # The frames below exist only on the test socket. Let the
                # socket-open census land before emitting them: a complete
                # census whose request starts after a frame concludes that card
                # by absence, exactly as it would a task the queue really lost.
                _wait_state_reads_quiescent(page)

                run_thread_flow(
                    page,
                    page.locator("#page-chat"),
                    "#page-chat",
                    "#chat-status",
                    1,
                    "main",
                )

                project_row = page.locator(
                    '.nav-project-row[data-project-id="status-attention"]'
                )
                if width <= 640 or not project_row.is_visible():
                    page.locator("#page-chat [data-mobile-nav-toggle]").click()
                    project_row.wait_for(state="visible", timeout=5_000)
                project_row.scroll_into_view_if_needed()
                # Same hydration gate for the Project instance: its chat
                # column mints on open and replays its own durable history;
                # emits must not race that rebuild (see the Main gate above).
                with page.expect_response(
                    lambda response: f"/api/chat/history?chat_id={project_chat_id}"
                    in response.url,
                    timeout=30_000,
                ) as project_hydration:
                    with page.expect_response(
                        lambda response: response.url.endswith("/api/ui/preferences")
                        and response.request.method == "POST",
                        timeout=30_000,
                    ):
                        project_row.click()
                project_hydration.value.finished()
                page.wait_for_selector("#project-panel:not([hidden])", timeout=30_000)
                project_scope = page.locator("#project-panel .chat-instance-panel")
                project_scope.wait_for(state="visible", timeout=30_000)
                page.evaluate(
                    "() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))"
                )
                # The panel's own hydrating census read (forced on mount) may
                # start behind an in-flight page read; let it land before the
                # panel's synthetic frames, for the same reason as in Main.
                _wait_state_reads_quiescent(page)
                run_thread_flow(
                    page,
                    project_scope,
                    "#project-panel .chat-instance-panel",
                    f"#pchat-{project['id']}-status",
                    project_chat_id,
                    "project",
                )

                assert page.locator("#toast-stack .toast").count() == 0
                assert page.locator('[data-nav-page="chat"] .unread-badge').count() == 0
                assert project_row.locator(".nav-unread-dot").count() == 0
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.mark.ui_browser
def test_history_replay_keeps_finalizing_and_finishes_bare_final(
    direct_server_with_data,
):
    """Open summaries stay live; a keyed final without summary falls back to Done.

    #1110: a finalizing task whose lifecycle already settled paints that KNOWN
    outcome as the chip and holds "Finalizing…" as the secondary fact — the card
    stays unfinished until the settled task_done lands.
    """
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    messages = [
        {
            "role": "assistant", "is_progress": True, "task_id": "replay-finalizing",
            "text": "Post-task synthesis is still running.",
            "content": "Post-task synthesis is still running.",
            "task_phase": "finalizing", "ts": "2026-08-17T10:00:00+00:00",
        },
        {
            "role": "system", "system_type": "task_summary",
            "task_id": "replay-finalizing", "status": "completed",
            "outcome_final": False, "task_phase": "finalizing",
            "tool_calls": 1, "rounds": 2, "text": "Authored summary.",
            "ts": "2026-08-17T10:00:01+00:00",
        },
        {
            "role": "assistant", "task_id": "replay-finalizing",
            "task_phase": "finalizing", "text": "Early answer.",
            "ts": "2026-08-17T10:00:02+00:00",
        },
        {
            "role": "assistant", "is_progress": True, "task_id": "replay-bare-final",
            "text": "Last retained progress.", "content": "Last retained progress.",
            "ts": "2026-08-17T10:01:00+00:00",
        },
        {
            "role": "assistant", "task_id": "replay-bare-final",
            "text": "Retained final answer.", "ts": "2026-08-17T10:01:01+00:00",
        },
    ]
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            try:
                page.route(
                    "**/api/chat/history*",
                    lambda route: route.fulfill(
                        content_type="application/json",
                        body=json.dumps({"messages": messages}),
                    ),
                )
                page.goto(
                    direct_server_with_data["url"],
                    wait_until="domcontentloaded",
                    timeout=30_000,
                )

                open_card = page.locator(
                    '.chat-live-card[data-task-id="replay-finalizing"]'
                )
                done_card = page.locator(
                    '.chat-live-card[data-task-id="replay-bare-final"]'
                )
                open_card.wait_for(state="visible", timeout=10_000)
                done_card.wait_for(state="visible", timeout=10_000)

                assert open_card.get_attribute("data-finished") == "0"
                assert open_card.locator(".chat-live-phase").inner_text().strip() == "Done"
                # The hold is the SECONDARY fact beside the outcome; the chip's accessible
                # name states both (task_phase_chip.setLiveCardPhase). This replay card has
                # no work rows, so its chip is hidden and the visible secondary is proven by
                # tests/test_ui_failed_finalizing_browser.py on a real card with work.
                assert open_card.locator(".chat-live-phase").get_attribute("aria-label") == "Task status: Done, Finalizing…"
                assert done_card.get_attribute("data-finished") == "1"
                assert done_card.locator(".chat-live-phase").inner_text().strip() == "Done"
                assert "Finalizing" not in (done_card.locator(".chat-live-phase").get_attribute("aria-label") or "")
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise
