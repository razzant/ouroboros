"""Production-shaped Chat viewport acceptance, split from the giant UI smoke module."""

from __future__ import annotations

import json

import pytest

_CAPTURE_TEST_SOCKET = """() => {
    const NativeWebSocket = window.WebSocket;
    window.__testSockets = [];
    window.WebSocket = class TestWebSocket extends NativeWebSocket {
        constructor(...args) {
            super(...args);
            window.__testSockets.push(this);
        }
    };
}"""
_SETTLE_TWO_FRAMES = "() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))"
_SETTLE_RESTORE_FRAMES = """() => new Promise(resolve => {
    let remaining = 14;
    const tick = () => { if (--remaining <= 0) resolve(); else requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
})"""


def _emit_ws_frame(page, frame):
    page.evaluate(
        """frame => {
            const socket = window.__testSockets?.[0];
            if (!socket) throw new Error('test socket not captured');
            socket.dispatchEvent(new MessageEvent('message', {data: JSON.stringify(frame)}));
        }""",
        frame,
    )
    page.evaluate(_SETTLE_TWO_FRAMES)


def run_chat_viewport_smoke(
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

    def set_remaining(page, target, *, dispatch=True):
        result = page.evaluate(
            """({target, dispatch}) => {
                const messages = document.querySelector('#chat-messages');
                messages.scrollTop = Math.max(0, messages.scrollHeight - messages.clientHeight - target);
                if (dispatch) messages.dispatchEvent(new Event('scroll'));
                return {scrollTop: messages.scrollTop,
                    remaining: messages.scrollHeight - messages.scrollTop - messages.clientHeight};
            }""",
            {"target": target, "dispatch": dispatch},
        )
        page.evaluate(_SETTLE_TWO_FRAMES)
        return result

    def jump_state(page):
        return page.evaluate(
            """() => {
                const messages = document.querySelector('#chat-messages');
                const button = document.querySelector('#chat-scroll-bottom');
                const dot = button.querySelector('.chat-scroll-activity-dot');
                return {
                    remaining: messages.scrollHeight - messages.scrollTop - messages.clientHeight,
                    scrollTop: messages.scrollTop,
                    visible: button.classList.contains('visible'),
                    dotHidden: dot.hidden,
                    dotCount: button.querySelectorAll('.chat-scroll-activity-dot').length,
                    dotAriaHidden: dot.getAttribute('aria-hidden'),
                    dotLive: dot.hasAttribute('aria-live'),
                    label: button.getAttribute('aria-label'),
                    title: button.getAttribute('title'),
                };
            }"""
        )

    def begin_noop_read(page):
        set_remaining(page, 0)
        return set_remaining(page, 40)["scrollTop"]

    def assert_noop_read(page, before):
        state = jump_state(page)
        assert abs(state["scrollTop"] - before) <= 1, state
        assert abs(state["remaining"] - 40) <= 2 and state["dotHidden"], state

    def assert_noop_frame(page, frame):
        before = begin_noop_read(page)
        _emit_ws_frame(page, frame)
        assert_noop_read(page, before)

    def hold_first_route(routes):
        return lambda route: routes.append(route) if not routes else route.fallback()

    def visible_card_anchor(page):
        return page.evaluate(
            """() => {
                const messages = document.querySelector('#chat-messages');
                const box = messages.getBoundingClientRect();
                const card = [...messages.querySelectorAll(':scope > .chat-live-card')]
                    .find(node => {
                        const rect = node.getBoundingClientRect();
                        return rect.bottom > box.top && rect.top < box.bottom;
                    });
                if (!card) throw new Error('no visible top-level card');
                return {id: card.dataset.taskId, top: card.getBoundingClientRect().top};
            }"""
        )

    try:
        with sync_playwright() as pw:
            browser_type = getattr(pw, browser_engine)
            try:
                browser = browser_type.launch(headless=True)
            except PlaywrightError as exc:
                if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
                    pytest.fail(f"required Playwright {browser_engine} browser is not installed: {exc}")
                raise
            page = browser.new_page(viewport={"width": 1280, "height": 760})
            try:
                page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                page.add_style_tag(
                    content="#chat-messages, #chat-messages * { overflow-anchor: none !important; }"
                )
                assert page.locator("#chat-messages").evaluate(
                    "node => getComputedStyle(node).overflowAnchor"
                ) == "none"
                page.wait_for_function(
                    "() => window.__testSockets?.some(socket => socket.readyState === WebSocket.OPEN)",
                    timeout=30_000,
                )
                page.wait_for_function(
                    "() => document.querySelector('#chat-messages')?.innerText.includes('Ouroboros has awakened')",
                    timeout=30_000,
                )
                # Threshold assertions start after the page-show restore lease;
                # WebKit otherwise applies its final scheduled pin mid-scenario.
                page.evaluate(_SETTLE_RESTORE_FRAMES)

                _emit_ws_frame(page, {
                    "type": "chat", "role": "assistant", "is_progress": True,
                    "chat_id": 1, "task_id": "vp-parent",
                    "content": "Parent begins", "ts": "2026-08-03T10:00:00+00:00",
                })
                _emit_ws_frame(page, {
                    "type": "chat", "role": "assistant", "is_progress": True,
                    "chat_id": 1, "task_id": "vp-review",
                    "content": "Review target begins", "ts": "2026-08-03T10:00:00.500000+00:00",
                })
                for idx in range(1, 5):
                    _emit_ws_frame(page, {
                        "type": "chat", "role": "assistant", "is_progress": True,
                        "chat_id": 1, "task_id": f"vp-child-{idx}",
                        "delegation_role": "subagent", "subagent_event": "scheduled",
                        "subagent_task_id": f"vp-child-{idx}", "parent_task_id": "vp-parent",
                        "root_task_id": "vp-parent", "subagent_role": f"reader-{idx}",
                        "content": f"Child {idx} scheduled",
                        "ts": f"2026-08-03T10:00:0{idx}+00:00",
                    })
                for idx in range(1, 11):
                    _emit_ws_frame(page, {
                        "type": "chat", "role": "assistant", "is_progress": True,
                        "chat_id": 1, "task_id": f"vp-follow-{idx}",
                        "content": (f"Following task {idx} " * 14),
                        "ts": f"2026-08-03T10:01:{idx:02d}+00:00",
                    })
                page.wait_for_selector('.chat-live-card[data-task-id="vp-follow-10"]', timeout=30_000)
                _emit_ws_frame(page, {
                    "type": "chat", "role": "user", "chat_id": 1,
                    "client_message_id": "vp-routing", "sender_session_id": "routing-test",
                    "content": "Route this existing message", "ts": "2026-08-03T10:00:00.250000+00:00",
                })
                assert page.locator('[data-client-message-id="vp-routing"]').evaluate(
                    "node => node.previousElementSibling?.dataset.taskId === 'vp-parent' && node.nextElementSibling?.dataset.taskId === 'vp-review'"
                )

                # Merely leaving the live edge shows navigation, not a false marker.
                assert abs(set_remaining(page, 300)["remaining"] - 300) <= 2
                state = jump_state(page)
                assert state["visible"] and state["dotHidden"], state
                assert state["label"] == state["title"] == "Scroll to latest message", state

                # The visible pre-mutation distance is the only live-follow truth.
                for case, target in enumerate((0, 40)):
                    assert abs(set_remaining(page, target)["remaining"] - target) <= 2
                    _emit_ws_frame(page, {
                        "type": "chat", "role": "assistant", "is_progress": True,
                        "chat_id": 1, "task_id": f"vp-threshold-follow-{case}",
                        "content": f"Follow at {target}px " * 18,
                        "ts": f"2026-08-03T10:02:0{case}+00:00",
                    })
                    state = jump_state(page)
                    assert state["remaining"] <= 6 and state["dotHidden"], state

                for case, target in enumerate((49, 300)):
                    set_remaining(page, 0)
                    assert jump_state(page)["dotHidden"]
                    assert abs(set_remaining(page, target)["remaining"] - target) <= 2
                    anchor = visible_card_anchor(page)
                    _emit_ws_frame(page, {
                        "type": "chat", "role": "assistant", "is_progress": True,
                        "chat_id": 1, "task_id": f"vp-threshold-freeze-{case}",
                        "content": f"Freeze at {target}px " * 18,
                        "ts": f"2026-08-03T10:02:1{case}+00:00",
                    })
                    after_top = card_top(page, anchor["id"])
                    state = jump_state(page)
                    assert abs(after_top - anchor["top"]) <= 6, (anchor, after_top, state)
                    assert state["remaining"] > 48 and state["visible"], state
                    assert not state["dotHidden"] and state["dotCount"] == 1, state
                    assert state["dotAriaHidden"] == "true" and not state["dotLive"], state
                    assert state["label"] == state["title"] == (
                        "New activity — scroll to latest message"
                    ), state

                    # Entering the 48px follow zone does not clear unseen activity;
                    # only the actual 6px landing tolerance does.
                    set_remaining(page, 40)
                    near = jump_state(page)
                    assert not near["dotHidden"] and not near["visible"], near
                    set_remaining(page, 0)
                    assert jump_state(page)["dotHidden"]

                # `_savedStick` is deliberately stale here: scroll and delivery
                # happen in one JS turn, before a native scroll event can repair it.
                set_remaining(page, 0)
                stale = page.evaluate(
                    """frame => {
                        const messages = document.querySelector('#chat-messages');
                        messages.scrollTop = Math.max(
                            0, messages.scrollHeight - messages.clientHeight - 300
                        );
                        const box = messages.getBoundingClientRect();
                        const anchor = [...messages.querySelectorAll(':scope > .chat-live-card')]
                            .find(node => {
                                const rect = node.getBoundingClientRect();
                                return rect.bottom > box.top && rect.top < box.bottom;
                            });
                        window.__testSockets[0].dispatchEvent(new MessageEvent('message', {
                            data: JSON.stringify(frame),
                        }));
                        return {id: anchor.dataset.taskId, top: anchor.getBoundingClientRect().top};
                    }""",
                    {
                        "type": "chat", "role": "assistant", "is_progress": True,
                        "chat_id": 1, "task_id": "vp-stale-stick",
                        "content": "Stale saved intent must not win " * 16,
                        "ts": "2026-08-03T10:02:20+00:00",
                    },
                )
                page.evaluate(_SETTLE_TWO_FRAMES)
                assert abs(card_top(page, stale["id"]) - stale["top"]) <= 6
                assert not jump_state(page)["dotHidden"]

                button = page.locator("#chat-scroll-bottom")
                button.click()
                page.evaluate(_SETTLE_TWO_FRAMES)
                assert jump_state(page)["remaining"] <= 6
                assert jump_state(page)["dotHidden"]

                # Native keyboard activation uses the same explicit landing path.
                set_remaining(page, 300)
                _emit_ws_frame(page, {
                    "type": "chat", "role": "assistant", "is_progress": True,
                    "chat_id": 1, "task_id": "vp-keyboard-jump",
                    "content": "Keyboard jump target", "ts": "2026-08-03T10:02:21+00:00",
                })
                button.focus()
                button.press("Enter")
                page.evaluate(_SETTLE_TWO_FRAMES)
                state = jump_state(page)
                assert state["remaining"] <= 6 and state["dotHidden"], state
                assert button.evaluate("node => document.activeElement === node")

                # Duplicate and intentionally hidden decision frames are no-ops,
                # not new visible activity.
                duplicate = {
                    "type": "chat", "role": "user", "chat_id": 1,
                    "client_message_id": "vp-duplicate-user",
                    "sender_session_id": "remote-duplicate-test",
                    "content": "Exactly once", "ts": "2026-08-03T10:02:22+00:00",
                }
                set_remaining(page, 300)
                _emit_ws_frame(page, duplicate)
                button.click()
                page.evaluate(_SETTLE_TWO_FRAMES)
                set_remaining(page, 300)
                _emit_ws_frame(page, duplicate)
                state = jump_state(page)
                assert state["remaining"] > 48 and state["dotHidden"], state
                _emit_ws_frame(page, {
                    "type": "chat", "role": "assistant", "is_progress": True,
                    "ephemeral_decision": True, "chat_id": 1,
                    "task_id": "vp-hidden-decision", "content": "Hidden decision",
                    "ts": "2026-08-03T10:02:23+00:00",
                })
                assert jump_state(page)["dotHidden"]
                _emit_ws_frame(page, {
                    "type": "chat", "role": "assistant", "is_progress": True, "chat_id": 1,
                    "task_id": "vp-hidden-decision", "content": "Late hidden decision",
                })
                assert page.locator('[data-task-id="vp-hidden-decision"]').count() == 0 and jump_state(page)["dotHidden"]

                # Browser visibility is a lifecycle seam. A hidden pinned
                # reader re-follows; a hidden history reader keeps its saved
                # numeric position and receives the coalesced activity bit.
                set_remaining(page, 0)
                page.evaluate(
                    """() => {
                        window.__testDocumentHidden = true;
                        Object.defineProperty(document, 'hidden', {
                            configurable: true,
                            get: () => window.__testDocumentHidden,
                        });
                        document.dispatchEvent(new Event('visibilitychange'));
                    }"""
                )
                _emit_ws_frame(page, {
                    "type": "chat", "role": "assistant", "is_progress": True,
                    "chat_id": 1, "task_id": "vp-hidden-pinned",
                    "content": "Hidden pinned update", "ts": "2026-08-03T10:02:24+00:00",
                })
                page.evaluate(
                    """() => {
                        window.__testDocumentHidden = false;
                        document.dispatchEvent(new Event('visibilitychange'));
                    }"""
                )
                page.evaluate(_SETTLE_RESTORE_FRAMES)
                state = jump_state(page)
                assert state["remaining"] <= 6 and state["dotHidden"], state

                set_remaining(page, 300)
                hidden_top = page.locator("#chat-messages").evaluate("node => node.scrollTop")
                page.evaluate(
                    """() => {
                        window.__testDocumentHidden = true;
                        document.dispatchEvent(new Event('visibilitychange'));
                    }"""
                )
                _emit_ws_frame(page, {
                    "type": "chat", "role": "assistant", "is_progress": True,
                    "chat_id": 1, "task_id": "vp-hidden-reader",
                    "content": "Hidden history update", "ts": "2026-08-03T10:02:25+00:00",
                })
                page.evaluate(
                    """() => {
                        window.__testDocumentHidden = false;
                        document.dispatchEvent(new Event('visibilitychange'));
                    }"""
                )
                page.evaluate(_SETTLE_RESTORE_FRAMES)
                restored_top = page.locator("#chat-messages").evaluate("node => node.scrollTop")
                assert abs(restored_top - hidden_top) <= 2
                assert not jump_state(page)["dotHidden"]
                button.click()
                page.evaluate(_SETTLE_TWO_FRAMES)

                # The newly landed routing picker is another height writer on
                # an existing bubble. Remote refusal marks activity; local
                # show-all and the resulting receipt preserve position quietly.
                set_remaining(page, 300)
                routing_anchor = visible_card_anchor(page)
                routing_frame = {
                    "type": "message_annotation", "annotation_type": "routing_ack",
                    "chat_id": 1, "client_message_id": "vp-routing",
                    "status": "needs_manual_target", "routing_token": "vp-route-token",
                    "options": [
                        {"action": "steer_task", "task_id": f"vp-route-{idx}", "title": f"Route {idx}"}
                        for idx in range(10)
                    ],
                }
                _emit_ws_frame(page, routing_frame)
                routing_card = page.locator('.chat-routing-card[data-routing-token="vp-route-token"]')
                routing_card.wait_for(state="attached", timeout=10_000)
                assert abs(card_top(page, routing_anchor["id"]) - routing_anchor["top"]) <= 6
                assert not jump_state(page)["dotHidden"]
                button.click()
                page.evaluate(_SETTLE_TWO_FRAMES)
                set_remaining(page, 300)
                routing_anchor = visible_card_anchor(page)
                routing_card.locator('.chat-quiz-more').evaluate("el => el.click()")
                page.evaluate(_SETTLE_TWO_FRAMES)
                assert abs(card_top(page, routing_anchor["id"]) - routing_anchor["top"]) <= 6
                assert jump_state(page)["dotHidden"]
                _emit_ws_frame(page, {
                    "type": "message_annotation", "annotation_type": "routing_ack",
                    "chat_id": 1, "client_message_id": "vp-routing", "status": "delivered",
                    "action": "steer_task", "target": "vp-route-0", "target_label": "Route 0",
                })
                assert abs(card_top(page, routing_anchor["id"]) - routing_anchor["top"]) <= 6
                assert jump_state(page)["dotHidden"]
                assert_noop_frame(page, {
                    "type": "message_annotation", "annotation_type": "routing_ack",
                    "chat_id": 1, "client_message_id": "vp-routing", "status": "delivered",
                    "action": "steer_task", "target": "vp-route-0", "target_label": "Route 0",
                })

                # Same-value lifecycle frames must perform no connected DOM
                # writes: WebKit can otherwise move the viewport on the write.
                set_remaining(page, 0)
                _emit_ws_frame(page, {
                    "type": "quiz", "chat_id": 1, "quiz_id": "vp-quiz",
                    "task_id": "vp-quiz-task", "question": "Keep reading?",
                    "state": "open", "options": [{"label": "Yes"}, {"label": "No"}],
                    "ts": "2026-08-03T10:00:00.250000+00:00",
                })
                quiz_state = {
                    "type": "quiz_state", "quiz_id": "vp-quiz",
                    "task_id": "vp-quiz-task", "state": "answered", "answered_index": 0,
                }
                _emit_ws_frame(page, quiz_state)
                assert_noop_frame(page, quiz_state)
                set_remaining(page, 300)

                # Cleanup is the inverse no-op case: a duplicate final bubble
                # still removes a pending routing annotation, so that real height
                # change must preserve the reader and mark remote activity.
                _emit_ws_frame(page, {
                    "type": "chat", "role": "user", "chat_id": 1,
                    "client_message_id": "vp-routing-cleanup", "sender_session_id": "routing-test",
                    "content": "Route cleanup target", "ts": "2026-08-03T10:00:00.300000+00:00",
                })
                duplicate_final = {
                    "type": "chat", "role": "assistant", "chat_id": 1,
                    "task_id": "vp-routing-cleanup-final", "content": "Routing settled",
                    "ts": "2026-08-03T10:00:00.400000+00:00",
                }
                _emit_ws_frame(page, duplicate_final)
                _emit_ws_frame(page, {
                    "type": "message_annotation", "annotation_type": "routing_ack",
                    "chat_id": 1, "client_message_id": "vp-routing-cleanup",
                    "status": "pending",
                })
                assert page.locator(
                    '.msg-routing-annotation[data-annotation-status="pending"]'
                ).count() == 1
                button.click()
                page.evaluate(_SETTLE_TWO_FRAMES)
                set_remaining(page, 300)
                routing_anchor = visible_card_anchor(page)
                _emit_ws_frame(page, duplicate_final)
                assert abs(card_top(page, routing_anchor["id"]) - routing_anchor["top"]) <= 6
                assert not jump_state(page)["dotHidden"]
                assert page.locator(
                    '.msg-routing-annotation[data-annotation-status="pending"]'
                ).count() == 0
                button.click()
                page.evaluate(_SETTLE_TWO_FRAMES)

                # Local composer reflow preserves reading position and never
                # announces remote activity.
                set_remaining(page, 300)
                composer_anchor = visible_card_anchor(page)
                page.fill("#chat-input", "one\ntwo\nthree\nfour\nfive")
                page.evaluate(_SETTLE_TWO_FRAMES)
                composer_after = card_top(page, composer_anchor["id"])
                assert abs(composer_after - composer_anchor["top"]) <= 6, (
                    composer_anchor, composer_after, jump_state(page),
                )
                assert jump_state(page)["dotHidden"]
                page.fill("#chat-input", "")
                page.evaluate(_SETTLE_TWO_FRAMES)

                # The reader is inside a child. Naming the parent and growing its
                # visible timeline above that child must keep the child stationary.
                parent = page.locator('.chat-live-card[data-task-id="vp-parent"]')
                parent.locator(':scope > [data-live-summary-button]').click()
                child_selector = '.chat-live-card[data-task-id="vp-child-2"] > [data-live-summary-button]'
                mid = put_at_viewport_top(page, child_selector)
                page.evaluate(_SETTLE_TWO_FRAMES)
                assert mid["remaining"] > 160, mid
                child_before = page.locator(child_selector).evaluate("el => el.getBoundingClientRect().top")
                parent_height_before = parent.evaluate("card => card.getBoundingClientRect().height")
                parent_name = {
                    "type": "task_named", "task_id": "vp-parent",
                    "suggested_name": "A deliberately long generated project name " * 12,
                }
                _emit_ws_frame(page, parent_name)
                child_after_name = page.locator(child_selector).evaluate("el => el.getBoundingClientRect().top")
                parent_height_named = parent.evaluate("card => card.getBoundingClientRect().height")
                assert parent_height_named > parent_height_before + 20
                assert abs(child_after_name - child_before) <= 6

                for idx in range(8):
                    _emit_ws_frame(page, {
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
                page.evaluate(_SETTLE_TWO_FRAMES)
                assert mid["remaining"] > 160, mid
                anchor_before = card_top(page, anchor_id)
                parent_before_mount = parent.evaluate("card => card.getBoundingClientRect().height")
                late_child_frame = {
                    "type": "chat", "role": "assistant", "is_progress": True,
                    "chat_id": 1, "task_id": "vp-late-child",
                    "delegation_role": "subagent", "subagent_event": "scheduled",
                    "subagent_task_id": "vp-late-child", "parent_task_id": "vp-parent",
                    "root_task_id": "vp-parent", "subagent_role": "late-reader",
                    "content": "Late child mounted above the reader",
                    "ts": "2026-08-03T10:04:00+00:00",
                }
                _emit_ws_frame(page, late_child_frame)
                assert parent.evaluate("card => card.getBoundingClientRect().height") > parent_before_mount + 30
                assert abs(card_top(page, anchor_id) - anchor_before) <= 6
                parent.evaluate("""card => { window.__subagentNoopMutations = []; window.__subagentNoopObserver = new MutationObserver(records => window.__subagentNoopMutations.push(...records)); window.__subagentNoopObserver.observe(card, {attributes: true, attributeOldValue: true, childList: true, characterData: true, subtree: true}); }""")
                assert_noop_frame(page, late_child_frame)
                noop_mutations = page.evaluate("""() => { window.__subagentNoopObserver.disconnect(); return window.__subagentNoopMutations.map(record => `${record.type}:${record.target.dataset?.taskId || record.target.dataset?.subagentsFor || record.target.className}:${record.attributeName || ''}:${record.oldValue || ''}->${record.target.getAttribute?.(record.attributeName) || ''}`); }""")
                assert noop_mutations == [], noop_mutations
                put_at_viewport_top(page, anchor_selector)
                enriched_child = {
                    **late_child_frame,
                    "review_projection": {"panels": [{
                        "panel_id": "scheduled-review", "surface": "task_acceptance",
                        "aggregate_signal": "PASS", "reason": "late review evidence", "actors": [],
                    }]},
                }
                anchor_before = card_top(page, anchor_id)
                _emit_ws_frame(page, enriched_child)
                assert page.locator(
                    '.chat-live-card[data-task-id="vp-late-child"] [data-live-review-summary]'
                ).text_content() == "Reviews 1"
                assert abs(card_top(page, anchor_id) - anchor_before) <= 6
                assert not jump_state(page)["dotHidden"]
                button.click()
                page.evaluate(_SETTLE_TWO_FRAMES)
                put_at_viewport_top(page, anchor_selector)
                anchor_before = card_top(page, anchor_id)

                parent_before_finish = parent.evaluate("card => card.getBoundingClientRect().height")
                parent_summary = {
                    "type": "chat", "role": "system", "system_type": "task_summary",
                    "chat_id": 1, "task_id": "vp-parent", "content": "Parent completed",
                    "ts": "2026-08-03T10:05:00+00:00",
                    "outcome_axes": {
                        "lifecycle": {"status": "completed"}, "execution": {"status": "ok"},
                        "objective": {"status": "pass"}, "review": {"status": "pass"},
                        "artifacts": {"status": "ready"},
                    },
                }
                _emit_ws_frame(page, parent_summary)
                assert parent.get_attribute("data-expanded") == "1"
                # A terminal summary is legally shorter than live narration by a
                # couple of wrapped lines; three --type-body line boxes
                # (14px * 1.45 * 3 ~ 61) bound that legitimate shrink after the
                # owner-approved chat scale migration, while a collapsed
                # reserved band would shrink far more and still fail here.
                assert parent.evaluate("card => card.getBoundingClientRect().height") >= parent_before_finish - 61
                assert abs(card_top(page, anchor_id) - anchor_before) <= 6
                assert_noop_frame(page, parent_summary)
                assert_noop_frame(page, parent_name)

                # Internal metrics and a stable-key task_done may reconcile
                # state, but their exact repeats cannot become scroll authors.
                _emit_ws_frame(page, {"type": "log", "data": {
                    "chat_id": 1, "type": "task_started", "task_id": "vp-title-noop",
                }})
                terminal = {"type": "chat", "role": "assistant", "chat_id": 1,
                            "task_id": "vp-title-noop", "task_terminal_status": "completed",
                            "content": "Title terminal"}
                _emit_ws_frame(page, terminal)
                assert_noop_frame(page, terminal)
                _emit_ws_frame(page, {
                    "type": "chat", "role": "assistant", "is_progress": True,
                    "chat_id": 1, "task_id": "vp-log-noop", "content": "Log target",
                    "ts": "2026-08-03T10:05:01+00:00",
                })
                metric = {"type": "log", "data": {
                    "chat_id": 1, "type": "task_metrics_event", "task_id": "vp-log-noop",
                    "tool_calls": 1, "tool_errors": 0, "ts": "2026-08-03T10:05:02+00:00",
                }}
                _emit_ws_frame(page, metric)
                assert_noop_frame(page, metric)
                error_frame = {"type": "log", "data": {
                    "chat_id": 1, "type": "tool_timeout", "task_id": "vp-log-noop",
                    "tool": "browser", "error": "timed out",
                    "ts": "2026-08-03T10:05:02.500000+00:00",
                }}
                _emit_ws_frame(page, error_frame)
                assert_noop_frame(page, error_frame)
                task_done = {"type": "log", "data": {
                    "chat_id": 1, "type": "task_done", "task_id": "vp-log-noop",
                    "status": "completed", "ts": "2026-08-03T10:05:03+00:00",
                }}
                _emit_ws_frame(page, task_done)
                assert_noop_frame(page, task_done)

                # A stable-key repeat can add only the newly-authorized Stop
                # control. That genuine remote geometry change still preserves
                # the reader and sets the single activity bit.
                cancel_authority = {
                    "type": "chat", "role": "assistant", "is_progress": True,
                    "chat_id": 1, "task_id": "vp-cancel-authority",
                    "content": "Authority target", "ts": "2026-08-03T10:05:03.500000+00:00",
                }
                set_remaining(page, 0)
                _emit_ws_frame(page, cancel_authority)
                assert page.locator(
                    '.chat-live-card[data-task-id="vp-cancel-authority"] [data-cancel-run]'
                ).count() == 0
                cancel_active_ids = page.locator(
                    '.chat-live-card[data-finished="0"]'
                ).evaluate_all("cards => cards.map(card => card.dataset.taskId)")
                cancel_active_ids.append("vp-cancel-noop")
                page.route(
                    "**/api/state",
                    lambda route: route.fulfill(
                        status=200,
                        content_type="application/json",
                        body=json.dumps({"active_chat_activities": [{
                            "activity_id": task_id, "task_id": task_id, "chat_id": 1,
                            "kind": "managed_task", "phase": "working",
                        } for task_id in cancel_active_ids]}),
                    ),
                )
                set_remaining(page, 300)
                authority_anchor = visible_card_anchor(page)
                _emit_ws_frame(page, {**cancel_authority, "cancelable": True})
                assert page.locator(
                    '.chat-live-card[data-task-id="vp-cancel-authority"] [data-cancel-run]'
                ).count() == 1
                assert abs(card_top(page, authority_anchor["id"]) - authority_anchor["top"]) <= 6
                assert not jump_state(page)["dotHidden"]
                button.click()
                page.evaluate(_SETTLE_TWO_FRAMES)
                page.wait_for_timeout(800)  # drain the prior task_done history debounce

                # A delayed, unchanged pending detail only re-enables the
                # control. Disabled styling has no transcript geometry and
                # must not consume the 40px follow zone after the await.
                held_cancel_details = []
                page.route("**/api/tasks/vp-cancel-noop", hold_first_route(held_cancel_details))
                page.route(
                    "**/api/tasks/vp-cancel-noop/cancel",
                    lambda route: route.fulfill(status=202, content_type="application/json", body=json.dumps({
                        "task_id": "vp-cancel-noop", "cancel_state": "pending",
                        "stop_policy": "finalize_then_cancel",
                    })),
                )
                _emit_ws_frame(page, {
                    "type": "chat", "role": "assistant", "is_progress": True, "cancelable": True,
                    "chat_id": 1, "task_id": "vp-cancel-noop", "content": "Cancelable target",
                    "ts": "2026-08-03T10:05:04+00:00",
                })
                page.locator(
                    '.chat-live-card[data-task-id="vp-cancel-noop"] [data-cancel-run]'
                ).evaluate("el => el.click()")
                page.locator('[data-task-control="finalize"]').evaluate("el => el.click()")
                for _ in range(100):
                    if held_cancel_details:
                        break
                    page.wait_for_timeout(10)
                assert len(held_cancel_details) == 1
                noop_top = begin_noop_read(page)
                held_cancel_details[0].fulfill(
                    status=200, content_type="application/json", body=json.dumps({
                        "task_id": "vp-cancel-noop", "status": "running",
                        "cancel_state": "pending", "stop_policy": "finalize_then_cancel",
                    }),
                )
                page.wait_for_function(
                    "() => !document.querySelector('.chat-live-card[data-task-id=\"vp-cancel-noop\"] [data-cancel-run]').disabled",
                    timeout=10_000,
                )
                page.evaluate(_SETTLE_TWO_FRAMES)
                assert_noop_read(page, noop_top)
                page.unroute("**/api/state")

                # A production-shaped review reference hydrates asynchronously;
                # both the fetch result and its review DOM reconcile stay anchored.
                page.route(
                    "**/api/tasks/vp-review",
                    lambda route: route.fulfill(
                        status=200,
                        content_type="application/json",
                        body=json.dumps({
                            "task_id": "vp-review",
                            "review_projection": {"panels": [{
                                "panel_id": "hydrated-review",
                                "surface": "task_acceptance",
                                "aggregate_signal": "PASS",
                                "reason": "hydrated viewport evidence",
                                "actors": [],
                            }]},
                        }),
                    ),
                )
                set_remaining(page, 0)
                put_at_viewport_top(page, anchor_selector)
                anchor_before = card_top(page, anchor_id)
                review_reference = {
                    "type": "chat", "role": "system", "system_type": "review_reference",
                    "chat_id": 1, "task_id": "vp-review", "surface": "plan_review",
                    "state_revision": "a" * 64, "ts": "2026-08-03T10:05:10+00:00",
                }
                _emit_ws_frame(page, review_reference)
                page.wait_for_function(
                    "() => document.querySelector('.chat-live-card[data-task-id=\"vp-review\"] "
                    "[data-live-review-summary]')?.textContent === 'Reviews 1'",
                    timeout=10_000,
                )
                assert abs(card_top(page, anchor_id) - anchor_before) <= 6
                assert not jump_state(page)["dotHidden"]

                # Repeating the exact applied revision consumes the frame but
                # changes no projection, so it must not manufacture activity.
                button.click()
                page.evaluate(_SETTLE_TWO_FRAMES)
                set_remaining(page, 300)
                duplicate_before = page.evaluate(
                    """() => {
                        const card = document.querySelector('.chat-live-card[data-task-id="vp-review"]');
                        const messages = document.querySelector('#chat-messages');
                        return {html: card.outerHTML, height: card.getBoundingClientRect().height,
                            scrollHeight: messages.scrollHeight};
                    }"""
                )
                _emit_ws_frame(page, review_reference)
                duplicate_after = page.evaluate(
                    """() => {
                        const card = document.querySelector('.chat-live-card[data-task-id="vp-review"]');
                        const messages = document.querySelector('#chat-messages');
                        return {html: card.outerHTML, height: card.getBoundingClientRect().height,
                            scrollHeight: messages.scrollHeight};
                    }"""
                )
                assert duplicate_after == duplicate_before
                assert jump_state(page)["dotHidden"]

                # Remote provenance survives the GET: the reference can arrive
                # at bottom, then the reader can move away before Reviews lands.
                held_review_routes = []
                page.route("**/api/tasks/vp-review-race", hold_first_route(held_review_routes))
                set_remaining(page, 0)
                delayed_reference = {
                    "type": "chat", "role": "system", "system_type": "review_reference",
                    "chat_id": 1, "task_id": "vp-review-race", "surface": "plan_review",
                    "state_revision": "b" * 64, "ts": "2026-08-03T10:05:11+00:00",
                }
                with page.expect_request("**/api/tasks/vp-review-race"):
                    _emit_ws_frame(page, delayed_reference)
                for _ in range(100):
                    if held_review_routes:
                        break
                    page.wait_for_timeout(10)
                assert len(held_review_routes) == 1
                put_at_viewport_top(page, anchor_selector)
                delayed_anchor_before = card_top(page, anchor_id)
                held_review_routes[0].fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({
                        "task_id": "vp-review-race",
                        "review_projection": {"panels": [{
                            "panel_id": "hydrated-review-race",
                            "surface": "task_acceptance",
                            "aggregate_signal": "PASS",
                            "reason": "delayed viewport evidence",
                            "actors": [],
                        }]},
                    }),
                )
                page.wait_for_function(
                    "() => document.querySelector('.chat-live-card[data-task-id=\"vp-review-race\"] "
                    "[data-live-review-summary]')?.textContent === 'Reviews 1'",
                    timeout=10_000,
                )
                assert abs(card_top(page, anchor_id) - delayed_anchor_before) <= 6
                assert not jump_state(page)["dotHidden"]

                # Mermaid renders a detached clone after the remote handler has
                # returned. Its late mount must use the captured remote boundary.
                set_remaining(page, 0)
                put_at_viewport_top(page, anchor_selector)
                page.evaluate(
                    """() => {
                        window.__releaseMermaid = null;
                        window.mermaid = {
                            initialize() {},
                            async run({nodes}) {
                                await new Promise(resolve => { window.__releaseMermaid = resolve; });
                                for (const node of nodes) {
                                    node.style.height = '420px';
                                    node.innerHTML = '<svg viewBox="0 0 100 420" height="420"><text y="20">Late diagram</text></svg>';
                                }
                            },
                        };
                    }"""
                )
                _emit_ws_frame(page, {
                    "type": "chat", "role": "assistant", "chat_id": 1,
                    "markdown": True, "ts": "2026-08-03T10:00:30+00:00",
                    "content": "Late viewport diagram\n\n```mermaid\ngraph TD; A-->B\n```",
                })
                page.wait_for_function("() => typeof window.__releaseMermaid === 'function'")
                diagram_anchor_before = card_top(page, anchor_id)
                page.evaluate("() => window.__releaseMermaid()")
                page.locator(
                    ".chat-bubble.assistant", has_text="Late viewport diagram"
                ).locator(".md-mermaid svg").wait_for(state="attached", timeout=10_000)
                page.evaluate(_SETTLE_TWO_FRAMES)
                assert abs(card_top(page, anchor_id) - diagram_anchor_before) <= 6
                assert jump_state(page)["dotCount"] == 1

                review_child = page.locator('.chat-live-card[data-task-id="vp-late-child"]')
                review_before = review_child.evaluate("card => card.getBoundingClientRect().height")
                _emit_ws_frame(page, {
                    "type": "chat", "role": "assistant", "is_progress": True,
                    "chat_id": 1, "task_id": "vp-late-child",
                    "delegation_role": "subagent", "subagent_event": "completed",
                    "subagent_task_id": "vp-late-child", "parent_task_id": "vp-parent",
                    "root_task_id": "vp-parent", "subagent_role": "late-reader",
                    "result": "Review-bearing result " * 16,
                    "result_truncated": True, "status": "completed",
                    "ts": "2026-08-03T10:06:00+00:00",
                    "review_projection": {"panels": [{
                        "panel_id": "viewport-review", "surface": "task_acceptance",
                        "authority": "host_root", "aggregate_signal": "PASS",
                        "transport_status": "success", "parse_status": "valid",
                        "quorum": {"required": 1, "contributed": 1, "configured": 1},
                        "enforcement_impact": "supports_pass", "reason": "viewport evidence",
                        "actors": [],
                    }]},
                })
                assert review_child.get_attribute("data-expanded") == "0"
                set_remaining(page, 0)
                put_at_viewport_top(page, anchor_selector)
                anchor_before = card_top(page, anchor_id)
                assert jump_state(page)["dotHidden"]
                review_child.locator(":scope > [data-live-summary-button]").evaluate("el => el.click()")
                page.evaluate(_SETTLE_TWO_FRAMES)
                assert review_child.get_attribute("data-expanded") == "1"
                assert review_child.evaluate("card => card.getBoundingClientRect().height") > review_before + 20
                assert abs(card_top(page, anchor_id) - anchor_before) <= 6
                assert jump_state(page)["dotHidden"]
                held_full_line = []
                page.route(
                    "**/api/tasks/vp-late-child",
                    hold_first_route(held_full_line),
                )
                review_child.locator("[data-live-line-toggle]").first.click()
                for _ in range(100):
                    if held_full_line:
                        break
                    page.wait_for_timeout(10)
                assert len(held_full_line) == 1
                noop_top = begin_noop_read(page)
                held_full_line[0].fulfill(
                    status=200, content_type="application/json", body="{}",
                )
                page.wait_for_timeout(50)
                page.evaluate(_SETTLE_TWO_FRAMES)
                assert_noop_read(page, noop_top)
                page.screenshot(
                    path=str(data_dir.parent / f"live-card-viewport-{browser_engine}.png"),
                    full_page=True,
                )

                # A missing-task detail can return after routine history has
                # already projected the same terminal summary. The late read
                # is internal reconciliation, not another viewport mutation.
                page.unroute("**/api/tasks/vp-late-child")
                detail_progress = {
                    "type": "chat", "role": "assistant", "is_progress": True,
                    "chat_id": 1, "task_id": "vp-detail-noop",
                    "content": "Delayed detail target", "ts": "2026-08-03T10:07:00+00:00",
                }
                set_remaining(page, 0)
                _emit_ws_frame(page, detail_progress)
                held_missing_detail = []
                page.route(
                    "**/api/tasks/vp-detail-noop",
                    hold_first_route(held_missing_detail),
                )
                active_ids = page.locator(
                    '.chat-live-card[data-finished="0"]'
                ).evaluate_all(
                    "cards => cards.map(card => card.dataset.taskId).filter(id => id !== 'vp-detail-noop')"
                )
                page.route(
                    "**/api/state",
                    lambda route: route.fulfill(
                        status=200,
                        content_type="application/json",
                        body=json.dumps({"active_chat_activities": [{
                            "activity_id": task_id, "task_id": task_id, "chat_id": 1,
                            "kind": "managed_task", "phase": "working",
                        } for task_id in active_ids]}),
                    ),
                )
                for _ in range(500):
                    if held_missing_detail:
                        break
                    page.wait_for_timeout(10)
                assert len(held_missing_detail) == 1

                terminal_detail = {
                    "task_id": "vp-detail-noop", "status": "completed",
                    "content": "Delayed detail completed",
                    "ts": "2026-08-03T10:07:01+00:00",
                    "outcome_axes": {
                        "lifecycle": {"status": "completed"}, "execution": {"status": "ok"},
                        "objective": {"status": "pass"}, "review": {"status": "pass"},
                        "artifacts": {"status": "ready"},
                    },
                }
                history_hits = []

                def serve_history(route):
                    history_hits.append(True)
                    route.fulfill(
                        status=200,
                        content_type="application/json",
                        body=json.dumps({"messages": [{
                            **terminal_detail, "role": "system", "system_type": "task_summary",
                        }], "window": {}}),
                    )

                page.route("**/api/chat/history*", serve_history)
                _emit_ws_frame(page, {
                    "type": "chat", "role": "assistant", "is_progress": True,
                    "chat_id": 1, "task_id": "vp-history-trigger",
                    "content": "History trigger", "ts": "2026-08-03T10:07:02+00:00",
                })
                _emit_ws_frame(page, {
                    "type": "chat", "role": "system", "system_type": "task_summary",
                    "chat_id": 1, "task_id": "vp-history-trigger", "content": "Done",
                    "ts": "2026-08-03T10:07:03+00:00",
                })
                for _ in range(300):
                    if history_hits:
                        break
                    page.wait_for_timeout(10)
                assert history_hits
                page.wait_for_function(
                    "() => document.querySelector('.chat-live-card[data-task-id=\"vp-detail-noop\"]')?.dataset.finished === '1'",
                    timeout=10_000,
                )
                noop_top = begin_noop_read(page)
                held_missing_detail[0].fulfill(
                    status=200, content_type="application/json", body=json.dumps(terminal_detail),
                )
                page.wait_for_timeout(50)
                page.evaluate(_SETTLE_TWO_FRAMES)
                assert_noop_read(page, noop_top)
                set_remaining(page, 300)
                healing_anchor = visible_card_anchor(page)
                _emit_ws_frame(page, {
                    "type": "chat", "role": "assistant", "is_progress": True, "ephemeral_decision": True,
                    "chat_id": 1, "task_id": "vp-threshold-freeze-0", "content": "Late decision marker",
                })
                assert abs(card_top(page, healing_anchor["id"]) - healing_anchor["top"]) <= 6
                assert page.locator('[data-task-id="vp-threshold-freeze-0"]').count() == 0 and not jump_state(page)["dotHidden"]
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.fail(f"required Playwright {browser_engine} browser is not installed: {exc}")
        raise
