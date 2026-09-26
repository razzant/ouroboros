"""Browser smoke tests for Project chat continuity and panel lifecycle.

Exercise the shared Main/Project state, instance disposal and visible history
through real browser flows using the common server fixture.

Observable scenarios include:
1. A stale history rebuild (response assembled before the send was logged)
   must not erase the owner's optimistic bubble or its routing receipt.
2. A chat opened AFTER a managed task started shows Working... from the
   /api/state activity snapshot, and the snapshot's absence concludes it.
3. An early final answer holds the card on "Finalizing…" —
   task_cost_finalized does not resolve it, the settled task_done does.
4. Queue loss removes Stop before a single-flight task-detail read settles;
   null/nonterminal reads retry, while terminal detail concludes exactly once.
5. An already-open Project panel consumes the app's existing state snapshot
   fanout and heals a lost task_done without acquiring its own poll.
6. Repeated opening, closing and switching destroys panels and subscriptions
   without counting asynchronous startup log paint as leaked chat DOM.
"""

from __future__ import annotations

import json

import pytest

from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401 - pytest fixture import
from tests.ui_chat_viewport_smoke import _OBSERVE_STATE_READS, _wait_state_reads_quiescent


def _launch(pw):
    browser = pw.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1280, "height": 800})
    page.add_init_script(f"({_OBSERVE_STATE_READS})()")
    return browser, page


def _wait_status(page, expected, timeout=10_000):
    page.wait_for_function(
        """(expected) => {
            const el = document.querySelector('#chat-status');
            return el && el.textContent.trim() === expected;
        }""",
        arg=expected,
        timeout=timeout,
    )


def _goto_main_ready(page, url, expected="Online"):
    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    _wait_status(page, expected, timeout=30_000)
    page.get_by_text("Ouroboros has awakened", exact=True).wait_for(timeout=30_000)
    # The socket-open census may still be reading (a forced refresh coalesced
    # behind an in-flight read starts after it settles); a card minted on the
    # test socket before that read lands would be concluded by its absence.
    _wait_state_reads_quiescent(page)


def _install_task_detail_gate(page):
    """Hold task-detail fetches in page JS so assertions can inspect in-flight UI."""
    page.add_init_script(
        r"""(() => {
            const realFetch = window.fetch.bind(window);
            window.__taskDetailCalls = [];
            window.__taskDetailPending = [];
            window.fetch = (input, init) => {
                const raw = typeof input === 'string' ? input : input?.url || '';
                const path = new URL(raw, window.location.href).pathname;
                if (/^\/api\/tasks\/[^/]+$/.test(path)) {
                    const taskId = decodeURIComponent(path.split('/').pop());
                    window.__taskDetailCalls.push(taskId);
                    return new Promise((resolve) => {
                        window.__taskDetailPending.push({ taskId, resolve });
                    });
                }
                return realFetch(input, init);
            };
            window.__settleTaskDetail = (payload, status = 200) => {
                const pending = window.__taskDetailPending.shift();
                if (!pending) return false;
                pending.resolve(new Response(JSON.stringify(payload), {
                    status,
                    headers: { 'Content-Type': 'application/json' },
                }));
                return true;
            };
        })()"""
    )


_CLICK_CONTEXT_MODE = """() => {
    const control = document.querySelector('#chat-context-mode');
    const next = control.dataset.contextMode === 'low' ? 'max' : 'low';
    control.querySelector(`[data-mode="${next}"]`).click();
}"""


def _start_stale_state_response_episode(page):
    """Hold the page-wide /api/state read, prove the gate, then make it stale.

    One read is in flight page-wide (chat_activity.js createStateSnapshotSequencer
    .gate): a forced chat refresh that lands behind the held app read starts no
    second read and coalesces into ONE follow-up that begins once the held read
    settles. The socket's own generation bump is the only thing that outruns a
    held read, so a synthetic first-connection `open` (no reconnect rebuild, so
    the feed and its live cards survive) makes the held read stale. Afterwards
    index 0 is that stale read and index 1 the follow-up, which starts only when
    index 0 is settled — the order the gate produces.
    """
    page.route(
        "**/api/owner/context-mode",
        lambda route: route.fulfill(
            content_type="application/json", body=json.dumps({"ok": True})
        ),
    )
    page.evaluate(
        r"""(() => {
            const realFetch = window.fetch.bind(window);
            const pending = [];
            window.__stateResponsePending = pending;
            window.fetch = (input, init) => {
                const raw = typeof input === 'string' ? input : input?.url || '';
                if (new URL(raw, location.href).pathname !== '/api/state') {
                    return realFetch(input, init);
                }
                return new Promise((resolve) => pending.push({input, init, resolve}));
            };
            window.__settleStateResponse = async (index, activities, patch = {}) => {
                const item = pending[index];
                if (!item || item.settled) return false;
                item.settled = true;
                const response = await realFetch(item.input, item.init);
                const payload = await response.json();
                Object.assign(payload, patch || {});
                payload.active_chat_activities = activities;
                item.resolve(new Response(JSON.stringify(payload), {
                    status: response.status,
                    headers: {'Content-Type': 'application/json'},
                }));
                return true;
            };
            window.__failStateResponse = (index, status = 500) => {
                const item = pending[index];
                if (!item || item.settled) return false;
                item.settled = true;
                item.resolve(new Response('{}', {
                    status, headers: {'Content-Type': 'application/json'},
                }));
                return true;
            };
            // app.js request is generation N and remains pending.
            window.__ouroWs.emit('projects_changed', {});
        })()"""
    )
    page.wait_for_function("() => window.__stateResponsePending.length === 1")
    # chat.js forces its own refresh through the existing control: gated, it
    # joins the follow-up instead of opening a second read.
    page.evaluate(_CLICK_CONTEXT_MODE)
    page.wait_for_timeout(300)
    assert page.evaluate("() => window.__stateResponsePending.length") == 1, (
        "a forced refresh behind an in-flight read must not start a second read")
    _bump_state_generation(page)


def _bump_state_generation(page):
    """The ungated socket-open bump: every read in flight is stale from here on.

    A first-connection open (previouslyConnected false) never rebuilds history,
    so the synthetic cards the tests minted stay in the feed; the chat's forced
    open refresh coalesces into the follow-up the gate already owes.
    """
    page.evaluate("() => window.__ouroWs.emit('open', { previouslyConnected: false })")


@pytest.mark.ui_browser
def test_ui_smoke_stale_history_rebuild_keeps_owner_bubble(direct_server_with_data):  # noqa: F811
    """Reconnect rebuild against a stale (empty) history keeps bubble + ack."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    try:
        with sync_playwright() as pw:
            browser, page = _launch(pw)
            try:
                # Every history fetch in this scenario returns a STALE empty
                # window — as if it were assembled before the send was logged.
                page.route(
                    "**/api/chat/history*",
                    lambda route: route.fulfill(
                        content_type="application/json",
                        body=json.dumps({"messages": []}),
                    ),
                )
                _goto_main_ready(page, url)

                page.fill("#chat-input", "please keep this message visible")
                page.click("#chat-send")
                bubble = page.locator(".chat-bubble.user[data-client-message-id]").last
                bubble.wait_for(state="visible", timeout=10_000)
                cmid = bubble.get_attribute("data-client-message-id")
                assert cmid

                # Typed routing receipt lands on the bubble.
                page.evaluate(
                    """(cmid) => {
                        window.__ouroWs.emit('message_annotation', {
                            type: 'message_annotation',
                            annotation_type: 'routing_ack',
                            chat_id: 1,
                            client_message_id: cmid,
                            action: 'mailbox_delivery',
                            target: 'root-1',
                            status: 'delivered',
                        });
                    }""",
                    cmid,
                )
                annotation = page.locator(
                    f'.chat-bubble.user[data-client-message-id="{cmid}"] .msg-routing-annotation'
                )
                annotation.wait_for(state="visible", timeout=5_000)
                assert "Delivered to task" in annotation.inner_text()

                # Reconnect-driven full rebuild replays the STALE response.
                page.evaluate(
                    "() => window.__ouroWs.emit('open', { previouslyConnected: true })"
                )
                page.wait_for_selector(
                    '.chat-bubble[data-system-type="reconnect"]', timeout=10_000
                )
                survivor = page.locator(
                    f'.chat-bubble.user[data-client-message-id="{cmid}"]'
                )
                assert survivor.count() == 1
                assert "please keep this message visible" in survivor.inner_text()
                assert "Delivered to task" in survivor.locator(
                    ".msg-routing-annotation"
                ).inner_text()
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.mark.ui_browser
def test_ui_smoke_late_open_chat_shows_managed_activity(direct_server_with_data):  # noqa: F811
    """A chat that missed every typing frame still shows queue-backed Working...;
    the next authoritative snapshot without the task concludes it."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    try:
        with sync_playwright() as pw:
            browser, page = _launch(pw)
            try:
                state = {"managed": True}

                def _inject(route):
                    response = route.fetch()
                    payload = response.json()
                    payload["active_chat_activities"] = (
                        [{
                            "activity_id": "managed-root-1",
                            "chat_id": 1,
                            "project_id": "",
                            "client_message_id": "",
                            "kind": "managed_task",
                            "phase": "working",
                            "started_at": 1.0,
                        }]
                        if state["managed"] else []
                    )
                    route.fulfill(
                        content_type="application/json",
                        body=json.dumps(payload),
                    )

                page.route("**/api/state*", _inject)
                _goto_main_ready(page, url, "Working...")
                # The managed activity arrives purely from the snapshot: this
                # page never saw a typing frame or a progress card.

                # Queue authority concludes it: the task left PENDING/RUNNING.
                state["managed"] = False
                _wait_status(page, "Online", timeout=15_000)
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.mark.ui_browser
def test_ui_smoke_snapshot_apply_time_cannot_resurrect_root(
    direct_server_with_data,  # noqa: F811
):
    """A progress root predating both reads survives the stale empty census, not the follow-up."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    card = '.chat-live-card[data-task-id="snapshot-root"]'
    try:
        with sync_playwright() as pw:
            browser, page = _launch(pw)
            try:
                _install_task_detail_gate(page)
                _goto_main_ready(page, url)
                page.evaluate("() => { window.__realDateNow = Date.now; window.__testNow = Date.now(); Date.now = () => window.__testNow; }")
                # The live frame predates both snapshot requests.
                page.evaluate(
                    """() => window.__ouroWs.emit('chat', {
                        type: 'chat', role: 'assistant', is_progress: true,
                        chat_id: 1, task_id: 'snapshot-root', cancelable: true,
                        content: 'Snapshot-only root is running.',
                    })"""
                )
                page.wait_for_selector(f"{card} [data-cancel-run]")
                page.evaluate("() => { Date.now = () => window.__testNow + 1; }")
                _start_stale_state_response_episode(page)

                # The stale read settles as a complete empty census whose request
                # started after the frame: were apply time live provenance, or the
                # stale generation applied at all, Stop would go now. It stays.
                assert page.evaluate("() => window.__settleStateResponse(0, [])")
                page.wait_for_function("() => window.__stateResponsePending.length === 2")
                assert page.locator(f"{card} [data-cancel-run]").count() == 1
                assert page.evaluate("() => window.__taskDetailCalls.length") == 0
                # The follow-up began only once the stale read settled; its own
                # request barrier is what the same empty census is judged against.
                assert page.evaluate("() => window.__settleStateResponse(1, [])")
                page.evaluate("() => { Date.now = window.__realDateNow; }")
                page.wait_for_function("() => window.__taskDetailCalls.length === 1")
                assert page.locator(f"{card} [data-cancel-run]").count() == 0
                assert page.evaluate(
                    """() => window.__settleTaskDetail({
                        status: 'completed',
                        root_phase_checkpoint: {post_task_synthesis: 'completed'},
                        outcome_axes: {lifecycle: {status: 'completed'}},
                    }, 200)"""
                )
                page.wait_for_function(
                    "(sel) => document.querySelector(sel)?.dataset.finished === '1'", arg=card
                )
                assert page.locator(f"{card} .chat-live-phase").inner_text().strip() == "Done"
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.mark.ui_browser
def test_ui_smoke_reversed_state_responses_do_not_resurrect_main_root(
    direct_server_with_data,  # noqa: F811
):
    """A newer empty chat snapshot wins over an older active app snapshot."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    card = '.chat-live-card[data-task-id="race-main-root"]'
    activity = {
        "activity_id": "race-main-root",
        "chat_id": 1,
        "project_id": "",
        "client_message_id": "",
        "kind": "managed_task",
        "phase": "working",
        "started_at": 1.0,
    }
    try:
        with sync_playwright() as pw:
            browser, page = _launch(pw)
            try:
                _install_task_detail_gate(page)
                _goto_main_ready(page, url)
                # The progress row alone mints the card with Stop authority; a
                # typing frame is a submission receipt and never registers liveness.
                page.evaluate(
                    """() => window.__ouroWs.emit('chat', {
                        type: 'chat', role: 'assistant', is_progress: true,
                        chat_id: 1, task_id: 'race-main-root', cancelable: true,
                        content: 'Main race is running.',
                    })"""
                )
                page.wait_for_selector(f"{card} [data-cancel-run]")
                page.wait_for_timeout(20)
                _start_stale_state_response_episode(page)

                # The stale generation N settles as an ACTIVE census carrying a
                # stale project, a stale binding and an enabled evolution flag. It
                # must mutate neither chat (header nor card), Projects nav, nor
                # the task-binding projection.
                assert page.evaluate(
                    """(activity) => window.__settleStateResponse(0, [activity], {
                        projects: [{id: 'stale-project', name: 'Stale project',
                                    chat_id: 999, lifecycle: 'active'}],
                        project_chat_ids: [999],
                        task_bindings: {'stale-binding': {project_id: 'stale-project', chat_id: 999}},
                        evolution_enabled: true,
                    })""",
                    activity,
                )
                page.wait_for_function("() => window.__stateResponsePending.length === 2")
                assert page.locator('[data-project-id="stale-project"]').count() == 0
                assert page.evaluate("() => !window.__ouroTaskBindings['stale-binding']")
                assert not page.locator('[data-chat-command="evolve"]').evaluate(
                    "(node) => node.classList.contains('on')"
                )
                assert page.locator(f"{card} [data-cancel-run]").count() == 1

                # The follow-up N+1, started only once N settled, proves queue
                # loss and carries the fresh binding.
                assert page.evaluate(
                    """() => window.__settleStateResponse(1, [], {
                        task_bindings: {'fresh-binding': {project_id: 'fresh', chat_id: 77}},
                    })"""
                )
                page.wait_for_function("() => window.__taskDetailCalls.length === 1")
                assert page.locator(f"{card} [data-cancel-run]").count() == 0
                page.wait_for_timeout(100)
                assert page.locator('[data-project-id="stale-project"]').count() == 0
                assert page.evaluate(
                    "() => Boolean(window.__ouroTaskBindings['fresh-binding'])"
                    " && !window.__ouroTaskBindings['stale-binding']"
                )

                assert page.evaluate(
                    """() => window.__settleTaskDetail({
                        status: 'completed',
                        root_phase_checkpoint: {post_task_synthesis: 'completed'},
                        outcome_axes: {lifecycle: {status: 'completed'}},
                    }, 200)"""
                )
                page.wait_for_function(
                    "(sel) => document.querySelector(sel)?.dataset.finished === '1'",
                    arg=card,
                )
                assert page.locator(f"{card} .chat-live-phase").inner_text().strip() == "Done"
                assert page.evaluate("() => window.__taskDetailCalls") == ["race-main-root"]
                _wait_status(page, "Online")

                # A failed STALE chat request is presentation-only and must not
                # roll back the header state a newer successful read applied.
                # A held read now blocks every later gated read, so settle
                # whatever the header poll parked first, in the same turn as the
                # forced app refresh that then surfaces as the next pending read.
                before = page.evaluate("() => window.__stateResponsePending.length")
                page.evaluate(
                    """() => {
                        window.__stateResponsePending.forEach((item, index) => {
                            if (!item.settled) window.__settleStateResponse(index, []);
                        });
                        window.__ouroWs.emit('projects_changed', {});
                    }"""
                )
                page.wait_for_function(
                    "(before) => window.__stateResponsePending.length > before",
                    arg=before,
                )
                success_index = page.evaluate("() => window.__stateResponsePending.length - 1")
                assert page.evaluate(
                    "(index) => window.__settleStateResponse(index, [], {evolution_enabled: true})",
                    success_index,
                )
                page.wait_for_function(
                    "() => document.querySelector('[data-chat-command=evolve]').classList.contains('on')"
                )
                # The chat's own forced read (or a header poll that beat it to the
                # free gate: both are chat reads) is held, then made stale by the
                # socket bump, then fails. Not current, it may not touch the header.
                page.evaluate(_CLICK_CONTEXT_MODE)
                page.wait_for_function(
                    "(index) => window.__stateResponsePending.length > index + 1",
                    arg=success_index,
                )
                failed_index = page.evaluate("() => window.__stateResponsePending.length - 1")
                _bump_state_generation(page)
                assert page.evaluate(
                    "(index) => window.__failStateResponse(index)", failed_index
                )
                page.wait_for_timeout(100)
                assert page.locator('[data-chat-command="evolve"]').evaluate(
                    "(node) => node.classList.contains('on')"
                )
                # The failure released the gate: the coalesced follow-up starts
                # and a current read applies as usual.
                page.wait_for_function(
                    "(index) => window.__stateResponsePending.length > index + 1",
                    arg=failed_index,
                )
                assert page.evaluate(
                    """() => window.__settleStateResponse(
                        window.__stateResponsePending.length - 1, [], {evolution_enabled: true})"""
                )
                page.wait_for_timeout(100)
                assert page.locator('[data-chat-command="evolve"]').evaluate(
                    "(node) => node.classList.contains('on')"
                )
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.mark.ui_browser
def test_ui_smoke_queue_loss_converges_terminal_card_once(direct_server_with_data):  # noqa: F811
    """Lost task_done heals from existing state snapshots plus durable detail."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    card = '.chat-live-card[data-task-id="truth-root"]'
    try:
        with sync_playwright() as pw:
            browser, page = _launch(pw)
            try:
                _install_task_detail_gate(page)

                def _activity(task_id, chat_id=1):
                    return {
                        "activity_id": task_id,
                        "chat_id": chat_id,
                        "project_id": "",
                        "client_message_id": "",
                        "kind": "managed_task",
                        "phase": "working",
                        "started_at": 1.0,
                    }

                state = {"activities": [_activity("truth-root")]}
                def _inject(route):
                    response = route.fetch()
                    payload = response.json()
                    payload["active_chat_activities"] = list(state["activities"])
                    payload["project_chat_ids"] = [9] if any(a.get("chat_id") == 9 for a in state["activities"]) else []
                    route.fulfill(content_type="application/json", body=json.dumps(payload))

                history = {"messages": []}
                page.route("**/api/state*", _inject)
                page.route("**/api/chat/history*", lambda route: route.fulfill(content_type="application/json", body=json.dumps(history)))
                _goto_main_ready(page, url, "Working...")

                page.evaluate(
                    """() => window.__ouroWs.emit('chat', {
                        type: 'chat', role: 'assistant', is_progress: true,
                        chat_id: 1, task_id: 'truth-root', cancelable: true,
                        content: 'Still working.', ts: '2026-08-19T10:00:00+00:00',
                    })"""
                )
                page.wait_for_selector(f"{card} [data-cancel-run]", timeout=10_000)

                page.evaluate(
                    """() => window.__ouroWs.emit('chat', {
                        type: 'chat', role: 'assistant', chat_id: 1,
                        task_id: 'truth-root', task_phase: 'finalizing',
                        content: 'The early answer is already visible.',
                        ts: '2026-08-19T10:00:01+00:00',
                    })"""
                )
                page.wait_for_function(
                    """(sel) => document.querySelector(sel + ' .chat-live-phase')
                        ?.textContent.trim() === 'Finalizing…'""",
                    arg=card,
                )
                page.evaluate(
                    """(sel) => {
                        window.__finishTransitions = 0;
                        window.__finishObserver = new MutationObserver((rows) => {
                            for (const row of rows) {
                                if (row.attributeName === 'data-finished'
                                    && row.oldValue === '0'
                                    && row.target.dataset.finished === '1') {
                                    window.__finishTransitions += 1;
                                }
                            }
                        });
                        window.__finishObserver.observe(document.querySelector(sel), {
                            attributes: true, attributeFilter: ['data-finished'],
                            attributeOldValue: true,
                        });
                    }""",
                    card,
                )

                # Overlapping refreshes revoke Stop and share one detail read.
                state["activities"] = []
                page.evaluate(
                    """() => {
                        window.__ouroWs.emit('projects_changed', {});
                        window.__ouroWs.emit('projects_changed', {});
                    }"""
                )
                page.wait_for_function("() => window.__taskDetailCalls.length === 1")
                assert page.locator(f"{card} [data-cancel-run]").count() == 0
                assert page.locator(card).get_attribute("data-finished") == "0"

                # Unreachable and completed-but-open detail prove nothing.
                assert page.evaluate("() => window.__settleTaskDetail({}, 404)")
                page.wait_for_timeout(100)
                page.evaluate("() => window.__ouroWs.emit('projects_changed', {})")
                page.wait_for_function("() => window.__taskDetailCalls.length === 2")
                assert page.evaluate(
                    """() => window.__settleTaskDetail({
                        status: 'completed',
                        root_phase_checkpoint: {post_task_synthesis: 'pending_once'},
                    }, 200)"""
                )
                page.wait_for_timeout(100)
                assert page.locator(card).get_attribute("data-finished") == "0"

                page.evaluate(
                    """() => {
                        window.__ouroWs.emit('projects_changed', {});
                        window.__ouroWs.emit('projects_changed', {});
                    }"""
                )
                page.wait_for_function("() => window.__taskDetailCalls.length === 3")
                page.wait_for_timeout(150)
                assert page.evaluate("() => window.__taskDetailCalls.length") == 3
                assert page.evaluate(
                    """() => window.__settleTaskDetail({
                        status: 'completed',
                        root_phase_checkpoint: {post_task_synthesis: 'completed'},
                        outcome_axes: {lifecycle: {status: 'completed'}},
                    }, 200)"""
                )
                page.wait_for_function(
                    "(sel) => document.querySelector(sel)?.dataset.finished === '1'",
                    arg=card,
                )
                assert page.locator(f"{card} .chat-live-phase").inner_text().strip() == "Done"
                _wait_status(page, "Online")
                assert page.evaluate("() => window.__finishTransitions") == 1

                state["activities"] = [_activity("truth-root")]
                page.evaluate("() => window.__ouroWs.emit('projects_changed', {})")
                page.wait_for_timeout(150)
                page.evaluate(
                    """() => window.__ouroWs.emit('typing', {
                        type: 'typing', chat_id: 1, activity_id: 'truth-root',
                        kind: 'managed_task', phase: 'working',
                    })"""
                )
                page.wait_for_timeout(100)
                assert page.evaluate("() => window.__taskDetailCalls.length") == 3
                assert page.evaluate("() => window.__finishTransitions") == 1
                _wait_status(page, "Online")

                rehome_card = '.chat-live-card[data-task-id="rehome-root"]'
                state["activities"] = [_activity("rehome-root")]
                page.evaluate("() => window.__ouroWs.emit('projects_changed', {})")
                _wait_status(page, "Working...")
                page.evaluate(
                    """() => window.__ouroWs.emit('chat', {
                        type: 'chat', role: 'assistant', is_progress: true,
                        chat_id: 1, task_id: 'rehome-root', cancelable: true,
                        content: 'Moving to its Project.',
                    })"""
                )
                page.wait_for_selector(f"{rehome_card} [data-cancel-run]")
                page.wait_for_timeout(20)
                state["activities"] = [_activity("rehome-root", 9)]
                with page.expect_response("**/api/state*", timeout=10_000):
                    page.evaluate("() => window.__ouroWs.emit('projects_changed', {})")
                page.wait_for_function(
                    "(sel) => document.querySelector(sel + ' [data-cancel-run]') === null",
                    arg=rehome_card,
                )
                assert page.evaluate("() => window.__taskDetailCalls.length") == 3
                assert page.locator(rehome_card).get_attribute("data-finished") == "0"
                page.evaluate("() => window.__ouroWs.emit('chat', {type: 'chat', role: 'assistant', is_progress: true, chat_id: 9, task_id: 'rehome-root', cancelable: true, content: 'Project work continues.'})")
                assert page.locator(f"{rehome_card} [data-cancel-run]").count() == 0
                history["messages"] = [{"text": "Historical project progress.", "role": "assistant", "is_progress": True, "task_id": "rehome-root", "cancelable": True, "project_mirror": True}]
                with page.expect_response("**/api/chat/history*", timeout=10_000):
                    page.evaluate("() => window.__ouroWs.emit('open', {previouslyConnected: true})")
                page.wait_for_selector(rehome_card)
                assert page.get_by_text("Historical project progress.").count() >= 1
                assert page.locator(f"{rehome_card} [data-cancel-run]").count() == 0

                # A later rehome snapshot clears an earlier candidate and response.
                # cannot finish the still-running card.
                state["activities"] = [_activity("rehome-root")]
                page.evaluate("() => window.__ouroWs.emit('projects_changed', {})")
                page.wait_for_timeout(100)
                page.evaluate(
                    """() => window.__ouroWs.emit('chat', {
                        type: 'chat', role: 'assistant', is_progress: true,
                        chat_id: 1, task_id: 'rehome-root', cancelable: true,
                        content: 'Still moving.',
                    })"""
                )
                page.wait_for_selector(f"{rehome_card} [data-cancel-run]")
                state["activities"] = []
                page.evaluate("() => window.__ouroWs.emit('projects_changed', {})")
                page.wait_for_function("() => window.__taskDetailCalls.length === 4")
                state["activities"] = [_activity("rehome-root", 9)]
                with page.expect_response("**/api/state*", timeout=10_000):
                    page.evaluate("() => window.__ouroWs.emit('projects_changed', {})")
                page.wait_for_timeout(50)
                assert page.evaluate(
                    """() => window.__settleTaskDetail({
                        status: 'completed',
                        outcome_axes: {lifecycle: {status: 'completed'}},
                    }, 200)"""
                )
                page.wait_for_timeout(100)
                assert page.locator(rehome_card).get_attribute("data-finished") == "0"

                # Fast terminal truth ledgers a root even though rehome left no
                # local activity/missing entry; late typing and old state stay inert.
                page.evaluate(
                    """() => window.__ouroWs.emit('log', {
                        type: 'log', chat_id: 1,
                        data: {type: 'task_done', task_id: 'rehome-root', status: 'completed'},
                    })"""
                )
                page.wait_for_function(
                    "(sel) => document.querySelector(sel)?.dataset.finished === '1'",
                    arg=rehome_card,
                )
                state["activities"] = [_activity("rehome-root")]
                page.evaluate(
                    """() => {
                        window.__ouroWs.emit('typing', {
                            type: 'typing', chat_id: 1, activity_id: 'rehome-root',
                            kind: 'managed_task', phase: 'working',
                        });
                        window.__ouroWs.emit('projects_changed', {});
                    }"""
                )
                page.wait_for_timeout(150)
                assert page.evaluate("() => window.__taskDetailCalls.length") == 4
                _wait_status(page, "Online")

                # Durable fallback settles a reusable logical slot without
                # permanently blocking its next Working cycle.
                reusable_card = '.chat-live-card[data-task-id="active"]'
                state["activities"] = [_activity("active")]
                page.evaluate("() => window.__ouroWs.emit('projects_changed', {})")
                _wait_status(page, "Working...")
                page.evaluate(
                    """() => window.__ouroWs.emit('chat', {
                        type: 'chat', role: 'assistant', is_progress: true,
                        chat_id: 1, task_id: 'active', cancelable: true,
                        content: 'Reusable cycle one.',
                    })"""
                )
                page.wait_for_selector(reusable_card)
                page.wait_for_timeout(20)
                state["activities"] = []
                page.evaluate("() => window.__ouroWs.emit('projects_changed', {})")
                page.wait_for_function("() => window.__taskDetailCalls.length === 5")
                assert page.evaluate(
                    """() => window.__settleTaskDetail({
                        status: 'completed',
                        root_phase_checkpoint: {post_task_synthesis: 'completed'},
                        outcome_axes: {lifecycle: {status: 'completed'}},
                    }, 200)"""
                )
                page.wait_for_function(
                    "(sel) => document.querySelector(sel)?.dataset.finished === '1'",
                    arg=reusable_card,
                )
                _wait_status(page, "Online")
                # Cycle two reopens through its progress card (typing frames are
                # receipts and never light the header).
                page.evaluate(
                    """() => window.__ouroWs.emit('chat', {
                        type: 'chat', role: 'assistant', is_progress: true,
                        chat_id: 1, task_id: 'active', cancelable: true,
                        content: 'Reusable cycle two.',
                    })"""
                )
                page.wait_for_function(
                    """(sel) => {
                        const card = document.querySelector(sel);
                        return card?.dataset.finished === '0'
                            && card.textContent.includes('Reusable cycle two.');
                    }""",
                    arg=reusable_card,
                )
                _wait_status(page, "Working...")
                page.evaluate(
                    """() => window.__ouroWs.emit('log', {
                        type: 'log', chat_id: 1,
                        data: {type: 'task_done', task_id: 'active', status: 'completed'},
                    })"""
                )

                progress_card = '.chat-live-card[data-task-id="progress-only-root"]'
                page.evaluate(
                    """() => window.__ouroWs.emit('chat', {
                        type: 'chat', role: 'assistant', is_progress: true, chat_id: 1,
                        task_id: 'progress-only-root', cancelable: true, content: 'Working.',
                    })"""
                )
                page.wait_for_selector(f"{progress_card} [data-cancel-run]")
                page.evaluate("() => window.__ouroWs.emit('projects_changed', {})")
                page.wait_for_function("() => window.__taskDetailCalls.length === 6")
                assert page.locator(f"{progress_card} [data-cancel-run]").count() == 0
                assert page.locator(progress_card).get_attribute("data-finished") == "0"
                assert page.evaluate("() => window.__settleTaskDetail({status: 'completed'}, 200)")
                page.wait_for_function(
                    "(sel) => document.querySelector(sel)?.dataset.finished === '1'",
                    arg=progress_card,
                )
                assert page.locator(f"{progress_card} .chat-live-phase").inner_text().strip() == "Done"
                _wait_status(page, "Online")

                # A kind-stamped known subagent still gains no convergence
                # authority; the minted parent ROOT card is now named by the
                # card-set reconcile — hence one extra parent-root read.
                state["activities"] = []
                page.evaluate(
                    """() => {
                        window.__ouroWs.emit('chat', {
                            type: 'chat', role: 'assistant', is_progress: true,
                            chat_id: 1, task_id: 'parent-root',
                            parent_task_id: 'parent-root', subagent_task_id: 'child-root',
                            delegation_role: 'subagent', subagent_role: 'reviewer',
                            subagent_event: 'running', content: 'Reviewing.',
                        });
                        window.__ouroWs.emit('typing', {
                            type: 'typing', chat_id: 1, activity_id: 'child-root',
                            kind: 'managed_task', phase: 'working',
                        });
                        window.__ouroWs.emit('projects_changed', {});
                    }"""
                )
                page.wait_for_timeout(200)
                assert page.evaluate("() => window.__taskDetailCalls") == [
                    "truth-root", "truth-root", "truth-root", "rehome-root", "active",
                    "progress-only-root", "parent-root",
                ]
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.mark.ui_browser
def test_ui_smoke_early_final_holds_finalizing_until_task_done(direct_server_with_data):  # noqa: F811
    """Finalizing hold: early final -> Finalizing…, cost checkpoint does not
    resolve the card, task_done does."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    card = '.chat-live-card[data-task-id="fin-1"]'
    try:
        with sync_playwright() as pw:
            browser, page = _launch(pw)
            try:
                _goto_main_ready(page, url)

                # A tool call makes the root card visible.
                page.evaluate(
                    """() => window.__ouroWs.emit('log', {
                        type: 'log', chat_id: 1,
                        data: { type: 'tool_call_started', task_id: 'fin-1',
                                tool: 'run_command', ts: '2026-08-17T10:00:00+00:00' },
                    })"""
                )
                page.wait_for_selector(card, timeout=10_000)

                # Early final answer: delivered while post-task still runs.
                page.evaluate(
                    """() => window.__ouroWs.emit('chat', {
                        type: 'chat', role: 'assistant', chat_id: 1,
                        content: 'Here is the delivered answer.',
                        task_id: 'fin-1', task_phase: 'finalizing',
                        ts: '2026-08-17T10:00:05+00:00',
                    })"""
                )
                page.wait_for_function(
                    """(sel) => {
                        const el = document.querySelector(sel + ' .chat-live-phase');
                        return el && el.textContent.trim() === 'Finalizing…';
                    }""",
                    arg=card,
                    timeout=5_000,
                )
                assert page.locator(card).get_attribute("data-finished") == "0"
                # The answer itself is visible as an ordinary bubble.
                assert page.locator(
                    ".chat-bubble.assistant", has_text="Here is the delivered answer."
                ).count() >= 1

                # The cost checkpoint is bookkeeping — the card stays open.
                page.evaluate(
                    """() => window.__ouroWs.emit('log', {
                        type: 'log', chat_id: 1,
                        data: { type: 'task_cost_finalized', task_id: 'fin-1',
                                post_task_status: 'completed',
                                cost_accounting_status: 'available',
                                accounted_upper_bound_usd: 0.42, cost_final: true,
                                ts: '2026-08-17T10:00:07+00:00' },
                    })"""
                )
                page.wait_for_timeout(300)
                assert page.locator(card).get_attribute("data-finished") == "0"
                assert page.locator(f"{card} .chat-live-phase").inner_text().strip() == "Finalizing…"

                # The settled task_done resolves the card.
                page.evaluate(
                    """() => window.__ouroWs.emit('log', {
                        type: 'log', chat_id: 1,
                        data: { type: 'task_done', task_id: 'fin-1', status: 'completed',
                                outcome_axes: { lifecycle: { status: 'completed' } },
                                ts: '2026-08-17T10:00:09+00:00' },
                    })"""
                )
                page.wait_for_function(
                    """(sel) => document.querySelector(sel)?.dataset.finished === '1'""",
                    arg=card,
                    timeout=5_000,
                )
                assert page.locator(f"{card} .chat-live-phase").inner_text().strip() == "Done"
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.mark.ui_browser
def test_ui_smoke_open_project_panel_heals_lost_task_done_from_state_fanout(
    direct_server_with_data,  # noqa: F811
):
    """The app's existing state refresh converges an already-open panel."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    from ouroboros.projects_registry import create_project

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    project = create_project(data_dir, "cont-panel", name="Continuity panel")
    project_chat = int(project["chat_id"])
    status_sel = '[id="pchat-cont-panel-status"]'
    card = '#panel-pchat-cont-panel .chat-live-card[data-task-id="panel-root-1"]'

    def _panel_status_is(page, expected, timeout=10_000):
        page.wait_for_function(
            """({ sel, expected }) => {
                const el = document.querySelector(sel);
                return el && el.textContent.trim() === expected;
            }""",
            arg={"sel": status_sel, "expected": expected},
            timeout=timeout,
        )

    try:
        with sync_playwright() as pw:
            browser, page = _launch(pw)
            try:
                _install_task_detail_gate(page)
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                with page.expect_response(lambda response: response.url.endswith("/api/ui/preferences") and response.request.method == "POST", timeout=30_000):
                    page.click('.nav-project-row[data-project-id="cont-panel"]')
                page.wait_for_selector("#project-panel:not([hidden])", timeout=30_000)
                _panel_status_is(page, "Online", timeout=30_000)
                # The panel's forced mount refresh may be coalesced behind an
                # in-flight page read; let it land before minting the card below.
                _wait_state_reads_quiescent(page)

                # A visible root card carries host-attested Stop authority and is
                # what lights the panel header (typing frames are receipts only).
                page.evaluate(
                    """(chatId) => window.__ouroWs.emit('chat', {
                        type: 'chat', role: 'assistant', is_progress: true,
                        chat_id: chatId, task_id: 'panel-root-1', cancelable: true,
                        content: 'Project work is running.',
                        ts: '2026-08-19T11:00:00+00:00',
                    })""",
                    project_chat,
                )
                page.wait_for_selector(f"{card} [data-cancel-run]", timeout=10_000)
                _panel_status_is(page, "Working...")

                # Early final prose does not conclude while post-task work runs.
                page.evaluate(
                    """(chatId) => window.__ouroWs.emit('chat', {
                        type: 'chat', role: 'assistant', chat_id: chatId,
                        content: 'Early answer.', task_id: 'panel-root-1',
                        task_phase: 'finalizing', ts: '2026-08-17T11:00:00+00:00',
                    })""",
                    project_chat,
                )
                _panel_status_is(page, "Working...")
                # Make the next request barrier strictly newer than the live
                # progress frame even on a millisecond-resolution browser clock.
                page.wait_for_timeout(20)

                # The stale app read settles as a complete empty census: dropped,
                # so the panel's Stop survives. The follow-up — which the panel's
                # own forced open refresh joined instead of opening a poll of its
                # own — is the queue-loss fact that stays authoritative in-panel.
                _start_stale_state_response_episode(page)
                assert page.evaluate("() => window.__settleStateResponse(0, [])")
                page.wait_for_function("() => window.__stateResponsePending.length === 2")
                assert page.locator(f"{card} [data-cancel-run]").count() == 1
                assert page.evaluate("() => window.__taskDetailCalls.length") == 0
                assert page.evaluate("() => window.__settleStateResponse(1, [])")
                page.wait_for_function("() => window.__taskDetailCalls.length === 1")
                assert page.locator(f"{card} [data-cancel-run]").count() == 0
                page.wait_for_timeout(100)
                assert page.evaluate(
                    """() => window.__settleTaskDetail({
                        status: 'completed',
                        root_phase_checkpoint: {post_task_synthesis: 'completed'},
                        outcome_axes: {lifecycle: {status: 'completed'}},
                    }, 200)"""
                )
                page.wait_for_function(
                    "(sel) => [...document.querySelectorAll(sel)]"
                    ".some((node) => node.dataset.finished === '1')",
                    arg=card,
                    timeout=10_000,
                )
                _panel_status_is(page, "Online")
                assert page.locator(f"{card} .chat-live-phase").inner_text().strip() == "Done"
                assert page.evaluate("() => window.__taskDetailCalls") == ["panel-root-1"]
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.mark.ui_browser
def test_ui_smoke_project_pointer_is_a_main_root_affordance(direct_server_with_data):  # noqa: F811
    """The bound-task pointer (the Project reference) belongs to the LIVE Main ROOT card:
    a task that scopes itself into a project mid-run keeps its Main card, which
    gains the pointer; the same task's card inside the project panel carries none;
    and clicking the Main pointer while that panel is already open leaves it open."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    from ouroboros.projects_registry import bind_task_to_project, create_project

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    project = create_project(data_dir, "ptr-panel", name="Pointer panel")
    project_chat = int(project["chat_id"])
    main_card = '#page-chat .chat-live-card[data-task-id="ptr-root"]'
    panel_card = '#panel-pchat-ptr-panel .chat-live-card[data-task-id="ptr-root"]'
    try:
        with sync_playwright() as pw:
            browser, page = _launch(pw)
            try:
                _goto_main_ready(page, url)
                # The task starts in Main as a live root card...
                page.evaluate(
                    """() => window.__ouroWs.emit('chat', {
                        type: 'chat', role: 'assistant', is_progress: true, chat_id: 1,
                        task_id: 'ptr-root', content: 'Started in Main',
                        ts: '2026-08-19T10:00:00+00:00',
                    })"""
                )
                page.wait_for_selector(main_card, state="attached", timeout=30_000)
                assert page.locator(f"{main_card} .chat-live-bound-pointer").count() == 0
                # ...then scopes itself into the project (durable binding + project-thread progress).
                bind_task_to_project(data_dir, "ptr-root", "ptr-panel", project_chat, origin={"absent": "system"})
                logs_dir = data_dir / "logs"
                logs_dir.mkdir(parents=True, exist_ok=True)
                with (logs_dir / "progress.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "ts": "2026-08-19T10:00:01+00:00", "chat_id": project_chat,
                        "task_id": "ptr-root", "content": "Continues in the project",
                        "is_progress": True,
                    }) + "\n")
                page.evaluate("() => window.__ouroWs.emit('projects_changed', {})")
                page.wait_for_selector(f"{main_card} .chat-live-bound-pointer", state="attached", timeout=30_000)
                assert page.locator(f"{main_card} .chat-live-project-name").inner_text().strip() == "Pointer panel"
                assert page.locator(f"{main_card} .chat-live-project-icon svg").count() == 1
                page_errors = []
                page.on("pageerror", lambda exc: page_errors.append(str(exc)))
                # Positive path: the pointer OPENS its project panel from Main.
                assert page.locator("#project-panel:not([hidden])").count() == 0
                page.locator(f"{main_card} .chat-live-bound-pointer").click()
                page.wait_for_selector("#project-panel:not([hidden])", timeout=30_000)
                page.wait_for_selector(panel_card, state="attached", timeout=30_000)
                # Re-apply the bindings now that the panel card exists: it stays pointer-free.
                page.evaluate("() => window.__ouroWs.emit('projects_changed', {})")
                page.wait_for_timeout(600)
                assert page.locator(f"{panel_card} .chat-live-bound-pointer").count() == 0
                assert page.locator(f"{main_card} .chat-live-bound-pointer").count() == 1
                # Open-or-noop: the pointer never closes the panel it points at. The
                # panel backdrop covers Main, so the handler is exercised directly.
                page.locator(f"{main_card} .chat-live-bound-pointer").dispatch_event("click")
                page.wait_for_timeout(400)
                assert page.locator("#project-panel:not([hidden])").count() == 1
                assert page.locator(panel_card).count() == 1
                assert page_errors == [], page_errors
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.mark.ui_browser
def test_ui_smoke_project_panel_lifecycle_does_not_leak(direct_server_with_data):  # noqa: F811
    """Open/close cycles keep one live panel, flat ws listeners, and flat DOM.

    P3 lifecycle concrete: closing or switching a project DESTROYS its chat
    instance (disposing every ws.on subscription, the ResizeObserver, the
    window/document listeners, and all timers), so repeated open/close cycles
    cannot accumulate hidden panels, listeners, or DOM nodes. Panels marked
    data-pending-work (staged attachments / in-flight upload) are the one
    sanctioned exception and are excluded from the live-panel count.
    """
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    from ouroboros.projects_registry import create_project

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    project_ids = [f"leak-{idx}" for idx in range(1, 4)]
    for idx, project_id in enumerate(project_ids, start=1):
        create_project(data_dir, project_id, name=f"Leak project {idx}")

    # window.__ouroWs is the loopback debug hook app.js exposes for exactly
    # this count; the module-scoped ws is unreachable from page.evaluate.
    count_listeners = """() => {
        const ws = window.__ouroWs;
        return Object.values(ws.listeners).reduce((total, set) => total + set.size, 0);
    }"""
    live_panels = """() => [...document.querySelectorAll('.chat-instance-panel')]
        .filter((panel) => panel.dataset.pendingWork !== '1').length"""
    dom_count = "() => document.getElementsByTagName('*').length"

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                for project_id in project_ids:
                    page.wait_for_selector(
                        f'.nav-project-row[data-project-id="{project_id}"]', timeout=30_000
                    )

                def open_project(project_id):
                    page.click(f'.nav-project-row[data-project-id="{project_id}"]')
                    page.wait_for_selector("#project-panel:not([hidden])", timeout=30_000)
                    page.wait_for_selector(
                        f'[id="panel-pchat-{project_id}"]:not([hidden])', timeout=30_000
                    )

                def close_project():
                    page.click("#project-panel-close")
                    page.wait_for_function(
                        "() => !document.getElementById('project-panel')"
                        ".classList.contains('open')",
                        timeout=30_000,
                    )

                # Baseline AFTER one full open/close cycle so one-time lazy
                # registrations cannot masquerade as leaks.
                open_project(project_ids[0])
                close_project()
                # The server's readiness response precedes its worker/startup
                # log paint. These legitimate rows add 37 nodes outside the
                # project panel; observe them before measuring flat DOM.
                page.wait_for_function(
                    """() => {
                        const painted = new Set([...document.querySelectorAll('#log-entries .log-type')]
                            .map(node => node.textContent));
                        return ['worker_sha_verify', 'startup_verification', 'worker_ready']
                            .every(type => painted.has(type));
                    }""",
                    timeout=30_000,
                )
                listeners_baseline = page.evaluate(count_listeners)
                dom_baseline = page.evaluate(dom_count)
                assert listeners_baseline > 0

                # Small slack for churn outside the panel (badges, toasts);
                # a leaked panel or card timeline is hundreds of nodes.
                dom_slack = 30
                for project_id in project_ids:
                    open_project(project_id)
                    assert page.evaluate(live_panels) <= 1
                    close_project()
                    assert page.evaluate(live_panels) == 0
                    # Every cycle returns to the baseline: no monotonic growth.
                    cycle_dom = page.evaluate(dom_count)
                    assert cycle_dom <= dom_baseline + dom_slack, (dom_baseline, cycle_dom)
                    assert page.evaluate(count_listeners) == listeners_baseline

                # Direct project-to-project switch (no explicit close) also
                # destroys the previous instance: one live panel, ever.
                open_project(project_ids[0])
                open_project(project_ids[1])
                assert page.evaluate(live_panels) == 1
                close_project()
                assert page.evaluate(live_panels) == 0

                assert page.evaluate(count_listeners) == listeners_baseline
                final_dom = page.evaluate(dom_count)
                assert final_dom <= dom_baseline + dom_slack, (dom_baseline, final_dom)
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise
