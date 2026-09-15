"""S3 (Q1/Q2/HQ1) focused browser proofs for the shared task control dropdown.

A NEW marker-gated module (the giant smoke file is byte-pinned): real server +
real bundles in Chromium, with the mutating control endpoints stubbed at the
`window.fetch` seam so the proofs stay deterministic and the no-chat contract
is asserted against the LIVE chat DOM, not a unit model.

Covered here (both owner surfaces):
  * Chat live card: trigger label, the exact frozen three-action dropdown,
    dismiss-continues-the-run, "Hurry up" = local toast + stable request_id
    idempotent retry + ZERO new chat bubbles, "Wrap up" = "Finalizing…"
    card with the pending menu collapsed to "Stop now" only.
  * Activity tab: the SAME shared dropdown (product-wide parity) with the same
    labels and the same no-chat hurry acknowledgement.

Run: OUROBOROS_RUN_UI_SMOKE=1 python -m pytest -o addopts="" -m ui_browser \
         tests/test_s3_task_control_browser.py
"""
from __future__ import annotations

import json

import pytest

from tests.test_ui_smoke_playwright import (
    direct_server_with_data as _direct_server_with_data,
)

# Re-export the shared server fixture under its canonical name so the tests
# below can request it (aliased import keeps ruff F811 clean).
direct_server_with_data = _direct_server_with_data

# The frozen owner wording (Q2/HQ1) — asserted verbatim, in order.
EXPECTED_ACTIONS = [
    ("finalize", "Wrap up"),
    ("hurry", "Hurry up"),
    ("stop_now", "Stop now"),
]

# One fetch-seam stub shared by both surfaces: captures every mutating control
# call for later assertion and answers the S3 contract shapes without touching
# the real queue (GET traffic falls through to the real server).
_FETCH_STUB_JS = """
(taskId) => {
    const real = window.fetch.bind(window);
    window.__s3Calls = [];
    const jsonResp = (obj, status = 200) => Promise.resolve(new Response(
        JSON.stringify(obj), { status, headers: { 'Content-Type': 'application/json' } },
    ));
    window.__s3PendingCancel = false;
    window.fetch = (input, init = {}) => {
        const url = String(input && input.url ? input.url : input);
        const method = String((init && init.method) || (input && input.method) || 'GET').toUpperCase();
        const record = (kind) => window.__s3Calls.push({
            kind, url, method, body: init && init.body ? JSON.parse(init.body) : null,
        });
        if (url.includes(`/api/tasks/${taskId}/hurry`)) {
            record('hurry');
            const dup = window.__s3Calls.filter((c) => c.kind === 'hurry').length > 1;
            return jsonResp({
                ok: true, task_id: taskId,
                request_id: (init && init.body) ? JSON.parse(init.body).request_id : '',
                state: 'requested', attempt_key: 1, duplicate: dup,
            });
        }
        if (url.includes(`/api/tasks/${taskId}/cancel`)) {
            record('cancel');
            window.__s3PendingCancel = true;
            return jsonResp({
                ok: true, task_id: taskId, cancelled: [], cascade: true,
                cancel_state: 'pending', stop_policy: 'finalize_then_cancel',
            }, 202);
        }
        if (method === 'GET' && url.endsWith(`/api/tasks/${taskId}`) && window.__s3PendingCancel) {
            return jsonResp({
                ok: true, task_id: taskId, status: 'running',
                cancel_state: 'pending', stop_policy: 'finalize_then_cancel',
            });
        }
        return real(input, init);
    };
}
"""


def _menu_labels(menu) -> list[tuple[str, str]]:
    items = menu.locator(".task-control-item")
    return [
        (
            items.nth(i).get_attribute("data-task-control") or "",
            (items.nth(i).inner_text() or "").strip(),
        )
        for i in range(items.count())
    ]


def _chat_bubble_count(page) -> int:
    return page.locator(".chat-bubble").count()


def _menu_metrics(page, menu, trigger) -> dict:
    return page.evaluate(
        """
        ([menu, trigger]) => {
            const menuRect = menu.getBoundingClientRect();
            const triggerRect = trigger.getBoundingClientRect();
            const hitItems = Array.from(menu.querySelectorAll('.task-control-item')).map((item) => {
                const rect = item.getBoundingClientRect();
                const hit = document.elementFromPoint(
                    rect.left + rect.width / 2,
                    rect.top + rect.height / 2,
                );
                return Boolean(hit && item.contains(hit));
            });
            return {
                parentIsBody: menu.parentElement === document.body,
                position: getComputedStyle(menu).position,
                viewport: { width: window.innerWidth, height: window.innerHeight },
                menu: {
                    top: menuRect.top, right: menuRect.right,
                    bottom: menuRect.bottom, left: menuRect.left,
                },
                trigger: {
                    top: triggerRect.top, right: triggerRect.right,
                    bottom: triggerRect.bottom, left: triggerRect.left,
                },
                hitItems,
            };
        }
        """,
        [menu.element_handle(), trigger.element_handle()],
    )


def _assert_menu_geometry(metrics: dict, *, placement: str | None = None) -> None:
    assert metrics["parentIsBody"] is True
    assert metrics["position"] == "fixed"
    assert all(metrics["hitItems"]), metrics
    rect = metrics["menu"]
    viewport = metrics["viewport"]
    assert rect["left"] >= 7.5, metrics
    assert rect["right"] <= viewport["width"] - 7.5, metrics
    assert rect["top"] >= 7.5, metrics
    assert rect["bottom"] <= viewport["height"] - 7.5, metrics
    if placement == "below":
        assert rect["top"] >= metrics["trigger"]["bottom"] + 3, metrics
    elif placement == "above":
        assert rect["bottom"] <= metrics["trigger"]["top"] - 3, metrics


@pytest.mark.ui_browser

def _hold_live_root(data_dir, task_id: str, *, hold_seconds: float = 90.0) -> None:
    """Make ``task_id`` a GENUINELY running managed root for the test's duration: a queued
    row restored from the queue snapshot at boot, dispatched by the pool, and held inside
    its first model call by the mock model (``fixtures_mock_llm.HOLD_SECONDS``). Since
    54dfdc727 a replayed progress row's marker offers Stop only once the activity census
    or a durable record vouches for the root, so the fixture must vouch, not just replay.
    Reset the hold with ``_release_mock_model()`` in the test's ``finally``."""
    import datetime as _dt

    from tests import fixtures_mock_llm
    from ouroboros.task_results import write_task_result

    fixtures_mock_llm.HOLD_RELEASE.clear()
    fixtures_mock_llm.HOLD_SECONDS = float(hold_seconds)
    (data_dir / "state").mkdir(parents=True, exist_ok=True)
    live_task = {
        "id": task_id, "type": "task", "chat_id": 1, "priority": 0, "text": "the big thing",
        "description": "the big thing", "objective": "the big thing", "title": "The big thing",
        "root_task_id": task_id, "delegation_role": "root",
    }
    write_task_result(
        data_dir, task_id, "scheduled", result="Task is queued.", description="the big thing",
        objective="the big thing", chat_id=1, title="The big thing", delegation_role="root", root_task_id=task_id,
    )
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    (data_dir / "state" / "queue_snapshot.json").write_text(json.dumps({
        "_schema_version": 1, "ts": now, "reason": "ui_smoke_seed",
        "pending_count": 1, "running_count": 0, "reaping_count": 0,
        "acceptance_fences": [], "budget_root_fences": [],
        "pending": [{"id": task_id, "type": "task", "priority": 0, "attempt": 1, "queued_at": now,
                     "queue_seq": 1, "task": live_task}],
        "running": [],
    }), encoding="utf-8")


def _release_mock_model() -> None:
    from tests import fixtures_mock_llm

    fixtures_mock_llm.HOLD_SECONDS = 0.0
    fixtures_mock_llm.HOLD_RELEASE.set()


def test_s3_chat_card_dropdown_hurry_and_soft_stop(direct_server_with_data):
    """Chat surface: frozen dropdown, no-chat hurry with idempotent request_id
    retry, and the soft stop collapsing the pending menu to the escalation."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    logs_dir = data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    # Seed while no server runs (the live loop rewrites the queue snapshot every tick).
    direct_server_with_data["stop_server"]()
    (logs_dir / "chat.jsonl").write_text("", encoding="utf-8")
    (logs_dir / "progress.jsonl").write_text(
        json.dumps({
            "ts": "2026-08-15T10:00:00+00:00", "chat_id": 1, "task_id": "live-root",
            "content": "Working on the big thing", "cancelable": True,
        }) + "\n",
        encoding="utf-8",
    )
    _hold_live_root(data_dir, "live-root")
    direct_server_with_data["start_server"]()

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            try:
                # Retain each observer callback so the singleton-generation
                # guard can be tested against a late delivery from a closed
                # predecessor. Native observation still drives the page.
                page.add_init_script(
                    """
                    (() => {
                        const NativeIntersectionObserver = window.IntersectionObserver;
                        window.__s3IntersectionObservers = [];
                        window.IntersectionObserver = class {
                            constructor(callback, options) {
                                this.callback = callback;
                                this.native = new NativeIntersectionObserver(callback, options);
                                window.__s3IntersectionObservers.push(this);
                            }
                            observe(target) { return this.native.observe(target); }
                            unobserve(target) { return this.native.unobserve(target); }
                            disconnect() { return this.native.disconnect(); }
                            takeRecords() { return this.native.takeRecords(); }
                        };
                    })();
                    """,
                )
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                live = page.locator('.chat-live-card[data-task-id="live-root"]')
                live.wait_for(state="attached", timeout=30_000)
                trigger = live.locator("[data-cancel-run]")
                trigger.wait_for(state="attached", timeout=30_000)
                assert trigger.inner_text().strip() == "Stop…"
                page.evaluate(_FETCH_STUB_JS, "live-root")
                bubbles_before = _chat_bubble_count(page)

                # 1) The dropdown offers EXACTLY the three frozen actions, in
                #    order and verbatim; Escape dismisses = the run continues.
                trigger.click()
                menu = page.locator("body > .task-control-menu")
                menu.wait_for(state="visible", timeout=10_000)
                assert _menu_labels(menu) == EXPECTED_ACTIONS
                _assert_menu_geometry(_menu_metrics(page, menu, trigger), placement="below")
                page.screenshot(
                    path=str(data_dir.parent / "s3-chat-control-open.png"), full_page=False,
                )
                page.keyboard.press("Escape")
                menu.wait_for(state="detached", timeout=10_000)
                assert trigger.is_enabled()
                assert trigger.get_attribute("aria-expanded") == "false"
                assert trigger.evaluate("(node) => document.activeElement === node") is True
                assert page.evaluate("() => window.__s3Calls.length") == 0

                # Owner decision 2A survives the browser's default pointer
                # focus without cancelling the outside element's actual click.
                page.evaluate(
                    """
                    () => {
                        const outside = document.createElement('button');
                        outside.id = 's3-outside-focus-target';
                        outside.textContent = 'Outside';
                        Object.assign(outside.style, {
                            position: 'fixed', left: '8px', top: '8px', zIndex: '200',
                        });
                        window.__s3OutsideClicks = 0;
                        outside.addEventListener('click', () => { window.__s3OutsideClicks += 1; });
                        document.body.appendChild(outside);
                    }
                    """,
                )
                trigger.click()
                menu.wait_for(state="visible", timeout=10_000)
                page.locator("#s3-outside-focus-target").click()
                menu.wait_for(state="detached", timeout=10_000)
                page.wait_for_function(
                    "() => document.activeElement?.matches('[data-cancel-run]')",
                    timeout=10_000,
                )
                assert page.evaluate("() => window.__s3OutsideClicks") == 1
                assert page.evaluate("() => window.__s3Calls.length") == 0
                page.locator("#s3-outside-focus-target").evaluate("node => node.remove()")

                # A queued callback from observer A must not close successor B.
                observer_count = page.evaluate("() => window.__s3IntersectionObservers.length")
                trigger.click()
                menu.wait_for(state="visible", timeout=10_000)
                page.keyboard.press("Escape")
                menu.wait_for(state="detached", timeout=10_000)
                trigger.click()
                menu.wait_for(state="visible", timeout=10_000)
                page.evaluate(
                    """
                    ([index, trigger]) => {
                        window.__s3IntersectionObservers[index].callback([{
                            target: trigger, isIntersecting: false,
                        }]);
                    }
                    """,
                    [observer_count, trigger.element_handle()],
                )
                page.wait_for_timeout(50)
                assert menu.count() == 1
                assert trigger.get_attribute("aria-expanded") == "true"
                page.keyboard.press("Escape")
                menu.wait_for(state="detached", timeout=10_000)

                # The stable Chat DOM owns the deterministic viewport-flip proof.
                page.evaluate(
                    """
                    () => {
                        const live = document.querySelector(
                            '.chat-live-card[data-task-id="live-root"]',
                        );
                        window.__s3OriginalLiveStyle = live.getAttribute('style');
                        Object.assign(live.style, {
                            position: 'fixed', left: '16px', right: 'auto',
                            bottom: '8px', top: 'auto', margin: '0', zIndex: '6',
                        });
                    }
                    """,
                )
                trigger.click()
                menu.wait_for(state="visible", timeout=10_000)
                _assert_menu_geometry(_menu_metrics(page, menu, trigger), placement="above")
                page.screenshot(
                    path=str(data_dir.parent / "s3-chat-control-flipped.png"), full_page=False,
                )
                page.keyboard.press("Escape")
                menu.wait_for(state="detached", timeout=10_000)
                page.evaluate(
                    """
                    () => {
                        const live = document.querySelector(
                            '.chat-live-card[data-task-id="live-root"]',
                        );
                        if (window.__s3OriginalLiveStyle === null) live.removeAttribute('style');
                        else live.setAttribute('style', window.__s3OriginalLiveStyle);
                    }
                    """,
                )

                # Owner decision 1D: any nested scroll or resize dismisses
                # without issuing a task-control action.
                trigger.click()
                menu.wait_for(state="visible", timeout=10_000)
                page.locator("#chat-messages").dispatch_event("scroll")
                menu.wait_for(state="detached", timeout=10_000)
                assert trigger.get_attribute("aria-expanded") == "false"
                assert trigger.evaluate("(node) => document.activeElement === node") is True
                assert page.evaluate("() => window.__s3Calls.length") == 0

                trigger.click()
                menu.wait_for(state="visible", timeout=10_000)
                page.set_viewport_size({"width": 1360, "height": 920})
                menu.wait_for(state="detached", timeout=10_000)
                assert trigger.get_attribute("aria-expanded") == "false"
                assert trigger.evaluate("(node) => document.activeElement === node") is True
                assert page.evaluate("() => window.__s3Calls.length") == 0

                # Narrow viewports must shrink the menu instead of allowing the
                # historical 220px minimum to overflow the page.
                page.set_viewport_size({"width": 320, "height": 700})
                trigger.click()
                menu.wait_for(state="visible", timeout=10_000)
                _assert_menu_geometry(_menu_metrics(page, menu, trigger))
                page.screenshot(
                    path=str(data_dir.parent / "s3-chat-control-narrow.png"), full_page=False,
                )
                page.keyboard.press("Escape")
                menu.wait_for(state="detached", timeout=10_000)

                # A genuinely short viewport height-clamps the portal. Its own
                # scroll must keep it open so every action remains reachable;
                # only scroll that can stale anchor geometry invokes 1D close.
                page.set_viewport_size({"width": 320, "height": 150})
                page.evaluate(
                    """
                    () => {
                        const live = document.querySelector(
                            '.chat-live-card[data-task-id="live-root"]',
                        );
                        window.__s3ShortLiveStyle = live.getAttribute('style');
                        Object.assign(live.style, {
                            position: 'fixed', left: '8px', right: 'auto',
                            top: '0px', bottom: 'auto', width: '304px',
                            margin: '0', zIndex: '6',
                        });
                        const trigger = live.querySelector('[data-cancel-run]');
                        live.style.top = `${70 - trigger.getBoundingClientRect().top}px`;
                    }
                    """,
                )
                trigger.click()
                menu.wait_for(state="visible", timeout=10_000)
                short_metrics = menu.evaluate(
                    """
                    node => ({
                        clientHeight: node.clientHeight,
                        scrollHeight: node.scrollHeight,
                        top: node.getBoundingClientRect().top,
                        bottom: node.getBoundingClientRect().bottom,
                    })
                    """,
                )
                assert short_metrics["scrollHeight"] > short_metrics["clientHeight"], short_metrics
                assert short_metrics["top"] >= 7.5, short_metrics
                assert short_metrics["bottom"] <= 142.5, short_metrics
                items = menu.locator(".task-control-item")
                for index in range(items.count()):
                    menu.evaluate(
                        """
                        (node, itemIndex) => {
                            node.scrollTop = node.querySelectorAll('.task-control-item')[itemIndex]
                                .offsetTop;
                        }
                        """,
                        index,
                    )
                    page.wait_for_timeout(50)
                    assert menu.count() == 1
                    assert trigger.get_attribute("aria-expanded") == "true"
                    assert items.nth(index).evaluate(
                        """
                        item => {
                            const rect = item.getBoundingClientRect();
                            const hit = document.elementFromPoint(
                                rect.left + rect.width / 2,
                                rect.top + rect.height / 2,
                            );
                            return Boolean(hit && item.contains(hit));
                        }
                        """,
                    ) is True
                assert menu.evaluate("node => node.scrollTop") > 0
                page.screenshot(
                    path=str(data_dir.parent / "s3-chat-control-short-scroll.png"),
                    full_page=False,
                )
                page.keyboard.press("Escape")
                menu.wait_for(state="detached", timeout=10_000)
                page.evaluate(
                    """
                    () => {
                        const live = document.querySelector(
                            '.chat-live-card[data-task-id="live-root"]',
                        );
                        if (window.__s3ShortLiveStyle === null) live.removeAttribute('style');
                        else live.setAttribute('style', window.__s3ShortLiveStyle);
                    }
                    """,
                )
                page.set_viewport_size({"width": 1440, "height": 1000})

                # 2) "Hurry up": typed control + LOCAL toast; the chat DOM
                #    gains ZERO bubbles (HQ1's no-chat contract, live DOM).
                trigger.click()
                menu.wait_for(state="visible", timeout=10_000)
                menu.locator('[data-task-control="hurry"]').click()
                toast = page.locator(".toast", has_text="Hurry up: accepted")
                toast.wait_for(state="visible", timeout=10_000)
                calls = page.evaluate("() => window.__s3Calls")
                assert [c["kind"] for c in calls] == ["hurry"]
                first_body = calls[0]["body"]
                assert set(first_body.keys()) == {"request_id"}, first_body
                assert first_body["request_id"].startswith("hurry-")
                assert _chat_bubble_count(page) == bubbles_before

                # 3) Retry of the SAME owner intent reuses the SAME request_id
                #    (idempotent endpoint ack -> the duplicate toast wording).
                trigger.click()
                menu.wait_for(state="visible", timeout=10_000)
                menu.locator('[data-task-control="hurry"]').click()
                page.locator(".toast", has_text="already accepted").wait_for(
                    state="visible", timeout=10_000,
                )
                calls = page.evaluate("() => window.__s3Calls")
                assert [c["kind"] for c in calls] == ["hurry", "hurry"]
                assert calls[1]["body"]["request_id"] == first_body["request_id"]
                assert _chat_bubble_count(page) == bubbles_before

                # 4) "Wrap up": 202 keeps the card honestly LIVE as
                #    "Finalizing…", the trigger stays reachable, and the
                #    pending menu offers ONLY "Stop now".
                trigger.click()
                menu.wait_for(state="visible", timeout=10_000)
                menu.locator('[data-task-control="finalize"]').click()
                page.wait_for_function(
                    "() => document.querySelector('.chat-live-card[data-task-id=\"live-root\"]"
                    " [data-live-phase]')?.textContent === 'Finalizing…'",
                    timeout=10_000,
                )
                calls = page.evaluate("() => window.__s3Calls")
                assert calls[-1]["kind"] == "cancel"
                assert calls[-1]["body"] == {"cascade": True, "stop_policy": "finalize_then_cancel"}
                trigger.wait_for(state="attached", timeout=10_000)
                page.wait_for_function(
                    "() => !document.querySelector('.chat-live-card[data-task-id=\"live-root\"]"
                    " [data-cancel-run]')?.disabled",
                    timeout=10_000,
                )
                trigger.click()
                menu.wait_for(state="visible", timeout=10_000)
                assert _menu_labels(menu) == [("stop_now", "Stop now")]
                page.keyboard.press("Escape")
                menu.wait_for(state="detached", timeout=10_000)
                assert _chat_bubble_count(page) == bubbles_before
                page.screenshot(
                    path=str(data_dir.parent / "s3-chat-control.png"), full_page=True,
                )
            finally:
                browser.close()
                _release_mock_model()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.mark.ui_browser
def test_s3_activity_tab_dropdown_parity_and_no_chat_hurry(direct_server_with_data):
    """Activity surface: the SAME shared dropdown (labels, order, hurry flow)
    and the same no-chat acknowledgement — product-wide parity (Q2/HQ1)."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    logs_dir = data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "chat.jsonl").write_text("", encoding="utf-8")
    (logs_dir / "progress.jsonl").write_text("", encoding="utf-8")

    # The activity queue view is fed by GET /api/tasks?queue_only=1; stub it at
    # the same fetch seam (no real queue mutation from a UI test).
    activity_stub = """
    (taskId) => {
        const real = window.fetch.bind(window);
        window.fetch = (input, init = {}) => {
            const url = String(input && input.url ? input.url : input);
            const method = String((init && init.method) || (input && input.method) || 'GET').toUpperCase();
            if (method === 'GET' && url.includes('/api/tasks?queue_only=1')) {
                return Promise.resolve(new Response(JSON.stringify({
                    ok: true,
                    queue: {
                        running: [{ id: taskId, type: 'task', runtime_sec: 42,
                                    task: { id: taskId, title: 'Active task' } }],
                        pending: [],
                    },
                }), { status: 200, headers: { 'Content-Type': 'application/json' } }));
            }
            if (method === 'GET' && url.endsWith(`/api/tasks/${taskId}`)) {
                return Promise.resolve(new Response(JSON.stringify({
                    ok: true, task_id: taskId, status: 'running',
                }), { status: 200, headers: { 'Content-Type': 'application/json' } }));
            }
            return real(input, init);
        };
    }
    """

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                page.wait_for_selector("#page-chat", timeout=30_000)
                page.evaluate(activity_stub, "act-1")
                page.evaluate(_FETCH_STUB_JS.replace(
                    "const real = window.fetch.bind(window);",
                    "const real = window.fetch;",
                ), "act-1")

                page.click('[data-nav-page="dashboard"]')
                page.click('[data-dashboard-tab="activity"]')
                row_btn = page.locator('#dashboard-panel-activity [data-act="task-control"][data-id="act-1"]')
                row_btn.wait_for(state="visible", timeout=30_000)
                assert row_btn.inner_text().strip() == "Stop…"

                # Parity: the same shared menu with the exact frozen actions.
                row_btn.click()
                menu = page.locator("body > .task-control-menu")
                menu.wait_for(state="visible", timeout=10_000)
                assert _menu_labels(menu) == EXPECTED_ACTIONS
                assert menu.evaluate("(node) => node.parentElement === document.body") is True
                assert menu.evaluate("(node) => getComputedStyle(node).position") == "fixed"
                page.screenshot(
                    path=str(data_dir.parent / "s3-activity-control-open.png"), full_page=True,
                )

                # Activity refresh replaces its subtree before awaiting network
                # reads. The trigger observer must dispose the body portal.
                page.evaluate(
                    """
                    () => window.dispatchEvent(new CustomEvent(
                        'ouro:dashboard-subtab-shown', { detail: { tab: 'activity' } },
                    ))
                    """,
                )
                menu.wait_for(state="detached", timeout=10_000)
                assert page.evaluate("() => window.__s3Calls.length") == 0
                row_btn.wait_for(state="visible", timeout=30_000)

                # "Hurry up" from Activity: local toast only, no chat bubble.
                # Baseline captured HERE (post-navigation) so async startup
                # bubbles (awaken banner, history replay) are already settled.
                bubbles_before = _chat_bubble_count(page)
                row_btn.click()
                menu.wait_for(state="visible", timeout=10_000)
                menu.locator('[data-task-control="hurry"]').click()
                page.locator(".toast", has_text="Hurry up: accepted").wait_for(
                    state="visible", timeout=10_000,
                )
                calls = page.evaluate("() => window.__s3Calls")
                assert [c["kind"] for c in calls] == ["hurry"]
                assert set(calls[0]["body"].keys()) == {"request_id"}
                assert _chat_bubble_count(page) == bubbles_before
                # Belt-and-suspenders: no bubble anywhere mentions the control.
                assert page.locator(".chat-bubble", has_text="Hurry up").count() == 0
                page.screenshot(
                    path=str(data_dir.parent / "s3-activity-control.png"), full_page=True,
                )
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise
