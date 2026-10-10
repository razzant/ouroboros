"""Native accounting waits in real Chat/Logs, including source-backed history reload.

Only the provider, bootstrap/state envelope and WebSocket transport are fixtures.
Wait production, supervisor persistence, history/log endpoints and UI reducers run
their production code against isolated files; there is no supervisor process.
"""
from __future__ import annotations

import json
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import usage_accounting as ua
from ouroboros.gateway.history import make_chat_history_endpoint
from ouroboros.gateway.logs import api_logs_tail
from ouroboros.gateway.tasks import api_task_get
from ouroboros.model_wait import task_model_wait_scope
from ouroboros.task_results import write_task_result
from ouroboros.utils import append_jsonl
from supervisor.events_worker_reports import _handle_log_event
from tests.test_subscription_setup_browser import subscription_ui as subscription_ui, capture
from tests.test_usage_lock_continuity import held_lock
from tests._usage_store_testing import request, root as root

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


def test_two_native_accounting_waits_in_chat_logs_and_reload(subscription_ui, root, monkeypatch):
    ui, task = subscription_ui, "accounting-wait-task"
    page, events, sockets, published = ui["page"], queue.Queue(), [], []
    evidence = Path(os.environ["OUROBOROS_TEST_TEMP_ROOT"]) / "usage-wait-browser"
    monkeypatch.setenv("OUROBOROS_UI_EVIDENCE_DIR", str(evidence))
    task_row = {"id": task, "chat_id": 1, "type": "task", "objective": "Check accounting access"}
    write_task_result(root, task, "running", result="", chat_id=1)
    ctx = SimpleNamespace(RUNNING={task: {"task": task_row}}, DRIVE_ROOT=root,
                          append_jsonl=append_jsonl, bridge=SimpleNamespace(push_log=published.append))
    page.route_web_socket("**/ws", lambda socket: sockets.append(socket))
    app = Starlette(routes=[Route("/api/chat/history", make_chat_history_endpoint(root)),
                            Route("/api/logs/{name}", api_logs_tail), Route("/api/tasks/{task_id}", api_task_get)])
    app.state.drive_root = root
    second, finish = threading.Event(), threading.Event()
    sends = []

    def execute():
        with task_model_wait_scope(task=task_row, drive_root=root, event_queue=events, worker_slot_held=False):
            for index in range(2):
                if index:
                    assert second.wait(20)
                ua.execute_physical_attempt(request(root, task_id=task, root_task_id=task),
                                            lambda: sends.append(index) or {"usage": {"cost": .1}})
            finish.set()

    def pump(phase):
        try:
            event = events.get(timeout=10)
        except queue.Empty:
            # Report a finished producer's failure, not only its missing event.
            if future.done():
                future.result()
            raise
        assert event["data"]["phase"] == phase
        _handle_log_event(event, ctx)
        payload = published[-1]
        assert payload["role"] == "system" and payload["chat_id"] == 1
        sockets[-1].send(json.dumps({"type": "log", "chat_id": 1, "data": payload}))
        return payload

    def labels(selector):
        return page.locator(selector).all_text_contents()

    chat_lines = '#chat-messages .chat-live-line-title'
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=1) as pool:
        def forward(route):
            parsed = urlparse(route.request.url)
            response = client.get(parsed.path + ("?" + parsed.query if parsed.query else ""))
            route.fulfill(status=response.status_code, content_type="application/json", body=response.content)
        page.route("**/api/chat/history*", forward)
        page.route("**/api/logs/**", forward)
        page.route("**/api/tasks/**", forward)
        page.goto(ui["url"] + "/")
        page.wait_for_selector('#chat-messages > .chat-load-older', state="attached")
        with held_lock(root, timeout=30) as release:
            future = pool.submit(execute)
            try:
                first = pump("entered")
                page.locator('#chat-messages .chat-live-card').first.wait_for(state="attached")
                card = page.locator('#chat-messages .chat-live-card').first
                card.locator('.chat-live-summary-button').click()
                capture(page, "chat-first-entered")
                release.set()
                pump("ended")
                page.get_by_text("Accounting wait ended", exact=True).first.wait_for(state="attached")
                capture(page, "chat-first-ended")
                with held_lock(root, timeout=30) as release_second:
                    second.set()
                    next_wait = pump("entered")
                    assert next_wait["episode_id"] != first["episode_id"]
                    page.wait_for_function("() => [...document.querySelectorAll('#chat-messages .chat-live-line-title')].filter(n => n.textContent === 'Waiting for accounting access').length === 2")
                    assert labels(chat_lines)[-1] == "Waiting for accounting access"
                    capture(page, "chat-second-entered")
                    page.click('[data-nav-page="dashboard"]')
                    page.locator('#log-entries .log-task-details > summary').first.click()
                    assert page.locator('#log-entries .log-task-timeline').get_by_text("Waiting for accounting access", exact=True).count() == 2
                    capture(page, "logs-second-entered")
                    page.click('[data-nav-page="chat"]')
                    release_second.set()
                    pump("ended")
                future.result(timeout=10)
            finally:
                release.set()
                second.set()
        page.wait_for_function("() => [...document.querySelectorAll('#chat-messages .chat-live-line-title')].filter(n => n.textContent === 'Accounting wait ended').length === 2")
        expected = ["Waiting for accounting access", "Accounting wait ended"] * 2
        assert labels(chat_lines) == expected
        capture(page, "chat-two-complete")
        page.click('[data-nav-page="dashboard"]')
        page.wait_for_selector('#page-logs', state="visible")
        if not page.locator('#log-entries .log-task-details').first.get_attribute('open') == '':
            page.locator('#log-entries .log-task-details > summary').first.click()
        assert page.locator('#log-entries .log-task-timeline').get_by_text("Waiting for accounting access", exact=True).count() == 2
        assert page.locator('#log-entries .log-task-timeline').get_by_text("Accounting wait ended", exact=True).count() == 2
        capture(page, "logs-two-complete")
        page.reload()
        page.click('[data-nav-page="chat"]')
        page.locator('#chat-messages .chat-live-card').first.locator('.chat-live-summary-button').click()
        page.wait_for_function("() => document.querySelectorAll('#chat-messages .chat-live-line-title').length === 4")
        assert labels(chat_lines) == expected
        capture(page, "chat-history-reloaded")
        page.click('[data-nav-page="dashboard"]')
        page.wait_for_selector('#page-logs', state="visible")
        assert page.locator('#log-entries .log-task-timeline').get_by_text("Waiting for accounting access", exact=True).count() == 2
        assert page.locator('#log-entries .log-task-timeline').get_by_text("Accounting wait ended", exact=True).count() == 2
        page.locator('#log-entries .log-task-details > summary').first.click()
        capture(page, "logs-history-reloaded")
        history = client.get('/api/chat/history?chat_id=1').json()
    assert sends == [0, 1] and finish.is_set()
    evidence.mkdir(exist_ok=True)
    (evidence / "native-events.json").write_text(json.dumps(published, indent=2))
    (evidence / "history.json").write_text(json.dumps(history, indent=2))
    print(f"USAGE_WAIT_BROWSER_EVIDENCE {evidence}")
