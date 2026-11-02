"""Native accounting waits in real Chat/Logs, including source-backed history reload.

Only the provider, bootstrap/state envelope and WebSocket transport are fixtures.
Wait production, supervisor persistence, history/log endpoints and UI reducers run
their production code against isolated files; there is no supervisor process.
"""
from __future__ import annotations

import json
import queue
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import dialogue_store as ds, usage_accounting as ua
from ouroboros.gateway.history import make_chat_history_endpoint
from ouroboros.gateway.logs import api_logs_tail
from ouroboros.gateway.tasks import api_task_get
from ouroboros.model_wait import task_model_wait_scope
from ouroboros.task_results import write_task_result
from ouroboros.utils import append_jsonl, utc_now_iso
from supervisor.events_worker_reports import _handle_log_event
from tests._shared import producer_evidence_dir
from tests.test_subscription_setup_browser import subscription_ui as subscription_ui, capture
from tests.test_usage_lock_continuity import held_lock
from tests._usage_store_testing import request, root as root

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


def test_accounting_waits_fold_into_one_receipt_row_while_logs_keep_each_phase(subscription_ui, root, monkeypatch, tmp_path):
    """Two real accounting waits. A wait is how the host ran, not the turn's work: alone it makes no
    card; beside the turn's own note both episodes are ONE receipt row, live and on reload, while
    Logs keep every phase."""
    ui, task = subscription_ui, "accounting-wait-task"
    page, events, sockets, published = ui["page"], queue.Queue(), [], []
    evidence = producer_evidence_dir(tmp_path / "usage-wait-browser", "usage-wait")
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

    def wait_row(pattern):
        """The card's one accounting-wait row, once it reads ``pattern``; its text."""
        page.wait_for_function("""source => {
            const rows = [...document.querySelectorAll('#chat-messages .chat-live-card .chat-live-line-title')]
                .map(node => node.textContent).filter(text => /^(Waiting for accounting access|Accounting wait)/.test(text));
            return rows.length === 1 && new RegExp(source).test(rows[0]);
        }""", arg=pattern)
        return page.locator('#chat-messages .chat-live-card .chat-live-line-title').filter(
            has_text=re.compile(r"^(Waiting for accounting access|Accounting wait)")).first.text_content()

    def logs_phases(expected):
        page.click('[data-nav-page="dashboard"]')
        page.wait_for_selector('#page-logs', state="visible")
        if not page.locator('#log-entries .log-task-details').first.get_attribute('open') == '':
            page.locator('#log-entries .log-task-details > summary').first.click()
        timeline = page.locator('#log-entries .log-task-timeline')
        for label, count in expected.items():
            assert timeline.get_by_text(label, exact=True).count() == count, label

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
                # A later frame that does render proves Chat already handled the wait, which made no card.
                sockets[-1].send(json.dumps({"type": "chat", "role": "assistant", "chat_id": 1,
                                             "content": "Control reply after the first wait"}))
                page.get_by_text("Control reply after the first wait", exact=True).wait_for()
                assert page.locator('#chat-messages .chat-live-card').count() == 0, "a wait alone made a card"
                capture(page, "chat-wait-alone")
                note = {"ts": utc_now_iso(), "task_id": task, "chat_id": 1, "role": "assistant", "is_progress": True,
                        "narration": True, "content": "Checking accounting access", "text": "Checking accounting access"}
                ds.append(root, "progress", note)
                sockets[-1].send(json.dumps({"type": "chat", **note}))
                card = page.locator('#chat-messages .chat-live-card').first
                card.wait_for(state="attached")
                card.locator('.chat-live-summary-button [data-live-disclosure]').click()
                wait_row(r"^Waiting for accounting access$")
                capture(page, "chat-first-entered")
                release.set()
                pump("ended")
                wait_row(r"^Accounting wait ended · \S")
                capture(page, "chat-first-ended")
                with held_lock(root, timeout=30) as release_second:
                    second.set()
                    next_wait = pump("entered")
                    assert next_wait["episode_id"] != first["episode_id"]
                    wait_row(r"^Waiting for accounting access · Accounting wait ended · \S")
                    capture(page, "chat-second-entered")
                    logs_phases({"Waiting for accounting access": 2, "Accounting wait ended": 1})
                    capture(page, "logs-second-entered")
                    page.click('[data-nav-page="chat"]')
                    release_second.set()
                    pump("ended")
                future.result(timeout=10)
            finally:
                release.set()
                second.set()
        live = wait_row(r"^Accounting waits ×2 · longest \S")
        assert page.locator('#chat-messages .chat-live-card').count() == 1
        capture(page, "chat-two-complete")
        logs_phases({"Waiting for accounting access": 2, "Accounting wait ended": 2})
        capture(page, "logs-two-complete")
        ui["errors"].watch(page)  # the reload may cut off the page's own reads (tests/ui_console_errors.py)
        page.reload()
        page.click('[data-nav-page="chat"]')
        page.locator('#chat-messages .chat-live-card').first.locator('.chat-live-summary-button [data-live-disclosure]').click()
        assert wait_row(r"^Accounting waits ×2 · longest \S") == live
        capture(page, "chat-history-reloaded")
        logs_phases({"Waiting for accounting access": 2, "Accounting wait ended": 2})
        capture(page, "logs-history-reloaded")
        history = client.get('/api/chat/history?chat_id=1').json()
    assert sends == [0, 1] and finish.is_set()
    evidence.mkdir(exist_ok=True)
    (evidence / "native-events.json").write_text(json.dumps(published, indent=2))
    (evidence / "history.json").write_text(json.dumps(history, indent=2))
    print(f"USAGE_WAIT_BROWSER_EVIDENCE {evidence}")
