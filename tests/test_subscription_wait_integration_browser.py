"""Browser actions through real wait ingress, mailbox, persistence and LLM continuation.

The task is represented by isolated RUNNING metadata, not a supervisor worker
process. LLMClient, TaskModelWait, the physical ledger, decision endpoint,
supervisor wait projection, history and settings writer are real. Only provider
replies and the browser's unrelated bootstrap APIs are controlled fixtures.
"""
from __future__ import annotations

import copy
import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from urllib.parse import urlparse

import httpx
import pytest
from openai import OpenAI
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from tests import test_llm_claudexor as llm_fixture
from tests import test_model_wait as wait_fixture
from tests import test_subscription_setup_browser as browser_fixture

subscription_ui = browser_fixture.subscription_ui
setup = llm_fixture.setup
live_wait = wait_fixture.live_wait
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
TASK = "task-one"
MODEL = llm_fixture.MODEL


@pytest.fixture
def integrated_wait(subscription_ui, live_wait, monkeypatch):
    from ouroboros import config, model_wait, pricing, usage_accounting as accounting
    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros.gateway.task_decision import api_decision_answer
    from ouroboros.gateway.tasks import api_task_get
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.utils import append_jsonl
    from supervisor import queue as task_queue
    from supervisor.task_model_wait import handle_task_model_wait

    ui = subscription_ui
    page = ui["page"]
    root, gateway, llm, controller, events, _unused_direct_decide = live_wait
    controller.task["chat_id"] = 1
    monkeypatch.setattr(config, "SETTINGS_PATH", root / "settings.json")
    monkeypatch.setattr(config, "CLAUDEXOR_MODEL_POLL_INTERVAL_SEC", 0.01)
    monkeypatch.setattr(model_wait, "time", time)
    monkeypatch.setattr(pricing, "_fetch_live_rows", lambda *_: [])
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-not-a-live-key")
    monkeypatch.setenv("OUROBOROS_MODEL_ACCOUNTS", json.dumps({"main": "personal", "light": "work"}))
    settings = {"OUROBOROS_MODEL": MODEL, "OUROBOROS_MODEL_LIGHT": MODEL,
                "OUROBOROS_MODEL_ACCOUNTS": json.dumps({"main": "personal", "light": "work"}),
                "OUROBOROS_CONTEXT_MODE": "max", "OUROBOROS_RUNTIME_MODE": "advanced"}
    root.mkdir(parents=True, exist_ok=True)
    config.SETTINGS_PATH.write_text(json.dumps(settings), encoding="utf-8")
    original_settings = config.SETTINGS_PATH.read_bytes()
    ui["settings"].update(settings)
    # The wait picker offers an API lane per provider whose credential is stored.
    ui["settings"]["OPENAI_API_KEY"] = "***set***"

    source_entered, source_release = threading.Event(), threading.Event()
    catalog_entered, catalog_release = threading.Event(), threading.Event()
    def sources():
        source_entered.set()
        if not source_release.wait(20):
            raise AssertionError("test did not release source metadata")
        return {"sources": [{"id": "codex", "credentialHarness": "codex"}]}

    def catalog(*_, **_kw):
        catalog_entered.set()
        if not catalog_release.wait(20):
            raise AssertionError("test did not release the unavailable catalog")
        raise llm_fixture.ClaudexorUnavailable("subscription_window_exhausted", "controlled quota remains exhausted")

    monkeypatch.setattr(llm, "claudexor_model_sources", sources)
    monkeypatch.setattr(llm, "claudexor_model_catalog", catalog)

    def served(account):
        return {**llm_fixture.ROUTE, "credentialProfileId": account, "accountFingerprint": f"fixture-{account}"}

    main_reply = llm_fixture.result(route=served("personal"))
    review_reply = llm_fixture.result(route=served("personal"))
    review_reply["message"] = {"role": "assistant", "content": "Completed independent review."}
    refusal = llm_fixture.result(outcome="failed", route=served("work"), problem={
        "code": "subscription_window_exhausted", "message": "controlled quota exhausted",
        "context": {"resetsAt": "2099-01-01T00:00:00Z"},
    })
    gateway.results = [main_reply, review_reply, refusal]
    gateway.dispatch = ["response_received", "response_received", "not_started"]
    completed = []
    main, _ = llm.chat([{"role": "user", "content": "Read two files."}], MODEL, model_role="main")
    tools = []
    for index, tool in enumerate(main["tool_calls"], 1):
        # Deterministic fixture tools really write once; the waiter must resume
        # the pending LLM call, not restart this already-completed work.
        artifact = root / f"completed-tool-{index}.txt"
        artifact.write_text(f"tool result {index}", encoding="utf-8")
        completed.append(tool["id"])
        tools.append({"role": "tool", "tool_call_id": tool["id"], "content": artifact.read_text()})
    review, _ = llm.chat([{"role": "user", "content": "Review completed work."}], MODEL,
                         model_role="reviewer:finished", model_account_override="personal")
    completed.append("review")
    messages = [{"role": "user", "content": "Summarize completed work."}, main, *tools,
                {"role": "user", "content": review["content"]}]
    original_messages = copy.deepcopy(messages)

    api_requests = []
    def api_provider(request):
        payload = json.loads(request.content)
        api_requests.append(payload)
        return httpx.Response(200, json={
            "id": "fixture-api-generation", "object": "chat.completion", "created": 1,
            "model": payload["model"], "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Continued exactly once."}}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 6, "total_tokens": 46},
        })
    http_client = httpx.Client(transport=httpx.MockTransport(api_provider))
    sdk = OpenAI(api_key="fixture-not-a-live-key", base_url="http://provider.invalid/v1",
                 http_client=http_client, max_retries=0)
    monkeypatch.setattr(llm, "_get_remote_client", lambda _target: sdk)

    sockets, decisions, responses, published = [], [], [], []
    page.route_web_socket("**/ws", lambda socket: sockets.append(socket))
    app = Starlette(routes=[
        Route("/api/decisions", api_decision_answer, methods=["POST"]),
        Route("/api/tasks/{task_id}", api_task_get),
        Route("/api/chat/history", make_chat_history_endpoint(root)),
    ])
    app.state.drive_root = root
    projection_context = SimpleNamespace(RUNNING=task_queue.RUNNING, DRIVE_ROOT=root,
        append_jsonl=append_jsonl, bridge=SimpleNamespace(push_log=published.append))

    def project(event):
        before = len(published)
        handle_task_model_wait(event, projection_context)
        if sockets and len(published) > before:
            sockets[-1].send(json.dumps({"type": "log", "chat_id": 1, "data": published[-1]}))

    def pump(predicate):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                event = events.get(timeout=max(0.01, deadline - time.monotonic()))
            except queue.Empty:
                break
            project(event)
            if predicate(event):
                return event
        raise AssertionError("expected real TaskModelWait event was not published")

    def pending_call():
        with accounting.usage_scope(accounting.UsageScope(
                drive_root=root, task_id=TASK, root_task_id=TASK)):
            # A never-dispatched quota refusal releases its claim. Exactly
            # one resumed provider generation can use this existing limiter.
            with accounting.physical_attempt_limit(1):
                return llm.chat(messages, MODEL, model_role="light")

    def finish(answer):
        result = write_task_result(root, TASK, "completed", result=answer["content"])
        # The activity envelope belongs to this process-free fixture. Ending
        # it must remove its RUNNING metadata as an actual worker finish does.
        with task_queue._queue_lock:
            task_queue.RUNNING.pop(TASK, None)
        return result

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="subscription-wait-acceptance")
    future = pool.submit(model_wait.copy_wait_context().run, pending_call)
    try:
        assert source_entered.wait(5)
        first = pump(lambda event: event["state"] == "waiting")
        assert first["role"] == "light" and first["credential_profile_id"] == "work"
        with TestClient(app) as client:
            def forward(route):
                request = route.request
                parsed = urlparse(request.url)
                path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
                if request.method == "POST":
                    body = request.post_data_json
                    decisions.append(body)
                    response = client.post(path, json=body)
                    responses.append({"status": response.status_code, "body": response.json()})
                else:
                    response = client.get(path)
                route.fulfill(status=response.status_code, content_type="application/json", body=response.content)

            page.route("**/api/decisions", forward)
            page.route(f"**/api/tasks/{TASK}", forward)
            page.route("**/api/chat/history*", forward)
            def state_response(route):
                # Only activity envelope is fixture-owned. Its wait projection
                # is the real supervisor handler's current queue metadata.
                task = task_queue.RUNNING.get(TASK, {}).get("task", {})
                route.fulfill(content_type="application/json", body=json.dumps({
                    "supervisor_ready": True, "projects": [], "active_chat_activities": [{
                        "activity_id": TASK, "kind": "managed_task", "chat_id": 1,
                        "phase": "working", "model_waits": task.get("model_waits", {}),
                    }] if task else [],
                }))
            page.route("**/api/state", state_response)
            with page.expect_response(lambda response: urlparse(response.url).path == "/api/chat/history") as initial_history:
                page.goto(ui["url"] + "/")
            assert initial_history.value.status == 200
            # State can paint the wait first. Finish the real history paint
            # while metadata is still held, before deliberately making it stale.
            page.wait_for_selector("#chat-messages > .chat-load-older", state="attached")
            page.wait_for_selector(f'[data-wait-id="{first["wait_id"]}"]')
            yield SimpleNamespace(ui=ui, page=page, root=root, controller=controller, gateway=gateway,
                client=client, wait_id=first["wait_id"], first=first, source_release=source_release,
                catalog_entered=catalog_entered, catalog_release=catalog_release,
                pump=pump, decisions=decisions, responses=responses, future=future,
                api_requests=api_requests, completed=completed, messages=messages,
                original_messages=original_messages, original_settings=original_settings,
                canonical=lambda: load_task_result(root, TASK), settings_path=config.SETTINGS_PATH,
                finish=finish)
    finally:
        controller.close()
        source_release.set()
        catalog_release.set()
        pool.shutdown(wait=True, cancel_futures=True)
        sdk.close()
        http_client.close()


@pytest.mark.parametrize("persist_role", [False, True])
def test_browser_wait_controls_apply_once_and_only_to_light(integrated_wait, persist_role):
    flow = integrated_wait
    page = flow.page
    row = page.locator(f'[data-wait-id="{flow.wait_id}"]')
    card = page.locator(f'.chat-live-card[data-task-id="{TASK}"]')
    assert page.locator(".model-wait-row").count() == 1
    assert "Light" in row.inner_text() and "Account: work" in row.inner_text()
    assert row.locator("[data-wait-auto]").is_checked()
    # A quota wait that resumes by itself shows no computation: still dots on a neutral chip.
    assert card.locator("[data-live-phase] > .status-mark").get_attribute("data-shape") == "still"
    assert card.locator("[data-live-phase]").get_attribute("data-chip-tone") == "neutral"
    assert set(flow.canonical()["model_waits"]) == {flow.wait_id}

    # Let real source metadata publish a newer revision without delivering
    # that WS event yet. The stale browser action must traverse the real 409.
    flow.source_release.set()
    assert flow.catalog_entered.wait(5)
    latest = flow.canonical()["model_waits"][flow.wait_id]
    assert latest["revision"] > flow.first["revision"]
    row.locator("[data-wait-auto]").click()  # A rejected toggle returns to checked.
    page.wait_for_function("() => document.querySelector('[data-wait-notice]').textContent.includes('stale_model_wait')")
    assert flow.responses[-1]["status"] == 409
    assert flow.responses[-1]["body"]["reason_code"] == "stale_model_wait"
    assert "pending_action" not in flow.canonical()["model_waits"][flow.wait_id]
    assert not (flow.root / "memory/owner_mailbox" / f"{TASK}.jsonl").exists()
    assert row.locator("[data-wait-auto]").is_checked()

    row.locator("[data-wait-auto]").uncheck()
    page.wait_for_function("() => document.querySelector('[data-wait-notice]').textContent.includes('Request accepted')")
    toggle = flow.decisions[-1]
    assert toggle["revision"] == latest["revision"] and toggle["action"] == "auto_continue"
    canonical = flow.canonical()["model_waits"][flow.wait_id]
    assert canonical["auto_continue"] is True
    assert canonical["pending_action"]["request_id"] == toggle["request_id"]
    assert flow.responses[-1]["status"] == 202 and flow.responses[-1]["body"]["applied"] is False
    browser_fixture.capture(page, f"integrated-wait-accepted-{persist_role}")
    flow.catalog_release.set()
    applied = flow.pump(lambda event: event.get("applied_request_id") == toggle["request_id"])
    assert applied["auto_continue"] is False and applied["revision"] > toggle["revision"]
    page.wait_for_selector("[data-wait-change]:not([disabled])")
    assert not row.locator("[data-wait-auto]").is_checked()

    row.locator("[data-wait-change]").click()
    row.locator("[data-model-role-source]").select_option("api:openai")
    row.locator("[data-model-role-model]").fill("owner-model")
    assert not row.locator("[data-wait-persist]").is_checked()
    if persist_role:
        row.locator("[data-wait-persist]").check()
    # Hold only the real worker's drain lock to expose accepted != applied.
    # The real endpoint still validates, locks, persists and writes the mailbox.
    with flow.controller.lock:
        row.locator("[data-wait-apply]").click()
        page.wait_for_function("() => document.querySelector('[data-wait-notice]').textContent.includes('Request accepted')")
        command = flow.decisions[-1]
        assert command["persist_role"] is persist_role
        assert command["model"] == "openai::owner-model" and command["credential_profile_id"] == ""
        assert flow.responses[-1]["status"] == 202 and flow.responses[-1]["body"]["applied"] is False
        assert flow.responses[-1]["body"]["saved"] is persist_role
        assert flow.canonical()["model_waits"][flow.wait_id]["state"] == "waiting"
        assert page.locator(".model-wait-row").count() == 1
        if persist_role:
            saved = json.loads(flow.settings_path.read_text())
            assert saved["OUROBOROS_MODEL"] == MODEL
            assert saved["OUROBOROS_MODEL_LIGHT"] == "openai::owner-model"
            accounts = saved["OUROBOROS_MODEL_ACCOUNTS"]
            if isinstance(accounts, str):
                accounts = json.loads(accounts)
            assert accounts["main"] == "personal" and accounts["light"] == ""
        else:
            assert flow.settings_path.read_bytes() == flow.original_settings
        browser_fixture.capture(page, f"integrated-wait-switch-{persist_role}")

    resolved = flow.pump(lambda event: event["state"] == "resolved")
    assert resolved["resolution"] == "model_switched"
    assert resolved["applied_request_id"] == command["request_id"]
    page.wait_for_selector(f'[data-wait-id="{flow.wait_id}"]', state="detached")
    answer, usage = flow.future.result(timeout=5)
    assert answer["content"] == "Continued exactly once."
    assert set(flow.controller.overrides) == {"light"}
    assert usage["model_role_route"]["role"] == "light"
    assert len(flow.api_requests) == 1 and flow.api_requests[0]["model"] == "owner-model"
    assert [entry[0]["account"] for entry in flow.gateway.uploads] == [
        {"mode": "pin", "profileId": "personal"}, {"mode": "pin", "profileId": "personal"},
        {"mode": "pin", "profileId": "work"}]
    assert flow.completed == ["a", "b", "review"]
    assert flow.messages == flow.original_messages
    sent_tools = [message for message in flow.api_requests[0]["messages"] if message["role"] == "tool"]
    assert sent_tools == [message for message in flow.messages if message["role"] == "tool"]
    assert any(message.get("content") == "Completed independent review."
               for message in flow.api_requests[0]["messages"])
    ledger = llm_fixture.ledger(flow.root)
    assert [entry["state"] for entry in ledger].count("settled") == 3
    assert [entry["state"] for entry in ledger].count("released") == 1
    assert len(usage["ledger_attempt_ids"]) == 2

    # A retry of the already-applied action reaches the real idempotency record;
    # neither a second generation nor another settings rewrite is allowed.
    stamp = flow.settings_path.stat().st_mtime_ns
    replay = flow.client.post("/api/decisions", json=command)
    assert replay.status_code == 200 and replay.json()["duplicate"] and replay.json()["applied"]
    assert flow.settings_path.stat().st_mtime_ns == stamp and len(flow.api_requests) == 1
    flow.finish(answer)
    actual = flow.client.get(f"/api/tasks/{TASK}").json()
    assert actual["status"] == "completed"
    assert actual["model_waits"][flow.wait_id]["applied_request_id"] == command["request_id"]
    history = flow.client.get("/api/chat/history").json()
    waits = [message for message in history["messages"] if message.get("system_type") == "task_model_wait"]
    assert waits and all(message["task_terminal_status"] == "completed" for message in waits)
    flow.ui["errors"].watch(page)  # the reload may cut off the page's own reads (tests/ui_console_errors.py)
    with page.expect_response(lambda response: urlparse(response.url).path == "/api/chat/history") as loaded:
        page.reload()
    hydrated = loaded.value.json()
    assert any(message.get("task_terminal_status") == "completed" for message in hydrated["messages"])
    # This fixture never emitted a task_summary, so terminal wait references
    # may correctly disappear altogether instead of inventing a live card.
    page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
    assert page.locator(".model-wait-row").count() == 0
    browser_fixture.capture(page, f"integrated-wait-resolved-{persist_role}")
