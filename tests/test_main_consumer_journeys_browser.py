"""Focused #1660 Main consumer regressions against an attested real candidate.

Only provider judgments, producer inputs and deliberate client disconnection are
fixtures. Every browser consumes the actual gateway and SPA. Diagnostics use the
managed evidence channel; OUROBOROS_BROWSER_EVIDENCE_OUT overrides the destination.
"""

from tests import _dialogue as dialogue_test


import json
import uuid

import pytest

from tests._shared import producer_evidence_dir
from devtools.benchmarks.common.server_runner import _api
from tests.test_native_owner_wait_browser import chat_connection
from tests.test_owner_wait_integration import wait_clone as clone_fixture
from tests.system_e2e.harness import (
    ArtifactOracle, KeylessIsolatedServer, ModelGate, ScriptedStubModel, body_text,
    keyless_settings, wait_durable_result, wait_until, write_settings_file,
)

wait_clone = clone_fixture
pytestmark = [pytest.mark.serial, pytest.mark.browser]


def _controls_evidence(tmp_path, name):
    destination = producer_evidence_dir(tmp_path / "screenshots", "main-consumer-journeys",
                                        env="OUROBOROS_BROWSER_EVIDENCE_OUT") / name
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _controls_capture(page, destination, name):
    if page and not page.is_closed():
        page.screenshot(path=str(destination / (name + ".png")), animations="disabled", full_page=True)


def _controls_owner_action(page, task_id):
    """The one System message in Main's feed through which a task that needs the owner is acted on."""
    return page.locator(f'#chat-messages .chat-bubble.owner-action[data-owner-action="{task_id}"]')


def _controls_project_closed(page):
    assert not page.locator('#project-panel').is_visible(), "The action must be usable from Main with Project closed"


async def _controls_capture_async(page, destination, name):
    if page and not page.is_closed():
        await page.screenshot(path=str(destination / (name + ".png")), animations="disabled", full_page=True)


async def _controls_project_closed_async(page):
    assert not await page.locator('#project-panel').is_visible(), "The action must be usable from Main with Project closed"


@pytest.mark.parametrize("interrupt_second_client", [False, True], ids=["live-second-client", "reconnect-second-client"])
def test_main_close_wakes_same_author_without_approval_and_reconnects_second_client(wait_clone, tmp_path, interrupt_second_client):
    """Both browsers consume live production state; network loss injects no frames.

    The async Playwright client lets the routed socket handler refuse a reconnect
    with ``await route.close()``; the sync client cannot await inside that handler.
    Blocking producer waits run off the event loop so live frames keep flowing.
    """
    import asyncio
    from playwright.async_api import async_playwright, expect
    from ouroboros.owner_quiz import OWNER_CLOSE_REASON

    destination = _controls_evidence(tmp_path, "main-close-reconnect" if interrupt_second_client else "main-close-two-clients")
    data_root = tmp_path / "instance" / "data"
    data_root.mkdir(parents=True)
    # One tap closes without the owner's words: the closure records the standard sentence.
    reason = OWNER_CLOSE_REASON
    resumed_inputs = []

    def after_close(body):
        text = body_text(body)
        resumed_inputs.append(text)
        assert reason in text, "The waiting author must receive the recorded closing reason"
        assert any(phrase in text.lower() for phrase in ("not owner approval", "not approval", "without approval", "no approval")), "Closure must deny approval to the resumed author"
        return {"final": "The question was closed without approval; the draft remains unsubmitted."}

    steps = [
        {"tool": "escalate", "arguments": {
            "question": "May I submit this draft?", "wait_for_answer": True,
            "options": [{"label": "Submit"}, {"label": "Keep editing"}],
            "stake": "Submission requires an explicit answer; closing is not permission."}},
        after_close,
    ]
    receipt = {"journey": "Main close with a waiting live author and disconnected second client"}

    async def journey(server, oracle, project):
        page = second = None
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            first_context = await browser.new_context(viewport={"width": 1100, "height": 900}, reduced_motion="reduce")
            second_context = await browser.new_context(viewport={"width": 1100, "height": 900}, reduced_motion="reduce")
            page, second = await first_context.new_page(), await second_context.new_page()
            sockets = []
            network = {"offline": False}

            async def routed_socket(route):
                if network["offline"]:
                    await route.close(code=1000, reason="Controlled client network interruption")
                else:
                    # Retain both ends so the deliberate outage closes the
                    # actual server connection too. Native WebSocket.close
                    # permits 1000 (not reserved close code 1001).
                    sockets.append((route, route.connect_to_server()))

            await second.route_web_socket("**/ws", routed_socket)
            try:
                await page.goto(server.base_url, wait_until="domcontentloaded")
                await second.goto(server.base_url, wait_until="domcontentloaded")
                with chat_connection(server) as ws:
                    message_id = uuid.uuid4().hex
                    ws.send(json.dumps({"type": "chat", "content": "Ask my permission before submitting the draft, and wait.",
                        "client_message_id": message_id, "chat_id": project["chat_id"], "project_id": project["id"]}))
                    task = await asyncio.to_thread(wait_until, lambda: next((row["task"] for row in oracle.events("task_received")
                        if (row.get("task", {}).get("metadata", {}).get("origin_message_ref") or {}).get("client_message_id") == message_id), None), 90)
                    assert task
                    task_id = task["id"]
                    wait = await asyncio.to_thread(wait_until, lambda: (row if (row := oracle.task_result(task_id).get("owner_wait", {})).get("state") == "waiting" else None), 90)
                    assert wait, oracle.task_result(task_id)
                    quiz_id = wait["quiz_id"]
                    receipt.update(task_id=task_id, quiz_id=quiz_id, initial_wait=wait)
                    selector = f'#chat-messages .project-question-card[data-task-id="{task_id}"][data-quiz-id="{quiz_id}"]'
                    for client in (page, second):
                        await client.locator(selector).get_by_text("Waiting for your answer", exact=True).wait_for(timeout=30000)
                        await _controls_project_closed_async(client)
                    stale = second.locator(selector)
                    await stale.locator('.chat-quiz-comment').fill("This unsent draft must not become an answer")
                    # A typed answer holds the card's quiet Close: a mis-tap never discards the owner's words.
                    await expect(stale.locator('.chat-quiz-close')).to_be_disabled()
                    await _controls_capture_async(second, destination, "second-client-before-disconnect")
                    if interrupt_second_client:
                        network["offline"] = True
                        assert sockets, "Second client never established its real WebSocket"
                        for socket, server_socket in list(sockets):
                            server_socket.close(code=1000, reason="Controlled client network interruption")
                            await socket.close(code=1000, reason="Controlled client network interruption")
                        await second_context.set_offline(True)
                        await expect(stale.locator('.chat-quiz-option').first).to_be_disabled(timeout=15000)
                        await expect(stale.locator('.chat-quiz-comment')).to_be_disabled()
                        await _controls_capture_async(second, destination, "second-client-disconnected-controls-disabled")

                    card = page.locator(selector)
                    close = card.get_by_role("button", name="Close question without answering", exact=True)
                    await expect(close).to_be_enabled()
                    await card.locator('.chat-quiz-comment').fill("Typed, then taken back")
                    await expect(close).to_be_disabled()
                    await card.locator('.chat-quiz-comment').fill("")
                    await expect(close).to_be_enabled()
                    await _controls_capture_async(page, destination, "main-before-close")
                    async with page.expect_response(lambda response: response.url.endswith('/api/decisions')
                                                    and response.request.method == 'POST') as close_response:
                        await close.click()
                    response = await close_response.value
                    confirmation = await response.json()
                    receipt["close_response"] = {"status": response.status, "body": confirmation,
                                                 "request": response.request.post_data_json}
                    assert response.ok and confirmation.get("ok"), confirmation
                    sent = response.request.post_data_json
                    assert sent == {"request_id": sent["request_id"], "decision_id": f"quiz:{task_id}:{quiz_id}",
                                    "action": "close", "reason": ""}, sent
                    # The copy reads the closure for five seconds, then only the Main copy goes.
                    await expect(card.locator('.chat-quiz-status-text')).to_have_text("Question closed", timeout=30000)
                    shown = await page.evaluate("performance.now()")
                    await expect(card.locator('.chat-quiz-closed-reason')).to_have_text(reason)
                    await expect(card.locator('.chat-quiz-close, .chat-quiz-comment')).to_have_count(0)
                    await _controls_capture_async(page, destination, "main-closed-result")
                    await expect(card).to_have_count(0, timeout=15000)
                    receipt["closed_copy_ms"] = await page.evaluate("performance.now()") - shown
                    assert 4000 <= receipt["closed_copy_ms"] <= 8000, receipt["closed_copy_ms"]
                    final = await asyncio.to_thread(wait_durable_result, oracle, task_id, timeout=90)
                    receipt["final"] = final
                    block = final["owner_quiz"][quiz_id]
                    assert block["state"] == "closed" and block["closed_actor"] == "owner", block
                    assert block["closed_reason"] == reason
                    assert block.get("answered_index") is None and not block.get("answered_at"), block
                    assert block["publication_chat_id"] == project["chat_id"]
                    assert final["owner_wait"]["wait_id"] == wait["wait_id"]
                    assert final["owner_wait"]["task_attempt"] == wait["task_attempt"]
                    assert final["owner_wait"]["state"] == "resumed"
                    assert len(resumed_inputs) == 1
                    assert len([row for row in oracle.events("task_received") if row.get("task", {}).get("id") == task_id]) == 1
                    assert {row.get("task", {}).get("id") for row in oracle.events("task_received")} == {task_id}, "Closing the live question must not create a terminal follow-up turn"
                    await _controls_project_closed_async(page)
                    await _controls_capture_async(page, destination, "main-closed")

                    if interrupt_second_client:
                        network["offline"] = False
                        await second_context.set_offline(False)
                    await expect(stale).to_have_count(0, timeout=30000)
                    await _controls_project_closed_async(second)
                    await _controls_capture_async(second, destination, "second-client-reconnected-no-stale-answer")
                    await second.reload(wait_until="domcontentloaded")
                    await second.wait_for_function("() => document.querySelector('#chat-messages') !== null")
                    await expect(second.locator(selector)).to_have_count(0)
                    state = await asyncio.to_thread(_api, server.base_url, "GET", "/api/state")
                    assert state["current_questions_complete"]
                    assert not any(row.get("quiz_id") == quiz_id for row in state["current_questions"])
                    receipt["state_after_reconnect"] = state
                    receipt["resumed_author_input"] = resumed_inputs
            finally:
                network["offline"] = False
                await second_context.set_offline(False)
                await _controls_capture_async(page, destination, "first-client-final")
                await _controls_capture_async(second, destination, "second-client-final")
                receipt["state_at_exit"] = await asyncio.to_thread(_api, server.base_url, "GET", "/api/state")
                if receipt.get("task_id"):
                    receipt["task_at_exit"] = oracle.task_result(receipt["task_id"])
                (destination / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
                await browser.close()

    with ScriptedStubModel(steps) as stub:
        settings_path = data_root / "settings.json"
        write_settings_file(settings_path, keyless_settings(stub, OUROBOROS_MAX_WORKERS=1))
        server = KeylessIsolatedServer(wait_clone, data_root, settings_path)
        server.start(ready_timeout=120)
        oracle = ArtifactOracle(data_root)
        try:
            project = _api(server.base_url, "POST", "/api/projects", {"name": "Main close proof"})["project"]
            receipt.update(project=project, server_url=server.base_url, server_pid=server.proc.pid)
            asyncio.run(journey(server, oracle, project))
        finally:
            server.stop()


class _ControlsPricedStub(ScriptedStubModel):
    """Report final provider cost through the existing loopback transport."""
    def _completion(self, body):
        completion = super()._completion(body)
        completion["usage"]["cost"] = 0.25
        return completion


def test_main_resume_uses_same_paused_project_task_without_opening_project(wait_clone, tmp_path):
    """The real ledger exhausts; Main exposes and executes the ordinary Resume."""
    from playwright.sync_api import sync_playwright, expect

    destination = _controls_evidence(tmp_path, "main-resume-project-closed")
    data_root = tmp_path / "instance" / "data"
    data_root.mkdir(parents=True)
    receipt = {"journey": "Main Resume of a real budget-paused Project task"}
    script = [{"tool": "list_files", "arguments": {"root": "system_repo", "path": "."}} for _ in range(10)]
    with _ControlsPricedStub(script, final_answer="The same Project task completed after your explicit Resume.") as stub:
        settings_path = data_root / "settings.json"
        write_settings_file(settings_path, keyless_settings(stub, OUROBOROS_MAX_WORKERS=1,
                            TOTAL_BUDGET=1.0, OUROBOROS_PER_TASK_COST_USD=10.0))
        server = KeylessIsolatedServer(wait_clone, data_root, settings_path)
        server.start(ready_timeout=120)
        oracle = ArtifactOracle(data_root)
        page = None
        try:
            project = _api(server.base_url, "POST", "/api/projects", {"name": "Resume from Main"})["project"]
            created = _api(server.base_url, "POST", "/api/tasks", {
                "description": "Keep listing the repository root; wait for my explicit Resume if the budget pauses you.",
                "project_id": project["id"], "chat_id": project["chat_id"], "memory_mode": "forked",
                "actor_id": "browser-regression", "source": "browser-regression", "timeout_sec": 1800,
                "metadata": {"source": "browser-regression", "delegation_role": "root",
                             "budget_profile": {"cost_hard_stop_pct": 0}},
            }, timeout=60)
            task_id = created["task_id"]
            receipt.update(project=project, task_id=task_id, created=created,
                           server_url=server.base_url, server_pid=server.proc.pid)
            paused = wait_until(lambda: (row if (row := oracle.task_result(task_id).get("budget_pause", {})).get("state") == "paused" else None), 180)
            assert paused, oracle.task_result(task_id)
            receipt["paused"] = paused
            calls_at_pause = len(stub.kinds())
            assert paused["exact_continuation"] and paused["reason"] != "owner", paused
            assert task_id not in oracle.running_ids()
            with sync_playwright() as pw:
                browser = pw.chromium.launch()
                page = browser.new_page(viewport={"width": 1100, "height": 900}, reduced_motion="reduce")
                try:
                    page.goto(server.base_url, wait_until="domcontentloaded")
                    card = _controls_owner_action(page, task_id)
                    card.get_by_role("button", name="Resume", exact=True).wait_for(timeout=30000)
                    expect(card.locator('.sender')).to_have_text("System")
                    expect(card.locator('.owner-action-reason')).to_have_text(
                        "Paused at the spending limit. Change the limit if needed, then Resume.")
                    _controls_project_closed(page)
                    _controls_capture(page, destination, "main-budget-paused")
                    # A click cannot silently raise money. With the wallet still
                    # exhausted, the same real ingress must refuse visibly.
                    with page.expect_response(lambda response: response.url.endswith(f'/api/tasks/{task_id}/resume')
                                              and response.request.method == 'POST') as refused:
                        card.get_by_role("button", name="Resume", exact=True).click()
                    refused_body = refused.value.json()
                    receipt["refused_resume"] = {"status": refused.value.status, "body": refused_body}
                    assert refused.value.status == 409 and refused_body["error"] == "budget_still_exhausted", refused_body
                    page.get_by_text("Resume refused: budget_still_exhausted", exact=True).wait_for()
                    assert oracle.task_result(task_id)["budget_pause"]["state"] == "paused"
                    assert len(stub.kinds()) == calls_at_pause
                    _controls_capture(page, destination, "main-resume-refused-budget-unchanged")

                    # This isolated test owner's existing settings ingress makes
                    # headroom available. It does not grant Resume by itself.
                    receipt["isolated_budget_update"] = _api(server.base_url, "POST", "/api/settings", {"TOTAL_BUDGET": 50.0}, timeout=120)
                    expect(card.get_by_role("button", name="Resume", exact=True)).to_be_enabled()
                    assert oracle.task_result(task_id)["budget_pause"]["state"] == "paused"
                    assert len(stub.kinds()) == calls_at_pause
                    with page.expect_response(lambda response: response.url.endswith(f'/api/tasks/{task_id}/resume')
                                              and response.request.method == 'POST') as resumed:
                        card.get_by_role("button", name="Resume", exact=True).click()
                    resumed_body = resumed.value.json()
                    receipt["accepted_resume"] = {"status": resumed.value.status, "body": resumed_body}
                    assert resumed.value.ok and resumed_body["ok"] and resumed_body["exact_continuation"], resumed_body
                    assert resumed_body["task_id"] == task_id
                    final = wait_durable_result(oracle, task_id, timeout=180)
                    receipt["final"] = final
                    assert final["status"] == "completed", final
                    continuation = final["budget_pause"]
                    assert continuation["pause_id"] == paused["pause_id"]
                    assert continuation["state"] == "resumed"
                    assert continuation["grant"]["grant_id"] == resumed_body["grant_id"]
                    assert continuation["grant"]["consumed_at"]
                    expect(card).to_have_count(0, timeout=30000)
                    _controls_project_closed(page)
                    _controls_capture(page, destination, "main-resume-complete")
                    page.reload(wait_until="domcontentloaded")
                    expect(_controls_owner_action(page, task_id)).to_have_count(0)
                    receipt["state_after_resume"] = _api(server.base_url, "GET", "/api/state")
                    assert not any(row.get("task_id") == task_id for row in receipt["state_after_resume"]["main_attention"])
                finally:
                    _controls_capture(page, destination, "final")
                    receipt["model_calls"] = stub.kinds()
                    receipt["queue_snapshot"] = oracle.queue_snapshot()
                    receipt["state_at_exit"] = _api(server.base_url, "GET", "/api/state")
                    if receipt.get("task_id"):
                        receipt["task_at_exit"] = oracle.task_result(receipt["task_id"])
                    (destination / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
                    browser.close()
        finally:
            server.stop()


class _ControlsAuthRefusalStub(ScriptedStubModel):
    """The real HTTP provider refuses after the agent declares primary wait.

    Only this provider's response is controlled. Successful responses, catalogs,
    SSE, the agent tool, wait producer, gateway and browser use existing code.
    """
    def __init__(self, marker, resumed_inputs):
        import io
        import threading

        def finish(body):
            resumed_inputs.append(body)
            return {"final": "Model access recovered after your Try again; the same Project task is complete."}

        super().__init__([{"tool": "switch_model", "arguments": {"primary": "wait"}}, finish])
        self.allow_retry = threading.Event()
        self.refused_requests = []
        self.marker = marker
        outer = self
        base_handler = self._server.RequestHandlerClass

        class AuthRefusalHandler(base_handler):
            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                body = json.loads(raw or b"{}")
                with outer._lock:
                    refuses = (bool(body.get("tools")) and outer.marker in body_text(body)
                               and outer._script_index == 1 and not outer.allow_retry.is_set())
                    if refuses:
                        outer.refused_requests.append(body)
                if refuses:
                    return self._send({"error": {"code": "invalid_api_key", "type": "authentication_error",
                        "message": "Controlled loopback credential refusal; the owner must retry."}}, status=401)
                original = self.rfile
                self.rfile = io.BytesIO(raw)
                try:
                    return super().do_POST()
                finally:
                    self.rfile = original

        self._server.RequestHandlerClass = AuthRefusalHandler


def test_main_modelwait_try_again_resumes_same_project_author(wait_clone, tmp_path):
    """Main controls a wait produced by an actual provider HTTP refusal."""
    from playwright.sync_api import sync_playwright, expect

    destination = _controls_evidence(tmp_path, "main-modelwait-project-closed")
    data_root = tmp_path / "instance" / "data"
    data_root.mkdir(parents=True)
    marker = "MAIN1660_AUTH_WAIT"
    resumed_inputs = []
    receipt = {"journey": "Main model wait Try again with Project closed", "marker": marker}
    with _ControlsAuthRefusalStub(marker, resumed_inputs) as stub:
        settings_path = data_root / "settings.json"
        write_settings_file(settings_path, keyless_settings(stub, OUROBOROS_MAX_WORKERS=1))
        server = KeylessIsolatedServer(wait_clone, data_root, settings_path)
        server.start(ready_timeout=120)
        oracle = ArtifactOracle(data_root)
        page = None
        try:
            project = _api(server.base_url, "POST", "/api/projects", {"name": "Model access from Main"})["project"]
            receipt.update(project=project, server_url=server.base_url, server_pid=server.proc.pid)
            with sync_playwright() as pw:
                browser = pw.chromium.launch()
                page = browser.new_page(viewport={"width": 1100, "height": 900}, reduced_motion="reduce")
                try:
                    page.goto(server.base_url, wait_until="domcontentloaded")
                    with chat_connection(server) as ws:
                        message_id = uuid.uuid4().hex
                        ws.send(json.dumps({"type": "chat", "content": f"[{marker}] Keep this task on its primary model and wait for my help if access fails.",
                            "client_message_id": message_id, "chat_id": project["chat_id"], "project_id": project["id"]}))
                        task = wait_until(lambda: next((row["task"] for row in oracle.events("task_received")
                            if (row.get("task", {}).get("metadata", {}).get("origin_message_ref") or {}).get("client_message_id") == message_id), None), 90)
                        assert task and task.get("_is_direct_chat"), task
                        task_id = task["id"]
                        receipt["task_id"] = task_id
                        waiting = wait_until(lambda: next((row for row in oracle.task_result(task_id).get("model_waits", {}).values()
                                                          if row.get("state") == "waiting"), None), 90)
                        assert waiting, oracle.task_result(task_id)
                        receipt["initial_wait"] = waiting
                        wait_id = waiting["wait_id"]
                        assert waiting["reason"] == "auth" and waiting["requires_owner_action"] is True
                        assert waiting["auto_continue"] is False
                        assert len(stub.refused_requests) == 1
                        assert not resumed_inputs
                        calls_at_wait = len(stub.kinds())
                        card = _controls_owner_action(page, task_id)
                        control = card.locator(f'.model-wait-row[data-wait-id="{wait_id}"]')
                        retry = control.get_by_role("button", name="Try again", exact=True)
                        expect(retry).to_be_enabled(timeout=30000)
                        expect(card.locator('.owner-action-reason')).to_have_text("Needs you to restore model access.")
                        _controls_project_closed(page)
                        _controls_capture(page, destination, "main-waiting-for-model-access")
                        page.wait_for_timeout(1500)
                        assert len(stub.kinds()) == calls_at_wait and len(stub.refused_requests) == 1
                        assert not resumed_inputs, "Waiting for the owner must not generate additional model turns"
                        receipt["state_while_waiting"] = _api(server.base_url, "GET", "/api/state")
                        projected = next(row for row in receipt["state_while_waiting"]["main_attention"] if row["task_id"] == task_id)
                        assert projected["model_waits"][wait_id]["task_attempt"] == waiting["task_attempt"]

                        # Provider access recovers externally; this API-only
                        # route still needs the actual owner control to resend.
                        stub.allow_retry.set()
                        page.wait_for_timeout(1500)
                        assert len(stub.kinds()) == calls_at_wait and not resumed_inputs
                        with page.expect_response(lambda response: response.url.endswith('/api/decisions')
                                                  and response.request.method == 'POST') as retried:
                            retry.click()
                        response = retried.value
                        request_body, confirmation = response.request.post_data_json, response.json()
                        receipt["retry"] = {"request": request_body, "status": response.status, "response": confirmation}
                        assert response.ok and confirmation["ok"], confirmation
                        assert request_body["decision_id"] == f"model_wait:{task_id}:{wait_id}"
                        assert request_body["action"] == "retry" and request_body["revision"] == waiting["revision"]
                        assert confirmation["wait"]["wait_id"] == wait_id
                        assert confirmation["wait"]["task_attempt"] == waiting["task_attempt"]
                        final = wait_durable_result(oracle, task_id, timeout=90)
                        receipt["final"] = final
                        assert final["status"] == "completed", final
                        resolved = final["model_waits"][wait_id]
                        assert resolved["state"] == "resolved" and resolved["resolution"] == "retry_requested"
                        assert resolved["task_attempt"] == waiting["task_attempt"]
                        assert resolved["applied_request_id"] == request_body["request_id"]
                        assert len(resumed_inputs) == 1 and marker in body_text(resumed_inputs[0])
                        assert len(stub.refused_requests) == 1
                        assert {row.get("task", {}).get("id") for row in oracle.events("task_received")} == {task_id}
                        expect(card).to_have_count(0, timeout=30000)
                        _controls_project_closed(page)
                        _controls_capture(page, destination, "main-after-model-retry")
                        page.reload(wait_until="domcontentloaded")
                        expect(_controls_owner_action(page, task_id)).to_have_count(0)
                finally:
                    _controls_capture(page, destination, "final")
                    receipt["refused_provider_requests"] = stub.refused_requests
                    receipt["resumed_author_requests"] = resumed_inputs
                    receipt["model_calls"] = stub.kinds()
                    receipt["state_at_exit"] = _api(server.base_url, "GET", "/api/state")
                    if receipt.get("task_id"):
                        receipt["task_at_exit"] = oracle.task_result(receipt["task_id"])
                    (destination / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
                    browser.close()
        finally:
            server.stop()


def test_main_closed_question_stays_closed_after_canonical_gc_across_pages_and_clients(wait_clone, tmp_path):
    """Cold clients must carry the durable closure from a newer page to its old ask."""
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError, sync_playwright, expect
    from ouroboros.gateway.history import _DEFAULT_N_HUMAN
    from ouroboros.owner_quiz import (
        _QUIZ_CAP, OWNER_CLOSE_REASON, mark_discharged, mark_published, mark_quiz_history_written,
        question_delivery_settled, quiz_states, reconcile_terminal,
        record_asked, record_closed, record_publication,
    )
    from ouroboros.task_results import write_task_result
    from supervisor.message_bus import log_chat
    from urllib.parse import urlencode

    destination = _controls_evidence(tmp_path, "main-closed-canonical-gc-two-clients")
    data_root = tmp_path / "instance" / "data"
    data_root.mkdir(parents=True)
    task_id, quiz_id = "main-closed-history-author", "old-edition-question"
    question = "Which edition should the next report use?"
    reason = OWNER_CLOSE_REASON  # one tap, no words: the standard sentence
    write_task_result(data_root, task_id, "completed", chat_id=1,
                      result="The current report is complete while the future-edition question remains open.")
    record_asked(data_root, task_id, quiz_id=quiz_id, question=question,
                 options=["Printed", "Searchable"], assumption="Keep the current searchable edition", chat_id=1)
    record_publication(data_root, task_id, quiz_id, 1)
    block = quiz_states(data_root, task_id)[quiz_id]
    log_chat("out", 1, 1, question, task_id=task_id, record_type="quiz", quiz=block,
             drive_root=data_root, require_write=True)
    mark_published(data_root, task_id, quiz_id)
    reconcile_terminal(data_root, task_id)
    receipt = {"task_id": task_id, "quiz_id": quiz_id, "question": question, "reason": reason,
               "journey": "Owner close in Main, canonical bounded GC, cold cross-page replay in two clients"}
    with ScriptedStubModel([]) as stub:
        settings_path = data_root / "settings.json"
        write_settings_file(settings_path, keyless_settings(stub, OUROBOROS_MAX_WORKERS=1))
        server = KeylessIsolatedServer(wait_clone, data_root, settings_path)
        server.start(ready_timeout=120)
        pages, history_reads = [], [[], []]
        try:
            receipt.update(server_url=server.base_url, server_pid=server.proc.pid)
            with sync_playwright() as pw:
                browser = pw.chromium.launch()
                try:
                    for index in range(2):
                        context = browser.new_context(viewport={"width": 1100, "height": 900}, reduced_motion="reduce")
                        page = context.new_page()
                        pages.append(page)

                        def retain_history(response, client=index):
                            if '/api/chat/history' in response.url and response.ok:
                                history_reads[client].append({"url": response.url, "body": response.json()})

                        page.on("response", retain_history)
                        page.goto(server.base_url, wait_until="domcontentloaded")
                    selector = f'#chat-messages .chat-quiz-card[data-task-id="{task_id}"][data-quiz-id="{quiz_id}"]'
                    for index, page in enumerate(pages):
                        card = page.locator(selector)
                        card.get_by_text(question, exact=True).wait_for(timeout=30000)
                        expect(card.locator('.chat-quiz-option').first).to_be_enabled()
                        _controls_project_closed(page)
                        _controls_capture(page, destination, f"client-{index + 1}-original-optional")
                    # These are real retained human history rows, not browser
                    # inserts. They physically separate the ask and its closure.
                    for index in range(_DEFAULT_N_HUMAN + 20):
                        log_chat("in", 1, 1, f"Later retained Main note {index:03d}",
                                 client_message_id=f"gc-history-gap-{index}", drive_root=data_root, require_write=True)
                    first = pages[0].locator(selector)
                    with pages[0].expect_response(lambda response: response.url.endswith('/api/decisions')
                                                 and response.request.method == 'POST') as closed:
                        first.get_by_role("button", name="Close question without answering", exact=True).click()
                    receipt["close_response"] = {"status": closed.value.status, "body": closed.value.json()}
                    assert closed.value.ok and receipt["close_response"]["body"]["state"] == "closed"
                    assert wait_until(lambda: question_delivery_settled(quiz_states(data_root, task_id).get(quiz_id)), 30)
                    for index, page in enumerate(pages):
                        expect(page.locator(selector)).to_have_attribute("data-state", "closed", timeout=30000)
                        page.locator(selector).get_by_text(reason, exact=True).wait_for()
                        assert page.locator(selector).locator('.chat-quiz-option:enabled, .chat-quiz-comment').count() == 0
                        _controls_capture(page, destination, f"client-{index + 1}-live-closed")
                    receipt["canonical_closed_before_gc"] = quiz_states(data_root, task_id)[quiz_id]

                    # Exercise the real hot-set GC in owner_quiz's locked
                    # mutation, by publishing and settling later questions.
                    # Never delete or rewrite the target block or its history.
                    for index in range(_QUIZ_CAP):
                        newer_id = f"settled-later-{index}"
                        newer_question = f"Retained later decision {index}?"
                        record_asked(data_root, task_id, quiz_id=newer_id, question=newer_question,
                                     options=["Keep"], chat_id=1)
                        record_publication(data_root, task_id, newer_id, 1)
                        newer = quiz_states(data_root, task_id)[newer_id]
                        log_chat("out", 1, 1, newer_question, task_id=task_id, record_type="quiz", quiz=newer,
                                 drive_root=data_root, require_write=True)
                        mark_published(data_root, task_id, newer_id)
                        outcome = record_closed(data_root, task_id, quiz_id=newer_id,
                                                reason="Superseded by the completed edition", actor_task_id=task_id)
                        assert outcome["ok"], outcome
                        log_chat("system", 1, 0, outcome["block"]["closed_reason"], task_id=task_id,
                                 record_type="quiz_closed", quiz=outcome["block"],
                                 drive_root=data_root, require_write=True)
                        mark_quiz_history_written(data_root, task_id, newer_id)
                        mark_discharged(data_root, task_id, newer_id)
                    canonical = quiz_states(data_root, task_id)
                    receipt["canonical_after_gc"] = canonical
                    assert quiz_id not in canonical, "The real bounded canonical GC did not retire the discharged target"
                    assert len(canonical) == _QUIZ_CAP
                    recent = _api(server.base_url, "GET", "/api/chat/history?chat_id=1")
                    receipt["recent_api_after_gc"] = recent
                    assert any(row.get("system_type") == "quiz_closed" and row.get("quiz", {}).get("quiz_id") == quiz_id
                               for row in recent["messages"]), "The newer page must carry the durable target closure"
                    assert not any(row.get("msg_type") == "quiz" and row.get("quiz", {}).get("quiz_id") == quiz_id
                                   for row in recent["messages"]), "The original ask must be in a different history page"
                    cursor = recent["next_cursor"]
                    older_pages, old_ask = [], None
                    for _ in range(5):
                        assert cursor, "No older cursor reaches the retained original ask"
                        older = _api(server.base_url, "GET", "/api/chat/history?" + urlencode({"chat_id": 1, "cursor": cursor}))
                        older_pages.append(older)
                        old_ask = next((row for row in older["messages"] if row.get("msg_type") == "quiz"
                                        and row.get("quiz", {}).get("quiz_id") == quiz_id), None)
                        if old_ask:
                            break
                        cursor = older["next_cursor"]
                    assert old_ask, "The original question was not retained in older history"
                    receipt["older_api_pages"] = older_pages
                    observations = []
                    for index, page in enumerate(pages):
                        history_reads[index].clear()
                        page.reload(wait_until="domcontentloaded")
                        page.get_by_text(f"Retained later decision {_QUIZ_CAP - 1}?", exact=True).wait_for(timeout=30000)
                        assert any(any(row.get("system_type") == "quiz_closed"
                                           and row.get("quiz", {}).get("quiz_id") == quiz_id
                                           for row in read["body"]["messages"])
                                   for read in history_reads[index]), "Cold browser never consumed the closure page"
                        target = page.locator(selector)
                        for _ in range(5):
                            if target.count():
                                break
                            button = page.locator('#chat-messages .chat-load-older-btn')
                            # Exhausted history hides the button; a hidden one is still "enabled".
                            expect(button).to_be_visible(timeout=30000)
                            expect(button).to_be_enabled(timeout=30000)
                            button.click()
                            # The older read renders the ask or exposes the next page; a fixed
                            # pause raced the render and clicked the now-hidden button.
                            try:
                                target.wait_for(state="attached", timeout=10000)
                            except PlaywrightTimeoutError:
                                pass
                        target.wait_for(state="attached", timeout=15000)
                        target.scroll_into_view_if_needed()
                        observations.append({"client": index + 1, "state": target.get_attribute('data-state'),
                            "text": target.inner_text(), "enabled_answers": target.locator('.chat-quiz-option:enabled').count(),
                            "comment_fields": target.locator('.chat-quiz-comment').count(),
                            "close_controls": target.locator('.chat-quiz-close').count()})
                        _controls_project_closed(page)
                        _controls_capture(page, destination, f"client-{index + 1}-cold-older-question")
                    receipt["cold_observations"] = observations
                    assert all(item["state"] == "closed" for item in observations), observations
                    assert all(reason in item["text"] for item in observations), observations
                    assert all(item["enabled_answers"] == item["comment_fields"] == item["close_controls"] == 0
                               for item in observations), observations
                    assert stub.kinds() == [], "Closing a terminal question must not invent a model turn"
                finally:
                    receipt["browser_history_reads"] = history_reads
                    receipt["state_at_exit"] = _api(server.base_url, "GET", "/api/state")
                    receipt["canonical_at_exit"] = quiz_states(data_root, task_id)
                    for index, page in enumerate(pages):
                        _controls_capture(page, destination, f"client-{index + 1}-final")
                    (destination / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
                    browser.close()
        finally:
            server.stop()


class _ChildControlsAuthRefusalStub(ScriptedStubModel):
    """Real delegated actor gets an HTTP 401 after selecting primary wait.

    The root uses schedule_subagent: external /api/tasks deliberately refuses
    fabricated child lineage. Its next provider call is held while the real
    child needs the owner, so the root cannot narrate a replacement notice.
    """
    def __init__(self, parent_marker, child_marker):
        import io
        import threading

        self.child_model = "main1660-wait-child"
        self.parent_marker, self.child_marker = parent_marker, child_marker
        self.author_calls = {"parent": [], "child": []}
        self.allow_retry = threading.Event()
        self.refused_requests = []
        self.parent_hold = ModelGate(lambda body: bool(body.get("tools"))
            and body.get("model") == "mock-model"
            and parent_marker in body_text(body)
            and len(self.author_calls["parent"]) >= 1, timeout=300)

        def respond(body):
            child = body.get("model") == self.child_model
            actor = "child" if child else "parent"
            self.author_calls[actor].append(body)
            turn = len(self.author_calls[actor])
            assert turn <= 2, "the child-wait probe bought an unexpected author turn"
            if child:
                assert child_marker in body_text(body)
                if turn == 1:
                    return {"tool": "switch_model", "arguments": {"primary": "wait"}}
                return {"final": "The delegated verification resumed on its original model after your Try again."}
            assert parent_marker in body_text(body)
            if turn == 1:
                return {"tool": "schedule_subagent", "arguments": {
                    "subagent_id": self.child_model,
                    "objective": child_marker + ": keep your primary model; wait for owner help if access fails.",
                    "expected_output": "A verification result from the same delegated task after model access recovers.",
                }}
            return {"final": "The delegated verification probe has ended."}

        super().__init__([respond] * 4, gate=self.parent_hold,
                         model_ids=["mock-model", self.child_model])
        outer = self
        base_handler = self._server.RequestHandlerClass

        class ChildAuthRefusalHandler(base_handler):
            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                body = json.loads(raw or b"{}")
                with outer._lock:
                    refuses = (bool(body.get("tools")) and body.get("model") == outer.child_model
                               and outer.child_marker in body_text(body)
                               and len(outer.author_calls["child"]) == 1
                               and not outer.allow_retry.is_set())
                    if refuses:
                        outer.refused_requests.append(body)
                if refuses:
                    return self._send({"error": {"code": "invalid_api_key", "type": "authentication_error",
                        "message": "Controlled child-model credential refusal; the owner must retry."}}, status=401)
                original = self.rfile
                self.rfile = io.BytesIO(raw)
                try:
                    return super().do_POST()
                finally:
                    self.rfile = original

        self._server.RequestHandlerClass = ChildAuthRefusalHandler


def test_main_child_modelwait_try_again_uses_same_delegated_author(wait_clone, tmp_path):
    """Every owner-required model wait is actionable from Main, including children."""
    from playwright.sync_api import expect, sync_playwright

    destination = _controls_evidence(tmp_path, "main-child-modelwait-project-closed")
    data_root = tmp_path / "instance" / "data"
    data_root.mkdir(parents=True)
    parent_marker = "MAIN1660_CHILD_WAIT_PARENT_" + uuid.uuid4().hex
    child_marker = "MAIN1660_CHILD_AUTH_WAIT_" + uuid.uuid4().hex
    receipt = {"journey": "Main child model wait Try again with Project closed", "passed": False}
    with _ChildControlsAuthRefusalStub(parent_marker, child_marker) as stub:
        settings = keyless_settings(stub, OUROBOROS_MAX_WORKERS=2, OUROBOROS_SUBAGENTS=json.dumps({
            "enabled": True, "items": [{"subagent_id": stub.child_model,
                "recommended_use": "Local child-model-wait browser probe",
                "route": {"kind": "api_model", "target_id": "openai-compatible::" + stub.child_model},
                "effort": "low"}]}))
        settings_path = data_root / "settings.json"
        write_settings_file(settings_path, settings)
        server = KeylessIsolatedServer(wait_clone, data_root, settings_path)
        server.start(ready_timeout=120)
        oracle = ArtifactOracle(data_root)
        page = None
        try:
            project = _api(server.base_url, "POST", "/api/projects", {"name": "Delegated model access"})["project"]
            receipt.update(project=project, server_url=server.base_url, server_pid=server.proc.pid,
                           candidate_identity=wait_clone.identity)
            with sync_playwright() as pw:
                browser = pw.chromium.launch()
                page = browser.new_page(viewport={"width": 1100, "height": 900}, reduced_motion="reduce")
                page_errors = []
                page.on("pageerror", lambda error: page_errors.append(str(error)))
                try:
                    page.goto(server.base_url, wait_until="domcontentloaded")
                    page.locator("#chat-input").wait_for(state="visible", timeout=30000)
                    _controls_project_closed(page)
                    created = _api(server.base_url, "POST", "/api/tasks", {
                        "description": parent_marker + ": delegate the model-access verification to your configured child.",
                        "title": "Delegated model access", "chat_id": project["chat_id"],
                        "project_id": project["id"], "source": "web", "memory_mode": "forked",
                        "metadata": {"delegation_role": "root"},
                    }, timeout=60)
                    parent_id = receipt["parent_id"] = created["task_id"]
                    children = wait_until(lambda: oracle.child_task_ids(parent_id), 90)
                    assert children and len(children) == 1, children
                    child_id = receipt["child_id"] = children[0]
                    waiting = wait_until(lambda: next((row for row in oracle.task_result(child_id).get("model_waits", {}).values()
                                                      if row.get("state") == "waiting"), None), 90)
                    assert waiting, oracle.task_result(child_id)
                    child = receipt["child_while_waiting"] = oracle.task_result(child_id)
                    receipt["initial_wait"] = waiting
                    assert child["parent_task_id"] == parent_id and child["root_task_id"] == parent_id
                    assert child["delegation_role"] == "subagent"
                    assert child["project_id"] == project["id"]
                    assert waiting["reason"] == "auth" and waiting["requires_owner_action"] is True
                    assert waiting["auto_continue"] is False
                    assert len(stub.refused_requests) == 1 and len(stub.author_calls["child"]) == 1
                    assert stub.parent_hold.arrived.wait(30), "the root did not reach its controlled second round"
                    assert not stub.parent_hold.release.is_set()
                    wait_id = waiting["wait_id"]
                    receipt["state_while_child_waiting"] = _api(server.base_url, "GET", "/api/state")
                    receipt["child_detail_while_waiting"] = _api(server.base_url, "GET", f"/api/tasks/{child_id}")
                    # Reload hydrates the actual canonical gateway projection;
                    # no hand-authored state or socket frame can supply the row.
                    page.reload(wait_until="domcontentloaded")
                    page.locator("#chat-input").wait_for(state="visible", timeout=30000)
                    # The actual /api/state snapshot posts the child's own System message in Main's feed.
                    card = _controls_owner_action(page, child_id)
                    card.wait_for(state="attached", timeout=30000)
                    _controls_capture(page, destination, "main-child-wait-before-control")
                    retry = card.locator(f'.model-wait-row[data-wait-id="{wait_id}"]').get_by_role(
                        "button", name="Try again", exact=True)
                    expect(retry).to_be_enabled(timeout=30000)
                    _controls_project_closed(page)
                    _controls_capture(page, destination, "main-child-wait-actionable")
                    projected = next(row for row in receipt["state_while_child_waiting"]["main_attention"]
                                     if row["task_id"] == child_id)
                    assert projected["model_waits"][wait_id]["task_attempt"] == waiting["task_attempt"]
                    stub.allow_retry.set()
                    with page.expect_response(lambda response: response.url.endswith('/api/decisions')
                                              and response.request.method == 'POST') as retried:
                        retry.click()
                    response = retried.value
                    request_body, confirmation = response.request.post_data_json, response.json()
                    receipt["retry"] = {"request": request_body, "status": response.status, "response": confirmation}
                    assert response.ok and confirmation["ok"], confirmation
                    assert request_body["decision_id"] == f"model_wait:{child_id}:{wait_id}"
                    assert request_body["action"] == "retry" and request_body["revision"] == waiting["revision"]
                    assert confirmation["wait"]["task_attempt"] == waiting["task_attempt"]
                    final = receipt["child_final"] = wait_durable_result(oracle, child_id, timeout=90)
                    assert final["status"] == "completed", final
                    resolved = final["model_waits"][wait_id]
                    assert resolved["state"] == "resolved" and resolved["resolution"] == "retry_requested"
                    assert resolved["task_attempt"] == waiting["task_attempt"]
                    assert resolved["applied_request_id"] == request_body["request_id"]
                    assert len(stub.author_calls["child"]) == 2 and len(stub.refused_requests) == 1
                    assert oracle.child_task_ids(parent_id) == [child_id]
                    expect(card).to_have_count(0, timeout=30000)
                    _controls_project_closed(page)
                    _controls_capture(page, destination, "main-child-retry-complete")
                    assert not page_errors, page_errors
                    receipt["passed"] = True
                finally:
                    _controls_capture(page, destination, "final")
                    receipt.update(refused_provider_requests=stub.refused_requests,
                                   author_requests=stub.author_calls, model_calls=stub.kinds(), page_errors=page_errors,
                                   state_at_exit=_api(server.base_url, "GET", "/api/state"))
                    for name in ("parent_id", "child_id"):
                        if receipt.get(name):
                            receipt[name + "_at_exit"] = oracle.task_result(receipt[name])
                    (destination / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
                    browser.close()
        finally:
            stub.parent_hold.release.set()
            server.stop()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_main_removes_an_unknown_late_answer_across_clients_without_resend(wait_clone, tmp_path, engine):
    """Owner 2026-10-12: an accepted late answer whose dispatch witness died with the previous
    process stops counting as pending only by the owner's removal, and its Main copy leaves;
    nothing is resent or marked delivered."""
    from playwright.sync_api import sync_playwright, expect
    from ouroboros.owner_quiz import (
        mark_published, mark_quiz_history_written, quiz_answer_frame, quiz_states, reconcile_terminal,
        record_answered, record_asked, record_publication, record_quiz_delivery_failure,
    )
    from ouroboros.projects_registry import create_project
    from ouroboros.task_results import write_task_result
    from supervisor.message_bus import log_chat

    destination = _controls_evidence(tmp_path, f"main-remove-unknown-{engine}")
    data_root = tmp_path / "instance" / "data"
    data_root.mkdir(parents=True)
    task_id, quiz_id, question = "late-unknown-author", "edition-question", "Which edition should ship?"
    project = create_project(data_root, "removal", name="Removal Project")
    chat = project["chat_id"]
    write_task_result(data_root, task_id, "completed", chat_id=chat, project_id=project["id"], result="Shipped.")
    record_asked(data_root, task_id, quiz_id=quiz_id, question=question, options=["Printed", "Searchable"],
                 assumption="Ship the searchable edition", chat_id=chat)
    record_publication(data_root, task_id, quiz_id, chat)
    log_chat("out", chat, 1, question, task_id=task_id, record_type="quiz", quiz=quiz_states(data_root, task_id)[quiz_id],
             drive_root=data_root, require_write=True)
    mark_published(data_root, task_id, quiz_id)
    reconcile_terminal(data_root, task_id)
    # What the previous process recorded before its restart: the accepted answer, its history row
    # and the owner's own message, whose dispatch witness is gone.
    answered = record_answered(data_root, task_id, quiz_id=quiz_id, option_index=1, request_id="late-owner",
                               allow_expired=True)["block"]
    log_chat("system", chat, 0, quiz_answer_frame(answered, 1, ""), task_id=task_id, record_type="quiz_answer",
             client_message_id=f"quiz_answer:{task_id}:{quiz_id}", quiz=dict(answered), drive_root=data_root, require_write=True)
    mark_quiz_history_written(data_root, task_id, quiz_id)
    log_chat("in", chat, 1, "2. Searchable", source="web", client_message_id=f"quiz_late_answer:{task_id}:{quiz_id}",
             drive_root=data_root, require_write=True)
    record_quiz_delivery_failure(data_root, task_id, quiz_id, "late_answer_delivery_unknown")
    def own_rows():
        rows = dialogue_test.log_rows(data_root / "logs" / "chat.jsonl")
        # This question's rows (its ask, answer, the owner's message), not the task's own terminal summary.
        return [row for row in rows if (row.get("quiz") or {}).get("quiz_id") == quiz_id
                or quiz_id in str(row.get("client_message_id") or "")]

    before = own_rows()
    receipt = {"engine": engine, "task_id": task_id, "quiz_id": quiz_id,
               "journey": "Remove an unknown late answer from pending: two clients, duplicate and stale actions, reload"}
    selector = f'#chat-messages .chat-quiz-card[data-task-id="{task_id}"][data-quiz-id="{quiz_id}"]'
    with ScriptedStubModel([]) as stub:
        settings_path = data_root / "settings.json"
        write_settings_file(settings_path, keyless_settings(stub, OUROBOROS_MAX_WORKERS=1))
        server = KeylessIsolatedServer(wait_clone, data_root, settings_path)
        server.start(ready_timeout=120)
        pages, posts = [], []
        try:
            receipt.update(server_url=server.base_url, server_pid=server.proc.pid)
            with sync_playwright() as pw:
                browser = getattr(pw, engine).launch()
                try:
                    for viewport, touch in (({"width": 1100, "height": 900}, False), ({"width": 390, "height": 844}, True)):
                        page = browser.new_context(viewport=viewport, has_touch=touch, reduced_motion="reduce").new_page()
                        pages.append(page)
                        page.goto(server.base_url, wait_until="domcontentloaded")
                    pages[0].on("request", lambda request: posts.append(request.post_data_json)
                                if request.url.endswith('/api/decisions') and request.method == 'POST' else None)
                    for index, page in enumerate(pages):
                        card = page.locator(selector)
                        expect(card.locator('.chat-quiz-status-text')).to_have_text(
                            'Answer recorded · delivery unknown', timeout=30000)
                        expect(card.locator('.chat-quiz-dismiss')).to_have_text('Remove from pending')
                        _controls_project_closed(page)
                        _controls_capture(page, destination, f"client-{index + 1}-pending")
                    with pages[0].expect_response(lambda response: response.url.endswith('/api/decisions')
                                                  and response.request.method == 'POST') as removed:
                        pages[0].locator(selector).locator('.chat-quiz-dismiss').dblclick()
                    receipt["remove_response"] = {"status": removed.value.status, "body": removed.value.json()}
                    assert removed.value.ok and receipt["remove_response"]["body"]["question_delivery"] == {
                        "kind": "answer", "state": "owed", "reason_code": "late_answer_delivery_unknown", "dismissed": True}
                    for index, page in enumerate(pages):  # the other client learns it live
                        expect(page.locator(selector)).to_have_count(0, timeout=30000)
                        _controls_capture(page, destination, f"client-{index + 1}-removed")
                    assert [post["action"] for post in posts] == ["dismiss"], "a double click posts one removal"
                    stale = _api(server.base_url, "POST", "/api/decisions", {
                        "request_id": "stale-tab", "decision_id": f"quiz:{task_id}:{quiz_id}", "action": "dismiss"})
                    assert stale["ok"] is True and stale["duplicate"] is True
                    for index, page in enumerate(pages):
                        page.reload(wait_until="domcontentloaded")
                        page.locator('#chat-input').wait_for(timeout=30000)
                        expect(page.locator(selector)).to_have_count(0, timeout=30000)
                        _controls_capture(page, destination, f"client-{index + 1}-reloaded")
                    canonical = quiz_states(data_root, task_id)[quiz_id]
                    receipt["canonical_after_removal"] = canonical
                    assert canonical["state"] == "answered" and canonical["answered_index"] == 1
                    assert canonical["delivery"]["state"] == "owed" and canonical["delivery"]["dismissed_at"]
                    assert canonical["delivery"]["reason_code"] == "late_answer_delivery_unknown"
                    assert not canonical.get("discharged_at") and "delivered_at" not in canonical["delivery"]
                    history = _api(server.base_url, "GET", f"/api/chat/history?chat_id={chat}")["messages"]
                    receipt["project_history"] = history
                    assert any(row.get("text") == "2. Searchable" for row in history), "the owner's answer stays readable"
                    assert own_rows() == before, "no row of this answer was written or resent"
                    assert stub.kinds() == [], "removal starts no model turn and resends nothing"
                finally:
                    receipt["state_at_exit"] = _api(server.base_url, "GET", "/api/state")
                    for index, page in enumerate(pages):
                        _controls_capture(page, destination, f"client-{index + 1}-final")
                    (destination / "receipt.json").write_text(json.dumps(receipt, indent=2, default=str), encoding="utf-8")
                    browser.close()
        finally:
            server.stop()
