"""Missing #1660 controls/quietness only; the existing consumer journeys stay intact.

A task that cannot continue without the owner is one System message in Main's feed with the
task's own controls; a task waiting on nothing the owner must do posts none, and Main's waiting
pill stays hidden.

The gateway, SPA, decision transport and task producers are real. Model judgments
and provider access are fixture-local. Evidence is retained on both RED and GREEN
via OUROBOROS_BROWSER_EVIDENCE_OUT; reached/not-reached stages name the proof limit.
"""

from contextlib import contextmanager
import json
import shutil
import uuid

import pytest

from devtools.benchmarks.common.server_runner import _api
from tests.system_e2e.harness import (
    ArtifactOracle, KeylessIsolatedServer, ModelGate, ScriptedStubModel,
    body_text, keyless_settings, wait_durable_result, wait_until, write_settings_file,
)
from tests.system_e2e.test_system_scenarios_w3a import _completion_step
from tests.test_main_consumer_journeys_browser import (
    _ControlsAuthRefusalStub, _controls_capture, _controls_evidence,
    _controls_owner_action, _controls_project_closed,
)
from tests.test_main_materials_browser import _m1660_material_question
from tests.test_main_results_browser import _nf_observe, _nf_retain
from tests.test_owner_wait_integration import wait_clone as wait_clone


pytestmark = [pytest.mark.serial, pytest.mark.browser]


@contextmanager
def _session(candidate, tmp_path, stub, name, stages, *, engine="chromium", seed=None, settings=None):
    """Reuse candidate attestation and evidence helpers, including failure screens."""
    from playwright.sync_api import sync_playwright

    root = tmp_path / "instance" / "data"
    root.mkdir(parents=True)
    facts = {"engine": engine, "planned": stages, "reached": [], "passed": False,
             "boundary": "Loopback scripted provider; real task, gateway, SPA and browser. No owner credentials."}
    if seed:
        facts["seed"] = seed(root)
    destination = _controls_evidence(tmp_path, name)
    settings_path = root / "settings.json"
    write_settings_file(settings_path, keyless_settings(stub, OUROBOROS_MAX_WORKERS=1, **(settings or {})))
    server = KeylessIsolatedServer(candidate, root, settings_path)
    oracle = ArtifactOracle(root)
    server.start(ready_timeout=120)
    try:
        with sync_playwright() as pw:
            browser = getattr(pw, engine).launch()
            page = browser.new_page(viewport={"width": 1100, "height": 900}, reduced_motion="reduce")
            _nf_observe(page, facts, engine)
            try:
                page.goto(server.base_url, wait_until="domcontentloaded")
                page.locator("#chat-input").wait_for(state="visible", timeout=30000)
                yield page, server, oracle, facts, destination
                assert not facts["page_errors"], facts["page_errors"]
                facts["passed"] = True
            finally:
                facts["not_reached"] = [stage for stage in stages if stage not in facts["reached"]]
                facts["model_calls"] = stub.kinds()
                facts["task_at_exit"] = oracle.task_result(facts["task_id"]) if facts.get("task_id") else None
                facts["queue_at_exit"] = oracle.queue_snapshot()
                _nf_retain(page, server, candidate, destination, facts)
                browser.close()
    finally:
        server.stop()
        if (root / "logs").exists():
            shutil.copytree(root / "logs", destination / "logs", dirs_exist_ok=True)
        (destination / "shutdown.json").write_text(json.dumps({"server_pid": server.proc.pid,
            "returncode": server.proc.poll(), "data_root": str(root)}, indent=2), encoding="utf-8")


def _start(server, facts, marker):
    project = _api(server.base_url, "POST", "/api/projects", {"name": marker})["project"]
    created = _api(server.base_url, "POST", "/api/tasks", {
        "description": marker + ": continue the assigned work using the primary model.",
        "title": marker, "project_id": project["id"], "chat_id": project["chat_id"],
        "source": "web", "memory_mode": "forked", "metadata": {"delegation_role": "root"},
    }, timeout=60)
    facts.update(project=project, task_id=created["task_id"], created=created)
    return created["task_id"]


def _wait(oracle, task_id):
    row = wait_until(lambda: next((row for row in oracle.task_result(task_id).get("model_waits", {}).values()
                                  if row.get("state") == "waiting"), None), 90)
    assert row, oracle.task_result(task_id)
    return row


def _census(page, facts, label):
    """A cold real browser must consume a complete census before absence counts."""
    url = page.url
    # Retire the previous document's in-flight periodic state request before
    # observing the new document's initial response.
    page.goto("about:blank")
    with page.expect_response(lambda response: response.url.endswith("/api/state")
                              and response.request.method == "GET", timeout=30000) as read:
        page.goto(url, wait_until="domcontentloaded")
    state = read.value.json()
    facts[label] = state
    assert read.value.ok and state["main_attention_complete"] and state["current_questions_complete"], state
    page.locator("#chat-input").wait_for(state="visible", timeout=30000)
    page.wait_for_function("() => { const s = document.querySelector('#chat-status');"
                           "return s && !s.classList.contains('offline'); }")
    _controls_project_closed(page)
    return state


def _manual_card(page, oracle, facts, task_id):
    waiting = facts["initial_wait"] = _wait(oracle, task_id)
    assert waiting["reason"] == "auth" and waiting["requires_owner_action"] is True
    assert waiting["auto_continue"] is False and waiting["availability_observation"] == "unavailable"
    state = _census(page, facts, "manual_census")
    projected = next(row for row in state["main_attention"] if row["task_id"] == task_id)
    assert projected["model_waits"][waiting["wait_id"]]["revision"] == waiting["revision"]
    _controls_project_closed(page)
    control = _controls_owner_action(page, task_id).locator(f'.model-wait-row[data-wait-id="{waiting["wait_id"]}"]')
    control.get_by_role("button", name="Change model or account", exact=True).wait_for()
    return waiting, control


def test_main_change_model_applies_wait_role_and_reads_resumed_result(wait_clone, tmp_path):
    from playwright.sync_api import expect

    marker, resumed = "MAIN1660_CHANGE_MODEL", []
    answer = "The same Project task completed on the replacement model selected in Main."
    with _ControlsAuthRefusalStub(marker, resumed) as stub:
        stub._explicit_model_ids = ["mock-model", "replacement-model"]
        complete = _completion_step(answer, answer_form="answer")
        def finish(body):
            resumed.append(body)
            # A real handover requires a substantive step on the replacement.
            # Then select completion, acknowledging the host's source if offered.
            if len(resumed) == 1:
                return {"tool": "list_files", "arguments": {"root": "system_repo", "path": "."}}
            return complete(body)
        stub.script[1:] = [finish] * 4
        with _session(wait_clone, tmp_path, stub, "main-change-model", [
            "manual wait visible", "replacement selected", "revision-bound switch accepted",
            "same author resumed on replacement", "Main result read", "cold census retired wait",
        ]) as (page, server, oracle, facts, evidence):
            task_id = _start(server, facts, marker)
            waiting, control = _manual_card(page, oracle, facts, task_id)
            facts["reached"].append("manual wait visible")
            before = _api(server.base_url, "GET", "/api/settings")
            control.get_by_role("button", name="Change model or account", exact=True).click()
            model = control.locator("[data-model-role-model]")
            expect(model).to_be_visible()
            model.fill("replacement-model")
            model.press("Tab")
            expect(control.locator("[data-wait-persist]")).not_to_be_checked()
            expect(control.locator("[data-wait-scope]")).to_have_text("This role changes until the task ends.")
            _controls_capture(page, evidence, "model-choice-before-apply")
            facts["reached"].append("replacement selected")
            stub.allow_retry.set()
            page.wait_for_timeout(1200)
            assert not resumed, "An API wait cannot resume without an explicit control"
            with page.expect_response(lambda response: response.url.endswith("/api/decisions")
                                      and response.request.method == "POST") as switched:
                control.get_by_role("button", name="Apply to this task", exact=True).click()
            response = switched.value
            request_body, confirmation = response.request.post_data_json, response.json()
            facts["switch"] = {"request": request_body, "status": response.status, "response": confirmation}
            assert response.ok and confirmation["ok"], confirmation
            assert request_body == {"request_id": request_body["request_id"],
                "decision_id": f'model_wait:{task_id}:{waiting["wait_id"]}', "revision": waiting["revision"],
                "action": "switch", "model": "openai-compatible::replacement-model",
                "credential_profile_id": "", "use_local": False, "persist_role": False}
            assert confirmation["wait"]["task_attempt"] == waiting["task_attempt"]
            assert waiting["role"] == "main"
            facts["reached"].append("revision-bound switch accepted")
            final = facts["final"] = wait_durable_result(oracle, task_id, timeout=90)
            resolved = final["model_waits"][waiting["wait_id"]]
            assert final["status"] == "completed" and resolved["resolution"] == "model_switched", final
            assert resolved["state"] == "resolved" and resolved["task_attempt"] == waiting["task_attempt"]
            assert resolved["applied_request_id"] == request_body["request_id"]
            facts["resumed_provider_requests"] = resumed
            assert 2 <= len(resumed) <= 3 and all(body["model"] == "replacement-model" for body in resumed)
            assert marker in body_text(resumed[0]) and len(stub.refused_requests) == 1
            # Public /api/tasks admission does not emit the chat task_received event.
            # The original admission and complete durable identity set prove no new task.
            assert final["api_admission"]["status"] == "accepted"
            from ouroboros.task_store import task_facts
            facts["durable_task_ids"] = sorted(task_facts(server.data_root))
            assert facts["durable_task_ids"] == [task_id]
            assert wait_until(lambda: [row for row in oracle.events("task_done") if row.get("task_id") == task_id], 30)
            assert len([row for row in oracle.events("task_done") if row.get("task_id") == task_id]) == 1
            after = _api(server.base_url, "GET", "/api/settings")
            facts["role_settings"] = {key: [before.get(key), after.get(key)]
                                     for key in ("OUROBOROS_MODEL", "OUROBOROS_MODEL_ACCOUNTS")}
            assert all(old == new for old, new in facts["role_settings"].values())
            facts["reached"].append("same author resumed on replacement")
            page.locator(f'.project-answer[data-task-id="{task_id}"]').get_by_text(answer, exact=True).wait_for(timeout=30000)
            _controls_project_closed(page)
            _controls_capture(page, evidence, "main-resumed-result")
            facts["reached"].append("Main result read")
            state = _census(page, facts, "resolved_census")
            assert not any(row["task_id"] == task_id for row in state["main_attention"])
            expect(_controls_owner_action(page, task_id)).to_have_count(0)
            facts["reached"].append("cold census retired wait")


def test_webkit_main_material_reader_then_model_wait_settings_entry(wait_clone, tmp_path):
    """One targeted WebKit flow; Settings navigation makes no authentication claim."""
    from playwright.sync_api import expect

    marker, resumed = "MAIN1660_SETTINGS", []
    with _ControlsAuthRefusalStub(marker, resumed) as stub:
        with _session(wait_clone, tmp_path, stub, "webkit-main-material-settings", [
            "manual wait visible", "PNG decoded", "Markdown reader read", "Settings Providers reached",
            "navigation did not resume task",
        ], engine="webkit", seed=lambda root: _m1660_material_question(root)[0]) as (page, server, oracle, facts, evidence):
            task_id = _start(server, facts, marker)
            waiting, control = _manual_card(page, oracle, facts, task_id)
            facts["reached"].append("manual wait visible")
            card = page.locator(f'#chat-messages .project-question-card[data-quiz-id="{facts["seed"]["quiz_id"]}"]')
            image = card.locator("img.chat-file-thumb")
            expect(image).to_be_visible()
            page.wait_for_function("img => img.complete && img.naturalWidth === 240 && img.naturalHeight === 120",
                                   arg=image.element_handle())
            facts["reached"].append("PNG decoded")
            card.get_by_role("button", name="decision-brief.md", exact=False).click()
            dialog = page.locator("dialog.document-reader")
            expect(dialog).to_be_visible()
            expect(dialog.get_by_role("heading", name="Decision brief", exact=True)).to_be_visible()
            expect(dialog.get_by_text("Choose the amber panel for the printed edition.", exact=True)).to_be_visible()
            _controls_project_closed(page)
            _controls_capture(page, evidence, "webkit-material-reader")
            page.keyboard.press("Escape")
            expect(dialog).to_have_count(0)
            facts["reached"].append("Markdown reader read")
            expect(control.get_by_role("button", name="Sign in again", exact=True)).to_be_hidden()
            control.get_by_role("button", name="Settings", exact=True).click()
            expect(page.locator("#page-settings.active")).to_be_visible()
            expect(page.locator('[data-settings-tab="providers"]')).to_have_attribute("aria-selected", "true")
            expect(page.locator('[data-settings-panel="providers"]')).to_be_visible()
            _controls_project_closed(page)
            _controls_capture(page, evidence, "webkit-settings-providers")
            facts["reached"].append("Settings Providers reached")
            latest = oracle.task_result(task_id)["model_waits"][waiting["wait_id"]]
            assert latest["state"] == "waiting" and latest["revision"] == waiting["revision"]
            assert not resumed and len(stub.refused_requests) == 1
            assert not any(row["method"] == "POST" and any(path in row["url"]
                           for path in ("/api/decisions", "/login", "/resume")) for row in facts["requests"])
            facts["reached"].append("navigation did not resume task")


def test_main_own_pause_stays_quiet_after_complete_census(wait_clone, tmp_path):
    from playwright.sync_api import expect

    marker = "MAIN1660_OWN_PAUSE"
    gate = ModelGate(lambda body: bool(body.get("tools")) and marker in body_text(body), timeout=180)
    with ScriptedStubModel([{"tool": "list_files", "arguments": {"root": "system_repo", "path": "."}}], gate=gate) as stub:
        try:
            with _session(wait_clone, tmp_path, stub, "main-own-pause-quiet", [
                "ordinary task running", "owner Pause accepted", "owner Pause settled", "complete census quiet",
            ]) as (page, server, oracle, facts, evidence):
                task_id = _start(server, facts, marker)
                assert gate.arrived.wait(90), "The task never reached its controlled provider round"
                facts["reached"].append("ordinary task running")
                response = page.request.post(server.base_url + f"/api/tasks/{task_id}/pause",
                                             data={"request_id": uuid.uuid4().hex})
                facts["pause_response"] = response.json()
                assert response.ok and facts["pause_response"]["ok"], facts["pause_response"]
                facts["reached"].append("owner Pause accepted")
                gate.release.set()
                # The member's own pause row settles first; the root fence, the truth of
                # Paused (``owner_pause_control.settle_requested_owner_pauses``), follows.
                paused = wait_until(lambda: (row if (row := oracle.task_result(task_id)).get("budget_pause", {}).get("state") == "paused"
                                             and row.get("owner_pause", {}).get("state") == "paused" else None), 90)
                assert paused and paused["budget_pause"]["reason"] == "owner", oracle.task_result(task_id)
                assert paused["owner_pause"]["state"] == "paused", paused
                facts["paused"] = paused
                facts["reached"].append("owner Pause settled")
                state = _census(page, facts, "paused_census")
                activity = next(row for row in state["active_chat_activities"] if row["activity_id"] == task_id)
                assert activity["phase"] == "budget_paused" and activity["pause_cause"] == "owner", activity
                assert not state["main_attention"] and not state["current_questions"]
                expect(page.locator("#chat-messages .chat-bubble.owner-action")).to_have_count(0)
                expect(page.locator("#page-chat .chat-question-nav")).to_be_hidden()
                _controls_capture(page, evidence, "main-owner-pause-quiet")
                facts["reached"].append("complete census quiet")
                assert not gate.timed_out and len(stub.kinds()) == 1
        finally:
            gate.release.set()


def test_main_observable_automatic_wait_is_quiet_until_manual_action_required(wait_clone, tmp_path):
    """Real producer/catalog wire and controls; no claim of a worker engine refusal.

    A fixture invokes TaskModelWait for a real public task held at its provider.
    The existing fake daemon supplies catalog observations only. This isolates
    the quietness rule without adding a fake model-operation engine or UI state.
    """
    from concurrent.futures import ThreadPoolExecutor
    import queue
    import threading
    from types import SimpleNamespace
    from playwright.sync_api import expect
    from ouroboros.gateways.claudexor import (
        ClaudexorGateway, ClaudexorSubscriptionWindowExhausted, discover_daemon_at,
    )
    from ouroboros.model_wait import TaskModelWait
    from tests.system_e2e.interfaces import FakeClaudexorDaemon

    recovered = threading.Event()
    cancelled = threading.Event()

    class CatalogDaemon(FakeClaudexorDaemon):
        def _route(self, method, path, record):
            clean = path.split("?", 1)[0]
            if method == "GET" and clean == "/v2/model-sources":
                return 200, {"sources": [{"id": "codex", "credentialHarness": "codex"}]}
            if method == "GET" and clean == "/v2/model-sources/codex/models":
                if not recovered.is_set():
                    return 503, {"code": "subscription_window_exhausted", "message": "Fixture catalog still unavailable"}
                return 200, {"source": "codex", "credentialProfileId": "fixture-profile",
                             "models": [{"id": "fixture-auto"}]}
            return super()._route(method, path, record)

    marker = "MAIN1660_AUTO_WAIT"
    gate = ModelGate(lambda body: bool(body.get("tools")) and marker in body_text(body), timeout=240)
    with CatalogDaemon() as daemon, ScriptedStubModel([{"tool": "finish_task", "arguments": {
            "action": "finish", "answer": "Automatic wait fixture finished."}}], gate=gate) as stub:
        daemon.install(tmp_path / "catalog")
        with ClaudexorGateway(discover_daemon_at(tmp_path / "catalog")) as gateway:
            gateway.handshake()
            adapter = SimpleNamespace(claudexor_model_sources=gateway.list_model_sources,
                                      claudexor_model_catalog=gateway.list_source_models)
            try:
                with _session(wait_clone, tmp_path, stub, "main-automatic-manual-wait", [
                    "automatic catalog wait produced", "complete automatic census quiet",
                    "manual continuation selected", "same wait visible in Main",
                    "Main restored automatic continuation", "catalog recovery resolved same wait",
                ]) as (page, server, oracle, facts, evidence):
                    facts["boundary"] = ("Fixture invokes production TaskModelWait on an actual public task held at its loopback provider. "
                        "Existing fake daemon supplies catalog refusal/recovery; real gateway, mailbox, census and SPA. "
                        "Producer events are retained in a fixture queue; the browser reads canonical census, not worker event delivery. "
                        "Does not prove a worker-generated engine refusal or vendor recovery.")
                    task_id = _start(server, facts, marker)
                    assert gate.arrived.wait(90)
                    events = queue.Queue()
                    controller = TaskModelWait(task={"id": task_id, "_attempt": 1,
                        "budget_drive_root": str(server.data_root)}, drive_root=oracle.task_drive(task_id).data_root,
                        event_queue=events, worker_slot_held=True)
                    arguments = {"model": "claudexor::codex=fixture-auto", "model_role": "main",
                                 "model_account_override": "fixture-profile"}
                    refusal = ClaudexorSubscriptionWindowExhausted("Fixture automatic quota", reset_at="2099-01-01T00:00:00Z")
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        future = pool.submit(controller.wait, adapter, refusal, arguments, cancelled)
                        try:
                            assert wait_until(lambda: daemon.calls("GET", "/v2/model-sources/codex/models"), 30)
                            waiting = facts["automatic_wait"] = _wait(oracle, task_id)
                            assert waiting["auto_continue"] is True and waiting["requires_owner_action"] is False
                            assert waiting["source"] == "codex" and waiting["credential_profile_id"] == "fixture-profile"
                            facts["reached"].append("automatic catalog wait produced")
                            state = _census(page, facts, "automatic_census")
                            activity = next(row for row in state["active_chat_activities"] if row["activity_id"] == task_id)
                            assert activity["model_waits"][waiting["wait_id"]] == waiting
                            assert not state["main_attention"] and not state["current_questions"]
                            expect(page.locator("#chat-messages .chat-bubble.owner-action")).to_have_count(0)
                            expect(page.locator("#page-chat .chat-question-nav")).to_be_hidden()
                            expect(page.locator(f'#chat-messages .chat-live-card[data-task-id="{task_id}"]')).to_have_count(0)
                            _controls_capture(page, evidence, "main-automatic-wait-quiet")
                            facts["reached"].append("complete automatic census quiet")
                            # Ordinary public control selects manual continuation;
                            # the producer derives requires_owner_action itself.
                            decision = {"request_id": uuid.uuid4().hex,
                                "decision_id": f'model_wait:{task_id}:{waiting["wait_id"]}',
                                "revision": waiting["revision"], "action": "auto_continue", "auto_continue": False}
                            response = page.request.post(server.base_url + "/api/decisions", data=decision)
                            facts["manual_decision"] = {"request": decision, "status": response.status, "response": response.json()}
                            assert response.ok and response.json()["ok"], facts["manual_decision"]
                            manual = wait_until(lambda: (row if (row := oracle.task_result(task_id)["model_waits"][waiting["wait_id"]]).get("applied_request_id") == decision["request_id"] else None), 30)
                            assert manual and manual["auto_continue"] is False and manual["requires_owner_action"] is True, manual
                            facts["reached"].append("manual continuation selected")
                            state = _census(page, facts, "manual_census")
                            projected = next(row for row in state["main_attention"] if row["task_id"] == task_id)
                            assert projected["model_waits"][waiting["wait_id"]] == manual
                            control = _controls_owner_action(page, task_id).locator(
                                f'.model-wait-row[data-wait-id="{waiting["wait_id"]}"]')
                            automatic = control.locator("[data-wait-auto]")
                            expect(automatic).to_be_visible()
                            expect(automatic).not_to_be_checked()
                            _controls_capture(page, evidence, "main-manual-wait-visible")
                            facts["reached"].append("same wait visible in Main")
                            with page.expect_response(lambda response: response.url.endswith("/api/decisions")
                                                      and response.request.method == "POST") as resumed:
                                automatic.check()
                            response = resumed.value
                            facts["automatic_decision"] = {"request": response.request.post_data_json,
                                                           "status": response.status, "response": response.json()}
                            assert response.ok and response.json()["ok"], facts["automatic_decision"]
                            submitted = response.request.post_data_json
                            assert submitted["decision_id"] == decision["decision_id"] and submitted["revision"] == manual["revision"]
                            assert submitted["action"] == "auto_continue" and submitted["auto_continue"] is True
                            assert wait_until(lambda: oracle.task_result(task_id)["model_waits"][waiting["wait_id"]].get("applied_request_id") == submitted["request_id"], 30)
                            state = _census(page, facts, "automatic_again_census")
                            assert not state["main_attention"]
                            expect(page.locator("#chat-messages .chat-bubble.owner-action")).to_have_count(0)
                            expect(page.locator("#page-chat .chat-question-nav")).to_be_hidden()
                            facts["reached"].append("Main restored automatic continuation")
                            recovered.set()
                            assert future.result(timeout=60) == arguments
                            resolved = facts["resolved_wait"] = oracle.task_result(task_id)["model_waits"][waiting["wait_id"]]
                            assert resolved["state"] == "resolved" and resolved["resolution"] == "resource_available"
                            assert resolved["task_attempt"] == waiting["task_attempt"]
                            facts["reached"].append("catalog recovery resolved same wait")
                        finally:
                            cancelled.set()
                            facts["daemon_calls"] = daemon.calls()
                            controller.close()
                            facts["producer_events"] = list(events.queue)
                    gate.release.set()
                    facts["final"] = wait_durable_result(oracle, task_id, timeout=90)
                    assert facts["final"]["status"] == "completed"
                    assert not gate.timed_out
            finally:
                gate.release.set()
                cancelled.set()
