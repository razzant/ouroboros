"""Main result limitations and notification destinations in real browsers.

Scripted providers feed the production server; Chromium and WebKit consume the
actual gateway and SPA. Receipts and screenshots are retained on failure too.
"""

import uuid
import json

import pytest

from devtools.benchmarks.common.server_runner import _api
from tests.test_owner_wait_integration import wait_clone as clone_fixture
from tests._shared import producer_evidence_dir
from tests.system_e2e.harness import (
    ArtifactOracle, KeylessIsolatedServer, ModelGate, ScriptedStubModel,
    body_text, keyless_settings, start_server, wait_durable_result,
)
from tests.ui_console_errors import PageErrors
from tests.ui_failure_evidence import FailureEvidence

wait_clone = clone_fixture
pytestmark = [pytest.mark.serial, pytest.mark.browser]



def _nf_environment(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    original = KeylessIsolatedServer._env
    monkeypatch.setattr(KeylessIsolatedServer, "_env", lambda server: {
        **original(server), "HOME": str(home), "USERPROFILE": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
    })


def _nf_evidence(tmp_path, stem):
    path = producer_evidence_dir(tmp_path / "evidence", "main-results",
                                 env="OUROBOROS_BROWSER_EVIDENCE_OUT") / stem
    path.mkdir(parents=True, exist_ok=True)
    return path


def _nf_observe(page, facts, engine):
    facts["frames"], facts["requests"], facts["page_errors"] = [], [], PageErrors(page, engine)
    page.on("websocket", lambda socket: socket.on("framereceived", lambda payload:
            facts["frames"].append(json.loads(payload))))
    page.on("request", lambda request: facts["requests"].append({
        "method": request.method, "url": request.url}))


def _nf_retain(page, server, candidate, evidence, facts):
    """Keep both successful and failing consumer evidence before browser close."""
    facts.update(candidate_identity=candidate.identity,
                 source_head=candidate.state.head.decode("ascii").strip(),
                 served_checkout=str(candidate.path), server_url=server.base_url,
                 server_pid=server.proc.pid)
    errors = []
    for name, action in (
        ("screenshot", lambda: page.screenshot(path=str(evidence / "last-view.png"), animations="disabled")),
        ("html", lambda: (evidence / "last-view.html").write_text(page.content(), encoding="utf-8")),
        ("state", lambda: facts.update(last_state=_api(server.base_url, "GET", "/api/state"))),
        ("main_history", lambda: facts.update(main_history=_api(server.base_url, "GET", "/api/chat/history?chat_id=1"))),
    ):
        try:
            action()
        except Exception as error:
            errors.append({"operation": name, "error": repr(error)})
    facts["evidence_errors"] = errors
    (evidence / "receipt.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")


@pytest.mark.parametrize("engine,width", [("chromium", 1100), ("webkit", 390)])
def test_main_partial_final_has_one_compact_limitation_with_project_closed(
    wait_clone, tmp_path, monkeypatch, request, engine, width,
):
    from playwright.sync_api import expect, sync_playwright

    marker = "MAIN_PARTIAL_" + uuid.uuid4().hex
    answer = "The Linux report is ready. The Windows measurements are still missing."
    rationale = "The Windows verification was not performed."
    calls = []

    def respond(body):
        assert marker in body_text(body), "an unrelated actor consumed the result script"
        calls.append(body)
        assert len(calls) == 1, "stopping must not purchase another author turn"
        return {"tool": "finish_task", "arguments": {
            "action": "stop", "answer": answer, "rationale": rationale,
        }}

    _nf_environment(tmp_path, monkeypatch)
    evidence = _nf_evidence(tmp_path, f"main-partial-final-{engine}-{width}")
    facts = {"engine": engine, "width": width, "answer": answer, "rationale": rationale,
             "stages": {}, "passed": False}
    with ScriptedStubModel([respond], final_answer="Partial verification") as stub:
        server = start_server(wait_clone, tmp_path / "instance", keyless_settings(stub, OUROBOROS_MAX_WORKERS=1))
        oracle = ArtifactOracle(server.data_root)
        try:
            folder = tmp_path / "result-project"
            folder.mkdir()
            project = _api(server.base_url, "POST", "/api/projects", {
                "name": "Partial verification", "path": str(folder),
            }, timeout=60)["project"]
            facts["project"] = project
            with sync_playwright() as pw:
                browser = getattr(pw, engine).launch()
                page = browser.new_page(viewport={"width": width, "height": 900}, reduced_motion="reduce")
                _nf_observe(page, facts, engine)
                with FailureEvidence(page, browser, evidence, request.node.nodeid, engine) as capture:
                    try:
                        page.goto(server.base_url, wait_until="domcontentloaded")
                        page.locator("#chat-input").wait_for(state="visible", timeout=30000)
                        assert page.locator("#project-panel.open").count() == 0
                        created = _api(server.base_url, "POST", "/api/tasks", {
                            "description": marker + ": deliver the Linux result and stop; Windows was not verified.",
                            "title": "Partial verification", "chat_id": project["chat_id"],
                            "project_id": project["id"], "source": "web", "memory_mode": "forked",
                            "metadata": {"delegation_role": "root"},
                        }, timeout=60)
                        task_id = facts["task_id"] = created["task_id"]
                        facts["result"] = wait_durable_result(oracle, task_id, timeout=120)
                        assert facts["result"]["outcome_axes"]["execution"]["task_completion"]["action"] == "stop"
                        assert facts["result"]["terminal_origin"] == "model_final"
                        for stage in ("live", "reloaded"):
                            capture.checkpoint(stage, task_id=task_id)
                            if stage == "reloaded":
                                page.reload(wait_until="domcontentloaded")
                            row = page.locator('#chat-messages .project-answer'
                                               f'[data-task-id="{task_id}"]')
                            row.get_by_text(answer, exact=True).wait_for(timeout=30000)
                            expect(row).to_have_count(1)
                            limitation = row.locator(".project-answer-limitation")
                            expect(limitation).to_have_count(1)
                            expect(limitation).to_be_visible()
                            text = limitation.inner_text()
                            assert text.count(rationale) == 1, text
                            assert "no review approval" in text.lower(), text
                            assert len(text) < 320, "the summary should be a short limitation, not full review detail"
                            assert row.inner_text().count(rationale) == 1
                            assert page.locator("#project-panel.open").count() == 0
                            assert page.locator(f'#chat-messages .chat-live-card[data-task-id="{task_id}"]').count() == 0
                            assert page.locator("#chat-messages").get_by_text(answer, exact=True).count() == 1
                            layout = limitation.evaluate("node => { const r = node.getBoundingClientRect();"
                                " return {width:r.width, height:r.height, left:r.left, right:r.right,"
                                " viewport:innerWidth, lineHeight:getComputedStyle(node).lineHeight}; }")
                            assert layout["left"] >= 0 and layout["right"] <= width + 1, layout
                            facts["stages"][stage] = {"limitation": text, "layout": layout, "row_text": row.inner_text()}
                            page.screenshot(path=str(evidence / f"{stage}.png"), animations="disabled")
                        facts["model_calls"] = len(calls)
                        assert len(calls) == 1
                        assert not facts["page_errors"].unexpected(), facts["page_errors"]
                        facts["passed"] = True
                    finally:
                        _nf_retain(page, server, wait_clone, evidence, facts)
        finally:
            server.stop()
            assert server.proc.poll() is not None


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_real_notification_opens_exact_question_in_project_outside_first_fifty(
    wait_clone, tmp_path, monkeypatch, request, engine,
):
    from playwright.sync_api import expect, sync_playwright
    from ouroboros.projects_registry import create_project

    marker = "OLD_PROJECT_NOTICE_" + uuid.uuid4().hex
    question = "Which edition should this older Project publish?"
    gate = ModelGate(lambda body: bool(body.get("tools")) and marker in body_text(body), timeout=240)
    calls = []

    def respond(body):
        assert marker in body_text(body), "an unrelated actor consumed the notification script"
        calls.append(body)
        if len(calls) == 1:
            return {"tool": "escalate", "arguments": {
                "question": question, "options": [{"label": "Printable PDF"}, {"label": "Web edition"}],
                "stake": "The publication format needs your decision.", "wait_for_answer": True,
            }}
        assert len(calls) == 2, "the addressed question resumed more than once"
        return {"final": "The older Project will publish the selected PDF edition."}

    _nf_environment(tmp_path, monkeypatch)
    evidence = _nf_evidence(tmp_path, f"notification-older-project-{engine}")
    facts = {"engine": engine, "passed": False, "notification_surface": "production in-app toast",
             "simulation_boundary": "client-local opt-in preferences and scripted model only; browser Notification permission unchanged"}
    with ScriptedStubModel([respond] * 3, gate=gate, final_answer="Older Project decision") as stub:
        server = start_server(wait_clone, tmp_path / "instance", keyless_settings(stub, OUROBOROS_MAX_WORKERS=1))
        oracle = ArtifactOracle(server.data_root)
        try:
            folder = tmp_path / "older-project"
            folder.mkdir()
            project = _api(server.base_url, "POST", "/api/projects", {
                "name": "Older publication Project", "path": str(folder),
            }, timeout=60)["project"]
            facts["project"] = project
            created = _api(server.base_url, "POST", "/api/tasks", {
                "description": marker + ": ask which edition to publish, then wait for the owner's answer.",
                "title": "Older publication decision", "chat_id": project["chat_id"],
                "project_id": project["id"], "source": "web", "memory_mode": "forked",
                "metadata": {"delegation_role": "root"},
            }, timeout=60)
            task_id = facts["task_id"] = created["task_id"]
            assert gate.arrived.wait(90), "the real actor never reached the held model round"
            # Production registry producer: place fifty-five newer rooms ahead
            # of the admitted actor before the client reads its sidebar slice.
            for index in range(55):
                create_project(server.data_root, f"later-room-{index}", name=f"Later room {index}")
            initial = facts["before_question_state"] = _api(server.base_url, "GET", "/api/state")
            assert len(initial["projects"]) == 50
            assert project["id"] not in {row["id"] for row in initial["projects"]}
            assert project["chat_id"] in initial["project_chat_ids"]
            with sync_playwright() as pw:
                browser = getattr(pw, engine).launch()
                page = browser.new_page(viewport={"width": 1100, "height": 900}, reduced_motion="reduce")
                # Client-local fixture preference, through its existing storage
                # reader. No Notification API or application method is stubbed.
                page.add_init_script("localStorage.setItem('ouroboros.notifications', JSON.stringify({"
                    "enabled:true, needs_answer:true, task_done:false, important:false, main_reply:false,"
                    "sound:false, show_text:false}));")
                _nf_observe(page, facts, engine)
                with FailureEvidence(page, browser, evidence, request.node.nodeid, engine) as capture:
                    try:
                        page.goto(server.base_url, wait_until="domcontentloaded")
                        page.locator("#chat-input").wait_for(state="visible", timeout=30000)
                        page.wait_for_function("() => { const status = document.querySelector('#chat-status');"
                                               " return status && !status.classList.contains('offline'); }", timeout=30000)
                        facts["notification_permission"] = page.evaluate("typeof Notification === 'undefined' ? 'unavailable' : Notification.permission")
                        assert facts["notification_permission"] != "granted"
                        assert page.locator("#project-panel.open").count() == 0
                        assert page.locator(f'.nav-project-row[data-project-id="{project["id"]}"]').count() == 0
                        capture.checkpoint("ready_with_destination_outside_sidebar", task_id=task_id)
                        gate.release.set()
                        toast = page.locator("#toast-stack .toast").filter(has_text="Ouroboros is waiting for your answer")
                        toast.wait_for(state="visible", timeout=90000)
                        assert page.locator("#project-panel.open").count() == 0
                        assert page.locator(f'.nav-project-row[data-project-id="{project["id"]}"]').count() == 0
                        facts["notice_text"] = toast.inner_text()
                        page.screenshot(path=str(evidence / "notice-main-project-closed.png"), animations="disabled")
                        with page.expect_response(lambda response: "/api/projects?chat_id=" in response.url
                                                  and f"chat_id={project['chat_id']}" in response.url) as addressed:
                            toast.click()
                        facts["addressed_response"] = addressed.value.json()
                        assert [row["id"] for row in facts["addressed_response"]["projects"]] == [project["id"]]
                        feed = page.locator(f'[id="pchat-{project["id"]}-messages"]')
                        feed.wait_for(state="visible", timeout=30000)
                        quiz = feed.locator(f'.chat-quiz-card[data-task-id="{task_id}"]')
                        quiz.get_by_text(question, exact=True).wait_for(timeout=30000)
                        expect(quiz).to_have_count(1)
                        facts["quiz_id"] = quiz.get_attribute("data-quiz-id")
                        assert facts["quiz_id"]
                        assert page.locator("#project-panel.open").count() == 1
                        page.screenshot(path=str(evidence / "activated-exact-project-question.png"), animations="disabled")
                        capture.checkpoint("answer_exact_activated_question", task_id=task_id, quiz_id=facts["quiz_id"])
                        with page.expect_response(lambda response: response.url.endswith("/api/decisions")
                                                  and response.request.method == "POST") as answered:
                            quiz.get_by_role("button", name="Printable PDF", exact=True).click()
                        facts["answer_confirmation"] = answered.value.json()
                        assert facts["answer_confirmation"]["ok"]
                        result = facts["result"] = wait_durable_result(oracle, task_id, timeout=120)
                        stored = result["owner_quiz"][facts["quiz_id"]]
                        assert stored["state"] == "answered" and stored["answered_index"] == 0
                        assert stored["publication_chat_id"] == project["chat_id"]
                        assert len(calls) == 2
                        assert not gate.timed_out
                        assert not facts["page_errors"], facts["page_errors"]
                        facts["passed"] = True
                    finally:
                        _nf_retain(page, server, wait_clone, evidence, facts)
        finally:
            gate.release.set()
            server.stop()
            assert server.proc.poll() is not None
