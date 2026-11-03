"""Real gateway/SPA result projections plus a native required-question journey."""
import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from devtools.benchmarks.common.server_runner import _api
from tests.test_owner_wait_integration import wait_clone as clone_fixture
from tests.test_native_owner_wait_browser import chat_connection
from tests._shared import producer_evidence_dir
from tests.test_ui_coherence_browser import pill_landing
from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data  # noqa: F401
from tests.ui_console_errors import PageErrors
from tests.test_chat_history_paging_browser import _OBSERVE_HISTORY, _FRAMES, _idle, _reads
from tests.system_e2e.harness import (
    ArtifactOracle, KeylessIsolatedServer, ScriptedStubModel, keyless_settings,
    wait_durable_result, wait_until, write_settings_file,
)

wait_clone = clone_fixture
pytestmark = [pytest.mark.serial, pytest.mark.browser]


def _journey_history_painted(page, project, task_id, since):
    """Synchronize with the newly opened room's source, never with its desired title."""
    page.wait_for_function("""([since, cid, tid]) => window.__historyReads.slice(since).some(r =>
        r.chatId === cid && r.done && r.body?.messages.some(m => m.task_id === tid && m.is_progress))""",
        arg=[since, project['chat_id'], task_id], timeout=30000)
    feed = f'#pchat-{project["id"]}-messages'
    _idle(page, feed)
    read = next(read for read in reversed(_reads(page, project['chat_id']))
                if any(row.get('task_id') == task_id and row.get('is_progress') for row in read['body']['messages']))
    witness = next(row for row in read['body']['messages'] if row.get('role') == 'user' and row.get('history_id'))
    page.locator(f'{feed} [data-history-id="{witness["history_id"]}"]').wait_for(state='attached')
    return {'paint_witness': witness['history_id'], 'page_cursor': read['body']['page_cursor'],
            'narration': [row for row in read['body']['messages']
                          if row.get('task_id') == task_id and row.get('is_progress') and row.get('narration') is not False]}


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_quiet_cards_keep_live_and_reloaded_presentation(direct_server_with_data, tmp_path, engine):
    from playwright.sync_api import expect, sync_playwright
    from ouroboros import dialogue_store
    from ouroboros.task_results import write_task_result
    from ouroboros.task_store import delete_document
    from ouroboros.terminal_projection import _project_row
    from supervisor.message_bus import log_chat
    from tests.ui_chat_viewport_smoke import (
        _CAPTURE_TEST_SOCKET, _OBSERVE_STATE_READS, _emit_ws_frame, _wait_socket_open_quiescent,
    )

    fixture = direct_server_with_data
    root = fixture["data_dir"]
    evidence = producer_evidence_dir(tmp_path / "screenshots", "quiet-cards", env="OUROBOROS_BROWSER_EVIDENCE_OUT")
    evidence.mkdir(parents=True, exist_ok=True)
    notice = "A tool this task used failed and nothing recovered it"
    cases = [("quiet", True, "", "Done · no message"),
             ("notice", True, notice, "Done with warnings"), ("ordinary", False, "", "Done")]
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        page = browser.new_page(viewport={"width": 1100, "height": 900}, reduced_motion="reduce")
        errors = PageErrors(page, engine)
        page.add_init_script(f"({_CAPTURE_TEST_SOCKET})(); ({_OBSERVE_STATE_READS})();")
        try:
            page.goto(fixture["url"], wait_until="domcontentloaded")
            _wait_socket_open_quiescent(page)
            for name, silent, host_notice, _headline in cases:
                tid = f"quiet-card-{name}"
                result = write_task_result(root, tid, "completed", chat_id=1, root_task_id=tid,
                    delegation_role="root", result="Internal quiet words" if silent else "The published answer.",
                    terminal_origin="model_final", wake_silent=silent, terminal_host_notice=host_notice,
                    reason_code="tool_failure" if host_notice else "final_message",
                    outcome_axes={"execution": {"status": "degraded" if host_notice else "ok"}})
                summary = _project_row(tid, result, {}, {"chat_id": 1})
                progress = {"task_id": tid, "chat_id": 1, "content": f"Checking {name}", "ts": summary["ts"],
                            "narration": True}
                dialogue_store.append(root, "progress", progress)
                _emit_ws_frame(page, {**progress, "type": "chat", "role": "assistant", "is_progress": True})
                if not silent:
                    log_chat("out", 1, 1, result["result"], task_id=tid, drive_root=root)
                    _emit_ws_frame(page, {"type": "chat", "role": "assistant", "chat_id": 1,
                                         "task_id": tid, "content": result["result"]})
                dialogue_store.append(root, "chat", summary)
                _emit_ws_frame(page, {**summary, "type": "chat", "role": "system",
                                     "system_type": "task_summary", "content": summary["text"]})

            for visit in ("live", "reloaded", "pruned"):
                if visit == "pruned":
                    assert delete_document(root, "quiet-card-quiet")
                if visit != "live":
                    page.reload(wait_until="domcontentloaded")
                for name, _silent, _notice, headline in cases:
                    card = page.locator(f'.chat-live-card[data-task-id="quiet-card-{name}"]')
                    expect(card.locator('[data-live-phase]')).to_have_text(headline, timeout=30000)
                    expect(card).to_have_attribute("data-finished", "1")
                    if card.get_attribute("data-expanded") != "1":
                        card.locator('[data-live-disclosure]').click()
                    if name == "notice":
                        expect(card).to_contain_text(notice)
                    elif name == "quiet":
                        assert card.locator('.chat-live-line-body').count() == 0
                        # Pruned results can settle from hidden evidence without retaining a timeline row.
                        assert card.locator('.chat-live-line.done .chat-live-line-title').all_text_contents() in ([], [headline])
                        card.screenshot(path=str(evidence / f"{engine}-{visit}-quiet.png"), animations="disabled")
                expect(page.get_by_text("The published answer.", exact=True)).to_have_count(1)
                assert "Internal quiet words" not in page.locator('#chat-messages').inner_text()
                page.screenshot(path=str(evidence / f"{engine}-{visit}.png"), animations="disabled")
            assert not errors.unexpected(), errors
        finally:
            browser.close()


def _seed_history(root):
    """Use production result/chat producers; only reviewer/model facts are fixtures."""
    from ouroboros import agent_task_pipeline as pipeline, dialogue_store
    from ouroboros.project_dialogue import append_terminal_task_projection
    from ouroboros.store import document_text
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.task_store import delete_document, read_document
    from supervisor.message_bus import log_chat
    from tests.test_delivery_forced_finalization import _forced_test_context, _bind_host_pass

    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for task_id in ("old-cancelled", "old-unknown"):
        dialogue_store.append(root, "progress", {"task_id": task_id, "chat_id": 1,
            "content": f"Inspecting retained history for {task_id}", "ts": "2026-09-08T10:00:00Z"})
    task = {"id": "old-cancelled", "chat_id": 1, "root_task_id": "old-cancelled", "delegation_role": "root"}
    result = write_task_result(root, task["id"], "cancelled", result="Retained cancellation")
    log_chat("out", 1, 1, "Retained ordinary answer before cancellation", task_id=task["id"], drive_root=root)
    assert append_terminal_task_projection(root, task["id"], task, result, {"chat_id": 1})
    # The old result is lost. Results live in the store now, not in task_results/<id>.json:
    # set its exact document aside and remove it there.
    preserved = root / "old-cancelled-preserved.json"
    preserved.write_text(document_text(read_document(root, task["id"])), encoding="utf-8")
    assert delete_document(root, task["id"]) and load_task_result(root, task["id"]) is None

    loop, registry, ctx, trace = _forced_test_context(root)
    candidate = loop._replace_delivery_candidate(registry, ctx, trace, "The accepted report is available.", control="candidate")
    _bind_host_pass(loop, registry, trace, candidate)
    trace["delivery_candidate"].update(degraded=True, degraded_reason="advisory_plan_review_open")
    env = SimpleNamespace(drive_root=root, repo_dir=root)
    task = {"id": "parent1", "type": "task", "chat_id": 1, "text": "Produce a verified report",
            "suggested_name": "Accepted report", "_skip_post_task_synthesis": True}
    pipeline._store_task_result(env, task, "The accepted report is available.", {}, trace)
    stored = load_task_result(root, "parent1")
    assert stored["outcome_axes"]["objective"]["status"] == "pass"
    assert stored["outcome_axes"]["execution"]["status"] == "degraded"
    dialogue_store.append(root, "progress", {"task_id": "parent1", "chat_id": 1,
        "content": "Verifying the report", "ts": "2026-09-08T10:01:00Z"})
    log_chat("out", 1, 1, "The accepted report is available.", task_id="parent1", drive_root=root)

    pending = []
    direct = {"id": "routing-history", "type": "task", "chat_id": 1,
              "text": "Read the evidence and route the follow-up into its Project.",
              "_is_direct_chat": True, "_skip_post_task_synthesis": True}
    direct_trace = {"tool_calls": [{"tool": "read_file"}, {"tool": "route_to_project"}], "reasoning_notes": []}
    pipeline.emit_task_results(env, None, None, pending, direct, "The work continues in the Project.",
        {"rounds": 2}, direct_trace,
        start_time=0.0, drive_logs=logs)
    final = next(row for row in pending if row["type"] == "send_message")
    log_chat("out", 1, 1, final["text"], task_id=direct["id"], message_meta=final.get("progress_meta", {}), drive_root=root)
    # Even a stopped native turn receives the free facts row from the real
    # emission path; replay consumes it without a second fixture write.
    from ouroboros.gateway.history import _assemble_history_response
    summaries = [row for row in json.loads(_assemble_history_response(root, 1, 50, 200))["messages"]
                 if row.get("task_id") == direct["id"] and row.get("system_type") == "task_summary"]
    assert len(summaries) == 1 and summaries[0]["tool_calls"] == 2 and summaries[0]["text"] == ""
    return {"preserved_path": str(preserved), "preserved_bytes": preserved.read_bytes()}


def _seed_early_failure(root):
    from ouroboros import dialogue_store
    from ouroboros.task_results import write_task_result
    from supervisor.message_bus import log_chat
    from ouroboros.task_finalization import stamp_root_final_phase

    early = {"type": "send_message", "progress_meta": {"outcome_axes": {"execution": {"status": "infra_failed"}}}}
    stamp_root_final_phase(early, {}, post_task_open=True, terminal_status="failed")
    write_task_result(root, "known-failure", "failed", title="Known failed work",
                      root_phase_checkpoint={"post_task_synthesis": "running"},
                      outcome_axes={"execution": {"status": "infra_failed"}})
    dialogue_store.append(root, "progress", {"task_id": "known-failure", "chat_id": 1,
        "content": "Checking the requested result", "ts": "2026-09-08T10:00:00Z"})
    log_chat("out", 1, 1, "The provider failed; retained work remains available.",
             task_id="known-failure", message_meta=early["progress_meta"], drive_root=root)


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_main_current_optional_question_survives_history_horizon_and_reload(wait_clone, tmp_path, engine):
    """A real candidate serves a terminal optional ask absent from initial Main history.

    The census names it and mounts nothing; Main's waiting pill counts it, mounts its copy at its
    own date, lands the card under the chat header with the question focused; the copy takes a
    late answer and leaves, and the pill with it. The same holds after a reload."""
    from playwright.sync_api import expect, sync_playwright
    from ouroboros.gateway.history import _DEFAULT_N_HUMAN
    from ouroboros.owner_quiz import record_asked, record_publication, reconcile_terminal, quiz_states, mark_published
    from ouroboros.project_dialogue import project_question_pointer
    from ouroboros.projects_registry import create_project
    from ouroboros.task_results import write_task_result
    from supervisor.message_bus import log_chat

    root = tmp_path / "instance" / "data"
    root.mkdir(parents=True)
    project = create_project(root, "old-optional", name="Older optional decision")
    task_id, quiz_id = "old-optional-task", "old-optional-question"
    write_task_result(root, task_id, "completed", project_id=project["id"], chat_id=project["chat_id"],
                      result="The draft uses HTML while the format decision remains open.")
    record_asked(root, task_id, quiz_id=quiz_id, question="Which format should the next edition use?",
                 options=["PDF", "HTML"], option_details=["Printable edition", "Accessible web edition"],
                 assumption="HTML for the current draft", chat_id=project["chat_id"])
    record_publication(root, task_id, quiz_id, project["chat_id"])
    reconcile_terminal(root, task_id)
    block = quiz_states(root, task_id)[quiz_id]
    log_chat("out", project["chat_id"], 1, block["question"], task_id=task_id, record_type="quiz",
             quiz=block, drive_root=root, require_write=True)
    mark_published(root, task_id, quiz_id)
    block = quiz_states(root, task_id)[quiz_id]
    pointer = project_question_pointer({"task_id": task_id, "quiz_id": quiz_id}, block, project)
    log_chat("out", 1, 1, pointer["text"], task_id=task_id, message_meta=pointer, drive_root=root, require_write=True)
    for index in range(_DEFAULT_N_HUMAN + 12):
        log_chat("in", 1, 1, f"Later Main history row {index}", drive_root=root)

    evidence = producer_evidence_dir(tmp_path / "screenshots", "ui-result",
                                     env="OUROBOROS_BROWSER_EVIDENCE_OUT")
    evidence.mkdir(parents=True, exist_ok=True)
    with ScriptedStubModel([{"final": "Future editions will use PDF."}]) as stub:
        settings_path = root / "settings.json"
        write_settings_file(settings_path, keyless_settings(stub, OUROBOROS_MAX_WORKERS=1))
        server = KeylessIsolatedServer(wait_clone, root, settings_path)
        server.start(ready_timeout=120)
        try:
            history = _api(server.base_url, "GET", "/api/chat/history?chat_id=1")["messages"]
            assert not any(row.get("quiz_id") == quiz_id for row in history)
            canonical = _api(server.base_url, "GET", "/api/state")
            assert canonical["current_questions_complete"]
            assert any(row["quiz_id"] == quiz_id for row in canonical["current_questions"])
            with sync_playwright() as pw:
                browser = getattr(pw, engine).launch()
                page = browser.new_page(viewport={"width": 1100, "height": 900}, reduced_motion="reduce")
                try:
                    page.goto(server.base_url, wait_until="domcontentloaded")
                    card = page.locator(f'#chat-messages .project-question-card[data-quiz-id="{quiz_id}"]')
                    pill = page.locator('#page-chat .chat-toolbar-row > button.chat-question-nav')
                    landings = []
                    for visit in ("first", "reloaded"):
                        if visit == "reloaded":
                            page.reload()
                        # The census names the question beyond the loaded history; it mounts no copy.
                        expect(pill).to_be_visible(timeout=30000)
                        expect(pill).to_have_text("1 question")
                        expect(pill).to_have_attribute("aria-label", "1 unanswered: show the next question")
                        assert card.count() == 0, "the census never mounts a copy"
                        pill.click()
                        card.get_by_text("Unanswered · task finished", exact=True).wait_for()
                        landing = pill_landing(page, quiz_id)
                        landings.append(landing)
                        assert landing["focused"], landing
                        assert landing["below_header"] >= 0 and landing["above_composer"] >= 0, landing
                        assert abs(landing["below_header"] - 12) <= 3 or landing["scroll_top"] == 0, landing
                        expect(pill).to_be_hidden()  # its only target is on screen
                        assert card.locator('.chat-quiz-comment').count() == 1
                    card.get_by_text("Accessible web edition", exact=True).wait_for()
                    page.screenshot(path=str(evidence / f"{engine}-main-old-optional.png"), animations="disabled")
                    # One quiet x closes the question; it waits while an answer is typed.
                    close = card.get_by_role("button", name="Close question without answering", exact=True)
                    expect(close).to_be_enabled()
                    card.locator('.chat-quiz-comment').fill("A draft that must not be discarded")
                    expect(close).to_be_disabled()
                    card.locator('.chat-quiz-comment').fill("")
                    expect(close).to_be_enabled()
                    with page.expect_response(lambda response: response.url.endswith('/api/decisions')
                                              and response.request.method == 'POST') as answer:
                        card.get_by_role("button", name="PDF Printable edition").click()
                    confirmation = answer.value.json()
                    assert confirmation["ok"] and confirmation["answered_after_terminal"] and confirmation["forwarded"]
                    card.get_by_text("You answered", exact=True).wait_for()
                    page.screenshot(path=str(evidence / f"{engine}-main-old-optional-answered.png"), animations="disabled")
                    stored = _api(server.base_url, "GET", f"/api/tasks/{task_id}")["owner_quiz"][quiz_id]
                    assert stored["state"] == "answered" and stored["answered_index"] == 0
                    assert stored["publication_chat_id"] == project["chat_id"]
                    card.wait_for(state="detached", timeout=15000)
                    expect(pill).to_be_hidden(timeout=15000)
                    (evidence / f"{engine}-main-old-optional.json").write_text(json.dumps({
                        "task_id": task_id, "quiz_id": quiz_id, "project": project,
                        "initial_history_rows": len(history), "confirmation": confirmation,
                        "landings": landings, "server_pid": server.proc.pid, "server_url": server.base_url,
                    }, indent=2))
                finally:
                    browser.close()
        finally:
            server.stop()


@pytest.mark.parametrize("live_engine", ["chromium", "webkit"])
def test_ui_results_and_required_question_journey(wait_clone, tmp_path, monkeypatch, live_engine):
    from playwright.sync_api import sync_playwright

    root = tmp_path / "instance" / "data"
    root.mkdir(parents=True)
    seeded = _seed_history(root)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    original_env = KeylessIsolatedServer._env
    def isolated_env(server):
        return {**original_env(server), "HOME": str(fake_home), "USERPROFILE": str(fake_home),
                "XDG_CONFIG_HOME": str(fake_home / ".config")}
    monkeypatch.setattr(KeylessIsolatedServer, "_env", isolated_env)
    screenshots = producer_evidence_dir(tmp_path / "screenshots", "ui-result",
                                        env="OUROBOROS_BROWSER_EVIDENCE_OUT")
    screenshots.mkdir(parents=True, exist_ok=True)
    steps = [
        {"tool": "escalate", "arguments": {"question": "Which evidence should the report use?",
            "options": [{"label": "Primary sources", "detail": "Use the complete original measurements."},
                        {"label": "Both sources", "detail": "Include the independent replication too."}],
            "stake": "The answer determines the report's evidence.", "wait_for_answer": True}},
        {"final": "The report will include both sources, as requested."},
    ]
    with ScriptedStubModel(steps) as stub:
        settings_path = root / "settings.json"
        write_settings_file(settings_path, keyless_settings(stub, OUROBOROS_MAX_WORKERS=1))
        server = KeylessIsolatedServer(wait_clone, root, settings_path)
        server.start(ready_timeout=120)
        oracle = ArtifactOracle(root)
        try:
            project = _api(server.base_url, "POST", "/api/projects", {"name": "Evidence review"})["project"]
            _seed_early_failure(root)
            with sync_playwright() as pw:
                browser = getattr(pw, live_engine).launch()
                theme = 'light' if live_engine == 'webkit' else 'dark'
                page = browser.new_page(viewport={"width": 1100, "height": 750}, device_scale_factor=2,
                                        color_scheme=theme, reduced_motion="reduce")
                page.add_init_script(f'({_OBSERVE_HISTORY})()')
                errors = PageErrors(page, live_engine)
                posts = []
                page.on('request', lambda request: posts.append(request.post_data_json)
                        if request.method == 'POST' and request.url.endswith('/api/decisions') else None)
                try:
                    page.goto(server.base_url, wait_until="domcontentloaded")
                    early_card = page.locator('.chat-live-card[data-task-id="known-failure"]')
                    early_card.locator('[data-live-phase]').filter(has_text="Failed").wait_for(timeout=30000)
                    early_card.get_by_text("Finalizing…", exact=True).wait_for()
                    # This seeded row has no live actor. The real census can
                    # correctly retire it; it is a replay metadata assertion,
                    # not a screenshot claim of a live finalizing process.
                    cancelled = page.locator('.chat-live-card[data-task-id="old-cancelled"]')
                    cancelled.locator('[data-live-phase]').filter(has_text="Stopped").wait_for(timeout=30000)
                    assert page.get_by_text("Retained ordinary answer before cancellation", exact=True).count() == 1
                    cancelled.screenshot(animations="disabled", path=str(screenshots / f"{live_engine}-retained-cancelled.png"))
                    unknown = page.locator('.chat-live-card[data-task-id="old-unknown"]')
                    unknown.locator('[data-live-meta]').filter(has_text="Outcome unavailable").wait_for(timeout=30000)
                    assert unknown.locator('[data-cancel-run]').count() == 0
                    route = page.locator('.chat-live-card[data-task-id="routing-history"]')
                    route_meta = route.locator(':scope > [data-live-foot] > [data-live-meta]')
                    route_meta.filter(has_text="2 tool calls").wait_for(state="visible")
                    assert "2 tool calls" in route_meta.inner_text()
                    # Synthesis is skipped, so the producer's final is this direct turn's terminal
                    # word (no finalizing hold): one Done line, and the card settles on it.
                    assert route.locator('.chat-live-line.done').count() == 1
                    assert route.get_attribute('data-finished') == '1'
                    assert route.locator('[data-live-phase]').get_attribute('data-phase') == 'done'
                    accepted = page.locator('.chat-live-card[data-task-id="parent1"]')
                    accepted.locator('[data-live-phase]').filter(has_text="Done with warnings").wait_for()
                    # The quiet Reviews door opens independently of Details; the review's
                    # passed state never replaces the task's Done with warnings.
                    door = accepted.locator('[data-live-review-summary]').filter(has_text="Reviews")
                    door.wait_for()
                    assert accepted.get_attribute('data-expanded') == '0'
                    door.click()
                    assert accepted.get_attribute('data-expanded') == '0'
                    accepted.get_by_text("Task acceptance", exact=True).wait_for()
                    accepted.locator('[data-review-group-toggle]').click()
                    assert "passed" in accepted.locator('.review-check-head .review-state').inner_text()
                    accepted.locator('[data-review-facts] > summary').click()
                    assert "verdict=PASS" in accepted.locator('[data-review-attempt-detail]').inner_text()
                    assert "Done with warnings" in accepted.locator('[data-live-phase]').inner_text()
                    page.screenshot(animations="disabled", path=str(screenshots / f"{live_engine}-result-history.png"), full_page=True)
                    with chat_connection(server) as ws:
                        message_id = uuid.uuid4().hex
                        ws.send(json.dumps({"type": "chat", "content": "Ask which evidence to use, then wait for my answer.",
                            "client_message_id": message_id, "chat_id": project["chat_id"], "project_id": project["id"]}))
                        task = wait_until(lambda: next((row["task"] for row in oracle.events("task_received")
                            if (row.get("task", {}).get("metadata", {}).get("origin_message_ref") or {}).get("client_message_id") == message_id), None), 90)
                        assert task
                        wait = wait_until(lambda: (block if (block := oracle.task_result(task["id"]).get("owner_wait", {})).get("state") == "waiting" else None), 90)
                        assert wait
                        pointer = page.locator(f'#chat-messages .project-question-card[data-task-id="{task["id"]}"]')
                        pointer.get_by_text("Waiting for your answer", exact=True).wait_for(timeout=30000)
                        # Main mirrors the Project's own form: the options with their details, the
                        # stake and the own-answer field, plus the chip that opens the Project.
                        main_question = page.locator('#chat-messages .chat-bubble.project-question').filter(
                            has=page.locator(f'.project-question-card[data-task-id="{task["id"]}"]'))
                        assert main_question.locator('.chat-quiz-option').count() == 2
                        assert main_question.locator('.chat-quiz-comment').count() == 1
                        main_question.get_by_text("Include the independent replication too.", exact=True).wait_for()
                        main_question.get_by_text("At stake: The answer determines the report's evidence.", exact=True).wait_for()
                        page.screenshot(animations="disabled", path=str(screenshots / f"{live_engine}-main-question-pointer.png"), full_page=True)
                        # Age the original question beyond the ordinary Project
                        # window. Navigation must reconstruct this exact task/quiz
                        # from its existing detail, including both option details.
                        from ouroboros.gateway.history import _DEFAULT_N_HUMAN
                        from supervisor.message_bus import log_chat
                        # The Main pointer can precede the Project's own question row, which keeps its
                        # ask-time ts when appended: age the window only once that row is written.
                        assert wait_until(lambda: any(row.get("msg_type") == "quiz" for row in _api(
                            server.base_url, "GET", f"/api/chat/history?chat_id={project['chat_id']}")["messages"]), 30)
                        for index in range(_DEFAULT_N_HUMAN + 1):
                            log_chat("in", project["chat_id"], 1, f"Retained later project note {index}", drive_root=root)
                        project_history = _api(server.base_url, "GET", f"/api/chat/history?chat_id={project['chat_id']}")["messages"]
                        assert not any(row.get("msg_type") == "quiz" for row in project_history)
                        # The Project reference is in the bubble's voice line, beside `Ouroboros`, outside the card.
                        main_question.locator('.chat-bubble-voice > .chat-quiz-project').click()
                        quiz = page.locator(f'.chat-quiz-card[data-task-id="{task["id"]}"][data-quiz-id="{wait["quiz_id"]}"]'
                                            ':not(.project-question-card)')
                        quiz.get_by_text("Include the independent replication too.", exact=True).wait_for()
                        # This new Project instance cold-replayed cancelable history;
                        # current activity must restore the existing Stop control.
                        waiting_card = page.locator(f'.chat-live-card[data-task-id="{task["id"]}"]')
                        waiting_card.locator('[data-cancel-run]').wait_for(state="visible", timeout=15000)
                        page.wait_for_function('qid => { const el=document.querySelector(`[data-quiz-id="${qid}"] .chat-quiz-question`); if(!el) return false; const r=el.getBoundingClientRect(); return r.top>=0 && r.bottom<=innerHeight; }', arg=wait["quiz_id"], timeout=15000)
                        waiting_card.screenshot(animations="disabled", path=str(screenshots / f"{live_engine}-cold-stop-control.png"))
                        # The cropped control capture intentionally scrolled;
                        # restore the quiz for the overview after its viewport assertion.
                        quiz.scroll_into_view_if_needed()
                        page.screenshot(animations="disabled", path=str(screenshots / f"{live_engine}-project-question.png"), full_page=True)
                        # One touch in Main answers it: the Main copy shows the recorded answer, then
                        # leaves; the Project card keeps the record (replayed below, in a fresh browser).
                        page.locator('#project-panel-close').click()
                        # Qualify ACK-before-census here; the inverse ordering is the unselected
                        # #1993 qualification. Gate only transport, keeping every real response body.
                        page.evaluate("""tid => { window.__answerShown = false; window.__answerPaint = [];
                            const fetch = window.fetch.bind(window);
                            const gate = new Promise(resolve => window.__releaseAnswerCensus = resolve);
                            window.fetch = async (input,init) => {
                                if(new URL(typeof input === 'string' ? input : input.url,location.href).pathname === '/api/state') await gate;
                                return fetch(input,init);
                            };
                            const card = () => document.querySelector(`#chat-messages .project-question-card[data-task-id="${tid}"]`);
                            const look = () => { const el = card(); if (el && [...el.querySelectorAll('*')].some(n => n.childElementCount === 0
                                && n.textContent.trim() === 'You answered') && [...el.querySelectorAll('.chat-quiz-option.chosen')]
                                .some(o => o.textContent.includes('Both sources'))) window.__answerShown = true; };
                            new MutationObserver(look).observe(document.querySelector('#chat-messages'),
                                {subtree: true, childList: true, characterData: true, attributes: true});
                            const paint = () => { const el=card(), rect=el?.getBoundingClientRect();
                                __answerPaint.push({at:performance.now(), present:!!el, answered:el?.dataset.state === 'answered',
                                    visible:!!rect && rect.bottom>0 && rect.top<innerHeight});
                                if(el) requestAnimationFrame(paint); };
                            requestAnimationFrame(paint); }""", task["id"])
                        main_question.get_by_role("button", name="Both sources").click()
                        main_question.get_by_text('You answered', exact=True).wait_for()
                        page.evaluate(_FRAMES)
                        page.screenshot(animations='disabled', path=str(screenshots / f'{live_engine}-main-answer-confirmed.png'))
                        page.evaluate('window.__releaseAnswerCensus()')
                        main_question.wait_for(state="detached", timeout=15000)
                        assert page.evaluate("window.__answerShown"), "the Main copy left without showing the recorded answer"
                        paint = page.evaluate('window.__answerPaint')
                        confirmed = [frame for frame in paint if frame['answered'] and frame['visible']]
                        assert len(confirmed) >= 2 and confirmed[-1]['at'] - confirmed[0]['at'] >= 4500, paint
                        assert len(posts) == 1 and posts[0]['option_index'] == 1, posts
                        result = wait_durable_result(oracle, task["id"], timeout=90)
                        final_phase = {'completed': 'done', 'failed': 'error', 'cancelled': 'cancelled'}[result['status']]
                        assert result["owner_quiz"][wait["quiz_id"]]["answered_index"] == 1
                        # Only the Main copy went: the exact question still opens in its Project.
                        since = page.evaluate('window.__historyReads.length')
                        page.evaluate("""([project, task, quiz]) => window.dispatchEvent(new CustomEvent('ouro:open-project',
                            {detail: {project, task_id: task, quiz_id: quiz}}))""", [project, task["id"], wait["quiz_id"]])
                        quiz.locator('.chat-quiz-option.chosen').filter(has_text="Both sources").wait_for(timeout=30000)
                        task_card = page.locator(f'.chat-live-card[data-task-id="{task["id"]}"]')
                        history_proof = _journey_history_painted(page, project, task['id'], since)
                        assert task_card.locator('[data-live-title]').inner_text() == "still working"
                        assert task_card.get_attribute('data-finished') == '1'
                        assert task_card.locator('[data-live-phase]').get_attribute('data-phase') == final_phase
                        # Record completion metadata now, then require it after all replays so a
                        # failed result write cannot prevent evidence of the answer/reopen path.
                        model_facts = [task_card.locator('[data-live-facts]').filter(has_text="Last solve response").count()]
                        page.screenshot(animations="disabled", path=str(screenshots / f"{live_engine}-question-answered.png"), full_page=True)
                        task_card.screenshot(animations='disabled', path=str(screenshots / f'{live_engine}-terminal-task.png'))
                        assert not errors.unexpected(), errors
                        (screenshots / f"{live_engine}-journey.json").write_text(json.dumps({"task_id": task["id"], "quiz_id": wait["quiz_id"],
                            "project": project, "model_execution": result.get("model_execution"), "server_pid": server.proc.pid,
                            "server_url": server.base_url, "owner_quiz": result["owner_quiz"][wait["quiz_id"]],
                            "history_proof": history_proof, "posts": posts, "answer_paint": paint, "theme": theme,
                            "delivery_order": "ACK before census", "page_errors": list(errors),
                            "status": result['status'], "model_facts": model_facts[0]}, indent=2))
                finally:
                    page.screenshot(animations='disabled', path=str(screenshots / f'{live_engine}-journey-end.png'))
                    (screenshots / f'{live_engine}-journey-end.json').write_text(json.dumps({
                        'page_errors': list(errors), 'posts': posts, 'history': page.evaluate('window.__historyReads'),
                        'answer_paint': page.evaluate('window.__answerPaint || []'),
                        'task_result': oracle.task_result(task['id']) if 'task' in locals() else {},
                        'events': oracle.events(), 'stub_kinds': stub.kinds()}, indent=2))
                    browser.close()
                for engine, viewport in (("webkit", {"width": 1100, "height": 750}), ("chromium", {"width": 390, "height": 844})):
                    browser = getattr(pw, engine).launch()
                    page = browser.new_page(viewport=viewport, device_scale_factor=2, color_scheme=theme,
                                            has_touch=viewport["width"] < 500, reduced_motion="reduce")
                    page.add_init_script(f'({_OBSERVE_HISTORY})()')
                    replay_errors = PageErrors(page, engine)
                    replay_posts = []
                    page.on('request', lambda request: replay_posts.append(request.url)
                            if request.method == 'POST' and request.url.endswith('/api/decisions') else None)
                    try:
                        page.goto(server.base_url, wait_until="domcontentloaded")
                        # An answered question never enters Main again; its Project replays the record.
                        page.locator('#chat-messages .chat-live-card[data-task-id="parent1"]').wait_for(timeout=30000)
                        assert page.locator('#chat-messages .project-question-card').count() == 0
                        since = page.evaluate('window.__historyReads.length')
                        page.evaluate("""([project, task, quiz]) => window.dispatchEvent(new CustomEvent('ouro:open-project',
                            {detail: {project, task_id: task, quiz_id: quiz}}))""", [project, task["id"], wait["quiz_id"]])
                        replay_quiz = page.locator('.chat-quiz-card').filter(has_text="Which evidence should the report use?")
                        replay_quiz.wait_for()
                        replay_quiz.locator('.chat-quiz-option.chosen').filter(has_text="Both sources").wait_for()
                        # Existing scroll restoration spans 12 animation frames;
                        # the explicit question navigation must still own the
                        # viewport after that restoration would have finished.
                        page.evaluate("() => new Promise(resolve => { let n=0; const step=()=>++n>=14 ? resolve() : requestAnimationFrame(step); requestAnimationFrame(step); })")
                        assert replay_quiz.evaluate("el => { const r=el.getBoundingClientRect(); return r.top >= 0 && r.bottom <= innerHeight; }")
                        replay_card = page.locator(f'.chat-live-card[data-task-id="{task["id"]}"]')
                        replay_history = _journey_history_painted(page, project, task['id'], since)
                        assert replay_card.locator('[data-live-title]').inner_text() == "still working"
                        assert replay_card.get_attribute('data-finished') == '1'
                        assert replay_card.locator('[data-live-phase]').get_attribute('data-phase') == final_phase
                        model_facts.append(replay_card.locator('[data-live-facts]').filter(has_text="Last solve response").count())
                        page.screenshot(animations="disabled", path=str(screenshots / f'{live_engine}-{engine}-{viewport["width"]}-question-replay.png'), full_page=True)
                        assert not replay_posts and not replay_errors.unexpected()
                        (screenshots / f'{live_engine}-{engine}-{viewport["width"]}-replay.json').write_text(json.dumps({
                            'history_proof': replay_history, 'posts': replay_posts, 'page_errors': list(replay_errors),
                            'title': replay_card.locator('[data-live-title]').inner_text(),
                            'phase': replay_card.locator('[data-live-phase]').get_attribute('data-phase'),
                            'model_facts': model_facts[-1]}, indent=2))
                    finally:
                        browser.close()
            assert Path(seeded["preserved_path"]).read_bytes() == seeded["preserved_bytes"]
            from ouroboros.task_results import load_task_result
            assert load_task_result(root, "old-cancelled") is None  # its card came from history alone
            # Keep a failure red after collecting the whole answer/reopen path and its actual lifecycle.
            assert result['status'] == 'completed', result
            assert model_facts == [1, 1, 1], model_facts
        finally:
            server.stop()
            (screenshots / f'{live_engine}-server.log').write_bytes((root / 'logs/server.log').read_bytes())
            assert server.proc.poll() is not None
