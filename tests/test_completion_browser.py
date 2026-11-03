"""Real completion producers and Direct/Project consumers, live and recovered."""
import hashlib
import json
import uuid
from urllib.parse import parse_qs, urlsplit

import pytest

from devtools.benchmarks.common.server_runner import _api
from tests.ci_evidence import output_dir
from tests.system_e2e.harness import (
    ArtifactOracle, KeylessIsolatedServer, ModelGate, keyless_settings, message_text,
    start_server, wait_durable_result, wait_until,
)
from tests.test_chat_addressing_browser import ToolCallOnlyModel, _OBSERVE_CARDS
from tests.test_chat_history_recovery_browser import _click_project
from tests.test_owner_wait_integration import wait_clone as clone_fixture
from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET
from tests.ui_console_errors import PageErrors
from tests.ui_failure_evidence import FailureEvidence

wait_clone = clone_fixture
pytestmark = [pytest.mark.serial, pytest.mark.ui_browser]
_ENGINES = [("chromium", 1440), ("webkit", 390)]
_STOP_REASON = "The requested verification is unfinished; this is the selected partial answer."
_OBSERVE = f"({_CAPTURE_TEST_SOCKET})();({_OBSERVE_CARDS.replace('document.documentElement', 'document')})();" + """
(() => {
    window.__completionFrames = [];
    window.__completionMounts = [];
    const observeMount = (card, when) => {
        const rect = card.getBoundingClientRect();
        const ancestors = [];
        for (let node = card; node instanceof Element; node = node.parentElement) {
            const css = getComputedStyle(node);
            ancestors.push({tag: node.tagName, id: node.id, display: css.display,
                visibility: css.visibility, opacity: css.opacity, hidden: node.hidden});
        }
        window.__completionMounts.push({when, time_ms: performance.now(),
            task_id: card.dataset.taskId, is_connected: card.isConnected,
            frame_count: window.__completionFrames.length,
            rect: {x: rect.x, y: rect.y, width: rect.width, height: rect.height},
            inside_viewport: rect.bottom > 0 && rect.right > 0 && rect.top < innerHeight && rect.left < innerWidth,
            css_visible: card.isConnected && rect.width > 0 && rect.height > 0 &&
                ancestors.every(a => a.display !== 'none' && a.visibility !== 'hidden' && Number(a.opacity) !== 0),
            ancestors, markup: card.outerHTML, text: card.textContent,
            title: card.querySelector('[data-live-title]')?.textContent,
            lines: [...card.querySelectorAll(':scope > [data-live-timeline] > .chat-live-line')]
                .map(line => ({key: line.dataset.liveLineKey, class: line.className, text: line.textContent,
                    headline: line.querySelector('.chat-live-line-title')?.textContent,
                    body: line.querySelector('.chat-live-line-body')?.textContent || ''})),
            review_groups: card.querySelectorAll('[data-review-group-toggle]').length,
            child_cards: card.querySelectorAll('.chat-live-card.subagent').length,
            actions: [...card.querySelectorAll('button')].map(b => ({text: b.textContent,
                cancel: b.hasAttribute('data-cancel-run'), convert: b.hasAttribute('data-turn-into-project'),
                hidden: b.hidden, disabled: b.disabled}))});
    };
    new MutationObserver(rows => {
        for (const row of rows) for (const node of row.addedNodes) {
            if (!(node instanceof Element)) continue;
            const cards = [...node.querySelectorAll('.chat-live-card')];
            if (node.matches('.chat-live-card')) cards.push(node);
            for (const card of cards) {
                observeMount(card, 'mutation_observer');
                requestAnimationFrame(() => observeMount(card, 'next_animation_frame'));
            }
        }
    }).observe(document, {childList: true, subtree: true});
    const CapturedSocket = window.WebSocket;
    window.WebSocket = class extends CapturedSocket {
        constructor(...args) {
            super(...args);
            this.addEventListener('message', event => {
                window.__completionFrames.push(JSON.parse(event.data));
            });
        }
    };
})();"""


def _finish(answer, *, stop=False):
    return {"tool": "finish_task", "arguments": {"action": "stop" if stop else "finish",
        "answer": answer, **({"rationale": _STOP_REASON} if stop else {})}}


def _environment(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    original_env = KeylessIsolatedServer._env
    monkeypatch.setattr(KeylessIsolatedServer, "_env", lambda server: {
        **original_env(server), "HOME": str(home), "USERPROFILE": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
    })


def _evidence(request, tmp_path):
    directory = output_dir(request.config) or tmp_path / "evidence"
    directory = directory / ("completion-" + hashlib.sha256(request.node.nodeid.encode()).hexdigest()[:16])
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _project(server, tmp_path):
    room = tmp_path / "attached-room"
    room.mkdir()
    project = _api(server.base_url, "POST", "/api/projects", {
        "name": "Completion acceptance room", "path": str(room),
    }, timeout=60).get("project") or {}
    assert project.get("id") and project.get("chat_id"), project
    return project


def _open(page, server, project):
    page.goto(server.base_url, wait_until="domcontentloaded")
    page.wait_for_function("() => window.__ouroWs?.ws?.readyState === 1", timeout=30000)
    feed = _click_project(page, project) if project else "#chat-messages"
    # Visibility precedes the phone drawer's release of the arriving room.
    # Fill cannot focus an inert composer, even when Playwright calls it editable.
    page.wait_for_function("feed => { const node = document.querySelector(feed); return node && !node.closest('[inert]'); }",
                           arg=feed)
    return feed


def _reconnect(page, chat_id):
    page.wait_for_function("() => Boolean(window.__ouroWs?._lastSha)", timeout=30000)
    page.wait_for_function("() => window.__ouroWs?.ws?.readyState === 1", timeout=30000)
    page.evaluate("window.__completionDocument = true; window.__oldSocket = window.__ouroWs.ws")
    def history(response):
        url = urlsplit(response.url)
        return url.path == "/api/chat/history" and parse_qs(url.query).get("chat_id", ["1"]) == [str(chat_id)]
    with page.expect_response(history, timeout=60000):
        page.evaluate("window.__oldSocket.close()")
    page.wait_for_function("() => window.__ouroWs.ws !== window.__oldSocket && window.__ouroWs.ws?.readyState === 1",
                           timeout=30000)
    assert page.evaluate("window.__completionDocument === true"), "reconnect reloaded the document"


def _metrics(page, oracle, task_id):
    page.wait_for_function("id => window.__completionFrames.some(f => f.type === 'log'"
                          " && f.data?.type === 'task_metrics_event' && f.data.task_id === id)",
                          arg=task_id, timeout=30000)
    rows = [row for row in oracle.supervisor_rows("task_metrics_event") if row.get("task_id") == task_id]
    assert rows
    return rows


def _delivery(oracle, task_id, result, answer, chat_id, *, stop=False):
    from ouroboros.artifacts import read_actor_source_bytes
    from supervisor.terminal_delivery import terminal_answer_receipts

    assert result["result"] == answer, result
    assert result["terminal_origin"] == "model_final", result
    digest = hashlib.sha256(answer.encode()).hexdigest()
    facts = wait_until(lambda: (row if (row := terminal_answer_receipts(oracle.data_root, task_id))["state"] == "delivered" else None), 30)
    assert facts and not facts["owed_delivery_ids"] and not facts["unverified_delivery_ids"], facts
    assert len(facts["delivered"]) == 1, facts
    receipt = facts["delivered"][0]
    assert receipt["chat_id"] == chat_id and receipt["text_sha256"] == digest, receipt
    assert receipt["basis"] == "send_handler_returned"
    raw = read_actor_source_bytes(oracle.data_root, task_id, receipt["source_ref"])
    source = json.loads(raw)
    assert source["text"] == answer and source["terminal_origin"] == "model_final", source
    if stop:
        axes = result["outcome_axes"]
        assert axes["objective"]["reason"] == "author_stop"
        assert axes["execution"]["task_completion"]["action"] == "stop"
        assert axes["execution"]["task_completion"]["rationale"] == _STOP_REASON
        assert axes["review"]["status"] != "pass"
    return {"selected_sha256": digest, "outbox": facts, "emitted_source": source}


def _receipt_only_mounts(page, task_id):
    """Host-attested Stop is attention, not substantive completion work."""
    facts = page.evaluate("() => ({frames: window.__completionFrames, mounts: window.__completionMounts})")
    # DOM lineKey is a random rendering identity, not the private tools| dedupe
    # key. Replay the real host frames through the production receipt projector.
    projected = page.evaluate("""async ({frames, mounts, taskId}) => {
        const {summarizeChatLiveEvent} = await import('/static/modules/log_events.js');
        const {applyToolObservation} = await import('/static/modules/chat_activity.js');
        const result = {};
        for (const mount of mounts.filter(m => m.task_id === taskId)) {
            const record = {items: []};
            let view = null;
            for (const frame of frames.slice(0, mount.frame_count)) {
                if (frame.type !== 'log' || frame.data?.task_id !== taskId) continue;
                const summary = summarizeChatLiveEvent({...frame.data, _live_tool_frame: true});
                if (summary?.toolCall) view = applyToolObservation(record, summary.toolCall);
            }
            result[mount.frame_count] = view;
        }
        return result;
    }""", {**facts, "taskId": task_id})
    for mount in facts["mounts"]:
        if mount["task_id"] != task_id or not mount["is_connected"] or not mount["css_visible"]:
            continue
        assert not mount["review_groups"] and not mount["child_cards"], mount
        assert not any(action["convert"] for action in mount["actions"]), mount
        view = projected[str(mount["frame_count"])]
        assert view and view["receipt"] is True and view["calls"] == 1 and view["errors"] == 0, view
        assert len(mount["lines"]) == 1, mount
        assert mount["lines"][0]["headline"] == view["headline"], (mount, view)
        assert not mount["lines"][0]["body"].strip(), mount
        observed = [frame.get("data", {}) if frame.get("type") == "log" else frame
                    for frame in facts["frames"][:mount["frame_count"]]]
        observed = [row for row in observed if row.get("task_id") == task_id]
        calls = [row for row in observed if row.get("type") in
                 {"tool_call_started", "tool_call", "tool_call_finished"}]
        assert calls and all(row.get("completion_control") is True and not row.get("is_error")
                             for row in calls), calls
        if mount["is_connected"] and mount["css_visible"]:
            assert any(action["cancel"] and not action["hidden"] and not action["disabled"]
                       for action in mount["actions"]), mount
            assert any(row.get("cancelable") is True for row in observed), observed
            assert not any(row.get("type") == "task_done" for row in observed), observed


def _presentation(page, feed, task_id, answer, case, project, *, managed=False):
    from playwright.sync_api import expect

    output = page.locator(feed).get_by_text(answer, exact=True)
    output.wait_for(timeout=30000)
    assert output.count() == 1
    card = page.locator(f'{feed} .chat-live-card[data-task-id="{task_id}"]')
    if case == "finish_only":
        expect(card).to_have_count(0, timeout=30000)
        _receipt_only_mounts(page, task_id)
    else:
        card.wait_for(timeout=30000)
        assert card.count() == 1
        if case != "unfinished_stop":
            assert card.locator(":scope > [data-live-summary-button] [data-live-title]").inner_text().strip()
        if case == "unfinished_stop":
            assert card.locator(":scope > [data-live-summary-button] [data-live-phase]").inner_text() == "Failed"
            if card.get_attribute("data-expanded") != "1":
                card.locator(":scope > [data-live-summary-button] [data-live-disclosure]").click()
            assert _STOP_REASON in card.inner_text()
        if case == "invalid_finish":
            if card.get_attribute("data-expanded") != "1":
                card.locator(":scope > [data-live-summary-button] [data-live-disclosure]").click()
            fold = card.locator(':scope > [data-live-timeline] > .chat-live-line.expandable').filter(has_text="1 tool call · 1 error")
            fold.wait_for(state="visible", timeout=30000)
            if fold.get_attribute("data-expanded") != "1":
                fold.locator("[data-live-line-toggle]").click()
            assert "finish_task" in fold.inner_text()
            assert "1 error" in card.inner_text(), card.inner_text()
    if project:
        assert page.locator(f'#chat-messages .chat-live-card[data-task-id="{task_id}"]').count() == 0
        main_answer = page.locator("#chat-messages").get_by_text(answer, exact=True)
        if managed:
            mirror = page.locator(f'#chat-messages [data-system-type="project_completion_summary"][data-task-id="{task_id}"][data-project-id="{project["id"]}"]')
            mirror.get_by_text(answer, exact=True).wait_for(timeout=30000)
            assert mirror.count() == 1 and main_answer.count() == 1
        else:
            assert main_answer.count() == 0
    return {"answer_count": output.count(), "card_count": card.count(),
            "mounted_cards": page.evaluate("() => window.__mountedCards")}


def _wire_order(page, task_id):
    page.wait_for_function("id => window.__completionFrames.some(f => f.type === 'log'"
                          " && f.data?.type === 'task_done' && f.data.task_id === id)",
                          arg=task_id, timeout=30000)
    frames = page.evaluate("() => window.__completionFrames")
    finals = [i for i, f in enumerate(frames) if f.get("type") == "chat"
              and f.get("task_id") == task_id and f.get("role") == "assistant" and not f.get("is_progress")]
    terminals = [i for i, f in enumerate(frames) if f.get("type") == "log"
                 and f.get("data", {}).get("type") == "task_done" and f["data"].get("task_id") == task_id]
    assert len(finals) == 1 and terminals and finals[0] < terminals[0], (finals, terminals)
    return {"final_indices": finals, "terminal_indices": terminals}


def _write_receipt(path, candidate, **facts):
    path.write_text(json.dumps({"candidate_identity": candidate.identity,
        "source_head": candidate.state.head.decode("ascii").strip(), "served_checkout": str(candidate.path),
        **facts}, ensure_ascii=False, indent=2), encoding="utf-8")


@pytest.mark.parametrize("engine,width", _ENGINES)
@pytest.mark.parametrize("surface", ["direct", "project"])
@pytest.mark.parametrize("case", ["finish_only", "read_then_finish", "invalid_finish", "unfinished_stop"])
def test_native_completion_receipt_keeps_work_and_errors_visible(
    wait_clone, tmp_path, monkeypatch, request, engine, width, surface, case,
):
    from playwright.sync_api import sync_playwright

    marker = "NATIVE_COMPLETION_" + uuid.uuid4().hex
    answer = "The requested result is complete." if case != "unfinished_stop" else "Selected partial answer: проверка ещё не закончена."
    steps = {
        "finish_only": [_finish(answer)],
        "read_then_finish": [{"tool": "read_file", "arguments": {"root": "system_repo", "path": "VERSION"}}, _finish(answer)],
        "invalid_finish": [_finish(""), {"final": answer}],
        "unfinished_stop": [_finish(answer, stop=True)],
    }[case]
    calls = []
    def response(body):
        assert any(marker in message_text(m) for m in body.get("messages", []) if m.get("role") == "user")
        index = len(calls)
        calls.append(body)
        assert index < len(steps), "completion bought an unexpected extra model turn"
        return steps[index]

    _environment(tmp_path, monkeypatch)
    evidence = _evidence(request, tmp_path)
    with ToolCallOnlyModel([response] * len(steps)) as stub:
        server = start_server(wait_clone, tmp_path / "instance", keyless_settings(stub, OUROBOROS_MAX_WORKERS=1))
        oracle = ArtifactOracle(server.data_root)
        try:
            project = _project(server, tmp_path) if surface == "project" else None
            chat_id = project["chat_id"] if project else 1
            with sync_playwright() as pw:
                browser = getattr(pw, engine).launch()
                page = browser.new_page(viewport={"width": width, "height": 900}, has_touch=width < 980)
                errors = PageErrors(page, engine)
                page.add_init_script(_OBSERVE)
                with FailureEvidence(page, browser, evidence, request.node.nodeid, engine) as capture:
                    feed = _open(page, server, project)
                    prefix = f'#pchat-{project["id"]}' if project else "#chat"
                    page.locator(prefix + "-input").fill(marker)
                    page.locator(prefix + "-send").click()
                    task = wait_until(lambda: next((row["task"] for row in oracle.events("task_received")
                        if row.get("task", {}).get("_is_direct_chat") and marker in row["task"].get("text", "")), None), 90)
                    assert task, "composer did not admit the native direct turn"
                    task_id = task["id"]
                    assert task["chat_id"] == chat_id
                    if project:
                        assert task["project_id"] == project["id"]
                    result = wait_durable_result(oracle, task_id, timeout=90)
                    metrics = _metrics(page, oracle, task_id)
                    assert metrics[-1]["completion_tool_calls"] == (0 if case == "invalid_finish" else 1)
                    assert metrics[-1]["tool_calls"] == (2 if case == "read_then_finish" else 1)
                    assert metrics[-1]["tool_errors"] == (1 if case == "invalid_finish" else 0)
                    assert len(calls) == len(steps)
                    # The answer and metrics precede task_done. Until that frame
                    # settles the turn, its host-attested Stop remains legitimate.
                    order = _wire_order(page, task_id)
                    capture.checkpoint("live_presentation", task_id=task_id)
                    try:
                        views = {"live": _presentation(page, feed, task_id, answer, case, project)}
                    except Exception:
                        capture._attempt("completion_mount_diagnostics", lambda: capture.details.update(
                            page.evaluate("() => ({frames: window.__completionFrames, mounts: window.__completionMounts})")))
                        raise
                    delivery = _delivery(oracle, task_id, result, answer, chat_id, stop=case == "unfinished_stop")
                    page.screenshot(path=str(evidence / "live.png"), full_page=True, animations="disabled")
                    capture.checkpoint("reconnect", task_id=task_id)
                    _reconnect(page, chat_id)
                    views["reconnected"] = _presentation(page, feed, task_id, answer, case, project)
                    page.screenshot(path=str(evidence / "reconnected.png"), full_page=True, animations="disabled")
                    capture.checkpoint("reload", task_id=task_id)
                    page.reload(wait_until="domcontentloaded")
                    if project:
                        feed = _click_project(page, project)
                    views["reloaded"] = _presentation(page, feed, task_id, answer, case, project)
                    page.screenshot(path=str(evidence / "reloaded.png"), full_page=True, animations="disabled")
                    assert not errors.unexpected(), errors
                    _write_receipt(evidence / "receipt.json", wait_clone, case=case, surface=surface,
                        engine=engine, width=width, task=task, result=result, task_metrics=metrics,
                        observed_model_calls=len(calls), errors=errors, views=views, delivery=delivery, wire_order=order)
        finally:
            server.stop()
            assert server.proc.poll() is not None


@pytest.mark.parametrize("engine,width", _ENGINES)
def test_unfinished_project_parent_preserves_running_child_and_full_output(
    wait_clone, tmp_path, monkeypatch, request, engine, width,
):
    from playwright.sync_api import sync_playwright

    marker = "COMPLETION_PARENT_" + uuid.uuid4().hex
    child_marker = "COMPLETION_CHILD_" + uuid.uuid4().hex
    parent_answer = "Selected partial answer: child work is still running."
    child_answer = "CHILD_OUTPUT_HEAD\n" + "Exact child result: яё𐍈🚀.\n" * 240 + "CHILD_OUTPUT_TAIL"
    gate = ModelGate(lambda body: body.get("model") == "completion-child" and bool(body.get("tools")), timeout=300)
    calls = {"parent": [], "child": []}
    def response(body):
        child = body.get("model") == "completion-child"
        actor = "child" if child else "parent"
        calls[actor].append(body)
        index = len(calls[actor])
        if child:
            assert index <= 2, "child took an unexpected extra model turn"
            return ({"tool": "read_file", "arguments": {"root": "system_repo", "path": "VERSION"}}
                    if index == 1 else _finish(child_answer))
        assert any(marker in message_text(m) for m in body.get("messages", []) if m.get("role") == "user")
        assert index <= 2, "stopped parent bought another model turn"
        if index == 1:
            return {"tool": "schedule_subagent", "arguments": {"subagent_id": "completion-child",
                "objective": child_marker + ": read VERSION and report your complete result.",
                "expected_output": "The full selected child result including its final marker."}}
        assert gate.arrived.wait(90), "child never reached its held model call"
        return _finish(parent_answer, stop=True)

    _environment(tmp_path, monkeypatch)
    evidence = _evidence(request, tmp_path)
    with ToolCallOnlyModel([response] * 4, gate=gate, model_ids=["mock-model", "completion-child"]) as stub:
        settings = keyless_settings(stub, OUROBOROS_MAX_WORKERS=2, OUROBOROS_SUBAGENTS=json.dumps({
            "enabled": True, "items": [{"subagent_id": "completion-child", "recommended_use": "Local completion fixture",
                "route": {"kind": "api_model", "target_id": "openai-compatible::completion-child"}, "effort": "max"}]}))
        server = start_server(wait_clone, tmp_path / "instance", settings)
        oracle = ArtifactOracle(server.data_root)
        try:
            project = _project(server, tmp_path)
            with sync_playwright() as pw:
                browser = getattr(pw, engine).launch()
                page = browser.new_page(viewport={"width": width, "height": 900}, has_touch=width < 980)
                page.add_init_script(_OBSERVE)
                with FailureEvidence(page, browser, evidence, request.node.nodeid, engine) as capture:
                    feed = _open(page, server, project)
                    created = _api(server.base_url, "POST", "/api/tasks", {
                        "description": marker + ": delegate the read, then stop with your partial answer while it runs.",
                        "title": "Unfinished parent", "chat_id": project["chat_id"], "project_id": project["id"],
                        "source": "web", "memory_mode": "forked", "metadata": {"delegation_role": "root"}})
                    parent_id = created["task_id"]
                    assert gate.arrived.wait(90)
                    result = wait_durable_result(oracle, parent_id, timeout=90)
                    [child_id] = oracle.child_task_ids(parent_id)
                    child_before = oracle.task_result(child_id)
                    assert child_before["status"] in {"scheduled", "running"}, child_before
                    assert not gate.release.is_set() and not gate.timed_out
                    assert child_id not in oracle.cancel_intents()
                    assert len(calls["parent"]) == 2
                    capture.checkpoint("parent_stopped_child_running", parent_id=parent_id, child_id=child_id)
                    parent_delivery = _delivery(oracle, parent_id, result, parent_answer, project["chat_id"], stop=True)
                    _presentation(page, feed, parent_id, parent_answer, "unfinished_stop", project, managed=True)
                    page.locator(f'{feed} .chat-live-card[data-task-id="{child_id}"]').wait_for(timeout=30000)
                    page.screenshot(path=str(evidence / "parent-stopped-child-running.png"), full_page=True, animations="disabled")
                    gate.release.set()
                    child_result = wait_durable_result(oracle, child_id, timeout=120)
                    assert child_result["result"] == child_answer and child_result["terminal_origin"] == "model_final"
                    assert child_result["status"] == "completed" and child_id not in oracle.cancel_intents()
                    assert len(calls["parent"]) == 2 and len(calls["child"]) == 2
                    snapshots = {}
                    for stage in ("live", "reconnected", "reloaded"):
                        capture.checkpoint("child_output_" + stage, parent_id=parent_id, child_id=child_id)
                        if stage == "reconnected":
                            _reconnect(page, project["chat_id"])
                        elif stage == "reloaded":
                            page.reload(wait_until="domcontentloaded")
                            feed = _click_project(page, project)
                        parent = page.locator(f'{feed} .chat-live-card[data-task-id="{parent_id}"]')
                        child_card = page.locator(f'{feed} .chat-live-card[data-task-id="{child_id}"]')
                        for card in (parent, child_card):
                            card.wait_for(state="attached", timeout=30000)
                            if card.get_attribute("data-expanded") != "1":
                                card.locator(":scope > [data-live-summary-button] [data-live-disclosure]").click()
                        # The authored result precedes its lifecycle notification;
                        # the latter may repeat an explicitly shortened [RESULT].
                        line = child_card.locator(":scope > [data-live-timeline] > .chat-live-line.expandable").filter(has_text="CHILD_OUTPUT_HEAD").first
                        if line.get_attribute("data-expanded") != "1":
                            line.locator("[data-live-line-toggle]").click()
                        # Inline complete bodies and fetched full bodies share this
                        # node; the -full class only denotes the fetched variant.
                        full = line.locator(":scope > .chat-live-line-body")
                        full.get_by_text("CHILD_OUTPUT_TAIL", exact=False).wait_for(timeout=30000)
                        rendered = full.inner_text()
                        selected = rendered[rendered.index("CHILD_OUTPUT_HEAD"):rendered.index("CHILD_OUTPUT_TAIL") + len("CHILD_OUTPUT_TAIL")]
                        assert selected.split() == child_answer.split(), "the visible full output lost content"
                        full.evaluate("node => { const range = document.createRange(); range.selectNodeContents(node);"
                                      " const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range); }")
                        assert "CHILD_OUTPUT_TAIL" in page.evaluate("window.getSelection().toString()")
                        assert page.locator(feed).get_by_text(parent_answer, exact=True).count() == 1
                        snapshots[stage] = {"parent_cards": parent.count(), "child_cards": child_card.count(), "full_chars": len(full.inner_text())}
                        page.screenshot(path=str(evidence / (stage + ".png")), full_page=True, animations="disabled")
                    _write_receipt(evidence / "receipt.json", wait_clone, engine=engine, width=width,
                        parent_id=parent_id, child_id=child_id, parent_result=result, child_before=child_before,
                        child_result=child_result, parent_delivery=parent_delivery,
                        selected_child_sha256=hashlib.sha256(child_answer.encode()).hexdigest(),
                        model_calls={name: len(rows) for name, rows in calls.items()}, snapshots=snapshots)
        finally:
            gate.release.set()
            server.stop()
            assert server.proc.poll() is not None
