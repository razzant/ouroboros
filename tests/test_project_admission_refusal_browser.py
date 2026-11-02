"""#1381: a never-admitted Project refusal through real producers, server and SPA.

Producers run in this process against the candidate server's isolated data
root while that server is STOPPED (the fixture's seed rule): the owner's
row is written by the ingress writer, and the real routing picker, promotion,
queue admission and refusal emitters refuse it after a completed registry
rebind-and-back during workspace preflight. Beside it sit an admitted root that
started and failed (canonical result writers) and a refusal recorded by the
legacy producer branch without the positive never-admitted fact. Only the
absent server's live transport is recorded (bridge frames and the terminal
event queue); every owner-visible row is durable.

The server then publishes owed terminal projections itself: its startup scan
on every boot, and its outbox replay for rows the 600 s custody duty
(``reconcile_terminal_projections``, called here while the server serves and
the replacement works) registers. The replacement is admitted by the served
HTTP API while this process holds the registry writer lock, and runs inside
the server, held in its first mock-model call. Main and the Project panel are
read live, after reload, and after a server restart with the page left open.
"""
from __future__ import annotations

from tests import _dialogue as dialogue_test

import copy
import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests._shared import producer_evidence_dir
from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401
from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

PROJECT, PROJECT_NAME = "admission-room", "Admission research"
ADMITTED_FAILURE, REPLACEMENT = "admitted-fail-1381", "replacement-1381"
LEGACY_AT_BOOT, LEGACY_AT_DUTY = "legacy-refused-1381", "legacy-late-1381"
OWNER_TEXT = "Research how Project admission should survive harmless registry activity."
REFUSAL_CAUSE = "Not started: the project changed while the task was being prepared"


class _Bridge:
    """The absent server's live transport: frames are recorded, never delivered."""

    def __init__(self):
        self.sent, self.frames, self.acks = [], [], []

    def send_message(self, chat_id, text, **kwargs):
        self.sent.append({"chat_id": chat_id, "text": text, **copy.deepcopy(kwargs)})

    def broadcast(self, payload):
        self.frames.append(copy.deepcopy(payload))

    def send_routing_ack(self, chat_id, **payload):
        self.acks.append({"chat_id": chat_id, **copy.deepcopy(payload)})


def _producers(monkeypatch, data_dir: Path, repo_dir: Path) -> SimpleNamespace:
    """Point the real supervisor owners at the server's data root."""
    from ouroboros import config
    from ouroboros.utils import append_jsonl
    from supervisor import git_ops, message_bus, queue, state, workers
    from supervisor import worker_chat_lane as lane

    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "SETTINGS_PATH", data_dir / "settings.json")
    for name, value in (("DRIVE_ROOT", data_dir), ("STATE_PATH", data_dir / "state" / "state.json"),
                        ("STATE_LAST_GOOD_PATH", data_dir / "state" / "state.last_good.json"),
                        ("STATE_LOCK_PATH", data_dir / "locks" / "state.lock")):
        monkeypatch.setattr(state, name, value)
    pending, running, sequence = [], {}, {"value": 0}
    pool = {0: SimpleNamespace(active_capacity=True, readiness_exhausted=False, reaping=False, busy_task_id=None)}
    for module in (workers, queue):
        for name, value in (("DRIVE_ROOT", data_dir), ("PENDING", pending), ("RUNNING", running),
                            ("QUEUE_SEQ_COUNTER_REF", sequence)):
            monkeypatch.setattr(module, name, value)
    monkeypatch.setattr(queue, "INITIALIZED", True)
    monkeypatch.setattr(queue, "OWNER_PID", os.getpid())
    monkeypatch.setattr(queue, "QUEUE_SNAPSHOT_PATH", data_dir / "state" / "queue_snapshot.json")
    for name in ("ADMISSION_RESERVATIONS", "ACCEPTANCE_FENCES", "BUDGET_ROOT_FENCES"):
        monkeypatch.setattr(queue, name, {})
    for name, value in (("WORKERS", pool), ("_WORKER_POOL_DISABLED_REASON", ""),
                        ("_repo_writer_gate_reason", ""), ("REPO_DIR", repo_dir)):
        monkeypatch.setattr(workers, name, value)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", data_dir)
    monkeypatch.setattr(git_ops, "REPO_DIR", repo_dir)
    bridge, live_sends = _Bridge(), []
    monkeypatch.setattr(message_bus, "DATA_DIR", data_dir)
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    from supervisor.event_observation import ObservedEventQueue

    class _Bus(ObservedEventQueue):
        """The supervisor's real two-lane bus (membership and atomic put included), recording each live send."""

        def _put(self, item):
            live_sends.append(item)
            super()._put(item)

    bus = _Bus()
    monkeypatch.setattr(workers, "get_event_q", lambda *, create=True: bus)
    ctx = SimpleNamespace(
        DRIVE_ROOT=data_dir, PENDING=pending, RUNNING=running, WORKERS=pool, bridge=bridge,
        enqueue_task=queue.enqueue_task, persist_queue_snapshot=queue.persist_queue_snapshot,
        load_state=state.load_state, append_jsonl=append_jsonl,
        consciousness=SimpleNamespace(inject_observation=lambda *_: None, pause=lambda: None, resume=lambda: None),
        handle_chat_direct=lane.handle_chat_direct, send_with_budget=message_bus.send_with_budget,
        get_chat_agent=lambda: SimpleNamespace(_busy=False),
    )
    return SimpleNamespace(ctx=ctx, bridge=bridge, bus=bus, live_sends=live_sends, pending=pending)


def _rows(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] \
        if path.exists() else []


def _legacy_refusal(host, data_dir: Path, task_id: str, chat: int, objective: str) -> dict:
    """The refusal shape pre-repair producers wrote: bound, failed, no never-admitted fact."""
    from ouroboros import projects_registry as registry
    from ouroboros.task_results import load_task_result
    from supervisor.events_project_routing import _persist_promote_rejection

    registry.bind_task_to_project(data_dir, task_id, PROJECT, chat, origin={"absent": "system"})
    _persist_promote_rejection(host.ctx, {"task_id": task_id, "project_id": PROJECT, "objective": objective,
                                          "routing_token": f"{task_id}-token"},
                               {"task_id": task_id, "reason": "project_routing_fence_changed"})
    stored = load_task_result(data_dir, task_id, strict=True)
    assert stored["status"] == "failed" and "admission_outcome" not in stored
    return stored


def _seed(host, monkeypatch, data_dir: Path, tmp_path: Path) -> dict:
    from ouroboros import projects_registry as registry
    from ouroboros import workspace_admission
    from ouroboros.gateway.routing_decision import _derived_identity, handle_routing_decision
    from ouroboros.project_dialogue import append_chat_annotation
    from ouroboros import routing_wait
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.utils import utc_now_iso
    from supervisor.message_bus import log_chat, send_with_budget

    folder = tmp_path / "room-folder"
    folder.mkdir()
    project = registry.create_project(data_dir, PROJECT, name=PROJECT_NAME, working_dir=str(folder))
    registry.create_project(data_dir, "neighbour-room", name="Neighbour room")
    chat = int(project["chat_id"])

    # An ADMITTED root that really started and failed, recorded by the canonical
    # writers with its terminal publication still owed.
    registry.bind_task_to_project(data_dir, ADMITTED_FAILURE, PROJECT, chat, origin={"absent": "system"})
    write_task_result(data_dir, ADMITTED_FAILURE, "running", project_id=PROJECT, chat_id=chat,
                      root_task_id=ADMITTED_FAILURE, delegation_role="root", title="Admitted comparison run",
                      description="Admitted comparison run", started_at=utc_now_iso())
    send_with_budget(chat, "Comparing the saved registry snapshots.", is_progress=True,
                     task_id=ADMITTED_FAILURE, narration=True)
    write_task_result(data_dir, ADMITTED_FAILURE, "failed", result="The comparison step failed after it started.")
    legacy = _legacy_refusal(host, data_dir, LEGACY_AT_BOOT, chat, "Earlier refused attempt (historical record)")

    # The owner's new-task picker choice in the Project room, from retained ingress.
    client_message_id, picker_token = "owner-picker-1381", "choose-project-1381"
    ts = utc_now_iso()
    log_chat("in", chat, 1, OWNER_TEXT, ts=ts, source="web", client_message_id=client_message_id,
             require_write=True, ensure_record_boundary=True)
    options = [{"action": "new_task_in_project", "project_id": PROJECT, "label": PROJECT_NAME}]
    assert append_chat_annotation(data_dir, client_message_id, action="route_decision",
        status="needs_manual_target", routing_token=picker_token, options=options)
    dispatch_token, refused = _derived_identity(client_message_id, picker_token, 0)
    original = workspace_admission.bounded_workspace_preflight
    route_changes = []

    def preflight_during_rebind_and_back(root, *args, **kwargs):
        # A genuine routing change completes while the task is being prepared.
        registry.update_project(data_dir, PROJECT, working_dir=str(tmp_path / "elsewhere"))
        registry.update_project(data_dir, PROJECT, working_dir=str(folder))
        route_changes.append(str(root))
        return original(root, *args, **kwargs)

    wait = routing_wait.wait_for_promotion_admission

    def consume_then_wait(*args, **kwargs):
        # The absent supervisor consumes the picker's actual queued event at its
        # wait boundary. Dispatch, admission and receipt reads remain real.
        from supervisor.events import dispatch_event

        while not host.bus.empty():
            event = host.bus.get_nowait()
            if event.get("type") == "promote_chat_to_task":
                assert event["objective"] == event["source_text"] == OWNER_TEXT
                assert event["source_ref"]["client_message_id"] == client_message_id
                dispatch_event(event, host.ctx)
                break
        else:
            pytest.fail("the picker did not dispatch a new task")
        return wait(*args, **kwargs)

    with monkeypatch.context() as dispatch:
        dispatch.setattr(workspace_admission, "bounded_workspace_preflight", preflight_during_rebind_and_back)
        dispatch.setattr(routing_wait, "wait_for_promotion_admission", consume_then_wait)
        status, response = handle_routing_decision(data_dir, request_id="owner-pick-1381",
            decision_id=f"routing:{client_message_id}:{picker_token}", option_index=0)
    assert status == 409 and response["error"] == "dispatch_rejected", response
    assert response["state"] == "open" and response["cause"] == REFUSAL_CAUSE
    assert route_changes, "the promotion never reached workspace preflight"
    assert not host.pending

    refused_result = load_task_result(data_dir, refused, strict=True)
    annotations = [row for row in _rows(data_dir / "logs" / "chat_annotations.jsonl")
                   if row.get("client_message_id") == client_message_id]
    notices = [row for row in dialogue_test.log_rows(data_dir / "logs" / "chat.jsonl")
               if row.get("task_id") == refused and row.get("type") == "task_not_started"]
    assert refused_result["status"] == "failed" and refused_result["admission_outcome"] == "never_admitted"
    assert refused_result["reason_code"] == "project_routing_fence_changed"
    assert not refused_result.get("started_at")
    # Bound to the room, so ordinary membership alone would make it Main-eligible.
    assert registry.project_binding_for_task(data_dir, refused)["project_id"] == PROJECT
    refusal = next(row for row in annotations if row.get("routing_token") == dispatch_token)
    assert refusal["status"] == "needs_manual_target" and refusal["cause"] == REFUSAL_CAUSE
    assert annotations[-1]["routing_token"] == picker_token and annotations[-1]["status"] == "needs_manual_target"
    assert annotations[-1]["options"] == options  # the owner's picker reopens after refusal
    assert len(notices) == 1 and notices[0]["chat_id"] == chat and REFUSAL_CAUSE in notices[0]["text"]
    return {"chat": chat, "refused": refused, "client_message_id": client_message_id, "picker_token": picker_token,
            "refusal_title": notices[0]["text"].split(" · ", 1)[0], "refused_result": refused_result,
            "legacy_result": legacy, "annotation": refusal, "notice": notices[0]}


_MAIN_FACTS = """() => [...document.querySelectorAll('#chat-messages .chat-bubble[data-system-type='
    + '"project_completion_summary"], #chat-messages .chat-bubble.project-answer')]
    .map(n => ({task: n.dataset.taskId || '', text: n.innerText,
                phase: n.querySelector('.project-ending-head .ui-status')?.innerText.trim() || '',
                voice: n.classList.contains('project-answer') ? 'answer' : 'pointer'}))"""
_PANEL_OPEN = """(pid) => {
    const host = document.getElementById('project-panel');
    const panel = document.getElementById(`panel-pchat-${pid}`);
    return !!host && !!panel && !host.hidden && !panel.hidden && host.classList.contains('open')
        && getComputedStyle(host).opacity === '1' && panel.getClientRects().length > 0;
}"""
_PANEL_FACTS = """(pid) => {
    const panel = document.getElementById(`panel-pchat-${pid}`);
    if (!panel) return null;
    const all = (sel) => [...panel.querySelectorAll(sel)];
    return {
        visible: (""" + _PANEL_OPEN + """)(pid),
        notices: all('.chat-bubble[data-system-type="task_not_started"]')
            .map(n => ({task: n.dataset.taskId || '', text: n.innerText})),
        cards: all('.chat-live-card[data-task-id]').map(c => ({task: c.dataset.taskId,
            phase: (c.querySelector('.chat-live-phase') || {}).textContent || '', text: c.innerText})),
        bubbles: all('.chat-bubble[data-task-id]').map(n => ({task: n.dataset.taskId,
            type: n.dataset.systemType || '', text: n.innerText})),
        pickers: all('.chat-routing-card').map(c => ({token: c.dataset.routingToken, state: c.dataset.state,
            options: [...c.querySelectorAll('.chat-quiz-option')].map(b => ({text: b.innerText, disabled: b.disabled}))})),
        completions: all('.chat-bubble[data-system-type="project_completion_summary"]').length,
        text: panel.innerText,
    };
}"""


def _wait_main(page, tasks, timeout):
    page.wait_for_function(f"(tasks) => {{ const rows = ({_MAIN_FACTS})(); "
                           "return tasks.every(t => rows.some(r => r.task === t)); }", arg=list(tasks),
                           timeout=timeout)
    return page.evaluate(_MAIN_FACTS)


def _open_project(page):
    """Open the room visibly; clicking the ACTIVE row would close it (toggle)."""
    row = page.locator(f'.nav-project-row[data-project-id="{PROJECT}"]')
    row.wait_for(state="visible", timeout=30_000)
    if not page.evaluate(_PANEL_OPEN, PROJECT) and "active" not in (row.get_attribute("class") or "").split():
        row.click()
    page.wait_for_function(_PANEL_OPEN, arg=PROJECT, timeout=30_000)


def _wait_panel(page, predicate: str, timeout=30_000):
    page.wait_for_function(f"(pid) => {{ const f = ({_PANEL_FACTS})(pid); return !!f && f.visible && ({predicate}); }}",
                           arg=PROJECT, timeout=timeout)
    return page.evaluate(_PANEL_FACTS, PROJECT)


def _assert_refusal_is_origin_receipt_only(main, panel, seed):
    refused = seed["refused"]
    assert panel["visible"], "the Project panel facts were not read from a visibly open panel"
    assert all(row["task"] != refused for row in main), main
    assert not any(seed["refusal_title"] in row["text"] for row in main), main
    assert [row["task"] for row in panel["notices"]] == [refused], panel["notices"]
    assert REFUSAL_CAUSE in panel["notices"][0]["text"]
    [picker] = panel["pickers"]
    assert picker["token"] == seed["picker_token"] and picker["state"] == "open"
    assert picker["options"] == [{"text": PROJECT_NAME, "disabled": False}]
    assert all(card["task"] != refused for card in panel["cards"]), panel["cards"]
    assert [row["type"] for row in panel["bubbles"] if row["task"] == refused] == ["task_not_started"]
    assert panel["completions"] == 0


def _assert_failures_stay_visible(main, tasks):
    by_task = {row["task"]: row for row in main}
    for task_id in tasks:
        row = by_task.get(task_id, {})
        # Typed endings give the outcome its own status, separate from the title.
        assert row.get("phase") == "Failed" and row.get("voice") == "pointer", (task_id, main)
        assert "Failed" in row["text"].splitlines(), (task_id, main)


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_never_admitted_project_refusal_stays_an_origin_receipt(
        direct_server_with_data, monkeypatch, tmp_path, engine):  # noqa: F811
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    from ouroboros import projects_registry as registry
    from ouroboros.task_results import load_task_result
    from ouroboros.terminal_projection import reconcile_terminal_projections
    from tests import fixtures_mock_llm

    server = direct_server_with_data
    url, data_dir = server["url"], server["data_dir"]
    evidence = producer_evidence_dir(tmp_path / "admission-consumer" / engine, "admission-consumer-" + engine)
    evidence.mkdir(parents=True, exist_ok=True)
    print(f"ADMISSION_EVIDENCE={evidence}")
    facts: dict = {"engine": engine}

    def record(name, value):
        facts[name] = value
        (evidence / "facts.json").write_text(json.dumps(facts, indent=2, default=str), encoding="utf-8")

    hold, entered = threading.Event(), threading.Event()
    fixtures_mock_llm.HOLD_RELEASE.clear()

    def completion(handler):
        payload = json.loads(handler.rfile.read(int(handler.headers.get("Content-Length", 0))))
        if hold.is_set():
            entered.set()
            fixtures_mock_llm.HOLD_RELEASE.wait(120)
        streaming = payload.get("stream") is True
        answer = {"id": "held-completion", "choices": [{"index": 0, "finish_reason": "stop",
                  "delta" if streaming else "message": {"role": "assistant", "content": "OK"}}],
                  "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        body = ("data: " + json.dumps(answer) + "\n\ndata: [DONE]\n\n" if streaming else json.dumps(answer)).encode()
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream" if streaming else "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    monkeypatch.setattr(fixtures_mock_llm._Handler, "do_POST", completion)

    server["stop_server"]()
    host = _producers(monkeypatch, data_dir, server["repo_dir"])
    seed = _seed(host, monkeypatch, data_dir, tmp_path)
    record("seed", {**seed, "bridge_frames": host.bridge.frames, "bridge_acks": host.bridge.acks,
                    "live_sends": host.live_sends})
    server["start_server"]()
    published = [ADMITTED_FAILURE, LEGACY_AT_BOOT]

    try:
        with sync_playwright() as pw:
            browser = getattr(pw, engine).launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                page.wait_for_function("() => window.__testSockets?.some(s => s.readyState === 1)", timeout=30_000)
                # The server's startup scan owes the actual and the ambiguous failure, not the refusal.
                main = _wait_main(page, published, timeout=60_000)
                _open_project(page)
                panel = _wait_panel(page, "f.notices.length > 0 && f.pickers.some(p => p.state === 'open')"
                                          f" && f.cards.some(c => c.task === '{ADMITTED_FAILURE}')")
                record("boot_publication", {"main": main, "panel": panel,
                                            "refused_after_boot": load_task_result(data_dir, seed["refused"], strict=True)})
                _assert_refusal_is_origin_receipt_only(main, panel, seed)
                _assert_failures_stay_visible(main, published)
                assert {card["task"]: card["phase"].strip() for card in panel["cards"]}[ADMITTED_FAILURE] == "Failed"
                page.screenshot(path=str(evidence / "01-after-boot-publication.png"))

                # The replacement is admitted by the served API while this process
                # holds the registry writer lock (occupancy is not a route change).
                lock_held, release_lock = threading.Event(), threading.Event()

                def hold_registry_writer():
                    with registry._file_write_lock(registry._registry_path(data_dir)):
                        lock_held.set()
                        release_lock.wait(3.0)

                holder = threading.Thread(target=hold_registry_writer)
                hold.set()
                holder.start()
                assert lock_held.wait(5)
                started = time.monotonic()
                response = page.request.post(url + "/api/tasks", data={
                    "task_id": REPLACEMENT, "project_id": PROJECT, "memory_mode": "empty",
                    "title": "Admission research (replacement)",
                    "description": "Retry the admission research as a new attempt."}, timeout=60_000)
                elapsed = time.monotonic() - started
                answered_while_held = holder.is_alive()
                release_lock.set()
                holder.join(5)
                body = response.json()
                record("replacement_admission", {"status": response.status, "body": body,
                                                 "elapsed_sec": round(elapsed, 3),
                                                 "answered_while_lock_held": answered_while_held})
                assert response.status == 200 and body["status"] == "scheduled", body
                assert entered.wait(60), "the replacement never reached its model call"
                working = []
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline and not working:
                    state = page.request.get(url + "/api/state").json()
                    working = [a for a in state.get("active_chat_activities", [])
                               if REPLACEMENT in (a.get("task_id"), a.get("activity_id"))]
                    page.wait_for_timeout(250)
                assert working, state.get("active_chat_activities")
                record("replacement_working", working)
                page.wait_for_timeout(1_000)
                page.screenshot(path=str(evidence / "02-replacement-working.png"))

                # The 600 s custody duty's pass while the replacement works: it owes a
                # newly discovered legacy-shaped refusal, never the positive refusal.
                late_legacy = _legacy_refusal(host, data_dir, LEGACY_AT_DUTY, seed["chat"],
                                              "Refused attempt discovered by the custody duty")
                settled = reconcile_terminal_projections(data_dir)
                live = [row.get("delivery_id") for row in host.live_sends if row.get("type") == "send_message"]
                refused_after = load_task_result(data_dir, seed["refused"], strict=True)
                record("custody_duty", {"settled": settled, "live_sends": live, "late_legacy": late_legacy,
                                        "refused_after": refused_after})
                # Main debt settles only on the sender's strict receipt; this live send is lost.
                assert settled == 0 and live == [f"project-completion:{LEGACY_AT_DUTY}"], (settled, live)
                assert not refused_after.get("canonical_terminal_projection_ready")
                assert not refused_after.get("canonical_terminal_projection")
                published.append(LEGACY_AT_DUTY)
                # Delivered by the serving process's own outbox replay (60 s minimum age).
                main = _wait_main(page, published, timeout=180_000)
                panel = page.evaluate(_PANEL_FACTS, PROJECT)
                record("after_custody_duty_live", {"main": main, "panel": panel})
                _assert_refusal_is_origin_receipt_only(main, panel, seed)
                _assert_failures_stay_visible(main, published)
                page.screenshot(path=str(evidence / "03-live-after-custody-duty.png"))

                # The replacement finishes; history then reloads and the server restarts.
                hold.clear()
                fixtures_mock_llm.HOLD_RELEASE.set()
                replacement = {}
                deadline = time.monotonic() + 90
                while time.monotonic() < deadline:
                    replacement = load_task_result(data_dir, REPLACEMENT, strict=True) or {}
                    if replacement.get("status") in {"completed", "failed", "cancelled"}:
                        break
                    page.wait_for_timeout(500)
                record("replacement_terminal", {key: replacement.get(key) for key in ("status", "reason_code", "result")})
                assert replacement.get("status") == "completed", replacement.get("status")
                # A final model answer precedes files/post-task settlement and
                # durable outbox publication. Observe that publication before
                # asking a reload to replay it (the replay beat alone is 60 s).
                _wait_main(page, [*published, REPLACEMENT], timeout=180_000)
                record("replacement_publication", {
                    "result": load_task_result(data_dir, REPLACEMENT, strict=True),
                    "main_history": [row for row in dialogue_test.log_rows(data_dir / "logs/chat.jsonl")
                                     if row.get("task_id") == REPLACEMENT],
                })

                for label in ("reload", "restart_reconnect", "restart_reload"):
                    if label == "restart_reconnect":
                        sockets = page.evaluate("() => window.__testSockets.length")
                        server["restart_server"]()
                        page.wait_for_function(
                            "(n) => window.__testSockets.length > n && window.__testSockets.at(-1).readyState === 1",
                            arg=sockets, timeout=60_000)
                    else:
                        page.reload(wait_until="domcontentloaded", timeout=30_000)
                        page.wait_for_function("() => window.__testSockets?.some(s => s.readyState === 1)",
                                               timeout=30_000)
                    _wait_main(page, [*published, REPLACEMENT], timeout=60_000)
                    _open_project(page)
                    panel = _wait_panel(page, f"f.notices.length > 0 && f.cards.some(c => c.task === '{ADMITTED_FAILURE}')"
                                              f" && f.bubbles.some(b => b.task === '{REPLACEMENT}')", timeout=60_000)
                    main = page.evaluate(_MAIN_FACTS)
                    record(label, {"main": main, "panel": panel})
                    _assert_refusal_is_origin_receipt_only(main, panel, seed)
                    _assert_failures_stay_visible(main, published)
                    phases = {card["task"]: card["phase"].strip() for card in panel["cards"]}
                    assert phases[ADMITTED_FAILURE] == "Failed", phases
                    assert phases.get(REPLACEMENT, "Done") == "Done", phases
                    replacement_main = [row for row in main if row["task"] == REPLACEMENT]
                    assert len(replacement_main) == 1 and "Failed" not in replacement_main[0]["text"], main
                    page.screenshot(path=str(evidence / f"04-{label}.png"))

                durable = page.request.get(url + f"/api/tasks/{seed['refused']}").json()
                record("refused_api_readback", durable)
                assert durable.get("status") == "failed" and durable.get("admission_outcome") == "never_admitted"
            except Exception:
                try:  # evidence for the parent; never masks the original failure
                    page.screenshot(path=str(evidence / "zz-failure.png"))
                    record("failure_view", {"main": page.evaluate(_MAIN_FACTS),
                                            "panel": page.evaluate(_PANEL_FACTS, PROJECT)})
                except Exception as capture_error:
                    record("failure_view_error", repr(capture_error))
                raise
            finally:
                fixtures_mock_llm.HOLD_RELEASE.set()
                browser.close()
    except PlaywrightError as exc:
        if not os.environ.get("OUROBOROS_EXPECT_BROWSER_ENGINES") and (
                "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower()):
            pytest.skip(str(exc))
        raise
    finally:
        fixtures_mock_llm.HOLD_RELEASE.set()
