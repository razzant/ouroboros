"""Batch4 owner controls through the REAL server, worker pool and SPA (browser consumers).

Each test runs the shared ``direct_server_with_data`` fixture (a copied candidate
checkout, a disposable data root, the in-process mock model) with ONE worker and
managed roots seeded as a queued snapshot, the way a previous server generation
leaves them when the application was quit. Accepted work survives that stop
under its own id and waits for the owner's explicit Resume (owner S1, quiz
a524d73f), so each test first resumes the roots it drives. The mock model is
scripted per root, so every state below is produced by the product itself,
never written as a result:

``test_pause_restart_retention_and_resume``
  ALPHA runs with its first model call held (sent work); the owner presses
  **Pause** in the chat card's menu. Pause stops waiting for that call and saves
  ALPHA at its last ready point (owner S1, quiz 9312a119): the tree is saved,
  BRAVO starts and CHARLIE queues behind it while the provider still holds the
  call, and its late answer runs no tool. BOTH Restart buttons (chat header,
  Settings "Restart now") open the SAME confirmation; the owner confirms, and the
  real owner Restart re-executes the server. Restart returns active work and the
  runnable queue (owner S1, quiz d2f7532b): BRAVO continues from its saved state
  under its own id and CHARLIE is admitted as ordinary queued work, both to
  completion without a click, while ALPHA is STILL paused (not cancelled, not
  auto-resumed) until the owner's Resume runs it to completion. Its card says
  ``Paused · owner pause`` once saved, including after Restart.

``test_owner_restart_returns_running_work_under_its_own_id``
  The owner Restart stops a running root whose working state is saved; the
  restarted generation returns it under the SAME id (no Continue, no
  successor), it finishes, and after a reload its card still says Done.

``test_continue_after_a_crash_rejoins_a_lost_answer_and_survives_reload``
  DELTA is running when the server process dies (SIGKILL): the next boot shows
  its saved work held under its own id until the owner's Resume (owner S1:
  after a crash, save and show, continue manually). Resumed, DELTA runs again
  until its worker process dies (SIGKILL), which existing policy keeps
  terminal as a technical interruption. Continue with a LOST answer (the
  admission lands, its response is dropped) says "not confirmed"; pressing
  again answers the SAME successor; after a reload the collapsed card still
  points to it (the history projection carries the offer; nothing has to be
  opened); exactly one successor exists.

``test_the_whole_tree_pause_is_offered_only_on_a_root``
  A root runs and a child of it is queued: the Activity row of the root offers
  Pause, the child's row does not (the server would refuse a child's Pause).

The fixture's server keeps its checkout across the real Restart
(``OUROBOROS_DISABLE_MANAGED_UPDATES=1``, the stand lever of ``safe_restart``),
so the relaunched generation serves the same candidate bytes. Screenshots of
every owner-visible state go to ``OUROBOROS_UI_EVIDENCE_DIR``.
"""
from __future__ import annotations

import json
import os
import pathlib
import signal
import threading
import time

import pytest

from tests import test_ui_smoke_playwright as smoke
from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET

direct_server_with_data = smoke.direct_server_with_data

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

ALPHA, BRAVO, CHARLIE, DELTA = "b4-alpha", "b4-bravo", "b4-charlie", "b4-delta"
_TEXT = {ALPHA: "ALPHA-7Q: inventory the available tools, then report.",
         BRAVO: "BRAVO-7Q: draft the weekly summary.",
         CHARLIE: "CHARLIE-7Q: tidy the notes.",
         DELTA: "DELTA-7Q: collect the open questions."}
_RESTART_SENDS = """(() => {
    const send = WebSocket.prototype.send;
    window.__restartSends = 0;
    WebSocket.prototype.send = function(data) {
        try { if (JSON.parse(data).cmd === '/restart') window.__restartSends += 1; } catch {}
        return send.call(this, data);
    };
})();"""


@pytest.fixture(autouse=True)
def _keep_checkout_across_restart(monkeypatch):
    original = smoke.isolated_environment

    def isolated_environment(root, repo, **kwargs):
        return {**original(root, repo, **kwargs), "OUROBOROS_DISABLE_MANAGED_UPDATES": "1"}

    monkeypatch.setattr(smoke, "isolated_environment", isolated_environment)


def _user_text(message):
    content = message.get("content")
    if isinstance(content, list):
        return "\n".join(str(block.get("text") or "") for block in content if isinstance(block, dict))
    return str(content or "")


class _ScriptedModel:
    """The mock model's answers per root: a held first ALPHA call that returns a tool
    call, BRAVO's first call and every DELTA call held until teardown, and ``OK`` for
    everything else. ``returned_work`` makes BRAVO's returned attempt call a tool first."""

    def __init__(self, monkeypatch, *, returned_work=False):
        from tests import fixtures_mock_llm

        self.returned_work = returned_work
        self.alpha_sent, self.alpha_release = threading.Event(), threading.Event()
        self.bravo_sent, self.delta_sent, self.teardown = threading.Event(), threading.Event(), threading.Event()
        self.lock = threading.Lock()
        self.calls: list[tuple[str, bool]] = []
        monkeypatch.setattr(fixtures_mock_llm._Handler, "do_POST", lambda handler: self.handle(handler))

    def main_calls(self, marker):
        with self.lock:
            return self.calls.count((marker, True))

    def marker(self, payload):
        text = "\n".join(_user_text(m) for m in payload.get("messages") or []
                         if isinstance(m, dict) and m.get("role") == "user")
        if "[CONTINUE]" in text:
            return "SUCCESSOR"
        return next((name for name in ("ALPHA", "BRAVO", "CHARLIE", "DELTA") if f"{name}-7Q" in text), "OTHER")

    def handle(self, handler):
        length = int(handler.headers.get("Content-Length", 0) or 0)
        payload = json.loads(handler.rfile.read(length) or b"{}")
        marker, main = self.marker(payload), bool(payload.get("tools"))
        with self.lock:
            self.calls.append((marker, main))
            ordinal = self.calls.count((marker, True))
        message, finish = {"role": "assistant", "content": "OK"}, "stop"
        if main and marker == "ALPHA" and ordinal == 1:
            self.alpha_sent.set()
            self.alpha_release.wait(240)
            message, finish = {"role": "assistant", "content": "", "tool_calls": [{
                "id": "call_inventory", "type": "function",
                "function": {"name": "list_available_tools", "arguments": "{}"}}]}, "tool_calls"
        elif main and marker == "BRAVO" and self.returned_work and ordinal == 2:
            message, finish = {"role": "assistant", "content": "", "tool_calls": [{
                "id": "call_returned_inventory", "type": "function",
                "function": {"name": "list_available_tools", "arguments": "{}"}}]}, "tool_calls"
        elif main and (marker == "DELTA" or marker == "BRAVO" and ordinal == 1):
            (self.bravo_sent if marker == "BRAVO" else self.delta_sent).set()
            self.teardown.wait(600)
        _reply(handler, payload, message, finish)


def _reply(handler, payload, message, finish):
    streaming = payload.get("stream") is True
    if streaming and message.get("tool_calls"):
        message = {**message, "tool_calls": [{**call, "index": index}
                                             for index, call in enumerate(message["tool_calls"])]}
    choice = {"index": 0, "finish_reason": finish, ("delta" if streaming else "message"): message}
    answer = {"id": "b4-completion", "object": "chat.completion", "choices": [choice],
              "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
    body = (f"data: {json.dumps(answer)}\n\ndata: [DONE]\n\n" if streaming else json.dumps(answer)).encode()
    try:
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream" if streaming else "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)
    except OSError:
        pass  # the calling worker was stopped while this call was held


def _seed_roots(data_dir, task_ids, children=None, workspace=None):
    """Queued roots in the queue snapshot (seeded while no server runs), as this
    version's queue persists them (``admitted_dispatch: none`` — never dispatched),
    each with the owner's own chat message, its scheduled result and a progress row.
    ``children`` maps a queued child id to its root (a delegated, never-started child)."""
    import datetime as _dt

    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.task_results import write_task_result

    now = _dt.datetime.now(_dt.timezone.utc)
    (data_dir / "state").mkdir(parents=True, exist_ok=True)
    (data_dir / "logs").mkdir(parents=True, exist_ok=True)
    pending, progress, chat = [], [], []
    for seq, task_id in enumerate(task_ids, start=1):
        text = _TEXT[task_id]
        title = task_id.replace("b4-", "Batch4 ").title()
        queued_at = (now + _dt.timedelta(seconds=seq)).isoformat()
        # The owner's message, with the identity the chat ingress mints for it.
        ref = build_owner_message_ref(chat_id=1, client_message_id=f"seed-{task_id}", ts=queued_at, text=text)
        chat.append({"ts": queued_at, "direction": "in", "chat_id": 1, "text": text,
                     "client_message_id": f"seed-{task_id}"})
        task = {"id": task_id, "type": "task", "chat_id": 1, "priority": 0, "text": text,
                "description": text, "objective": text, "title": title, "root_task_id": task_id,
                "delegation_role": "root", "origin_message_text": text, "origin_message_ref": ref,
                "admitted_dispatch": "none"}
        folder = {"workspace_root": str(workspace), "workspace_mode": "external"} if workspace else {}
        task.update(folder)
        write_task_result(data_dir, task_id, "scheduled", result="Task is queued.", description=text,
                          objective=text, chat_id=1, title=title, delegation_role="root", root_task_id=task_id,
                          origin_message_text=text, origin_message_ref=ref, **folder)
        pending.append({"id": task_id, "type": "task", "priority": 0, "attempt": 1, "queued_at": queued_at,
                        "queue_seq": seq, "task": task})
        progress.append({"ts": queued_at, "chat_id": 1, "task_id": task_id,
                         "content": f"Working on {title}", "cancelable": True})
    for seq, (child_id, root_id) in enumerate((children or {}).items(), start=len(pending) + 1):
        queued_at = (now + _dt.timedelta(seconds=seq)).isoformat()
        child = {"id": child_id, "type": "task", "chat_id": 1, "priority": 0, "text": f"{child_id}: helper step",
                 "description": "helper step", "title": "Helper step", "root_task_id": root_id,
                 "parent_task_id": root_id, "delegation_role": "subagent", "admitted_dispatch": "none"}
        write_task_result(data_dir, child_id, "scheduled", result="Task is queued.", description="helper step",
                          chat_id=1, title="Helper step", delegation_role="subagent", root_task_id=root_id,
                          parent_task_id=root_id)
        pending.append({"id": child_id, "type": "task", "priority": 0, "attempt": 1, "queued_at": queued_at,
                        "queue_seq": seq, "task": child})
    (data_dir / "state" / "queue_snapshot.json").write_text(json.dumps({
        "_schema_version": 1, "ts": now.isoformat(), "reason": "b4_browser_seed",
        "pending_count": len(pending), "running_count": 0, "reaping_count": 0,
        "acceptance_fences": [], "budget_root_fences": [], "pending": pending, "running": [],
    }), encoding="utf-8")
    (data_dir / "logs" / "chat.jsonl").write_text("".join(json.dumps(row) + "\n" for row in chat), encoding="utf-8")
    (data_dir / "logs" / "progress.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in progress), encoding="utf-8")


def _http(url, method, path):
    import urllib.request

    request = urllib.request.Request(url + path, method=method, data=b"{}" if method == "POST" else None,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read() or b"{}")


def _resume_after_app_stop(url, *task_ids):
    """The seeded snapshot is what a Quit leaves: accepted roots keep their ids under
    ``saved_work_hold`` at any age, and nothing starts until the owner's explicit
    Resume (owner S1, quiz a524d73f). Each Resume admits its row as ordinary queued
    work, in the order given."""
    for task_id in task_ids:
        _wait(lambda: _http(url, "GET", f"/api/tasks/{task_id}").get("reason_code") == "saved_work_hold",
              30, f"{task_id} held after the application stop")
        answer = _http(url, "POST", f"/api/tasks/{task_id}/resume")
        assert answer.get("ok"), answer


def _get(page, url, path):
    response = page.request.get(url + path)
    assert response.ok, f"{path}: {response.status} {response.text()[:400]}"
    return response.json()


def _wait(predicate, timeout, what, interval=0.5):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            last = predicate()
        except Exception as exc:  # a server between generations answers nothing
            last = exc
        else:
            if last:
                return last
        time.sleep(interval)
    raise AssertionError(f"timed out waiting for {what}; last={last!r}")


def _phase(page, url, task_id):
    state = _get(page, url, "/api/state")
    row = next((row for row in state.get("active_chat_activities") or [] if row.get("activity_id") == task_id), {})
    return str(row.get("phase") or "")


def _task(page, url, task_id):
    return _get(page, url, f"/api/tasks/{task_id}")


def _events(data_dir, kind):
    path = data_dir / "logs" / "events.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] \
        if path.exists() else []
    return [row for row in rows if row.get("type") == kind]


def _tool_names(data_dir, task_id):
    path = data_dir / "logs" / "tools.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] \
        if path.exists() else []
    return [str(row.get("tool") or "") for row in rows if row.get("task_id") == task_id]


def _fence(data_dir, task_id):
    from ouroboros.owner_pause import read_fence

    return read_fence(data_dir, task_id)


def _card(page, task_id):
    return page.locator(f'.chat-live-card[data-task-id="{task_id}"]')


def _chip(page, task_id):
    return _card(page, task_id).locator("[data-live-phase]").first


def _menu_action(page, trigger, action, *, attempts=4):
    """Open the shared control menu, read its actions atomically and choose ``action``.

    The menu is a fixed overlay that closes when the chat scrolls (a late system
    row auto-scrolls it), so a menu closed under the click is reopened."""
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

    trigger.wait_for(state="visible", timeout=30_000)
    for attempt in range(attempts):
        trigger.scroll_into_view_if_needed()
        trigger.click()
        menu = page.locator("body > .task-control-menu")
        try:
            menu.wait_for(state="visible", timeout=5_000)
            actions = menu.evaluate(
                "node => [...node.querySelectorAll('.task-control-item')].map(item => item.dataset.taskControl)")
            menu.locator(f'[data-task-control="{action}"]').click(timeout=5_000)
            return actions
        except PlaywrightTimeoutError:
            if attempt == attempts - 1:
                raise
            page.keyboard.press("Escape")
    raise AssertionError("unreachable")


def _menu_actions(page, trigger, screenshot):
    """Open the shared menu, read its offered actions, capture it and dismiss it (no action runs)."""
    trigger.wait_for(state="visible", timeout=30_000)
    trigger.scroll_into_view_if_needed()
    trigger.click()
    menu = page.locator("body > .task-control-menu")
    menu.wait_for(state="visible", timeout=10_000)
    actions = menu.evaluate(
        "node => [...node.querySelectorAll('.task-control-item')].map(item => item.dataset.taskControl)")
    page.screenshot(path=str(screenshot))
    page.keyboard.press("Escape")
    menu.wait_for(state="detached", timeout=10_000)
    return actions


def _restart_dialog(page, opener):
    opener.click()
    dialog = page.locator(".confirm-dialog")
    dialog.wait_for(state="visible", timeout=10_000)
    return (dialog, dialog.locator("#confirm-dialog-title").inner_text(),
            dialog.locator(".marketplace-modal-body p").inner_text())


def _open_chat(page, url):
    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    page.wait_for_selector("#page-chat", timeout=30_000)
    page.wait_for_function("() => window.__testSockets?.some(s => s.readyState === 1)", timeout=30_000)


def _open_activity(page):
    page.locator('[data-nav-page="dashboard"]').click()
    page.locator('[data-dashboard-tab="activity"]').click()
    return page.locator("#dashboard-panel-activity")


def _evidence_dir(data_dir, name):
    root = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(data_dir.parent)))
    path = root / "batch4" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _owner_restart(page, url, data_dir):
    """Confirm the header Restart and wait for the re-executed generation to serve."""
    boots = len(_events(data_dir, "startup_verification"))
    page.locator(".confirm-dialog [data-confirm-ok]").click()
    page.wait_for_function("window.__restartSends === 1", timeout=10_000)
    _wait(lambda: len(_events(data_dir, "startup_verification")) > boots, 120, "the restarted generation")
    smoke._wait_health(url, 60)
    smoke._wait_supervisor_ready(url, 60)
    page.wait_for_function("() => window.__testSockets?.some(s => s.readyState === 1)", timeout=60_000)


def _launch(pw):
    browser = pw.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    errors: list[str] = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
    page.add_init_script(_RESTART_SENDS)
    return browser, page, errors


def test_pause_restart_retention_and_resume(direct_server_with_data, monkeypatch):
    from playwright.sync_api import expect, sync_playwright

    server = direct_server_with_data
    url, data_dir = server["url"], server["data_dir"]
    evidence = _evidence_dir(data_dir, "pause-restart-resume")
    model = _ScriptedModel(monkeypatch)
    server["stop_server"]()
    _seed_roots(data_dir, (ALPHA, BRAVO, CHARLIE))
    server["start_server"]()
    _resume_after_app_stop(url, ALPHA, BRAVO, CHARLIE)
    record: dict = {}
    try:
        with sync_playwright() as pw:
            browser, page, errors = _launch(pw)

            def shot(name):
                page.screenshot(path=str(evidence / f"{name}.png"))

            try:
                # 1) ALPHA genuinely runs, its first model call in flight; Pause from its card.
                assert model.alpha_sent.wait(90), "ALPHA never reached its first model call"
                _open_chat(page, url)
                _wait(lambda: _phase(page, url, ALPHA) == "working", 30, "ALPHA working")
                actions = _menu_action(page, _card(page, ALPHA).locator("[data-cancel-run]"), "pause")
                assert actions == ["finalize", "hurry", "pause", "stop_now"]
                toast = page.locator(".toast").last
                expect(toast).to_contain_text("Pausing", timeout=15_000)
                # The toast never promises that work already sent finishes (owner S1, quiz 9312a119).
                expect(toast).to_contain_text("running work stops at its last saved point")
                expect(toast).not_to_contain_text("already sent")
                shot("00-pause-toast")
                # Owner S1 (quiz 9312a119): Pause stops waiting for the sent model call and
                # saves ALPHA at its last ready point while the provider still holds that call.
                _wait(lambda: _fence(data_dir, ALPHA).get("state") == "paused", 60, "ALPHA's tree saved")
                assert not model.alpha_release.is_set()
                assert model.bravo_sent.wait(60), "BRAVO never started after ALPHA parked"
                _wait(lambda: _phase(page, url, ALPHA) == "budget_paused", 30, "ALPHA paused")
                assert _phase(page, url, CHARLIE) == "queued"
                expect(_chip(page, ALPHA)).to_have_text("Paused · owner pause", timeout=30_000)
                expect(_card(page, ALPHA).locator("[data-resume-run]")).to_be_visible()
                expect(_card(page, BRAVO).locator("[data-live-phase]")).to_have_text("Working", timeout=30_000)
                shot("01-paused-while-its-call-was-sent")

                # 2) The abandoned call answers late: its tool call never runs and ALPHA stays saved.
                model.alpha_release.set()
                time.sleep(5)  # room for an unwanted late tool run or settlement
                assert _fence(data_dir, ALPHA)["state"] == "paused" and _phase(page, url, ALPHA) == "budget_paused"
                assert _task(page, url, ALPHA)["status"] == "scheduled"
                assert "list_available_tools" not in _tool_names(data_dir, ALPHA)

                # 3) The ONE shared Restart confirmation, from both surfaces.
                dialog, title, body = _restart_dialog(page, page.locator('[data-chat-command="restart"]'))
                assert title == "Restart agent"
                assert "Tasks already paused stay paused" in body
                assert "Runnable queued tasks return to the queue" in body
                assert "still pausing" not in body and "could not be read" not in body
                shot("02-header-restart-confirm")
                dialog.locator(".marketplace-modal-actions [data-confirm-cancel]").click()
                expect(dialog).to_be_hidden()
                record["header_body"] = body
                page.locator('[data-nav-page="settings"]').click()
                page.locator("#s-workers").evaluate(
                    "(node) => {node.value='2'; node.dispatchEvent(new Event('input',{bubbles:true}));}")
                page.locator("#btn-save-settings").click()
                expect(page.locator("#btn-restart-now")).to_be_visible(timeout=30_000)
                dialog, s_title, s_body = _restart_dialog(page, page.locator("#btn-restart-now"))
                assert (s_title, s_body) == (title, body), "header and Settings must open the SAME confirmation"
                shot("03-settings-restart-confirm")
                dialog.locator(".marketplace-modal-actions [data-confirm-cancel]").click()
                expect(dialog).to_be_hidden()
                assert page.evaluate("window.__restartSends") == 0, "a cancelled confirmation sent /restart"
                page.locator('[data-nav-page="chat"]').click()
                _restart_dialog(page, page.locator('[data-chat-command="restart"]'))

                # 4) The real owner Restart; the page stays open across the new generation.
                # Restart returns active work and the runnable queue (owner S1, quiz
                # d2f7532b): BRAVO continues from its saved state under its own id and
                # CHARLIE is admitted as ordinary queued work; the owner's Pause still
                # holds ALPHA (a Restart grants no execution authority).
                _owner_restart(page, url, data_dir)
                bravo, charlie = (_wait(lambda t=t: (lambda row: row if row["status"] == "completed" else None)(
                    _task(page, url, t)), 120, f"{t} completed after the Restart") for t in (BRAVO, CHARLIE))
                time.sleep(8)  # room for an unwanted automatic resume to dispatch on the idle worker
                alpha = _task(page, url, ALPHA)
                record["after_restart"] = {name: {
                    key: t.get(key) for key in ("status", "reason_code", "cancel_origin", "continuation_offer",
                                                "root_task_id", "continued_by")}
                    for name, t in (("alpha", alpha), ("bravo", bravo), ("charlie", charlie))}
                assert alpha["status"] == "scheduled" and alpha["reason_code"] == "owner_paused", alpha["status"]
                assert _fence(data_dir, ALPHA)["state"] == "paused"
                assert _phase(page, url, ALPHA) == "budget_paused"
                assert model.main_calls("ALPHA") == 1, "a paused root was resumed without the owner"
                assert bravo["root_task_id"] == BRAVO and not bravo.get("continued_by")
                assert model.main_calls("BRAVO") == 2 and model.main_calls("SUCCESSOR") == 0, \
                    "BRAVO returns under its own id, never as a Continue successor"
                assert model.main_calls("CHARLIE") == 1
                assert not _events(data_dir, "owner_continue_admitted")
                expect(_chip(page, ALPHA)).to_have_text("Paused · owner pause", timeout=60_000)
                expect(_card(page, ALPHA).locator("[data-resume-run]")).to_be_visible()
                for task_id in (BRAVO, CHARLIE):
                    expect(_chip(page, task_id)).to_have_text("Done", timeout=60_000)
                expect(page.locator("#chat-status")).to_have_text("Paused · owner pause", timeout=30_000)
                shot("06-after-restart-chat")
                panel = _open_activity(page)
                expect(panel.locator(".activity-row", has_text="Batch4 Alpha")).to_contain_text("paused", timeout=30_000)
                expect(panel).not_to_contain_text("held after Restart")
                shot("07-after-restart-activity")

                # 5) The owner's Resume from ALPHA's chat card.
                page.locator('[data-nav-page="chat"]').click()
                shot("08-paused-card-before-resume")
                resume = _card(page, ALPHA).locator("[data-resume-run]")
                expect(resume).to_be_visible()
                actions = _menu_actions(page, _card(page, ALPHA).locator("[data-cancel-run]"),
                                        evidence / "08-paused-card-menu.png")
                assert actions == ["stop_now"], "Resume is direct on the card, not duplicated in its menu"
                resume.click()
                expect(page.locator(".toast").last).to_contain_text("Resuming", timeout=15_000)
                _wait(lambda: _task(page, url, ALPHA)["status"] == "completed", 120, "ALPHA completed")
                assert model.main_calls("ALPHA") >= 2
                page.locator('[data-nav-page="chat"]').click()
                expect(_card(page, ALPHA).locator("[data-live-phase]")).to_have_text("Done", timeout=60_000)
                shot("09-resumed-completed")
                assert errors == []
            except Exception:
                shot("failure")
                raise
            finally:
                record["page_errors"] = list(errors)
                browser.close()
    finally:
        model.alpha_release.set()
        model.teardown.set()
        record["model_calls"] = list(model.calls)
        (evidence / "record.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")


@pytest.mark.parametrize("returned_work", [False, True], ids=["text-only", "tool-work"])
def test_owner_restart_returns_running_work_under_its_own_id(direct_server_with_data, monkeypatch, returned_work):
    """Owner S1 (quiz d2f7532b): the owner's Restart returns active work. A running root
    with saved working state continues under its OWN id in the next generation,
    in its own folder and chat: no Continue is offered and no successor exists."""
    from playwright.sync_api import expect, sync_playwright

    server = direct_server_with_data
    url, data_dir = server["url"], server["data_dir"]
    evidence = _evidence_dir(data_dir, "restart-returns-running-" + ("work" if returned_work else "text"))
    model = _ScriptedModel(monkeypatch, returned_work=returned_work)
    server["stop_server"]()
    workspace = data_dir.parent / "returned-work"
    workspace.mkdir()
    _seed_roots(data_dir, (BRAVO,), workspace=workspace)
    server["start_server"]()
    _resume_after_app_stop(url, BRAVO)
    record: dict = {}
    try:
        with sync_playwright() as pw:
            browser, page, errors = _launch(pw)

            def shot(name):
                page.screenshot(path=str(evidence / f"{name}.png"))

            try:
                assert model.bravo_sent.wait(90), "BRAVO never reached its model call"
                _open_chat(page, url)
                _wait(lambda: _phase(page, url, BRAVO) == "working", 30, "BRAVO working")
                before = _task(page, url, BRAVO)
                _restart_dialog(page, page.locator('[data-chat-command="restart"]'))
                _owner_restart(page, url, data_dir)
                done = _wait(lambda: (lambda t: t if t["status"] in {"completed", "failed", "cancelled"} else None)(
                    _task(page, url, BRAVO)), 180, "BRAVO settled after the Restart")
                record["bravo"] = {key: done.get(key) for key in (
                    "status", "reason_code", "cancel_origin", "continuation_offer", "continued_by", "chat_id",
                    "root_task_id", "workspace_root", "workspace_mode")}
                assert done["status"] == "completed" and done["root_task_id"] == BRAVO, record["bravo"]
                assert not done.get("cancel_origin") and not done.get("continued_by")
                assert (done.get("continuation_offer") or {}).get("eligible") is not True
                assert model.main_calls("BRAVO") >= 2 and model.main_calls("SUCCESSOR") == 0, model.calls
                assert not _events(data_dir, "owner_continue_admitted")
                # The returned attempt does its own work: its tool runs under BRAVO's id.
                assert ("list_available_tools" in _tool_names(data_dir, BRAVO)) is returned_work
                assert done["chat_id"] == before["chat_id"] == 1
                assert done["workspace_root"] == before["workspace_root"] == str(workspace)
                assert done["workspace_mode"] == before["workspace_mode"] == "external"
                expect(_chip(page, BRAVO)).to_have_text("Done", timeout=60_000)
                expect(_card(page, BRAVO).locator("[data-continue-task]")).to_have_count(0)
                expect(page.locator(".chat-bubble").get_by_text("OK", exact=True).last).to_be_visible(timeout=60_000)
                expect(page.locator("#chat-status")).not_to_contain_text("Working", timeout=60_000)
                shot("01-returned-and-completed")

                page.reload(wait_until="domcontentloaded")
                page.wait_for_selector("#page-chat", timeout=30_000)
                expect(_chip(page, BRAVO)).to_have_text("Done", timeout=30_000)
                expect(_card(page, BRAVO).locator("[data-continue-task]")).to_have_count(0)
                expect(page.locator(".chat-bubble").get_by_text("OK", exact=True).last).to_be_visible()
                shot("02-done-after-reload")
                assert errors == []
            except Exception:
                shot("failure")
                raise
            finally:
                record["page_errors"] = list(errors)
                browser.close()
    finally:
        model.teardown.set()
        record["model_calls"] = list(model.calls)
        (evidence / "record.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")


def test_continue_after_a_crash_rejoins_a_lost_answer_and_survives_reload(direct_server_with_data, monkeypatch):
    from playwright.sync_api import expect, sync_playwright

    from ouroboros.server_process import read_service_bindings

    server = direct_server_with_data
    url, data_dir = server["url"], server["data_dir"]
    evidence = _evidence_dir(data_dir, "continue-after-crash")
    model = _ScriptedModel(monkeypatch)
    server["stop_server"]()
    _seed_roots(data_dir, (DELTA,))
    server["start_server"]()
    _resume_after_app_stop(url, DELTA)
    record: dict = {}
    try:
        assert model.delta_sent.wait(90), "DELTA never reached its model call"
        # The server process dies with DELTA running (a crash, not a stop door).
        os.kill(int(read_service_bindings(data_dir)["main"]["pid"]), signal.SIGKILL)
        server["stop_server"]()  # reaps the orphaned worker tree
        server["start_server"]()
        with sync_playwright() as pw:
            browser, page, errors = _launch(pw)

            def shot(name):
                page.screenshot(path=str(evidence / f"{name}.png"))

            try:
                _open_chat(page, url)
                # Owner S1: after an application crash the saved work is shown and
                # waits under its own id for the owner's Resume; nothing continues by itself.
                held = _wait(lambda: (lambda t: t if t.get("reason_code") == "saved_work_hold" else None)(
                    _task(page, url, DELTA)), 60, "DELTA's saved work held after the crash")
                record["held"] = {key: held.get(key) for key in ("status", "reason_code", "cancel_origin",
                                                                 "continuation_offer")}
                assert held["status"] == "scheduled" and not held.get("cancel_origin")
                card = _card(page, DELTA)
                expect(_chip(page, DELTA)).to_contain_text("Paused", timeout=60_000)
                expect(card.locator("[data-continue-task]")).to_have_count(0)
                resume = card.locator("[data-resume-run]")
                expect(resume).to_be_visible()
                assert model.main_calls("DELTA") == 1, "an application crash never continues work by itself"
                shot("01-held-after-crash")
                # Activity names the same hold as held after the stop, never as a money pause.
                held_row = _open_activity(page).locator(".activity-row", has_text="Batch4 Delta")
                expect(held_row).to_contain_text("held after Restart", timeout=30_000)
                expect(held_row).not_to_contain_text("budget")
                shot("01b-held-after-crash-activity")
                page.locator('[data-nav-page="chat"]').click()
                expect(resume).to_be_visible(timeout=30_000)
                resume.click()
                expect(page.locator(".toast").last).to_contain_text("Resuming", timeout=15_000)
                _wait(lambda: model.main_calls("DELTA") == 2, 90, "DELTA resumed under its own id")
                assert _task(page, url, DELTA)["status"] == "running"

                # Its worker process dies (SIGKILL) while the application stays up: the
                # existing policy keeps a signal death terminal, so Continue is its manual path.
                workers = json.loads((data_dir / "state" / "worker_pids.json").read_text(encoding="utf-8"))["workers"]
                assert len(workers) == 1, workers
                os.kill(int(workers[0]["pid"]), signal.SIGKILL)
                delta = _wait(lambda: (lambda t: t if t["status"] == "failed" else None)(_task(page, url, DELTA)),
                              120, "DELTA settled by its worker's death")
                record["delta"] = {key: delta.get(key) for key in ("status", "reason_code", "cancel_origin",
                                                                   "continuation_offer")}
                assert delta["reason_code"] == "worker_crash_signal", record["delta"]
                assert (delta.get("continuation_offer") or {}).get("eligible") is True
                # Opened afresh, as after a reboot: the settled card is rebuilt from its
                # history rows, which carry the offer.
                page.reload(wait_until="domcontentloaded")
                page.wait_for_selector("#page-chat", timeout=30_000)
                card = _card(page, DELTA)
                button = card.locator("[data-continue-task]")
                expect(button).to_have_text("Continue", timeout=60_000)
                shot("02-continue-offered")

                dropped: list[int] = []

                def lose_answer(route):
                    dropped.append(route.fetch().status)  # the admission lands server-side...
                    route.abort()                          # ...and its answer never arrives

                page.route(f"**/api/tasks/{DELTA}/continue", lose_answer)
                button.click()
                expect(page.locator(".toast").last).to_contain_text("Continue not confirmed", timeout=15_000)
                expect(button).to_be_enabled()
                assert dropped == [200]
                page.unroute(f"**/api/tasks/{DELTA}/continue")
                offer = _task(page, url, DELTA).get("continuation_offer") or {}
                successor = offer.get("successor_task_id")
                assert successor and offer.get("refusal") == "already_continued", offer
                shot("03-continue-answer-lost")
                button.click()
                expect(button).to_have_text("Continued", timeout=15_000)
                expect(button).to_have_attribute("data-continue-successor", successor)
                expect(page.locator(".toast").last).to_contain_text("Continue accepted.")
                nonce = page.evaluate(f"localStorage.getItem('ouro_continue_nonce:{DELTA}')")
                assert nonce
                shot("04-continue-retry-same-successor")

                page.reload(wait_until="domcontentloaded")
                page.wait_for_selector("#page-chat", timeout=30_000)
                card = _card(page, DELTA)
                card.wait_for(state="attached", timeout=30_000)
                button = card.locator("[data-continue-task]")
                # The replayed history rows carry the offer: the collapsed card
                # points to its successor without being opened.
                expect(button).to_have_text("Continued", timeout=30_000)
                expect(button).to_have_attribute("data-continue-successor", successor)
                expect(button).to_be_disabled()
                assert card.locator(":scope > [data-live-summary-button]").get_attribute("aria-expanded") != "true"
                assert page.evaluate(f"localStorage.getItem('ouro_continue_nonce:{DELTA}')") == nonce
                shot("05-continue-after-reload")
                admitted = [row["task_id"] for row in _events(data_dir, "owner_continue_admitted")
                            if row.get("predecessor_task_id") == DELTA]
                assert admitted == [successor], admitted
                succ = _task(page, url, successor)
                record["successor"] = {key: succ.get(key) for key in ("status", "chat_id", "root_task_id")}
                assert succ["chat_id"] == 1 and succ["root_task_id"] == successor
                assert errors == []
            except Exception:
                shot("failure")
                raise
            finally:
                record["page_errors"] = list(errors)
                browser.close()
    finally:
        model.teardown.set()
        record["model_calls"] = list(model.calls)
        (evidence / "record.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")


def test_the_whole_tree_pause_is_offered_only_on_a_root(direct_server_with_data, monkeypatch):
    """F14: the shared menu offered Pause on a child's Activity row, which the server
    refuses (``not_a_root_task``). The root's row and card keep it; a child's row
    offers its other actions but not a Pause that would not name its own tree."""
    from playwright.sync_api import sync_playwright

    server = direct_server_with_data
    url, data_dir = server["url"], server["data_dir"]
    evidence = _evidence_dir(data_dir, "pause-root-only")
    model = _ScriptedModel(monkeypatch)
    child = "b4-delta-child"
    server["stop_server"]()
    _seed_roots(data_dir, (DELTA,), children={child: DELTA})
    server["start_server"]()
    _resume_after_app_stop(url, DELTA, child)
    record: dict = {}
    try:
        with sync_playwright() as pw:
            browser, page, errors = _launch(pw)
            try:
                assert model.delta_sent.wait(90), "DELTA never reached its model call"
                _open_chat(page, url)
                _wait(lambda: _phase(page, url, DELTA) == "working", 30, "DELTA working")
                record["card_actions"] = _menu_actions(page, _card(page, DELTA).locator("[data-cancel-run]"),
                                                       evidence / "01-root-card-menu.png")
                assert record["card_actions"] == ["finalize", "hurry", "pause", "stop_now"]
                panel = _open_activity(page)
                root_row = panel.locator(f'[data-act="task-control"][data-id="{DELTA}"]')
                child_row = panel.locator(f'[data-act="task-control"][data-id="{child}"]')
                child_row.wait_for(state="visible", timeout=30_000)
                assert root_row.get_attribute("data-root") == "1" and child_row.get_attribute("data-root") is None
                record["child_actions"] = _menu_actions(page, child_row, evidence / "02-child-row-menu.png")
                assert record["child_actions"] == ["finalize", "hurry", "stop_now"]
                record["root_actions"] = _menu_actions(page, root_row, evidence / "03-root-row-menu.png")
                assert record["root_actions"] == ["finalize", "hurry", "pause", "stop_now"]
                # The server agrees: a child's Pause is refused and nothing is fenced.
                refused = page.request.post(f"{url}/api/tasks/{child}/pause", data={"request_id": "child-pause-1"})
                record["child_pause_http"] = [refused.status, refused.json()]
                assert refused.status == 409 and refused.json().get("reason_code") == "not_a_root_task"
                assert not _fence(data_dir, DELTA)
                assert errors == []
            finally:
                record["page_errors"] = list(errors)
                browser.close()
    finally:
        model.teardown.set()
        record["model_calls"] = list(model.calls)
        (evidence / "record.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
