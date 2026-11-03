"""#1403 in the real SPA over synthetic HTTP/WS: unrelated System rows end no turn, and
the reworded host rows render as the owner reads them, live and after reload.

Frames come from the real producers through the real ``LocalChatBridge``; reload reads
the real history endpoint over the same data root. No runtime, provider or Telegram.
"""
from __future__ import annotations

from tests import _dialogue as dialogue_test

import json
import os
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from tests import test_subscription_setup_browser as setup
from tests.test_sidebar_activity_browser import FRAMES, activity, mount_sidebar
from tests.ui_chat_viewport_smoke import wait_drawer_closed

subscription_ui = setup.subscription_ui
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
ROOM = [dict(id="p-fixture", name="Synthetic room", chat_id=42, lifecycle="active", visible_revision=0)]


def _evidence(tmp_path) -> Path:
    out = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR") or tmp_path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _host(tmp_path, monkeypatch):
    """The real bridge and send seam over a disposable data root; broadcast frames are kept."""
    from supervisor import message_bus

    (tmp_path / "logs").mkdir(exist_ok=True)
    bridge = message_bus.LocalChatBridge({})
    frames: list = []
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    monkeypatch.setattr(message_bus, "publish_event", lambda *_a, **_k: None)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 7, "owner_chat_id": 1})
    return bridge, frames


def _serve_history(page, data_root):
    """The real history endpoint over the same data root (Playwright's routes run in its loop)."""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway.history import make_chat_history_endpoint

    client = TestClient(Starlette(routes=[Route("/api/chat/history", make_chat_history_endpoint(data_root))]))

    def history(route):
        url = urlsplit(route.request.url)
        result = client.get(url.path + ("?" + url.query if url.query else ""))
        route.fulfill(status=result.status_code, content_type="application/json", body=result.content)

    page.route("**/api/chat/history**", history)


def _push(mounted, frames, start=0):
    for frame in frames[start:]:
        mounted["sockets"][-1].send(json.dumps(frame))
    return len(frames)


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_unrelated_system_rows_keep_working_and_sending(subscription_ui, tmp_path, monkeypatch, theme):
    from ouroboros.task_finalization import host_operation_reply_kwargs
    from supervisor import message_bus

    ui, out = subscription_ui, _evidence(tmp_path)
    page = ui["page"]
    _bridge, frames = _host(tmp_path, monkeypatch)
    _serve_history(page, tmp_path)
    status = page.locator("#page-chat #chat-status")

    mounted = mount_sidebar(ui, 1360, theme, False, ROOM, [activity("turn-main")])
    page.wait_for_function("() => document.querySelector('#page-chat #chat-status').textContent.startsWith('Working')")
    mounted["mode"]["fault"] = "hold"  # the notice's own effect, not a later census correction
    mounted["sockets"][-1].send(json.dumps({"type": "projects_changed"}))
    page.wait_for_function("() => sidebarHeldCount().then(n => n > 0)")
    sent = 0
    message_bus.send_with_budget(1, "🚫 Model budget reached. Unrelated runtime fact.",
                                 role="system", system_type="budget_notice")
    sent = _push(mounted, frames, sent)
    page.get_by_text("Model budget reached. Unrelated runtime fact.", exact=False).wait_for()
    page.evaluate(FRAMES)
    assert status.inner_text().startswith("Working"), "an unrelated System row ends no turn"
    page.screenshot(path=str(out / f"header-working-after-notice-{theme}.png"), animations="disabled")

    # The census answers: no turn. The owner's own sends show Sending... until answered.
    mounted["mode"]["fault"] = ""
    for route, snapshot in mounted["held"]:
        route.fulfill(json=snapshot)
    mounted["held"].clear()
    mounted["refresh"](active_chat_activities=[])
    page.wait_for_function("() => document.querySelector('#page-chat #chat-status').textContent.startsWith('Online')")
    mounted["mode"]["fault"] = "hold"
    page.locator("#chat-input").fill("/bg")
    page.locator("#chat-send").click()
    page.wait_for_function("() => document.querySelector('#page-chat #chat-status').textContent.startsWith('Sending')")
    cmid = page.locator(".chat-bubble.user[data-client-message-id]").last.get_attribute("data-client-message-id")
    assert cmid
    message_bus.send_with_budget(1, "Reminder · Ouroboros · written today · for now\nCheck the export.",
                                 role="system", system_type="reminder")
    message_bus.send_with_budget(1, "🚫 Model budget reached. Another unrelated fact.",
                                 role="system", system_type="budget_notice")
    sent = _push(mounted, frames, sent)
    page.get_by_text("Another unrelated fact.", exact=False).wait_for()
    page.evaluate(FRAMES)
    assert status.inner_text().startswith("Sending"), "nor retires a submission it does not name"
    page.screenshot(path=str(out / f"header-sending-after-notice-{theme}.png"), animations="disabled")

    # The command reply names the submission it answers: that Sending settles.
    message_bus.send_with_budget(1, "🧠 Background consciousness: off — not running.", role="system",
                                 system_type="command_reply", **host_operation_reply_kwargs(None, "", cmid))
    sent = _push(mounted, frames, sent)
    page.get_by_text("Background consciousness: off", exact=False).wait_for()
    page.evaluate(FRAMES)
    assert status.inner_text().startswith("Online"), status.inner_text()
    page.screenshot(path=str(out / f"header-answered-{theme}.png"), animations="disabled")
    stored = dialogue_test.log_rows(tmp_path / "logs" / "chat.jsonl")
    assert all(not row.get("client_message_id") for row in stored if row["direction"] == "system")
    mounted["mode"]["fault"] = ""
    for route, snapshot in mounted["held"]:
        route.fulfill(json=snapshot)
    mounted["held"].clear()
    assert ui["errors"] == []


@pytest.fixture(params=["chromium", "webkit"])
def engine_ui(request, monkeypatch):
    """``subscription_ui`` in Chromium and WebKit; the browser that actually launched is checked, not assumed."""
    monkeypatch.setenv("OUROBOROS_UI_BROWSER_ENGINE", request.param)
    if os.environ.get("OUROBOROS_UI_EVIDENCE_DIR"):  # one engine's screenshots never replace the other's
        monkeypatch.setenv("OUROBOROS_UI_EVIDENCE_DIR", str(Path(os.environ["OUROBOROS_UI_EVIDENCE_DIR"]) / request.param))
    ui = request.getfixturevalue("subscription_ui")
    assert ui["page"].context.browser.browser_type.name == request.param
    return ui


@pytest.mark.parametrize("theme,width", [("light", 1360), ("dark", 1360), ("light", 390)])
def test_reworded_host_rows_live_and_after_reload(engine_ui, tmp_path, monkeypatch, theme, width):
    import supervisor.queue as q
    from ouroboros.cancel_intents import active_intent, request_cancel
    from ouroboros.server_owner_routing import _record_routing_receipt
    from ouroboros.task_finalization import prepare_terminal_send_event
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from supervisor import events_chat_delivery, message_bus
    from supervisor.events_runtime_controls import _drive_cancel_task_event
    from supervisor.events_schedule_task import _reject_schedule_task
    from supervisor.terminal_delivery import build_unreviewed_salvage_event, deliver_miss_lane_outcome

    ui, out = engine_ui, _evidence(tmp_path)
    page = ui["page"]
    bridge, frames = _host(tmp_path, monkeypatch)
    # Each run is its own install: the process-wide delivered-id memo must not dedupe it.
    monkeypatch.setattr(events_chat_delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))
    _serve_history(page, tmp_path)
    sender = SimpleNamespace(DRIVE_ROOT=tmp_path, task_registry={}, RUNNING={},
                             send_with_budget=message_bus.send_with_budget, append_jsonl=lambda *_a: None,
                             load_state=lambda: {"owner_chat_id": 1})
    mounted = mount_sidebar(ui, width, theme, False, ROOM, [])
    if width < 700:  # the narrow layout opened the navigation drawer; choosing Main Chat closes it
        page.locator('#primary-sidebar [data-nav-page="chat"]').click()
        wait_drawer_closed(page)  # the room stays inert until its close motion ends
    for tid in ("t-custody", "t-parent", "t-stop"):
        write_task_result(tmp_path, tid, STATUS_RUNNING, chat_id=1, result="working")
        message_bus.send_with_budget(1, f"Working on {tid}.", is_progress=True, task_id=tid, narration=True)

    # 1. Delegated custody: the owner's card row lists only what holds something. The delivery reads the task's
    # current custody (the pipeline's disclosure, handed over as custody_audit), not a stored copy.
    custody = {"audit_status": "ok", "open_run_ids": [], "pending_invocation_ids": [],
               "undisposed_patch_run_ids": [], "terminal_runs": [
                   {"run_id": "run-7c1", "state": "failed", "model": "leaf-model"}]}
    write_task_result(tmp_path, "t-custody", "completed", chat_id=1, result="Report done.",
                      delegate_terminal_reconciliation=custody)
    event = prepare_terminal_send_event(
        tmp_path, {"id": "t-custody", "chat_id": 1}, "Report done.", {"terminal_origin": "model_final"},
        {"type": "send_message", "task_id": "t-custody", "chat_id": 1, "text": "Report done.",
         "progress_meta": {"task_terminal_status": "completed"}}, presence=False, custody_audit=custody)
    events_chat_delivery._handle_send_message(event, sender)
    # 2. A cancellation receipt without a saved message promises nothing below.
    write_task_result(tmp_path, "t-cancel", "cancelled", chat_id=1, result="stopped")
    events_chat_delivery._handle_send_message(build_unreviewed_salvage_event(
        tmp_path, {"chat_id": 1}, "t-cancel", outcome="stopped", settled_status="cancelled"), sender)
    # A restore-fenced cancellation names the host cause, not an apparent owner Stop.
    from supervisor import terminal_delivery

    restored = {"status": "cancelled", "chat_id": 1, "cancel_origin": {
        "source": "snapshot_restore", "reason": "server_shutdown"}}
    write_task_result(tmp_path, "t-restore", "cancelled", **{k: v for k, v in restored.items() if k != "status"})
    monkeypatch.setattr(terminal_delivery, "enqueue_terminal_delivery",
                        lambda _root, event, **_kw: events_chat_delivery._handle_send_message(event, sender))
    deliver_miss_lane_outcome(tmp_path, tmp_path, restored, "t-restore", "cancelled")
    # 3. A helper the pool refused: the parent's card reads the owner sentence.
    _reject_schedule_task(sender, tid="kid12345", chat_id=1, delegation_role="subagent", parent_id="t-parent",
                          root_task_id="t-parent", role="researcher", result_fields={},
                          detail="Subagent not scheduled: the worker pool is currently unavailable. It was NOT "
                                 "left scheduled — do the work inline yourself, or retry after /restart.",
                          fallback_message="⚠️ Task kid12345 not scheduled: worker pool unavailable.")
    # 4. A cancel_task event drives an already recorded intent; even a failed
    # teardown emits no chat row. The card/detail and durable intent own its state.
    monkeypatch.setattr(q, "DRIVE_ROOT", tmp_path)
    request_cancel(tmp_path, "t-stop", source="test")
    monkeypatch.setattr(q, "cancel_task_custody", lambda task_id, **_k: q.CANCEL_FAILED)
    before_frames = list(frames)
    before_history = dialogue_test.log_bytes(tmp_path / "logs" / "chat.jsonl")
    _drive_cancel_task_event({"task_id": "t-stop"}, sender)
    assert frames == before_frames
    assert dialogue_test.log_bytes(tmp_path / "logs" / "chat.jsonl") == before_history
    assert active_intent(tmp_path, "t-stop") is not None
    sent = _push(mounted, frames)

    texts = {
        "custody": "Confirmed delegated terminal receipts: run-7c1: failed on leaf-model.",
        "cancel": "Task t-cancel was stopped.",
        "restore": "Task t-restore was stopped. The server stopped while this task was still running.",
        "helper": "Task kid12345 not scheduled: worker pool unavailable.",
    }
    for text in texts.values():
        # A collapsed card's one line is cut on a word to fit (text_fit.js), so on a phone a long line ends
        # `worker…`; its whole sentence stays in the card's timeline. The words before the cut must still be there.
        shown = text if width >= 700 else text.split(":")[0]
        page.get_by_text(shown, exact=False).first.wait_for(state="attached")

    # 5. A message saved into a running task's mailbox: a write, not a read receipt.
    page.locator("#chat-input").fill("Add the appendix too.")
    page.locator("#chat-send").click()
    cmid = page.locator(".chat-bubble.user[data-client-message-id]").last.get_attribute("data-client-message-id")
    message_bus.log_chat("in", 1, 7, "Add the appendix too.", source="web", client_message_id=cmid,
                         drive_root=tmp_path)
    _record_routing_receipt(bridge, sender, chat_id=1, client_message_id=cmid, action="mailbox_delivery",
                            target="t-docs", target_label="Docs task", status="delivered")
    _push(mounted, frames, sent)
    page.locator(".msg-routing-annotation").filter(has_text="Message saved for task · Docs task").wait_for()

    def check(phase):
        page.evaluate(FRAMES)
        for card in page.locator(".chat-live-card").all():
            # A card opens by its own Details button (a portal has none: its room holds the details).
            disclosure = card.locator(":scope > [data-live-summary-button] [data-live-disclosure]")
            if card.get_attribute("data-expanded") != "1" and disclosure.is_visible():
                disclosure.click()
        page.evaluate(FRAMES)
        body = page.locator("#page-chat").inner_text()
        found = {key: text in body for key, text in texts.items()}
        found["mailbox"] = "Message saved for task · Docs task" in body
        for gone in ("none recorded by the current audit", "Below is the last persisted", "inline yourself",
                     "/restart", "durable cancel intent", "watchdog", "Delivered to task",
                     "Stopping task t-stop", "cancel t-stop did not settle"):
            assert gone not in body, (phase, gone)
        page.screenshot(path=str(out / f"rows-{phase}-{theme}-{width}.png"), full_page=True, animations="disabled")
        # Each reworded row as painted, in its own carrier (a card or a bubble), for human inspection.
        for key, text in {**texts, "mailbox": "Message saved for task · Docs task"}.items():
            shown = page.get_by_text(text, exact=False).first
            if shown.count():
                holder = shown.locator("xpath=ancestor-or-self::*[contains(@class,'chat-live-card') "
                                       "or contains(@class,'chat-bubble')][1]")
                holder.scroll_into_view_if_needed()
                holder.screenshot(path=str(out / f"row-{key}-{phase}-{theme}-{width}.png"), animations="disabled")
        return found

    live = check("live")
    assert all(live.values()), live
    ui["errors"].watch(page)  # the reload may cut off the page's own reads (tests/ui_console_errors.py)
    page.reload(wait_until="domcontentloaded")
    page.wait_for_function("() => window.__ouroWs?.ws?.readyState === WebSocket.OPEN")
    page.locator(".msg-routing-annotation").filter(has_text="Message saved for task").wait_for()
    replay = check("reload")
    (out / f"rows-{theme}-{width}.json").write_text(json.dumps({"live": live, "reload": replay}, indent=1))
    # The production history endpoint carries the card rows as well as the
    # standalone receipt and mailbox annotation; pin every observed live/replay pair.
    assert all(replay.values()), replay
    assert not ui["errors"].unexpected(), ui["errors"].report()


@pytest.mark.parametrize("theme,width", [("light", 1360), ("dark", 1360), ("light", 390)])
@pytest.mark.parametrize("outcome", ["pending", "write_failure"])
def test_stop_response_reaches_card_from_real_endpoints(subscription_ui, tmp_path, monkeypatch, theme, width, outcome):
    from playwright.sync_api import expect
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.cancel_intents import active_intent
    from ouroboros.gateway.tasks import api_task_cancel, api_task_get
    from ouroboros.task_results import write_task_result
    from tests.test_task_cancel_endpoint_v682 import _isolate_queue

    page = subscription_ui["page"]
    tid = "stop-consumer"
    queue, _pending = _isolate_queue(monkeypatch, tmp_path, [{"id": tid, "chat_id": 1}])
    custody_calls = []
    monkeypatch.setattr(queue, "cancel_task_custody",
                        lambda task_id, **_kw: custody_calls.append(task_id) or queue.CANCEL_FAILED)
    if outcome == "write_failure":
        def fail_write(*_a, **_kw):
            raise OSError("synthetic intent write failure")
        monkeypatch.setattr("ouroboros.cancel_intents.request_cancel", fail_write)
    write_task_result(tmp_path, tid, "running", chat_id=1, result="Checking the report.")
    _host(tmp_path, monkeypatch)
    _serve_history(page, tmp_path)
    mounted = mount_sidebar(subscription_ui, width, theme, False, ROOM, [activity(tid)])
    if width < 700:
        page.locator('#primary-sidebar [data-nav-page="chat"]').click()
        page.wait_for_function("() => !document.body.classList.contains('nav-drawer-open')")
    mounted["sockets"][-1].send(json.dumps({"type": "chat", "chat_id": 1, "role": "assistant",
        "task_id": tid, "is_progress": True, "narration": True, "cancelable": True,
        "content": "Checking the report."}))
    replies, details = [], []
    app = Starlette(routes=[Route("/api/tasks/{task_id}/cancel", api_task_cancel, methods=["POST"]),
                            Route("/api/tasks/{task_id}", api_task_get)])
    app.state.drive_root = tmp_path
    with TestClient(app) as client:
        def cancel(route):
            response = client.post(f"/api/tasks/{tid}/cancel", json=route.request.post_data_json)
            replies.append({"status": response.status_code, "body": response.json()})
            route.fulfill(status=response.status_code, content_type="application/json", body=response.content)

        def detail(route):
            response = client.get(f"/api/tasks/{tid}")
            details.append({"status": response.status_code, "body": response.json()})
            route.fulfill(status=response.status_code, content_type="application/json", body=response.content)

        page.route(f"**/api/tasks/{tid}", detail)
        page.route(f"**/api/tasks/{tid}/cancel", cancel)
        button = page.locator("#page-chat [data-cancel-run]")
        button.click()
        with page.expect_response(lambda response: urlsplit(response.url).path == f"/api/tasks/{tid}"
                                  and response.request.method == "GET") as fetched:
            page.locator('[data-task-control="stop_now"]').click()
        assert fetched.value.status == 200
        fetched.value.finished()
        page.evaluate(FRAMES)
        if outcome == "pending":
            expect(page.locator("#page-chat .chat-live-phase").filter(has_text="Cancelling…")).to_be_visible()
            expect(button).to_be_disabled()
        else:
            toast = page.locator(".toast").filter(has_text="Cancel failed:")
            expect(toast).to_be_visible()
            expect(button).to_be_enabled()
            expected = "durable cancel intent could not be recorded; nothing was cancelled — retry"
            assert replies == [{"status": 503, "body": {
                "error": expected, "task_id": tid, "reason_code": "cancel_intent_write_failed"}}]
            assert toast.inner_text() == "Cancel failed: " + expected
            assert "toast-danger" in toast.get_attribute("class") and toast.get_attribute("role") == "alert"
            box = toast.bounding_box()
            assert box and box["x"] >= 0 and box["x"] + box["width"] <= width
            assert active_intent(tmp_path, tid) is None and custody_calls == []
        page.evaluate(FRAMES)
        assert details and all(row["status"] == 200 for row in details)
        if outcome == "pending":
            intent = active_intent(tmp_path, tid)
            assert replies == [{"status": 202, "body": {
                "ok": True, "task_id": tid, "cascade": True, "cancel_state": "pending",
                "stop_policy": "immediate", "request_id": intent["request_id"],
                "reason_code": "teardown_unconfirmed"}}]
            # Cascade custody may retry its unresolved member during bounded sweeps.
            # The request stays one recorded Stop and must touch only that member.
            assert custody_calls and set(custody_calls) == {tid}
            assert details[-1]["body"]["cancel_state"] == "pending"
            assert page.locator(".toast").count() == 0
        else:
            assert details[-1]["body"].get("cancel_state") != "pending"
            assert "Cancelling…" not in page.locator("#page-chat .chat-live-phase").all_text_contents()
        out = _evidence(tmp_path)
        page.screenshot(path=str(out / f"stop-{outcome}-{theme}-{width}.png"), animations="disabled")
        (out / f"stop-{outcome}-{theme}-{width}.json").write_text(json.dumps({
            "cancel": replies, "detail": details, "custody_calls": custody_calls}, indent=2))
    assert subscription_ui["errors"] == []


def test_selected_claim_bytes_render_without_renderer_normalization(subscription_ui, tmp_path, monkeypatch):
    """#1725: the selected answer's bytes reach the painted bubble as written. Spacing the
    author wrote is neither repaired nor lost by the host or the renderer; markdown structure
    renders as structure. (Model register is a source fact the field description addresses.)"""
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from supervisor import events_chat_delivery, message_bus
    from tests.test_owner_field_affordances import CLAIMS

    ui, out = subscription_ui, _evidence(tmp_path)
    page = ui["page"]
    _bridge, frames = _host(tmp_path, monkeypatch)
    monkeypatch.setattr(events_chat_delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))
    _serve_history(page, tmp_path)
    sender = SimpleNamespace(DRIVE_ROOT=tmp_path, task_registry={}, RUNNING={},
                             send_with_budget=message_bus.send_with_budget, append_jsonl=lambda *_a: None)
    mounted = mount_sidebar(ui, 1360, "light", False, ROOM, [])
    for index, claim in enumerate(CLAIMS):
        tid = f"t-claim-{index}"
        write_task_result(tmp_path, tid, STATUS_RUNNING, chat_id=1, result="working")
        message_bus.send_with_budget(1, f"Working on {tid}.", is_progress=True, task_id=tid, narration=True)
        events_chat_delivery._handle_send_message({"type": "send_message", "chat_id": 1, "task_id": tid,
                                                   "text": claim, "format": "markdown",
                                                   "progress_meta": {"task_terminal_status": "completed"}}, sender)
    _push(mounted, frames)
    page.get_by_text("Fixed export.Ran tests;all green.Docs updated", exact=False).first.wait_for()
    page.evaluate(FRAMES)
    painted = " ".join(page.locator("#page-chat").inner_text().split())
    page.screenshot(path=str(out / "claims-rendered.png"), full_page=True, animations="disabled")
    # Every word sequence the author wrote is painted with its own spacing (HTML collapses runs).
    for claim in CLAIMS:
        for line in claim.splitlines():
            words = " ".join(line.lstrip("#- ").split())
            assert words in painted, (words, painted)
    assert "Fixed export. Ran" not in painted and "Fixedexport" not in painted
    assert ui["errors"] == []


def test_project_ingress_refusal_settles_only_its_own_sending(subscription_ui, tmp_path, monkeypatch):
    """Exercise the actual socket's refusal in the real Project UI; Main stays busy."""
    from starlette.applications import Starlette
    from starlette.routing import WebSocketRoute
    from starlette.testclient import TestClient

    from ouroboros.gateway.ws import ws_endpoint
    from supervisor import message_bus

    page = subscription_ui["page"]
    _serve_history(page, tmp_path)
    mounted = mount_sidebar(subscription_ui, 1360, "light", False, ROOM, [activity("main-other")])
    page.locator('.nav-project-row[data-project-id="p-fixture"]').click()
    prefix = "pchat-p-fixture"
    status = page.locator(f"[id='{prefix}-status']")
    status.wait_for(state="visible")
    mounted["mode"]["fault"] = "hold"
    page.locator(f"[id='{prefix}-input']").fill("Check the report.")
    page.locator(f"[id='{prefix}-send']").click()
    page.wait_for_function("id => document.getElementById(id)?.textContent.startsWith('Sending')",
                           arg=f"{prefix}-status")
    cmid = page.locator(f"[id='{prefix}-messages'] .chat-bubble.user[data-client-message-id]").last.get_attribute(
        "data-client-message-id")
    assert cmid
    monkeypatch.setattr(message_bus, "try_get_bridge", lambda: None)
    app = Starlette(routes=[WebSocketRoute("/ws", ws_endpoint)])

    def refusal(room, message_id):
        with TestClient(app) as client, client.websocket_connect("/ws") as socket:
            socket.send_json({"type": "chat", "content": "hello", "chat_id": room,
                              "client_message_id": message_id})
            return socket.receive_json()

    # Even a matching id cannot turn another room's notice into this room's answer.
    mounted["sockets"][-1].send(json.dumps(refusal(1, cmid)))
    page.evaluate(FRAMES)
    assert status.inner_text().startswith("Sending")
    mounted["sockets"][-1].send(json.dumps(refusal(42, "other-message")))
    page.evaluate(FRAMES)
    assert status.inner_text().startswith("Sending")
    mounted["sockets"][-1].send(json.dumps(refusal(42, cmid)))
    page.wait_for_function("id => document.getElementById(id)?.textContent.startsWith('Online')",
                           arg=f"{prefix}-status")
    assert page.locator("#page-chat #chat-status").inner_text().startswith("Working")
    page.screenshot(path=str(_evidence(tmp_path) / "project-refusal-own-submission.png"), animations="disabled")
    assert subscription_ui["errors"] == []


def test_initialization_notice_kept_send_controls_are_room_and_message_scoped(subscription_ui, tmp_path):
    """Two real chat instances share one socket; only the refused submission gains retry controls."""
    page = subscription_ui["page"]
    upload = "a" * 32 + "_plan.pdf"
    page.route("**/api/chat/upload", lambda route: route.fulfill(json={
        "ok": True, "filename": upload, "display_name": "plan.pdf", "mime": "application/pdf",
        "view": {"name": "plan.pdf", "kind": "file", "mime": "application/pdf", "size": 4,
                 "available": True, "url": f"/api/files/download?upload={upload}"},
    }))
    _serve_history(page, tmp_path)
    # Main accepts non-Project chat ids too; a foreign room must be a known Project.
    foreign = dict(id="p-foreign", name="Foreign room", chat_id=99, lifecycle="active", visible_revision=0)
    mounted = mount_sidebar(subscription_ui, 1360, "light", False, [*ROOM, foreign], [])
    sent = []
    mounted["sockets"][-1].on_message(lambda message: sent.append(json.loads(message)))

    def send(prefix, text):
        page.locator(f"[id='{prefix}-file-input']").set_input_files({
            "name": "plan.pdf", "mimeType": "application/pdf", "buffer": b"%PDF"})
        page.locator(f"[id='{prefix}-input']").fill(text)
        page.locator(f"[id='{prefix}-send']").click()
        bubble = page.locator(f"[id='{prefix}-messages'] .chat-bubble.user[data-client-message-id]").filter(has_text=text)
        bubble.wait_for()
        return bubble.get_attribute("data-client-message-id")

    def notice(room, cmid=""):
        mounted["sockets"][-1].send(json.dumps({"type": "chat", "role": "system",
            "system_type": "initialization_notice", "chat_id": room, "client_message_id": cmid,
            "content": "Still starting. Send again when ready."}))
        page.evaluate(FRAMES)

    def actions(cmid):
        return page.locator(f'.chat-bubble.user[data-client-message-id="{cmid}"] [data-unconfirmed-action]')

    a, b = send("chat", "First Main message"), send("chat", "Second Main message")
    page.locator('.nav-project-row[data-project-id="p-fixture"]').click()
    prefix = "pchat-p-fixture"
    page.locator(f"[id='{prefix}-status']").wait_for(state="visible")
    c, d = send(prefix, "First Project message"), send(prefix, "Second Project message")
    def counts():
        return [actions(cmid).count() for cmid in (a, b, c, d)]
    assert counts() == [0, 0, 0, 0]
    notice(99, a)
    notice(99)
    assert counts() == [0, 0, 0, 0], "foreign named and nameless notices change neither room"
    notice(42, c)
    assert counts() == [0, 0, 2, 0], "only the named Project submission gains retry controls"
    assert page.locator(f"[id='{prefix}-status']").inner_text().startswith("Sending")
    assert actions(c).all_text_contents() == ["Send again", "Discard"]
    page.screenshot(path=str(_evidence(tmp_path) / "project-named-send.png"), animations="disabled")
    original = next(frame for frame in sent if frame.get("client_message_id") == c)
    actions(c).filter(has_text="Send again").click()
    page.evaluate(FRAMES)
    assert [frame for frame in sent if frame.get("client_message_id") == c] == [original, original]
    assert counts() == [0, 0, 0, 0]
    notice(1, a)
    assert counts() == [2, 0, 0, 0]
    notice(42)
    assert counts() == [2, 0, 2, 2], "nameless notice retains room-wide doubt"
    page.screenshot(path=str(_evidence(tmp_path) / "project-kept-sends.png"), animations="disabled")
    notice(1)
    assert counts() == [2, 2, 2, 2]
    page.locator('#primary-sidebar [data-nav-page="chat"]').click()
    page.screenshot(path=str(_evidence(tmp_path) / "main-kept-sends.png"), animations="disabled")
    assert len([frame for frame in sent if frame.get("type") == "chat"]) == 5, "no automatic resend"
    assert subscription_ui["errors"] == []
