"""Real Chat consumer over actual queue/census gateway facts from an inert host.

The SPA is served by the isolated candidate server. Only state/task transport is
substituted with production gateway responses from a separately rooted inert
supervisor, so receipt loss and exact same-ID recovery are under test control.
No provider is called. Live scheduled/terminal frames use the normal WS consumer.
"""
import copy
import json
import os
import time
from pathlib import Path

import pytest

from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401
from tests.test_project_admission_refusal_browser import _producers
from tests.test_project_authority_outages import _uncertain_child
from tests.test_project_hold_recovery import worker
from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET, _emit_ws_frame

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
WAIT = "Waiting for Project verification"
CHILD = '.chat-live-card[data-task-id="held"]'


@pytest.mark.parametrize("engine,ending", [("chromium", "recovery"), ("webkit", "recovery"), ("chromium", "cancel")])
def test_nested_child_wait_and_same_id_recovery(direct_server_with_data, tmp_path, monkeypatch, engine, ending):  # noqa: F811
    from playwright.sync_api import sync_playwright
    from ouroboros import projects_registry as registry
    from ouroboros.gateway.state import _chat_activities_snapshot_safe
    from ouroboros.gateway.tasks import _tasks_list_payload
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.utils import append_jsonl
    from supervisor import queue, workers

    server = direct_server_with_data
    root = tmp_path / "producer-data"
    root.mkdir()
    host = _producers(monkeypatch, root, server["repo_dir"])
    host.root = root
    registry.create_project(root, "target", name="Synthetic Project")
    child, original = _uncertain_child(host, monkeypatch)
    # Start from the accepted scheduled receipt, then lose it while the card is open.
    result_path = root / "task_results" / "held.json"
    result_path.write_bytes(original)
    queue.RUNNING["parent"] = {"task": {"id": "parent", "chat_id": 1}, "started_at": time.time(), "attempt": 1}
    sent = worker(host, monkeypatch)
    assert queue.persist_queue_snapshot()
    snapshots, request_count, errors = [], [], []
    evidence = Path(os.environ.get("OUROBOROS_TEST_TEMP_ROOT") or tmp_path) / "child-hold" / f"{engine}-{ending}"
    evidence.mkdir(parents=True, exist_ok=True)
    print(f"CHILD_HOLD_EVIDENCE={evidence}")

    def task_transport(route):
        request_count.append(route.request.url)
        if "queue_only=1" in route.request.url:
            body = _tasks_list_payload(root, None, 10, True)
            snapshots.append(copy.deepcopy(body))
            route.fulfill(status=200, json=body)
        else:
            tid = route.request.url.rsplit("/", 1)[-1]
            try:
                row = load_task_result(root, tid, strict=True)
            except ValueError:
                route.fulfill(status=503, json={"error": "result unavailable"})
                return
            route.fulfill(status=200 if row else 404, json=row or {"error": "not found"})

    def state_transport(route):
        response = route.fetch()
        body = response.json()
        body.update(active_chat_activities=_chat_activities_snapshot_safe(root), active_chat_activities_complete=True,
                    supervisor_ready=True, task_bindings={})
        route.fulfill(response=response, json=body)

    def frame(page, tid, **fields):
        row = {"type": "chat", "role": "assistant", "is_progress": True,
               "chat_id": 1, "task_id": tid, "ts": "2026-09-30T10:00:00Z", **fields}
        append_jsonl(server["data_dir"] / "logs" / "progress.jsonl", row)
        _emit_ws_frame(page, row)

    def shot(page, name):
        page.locator(CHILD).scroll_into_view_if_needed()
        page.screenshot(path=str(evidence / f"{name}.png"), animations="disabled")

    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 900}, reduced_motion="reduce")
            page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.route("**/api/tasks*", task_transport)
            page.route("**/api/tasks/**", task_transport)
            page.route("**/api/state", state_transport)
            page.goto(server["url"], wait_until="domcontentloaded")
            page.wait_for_function("() => window.__testSockets?.some(s => s.readyState === 1)")
            page.wait_for_selector('#chat-messages[data-history-hydrated="true"]')
            frame(page, "parent", content="Inspecting the Project with a child researcher.")
            frame(page, "held", content="Child research scheduled", delegation_role="subagent", subagent_event="scheduled",
                  subagent_task_id="held", parent_task_id="parent", root_task_id="parent", subagent_role="researcher")
            page.locator('.chat-live-card[data-task-id="parent"] > [data-live-summary-button]').click()
            child_card = page.locator(CHILD)
            child_card.wait_for(state="visible")
            # The scheduled receipt precedes the authoritative pending queue read.
            page.wait_for_function("() => document.querySelector('.chat-live-card[data-task-id=held] [data-live-phase]')?.textContent === 'Queued'")
            assert child_card.locator('[data-live-phase]').get_attribute('data-motion') == '0'
            assert not child_card.locator('[data-live-typing]').is_visible()
            page.evaluate("() => window.__heldCard = document.querySelector('.chat-live-card[data-task-id=held]')")
            shot(page, "scheduled")
            result_path.write_text("{torn", encoding="utf-8")
            workers.assign_tasks()
            assert not sent
            page.wait_for_function("text => document.querySelector('.chat-live-card[data-task-id=held] [data-live-phase]')?.textContent === text", arg=WAIT)
            assert "unreadable" in child_card.locator('[data-live-meta]').inner_text()
            assert not child_card.locator('[data-live-typing]').is_visible()
            assert child_card.locator('[data-live-phase]').get_attribute('class') == 'chat-live-phase warn'
            assert [row["activity_id"] for row in _chat_activities_snapshot_safe(root)] == ["parent"]
            shot(page, "held-desktop")
            for episode in ("reload", "reconnect"):
                if episode == "reload":
                    page.reload(wait_until="domcontentloaded")
                    page.wait_for_selector('#chat-messages[data-history-hydrated="true"]')
                else:
                    count = page.evaluate("() => window.__testSockets.length")
                    page.evaluate("() => window.__testSockets.at(-1).close()")
                    page.wait_for_function("n => window.__testSockets.length > n && window.__testSockets.at(-1).readyState === 1", arg=count)
                page.wait_for_function("text => document.querySelector('.chat-live-card[data-task-id=held] [data-live-phase]')?.textContent === text", arg=WAIT)
                parent = page.locator('.chat-live-card[data-task-id="parent"]')
                if parent.get_attribute("data-expanded") != "1":
                    parent.locator(':scope > [data-live-summary-button]').click()
                assert not child_card.locator('[data-live-typing]').is_visible()
                shot(page, episode)
            page.evaluate("() => window.__heldCard = document.querySelector('.chat-live-card[data-task-id=held]')")
            page.set_viewport_size({"width": 390, "height": 844})
            shot(page, "held-narrow")
            assert child_card.evaluate("el => el.scrollWidth <= el.clientWidth + 1")
            # Typed unknown-dispatch fact uses the same host projection, without a
            # promise of automatic continuation (no client text classification).
            held = next(row for row in host.pending if row["id"] == child["id"])
            previous_hold = held["_project_admission_restore_hold"]
            held["_project_admission_restore_hold"] = {"reason": "project_dispatch_unconfirmed",
                "detail": "The previous run is unconfirmed; automatic recovery is not authorized."}
            assert queue.persist_queue_snapshot()
            page.wait_for_function("() => document.querySelector('.chat-live-card[data-task-id=held] [data-live-phase]')?.textContent === 'Waiting: previous run unconfirmed'")
            assert "automatic recovery is not authorized" in child_card.locator('[data-live-meta]').inner_text()
            assert not child_card.locator('[data-live-typing]').is_visible()
            shot(page, "previous-run-unconfirmed-narrow")
            held["_project_admission_restore_hold"] = previous_hold
            result_path.write_bytes(original)
            if ending == "recovery":
                # Same accepted receipt returns; the assignment owner dispatches exactly once.
                workers.assign_tasks()
                workers.assign_tasks()
                assert [row["id"] for row in sent] == [child["id"]]
                page.wait_for_function("() => document.querySelector('.chat-live-card[data-task-id=held] [data-live-phase]')?.textContent === 'Working'")
                assert "unreadable" not in child_card.locator('[data-live-meta]').inner_text()
                shot(page, "recovered-narrow")
            else:
                from ouroboros.cancel_intents import request_cancel
                request_cancel(root, "held", reason="owner stopped")
                workers.assign_tasks()
                workers.assign_tasks()
                assert not sent and load_task_result(root, "held", strict=True)["status"] == "cancelled"
            assert page.evaluate("() => window.__heldCard === document.querySelector('.chat-live-card[data-task-id=held]')")
            # Stop/terminal truth wins while held as well as after recovery.
            write_task_result(root, "held", "cancelled", result="Stopped by the owner.")
            frame(page, "held", content="Stopped by the owner.", delegation_role="subagent", subagent_event="cancelled",
                  subagent_task_id="held", parent_task_id="parent", root_task_id="parent", subagent_role="researcher",
                  status="cancelled")
            page.wait_for_function("() => document.querySelector('.chat-live-card[data-task-id=held]')?.dataset.finished === '1'")
            assert child_card.locator('[data-live-phase]').inner_text() == "Cancelled"
            assert not child_card.locator('[data-live-typing]').is_visible()
            shot(page, "cancelled-narrow")
            assert not errors, errors
        except Exception:
            page.screenshot(path=str(evidence / "zz-failure.png"))
            raise
        finally:
            (evidence / "facts.json").write_text(json.dumps({"requests": request_count, "snapshots": snapshots,
                "physical_handoffs": [row["id"] for row in sent], "page_errors": errors}, indent=2), encoding="utf-8")
            browser.close()


@pytest.mark.parametrize("engine,ending", [("chromium", "header-recovery")])
def test_held_tree_header_updates_on_child_reply_without_detail_polling(direct_server_with_data, tmp_path, monkeypatch, engine, ending):  # noqa: F811
    from playwright.sync_api import sync_playwright
    from ouroboros import projects_registry as registry
    from ouroboros.gateway.state import _chat_activities_snapshot_safe
    from ouroboros.gateway.tasks import _tasks_list_payload
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.utils import append_jsonl
    from supervisor import queue, workers

    server = direct_server_with_data
    root = tmp_path / "producer-data"
    root.mkdir()
    host = _producers(monkeypatch, root, server["repo_dir"])
    host.root = root
    registry.create_project(root, "target", name="Synthetic Project")
    child, original = _uncertain_child(host, monkeypatch)
    # Start from the accepted scheduled receipt, then lose it while the card is open.
    result_path = root / "task_results" / "held.json"
    result_path.write_bytes(original)
    queue.RUNNING["parent"] = {"task": {"id": "parent", "chat_id": 1}, "started_at": time.time(), "attempt": 1}
    sent = worker(host, monkeypatch)
    assert queue.persist_queue_snapshot()
    snapshots, request_count, errors = [], [], []
    # Preserve an earlier RUNNING child reply while later producer state holds it.
    # A pending row means Queued and cannot justify this test's Working header.
    host.pending.remove(child)
    queue.RUNNING["held"] = {"task": child, "started_at": time.time(), "attempt": 1}
    assert queue.persist_queue_snapshot()
    delayed_queue = _tasks_list_payload(root, None, 10, True)
    host.pending.append(queue.RUNNING.pop("held")["task"])
    assert queue.persist_queue_snapshot()
    show_child_hold = freeze_next_child_reply = block_state = False
    completed_states, deferred_states = [], []
    state_count_at_child_reply = None
    evidence = Path(os.environ.get("OUROBOROS_TEST_TEMP_ROOT") or tmp_path) / "child-hold" / f"{engine}-{ending}"
    evidence.mkdir(parents=True, exist_ok=True)
    print(f"CHILD_HOLD_EVIDENCE={evidence}")

    def task_transport(route):
        nonlocal freeze_next_child_reply, block_state, state_count_at_child_reply
        request_count.append(route.request.url)
        if "queue_only=1" in route.request.url:
            body = _tasks_list_payload(root, None, 10, True) if show_child_hold else delayed_queue
            if freeze_next_child_reply:
                freeze_next_child_reply, block_state = False, True
                state_count_at_child_reply = len(completed_states)
            snapshots.append(copy.deepcopy(body))
            route.fulfill(status=200, json=body)
        else:
            tid = route.request.url.rsplit("/", 1)[-1]
            try:
                row = load_task_result(root, tid, strict=True)
            except ValueError:
                route.fulfill(status=503, json={"error": "result unavailable"})
                return
            route.fulfill(status=200 if row else 404, json=row or {"error": "not found"})

    def state_transport(route):
        if block_state:
            deferred_states.append(route)
            return
        response = route.fetch()
        body = response.json()
        body.update(active_chat_activities=_chat_activities_snapshot_safe(root), active_chat_activities_complete=True,
                    supervisor_ready=True, task_bindings={})
        completed_states.append(body)
        route.fulfill(response=response, json=body)

    def frame(page, tid, **fields):
        row = {"type": "chat", "role": "assistant", "is_progress": True,
               "chat_id": 1, "task_id": tid, "ts": "2026-09-30T10:00:00Z", **fields}
        append_jsonl(server["data_dir"] / "logs" / "progress.jsonl", row)
        _emit_ws_frame(page, row)

    def shot(page, name):
        page.locator(CHILD).scroll_into_view_if_needed()
        page.screenshot(path=str(evidence / f"{name}.png"), animations="disabled")

    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 900}, reduced_motion="reduce")
            page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.route("**/api/tasks*", task_transport)
            page.route("**/api/tasks/**", task_transport)
            page.route("**/api/state", state_transport)
            page.goto(server["url"], wait_until="domcontentloaded")
            page.wait_for_function("() => window.__testSockets?.some(s => s.readyState === 1)")
            page.wait_for_selector('#chat-messages[data-history-hydrated="true"]')
            frame(page, "parent", content="Inspecting the Project with a child researcher.")
            frame(page, "held", content="Child research scheduled", delegation_role="subagent", subagent_event="scheduled",
                  subagent_task_id="held", parent_task_id="parent", root_task_id="parent", subagent_role="researcher")
            page.locator('.chat-live-card[data-task-id="parent"] > [data-live-summary-button]').click()
            child_card = page.locator(CHILD)
            child_card.wait_for(state="visible")
            assert child_card.locator('[data-live-phase]').inner_text() == "Working"
            page.evaluate("() => window.__heldCard = document.querySelector('.chat-live-card[data-task-id=held]')")
            shot(page, "scheduled")
            result_path.write_text("{torn", encoding="utf-8")
            workers.assign_tasks()
            assert not sent
            # Both real producer rows now hold, but the child's earlier running
            # snapshot remains in transit until the parent census has painted.
            parent = page.locator('.chat-live-card[data-task-id="parent"]')
            parent_row = queue.RUNNING.pop("parent")["task"]
            parent_row["_project_admission_restore_hold"] = {"reason": "project_routing_fence_lookup_failed",
                "detail": "Original Project authority is unavailable."}
            host.pending.append(parent_row)
            assert queue.persist_queue_snapshot()
            page.wait_for_function("text => document.querySelector('.chat-live-card[data-task-id=parent] [data-live-phase]')?.textContent === text", arg=WAIT)
            assert child_card.locator('[data-live-phase]').inner_text() == "Working"
            assert page.locator('#chat-status').inner_text() == 'Working...'
            # Deliver the real child hold response, then keep subsequent state
            # reads unresolved: another census cannot mask a missing async sync.
            show_child_hold = freeze_next_child_reply = True
            with page.expect_response(lambda response: "queue_only=1" in response.url):
                pass
            page.wait_for_function("text => document.querySelector('#chat-status')?.textContent === text", arg=WAIT, timeout=10000)
            assert len(completed_states) == state_count_at_child_reply
            assert parent.locator(':scope > [data-live-summary-button] [data-live-phase]').inner_text() == WAIT
            assert child_card.locator('[data-live-phase]').inner_text() == WAIT
            assert not parent.locator(':scope > [data-live-summary-button] [data-live-typing]').is_visible()
            assert not child_card.locator('[data-live-typing]').is_visible()
            assert [row["activity_id"] for row in _chat_activities_snapshot_safe(root)] == ["parent"]
            shot(page, "both-held-header-waiting")
            page.set_viewport_size({"width": 390, "height": 844})
            shot(page, "both-held-header-waiting-narrow")
            block_state = False
            for route in deferred_states:
                state_transport(route)
            deferred_states.clear()
            # Once the child is terminal, the same held parent alone says Waiting.
            result_path.write_bytes(original)
            write_task_result(root, "held", "cancelled", result="Stopped by the owner.")
            frame(page, "held", content="Stopped by the owner.", delegation_role="subagent", subagent_event="cancelled",
                  subagent_task_id="held", parent_task_id="parent", root_task_id="parent", subagent_role="researcher",status="cancelled")
            page.wait_for_function("text => document.querySelector('#chat-status')?.textContent === text", arg=WAIT)
            shot(page, "held-parent-header-waiting")
            # Real working sibling is still entitled to Working in the same room.
            queue.RUNNING["working-sibling"]={"task":{"id":"working-sibling","chat_id":1},"started_at":time.time(),"attempt":1}
            write_task_result(root,"working-sibling","running",chat_id=1)
            frame(page,"working-sibling",content="Independent sibling is working.")
            page.wait_for_function("() => document.querySelector('#chat-status')?.textContent === 'Working...'")
            shot(page, "real-working-sibling")
            # A live child's first review group must not enroll it in the root
            # census's missing-task polling; its review body is already present.
            queue.RUNNING["review-child"] = {"task": {"id": "review-child", "chat_id": 1,
                "parent_task_id": "parent", "root_task_id": "parent", "delegation_role": "subagent"},
                "started_at": time.time(), "attempt": 1}
            write_task_result(root, "review-child", "running", chat_id=1, parent_task_id="parent", root_task_id="parent",
                              delegation_role="subagent")
            assert queue.persist_queue_snapshot()
            frame(page, "review-child", content="Review child is working.", delegation_role="subagent",
                  subagent_event="running", subagent_task_id="review-child", parent_task_id="parent",
                  root_task_id="parent", subagent_role="reviewer")
            frame(page, "review-child", role="system", system_type="skill_review", content="Review feedback received.",
                  review_group={"surface": "skill", "id": "task:review-child:alpha", "presentation_owner_task_id": "review-child",
                                "skill": "alpha", "status": "clean", "attempts": [{"job_id": "child-review", "skill": "alpha", "status": "clean"}]})
            page.wait_for_function("() => document.querySelector('.chat-live-card[data-task-id=review-child] [data-live-review-summary]')?.textContent.includes('Reviews')")
            for _pass in range(3):
                with page.expect_response(lambda response: "queue_only=1" in response.url, timeout=10000):
                    pass
            assert not [url for url in request_count if url.split("?")[0].endswith("/api/tasks/review-child")]
            assert page.locator('#chat-status').inner_text() == "Working..."
            shot(page, "working-review-child-no-detail-poll")
            assert not errors, errors
        except Exception:
            page.screenshot(path=str(evidence / "zz-failure.png"))
            raise
        finally:
            (evidence / "facts.json").write_text(json.dumps({"requests": request_count, "snapshots": snapshots,
                "physical_handoffs": [row["id"] for row in sent], "page_errors": errors}, indent=2), encoding="utf-8")
            browser.close()
