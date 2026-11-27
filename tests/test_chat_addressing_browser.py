"""The real SPA hides only successful addressing-only native conversations and routes work Replies.

Use ordinary composer sends, real native model/tool execution, the host's own
annotations and terminal metrics, then reload the same gateway history. The
observer records every mounted live card so a briefly created/removed empty
card cannot pass by disappearing before the final assertion.
"""

import json
import re
import uuid
from pathlib import Path

import pytest

from tests.system_e2e.harness import (
    ArtifactOracle, ModelGate, ScriptedStubModel, keyless_settings, message_text,
    start_server, wait_durable_result, wait_until,
)
from tests.test_owner_wait_integration import wait_clone as clone_fixture
from tests._shared import producer_evidence_dir
from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET

wait_clone = clone_fixture
pytestmark = [pytest.mark.ui_browser]

_OBSERVE_CARDS = """() => {
    window.__mountedCards = [];
    const record = node => {
        if (node.nodeType !== Node.ELEMENT_NODE) return;
        const cards = [...node.querySelectorAll('.chat-live-card')];
        if (node.matches('.chat-live-card')) cards.push(node);
        for (const card of cards) window.__mountedCards.push(card.dataset.taskId);
    };
    new MutationObserver(rows => rows.forEach(row => row.addedNodes.forEach(record)))
        .observe(document.documentElement, {childList:true, subtree:true});
}"""


def _work_order(text):
    """The objective an actor was handed, read past the drafted-objective frame.

    A started follow-up's first user row opens with the host's
    ``[OBJECTIVE_AUTHOR]`` notice (a task, not the owner, drafted it); the
    fixture routes actors by the work order that follows, never by the frame.
    """
    frame, closed, rest = text.partition("[/OBJECTIVE_AUTHOR]\n\n")
    return rest if closed and frame.startswith("[OBJECTIVE_AUTHOR]") else text


def test_work_order_reads_the_objective_under_the_drafted_objective_frame():
    from ouroboros.context import build_user_content

    rendered = build_user_content({"text": "MANAGED_FOLLOWUP_x", "metadata": {
        "objective_author": {"kind": "task", "task_id": "addressing-1"}}})
    assert rendered.startswith("[OBJECTIVE_AUTHOR]") and _work_order(rendered) == "MANAGED_FOLLOWUP_x"
    assert _work_order("NATIVE_ADDRESSING_x") == "NATIVE_ADDRESSING_x"


class ToolCallOnlyModel(ScriptedStubModel):
    """Use the harness's full-message seam for genuine tool-only turns.

    ScriptedStubModel normally narrates "still working" on every tool call.
    Authored progress is independently card-worthy, so this scenario supplies
    standard content:null tool calls. Final authored prose remains unchanged.
    """

    def _answer(self, body, seq):
        kind, message = super()._answer(body, seq)
        return kind, ({**message, "content": None} if message.get("tool_calls") else message)


def _user_rows(body):
    return [message_text(row) for row in body["messages"] if row.get("role") == "user"]


def _belongs(body, marker):
    return bool(body.get("tools")) and any(_work_order(text).startswith(marker) for text in _user_rows(body))


def _task_for(oracle, words):
    return next((event["task"] for event in oracle.events("task_received")
                 if event.get("task", {}).get("text") == words), None)


def _history(page, server, chat_id=1):
    response = page.request.get(f"{server.base_url}/api/chat/history?chat_id={chat_id}")
    assert response.ok, response.text()
    return response.json()["messages"]


def _saved_message(page, server, words, role="user", chat_id=1):
    rows = [row for row in _history(page, server, chat_id) if (row.get("role"), row.get("text")) == (role, words)]
    assert len(rows) <= 1, rows
    return rows[0] if rows else None


def _mail_receipts(oracle, row):
    return [item for item in oracle._jsonl("logs/chat_annotations.jsonl")
            if item.get("client_message_id") == row["client_message_id"] and item.get("action") == "mailbox_delivery"]


def _expect_delivery(oracle, row, tid):
    found = wait_until(lambda: _mail_receipts(oracle, row), 25)
    assert len(found or []) == 1, found
    assert (found[0]["status"], found[0]["target"], found[0]["source_row_address"]) == (
        "delivered", tid, row["row_address"]), found


def _offered_work_reply(body):
    """The keyed return handle the host offered this work for the owner's Reply, or None."""
    hit = re.search(r"reply_to_relay='(work-reply:[0-9a-f]+)'", "\n".join(_user_rows(body)))
    return hit[1] if hit else None


def _bubble(page, feed, role, words):
    # A later reply may repeat those exact words in its quote. Count only the bubble's own body.
    return feed.locator(f".chat-bubble.{role}[data-row-address]").filter(
        has=page.locator(":scope > .message").filter(has_text=re.compile(rf"^\s*{re.escape(words)}\s*$")))


def _open_reply(page, row, quote_words):
    from playwright.sync_api import expect

    row.click(button="right")
    page.locator('body > .chat-row-menu[role="menu"] [data-row-action="reply"]').click()
    strip = page.locator("#page-chat .chat-quote--composer")
    expect(strip.get_by_role("button", name=re.compile("Replying to.*" + re.escape(quote_words)))).to_be_visible()
    return strip


def _send_main(page, words):
    page.locator("#chat-input").fill(words)
    page.locator("#chat-send").click()


def _new_project(page, server, name):
    created = page.request.post(server.base_url + "/api/projects", data={"name": name, "no_folder": True})
    assert created.ok, created.text()
    return created.json()["project"]


def _open_project(page, pid):
    pane = page.locator(f'[id="pchat-{pid}-messages"][data-history-hydrated=true]')
    if not pane.is_visible():
        page.locator(f'.nav-project-row[data-project-id="{pid}"]').click(timeout=20000)
    pane.wait_for(state="visible", timeout=30000)
    return pane


@pytest.mark.parametrize("destination", ["main", "project"])
def test_reply_to_finished_launcher_reaches_only_its_live_work(tmp_path, destination):
    """A confirmed handoff outlives its speaker; a plain Main follow-up still reaches the Main author."""
    from playwright.sync_api import expect, sync_playwright
    from tests.candidate_checkout import candidate_checkout

    nonce = uuid.uuid4().hex[:10]
    launch, work, neighbor, explicit, plain, final = (
        f"{label}_{nonce}" for label in ("LAUNCH", "WORK", "NEIGHBOR", "EXPLICIT", "PLAIN", "HANDOFF")
    )
    actors = {marker: [] for marker in (work, neighbor, launch)}
    answered = set()
    target, project = {}, {}
    work_gate = ModelGate(lambda body: _belongs(body, work), timeout=160)
    neighbor_gate = ModelGate(lambda body: _belongs(body, neighbor), timeout=160)

    def response(body):
        actor = next(marker for marker in actors if _belongs(body, marker))
        actors[actor].append(body)
        first = len(actors[actor]) == 1
        if actor == launch:
            if first:
                return {"tool": "start_task", "arguments": {"workspace": "none", "objective": work,
                        "title": "Exact Reply target", **({"project_id": project["id"]} if project else {})}}
            for row in reversed(body["messages"]):
                if row.get("role") == "tool" and (hit := re.search(
                        r"finish_task\(about_work=['\"]([a-zA-Z0-9_-]+)", message_text(row))):
                    target["id"] = hit[1]
                    return {"tool": "finish_task", "arguments": {"action": "finish", "answer": final,
                            "about_work": hit[1]}}
            raise AssertionError("No host-confirmed work in start_task's result")
        if first:
            return {"tool": "read_file", "arguments": {"root": "system_repo", "path": "VERSION"}}
        # #3759: a Reply addressed to this work holds a regular finish until the work answers it in its source room
        # (loop_messages.work_reply_finish_hold). The work does what a real worker does: keyed answer, then finish.
        relay = _offered_work_reply(body)
        if actor == work and relay and relay not in answered:
            answered.add(relay)
            return {"tool": "send_user_message", "arguments": {"text": f"Answer from {work}", "reply_to_relay": relay}}
        return {"final": f"Done {actor}"}

    evidence = producer_evidence_dir(tmp_path / "evidence", "reply-reader")
    with candidate_checkout(Path(__file__).resolve().parents[1], tmp_path / "clone", origin_proof=True) as clone, \
            ToolCallOnlyModel([response] * 16, gate=lambda body: (work_gate(body), neighbor_gate(body))) as stub:
        server = start_server(clone, tmp_path / "instance", keyless_settings(stub, OUROBOROS_MAX_WORKERS=1),
                              ready_timeout=115)
        oracle = ArtifactOracle(server.data_root)
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(timeout=15000)
                try:
                    page = browser.new_page(viewport={"width": 1280, "height": 850}, reduced_motion="reduce")
                    page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
                    page.goto(server.base_url, wait_until="domcontentloaded", timeout=20000)
                    page.wait_for_function("() => window.__testSockets?.some(s => s.readyState === WebSocket.OPEN)",
                                           timeout=20000)
                    feed = page.locator("#chat-messages[data-history-hydrated=true]")
                    feed.wait_for(state="attached", timeout=30000)
                    if destination == "project":
                        project.update(_new_project(page, server, "Project Reply target"))
                    _send_main(page, launch)
                    source = _bubble(page, feed, "assistant", final)
                    expect(source).to_be_visible(timeout=90000)
                    address = source.get_attribute("data-row-address")
                    launcher = wait_until(lambda: _task_for(oracle, launch), 25)
                    assert launcher and wait_durable_result(oracle, launcher["id"], timeout=45)["status"] == "completed"
                    assert work_gate.arrived.wait(90), "scheduled work never entered its first request"
                    managed = wait_until(lambda: _task_for(oracle, work), 25)
                    assert managed and managed["id"] == target["id"] != launcher["id"]
                    if project:
                        assert managed["chat_id"] == project["chat_id"] and managed["project_id"] == project["id"]
                    _send_main(page, neighbor)
                    assert neighbor_gate.arrived.wait(90), "other Main turn never entered its first request"
                    strip = _open_reply(page, source, final)
                    _send_main(page, explicit)
                    expect(strip).to_have_count(0)
                    msg = wait_until(lambda: _saved_message(page, server, explicit), 25)
                    assert msg and (msg["reply_to"]["address"], msg["reply_to"]["work_task_id"]) == (
                        address, target["id"]), msg
                    _expect_delivery(oracle, msg, target["id"])
                    _send_main(page, plain)
                    ordinary = wait_until(lambda: _saved_message(page, server, plain), 25)
                    reader = wait_until(lambda: _task_for(oracle, neighbor), 25)
                    assert ordinary and not ordinary.get("reply_to") and reader and reader["id"] != target["id"]
                    _expect_delivery(oracle, ordinary, reader["id"])
                    work_gate.release.set()
                    neighbor_gate.release.set()
                    for tid in (target["id"], reader["id"]):
                        assert wait_durable_result(oracle, tid, timeout=65)["status"] == "completed"
                    work_input, main_input = ("\n".join(_user_rows(actors[actor][1])) for actor in (work, neighbor))
                    assert explicit in work_input.splitlines() and "The owner addressed this work directly" in work_input
                    assert msg["row_address"] in work_input and f"[Message from my human]: {plain}" not in work_input
                    assert f"[Message from my human]: {plain}" in main_input
                    assert f"[Message from my human]: {explicit}" not in main_input
                    assert "The owner addressed this work directly" not in main_input
                    assert explicit in main_input.splitlines() and f"went to {target['id']}" in main_input
                    assert not _task_for(oracle, explicit) and not _task_for(oracle, plain)

                    for replay in (False, True):
                        if replay:
                            page.reload(wait_until="domcontentloaded", timeout=20000)
                            feed.wait_for(state="visible", timeout=30000)
                        line = _bubble(page, feed, "user", explicit)
                        expect(line).to_have_count(1, timeout=20000)
                        expect(line).to_be_visible()
                        expect(line).to_have_attribute("data-row-address", msg["row_address"])
                        expect(line.locator(".chat-quote-main")).to_have_accessible_name(re.compile(re.escape(final)))
                        line.scroll_into_view_if_needed()
                        page.screenshot(path=str(evidence / ("replay.png" if replay else "live.png")), animations="disabled")
                    if project:
                        pane = _open_project(page, project["id"])
                        line = _bubble(page, pane, "user", explicit)
                        expect(line).to_have_count(1, timeout=20000)
                        expect(line).to_be_visible()
                        expect(line).to_have_attribute("data-row-address", msg["row_address"])
                except Exception:
                    page.screenshot(path=str(evidence / "failure.png"), animations="disabled")
                    (evidence / "failure-dom.html").write_text(page.content(), encoding="utf-8")
                    raise
                finally:
                    browser.close()
        finally:
            work_gate.release.set()
            neighbor_gate.release.set()
            server.stop()


def test_reply_wakes_its_parked_work_which_answers_under_it_in_main(tmp_path):
    """Reply from Main to a real parked Project work, then to its same card after completion."""
    from playwright.sync_api import expect, sync_playwright
    from tests.candidate_checkout import candidate_checkout

    nonce = uuid.uuid4().hex[:10]
    launch, work, question, answer, work_done, second, room_answer = (
        f"{label}_{nonce}" for label in ("LAUNCH", "WORK", "QUESTION", "ANSWER", "WORK_DONE", "SECOND", "ROOM_ANSWER")
    )
    title = "Parked Reply target"
    project, target = {}, {}
    launcher_calls, work_calls, room_calls, unexpected = [], [], [], []

    def response(body):
        texts = _user_rows(body)
        if _belongs(body, second):
            room_calls.append(body)
            if len(room_calls) == 1:
                return {"tool": "get_task_result", "arguments": {"task_id": target["id"], "view": "answer"}}
            assert any(work_done in message_text(row) for row in body["messages"] if row.get("role") == "tool"), (
                "the fall-through turn never read the previous work's result")
            return {"final": room_answer}
        if _belongs(body, work):
            work_calls.append(body)
            match len(work_calls):
                case 1:
                    # No clock and no watched dependency: only a real waking input can return it.
                    return {"tool": "await_messages", "arguments": {"mode": "warm"}}
                case 2:
                    wire = "\n".join(texts)
                    assert question in wire.splitlines(), "the first request after wake lost the exact owner words"
                    assert "(warm) and were woken by the owner's message or control" in wire, wire[-3000:]
                    assert f"task:{target['id']}" in wire, "the Reply's quoted work identity was lost"
                    match = re.search(r"reply_to_relay='(work-reply:[0-9a-f]+)'", wire)
                    assert match, "receiver did not get the addressed Reply's exact return handle"
                    return {"tool": "send_user_message", "arguments": {"text": answer, "reply_to_relay": match[1]}}
                case _:
                    return {"tool": "finish_task", "arguments": {"action": "finish", "answer": work_done}}
        if _belongs(body, launch):
            launcher_calls.append(body)
            if len(launcher_calls) == 1:
                return {"tool": "start_task", "arguments": {"workspace": "none", "objective": work,
                        "title": title, "project_id": project["id"]}}
            return {"final": f"LAUNCH_DONE_{nonce}"}
        unexpected.append(texts)
        return {"final": f"Unexpected reader {nonce}"}

    evidence = producer_evidence_dir(tmp_path / "evidence", "sleeping-reply")
    with candidate_checkout(Path(__file__).resolve().parents[1], tmp_path / "clone", origin_proof=True) as clone, \
            ToolCallOnlyModel([response] * 16) as stub:
        server = start_server(clone, tmp_path / "instance", keyless_settings(stub, OUROBOROS_MAX_WORKERS=1),
                              ready_timeout=115)
        oracle = ArtifactOracle(server.data_root)
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(timeout=15000)
                try:
                    page = browser.new_page(viewport={"width": 1280, "height": 850}, reduced_motion="reduce")
                    page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
                    page.goto(server.base_url, wait_until="domcontentloaded", timeout=20000)
                    page.wait_for_function("() => window.__testSockets?.some(s => s.readyState === WebSocket.OPEN)",
                                           timeout=20000)
                    feed = page.locator("#chat-messages[data-history-hydrated=true]")
                    feed.wait_for(state="attached", timeout=30000)
                    project.update(_new_project(page, server, title))
                    _send_main(page, launch)
                    task = wait_until(lambda: _task_for(oracle, launch), 35)
                    assert task and wait_durable_result(oracle, task["id"], timeout=60)["status"] == "completed"
                    managed = wait_until(lambda: _task_for(oracle, work), 50)
                    assert managed and managed["id"] != task["id"] and managed["project_id"] == project["id"]
                    work_id = target["id"] = managed["id"]
                    assert managed["chat_id"] == project["chat_id"] != 1
                    card_address = f"task:{work_id}"
                    card = feed.locator(f'.chat-live-card[data-row-address="{card_address}"]')
                    expect(card).to_be_visible(timeout=30000)

                    def parked():
                        row = oracle.running_row(work_id)
                        state = row.get("owner_wait") or {}
                        return state if state.get("state") == "waiting" and state.get("reason") == "sleep" else None

                    sleep = wait_until(parked, 50)
                    assert sleep and len(work_calls) == 1, f"work has not reached its real sleep: {oracle.running_row(work_id)}"
                    strip = _open_reply(page, card, title)
                    _send_main(page, question)
                    expect(strip).to_have_count(0)
                    sent = wait_until(lambda: _saved_message(page, server, question), 25)
                    assert sent and (sent["reply_to"]["address"], sent["reply_to"]["work_task_id"]) == (
                        card_address, work_id), sent
                    _expect_delivery(oracle, sent, work_id)
                    returned = wait_until(lambda: _saved_message(page, server, answer, "assistant"), 45)
                    assert returned and returned["task_id"] == work_id, returned
                    assert returned["reply_to"]["address"] == sent["row_address"], returned
                    final = wait_durable_result(oracle, work_id, timeout=60)
                    assert final["status"] == "completed", final
                    assert not _task_for(oracle, question), "live Reply started an ordinary room turn too"
                    assert not unexpected, unexpected
                    assert len([row for row in _history(page, server) if row.get("role") == "assistant"
                                and (row.get("reply_to") or {}).get("address") == sent["row_address"]]) == 1
                    for words, role, expected in ((question, "user", sent), (answer, "assistant", returned)):
                        shared = wait_until(lambda: _saved_message(page, server, words, role, project["chat_id"]), 25)
                        assert shared and shared["row_address"] == expected["row_address"], shared

                    question_row = _bubble(page, feed, "user", question)
                    answer_row = _bubble(page, feed, "assistant", answer)
                    for replay in (False, True):
                        if replay:
                            page.reload(wait_until="domcontentloaded", timeout=20000)
                            feed.wait_for(state="attached", timeout=30000)
                        for row, saved in ((question_row, sent), (answer_row, returned)):
                            expect(row).to_have_count(1, timeout=25000)
                            expect(row).to_be_visible()
                            expect(row).to_have_attribute("data-row-address", saved["row_address"])
                        expect(question_row.locator(":scope > .chat-quote .chat-quote-main")).to_be_visible()
                        expect(answer_row.locator(":scope > .chat-quote")).to_have_attribute("data-address", sent["row_address"])
                        assert question_row.evaluate("(row, address) => Boolean(row.compareDocumentPosition("
                            "[...row.parentElement.children].find(x => x.dataset.rowAddress === address)) & 4)",
                            returned["row_address"]), "the answer did not follow the owner's Reply in the Main feed"
                        answer_row.scroll_into_view_if_needed()
                        page.screenshot(path=str(evidence / ("parked-reply-replay.png" if replay else "parked-reply-live.png")),
                                        animations="disabled")

                    # Same callback, same exact work card, now a terminal record. A normal Main turn owns the answer.
                    _open_reply(page, card, title)
                    _send_main(page, second)
                    fallback = wait_until(lambda: _task_for(oracle, second), 35)
                    assert fallback and fallback["id"] != work_id and fallback["_is_direct_chat"] and fallback["chat_id"] == 1
                    facts = fallback["metadata"]["routing_contract"]["work_reply"]
                    assert (facts["target"], facts["state"]) == (work_id, "not_reading"), facts
                    expect(_bubble(page, feed, "assistant", room_answer)).to_be_visible(timeout=50000)
                    assert wait_durable_result(oracle, fallback["id"], timeout=45)["status"] == "completed"
                    second_row = _saved_message(page, server, second)
                    assert second_row and second_row["reply_to"]["address"] == card_address, second_row
                    assert not _mail_receipts(oracle, second_row), "a finished work got an active delivery receipt"
                    second_answer = _saved_message(page, server, room_answer, "assistant")
                    assert second_answer and second_answer["task_id"] == fallback["id"]
                    assert second_answer["reply_to"]["address"] == second_row["row_address"], second_answer
                    assert len(work_calls) == 3 and len(room_calls) == 2 and not unexpected
                    assert oracle.task_result(work_id)["status"] == "completed", "the Reply resurrected the finished work"
                    _bubble(page, feed, "assistant", room_answer).scroll_into_view_if_needed()
                    page.screenshot(path=str(evidence / "finished-work-reply.png"), animations="disabled")
                except Exception:
                    page.screenshot(path=str(evidence / "failure.png"), animations="disabled")
                    (evidence / "failure-dom.html").write_text(page.content(), encoding="utf-8")
                    raise
                finally:
                    browser.close()
        finally:
            server.stop()


def test_live_reply_to_same_turn_after_it_moves_to_project(tmp_path):
    """Stage in Main before the author moves; deliver that exact row to the open Project over WS."""
    from playwright.sync_api import expect, sync_playwright
    from ouroboros.projects_registry import project_id_for_task
    from tests.candidate_checkout import candidate_checkout

    nonce = uuid.uuid4().hex[:10]
    start, opening, reply, plain = (f"{label}_{nonce}" for label in ("MOVE", "OPENING", "REPLY", "MAIN"))
    project_name = f"Moved Reply {nonce}"
    moved_calls, main_calls, unexpected = [], [], []
    before_move = ModelGate(lambda body: _belongs(body, start) and len(moved_calls) == 1, timeout=180)
    after_move = ModelGate(lambda body: _belongs(body, start) and len(moved_calls) == 2, timeout=180)

    def response(body):
        if _belongs(body, start):
            moved_calls.append(body)
            match len(moved_calls):
                case 1:
                    return {"tool": "send_user_message", "arguments": {"text": opening}}
                case 2:
                    return {"tool": "move_to_project", "arguments": {"project_name": project_name}}
                case 3:
                    return {"tool": "read_file", "arguments": {"root": "system_repo", "path": "VERSION"}}
                case 4 if (relay := _offered_work_reply(body)):
                    # #3759: the Reply addressed to this work must be answered under its keyed relay before it may finish.
                    return {"tool": "send_user_message", "arguments": {"text": "Moved work answers.", "reply_to_relay": relay}}
                case _:
                    return {"final": "Moved work completed."}
        if _belongs(body, plain):
            main_calls.append(body)
            return {"final": "Main independently answered."}
        unexpected.append(_user_rows(body))
        return {"final": "Unexpected new reader."}

    evidence = producer_evidence_dir(tmp_path / "evidence", "moved-live-reply")
    with candidate_checkout(Path(__file__).resolve().parents[1], tmp_path / "clone", origin_proof=True) as clone, \
            ToolCallOnlyModel([response] * 16, gate=lambda body: (before_move(body), after_move(body))) as stub:
        server = start_server(clone, tmp_path / "instance", keyless_settings(stub), ready_timeout=115)
        oracle = ArtifactOracle(server.data_root)
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(timeout=15000)
                context = browser.new_context(viewport={"width": 1280, "height": 850}, reduced_motion="reduce")
                main, project = context.new_page(), context.new_page()
                try:
                    for page in (main, project):
                        page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
                        page.goto(server.base_url, wait_until="domcontentloaded", timeout=20000)
                        page.wait_for_function("() => window.__testSockets?.some(s => s.readyState === WebSocket.OPEN)",
                                               timeout=20000)
                    _send_main(main, start)
                    assert before_move.arrived.wait(90), "the author's opening did not reach the next round"
                    task = wait_until(lambda: _task_for(oracle, start), 25)
                    assert task and task["chat_id"] == 1
                    tid = task["id"]
                    main_feed = main.locator("#chat-messages[data-history-hydrated=true]")
                    source = _bubble(main, main_feed, "assistant", opening)
                    expect(source).to_be_visible(timeout=20000)
                    address = source.get_attribute("data-row-address")
                    strip = _open_reply(main, source, opening)
                    before_move.release.set()
                    assert after_move.arrived.wait(90), "the same author did not reach a model round after moving"
                    pid = project_id_for_task(server.data_root, tid)
                    assert pid, "the work never acquired a durable Project binding"
                    feed = _open_project(project, pid)
                    expect(feed.locator(f'.chat-live-card[data-task-id="{tid}"]:not([data-portal="1"])')).to_be_visible(timeout=30000)
                    history_pattern = "**/api/chat/history*"
                    abort_history = lambda route: route.abort()
                    project.route(history_pattern, abort_history)  # solely live frames may supply the new row
                    expect(strip).to_be_visible()
                    _send_main(main, reply)
                    line = _bubble(project, feed, "user", reply)
                    expect(line).to_have_count(1, timeout=30000)
                    expect(line).to_be_visible()
                    expect(line.locator(".chat-quote-main")).to_have_accessible_name(re.compile(re.escape(opening)))
                    saved = wait_until(lambda: _saved_message(main, server, reply), 25)
                    assert saved and (saved["reply_to"]["address"], saved["reply_to"]["work_task_id"]) == (
                        address, tid), saved
                    expect(line).to_have_attribute("data-row-address", saved["row_address"])
                    expect(_bubble(main, main_feed, "user", reply)).to_have_attribute("data-row-address", saved["row_address"])
                    line.scroll_into_view_if_needed()
                    project.screenshot(path=str(evidence / "live.png"), animations="disabled")
                    _expect_delivery(oracle, saved, tid)
                    _send_main(main, plain)
                    expect(main.get_by_text("Main independently answered.", exact=True)).to_be_visible(timeout=40000)
                    assert len(main_calls) == 1 and "The owner addressed this work directly" not in "\n".join(_user_rows(main_calls[0]))
                    after_move.release.set()
                    assert wait_durable_result(oracle, tid, timeout=70)["status"] == "completed"
                    next_words = "\n".join(_user_rows(moved_calls[3]))
                    assert reply in next_words.splitlines() and saved["row_address"] in next_words
                    assert "The owner addressed this work directly" in next_words
                    assert f"[Message from my human]: {plain}" not in next_words
                    assert not _task_for(oracle, reply) and not unexpected

                    project.unroute(history_pattern, abort_history)
                    project.reload(wait_until="domcontentloaded", timeout=20000)
                    _open_project(project, pid)
                    expect(line).to_have_count(1, timeout=20000)
                    expect(line).to_be_visible()
                    expect(line).to_have_attribute("data-row-address", saved["row_address"])
                    line.scroll_into_view_if_needed()
                    project.screenshot(path=str(evidence / "replay.png"), animations="disabled")
                except Exception:
                    for label, page in (("main", main), ("project", project)):
                        page.screenshot(path=str(evidence / f"failure-{label}.png"), animations="disabled")
                        (evidence / f"failure-{label}.html").write_text(page.content(), encoding="utf-8")
                    raise
                finally:
                    browser.close()
        finally:
            before_move.release.set()
            after_move.release.set()
            server.stop()


@pytest.mark.parametrize("case,width", [
    ("start_only", 1440), ("start_only", 390),
    ("read_start", 1440), ("failed_tell", 390),
])
def test_ordinary_addressing_card_tracks_real_work_through_metrics_and_reload(
    wait_clone, tmp_path, monkeypatch, case, width,
):
    from playwright.sync_api import sync_playwright
    from tests.system_e2e.harness import KeylessIsolatedServer

    direct_marker = "NATIVE_ADDRESSING_" + uuid.uuid4().hex
    child_marker = "MANAGED_FOLLOWUP_" + uuid.uuid4().hex
    start = {"tool": "start_task", "arguments": {
        "objective": child_marker, "workspace": "none", "title": "Admitted follow-up",
    }}
    steps = {
        "start_only": [start],
        "read_start": [{"tool": "read_file", "arguments": {"root": "system_repo", "path": "VERSION"}}, start],
        "failed_tell": [{"tool": "tell_task", "arguments": {"task_id": "missing-live-task", "note": "Please continue"}}],
    }[case]
    direct_calls = []
    child_calls = []

    def response(body):
        user_messages = [message_text(m) for m in body.get("messages", []) if m.get("role") == "user"]
        if any(_work_order(text).startswith(child_marker) for text in user_messages):
            child_calls.append(body)
            return ({"tool": "read_file", "arguments": {"root": "system_repo", "path": "VERSION"}}
                    if len(child_calls) == 1 else {"final": "The admitted follow-up is complete."})
        assert any(text.startswith(direct_marker) for text in user_messages), "unexpected model actor consumed the native script"
        index = len(direct_calls)
        direct_calls.append(body)
        return steps[index] if index < len(steps) else {"final": "The addressing attempt is complete."}

    home = tmp_path / "home"
    home.mkdir()
    original_env = KeylessIsolatedServer._env
    monkeypatch.setattr(KeylessIsolatedServer, "_env", lambda server: {
        **original_env(server), "HOME": str(home), "USERPROFILE": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
    })
    evidence = producer_evidence_dir(tmp_path / "evidence", "chat-addressing",
                                     env="OUROBOROS_BROWSER_EVIDENCE_OUT") / f"addressing-{case}-{width}"
    evidence.mkdir(parents=True, exist_ok=True)
    with ToolCallOnlyModel([response] * 16) as stub:
        server = start_server(wait_clone, tmp_path / "instance", keyless_settings(stub, OUROBOROS_MAX_WORKERS=1))
        oracle = ArtifactOracle(server.data_root)
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch()
                page = browser.new_page(viewport={"width": width, "height": 900}, has_touch=width < 980)
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
                try:
                    page.goto(server.base_url, wait_until="domcontentloaded")
                    page.wait_for_function("() => window.__testSockets?.[0]?.readyState === WebSocket.OPEN")
                    page.evaluate(_OBSERVE_CARDS)
                    page.evaluate("""() => {
                        window.__taskMetricsFrames = [];
                        window.__testSockets[0].addEventListener('message', event => {
                            const frame = JSON.parse(event.data);
                            if (frame.type === 'log' && frame.data?.type === 'task_metrics_event')
                                window.__taskMetricsFrames.push(frame.data);
                        });
                    }""")
                    page.locator("#chat-input").fill(direct_marker)
                    page.locator("#chat-send").click()
                    task = wait_until(lambda: next((row["task"] for row in oracle.events("task_received")
                        if row.get("task", {}).get("_is_direct_chat") and direct_marker in row["task"].get("text", "")), None), 90)
                    assert task, "composer send never reached an ordinary native turn"
                    task_id = task["id"]
                    result = wait_durable_result(oracle, task_id, timeout=90)
                    assert len(direct_calls) >= len(steps) + 1
                    message_id = task["metadata"]["origin_message_ref"]["client_message_id"]
                    if case != "failed_tell":
                        annotations = [row for row in oracle._jsonl("logs/chat_annotations.jsonl")
                                       if row.get("client_message_id") == message_id and row.get("status") == "scheduled"]
                        assert len(annotations) == 1
                        child_id = annotations[0]["target"]
                        assert child_id != task_id
                        wait_durable_result(oracle, child_id, timeout=90)
                        page.wait_for_function("id => window.__taskMetricsFrames.some(row => row.task_id === id)", arg=child_id, timeout=30000)
                        child_metrics = [row for row in oracle.supervisor_rows("task_metrics_event") if row.get("task_id") == child_id]
                        assert child_metrics and child_metrics[-1]["tool_calls"] == 1
                        assert len(child_calls) >= 2
                        # The follow-up is told the addressing turn drafted its objective.
                        drafted = next(message_text(m) for m in child_calls[0]["messages"] if m.get("role") == "user"
                                       and _work_order(message_text(m)).startswith(child_marker))
                        assert drafted.startswith("[OBJECTIVE_AUTHOR]") and f"drafted by task {task_id}," in drafted, drafted
                        page.locator('.msg-routing-annotation[data-annotation-status="scheduled"]').wait_for(timeout=30000)
                        page.locator(f'.chat-live-card[data-task-id="{child_id}"]').wait_for(state="attached", timeout=30000)
                        assert page.locator(f'.chat-live-card[data-task-id="{child_id}"]').count() == 1
                    else:
                        rows = oracle._jsonl("logs/chat_annotations.jsonl")
                        # A tell that quotes none of his words is the turn's own note: it never speaks
                        # for his message, and its refusal is the turn's own receipt.
                        assert [row for row in rows if row.get("client_message_id") == message_id] == []
                        annotations = [row for row in rows if row.get("target") == "missing-live-task"]
                        assert len(annotations) == 1
                        assert annotations[0]["action"] == "steer_task"
                        assert annotations[0]["status"] == "needs_manual_target"
                        assert annotations[0]["reason"] == "target_unknown"
                    # Wait for the actual aggregate task_metrics frame from the
                    # terminal producer, not a hand-authored approximation.
                    page.wait_for_function("id => window.__taskMetricsFrames.some(row => row.task_id === id)", arg=task_id, timeout=30000)
                    metrics = [row for row in oracle.supervisor_rows("task_metrics_event") if row.get("task_id") == task_id]
                    assert metrics and metrics[-1]["tool_calls"] == len(steps)
                    page.get_by_text("The addressing attempt is complete.", exact=True).wait_for(timeout=30000)
                    card = page.locator(f'.chat-live-card[data-task-id="{task_id}"]')

                    if case == "start_only":
                        # Addressing alone earns no card, and a host wait the turn sat through as it
                        # started (accounting, control storage) is a receipt of how the host ran, not
                        # work (owner decisions 11.09 and 16.09): it never makes one, live or on reload.
                        assert card.count() == 0
                        assert task_id not in page.evaluate("() => window.__mountedCards")
                    else:
                        card.wait_for(timeout=30000)
                        assert card.count() == 1
                        # A tool row or a tool error is content: the direct turn
                        # wears the task card (chrome follows content, owner
                        # decision 16.09) — a visible chip, a title, and the
                        # Main conversion control on an unbound turn. The turn's
                        # end lands after its reply and metrics, and an unnarrated
                        # card is named once it has ended (#1369): wait for it.
                        page.wait_for_function("id => document.querySelector(`.chat-live-card[data-task-id=\"${id}\"]`)"
                                               "?.dataset.finished === '1'", arg=task_id, timeout=30000)
                        assert card.locator("[data-live-phase]").is_visible()
                        assert card.locator("[data-live-title]").inner_text().strip() != ""
                        assert card.locator("[data-turn-into-project]").count() == 1
                    page.screenshot(path=str(evidence / "live.png"), full_page=True, animations="disabled")
                    page.reload(wait_until="domcontentloaded")
                    # A reload first paints the room's saved text preview (PR #2765): its words come before the
                    # history read that draws cards and receipts, so the read itself is waited for.
                    page.locator('#chat-messages[data-history-hydrated="true"]').wait_for(state="attached", timeout=30000)
                    page.get_by_text("The addressing attempt is complete.", exact=True).wait_for(timeout=30000)
                    assert page.locator(f'.chat-live-card[data-task-id="{task_id}"]').count() == (0 if case == "start_only" else 1)
                    if case != "failed_tell":
                        assert page.locator('.msg-routing-annotation[data-annotation-status="scheduled"]').count() == 1
                    page.screenshot(path=str(evidence / "reloaded.png"), full_page=True, animations="disabled")
                    assert not errors, errors
                    (evidence / "receipt.json").write_text(json.dumps({
                        "case": case, "task": task, "result": result,
                        "annotations": annotations,
                        "task_metrics": metrics,
                        "child_metrics": child_metrics if case != "failed_tell" else [],
                        "observed_direct_calls": len(direct_calls), "errors": errors,
                    }, ensure_ascii=False, indent=2), encoding="utf-8")
                    if case == "failed_tell":
                        # The integrated P5 producer types a rejected tell as
                        # TOOL_REPORTED_FAILURE. Preserve this acceptance even
                        # when an older preparation base still reports zero.
                        assert metrics[-1]["tool_errors"] == 1
                except Exception:
                    page.screenshot(path=str(evidence / "failure.png"), full_page=True, animations="disabled")
                    (evidence / "failure-dom.html").write_text(page.content(), encoding="utf-8")
                    raise
                finally:
                    browser.close()
        finally:
            server.stop()
            assert server.proc.poll() is not None
