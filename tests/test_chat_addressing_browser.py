"""The real SPA hides only successful addressing-only native conversations.

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


@pytest.mark.parametrize("destination", ["main", "project"])
def test_reply_to_finished_launcher_reaches_only_its_live_work(tmp_path, destination):
    """Real composer -> host ingress -> two independent readers -> next model wire -> replay."""
    from playwright.sync_api import expect, sync_playwright
    from tests.candidate_checkout import candidate_checkout

    nonce = uuid.uuid4().hex[:10]
    launch, work, neighbor, explicit, plain, final = [
        f"{label}_{nonce}" for label in ("LAUNCH", "WORK", "NEIGHBOR", "EXPLICIT", "PLAIN", "HANDOFF")]
    actors = {marker: [] for marker in (work, neighbor, launch)}
    target = {}
    project = {}

    def user_rows(body):
        return [message_text(row) for row in body.get("messages", []) if row.get("role") == "user"]

    def belongs(body, marker):
        return bool(body.get("tools")) and any(_work_order(text).startswith(marker) for text in user_rows(body))

    work_gate = ModelGate(lambda body: belongs(body, work), timeout=160)
    neighbor_gate = ModelGate(lambda body: belongs(body, neighbor), timeout=160)

    def response(body):
        actor = next(marker for marker in actors if belongs(body, marker))
        actors[actor].append(body)
        first = len(actors[actor]) == 1
        if actor == launch:
            if first:
                return {"tool": "start_task", "arguments": {"workspace": "none", "objective": work,
                                                                  "title": "Exact Reply target",
                                                                  **({"project_id": project["id"]} if project else {})}}
            for row in reversed(body["messages"]):
                if row.get("role") == "tool" and (hit := re.search(
                        r"finish_task\(about_work=['\"]([a-zA-Z0-9_-]+)", message_text(row))):
                    target["id"] = hit[1]
                    return {"tool": "finish_task", "arguments": {"action": "finish", "answer": final,
                                                                     "about_work": hit[1]}}
            raise AssertionError("No host-confirmed work in start_task's result")
        return ({"tool": "read_file", "arguments": {"root": "system_repo", "path": "VERSION"}}
                if first else {"final": f"Done {actor}"})

    def send(page, words):
        page.locator("#chat-input").fill(words)
        page.locator("#chat-send").click()

    evidence = producer_evidence_dir(tmp_path / "evidence", "reply-reader", env="OUROBOROS_BROWSER_EVIDENCE_OUT")
    evidence.mkdir(parents=True, exist_ok=True)
    with candidate_checkout(Path(__file__).resolve().parents[1], tmp_path / "clone", origin_proof=True) as clone, \
            ToolCallOnlyModel([response] * 16, gate=lambda body: (work_gate(body), neighbor_gate(body))) as stub:
        server = start_server(clone, tmp_path / "instance", keyless_settings(stub, OUROBOROS_MAX_WORKERS=1),
                              ready_timeout=115)
        print("REPLY_WITNESS_HOST_READY", flush=True)
        oracle = ArtifactOracle(server.data_root)
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(timeout=15000)
                print("REPLY_WITNESS_CHROMIUM_STARTED", flush=True)
                try:
                    page = browser.new_page(viewport={"width": 1280, "height": 850}, reduced_motion="reduce")
                    page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
                    page.goto(server.base_url, wait_until="domcontentloaded", timeout=20000)
                    page.wait_for_function("() => window.__testSockets?.some(s => s.readyState === WebSocket.OPEN)",
                                           timeout=20000)
                    feed = page.locator("#chat-messages[data-history-hydrated=true]")
                    feed.wait_for(state="attached", timeout=30000)
                    print("REPLY_WITNESS_BROWSER_READY", flush=True)
                    if destination == "project":
                        created = page.request.post(server.base_url + "/api/projects",
                                                    data={"name": "Project Reply target", "no_folder": True})
                        assert created.ok, created.text()
                        project.update(created.json()["project"])
                    send(page, launch)
                    source = feed.locator("[data-row-address]").filter(has=page.get_by_text(final, exact=True)).last
                    expect(source).to_be_visible(timeout=90000)
                    address = source.get_attribute("data-row-address")
                    launch_id = wait_until(lambda: next((r["task"]["id"] for r in oracle.events("task_received")
                        if r.get("task", {}).get("text") == launch), None), 25)
                    assert launch_id and wait_durable_result(oracle, launch_id, timeout=45)["status"] == "completed"
                    assert launch_id != target["id"]
                    assert work_gate.arrived.wait(90), "scheduled work never entered its actual first request"
                    if project:
                        child = next(r["task"] for r in oracle.events("task_received") if r["task"]["id"] == target["id"])
                        assert child["chat_id"] == project["chat_id"] and child["project_id"] == project["id"]
                    send(page, neighbor)
                    assert neighbor_gate.arrived.wait(90), "other Main turn never entered its actual first request"
                    source.click(button="right")
                    page.locator('body > .chat-row-menu[role="menu"] [data-row-action="reply"]').click()
                    strip = page.locator("#page-chat .chat-quote--composer")
                    expect(strip.get_by_role("button", name=re.compile("Replying to.*" + final))).to_be_visible()
                    page.screenshot(path=str(evidence / "composer-reply.png"), animations="disabled")
                    send(page, explicit)
                    expect(strip).to_have_count(0)
                    send(page, plain)

                    def delivered(words):
                        rows = page.request.get(f"{server.base_url}/api/chat/history").json()["messages"]
                        row = next((row for row in rows if row.get("text") == words and row.get("role") == "user"), None)
                        if not row:
                            return None
                        receipt = next((r for r in oracle._jsonl("logs/chat_annotations.jsonl")
                                        if r.get("client_message_id") == row.get("client_message_id")
                                        and r.get("action") == "mailbox_delivery"), None)
                        return (row, receipt) if receipt else None

                    a, ar = wait_until(lambda: delivered(explicit), 25) or ({}, {})
                    b, br = wait_until(lambda: delivered(plain), 25) or ({}, {})
                    assert a.get("reply_to", {}).get("address") == address and not b.get("reply_to"), (a, b)
                    assert ar.get("status") == "delivered" and ar.get("target") == target["id"], ar
                    neighbor_id = wait_until(lambda: next((r["task"]["id"] for r in oracle.events("task_received")
                        if r.get("task", {}).get("text") == neighbor), None), 25)
                    assert neighbor_id and br.get("status") == "delivered" and br.get("target") == neighbor_id, br
                    print("REPLY_WITNESS_EXACT_RECEIPTS", flush=True)
                    work_gate.release.set()
                    neighbor_gate.release.set()
                    for actor, wanted, forbidden in ((work, explicit, plain), (neighbor, plain, explicit)):
                        assert wait_until(lambda: len(actors[actor]) >= 2, 45), actor
                        words = "\n".join(user_rows(actors[actor][1]))
                        assert f"[Message from my human]: {wanted}" in words
                        assert f"[Message from my human]: {forbidden}" not in words
                    project_feed = f'[id="pchat-{project["id"]}-messages"][data-history-hydrated=true]' if project else None
                    for replay in (False, True):
                        if replay:
                            page.reload(wait_until="domcontentloaded", timeout=20000)
                            feed.wait_for(state="attached", timeout=30000)
                        if project_feed:
                            pane = page.locator(project_feed)
                            if not pane.is_visible():
                                page.locator(f'.nav-project-row[data-project-id="{project["id"]}"]').click(timeout=20000)
                            pane.wait_for(state="visible", timeout=30000)
                        else:
                            pane = feed
                        card = pane.locator(f'.chat-live-card[data-task-id="{target["id"]}"]')
                        line = card.locator("[data-live-work-replies] .chat-bubble.user").filter(has_text=explicit)
                        expect(pane.get_by_text(explicit, exact=True)).to_have_count(1, timeout=20000)
                        expect(line).to_have_count(1)
                        if not line.is_visible():
                            card.locator(":scope > [data-live-summary-button] [data-live-disclosure]").click()
                        expect(line.get_by_role("button", name=re.compile("Reply to.*" + final))).to_be_visible()
                        expect(line).to_be_visible()
                        page.screenshot(path=str(evidence / ("replay.png" if replay else "live.png")), animations="disabled")
                finally:
                    browser.close()
        finally:
            work_gate.release.set()
            neighbor_gate.release.set()
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
