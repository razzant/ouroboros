"""Converting a RUNNING direct Main turn into a project, end to end in a browser.

Owner decision 16.09: a direct conversation turn that does real work IS a full
task card in Main, and it offers "Turn into project" WHILE IT RUNS — not only
after it ends. The running half is what no unit test can certify: the click has
to land while the turn is inside a model round, the durable binding has to be
written under the live turn, and the turn's own final answer has to follow that
binding into the project room instead of landing back in Main.

The turn is held inside its SECOND model round by a ``ModelGate`` (the same
event-gated hold S26/S27 use), so "still running" is a fact of the HTTP boundary,
never a sleep or a race: round one performs a real ``read_file`` (the work the
card stands on), round two blocks until this test releases it, and its release
produces the final answer whose delivery room is the whole point.
"""

import json
import os
import uuid
from pathlib import Path

import pytest

from ouroboros.projects_registry import project_binding_for_task, project_id_for_task
from tests.system_e2e.harness import (
    ArtifactOracle, ModelGate, body_text, keyless_settings,
    start_server, wait_durable_result, wait_until,
)
from tests.test_chat_addressing_browser import ToolCallOnlyModel
from tests.test_owner_wait_integration import wait_clone as clone_fixture
from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET

wait_clone = clone_fixture
pytestmark = [pytest.mark.serial, pytest.mark.ui_browser]

ANSWER = "The converted turn finished its work.\n\n" + "\n\n".join(
    f"Verified section {i}: the complete answer remains readable in the Project and selectable in Main."
    for i in range(12)
)
# The turn namer and the project namer are TOOL-LESS model calls, which the stub
# answers with its default text — so pinning it to something the answer does not
# contain keeps "the project name" and "the answer" separable strings, and the
# room assertions below cannot pass on an accidental substring.
CARD_NAME = "Held work of a direct turn"


def _out_rows(oracle, text):
    return [row for row in oracle._jsonl("logs/chat.jsonl")
            if row.get("direction") == "out" and text in str(row.get("text") or "")]


_MAIN_CARD_FACTS = """id => {
    const card = document.querySelector(`#page-chat .chat-live-card[data-task-id="${id}"]`);
    return {
        card: !!card,
        converted: card?.dataset.projectCreated || '',
        bound: card?.dataset.projectBound || '',
        pointer: !!card?.querySelector('.chat-live-bound-pointer'),
        convert_buttons: document.querySelectorAll('#page-chat [data-turn-into-project]').length,
    };
}"""


def _task_rows(oracle, task_id):
    return [{"direction": row.get("direction"), "chat_id": row.get("chat_id"),
             "type": row.get("type", ""), "text": str(row.get("text") or "")[:80]}
            for row in oracle._jsonl("logs/chat.jsonl")
            if str(row.get("task_id") or "") == task_id]


@pytest.mark.parametrize("width", [1440, 390], ids=["desktop", "mobile"])
@pytest.mark.parametrize("engine,theme", [("chromium", "dark"), ("webkit", "light")])
def test_running_direct_turn_converts_to_project_and_its_answer_follows(
    wait_clone, tmp_path, monkeypatch, width, engine, theme,
):
    from playwright.sync_api import sync_playwright
    from tests.system_e2e.harness import KeylessIsolatedServer

    marker = "CONVERT_RUNNING_" + uuid.uuid4().hex
    # Round 1 does real work; round 2 (held) carries the tool RESULT message and
    # answers once released. The script is exactly those two rounds.
    steps = [
        {"tool": "read_file", "arguments": {"root": "system_repo", "path": "VERSION"}},
        {"final": ANSWER},
    ]

    def held_round(body):
        """The turn's own round that already carries a tool result — i.e. the
        round AFTER the card's work exists. Naming/review calls carry no tools
        and are never held, so the hold cannot starve the conversion itself."""
        messages = [m for m in body.get("messages", []) if isinstance(m, dict)]
        return (bool(body.get("tools")) and marker in body_text(body)
                and any(str(m.get("role") or "") == "tool" for m in messages))

    gate = ModelGate(held_round, timeout=300)
    home = tmp_path / "home"
    home.mkdir()
    original_env = KeylessIsolatedServer._env
    monkeypatch.setattr(KeylessIsolatedServer, "_env", lambda server: {
        **original_env(server), "HOME": str(home), "USERPROFILE": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
    })
    evidence = Path(os.environ.get("OUROBOROS_BROWSER_EVIDENCE_OUT") or tmp_path / "evidence")
    evidence = evidence / f"convert-running-{engine}-{theme}-{width}"
    evidence.mkdir(parents=True, exist_ok=True)
    with ToolCallOnlyModel(steps, final_answer=CARD_NAME, gate=gate) as stub:
        server = start_server(wait_clone, tmp_path / "instance", keyless_settings(stub, OUROBOROS_MAX_WORKERS=1))
        oracle = ArtifactOracle(server.data_root)
        try:
            with sync_playwright() as pw:
                browser = getattr(pw, engine).launch()
                page = browser.new_page(viewport={"width": width, "height": 900}, has_touch=width < 980)
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
                page.add_init_script(f"localStorage.setItem('ouroboros.theme', '{theme}')")
                try:
                    page.goto(server.base_url, wait_until="domcontentloaded")
                    page.wait_for_function("() => window.__testSockets?.[0]?.readyState === WebSocket.OPEN")
                    page.locator("#chat-input").fill(marker)
                    page.locator("#chat-send").click()

                    task = wait_until(lambda: next((row["task"] for row in oracle.events("task_received")
                        if row.get("task", {}).get("_is_direct_chat") and marker in row["task"].get("text", "")), None), 90)
                    assert task, "composer send never reached an ordinary direct turn"
                    task_id = task["id"]
                    assert not task.get("_ephemeral_turn"), task
                    assert gate.arrived.wait(120), "the direct turn never reached its second model round"

                    # ---- the turn is provably RUNNING, inside a model round ----
                    card = page.locator(f'.chat-live-card[data-task-id="{task_id}"]')
                    card.wait_for(timeout=30000)
                    convert = card.locator("[data-turn-into-project]")
                    convert.wait_for(timeout=30000)
                    assert page.locator(
                        f'.chat-live-card[data-task-id="{task_id}"][data-finished="0"]'
                    ).count() == 1, "the conversion control is not on a RUNNING card"
                    assert convert.count() == 1
                    assert task_id not in oracle.running_ids(), "a direct turn must not take a pool worker"
                    at_click = {
                        "stored_status": str(oracle.task_result(task_id).get("status") or ""),
                        "gate_held": gate.held, "gate_released": gate.release.is_set(),
                        "binding": project_id_for_task(server.data_root, task_id),
                        "card_finished": card.get_attribute("data-finished"),
                        "title": card.locator("[data-live-title]").inner_text().strip(),
                    }
                    assert at_click["stored_status"] == "running", at_click
                    assert at_click["gate_held"] == 1 and at_click["gate_released"] is False
                    assert at_click["binding"] == "", "the turn is bound before the owner clicked"
                    page.screenshot(path=str(evidence / "live-running.png"), full_page=True, animations="disabled")

                    # ---- the one click, and the server's own answer to it ----
                    with page.expect_response(
                        lambda response: "/api/projects/from-task" in response.url, timeout=60000,
                    ) as captured:
                        convert.click()
                    response = captured.value
                    assert response.status == 200, (response.status, response.text())
                    payload = response.json()
                    assert payload["adopted"] is False, payload
                    project = payload["project"]
                    assert project.get("id") and project.get("chat_id"), project
                    # Named from the work (owner P1: one click, no prompt), never
                    # from the answer the turn has not even produced yet.
                    assert str(project.get("name") or "").strip(), project
                    assert ANSWER not in str(project["name"]), project
                    project_chat = int(project["chat_id"])
                    assert project_chat != 1, project
                    # The receipt is one typed word beside the binding: a captured
                    # Main origin makes it durable, so the card carries no gap mark.
                    assert payload["handoff_receipt"] == "durable", payload
                    assert payload["handoff_id"].startswith("project-handoff:"), payload

                    # The Main card became the project chip, and stopped offering
                    # a second conversion of the same work.
                    converted = page.locator(
                        f'.chat-live-card[data-task-id="{task_id}"][data-project-created="1"]')
                    converted.wait_for(timeout=30000)
                    page.wait_for_selector(
                        f'.chat-live-card[data-task-id="{task_id}"].is-project', timeout=30000)
                    assert converted.get_attribute("data-project-id") == project["id"]
                    assert converted.get_attribute("data-handoff-id") == payload["handoff_id"]
                    assert converted.get_attribute("data-receipt") is None, "a durable receipt is not marked as a gap"
                    assert converted.locator(".chat-live-project-name").inner_text().strip()
                    assert page.locator(f'.chat-live-card[data-task-id="{task_id}"]'
                                        ' [data-turn-into-project]').count() == 0
                    page.screenshot(path=str(evidence / "live-converted.png"), full_page=True, animations="disabled")

                    # The durable binding exists while the turn is STILL held.
                    assert gate.held == 1 and not gate.release.is_set()
                    assert project_id_for_task(server.data_root, task_id) == project["id"]
                    binding = project_binding_for_task(server.data_root, task_id)
                    assert int(binding["project_chat_id"]) == project_chat, binding
                    assert str(oracle.task_result(task_id).get("status") or "") == "running"

                    # ---- release: the turn's own answer follows the binding ----
                    gate.release.set()
                    stored = wait_durable_result(oracle, task_id, timeout=120)
                    assert stored["status"] == "completed", stored
                    assert ANSWER in str(stored.get("result") or ""), stored
                    delivered = wait_until(lambda: _out_rows(oracle, ANSWER), 60) or []
                    assert len(delivered) == 1, delivered
                    final_row = delivered[0]
                    assert final_row["chat_id"] == project_chat, final_row
                    assert final_row["task_id"] == task_id, final_row
                    assert not [row for row in _out_rows(oracle, ANSWER) if row["chat_id"] == 1], \
                        "the final answer of a converted turn was also delivered to Main"
                    # Main keeps the transfer receipt and a separate final-answer
                    # mirror; the original authored row belongs to the Project.
                    # The completion mirror is the supervisor's task_done row, written after
                    # the worker's own final delivery: wait for it like every other durable fact.
                    rows = wait_until(lambda: (lambda r: r if any(
                        row["chat_id"] == 1 and row["type"] == "project_completion_summary" for row in r) else None)(
                        _task_rows(oracle, task_id)), 60) or _task_rows(oracle, task_id)
                    main_rows = [row for row in rows if row["chat_id"] == 1]
                    handoff_rows = [row for row in main_rows if row["type"] == "project_handoff"]
                    completion_rows = [row for row in main_rows if row["type"] == "project_completion_summary"]
                    assert len(handoff_rows) == 1, rows
                    assert len(completion_rows) == 1, rows
                    assert project["name"] in handoff_rows[0]["text"], main_rows
                    assert [row for row in rows if row["chat_id"] == project_chat
                            and row["direction"] == "out"], rows

                    # ---- reload: the Project room owns the work; Main offers no second conversion ----
                    page.reload(wait_until="domcontentloaded")
                    page.wait_for_function("() => window.__testSockets?.[0]?.readyState === WebSocket.OPEN")
                    page.wait_for_selector("#page-chat")
                    page.get_by_text(marker, exact=False).first.wait_for(timeout=30000)
                    # Zero convert buttons is also the state of a card whose
                    # binding has not landed yet. Wait for the fact the
                    # assertion below requires: no card, or a card that names
                    # its project.
                    reloaded = wait_until(
                        lambda: (lambda facts: facts if (
                            not facts["card"] or facts["converted"] == "1" or facts["bound"] == "1"
                        ) else None)(
                            page.evaluate(_MAIN_CARD_FACTS, task_id)),
                        30) or page.evaluate(_MAIN_CARD_FACTS, task_id)
                    assert reloaded["convert_buttons"] == 0, \
                        f"Main still offers to convert work that already has a project: {reloaded}"
                    # A surviving card names its Project instead of offering
                    # a second conversion; the durable receipt survives separately.
                    assert not reloaded["card"] or reloaded["converted"] == "1" or reloaded["bound"] == "1", \
                        f"a surviving Main card does not name its project: {reloaded}"
                    main_text = page.locator("#page-chat").inner_text()
                    page.locator('#page-chat .project-answer').filter(has_text='The converted turn finished its work.').wait_for(timeout=30000)
                    page.wait_for_selector('#page-chat .project-answer.is-folded', timeout=30000)
                    page.wait_for_function("() => [...document.querySelectorAll('#page-chat .project-handoff:not([hidden]) .chat-live-phase')].some(n => n.textContent === 'Done')", timeout=30000)
                    assert page.locator('#page-chat .project-handoff:not([hidden])').count() == 1
                    assert marker in main_text, "Main lost the owner message the turn started from"
                    main_history = page.evaluate(
                        "async () => (await (await fetch('/api/chat/history?chat_id=1')).json())")
                    replayed = main_history.get("messages") or []
                    assert len([row for row in replayed if row.get("system_type") == "project_handoff"]) == 1
                    answers = [row for row in replayed if row.get("system_type") == "project_completion_summary"]
                    assert len(answers) == 1 and answers[0].get("completion_answer") == ANSWER
                    page.screenshot(path=str(evidence / "reloaded-main.png"), full_page=True, animations="disabled")
                    page.evaluate(
                        "project => window.dispatchEvent(new CustomEvent('ouro:open-project', {detail:{project}}))",
                        project)
                    page.wait_for_selector("#project-panel:not([hidden])")
                    page.locator("#project-panel").get_by_text('The converted turn finished its work.', exact=False).first.wait_for(timeout=30000)
                    page.screenshot(path=str(evidence / "reloaded-project.png"), full_page=True, animations="disabled")
                    assert not errors, errors
                    (evidence / "receipt.json").write_text(json.dumps({
                        "engine": engine, "theme": theme, "width": width, "task": task, "at_click": at_click,
                        "from_task_response": payload, "binding": binding,
                        "stored_result_status": stored["status"],
                        "final_chat_row": final_row, "project_chat_id": project_chat,
                        "reloaded_main": reloaded, "task_chat_rows": rows, "reloaded_main_history": main_history,
                        "progress_chat_ids": sorted({int(row.get("chat_id") or 0) for row in
                                                      oracle._jsonl("logs/progress.jsonl")
                                                      if str(row.get("task_id") or "") == task_id}),
                        "model_rounds": stub.kinds(), "errors": errors,
                    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
                except Exception:
                    page.screenshot(path=str(evidence / "failure.png"), full_page=True, animations="disabled")
                    (evidence / "failure-dom.html").write_text(page.content(), encoding="utf-8")
                    raise
                finally:
                    gate.release.set()
                    browser.close()
        finally:
            gate.release.set()
            server.stop()
            assert server.proc.poll() is not None
            assert not gate.timed_out
