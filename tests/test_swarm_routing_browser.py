"""Real websocket -> host admission -> pooled root -> model and SPA consumers.

The existing keyless SystemHarness owns transport and server isolation. The
first model request is held at its real HTTP boundary while durable admission
and replay are checked. No production route, worker, tool, or UI handler is
mocked. Text cases use the real Swarm toggle, composer and file input; caption
cases retain the direct WS ingress proof. The model chooses a real read followed by an owner question; the test
then uses the supported Stop control to retire its isolated work.
"""

import base64
import hashlib
import json
import os
import uuid
from pathlib import Path

import httpx
import pytest

from devtools.benchmarks.common.server_runner import _api
from ouroboros.gateway.routing_decision import _derived_identity
from ouroboros.project_dialogue import build_owner_message_ref
from ouroboros.projects_registry import project_binding_for_task
from tests.system_e2e.harness import (
    ArtifactOracle, ModelGate, ScriptedStubModel, body_text, keyless_settings,
    start_server, wait_durable_result, wait_until,
)
from tests.test_owner_wait_integration import wait_clone as clone_fixture
from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET

wait_clone = clone_fixture
pytestmark = [pytest.mark.serial, pytest.mark.ui_browser]

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aKs8AAAAASUVORK5CYII="
)


def _send(page, message):
    page.evaluate("message => window.__testSockets[0].send(JSON.stringify(message))", message)


# Every owner echo the server sends this page's socket, in arrival order, plus the
# failure notice a refused chat frame answers its sender with instead.
_RECORD_ECHOES = """() => {
    window.__ownerEchoes = [];
    window.__testSockets[0].addEventListener('message', event => {
        const frame = JSON.parse(event.data);
        if (frame.type === 'chat' && (frame.role === 'user' || frame.system_type === 'initialization_notice'))
            window.__ownerEchoes.push(frame);
    });
}"""


def _echoes(page, message_id):
    return page.evaluate("""id => window.__ownerEchoes.filter(frame =>
        frame.client_message_id === id || frame.system_type === 'initialization_notice')""", message_id)


def _annotations(oracle, message_id):
    return [row for row in oracle._jsonl("logs/chat_annotations.jsonl")
            if row.get("client_message_id") == message_id]


def _received(oracle, task_id):
    # Project roots execute in their existing forked data root. Queue and
    # admission are still budget-root facts; execution evidence belongs here.
    return [row["task"] for row in oracle.task_drive(task_id).events("task_received")
            if row.get("task", {}).get("id") == task_id]


def _queued(oracle, task_id):
    snapshot = oracle.queue_snapshot()
    return [row for phase in ("pending", "running") for row in snapshot.get(phase, [])
            if row.get("id") == task_id]


@pytest.mark.parametrize("room,width,caption_only", [
    ("main", 1440, False), ("project", 390, False),
    ("main", 390, True), ("project", 1440, True),
], ids=["main-desktop-text", "project-mobile-text", "main-mobile-caption", "project-desktop-caption"])
def test_swarm_admission_precedes_model_and_survives_replay_and_reload(
    wait_clone, tmp_path, monkeypatch, room, width, caption_only,
):
    from playwright.sync_api import sync_playwright
    from tests.system_e2e.harness import KeylessIsolatedServer

    marker = "SWARM_BROWSER_" + uuid.uuid4().hex
    message_id = uuid.uuid4().hex if caption_only else ""
    token, task_id = _derived_identity(message_id, "swarm", 0) if message_id else ("", "")
    raw = f"  {marker}\nRead the repository, then ask which evidence to use.  \n"
    # A wordless send is logged under the host placeholder (the gateway no longer
    # captions uploads); the marker reaches the model as the image's own label.
    caption = "(image attached)"
    first_request = []

    def for_this_message(task):
        # The placeholder objective names no marker; a caption case's id is fixed up front.
        if caption_only:
            return (task.get("origin_message_ref") or {}).get("client_message_id") == message_id
        return marker in str(task.get("objective") or "")

    def first_agent_request(body):
        match = bool(body.get("tools")) and marker in body_text(body)
        if match and not first_request:
            # Read these inside the HTTP request callback, before ModelGate
            # announces arrival or releases a response. A later observation
            # alone could hide a routing actor admitting a root mid-request.
            snapshot = oracle.queue_snapshot()
            admitted = [row["task"] for phase in ("pending", "running") for row in snapshot.get(phase, [])
                        if for_this_message(row.get("task", {}))]
            observed_id = admitted[0]["id"] if len(admitted) == 1 else ""
            first_request.append({
                "body": body, "admitted_roots": admitted, "observed_root_id": observed_id,
                "received": _received(oracle, observed_id) if observed_id else [],
                "queue": snapshot, "result": oracle.task_result(observed_id) if observed_id else {},
                "previous_model_kinds": stub.kinds(),
            })
        return match

    gate = ModelGate(first_agent_request)
    steps = [
        {"tool": "list_files", "arguments": {"root": "system_repo", "path": "."}},
        {"tool": "escalate", "arguments": {
            "question": "Which evidence should this admitted root use?",
            "options": [{"label": "Repository"}, {"label": "Attachment"}],
            "wait_for_answer": True,
        }},
    ]
    home = tmp_path / "home"
    home.mkdir()
    original_env = KeylessIsolatedServer._env
    monkeypatch.setattr(KeylessIsolatedServer, "_env", lambda server: {
        **original_env(server), "HOME": str(home), "USERPROFILE": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
    })
    evidence = Path(os.environ.get("OUROBOROS_BROWSER_EVIDENCE_OUT") or tmp_path / "evidence")
    evidence = evidence / f"swarm-{room}-{width}-{'caption' if caption_only else 'text'}"
    evidence.mkdir(parents=True, exist_ok=True)
    with ScriptedStubModel(steps, gate=gate) as stub:
        server = start_server(wait_clone, tmp_path / "instance", keyless_settings(stub, OUROBOROS_MAX_WORKERS=1))
        oracle = ArtifactOracle(server.data_root)
        try:
            project = (_api(server.base_url, "POST", "/api/projects", {"name": "Swarm evidence"})["project"]
                       if room == "project" else {})
            chat_id = project.get("chat_id", 1)
            uploads = []
            payloads = {"owner-notes.txt": b"Exact owner attachment\nsecond line\n", f"{marker}.png": _PNG}
            with httpx.Client(trust_env=False) as client:
                for name, payload in (payloads.items() if caption_only else []):
                    mime = "image/png" if name.endswith(".png") else "text/plain"
                    response = client.post(server.base_url + "/api/chat/upload", files={"file": (name, payload, mime)})
                    response.raise_for_status()
                    uploads.append({**response.json(), "mime": mime})
            surface = {"viewport": {"w": width, "h": 900}, "narrow_layout": width < 980,
                       "coarse_pointer": width < 980, "pywebview": False, "ua": "Swarm browser proof"}
            message = {"type": "chat", "content": " " if caption_only else raw,
                       "client_message_id": message_id, "chat_id": chat_id,
                       "force_plan": True, "attachments": uploads, "client_surface": surface,
                       **({"project_id": project["id"]} if project else {})}
            with sync_playwright() as pw:
                browser = pw.chromium.launch()
                page = browser.new_page(viewport={"width": width, "height": 900}, has_touch=width < 980)
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
                try:
                    page.goto(server.base_url, wait_until="domcontentloaded")
                    page.wait_for_function("() => window.__testSockets?.[0]?.readyState === WebSocket.OPEN")
                    page.evaluate(_RECORD_ECHOES)
                    if project:
                        # The same existing navigation event is used by the Project reference
                        # annotations and the side navigation, including mobile.
                        page.evaluate("project => window.dispatchEvent(new CustomEvent('ouro:open-project', {detail:{project}}))", project)
                        page.wait_for_selector('#project-panel:not([hidden])')
                    if caption_only:
                        _send(page, message)
                    else:
                        # Observe the real ws.send output, including its own
                        # generated client_message_id. Never preset or rewrite it.
                        page.evaluate("""() => {
                            window.__sentChatFrames = [];
                            const socket = window.__testSockets[0];
                            const send = socket.send.bind(socket);
                            socket.send = payload => {
                                const frame = JSON.parse(payload);
                                if (frame.type === 'chat') window.__sentChatFrames.push(frame);
                                return send(payload);
                            };
                        }""")
                        composer = page.locator('#project-panel:not([hidden])' if project else '#page-chat')
                        composer.locator('.chat-text-row textarea').fill(raw)
                        composer.locator('.chat-file-input-hidden').set_input_files([
                            {"name": name, "mimeType": "image/png" if name.endswith(".png") else "text/plain", "buffer": payload}
                            for name, payload in payloads.items()
                        ])
                        composer.locator('.attach-name').filter(has_text='owner-notes.txt').wait_for()
                        swarm = composer.locator('.chat-swarm')
                        swarm.click()
                        assert swarm.get_attribute('data-armed') == 'true'
                        page.screenshot(path=str(evidence / 'button-armed.png'), full_page=True, animations='disabled')
                        composer.locator('.chat-send-inline').click()
                        page.wait_for_function('() => window.__sentChatFrames.length === 1')
                        message = page.evaluate('() => window.__sentChatFrames[0]')
                        message_id = message['client_message_id']
                        assert message_id and message['force_plan'] is True
                        assert int(message.get('chat_id') or 1) == chat_id
                        assert message.get('project_id', '') == project.get('id', '')
                        assert message['content'].startswith(raw.strip())
                        assert len(message['attachments']) == len(payloads)
                        assert swarm.get_attribute('data-armed') == 'false'
                        token, task_id = _derived_identity(message_id, 'swarm', 0)
                        surface = message['client_surface']
                    assert gate.arrived.wait(90), "Swarm never reached the first managed model request"
                    at_call = first_request[0]
                    assert len(at_call['admitted_roots']) == 1
                    assert at_call['observed_root_id'] == task_id
                    assert at_call['admitted_roots'][0]['origin_message_ref']['client_message_id'] == message_id
                    assert len(at_call["received"]) == 1
                    assert at_call["result"]["promotion_admission"]["routing_token"] == token
                    assert len([row for phase in ("pending", "running") for row in at_call["queue"].get(phase, [])
                                if row.get("id") == task_id]) == 1
                    assert "agent" not in at_call["previous_model_kinds"]
                    received = wait_until(lambda: _received(oracle, task_id), 15)
                    assert len(received) == 1, "first model call must already belong to the one durable root"
                    task = received[0]
                    ref = task["origin_message_ref"]
                    canonical_rows = [row for row in oracle._jsonl("logs/chat.jsonl")
                                      if row.get("direction") == "in" and row.get("client_message_id") == message_id]
                    assert len(canonical_rows) == 1
                    canonical = canonical_rows[0]
                    expected = canonical["text"]
                    # PR-1 preserves canonical ingress. Common web/bus trimming
                    # predates host admission and remains an explicit boundary.
                    assert expected == (caption if caption_only else message['content'].strip())
                    assert task["root_task_id"] == task_id and task["delegation_role"] == "root"
                    assert not task.get("_is_direct_chat") and not task.get("_ephemeral_turn")
                    assert task["objective"] == expected
                    assert task["chat_id"] == chat_id and str(task.get("project_id") or "") == str(project.get("id") or "")
                    assert task["metadata"]["force_plan"] is True
                    assert task["metadata"]["force_plan_source"] == "swarm"
                    assert task["origin_message_text"] == expected
                    assert ref["client_message_id"] == message_id and ref["chat_id"] == chat_id
                    assert ref == build_owner_message_ref(
                        chat_id=chat_id, client_message_id=message_id,
                        ts=canonical["ts"], text=expected,
                    )
                    assert all(task["metadata"]["client_surface"][key] == value for key, value in surface.items())
                    manifest = task["attachment_manifest"]
                    assert len(manifest) == len(payloads)
                    assert {row["label"]: Path(row["abs_path"]).read_bytes() for row in manifest} == payloads
                    assert len(_queued(oracle, task_id)) == 1
                    assert oracle.task_result(task_id)["promotion_admission"]["routing_token"] == token
                    assert len(_annotations(oracle, message_id)) == 1
                    assert _annotations(oracle, message_id)[0]["status"] == "scheduled"
                    assert "[SWARM_INITIATIVE]" in body_text(at_call["body"])
                    assert "[SWARM_ROUTING_INTENT]" not in body_text(at_call["body"])
                    if project:
                        assert project_binding_for_task(server.data_root, task_id)["source_ref"] == ref
                    accepted_echo = wait_until(lambda: _echoes(page, message_id), 15)
                    assert len(accepted_echo) == 1 and accepted_echo[0]["ingress_accepted"] is True
                    assert accepted_echo[0]["ts"] == canonical["ts"] and accepted_echo[0]["chat_id"] == chat_id
                    before = {"annotations": _annotations(oracle, message_id),
                              "admissions": oracle.supervisor_rows("promote_chat_to_task_admitted")}
                    _send(page, message)
                    # The same message under its id rejoins the accepted row: the server
                    # re-echoes that row (a mismatch would get the failure notice) and
                    # writes or dispatches nothing. This socket's second echo is the
                    # replay's own receipt; no fixed sleep can certify the negatives below.
                    replay_echoes = wait_until(lambda: rows if len(rows := _echoes(page, message_id)) >= 2 else None, 15)
                    assert replay_echoes == accepted_echo * 2, replay_echoes
                    assert len([r for r in oracle._jsonl("logs/chat.jsonl")
                                if r.get("client_message_id") == message_id and r.get("direction") == "in"]) == 1
                    assert {"annotations": _annotations(oracle, message_id),
                            "admissions": oracle.supervisor_rows("promote_chat_to_task_admitted")} == before
                    assert len(_queued(oracle, task_id)) == 1 and len(_received(oracle, task_id)) == 1
                    assert gate.held == 1 and not gate.release.is_set()
                    gate.release.set()
                    owner_wait = wait_until(lambda: (wait if (wait := oracle.task_result(task_id).get("owner_wait", {})).get("state") == "waiting" else None), 90)
                    assert owner_wait, "the admitted root must execute real work and reach its own question"
                    assert {"annotations": _annotations(oracle, message_id),
                            "admissions": oracle.supervisor_rows("promote_chat_to_task_admitted")} == before
                    assert len(_received(oracle, task_id)) == 1
                    tools = oracle.task_drive(task_id).tools_rows()
                    assert any(row.get("tool") == "list_files" or row.get("name") == "list_files" for row in tools)
                    card = page.locator(f'.chat-live-card[data-task-id="{task_id}"]')
                    card.wait_for(timeout=30000)
                    assert card.count() == 1
                    assert page.locator('.msg-routing-annotation[data-annotation-status="scheduled"]').count() == 1
                    assert _echoes(page, message_id) == replay_echoes, "a rejoined replay must never be redelivered"
                    page.screenshot(path=str(evidence / "live.png"), full_page=True, animations="disabled")
                    page.reload(wait_until="domcontentloaded")
                    page.wait_for_function("() => window.__testSockets?.[0]?.readyState === WebSocket.OPEN")
                    if project:
                        page.evaluate("project => window.dispatchEvent(new CustomEvent('ouro:open-project', {detail:{project}}))", project)
                    page.locator(f'.chat-live-card[data-task-id="{task_id}"]').wait_for(timeout=30000)
                    assert page.locator('.msg-routing-annotation[data-annotation-status="scheduled"]').count() == 1
                    page.screenshot(path=str(evidence / "reloaded.png"), full_page=True, animations="disabled")
                    assert not errors, errors
                    assert server.cancel_task(task_id)["status"] == 200
                    terminal = wait_durable_result(oracle, task_id, timeout=90)
                    assert terminal["status"] in {"cancelled", "failed"}
                    (evidence / "receipt.json").write_text(json.dumps({
                        "task": task, "first_model_entry": at_call, "admission": before,
                        "raw_ws_text": message["content"], "canonical_owner_row": canonical,
                        "ingress_journey": "direct_ws_caption" if caption_only else "actual_swarm_toggle_composer_file_input",
                        "composer_input_text": raw if not caption_only else None,
                        "actual_sent_frame_for_replay": message, "replay_echoes": replay_echoes,
                        "canonical_text_sha256": hashlib.sha256(expected.encode("utf-8")).hexdigest(),
                        "existing_ingress_normalization": "web/message_bus trims outer whitespace before canonical admission",
                        "owner_wait": owner_wait, "terminal_status": terminal["status"], "errors": errors,
                    }, ensure_ascii=False, indent=2), encoding="utf-8")
                finally:
                    gate.release.set()
                    browser.close()
        finally:
            gate.release.set()
            server.stop()
            assert server.proc.poll() is not None
            assert not gate.timed_out
