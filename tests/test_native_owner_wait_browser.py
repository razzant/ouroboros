"""Ordinary Main/Project work keeps its real browser while the owner answers."""

import json
import os
import shutil
import subprocess
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest

from devtools.benchmarks.common.server_runner import _api
from ouroboros.owner_mailbox import write_owner_message
from tests.test_owner_wait_integration import local_form as form_fixture, wait_clone as clone_fixture
from tests.system_e2e.harness import (
    ArtifactOracle, ScriptedStubModel, body_text, keyless_settings,
    start_server, wait_durable_result, wait_until, ws_url,
)

pytestmark = [pytest.mark.browser]
local_form = form_fixture
wait_clone = clone_fixture


@contextmanager
def chat_connection(server):
    """Consume live frames like the UI; an unread client stalls WS flow control."""
    from websockets.exceptions import ConnectionClosed
    from websockets.sync.client import connect

    stopped, errors = threading.Event(), []
    with connect(ws_url(server), open_timeout=30, proxy=None) as ws:
        def receive():
            while not stopped.is_set():
                try:
                    ws.recv(timeout=1)
                except TimeoutError:
                    continue
                except ConnectionClosed as exc:
                    if not stopped.is_set():
                        errors.append(exc)
                    return
        reader = threading.Thread(target=receive)
        reader.start()
        try:
            yield ws
        finally:
            stopped.set()
            ws.close()
            reader.join(5)
            assert not reader.is_alive()
            assert not errors, errors


@pytest.mark.parametrize("surface", ["main", "project"])
@pytest.mark.parametrize("answer_path", ["text", "quiz", "stop"])
def test_native_owner_wait_retains_form_and_remains_addressable(
    wait_clone, local_form, tmp_path, surface, answer_path,
):
    origin, requests = local_form
    marker, other = "NATIVE_FORM_WAIT", "INDEPENDENT_NATIVE_TURN"
    draft, answer = "Keep this live native draft", "Submit that same saved draft now."
    steps = [
        {"tool": "browse_page", "arguments": {"url": origin}},
        {"tool": "browser_action", "arguments": {"action": "fill", "selector": "#draft", "value": draft}},
        {"tool": "browser_action", "arguments": {"action": "click", "selector": "#save"}},
        {"tool": "browser_action", "arguments": {"action": "screenshot"}},
        {"tool": "send_photo", "arguments": {"image_base64": "__last_screenshot__", "caption": "Native form before waiting"}},
        {"tool": "escalate", "arguments": {"question": "Submit this draft?",
            "options": [{"label": "Submit"}, {"label": "Edit"}], "wait_for_answer": True}},
        {"tool": "browser_action", "arguments": {"action": "click", "selector": "#submit"}},
        {"tool": "browser_action", "arguments": {"action": "screenshot"}},
        {"tool": "send_photo", "arguments": {"image_base64": "__last_screenshot__", "caption": "Native form after answer"}},
        {"final": "Submitted the original native form once."},
    ]
    calls = {marker: 0, other: 0}

    def response(body):
        actor = other if other in body_text(body) else marker
        index = calls[actor]
        calls[actor] += 1
        if actor == other:
            return ({"tool": "list_files", "arguments": {"root": "system_repo", "path": "."}}
                    if index == 0 else {"final": "Independent ordinary work completed."})
        if index == 6:
            assert answer in body_text(body), "native continuation lost the addressed owner answer"
        assert index < len(steps), "completed native effects were replayed"
        return steps[index]

    with ScriptedStubModel([response] * 20) as stub:
        server = start_server(wait_clone, tmp_path / "instance", keyless_settings(stub, OUROBOROS_MAX_WORKERS=1))
        oracle = ArtifactOracle(server.data_root)
        try:
            project = {}
            if surface == "project":
                folder = tmp_path / "project-folder"
                folder.mkdir()
                subprocess.run(["git", "init", "-q", str(folder)], check=True, capture_output=True)
                project = _api(server.base_url, "POST", "/api/projects", {
                    "name": "Native waiting", "path": str(folder),
                })["project"]
            with chat_connection(server) as ws:
                def submit(text, *, in_project=False):
                    message_id = uuid.uuid4().hex
                    ws.send(json.dumps({"type": "chat", "content": text, "client_message_id": message_id,
                        "chat_id": project["chat_id"] if in_project else 1,
                        **({"project_id": project["id"]} if in_project else {})}))
                    task = wait_until(lambda: next((row["task"] for row in oracle.events("task_received")
                        if (row.get("task", {}).get("metadata", {}).get("origin_message_ref") or {}).get("client_message_id") == message_id), None), 90)
                    assert task and task.get("_is_direct_chat")
                    return task["id"]

                task_id = submit(f"[{marker}] Fill and save the form, then wait for my answer before submitting.", in_project=bool(project))
                wait = wait_until(lambda: (row if (row := (oracle.task_result(task_id).get("owner_wait") or {})).get("state") == "waiting" else None), 120)
                assert wait, oracle.task_result(task_id)
                assert calls[marker] == 6
                assert task_id not in oracle.running_ids()  # no synthetic pool admission
                live = _api(server.base_url, "GET", "/api/state")["active_direct_turns"]
                assert task_id in {row["activity_id"] for row in live}
                assert not oracle.task_drive(task_id).events("task_done")
                other_id = submit(f"[{other}] List the repository and finish independently.")
                assert wait_durable_result(oracle, other_id)["status"] == "completed"
                assert calls[marker] == 6 and calls[other] == 2
                assert [row["path"] for row in requests if row["method"] == "POST"] == ["/save"]
                if answer_path == "stop":
                    cancelled = server.cancel_task(task_id)
                    assert cancelled["status"] == 200, cancelled
                elif answer_path == "quiz":
                    answered = _api(server.base_url, "POST", "/api/decisions", {
                        "decision_id": f"quiz:{task_id}:{wait['quiz_id']}",
                        "request_id": "native-owner-answer", "comment": answer,
                    })
                    assert answered["ok"], answered
                else:
                    assert write_owner_message(Path(wait["execution_drive_root"]), answer,
                                               task_id, msg_id="native-owner-answer")
                final = wait_durable_result(oracle, task_id, timeout=120)
                assert final["owner_wait"]["wait_id"] == wait["wait_id"]
                assert final["owner_wait"]["state"] == "resumed"
                assert wait_until(lambda: task_id not in {r["activity_id"] for r in
                    _api(server.base_url, "GET", "/api/state")["active_direct_turns"]}, 30)
                effects = [row for row in requests if row["method"] == "POST"]
                if answer_path == "stop":
                    assert final["status"] in {"cancelled", "failed"}, final
                    assert calls[marker] == 6 and len(effects) == 1  # no paid finalizer or submit
                else:
                    assert final["status"] == "completed", final
                    assert calls[marker] == len(steps)
                    assert [row["path"] for row in effects] == ["/save", "/submit"]
                    assert effects[0]["instance"] == effects[1]["instance"]
                    assert effects[0]["value"] == effects[1]["value"] == draft
                assert len([row for row in requests if row["method"] == "GET" and row["path"] == "/"]) == 1
                assert len([row for row in oracle.task_drive(task_id).events("task_received")
                            if row.get("task", {}).get("id") == task_id]) == 1
                if output := os.environ.get("OUROBOROS_BROWSER_EVIDENCE_OUT"):
                    dest = Path(output) / f"native-{surface}-{answer_path}"
                    dest.mkdir(parents=True, exist_ok=True)
                    for path in (server.data_root / "task_results/artifacts" / task_id).rglob("*.png"):
                        shutil.copyfile(path, dest / path.name)
                    (dest / "receipt.json").write_text(json.dumps({
                        "task_id": task_id, "other_task_id": other_id, "wait": wait,
                        "effects": effects, "calls": calls, "result": final,
                    }, indent=2), encoding="utf-8")
        finally:
            server.stop()
            assert server.proc.poll() is not None
