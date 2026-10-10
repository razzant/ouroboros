"""A required owner answer preserves a real form while another worker runs.

The existing keyless system harness owns the real server, pool and model wire.
Only model judgment is scripted: browser tools, escalation, queue transitions,
Stop custody and mailbox/quiz delivery all execute their production paths.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from devtools.benchmarks.common.server_runner import _api
from ouroboros.owner_mailbox import write_owner_message
from tests.candidate_checkout import candidate_checkout
from tests.system_e2e.harness import (
    ArtifactOracle,
    ModelGate,
    ScriptedStubModel,
    body_text,
    keyless_settings,
    start_server,
    wait_durable_result,
    wait_until,
)
from tests._usage_store_testing import ledger_rows


pytestmark = [pytest.mark.serial, pytest.mark.browser]
_FORM_TASK = "OWNER_WAIT_FORM_A"
_OTHER_TASK = "OWNER_WAIT_BROWSER_B"
_OWNER_ANSWER = "Keep the filled form and submit exactly that draft now."
_DRAFT = "A draft that exists only in this live browser form"


@pytest.fixture
def local_form():
    """An ordinary local form: a fresh document has a new identity and no draft."""
    requests = []
    lock = threading.Lock()
    html = b'''<!doctype html><meta charset="utf-8"><title>Owner wait form</title>
        <style>body{font:22px system-ui;background:#edf4f2;color:#172a23;padding:48px}
        main{max-width:720px}label,input,button{display:block;margin:20px 0}
        input{font:18px system-ui;width:650px;padding:12px}button{font:18px system-ui;padding:10px}
        #status{font-weight:600}</style><main><h1>Preserved owner draft</h1>
        <label for="draft">Unsubmitted form</label><input id="draft" autocomplete="off">
        <button id="save">Save draft</button><button id="submit">Submit draft</button>
        <p id="status">Waiting for a draft</p></main><script>
        const instance = crypto.randomUUID();
        for (const operation of ['save','submit']) {
          document.getElementById(operation).onclick = async () => {
            const result = await fetch('/' + operation, {method:'POST',
              body:JSON.stringify({instance,value:document.getElementById('draft').value})});
            document.getElementById('status').textContent = await result.text();
          };
        }</script>'''

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            with lock:
                requests.append({"method": "GET", "path": self.path})
            payload = b"<h1>Independent browser ready</h1>" if self.path == "/independent" else html
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):  # noqa: N802
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                requests.append({"method": "POST", "path": self.path, **payload})
            body = b"Draft saved, awaiting owner" if self.path == "/save" else b"Draft submitted once"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.fixture
def wait_clone(tmp_path):
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import Error, sync_playwright
    from ouroboros.tools.browser import _set_playwright_browsers_path_if_bundled

    _set_playwright_browsers_path_if_bundled()
    try:
        # launch() never downloads an engine, and completing a real launch also
        # settles the Playwright driver before this preflight context closes.
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            browser.close()
    except Error as exc:
        if os.environ.get("OUROBOROS_EXPECT_BROWSER_ENGINES"):
            pytest.fail(str(exc))
        pytest.skip(str(exc))
    source = Path(__file__).resolve().parents[1]
    with candidate_checkout(source, tmp_path / "clone", origin_proof=True) as candidate:
        yield candidate


def _running(oracle, task_id):
    return next((row for row in oracle.queue_snapshot().get("running", [])
                 if row.get("id") == task_id), {})


def _rounds(stub, marker):
    with stub._lock:
        return [body for _, body in stub.calls
                if body.get("tools") and marker in body_text(body)]


def _tool_messages(body):
    return [message for message in body.get("messages", []) if message.get("role") == "tool"]


@pytest.mark.parametrize("answer_path", ["ordinary_text", "quiz"])
def test_owner_wait_preserves_form_and_lends_one_worker(wait_clone, local_form, tmp_path, answer_path):
    origin, requests = local_form
    # B owns a SECOND real browser and is held only after its navigation completed.
    # This gives Stop a concrete worker/browser tree to tear down while A is parked.
    gate = ModelGate(lambda body: bool(body.get("tools"))
                     and _OTHER_TASK in body_text(body)
                     and any("Independent browser ready" in str(row.get("content"))
                             for row in _tool_messages(body)), timeout=180)
    steps = [
        {"tool": "browse_page", "arguments": {"url": origin, "engine": "chromium"}},
        {"tool": "browser_action", "arguments": {"action": "fill", "selector": "#draft", "value": _DRAFT}},
        {"tool": "browser_action", "arguments": {"action": "click", "selector": "#save"}},
        {"tool": "browser_action", "arguments": {"action": "screenshot"}},
        {"tool": "send_photo", "arguments": {"image_base64": "__last_screenshot__", "caption": "Before owner wait"}},
        {"tool": "escalate", "arguments": {"question": "Submit the saved draft?",
            "options": [{"label": "Submit"}, {"label": "Keep editing"}],
            "stake": "The filled form must remain open until the owner answers.", "wait_for_answer": True}},
        {"tool": "browser_action", "arguments": {"action": "click", "selector": "#submit"}},
        {"tool": "browser_action", "arguments": {"action": "screenshot"}},
        {"tool": "send_photo", "arguments": {"image_base64": "__last_screenshot__", "caption": "After owner answer"}},
        {"final": "The original draft was submitted once."},
    ]
    indices = {_FORM_TASK: 0, _OTHER_TASK: 0}

    def next_step(body):
        actor = _OTHER_TASK if _OTHER_TASK in body_text(body) else _FORM_TASK
        index = indices[actor]
        indices[actor] += 1
        if actor == _OTHER_TASK:
            return ({"tool": "browse_page", "arguments": {"url": origin + "/independent"}}
                    if index == 0 else {"final": "Independent browser work stopped."})
        if index == 6:
            assert _OWNER_ANSWER in body_text(body), "the resumed round lost the addressed owner answer"
        assert index < len(steps), "the task replayed or took an unexpected extra model round"
        return steps[index]

    with ScriptedStubModel([next_step] * 20, gate=gate) as stub:
        server = start_server(wait_clone, tmp_path / "instance", keyless_settings(stub, OUROBOROS_MAX_WORKERS=1))
        oracle = ArtifactOracle(server.data_root)
        try:
            task_a = server.submit(f"[{_FORM_TASK}] Fill the local form, save its draft, ask before submitting.")
            parked = wait_until(lambda: (row if ((row := _running(oracle, task_a)).get("owner_wait") or {}).get("state") == "waiting" else None), 120)
            assert parked, f"A never parked: {oracle.task_result(task_a)}"
            wait = parked["owner_wait"]
            assert wait["source_ref"] and wait["task_attempt"] == parked["attempt"]
            assert len(_rounds(stub, _FORM_TASK)) == 6
            assert _api(server.base_url, "GET", f"/api/tasks/{task_a}")["status"] == "running"
            assert [row["path"] for row in requests if row["method"] == "POST"] == ["/save"]
            assert not oracle.task_drive(task_a).events("task_done")

            task_b = server.submit(f"[{_OTHER_TASK}] Open the independent local browser page.")
            assert gate.arrived.wait(120), f"B never executed in the lent slot: {oracle.queue_snapshot()}"
            both = oracle.queue_snapshot()
            assert {task_a, task_b} <= oracle.running_ids()
            assert both["active_worker_count"] == 1 and both["parked_worker_count"] == 1
            assert _running(oracle, task_b)["worker_id"] != parked["worker_id"]
            assert len(_rounds(stub, _FORM_TASK)) == 6, "parked A bought an empty round while B ran"

            stopped = server.cancel_task(task_b)
            assert stopped["status"] == 200, stopped
            assert wait_durable_result(oracle, task_b)["status"] == "cancelled"
            gate.release.set()
            assert wait_until(lambda: indices[_OTHER_TASK] == 2, 10)
            assert _running(oracle, task_a)["owner_wait"]["state"] == "waiting"
            assert len(_rounds(stub, _FORM_TASK)) == 6, "Stop B woke unrelated A"

            if answer_path == "quiz":
                answer = _api(server.base_url, "POST", "/api/decisions", {
                    "decision_id": f"quiz:{task_a}:{wait['quiz_id']}",
                    "request_id": "answer-the-original-form", "option_index": 0, "comment": _OWNER_ANSWER,
                })
                assert answer["ok"], answer
            else:
                assert write_owner_message(Path(wait["execution_drive_root"]), _OWNER_ANSWER,
                                           task_id=task_a, msg_id="answer-the-original-form")

            final = wait_durable_result(oracle, task_a, timeout=120)
            assert final["status"] == "completed", final
            assert final["owner_wait"]["state"] == "resumed"
            assert final["owner_wait"]["wait_id"] == wait["wait_id"]
            assert final["owner_wait"]["task_attempt"] == parked["attempt"]
            assert final["owner_wait"]["started_at"] == parked["started_at"]
            assert len(_rounds(stub, _FORM_TASK)) == len(steps)
            assert indices == {_FORM_TASK: len(steps), _OTHER_TASK: 2}
            effects = [row for row in requests if row["method"] == "POST"]
            assert [row["path"] for row in effects] == ["/save", "/submit"]
            assert effects[0]["instance"] == effects[1]["instance"]
            assert effects[0]["value"] == effects[1]["value"] == _DRAFT
            assert len([row for row in requests if row["method"] == "GET" and row["path"] == "/"]) == 1
            assert len(oracle.task_drive(task_a).events("task_received")) == 1
            assert wait_until(lambda: task_a not in oracle.running_ids(), 30)
            assert gate.timed_out is False
            usage = [row for row in ledger_rows(server.data_root)
                     if row.get("task_id") == task_a and row.get("category") == "task"]
            # One current row per attempt: every step's attempt went the whole way.
            attempts = [row["attempt_id"] for row in usage if row.get("state") == "settled"]
            assert len(attempts) == len(set(attempts)) == len(steps) == len(usage)
            if output := os.environ.get("OUROBOROS_BROWSER_EVIDENCE_OUT"):
                destination = Path(output) / answer_path
                destination.mkdir(parents=True, exist_ok=True)
                for path in (server.data_root / "task_results" / "artifacts" / task_a).rglob("*.png"):
                    shutil.copyfile(path, destination / path.name)
                (destination / "receipt.json").write_text(json.dumps({
                    "task_id": task_a, "other_task_id": task_b, "parked": parked,
                    "effects": effects, "rounds": indices, "result": final,
                }, indent=2), encoding="utf-8")
        finally:
            gate.release.set()
            server.stop()
            assert server.proc.poll() is not None
