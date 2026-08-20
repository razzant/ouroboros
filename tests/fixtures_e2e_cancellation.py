"""Harness for the E1-E12 end-to-end owner-control scenarios.

Not a test module (pytest collects ``test_*.py`` only) — this is the machinery
``tests/test_e2e_cancellation_scenarios.py`` drives: the local stub model that lets a real
isolated server run an agent loop for free, a loopback request recorder for pinning the
driver's wire contract, the isolated settings builder, and the readers for the durable
artifacts every scenario asserts against. Same split as ``tests/fixtures_mock_llm.py`` and
``tests/_shared.py``; the scenario semantics and the paid-pass contract live in the test
module's docstring.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devtools.benchmarks.common.server_runner import IsolatedServer  # noqa: E402

LANE_MOCK = "mock"
LANE_PAID = "paid"

# The scenario inventory of S5_GAP_ANALYSIS.md §5, with the lane each one is driven in.
# The test module fails if an id loses its test — a scenario must be retired deliberately,
# not by deletion.
SCENARIOS = {
    "E1": ("delegate_start -> wait -> answer -> cancel", LANE_PAID),
    "E2": ("integrate_delegated_patch, clean", LANE_PAID),
    "E3": ("integrate_delegated_patch, conflicting", LANE_PAID),
    "E4": ("cancel, single", LANE_MOCK),
    "E5": ("cancel, cascade", LANE_MOCK),
    "E6": ("cancel x completion race", LANE_MOCK),
    "E7": ("kill/replay of delivery (owed outbox)", LANE_MOCK),
    "E8": ("budget-drain fail_tasks", LANE_PAID),
    "E9": ("boot migrate_legacy", LANE_MOCK),
    "E10": ("owner graceful stop (finalize_then_cancel)", LANE_MOCK),
    "E11": ("stop-now hardening mid-episode", LANE_MOCK),
    "E12": ("owner hurry", LANE_MOCK),
}

MOCK_SLUG = "openai-compatible::mock-model"


# ---------------------------------------------------------------------------
# Opt-in gate
# ---------------------------------------------------------------------------

def lane_enabled(lane: str) -> bool:
    selected = str(os.environ.get("OUROBOROS_E2E_CANCEL") or "").strip().lower()
    if lane == LANE_MOCK:
        return selected in {LANE_MOCK, LANE_PAID}
    return selected == LANE_PAID


def require_lane(lane: str) -> None:
    if not lane_enabled(lane):
        pytest.skip(
            f"set OUROBOROS_E2E_CANCEL={lane} to run the {lane} E2E cancellation lane "
            "(spawns a real isolated server; see test_e2e_cancellation_scenarios.py)"
        )


# ---------------------------------------------------------------------------
# The local stub model: an OpenAI-compatible endpoint on loopback
# ---------------------------------------------------------------------------

class StubModelServer:
    """Keep-alive OpenAI-compatible stub model.

    ``mode`` drives the agent loop from the test process (the stub runs in-process, so a
    scenario just assigns the attribute):

    - ``keepalive`` — answer every tool-bearing call with ``list_files`` (a POLICY_SKIP
      read-only tool: no safety model call, no side effect), so the task stays RUNNING
      until the scenario cancels it.
    - ``spawn``     — the FIRST tool-bearing call schedules one read-only subagent, then
      keepalive; this is how a live subtree exists for the cascade scenario.
    - ``finish``    — answer with plain text and no tool call, the loop's final-answer path.

    A JSON-object ``response_format`` request is the safety supervisor's shape; it always
    gets a SAFE verdict so a scenario can also run with safety on.
    """

    def __init__(self, *, mode: str = "keepalive", latency_sec: float = 0.0) -> None:
        self.mode = mode
        self.latency_sec = latency_sec
        self.calls: list = []
        self.spawned = 0
        outer = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - stdlib callback name
                if self.path.rstrip("/").endswith("/models"):
                    return self._send({"data": [{"id": "mock-model", "max_model_len": 400000}]})
                self.send_error(404)

            def do_POST(self):  # noqa: N802 - stdlib callback name
                length = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads((self.rfile.read(length) or b"{}").decode("utf-8"))
                except ValueError:
                    body = {}
                if not isinstance(body, dict):
                    body = {}
                outer.calls.append(body)
                if outer.latency_sec:
                    time.sleep(outer.latency_sec)
                return self._send(outer._completion(body, len(outer.calls)))

            def _send(self, payload):
                data = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *_args):
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @staticmethod
    def _is_finalization_turn(body: dict) -> bool:
        """The runtime's forced-finalization turns are self-identifying in the prompt.

        A stub that kept emitting tool calls through them would make the owner-stop
        outcome depend on scheduling luck: whether the grace window happened to expire
        before the agent produced anything. Answering these turns with a tool-less final
        answer is what a compliant agent does, and it makes the scenario deterministic.
        """
        for message in body.get("messages") or []:
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, str) and ("[OWNER_STOP]" in content or "[FINALIZE_NOW]" in content):
                return True
        return False

    def _completion(self, body: dict, seq: int) -> dict:
        fmt = body.get("response_format")
        if isinstance(fmt, dict) and fmt.get("type") == "json_object":
            message = {"role": "assistant",
                       "content": json.dumps({"status": "SAFE", "reason": "stub"})}
        elif self._is_finalization_turn(body):
            message = {"role": "assistant",
                       "content": "Final answer: the repository root was listed; stopping as asked."}
        elif body.get("tools") and self.mode != "finish":
            names = {
                (tool.get("function") or {}).get("name")
                for tool in body.get("tools") or [] if isinstance(tool, dict)
            }
            if self.mode == "spawn" and self.spawned < 1 and "schedule_subagent" in names:
                self.spawned += 1
                call = {"name": "schedule_subagent", "arguments": json.dumps({
                    "objective": "List the repository root and report what is there.",
                    "expected_output": "A list of file names.",
                })}
            else:
                call = {"name": "list_files", "arguments": json.dumps({"path": "."})}
            message = {"role": "assistant", "content": "still working",
                       "tool_calls": [{"id": f"call_{seq}", "type": "function", "function": call}]}
        else:
            message = {"role": "assistant", "content": "Done."}
        return {
            "id": f"stub-{seq}",
            "object": "chat.completion",
            "model": str(body.get("model") or "mock-model"),
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def __enter__(self) -> "StubModelServer":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._server.shutdown()
        self._server.server_close()


class RecordingEndpoint:
    """A loopback HTTP recorder: captures ``(method, path, body)`` and answers a scripted
    status/payload. Pins the DRIVER's wire contract without a runtime behind it."""

    def __init__(self, status: int = 200, payload: dict | None = None) -> None:
        self.status = status
        self.payload = dict(payload or {"ok": True})
        self.requests: list = []
        outer = self

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - stdlib callback name
                length = int(self.headers.get("Content-Length") or 0)
                raw = (self.rfile.read(length) if length else b"")
                try:
                    parsed = json.loads(raw.decode("utf-8")) if raw.strip() else None
                except ValueError:
                    parsed = {"__unparseable__": raw.decode("utf-8", "replace")}
                outer.requests.append({"method": "POST", "path": self.path, "body": parsed})
                data = json.dumps(outer.payload).encode("utf-8")
                self.send_response(outer.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *_args):
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def __enter__(self) -> "RecordingEndpoint":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._server.shutdown()
        self._server.server_close()


def driver_at(recorder: RecordingEndpoint) -> IsolatedServer:
    """An ``IsolatedServer`` whose HTTP calls land on the recorder instead of a runtime."""
    server = IsolatedServer(pathlib.Path("/nonexistent-clone"),
                            pathlib.Path("/nonexistent-data"),
                            pathlib.Path("/nonexistent-settings.json"))
    server.base_url = recorder.base_url
    return server


# ---------------------------------------------------------------------------
# Isolated server construction
# ---------------------------------------------------------------------------

def paid_model_and_key() -> tuple:
    """Resolve the paid lane's model slug and credential BY NAME (never by value)."""
    model = str(os.environ.get("OUROBOROS_E2E_PAID_MODEL") or "").strip()
    key_env = str(os.environ.get("OUROBOROS_E2E_PAID_KEY_ENV") or "").strip()
    if not model or not key_env:
        pytest.skip(
            "paid lane needs OUROBOROS_E2E_PAID_MODEL (exact slug) and "
            "OUROBOROS_E2E_PAID_KEY_ENV (the NAME of the env var holding the key)"
        )
    value = os.environ.get(key_env)
    if not value:
        pytest.skip(f"paid lane: env var {key_env!r} named by OUROBOROS_E2E_PAID_KEY_ENV is empty")
    return model, key_env, value


def isolated_settings(*, stub: StubModelServer | None, paid: bool = False, **overrides) -> dict:
    """The isolated settings.json for a scenario server.

    Every routed model slot is pinned explicitly. An UN-prefixed slug routes to OpenRouter
    by default, so a slot left at its packaged default would be a live-egress attempt from
    a lane that promises not to make one — hence the exhaustive list rather than a few
    interesting keys.
    """
    cfg: dict = {
        "OUROBOROS_MODEL_FALLBACKS": "",
        "OUROBOROS_MODEL_VISION": "",
        "OUROBOROS_MODEL_CONSCIOUSNESS": "",
        "OUROBOROS_MODEL_DEEP_SELF_REVIEW": "",
        "OUROBOROS_WEBSEARCH_MODEL": "",
        "CLAUDE_CODE_MODEL": "",
        # Disk-authored keys: config.apply_settings_to_env cannot author these from the
        # environment, so they have to be in the file, written fresh (both carry a
        # lowering ratchet against the previous file value).
        "OUROBOROS_SAFETY_MODE": "off",
        "OUROBOROS_CONTEXT_MODE": "low",
        "OUROBOROS_RUNTIME_MODE": "light",
        "OUROBOROS_TASK_REVIEW_MODE": "off",
        "OUROBOROS_POST_TASK_EVOLUTION": "false",
        "OUROBOROS_MAX_WORKERS": 4,
        "TOTAL_BUDGET": 10.0,
        "OUROBOROS_PER_TASK_COST_USD": 10.0,
    }
    if paid:
        model, key_env, value = paid_model_and_key()
        cfg[key_env] = value
        slug = model
    else:
        assert stub is not None
        cfg["OPENAI_COMPATIBLE_BASE_URL"] = stub.base_url
        cfg["OPENAI_COMPATIBLE_API_KEY"] = "stub-key-not-a-credential"
        slug = MOCK_SLUG
    for slot in ("OUROBOROS_MODEL", "OUROBOROS_MODEL_HEAVY", "OUROBOROS_MODEL_LIGHT",
                 "OUROBOROS_REVIEW_MODELS", "OUROBOROS_SCOPE_REVIEW_MODELS",
                 "OUROBOROS_SCOPE_REVIEW_MODEL"):
        cfg[slot] = slug
    cfg.update(overrides)
    return cfg


def clone_repo(destination: pathlib.Path) -> pathlib.Path:
    """One throwaway clone of the checkout under test.

    A clone (not the working tree) is what the runtime is allowed to run against: the
    server owns its repo directory, so an E2E server must never be pointed at a live
    worktree.
    """
    clone = pathlib.Path(destination) / "clone"
    subprocess.run(["git", "clone", "--no-hardlinks", "-q", str(REPO_ROOT), str(clone)],
                   check=True, capture_output=True)
    subprocess.run(["git", "checkout", "-B", "ouroboros"], cwd=str(clone),
                   check=True, capture_output=True)
    subprocess.run(["git", "remote", "remove", "origin"], cwd=str(clone),
                   check=False, capture_output=True)
    return clone


def write_settings_file(settings_path: pathlib.Path, settings: dict) -> None:
    """The paid lane's settings carry a live API key on a shared host: the file
    must exist at 0600 BEFORE the key bytes land (a default-umask write_text
    briefly published a live key world-readable)."""
    fd = os.open(settings_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)  # O_CREAT's mode only applies on creation; an existing wider file keeps its bits
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(settings, indent=2))


def start_server(clone, root, settings: dict, *, ready_timeout: float = 300) -> IsolatedServer:
    data_root = pathlib.Path(root) / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    settings_path = data_root / "settings.json"
    write_settings_file(settings_path, settings)
    server = IsolatedServer(clone, data_root, settings_path)
    server.start(ready_timeout=ready_timeout)
    return server


# ---------------------------------------------------------------------------
# Readers of the durable artifacts every scenario asserts against
# ---------------------------------------------------------------------------

def intents(data_root) -> dict:
    path = pathlib.Path(data_root) / "state" / "cancel_intents.json"
    if not path.exists():
        return {}
    blob = json.loads(path.read_text(encoding="utf-8"))
    return blob.get("intents") if isinstance(blob.get("intents"), dict) else {}


def forensics(data_root, *, task_id: str = "", event: str = "") -> list:
    """``cancel_intent`` rows from logs/supervisor.jsonl, optionally filtered."""
    path = pathlib.Path(data_root) / "logs" / "supervisor.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or "cancel_intent" not in line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("type") != "cancel_intent":
            continue
        if task_id and str(row.get("task_id") or "") != task_id:
            continue
        if event and str(row.get("event") or "") != event:
            continue
        rows.append(row)
    return rows


def events(data_root, event_type: str) -> list:
    path = pathlib.Path(data_root) / "logs" / "events.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or event_type not in line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("type") == event_type:
            rows.append(row)
    return rows


def task_result(data_root, task_id: str) -> dict:
    path = pathlib.Path(data_root) / "task_results" / f"{task_id}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def task_result_bytes(data_root, task_id: str) -> bytes:
    return (pathlib.Path(data_root) / "task_results" / f"{task_id}.json").read_bytes()


def chat_bytes(data_root) -> bytes:
    path = pathlib.Path(data_root) / "logs" / "chat.jsonl"
    return path.read_bytes() if path.exists() else b""


def queue_snapshot(data_root) -> dict:
    path = pathlib.Path(data_root) / "state" / "queue_snapshot.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def wait_until(predicate, timeout: float, interval: float = 0.5):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(interval)
    return last


def submit_running(server: IsolatedServer, description: str, *, timeout: float = 120) -> str:
    """Submit a task and wait until the supervisor actually has it RUNNING — a scenario
    that cancels a task still sitting in PENDING would assert a different protocol path
    than the one it names."""
    task_id = server.submit(description)
    assert task_id, "submit returned no task id"
    running = wait_until(
        lambda: any(
            str(row.get("id") or "") == task_id
            for row in (queue_snapshot(server.data_root).get("running") or [])
        ),
        timeout,
    )
    assert running, f"task {task_id} never reached the RUNNING set"
    return task_id
