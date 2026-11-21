"""Opt-in, keyless macOS qualification of a capture that belongs to a real task.

The loopback model holds a request after receiving the worker's tool result.
Pytest owns the windows and observes the worker and capture process from outside
both. All exit checks precede fixture teardown. System-wide Panic has no case
here because it also tears down the services needed for these observations.
"""

from __future__ import annotations

import contextlib
import queue
import threading
import uuid

import pytest

from devtools.benchmarks.common.server_runner import _api_status, seed_owner_state
from tests._shared import live_worker_slots
from tests._window_capture_observer import (
    ProcessIdentity,
    capture_server,
    confirm_idle_window_stop,
    pyobjc_mapped,
    require_capture_grant,
    wait_for,
    window_key,
    window_rows,
)
from tests.system_e2e.harness import (
    ArtifactOracle,
    ModelGate,
    ScriptedStubModel,
    classify_call,
    keyless_settings,
    message_text,
    start_server,
    submit_running,
    wait_durable_result,
)
from tests.test_mcp_result_images import _pixels
from tests.test_unix_computer_use_stream_native import (
    BEGIN,
    FRAME,
    GREEN,
    SERVER,
    capture_only,
    connected_skill,
    desktop_only,
    mcp_facts,
    owned_windows,
    pixels_of_target,
    tool,
)

pytestmark = [desktop_only, pytest.mark.integration]


class CaptureResults:
    """Hold each new capture tool result at the model wire until its observer releases it."""

    def __init__(self):
        self.incoming = queue.Queue()
        self.gates = {}
        self.lock = threading.Lock()
        self.closed = False

    def __call__(self, body):
        if not body.get("tools") or classify_call(body) != "agent":
            return
        messages = body["messages"]
        # An image follows the tool message. Do not pick an older result from
        # a later round, or an old image retained elsewhere in model history.
        idx = next((i for i in reversed(range(len(messages)))
                    if messages[i].get("role") in {"tool", "assistant"}), None)
        if idx is None or messages[idx].get("role") != "tool":
            return
        call_id = messages[idx]["tool_call_id"]
        assistant = next((m for m in reversed(messages[:idx]) if m.get("role") == "assistant"), {})
        name = next((c["function"]["name"] for c in assistant.get("tool_calls", []) if c["id"] == call_id), "")
        if name not in {tool(BEGIN), tool(FRAME)}:
            return
        with self.lock:
            if self.closed:
                return
            if call_id in self.gates:
                # The finish door may present an already observed result again.
                # A concurrent retry must never release the held response.
                assert self.gates[call_id].release.is_set(), "duplicate request while its result is held"
                return
            gate = self.gates[call_id] = ModelGate(lambda _: True, timeout=240)
            self.incoming.put((name, messages[idx:], gate))
        gate(body)

    def take(self, method):
        name, tail, gate = self.incoming.get(timeout=180)
        assert gate.arrived.wait(5) and name == tool(method)
        label = f"External MCP tool result from {SERVER!r}/{method!r}."
        text = message_text(tail[0])
        assert label in text, "worker did not return a result from the selected MCP tool"
        return mcp_facts(text[text.index(label):]), _pixels(tail), gate.release

    def close(self):
        with self.lock:
            self.closed = True
            for gate in self.gates.values():
                gate.release.set()


def carrier(oracle, task_id):
    row = oracle.running_row(task_id)
    assert row.get("worker_id") is not None, "this task has no running worker"
    pid = live_worker_slots(oracle.data_root)[int(row["worker_id"])]
    return ProcessIdentity.live(pid)


@pytest.mark.parametrize("ending", ["finish", "immediate_stop", "window_gone"])
@pytest.mark.timeout(900)
def test_managed_window_task_lifetime(e2e_clone, tmp_path, monkeypatch, record_property, ending):
    require_capture_grant()
    with owned_windows(tmp_path) as (target, occluder), contextlib.ExitStack() as resources:
        turns = CaptureResults()
        main_reply = threading.Event()
        final = "Synthetic window observation complete."
        probe_final = "Fresh window session checked."

        def finish(_body):
            main_reply.set()
            return {"final": final}

        frame = {"tool": tool(FRAME), "arguments": {"wait_ms": 3000}}
        steps = [
            frame,
            {"tool": tool(BEGIN), "arguments": {"app": f"pid:{target.pid}", "window": target.title}},
            frame,
            *([{"tool": tool(FRAME), "arguments": {}}] if ending == "window_gone" else []),
            finish,
            frame, {"final": probe_final},  # a second, independent managed task
        ]
        model = resources.enter_context(ScriptedStubModel(steps, gate=turns, select_retained=True))
        settings = keyless_settings(model, OUROBOROS_RUNTIME_MODE="advanced", OUROBOROS_IMAGE_INPUT_MODE="inline",
                                    OUROBOROS_MAX_WORKERS=1)
        root = tmp_path / "service"
        drive = root / "data"
        drive.mkdir(parents=True)
        seed_owner_state(drive)
        settings, entry = resources.enter_context(connected_skill(drive, e2e_clone, settings, monkeypatch, record_property))
        service = start_server(e2e_clone, root, settings)
        resources.callback(service.stop)
        resources.callback(turns.close)
        # Use the serving process's own manager and published catalog as well.
        refresh = _api_status(service.base_url, "POST", "/api/mcp/refresh", {"server_id": SERVER})
        assert refresh["status"] == 200 and refresh["body"].get("ok") is True
        host = ProcessIdentity.live(service.proc.pid)
        oracle = ArtifactOracle(drive)
        task = submit_running(service, "Observe the synthetic window and report its color.")
        idle, images, next_round = turns.take(FRAME)
        assert idle.get("reason") == "no_session" and not images
        worker = carrier(oracle, task)
        worker.record(record_property, "worker")
        producer = capture_server(worker.pid, entry)
        producer.record(record_property, "server")
        assert not pyobjc_mapped(producer)
        next_round.set()

        begun, images, next_round = turns.take(BEGIN)
        assert begun.get("ok") and begun.get("state") == "live" and not images
        assert (begun["app"]["pid"], begun["window"]["window_id"]) == window_key(target)
        assert capture_server(worker.pid, entry) == producer
        next_round.set()

        live, images, held = turns.take(FRAME)
        assert live.get("ok") and live.get("image") is True and live["state"] == "live" and live["seq"] > 0
        assert (live["window"]["pid"], live["window"]["window_id"]) == window_key(target)
        capture_only(live)
        image, = images
        pixels_of_target(image, GREEN, record_property, "model")
        assert carrier(oracle, task) == worker and capture_server(worker.pid, entry) == producer
        assert pyobjc_mapped(producer)
        assert window_key(target) in window_rows(on_screen=True)
        assert not main_reply.is_set()
        record_property("frame_seq", live["seq"])

        if ending == "window_gone":
            def first_terminal_frame():
                nonlocal held
                held.set()  # only now may the model return its next frame call
                gone, images, held = turns.take(FRAME)
                return gone, images

            confirm_idle_window_stop(target, occluder, first_terminal_frame, record_property)
            assert carrier(oracle, task) == worker and capture_server(worker.pid, entry) == producer
            assert not main_reply.is_set()

        if ending == "immediate_stop":
            # This is the task card's Stop endpoint. Keep the answer held
            # through settlement so ordinary completion cannot win the race.
            response = _api_status(service.base_url, "POST", f"/api/tasks/{task}/cancel", {
                "stop_policy": "immediate", "stop_action_id": uuid.uuid4().hex,
            }, timeout=90)
            assert response["status"] in {200, 202}
            assert response["body"].get("ok") is True and response["body"].get("task_id") == task
            record_property("stop_http", response["status"])
            result = wait_durable_result(oracle, task, timeout=90)
            assert result["status"] == "cancelled"
            wait_for(worker.gone)
            record_property("worker_gone", 1)
        else:
            held.set()
            result = wait_durable_result(oracle, task)
            assert result["status"] == "completed" and final in str(result.get("result") or "")

        wait_for(producer.gone)
        assert ProcessIdentity.live(service.proc.pid) == host
        target_key = window_key(target)
        visible = target_key in window_rows(on_screen=True)
        assert visible == (ending != "window_gone")
        assert window_key(occluder) in window_rows(on_screen=True)
        record_property("server_gone", 1)
        record_property("target_app_alive", 1)
        record_property("target_window_visible", int(visible))
        record_property("task_" + result["status"], 1)
        if ending == "immediate_stop":
            assert not main_reply.is_set() and not held.is_set()
            held.set()  # let the stub finish the abandoned HTTP request

        assert main_reply.wait(20)
        probe = submit_running(service, "Check for a held observation and finish.")
        assert probe != task
        idle, images, proceed = turns.take(FRAME)
        assert idle.get("ok") is False and idle.get("reason") == "no_session" and not images
        fresh = capture_server(carrier(oracle, probe).pid, entry)
        assert fresh != producer and not pyobjc_mapped(fresh)
        proceed.set()
        result = wait_durable_result(oracle, probe)
        assert result["status"] == "completed" and probe_final in str(result.get("result") or "")
        wait_for(fresh.gone)
        assert model.script_consumed() and all(not g.timed_out for g in turns.gates.values())
        record_property("fresh_session_idle", 1)

    assert not service.stop_unconfirmed
