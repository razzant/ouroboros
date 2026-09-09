"""Secretless ACP framing and custody tests with mocked subprocess pipes."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ouroboros.gateways import copilot_acp as acp


def response(request_id, result):
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def startup():
    return [
        response(1, {"protocolVersion": 1, "agentCapabilities": {}}),
        response(2, {"sessionId": "session-1"}),
    ]


class _Input(io.BytesIO):
    captured = b""

    def close(self):
        if not self.closed:
            self.captured = self.getvalue()
        super().close()


@pytest.fixture
def process_factory(monkeypatch, tmp_path):
    runs = []

    def make(frames, *, stderr=b"", cleanup_error=""):
        raw = frames if isinstance(frames, bytes) else b"".join(
            json.dumps(frame).encode() + b"\n" for frame in frames
        )
        process = SimpleNamespace(
            pid=12345678, stdin=_Input(), stdout=io.BytesIO(raw),
            stderr=io.BytesIO(stderr), wait=lambda **_kwargs: 0, poll=lambda: 0,
        )
        state = {"process": process, "reaped": False, "closed": False, "records": []}

        class Container:
            def spawn(self, command, **kwargs):
                state.update(command=command, kwargs=kwargs)
                return process

            def reap(self):
                state["reaped"] = True
                return cleanup_error

            def close(self):
                state["closed"] = True

        monkeypatch.setattr("ouroboros.process_containment.ProcessContainer", Container)
        monkeypatch.setattr("ouroboros.process_custody.record_process", lambda root, **kwargs: state["records"].append((root, kwargs)))
        spec = acp.ACPProcessSpec(
            command=["copilot", "--acp"], cwd=tmp_path, env={"HOME": str(tmp_path)},
            drive_root=tmp_path / "data", task_id="task-1", startup_timeout=1, shutdown_timeout=0.1,
        )
        state["spec"] = spec
        runs.append(state)
        return spec, state

    yield make
    assert all(state["closed"] and state["reaped"] for state in runs)


def test_handshake_prompt_permissions_and_guardian_custody(process_factory):
    permission = {
        "jsonrpc": "2.0", "id": "permission-1", "method": "session/request_permission",
        "params": {"sessionId": "session-1", "toolCall": {"kind": "edit"}, "options": []},
    }
    spec, state = process_factory([*startup(), permission, response(3, {"stopReason": "end_turn"})])
    seen = []
    decision = {"outcome": {"outcome": "selected", "optionId": "once"}}
    with acp.CopilotACPClient(
        spec, permission_handler=lambda params: decision,
        observe=lambda direction, frame: seen.append((direction, frame)),
    ) as client:
        assert client.initialize()["protocolVersion"] == 1
        assert client.new_session()["sessionId"] == "session-1"
        assert list(client.prompt("Do the complete task"))[-1]["result"]["stopReason"] == "end_turn"
        assert client.prompt_sent and client.completed
    sent = [json.loads(line) for line in state["process"].stdin.captured.splitlines()]
    assert [row.get("method") for row in sent] == ["initialize", "session/new", "session/prompt", None]
    assert sent[0]["params"]["clientCapabilities"] == {}
    assert sent[1]["params"] == {"cwd": str(spec.cwd), "mcpServers": []}
    assert sent[2]["params"]["prompt"] == [{"type": "text", "text": "Do the complete task"}]
    assert sent[3]["id"] == "permission-1" and sent[3]["result"] == decision
    assert state["command"][:3] == [sys.executable, "-m", "ouroboros.gateways.copilot_acp"]
    from ouroboros.process_custody import current_custody_session_id
    assert state["command"][5] == current_custody_session_id()
    assert state["command"][-2:] == ["copilot", "--acp"]
    assert state["records"][0][1]["scope"] == "task"
    assert state["records"][0][1]["owner_task_id"] == "task-1"
    assert state["kwargs"]["stdin"] == subprocess.PIPE
    assert seen[0][0] == "client" and any(frame == permission for direction, frame in seen if direction == "agent")


@pytest.mark.parametrize("foreign", [True, False])
def test_foreign_permissions_and_unadvertised_client_capabilities_fail_closed(process_factory, foreign):
    request = {
        "jsonrpc": "2.0", "id": 40,
        "method": "session/request_permission" if foreign else "fs/write_text_file",
        "params": {"sessionId": "other" if foreign else "session-1"},
    }
    spec, state = process_factory([*startup(), request, response(3, {"stopReason": "end_turn"})])
    with acp.CopilotACPClient(spec, permission_handler=lambda _: pytest.fail("untrusted permission reached policy")) as client:
        client.initialize()
        client.new_session()
        list(client.prompt("hello"))
    reply = [json.loads(line) for line in state["process"].stdin.captured.splitlines()][-1]
    assert reply.get("result") == {"outcome": {"outcome": "cancelled"}} if foreign else reply["error"]["code"] == -32601


@pytest.mark.parametrize("frame, code", [
    (b"not json\n", "acp_protocol_error"),
    (b'{"jsonrpc":"2.0","id":1}', "acp_protocol_error"),
    ([response(99, {})], "acp_protocol_error"),
    ([{"jsonrpc": "2.0", "id": True, "result": {}}], "acp_protocol_error"),
    ([{"jsonrpc": "2.0", "id": 1, "result": {}, "error": {}}], "acp_protocol_error"),
    ([{"jsonrpc": "1.0", "id": 1, "result": {}}], "acp_protocol_error"),
    ([response(1, [])], "acp_protocol_error"),
    ([response(1, {"protocolVersion": 2, "agentCapabilities": {}})], "acp_protocol_incompatible"),
    ([response(1, {"protocolVersion": 1})], "acp_protocol_error"),
    ([{"jsonrpc": "2.0", "id": 1, "error": {"code": -32000, "message": "Sign in"}}], "acp_auth_required"),
    (b"", "acp_process_exited"),
])
def test_malformed_or_failed_handshake_never_sends_task(process_factory, frame, code):
    spec, state = process_factory(frame)
    with pytest.raises(acp.ACPError) as raised:
        with acp.CopilotACPClient(spec, permission_handler=lambda _: {}) as client:
            client.initialize()
    assert raised.value.code == code
    assert b"session/prompt" not in state["process"].stdin.captured


def test_stderr_drains_but_is_bounded(process_factory):
    spec, state = process_factory([*startup()], stderr=b"x" * (acp.STDERR_CAP_BYTES + 10000))
    with acp.CopilotACPClient(spec, permission_handler=lambda _: {}) as client:
        client.initialize()
        client.new_session()
    assert len(client.stderr) == acp.STDERR_CAP_BYTES
    assert client.overflow["stderr"]


def test_oversized_stdout_is_rejected_before_json_decoding(process_factory, monkeypatch):
    monkeypatch.setattr(acp, "MAX_FRAME_BYTES", 512)
    spec, _ = process_factory(b" " * 513 + b"\n")
    with pytest.raises(acp.ACPError, match="byte limit"):
        with acp.CopilotACPClient(spec, permission_handler=lambda _: {}) as client:
            client.initialize()


def test_oversized_input_is_not_silently_truncated_or_resent(process_factory, monkeypatch):
    monkeypatch.setattr(acp, "MAX_FRAME_BYTES", 512)
    spec, state = process_factory(startup())
    with acp.CopilotACPClient(spec, permission_handler=lambda _: {}) as client:
        client.initialize()
        client.new_session()
        with pytest.raises(acp.ACPError, match="nothing was truncated"):
            list(client.prompt("x" * 600))
        with pytest.raises(acp.ACPError, match="never automatically retried"):
            list(client.prompt("shorter"))
    assert b"session/prompt" not in state["process"].stdin.captured


def test_disconnect_after_dispatch_cancels_and_never_replays(process_factory):
    spec, state = process_factory(startup())
    with acp.CopilotACPClient(spec, permission_handler=lambda _: {}) as client:
        client.initialize()
        client.new_session()
        with pytest.raises(acp.ACPError, match="exited"):
            list(client.prompt("edit the workspace"))
    sent = [json.loads(line) for line in state["process"].stdin.captured.splitlines()]
    assert sum(row.get("method") == "session/prompt" for row in sent) == 1
    assert sent[-1]["method"] == "session/cancel"


def test_cleanup_failure_is_not_a_success(process_factory):
    spec, _ = process_factory(startup(), cleanup_error="a child remains alive")
    with pytest.raises(acp.ACPError) as raised:
        with acp.CopilotACPClient(spec, permission_handler=lambda _: {}) as client:
            client.initialize()
    assert raised.value.code == "acp_cleanup_unconfirmed"


def test_control_checked_before_spawn(monkeypatch, tmp_path):
    monkeypatch.setattr("ouroboros.process_containment.ProcessContainer", lambda: pytest.fail("cancelled task spawned"))
    spec = acp.ACPProcessSpec(["copilot", "--acp"], tmp_path, {}, tmp_path, "task")

    def stopped():
        raise acp.ACPError("owner stopped", "acp_cancel_requested")

    with pytest.raises(acp.ACPError, match="owner stopped"):
        with acp.CopilotACPClient(spec, permission_handler=lambda _: {}, check_control=stopped):
            pytest.fail("entered cancelled session")
