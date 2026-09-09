"""Stdio ACP transport for a single Copilot agent session, not model completions.

Wire semantics adapted from Q00/ouroboros's Copilot ACP client:
Copyright (c) 2025 Q00

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

A small Python guardian inherits the protocol pipes unchanged. Its existing
parent lifeline ends the shared guardian/CLI group when the worker dies;
durable task custody and Windows Job Objects provide the other cleanup rails.
No restart or resend is allowed after a prompt might have reached the agent.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import pathlib
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator

from ouroboros.config import COPILOT_ACP_SHUTDOWN_TIMEOUT_SEC, COPILOT_ACP_STARTUP_TIMEOUT_SEC

MAX_FRAME_BYTES = 8 * 1024 * 1024
STDERR_CAP_BYTES = 64 * 1024


class ACPError(RuntimeError):
    """A typed transport failure; never evidence that a sent task did not run."""

    def __init__(self, message: str, code: str = "acp_protocol_error"):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ACPProcessSpec:
    command: list[str]
    cwd: pathlib.Path
    env: dict[str, str]
    drive_root: pathlib.Path
    task_id: str
    startup_timeout: float = COPILOT_ACP_STARTUP_TIMEOUT_SEC
    shutdown_timeout: float = COPILOT_ACP_SHUTDOWN_TIMEOUT_SEC


class CopilotACPClient:
    """One sequential JSON-RPC client with bounded, independently drained pipes."""

    def __init__(
        self,
        spec: ACPProcessSpec,
        *,
        permission_handler: Callable[[dict], dict],
        observe: Callable[[str, dict], None] = lambda _direction, _frame: None,
        check_control: Callable[[], None] = lambda: None,
    ):
        for seconds in (spec.startup_timeout, spec.shutdown_timeout):
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("ACP process waits must be finite positive seconds")
        self.spec = spec
        self.permission_handler = permission_handler
        self.observe = observe
        self.check_control = check_control
        self.process: subprocess.Popen | None = None
        self.session_id = ""
        self.prompt_sent = False
        self.completed = False
        self.stderr = bytearray()
        self.overflow = {"stderr": False}
        self._frames: queue.Queue = queue.Queue(maxsize=16)
        self._writes: queue.Queue = queue.Queue(maxsize=2)
        self._threads: list[threading.Thread] = []
        self._closed = threading.Event()
        self._next_id = 0
        self._container: Any = None

    def __enter__(self) -> CopilotACPClient:
        from ouroboros.extension_companion import _drain_companion_pipe
        from ouroboros.platform_layer import merge_hidden_kwargs
        from ouroboros.process_containment import ProcessContainer
        from ouroboros.process_custody import current_custody_session_id, record_process

        self.check_control()
        self._container = ProcessContainer()
        command = [
            sys.executable, "-m", "ouroboros.gateways.copilot_acp",
            str(self.spec.drive_root), self.spec.task_id, current_custody_session_id(), *self.spec.command,
        ]
        # The guardian needs the installed host package; this import path does
        # not reach Copilot's tools (the guardian removes it before CLI spawn).
        env = {**self.spec.env, "PYTHONPATH": str(pathlib.Path(__file__).resolve().parents[2])}
        try:
            self.process = self._container.spawn(command, **merge_hidden_kwargs({
                "cwd": str(self.spec.cwd), "env": env,
                "stdin": subprocess.PIPE, "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE, "bufsize": 0,
            }))
            record_process(
                self.spec.drive_root, pid=self.process.pid, cmd=command,
                purpose="task_runtime:copilot_acp", scope="task",
                owner_task_id=self.spec.task_id,
            )
            self._start_thread(self._read_stdout, (), "acp-stdout")
            self._start_thread(self._write_stdin, (), "acp-stdin")
            self._start_thread(
                _drain_companion_pipe,
                (self.process.stderr, STDERR_CAP_BYTES, self.stderr, self.overflow, "stderr"),
                "acp-stderr",
            )
            return self
        except BaseException:
            self.close()
            raise

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _start_thread(self, target: Callable, args: tuple, name: str) -> None:
        thread = threading.Thread(target=target, args=args, name=name, daemon=True)
        self._threads.append(thread)
        thread.start()

    def _publish(self, value: Any) -> None:
        while not self._closed.is_set():
            try:
                self._frames.put(value, timeout=0.1)
                return
            except queue.Full:
                continue

    def _read_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        try:
            stream = io.BufferedReader(self.process.stdout)
            while not self._closed.is_set():
                line = stream.readline(MAX_FRAME_BYTES + 1)
                if not line:
                    raise ACPError("Copilot ACP exited before its response.", "acp_process_exited")
                if len(line) > MAX_FRAME_BYTES:
                    raise ACPError("Copilot ACP frame exceeds the byte limit.", "acp_frame_too_large")
                if not line.endswith(b"\n"):
                    raise ACPError("Copilot ACP returned a truncated frame.")
                try:
                    frame = json.loads(line)
                except (ValueError, UnicodeError, RecursionError) as exc:
                    raise ACPError("Copilot ACP returned invalid JSON.") from exc
                if not isinstance(frame, dict) or frame.get("jsonrpc") != "2.0":
                    raise ACPError("Copilot ACP returned an invalid JSON-RPC envelope.")
                self._publish(frame)
        except (ACPError, OSError, ValueError) as exc:
            self._publish(exc if isinstance(exc, ACPError) else ACPError("Copilot ACP output closed."))

    def _write_stdin(self) -> None:
        assert self.process is not None and self.process.stdin is not None
        while not self._closed.is_set():
            try:
                data, done, errors = self._writes.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                view = memoryview(data)
                while view:
                    written = self.process.stdin.write(view)
                    if not written:
                        raise BrokenPipeError("closed ACP input")
                    view = view[written:]
                self.process.stdin.flush()
            except (OSError, ValueError):
                errors.append(ACPError("Copilot ACP closed its input.", "acp_process_exited"))
            finally:
                done.set()

    def _send(self, frame: dict, *, closing: bool = False) -> None:
        data = json.dumps(frame, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
        if len(data) > MAX_FRAME_BYTES:
            raise ACPError("ACP request exceeds the byte limit; nothing was truncated.", "acp_request_too_large")
        self.observe("client", frame)
        done, errors = threading.Event(), []
        limit = self.spec.shutdown_timeout if closing else self.spec.startup_timeout
        deadline = time.monotonic() + limit
        try:
            self._writes.put((data, done, errors), timeout=limit)
        except queue.Full as exc:
            raise ACPError("Copilot ACP input is blocked.", "acp_transport_timeout") from exc
        while not done.wait(0.1):
            if not closing:
                self.check_control()
            if time.monotonic() >= deadline:
                raise ACPError("Copilot ACP input did not drain.", "acp_transport_timeout")
        if errors:
            raise errors[0]

    def exchange(self, method: str, params: dict, *, startup: bool = False) -> Iterator[dict]:
        """Deliver server notifications/requests in order, then the matching result."""
        self._next_id += 1
        request_id = self._next_id
        deadline = time.monotonic() + self.spec.startup_timeout if startup else None
        self.check_control()
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        while True:
            self.check_control()
            if deadline is not None and time.monotonic() >= deadline:
                raise ACPError(f"Copilot ACP {method} handshake timed out.", "acp_transport_timeout")
            try:
                frame = self._frames.get(timeout=0.1)
            except queue.Empty:
                continue
            if isinstance(frame, Exception):
                raise frame
            self.observe("agent", frame)
            if "method" in frame:
                self._handle_server_frame(frame)
                yield frame
                continue
            if type(frame.get("id")) is not int or frame["id"] != request_id:
                raise ACPError("Copilot ACP returned an unexpected response id.")
            if ("result" in frame) == ("error" in frame):
                raise ACPError("Copilot ACP response must contain result or error.")
            if "error" in frame:
                error = frame["error"]
                if not isinstance(error, dict):
                    raise ACPError("Copilot ACP returned a malformed error.")
                code = "acp_auth_required" if error.get("code") in (-32000, 401, 403) else "acp_request_failed"
                raise ACPError(f"Copilot ACP {method}: {error.get('message', 'request refused')}", code)
            if not isinstance(frame["result"], dict):
                raise ACPError("Copilot ACP returned a non-object result.")
            yield frame
            return

    def _handle_server_frame(self, frame: dict) -> None:
        if not isinstance(frame["method"], str) or not isinstance(frame.get("params", {}), dict):
            raise ACPError("Copilot ACP returned malformed notification parameters.")
        if "id" not in frame:
            return
        if type(frame["id"]) not in (int, str):
            raise ACPError("Copilot ACP returned an invalid server request id.")
        response: dict = {"jsonrpc": "2.0", "id": frame["id"]}
        if frame["method"] == "session/request_permission":
            params = frame.get("params", {})
            response["result"] = {"outcome": {"outcome": "cancelled"}}
            if self.session_id and params.get("sessionId") == self.session_id:
                response["result"] = self.permission_handler(params)
        else:
            response["error"] = {"code": -32601, "message": "Client capability not provided"}
        self._send(response)

    def _handshake(self, method: str, params: dict, on_update: Callable[[dict], None]) -> dict:
        for frame in self.exchange(method, params, startup=True):
            if "method" not in frame:
                return frame["result"]
            on_update(frame)
        raise ACPError("Copilot ACP handshake returned no result.")

    def initialize(self, on_update: Callable[[dict], None] = lambda _frame: None) -> dict:
        result = self._handshake("initialize", {
            "protocolVersion": 1, "clientCapabilities": {},
            "clientInfo": {"name": "ouroboros", "version": "1"},
        }, on_update)
        if type(result.get("protocolVersion")) is not int or result["protocolVersion"] != 1:
            raise ACPError("Copilot ACP protocol version is incompatible.", "acp_protocol_incompatible")
        if not isinstance(result.get("agentCapabilities"), dict):
            raise ACPError("Copilot ACP initialize omitted agentCapabilities.")
        return result

    def new_session(self, on_update: Callable[[dict], None] = lambda _frame: None) -> dict:
        result = self._handshake("session/new", {
            "cwd": str(self.spec.cwd), "mcpServers": [],
        }, on_update)
        session_id = result.get("sessionId")
        if not isinstance(session_id, str) or not session_id.strip():
            raise ACPError("Copilot ACP session/new omitted sessionId.")
        self.session_id = session_id
        return result

    def prompt(self, text: str) -> Iterator[dict]:
        if not self.session_id or self.prompt_sent:
            raise ACPError("ACP requires a fresh session; prompts are never automatically retried.")
        self.prompt_sent = True
        for frame in self.exchange("session/prompt", {
            "sessionId": self.session_id, "prompt": [{"type": "text", "text": text}],
        }):
            if "method" not in frame:
                if not isinstance(frame["result"].get("stopReason"), str) or not frame["result"]["stopReason"]:
                    raise ACPError("Copilot ACP prompt omitted stopReason.")
                self.completed = True
            yield frame

    def close(self) -> None:
        """Cancel once, close stdin, then verify cleanup of the entire owned tree."""
        if self._closed.is_set():
            return
        proc = self.process
        cleanup_error = ""
        try:
            if proc is not None:
                if self.prompt_sent and not self.completed:
                    with contextlib.suppress(Exception):
                        self._send({
                            "jsonrpc": "2.0", "method": "session/cancel",
                            "params": {"sessionId": self.session_id},
                        }, closing=True)
                if proc.stdin is not None:
                    with contextlib.suppress(OSError, ValueError):
                        proc.stdin.close()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=self.spec.shutdown_timeout)
            if self._container is not None:
                cleanup_error = self._container.reap()
            if proc is not None:
                try:
                    proc.wait(timeout=self.spec.shutdown_timeout)
                except subprocess.TimeoutExpired:
                    cleanup_error = cleanup_error or "ACP process exit could not be confirmed."
        finally:
            self._closed.set()
            if self._container is not None:
                self._container.close()
            for thread in self._threads:
                thread.join(timeout=self.spec.shutdown_timeout)
            if proc is not None:
                for pipe in (proc.stdin, proc.stdout, proc.stderr):
                    if pipe is not None:
                        with contextlib.suppress(OSError, ValueError):
                            pipe.close()
        if cleanup_error:
            raise ACPError(cleanup_error, "acp_cleanup_unconfirmed")


def _run_guardian() -> int:
    """Keep the vendor CLI in the existing parent-lifeline process group."""
    import os
    from ouroboros.process_custody import adopt_session_id, spawn_supervised, start_parent_lifeline

    drive, task, session, *command = sys.argv[1:]
    adopt_session_id(session)
    start_parent_lifeline(label="copilot-acp")
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    proc = spawn_supervised(
        command, drive_root=pathlib.Path(drive), purpose="task_runtime:copilot_acp:agent",
        scope="task", owner_task_id=task, new_process_group=False, env=env,
    )
    return proc.wait()


if __name__ == "__main__":
    raise SystemExit(_run_guardian())
