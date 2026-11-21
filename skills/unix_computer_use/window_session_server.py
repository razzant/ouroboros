"""MCP stdio server: one capture-only ScreenCaptureKit session of one exact macOS window, for the calling task.

Declared in SKILL.md (``optional_mcp_servers``, ``task_session: true``, ``runtime: agent_python``) and connected by the owner
through Connect tools. The host starts one process for a task loop's first call, as
``<agent python> -I -B window_session_server.py``, keeps it for that loop's later calls and ends it with the loop
(``ouroboros/mcp_task_session.py``); no other lifetime exists here. Listing tools and starting touch nothing
native: PyObjC and ScreenCaptureKit are reached by ``begin`` alone, and off macOS 14+ every ``begin`` is a typed
refusal. While a session is held, every jobless moment between calls gives it one bounded window-server look
(``idle_check``) — and proven destruction its single native stop — so a destroyed window's capture does not
outlive the last call; with no session held an idle moment does nothing and touches nothing native.

Tools (``macos_stream.WindowSessions`` holds their contract): ``begin`` selects one window once and starts its
stream, ``frame`` returns the latest complete image as MCP ``ImageContent`` beside one JSON object of facts,
``release`` ends the session. A result is an error result exactly when its facts say ``ok`` is not true. Images are
capture-only: no observation id, no display mapping, nothing an input tool may use.

Threads. ScreenCaptureKit is used from the process's main thread, the thread the one-shot helper was proved on, so
the main thread runs the tools one at a time (``MainThread``), services its run loop while one of them waits and
spends its jobless moments on the session's idle look. The MCP protocol runs on a second thread and hands each call
over. When the host closes stdin the protocol ends; the main thread then asks the stream to stop within a bound and
the process exits without interpreter teardown, also when that stop or anything before it raised. An idle look
already under way — bounded by one stop budget — finishes before the close is taken, so the exit can lag by that
bound; a killed process ends its stream too: the stream belongs to the process.
"""

from __future__ import annotations

import base64
import concurrent.futures
import importlib.util
import json
import os
import pathlib
import queue
import sys
import threading
import traceback
from typing import Any, Callable

SERVER_NAME = "mac_window_session"
IDLE_CHECK_S = 2.0  # how often the main thread, between jobs, gives a held session one bounded idle look
INSTRUCTIONS = (
    "Capture-only watching of ONE macOS window for this task (local macOS 14+, Screen Recording granted). begin(app, "
    "window) binds the session to one exact window once; frame() returns its latest complete image with seq and "
    "frame_age_ms; frame(after_seq=<last seq>, wait_ms=...) waits for a newer image and sends none when nothing "
    "changed; release() ends it (the task's end does too). stream.last_report is separate evidence: idle means the "
    "window had not changed at that moment. A stopped stream, a withheld frame or a gone window gives no image and "
    "nothing restarts by itself. When the window is confirmed gone, this session serves no more images and "
    "asks macOS to stop capture. The window_gone and stream_stopped refusals carry "
    "cleanup: confirmed is true only from a native answer or the stream's own end, and an unconfirmed stop is "
    "release's to retry. The images carry no input coordinates: act through element_act or a mapped screenshot."
)
_WINDOW = {"type": ["integer", "string"], "description": "Positive integer window_id or exact title string, as window_list prints them"}
TOOLS: tuple[dict[str, Any], ...] = (
    {"name": "begin",
     "description": "Start this task's one window session: resolve app (name, bundle id or pid:<n>; never launched), "
                    "select exactly one of its windows and stream it (at most 1280x800, 5 fps, no cursor). Refused "
                    "while a session is held, when the window is not exactly one, or without the Screen Recording "
                    "grant (nothing is asked). Returns facts, no image.",
     "inputSchema": {"type": "object", "additionalProperties": False, "required": ["app", "window"],
                     "properties": {"app": {"type": "string"}, "window": _WINDOW}}},
    {"name": "frame",
     "description": "The session's latest complete image with its facts (seq, frame_age_ms, stream.last_report). With "
                    "after_seq, only a newer image is sent; wait_ms (at most 10000) waits for one. No image when the "
                    "stream stopped, the frame was withheld (blank, suspended) or the window is gone.",
     "inputSchema": {"type": "object", "additionalProperties": False,
                     "properties": {"after_seq": {"type": "integer", "minimum": 0},
                                    "wait_ms": {"type": "integer", "minimum": 0, "default": 0}}}},
    {"name": "release",
     "description": "End the window session and its stream; safe to call without one. If the stop is not confirmed, "
                    "call it again: no other session can begin until it is.",
     "inputSchema": {"type": "object", "additionalProperties": False, "properties": {}}},
)


def load_stream() -> Any:
    """``macos_stream.py`` beside this file, by its path (``-I`` keeps this folder off ``sys.path``)."""
    path = pathlib.Path(__file__).resolve().with_name("macos_stream.py")
    spec = importlib.util.spec_from_file_location("_unix_computer_use_macos_stream", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dispatch(sessions: Any, name: str, arguments: Any) -> tuple[dict[str, Any], bytes | None]:
    """One tool call on the main thread -> (facts, png or None); nothing raised leaves here."""
    if name not in {tool["name"] for tool in TOOLS}:
        return {"ok": False, "reason": "unknown_tool", "error": f"no tool {name!r}"}, None
    try:
        return getattr(sessions, name)(arguments if arguments is not None else {})
    except Exception as exc:  # noqa: BLE001 - the model reads the failure; the server and its session stay
        return {"ok": False, "reason": "server_error", "error": f"{type(exc).__name__}: {exc}"}, None


class MainThread:
    """Calls handed to the process's main thread, run there one at a time in the order they were handed over.

    With ``idle`` set, a moment without jobs longer than ``idle_s`` runs it once on this thread and keeps waiting:
    that is when a held session whose window was destroyed stops its native stream even though no call is in
    flight. Jobs always take precedence over the next idle moment, and an ``idle`` that raises is reported on
    stderr and ends neither the loop nor the process."""

    def __init__(self, idle: Callable[[], None] | None = None, idle_s: float = 0.0) -> None:
        self._jobs: queue.SimpleQueue = queue.SimpleQueue()
        self._idle, self._idle_s = idle, idle_s

    def call(self, work: Callable[[], Any]) -> Any:
        """Run ``work`` on the main thread and return its result (from any other thread)."""
        future: concurrent.futures.Future = concurrent.futures.Future()
        self._jobs.put((work, future))
        return future.result()

    def close(self) -> None:
        self._jobs.put(None)

    def run(self) -> None:
        """Serve calls until ``close``; a waiting main thread costs nothing."""
        while True:
            job = self._next_job()
            if job is None:
                return
            work, future = job
            try:
                future.set_result(work())
            except BaseException as exc:  # noqa: BLE001 - the caller is waiting on this future
                future.set_exception(exc)
                if not isinstance(exc, Exception):
                    raise

    def _next_job(self) -> Any:
        """The next job, or ``None`` at ``close``; a jobless moment runs the idle look, if one is set."""
        while True:
            if self._idle is None or self._idle_s <= 0:
                return self._jobs.get()
            try:
                return self._jobs.get(timeout=self._idle_s)
            except queue.Empty:
                try:
                    self._idle()
                except Exception:  # noqa: BLE001 - an idle look may never end the server
                    traceback.print_exc()


def build_server(call: Callable[[str, Any], tuple[dict[str, Any], bytes | None]]) -> Any:
    """The MCP server whose tools answer through ``call(name, arguments) -> (facts, png)``."""
    import anyio
    import mcp.types as types
    from mcp.server.lowlevel import Server

    server = Server(SERVER_NAME, instructions=INSTRUCTIONS)

    @server.list_tools()
    async def list_tools() -> list[Any]:
        return [types.Tool(**tool) for tool in TOOLS]

    @server.call_tool(validate_input=False)  # WindowSessions validates: one owner, typed refusals
    async def call_tool(name: str, arguments: Any) -> Any:
        facts, png = await anyio.to_thread.run_sync(call, name, arguments)
        content: list[Any] = [types.TextContent(type="text", text=json.dumps(facts, ensure_ascii=False))]
        if png is not None and facts.get("image") is True:
            content.append(types.ImageContent(type="image", mimeType="image/png",
                                              data=base64.b64encode(png).decode("ascii")))
        return types.CallToolResult(content=content, isError=facts.get("ok") is not True)

    return server


def serve(call: Callable[[str, Any], tuple[dict[str, Any], bytes | None]]) -> None:
    """Speak MCP over this process's stdio until the host closes stdin."""
    import anyio
    from mcp.server.stdio import stdio_server

    server = build_server(call)

    async def run() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    anyio.run(run)


def main() -> int:
    import mcp  # noqa: F401 - a missing SDK is named on stderr before anything starts

    sessions = load_stream().WindowSessions()
    # getattr: a stand-in stream module without idle_check keeps working, with no idle looks
    main_thread = MainThread(idle=getattr(sessions, "idle_check", None), idle_s=IDLE_CHECK_S)
    failed: list[bool] = []

    def protocol() -> None:
        try:
            serve(lambda name, arguments: main_thread.call(lambda: dispatch(sessions, name, arguments)))
        except BaseException:  # noqa: BLE001 - reported, then the process ends as on a closed stdin
            failed.append(True)
            traceback.print_exc()
        finally:
            main_thread.close()

    threading.Thread(target=protocol, name="mcp-protocol", daemon=True).start()
    try:
        main_thread.run()
    finally:
        sessions.shutdown()
    return 1 if failed else 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except BaseException:  # noqa: BLE001 - reported; whatever ended main, the exit below must still happen
        traceback.print_exc()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except ValueError:  # the SDK's transport closed the stdout buffer it had wrapped
            pass
    # A callback of a stream that did not confirm its stop may still be held by a framework queue: ending without
    # interpreter teardown keeps it from running into a finalizing interpreter (as macos_capture.py ends).
    os._exit(code)
