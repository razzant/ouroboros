"""ASGI response frames over the extension child's existing portable stdio pipes.

The runner owns process staging/custody; this module owns response delivery and
backpressure. No response lifetime timer or durable stream registry is involved.

Frame bounds come from `runtime_limits.py`: `EXTENSION_STREAM_CHUNK_BYTES` (body
frame), `EXTENSION_STREAM_METADATA_BYTES` (metadata frame) and
`EXTENSION_CHILD_CLEANUP_GRACE_SEC` (post-response child exit); none is a response
deadline, and a slow consumer's backpressure is never an idle failure. Headers keep
their order; HEAD and bodiless statuses finish on the headers, and a child's
background work survives a client that closes once the wire body is complete.
Cancellation during startup keeps the worker future and process context until the
child exits. A failed final send is a delivery failure; a child error after the
final body is only a diagnostic and leaves the delivered body intact. An abnormal
child exit logs its exit code and bounded, sanitized stderr.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
import logging
import os
import struct
import sys
import threading

from starlette.responses import JSONResponse, Response

from ouroboros.utils import sanitize_tool_result_for_log
from ouroboros.config import (
    EXTENSION_STREAM_CHUNK_BYTES, EXTENSION_STREAM_METADATA_BYTES,
    EXTENSION_CHILD_CLEANUP_GRACE_SEC,
)

log = logging.getLogger(__name__)


def _write_frame(stream, kind: bytes, payload: bytes = b"") -> None:
    frame = kind + payload
    for value in (struct.pack("!I", len(frame)), frame):
        remaining = memoryview(value)
        while remaining:
            written = stream.write(remaining)
            if not written:
                raise OSError("extension response channel stopped accepting bytes")
            remaining = remaining[written:]
    stream.flush()


def _read_exact(stream, size: int) -> bytes:
    parts = bytearray()
    while len(parts) < size:
        chunk = stream.read(size - len(parts))
        if not chunk:
            raise EOFError("extension response channel closed")
        parts.extend(chunk)
    return bytes(parts)


def _read_frame(stream):
    size = struct.unpack("!I", _read_exact(stream, 4))[0]
    if size < 1:
        raise ValueError("empty extension response frame")
    kind = _read_exact(stream, 1)
    payload_limit = EXTENSION_STREAM_CHUNK_BYTES + 1 if kind == b"B" else EXTENSION_STREAM_METADATA_BYTES
    if size - 1 > payload_limit:
        raise ValueError("extension response frame exceeds its channel bound")
    return kind, _read_exact(stream, size - 1)


class ChildResponseChannel:
    """Child ASGI send/receive: reserve stdout before loading plugin code."""

    def __init__(self):
        import sys
        sys.stdout.flush()
        self.output = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        self.input = os.fdopen(os.dup(sys.stdin.fileno()), "rb", buffering=0)
        self.body_complete = False
        self.started = False
        self.disconnected = None
        self.control_thread = None

    def bind(self):
        loop = asyncio.get_running_loop()
        self.disconnected = asyncio.Event()

        def control():
            try:
                _read_frame(self.input)
            except (EOFError, OSError, ValueError):
                pass
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(self.disconnected.set)

        self.control_thread = threading.Thread(target=control, daemon=True, name="extension-route-control")
        self.control_thread.start()

    async def receive(self):
        await self.disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(self, message):
        if self.disconnected.is_set():
            raise OSError("extension response client disconnected")
        kind = message["type"]
        if kind == "http.response.start":
            payload = json.dumps({
                "status": message["status"],
                "headers": [[bytes(k).decode("latin-1"), bytes(v).decode("latin-1")]
                            for k, v in message.get("headers", [])],
            }).encode()
            await asyncio.to_thread(_write_frame, self.output, b"S", payload)
            self.started = True
        elif kind == "http.response.body":
            body = memoryview(message.get("body") or b"")
            more = bool(message.get("more_body", False))
            for start in range(0, max(1, len(body)), EXTENSION_STREAM_CHUNK_BYTES):
                end = min(start + EXTENSION_STREAM_CHUNK_BYTES, len(body))
                payload = bytes([int(more or end < len(body))]) + body[start:end].tobytes()
                await asyncio.to_thread(_write_frame, self.output, b"B", payload)
            self.body_complete = not more
        else:
            raise ValueError(f"unsupported extension response message {kind!r}")

    def error(self, exc):
        message = sanitize_tool_result_for_log(f"{type(exc).__name__}: {exc}")
        _write_frame(self.output, b"C" if self.body_complete else b"E", message.encode("utf-8"))

    def finish(self, ws_relay_failures=None):
        payload = json.dumps({"ws_relay_failures": ws_relay_failures}).encode() if ws_relay_failures else b""
        _write_frame(self.output, b"X", payload)
        self.output.close()
        # The control thread is daemon-owned and ends with this per-call child.
        # Closing stdin here could block on its pending read's file-object lock.


class RouteStreamResponse(Response):
    """A response whose handle belongs to one currently published skill bundle."""

    def __init__(self, spec, child_factory):
        super().__init__(status_code=200)
        self.spec = spec
        self.child_factory = child_factory
        self.cancelled = False
        self.task = self.loop = None
        self.body_complete = False
        self.started = False
        self.client_disconnected = False
        self._wire_expected = None
        self._wire_sent = 0
        self._finishing_send = False
        self._head = False
        self.ws_relay_failures = None

    def cancel(self):
        if self.cancelled:
            return
        self.cancelled = True
        if self.loop is not None and self.task is not None:
            with contextlib.suppress(RuntimeError):
                self.loop.call_soon_threadsafe(self.task.cancel)

    def _wire_complete(self):
        return self.started and self._wire_expected is not None and self._wire_sent >= self._wire_expected

    async def _wait_worker(self, callback, *args):
        """Keep custody of blocking work until it settles, even after cancellation."""
        future = asyncio.create_task(asyncio.to_thread(callback, *args))
        cancelled = False
        while not future.done():
            try:
                await asyncio.shield(future)
            except asyncio.CancelledError:
                cancelled = True
            except Exception:
                if not cancelled:
                    raise
                break
        if cancelled:
            try:
                future.result()
            except Exception as exc:
                log.warning("extension response work failed while cancellation settled: %s", exc)
            raise asyncio.CancelledError
        return future.result()

    @contextlib.asynccontextmanager
    async def _child_scope(self):
        """The existing process context stays owned while startup runs off-loop."""
        from ouroboros.extension_registry_state import extension_work_scope

        stack = contextlib.ExitStack()
        def start():
            stack.enter_context(extension_work_scope(self.spec, self))
            if self.cancelled:
                raise asyncio.CancelledError
            return stack.enter_context(self.child_factory())
        try:
            yield await self._wait_worker(start)
        finally:
            # Startup is settled before _wait_worker propagates cancellation;
            # this stack therefore still owns any process that actually spawned.
            await self._wait_worker(stack.__exit__, *sys.exc_info())

    async def _send_to_client(self, send, message):
        try:
            await send(message)
        except OSError:
            # HTTP clients may close after Content-Length bytes (or HEAD
            # headers), before FileResponse's final empty ASGI body. Preserve
            # the child's background work when the wire body is already done.
            if self._wire_complete() and (self._head or not message.get("body")):
                return
            self.client_disconnected = True
            raise

    async def _pump(self, frames, send):
        from ouroboros.extension_process_runner import ExtensionProcessError
        while True:
            kind, payload = await frames.get()
            if kind == b"S":
                fields = json.loads(payload)
                self.status_code = int(fields["status"])
                self.raw_headers = [(k.encode("latin-1"), v.encode("latin-1"))
                                    for k, v in fields["headers"]]
                lengths = [v for k, v in self.raw_headers if k.lower() == b"content-length"]
                try:
                    self._wire_expected = int(lengths[0]) if lengths and all(value.strip().isdigit() for value in lengths) else None
                    if self._wire_expected is not None and any(int(value) != self._wire_expected for value in lengths):
                        self._wire_expected = None
                except ValueError:
                    self._wire_expected = None
                if self._head or self.status_code in {204, 205, 304}:
                    self._wire_expected = 0
                self._finishing_send = self._wire_expected == 0
                await self._send_to_client(send, {"type": "http.response.start", "status": self.status_code,
                                                  "headers": self.raw_headers})
                self.started = True
                self._finishing_send = False
            elif kind == b"B":
                if not self.started or not payload:
                    raise ExtensionProcessError("extension body preceded response headers")
                final = payload[0] == 0
                self._finishing_send = final or (
                    self._wire_expected is not None and self._wire_sent + len(payload) - 1 >= self._wire_expected)
                try:
                    await self._send_to_client(send, {"type": "http.response.body", "body": payload[1:],
                                                      "more_body": not final})
                except BaseException:
                    # A failed final send is not a completed delivery. The flag
                    # only suppresses the server's normal post-response disconnect.
                    self.body_complete = False
                    raise
                self._wire_sent += len(payload) - 1
                self.body_complete = final
                self._finishing_send = False
            elif kind in {b"E", b"C"}:
                detail = payload.decode("utf-8", errors="replace")
                if self.body_complete:
                    log.warning("extension response cleanup failed after delivery: %s", detail)
                    if kind == b"E":
                        return
                else:
                    raise ExtensionProcessError(detail)
            elif kind == b"X":
                if not self.body_complete:
                    raise ExtensionProcessError("extension response ended before its final body")
                self.ws_relay_failures = json.loads(payload).get("ws_relay_failures") if payload else None
                return
            else:
                raise ExtensionProcessError("invalid extension response frame")

    async def __call__(self, scope, receive, send):
        from ouroboros.extension_process_runner import (
            _drain, _STDERR_CAP, _publish_child_facts, _format_child_returncode,
        )

        self.loop, self.task = asyncio.get_running_loop(), asyncio.current_task()
        self._head = str(scope.get("method") or "GET").upper() == "HEAD"
        if self.cancelled:
            raise asyncio.CancelledError
        try:
            async with self._child_scope() as child:
                proc = child.proc
                frames = asyncio.Queue(maxsize=1)
                stop_reader = threading.Event()
                submitted = []
                reader_lock = threading.Lock()
                stderr, overflow = bytearray(), {"stderr": False}

                def reader():
                    while not stop_reader.is_set():
                        try:
                            frame = _read_frame(proc.stdout)
                        except (EOFError, OSError, ValueError) as exc:
                            frame = (b"E", str(exc).encode())
                        with reader_lock:
                            if stop_reader.is_set():
                                return
                            future = asyncio.run_coroutine_threadsafe(frames.put(frame), self.loop)
                            submitted[:] = [future]
                        try:
                            future.result()
                        except concurrent.futures.CancelledError:
                            return
                        if frame[0] in {b"X", b"E"}:
                            return

                read_thread = threading.Thread(target=reader, daemon=True, name="extension-route-response")
                err_thread = threading.Thread(target=_drain,
                    args=(proc.stderr, _STDERR_CAP, stderr, overflow, "stderr"),
                    kwargs={"discard_excess": True}, daemon=True, name="extension-route-stderr")
                read_thread.start()
                err_thread.start()

                async def disconnect():
                    while True:
                        if (await receive())["type"] == "http.disconnect":
                            return
                        await asyncio.sleep(0)

                pump_task = asyncio.create_task(self._pump(frames, send))
                disconnect_task = asyncio.create_task(disconnect())
                try:
                    done, _ = await asyncio.wait({pump_task, disconnect_task}, return_when=asyncio.FIRST_COMPLETED)
                    if disconnect_task in done and not (
                        self.body_complete or self._wire_complete() or self._finishing_send
                    ):
                        self.client_disconnected = True
                        return
                    await pump_task
                finally:
                    with reader_lock:
                        stop_reader.set()
                        for future in submitted:
                            future.cancel()
                    for task in (pump_task, disconnect_task):
                        task.cancel()
                    await asyncio.gather(pump_task, disconnect_task, return_exceptions=True)
                    # A cancellation grants the child's existing ASGI cleanup a
                    # short exit opportunity; this is not a response timer.
                    killed_by_host = False
                    def cleanup():
                        nonlocal killed_by_host
                        with contextlib.suppress(OSError, ValueError, BrokenPipeError):
                            _write_frame(proc.stdin, b"D")
                        try:
                            proc.wait(timeout=EXTENSION_CHILD_CLEANUP_GRACE_SEC)
                        except Exception:
                            from ouroboros.tools.shell import _kill_process_group
                            _kill_process_group(proc)
                            killed_by_host = True
                            with contextlib.suppress(Exception):
                                proc.wait(timeout=EXTENSION_CHILD_CLEANUP_GRACE_SEC)
                        read_thread.join(EXTENSION_CHILD_CLEANUP_GRACE_SEC)
                        err_thread.join(EXTENSION_CHILD_CLEANUP_GRACE_SEC)
                    try:
                        await self._wait_worker(cleanup)
                    finally:
                        # Keep the existing thread-local process-fact publisher.
                        _publish_child_facts(proc, child.started_ts, killed_by_host=killed_by_host,
                                             ws_relay_failures=self.ws_relay_failures,
                                             skill_name=str(self.spec.get("skill") or ""))
                    if proc.returncode not in (None, 0) and not killed_by_host:
                        detail = sanitize_tool_result_for_log(stderr.decode("utf-8", errors="replace").strip())[-2000:]
                        log.warning("extension route child exited abnormally: %s; %s",
                                    _format_child_returncode(proc.returncode), detail)
                    if overflow["stderr"]:
                        log.warning("extension route diagnostic output was truncated")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self.client_disconnected:
                return
            if self.body_complete:
                log.warning("extension response cleanup failed after delivery: %s", exc)
            elif self.started:
                raise
            else:
                self.status_code = 502
                await JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=502)(scope, receive, send)
