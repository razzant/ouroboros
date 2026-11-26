"""One serving generation's history reader. It receives the admitted GET inputs,
not a route or a store name; the server retains authentication and HTTP delivery.
The child is one custodied interpreter, with no shared port or on-disk request queue.
"""
from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import logging
import os
import struct
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import ContextVar
from pathlib import Path

log = logging.getLogger(__name__)
PURPOSE = "gateway.history_reader"
PARALLEL_READS = 2
_HEADER = struct.Struct("!II")
# This is a private inherited channel. The metadata is bounded separately from the
# response body, which is transferred verbatim rather than encoded and decoded again.
_MAX_METADATA = 16 * 1024 * 1024
_HANDOFF_SECONDS = 2
READ_PATH = ContextVar("room_history_read_path", default=None)  # optional observer for local acceptance instrumentation
_DIAGNOSTICS = {
    ("ouroboros.projects_registry", "readable_rooms"): "The Project registry vouches for no room; Main's saved history is withheld",
    ("ouroboros.gateway.history", "_collect_chat_rows"): "Failed to read chat history",
    ("ouroboros.gateway.history", "_collect_progress_rows"): "Failed to read progress log",
    ("ouroboros.i18n_memory", "cached_memory"): "Malformed translation memory ignored",
    ("ouroboros.delegate_custody_usage", "cost_snapshot"): "Cost custody unavailable",
}
_DIAGNOSTIC_LINES = {message.encode("ascii") + b"\n" for message in _DIAGNOSTICS.values()}
_CURSOR_KINDS = (None, "after", "older", "page")


class ReaderUnavailable(Exception):
    """The process could not give a complete answer. An HTTP refusal is an answer."""


class ReaderSwitching(Exception):
    """The serving generation must not open another store lease."""


class ReaderDisconnected(Exception):
    """The HTTP peer left while waiting for an unused reader slot."""


def _frame(out, metadata, body=b""):
    meta = json.dumps(metadata, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("ascii")
    if len(meta) > _MAX_METADATA:
        raise ValueError("history reader metadata is too large")
    out.write(_HEADER.pack(len(meta), len(body)))
    out.write(meta)
    if body:
        out.write(body)
    out.flush()


def _receive_metadata(stream):
    header = stream.read(_HEADER.size)
    if len(header) != _HEADER.size:
        raise EOFError
    size, length = _HEADER.unpack(header)
    if not 0 < size <= _MAX_METADATA:
        raise ValueError("invalid history reader metadata length")
    meta = stream.read(size)
    if len(meta) != size:
        raise EOFError
    return json.loads(meta), length


def _receive_body(stream, length):
    body = stream.read(length)
    if len(body) != length:
        raise EOFError
    return body


def _receive(stream):
    meta, length = _receive_metadata(stream)
    return meta, _receive_body(stream, length)


async def _disconnected(receive):
    # ASGI delivers this event even when the server does not cancel the GET's
    # handler. Receive blocks after the request body; no polling or response I/O.
    while (await receive())["type"] != "http.disconnect":
        pass


class HistoryReader:
    def __init__(self):
        self._lock = threading.Condition()
        self._write_lock = threading.Lock()
        self._process = None
        self._thread = None
        self._stopped = False
        self._handoff = None
        self._unconfirmed_handoffs = set()
        self._closing = False
        self._root = None
        self._requests = {}
        self._next_id = itertools.count(1)
        self._server_reads = 0
        self.ready = threading.Event()
        self._slots = asyncio.Semaphore(PARALLEL_READS)
        self._unavailable = asyncio.Event()
        self._wait_loop = None
        self._failure = "not_started"
        self._reported_fallback = None

    def bind(self, root):
        """Name the serving root before the boot callbacks; the handoff can fence even a late spawn."""
        self._root = Path(root).resolve()

    @property
    def pid(self):
        process = self._process
        return process.pid if self.ready.is_set() and process is not None else None

    def start(self, data_root, repo_dir, *, check_checkout=False):
        """Called once on publication, even providerless. Does not wait for spawn or readiness."""
        with self._lock:
            if self._thread is not None or self._stopped:
                return
            if self._root is None:
                self.bind(data_root)
            self._failure = "starting"
            thread = threading.Thread(target=self._run, args=(self._root, Path(repo_dir), check_checkout),
                                      name="room-history-reader", daemon=True)
            self._thread = thread
            try:
                thread.start()
            except RuntimeError:
                self._thread = None
                self._failure = "start_failed"
                log.warning("room_history_reader could not start its launch thread")

    def _own(self, process):
        # Publish immediately after Popen, before custody I/O. A concurrent stop
        # either sees it, or its flag is seen here. Neither waits for that I/O.
        self._process = process
        if self._stopped:
            self._kill(process)

    @staticmethod
    def _kill(process):
        # A captured Popen; the reader makes no subprocesses of its own.
        with contextlib.suppress(OSError):
            process.kill()

    def stop(self):
        """Immediate request only. No reader lock, pipe I/O, join or log on Panic."""
        self._closing = True
        self._stop_process()

    def _stop_process(self):
        self._stopped = True
        process = self._process
        if process is not None:
            self._kill(process)
        self._wake_slot_waiters()

    def _wake_slot_waiters(self):
        # The owning HTTP loop wakes all undispatched reads. The stop thread
        # neither acquires the reader lock nor walks the requests on Panic.
        if (loop := self._wait_loop) is not None:
            with contextlib.suppress(RuntimeError):  # the loop may already have closed
                loop.call_soon_threadsafe(self._unavailable.set)

    def live_children(self, *, include_server_reads=True):
        with self._lock:
            process, thread = self._process, self._thread
            live = [PURPOSE + ":server_reads"] if include_server_reads and self._server_reads else []
        if process is not None and process.poll() is None:
            live.append(process.pid)
        elif thread is not None and thread.is_alive():
            live.append(PURPOSE + ":launch_or_reap")
        return live

    def retire_for_checkout(self, root):
        """Keep the existing server read path during a live checkout mutation."""
        if self._root != Path(root).resolve():
            return ""
        with self._lock:
            self._failure = "checkout_changing"
            self._stop_process()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=_HANDOFF_SECONDS)
        return "room history reader stop unconfirmed" if self.live_children(include_server_reads=False) else ""

    def prepare_handoff(self, root):
        """Fence reads and give this transfer an identity, even when quiescence is refused."""
        if self._root != root:
            return None, ""
        with self._lock:
            token = self._handoff = object()
            self._unconfirmed_handoffs.add(token)
        self._stop_process()
        return token, self.wait_stopped()

    def wait_stopped(self):
        """Bounded handoff/restart observation. Panic only calls stop()."""
        deadline = time.monotonic() + _HANDOFF_SECONDS
        with self._lock:
            if not self._lock.wait_for(lambda: not self._server_reads, max(0, deadline - time.monotonic())):
                return "room history server read still in use"
            thread = self._thread
        if thread is not None:
            thread.join(timeout=max(0, deadline - time.monotonic()))
        return "room history reader stop unconfirmed" if self.live_children() else ""

    def continue_inline(self, token):
        # The transfer owner has proof this process continues. Another attempt's
        # fence (including an old unknown outcome) is not ours to release.
        with self._lock:
            self._unconfirmed_handoffs.discard(token)

    def _reopen_inline(self, root, token):
        from ouroboros.store_switch import inline_history_ready, no_switch_intent
        from ouroboros.store import forget_handoffs

        ready = inline_history_ready(root)  # only a real read, never under the admission lock
        with self._lock:
            if self._closing:
                raise ReaderSwitching
            if self._handoff is not None:
                if self._handoff is not token or not ready or not no_switch_intent(root):
                    raise ReaderSwitching
                forget_handoffs()
                self._handoff = None

    def _run(self, root, repo, check_checkout=False):
        failure = "start_failed"
        diagnostics = None
        try:
            # Optional launch admission is not part of server startup. An
            # unreadable checkout selects the existing in-server reader.
            if check_checkout:
                failure = "checkout_probe_failed"
                from supervisor.update_merge import assisted_checkout_active

                if assisted_checkout_active():
                    failure = "checkout_changing"
                    return
                failure = "start_failed"
            from ouroboros.deployment_policy import child_policy_env
            from ouroboros.startup_historical_audit import spawn_maintenance_job

            if self._stopped:
                return
            # The packaged launcher installs runtime dependencies into this
            # user site. Retain its Python selection, not its whole environment.
            env = {key: os.environ[key] for key in ("PYTHONUSERBASE", "PYTHONNOUSERSITE") if key in os.environ}
            process = spawn_maintenance_job(root, repo, PURPOSE, direct_interpreter=True,
                                            extra_env={**child_policy_env(), **env, "PYTHONIOENCODING": "utf-8"},
                                            stdin=subprocess.PIPE, stderr=subprocess.PIPE, on_spawn=self._own)
            diagnostics = threading.Thread(target=_log_stderr, args=(process.stderr,),
                                           name="room-history-stderr", daemon=True)
            diagnostics.start()
            hello, body = _receive(process.stdout)
            if hello != {"ready": process.pid} or body:
                raise ValueError("invalid history reader readiness")
            failure = "connection_lost"
            if self._stopped:
                return
            log.info("room_history_reader ready pid=%s", process.pid)
            self.ready.set()
            while not self._stopped:
                self._read_reply(process.stdout)
        except Exception as exc:
            if not self._stopped:
                log.warning("room_history_reader %s (%s)", failure, type(exc).__name__)
        finally:
            self.ready.clear()
            with self._lock:
                if self._failure != "checkout_changing":
                    self._failure = failure
                requests = list(self._requests.values())
                self._requests.clear()
                for future, _ in requests:
                    future.set_exception(ReaderUnavailable(self._failure))
            self._wake_slot_waiters()
            if (process := self._process) is not None:
                if process.poll() is None:
                    self._kill(process)
                # Outside Panic. Close the exact pipe ends after the process is gone.
                process.wait()
                with contextlib.suppress(OSError):
                    process.stdout.close()
                with self._write_lock, contextlib.suppress(OSError):
                    process.stdin.close()  # a failed send can leave bytes to flush to a dead child
                if diagnostics is None or diagnostics.ident is None:
                    process.stderr.close()

    def _read_reply(self, stream):
        reply, length = _receive_metadata(stream)
        request_id, failed, status, kind = reply.get("id"), reply.get("failed"), reply.get("status"), reply.get("cursor_kind")
        if (type(request_id) is not int or request_id <= 0 or
                (failed is not None and (failed is not True or length or status is not None or kind is not None)) or
                (failed is None and (type(status) is not int or not 100 <= status <= 599 or kind not in _CURSOR_KINDS))):
            raise ValueError("invalid history reader reply")
        with self._lock:
            _, decoded = self._requests.get(request_id, (None, None))
            if kind is not None and decoded is not None:
                decoded["kind"] = kind  # one matching frame's metadata, even if delivery ends before its body
        body = _receive_body(stream, length)
        with self._lock:
            if entry := self._requests.pop(request_id, None):
                future, _ = entry
                if failed:
                    future.set_exception(ReaderUnavailable("worker_assembly_failed"))
                else:
                    future.set_result((status, body, kind))
            # An abandoned dispatch may already have left the table. Drain its
            # frame to keep the next reply readable; do not invent a broken pipe.

    def _send(self, request_id, selectors, live):
        with self._write_lock:
            process = self._process
            if not self.ready.is_set() or self._stopped:
                raise ReaderUnavailable(self._failure)
            try:
                _frame(process.stdin, {"id": request_id, "selectors": selectors, "live": live})
            except (OSError, TypeError, ValueError) as exc:
                # An uncertain write never resubmits to this transport. The read
                # itself is effect-free; this request may go inline once.
                if isinstance(exc, OSError):
                    self._kill(process)
                raise ReaderUnavailable("send_failed") from exc

    def _server_read(self, root, produce, *args):
        # Count the physical reader, even if the event loop tears down its
        # awaiting coroutine. Admission and the switch share this lock.
        if self._root != root:
            return produce(*args)
        with self._lock:
            if self._closing or self._unconfirmed_handoffs:
                raise ReaderSwitching
            handoff = self._handoff
            self._server_reads += 1
        try:
            if handoff is not None:
                self._reopen_inline(root, handoff)
            return produce(*args)
        finally:
            with self._lock:
                self._server_reads -= 1
                self._lock.notify_all()

    def _fallback(self, root, inline, reason):
        return self._server_read(root, self._read_inline, root, inline, reason)

    def _read_inline(self, root, inline, reason):
        if self._root == root:
            with self._lock:
                changed = self._reported_fallback != reason
                self._reported_fallback = reason
            if changed:
                log.warning("room_history_reader_fallback cause=%s", reason)
        return inline()

    async def _acquire(self, receive):
        if not self._slots.locked():
            await self._slots.acquire()
            return
        self._wait_loop = asyncio.get_running_loop()
        if self._stopped or not self.ready.is_set():
            raise ReaderUnavailable(self._failure)
        acquire = asyncio.create_task(self._slots.acquire())
        unavailable = asyncio.create_task(self._unavailable.wait())
        disconnect = asyncio.create_task(_disconnected(receive)) if receive is not None else None
        waits = (acquire, unavailable, disconnect) if disconnect is not None else (acquire, unavailable)
        admitted = False
        try:
            done, _ = await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
            if disconnect in done:
                disconnect.result()
                raise ReaderDisconnected
            if unavailable in done:
                raise ReaderUnavailable(self._failure)
            admitted = acquire.result()
        finally:
            for wait in waits[1:]:
                if not wait.done():
                    wait.cancel()
                elif not wait.cancelled():
                    wait.exception()
            if not acquire.done():
                acquire.cancel()  # Semaphore returns a permit granted during cancellation
            elif not acquire.cancelled() and acquire.exception() is None and not admitted:
                self._slots.release()  # both completed, or the outer task was cancelled

    async def read(self, root, selectors, inline, *, receive=None):
        """Use a resolved route root and reserve slots only for a ready reader."""
        reason = self._failure
        try:
            if self._root == root and self.ready.is_set() and not self._stopped:
                from ouroboros.gateway.history_live import capture

                # A cancelled waiter has not started work; a dispatched build
                # keeps its slot after its HTTP client goes away.
                try:
                    await self._acquire(receive)
                    if self.ready.is_set() and not self._stopped:
                        task = asyncio.create_task(self._dispatch(root, selectors, capture))
                        task.add_done_callback(self._release_slot)
                        result = await asyncio.shield(task)
                        _observed("child")
                        return result
                    else:
                        self._slots.release()
                        reason = self._failure
                except ReaderUnavailable as exc:
                    reason = str(exc)
            # No child reservation holds server fallback, including when the
            # reader died while this request was waiting for a slot.
            result = await asyncio.to_thread(self._fallback, root, inline, reason)
            _observed("server")
            return result
        except ReaderSwitching:
            return switching_response(selectors["cursor"])

    async def _dispatch(self, root, selectors, capture):
        from ouroboros.gateway.history_paging import DECODED_CURSOR

        try:
            live = await asyncio.to_thread(self._server_read, root, capture, root, selectors)
        except ReaderSwitching:
            raise
        except Exception as exc:
            raise ReaderUnavailable("live_inputs_unavailable") from exc
        request_id, future = next(self._next_id), Future()
        future.set_running_or_notify_cancel()  # loop shutdown cannot cancel the physical reply
        with self._lock:
            self._requests[request_id] = (future, DECODED_CURSOR.get())
        try:
            await asyncio.to_thread(self._send, request_id, selectors, live)
            return await asyncio.wrap_future(future)
        finally:
            with self._lock:
                self._requests.pop(request_id, None)

    def _release_slot(self, task):
        self._slots.release()
        if not task.cancelled():
            task.exception()


def _log_stderr(stream):
    # Drain everything, publishing only the reader's fixed statements. Do not
    # relay arbitrary exceptions, raw input, or any part of a framed response.
    with stream:
        while raw := stream.readline(4096):
            if raw.endswith(b"\r\n"):
                raw = raw[:-2] + b"\n"
            if raw in _DIAGNOSTIC_LINES:
                log.warning("room_history_reader: %s", raw.decode("ascii").rstrip())


class _ReaderDiagnostics(logging.Handler):
    def emit(self, record):
        if message := _DIAGNOSTICS.get((record.name, record.funcName)):
            print(message, file=sys.stderr, flush=True)


def _observed(path):
    if observer := READ_PATH.get():
        with contextlib.suppress(Exception):  # measurement cannot change the selected answer
            observer(path)


def switching_response(cursor):
    from ouroboros.gateway.history_response import refusal

    _observed("handoff")
    return 503, refusal("migrating", cursor), None


reader = HistoryReader()


def begin_generation(root):
    global reader
    reader = HistoryReader()
    reader.bind(root)
    return reader


def _responses(source, output, data_root):
    from ouroboros.gateway import history  # complete imports before READY
    from ouroboros.gateway.history_response import assemble
    from ouroboros.store import thread_read_connections

    write_lock = threading.Lock()

    def respond(request):
        try:
            with thread_read_connections():
                status, body, kind = assemble(history._assemble_history_response, data_root,
                                              **request["selectors"], live=request["live"])
            meta = {"id": request["id"], "status": status, "cursor_kind": kind}
        except Exception:
            meta, body = {"id": request["id"], "failed": True}, b""
        try:
            with write_lock:
                _frame(output, meta, body)
        except (OSError, ValueError, struct.error):
            # Make the controller observe EOF, not wait on a detached worker
            # that could no longer deliver a frame. This interpreter owns only reads.
            os._exit(1)

    _frame(output, {"ready": os.getpid()})
    with ThreadPoolExecutor(PARALLEL_READS, thread_name_prefix="history-read") as threads:
        while True:
            try:
                request, body = _receive(source)
            except EOFError:
                return
            if body:
                raise ValueError("unexpected history request body")
            threads.submit(respond, request)


def main():
    import argparse

    # Hold private duplicates; even a native print or stdin read in an import
    # cannot produce frames or consume a request. Do this before importing readers.
    source, output = os.fdopen(os.dup(0), "rb"), os.fdopen(os.dup(1), "wb")
    with open(os.devnull, "r+b") as devnull:
        os.dup2(devnull.fileno(), 0)
        os.dup2(devnull.fileno(), 1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--custody-session", required=True)
    args = parser.parse_args()
    from ouroboros.process_custody import adopt_session_id, start_parent_lifeline

    adopt_session_id(args.custody_session)
    start_parent_lifeline()
    logging.basicConfig(level=logging.WARNING, handlers=[_ReaderDiagnostics()], force=True)
    with source, output:
        _responses(source, output, args.data_root)


if __name__ == "__main__":
    main()
