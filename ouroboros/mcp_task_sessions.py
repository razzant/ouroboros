"""Task-held MCP sessions for servers whose state lives in one MCP session.

``mcp_client`` opens a fresh session for every call. A server configured with
``"session_scope": "task"`` instead keeps ONE session per Ouroboros task: it is
opened by that task's first call and reused by its later calls. The motivating
server is Microsoft's Playwright Extension bridge (``@playwright/mcp
--extension``): its connection to the owner's running Chrome, and the tab group
that connection controls, live exactly as long as one MCP session, so a fresh
session per call would request a new connection every time and lose the page
between calls.

A session is keyed by ``(server id, task id)`` and is never handed to another
task; which callers may open one at all is the registry's decision
(``registry_guards.task_session_block_reason``). A session closes when its
task's loop exits (``close_task_sessions``), when its server's Settings entry
changes, is disabled or is removed (``retain_sessions``), and when one of its
calls fails or times out. A task whose loop closed its sessions is terminal
here: a later call under that task id (a tool thread that outlived its round)
never opens a new connection in this process. Whether a NEW session reaches the
owner's browser without a click is the server's own decision; nothing here
claims it.

A ``stdio`` server is spawned here, not by the SDK's stdio transport, and the
public ``ClientSession`` speaks JSON-RPC over its pipes. The SDK ends only a
leader that ignores stdin EOF, so a descendant a leader left behind kept running
after an ordinary close. Here ``spawn_supervised`` starts the server as the
leader of its own process group and records a durable ``task``-scope custody row
(pid, pgid, owning task, spawning process) before any byte is exchanged. Every
close is physical: the whole group is signalled while its owned leader still
identifies it, and descendants captured from the leader's tree are killed; the
row is released only
once all of them are gone. Otherwise the row and the session record stay, no
replacement starts and the receipt says so. Other owners reach the same group
through that row: a worker tree kill stops the sessions its worker spawned,
Panic signals this process's groups at once and settles recorded groups with
identified leaders. A group that outlived a dead leader remains unresolved:
its numeric PGID alone cannot authorize a later signal.

Limits: a descendant that left the group (``setsid``) and the leader's tree
before close escapes; a spawn killed between ``Popen`` and its custody row is
unrecorded (``spawn_supervised``). Windows has no process groups here: close
kills the leader and its captured tree only, and none of this is verified on
Windows.

A stdio server without a configured ``cwd`` runs in a private temporary
directory removed when its session closes: Playwright MCP writes page snapshots
under its working directory, and the owner's pages must not land in whatever
directory the host process happens to run in.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import logging
import os
import pathlib
import shutil
import subprocess
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Tuple

from ouroboros import mcp_client, process_custody
from ouroboros import platform_layer as _pl
from ouroboros.process_containment import pid_is_zombie, process_group_has_live_members
from ouroboros.tools.tool_result import ToolResult

log = logging.getLogger(__name__)

_CLOSE_WAIT_SEC = 5.0
_EOF_GRACE_SEC = 2.0  # the SDK's own grace between stdin EOF and termination
_SETTLE_SEC = 3.0
_STDERR_TAIL_BYTES = 4000


def _custody_root() -> pathlib.Path:
    """The data root whose ledger the reaper, worker tree kill and Panic read."""
    from ouroboros import config

    return pathlib.Path(config.DATA_DIR)


class _Stdin:
    """The server's stdin: writes and close never contend on a buffered lock.

    A writer blocked on a full pipe holds ``_lock``; close then gives up and the
    group kill that follows breaks the write, so EOF can never stall a close.
    """

    def __init__(self, pipe: Any) -> None:
        self._pipe, self._lock, self._closed = pipe, threading.Lock(), False

    def write(self, data: bytes) -> None:
        with self._lock:
            if self._closed:
                raise BrokenPipeError("MCP server stdin is closed")
            view = memoryview(data)
            while view:
                view = view[os.write(self._pipe.fileno(), view):]

    def close(self, timeout: float) -> bool:
        if not self._lock.acquire(timeout=max(0.0, timeout)):
            return False
        try:
            if not self._closed:
                self._closed = True
                self._pipe.close()
            return True
        finally:
            self._lock.release()


class _StdioCustody:
    """One task session's spawned server: its process group and custody row.

    Safe from any thread. The leader is reaped only under ``_signal_lock`` after
    its group was signalled, so no signal here can reach a recycled group id.
    ``released`` goes True once, only when leader, group and captured
    descendants are gone; a session that never spawned holds nothing.
    """

    def __init__(self, cfg: mcp_client.MCPServerConfig, task_id: str) -> None:
        self.task_id, self.purpose = task_id, process_custody.TASK_SESSION_PURPOSE_PREFIX + cfg.id
        self.root = _custody_root()
        self.proc: Any = None
        self.stdin: Any = None
        self.pgid = 0
        self.children: set = set()
        self.released, self.detail, self.refused = True, "", ""
        self._signal_lock, self._settle_lock = threading.Lock(), threading.Lock()

    def publish(self, proc: subprocess.Popen) -> None:
        """``spawn_supervised`` callback: owned before custody I/O, or refused."""
        with self._signal_lock:
            self.proc, self.stdin, self.released = proc, _Stdin(proc.stdin), False
            self.pgid = _pl.process_group_id(proc.pid)
            refused = self.refused or _emergency
        if refused:
            raise RuntimeError(refused)  # spawn_supervised kills it before recording

    def request_stop(self) -> Dict[str, Any]:
        """Emergency Stop's request: signal the group now, no scan, wait or disk."""
        self.refused = "Emergency Stop requested"
        if not self._signal_lock.acquire(timeout=0.05):
            return {"requested": False, "error": "custody busy; settlement still owed"}
        try:
            proc = self.proc
            if proc is None or proc.returncode is not None:
                return {"requested": False, "error": "no unreaped server process"}
            if _pl.IS_WINDOWS:
                return _pl.request_process_tree_kill(proc)
            if self.pgid == proc.pid and _pl.process_group_id(proc.pid) == self.pgid:
                _pl.kill_process_group_id(self.pgid)
            elif self.pgid != proc.pid:
                _pl.force_kill_pid(proc.pid)
            else:
                return {"pid": proc.pid, "pgid": self.pgid, "requested": False,
                        "error": "server group identity unconfirmed; custody retained"}
            return {"pid": proc.pid, "pgid": self.pgid, "requested": True,
                    "scope": "group" if self.pgid == proc.pid else "process"}
        finally:
            self._signal_lock.release()

    def _leader_exited(self, proc: subprocess.Popen) -> bool:
        if _pl.IS_WINDOWS:
            return proc.poll() is not None
        # Never reap here: a reaped leader's group id could be reused before the kill.
        return proc.returncode is not None or pid_is_zombie(proc.pid) or not _pl.pid_is_alive(proc.pid)

    def _quiescent(self, proc: subprocess.Popen) -> bool:
        group = self.pgid > 0 and self.pgid == proc.pid and process_group_has_live_members(self.pgid)
        return (proc.returncode is not None and not group
                and not any(_pl.pid_is_alive(pid) and not pid_is_zombie(pid) for pid in self.children))

    def _is_row(self, entry: Dict[str, Any]) -> bool:
        return (int(entry.get("pid") or 0) == self.proc.pid and entry.get("purpose") == self.purpose
                and entry.get("owner_task") == self.task_id
                and int(entry.get("spawner_pid") or 0) == os.getpid())

    def settle(self, *, grace: float, timeout: float) -> bool:
        """End the server physically, then release its custody row; True once released.

        Signal the group while the owned leader identifies it, then close stdin
        and wait up to ``grace``. An unconfirmed stop keeps the row and says why.
        """
        if not self._settle_lock.acquire(timeout=max(0.0, grace + timeout)):
            return self.released
        try:
            proc = self.proc
            if proc is None or self.released:
                return self.released
            grace_end = time.monotonic() + max(0.0, grace)
            deadline = grace_end + max(0.0, timeout)
            # Signal while the owned Popen still reserves its leader PID. A
            # server may exit immediately on EOF; after it is reaped the same
            # numeric pgid could belong to somebody else.
            with self._signal_lock:
                if proc.returncode is None:
                    self.children.update(_pl.collect_descendant_pids(proc.pid))
                    if (self.pgid > 0 and self.pgid == proc.pid
                            and _pl.process_group_id(proc.pid) == self.pgid):
                        _pl.kill_process_group_id(self.pgid)
                    for pid in [*self.children, proc.pid]:
                        _pl.force_kill_pid(pid)
            if self.stdin.close(timeout=grace):
                while time.monotonic() < grace_end and not self._leader_exited(proc):
                    time.sleep(0.05)
            while time.monotonic() < deadline:
                with self._signal_lock:
                    if proc.poll() is not None:
                        break
                time.sleep(0.05)
            failures: List[str] = []
            stopped = process_custody.stop_group_custody(
                self.root, self._is_row, timeout_sec=max(0.0, deadline - time.monotonic()),
                reason="task_session_close", unconfirmed=failures,
            )
            self.stdin.close(timeout=0)
            # Ledger settlement can race a descendant's survival observation.
            # Neither a removed row nor a quiet leader alone proves physical close.
            self.released = (proc.pid in stopped or not failures) and self._quiescent(proc)
            self.detail = "" if self.released else (
                "; ".join(failures) or f"server process {proc.pid} group {self.pgid} not confirmed stopped")
            return self.released
        finally:
            self._settle_lock.release()


@asynccontextmanager
async def _custodied_stdio(cfg: mcp_client.MCPServerConfig, task_id: str, custody: _StdioCustody):
    """``stdio`` streams for ``ClientSession`` over a custodied server process."""
    import anyio
    from mcp import types
    from mcp.client.stdio import get_default_environment

    try:
        from mcp.shared.message import SessionMessage
    except ImportError:  # SDKs before 1.8 exchange bare JSON-RPC messages
        SessionMessage = None
    # The SDK's environment contract: its small default set plus configured names.
    env = {**get_default_environment(), **{key.upper() if _pl.IS_WINDOWS else key: value
                                           for key, value in cfg.env.items()}}
    command = (shutil.which(cfg.command, path=env.get("PATH")) or cfg.command) if _pl.IS_WINDOWS else cfg.command
    stderr = tempfile.TemporaryFile()
    try:
        proc = process_custody.spawn_supervised(
            [command, *cfg.args], drive_root=custody.root, purpose=custody.purpose, scope="task",
            owner_task_id=task_id, spawner_pid=os.getpid(), on_spawn=custody.publish,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, cwd=cfg.cwd or None, env=env,
        )
    except BaseException:
        custody.settle(grace=0, timeout=_SETTLE_SEC)
        stderr.close()
        raise
    read_writer, read_stream = anyio.create_memory_object_stream(0)
    write_stream, write_reader = anyio.create_memory_object_stream(0)

    async def stdout_reader() -> None:
        try:
            async with read_writer:
                while line := await anyio.to_thread.run_sync(proc.stdout.readline, abandon_on_cancel=True):
                    try:
                        message: Any = types.JSONRPCMessage.model_validate_json(line.decode("utf-8", "replace"))
                    except Exception as exc:  # noqa: BLE001 - delivered to the session, as the SDK does
                        await read_writer.send(exc)
                        continue
                    await read_writer.send(SessionMessage(message) if SessionMessage else message)
            proc.stdout.close()  # EOF: no reader thread is left on it
        except (anyio.ClosedResourceError, anyio.BrokenResourceError):
            pass

    async def stdin_writer() -> None:
        try:
            async with write_reader:
                async for item in write_reader:
                    message = getattr(item, "message", item)
                    data = (message.model_dump_json(by_alias=True, exclude_none=True) + "\n").encode("utf-8")
                    await anyio.to_thread.run_sync(custody.stdin.write, data, abandon_on_cancel=True)
        except (OSError, anyio.ClosedResourceError, anyio.BrokenResourceError):
            pass

    try:
        async with anyio.create_task_group() as tg:
            tg.start_soon(stdout_reader)
            tg.start_soon(stdin_writer)
            try:
                yield read_stream, write_stream
            finally:
                # Synchronous on purpose: cancellation of this task cannot skip it.
                if not custody.settle(grace=_EOF_GRACE_SEC, timeout=_SETTLE_SEC):
                    log.error("MCP task session %r of task %s: %s", cfg.id, task_id, custody.detail)
                tg.cancel_scope.cancel()
                for stream in (read_stream, write_stream, read_writer, write_reader):
                    stream.close()
    finally:
        stderr.seek(max(0, stderr.seek(0, os.SEEK_END) - _STDERR_TAIL_BYTES))
        diagnostic = mcp_client._redact_error_text(stderr.read().decode("utf-8", "replace"), cfg)
        stderr.close()
        if diagnostic.strip():
            log.info("MCP task session stderr tail (%s): %s", cfg.id, diagnostic)


@asynccontextmanager
async def _client_session(cfg: mcp_client.MCPServerConfig, task_id: str, custody: _StdioCustody):
    """Open and initialize one MCP session named after the owning task."""
    if not mcp_client._MCP_SDK_AVAILABLE:
        raise RuntimeError("MCP client SDK not installed. Add `mcp>=1.6` to the runtime.")
    from mcp import types

    # The Playwright Extension names the connection's tab group after this.
    client_info = types.Implementation(name=f"Ouroboros task {task_id}", version="1")
    scratch = tempfile.mkdtemp(prefix="ouroboros-mcp-task-") if cfg.transport == "stdio" and not cfg.cwd else ""
    cfg = dataclasses.replace(cfg, cwd=scratch or cfg.cwd)
    try:
        transport = (_custodied_stdio(cfg, task_id, custody) if cfg.transport == "stdio"
                     else mcp_client._transport_factory(cfg))
        async with transport as streams:
            read, write = streams[0], streams[1]
            async with mcp_client.ClientSession(read, write, client_info=client_info) as session:
                await session.initialize()
                yield session
    finally:
        if scratch:
            shutil.rmtree(scratch, ignore_errors=True)


class _HeldSession:
    """One task's live MCP session, served by a private event-loop thread.

    Monotonic: once closing it never serves another call, and it counts as
    closed only when its thread ended AND its server's custody was released.
    """

    def __init__(self, cfg: mcp_client.MCPServerConfig, task_id: str, opener: Any) -> None:
        self.cfg, self.task_id, self._opener = cfg, task_id, opener
        self.custody = _StdioCustody(cfg, task_id)
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._loop: Any = None
        self._main: Any = None
        self._closing: Any = None
        self._session: Any = None
        self._error: BaseException | None = None
        self._close_requested = False
        self._thread = threading.Thread(target=self._run, name=f"mcp-task-session-{cfg.id}", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            asyncio.run(self._hold())
        except BaseException as exc:  # noqa: BLE001 - surfaced by the next call on this session
            self._error = self._error or exc
        finally:
            self._session = None
            self._ready.set()

    async def _hold(self) -> None:
        with self._lock:
            if self._close_requested:
                return
            self._closing = asyncio.Event()
            self._loop, self._main = asyncio.get_running_loop(), asyncio.current_task()
        async with self._opener(self.cfg, self.task_id, self.custody) as session:
            self._session = session
            self._ready.set()
            await self._closing.wait()  # an ordinary exit runs the transport's own shutdown

    def _stop_on_loop(self) -> None:
        if self._session is not None:
            self._closing.set()
        elif self._main is not None:
            self._main.cancel()  # still opening: nothing to shut down gracefully yet

    def alive(self) -> bool:
        return self._thread.is_alive() and not self._close_requested

    def closed(self) -> bool:
        return not self._thread.is_alive() and self.custody.released

    def call(self, raw_name: str, arguments: Dict[str, Any], timeout: int) -> Any:
        deadline = time.monotonic() + timeout
        if not self._ready.wait(timeout):
            raise asyncio.TimeoutError()
        with self._lock:
            if self._close_requested:
                raise RuntimeError("task session is closing; no new MCP call was submitted")
            session, loop = self._session, self._loop
            if session is None or loop is None:
                error = self._error
                raise RuntimeError(f"task session is not open ({type(error).__name__}: {error})" if error
                                   else "task session is closed")
            # The same lock serializes the admission with close(). A close that
            # won before this point cannot be followed by a late tool submit.
            future = asyncio.run_coroutine_threadsafe(session.call_tool(raw_name, arguments), loop)
        try:
            return future.result(timeout=max(0.0, deadline - time.monotonic()))
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise asyncio.TimeoutError() from None

    def close(self, *, wait: float) -> bool:
        """Request shutdown; True only once the thread ended and custody was released.

        A waiting caller that outlives the loop's own EOF-then-kill path ends the
        server from here: that does not depend on the session thread.
        """
        with self._lock:
            self._close_requested = True
            loop = self._loop
        self.custody.refused = self.custody.refused or "task session is closing"
        if loop is not None:
            try:
                loop.call_soon_threadsafe(self._stop_on_loop)
            except RuntimeError:
                pass  # the loop already finished
        if wait > 0:
            deadline = time.monotonic() + wait
            self._thread.join(min(wait, _EOF_GRACE_SEC + 1.0))
            if not self.closed():
                self.custody.settle(grace=0, timeout=max(0.0, deadline - time.monotonic()))
                self._thread.join(max(0.0, deadline - time.monotonic()))
        return self.closed()


_lock = threading.Lock()
_sessions: Dict[Tuple[str, str], _HeldSession] = {}
_terminal_tasks: set = set()  # monotonic: a task id never leaves it
_emergency = ""  # monotonic: set once by Emergency Stop in this process
_opener: Any = _client_session  # test hook; production opens the configured transport


def call_task_session(cfg: mcp_client.MCPServerConfig, task_id: str, raw_name: str,
                      arguments: Dict[str, Any], timeout: int) -> ToolResult:
    """Call ``raw_name`` on the calling task's own session, opening it if needed.

    A failed or timed-out call requests closure. Until that session is confirmed
    closed its owner cannot open a replacement; a terminal task never can.
    """
    task_id = str(task_id or "").strip()
    if not task_id:
        raise ValueError(f"MCP server {cfg.id!r} keeps one session per task and the caller has no task id")
    key = (cfg.id, task_id)
    with _lock:
        held = _sessions.get(key)
        if held is not None and held.closed():
            del _sessions[key]
            held = None
        if _emergency:
            reason = f"{_emergency}; no MCP task session may open in this process"
        elif task_id in _terminal_tasks:
            reason = (f"task {task_id} already closed its MCP task sessions when its run exited; "
                      "a later call cannot reopen one")
        elif held is not None and (held.cfg != cfg or not held.alive()):
            reason = f"MCP task session {cfg.id!r} has not confirmed disconnection; no replacement started"
        else:
            reason = ""
        if held is None and not reason:
            held = _sessions[key] = _HeldSession(cfg, task_id, _opener)
    if reason:
        if held is not None:
            held.close(wait=0)
        raise RuntimeError(reason)
    try:
        result = held.call(raw_name, arguments, timeout)
    except BaseException:
        if held.close(wait=0):
            with _lock:
                if _sessions.get(key) is held:
                    del _sessions[key]
        raise
    return mcp_client._tool_result_from_call_result(result)


def close_task_sessions(task_id: str, *, wait: float = _CLOSE_WAIT_SEC) -> List[Dict[str, Any]]:
    """End every session of a task whose run exited; the task becomes terminal here.

    ``closed`` is True only when a session's server process group is confirmed
    gone. An unconfirmed one keeps its record and custody row for another attempt,
    for Panic and for the reaper, and its receipt carries the reason.
    """
    task_id = str(task_id or "")
    with _lock:
        if task_id:
            _terminal_tasks.add(task_id)
        held = [(key, item) for key, item in _sessions.items() if key[1] == task_id]
    deadline = time.monotonic() + wait
    receipts = []
    for key, item in held:
        closed = item.close(wait=max(0.0, deadline - time.monotonic()))
        if closed:
            with _lock:
                if _sessions.get(key) is item:
                    del _sessions[key]
        receipts.append({"server_id": item.cfg.id, "task_id": item.task_id, "closed": closed,
                         **({} if closed else {"detail": item.custody.detail or "session thread still running"})})
    for receipt in receipts:
        log.info("MCP task session %s: %s", "closed" if receipt["closed"] else "close requested, not yet confirmed",
                 receipt)
    return receipts


def request_emergency_stop(*, request_only: bool = True) -> List[Dict[str, Any]]:
    """Panic's request phase: refuse new sessions and signal this process's groups.

    No lock, scan, wait or disk. Settlement is the recorded-group stop
    (``process_custody.stop_task_session_groups``), which covers every process.
    """
    global _emergency
    _emergency = "Emergency Stop requested"
    return [{"server_id": item.cfg.id, "task_id": item.task_id, **item.custody.request_stop()}
            for item in _sessions.copy().values()]


def retain_sessions(live: Dict[str, mcp_client.MCPServerConfig]) -> None:
    """Close, without waiting, sessions whose server is no longer ``live`` with the same config."""
    with _lock:
        stale = [(key, item) for key, item in _sessions.items() if live.get(key[0]) != item.cfg]
    for key, item in stale:
        log.info("MCP task session of task %s on server %r closed: its Settings entry changed", item.task_id, item.cfg.id)
        if item.close(wait=0):
            with _lock:
                if _sessions.get(key) is item:
                    del _sessions[key]
