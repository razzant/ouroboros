"""Task-owned stdio MCP sessions: one started, initialized server for one live task loop.

An owner opts a stdio server in with ``task_session: true`` (``mcp_client``). Its
discovery is unchanged: boot, Save, Refresh and Settings tests each list through a
fresh process. A CALL from a task's own tool loop starts one process, initializes
it once, and every later call of that same loop reuses it, so a stateful server (a
browser) keeps its state between the model's calls.

Ownership. A session lives on the exact ToolContext of the task's live loop: the
context the task's model-wait owner is bound to (``TaskModelWait.tool_context``),
never a copy (review collectors, voice views, readers) and never a process-wide
task/attempt table; the attempt only has to agree with that owner. The loop's exit
(``loop_budget._cleanup_loop_resources``, before its cold-pause early return) and
the agent turn's ``finally`` close it and retire the context, so a late thread
cannot start another. A warm owner wait inside the loop keeps it, except a moved
Presence author's park under the owner's Pause: its own processes settle before it
lends the conversation (``presence_continuation``), so that park closes it without
retiring the context. Nothing is restored after a restart, a cold Resume or that
park's Resume: the next call starts a new process. A
Settings change that alters the server, or disables it or MCP, closes it
(``retire_changed``). A call that resolved the server before such a change starts
nothing: the manager re-reads its configuration and registers the new session under
the lock its reconfiguration retires sessions under, before the worker can launch.
When a later call of the same loop starts a new process in place of an ended one
(the server left, a call timed out, a Settings change, that park), the first call
the new one answers carries one host note naming the restart and the ended
session's ``close_reason``. This connection transfers no in-memory session state;
a server may restore its own persisted state.

One worker thread runs one event loop in a clean context: it never carries, holds
or settles the invocation that started it. One coroutine enters and leaves the SDK
transport and client-session contexts, lets the session answer the server
(``ping``) while idle and notices a server that left. It waits for ``initialize``
only until the session is asked to close (a caller's deadline, a Settings change,
the loop's end): a server that never answers is then shut down, not waited for.

Per call the caller keeps the existing gates: Pause admission
(``owner_pause.operation_start``) around the submission only, the external MCP duty
``mcp_client`` retains, and a per-call hold (``tool_custody.hold_invocation``) so
the invocation lives until the call's coroutine has ended. Calls are serialized;
closing, the server's requests and a ``browser_policy`` callback
(``mcp_browser_guard``) never wait for that lock. A server-answered ``isError`` is
an ordinary result. A host timeout after the request may have been sent closes the
session: the outcome stays unknown and nothing is resent. When a call's own deadline
closes the session before its request left, the TOOL is reported unsent and the
TRANSPORT's end only as observed; like a per-call runner through its transport's exit,
the invocation stays held until the worker ends, and only a confirmed end discharges
its MCP duty. After a close whose end is not confirmed, this context starts no second
process for that server: while that shutdown still runs a later call is refused and
may start one once the end is confirmed; a worker that ended without confirming it
leaves the server unavailable to this context for the rest of its loop, since nothing
can confirm that end any more. The raw ``CallToolResult`` goes back to ``mcp_client``,
which projects it.

The process carries the installation marker and this task's process-scope marker
(``process_scope.scoped_child_env``), so task end and Panic reach it and its
descendants. "Confirmed" means only that the SDK's own stdio shutdown returned
(stdin closed; the child waited for, or its group terminated and the child reaped)
and the worker ended; the SDK returns no pid receipt and none is invented here. The
client session reads and writes through its own clones of the transport's streams,
so its exit closes only those: a server message that arrives after it waits in the
SDK's reader until that shutdown has run and closed the transport's own streams. The
closed-stream error the reader then meets, which its task group reports only after
that shutdown, does not unconfirm it; any other transport failure does.

The one-worker, one-coroutine holder shape is adapted from razzant/ouroboros#1401
``mcp_task_sessions.py`` (Andrei Kaznacheev with Ouroboros, head 4384d057), without its
task/attempt table, browser tool knowledge or handoff state.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import dataclasses
import logging
import threading
import time
import weakref
from typing import Any, Callable, Dict, Optional

from ouroboros.config import (
    MCP_TASK_SESSION_CLOSE_WAIT_SEC,
    MCP_TASK_SESSION_IDLE_POLL_SEC,
    MCP_TASK_SESSION_RETURN_WAIT_SEC,
)

log = logging.getLogger(__name__)

_SLOT = "_mcp_task_sessions"  # the owning ToolContext's attribute
_LIVE: "weakref.WeakSet[TaskSession]" = weakref.WeakSet()  # only for retire_changed; never a lookup authority
_LIVE_LOCK = threading.Lock()  # _LIVE and a context's slot lookup/create; nothing slow or blocking runs under it
_SHUTTING_DOWN = "unconfirmed: the server's shutdown is still running"  # wait_closed while the worker runs, only


class TaskSessionUnavailable(RuntimeError):
    """This call was not submitted: the context has no session that can take it."""

    mcp_unsent_joined = True

    def __init__(self, reason: str) -> None:
        super().__init__(f"MCP_TASK_SESSION_UNAVAILABLE: {reason}")


def _unsent(exc: BaseException) -> BaseException:
    exc.mcp_unsent_joined = True  # nothing of this call was handed to the session
    return exc


def _closed_streams_only(exc: BaseException) -> bool:
    """A task group's report that the SDK's reader/writer only met a stream the shutdown had already closed.

    ``stdio_client`` runs its shutdown (stdin closed, the child waited for or terminated) in the body's
    ``finally`` and closes its own streams only after it, before its task group ends; the session holds clones
    (``_serve``), so only that close can fail the reader. A group of these errors (a server message that arrived
    after the session ended) therefore says nothing against the shutdown. Anything else, or a bare error, is not
    read as a clean exit."""
    import anyio

    members = getattr(exc, "exceptions", None)
    return bool(members) and all(
        isinstance(member, (anyio.BrokenResourceError, anyio.ClosedResourceError)) or _closed_streams_only(member)
        for member in members)


def session_identity(cfg: Any) -> str:
    """What a live session was started from: the server's identity plus every call-lifetime choice."""
    from ouroboros import mcp_catalog

    return f"{mcp_catalog.identity(cfg)}/{cfg.enabled}/{cfg.task_session}/{cfg.browser_policy}"


def _not_live_owner(ctx: Any) -> str:
    """Why ``ctx`` may not own a session; empty for the task loop's own live context."""
    from ouroboros.model_wait import current_model_wait

    owner = current_model_wait()
    if ctx is None or owner is None or owner.closed or getattr(owner, "tool_context", None) is not ctx:
        return "this call does not come from a task's own live tool loop (a review, voice view or other derived context)"
    if str(getattr(ctx, "task_id", "") or "") != owner.task_id:
        return "the tool context and its model-wait owner name different tasks"
    attempt = getattr(ctx, "task_attempt", None)
    try:
        consistent = attempt in (None, "") or int(attempt) == owner.attempt  # agreement only, never a default
    except (TypeError, ValueError):
        consistent = False
    return "" if consistent else "the tool context and its model-wait owner name different attempts"


class _Owned:
    """The sessions one exact ToolContext owns, stored on that context."""

    def __init__(self, ctx: Any) -> None:
        self.owner = ctx  # identity, not a copy's view: a plain reference works for every context type
        self.lock = threading.Lock()
        self.sessions: Dict[str, "TaskSession"] = {}
        self.retired = False  # its loop ended: no new session in this context


def _owned(ctx: Any) -> _Owned:
    with _LIVE_LOCK:  # find-or-create-and-attach is one step: two first callers must share ONE slot (and its lock)
        owned = vars(ctx).get(_SLOT)
        if owned is None or owned.owner is not ctx:  # a copied context carries its original's reference
            owned = _Owned(ctx)
            setattr(ctx, _SLOT, owned)
        return owned


class TaskSession:
    """One server process and its initialized client session, owned by one ToolContext."""

    def __init__(self, cfg: Any) -> None:
        self.cfg, self.server_id, self.identity = cfg, cfg.id, session_identity(cfg)
        self.guard: Any = None  # its BrowserGuard (bound to the owning context) when browser_policy is on
        self.closure = ""  # once the worker ended: "closed" or "unconfirmed: ..."
        self.close_reason = ""  # why it was asked to end; once the worker ended, also how it ended by itself
        self.restarted = ""  # the host note its first answered call carries when it replaced an ended session
        self._host_env: Dict[str, str] = {}
        self._ready: concurrent.futures.Future = concurrent.futures.Future()
        self._stop, self._finished = threading.Event(), threading.Event()
        self._call_lock, self._submit_lock = threading.Lock(), threading.Lock()
        self._accepting = self._entered = self._transport_closed = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_context: Optional[contextvars.Context] = None
        self._wake: Optional[asyncio.Event] = None
        self._session: Any = None
        self._calls: set = set()
        self._end_lock, self._at_end = threading.Lock(), []  # reports owed when the worker has ended
        self._thread = threading.Thread(target=contextvars.Context().run, args=(self._run,),
                                        name=f"mcp-task-session-{cfg.id}", daemon=True)

    # -- lifetime ---------------------------------------------------------------

    def usable(self) -> bool:
        return not self._stop.is_set() and not self._finished.is_set()

    def start(self) -> None:
        """Start the worker; the caller holds Pause admission around this call only."""
        self._thread.start()

    def never_started(self) -> None:
        self._stop.set()
        if self.guard is not None:
            self.guard.close()
        self.closure = "closed"
        self._finished.set()

    def close(self, reason: str) -> None:
        """Ask the session to end (idempotent; never waits, never takes the call lock)."""
        if self._stop.is_set():
            return
        self.close_reason = reason
        self._stop.set()
        if self.guard is not None:
            self.guard.closing()
        with self._submit_lock:  # the worker closes its loop under this lock
            if self._loop is not None and self._wake is not None and not self._loop.is_closed():
                self._loop.call_soon_threadsafe(self._wake.set)

    def wait_closed(self, timeout: float) -> str:
        if self._finished.wait(max(0.0, timeout)):
            return self.closure
        return _SHUTTING_DOWN

    def _after_end(self, report: Callable[[], None]) -> None:
        """Run ``report`` once the worker has ended: now if it has, else on the worker as it ends."""
        with self._end_lock:
            if not self._finished.is_set():
                self._at_end.append(report)
                return
        report()

    def _run(self) -> None:
        loop = self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)  # this worker's own loop; nothing else runs on this thread
        failure: Optional[BaseException] = None
        try:
            try:
                loop.run_until_complete(self._serve())
            except BaseException as exc:  # noqa: BLE001 - the worker reports; nothing above it would see a raise
                failure = exc
                if self._ready.done() and not self._stop.is_set():
                    log.warning("MCP task session %s ended: %s", self.server_id, type(exc).__name__)
            with self._submit_lock:
                self._accepting = False
            # Queued submissions are refused and leftover tasks end on the loop that owns them.
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
        except BaseException:  # noqa: BLE001 - the end record below is written whatever happened
            log.warning("MCP task session %s worker cleanup failed", self.server_id, exc_info=True)
        finally:
            with self._submit_lock:
                loop.close()
            asyncio.set_event_loop(None)
            if not self._ready.done():
                self._ready.set_exception(failure or TaskSessionUnavailable("the session closed before it was ready"))
            if self.guard is not None:
                self.guard.close()  # after the process is gone: it could still ask until then
            if not self.close_reason:  # nobody asked it to end: say how it ended, for a replacement's note
                self.close_reason = (f"its session failed: {type(failure).__name__}" if failure
                                     else "its transport ended")
            self._stop.set()
            self.closure = ("closed" if self._transport_closed or not self._entered
                            else f"unconfirmed: {type(failure).__name__ if failure else 'transport exit'}")
            self._finished.set()
            with self._end_lock:
                owed, self._at_end = self._at_end, []
            for report in owed:
                try:
                    report()
                except Exception:  # noqa: BLE001 - one report never strands the next
                    log.warning("MCP task session %s end report failed", self.server_id, exc_info=True)

    async def _serve(self) -> None:
        from ouroboros.mcp_client import ClientSession, _transport_factory

        if self._stop.is_set():  # closed (a Settings change, the loop's end) before its server was launched
            raise TaskSessionUnavailable("the session closed before its server was started")
        launch = self.cfg if self.guard is None else dataclasses.replace(
            self.cfg, secret_values=(*self.cfg.secret_values, self.guard.token))
        self._wake = asyncio.Event()
        failure: Optional[BaseException] = None
        try:
            async with _transport_factory(launch, self._host_env) as streams:
                self._entered = True
                try:
                    # Clones: the session's exit closes only these. The transport's own streams stay open through
                    # the SDK's shutdown (stdin closed, the wait, its process-group escalation), which closes them
                    # last; a server message in that window waits instead of failing the reader and cutting it short.
                    async with ClientSession(streams[0].clone(), streams[1].clone()) as session:
                        initialized = await self._initialize(session)
                        self._session, self._loop_context = session, contextvars.copy_context()
                        with self._submit_lock:
                            self._accepting = initialized and not self._stop.is_set()
                        if self._accepting:
                            self._ready.set_result(True)
                            await self._idle(streams[0])
                        with self._submit_lock:
                            self._accepting = False
                        calls = list(self._calls)
                        for call in calls:
                            call.cancel()
                        await asyncio.gather(*calls, return_exceptions=True)
                except BaseException as exc:  # noqa: BLE001 - raised again once the transport's own exit has run
                    failure = exc
        except BaseException as exc:  # noqa: BLE001 - the transport's own exit; a failed start is never tolerated
            if not (self._entered and _closed_streams_only(exc)):
                raise
        self._transport_closed = True  # the SDK's stdio shutdown sequence returned
        if failure is not None:
            raise failure

    async def _initialize(self, session: Any) -> bool:
        """``initialize``, unless the session is asked to close first (a server that never answers it).

        True once answered; False when closing came first: the pending request is cancelled, so the
        transport's own shutdown ends the process instead of this worker waiting for a reply forever."""
        work = asyncio.ensure_future(session.initialize())
        try:
            while not self._stop.is_set():
                done, _ = await asyncio.wait({work}, timeout=MCP_TASK_SESSION_IDLE_POLL_SEC)
                if done:
                    work.result()
                    return True
            return False
        finally:
            if not work.done():
                work.cancel()
                await asyncio.gather(work, return_exceptions=True)

    async def _idle(self, read_stream: Any) -> None:
        """Until asked to stop or the server leaves; the SDK session answers the server meanwhile."""
        while not self._stop.is_set():
            if read_stream.statistics().open_send_streams == 0:
                self.close("the server closed its connection")
                return
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=MCP_TASK_SESSION_IDLE_POLL_SEC)
            except asyncio.TimeoutError:
                pass

    # -- calls ------------------------------------------------------------------

    def _submit(self, name: str, arguments: Dict[str, Any], sent: threading.Event) -> concurrent.futures.Future:
        future: concurrent.futures.Future = concurrent.futures.Future()

        async def invoke() -> Any:
            sent.set()  # from here the request may reach the server
            return await self._session.call_tool(name, arguments)

        def begin() -> None:  # on the session's loop, in the session's own context
            if not self._accepting:
                future.set_exception(TaskSessionUnavailable("the session closed before this call was sent"))
                return
            task = self._loop.create_task(invoke())
            self._calls.add(task)

            def settle(done: asyncio.Task) -> None:
                self._calls.discard(done)
                if done.cancelled():
                    future.cancel()
                elif done.exception() is not None:
                    future.set_exception(done.exception())
                else:
                    future.set_result(done.result())
            task.add_done_callback(settle)

        with self._submit_lock:
            if not self._accepting:
                raise TaskSessionUnavailable("the session is closing")
            self._loop.call_soon_threadsafe(begin, context=self._loop_context)
        return future

    def _exit_at_end(self, exited: Callable[[Any, Optional[BaseException]], None], caller: contextvars.Context,
                     release: Callable[[], None], ended: Optional[threading.Event] = None) -> None:
        """This call's own deadline closed the session before its request left: the call ends with the worker.

        ``release`` (this invocation's hold) waits for the worker's end, as a per-call runner holds its
        invocation through its transport's exit; only a confirmed end gives ``exited`` the joined no-send
        proof that discharges the call's MCP duty, an unconfirmed one leaves that duty open."""
        def report() -> None:
            try:
                caller.run(exited, None, _unsent(TaskSessionUnavailable("never sent; the session's end is confirmed"))
                           if self.closure == "closed" else ConnectionError(f"never sent; the session's end is "
                                                                             f"{self.closure}"))
            finally:
                release()
                if ended is not None:
                    ended.set()
        self._after_end(report)

    def _closing_unsent(self, message: str) -> TimeoutError:
        """The raise of a call whose own deadline closed the session before its request left.

        The TOOL was not sent (``mcp_unsent_closing``); the TRANSPORT is claimed ended (``mcp_unsent_joined``)
        only when this moment observes the session's confirmed end."""
        closure = self.wait_closed(0)
        error = TimeoutError(f"{message}; the session's end: {closure}")
        if closure == "closed":
            return _unsent(error)
        error.mcp_unsent_closing = True
        return error

    def call(self, name: str, arguments: Dict[str, Any], *, deadline: float,
             exited: Callable[[Any, Optional[BaseException]], None]) -> Any:
        """One serialized call: the raw ``CallToolResult`` and the session's restart note, or a raise.

        The note is ``restarted`` for the first call this session answers, then ``""``; ``exited`` runs exactly once."""
        from ouroboros.owner_pause import operation_start
        from ouroboros.tool_custody import hold_invocation

        if not self._call_lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
            exc = TaskSessionUnavailable("another call to this server is still running in this task")
            exited(None, exc)
            raise exc
        try:
            sent, ended, closer = threading.Event(), threading.Event(), threading.Event()
            caller = contextvars.copy_context()
            release: Optional[Callable[[], None]] = None
            try:
                self._ready.result(timeout=max(0.0, deadline - time.monotonic()))
                release = hold_invocation()  # this invocation lives until the call's coroutine has ended
                with operation_start():  # Pause admission for the submission, never for the wait
                    future = self._submit(name, arguments, sent)
            except BaseException as exc:
                if release is not None:
                    release()
                if isinstance(exc, concurrent.futures.TimeoutError) and not self._ready.done():
                    self.close("the server did not finish starting within the call's time")
                    self._exit_at_end(exited, caller, hold_invocation())
                    raise self._closing_unsent(
                        f"MCP server {self.server_id!r} did not finish starting in time") from None
                exited(None, _unsent(exc))  # never handed to the session
                raise exc

            def finished(done: concurrent.futures.Future) -> None:  # on the session's loop thread
                if closer.is_set() and not sent.is_set():  # this call's deadline closed the session before it left
                    self._exit_at_end(exited, caller, release, ended)
                    return
                try:
                    if done.cancelled():
                        error: Optional[BaseException] = (
                            ConnectionError("the session closed during this call; its outcome is unknown")
                            if sent.is_set() else _unsent(TaskSessionUnavailable("the session closed first")))
                        caller.run(exited, None, error)
                    else:
                        caller.run(exited, None if done.exception() else done.result(), done.exception())
                finally:
                    release()
                    ended.set()
            future.add_done_callback(finished)
            try:
                value = future.result(timeout=max(0.0, deadline - time.monotonic()))
            except concurrent.futures.TimeoutError:
                closer.set()
                self.close("a call did not answer in time")  # the process ends on the worker; nothing is resent
                ended.wait(MCP_TASK_SESSION_CLOSE_WAIT_SEC)  # the session cancels its calls before it leaves the transport
                if future.done() and not sent.is_set():
                    raise self._closing_unsent(f"MCP server {self.server_id!r}: the call was never sent") from None
                raise TimeoutError(f"MCP server {self.server_id!r} did not answer; its session is closing and "
                                   "the call's outcome is unknown") from None
            except concurrent.futures.CancelledError:
                ended.wait(MCP_TASK_SESSION_RETURN_WAIT_SEC)
                if not sent.is_set():
                    raise _unsent(TaskSessionUnavailable("the session closed before this call was sent")) from None
                raise ConnectionError("the session closed during this call; its outcome is unknown") from None
            ended.wait(MCP_TASK_SESSION_RETURN_WAIT_SEC)
            note, self.restarted = self.restarted, ""  # under the call lock: named by one answered call only
            return value, note
        finally:
            self._call_lock.release()


def _register(session: TaskSession) -> None:
    with _LIVE_LOCK:
        _LIVE.add(session)


def _session_for(ctx: Any, cfg: Any, deadline: float, live: Callable[..., bool]) -> TaskSession:
    """This context's usable session for ``cfg``, starting one under Pause admission if needed.

    ``live(cfg, register=None)`` is the manager's own answer whether ``cfg`` is still its configuration,
    ``register`` run under the lock its reconfiguration retires sessions under: a new session is registered
    before its worker can launch the server, so a change either refuses it here or retires it there."""
    from ouroboros.mcp_browser_guard import BrowserGuard
    from ouroboros.owner_pause import operation_start
    from ouroboros.process_scope import scoped_child_env, task_launch_target
    from ouroboros.tool_custody import invocation_binding

    stale = "its server configuration changed after this call resolved it; nothing was started or sent"
    if reason := _not_live_owner(ctx):
        raise TaskSessionUnavailable(reason + "; nothing was started or sent")
    owned, identity = _owned(ctx), session_identity(cfg)
    with owned.lock:  # one starter per context; close() and the server's requests never take it
        if owned.retired:
            raise TaskSessionUnavailable("this task loop has ended; no new server process is started")
        previous = owned.sessions.get(cfg.id)
        if previous is not None and previous.identity == identity and previous.usable():
            return previous
        if not live(cfg):  # a stale call neither replaces the current session nor starts its own
            raise TaskSessionUnavailable(stale)
        if previous is not None and previous.identity != identity:
            previous.close("its server configuration changed")
        if previous is not None:
            closure = previous.wait_closed(min(MCP_TASK_SESSION_CLOSE_WAIT_SEC, deadline - time.monotonic()))
            if closure == _SHUTTING_DOWN:
                raise TaskSessionUnavailable("the previous session of this server has not confirmed its end yet (its "
                                             "shutdown is still running); no second server process is started "
                                             "before it does, so a later call may succeed")
            if closure != "closed":  # its worker ended: nothing can confirm that end any more
                raise TaskSessionUnavailable(f"the previous session of this server ended without confirming its "
                                             f"server's exit ({closure}); no second server process is started, so "
                                             "this server stays unavailable for the rest of this task loop")
        session = TaskSession(cfg)
        if previous is not None:  # a new process in place of an ended one: its first answered call says so
            session.restarted = (f"MCP server {cfg.id!r} was started anew for this call: its previous process in this "
                                 f"task loop ended ({previous.close_reason}), and this connection transferred no "
                                 "in-memory session state. The server may restore its own persisted state.")
        try:
            session.guard = BrowserGuard(ctx) if cfg.browser_policy else None
            env = session.guard.environment() if session.guard is not None else {}
            session._host_env = scoped_child_env(env, task_launch_target(invocation_binding()))
            if not live(cfg, lambda: _register(session)):  # seen by every later retirement, or refused now
                raise TaskSessionUnavailable(stale)
            with operation_start():  # starting the worker starts the server process: Pause excludes it
                session.start()
        except BaseException:
            session.never_started()
            raise
        owned.sessions[cfg.id] = session
        return session


def call_tool(cfg: Any, raw_name: str, arguments: Dict[str, Any], timeout_sec: float,
              exited: Callable[[Any, Optional[BaseException]], None], ctx: Any = None, *,
              live: Callable[..., bool]) -> Any:
    """``cfg``'s call through the calling task loop's own session: the raw ``CallToolResult`` and a host note,
    ``""`` unless this is the first call a session that replaced an ended one answers (``TaskSession.call``).

    ``ctx`` is the caller's context, else the active invocation's own (its registry's); ``live`` is
    the resolving manager's configuration check (``_session_for``). Raises like the per-call transport:
    ``TimeoutError`` (``mcp_unsent_joined`` only when the request was never handed over and nothing it
    reached is still ending; ``mcp_unsent_closing`` while the session its deadline closed still is), a
    refusal before any send (``TaskSessionUnavailable``, Pause or store refusals) or the SDK's error
    after a possible send. ``exited`` runs once, when this call's local work is over."""
    from ouroboros.owner_pause import _TOOL_OPERATION

    active = _TOOL_OPERATION.get()
    deadline = time.monotonic() + max(0.0, float(timeout_sec))
    try:
        session = _session_for(ctx if ctx is not None else active[0] if active else None, cfg, deadline, live)
    except BaseException as exc:
        exited(None, _unsent(exc))
        raise
    return session.call(raw_name, arguments, deadline=deadline, exited=exited)


def close_context_sessions(ctx: Any, *, reason: str = "its task loop ended", retire: bool = True) -> None:
    """Close what this exact context owns and accept no new session in it (idempotent; never raises).

    ``retire=False`` (the moved Presence author's Pause park) leaves the live loop able to start a new one,
    once Pause admission lets it, and adds nothing to a context that never started one. Waits a bounded
    time; a slower end continues on its worker and reads as unconfirmed."""
    try:
        if ctx is None:
            return
        owned = _owned(ctx) if retire else vars(ctx).get(_SLOT)
        if owned is None or owned.owner is not ctx:
            return
        with owned.lock:
            owned.retired = owned.retired or retire
            sessions = list(owned.sessions.values())
        for session in sessions:
            session.close(reason)
        deadline = time.monotonic() + MCP_TASK_SESSION_CLOSE_WAIT_SEC
        for session in sessions:
            closure = session.wait_closed(deadline - time.monotonic())
            if closure != "closed":
                log.warning("MCP task session %s did not confirm its end: %s", session.server_id, closure)
    except Exception:
        log.warning("MCP task sessions could not be closed", exc_info=True)


def retire_changed(configs: Any) -> None:
    """Close every live session whose server is gone, disabled or changed; never waits."""
    wanted = {cfg.id: session_identity(cfg) for cfg in configs if cfg.enabled and cfg.task_session}
    with _LIVE_LOCK:
        live = list(_LIVE)
    for session in live:
        if session.usable() and wanted.get(session.server_id) != session.identity:
            session.close("its MCP configuration changed")
