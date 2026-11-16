"""One browser generation's Playwright on its own event loop, and the request guard it keeps answering.

A builtin tab lives between tool calls: its timers fire and its requests reach the network
while the model thinks. Synchronous Playwright dispatched driver events only inside a call, so
a request the page made between calls waited unanswered at the route guard until the next call
(a module loader's 25 s bound expired while nothing was wrong, #3389). Each non-empty generation
(``BrowserState``) therefore runs ``playwright.async_api`` on one daemon thread with its own
asyncio loop (``BrowserOwner``), started on first use; the tools keep a synchronous facade.

- Every Playwright object, call and property belongs to the loop. Tool results, messages,
  screenshots and browser preparation stay on the calling thread.
- The session is opened in an empty ``contextvars.Context`` (``run(..., clean=True)``): route and
  event callbacks never inherit a call's browser pin, deadline or operation. A command runs in a
  copy of its caller's context (``run_coroutine_threadsafe``); its changes never flow back.
- The bridge has no timeout of its own: the call's outer tool timeout bounds the wait. Submission
  and stop are atomic under one lock that is never held while waiting; a closing, closed or dead
  loop refuses at once; closing settles every pending command; the loop's own thread may not wait
  on itself; only the first close runs its closing work, on the loop (``browser.py`` reads and
  clears the generation's handles there, so one caller closes them); a second reports its outcome.
- ``request_guard`` answers every routed request, between calls too, with ``browser_policy`` as
  the only policy, decided off the loop on a small worker pool (DNS and service identity block).
  A request that reaches Python while a command of this generation runs belongs to that
  already-handed operation, as before. Any other request is a new effect of the task: one strict
  read of its tree's Pause fence and of an immediate Stop over it (its own, or an ancestor's
  cascade) admits or refuses it before the policy; no lock, claim, queue or cached grant. Those
  reads wait on their own workers, so a busy store never delays a command's decisions. Every
  answer is continue or abort; a refusal is kept on its own generation for the next tool result
  and consumed only when shown there.

Interception keeps its limits: native redirect hops, service workers and WebSockets bypass the
route (``browser.py`` withholds a redirect chain's content afterwards; a blocked hop has already
left), cookies and bodies pass unchanged, and there is no DNS-rebinding or pre-request guarantee,
proxy, CDP fork, pre-probe or refetch.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import functools
import logging
import threading
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Optional

from ouroboros import browser_policy

log = logging.getLogger(__name__)

_DECISION_WORKERS = 8  # per pool; a page-load burst waits for these, the synchronous session decided one at a time
_LOOP_START_SEC = 10.0
_CLOSE_STEP_SEC = 5.0  # one handle's close: page, context, browser, then the driver
_CANCEL_WAIT_SEC = 2.0  # what is still pending on the loop after the handles closed
_CLOSE_WAIT_SEC = 4 * _CLOSE_STEP_SEC + _CANCEL_WAIT_SEC + 3.0

CLOSED = "BROWSER_SESSION_CLOSED: this call's browser session was closed or had stopped; nothing more ran in it."
_IDLE = "a page request made between browser calls was not sent"
_RETIRED = "BROWSER_SESSION_RETIRED: this browser session was retired; a page request made after its last call was not sent."


class BrowserSessionClosed(RuntimeError):
    """The generation's loop is closing, closed or gone: nothing (more) of the call ran there."""


class BrowserOwner:
    """The event loop that owns one generation's Playwright, behind a synchronous bridge."""

    def __init__(self) -> None:
        self._lock = threading.Lock()  # submission and stop, never held while waiting
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._decisions: dict = {}  # between_calls -> its worker pool
        self._pending: set = set()
        self._closing = False
        self._closed = threading.Event()
        self._confirmed = False
        self.commands = 0  # command coroutines running now; read and written only on the loop

    @property
    def started(self) -> bool:
        return self._loop is not None

    def run(self, coro: Any, *, clean: bool = False) -> Any:
        """Run ``coro`` on this generation's loop and wait for its outcome; ``clean`` submits it
        from an empty context (the session's own callbacks start there)."""
        self._refuse_own_thread(coro)
        with self._lock:
            if self._closing or (self._thread is not None and not self._thread.is_alive()):
                coro.close()
                raise BrowserSessionClosed(CLOSED)
            if self._loop is None:
                try:
                    self._start()
                except BaseException:
                    coro.close()
                    raise
            submit = functools.partial(asyncio.run_coroutine_threadsafe, self._command(coro), self._loop)
            future = contextvars.Context().run(submit) if clean else submit()
            self._pending.add(future)
        try:
            return future.result()
        except concurrent.futures.CancelledError:
            raise BrowserSessionClosed(CLOSED) from None
        finally:
            with self._lock:
                self._pending.discard(future)

    async def _command(self, coro: Any) -> Any:
        self.commands += 1
        try:
            return await coro
        finally:
            self.commands -= 1

    async def decide(self, function: Callable[..., Any], *args: Any, between_calls: bool = False) -> Any:
        """Run a blocking decision off the loop, in an empty context of its own; a between-call
        request's (which may wait on the store) on workers of its own."""
        return await asyncio.get_running_loop().run_in_executor(
            self._decisions[between_calls], functools.partial(contextvars.Context().run, function, *args))

    def _start(self) -> None:
        loop, ready = asyncio.new_event_loop(), threading.Event()

        def serve() -> None:
            asyncio.set_event_loop(loop)
            loop.call_soon(ready.set)
            loop.run_forever()

        thread = threading.Thread(target=contextvars.Context().run, args=(serve,), name="browser-session",
                                  daemon=True)
        thread.start()
        if not ready.wait(_LOOP_START_SEC):
            loop.call_soon_threadsafe(loop.stop)
            raise BrowserSessionClosed("BROWSER_SESSION_CLOSED: the browser session's event loop did not start.")
        self._loop, self._thread = loop, thread
        self._decisions = {between: concurrent.futures.ThreadPoolExecutor(
            max_workers=_DECISION_WORKERS, thread_name_prefix=f"browser-{name}") for between, name in
            ((False, "decision"), (True, "idle-decision"))}  # threads start with their first decision

    def close(self, closing: Optional[Callable[[], Awaitable[bool]]] = None) -> bool:
        """Close once, from any thread but the loop's own: run the first caller's ``closing`` there
        (it says whether the handles closed), end what is still pending, stop and join the loop.
        True only when all of it is confirmed; a concurrent second call runs nothing of its own and
        reports the first one's outcome."""
        self._refuse_own_thread()
        with self._lock:
            first, self._closing = not self._closing, True
            loop, thread = self._loop, self._thread
        if not first:
            return self._closed.wait(_CLOSE_WAIT_SEC + 5.0) and self._confirmed
        confirmed = loop is None  # an empty generation never started a thread
        try:
            if loop is not None:
                if thread.is_alive():
                    shutdown = contextvars.Context().run(asyncio.run_coroutine_threadsafe,
                                                         self._shutdown(closing), loop)
                    confirmed = bool(shutdown.result(timeout=_CLOSE_WAIT_SEC))
                    loop.call_soon_threadsafe(loop.stop)
                thread.join(_CLOSE_STEP_SEC)
                confirmed = confirmed and not thread.is_alive()
        except Exception:
            log.warning("Browser session close is unconfirmed", exc_info=True)
            confirmed = False
        finally:
            if loop is not None:
                self._settle_pending()
                for pool in self._decisions.values():
                    pool.shutdown(wait=False, cancel_futures=True)
                if not thread.is_alive():
                    loop.close()
            self._confirmed = confirmed
            self._closed.set()
        return confirmed

    async def _shutdown(self, closing: Optional[Callable[[], Awaitable[bool]]]) -> bool:
        confirmed = True
        if closing is not None:
            try:
                confirmed = bool(await closing())
            except Exception:
                log.warning("Browser session handles did not close", exc_info=True)
                confirmed = False
        current = asyncio.current_task()
        tasks = [task for task in asyncio.all_tasks() if task is not current]
        for task in tasks:  # hung commands and requests still waiting for their decision
            task.cancel()
        if tasks:
            _done, pending = await asyncio.wait(tasks, timeout=_CANCEL_WAIT_SEC)
            confirmed = confirmed and not pending
        return confirmed

    def _settle_pending(self) -> None:
        """Cancel what still waits (its caller hears the session closed); an outcome the loop already
        gave a future, a real failure or its own cancellation, stays: nothing moves a future twice."""
        with self._lock:
            pending = list(self._pending)
        for future in pending:
            future.cancel()

    def _refuse_own_thread(self, coro: Any = None) -> None:
        # The loop's own thread by identity: its number goes to a later thread once it has ended.
        if self._thread is not None and self._thread is threading.current_thread():
            if coro is not None:
                coro.close()
            raise RuntimeError("a browser session's own event loop cannot wait for itself")


_OWNERS_LOCK = threading.Lock()


def owner_of(generation: Any) -> BrowserOwner:
    """The generation's owner; created without a thread until its first command."""
    with _OWNERS_LOCK:
        owner = getattr(generation, "_owner", None)
        if owner is None:
            owner = BrowserOwner()
            setattr(generation, "_owner", owner)
        return owner


def _target_closed(exc: BaseException) -> bool:
    """Playwright's answer for a page, context, browser or driver that is already gone."""
    return type(exc).__name__ == "TargetClosedError" or "has been closed" in str(exc).lower()


class PlaywrightDriver:
    """Own Playwright's connection runner as well as its transport, before the first await.

    The public context manager exposes its driver only after the handshake. Retain its provider
    objects first: the Connection runner, or the PlaywrightClient introduced in 1.65 (unlocked
    installs resolve Playwright independently of uv.lock). Both need a connected pipe to stop;
    their stopped futures can resolve on reader cancellation, before the process exits. A close
    cancels an unfinished spawn (asyncio reaps it) or closes its pipe, then waits for the process.
    No process discovery or global patching: these objects belong to this generation.
    """

    def __init__(self) -> None:
        from playwright.async_api import _context_manager

        client = getattr(_context_manager, "PlaywrightClient", None)
        self._client = client() if client is not None else None
        if self._client is not None:
            self._transport = self._client._transport
            self._connection = self._client._connection
            self._runner = self._client._playwright
            return
        from playwright._impl._connection import Connection
        from playwright._impl._object_factory import create_remote_object
        from playwright._impl._transport import PipeTransport

        loop = asyncio.get_running_loop()
        self._transport = PipeTransport(loop)
        self._connection = Connection(None, create_remote_object, self._transport, loop)
        self._runner = None

    async def start(self) -> Any:
        from playwright.async_api._generated import Playwright

        if self._client is not None:
            playwright = Playwright(await self._runner)
            playwright.stop = self.stop
            return playwright
        self._runner = asyncio.create_task(self._connection.run())
        future = self._connection.playwright_future
        done, _ = await asyncio.wait({self._transport.on_error_future, future},
                                     return_when=asyncio.FIRST_COMPLETED)
        if not future.done():
            future.cancel()
        playwright = Playwright(next(iter(done)).result())
        playwright.stop = self.stop
        return playwright

    async def stop(self) -> None:
        if self._runner is None:  # a close before start ran: nothing was spawned
            return
        if getattr(self._transport, "_output", None) is None:
            # connect is still in asyncio.create_subprocess_exec, or failed: cancelling the
            # runner settles that spawn before we inspect its result, not the handshake task.
            self._runner.cancel()
        else:
            self._transport.request_stop()
            if self._client is not None:
                await self._client.stop()  # its reader finishes separately from the handshake
                self._runner.cancel()  # an unfinished handshake need not finish to close its pipe
        await asyncio.gather(self._runner, return_exceptions=True)
        process = getattr(self._transport, "_proc", None)
        if process is not None:
            await process.wait()  # a cancelled reader's stopped future is not proof of exit
        if self._client is not None:
            self._connection.close()
        else:
            self._connection.cleanup()
            self._connection.playwright_future.cancel()
        if self._transport.on_error_future.done() and not self._transport.on_error_future.cancelled():
            self._transport.on_error_future.exception()


async def close_handles(page: Any, browser_context: Any, browser: Any, pw_instance: Any,
                        driver: Any = None) -> bool:
    """Close page, context and browser, then stop the driver: in order and each bounded; every step runs
    whatever an earlier one did, and one already gone counts as closed. The driver's stop confirms the
    close: Playwright's stop returns once the driver process has exited, and that exit closes or kills
    every browser it launched, so a slower or failed close before it leaves nothing open. A driver stop
    that hangs or fails leaves the close unconfirmed; without a driver every handle present must close.
    The transport owner can stop a driver whose handshake has not returned or failed."""
    driver = driver if driver is not None else pw_instance
    closed = []
    for name, handle, method in (("page", page, "close"), ("context", browser_context, "close"),
                                 ("browser", browser, "close"), ("Playwright driver", driver, "stop")):
        if handle is None:
            continue
        try:
            await asyncio.wait_for(getattr(handle, method)(), _CLOSE_STEP_SEC)
            closed.append(True)
        except asyncio.TimeoutError:
            log.warning("Browser %s did not close within %s s", name, _CLOSE_STEP_SEC)
            closed.append(False)
        except Exception as exc:
            if _target_closed(exc):
                log.debug("Browser %s was already closed", name, exc_info=True)
            else:
                log.warning("Browser %s did not close", name, exc_info=True)
            closed.append(_target_closed(exc))
    return closed[-1] if driver is not None else all(closed)


# --- the request guard ----------------------------------------------------------------------

def request_guard(owner: BrowserOwner, ctx: Any, generation: Any) -> Callable[[Any], Awaitable[None]]:
    """The generation's one route handler: every request the page makes is decided, between calls too."""

    async def route_request(route: Any) -> None:
        affiliated = owner.commands > 0  # the handed command this request belongs to, read before any await
        facts = _request_facts(route.request)
        between = not affiliated
        reason = _RETIRED if between and getattr(generation, "_retired", False) else None
        if reason is None:
            try:
                reason = await owner.decide(_decide, ctx, facts, affiliated, between_calls=between)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.warning("Browser request decision failed", exc_info=True)
                reason = "BROWSER_POLICY_UNAVAILABLE: the request decision failed; it was not sent"
            if not reason and between and getattr(generation, "_retired", False):
                reason = _RETIRED  # retired while it was decided: an old session starts nothing new
        await _answer(route, reason, generation, between_calls=between)

    return route_request


def _request_facts(request: Any) -> Optional[SimpleNamespace]:
    """The decision's inputs, read on the loop: url, method and the body in three states (text; none;
    present but unreadable, which Playwright's strict decode raises for), as ``mcp_browser_guard`` asks."""
    try:
        url, method, resource_type = request.url, request.method, request.resource_type
    except Exception:
        log.warning("Browser request facts could not be read", exc_info=True)
        return None
    try:
        body = request.post_data
    except Exception:
        body, present = None, True
    else:
        present = body is not None
    return SimpleNamespace(url=url, method=method, post_data=body, body_present=present,
                           resource_type=resource_type)


def _decide(ctx: Any, facts: Optional[SimpleNamespace], affiliated: bool) -> str:
    if not affiliated and (refused := _idle_admission(ctx)):
        return refused
    if facts is None:
        return "BROWSER_POLICY_UNAVAILABLE: the request's url, method or body could not be read; it was not sent"
    try:
        from ouroboros.tools.core import is_restricted_subagent_profile

        return browser_policy.browser_request_block_reason(
            facts, ctx, restricted=is_restricted_subagent_profile(ctx),
            runtime_mode=browser_policy.browser_runtime_mode(ctx))
    except Exception:
        log.warning("Browser request policy could not read target identity", exc_info=True)
        return "BROWSER_POLICY_UNAVAILABLE: runtime service identity could not be read"


def _idle_admission(ctx: Any) -> str:
    """The one admission point of a request no call made, read strictly once: the tree's Pause fence
    and an immediate Stop over the task (``cancel_intents.immediate_stop_pending``, the predicate
    ``ModelWait.control_reason`` uses, here with its ancestors' cascades too), which closes the window
    before its worker is killed. Warm waits, model rounds, owner questions, a graceful Stop's wrap-up
    and the task's own sleep refuse nothing here."""
    from ouroboros import owner_pause
    from ouroboros._usage_wait import ACQUISITION_SLICE_SEC
    from ouroboros.cancel_intents import immediate_stop_pending
    from ouroboros.store import acquisition_budget

    root, tree, task_id = owner_pause._member_coordinates(ctx)
    try:
        with acquisition_budget(ACQUISITION_SLICE_SEC):
            fence = owner_pause.member_fence(ctx)
            stopping = bool(root and task_id and immediate_stop_pending(root, task_id, tree_root_id=tree))
    except Exception as exc:  # a busy or unavailable store is unknown authority, never an open fence
        log.warning("Pause or Stop authority for a between-call browser request is unavailable", exc_info=True)
        return (f"BROWSER_POLICY_UNAVAILABLE: this task's Pause or Stop could not be read ({type(exc).__name__}); "
                f"{_IDLE}.")
    if fence.get("state") == "unknown":
        return f"BROWSER_POLICY_UNAVAILABLE: this task's Pause could not be read; {_IDLE}."
    if fence:
        return ("OWNER_PAUSE_NOT_STARTED: a Pause of this task tree stopped a page request made between browser "
                "calls; it was not sent.")
    if stopping:
        return f"OWNER_OPERATION_REFUSED: this task is stopping; {_IDLE}."
    return ""


async def _answer(route: Any, reason: Optional[str], generation: Any, *, between_calls: bool) -> None:
    """Continue or abort, never fall through. A refusal is recorded on its own generation before its
    abort. A continue that fails is never retried: whether that request was sent is unknown, so its
    note says so beside the outcome of the one abort that follows. A closed target is owed nothing."""
    if reason:
        record_block(generation, reason, between_calls=between_calls)
        if (failure := await _attempt(route.abort)) and not _target_closed(failure):
            log.warning("A refused browser request could not be aborted", exc_info=failure)
        return
    if not (failure := await _attempt(route.continue_)) or _target_closed(failure):
        return
    log.warning("A browser request could not be continued", exc_info=failure)
    abort = await _attempt(route.abort)
    record_block(generation, f"BROWSER_REQUEST_OUTCOME_UNKNOWN: continuing a page request failed "
                             f"({type(failure).__name__}); whether it was sent is unknown; the abort that followed "
                             + (f"failed too ({type(abort).__name__})" if abort else "was accepted"),
                 between_calls=between_calls)


async def _attempt(answer: Callable[[], Awaitable[None]]) -> Optional[Exception]:
    """One answer to Playwright: None once accepted, else its failure (a closed target's included)."""
    try:
        await answer()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        log.debug("Browser request answer failed", exc_info=True)
        return exc
    return None


# --- refusal notes: kept per generation until a result shows them ---------------------------

_NOTES_LOCK = threading.Lock()


def record_block(generation: Any, reason: str, *, between_calls: bool = False) -> None:
    """Keep one refusal on ``generation``; repeats of the same refusal are counted, not listed."""
    with _NOTES_LOCK:
        seq = getattr(generation, "_block_seq", 0) + 1
        setattr(generation, "_block_seq", seq)
        rows = getattr(generation, "_block_rows", None)
        if rows is None:
            rows = {}
            setattr(generation, "_block_rows", rows)
        _seq, count = rows.pop((reason, between_calls), (0, 0))
        rows[(reason, between_calls)] = (seq, count + 1)


def block_mark(generation: Any) -> int:
    """The latest refusal so far: a call reclassifies its failure only by refusals after its own mark."""
    with _NOTES_LOCK:
        return getattr(generation, "_block_seq", 0)


def take_block_since(generation: Any, mark: int) -> str:
    """The newest refusal of a request a command made after ``mark``, consumed (the failed call shows it).
    Older ones and those made between calls stay for the next result: they never replace a call's own error."""
    with _NOTES_LOCK:
        rows = getattr(generation, "_block_rows", None) or {}
        latest = max(((seq, key) for key, (seq, _count) in rows.items() if seq > mark and not key[1]),
                     default=None)
        if latest is None:
            return ""
        rows.pop(latest[1])
        return latest[1][0]


def take_block_notes(generation: Any) -> str:
    """Every refusal not shown yet, consumed by the result that shows it; later ones wait for the next."""
    with _NOTES_LOCK:
        rows = getattr(generation, "_block_rows", None)
        if not rows:
            return ""
        taken = dict(rows)
        rows.clear()
    return "\n".join(
        ("Between browser calls: " if between else "") + reason + (f" ({count} requests)" if count > 1 else "")
        for (reason, between), (_seq, count) in taken.items())
