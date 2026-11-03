"""The Host's live Presence turns: one execution per durable turn id, joined by retries, resolved once.

``PresenceTurnExecutions`` starts a turn or joins the one already live. A Host turn's body
runs in a conversation-lane carrier (``presence_carrier.dispatch_presence_turn``): the futures
here resolve from the carrier's frames. ``start_thread`` keeps the proactive form (an
initiating tool's turn runs where the tool runs) and a live call. Every registry of this
process is weakly registered so a carrier's frame finds its waiting execution by turn id.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import logging
import threading
import weakref
from typing import Any, Callable

from ouroboros.presence_events import PresenceTurnError
from ouroboros.conversation_lease import PresenceTurnLease

log = logging.getLogger(__name__)


class PresenceTurnNotStarted(RuntimeError):
    """Nothing ran: admission was interrupted or no thread could start; retry the same event."""


class PresenceTurnInterrupted(RuntimeError):
    """The turn's carrier ended without reporting it (a Stop's kill, a crash, a pool stop) and the server has
    published its terminal; resend the same event: its durable row answers (a replay, or a retained refusal)."""


class PresenceTurnExecution:
    """One live host turn: the results every waiter shares and the admission that starts it.

    ``initial`` is the first envelope (a continuing author's, or the terminal itself) and
    ``result`` the author's terminal (#1536). The reserved capacity returns exactly once,
    at whichever comes first; the turn stays live until its terminal.
    """

    __slots__ = ("turn_id", "identity", "result", "initial", "admission", "_capacity", "_returned", "_lock")

    def __init__(self, turn_id: str, identity: str = "") -> None:
        self.turn_id = turn_id
        self.identity = identity
        self.result: concurrent.futures.Future = concurrent.futures.Future()
        self.initial: concurrent.futures.Future = concurrent.futures.Future()
        # RUNNING from birth: a cancelled waiter's wrapper calls Future.cancel(), which a running
        # future refuses, so no HTTP waiter can cancel the result other waiters share.
        self.result.set_running_or_notify_cancel()
        self.initial.set_running_or_notify_cancel()
        self.admission: asyncio.Task | None = None
        self._capacity: Callable[[], None] | None = None
        self._returned = False
        self._lock = threading.Lock()

    def return_capacity(self, fallback: Callable[[], None] | None = None) -> None:
        with self._lock:
            if self._returned:
                return
            self._returned, release = True, self._capacity or fallback
        if release is not None:
            try:
                release()
            except Exception:
                log.warning("presence turn %s could not return its capacity", self.turn_id, exc_info=True)

    def publish_initial(self, value: Any, *, return_capacity: bool = True) -> bool:
        """The durable initial envelope of a continuing author: its waiters answer, its capacity returns.

        An author that answered early (``return_capacity=False``: it still runs and holds its
        lease) returns the capacity at its first real park, whose envelope is then already out.
        """
        with self._lock:
            published = not self.initial.done()
            if published:
                self.initial.set_result(value)
        if return_capacity:
            self.return_capacity()
        return published

    def publish_retained(self, error: BaseException) -> None:
        """Every waiter not yet answered hears that the queue keeps this turn (a moved turn's park): unlike a
        settlement nothing returns or retires here; the carrier's own settlement does that after its hold."""
        with self._lock:
            for future in (self.initial, self.result):
                if not future.done():
                    future.set_exception(error)


class PresenceTurnExecutions:
    """The Host's live presence turns by durable turn id: start one, or join the one already running.

    A Host turn's BODY runs in a conversation-lane carrier (``dispatch_presence_turn``,
    owner decision 1A): the Host admits and reserves; the carrier acquires the gate, runs
    ``run_presence_turn`` and reports its initial envelope and its terminal as frames
    (``presence_turn_initial`` / ``presence_turn_settled``) that resolve the futures here.
    ``start_thread`` keeps the proactive form (an initiating tool's turn runs where the tool
    runs) and a live call. Both versions return ``reserve``d capacity at a qualified park or
    terminal settlement; version 0's HTTP waiter stays open until terminal. Cancelling or
    disconnecting a waiter leaves the turn and any held capacity in place; retries join it.
    A process exit leaves the turn host-lost for retry.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._live: dict[str, PresenceTurnExecution] = {}
        _EXECUTION_REGISTRIES.add(self)

    def live(self) -> list[str]:
        with self._lock:
            return list(self._live)

    def find(self, turn_id: str) -> PresenceTurnExecution | None:
        with self._lock:
            return self._live.get(turn_id)

    def start_or_dispatch(
        self,
        turn_id: str,
        *,
        identity: str = "",
        reserve: Callable[[], bool],
        release: Callable[[], None],
        dispatch: Callable[[PresenceTurnExecution], None],
    ) -> tuple[PresenceTurnExecution | None, bool]:
        """``(execution, started)``; ``(None, False)`` when ``reserve`` refuses a new turn.

        ``dispatch(execution)`` admits the turn's body to a carrier (``dispatch_presence_turn``);
        the futures resolve from the carrier's frames. A dispatch that raises settles the
        execution with its error and re-raises. A live turn is joined before capacity is consulted.
        """
        with self._lock:
            live = self._live.get(turn_id)
            if live is not None:
                if identity and live.identity and identity != live.identity:
                    raise PresenceTurnError("presence_event_identity_conflict", "source_event_id", turn_ref=turn_id)
                return live, False
            if not reserve():
                return None, False
            execution = self._live[turn_id] = PresenceTurnExecution(turn_id, identity)
            execution._capacity = release
        try:
            dispatch(execution)
        except BaseException as exc:
            self._settle(execution, release, error=exc)
            raise
        return execution, True

    def settle_from_frame(self, execution: PresenceTurnExecution, *, result: Any = None,
                          error: BaseException | None = None) -> None:
        self._settle(execution, lambda: None, result=result, error=error)

    def start_or_join(
        self,
        turn_id: str,
        *,
        identity: str = "",
        reserve: Callable[[], bool],
        release: Callable[[], None],
        admit: Callable[[], Any],
        run: Callable[[PresenceTurnLease], Any],
    ) -> tuple[PresenceTurnExecution | None, bool]:
        """``(execution, started)``; ``(None, False)`` when ``reserve`` refuses a new turn.

        A live turn is joined before capacity is consulted: a retry never needs a new slot.
        Call from the event loop; ``admit`` is a coroutine function returning the lease.
        """
        with self._lock:
            live = self._live.get(turn_id)
            if live is not None:
                if identity and live.identity and identity != live.identity:
                    raise PresenceTurnError("presence_event_identity_conflict", "source_event_id", turn_ref=turn_id)
                return live, False
            if not reserve():
                return None, False
            execution = self._live[turn_id] = PresenceTurnExecution(turn_id, identity)
            execution._capacity = release
        admission = self._admit_and_start(execution, admit, run, release)
        try:
            execution.admission = asyncio.get_running_loop().create_task(admission)
        except BaseException:
            admission.close()
            self._settle(execution, release, error=PresenceTurnNotStarted("presence turn admission did not start"))
            raise
        execution.admission.add_done_callback(lambda task: self._admission_ended(execution, release, task))
        return execution, True

    def start_thread(self, turn_id: str, *, identity: str, admit: Callable[[], PresenceTurnLease],
                     run: Callable[[PresenceTurnLease], Any],
                     context: contextvars.Context) -> tuple[PresenceTurnExecution, bool]:
        """The proactive form of ``start_or_join``: admission waits on the turn's own thread."""
        with self._lock:
            live = self._live.get(turn_id)
            if live is not None:
                if identity and live.identity and identity != live.identity:
                    raise PresenceTurnError("presence_event_identity_conflict", "source_event_id", turn_ref=turn_id)
                return live, False
            execution = self._live[turn_id] = PresenceTurnExecution(turn_id, identity)

        def body() -> None:
            try:
                lease = admit()
            except BaseException as exc:
                self._settle(execution, lambda: None, error=exc)
                return
            self._execute(execution, run, lease, lambda: None)

        try:
            threading.Thread(target=context.run, args=(body,), name=f"presence-turn-{turn_id}", daemon=True).start()
        except BaseException as exc:
            error = PresenceTurnNotStarted("presence turn thread could not start")
            error.__cause__ = exc
            self._settle(execution, lambda: None, error=error)
            raise error from exc
        return execution, True

    def _admission_ended(self, execution: PresenceTurnExecution, release: Callable[[], None],
                         task: asyncio.Task) -> None:
        # Cancellation lands only before the thread starts — in ``admit`` or before the first step,
        # where the coroutine's own handlers never run (e.g. a loop shutting down).
        if task.cancelled() and not execution.result.done():
            self._settle(execution, release, error=PresenceTurnNotStarted("presence turn admission was interrupted"))

    async def _admit_and_start(self, execution: PresenceTurnExecution, admit: Callable[[], Any],
                               run: Callable[[PresenceTurnLease], Any], release: Callable[[], None]) -> None:
        try:
            lease = await admit()
        except asyncio.CancelledError:
            raise  # settled by _admission_ended
        except BaseException as exc:
            self._settle(execution, release, error=exc)
            if not isinstance(exc, Exception):
                raise
            return
        try:
            thread = threading.Thread(
                target=contextvars.copy_context().run, args=(self._execute, execution, run, lease, release),
                name=f"presence-turn-{execution.turn_id}", daemon=True,
            )
            thread.start()
        except BaseException as exc:  # e.g. "can't start new thread": nothing ran, so nothing stays held
            self._release_lease(execution, lease)
            error = PresenceTurnNotStarted("presence turn thread could not start")
            error.__cause__ = exc
            self._settle(execution, release, error=error)
            if not isinstance(exc, Exception):
                raise

    def _execute(self, execution: PresenceTurnExecution, run: Callable[[PresenceTurnLease], Any],
                 lease: PresenceTurnLease, release: Callable[[], None]) -> None:
        outcome: Any = None
        error: BaseException | None = None
        lease.execution = execution  # the author may publish its initial envelope through its lease
        try:
            outcome = run(lease)
        except BaseException as exc:  # relayed unchanged to every waiter
            error = exc
        self._release_lease(execution, lease)
        self._settle(execution, release, result=outcome, error=error)

    @staticmethod
    def _release_lease(execution: PresenceTurnExecution, lease: PresenceTurnLease) -> None:
        try:
            lease.release()
        except Exception:  # every release callback still ran; settlement must not be lost with it
            log.warning("presence turn %s could not release its gate lease", execution.turn_id, exc_info=True)

    def _settle(self, execution: PresenceTurnExecution, release: Callable[[], None], *,
                result: Any = None, error: BaseException | None = None) -> None:
        # Capacity first, then the id retires, then every waiter sees the outcome. The live set is
        # custody the Host reads the moment a waiter observes the terminal, and this usually runs on
        # the turn's own thread while the waiter wakes on the loop: an id published after its outcome
        # could outlive a response already on the wire. A retry that takes the lock before this joins
        # the running turn; one that lands after retirement rechecks durable authority — a turn that
        # reached the model left its RUNNING/terminal row, so it replays or refuses; only a turn that
        # failed before the start barrier may have no row, and it never generated anything to repeat.
        # A continuing author returned its capacity with its initial envelope; that is never repeated.
        execution.return_capacity(release)
        with self._lock:
            if self._live.get(execution.turn_id) is execution:
                del self._live[execution.turn_id]
        for future in (execution.initial, execution.result):
            with execution._lock:
                if future.done():
                    continue
                if error is None:
                    future.set_result(result)
                else:
                    future.set_exception(error)


# Every execution registry of this process (the Host's, the proactive one): a carrier's frame
# finds its waiting execution by turn id through them.
_EXECUTION_REGISTRIES: "weakref.WeakSet[PresenceTurnExecutions]" = weakref.WeakSet()

# Initiated turns run under this process's own execution registry (the Host keeps its own).
PROACTIVE_TURNS = PresenceTurnExecutions()


def _find_execution(turn_id: str) -> tuple[PresenceTurnExecutions | None, PresenceTurnExecution | None]:
    for registry in list(_EXECUTION_REGISTRIES):
        execution = registry.find(turn_id)
        if execution is not None:
            return registry, execution
    return None, None


__all__ = [
    "PROACTIVE_TURNS", "PresenceTurnExecution", "PresenceTurnExecutions", "PresenceTurnInterrupted",
    "PresenceTurnNotStarted",
]
