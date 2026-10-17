"""The Presence turn gate: one active turn per conversation under a cross-process cap.

Every admitted Presence turn, whatever started it (a transport event, an owner
initiation, a live call), takes the same two resources before any model or tool
step: its conversation's lock and one installation-wide active slot. The gate
lives apart from the runner so the runner stays the turn pipeline and the gate
stays the one place where the cap and the serialization order are defined.
``presence_runner`` re-exports every name here; its callers and tests keep them.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import threading
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Callable

_GATE_POLL_SEC = 0.05
# A stop check reads task controls from disk; a queued reacquisition asks at this cadence.
_GATE_STOP_POLL_SEC = 1.0


def _try_exclusive(fd: int) -> bool:
    from ouroboros.platform_layer import file_lock_exclusive_nb

    try:
        file_lock_exclusive_nb(fd)
    except OSError:
        return False
    return True


class PresenceTurnLease:
    """The gate resources one admitted turn holds; its owner releases them exactly once.

    A lease minted by a gate can also be LENT at a qualified review wait (#1536): the
    conversation and active slot return to the gate while the same author stays live, and
    ``reacquire`` takes both again, in admission order, before any model or tool step.
    ``execution`` is the Host turn whose initial envelope that author may publish.
    """

    def __init__(self, conversation_key: str, resources: ExitStack,
                 gate: "PresenceTurnGate | None" = None) -> None:
        self.conversation_key = conversation_key
        self._resources: ExitStack | None = resources
        self._lock = threading.Lock()
        self._gate = gate
        self._ended = False
        self.execution: Any | None = None  # a presence_runner.PresenceTurnExecution

    def release(self) -> None:
        with self._lock:
            resources, self._resources, self._ended = self._resources, None, True
        if resources is not None:
            resources.close()

    def lendable(self) -> bool:
        """Held, minted by a gate and not ended: ``lend`` can return it and ``reacquire`` take it back."""
        with self._lock:
            return self._gate is not None and not self._ended and self._resources is not None

    def lend(self) -> bool:
        """Return the held resources while keeping the lease reacquirable; False when it cannot."""
        with self._lock:
            if self._gate is None or self._ended or self._resources is None:
                return False
            resources, self._resources = self._resources, None
        resources.close()
        return True

    def held(self) -> bool:
        with self._lock:
            return self._resources is not None

    def reacquire(self, should_stop: Callable[[], bool]) -> bool:
        """Wait for the conversation and a slot again; False when ``should_stop`` ends the wait.

        A stop seen before the first attempt or right after the last one leaves nothing held,
        so a stopped author never acts even when both resources were free at once.
        """
        if self._gate is None:
            return False
        if self.held():
            return not should_stop()
        resources = self._gate.acquire_resources(self.conversation_key, should_stop)
        if resources is None:
            return False
        with self._lock:
            if self._ended or self._resources is not None:
                stale = resources
            else:
                self._resources, stale = resources, None
        if stale is not None:
            stale.close()
        return stale is None


class PresenceTurnGate:
    """Cross-process cap plus one active turn for each conversation.

    ``run`` waits on a thread (the tool path); ``admit`` takes the same resources as a
    coroutine for the Host, whose queued turns must hold no thread while they wait.
    """

    def __init__(self, max_active: int = 2, *, state_root: Path | None = None) -> None:
        self._max_active = max(1, int(max_active))
        self._slots = threading.BoundedSemaphore(max(1, int(max_active)))
        self._guard = threading.Lock()
        self._conversations: dict[str, threading.Lock] = {}
        self._state_root = Path(state_root).resolve(strict=False) if state_root is not None else None
        self._claimed_slots: set[int] = set()
        self._local_active = 0  # in-process slots held when no state root coordinates processes

    def _gate_file(self, name: str) -> Path:
        root = self._state_root / "presence_turn_gate"
        root.mkdir(parents=True, exist_ok=True)
        return root / name

    def _conversation(self, conversation_key: str) -> tuple[str, threading.Lock]:
        key = str(conversation_key or "").strip()
        if not key:
            from ouroboros.presence_runner import PresenceTurnError

            raise PresenceTurnError("presence_conversation_key_required", "conversation_key")
        with self._guard:
            return key, self._conversations.setdefault(key, threading.Lock())

    def _conversation_file(self, key: str) -> Path:
        return self._gate_file(f"conversation-{hashlib.sha256(key.encode('utf-8')).hexdigest()}.lock")

    @contextmanager
    def _file_lock(self, path: Path):
        from ouroboros.platform_layer import file_lock_exclusive, file_unlock

        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            file_lock_exclusive(fd)
            yield
        finally:
            try:
                file_unlock(fd)
            finally:
                os.close(fd)

    def _try_slot(self) -> Callable[[], None] | None:
        """Claim one free active slot without waiting: its release, or None while all are taken."""
        if self._state_root is None:
            if not self._slots.acquire(blocking=False):
                return None
            with self._guard:
                self._local_active += 1

            def release_local() -> None:
                with self._guard:
                    self._local_active -= 1
                self._slots.release()

            return release_local
        from ouroboros.platform_layer import file_unlock

        for index in range(self._max_active):
            with self._guard:
                if index in self._claimed_slots:
                    continue
                self._claimed_slots.add(index)
            try:
                fd = os.open(self._gate_file(f"slot-{index}.lock"), os.O_CREAT | os.O_RDWR, 0o600)
            except BaseException:  # an unopenable slot is an error, never a slot left claimed
                with self._guard:
                    self._claimed_slots.discard(index)
                raise
            if not _try_exclusive(fd):  # another process holds it
                os.close(fd)
                with self._guard:
                    self._claimed_slots.discard(index)
                continue

            def release(index: int = index, fd: int = fd) -> None:
                try:
                    file_unlock(fd)
                finally:
                    os.close(fd)
                    with self._guard:
                        self._claimed_slots.discard(index)

            return release
        return None

    def run(self, conversation_key: str, callback: Callable[[], Any]) -> Any:
        lease = self.acquire(conversation_key)
        try:
            return callback()
        finally:
            lease.release()

    def _acquisition(self, conversation_key: str):
        """Take what ``run`` takes, in the same order, without ever blocking.

        A generator that yields while a resource is busy and returns the ExitStack holding
        everything; closing it early releases whatever was already taken. The coroutine and
        the thread admissions differ only in how they sleep between attempts.
        """
        from ouroboros.platform_layer import file_unlock

        key, conversation_lock = self._conversation(conversation_key)
        with ExitStack() as held:
            while not conversation_lock.acquire(blocking=False):
                yield
            held.callback(conversation_lock.release)
            if self._state_root is not None:
                fd = os.open(self._conversation_file(key), os.O_CREAT | os.O_RDWR, 0o600)
                held.callback(os.close, fd)
                while not _try_exclusive(fd):
                    yield
                held.callback(file_unlock, fd)
            while (release_slot := self._try_slot()) is None:
                yield
            held.callback(release_slot)
            return held.pop_all()

    async def admit(self, conversation_key: str) -> PresenceTurnLease:
        """Take what ``run`` takes, in the same order, as a coroutine.

        Every attempt is non-blocking and every wait is an asyncio sleep, so a turn queued
        behind its conversation or the active cap parks no thread. Cancellation releases
        whatever was already taken; success hands it all to the returned lease.
        """
        key, _lock = self._conversation(conversation_key)
        steps = self._acquisition(key)
        try:
            while True:
                try:
                    next(steps)
                except StopIteration as done:
                    return PresenceTurnLease(key, done.value, self)
                await asyncio.sleep(_GATE_POLL_SEC)
        finally:
            steps.close()

    def acquire_resources(self, conversation_key: str,
                          should_stop: Callable[[], bool] | None = None) -> ExitStack | None:
        """The thread form of ``admit``; None when ``should_stop`` ends the wait.

        ``should_stop`` is asked before the first attempt, every ``_GATE_STOP_POLL_SEC``
        while waiting, and once more after the last attempt succeeds: what was taken as a
        stop landed is returned at once instead of handed to a stopped caller.
        """
        if should_stop is not None and should_stop():
            return None
        steps = self._acquisition(conversation_key)
        checked = time.monotonic()
        try:
            while True:
                try:
                    next(steps)
                except StopIteration as done:
                    resources = done.value
                    break
                if should_stop is not None and time.monotonic() - checked >= _GATE_STOP_POLL_SEC:
                    checked = time.monotonic()
                    if should_stop():
                        return None
                time.sleep(_GATE_POLL_SEC)
        finally:
            steps.close()
        if should_stop is not None and should_stop():
            resources.close()
            return None
        return resources

    def acquire(self, conversation_key: str) -> PresenceTurnLease:
        key, _lock = self._conversation(conversation_key)
        return PresenceTurnLease(key, self.acquire_resources(key), self)

    def try_acquire(self, conversation_key: str) -> PresenceTurnLease | None:
        """One attempt at what ``acquire`` takes; None at once while any of it is busy.

        A door that must answer now (a call open) uses this instead of parking a request
        in the gate queue; whatever the attempt took before the busy resource is released.
        """
        key, _lock = self._conversation(conversation_key)
        steps = self._acquisition(key)
        try:
            next(steps)
        except StopIteration as done:
            return PresenceTurnLease(key, done.value, self)
        steps.close()
        return None

    @property
    def max_active(self) -> int:
        return self._max_active

    def active_count(self) -> int:
        """Active slots held right now, by this process or another; a diagnostic read for
        a refusal message, never an admission decision (the lock attempt is that)."""
        if self._state_root is None:
            with self._guard:
                return self._local_active
        from ouroboros.platform_layer import file_unlock

        with self._guard:
            claimed = set(self._claimed_slots)
        held = len(claimed)
        for index in range(self._max_active):
            if index in claimed:
                continue
            try:
                fd = os.open(self._gate_file(f"slot-{index}.lock"), os.O_CREAT | os.O_RDWR, 0o600)
            except OSError:
                continue
            try:
                if _try_exclusive(fd):
                    file_unlock(fd)
                else:
                    held += 1
            finally:
                os.close(fd)
        return held


_GATES_LOCK = threading.Lock()
_GATES: dict[tuple[str, int], PresenceTurnGate] = {}
_LIVE_LOCK = threading.Lock()
_LIVE_PRESENCE_TASKS: set[tuple[str, str]] = set()


def presence_turn_is_live(task_id: str, *, drive_root: Path | None = None) -> bool:
    """Root-scoped local liveness; never an owner-addressable actor."""
    with _LIVE_LOCK:
        if drive_root is not None:
            return (str(Path(drive_root).resolve()), str(task_id or "")) in _LIVE_PRESENCE_TASKS
        return any(tid == str(task_id or "") for _root, tid in _LIVE_PRESENCE_TASKS)


def _configured_gate(drive_root: Path | None = None) -> PresenceTurnGate:
    from ouroboros.config import SETTINGS_DEFAULTS, _bounded_positive_int_setting

    limit = _bounded_positive_int_setting(
        "OUROBOROS_PRESENCE_MAX_ACTIVE",
        default=int(SETTINGS_DEFAULTS["OUROBOROS_PRESENCE_MAX_ACTIVE"]),
        hard_max=20,
    )
    state_root = Path(drive_root).resolve(strict=False) / "state" if drive_root is not None else None
    key = (str(state_root or ""), limit)
    with _GATES_LOCK:
        return _GATES.setdefault(key, PresenceTurnGate(limit, state_root=state_root))


async def admit_configured_gate(drive_root: Path, conversation_key: str) -> PresenceTurnLease:
    """Coroutine admission into the configured gate ``run_presence_turn`` would otherwise take."""
    return await _configured_gate(Path(drive_root)).admit(conversation_key)


__all__ = [
    "PresenceTurnGate",
    "PresenceTurnLease",
    "admit_configured_gate",
    "presence_turn_is_live",
]
