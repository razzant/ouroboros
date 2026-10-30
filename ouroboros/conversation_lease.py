"""Conversation authorship and the cross-process physical conversation bound.

The existing Presence gate retains lock order, coroutine admission, lending and
stop-aware reacquisition. Author records are projections under the held lock;
only a held conversation is addressable, while Stop also sees lent live authors.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
import time
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Callable

from ouroboros.utils import atomic_write_json

_AUTHORS_LOCK = threading.Lock()
_AUTHORS: dict[tuple[str, str], "PresenceTurnLease"] = {}

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
        self.execution: Any = None
        self.task_id = ""
        self.intake_agent: Any = None
        self._live_fd: int | None = None
        self._live_path: Path | None = None

    def bind_author(self, task_id: str) -> None:
        self.task_id = task_id
        if self._gate and self._gate._state_root and self._live_fd is None:
            # One unique inode per live stack, held across lending. A PID alone
            # would mistake a reused process id for the original author.
            with self._gate._file_lock(self._gate._gate_file("authors-census.lock")):
                self._live_path = self._gate._gate_file(f"author-{uuid.uuid4().hex}.live")
                self._live_fd = os.open(self._live_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
                from ouroboros.platform_layer import file_lock_exclusive
                file_lock_exclusive(self._live_fd)
                os.write(self._live_fd, json.dumps({"task_id": task_id}).encode())
        with _AUTHORS_LOCK:
            _AUTHORS[(self._root(), task_id)] = self
        self._publish_author()

    def _root(self) -> str:
        return str(self._gate._state_root.parent) if self._gate and self._gate._state_root else ""

    def _author_path(self) -> Path | None:
        return (self._gate._conversation_file(self.conversation_key).with_suffix(".author.json")
                if self._gate and self._gate._state_root else None)

    def _publish_author(self) -> None:
        path = self._author_path()
        if path and self.task_id:
            atomic_write_json(path, {"task_id": self.task_id, "conversation_key": self.conversation_key,
                                     "pid": os.getpid()})

    def _clear_author(self) -> None:
        path = self._author_path()
        if path and self.task_id:
            path.unlink(missing_ok=True)

    def _clear_live_author(self) -> None:
        fd, self._live_fd = self._live_fd, None
        try:
            if fd is not None:
                os.close(fd)
        finally:
            if self._live_path:
                self._live_path.unlink(missing_ok=True)

    def release(self) -> None:
        resources = None
        try:
            with self._lock:
                resources, self._resources, self._ended = self._resources, None, True
                try:
                    if resources is not None:
                        self._clear_author()
                finally:
                    self.intake_agent = None
                    with _AUTHORS_LOCK:
                        _AUTHORS.pop((self._root(), self.task_id), None)
                    self._clear_live_author()
        finally:
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
            self._clear_author()
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
                self._publish_author()
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
            return self._slots.release if self._slots.acquire(blocking=False) else None
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

        A generator that yields the busy resource (``conversation`` or ``capacity``) and returns
        the ExitStack holding everything; closing it early releases whatever was already taken.
        The admissions differ only in how they wait between attempts, or (``try_admit``) not at all.
        """
        from ouroboros.platform_layer import file_unlock

        key, conversation_lock = self._conversation(conversation_key)
        with ExitStack() as held:
            while not conversation_lock.acquire(blocking=False):
                yield "conversation"
            held.callback(conversation_lock.release)
            if self._state_root is not None:
                fd = os.open(self._conversation_file(key), os.O_CREAT | os.O_RDWR, 0o600)
                held.callback(os.close, fd)
                while not _try_exclusive(fd):
                    yield "conversation"
                held.callback(file_unlock, fd)
            while (release_slot := self._try_slot()) is None:
                yield "capacity"
            held.callback(release_slot)
            return held.pop_all()

    def try_admit(self, conversation_key: str) -> tuple[PresenceTurnLease | None, str]:
        """One non-blocking pass of ``admit``: ``(lease, "")``, or ``(None, busy)`` naming the first
        busy resource after releasing everything this pass took. Nothing waits or survives it."""
        key, _lock = self._conversation(conversation_key)
        steps = self._acquisition(key)
        try:
            busy = next(steps)
        except StopIteration as done:
            return PresenceTurnLease(key, done.value, self), ""
        finally:
            steps.close()
        return None, busy

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

    def try_acquire(self, conversation_key: str) -> PresenceTurnLease | None:
        """One admission attempt for a live peer door; no waiting worker."""
        key, _lock = self._conversation(conversation_key)
        steps = self._acquisition(key)
        try:
            try:
                next(steps)
            except StopIteration as done:
                return PresenceTurnLease(key, done.value, self)
            return None
        finally:
            steps.close()

    def acquire(self, conversation_key: str) -> PresenceTurnLease:
        key, _lock = self._conversation(conversation_key)
        return PresenceTurnLease(key, self.acquire_resources(key), self)



def live_author(conversation_key: str, *, drive_root: Any = None) -> str:
    root = str(Path(drive_root).resolve()) if drive_root is not None else ""
    with _AUTHORS_LOCK:
        local = list(_AUTHORS.items())
    for (where, task), lease in local:
        if (not root or where == root) and lease.conversation_key == conversation_key and lease.held():
            return task
    if not root:
        return ""
    path = Path(root) / "state" / "presence_turn_gate" / (
        "conversation-" + hashlib.sha256(conversation_key.encode()).hexdigest() + ".lock")
    try:
        fd = os.open(path, os.O_RDWR)
    except FileNotFoundError:
        return ""
    try:
        if _try_exclusive(fd):
            from ouroboros.platform_layer import file_unlock
            file_unlock(fd)
            return ""
        row = json.loads(path.with_suffix(".author.json").read_text(encoding="utf-8"))
        return str(row.get("task_id") or "") if row.get("conversation_key") == conversation_key else ""
    except (FileNotFoundError, ValueError):
        return ""
    finally:
        os.close(fd)


def presence_author_agent(drive_root, author: str):
    """The local inline author's intake owner, including while its conversation is lent.

    Question delivery uses this existing actor's final-drain lock. A census on
    another process never invents a local admission endpoint.
    """
    with _AUTHORS_LOCK:
        lease = _AUTHORS.get((str(Path(drive_root).resolve()), author))
    return lease.intake_agent if lease is not None and not lease._ended else None


def accept_author_mail(drive_root, author: str, conversation_key: str, append: Callable[[], bool]) -> bool:
    """Accept under the author's own intake/close lock, never a census-to-append race.

    A remote author has no local admission endpoint: leave the arrival to ordinary
    conversation admission instead of pretending a mailbox append is acceptance.
    """
    root = str(Path(drive_root).resolve())
    with _AUTHORS_LOCK:
        lease = _AUTHORS.get((root, author))
    agent = lease.intake_agent if lease is not None else None
    lock = getattr(agent, "_owner_message_admission_lock", None)
    if lock is None:
        return False
    with lock, lease._lock:
        if (lease._ended or lease._resources is None or lease.conversation_key != conversation_key
                or not getattr(agent, "_accepting_owner_messages", False) or not getattr(agent, "_busy", False)
                or getattr(agent, "_current_task_id", None) != author):
            return False
        return append()


def live_authors(*, drive_root: Any = None) -> list[str]:
    root = str(Path(drive_root).resolve()) if drive_root is not None else ""
    with _AUTHORS_LOCK:
        found = {task for (where, task) in _AUTHORS if not root or where == root}
    if root:
        found.update(_live_stack_authors(Path(root)))
        for path in (Path(root) / "state" / "presence_turn_gate").glob("*.author.json"):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
                task = live_author(row["conversation_key"], drive_root=root)
                if task:
                    found.add(task)
            except (FileNotFoundError, ValueError, KeyError):
                continue  # release can remove a projection during this census
    return sorted(found)


def _live_stack_authors(root: Path) -> set[str]:
    """O(live stacks + crash leftovers); reclaim only unlocked unique inodes."""
    gate = PresenceTurnGate(state_root=root / "state")
    with gate._file_lock(gate._gate_file("authors-census.lock")):
        return _scan_live_stack_authors(root)


def _scan_live_stack_authors(root: Path) -> set[str]:
    found = set()
    for path in (root / "state" / "presence_turn_gate").glob("author-*.live"):
        try:
            fd = os.open(path, os.O_RDWR)
        except FileNotFoundError:
            continue
        retire = False
        try:
            if _try_exclusive(fd):
                retire = True  # its stack ended, including a hard crash
            else:
                row = json.loads(os.read(fd, 8192))
                if row.get("task_id"):
                    found.add(str(row["task_id"]))
        except (FileNotFoundError, ValueError):
            continue
        finally:
            os.close(fd)
            if retire:
                path.unlink(missing_ok=True)  # Windows requires closing the locked handle first
    return found
