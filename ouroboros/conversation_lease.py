"""Conversation authorship and the cross-process physical conversation bound.

The existing Presence gate retains lock order, coroutine admission, lending and
stop-aware reacquisition. Author records are projections under the held lock;
only a held conversation is addressable, while Stop also sees lent live authors.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Callable

from ouroboros.utils import atomic_write_json

log = logging.getLogger(__name__)

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


def _remove_witness(path: Path) -> None:
    """Remove a stack witness nothing holds. Windows refuses the delete while a passive census reader has it open
    (``os.open`` shares no delete); that one refusal leaves the unlocked leftover, which names no live stack, for
    the next reclaim. Every other failure stays its caller's."""
    try:
        path.unlink(missing_ok=True)
    except PermissionError:
        log.debug("Presence author witness %s stays for the next census", path, exc_info=True)


class PresenceTurnLease:
    """The gate resources one admitted turn holds; its owner releases them exactly once.

    A lease minted by a gate can also be LENT at a qualified review wait (#1536): the
    conversation and active slot return to the gate while the same author stays live, and
    ``reacquire`` takes both again, in admission order, before any model or tool step.
    ``execution`` is the Host turn whose initial envelope that author may publish; ``review_wait`` the
    carrier's pooled owner wait its qualified review/private-question park lends the slot through (never the
    agent's generic owner-wait capability: a Presence author gets no sleep, detach, Pause or budget pause).
    """

    def __init__(self, conversation_key: str, resources: ExitStack,
                 gate: "PresenceTurnGate | None" = None) -> None:
        self.conversation_key = conversation_key
        self._resources: ExitStack | None = resources
        self._lock = threading.Lock()
        self._gate = gate
        self._ended = False
        self.execution: Any = None
        self.review_wait: Any = None
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
                _remove_witness(self._live_path)

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


def _intake_path(drive_root: Any, task_id: str) -> Path:
    from ouroboros.task_results import validate_task_id

    return Path(drive_root) / "memory" / "owner_mailbox" / f"{validate_task_id(task_id)}.intake.json"


def author_intake_accepting(drive_root: Any, task_id: str, conversation_key: str = "", *, attempt: Any = None) -> bool:
    """The author's durable intake fact: True while its loop reads this conversation's mail.

    Read under ``task_custody.task_mail_lock`` by the acceptance transaction below. ``attempt``: the dispatcher's
    current attempt, which a root's fact must name (a superseded attempt's loop is not this work's reader).
    """
    try:
        row = json.loads(_intake_path(drive_root, task_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not row.get("accepting") or (conversation_key and row.get("conversation_key") != conversation_key):
        return False
    if attempt is not None and row.get("attempt") != attempt:
        return False
    # The fact is the writer's: a carrier that died before its release left it standing, and a dead
    # author reads no mail (an arrival appended to it would be swallowed). Its process must be alive.
    return _writer_alive(row.get("pid"))


def author_intake_closed(drive_root: Any, task_id: str, conversation_key: str, *, attempt: Any) -> bool:
    """The positive opposite of ``author_intake_accepting``: this conversation's fact for this attempt says its loop
    reads no more (its final drain closed it, ``follow_loop_admission``). No fact, an unreadable or foreign one, or
    another attempt's proves nothing."""
    try:
        row = json.loads(_intake_path(drive_root, task_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (isinstance(row, dict) and row.get("accepting") is False and row.get("conversation_key") == conversation_key
            and row.get("attempt") == attempt)


def _writer_alive(pid: Any) -> bool:
    from ouroboros.platform_layer import pid_is_alive  # never os.kill(pid, 0): CTRL_C_EVENT on Windows

    try:
        return pid_is_alive(int(pid))
    except (TypeError, ValueError):
        return False


def _write_intake(drive_root: Any, task_id: str, row: dict) -> None:
    atomic_write_json(_intake_path(drive_root, task_id), row)


def open_author_intake(drive_root: Any, task_id: str, conversation_key: str, *, attempt: Any = None) -> bool:
    """The author's loop reads this conversation's mail from here on (under the mail lock).

    False when the lock could not be taken: nothing is opened, so no arrival is accepted that the lock
    could not order (the author still reads its own mailbox; a remote Host refuses, and the arrival
    becomes its own turn)."""
    from ouroboros.task_custody import task_mail_lock

    with task_mail_lock(drive_root, task_id) as locked:
        if not locked:
            return False
        _write_intake(drive_root, task_id, {"accepting": True, "conversation_key": conversation_key, "pid": os.getpid(),
                                            **({"attempt": attempt} if attempt is not None else {})})
        return True


def _own_row(row: Any, attempt: Any) -> bool:
    """A fact this process wrote for this attempt (a root's carries its attempt; an author's carries none)."""
    return isinstance(row, dict) and row.get("pid") == os.getpid() and row.get("attempt") == attempt


def close_author_intake(drive_root: Any, task_id: str, *, own: bool = False, attempt: Any = None) -> bool:
    """The author's loop stopped reading: an arrival offered from now on is refused (under the mail lock).

    The fact is removed even when the lock could not be taken (it must never outlive its reader); False
    then discloses that an arrival racing the close may have landed unread. A recipient-claim COMMIT
    delayed beyond this wait can finish after the last admission check; the following append may then
    miss the final drain. ``own`` removes only this process/attempt's fact, never a successor's."""
    from ouroboros.task_custody import task_mail_lock

    with task_mail_lock(drive_root, task_id) as locked:
        path = _intake_path(drive_root, task_id)
        try:
            if not own or _own_row(json.loads(path.read_text(encoding="utf-8")), attempt):
                path.unlink()
        except (OSError, ValueError):
            pass
        if not locked:
            log.warning("author intake of %s closed without its mail lock; a racing arrival may be unread", task_id)
        return bool(locked)


def retire_author_intake(drive_root: Any, task_id: str, conversation_key: str, *, attempt: Any) -> None:
    """A failed open leaves no fact of this conversation and attempt standing, whichever process wrote it: a carrier's
    failed take would leave the host's, which its loop's final close cannot move (``_own_row``) while its live writer
    keeps it accepting. Another conversation's or attempt's fact stays. Removed as ``close_author_intake`` removes,
    its lock refused or not: this attempt's loop has not read yet, so an arrival racing it waits for the first drain,
    and no other writer can replace the fact meanwhile (the host wrote it before the row existed; a later attempt is
    admitted only once this carrier is confirmed dead, ``worker_health.recover_confirmed_dead_worker``)."""
    from ouroboros.task_custody import task_mail_lock

    path = _intake_path(drive_root, task_id)
    if not path.exists():
        return
    with task_mail_lock(drive_root, task_id) as locked:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            if (isinstance(row, dict) and row.get("conversation_key") == conversation_key
                    and row.get("attempt") == attempt):
                path.unlink()
                if not locked:
                    log.warning("author intake of %s retired without its mail lock", task_id)
        except (OSError, ValueError):
            log.debug("author intake of %s not retired", task_id, exc_info=True)


def root_conversation_key(task: Any) -> str:
    """The conversation whose new text this live root reads, else "".

    A promoted or follow-up ROOT keeps the speaker carrier of the conversation it answers
    (``presence_root_carrier``, that carrier alone). A turn itself is its conversation's lease author,
    resumed under its own id too (it keeps its event identity, and its carrier opens intake only once it
    holds the conversation again); related work holding only a binding, a child and an owner root answer none."""
    from ouroboros.dialogue_provenance import presence_record_binding, presence_related_work

    metadata = task.get("metadata") if isinstance(task, dict) and isinstance(task.get("metadata"), dict) else {}
    if (not isinstance(task, dict) or task.get("_presence_turn") or task.get("_presence_run") or task.get("_is_direct_chat")
            or metadata.get("presence_event_identity") or not presence_related_work(presence_record_binding(task), task)):
        return ""
    presence = metadata.get("presence") if isinstance(metadata.get("presence"), dict) else {}
    event = presence.get("event") if isinstance(presence.get("event"), dict) else {}
    return str(event.get("conversation_key") or "")


OWNER_CONVERSATION = "owner:"  # an owner turn's intake key: this prefix and its chat id


def owner_conversation_key(task: Any) -> str:
    """``owner:<chat>`` for my human's own foreground turn: the conversation whose ordinary messages it reads; else "".

    Facts the host already stamps, never a mark of its own: the direct origin with the owner door's ingress
    (``dialogue_provenance.run_origin``: a wake or an auto-resume has none), on the conversation lane or not yet
    carried, any executor but a live call's (``inline_call``), in a chat of the owner's and bound to no Project.
    Presence, detached, Project and pooled work read none. Its fact is the host's from admission, then its carrier's
    (``worker_chat_lane.CarrierTurn``); the final drain closes it (``follow_loop_admission``)."""
    from ouroboros.dialogue_provenance import is_presence_task, run_origin
    from ouroboros.task_carrier import (
        EXECUTOR_INLINE_CALL,
        LANE_CONVERSATION,
        carrier_executor,
        execution_kind,
        execution_owner,
    )

    if not isinstance(task, dict):
        return ""
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    try:
        chat_id = int(task.get("chat_id") or 0)
    except (TypeError, ValueError):
        return ""
    if (chat_id <= 0 or execution_kind(task) != "direct"
            or execution_owner(task).get("lane") not in (None, LANE_CONVERSATION)
            or carrier_executor(task) == EXECUTOR_INLINE_CALL or not run_origin(task)["owner_ingress"]
            or is_presence_task(task) or task.get("detached_at") or metadata.get("detached_at")
            or str(task.get("project_id") or "").strip()):
        return ""
    return f"{OWNER_CONVERSATION}{chat_id}"


def task_attempt(task: Any) -> int:
    """The attempt the dispatcher gave this task (the published running row's ``attempt``)."""
    try:
        return int((task or {}).get("_attempt") or 1)
    except (TypeError, ValueError):
        return 1


def open_root_intake(drive_root: Any, task: Any) -> bool:
    """A root's loop reads its conversation from here on, on its own drive beside its mailbox; never fails the loop."""
    key = root_conversation_key(task)
    try:
        return bool(key) and open_author_intake(drive_root, str(task.get("id") or ""), key, attempt=task_attempt(task))
    except Exception:
        log.debug("root intake fact not written", exc_info=True)
        return False


def close_root_intake(drive_root: Any, task: Any) -> None:
    try:
        close_author_intake(drive_root, str(task.get("id") or ""), own=True, attempt=task_attempt(task))
    except Exception:
        log.debug("root intake of %s not closed", task.get("id"), exc_info=True)


@contextmanager
def root_intake(drive_root: Any, task: Any):
    """The scope of a root's loop: open before its first drain, removed as the loop ends however it ends;
    the final drain moves this same fact in between (``follow_loop_admission``)."""
    opened = open_root_intake(drive_root, task)
    try:
        yield
    finally:
        if opened:
            close_root_intake(drive_root, task)


def follow_loop_admission(ctx: Any, accepting: bool) -> None:
    """A carried Presence author's, root's or owner turn's durable intake follows its loop's own admission.

    A Host or a server in another process reads only this fact, so it moves with the loop under the mail lock:
    closed BEFORE the final drain reads (an arrival either lands first and is drained, or is refused
    and becomes its own turn), reopened when the loop reads mail again (an owner follow-up, a reopened
    final). Only the writer's own process moves its fact; a turn with no intake row has none to follow."""
    metadata = getattr(ctx, "task_metadata", None)
    root, task_id = getattr(ctx, "drive_root", None), str(getattr(ctx, "task_id", "") or "")
    if not isinstance(metadata, dict) or root is None or not task_id:
        return
    from ouroboros.task_custody import task_mail_lock

    try:
        path = _intake_path(root, task_id)
        if not path.exists():
            return
        with task_mail_lock(root, task_id) as locked:
            row = json.loads(path.read_text(encoding="utf-8"))
            if not (isinstance(metadata.get("presence"), dict)
                    or str(row.get("conversation_key") or "").startswith(OWNER_CONVERSATION)):
                return
            attempt = task_attempt({"_attempt": getattr(ctx, "task_attempt", None)}) if "attempt" in row else None
            if not _own_row(row, attempt) or bool(row.get("accepting")) is bool(accepting):
                return
            if not locked:
                if accepting:
                    return  # never reopened unordered: the loop's own mailbox stays its only intake
                log.warning("author intake of %s closed without its mail lock; a racing arrival may be unread",
                            task_id)
            _write_intake(root, task_id, {**row, "accepting": bool(accepting)})
    except (OSError, ValueError):
        log.debug("author intake of %s did not follow its loop's admission", task_id, exc_info=True)


def accept_author_mail(drive_root, author: str, conversation_key: str, append: Callable[..., bool]) -> bool:
    """Accept an arrival for a live author while it still reads, never a census-to-append race.

    A LOCAL author (an inline executor's actor in this process) is asked under its own
    intake/close lock as before. A REMOTE author (a conversation-lane carrier) is asked
    through one transaction under its mailbox lock (``task_custody.task_mail_lock``): the
    durable intake fact (``open_author_intake`` / ``close_author_intake``, written under the
    same lock) and the append are one step, so no row lands after the author stopped
    reading. ``append(admit=...)`` performs the write; it receives the check to run inside
    its own locked append (``owner_mailbox.write_task_message(admit=...)``).
    """
    root = str(Path(drive_root).resolve())
    with _AUTHORS_LOCK:
        lease = _AUTHORS.get((root, author))
    agent = lease.intake_agent if lease is not None else None
    lock = getattr(agent, "_owner_message_admission_lock", None)
    if lock is not None:
        with lock, lease._lock:
            if (lease._ended or lease._resources is None or lease.conversation_key != conversation_key
                    or not getattr(agent, "_accepting_owner_messages", False) or not getattr(agent, "_busy", False)
                    or getattr(agent, "_current_task_id", None) != author):
                return False
            return append()
    if not author_intake_accepting(drive_root, author, conversation_key):
        return False
    try:
        return bool(append(admit=lambda: author_intake_accepting(drive_root, author, conversation_key)))
    except TypeError:
        return False  # an append that takes no admission check cannot be made one transaction


def accept_lent_author_mail(drive_root, author: str, conversation_key: str, append: Callable[..., bool],
                            qualified: Callable[[], bool]) -> bool:
    """Accept an arrival for an author parked with its conversation lent: it holds no lease, and its wait reads
    its mailbox. A local author is asked under its own admission lock (its final closes there); a remote one has only
    its durable facts. Both run ``qualified`` inside the append's mail lock, so the two mean the same thing."""
    with _AUTHORS_LOCK:
        lease = _AUTHORS.get((str(Path(drive_root).resolve()), author))
    agent = lease.intake_agent if lease is not None else None
    lock = getattr(agent, "_owner_message_admission_lock", None)
    if lock is None:
        return bool(append(admit=qualified))
    with lock:
        if (lease._ended or lease.conversation_key != conversation_key or not getattr(agent, "_accepting_owner_messages", False)
                or not getattr(agent, "_busy", False) or getattr(agent, "_current_task_id", None) != author):
            return False
        return bool(append(admit=qualified))


def live_presence_roots(drive_root: Any) -> dict:
    """Every live inline Presence author of ``drive_root``: a read-only projection, never an owner or registry.

    ``local`` authors run in this process (their lease here and their stack in the runner's live set);
    another process's authors are known only by their stack inode. The conversation locks are never
    probed, so a reader cannot make an admission meet a busy conversation, and the inode census is
    observed, never reclaimed (``_observe_live_stack_authors``). ``complete`` is False when it could not
    be read whole. Liveness only: whether an author takes mail is ``author_intake_open``, and every door
    rechecks both under that author's own lock. Never raises.
    """
    from ouroboros.presence_runner import presence_turn_is_live

    root = Path(drive_root).resolve()
    with _AUTHORS_LOCK:
        mine = {task for (where, task) in _AUTHORS if where == str(root)}
    try:  # no gate directory: no author ever ran here, and a read creates nothing
        everywhere, complete = (_observe_live_stack_authors(root)
                                if (root / "state" / "presence_turn_gate").is_dir() else (set(), True))
    except Exception:
        everywhere, complete = set(), False
    local = sorted(task for task in mine if presence_turn_is_live(task, drive_root=root))
    return {"roots": [{"task_id": task, "local": True} for task in local]
            + [{"task_id": task, "local": False} for task in sorted(everywhere - mine)], "complete": complete}


def resolved_presence_turns(drive_root: Any) -> dict:
    """The one reading every Presence reader shares: ``live_presence_roots`` with each author's durable row read once.

    ``state`` says what ``row`` proves: ``running`` (the unsettled row its runner owns), ``preparing`` (nothing stored
    yet: a runner publishes its witness before it builds the agent and writes RUNNING, so absence here is a fact of a
    starting turn, never of an ended one; ``row`` holds only the room a local one's conversation names, {} elsewhere)
    or ``unreadable`` (``row`` {}, ``error`` names why; never called ended). A settled row's stack is only unwinding
    and is left out. ``complete`` also says whether every row could be read. Facts only: intake and every door stay
    the author's own. Never raises.
    """
    from ouroboros.presence_runner import presence_chat_id
    from ouroboros.task_results import load_task_result
    from supervisor.cancel_publication import live_presence_row

    census = live_presence_roots(drive_root)
    roots, complete = [], census["complete"]
    for fact in census["roots"]:
        task_id = fact["task_id"]
        try:
            row = load_task_result(drive_root, task_id, strict=True) or {}
            if row and not live_presence_row(row):
                continue
        except Exception as exc:
            roots.append({**fact, "state": "unreadable", "row": {}, "error": type(exc).__name__})
            complete = False
            continue
        if row:
            from ouroboros.lead_fact import lead_fields

            actor = presence_author_agent(drive_root, task_id) if fact["local"] else None
            owner = getattr(getattr(actor, "tools", None), "_ctx", None)
            row.update(lead_fields(getattr(owner, "_lead_last_sent", None)))
            roots.append({**fact, "state": "running", "row": row})
            continue
        with _AUTHORS_LOCK:
            lease = _AUTHORS.get((str(Path(drive_root).resolve()), task_id))
        roots.append({**fact, "state": "preparing",
                      "row": {"chat_id": presence_chat_id(lease.conversation_key)} if lease is not None else {}})
    return {"roots": roots, "complete": complete}


def author_intake_open(drive_root: Any, author: str) -> bool:
    """Whether this process's live ``author`` takes owner and peer mail now, read under its admission lock.

    A lent author still drains its mailbox (owner mail wakes its wait); place arrivals keep the
    stricter ``accept_author_mail`` and ``accept_lent_author_mail``. A closing author is live and takes nothing.
    """
    agent = presence_author_agent(drive_root, author)
    lock = getattr(agent, "_owner_message_admission_lock", None)
    if lock is None:
        return False
    with lock:
        return bool(getattr(agent, "_busy", False) and getattr(agent, "_accepting_owner_messages", False)
                    and getattr(agent, "_current_task_id", None) == author)


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
        return _scan_live_stack_authors(root)[0]


def _observe_live_stack_authors(root: Path) -> tuple[set[str], bool]:
    """The same census as a pure observation: it creates, reclaims and writes nothing, and ``complete`` is False
    when a held witness cannot be read. It waits on the existing census lock as a writer does, so it never meets a
    half-made witness; ended witnesses stay for ``live_authors`` to reclaim."""
    from ouroboros.platform_layer import file_lock_exclusive, file_unlock

    gate = root / "state" / "presence_turn_gate"
    try:
        fd = os.open(gate / "authors-census.lock", os.O_RDONLY)
    except FileNotFoundError:  # the lock precedes every witness (``bind_author``): one without it is unreadable here
        return set(), not any(gate.glob("author-*.live"))
    try:
        file_lock_exclusive(fd)
        try:
            return _scan_live_stack_authors(root, reclaim=False)
        finally:
            file_unlock(fd)
    finally:
        os.close(fd)


def _scan_live_stack_authors(root: Path, *, reclaim: bool = True) -> tuple[set[str], bool]:
    found, complete = set(), True
    for path in (root / "state" / "presence_turn_gate").glob("author-*.live"):
        try:
            fd = os.open(path, os.O_RDWR if reclaim else os.O_RDONLY)
        except FileNotFoundError:
            continue  # its author released it during this census
        except OSError:
            if reclaim:
                raise
            complete = False
            continue
        retire = False
        try:
            if _try_exclusive(fd):
                retire = reclaim  # its stack ended, including a hard crash
            else:
                row = json.loads(os.read(fd, 8192))
                task = row.get("task_id") if isinstance(row, dict) else None
                if task:
                    found.add(str(task))
                else:
                    complete = False  # a held witness this read cannot name
        except (FileNotFoundError, ValueError):
            complete = False
            continue
        finally:
            os.close(fd)
            if retire:
                _remove_witness(path)  # Windows requires closing the locked handle first
    return found, complete
