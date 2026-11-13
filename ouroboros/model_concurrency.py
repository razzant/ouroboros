"""Fair physical-call admission shared by the processes of one installation.

The owner's limit (``settings_scales.MODEL_CAP_KEY``: not chosen is no limit for an API route and
three for a local model; <=0 disables) belongs to one model on a resolved
provider, credential and endpoint, across roles and every process of the installation.
Routes the managed Claudexor engine serves are not queued here: that one daemon is
already the shared admission point for its accounts (quota-aware routing and
rotation), and a host cap would only shrink an install's parallel work to the cap
across all its models. Callers bind their deadline with
``model_call_scope``; adapters take ``model_call_slot(route=target)`` INSIDE the
physical sender, so retries, tools, backoff and abandoned logical waits own no slot.
A waiter has no clock of its own: it keeps its place until a slot frees, its caller's
deadline passes, or Stop, Pause or the dispatch fence it already polls ends it. The work
is never sent elsewhere for waiting (``loop_transport.fallback_chain_allowed``); while it
waits it says so through the note its loop bound (``slot_wait_notes``).

``state/model_concurrency/<route digest>/`` contains zero-byte ``*.ticket`` and
``*.slot`` files: their names carry enqueue ordinal, PID and a random identity.
Under a short portable ``queue.lock``, the first live ticket may become a slot
only below the current cap. A call of a turn the owner awaits sorts ahead of every
background ticket that has not started (a background ordinal carries an offset);
each kind keeps its arrival order, and a started call keeps its slot.
A process-local counter avoids contending on that lock
while this process already fills the cap. Lowering the cap drains existing calls;
it never replaces the pool or evicts a live holder. A passed deadline refuses, never bypasses.

Release/cancellation removes its file; admission sweeps files whose PID is dead.
No age expires a live call. Empty route directories remain (one per used route),
with no history/counter file. One host/PID namespace is required; remote work that
outlives its dead host and conservative PID-reuse retention are outside this cap.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Callable
import uuid

from ouroboros.config import runtime_setting
from ouroboros import platform_layer
from ouroboros.transport_custody import ProviderNotDispatched

_POLL_SEC = 0.05
_BACKGROUND = 10 ** 18  # a background ticket's ordinal offset: an awaited turn's ticket sorts ahead of it
_LOCK = threading.Lock()
_LOCAL: dict[tuple[int, str], _LocalRoute] = {}
_DEADLINE = contextvars.ContextVar("model_slot_deadline", default=None)
_NOTE = contextvars.ContextVar("model_slot_note", default=None)


class ModelSlotUnavailable(ProviderNotDispatched):
    """Local capacity was not acquired; no provider request was sent."""

    code = "model_capacity_unavailable"
    retryable = False


class ModelSlotTimeout(ModelSlotUnavailable, TimeoutError):
    """The caller's deadline passed before a slot freed."""


class ModelSlotCancelled(ModelSlotUnavailable):
    """The waiter was cancelled before dispatch."""


def capacity_refusal(usage: object) -> bool:
    """Whether a turn's latest attempt is this limiter's refusal (no slot, nothing sent), from the
    per-failure code ``loop_llm_call`` stamps. It words the outcome and keeps the work on its route: the
    install's own queue is no failure of the model, so it never opens the fallback list
    (``loop_transport.fallback_chain_allowed``). Silent about the episode's earlier attempts."""
    return isinstance(usage, dict) and str(usage.get("_last_llm_provider_code") or "") == ModelSlotUnavailable.code


def _owner_awaits() -> bool:
    """This call belongs to a turn the owner awaits; an unreadable binding queues it as background."""
    try:
        from ouroboros.model_wait import owner_awaits_current_call

        return owner_awaits_current_call()
    except Exception:
        return False


class _LocalRoute:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0


def _cap(use_local: bool = False) -> int:
    from ouroboros.config import SETTINGS_PATH
    from ouroboros.settings_integrity import read_settings_json_verified
    from ouroboros.settings_scales import MODEL_CAP_KEY, MODEL_CAP_LOCAL, model_cap_value, unchosen_saved_document

    # Capacity belongs to the install, not an older task's frozen settings.
    # Every process observes an owner cap change through the same document (read as the settings seam reads it:
    # a 3 saved before the choice became optional is not a choice); an unreadable document falls back to the
    # process's own value, never refuses a call.
    try:
        saved = unchosen_saved_document(read_settings_json_verified(SETTINGS_PATH))
    except (OSError, ValueError):
        saved = None
    cap = model_cap_value((saved or {}).get(MODEL_CAP_KEY, runtime_setting(MODEL_CAP_KEY, "")))
    # Not chosen: a provider's API keeps its own limit; a model on this computer computes only what its slots allow.
    return (MODEL_CAP_LOCAL if use_local else 0) if cap is None else cap


def enabled(use_local: bool = False) -> bool:
    return _cap(use_local) > 0


@contextlib.contextmanager
def slot_wait_notes(note: Callable[[float, bool], None] | None):
    """Bind a work's note for a queued send: ``note(waited_sec, False)`` when the send first finds its queue full and
    at ``NETWORK_WAIT_NOTE_INTERVAL_SEC`` while it waits, ``note(waited_sec, True)`` once when it gets its slot. A send
    that never queued notes nothing. A note is a record only: it never ends or shortens the wait."""
    token = _NOTE.set(note)
    try:
        yield
    finally:
        _NOTE.reset(token)


@contextlib.contextmanager
def model_call_scope(model: str, use_local: bool = False, deadline_ts: float | None = None):
    """Carry a caller's wait bound through routing without holding physical capacity."""
    deadlines = [value for value in (_DEADLINE.get(), deadline_ts) if value is not None]
    token = _DEADLINE.set(min(deadlines) if deadlines else None)
    try:
        yield
    finally:
        _DEADLINE.reset(token)


def _managed(route: dict | None) -> bool:
    """A route the managed Claudexor engine serves (its daemon arbitrates accounts itself)."""
    return isinstance(route, dict) and str(route.get("provider") or "") == "claudexor"


def _route_digest(model: str, use_local: bool, route: dict | None) -> str:
    if route is None:
        if use_local:
            route = {"provider": "local"}
        else:
            # Compatibility for standalone callers. Physical adapters always pass
            # their actual target, including client-specific endpoint/key overrides.
            from ouroboros.llm import LLMClient
            route = LLMClient().resolve_remote_target(model)
    provider = str(route.get("provider") or "unknown")
    endpoint = str(route.get("base_url") or "").strip().rstrip("/")
    if provider == "local" and not endpoint:
        endpoint = f"http://127.0.0.1:{runtime_setting('LOCAL_MODEL_PORT', 8766)}/v1"
    account = {key: str(route.get(key) or "") for key in ("api_key", "user", "password", "credential_profile_id")}
    identity = [provider, str(route.get("source") or ""), endpoint, account, str(model or "")]
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _queue_root() -> Path:
    from ouroboros.config import resolve_data_dir
    return resolve_data_dir() / "state" / "model_concurrency"


class _Waiter:
    def __init__(self, model, use_local, deadline_ts, route, cancel_event, check):
        self.check = check
        self.cancel_event = cancel_event
        deadlines = [float(deadline) for deadline in (_DEADLINE.get(), deadline_ts) if deadline is not None]
        self.model, self.use_local = str(model or ""), bool(use_local)
        self.started = time.monotonic()
        # No clock of its own: only the caller's deadline bounds the wait (None: until a slot, Stop or Pause).
        self.until = self.started + max(0.0, min(deadlines) - time.time()) if deadlines else None
        self.queued, self.noted = False, None
        self.directory = _queue_root() / _route_digest(model, use_local, route)
        self.path: Path | None = None
        self.acquired = False
        self.awaited = _owner_awaits()
        key = (os.getpid(), str(self.directory))
        with _LOCK:
            self.local = _LOCAL.setdefault(key, _LocalRoute())

    def remaining(self):
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise ModelSlotCancelled(f"Local model slot wait cancelled before dispatch{self._subject()}; the request was not sent")
        if self.check is not None:
            self.check()
            from ouroboros.owner_pause import RAIL_OWNER_PAUSE, scope_fence
            from ouroboros.llm_attempt import _PhysicalSendNotStarted

            # Accounting's executor handoff precedes this queue. A queued waiter
            # has not started provider work, so Pause still cancels admission.
            if self.path is not None and scope_fence():
                raise _PhysicalSendNotStarted(RAIL_OWNER_PAUSE)
        if self.until is None:
            return math.inf
        remaining = self.until - time.monotonic()
        if remaining <= 0:
            # The install's own per-model slot, never a provider: the raw llm_api_error row used to read
            # "Provider capacity" for waits no provider ever saw. Only observed facts: no slot was acquired
            # within the measured wait, and the caller's deadline ended it.
            raise ModelSlotTimeout(f"Local model slot wait expired before dispatch{self._subject()}; no slot was acquired "
                                   f"within {time.monotonic() - self.started:.0f} s (the caller's deadline); the request was not sent")
        return remaining

    def note(self, admitted=False):
        """Tell the bound work that this send waits in line (at once, then at the note interval), and how long it
        waited when it gets its slot."""
        from ouroboros.config import NETWORK_WAIT_NOTE_INTERVAL_SEC

        note, now = _NOTE.get(), time.monotonic()
        if note is None or not self.queued or (
                not admitted and self.noted is not None and now - self.noted < NETWORK_WAIT_NOTE_INTERVAL_SEC):
            return
        self.noted = now
        try:
            note(now - self.started, admitted)
        except Exception:  # a note never ends a wait or a send
            pass

    def _subject(self):
        return f": model {self.model}" if self.model else ""

    def _live_files(self):
        files = []
        alive = {}
        for path in self.directory.iterdir():
            if path.suffix not in {".ticket", ".slot"}:
                continue
            try:
                ordinal, pid, _identity = path.stem.split("-")
                pid = int(pid)
                ordinal = int(ordinal)
            except ValueError as error:
                raise ModelSlotUnavailable("Invalid local model slot record") from error
            if pid not in alive:
                alive[pid] = platform_layer.pid_is_alive(pid)
            if alive[pid]:
                files.append((ordinal, path))
            else:
                path.unlink(missing_ok=True)
        return files

    def step(self):
        self.remaining()
        cap = _cap(self.use_local)
        if cap <= 0:
            return True
        with self.local.lock:
            # Register EVERY waiter before taking the local fast path; otherwise
            # a new process could repeatedly jump ahead of local waiting threads.
            if self.path is not None and self.local.active >= cap:
                self.queued = True
                return False
            lock_path = self.directory / "queue.lock"
            outcome = {}
            fd = platform_layer.acquire_exclusive_file_lock(
                lock_path, timeout_sec=0, owner_aware_stale=True, outcome=outcome)
            if fd is None:
                if outcome.get("reason") == "contention":
                    return False
                raise ModelSlotUnavailable("Local model slot queue could not be locked")
            try:
                self.remaining()
                files = self._live_files()
                if self.path is None:
                    sequence = max((number % _BACKGROUND for number, _ in files), default=0) + 1
                    ordinal = sequence if self.awaited else _BACKGROUND + sequence
                    self.path = self.directory / f"{ordinal:020d}-{os.getpid()}-{uuid.uuid4().hex}.ticket"
                    # All facts live in the filename: O_EXCL publishes a complete
                    # record even if the process dies immediately after creation.
                    self.path.touch(exist_ok=False)
                    files.append((ordinal, self.path))
                tickets = sorted(path.name for _, path in files if path.suffix == ".ticket")
                holders = sum(path.suffix == ".slot" for _, path in files)
                if holders >= cap or tickets[0] != self.path.name:
                    self.queued = True  # a full queue, not a busy lock: the work may say it waits
                    return False
                slot = self.path.with_suffix(".slot")
                self.path.rename(slot)
                self.path = slot
                self.local.active += 1
                self.acquired = True
                return True
            finally:
                platform_layer.release_exclusive_file_lock(lock_path, fd)

    def close(self):
        with self.local.lock:
            if self.path is not None:
                self.path.unlink(missing_ok=True)
                self.path = None
            if self.acquired:
                self.local.active -= 1
                self.acquired = False


@contextlib.contextmanager
def model_call_slot(model: str, use_local: bool = False, deadline_ts: float | None = None,
                    *, route: dict | None = None, cancel_event=None, check: Callable | None = None):
    """Hold capacity for ONE physical send; cancel/check is polled while waiting.

    ``route`` is the adapter's resolved target, never a requested model alias.
    The owning sender must exit this context, even if its caller stops waiting.
    """
    if _managed(route) or not enabled(use_local):
        yield
        return
    waiter = _Waiter(model, use_local, deadline_ts, route, cancel_event, check)
    try:
        try:
            while not waiter.step():
                waiter.note()
                delay = min(_POLL_SEC, waiter.remaining())
                if cancel_event is not None:
                    cancel_event.wait(delay)
                else:
                    time.sleep(delay)
        except ModelSlotUnavailable:
            raise
        except OSError as error:
            raise ModelSlotUnavailable("Local model slot storage unavailable") from error
        waiter.note(admitted=True)
        yield
    finally:
        waiter.close()


@contextlib.asynccontextmanager
async def model_call_slot_async(model: str, use_local: bool = False, deadline_ts: float | None = None,
                                *, route: dict | None = None, check: Callable | None = None):
    """Same queue with cancellable async waits, with no orphan acquiring thread."""
    if _managed(route) or not enabled(use_local):
        yield
        return
    waiter = _Waiter(model, use_local, deadline_ts, route, None, check)
    try:
        try:
            while not waiter.step():
                waiter.note()
                await asyncio.sleep(min(_POLL_SEC, waiter.remaining()))
        except ModelSlotUnavailable:
            raise
        except OSError as error:
            raise ModelSlotUnavailable("Local model slot storage unavailable") from error
        waiter.note(admitted=True)
        yield
    finally:
        waiter.close()


def reset_for_tests() -> None:
    """Reset only process-local counters; never delete another process's claims."""
    with _LOCK:
        _LOCAL.clear()


def limit_physical_send(route: dict, send: Callable) -> Callable:
    """Wrap the actual sender, preserving accounting's late-sender ownership."""
    from ouroboros.llm_attempt import require_physical_dispatch_window

    model = str(route.get("usage_model") or route.get("resolved_model") or "")
    local = str(route.get("provider") or "") == "local"  # the built-in server: it keeps three while no number is chosen

    def dispatch():
        with model_call_slot(model, local, route=route, check=require_physical_dispatch_window):
            require_physical_dispatch_window()
            return send()

    async def dispatch_async():
        async with model_call_slot_async(model, local, route=route, check=require_physical_dispatch_window):
            require_physical_dispatch_window()
            return await send()

    return dispatch_async if inspect.iscoroutinefunction(send) else dispatch
