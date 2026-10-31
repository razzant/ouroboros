"""Fair physical-call admission shared by the processes of one installation.

The configured cap (default three, <=0 disables) belongs to one model on a resolved
provider, credential and endpoint, across roles and every process of the installation.
Routes the managed Claudexor engine serves are not queued here: that one daemon is
already the shared admission point for its accounts (quota-aware routing and
rotation), and a host cap would only shrink an install's parallel work to the cap
across all its models. Callers bind their deadline with
``model_call_scope``; adapters take ``model_call_slot(route=target)`` INSIDE the
physical sender, so retries, tools, backoff and abandoned logical waits own no slot.

``state/model_concurrency/<route digest>/`` contains zero-byte ``*.ticket`` and
``*.slot`` files: their names carry enqueue ordinal, PID and a random identity.
Under a short portable ``queue.lock``, the first live ticket may become a slot
only below the current cap. A call of a turn the owner awaits sorts ahead of every
background ticket that has not started (a background ordinal carries an offset);
each kind keeps its arrival order, and a started call keeps its slot.
A process-local counter avoids contending on that lock
while this process already fills the cap. Lowering the cap drains existing calls;
it never replaces the pool or evicts a live holder. Timeout refuses, never bypasses.

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


class ModelSlotUnavailable(ProviderNotDispatched):
    """Local capacity was not acquired; no provider request was sent."""

    code = "model_capacity_unavailable"
    retryable = False


class ModelSlotTimeout(ModelSlotUnavailable, TimeoutError):
    """The bounded admission wait ended without capacity."""


class ModelSlotCancelled(ModelSlotUnavailable):
    """The waiter was cancelled before dispatch."""


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


def _max_slot_wait_sec() -> float:
    from ouroboros.config import SETTINGS_DEFAULTS

    default = SETTINGS_DEFAULTS["OUROBOROS_MODEL_SLOT_MAX_WAIT_SEC"]
    try:
        value = float(runtime_setting("OUROBOROS_MODEL_SLOT_MAX_WAIT_SEC", default))
        return max(0.0, value) if math.isfinite(value) else float(default)
    except (TypeError, ValueError):
        return float(default)


def _cap() -> int:
    from ouroboros.config import SETTINGS_DEFAULTS, SETTINGS_PATH
    from ouroboros.settings_integrity import read_settings_json_verified

    default = SETTINGS_DEFAULTS.get("OUROBOROS_MODEL_MAX_CONCURRENCY", 3)
    # Capacity belongs to the install, not an older task's frozen settings.
    # Every process observes an owner cap change through the same document; an
    # unreadable document falls back to the process's own value, never refuses a call.
    try:
        saved = read_settings_json_verified(SETTINGS_PATH)
    except (OSError, ValueError):
        saved = None
    value = (saved or {}).get("OUROBOROS_MODEL_MAX_CONCURRENCY",
                             runtime_setting("OUROBOROS_MODEL_MAX_CONCURRENCY", default))
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def enabled() -> bool:
    return _cap() > 0


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
        timeout = _max_slot_wait_sec()
        for deadline in (_DEADLINE.get(), deadline_ts):
            if deadline is not None:
                timeout = min(timeout, max(0.0, float(deadline) - time.time()))
        self.until = time.monotonic() + timeout
        self.directory = _queue_root() / _route_digest(model, use_local, route)
        self.path: Path | None = None
        self.acquired = False
        self.awaited = _owner_awaits()
        key = (os.getpid(), str(self.directory))
        with _LOCK:
            self.local = _LOCAL.setdefault(key, _LocalRoute())

    def remaining(self):
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise ModelSlotCancelled("Provider capacity wait cancelled before dispatch")
        if self.check is not None:
            self.check()
            from ouroboros.owner_pause import RAIL_OWNER_PAUSE, scope_fence
            from ouroboros.llm_attempt import _PhysicalSendNotStarted

            # Accounting's executor handoff precedes this queue. A queued waiter
            # has not started provider work, so Pause still cancels admission.
            if self.path is not None and scope_fence():
                raise _PhysicalSendNotStarted(RAIL_OWNER_PAUSE)
        remaining = self.until - time.monotonic()
        if remaining <= 0:
            raise ModelSlotTimeout("Provider capacity wait expired before dispatch")
        return remaining

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
                raise ModelSlotUnavailable("Invalid provider capacity record") from error
            if pid not in alive:
                alive[pid] = platform_layer.pid_is_alive(pid)
            if alive[pid]:
                files.append((ordinal, path))
            else:
                path.unlink(missing_ok=True)
        return files

    def step(self):
        self.remaining()
        cap = _cap()
        if cap <= 0:
            return True
        with self.local.lock:
            # Register EVERY waiter before taking the local fast path; otherwise
            # a new process could repeatedly jump ahead of local waiting threads.
            if self.path is not None and self.local.active >= cap:
                return False
            lock_path = self.directory / "queue.lock"
            outcome = {}
            fd = platform_layer.acquire_exclusive_file_lock(
                lock_path, timeout_sec=0, owner_aware_stale=True, outcome=outcome)
            if fd is None:
                if outcome.get("reason") == "contention":
                    return False
                raise ModelSlotUnavailable("Provider capacity queue could not be locked")
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
    if _managed(route) or not enabled():
        yield
        return
    waiter = _Waiter(model, use_local, deadline_ts, route, cancel_event, check)
    try:
        try:
            while not waiter.step():
                delay = min(_POLL_SEC, waiter.remaining())
                if cancel_event is not None:
                    cancel_event.wait(delay)
                else:
                    time.sleep(delay)
        except ModelSlotUnavailable:
            raise
        except OSError as error:
            raise ModelSlotUnavailable("Provider capacity storage unavailable") from error
        yield
    finally:
        waiter.close()


@contextlib.asynccontextmanager
async def model_call_slot_async(model: str, use_local: bool = False, deadline_ts: float | None = None,
                                *, route: dict | None = None, check: Callable | None = None):
    """Same queue with cancellable async waits, with no orphan acquiring thread."""
    if _managed(route) or not enabled():
        yield
        return
    waiter = _Waiter(model, use_local, deadline_ts, route, None, check)
    try:
        try:
            while not waiter.step():
                await asyncio.sleep(min(_POLL_SEC, waiter.remaining()))
        except ModelSlotUnavailable:
            raise
        except OSError as error:
            raise ModelSlotUnavailable("Provider capacity storage unavailable") from error
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

    def dispatch():
        with model_call_slot(model, route=route, check=require_physical_dispatch_window):
            require_physical_dispatch_window()
            return send()

    async def dispatch_async():
        async with model_call_slot_async(model, route=route, check=require_physical_dispatch_window):
            require_physical_dispatch_window()
            return await send()

    return dispatch_async if inspect.iscoroutinefunction(send) else dispatch
