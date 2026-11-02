"""Small shared helpers for deadline-aware task behavior."""

from __future__ import annotations

from datetime import datetime, timezone
import contextlib
import functools
import inspect
import logging
import math
import os
import time
from typing import Any, Optional

log = logging.getLogger(__name__)


def caller_deadline_scoped(function):
    """Carry caller-owned absolute bounds through preparation and all recovery sends.

    The caller subtracts its finalization reserve once. Relative socket timeouts
    never create an execution deadline; absent bounds preserve existing defaults.
    """
    signature = inspect.signature(function)

    @contextlib.contextmanager
    def scope(args, kwargs):
        from ouroboros.model_wait import calendar_scope, execution_deadline_scope

        values = signature.bind(*args, **kwargs).arguments
        with contextlib.ExitStack() as stack:
            calendar = values.get("caller_deadline_ts")
            execution = values.get("caller_execution_deadline")
            if calendar is not None:
                stack.enter_context(calendar_scope(datetime.fromtimestamp(
                    float(calendar), timezone.utc).isoformat()))
            if execution is not None:
                stack.enter_context(execution_deadline_scope(float(execution)))
            yield

    if inspect.iscoroutinefunction(function):
        @functools.wraps(function)
        async def asynchronous(*args, **kwargs):
            with scope(args, kwargs):
                return await function(*args, **kwargs)
        return asynchronous

    @functools.wraps(function)
    def synchronous(*args, **kwargs):
        with scope(args, kwargs):
            return function(*args, **kwargs)
    return synchronous


def physical_dispatch_timeout(explicit: Any = None) -> Any:
    """Narrow every socket phase at dispatch, keeping unset transport defaults.

    Phase timeouts are not a total wall-clock bound. The existing caller retains
    custody if an already-dispatched request outlives its logical wait.
    """
    from ouroboros.llm_attempt import require_physical_dispatch_window

    remaining = require_physical_dispatch_window()
    if remaining is None:
        return explicit
    import httpx

    if isinstance(explicit, httpx.Timeout):
        return httpx.Timeout(**{
            phase: remaining if value is None else min(value, remaining)
            for phase, value in explicit.as_dict().items()
        })
    return min(llm_transport_timeout_sec(explicit), remaining)


def caller_deadline_arguments(deadline_at: Any, execution_deadline: Any, *, reserve_sec: float = 0.0) -> dict:
    """Project an existing caller's two clocks into the optional LLM arguments."""
    deadline = parse_deadline_ts(deadline_at)
    values = {}
    if deadline is not None:
        values["caller_deadline_ts"] = deadline.timestamp() - reserve_sec
    if execution_deadline is not None:
        values["caller_execution_deadline"] = execution_deadline
    return values


def parse_deadline_ts(value: Any) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        parsed = datetime.fromisoformat(raw)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def seconds_until(value: Any) -> Optional[float]:
    """Non-negative wall-clock seconds until an ISO instant; None when unparsable."""
    parsed = parse_deadline_ts(value)
    if parsed is None:
        return None
    return max(0.0, (parsed - utc_now()).total_seconds())


def deadline_remaining_sec(ctx: Any) -> float:
    meta = getattr(ctx, "task_metadata", {})
    if not isinstance(meta, dict):
        return 0.0
    deadline = parse_deadline_ts(meta.get("deadline_at"))
    return (deadline - utc_now()).total_seconds() if deadline is not None else 0.0


def has_deadline(ctx: Any) -> bool:
    """Whether a deadline EXISTS, which the remaining seconds alone cannot answer.

    ``deadline_remaining_sec`` returns 0.0 both for "no deadline set" and for one that
    expires exactly now, and goes negative once it is spent — so a caller deciding
    whether to bound itself has to ask this separately or it will read a spent deadline
    (negative) and a sub-second one (0.x truncated to 0) as "unbounded".
    """
    meta = getattr(ctx, "task_metadata", {})
    if not isinstance(meta, dict):
        return False
    return parse_deadline_ts(meta.get("deadline_at")) is not None


def window_within_deadline(ctx: Any, requested: int) -> int:
    """``requested`` seconds, narrowed so a held window cannot outlive the deadline.

    NARROW-ONLY, and keyed on EXISTENCE rather than sign: `int(remaining) > 0` let a
    wait hold its whole window past a deadline in two shapes — a sub-second remainder
    truncated to 0, and a spent deadline, whose remainder is negative. Only "no deadline
    set" takes the full ask; both spent shapes land on the floor. The finalization GRACE
    is subtracted for the reason the network tools subtract it: targeting the whole
    remaining deadline returns at the instant there is no time left to answer at all.
    """
    if not has_deadline(ctx):
        return max(1, int(requested))
    from ouroboros.task_pacing import effective_finalization_reserve_sec

    remaining = float(deadline_remaining_sec(ctx) or 0.0)
    return max(1, int(min(requested, remaining - float(effective_finalization_reserve_sec(ctx)))))


def deadline_clamped_timeout_sec(ctx: Any, base_timeout: Any) -> Any:
    """``min(base, remaining − finalization reserve)`` for a wait that must return before its task.

    ``remaining`` is the nearer explicit bound: ``deadline_at`` or the explicit execution window
    (#939); with neither, or one already elapsed (forced finalization owns that), ``base`` is
    returned unchanged. A ``None`` base (no window of its own) takes the clamped remainder. The
    clamp never floors PAST the reserve: inside it the wait gets 1 s, a fast typed failure, not a
    grace-eating floor (review round 1). The plain grace, not a profile's percentage reserve
    (v6.55.0): clamping out the last pct-of-total strands long tasks idle. One owner for the
    network tools' outer waits (``loop_tool_execution``) and any command resolution that adopts it.
    """
    try:
        remaining = deadline_remaining_sec(ctx)
        from ouroboros.model_wait import explicit_execution_window_remaining
        execution_left = explicit_execution_window_remaining(ctx)  # #939
        if execution_left is not None:
            remaining = min(remaining, execution_left) if remaining > 0 else execution_left
    except Exception:
        return base_timeout
    if remaining <= 0:
        return base_timeout
    try:
        from ouroboros.task_pacing import effective_finalization_reserve_sec

        reserve = effective_finalization_reserve_sec(ctx)
    except Exception:
        from ouroboros.config import get_finalization_grace_sec

        reserve = float(get_finalization_grace_sec())
    window = remaining - reserve
    if base_timeout is not None and window >= base_timeout:
        return base_timeout
    return int(max(1.0, window if base_timeout is None else min(float(base_timeout), window)))


def llm_transport_timeout_sec(explicit: Any = None) -> float:
    """Return the transport/dead-socket timeout for one LLM request.

    A caller may narrow this value explicitly.  The shared default is deliberately
    read from ``config.py`` and is never used as a logical cognition deadline.
    """
    try:
        requested = float(explicit)
    except (TypeError, ValueError):
        requested = 0.0
    if requested > 0:
        return requested
    from ouroboros.config import get_llm_transport_read_timeout_sec

    return float(get_llm_transport_read_timeout_sec())


def dispatch_window_remaining_sec(
    *,
    deadline_at: Any = None,
    deadline_ts: Any = None,
    reserve_sec: Any = 0.0,
) -> Optional[float]:
    """Return owner time available for a *new* physical dispatch.

    ``None`` means that no owner deadline is present.  ``0.0`` means the
    deadline or its finalization reserve is spent.  This is deliberately
    separate from :func:`transport_timeout_with_deadline`, whose positive
    floor keeps an already-admitted transport call bounded without pretending
    that a new call is still admissible.
    """
    remaining: Optional[float] = None
    try:
        if isinstance(deadline_ts, (int, float, str)) and not isinstance(deadline_ts, bool):
            candidate = float(deadline_ts) - time.time()
            if math.isfinite(candidate):
                remaining = candidate
    except (TypeError, ValueError, OverflowError):
        remaining = None
    if remaining is None:
        remaining = seconds_until(deadline_at)
    if remaining is None:
        return None
    try:
        reserve = max(0.0, float(reserve_sec or 0.0))
    except (TypeError, ValueError, OverflowError):
        reserve = 0.0
    return max(0.0, remaining - reserve)


def owner_deadline_exhausted(
    *, deadline_at: Any = None, deadline_ts: Any = None, reserve_sec: Any = 0.0,
) -> bool:
    """Whether the owner window, optionally minus a settlement reserve, is spent."""
    remaining = dispatch_window_remaining_sec(
        deadline_at=deadline_at, deadline_ts=deadline_ts, reserve_sec=reserve_sec,
    )
    return remaining is not None and remaining <= 0


def owner_deadline_exhausted_for_context(ctx: Any, *, reserve_sec: Any = 0.0) -> bool:
    """Apply the same admission rule to a task context's ISO or epoch deadline."""
    metadata = getattr(ctx, "task_metadata", {})
    deadline_at = metadata.get("deadline_at") if isinstance(metadata, dict) else None
    return owner_deadline_exhausted(
        deadline_at=deadline_at, deadline_ts=getattr(ctx, "deadline_ts", None), reserve_sec=reserve_sec,
    )


def transport_timeout_with_deadline(
    explicit: Any = None,
    *,
    deadline_at: Any = None,
    deadline_ts: Any = None,
    reserve_sec: Any = 0.0,
) -> float:
    """Narrow a transport read/process timeout to the owner's remaining time.

    This is deliberately a transport bound, not a logical-operation deadline:
    absent an owner deadline it returns the configured dead-socket timeout, while
    an existing deadline can only shorten it. Numeric epoch deadlines are useful
    for in-process loop contexts; ISO deadlines remain the public task contract.
    """
    base = llm_transport_timeout_sec(explicit)
    remaining = dispatch_window_remaining_sec(
        deadline_at=deadline_at, deadline_ts=deadline_ts, reserve_sec=reserve_sec,
    )
    if remaining is None:
        return base
    return max(0.001, min(base, remaining))


def bounded_seconds(value: Any, *, default: float, maximum: float) -> int:
    """Encode a positive engine horizon without widening an exhausted one."""
    if value is None or (isinstance(value, str) and not value.strip()):
        raw = float(default)
    else:
        try:
            raw = float(value)
        except (TypeError, ValueError):
            raw = float(default)
    if not math.isfinite(raw):
        raw = float(default)
    if raw <= 0:
        return 1
    return max(1, min(math.ceil(raw), int(maximum)))


def logical_operation_timeout_sec(
    explicit: Any = None,
    *,
    deadline_at: Any = None,
    fallback: Any = None,
    reserve_sec: Any = 0.0,
) -> float:
    """Choose a logical operation wait without borrowing a socket timeout.

    An explicit slot/task value is the operation's requested window, but an
    existing owner deadline always narrows it.  Absent both, ``fallback`` is
    used (normally the transport bound only as a settlement wait, not as a
    cognition target).
    """
    # ``None``/blank means "use the route fallback"; an explicit zero is a
    # caller-owned one-second horizon, matching ``bounded_seconds``. Treating
    # both shapes as the same sentinel silently widened a zero slot to the
    # 2700-second transport fallback.
    explicit_set = not (
        explicit is None or (isinstance(explicit, str) and not explicit.strip())
    )
    try:
        requested = float(explicit) if explicit_set else 0.0
    except (TypeError, ValueError):
        explicit_set = False
        requested = 0.0
    if explicit_set and not math.isfinite(requested):
        explicit_set = False
    remaining = seconds_until(deadline_at)
    if remaining is not None:
        try:
            reserve = max(0.0, float(reserve_sec or 0.0))
        except (TypeError, ValueError):
            reserve = 0.0
        remaining = max(0.0, remaining - reserve)
        if remaining <= 0:
            return 0.0
        if explicit_set:
            requested = 1.0 if requested <= 0 else requested
            return min(requested, remaining)
        return remaining
    if explicit_set:
        return 1.0 if requested <= 0 else requested
    try:
        value = float(fallback)
    except (TypeError, ValueError):
        value = llm_transport_timeout_sec()
    return max(0.001, value)


def review_logical_fallback_timeout_sec() -> Optional[float]:
    """Return an explicit review-window override, else leave routing in charge."""
    raw = os.environ.get("OUROBOROS_REVIEW_MODEL_TIMEOUT_SEC", "")
    if not raw.strip():
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 0.0
    if math.isfinite(value) and value > 0:
        return value
    log.warning(
        "Invalid or non-positive OUROBOROS_REVIEW_MODEL_TIMEOUT_SEC=%r; "
        "using the route-owned logical fallback",
        raw,
    )
    return None


def task_execution_window_sec() -> Optional[float]:
    """Execution time an operation of the current task may still use: the bound task's
    remaining window (its explicit budget or the global lifetime, the narrower, pauses
    excluded), else the global lifetime; ``None`` when neither exists, never an invented one."""
    from ouroboros.config import get_task_abs_ceiling_sec
    from ouroboros.model_wait import current_model_wait

    owner = current_model_wait()
    remaining = owner.execution_window_remaining() if owner is not None else None
    if remaining is not None:
        return remaining
    lifetime = get_task_abs_ceiling_sec()
    return None if lifetime is None else float(lifetime)


def review_operation_timeout_sec(
    explicit: Any = None,
    *,
    retrieves: bool = False,
    deadline_at: Any = None,
    transport_timeout_sec: Any = None,
    reserve_sec: Any = 0.0,
) -> Optional[float]:
    """Resolve a review slot's logical window without turning transport into cognition policy.

    A packet row (``retrieves`` false, ``review_execution.delivery_retrieves``) is ONE physical
    request, which ends at its dead-socket bound, so that bound is its settlement fallback. A
    retrieving row (an agent session, a native episode) sends many requests, each keeping its
    own socket bound; its window is only what is true: an explicit window, the owner deadline
    less the reserve, the task's finite execution window. ``None`` = none exists; a session
    then sends no ``maxSeconds``.
    """
    requested = explicit
    if explicit is None or (isinstance(explicit, str) and not explicit.strip()):
        requested = review_logical_fallback_timeout_sec()
    if not retrieves:
        return logical_operation_timeout_sec(
            requested,
            deadline_at=deadline_at,
            fallback=llm_transport_timeout_sec(transport_timeout_sec),
            reserve_sec=reserve_sec,
        )
    # ``inf`` is the "no window" fallback of the shared parser, returned as ``None``.
    window = logical_operation_timeout_sec(requested, deadline_at=deadline_at, fallback=math.inf,
                                           reserve_sec=reserve_sec)
    execution = task_execution_window_sec()
    if execution is not None:
        window = min(window, execution)
    return None if math.isinf(window) else window


def main_transport_timeout_sec(
    model: str, deadline_ts: Optional[float], *, reserve_sec: Optional[float] = None,
) -> float:
    """Main's transport projection lives beside the corresponding review policy."""
    from ouroboros.provider_models import provider_for_model

    # Preserve the native Anthropic default while narrowing every route to an
    # owner deadline. Other routes use the shared dead-socket bound; local models
    # receive the same explicit bound their client already supports.
    explicit = 120 if provider_for_model(model) == "anthropic" else None
    # ``None`` is the low-level raw-deadline contract. The production round
    # dispatcher passes the finalization reserve explicitly; keeping this
    # default raw prevents admission from accepting a call and then shrinking
    # its transport to the 0.001-second floor unexpectedly.
    return transport_timeout_with_deadline(
        explicit, deadline_ts=deadline_ts,
        reserve_sec=0.0 if reserve_sec is None else reserve_sec,
    )


def review_transport_timeout(model: Any, explicit: Any = None, deadline_at: Any = None) -> Optional[float]:
    """Resolve one review route's physical timeout without hiding an owner deadline."""
    from ouroboros.config import get_finalization_grace_sec
    from ouroboros.provider_models import provider_for_model

    provider = provider_for_model(model)
    deadline = str(deadline_at or "").strip()
    if explicit is None and provider == "anthropic" and not deadline:
        return None
    native = 120 if provider == "anthropic" else None
    return transport_timeout_with_deadline(
        explicit if explicit is not None else native,
        deadline_at=deadline or None,
        reserve_sec=get_finalization_grace_sec(),
    )


def deadline_expired(ctx) -> bool:
    """True when the task HAS a deadline and it has already passed.

    The distinction ``deadline_remaining_sec`` alone cannot make: it answers
    0.0 both for "no deadline" and for "the deadline is behind us", and
    collapsing them let an EXPIRED nanny hand a fresh run the absolute task
    ceiling.
    """
    meta = getattr(ctx, "task_metadata", {})
    meta = meta if isinstance(meta, dict) else {}
    if parse_deadline_ts(meta.get("deadline_at")) is None:
        return False
    return deadline_remaining_sec(ctx) <= 0
