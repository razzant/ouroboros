"""The own clock's named time beside the alarm: a time the mind named for its next wake.

The named time is one overwritten state record (``WAKE_TARGET_STATE_KEY``) that ``set_next_wakeup(at=…)``
writes through ``store_choice``: the instant as given (ISO 8601 with an explicit UTC offset or ``Z``; ``named_time``
refuses a naive, unreadable or past one and never reads it as UTC), when it was stored, and the appointment that
stood then. ``seconds``, the rhythm, leaves it standing; a later ``at`` replaces it; nothing withdraws it. The alarm
(``consciousness.BackgroundConsciousness._next_delay``) lets it shorten a delay only while it lies strictly ahead,
never below MIN nor above MAX after a wake's end, so a time beyond MAX is reached through intermediate wakes. The
settlement of a wake appointed after it was stored consumes it once that wake ends at or after it, success or
failure (``consumed_by``), by compare-and-delete inside the settlement's own state write, so a newer ``at`` written
meanwhile stays. The wake that stood appointed when it was stored, pending or running, never consumes it: each named
time is consumed by one wake that started with it stored, and a passed one never adds a wake. With consciousness on,
a time at or before the pending appointment is refused: that wake keeps its time (owner 12.10, #1603), and
``schedule_followup(run_at=…)`` serves work at that moment.
"""

from __future__ import annotations

import datetime as _dt
import math
from typing import Any, Dict, Optional, Tuple

from ouroboros.deadline_utils import naive_instant, parse_deadline_ts

WAKE_TARGET_STATE_KEY = "consciousness_wake_target"  # {at, stored_at, standing_appointment}; one overwritten record
_EXAMPLE = "2026-11-21T09:00:00+03:00"


def _utc(ts: float) -> str:
    return _dt.datetime.fromtimestamp(float(ts), tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _seconds(value: Any) -> float:
    """A stored Unix time, 0.0 when absent or not one."""
    try:
        number = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) and number > 0 else 0.0


def named_time(raw: Any, *, now: float) -> Tuple[Optional[_dt.datetime], str]:
    """``(instant, "")`` for an ISO 8601 time with an explicit UTC offset or ``Z`` strictly after ``now``; else
    ``(None, why)``. A time without an offset is refused, never read as UTC as ``parse_deadline_ts`` would."""
    text = raw.strip() if isinstance(raw, str) else ""
    naive = bool(text) and naive_instant(text)
    instant = None if naive else parse_deadline_ts(text)
    if instant is None:
        problem = "carries no UTC offset" if naive else "is not an ISO 8601 time"
        return None, f"at={raw!r} {problem}; give the time with its offset, e.g. {_EXAMPLE}"
    if instant.timestamp() <= now:
        return None, f"at={raw!r} is not in the future (now {_utc(now)})"
    return instant, ""


def target_ts(record: Any) -> Optional[float]:
    """The stored named time in Unix seconds; None when none stands."""
    instant = parse_deadline_ts(record.get("at")) if isinstance(record, dict) else None
    return instant.timestamp() if instant is not None else None


def consumed_by(record: Any, *, appointment_at: float, now: float) -> bool:
    """Whether the settlement at ``now`` of the wake appointed for ``appointment_at`` consumes the named time:
    that wake ends at or after it and is not the one that stood appointed when it was stored."""
    target = target_ts(record)
    if target is None or now < target:
        return False
    standing = _seconds(record.get("standing_appointment"))
    return not standing or standing != _seconds(appointment_at)


class PendingWakeKeepsItsTime(Exception):
    """Raised inside the state write for a named time at or before the pending appointment: nothing is written."""

    def __init__(self, appointment_at: float) -> None:
        super().__init__(appointment_at)
        self.appointment_at = appointment_at


def store_choice(*, interval: Optional[int], at: str, target: Optional[_dt.datetime], now: float) -> Dict[str, Any]:
    """Store the rhythm and/or the named time in one state write and return what the receipt names: whether
    consciousness is known on, a running wake, the pending appointment and the next attempt the clock last stored.
    Raises ``PendingWakeKeepsItsTime`` (nothing written) or ``StateUnavailable``."""
    from ouroboros.consciousness import APPOINTMENT_STATE_KEY, INTERVAL_STATE_KEY, NEXT_WAKE_STATE_KEY, OWN_WAKE_STATE_KEY
    from supervisor.state import control_value, update_state

    seen: Dict[str, Any] = {}

    def write(st: Dict[str, Any]) -> None:
        known, enabled = control_value(st, "bg_consciousness_enabled")
        on = known and enabled is True
        appointment = _seconds(st.get(APPOINTMENT_STATE_KEY)) if on else 0.0
        if target is not None and appointment and target.timestamp() <= appointment:
            raise PendingWakeKeepsItsTime(appointment)
        if interval is not None:
            st[INTERVAL_STATE_KEY] = interval
        if target is not None:
            st[WAKE_TARGET_STATE_KEY] = {"at": at, "stored_at": _utc(now), "standing_appointment": appointment or None}
        own = st.get(OWN_WAKE_STATE_KEY) if isinstance(st.get(OWN_WAKE_STATE_KEY), dict) else {}
        seen.update(known=known, on=on, running=on and bool(own.get("task_id")) and not own.get("finished"),
                    appointment=appointment, next_attempt=_seconds(st.get(NEXT_WAKE_STATE_KEY)) if on else 0.0,
                    record=st.get(WAKE_TARGET_STATE_KEY))

    update_state(write)
    return seen


def pending_refusal(at: str, appointment_at: float) -> str:
    return (f"⚠️ TOOL_ARG_ERROR (set_next_wakeup): at={at!r} is at or before the wake-up already appointed for "
            f"{_utc(appointment_at)}, which keeps its time; nothing was stored. To have work done at that moment, "
            "use schedule_followup(run_at=…); a later at applies once that wake-up ends.")


def choice_receipt(seen: Dict[str, Any], *, interval: Optional[int], clamp_note: str, at: str,
                   target: Optional[_dt.datetime], low: int, high: int) -> str:
    """What the write set, and nothing the alarm will not do: the stored time, MIN and MAX, the pending appointment
    and, when a skip or refusal moved it, the next attempt."""
    known, on, record = seen.get("known"), seen.get("on"), seen.get("record")
    state = "off" if known else "unknown (runtime state is recovering)"
    if target is None:  # the rhythm alone: today's receipt, and the named time it leaves standing
        text = (f"OK: the wake-up interval is now {interval} s{clamp_note}; it applies from the end of the next wake-up "
                "(a wake-up already pending keeps its time; a wake-up calling this sets its own next one)." if on else
                f"OK: consciousness is {state}; the next wake-up interval of {interval} s{clamp_note} is stored for "
                "when it is enabled.")
        standing = record.get("at") if isinstance(record, dict) and target_ts(record) is not None else ""
        return text + (f" The named time {standing} still stands: the next wake-up comes at the earlier of the two."
                       if standing else "")
    text = f"OK: the named next wake-up time is now {at} ({target.strftime('%Y-%m-%dT%H:%M:%SZ')})."
    if not on:
        text += f" Consciousness is {state}: it is stored for when it is enabled."
    elif seen.get("running"):
        text += " A wake-up is running: from its end the next one comes at the earlier of the interval and this time."
    elif seen.get("appointment"):
        moved, appointment = seen.get("next_attempt"), seen["appointment"]
        text += (f" The wake-up appointed for {_utc(appointment)} keeps its time and does not consume it"
                 + (f" (a skip or refused admission moved its next attempt to {_utc(moved)})"
                    if moved and moved != appointment else "") + ".")
    text += (f" Each next wake-up comes no sooner than MIN ({low} s) and no later than MAX ({high} s) after a "
             "wake-up ends, so a time beyond MAX is reached through intermediate wake-ups; the first wake-up appointed "
             "after now that ends at or after it consumes it. seconds leaves it standing; a later at replaces it.")
    return text + (f" The wake-up interval is now {interval} s{clamp_note}." if interval is not None else "")

