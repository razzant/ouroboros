"""The own clock's two facts beside the alarm: a time the mind named for its next wake, and my human's last
reported time zone.

The named time is one overwritten state record (``WAKE_TARGET_STATE_KEY``) that ``set_next_wakeup(at=…)``
writes through ``store_choice``: the instant as given (ISO 8601 with an explicit UTC offset or ``Z``; ``named_time``
refuses a naive, unreadable or past one and never reads it as UTC), when it was stored, and the appointment that
stood then. ``seconds``, the rhythm, leaves it standing; a later ``at`` replaces it; nothing withdraws it. The alarm
(``consciousness.BackgroundConsciousness._next_delay``) lets it shorten a delay only while it lies strictly ahead,
never below MIN nor above MAX after a wake's end, so a time beyond MAX is reached through intermediate wakes. The
settlement of a wake appointed after it was stored consumes it once that wake ends at or after it, success or
failure (``consumed_by``), decided inside the settlement's own state write on the record it holds, so a newer ``at``
written meanwhile stays and sets the next wake. The wake that stood appointed when it was stored, pending or running,
never consumes it: each named time is consumed by one wake that started with it stored, and a passed one never adds
a wake. With consciousness on, a time at or before the pending appointment is refused: that wake keeps its time
(owner 12.10, #1603), and ``schedule_followup(run_at=…)`` serves work at that moment. Turning consciousness on
(``consciousness.start``) appoints the earlier of the next attempt Off left (its appointment, or the retry a skip or
refused admission moved it to) and a named time still ahead, no sooner than MIN after the last wake, boot or skip.

My human's last reported zone (``OWNER_ZONE_STATE_KEY``) is overwritten at owner ingress (``note_owner_zone``) from
their own message whose client reported a valid IANA zone, dated by that message's own time, never over a report
dated later; any other message leaves the dated fact as it is.

``own_clock_lines`` renders both for a wake with its appointment and the owner's bounds: facts with their meanings,
no instruction and no cause beyond them (P5). They are the "Own clock" standing section
(``consciousness_standing``) of every wake prepared while either stands, from the state the alarm read when it
admitted that wake (``consciousness._launch``).
"""

from __future__ import annotations

import datetime as _dt
import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ouroboros.deadline_utils import naive_instant, parse_deadline_ts, span_text, utc_offset_text

WAKE_TARGET_STATE_KEY = "consciousness_wake_target"  # {at, stored_at, standing_appointment}; one overwritten record
OWNER_ZONE_STATE_KEY = "owner_last_reported_zone"  # {zone, at, message}; overwritten by my human's zoned messages
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
    """Raised inside the state write for a named time at or before the pending appointment: nothing is written.
    ``next_attempt``: the attempt the clock last stored (a skip or refused admission may have moved it later)."""

    def __init__(self, appointment_at: float, next_attempt: float = 0.0) -> None:
        super().__init__(appointment_at)
        self.appointment_at, self.next_attempt = appointment_at, next_attempt


def store_choice(*, interval: Optional[int], at: str, target: Optional[_dt.datetime], now: float) -> Dict[str, Any]:
    """Store the rhythm and/or the named time in one state write and return what the receipt names: whether
    consciousness is known on, a running wake, the pending appointment and the next attempt the clock last stored.
    Raises ``PendingWakeKeepsItsTime`` or ``StateUnavailable`` when the choice was not written; when only the backup
    copy of this write's own save failed (``primary_written``) it stands, and ``unsaved`` is the receipt's note."""
    from ouroboros.consciousness import APPOINTMENT_STATE_KEY, INTERVAL_STATE_KEY, NEXT_WAKE_STATE_KEY, OWN_WAKE_STATE_KEY
    from supervisor.state import StateUnavailable, control_value, update_state

    seen: Dict[str, Any] = {}

    def write(st: Dict[str, Any]) -> None:
        known, enabled = control_value(st, "bg_consciousness_enabled")
        on = known and enabled is True
        appointment = _seconds(st.get(APPOINTMENT_STATE_KEY)) if on else 0.0
        if target is not None and appointment and target.timestamp() <= appointment:
            raise PendingWakeKeepsItsTime(appointment, _seconds(st.get(NEXT_WAKE_STATE_KEY)))
        if interval is not None:
            st[INTERVAL_STATE_KEY] = interval
        if target is not None:
            st[WAKE_TARGET_STATE_KEY] = {"at": at, "stored_at": _utc(now), "standing_appointment": appointment or None}
        own = st.get(OWN_WAKE_STATE_KEY) if isinstance(st.get(OWN_WAKE_STATE_KEY), dict) else {}
        seen.update(known=known, on=on, running=on and bool(own.get("task_id")) and not own.get("finished"),
                    appointment=appointment, next_attempt=_seconds(st.get(NEXT_WAKE_STATE_KEY)) if on else 0.0,
                    record=st.get(WAKE_TARGET_STATE_KEY))

    try:
        update_state(write)
    except StateUnavailable as exc:
        # ``seen`` is filled once the write ran, and only its final save follows: a restore of a missing primary
        # before it (``state._state_for_write``) can fail on its backup with nothing of this choice written.
        if not (exc.primary_written and seen):
            raise
        from ouroboros.consciousness import _BACKUP_UNSAVED  # the words Start and Stop say for the same partial write

        seen["unsaved"] = _BACKUP_UNSAVED.format(exc.reason)
    return seen


def _moved(appointment: float, attempt: Any) -> str:
    """The clause naming a next attempt a skip or refused admission moved away from the appointment, else ""."""
    return (f" (a skip or refused admission moved its next attempt to {_utc(attempt)})"
            if attempt and attempt != appointment else "")


def pending_refusal(at: str, appointment_at: float, next_attempt: float = 0.0) -> str:
    return (f"⚠️ TOOL_ARG_ERROR (set_next_wakeup): at={at!r} is at or before the wake-up already appointed for "
            f"{_utc(appointment_at)}, which keeps its time{_moved(appointment_at, next_attempt)}; nothing was stored. "
            "To have work done at that moment, use schedule_followup(run_at=…); a later at applies once that wake-up "
            "ends.")


def choice_receipt(seen: Dict[str, Any], *, interval: Optional[int], clamp_note: str, at: str,
                   target: Optional[_dt.datetime], low: int, high: int, now: float) -> str:
    """What the write set, and nothing the alarm will not do: the stored time, MIN and MAX, the pending appointment
    and, when a skip or refusal moved it, the next attempt. A named time sets a wake only while it is still ahead at
    a wake's end (``consciousness._next_delay``), so a time a wake-up outlives is said to add none."""
    known, on, record = seen.get("known"), seen.get("on"), seen.get("record")
    state = "off" if known else "unknown (runtime state is recovering)"
    if target is None:  # the rhythm alone: today's receipt, and the named time it leaves standing
        text = (f"OK: the wake-up interval is now {interval} s{clamp_note}; it applies from the end of the next wake-up "
                "(a wake-up already pending keeps its time; a wake-up calling this sets its own next one)." if on else
                f"OK: consciousness is {state}; the next wake-up interval of {interval} s{clamp_note} is stored for "
                "when it is enabled.")
        stored = target_ts(record)
        if stored is None:
            return text
        return text + (f" The named time {record['at']} still stands: from a wake-up's end the next one is appointed "
                       "at the earlier of the two while it is still ahead." if stored > now else
                       f" The named time {record['at']} has passed and adds no wake-up; a wake-up appointed after it "
                       "was stored removes it when it ends.")
    text = f"OK: the named next wake-up time is now {at} ({target.strftime('%Y-%m-%dT%H:%M:%SZ')})."
    if not on:
        text += f" Consciousness is {state}: it is stored for when it is enabled."
    elif seen.get("running"):
        text += " A wake-up is running; what follows applies from its end."
    elif seen.get("appointment"):
        text += (f" The wake-up appointed for {_utc(seen['appointment'])} keeps its time and does not consume it"
                 f"{_moved(seen['appointment'], seen.get('next_attempt'))}.")
    text += (f" From a wake-up's end the next one is appointed at the earlier of the interval and this time while it "
             f"is still ahead, no sooner than MIN ({low} s) and no later than MAX ({high} s) after that end: a time "
             "beyond MAX is reached through intermediate wake-ups, and one a wake-up outlives adds none. A wake-up "
             "appointed after it was stored that ends at or after it removes it; seconds leaves it standing, a later at "
             "replaces it.")
    return text + (f" The wake-up interval is now {interval} s{clamp_note}." if interval is not None else "")


def note_owner_zone(live: Dict[str, Any], metadata: Any, *, own: bool, message: Any, at: str, accepted: Any = None) -> None:
    """At owner ingress, inside the write that stamps their activity: my human's own message (``own``) whose client
    reported a valid IANA zone overwrites the one dated fact, dated by the ``ts`` its reference carries (a message
    accepted earlier and processed later keeps when it came), so it never replaces a report dated later; a reference
    without one is dated ``at``. The zone is the surface its accepted row (``accepted``) recorded, since a same-id
    retry carries the first acceptance's row beside its own metadata; a row with none (named external ingress,
    ``message_ingress.accept_local_message``) leaves the metadata's. Any other message leaves the fact as it is."""
    from ouroboros.client_surface import normalize_timezone_name

    carried = [source.get("client_surface") for source in (accepted, metadata) if isinstance(source, dict)]
    surface = next((item for item in carried if isinstance(item, dict) and item), None) if own else None
    zone = normalize_timezone_name(surface.get("timezone")) if isinstance(surface, dict) else ""
    if not zone:
        return
    sent = parse_deadline_ts(message.get("ts")) if isinstance(message, dict) else None
    if sent is not None:
        held = live.get(OWNER_ZONE_STATE_KEY)
        reported = parse_deadline_ts(held.get("at")) if isinstance(held, dict) else None
        if reported is not None and reported > sent:
            return
        at = str(message["ts"])
    live[OWNER_ZONE_STATE_KEY] = {"zone": zone, "at": at,
                                  **({"message": dict(message)} if isinstance(message, dict) else {})}


def _zone_name(fact: Any) -> str:
    from ouroboros.client_surface import normalize_timezone_name

    return normalize_timezone_name(fact.get("zone")) if isinstance(fact, dict) else ""


def _zone_line(fact: Any, *, now: float) -> str:
    from zoneinfo import ZoneInfo

    zone = _zone_name(fact)
    if not zone:
        return "- my human's last reported time zone: unknown"
    offset = utc_offset_text(_dt.datetime.fromtimestamp(now, ZoneInfo(zone)))
    reported = parse_deadline_ts(fact.get("at"))
    at = reported.strftime("%Y-%m-%dT%H:%M:%SZ") if reported is not None else str(fact.get("at"))
    return f"- my human's last reported time zone: {zone} ({offset} now), from their message at {at}"


def own_clock_lines(state: Mapping[str, Any], *, now: float, floor: int, ceiling: int) -> List[str]:
    """The "Own clock" facts of a wake from the runtime state it was prepared with: the named time if one stands,
    how far ahead and whether this wake's end consumes it; this wake's appointment (the reason line says whether it
    came on time); the owner's MIN and MAX; my human's last reported zone with its UTC offset now."""
    from ouroboros.consciousness import APPOINTMENT_STATE_KEY

    appointment = _seconds(state.get(APPOINTMENT_STATE_KEY))
    record = state.get(WAKE_TARGET_STATE_KEY)
    target = target_ts(record)
    if target is None:
        lines = ["- named next wake-up time: none; the interval alone sets the next wake-up"]
    else:
        when = f"in {span_text(target - now)}" if target > now else f"passed {span_text(now - target)} ago"
        stood = bool(appointment) and _seconds(record.get("standing_appointment")) == appointment
        lines = [f"- named next wake-up time: {record['at']}, {when}; the settlement of a wake-up appointed after it was "
                 "stored consumes it once that wake-up ends at or after it"
                 + ("; this wake was appointed before it was stored, so its end does not" if stood else "")]
    if appointment:
        lines.append(f"- this wake's appointment: {_utc(appointment)} (whether it came on time: the reason line)")
    lines.append(f"- each next wake-up is appointed no sooner than MIN {floor} s and no later than MAX {ceiling} s after "
                 "a wake-up ends (owner settings); a named time beyond MAX is reached through intermediate wake-ups")
    lines.append(_zone_line(state.get(OWNER_ZONE_STATE_KEY), now=now))
    return lines


def has_own_clock_fact(state: Mapping[str, Any]) -> bool:
    """Whether ``own_clock_lines`` has a fact of its own to say: a named time stands (ahead or passed), or my human
    reported a zone. Its "none" and "unknown" lines, this wake's appointment (the reason line dates it) and the
    owner's MIN and MAX alone are not one, so a wake with neither shows no "Own clock" section."""
    return (target_ts(state.get(WAKE_TARGET_STATE_KEY)) is not None
            or bool(_zone_name(state.get(OWNER_ZONE_STATE_KEY))))
