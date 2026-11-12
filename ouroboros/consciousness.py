"""The alarm clock of Background Consciousness: when Ouroboros wakes up on its own.

A wake-up is an ORDINARY Main direct turn nobody typed (owner decisions В13/В15, PLAN 5.1–5.3):
Main's prompt, memory, tools and loop, started through ``supervisor.workers.handle_wake_direct``
with the wake's envelope, the Main lane's own routing facts and rendered message. This module owns only
the clock — WHEN such a turn starts and what to say about it afterwards. No thread, no private
registry, no observation inbox, no pause/resume: the supervisor loop calls ``tick(now)`` once
per pass; ``DirectActivityRegistry`` and the ordinary pool census supply liveness.
Only its own wake (``wake_cause=own_clock``, known by id) blocks this clock: a wake runs beside
the owner's work in Main or any room and beside the focuses, which never hold it.
Tick order, each step a typed outcome: disabled → its own live wake → not yet due → the
rolling-24h allowance (an unreadable ledger is the disclosed skip
``allowance_unknown``, never a silent block; known spend at the daily limit is
``allowance_exhausted``) → an owner chat must be bound → launch. A skip keeps the window it met
(``allowance_fact``, the projection a wake's preparation carries) in its event and in the one
overwritten ``consciousness_last_skip`` state key, which the mind's status read shows. An exhausted
allowance waits for its window to free, a time chosen under the limit it met: a changed daily limit
retries it like any skip, MIN after the skip and after boot (``_limit_reopens_skip``; appointment untouched).
What is left of that allowance is the launched tree's producer ceiling
(``metadata.root_cost_ceiling_usd``, В26=A): the in-task stop pauses the wake when its known
spend reaches it — no extra planning margin is subtracted or skipped (owner 2026-10-07) — while the
ledger fence stays at the owner's per-task cap, so a wake never dies before its first call on a
nearly spent day. The next
wake is ``last finish + interval``: the MODEL's choice (``set_next_wakeup`` persists
``consciousness_next_interval_sec``) or ``WAKE_DEFAULT_SEC``, clamped into the owner's
[min, max]; a runner failure — or a wake the lane could not admit — doubles it up to max until a
wake succeeds. The persisted own actor id and cause fence completion; a carrier handoff
does not finish the logical wake, and repeated terminal callbacks do not advance it twice.
A wake a pending Restart refuses before its start stays a refusal, like every closed door:
no failure, no row, no owner and no observation (``handle_wake_direct``). A pooled terminal
no dispatcher reported settles only at the tick that would start over it.
Nothing else moves the alarm (owner 12 Oct, #1603: no accelerated wake, only the
scheduled time): a task finishing, a project digest or an orphan-heal sweep never pulls the next
wake earlier — they wait as observed facts for the scheduled wake (``observe_wake``), so every
wake's reason is the heartbeat. Neither does an owner message, nor a health event (``build_health_invariants``
is read-side; health is visible in the next wake's context).
Panic and ``/bg stop`` arrive through ``stop()``, which arms ordinary graceful cascade
cancellation for own wakes (pooled ones too), live/lent focuses and autonomous work, off-thread. Who stopped it is
saved beside the switch (``STOP_SOURCE_KEY``): the mind's tool never lifts the owner's, Panic's or an
unknown stop, after a restart or from a queued request alike.
The legacy observation inbox, if present, is moved once to the archive unread.
What a wake OBSERVES is not this clock's business: it starts after the chat-chain position and
the task-result transition state the last admitted wake observed
(``consciousness_observation_boundary``, persisted only once the lane admits the wake;
``consciousness_wake.observe_wake``), never at a finish time.
"""

from __future__ import annotations

import datetime as _dt
import logging
import math
import pathlib
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

from ouroboros.config import (
    WAKE_DEFAULT_SEC,
    get_bg_wakeup_max_sec,
    get_bg_wakeup_min_sec,
    get_consciousness_autonomy,
    get_consciousness_daily_usd,
    get_consciousness_max_tasks,
    runtime_setting,
)
from ouroboros.consciousness_allowance import STATUS_EXHAUSTED, STATUS_UNKNOWN, allowance_window
from ouroboros.consciousness_authority import is_consciousness_origin, is_own_clock_wake
from ouroboros.consciousness_wake import WAKE_OBSERVATION_KEY, wake_continuity, wake_continuity_after, wake_task_metadata
from ouroboros.deadline_utils import parse_deadline_ts
from ouroboros.utils import append_jsonl, utc_now_iso

log = logging.getLogger(__name__)

NEXT_WAKE_STATE_KEY = "consciousness_next_wake_at"
INTERVAL_STATE_KEY = "consciousness_next_interval_sec"
LAST_WAKE_STATE_KEY = "consciousness_last_wake_at"
APPOINTMENT_STATE_KEY = "consciousness_appointment_at"  # the scheduled time; skips and refusals move only the next attempt
LAST_SKIP_STATE_KEY = "consciousness_last_skip"  # the newest skipped wake with the allowance window it met; overwritten
OWN_WAKE_STATE_KEY = "consciousness_own_wake"  # the admitted clock actor, retained across carrier changes/restart
OBSERVATION_STATE_KEY = "consciousness_observation_boundary"  # chat position + transition state the last ACCEPTED wake observed
OBSERVATION_BOOTSTRAP_STATE_KEY = "consciousness_observation_bootstrap_since"
CONTINUITY_STATE_KEY = "consciousness_wake_continuity"  # notes the next wake carries when its predecessor's row cannot say
LEGACY_INBOX_REL = pathlib.Path("state") / "consciousness_observations.jsonl"
ARCHIVED_INBOX_REL = pathlib.Path("archive") / "consciousness_observations.jsonl"
HEARTBEAT = "heartbeat"
ALLOWANCE_VIEW_TTL_SEC = 60.0
SNAPSHOT_KEYS = ("enabled", "level", "next_wake_at", "last_wake_at", "last_wake_task_id",
                 "last_wake_outcome", "last_error", "spent_24h_usd", "daily_usd", "allowance_resets_at",
                 "tasks_running", "max_tasks", "live_wake_task_id", "unknown_unmetered", "integrity_degraded")


def _iso(ts: float) -> str:
    return _dt.datetime.fromtimestamp(float(ts), tz=_dt.timezone.utc).isoformat() if ts else ""


def model_route() -> Optional[Dict[str, Any]]:
    """The model the next wake thinks with, read from the settings as they are now (no network, no store), with
    the route's local flag: ``layer`` ``consciousness`` when its own slot decides the route (a model of its own, or
    Main's run locally when its box is checked under a cloud Main: ``route_choice.model_role_slot_override``), else
    ``main``, the route a wake follows (``following_route``). ``None`` when it cannot be read."""
    from ouroboros.route_choice import following_route, model_role_slot_override

    try:
        slot = model_role_slot_override({"model_role": "consciousness"})
        model, local = slot or following_route("consciousness")
    except Exception:  # every status read asks again; the unknown is the null itself, so the log stays quiet
        log.debug("The consciousness model route could not be read from the settings", exc_info=True)
        return None
    model, layer = str(model or "").strip(), "main" if slot is None else "consciousness"
    return {"value": model, "layer": layer, "local": bool(local)} if model else None


def allowance_fact(window: Dict[str, Any]) -> Dict[str, Any]:
    """The one projection of the allowance window a wake decision met: a wake's preparation and a skip's
    record carry the same keys (known spend, unmetered calls, the limit, spend by cause, the reset time);
    null is unknown, never 0."""
    return {**{key: window.get(key) for key in ("settled_usd", "unknown_unmetered", "limit_usd", "spend_by_cause")},
            "resets_at": window.get("resets_at") or None}


def _money_skip_basis(skip: Any, next_wake_at: float) -> Optional[Tuple[float, float, float]]:
    """``(the time it chose, the limit it met, when it skipped)`` when ``skip`` is an exhausted-allowance skip
    record that set this next wake: the facts ``_limit_reopens_skip`` compares the setting and the clock with."""
    if not isinstance(skip, dict) or skip.get("reason") != "allowance_exhausted":
        return None
    allowance = skip.get("allowance") if isinstance(skip.get("allowance"), dict) else {}
    limit, skipped = allowance.get("limit_usd"), parse_deadline_ts(skip.get("at"))
    try:  # a stored time that cannot be a date is no basis
        chose_this_wake = skip.get("next_wake_at") == _iso(next_wake_at)
    except (OverflowError, OSError, ValueError):
        return None
    if not chose_this_wake or type(limit) not in (int, float) or skipped is None:
        return None
    return float(next_wake_at), float(limit), skipped.timestamp()


def panic_blocks_wake(drive_root: Any) -> bool:
    """An unreceived Panic input bars automatic wake grants."""
    from ouroboros.installation_stop import installation_panic

    return installation_panic(drive_root)


# Who turned consciousness off, saved in the same state write as ``bg_consciousness_enabled`` (owner 11.11, 5A; the
# twin of /evolve off's provenance): the owner's doors (/bg from the app, Telegram or onboarding) and Panic leave a stop
# only the owner's start lifts, in every mode (P0: the owner's decision binds in Cyber Pro too); toggle_consciousness lifts only
# a stop it placed. A stop with no saved source (placed before this key, a recovered state) is unknown: owner's.
STOP_SOURCE_KEY = "bg_consciousness_stop_source"
OWNER_STOP, PANIC_STOP, AGENT_STOP = "owner", "panic", "agent_tool"
_STOPPED_BY = {OWNER_STOP: "the owner turned it off", PANIC_STOP: "Panic turned it off"}


class _StopHolds(Exception):
    """Raised inside the state write: the standing stop is not the mind's to lift, and nothing is written."""


def held_stop_text(source: Any) -> str:
    who = _STOPPED_BY.get(str(source or ""), "who turned it off is not recorded")
    return (f"Background consciousness stayed off: {who}, and that stop holds against toggle_consciousness. "
            "Only the owner's /bg start (Start in the app or in Telegram) turns it back on.")


# The primary copy written, its backup not (``StateUnavailable.primary_written``): the change stands and the next tick
# reads it, so the reply says it was saved and what it lacks, never that nothing changed.
_BACKUP_UNSAVED = " Saved, but the state's backup copy was not written ({})."


def _persist_start(source: str) -> Tuple[str, str]:
    """Save an enable; the mind's right to it is decided inside the same locked write. ``(refusal, note)``: why
    nothing changed, or what the saved enable lacks."""
    from supervisor.state import StateUnavailable, control_value, update_state

    def enable(live: Dict[str, Any]) -> None:
        known, enabled = control_value(live, "bg_consciousness_enabled")
        source_known, stopped_by = control_value(live, STOP_SOURCE_KEY)
        if source != OWNER_STOP and not (known and (enabled is True or (source_known and stopped_by == AGENT_STOP))):
            raise _StopHolds(stopped_by if known and source_known else "")
        live["bg_consciousness_enabled"] = True
        live.pop(STOP_SOURCE_KEY, None)

    try:
        update_state(enable, confirm=("bg_consciousness_enabled", STOP_SOURCE_KEY))
    except _StopHolds as held:
        return held_stop_text(held.args[0]), ""
    except StateUnavailable as exc:
        if exc.primary_written:
            return "", _BACKUP_UNSAVED.format(exc.reason)
        return (f"Background consciousness stayed off: runtime state is unavailable ({exc.reason}); "
                "nothing was started."), ""
    return "", ""


def _persist_stop(source: str) -> str:
    """Save a stop with who placed it: "" or the disclosure that it holds only in this process. The mind's stop never
    relabels a standing stop that is not its own (the owner's, Panic's or an unknown one)."""
    from supervisor.state import StateUnavailable, control_value, update_state

    def disable(live: Dict[str, Any]) -> None:
        known, enabled = control_value(live, "bg_consciousness_enabled")
        source_known, stopped_by = control_value(live, STOP_SOURCE_KEY)
        if source == AGENT_STOP and not (known and enabled is True) and not (source_known and stopped_by == AGENT_STOP):
            kept = stopped_by if source_known and stopped_by in _STOPPED_BY else ""  # "" stays unknown
        else:
            kept = source
        live["bg_consciousness_enabled"] = False
        if kept:
            live[STOP_SOURCE_KEY] = kept
        else:
            live.pop(STOP_SOURCE_KEY, None)

    try:
        update_state(disable, confirm=("bg_consciousness_enabled", STOP_SOURCE_KEY))
    except StateUnavailable as exc:
        if exc.primary_written:
            return _BACKUP_UNSAVED.format(exc.reason)
        return f" Not saved (runtime state unavailable, {exc.reason}): the stop holds in this process only."
    return ""


class BackgroundConsciousness:
    """The alarm clock; one instance per supervisor, ticked from its loop."""

    def __init__(self, drive_root: Any, repo_dir: Any, owner_chat_id_fn: Callable[[], Optional[int]],
                 *, routing_metadata_fn: Optional[Callable[[int], Dict[str, Any]]] = None,
                 now: Optional[float] = None) -> None:
        self._drive_root, self._repo_dir = pathlib.Path(drive_root), pathlib.Path(repo_dir)
        self._owner_chat_id_fn, self._lock = owner_chat_id_fn, threading.RLock()
        self._routing_metadata_fn = routing_metadata_fn
        self._booted_at = time.time() if now is None else float(now)
        state = self._read_state()
        from supervisor.state import control_is

        self._stopped = panic_blocks_wake(self._drive_root)  # unknown is suspension, Panic is a stop
        self._stop_source = PANIC_STOP if self._stopped else ""  # who stopped this process, whatever its write did
        self._enabled = not self._stopped and control_is(state, "bg_consciousness_enabled", True)  # unknown is not on (#1307)
        try:
            persisted = float(state.get(NEXT_WAKE_STATE_KEY) or 0.0)
        except (TypeError, ValueError):
            persisted = 0.0
        # Boot floor (PLAN 5.13 п.8): an overdue persisted value never wakes in the first second.
        self._next_wake_at = max(persisted, self._booted_at + self.floor)
        try:
            appointment = float(state.get(APPOINTMENT_STATE_KEY) or 0.0)
        except (TypeError, ValueError):
            appointment = 0.0
        self._appointment_at = appointment if math.isfinite(appointment) else 0.0
        raw_last_wake = state.get(LAST_WAKE_STATE_KEY)
        try:
            persisted_last_wake = float(raw_last_wake or 0.0)
            if not math.isfinite(persisted_last_wake):
                raise ValueError("non-finite timestamp")
        except (TypeError, ValueError):
            persisted_last_wake = 0.0
            log.warning(
                "consciousness: invalid persisted last wake boundary; using process-start boundary",
            )
        if persisted_last_wake < 0 or persisted_last_wake > self._booted_at:
            log.warning(
                "consciousness: persisted last wake boundary is outside the current boot window; "
                "using process-start boundary",
            )
            persisted_last_wake = 0.0
        self._last_wake_at, self._last_wake_task_id, self._last_wake_outcome, self._last_error = (
            persisted_last_wake, "", "", ""
        )
        admitted = state.get(OWN_WAKE_STATE_KEY)
        self._own_wake = dict(admitted) if isinstance(admitted, dict) else {}
        self._last_wake_task_id = str(self._own_wake.get("task_id") or "")
        self._continuity = wake_continuity(state.get(CONTINUITY_STATE_KEY))
        try:
            bootstrap = float(state.get(OBSERVATION_BOOTSTRAP_STATE_KEY, persisted_last_wake or self._booted_at))
            if not math.isfinite(bootstrap) or not 0 <= bootstrap <= self._booted_at:
                raise ValueError("invalid observation bootstrap")
        except (TypeError, ValueError):
            bootstrap = persisted_last_wake or self._booted_at
        self._observation_bootstrap_since = bootstrap
        self._backoff, self._allowance = 1, (0.0, {}, False)
        self._skip_basis = _money_skip_basis(state.get(LAST_SKIP_STATE_KEY), persisted)  # memory only
        self._in_flight = ""  # the id this process is binding, until its admission answers
        self._archive_legacy_inbox()

    floor = property(lambda self: int(get_bg_wakeup_min_sec()))
    ceiling = property(lambda self: int(get_bg_wakeup_max_sec()))
    enabled = property(lambda self: self._enabled)
    next_wake_at = property(lambda self: self._next_wake_at)

    @staticmethod
    def _read_state() -> Dict[str, Any]:
        from supervisor import state

        try:
            return dict(state.load_state() or {})
        except Exception:  # an unreadable state is a fresh clock, not a crash
            log.debug("consciousness: runtime state unreadable", exc_info=True)
            return {}

    def _set_next_wake(self, at: float, *, appointment: bool = False, skip: Optional[Dict[str, Any]] = None) -> None:
        from supervisor import state

        self._next_wake_at = float(at)
        keys = {NEXT_WAKE_STATE_KEY: self._next_wake_at, **({LAST_SKIP_STATE_KEY: skip} if skip else {})}
        if appointment:
            self._appointment_at = self._next_wake_at
            keys[APPOINTMENT_STATE_KEY] = self._appointment_at
        try:
            state.update_state(lambda st: st.update(keys))
        except Exception:
            log.debug("consciousness: next wake time not persisted", exc_info=True)

    @staticmethod
    def _set_state(key: str, value: Any) -> None:
        from supervisor import state

        try:
            state.update_state(lambda st: st.__setitem__(key, value))
        except Exception:
            log.debug("consciousness: %s not persisted", key, exc_info=True)

    def _interval(self) -> int:
        """The model's chosen interval (``set_next_wakeup``) or the default, clamped."""
        try:
            chosen = int(self._read_state().get(INTERVAL_STATE_KEY) or WAKE_DEFAULT_SEC)
        except (TypeError, ValueError):
            chosen = WAKE_DEFAULT_SEC
        return max(self.floor, min(self.ceiling, chosen))

    def _archive_legacy_inbox(self) -> None:
        """One-time move of the retired observation inbox into the archive (never read)."""
        source, target = self._drive_root / LEGACY_INBOX_REL, self._drive_root / ARCHIVED_INBOX_REL
        try:
            if source.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    target = target.with_name(f"consciousness_observations_{int(self._booted_at)}.jsonl")
                source.replace(target)
        except OSError:
            log.warning("consciousness: legacy observation inbox could not be archived", exc_info=True)

    def live_wake(self) -> str:
        """Its own live wake by identity, or "": this process's custody from binding until the direct
        callback returns, the admitted actor while the lane holds it, an own-clock turn (a revived
        clock's), else an own wake the pool carries. Owner work and focuses are never its wake."""
        from supervisor.active_activity import get_direct_activity_registry
        from supervisor.consciousness_carriers import queued_wakes
        from supervisor.workers import direct_chat_turn

        if self._in_flight:
            return self._in_flight
        registry = get_direct_activity_registry()
        admitted = "" if self._own_wake.get("finished") else str(self._own_wake.get("task_id") or "")
        if admitted and registry.get(admitted) is not None:
            return admitted
        pooled = queued_wakes()
        if admitted in pooled:
            return admitted
        for entry in registry.actors():
            if is_own_clock_wake((direct_chat_turn(str(entry.activity_id)) or {}).get("metadata") or {}):
                return str(entry.activity_id)
        return pooled[0] if pooled else ""

    def tick(self, now: Optional[float] = None) -> str:
        """One supervisor pass; never blocks on a wake. Returns the typed decision."""
        now = time.time() if now is None else float(now)
        wake = self.live_wake()  # never wait for Q while holding the clock lock
        ended_live = not wake and self._enabled and now >= self._next_wake_at and self._settle_unreported_terminal()
        with self._lock:
            from supervisor.state import control_is

            if panic_blocks_wake(self._drive_root):
                self._stopped, self._enabled, self._stop_source = True, False, PANIC_STOP
                return "panic_stop"
            self._enabled = not self._stopped and control_is(self._read_state(), "bg_consciousness_enabled", True)
            if not self._enabled:
                return "disabled"
            if wake or ended_live:
                return "wake_live"
            if now < self._next_wake_at and not self._limit_reopens_skip(now):
                return "not_due"
            window = self._allowance_view(now, fresh=True)
            if window.get("status") == STATUS_UNKNOWN:
                return self._skip("allowance_unknown", now + self.floor, window, now=now,
                                  error=str(window.get("error") or ""))
            # Known spend reached the daily allowance (#1487: unresolved bounds are
            # disclosed exposure, not spending). No extra early margin skips a wake.
            if window.get("status") == STATUS_EXHAUSTED or float(window.get("remaining_usd") or 0.0) <= 0.0:
                resets = parse_deadline_ts(str(window.get("resets_at") or ""))
                return self._skip("allowance_exhausted", max(resets.timestamp() if resets else 0.0, now + self.floor),
                                  window, now=now)
            if not self._owner_chat_id():
                return self._skip("waiting_for_first_conversation", now + self.floor, window, now=now)
        return self._launch(now, int(self._owner_chat_id() or 0), window)

    def _settle_unreported_terminal(self) -> bool:
        """Settle its own known wake from a pooled terminal no dispatcher reported (an orphan sweep, a
        boot restore, a lost event): one addressed row and one ownership probe, only when this tick
        would otherwise start over it. True while that wake's effects may still live: the next one
        waits. An unreadable, absent, unsettled or legacy in-process row (one that predates the carrier facts)
        settles nothing and blocks nothing; a lane carrier's row settles whichever carrier ran it, the
        disclosed degraded executor included; a cancelled one is no runner failure, and its stop door's next
        start stands."""
        from ouroboros.task_results import load_task_result
        from supervisor.consciousness_carriers import pooled_terminal_ok
        from supervisor.queue import task_settlement_liveness

        task_id = "" if self._own_wake.get("finished") else str(self._own_wake.get("task_id") or "")
        if not task_id:
            return False
        try:
            row = load_task_result(self._drive_root, task_id, strict=True) or {}
        except Exception:
            log.debug("Own wake %s is unreadable; its terminal stays unknown", task_id, exc_info=True)
            self._note_continuity(task_id, None)
            return False
        ok = pooled_terminal_ok(row)
        if ok is None or row.get("status") == "cancelled":
            self._note_continuity(task_id, row)  # the next start replaces it
            return False
        if task_settlement_liveness(task_id) is not False:
            return True
        self._wake_finished(task_id, ok, pooled=True)
        return False

    def _note_continuity(self, task_id: str, row: Any) -> None:
        """The notes past an own wake the next start replaces, from the row this clock already read (``None``:
        unreadable); a wake that settles writes them with its settlement (``_settle_wake``)."""
        from supervisor import state

        self._continuity = wake_continuity_after(row, task_id, self._continuity)
        carried = dict(self._continuity)
        try:
            state.update_state(lambda st: st.__setitem__(CONTINUITY_STATE_KEY, carried))
        except Exception:  # the next wake's carrier still reads that row's own facts (``previous_wake_text``)
            log.debug("consciousness: notes past %s not persisted", task_id, exc_info=True)

    def _skip(self, reason: str, next_at: float, window: Dict[str, Any], *, now: float, error: str = "") -> str:
        """A skipped wake keeps the allowance window it met (``allowance_fact``) in its event and in the one
        overwritten ``consciousness_last_skip`` key, written with the next wake time: a status read answers
        whether it stopped for money with no second reader or writer."""
        self._last_wake_outcome, self._last_error = f"skipped:{reason}", error
        skip = {"reason": reason, "next_wake_at": _iso(next_at), "error": error, "allowance": allowance_fact(window)}
        record = {"at": _iso(now), **skip}
        self._set_next_wake(next_at, skip=record)
        self._skip_basis = _money_skip_basis(record, self._next_wake_at)
        self._record("consciousness_wake_skipped", **skip)
        return f"skipped:{reason}"

    def _limit_reopens_skip(self, now: float) -> bool:
        """An exhausted allowance chose its time (the window's reset) under the limit it met. While that time
        stands, a different daily limit turns the wait into any skip's retry: the same window is re-checked no
        sooner than MIN after that skip and after boot, and admits or skips again under the new limit. The
        skip's own fact changed, no event moved the schedule. The check reads only the settings; it adds no
        ledger or state read to a tick (#2654)."""
        return now >= self._due_at() < self._next_wake_at

    def _due_at(self) -> float:
        """When the next check can come: the stored time, or, after a changed limit turned a money skip's wait into
        any skip's retry (``_limit_reopens_skip``), MIN after that skip and after boot. The status shows this time."""
        basis = self._skip_basis
        if basis is None or basis[0] != self._next_wake_at or basis[1] == float(get_consciousness_daily_usd()):
            return self._next_wake_at
        return min(self._next_wake_at, max(basis[2], self._booted_at) + self.floor)

    def _launch(self, now: float, chat_id: int, window: Dict[str, Any]) -> str:
        from supervisor import workers

        reason = HEARTBEAT  # the schedule is the only cause of a wake
        level, remaining = get_consciousness_autonomy(), float(window.get("remaining_usd") or 0.0)
        try:
            per_task_cap = float(runtime_setting("OUROBOROS_PER_TASK_COST_USD", "0") or 0)
        except (TypeError, ValueError):
            per_task_cap = 0.0
        # В26=A: a wake's whole tree may spend at most what is left of the rolling-24h
        # allowance, and never more than the owner's per-task cap (a cap of 0 disables
        # that half). It travels as the tree's producer ceiling (`root_cost_ceiling_usd`,
        # honored for the root itself by `task_pacing.resolve_cost_ceiling` and by the
        # members): the in-task stop pauses the tree when its known spend reaches it, with
        # no planning margin subtracted (owner 2026-10-07), while the ledger fence keeps the
        # owner's per-task cap — one Main attempt reserves several dollars up front, and a
        # fence narrowed below that refused every wake of a nearly spent day before its
        # first call.
        ceiling = min(per_task_cap, remaining) if per_task_cap > 0 else remaining
        # Which of the two set it, observed for the views that explain a pause (never read by a stop or a Resume).
        source = ("allowance" if per_task_cap <= 0 or remaining < per_task_cap
                  else "per_task_cap" if per_task_cap < remaining else "both")
        metadata = {**self._routing_facts(chat_id),
                    **wake_task_metadata(level, reason, root_cost_ceiling_usd=ceiling, root_cost_ceiling_source=source)}
        # Everything since the last ACCEPTED observation — chain positions and transition
        # identities, not this alarm's finish time. Before a verified boundary,
        # the bootstrap survives source failure, completion and restart.
        self._set_state(OBSERVATION_BOOTSTRAP_STATE_KEY, self._observation_bootstrap_since)
        # The clock decides WHEN and admits; the wake's preparation — observing the window,
        # the context facts, the render, the bound source — is the carrier's first step
        # (``consciousness_wake.prepare_wake``), never this thread's (owner decision 1A, L7).
        # Only the facts that live in this process travel with the admission.
        from supervisor.queue import live_consciousness_roots

        persisted = self._read_state()
        preparation = {
            "boundary": persisted.get(OBSERVATION_STATE_KEY), "since": self._observation_bootstrap_since,
            "now": now, "reason": reason, "level": level, "repo_dir": str(self._repo_dir),
            "last_wake_at": self._last_wake_at, "last_wake_task_id": self._last_wake_task_id,
            "continuity": dict(self._continuity),
            "disabled_tools": list(metadata.get("disabled_tools") or []),
            "window": allowance_fact(window),
            "counted_roots": live_consciousness_roots(), "max_tasks": get_consciousness_max_tasks(),
            "interval": self._interval(), "appointment_at": self._appointment_at,
        }
        bound_over, stopped_before_binding = None, False

        def bind_input(task):
            nonlocal bound_over, stopped_before_binding
            if not is_own_clock_wake(task.get("metadata")):
                return
            # Bind before the dispatch: even an immediate completion has an owner.
            with self._lock:
                bound_over = self._own_wake, self._last_wake_task_id, self._last_wake_outcome, self._last_error
                self._in_flight = str(task["id"])
                self._own_wake = {"task_id": str(task["id"]), "wake_cause": metadata["wake_cause"],
                                  "initiator": metadata["initiator"], "finished": False}
                self._set_state(OWN_WAKE_STATE_KEY, self._own_wake)
                self._last_wake_task_id, self._last_wake_outcome, self._last_error = str(task["id"]), "running", ""
                stopped_before_binding = self._stopped  # a Stop that found no wake to address yet
            task["_wake_prepare"] = dict(preparation)

        # The lane checks its gates under its own lock order (Q -> gate); only this tick starts own wakes.
        receipt: Dict[str, Any] = {}
        try:
            receipt = workers.handle_wake_direct(chat_id, f"[Wake-up · {reason}]", metadata, bind_input=bind_input)
        finally:
            # The admitted row carries the wake from here (``live_wake`` reads the pool); a refused
            # or raised admission after binding ran nothing, so this id never was its wake.
            self._in_flight = ""
            if not receipt.get("admitted") and bound_over is not None:
                with self._lock:
                    self._own_wake, self._last_wake_task_id, self._last_wake_outcome, self._last_error = bound_over
                    self._set_state(OWN_WAKE_STATE_KEY, self._own_wake)
        if not receipt.get("admitted"):
            why = str(receipt.get("reason") or "refused")
            self._last_wake_outcome = f"rejected:{why}"
            # A wake the lane could not admit (its admission raised; the lane logged it) backs off
            # like a failed wake. A closed door is retried quietly — the repo-writer gate at
            # the floor, the owner's budget at the interval.
            if why == "admission_failed":
                self._backoff = min(self._backoff * 2, 1024)
            transient = why == "repo_writer_gate_closed"
            self._set_next_wake(now + (self.floor if transient else min(self.ceiling, self._interval() * self._backoff)))
            self._record("consciousness_wake_rejected", reason=why, wake_reason=reason)
            return f"rejected:{why}"
        if stopped_before_binding:
            # That Stop found nothing to address; the admitted row exists now. The durable intent is
            # enough: the ordinary sweep routes it to whichever carrier owns the id (a later Stop
            # addressed the live id itself).
            from supervisor.consciousness_carriers import stop_pooled_wake

            stop_pooled_wake(str(self._last_wake_task_id))
        # The accepted observation boundary is the carrier's to bind (its first step); it reaches
        # this clock with the wake's terminal (``_settle_wake``), so the next wake starts after it.
        # A wake that dies before its terminal replays the same window (at-least-once, never a silent skip).
        self._record("consciousness_wake_started", task_id=self._last_wake_task_id, wake_reason=reason,
                     level=level, root_cost_ceiling_usd=ceiling, root_cost_ceiling_source=source)
        return "launched"

    def _wake_finished(self, task_id: str, ok: bool, *, metadata=None, pooled: bool = False) -> None:
        """Only the admitted own-clock actor's terminal advances the clock, once.

        A direct stack handing the same id to the pool has not finished the wake.
        A pooled terminal carries its stored cause and the persisted admission still
        names its actor after restart; descendants and stale callbacks own neither.
        The custody this process took at binding ends only here, after the next wake
        is written: the lane unregisters a direct turn before it calls back.
        """
        try:
            self._settle_wake(task_id, ok, metadata=metadata, pooled=pooled)
        finally:
            if not pooled and self._in_flight == task_id:
                self._in_flight = ""

    def _settle_wake(self, task_id: str, ok: bool, *, metadata=None, pooled: bool = False) -> None:
        if metadata is not None and not is_own_clock_wake(metadata):
            return
        from supervisor.consciousness_carriers import queued_wakes

        if not pooled and task_id in queued_wakes():
            return
        with self._lock:
            if (task_id != self._own_wake.get("task_id") or self._own_wake.get("finished")
                    or not is_own_clock_wake(self._own_wake)):
                return
        outcome = "unknown"
        row = None
        try:
            from ouroboros.task_results import load_task_result
            row = load_task_result(self._drive_root, task_id, strict=True)
            owner = (row or {}).get("execution_owner") or {}
            if not pooled and owner.get("kind") == "pooled":
                return  # durable handoff can precede queue restoration / follow the pool's terminal
            # The carrier bound the observation this wake accepted (``prepare_wake``): the
            # next wake starts after it. Nothing bound (a binder failure) keeps the window.
            bound = ((row or {}).get("metadata") or {}).get(WAKE_OBSERVATION_KEY)
            accepted = bound.get("accepted_boundary") if isinstance(bound, dict) else None
            if isinstance(accepted, dict) and accepted.get("task_id") == task_id:
                self._set_state(OBSERVATION_STATE_KEY, accepted)
                # Every time bootstrap starts at this capture, never at the installation's first start: a body that cannot
                # read the boundary's transition chain, and a chat window re-read by time after a chat position that no
                # longer matches (docs/PERSISTENCE.md). Persisted with the settlement below.
                captured = parse_deadline_ts(accepted.get("captured_at"))
                if captured is not None:
                    self._observation_bootstrap_since = captured.timestamp()
            if ok:
                pause = (row or {}).get("budget_pause") or {}
                owner_pause = (row or {}).get("owner_pause") or {}
                if owner_pause.get("state") == "requested" or pause.get("state") == "pausing":
                    outcome = "pausing"
                elif (owner_pause.get("state") == "paused" or pause.get("state") == "paused"
                      or row and row.get("status") == "scheduled"
                      and row.get("reason_code") in {"budget_paused", "budget_exhausted", "owner_paused",
                                                     "authority_paused"}):
                    outcome = "paused"
                elif row:
                    outcome = "done"  # the ordinary successful-turn contract is unchanged
        except Exception:
            log.debug("Wake outcome is unreadable for %s", task_id, exc_info=True)
        with self._lock:
            if task_id != self._own_wake.get("task_id") or self._own_wake.get("finished"):
                return
            now = time.time()
            self._last_wake_at, self._last_wake_task_id = now, str(task_id)
            self._own_wake = {**self._own_wake, "finished": True}
            # Like the predecessor id, retain this computation when the state write fails.
            self._continuity = wake_continuity_after(row, task_id, self._continuity)
            if ok:
                self._backoff, self._last_wake_outcome, self._last_error = 1, outcome, ""
            else:
                self._backoff, self._last_wake_outcome = min(self._backoff * 2, 1024), "failed"
                self._last_error = f"wake-up {task_id} failed in its runner (see the chat and events.jsonl)"
            self._next_wake_at = self._appointment_at = now + min(self.ceiling, self._interval() * self._backoff)
            from supervisor import state
            try:  # the notes past it (``None``: an unreadable row) persist with the settlement itself
                state.update_state(lambda st: st.update({OWN_WAKE_STATE_KEY: self._own_wake,
                    LAST_WAKE_STATE_KEY: now, NEXT_WAKE_STATE_KEY: self._next_wake_at,
                    APPOINTMENT_STATE_KEY: self._appointment_at,
                    OBSERVATION_BOOTSTRAP_STATE_KEY: self._observation_bootstrap_since,
                    CONTINUITY_STATE_KEY: dict(self._continuity)}))
            except Exception:
                log.debug("consciousness: wake completion not persisted", exc_info=True)
        if (row or {}).get("wake_summary") and not row.get("title"):
            from supervisor.worker_chat_lane import _broadcast_task_named
            from ouroboros.text_sentences import first_sentence
            _broadcast_task_named({"type": "task_named", "task_id": task_id,
                                   "suggested_name": first_sentence(row["wake_summary"])})

    def start(self, source: str = OWNER_STOP) -> str:
        """Turn on by ``source``'s right (the owner's doors; ``AGENT_STOP`` for the mind's tool). A stop this process
        holds and the saved one are both decided, under the clock's lock, before anything turns on."""
        with self._lock:
            if panic_blocks_wake(self._drive_root):
                return "Background consciousness stays disabled while Panic controls await persistence."
            if source != OWNER_STOP and self._stopped and self._stop_source != AGENT_STOP:
                return held_stop_text(self._stop_source)  # held here even when its own write failed
            refusal, note = _persist_start(source)
            if refusal:
                return refusal
            if self._enabled:
                return "Background consciousness is already enabled." + note
            self._stopped, self._enabled, self._stop_source = False, True, ""
            # The clock did not advance while disabled: never announce a wake in the past.
            self._set_next_wake(max(self._next_wake_at, time.time()), appointment=True)
            at = time.strftime('%H:%M', time.localtime(self._next_wake_at))
            return f"Background consciousness enabled; next wake-up at {at}." + note

    def stop(self, source: str = "") -> str:
        """Stop this process first, then save the stop with ``source`` (``OWNER_STOP`` or ``AGENT_STOP``); Panic names
        none, its own bounded writer saves its controls. The mind's stop never relabels one this process holds."""
        with self._lock:
            if source != AGENT_STOP or not self._stopped or self._stop_source == AGENT_STOP:
                self._stop_source = source or OWNER_STOP
            self._stopped, held_by = True, self._stop_source
            was_enabled, self._enabled = self._enabled, False
        saved = _persist_stop(held_by) if source else ""  # a stop the owner holds here is saved as the owner's
        wake = self.live_wake()
        from supervisor.consciousness_carriers import autonomous_tasks, live_task_rows
        from ouroboros.conversation_lease import live_authors
        from supervisor.active_activity import get_direct_activity_registry
        from supervisor.workers import direct_chat_turn
        direct, turns = [], {}
        for entry in get_direct_activity_registry().actors():
            task_id = str(entry.activity_id)
            turn = direct_chat_turn(task_id) or {}
            metadata = turn.get("metadata") or {}
            if turn:  # its title and metadata name it if the switch ends it
                turns[task_id] = {"title": str(turn.get("title") or turn.get("suggested_name") or ""), "metadata": metadata}
            if is_consciousness_origin(metadata) and metadata.get("work_kind") != "owner":
                direct.append(task_id)
        # Its own wake by identity too: bound before delivery, a direct turn is no chat turn yet.
        authors = live_authors(drive_root=self._drive_root)
        wakes = list(dict.fromkeys(([wake] if wake else []) + direct + autonomous_tasks() + authors))
        # Work started on my human's cited words is theirs: the switch leaves it running and says so, quote and all,
        # in the same Main row every surface shows (owner 06.11, 3A); they cancel it from its card if the cite was wrong.
        from supervisor.consciousness_carriers import cited_owner_roots, cited_owner_schedules
        kept = "".join(f"\nKept running on your words (task {task_id}): «{quote}»" for task_id, quote in cited_owner_roots())
        kept += "".join(f"\nKept firing on your words (schedule {sid}): «{quote}»"
                        for sid, quote in cited_owner_schedules(self._drive_root))
        if not wakes and not was_enabled:
            return "Background consciousness is already disabled." + saved + kept
        for task_id in wakes:
            threading.Thread(target=self._stop_live_wake, args=(task_id,), name="wake-stop", daemon=True).start()
        # What Off leaves and what it ends, in the same row (owner 09.11, variant A). The stops are already on their
        # way: the place count is one read that never holds or fails them; what it ends is named from memory.
        from ouroboros.consciousness_view import off_sentence, stopping_words
        try:
            from ouroboros.places import load_places
            places = load_places(self._drive_root)
        except Exception:
            log.debug("consciousness: places unreadable for the stop sentence", exc_info=True)
            places = None
        ending = stopping_words(wakes, wake=wake, rows={**live_task_rows(wakes), **turns}, authors=set(authors),
                                places=places) if wakes else ""
        return " ".join(part for part in (off_sentence(places), ending) if part) + saved + kept

    @staticmethod
    def _stop_live_wake(task_id: str) -> None:
        from supervisor import worker_chat_lane, workers

        try:
            turn = workers.direct_chat_turn(task_id)
            from supervisor.consciousness_carriers import stop_pooled_wake
            if turn is not None:
                worker_chat_lane.stop_direct_chat_turn(task_id, turn)
            # A handoff can race this read. An intent survives it and the ordinary
            # sweep routes the stop to whichever carrier owns the id now.
            stop_pooled_wake(task_id)
        except Exception:
            log.debug("consciousness: graceful stop of wake %s failed", task_id, exc_info=True)

    def status_snapshot(self) -> Dict[str, Any]:
        from supervisor.consciousness_carriers import queued_wakes

        wake = self.live_wake()
        window = self._allowance_view(time.time())
        context = self._context_facts(window)
        values = (self._enabled, get_consciousness_autonomy(), _iso(self._due_at()),
                  _iso(self._last_wake_at), self._last_wake_task_id, self._last_wake_outcome, self._last_error,
                  window.get("settled_usd"), window.get("limit_usd"), str(window.get("resets_at") or ""),
                  len(context["counted_roots"]), int(get_consciousness_max_tasks()), wake,
                  int(window.get("unknown_unmetered") or 0), bool(window.get("integrity_degraded")))
        places = context.pop("places")
        # Held: its own wake waits in the pool (paused, held or not yet dispatched), not running.
        held = bool(wake) and wake in queued_wakes(held_only=True)
        from ouroboros.consciousness_view import held_wake_cause
        return {**dict(zip(SNAPSHOT_KEYS, values)), **context, "appointment_at": _iso(self._appointment_at),
                "live_wake_held": held,
                # What holds it, proven from its records; the current window's own verdict stays beside it.
                "live_wake_hold": held_wake_cause(self._drive_root, wake) if held else None,
                "allowance_status": str(window.get("status") or ""),
                "allowance_open_rows": window.get("accounting_open_rows"),
                "allowance_room_after": str(window.get("room_after") or ""),
                "model_route": model_route(),
                "places_brief": [{key: place.get(key) for key in ("place_id", "label", "state", "unanswered")}
                                 for place in places] if places is not None else None}

    def details_snapshot(self, requested_enabled: bool | None) -> Dict[str, Any]:
        """The full read-only panel; the state endpoint keeps its small snapshot."""
        from ouroboros.consciousness_view import RECENT_WAKE_LIMIT, wake_ids, describe_consciousness_state, recent_wakes
        from ouroboros.store import StoreBusy
        from ouroboros.usage_store import UsageLockUnavailable

        snapshot = describe_consciousness_state(self.status_snapshot(), requested_enabled)
        # The lists' caption, a sentence of its own: what they hold, or how far a full window reaches.
        coverage = "Wakes that reached a model call in the last 48 h"
        try:
            ids = wake_ids(self._drive_root, time.time())
            wakes = recent_wakes(self._drive_root, snapshot.get("places_brief"),
                                 live_wake_task_id=snapshot.get("live_wake_task_id", ""), ids=ids)
            if len(ids) == RECENT_WAKE_LIMIT:
                places = sum(wake.get("cause") == "focus" for wake in wakes)
                own = len(wakes) - places
                coverage = (f"The latest {own} of its own wakes and {places} wakes in places "
                            "that reached a model call in the last 48 h" if own and places
                            else f"The latest {RECENT_WAKE_LIMIT} wakes that reached a model call in the last 48 h")
        except UsageLockUnavailable:
            wakes, coverage = None, "Recent wakes could not be read: usage ledger busy."
        except StoreBusy as exc:  # the read that met it names the store (consciousness_view._reading)
            store = getattr(exc, "store_name", "") or "data"  # unnamed stays unknown, never the task store
            wakes, coverage = None, f"Recent wakes could not be read: {store} store busy."
        return {**snapshot, "recent_wakes": wakes, "recent_wakes_coverage": coverage}

    def _context_facts(self, window) -> dict:
        from ouroboros.consciousness_view import cause_views, place_views, root_views
        from supervisor.queue import live_consciousness_roots
        try:
            places = place_views(self._drive_root, window)
        except Exception:
            places = None
        return {"spend_by_cause": cause_views(window.get("spend_by_cause"), places),
                "counted_roots": root_views(live_consciousness_roots(), places), "places": places}

    def _allowance_view(self, now: float, *, fresh: bool = False) -> Dict[str, Any]:
        """The status view's window (with its display forecast) or a wake's exact admission read (without one).
        A view reuses a cached window only while it is still this window: a display read, under the same limit,
        before its first counted row leaves and inside the TTL; a wake's forecast-less read never stands in."""
        cached_at, cached, shown = self._allowance
        changes_at = cached.get("window_changes_at") if cached else None
        if (not fresh and shown and cached and now - cached_at < ALLOWANCE_VIEW_TTL_SEC
                and cached.get("limit_usd") == float(get_consciousness_daily_usd())
                and (changes_at is None or now <= changes_at)):
            return cached
        try:  # status shows; a wake admits
            window = allowance_window(self._drive_root, now=now, allow_stale=not fresh, **({} if fresh else {"forecast": True}))
        except Exception as exc:  # the reader types every failure; this is the last net
            window = {"status": STATUS_UNKNOWN, "error": f"{type(exc).__name__}: {exc}"}
        self._allowance = (now, window, not fresh)
        return window

    def _routing_facts(self, chat_id: int) -> Dict[str, Any]:
        """The Main-lane host facts an owner turn in this chat gets (the routing manifest,
        the addressable roots). A wake is an ordinary Main turn, so the results it may
        continue must be addressable the same way; a failed seam is disclosed and the wake
        still starts fresh work with ``predecessor_task_id=""``."""
        try:
            facts = self._routing_metadata_fn(int(chat_id)) if self._routing_metadata_fn else {}
        except Exception:
            log.warning("consciousness: Main routing facts unavailable for this wake", exc_info=True)
            return {}
        return dict(facts) if isinstance(facts, dict) else {}

    def _owner_chat_id(self) -> Optional[int]:
        try:
            return int(self._owner_chat_id_fn() or 0) or None
        except Exception:
            return None

    def _record(self, event_type: str, **fields: Any) -> None:
        try:
            append_jsonl(self._drive_root / "logs" / "events.jsonl", {"ts": utc_now_iso(), "type": event_type, **fields})
        except Exception:
            log.debug("consciousness: %s not recorded", event_type, exc_info=True)
