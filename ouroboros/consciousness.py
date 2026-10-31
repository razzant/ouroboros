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
``allowance_exhausted``) → an owner chat must be bound → launch.
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
Nothing else moves the alarm (owner 12 Oct, #1603: no accelerated wake, only the
scheduled time): a task finishing, a project digest or an orphan-heal sweep never pulls the next
wake earlier — they wait as observed facts for the scheduled wake (``observe_wake``), so every
wake's reason is the heartbeat. Neither does an owner message, nor a health event (``build_health_invariants``
is read-side; health is visible in the next wake's context).
Panic and ``/bg stop`` arrive through ``stop()``, which arms ordinary graceful cascade
cancellation for own wakes, live/lent focuses and autonomous work, off-thread.
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
from typing import Any, Callable, Dict, Optional

from ouroboros.config import (
    WAKE_DEFAULT_SEC,
    get_bg_wakeup_max_sec,
    get_bg_wakeup_min_sec,
    get_consciousness_autonomy,
    get_consciousness_max_tasks,
    runtime_setting,
)
from ouroboros.consciousness_allowance import STATUS_EXHAUSTED, STATUS_UNKNOWN, allowance_window
from ouroboros.consciousness_authority import is_consciousness_origin, is_own_clock_wake
from ouroboros.consciousness_wake import bind_wake_observation, observe_wake, previous_wake_text, render_wake_message, wake_task_metadata
from ouroboros.deadline_utils import parse_deadline_ts
from ouroboros.utils import append_jsonl, utc_now_iso

log = logging.getLogger(__name__)

NEXT_WAKE_STATE_KEY = "consciousness_next_wake_at"
INTERVAL_STATE_KEY = "consciousness_next_interval_sec"
LAST_WAKE_STATE_KEY = "consciousness_last_wake_at"
APPOINTMENT_STATE_KEY = "consciousness_appointment_at"  # the scheduled time; skips and refusals move only the next attempt
OWN_WAKE_STATE_KEY = "consciousness_own_wake"  # the admitted clock actor, retained across carrier changes/restart
OBSERVATION_STATE_KEY = "consciousness_observation_boundary"  # chat position + transition state the last ACCEPTED wake observed
OBSERVATION_BOOTSTRAP_STATE_KEY = "consciousness_observation_bootstrap_since"
LEGACY_INBOX_REL = pathlib.Path("state") / "consciousness_observations.jsonl"
ARCHIVED_INBOX_REL = pathlib.Path("archive") / "consciousness_observations.jsonl"
HEARTBEAT = "heartbeat"
ALLOWANCE_VIEW_TTL_SEC = 60.0
SNAPSHOT_KEYS = ("enabled", "level", "next_wake_at", "last_wake_at", "last_wake_task_id",
                 "last_wake_outcome", "last_error", "spent_24h_usd", "daily_usd", "allowance_resets_at",
                 "tasks_running", "max_tasks", "live_wake_task_id", "unknown_unmetered", "integrity_degraded")


def _iso(ts: float) -> str:
    return _dt.datetime.fromtimestamp(float(ts), tz=_dt.timezone.utc).isoformat() if ts else ""


def panic_blocks_wake(drive_root: Any) -> bool:
    """An unreceived Panic input bars automatic wake grants."""
    from ouroboros.installation_stop import installation_panic

    return installation_panic(drive_root)


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
        try:
            bootstrap = float(state.get(OBSERVATION_BOOTSTRAP_STATE_KEY, persisted_last_wake or self._booted_at))
            if not math.isfinite(bootstrap) or not 0 <= bootstrap <= self._booted_at:
                raise ValueError("invalid observation bootstrap")
        except (TypeError, ValueError):
            bootstrap = persisted_last_wake or self._booted_at
        self._observation_bootstrap_since = bootstrap
        self._backoff, self._allowance = 1, (0.0, {})
        self._in_flight = ""  # the id this process bound, until its direct callback returns
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

    def _set_next_wake(self, at: float, *, appointment: bool = False) -> None:
        from supervisor import state

        self._next_wake_at = float(at)
        keys = {NEXT_WAKE_STATE_KEY: self._next_wake_at}
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
        with self._lock:
            from supervisor.state import control_is

            if panic_blocks_wake(self._drive_root):
                self._stopped, self._enabled = True, False
                return "panic_stop"
            self._enabled = not self._stopped and control_is(self._read_state(), "bg_consciousness_enabled", True)
            if not self._enabled:
                return "disabled"
            if wake:
                return "wake_live"
            if now < self._next_wake_at:
                return "not_due"
            window = self._allowance_view(now, fresh=True)
            if window.get("status") == STATUS_UNKNOWN:
                return self._skip("allowance_unknown", now + self.floor, now=now, error=str(window.get("error") or ""))
            # Known spend reached the daily allowance (#1487: unresolved bounds are
            # disclosed exposure, not spending). No extra early margin skips a wake.
            if window.get("status") == STATUS_EXHAUSTED or float(window.get("remaining_usd") or 0.0) <= 0.0:
                resets = parse_deadline_ts(str(window.get("resets_at") or ""))
                return self._skip("allowance_exhausted", max(resets.timestamp() if resets else 0.0, now + self.floor), now=now)
            if not self._owner_chat_id():
                return self._skip("waiting_for_first_conversation", now + self.floor, now=now)
        return self._launch(now, int(self._owner_chat_id() or 0), window)

    def _skip(self, reason: str, next_at: float, *, now: float, error: str = "") -> str:
        self._last_wake_outcome, self._last_error = f"skipped:{reason}", error
        self._set_next_wake(next_at)
        self._record("consciousness_wake_skipped", reason=reason, next_wake_at=_iso(next_at), error=error)
        return f"skipped:{reason}"

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
        metadata = {**self._routing_facts(chat_id),
                    **wake_task_metadata(level, reason, root_cost_ceiling_usd=ceiling)}
        # Everything since the last ACCEPTED observation — chain positions and transition
        # identities, not this alarm's finish time. Before a verified boundary,
        # the bootstrap survives source failure, completion and restart.
        self._set_state(OBSERVATION_BOOTSTRAP_STATE_KEY, self._observation_bootstrap_since)
        observation = observe_wake(self._drive_root, boundary=self._read_state().get(OBSERVATION_STATE_KEY),
                                   since=self._observation_bootstrap_since, now=now, reason=reason)
        context = self._context_facts(window)
        previous = previous_wake_text(self._drive_root, self._last_wake_task_id, now=now)

        def render(events: str) -> str:
            return render_wake_message(
                self._repo_dir, drive_root=self._drive_root, reason=reason, last_wake_at=self._last_wake_at, now=now, level=level,
                disabled_tools=list(metadata.get("disabled_tools") or []), spent_usd=window.get("settled_usd"),
                spent_is_floor=int(window.get("unknown_unmetered") or 0) > 0,
                daily_usd=window.get("limit_usd") or 0.0, running=len(context["counted_roots"]),
                max_tasks=get_consciousness_max_tasks(), interval=self._interval(), events=events, previous_wake=previous,
                appointment_at=self._appointment_at, **context)

        accepted_boundary = None

        def bind_input(task):
            nonlocal accepted_boundary
            if not is_own_clock_wake(task.get("metadata")):
                return
            # Bind before the thread starts: even an immediate completion has an owner.
            with self._lock:
                self._in_flight = str(task["id"])
                self._own_wake = {"task_id": str(task["id"]), "wake_cause": metadata["wake_cause"],
                                  "initiator": metadata["initiator"], "finished": False}
                self._set_state(OWN_WAKE_STATE_KEY, self._own_wake)
                self._last_wake_task_id, self._last_wake_outcome, self._last_error = str(task["id"]), "running", ""
                stopped = self._stopped
            if stopped:  # Stop can arrive during admission, before the clock knows this id.
                self._stop_live_wake(str(task["id"]))
            accepted_boundary = bind_wake_observation(self._drive_root, task, observation, render)

        # The lane checks its gates under its own lock order (Q -> gate); only this tick starts own wakes.
        receipt: Dict[str, Any] = {}
        try:
            receipt = workers.handle_wake_direct(
                chat_id, render(observation.full_text()), metadata, on_finished=self._wake_finished,
                bind_input=bind_input)
        finally:
            if not receipt.get("admitted"):
                self._in_flight = ""  # refused or raised after binding: no callback will come
        if not receipt.get("admitted"):
            why = str(receipt.get("reason") or "refused")
            self._last_wake_outcome = f"rejected:{why}"
            # A wake the lane could not admit already left an error in the chat: back off
            # like a failed wake. A closed door is retried quietly — the repo-writer gate at
            # the floor, the owner's budget at the interval.
            if why == "admission_failed":
                self._backoff = min(self._backoff * 2, 1024)
            transient = why == "repo_writer_gate_closed"
            self._set_next_wake(now + (self.floor if transient else min(self.ceiling, self._interval() * self._backoff)))
            self._record("consciousness_wake_rejected", reason=why, wake_reason=reason)
            return f"rejected:{why}"
        # Accepted: the running wake owns this observation, so the next one starts
        # after it. A refused launch above consumed nothing; a crash before this
        # write replays the same window (at-least-once, never a silent skip).
        if accepted_boundary is not None and str(receipt["task_id"]) == self._own_wake.get("task_id"):
            self._set_state(OBSERVATION_STATE_KEY, accepted_boundary)
        self._record("consciousness_wake_started", task_id=self._last_wake_task_id, wake_reason=reason,
                     level=level, root_cost_ceiling_usd=ceiling)
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
            if ok:
                self._backoff, self._last_wake_outcome, self._last_error = 1, outcome, ""
            else:
                self._backoff, self._last_wake_outcome = min(self._backoff * 2, 1024), "failed"
                self._last_error = f"wake-up {task_id} failed in its runner (see the chat and events.jsonl)"
            self._next_wake_at = self._appointment_at = now + min(self.ceiling, self._interval() * self._backoff)
            from supervisor import state
            try:
                state.update_state(lambda st: st.update({OWN_WAKE_STATE_KEY: self._own_wake,
                    LAST_WAKE_STATE_KEY: now, NEXT_WAKE_STATE_KEY: self._next_wake_at,
                    APPOINTMENT_STATE_KEY: self._appointment_at,
                    OBSERVATION_BOOTSTRAP_STATE_KEY: self._observation_bootstrap_since}))
            except Exception:
                log.debug("consciousness: wake completion not persisted", exc_info=True)
        if (row or {}).get("wake_summary"):
            from supervisor.worker_chat_lane import _broadcast_task_named
            _broadcast_task_named({"type": "task_named", "task_id": task_id, "suggested_name": row["wake_summary"]})

    def start(self) -> str:
        with self._lock:
            if panic_blocks_wake(self._drive_root):
                return "Background consciousness stays disabled while Panic controls await persistence."
            if self._enabled:
                return "Background consciousness is already enabled."
            self._stopped, self._enabled = False, True
            # The clock did not advance while disabled: never announce a wake in the past.
            self._set_next_wake(max(self._next_wake_at, time.time()), appointment=True)
            return f"Background consciousness enabled; next wake-up at {time.strftime('%H:%M', time.localtime(self._next_wake_at))}."

    def stop(self) -> str:
        with self._lock:
            self._stopped = True
            was_enabled, self._enabled = self._enabled, False
        wake = self.live_wake()
        from supervisor.consciousness_carriers import autonomous_tasks
        from ouroboros.conversation_lease import live_authors
        from supervisor.active_activity import get_direct_activity_registry
        from supervisor.workers import direct_chat_turn
        direct = []
        for entry in get_direct_activity_registry().actors():
            task_id = str(entry.activity_id)
            metadata = (direct_chat_turn(task_id) or {}).get("metadata") or {}
            if is_consciousness_origin(metadata) and metadata.get("work_kind") != "owner":
                direct.append(task_id)
        # Its own wake by identity too: bound before delivery, a direct turn is no chat turn yet.
        wakes = list(dict.fromkeys(([wake] if wake else []) + direct + autonomous_tasks()
                                   + live_authors(drive_root=self._drive_root)))
        if wakes:
            for task_id in wakes:
                threading.Thread(target=self._stop_live_wake, args=(task_id,), name="wake-stop", daemon=True).start()
            return (f"Background consciousness disabled; wake-up {wake} ends at its next step." if wake
                    else "Background consciousness disabled; autonomous work ends at its next step.")
        return "Background consciousness disabled." if was_enabled else "Background consciousness is already disabled."

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
        values = (self._enabled, get_consciousness_autonomy(), _iso(self._next_wake_at),
                  _iso(self._last_wake_at), self._last_wake_task_id, self._last_wake_outcome, self._last_error,
                  window.get("settled_usd"), window.get("limit_usd"), str(window.get("resets_at") or ""),
                  len(context["counted_roots"]), int(get_consciousness_max_tasks()), wake,
                  int(window.get("unknown_unmetered") or 0), bool(window.get("integrity_degraded")))
        places = context.pop("places")
        # Held: its own wake waits in the pool (paused, held or not yet dispatched), not running.
        return {**dict(zip(SNAPSHOT_KEYS, values)), **context, "appointment_at": _iso(self._appointment_at),
                "live_wake_held": bool(wake) and wake in queued_wakes(held_only=True),
                "places_brief": [{key: place.get(key) for key in ("place_id", "label", "state", "unanswered")}
                                 for place in places] if places is not None else None}

    def details_snapshot(self, requested_enabled: bool | None) -> Dict[str, Any]:
        """The full read-only panel; the state endpoint keeps its small snapshot."""
        from ouroboros.consciousness_view import RECENT_WAKE_LIMIT, wake_ids, describe_consciousness_state, recent_wakes
        from ouroboros.store import StoreBusy
        from ouroboros.usage_store import UsageLockUnavailable

        snapshot = describe_consciousness_state(self.status_snapshot(), requested_enabled)
        coverage = "wakes that reached a model call in the last 48 h"
        try:
            ids = wake_ids(self._drive_root, time.time())
            wakes = recent_wakes(self._drive_root, snapshot.get("places_brief"),
                                 live_wake_task_id=snapshot.get("live_wake_task_id", ""), ids=ids)
            if len(ids) == RECENT_WAKE_LIMIT:
                places = sum(wake.get("cause") == "focus" for wake in wakes)
                own = len(wakes) - places
                coverage = (f"the latest {own} of its own wakes and {places} wakes in places "
                            "that reached a model call in the last 48 h" if own and places
                            else f"the latest {RECENT_WAKE_LIMIT} " + coverage)
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
        cached_at, cached = self._allowance
        if not fresh and cached and now - cached_at < ALLOWANCE_VIEW_TTL_SEC:
            return cached
        try:
            window = allowance_window(self._drive_root, now=now, allow_stale=not fresh)  # status shows; a wake admits
        except Exception as exc:  # the reader types every failure; this is the last net
            window = {"status": STATUS_UNKNOWN, "error": f"{type(exc).__name__}: {exc}"}
        self._allowance = (now, window)
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
