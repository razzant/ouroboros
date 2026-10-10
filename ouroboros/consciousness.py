"""The alarm clock of Background Consciousness: when Ouroboros wakes up on its own.

A wake-up is an ORDINARY Main direct turn nobody typed (owner decisions В13/В15, PLAN 5.1–5.3):
Main's prompt, memory, tools and loop, started through ``supervisor.workers.handle_wake_direct``
with the wake's envelope, the Main lane's own routing facts and rendered message. This module owns only
the clock — WHEN such a turn starts and what to say about it afterwards. No thread, no private
registry, no observation inbox, no pause/resume: the supervisor loop calls ``tick(now)`` once
per pass and the ``DirectActivityRegistry`` (the census of live direct turns) is the only
liveness truth. Tick order, each step a typed outcome: disabled → a live wake → a live owner
turn → not yet due → the rolling-24h allowance (an unreadable ledger is the disclosed skip
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
wake succeeds. ``notify(reason)`` (a root task finished, a project digest, an orphan-heal sweep)
pulls the next wake to ``max(now, (last wake or boot) + min)`` — arithmetic debounce, one floor
everywhere, and the boot floor holds against the first post-restart event. An owner message never
wakes it (nor does the owner's own turn finishing), and a wake's own finish (or that of a task it
started) never re-arms it. A health
WARNING/CRITICAL trigger is deliberately NOT implemented: nothing emits a health event
(``build_health_invariants`` is read-side), so health is visible in the next wake's context.
Panic and ``/bg stop`` arrive through ``stop()``, which arms a graceful stop of a live wake
off-thread. The legacy observation inbox, if present, is moved once to the archive unread.
What a wake OBSERVES is not this clock's business: it starts after the chat-chain position and
the task-result transition state the last admitted wake observed
(``consciousness_observation_boundary``, persisted only once the lane admits the wake;
``consciousness_wake.observe_wake``), never at a finish time.
"""

from __future__ import annotations

import datetime as _dt
import logging
import math
import os
import pathlib
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

from ouroboros.config import (
    WAKE_DEFAULT_SEC,
    get_bg_wakeup_max_sec,
    get_bg_wakeup_min_sec,
    get_consciousness_autonomy,
    get_consciousness_max_tasks,
    runtime_setting,
)
from ouroboros.consciousness_allowance import STATUS_EXHAUSTED, STATUS_UNKNOWN, allowance_window
from ouroboros.consciousness_authority import is_consciousness_origin
from ouroboros.consciousness_wake import bind_wake_observation, observe_wake, render_wake_message, wake_task_metadata
from ouroboros.deadline_utils import parse_deadline_ts
from ouroboros.utils import append_jsonl, utc_now_iso

log = logging.getLogger(__name__)

NEXT_WAKE_STATE_KEY = "consciousness_next_wake_at"
INTERVAL_STATE_KEY = "consciousness_next_interval_sec"
LAST_WAKE_STATE_KEY = "consciousness_last_wake_at"
OBSERVATION_STATE_KEY = "consciousness_observation_boundary"  # chat position + transition state the last ACCEPTED wake observed
OBSERVATION_BOOTSTRAP_STATE_KEY = "consciousness_observation_bootstrap_since"
LEGACY_INBOX_REL = pathlib.Path("state") / "consciousness_observations.jsonl"
ARCHIVED_INBOX_REL = pathlib.Path("archive") / "consciousness_observations.jsonl"
HEARTBEAT = "heartbeat"
ALLOWANCE_VIEW_TTL_SEC = 60.0
SNAPSHOT_KEYS = ("enabled", "level", "next_wake_at", "pending_reason", "last_wake_at", "last_wake_task_id",
                 "last_wake_outcome", "last_error", "spent_24h_usd", "daily_usd", "allowance_resets_at",
                 "tasks_running", "max_tasks", "live_wake_task_id", "unknown_unmetered", "integrity_degraded")


def _iso(ts: float) -> str:
    return _dt.datetime.fromtimestamp(float(ts), tz=_dt.timezone.utc).isoformat() if ts else ""


def panic_blocks_wake(drive_root: Any) -> bool:
    """A kept Panic intent bars all automatic wake grants. Only a proven absence of the flag
    (``confirm_absent``) permits one: a dangling link is present, and an unreadable flag or
    an absence no directory ancestor proves is unknown, so it bars too."""
    from supervisor.state_initialization import confirm_absent

    path = pathlib.Path(drive_root) / "state" / "panic_stop.flag"
    try:
        try:
            os.lstat(path)  # lstat, not stat: a dangling link IS present
        except FileNotFoundError:
            confirm_absent(path)
            return False
        return True
    except OSError:
        return True


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
        self._pending_reason: Optional[str] = None
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
        try:
            bootstrap = float(state.get(OBSERVATION_BOOTSTRAP_STATE_KEY, persisted_last_wake or self._booted_at))
            if not math.isfinite(bootstrap) or not 0 <= bootstrap <= self._booted_at:
                raise ValueError("invalid observation bootstrap")
        except (TypeError, ValueError):
            bootstrap = persisted_last_wake or self._booted_at
        self._observation_bootstrap_since = bootstrap
        self._last_skip_at = 0.0  # a skipped wake debounces the next event like a wake does
        self._backoff, self._allowance = 1, (0.0, {})
        self._archive_legacy_inbox()

    floor = property(lambda self: int(get_bg_wakeup_min_sec()))
    ceiling = property(lambda self: int(get_bg_wakeup_max_sec()))
    enabled = property(lambda self: self._enabled)
    next_wake_at = property(lambda self: self._next_wake_at)
    pending_reason = property(lambda self: self._pending_reason)

    @staticmethod
    def _read_state() -> Dict[str, Any]:
        from supervisor import state

        try:
            return dict(state.load_state() or {})
        except Exception:  # an unreadable state is a fresh clock, not a crash
            log.debug("consciousness: runtime state unreadable", exc_info=True)
            return {}

    def _set_next_wake(self, at: float) -> None:
        from supervisor import state

        self._next_wake_at = float(at)
        try:
            state.update_state(lambda st: st.__setitem__(NEXT_WAKE_STATE_KEY, self._next_wake_at))
        except Exception:
            log.debug("consciousness: next wake time not persisted", exc_info=True)

    def _set_last_wake_at(self, at: float) -> None:
        from supervisor import state

        try:
            # Even if the first bootstrap write failed, alarm completion may
            # never persist without the observation lower bound beside it.
            state.update_state(lambda st: st.update({LAST_WAKE_STATE_KEY: float(at),
                OBSERVATION_BOOTSTRAP_STATE_KEY: self._observation_bootstrap_since}))
        except Exception:
            log.debug("consciousness: wake completion not persisted", exc_info=True)

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

    def live_turns(self) -> Tuple[str, bool]:
        """``(live wake task id or "", owner turn live?)`` read off the direct-activity census."""
        from supervisor.active_activity import get_direct_activity_registry
        from supervisor.workers import direct_chat_turn

        wake, owner = "", False
        for entry in get_direct_activity_registry().actors():
            activity_id = str(entry.activity_id)
            metadata = (direct_chat_turn(activity_id) or {}).get("metadata") or {}
            if activity_id == self._last_wake_task_id or is_consciousness_origin(metadata):
                wake = activity_id
            else:
                owner = True
        return wake, owner

    def tick(self, now: Optional[float] = None) -> str:
        """One supervisor pass; never blocks on a wake. Returns the typed decision."""
        now = time.time() if now is None else float(now)
        with self._lock:
            from supervisor.state import control_is

            if panic_blocks_wake(self._drive_root):
                self._stopped, self._enabled = True, False
                return "panic_stop"
            self._enabled = not self._stopped and control_is(self._read_state(), "bg_consciousness_enabled", True)
            if not self._enabled:
                return "disabled"
            wake, owner_live = self.live_turns()
            if wake or owner_live:
                return "wake_live" if wake else "owner_turn_live"
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
        self._last_wake_outcome, self._last_error, self._last_skip_at = f"skipped:{reason}", error, float(now)
        self._set_next_wake(next_at)
        self._record("consciousness_wake_skipped", reason=reason, next_wake_at=_iso(next_at), error=error)
        return f"skipped:{reason}"

    def _launch(self, now: float, chat_id: int, window: Dict[str, Any]) -> str:
        from supervisor import workers

        reason, self._pending_reason = (self._pending_reason or HEARTBEAT), None  # capture-and-clear
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

        def render(events: str) -> str:
            return render_wake_message(
                self._repo_dir, reason=reason, last_wake_at=self._last_wake_at, now=now, level=level,
                disabled_tools=list(metadata.get("disabled_tools") or []), spent_usd=window.get("settled_usd"),
                spent_is_floor=int(window.get("unknown_unmetered") or 0) > 0,
                daily_usd=window.get("limit_usd") or 0.0, running=self._running_roots(),
                max_tasks=get_consciousness_max_tasks(), interval=self._interval(), events=events)

        accepted_boundary = None

        def bind_input(task):
            nonlocal accepted_boundary
            accepted_boundary = bind_wake_observation(self._drive_root, task, observation, render)

        # Check-and-register under the lane's own re-entrant gate lock: atomic with the census.
        with workers._repo_writer_gate_lock:
            wake, owner_live = self.live_turns()
            if wake or owner_live:
                self._pending_reason = self._pending_reason or reason
                return "wake_live" if wake else "owner_turn_live"
            receipt = workers.handle_wake_direct(
                chat_id, render(observation.full_text()), metadata, on_finished=self._wake_finished,
                bind_input=bind_input)
        if not receipt.get("admitted"):
            why = str(receipt.get("reason") or "refused")
            self._last_wake_outcome, self._last_skip_at = f"rejected:{why}", now  # an event never undoes this backoff below the floor
            # A wake the lane could not admit already left an error in the chat: back off
            # like a failed wake. A closed door is retried quietly — the repo-writer gate at
            # the floor, the owner's budget at the interval. The event that asked for this
            # wake is kept: a refused launch does not consume it.
            self._pending_reason = self._pending_reason or reason
            if why == "admission_failed":
                self._backoff = min(self._backoff * 2, 1024)
            transient = why == "repo_writer_gate_closed"
            self._set_next_wake(now + (self.floor if transient else min(self.ceiling, self._interval() * self._backoff)))
            self._record("consciousness_wake_rejected", reason=why, wake_reason=reason)
            return f"rejected:{why}"
        self._last_wake_task_id, self._last_wake_outcome, self._last_error = str(receipt["task_id"]), "running", ""
        # Accepted: the running wake owns this observation, so the next one starts
        # after it. A refused launch above consumed nothing; a crash before this
        # write replays the same window (at-least-once, never a silent skip).
        if accepted_boundary is not None:
            self._set_state(OBSERVATION_STATE_KEY, accepted_boundary)
        self._record("consciousness_wake_started", task_id=self._last_wake_task_id, wake_reason=reason,
                     level=level, root_cost_ceiling_usd=ceiling)
        return "launched"

    def _wake_finished(self, task_id: str, ok: bool) -> None:
        """Runner success controls backoff; stored work facts describe its outcome."""
        outcome = "unknown"
        if ok:
            try:
                from ouroboros.task_results import load_task_result
                row = load_task_result(self._drive_root, task_id, strict=True)
                pause = (row or {}).get("budget_pause") or {}
                owner_pause = (row or {}).get("owner_pause") or {}
                if owner_pause.get("state") == "requested" or pause.get("state") == "pausing":
                    outcome = "pausing"
                elif (owner_pause.get("state") == "paused" or pause.get("state") == "paused"
                      or row and row.get("status") == "scheduled"
                      and row.get("reason_code") in {"budget_paused", "budget_exhausted", "owner_paused"}):
                    outcome = "paused"
                elif row:
                    outcome = "done"  # the ordinary successful-turn contract is unchanged
            except Exception:
                log.debug("Wake outcome is unreadable for %s", task_id, exc_info=True)
        with self._lock:
            now = time.time()
            self._last_wake_at, self._last_wake_task_id = now, str(task_id)
            self._set_last_wake_at(now)
            if ok:
                self._backoff, self._last_wake_outcome, self._last_error = 1, outcome, ""
            else:
                self._backoff, self._last_wake_outcome = min(self._backoff * 2, 1024), "failed"
                self._last_error = f"wake-up {task_id} failed in its runner (see the chat and events.jsonl)"
            interval = min(self.ceiling, self._interval() * self._backoff)
            self._set_next_wake(now + (self.floor if self._pending_reason else interval))

    def notify(self, reason: str) -> None:
        """An event worth waking for: keep the last reason, pull the next wake to the floor."""
        with self._lock:
            self._pending_reason = str(reason or "event")
            target = max(time.time(), max(self._last_wake_at, self._booted_at, self._last_skip_at) + self.floor)
            if target < self._next_wake_at:
                self._set_next_wake(target)

    def start(self) -> str:
        with self._lock:
            if panic_blocks_wake(self._drive_root):
                return "Background consciousness stays disabled while Panic controls await persistence."
            if self._enabled:
                return "Background consciousness is already enabled."
            self._stopped, self._enabled = False, True
            # The clock did not advance while disabled: never announce a wake in the past.
            self._set_next_wake(max(self._next_wake_at, time.time()))
            return f"Background consciousness enabled; next wake-up at {time.strftime('%H:%M', time.localtime(self._next_wake_at))}."

    def stop(self) -> str:
        with self._lock:
            self._stopped = True
            was_enabled, self._enabled = self._enabled, False
        wake, _owner = self.live_turns()
        if wake:
            threading.Thread(target=self._stop_live_wake, args=(wake,), name="wake-stop", daemon=True).start()
            return f"Background consciousness disabled; wake-up {wake} ends at its next step."
        return "Background consciousness disabled." if was_enabled else "Background consciousness is already disabled."

    @staticmethod
    def _stop_live_wake(task_id: str) -> None:
        from supervisor import worker_chat_lane, workers

        try:
            turn = workers.direct_chat_turn(task_id)
            turn is None or worker_chat_lane.stop_direct_chat_turn(task_id, turn)
        except Exception:
            log.debug("consciousness: graceful stop of wake %s failed", task_id, exc_info=True)

    def status_snapshot(self) -> Dict[str, Any]:
        wake, _owner = self.live_turns()
        window = self._allowance_view(time.time())
        values = (self._enabled, get_consciousness_autonomy(), _iso(self._next_wake_at), self._pending_reason or "",
                  _iso(self._last_wake_at), self._last_wake_task_id, self._last_wake_outcome, self._last_error,
                  window.get("settled_usd"), window.get("limit_usd"), str(window.get("resets_at") or ""),
                  self._running_roots(), int(get_consciousness_max_tasks()), wake,
                  int(window.get("unknown_unmetered") or 0), bool(window.get("integrity_degraded")))
        return dict(zip(SNAPSHOT_KEYS, values))

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

    @staticmethod
    def _running_roots() -> int:
        try:
            from supervisor.queue import live_consciousness_root_count

            return int(live_consciousness_root_count())
        except Exception:
            return 0

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
