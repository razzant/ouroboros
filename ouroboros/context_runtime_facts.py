"""The runtime section's FACT builders: what the host can honestly say it knows.

Extracted whole from ``context.py`` at its module ceiling (v7 leaf) so the
facts the runtime section renders keep one home: the project room a task sits in,
the budget rails it runs under, how the run learns the time (its capture instant
labels the Recent/Drive snapshots), the body's own dated facts (``vitals_fact``, beside
``storage_fact``), and the
configured delegation route with its honestly-labeled historical observations.
Each returns a plain projection and reads no context state, so nothing here can
change what the section MEANS — only what it reports. ``context`` re-exports the
start-of-task builders, so historical imports and monkeypatch targets keep working
unchanged; ``round_clock_fact`` is the per-round time segment of the facts line
(``loop_messages``), read from the same clock owners.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, Optional, Tuple

from ouroboros.task_pacing import in_task_cost_ceiling_disclosure as _in_task_cost_ceiling
from ouroboros.config import runtime_setting

log = logging.getLogger(__name__)


def schedule_store_fact(env: Any) -> Dict[str, Any]:
    """Observed schedule controls for context; an unreadable table is not empty."""
    from supervisor.followup_policy import observed_store
    from supervisor.queue import ScheduleStoreUnreadable, load_schedule_store

    root = getattr(env, "drive_root", None) or pathlib.Path(env.drive_path("state/scheduled_tasks.json")).parent.parent
    try:
        return observed_store(root, load_schedule_store(root))
    except ScheduleStoreUnreadable as exc:
        return {"unavailable": str(exc), "pointer": "manage_schedules(action='list')"}


def task_schedule_fact(task: Dict[str, Any]) -> Dict[str, Any]:
    """The admitted occurrence's original clock, even after its row advances.

    Lateness is for the mind to judge. A missing legacy date stays unknown;
    the schedule's next firing point cannot supply this occurrence's due time.
    """
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    occurrence = metadata.get("schedule_occurrence")
    if not isinstance(occurrence, dict):
        return {}
    fact = {
        key: occurrence.get(key) for key in ("schedule_id", "due_at", "claimed_at")
    }
    if "previous" in occurrence:
        fact["previous"] = occurrence["previous"]
    return {"schedule_occurrence": fact}


def task_execution_clock_fact(task: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    """Current finite execution-ceiling estimate, not a calendar deadline.

    Quota/budget pauses can move the estimate after this context is assembled;
    an unknown start or unlimited ceiling yields null instead of a false date.
    """
    import datetime
    import math
    import time
    from ouroboros.config import explicit_execution_timeout_sec, resolve_task_execution_timeout_sec
    from ouroboros.deadline_utils import parse_deadline_ts
    from ouroboros.model_wait import current_model_wait, execution_elapsed_seconds

    raw = getattr(ctx, "task_started_at", None) or task.get("started_at")
    try:
        start = float(raw)
    except (TypeError, ValueError):
        parsed = parse_deadline_ts(raw)
        start = parsed.timestamp() if parsed is not None else 0.0
    ceiling = resolve_task_execution_timeout_sec(task)  # the narrower of explicit budget and global ceiling
    started = datetime.datetime.fromtimestamp(start, datetime.timezone.utc).isoformat() if start > 0 and math.isfinite(start) else None
    projected = None
    if started and ceiling is not None:
        now = time.time()
        owner = current_model_wait()
        elapsed = (owner.executed_seconds() if owner is not None and owner.task_id == str(task.get("id") or "")
                   else execution_elapsed_seconds({**task, "started_at": start}, now))
        try:
            projected = datetime.datetime.fromtimestamp(now + max(0.0, ceiling - elapsed),
                                                        datetime.timezone.utc).isoformat()
        except (OverflowError, ValueError, OSError):
            pass  # a finite duration need not fit the platform's calendar range: stays not_set
    # The same absolute_ceiling rail either way; the basis names which bound it is (#939).
    bound = "task execution budget" if ceiling is not None and ceiling == explicit_execution_timeout_sec(task) else "global ceiling"
    return {"started_at": started, "absolute_ceiling_at": projected,
            "absolute_ceiling_at_basis": (f"current estimate of the {bound}; quota or budget pauses may move it"
                                          if projected else "not_set")}


def _start_epoch(raw: Any) -> Optional[float]:
    import math
    from ouroboros.deadline_utils import parse_deadline_ts

    try:
        start = float(raw)
    except (TypeError, ValueError):
        parsed = parse_deadline_ts(raw)
        start = parsed.timestamp() if parsed is not None else 0.0
    return start if start > 0 and math.isfinite(start) else None


def round_clock_fact(ctx: Any, now: Optional[float] = None) -> str:
    """The facts line's time segment, read from the Time clock owners (#1786) per round.

    Calendar time since this task's clock origin and execution time (the carried
    offset, quota waits, budget pauses and warm sleep excluded) are named apart; the
    execution limit names its basis; the calendar deadline stays its own axis; the
    host's local finalization instant exists only for an explicit deadline or
    execution budget (``loop_round_limits._maybe_deadline_local_finalize``). A
    missing start is unknown, never zero; every value is as of the one sample
    instant, with the live wait owner's clock when it owns this task and otherwise
    the same carriers ``model_wait.execution_window_remaining_for_context`` reads.
    An owner turn in this process ends the segment with its own wait so far
    (``_owner_wait_fact``); a wake, a pooled task or a resume has none.
    """
    import datetime
    import time
    from ouroboros.config import explicit_execution_timeout_sec, resolve_task_execution_timeout_sec
    from ouroboros.deadline_utils import parse_deadline_ts
    from ouroboros.model_wait import budget_paused_seconds, current_model_wait, execution_elapsed_seconds
    from ouroboros.task_pacing import effective_finalization_reserve_sec

    def stamp(epoch: float) -> str:
        return datetime.datetime.fromtimestamp(epoch, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    now = time.time() if now is None else float(now)
    meta = getattr(ctx, "task_metadata", None)
    meta = meta if isinstance(meta, dict) else {}
    limit = resolve_task_execution_timeout_sec(meta)
    parts, executed, left = [f"time sampled {stamp(now)}"], None, None
    owner = getattr(ctx, "model_wait_context", None) or current_model_wait()
    start = _start_epoch(getattr(ctx, "task_started_at", None) or meta.get("started_at"))
    if owner is not None and owner.task_id == str(getattr(ctx, "task_id", "") or ""):
        executed, left = owner.executed_seconds(), owner.execution_window_remaining()
    elif start is not None:
        paused = getattr(ctx, "_budget_paused_sec", None)
        if paused is None:
            paused = budget_paused_seconds(getattr(ctx, "budget_pause_resume", None) or {})
        executed = execution_elapsed_seconds({**meta, "started_at": start, "budget_paused_sec": paused}, now)
        left = None if limit is None else max(0.0, limit - executed)
    if start is None:
        parts.append("wall elapsed unknown (no recorded start)")
    else:
        parts.append(f"wall {max(0.0, now - start) / 60:.1f} min since start {stamp(start)}")
    parts.append("execution elapsed unknown" if executed is None
                 else f"execution {executed / 60:.1f} min (quota waits and pauses excluded)")
    if limit is None:
        parts.append("no execution limit")
    elif left is None:
        parts.append(f"execution limit {limit / 60:.1f} min, remaining unknown")
    else:
        basis = "task execution budget" if limit == explicit_execution_timeout_sec(meta) else "global execution ceiling"
        parts.append(f"{basis} {limit / 60:.1f} min, {left / 60:.1f} min left")
    deadline = parse_deadline_ts(meta.get("deadline_at"))
    bounds = [left] if left is not None and explicit_execution_timeout_sec(meta) is not None else []
    if deadline is not None:
        bounds.append(deadline.timestamp() - now)
        parts.append(f"deadline {stamp(deadline.timestamp())}, {(deadline.timestamp() - now) / 60:.1f} min left")
    if bounds:
        grace = effective_finalization_reserve_sec(ctx)
        parts.append(f"host requests one tool-less final answer at ~{stamp(now + max(0.0, min(bounds) - grace))} "
                     f"({grace / 60:.1f} min before the nearer explicit bound)")
    parts.extend(filter(None, [_owner_wait_fact(ctx)]))
    return "; ".join(parts)


def _owner_wait_fact(ctx: Any) -> str:
    """This turn's own wait, from the process-local registry (``TurnExecution.receipt``): when my human's
    message was accepted and how much later this request comes, so far, with the slowest phase; once
    the first dispatch was observed, when it went out and on which basis. Elapsed only, never a
    dispatch that has not happened; "" for a turn nobody awaits or a process that serves none."""
    from supervisor.active_activity import get_direct_activity_registry

    entry = get_direct_activity_registry().get(str(getattr(ctx, "task_id", "") or ""))
    execution = getattr(entry, "execution", None)
    if execution is None or not execution.owner_turn:
        return ""
    receipt = execution.receipt()
    slowest = (f"; slowest phase so far {receipt['slowest_phase']} {receipt['slowest_phase_sec']:.1f} s"
               if receipt.get("slowest_phase") else "")
    if receipt.get("accepted_at"):
        origin, wait = f"my human's message accepted at {receipt['accepted_at']}", receipt["owner_wait_sec"]
    else:
        origin, wait = f"my turn began at {receipt['entered_at']} (acceptance stamp unknown)", receipt["handler_wait_sec"]
    if receipt["dispatch"] == "not_observed":
        return f"{origin}; this request {wait:.1f} s later{slowest}"
    return f"{origin}; my first request went out {wait:.1f} s later ({receipt['dispatch']}){slowest}"


def _context_clock_note(task: Dict[str, Any]) -> str:
    """How this run learns the time: Main gets a clock line per request (``send_clock``)."""
    from ouroboros.send_clock import main_clock_policy

    meta = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    if task.get("delegation_role") and "delegation_role" not in meta:
        meta = {**meta, "delegation_role": task.get("delegation_role")}
    if main_clock_policy(meta, task_type=str(task.get("type") or "")) is None:
        return ("context_captured_at is when this context was built; it does not advance "
                "during this run.")
    return ("context_captured_at is when this context was built, not the current time. "
            "Each of your model requests ends with a host clock line sampled for that request.")


def snapshot_labelled(section: str, captured_at: str) -> str:
    """Label a captured Recent/Drive section with its capture time, below its heading.

    These sections are rendered once per run and stay byte-stable in the cached
    prefix; the label says so instead of refreshing them every round.
    """
    heading, sep, body = str(section or "").partition("\n")
    if not heading.startswith(("## Recent ", "## Drive state")) or not captured_at:
        return section
    label = (f"_Snapshot captured at {captured_at} when this context was built; "
             "not refreshed during this run._")
    return heading + "\n" + label + ((sep + body) if body else "")


def _queue_context_fact(task: Dict[str, Any]) -> Dict[str, Any]:
    """One dated canonical-queue view, frozen with the task's ContextCore."""
    from ouroboros.config import DATA_DIR, get_max_active_subagents_per_root, get_max_workers
    from ouroboros.task_status import _load_queue_snapshot, queue_snapshot_observation

    root = pathlib.Path(task.get("budget_drive_root") or DATA_DIR)
    snapshot = _load_queue_snapshot(root)
    fact = {**queue_snapshot_observation(snapshot), "source_root": str(root),
            "max_workers": int(get_max_workers()),
            "max_active_subagents_per_root": int(get_max_active_subagents_per_root())}
    if snapshot.get("_snapshot_missing") or snapshot.get("_snapshot_invalid"):
        return {**fact, "note": "Queue observation unavailable; current capacity is unknown."}
    for key in ("running", "pending"):
        rows = snapshot.get(key)
        fact[key + "_count"] = sum(isinstance(row, dict) for row in rows) if isinstance(rows, list) else None
    fact["reaping_count"] = snapshot.get("reaping_count")
    fact["worker_total"] = snapshot.get("worker_total")
    assignable = snapshot.get("assignable_idle_workers")
    if assignable is not None:
        fact["free_worker_slots"] = max(0, int(assignable))
        fact["free_worker_slots_basis"] = "recorded_assignable_idle_workers"
    elif fact["running_count"] is not None:
        fact["free_worker_slots"] = max(0, fact["max_workers"] - fact["running_count"]
                                          - int(fact["reaping_count"] or 0))
        fact["free_worker_slots_basis"] = "legacy_estimate_from_configured_limit"
    else:
        fact["free_worker_slots"] = None
        fact["free_worker_slots_basis"] = "unknown"
    fact["note"] = (
        "Last recorded queue counts at ts; freshness and age were measured when this context "
        "was built and do not refresh during the task. Stale or unknown observations do not "
        "establish current load. Legacy free-slot estimates are not measured capacity. "
        "Scheduling owns admission; these observations reserve no slots."
    )
    return fact


def _project_room_fact(task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The room's active folder and ordinary tool target, or None.

    Conversation keeps its own lifecycle; selecting a room changes the physical
    target without requiring a managed workspace or Git repository. Promotion
    continues to use its separate workspace admission contract.
    """
    try:
        _room_pid = str(task.get("project_id") or "").strip()
        if _room_pid and not str(task.get("workspace_root") or "").strip():
            from ouroboros.config import DATA_DIR as _DATA_DIR
            from ouroboros.projects_registry import get_project as _get_project
            from ouroboros.workspace_admission import room_chat_lens_dir as _room_lens, room_lens_applies

            _room = _get_project(_DATA_DIR, _room_pid) or {}
            _room_wd = str(_room.get("working_dir") or "").strip()
            if _room_wd:
                # Same resolver the agent uses for the tool lens, so the stated rule
                # and the actual tool surface cannot diverge (the robot incident).
                _lens_dir, _room_note = _room_lens(_DATA_DIR, _room_pid)
                _lens_active = room_lens_applies(task) and bool(_lens_dir)
                fact = {
                    "project_id": _room_pid,
                    "working_dir": _room_wd,
                    "rule": (
                        (
                            "This room's active_workspace is working_dir for file reads, "
                            "writes and edits, the default shell cwd, VCS and selected "
                            "delegation. The tools retain their own requirements and "
                            "explicit task constraints. Ouroboros governance remains at "
                            "system_repo; select that root explicitly to work on the body. "
                            "Direct work and promotion are both available."
                        )
                        if _lens_active
                        else (
                            "This project has a working folder. Tasks promoted from this room "
                            "run with it as their active workspace by default; pass "
                            "workspace='none' to promote a folder-less task."
                        )
                    ),
                }
                if _room_note:
                    fact["working_dir_warning"] = _room_note
                return fact
    except Exception:
        log.debug("Failed to inject project_room working_dir fact", exc_info=True)
    return None


def _runtime_budget_info(env: Any, task: Dict[str, Any], ctx: Any = None) -> Dict[str, Any]:
    """Start-of-task budget block: global projection + the STATIC per-task tree cap,
    written once at task start so the cached prefix stays byte-stable (DEVELOPMENT
    cache_friendliness item 24); live wallet and tree spend ride the per-round facts
    line (``loop_messages.context_facts_line``)."""
    try:
        from ouroboros.usage_accounting import usage_projection
        from ouroboros.settings_setup_contract import resolve_total_budget_usd

        total_usd = resolve_total_budget_usd()
        budget_root = pathlib.Path(task.get("budget_drive_root") or env.drive_root)
        projection = usage_projection(budget_root, global_limit_usd=total_usd)
        # Known spend (confirmed + estimated) is what the limit decides on; open
        # holds ride beside it, never added in (#1487).
        spent_usd = float(projection.get("settled_usd") or 0.0)
        budget_info = {
            "status": "available" if total_usd is not None else "no_global_limit", "total_usd": total_usd,
            "spent_usd": spent_usd, "spent_basis": "known_settled_incl_estimates",
            "remaining_usd": None if total_usd is None else max(0.0, total_usd - spent_usd),
            "reserved_usd": float(projection.get("reserved_usd") or 0.0),
            "unresolved_upper_bound_usd": float(projection.get("unresolved_upper_bound_usd") or 0.0),
            "unknown_unmetered": int(projection.get("unknown_unmetered") or 0),
        }
    except Exception:
        log.error("Budget authority unavailable for runtime context", exc_info=True)
        budget_info = {"status": "unavailable"}
    try:
        root_cap = float(runtime_setting("OUROBOROS_PER_TASK_COST_USD", "0") or 0)
    except (TypeError, ValueError):
        root_cap = 0.0
    if root_cap > 0:
        budget_info["per_task_tree_cap_usd"] = root_cap
        budget_info["per_task_tree_cap_rule"] = (
            "Hard cap for THIS task's WHOLE tree (own model calls + all subagents), enforced "
            "by the physical-attempt ledger: new dispatches are refused once the tree's KNOWN "
            "spend (confirmed and estimated) reaches it. A task with an exact continuation then "
            "pauses with its work saved; an actor without one (no continuation owner, task id or "
            "durable root) ends with a typed budget_exhausted result and no paid recap. "
            "Open reservations and unresolved charges are not counted as spending, and calls "
            "in flight or charged late can take the tree past the cap. The facts line of each "
            "round reports the live tree number."
        )
    if ctx is not None:
        budget_info["in_task_cost_ceiling"] = _in_task_cost_ceiling(ctx, budget_info.get("remaining_usd"))
    return budget_info


def storage_fact(data_root: Any) -> Optional[Dict[str, Any]]:
    """The last finished storage count as one compact fact for the mind: when it was counted,
    what Ouroboros owns and how many items it could not read, the growth since the count before
    (flagged when either count had unreadable items), free and total space of the data folder's
    disk, and one ``removers`` line only while a maintenance job's last outcome is not clean
    (``storage_removers.not_ok`` over the local job rows: never an engine call while a turn is prepared;
    the engine's record is ``query_data(kind="storage")``'s and the Storage page's), present before any
    count too. None when nothing was counted and every job is clean (never zero, never an exception). No
    threshold, tier or warning stands in for judgment here: the mind reads the numbers. The saved file is
    read through the storage owner's identity cache; free space and the job rows are read fresh."""
    import shutil

    from ouroboros import storage, storage_removers

    try:
        removers = "; ".join(storage_removers.not_ok(storage_removers.place_removers(data_root, engine=False)))
    except Exception:  # the readers already answer unknown; anything else is said, never a broken section
        log.debug("storage removers unreadable for the runtime fact", exc_info=True)
        removers = "the removers' facts could not be read"
    try:
        saved = storage.count.saved(data_root)
        count = storage.with_growth(saved["latest"], saved["previous"]) if saved else None
    except Exception:  # an unreadable or malformed file is no fact, never a broken Runtime section
        log.debug("storage count unreadable for the runtime fact", exc_info=True)
        count = None
    if count is None:
        return {"removers": removers} if removers else None
    try:
        usage = shutil.disk_usage(str(data_root))
        free, total = usage.free, usage.total
    except OSError:
        free = total = None
    fact = {
        "counted_at": count["finished_at"],
        "bytes": count["bytes"],
        "unreadable": count["unreadable"],  # items the count could not read: the headline excludes them
        "basis": count.get("basis"),
        "previous_at": count.get("growth_since"),
        "growth_bytes": count.get("growth_bytes"),
        "free_bytes": free,
        "volume_total_bytes": total,
    }
    if count.get("growth_flag"):
        fact["growth_flag"] = count["growth_flag"]  # either count had unreadable items: the growth is qualified
    if removers:
        fact["removers"] = removers
    return fact


# The body's own facts: bounds here are transport and memory limits, never judgments. Each item is one
# short line of text (the Runtime section is indented JSON), kept small for small context windows: clocks
# are UTC HH:MM of observed_at's day (MM-DD HH:MM on another day), and no phrase carries a verdict.
_SUPERVISOR_TAIL_BYTES = 256_000  # the same bounded tail the health block reads
_OWNER_TURNS_SHOWN = 2  # receipts rendered per context; the ring keeps 100 and the journal every one
_STALL_ENDS = {"direct_turn_stall": "direct_turn_stall_end", "host_duty_stall": "host_duty_stall_end",
               "supervisor_loop_stall": "supervisor_loop_stall_end"}
_STALL_WORDS = {"direct_turn_stall": "turn", "host_duty_stall": "duty", "supervisor_loop_stall": "loop"}
_STALLS_KEY = "stalls, no end in window"
_ROUTES_SHOWN = 4  # failing routes rendered in the line; the row keeps every route
_CAUSE_CHARS = 80  # of a top cause's message in the line; the row keeps the engine's whole first line
_STORE_FILES = ("state/current.sqlite", "state/dialog.sqlite", "state/usage.sqlite")
_JOURNAL_MODES: Dict[str, str] = {}  # per process: each store file's header, read once
VITALS_READER = "query_data(kind='supervisor_log', op='jsonl')"  # the minimal working call: the kind requires op='jsonl'
VITALS_NOTE = f"No verdict here, the weighing is mine; clocks UTC; rows: {VITALS_READER}"
WAL_GATE = ("WAL needs a SQLite with the WAL-reset fix (3.51.3+, or 3.50.7+ / 3.44.6+ within those series),"
            " enforced kernel locks and a local filesystem; an Update keeps this interpreter")


def _clock(stamp: Any, today: Any) -> str:
    """HH:MM of a UTC stamp, with its day (MM-DD) only when that is not observed_at's day."""
    text = str(stamp or "")
    if len(text) < 16:
        return text
    return text[11:16] if text[:10] == str(today or "")[:10] else f"{text[5:10]} {text[11:16]}"


def _sec(value: Any) -> str:
    return f"{float(value):.1f} s" if isinstance(value, (int, float)) else "unknown"


def _site(site: Any) -> str:
    text = str(site or "")
    return text.split(".", 1)[1] if text.startswith(("ouroboros.", "supervisor.")) else text


def _stall_key(row: Dict[str, Any]) -> tuple:
    return (str(row.get("execution_id") or ""), str(row.get("duty") or ""), str(row.get("phase") or ""))


def _supervisor_tail(root: pathlib.Path) -> Dict[str, Any]:
    """One pass over the journal's bounded tail: the newest boot and ``supervisor_load`` rows, the newest
    ``direct_turn`` receipts (a pooled worker has no ring), stall onsets whose end is not in the window
    (an onset's ``_end`` row with the same identity closes it; a closed episode's ``top_frames`` is kept
    by execution for the receipts). The tail is the one the health block reads, memoized on the file's
    identity (``jsonl_tail.cached_tail_objects``): one read per context build serves both. An
    unreadable or absent journal is no fact: empty, never an error. The newest route-health row is not
    read here: its own file keeps it past this tail (``claudexor_exit_facts.latest_route_health``)."""
    from ouroboros.jsonl_tail import cached_tail_objects

    found: Dict[str, Any] = {"open": {}, "ended": {}, "turns": []}
    try:
        rows = cached_tail_objects(root / "logs" / "supervisor.jsonl", _SUPERVISOR_TAIL_BYTES)
    except Exception:
        log.debug("supervisor journal tail unreadable for the body fact", exc_info=True)
        return found
    for row in rows:
        kind = str(row.get("type") or "")
        if kind in ("launcher_start", "restart", "boot"):
            found["boot"] = row
        elif kind == "supervisor_load":
            found["load"] = row
        elif kind == "direct_turn":
            found["turns"] = [*found["turns"], row][-_OWNER_TURNS_SHOWN:]
        elif kind in _STALL_ENDS:
            found["open"][(kind, _stall_key(row))] = row
        elif kind in _STALL_ENDS.values():
            onset = kind[: -len("_end")]
            found["open"].pop((onset, _stall_key(row)), None)
            if onset == "direct_turn_stall" and row.get("execution_id"):
                found["ended"][str(row["execution_id"])] = row
    return found


def _owner_turn(row: Dict[str, Any], ended: Dict[str, Any], today: Any) -> str:
    """One receipt in one line: when, the chat, how long my human waited to my first request (the
    handler's own wait when the acceptance stamp was unknown), the slowest phase, the dispatch basis
    when it was not a seen send, the turn's transactions per store (r/w) and chronicle opens, the stall
    episode's top frame when its end row is still in the tail, and a journal row that was not written."""
    wait = row.get("owner_wait_sec")
    text = f"{_clock(row.get('ts'), today)} chat {row.get('chat_id')} waited " + (
        _sec(wait) if wait is not None else f"{_sec(row.get('handler_wait_sec'))} from entry (acceptance stamp unknown)")
    if row.get("slowest_phase"):
        text += f", slowest {row['slowest_phase']} {_sec(row.get('slowest_phase_sec'))}"
    if row.get("dispatch") not in (None, "observed_send"):
        text += f", {row['dispatch']}"
    meter = row.get("meter") if isinstance(row.get("meter"), dict) else {}
    stores: Dict[str, str] = {}
    for key, value in meter.items():
        if key.endswith("_sec") or key.endswith(".sec"):
            continue
        kind, _dot, side = key.partition(".")
        stores[kind] = stores.get(kind, "") + (f" {value} locked" if side == "locked" else f" {value}{side[:1]}")
    if stores:
        text += "; txns " + " · ".join(f"{kind}{counts}" for kind, counts in stores.items())
    frames = (ended.get(str(row.get("execution_id") or "")) or {}).get("top_frames")
    if isinstance(frames, list) and frames and isinstance(frames[0], dict) and frames[0].get("frame"):
        text += f"; top {str(frames[0]['frame']).rsplit('/', 1)[-1]}"
    if row.get("journal") == "not_written":
        text += "; journal row not written"
    return text


def _stall(kind: str, row: Dict[str, Any], today: Any) -> str:
    where = " ".join(str(row[key]) for key in ("phase", "duty") if row.get(key))
    ident = str(row.get("task_id") or row.get("execution_id") or "")
    return " ".join(part for part in (_STALL_WORDS.get(kind, kind), ident, where, "since", _clock(row.get("ts"), today)) if part)


def _busy_minute(row: Dict[str, Any], today: Any) -> str:
    """The newest ``supervisor_load`` row in one line: the worst event lag; per store, holds as
    count/summed seconds by side, the longest hold's site after its store; the queue lock's holds."""
    text = f"{_clock(row.get('ts'), today)}: lag ≤{_sec(row.get('max_event_lag_sec'))}"
    locks = row.get("store_locks")
    if isinstance(locks, dict) and locks and "error" not in locks:
        longest, parts = (0.0, "", ""), []
        for kind, sides in locks.items():
            if not isinstance(sides, dict):
                continue
            parts.append(f"{kind} " + " ".join(f"{side[:1]}{bucket.get('count')}/{float(bucket.get('total_sec') or 0.0):.1f} s"
                                               for side, bucket in sides.items() if isinstance(bucket, dict)))
            for bucket in sides.values():
                if isinstance(bucket, dict) and float(bucket.get("max_sec") or 0.0) >= longest[0] and bucket.get("max_site"):
                    longest = (float(bucket.get("max_sec") or 0.0), kind, _site(bucket["max_site"]))
        if parts:
            text += "; holds " + ", ".join(part + (f" ({longest[2]})" if part.startswith(longest[1] + " ") else "") for part in parts)
    queue = row.get("queue_lock")
    if isinstance(queue, dict) and queue.get("count"):
        text += f"; queue {queue.get('count')}/{float(queue.get('hold_total_sec') or 0.0):.1f} s"
    return text


def _route_health(row: Dict[str, Any], today: Any) -> str:
    """The newest ``claudexor_route_health`` row in one line: the engine's window and the routes it saw
    (a harness with its access profile), the routes with failed runs, most failed first, each as
    failed/runs with its top failure group, and the scan's own incompleteness. A window the engine
    counted without a run says so, with its scan's completeness. Counts only: no share, no threshold,
    no verdict. "" for a row without its routes (then no key, never "none failed" or "no engine runs")."""
    raw = row.get("routes")
    routes = [route for route in (raw if isinstance(raw, list) else [])
              if isinstance(route, dict) and type(route.get("runs")) is int and type(route.get("failed")) is int]
    if not isinstance(raw, list) or len(routes) != len(raw):
        return ""
    text = f"{_clock(row.get('generated_at') or row.get('ts'), today)}: {row.get('window_days')}-day window"
    if not routes:
        return text + ", no engine runs" + (", scan complete" if row.get("complete") is True else _scan_note(row))
    text += f", {len(routes)} routes, {sum(route['runs'] for route in routes)} engine runs"
    failing = sorted((route for route in routes if route["failed"] > 0),
                     key=lambda route: (-route["failed"], str(route.get("harness")), str(route.get("access"))))
    if not failing:
        return text + ", none failed" + _scan_note(row)
    parts = []
    for route in failing[:_ROUTES_SHOWN]:
        part = f"{route.get('harness')} {route.get('access')} {route['failed']}/{route['runs']}"
        top = next(iter(route.get("top_failures") or []), None)
        if isinstance(top, dict) and top.get("count") is not None:
            cause = str(top.get("message") or top.get("code") or "")[:_CAUSE_CHARS]
            part += f" ({top.get('category')}{': ' + cause if cause else ''} ×{top['count']})"
        parts.append(part)
    text += "; failed: " + ", ".join(parts)
    if len(failing) > _ROUTES_SHOWN:
        text += f", +{len(failing) - _ROUTES_SHOWN} more routes with failed runs"
    return text + _scan_note(row)


def _scan_note(row: Dict[str, Any]) -> str:
    return f"; scan incomplete ({row.get('incomplete_reason') or 'reason unknown'})" if row.get("complete") is False else ""


def _route_health_line(root: pathlib.Path, today: Any) -> str:
    """The newest route-health row from its own file (``claudexor_exit_facts.latest_route_health``, which
    its writer owns), rendered; "" without a row or for one this line cannot read, so a damaged file
    drops only its own key."""
    from ouroboros.claudexor_exit_facts import latest_route_health

    try:
        row = latest_route_health(root)
        return _route_health(row, today) if isinstance(row, dict) else ""
    except Exception:
        log.debug("route health fact unreadable for the body fact", exc_info=True)
        return ""


def _runtime_facts(root: pathlib.Path) -> Tuple[str, str]:
    """CPython and SQLite as this process runs them with each present store's journal as its header
    says (read once per process: a boot or a handoff is the only writer of that choice), and the WAL
    gate while a store runs a rollback journal ("" otherwise)."""
    import platform
    import sqlite3

    from ouroboros import store

    journal: Dict[str, str] = {}
    for rel in _STORE_FILES:
        path = root / rel
        key = str(path)
        try:
            if key not in _JOURNAL_MODES:
                if not path.is_file():
                    continue
                _JOURNAL_MODES[key] = "wal" if store.wal_file(path) else "rollback"
            journal[path.stem] = _JOURNAL_MODES[key]
        except Exception:
            log.debug("journal mode unreadable: %s", path, exc_info=True)
    runtime = f"python {platform.python_version()}, sqlite {sqlite3.sqlite_version}"
    if journal:
        runtime += ", journal " + ", ".join(f"{name}={mode}" for name, mode in journal.items())
    return runtime, WAL_GATE if "rollback" in journal.values() else ""


def _last_update(root: pathlib.Path) -> str:
    """The retained managed update in one line: outcome and target, the first request's wait before
    and after, RSS before and after (named peak when a side only knew its high-water mark), the duty
    overruns the newest side counted; "" without a retained attempt."""
    from ouroboros.update_generation_facts import projection

    facts = projection(drive_root=root)
    if not isinstance(facts, dict):
        return ""
    sides = [(name, facts[name]) for name in ("before", "after") if isinstance(facts.get(name), dict)]
    text = f"{facts.get('outcome')} {str(facts.get('target_sha') or '')[:12]}"
    if not sides:
        return text
    dispatch = {name: (sample.get("direct_message_first_dispatch") or {}).get("seconds") for name, sample in sides}
    text += ": first request from handler entry " + " → ".join(
        f"{name} {_sec(dispatch[name]) if dispatch[name] is not None else 'none'}" for name, _s in sides)
    if any(sample.get("rss_bytes") is not None for _n, sample in sides):
        text += "; rss " + " → ".join(f"{int(sample['rss_bytes']) // 1048576}" if sample.get("rss_bytes") is not None else "?"
                                      for _n, sample in sides) + " MiB"
        if any(sample.get("rss_measure") == "peak" for _n, sample in sides):
            text += " (peak)"
    overruns = next((sample["duty_overruns"] for _n, sample in reversed(sides) if sample.get("duty_overruns")), None)
    if overruns:
        text += "; duty overruns " + ", ".join(f"{name} {count}" for name, count in overruns.items())
    return text + "; record state/state.json managed_update_attempt"


def vitals_fact(data_root: Any, *, observed_at: str, captured_at: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The body's own dated facts beside ``storage``, one line each: the last owner turns from this serving
    process's receipt ring, or in a pooled worker from the journal's tail (wait, slowest phase, dispatch
    basis when unobserved, own transactions; the top frame of a stall episode whose end row is still in
    the tail), stall onsets whose end was not observed in the journal's bounded tail (never «current»),
    the last busy minute (counts and summed seconds per store and side, the longest hold's site, never a
    percentage), the engine's per-route failure shares (the newest ``claudexor_route_health`` row, kept
    in its own file, which the custody sweep replaces only when they change: failed/runs and the top
    cause per route, most failed first, or a window without runs; no row, no key),
    the runtime (CPython, SQLite, journal per store, the WAL gate while a store runs a
    rollback journal), the retained update's before/after, the boot row, and a note that names the
    journal's reader. ``observed_at`` is the fact's own instant (the clocks' day); it is written only
    when it differs from ``captured_at``, the Runtime JSON's ``context_captured_at`` beside it (a stale
    snapshot), so the same instant is not printed twice. Every absent source omits its key. Never
    raises."""
    import os

    root = pathlib.Path(data_root)
    fact: Dict[str, Any] = {} if observed_at == captured_at else {"observed_at": observed_at}
    try:
        tail = _supervisor_tail(root)
        if os.environ.get("OUROBOROS_IN_WORKER") == "1":  # no ring in a pooled worker: the journal's receipts
            turns, none = tail["turns"], "none in the journal's tail"
        else:
            from supervisor.active_activity import direct_turns

            turns, none = direct_turns(limit=_OWNER_TURNS_SHOWN), "none since this process started"
        if turns:
            fact["owner_turns"] = [_owner_turn(row, tail["ended"], observed_at) for row in turns]
        else:
            fact["owner_turns_note"] = none
        stalls = [_stall(kind, row, observed_at) for (kind, _key), row in tail["open"].items()]
        if stalls:
            fact[_STALLS_KEY] = stalls
        if isinstance(tail.get("load"), dict):
            fact["busy_minute"] = _busy_minute(tail["load"], observed_at)
        routes = _route_health_line(root, observed_at)
        if routes:
            fact["route_health"] = routes
        fact["runtime"], gate = _runtime_facts(root)
        if gate:
            fact["wal_gate"] = gate
        update = _last_update(root)
        if update:
            fact["last_update"] = update
        boot = tail.get("boot")
        if isinstance(boot, dict):
            fact["boot"] = (f"{_clock(boot.get('ts'), observed_at)} {boot.get('branch') or boot.get('git_branch')}@"
                            f"{str(boot.get('sha') or boot.get('git_sha') or '')[:12]}")
        fact["note"] = VITALS_NOTE
    except Exception:  # a fact about the body never breaks the Runtime section
        log.debug("vitals fact unavailable", exc_info=True)
        return None
    return fact


def body_facts(data_root: Any, *, observed_at: str) -> Dict[str, Any]:
    """The Runtime section's ``storage`` and ``vitals`` keys, each present only with its fact: one call
    site, ``context.build_runtime_section``."""
    facts = {"storage": storage_fact(data_root),
             "vitals": vitals_fact(data_root, observed_at=observed_at, captured_at=observed_at)}
    return {key: value for key, value in facts.items() if value is not None}


def vitals_line(fact: Optional[Dict[str, Any]]) -> str:
    """The same facts in one line (a deep self-review's inline rendering); "" for none."""
    if not isinstance(fact, dict):
        return ""
    parts = [f"body as of {fact['observed_at']}" if fact.get("observed_at") else "body"]
    for key in ("owner_turns", "owner_turns_note", _STALLS_KEY, "busy_minute", "route_health", "runtime", "wal_gate",
                "last_update", "boot"):
        value = fact.get(key)
        if isinstance(value, list) and value:
            parts.append(f"{key.replace('_', ' ')} " + " | ".join(str(item) for item in value))
        elif isinstance(value, str) and value:
            parts.append(f"{key.replace('_', ' ')} {value}")
    return "; ".join(parts)


def _delegation_capability_fact() -> Optional[Dict[str, Any]]:
    """Historical identities and readers; execution bodies are chosen reads."""
    try:
        from ouroboros.reviewer_slot_config import reviewer_slot_last_executions, LAST_EXECUTION_FILENAME
        from ouroboros.subagent_history import recorded_handle, LAST_DELEGATION_FILENAME
        from ouroboros.subagents import subagent_last_delegation

        slots = reviewer_slot_last_executions()
        last = subagent_last_delegation() or {}
        if not slots and not last:
            return None
        records = last.get("latest_by_subagent")
        records = list(records.values()) if isinstance(records, dict) and records else ([last] if last else [])
        return {"note": "Historical observations, not live health; read the records for details.",
                "reviewer_slots_last": [{"slot": str(slot), "outcome": row.get("status", "unknown"),
                    "observed": row.get("ts")} for slot, row in sorted(slots.items()) if isinstance(row, dict)],
                "subagents_last_executions": [{"selected_subagent_id": recorded_handle(row),
                    "outcome": row.get("outcome", "unknown"), "observed": row.get("observed_at") or row.get("ts"),
                    **{key: row[key] for key in ("task_id", "run_id", "attempt_id") if row.get(key)}}
                    for row in records if isinstance(row, dict)],
                "readers": [{"tool": "read_file", "arguments": {"root": "runtime_data", "path": "state/" + name}}
                            for name in (LAST_EXECUTION_FILENAME, LAST_DELEGATION_FILENAME)]}
    except Exception:
        log.debug("Failed to build delegation capability fact", exc_info=True)
        return None
