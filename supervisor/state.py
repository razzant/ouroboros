"""Supervisor persistent state, atomic writes, locks, and budget accounting."""

from __future__ import annotations

import json
import logging
import os
import pathlib
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
from ouroboros.utils import append_jsonl, iter_llm_usage_events, llm_usage_cost, utc_now_iso

log = logging.getLogger(__name__)


DRIVE_ROOT: pathlib.Path = pathlib.Path.home() / "Ouroboros" / "data"
STATE_PATH: pathlib.Path = DRIVE_ROOT / "state" / "state.json"
STATE_LAST_GOOD_PATH: pathlib.Path = DRIVE_ROOT / "state" / "state.last_good.json"
STATE_LOCK_PATH: pathlib.Path = DRIVE_ROOT / "locks" / "state.lock"
QUEUE_SNAPSHOT_PATH: pathlib.Path = DRIVE_ROOT / "state" / "queue_snapshot.json"

# Explicit marker a benchmark/evolution driver writes into its THROWAWAY data root. A live
# data root (the default ~/Ouroboros/data OR a custom/Drive-backed OUROBOROS_DATA_DIR) never
# has it, so reset_per_task_budget can refuse on it regardless of how the path resolves —
# closing the budget-reset guard for custom-data-root installs (BIBLE P8).
ISOLATED_BENCHMARK_SENTINEL = ".ouroboros_isolated_benchmark"


def init(drive_root: pathlib.Path, total_budget_limit: float = 0.0) -> None:
    global DRIVE_ROOT, STATE_PATH, STATE_LAST_GOOD_PATH, STATE_LOCK_PATH, QUEUE_SNAPSHOT_PATH
    DRIVE_ROOT = drive_root
    STATE_PATH = drive_root / "state" / "state.json"
    STATE_LAST_GOOD_PATH = drive_root / "state" / "state.last_good.json"
    STATE_LOCK_PATH = drive_root / "locks" / "state.lock"
    QUEUE_SNAPSHOT_PATH = drive_root / "state" / "queue_snapshot.json"
    set_budget_limit(total_budget_limit)


def atomic_write_text(path: pathlib.Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        data = content.encode("utf-8")
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(str(tmp), str(path))


def json_load_file(path: pathlib.Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        log.debug(f"Failed to load JSON from {path}", exc_info=True)
        return None


def acquire_file_lock(lock_path: pathlib.Path, timeout_sec: float = 4.0,
                      stale_sec: float = 90.0) -> Optional[int]:
    return acquire_exclusive_file_lock(
        lock_path,
        timeout_sec=timeout_sec,
        stale_sec=stale_sec,
        metadata=f"pid={os.getpid()} ts={utc_now_iso()}\n",
    )


# Direct alias: the platform helper already has the exact signature.
release_file_lock = release_exclusive_file_lock


def ensure_state_defaults(st: Dict[str, Any]) -> Dict[str, Any]:
    st.setdefault("created_at", utc_now_iso())
    st.setdefault("owner_id", None)
    st.setdefault("owner_chat_id", None)
    # Separate slot authorizing owner slash commands from external transports
    # (e.g. Telegram), so the local web owner never locks out a real chat owner.
    st.setdefault("owner_external_id", None)
    st.setdefault("owner_external_chat_id", None)
    st.setdefault("owner_external_bound_at", None)
    st.setdefault("message_offset", 0)
    if "tg_offset" in st:
        st.setdefault("message_offset", st.pop("tg_offset"))
    st.setdefault("spent_usd", 0.0)
    st.setdefault("spent_calls", 0)
    st.setdefault("spent_tokens_prompt", 0)
    st.setdefault("spent_tokens_completion", 0)
    st.setdefault("spent_tokens_cached", 0)
    st.setdefault("session_id", uuid.uuid4().hex)
    st.setdefault("current_branch", None)
    st.setdefault("current_sha", None)
    st.setdefault("last_owner_message_at", "")
    st.setdefault("last_evolution_task_at", "")
    st.setdefault("budget_messages_since_report", 0)
    st.setdefault("evolution_mode_enabled", False)
    # Durable owner-stop sentinel: set True by the owner-stop sites, cleared by an
    # owner-authorized start (/evolve start or the owner-directed toggle_evolution(True)
    # tool). apply_pending_request refuses to autonomously re-arm while True.
    st.setdefault("evolution_owner_stopped", False)
    st.setdefault("evolution_cycle", 0)
    st.setdefault("session_total_snapshot", None)
    st.setdefault("session_spent_snapshot", None)
    st.setdefault("budget_drift_pct", None)
    st.setdefault("budget_drift_alert", False)
    st.setdefault("evolution_consecutive_failures", 0)
    st.setdefault("bg_consciousness_enabled", False)
    for legacy_key in ("approvals", "idle_cursor", "idle_stats", "last_idle_task_at",
                        "last_auto_review_at", "last_review_task_id", "session_daily_snapshot"):
        st.pop(legacy_key, None)
    return st


def _load_state_unlocked() -> Dict[str, Any]:
    """Load state; caller must hold STATE_LOCK."""
    recovered = False
    st_obj = json_load_file(STATE_PATH)
    if st_obj is None:
        st_obj = json_load_file(STATE_LAST_GOOD_PATH)
        recovered = st_obj is not None

    if st_obj is None:
        st = ensure_state_defaults({})
        _save_state_unlocked(st)
        return st

    st = ensure_state_defaults(st_obj)
    if recovered:
        _save_state_unlocked(st)
    return st


def _save_state_unlocked(st: Dict[str, Any]) -> None:
    """Save state; caller must hold STATE_LOCK."""
    st = ensure_state_defaults(st)
    payload = json.dumps(st, ensure_ascii=False, indent=2)
    atomic_write_text(STATE_PATH, payload)
    atomic_write_text(STATE_LAST_GOOD_PATH, payload)


def _warn_state_unlocked(op: str, lock_fd: Optional[int]) -> None:
    """Loud trail when the state lock could not be acquired.

    Proceeding unlocked is a deliberate availability tradeoff (a wedged lock
    must not freeze the supervisor), but it must never be silent: an unlocked
    write is exactly the lost-update class this lock exists to prevent.
    """
    if lock_fd is None:
        log.error("state.json %s proceeding WITHOUT lock (timeout on %s)", op, STATE_LOCK_PATH)


def load_state() -> Dict[str, Any]:
    lock_fd = acquire_file_lock(STATE_LOCK_PATH)
    _warn_state_unlocked("load", lock_fd)
    try:
        return _load_state_unlocked()
    finally:
        release_file_lock(STATE_LOCK_PATH, lock_fd)


def save_state(st: Dict[str, Any]) -> None:
    lock_fd = acquire_file_lock(STATE_LOCK_PATH)
    _warn_state_unlocked("save", lock_fd)
    try:
        _save_state_unlocked(st)
    finally:
        release_file_lock(STATE_LOCK_PATH, lock_fd)


def update_state(mutator) -> Dict[str, Any]:
    """Atomically read-modify-write state under a single held lock.

    Loads the current state, applies ``mutator(st)`` in place, and persists the
    result while holding STATE_LOCK for the WHOLE operation, so concurrent
    updates cannot lose each other (load and save are one critical section — the
    racy ``st = load_state(); st[...] = ...; save_state(st)`` pattern drops the
    other writer's change). Returns the saved state.

    This is also the canonical home of ``update_state`` that
    ``supervisor.events`` imports — it previously lived only in
    ``ouroboros.review_state``, so ``from supervisor.state import update_state``
    raised ImportError (e.g. toggling background consciousness via tool).

    ``mutator`` must NOT call ``load_state``/``save_state``/``update_state`` itself:
    STATE_LOCK is not re-entrant within a process, so re-entering would block.
    """
    lock_fd = acquire_file_lock(STATE_LOCK_PATH)
    _warn_state_unlocked("update", lock_fd)
    try:
        st = _load_state_unlocked()
        mutator(st)
        _save_state_unlocked(st)
        return st
    finally:
        release_file_lock(STATE_LOCK_PATH, lock_fd)


def init_state() -> Dict[str, Any]:
    """Initialize session snapshots for budget drift detection."""
    lock_fd = acquire_file_lock(STATE_LOCK_PATH)
    try:
        st = _load_state_unlocked()

        st["session_spent_snapshot"] = float(st.get("spent_usd") or 0.0)

        ground_truth = check_openrouter_ground_truth()
        if ground_truth is not None:
            st["session_total_snapshot"] = ground_truth["total_usd"]
            st["openrouter_total_usd"] = ground_truth["total_usd"]
            st["openrouter_daily_usd"] = ground_truth["daily_usd"]
            st["openrouter_last_check_at"] = utc_now_iso()
        else:
            st["session_total_snapshot"] = 0.0

        st["budget_drift_pct"] = None
        st["budget_drift_alert"] = False

        _save_state_unlocked(st)
        return st
    finally:
        release_file_lock(STATE_LOCK_PATH, lock_fd)


TOTAL_BUDGET_LIMIT: float = 0.0
EVOLUTION_BUDGET_RESERVE: float = 2.0  # Stop evolution when remaining < this


def set_budget_limit(limit: float) -> None:
    """Set total budget limit for budget_pct."""
    global TOTAL_BUDGET_LIMIT
    TOTAL_BUDGET_LIMIT = limit


def refresh_budget_from_settings(settings: Dict[str, Any]) -> None:
    """Hot-reload TOTAL_BUDGET; bad/missing values mean no limit."""
    try:
        raw = settings.get("TOTAL_BUDGET")
        value = float(raw) if raw is not None else 0.0
        set_budget_limit(value)
    except (TypeError, ValueError):
        pass


def budget_remaining(st: Dict[str, Any]) -> float:
    """Return remaining budget in USD."""
    spent = float(st.get("spent_usd") or 0.0)
    total = float(TOTAL_BUDGET_LIMIT or 0.0)
    if total <= 0:
        return float('inf')
    return max(0.0, total - spent)


def reset_per_task_budget(data_root: Any, *, confirm_isolated: bool = False) -> bool:
    """Zero the per-task budget ledger (spent_usd + call/token counters) in an
    ISOLATED benchmark/evolution data root.

    CRITICAL safety guard (BIBLE P8): the live TOTAL_BUDGET / Emergency-Stop
    contract must never be defeated by a reset. This refuses unless ALL hold:
    the target is NOT the live ``~/Ouroboros/data`` dir, the caller passes
    ``confirm_isolated=True`` (explicit bench intent), and ``OUROBOROS_DATA_DIR``
    is set (a non-default, isolated data dir). Evolutionary drivers call this
    between tasks so each instance starts with a fresh per-task allowance while
    learned knowledge/identity/code carry forward. Returns True only when a reset
    was actually written.
    """
    try:
        target = pathlib.Path(str(data_root)).resolve(strict=False)
    except Exception:
        return False
    live = (pathlib.Path.home() / "Ouroboros" / "data").resolve(strict=False)
    if target == live:
        return False
    if not confirm_isolated:
        return False
    env_dir = str(os.environ.get("OUROBOROS_DATA_DIR", "") or "").strip()
    if not env_dir:
        return False
    try:
        if pathlib.Path(env_dir).resolve(strict=False) != target:
            return False
    except Exception:
        return False
    # Final guard: the target MUST carry the isolated-benchmark sentinel. A live root (default
    # or custom/Drive-backed) never has it, so this reset can never zero a live budget even if
    # the home-path comparison above does not match a non-default live data root (BIBLE P8).
    if not (target / ISOLATED_BENCHMARK_SENTINEL).exists():
        return False
    state_path = target / "state" / "state.json"
    # Lock on the TARGET root's own state.lock (the isolated server holds the same
    # path as its STATE_LOCK), so this between-instance reset and a concurrent server
    # save_state cannot lost-update each other in the B-full server-driven model.
    lock_path = target / "locks" / "state.lock"
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    lock_fd = acquire_file_lock(lock_path)
    if lock_fd is None:
        # Lock acquisition timed out (the isolated server is actively writing state):
        # skip rather than run an UNLOCKED read-modify-write that would race save_state.
        log.warning("reset_per_task_budget: could not acquire state lock for %s; skipping reset", state_path)
        return False
    budget_keys = ("spent_usd", "spent_calls", "spent_tokens_prompt",
                   "spent_tokens_completion", "spent_tokens_cached")
    try:
        st = json_load_file(state_path) or {}
        st["spent_usd"] = 0.0
        st["spent_calls"] = 0
        st["spent_tokens_prompt"] = 0
        st["spent_tokens_completion"] = 0
        st["spent_tokens_cached"] = 0
        atomic_write_text(state_path, json.dumps(st, ensure_ascii=False, indent=2))
        # Also zero the budget counters in the last-good snapshot. _load_state
        # falls back to it when state.json is missing/corrupt; leaving stale
        # spend there could re-inflate the per-task ledger after a mid-run
        # crash+recovery, defeating the reset (the kit reset both files).
        lg_path = target / "state" / "state.last_good.json"
        lg = json_load_file(lg_path)
        if isinstance(lg, dict):
            for key in budget_keys:
                if key in lg:
                    lg[key] = 0 if key != "spent_usd" else 0.0
            atomic_write_text(lg_path, json.dumps(lg, ensure_ascii=False, indent=2))
    except Exception:
        log.warning("reset_per_task_budget: failed to write %s", state_path, exc_info=True)
        return False
    finally:
        release_file_lock(lock_path, lock_fd)
    return True


def check_openrouter_ground_truth() -> Optional[Dict[str, float]]:
    """Return OpenRouter total/daily usage, or None on error."""
    try:
        import urllib.request
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            return None
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        # OpenRouter usage is dollars, not cents.
        usage_total = data.get("data", {}).get("usage", 0)
        usage_daily = data.get("data", {}).get("usage_daily", 0)
        return {
            "total_usd": float(usage_total),
            "daily_usd": float(usage_daily),
        }
    except Exception:
        log.warning("Failed to fetch OpenRouter ground truth", exc_info=True)
        return None


def budget_pct(st: Dict[str, Any]) -> float:
    """Return budget percent used."""
    spent = float(st.get("spent_usd") or 0.0)
    total = float(TOTAL_BUDGET_LIMIT or 0.0)
    if total <= 0:
        return 0.0
    return (spent / total) * 100.0


def update_budget_from_usage(usage: Dict[str, Any]) -> None:
    """Update LLM cost/token counters and periodically compare OpenRouter truth."""
    def _to_float(v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            log.debug(f"Failed to convert value to float: {v!r}", exc_info=True)
            return default

    def _to_int(v: Any, default: int = 0) -> int:
        try:
            return int(v)
        except Exception:
            log.debug(f"Failed to convert value to int: {v!r}", exc_info=True)
            return default

    # Keep the lock around local counters only; network check runs outside it.
    lock_fd = acquire_file_lock(STATE_LOCK_PATH)
    _warn_state_unlocked("budget-update", lock_fd)
    try:
        st = _load_state_unlocked()
        cost = usage.get("cost") if isinstance(usage, dict) else None
        if cost is None:
            cost = 0.0
        st["spent_usd"] = _to_float(st.get("spent_usd") or 0.0) + _to_float(cost)
        rounds = _to_int(usage.get("rounds") if isinstance(usage, dict) else 0, default=1)
        st["spent_calls"] = int(st.get("spent_calls") or 0) + rounds
        st["spent_tokens_prompt"] = _to_int(st.get("spent_tokens_prompt") or 0) + _to_int(
            usage.get("prompt_tokens") if isinstance(usage, dict) else 0)
        st["spent_tokens_completion"] = _to_int(st.get("spent_tokens_completion") or 0) + _to_int(
            usage.get("completion_tokens") if isinstance(usage, dict) else 0)
        st["spent_tokens_cached"] = _to_int(st.get("spent_tokens_cached") or 0) + _to_int(
            usage.get("cached_tokens") if isinstance(usage, dict) else 0)
        should_check_ground_truth = (st["spent_calls"] % 50 == 0)
        _save_state_unlocked(st)
    finally:
        release_file_lock(STATE_LOCK_PATH, lock_fd)

    if should_check_ground_truth:
        ground_truth = check_openrouter_ground_truth()
        if ground_truth is not None:
            lock_fd = acquire_file_lock(STATE_LOCK_PATH)
            try:
                st = _load_state_unlocked()
                st["openrouter_total_usd"] = ground_truth["total_usd"]
                st["openrouter_daily_usd"] = ground_truth["daily_usd"]
                st["openrouter_last_check_at"] = utc_now_iso()

                session_total_snap = st.get("session_total_snapshot")
                session_spent_snap = st.get("session_spent_snapshot")

                if session_total_snap is not None and session_spent_snap is not None:
                    current_total_usd = ground_truth["total_usd"]
                    current_spent_usd = _to_float(st.get("spent_usd") or 0.0)
                    or_delta = current_total_usd - _to_float(session_total_snap)
                    our_delta = current_spent_usd - _to_float(session_spent_snap)

                    if or_delta > 0.001:
                        drift_pct = abs(or_delta - our_delta) / max(abs(or_delta), 0.01) * 100.0
                        st["budget_drift_pct"] = drift_pct
                        abs_diff = abs(or_delta - our_delta)
                        if drift_pct > 50.0 and abs_diff > 5.0:
                            st["budget_drift_alert"] = True
                            append_jsonl(
                                DRIVE_ROOT / "logs" / "events.jsonl",
                                {
                                    "ts": utc_now_iso(),
                                    "event": "budget_drift_warning",
                                    "drift_pct": round(drift_pct, 2),
                                    "our_delta": round(our_delta, 4),
                                    "or_delta": round(or_delta, 4),
                                    "abs_diff": round(abs_diff, 4),
                                    "spent_calls": st["spent_calls"],
                                    "note": "High drift expected if OR key is shared or tracking had early bugs",
                                }
                            )
                        else:
                            st["budget_drift_alert"] = False
                    else:
                        st["budget_drift_pct"] = 0.0
                        st["budget_drift_alert"] = False

                _save_state_unlocked(st)
            finally:
                release_file_lock(STATE_LOCK_PATH, lock_fd)


def budget_breakdown(st: Dict[str, Any]) -> Dict[str, float]:
    """Aggregate llm_usage cost by category from events.jsonl."""
    events_path = DRIVE_ROOT / "logs" / "events.jsonl"
    breakdown: Dict[str, float] = {}
    try:
        for event in iter_llm_usage_events(events_path):
            category = event.get("category", "other")
            cost = llm_usage_cost(event)
            if cost > 0:
                breakdown[category] = breakdown.get(category, 0.0) + cost
    except Exception:
        log.warning("Failed to calculate budget breakdown", exc_info=True)

    return breakdown


def model_breakdown(st: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Aggregate llm_usage cost/calls/tokens by model."""
    events_path = DRIVE_ROOT / "logs" / "events.jsonl"
    breakdown: Dict[str, Dict[str, float]] = {}
    try:
        for event in iter_llm_usage_events(events_path):
            model = event.get("model") or "unknown"
            stats = breakdown.setdefault(model, {
                "cost": 0.0, "calls": 0, "prompt_tokens": 0,
                "completion_tokens": 0, "cached_tokens": 0,
            })
            stats["cost"] += llm_usage_cost(event)
            stats["calls"] += 1
            for key in ("prompt_tokens", "completion_tokens", "cached_tokens"):
                try:
                    stats[key] += int(event.get(key, 0) or 0)
                except (TypeError, ValueError):
                    pass
    except Exception:
        log.warning("Failed to calculate model breakdown", exc_info=True)

    return breakdown


def per_task_cost_summary(max_tasks: int = 10, tail_bytes: int = 512_000) -> List[Dict[str, Any]]:
    """Return recent task cost summary from the tail of events.jsonl."""
    events_path = DRIVE_ROOT / "logs" / "events.jsonl"
    tasks: Dict[str, Dict[str, Any]] = {}
    try:
        for event in iter_llm_usage_events(events_path, tail_bytes=tail_bytes):
            tid = event.get("task_id") or "unknown"
            task = tasks.setdefault(
                tid,
                {"task_id": tid, "cost": 0.0, "rounds": 0, "model": event.get("model", "")},
            )
            task["cost"] += llm_usage_cost(event)
            task["rounds"] += 1
    except Exception:
        log.warning("Failed to calculate per-task cost summary", exc_info=True)

    sorted_tasks = sorted(tasks.values(), key=lambda x: x["cost"], reverse=True)
    return sorted_tasks[:max_tasks]


def reconstruct_task_cost(task_id: str) -> Tuple[float, int, int, int]:
    """Reconstruct ``(cost_usd, rounds, prompt_tokens, completion_tokens)`` for a
    task from its durable ``llm_usage`` events.

    On abnormal termination (hard-timeout kill, cancel, worker loss) the worker is
    SIGKILLed before normal finalization aggregates cost, so the terminal event
    carries zeros — which silently understates per-task rollups, the evolution
    campaign tally, and the failure heuristic. The per-round ``llm_usage`` rows in
    ``events.jsonl`` are the budget SSOT (global ``spent_usd`` is summed from them)
    and are already durable before any kill, so we re-derive the per-task totals
    from them here. Full-scan (not tail): a long task's early rounds can sit far
    behind the file tail.
    """
    want = str(task_id or "")
    cost = 0.0
    rounds = 0
    prompt_tokens = 0
    completion_tokens = 0
    if not want:
        return (0.0, 0, 0, 0)
    events_path = DRIVE_ROOT / "logs" / "events.jsonl"
    try:
        for event in iter_llm_usage_events(events_path):
            if str(event.get("task_id") or "") != want:
                continue
            usage = event.get("usage") if isinstance(event.get("usage"), dict) else {}
            rounds += 1
            cost += llm_usage_cost(event)
            try:
                prompt_tokens += int(event.get("prompt_tokens") or usage.get("prompt_tokens") or 0)
            except (TypeError, ValueError):
                pass
            try:
                completion_tokens += int(event.get("completion_tokens") or usage.get("completion_tokens") or 0)
            except (TypeError, ValueError):
                pass
    except Exception:
        log.warning("Failed to reconstruct task cost for %s", task_id, exc_info=True)
    return (round(cost, 6), rounds, prompt_tokens, completion_tokens)


def status_text(workers_dict: Dict[int, Any], pending_list: list, running_dict: Dict[str, Dict[str, Any]],
                soft_timeout_sec: int, hard_timeout_sec: int) -> str:
    """Build status text from worker and queue state."""
    st = load_state()
    now = time.time()
    lines = []
    lines.append(f"owner_id: {st.get('owner_id')}")
    lines.append(f"session_id: {st.get('session_id')}")
    lines.append(f"version: {st.get('current_branch')}@{(st.get('current_sha') or '')[:8]}")
    busy_count = sum(1 for w in workers_dict.values() if getattr(w, 'busy_task_id', None) is not None)
    lines.append(f"workers: {len(workers_dict)} (busy: {busy_count})")
    lines.append(f"pending: {len(pending_list)}")
    lines.append(f"running: {len(running_dict)}")
    if pending_list:
        preview = []
        for t in pending_list[:10]:
            preview.append(
                f"{t.get('id')}:{t.get('type')}:pr{t.get('priority')}:a{int(t.get('_attempt') or 1)}")
        lines.append("pending_queue: " + ", ".join(preview))
    if running_dict:
        lines.append("running_ids: " + ", ".join(list(running_dict.keys())[:10]))
    busy = [f"{getattr(w, 'wid', '?')}:{getattr(w, 'busy_task_id', '?')}"
            for w in workers_dict.values() if getattr(w, 'busy_task_id', None)]
    if busy:
        lines.append("busy: " + ", ".join(busy))
    if running_dict:
        details = []
        for task_id, meta in list(running_dict.items())[:10]:
            task = meta.get("task") if isinstance(meta, dict) else {}
            started = float(meta.get("started_at") or 0.0) if isinstance(meta, dict) else 0.0
            hb = float(meta.get("last_heartbeat_at") or 0.0) if isinstance(meta, dict) else 0.0
            runtime_sec = int(max(0.0, now - started)) if started > 0 else 0
            hb_lag_sec = int(max(0.0, now - hb)) if hb > 0 else -1
            details.append(
                f"{task_id}:type={task.get('type')} pr={task.get('priority')} "
                f"attempt={meta.get('attempt')} runtime={runtime_sec}s hb_lag={hb_lag_sec}s")
        if details:
            lines.append("running_details:")
            lines.extend([f"  - {d}" for d in details])
    if running_dict and busy_count == 0:
        lines.append("queue_warning: running>0 while busy=0")
    spent = float(st.get("spent_usd") or 0.0)
    pct = budget_pct(st)
    budget_remaining_usd = max(0, TOTAL_BUDGET_LIMIT - spent)
    lines.append(f"budget_total: ${TOTAL_BUDGET_LIMIT:.0f}")
    lines.append(f"budget_remaining: ${budget_remaining_usd:.0f}")
    if pct > 0:
        lines.append(f"spent_usd: ${spent:.2f} ({pct:.1f}% of budget)")
    else:
        lines.append(f"spent_usd: ${spent:.2f}")
    lines.append(f"spent_calls: {st.get('spent_calls')}")
    lines.append(f"prompt_tokens: {st.get('spent_tokens_prompt')}, completion_tokens: {st.get('spent_tokens_completion')}, cached_tokens: {st.get('spent_tokens_cached')}")

    breakdown = budget_breakdown(st)
    if breakdown:
        sorted_categories = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)
        breakdown_parts = [f"{cat}=${cost:.2f}" for cat, cost in sorted_categories if cost > 0]
        if breakdown_parts:
            lines.append(f"budget_breakdown: {', '.join(breakdown_parts)}")

    drift_pct = st.get("budget_drift_pct")
    if drift_pct is not None:
        session_total_snap = st.get("session_total_snapshot")
        session_spent_snap = st.get("session_spent_snapshot")
        or_total = st.get("openrouter_total_usd")

        if session_total_snap is not None and session_spent_snap is not None and or_total is not None:
            or_delta = or_total - session_total_snap
            our_delta = spent - session_spent_snap

            drift_icon = " ⚠️" if st.get("budget_drift_alert") else ""
            lines.append(
                f"budget_drift: {drift_pct:.1f}%{drift_icon} "
                f"(tracked: ${our_delta:.2f} vs OpenRouter: ${or_delta:.2f})"
            )

    models = model_breakdown(st)
    if models:
        sorted_models = sorted(models.items(), key=lambda x: x[1]["cost"], reverse=True)
        lines.append("model_breakdown:")
        for model_name, stats in sorted_models:
            if stats["cost"] > 0 or stats["calls"] > 0:
                cost = stats["cost"]
                calls = int(stats["calls"])
                pt = int(stats["prompt_tokens"])
                ct = int(stats["completion_tokens"])
                lines.append(f"  {model_name}: ${cost:.2f} ({calls} calls, {pt:,}p/{ct:,}c tok)")

    lines.append(
        "evolution: "
        + f"enabled={int(bool(st.get('evolution_mode_enabled')))}, "
        + f"cycle={int(st.get('evolution_cycle') or 0)}")
    lines.append(f"last_owner_message_at: {st.get('last_owner_message_at') or '-'}")
    lines.append(f"timeouts: soft={soft_timeout_sec}s, hard={hard_timeout_sec}s")
    return "\n".join(lines)


def rotate_chat_log_if_needed(drive_root: pathlib.Path, max_bytes: int = 800_000) -> None:
    """Rotate chat log if it exceeds max_bytes.

    Rotation is an atomic ``os.replace`` rename performed under the SAME
    sidecar lock that ``append_jsonl`` writers take — the old copy+truncate
    destroyed any line appended between the read and the truncate.
    """
    chat = drive_root / "logs" / "chat.jsonl"
    if not chat.exists():
        return
    if chat.stat().st_size < max_bytes:
        return
    ts = utc_now_iso().replace("-", "").replace(":", "").split(".")[0]
    archive_path = drive_root / "archive" / f"chat_{ts}.jsonl"
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    from ouroboros.utils import jsonl_append_lock_path

    lock_path = jsonl_append_lock_path(chat)
    lock_fd = acquire_exclusive_file_lock(lock_path, timeout_sec=2.0, stale_sec=10.0)
    if lock_fd is None:
        log.warning("chat.jsonl rotation skipped: append lock busy")
        return
    try:
        if not chat.exists() or chat.stat().st_size < max_bytes:
            return
        os.replace(chat, archive_path)
        chat.touch()
    finally:
        release_exclusive_file_lock(lock_path, lock_fd)
