"""
Supervisor — State management.

Persistent state on Google Drive: load, save, atomic writes, file locks.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import pathlib
import time
import uuid
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level config (set via init())
# ---------------------------------------------------------------------------
DRIVE_ROOT: pathlib.Path = pathlib.Path("/content/drive/MyDrive/Ouroboros")
STATE_PATH: pathlib.Path = DRIVE_ROOT / "state" / "state.json"
STATE_LAST_GOOD_PATH: pathlib.Path = DRIVE_ROOT / "state" / "state.last_good.json"
STATE_LOCK_PATH: pathlib.Path = DRIVE_ROOT / "locks" / "state.lock"
QUEUE_SNAPSHOT_PATH: pathlib.Path = DRIVE_ROOT / "state" / "queue_snapshot.json"


def init(drive_root: pathlib.Path, total_budget_limit: float = 0.0) -> None:
    global DRIVE_ROOT, STATE_PATH, STATE_LAST_GOOD_PATH, STATE_LOCK_PATH, QUEUE_SNAPSHOT_PATH
    DRIVE_ROOT = drive_root
    STATE_PATH = drive_root / "state" / "state.json"
    STATE_LAST_GOOD_PATH = drive_root / "state" / "state.last_good.json"
    STATE_LOCK_PATH = drive_root / "locks" / "state.lock"
    QUEUE_SNAPSHOT_PATH = drive_root / "state" / "queue_snapshot.json"
    set_budget_limit(total_budget_limit)


# ---------------------------------------------------------------------------
# Atomic file operations
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# File locks
# ---------------------------------------------------------------------------

def acquire_file_lock(lock_path: pathlib.Path, timeout_sec: float = 4.0,
                      stale_sec: float = 90.0) -> Optional[int]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    while (time.time() - started) < timeout_sec:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            try:
                os.write(fd, f"pid={os.getpid()} ts={datetime.datetime.now(datetime.timezone.utc).isoformat()}\n".encode("utf-8"))
            except Exception:
                log.debug(f"Failed to write lock metadata to {lock_path}", exc_info=True)
                pass
            return fd
        except FileExistsError:
            try:
                age = time.time() - lock_path.stat().st_mtime
                if age > stale_sec:
                    lock_path.unlink()
                    continue
            except Exception:
                log.debug(f"Failed to check/remove stale lock at {lock_path}", exc_info=True)
                pass
            time.sleep(0.05)
        except Exception:
            log.warning(f"Failed to acquire lock at {lock_path}", exc_info=True)
            break
    return None


def release_file_lock(lock_path: pathlib.Path, lock_fd: Optional[int]) -> None:
    if lock_fd is None:
        return
    try:
        os.close(lock_fd)
    except Exception:
        log.debug(f"Failed to close lock fd {lock_fd} for {lock_path}", exc_info=True)
        pass
    try:
        if lock_path.exists():
            lock_path.unlink()
    except Exception:
        log.debug(f"Failed to unlink lock file {lock_path}", exc_info=True)
        pass


# Re-export append_jsonl from ouroboros.utils (single source of truth)
from ouroboros.utils import append_jsonl  # noqa: F401


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

def ensure_state_defaults(st: Dict[str, Any]) -> Dict[str, Any]:
    st.setdefault("created_at", datetime.datetime.now(datetime.timezone.utc).isoformat())
    st.setdefault("owner_id", None)
    st.setdefault("owner_chat_id", None)
    st.setdefault("tg_offset", 0)
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
    st.setdefault("evolution_cycle", 0)
    st.setdefault("session_total_snapshot", None)
    st.setdefault("session_spent_snapshot", None)
    st.setdefault("budget_drift_pct", None)
    st.setdefault("budget_drift_alert", False)
    st.setdefault("evolution_consecutive_failures", 0)
    for legacy_key in ("approvals", "idle_cursor", "idle_stats", "last_idle_task_at",
                        "last_auto_review_at", "last_review_task_id", "session_daily_snapshot"):
        st.pop(legacy_key, None)
    return st


def default_state_dict() -> Dict[str, Any]:
    """Create a fresh state dict. Single source of truth: ensure_state_defaults."""
    return ensure_state_defaults({})


# ---------------------------------------------------------------------------
# Load / Save
# ---------------------------------------------------------------------------

def _load_state_unlocked() -> Dict[str, Any]:
    """Load state without acquiring lock. Caller must hold STATE_LOCK."""
    recovered = False
    st_obj = json_load_file(STATE_PATH)
    if st_obj is None:
        st_obj = json_load_file(STATE_LAST_GOOD_PATH)
        recovered = st_obj is not None

    if st_obj is None:
        st = ensure_state_defaults(default_state_dict())
        _save_state_unlocked(st)
        return st

    st = ensure_state_defaults(st_obj)
    if recovered:
        _save_state_unlocked(st)
    return st


def _save_state_unlocked(st: Dict[str, Any]) -> None:
    """Save state without acquiring lock. Caller must hold STATE_LOCK."""
    st = ensure_state_defaults(st)
    payload = json.dumps(st, ensure_ascii=False, indent=2)
    atomic_write_text(STATE_PATH, payload)
    atomic_write_text(STATE_LAST_GOOD_PATH, payload)


def load_state() -> Dict[str, Any]:
    lock_fd = acquire_file_lock(STATE_LOCK_PATH)
    try:
        return _load_state_unlocked()
    finally:
        release_file_lock(STATE_LOCK_PATH, lock_fd)


def save_state(st: Dict[str, Any]) -> None:
    lock_fd = acquire_file_lock(STATE_LOCK_PATH)
    try:
        _save_state_unlocked(st)
    finally:
        release_file_lock(STATE_LOCK_PATH, lock_fd)


def init_state() -> Dict[str, Any]:
    """
    Initialize state at session start, capturing snapshots for budget drift detection.

    Fetches OpenRouter ground truth and stores session_daily_snapshot and
    session_spent_snapshot for drift calculation.
    """
    lock_fd = acquire_file_lock(STATE_LOCK_PATH)
    try:
        st = _load_state_unlocked()

        # Capture session snapshots for drift detection
        st["session_spent_snapshot"] = float(st.get("spent_usd") or 0.0)

        # Fetch OpenRouter ground truth to capture total_usd baseline
        ground_truth = check_openrouter_ground_truth()
        if ground_truth is not None:
            st["session_total_snapshot"] = ground_truth["total_usd"]
            st["openrouter_total_usd"] = ground_truth["total_usd"]
            st["openrouter_daily_usd"] = ground_truth["daily_usd"]
            st["openrouter_last_check_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        else:
            # If we can't fetch ground truth, use 0 as baseline
            st["session_total_snapshot"] = 0.0

        # Reset drift tracking
        st["budget_drift_pct"] = None
        st["budget_drift_alert"] = False

        _save_state_unlocked(st)
        return st
    finally:
        release_file_lock(STATE_LOCK_PATH, lock_fd)


# ---------------------------------------------------------------------------
# Budget tracking (moved from workers.py)
# ---------------------------------------------------------------------------
TOTAL_BUDGET_LIMIT: float = 0.0
EVOLUTION_BUDGET_RESERVE: float = 50.0  # Stop evolution when remaining < this


def set_budget_limit(limit: float) -> None:
    """Set total budget limit for budget_pct calculation."""
    global TOTAL_BUDGET_LIMIT
    TOTAL_BUDGET_LIMIT = limit


def budget_remaining(st: Dict[str, Any]) -> float:
    """Calculate remaining budget in USD."""
    spent = float(st.get("spent_usd") or 0.0)
    total = float(TOTAL_BUDGET_LIMIT or 0.0)
    if total <= 0:
        return float('inf')  # No limit set
    return max(0.0, total - spent)


def check_openrouter_ground_truth() -> Optional[Dict[str, float]]:
    """
    No-op: Google AI Studio free tier has no billing API to query.
    Kept for API compatibility — always returns $0 usage.
    """
    log.debug("check_openrouter_ground_truth: no-op (using Google AI Studio free tier)")
    return {"total_usd": 0.0, "daily_usd": 0.0}


def budget_pct(st: Dict[str, Any]) -> float:
    """Calculate budget percentage used."""
    spent = float(st.get("spent_usd") or 0.0)
    total = float(TOTAL_BUDGET_LIMIT or 0.0)
    if total <= 0:
        return 0.0
    return (spent / total) * 100.0


def update_budget_from_usage(usage: Dict[str, Any]) -> None:
    """Update state with LLM usage costs and tokens.

    Uses a single lock scope for the read-modify-write cycle to prevent
    concurrent writes from losing budget updates.

    Every 50 calls, fetches OpenRouter ground truth for comparison.
    """
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

    # Step 1: Update budget counters under lock (fast, no I/O beyond Drive)
    lock_fd = acquire_file_lock(STATE_LOCK_PATH)
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

    # Step 2: HTTP to OpenRouter OUTSIDE the lock (can take up to 10s)
    if should_check_ground_truth:
        ground_truth = check_openrouter_ground_truth()
        if ground_truth is not None:
            lock_fd = acquire_file_lock(STATE_LOCK_PATH)
            try:
                st = _load_state_unlocked()
                st["openrouter_total_usd"] = ground_truth["total_usd"]
                st["openrouter_daily_usd"] = ground_truth["daily_usd"]
                st["openrouter_last_check_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

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
                                    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
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


# ---------------------------------------------------------------------------
# Budget breakdown by category
# ---------------------------------------------------------------------------

def budget_breakdown(st: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate budget breakdown by category from events.jsonl.

    Reads llm_usage events and aggregates cost_usd by category field.
    Returns dict like {"task": 12.5, "evolution": 45.2, ...}
    """
    events_path = DRIVE_ROOT / "logs" / "events.jsonl"
    if not events_path.exists():
        return {}

    breakdown: Dict[str, float] = {}
    try:
        with events_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    if event.get("type") != "llm_usage":
                        continue

                    # Get category (default to "other" if not present)
                    category = event.get("category", "other")

                    # Get cost from either top-level "cost" or nested "usage.cost"
                    cost = 0.0
                    if "cost" in event:
                        cost = float(event.get("cost", 0))
                    elif "usage" in event and isinstance(event["usage"], dict):
                        cost = float(event["usage"].get("cost", 0))

                    if cost > 0:
                        breakdown[category] = breakdown.get(category, 0.0) + cost

                except (json.JSONDecodeError, ValueError, TypeError):
                    continue
    except Exception:
        log.warning("Failed to calculate budget breakdown", exc_info=True)

    return breakdown


def model_breakdown(st: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Calculate budget breakdown by model from events.jsonl.

    Returns dict like:
    {
        "anthropic/claude-sonnet-4.6": {"cost": 12.5, "calls": 120, "prompt_tokens": 50000, "completion_tokens": 3000},
        "openai/gpt-4o": {"cost": 3.2, "calls": 15, ...},
    }
    """
    events_path = DRIVE_ROOT / "logs" / "events.jsonl"
    if not events_path.exists():
        return {}

    breakdown: Dict[str, Dict[str, float]] = {}
    try:
        with events_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    if event.get("type") != "llm_usage":
                        continue

                    model = event.get("model") or "unknown"
                    if not model:
                        model = "unknown"

                    # Get cost
                    cost = 0.0
                    if "cost" in event:
                        cost = float(event.get("cost", 0))
                    elif "usage" in event and isinstance(event["usage"], dict):
                        cost = float(event["usage"].get("cost", 0))

                    # Get tokens
                    prompt_tokens = int(event.get("prompt_tokens", 0) or 0)
                    completion_tokens = int(event.get("completion_tokens", 0) or 0)
                    cached_tokens = int(event.get("cached_tokens", 0) or 0)

                    if model not in breakdown:
                        breakdown[model] = {"cost": 0.0, "calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}

                    breakdown[model]["cost"] += cost
                    breakdown[model]["calls"] += 1
                    breakdown[model]["prompt_tokens"] += prompt_tokens
                    breakdown[model]["completion_tokens"] += completion_tokens
                    breakdown[model]["cached_tokens"] += cached_tokens

                except (json.JSONDecodeError, ValueError, TypeError):
                    continue
    except Exception:
        log.warning("Failed to calculate model breakdown", exc_info=True)

    return breakdown


def per_task_cost_summary(max_tasks: int = 10, tail_bytes: int = 512_000) -> List[Dict[str, Any]]:
    """Return cost summary for recent tasks from events.jsonl.

    Only reads the last `tail_bytes` of the file to avoid scanning
    megabytes of history on every LLM round.

    Returns list of dicts: [{task_id, cost, rounds, model}, ...]
    sorted by cost descending, limited to max_tasks.
    """
    events_path = DRIVE_ROOT / "logs" / "events.jsonl"
    if not events_path.exists():
        return []

    tasks: Dict[str, Dict[str, Any]] = {}
    try:
        file_size = events_path.stat().st_size
        with events_path.open("r", encoding="utf-8") as f:
            if file_size > tail_bytes:
                f.seek(file_size - tail_bytes)
                f.readline()  # skip partial first line
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    if event.get("type") != "llm_usage":
                        continue
                    tid = event.get("task_id") or "unknown"
                    cost = float(event.get("cost", 0) or 0)
                    if tid not in tasks:
                        tasks[tid] = {"task_id": tid, "cost": 0.0, "rounds": 0, "model": event.get("model", "")}
                    tasks[tid]["cost"] += cost
                    tasks[tid]["rounds"] += 1
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue
    except Exception:
        log.warning("Failed to calculate per-task cost summary", exc_info=True)

    sorted_tasks = sorted(tasks.values(), key=lambda x: x["cost"], reverse=True)
    return sorted_tasks[:max_tasks]


# ---------------------------------------------------------------------------
# Status text (moved from workers.py)
# ---------------------------------------------------------------------------

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

    # Add budget breakdown by category
    breakdown = budget_breakdown(st)
    if breakdown:
        # Sort by cost descending
        sorted_categories = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)
        breakdown_parts = [f"{cat}=${cost:.2f}" for cat, cost in sorted_categories if cost > 0]
        if breakdown_parts:
            lines.append(f"budget_breakdown: {', '.join(breakdown_parts)}")

    # Display budget drift if available
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

    # Model breakdown
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
    """Rotate chat log if it exceeds max_bytes."""
    chat = drive_root / "logs" / "chat.jsonl"
    if not chat.exists():
        return
    if chat.stat().st_size < max_bytes:
        return
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_path = drive_root / "archive" / f"chat_{ts}.jsonl"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_bytes(chat.read_bytes())
    chat.write_text("", encoding="utf-8")
