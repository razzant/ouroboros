"""Supervisor persistent state, atomic writes, locks, and budget accounting."""

from __future__ import annotations

import json
import logging
import os
import pathlib
import time
import uuid
from typing import Any, Dict, List, Optional

from ouroboros.config import DATA_DIR
from ouroboros.contracts.schema_versions import SCHEMA_VERSION_KEY
from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
from ouroboros.utils import append_jsonl, assert_test_data_path, utc_now_iso, write_bytes_atomic  # noqa: F401 -- assert_test_data_path re-exported: the guard moved to the utils leaf so append_jsonl shares it; state.assert_test_data_path callers keep working

log = logging.getLogger(__name__)

# ABI 7.0 (Q8=B): the durable ``state.json`` snapshot names its schema on
# every write. Stamp-on-write ONLY — readers do not require the stamp (no
# compat branching), so a pre-7.0 file and a post-rollback stamped file both
# load unchanged.
STATE_SCHEMA_VERSION = 1


DRIVE_ROOT: pathlib.Path = pathlib.Path(DATA_DIR)
STATE_PATH: pathlib.Path = DRIVE_ROOT / "state" / "state.json"
STATE_LAST_GOOD_PATH: pathlib.Path = DRIVE_ROOT / "state" / "state.last_good.json"
STATE_LOCK_PATH: pathlib.Path = DRIVE_ROOT / "locks" / "state.lock"

# Explicit marker a benchmark/evolution driver writes into its THROWAWAY data root. A live
# data root (the default ~/Ouroboros/data OR a custom/Drive-backed OUROBOROS_DATA_DIR) never
# has it, so reset_per_task_budget can refuse on it regardless of how the path resolves —
# closing the budget-reset guard for custom-data-root installs (BIBLE P8).
ISOLATED_BENCHMARK_SENTINEL = ".ouroboros_isolated_benchmark"


def init(drive_root: pathlib.Path, total_budget_limit: float = 0.0) -> None:
    global DRIVE_ROOT, STATE_PATH, STATE_LAST_GOOD_PATH, STATE_LOCK_PATH
    DRIVE_ROOT = drive_root
    STATE_PATH = drive_root / "state" / "state.json"
    STATE_LAST_GOOD_PATH = drive_root / "state" / "state.last_good.json"
    STATE_LOCK_PATH = drive_root / "locks" / "state.lock"
    set_budget_limit(total_budget_limit)


def atomic_write_text(path: pathlib.Path, content: str) -> None:
    """Durable state write: byte-exact, fsync'd, every byte landed.

    Rides the utils atomic SSOT — its write loop survives a short ``os.write``
    (the old single call could publish a truncated ``state.json`` behind a
    successful rename), and the pytest live-data guard now sits on that same
    seam, so every writer through it is guarded rather than this one alone.
    """
    write_bytes_atomic(path, content.encode("utf-8"), fsync=True)


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
    # Drift compares like with like: the OpenRouter-only settled ledger total vs
    # the queried OpenRouter key's usage. The all-provider spent_usd delta is NOT
    # comparable once direct-provider lanes (e.g. Anthropic advisory) carry real
    # spend — that shape kept budget_drift_alert latched at ~88% while nothing
    # was wrong with the ledger.
    st.setdefault("session_openrouter_settled_snapshot", None)
    st.setdefault("session_openrouter_key_fp", "")
    st.setdefault("openrouter_ledger_settled_usd", None)
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
    st[SCHEMA_VERSION_KEY] = STATE_SCHEMA_VERSION
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
    assert_test_data_path(STATE_PATH)
    lock_fd = acquire_file_lock(STATE_LOCK_PATH)
    _warn_state_unlocked("load", lock_fd)
    try:
        return _load_state_unlocked()
    finally:
        release_file_lock(STATE_LOCK_PATH, lock_fd)


def save_state(st: Dict[str, Any]) -> None:
    assert_test_data_path(STATE_PATH)
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
    assert_test_data_path(STATE_PATH)
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
    assert_test_data_path(STATE_PATH)
    lock_fd = acquire_file_lock(STATE_LOCK_PATH)
    try:
        st = _load_state_unlocked()

        st["session_spent_snapshot"] = float(st.get("spent_usd") or 0.0)
        or_settled = _openrouter_ledger_settled()
        st["session_openrouter_settled_snapshot"] = or_settled
        st["openrouter_ledger_settled_usd"] = or_settled
        st["session_openrouter_key_fp"] = _openrouter_key_fingerprint()

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


def budget_remaining(
    st: Dict[str, Any],
    *,
    strict: bool = False,
    projection: Optional[Dict[str, Any]] = None,
) -> float:
    """Return ledger-derived remaining budget in USD.

    ``state.json`` is only a compatibility projection.  A corrupt or
    unavailable monetary ledger fails closed while a configured limit is in
    force, so the supervisor cannot dispatch against stale counters.

    ``projection`` is an optional pre-computed global usage projection (same
    ``global_limit_usd`` and drive root as this function would use itself) so a
    caller that already replayed the ledger — e.g. ``/api/state`` — does not
    trigger a second replay. It is accepted only when its ``limit_usd`` equals
    the limit this function reads itself (``round(max(0.0, total), 6)``, the
    exact value ``usage_projection`` stamps); a mismatch — e.g. a settings
    hot-reload between the caller's computation and this call, or a caller
    passing a projection built for a different limit — falls through to
    self-computation. Default ``None`` preserves the exact prior behavior,
    including the strict fail-closed path.
    """
    total = float(TOTAL_BUDGET_LIMIT or 0.0)
    if total <= 0:
        return float('inf')
    if projection is not None and projection.get("limit_usd") != round(max(0.0, total), 6):
        projection = None
    try:
        if projection is None:
            from ouroboros.usage_accounting import ensure_legacy_imported, usage_projection

            ensure_legacy_imported(DRIVE_ROOT)
            projection = usage_projection(DRIVE_ROOT, global_limit_usd=total)
        return float(projection.get("remaining_known_usd") or 0.0)
    except Exception:
        log.exception("Budget ledger unavailable; refusing new model dispatch")
        if strict:
            raise
        return 0.0


def reset_per_task_budget(data_root: Any, *, confirm_isolated: bool = False) -> bool:
    """Zero legacy budget *projection* fields in an isolated benchmark root.

    The append-only physical-attempt ledger is deliberately untouched.  New
    runs receive their allowance through a new root task id/root limit, while a
    campaign limit continues to cover every prior physical attempt.  This
    compatibility helper only prevents stale pre-ledger state fields from being
    mistaken for a current per-task counter by old benchmark tooling.

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


def _openrouter_key_fingerprint() -> str:
    """Non-secret identity of the currently configured OpenRouter key.

    Drift comparison is only meaningful while the ledger baseline and the
    ``/auth/key`` ground truth describe the SAME key; a settings hot-reload can
    swap the key mid-session. Returns a short sha256 prefix (never the key)."""
    import hashlib

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return ""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]


def _openrouter_ledger_settled(breakdown: Optional[Dict[str, Any]] = None) -> Optional[float]:
    """Cumulative settled USD attributed to provider=openrouter in the attempt
    ledger, or None when the ledger is unavailable. Settled-only on purpose:
    reservations/unresolved bounds are conservative estimates and would inflate
    the tracked side of the drift comparison."""
    try:
        if breakdown is None:
            from ouroboros.usage_accounting import ensure_legacy_imported, usage_breakdown

            ensure_legacy_imported(DRIVE_ROOT)
            breakdown = usage_breakdown(DRIVE_ROOT)
        bucket = dict(breakdown.get("by_provider") or {}).get("openrouter") or {}
        return float(bucket.get("settled_usd") or 0.0)
    except Exception:
        log.debug("OpenRouter ledger settled total unavailable", exc_info=True)
        return None


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
    """Return ledger-derived budget percent used."""
    total = float(TOTAL_BUDGET_LIMIT or 0.0)
    if total <= 0:
        return 0.0
    try:
        from ouroboros.usage_accounting import ensure_legacy_imported, usage_projection

        ensure_legacy_imported(DRIVE_ROOT)
        projection = usage_projection(DRIVE_ROOT, global_limit_usd=total)
        return (float(projection.get("accounted_usd") or 0.0) / total) * 100.0
    except Exception:
        log.exception("Budget ledger unavailable while calculating percent")
        return 100.0


def update_budget_from_usage(usage: Dict[str, Any]) -> None:
    """Refresh the legacy state projection from the physical-attempt ledger.

    ``usage`` is retained for caller compatibility but is never added to the
    monetary total: every core-mediated provider attempt has already been
    persisted by the transport wrapper.  This prevents logical usage events,
    retries, and review aggregation from charging the same attempt twice.
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

    from ouroboros.usage_accounting import ensure_legacy_imported, usage_breakdown, usage_projection

    # Serialize the ledger snapshot with its compatibility write. Otherwise an
    # older concurrent reader can acquire STATE_LOCK later and regress state.json.
    lock_fd = acquire_file_lock(STATE_LOCK_PATH)
    _warn_state_unlocked("budget-update", lock_fd)
    try:
        ensure_legacy_imported(DRIVE_ROOT)
        breakdown = usage_breakdown(DRIVE_ROOT)
        total_limit = float(TOTAL_BUDGET_LIMIT or 0.0)
        projection = (
            usage_projection(DRIVE_ROOT, global_limit_usd=total_limit)
            if total_limit > 0
            else {key: breakdown.get(key) for key in (
                "settled_usd", "confirmed_usd", "estimated_usd", "reserved_usd",
                "unresolved_upper_bound_usd", "accounted_usd", "unknown_unmetered",
                "cost_final", "attempt_counts", "integrity_degraded",
            )}
        )
        st = _load_state_unlocked()
        st["spent_usd"] = _to_float(breakdown.get("accounted_usd"))
        st["spent_calls"] = _to_int(breakdown.get("physical_calls"))
        st["spent_tokens_prompt"] = _to_int(breakdown.get("prompt_tokens"))
        st["spent_tokens_completion"] = _to_int(breakdown.get("completion_tokens"))
        st["spent_tokens_cached"] = _to_int(breakdown.get("cached_tokens"))
        st["usage_accounting"] = projection
        st["openrouter_ledger_settled_usd"] = _openrouter_ledger_settled(breakdown)
        previous_check_call = _to_int(st.get("openrouter_last_check_call"), -1)
        should_check_ground_truth = bool(
            st["spent_calls"] > 0
            and st["spent_calls"] % 50 == 0
            and st["spent_calls"] != previous_check_call
        )
        if should_check_ground_truth:
            st["openrouter_last_check_call"] = st["spent_calls"]
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

                # Drift compares the OpenRouter-only settled ledger delta with the
                # queried OpenRouter key's usage delta — the only like-for-like
                # pair. Direct-provider spend (Anthropic/OpenAI/local) is invisible
                # to /auth/key by construction and must not count as "drift".
                session_total_snap = st.get("session_total_snapshot")
                session_or_settled_snap = st.get("session_openrouter_settled_snapshot")
                or_ledger_settled = st.get("openrouter_ledger_settled_usd")
                current_fp = _openrouter_key_fingerprint()
                baseline_fp = str(st.get("session_openrouter_key_fp") or "")
                integrity_degraded = bool(breakdown.get("integrity_degraded"))
                key_changed = bool(current_fp) and bool(baseline_fp) and current_fp != baseline_fp

                if integrity_degraded:
                    # A quarantined ledger tail makes the tracked side non-final;
                    # a confident percentage would be dishonest. Comparison is
                    # suppressed, not zeroed.
                    st["budget_drift_pct"] = None
                    st["budget_drift_alert"] = False
                elif (
                    key_changed
                    or session_total_snap is None
                    or session_or_settled_snap is None
                    or or_ledger_settled is None
                ):
                    # Rebaseline (key swapped mid-session, or pre-upgrade state
                    # lacks the OpenRouter-only snapshot) and skip this cycle:
                    # the old baseline describes a different key/metric.
                    st["session_total_snapshot"] = ground_truth["total_usd"]
                    st["session_openrouter_settled_snapshot"] = or_ledger_settled
                    st["session_openrouter_key_fp"] = current_fp
                    st["budget_drift_pct"] = None
                    st["budget_drift_alert"] = False
                else:
                    or_delta = ground_truth["total_usd"] - _to_float(session_total_snap)
                    our_delta = _to_float(or_ledger_settled) - _to_float(session_or_settled_snap)

                    if or_delta > 0.001:
                        drift_pct = abs(or_delta - our_delta) / max(abs(or_delta), 0.01) * 100.0
                        st["budget_drift_pct"] = drift_pct
                        abs_diff = abs(or_delta - our_delta)
                        if drift_pct > 50.0 and abs_diff > 5.0:
                            st["budget_drift_alert"] = True
                            all_provider_delta = _to_float(st.get("spent_usd") or 0.0) - _to_float(
                                st.get("session_spent_snapshot") or 0.0
                            )
                            append_jsonl(
                                DRIVE_ROOT / "logs" / "events.jsonl",
                                {
                                    "ts": utc_now_iso(),
                                    # "type" is the events.jsonl schema key every other
                                    # event uses; type-keyed aggregations used to lose
                                    # this row when it was written under "event".
                                    "type": "budget_drift_warning",
                                    "drift_pct": round(drift_pct, 2),
                                    "our_delta": round(our_delta, 4),
                                    "or_delta": round(or_delta, 4),
                                    "abs_diff": round(abs_diff, 4),
                                    "all_provider_delta": round(all_provider_delta, 4),
                                    "spent_calls": st["spent_calls"],
                                    "note": (
                                        "OpenRouter-only ledger delta vs /auth/key usage delta. "
                                        "High drift usually means a shared OR key or missing ledger rows."
                                    ),
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
    """Aggregate accounted physical-attempt cost by category."""
    breakdown: Dict[str, float] = {}
    try:
        from ouroboros.usage_accounting import ensure_legacy_imported, usage_breakdown

        ensure_legacy_imported(DRIVE_ROOT)
        ledger = usage_breakdown(DRIVE_ROOT)
        for category, bucket in dict(ledger.get("by_category") or {}).items():
            breakdown[str(category)] = float(bucket.get("accounted_usd") or 0.0)
        unattributed = dict(ledger.get("unattributed") or {}).get("category") or {}
        if float(unattributed.get("accounted_usd") or 0.0) > 0:
            breakdown["(unattributed)"] = float(unattributed.get("accounted_usd") or 0.0)
    except Exception:
        log.error("Failed to calculate ledger budget breakdown", exc_info=True)

    return breakdown


def model_breakdown(st: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Aggregate physical calls/tokens/accounted cost by model."""
    breakdown: Dict[str, Dict[str, float]] = {}
    try:
        from ouroboros.usage_accounting import ensure_legacy_imported, usage_breakdown

        ensure_legacy_imported(DRIVE_ROOT)
        ledger = usage_breakdown(DRIVE_ROOT)
        buckets = dict(ledger.get("by_model") or {})
        unattributed = dict(ledger.get("unattributed") or {}).get("model") or {}
        if int(unattributed.get("physical_calls") or 0) or float(unattributed.get("accounted_usd") or 0.0):
            buckets["(unattributed)"] = unattributed
        for model, bucket in buckets.items():
            breakdown[str(model)] = {
                "cost": float(bucket.get("accounted_usd") or 0.0),
                "calls": int(bucket.get("physical_calls") or 0),
                "prompt_tokens": int(bucket.get("prompt_tokens") or 0),
                "completion_tokens": int(bucket.get("completion_tokens") or 0),
                "cached_tokens": int(bucket.get("cached_tokens") or 0),
            }
    except Exception:
        log.error("Failed to calculate ledger model breakdown", exc_info=True)

    return breakdown


def per_task_cost_summary(max_tasks: int = 10, tail_bytes: int = 512_000) -> List[Dict[str, Any]]:
    """Return task cost summary from ledger-attributed physical attempts."""
    del tail_bytes  # compatibility-only; the append-only ledger is replayed in full
    tasks: Dict[str, Dict[str, Any]] = {}
    try:
        from ouroboros.usage_accounting import ensure_legacy_imported, usage_breakdown

        ensure_legacy_imported(DRIVE_ROOT)
        ledger = usage_breakdown(DRIVE_ROOT)
        for task_id, bucket in dict(ledger.get("by_task") or {}).items():
            tasks[str(task_id)] = {
                "task_id": str(task_id),
                "cost": float(bucket.get("accounted_usd") or 0.0),
                "rounds": int(bucket.get("physical_calls") or 0),
                "model": "",
            }
    except Exception:
        log.error("Failed to calculate ledger per-task cost summary", exc_info=True)
        raise

    sorted_tasks = sorted(tasks.values(), key=lambda x: x["cost"], reverse=True)
    return sorted_tasks[:max_tasks]


def reconstruct_task_cost(
    task_id: str, *, fields: bool = False, drive_root: Optional[pathlib.Path] = None,
) -> Any:
    """Reconstruct cost, or return honest terminal fields when requested."""
    want = str(task_id or "")
    if not want:
        projection = {
            "cost_accounting_status": "available", "accounted_upper_bound_usd": 0.0,
            "total_rounds": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "cost_final": True, "reserved_usd": 0.0,
            "unresolved_upper_bound_usd": 0.0, "unknown_unmetered": 0,
            "non_final_rows": 0,
        }
    else:
        try:
            from ouroboros.cost_projection import honest_accounted_amount
            from ouroboros.usage_accounting import ensure_legacy_imported, usage_breakdown

            authority_root = pathlib.Path(drive_root) if drive_root is not None else DRIVE_ROOT
            ensure_legacy_imported(authority_root)
            bucket = usage_breakdown(authority_root, task_id=want)
            projection = {
                "cost_accounting_status": "available",
                "accounted_upper_bound_usd": (
                    round(amount, 6)
                    if (amount := honest_accounted_amount(bucket)) is not None
                    else None
                ),
                "total_rounds": int(bucket.get("physical_calls") or 0),
                "prompt_tokens": int(bucket.get("prompt_tokens") or 0),
                "completion_tokens": int(bucket.get("completion_tokens") or 0),
                "cost_final": bool(bucket.get("cost_final")),
                "reserved_usd": float(bucket.get("reserved_usd") or 0.0),
                "unresolved_upper_bound_usd": float(
                    bucket.get("unresolved_upper_bound_usd") or 0.0
                ),
                "unknown_unmetered": int(bucket.get("unknown_unmetered") or 0),
                # The disclosed CAUSE of cost_final=false, carried with the flag.
                "non_final_rows": int(bucket.get("non_final_rows") or 0),
                "ledger_integrity_degraded": bool(bucket.get("integrity_degraded")),
            }
        except Exception:
            log.error("Failed to reconstruct ledger task cost for %s", task_id, exc_info=True)
            projection = {
                "cost_accounting_status": "unavailable", "cost_final": False,
                "cost_accounting_error": "ledger_unavailable",
                "accounted_upper_bound_usd": None, "total_rounds": None,
                "prompt_tokens": None, "completion_tokens": None,
                "reserved_usd": None, "unresolved_upper_bound_usd": None,
                "unknown_unmetered": None, "non_final_rows": None,
                "ledger_integrity_degraded": True,
            }
    if fields:
        # SSOT cost naming (C2/ABI-3): the authority assembles the honest
        # name directly (Ф3.1 fix-round — no producer touches the retired
        # alias); the seam stays as the idempotent amount-normalization and
        # would strip any retired key a future mutation leaked.
        from ouroboros.cost_projection import with_cost_aliases

        return with_cost_aliases(projection)
    if projection.get("cost_accounting_status") != "available":
        from ouroboros.usage_accounting import UsageAccountingError

        raise UsageAccountingError(f"task cost authority unavailable for {task_id}")
    return (
        float(projection["accounted_upper_bound_usd"]), int(projection["total_rounds"]),
        int(projection["prompt_tokens"]), int(projection["completion_tokens"]),
    )


def status_text(workers_dict: Dict[int, Any], pending_list: list,
                running_dict: Dict[str, Dict[str, Any]]) -> str:
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
    accounting_available = True
    try:
        from ouroboros.usage_accounting import ensure_legacy_imported, usage_breakdown, usage_projection

        ensure_legacy_imported(DRIVE_ROOT)
        ledger_breakdown = usage_breakdown(DRIVE_ROOT)
        ledger_projection = (
            usage_projection(DRIVE_ROOT, global_limit_usd=TOTAL_BUDGET_LIMIT)
            if TOTAL_BUDGET_LIMIT > 0
            else ledger_breakdown
        )
        spent = float(ledger_projection.get("accounted_usd") or 0.0)
        pct = (spent / TOTAL_BUDGET_LIMIT * 100.0) if TOTAL_BUDGET_LIMIT > 0 else 0.0
        budget_remaining_usd = (
            float(ledger_projection.get("remaining_known_usd") or 0.0)
            if TOTAL_BUDGET_LIMIT > 0
            else float("inf")
        )
        spent_calls = int(ledger_breakdown.get("physical_calls") or 0)
        prompt_tokens = int(ledger_breakdown.get("prompt_tokens") or 0)
        completion_tokens = int(ledger_breakdown.get("completion_tokens") or 0)
        cached_tokens = int(ledger_breakdown.get("cached_tokens") or 0)
    except Exception:
        log.exception("Budget ledger unavailable while building status")
        accounting_available = False
        spent = pct = budget_remaining_usd = None
        spent_calls = prompt_tokens = completion_tokens = cached_tokens = None
    lines.append(f"budget_total: ${TOTAL_BUDGET_LIMIT:.0f}")
    if not accounting_available:
        lines.append("accounting: unavailable (physical-attempt ledger read failed; dispatch is fail-closed)")
        lines.append("budget_remaining: unavailable")
        lines.append("spent_usd: unavailable")
        lines.append("spent_calls: unavailable")
        lines.append("prompt_tokens: unavailable, completion_tokens: unavailable, cached_tokens: unavailable")
    else:
        if ledger_projection.get("integrity_degraded"):
            lines.append("accounting_integrity: DEGRADED (quarantined ledger tail; cost is not final)")
        lines.append(f"budget_remaining: ${budget_remaining_usd:.0f}")
        if pct > 0:
            lines.append(f"spent_usd: ${spent:.2f} ({pct:.1f}% of budget)")
        else:
            lines.append(f"spent_usd: ${spent:.2f}")
        lines.append(f"spent_calls: {spent_calls}")
        lines.append(
            f"prompt_tokens: {prompt_tokens}, completion_tokens: {completion_tokens}, "
            f"cached_tokens: {cached_tokens}"
        )

    breakdown = budget_breakdown(st)
    if breakdown:
        sorted_categories = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)
        breakdown_parts = [f"{cat}=${cost:.2f}" for cat, cost in sorted_categories if cost > 0]
        if breakdown_parts:
            lines.append(f"budget_breakdown: {', '.join(breakdown_parts)}")

    drift_pct = st.get("budget_drift_pct")
    if accounting_available and drift_pct is not None:
        # Same fields as the update_budget computation: OpenRouter-only settled
        # ledger delta vs the key's usage delta. Rendering the all-provider
        # spent delta here used to contradict the percentage next to it.
        session_total_snap = st.get("session_total_snapshot")
        session_or_settled_snap = st.get("session_openrouter_settled_snapshot")
        or_ledger_settled = st.get("openrouter_ledger_settled_usd")
        or_total = st.get("openrouter_total_usd")

        if (
            session_total_snap is not None
            and session_or_settled_snap is not None
            and or_ledger_settled is not None
            and or_total is not None
        ):
            or_delta = or_total - session_total_snap
            our_delta = float(or_ledger_settled) - float(session_or_settled_snap)

            drift_icon = " ⚠️" if st.get("budget_drift_alert") else ""
            lines.append(
                f"budget_drift: {drift_pct:.1f}%{drift_icon} "
                f"(openrouter tracked: ${our_delta:.2f} vs OpenRouter key: ${or_delta:.2f})"
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
    lines.append("active_liveness: idle+deadline+absolute_ceiling+reaper")
    return "\n".join(lines)


def rotate_jsonl_log_if_needed(
    drive_root: pathlib.Path,
    name: str,
    archive_prefix: str,
    max_bytes: int = 800_000,
) -> None:
    """Rotate ``logs/<name>`` to ``archive/<archive_prefix>_<ts>.jsonl`` when it
    exceeds ``max_bytes``.

    Rotation is an atomic ``os.replace`` rename performed under the SAME sidecar
    lock that ``append_jsonl`` writers take — the old copy+truncate destroyed any
    line appended between the read and the truncate.

    Suppressed in isolated benchmark data roots (``ISOLATED_BENCHMARK_SENTINEL``):
    bench harnesses read trial-local logs as one file from birth, the roots are
    throwaway, and not every harness reader is archive-chain-aware.

    Archives are durable history, NOT GC targets: no retention sweep touches
    ``archive/`` (retention.py governs subagent worktrees, task drives, and
    service logs only) and none may be added — readers backfill from these
    segments, so pruning them would silently erase visible history (BIBLE P1).
    """
    if (drive_root / ISOLATED_BENCHMARK_SENTINEL).exists():
        return
    path = drive_root / "logs" / name
    if not path.exists():
        return
    if path.stat().st_size < max_bytes:
        return
    ts = utc_now_iso().replace("-", "").replace(":", "").split(".")[0]
    archive_path = drive_root / "archive" / f"{archive_prefix}_{ts}.jsonl"
    # Second-resolution names can collide when a fast writer forces two rotations
    # within one second; os.replace onto an existing archive would destroy it. The
    # "_<n>" suffix sorts lexicographically AFTER "<ts>.jsonl" ("_" > "."), so
    # name-ordered readers keep the true chronological chain.
    suffix = 0
    while archive_path.exists():
        suffix += 1
        archive_path = drive_root / "archive" / f"{archive_prefix}_{ts}_{suffix}.jsonl"
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    from ouroboros.utils import jsonl_append_lock_path

    lock_path = jsonl_append_lock_path(path)
    lock_fd = acquire_exclusive_file_lock(lock_path, timeout_sec=2.0, stale_sec=10.0)
    if lock_fd is None:
        log.warning("%s rotation skipped: append lock busy", name)
        return
    try:
        if not path.exists() or path.stat().st_size < max_bytes:
            return
        os.replace(path, archive_path)
        path.touch()
    finally:
        release_exclusive_file_lock(lock_path, lock_fd)


def rotate_chat_log_if_needed(drive_root: pathlib.Path, max_bytes: int = 800_000) -> None:
    """Compatibility wrapper: chat.jsonl rotation via the generalized rotator."""
    rotate_jsonl_log_if_needed(drive_root, "chat.jsonl", "chat", max_bytes)
