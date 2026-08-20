"""Ouroboros — the numeric runtime knobs and their clamps.

Worker count, task liveness windows, per-call ceilings, reviewer and acceptance
budgets, subagent caps and delegation windows. Every one of them is an
environment-or-default scalar clamped into a documented band, so a typo falls
back to the shipped value instead of disabling a rail.
"""

from __future__ import annotations

import os
from typing import Optional

from ouroboros.settings_defaults import (
    PACING_INTERVAL_DEFAULT_SEC,
    SETTINGS_DEFAULTS,
    SUPERVISOR_LIVENESS_DEADLINE_DEFAULT_SEC,
)


def _clamped_number_setting(key: str, *, low, high=float("inf"), cast=float):
    """Env-or-default numeric setting clamped to [low, high]; a typo falls back to the
    shipped default. SSOT for the clamped scalar getters below — the seven of them were
    byte-identical except for key, caster and bounds (P7 DRY)."""
    try:
        value = cast(os.environ.get(key, "") or SETTINGS_DEFAULTS[key])
    except (TypeError, ValueError):
        value = cast(SETTINGS_DEFAULTS[key])
    return max(low, min(value, high))


def _bounded_positive_int_setting(key: str, *, default: int, hard_max: int, min_value: int = 1) -> int:
    """Bounded int setting; below ``min_value`` it is a typo and falls back to ``default``. Only
    subagent depth passes 0 — there an explicit 0 is a real owner choice, not unset (owner Q26)."""
    raw = os.environ.get(key, SETTINGS_DEFAULTS.get(key, default))
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        parsed = default
    if parsed < min_value:
        parsed = default
    return max(min_value, min(parsed, hard_max))


def get_max_workers() -> int:
    return _clamped_number_setting("OUROBOROS_MAX_WORKERS", low=1, cast=int)


def get_task_idle_timeout_sec() -> int:
    """Idle window before a task is eligible for an activity-based stop: it has made
    no REAL progress (its own last_progress_at) AND has no progressing subtree for
    this long. The periodic 30s process heartbeat is liveness, NOT progress."""
    return _clamped_number_setting("OUROBOROS_TASK_IDLE_TIMEOUT_SEC", low=60, cast=int)


def get_task_abs_ceiling_sec() -> int:
    """Absolute wall-clock backstop per task, independent of activity — the only hard
    time axis (budget/cost is the other, separate hard axis). A productively-waiting
    orchestrator survives to this ceiling instead of a flat 1800s wall-clock kill."""
    return _clamped_number_setting("OUROBOROS_TASK_ABS_CEILING_SEC", low=300, cast=int)


def get_per_call_timeout_ceiling_sec() -> int:
    """SSOT ceiling for an explicit per-call run_command/run_script timeout_sec
    (and the outer tool-execution cap that accommodates it)."""
    return _clamped_number_setting("OUROBOROS_PER_CALL_TIMEOUT_CEILING_SEC", low=1, cast=int)


def get_restart_drain_max_sec() -> int:
    return _clamped_number_setting(
        "OUROBOROS_RESTART_DRAIN_MAX_SEC", low=0, cast=lambda v: int(float(v)))


def get_safety_max_tokens() -> int:
    """Output-token budget for safety-supervisor LLM calls (parse-bug fix)."""
    return _clamped_number_setting("OUROBOROS_SAFETY_MAX_TOKENS", low=256, high=16384, cast=int)


def get_safety_call_timeout_sec() -> float:
    """Transport timeout for safety-supervisor LLM calls (prevents indefinite hang)."""
    return _clamped_number_setting("OUROBOROS_SAFETY_CALL_TIMEOUT_SEC", low=5.0, high=600.0)


def get_websearch_timeout_sec() -> float:
    """Transport timeout for the web_search OpenAI streaming call (v6.54.3, D)."""
    return _clamped_number_setting("OUROBOROS_WEBSEARCH_TIMEOUT_SEC", low=30.0, high=3600.0)


def get_llm_transport_read_timeout_sec() -> float:
    """Default httpx read/write timeout for no_proxy LLM clients (v6.54.3, D).

    The DEAD-SOCKET bound, not a latency target; explicit per-call timeouts win."""
    return _clamped_number_setting("OUROBOROS_LLM_TRANSPORT_READ_TIMEOUT_SEC", low=60.0, high=7200.0)


def get_acceptance_review_est_sec() -> float:
    """Estimated duration of one acceptance review/improvement pass (v6.54.4)."""
    return _clamped_number_setting("OUROBOROS_ACCEPTANCE_REVIEW_EST_SEC", low=10.0, high=3600.0)


def get_acceptance_reserve_pct() -> int:
    """Default finalization-reserve percentage of the total budget (v6.54.4)."""
    return _clamped_number_setting("OUROBOROS_ACCEPTANCE_RESERVE_PCT", low=0, high=50, cast=int)


def get_plan_task_deadline_min_sec() -> float:
    """Minimum useful deadline-scaled planning-swarm window (v6.54.3, 1.5)."""
    return _clamped_number_setting("OUROBOROS_PLAN_TASK_DEADLINE_MIN_SEC", low=30.0, high=3600.0)


def get_vision_caption_timeout_sec() -> int:
    return _clamped_number_setting("OUROBOROS_VISION_CAPTION_TIMEOUT_SEC", low=1, cast=int)


def get_pacing_interval_sec(settings: Optional[dict] = None) -> int:
    """Intrinsic self-pacing checkpoint cadence in seconds (0 disables)."""
    raw = os.environ.get("OUROBOROS_PACING_INTERVAL_SEC")
    if raw is None and isinstance(settings, dict):
        raw = settings.get("OUROBOROS_PACING_INTERVAL_SEC")
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        parsed = int(PACING_INTERVAL_DEFAULT_SEC)
    return max(0, parsed)


def get_supervisor_liveness_deadline_sec(settings: Optional[dict] = None) -> int:
    """Supervisor-loop stall deadline in seconds (0 disables the watchdog)."""
    raw = os.environ.get("OUROBOROS_SUPERVISOR_LIVENESS_DEADLINE_SEC")
    if raw is None and isinstance(settings, dict):
        raw = settings.get("OUROBOROS_SUPERVISOR_LIVENESS_DEADLINE_SEC")
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        parsed = int(SUPERVISOR_LIVENESS_DEADLINE_DEFAULT_SEC)
    return max(0, parsed)


def get_post_task_evolution_budget_usd() -> float:
    """Optional per-window USD budget for post-task evolution (0 = use the
    existing EVOLUTION_BUDGET_RESERVE / TOTAL_BUDGET gating only)."""
    return _clamped_number_setting("OUROBOROS_POST_TASK_EVOLUTION_BUDGET_USD", low=0.0)


# ONE per-root subagent ceiling (v6.82: 50->500): clamp below, supervisor/events.py, wait_tasks; ARCHITECTURE §7.
MAX_ACTIVE_SUBAGENTS_HARD_CAP = 500


def get_max_active_subagents_per_root() -> int:
    return _bounded_positive_int_setting(
        "OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT",
        default=int(SETTINGS_DEFAULTS["OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT"]),
        hard_max=MAX_ACTIVE_SUBAGENTS_HARD_CAP,
    )


def get_max_subagent_depth() -> int:
    """Structural nesting cap; 0 = NO delegation at all (every child refused, root tasks still
    run). Before v6.79.0 a configured 0 was silently rewritten to 2, so "no-swarm" delegated."""
    return _bounded_positive_int_setting(
        "OUROBOROS_MAX_SUBAGENT_DEPTH",
        default=int(SETTINGS_DEFAULTS["OUROBOROS_MAX_SUBAGENT_DEPTH"]),
        hard_max=10,
        min_value=0,
    )


# delegate_wait's ToolEntry per-call timeout (above it a configured ceiling buys a
# KILLED call, not a longer wait; pinned by test) and the hard max WINDOW per call
# (F5): 1800 < 2100 (kill) < 2400 (lease) — decoupled, a raised timeout never widens it.
DELEGATE_WAIT_CEILING_SEC = 2100
DELEGATE_WAIT_WINDOW_MAX_SEC = 1800


def get_delegate_wait_max_sec() -> int:
    """delegate_wait window ceiling: the setting NARROWS, never widens past 1800."""
    return _clamped_number_setting(
        "OUROBOROS_DELEGATE_WAIT_MAX_SEC", low=1, high=DELEGATE_WAIT_WINDOW_MAX_SEC, cast=int)


def get_delegate_wait_sec() -> int:
    """Default WINDOW one ``delegate_wait`` call holds — not a quiet cutoff: the
    wait holds, returns its advances, and bounds the nanny's mailbox absence."""
    return _clamped_number_setting(
        "OUROBOROS_DELEGATE_WAIT_SEC", low=1, high=get_delegate_wait_max_sec(), cast=int)


def get_search_code_wall_sec() -> float:
    """Total wall-clock budget (seconds) for ONE search_code call — bounds both the rg
    directory walk and the batched rg loop so a scan over a very large root cannot run
    unbounded. Env/setting: ``OUROBOROS_SEARCH_CODE_WALL_SEC`` (floored at 5s)."""
    raw = (os.environ.get("OUROBOROS_SEARCH_CODE_WALL_SEC", "")
           or str(SETTINGS_DEFAULTS.get("OUROBOROS_SEARCH_CODE_WALL_SEC", "45")))
    try:
        return max(5.0, float(raw))
    except (TypeError, ValueError):
        return 45.0
