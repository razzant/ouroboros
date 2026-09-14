"""Ouroboros — the numeric runtime knobs and their clamps.

Worker count, task liveness windows, per-call ceilings, reviewer and acceptance
budgets, subagent caps and delegation windows, plus fixed process and transport
bounds. Environment-or-default getters clamp into documented bands, so a typo
falls back to the shipped value instead of disabling a rail.
"""

from __future__ import annotations

from typing import Optional

from ouroboros.settings_defaults import (
    FINALIZATION_GRACE_DEFAULT_SEC,
    PACING_INTERVAL_DEFAULT_SEC,
    SETTINGS_DEFAULTS,
    SUPERVISOR_LIVENESS_DEADLINE_DEFAULT_SEC,
)
from ouroboros.settings_integrity import runtime_setting

# Local model-operation status polling; not a provider deadline or quota timer.
CLAUDEXOR_MODEL_POLL_INTERVAL_SEC = 0.25
# Existing CLI RPC (10s), termination confirmation (20s), and process startup slack.
CLAUDEXOR_OPERATOR_STOP_TIMEOUT_SEC = 35.0
# Physical exit observation after a clean operator-stop receipt, not a task deadline.
CLAUDEXOR_STOP_EXIT_WAIT_SEC = 5.0
# Phone-native source compilation exceeds ten minutes; one contained platform
# preparation may run for an hour, independently of ordinary tool/harness calls.
EXTERNAL_PLATFORM_UPDATE_TIMEOUT_SEC = 3600.0


EXTENSION_STREAM_CHUNK_BYTES = 64 * 1024
# Exit/pipe-drain grace after a response ends; never a response lifetime timer.
EXTENSION_CHILD_CLEANUP_GRACE_SEC = 2
NESTED_SETTLEMENT_MARGIN_SEC = 30  # Structural ordering margin, not a cognition timeout.
# Owner-note cadence while a task waits out a provider-connection outage; the effective interval is min(this, idle_timeout/2) so the notes also keep the idle rail alive.
NETWORK_WAIT_NOTE_INTERVAL_SEC = 300
# First free-redial pause of a transport-wait episode; doubles per wait iteration up to the existing 60s transient backoff cap (Q10: an existing bound, not a new knob).
NETWORK_WAIT_BACKOFF_START_SEC = 4.0
NETWORK_WAIT_BACKOFF_MAX_SEC = 60.0
# TCP keepalive for long-lived remote LLM sockets (idle threshold, probe interval, probe count): kernel probes
# detect a silently dropped NAT/VPN mapping instead of hanging to the read timeout; platform_layer builds the options.
TCP_KEEPALIVE_IDLE_SEC = 60
TCP_KEEPALIVE_INTERVAL_SEC = 60
TCP_KEEPALIVE_PROBE_COUNT = 5
# One response frame may carry metadata or a body chunk; never a total response cap.
EXTENSION_STREAM_METADATA_BYTES = 512 * 1024
# Only the out-of-process WS relay is limited: the existing reserve refills gradually.
WS_RELAY_BURST = 60
WS_RELAY_REFILL_PER_SEC = 1.0


# Worker-pool spawn bounds (structural constants, not env knobs). Grace after a full-pool spawn before the crash
# detector counts dead workers (up to ~60s to init: spawn + pip); workers.py binds it as `_SPAWN_GRACE_SEC`, the extension import-staging sweep reads it too.
WORKER_SPAWN_GRACE_SEC = 90.0
# Readiness window for ONE spawned/respawned slot: unassignable until the child's own `worker_ready` row lands; alive
# but silent past this = torn down and replaced. Sized to the spawn grace (the pool's existing init budget): a warm
# forkserver child boots in ~3-4s (G13 mock lane: 3.5-4.9s startup, 2.5-3.2s respawn), a cold 4-vCPU CI runner well under 60s (its 21-scenario mock lane runs in ~80s), and the E2E
# scenarios wait 240s per task, so a wedged child is a fast, named failure. A contract distinct from process liveness
# (`proc.is_alive`, worker_health.py) and from the task idle rail (queue_timeouts.py): a deadlocked child is alive.
WORKER_READY_WINDOW_SEC = 90.0
# Consecutive readiness failures of one slot before it is parked and reported (three strikes, like the crash-storm fence).
WORKER_READY_MAX_ATTEMPTS = 3


def _clamped_number_setting(key: str, *, low, high=float("inf"), cast=float):
    """Env-or-default numeric setting clamped to [low, high]; a typo falls back to the
    shipped default. SSOT for the clamped scalar getters below — the seven of them were
    byte-identical except for key, caster and bounds (P7 DRY)."""
    try:
        value = cast(runtime_setting(key, "") or SETTINGS_DEFAULTS[key])
    except (TypeError, ValueError):
        value = cast(SETTINGS_DEFAULTS[key])
    return max(low, min(value, high))


def _bounded_positive_int_setting(key: str, *, default: int, hard_max: int, min_value: int = 1) -> int:
    """Bounded int setting; below ``min_value`` it is a typo and falls back to ``default``. Only
    subagent depth passes 0 — there an explicit 0 is a real owner choice, not unset (owner Q26)."""
    raw = runtime_setting(key, SETTINGS_DEFAULTS.get(key, default))
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


def get_update_letter_timeout_sec() -> float:
    """Transport ceiling of the update-letter LIGHT one-shot (`update_letter.py`)."""
    return _clamped_number_setting("OUROBOROS_UPDATE_LETTER_TIMEOUT_SEC", low=10.0, high=600.0)


def get_websearch_timeout_sec() -> float:
    """Per-attempt transport timeout for provider-backed web_search calls."""
    return _clamped_number_setting("OUROBOROS_WEBSEARCH_TIMEOUT_SEC", low=30.0, high=3600.0)


def get_llm_transport_read_timeout_sec() -> float:
    """Default httpx read/write timeout for no_proxy LLM clients (v6.54.3, D).

    The DEAD-SOCKET bound, not a latency target; explicit per-call timeouts win."""
    return _clamped_number_setting("OUROBOROS_LLM_TRANSPORT_READ_TIMEOUT_SEC", low=60.0, high=7200.0)


def get_acceptance_review_est_sec() -> float:
    """The configurable acceptance admission floor, clamped to >=200 s by task_pacing."""
    return _clamped_number_setting("OUROBOROS_ACCEPTANCE_REVIEW_EST_SEC", low=10.0, high=3600.0)


def get_acceptance_reserve_pct() -> int:
    """Default finalization-reserve percentage of the total budget (v6.54.4)."""
    return _clamped_number_setting("OUROBOROS_ACCEPTANCE_RESERVE_PCT", low=0, high=50, cast=int)


def get_plan_task_deadline_min_sec() -> float:
    """Minimum useful deadline-scaled planning-swarm window (v6.54.3, 1.5)."""
    return _clamped_number_setting("OUROBOROS_PLAN_TASK_DEADLINE_MIN_SEC", low=30.0, high=3600.0)


def get_claudexor_quota_refresh_timeout_sec() -> int:
    return _clamped_number_setting("OUROBOROS_CLAUDEXOR_QUOTA_REFRESH_TIMEOUT_SEC", low=1, high=90, cast=int)


def get_claudexor_harness_install_timeout_sec() -> int:
    return _clamped_number_setting("OUROBOROS_CLAUDEXOR_HARNESS_INSTALL_TIMEOUT_SEC", low=1, cast=int)


def get_onboarding_snapshot_timeout_sec() -> int:
    """Bound of the single Claudexor snapshot read during onboarding completion (#464)."""
    return _clamped_number_setting("OUROBOROS_ONBOARDING_SNAPSHOT_TIMEOUT_SEC", low=1, high=300, cast=int)


def get_settings_document_lock_timeout_sec() -> int:
    """Bound of the in-process settings-document lock (one read-merge-write plus its effects)."""
    return _clamped_number_setting("OUROBOROS_SETTINGS_DOCUMENT_LOCK_TIMEOUT_SEC", low=1, high=300, cast=int)


def get_direct_turn_stop_wait_sec() -> float:
    """How long custody waits for a stopped direct-chat turn to reach its next round boundary."""
    return _clamped_number_setting("OUROBOROS_DIRECT_TURN_STOP_WAIT_SEC", low=0, high=10, cast=float)


def get_vision_caption_timeout_sec() -> int:
    return _clamped_number_setting("OUROBOROS_VISION_CAPTION_TIMEOUT_SEC", low=1, cast=int)


def get_pacing_interval_sec(settings: Optional[dict] = None) -> int:
    """Intrinsic self-pacing checkpoint cadence in seconds (0 disables)."""
    raw = runtime_setting("OUROBOROS_PACING_INTERVAL_SEC")
    if raw is None and isinstance(settings, dict):
        raw = settings.get("OUROBOROS_PACING_INTERVAL_SEC")
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        parsed = int(PACING_INTERVAL_DEFAULT_SEC)
    return max(0, parsed)


def get_supervisor_liveness_deadline_sec(settings: Optional[dict] = None) -> int:
    """Supervisor-loop stall deadline in seconds (0 disables the watchdog)."""
    raw = runtime_setting("OUROBOROS_SUPERVISOR_LIVENESS_DEADLINE_SEC")
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


# Per-root active-child ceiling (v6.82: 50->500) and absolute host-visible nesting ceiling, used by supervisor gates and ARCHITECTURE §7.
MAX_ACTIVE_SUBAGENTS_HARD_CAP, MAX_SUBAGENT_DEPTH_HARD_CAP = 500, 10


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
        hard_max=MAX_SUBAGENT_DEPTH_HARD_CAP,
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
    return _clamped_number_setting("OUROBOROS_SEARCH_CODE_WALL_SEC", low=5.0)


def get_finalization_grace_sec(settings: Optional[dict] = None) -> int:
    """Grace window in seconds: env, else the ``settings`` argument, else the
    shipped default — the ``_clamped_number_setting`` shape. Deliberately NO
    ``load_settings()`` fallback: a READ must never persist settings, and that
    call runs the context-mode compatibility migration, which can WRITE a
    normalized file under read-only observers (``task_pacing._reserve_sec``)."""
    raw = runtime_setting("OUROBOROS_FINALIZATION_GRACE_SEC")
    if raw is None and isinstance(settings, dict):
        raw = settings.get("OUROBOROS_FINALIZATION_GRACE_SEC")
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        parsed = int(FINALIZATION_GRACE_DEFAULT_SEC)
    return max(0, min(parsed, 300))
