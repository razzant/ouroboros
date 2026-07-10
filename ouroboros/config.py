"""
Ouroboros — Shared configuration (single source of truth).

Paths, settings defaults, load/save with file locking.
Only imports ouroboros.platform_layer (platform abstraction, no circular deps).
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import sys
import time
from typing import Any, Optional

from ouroboros.platform_layer import pid_lock_acquire as _compat_pid_lock_acquire
from ouroboros.platform_layer import pid_lock_release as _compat_pid_lock_release
from ouroboros.provider_models import compute_direct_review_models_fallback, migrate_model_value


# Paths
HOME = pathlib.Path.home()
APP_ROOT = pathlib.Path(os.environ.get("OUROBOROS_APP_ROOT", HOME / "Ouroboros"))
REPO_DIR = pathlib.Path(os.environ.get("OUROBOROS_REPO_DIR", APP_ROOT / "repo"))
DATA_DIR = pathlib.Path(os.environ.get("OUROBOROS_DATA_DIR", APP_ROOT / "data"))
SETTINGS_PATH = pathlib.Path(os.environ.get("OUROBOROS_SETTINGS_PATH", DATA_DIR / "settings.json"))
PID_FILE = pathlib.Path(os.environ.get("OUROBOROS_PID_FILE", APP_ROOT / "ouroboros.pid"))
PORT_FILE = pathlib.Path(os.environ.get("OUROBOROS_PORT_FILE", DATA_DIR / "state" / "server_port"))

RESTART_EXIT_CODE = 42
PANIC_EXIT_CODE = 99
AGENT_SERVER_PORT = 8765
FINALIZATION_GRACE_DEFAULT_SEC = 120
# Cadence for intrinsic self-pacing checkpoints when a task has NO deadline_at
# (e.g. headless benchmark runs). Advisory only — surfaces elapsed/rounds/cost so
# the model can self-pace; it is not a stop gate. 0 disables.
PACING_INTERVAL_DEFAULT_SEC = 600
# Supervisor-loop liveness deadline (WS3, v6.34.0): a dedicated watchdog thread
# flags the main supervisor loop as STALLED if it has not ticked within this many
# seconds (it normally ticks every ~0.5s). Far above any healthy tick so it only
# fires on a real wedge (a blocking step starving new-message intake). 0 disables.
SUPERVISOR_LIVENESS_DEADLINE_DEFAULT_SEC = 90


def _guard_live_settings_write() -> None:
    if os.environ.get("OUROBOROS_ALLOW_LIVE_DATA_TESTS") == "1":
        return
    try:
        live_settings = SETTINGS_PATH.resolve(strict=False) == (
            HOME / "Ouroboros" / "data" / "settings.json"
        ).resolve(strict=False)
    except OSError:
        live_settings = False
    if ("PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules) and live_settings:
        raise RuntimeError(
            "Refusing to write live Ouroboros settings.json from pytest. "
            "Set OUROBOROS_SETTINGS_PATH/OUROBOROS_DATA_DIR to a temp path, "
            "or OUROBOROS_ALLOW_LIVE_DATA_TESTS=1 for an explicit live-data test."
        )


# Settings defaults
SETTINGS_DEFAULTS = {
    "OPENROUTER_API_KEY": "",
    "OPENAI_API_KEY": "",
    "OPENAI_BASE_URL": "",
    "OPENAI_COMPATIBLE_API_KEY": "",
    "OPENAI_COMPATIBLE_BASE_URL": "",
    "CLOUDRU_FOUNDATION_MODELS_API_KEY": "",
    "CLOUDRU_FOUNDATION_MODELS_BASE_URL": "https://foundation-models.api.cloud.ru/v1",
    "GIGACHAT_CREDENTIALS": "",
    "GIGACHAT_USER": "",
    "GIGACHAT_PASSWORD": "",
    "GIGACHAT_SCOPE": "GIGACHAT_API_PERS",
    "GIGACHAT_BASE_URL": "https://gigachat.devices.sberbank.ru/api/v1",
    "GIGACHAT_VERIFY_SSL_CERTS": "true",
    "GIGACHAT_PROFANITY_CHECK": "",
    "ANTHROPIC_API_KEY": "",

    "OUROBOROS_NETWORK_PASSWORD": "",
    "OUROBOROS_SERVER_HOST": "127.0.0.1",
    "OUROBOROS_HOST_SERVICE_PORT": 8767,
    "OUROBOROS_MODEL": "google/gemini-3.5-flash",
    # Worker lanes. Empty means "use OUROBOROS_MODEL" (same shape as consciousness),
    # so the owner sets ONE model by default and optionally overrides a lane. HEAVY is
    # the strong acting/coding lane (mutative first-level subagents); LIGHT is the cheap
    # bulk lane (auto / deep subagents). (HEAVY renamed from the legacy MODEL_CODE.)
    "OUROBOROS_MODEL_HEAVY": "",
    "OUROBOROS_MODEL_LIGHT": "",
    "OUROBOROS_MODEL_VISION": "",
    "OUROBOROS_IMAGE_INPUT_MODE": "auto",
    # Background consciousness is a high-horizon cognitive loop, not a cheap
    # helper lane. Empty means "use OUROBOROS_MODEL".
    "OUROBOROS_MODEL_CONSCIOUSNESS": "",
    # Cross-model resilience CHAIN (comma-separated, ordered). A single model is a
    # 1-element chain; empty disables cross-model fallback. Resilience slot — keeps a
    # real default, unlike the worker lanes. (Renamed from the singular MODEL_FALLBACK.)
    "OUROBOROS_MODEL_FALLBACKS": "anthropic/claude-sonnet-4.6",
    "OUROBOROS_MODEL_DEEP_SELF_REVIEW": "openai/gpt-5.5-pro",
    "CLAUDE_CODE_MODEL": "opus[1m]",
    "OUROBOROS_MAX_WORKERS": 10,
    "OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT": 6,
    "OUROBOROS_MAX_SUBAGENT_DEPTH": 2,
    # Mutative ("acting") subagents master toggle. Empty = follow runtime mode
    # (ON in advanced/pro, OFF in light); explicit true/false overrides. Owner-
    # controlled; light-mode self-repo writes stay blocked by the sandbox.
    "OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS": "",
    # Acting self_worktree base location + durable genesis projects root (both
    # outside repo/ and data/). genesis projects are durable and never GC'd.
    "OUROBOROS_SUBAGENT_WORKTREE_ROOT": "",
    "OUROBOROS_SUBAGENT_PROJECTS_ROOT": "",
    "OUROBOROS_DELIVERABLES_ROOT": "",
    # Unified age-based GC retention (days) for ALL disposable runtime artifacts:
    # subagent worktrees, headless/direct task drives, and leftover service logs.
    # Single owner-facing knob (math SSOT in ouroboros/retention.py); deprecated
    # per-subsystem keys are migrated to this on settings load.
    "OUROBOROS_GC_RETENTION_DAYS": 7,
    "OUROBOROS_PLAN_TASK_SWARM_TIMEOUT_SEC": 120,
    "OUROBOROS_PLAN_TASK_SWARM_MAX_WAIT_SEC": 900,
    "OUROBOROS_PLAN_TASK_SWARM_HEARTBEAT_STALE_SEC": 120,
    "TOTAL_BUDGET": 10.0,
    "OUROBOROS_PER_TASK_COST_USD": 20.0,
    # cloud.ru Foundation Models catalog prices token costs in RUB per 1M; budget is
    # USD. Owner-configurable RUB->USD divisor for converting cloud.ru cost to USD.
    "OUROBOROS_RUB_USD_RATE": 95.0,
    # Live-pricing (OpenRouter + cloud.ru catalog) refetch interval; prices/FX drift.
    "OUROBOROS_PRICING_TTL_SEC": 21600,
    # Main-loop round ceiling (was an inline literal in loop.py — hot-reloadable now).
    "OUROBOROS_MAX_ROUNDS": 200,
    # Same-model attempt budget for TRANSIENT provider failure classes
    # (finish_reason=null, 429/5xx/overloaded); floored at the caller's base
    # retry budget. Permanent classes fail fast regardless.
    "OUROBOROS_TRANSIENT_RETRY_MAX": 6,
    # #4 self-DoS guard: max concurrent provider calls per (model, use_local) route; excess
    # worker threads wait (deadline-bounded) instead of storming one model's rate limit. <=0
    # disables. Default-on, fail-soft (see ouroboros/model_concurrency.py).
    "OUROBOROS_MODEL_MAX_CONCURRENCY": 3,
    # Hard ceiling (seconds) a provider call waits for a concurrency slot when the task has
    # NO deadline; past it the call proceeds WITHOUT a slot (never blocks forever). SSOT here.
    "OUROBOROS_MODEL_SLOT_MAX_WAIT_SEC": 180,
    # Project-naming LIGHT-call waits (v6.40): the provider-call transport timeout and the
    # gateway's hard wait for the inline turn-into-project name. SSOT here (not magic numbers
    # in project_naming.py) per DEVELOPMENT "Timeout & Wait Control".
    "OUROBOROS_PROJECT_NAMING_TIMEOUT_SEC": 60,
    "OUROBOROS_PROJECT_NAMING_ASYNC_TIMEOUT_SEC": 8,
    # Skill lifecycle lane deadline (wedged-job loud-failure bound).
    "OUROBOROS_SKILL_LIFECYCLE_TIMEOUT_SEC": 1800,
    "OUROBOROS_SOFT_TIMEOUT_SEC": 600,
    # NOTE: OUROBOROS_HARD_TIMEOUT_SEC no longer terminates tasks — the flat wall-clock
    # kill was replaced by the activity model below (idle + subtree-liveness, abs ceiling).
    # It survives only as a soft-warning/status display input; runtime is governed by
    # OUROBOROS_TASK_IDLE_TIMEOUT_SEC and OUROBOROS_TASK_ABS_CEILING_SEC.
    "OUROBOROS_HARD_TIMEOUT_SEC": 1800,
    # Activity-based liveness (replaces flat wall-clock as the primary stop):
    # idle window = no real progress AND no progressing subtree; abs ceiling = the
    # unconditional per-task backstop (budget/cost stays a separate hard axis).
    "OUROBOROS_TASK_IDLE_TIMEOUT_SEC": 900,
    "OUROBOROS_TASK_ABS_CEILING_SEC": 21600,
    "OUROBOROS_PER_CALL_TIMEOUT_CEILING_SEC": 1800,
    "OUROBOROS_FINALIZATION_GRACE_SEC": FINALIZATION_GRACE_DEFAULT_SEC,
    "OUROBOROS_SUPERVISOR_LIVENESS_DEADLINE_SEC": SUPERVISOR_LIVENESS_DEADLINE_DEFAULT_SEC,
    "OUROBOROS_PACING_INTERVAL_SEC": PACING_INTERVAL_DEFAULT_SEC,
    "OUROBOROS_TOOL_TIMEOUT_SEC": 600,
    "OUROBOROS_VISION_CAPTION_TIMEOUT_SEC": 90,
    "OUROBOROS_BG_MAX_ROUNDS": 10,
    "OUROBOROS_BG_WAKEUP_MIN": 30,
    "OUROBOROS_BG_WAKEUP_MAX": 7200,
    # Post-task self-evolution envelope (V4). Owner-enabled capability whose
    # CONTENT stays LLM-first; default OFF. When enabled, after a qualifying task
    # the worker may promote one high-value code-class backlog item into the
    # existing (gated) evolution campaign. Cadence: off | llm | every_n:<k>.
    "OUROBOROS_POST_TASK_EVOLUTION": "false",
    "OUROBOROS_POST_TASK_EVOLUTION_CADENCE": "llm",
    "OUROBOROS_POST_TASK_EVOLUTION_BUDGET_USD": 0.0,
    # Optional owner steer appended to each evolution cycle's objective (never
    # overrides the LLM-first promotion). Empty = pure LLM choice.
    "OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE": "",
    "OUROBOROS_WEBSEARCH_MODEL": "gpt-5.2",
    # web_search backend pin: auto (default OpenAI-first cascade) | ddgs (pure
    # retrieval, no second LLM — for fixed-model runs) | openai | openrouter | anthropic.
    "OUROBOROS_WEBSEARCH_BACKEND": "auto",
    # Main-loop OpenRouter server web-search tool. Off by default: provider-
    # specific capability, not a core provider-independence requirement.
    "OUROBOROS_MAIN_WEB_SEARCH": "off",
    "OUROBOROS_MAIN_WEB_SEARCH_ENGINE": "auto",
    "OUROBOROS_MAIN_WEB_SEARCH_MAX_TOTAL_RESULTS": 10,
    # OpenRouter provider routing: "" (off) | resilience (same-model failover, cache-warm)
    # | repro (pin, no failover — fixed-model runs) | a raw JSON `provider` object.
    "OUROBOROS_OR_PROVIDER": "",
    # search_code total wall-clock budget (seconds) bounding the rg walk + the fallback walk.
    "OUROBOROS_SEARCH_CODE_WALL_SEC": "45",
    # NOTE: OUROBOROS_OBSERVABILITY_KEEP_RAW (writes UNREDACTED secret-bearing payloads to
    # disk) is intentionally NOT a settings/UI carrier — it is an env-only operator debug
    # override so a self-change or non-owner save can never enable secret logging.
    # Generative context-window probe (Max gate): on (default) confirms a route's >=1M
    # window from a FREE over-window reject; *_CHARS sizes the oversized padding.
    "OUROBOROS_GENERATIVE_PROBE": "1",
    "OUROBOROS_GENERATIVE_PROBE_CHARS": "5000000",
    # Pre-commit review: comma-separated provider-tagged model list
    "OUROBOROS_REVIEW_MODELS": "openai/gpt-5.5,google/gemini-3.5-flash,anthropic/claude-fable-5",
    # Pre-commit review enforcement: advisory | blocking
    "OUROBOROS_REVIEW_ENFORCEMENT": "advisory",
    # Auto-grant reviewed-skill requests by default; grants stay bound to the
    # reviewed content hash and editing a skill still invalidates them.
    "OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS": "true",
    # Launcher-seeded native skills carry a hash-pinned native-trust review
    # verdict (the payload bytes shipped through the repo commit gate); the
    # zero-grant ones also auto-enable. Editing the payload still goes stale.
    # Owner opt-out: set to false to keep manual review for native seeds.
    "OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS": "true",
    # Agent-requested restarts drain running tasks first: while any RUNNING
    # task still heartbeats, the restart waits up to this many seconds before
    # proceeding fail-closed (0 = no drain, restart immediately).
    "OUROBOROS_RESTART_DRAIN_MAX_SEC": 120,
    # Runtime mode: light | advanced | pro; pro still requires review gates.
    "OUROBOROS_RUNTIME_MODE": "advanced",
    # Context mode: low | max. Owner-only working-context size profile. max =
    # full always-on docs + current memory granularity; low = ARCHITECTURE as a
    # navigation map + deeper memory consolidation, sized for ~200k / local models.
    # Cognitive-horizon knob (BIBLE P1): the agent cannot lower it (owner-only),
    # and it never changes model / reasoning-effort / output-token budgets.
    "OUROBOROS_CONTEXT_MODE": "max",
    # Optional extra user-managed skills checkout; Ouroboros never clones/pulls it.
    "OUROBOROS_SKILLS_REPO_PATH": "",
    "OUROBOROS_CLAWHUB_REGISTRY_URL": "https://clawhub.ai/api/v1",
    "OUROBOROS_HUB_CATALOG_URL": "https://raw.githubusercontent.com/razzant/OuroborosHub/main/catalog.json",
    "MCP_ENABLED": False,
    "MCP_SERVERS": [],
    "MCP_TOOL_TIMEOUT_SEC": 60,
    # Scope review: one or more reviewer slots; enforcement follows OUROBOROS_REVIEW_ENFORCEMENT.
    "OUROBOROS_SCOPE_REVIEW_MODELS": "anthropic/claude-fable-5",
    "OUROBOROS_SCOPE_REVIEW_MODEL": "anthropic/claude-fable-5",
    # Opt-in (default off): in low context mode, after the normal scope-review
    # prompt cannot fit and routes to the non-blocking skip, run a supplemental
    # window-fitting ADVISORY degraded scope review (top-scored touched +
    # import-seam + contracts full; the rest manifest-only). Does NOT claim full
    # coverage and does NOT replace the >=1M blocking scope floor.
    "OUROBOROS_SCOPE_REVIEW_DEGRADED": "false",
    # P3 scope-reviewer capability floor: blocking_1m (default; the reviewer is the
    # >=1M blocking gate) | advisory (sub-1M reviewer, supplementary only — can
    # never satisfy a required blocking scope gate). See get_scope_review_floor.
    "OUROBOROS_SCOPE_REVIEW_FLOOR": "blocking_1m",
    "OUROBOROS_TASK_REVIEW_MODE": "auto",
    # LLM safety-supervisor coverage (owner-only, like runtime/context mode):
    #   full (default)  — LLM check on every POLICY_CHECK tool + non-whitelisted
    #                     POLICY_CHECK_CONDITIONAL shell (today's behavior).
    #   light           — LLM check ONLY on POLICY_CHECK integration tools;
    #                     POLICY_CHECK_CONDITIONAL shell/verify fall to the
    #                     deterministic whitelist + registry guards (no LLM).
    #   off             — no LLM safety calls at all; the deterministic registry
    #                     sandbox, protected-path policy, and light-mode guards
    #                     STAY ON. Every non-full mode emits a durable audit event.
    "OUROBOROS_SAFETY_MODE": "full",
    # Safety-supervisor LLM call shaping (v6.54.3 parse-bug fix): a tight output
    # budget + no reasoning keeps the light model from spending its whole budget on
    # hidden reasoning and returning a 1-token/empty body that fails JSON parse and
    # then fail-closed blocks a benign command. Registered numeric SSOT (no inline literals).
    "OUROBOROS_SAFETY_MAX_TOKENS": 2000,
    "OUROBOROS_SAFETY_CALL_TIMEOUT_SEC": 60,
    # v6.54.3 transport-timeout SSOT (deadline package D). web_search: the OpenAI
    # streaming SDK call ran with NO client timeout, so the ToolEntry 540s outer cap
    # was the only (thread-kill) bound; 480 keeps the transport failure cleanly
    # messaged below that cap. LLM no_proxy read/write floor: was a hardcoded 3600s —
    # 2700 still leaves generous headroom for long silent reasoning (scope review /
    # deep self-review can think 20-40 min before the first byte) while a dead
    # socket no longer pins a worker for a full hour.
    "OUROBOROS_WEBSEARCH_TIMEOUT_SEC": 480,
    "OUROBOROS_LLM_TRANSPORT_READ_TIMEOUT_SEC": 2700,
    # v6.54.3 (1.5): plan_task deadline scaling. With a task deadline, the planning
    # swarm's wait ceiling is min(configured ceiling, remaining/4); below this floor
    # planning cannot return anything useful in time, so plan_task SKIPS with a typed
    # reason + telemetry instead of eating the tail of the budget (TB2.1: plan_task
    # was structurally irrational under a 900s deadline — ceiling 900s + wrapper 1520s).
    "OUROBOROS_PLAN_TASK_DEADLINE_MIN_SEC": 300,
    # v6.54.4 acceptance-review budget layer (task_pacing SSOT). est_sec: how long
    # one review/improvement pass roughly takes (gates review launch above the
    # finalization reserve). max passes default 1 = the historical single bounded
    # improvement pass. reserve pct: finalization reserve = max(grace, pct×total).
    "OUROBOROS_ACCEPTANCE_REVIEW_EST_SEC": 90,
    "OUROBOROS_ACCEPTANCE_MAX_IMPROVEMENT_PASSES": 1,
    "OUROBOROS_ACCEPTANCE_RESERVE_PCT": 5,
    # Reasoning effort per task type: none | low | medium | high
    "OUROBOROS_EFFORT_TASK": "medium",
    "OUROBOROS_EFFORT_EVOLUTION": "high",
    "OUROBOROS_EFFORT_REVIEW": "medium",
    "OUROBOROS_EFFORT_SCOPE_REVIEW": "high",
    "OUROBOROS_EFFORT_DEEP_SELF_REVIEW": "high",
    "OUROBOROS_EFFORT_CONSCIOUSNESS": "high",
    "OUROBOROS_RETURN_REASONING": True,
    "OUROBOROS_REASONING_SUMMARY": "auto",
    "GITHUB_TOKEN": "",
    "GITHUB_REPO": "",
    # Local model (llama-cpp-python server)
    "LOCAL_MODEL_SOURCE": "",
    "LOCAL_MODEL_FILENAME": "",
    "LOCAL_MODEL_PORT": 8766,
    "LOCAL_MODEL_N_GPU_LAYERS": 0,
    "LOCAL_MODEL_CONTEXT_LENGTH": 16384,
    "LOCAL_MODEL_CHAT_FORMAT": "",
    "USE_LOCAL_MAIN": False,
    "USE_LOCAL_HEAVY": False,
    "USE_LOCAL_LIGHT": False,
    "USE_LOCAL_CONSCIOUSNESS": False,
    "USE_LOCAL_FALLBACK": False,
    "OUROBOROS_FILE_BROWSER_DEFAULT": "",
    # Subagent depth at/below which an EXPLICIT main/heavy lane is honored; deeper
    # descendants (grandchildren) fall to light as a cost guard. Owner-configurable
    # (advanced); a visible note is surfaced when an explicit request is depth-capped.
    "OUROBOROS_SUBAGENT_CAPABILITY_DEPTH_LIMIT": 1,
    # 429-aware cross-model fallback: process-local cooldown for transiently failing
    # models (429/5xx/overloaded), passive heal-back. Owner-tunable; default-on, fail-soft.
    "OUROBOROS_FALLBACK_COOLDOWN_ENABLED": True,
    "OUROBOROS_FALLBACK_COOLDOWN_SEC": 120,
    "OUROBOROS_FALLBACK_ATTEMPTS_PER_MODEL": 1,
}


def _main_model() -> str:
    return (
        str(os.environ.get("OUROBOROS_MODEL", "") or "").strip()
        or str(SETTINGS_DEFAULTS["OUROBOROS_MODEL"])
    )


def get_light_model() -> str:
    """Return the light-model slot; empty falls back to OUROBOROS_MODEL (only main
    carries a real default — heavy/light/consciousness are empty->main)."""
    return str(os.environ.get("OUROBOROS_MODEL_LIGHT", "") or "").strip() or _main_model()


def get_heavy_model() -> str:
    """Return the heavy (strong acting/coding) lane slot; empty falls back to
    OUROBOROS_MODEL. Renamed from the legacy code slot."""
    return str(os.environ.get("OUROBOROS_MODEL_HEAVY", "") or "").strip() or _main_model()


def get_vision_model() -> str:
    """Return the vision/caption model slot; empty falls back to OUROBOROS_MODEL."""
    return str(os.environ.get("OUROBOROS_MODEL_VISION", "") or "").strip() or _main_model()


def get_image_input_mode() -> str:
    raw = str(os.environ.get("OUROBOROS_IMAGE_INPUT_MODE", SETTINGS_DEFAULTS["OUROBOROS_IMAGE_INPUT_MODE"]) or "").strip().lower()
    return raw if raw in {"auto", "caption", "inline", "off"} else "auto"


def parse_fallback_chain() -> list[str]:
    """Parse the raw ordered cross-model fallback chain — SSOT for every consumer
    (resilience walk, pricing categorization, credentialed-model resolution).

    Reads OUROBOROS_MODEL_FALLBACKS, then the legacy singular OUROBOROS_MODEL_FALLBACK
    (env-only back-compat). No dedup, no active-model drop, and NO SETTINGS_DEFAULTS
    injection: an EXPLICITLY empty Fallbacks slot means "no cross-model fallback" (so an
    OpenAI-compatible / local owner who clears it is not silently routed to the default
    Anthropic chain into an unconfigured provider). The shipped default reaches a default
    install through apply_settings_to_env, which writes the non-empty default into env."""
    raw = (
        str(os.environ.get("OUROBOROS_MODEL_FALLBACKS", "") or "").strip()
        or str(os.environ.get("OUROBOROS_MODEL_FALLBACK", "") or "").strip()
    )
    return [m.strip() for m in _parse_model_list(raw) if str(m or "").strip()]


def get_fallback_models(active_model: str = "") -> list[str]:
    """Return the ordered cross-model resilience CHAIN (deduped, with the active model
    removed so a benchmark all-slots-one-model setup collapses the chain to a no-op)."""
    out: list[str] = []
    seen = set()
    active = str(active_model or "").strip()
    for m in parse_fallback_chain():
        if m and m != active and m not in seen:
            seen.add(m)
            out.append(m)
    return out


# v6.39 slot rename-alias migration (same shape as the retention-key rename):
# OUROBOROS_MODEL_CODE -> _HEAVY, USE_LOCAL_CODE -> USE_LOCAL_HEAVY,
# OUROBOROS_MODEL_FALLBACK -> _FALLBACKS.
_LEGACY_SLOT_RENAMES = (
    ("OUROBOROS_MODEL_CODE", "OUROBOROS_MODEL_HEAVY"),
    ("OUROBOROS_VISION_MODEL", "OUROBOROS_MODEL_VISION"),
    ("USE_LOCAL_CODE", "USE_LOCAL_HEAVY"),
    ("OUROBOROS_MODEL_FALLBACK", "OUROBOROS_MODEL_FALLBACKS"),
)


def migrate_legacy_slot_keys(settings: dict) -> dict:
    """In-place v6.39 slot rename-alias migration. Preserves a stored value (never orphans
    an owner customization), then drops the legacy key so it does not linger. Shared SSOT
    for every settings entry point (load_settings AND the Colab settings builder) so a
    Drive/legacy settings file is migrated the same way regardless of how it is loaded."""
    for _old, _new in _LEGACY_SLOT_RENAMES:
        if _new not in settings and _old in settings:
            settings[_new] = settings[_old]
        settings.pop(_old, None)
    return settings


def get_consciousness_model() -> str:
    """Return the high-horizon background-consciousness model slot."""

    configured = str(os.environ.get("OUROBOROS_MODEL_CONSCIOUSNESS", "") or "").strip()
    if configured:
        return configured
    return (
        str(os.environ.get("OUROBOROS_MODEL", "") or "").strip()
        or str(SETTINGS_DEFAULTS["OUROBOROS_MODEL"])
    )

# v6.57.0 — EFFORT_SCALE: ORDERED reasoning-effort SSOT (low→high), the single place a tier
# is defined (settings, llm.py builder, switch_model enum, subagent lanes). xhigh/max extend
# none..high; llm.py clamps a request DOWN to each model's learned ceiling (BIBLE P1: disclosed).
EFFORT_SCALE: tuple[str, ...] = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
_VALID_EFFORTS = EFFORT_SCALE


def effort_rank(value: str) -> int:
    """Index of an effort in EFFORT_SCALE (−1 if unknown). Strength-ordering SSOT."""
    v = str(value or "").strip().lower()
    return EFFORT_SCALE.index(v) if v in EFFORT_SCALE else -1


def clamp_effort_to(value: str, ceiling: str) -> str:
    """Clamp ``value`` down to ``ceiling`` on EFFORT_SCALE; unknown inputs pass through."""
    vi, ci = effort_rank(value), effort_rank(ceiling)
    return ceiling if (vi >= 0 and ci >= 0 and vi > ci) else str(value or "").strip().lower()


def effort_one_step_down(value: str) -> str:
    """Next-lower effort on EFFORT_SCALE (reject-and-retry walk); floors at `none`."""
    idx = effort_rank(value)
    return EFFORT_SCALE[idx - 1] if idx > 0 else ("none" if idx == 0 else "medium")


_DIRECT_PROVIDER_REVIEW_RUNS = 3

# Runtime mode and review enforcement are separate axes.
VALID_RUNTIME_MODES = ("light", "advanced", "pro")

# Context mode is an independent, owner-controlled working-context size profile
# (low/max). Unlike runtime mode it is NOT boot-pinned — it is not a privilege
# boundary, so it hot-applies on the next task.
VALID_CONTEXT_MODES = ("low", "max")

# Lower rank = stricter scope. ``save_settings`` refuses agent self-elevation.
_RUNTIME_MODE_RANK = {"light": 0, "advanced": 1, "pro": 2}

# Boot-time runtime-mode baseline. Pinning the owner-selected mode after
# settings load prevents an out-of-process settings edit from becoming the new
# baseline through a later load/save round-trip. The pin is also exported via
# ``OUROBOROS_BOOT_RUNTIME_MODE`` so fresh subprocess imports inherit the same
# ratchet; a child can clobber only its own env, not the parent's in-memory pin.
_BOOT_RUNTIME_MODE: Optional[str] = None
BOOT_RUNTIME_MODE_ENV_KEY = "OUROBOROS_BOOT_RUNTIME_MODE"


def _resolve_baseline_from_env() -> Optional[str]:
    """Return the parent-pinned runtime-mode baseline inherited via env."""
    raw = os.environ.get(BOOT_RUNTIME_MODE_ENV_KEY, "")
    if not raw:
        return None
    return normalize_runtime_mode(raw)


def initialize_runtime_mode_baseline(mode: Optional[str] = None) -> None:
    """Pin the immutable runtime-mode baseline before any agent code runs.

    Call after ``load_settings``/``apply_settings_to_env`` and before worker or
    supervisor startup. The pin is exported as ``OUROBOROS_BOOT_RUNTIME_MODE``
    so subprocesses enforce the same owner-selected baseline.
    """
    global _BOOT_RUNTIME_MODE
    if _BOOT_RUNTIME_MODE is not None:
        return
    if mode is None:
        # Prefer the parent-exported BOOT key; RUNTIME_MODE is mutable app state.
        inherited = _resolve_baseline_from_env()
        if inherited is not None:
            mode = inherited
        else:
            mode = os.environ.get("OUROBOROS_RUNTIME_MODE", "advanced") or "advanced"
    _BOOT_RUNTIME_MODE = normalize_runtime_mode(mode)
    # Propagate the pin to subprocesses.
    os.environ[BOOT_RUNTIME_MODE_ENV_KEY] = _BOOT_RUNTIME_MODE


def reset_runtime_mode_baseline_for_tests() -> None:
    """Test-only helper to clear the pinned baseline and env export."""
    global _BOOT_RUNTIME_MODE
    _BOOT_RUNTIME_MODE = None
    os.environ.pop(BOOT_RUNTIME_MODE_ENV_KEY, None)


def _parse_model_list(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _exclusive_direct_remote_provider_env() -> str:
    has_openrouter = bool(str(os.environ.get("OPENROUTER_API_KEY", "") or "").strip())
    has_openai = bool(str(os.environ.get("OPENAI_API_KEY", "") or "").strip())
    has_anthropic = bool(str(os.environ.get("ANTHROPIC_API_KEY", "") or "").strip())
    has_legacy_base = bool(str(os.environ.get("OPENAI_BASE_URL", "") or "").strip())
    has_compatible = bool(str(os.environ.get("OPENAI_COMPATIBLE_BASE_URL", "") or "").strip())
    has_cloudru = bool(str(os.environ.get("CLOUDRU_FOUNDATION_MODELS_API_KEY", "") or "").strip())
    has_gigachat = bool(str(os.environ.get("GIGACHAT_CREDENTIALS", "") or "").strip()) or (
        bool(str(os.environ.get("GIGACHAT_USER", "") or "").strip())
        and bool(str(os.environ.get("GIGACHAT_PASSWORD", "") or "").strip())
    )
    # OpenRouter / legacy OpenAI base / OpenAI-compatible all route through the
    # OpenRouter-style stack, so their presence means "not an exclusive direct
    # provider". Among the real direct providers (official OpenAI, Anthropic,
    # Cloud.ru, GigaChat), return one only when exactly one is configured.
    if has_openrouter or has_legacy_base or has_compatible:
        return ""
    direct = [
        name for name, present in (
            ("openai", has_openai),
            ("anthropic", has_anthropic),
            ("cloudru", has_cloudru),
            ("gigachat", has_gigachat),
        ) if present
    ]
    return direct[0] if len(direct) == 1 else ""


def resolve_effort(task_type: str) -> str:
    """Return the configured reasoning effort for the given task type."""
    t = (task_type or "").lower().strip()

    if t == "evolution":
        key = "OUROBOROS_EFFORT_EVOLUTION"
        default = "high"
    elif t == "review":
        key = "OUROBOROS_EFFORT_REVIEW"
        default = "medium"
    elif t == "deep_self_review":
        key = "OUROBOROS_EFFORT_DEEP_SELF_REVIEW"
        default = "high"
    elif t in ("scope_review", "scope-review"):
        key = "OUROBOROS_EFFORT_SCOPE_REVIEW"
        default = "high"
    elif t == "consciousness":
        key = "OUROBOROS_EFFORT_CONSCIOUSNESS"
        default = "high"
    else:
        # Legacy INITIAL_REASONING_EFFORT is retired; use EFFORT_TASK.
        key = "OUROBOROS_EFFORT_TASK"
        default = "medium"

    raw = os.environ.get(key, default)
    return raw if raw in _VALID_EFFORTS else default


def direct_provider_review_models_fallback(provider: str) -> list[str]:
    """Return the exact review-models list a direct-provider fallback emits."""
    if provider not in ("openai", "anthropic", "cloudru", "gigachat"):
        return []
    main_model = str(
        os.environ.get("OUROBOROS_MODEL", SETTINGS_DEFAULTS["OUROBOROS_MODEL"]) or ""
    ).strip()
    main_model = migrate_model_value(provider, main_model)
    user_light_raw = str(os.environ.get("OUROBOROS_MODEL_LIGHT", "") or "").strip()
    return compute_direct_review_models_fallback(
        provider,
        main_model,
        user_light_raw,
        review_runs=_DIRECT_PROVIDER_REVIEW_RUNS,
    )


def adaptive_quorum(n_slots: int) -> int:
    """Reviewer-quorum SSOT for an ARBITRARY configured slot count, reused by
    triad/scope/plan/skill/acceptance review. A single configured reviewer needs
    1 (a loud single_reviewer_no_diversity degraded mode), 2 need both, 3+ keep
    the classic 2-of-N majority. This honors an explicit small reviewer config
    (Bible P3 stays loud); it is DISTINCT from "configured >= quorum but fewer
    responded", which remains a loud infra quorum FAILURE at the call site."""
    return 2 if n_slots >= 3 else max(1, n_slots)


def get_review_models() -> list[str]:
    """Return the configured pre-commit review model list."""
    default_str = SETTINGS_DEFAULTS["OUROBOROS_REVIEW_MODELS"]
    models_str = os.environ.get("OUROBOROS_REVIEW_MODELS", default_str) or default_str
    models = _parse_model_list(models_str)
    provider = _exclusive_direct_remote_provider_env()
    if not provider:
        return models

    main_model = str(os.environ.get("OUROBOROS_MODEL", SETTINGS_DEFAULTS["OUROBOROS_MODEL"]) or "").strip()
    main_model = migrate_model_value(provider, main_model)
    provider_prefix = f"{provider}::"
    if not main_model.startswith(provider_prefix):
        return models

    migrated = [migrate_model_value(provider, model) for model in models]
    if not migrated or any(not model.startswith(provider_prefix) for model in migrated):
        # Auto-expand to the [main]*N stochastic fallback ONLY when nothing
        # usable is configured (empty, or foreign models in an exclusive
        # direct-provider setup). An explicit provider-matching list — including
        # a single model — is honored exactly (duplicates are valid stochastic
        # slots, at the owner's discretion).
        return direct_provider_review_models_fallback(provider)
    return migrated


def get_review_enforcement() -> str:
    """Return the configured pre-commit review enforcement mode."""
    default_val = str(SETTINGS_DEFAULTS["OUROBOROS_REVIEW_ENFORCEMENT"])
    raw = (os.environ.get("OUROBOROS_REVIEW_ENFORCEMENT", default_val) or default_val).strip().lower()
    return raw if raw in {"advisory", "blocking"} else default_val


def get_scope_review_models() -> list[str]:
    """Return configured scope reviewer slots, preserving duplicate model IDs."""
    default_str = str(SETTINGS_DEFAULTS["OUROBOROS_SCOPE_REVIEW_MODELS"])
    raw = os.environ.get("OUROBOROS_SCOPE_REVIEW_MODELS", "") or ""
    if not raw.strip():
        raw = os.environ.get("OUROBOROS_SCOPE_REVIEW_MODEL", default_str) or default_str
    models = _parse_model_list(raw)
    singular = str(os.environ.get("OUROBOROS_SCOPE_REVIEW_MODEL", SETTINGS_DEFAULTS["OUROBOROS_SCOPE_REVIEW_MODEL"]) or "").strip()
    if not models and singular:
        models = [singular]
    if not models:
        models = _parse_model_list(default_str)
    provider = _exclusive_direct_remote_provider_env()
    if not provider:
        return models
    migrated = [migrate_model_value(provider, model) for model in models]
    provider_prefix = f"{provider}::"
    if migrated and all(model.startswith(provider_prefix) for model in migrated):
        return migrated
    migrated_singular = migrate_model_value(provider, singular or SETTINGS_DEFAULTS["OUROBOROS_SCOPE_REVIEW_MODEL"])
    if migrated_singular.startswith(provider_prefix):
        return [migrated_singular]
    fallback = direct_provider_review_models_fallback(provider)
    return fallback[:1] if fallback else migrated


def get_deep_self_review_model() -> str:
    """Return the configured deep self-review model slot."""

    return (
        str(os.environ.get("OUROBOROS_MODEL_DEEP_SELF_REVIEW", "") or "").strip()
        or str(SETTINGS_DEFAULTS["OUROBOROS_MODEL_DEEP_SELF_REVIEW"])
    )


def get_max_workers() -> int:
    raw = os.environ.get("OUROBOROS_MAX_WORKERS", SETTINGS_DEFAULTS["OUROBOROS_MAX_WORKERS"])
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        parsed = int(SETTINGS_DEFAULTS["OUROBOROS_MAX_WORKERS"])
    return max(1, parsed)


def get_task_idle_timeout_sec() -> int:
    """Idle window before a task is eligible for an activity-based stop: it has made
    no REAL progress (its own last_progress_at) AND has no progressing subtree for
    this long. The periodic 30s process heartbeat is liveness, NOT progress."""
    raw = os.environ.get(
        "OUROBOROS_TASK_IDLE_TIMEOUT_SEC", SETTINGS_DEFAULTS["OUROBOROS_TASK_IDLE_TIMEOUT_SEC"]
    )
    try:
        return max(60, int(raw))
    except (TypeError, ValueError):
        return int(SETTINGS_DEFAULTS["OUROBOROS_TASK_IDLE_TIMEOUT_SEC"])


def get_task_abs_ceiling_sec() -> int:
    """Absolute wall-clock backstop per task, independent of activity — the only hard
    time axis (budget/cost is the other, separate hard axis). A productively-waiting
    orchestrator survives to this ceiling instead of a flat 1800s wall-clock kill."""
    raw = os.environ.get(
        "OUROBOROS_TASK_ABS_CEILING_SEC", SETTINGS_DEFAULTS["OUROBOROS_TASK_ABS_CEILING_SEC"]
    )
    try:
        return max(300, int(raw))
    except (TypeError, ValueError):
        return int(SETTINGS_DEFAULTS["OUROBOROS_TASK_ABS_CEILING_SEC"])


def get_per_call_timeout_ceiling_sec() -> int:
    """SSOT ceiling for an explicit per-call run_command/run_script timeout_sec
    (and the outer tool-execution cap that accommodates it)."""
    raw = os.environ.get(
        "OUROBOROS_PER_CALL_TIMEOUT_CEILING_SEC",
        SETTINGS_DEFAULTS["OUROBOROS_PER_CALL_TIMEOUT_CEILING_SEC"],
    )
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return int(SETTINGS_DEFAULTS["OUROBOROS_PER_CALL_TIMEOUT_CEILING_SEC"])


def get_plan_task_swarm_timeout_sec() -> float:
    raw = os.environ.get(
        "OUROBOROS_PLAN_TASK_SWARM_TIMEOUT_SEC",
        SETTINGS_DEFAULTS["OUROBOROS_PLAN_TASK_SWARM_TIMEOUT_SEC"],
    )
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        parsed = float(SETTINGS_DEFAULTS["OUROBOROS_PLAN_TASK_SWARM_TIMEOUT_SEC"])
    return max(0.0, parsed)


def get_plan_task_swarm_max_wait_sec() -> float:
    raw = os.environ.get(
        "OUROBOROS_PLAN_TASK_SWARM_MAX_WAIT_SEC",
        SETTINGS_DEFAULTS["OUROBOROS_PLAN_TASK_SWARM_MAX_WAIT_SEC"],
    )
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        parsed = float(SETTINGS_DEFAULTS["OUROBOROS_PLAN_TASK_SWARM_MAX_WAIT_SEC"])
    return max(0.0, parsed)


def get_plan_task_swarm_heartbeat_stale_sec() -> float:
    raw = os.environ.get(
        "OUROBOROS_PLAN_TASK_SWARM_HEARTBEAT_STALE_SEC",
        SETTINGS_DEFAULTS["OUROBOROS_PLAN_TASK_SWARM_HEARTBEAT_STALE_SEC"],
    )
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        parsed = float(SETTINGS_DEFAULTS["OUROBOROS_PLAN_TASK_SWARM_HEARTBEAT_STALE_SEC"])
    return max(0.0, parsed)


def get_restart_drain_max_sec() -> int:
    raw = os.environ.get(
        "OUROBOROS_RESTART_DRAIN_MAX_SEC",
        SETTINGS_DEFAULTS["OUROBOROS_RESTART_DRAIN_MAX_SEC"],
    )
    try:
        parsed = int(float(raw))
    except (TypeError, ValueError):
        parsed = int(SETTINGS_DEFAULTS["OUROBOROS_RESTART_DRAIN_MAX_SEC"])
    return max(0, parsed)


def get_post_task_evolution_enabled() -> bool:
    """V4 envelope: is owner-enabled post-task self-evolution on? Default OFF."""
    raw = str(os.environ.get(
        "OUROBOROS_POST_TASK_EVOLUTION",
        SETTINGS_DEFAULTS["OUROBOROS_POST_TASK_EVOLUTION"],
    ) or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


_EVERY_N_CADENCE_RE = re.compile(r"^every_n:[1-9][0-9]*$")


def is_valid_post_task_evolution_cadence(raw: str) -> bool:
    """SSOT predicate: True iff `raw` is an exact valid cadence — 'off' | 'llm' |
    'every_n:<positive int>'. Used both at read time (normalize) and at the API
    boundary (reject), so a malformed value (every_n:0, every_nonsense, typos) can
    never silently force an evolution cycle after every task."""
    value = str(raw or "").strip().lower()
    return value in {"off", "llm"} or bool(_EVERY_N_CADENCE_RE.match(value))


def get_post_task_evolution_cadence() -> str:
    """Cadence for post-task evolution: 'off' | 'llm' | 'every_n:<k>'. Default 'llm'.
    Unknown/malformed values normalize to 'llm' so a typo can never silently force
    an evolution cycle after every task."""
    raw = str(os.environ.get(
        "OUROBOROS_POST_TASK_EVOLUTION_CADENCE",
        SETTINGS_DEFAULTS["OUROBOROS_POST_TASK_EVOLUTION_CADENCE"],
    ) or "").strip().lower()
    return raw if is_valid_post_task_evolution_cadence(raw) else "llm"


def get_evolution_persistent_objective() -> str:
    """Optional owner-set standing steer APPENDED to each evolution cycle's
    objective. Never overrides the LLM-first promotion; empty = pure LLM choice."""
    return str(os.environ.get(
        "OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE",
        SETTINGS_DEFAULTS["OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE"],
    ) or "").strip()


def get_post_task_evolution_budget_usd() -> float:
    """Optional per-window USD budget for post-task evolution (0 = use the
    existing EVOLUTION_BUDGET_RESERVE / TOTAL_BUDGET gating only)."""
    raw = os.environ.get(
        "OUROBOROS_POST_TASK_EVOLUTION_BUDGET_USD",
        SETTINGS_DEFAULTS["OUROBOROS_POST_TASK_EVOLUTION_BUDGET_USD"],
    )
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.0


def _bounded_positive_int_setting(key: str, *, default: int, hard_max: int) -> int:
    raw = os.environ.get(key, SETTINGS_DEFAULTS.get(key, default))
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        parsed = default
    if parsed < 1:
        parsed = default
    return max(1, min(parsed, hard_max))


def get_max_active_subagents_per_root() -> int:
    return _bounded_positive_int_setting(
        "OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT",
        default=int(SETTINGS_DEFAULTS["OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT"]),
        hard_max=50,
    )


def get_max_subagent_depth() -> int:
    return _bounded_positive_int_setting(
        "OUROBOROS_MAX_SUBAGENT_DEPTH",
        default=int(SETTINGS_DEFAULTS["OUROBOROS_MAX_SUBAGENT_DEPTH"]),
        hard_max=10,
    )


def get_allow_mutative_subagents() -> bool:
    """Whether the parent may spawn mutative (acting) subagents.

    Owner-controlled. Empty/unset => follow runtime mode (ON in advanced/pro,
    OFF in light). Explicit truthy/falsey overrides. This only gates whether
    acting subagents may be SCHEDULED; light-mode self-repo writes still stay
    blocked by the runtime sandbox regardless of this toggle.
    """
    key = "OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS"
    raw = os.environ.get(key, SETTINGS_DEFAULTS.get(key, ""))
    text = str(raw or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return get_runtime_mode() in {"advanced", "pro"}


def get_subagent_worktree_root() -> str:
    """Filesystem root for acting self_worktree checkouts (outside repo/ and data/)."""
    raw = str(
        os.environ.get("OUROBOROS_SUBAGENT_WORKTREE_ROOT", "")
        or SETTINGS_DEFAULTS.get("OUROBOROS_SUBAGENT_WORKTREE_ROOT", "")
    ).strip()
    return raw or os.path.expanduser(os.path.join("~", "Ouroboros", "subagent_worktrees"))


def get_subagent_projects_root() -> str:
    """Durable root for genesis ("from scratch") subagent projects.

    Outside repo/ and data/. Unlike self_worktree checkouts, genesis projects are
    durable deliverables and are never age-pruned by the GC retention sweep."""
    raw = str(
        os.environ.get("OUROBOROS_SUBAGENT_PROJECTS_ROOT", "")
        or SETTINGS_DEFAULTS.get("OUROBOROS_SUBAGENT_PROJECTS_ROOT", "")
    ).strip()
    return raw or os.path.expanduser(os.path.join("~", "Ouroboros", "projects"))


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


def get_deliverables_root() -> str:
    """Visible container for UNNAMED user deliverables: a bare filename (no directory) lands here
    instead of cluttering the home root. Sibling of the genesis projects root under ~/Ouroboros,
    outside data/, and never GC-pruned. An explicit placement (Desktop/..., Downloads/..., or any
    path WITH a directory) is always honored as given. Override with OUROBOROS_DELIVERABLES_ROOT."""
    raw = str(
        os.environ.get("OUROBOROS_DELIVERABLES_ROOT", "")
        or SETTINGS_DEFAULTS.get("OUROBOROS_DELIVERABLES_ROOT", "")
    ).strip()
    return raw or os.path.expanduser(os.path.join("~", "Ouroboros", "Deliverables"))


def get_task_review_mode() -> str:
    default_val = str(SETTINGS_DEFAULTS["OUROBOROS_TASK_REVIEW_MODE"])
    raw = (os.environ.get("OUROBOROS_TASK_REVIEW_MODE", default_val) or default_val).strip().lower()
    return raw if raw in {"off", "auto", "required"} else default_val


def get_auto_grant_enabled() -> bool:
    """Return whether reviewed skills should receive requested grants."""
    key = "OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS"
    raw = None
    try:
        if SETTINGS_PATH.exists():
            disk = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(disk, dict) and key in disk:
                raw = disk.get(key)
    except Exception:
        raw = None
    if raw is None:
        raw = os.environ.get(key, SETTINGS_DEFAULTS[key])
    raw = str(raw or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def get_trust_native_seeded_skills() -> bool:
    """Whether launcher-seeded native skills get the hash-pinned trust verdict."""
    key = "OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS"
    raw = None
    try:
        if SETTINGS_PATH.exists():
            disk = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(disk, dict) and key in disk:
                raw = disk.get(key)
    except Exception:
        raw = None
    if raw is None:
        raw = os.environ.get(key, SETTINGS_DEFAULTS[key])
    raw = str(raw or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def normalize_runtime_mode(value: Any) -> str:
    """Clamp caller-supplied runtime mode to the canonical closed enum."""
    default_val = str(SETTINGS_DEFAULTS["OUROBOROS_RUNTIME_MODE"])
    text = str(value or "").strip().lower()
    return text if text in VALID_RUNTIME_MODES else default_val


def get_runtime_mode() -> str:
    """Return the configured runtime mode (light / advanced / pro)."""
    default_val = str(SETTINGS_DEFAULTS["OUROBOROS_RUNTIME_MODE"])
    if _BOOT_RUNTIME_MODE is not None:
        return normalize_runtime_mode(_BOOT_RUNTIME_MODE)
    inherited = _resolve_baseline_from_env()
    if inherited is not None:
        return normalize_runtime_mode(inherited)
    return normalize_runtime_mode(os.environ.get("OUROBOROS_RUNTIME_MODE", default_val) or default_val)


VALID_SAFETY_MODES = ("full", "light", "off")


def normalize_safety_mode(value: Any) -> str:
    """Clamp caller-supplied safety mode to the closed enum (full / light / off)."""
    default_val = str(SETTINGS_DEFAULTS["OUROBOROS_SAFETY_MODE"])
    text = str(value or "").strip().lower()
    return text if text in VALID_SAFETY_MODES else default_val


def get_safety_mode() -> str:
    """Return the owner-selected LLM-safety-supervisor coverage (full | light | off).

    Owner-only at the write surface (dropped from the agent-reachable /api/settings
    POST; flows only through the dedicated audited owner endpoint), so the agent
    cannot lower its own safety coverage. Deterministic registry sandbox, protected
    paths, and light-mode guards run regardless of this mode (BIBLE P3: the LLM
    supervisor is a layer, not the floor)."""
    default_val = str(SETTINGS_DEFAULTS["OUROBOROS_SAFETY_MODE"])
    return normalize_safety_mode(os.environ.get("OUROBOROS_SAFETY_MODE", default_val) or default_val)


def get_safety_max_tokens() -> int:
    """Output-token budget for safety-supervisor LLM calls (parse-bug fix)."""
    try:
        val = int(os.environ.get("OUROBOROS_SAFETY_MAX_TOKENS", "") or SETTINGS_DEFAULTS["OUROBOROS_SAFETY_MAX_TOKENS"])
    except (TypeError, ValueError):
        val = int(SETTINGS_DEFAULTS["OUROBOROS_SAFETY_MAX_TOKENS"])
    return max(256, min(val, 16384))


def get_safety_call_timeout_sec() -> float:
    """Transport timeout for safety-supervisor LLM calls (prevents indefinite hang)."""
    try:
        val = float(os.environ.get("OUROBOROS_SAFETY_CALL_TIMEOUT_SEC", "") or SETTINGS_DEFAULTS["OUROBOROS_SAFETY_CALL_TIMEOUT_SEC"])
    except (TypeError, ValueError):
        val = float(SETTINGS_DEFAULTS["OUROBOROS_SAFETY_CALL_TIMEOUT_SEC"])
    return max(5.0, min(val, 600.0))


def get_websearch_timeout_sec() -> float:
    """Transport timeout for the web_search OpenAI streaming call (v6.54.3, D)."""
    try:
        val = float(os.environ.get("OUROBOROS_WEBSEARCH_TIMEOUT_SEC", "") or SETTINGS_DEFAULTS["OUROBOROS_WEBSEARCH_TIMEOUT_SEC"])
    except (TypeError, ValueError):
        val = float(SETTINGS_DEFAULTS["OUROBOROS_WEBSEARCH_TIMEOUT_SEC"])
    return max(30.0, min(val, 3600.0))


def get_llm_transport_read_timeout_sec() -> float:
    """Default httpx read/write timeout for no_proxy LLM clients (v6.54.3, D).

    Generous by design: long silent reasoning (scope review, deep self-review)
    can take 20-40 min before the first byte. This is the DEAD-SOCKET bound,
    not a latency target; explicit per-call timeouts always win."""
    try:
        val = float(os.environ.get("OUROBOROS_LLM_TRANSPORT_READ_TIMEOUT_SEC", "") or SETTINGS_DEFAULTS["OUROBOROS_LLM_TRANSPORT_READ_TIMEOUT_SEC"])
    except (TypeError, ValueError):
        val = float(SETTINGS_DEFAULTS["OUROBOROS_LLM_TRANSPORT_READ_TIMEOUT_SEC"])
    return max(60.0, min(val, 7200.0))


def get_acceptance_review_est_sec() -> float:
    """Estimated duration of one acceptance review/improvement pass (v6.54.4)."""
    try:
        val = float(os.environ.get("OUROBOROS_ACCEPTANCE_REVIEW_EST_SEC", "") or SETTINGS_DEFAULTS["OUROBOROS_ACCEPTANCE_REVIEW_EST_SEC"])
    except (TypeError, ValueError):
        val = float(SETTINGS_DEFAULTS["OUROBOROS_ACCEPTANCE_REVIEW_EST_SEC"])
    return max(10.0, min(val, 3600.0))


def get_acceptance_max_improvement_passes() -> int:
    """Default COUNT cap for acceptance-review improvement passes (v6.54.4)."""
    try:
        val = int(os.environ.get("OUROBOROS_ACCEPTANCE_MAX_IMPROVEMENT_PASSES", "") or SETTINGS_DEFAULTS["OUROBOROS_ACCEPTANCE_MAX_IMPROVEMENT_PASSES"])
    except (TypeError, ValueError):
        val = int(SETTINGS_DEFAULTS["OUROBOROS_ACCEPTANCE_MAX_IMPROVEMENT_PASSES"])
    return max(0, min(val, 20))


def get_acceptance_reserve_pct() -> int:
    """Default finalization-reserve percentage of the total budget (v6.54.4)."""
    try:
        val = int(os.environ.get("OUROBOROS_ACCEPTANCE_RESERVE_PCT", "") or SETTINGS_DEFAULTS["OUROBOROS_ACCEPTANCE_RESERVE_PCT"])
    except (TypeError, ValueError):
        val = int(SETTINGS_DEFAULTS["OUROBOROS_ACCEPTANCE_RESERVE_PCT"])
    return max(0, min(val, 50))


def get_plan_task_deadline_min_sec() -> float:
    """Minimum useful deadline-scaled planning-swarm window (v6.54.3, 1.5)."""
    try:
        val = float(os.environ.get("OUROBOROS_PLAN_TASK_DEADLINE_MIN_SEC", "") or SETTINGS_DEFAULTS["OUROBOROS_PLAN_TASK_DEADLINE_MIN_SEC"])
    except (TypeError, ValueError):
        val = float(SETTINGS_DEFAULTS["OUROBOROS_PLAN_TASK_DEADLINE_MIN_SEC"])
    return max(30.0, min(val, 3600.0))


def normalize_context_mode(value: Any) -> str:
    """Clamp caller-supplied context mode to the closed enum (low / max)."""
    default_val = str(SETTINGS_DEFAULTS["OUROBOROS_CONTEXT_MODE"])
    text = str(value or "").strip().lower()
    return text if text in VALID_CONTEXT_MODES else default_val


def get_context_mode() -> str:
    """Return the owner-selected working-context mode (low | max).

    Unlike runtime mode there is NO boot-pin: context mode is not a privilege
    boundary, so it hot-applies on the next task. It stays owner-only at the
    write surface (dropped from the agent-reachable /api/settings POST), so the
    agent cannot lower its own cognitive horizon (BIBLE P1 cognitive-horizon).
    """
    default_val = str(SETTINGS_DEFAULTS["OUROBOROS_CONTEXT_MODE"])
    return normalize_context_mode(
        os.environ.get("OUROBOROS_CONTEXT_MODE", default_val) or default_val
    )


def _settings_file_context_mode(default: str = "max") -> str:
    """Read the persisted/current context mode without normalizing whole settings."""
    if SETTINGS_PATH.exists():
        try:
            disk_settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(disk_settings, dict):
                return normalize_context_mode(disk_settings.get("OUROBOROS_CONTEXT_MODE", default))
        except (OSError, json.JSONDecodeError):
            pass
    return normalize_context_mode(os.environ.get("OUROBOROS_CONTEXT_MODE", default) or default)


def _guard_context_mode_lowering(settings: dict, *, allow_context_lowering: bool = False) -> None:
    """Refuse agent-reachable settings writes that lower the cognitive horizon."""
    previous_mode = _settings_file_context_mode()
    next_mode = normalize_context_mode(settings.get("OUROBOROS_CONTEXT_MODE", previous_mode))
    if previous_mode == "max" and next_mode == "low" and not allow_context_lowering:
        raise PermissionError(
            "OUROBOROS_CONTEXT_MODE lowering refused: 'max' -> 'low'. "
            "Context mode is owner-controlled — use the dedicated owner endpoint/UI/CLI."
        )


_SAFETY_MODE_RANK = {"full": 2, "light": 1, "off": 0}


def _settings_file_safety_mode(default: str = "full") -> str:
    """Read the persisted/current safety mode without normalizing whole settings."""
    if SETTINGS_PATH.exists():
        try:
            disk_settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(disk_settings, dict):
                return normalize_safety_mode(disk_settings.get("OUROBOROS_SAFETY_MODE", default))
        except (OSError, json.JSONDecodeError):
            pass
    return normalize_safety_mode(os.environ.get("OUROBOROS_SAFETY_MODE", default) or default)


def _guard_safety_mode_lowering(settings: dict, *, allow_safety_lowering: bool = False) -> None:
    """Refuse agent-reachable settings writes that lower LLM-safety coverage.

    ``full -> light -> off`` is a strictly decreasing coverage ladder; any downward
    step is owner-only (mirrors the context-mode ratchet — the agent must not reduce
    its own supervision to remove friction, BIBLE P3)."""
    previous_mode = _settings_file_safety_mode()
    next_mode = normalize_safety_mode(settings.get("OUROBOROS_SAFETY_MODE", previous_mode))
    if _SAFETY_MODE_RANK[next_mode] < _SAFETY_MODE_RANK[previous_mode] and not allow_safety_lowering:
        raise PermissionError(
            f"OUROBOROS_SAFETY_MODE lowering refused: {previous_mode!r} -> {next_mode!r}. "
            "Safety mode is owner-controlled — use the dedicated /api/owner/safety-mode endpoint."
        )


def get_skills_repo_path() -> str:
    """Return the configured external skills checkout path, expanding ``~``."""
    raw = (
        os.environ.get("OUROBOROS_SKILLS_REPO_PATH", "") or ""
    ).strip()
    if not raw:
        return ""
    try:
        return str(pathlib.Path(raw).expanduser())
    except Exception:
        return raw


# Skills data layout
#
# Runtime skill packages live under ``data/skills/<source>/<slug>/``. The
# git-tracked ``repo/skills/`` tree is only a launcher seed; the optional
# ``OUROBOROS_SKILLS_REPO_PATH`` adds a user-managed checkout.

SKILL_SOURCE_NATIVE = "native"
SKILL_SOURCE_CLAWHUB = "clawhub"
SKILL_SOURCE_EXTERNAL = "external"
SKILL_SOURCE_OUROBOROSHUB = "ouroboroshub"
SKILL_SOURCE_SELF_AUTHORED = "self_authored"
SKILL_SOURCE_USER_REPO = "user_repo"

SKILL_SOURCE_SUBDIRS = (
    SKILL_SOURCE_NATIVE,
    SKILL_SOURCE_CLAWHUB,
    SKILL_SOURCE_EXTERNAL,
    SKILL_SOURCE_OUROBOROSHUB,
)


def ensure_data_skills_dir(data_dir: pathlib.Path) -> pathlib.Path:
    """Create and return the data skills root plus source subdirectories."""
    root = data_dir / "skills"
    try:
        root.mkdir(parents=True, exist_ok=True)
        for sub in SKILL_SOURCE_SUBDIRS:
            (root / sub).mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return root


def resolve_data_skills_dir(data_dir: pathlib.Path) -> Optional[pathlib.Path]:
    """Return existing ``<data_dir>/skills/`` without creating it."""
    candidate = data_dir / "skills"
    return candidate if candidate.is_dir() else None


def get_ouroboroshub_catalog_url() -> str:
    """Return the official OuroborosHub static catalog URL."""

    return str(load_settings().get("OUROBOROS_HUB_CATALOG_URL") or SETTINGS_DEFAULTS["OUROBOROS_HUB_CATALOG_URL"]).strip()


def get_ouroboroshub_skills_dir() -> pathlib.Path:
    """Return ``<DATA_DIR>/skills/ouroboroshub/`` (created on demand)."""

    target = ensure_data_skills_dir(DATA_DIR) / SKILL_SOURCE_OUROBOROSHUB
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return target


def get_clawhub_registry_url() -> str:
    """Return the normalized ClawHub registry URL; callers enforce host allowlists."""
    raw = (os.environ.get("OUROBOROS_CLAWHUB_REGISTRY_URL", "") or "").strip()
    default_url = "https://clawhub.ai/api/v1"
    if not raw:
        return default_url
    import urllib.parse as _urlparse
    components = _urlparse.urlparse(raw)
    cleaned = _urlparse.urlunparse(
        (components.scheme, components.netloc, components.path.rstrip("/"), "", "", "")
    )
    return cleaned


# Version
def read_version() -> str:
    try:
        if getattr(sys, "frozen", False):
            vp = pathlib.Path(sys._MEIPASS) / "VERSION"
        else:
            vp = pathlib.Path(__file__).parent.parent / "VERSION"
        return vp.read_text(encoding="utf-8").strip()
    except Exception:
        return "0.0.0"


# Settings file locking
_SETTINGS_LOCK = pathlib.Path(str(SETTINGS_PATH) + ".lock")


def _acquire_settings_lock(timeout: float = 2.0) -> Optional[int]:
    start = time.time()
    while time.time() - start < timeout:
        try:
            fd = os.open(str(_SETTINGS_LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            return fd
        except FileExistsError:
            try:
                if time.time() - _SETTINGS_LOCK.stat().st_mtime > 10:
                    _SETTINGS_LOCK.unlink()
                    continue
            except Exception:
                pass
            time.sleep(0.01)
        except Exception:
            break
    return None


def _release_settings_lock(fd: Optional[int]) -> None:
    if fd is not None:
        try:
            os.close(fd)
        except Exception:
            pass
    try:
        _SETTINGS_LOCK.unlink()
    except Exception:
        pass


def _coerce_setting_value(key: str, value):
    default = SETTINGS_DEFAULTS.get(key)
    # Normalize runtime mode on read so all consumers see the closed enum.
    if key == "OUROBOROS_RUNTIME_MODE":
        return normalize_runtime_mode(value)
    if key == "OUROBOROS_CONTEXT_MODE":
        return normalize_context_mode(value)
    # Trim so whitespace-only config is not treated as a configured skills repo.
    if key == "OUROBOROS_SKILLS_REPO_PATH":
        return str(value or "").strip()
    if key == "MCP_SERVERS":
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, dict)]
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            try:
                parsed = json.loads(text)
            except (TypeError, ValueError):
                return []
            if isinstance(parsed, list):
                return [dict(item) for item in parsed if isinstance(item, dict)]
        return []
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        return str(value or "").strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int) and not isinstance(default, bool):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    if isinstance(default, float):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    return str(value or "")


# Load / Save
def load_settings() -> dict:
    fd = _acquire_settings_lock()
    try:
        loaded: dict = {}
        if SETTINGS_PATH.exists():
            try:
                raw = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    loaded = {
                        key: _coerce_setting_value(key, value) if key in SETTINGS_DEFAULTS else value
                        for key, value in raw.items()
                    }
            except Exception:
                pass
        # Rename-alias migration: fold deprecated per-subsystem retention keys into
        # the unified OUROBOROS_GC_RETENTION_DAYS, then drop the legacy keys so they
        # do not linger. Prefer a CUSTOMIZED legacy value (one that differs from its
        # former default) so a user's customization is never orphaned by the rename;
        # an all-defaults file collapses to the unified default (e.g. service 14->7).
        from ouroboros.retention import LEGACY_RETENTION_KEYS, pick_legacy_retention_seed
        if "OUROBOROS_GC_RETENTION_DAYS" not in loaded:
            seed = pick_legacy_retention_seed(loaded.get)
            if seed is not None:
                loaded["OUROBOROS_GC_RETENTION_DAYS"] = seed
        for _legacy in LEGACY_RETENTION_KEYS:
            loaded.pop(_legacy, None)
        migrate_legacy_slot_keys(loaded)
        settings = dict(SETTINGS_DEFAULTS)
        settings.update(loaded)
        for key in SETTINGS_DEFAULTS:
            raw_env = os.environ.get(key)
            if raw_env is None:
                continue
            if key == "OUROBOROS_RETURN_REASONING" and raw_env == "":
                settings[key] = ""
                continue
            if raw_env == "":
                continue
            if key in loaded and settings.get(key) not in {None, ""}:
                continue
            settings[key] = _coerce_setting_value(key, raw_env)
        return settings
    finally:
        _release_settings_lock(fd)


def save_settings(
    settings: dict,
    *,
    allow_elevation: bool = False,
) -> None:
    """Persist settings and enforce owner-only mode ratchets.

    Elevation above the boot baseline is refused after initialization; then
    ``allow_elevation=True`` is inert to agent-reachable subprocesses. Production
    entry points must call ``initialize_runtime_mode_baseline`` before agent code.
    Context-mode lowering (max -> low) likewise requires an explicit owner path.
    """
    _guard_live_settings_write()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fd = _acquire_settings_lock()
    try:
        _guard_context_mode_lowering(settings)
        _guard_safety_mode_lowering(settings)
        # Baseline order: in-process pin, inherited env pin, on-disk fallback.
        baseline_pinned_in_process = _BOOT_RUNTIME_MODE is not None
        baseline_inherited_from_env = (
            not baseline_pinned_in_process and _resolve_baseline_from_env() is not None
        )
        if baseline_pinned_in_process:
            baseline_mode = _BOOT_RUNTIME_MODE
        elif baseline_inherited_from_env:
            baseline_mode = _resolve_baseline_from_env()
        else:
            baseline_mode = "advanced"
            if SETTINGS_PATH.exists():
                try:
                    disk_settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
                    baseline_mode = normalize_runtime_mode(disk_settings.get("OUROBOROS_RUNTIME_MODE"))
                except (OSError, json.JSONDecodeError):
                    pass
        new_mode = normalize_runtime_mode(settings.get("OUROBOROS_RUNTIME_MODE"))
        # Once a boot baseline is pinned, allow_elevation is inert.
        baseline_pinned = baseline_pinned_in_process or baseline_inherited_from_env
        consent_honoured = allow_elevation and not baseline_pinned
        if (_RUNTIME_MODE_RANK[new_mode] > _RUNTIME_MODE_RANK[baseline_mode]
                and not consent_honoured):
            if baseline_pinned and allow_elevation:
                hint = (
                    " The boot baseline is pinned for this run "
                    f"(source={'in-process' if baseline_pinned_in_process else 'env-var'}); "
                    "``allow_elevation=True`` is inert post-init. To "
                    "change the mode, stop the agent and edit "
                    "settings.json directly, then restart."
                )
            else:
                hint = (
                    " Runtime mode is owner-controlled — change it by "
                    "editing settings.json directly while the agent is "
                    "stopped, then restart."
                )
            raise PermissionError(
                f"OUROBOROS_RUNTIME_MODE elevation refused: "
                f"{baseline_mode!r} -> {new_mode!r}.{hint}"
            )
        try:
            tmp = SETTINGS_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(settings, indent=2), encoding="utf-8")
            os.replace(str(tmp), str(SETTINGS_PATH))
        except OSError:
            SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    finally:
        _release_settings_lock(fd)


def get_mcp_servers() -> list:
    return list(_coerce_setting_value("MCP_SERVERS", load_settings().get("MCP_SERVERS")))


def get_mcp_tool_timeout_sec() -> int:
    raw = os.environ.get("MCP_TOOL_TIMEOUT_SEC")
    if raw:
        try:
            parsed = int(raw)
            if parsed > 0:
                return parsed
        except (TypeError, ValueError):
            pass
    try:
        parsed = int(load_settings().get("MCP_TOOL_TIMEOUT_SEC") or 0)
    except (TypeError, ValueError):
        parsed = 0
    return parsed if parsed > 0 else int(SETTINGS_DEFAULTS["MCP_TOOL_TIMEOUT_SEC"])


def get_vision_caption_timeout_sec() -> int:
    raw = os.environ.get("OUROBOROS_VISION_CAPTION_TIMEOUT_SEC", SETTINGS_DEFAULTS["OUROBOROS_VISION_CAPTION_TIMEOUT_SEC"])
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return int(SETTINGS_DEFAULTS["OUROBOROS_VISION_CAPTION_TIMEOUT_SEC"])


def get_finalization_grace_sec(settings: Optional[dict] = None) -> int:
    raw = os.environ.get("OUROBOROS_FINALIZATION_GRACE_SEC")
    if raw is None and isinstance(settings, dict):
        raw = settings.get("OUROBOROS_FINALIZATION_GRACE_SEC")
    if raw is None:
        try:
            raw = load_settings().get("OUROBOROS_FINALIZATION_GRACE_SEC")
        except Exception:
            raw = None
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        parsed = int(FINALIZATION_GRACE_DEFAULT_SEC)
    return max(0, min(parsed, 300))


def get_scope_review_floor(settings: Optional[dict] = None) -> str:
    """P3 scope-reviewer capability floor: 'blocking_1m' (default) or 'advisory'.

    blocking_1m: the scope reviewer is treated as the >=1M blocking gate (the
    default gpt-5.5 reviewer IS 1M). advisory: a sub-1M reviewer may run, but its
    scope output is SUPPLEMENTARY ONLY and can NEVER satisfy a required blocking
    constitutional/release scope gate. Binary by design — no chunked tier."""
    raw = os.environ.get("OUROBOROS_SCOPE_REVIEW_FLOOR")
    if raw is None and isinstance(settings, dict):
        raw = settings.get("OUROBOROS_SCOPE_REVIEW_FLOOR")
    if raw is None:
        try:
            raw = load_settings().get("OUROBOROS_SCOPE_REVIEW_FLOOR")
        except Exception:
            raw = None
    value = str(raw or "blocking_1m").strip().lower()
    return "advisory" if value == "advisory" else "blocking_1m"


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


def apply_settings_to_env(settings: dict) -> None:
    """Push settings into environment variables for supervisor modules."""
    env_keys = [
        "OPENROUTER_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL",
        "OPENAI_COMPATIBLE_API_KEY", "OPENAI_COMPATIBLE_BASE_URL",
        "CLOUDRU_FOUNDATION_MODELS_API_KEY", "CLOUDRU_FOUNDATION_MODELS_BASE_URL",
        "GIGACHAT_CREDENTIALS", "GIGACHAT_USER", "GIGACHAT_PASSWORD",
        "GIGACHAT_SCOPE", "GIGACHAT_BASE_URL", "GIGACHAT_VERIFY_SSL_CERTS",
        "GIGACHAT_PROFANITY_CHECK",
        "ANTHROPIC_API_KEY",
        "OUROBOROS_NETWORK_PASSWORD",
        "OUROBOROS_MODEL", "OUROBOROS_MODEL_HEAVY", "OUROBOROS_MODEL_LIGHT", "OUROBOROS_MODEL_VISION",
        "OUROBOROS_MODEL_CONSCIOUSNESS",
        "OUROBOROS_MODEL_FALLBACKS", "OUROBOROS_MODEL_DEEP_SELF_REVIEW", "CLAUDE_CODE_MODEL",
        "OUROBOROS_FALLBACK_COOLDOWN_ENABLED", "OUROBOROS_FALLBACK_COOLDOWN_SEC",
        "OUROBOROS_FALLBACK_ATTEMPTS_PER_MODEL", "OUROBOROS_MODEL_MAX_CONCURRENCY",
        "OUROBOROS_MODEL_SLOT_MAX_WAIT_SEC",
        "OUROBOROS_PROJECT_NAMING_TIMEOUT_SEC", "OUROBOROS_PROJECT_NAMING_ASYNC_TIMEOUT_SEC",
        "OUROBOROS_SUBAGENT_CAPABILITY_DEPTH_LIMIT",
        "OUROBOROS_MAX_WORKERS", "OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT",
        "OUROBOROS_MAX_SUBAGENT_DEPTH", "OUROBOROS_PLAN_TASK_SWARM_TIMEOUT_SEC",
        "OUROBOROS_PLAN_TASK_SWARM_MAX_WAIT_SEC",
        "OUROBOROS_PLAN_TASK_SWARM_HEARTBEAT_STALE_SEC",
        "TOTAL_BUDGET", "OUROBOROS_PER_TASK_COST_USD", "GITHUB_TOKEN", "GITHUB_REPO",
        "OUROBOROS_RUB_USD_RATE", "OUROBOROS_PRICING_TTL_SEC",
        "OUROBOROS_TOOL_TIMEOUT_SEC", "OUROBOROS_PER_CALL_TIMEOUT_CEILING_SEC", "OUROBOROS_FINALIZATION_GRACE_SEC",
        "OUROBOROS_VISION_CAPTION_TIMEOUT_SEC",
        "OUROBOROS_TASK_IDLE_TIMEOUT_SEC", "OUROBOROS_TASK_ABS_CEILING_SEC",
        "OUROBOROS_PACING_INTERVAL_SEC", "OUROBOROS_SUPERVISOR_LIVENESS_DEADLINE_SEC",
        "OUROBOROS_MAX_ROUNDS", "OUROBOROS_TRANSIENT_RETRY_MAX",
        "OUROBOROS_IMAGE_INPUT_MODE",
        "OUROBOROS_BG_MAX_ROUNDS", "OUROBOROS_BG_WAKEUP_MIN", "OUROBOROS_BG_WAKEUP_MAX",
        "OUROBOROS_WEBSEARCH_MODEL", "OUROBOROS_WEBSEARCH_BACKEND",
        "OUROBOROS_MAIN_WEB_SEARCH", "OUROBOROS_MAIN_WEB_SEARCH_ENGINE",
        "OUROBOROS_MAIN_WEB_SEARCH_MAX_TOTAL_RESULTS",
        "OUROBOROS_OR_PROVIDER",
        "OUROBOROS_SEARCH_CODE_WALL_SEC",
        "OUROBOROS_GENERATIVE_PROBE", "OUROBOROS_GENERATIVE_PROBE_CHARS",
        "OUROBOROS_POST_TASK_EVOLUTION", "OUROBOROS_POST_TASK_EVOLUTION_CADENCE",
        "OUROBOROS_POST_TASK_EVOLUTION_BUDGET_USD", "OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE",
        "OUROBOROS_REVIEW_MODELS", "OUROBOROS_REVIEW_ENFORCEMENT",
        "OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS",
        "OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS",
        "OUROBOROS_RESTART_DRAIN_MAX_SEC",
        "OUROBOROS_SCOPE_REVIEW_MODELS", "OUROBOROS_SCOPE_REVIEW_MODEL",
        "OUROBOROS_SCOPE_REVIEW_DEGRADED", "OUROBOROS_SCOPE_REVIEW_FLOOR",
        "OUROBOROS_TASK_REVIEW_MODE",
        "OUROBOROS_SAFETY_MODE", "OUROBOROS_SAFETY_MAX_TOKENS", "OUROBOROS_SAFETY_CALL_TIMEOUT_SEC",
        "OUROBOROS_WEBSEARCH_TIMEOUT_SEC", "OUROBOROS_LLM_TRANSPORT_READ_TIMEOUT_SEC",
        "OUROBOROS_PLAN_TASK_DEADLINE_MIN_SEC",
        "OUROBOROS_ACCEPTANCE_REVIEW_EST_SEC", "OUROBOROS_ACCEPTANCE_MAX_IMPROVEMENT_PASSES",
        "OUROBOROS_ACCEPTANCE_RESERVE_PCT",
        # Unified disposable-artifact GC retention (replaces per-subsystem keys).
        "OUROBOROS_GC_RETENTION_DAYS",
        # Runtime-mode, context-mode, and skills-repo plumbing.
        "OUROBOROS_RUNTIME_MODE", "OUROBOROS_CONTEXT_MODE", "OUROBOROS_SKILLS_REPO_PATH",
        "OUROBOROS_HOST_SERVICE_PORT",
        # Acting (mutative) subagents: owner toggle + worktree/projects roots.
        "OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "OUROBOROS_SUBAGENT_WORKTREE_ROOT",
        "OUROBOROS_SUBAGENT_PROJECTS_ROOT",
        "OUROBOROS_DELIVERABLES_ROOT",
        # ClawHub marketplace registry URL.
        "OUROBOROS_CLAWHUB_REGISTRY_URL",
        "MCP_ENABLED", "MCP_TOOL_TIMEOUT_SEC",
        "OUROBOROS_EFFORT_TASK", "OUROBOROS_EFFORT_EVOLUTION",
        "OUROBOROS_EFFORT_REVIEW", "OUROBOROS_EFFORT_SCOPE_REVIEW",
        "OUROBOROS_EFFORT_DEEP_SELF_REVIEW",
        "OUROBOROS_EFFORT_CONSCIOUSNESS",
        "OUROBOROS_RETURN_REASONING",
        "OUROBOROS_REASONING_SUMMARY",
        "LOCAL_MODEL_SOURCE", "LOCAL_MODEL_FILENAME",
        "LOCAL_MODEL_PORT", "LOCAL_MODEL_N_GPU_LAYERS", "LOCAL_MODEL_CONTEXT_LENGTH",
        "LOCAL_MODEL_CHAT_FORMAT",
        "USE_LOCAL_MAIN", "USE_LOCAL_HEAVY", "USE_LOCAL_LIGHT", "USE_LOCAL_CONSCIOUSNESS", "USE_LOCAL_FALLBACK",
        "OUROBOROS_FILE_BROWSER_DEFAULT",
    ]
    for k in env_keys:
        val = settings.get(k)
        if k == "OUROBOROS_RETURN_REASONING" and val == "":
            os.environ[k] = ""
            continue
        if val is None or val == "":
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(val)
    if not os.environ.get("OUROBOROS_REVIEW_MODELS"):
        os.environ["OUROBOROS_REVIEW_MODELS"] = str(SETTINGS_DEFAULTS["OUROBOROS_REVIEW_MODELS"])
    if not os.environ.get("OUROBOROS_REVIEW_ENFORCEMENT"):
        os.environ["OUROBOROS_REVIEW_ENFORCEMENT"] = str(SETTINGS_DEFAULTS["OUROBOROS_REVIEW_ENFORCEMENT"])
    if not os.environ.get("OUROBOROS_SCOPE_REVIEW_MODELS") and not os.environ.get("OUROBOROS_SCOPE_REVIEW_MODEL"):
        os.environ["OUROBOROS_SCOPE_REVIEW_MODELS"] = str(SETTINGS_DEFAULTS["OUROBOROS_SCOPE_REVIEW_MODELS"])
    if not os.environ.get("OUROBOROS_TASK_REVIEW_MODE"):
        os.environ["OUROBOROS_TASK_REVIEW_MODE"] = str(SETTINGS_DEFAULTS["OUROBOROS_TASK_REVIEW_MODE"])


# PID lock: platform_layer uses OS-released locks on Unix and Windows.

def acquire_pid_lock() -> bool:
    APP_ROOT.mkdir(parents=True, exist_ok=True)
    return _compat_pid_lock_acquire(str(PID_FILE))


def release_pid_lock() -> None:
    _compat_pid_lock_release(str(PID_FILE))
