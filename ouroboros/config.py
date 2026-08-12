"""
Ouroboros — Shared configuration (single source of truth).

Paths, settings defaults, load/save with file locking and cycle-free setting metadata.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import sys
import time
from typing import Any, Optional, Sequence

from ouroboros.platform_layer import pid_lock_acquire as _compat_pid_lock_acquire
from ouroboros.platform_layer import pid_lock_release as _compat_pid_lock_release
from ouroboros.provider_models import compute_direct_review_models_fallback, local_only_review_route_env, migrate_model_value, review_model_uses_local as review_model_uses_local
from ouroboros.update_channels import UPDATE_SETTINGS_DEFAULTS, normalize_update_channel


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
# Supervisor-loop liveness deadline (WS3, v6.34.0): a watchdog thread flags the main
# supervisor loop STALLED if it has not ticked within this many seconds (healthy tick
# ~0.5s), so it only fires on a real wedge. 0 disables.
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
SETTINGS_DEFAULTS = {**UPDATE_SETTINGS_DEFAULTS,
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
    "GIGACHAT_BASE_URL": "https://api.giga.chat/v1",
    "GIGACHAT_VERIFY_SSL_CERTS": "true",
    "GIGACHAT_PROFANITY_CHECK": "",
    "ANTHROPIC_API_KEY": "",
    "MINIMAX_API_KEY": "",
    "MINIMAX_REGION": "",

    "OUROBOROS_NETWORK_PASSWORD": "",
    "OUROBOROS_SERVER_HOST": "127.0.0.1",
    "OUROBOROS_HOST_SERVICE_PORT": 8767,
    "OUROBOROS_MODEL": "x-ai/grok-4.5",
    # Worker lanes. Empty means "use OUROBOROS_MODEL" (same shape as consciousness),
    # so the owner sets ONE model by default and optionally overrides a lane. HEAVY is
    # the strong acting/coding lane (mutative first-level subagents); LIGHT is the cheap
    # bulk lane (auto / deep subagents); real cheap default since v6.82.0.
    "OUROBOROS_MODEL_HEAVY": "",
    "OUROBOROS_MODEL_LIGHT": "google/gemini-3.6-flash",
    "OUROBOROS_MODEL_VISION": "",
    "OUROBOROS_IMAGE_INPUT_MODE": "auto",
    # Background consciousness is a high-horizon cognitive loop, not a cheap
    # helper lane. Empty means "use OUROBOROS_MODEL".
    "OUROBOROS_MODEL_CONSCIOUSNESS": "",
    # Cross-model resilience CHAIN (comma-separated, ordered). A single model is a
    # 1-element chain; empty disables cross-model fallback. Resilience slot — keeps a
    # real default, unlike the worker lanes. (Renamed from the singular MODEL_FALLBACK.)
    "OUROBOROS_MODEL_FALLBACKS": "openai/gpt-5.6-luna",
    "OUROBOROS_MODEL_DEEP_SELF_REVIEW": "openai/gpt-5.6-sol-pro",
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
    # One-minor deprecated no-op: the v6.65 shared terminal-or-cutoff boundary
    # no longer stops on heartbeat staleness, but custom saved values stay loud.
    "OUROBOROS_PLAN_TASK_SWARM_HEARTBEAT_STALE_SEC": 120,
    "TOTAL_BUDGET": 10.0,
    "OUROBOROS_PER_TASK_COST_USD": 20.0,
    # cloud.ru catalog prices are RUB per 1M while the budget is USD. No implicit
    # exchange rate: the owner must explicitly configure the divisor.
    "OUROBOROS_RUB_USD_RATE": "",
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
    # Wall clock for ONE `git` invocation behind GET /api/tasks/{id}/diff. It bounds an
    # owner-facing READ, not a task: a repo whose git is wedged must fail the request
    # with a blocker rather than hold the thread.
    "OUROBOROS_TASK_DIFF_GIT_TIMEOUT_SEC": 30,
    # Wall clock for ONE `git` invocation behind the thread BRANCH OFF / MERGE BACK
    # gestures (`thread_branching._git`). Deliberately NOT the task-diff ceiling
    # above: that one bounds a bounded READ (`git diff` against one commit), while
    # this one bounds `git add -A` + `git commit` over a working tree of unknown
    # size, and a snapshot that times out is the exact failure the branch-off
    # refusals exist to contain. Same clamp, its own knob.
    "OUROBOROS_THREAD_GIT_TIMEOUT_SEC": 120,
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
    "OUROBOROS_REVIEW_MODELS": "openai/gpt-5.6-luna,google/gemini-3.6-flash,anthropic/claude-sonnet-5",
    "OUROBOROS_REVIEWER_SLOTS": "",  # structured slot SSOT (reviewer_slot_config.py); "" = legacy comma keys
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
    # Derived system state, never an owner choice (see get_owner_context_mode).
    # TRI-STATE, fail-CLOSED: "" is UNKNOWN, not "the owner chose low". Only an explicit
    # "false" (written by api_owner_context_mode alone) makes a stored `low` an owner
    # declaration, so a pre-v6.80.0 settings.json and an env allowlist forwarding the
    # mode without this key both leave the BIBLE P3 scope gate ON.
    "OUROBOROS_CONTEXT_MODE_AUTO_LOW": "",
    # Optional extra user-managed skills checkout; Ouroboros never clones/pulls it.
    "OUROBOROS_SKILLS_REPO_PATH": "",
    "OUROBOROS_CLAWHUB_REGISTRY_URL": "https://clawhub.ai/api/v1",
    "OUROBOROS_HUB_CATALOG_URL": "https://raw.githubusercontent.com/razzant/OuroborosHub/main/catalog.json",
    "MCP_ENABLED": False,
    "MCP_SERVERS": [],
    "MCP_TOOL_TIMEOUT_SEC": 60,
    # Scope review: one or more reviewer slots; enforcement follows OUROBOROS_REVIEW_ENFORCEMENT.
    "OUROBOROS_SCOPE_REVIEW_MODELS": "openai/gpt-5.6-terra",
    "OUROBOROS_SCOPE_REVIEW_MODEL": "openai/gpt-5.6-terra",
    # DEPRECATED, enforcement-inert (v6.80.0): stored, owner-only (dedicated audited
    # endpoint), but NOTHING consults it — whether the BIBLE P3 blocking scope review
    # applies follows owner-only OUROBOROS_CONTEXT_MODE. Degraded opt-in key: removed.
    "OUROBOROS_SCOPE_REVIEW_FLOOR": "blocking_1m",
    "OUROBOROS_TASK_REVIEW_MODE": "auto",
    # LLM safety-supervisor coverage (owner-only, like runtime/context mode):
    #   full (shipped default; fail-closed fallbacks land here; a FRESH wizard
    #                     authors "light") — LLM check on POLICY_CHECK + cond. shell.
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
    # v6.54.3 transport-timeout SSOT (deadline package D). web_search: 480 keeps the
    # transport failure messaged below the ToolEntry 540s outer thread-kill cap. LLM
    # no_proxy read/write floor: 2700 leaves headroom for long silent reasoning without
    # pinning a worker on a dead socket.
    "OUROBOROS_WEBSEARCH_TIMEOUT_SEC": 480,
    "OUROBOROS_LLM_TRANSPORT_READ_TIMEOUT_SEC": 2700,
    # v6.54.3 (1.5): plan_task deadline scaling. With a task deadline the planning swarm's
    # wait ceiling is min(configured ceiling, remaining/4); below this floor plan_task SKIPS
    # with a typed reason + telemetry rather than eat the tail of the budget.
    "OUROBOROS_PLAN_TASK_DEADLINE_MIN_SEC": 300,
    # Acceptance-review budget layer (task_pacing SSOT). The first final review
    # reserves at least 200s; later passes use max(this floor, 1.5×timing EWMA).
    # max passes remains the legacy default outside Required+Blocking; an explicit
    # task-local cap is always authoritative.
    "OUROBOROS_ACCEPTANCE_REVIEW_EST_SEC": 200,
    "OUROBOROS_ACCEPTANCE_MAX_IMPROVEMENT_PASSES": 1,
    "OUROBOROS_ACCEPTANCE_RESERVE_PCT": 5,
    # Prompt-cache TTL, one honest GLOBAL override (owner decision 2026-08-08, batch #2 Q2=A): applied to
    # EVERY cache_control breakpoint on the Anthropic-normalizing family — main loop, review lanes, safety
    # supervisor alike — at the ONE send-time finalizer (llm._normalize_payload_cache_ttl). 'default' = bare
    # markers (provider default 5m tier); '5m'/'1h' = the explicit Anthropic ephemeral tiers ('1h' bills cache
    # writes at the documented 2x-vs-1.25x ratio). Non-Anthropic wire formats are a NO-OP by construction
    # (Gemini documents no ttl field — the v5.30.0 outage class).
    "OUROBOROS_PROMPT_CACHE_TTL": "1h",
    # Reasoning effort per task type: none | low | medium | high
    "OUROBOROS_EFFORT_TASK": "medium",
    "OUROBOROS_EFFORT_EVOLUTION": "high",
    "OUROBOROS_EFFORT_REVIEW": "high",
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
    # 429-aware cross-model fallback: process-local cooldown for transiently failing
    # models (429/5xx/overloaded), passive heal-back. Owner-tunable; default-on, fail-soft.
    "OUROBOROS_FALLBACK_COOLDOWN_ENABLED": True,
    "OUROBOROS_FALLBACK_COOLDOWN_SEC": 120,
    "OUROBOROS_FALLBACK_ATTEMPTS_PER_MODEL": 1,
    # Delegated subagents. NARROW key, read ONLY by the subagent scheduler; deliberately
    # absent from provider_models.MODEL_SETTING_KEYS (see ARCHITECTURE "Delegated
    # subagents"). Empty = delegation off AND undecided (Settings' Subagents section
    # offers the connected-subscription default); the literal `off` = delegation off
    # because the owner said so. Wait keys bound the nanny's QUIET wait only.
    "OUROBOROS_SUBAGENT_HARNESS": "",
    "OUROBOROS_DELEGATE_WAIT_SEC": 120,
    "OUROBOROS_DELEGATE_WAIT_MAX_SEC": 1800,
}

# Claudexor control-plane contract, checked at handshake so an old daemon is a typed
# lane refusal rather than a mid-run schema surprise.
CLAUDEXOR_PROTOCOL_MAJOR: int = 3
# The TRANSPORT floor: the lowest engine that serves the READ-ONLY lane, which sends no
# `execution` block at all. 3.2.0 schema-accepts every field that lane does send (verified
# live: the body comes back with only the fake-root error, never a field error), and a
# read-only run is already scoped by Claudexor's ordinary envelope, so it needs nothing
# newer. Keeping this floor AT the oldest serving engine is the point: it is what lets an
# older daemon keep read-only delegation instead of losing the lane entirely, which is the
# owner's explicit decision. A floor set to the newest daemon anyone happened to be running
# is not conservative, it is an outage — 3.2.1 here refused the operator's own 3.2.0 engine
# and took read-only delegation down with it.
CLAUDEXOR_MIN_VERSION: str = "3.2.0"
# The MARKER floor: the oldest engine whose SCHEMA ACCEPTS `execution.delegated`, which is
# the only delegated-lane question a version can answer honestly. Measured: `RunExecution`
# is `.strict()` and has no `delegated` key below 3.3.0, so the field is a 400 (live against
# the running 3.2.0 daemon, which names `/execution/delegated` in `fieldErrors`) and the run
# never starts. Refusing here turns a certain failure into a typed one before a token is
# spent — the only work this number does. It cannot be a probe either: the marker is nested
# under `execution` and the catalog lists TOP-LEVEL keys, and the one behavioural probe is
# to send it, which on an engine that accepts it STARTS THE RUN.
#
# It is NOT the floor for a BOUNDARY existing. It used to be, pinned at 3.3.2 (macOS
# Seatbelt), and a version standing in for "a boundary was applied" lies in both directions:
# Claudexor's `docs/DELEGATED_CONFINEMENT.md` says the mechanism is macOS-only ("On every
# other platform `confinement_mechanism` is null, `confinement_verified_denied_path` is null,
# and `confinement_unavailable_reason` says why"), so a build declares the same number on a host where it
# applies nothing — and a version describes a BUILD, never what THIS attempt did. That
# question goes to the attempt record (`gateways.claudexor.attempt_containment`), and a run
# reporting no mechanism is DISCLOSED, not refused: the child already holds a shell here, so
# the step to "shell plus token" does not buy a lane-wide refusal (AGENTS.md "Disclose instead
# of forbid"). Two gates, two questions, no overlap; bands: docs/DELEGATED_ADMISSION.md.
CLAUDEXOR_DELEGATED_MARKER_MIN_VERSION: str = "3.3.0"


def _main_model() -> str:
    return (
        str(os.environ.get("OUROBOROS_MODEL", "") or "").strip()
        or str(SETTINGS_DEFAULTS["OUROBOROS_MODEL"])
    )


def get_light_model() -> str:
    """Light slot; empty falls back to Main (heavy/consciousness stay empty->main)."""
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
    injection: an EXPLICITLY empty Fallbacks slot means "no cross-model fallback". The
    shipped default reaches a default install through apply_settings_to_env."""
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
    """In-place settings migration, applied BEFORE defaults are merged.

    Preserves a stored value (never orphans an owner customization), then drops the legacy
    key. Shared SSOT for every settings entry point (load_settings AND the Colab builder).
    Order matters: the singular scope-review pin is promoted HERE, before ``SETTINGS_DEFAULTS``
    supplies the plural that WINS in get_scope_review_models."""
    for _old, _new in _LEGACY_SLOT_RENAMES:
        if _new not in settings and _old in settings:
            settings[_new] = settings[_old]
        settings.pop(_old, None)
    _pin = str(settings.get("OUROBOROS_SCOPE_REVIEW_MODEL") or "").strip()
    if _pin and not str(settings.get("OUROBOROS_SCOPE_REVIEW_MODELS") or "").strip():
        settings["OUROBOROS_SCOPE_REVIEW_MODELS"] = _pin
    return settings


def get_consciousness_model() -> str:
    """Return the high-horizon background-consciousness model slot."""
    return str(os.environ.get("OUROBOROS_MODEL_CONSCIOUSNESS", "") or "").strip() or _main_model()

# v6.57.0 — EFFORT_SCALE: ORDERED reasoning-effort SSOT (low→high), the single place a tier
# is defined (settings, llm.py builder, switch_model enum, subagent lanes). xhigh/max extend
# none..high; llm.py clamps a request DOWN to each model's learned ceiling (BIBLE P1: disclosed).
EFFORT_SCALE: tuple[str, ...] = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


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
    """Pin the immutable runtime-mode baseline before any agent code runs: call it after
    ``load_settings``/``apply_settings_to_env``, before worker or supervisor startup."""
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
    has_minimax = bool(str(os.environ.get("MINIMAX_API_KEY", "") or "").strip())
    has_legacy_base = bool(str(os.environ.get("OPENAI_BASE_URL", "") or "").strip())
    has_compatible = bool(str(os.environ.get("OPENAI_COMPATIBLE_BASE_URL", "") or "").strip())
    has_cloudru = bool(str(os.environ.get("CLOUDRU_FOUNDATION_MODELS_API_KEY", "") or "").strip())
    has_gigachat = bool(str(os.environ.get("GIGACHAT_CREDENTIALS", "") or "").strip()) or (
        bool(str(os.environ.get("GIGACHAT_USER", "") or "").strip())
        and bool(str(os.environ.get("GIGACHAT_PASSWORD", "") or "").strip())
    )
    # OpenRouter / legacy OpenAI base / OpenAI-compatible all route through the
    # OpenRouter-style stack, so their presence means "not an exclusive direct
    # provider". Among the registered direct providers, return one only when
    # exactly one is configured.
    if has_openrouter or has_legacy_base or has_compatible:
        return ""
    direct = [name for name, present in (
        ("openai", has_openai), ("anthropic", has_anthropic), ("minimax", has_minimax),
        ("cloudru", has_cloudru), ("gigachat", has_gigachat),
    ) if present]
    return direct[0] if len(direct) == 1 else ""


def resolve_effort(task_type: str) -> str:
    """Return the configured reasoning effort for the given task type."""
    t = (task_type or "").lower().strip()

    if t == "evolution":
        key = "OUROBOROS_EFFORT_EVOLUTION"
        default = "high"
    elif t == "review":
        key = "OUROBOROS_EFFORT_REVIEW"
        default = "high"
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
    return raw if raw in EFFORT_SCALE else default


# Prompt-cache TTL scale (owner decision 2026-08-08): 'default' = bare markers (provider default tier),
# '5m'/'1h' = the two documented Anthropic ephemeral tiers. Deliberately NO 'auto' (a dead value until an
# adaptive design exists) and NO '24h' (Anthropic would clamp it — a value that mostly lies).
PROMPT_CACHE_TTL_SCALE: tuple[str, ...] = ("default", "5m", "1h")


def resolve_prompt_cache_ttl() -> str:
    """The owner-configured global prompt-cache TTL ('default' | '5m' | '1h').

    Validated like ``resolve_effort``: an unknown value falls back to the shipped default.
    Consumed ONLY by the finalizer (``llm.LLMClient._normalize_payload_cache_ttl``), by
    ``review_helpers.cached_prompt_blocks`` (its marker gets stamped to the same value anyway),
    and by ``usage_accounting._reservation_cost`` as the payload-free admission fallback
    (payload-carrying sites use the finalizer's applied TTL) — never by per-builder marking
    sites (docs/DEVELOPMENT.md cache-friendliness invariant)."""
    default = str(SETTINGS_DEFAULTS["OUROBOROS_PROMPT_CACHE_TTL"])
    raw = str(os.environ.get("OUROBOROS_PROMPT_CACHE_TTL", default) or "").strip().lower()
    return raw if raw in PROMPT_CACHE_TTL_SCALE else default


def direct_provider_review_models_fallback(provider: str) -> list[str]:
    """Return the exact review-models list a direct-provider fallback emits."""
    if provider not in ("openai", "anthropic", "minimax", "cloudru", "gigachat"):
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
    triad/scope/plan/skill/acceptance review. One configured reviewer needs 1 (a loud
    single_reviewer_no_diversity degraded mode), 2 need both, 3+ keep the classic 2-of-N
    majority. DISTINCT from "configured >= quorum but fewer responded", which stays a loud
    infra quorum FAILURE at the call site."""
    return 2 if n_slots >= 3 else max(1, n_slots)


def get_review_models() -> list[str]:
    """Return the configured pre-commit review model list."""
    default_str = SETTINGS_DEFAULTS["OUROBOROS_REVIEW_MODELS"]
    models_str = os.environ.get("OUROBOROS_REVIEW_MODELS", default_str) or default_str
    models = _parse_model_list(models_str)
    models = [_main_model()] * max(1, len(models)) if local_only_review_route_env() else models
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
        # Auto-expand to the [main]*N stochastic fallback ONLY when nothing usable is
        # configured (empty, or foreign models in an exclusive direct-provider setup). An
        # explicit provider-matching list is honored exactly, duplicates included.
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
    models = [_main_model()] * max(1, len(models)) if local_only_review_route_env() else models
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
    return (str(os.environ.get("OUROBOROS_MODEL_DEEP_SELF_REVIEW", "") or "").strip()
            or str(SETTINGS_DEFAULTS["OUROBOROS_MODEL_DEEP_SELF_REVIEW"]))


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


def get_plan_task_swarm_timeout_sec() -> float:
    return _clamped_number_setting("OUROBOROS_PLAN_TASK_SWARM_TIMEOUT_SEC", low=0.0)


def get_plan_task_swarm_max_wait_sec() -> float:
    return _clamped_number_setting("OUROBOROS_PLAN_TASK_SWARM_MAX_WAIT_SEC", low=0.0)


def get_restart_drain_max_sec() -> int:
    return _clamped_number_setting(
        "OUROBOROS_RESTART_DRAIN_MAX_SEC", low=0, cast=lambda v: int(float(v)))


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
    return _clamped_number_setting("OUROBOROS_POST_TASK_EVOLUTION_BUDGET_USD", low=0.0)


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


def get_allow_mutative_subagents(write_surface: str = "") -> bool:
    """Whether the parent may spawn mutative (acting) subagents.

    Owner-controlled. An explicit truthy/falsey value applies to EVERY surface.
    Empty/unset follows the runtime mode: advanced/pro allow every acting
    surface; light is SURFACE-AWARE (Q4 sandbox unwind, owner 2026-08-08) —
    ``external_workspace``/``genesis`` children build OUTSIDE the Ouroboros
    runtime and stay allowed (light is a self-modification boundary, not an OS
    sandbox), while ``self_worktree`` (a checkout of the live body) stays off.
    A bare call (no surface) answers "may ANY acting child be scheduled".
    Gates only SCHEDULING: light-mode self-repo writes stay blocked by the
    runtime sandbox regardless."""
    key = "OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS"
    raw = os.environ.get(key, SETTINGS_DEFAULTS.get(key, ""))
    text = str(raw or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    if get_runtime_mode() in {"advanced", "pro"}:
        return True
    surface = str(write_surface or "").strip().lower()
    # Unset + light (or unknown mode): allowed for the external build surfaces,
    # off for self_worktree; an unknown surface string fails closed (the surface
    # validity gate elsewhere rejects it with its own message). A bare query
    # reports True because SOME acting children are allowed.
    return not surface or surface in {"external_workspace", "genesis"}


def get_subagent_worktree_root() -> str:
    """Filesystem root for acting self_worktree checkouts (outside repo/ and data/)."""
    raw = str(
        os.environ.get("OUROBOROS_SUBAGENT_WORKTREE_ROOT", "")
        or SETTINGS_DEFAULTS.get("OUROBOROS_SUBAGENT_WORKTREE_ROOT", "")
    ).strip()
    return raw or os.path.expanduser(os.path.join("~", "Ouroboros", "subagent_worktrees"))


# The delegate_wait ToolEntry's own per-call timeout: a configured ceiling above it
# buys a KILLED tool call, not a longer wait. Pinned to that ToolEntry by test.
DELEGATE_WAIT_CEILING_SEC = 2100


def get_delegate_wait_max_sec() -> int:
    """Ceiling for a caller-supplied ``delegate_wait`` window, in seconds."""
    return _clamped_number_setting(
        "OUROBOROS_DELEGATE_WAIT_MAX_SEC", low=1, high=DELEGATE_WAIT_CEILING_SEC, cast=int)


def get_delegate_wait_sec() -> int:
    """Default WINDOW one ``delegate_wait`` call holds, in seconds.

    Not a quiet cutoff: the wait holds this long and returns the advances it saw,
    so this also bounds how long the nanny stays out of its own mailbox."""
    return _clamped_number_setting(
        "OUROBOROS_DELEGATE_WAIT_SEC", low=1, high=get_delegate_wait_max_sec(), cast=int)


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


def _settings_flag_enabled(key: str) -> bool:
    """Disk-then-env-then-default boolean: an explicitly STORED value wins, so a UI toggle
    applies without a restart, while env still seeds a key the file never mentions."""
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
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


def get_auto_grant_enabled() -> bool:
    """Return whether reviewed skills should receive requested grants."""
    return _settings_flag_enabled("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS")


def get_trust_native_seeded_skills() -> bool:
    """Whether launcher-seeded native skills get the hash-pinned trust verdict."""
    return _settings_flag_enabled("OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS")


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

    Owner-only at the write surface (dropped from the agent-reachable /api/settings POST),
    so the agent cannot lower its own safety coverage. Deterministic registry sandbox,
    protected paths and light-mode guards run regardless (BIBLE P3: the LLM supervisor is a
    layer, not the floor)."""
    default_val = str(SETTINGS_DEFAULTS["OUROBOROS_SAFETY_MODE"])
    return normalize_safety_mode(os.environ.get("OUROBOROS_SAFETY_MODE", default_val) or default_val)


def _clamped_number_setting(key: str, *, low, high=float("inf"), cast=float):
    """Env-or-default numeric setting clamped to [low, high]; a typo falls back to the
    shipped default. SSOT for the clamped scalar getters below — the seven of them were
    byte-identical except for key, caster and bounds (P7 DRY)."""
    try:
        value = cast(os.environ.get(key, "") or SETTINGS_DEFAULTS[key])
    except (TypeError, ValueError):
        value = cast(SETTINGS_DEFAULTS[key])
    return max(low, min(value, high))


def get_safety_max_tokens() -> int:
    """Output-token budget for safety-supervisor LLM calls (parse-bug fix)."""
    return _clamped_number_setting("OUROBOROS_SAFETY_MAX_TOKENS", low=256, high=16384, cast=int)


def get_safety_call_timeout_sec() -> float:
    """Transport timeout for safety-supervisor LLM calls (prevents indefinite hang)."""
    return _clamped_number_setting("OUROBOROS_SAFETY_CALL_TIMEOUT_SEC", low=5.0, high=600.0)


def get_task_diff_git_timeout_sec() -> float:
    """Per-invocation `git` timeout for the task-diff endpoint (see SETTINGS_DEFAULTS).

    Clamped, like its siblings, so a hand-edited setting cannot turn an owner-facing
    read into an unbounded wait (or into a timeout too short for a large repo)."""
    return _clamped_number_setting("OUROBOROS_TASK_DIFF_GIT_TIMEOUT_SEC", low=5.0, high=300.0)


def get_thread_git_timeout_sec() -> float:
    """Per-invocation `git` timeout for thread BRANCH OFF / MERGE BACK (see SETTINGS_DEFAULTS).

    Clamped 5-300 exactly like the task-diff sibling, but a SEPARATE knob with a
    higher default: branch-off's snapshot runs `git add -A` and `git commit` over
    the owner's whole working tree, where the diff endpoint runs one bounded read.
    Borrowing the diff's 30s ceiling silently narrowed this path and made a large
    repository's snapshot time out where it used to succeed."""
    return _clamped_number_setting("OUROBOROS_THREAD_GIT_TIMEOUT_SEC", low=5.0, high=300.0)


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


def get_acceptance_max_improvement_passes() -> int:
    """Default COUNT cap for acceptance-review improvement passes (v6.54.4)."""
    return _clamped_number_setting("OUROBOROS_ACCEPTANCE_MAX_IMPROVEMENT_PASSES", low=0, high=20, cast=int)


def get_acceptance_reserve_pct() -> int:
    """Default finalization-reserve percentage of the total budget (v6.54.4)."""
    return _clamped_number_setting("OUROBOROS_ACCEPTANCE_RESERVE_PCT", low=0, high=50, cast=int)


def get_plan_task_deadline_min_sec() -> float:
    """Minimum useful deadline-scaled planning-swarm window (v6.54.3, 1.5)."""
    return _clamped_number_setting("OUROBOROS_PLAN_TASK_DEADLINE_MIN_SEC", low=30.0, high=3600.0)


def normalize_context_mode(value: Any) -> str:
    """Clamp caller-supplied context mode to the closed enum (low / max)."""
    default_val = str(SETTINGS_DEFAULTS["OUROBOROS_CONTEXT_MODE"])
    text = str(value or "").strip().lower()
    return text if text in VALID_CONTEXT_MODES else default_val


def get_context_mode() -> str:
    """The EFFECTIVE working-context mode (low | max) used by context sizing.

    Owner selection, possibly narrowed by the /api/settings model auto-downgrade; the
    P3 scope gate reads get_owner_context_mode instead. No boot-pin: hot-applies on the
    next task. The key is dropped from the agent-reachable /api/settings POST (P1)."""
    default_val = str(SETTINGS_DEFAULTS["OUROBOROS_CONTEXT_MODE"])
    return normalize_context_mode(
        os.environ.get("OUROBOROS_CONTEXT_MODE", default_val) or default_val
    )


def get_owner_context_mode() -> str:
    """The OWNER-SELECTED context mode, ignoring any system auto-downgrade.

    The P3 scope gate reads THIS, not the effective mode: the agent-reachable
    /api/settings model auto-downgrade narrows the effective mode without the owner
    choosing it, and honouring that would let the agent switch an immune gate off.

    The derived flag is TRI-STATE and resolved FAIL-CLOSED: absent or unrecognised is
    UNKNOWN, never an owner declaration, so it reports `max` and the gate stays ON. Only
    an explicit stored `false` counts, written by api_owner_context_mode alone (an owner
    write always clears the flag) — otherwise a pre-v6.80.0 settings.json or an isolated
    server forwarding the mode alone would read as "the owner switched scope review off"."""
    if get_context_mode() != "low":
        return "max"
    return "low" if _owner_declared_low(os.environ.get("OUROBOROS_CONTEXT_MODE_AUTO_LOW", "")) else "max"


def _settings_file_value(key: str, default: str) -> str:
    """Read ONE persisted setting off disk, without normalizing the whole file. DISK ONLY, for EVERY caller: env
    is inherited and freely rewritten by any subprocess, so it can never be a ratchet's PREVIOUS value — reading it
    there turns ``max -> low`` into ``low -> low`` and the gate opens. Absent/corrupt = the fail-closed default."""
    if SETTINGS_PATH.exists():
        try:
            disk_settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(disk_settings, dict):
                return str(disk_settings.get(key, default) or default)
        except (OSError, json.JSONDecodeError):
            pass
    return default


# The same keys from the other side: load_settings overlays env onto disk-ABSENT keys, so without this an
# ordinary load->save round-trip in a process whose env says low/off would launder that value onto disk
# unauthorised — or, once the guard reads disk, raise a PermissionError nobody authored. Owner endpoints
# write BOTH disk and env, so the owner path is unaffected.
_DISK_AUTHORED_SETTINGS = ("OUROBOROS_CONTEXT_MODE", "OUROBOROS_CONTEXT_MODE_AUTO_LOW", "OUROBOROS_SAFETY_MODE")


def _owner_declared_low(value: Any) -> bool:
    """True when the derived auto-downgrade flag reads as an OWNER-authored ``low``
    (get_owner_context_mode's own tri-state: absent/unrecognised is UNKNOWN, not owner)."""
    return str(value or "").strip().lower() in ("0", "false", "no", "off")


def _guard_context_mode_lowering(settings: dict, *, allow_context_lowering: bool = False) -> None:
    """Refuse agent-reachable settings writes that lower the cognitive horizon.

    TWO ratchets on the SAME owner authorisation (``allow_context_lowering``, held only by
    the dedicated owner endpoint): the mode may not step ``max -> low``, AND the derived
    auto-downgrade flag may not be CLEARED. Since v6.80.0 that flag is an AUTHORITY BIT —
    clearing it makes get_owner_context_mode report ``low``, switching the BIBLE P3 scope
    gate off: the same horizon-lowering spelled through another key. So ``unknown/true ->
    false`` needs the owner path, while SETTING it true (the system auto-downgrade on the
    model route) stays free. It belongs HERE because save_settings is the seam every writer
    funnels through, and a second lexical shell check would not do — a key can always be
    constructed dynamically (BIBLE P5)."""
    previous_mode = normalize_context_mode(_settings_file_value("OUROBOROS_CONTEXT_MODE", "max"))
    next_mode = normalize_context_mode(settings.get("OUROBOROS_CONTEXT_MODE", previous_mode))
    if previous_mode == "max" and next_mode == "low" and not allow_context_lowering:
        raise PermissionError(
            "OUROBOROS_CONTEXT_MODE lowering refused: 'max' -> 'low'. "
            "Context mode is owner-controlled — use the dedicated owner endpoint/UI/CLI."
        )
    if allow_context_lowering or "OUROBOROS_CONTEXT_MODE_AUTO_LOW" not in settings:
        return
    previous_flag = _settings_file_value("OUROBOROS_CONTEXT_MODE_AUTO_LOW", "")
    if _owner_declared_low(settings["OUROBOROS_CONTEXT_MODE_AUTO_LOW"]) and not _owner_declared_low(previous_flag):
        raise PermissionError(
            f"OUROBOROS_CONTEXT_MODE_AUTO_LOW clearing refused: {previous_flag or 'unknown'!r} -> 'false'. "
            "Clearing the derived flag turns a system auto-downgrade into an owner-declared scope-review "
            "skip — it is owner-controlled, use the dedicated owner endpoint/UI/CLI."
        )


def prepare_settings_for_persist(settings: dict, *, authored_keys: Sequence[str] = (),
        allow_context_lowering: bool = False, allow_safety_lowering: bool = False) -> dict:
    """THE prologue EVERY writer that persists settings.json must call; returns the dict to write.

    ONE enforcement point: three review rounds found this rule on one path while a sibling bypassed it. Ratchets
    are enforced here, and SILENCE STAYS SILENCE — a disk-authored key the file does not carry, arriving as nothing
    but the shipped default, is a gap filled by a defaults merge (load_settings / _owner_read_settings_raw), not
    authorship: persisting it ends a forwarded env override mid-run and labels a benchmark artifact with a mode it
    never ran under (mirror: apply_settings_to_env). AUTHORSHIP IS INFORMATION ONLY THE CALLER HAS — one that
    really authors such a key names it in ``authored_keys``; a POST never about these keys authors nothing."""
    authored = set(authored_keys or ())
    prepared = {k: v for k, v in settings.items() if not (
        k in _DISK_AUTHORED_SETTINGS and k not in authored and not _settings_file_value(k, "")
        and str(v) == str(SETTINGS_DEFAULTS.get(k, "")))}
    _guard_context_mode_lowering(prepared, allow_context_lowering=allow_context_lowering)
    _guard_safety_mode_lowering(prepared, allow_safety_lowering=allow_safety_lowering)
    return prepared


_SAFETY_MODE_RANK = {"full": 2, "light": 1, "off": 0}


def _guard_safety_mode_lowering(settings: dict, *, allow_safety_lowering: bool = False) -> None:
    """Refuse agent-reachable settings writes that lower LLM-safety coverage.

    ``full -> light -> off`` is a strictly decreasing coverage ladder; any downward step is
    owner-only (mirrors the context-mode ratchet, BIBLE P3)."""
    previous_mode = normalize_safety_mode(_settings_file_value("OUROBOROS_SAFETY_MODE", "full"))
    next_mode = normalize_safety_mode(settings.get("OUROBOROS_SAFETY_MODE", previous_mode))
    if _SAFETY_MODE_RANK[next_mode] < _SAFETY_MODE_RANK[previous_mode] and not allow_safety_lowering:
        raise PermissionError(
            f"OUROBOROS_SAFETY_MODE lowering refused: {previous_mode!r} -> {next_mode!r}. "
            "Safety mode is owner-controlled — use the dedicated /api/owner/safety-mode endpoint."
        )


def get_skills_repo_path() -> str:
    """Return the configured external skills checkout path, expanding ``~``."""
    raw = (os.environ.get("OUROBOROS_SKILLS_REPO_PATH", "") or "").strip()
    if not raw:
        return ""
    try:
        return str(pathlib.Path(raw).expanduser())
    except Exception:
        return raw


# Skills data layout: runtime skill packages live under ``data/skills/<source>/<slug>/``.
# The git-tracked ``repo/skills/`` tree is only a launcher seed; the optional
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
    """Return ``<DATA_DIR>/skills/ouroboroshub/`` (created on demand by
    ``ensure_data_skills_dir``, which makes every source subdir)."""
    return ensure_data_skills_dir(DATA_DIR) / SKILL_SOURCE_OUROBOROSHUB


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
def _settings_lock_path() -> pathlib.Path:
    # Call-time so a repointed SETTINGS_PATH locks beside its own file.
    return pathlib.Path(str(SETTINGS_PATH) + ".lock")


def _acquire_settings_lock(timeout: float = 2.0) -> Optional[int]:
    start = time.time()
    lock_path = _settings_lock_path()
    while time.time() - start < timeout:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            return fd
        except FileExistsError:
            try:
                if time.time() - lock_path.stat().st_mtime > 10:
                    lock_path.unlink()
                    continue
            except Exception:
                pass
            time.sleep(0.01)
        except Exception:
            break
    return None


def _release_settings_lock(fd: Optional[int]) -> None:
    if fd is None:  # never acquired; a concurrent writer's lock must stay
        return
    try:
        os.close(fd)
    except Exception:
        pass
    try:
        _settings_lock_path().unlink()
    except Exception:
        pass


def _coerce_setting_value(key: str, value):
    default = SETTINGS_DEFAULTS.get(key)
    # Normalize runtime mode on read so all consumers see the closed enum.
    if key == "OUROBOROS_RUNTIME_MODE":
        return normalize_runtime_mode(value)
    if key == "OUROBOROS_UPDATE_CHANNEL":
        return normalize_update_channel(value)
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
# Setting keys a release DELETED. `load_settings` keeps unrecognized keys so an owner
# customization is never destroyed by a rename, which means a key removed from
# SETTINGS_DEFAULTS would otherwise live in an existing data/settings.json forever and
# keep being served by GET /api/settings as something the runtime no longer honors.
# Retiring a key is a decision; leaving its ghost to answer for it is not.
RETIRED_SETTING_KEYS: tuple[str, ...] = (
    # v6.87.7: the capability depth cap conflated how DEEP delegation nests with how
    # STRONG a descendant may be. Depth never rewrites the lane now.
    "OUROBOROS_SUBAGENT_CAPABILITY_DEPTH_LIMIT",
)


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
        # Rename-alias migration: fold deprecated per-subsystem retention keys into the
        # unified OUROBOROS_GC_RETENTION_DAYS, then drop the legacy keys. Prefer a CUSTOMIZED
        # legacy value so a rename never orphans it; an all-defaults file collapses to the
        # unified default (e.g. service 14->7).
        from ouroboros.retention import LEGACY_RETENTION_KEYS, pick_legacy_retention_seed
        if "OUROBOROS_GC_RETENTION_DAYS" not in loaded:
            seed = pick_legacy_retention_seed(loaded.get)
            if seed is not None:
                loaded["OUROBOROS_GC_RETENTION_DAYS"] = seed
        for _legacy in LEGACY_RETENTION_KEYS:
            loaded.pop(_legacy, None)
        for _retired in RETIRED_SETTING_KEYS:
            loaded.pop(_retired, None)
        migrate_legacy_slot_keys(loaded)
        settings = dict(SETTINGS_DEFAULTS)
        settings.update(loaded)
        for key in SETTINGS_DEFAULTS:
            raw_env = os.environ.get(key)
            if raw_env is None or key in _DISK_AUTHORED_SETTINGS:  # owner ratchets: DISK-authored, never env
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
    onboarding_safety_default: bool = False,
) -> None:
    """Persist settings and enforce owner-only mode ratchets.

    Elevation above the boot baseline is refused after initialization (``allow_elevation``
    is then inert to agent-reachable subprocesses; production entry points must call
    ``initialize_runtime_mode_baseline`` before agent code). Context-mode lowering and
    clearing the derived auto-low flag likewise require the explicit owner path.

    ``onboarding_safety_default`` is a NARROW boolean authorizing exactly one
    transition — a FRESH install (no settings file yet) authoring
    ``OUROBOROS_SAFETY_MODE="light"``; everything else keeps the ratchet."""
    _guard_live_settings_write()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fd = _acquire_settings_lock()
    if fd is None:
        raise TimeoutError(f"Could not acquire settings lock {_settings_lock_path()} within 2 seconds")
    try:
        authored_keys: tuple[str, ...] = ()
        allow_safety_lowering = False
        if onboarding_safety_default:
            wants_light = str(settings.get("OUROBOROS_SAFETY_MODE", "") or "").strip().lower() == "light"
            if wants_light and not SETTINGS_PATH.exists():
                authored_keys = ("OUROBOROS_SAFETY_MODE",)
                allow_safety_lowering = True
        settings = prepare_settings_for_persist(
            settings, authored_keys=authored_keys,
            allow_safety_lowering=allow_safety_lowering)
        # Baseline: the in-process pin, else the STRICTEST of the inherited env pin and disk. The env pin only
        # exists so a subprocess inherits the parent's ratchet, so it may only TIGHTEN — letting it RAISE the
        # baseline is the caller-controlled "previous value" hole of _settings_file_value on a fourth key (a
        # subprocess exporting BOOT_RUNTIME_MODE=pro could otherwise persist its own elevation).
        baseline_pinned_in_process = _BOOT_RUNTIME_MODE is not None
        inherited_baseline = None if baseline_pinned_in_process else _resolve_baseline_from_env()
        baseline_inherited_from_env = inherited_baseline is not None
        disk_baseline = normalize_runtime_mode(_settings_file_value("OUROBOROS_RUNTIME_MODE", "advanced"))
        baseline_mode = _BOOT_RUNTIME_MODE or min(
            [m for m in (inherited_baseline, disk_baseline) if m], key=_RUNTIME_MODE_RANK.__getitem__)
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
    return _clamped_number_setting("OUROBOROS_VISION_CAPTION_TIMEOUT_SEC", low=1, cast=int)


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


# Settings keys deliberately NOT projected into the environment. Everything else in
# SETTINGS_DEFAULTS IS exported, by derivation rather than by a parallel hand-kept list:
# a list maintained beside the defaults drifts silently, and the failure it produces is
# invisible — settings accept the key, the UI shows it saved, and the consuming module
# goes on reading os.environ and falling back to its hardcoded constant
# (OUROBOROS_SKILL_LIFECYCLE_TIMEOUT_SEC sat like that behind a hardcoded 1800). Deriving
# makes export the DEFAULT for a new key and makes an exclusion a decision someone has to
# write down here.
SETTINGS_KEYS_NOT_EXPORTED_TO_ENV = frozenset({
    # Structured list value: `str(value)` is a Python repr no reader parses back, and
    # every consumer already reads it from the settings dict (mcp_client.parse_servers,
    # gateway.mcp), never from the environment.
    "MCP_SERVERS",
    # ENV IS THE AUTHORITY for the bind host, not settings. `ouroboros server --host
    # 0.0.0.0` puts the choice in the environment, and both consumers (server.main and
    # server_control.restart_current_process) deliberately read env BEFORE settings.
    # Exporting this key stamped the settings value — usually the shipped 127.0.0.1
    # default, which no owner ever authored — back over that environment, so the operator's
    # LAN-reachable server silently became loopback at the first self-restart and stayed
    # there. A default standing in for an absent key is not an owner decision.
    "OUROBOROS_SERVER_HOST",
})


def settings_env_keys() -> list:
    """Settings keys projected into os.environ, derived from SETTINGS_DEFAULTS."""
    return [k for k in SETTINGS_DEFAULTS if k not in SETTINGS_KEYS_NOT_EXPORTED_TO_ENV]


def apply_settings_to_env(settings: dict) -> None:
    """Push settings into environment variables for supervisor modules."""
    env_keys = settings_env_keys()
    # Disk-authored ratchets PROJECT ONLY WHAT THE FILE ACTUALLY SAYS: a default standing in for an absent
    # key is not an owner decision, so overwriting/popping the env entry would clobber a legitimately
    # forwarded value (harbor_installed_agent runs with NO settings.json; server_runner documents the same
    # "settings.json over env" clobber). Silence stays silent. ONE fail-closed exception: env may not author
    # the owner-declared-low CLAIM (auto-low `false`), which would switch the BIBLE P3 scope gate off.
    unauthored = {k for k in _DISK_AUTHORED_SETTINGS if not _settings_file_value(k, "")}
    for k in env_keys:
        val = settings.get(k)
        if k in unauthored and not _owner_declared_low(
                os.environ.get(k) if k == "OUROBOROS_CONTEXT_MODE_AUTO_LOW" else ""):
            continue
        if k == "OUROBOROS_RETURN_REASONING" and val == "":
            os.environ[k] = ""
            continue
        if val is None or val == "":
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(val)
    # Reviewer-model floors moved into the structured-slot projection (6.1):
    from ouroboros.reviewer_slot_config import project_reviewer_slots_into_env
    project_reviewer_slots_into_env()
    if not os.environ.get("OUROBOROS_REVIEW_ENFORCEMENT"):
        os.environ["OUROBOROS_REVIEW_ENFORCEMENT"] = str(SETTINGS_DEFAULTS["OUROBOROS_REVIEW_ENFORCEMENT"])
    if not os.environ.get("OUROBOROS_TASK_REVIEW_MODE"):
        os.environ["OUROBOROS_TASK_REVIEW_MODE"] = str(SETTINGS_DEFAULTS["OUROBOROS_TASK_REVIEW_MODE"])


# PID lock: platform_layer uses OS-released locks on Unix and Windows.

def acquire_pid_lock() -> bool:
    APP_ROOT.mkdir(parents=True, exist_ok=True)
    return _compat_pid_lock_acquire(str(PID_FILE))


def release_pid_lock() -> None:
    _compat_pid_lock_release(str(PID_FILE))
