"""Ouroboros — the settings vocabulary.

What settings exist, the values a fresh install ships, which keys a release has
retired, and which keys never travel between disk and the environment. Data and
derivations only: nothing here reads or writes settings.json.
"""

from __future__ import annotations

from ouroboros.update_channels import UPDATE_SETTINGS_DEFAULTS

FINALIZATION_GRACE_DEFAULT_SEC = 120
# Owner finalize-then-stop OUTER safety cap (S3, owner decisions 2026-08-15),
# from the stop REQUEST; the grace budget above starts only at control DELIVERY
# (the loop's mailbox drain). No summary by this cap -> honest custody cancel.
OWNER_STOP_OUTER_CAP_SEC = 600
# Cadence for intrinsic self-pacing checkpoints when a task has NO deadline_at
# (e.g. headless benchmark runs). Advisory only — surfaces elapsed/rounds/cost so
# the model can self-pace; it is not a stop gate. 0 disables.
PACING_INTERVAL_DEFAULT_SEC = 600
# Supervisor-loop liveness deadline (WS3, v6.34.0): a watchdog thread flags the main
# supervisor loop STALLED if it has not ticked within this many seconds (healthy tick
# ~0.5s), so it only fires on a real wedge. 0 disables.
SUPERVISOR_LIVENESS_DEADLINE_DEFAULT_SEC = 90


# Shipped router profile. Keeping the root-loop role policy beside the direct
# provider profiles gives onboarding, runtime defaults, and tests one vocabulary
# instead of repeating model ids across those surfaces.
OPENROUTER_DEFAULTS = {
    "main": "google/gemini-3.7-flash",
    "heavy": "",
    "light": "openai/gpt-5.6-luna",
    "vision": "",
    "consciousness": "",
    "fallback": "openai/gpt-5.6-luna",
    "deep_self_review": "openai/gpt-5.6-sol-pro",
}

OPENROUTER_REVIEW_DEFAULTS = {
    "triad": (
        "google/gemini-3.7-flash",
        "openai/gpt-5.6-terra",
        "anthropic/claude-opus-5",
    ),
    "scope": ("openai/gpt-5.6-terra",),
    # Claude Agent SDK spelling, not an OpenRouter model id. With no direct
    # Anthropic key the existing advisory gate records an audited bypass.
    "advisory": "claude-sonnet-5",
}


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
    "OUROBOROS_MODEL": OPENROUTER_DEFAULTS["main"],
    # Worker lanes; empty means "use OUROBOROS_MODEL" (one model by default, per-lane
    # override optional). HEAVY = mutative first-level subagents; LIGHT = auto/deep bulk.
    "OUROBOROS_MODEL_HEAVY": OPENROUTER_DEFAULTS["heavy"],
    "OUROBOROS_MODEL_LIGHT": OPENROUTER_DEFAULTS["light"],
    "OUROBOROS_MODEL_VISION": OPENROUTER_DEFAULTS["vision"],
    "OUROBOROS_IMAGE_INPUT_MODE": "auto",
    # Background consciousness is a high-horizon loop, not a cheap helper lane.
    "OUROBOROS_MODEL_CONSCIOUSNESS": OPENROUTER_DEFAULTS["consciousness"],
    # Cross-model resilience CHAIN (comma-separated, ordered). A single model is a
    # 1-element chain; empty disables cross-model fallback. Resilience slot — keeps a
    # real default, unlike the worker lanes. (Renamed from the singular MODEL_FALLBACK.)
    "OUROBOROS_MODEL_FALLBACKS": OPENROUTER_DEFAULTS["fallback"],
    "OUROBOROS_MODEL_DEEP_SELF_REVIEW": OPENROUTER_DEFAULTS["deep_self_review"],
    "CLAUDE_CODE_MODEL": OPENROUTER_REVIEW_DEFAULTS["advisory"],
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
    "TOTAL_BUDGET": 200.0,
    "OUROBOROS_PER_TASK_COST_USD": 50.0,
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
    # Generative context-window probe machinery: when enabled AND a caller passes
    # allow_generative=True, confirms a route's >=1M window from a FREE over-window
    # reject; *_CHARS sizes the padding. Since the settings-time Max gate retirement
    # no production surface passes allow_generative=True (dormant; kept for tests
    # and future explicit owner probes).
    "OUROBOROS_GENERATIVE_PROBE": "1",
    "OUROBOROS_GENERATIVE_PROBE_CHARS": "5000000",
    # Pre-commit review: comma-separated provider-tagged model list
    "OUROBOROS_REVIEW_MODELS": ",".join(OPENROUTER_REVIEW_DEFAULTS["triad"]),
    "OUROBOROS_REVIEWER_SLOTS": "",  # structured slot SSOT (reviewer_slot_config.py); "" = legacy comma keys
    # INSTALL-TIME facts: the agent-preset generation this install received, and WHEN onboarding last completed
    # (recorded on EVERY completion). Endpoint-authored and disk-only — see ENDPOINT_AUTHORED_SETTINGS.
    "OUROBOROS_SUBSCRIPTION_PRESET_VERSION": "",
    "OUROBOROS_ONBOARDING_COMPLETED_AT": "",
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
    # Context mode: low | max. Owner-only working-context size profile. max = full always-on docs +
    # current memory granularity; low = ARCHITECTURE as a navigation map + deeper memory consolidation,
    # sized for ~200k / local models. Cognitive-horizon knob (BIBLE P1): the agent cannot lower it
    # (owner-only), and it never changes model / reasoning-effort / output-token budgets.
    "OUROBOROS_CONTEXT_MODE": "max",
    # One-window compatibility tombstone for the retired persistent auto-Low mechanism.
    # It never sizes or routes context and no runtime writer may set it true.  An explicit
    # false still distinguishes owner-authored Low from a bare forwarded env Low for P3.
    "OUROBOROS_CONTEXT_MODE_AUTO_LOW": "false",
    # Optional extra user-managed skills checkout; Ouroboros never clones/pulls it.
    "OUROBOROS_SKILLS_REPO_PATH": "",
    "OUROBOROS_CLAWHUB_REGISTRY_URL": "https://clawhub.ai/api/v1",
    "OUROBOROS_HUB_CATALOG_URL": "https://raw.githubusercontent.com/razzant/OuroborosHub/main/catalog.json",
    "MCP_ENABLED": False,
    "MCP_SERVERS": [],
    "MCP_TOOL_TIMEOUT_SEC": 60,
    # Scope review: one or more reviewer slots; enforcement follows OUROBOROS_REVIEW_ENFORCEMENT.
    "OUROBOROS_SCOPE_REVIEW_MODELS": ",".join(OPENROUTER_REVIEW_DEFAULTS["scope"]),
    "OUROBOROS_SCOPE_REVIEW_MODEL": OPENROUTER_REVIEW_DEFAULTS["scope"][0],
    # DEPRECATED, enforcement-inert (v6.80.0): stored, owner-only (dedicated audited endpoint), but
    # NOTHING consults it — whether the BIBLE P3 blocking scope review applies follows owner-only
    # OUROBOROS_CONTEXT_MODE. Degraded opt-in key: removed.
    "OUROBOROS_SCOPE_REVIEW_FLOOR": "blocking_1m",
    "OUROBOROS_TASK_REVIEW_MODE": "auto",
    # LLM safety-supervisor coverage (owner-only, like runtime/context mode):
    #   full  (shipped default; fail-closed fallbacks land here; a FRESH wizard authors
    #          "light") — LLM check on POLICY_CHECK + conditional shell.
    #   light — LLM check ONLY on POLICY_CHECK integration tools; POLICY_CHECK_CONDITIONAL
    #           shell/verify fall to the deterministic whitelist + registry guards (no LLM).
    #   off   — no LLM safety calls at all; the deterministic registry sandbox, protected-path
    #           policy and light-mode guards STAY ON. Every non-full mode audits durably.
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
    "OUROBOROS_ACCEPTANCE_REVIEW_EST_SEC": 200,
    # Shared paid-review-cycle cap (SSOT + per-gate meaning: ouroboros/review_cycles.py):
    # STRING "N"|"unlimited": plan review, acceptance (passes = cycles - 1), commit-gate cap.
    "OUROBOROS_REVIEW_MAX_CYCLES": "2",
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
    # Optional Delegation account pin (D-U5): a credential-profile id sent as
    # `credentialProfileId`; empty = engine rotation pool (D28; presets never
    # author it). Read ONLY by get_subagent_harness -> DelegationRoute.profile_id.
    "OUROBOROS_SUBAGENT_PROFILE": "",
    "OUROBOROS_DELEGATE_WAIT_SEC": 120,
    "OUROBOROS_DELEGATE_WAIT_MAX_SEC": 1800,
}


# Setting keys a release DELETED. `load_settings` keeps unrecognized keys so a rename never destroys
# an owner customization — which would otherwise leave a removed key living in data/settings.json
# forever, still served by GET /api/settings. Retiring a key is a decision; its ghost is not.
RETIRED_SETTING_KEYS: tuple[str, ...] = (
    # v6.87.7: the depth cap conflated how DEEP delegation nests with how STRONG a descendant is.
    "OUROBOROS_SUBAGENT_CAPABILITY_DEPTH_LIMIT",
    # The flat wall-clock stop these two named was replaced by the activity model
    # (idle window + subtree liveness + absolute ceiling), and their one-minor
    # deprecation window — during which a customized value still emitted a
    # deprecated_settings_ignored event — ended. A knob whose only remaining job is
    # to announce that it does nothing is a knob the settings surface still offers.
    "OUROBOROS_SOFT_TIMEOUT_SEC",
    "OUROBOROS_HARD_TIMEOUT_SEC",
    # Same window, same reason: the shared terminal-or-cutoff planning boundary never
    # stopped on heartbeat staleness.
    "OUROBOROS_PLAN_TASK_SWARM_HEARTBEAT_STALE_SEC",
    # knobs are retired (the review-cycle cap OUROBOROS_REVIEW_MAX_CYCLES bounds plan review).
    "OUROBOROS_ACCEPTANCE_MAX_IMPROVEMENT_PASSES",
    "OUROBOROS_PLAN_TASK_SWARM_TIMEOUT_SEC",
    "OUROBOROS_PLAN_TASK_SWARM_MAX_WAIT_SEC",
)


# The same keys from the other side: load_settings overlays env onto disk-ABSENT keys, so without this an
# ordinary load->save round-trip in a process whose env says low/off would launder that value onto disk
# unauthorised — or, once the guard reads disk, raise a PermissionError nobody authored. Owner endpoints
# write BOTH disk and env, so the owner path is unaffected.
_DISK_AUTHORED_SETTINGS = ("OUROBOROS_CONTEXT_MODE", "OUROBOROS_CONTEXT_MODE_AUTO_LOW", "OUROBOROS_SAFETY_MODE")

# ENDPOINT-AUTHORED, DISK-ONLY: install-time facts POST /api/onboarding/complete alone writes. The ratchets above
# are disk-authored yet DO project once the file carries them; these never leave disk in EITHER direction — an env
# timestamp alone closed the onboarding window on a fresh install, and an env marker was then persisted by a save.
ENDPOINT_AUTHORED_SETTINGS = frozenset({"OUROBOROS_SUBSCRIPTION_PRESET_VERSION", "OUROBOROS_ONBOARDING_COMPLETED_AT"})


# Settings keys deliberately NOT projected into the environment. Everything else in SETTINGS_DEFAULTS IS
# exported, by derivation rather than a parallel hand-kept list: such a list drifts silently and the failure
# is invisible — settings accept the key, the UI shows it saved, and the consumer goes on reading os.environ
# and falling back to its hardcoded constant (OUROBOROS_SKILL_LIFECYCLE_TIMEOUT_SEC sat like that behind a
# hardcoded 1800). Deriving makes export the DEFAULT for a new key and an exclusion a decision written here.
SETTINGS_KEYS_NOT_EXPORTED_TO_ENV = frozenset({
    # Structured list value: `str(value)` is a Python repr no reader parses back, and every consumer already reads
    # it from the settings dict (mcp_client.parse_servers, gateway.mcp), never from the environment.
    "MCP_SERVERS",
    # ENV IS THE AUTHORITY for the bind host, not settings. `ouroboros server --host 0.0.0.0` puts the choice in
    # the environment, and both consumers (server.main, server_control.restart_current_process) deliberately read
    # env BEFORE settings. Exporting this key stamped the settings value — usually the shipped 127.0.0.1 default,
    # which no owner authored — back over that environment, so the operator's LAN-reachable server silently became
    # loopback at the first self-restart. A default standing in for an absent key is not a decision.
    "OUROBOROS_SERVER_HOST",
}) | ENDPOINT_AUTHORED_SETTINGS  # disk-only in BOTH directions (never read from env, never exported to it)


def settings_env_keys() -> list:
    """Settings keys projected into os.environ, derived from SETTINGS_DEFAULTS."""
    return [k for k in SETTINGS_DEFAULTS if k not in SETTINGS_KEYS_NOT_EXPORTED_TO_ENV]
