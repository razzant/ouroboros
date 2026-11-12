"""Ouroboros — the closed scales a settings value is clamped to.

Reasoning effort, prompt-cache tier, runtime mode, safety-supervisor coverage and the
optional positive bounds (a positive integer or "unlimited") are ordered or enumerated
vocabularies. Each one is defined once here, with the
clamp that turns any caller-supplied or environment-supplied text into a member
of it, so an unknown value can never reach a consumer.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from ouroboros.settings_defaults import SETTINGS_DEFAULTS
from ouroboros.settings_integrity import runtime_setting

# v6.57.0 — EFFORT_SCALE: ORDERED reasoning-effort SSOT (low→high), the single place a tier is
# defined (settings, llm.py builder, switch_model enum, subagent lanes). `ultra` = the codex
# vendor tier above `max`; above-ceiling tiers adapt per route (API wire recovery / delegated).
EFFORT_SCALE: tuple[str, ...] = ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")

# The primary effort and independently ordered helper bounds (``effort_range``).
EFFORT_RANGE_KEYS: tuple[str, ...] = ("OUROBOROS_EFFORT_MIN", "OUROBOROS_EFFORT_TASK", "OUROBOROS_EFFORT_MAX")
EFFORT_FORMAT_KEY = "_effort_range_format"  # disk format, never an environment setting or owner knob
# The tiers the owner's range control offers; `minimal` stays a runtime tier a stored value may name.
OWNER_EFFORT_TIERS: tuple[str, ...] = tuple(tier for tier in EFFORT_SCALE if tier != "minimal")
# Who decided one actor's effort (``choose_effort``); task records and chat frames carry it verbatim.
EFFORT_SOURCES: tuple[str, ...] = ("auto", "pin", "model_name", "cyber")
# The lane-era global review effort, kept ONLY as the frozen legacy skill-review fingerprint
# literal: an empty reviewer row reviews at the range's top, never here.
REVIEW_POOL_DEFAULT_EFFORT = "high"


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


def requested_effort(value: Any) -> str:
    """A caller's explicit starting effort for a new root task, as its EFFORT_SCALE tier.

    The field is optional and its caller decides whether it was supplied; a value that
    WAS supplied must name a tier — blank, unknown or non-string raises ``ValueError``
    before the caller's first effect, never a silent default. It is a request: the
    route may adapt it, and what was applied is recorded apart from it.
    """
    tier = value.strip().lower() if isinstance(value, str) else ""
    if tier not in EFFORT_SCALE:
        raise ValueError(f"reasoning_effort must be one of {', '.join(EFFORT_SCALE)}; got {value!r}")
    return tier


def _tier(value: Any) -> str:
    """``value`` as an EFFORT_SCALE tier, '' for blank or unknown."""
    text = str(value or "").strip().lower()
    return text if text in EFFORT_SCALE else ""


def normalize_legacy_effort_settings(settings: Mapping[str, Any], *, legacy_order: bool = False) -> dict[str, Any]:
    """Materialize the old effective bounds BEFORE defaults erase their absence.

    Partial choices widened their bounds around Main. Canonical complete triples
    are independent; the document reader alone opts into the old ordering.
    """
    result = dict(settings)
    if legacy_order or (any(key in settings for key in EFFORT_RANGE_KEYS) and not all(
            _tier(settings.get(key)) for key in (EFFORT_RANGE_KEYS[0], EFFORT_RANGE_KEYS[2]))):
        low, primary, high = (_tier(settings.get(key)) or str(SETTINGS_DEFAULTS[key]) for key in EFFORT_RANGE_KEYS)
        result.update(zip(EFFORT_RANGE_KEYS, (min(low, primary, key=effort_rank), primary,
                                             max(high, primary, key=effort_rank))))
    return result


def normalize_effort_document(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Keep pre-independent effective bounds, even after old writers filled defaults.

    Call after selecting authored sources, before defaults hide absence. The returned
    format marker survives ordinary saves; canonical actor/project triples need none.
    """
    return {**normalize_legacy_effort_settings(settings, legacy_order=settings.get(EFFORT_FORMAT_KEY) != 1),
            EFFORT_FORMAT_KEY: 1}


def effort_range(settings: Optional[Mapping[str, Any]] = None) -> dict[str, str]:
    """Snapshot-aware primary level and helper bounds; only the bounds are ordered.

    Missing legacy bounds retain their former effective values. Unknown tiers
    take their defaults. Reversed helper bounds sort without moving the primary.
    """
    raw = settings if settings is not None else {key: runtime_setting(key) for key in EFFORT_RANGE_KEYS
                                                if runtime_setting(key) is not None}
    values = normalize_legacy_effort_settings(raw)
    low, primary, high = (_tier(values.get(key)) or str(SETTINGS_DEFAULTS[key]) for key in EFFORT_RANGE_KEYS)
    return {"min": min(low, high, key=effort_rank), "recommended": primary, "max": max(low, high, key=effort_rank)}


def clamp_effort_into(value: str, rng: Mapping[str, str]) -> str:
    """``value`` at the nearest bound of ``rng`` when it lies outside; unknown passes through."""
    rank = effort_rank(value)
    if rank < 0:
        return str(value or "").strip().lower()
    if rank < effort_rank(rng["min"]):
        return rng["min"]
    if rank > effort_rank(rng["max"]):
        return rng["max"]
    return EFFORT_SCALE[rank]


def choose_effort(
    requested: Any = "", *, pin: Any = "", model_named: Any = "",
    default_top: bool = False, binds: Optional[bool] = None,
    rng: Optional[Mapping[str, str]] = None,
) -> tuple[str, str]:
    """THE effort decision for any actor: ``(level, source)``, ``source`` in EFFORT_SOURCES.

    A level encoded in the model name wins in every mode. An owner pin — a row's ``effort``,
    an explicit root effort — wins over ``requested`` while the range binds; in Cyber Pro an
    explicit request beats the pin and silence still sits on it. Otherwise the request is
    clamped into the range while it binds (``auto``) and applied as asked in Cyber Pro
    (``cyber``); with no request the helper starts nearest the primary within its bounds,
    or at ``max`` for ``default_top`` roles (reviewers, evolution, consciousness). Blank or
    unknown tiers read as absent; callers refuse an unknown request on their own surface.
    ``binds`` defaults to the runtime mode (``runtime_mode_policy.effort_range_binds``);
    ``rng`` to the task's snapshot read.
    """
    requested, pin, model_named = _tier(requested), _tier(pin), _tier(model_named)
    if model_named:
        return model_named, "model_name"
    if binds is None:
        from ouroboros.runtime_mode_policy import effort_range_binds

        binds = effort_range_binds()
    binding = bool(binds)
    if pin and (binding or not requested):
        return pin, "pin"
    rng = dict(rng) if rng is not None else effort_range()
    if requested:
        return (clamp_effort_into(requested, rng), "auto") if binding else (requested, "cyber")
    return (rng["max"] if default_top else clamp_effort_into(rng["recommended"], rng)), "auto"


def effort_fact(requested: Any, level: str, source: str) -> dict[str, str]:
    """One actor's recorded effort decision ``{requested, applied, source}``."""
    return {"requested": _tier(requested), "applied": str(level or ""), "source": str(source or "")}


def effort_fact_says(fact: Mapping[str, Any]) -> bool:
    """Whether a decision is worth a line: a request that was moved into the range or set
    aside by a pin or a model name. A request that simply applied, a pin or a name deciding
    with nothing asked, and the plain default are noise (the fact's fields still ride)."""
    requested = _tier(fact.get("requested"))
    return bool(requested) and requested != str(fact.get("applied") or "")


def effort_fact_phrase(fact: Mapping[str, Any], rng: Optional[Mapping[str, str]] = None) -> str:
    """The one clause for a decision worth saying (``effort_fact_says``); '' otherwise."""
    if not effort_fact_says(fact):
        return ""
    requested = _tier(fact.get("requested"))
    applied, source = str(fact.get("applied") or ""), str(fact.get("source") or "")
    if source == "model_name":
        return f"effort {applied}: the level in the model name; requested {requested} not applied"
    if source == "pin":
        return f"effort {applied}: pinned by my human; requested {requested} not applied"
    bounds = dict(rng) if rng is not None else effort_range()
    return f"effort {applied}: your request {requested} moved into my human's range {bounds['min']}..{bounds['max']}"


def resolve_effort(task_type: str) -> str:
    """The effort a role starts on with no pin, request or model-named level: the range's
    recommended level for ordinary work (a chat turn, a root task, a Light synthesis), its
    top for an evolution task and a consciousness wake (``effort_range``)."""
    rng = effort_range()
    return rng["max"] if (task_type or "").strip().lower() in ("evolution", "consciousness") else rng["recommended"]


# Prompt-cache TTL scale (owner decision 2026-08-08): 'default' = bare markers (provider default tier), '5m'/'1h' =
# the two documented Anthropic ephemeral tiers. Deliberately NO 'auto' (dead until an adaptive design exists) and NO '24h' (Anthropic would clamp it — a value that mostly lies).
PROMPT_CACHE_TTL_SCALE: tuple[str, ...] = ("default", "5m", "1h")


def resolve_prompt_cache_ttl() -> str:
    """The owner-configured global prompt-cache TTL ('default' | '5m' | '1h').

    Validated like ``resolve_effort``: an unknown value falls back to the shipped default.
    Consumed ONLY by the finalizer (``llm.LLMClient._normalize_payload_cache_ttl``), by
    ``review_helpers.cached_prompt_blocks`` (its marker gets stamped to the same value anyway),
    by ``effort_evidence.session_prompt_cache_ttl_request`` for native session starts,
    and by ``usage_accounting._reservation_cost`` as the payload-free admission fallback
    (payload-carrying sites use the finalizer's applied TTL) — never by per-builder marking
    sites (docs/DEVELOPMENT.md cache-friendliness invariant)."""
    default = str(SETTINGS_DEFAULTS["OUROBOROS_PROMPT_CACHE_TTL"])
    raw = str(runtime_setting("OUROBOROS_PROMPT_CACHE_TTL", default) or "").strip().lower()
    return raw if raw in PROMPT_CACHE_TTL_SCALE else default


# ``cyber_pro`` is the owner-selected high-power access level; it remains an ordinary member of the
# same closed scale so every consumer shares one vocabulary and rank.
VALID_RUNTIME_MODES = ("light", "advanced", "pro", "cyber_pro")

# Lower rank = stricter scope. ``save_settings`` refuses agent self-elevation.
_RUNTIME_MODE_RANK = {"light": 0, "advanced": 1, "pro": 2, "cyber_pro": 3}


def normalize_runtime_mode(value: Any) -> str:
    """Clamp caller-supplied runtime mode to the canonical closed enum."""
    default_val = str(SETTINGS_DEFAULTS["OUROBOROS_RUNTIME_MODE"])
    text = str(value or "").strip().lower()
    return text if text in VALID_RUNTIME_MODES else default_val


VALID_SUBAGENTS_MODES = ("off", "auto", "swarm")


def normalize_subagents_mode(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text in VALID_SUBAGENTS_MODES else str(SETTINGS_DEFAULTS["OUROBOROS_SUBAGENTS_MODE"])


VALID_SAFETY_MODES = ("full", "light", "off")


def normalize_safety_mode(value: Any) -> str:
    """Clamp caller-supplied safety mode to the closed enum (full / light / off)."""
    default_val = str(SETTINGS_DEFAULTS["OUROBOROS_SAFETY_MODE"])
    text = str(value or "").strip().lower()
    return text if text in VALID_SAFETY_MODES else default_val


_SAFETY_MODE_RANK = {"full": 2, "light": 1, "off": 0}


# Effect vocabulary shared by the owner gateway and task-local runtime readers.
IMMEDIATE_SETTINGS = frozenset({
    "TOTAL_BUDGET",
    "OUROBOROS_SUBAGENTS_MODE",  # live room rounds/admissions; never a task snapshot authority
    # The interface language is read live by the gateway (every /api/ui/i18n read) and by
    # the Telegram skill; a running task's runtime block keeps the tag it started with.
    "OUROBOROS_UI_LANGUAGE",
    # The OUTER per-call tool cap reads settings.json BEFORE env on every tool
    # call in every process (loop_tool_execution.py), so a saved change bites
    # the currently running task's next tool call. The inner shell subprocess
    # timeout still prefers the worker env (next task) — disclosed residual.
    "OUROBOROS_TOOL_TIMEOUT_SEC",
    # model_concurrency._cap() reads the settings document at every physical call in every
    # process, so a saved cap reaches running work at its next call, not at the next task.
    "OUROBOROS_MODEL_MAX_CONCURRENCY",
    # The owner's cap on acting tasks: the supervisor reads it on every assignment pass.
    "OUROBOROS_MAX_WORKERS",
    "GITHUB_TOKEN",
    "GITHUB_REPO",
    "OUROBOROS_UPDATE_CHANNEL",
    # The save handler hot-reconfigures MCP itself before responding
    # (_apply_settings_save_side_effects), and worker processes re-check the
    # settings mtime on their next tool-schema read; a reconfigure failure is
    # surfaced as a save warning instead of silently keeping the claim.
    "MCP_ENABLED",
    "MCP_SERVERS",
    "MCP_TOOL_TIMEOUT_SEC",
})

RESTART_REQUIRED_SETTINGS = frozenset({
    "OUROBOROS_SERVER_HOST",
    # The host-service port is bound once at server startup.
    "OUROBOROS_HOST_SERVICE_PORT",
    # The main interface is selected once per server generation (server_web.ServedUI.apply).
    "OUROBOROS_UI_DIR",
    # Workers adopt the server's extension generation before task-start settings
    # refresh; a changed repository path still requires a restart of the pool.
    "OUROBOROS_SKILLS_REPO_PATH",
    "LOCAL_MODEL_SOURCE",
    "LOCAL_MODEL_FILENAME",
    "LOCAL_MODEL_PORT",
    "LOCAL_MODEL_N_GPU_LAYERS",
    "LOCAL_MODEL_CONTEXT_LENGTH",
    "LOCAL_MODEL_CHAT_FORMAT",
    # The consciousness keys (autonomy, allowance, concurrency, wake-up bounds) are
    # deliberately NOT here: the alarm clock reads them at each decision
    # (consciousness.tick / set_next_wakeup), so a save applies without a restart.
})


# Presentation only: these getters run in the clock, outside an actor snapshot. A launched
# wake keeps its admission facts; bounds apply when scheduling again, not to an existing alarm.
CONSCIOUSNESS_CLOCK_SETTINGS = frozenset({
    "OUROBOROS_CONSCIOUSNESS_AUTONOMY", "OUROBOROS_CONSCIOUSNESS_DAILY_USD",
    "OUROBOROS_CONSCIOUSNESS_MAX_TASKS", "OUROBOROS_BG_WAKEUP_MIN", "OUROBOROS_BG_WAKEUP_MAX",
})


def setting_effect(key: str) -> str:
    """When an owner choice is consumed; never changes the actor's snapshot policy."""
    from ouroboros.model_slots import MODEL_CONTEXT_WINDOWS_KEY
    from ouroboros.route_choice import ROUTE_SETTING_KEYS

    if key in RESTART_REQUIRED_SETTINGS:
        return "local_model" if key.startswith("LOCAL_MODEL_") else "restart"
    if key == "OUROBOROS_RUNTIME_MODE":
        return "restart"
    if key in CONSCIOUSNESS_CLOCK_SETTINGS:
        return "next_wake"
    if key in IMMEDIATE_SETTINGS or key in {
        "OUROBOROS_NETWORK_PASSWORD",  # server_auth.NetworkAuthGate: every request
        "OUROBOROS_FINALIZATION_GRACE_SEC",  # queue.refresh_timeouts_from_settings -> queue_timeouts
        "OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS",  # config._settings_flag_enabled: disk on every read
    }:
        return "now"
    if key in ROUTE_SETTING_KEYS and key != MODEL_CONTEXT_WINDOWS_KEY:
        return "next_call"
    return "next_task"


# Optional positive bounds: a positive integer, or the literal "unlimited" for no bound. ONE
# vocabulary for every such knob — the shared paid-review-cycle cap (review_cycles.py), the task
# round limit and the absolute task lifetime — so the Settings UI, the settings write boundary and
# every runtime reader spell "no limit" alike. "none" is deliberately NOT an alias: it reads as
# "zero" as easily as "no cap".
UNLIMITED = "unlimited"
UNLIMITED_ALIASES = frozenset({UNLIMITED, "inf", "∞"})


def parse_positive_or_unlimited(raw: Any) -> Optional[int]:
    """Strict parser: positive-integer text → int; an unlimited alias → None.

    Raises ``ValueError`` for anything else (empty, zero, negative, non-integer,
    unknown word) so each caller decides between its own fail-closed read and a
    400 at the write boundary."""
    text = str(raw if raw is not None else "").strip().lower()
    if text in UNLIMITED_ALIASES:
        return None
    if not text:
        raise ValueError("empty bound")
    value = int(text)  # ValueError on non-integer text (incl. "true"/"1.5")
    if value < 1:
        raise ValueError(f"bound must be a positive integer, got {value}")
    return value


# The task round limit and the absolute task lifetime ship as "unlimited" (#1196): a fresh
# install — no settings document yet — bounds a task by money, deadlines, Stop/Panic and the idle
# rail, not by its age or round count. A document an earlier release wrote without one of these
# keys ran under that release's finite default, so readers keep THAT value as the key's default for
# such a document (``defaults_for_settings_document``) and as the typed fallback for a malformed
# value: an update never silently lifts a bound an install was running under, a typo never means
# "no bound", and a read never rewrites the document.
OPTIONAL_BOUND_LEGACY: dict[str, int] = {
    "OUROBOROS_MAX_ROUNDS": 200,
    "OUROBOROS_TASK_ABS_CEILING_SEC": 21600,
}
_WARNED_OPTIONAL_BOUNDS: set = set()


def optional_bound_value(key: str, raw: Any) -> Optional[int]:
    """One read of an optional bound or the owner cap: ``None`` = no bound, else a positive int.

    An integral float is its integer (a harness may write ``10800.0``). Blank or null means
    no cap only for the owner cap. Everything else the strict parser refuses — including
    blank/null for other bounds — is a typo, never "no bound": it takes the key's finite
    legacy value, reported once per process and value."""
    owner_cap = key == OWNER_CAP_KEY
    if owner_cap and (raw is None or isinstance(raw, str) and not raw.strip()):
        return None
    value = int(raw) if isinstance(raw, float) and raw.is_integer() else raw
    try:
        return parse_positive_or_unlimited(value)
    except (TypeError, ValueError):
        fallback = OWNER_CAP_LEGACY if owner_cap else OPTIONAL_BOUND_LEGACY[key]
        if (key, repr(raw)) not in _WARNED_OPTIONAL_BOUNDS:
            _WARNED_OPTIONAL_BOUNDS.add((key, repr(raw)))
            logging.getLogger(__name__).warning(
                "%s=%r is not a positive integer or %s; using the finite fallback %s",
                key, raw, "blank" if owner_cap else repr(UNLIMITED), fallback)
        return fallback


def defaults_for_settings_document(document_present: bool) -> dict:
    """The defaults a reader merges under the settings document: the shipped values, with each
    optional bound's finite legacy value while a document exists (``OPTIONAL_BOUND_LEGACY``).
    Every writer that creates a document persists a defaults-merged one (the context-mode
    compatibility pass only rewrites an existing document), so only a document an earlier
    release or a harness wrote without the key falls to the legacy value. Every view carries
    the owner cap's format marker, so the first save of a fresh install keeps a cap it sets."""
    defaults = {**SETTINGS_DEFAULTS, OWNER_CAP_FORMAT_KEY: 1}
    if document_present:
        defaults.update(OPTIONAL_BOUND_LEGACY)
    return defaults


# The owner's optional cap on acting project tasks (pool-capacity Q1 = D): blank, the shipped default, is no number
# (the machine's memory is the limit, ``supervisor.pool_admission``), a positive integer the cap ("unlimited" reads
# blank), anything else a typo that reads finite, never "no cap" (the ``OPTIONAL_BOUND_LEGACY`` rule). A document
# without the format marker was written while the settings form saved 10 on every save: its 10 reads once as blank
# (owner decision 09.11, migration A; Colab's first-launch 1 likewise, ``colab_bootstrap``); 40, say, stays the cap.
OWNER_CAP_KEY = "OUROBOROS_MAX_WORKERS"
OWNER_CAP_LEGACY = 10
OWNER_CAP_FORMAT_KEY = "_owner_cap_format"  # disk format, never an environment setting or owner knob


def owner_cap_document(raw: Any) -> Any:
    """The read seam's migration of a document from before the optional cap (no format marker): its saved 10 reads
    as blank. A read stays a read; every view carries the marker (``defaults_for_settings_document``), so the next
    whole-document save ends the migration, and a partial save writes the blank the migration read."""
    if (not isinstance(raw, Mapping) or raw.get(OWNER_CAP_FORMAT_KEY) == 1
            or raw.get(OWNER_CAP_KEY) not in (OWNER_CAP_LEGACY, str(OWNER_CAP_LEGACY))):
        return raw
    return {**raw, OWNER_CAP_KEY: ""}


def document_optional_value(key: str, value: Any) -> Any:
    """The document spelling of an optional bound or the owner cap: a positive int, else "unlimited" (a bound) or
    blank (the cap); a typo reads finite."""
    bound = optional_bound_value(key, value)
    if bound is not None:
        return bound
    return "" if key == OWNER_CAP_KEY else UNLIMITED
