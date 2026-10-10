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

# The owner's effort range: three flat keys read as one ordered triple (``effort_range``).
EFFORT_RANGE_KEYS: tuple[str, ...] = ("OUROBOROS_EFFORT_MIN", "OUROBOROS_EFFORT_TASK", "OUROBOROS_EFFORT_MAX")
# The tiers the owner's range control offers; `minimal` stays a runtime tier a stored value may name.
OWNER_EFFORT_TIERS: tuple[str, ...] = tuple(tier for tier in EFFORT_SCALE if tier != "minimal")
# Who decided one actor's effort (``choose_effort``); task records and chat frames carry it verbatim.
EFFORT_SOURCES: tuple[str, ...] = ("auto", "pin", "model_name", "cyber")
# The lane-era global review effort, kept ONLY as the frozen legacy skill-review fingerprint
# literal (``skill_review_cycles``): an empty reviewer row reviews at the range's top, never here.
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


def effort_range(settings: Optional[Mapping[str, Any]] = None) -> dict[str, str]:
    """The owner's effort range ``{min, recommended, max}`` — THE tolerant read.

    ``recommended`` is ``OUROBOROS_EFFORT_TASK``, the level Main works at. An unknown or
    blank value takes its key's shipped default; ``min`` is never above and ``max`` never
    below ``recommended`` (a document that carries only TASK=high reads low/high/high).
    Snapshot-aware through ``runtime_setting`` (a running task keeps the range it started
    with) unless a settings document is given (``settings_integrity.live_effort_range``
    reads the owner's current one for a participant starting inside a running task). A
    read never rewrites the document.
    """
    def read(key: str) -> str:
        default = str(SETTINGS_DEFAULTS[key])
        raw = settings.get(key, default) if settings is not None else runtime_setting(key, default)
        return _tier(raw) or default

    low, recommended, high = (read(key) for key in EFFORT_RANGE_KEYS)
    return {"min": min(low, recommended, key=effort_rank), "recommended": recommended,
            "max": max(high, recommended, key=effort_rank)}


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
    (``cyber``); with no request the role default applies: the range's ``recommended``, or
    its ``max`` for ``default_top`` roles (reviewers, evolution, consciousness). Blank or
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
    return (rng["max"] if default_top else rng["recommended"]), "auto"


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
    and by ``usage_accounting._reservation_cost`` as the payload-free admission fallback
    (payload-carrying sites use the finalizer's applied TTL) — never by per-builder marking
    sites (docs/DEVELOPMENT.md cache-friendliness invariant)."""
    default = str(SETTINGS_DEFAULTS["OUROBOROS_PROMPT_CACHE_TTL"])
    raw = str(runtime_setting("OUROBOROS_PROMPT_CACHE_TTL", default) or "").strip().lower()
    return raw if raw in PROMPT_CACHE_TTL_SCALE else default


# Runtime mode and review enforcement are separate axes.  ``cyber_pro`` is the
# owner-selected high-power access level; it remains an ordinary member of the
# same closed scale so every consumer shares one vocabulary and rank.
VALID_RUNTIME_MODES = ("light", "advanced", "pro", "cyber_pro")

# Lower rank = stricter scope. ``save_settings`` refuses agent self-elevation.
_RUNTIME_MODE_RANK = {"light": 0, "advanced": 1, "pro": 2, "cyber_pro": 3}


def normalize_runtime_mode(value: Any) -> str:
    """Clamp caller-supplied runtime mode to the canonical closed enum."""
    default_val = str(SETTINGS_DEFAULTS["OUROBOROS_RUNTIME_MODE"])
    text = str(value or "").strip().lower()
    return text if text in VALID_RUNTIME_MODES else default_val


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
    # The interface language is read live by the gateway (every /api/ui/i18n read) and by
    # the Telegram skill; a running task's runtime block keeps the tag it started with.
    "OUROBOROS_UI_LANGUAGE",
    # The OUTER per-call tool cap reads settings.json BEFORE env on every tool
    # call in every process (loop_tool_execution.py), so a saved change bites
    # the currently running task's next tool call. The inner shell subprocess
    # timeout still prefers the worker env (next task) — disclosed residual.
    "OUROBOROS_TOOL_TIMEOUT_SEC",
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
    "OUROBOROS_MAX_WORKERS",
    "OUROBOROS_SERVER_HOST",
    # The host-service port is bound once at server startup.
    "OUROBOROS_HOST_SERVICE_PORT",
    # Pooled workers load the extension registry once at spawn and never
    # reload it per task; the save-time server reload keeps the skills UI
    # fresh, but agent tasks see the new repo only after a restart.
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
    """One read of an ``OPTIONAL_BOUND_LEGACY`` key: ``None`` = no bound, else a positive int.

    An integral float is its integer (a harness may write ``10800.0``). Whatever the strict
    parser refuses — blank, null, zero, negative, a fraction, a word — is a typo, never "no
    bound": it takes the key's finite legacy value, reported once per process and value."""
    value = int(raw) if isinstance(raw, float) and raw.is_integer() else raw
    try:
        return parse_positive_or_unlimited(value)
    except (TypeError, ValueError):
        fallback = OPTIONAL_BOUND_LEGACY[key]
        if (key, repr(raw)) not in _WARNED_OPTIONAL_BOUNDS:
            _WARNED_OPTIONAL_BOUNDS.add((key, repr(raw)))
            logging.getLogger(__name__).warning(
                "%s=%r is not a positive integer or %r; using the finite fallback %s",
                key, raw, UNLIMITED, fallback)
        return fallback


def defaults_for_settings_document(document_present: bool) -> dict:
    """The defaults a reader merges under the settings document: the shipped values, with each
    optional bound's finite legacy value while a document exists (``OPTIONAL_BOUND_LEGACY``).
    Every writer that creates a document persists a defaults-merged one (the context-mode
    compatibility pass only rewrites an existing document), so only a document an earlier
    release or a harness wrote without the key falls to the legacy value."""
    defaults = dict(SETTINGS_DEFAULTS)
    if document_present:
        defaults.update(OPTIONAL_BOUND_LEGACY)
    return defaults
