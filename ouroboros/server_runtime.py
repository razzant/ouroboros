"""Helpers shared by server startup, onboarding, and WebSocket liveness."""

from __future__ import annotations

import asyncio
import os
from typing import Awaitable, Callable

from ouroboros.provider_models import (
    DIRECT_PROVIDER_DEFAULTS,
    compute_direct_review_models_fallback,
    migrate_model_value,
)
from ouroboros.config import SETTINGS_DEFAULTS, _parse_model_list
from ouroboros.utils import utc_now_iso


# The deep-review model key is a RETIRED setting (PR-3): the read seam migrates a
# stored value into a catalog row, so no provider default is applied to it here.
_MODEL_ROLE_SETTING_KEYS = {
    "main": "OUROBOROS_MODEL",
    "light": "OUROBOROS_MODEL_LIGHT",
    "fallback": "OUROBOROS_MODEL_FALLBACKS",
}
_DIRECT_PROVIDER_AUTO_DEFAULTS = {
    provider: {
        setting_key: defaults[role]
        for role, setting_key in _MODEL_ROLE_SETTING_KEYS.items()
        if role in defaults
    }
    for provider, defaults in DIRECT_PROVIDER_DEFAULTS.items()
}
# Heavy is no longer an active role, but its bounded migration reader still
# needs to distinguish an owner's custom value from values Ouroboros itself
# shipped as Heavy/Code defaults (both spellings, every release that shipped
# one, plus each provider's current Heavy default), so an old default is not
# laundered into a new explicit API actor. This retired key is the only place
# equality with a former default still classifies a value: active model slots
# keep whatever was saved (``provider_slot_proposals`` fills only absent ones).
_SHIPPED_LEGACY_HEAVY_DEFAULTS = frozenset({
    "google/gemini-" + "3.1-flash-lite", "google/gemini-3.5-flash",
    "anthropic/claude-opus-4.6", "anthropic::claude-opus-4-6",
    "anthropic/claude-opus-4.7", "anthropic::claude-opus-4-7",
    "anthropic/claude-opus-4.8", "anthropic::claude-opus-4-8",
    "openai/gpt-5.5", "openai::gpt-5.5", "openai::gpt-" + "5.4",
    "openai::gpt-5.6-sol", "anthropic::claude-opus-5", "gigachat::GigaChat-3-Ultra",
    *(
        str(provider_defaults.get("heavy", "") or "").strip()
        for provider_defaults in DIRECT_PROVIDER_DEFAULTS.values()
        if str(provider_defaults.get("heavy", "") or "").strip()
    ),
})
# The slots that take a provider proposal while nobody authored them.
MODEL_SLOT_KEYS = _ALL_MODEL_SLOT_KEYS = tuple(_MODEL_ROLE_SETTING_KEYS.values())
# The model roles whose saved value, blank included, beats an operator variable: those slots plus
# Vision and Consciousness, whose blank is the owner's "use Main" and which take no proposal.
SAVED_MODEL_ROLE_KEYS = (*MODEL_SLOT_KEYS, "OUROBOROS_MODEL_VISION", "OUROBOROS_MODEL_CONSCIOUSNESS")
_SCOPE_REVIEW_LEGACY_DEFAULTS = frozenset({
    "",
    "anthropic/claude-opus-4.6",
    "anthropic::claude-opus-4-6",
    "openai/gpt-5.5",
    "openai::gpt-5.5",
    "openai/gpt-5.5-pro",
    "openai::gpt-5.5-pro",
    "openai/gpt-" + "5.4",
    "openai::gpt-" + "5.4",
    "openai/gpt-" + "5.4-pro",
    "openai::gpt-" + "5.4-pro",
    "openai/gpt-" + "5.4-mini",
    "openai::gpt-" + "5.4-mini",
    # The v6.55.0-v6.81 shipped scope-review default (both spellings).
    "anthropic/claude-fable-5",
    "anthropic::claude-fable-5",
})
# The prior shipped scope-review defaults (gpt-5.5 pre-v6.55.0, claude-fable-5
# v6.55.0-v6.81; both spellings each). On the aggregator/general path (no exclusive
# direct provider) a saved value equal to one of them was an old shipped DEFAULT, not
# an explicit user choice — remap it to the current shipped default (gpt-5.6-terra as
# of v6.82.0) so an upgraded install picks up the current scope reviewer instead of
# silently keeping an off-default slot. Scope-review keys ONLY (gpt-5.5 and fable-5
# stay first-class for main/triad slots); the direct-provider path is untouched —
# it already migrates these values via _SCOPE_REVIEW_LEGACY_DEFAULTS above.
_SCOPE_REVIEW_PRIOR_DEFAULTS = frozenset({
    "openai/gpt-5.5", "openai::gpt-5.5",
    "anthropic/claude-fable-5", "anthropic::claude-fable-5",
})
# No table here declares an external model "retired": a saved model id is the owner's
# choice and stays as written, and a model a provider really withdrew answers with that
# provider's own error. Only equality with one of OUR former shipped defaults migrates
# (the sets above), because only then is the value ours rather than the owner's.


def _truthy_setting(value) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes", "on"}


def _setting_text(settings: dict, key: str) -> str:
    return str(settings.get(key, "") or "").strip()


def _serialize_model_list(models: list[str]) -> str:
    return ",".join(model.strip() for model in models if str(model or "").strip())


def _unique_changed_keys(keys: list[str]) -> list[str]:
    return list(dict.fromkeys(keys))


def _migrate_scope_review_prior_default(settings: dict) -> tuple[dict, list[str]]:
    # ABI 7.0 (ABI-10): the comma keys are RETIRED settings — load_settings
    # purges them before this runs, so the branch fires only on a raw dict fed
    # directly (tests/tools). The replacement source is the shipped default
    # (the keys left SETTINGS_DEFAULTS with the retirement).
    from ouroboros.settings_defaults import OPENROUTER_REVIEW_DEFAULTS

    normalized = dict(settings)
    changed: list[str] = []
    for key in ("OUROBOROS_SCOPE_REVIEW_MODEL", "OUROBOROS_SCOPE_REVIEW_MODELS"):
        if _setting_text(normalized, key) in _SCOPE_REVIEW_PRIOR_DEFAULTS:
            normalized[key] = ",".join(OPENROUTER_REVIEW_DEFAULTS["scope"])
            changed.append(key)
    return normalized, changed


def _provider_prefix(provider: str) -> str:
    return f"{provider}::"


def _exclusive_direct_remote_provider(settings: dict) -> str:
    """The one direct provider the settings reach, else '' (``provider_models.exclusive_direct_kind``)."""
    from ouroboros.provider_models import exclusive_direct_kind

    return exclusive_direct_kind(settings or {})


def _normalize_direct_review_models(settings: dict, provider: str) -> str:
    main_model = migrate_model_value(provider, _setting_text(settings, "OUROBOROS_MODEL"))
    current_models = _parse_model_list(_setting_text(settings, "OUROBOROS_REVIEW_MODELS"))
    migrated_models = [migrate_model_value(provider, model) for model in current_models]
    provider_prefix = _provider_prefix(provider)

    if not main_model.startswith(provider_prefix):
        return _serialize_model_list(migrated_models)

    has_foreign_models = any(not model.startswith(provider_prefix) for model in migrated_models)
    # Honor an explicit provider-matching list exactly (a single model stays a
    # single slot); expand to the stochastic fallback only when empty/foreign.
    if not migrated_models or has_foreign_models:
        user_light_raw = _setting_text(settings, "OUROBOROS_MODEL_LIGHT")
        fallback = compute_direct_review_models_fallback(provider, main_model, user_light_raw)
        return _serialize_model_list(fallback)
    return _serialize_model_list(migrated_models)


def _normalize_direct_scope_review_model(settings: dict, provider: str) -> str:
    current_raw = _setting_text(settings, "OUROBOROS_SCOPE_REVIEW_MODEL")
    default_raw = _setting_text(SETTINGS_DEFAULTS, "OUROBOROS_SCOPE_REVIEW_MODEL")
    current = migrate_model_value(provider, current_raw) if current_raw else ""
    default = migrate_model_value(provider, default_raw) if default_raw else ""
    provider_prefix = _provider_prefix(provider)
    # The scope lane's per-provider default was the provider's Main (the former
    # ``DIRECT_PROVIDER_SCOPE_DEFAULTS``, retired with the lane).
    auto_value = migrate_model_value(
        provider, DIRECT_PROVIDER_DEFAULTS.get(provider, {}).get("main", ""),
    )
    legacy_defaults = {
        migrate_model_value(provider, item) for item in _SCOPE_REVIEW_LEGACY_DEFAULTS
    }
    if current_raw in {"", default_raw, *_SCOPE_REVIEW_LEGACY_DEFAULTS} or current in {"", default, *legacy_defaults}:
        return auto_value
    if current.startswith(provider_prefix) and current_raw:
        return current
    return current or auto_value


def _normalize_direct_scope_review_models(settings: dict, provider: str) -> str:
    raw = _setting_text(settings, "OUROBOROS_SCOPE_REVIEW_MODELS")
    models = _parse_model_list(raw)
    if not models:
        singular = _normalize_direct_scope_review_model(settings, provider)
        return _serialize_model_list([singular] if singular else [])
    migrated = [migrate_model_value(provider, model) for model in models]
    provider_prefix = _provider_prefix(provider)
    if raw == _setting_text(SETTINGS_DEFAULTS, "OUROBOROS_SCOPE_REVIEW_MODELS"):
        return _serialize_model_list([_normalize_direct_scope_review_model(settings, provider)])
    # A SINGLETON list still holding a prior shipped scope default migrates exactly
    # like the singular key. Without this the two keys go split-brain — the runtime
    # prefers the plural one, so an upgraded install would keep the retired
    # sub-floor reviewer and fail every Max-mode commit against the BIBLE P3 floor.
    legacy_scope_defaults = _SCOPE_REVIEW_LEGACY_DEFAULTS | _SCOPE_REVIEW_PRIOR_DEFAULTS
    if len(models) == 1 and (
        models[0] in legacy_scope_defaults or migrated[0] in legacy_scope_defaults
    ):
        return _serialize_model_list([_normalize_direct_scope_review_model(settings, provider)])
    if all(model.startswith(provider_prefix) for model in migrated):
        return _serialize_model_list(migrated)
    return _serialize_model_list([_normalize_direct_scope_review_model(settings, provider)])


def classify_runtime_provider_change(before: dict, after: dict) -> str:
    """Classify what kind of normalization ``apply_runtime_provider_defaults`` did.

    Returns one of:

    - ``"none"`` — no change, or change was purely cosmetic.
    - ``"direct_normalize"`` — OpenRouter is NOT configured, and the function
      auto-filled direct-provider defaults.  This is the only case where a
      user-facing warning is appropriate.
    - ``"reverse_migrate"`` — OpenRouter IS configured (so no exclusive-direct
      provider is active).  ``apply_runtime_provider_defaults`` returned early
      without making any changes, so this is pure housekeeping and should NOT
      produce a warning.
    """
    from ouroboros.provider_models import configured_kinds

    provider_after = _exclusive_direct_remote_provider(after)
    if provider_after:
        return "direct_normalize"
    if "openrouter" in configured_kinds(after or {}):
        return "reverse_migrate"
    return "none"


def has_remote_provider(settings: dict) -> bool:
    """Return configured access, independently of a source's momentary health."""
    from ouroboros.provider_models import parse_claudexor_model, provider_for_model

    main = str(settings.get("OUROBOROS_MODEL") or "")
    if provider_for_model(main) == "claudexor":
        try:
            parse_claudexor_model(main)
            return True
        except ValueError:
            pass
    from ouroboros.provider_models import any_api_connection_configured

    return any_api_connection_configured(settings)


def has_local_routing(settings: dict) -> bool:
    """Return True when a task-capable model slot is routed to local."""
    return any(
        _truthy_setting(settings.get(k))
        for k in ("USE_LOCAL_MAIN", "USE_LOCAL_LIGHT", "USE_LOCAL_FALLBACK")
    )


def needs_local_model_autostart(settings: dict) -> bool:
    """Return True when a configured root lane or API subagent needs local serving."""
    if any(
        _truthy_setting(settings.get(k))
        for k in (
            "USE_LOCAL_MAIN", "USE_LOCAL_LIGHT",
            "USE_LOCAL_CONSCIOUSNESS", "USE_LOCAL_FALLBACK",
        )
    ):
        return True
    try:
        from ouroboros.configured_subagents import resolve_configured_subagents

        resolution = resolve_configured_subagents(settings)
        config = resolution.config
        return bool(
            config
            and config.enabled
            and any(
                row.enabled
                and row.route.kind == "api_model"
                and row.route.target_id.endswith(" (local)")
                for row in config.items
            )
        )
    except Exception:
        return False


def has_startup_ready_provider(settings: dict) -> bool:
    """Return True when the runtime has enough provider config to serve chat.

    Used both by startup/onboarding gating and by the supervisor-start gate.
    A local model source alone is not enough unless at least one lane is
    routed to that local runtime. (The packaged launcher imports this name —
    keep it stable.)
    """
    return has_remote_provider(settings) or has_local_routing(settings)


_LOCAL_LANE_FLAG_FOR_SLOT = {
    "OUROBOROS_MODEL_LIGHT": "USE_LOCAL_LIGHT",
    "OUROBOROS_MODEL_FALLBACKS": "USE_LOCAL_FALLBACK",
}


def authored_model_slots(document) -> set[str]:
    """The active model slots ``document`` carries as a choice: any saved value, blank included
    (blank Light follows Main, blank Fallbacks means no cross-model fallback, a blank Main runs
    as ``model_slots._main_model`` reads it). Only an absent slot takes a proposal."""
    return {key for key in _ALL_MODEL_SLOT_KEYS if key in document}


def provider_slot_proposals(settings: dict) -> dict:
    """What an UNAUTHORED Main/Light/Fallbacks slot takes, from the configured access.

    One exclusive direct provider: its own defaults. A local-only install: blank Light (it
    follows Main) and blank Fallbacks (no chain onto the same runtime) — the setup contract's
    ``local`` profile proposes the same — except a lane routed local while Main is not, whose
    shipped id is then a route the install runs (for Fallbacks the only chain entry). Otherwise
    the shipped defaults. A proposal never replaces a saved value (DEVELOPMENT: Provider Independence).
    """
    provider = _exclusive_direct_remote_provider(settings)
    if provider:
        return dict(_DIRECT_PROVIDER_AUTO_DEFAULTS[provider])
    proposals = {key: _setting_text(SETTINGS_DEFAULTS, key) for key in _ALL_MODEL_SLOT_KEYS}
    if not has_remote_provider(settings) and _setting_text(settings, "LOCAL_MODEL_SOURCE"):
        local_main = _truthy_setting(settings.get("USE_LOCAL_MAIN"))
        proposals.update({key: "" for key, flag in _LOCAL_LANE_FLAG_FOR_SLOT.items()
                          if local_main or not _truthy_setting(settings.get(flag))})
    return proposals


def propose_unauthored_model_slots(settings: dict, authored) -> dict:
    """``settings`` with every slot outside ``authored`` set to its proposal (in place)."""
    proposals = provider_slot_proposals(settings)
    settings.update({key: proposals[key] for key in _ALL_MODEL_SLOT_KEYS if key not in authored})
    return settings


def repropose_unedited_slots(settings: dict, document, written, shown) -> set[str]:
    """The Settings save's rule: the ``unedited_unauthored_slots`` are re-proposed in ``settings``
    for the access the save now carries (Vision/Consciousness take none), and returned so the caller
    does not persist them. One an operator's variable sets (``settings_integrity.operator_variables``)
    keeps that value, as the next read takes it: in this process, its children and the next task."""
    from ouroboros.settings_integrity import operator_variables

    unauthored = unedited_unauthored_slots(document, written, shown)
    operator = operator_variables(os.environ, unauthored)
    propose_unauthored_model_slots(settings, set(_ALL_MODEL_SLOT_KEYS) - (unauthored - set(operator)))
    settings.update(operator)
    return unauthored


def unedited_unauthored_slots(document, written, shown) -> set[str]:
    """The ``SAVED_MODEL_ROLE_KEYS`` a write leaves unauthored: absent from the ``document`` on disk and
    either not ``written`` or written exactly as the writer read them (``shown``; for the Settings save, what
    GET showed, respelled onto the exclusive provider: an operator's ``openai/x`` reads ``openai::x``).
    Settings posts every model role and an owner endpoint writes back the whole document, so posting
    a proposal back unchanged is not a choice; the key stays absent and keeps following the configured access."""
    return {key for key in SAVED_MODEL_ROLE_KEYS if key not in document
            and (key not in written or str(written[key]) == str(shown.get(key)))}


def settings_read_catalog(settings: dict):
    """The subagent catalog a Settings read of the loaded ``settings`` shows: the one in force, else the
    unsaved candidate offered in its place (``resolve_settings_subagent_candidate``), else ``None``."""
    from ouroboros.configured_subagents import (
        SUBAGENTS_SETTING,
        configured_subagents_dict,
        resolve_settings_subagent_candidate,
    )

    shown = settings.get(SUBAGENTS_SETTING)
    if shown in (None, ""):
        config = resolve_settings_subagent_candidate(apply_runtime_provider_defaults(dict(settings))[0])[0].config
        shown = configured_subagents_dict(config) if config is not None else None
    return shown


def repropose_unedited_catalog(settings: dict, document, written, shown) -> set[str]:
    """The Settings save's rule for the subagent catalog (owner decision: an Agents edit or onboarding completion
    authors one, an unrelated save does not). A never-configured ``document`` (``review_pool_migration.
    migration_trigger``) authors none: the read shows its factory rows, an operator's environment catalog or the
    unsaved candidate. A write leaving that catalog as ``shown`` (not ``written``, or equal once canonical: Settings
    posts the loaded catalog with every save) keeps it unauthored; ``settings`` then carries what the next read
    takes for the access the write carries. Returns the key the caller does not persist."""
    from ouroboros.configured_subagents import SUBAGENTS_SETTING, canonical_subagents
    from ouroboros.review_pool_migration import TRIGGER_NEVER_CONFIGURED, apply_at_read_seam, migration_trigger
    from ouroboros.settings_integrity import operator_variables

    posted = written.get(SUBAGENTS_SETTING)
    if migration_trigger(document) != TRIGGER_NEVER_CONFIGURED or (
            posted not in (None, "") and canonical_subagents(posted) != canonical_subagents(settings_read_catalog(shown))):
        return set()
    if automatic_catalog(document):  # our own projection is no operator catalog: the factory rows follow the access
        settings[SUBAGENTS_SETTING] = ""
        apply_at_read_seam(settings, merged=True)
    else:  # an operator's catalog keeps standing in, byte for byte
        settings.update(operator_variables(os.environ, (SUBAGENTS_SETTING,)))
    return {SUBAGENTS_SETTING}


def automatic_catalog(document) -> bool:
    """Whether the catalog a read shows for ``document`` is the read's own, following Main and access (Settings →
    Agents says Automatic): the document authors none (never-configured) and no operator's variable stands in."""
    from ouroboros.configured_subagents import SUBAGENTS_SETTING
    from ouroboros.review_pool_migration import TRIGGER_NEVER_CONFIGURED, migration_trigger
    from ouroboros.settings_integrity import operator_variables

    return (migration_trigger(document) == TRIGGER_NEVER_CONFIGURED
            and not operator_variables(os.environ, (SUBAGENTS_SETTING,)))


def _clear_shipped_legacy_heavy(settings: dict) -> list[str]:
    """Remove only old product-authored Heavy defaults before actor migration.

    ``USE_LOCAL_HEAVY`` is explicit saved owner intent, so it wins even while
    the local source is temporarily unavailable.  Arbitrary Heavy values also
    survive verbatim; only values already classified by the existing default
    migration tables are removed.
    """
    current = _setting_text(settings, "OUROBOROS_MODEL_HEAVY")
    if not current or _truthy_setting(settings.get("USE_LOCAL_HEAVY")):
        return []
    if current not in _SHIPPED_LEGACY_HEAVY_DEFAULTS:
        return []
    settings["OUROBOROS_MODEL_HEAVY"] = ""
    return ["OUROBOROS_MODEL_HEAVY"]


def apply_runtime_provider_defaults(settings: dict) -> tuple[dict, bool, list[str]]:
    """Every route consumer's normalization of a loaded mapping; it never replaces a model value.

    The retired Heavy key's shipped defaults are cleared, and with exactly one direct
    provider a saved id is respelled onto that provider's prefix (``migrate_model_value``:
    the same model, ``openai/x`` -> ``openai::x``). Unauthored slots already hold their
    ``provider_slot_proposals`` (the loader); a saved value — blank, former default or a
    route without access — stays as written, honestly unavailable when unreachable."""
    normalized = dict(settings)
    changed_keys = _clear_shipped_legacy_heavy(normalized)
    provider = _exclusive_direct_remote_provider(normalized)

    if not provider:
        normalized, scope_changed = _migrate_scope_review_prior_default(normalized)
        changed_keys = _unique_changed_keys(changed_keys + scope_changed)
        return normalized, bool(changed_keys), changed_keys

    for key in _ALL_MODEL_SLOT_KEYS:
        raw_current = _setting_text(normalized, key)
        current = migrate_model_value(provider, raw_current)
        if current != raw_current:
            normalized[key] = current
            changed_keys.append(key)

    # ABI 7.0 (ABI-10): the comma keys are RETIRED settings. A ghost value fed
    # directly still normalizes (owner-value preservation on a stale dict), but
    # an absent key is never INTRODUCED. Live reviews read marked catalog rows;
    # this tolerance for stale dicts does not configure the review pool.
    if _setting_text(normalized, "OUROBOROS_REVIEW_MODELS"):
        review_models = _normalize_direct_review_models(normalized, provider)
        if review_models != _setting_text(normalized, "OUROBOROS_REVIEW_MODELS"):
            normalized["OUROBOROS_REVIEW_MODELS"] = review_models
            changed_keys.append("OUROBOROS_REVIEW_MODELS")

    if _setting_text(normalized, "OUROBOROS_SCOPE_REVIEW_MODEL"):
        scope_review_model = _normalize_direct_scope_review_model(normalized, provider)
        if scope_review_model != _setting_text(normalized, "OUROBOROS_SCOPE_REVIEW_MODEL"):
            normalized["OUROBOROS_SCOPE_REVIEW_MODEL"] = scope_review_model
            changed_keys.append("OUROBOROS_SCOPE_REVIEW_MODEL")

    if _setting_text(normalized, "OUROBOROS_SCOPE_REVIEW_MODELS"):
        scope_review_models = _normalize_direct_scope_review_models(normalized, provider)
        if scope_review_models != _setting_text(normalized, "OUROBOROS_SCOPE_REVIEW_MODELS"):
            normalized["OUROBOROS_SCOPE_REVIEW_MODELS"] = scope_review_models
            changed_keys.append("OUROBOROS_SCOPE_REVIEW_MODELS")

    changed_keys = _unique_changed_keys(changed_keys)
    return normalized, bool(changed_keys), changed_keys


def setup_remote_if_configured(settings: dict, log) -> None:
    """Set up GitHub remote when credentials are configured."""
    slug = settings.get("GITHUB_REPO", "")
    token = settings.get("GITHUB_TOKEN", "")
    if not token:
        return
    from supervisor.git_ops import configure_personal_remote

    # configure_personal_remote ensures the official `managed` remote exists before
    # repointing `origin`, so a plain official clone never loses official updates.
    remote_ok, remote_msg, _resolved = configure_personal_remote(
        slug,
        token,
        auto_fork=not bool(str(slug or "").strip()),
    )
    if not remote_ok:
        log.warning("Remote configuration failed on startup: %s", remote_msg)


async def ws_heartbeat_loop(
    has_clients_fn: Callable[[], bool],
    broadcast_fn: Callable[[dict], Awaitable[None]],
    interval_sec: float = 15.0,
) -> None:
    """Keep embedded clients active and give watchdogs a steady liveness signal."""
    while True:
        await asyncio.sleep(interval_sec)
        if not has_clients_fn():
            continue
        await broadcast_fn({
            "type": "heartbeat",
            "ts": utc_now_iso(),
        })
