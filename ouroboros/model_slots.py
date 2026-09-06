"""Ouroboros — model slot resolution.

The Main/Heavy/Light/Vision/Consciousness/deep-review slots and the ordered
cross-model fallback chain, resolved from the environment with the shipped
defaults as the floor, plus the rename-alias migration that keeps a slot the
owner customized under its former key. Imported by ``provider_models`` as well
as by ``config``, which is why it holds no settings-file knowledge.
"""

from __future__ import annotations

import dataclasses
import os

from ouroboros.settings_defaults import SETTINGS_DEFAULTS


@dataclasses.dataclass(frozen=True, slots=True)
class ResolvedModelTarget:
    """One fully RESOLVED model destination — the output of route resolution (ABI-4).

    Constructed ONLY at the existing resolution seams (the cross-model fallback
    ladder, the reviewer model lists, the delegated-route parse) — see
    ``provider_models.resolve_model_target`` — and consumed downstream as a
    value, never re-parsed from a comma/at string. Absent facts are typed
    sentinels (``""`` / ``0``), never None-vs-missing ambiguity. Deliberately
    NO pricing fields: cost stays with the provider-route pricing SSOT
    (hardcoded price tables remain banned).
    """

    # Exact provider model id, e.g. "anthropic/claude-sonnet-4.6" or "openai::gpt-5.6-sol".
    model_id: str
    # Resolved transport lane: "openrouter" | "openai" | ... | "local" for API
    # routes (``provider_for_model`` vocabulary), or the OPAQUE harness route id
    # on delegated agent-session lanes (never interpreted — AGENTS.md).
    provider_route: str
    # Which configured credential/profile serves the call ("" = the provider default).
    credential_ref: str = ""
    # Normalized reasoning-effort label ("" when N/A at this seam).
    effort: str = ""
    # Tokens; 0 = unknown (fail-open per the cost-unknown rule — windows stay
    # Capability Evidence's fact, this seam never probes for one).
    context_window: int = 0


def _parse_model_list(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


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
    key. One step of ``config.normalize_settings_raw``, the raw-stage seam every settings
    reader applies (``load_settings``, the owner reader and the Colab builder alike).
    (ABI 7.0/ABI-10: the singular scope-review pin promotion is gone — both
    comma spellings are retired settings keys, purged before this runs.)"""
    for _old, _new in _LEGACY_SLOT_RENAMES:
        if _new not in settings and _old in settings:
            settings[_new] = settings[_old]
        settings.pop(_old, None)
    return settings


def get_consciousness_model() -> str:
    """Return the high-horizon background-consciousness model slot."""
    return str(os.environ.get("OUROBOROS_MODEL_CONSCIOUSNESS", "") or "").strip() or _main_model()


def get_deep_self_review_model() -> str:
    """Return the configured deep self-review model slot."""
    return (str(os.environ.get("OUROBOROS_MODEL_DEEP_SELF_REVIEW", "") or "").strip()
            or str(SETTINGS_DEFAULTS["OUROBOROS_MODEL_DEEP_SELF_REVIEW"]))
