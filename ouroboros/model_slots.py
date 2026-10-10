"""Ouroboros — model slot resolution.

The Main/Heavy/Light/Vision/Consciousness/deep-review slots and the ordered
cross-model fallback chain, resolved from the environment with the shipped
defaults as the floor, plus the rename-alias migration that keeps a slot the
owner customized under its former key. Imported by ``provider_models`` as well
as by ``config``, which is why it holds no settings-file knowledge.
"""

from __future__ import annotations

import dataclasses
import copy
import json

from ouroboros.settings_defaults import SETTINGS_DEFAULTS
from ouroboros.settings_integrity import runtime_setting


MODEL_ACCOUNTS_KEY = "OUROBOROS_MODEL_ACCOUNTS"
MODEL_CONTEXT_WINDOWS_KEY = "OUROBOROS_MODEL_CONTEXT_WINDOWS"
PROCESSING_PREFERENCE_KEY = "OUROBOROS_PROCESSING_PREFERENCE"
MODEL_PROCESSING_PREFERENCES_KEY = "OUROBOROS_MODEL_PROCESSING_PREFERENCES"
PROCESSING_PREFERENCES = ("standard", "fast", "economy")
MODEL_ROLE_SETTINGS = {
    "main": "OUROBOROS_MODEL",
    "light": "OUROBOROS_MODEL_LIGHT",
    "vision": "OUROBOROS_MODEL_VISION",
    "consciousness": "OUROBOROS_MODEL_CONSCIOUSNESS",
    "deep_review": "OUROBOROS_MODEL_DEEP_SELF_REVIEW",
    "websearch": "OUROBOROS_WEBSEARCH_MODEL",
    "fallback": "OUROBOROS_MODEL_FALLBACKS",
}


def normalize_processing_preference(value: object) -> str:
    """One service preference vocabulary; empty preserves native/legacy defaults."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("Processing preference must be a string")
    result = value.strip().lower()
    if result and result not in PROCESSING_PREFERENCES:
        raise ValueError("Processing preference must be standard, fast, economy, or empty to inherit")
    return result


def normalize_model_role_options(key: str, raw: object) -> tuple[dict, str]:
    """Validate role-owned options, never infer roles from model IDs.

    Empty account means Auto; zero/absent window means Auto, NOT a capacity.
    Empty processing means inherit; explicit standard is an ordinary service request.
    Fallback arrays retain order, matching the existing ordered model chain.
    Invalid pins must never silently degrade into automatic account selection.
    """
    if key not in (MODEL_ACCOUNTS_KEY, MODEL_CONTEXT_WINDOWS_KEY, MODEL_PROCESSING_PREFERENCES_KEY):
        raise ValueError(f"Unknown model role option: {key}")
    if raw is None or raw == "":
        raw = {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{key} must be a JSON object") from exc
    if not isinstance(raw, dict) or set(raw) - set(MODEL_ROLE_SETTINGS):
        raise ValueError(f"{key} must contain only known model roles")
    def value(item: object) -> str | int:
        if key == MODEL_PROCESSING_PREFERENCES_KEY:
            return normalize_processing_preference(item)
        if key == MODEL_ACCOUNTS_KEY:
            if not isinstance(item, str):
                raise ValueError(f"{key}: an account must be a profile name or an empty Auto value")
            return item.strip()
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"{key}: a context window must be a nonnegative integer")
        return item
    parsed = {}
    for role, option in raw.items():
        if role == "fallback":
            if not isinstance(option, list):
                raise ValueError(f"{key}: fallback must be an ordered array")
            parsed[role] = [value(item) for item in option]
        else:
            parsed[role] = value(option)
    return parsed, json.dumps(parsed, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def model_role_option(key: str, role: str, *, settings: dict | None = None) -> str | int:
    """Read one explicit role (``fallback:<index>`` for a chain entry).

    Unlabelled calls use Auto, not Main inferred by string equality. Reviewers
    and configured agents carry their own existing route credential field.
    """
    default = 0 if key == MODEL_CONTEXT_WINDOWS_KEY else ""
    raw = (settings or {}).get(key, "") if settings is not None else runtime_setting(key, "")
    options, _canonical = normalize_model_role_options(key, raw)
    family, separator, position = role.partition(":")
    if family == "fallback" and separator:
        try:
            index = int(position)
        except ValueError:
            raise ValueError("Fallback model role requires its ordered index") from None
        rows = options.get("fallback", [])
        return rows[index] if 0 <= index < len(rows) else default
    return options.get(role, default) if role != "fallback" else default


def resolve_processing_preference(role: str = "", *, override: str | None = None,
                                 settings: dict | None = None) -> str:
    """Resolve a captured call once: explicit call, authored role, then global.

    A call's explicit empty value preserves its captured legacy behavior. Authored
    empty role/actor values instead inherit before that call is captured. Native
    exact overrides are applied at the transport boundary, outside this preference.
    """
    if override is not None:
        return normalize_processing_preference(override)
    role_value = model_role_option(MODEL_PROCESSING_PREFERENCES_KEY, role, settings=settings)
    global_value = (settings or {}).get(PROCESSING_PREFERENCE_KEY, "") if settings is not None else runtime_setting(PROCESSING_PREFERENCE_KEY, "")
    return normalize_processing_preference(role_value or global_value)


def task_model_binding(task: dict, *, context_fit_plan: object = None,
                       overrides: dict | None = None) -> tuple[str, str | None]:
    """Bind an acting call to its explicit role, active fit plan or frozen actor.

    Callers may pass their live owner's overrides; this pure projection never
    captures another task's ambient wait or turns an observed Auto account into a pin.
    """
    metadata = task.get("task_metadata", task.get("metadata", {}))
    metadata = metadata if isinstance(metadata, dict) else {}
    actor = task.get("configured_subagent", metadata.get("configured_subagent", {}))
    actor = actor if isinstance(actor, dict) else {}
    route = actor.get("route") or {}
    actor_id = str(actor.get("selected_subagent_id") or "")
    actor_role = f"subagent:{actor_id}" if actor_id and route.get("kind") == "api_model" else "main"
    role = str(task.get("model_role") or getattr(context_fit_plan, "model_role", "")
               or metadata.get("model_role") or actor_role)
    pin = task.get("credential_profile_id")
    if pin is None and role == actor_role and actor_role != "main":
        pin = str(route.get("credential_profile_id") or "")
    pin = (overrides or {}).get(role, {}).get("model_account_override", pin)
    return role, pin


def route_binding(model: str, use_local: bool, role: str, *,
                  overrides: dict | None = None) -> tuple[str, bool, str]:
    """The complete identity a send on ``role`` uses: model, locality and account.

    A live owner choice for the role (a wait-card switch) replaces the configured
    route exactly as the send applies it; the account is the configured pin or
    empty Auto, never an observed Auto account, and routes without accounts have
    none. Equal model strings with different accounts are different routes.
    """
    from ouroboros.provider_models import provider_for_model

    chosen = (overrides or {}).get(role) or {}
    model, use_local = str(chosen.get("model", model) or ""), bool(chosen.get("use_local", use_local))
    account = chosen.get("model_account_override")
    if account is None:
        account = model_role_option(MODEL_ACCOUNTS_KEY, role) if role else ""
    if use_local or provider_for_model(model) != "claudexor":
        account = ""
    return model, use_local, str(account or "").strip()


def task_processing_preference(task: dict, *, model_role: str = "",
                              override: str | None = None) -> str:
    """Capture the actor's intent; an unconfigured fallback preserves that intent."""
    if override is not None:
        return resolve_processing_preference(override=override)
    metadata = task.get("task_metadata", task.get("metadata", {}))
    metadata = metadata if isinstance(metadata, dict) else {}
    actor = task.get("configured_subagent", metadata.get("configured_subagent", {}))
    actor = actor if isinstance(actor, dict) else {}
    actor_id = str(actor.get("selected_subagent_id") or "")
    role = model_role or task_model_binding(task)[0]
    if role.startswith("fallback:"):
        authored = model_role_option(MODEL_PROCESSING_PREFERENCES_KEY, role)
        if authored:
            return resolve_processing_preference(override=str(authored))
        role = (f"subagent:{actor_id}" if actor_id and (actor.get("route") or {}).get("kind") == "api_model"
                else "main")
    if actor_id and role == f"subagent:{actor_id}":
        # Missing in a historical snapshot means captured legacy behavior.
        return resolve_processing_preference(override=actor.get("processing_preference") or "")
    return resolve_processing_preference(role)


def apply_model_role_override(settings: dict, *, role: str, model: str,
                              credential_profile_id: str, use_local: bool) -> dict:
    """Apply explicit wait-card persistence to one role; never infer it from a model.

    The gateway owns locking and writes. A ``reviewer:<id>`` or ``subagent:<id>``
    role names ONE catalog row by its stored id (a review pool seat's identity is
    its row), and the override changes that row's route — its mark, delivery,
    effort and every other field stay as the owner saved them.
    """
    from ouroboros.provider_models import provider_for_model, parse_claudexor_model

    if not isinstance(model, str) or not model.strip():
        raise ValueError("A model is required")
    if not isinstance(credential_profile_id, str) or not isinstance(use_local, bool):
        raise ValueError("Account and local routing must be explicit")
    model, pin = model.strip(), credential_profile_id.strip()
    subscription = not use_local and provider_for_model(model) == "claudexor"
    if subscription:
        parse_claudexor_model(model)
    if pin and not subscription:
        raise ValueError("An account pin requires a managed model source")
    result = copy.deepcopy(settings)
    family, _, identity = role.partition(":")
    if family == "reviewer" and identity == "main":
        # ``/review`` runs on the direct Main row when the call names no reviewer
        # (deep_self_review.main_review_row, decision 3A): there is no catalog row
        # behind that seat, so its wait card persists as Main's own role.
        family, identity = "main", ""
    if family in MODEL_ROLE_SETTINGS:
        key = MODEL_ROLE_SETTINGS[family]
        accounts, _ = normalize_model_role_options(MODEL_ACCOUNTS_KEY, result.get(MODEL_ACCOUNTS_KEY))
        if family == "fallback":
            values = _parse_model_list(result.get(key, ""))
            if not identity.isdigit() or int(identity) >= len(values):
                raise ValueError("The selected fallback role no longer exists")
            shared_local = result.get("USE_LOCAL_FALLBACK", SETTINGS_DEFAULTS["USE_LOCAL_FALLBACK"])
            if use_local != (shared_local is True or shared_local == "true"):
                raise ValueError("Local applies to all fallbacks in Settings. Change it only for this task, or edit Models.")
            index = int(identity)
            values[index] = model
            pins = list(accounts.get("fallback", []))
            pins.extend([""] * max(0, len(values) - len(pins)))
            pins[index] = pin
            result[key], accounts[family] = ", ".join(values), pins
        else:
            if identity:
                raise ValueError("Unknown model role")
            result[key], accounts[family] = model, pin
        local_key = f"USE_LOCAL_{family.upper()}"
        if local_key in SETTINGS_DEFAULTS and family != "fallback":
            result[local_key] = use_local
        elif local_key not in SETTINGS_DEFAULTS and use_local:
            result[key] = model if model.endswith(" (local)") else f"{model} (local)"
        result[MODEL_ACCOUNTS_KEY] = normalize_model_role_options(MODEL_ACCOUNTS_KEY, accounts)[1]
        return result
    from ouroboros.configured_subagents import normalize_configured_subagents, configured_subagents_dict

    if family not in ("reviewer", "subagent"):
        raise ValueError("Unknown waiting model role")
    routed_model = model if not use_local or model.endswith(" (local)") else f"{model} (local)"
    roster = configured_subagents_dict(normalize_configured_subagents(result.get("OUROBOROS_SUBAGENTS"))[0])
    actor = next((item for item in roster["items"] if item["subagent_id"] == identity), None)
    if actor is None:
        if family == "reviewer":
            from ouroboros.reviewer_slot_config import review_pool_rows

            handles = ", ".join(row.slot_id for row in review_pool_rows(result)) or "empty"
            raise ValueError(f"The selected reviewer no longer exists (the review pool is: {handles})")
        raise ValueError("The selected task agent no longer exists")
    actor["route"] = {"kind": "api_model", "target_id": routed_model, "credential_profile_id": pin}
    actor.pop("access", None)  # Native/API rows have no session access profile.
    result["OUROBOROS_SUBAGENTS"] = normalize_configured_subagents(roster)[1]
    return result


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
        str(runtime_setting("OUROBOROS_MODEL", "") or "").strip()
        or str(SETTINGS_DEFAULTS["OUROBOROS_MODEL"])
    )


def get_light_model() -> str:
    """Light slot; empty falls back to Main (heavy/consciousness stay empty->main)."""
    return str(runtime_setting("OUROBOROS_MODEL_LIGHT", "") or "").strip() or _main_model()


def get_heavy_model() -> str:
    """Return the heavy (strong acting/coding) lane slot; empty falls back to
    OUROBOROS_MODEL. Renamed from the legacy code slot."""
    return str(runtime_setting("OUROBOROS_MODEL_HEAVY", "") or "").strip() or _main_model()


def get_vision_model() -> str:
    """Return the vision/caption model slot; empty falls back to OUROBOROS_MODEL."""
    return str(runtime_setting("OUROBOROS_MODEL_VISION", "") or "").strip() or _main_model()


_SLOT_LOCAL_FLAGS = {"main": "USE_LOCAL_MAIN", "light": "USE_LOCAL_LIGHT", "fallback": "USE_LOCAL_FALLBACK"}


def local_lane_label(model: object, use_local: bool) -> str:
    """``model`` labelled ``" (local)"`` when it runs on our local lane (``provider_for_model`` reads the label)."""
    text = str(model or "").strip()
    return f"{text} (local)" if text and use_local and not text.endswith(" (local)") else text


def slot_lane_label(role: str, model: object) -> str:
    """A configured slot's model as it routes: labelled local when that slot runs on our local lane.

    The local flag is a separate setting, so a bare id alone does not say which lane
    carries it. An empty vision or light slot inherits Main and Main's flag; a vision
    slot set to a local model is stored with its label already.
    """
    inherits = {"vision": "OUROBOROS_MODEL_VISION", "light": "OUROBOROS_MODEL_LIGHT"}.get(role, "")
    if inherits and not str(runtime_setting(inherits, "") or "").strip():
        role = "main"
    flag = str(runtime_setting(_SLOT_LOCAL_FLAGS.get(role, ""), "") or "") if role in _SLOT_LOCAL_FLAGS else ""
    return local_lane_label(model, flag.strip().lower() in {"1", "true", "yes", "on"})


def get_image_input_mode() -> str:
    raw = str(runtime_setting("OUROBOROS_IMAGE_INPUT_MODE", SETTINGS_DEFAULTS["OUROBOROS_IMAGE_INPUT_MODE"]) or "").strip().lower()
    return raw if raw in {"auto", "caption", "inline", "off"} else "auto"


def parse_fallback_chain() -> list[str]:
    """Parse the raw ordered cross-model fallback chain — SSOT for every consumer
    (resilience walk, pricing categorization, credentialed-model resolution).

    Reads OUROBOROS_MODEL_FALLBACKS, then the legacy singular OUROBOROS_MODEL_FALLBACK
    (env-only back-compat). No dedup, no active-model drop, and NO SETTINGS_DEFAULTS
    injection: an EXPLICITLY empty Fallbacks slot means "no cross-model fallback". The
    shipped default reaches a default install through apply_settings_to_env."""
    raw = (
        str(runtime_setting("OUROBOROS_MODEL_FALLBACKS", "") or "").strip()
        or str(runtime_setting("OUROBOROS_MODEL_FALLBACK", "") or "").strip()
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
    return str(runtime_setting("OUROBOROS_MODEL_CONSCIOUSNESS", "") or "").strip() or _main_model()
