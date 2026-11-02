"""Provider-specific model ID helpers, direct-provider defaults, and the
provider registry (SSOT for prefix→provider→credentials knowledge that was
previously duplicated across llm.py, pricing.py, agent_task_pipeline.py and
deep_self_review.py)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ouroboros.model_slots import ResolvedModelTarget, parse_fallback_chain
from ouroboros.settings_defaults import OPENROUTER_DEFAULTS, OPENROUTER_REVIEW_DEFAULTS, SETTINGS_DEFAULTS  # noqa: F401
from ouroboros.settings_integrity import runtime_setting

# MiniMax exposes the same OpenAI-compatible API on two regional hosts. Keep the
# mapping centralized so transport, capability evidence, and settings diagnostics
# fingerprint the exact endpoint selected by the owner.
MINIMAX_REGION_ENDPOINTS: dict[str, str] = {
    "global_en": "https://api.minimax.io/v1",
    "cn_zh": "https://api.minimaxi.com/v1",
}
MINIMAX_DEFAULT_REGION = "global_en"


def resolve_minimax_base_url(region: str = "") -> str:
    """Return the configured MiniMax OpenAI-compatible endpoint."""
    selected = str(region or "").strip().lower() or MINIMAX_DEFAULT_REGION
    return MINIMAX_REGION_ENDPOINTS.get(selected, MINIMAX_REGION_ENDPOINTS[MINIMAX_DEFAULT_REGION])


# DeepSeek serves one official OpenAI-compatible endpoint (no regions, no
# owner-configurable base URL — a proxy/mirror setup belongs to the generic
# openai-compatible slot). Kept as a module constant so transport and the
# provider test resolve the same host; the reviewer-window/base-url fingerprint
# maps deliberately have NO deepseek branch (both default to "" consistently —
# the fingerprint is already unique per provider+model).
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# Z.ai (Zhipu / GLM) serves one OpenAI-compatible API surface on two plans that
# share the same key: pay-as-you-go and the subscription Coding Plan. The plan
# selects the endpoint (analogous to MiniMax regions); the PAYG endpoint is the
# default because the Coding Plan endpoint is officially intended for supported
# coding tools only.
ZAI_PLAN_ENDPOINTS: dict[str, str] = {
    "payg": "https://api.z.ai/api/paas/v4",
    "coding": "https://api.z.ai/api/coding/paas/v4",
}
ZAI_DEFAULT_PLAN = "payg"


def resolve_zai_base_url(plan: str = "") -> str:
    """Return the configured Z.ai OpenAI-compatible endpoint for the plan."""
    selected = str(plan or "").strip().lower() or ZAI_DEFAULT_PLAN
    return ZAI_PLAN_ENDPOINTS.get(selected, ZAI_PLAN_ENDPOINTS[ZAI_DEFAULT_PLAN])


# DeepSeek's Chat Completions ``reasoning_effort`` enum is low/high/max
# (medium/xhigh are documented aliases of high) and thinking is switched off by
# ``thinking.type=disabled``, not by an effort value. This is the wire dialect
# of one provider, projected at the physical-send boundary; the canonical
# Ouroboros effort scale stays the SSOT everywhere else. Not a model or pricing
# table and never an admission gate.
DEEPSEEK_REASONING_EFFORT_ALIASES = {
    "minimal": "low",
    "medium": "high",
    "xhigh": "high",
    "ultra": "max",
}


def normalize_deepseek_reasoning_effort(value: str) -> str:
    """Project one canonical effort tier onto DeepSeek's Chat wire enum."""
    normalized = str(value or "").strip().lower()
    return DEEPSEEK_REASONING_EFFORT_ALIASES.get(normalized, normalized)


# Z.ai (GLM) serves the same Chat Completions ``reasoning_effort`` shape but a
# DIFFERENT enum mapping than DeepSeek — do not reuse the DeepSeek table. The
# provider has exactly three tiers (low | high | max); ``medium`` does not
# exist, thinking cannot be disabled (``thinking={"type":"disabled"}`` answers
# 400 code 1210 "please use low, high or max" on PAYG), and an ABSENT
# parameter is served at MAX — so a silently dropped tier means every call
# runs (and bills) at max. Projection of the canonical scale, measured live
# 2026-09-21 on the Coding Plan endpoint: none/minimal/low -> low,
# medium/high -> high, xhigh/ultra -> max.
ZAI_REASONING_EFFORT_ALIASES = {
    "none": "low",
    "minimal": "low",
    "low": "low",
    "medium": "high",
    "high": "high",
    "xhigh": "max",
    "max": "max",
    "ultra": "max",
}


def normalize_zai_reasoning_effort(value: str) -> str:
    """Project one canonical effort tier onto Z.ai's Chat wire enum.

    Every canonical tier maps to a concrete provider tier; unlike DeepSeek
    there is no "off" arm — GLM reasoning cannot be disabled, so an unmapped
    value still resolves to a tier rather than being dropped (a dropped tier
    is served at max).
    """
    normalized = str(value or "").strip().lower()
    return ZAI_REASONING_EFFORT_ALIASES.get(normalized, "low")


@dataclass(frozen=True)
class KindSpec:
    """One API connection kind: its label, its fields in order as ``(row_field,
    flat_setting_key, secret)``, and the field groups that make a connection of it
    usable (any ONE group whose every field is set)."""

    label: str
    fields: tuple[tuple[str, str, bool], ...]
    ready: tuple[tuple[str, ...], ...] = (("api_key",),)

    def flat_key(self, row_field: str) -> str:
        return next((flat for field, flat, _secret in self.fields if field == row_field), "")


# The kind table, SSOT for every credential table below. A kind's BUILT-IN connection
# reads these flat settings; a named connection (``OUROBOROS_CONNECTIONS``,
# ``provider_connections``) carries the same fields as one row. Order is the
# owner-facing order (OpenRouter first: an unprefixed model id routes through it).
PROVIDER_KINDS: dict[str, KindSpec] = {
    "openrouter": KindSpec("OpenRouter", (("api_key", "OPENROUTER_API_KEY", True),)),
    "openai": KindSpec("OpenAI", (("api_key", "OPENAI_API_KEY", True),)),
    "anthropic": KindSpec("Anthropic", (("api_key", "ANTHROPIC_API_KEY", True),)),
    "deepseek": KindSpec("DeepSeek", (("api_key", "DEEPSEEK_API_KEY", True),)),
    "zai": KindSpec("Z.ai (GLM)", (("api_key", "ZAI_API_KEY", True), ("plan", "ZAI_PLAN", False))),
    "minimax": KindSpec("MiniMax", (("api_key", "MINIMAX_API_KEY", True), ("region", "MINIMAX_REGION", False))),
    "cloudru": KindSpec("Cloud.ru Foundation Models", (
        ("api_key", "CLOUDRU_FOUNDATION_MODELS_API_KEY", True),
        ("base_url", "CLOUDRU_FOUNDATION_MODELS_BASE_URL", False))),
    "gigachat": KindSpec("GigaChat", (
        ("credentials", "GIGACHAT_CREDENTIALS", True), ("user", "GIGACHAT_USER", False),
        ("password", "GIGACHAT_PASSWORD", True), ("base_url", "GIGACHAT_BASE_URL", False),
        ("scope", "GIGACHAT_SCOPE", False), ("verify_ssl_certs", "GIGACHAT_VERIFY_SSL_CERTS", False)),
        ready=(("credentials",), ("user", "password"))),
    # An empty key is a keyless endpoint; the endpoint itself is what configures it.
    "openai-compatible": KindSpec("OpenAI-compatible endpoint", (
        ("api_key", "OPENAI_COMPATIBLE_API_KEY", True),
        ("base_url", "OPENAI_COMPATIBLE_BASE_URL", False)), ready=(("base_url",),)),
}
# The extra connections, one JSON list (secrets masked by row identity, never exported to env).
CONNECTIONS_KEY = "OUROBOROS_CONNECTIONS"
# The built-in OpenAI-compatible connection also honours the legacy pair (``compatible_endpoint``).
_LEGACY_COMPATIBLE_KEYS = ("OPENAI_API_KEY", "OPENAI_BASE_URL")
# Every kind a route head may name: the API kinds plus the owned subscription engine.
ROUTE_KINDS: tuple[str, ...] = ("claudexor", *PROVIDER_KINDS)

# Direct-provider prefix → canonical provider name. Un-prefixed models route
# through OpenRouter; prefixes are disjoint.
PROVIDER_PREFIXES: tuple[tuple[str, str], ...] = tuple((f"{kind}::", kind) for kind in ROUTE_KINDS)

# Primary credential setting per provider (single-key providers).
PROVIDER_ENV_KEYS: dict[str, str] = {
    kind: spec.fields[0][1] for kind, spec in PROVIDER_KINDS.items() if spec.ready == (("api_key",),)
}

# Settings whose content-bound grant lets an external skill bypass the core LLM
# transport and incur model spend.  This belongs with the provider/credential
# registry, not in the frozen PluginAPI contract.
MODEL_PROVIDER_CREDENTIAL_KEYS: frozenset[str] = frozenset({
    *(flat for spec in PROVIDER_KINDS.values() for _field, flat, secret in spec.fields if secret),
    CONNECTIONS_KEY,
})

# EVERY flat settings key the built-in connection of a kind reads, GROUPED so a credential
# and the fields it is useless without travel together or not at all (GigaChat needs
# CREDENTIALS *or* USER+PASSWORD plus its endpoint/scope; Cloud.ru's key is meaningless
# against the wrong base_url; the openai-compatible lane legitimately falls back to the
# legacy OPENAI_* pair).  Deriving a per-run credential set from anything but this table is
# guessing: `anthropic/claude-sonnet-4.6` is an OPENROUTER model id, only `anthropic::…` is direct.
PROVIDER_CREDENTIAL_GROUPS: dict[str, tuple[str, ...]] = {
    **{kind: tuple(flat for _field, flat, _secret in spec.fields)
       + (_LEGACY_COMPATIBLE_KEYS if kind == "openai-compatible" else ())
       for kind, spec in PROVIDER_KINDS.items()},
    "local": (),
    # The owned engine holds credentials; no control token reaches model settings/env.
    "claudexor": (),
}

# Active settings keys that hold a ROUTED model identity (prefix -> provider via
# provider_for_model). Heavy and the retired review scalars are migration/history
# inputs, not live route selectors; the active set keeps consumers (including
# Provider Test) from accidentally resurrecting them.
# Superset of the live slots; a key absent from settings still declares whatever
# ``config.SETTINGS_DEFAULTS`` will hand the runtime, which is why declared_model_settings()
# fills the defaults in rather than treating "unset" as "unused".
ACTIVE_MODEL_SETTING_KEYS: tuple[str, ...] = (
    "OUROBOROS_MODEL", "OUROBOROS_MODEL_LIGHT",
    "OUROBOROS_MODEL_VISION", "OUROBOROS_MODEL_CONSCIOUSNESS",
    "OUROBOROS_MODEL_FALLBACKS", "OUROBOROS_MODEL_FALLBACK",
    "OUROBOROS_WEBSEARCH_MODEL", "OUROBOROS_MODEL_SAFETY",
)
LEGACY_MODEL_SETTING_KEYS: tuple[str, ...] = ("OUROBOROS_MODEL_HEAVY",)
# Compatibility import name. Its meaning is now explicitly the active set.
MODEL_SETTING_KEYS = ACTIVE_MODEL_SETTING_KEYS

# Settings keys whose value is a Claude Agent SDK / Claude Code model NAME (``opus[1m]``),
# NOT a routed model identity: they carry no provider prefix, so provider_for_model would
# mis-route them to OpenRouter.  Their transport is the Anthropic SDK subprocess, which
# authenticates with ANTHROPIC_API_KEY, so a non-empty value DECLARES the anthropic provider.


_CONNECTION_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}")
# Reserved for a kind's built-in connection: ``K@default::m`` is ``K::m``.
BUILTIN_CONNECTION = "default"


@dataclass(frozen=True)
class RouteParts:
    """One model route string read by the route grammar (``parse_model_route``)."""

    kind: str
    connection: str
    model: str
    local: bool = False
    valid: bool = True


def parse_model_route(value: Any) -> RouteParts:
    """The ONE route grammar; pure, it reads no settings.

    A bare id is OpenRouter's, ``kind::model`` a kind's built-in connection,
    ``kind@name::model`` a named API connection (``@default`` is the built-in one),
    ``claudexor::source=model`` the subscription engine, a trailing `` (local)`` the
    local lane. A ``claudexor@`` head (Claudexor keeps its own account axis) or a
    malformed name is ``valid=False``; an unknown head stays an opaque OpenRouter id.
    The model part is returned as written."""
    text = str(value or "").strip()
    if text.endswith(" (local)"):
        return RouteParts("local", "", text[:-len(" (local)")], local=True)
    head, separator, model = text.partition("::")
    kind, at, name = head.partition("@")
    if not separator or kind not in ROUTE_KINDS:
        return RouteParts("openrouter", "", text)
    if not at:
        return RouteParts(kind, "", model)
    if kind not in PROVIDER_KINDS or not _CONNECTION_NAME.fullmatch(name):
        return RouteParts(kind, name, model, valid=False)
    return RouteParts(kind, "" if name == BUILTIN_CONNECTION else name, model)


def connection_id(kind: str, name: str = "") -> str:
    """The picker source id of a connection: ``kind`` for the built-in, ``kind@name`` otherwise."""
    return f"{kind}@{name}" if name else kind


def provider_for_model(model: str) -> str:
    """Return the execution provider for a model id (``local`` for local lanes)."""
    return parse_model_route(model).kind


# Physical per-request ceilings on tool schemas, by execution provider; a provider
# absent here has no known ceiling. OpenAI's own API refuses a longer `tools` array
# (400 array_above_max_length); the same models through OpenRouter accepted 129
# (provider canary, CI run 37204372307, 2026-10-04).
PROVIDER_TOOL_SCHEMA_LIMITS: dict[str, int] = {"openai": 128}


def tool_schema_limit(model: str, *, use_local: bool = False) -> int | None:
    """The route's ceiling on tool schemas in one request, or None when none is known."""
    return None if use_local else PROVIDER_TOOL_SCHEMA_LIMITS.get(provider_for_model(model))


def parse_claudexor_model(model: str) -> tuple[str, str]:
    """Split the model transport's opaque source and model, never an account pin."""
    if not str(model).startswith("claudexor::"):
        raise ValueError("Not a Claudexor model identity")
    source, separator, native_model = str(model)[len("claudexor::"):].partition("=")
    if not separator or not source.strip() or not native_model.strip():
        raise ValueError("Claudexor models use claudexor::<source>=<model>")
    return source.strip(), native_model.strip()


def resolve_model_target(
    model: str,
    *,
    effort: str = "",
    credential_ref: str = "",
    context_window: int = 0,
) -> ResolvedModelTarget:
    """Construct the ABI-4 typed target at an EXISTING resolution seam.

    Wraps what the resolution already computed (reuse-first): the transport
    lane comes from ``provider_for_model``, and the optional facts keep their
    typed sentinels unless the calling seam genuinely resolved them. No
    parallel resolver, no pricing, no window probing — a context window is
    Capability Evidence's fact (0 = unknown, fail-open).
    """
    model_id = str(model or "").strip()
    return ResolvedModelTarget(
        model_id=model_id,
        provider_route=provider_for_model(model_id),
        credential_ref=str(credential_ref or "").strip(),
        effort=str(effort or "").strip(),
        context_window=max(0, int(context_window or 0)),
    )


def fallback_candidate_targets(active_model: str = "", *, preserve_slots: bool = False) -> tuple[ResolvedModelTarget, ...]:
    """The cross-model fallback candidate ladder as typed targets (ABI-4).

    Same membership and order as ``model_slots.get_fallback_models`` — a typed
    view over the ONE chain SSOT, not a second resolver. ``preserve_slots`` keeps
    repeated model ids so callers can compare their distinct configured account bindings.
    Effort stays the ""
    sentinel: the ladder resolves destinations, the dispatching round owns the
    active effort. ``provider_route`` stays the ``""`` sentinel DELIBERATELY:
    the chain's local-vs-remote dispatch lane is the loop's single global
    ``USE_LOCAL_FALLBACK`` flag (the pre-existing contract, byte-identical
    through the ABI-4 sweep), so a per-candidate route here would be a
    fabricated fact no dispatcher consumes.
    """
    from ouroboros.model_slots import get_fallback_models, parse_fallback_chain

    return tuple(
        ResolvedModelTarget(model_id=model, provider_route="")
        for model in (parse_fallback_chain() if preserve_slots else get_fallback_models(active_model))
    )


# The dedicated compatible pair and the legacy pair the built-in connection falls back to.
COMPATIBLE_LEGACY_PAIR = {"OPENAI_COMPATIBLE_API_KEY": "OPENAI_API_KEY", "OPENAI_COMPATIBLE_BASE_URL": "OPENAI_BASE_URL"}


def compatible_endpoint(value: Callable[[str], str]) -> tuple[str, str]:
    """``(api_key, base_url)`` of the OpenAI-compatible lane, one pair on every path.

    A dedicated ``OPENAI_COMPATIBLE_BASE_URL`` travels only with its own key (blank = an
    endpoint without auth); without it the legacy ``OPENAI_BASE_URL`` travels only with
    ``OPENAI_API_KEY``. Neither pair borrows the other's key. ``value`` reads one stripped setting."""
    if base := value("OPENAI_COMPATIBLE_BASE_URL"):
        return value("OPENAI_COMPATIBLE_API_KEY"), base
    return value("OPENAI_API_KEY"), value("OPENAI_BASE_URL")


def compatible_endpoint_configured(value: Callable[[str], str]) -> bool:
    """Readiness of that pair: the dedicated endpoint (its key optional), or the legacy endpoint
    with its key. A dedicated key without its endpoint configures nothing."""
    key, base = compatible_endpoint(value)
    return bool(base and (key or value("OPENAI_COMPATIBLE_BASE_URL")))


def _route_has_credentials(route: RouteParts, settings: Mapping[str, Any] | None) -> bool:
    """Whether ``route``'s connection can serve: enabled, with its credential/endpoint."""
    if route.kind == "claudexor":
        return route.valid  # A selected engine route needs no API key in Ouroboros.
    if route.kind == "local":
        return settings is None or bool(str(settings.get("LOCAL_MODEL_SOURCE") or "").strip())
    from ouroboros.provider_connections import connection_configured, settings_view

    return connection_configured(route, *settings_view(settings))


def provider_has_credentials(provider: str) -> bool:
    """Whether a kind's built-in connection is configured (unknown kinds read OpenRouter's);
    a managed engine's live readiness is separate."""
    kind = provider if provider in ROUTE_KINDS or provider == "local" else "openrouter"
    return _route_has_credentials(RouteParts(kind, "", ""), None)


def provider_has_credentials_in_settings(provider: str, settings: dict) -> bool:
    """Mapping-based twin used by pure config/default compilers (no ambient env)."""
    kind = provider if provider in ROUTE_KINDS or provider == "local" else "openrouter"
    return _route_has_credentials(RouteParts(kind, "", ""), settings or {})


def model_has_credentials_in_settings(model: str, settings: dict) -> bool:
    return _route_has_credentials(parse_model_route(model), settings or {})


def model_has_credentials(model: str) -> bool:
    """Return True when the connection the model names has usable credentials configured."""
    return _route_has_credentials(parse_model_route(model), None)


def any_api_connection_configured(settings: Mapping[str, Any] | None = None) -> bool:
    """Whether any API connection, built-in or named, is enabled with its credential/endpoint."""
    from ouroboros.provider_connections import configured_connections

    return bool(configured_connections(settings))


def local_only_review_route_env() -> bool:
    """Whether review slots must inherit the configured local Main route."""
    local_main = str(runtime_setting("USE_LOCAL_MAIN", "") or "").strip().lower()
    if local_main not in {"1", "true", "yes", "on"}:
        return False
    return not any_api_connection_configured()


def review_model_uses_local(model: str) -> bool:
    """Return the transport route for a resolved review slot."""
    return local_only_review_route_env() or provider_for_model(str(model or "").strip()) == "local"


def resolve_credentialed_model(default_model: str) -> str:
    """Return ``default_model`` if its provider is credentialed, else the first
    configured model slot whose provider has credentials (light → fallback →
    main). Falls back to ``default_model`` when nothing is credentialed
    so callers surface the original provider error rather than a silent swap."""
    if model_has_credentials(default_model):
        return default_model
    # LIGHT/MAIN are single-model slots; FALLBACKS is a comma chain expanded via the
    # shared SSOT parser (the live Fallbacks slot, route_choice)
    # instead of testing the whole comma-string as one broken model id. Empty Light
    # (default -> Main) simply contributes nothing here.
    candidates: list[str] = []
    light = str(runtime_setting("OUROBOROS_MODEL_LIGHT", "") or "").strip()
    if light:
        candidates.append(light)
    candidates.extend(parse_fallback_chain())
    for env_name in ("OUROBOROS_MODEL",):
        raw = str(runtime_setting(env_name, "") or "").strip()
        if raw:
            candidates.append(raw)
    for candidate in candidates:
        if model_has_credentials(candidate):
            return candidate
    return default_model


def declared_model_settings(
    settings: dict,
    *,
    include_claude_sdk_defaults: bool = True,
) -> dict[str, str]:
    """Return the model slots a settings mapping DECLARES, with runtime defaults filled in.

    An absent or empty slot is not "unused": the server falls back to
    ``config.SETTINGS_DEFAULTS`` for it, so the default's provider is genuinely reachable and
    must be declared.  ``include_claude_sdk_defaults`` is a RETIRED no-op kept for caller
    compatibility: the Claude-SDK transport and its dedicated model slots are gone, so both
    values declare the same set."""
    declared: dict[str, str] = {}
    for key in MODEL_SETTING_KEYS:
        value = str((settings or {}).get(key) or "").strip()
        if not value:
            value = str(SETTINGS_DEFAULTS.get(key) or "").strip()
        if value:
            declared[key] = value
    return declared


def providers_for_declared_models(declared: dict) -> dict[str, list[str]]:
    """Map ``{settings key: model string}`` to ``{provider: sorted model strings}``.

    Comma chains (fallbacks, review triads) are expanded; the Claude-SDK slots resolve to
    ``anthropic`` by transport rather than by prefix."""
    found: dict[str, set] = {}
    for key, raw in (declared or {}).items():
        text = str(raw or "").strip()
        if not text:
            continue
        for part in text.split(","):
            model = part.strip()
            if model:
                found.setdefault(provider_for_model(model), set()).add(model)
    return {provider: sorted(models) for provider, models in sorted(found.items())}


def credential_keys_for_providers(providers) -> tuple[str, ...]:
    """Return the ordered, de-duplicated credential keys a provider set needs."""
    keys: list[str] = []
    for provider in providers:
        group = PROVIDER_CREDENTIAL_GROUPS.get(str(provider))
        if group is None:
            # Unknown provider: fail OPEN with its primary key rather than silently
            # handing a run no credential at all.
            group = tuple(filter(None, (PROVIDER_ENV_KEYS.get(str(provider), ""),)))
        for key in group:
            if key not in keys:
                keys.append(key)
    return tuple(keys)


ALL_PROVIDER_CREDENTIAL_KEYS: frozenset[str] = frozenset(
    key for group in PROVIDER_CREDENTIAL_GROUPS.values() for key in group
)
# Every setting a route's credential or endpoint is read from: the flat keys of the
# built-in connections (legacy compatible pair included) and the named connections.
ROUTE_CREDENTIAL_SETTING_KEYS: frozenset[str] = ALL_PROVIDER_CREDENTIAL_KEYS | {CONNECTIONS_KEY}


def provider_credential_plan(
    settings: dict,
    *,
    include_claude_sdk_defaults: bool = True,
) -> dict:
    """Derive WHICH provider credentials a settings mapping's declared models actually need.

    Returns ``{declared_model_slots, providers, planned_keys, fail_open}``.  ``fail_open`` is
    the disclosed escape hatch: when nothing resolves (a settings mapping with no model slot
    at all) the plan is the FULL credential universe, because a benchmark that dies on a
    missing key at hour six is worse than one that carries a spare.  The optional
    ``include_claude_sdk_defaults`` switch is RETIRED (no-op): the Claude-SDK transport and
    its dedicated model slots are gone, so both values produce the same plan."""
    declared = declared_model_settings(settings)
    providers = providers_for_declared_models(declared)
    planned = credential_keys_for_providers(providers)
    if any(parse_model_route(model).connection for models in providers.values() for model in models):
        planned += (CONNECTIONS_KEY,)  # a named connection travels as its row, not as flat keys
    fail_open = not planned
    if fail_open:
        planned = tuple(sorted(ALL_PROVIDER_CREDENTIAL_KEYS))
    return {
        "declared_model_slots": declared,
        "providers": providers,
        "planned_keys": sorted(planned),
        "fail_open": fail_open,
    }


OPENAI_DIRECT_DEFAULTS = {
    "main": "openai::gpt-5.6-terra",
    "heavy": "",
    "light": "openai::gpt-5.6-luna",
    "fallback": "openai::gpt-5.6-sol",
    # Deep self-review is a real slot with a SHIPPED default; without a
    # per-provider value a direct-only install keeps an unreachable
    # OpenRouter-form id it has no credential for (v6.82.0). Only providers whose
    # model genuinely carries the >=1M window this review sizes against get one —
    # Cloud.ru and GigaChat are documented BELOW that floor, so filling their slot
    # would advertise a deep review that is doomed to overflow its real route.
    #
    # Plain Sol, the same model the OpenRouter default names. A `-pro` suffix is an
    # OpenRouter routing slug, not an OpenAI model id: live-probed 2026-07-29 against
    # api.openai.com, `gpt-5.6-sol-pro` on /v1/chat/completions -> 404; the pro
    # reasoning mode exists only on /v1/responses as `reasoning.mode="pro"` (200),
    # and passing `reasoning` to /v1/chat/completions -> 400 "Unknown parameter".
    # Every LLM call in llm.py is a chat.completions call, so an owner's pinned
    # `-pro` slug lands here too (deep_self_review.deep_review_route).
    "deep_self_review": "openai::gpt-5.6-sol",
}

CLOUDRU_DIRECT_DEFAULTS = {
    "main": "cloudru::zai-org/GLM-4.7",
    "heavy": "cloudru::zai-org/GLM-4.7",
    "light": "cloudru::zai-org/GLM-4.7",
    "fallback": "cloudru::zai-org/GLM-4.7",
}

GIGACHAT_DIRECT_DEFAULTS = {
    # Ultra is available only to individuals in Freemium. GigaChat 2 Max is
    # available to personal and legal-entity paid plans as well, so it is the
    # strongest current default that does not strand B2B/CORP installs.
    "main": "gigachat::GigaChat-2-Max",
    "heavy": "gigachat::GigaChat-2-Max",
    "light": "gigachat::GigaChat-2-Max",
    "fallback": "gigachat::GigaChat-2-Max",
}

MINIMAX_DIRECT_DEFAULTS = {
    "main": "minimax::MiniMax-M3",
    "heavy": "minimax::MiniMax-M3",
    "light": "minimax::MiniMax-M3.1-Flash",
    "fallback": "minimax::MiniMax-M3.1-Flash",
    # NO deep_self_review default: MiniMax documents M3 as "up to 1M tokens" with a
    # GUARANTEED minimum of 512K (platform.minimax.io, 2026-08). Deep review sizes
    # against a firm 1M window, so the guaranteed floor decides — the slot follows
    # the Cloud.ru/GigaChat clear-instead-of-fill path; owners can opt in manually.
}

ZAI_DIRECT_DEFAULTS = {
    "main": "zai::glm-5.3",
    "heavy": "",
    "light": "zai::glm-5.3-flash",
    # Image input per docs.z.ai/guides/vlm/glm-5.3-flash.md (read 2026-10-08; glm-5.3
    # is text-only). A contributor probe (2026-10-06) covered the Coding Plan endpoint
    # only; pay-as-you-go rests on the docs. A new install's wizard proposes it; a
    # saved Vision choice, empty included, stays as saved.
    "vision": "zai::glm-5.3-flash",
    "fallback": "zai::glm-5.3-flash",
    # No deep_review default: the route publishes no window metadata and no live
    # measurement exists, so the slot follows the MiniMax clear-instead-of-fill path.
}

DEEPSEEK_DIRECT_DEFAULTS = {
    "main": "deepseek::deepseek-v4-pro",
    "heavy": "",
    "light": "deepseek::deepseek-v4-flash",
    "fallback": "deepseek::deepseek-v4-flash",
    # DeepSeek documents a FIRM 1M context on every v4 model (api-docs
    # 2026-08: "1M context, 384K max output" with no guaranteed-minimum
    # caveat), so the slot follows the OpenAI/Anthropic fill pattern rather
    # than the MiniMax clear-instead-of-fill path (whose guaranteed floor was
    # 512K). The route's /models endpoint publishes NO window metadata, so the
    # sizing remains evidence-driven; a missing window measurement does not
    # remove review authority (see ARCHITECTURE §7).
    "deep_self_review": "deepseek::deepseek-v4-pro",
    # No vision default: deepseek-v4-flash-vision-exp is experimental and is
    # for explicit owner selection only.
}

ANTHROPIC_DIRECT_DEFAULTS = {
    "main": "anthropic::claude-opus-5-5",
    "heavy": "",
    "light": "anthropic::claude-sonnet-5",
    "fallback": "anthropic::claude-sonnet-5",
    # Deep self-review is a real slot with a SHIPPED default; without a
    # per-provider value a direct-only install keeps an unreachable
    # OpenRouter-form id it has no credential for (v6.82.0).
    "deep_self_review": "anthropic::claude-opus-5-5",
}

DIRECT_PROVIDER_DEFAULTS = {
    "openai": OPENAI_DIRECT_DEFAULTS,
    "anthropic": ANTHROPIC_DIRECT_DEFAULTS,
    "cloudru": CLOUDRU_DIRECT_DEFAULTS,
    "gigachat": GIGACHAT_DIRECT_DEFAULTS,
    "minimax": MINIMAX_DIRECT_DEFAULTS,
    "deepseek": DEEPSEEK_DIRECT_DEFAULTS,
    "zai": ZAI_DIRECT_DEFAULTS,
}

# Review panels are declared as provider ROLE sequences, one seat per role, then
# compiled against the owner's current provider-prefixed Main/Light values. This
# keeps explicit custom models useful while making the shipped single-provider
# policy obvious: a provider reviews on its strongest Main ONCE (owner Q5=A —
# copies of one model would each pay a run and add no second opinion), and
# MiniMax keeps its mixed panel of two models.
DIRECT_PROVIDER_REVIEW_ROLES = {
    "openai": ("main",),
    "anthropic": ("main",),
    "cloudru": ("main",),
    "gigachat": ("main",),
    "minimax": ("main", "light", "light"),
    "deepseek": ("main",),
    "zai": ("main",),
}

_ANTHROPIC_MODEL_ALIASES = {
    "claude-opus-4.6": "claude-opus-4-6",
    "claude-opus-4.7": "claude-opus-4-7",
    "claude-opus-4.8": "claude-opus-4-8",
    "claude-sonnet-4.6": "claude-sonnet-4-6",
    "claude-opus-5.5": "claude-opus-5-5",
}


def normalize_anthropic_model_id(model_id: str) -> str:
    text = str(model_id or "").strip()
    return _ANTHROPIC_MODEL_ALIASES.get(text, text)


def migrate_model_value(provider: str, value: str) -> str:
    text = str(value or "").strip()
    if provider == "openai":
        if text.startswith("openai/"):
            return f"openai::{text[len('openai/'):]}"
        return text
    if provider == "anthropic":
        if text.startswith("anthropic::"):
            return f"anthropic::{normalize_anthropic_model_id(text[len('anthropic::'):])}"
        if text.startswith("anthropic/"):
            return f"anthropic::{normalize_anthropic_model_id(text[len('anthropic/'):])}"
        return text
    if provider == "cloudru":
        if text.startswith("cloudru::"):
            return text
        if text.startswith("cloudru/"):
            return f"cloudru::{text[len('cloudru/'):]}"
        return text
    if provider == "minimax":
        if text.startswith("minimax::"):
            return text
        if text.startswith("minimax/"):
            return f"minimax::{text[len('minimax/'):]}"
        return text
    if provider == "deepseek":
        if text.startswith("deepseek::"):
            return text
        if text.startswith("deepseek/"):
            return f"deepseek::{text[len('deepseek/'):]}"
        return text
    if provider == "zai":
        if text.startswith("zai/"):
            return f"zai::{text[len('zai/'):]}"
        return text
    return text


def legacy_openai_base(settings: Mapping[str, Any] | None = None) -> bool:
    """Whether the legacy compatible base URL is set: the built-in OpenAI key then belongs
    to that endpoint (``compatible_endpoint``), so it is not official OpenAI access."""
    read = runtime_setting if settings is None else settings.get
    return bool(str(read("OPENAI_BASE_URL", "") or "").strip())


def configured_kinds(settings: Mapping[str, Any] | None = None) -> set[str]:
    """API kinds with at least one enabled connection, built-in or named, that can serve."""
    from ouroboros.provider_connections import configured_connections

    return {kind for kind, _name in configured_connections(settings)}


def exclusive_direct_kind(settings: Mapping[str, Any] | None = None) -> str:
    """The one direct provider an install reaches, else '' (``settings`` None = this process's view).

    OpenRouter, an OpenAI-compatible endpoint or the legacy base URL each mean the
    router-style defaults stay; among the direct providers one is returned only when
    exactly one is configured."""
    kinds = configured_kinds(settings)
    if {"openrouter", "openai-compatible"} & kinds or legacy_openai_base(settings):
        return ""
    direct = [kind for kind in PROVIDER_KINDS if kind in kinds and kind in DIRECT_PROVIDER_DEFAULTS]
    return direct[0] if len(direct) == 1 else ""


# The settings whose presence decides whether a built-in connection is configured (and
# the named connections): credentials by presence only, never their values.
READINESS_SETTING_KEYS: frozenset[str] = frozenset({
    *(spec.flat_key(field) for spec in PROVIDER_KINDS.values() for group in spec.ready for field in group),
    *_LEGACY_COMPATIBLE_KEYS,
})


def compatible_only_main_model(settings: Mapping[str, Any]) -> str:
    """Main's exact OpenAI-compatible route when such endpoints are the install's ONLY
    remote access, else ''.

    #1116: the shipped reviewer defaults are OpenRouter ids such an install cannot
    reach, and the direct-provider remap deliberately excludes the compatible
    endpoint. Review defaults then use the one model this install demonstrably
    pays for — never a guessed id from another provider (less diversity, disclosed).
    """
    if configured_kinds(settings) != {"openai-compatible"} or legacy_openai_base(settings):
        return ""
    main = str(settings.get("OUROBOROS_MODEL") or "").strip()
    return main if parse_model_route(main).kind == "openai-compatible" else ""


# A built-in connection's endpoint as Capability Evidence fingerprints it, byte for byte
# what stored window evidence and owner acknowledgements were filed under: none for the
# fixed-endpoint kinds, the saved endpoint for these, the region/plan endpoint for MiniMax/Z.ai.
_FINGERPRINT_ENDPOINT_FIELD = {"openai-compatible": "base_url", "cloudru": "base_url", "gigachat": "base_url"}


def capability_base_url(model: str, settings: Mapping[str, Any] | None = None) -> str:
    """The base URL Capability Evidence fingerprints ``model``'s route by — not always
    the dispatch URL. Built-in OpenAI keeps the legacy base URL it always used; a named
    connection reads the same field from its row; one that cannot serve has none."""
    from ouroboros.llm_attempt import ConnectionUnavailable
    from ouroboros.provider_connections import connection_values, settings_view

    route = parse_model_route(model)
    if route.kind not in PROVIDER_KINDS:
        return ""
    read, rows = settings_view(settings)
    try:
        get = connection_values(route, read, rows)
    except ConnectionUnavailable:
        return ""
    if route.kind == "minimax":
        return resolve_minimax_base_url(get("region") or "")
    if route.kind == "zai":
        return resolve_zai_base_url(get("plan") or "")
    if route.kind == "openai":
        return "" if route.connection else str(read("OPENAI_BASE_URL", "") or "")
    field = _FINGERPRINT_ENDPOINT_FIELD.get(route.kind)
    return str(get(field) or "") if field else ""


_KIND_CHOICES = {("minimax", "region"): (MINIMAX_REGION_ENDPOINTS, "global_en or cn_zh"),
                 ("zai", "plan"): (ZAI_PLAN_ENDPOINTS, "payg or coding")}


def choice_error(kind: str, field: str, value: Any) -> str:
    """Why ``value`` is not one of a kind field's closed choices (MiniMax region, Z.ai plan), or ''."""
    choices = _KIND_CHOICES.get((kind, field))
    text = str(value or "").strip().lower()
    return f"must be {choices[1]}" if choices and text and text not in choices[0] else ""


def normalize_builtin_choices(settings: dict) -> str:
    """Lower-case the built-in connections' closed choices in place; the first invalid one's message, else ''."""
    for kind, field in _KIND_CHOICES:
        key = PROVIDER_KINDS[kind].flat_key(field)
        if error := choice_error(kind, field, settings.get(key)):
            return f"{key} {error}."
        settings[key] = str(settings.get(key) or "").strip().lower()
    return ""


def compute_direct_review_models_fallback(
    provider: str,
    main_model: str,
    light_model: str = "",
) -> list[str]:
    """Compile a direct-provider review panel from declarative role names."""
    return [model for _role, model in compute_direct_review_seats(provider, main_model, light_model)]


def compute_direct_review_seats(
    provider: str,
    main_model: str,
    light_model: str = "",
) -> list[tuple[str, str]]:
    """The same panel as ``(role, model)`` seats, one per declared role: ``role`` is the owner slot a
    seat runs (``main``, or ``light`` for a Light of this provider), '' where the provider's shipped
    Light stands in."""
    if provider not in DIRECT_PROVIDER_DEFAULTS:
        return []
    provider_prefix = f"{provider}::"
    main = migrate_model_value(provider, main_model)
    if not main.startswith(provider_prefix):
        return []
    light = migrate_model_value(provider, light_model) if light_model else ""
    default_light = migrate_model_value(provider, DIRECT_PROVIDER_DEFAULTS[provider].get("light", ""))
    seats = {"main": ("main", main), "light": ("light", light) if light.startswith(provider_prefix)
             else ("", default_light) if default_light else ("main", main)}
    roles = DIRECT_PROVIDER_REVIEW_ROLES.get(provider, ("main", "light", "light"))
    return [seats.get(role, seats["main"]) for role in roles]


def image_input_from_modalities(value: Any) -> bool | None:
    """Image input stated by one input-modality list: True/False, or None when it says nothing.

    Only a non-empty list of non-blank strings is a statement. A missing, null or empty
    list, another shape, or one item that is not a non-blank string leaves it unknown:
    the valid remainder of ``["text", 42]`` or ``["text", ""]`` is not a complete list
    that omits images.
    """
    if isinstance(value, list) and value and all(isinstance(item, str) and item.strip() for item in value):
        return "image" in {item.strip().lower() for item in value}
    return None


def supports_vision(model_id: str, *, model_role: str = "",
                    model_account_override: str | None = None) -> bool | None:
    """Image input of the exact route: True, False, or None when nothing sourced says.

    A Claudexor route answers from the input modalities in the catalog of this
    call's role/account (``image_input_from_modalities``), never
    a global overlay; an API route answers from the fresh route-scoped catalog
    record (``vision_routing.route_image_input``). A model's name is not
    evidence, so every other case is None: image senders preserve input when the
    fact is unknown and the actual call returns the route's own answer. Our own
    lanes that cannot carry bytes (local, GigaChat) are the send policy's
    transport fact, decided by lane, not a model fact. Metadata discovery never
    starts an engine or buys a model generation.
    """
    if provider_for_model(model_id) == "claudexor":
        from ouroboros.gateways.claudexor import ClaudexorUnavailable
        from ouroboros.llm import LLMClient
        from ouroboros.model_slots import MODEL_ACCOUNTS_KEY, model_role_option
        from ouroboros.model_wait import current_model_wait

        source, native_model = parse_claudexor_model(model_id)
        wait = current_model_wait()
        if model_account_override is None and wait is not None:
            model_account_override = wait.overrides.get(model_role, {}).get("model_account_override")
        account = (model_account_override if model_account_override is not None
                   else model_role_option(MODEL_ACCOUNTS_KEY, model_role))
        try:
            catalog = LLMClient.claudexor_model_catalog(
                source, account or None, requested_model=native_model)
        except ClaudexorUnavailable:
            return None
        if catalog.get("source") != source or (account and catalog.get("credentialProfileId") != account):
            return None
        item = next((row for row in catalog.get("models", []) if row.get("id") == native_model), {})
        return image_input_from_modalities(item.get("inputModalities"))
    from ouroboros.vision_routing import route_image_input

    return route_image_input(model_id).verdict


# NOTE (v6.33.0): the static per-model context-window table was REMOVED. It
# perpetually went stale (1M-beta models hard-coded to 200K, [1m] ignored). The
# agent's OWN operating window is the owner low/max context MODE (the SSOT — see
# context_budget.py / loop.py), and external-model windows are resolved by
# Capability Evidence (ouroboros.capability_evidence: confirmed provider metadata
# / local health, or route-fingerprinted owner-ack), fail-closed when unknown.


def normalize_model_identity(model: str) -> str:
    """One model identity across route spellings: ``kind[@name]::m`` -> ``kind/m``, an
    OpenRouter id as itself (``openrouter::`` folded). A connection name says which key
    pays, not which model answers, so it is dropped — except for OpenAI-compatible
    connections, which are different servers that must never share learned facts."""
    text = str(model or "").strip()
    if text.endswith(" (local)"):
        text = text[:-8]
    route = parse_model_route(text)
    if not route.valid or route.kind == "claudexor":
        return text
    native = route.model
    if route.kind == "openrouter":
        if not native.startswith("anthropic/"):
            return native
        route, native = RouteParts("anthropic", "", ""), native[len("anthropic/"):]
    if route.kind == "anthropic":
        native = normalize_anthropic_model_id(native)
    keep_name = route.kind == "openai-compatible" and route.connection
    return f"{connection_id(route.kind, route.connection) if keep_name else route.kind}/{native}"
