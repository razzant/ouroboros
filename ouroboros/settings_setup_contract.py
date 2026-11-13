"""Shared Settings/Onboarding setup contract and payload validation."""

from __future__ import annotations

import math
import os
import re
from typing import Any, Dict, Optional, Tuple

from ouroboros.config import SETTINGS_DEFAULTS, VALID_RUNTIME_MODES
from ouroboros.review_run_isolation import run_cap_from_env
from ouroboros.provider_models import DIRECT_PROVIDER_DEFAULTS, MINIMAX_REGION_ENDPOINTS
from ouroboros.secret_masking import (
    CONFIGURED_SECRET_PLACEHOLDER,
    MASKED_SECRET_SETTING_KEYS as SECRET_SETTING_KEYS,
)
from ouroboros.model_slots import (
    MODEL_ACCOUNTS_KEY, MODEL_CONTEXT_WINDOWS_KEY, MODEL_PROCESSING_PREFERENCES_KEY,
    PROCESSING_PREFERENCE_KEY, normalize_model_role_options, normalize_processing_preference,
)
from ouroboros.provider_models import compatible_endpoint_configured, parse_claudexor_model, provider_for_model
from ouroboros.provider_connections import configured_connections, connection_rows, normalize_connections
from ouroboros.gateway.secret_rows import mask_connections, rehydrate_connections
from ouroboros.settings_integrity import operator_variables, skill_environ



_SAVED_TOTAL_BUDGET: Dict[str, Any] = {"stamp": None, "raw": ""}


def _saved_total_budget() -> str:
    """The owner's CURRENT saved ``TOTAL_BUDGET``, or "" when the document has none.

    ``TOTAL_BUDGET`` is an immediate setting, but a worker process re-projects the
    settings document into its environment only when a task STARTS, so for the
    hours a task runs that environment answers with the budget of its first
    minute: an owner who topped the budget up mid-run watched the task stop at
    the old number. The document itself is the one channel every process already
    shares (``loop_tool_execution._get_tool_timeout`` reads it through the
    loader on every tool call; the MCP reload stat-gates it), so the resolver asks
    it first. The read is unlocked (saves are
    atomic, and only writers need the settings lock) and re-parses only when the
    file changed, so a reservation costs one ``stat``; any failure -- a missing
    file, a refused integrity pin -- leaves the environment answering, because a
    settings hiccup must never become a money-path exception."""
    from ouroboros import config
    from ouroboros.settings_integrity import read_settings_json_verified

    path = config.SETTINGS_PATH
    try:
        found = path.stat()
    except OSError:
        return ""
    stamp = (str(path), found.st_dev, found.st_ino, found.st_mtime_ns, found.st_size)
    if _SAVED_TOTAL_BUDGET["stamp"] != stamp:
        try:
            document = read_settings_json_verified(path)
        except Exception:
            # Never remembered: a refused pin or a transient read error must not
            # become a sticky answer; the environment (the last verified
            # projection of this same document) answers until a read succeeds.
            return ""
        if not isinstance(document, dict):
            # Without a pin the verified reader answers None for an absent,
            # unreadable or undecodable file instead of raising: the same
            # failure, the same rule -- the environment answers, nothing is kept.
            return ""
        saved = document.get("TOTAL_BUDGET")
        _SAVED_TOTAL_BUDGET.update(stamp=stamp, raw="" if saved is None else str(saved).strip())
    return str(_SAVED_TOTAL_BUDGET["raw"])


def resolve_total_budget_usd() -> Optional[float]:
    """The effective global money limit, or None when the owner set no limit.

    Lives beside this setting's owner-facing declaration below, because the
    question it answers is what the declared default MEANS at runtime. Every
    reader used to invent its own fallback for an absent key and they disagreed
    threefold on the same install ($1, no limit at all, $200): the same machine
    could reject a task at round one on the loop's money axis while the ledger
    fence still allowed $200 of work. An absent key is not an owner decision --
    ``config.apply_settings_to_env`` deliberately projects only what the
    settings file actually says, so a raw harness run with no settings.json has
    no entry at all -- so absence resolves to the product default here. A
    non-positive value IS an owner decision and keeps its historical meaning of
    no finite global budget.

    A run cap a launcher set BEFORE settings load (``review_run_isolation``; its one
    producer is the isolated contributor review lane) is that process tree's whole
    global limit and the saved document is not consulted: the isolated ledger
    starts empty, so a saved lifetime budget would otherwise become a fresh
    allowance there. An unreadable cap is a zero allowance, never "no limit".
    """
    run_cap = run_cap_from_env()
    if run_cap is not None:
        return run_cap
    raw = _saved_total_budget() or str(os.environ.get("TOTAL_BUDGET", "") or "").strip()
    default = float(SETTINGS_DEFAULTS["TOTAL_BUDGET"])
    if not raw:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value if value > 0 else None


def _rows(keys: tuple[str, ...], specs: tuple[tuple[Any, ...], ...]) -> list[dict]:
    rows = []
    for spec in specs:
        if len(spec) != len(keys):
            raise ValueError(f"setup contract row has {len(spec)} fields, expected {len(keys)}")
        rows.append(dict(zip(keys, spec)))
    return rows


_MODEL_DEFAULTS = {
    "openrouter": {
        "main": str(SETTINGS_DEFAULTS["OUROBOROS_MODEL"]),
        "light": str(SETTINGS_DEFAULTS["OUROBOROS_MODEL_LIGHT"]),
        "vision": str(SETTINGS_DEFAULTS["OUROBOROS_MODEL_VISION"]),
        "consciousness": str(SETTINGS_DEFAULTS["OUROBOROS_MODEL_CONSCIOUSNESS"]),
        "fallback": str(SETTINGS_DEFAULTS["OUROBOROS_MODEL_FALLBACKS"]),
    },
    # Every direct provider's own defaults (provider_models.DIRECT_PROVIDER_DEFAULTS), never a second list.
    **{kind: {key: value for key, value in defaults.items() if key != "heavy"}
       for kind, defaults in DIRECT_PROVIDER_DEFAULTS.items()},
    # No defaults: model names are server-specific; user must fill all slots.
    "openai-compatible": {"main": "", "light": "", "vision": "", "fallback": ""},
    # Several direct providers and no router: the owner chooses Main; no provider is guessed.
    "direct-multi": {"main": "", "light": "", "vision": "", "fallback": ""},
}
# Local-only: Light follows the local Main and Fallbacks is no chain (the boot proposal,
# server_runtime.provider_slot_proposals); Main keeps the shipped id as the local label.
_MODEL_DEFAULTS["local"] = {**_MODEL_DEFAULTS["openrouter"], "light": "", "fallback": ""}
for _profile_defaults in _MODEL_DEFAULTS.values():
    _profile_defaults.setdefault("consciousness", "")
    _profile_defaults.setdefault("vision", "")
    _profile_defaults.setdefault("safety", "")

_STEPS = _rows(("id", "title", "railCopy", "copy", "footer"), (
    ("accounts", "Connect your accounts", "Subscriptions + API", "Connect Codex to start without an API key, or add an API key or local model. The same account can serve models and agents.", "Add more subscriptions or API access later in Settings → Accounts and keys."),
    ("models", "Choose models", "model slots", "Review the visible model defaults derived from your current setup, then edit anything you want before launch.", "Each slot names its source. OpenRouter stays the source for ids such as openai/gpt-5.6-terra; a provider's own key routes that slot straight to the provider."),
    ("review_mode", "Choose access", "Reach", "Choose how much access Ouroboros gets.", ""),
    ("budget", "Review limits", "Quota + API budget", "Review subscription quotas and optional API spending limits.", ""),
    ("summary", "Review before launch", "Final check", "Check the final provider, model, and budget picture. Ouroboros will save these onboarding values before starting.", "The same onboarding values remain editable later in Settings."),
))
_STEP_ORDER = [step["id"] for step in _STEPS]

# The "group" column drives onboarding layout only: "primary" fields render in
# the always-visible access grid, "more" fields render inside the collapsed
# "More options" disclosure. Every input stays mounted in the DOM either way.
_PROVIDER_FIELDS = _rows(("id", "stateKey", "settingKey", "settingsInputId", "label", "placeholder", "note", "inputType", "group"), (
    ("openrouter-key", "openrouterKey", "OPENROUTER_API_KEY", "s-openrouter", "OpenRouter API Key", "sk-or-v1-...", "Best when you want one router for OpenAI, Anthropic, Google, and more.", "password", "primary"),
    ("openai-key", "openaiKey", "OPENAI_API_KEY", "s-openai", "OpenAI API Key", "sk-...", "", "password", "primary"),
    ("cloudru-key", "cloudruKey", "CLOUDRU_FOUNDATION_MODELS_API_KEY", "s-cloudru-key", "Cloud.ru Foundation Models API Key", "Cloud.ru API key", "", "password", "more"),
    # GigaChat is ready with an authorization key, or a user and password (provider_models.PROVIDER_KINDS).
    ("gigachat-credentials", "gigachatCredentials", "GIGACHAT_CREDENTIALS", "s-gigachat-credentials", "GigaChat Authorization Key", "Base64 client_id:secret", "", "password", "more"),
    ("gigachat-user", "gigachatUser", "GIGACHAT_USER", "s-gigachat-user", "GigaChat User", "username", "Basic auth instead of an authorization key, with the password below.", "text", "more"),
    ("gigachat-password", "gigachatPassword", "GIGACHAT_PASSWORD", "s-gigachat-password", "GigaChat Password", "password", "Basic auth, with the user above.", "password", "more"),
    ("minimax-key", "minimaxKey", "MINIMAX_API_KEY", "s-minimax-key", "MiniMax API Key", "MiniMax API key", "", "password", "more"),
    ("minimax-region", "minimaxRegion", "MINIMAX_REGION", "s-minimax-region", "MiniMax Region", "global_en or cn_zh", "Choose global_en for the global endpoint or cn_zh for the China endpoint.", "text", "more"),
    ("deepseek-key", "deepseekKey", "DEEPSEEK_API_KEY", "s-deepseek-key", "DeepSeek API Key", "sk-...", "", "password", "more"),
    ("zai-key", "zaiKey", "ZAI_API_KEY", "s-zai-key", "Z.ai API Key (GLM)", "...", "", "password", "more"),
    ("zai-plan", "zaiPlan", "ZAI_PLAN", "s-zai-plan", "Z.ai Plan", "payg or coding", "Choose payg (pay-as-you-go, default) or coding (Coding Plan endpoint; officially intended for supported coding tools only).", "text", "more"),
    ("anthropic-key", "anthropicKey", "ANTHROPIC_API_KEY", "s-anthropic", "Anthropic API Key", "sk-ant-...", "Saved for models routed straight to Anthropic, and for Claude tooling.", "password", "primary"),
    ("openai-compatible-url", "compatibleBaseUrl", "OPENAI_COMPATIBLE_BASE_URL", "s-compatible-url", "OpenAI-compatible Base URL", "http://localhost:11434/v1", "Base URL for your OpenAI-compatible endpoint (e.g. Ollama, LM Studio, vLLM). Required whenever a slot uses the OpenAI-compatible endpoint as its source.", "url", "more"),
    ("openai-compatible-key", "compatibleApiKey", "OPENAI_COMPATIBLE_API_KEY", "s-compatible-key", "OpenAI-compatible API Key", "Leave empty for no auth", "API key for the endpoint. Leave empty if your server does not require authentication.", "password", "more"),
))

# Secret-key membership and its configured marker are re-exported from the
# leaf wire contract so onboarding, Settings, repair, and persistence cannot drift.

_PROFILE_SPECS = {
    "openrouter": ("OpenRouter", "OpenRouter is present, so the next step keeps router-style defaults while still saving any extra direct keys you paste here.", "OpenRouter-style routing remains active. OpenRouter stays the source for ids such as openai/gpt-5.6-terra or anthropic/claude-sonnet-5."),
    "openai": ("OpenAI", "OpenAI is present, so the next step prefills OpenAI's own model ids.", "OpenAI-only setup detected. These defaults are explicit and official."),
    "gigachat": ("GigaChat", "GigaChat is present, so the next step prefills GigaChat's own model ids.", "GigaChat-only setup detected. These defaults use GigaChat's own model ids."),
    "cloudru": ("Cloud.ru Foundation Models", "Cloud.ru is present, so the next step prefills Cloud.ru's own model ids.", "Cloud.ru-only setup detected. These defaults use Cloud.ru's own model ids."),
    "minimax": ("MiniMax", "MiniMax is present, so the next step prefills MiniMax's own model ids.", "MiniMax-only setup detected. These defaults include MiniMax-M3 and MiniMax-M3.1-Flash."),
    "deepseek": ("DeepSeek", "DeepSeek is present, so the next step prefills DeepSeek's own model ids.", "DeepSeek-only setup detected. These defaults use deepseek-v4-pro for main work and deepseek-v4-flash for the light lane."),
    "zai": ("Z.ai (GLM)", "Z.ai is present, so the next step prefills Z.ai's own GLM model ids.", "Z.ai-only setup detected. These defaults use glm-5.3 for main work and glm-5.3-flash for the light lane. The plan setting selects the pay-as-you-go or Coding Plan endpoint."),
    "anthropic": ("Anthropic", "Anthropic is present, so the next step prefills Anthropic's own model ids.", "Anthropic-only setup detected. These defaults are explicit and official."),
    "openai-compatible": ("OpenAI-compatible endpoint", "An OpenAI-compatible base URL is configured. Enter the model names your server exposes in the next step.", "OpenAI-compatible endpoint detected. Choose the OpenAI-compatible endpoint as the source and enter the model names your server exposes. The model list is whatever your server supports."),
    "direct-multi": ("Direct multi-provider", "Multiple direct providers are present, so the next step keeps your model values editable without forcing one provider family.", "Multiple direct providers are configured. Start here, then split model slots across them if you want."),
    "local": ("Local-first", "No remote key is present yet, so local-only setup remains available below.", "Local-only setup detected. Review the model values and local routing before launch."),
}

_MODEL_SLOTS = _rows(("slot", "stateKey", "settingKey", "inputId", "label", "note", "settingsInputId", "settingsToggleId"), (
    ("main", "mainModel", "OUROBOROS_MODEL", "main-model", "Main Model", "Primary reasoning and long-form work.", "s-model", "s-local-main"),
    ("light", "lightModel", "OUROBOROS_MODEL_LIGHT", "light-model", "Light Model", "Fast summaries, lightweight internal work and the default Fast scout. Empty uses Main.", "s-model-light", "s-local-light"),
    ("safety", "safetyModel", "OUROBOROS_MODEL_SAFETY", "safety-model", "Safety Supervisor", "Empty uses the Light model.", "s-model-safety", ""),
    ("vision", "visionModel", "OUROBOROS_MODEL_VISION", "vision-model", "Vision Model", "Caption and VLM lane. Empty uses Main.", "s-model-vision", ""),
    ("consciousness", "consciousnessModel", "OUROBOROS_MODEL_CONSCIOUSNESS", "consciousness-model", "Consciousness Model", "High-horizon background consciousness. Empty uses Main.", "s-model-consciousness", "s-local-consciousness"),
    ("fallback", "fallbackModel", "OUROBOROS_MODEL_FALLBACKS", "fallback-model", "Fallback Model", "Fallback and resilience path.", "s-model-fallback", "s-local-fallback"),
))

_RUNTIME_MODES = _rows(("value", "label", "tone", "className", "copy"), (
    ("light", "Light", "Safest", "light", "Works on your projects. Never changes its own code."),
    ("advanced", "Advanced", "Default", "advanced", "May grow its own skills and outer layer."),
    ("pro", "Pro", "Power", "pro", "May change its protected core. Reads a review before each change."),
    ("cyber_pro", "Cyber Pro", "Maximum power", "cyber-pro", "Full say over itself and this computer. Asks for reviews when it judges."),
))

_LOCAL_ROUTING_MODES = _rows(("value", "buttonLabel", "label", "flags"), (
    ("cloud", "Cloud only", "Cloud models only", (False, False, False, False)),
    ("fallback", "Fallback local", "Fallback model local", (False, False, False, True)),
    ("all", "All models local", "All active models local", (True, True, True, True)),
))

_ACTIVE_LOCAL_KEYS = (
    "USE_LOCAL_MAIN", "USE_LOCAL_LIGHT", "USE_LOCAL_CONSCIOUSNESS", "USE_LOCAL_FALLBACK",
)

_BUDGET_NOTE = (
    "Spending limits use known spend only: confirmed and estimated prices count toward the applicable limit; "
    "known $0 local calls are priced at $0 and do not use the limit. Unpriced calls stay without a price "
    "and are not counted as $0: they add nothing to known spend, so unpriced spending alone never triggers a limit. "
    "Open reservations and unresolved charges are not counted as spending. The Total Budget covers all work: once "
    "known spend reaches it, new paid calls are refused, API calls with an unknown price included, while known $0 "
    "calls and subscription or local calls on a route that has charged nothing continue. The Per-task Cost Cap "
    "covers one task's whole tree, subagents included: once that tree's known spend reaches it, every new model call "
    "of the tree is refused, known $0 and unpriced calls included. A resumable task pauses with its work saved; "
    "without a continuation it ends budget-exhausted. Calls already in flight settle normally and can overshoot the "
    "limit. Raising the cap does not resume a paused task; it must be resumed explicitly."
)

_BUDGET_FIELDS = [
    {
        "stateKey": "totalBudget",
        "settingKey": "TOTAL_BUDGET",
        "inputId": "total-budget",
        "title": "Total budget",
        "label": "Total Budget (USD)",
        # Owner 2026-10-03 Q4-A / 2026-10-07: the cap is the limit itself,
        # decided on known spend; no default share of the wallet or margin.
        "note": _BUDGET_NOTE,
        "default": float(SETTINGS_DEFAULTS["TOTAL_BUDGET"]),
        "min": "0.01",
        "step": "any",
    },
    {
        "stateKey": "perTaskCostUsd",
        "settingKey": "OUROBOROS_PER_TASK_COST_USD",
        "inputId": "per-task-budget",
        "title": "Per-task cost cap",
        "label": "Per-task Cost Cap (USD)",
        "note": _BUDGET_NOTE,
        "default": float(SETTINGS_DEFAULTS.get("OUROBOROS_PER_TASK_COST_USD", 50.0)),
        "min": "0.01",
        "step": "any",
    },
]
_BUDGET_FIELDS_BY_KEY = {field["settingKey"]: field for field in _BUDGET_FIELDS}
BUDGET_SETTING_KEYS = tuple(_BUDGET_FIELDS_BY_KEY)

# Local presets (12.11.2026): the current Qwen generation, trained to 262,144 tokens with full attention in one
# layer of four, so a long window costs little memory. A fresh install's first turn needs about 34K tokens of input
# beside the reply's room (measured on Qwen3.8-27B), so each opens at 65,536. Measured memory: Qwen3.8-27B held
# 19.45 GB resident at 65,536 (macOS footprint showed 37 GB right after a first turn); the 9B and the 35B-A3B are not
# measured. Their GGUF files need a llama-cpp-python runtime from mid-2026 or later (0.3.36 loads the 27B).
_LOCAL_PRESETS: Dict[str, Dict[str, Any]] = {
    "qwen35-9b": {"label": "Qwen3.5-9B Q4_K_M (5.7 GB file)", "source": "unsloth/Qwen3.5-9B-GGUF", "filename": "Qwen3.5-9B-Q4_K_M.gguf", "contextLength": 65536, "chatFormat": ""},
    "qwen38-27b": {"label": "Qwen3.8-27B Q4_K_M (16.6 GB file)", "source": "unsloth/Qwen3.8-27B-GGUF", "filename": "Qwen3.8-27B-UD-Q4_K_M.gguf", "contextLength": 65536, "chatFormat": ""},
    "qwen36-35b-a3b": {"label": "Qwen3.6-35B-A3B Q4_K_M (22 GB file)", "source": "unsloth/Qwen3.6-35B-A3B-GGUF", "filename": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf", "contextLength": 65536, "chatFormat": ""},
}

# Agent SUBSCRIPTION intent (install-time presets, D-3/D-4). These two
# are NOT settings keys — they are wizard→server declarations about the
# onboarding run itself, so they are declared here (one setup vocabulary for
# every host) instead of being invented separately by each caller. The server
# never trusts the first one as authority for anything but "look at the daemon";
# what a connected subscription actually IS gets re-read from Claudexor.
SUBSCRIPTIONS_CONNECTED_FIELD = "subscriptionsConnected"
SKIP_SUBSCRIPTION_PRESETS_FIELD = "skipSubscriptionPresets"

# The durable "onboarding finished here" fact, written by every completion —
# including one that connected no subscription and one that skipped the preset.
# Absence of a working provider is a state an OLD install reaches too, so it
# cannot be the latch on its own; this key is what closes the window for good.
ONBOARDING_COMPLETED_KEY = "OUROBOROS_ONBOARDING_COMPLETED_AT"
# Which completion attempt wrote that fact: the id the CLIENT made before sending, so it can
# still ask about its own save when the answer is lost. The same write carries it.
ONBOARDING_ATTEMPT_KEY = "OUROBOROS_ONBOARDING_ATTEMPT_ID"
ATTEMPT_ID_FIELD = "attempt_id"
_ATTEMPT_ID = re.compile(r"[A-Za-z0-9_-]{16,128}")

_SUBSCRIPTION_FIELDS = _rows(("id", "payloadKey", "label", "note"), (
    ("subscriptions-connected", SUBSCRIPTIONS_CONNECTED_FIELD, "Agent subscription connected", "Set by the wizard when at least one agent account was signed in during onboarding. The server re-reads the live account state before applying anything."),
    ("skip-subscription-presets", SKIP_SUBSCRIPTION_PRESETS_FIELD, "Finish without agent defaults", "Completes onboarding without moving reviewers and subagents onto the connected subscriptions. Everything stays editable in Settings afterwards."),
))

_MODEL_SUGGESTIONS = list(dict.fromkeys(("google/gemini-3.8-flash", "x-ai/grok-4.6", "openai/gpt-6.1-sol", "openai/gpt-6-luna", "anthropic/claude-opus-5.5", "openai/gpt-5.6-terra", "openai/gpt-5.6-sol", "openai/gpt-5.6-luna", "openai::gpt-5.6-terra", "openai::gpt-5.6-sol", "openai::gpt-5.6-luna", "anthropic/claude-sonnet-5", "anthropic/claude-opus-5", "anthropic::claude-sonnet-5", "anthropic::claude-opus-5-5", "anthropic::claude-opus-5", "anthropic::claude-opus-4-6", "deepseek/deepseek-v4-pro", "deepseek::deepseek-v4-pro", "deepseek::deepseek-v4-flash", "zai::glm-5.3", "zai::glm-5.3-flash", "openai-compatible::meta-llama/compatible", "cloudru::zai-org/GLM-4.7", "minimax::MiniMax-M3", "minimax::MiniMax-M3.1-Flash")))


def _string(value: Any) -> str:
    return str(value or "").strip()


_truthy = lambda value: value is True or _string(value).lower() in {"1", "true", "yes", "on"}


def parse_budget_setting(
    key: str,
    raw_value: Any,
    *,
    use_default_for_blank: bool = False,
) -> Tuple[float | None, str | None]:
    """Parse one shared budget setting for onboarding and Settings saves."""
    field = _BUDGET_FIELDS_BY_KEY[key]
    name = "Budget" if key == "TOTAL_BUDGET" else "Per-task cost cap"
    if raw_value is None or raw_value == "":
        if use_default_for_blank:
            raw_value = field["default"]
        else:
            return None, f"{name} must be a number."
    if isinstance(raw_value, bool):
        return None, f"{name} must be a number."
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None, f"{name} must be a number."
    if not math.isfinite(value):
        return None, f"{name} must be a number."
    if value <= 0:
        return None, f"{name} must be greater than zero."
    min_value = float(field.get("min") or 0)
    if min_value > 0 and value < min_value:
        return None, f"{name} must be at least {field['min']}."
    return value, None


def secret_provider_setting_keys() -> frozenset:
    """Provider fields whose stored value must never leave the server verbatim.

    The union of the canonical secret keys and every password-typed provider
    field, so the rule holds for a field added on either side: a new credential
    listed in ``SECRET_SETTING_KEYS`` is covered even if someone renders it as a
    text input, and a new password input is covered before anyone remembers to
    add its key to the canonical set.
    """
    return frozenset(
        field["settingKey"]
        for field in _PROVIDER_FIELDS
        if field["settingKey"] in SECRET_SETTING_KEYS
        or (field.get("inputType") or "password") == "password"
    )


def _is_secret_placeholder(value: str) -> bool:
    # Imported lazily: the predicate's home is the MCP client, and the launcher
    # imports this contract long before any MCP machinery is wanted.
    from ouroboros.mcp_client import looks_masked_secret

    return looks_masked_secret(value)


def derive_provider_profile(settings: dict) -> str:
    flags = {field["settingKey"]: bool(_string(settings.get(field["settingKey"]))) for field in _PROVIDER_FIELDS}
    if flags["OPENROUTER_API_KEY"]:
        return "openrouter"
    # The legacy OPENAI_BASE_URL pair counts too: its OPENAI_API_KEY then belongs to that endpoint.
    if compatible_endpoint_configured(lambda key: _string(settings.get(key))):
        return "openai-compatible"
    direct = [
        ("OPENAI_API_KEY", "openai"),
        ("CLOUDRU_FOUNDATION_MODELS_API_KEY", "cloudru"),
        ("MINIMAX_API_KEY", "minimax"),
        ("DEEPSEEK_API_KEY", "deepseek"),
        ("ZAI_API_KEY", "zai"),
        ("ANTHROPIC_API_KEY", "anthropic"),
    ]
    configured = [name for key, name in direct if flags[key]]
    if len(configured) > 1:
        return "direct-multi"
    return configured[0] if configured else ("local" if _string(settings.get("LOCAL_MODEL_SOURCE")) else "openrouter")


def derive_local_routing_mode(settings: dict) -> str:
    flags = tuple(_truthy(settings.get(key)) for key in _ACTIVE_LOCAL_KEYS)
    if flags == (True, True, True, True):
        return "all"
    return "fallback" if flags == (False, False, False, True) else "cloud"


def local_routing_flags(mode: str, has_local: bool = True) -> tuple[bool, bool, bool, bool]:
    if not has_local:
        return (False, False, False, False)
    for item in _LOCAL_ROUTING_MODES:
        if item["value"] == mode:
            return tuple(bool(flag) for flag in item["flags"])  # type: ignore[return-value]
    return (False, False, False, False)


def model_defaults_for_profile(profile: str) -> dict:
    return dict(_MODEL_DEFAULTS.get(profile) or _MODEL_DEFAULTS["openrouter"])


def build_setup_contract(host_mode: str = "desktop") -> dict:
    """The choices Settings and the wizard offer: under a deployment policy only the Access levels it allows,
    and a Supervisor slot its checker fixes is ``managed`` (shown, never edited or written)."""
    from ouroboros.deployment_policy import CHECKER_KEY, checker, choice_refusal

    managed_checker = checker() is not None
    return {
        "version": 1,
        "hostMode": "web" if host_mode == "web" else "desktop",
        "steps": [dict(item) for item in _STEPS],
        "providerFields": [dict(item) for item in _PROVIDER_FIELDS],
        "providerProfiles": {key: {"label": spec[0], "providerCopy": spec[1], "modelCopy": spec[2]} for key, spec in _PROFILE_SPECS.items()},
        "modelSlots": [{**item, "managed": managed_checker and item["settingKey"] == CHECKER_KEY} for item in _MODEL_SLOTS],
        "runtimeModes": [dict(item) for item in _RUNTIME_MODES
                         if not choice_refusal("OUROBOROS_RUNTIME_MODE", item["value"])],
        "localRoutingModes": [dict(item) for item in _LOCAL_ROUTING_MODES],
        "budgetFields": [dict(item) for item in _BUDGET_FIELDS],
        "subscriptionFields": [dict(item) for item in _SUBSCRIPTION_FIELDS],
    }


def parse_subscription_intent(data: dict) -> Tuple[bool, bool]:
    """``(subscriptions_connected, skip_presets)`` from a wizard payload.

    Both default to False, and both are DECLARATIONS, not decisions: the first
    only tells the server to go look at the live Claudexor account state, and
    the second is the owner's explicit "finish without agent defaults" escape
    hatch. Neither can make the server apply a preset an install is not
    eligible for."""
    return (
        _truthy(data.get(SUBSCRIPTIONS_CONNECTED_FIELD)),
        _truthy(data.get(SKIP_SUBSCRIPTION_PRESETS_FIELD)),
    )


def parse_completion_attempt(data: Any) -> Tuple[str, str]:
    """``(attempt_id, error)`` from a completion payload: ``("", "")`` when the client
    sent none (older clients complete as before), an error for a malformed one."""
    raw = data.get(ATTEMPT_ID_FIELD) if isinstance(data, dict) else None
    if raw in (None, ""):
        return "", ""
    if isinstance(raw, str) and _ATTEMPT_ID.fullmatch(raw):
        return raw, ""
    return "", "attempt_id must be 16-128 letters, digits, '-' or '_'."


def build_initial_setup_state(settings: dict, host_mode: str = "desktop", document: Optional[dict] = None) -> dict:
    profile = derive_provider_profile(settings)
    defaults = model_defaults_for_profile(profile)
    local_source = _string(settings.get("LOCAL_MODEL_SOURCE"))
    local_filename = _string(settings.get("LOCAL_MODEL_FILENAME"))
    local_preset = next(
        (preset_id for preset_id, preset in _LOCAL_PRESETS.items() if local_source == preset["source"] and local_filename == preset["filename"]),
        "custom" if local_source else "",
    )
    try:
        raw_context_length = settings.get("LOCAL_MODEL_CONTEXT_LENGTH", SETTINGS_DEFAULTS["LOCAL_MODEL_CONTEXT_LENGTH"])
        local_context_length = int(raw_context_length if raw_context_length not in (None, "") else SETTINGS_DEFAULTS["LOCAL_MODEL_CONTEXT_LENGTH"])
    except (TypeError, ValueError):
        local_context_length = int(SETTINGS_DEFAULTS["LOCAL_MODEL_CONTEXT_LENGTH"])
    try:
        raw_gpu_layers = settings.get("LOCAL_MODEL_N_GPU_LAYERS", -1)
        local_gpu_layers = int(raw_gpu_layers if raw_gpu_layers not in (None, "") else -1)
    except (TypeError, ValueError):
        local_gpu_layers = -1
    budget_state: dict[str, float] = {}
    for field in _BUDGET_FIELDS:
        value, error = parse_budget_setting(field["settingKey"], settings.get(field["settingKey"]), use_default_for_blank=True)
        budget_state[field["stateKey"]] = float(field["default"] if error or value is None else value)
    state = {
        "connections": mask_connections(list(connection_rows(settings.get("OUROBOROS_CONNECTIONS")))),
        # Read-only here: an older install's endpoint, paired with OPENAI_API_KEY while the dedicated
        # address is empty (``compatible_endpoint``). No provider field: completion keeps it as saved.
        "legacyCompatibleBaseUrl": _string(settings.get("OPENAI_BASE_URL")),
        "providerProfile": profile,
        "runtimeMode": _string(settings.get("OUROBOROS_RUNTIME_MODE")) or str(SETTINGS_DEFAULTS["OUROBOROS_RUNTIME_MODE"]),
        "skillsRepoPath": _string(settings.get("OUROBOROS_SKILLS_REPO_PATH")),
        "localPreset": local_preset,
        "localSource": local_source,
        "localFilename": local_filename,
        "localContextLength": local_context_length,
        "localGpuLayers": local_gpu_layers,
        "localChatFormat": _string(settings.get("LOCAL_MODEL_CHAT_FORMAT")),
        "localRoutingMode": derive_local_routing_mode(settings),
    }
    # A credential is reported as CONFIGURED, never handed back. The onboarding
    # page is served by an unauthenticated GET on every host, and on a supported
    # non-loopback bind without a network password that page is retrievable by
    # anyone on the LAN — so the bootstrap must carry the same "configured or
    # not" fact /api/settings carries, and nothing more. The wizard needs only
    # that fact: which providers are already set up.
    secret_keys = secret_provider_setting_keys()
    state.update({
        field["stateKey"]: (
            CONFIGURED_SECRET_PLACEHOLDER
            if field["settingKey"] in secret_keys and _string(settings.get(field["settingKey"]))
            else _string(settings.get(field["settingKey"]))
        )
        for field in _PROVIDER_FIELDS
    })
    state.update(budget_state)
    state["modelAccounts"] = normalize_model_role_options(MODEL_ACCOUNTS_KEY, settings.get(MODEL_ACCOUNTS_KEY))[0]
    state["modelContextWindows"] = normalize_model_role_options(MODEL_CONTEXT_WINDOWS_KEY, settings.get(MODEL_CONTEXT_WINDOWS_KEY))[0]
    state["processingPreference"] = normalize_processing_preference(settings.get(PROCESSING_PREFERENCE_KEY))
    state["modelProcessingPreferences"] = normalize_model_role_options(MODEL_PROCESSING_PREFERENCES_KEY, settings.get(MODEL_PROCESSING_PREFERENCES_KEY))[0]
    # A loaded settings document carries every slot (defaults-merged), so a blank one
    # is a saved "inherit Main" and shows as saved; only a slot the document lacks
    # takes the profile default. A fresh install's wizard proposes its defaults itself.
    state.update({slot["stateKey"]: _string(settings[slot["settingKey"]]) if slot["settingKey"] in settings
                  else defaults[slot["slot"]] for slot in _MODEL_SLOTS})
    # The model slots the wizard proposes and may move to another key or a subscription's model (#2969): set by
    # neither the stored ``document`` (``settings`` when none is given) nor an operator's own variable. Any other
    # value, blank included, is kept.
    chosen = set(settings if document is None else document) | set(
        operator_variables(skill_environ(), [slot["settingKey"] for slot in _MODEL_SLOTS]))
    state["automaticModels"] = [slot["stateKey"] for slot in _MODEL_SLOTS if slot["settingKey"] not in chosen]
    return state


def consciousness_offer(settings: dict, *, enabled: Optional[bool] = None) -> dict:
    """The Summary's optional Consciousness row. It is offered while setup has never completed here
    (an unticked choice the owner makes now; a completed setup neither shows nor repeats one), with
    the limits consciousness runs under now, as the runtime's own readers report them. ``enabled``
    is the control as the state shows it (``None``: unknown). Nothing here is saved or started:
    a ticked row sends the ordinary Start after the save, from the wizard."""
    from ouroboros.runtime_limits import (
        get_consciousness_autonomy, get_consciousness_daily_usd, get_consciousness_max_tasks,
    )

    return {"offered": not _string(settings.get(ONBOARDING_COMPLETED_KEY)),
            "enabled": enabled if isinstance(enabled, bool) else None,
            "autonomy": get_consciousness_autonomy(), "dailyUsd": get_consciousness_daily_usd(),
            "maxTasks": get_consciousness_max_tasks()}


def build_setup_bootstrap(settings: dict, host_mode: str = "desktop", *, fresh_install: bool = False,
                          consciousness_enabled: Optional[bool] = None, document: Optional[dict] = None) -> dict:
    from ouroboros.deployment_policy import policy_state

    normalized_host = "web" if host_mode == "web" else "desktop"
    return {
        "hostMode": normalized_host,
        # Display provenance only; completion re-proves install eligibility.
        # Unknown/legacy callers preserve existing model choices conservatively.
        "freshInstall": fresh_install,
        "supportsLocalRuntimeControls": normalized_host == "web",
        "stepOrder": list(_STEP_ORDER),
        "modelDefaults": {key: dict(value) for key, value in _MODEL_DEFAULTS.items()},
        "localPresets": {key: dict(value) for key, value in _LOCAL_PRESETS.items()},
        "modelSuggestions": list(_MODEL_SUGGESTIONS),
        # The wizard must recognize its own "already configured" marker without
        # hardcoding it: a prefilled credential field holds this string, not a key.
        "secretPlaceholder": CONFIGURED_SECRET_PLACEHOLDER,
        "contract": build_setup_contract(normalized_host),
        "initialState": build_initial_setup_state(settings, normalized_host, document),
        "consciousnessOffer": consciousness_offer(settings, enabled=consciousness_enabled),
        "deployment": policy_state(),
    }


def wizard_authors_safety_light() -> bool:
    """Whether a DESKTOP first-run wizard save may author the ``light`` safety
    default. True only for a genuinely FRESH install — no settings file on disk yet.

    Key absence alone is deliberately NOT the test: an older install re-entering the
    wizard (or one whose file is unreadable) would then be silently moved off the
    fail-closed ``full``. This predicate is HOST-AGNOSTIC by design and its one
    caller is the onboarding gateway (`gateway/onboarding.py`) — the shared
    validator must not author it, because web/Docker onboarding posts the same
    payload through generic `/api/settings`, which is a non-owner path. The persist
    seam re-proves freshness under the settings lock, so this is the caller-side
    eligibility check, not the authority."""
    from ouroboros.config import SETTINGS_PATH

    return not SETTINGS_PATH.exists()


def validate_setup_payload(data: dict, current_settings: dict) -> Tuple[dict, str | None]:
    """Structural validation shared by the desktop and web wizard.

    Returns ``(prepared_settings, None)`` or ``({}, error)``. Requires a remote key
    or URL, a local model source, or an agent subscription (a pending connection or
    a selected Claudexor model); a local-only setup must route at least one active
    lane locally. Main is required unless a subscription is pending; Light, Vision,
    Consciousness and Fallback are not validated, so empty keeps its inheritance
    semantics. Enforcement and runtime mode are closed enums, budgets finite and
    positive, the MiniMax region is closed, and a Hugging Face local source needs a
    filename. Credential length is checked only on fields changed in the payload,
    so an unchanged short stored value cannot discard the whole form."""
    subscriptions_connected, skip_presets = parse_subscription_intent(data)
    pending_subscription = subscriptions_connected and not skip_presets
    selected_subscription = provider_for_model(_string(data.get("OUROBOROS_MODEL"))) == "claudexor"
    if selected_subscription:
        try:
            parse_claudexor_model(_string(data.get("OUROBOROS_MODEL")))
        except ValueError as exc:
            return {}, str(exc)
    secret_keys = secret_provider_setting_keys()
    keys: Dict[str, str] = {}
    connections = current_settings.get("OUROBOROS_CONNECTIONS", [])
    if "OUROBOROS_CONNECTIONS" in data:
        connections, error = normalize_connections(data["OUROBOROS_CONNECTIONS"])
        if error:
            return {}, error
        try:
            connections = rehydrate_connections(connections, current_settings.get("OUROBOROS_CONNECTIONS"))
        except ValueError as exc:
            return {}, str(exc)
    for field in _PROVIDER_FIELDS:
        setting_key = field["settingKey"]
        raw_value = data.get(setting_key)
        if raw_value is not None and not isinstance(raw_value, str):
            # A JSON object/array/number here is a malformed API post, never a
            # credential: str()-ing it would persist unusable garbage as a
            # "successful" setup. Honest rejection over silent stringification.
            return {}, f"{field['label']} must be a text value."
        value = _string(raw_value)
        # An untouched credential field posts back the marker it was prefilled
        # with. It means "keep the stored secret" and NEVER "store this string":
        # resolving it to the stored value here — before the length check, the
        # has-remote gate and the prepared merge — is what keeps the marker out
        # of settings.json on every host, since every save path (the atomic
        # completion endpoint, the desktop bridge) shares this validator. With
        # nothing stored the marker resolves to empty, so it cannot become a key.
        if setting_key in secret_keys and _is_secret_placeholder(value):
            value = _string(current_settings.get(setting_key))
        keys[setting_key] = value
    local_source = _string(data.get("LOCAL_MODEL_SOURCE"))
    local_filename = _string(data.get("LOCAL_MODEL_FILENAME"))
    local_chat_format = _string(data.get("LOCAL_MODEL_CHAT_FORMAT"))
    local_routing_mode = _string(data.get("LOCAL_ROUTING_MODE")) or "cloud"
    raw_runtime_mode = _string(data.get("OUROBOROS_RUNTIME_MODE"))
    runtime_mode = raw_runtime_mode.lower() if raw_runtime_mode else _string(current_settings.get("OUROBOROS_RUNTIME_MODE")) or str(SETTINGS_DEFAULTS["OUROBOROS_RUNTIME_MODE"])

    for field in _PROVIDER_FIELDS:
        setting_key = field["settingKey"]
        value = keys[setting_key]
        # Only a credential the owner authored in THIS payload is length-checked.
        # build_initial_setup_state prefills every provider field from disk, so a
        # stored value the owner never touched arrives here unchanged; rejecting it
        # deadlocks the wizard, because the rejection discards the WHOLE payload —
        # including the replacement key typed in the same form. The value that
        # triggers the error then becomes the one value that can never be replaced.
        if value == _string(current_settings.get(setting_key)):
            continue
        if (
            value
            and setting_key not in {"OPENAI_COMPATIBLE_API_KEY", "GIGACHAT_PASSWORD"}  # API keys only
            and (field.get("inputType") or "password") == "password"
            and len(value) < 10
        ):
            return {}, f"{field['label'].replace(' API Key', '')} API key looks too short."

    has_remote = any(
        value
        for setting_key, value in keys.items()
        if setting_key not in {"OPENAI_COMPATIBLE_API_KEY", "MINIMAX_REGION", "ZAI_PLAN", "GIGACHAT_USER", "GIGACHAT_PASSWORD"}
    )
    has_remote = has_remote or bool(configured_connections({**keys, "OUROBOROS_CONNECTIONS": connections}))
    has_local = bool(local_source)
    if not has_remote and not has_local and not (pending_subscription or selected_subscription):
        return {}, "Connect Codex, an API provider, or a local model before continuing."
    minimax_region = keys.get("MINIMAX_REGION", "").lower()
    if minimax_region and minimax_region not in MINIMAX_REGION_ENDPOINTS:
        return {}, "MiniMax Region must be global_en or cn_zh."
    keys["MINIMAX_REGION"] = minimax_region
    if has_local and "/" in local_source and not local_source.startswith(("/", "~")) and not local_filename:
        return {}, "Local HuggingFace sources need a GGUF filename."
    if runtime_mode not in VALID_RUNTIME_MODES:
        return {}, f"Choose a runtime mode from {sorted(VALID_RUNTIME_MODES)}."

    models = {slot["settingKey"]: _string(data.get(slot["settingKey"])) for slot in _MODEL_SLOTS}
    # Role-model (v6.39): only Main is required. Light/Consciousness fall back to
    # Main when empty, and Fallbacks carries a resilience default (empty = no cross-model
    # fallback) — so the owner is not forced to fill every slot. Mirrors the relaxed
    # onboarding-wizard validateModelsStep.
    if not models.get("OUROBOROS_MODEL") and not pending_subscription:
        return {}, "Confirm the Main model before starting Ouroboros."

    parsed_budget: dict[str, float] = {}
    for field in _BUDGET_FIELDS:
        key = field["settingKey"]
        value, error = parse_budget_setting(key, data.get(key), use_default_for_blank=True)
        if error:
            return {}, error
        parsed_budget[key] = float(value)

    try:
        local_context_length = int(data.get("LOCAL_MODEL_CONTEXT_LENGTH") or SETTINGS_DEFAULTS["LOCAL_MODEL_CONTEXT_LENGTH"])
        local_gpu_layers = int(data.get("LOCAL_MODEL_N_GPU_LAYERS") if data.get("LOCAL_MODEL_N_GPU_LAYERS") is not None else -1)
    except (TypeError, ValueError):
        return {}, "Local model context length and GPU layers must be integers."

    use_local = local_routing_flags(local_routing_mode, has_local)
    if has_local and not has_remote and not any(use_local):
        return {}, "Local-only setups must route at least one model to the local runtime."

    prepared = dict(current_settings)
    if "OUROBOROS_CONNECTIONS" in data:
        prepared["OUROBOROS_CONNECTIONS"] = connections
    for key in (MODEL_ACCOUNTS_KEY, MODEL_CONTEXT_WINDOWS_KEY, MODEL_PROCESSING_PREFERENCES_KEY):
        if key in data:
            try:
                prepared[key] = normalize_model_role_options(key, data[key])[1]
            except ValueError as exc:
                return {}, str(exc)
    if PROCESSING_PREFERENCE_KEY in data:
        try:
            prepared[PROCESSING_PREFERENCE_KEY] = normalize_processing_preference(data[PROCESSING_PREFERENCE_KEY])
        except ValueError as exc:
            return {}, str(exc)
    prepared.update(models)
    prepared.update(keys)
    prepared.update(parsed_budget)
    prepared.update({
        "OUROBOROS_RUNTIME_MODE": runtime_mode,
        "OUROBOROS_SKILLS_REPO_PATH": _string(data.get("OUROBOROS_SKILLS_REPO_PATH")),
        "LOCAL_MODEL_SOURCE": local_source if has_local else "",
        "LOCAL_MODEL_FILENAME": local_filename if has_local else "",
        "LOCAL_MODEL_CONTEXT_LENGTH": local_context_length,
        "LOCAL_MODEL_N_GPU_LAYERS": local_gpu_layers,
        "LOCAL_MODEL_CHAT_FORMAT": local_chat_format if has_local else "",
        "USE_LOCAL_MAIN": use_local[0],
        "USE_LOCAL_LIGHT": use_local[1],
        "USE_LOCAL_CONSCIOUSNESS": use_local[2],
        "USE_LOCAL_FALLBACK": use_local[3],
    })
    return prepared, None
