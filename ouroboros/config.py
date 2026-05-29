"""
Ouroboros — Shared configuration (single source of truth).

Paths, settings defaults, load/save with file locking.
Only imports ouroboros.platform_layer (platform abstraction, no circular deps).
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from typing import Optional

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
    "OUROBOROS_MODEL_CODE": "google/gemini-3.5-flash",
    "OUROBOROS_MODEL_LIGHT": "google/gemini-3.5-flash",
    # Background consciousness is a high-horizon cognitive loop, not a cheap
    # helper lane. Empty means "use OUROBOROS_MODEL".
    "OUROBOROS_MODEL_CONSCIOUSNESS": "",
    "OUROBOROS_MODEL_FALLBACK": "anthropic/claude-sonnet-4.6",
    "CLAUDE_CODE_MODEL": "opus[1m]",
    "OUROBOROS_MAX_WORKERS": 5,
    "TOTAL_BUDGET": 10.0,
    "OUROBOROS_PER_TASK_COST_USD": 20.0,
    "OUROBOROS_SOFT_TIMEOUT_SEC": 600,
    "OUROBOROS_HARD_TIMEOUT_SEC": 1800,
    "OUROBOROS_TOOL_TIMEOUT_SEC": 600,
    "OUROBOROS_BG_MAX_ROUNDS": 10,
    "OUROBOROS_BG_WAKEUP_MIN": 30,
    "OUROBOROS_BG_WAKEUP_MAX": 7200,
    "OUROBOROS_EVO_COST_THRESHOLD": 0.10,
    "OUROBOROS_WEBSEARCH_MODEL": "gpt-5.2",
    # Pre-commit review: comma-separated provider-tagged model list
    "OUROBOROS_REVIEW_MODELS": "openai/gpt-5.5,google/gemini-3.5-flash,anthropic/claude-opus-4.8",
    # Pre-commit review enforcement: advisory | blocking
    "OUROBOROS_REVIEW_ENFORCEMENT": "advisory",
    # Auto-grant reviewed-skill requests by default; grants stay bound to the
    # reviewed content hash and editing a skill still invalidates them.
    "OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS": "true",
    # Runtime mode: light | advanced | pro; pro still requires review gates.
    "OUROBOROS_RUNTIME_MODE": "advanced",
    # Optional extra user-managed skills checkout; Ouroboros never clones/pulls it.
    "OUROBOROS_SKILLS_REPO_PATH": "",
    "OUROBOROS_CLAWHUB_REGISTRY_URL": "https://clawhub.ai/api/v1",
    "OUROBOROS_HUB_CATALOG_URL": "https://raw.githubusercontent.com/razzant/OuroborosHub/main/catalog.json",
    "MCP_ENABLED": False,
    "MCP_SERVERS": [],
    "MCP_TOOL_TIMEOUT_SEC": 60,
    # Scope review: one or more reviewer slots; enforcement follows OUROBOROS_REVIEW_ENFORCEMENT.
    "OUROBOROS_SCOPE_REVIEW_MODELS": "openai/gpt-5.5",
    "OUROBOROS_SCOPE_REVIEW_MODEL": "openai/gpt-5.5",
    "OUROBOROS_TASK_REVIEW_MODE": "auto",
    "OUROBOROS_SERVICE_LOG_RETENTION_DAYS": 14,
    # Reasoning effort per task type: none | low | medium | high
    "OUROBOROS_EFFORT_TASK": "medium",
    "OUROBOROS_EFFORT_EVOLUTION": "high",
    "OUROBOROS_EFFORT_REVIEW": "medium",
    "OUROBOROS_EFFORT_SCOPE_REVIEW": "high",
    "OUROBOROS_EFFORT_CONSCIOUSNESS": "high",
    "OUROBOROS_RETURN_REASONING": True,
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
    "USE_LOCAL_CODE": False,
    "USE_LOCAL_LIGHT": False,
    "USE_LOCAL_CONSCIOUSNESS": False,
    "USE_LOCAL_FALLBACK": False,
    "OUROBOROS_FILE_BROWSER_DEFAULT": "",
}


def get_light_model() -> str:
    """Return the configured light-model slot with runtime env override."""

    return (
        str(os.environ.get("OUROBOROS_MODEL_LIGHT", "") or "").strip()
        or str(SETTINGS_DEFAULTS["OUROBOROS_MODEL_LIGHT"])
    )


def get_consciousness_model() -> str:
    """Return the high-horizon background-consciousness model slot."""

    configured = str(os.environ.get("OUROBOROS_MODEL_CONSCIOUSNESS", "") or "").strip()
    if configured:
        return configured
    return (
        str(os.environ.get("OUROBOROS_MODEL", "") or "").strip()
        or str(SETTINGS_DEFAULTS["OUROBOROS_MODEL"])
    )

_VALID_EFFORTS = ("none", "low", "medium", "high")
_DIRECT_PROVIDER_REVIEW_RUNS = 3

# Runtime mode and review enforcement are separate axes.
VALID_RUNTIME_MODES = ("light", "advanced", "pro")

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
    has_compatible = bool(str(os.environ.get("OPENAI_COMPATIBLE_API_KEY", "") or "").strip())
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
        key = "OUROBOROS_EFFORT_TASK"
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
    if provider not in ("openai", "anthropic", "cloudru"):
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
    if not migrated or len(migrated) < 2 or any(not model.startswith(provider_prefix) for model in migrated):
        # Duplicate model IDs are valid stochastic reviewer slots.
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


def get_data_skills_dir() -> pathlib.Path:
    """Return ``<DATA_DIR>/skills/`` (created on demand)."""
    return ensure_data_skills_dir(DATA_DIR)


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


def get_clawhub_skills_dir() -> pathlib.Path:
    """Return ``<DATA_DIR>/skills/clawhub/`` (created on demand)."""
    target = get_data_skills_dir() / SKILL_SOURCE_CLAWHUB
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return target


def get_ouroboroshub_catalog_url() -> str:
    """Return the official OuroborosHub static catalog URL."""

    return str(load_settings().get("OUROBOROS_HUB_CATALOG_URL") or SETTINGS_DEFAULTS["OUROBOROS_HUB_CATALOG_URL"]).strip()


def get_ouroboroshub_skills_dir() -> pathlib.Path:
    """Return ``<DATA_DIR>/skills/ouroboroshub/`` (created on demand)."""

    target = get_data_skills_dir() / SKILL_SOURCE_OUROBOROSHUB
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


def save_settings(settings: dict, *, allow_elevation: bool = False) -> None:
    """Persist settings and enforce the runtime-mode self-elevation ratchet.

    Elevation above the boot baseline is refused after initialization; then
    ``allow_elevation=True`` is inert to agent-reachable subprocesses. Production
    entry points must call ``initialize_runtime_mode_baseline`` before agent code.
    """
    _guard_live_settings_write()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fd = _acquire_settings_lock()
    try:
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


def get_mcp_enabled() -> bool:
    raw = str(os.environ.get("MCP_ENABLED", "") or "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool(load_settings().get("MCP_ENABLED"))


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
        "OUROBOROS_MODEL", "OUROBOROS_MODEL_CODE", "OUROBOROS_MODEL_LIGHT",
        "OUROBOROS_MODEL_CONSCIOUSNESS",
        "OUROBOROS_MODEL_FALLBACK", "CLAUDE_CODE_MODEL",
        "TOTAL_BUDGET", "OUROBOROS_PER_TASK_COST_USD", "GITHUB_TOKEN", "GITHUB_REPO",
        "OUROBOROS_TOOL_TIMEOUT_SEC",
        "OUROBOROS_BG_MAX_ROUNDS", "OUROBOROS_BG_WAKEUP_MIN", "OUROBOROS_BG_WAKEUP_MAX",
        "OUROBOROS_EVO_COST_THRESHOLD", "OUROBOROS_WEBSEARCH_MODEL",
        "OUROBOROS_REVIEW_MODELS", "OUROBOROS_REVIEW_ENFORCEMENT",
        "OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS",
        "OUROBOROS_SCOPE_REVIEW_MODELS", "OUROBOROS_SCOPE_REVIEW_MODEL",
        "OUROBOROS_TASK_REVIEW_MODE",
        "OUROBOROS_SERVICE_LOG_RETENTION_DAYS",
        # Runtime-mode and skills-repo plumbing.
        "OUROBOROS_RUNTIME_MODE", "OUROBOROS_SKILLS_REPO_PATH",
        "OUROBOROS_HOST_SERVICE_PORT",
        # ClawHub marketplace registry URL.
        "OUROBOROS_CLAWHUB_REGISTRY_URL",
        "MCP_ENABLED", "MCP_TOOL_TIMEOUT_SEC",
        "OUROBOROS_EFFORT_TASK", "OUROBOROS_EFFORT_EVOLUTION",
        "OUROBOROS_EFFORT_REVIEW", "OUROBOROS_EFFORT_SCOPE_REVIEW",
        "OUROBOROS_EFFORT_CONSCIOUSNESS",
        "OUROBOROS_RETURN_REASONING",
        "LOCAL_MODEL_SOURCE", "LOCAL_MODEL_FILENAME",
        "LOCAL_MODEL_PORT", "LOCAL_MODEL_N_GPU_LAYERS", "LOCAL_MODEL_CONTEXT_LENGTH",
        "LOCAL_MODEL_CHAT_FORMAT",
        "USE_LOCAL_MAIN", "USE_LOCAL_CODE", "USE_LOCAL_LIGHT", "USE_LOCAL_CONSCIOUSNESS", "USE_LOCAL_FALLBACK",
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
