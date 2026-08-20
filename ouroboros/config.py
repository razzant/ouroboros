"""
Ouroboros — Shared configuration (single source of truth).

Paths, the settings-file lifecycle (locked load/normalize/save plus environment
projection) and the owner-only mode ratchets. The vocabularies it reads through —
shipped defaults, closed scales, model slots, reviewer routes, numeric limits —
live in sibling leaves and are re-exported here, so ``ouroboros.config`` remains
the one import surface for settings knowledge.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import sys
import time
from typing import Any, Optional, Sequence  # noqa: F401

from ouroboros.context_mode_compat import (
    normalize_and_persist_context_mode_compat, normalize_context_mode, owner_declared_low,
)
from ouroboros.platform_layer import pid_lock_acquire as _compat_pid_lock_acquire, pid_lock_release as _compat_pid_lock_release
from ouroboros.provider_models import compute_direct_review_models_fallback, local_only_review_route_env, migrate_model_value, review_model_uses_local as review_model_uses_local  # noqa: F401
from ouroboros.secret_masking import strip_masked_secrets
from ouroboros.settings_defaults import (
    ENDPOINT_AUTHORED_SETTINGS,  # noqa: F401
    FINALIZATION_GRACE_DEFAULT_SEC,  # noqa: F401
    OPENROUTER_DEFAULTS,  # noqa: F401
    OPENROUTER_REVIEW_DEFAULTS,  # noqa: F401
    OWNER_STOP_OUTER_CAP_SEC,  # noqa: F401
    PACING_INTERVAL_DEFAULT_SEC,  # noqa: F401
    RETIRED_SETTING_KEYS,  # noqa: F401
    SETTINGS_DEFAULTS,  # noqa: F401
    SETTINGS_KEYS_NOT_EXPORTED_TO_ENV,  # noqa: F401
    SUPERVISOR_LIVENESS_DEADLINE_DEFAULT_SEC,  # noqa: F401
    _DISK_AUTHORED_SETTINGS,  # noqa: F401
    settings_env_keys,  # noqa: F401
)
from ouroboros.settings_scales import (
    EFFORT_SCALE,  # noqa: F401
    PROMPT_CACHE_TTL_SCALE,  # noqa: F401
    VALID_RUNTIME_MODES,  # noqa: F401
    VALID_SAFETY_MODES,  # noqa: F401
    _RUNTIME_MODE_RANK,  # noqa: F401
    _SAFETY_MODE_RANK,  # noqa: F401
    clamp_effort_to,  # noqa: F401
    effort_one_step_down,  # noqa: F401
    effort_rank,  # noqa: F401
    normalize_runtime_mode,  # noqa: F401
    normalize_safety_mode,  # noqa: F401
    resolve_effort,  # noqa: F401
    resolve_prompt_cache_ttl,  # noqa: F401
)
from ouroboros.model_slots import (
    _LEGACY_SLOT_RENAMES,  # noqa: F401
    _main_model,  # noqa: F401
    _parse_model_list,  # noqa: F401
    get_consciousness_model,  # noqa: F401
    get_deep_self_review_model,  # noqa: F401
    get_fallback_models,  # noqa: F401
    get_heavy_model,  # noqa: F401
    get_image_input_mode,  # noqa: F401
    get_light_model,  # noqa: F401
    get_vision_model,  # noqa: F401
    migrate_legacy_slot_keys,  # noqa: F401
    parse_fallback_chain,  # noqa: F401
)
from ouroboros.review_model_routes import (
    _DIRECT_PROVIDER_REVIEW_RUNS,  # noqa: F401
    _exclusive_direct_remote_provider_env,  # noqa: F401
    adaptive_quorum,  # noqa: F401
    direct_provider_review_models_fallback,  # noqa: F401
    get_review_enforcement,  # noqa: F401
    get_review_models,  # noqa: F401
    get_scope_review_models,  # noqa: F401
)
from ouroboros.runtime_limits import (
    DELEGATE_WAIT_CEILING_SEC,  # noqa: F401
    DELEGATE_WAIT_WINDOW_MAX_SEC,  # noqa: F401
    MAX_ACTIVE_SUBAGENTS_HARD_CAP,  # noqa: F401
    _bounded_positive_int_setting,  # noqa: F401
    _clamped_number_setting,  # noqa: F401
    get_acceptance_reserve_pct,  # noqa: F401
    get_acceptance_review_est_sec,  # noqa: F401
    get_delegate_wait_max_sec,  # noqa: F401
    get_delegate_wait_sec,  # noqa: F401
    get_llm_transport_read_timeout_sec,  # noqa: F401
    get_max_active_subagents_per_root,  # noqa: F401
    get_max_subagent_depth,  # noqa: F401
    get_max_workers,  # noqa: F401
    get_pacing_interval_sec,  # noqa: F401
    get_per_call_timeout_ceiling_sec,  # noqa: F401
    get_plan_task_deadline_min_sec,  # noqa: F401
    get_post_task_evolution_budget_usd,  # noqa: F401
    get_restart_drain_max_sec,  # noqa: F401
    get_safety_call_timeout_sec,  # noqa: F401
    get_safety_max_tokens,  # noqa: F401
    get_search_code_wall_sec,  # noqa: F401
    get_supervisor_liveness_deadline_sec,  # noqa: F401
    get_task_abs_ceiling_sec,  # noqa: F401
    get_task_idle_timeout_sec,  # noqa: F401
    get_vision_caption_timeout_sec,  # noqa: F401
    get_websearch_timeout_sec,  # noqa: F401
)
from ouroboros.update_channels import UPDATE_SETTINGS_DEFAULTS, normalize_update_channel  # noqa: F401


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


# Claudexor control-plane contract, checked at handshake so an old daemon is a typed
# lane refusal rather than a mid-run schema surprise.
CLAUDEXOR_PROTOCOL_MAJOR: int = 3
# The TRANSPORT floor: the lowest engine that serves the READ-ONLY lane, which sends no `execution` block at all.
# 3.2.0 schema-accepts every field that lane does send (verified live: the body comes back with only the fake-root
# error, never a field error), and a read-only run is already scoped by Claudexor's ordinary envelope. Keeping the
# floor AT the oldest serving engine is the owner's explicit decision — it lets an older daemon keep read-only
# delegation instead of losing the lane; a floor set to the newest daemon anyone happens to run is not conservative,
# it is an outage (3.2.1 here refused the operator's own 3.2.0 engine and took read-only delegation down with it).
CLAUDEXOR_MIN_VERSION: str = "3.2.0"
# The MARKER floor: the oldest engine whose SCHEMA ACCEPTS `execution.delegated`, which is
# the only delegated-lane question a version can answer honestly. Measured: `RunExecution`
# is `.strict()` and has no `delegated` key below 3.3.0, so the field is a 400 (live against
# the running 3.2.0 daemon, which names `/execution/delegated` in `fieldErrors`) and the run
# never starts. Refusing here turns a certain failure into a typed one before a token is
# spent — the only work this number does. It cannot be a probe either: the marker is nested
# under `execution` while the catalog lists TOP-LEVEL keys, and the one behavioural probe is
# to send it, which on an engine that accepts it STARTS THE RUN.
#
# It is NOT the floor for a BOUNDARY existing. It used to be, pinned at 3.3.2 (macOS
# Seatbelt), and a version standing in for "a boundary was applied" lies in both directions:
# Claudexor's `docs/DELEGATED_CONFINEMENT.md` says the mechanism is macOS-only, so a build
# declares the same number on a host where it applies nothing — and a version describes a
# BUILD, never what THIS attempt did. That question goes to the attempt record
# (`gateways.claudexor.attempt_containment`), and a run reporting no mechanism is DISCLOSED,
# not refused: the child already holds a shell here, so the step to "shell plus token" does
# not buy a lane-wide refusal (AGENTS.md "Disclose instead of forbid"). Two gates, two
# questions, no overlap; bands: docs/DELEGATED_ADMISSION.md.
CLAUDEXOR_DELEGATED_MARKER_MIN_VERSION: str = "3.3.0"


# Boot-time runtime-mode baseline. Pinning the owner-selected mode after settings load stops an
# out-of-process settings edit from becoming the new baseline through a later load/save round-trip.
# The pin is exported via ``OUROBOROS_BOOT_RUNTIME_MODE`` so fresh subprocess imports inherit the
# same ratchet; a child can clobber only its own env, not the parent's in-memory pin.
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


def get_subagent_projects_root() -> str:
    """Durable root for genesis ("from scratch") subagent projects.

    Outside repo/ and data/. Unlike self_worktree checkouts, genesis projects are
    durable deliverables and are never age-pruned by the GC retention sweep."""
    raw = str(
        os.environ.get("OUROBOROS_SUBAGENT_PROJECTS_ROOT", "")
        or SETTINGS_DEFAULTS.get("OUROBOROS_SUBAGENT_PROJECTS_ROOT", "")
    ).strip()
    return raw or os.path.expanduser(os.path.join("~", "Ouroboros", "projects"))


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


def get_runtime_mode() -> str:
    """Return the configured runtime mode (light / advanced / pro)."""
    default_val = str(SETTINGS_DEFAULTS["OUROBOROS_RUNTIME_MODE"])
    if _BOOT_RUNTIME_MODE is not None:
        return normalize_runtime_mode(_BOOT_RUNTIME_MODE)
    inherited = _resolve_baseline_from_env()
    if inherited is not None:
        return normalize_runtime_mode(inherited)
    return normalize_runtime_mode(os.environ.get("OUROBOROS_RUNTIME_MODE", default_val) or default_val)


def get_safety_mode() -> str:
    """Return the owner-selected LLM-safety-supervisor coverage (full | light | off).

    Owner-only at the write surface (dropped from the agent-reachable /api/settings POST),
    so the agent cannot lower its own safety coverage. Deterministic registry sandbox,
    protected paths and light-mode guards run regardless (BIBLE P3: the LLM supervisor is a
    layer, not the floor)."""
    default_val = str(SETTINGS_DEFAULTS["OUROBOROS_SAFETY_MODE"])
    return normalize_safety_mode(os.environ.get("OUROBOROS_SAFETY_MODE", default_val) or default_val)


def get_context_mode() -> str:
    """The EFFECTIVE working-context mode (low | max) used by context sizing.

    Owner selection or an explicitly forwarded benchmark/operator value.  The P3 scope
    gate reads get_owner_context_mode instead so a bare env Low cannot author owner intent.
    No boot-pin: hot-applies on the next task. The key is dropped from the
    agent-reachable /api/settings POST (P1)."""
    default_val = str(SETTINGS_DEFAULTS["OUROBOROS_CONTEXT_MODE"])
    return normalize_context_mode(os.environ.get("OUROBOROS_CONTEXT_MODE", default_val) or default_val)


def get_owner_context_mode() -> str:
    """The OWNER-SELECTED context mode during the auto-Low compatibility window.

    Persistent auto-Low is retired, but a bare forwarded env ``low`` still lacks owner
    provenance and therefore keeps P3 at Max.  Only explicit ``low`` + tombstone ``false``
    means owner Low.  Raw persisted legacy ambiguity is normalized before environment
    projection, so this distinction is needed only for env-only benchmark/operator runs."""
    if get_context_mode() != "low":
        return "max"
    return "low" if owner_declared_low(os.environ.get("OUROBOROS_CONTEXT_MODE_AUTO_LOW", "")) else "max"


def _settings_file_value(key: str, default: str) -> str:
    """Read ONE persisted setting off disk, without normalizing the whole file. DISK ONLY, for EVERY caller: env
    is inherited and freely rewritten by any subprocess, so it can never be a ratchet's PREVIOUS value — reading it
    there turns ``max -> low`` into ``low -> low`` and the gate opens. Absent/corrupt = the fail-closed default."""
    if SETTINGS_PATH.exists():
        try:
            disk_settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(disk_settings, dict):
                value = disk_settings.get(key, default)
                return str(default if value is None or value == "" else value)
        except (OSError, json.JSONDecodeError):
            pass
    return default


def _guard_context_mode_lowering(settings: dict, *, allow_context_lowering: bool = False) -> None:
    """Refuse agent-reachable settings writes that lower the cognitive horizon.

    The mode may not step ``max -> low`` without the dedicated owner endpoint.  During
    the compatibility window, changing an ambiguous legacy Low marker to false is also
    refused unless the same write restores Max; that exact Max+false rewrite is the
    migration and cannot disable the P3 gate."""
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
    if (next_mode == "low" and owner_declared_low(settings["OUROBOROS_CONTEXT_MODE_AUTO_LOW"])
            and not owner_declared_low(previous_flag)):
        raise PermissionError(
            f"OUROBOROS_CONTEXT_MODE_AUTO_LOW clearing refused: {previous_flag or 'unknown'!r} -> 'false'. "
            "Authoring explicit owner-Low is owner-controlled; use the dedicated owner endpoint/UI/CLI."
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
        and not (
            k == "OUROBOROS_CONTEXT_MODE_AUTO_LOW"
            and _settings_file_value("OUROBOROS_CONTEXT_MODE", "")
        )
        and str(v) == str(SETTINGS_DEFAULTS.get(k, "")))}
    _guard_context_mode_lowering(prepared, allow_context_lowering=allow_context_lowering)
    _guard_safety_mode_lowering(prepared, allow_safety_lowering=allow_safety_lowering)
    return strip_masked_secrets(prepared, known_setting_keys=SETTINGS_DEFAULTS)


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
    # None means the lock was NOT taken: every WRITER must abort on it (`save_settings` raises
    # TimeoutError, `gateway.owner_settings` SettingsLockUnavailable) — writing anyway makes
    # "atomic" a claim the code does not keep. Only READS may proceed unlocked.
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


def normalize_settings_raw(raw: dict) -> dict:
    """THE raw-stage normalization every settings READER applies BEFORE defaults.

    A settings document on disk is written by whatever release the owner last used, so a
    reader's first job is to translate it into today's vocabulary: coerce every known key to
    the type its default declares, fold the deprecated per-subsystem retention keys into the
    unified one, drop the keys a release retired, promote the renamed model slots (and the
    singular scope-review pin), and repair secret placeholders. Every step exists to PRESERVE
    an owner customization written under a former key, which is why the order matters: the
    singular pin is promoted here, before any defaults merge supplies the plural that wins.

    Pure — it reads no file, writes no file, and consults no environment, so a reader can
    apply it and a read stays a read. It is the seam BECAUSE it was previously inline in
    ``load_settings``: the owner endpoints' reader merged defaults over the raw document
    instead, and then wrote that document back, turning a wrong read into a lost setting."""
    from ouroboros.retention import LEGACY_RETENTION_KEYS, pick_legacy_retention_seed

    loaded = {
        key: _coerce_setting_value(key, value) if key in SETTINGS_DEFAULTS else value
        for key, value in dict(raw or {}).items()
    }
    if "OUROBOROS_GC_RETENTION_DAYS" not in loaded:
        seed = pick_legacy_retention_seed(loaded.get)
        if seed is not None:
            loaded["OUROBOROS_GC_RETENTION_DAYS"] = seed
    for _legacy in LEGACY_RETENTION_KEYS:
        loaded.pop(_legacy, None)
    # Rename alias: a customized acceptance-pass count seeds the shared review-cycle knob
    # (cycles = passes + 1) unless the owner authored one, then the legacy key is dropped.
    _seed_review_cycles_from_legacy_passes(loaded)
    for _retired in RETIRED_SETTING_KEYS:
        loaded.pop(_retired, None)
    migrate_legacy_slot_keys(loaded)
    return strip_masked_secrets(loaded, known_setting_keys=SETTINGS_DEFAULTS)


def serialize_settings(settings: dict) -> str:
    """THE bytes a settings document is persisted as, for every writer that persists one.

    ``ouroboros.utils.atomic_write_json`` produces exactly this text, which is what lets the
    owner-endpoint writer keep its atomic helper while the config saver and the packaged
    bootstrap saver produce byte-identical output through the same function (pinned by
    tests/test_settings_read_seam.py). Without one serializer the writers disagreed on
    ``ensure_ascii`` alone, so the same document had two spellings on disk."""
    return json.dumps(settings, ensure_ascii=False, indent=2)


def _seed_review_cycles_from_legacy_passes(loaded: dict) -> None:
    """Migrate the retired acceptance-pass key into ``OUROBOROS_REVIEW_MAX_CYCLES`` (cycles =
    passes + 1) at LOAD: a runtime "is it customized?" test cannot tell a deliberate "2" from
    an untouched default, and left acceptance on the legacy number."""
    legacy = loaded.pop("OUROBOROS_ACCEPTANCE_MAX_IMPROVEMENT_PASSES", None)
    try:
        passes = int(str(legacy).strip()) if legacy is not None else 1
    except (TypeError, ValueError):
        return
    if passes != 1 and "OUROBOROS_REVIEW_MAX_CYCLES" not in loaded:  # 1 = shipped legacy default
        loaded["OUROBOROS_REVIEW_MAX_CYCLES"] = str(max(0, passes) + 1)


def load_settings() -> dict:
    fd = _acquire_settings_lock()
    try:
        return load_settings_lock_held(_settings_lock_held=fd is not None)
    finally:
        _release_settings_lock(fd)


def load_settings_lock_held(*, _settings_lock_held: bool = True) -> dict:
    """The same read, for a caller that ALREADY holds the settings lock. The lock is not
    re-entrant, so a nested ``load_settings()`` burns the full 2s timeout and then reads
    anyway; a write-path precondition needs the effective settings as of NOW, so it reads here.

    ``load_settings`` passes whether it actually acquired the best-effort read lock. Direct
    callers are lock-owning write preconditions, so the default remains true. A raw context
    compatibility migration is persisted only while that lock is held; the write contains
    the raw mapping plus the normalized pair, never a defaults-merged settings document."""
    loaded: dict = {}
    if SETTINGS_PATH.exists():
        try:
            raw = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                raw = normalize_and_persist_context_mode_compat(
                    raw,
                    settings_path=SETTINGS_PATH,
                    lock_held=_settings_lock_held,
                    guard_live_write=_guard_live_settings_write,
                )
                loaded = normalize_settings_raw(raw)
        except Exception:
            pass
    settings = dict(SETTINGS_DEFAULTS)
    settings.update(loaded)
    for key in SETTINGS_DEFAULTS:
        raw_env = os.environ.get(key)
        if raw_env is None or key in _DISK_AUTHORED_SETTINGS or key in ENDPOINT_AUTHORED_SETTINGS:  # DISK-authored
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


def save_settings(
    settings: dict,
    *,
    allow_elevation: bool = False,
    onboarding_safety_default: bool = False,
) -> None:
    """Persist settings and enforce owner-only mode ratchets.

    Elevation above the boot baseline is refused after initialization (``allow_elevation`` is then
    inert to agent-reachable subprocesses; production entry points must call
    ``initialize_runtime_mode_baseline`` before agent code). Context-mode lowering likewise
    requires the explicit owner path; the retired auto-Low key is an inert false tombstone.
    ``onboarding_safety_default`` is a NARROW boolean authorizing exactly one transition —
    a FRESH install (no settings file yet) authoring ``OUROBOROS_SAFETY_MODE="light"``."""
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
        # Baseline: the in-process pin, else the STRICTEST of the inherited env pin and disk. The env pin
        # exists only so a subprocess inherits the parent's ratchet, so it may only TIGHTEN — letting it
        # RAISE the baseline would be the caller-controlled "previous value" hole of _settings_file_value
        # on a fourth key (a subprocess exporting BOOT_RUNTIME_MODE=pro persisting its own elevation).
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
            from ouroboros.utils import replace_atomic
            tmp = SETTINGS_PATH.with_suffix(".tmp")
            tmp.write_text(serialize_settings(settings), encoding="utf-8")
            replace_atomic(str(tmp), str(SETTINGS_PATH))
        except OSError:
            SETTINGS_PATH.write_text(serialize_settings(settings), encoding="utf-8")
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


def apply_settings_to_env(settings: dict) -> None:
    """Push settings into environment variables for supervisor modules."""
    env_keys = settings_env_keys()
    # Disk-authored ratchets PROJECT ONLY WHAT THE FILE ACTUALLY SAYS: a default standing in for an absent
    # key is not an owner decision, so overwriting/popping the env entry would clobber a legitimately
    # forwarded value (harbor_installed_agent runs with NO settings.json; server_runner documents the same
    # "settings.json over env" clobber). Silence stays silent. ONE fail-closed exception: env may not author
    # the explicit-false owner-Low provenance claim, which would switch the BIBLE P3 scope gate off.
    unauthored = {k for k in _DISK_AUTHORED_SETTINGS if not _settings_file_value(k, "")}
    for k in env_keys:
        val = settings.get(k)
        if k in unauthored and not owner_declared_low(
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
