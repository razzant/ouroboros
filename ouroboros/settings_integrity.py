"""Pinned settings snapshots, private actor reads and live room choices.

Benchmark OUROBOROS_SETTINGS_SHA256 pins verify complete bytes and refuse
every writer. Public integrity names are re-exported by config.py.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import contextlib
import contextvars
import copy
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

SETTINGS_INTEGRITY_ENV = "OUROBOROS_SETTINGS_SHA256"
_TASK_SETTINGS = contextvars.ContextVar("ouroboros_task_settings", default=None)
# The environment ``config.load_settings`` overlays on the keys its document leaves unset: the process's own,
# unless a read binds another (``settings_read_environ``: a skill read, an extension child's).
_READ_ENVIRON = contextvars.ContextVar("ouroboros_settings_read_environ", default=None)
# Only capture/projection holds this lock, never a task's execution lifetime.
SETTINGS_ENV_LOCK = threading.RLock()


@dataclass(frozen=True, repr=False)
class TaskSettingsSnapshot:
    """Private in-memory views; document values and env absence are distinct."""

    settings: Mapping
    environ: Mapping
    actor_scope: tuple[Path, Mapping] | None = None


def _next_task_setting(key: str) -> bool:
    from ouroboros.settings_scales import IMMEDIATE_SETTINGS, RESTART_REQUIRED_SETTINGS

    return key not in IMMEDIATE_SETTINGS and key not in RESTART_REQUIRED_SETTINGS and key != "OUROBOROS_RUNTIME_MODE"


def _projected_keys() -> set[str]:
    from ouroboros.settings_defaults import RETIRED_COMMA_LIST_SETTING_KEYS, settings_env_keys
    from ouroboros.model_slots import _LEGACY_SLOT_RENAMES

    return (set(settings_env_keys()) | set(RETIRED_COMMA_LIST_SETTING_KEYS)
            | {old for old, _new in _LEGACY_SLOT_RENAMES} | {PROJECTED_PROPOSALS_ENV, PROJECTED_SETTINGS_ENV})


# The values this process tree projected (``config.apply_settings_to_env``) for the keys whose absence takes a
# proposal — the model slots and the never-configured factory catalog — handed down with the environment: a read
# of a key the document does not set tells this tree's own echo from an operator's variable, so an absent slot
# keeps following the configured access and an absent catalog its factory (owner 2A).
PROJECTED_PROPOSALS_ENV = "OUROBOROS_PROJECTED_PROPOSALS"


def projected_proposals(environ: Mapping) -> dict:
    """``{key: value}`` this process tree last projected into ``environ`` for a proposal key."""
    try:
        echo = json.loads(environ.get(PROJECTED_PROPOSALS_ENV) or "{}")
    except ValueError:
        return {}
    return echo if isinstance(echo, dict) else {}


# The digest of every settings value this process tree's projection left in the environment as its own, handed
# down beside the echo. The loader never reads it; a skill read does (``skill_environ``), so a value this tree mirrored
# from an earlier document, in this process or one it inherited from, never passes for an operator's variable there.
PROJECTED_SETTINGS_ENV = "OUROBOROS_PROJECTED_SETTINGS"


def _value_digest(value: str) -> str:
    # 64 bits tell our own value of a key from an operator's, and keep the inherited variable small.
    return hashlib.sha256(str(value).encode("utf-8", "surrogatepass")).hexdigest()[:16]


def projected_settings(environ: Mapping) -> dict:
    """``{key: digest}`` of the values this process tree last projected into ``environ`` from its settings."""
    try:
        record = json.loads(environ.get(PROJECTED_SETTINGS_ENV) or "{}")
    except ValueError:
        return {}
    return record if isinstance(record, dict) else {}


def operator_variables(environ: Mapping, keys) -> dict:
    """``{key: value}`` of an operator's own variables in ``environ`` for proposal ``keys``: present, not blank and not
    this tree's projection — what the loader lets stand in for a key the document does not set."""
    echo = projected_proposals(environ)
    return {key: environ[key] for key in keys if environ.get(key) and environ[key] != echo.get(key)}


def note_proposal_projection(environ, before: Mapping) -> None:
    """Record the proposal-key values a projection left in ``environ`` (``before``: their values before it),
    except where it kept or respelled an operator's own variable (``operator_variables`` of ``before``): our echo
    must never turn the operator's choice into a proposal the next read replaces.

    Every other projected value is ours by digest (``PROJECTED_SETTINGS_ENV``) when this projection wrote it, when
    it is what the settings document itself sets (a launcher or an earlier release may have mirrored it there with
    no record), or when an earlier projection's record already held it; any other variable stays the operator's."""
    from ouroboros.configured_subagents import SUBAGENTS_SETTING
    from ouroboros.server_runtime import MODEL_SLOT_KEYS
    from ouroboros.settings_defaults import settings_env_keys

    keys = (*MODEL_SLOT_KEYS, SUBAGENTS_SETTING)
    echo, operator = projected_proposals(environ), operator_variables(before, keys)
    for key in keys:
        after = environ.get(key)
        if after is None or key in operator or (after == before.get(key) and after != echo.get(key)):
            echo.pop(key, None)
        else:
            echo[key] = after
    environ[PROJECTED_PROPOSALS_ENV] = json.dumps(echo, sort_keys=True)
    ours, document, record = projected_settings(before), _document_keys(), {}
    for key in settings_env_keys():
        value = environ.get(key)
        if value is None:
            continue
        digest = _value_digest(value)
        if key in document or value != before.get(key) or ours.get(key) == digest:
            record[key] = digest
    environ[PROJECTED_SETTINGS_ENV] = json.dumps(record, sort_keys=True)


def _document_keys() -> set:
    """The keys the settings document on disk sets to a value, as every reader normalizes it; none when unreadable."""
    from ouroboros import config

    try:
        raw = read_settings_json_verified(config.SETTINGS_PATH)
        document = config.normalize_settings_raw(raw) if isinstance(raw, dict) else {}
    except Exception:
        return set()
    return {key for key, value in document.items() if value not in (None, "")}


@contextlib.contextmanager
def task_settings_scope(snapshot):
    """Bind one task's settings in memory only; concurrent tasks keep their own view."""
    token = _TASK_SETTINGS.set(snapshot)
    try:
        yield
    finally:
        _TASK_SETTINGS.reset(token)


def copy_task_settings_context(context) -> None:
    """Carry settings through context transfers that intentionally omit Main call state."""
    context.run(_TASK_SETTINGS.set, _TASK_SETTINGS.get())


def runtime_setting(key: str, default=None):
    """Environment-shaped runtime read; absence in the snapshot stays absent.

    A route key (``route_choice.ROUTE_SETTING_KEYS``: model slots, accounts, provider keys) reads the
    settings document as it is now, in a task or not; every other next-task key keeps the task's snapshot."""
    from ouroboros.route_choice import ROUTE_SETTING_KEYS, live_setting

    if key in ROUTE_SETTING_KEYS:
        served, value = live_setting(key, default)
        if served:
            return value
    snapshot = _TASK_SETTINGS.get()
    if snapshot is not None and _next_task_setting(key):
        return snapshot.environ.get(key, default)
    return os.environ.get(key, default)


def runtime_environ() -> dict[str, str]:
    """Explicit child environment with this task's next-task settings overlaid."""
    with SETTINGS_ENV_LOCK:
        env = dict(os.environ)
    # Launcher authority belongs only to the launcher-owned server process. A
    # child shell/server (including an external-workspace test fixture) must not
    # inherit it and gain permission to run destructive managed bootstrap
    # against its own checkout.
    env.pop("OUROBOROS_MANAGED_BY_LAUNCHER", None)
    env.pop("OUROBOROS_MANAGED_REPO_DIR", None)
    snapshot = _TASK_SETTINGS.get()
    if snapshot is not None:
        for key in _projected_keys() | snapshot.settings.keys():
            if _next_task_setting(key):
                if key not in snapshot.environ:
                    env.pop(key, None)
                else:
                    env[key] = snapshot.environ[key]
    from ouroboros.route_choice import live_environ_overlay

    live_environ_overlay(env)  # a model call in a child (vision) uses the current model and key, never a cleared one
    return env


def _actor_project(scope: tuple[Path, Mapping]) -> str:
    """Current room from lineage or admitted facts, never a working directory."""
    from ouroboros.project_admission import reserved_project_for_chat
    from ouroboros.projects_registry import RoomResolver, get_reserved_project

    root, actor = scope
    if actor.get("presence"):
        return ""
    chat = RoomResolver(root, strict=True)(actor.get("id"), actor.get("parent_task_id"), actor.get("root_task_id"))
    if chat:
        return reserved_project_for_chat(root, chat)["id"]
    project = get_reserved_project(root, actor.get("project_id"), strict=True)
    if project:
        return project["id"]
    return reserved_project_for_chat(root, actor["chat_id"]).get("id", "") if actor.get("chat_id") else ""


def _with_cognition(snapshot: TaskSettingsSnapshot, cognition: Mapping) -> TaskSettingsSnapshot:
    from ouroboros.composer_cognition import cognition_overrides
    from ouroboros.context_mode_compat import VALID_CONTEXT_MODES
    from ouroboros.settings_scales import EFFORT_SCALE, effort_rank, VALID_SUBAGENTS_MODES

    effort = cognition.get("effort_range")
    if (cognition.get("subagents_mode", "auto") not in VALID_SUBAGENTS_MODES
            or cognition.get("context_mode") not in VALID_CONTEXT_MODES or not isinstance(effort, Mapping)
            or any(effort.get(key) not in EFFORT_SCALE for key in ("min", "recommended", "max"))
            or effort_rank(effort["min"]) > effort_rank(effort["max"])):
        raise ValueError("Saved actor cognition is invalid")
    values = cognition_overrides(cognition)
    values.pop("OUROBOROS_SUBAGENTS_MODE")  # this axis is live, never restored from an actor snapshot
    values["OUROBOROS_CONTEXT_MODE_AUTO_LOW"] = cognition.get("context_mode_auto_low", "false")
    return replace(snapshot, settings=MappingProxyType({**snapshot.settings, **values}),
                   environ=MappingProxyType({**snapshot.environ, **values}))


def _actor_scope(task: Mapping, drive_root: Path) -> tuple[Path, Mapping]:
    from ouroboros.consciousness_authority import is_consciousness_origin
    from ouroboros.dialogue_provenance import presence_metadata_binding

    metadata = task.get("metadata") if isinstance(task.get("metadata"), Mapping) else {}
    actor = {key: task.get(key) or metadata.get(key) for key in (
        "id", "parent_task_id", "root_task_id", "project_id", "chat_id")}
    actor["presence"] = bool(task.get("_presence_turn") or presence_metadata_binding(metadata) is not None)
    actor["autonomous"] = bool(task.get("type") == "evolution" or metadata.get("evolution_root_task_id")
                               or metadata.get("evolution_transaction") or is_consciousness_origin(metadata)
                               or metadata.get("work_kind") == "consciousness" or actor["presence"])
    return (Path(task.get("budget_drive_root") or metadata.get("budget_drive_root") or drive_root),
            MappingProxyType(actor))


def actor_settings_snapshot(snapshot: TaskSettingsSnapshot, task: Mapping, drive_root: Path) -> TaskSettingsSnapshot:
    """Capture this room privately; a new subordinate resolves its room again."""
    from ouroboros.composer_cognition import read_cognition

    scope = _actor_scope(task, drive_root)
    snapshot = replace(snapshot, actor_scope=scope)
    project_id = _actor_project(scope)
    if not project_id:
        return snapshot  # preserve Main's projected values and legacy Low provenance exactly
    return _with_cognition(snapshot, read_cognition(scope[0], project_id, settings=snapshot.environ))


def captured_cognition() -> dict | None:
    """Composer values and any legacy Low provenance, never credentials."""
    from ouroboros.composer_cognition import cognition_from_settings
    from ouroboros.context_mode_compat import owner_declared_low

    snapshot = _TASK_SETTINGS.get()
    if snapshot is None:
        return None
    cognition = cognition_from_settings(snapshot.environ)
    cognition.pop("subagents_mode")
    marker = snapshot.environ.get("OUROBOROS_CONTEXT_MODE_AUTO_LOW", "")
    if cognition["context_mode"] == "low" and not owner_declared_low(marker):
        cognition["context_mode_auto_low"] = marker
    return cognition


def restore_cognition(cognition: Mapping | None) -> None:
    """Replace the bound private view before rebuilding a returning actor's request.

    Legacy checkpoints have no choice recorded and retain their existing fallback.
    The enclosing task_settings_scope still resets the context at task exit.
    """
    snapshot = _TASK_SETTINGS.get()
    if cognition is not None and snapshot is not None:
        _TASK_SETTINGS.set(_with_cognition(snapshot, cognition))


def live_effort_range() -> dict[str, str]:
    """Current own-room bounds for a NEW review/delegate; the caller keeps its snapshot."""
    from ouroboros import config
    from ouroboros.composer_cognition import read_cognition
    from ouroboros.settings_scales import effort_range

    settings = config.load_settings()
    snapshot = _TASK_SETTINGS.get()
    if snapshot is None or snapshot.actor_scope is None:
        return effort_range(settings)
    root, _actor = snapshot.actor_scope
    return read_cognition(root, _actor_project(snapshot.actor_scope), settings=settings)["effort_range"]


def live_subagents_mode(*, task: Mapping | None = None, drive_root: Path | None = None,
                        settings: Mapping | None = None) -> str:
    """Live room choice from explicit or bound actor facts; autonomous work keeps its controls."""
    from ouroboros import config
    from ouroboros.composer_cognition import read_cognition

    settings = config.load_settings() if settings is None else settings
    snapshot = _TASK_SETTINGS.get()
    scope = _actor_scope(task, drive_root or config.DATA_DIR) if task is not None else (snapshot.actor_scope if snapshot else None)
    if scope is None:
        return config.normalize_subagents_mode(settings.get("OUROBOROS_SUBAGENTS_MODE"))
    if scope[1].get("autonomous"):
        return "auto"
    return read_cognition(scope[0], _actor_project(scope), settings=settings)["subagents_mode"]


def record_subagents_admission(task: dict, drive_root: Path, *, settings: Mapping | None = None,
                              project: Mapping | None = None) -> None:
    """Root telemetry. Queue callers supply settings and the admitted row (None for Main), without more IO."""
    if settings is not None:
        # Restate effective_subagents_mode from admitted facts, without IO under the queue lock.
        from ouroboros.settings_scales import normalize_subagents_mode
        from ouroboros.subagent_runtime import subagents_availability

        cognition = project.get("cognition") if project is not None else None
        mode = ((cognition if isinstance(cognition, Mapping) else {}).get("subagents_mode") if project is not None
                else settings.get("OUROBOROS_SUBAGENTS_MODE"))
        mode = "auto" if _actor_scope(task, drive_root)[1].get("autonomous") else normalize_subagents_mode(mode)
        mode = mode if subagents_availability(settings) == "available" else "off"
    else:
        # Direct turns have no queue admission row. Resolve their accepted room,
        # excluding unstarted lineage so the pre-start storage wait retains ownership.
        lineage = {"id", "parent_task_id", "root_task_id"}
        room = {key: value for key, value in task.items() if key not in lineage}
        room["metadata"] = {key: value for key, value in (task.get("metadata") or {}).items() if key not in lineage}
        mode = effective_subagents_mode(task=room, drive_root=drive_root)
    task.setdefault("metadata", {})["subagents_mode"] = mode


def room_subagents_mode_applies() -> bool:
    """Consciousness and Evolve keep their own controls, including their descendants."""
    snapshot = _TASK_SETTINGS.get()
    return not (snapshot and snapshot.actor_scope and snapshot.actor_scope[1].get("autonomous"))


def actor_subagents_availability() -> str | None:
    """Live Settings gate for a bound actor; pure catalog resolvers keep their supplied settings."""
    from ouroboros import config
    from ouroboros.subagent_runtime import subagents_availability

    snapshot = _TASK_SETTINGS.get()
    return subagents_availability(config.load_settings()) if snapshot and snapshot.actor_scope else None


def effective_subagents_mode(*, task: Mapping | None = None, drive_root: Path | None = None) -> str:
    """Settings can withhold the catalog without changing the saved room choice."""
    from ouroboros import config
    from ouroboros.subagent_runtime import subagents_availability

    if subagents_availability(config.load_settings()) != "available":
        return "off"
    return live_subagents_mode(task=task, drive_root=drive_root)


def read_environ() -> Mapping:
    """The environment ``config.load_settings`` overlays now: the process's, or the one a read binds."""
    environ = _READ_ENVIRON.get()
    return os.environ if environ is None else environ


@contextlib.contextmanager
def settings_read_environ(environ: Mapping):
    """Bind the environment the settings loader overlays for reads in this context only."""
    token = _READ_ENVIRON.set(environ)
    try:
        yield
    finally:
        _READ_ENVIRON.reset(token)


def skill_environ() -> dict:
    """The environment a skill read lets stand in for a key its document leaves unset: the task's own (or, outside
    a task, this process's) without the values this process tree mirrored there from its settings document
    (``projected_settings``) — the operator's own variables only."""
    snapshot = _TASK_SETTINGS.get()
    if snapshot is None:
        with SETTINGS_ENV_LOCK:
            source = dict(os.environ)
    else:
        source = dict(snapshot.environ)
    ours = projected_settings(source)
    return {key: value for key, value in source.items() if key not in ours or ours[key] != _value_digest(value)}


def skill_settings(settings_reader=None) -> dict:
    """What a skill reads: the settings document as it reads now, in a task or not (owner choice A, #1122).

    The one reader behind a skill's ``PluginAPI.get_settings``, its scripts' and companions' environment and the
    funded-credential disclosure of its handlers; grants and the manifest allowlist filter it afterwards; the
    task's review reads keep its snapshot while its model, account and key reads share this view
    (``route_choice``, ``runtime_settings``). The loader lets the
    environment stand in for a key the document leaves unset, and a long-lived process still holds what earlier
    projections mirrored there — this task's start, an earlier task's, the server's it inherited — so a key the owner
    cleared or deleted since would come back. The read overlays ``skill_environ`` instead: our mirrored values never
    stand in, an operator's own variable still does. Nothing here touches the process environment."""
    from ouroboros import config

    with settings_read_environ(skill_environ()):
        return dict((settings_reader or config.load_settings)() or {})


def runtime_settings(*, settings_reader=None) -> dict:
    """Runtime document view; owner writers continue using config.load_settings."""
    from ouroboros import config

    settings = dict((settings_reader or config.load_settings)() or {})
    snapshot = _TASK_SETTINGS.get()
    if snapshot is not None:
        for key in settings.keys() | snapshot.settings.keys():
            if not _next_task_setting(key):
                continue
            if key not in snapshot.settings:
                settings.pop(key, None)
            else:
                settings[key] = copy.deepcopy(snapshot.settings[key])
    from ouroboros.route_choice import live_settings_overlay

    live_settings_overlay(settings)  # route keys read the document as it is now (route_choice)
    return settings


def task_settings_snapshot(settings: dict, environ: dict) -> TaskSettingsSnapshot:
    """Keep document-only values and exact projected presence without serializing either."""
    return TaskSettingsSnapshot(
        MappingProxyType(copy.deepcopy(settings)), MappingProxyType(dict(environ)))


class SettingsIntegrityError(RuntimeError):
    """The settings snapshot changed or became unreadable under a strict pin."""


def guard_settings_snapshot_mutation() -> None:
    """Refuse every settings writer while a benchmark snapshot is pinned."""
    if os.environ.get(SETTINGS_INTEGRITY_ENV):
        raise SettingsIntegrityError("strict isolated settings snapshot is immutable")


def guard_live_settings_write(settings_path: Path, home: Path) -> None:
    """Every settings-write precondition: the snapshot pin, then the pytest
    guard on the LIVE settings file (an isolated test root passes through)."""
    guard_settings_snapshot_mutation()
    if os.environ.get("OUROBOROS_ALLOW_LIVE_DATA_TESTS") == "1":
        return
    try:
        live_settings = settings_path.resolve(strict=False) == (
            home / "Ouroboros" / "data" / "settings.json"
        ).resolve(strict=False)
    except OSError:
        live_settings = False
    if ("PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules) and live_settings:
        raise RuntimeError(
            "Refusing to write live Ouroboros settings.json from pytest. "
            "Set OUROBOROS_SETTINGS_PATH/OUROBOROS_DATA_DIR to a temp path, "
            "or OUROBOROS_ALLOW_LIVE_DATA_TESTS=1 for an explicit live-data test."
        )


def expected_settings_sha256() -> str:
    value = str(os.environ.get(SETTINGS_INTEGRITY_ENV, "") or "").strip().lower()
    if value and (len(value) != 64 or any(char not in "0123456789abcdef" for char in value)):
        raise SettingsIntegrityError("settings integrity digest is malformed")
    return value


def read_settings_bytes_verified(settings_path: Path) -> bytes | None:
    """Read one stable file descriptor and verify its complete byte stream."""
    expected = expected_settings_sha256()
    try:
        with settings_path.open("rb") as handle:
            raw = handle.read()
    except FileNotFoundError:
        if expected:
            raise SettingsIntegrityError("settings snapshot is missing") from None
        return None
    except OSError as exc:
        if expected:
            raise SettingsIntegrityError("settings snapshot is unreadable") from exc
        return None
    if expected and hashlib.sha256(raw).hexdigest() != expected:
        raise SettingsIntegrityError("settings snapshot changed")
    return raw


def read_settings_json_verified(settings_path: Path, *, behind_writer: bool = False):
    """Decode the verified snapshot; under a pin a decode failure is typed.

    Returns the parsed JSON value (any type) or ``None`` for an absent or —
    without a pin — unreadable/undecodable file. One helper so the two
    config.py read paths cannot drift apart in their pin semantics. Every
    writer replaces the document whole, except ``save_settings``' fallback for a
    filesystem that cannot rename, which truncates and writes in place under the
    settings lock: a prefix of one JSON object never decodes, so a lock-free read
    (``behind_writer``) that cannot decode reads once more behind that lock. So does
    one that cannot open a present file (Windows: an open meeting a writer's replace).
    """
    raw_bytes = read_settings_bytes_verified(settings_path)
    if raw_bytes is None:
        return _read_behind_the_writer(settings_path) if behind_writer and settings_path.exists() else None
    try:
        return json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        if expected_settings_sha256():
            raise SettingsIntegrityError("settings snapshot is unreadable") from exc
        return _read_behind_the_writer(settings_path) if behind_writer else None


def _read_behind_the_writer(settings_path: Path):
    """The same read once more, behind the lock a writer holds until its bytes are complete."""
    from ouroboros import config

    fd = config._acquire_settings_lock()
    try:
        return read_settings_json_verified(settings_path)
    finally:
        config._release_settings_lock(fd)


def verify_settings_integrity(settings_path: Path) -> str | None:
    """Verify the strict child pin, returning the observed digest when present."""
    raw = read_settings_bytes_verified(settings_path)
    return hashlib.sha256(raw).hexdigest() if raw is not None else None
