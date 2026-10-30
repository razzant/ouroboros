"""Saved Context, Effort, Subagents and model choice of Main and each registered Project room.

Projects copy Main's Context, Effort and Subagents once, at creation or migration. Reads stay
read-only; every settings writer freezes legacy rooms before changing Main. Registry writers take
their Main snapshot before the registry lock, never acquiring settings under it.
Actor lifetimes and room resolution belong to the task-settings owner.

The model choice is the room's alone and is never copied: absent, a room follows the configured
Main. Main's lives in the settings document (``MAIN_CHAT_MODEL_KEY``, never exported to an
environment, never a role slot), a Project's on its registry row (``model_choice``). What new owner
work snapshots from it, and how that runs, belongs to ``route_choice``.
"""
from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

from ouroboros.context_mode_compat import normalize_context_mode, owner_declared_low
from ouroboros.settings_scales import EFFORT_RANGE_KEYS, effort_range, normalize_subagents_mode

COGNITION_KEYS = ("OUROBOROS_CONTEXT_MODE", *EFFORT_RANGE_KEYS, "OUROBOROS_SUBAGENTS_MODE")
MAIN_CHAT_MODEL_KEY = "OUROBOROS_MAIN_CHAT_MODEL"
# The discriminated choice: the configured Main's reference (the default and the reset), the first
# configured Fallback's reference, or one explicit raw route. A reference is never the same model pinned.
MAIN_REFERENCE = {"kind": "main"}
MODEL_CHOICE_KINDS = ("main", "fallback", "model")
_MAX_ROUTE_CHARS = 400
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def normalize_model_choice(raw: Any) -> dict:
    """The canonical room choice; ``ValueError`` names what is wrong. Empty is Main's reference.

    A route is checked against the one route grammar only: the catalog suggests, it never admits, so an
    unknown or unlisted model stays the owner's choice. Locality is its own field (a `` (local)`` label
    becomes it); a subscription route is never local."""
    if raw is None or raw == "":
        return dict(MAIN_REFERENCE)
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            raise ValueError("The model choice must be a JSON object") from None
    if not isinstance(raw, Mapping):
        raise ValueError("The model choice must be an object with a kind")
    kind = raw.get("kind")
    if kind in ("main", "fallback"):
        return {"kind": kind}
    if kind != "model":
        raise ValueError("The model choice kind must be main, fallback or model")
    from ouroboros.provider_models import parse_claudexor_model, parse_model_route

    model, use_local = raw.get("model"), raw.get("use_local", False)
    if not isinstance(model, str) or not model.strip() or len(model) > _MAX_ROUTE_CHARS or _CONTROL.search(model):
        raise ValueError(f"A chosen model is a route of at most {_MAX_ROUTE_CHARS} printable characters")
    if not isinstance(use_local, bool):
        raise ValueError("use_local must be true or false")
    parts = parse_model_route(model)
    if parts.local:
        model, use_local = parts.model.strip(), True
    elif not parts.valid or not parts.model.strip():
        raise ValueError(f"{model.strip()!r} is not a model route")
    model = model.strip()
    if parts.kind == "claudexor":
        parse_claudexor_model(model)
        if use_local:
            raise ValueError("A subscription model never runs on the local lane")
    return {"kind": "model", "model": model, "use_local": use_local}


def model_choice_setting(choice: Mapping[str, Any]) -> str:
    """The settings spelling of Main's choice: empty for Main's reference, else canonical JSON."""
    current = normalize_model_choice(choice)
    return "" if current == MAIN_REFERENCE else json.dumps(current, ensure_ascii=False, sort_keys=True,
                                                           separators=(",", ":"))


def _tolerant_choice(raw: Any, where: str) -> dict:
    try:
        return normalize_model_choice(raw)
    except ValueError:
        import logging

        logging.getLogger(__name__).warning("Unreadable model choice in %s; the room follows Main", where)
        return dict(MAIN_REFERENCE)


def main_model_choice(settings: Mapping[str, Any] | None = None) -> dict:
    """Main's saved choice, read from the document; an unreadable value reads as the default it runs as."""
    if settings is None:
        from ouroboros import config

        settings = config.load_settings()
    return _tolerant_choice(settings.get(MAIN_CHAT_MODEL_KEY), "Main")


def project_model_choice(project: Mapping[str, Any]) -> dict:
    """A Project's saved choice; a row without one follows Main (it was never copied)."""
    return _tolerant_choice(project.get("model_choice"), f"Project {project.get('id')!r}")


def read_model_choice(drive_root: Any, project_id: str = "", *, settings: Mapping[str, Any] | None = None) -> dict:
    """The room's saved choice from its authority: the settings document for Main, the strict registry row
    for a Project (``ValueError``/``OSError`` when that row cannot be read or is no longer active)."""
    if not project_id:
        return main_model_choice(settings)
    from ouroboros import projects_registry as registry

    project = registry.get_reserved_project(drive_root, project_id, strict=True)
    if project is None or project.get("lifecycle") != registry.PROJECT_ACTIVE:
        raise ValueError(f"Project {project_id!r} is unavailable")
    return project_model_choice(project)


def update_project_model_choice(drive_root: Any, project_id: str, choice: Mapping[str, Any]) -> tuple[dict, dict]:
    """Save just this room's model choice under the registry's existing write lock."""
    from ouroboros import projects_registry as registry

    current = normalize_model_choice(choice)
    with registry._file_write_lock(registry._registry_path(drive_root)):
        data = registry._load(drive_root, strict=True, project_id=project_id)
        project = next((row for row in data["projects"] if row["id"] == project_id
                        and row.get("lifecycle") == registry.PROJECT_ACTIVE), None)
        if project is None:
            raise ValueError(f"Project {project_id!r} is unavailable")
        previous = project_model_choice(project)
        if current == MAIN_REFERENCE:
            project.pop("model_choice", None)
        else:
            project["model_choice"] = current
        registry._save(drive_root, data)
    return previous, current


def cognition_from_settings(settings: Mapping[str, Any]) -> dict:
    return {"context_mode": normalize_context_mode(settings.get("OUROBOROS_CONTEXT_MODE", "max")),
            "effort_range": effort_range(settings),
            "subagents_mode": normalize_subagents_mode(settings.get("OUROBOROS_SUBAGENTS_MODE"))}


def cognition_overrides(cognition: Mapping[str, Any]) -> dict:
    """The non-secret room values; snapshot callers retain only Context/Effort."""
    effort = cognition["effort_range"]
    return {"OUROBOROS_CONTEXT_MODE": cognition["context_mode"],
            "OUROBOROS_SUBAGENTS_MODE": normalize_subagents_mode(cognition.get("subagents_mode")),
            **{key: effort[field] for key, field in zip(EFFORT_RANGE_KEYS, ("min", "recommended", "max"))}}


def main_cognition(settings: Mapping[str, Any] | None = None) -> dict:
    from ouroboros import config
    from ouroboros.settings_integrity import SETTINGS_ENV_LOCK

    settings = config.load_settings() if settings is None else settings
    with SETTINGS_ENV_LOCK:
        projected = dict(os.environ)
        config.apply_settings_to_env(settings, environ=projected)
    cognition = cognition_from_settings(projected)
    # A retired auto-Low view is not an owner choice to copy into a new room.
    if cognition["context_mode"] == "low" and not owner_declared_low(projected.get("OUROBOROS_CONTEXT_MODE_AUTO_LOW")):
        cognition["context_mode"] = "max"
    return cognition


def project_cognition(project: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict:
    """Legacy rows use the pre-migration baseline; saved choices never follow Main."""
    saved = project.get("cognition")
    return (cognition_from_settings(cognition_overrides(saved)) if isinstance(saved, Mapping)
            else {**copy.deepcopy(dict(baseline)), "subagents_mode": "auto"})


def read_cognition(drive_root: Any, project_id: str = "", *, settings: Mapping[str, Any] | None = None) -> dict:
    baseline = main_cognition(settings)  # settings before registry: the write order, too
    if not project_id:
        return baseline
    from ouroboros.projects_registry import get_project

    project = get_project(drive_root, project_id, strict=True)
    if project is None:
        raise ValueError(f"Project {project_id!r} is unavailable")
    return project_cognition(project, baseline)


def _seed_rows(data: dict, baseline: Mapping[str, Any]) -> bool:
    changed = False
    for row in data["projects"]:
        if "cognition" not in row:
            row["cognition"] = {**copy.deepcopy(dict(baseline)), "subagents_mode": "auto"}
            changed = True
    return changed


def ensure_project_cognition(drive_root: Any, *, baseline: Mapping[str, Any] | None = None) -> None:
    """Freeze existing rooms before Main changes, or during ordinary boot reconcile."""
    from ouroboros import projects_registry as registry

    path = registry._registry_path(drive_root)
    if not path.exists():
        return
    baseline = main_cognition() if baseline is None else baseline
    with registry._file_write_lock(path):
        data = registry._load(drive_root, strict=True, project_id="")
        if _seed_rows(data, baseline):
            registry._save(drive_root, data)


def preserve_project_cognition(settings: Mapping[str, Any], settings_path: Path) -> None:
    """Settings-write prologue; the caller already owns the settings file lock."""
    from ouroboros import config

    current = main_cognition(config.load_settings_lock_held())
    if current != cognition_from_settings(settings):
        # The file's parent is its actual data root, including isolated readers
        # that replace SETTINGS_PATH without replacing the module's DATA_DIR.
        ensure_project_cognition(settings_path.parent, baseline=current)


def update_project_cognition(drive_root: Any, project_id: str, *, context_mode: str | None = None,
                             effort_range: Mapping[str, str] | None = None,
                             subagents_mode: str | None = None) -> tuple[dict, dict]:
    """Update just this room's decision under the registry's existing write lock."""
    from ouroboros import projects_registry as registry

    baseline = main_cognition()
    with registry._file_write_lock(registry._registry_path(drive_root)):
        data = registry._load(drive_root, strict=True, project_id=project_id)
        project = next((row for row in data["projects"] if row["id"] == project_id
                        and row.get("lifecycle") == registry.PROJECT_ACTIVE), None)
        if project is None:
            raise ValueError(f"Project {project_id!r} is unavailable")
        previous = project_cognition(project, baseline)
        current = copy.deepcopy(previous)
        if context_mode is not None:
            current["context_mode"] = context_mode
        if effort_range is not None:
            current["effort_range"] = dict(effort_range)
        if subagents_mode is not None:
            current["subagents_mode"] = subagents_mode
        project["cognition"] = current
        registry._save(drive_root, data)
    return previous, current


def request_project_id(body: Mapping[str, Any]) -> str:
    """Exact optional room identity; never sanitize a different id into this one."""
    from ouroboros.project_facts import explicit_project_id_ok

    value = body.get("project_id", "")
    if not isinstance(value, str) or (value and not explicit_project_id_ok(value)):
        raise ValueError("project_id must name an existing Project, or be empty for Main")
    return value
