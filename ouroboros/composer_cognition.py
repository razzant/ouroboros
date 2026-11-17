"""Saved Context, Effort, Subagents and model choice of Main and each registered Project room.

A new Project copies Main's Effort and Subagents once and starts its Context in Max, whatever Main's Context
is (``new_room_cognition``; owner 15.11.2026, 3A: a room never inherits Main's temporary Low); a legacy row frozen
at migration keeps Main's Context and Effort, its Subagents Auto, and a recovered row its saved cognition or
Main's. Reads stay read-only; every settings writer freezes legacy rooms before changing Main.
Registry writers take their Main snapshot before the registry lock, never acquiring settings under it.
Actor lifetimes and room resolution belong to the task-settings owner.

The model choice is the room's alone and is never copied: absent, a room follows the configured
Main. A Project keeps its choice on its registry row (``model_choice``); Main chat writes
Main itself. Admission and the work's lead belong to ``route_choice``. The room's processing tier
(``processing_choice``: standard, fast or economy) is an independent axis on the same row, also never
copied; absent, the room's work keeps the default resolution (Main's tier for work that follows Main,
the default tier for a model of the room's own).
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
# A room follows Main (the default and reset) or chooses one route; a reference is never a pin.
MAIN_REFERENCE = {"kind": "main"}
MODEL_CHOICE_KINDS = ("main", "model")
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
    if kind == "main":
        return {"kind": kind}
    if kind != "model":
        raise ValueError("The model choice kind must be main or model")
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


def _tolerant_choice(raw: Any, where: str) -> dict:
    try:
        return normalize_model_choice(raw)
    except ValueError:
        import logging

        logging.getLogger(__name__).warning("Unreadable model choice in %s; the room follows Main", where)
        return dict(MAIN_REFERENCE)


def project_model_choice(project: Mapping[str, Any]) -> dict:
    """A Project's saved choice; a row without one follows Main (it was never copied)."""
    return _tolerant_choice(project.get("model_choice"), f"Project {project.get('id')!r}")


def project_processing_choice(project: Mapping[str, Any]) -> str:
    """A Project's own processing tier, or ``""`` (Default) for a row without a readable one."""
    from ouroboros.model_slots import normalize_call_tier

    try:
        return normalize_call_tier(project.get("processing_choice"))
    except ValueError:
        import logging

        logging.getLogger(__name__).warning("Unreadable processing choice in Project %r; Default", project.get("id"))
        return ""


def read_room(drive_root: Any, project_id: str = "") -> Mapping[str, Any] | None:
    """The active Project's strict registry row, or None for work without a room (Main)."""
    if not project_id:
        return None
    from ouroboros import projects_registry as registry

    project = registry.get_reserved_project(drive_root, project_id, strict=True)
    if project is None or project.get("lifecycle") != registry.PROJECT_ACTIVE:
        raise ValueError(f"Project {project_id!r} is unavailable")
    return project


def read_model_choice(drive_root: Any, project_id: str = "") -> dict:
    """The Project's strict registry choice, or a recorded follow for work without a room."""
    project = read_room(drive_root, project_id)
    return project_model_choice(project) if project is not None else dict(MAIN_REFERENCE)


def _update_room(drive_root: Any, project_id: str, change) -> Any:
    """Change one active row under the registry's existing write lock; ``change(row)`` returns its answer."""
    from ouroboros import projects_registry as registry

    with registry._file_write_lock(registry._registry_path(drive_root)):
        data = registry._load(drive_root, strict=True, project_id=project_id)
        project = next((row for row in data["projects"] if row["id"] == project_id
                        and row.get("lifecycle") == registry.PROJECT_ACTIVE), None)
        if project is None:
            raise ValueError(f"Project {project_id!r} is unavailable")
        answer = change(project)
        registry._save(drive_root, data)
    return answer


def update_project_model_choice(drive_root: Any, project_id: str, choice: Mapping[str, Any]) -> tuple[dict, dict]:
    """Save just this room's model choice; its processing tier stays."""
    current = normalize_model_choice(choice)

    def change(project: dict) -> tuple[dict, dict]:
        previous = project_model_choice(project)
        if current == MAIN_REFERENCE:
            project.pop("model_choice", None)
        else:
            project["model_choice"] = current
        return previous, current
    return _update_room(drive_root, project_id, change)


def update_project_processing_choice(drive_root: Any, project_id: str, tier: Any) -> tuple[str, str]:
    """Save just this room's processing tier (``""``: Default); its model choice stays."""
    from ouroboros.model_slots import normalize_call_tier

    current = normalize_call_tier(tier)

    def change(project: dict) -> tuple[str, str]:
        previous = project_processing_choice(project)
        if current:
            project["processing_choice"] = current
        else:
            project.pop("processing_choice", None)
        return previous, current
    return _update_room(drive_root, project_id, change)


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


def new_room_cognition(baseline: Mapping[str, Any]) -> dict:
    """A new room's cognition: Main's Effort and Subagents, its Context in Max (owner 15.11.2026, 3A)."""
    return {**copy.deepcopy(dict(baseline)), "context_mode": "max"}


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
    baseline = main_cognition()

    def change(project: dict) -> tuple[dict, dict]:
        previous = project_cognition(project, baseline)
        current = copy.deepcopy(previous)
        if context_mode is not None:
            current["context_mode"] = context_mode
        if effort_range is not None:
            current["effort_range"] = dict(effort_range)
        if subagents_mode is not None:
            current["subagents_mode"] = subagents_mode
        project["cognition"] = current
        return previous, current
    return _update_room(drive_root, project_id, change)


def request_project_id(body: Mapping[str, Any]) -> str:
    """Exact optional room identity; never sanitize a different id into this one."""
    from ouroboros.project_facts import explicit_project_id_ok

    value = body.get("project_id", "")
    if not isinstance(value, str) or (value and not explicit_project_id_ok(value)):
        raise ValueError("project_id must name an existing Project, or be empty for Main")
    return value
