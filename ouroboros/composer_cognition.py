"""Saved Context, Effort, Subagents and model choice of Main and each registered Project room.

A new Project copies Main's Effort and Subagents once and starts its Context in Max, whatever Main's Context
is (``new_room_cognition``; owner 15.11.2026, 3A: a room never inherits Main's temporary Low); a legacy row frozen
at migration keeps Main's Context and Effort, its Subagents Auto, and a recovered row its saved cognition or
Main's. Reads stay read-only; every settings writer freezes legacy rooms before changing Main.
Registry writers take their Main snapshot before the registry lock, never acquiring settings under it.
Actor lifetimes and room resolution belong to the task-settings owner.

The model new work starts on is each room's own. A Project keeps it on its registry row (``model_choice``;
absent, the room follows the configured Main), the Main chat in two settings of its own (``MAIN_CHAT_MODEL_KEY``,
``MAIN_CHAT_PROCESSING_KEY``) that no role reads. A new Project copies the Main chat's model and tier once
(``new_room_choice``). A copied model is the room's from then on and is marked ``model_choice_copied`` until its
owner chooses there: a copy is not their choice for that room. While the Main chat follows Main the copy
follows too, so nothing turns Main's own binding (its account, window and tier) into a carried route unasked.
Admission and the work's lead belong to ``route_choice``. The processing tier (``processing_choice``) is an
independent axis beside the model; absent, work keeps the default resolution (Main's tier for work that follows
Main, the default tier for a model of the room's own).

The Main chat's Ultrafast is its owner's consent to one destination's price, named as they were shown it: the
model, its locality and the subscription profile its new work is pinned to (``""``: Auto, an API connection or the
local model). Only the chat's confirmed write saves it (``gateway.owner_effort``), with one word for that
destination (no key and nothing made from one). A settings write that moves the destination or its API connection
lands with the tier Default (``keep_chat_ultrafast``), and a read finds it in force only while the word names where
new work would run now (``chat_ultrafast_route``): otherwise new work takes the ordinary mode, said as withheld.
Work admitted on it carries both the destination and, for API, the settings writer's random connection revision.
Only its own calls to that destination ask Ultrafast (``route_choice.task_processing``), independently of later
chip picks. A named profile follows its name and Auto its source's pool; neither identifies an individual account.
A changed or unproven API connection takes the ordinary tier without inheriting old consent.
A new room copies every other tier and never Ultrafast.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

from ouroboros.context_mode_compat import normalize_context_mode, owner_declared_low
from ouroboros.settings_scales import EFFORT_RANGE_KEYS, effort_range, normalize_subagents_mode

COGNITION_KEYS = ("OUROBOROS_CONTEXT_MODE", *EFFORT_RANGE_KEYS, "OUROBOROS_SUBAGENTS_MODE")
# The Main chat's own model (a route label, blank: follows Main) and tier (blank: Default) for new work there.
MAIN_CHAT_MODEL_KEY, MAIN_CHAT_PROCESSING_KEY = "OUROBOROS_MAIN_CHAT_MODEL", "OUROBOROS_MAIN_CHAT_PROCESSING"
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


def _tolerant_tier(raw: Any, where: str) -> str:
    from ouroboros.model_slots import normalize_call_tier

    try:
        return normalize_call_tier(raw)
    except ValueError:
        import logging

        logging.getLogger(__name__).warning("Unreadable processing choice in %s; Default", where)
        return ""


def project_model_choice(project: Mapping[str, Any]) -> dict:
    """A Project's saved choice; a row without one follows Main."""
    return _tolerant_choice(project.get("model_choice"), f"Project {project.get('id')!r}")


def project_processing_choice(project: Mapping[str, Any]) -> str:
    """A Project's own processing tier, or ``""`` (Default) for a row without a readable one."""
    return _tolerant_tier(project.get("processing_choice"), f"Project {project.get('id')!r}")


def main_chat_saved(settings: Mapping[str, Any]) -> tuple[dict, str, str]:
    """The Main chat's settings as saved: its model choice, its tier's word and the consent word an Ultrafast was
    saved with (``""`` for any other tier). Whether that Ultrafast is in force is ``main_chat_choice``'s answer."""
    route = str(settings.get(MAIN_CHAT_MODEL_KEY) or "").strip()
    # The stored local label is the choice's locality (``main_chat_route_setting``): read as such before the grammar's
    # length rule, so a local model whose name fills that rule reads back as it was saved.
    local = route.endswith(" (local)")
    choice = (_tolerant_choice({"kind": "model", "model": route.removesuffix(" (local)"), "use_local": local},
                               "the Main chat") if route else dict(MAIN_REFERENCE))
    saved = settings.get(MAIN_CHAT_PROCESSING_KEY)
    word, _, consent = saved.partition("@") if isinstance(saved, str) else (saved, "", "")
    tier = _tolerant_tier(word, "the Main chat")
    return choice, (tier if tier == "ultrafast" or not consent else ""), consent  # only Ultrafast is saved with a word


def main_chat_choice(settings: Mapping[str, Any] | None = None, *, withheld: bool = False) -> dict:
    """The Main chat's ``model_choice`` and ``processing_choice`` for new work there, read from the settings document
    as it is now (or from ``settings``, a read of it the caller already holds). Never Main the role's own binding.

    A saved Ultrafast answers only while its consent names where that work would run now
    (``gateway.owner_effort.chat_ultrafast_route``: the same model, locality and profile), and then names that
    destination (``processing_route``: ``[model, use_local, account]``) for the work's carrier; otherwise the tier
    reads Default, and ``withheld`` adds ``processing_withheld: "ultrafast"`` so the page can say why."""
    if settings is None:
        from ouroboros import config

        settings = config.load_settings()
    choice, tier, consent = main_chat_saved(settings)
    bound = None
    if tier == "ultrafast":
        from ouroboros.gateway.owner_effort import chat_ultrafast_route

        bound = chat_ultrafast_route(settings, choice, consent)
        if bound is None:
            return {"model_choice": choice, "processing_choice": "",
                    **({"processing_withheld": "ultrafast"} if withheld else {})}
    result = {"model_choice": choice, "processing_choice": tier}
    if bound:
        result["processing_route"], connection = bound
        if connection:
            result["processing_connection"] = connection
    return result


def chat_consent_word(destination: list, connection: str = "") -> str:
    """What Main chat Ultrafast was confirmed for, by name and (API only) random saved revision.

    Neither a credential, a credential derivative nor the account signed into a profile is part of the word."""
    names = list(destination[:3])
    if connection:
        names.append(connection)
    said = json.dumps(names, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(said.encode("utf-8")).hexdigest()[:24]


def main_chat_route_setting(choice: Mapping[str, Any]) -> str:
    """A normalized choice as ``MAIN_CHAT_MODEL_KEY`` stores it: its route labelled local, blank for Main's reference."""
    from ouroboros.model_slots import local_lane_label

    return local_lane_label(choice["model"], choice["use_local"]) if choice.get("kind") == "model" else ""


def new_room_choice(settings: Mapping[str, Any] | None = None) -> dict:
    """The row fields a new Project copies from the Main chat, once: its model (marked a copy) and its tier.
    A Main chat that follows Main adds none, so the new room follows Main as it always did. Ultrafast is never
    copied: it was confirmed for the Main chat's route, and a room's own tier names none."""
    saved = main_chat_choice(settings)
    choice, tier = saved["model_choice"], saved["processing_choice"]
    return {**({"model_choice": choice, "model_choice_copied": True} if choice["kind"] == "model" else {}),
            **({"processing_choice": tier} if tier and tier != "ultrafast" else {})}


def model_choice_binds(project: Mapping[str, Any] | None) -> bool:
    """Whether the room's model is its owner's choice for that room: a copy made at registration is not."""
    return (project is not None and project_model_choice(project)["kind"] == "model"
            and project.get("model_choice_copied") is not True)


def model_choice_token(project_id: str, choice: Mapping[str, Any]) -> str:
    """A saved model choice as one opaque word: its writer answers with it, and a message sent as new work names
    it as the condition its acceptance compares (``saved_choice``). It says nothing of the tier."""
    said = [str(project_id or ""), choice.get("kind"), choice.get("model", ""), bool(choice.get("use_local"))]
    return hashlib.sha256(json.dumps(said, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()[:24]


def saved_choice(drive_root: Any, project_id: str = "", *, project: Mapping[str, Any] | None = None) -> dict:
    """The model and tier new work starts on in the named Project, or in the Main chat, as saved now, with the
    model's token. ``ValueError`` when the Project is unavailable. ``project``: its strict registry row where the
    caller's own read already holds it, so the registry is not read a second time; it meets the same rule."""
    if project is None:
        project = read_room(drive_root, project_id)
    else:
        from ouroboros.projects_registry import PROJECT_ACTIVE

        if project.get("id") != project_id or project.get("lifecycle") != PROJECT_ACTIVE:
            raise ValueError(f"Project {project_id!r} is unavailable")
    saved = (main_chat_choice() if project is None else
             {"model_choice": project_model_choice(project), "processing_choice": project_processing_choice(project)})
    return {**saved, "choice_token": model_choice_token(project_id, saved["model_choice"])}


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
        project.pop("model_choice_copied", None)  # every write here is a choice made for this room
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
