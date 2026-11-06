"""Skill-declared MCP stdio servers: their derived launch, the Settings binding, and Connect tools.

A skill declares servers in its manifest (``mcp_servers``, parsed by the manifest contract).
Connecting is one owner decision (the Skills card, or ``skill_owner_action(action="connect_mcp")``
carrying an owner source): it writes exactly one ``MCP_SERVERS`` row per declaration — id,
title, ``enabled``, stdio, the declared call-lifetime flags and ``skill`` (the binding) — and
``MCP_ENABLED``, through the owner settings writer with the previewed settings digest. Nothing
else in settings changes, nothing is written when the state already matches, and a row of the
same id that is not bound to this skill is never overwritten.

A bound row stores no command: ``bind_config`` derives the runtime binary (the skill runtime
precedence of ``skill_exec``), the entry inside the reviewed payload, the skill directory as cwd
and the declared ``task_session``/``browser_policy`` at every discovery, Settings Test and
dispatch, and carries the payload revision into the config so catalog and task-session
identities follow the accepted bytes. A row's ``enabled`` reads as the loader reads it
(``mcp_client.setting_flag``; absent is off). While the skill is
missing, not accepted for its current bytes, off, short of grants or dependencies, in conflict,
or no longer declares the server, only that server is unavailable and no process starts; its
Settings row stays as the owner left it. Unbound MCP rows never reach this module.
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

CONNECT_KEYS = ("MCP_SERVERS", "MCP_ENABLED")
_DERIVED_KEYS = ("command", "args", "cwd")
_STALE_DOCUMENT = "The MCP settings changed while connecting, so nothing was written. Open Connect tools again."


def _on(value: Any) -> bool:
    from ouroboros.mcp_client import setting_flag

    return setting_flag(value)


def _derive_launch(skill: Any, decl: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    """``({command, args, cwd}, "")`` for one declaration, or ``(None, reason)``."""
    from ouroboros.skill_loader import _SKILL_DIR_CACHE_NAMES
    from ouroboros.tools.skill_exec import _resolve_runtime_binary

    if any(part in _SKILL_DIR_CACHE_NAMES for part in pathlib.PurePosixPath(decl["entry"]).parts):
        return None, f"its entry {decl['entry']!r} lies in a cache or dependency folder, outside the reviewed files"
    root = pathlib.Path(skill.skill_dir).resolve()
    entry = (root / decl["entry"]).resolve()
    if not entry.is_relative_to(root) or not entry.is_file():
        return None, f"its entry {decl['entry']!r} is not a file inside the skill"
    binary, why = _resolve_runtime_binary(decl["runtime"])
    if not binary:
        return None, f"no usable {decl['runtime']} runtime was found" + (f" ({why})" if why else "")
    return {"command": binary, "args": [str(entry)], "cwd": str(root)}, ""


def _not_ready(drive_root: pathlib.Path, skill: Any) -> List[str]:
    """The facts that keep this payload from running now; empty when it may run."""
    from ouroboros.skill_readiness import skill_readiness_for_execution

    if skill.load_error:
        return [f"load_error: {skill.load_error}"]
    if not (skill.manifest.is_script() or skill.manifest.is_extension()):
        return ["only script and extension skills run servers"]
    return list(skill_readiness_for_execution(drive_root, skill).blockers)


def _find(drive_root: pathlib.Path, name: str) -> Any:
    from ouroboros.config import get_skills_repo_path
    from ouroboros.skill_loader import find_skill

    return find_skill(drive_root, name, repo_path=get_skills_repo_path())


def bind_config(cfg: Any, drive_root: Optional[pathlib.Path] = None) -> Any:
    """The launchable form of one parsed MCP config; an unbound config is returned unchanged.

    A ready binding gets the derived ``command``/``args``/``cwd`` and ``skill_revision``; any
    other state gets them empty and ``skill_error`` saying why, so it can never launch."""
    if not getattr(cfg, "skill", ""):
        return cfg
    from ouroboros import config

    try:
        root = pathlib.Path(drive_root or config.DATA_DIR)
        skill = _find(root, cfg.skill)
        decl = next((item for item in skill.manifest.mcp_servers if item["id"] == cfg.id), None) if skill else None
        blockers = (["it is not installed"] if skill is None else
                    [f"it no longer declares server {cfg.id!r}"] if decl is None else _not_ready(root, skill))
        launch, why = _derive_launch(skill, decl) if not blockers else (None, "")
    except Exception as exc:  # noqa: BLE001 - an unreadable skill is an unavailable server, never a crash
        launch, blockers, why = None, [f"{type(exc).__name__}: {exc}"], ""
    if launch is None:
        reason = "; ".join(blockers) or why
        return dataclasses.replace(cfg, command="", args=[], cwd="", skill_revision="", skill_error=(
            f"MCP_SKILL_UNAVAILABLE: server {cfg.id!r} runs from skill {cfg.skill!r}, which cannot run it now "
            f"({reason}). Nothing was started. Its Settings row is kept; it works again once the skill is "
            "ready on its Skills card."))
    # The accepted declaration, not the saved row, decides the call lifetime: an update that adds browser_policy
    # is never launched unguarded from an older row.
    return dataclasses.replace(cfg, **launch, task_session=decl["task_session"], browser_policy=decl["browser_policy"],
                               skill_revision=skill.content_hash, skill_error="")


def _row(skill_name: str, decl: Dict[str, Any]) -> Dict[str, Any]:
    return {"id": decl["id"], "name": decl["title"], "enabled": True, "transport": "stdio", "skill": skill_name,
            "task_session": decl["task_session"], "browser_policy": decl["browser_policy"]}


def plan_connect(skill: Any, document: Dict[str, Any]) -> Dict[str, Any]:
    """What Connect would write over the authored settings ``document`` (a pure function of both)."""
    from ouroboros.mcp_client import raw_server_id

    current = document.get("MCP_SERVERS") if isinstance(document.get("MCP_SERVERS"), list) else []
    rows = [dict(item) if isinstance(item, dict) else item for item in current]
    declared = {decl["id"] for decl in skill.manifest.mcp_servers}
    servers = []
    for decl in skill.manifest.mcp_servers:
        wanted, claimants = _row(skill.name, decl), [i for i, item in enumerate(current) if raw_server_id(item) == decl["id"]]
        conflict = ""
        if not claimants:
            rows.append(wanted)
            change = "add"
        elif len(claimants) > 1 or current[claimants[0]].get("skill") != skill.name:
            owner = current[claimants[0]].get("skill") if len(claimants) == 1 else ""
            change, conflict = "conflict", (
                f"{len(claimants)} Settings rows already use server id {decl['id']!r}" if len(claimants) > 1 else
                f"Settings already has server id {decl['id']!r} " + (f"from skill {owner!r}" if owner else "added by hand"))
        else:
            row = {key: value for key, value in current[claimants[0]].items() if key not in _DERIVED_KEYS}
            row.update({key: wanted[key] for key in ("transport", "skill", "task_session", "browser_policy")}, enabled=True)
            was = current[claimants[0]]
            change = ("unchanged" if row == was else
                      "enable" if not _on(was.get("enabled")) and {**row, "enabled": was.get("enabled")} == was else "repair")
            rows[claimants[0]] = row
        servers.append({"id": decl["id"], "title": decl["title"], "change": change, "conflict": conflict,
                        "row": wanted if change == "add" else rows[claimants[0]] if change != "conflict" else None,
                        "task_session": decl["task_session"], "browser_policy": decl["browser_policy"]})
    mcp_on = _on(document.get("MCP_ENABLED"))
    others = [] if mcp_on else [
        {"id": raw_server_id(item), "name": str(item.get("name") or raw_server_id(item))}
        for item in current if isinstance(item, dict) and _on(item.get("enabled")) and raw_server_id(item) not in declared]
    return {"servers": servers, "mcp_enabled": mcp_on, "enables_mcp": not mcp_on, "other_servers": others,
            "requires_other_servers_confirmation": bool(others),
            "changed": not mcp_on or any(item["change"] not in {"unchanged", "conflict"} for item in servers),
            "conflict": any(item["change"] == "conflict" for item in servers), "_rows": rows}


def _refusal(code: str, error: str, status_code: int = 409, **facts: Any) -> Dict[str, Any]:
    return {"ok": False, "saved": False, "code": code, "error": error, "status_code": status_code, **facts}


def _selected(drive_root: pathlib.Path, skill_name: str, expected_content_hash: str = "") -> Tuple[Any, Dict[str, Any]]:
    skill = _find(drive_root, skill_name)
    if skill is None:
        return None, _refusal("not_found", "skill not found", 404)
    if not skill.manifest.mcp_servers:
        return None, _refusal("no_mcp_servers", "This skill declares no MCP servers.", 400)
    if expected_content_hash and skill.content_hash != expected_content_hash:
        return None, _refusal("skill_revision_changed", "The skill's files changed; open Connect tools again.")
    return skill, {}


def preview_connect(drive_root: pathlib.Path, skill_name: str) -> Dict[str, Any]:
    """Passive preview: the exact rows, what will run, which other servers start; writes nothing."""
    from ouroboros.gateway import owner_settings as owner

    skill, refusal = _selected(drive_root, skill_name)
    if refusal:
        return refusal
    digest = owner.settings_document_digest()  # before the read: a later change makes the confirm refuse
    try:
        plan = plan_connect(skill, owner._owner_settings_document(strict=True))
    except owner.SettingsPreconditionFailed as exc:  # a present settings file that does not read as a document
        return _refusal("settings_unreadable", str(exc))
    blockers = _not_ready(drive_root, skill)
    for item, decl in zip(plan["servers"], skill.manifest.mcp_servers):
        launch, why = _derive_launch(skill, decl) if not blockers else (None, "")
        item["executable"] = launch or {}
        if why:
            blockers.append(f"server {decl['id']!r}: {why}")
    plan.pop("_rows")
    return {"ok": True, "skill": skill.name, "content_hash": skill.content_hash, "ready": not blockers,
            "blockers": blockers, "settings_digest": digest,
            "next_action": "" if not blockers else "Finish this skill's setup on its Skills card (review, turn on, "
                           "access, packages) before connecting its tools.", **plan}


def connect(drive_root: pathlib.Path, skill_name: str, *, expected_content_hash: str, settings_digest: str,
            confirm_other_servers: bool = False) -> Dict[str, Any]:
    """Apply the previewed Connect: one owner settings write of ``CONNECT_KEYS``, or a typed refusal."""
    from ouroboros.gateway import owner_settings as owner

    if not settings_digest:
        return _refusal("settings_digest_required", "Confirm with the settings_digest of the Connect tools preview.", 400)
    skill, refusal = _selected(drive_root, skill_name, expected_content_hash)
    if refusal:
        return refusal
    blockers = _not_ready(drive_root, skill) + [why for decl in skill.manifest.mcp_servers
                                                for why in [_derive_launch(skill, decl)[1]] if why]
    if blockers:
        return _refusal("skill_not_ready", "This skill can't run its tools yet: " + "; ".join(blockers) + ".")
    try:
        with owner.settings_document_mutation():
            plan = plan_connect(skill, owner._owner_settings_document(strict=True))
            facts = {key: plan[key] for key in ("servers", "other_servers")}
            if plan["conflict"]:
                return _refusal("mcp_server_conflict", next(s["conflict"] for s in plan["servers"] if s["conflict"])
                                + "; nothing was changed. Remove or rename that row to connect this skill.", **facts)
            if plan["requires_other_servers_confirmation"] and not confirm_other_servers:
                return _refusal("other_servers_unconfirmed", "Turning MCP on also starts the other servers that are "
                                "on; confirm that too.", **facts)
            if not plan["changed"]:
                return {"ok": True, "saved": False, "changed": False, "skill": skill.name, "mcp_enabled": True, **facts}
            changes = {"MCP_SERVERS": plan["_rows"], "MCP_ENABLED": True}

            def still_planned(document: Dict[str, Any]) -> str:
                return "" if plan_connect(skill, document)["_rows"] == plan["_rows"] else _STALE_DOCUMENT

            receipt = owner._owner_update_settings(owner.fields_on_saved_document(changes, still_planned),
                                                   settings_digest, authored_keys=CONNECT_KEYS)
    except owner.SettingsPreconditionFailed as exc:
        return _refusal("settings_precondition_failed", str(exc))
    except (owner.SettingsLockUnavailable, owner.SettingsDocumentBusy) as exc:
        return _refusal("settings_busy", str(exc), 503)
    return {"ok": True, "saved": True, "changed": True, "skill": skill.name, "mcp_enabled": True,
            "changed_keys": sorted(receipt), **facts}


def card_facts(skill: Any, settings: Dict[str, Any], status_rows: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Each declared server's passive Skills-card state: configured here, listed tools, refusal."""
    from ouroboros.mcp_client import raw_server_id

    rows = [item for item in settings.get("MCP_SERVERS") or [] if isinstance(item, dict)]
    facts = []
    for decl in skill.manifest.mcp_servers:
        claimants = [item for item in rows if raw_server_id(item) == decl["id"]]
        bound = len(claimants) == 1 and claimants[0].get("skill") == skill.name
        status = status_rows.get(decl["id"], {}) if bound else {}
        facts.append({"id": decl["id"], "title": decl["title"], "conflict": bool(claimants) and not bound,
                      "configured": bound and _on(claimants[0].get("enabled")) and _on(settings.get("MCP_ENABLED")),
                      "tool_count": int(status.get("tool_count") or 0), "listed": bool(status.get("listed_at")),
                      "error": str(status.get("skill_error") or status.get("last_error") or "")})
    return facts


def lifecycle_settled(skill_name: str = "", *, event_queue: Any = None, reason: str = "skill_lifecycle") -> None:
    """A skill lifecycle job (Connect included) ended here, or a worker reported one: the lister (the server) re-reads
    settings, configures and lists what it has not listed yet, else lists a bound server whose payload moved; a
    worker sends the content-free ``mcp_configuration_changed`` report on its event queue instead. Fail-soft."""
    from ouroboros import mcp_client

    try:
        manager = mcp_client.get_manager()
        if manager.is_lister:
            from ouroboros.config import load_settings

            settings = load_settings()
            if not manager.listed(settings):
                manager.configuration_changed(settings, reason=reason)
            elif manager.rebind(skill=skill_name):
                manager.refresh_all_background(reason=reason)
        elif event_queue is not None:
            event_queue.put_nowait({"type": "mcp_configuration_changed", "reason": reason, "skill": skill_name})
    except Exception:
        log.warning("MCP reconfiguration after a change of skill %r failed", skill_name, exc_info=True)


__all__ = ["CONNECT_KEYS", "bind_config", "card_facts", "connect", "lifecycle_settled", "plan_connect",
           "preview_connect"]
