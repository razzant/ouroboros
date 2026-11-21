"""Skill-declared MCP stdio servers: their derived launch, the Settings binding, and Connect tools.

The manifest contract combines a skill's stdio server declarations into ``manifest.mcp_servers``.
Connecting is one owner decision (the Skills card, or ``skill_owner_action(action="connect_mcp")``
carrying an owner source): it writes exactly one ``MCP_SERVERS`` row per declaration — id,
title, ``enabled``, stdio, the declared call-lifetime flags and ``skill`` (the binding) — and
``MCP_ENABLED``, through the owner settings writer with the previewed settings digest. Nothing
else in settings changes, nothing is written when the state already matches, and a row of the
same id that is not bound to this skill is never overwritten.

A bound row stores no command: ``bind_config`` derives the runtime binary (Node uses the skill
runtime precedence of ``skill_exec``; agent_python uses the agent interpreter), the entry inside the
reviewed payload, the skill directory as cwd and the declared ``task_session``/``browser_policy``
at every discovery, Settings Test and dispatch, and carries the payload revision into the config
so catalog and task-session identities follow the accepted bytes. A row's ``enabled`` reads as the loader reads it
(``mcp_client.setting_flag``; absent is off). Each real start (``work_start``) gets the environment skill_exec gives
the skill's scripts; a managed domain runs Node servers from verified work copies in its container and refuses
agent_python servers; binding and preview stay passive. While the skill is missing, not accepted for
its current bytes, off, short of grants or dependencies, in conflict, an extension the domain cannot
start, or no longer declares the server, only that server is unavailable and no process starts
anywhere; its Settings row stays as the owner left it. A deployment policy's refusal of the write is
typed (``managed_by_deployment``). Unbound MCP rows never reach this module.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

CONNECT_KEYS = ("MCP_SERVERS", "MCP_ENABLED")
_DERIVED_KEYS = ("command", "args", "cwd")
_STALE_DOCUMENT = "The MCP settings changed while connecting, so nothing was written. Open Connect tools again."


def _on(value: Any) -> bool:
    from ouroboros.mcp_client import setting_flag

    return setting_flag(value)


def entry_problem(skill_dir: Any, entry: str) -> str:
    """Why a declared server's ``entry`` cannot start from ``skill_dir``, "" when it can: a file of the reviewed
    payload, outside every cache or dependency folder and per-install note. The one rule of a start
    (``_derive_launch``) and of skill_preflight."""
    from ouroboros.skill_identity import is_per_install_note, payload_bucket
    from ouroboros.skill_loader import _SKILL_DIR_CACHE_NAMES

    if any(part in _SKILL_DIR_CACHE_NAMES for part in pathlib.PurePosixPath(entry).parts):
        return f"its entry {entry!r} lies in a cache or dependency folder, outside the reviewed files"
    root = pathlib.Path(skill_dir).resolve()
    target = (root / entry).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        return f"its entry {entry!r} is not a file inside the skill"
    if any(part in _SKILL_DIR_CACHE_NAMES for part in target.relative_to(root).parts):
        return f"its entry {entry!r} resolves into a cache or dependency folder, outside the reviewed files"
    bucket = payload_bucket(root)
    if any(is_per_install_note(rel, bucket) for rel in
           (pathlib.PurePosixPath(entry).as_posix(), target.relative_to(root).as_posix())):
        return f"its entry {entry!r} selects a per-install note, outside the reviewed files"
    return ""


def _derive_launch(skill: Any, decl: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    """``({command, args, cwd}, "")`` for one declaration, or ``(None, reason)``."""
    from ouroboros import work_domain
    if why := entry_problem(skill.skill_dir, decl["entry"]):
        return None, why
    root = pathlib.Path(skill.skill_dir).resolve()
    entry = (root / decl["entry"]).resolve()
    runtime = decl["runtime"]
    if runtime not in {"node", "agent_python"}:
        return None, f"unsupported skill MCP runtime {runtime!r}"
    if runtime == "agent_python":
        if work_domain.managed():
            return None, "agent_python MCP servers cannot start in this deployment's work container"
        from ouroboros.process_interpreters import _usable_executable

        binary = (_usable_executable(os.environ.get("OUROBOROS_AGENT_PYTHON", ""))
                  or _usable_executable(sys.executable))
        if not binary:
            return None, "no usable agent_python runtime was found"
        return {"command": binary, "args": ["-I", "-B", str(entry)], "cwd": str(root)}, ""
    if work_domain.managed():  # mapped or staged: the work container resolves its own runtime
        from ouroboros.work_skill_payloads import _CONTAINER_RUNTIMES

        return {"command": _CONTAINER_RUNTIMES[runtime], "args": [str(entry)],
                "cwd": str(root)}, ""
    from ouroboros.tools.skill_exec import _resolve_runtime_binary

    binary, why = _resolve_runtime_binary(runtime)
    if not binary:
        return None, f"no usable {runtime} runtime was found" + (f" ({why})" if why else "")
    return {"command": binary, "args": [str(entry)], "cwd": str(root)}, ""


def _work_copy(skill: Any) -> Tuple[Optional[pathlib.Path], str]:
    """Where a managed execution domain starts this skill's servers, read passively (a path, nothing copied):
    ``(home, "")`` for the work copy of its reviewed payload (``work_skill_payloads``), ``(None, "")`` where they start
    as installed (unmanaged, or a skill folder inside the domain's mappings), ``(None, why)`` where it cannot. That
    domain starts runnable stdio MCP servers in its work container (``execution_domain.attached_launch``), never on
    the service, and an extension's package stays on the service, never copied there
    (``work_skill_payloads.deps_home``)."""
    from ouroboros import work_domain

    if not work_domain.managed():
        return None, ""
    if skill.manifest.is_extension():  # trusted code runs on the service only; none of it is a work-domain process
        return None, ("this deployment starts MCP servers in its work container, and an extension's package stays on "
                      "the service, never copied there")
    from ouroboros.execution_domain import current

    domain = current()
    if domain is not None and not domain.skill_state_root:  # its start's state folder (``_skill_env``), as scripts'
        return None, ("this deployment's execution domain names no skill_state_root, where a skill's own processes "
                      "keep their state")
    if work_domain.is_work(pathlib.Path(skill.skill_dir)):
        return None, ""
    from ouroboros.work_skill_payloads import _home

    try:
        return pathlib.Path(_home(skill.name, skill.content_hash)), ""
    except work_domain.WorkDomainRefused as refused:  # e.g. a domain that names no staging_root
        return None, str(refused)


def _skill_env(drive_root: pathlib.Path, skill: Any, *, start: bool) -> Tuple[Dict[str, str], Tuple[str, ...], str]:
    """The environment ``skill_exec`` gives this skill's scripts, from its owners: the declared ``env_from_settings``
    filtered by the owner's grants for these bytes, the skill's name and state folder (``_scrub_env``; a start's is
    ``execution_domain.skill_state_dir``) and, for a start, its isolated dependencies (``augment_env_for_skill_deps``);
    with the values among them that Settings masks, for redaction, and a digest of the settings delivered (never their
    values), for the binding's identity. A start makes the state folder and refuses missing grants; a binding reads."""
    from ouroboros.config import SETTINGS_DEFAULTS
    from ouroboros.execution_domain import skill_state_dir  # managed: the work domain's own state folder
    from ouroboros.marketplace.isolated_deps import augment_env_for_skill_deps
    from ouroboros.secret_masking import MASKED_SECRET_SETTING_KEYS, is_custom_secret_setting_key
    from ouroboros.skill_loader import grant_status_for_skill, skill_state_dir_path
    from ouroboros.tools.skill_exec import _grant_refusal, _scrub_env

    grants = grant_status_for_skill(drive_root, skill)
    if start and (refusal := _grant_refusal(skill.name, grants)):
        raise ValueError(refusal)
    declared = {str(key).strip() for key in skill.manifest.env_from_settings or [] if str(key).strip()}
    env = _scrub_env(sorted(declared), (skill_state_dir if start else skill_state_dir_path)(drive_root, skill.name),
                     skill.name, granted_keys=list(grants.get("granted_keys") or []))
    named = {key.upper() for key in declared}
    secrets = tuple(value for key, value in env.items() if value and key.upper() in named and (
        key.upper() in MASKED_SECRET_SETTING_KEYS
        or is_custom_secret_setting_key(key, known_setting_keys=SETTINGS_DEFAULTS)))
    delivered = {key: value for key, value in env.items() if key.upper() in named}
    digest = hashlib.sha256(json.dumps(delivered, sort_keys=True).encode()).hexdigest() if delivered else ""
    return (augment_env_for_skill_deps(env, pathlib.Path(skill.skill_dir)) if start else env), secrets, digest


def work_start(cfg: Any, env: Dict[str, str]) -> Tuple[List[str], Optional[str], Dict[str, str]]:
    """``(argv, cwd, env)`` that one real stdio start of ``cfg`` hands ``execution_domain.attached_launch`` (listing,
    Test, calls and task sessions all start in ``mcp_client._transport_factory``). An unbound server starts as
    configured with ``env``. A skill-bound server starts with its skill's script environment (``_skill_env``) under
    ``env`` (the row's own env and settings references, then the host's task and guard entries), its name and state
    folder kept; under a managed execution domain, Node uses the verified work copy of its reviewed payload with
    the container's runtime and the copy's dependencies (``work_skill_payloads.launch``, as ``skill_exec`` does),
    while agent_python refuses before launch. The
    copy is made or checked only here, never by a passive read; changed payload or granted settings, missing
    grants, or an unavailable domain refuse before launch. Changed settings need a fresh binding and catalog."""
    from ouroboros import work_domain

    argv, cwd = [cfg.command, *cfg.args], cfg.cwd or None
    if not getattr(cfg, "skill", ""):
        return argv, cwd, env
    from ouroboros import config
    from ouroboros.workspace_executor import overlay_env

    root = pathlib.Path(config.DATA_DIR)
    skill = _find(root, cfg.skill)
    decl = next((item for item in skill.manifest.mcp_servers if item["id"] == cfg.id), None) if skill else None
    why = ("its skill changed since it was bound" if decl is None or skill.content_hash != cfg.skill_revision
           else "; ".join(_not_ready(root, skill)))  # what the binding checked, read again right before the start
    if not why and work_domain.managed() and decl["runtime"] == "agent_python":
        why = "agent_python MCP servers cannot start in this deployment's work container"
    try:
        if not why:
            own, _secrets, digest = _skill_env(root, skill, start=True)
            # Launch only the snapshot whose identity and redaction values the consumer already holds.
            # Never mutate shared cfg or replay a call: a normal rebind/relist admits the new settings.
            if digest != cfg.skill_settings:
                why = "its granted settings changed since it was bound; refresh its tools and try again"
    except ValueError as refused:
        why = str(refused)
    if why:
        raise work_domain.WorkDomainRefused(f"MCP_SKILL_UNAVAILABLE: server {cfg.id!r} of skill {cfg.skill!r} cannot "
                                            f"start ({why}); nothing was started, here or in the work domain")
    env = {**overlay_env(own, env), **{key: own[key] for key in ("OUROBOROS_SKILL_NAME", "OUROBOROS_SKILL_STATE_DIR")}}
    if not work_domain.managed():
        return argv, cwd, env
    from ouroboros.work_skill_payloads import launch

    return launch(root, skill, cfg.skill_revision, argv, env, decl["runtime"])


def _not_ready(drive_root: pathlib.Path, skill: Any) -> List[str]:
    """The facts that keep this payload from running now; empty when it may run."""
    from ouroboros.skill_readiness import skill_readiness_for_execution

    if skill.load_error:
        return [f"load_error: {skill.load_error}"]
    if refused := _work_copy(skill)[1]:
        return [refused]
    if not (skill.manifest.is_script() or skill.manifest.is_extension()):
        return ["only script and extension skills run servers"]
    from ouroboros.config import get_skills_repo_path
    from ouroboros.skill_peer_inventory import discover_conflict_peers

    peers = list(discover_conflict_peers(drive_root, skill.name, repo_path=get_skills_repo_path()))
    return list(skill_readiness_for_execution(drive_root, skill, skills=peers).blockers)


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

    skill = None
    try:
        root = pathlib.Path(drive_root or config.DATA_DIR)
        skill = _find(root, cfg.skill)
        decl = next((item for item in skill.manifest.mcp_servers if item["id"] == cfg.id), None) if skill else None
        blockers = (["it is not installed"] if skill is None else
                    [f"it no longer declares server {cfg.id!r}"] if decl is None else _not_ready(root, skill))
        launch, why = _derive_launch(skill, decl) if not blockers else (None, "")
        # Every text from or about the server is masked with the granted secrets its start receives (``_skill_env``),
        # and a change of the settings it receives is a new identity (its task sessions start anew).
        _env, granted, digest = _skill_env(root, skill, start=False) if launch else ({}, (), "")
    except Exception as exc:  # noqa: BLE001 - an unreadable skill is an unavailable server, never a crash
        launch, blockers, why = None, [f"{type(exc).__name__}: {exc}"], ""
    if launch is None:
        reason = "; ".join(blockers) or why
        until = ("this deployment decides where it runs." if skill is not None and _work_copy(skill)[1]
                 else "it works again once the skill is ready on its Skills card.")
        return dataclasses.replace(cfg, command="", args=[], cwd="", skill_revision="", skill_settings="", skill_error=(
            f"MCP_SKILL_UNAVAILABLE: server {cfg.id!r} runs from skill {cfg.skill!r}, which cannot run it now "
            f"({reason}). Nothing was started. Its Settings row is kept; {until}"))
    # The accepted declaration, not the saved row, decides the call lifetime: an update that adds browser_policy
    # is never launched unguarded from an older row.
    secrets = tuple(dict.fromkeys((*cfg.secret_values, *granted)))
    return dataclasses.replace(cfg, **launch, task_session=decl["task_session"], browser_policy=decl["browser_policy"],
                               skill_revision=skill.content_hash, skill_settings=digest, skill_error="",
                               secret_values=secrets)


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
    home, domain_refusal = _work_copy(skill)
    for item, decl in zip(plan["servers"], skill.manifest.mcp_servers):
        launch, why = _derive_launch(skill, decl) if not blockers else (None, "")
        if launch and home is not None:  # what the work domain starts (``work_start``), shown without copying
            root = pathlib.Path(skill.skill_dir).resolve()
            launch = {"command": launch["command"], "cwd": str(home),
                      "args": [str(home / pathlib.Path(arg).relative_to(root)) for arg in launch["args"]]}
        item["executable"] = launch or {}
        if why:
            blockers.append(f"server {decl['id']!r}: {why}")
    plan.pop("_rows")
    return {"ok": True, "skill": skill.name, "content_hash": skill.content_hash, "ready": not blockers,
            "blockers": blockers, "settings_digest": digest,
            "next_action": "" if not blockers or domain_refusal else "Finish this skill's setup on its Skills "
                           "card (review, turn on, access, packages) before connecting its tools.", **plan}


def connect(drive_root: pathlib.Path, skill_name: str, *, expected_content_hash: str, settings_digest: str,
            confirm_other_servers: bool = False) -> Dict[str, Any]:
    """Apply the previewed Connect: one owner settings write of ``CONNECT_KEYS``, or a typed refusal."""
    from ouroboros.deployment_policy import DeploymentPolicyRefusal
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
    except DeploymentPolicyRefusal as exc:  # the writer's typed refusal of a managed choice: nothing was written
        return _refusal(owner.MANAGED_BY_DEPLOYMENT, str(exc), 403)
    except owner.SettingsPreconditionFailed as exc:
        code = "settings_unreadable" if isinstance(exc, owner.SettingsDocumentUnreadable) else "settings_precondition_failed"
        return _refusal(code, str(exc))
    except (owner.SettingsLockUnavailable, owner.SettingsDocumentBusy) as exc:
        return _refusal("settings_busy", str(exc), 503)
    return {"ok": True, "saved": True, "changed": True, "skill": skill.name, "mcp_enabled": True,
            "changed_keys": sorted(receipt), **facts}


def card_facts(skill: Any, settings: Dict[str, Any], status_rows: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Passive binding, listing and diagnostic facts; a successful catalog note is not a refusal."""
    from ouroboros.mcp_client import raw_server_id

    rows = [item for item in settings.get("MCP_SERVERS") or [] if isinstance(item, dict)]
    facts = []
    for decl in skill.manifest.mcp_servers:
        claimants = [item for item in rows if raw_server_id(item) == decl["id"]]
        bound = len(claimants) == 1 and claimants[0].get("skill") == skill.name
        status = status_rows.get(decl["id"], {}) if bound else {}
        listed_ok = (status.get("latest_attempt") or {}).get("ok") is True
        diagnostic = str(status.get("last_error") or "")
        facts.append({"id": decl["id"], "title": decl["title"], "conflict": bool(claimants) and not bound, "bound": bound,
                      "configured": bound and _on(claimants[0].get("enabled")) and _on(settings.get("MCP_ENABLED")),
                      "tool_count": int(status.get("tool_count") or 0), "listed": bool(status.get("listed_at")),
                      "error": str(status.get("skill_error") or ("" if listed_ok else diagnostic)),
                      "notes": diagnostic if listed_ok else ""})
    return facts


def settings_references(settings: Dict[str, Any]) -> set[str]:
    """Declared settings of enabled bindings, using the existing discovery reuse.

    This only detects changes. Binding still checks current revision, readiness and grants, and missing-only
    discovery never starts a denied server. Include both spellings used by ``skill_exec._scrub_env``.
    """
    from ouroboros import config

    references = set()
    rows = settings.get("MCP_SERVERS")
    if not _on(settings.get("MCP_ENABLED")) or not isinstance(rows, list):
        return references
    names = {row["skill"] for row in rows if isinstance(row, dict) and isinstance(row.get("skill"), str)
             and row["skill"] and row.get("transport", "stdio") == "stdio" and _on(row.get("enabled"))}
    for name in names:
        try:
            skill = _find(pathlib.Path(config.DATA_DIR), name)
        except Exception:  # bind_config reports an unreadable skill; other servers still configure normally
            continue
        if skill is not None:
            declared = {str(key).strip() for key in skill.manifest.env_from_settings or [] if str(key).strip()}
            references.update(declared | {key.upper() for key in declared})
    return references


def lifecycle_settled(skill_name: str = "", *, event_queue: Any = None, reason: str = "skill_lifecycle") -> None:
    """Notify the lister's background owner; without bound rows, do not even import the MCP SDK.

    Binding, pruning and listing run on the manager's existing discovery thread, never on the supervisor's
    event intake. Workers send the same content-free report. The discovery owner exposes ``wait`` for teardown.
    """
    try:
        from ouroboros.config import load_settings

        settings = load_settings()
        rows = settings.get("MCP_SERVERS")
        if not isinstance(rows, list) or not any(isinstance(row, dict) and row.get("skill") for row in rows):
            return
        from ouroboros import mcp_client

        manager = mcp_client.get_manager()
        if manager.is_lister:
            manager.discovery.settle(reason)
        elif event_queue is not None:
            event_queue.put_nowait({"type": "mcp_configuration_changed", "reason": reason, "skill": skill_name})
    except Exception:
        log.warning("MCP reconfiguration after a change of skill %r failed", skill_name, exc_info=True)


__all__ = ["CONNECT_KEYS", "bind_config", "card_facts", "connect", "entry_problem", "lifecycle_settled",
           "plan_connect", "preview_connect", "work_start"]
