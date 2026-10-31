"""Shared skill grant/toggle effects and host-validated owner action inputs.

HTTP and task adapters provide their actual caller context (the desktop launcher posts to
the server after its native confirm, so no install ever runs in the frozen launcher). The model
interprets an owner's instruction; this owner verifies the referenced source,
selected payload revision and existing execution prerequisites. There is no new
permission store, inferred approval, or HTTP impersonation.

``reconcile_turn_on_when_ready`` is the one place an install's "turn on when
ready" request becomes enablement: after install, after a review or owner Skip
ends, after grants, at startup. It never overrides an explicit choice
(``turn_on_blocker`` is that intent's one owner). Every ``_toggle`` refusal is typed:
``code`` names the gate, ``error`` is the owner's one sentence (the toast prefixes the
skill name), ``detail`` keeps the mechanism's own words.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ouroboros.contracts.task_constraint import normalize_task_constraint
from ouroboros.owner_source import resolve_owner_source as resolve_skill_owner_source
from ouroboros.skill_lifecycle_queue import LifecycleJobOptions, run_lifecycle_job_blocking
from ouroboros.skill_review_status import executable_review_requirement
from ouroboros.skill_loader import (
    discover_skills, find_skill, grant_status_for_skill, requested_core_setting_keys,
    requested_skill_permissions, save_enabled, save_skill_grants, skill_conflict_status,
    skill_state_dir, skill_state_dir_path,
)
from ouroboros.tool_access import active_tool_profile, canonical_data_root
from ouroboros.utils import append_jsonl, read_json_dict, utc_now_iso

log = logging.getLogger(__name__)


def _refusal(message: str, status_code: int = 409, *, code: str = "", detail: str = "", **facts: Any) -> dict[str, Any]:
    return {"ok": False, "error": message, "status_code": status_code,
            **({"code": code} if code else {}), **({"detail": detail} if detail else {}), **facts}


def _review_facts(skill: Any, drive_root: Path) -> dict[str, Any]:
    stale = skill.review.is_stale_for(skill.content_hash)
    gate = skill.review.gate_for(skill.content_hash)
    return {
        "skill": skill.name, "source": skill.source, "content_hash": skill.content_hash,
        "review_status": skill.review.status, "review_stale": stale,
        "review_gate": gate, "executable_review": gate["executable_review"],
        "grants": grant_status_for_skill(drive_root, skill),
    }


def _strict_review(skill: Any) -> bool:
    """A source that reviews each skill: its blockers need the owner (skill_acceptance)."""
    return bool((getattr(skill.review, "acceptance", None) or {}).get("strict_review"))


def _caller_action_error(ctx: Any, action: str, owner_actor: str, owner_source: Any) -> tuple[str, dict]:
    if owner_actor:
        return ("", {}) if owner_actor in {"owner_ui", "owner_cli"} else ("unknown host owner actor", {})
    if active_tool_profile(ctx) not in {"self_modification", "workspace_task", "external_workspace_task", "operator_control"}:
        return "this caller cannot perform skill lifecycle actions", {}
    from ouroboros.cancel_intents import cancel_pending

    task_id = str(getattr(ctx, "task_id", "") or "")
    if task_id and cancel_pending(canonical_data_root(ctx), task_id):
        return "the task has a pending Stop; its skill action was not performed", {}
    constraint = normalize_task_constraint(getattr(ctx, "task_constraint", None))
    from ouroboros.presence_authority import presence_ceiling_from_context

    # A selected payload or a copied owner reference cannot widen Presence.
    if presence_ceiling_from_context(ctx) is not None:
        return "a Presence task cannot issue owner-only skill authority", {}
    needs_owner = action in {"grant", "attest", "delete"} or (
        action == "enable" and constraint is not None
        and (constraint.has_selected_skill or not constraint.allow_enable)
    )
    if not needs_owner:
        return "", {}
    if action == "enable" and owner_source is None:
        from ouroboros.task_status import load_effective_task_result

        task = load_effective_task_result(canonical_data_root(ctx), task_id, materialize_artifacts=False) if task_id else {}
        metadata = getattr(ctx, "task_metadata", None)
        origin = (task or {}).get("origin_message_ref")
        if not origin and isinstance(metadata, dict):
            origin = metadata.get("origin_message_ref")
        owner_source = {"kind": "chat", "ref": origin}
    resolved = resolve_skill_owner_source(ctx, owner_source)
    if not resolved:
        return "an existing owner message, answered quiz or owner mailbox entry in this task is required", {}
    return "", resolved


def _reconcile(skill: Any, drive_root: Path, repo_path: str, *, enabling: bool = False) -> dict[str, Any]:
    from ouroboros import extension_loader
    from ouroboros.config import load_settings

    if not skill.manifest.is_extension() and skill.name not in extension_loader.snapshot()["extensions"]:
        return {"action": None, "reason": "not_extension"}
    return extension_loader.reconcile_extension(
        skill.name, drive_root, load_settings, repo_path=repo_path, selected_skill=skill,
        retry_load_error=True, revert_enabled_on_error=enabling,
    )


def _toggle(skill: Any, drive_root: Path, repo_path: str, enabled: bool, actor: str, reason: str, *,
            progress: Any = None) -> dict:
    """Enable or disable one skill. Enabling establishes the declared packages itself (the
    seam is idempotent, so Enable on an enabled skill repairs it and Enable again retries a
    failure); ``progress`` receives the job's message while packages install."""
    facts = _review_facts(skill, drive_root)
    collision = skill.load_error.lower().startswith("skill name collision:")
    if enabled:
        if skill.load_error:
            return _refusal("Its files could not be loaded.", 400, code="load_error", detail=skill.load_error, **facts)
        conflict = skill_conflict_status(skill, discover_skills(drive_root, repo_path=repo_path))
        if conflict:
            return _refusal("Another skill that is on uses the same tool names; turn it off first.",
                            code="conflict", conflict=conflict, **facts)
        if not facts["executable_review"]:
            need = executable_review_requirement(facts["review_gate"], strict_review=_strict_review(skill))
            return _refusal("It turns on after a review passes.", code="acceptance", detail=need, **facts)
        if not facts["grants"].get("all_granted", True):
            return _refusal("Grant it the keys and permissions it asks for first.", code="grants", **facts)
        from ouroboros.skill_dependencies import dependency_sentence, ensure_skill_dependencies

        deps_status, detail = ensure_skill_dependencies(drive_root, skill, progress=progress)
        if deps_status not in {"installed", "not_required"}:
            code = "dependencies_failed" if deps_status == "failed" else "acceptance"
            return _refusal(dependency_sentence(code) if code == "dependencies_failed" else "It turns on after a review passes.",
                            code=code, detail=detail, deps_status=deps_status, **facts)
    elif collision:
        from ouroboros import extension_loader

        extension_loader.unload_extension(skill.name)
        return _refusal("cannot persist disable while the skill identity collides; rename one directory first", 400,
                        extension_action="extension_unloaded", extension_reason="name_collision", **facts)
    save_enabled(drive_root, skill.name, enabled, actor=actor, reason=reason)
    skill.enabled = enabled
    state = _reconcile(skill, drive_root, repo_path, enabling=enabled)
    failed = enabled and state.get("action") == "extension_load_error"
    return {
        "ok": not failed, **facts, "enabled": enabled and not failed,
        "extension_action": state.get("action"), "extension_reason": state.get("reason"),
        "process": str(state.get("process") or ""), "server_reconcile": str(state.get("server_reconcile") or ""),
        "load_error": state.get("load_error"),
        **({"error": "It could not start.", "code": "load_error", "status_code": 409,
            "detail": str(state.get("load_error") or "extension failed to load")} if failed else {}),
    }


# The only enablement writer whose Off is not a choice: a failed load reverts the toggle.
_REVERSIBLE_DISABLE_ACTORS = frozenset({"load_error_revert"})


def turn_on_blocker(drive_root: Path, skill_name: str) -> str:
    """Why an install's turn-on-when-ready request is not open: ``not_requested``, ``already_on``
    or ``owner_choice`` (``enabled.json`` written by anyone but a load-error revert); ``""`` while
    the request stands. The one owner of that intent: turn-on and the startup pass share it."""
    from ouroboros.skill_install_record import read_install_record
    from ouroboros.skill_loader import load_enabled

    drive = Path(drive_root)
    if (read_install_record(drive, skill_name) or {}).get("turn_on_when_ready") is not True:
        return "not_requested"
    if load_enabled(drive, skill_name):
        return "already_on"
    choice = read_json_dict(skill_state_dir_path(drive, skill_name) / "enabled.json")
    return "owner_choice" if choice is not None and choice.get("actor") not in _REVERSIBLE_DISABLE_ACTORS else ""


def reconcile_turn_on_when_ready(
    drive_root: Path, skill_name: str, *, repo_path: str | None = None, reason: str = "", progress: Any = None,
) -> dict[str, Any]:
    """Enable an installed skill whose install record asks to turn on when ready.

    Call it inside a lifecycle job (the caller holds the lane; this never takes it; it calls
    ``_toggle``, never ``run_skill_action``, because the lane lock is not re-entrant). The
    toggle's own checks decide: a typed refusal becomes ``not_ready`` naming what is
    ``missing`` (``acceptance``, ``grants``, ``conflict``, ``load_error``, ``dependencies``);
    a load failure after the write stays ``load_error``. Never over an explicit choice.
    Returns ``{skill, action, enabled, missing}``; never raises.
    """
    from ouroboros.config import get_skills_repo_path

    out: dict[str, Any] = {"skill": skill_name, "action": "not_requested", "enabled": False, "missing": []}
    try:
        drive = Path(drive_root)
        blocker = turn_on_blocker(drive, skill_name)
        if blocker:
            return {**out, "action": blocker, "enabled": blocker == "already_on"}
        repo = get_skills_repo_path() if repo_path is None else repo_path
        skill = find_skill(drive, skill_name, repo_path=repo)
        if skill is None:
            return {**out, "action": "not_found"}
        result = _toggle(skill, drive, repo, True, "turn_on_when_ready", reason or "install requested turn on when ready", progress=progress)
        if result.get("ok"):
            return {**out, "action": "enabled", "enabled": True}
        if result.get("extension_action") == "extension_load_error":
            return {**out, "action": "load_error", "error": str(result.get("error") or "")}
        code = str(result.get("code") or "")
        return {**out, "action": "not_ready", "missing": ["dependencies" if code == "dependencies_failed" else code],
                **({"error": result["error"], "detail": result.get("detail", "")} if code == "dependencies_failed" else {})}
    except Exception as exc:
        log.warning("turn-on-when-ready reconcile failed for %s", skill_name, exc_info=True)
        return {**out, "action": "error", "error": f"{type(exc).__name__}: {exc}"}


def prepare_skill_grant(skill: Any, drive_root: Path, items: list[str]) -> dict:
    """Validate the exact request before a launcher confirmation or grant write."""
    facts = _review_facts(skill, drive_root)
    if not (skill.manifest.is_script() or skill.manifest.is_extension()):
        return _refusal("key and permission grants are supported for script and extension skills", 400)
    if not facts["executable_review"]:
        need = executable_review_requirement(facts["review_gate"], strict_review=_strict_review(skill))
        return _refusal(f"key and permission grants need the review gate to permit the current bytes: {need}", **facts)
    allowed_keys = requested_core_setting_keys(list(skill.manifest.env_from_settings or []))
    allowed_permissions = requested_skill_permissions(skill.manifest.permissions, skill.manifest.subscribe_events)
    permissions = {value.lower(): value for value in allowed_permissions}
    keys, granted_permissions, rejected = [], [], []
    for item in dict.fromkeys(items):
        if item.upper() in allowed_keys:
            if item.upper() not in keys:
                keys.append(item.upper())
        elif item.lower() in permissions:
            if permissions[item.lower()] not in granted_permissions:
                granted_permissions.append(permissions[item.lower()])
        else:
            rejected.append(item)
    if not items or rejected:
        return _refusal("grant items must be requested by the current manifest", 400,
                        allowed_keys=allowed_keys, allowed_permissions=allowed_permissions, rejected_items=rejected)
    return {"ok": True, "skill": skill.name, "content_hash": skill.content_hash,
            "keys": keys, "permissions": granted_permissions,
            "allowed_keys": allowed_keys, "allowed_permissions": allowed_permissions}


def _grant(skill: Any, drive_root: Path, repo_path: str, items: list[str], *, progress: Any = None,
           enable_after_grant: bool = False) -> dict:
    request = prepare_skill_grant(skill, drive_root, items)
    if request.get("error"):
        return request
    keys, granted_permissions = request["keys"], request["permissions"]
    save_skill_grants(drive_root, skill.name, keys, content_hash=skill.content_hash,
                      requested_keys=request["allowed_keys"], granted_permissions=granted_permissions,
                      requested_permissions=request["allowed_permissions"])
    if enable_after_grant:
        # The owner's explicit Enable survives an answered-pending grant, without another job or pip attempt.
        try:
            state = _toggle(skill, drive_root, repo_path, True, "owner_ui", "enable with confirmed grants", progress=progress)
        except Exception as exc:
            state = {"error": f"Setup failed: {type(exc).__name__}: {exc}"}
        turn_on = {"action": "enabled" if state.get("ok") else "not_ready",
                   "enabled": bool(state.get("enabled")), "error": str(state.get("error") or "")}
    else:
        try:
            state = _reconcile(skill, drive_root, repo_path)
        except Exception as exc:
            state = {"reason": "reconcile_call_failed", "load_error": str(exc)}
        turn_on = reconcile_turn_on_when_ready(drive_root, skill.name, repo_path=repo_path, reason="grants approved", progress=progress)
    return {
        "turn_on_when_ready": turn_on["action"],
        **({"enabled": turn_on["enabled"]} if enable_after_grant else {}),
        **({"setup_error": turn_on["error"]} if turn_on.get("error") else {}),
        "ok": True, "skill": skill.name, "content_hash": skill.content_hash,
        "granted_keys": keys, "granted_permissions": granted_permissions,
        "extension_action": state.get("extension_action") or state.get("action"),
        "extension_reason": state.get("extension_reason") or state.get("reason"),
        "load_error": state.get("load_error"), "grants": grant_status_for_skill(drive_root, skill),
        "process": str(state.get("process") or ""), "server_reconcile": str(state.get("server_reconcile") or ""),
        "live_loaded": state.get("live_loaded"),
    }


def run_skill_action(
    ctx: Any, skill_name: str, action: str, *, expected_content_hash: str = "",
    items: list[str] | None = None, owner_source: Any = None, payload_root: str = "",
    repo_path: str | None = None, _owner_actor: str = "",
    _progress: Any = None, _enable_after_grant: bool = False, _reconcile_grant: Any = None,
) -> dict[str, Any]:
    """Apply one selected action through the existing lifecycle lane; ``_progress`` is the
    caller's job handle (one job binds one progress target), relayed into the job.
    The legacy keyword only diagnoses a source launcher straddling an update; it runs no callback."""
    if _owner_actor == "owner_launcher":
        return _refusal("Restart Ouroboros, then grant access again.", 409, code="launcher_restart_required")
    from ouroboros.config import get_skills_repo_path
    from ouroboros.tool_access import build_resolved_resource_binding

    if action not in {"grant", "enable", "disable", "attest", "delete"}:
        return _refusal("unknown skill action", 400)
    if items is not None and (not isinstance(items, list) or any(not isinstance(item, str) for item in items)):
        return _refusal("grant items must be a list of strings", 400)
    error, source = _caller_action_error(ctx, action, _owner_actor, owner_source)
    if error:
        return _refusal(error, 403)
    if not _owner_actor and action in {"grant", "attest", "delete"} and not expected_content_hash:
        return _refusal("the selected content hash is required for this owner action", 400)
    drive = canonical_data_root(ctx)
    repo_path = get_skills_repo_path() if repo_path is None else repo_path
    task_id = str(getattr(ctx, "task_id", "") or "")
    actor = _owner_actor or "agent_tool"

    def selected():
        nonlocal source
        error, current_source = _caller_action_error(ctx, action, _owner_actor, owner_source)
        if error:
            return None, _refusal(error, 403)
        source = current_source
        loaded = find_skill(drive, skill_name, repo_path=repo_path)
        if loaded is None:
            return None, _refusal("skill not found", 404)
        if expected_content_hash and loaded.content_hash != expected_content_hash:
            return None, _refusal("selected skill revision changed; inspect it before repeating the action")
        constraint = normalize_task_constraint(getattr(ctx, "task_constraint", None))
        if constraint and constraint.has_selected_skill:
            try:
                binding = build_resolved_resource_binding(ctx, root="skill_payload", operation="review", skill_name=skill_name)
            except ValueError as exc:
                return None, _refusal(str(exc), 403)
            if Path(loaded.skill_dir).resolve() != binding.base_path:
                return None, _refusal("selected skill payload does not match this task", 403)
            if action == "enable" and not _owner_actor:
                from ouroboros.deadline_utils import parse_deadline_ts

                enabled_state = read_json_dict(skill_state_dir(drive, skill_name) / "enabled.json") or {}
                # Only known direct owner actors attest an independent choice (owner_launcher: files
                # older launchers wrote). Load-error reverts and old unlabelled snapshots do not.
                if enabled_state.get("enabled") is False and enabled_state.get("actor") in {
                    "owner_ui", "owner_cli", "owner_launcher",
                }:
                    disabled_at = parse_deadline_ts(enabled_state.get("updated_at"))
                    source_at = parse_deadline_ts(source.get("ts"))
                    if disabled_at is None or source_at is None or source_at <= disabled_at:
                        return None, _refusal("the owner disabled this skill after the referenced request; use a newer owner instruction", 403)
        return loaded, None

    def audit(result):
        try:
            append_jsonl(drive / "logs" / "events.jsonl", {
                "ts": utc_now_iso(), "type": "owner_api_action",
                "action": "skill_owner_attest" if action == "attest" else f"skill_{action}",
                "actor": actor, "task_id": task_id, "skill": skill_name,
                "client_host": str(getattr(ctx, "_skill_owner_client_host", "") or ""),
                "content_hash": result.get("content_hash") or expected_content_hash,
                "requested_items": list(items or []), "ok": bool(result.get("ok")),
                "enable_after_grant": _enable_after_grant,
                "granted_key_count": len(result.get("granted_keys") or []),
                "granted_permission_count": len(result.get("granted_permissions") or []),
                "extension_action": result.get("extension_action"),
                "extension_reason": result.get("extension_reason"),
                "source_ref": {key: value for key, value in source.items() if key != "text"},
            })
        except Exception:
            log.warning("skill action audit failed", exc_info=True)
        return result

    if action == "attest":
        from ouroboros.skill_owner_attestation import review_skill_owner_attest
        from ouroboros.skill_review import SkillReviewOutcome
        from ouroboros.skill_review_runner import run_skill_review_lifecycle_blocking

        def attest(review_ctx, name):
            _loaded, refusal = selected()
            if refusal:
                return SkillReviewOutcome(skill_name=name, status="pending", error=refusal["error"])
            return review_skill_owner_attest(review_ctx, name)

        result = run_skill_review_lifecycle_blocking(ctx, skill_name, source="skills" if _owner_actor else "tool",
                                                    review_impl=attest, repo_path=repo_path,
                                                    owner_attestation=True, progress=_progress)
        result["ok"] = result.get("status") == "clean"
        if not result["ok"]:
            result["status_code"] = 409
        return audit(result)

    def apply():
        loaded, refusal = selected()
        if refusal:
            return refusal
        if action == "grant":
            return _grant(loaded, drive, repo_path, list(items or []), progress=_progress.set if _progress is not None else None,
                          enable_after_grant=_enable_after_grant)
        if action == "delete":
            from ouroboros.skill_uninstall_state import delete_local_skill

            return delete_local_skill(drive, loaded, payload_root=payload_root, repo_path=repo_path)
        return _toggle(loaded, drive, repo_path, action == "enable", actor,
                       json.dumps({key: value for key, value in source.items() if key != "text"}) if source else task_id,
                       progress=_progress.set if _progress is not None else None)

    result = run_lifecycle_job_blocking(
        kind=action, target=skill_name, source="skills" if _owner_actor else "tool", runner=apply,
        chat_id=int(getattr(ctx, "current_chat_id", 0) or 0),
        options=LifecycleJobOptions(drive_root=drive, result_error=lambda value: value.get("error", ""),
                                    result_message=(lambda value: "Access granted; " + value["setup_error"] if value.get("setup_error") else "") if action == "grant" else None,
                                    progress_target=_progress),
    )
    return audit(result)
