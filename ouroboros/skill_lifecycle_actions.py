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
    discover_skills, find_skill, grant_status_for_skill, load_skill_grants, requested_core_setting_keys,
    requested_skill_permissions, save_enabled, save_skill_grants, skill_conflict_status,
    skill_state_dir, skill_state_dir_path,
)
from ouroboros.skill_readiness import grant_gap_sentence
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


_OWNER_ACTIONS = frozenset({"grant", "attest", "delete", "connect_mcp"})


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
    needs_owner = action in _OWNER_ACTIONS or (
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
        press = metadata.get("ui_control") if isinstance(metadata, dict) and isinstance(metadata.get("ui_control"), dict) else {}
        if not origin and press.get("name") == "skill_repair" and constraint.has_selected_skill and press.get("pressed_at"):
            # My human pressed Repair for this selected skill ("turns it on when it is ready"): the press is the
            # request, held by the host at the door; its words were the host's, so nothing is quoted as theirs.
            return "", {"kind": "control", "control": "skill_repair", "ts": str(press["pressed_at"]), "text": ""}
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
            progress: Any = None, prepared_deps: tuple[str, str] | None = None, only_if_unset: bool = False) -> dict:
    """Enable or disable one skill. Enabling establishes the declared packages itself (the
    seam is idempotent, so Enable on an enabled skill repairs it and Enable again retries a
    failure); ``progress`` receives the job's message while packages install. A first Enable is
    atomic: a failed load saves Off again. A first Enable first asks the common Nano
    admission (``nano_activation``: the skill's card and any always-in-context body); a
    refusal leaves the switch untouched, and an admitted one holds the publication fence
    across the write. Enable on a skill whose saved switch was already On
    is a repair: its failure is answered and the owner's choice stays On. ``enabled`` answers
    the saved switch either way. A repair the live publication could not finish (a companion still failed,
    another awaiting exit, a reload deferred) answers ``ok`` with the reconcile's typed reason, its
    ``load_error`` words and ``retained_companions``, as the reconcile receipt does. ``prepared_deps`` is the
    dependency outcome this job already has (never installed twice); ``only_if_unset`` turns on only while nothing
    was chosen for the switch, read under the lifecycle lock (``owner_switch``)."""
    facts = _review_facts(skill, drive_root)
    collision = skill.load_error.lower().startswith("skill name collision:")
    from ouroboros.deployment_extensions import TRUST_SENTENCES, change_refusal, trust_gap
    from ouroboros.nano_publication import NanoPublicationRefused, source_bytes

    if not enabled and (managed := change_refusal(skill.name, "disable")):  # before the collision unload or a write
        return _refusal("Your organization requires it.", 403, code="managed_by_deployment", detail=managed, **facts)
    if enabled:
        if skill.load_error:
            return _refusal("Its files could not be loaded.", 400, code="load_error", detail=skill.load_error, **facts)
        if skill.manifest.is_extension() and (gap := trust_gap(skill.name, skill.skill_dir, skill.content_hash)):
            return _refusal("Only extension skills your organization delivers run here.", 403, code="managed_by_deployment",
                            detail=TRUST_SENTENCES[gap], **facts)
        conflict = skill_conflict_status(skill, discover_skills(drive_root, repo_path=repo_path))
        if conflict:
            return _refusal("Another skill that is on uses the same tool names; turn it off first.",
                            code="conflict", conflict=conflict, **facts)
        if not facts["executable_review"]:
            need = executable_review_requirement(facts["review_gate"], strict_review=_strict_review(skill))
            return _refusal("It turns on after a review passes.", code="acceptance", detail=need, **facts)
        if facts["grants"].get("blocked"):  # pending custom keys alone do not block
            return _refusal("Grant it the keys and permissions it asks for first.", code="grants",
                            detail=grant_gap_sentence(facts["grants"]), **facts)
        from ouroboros.skill_dependencies import dependency_sentence, ensure_skill_dependencies

        try:
            switch = source_bytes(skill_state_dir_path(drive_root, skill.name) / "enabled.json")
        except OSError as exc:
            return _switch_io_failure(drive_root, skill.name, exc, facts)
        deps_status, detail = (prepared_deps if prepared_deps is not None
                               else ensure_skill_dependencies(drive_root, skill, progress=progress))
        if deps_status not in {"installed", "not_required"}:
            code = "dependencies_failed" if deps_status == "failed" else "acceptance"
            return _refusal(dependency_sentence(code) if code == "dependencies_failed" else "It turns on after a review passes.",
                            code=code, detail=detail, deps_status=deps_status, **facts)
    elif collision:
        from ouroboros import extension_loader

        extension_loader.unload_extension(skill.name)
        return _refusal("cannot persist disable while the skill identity collides; rename one directory first", 400,
                        extension_action="extension_unloaded", extension_reason="name_collision", **facts)
    from ouroboros.extension_registry_state import _lifecycle_lock_for
    from ouroboros.nano_activation import check_skill_switch, refusal_words, skill_switch_publication
    from ouroboros.skill_loader import load_enabled

    # Under the lock a companion refresh holds from its final admission through its spawn (never its stop):
    # the refresh reads this choice, or the reconcile below finds its new process and stops it.
    try:
        with skill_switch_publication(skill, drive_root, enabled=enabled), _lifecycle_lock_for(skill.name):
            if enabled and only_if_unset and (choice := owner_switch(drive_root, skill.name)):
                return _refusal("The saved switch was left as it is.", code={"on": "already_on", "off": "owner_off"}.get(
                    choice, "switch_unreadable"), enabled=_saved_switch(drive_root, skill.name), **facts)
            was_on = load_enabled(drive_root, skill.name)  # the saved intent before this attempt, not what is live
            if enabled:
                check_skill_switch(drive_root, skill, switch, repo_path=repo_path)
            save_enabled(drive_root, skill.name, enabled, actor=actor, reason=reason)
    except NanoPublicationRefused as exc:
        happened, step = refusal_words(exc)
        return _refusal(f"{happened}, so its switch is unchanged; {step}.", 409,
                        code=f"nano_publication_{exc.reason}", detail=exc.detail, **facts)
    except OSError as exc:
        return _switch_io_failure(drive_root, skill.name, exc, facts)
    skill.enabled = enabled
    state = _reconcile(skill, drive_root, repo_path, enabling=enabled and not was_on)
    failed = enabled and state.get("action") == "extension_load_error"
    return {
        "ok": not failed, **facts, "enabled": enabled and not state.get("reverted_enabled"),
        "extension_action": state.get("action"), "extension_reason": state.get("reason"),
        "process": str(state.get("process") or ""), "server_reconcile": str(state.get("server_reconcile") or ""),
        "load_error": state.get("load_error"),
        **({"retained_companions": state["retained_companions"]} if state.get("retained_companions") else {}),
        **({"error": "It could not start.", "code": "load_error", "status_code": 409,
            "detail": str(state.get("load_error") or "extension failed to load")} if failed else {}),
    }


def _switch_io_failure(drive_root, name, exc, facts):
    # A file failure says nothing about size; the switch is reported as it reads now.
    return _refusal("A file could not be read or written while saving its switch.", 409,
                    code="switch_io_failed", detail=f"{type(exc).__name__}: {exc}",
                    enabled=_saved_switch(drive_root, name), **facts)


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
                **({"error": result["error"], "detail": result.get("detail", "")}
                   if result.get("error") and (code == "dependencies_failed" or code.startswith("nano_publication_")) else {})}
    except Exception as exc:
        log.warning("turn-on-when-ready reconcile failed for %s", skill_name, exc_info=True)
        return {**out, "action": "error", "error": f"{type(exc).__name__}: {exc}"}


def prepare_skill_grant(skill: Any, drive_root: Path, items: list[str]) -> dict:
    """Validate the exact request before a launcher confirmation or grant write."""
    facts = _review_facts(skill, drive_root)
    if not (skill.manifest.is_script() or skill.manifest.is_extension()):
        return _refusal("key and permission grants are supported for script and extension skills", 400,
                        code="unsupported_type")
    if not facts["executable_review"]:
        need = executable_review_requirement(facts["review_gate"], strict_review=_strict_review(skill))
        return _refusal(f"key and permission grants need the review gate to permit the current bytes: {need}",
                        code="acceptance", **facts)
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
        return _refusal("grant items must be requested by the current manifest", 400, code="not_requested",
                        allowed_keys=allowed_keys, allowed_permissions=allowed_permissions, rejected_items=rejected)
    return {"ok": True, "skill": skill.name, "content_hash": skill.content_hash,
            "keys": keys, "permissions": granted_permissions,
            "allowed_keys": allowed_keys, "allowed_permissions": allowed_permissions}


def _saved_switch(drive_root: Path, name: str) -> bool | None:
    """The saved switch, read strictly from ``enabled.json``: False if never saved (Off), None while it cannot be read
    or holds no boolean (``load_enabled`` reads both as Off). Never the effective switch: a package a deployment
    requires reads On over a saved Off."""
    from ouroboros.utils import json_snapshot

    try:
        saved = json_snapshot(skill_state_dir_path(drive_root, name) / "enabled.json", strict=True)
    except (OSError, ValueError):
        return None
    value = False if saved is None else saved.get("enabled")
    return value if isinstance(value, bool) else None


def _grant(skill: Any, drive_root: Path, repo_path: str, items: list[str], *, progress: Any = None,
           enable_after_grant: bool = False, explicit_via: str = "", form_installation: str = "",
           setup: bool = True) -> dict:
    """The one grant writer. ``form_installation``: the secret form's grant of keys only, kept apart from the ordinary
    decision for that installation; ``setup`` False leaves the one setup that follows to the caller."""
    request = prepare_skill_grant(skill, drive_root, items)
    if form_installation:
        request = _keys_only(request)
    if request.get("error"):
        return request
    keys, granted_permissions = request["keys"], request["permissions"]
    if form_installation:
        requested_keys = list(request["allowed_keys"])
    else:
        # The decision covers the keys it sent and what an earlier decision covered: a key saved later that the card did
        # not offer stays pending (B″), never covered and blocking (R2-A03).
        earlier = load_skill_grants(drive_root, skill.name)
        decided = {*keys, *(earlier.get("requested_keys") or []), *(earlier.get("granted_keys") or [])}
        requested_keys = [key for key in request["allowed_keys"] if key in decided]
    save_skill_grants(drive_root, skill.name, keys, content_hash=skill.content_hash,
                      requested_keys=requested_keys,
                      granted_permissions=granted_permissions, requested_permissions=request["allowed_permissions"],
                      form_installation=form_installation)
    if explicit_via and keys:  # the saved explicit grant only, never an automatic one (skill_grant_follow)
        from ouroboros.skill_grant_follow import record_explicit_grant

        record_explicit_grant(drive_root, skill, keys, via=explicit_via)
    if not setup:
        return {"ok": True, "skill": skill.name, "content_hash": skill.content_hash, "granted_keys": keys,
                "granted_permissions": granted_permissions}
    if enable_after_grant:
        # The owner's explicit Enable survives an answered-pending grant, without another job or pip attempt.
        try:
            state = _toggle(skill, drive_root, repo_path, True, "owner_ui", "enable with confirmed grants",
                            progress=progress, only_if_unset=bool(form_installation))
        except Exception as exc:
            state = {"error": f"Setup failed: {type(exc).__name__}: {exc}"}
        if "enabled" not in state:
            # No answer of the toggle's own: a precheck refused before any write or reconcile, or the attempt failed.
            # The saved switch as it reads now (unknown, not Off, while its file cannot be read) and this folder's own
            # pending exits (server memory only) still stand.
            from ouroboros.extension_liveness import retained_companions_for_directory

            retained = retained_companions_for_directory(skill.name, str(skill.skill_dir.resolve()))
            state = {**state, "enabled": _saved_switch(drive_root, skill.name),
                     **({"retained_companions": retained, "process": "server"} if retained else {})}
        turn_on = {"action": "enabled" if state.get("ok") else "not_ready",
                   "enabled": bool(state.get("enabled")), "error": str(state.get("error") or ""),
                   "code": "" if state.get("ok") else str(state.get("code") or "error")}
    else:
        try:
            state = _reconcile(skill, drive_root, repo_path)
        except Exception as exc:
            state = {"reason": "reconcile_call_failed", "load_error": str(exc)}
        turn_on = ({"action": "not_requested", "enabled": False, "missing": []} if form_installation
                   else reconcile_turn_on_when_ready(drive_root, skill.name, repo_path=repo_path,
                                                     reason="grants approved", progress=progress))
    return {
        "turn_on_when_ready": turn_on["action"],
        **({"enabled": turn_on["enabled"]} if enable_after_grant and state.get("enabled", False) is not None else {}),
        **({"setup_error": turn_on["error"]} if turn_on.get("error") else {}),
        **({"setup_code": turn_on["code"]} if turn_on.get("code") else {}),
        "ok": True, "skill": skill.name, "content_hash": skill.content_hash,
        "granted_keys": keys, "granted_permissions": granted_permissions,
        "extension_action": state.get("extension_action") or state.get("action"),
        "extension_reason": state.get("extension_reason") or state.get("reason"),
        "load_error": state.get("load_error"), "grants": grant_status_for_skill(drive_root, skill),
        "process": str(state.get("process") or ""), "server_reconcile": str(state.get("server_reconcile") or ""),
        "live_loaded": state.get("live_loaded"),
        **({"retained_companions": state["retained_companions"]} if state.get("retained_companions") else {}),
    }


def _keys_only(prepared: dict[str, Any]) -> dict[str, Any]:
    """The secret form grants Settings keys, never a host permission: an item the manifest names as a permission
    (a key spelled like one included) refuses the whole grant."""
    if prepared.get("ok") and prepared.get("permissions"):
        return _refusal("the secret form grants keys only", 400, code="not_requested")
    return prepared


def owner_switch(drive_root: Path, name: str) -> str:
    """The saved switch as the secret form's allow reads it: ``on``; ``off``, an Off saved by anyone but a load-error
    revert (a choice that stands whether it came before or after the allow); ``unknown`` while ``enabled.json`` cannot
    be read; ``""`` when nothing was chosen (or a load error reverted it). Only ``""`` lets an allow turn it on."""
    from ouroboros.utils import json_snapshot

    try:
        saved = json_snapshot(skill_state_dir_path(drive_root, name) / "enabled.json", strict=True)
    except (OSError, ValueError):
        return "unknown"
    value = None if saved is None else saved.get("enabled")
    if saved is None or (value is False and saved.get("actor") in _REVERSIBLE_DISABLE_ACTORS):
        return ""
    return "on" if value is True else "off" if value is False else "unknown"


def _secret_form_audit(drive: Path, facts: dict[str, Any], **fields: Any) -> None:
    try:
        append_jsonl(drive / "logs" / "events.jsonl", {
            "ts": utc_now_iso(), "type": "owner_api_action", "action": "skill_grant", "actor": "owner_ui",
            "via": "secret_form", "state": facts.get("state"), "reason": facts.get("reason", ""),
            "enabled": facts.get("enabled"), **fields})
    except Exception:
        log.warning("secret form grant audit failed", exc_info=True)


def _grant_shown_files(drive: Path, repo: str, skill_name: str, *, content_hash: str, installation: str,
                       keys: list[str], enable: bool, decision: dict[str, str], progress: Any) -> dict[str, Any]:
    """The allow's facts for one claimed press (inside the lane): a grant to exactly the shown files of the shown
    installation once they are admitted (``files_accepted``, never runtime authority alone), else the owner's
    consent for them; a skill turns on only when the press named it and no Off stands (``owner_switch``)."""
    from ouroboros.secret_requests import allow_result
    from ouroboros.skill_grant_follow import allow_installation, files_accepted, record_consent

    skill = find_skill(drive, skill_name, repo_path=repo)
    refusal = ("skill_unavailable" if skill is None or skill.load_error
               else "skill_changed" if skill.content_hash != content_hash
               else "installation_changed" if not installation or allow_installation(drive, skill) != installation
               else "unsupported" if not (skill.manifest.is_script() or skill.manifest.is_extension()) else "")
    if refusal:
        return allow_result({"ok": False, "reason": refusal}, enable=enable, attempted=False, switch="")
    switch = owner_switch(drive, skill.name)
    # An update still settling may be rolled back: its files get the consent, which its settlement decides.
    if update_pending(drive, skill.name) or not files_accepted(skill.review, skill.content_hash):
        status = record_consent(drive, skill, keys, installation=installation, enable=enable, decision=decision)
        result = ({"ok": True, "deferred": True} if status == "recorded"
                  else {"ok": False, "reason": "consent_unrecorded"})
        return allow_result(result, enable=enable, attempted=False, switch=switch)
    prepared = _keys_only(prepare_skill_grant(skill, drive, keys))
    if prepared.get("error"):
        return allow_result(prepared, enable=enable, attempted=False, switch=switch)
    attempted = enable and switch == ""
    result = _grant(skill, drive, repo, keys, progress=progress.set if progress is not None else None,
                    enable_after_grant=attempted, explicit_via="secret_form", form_installation=installation)
    return allow_result(result, enable=enable, attempted=attempted, switch=switch)


def update_pending(drive: Path, name: str) -> bool:
    """An update of the skill awaits its settlement (accepted or rolled back): its files are not yet the landing."""
    from ouroboros.skill_install_record import read_install_record, update_backup_path

    return bool((read_install_record(drive, name) or {}).get("pending_update")) or update_backup_path(drive, name).exists()


def grant_from_secret_form(drive_root: Path, skill_name: str, *, content_hash: str, installation: str,
                           keys: list[str], enable: bool, decision: dict[str, str], progress: Any = None) -> None:
    """The secret form's "Save and allow", after its save released the settings lock, as one lifecycle job: it claims
    the question's allow (``secret_requests.claim_allow``: a started step never runs twice), settles it there with
    ``_grant_shown_files``'s facts, inside the lane, so no admission slips between the check and the record. A job the
    lane never started raises with ``grant_step_not_started``: its runner only follows ``on_started`` (a busy lane's
    deadline, say), so that step provably never began."""
    from ouroboros.config import get_skills_repo_path
    from ouroboros.secret_requests import ALLOW_RUNNING, claim_allow, record_allow

    drive, repo, items, began = Path(drive_root), get_skills_repo_path(), list(keys), []

    def apply() -> dict[str, Any]:
        if not claim_allow(drive, decision):
            return {"ok": True, "skipped": True}
        facts = _grant_shown_files(drive, repo, skill_name, content_hash=content_hash, installation=installation,
                                   keys=items, enable=enable, decision=decision, progress=progress)
        record_allow(drive, decision, facts, expect=(ALLOW_RUNNING,))
        _secret_form_audit(drive, facts, decision_id=str(decision.get("decision_id") or ""), skill=skill_name,
                           content_hash=content_hash, requested_items=items)
        return {"ok": facts.get("state") in ("granted", "deferred"), **facts}

    try:
        run_lifecycle_job_blocking(
            kind="grant", target=skill_name, source="skills", runner=apply,
            options=LifecycleJobOptions(drive_root=drive, result_error=lambda value: "" if value.get("ok") else str(
                value.get("reason") or ""), progress_target=progress, on_started=began.append))
    except Exception as exc:
        exc.grant_step_not_started = not began
        raise


def grant_consent_on_admission(drive_root: Path, skill_name: str, *, repo_path: str | None = None,
                               progress: Any = None, deps_result: tuple[str, str] | None = None,
                               expected_skill: Any = None, normal_setup: Any = None) -> dict[str, Any] | None:
    """Inside the admission lane: claim and settle the owner's allow that waited for exactly these admitted files
    (``expected_skill``'s, when given), once, never over an Off. Its grant writes the keys alone; then ONE setup runs:
    the allow's own turn-on when its press named it and nothing was chosen for the switch (with the admission's
    ``deps_result``, never installed again here), else the caller's ``normal_setup``, else one ordinary extension
    reconcile when a grant landed so an already-enabled extension receives it. Returns that setup's receipt
    (``{}`` without one), or None when no allow was claimed: the caller runs its own setup then. A claimed step a
    restart cut is recorded unknown and never runs again; republishing a known result grants nothing."""
    from ouroboros.config import get_skills_repo_path
    from ouroboros.secret_requests import ALLOW_ORPHANED, allow_result
    from ouroboros.skill_grant_follow import (
        allow_installation, claim_consent, consent_of, files_accepted, finish_consent, publish_consent_outcomes,
    )

    drive = Path(drive_root)
    try:
        publish_consent_outcomes(drive, skill_name)
        waiting = consent_of(drive, skill_name)
        if waiting.get("claimed_at"):  # a step a restart cut: unknown, never run again
            finish_consent(drive, skill_name, waiting, ALLOW_ORPHANED)
        if not waiting or waiting.get("claimed_at"):
            return None
        repo = get_skills_repo_path() if repo_path is None else repo_path
        skill = find_skill(drive, skill_name, repo_path=repo)
        if (skill is None or skill.load_error or not files_accepted(skill.review, skill.content_hash)
                or (expected_skill is not None and (skill.content_hash != expected_skill.content_hash or Path(
                    skill.skill_dir).resolve() != Path(expected_skill.skill_dir).resolve()))):
            return None
        installation = allow_installation(drive, skill)
        status, consent = claim_consent(drive, skill, installation)  # other files or installation end it
    except Exception:
        log.warning("The waiting allow of %s was not read", skill_name, exc_info=True)
        return None
    if status == "interrupted":
        finish_consent(drive, skill.name, consent, ALLOW_ORPHANED)
    if status != "claimed":
        return None
    keys, enable = list(consent["keys"]), bool(consent["enable"])
    switch = owner_switch(drive, skill.name)
    ready = deps_result is None or deps_result[0] in {"installed", "not_required"}
    try:
        result = _keys_only(prepare_skill_grant(skill, drive, keys))
        if not result.get("error"):
            result = _grant(skill, drive, repo, keys, explicit_via="secret_form", form_installation=installation,
                            setup=False)
    except Exception:
        log.warning("The waiting allow of %s gave no answer", skill_name, exc_info=True)
        result = {}
    attempted, setup = bool(result.get("ok")) and enable and ready and switch == "", {}
    if attempted:
        try:
            state = _toggle(skill, drive, repo, True, "owner_ui", "enable with the secret form's grant",
                            progress=progress, prepared_deps=deps_result, only_if_unset=True)
            result["setup_code"] = str(state.get("code") or "")
            setup = {"action": state.get("extension_action"), "reason": state.get("extension_reason"),
                     "process": str(state.get("process") or ""), "load_error": state.get("load_error"),
                     "server_reconcile": str(state.get("server_reconcile") or ""), "live_loaded": None}
        except Exception:
            log.warning("Turning %s on with the form's grant gave no answer", skill.name, exc_info=True)
            result["setup_code"] = "error"
        result["enabled"] = _saved_switch(drive, skill.name)
    facts = allow_result(result, enable=enable, attempted=attempted, switch=switch,
                         enable_ready=ready) if result else ALLOW_ORPHANED
    finish_consent(drive, skill.name, consent, facts)
    _secret_form_audit(drive, facts, fulfilled=True, skill=skill.name, content_hash=skill.content_hash,
                       decision_ids=[str(row.get("decision_id") or "") for row in consent["decisions"]],
                       requested_items=keys)
    if attempted:
        return setup
    if normal_setup is not None:
        return normal_setup()
    if result.get("ok"):
        try:
            return _reconcile(skill, drive, repo)
        except Exception:
            log.warning("Reconciling %s after the form's grant failed", skill.name, exc_info=True)
    return {}


def preview_connect_mcp(ctx: Any, skill_name: str, owner_source: Any = None, *, _owner_actor: str = "") -> dict[str, Any]:
    """The passive Connect tools preview behind the same caller gates as its confirm (``skill_mcp``)."""
    from ouroboros.skill_mcp import preview_connect

    error, _source = _caller_action_error(ctx, "connect_mcp", _owner_actor, owner_source)
    return _refusal(error, 403) if error else preview_connect(canonical_data_root(ctx), skill_name)


def run_skill_action(
    ctx: Any, skill_name: str, action: str, *, expected_content_hash: str = "",
    items: list[str] | None = None, owner_source: Any = None, payload_root: str = "",
    repo_path: str | None = None, _owner_actor: str = "",
    _progress: Any = None, _enable_after_grant: bool = False, _reconcile_grant: Any = None,
    settings_digest: str = "", confirm_other_servers: bool = False,
) -> dict[str, Any]:
    """Apply one selected action through the existing lifecycle lane; ``_progress`` is the
    caller's job handle (one job binds one progress target), relayed into the job.
    ``connect_mcp`` writes the previewed MCP rows (``skill_mcp.connect``) against ``settings_digest``.
    The legacy keyword only diagnoses a source launcher straddling an update; it runs no callback."""
    if _owner_actor == "owner_launcher":
        return _refusal("Restart Ouroboros, then grant access again.", 409, code="launcher_restart_required")
    from ouroboros.config import get_skills_repo_path
    from ouroboros.tool_access import build_resolved_resource_binding

    if action not in {"grant", "enable", "disable", "attest", "delete", "connect_mcp"}:
        return _refusal("unknown skill action", 400)
    if items is not None and (not isinstance(items, list) or any(not isinstance(item, str) for item in items)):
        return _refusal("grant items must be a list of strings", 400)
    error, source = _caller_action_error(ctx, action, _owner_actor, owner_source)
    if error:
        return _refusal(error, 403)
    if not _owner_actor and action in _OWNER_ACTIONS and not expected_content_hash:
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
            return None, _refusal("skill not found", 404, code="not_found")
        if expected_content_hash and loaded.content_hash != expected_content_hash:
            return None, _refusal("selected skill revision changed; inspect it before repeating the action",
                                  code="revision_changed")
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
                "review_status": result.get("review_status"), "review_stale": result.get("review_stale"),
                "review_gate_reason": (result.get("review_gate") or {}).get("blocking_reason"),
                "requested_items": list(items or []), "ok": bool(result.get("ok")),
                "enable_after_grant": _enable_after_grant,
                "granted_key_count": len(result.get("granted_keys") or []),
                "granted_permission_count": len(result.get("granted_permissions") or []),
                "extension_action": result.get("extension_action"),
                "extension_reason": result.get("extension_reason"),
                "source_ref": {key: value for key, value in source.items() if key != "text"},
                **({"mcp_servers": [{key: item.get(key) for key in ("id", "change")} for item in result.get("servers") or []],
                    "other_servers": [item.get("id") for item in result.get("other_servers") or []],
                    "confirm_other_servers": confirm_other_servers, "saved": result.get("saved")}
                   if action == "connect_mcp" else {}),
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
            # Explicit: the Skills button (UI or CLI) for the revision it showed, or a verified owner source.
            via = (_owner_actor if _owner_actor in {"owner_ui", "owner_cli"} else "owner_source" if source else "")
            return _grant(loaded, drive, repo_path, list(items or []), progress=_progress.set if _progress is not None else None,
                          enable_after_grant=_enable_after_grant, explicit_via=via if expected_content_hash else "")
        if action == "connect_mcp":
            from ouroboros.skill_mcp import connect

            return connect(drive, loaded.name, expected_content_hash=loaded.content_hash, settings_digest=settings_digest,
                           confirm_other_servers=confirm_other_servers)
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
    if not _owner_actor and result.get("ok"):  # the mind's own action left the skill on: the work's card says over what
        from ouroboros.review_overrides import note_skill_over_review

        if action == "enable" and result.get("enabled"):
            note_skill_over_review(ctx, skill_name, result.get("review_gate"), "turned_on")
        elif action == "grant" and result.get("turn_on_when_ready") == "enabled":  # the grant let a waiting turn-on through
            try:
                granted = find_skill(drive, skill_name, repo_path=repo_path)
                if granted is not None and granted.enabled:
                    note_skill_over_review(ctx, skill_name, granted.review.gate_for(granted.content_hash), "turned_on")
            except Exception:  # the grant stands and its audit row follows; only the card's note is lost
                log.warning("A skill turned on by a grant was not named on its work's card", exc_info=True)
    return audit(result)
