"""Local acceptance facts beside the review verdict.

The review gate (``skill_review_status.skill_review_gate``) admits the current bytes of a
skill without a critic PASS when a host-owned fact binds them:

* the install record (``state/skills/<name>/install.json``, written by the installers and
  the built-in seeder) says the skill came unchanged from a trusted source (``trusted``)
  or that the owner skipped the review at install (``owner_skip``), and its release
  digest equals the current digest;
* an owner edit (``state/skills/<name>/owner_edit.json``) names the current digest: the
  owner, or Ouroboros working on the owner's own request, produced these exact bytes.

Both files live in the state plane, outside the payload and its digest, and are owner
state that generic agent writes cannot forge (``SKILL_OWNER_STATE_FILENAMES``). Edits
made anywhere else leave the skill "Not reviewed yet". A skill installed from a source
that reviews each skill keeps blocker findings non-executable until the owner skips
the review (``strict_review``); records converted from earlier installs keep the behaviour they had.

The owner-edit seam is the admitted selected-skill development task
(``skill_repair_admission.advance_repair_expected_hash``): after each payload write of a
task whose run the owner's own door started, the observed digest is recorded here.
Every acceptance event re-binds the installation's grants to the accepted bytes
(``carry_grants_to_current`` through ``accept_grants``); between acceptances its keys
follow the installation and its permissions wait (``skill_grant_scope``).
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

OWNER_EDIT_FILENAME = "owner_edit.json"
OWNER_EDIT_SCHEMA_VERSION = 1


def load_acceptance_facts(drive_root: pathlib.Path, name: str) -> Dict[str, Any]:
    """Read-only projection of the acceptance facts for one skill name."""
    from ouroboros.skill_install_record import read_install_record
    from ouroboros.skill_loader import skill_state_dir_path
    from ouroboros.utils import read_json_dict

    facts: Dict[str, Any] = {}
    record = read_install_record(pathlib.Path(drive_root), name)
    if record:
        release = record.get("release") if isinstance(record.get("release"), dict) else {}
        basis = str(record.get("basis") or "")
        facts.update({
            "basis": basis,
            "release_digest": str(release.get("digest") or ""),
            "source_id": str((record.get("source") or {}).get("id") or "") if isinstance(record.get("source"), dict) else "",
            # A review-each install stays strict when Ouroboros's own check no longer admits it.
            "strict_review": basis in {"review", "self_check"} and not record.get("converted_from"),
        })
    owner_edit = read_json_dict(skill_state_dir_path(pathlib.Path(drive_root), name) / OWNER_EDIT_FILENAME)
    if owner_edit and str(owner_edit.get("digest") or ""):
        facts["owner_edit_digest"] = str(owner_edit["digest"])
    return facts


def admitted_acceptance(facts: Dict[str, Any], manifest: Any) -> Dict[str, Any]:
    """Drop the bases that would make bytes executable when the manifest fails the
    PluginAPI admission every PASS-minting path shares (ABI-1); the review facts stay."""
    if not facts or manifest is None:
        return facts
    from ouroboros.contracts.plugin_api import extension_new_pass_admission_error

    error = extension_new_pass_admission_error(manifest)
    if not error:
        return facts
    kept = {key: value for key, value in facts.items() if key not in {"release_digest", "owner_edit_digest"}}
    kept["admission_error"] = error
    return kept


def carry_grants_to_current(drive_root: pathlib.Path, skill: Any, *, vouched: bool = False) -> bool:
    """Re-bind this installation's grants to the current, accepted bytes. Callers are acceptance events
    (``carry_if_accepted``: end of a review, the author's acceptance, the owner's Skip, an accepted release; and,
    ``vouched`` for their own payload, the owner's edit and a re-seed of a built-in its source trusts).

    A record of this installation (``skill_grant_scope.record_scope`` ``"install"``) carries the granted keys the
    manifest still declares, compared with its declaration and not with what Settings holds now, and the granted
    permissions when the skill still requests exactly the same ones; it is written even empty, so a key the
    accepted manifest dropped is retired. A legacy record (no installation named) carries only where the caller
    vouches for the payload and the request is unchanged, as before. With no record at all, the empty decision of
    these bytes is recorded (``record_decision``). Returns whether grants were carried. Never raises."""
    try:
        from ouroboros.skill_grant_scope import install_identity, scope_for
        from ouroboros.skill_loader import (
            load_skill_grants, requested_core_setting_keys, requested_skill_permissions, save_skill_grants,
        )

        if skill is None or skill.load_error or not skill.content_hash:
            return False
        grants = load_skill_grants(drive_root, skill.name)
        if "requested_keys" not in grants:
            record_decision(drive_root, skill)
            return False
        permission_hash = grants.get("permission_hash") or grants.get("content_hash")
        if str(grants.get("content_hash") or "") in {"", skill.content_hash} and permission_hash in {"", skill.content_hash}:
            return False  # keys and permissions already belong to these bytes
        declared = sorted({str(key or "").strip().upper() for key in skill.manifest.env_from_settings or []})
        permissions = requested_skill_permissions(
            list(skill.manifest.permissions or []), list(skill.manifest.subscribe_events or []),
        )
        same_request = sorted(grants.get("requested_permissions") or []) == sorted(permissions)
        if isinstance(grants.get("install_identity"), dict):
            if scope_for(drive_root, skill, grants) != "install":
                return False
            keys = [key for key in grants.get("granted_keys") or [] if key in declared]
            carried = list(grants.get("granted_permissions") or []) if same_request else []
        elif vouched and not grants.get("ended_identity") and not grants.get("identity_ended_at"):
            if (sorted(grants.get("requested_keys") or [])
                    != sorted(requested_core_setting_keys(list(skill.manifest.env_from_settings or [])))
                    or not same_request or not (grants.get("granted_keys") or grants.get("granted_permissions"))):
                return False
            keys, carried = list(grants.get("granted_keys") or []), list(grants.get("granted_permissions") or [])
        else:
            return False
        # The request the owner's decision covered stays the covered one: a key Settings gained later stays
        # pending (``skill_grant_scope``) instead of turning into a declined one.
        save_skill_grants(drive_root, skill.name, keys, content_hash=skill.content_hash,
                          requested_keys=[key for key in grants.get("requested_keys") or [] if key in declared],
                          granted_permissions=carried, requested_permissions=permissions,
                          install_identity=install_identity(drive_root, skill), scope="install",
                          declared_keys=[key for key in keys if key in declared])
        return True
    except Exception:
        log.warning("Grant carry failed for %s", getattr(skill, "name", "?"), exc_info=True)
        return False


def record_decision(drive_root: pathlib.Path, skill: Any) -> bool:
    """A skill that declares keys and has no grant record: record that its accepted bytes were granted nothing of
    their current request, so a key saved later finds a decision and is pending, not a reason to stop it, while a
    deleted record stays a revocation (``skill_grant_scope``). Returns whether one was written. Never raises."""
    try:
        from ouroboros.skill_grant_scope import install_identity
        from ouroboros.skill_loader import load_skill_grants, requested_core_setting_keys, save_skill_grants

        if not skill.manifest.env_from_settings or "requested_keys" in load_skill_grants(drive_root, skill.name):
            return False
        save_skill_grants(drive_root, skill.name, [], content_hash=skill.content_hash,
                          requested_keys=requested_core_setting_keys(list(skill.manifest.env_from_settings or [])),
                          install_identity=install_identity(drive_root, skill))
        return True
    except Exception:
        log.warning("Grant decision record failed for %s", getattr(skill, "name", "?"), exc_info=True)
        return False


def admit_pending_release(drive_root: pathlib.Path, skill: Any) -> bool:
    """A release that waited for its check (``install.json`` ``pending_update``) whose exact bytes a recorded basis
    now accepts loses the mark, so this installation's grants can follow it. A source update keeps its own
    settlement (``skill_install_record.settle_pending_update``, which owns the kept snapshot); this clears the mark
    of a re-seeded built-in, which keeps none. Never raises."""
    try:
        from ouroboros.skill_grant_scope import bytes_accepted
        from ouroboros.skill_install_record import read_install_record, update_backup_path, update_install_record

        pending = (read_install_record(pathlib.Path(drive_root), skill.name) or {}).get("pending_update")
        if (not isinstance(pending, dict) or pending.get("digest") != skill.content_hash
                or update_backup_path(pathlib.Path(drive_root), skill.name).exists() or not bytes_accepted(skill)):
            return False
        return update_install_record(pathlib.Path(drive_root), skill.name, pending_update=None) is not None
    except Exception:
        log.warning("Pending release admission failed for %s", getattr(skill, "name", "?"), exc_info=True)
        return False


def carry_if_accepted(drive_root: pathlib.Path, skill: Any) -> bool:
    """At an acceptance event: when a recorded basis accepts the current bytes (``skill_grant_scope.bytes_accepted``,
    the condition the automatic grant reads), a re-seeded release waiting for this check is admitted and this
    installation's grants follow (``carry_grants_to_current``). Never raises."""
    from ouroboros.skill_grant_scope import bytes_accepted

    try:
        if skill is None or skill.load_error or not bytes_accepted(skill):
            return False
    except Exception:
        log.warning("Acceptance read failed for %s", getattr(skill, "name", "?"), exc_info=True)
        return False
    admit_pending_release(pathlib.Path(drive_root), skill)
    return carry_grants_to_current(pathlib.Path(drive_root), skill)


def accept_grants(drive_root: pathlib.Path, skill: Any) -> Any:
    """An acceptance event of the current bytes: ``carry_if_accepted``, then the owner's automatic grant for what
    is still requested (``auto_grant_if_enabled``, which reads its own setting); returns its outcome."""
    from ouroboros.skill_loader import auto_grant_if_enabled

    carry_if_accepted(drive_root, skill)
    return auto_grant_if_enabled(drive_root, skill)


def record_missing_decisions(drive_root: pathlib.Path, skills: Any) -> List[str]:
    """Once per data root (``state/skills/.grant-decisions.json``): installed skills whose bytes a recorded basis
    accepts, that declare keys and have no grant record, get the decision acceptance records now, so a key saved
    later does not stop them. A restored snapshot without the mark runs it again. Never raises."""
    from ouroboros.skill_grant_scope import bytes_accepted
    from ouroboros.utils import atomic_write_json, utc_now_iso

    marker = pathlib.Path(drive_root) / "state" / "skills" / ".grant-decisions.json"
    if marker.exists():
        return []
    recorded = []
    for skill in skills or []:
        try:
            if (not skill.identity_collision and not skill.load_error and skill.content_hash
                    and bytes_accepted(skill) and record_decision(pathlib.Path(drive_root), skill)):
                recorded.append(skill.name)
        except Exception:
            log.warning("Grant decision migration failed for %s", getattr(skill, "name", "?"), exc_info=True)
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(marker, {"schema_version": 1, "recorded_at": utc_now_iso(), "recorded": recorded})
    except OSError:
        log.warning("Grant decision migration mark not written", exc_info=True)
    return recorded


def record_owner_edit(
    drive_root: pathlib.Path,
    skill_dir: pathlib.Path,
    *,
    task_id: str = "",
    attribution: str = "",
) -> Optional[str]:
    """Record that the current bytes of ``skill_dir`` are an owner edit; returns the digest.

    Callers are the owner-edit seam only (see the module docstring). The owner's edit is an
    acceptance of these bytes: this installation's grants follow it (``carry_grants_to_current``,
    vouched for this payload). Never raises.
    """
    try:
        from ouroboros.contracts.schema_versions import with_schema_version
        from ouroboros.skill_loader import load_skill, skill_state_dir
        from ouroboros.utils import atomic_write_json, utc_now_iso

        skill = load_skill(pathlib.Path(skill_dir), pathlib.Path(drive_root))
        if skill is None or skill.load_error or not skill.content_hash:
            return None
        atomic_write_json(skill_state_dir(pathlib.Path(drive_root), skill.name) / OWNER_EDIT_FILENAME,
                          with_schema_version({
                              "digest": skill.content_hash,
                              "task_id": str(task_id or ""),
                              "attribution": str(attribution or ""),
                              "recorded_at": utc_now_iso(),
                          }, OWNER_EDIT_SCHEMA_VERSION))
        carry_grants_to_current(pathlib.Path(drive_root), skill, vouched=True)
        return skill.content_hash
    except Exception:
        log.warning("Owner edit record failed for %s", skill_dir, exc_info=True)
        return None


__all__ = [
    "OWNER_EDIT_FILENAME",
    "accept_grants",
    "admit_pending_release",
    "admitted_acceptance",
    "carry_grants_to_current",
    "carry_if_accepted",
    "load_acceptance_facts",
    "record_decision",
    "record_missing_decisions",
    "record_owner_edit",
]
