"""Local acceptance facts beside the review verdict.

The review gate (``skill_review_status.skill_review_gate``) admits the current bytes of a
skill without a critic PASS when a host-owned fact binds them:

* the install record (``state/skills/<name>/install.json``, written by the installers and
  the built-in seeder) says the skill came unchanged from a trusted source (``trusted``)
  or that the owner skipped the review at install (``owner_skip``), and its release
  digest equals the current digest;
* an owner edit (``state/skills/<name>/owner_edit.json``) names the current digest: the
  owner's admitted Repair verified them against its predicted text edit.

Both files live in the state plane, outside the payload and its digest, and are owner
state that generic agent writes cannot forge (``SKILL_OWNER_STATE_FILENAMES``). Edits
made anywhere else leave the skill "Not reviewed yet". A skill installed from a source
that reviews each skill keeps blocker findings non-executable until the owner skips
the review (``strict_review``); records converted from earlier installs keep the behaviour they had.

The owner-edit seam is the admitted selected-skill development task
(``skill_repair_admission.advance_repair_expected_hash``). It records the revision
of a direct file-tool edit made in an owner-started task. The record admits code;
it does not issue keys or permissions for that revision.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

OWNER_EDIT_FILENAME = "owner_edit.json"
OWNER_EDIT_SCHEMA_VERSION = 1


def load_acceptance_facts(drive_root: pathlib.Path, name: str, *, install_record: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Read acceptance for one name, optionally with a verified prospective landing."""
    from ouroboros.skill_install_record import read_install_record
    from ouroboros.skill_loader import skill_state_dir_path
    from ouroboros.utils import read_json_dict

    facts: Dict[str, Any] = {}
    record = read_install_record(pathlib.Path(drive_root), name) if install_record is None else install_record
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


def carry_grants_to_current(drive_root: pathlib.Path, skill: Any, *, previous_digest: str) -> bool:
    """Carry the proven predecessor's exact grant when both request sets are unchanged.

    The built-in installer proves the predecessor before landing. This step never
    rewrites a decision for any other digest. Returns whether it carried a grant.
    """
    try:
        from ouroboros.skill_loader import (
            load_skill_grants,
            requested_core_setting_keys,
            requested_skill_permissions,
            save_skill_grants,
        )

        if skill is None or skill.load_error or not skill.content_hash:
            return False
        grants = load_skill_grants(drive_root, skill.name)
        old_hash = str(grants.get("content_hash") or "")
        if not previous_digest or old_hash != previous_digest or old_hash == skill.content_hash:
            return False
        granted_keys = list(grants.get("granted_keys") or [])
        granted_permissions = list(grants.get("granted_permissions") or [])
        if not granted_keys and not granted_permissions:
            return False
        keys = requested_core_setting_keys(list(skill.manifest.env_from_settings or []))
        permissions = requested_skill_permissions(
            list(skill.manifest.permissions or []), list(skill.manifest.subscribe_events or []),
        )
        if sorted(grants.get("requested_keys") or []) != sorted(keys):
            return False
        if sorted(grants.get("requested_permissions") or []) != sorted(permissions):
            return False
        save_skill_grants(drive_root, skill.name, granted_keys, content_hash=skill.content_hash,
                          requested_keys=keys, granted_permissions=granted_permissions,
                          requested_permissions=permissions)
        return True
    except Exception:
        log.warning("Grant carry failed for %s", getattr(skill, "name", "?"), exc_info=True)
        return False


def record_owner_edit(
    drive_root: pathlib.Path,
    skill_dir: pathlib.Path,
    *,
    expected_digest: str,
    task_id: str = "",
    attribution: str = "",
) -> Optional[str]:
    """Record that the current bytes of ``skill_dir`` are an owner edit; returns the digest.

    The admitted edit supplies its verified resulting digest; a changed payload is not stamped.
    An On skill whose edited text does not pass the Nano capacity check is not recorded either.
    Unlike a managed update this is not atomic: the edit already replaced the files, so the
    former revision is not kept and the new one stays on disk, not acting, until a smaller edit
    or an acceptance fits. No grant is rewritten here. Never raises.
    """
    try:
        from ouroboros.contracts.schema_versions import with_schema_version
        from ouroboros.nano_activation import review_publication
        from ouroboros.nano_publication import NanoPublicationRefused
        from ouroboros.skill_loader import load_skill, skill_state_dir
        from ouroboros.utils import atomic_write_json, utc_now_iso

        skill = load_skill(pathlib.Path(skill_dir), pathlib.Path(drive_root))
        if not expected_digest or skill is None or skill.load_error or skill.content_hash != expected_digest:
            return None
        try:
            with review_publication(pathlib.Path(drive_root), skill, facts={"owner_edit_digest": skill.content_hash}):
                atomic_write_json(skill_state_dir(pathlib.Path(drive_root), skill.name) / OWNER_EDIT_FILENAME,
                                  with_schema_version({
                                      "digest": skill.content_hash,
                                      "task_id": str(task_id or ""),
                                      "attribution": str(attribution or ""),
                                      "recorded_at": utc_now_iso(),
                                  }, OWNER_EDIT_SCHEMA_VERSION))
        except NanoPublicationRefused as exc:
            log.warning("Owner edit of %s is saved but not accepted (%s): %s", skill.name, exc.reason, exc.detail)
            return None
        return skill.content_hash
    except Exception:
        log.warning("Owner edit record failed for %s", skill_dir, exc_info=True)
        return None


__all__ = [
    "OWNER_EDIT_FILENAME",
    "admitted_acceptance",
    "carry_grants_to_current",
    "load_acceptance_facts",
    "record_owner_edit",
]
