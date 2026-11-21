"""What an owner grant covers: the installation it belongs to, and what of it the current bytes may use.

A grant record (``state/skills/<name>/grants.json``) names the bytes it was given for
(``content_hash``) and the installation (``install_identity``): the source row the installer
recorded (``install.json`` ``source.id``; empty for a payload no installer recorded) and the
payload's physical place (``data:<bucket>/<dir>`` under the data skills root, else
``path:<absolute directory>``). The release digest is not part of it: a local edit of the payload
and an accepted update from the same source stay the same installation, while a same-named payload
from another source or another folder is another installation and inherits nothing.

Owner decision 21.11.2026 (C10, 1A): a granted key stays usable while the same installation's
manifest still declares it, through local edits. Two conditions end that: ``install.json`` holds a
``pending_update`` (a release of a source that reviews each skill, not yet accepted: decision 05.10
D-A3, those bytes get no keys) and uninstall (``identity_ended_at``, written through the uninstall
tombstone, so a later same-named skill inherits no key; the exact ``content_hash`` binding of the
same bytes survives uninstall as before). Permissions stay bound to accepted bytes (``content_hash``,
the frozen manifest contract). A permission the host checks at each use (``USE_TIME_PERMISSIONS``)
that the owner granted to an earlier version of this installation does not keep the extension from
loading: that use is refused, saying why, until the bytes are accepted (``waiting_permissions``).
A permission checked only at load waits for acceptance as before.

A custom secret requested only because it now exists in Settings, which the grant decision never
covered (it is not in the record's ``requested_keys``), is not passed and keeps nothing from running
(``pending_keys``): the skill runs with its previous access and readiness names the key.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# Permissions whose only enforcement is the host's check at each use (``HostServiceContext.require_permission``
# behind ``POST /notify``, the chat, call and Presence endpoints). ``page_module``, ``context_snapshot``,
# ``live_conversation`` and the ``subscribe_event:chat.*`` topics act through what the load publishes, so the
# load gate is their check and they keep it.
USE_TIME_PERMISSIONS = frozenset({"notify_owner", "inject_chat", "call_peer", "presence"})


def payload_location(drive_root: pathlib.Path, skill_dir: Any) -> str:
    """The physical place of a payload: relative under the data skills root, absolute elsewhere."""
    resolved = pathlib.Path(skill_dir).resolve()
    try:
        return "data:" + resolved.relative_to((pathlib.Path(drive_root) / "skills").resolve()).as_posix()
    except ValueError:
        return "path:" + str(resolved)


def install_identity(drive_root: pathlib.Path, skill: Any) -> Optional[Dict[str, str]]:
    """``{"source", "location"}`` of the skill's installation, or ``None`` when it cannot be told (an
    install record that exists but cannot be read, a name collision): the exact bytes binding alone holds."""
    from ouroboros.skill_install_record import install_record_path, read_install_record

    if skill is None or getattr(skill, "identity_collision", False) or not getattr(skill, "skill_dir", None):
        return None
    record = read_install_record(pathlib.Path(drive_root), skill.name)
    if record is None and install_record_path(pathlib.Path(drive_root), skill.name).exists():
        return None
    source = (record or {}).get("source")
    return {"source": str(source.get("id") or "") if isinstance(source, dict) else "",
            "location": payload_location(drive_root, skill.skill_dir)}


def _pending_release(drive_root: pathlib.Path, name: str) -> bool:
    from ouroboros.skill_install_record import read_install_record

    return bool((read_install_record(pathlib.Path(drive_root), name) or {}).get("pending_update"))


def identity_holds(drive_root: pathlib.Path, skill: Any, grants: Dict[str, Any]) -> bool:
    """Whether ``grants`` belongs to the skill's current installation and its keys may follow it."""
    recorded = grants.get("install_identity")
    if not isinstance(recorded, dict) or grants.get("identity_ended_at"):
        return False
    current = install_identity(drive_root, skill)
    return current is not None and recorded == current and not _pending_release(drive_root, skill.name)


def grant_view(drive_root: pathlib.Path, skill: Any, grants: Dict[str, Any], requested: List[str],
               requested_permissions: List[str]) -> Dict[str, Any]:
    """What of ``grants`` the current bytes may use, and what of the rest keeps nothing from running."""
    from ouroboros.contracts.plugin_api import FORBIDDEN_SKILL_SETTINGS
    from ouroboros.skill_loader import REQUESTABLE_WHEN_ABSENT

    same_bytes = str(grants.get("content_hash") or "") == str(skill.content_hash or "")
    same_install = not same_bytes and identity_holds(drive_root, skill, grants)
    request_same = sorted(grants.get("requested_permissions") or []) == sorted(requested_permissions)
    keys = set(grants.get("granted_keys") or []) if same_bytes or same_install else set()
    permissions = set(grants.get("granted_permissions") or []) if same_bytes and request_same else set()
    granted = [key for key in requested if key in keys]
    granted_permissions = [perm for perm in requested_permissions if perm in permissions]
    earlier_grant = None
    if grants.get("content_hash") and not same_bytes:
        earlier_keys = [key for key in requested if key in set(grants.get("granted_keys") or []) - keys]
        earlier_permissions = [perm for perm in requested_permissions
                               if perm in set(grants.get("granted_permissions") or [])]
        if earlier_keys or earlier_permissions:
            earlier_grant = {"keys": earlier_keys, "permissions": earlier_permissions}
    always_requested = {key.upper() for key in FORBIDDEN_SKILL_SETTINGS} | set(REQUESTABLE_WHEN_ABSENT)
    covered = set(grants.get("requested_keys") or [])
    pending = [key for key in requested
               if key not in granted and key not in always_requested and key not in covered]
    waiting = [perm for perm in (earlier_grant or {}).get("permissions", [])
               if perm in USE_TIME_PERMISSIONS and same_install]
    return {"granted_keys": granted, "granted_permissions": granted_permissions, "earlier_grant": earlier_grant,
            "pending_keys": pending, "waiting_permissions": waiting}


def blocking_lists(status: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """The keys and permissions that keep a skill from running: missing beyond pending and waiting."""
    keys = [key for key in status.get("missing_keys") or [] if key not in (status.get("pending_keys") or [])]
    permissions = [perm for perm in status.get("missing_permissions") or []
                   if perm not in (status.get("waiting_permissions") or [])]
    return keys, permissions


def blocking_items(status: Dict[str, Any]) -> str:
    """``keys=[...], permissions=[...]`` of ``blocking_lists``."""
    keys, permissions = blocking_lists(status)
    return ", ".join(f"{label}={items}" for label, items in (("keys", keys), ("permissions", permissions)) if items)


def grant_notes(status: Dict[str, Any]) -> List[str]:
    """Readiness lines for what is not granted yet but keeps nothing from running."""
    notes = [f"key {key} not granted: it was saved in Settings after this skill's grant, so the skill runs "
             "without it until the owner grants it" for key in status.get("pending_keys") or []]
    notes += [f"permission {perm} waits for the check of the edited version: it was granted to an earlier version "
              "of this installation's files; the skill runs, and uses that need it are refused until a review or "
              "another acceptance of the current files" for perm in status.get("waiting_permissions") or []]
    return notes


def waiting_refusal(skill_name: str, permission: str) -> str:
    """The words a refused use of a waiting permission carries back to the skill."""
    return (f"skill {skill_name!r} holds the owner's {permission!r} grant for an earlier version of its files; "
            "it waits for the check of the edited version (a review or another acceptance of these bytes), "
            "after which it applies again without a new grant")


def end_identity(drive_root: pathlib.Path, name: str) -> None:
    """Uninstall: the record's keys no longer follow an installation (the exact bytes binding stays). Never raises."""
    from ouroboros.skill_loader import GRANTS_FILENAME, skill_state_dir_path
    from ouroboros.utils import atomic_write_json, read_json_dict, utc_now_iso

    path = skill_state_dir_path(pathlib.Path(drive_root), name) / GRANTS_FILENAME
    try:
        data = read_json_dict(path)
        if isinstance(data, dict) and "install_identity" in data:
            data.pop("install_identity", None)
            data["identity_ended_at"] = utc_now_iso()
            atomic_write_json(path, data)
    except Exception:
        log.warning("Could not end the grant installation of %s", name, exc_info=True)


def import_grant_identities(drive_root: pathlib.Path, *, log_obj: Any = None,
                            repo_path: Optional[str] = None) -> Dict[str, Any]:
    """The startup import (DEVELOPMENT «One stored format»): a grant record that covers the current bytes of
    its skill and names no installation gains this one; a record already stale stays as it is and an ended one
    stays ended. Done is the field itself, so a restored snapshot or an older body's write is imported again at
    the next startup. Without it the first edit after this update would cost every skill its keys. Never raises."""
    from ouroboros.skill_loader import GRANTS_FILENAME, discover_skills, skill_state_dir_path
    from ouroboros.utils import atomic_write_json, read_json_dict

    logger = log_obj or log
    summary: Dict[str, Any] = {"imported": [], "errors": []}
    try:
        skills = discover_skills(pathlib.Path(drive_root), repo_path=repo_path)
    except Exception as exc:
        logger.warning("Grant installation import could not list skills", exc_info=True)
        summary["errors"].append(f"discovery: {type(exc).__name__}: {exc}")
        return summary
    for skill in skills:
        if skill.identity_collision or skill.load_error or not skill.content_hash:
            continue
        path = skill_state_dir_path(pathlib.Path(drive_root), skill.name) / GRANTS_FILENAME
        try:
            data = read_json_dict(path)
            if (not isinstance(data, dict) or "install_identity" in data or data.get("identity_ended_at")
                    or str(data.get("content_hash") or "") != skill.content_hash):
                continue
            identity = install_identity(drive_root, skill)
            if identity is None:
                continue
            atomic_write_json(path, {**data, "install_identity": identity})
            summary["imported"].append(skill.name)
        except Exception as exc:
            logger.warning("Grant installation import failed for %s", skill.name, exc_info=True)
            summary["errors"].append(f"{skill.name}: {type(exc).__name__}: {exc}")
    if summary["imported"]:
        logger.info("Grant installations: imported records for %d skill(s)", len(summary["imported"]))
    return summary


__all__ = [
    "USE_TIME_PERMISSIONS", "blocking_items", "blocking_lists", "end_identity", "grant_notes", "import_grant_identities", "grant_view", "identity_holds", "install_identity",
    "payload_location", "waiting_refusal",
]
