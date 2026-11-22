"""What an owner grant covers: the installation it belongs to, and what of it the current bytes may use.

A grant record (``state/skills/<name>/grants.json``) names the bytes it was given for (``content_hash``), the
accepted bytes its permissions belong to (``permission_hash``, absent: ``content_hash``) and the installation
(``install_identity``): the source row the installer recorded (``install.json`` ``source.id``; empty for a
payload no installer recorded) and the payload's physical place (``data:<bucket>/<dir>`` under the data skills
root, else ``path:<absolute directory>``). The release digest is not part of it: a local edit of the payload and
an accepted update from the same source stay the same installation, while a same-named payload from another
source or another folder is another installation and inherits nothing, even with identical bytes.

Owner decision 21.11.2026 (C10, 1A): a granted key, core provider keys included, stays usable while the same
installation's manifest still declares it, through local edits. A release of a source that reviews each skill
that is not yet accepted (``install.json`` ``pending_update``: decision 05.10 D-A3, built-ins under a «review
each» source policy included) gets no key beyond its own exact accepted bytes; an uninstall ends the
installation (``ended_identity``): only a reinstall of the same installation with the same bytes keeps the
grant, as before. A record written before installations were named (legacy) binds the exact bytes only.
Permissions stay bound to accepted bytes, as the frozen manifest contract says; a partial key grant keeps the
last accepted version's permissions. A permission the host checks at each use (``USE_TIME_PERMISSIONS``) that
the owner granted to an earlier version of this installation does not keep the extension from loading: that use
is refused, saying why, until the bytes are accepted (``waiting_permissions``). A permission checked only at load
waits for acceptance as before. Only a recorded basis accepts bytes (``bytes_accepted``): Cyber Pro's own
authority, alone or with an author record, grants nothing (18.10).

A custom secret requested only because it now exists in Settings, which this installation's grant decision never
covered (it is not in the record's ``requested_keys``), is not passed and keeps nothing from running
(``pending_keys``): the skill runs with its previous access and readiness names the key. A key the decision
covered and did not grant blocks, and so does every missing key without such a decision: a deleted record is a
revocation. Every acceptance records a decision, empty included (``skill_acceptance``). An empty receipt, one
written with an empty request while a key was present, covers nothing: that key is pending, not revoked.

Pure scope logic takes its inputs from the caller; the few readers of the install record and the startup import
below read files, never through ``skill_loader``.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

log = logging.getLogger(__name__)

REQUESTABLE_WHEN_ABSENT = frozenset({"TELEGRAM_BOT_TOKEN"})  # requested before Settings holds it
# Permissions whose only enforcement is the host's check at each use (``HostServiceContext.require_permission``
# behind ``POST /notify``, the chat, call and Presence endpoints). ``page_module``, ``context_snapshot``,
# ``live_conversation`` and the ``subscribe_event:chat.*`` topics act through what the load publishes, so the
# load gate is their check and they keep it.
USE_TIME_PERMISSIONS = frozenset({"notify_owner", "inject_chat", "call_peer", "presence"})
GRANTS_FILE = "grants.json"
_IMPORT_KEPT = "grants.before-installation.json"  # the import keeps a record's original bytes beside it


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


def release_pending(drive_root: pathlib.Path, name: str) -> bool:
    """Whether ``install.json`` names a release that is not accepted yet (a review-each update or re-seed)."""
    from ouroboros.skill_install_record import read_install_record

    return bool((read_install_record(pathlib.Path(drive_root), name) or {}).get("pending_update"))


def record_scope(grants: Dict[str, Any], identity: Optional[Dict[str, str]], content_hash: str, *,
                 pending: bool) -> str:
    """``"install"`` (its keys follow this installation), ``"bytes"`` (they hold for these exact bytes only) or
    ``"none"``. Pure: the caller reads the current installation and whether its release is pending."""
    same_bytes = bool(content_hash) and str(grants.get("content_hash") or "") == str(content_hash)
    bound, ended = grants.get("install_identity"), grants.get("ended_identity")
    if isinstance(bound, dict):
        if identity is None:  # the install record cannot be read: the exact bytes binding alone holds
            return "bytes" if same_bytes else "none"
        if bound != identity:
            return "none"  # another installation, also with identical bytes
        return "install" if not pending else ("bytes" if same_bytes else "none")
    if isinstance(ended, dict) or grants.get("identity_ended_at"):
        return "bytes" if same_bytes and identity is not None and ended == identity else "none"
    return "bytes" if same_bytes else "none"  # legacy: a record of the exact bytes


def scope_for(drive_root: pathlib.Path, skill: Any, grants: Dict[str, Any]) -> str:
    """``record_scope`` of ``grants`` against the skill's current installation."""
    if not grants.get("content_hash") and not grants.get("install_identity"):
        return "none"
    return record_scope(grants, install_identity(drive_root, skill), str(getattr(skill, "content_hash", "") or ""),
                        pending=release_pending(drive_root, skill.name))


def accepted_verdict(gate: Dict[str, Any], *, strict: bool) -> bool:
    """Whether ``gate`` records an acceptance of the current bytes (owner 18.10: Cyber Pro alone accepts nothing).

    A recorded basis (trusted release, owner Skip or edit, Ouroboros's recorded check, the author's acceptance
    outside Cyber Pro) accepts; so does a finished verdict about these very bytes that admits them on its own.
    Cyber Pro's own authority, with or without an author record, and a verdict about earlier bytes never do; a
    source that reviews each skill needs a verdict without blockers."""
    from ouroboros.skill_review_status import ACCEPTANCE_REASONS, STATUS_BLOCKERS, STATUS_CLEAN, STATUS_WARNINGS

    if not gate.get("executable_review"):
        return False
    if gate.get("blocking_reason") in ACCEPTANCE_REASONS:
        return True
    status = gate.get("status")
    if gate.get("stale") or status not in (STATUS_CLEAN, STATUS_WARNINGS, STATUS_BLOCKERS):
        return False
    return status != STATUS_BLOCKERS or not strict


def bytes_accepted(skill: Any) -> bool:
    """``accepted_verdict`` of the skill's current bytes."""
    review = skill.review
    return accepted_verdict(review.gate_for(skill.content_hash),
                            strict=bool((getattr(review, "acceptance", None) or {}).get("strict_review")))


def _unique(values: Any, *, upper: bool = False) -> List[str]:
    out: List[str] = []
    for raw in values or []:
        item = str(raw or "").strip()
        item = item.upper() if upper else item
        if item and item not in out:
            out.append(item)
    return out


def merged_record(existing: Dict[str, Any], *, granted_keys: Iterable[str], allowed_keys: Iterable[str],
                  granted_permissions: Optional[Iterable[str]], allowed_permissions: Iterable[str],
                  content_hash: str, scope: Optional[str], install_identity: Optional[Dict[str, str]],
                  declared_keys: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """The record a grant write leaves (``skill_loader.save_skill_grants``). Approvals of this installation (or of
    these exact bytes) stay: a write adds, never revokes; only the current request is recorded, and a consent the
    manifest still declares stays whatever Settings holds now (``declared_keys``). ``granted_permissions=None`` is
    no permission decision: the last accepted version's permissions keep their bytes and request."""
    same_bytes = str(existing.get("content_hash") or "") == str(content_hash or "")
    scope = scope or ("bytes" if same_bytes else "none")
    keep_keys = scope == "install" or (scope == "bytes" and same_bytes)
    allowed = set(_unique(allowed_keys, upper=True))
    if declared_keys is not None and keep_keys:
        allowed |= set(_unique(existing.get("granted_keys"), upper=True)) & set(_unique(declared_keys, upper=True))
    keys = [key for key in _unique([*(existing.get("granted_keys") if keep_keys else []), *granted_keys], upper=True)
            if key in allowed]
    perm_hash = str(existing.get("permission_hash") or existing.get("content_hash") or "")
    if granted_permissions is None and scope != "none" and existing.get("granted_permissions"):
        permissions = _unique(existing.get("granted_permissions"))
        requested_permissions, perm_hash = _unique(existing.get("requested_permissions")), perm_hash
    else:
        allowed_perms = set(_unique(allowed_permissions))
        kept = existing.get("granted_permissions") if scope != "none" and perm_hash == str(content_hash or "") else []
        permissions = [perm for perm in _unique([*(kept or []), *(granted_permissions or [])]) if perm in allowed_perms]
        requested_permissions, perm_hash = sorted(allowed_perms), str(content_hash or "")
    record: Dict[str, Any] = {
        "granted_keys": keys, "requested_keys": sorted(allowed), "granted_permissions": sorted(permissions),
        "requested_permissions": sorted(requested_permissions), "content_hash": str(content_hash or ""),
    }
    if perm_hash != record["content_hash"]:
        record["permission_hash"] = perm_hash
    if install_identity:
        record["install_identity"] = dict(install_identity)
    return record


def grant_view(grants: Dict[str, Any], requested: List[str], requested_permissions: List[str], *, scope: str,
               content_hash: str) -> Dict[str, Any]:
    """What of ``grants`` the current bytes may use (``scope``: ``record_scope``), and what of the rest keeps
    nothing from running. Pure."""
    from ouroboros.contracts.plugin_api import FORBIDDEN_SKILL_SETTINGS

    recorded_keys = set(grants.get("granted_keys") or [])
    recorded_permissions = set(grants.get("granted_permissions") or [])
    keys = recorded_keys if scope != "none" else set()
    perm_hash = str(grants.get("permission_hash") or grants.get("content_hash") or "")
    request_same = sorted(grants.get("requested_permissions") or []) == sorted(requested_permissions)
    permissions = recorded_permissions if scope != "none" and perm_hash == content_hash and request_same else set()
    granted = [key for key in requested if key in keys]
    granted_permissions = [perm for perm in requested_permissions if perm in permissions]
    earlier_grant = None
    if grants.get("content_hash") and (perm_hash != content_hash or scope == "none"):
        earlier_keys = [key for key in requested if key in recorded_keys - keys]
        earlier_permissions = [perm for perm in requested_permissions
                               if perm in recorded_permissions and perm not in permissions]
        if earlier_keys or earlier_permissions:
            earlier_grant = {"keys": earlier_keys, "permissions": earlier_permissions}
    # Pending needs a decision of this installation to compare with: no record (a deleted one revokes), or one of
    # another installation, leaves every missing key blocking, as before.
    always_requested = {key.upper() for key in FORBIDDEN_SKILL_SETTINGS} | set(REQUESTABLE_WHEN_ABSENT)
    covered = set(grants.get("requested_keys") or [])
    decided_here = "requested_keys" in grants and scope != "none"
    pending = [key for key in requested if decided_here
               and key not in granted and key not in always_requested and key not in covered]
    waiting = [perm for perm in (earlier_grant or {}).get("permissions", [])
               if perm in USE_TIME_PERMISSIONS and scope == "install"]
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
    notes = [f"key {key} not granted: this installation's grant decision did not cover it (it was saved in "
             "Settings later), so the skill runs without it until the owner grants it"
             for key in status.get("pending_keys") or []]
    notes += [f"permission {perm} waits for the check of the edited version: it was granted to an earlier version "
              "of this installation's files; the skill runs, and uses that need it are refused until a review or "
              "another acceptance of the current files" for perm in status.get("waiting_permissions") or []]
    return notes


def waiting_refusal(skill_name: str, permission: str) -> str:
    """The words a refused use of a waiting permission carries back to the skill."""
    return (f"skill {skill_name!r} holds the owner's {permission!r} grant for an earlier version of its files; "
            "it waits for the check of the edited version (a review or another acceptance of these bytes), "
            "after which it applies again without a new grant")


def _grants_path(drive_root: pathlib.Path, name: str) -> pathlib.Path:
    from ouroboros.contracts.skill_manifest import canonical_skill_name

    return pathlib.Path(drive_root) / "state" / "skills" / canonical_skill_name(name) / GRANTS_FILE


def end_identity(drive_root: pathlib.Path, name: str) -> None:
    """Uninstall: the record's installation ends (``ended_identity``); only a reinstall of the same installation
    with the same bytes keeps its grant, as before. Never raises."""
    from ouroboros.utils import atomic_write_json, read_json_dict, utc_now_iso

    path = _grants_path(drive_root, name)
    try:
        data = read_json_dict(path)
        if isinstance(data, dict) and isinstance(data.get("install_identity"), dict):
            data["ended_identity"] = data.pop("install_identity")
            data["identity_ended_at"] = utc_now_iso()
            atomic_write_json(path, data)
    except Exception:
        log.warning("Could not end the grant installation of %s", name, exc_info=True)


def import_grant_identities(drive_root: pathlib.Path, find: Callable[[str], Any], *,
                            log_obj: Any = None) -> Dict[str, Any]:
    """The startup import (DEVELOPMENT «One stored format»), at the startup door before a re-seed carries grants.

    A record that names no installation is read once: one that covers its skill's current bytes gains this
    installation; one that does not (stale today) is marked checked (``install_identity: null``) and keeps binding
    its exact bytes. The original bytes stay beside the record (``grants.before-installation.json``). Done is the
    field itself, so a restored snapshot or an older body's write is imported again, and a start with nothing owed
    reads the small records only. ``find(name)`` resolves a skill the way discovery does (the connected skills
    folder included). Never raises."""
    from ouroboros.utils import atomic_write_json, read_json_dict

    logger = log_obj or log
    summary: Dict[str, Any] = {"imported": [], "checked": [], "errors": []}
    try:
        paths = sorted((pathlib.Path(drive_root) / "state" / "skills").glob(f"*/{GRANTS_FILE}"))
    except OSError:
        paths = []
    for path in paths:
        name = path.parent.name
        try:
            data = read_json_dict(path)
            if (not isinstance(data, dict) or "install_identity" in data or data.get("identity_ended_at")
                    or data.get("ended_identity")):
                continue
            skill = find(name)
            if skill is None or getattr(skill, "load_error", "") or not getattr(skill, "content_hash", ""):
                continue
            identity = install_identity(drive_root, skill)
            if identity is None:
                continue
            kept = path.with_name(_IMPORT_KEPT)
            if not kept.exists():
                kept.write_bytes(path.read_bytes())
            bound = identity if str(data.get("content_hash") or "") == skill.content_hash else None
            atomic_write_json(path, {**data, "install_identity": bound})
            summary["imported" if bound else "checked"].append(name)
        except Exception as exc:
            logger.warning("Grant installation import failed for %s", name, exc_info=True)
            summary["errors"].append(f"{name}: {type(exc).__name__}: {exc}")
    if summary["imported"]:
        logger.info("Grant installations: imported records for %d skill(s)", len(summary["imported"]))
    return summary


__all__ = [
    "REQUESTABLE_WHEN_ABSENT", "USE_TIME_PERMISSIONS", "accepted_verdict", "blocking_items", "blocking_lists",
    "bytes_accepted", "end_identity", "grant_notes", "grant_view", "import_grant_identities", "install_identity",
    "merged_record", "payload_location", "record_scope", "release_pending", "scope_for", "waiting_refusal",
]
