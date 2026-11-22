"""What of an owner grant follows a local edit of a skill's files (owner decision 22.11.2026, C10 «Упростить»).

A grant record (``state/skills/<name>/grants.json``) names the bytes it was decided for (``content_hash``). Its keys
stay valid for the skill's current, different bytes when those are a local edit of the same installed skill
(``follows_edit``), and only then:

* the grant is this installation's decision: made for the release its install record names (``release.digest``), or
  after that release landed (``updated_at`` not earlier than ``installed_at``, which every landing writes anew, so an
  update never inherits a grant decided before it). A skill made on this computer (no install record; the external
  bucket or the connected folder) counts its own state folder's decisions, from its self-authored marker on when it
  carries one, so an earlier same-named skill's surviving grant does not follow;
* nothing installs or awaits its check: no ``pending_update`` mark (a release of a source that reviews each skill,
  decision 05.10 D-A3), no retained update (``update_backup.json``, written before an update replaces any byte and
  kept through an interrupted landing) and no landing in flight (the installer's rollback copy of the payload,
  ``<bucket>/.rollback/<name>.<32 hex>``, made before new files are published and renamed only after the new install
  record is written);
* the current manifest still declares the key (``grant_status_for_skill`` grants requested keys only).

Nothing is written for it: ``skill_loader.grant_status_for_skill``, the one reader every gate asks (script and MCP
environments, extension loads and companions, the Host Service, liveness, readiness, the Skills listing), applies it,
and ``skill_loader.save_skill_grants`` keeps what follows when an existing writer runs. Updates, built-in re-seeds,
sources, the owner's Grant access, Skip and uninstall bind exact bytes as before. Permissions never follow (owner
1A, 21.11): one the host checks at each use (``USE_TIME_PERMISSIONS``) that was granted to the edited files' earlier
version keeps nothing from loading, and each use is refused saying why (``waiting_refusal``); one acting through what
the load publishes waits for the check as before.

A custom secret saved in Settings after the grant decision (outside its ``requested_keys``; no decision at all counts
the same) is not passed and keeps nothing from running (``pending_keys``, B″); a key the decision covered and did not
grant blocks as before.
"""

from __future__ import annotations

import datetime as _dt
import logging
import pathlib
import re
from typing import Any, Dict, List, Tuple

log = logging.getLogger(__name__)

REQUESTABLE_WHEN_ABSENT = frozenset({"TELEGRAM_BOT_TOKEN"})  # requested before Settings holds it
# Enforced only by the host's check at each use (``HostServiceContext.require_permission`` behind ``POST /notify``,
# the chat, call and Presence endpoints). ``page_module``, ``context_snapshot``, ``live_conversation`` and the
# ``subscribe_event:chat.*`` topics act through what the load publishes: the load gate is their check.
USE_TIME_PERMISSIONS = frozenset({"notify_owner", "inject_chat", "call_peer", "presence"})
_OWN_LOCATIONS = frozenset({"external", "user_repo"})
_ROLLBACK_COPY = re.compile(r"[0-9a-f]{32}")


def _decided_since(grants: Dict[str, Any], start: Any) -> bool:
    """Whether the grant record was written at or after ``start`` (ISO times; unreadable: no)."""
    try:
        decided, began = (_dt.datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
                          for value in (grants.get("updated_at"), start))
    except ValueError:
        return False
    utc = _dt.timezone.utc
    return (decided if decided.tzinfo else decided.replace(tzinfo=utc)) >= (began if began.tzinfo else began.replace(tzinfo=utc))


def _landing_in_flight(skill_dir: pathlib.Path) -> bool:
    """An installer's rollback copy of this payload (``snapshot_payload_state``) that no landing finished."""
    prefix = pathlib.Path(skill_dir).name + "."
    try:
        names = [entry.name for entry in (pathlib.Path(skill_dir).parent / ".rollback").iterdir()]
    except FileNotFoundError:
        return False
    except OSError:
        return True  # unreadable: nothing shows that no landing is in flight
    return any(name.startswith(prefix) and _ROLLBACK_COPY.fullmatch(name[len(prefix):]) for name in names)


def follows_edit(drive_root: pathlib.Path, skill: Any, grants: Dict[str, Any]) -> bool:
    """Whether the keys of ``grants``, decided for other bytes, are valid for the skill's current bytes (the module
    rule). Never raises: anything unreadable binds the exact bytes, as before."""
    from ouroboros.skill_install_record import install_record_path, read_install_record, update_backup_path
    from ouroboros.skill_loader import skill_state_dir_path
    from ouroboros.utils import read_json_dict

    current, decided_for = str(getattr(skill, "content_hash", "") or ""), str(grants.get("content_hash") or "")
    if (not current or not decided_for or decided_for == current or getattr(skill, "identity_collision", False)
            or getattr(skill, "load_error", "")):
        return False
    drive = pathlib.Path(drive_root)
    try:
        if update_backup_path(drive, skill.name).exists():
            return False  # an update is retained, landing or interrupted: these files are not a local edit
        record = read_install_record(drive, skill.name)
        if record is None:
            if install_record_path(drive, skill.name).exists() or getattr(skill, "location", "") not in _OWN_LOCATIONS:
                return False  # an unreadable record, or a sourced payload without one
            if getattr(skill, "source", "") != "self_authored":
                return True  # the owner's own folder: the state folder of its name is its own
            marker = read_json_dict(skill_state_dir_path(drive, skill.name) / "self_authored.json") or {}
            return _decided_since(grants, marker.get("created_at"))
        if record.get("pending_update") or _landing_in_flight(pathlib.Path(skill.skill_dir)):
            return False
        release = record.get("release") if isinstance(record.get("release"), dict) else {}
        return decided_for == str(release.get("digest") or "") or _decided_since(grants, record.get("installed_at"))
    except Exception:
        log.debug("Grant follow check failed for %s", getattr(skill, "name", "?"), exc_info=True)
        return False


def written_bytes_follow(drive_root: pathlib.Path, name: str, grants: Dict[str, Any], content_hash: str) -> bool:
    """Whether a grant write for ``content_hash`` keeps the keys of ``grants``: they are the current bytes of the skill
    of that name and the record follows them (``follows_edit``). Never raises."""
    if not grants.get("content_hash"):
        return False
    try:
        from ouroboros.config import get_skills_repo_path
        from ouroboros.skill_loader import find_skill

        skill = find_skill(pathlib.Path(drive_root), name, repo_path=get_skills_repo_path() or None)
        return skill is not None and skill.content_hash == content_hash and follows_edit(drive_root, skill, grants)
    except Exception:
        log.debug("Grant follow check for a write failed for %s", name, exc_info=True)
        return False


def grant_view(drive_root: pathlib.Path, skill: Any, grants: Dict[str, Any], requested: List[str],
               requested_permissions: List[str]) -> Dict[str, Any]:
    """What of ``grants`` the skill's current bytes may use (keys of these bytes or following a local edit;
    permissions of these bytes with an unchanged request), what an earlier version holds, and what of the missing
    rest keeps nothing from running (``pending_keys``, ``waiting_permissions``); ``blocks`` is whether anything else
    is missing."""
    from ouroboros.contracts.plugin_api import FORBIDDEN_SKILL_SETTINGS

    same_bytes = str(grants.get("content_hash") or "") == str(getattr(skill, "content_hash", "") or "")
    follows = not same_bytes and follows_edit(drive_root, skill, grants)
    keys, permissions = set(grants.get("granted_keys") or []), set(grants.get("granted_permissions") or [])
    request_same = sorted(grants.get("requested_permissions") or []) == sorted(requested_permissions)
    granted = [key for key in requested if key in keys and (same_bytes or follows)]
    granted_permissions = [perm for perm in requested_permissions if perm in permissions and same_bytes and request_same]
    missing = [key for key in requested if key not in granted]
    missing_permissions = [perm for perm in requested_permissions if perm not in granted_permissions]
    earlier = {"keys": [key for key in missing if key in keys],
               "permissions": [perm for perm in missing_permissions if perm in permissions]}
    always_requested = {key.upper() for key in FORBIDDEN_SKILL_SETTINGS} | REQUESTABLE_WHEN_ABSENT
    covered = set(grants.get("requested_keys") or [])
    pending = [key for key in missing if key not in always_requested and key not in covered]
    waiting = [perm for perm in earlier["permissions"] if follows and perm in USE_TIME_PERMISSIONS]
    return {"granted_keys": granted, "missing_keys": missing, "granted_permissions": granted_permissions,
            "missing_permissions": missing_permissions, "pending_keys": pending, "waiting_permissions": waiting,
            "earlier_grant": earlier if grants.get("content_hash") and not same_bytes and (
                earlier["keys"] or earlier["permissions"]) else None,
            "blocks": bool(set(missing) - set(pending)) or bool(set(missing_permissions) - set(waiting))}


def blocking_lists(status: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """The keys and permissions that keep a skill from running: the missing ones beyond pending and waiting."""
    pending, waiting = set(status.get("pending_keys") or []), set(status.get("waiting_permissions") or [])
    return ([key for key in status.get("missing_keys") or [] if key not in pending],
            [perm for perm in status.get("missing_permissions") or [] if perm not in waiting])


def blocking_items(status: Dict[str, Any]) -> str:
    """``keys=[...], permissions=[...]`` of ``blocking_lists``."""
    keys, permissions = blocking_lists(status)
    return ", ".join(f"{label}={items}" for label, items in (("keys", keys), ("permissions", permissions)) if items)


def grant_notes(status: Dict[str, Any]) -> List[str]:
    """Readiness lines for what is not granted but keeps nothing from running."""
    notes = [f"key {key} not granted: the skill's last grant decision did not cover it (it was saved in Settings or "
             "declared after it), so it is not passed and the skill runs without it until the owner grants it"
             for key in status.get("pending_keys") or []]
    notes += [f"permission {perm} waits: it was granted to an earlier version of these files; the skill runs and each "
              "use of it is refused until the current files are accepted (with automatic grants on, a review or "
              "another acceptance grants it again; otherwise the owner's Grant access)"
              for perm in status.get("waiting_permissions") or []]
    return notes


def waiting_refusal(skill_name: str, permission: str) -> str:
    """The words a refused use of a waiting permission carries back to the skill."""
    return (f"skill {skill_name!r} holds the owner's {permission!r} grant for an earlier version of its files, not "
            "for its current ones; the use waits for the check of the edited version (a review or another acceptance "
            "of the current files, or the owner's Grant access), after which it applies without a new request")


__all__ = [
    "REQUESTABLE_WHEN_ABSENT", "USE_TIME_PERMISSIONS", "blocking_items", "blocking_lists", "follows_edit",
    "grant_notes", "grant_view", "waiting_refusal", "written_bytes_follow",
]
