"""Store and project exact-byte skill grants, never the saved Settings values.

``grants.json`` sits in the skill's own state directory, the one its PluginAPI and scripts are handed: it keeps the
ordinary exact-byte decisions (``skill_loader.grant_status_for_skill`` reads them through ``grant_view``) and proves
nothing about the owner. For accepted current files, a custom key outside the last decision is pending (B″): not
passed, and not a blocker by itself. A requested but ungranted key the decision covered, a core key, or a missing
permission blocks.

Continuity across the owner's Repair is the host's own receipt (``state/skill_grant_receipts.json``, outside every
skill's state directory). Only a real, successful explicit grant writes one (the Skills button, or a verified owner
source; never an automatic grant, an older decision or a log): this installation (the bucket and folder identity of a
local or Hub payload), the digest the grant was saved for and the key names. A selected Repair's byte-exact text
write extends its chain once the observed digest equals the predicted one; only the ordinary clean/warnings review of
the chain's last digest rebinds the keys the earlier decision still grants and the manifest still requests (no
permission, no new key). A new landing of the name (Replace, reinstall, another source), a local delete or an
uninstall ends the receipt; Keep is no landing. Any other write leaves the ordinary grant. This guards the host's own
grant and Repair paths; it is no OS boundary for code that runs as this user.
"""

from __future__ import annotations

import logging
import pathlib
import stat
from typing import Any, Callable, Dict, List, Optional, Tuple

from ouroboros.contracts.plugin_api import FORBIDDEN_SKILL_SETTINGS
from ouroboros.contracts.schema_versions import with_schema_version
from ouroboros.contracts.skill_payload_policy import (
    SKILL_GRANT_RECEIPTS_FILENAME as RECEIPTS_FILENAME, skill_grant_receipts_path, skill_state_dir_path,
)
from ouroboros.contracts.skill_manifest import canonical_skill_name
from ouroboros.utils import read_json_dict, update_json_locked, utc_now_iso

log = logging.getLogger(__name__)
GRANTS_FILENAME = "grants.json"
SKILL_OWNER_STATE_SCHEMA_VERSION = 1  # authored skill owner state (ABI-2); no read-time rewrite
RECEIPTS_SCHEMA_VERSION = 1
RECEIPT_BUCKETS = ("external", "ouroboroshub")  # a payload made or repaired on this installation
REQUESTABLE_WHEN_ABSENT = frozenset({"TELEGRAM_BOT_TOKEN"})  # requested before Settings holds it


def requested_core_setting_keys(env_keys: List[str], settings: Optional[Dict[str, Any]] = None) -> List[str]:
    """Return manifest-requested setting keys that require explicit owner grants.

    A reader exposing values passes the ``settings`` it read them from: the same mapping names its custom
    secrets, so a key saved between two reads can never be exposed unclassified. Otherwise this reads them."""
    if not env_keys:
        return []  # nothing requested, nothing to classify: no settings read
    forbidden_upper = {key.upper() for key in FORBIDDEN_SKILL_SETTINGS}
    try:
        from ouroboros.config import SETTINGS_DEFAULTS, load_settings
        settings = load_settings() if settings is None else settings
        custom_secret_upper = {
            str(key).upper() for key in settings
            if str(key).upper() not in SETTINGS_DEFAULTS and str(key).upper().replace("_", "").isalnum()
        }
    except Exception:
        custom_secret_upper = set()
    out: List[str] = []
    for raw_key in env_keys or []:
        key = str(raw_key or "").strip().upper()
        if key and (key in forbidden_upper or key in custom_secret_upper or key in REQUESTABLE_WHEN_ABSENT) and key not in out:
            out.append(key)
    return out



# The privileged permissions an owner grants, in the order a request lists them; subscribe_event is granted by topic.
_GRANTABLE_PERMISSIONS = ("inject_chat", "presence", "notify_owner", "page_module", "context_snapshot",
                          "live_conversation", "call_peer")
_GRANTABLE_SKILL_PERMISSIONS = frozenset({*_GRANTABLE_PERMISSIONS, *(
    f"subscribe_event:chat.{topic}" for topic in ("outbound", "typing", "photo", "video"))})


def requested_skill_permissions(
    permissions: List[str],
    subscribe_events: Optional[List[str]] = None,
) -> List[str]:
    """Return manifest-requested privileged permissions needing owner grants."""
    permission_set = {str(item or "").strip() for item in (permissions or [])}
    requested = [permission for permission in _GRANTABLE_PERMISSIONS if permission in permission_set]
    if "subscribe_event" in permission_set:
        for raw_topic in subscribe_events or []:
            topic = str(raw_topic or "").strip()
            grant = f"subscribe_event:{topic}"
            if grant in _GRANTABLE_SKILL_PERMISSIONS and grant not in requested:
                requested.append(grant)
    return requested


def _text_items(values: Any, *, upper: bool = False) -> List[str]:
    items = (str(raw or "").strip() for raw in values) if isinstance(values, list) else ()
    return list(dict.fromkeys(item.upper() if upper else item for item in items if item))


def _grant_path(drive_root: pathlib.Path, name: str) -> pathlib.Path:
    return skill_state_dir_path(drive_root, name) / GRANTS_FILENAME


def load_skill_grants(drive_root: pathlib.Path, name: str) -> Dict[str, Any]:
    data = read_json_dict(_grant_path(drive_root, name))
    if not isinstance(data, dict):
        return {"granted_keys": [], "granted_permissions": [], "updated_at": ""}
    return {
        **{field: _text_items(data.get(field), upper=field.endswith("_keys")) for field in (
            "granted_keys", "requested_keys", "granted_permissions", "requested_permissions")},
        "content_hash": str(data.get("content_hash") or ""),
        "updated_at": str(data.get("updated_at") or ""),
    }


def save_skill_grants(
    drive_root: pathlib.Path, name: str, granted_keys: List[str], *, content_hash: str,
    requested_keys: List[str], granted_permissions: List[str] | None = None,
    requested_permissions: List[str] | None = None,
) -> None:
    """Merge exact-byte decisions: the same bytes keep earlier approvals of items still requested."""
    allowed = set(requested_core_setting_keys(requested_keys))
    allowed_permissions = set(requested_permissions or [])

    path = _grant_path(drive_root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    update_json_locked(path, lambda data: _grant_decision(data, content_hash, granted_keys, allowed,
                                                         granted_permissions or [], allowed_permissions))


def _grant_decision(data, digest, keys, requested, permissions, requested_permissions):
    prior = data if digest and data.get("content_hash") == digest else {}
    keys = _text_items([*_text_items(prior.get("granted_keys")), *keys], upper=True)
    permissions = _text_items([*_text_items(prior.get("granted_permissions")), *permissions])
    return with_schema_version({
        "granted_keys": [key for key in keys if key in requested], "requested_keys": sorted(requested),
        "granted_permissions": sorted(p for p in permissions if p in requested_permissions),
        "requested_permissions": sorted(requested_permissions),
        "content_hash": str(digest or ""), "updated_at": utc_now_iso(),
    }, SKILL_OWNER_STATE_SCHEMA_VERSION)


def installation_of(drive_root: pathlib.Path, name: str, payload_dir: pathlib.Path) -> Dict[str, Any] | None:
    """The installation a receipt names: a local or Hub payload's canonical folder, by bucket and folder identity
    (in-place edits and Keep keep it). ``None`` for another bucket, a link or an unreadable folder."""
    skills = pathlib.Path(drive_root).resolve() / "skills"
    try:
        if pathlib.Path(payload_dir).is_symlink() or skills.is_symlink():
            return None
        payload_dir = pathlib.Path(payload_dir).resolve()  # a linked bucket resolves outside ``skills``
        info = payload_dir.lstat()
    except OSError:
        return None
    bucket = payload_dir.parent
    if bucket.name not in RECEIPT_BUCKETS or bucket != skills / bucket.name or payload_dir.name != name:
        return None
    if not info.st_ino or not stat.S_ISDIR(info.st_mode):
        return None
    return {"bucket": bucket.name, "directory": [info.st_dev, info.st_ino]}


def _receipt(entry: Any) -> Dict[str, Any]:
    """One stored receipt exactly as the host writes it, else ``{}``: a malformed one proves nothing."""
    from ouroboros.skill_identity import is_release_digest

    installation = entry.get("installation") if isinstance(entry, dict) else None
    directory = installation.get("directory") if isinstance(installation, dict) else None
    keys = entry.get("keys") if isinstance(entry, dict) else None
    if (not isinstance(directory, list) or len(directory) != 2 or not all(type(item) is int for item in directory)
            or installation.get("bucket") not in RECEIPT_BUCKETS or not is_release_digest(entry.get("digest"))
            or not isinstance(keys, list) or not keys or not all(isinstance(key, str) and key for key in keys)
            or ("tip" in entry and not is_release_digest(entry["tip"]))):
        return {}
    return entry


def load_receipt(drive_root: pathlib.Path, name: str) -> Dict[str, Any]:
    data = read_json_dict(skill_grant_receipts_path(drive_root)) or {}
    skills = data.get("skills") if data.get("schema_version") == RECEIPTS_SCHEMA_VERSION else None
    return _receipt(skills.get(canonical_skill_name(name))) if isinstance(skills, dict) else {}


def _change_receipt(drive_root: pathlib.Path, name: str, change: Callable[[Dict[str, Any]], Any]) -> bool:
    """Apply ``change(current)`` to one skill's receipt under the document's lock: a receipt replaces it, ``{}``
    removes it, ``None`` leaves it. A document of another schema is never rewritten. Whether it changed."""
    key, changed = canonical_skill_name(name), []

    def mutate(data):
        if data and data.get("schema_version") != RECEIPTS_SCHEMA_VERSION:
            raise ValueError("skill grant receipts of another schema")
        skills = dict(data.get("skills") or {}) if isinstance(data.get("skills"), dict) else {}
        result = change(_receipt(skills.get(key)))
        if result is None or (not result and key not in skills):
            return None
        if result:
            skills[key] = result
        else:
            del skills[key]
        changed.append(key)
        return {"schema_version": RECEIPTS_SCHEMA_VERSION, "skills": skills}

    try:
        skill_grant_receipts_path(drive_root).parent.mkdir(parents=True, exist_ok=True)
        update_json_locked(skill_grant_receipts_path(drive_root), mutate, strict_existing_dict=True)
    except (OSError, ValueError):  # no continuity is recorded; the ordinary exact-byte grant stands
        log.warning("The grant receipt of %s was not changed", key, exc_info=True)
        return False
    return bool(changed)


def record_explicit_grant(drive_root: pathlib.Path, skill: Any, keys: List[str], *, via: str) -> None:
    """The host's receipt of a real explicit grant of ``keys`` that ``grants.json`` now holds for these files."""
    installation = installation_of(drive_root, skill.name, skill.skill_dir)
    stored = load_skill_grants(drive_root, skill.name)
    held = [key for key in _text_items(list(keys), upper=True)
            if stored.get("content_hash") == skill.content_hash and key in stored["granted_keys"]]
    if not installation or not held or not skill.content_hash:
        return

    def change(current):
        same = current.get("installation") == installation and current.get("digest") == skill.content_hash
        return {"installation": installation, "digest": skill.content_hash,
                "keys": sorted({*held, *(current["keys"] if same else ())}), "via": via, "recorded_at": utc_now_iso()}

    _change_receipt(drive_root, skill.name, change)


def receipt_tip(drive_root: pathlib.Path, name: str, installation: Dict[str, Any]) -> str:
    """The digest a verified Repair write of this installation continues from ("" without its receipt)."""
    current = load_receipt(drive_root, name)
    return current.get("tip", current["digest"]) if current and current["installation"] == installation else ""


def extend_verified_edit(drive_root: pathlib.Path, name: str, installation: Dict[str, Any],
                         before: str, after: str) -> bool:
    """One verified Repair write from the receipt's last digest ``before`` to the observed ``after``."""
    def change(current):
        if not current or current["installation"] != installation or current.get("tip", current["digest"]) != before:
            return None
        rest = {key: value for key, value in current.items() if key != "tip"}
        return rest if after == current["digest"] else {**rest, "tip": after}

    return _change_receipt(drive_root, name, change)


def end_receipt(drive_root: pathlib.Path, name: str) -> None:
    """This installation ended (a new landing of the name, a local delete or an uninstall)."""
    _change_receipt(drive_root, name, lambda current: {})


def _continuing_keys(receipt, grants, requested):
    if grants.get("content_hash") != receipt["digest"]:
        return []
    standing = set(receipt["keys"]) & set(_text_items(grants.get("granted_keys"), upper=True))
    return [key for key in requested if key in standing]


def finish_local_edit_grants(drive_root: pathlib.Path, skill: Any) -> None:
    """Rebind only still-granted, still-requested keys after an ordinary review of the verified tip."""
    gate = skill.review.gate_for(skill.content_hash)
    if (skill.load_error or gate["stale"] or not files_accepted(gate) or gate["status"] not in {"clean", "warnings"}
            or skill.review.review_profile == "owner_attested"
            or not (skill.manifest.is_script() or skill.manifest.is_extension())):
        return
    installation = installation_of(drive_root, skill.name, skill.skill_dir)
    receipt = load_receipt(drive_root, skill.name)
    if not installation or receipt.get("installation") != installation or receipt.get("tip") != skill.content_hash:
        return
    requested = requested_core_setting_keys(list(skill.manifest.env_from_settings or []))
    permissions = set(requested_skill_permissions(skill.manifest.permissions, skill.manifest.subscribe_events))

    def rebind_receipt(current):
        if current != receipt:
            return None

        keys: List[str] = []

        def rebind_grants(grants):
            keys.extend(_continuing_keys(current, grants, requested))
            if not keys:
                return None
            covered = set(_text_items(grants.get("requested_keys"), upper=True)) | set(keys)
            return _grant_decision(grants, skill.content_hash, keys, set(requested) & covered, [], permissions)

        # Receipt identity under its lock, and live grant/digest under its own lock.
        # Other writers finish their grant write before taking the receipt lock.
        update_json_locked(_grant_path(drive_root, skill.name), rebind_grants, strict_existing_dict=True)
        return ({**{k: v for k, v in current.items() if k != "tip"}, "digest": skill.content_hash, "keys": sorted(keys)}
                if keys else {})

    _change_receipt(drive_root, skill.name, rebind_receipt)


def files_accepted(gate: Dict[str, Any]) -> bool:
    """A recorded acceptance or a current clean/warnings verdict, not Cyber Pro authority alone."""
    from ouroboros.skill_review_status import ACCEPTANCE_REASONS

    return bool(gate.get("executable_review")) and not gate.get("preflight_failed") and (
        gate.get("blocking_reason") in ACCEPTANCE_REASONS or (
            not gate.get("stale") and gate.get("status") in {"clean", "warnings"}))


def grant_view(content_hash: str, grants: Dict[str, Any], requested: List[str],
               requested_permissions: List[str], gate: Dict[str, Any]) -> Dict[str, Any]:
    """Exact grants, missing items, and pending custom keys for accepted current files."""
    decided_for = grants.get("content_hash")
    same_bytes = bool(content_hash) and content_hash == decided_for
    keys, permissions = set(grants.get("granted_keys") or []), set(grants.get("granted_permissions") or [])
    request_same = sorted(grants.get("requested_permissions") or []) == sorted(requested_permissions)
    granted = [key for key in requested if key in keys and same_bytes]
    granted_permissions = [perm for perm in requested_permissions if perm in permissions and same_bytes and request_same]
    missing = [key for key in requested if key not in granted]
    missing_permissions = [perm for perm in requested_permissions if perm not in granted_permissions]
    earlier = {"keys": [key for key in missing if key in keys],
               "permissions": [perm for perm in missing_permissions if perm in permissions]}
    always_requested = {key.upper() for key in FORBIDDEN_SKILL_SETTINGS} | REQUESTABLE_WHEN_ABSENT
    covered = set(grants.get("requested_keys") or []) | keys
    pending = ([key for key in missing if key not in always_requested and key not in covered]
               if content_hash and files_accepted(gate) else [])
    return {"granted_keys": granted, "missing_keys": missing, "granted_permissions": granted_permissions,
            "missing_permissions": missing_permissions, "pending_keys": pending,
            "earlier_grant": earlier if decided_for and content_hash and not same_bytes and any(earlier.values()) else None,
            "blocks": bool(set(missing) - set(pending) or missing_permissions)}


def registration_view(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Registration keeps readable scalars; structured values cross only via their public projection."""
    from ouroboros.config import SETTINGS_DEFAULTS
    from ouroboros.gateway.secret_rows import public_rows

    view: Dict[str, Any] = {}
    for key, value in settings.items():
        if not isinstance(value, (list, dict)) and not isinstance(SETTINGS_DEFAULTS.get(key), (list, dict)):
            view[key] = value
        elif (rows := public_rows(key, value)) is not None:
            view[key] = rows
    return view


def declared_catalog_settings(keys: List[str], settings: Dict[str, Any], grants: Dict[str, Any]) -> Dict[str, Any]:
    """The declared registration inputs a catalog child receives from ``settings``: a key that needs the owner's grant
    there (a core key, or a custom secret that document holds; the PluginAPI's own rule) only with that grant, so a
    pending key never crosses into the child; a public setting as ``registration_view`` shows it."""
    protected, granted = set(requested_core_setting_keys(list(keys), settings)), set(grants.get("granted_keys") or [])
    return registration_view({key: settings[key] for key in keys
                              if key in settings and (key.upper() not in protected or key.upper() in granted)})


def blocking_lists(status: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """All missing permissions and all missing keys except pending ones."""
    pending = set(status.get("pending_keys") or [])
    return ([key for key in status.get("missing_keys") or [] if key not in pending],
            list(status.get("missing_permissions") or []))


def blocking_items(status: Dict[str, Any]) -> str:
    """``keys=[...], permissions=[...]`` of ``blocking_lists``."""
    keys, permissions = blocking_lists(status)
    return ", ".join(f"{label}={items}" for label, items in (("keys", keys), ("permissions", permissions)) if items)


def pending_fact(status: Dict[str, Any]) -> str:
    """The neutral line a run, a load or a server start of the skill carries about its pending keys ("" without)."""
    keys = list(status.get("pending_keys") or [])
    if not keys:
        return ""
    many = len(keys) > 1
    return (f"key{'s' if many else ''} {', '.join(keys)} {'are' if many else 'is'} saved in Settings but not granted to "
            f"this skill; {'they were' if many else 'it was'} not passed")


def grant_notes(status: Dict[str, Any]) -> List[str]:
    """Readiness notes about keys, not a claim that the skill runs or is enabled."""
    return [f"key {key} is saved in Settings but not granted for these files; it is not passed and does not"
            " by itself stop the skill" for key in status.get("pending_keys") or []]


__all__ = [
    "RECEIPTS_FILENAME", "REQUESTABLE_WHEN_ABSENT", "blocking_items", "blocking_lists", "declared_catalog_settings",
    "end_receipt", "extend_verified_edit", "files_accepted", "finish_local_edit_grants", "grant_notes", "grant_view",
    "installation_of", "load_receipt", "pending_fact", "receipt_tip", "record_explicit_grant", "registration_view",
]
