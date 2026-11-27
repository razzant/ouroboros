"""Store and project exact-byte skill grants, never the saved Settings values.

``grants.json`` sits in the skill's own state directory, the one its PluginAPI and scripts are handed: it keeps the
ordinary exact-byte decisions (``skill_loader.grant_status_for_skill`` reads them through ``grant_view``) and proves
nothing about the owner. Keys only the secret form granted stay apart (``form``: the installation and digest they were
granted for, and their names): they count for those accepted files of that installation alone, so a reinstall, older
bytes, or another landing read none of them while independent ordinary decisions stand. For accepted current files, a custom key outside the last
decision is pending (B″): not passed, and not a blocker by itself. A requested but ungranted key the decision covered,
a core key, or a missing permission blocks, except a held one (below), which is pending as well.

Continuity across the owner's Repair is the host's own receipt (``state/skill_grant_receipts.json``, outside every
skill's state directory). Only a real, successful explicit grant writes one (the Skills button, or a verified owner
source; never an automatic grant, an older decision or a log): this installation (the bucket and folder identity of a
local or Hub payload), the digest the grant was saved for and the key names. A selected Repair's byte-exact text
write extends its chain once the observed digest equals the predicted one; only the ordinary clean/warnings review of
the chain's last digest rebinds the keys the earlier decision still grants and the manifest still requests (no
permission, no new key). For an extension the receipt then holds each requested permission the host checks at every
use that was granted to the earlier files (``notify_owner``): it stays ungranted, so the extension loads with its keys
and the host refuses that use until the owner grants it for these files. A new landing of the name (Replace,
reinstall, another source), a local delete or an uninstall ends the receipt; Keep is no landing. Any other write
leaves the ordinary grant. This guards the host's own grant and Repair paths; it is no OS boundary for code that runs
as this user.

The same document keeps the owner's consent still waiting for its files (``consents``): the secret form's explicit
"Save and allow" for files not admitted yet records the known installation (``allow_installation``), the exact digest
it showed, the key names, whether it named turning on, and the questions it answered, never a value and never a Skip.
Admission of those files claims the consent once. Completed or ended consents leave metadata-only settlements, each
with the readers taken when its result was decided, until their exact questions record the results. A new allow or
installation can replace the waiting slot, not those facts; after a restart a claimed step is recorded unknown and
never runs again, while an unclaimed consent for already-admitted files finishes once (``recover_consent``). A press
whose card no live step settled reads what these records say (``orphan_facts``): still waiting while its consent does.
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
_LANDING_BUCKETS = {"builtin": "native", "registry": "clawhub"}  # the folder a record of these sources lands
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
# Checked by the host at each use against the grant for the current files (``POST /notify``: ``require_permission``),
# not only before a load; a Repair receipt may hold one for an extension, never grant it. Any other missing permission,
# and any for a script, still blocks as before (for several the load gate is the only check).
_USE_CHECKED_PERMISSIONS = frozenset({"notify_owner"})


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


def _form_keys(data: Any, installation: str, digest: str | None = None) -> List[str]:
    """The keys only the secret form granted, while ``installation`` (and ``digest``, when given) match."""
    form = data.get("form") if isinstance(data, dict) else None
    if not installation or not isinstance(form, dict) or form.get("installation") != installation:
        return []
    if digest is not None:
        stored = form["digest"] if "digest" in form else data.get("content_hash")
        if not digest or stored != digest:
            return []
    return _text_items(form.get("keys"), upper=True)


def _preserve_ordinary_with_form(data: Any, form: Dict[str, Any] | None) -> Dict[str, Any]:
    """Keep the ordinary exact-byte decision (including its coverage on older bytes) untouched beside ``form``."""
    prior = data if isinstance(data, dict) else {}
    return with_schema_version({
        "granted_keys": _text_items(prior.get("granted_keys"), upper=True),
        "requested_keys": _text_items(prior.get("requested_keys"), upper=True),
        "granted_permissions": _text_items(prior.get("granted_permissions")),
        "requested_permissions": _text_items(prior.get("requested_permissions")),
        "content_hash": str(prior.get("content_hash") or ""),
        "updated_at": utc_now_iso(),
        **({"form": form} if isinstance(form, dict) and form.get("keys") else {}),
    }, SKILL_OWNER_STATE_SCHEMA_VERSION)


def load_skill_grants(drive_root: pathlib.Path, name: str, *, skill: Any = None) -> Dict[str, Any]:
    """The exact-byte decision. With the installed ``skill`` it also attaches ``form_granted_keys`` while these are
    the accepted files of the installation and digest the secret form granted them for, without altering the ordinary
    decision's own digest or coverage. Without, the ordinary decision alone (what a carry, a merge and the Skills
    page's earlier decision read)."""
    data = read_json_dict(_grant_path(drive_root, name))
    if not isinstance(data, dict):
        return {"granted_keys": [], "granted_permissions": [], "updated_at": ""}
    out = {
        **{field: _text_items(data.get(field), upper=field.endswith("_keys")) for field in (
            "granted_keys", "requested_keys", "granted_permissions", "requested_permissions")},
        "content_hash": str(data.get("content_hash") or ""),
        "updated_at": str(data.get("updated_at") or ""),
    }
    if (skill is not None and data.get("form") and skill.name == name and not skill.load_error
            and skill.content_hash and files_accepted(skill.review, skill.content_hash)):
        scoped = _form_keys(data, allow_installation(drive_root, skill), skill.content_hash)
        if scoped:
            out["form_granted_keys"] = scoped
    return out


def save_skill_grants(
    drive_root: pathlib.Path, name: str, granted_keys: List[str], *, content_hash: str,
    requested_keys: List[str], granted_permissions: List[str] | None = None,
    requested_permissions: List[str] | None = None, form_installation: str = "",
) -> None:
    """Merge exact-byte decisions: the same bytes keep earlier approvals of items still requested. A secret form's grant
    (``form_installation``) stays apart from the ordinary decision, bound to that installation and digest (``form``)."""
    allowed = set(requested_core_setting_keys(requested_keys))
    allowed_permissions = set(requested_permissions or [])

    def decide(data):
        if not form_installation:
            return _grant_decision(data, content_hash, granted_keys, allowed, granted_permissions or [],
                                   allowed_permissions)
        if not content_hash:
            return _preserve_ordinary_with_form(data, None)
        scoped = sorted(({*_form_keys(data, form_installation, content_hash),
                          *_text_items(granted_keys, upper=True)}) & allowed)
        form = {"installation": form_installation, "digest": str(content_hash), "keys": scoped} if scoped else None
        return _preserve_ordinary_with_form(data, form)

    path = _grant_path(drive_root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    update_json_locked(path, decide)


def _grant_decision(data, digest, keys, requested, permissions, requested_permissions, *, form=None):
    prior = data if digest and data.get("content_hash") == digest else {}
    keys = _text_items([*_text_items(prior.get("granted_keys")), *keys], upper=True)
    granted_keys = [key for key in keys if key in requested]
    permissions = _text_items([*_text_items(prior.get("granted_permissions")), *permissions])
    if form is None:
        existing = data.get("form") if isinstance(data, dict) else None
        form_digest = existing["digest"] if isinstance(existing, dict) and "digest" in existing else (
            data.get("content_hash") if isinstance(data, dict) else None)
        if isinstance(existing, dict) and digest and form_digest == digest:
            kept = [key for key in _text_items(existing.get("keys"), upper=True) if key not in granted_keys]
            form = {**existing, "digest": str(digest), "keys": kept} if kept else None
    return with_schema_version({
        "granted_keys": granted_keys, "requested_keys": sorted(requested),
        "granted_permissions": sorted(p for p in permissions if p in requested_permissions),
        "requested_permissions": sorted(requested_permissions),
        "content_hash": str(digest or ""), "updated_at": utc_now_iso(),
        **({"form": form} if isinstance(form, dict) and form.get("keys") else {}),
    }, SKILL_OWNER_STATE_SCHEMA_VERSION)


def installation_of(drive_root: pathlib.Path, name: str, payload_dir: pathlib.Path) -> Dict[str, Any] | None:
    """The installation a receipt names: a local or Hub payload's canonical folder, by bucket and folder identity
    (in-place edits and Keep keep it). ``None`` for another bucket, a link or an unreadable folder."""
    return _folder_identity(drive_root, name, payload_dir, RECEIPT_BUCKETS)


def _folder_identity(drive_root: Any, name: str, payload_dir: Any, buckets: Tuple[str, ...]) -> Dict[str, Any] | None:
    skills = pathlib.Path(drive_root).resolve() / "skills"
    try:
        if pathlib.Path(payload_dir).is_symlink() or skills.is_symlink():
            return None
        payload_dir = pathlib.Path(payload_dir).resolve()  # a linked bucket resolves outside ``skills``
        info = payload_dir.lstat()
    except OSError:
        return None
    bucket = payload_dir.parent
    if bucket.name not in buckets or bucket != skills / bucket.name or payload_dir.name != name:
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


def _change_document(
    drive_root: pathlib.Path, key: str,
    change: Callable[[Dict[str, Any], Dict[str, Any], Dict[str, Any]], bool],
    *, unchanged_ok: bool = False,
) -> bool:
    """Change receipts, consents and results under one lock; unchanged_ok also acknowledges a read-only no-op."""
    changed: List[str] = []

    def mutate(data):
        if data and data.get("schema_version") != RECEIPTS_SCHEMA_VERSION:
            raise ValueError("skill grant receipts of another schema")
        skills, consents, settlements = (
            dict(data[field]) if isinstance(data.get(field), dict) else {}
            for field in ("skills", "consents", "settlements")
        )
        if not change(skills, consents, settlements):
            return None
        changed.append(key)
        return {"schema_version": RECEIPTS_SCHEMA_VERSION, "skills": skills,
                **({"consents": consents} if consents else {}),
                **({"settlements": settlements} if settlements else {})}

    try:
        skill_grant_receipts_path(drive_root).parent.mkdir(parents=True, exist_ok=True)
        update_json_locked(skill_grant_receipts_path(drive_root), mutate, strict_existing_dict=True)
    except (OSError, ValueError):
        log.warning("The grant receipt document was not changed for %s", key, exc_info=True)
        return False
    return bool(changed) or unchanged_ok


def _change_receipt(drive_root: pathlib.Path, name: str, change: Callable[[Dict[str, Any]], Any]) -> bool:
    """Apply ``change(current)`` to one skill's receipt under the document's lock: a receipt replaces it, ``{}``
    removes it, ``None`` leaves it. Whether it changed."""
    key = canonical_skill_name(name)

    def apply(skills, _consents, _settlements):
        result = change(_receipt(skills.get(key)))
        if result is None or (not result and key not in skills):
            return False
        if result:
            skills[key] = result
        else:
            del skills[key]
        return True

    return _change_document(drive_root, key, apply)


def record_explicit_grant(drive_root: pathlib.Path, skill: Any, keys: List[str], *, via: str) -> None:
    """The host's receipt of a real explicit grant of ``keys`` that ``grants.json`` now holds for these files."""
    installation = installation_of(drive_root, skill.name, skill.skill_dir)
    stored = load_skill_grants(drive_root, skill.name, skill=skill)  # the form's own keys too
    effective = set(stored.get("form_granted_keys") or [])
    if stored.get("content_hash") == skill.content_hash:
        effective.update(stored.get("granted_keys") or [])
    held = [key for key in _text_items(list(keys), upper=True) if key in effective]
    if not installation or not held or not skill.content_hash:
        return

    def change(current):
        same = current.get("installation") == installation and current.get("digest") == skill.content_hash
        return {"installation": installation, "digest": skill.content_hash,
                "keys": sorted({*held, *(current["keys"] if same else ())}), "via": via, "recorded_at": utc_now_iso(),
                **({"held_permissions": current["held_permissions"]} if same and "held_permissions" in current else {})}

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


def end_receipt(drive_root: pathlib.Path, name: str, *, reason: str = "installation_ended",
                restored_payload: pathlib.Path | None = None) -> bool:
    """End the receipt and waiting right durably, or confirm there is nothing to end; card delivery is separate.

    A retry of rollback settlement leaves any later decision for the already restored installation alone.
    """
    key = canonical_skill_name(name)
    owed = False
    restored = installation_token(drive_root, key, restored_payload) if restored_payload is not None else ""
    folder = installation_of(drive_root, key, restored_payload) if restored_payload is not None else None
    waiting = consent_of(drive_root, key)
    readers = _readers_now(drive_root, waiting) if waiting.get("installation") != restored else {}

    def end(skills, consents, settlements):
        nonlocal owed
        consent = _consent(consents.get(key))
        if not restored or consent.get("installation") != restored:
            consent = _consent(consents.pop(key, None))
        else:
            consent = {}
        had_receipt = (skills.pop(key, None) is not None
                       if not folder or _receipt(skills.get(key)).get("installation") != folder else False)
        if consent:
            _remember_settlements(settlements, key, _settlement_rows(key, consent, _ended_facts(
                consent, reason), readers if consent == waiting else {}))
        owed = bool(settlements.get(key))
        return had_receipt or bool(consent)

    recorded = _change_document(drive_root, key, end, unchanged_ok=True)
    if recorded and owed:
        publish_consent_outcomes(drive_root, key)
    return recorded


def end_superseded_consent(drive_root: pathlib.Path, name: str, digest: str) -> None:
    """Files other than the waiting consent's were admitted (an owner-verified edit): that consent can never be
    granted now, so it ends as ``skill_changed`` with its readers then. A claimed one stays with its own step."""
    key = canonical_skill_name(name)
    waiting = consent_of(drive_root, key)
    if not waiting or waiting.get("claimed_at") or waiting["digest"] == digest:
        return
    readers = _readers_now(drive_root, waiting)

    def end(_skills, consents, settlements):
        if _consent(consents.get(key)) != waiting:
            return False
        del consents[key]
        _remember_settlements(settlements, key, _settlement_rows(
            key, waiting, {"state": "refused", "reason": "skill_changed"}, readers))
        return True

    if _change_document(drive_root, key, end):
        publish_consent_outcomes(drive_root, key)


def allow_installation(drive_root: pathlib.Path, skill: Any) -> str:
    """The installation an owner's allow binds to, as a short token of host-owned facts (``installation_token``): its
    canonical folder's identity and its install record's landing (``landing_of``: a reinstall of the same release is
    another). ``""`` when that folder is not proven to be this installation: nothing is bound to an unknown one."""
    return installation_token(drive_root, skill.name, skill.skill_dir)


def installation_token(drive_root: pathlib.Path, name: str, payload_dir: Any) -> str:
    """``allow_installation`` of the payload folder ``payload_dir`` of ``name``: that canonical folder's identity and its
    install record's landing. The record is found by name alone, so a native or ClawHub folder counts only with a record
    of the source that lands there (``_LANDING_BUCKETS``); without that link, or for a link or any other folder, the
    token is ``""`` and the form offers no allow (Save, Replace and the Skills page's grants are unchanged)."""
    import hashlib
    import json

    from ouroboros.skill_install_record import landing_of, read_install_record

    folder = installation_of(drive_root, name, payload_dir) or _folder_identity(
        drive_root, name, payload_dir, tuple(_LANDING_BUCKETS.values()))
    record = read_install_record(pathlib.Path(drive_root), name)
    if folder is None or (folder["bucket"] not in RECEIPT_BUCKETS and _LANDING_BUCKETS.get(
            ((record or {}).get("source") or {}).get("kind")) != folder["bucket"]):
        return ""
    landing = list(landing_of(record)) if record else None
    return hashlib.sha256(json.dumps([folder, landing], sort_keys=True).encode()).hexdigest()[:16]


def rebind_restored_form_grant(drive_root: pathlib.Path, name: str, payload_dir: Any, previous: str) -> None:
    """A rollback restored the earlier payload as a copy (a new folder identity) with its earlier decision: the keys
    only the secret form granted to the installation it replaced (``previous``) follow it, nothing else, and nothing
    is granted anew. Raises while such keys are not rebound (an unreadable decision, an unproven restored copy): the
    rollback keeps its snapshot and that identity, so its ordinary retry only finishes this restoration."""
    restored = installation_token(drive_root, name, payload_dir)
    if not previous or restored == previous:
        return

    def rebind(data):
        if not _form_keys(data, previous):
            return None
        if not restored:
            raise ValueError("the restored payload is not a proven installation")
        return {**data, "form": {**data["form"], "installation": restored}}

    update_json_locked(_grant_path(drive_root, name), rebind, strict_existing_dict=True)


def _consent(entry: Any) -> Dict[str, Any]:
    """One stored consent exactly as the host writes it, else ``{}``: a malformed one proves nothing."""
    from ouroboros.skill_identity import is_release_digest

    entry = entry if isinstance(entry, dict) else {}
    keys, decisions = entry.get("keys"), entry.get("decisions")
    if (not is_release_digest(entry.get("digest")) or not isinstance(entry.get("installation"), str)
            or not entry["installation"] or not isinstance(entry.get("enable"), bool)
            or not isinstance(entry.get("recorded_at"), str) or not entry["recorded_at"]
            or not isinstance(keys, list) or not keys or not all(isinstance(key, str) and key for key in keys)
            or not isinstance(decisions, list) or not decisions
            or not all(isinstance(row, dict) and all(isinstance(row.get(field), str) and row[field]
                       for field in ("decision_id", "request_id")) for row in decisions)):
        return {}
    return entry


def consent_of(drive_root: pathlib.Path, name: str) -> Dict[str, Any]:
    """The owner's allow for exact files (including a claimed one until its result is saved)."""
    data = read_json_dict(skill_grant_receipts_path(drive_root)) or {}
    consents = data.get("consents") if data.get("schema_version") == RECEIPTS_SCHEMA_VERSION else None
    return _consent(consents.get(canonical_skill_name(name))) if isinstance(consents, dict) else {}


def orphan_facts(drive_root: pathlib.Path, decision: Dict[str, str], allow: Dict[str, Any], *,
                 began: bool = False) -> Dict[str, Any]:
    """What a pending press (``allow``) that no live step holds became, from the records alone (a read: it grants
    nothing). Never claimed, it did not start, unless its job may have begun (``began``: unknown). Claimed, it waits
    while the consent for exactly its files and installation still waits unclaimed with this press (its step ended
    there; only the card's write was lost), else its outcome is unknown."""
    from ouroboros.secret_requests import ALLOW_NOT_STARTED, ALLOW_ORPHANED, ALLOW_REQUESTED, allow_result

    if allow.get("state") == ALLOW_REQUESTED:
        return ALLOW_ORPHANED if began else ALLOW_NOT_STARTED
    waiting = consent_of(drive_root, str(allow.get("skill") or ""))
    if (not waiting or waiting.get("claimed_at") or waiting["digest"] != allow.get("content_hash")
            or waiting["installation"] != allow.get("installation") or dict(decision) not in waiting["decisions"]):
        return ALLOW_ORPHANED
    from ouroboros.skill_lifecycle_actions import owner_switch

    return allow_result({"ok": True, "deferred": True}, enable=bool(allow.get("enable")), attempted=False,
                        switch=owner_switch(drive_root, str(allow["skill"])))


def _ended_facts(consent: Dict[str, Any], reason: str) -> Dict[str, Any]:
    from ouroboros.secret_requests import ALLOW_ORPHANED

    return ALLOW_ORPHANED if consent.get("claimed_at") else {"state": "refused", "reason": reason}


def _readers_now(drive_root: pathlib.Path, consent: Dict[str, Any]) -> Dict[str, Any]:
    """Each decision's reader snapshot at this moment (``secret_requests.decision_consumers``), read outside the
    document lock and kept with the result it is decided with: a later publication never reads readers again."""
    if not consent:
        return {}
    from ouroboros.secret_requests import decision_consumers

    memo: Dict[Any, Any] = {}
    return {row["decision_id"]: decision_consumers(drive_root, row, memo) for row in consent["decisions"]}


def _settlement_rows(name: str, consent: Dict[str, Any], facts: Dict[str, Any],
                     readers: Dict[str, Any]) -> Dict[str, Any]:
    """One result per decision with its ``readers`` snapshot: a list, ``None`` (unknown, ``consumers_unknown``);
    a decision ``readers`` lacks (its consent changed meanwhile) is read when it is published."""
    stamp = utc_now_iso()
    binding = {"skill": name, "content_hash": consent["digest"], "installation": consent["installation"]}

    def snapshot(identity):
        found = readers.get(identity)
        return {"consumers": found} if isinstance(found, list) else {"consumers_unknown": True} if identity in readers else {}

    return {row["decision_id"]: {"request_id": row["request_id"], "binding": binding, "at": stamp, "facts": facts,
                                 **snapshot(row["decision_id"])} for row in consent["decisions"]}


def _remember_settlements(settlements: Dict[str, Any], name: str, rows: Dict[str, Any]) -> bool:
    prior = settlements.get(name)
    entries = dict(prior) if isinstance(prior, dict) else {}
    for decision_id, row in rows.items():
        existing = entries.get(decision_id)
        # A witnessed completion can resolve an unknown from a claimed press. Neither can replace known facts.
        if (existing is None or (existing.get("request_id") == row["request_id"]
                                 and existing.get("binding") == row["binding"]
                                 and existing.get("facts", {}).get("state") == "unknown"
                                 and row["facts"].get("state") != "unknown")):
            entries[decision_id] = row
    if entries == prior:
        return False
    settlements[name] = entries
    return True


def read_consent_outcomes(drive_root: pathlib.Path) -> Dict[str, Any]:
    """Read the metadata owed to cards, never a grant or a waiting consent."""
    data = read_json_dict(skill_grant_receipts_path(drive_root)) or {}
    rows = data.get("settlements") if data.get("schema_version") == RECEIPTS_SCHEMA_VERSION else None
    return rows if isinstance(rows, dict) else {}


def publish_consent_outcomes(drive_root: pathlib.Path, name: str, *, decision_id: str = "",
                             unstored: Dict[str, Any] | None = None) -> None:
    """Try the card projection; remove only the identical result that it no longer needs."""
    from ouroboros.secret_requests import settle_decisions

    key = canonical_skill_name(name)
    entries = read_consent_outcomes(drive_root).get(key) or {} if unstored is None else unstored
    if not isinstance(entries, dict):
        return
    done = {}
    for identity, entry in entries.items():
        if (decision_id and identity != decision_id) or not isinstance(entry, dict):
            continue
        if not isinstance(entry.get("facts"), dict) or not isinstance(entry.get("binding"), dict):
            continue
        if not entry.get("request_id") or not entry.get("at"):
            continue
        decision = {"decision_id": identity, "request_id": entry["request_id"], "binding": entry["binding"]}
        readers = ({"consumers": entry["consumers"]} if isinstance(entry.get("consumers"), list)
                   else {"consumers": None} if entry.get("consumers_unknown") else {})  # the decision's own snapshot
        if not settle_decisions(drive_root, [decision], entry["facts"], at=entry["at"], **readers):
            done[identity] = entry
    if not done or unstored is not None:
        return

    def acknowledge(_skills, _consents, settlements):
        current = settlements.get(key)
        if not isinstance(current, dict):
            return False
        remaining = {identity: entry for identity, entry in current.items() if done.get(identity) != entry}
        if remaining == current:
            return False
        if remaining:
            settlements[key] = remaining
        else:
            del settlements[key]
        return True

    _change_document(drive_root, key, acknowledge)


def record_consent(drive_root: pathlib.Path, skill: Any, keys: List[str], *, installation: str, enable: bool,
                   decision: Dict[str, str]) -> str:
    """Join an explicit allow for identical waiting files, or end the older right and record this one."""
    key, keys = canonical_skill_name(skill.name), _text_items(list(keys), upper=True)
    if not keys or not skill.content_hash or not installation:
        return "unrecorded"
    waiting = consent_of(drive_root, key)
    joins = (bool(waiting) and not waiting.get("claimed_at") and waiting["digest"] == skill.content_hash
             and waiting["installation"] == installation)
    readers = {} if joins else _readers_now(drive_root, waiting)  # for the consent this one replaces

    def change(_skills, consents, settlements):
        current = _consent(consents.get(key))
        same = (bool(current) and not current.get("claimed_at") and current["digest"] == skill.content_hash
                and current["installation"] == installation)
        earlier = [row for row in (current["decisions"] if same else []) if row["decision_id"] != decision["decision_id"]]
        if current and not same:
            reason = "skill_changed" if current["digest"] != skill.content_hash else "installation_changed"
            _remember_settlements(settlements, key, _settlement_rows(key, current, _ended_facts(current, reason),
                                                                     readers if current == waiting else {}))
        consents[key] = {"installation": installation, "digest": skill.content_hash,
                         "keys": sorted({*keys, *(current["keys"] if same else ())}),
                         "enable": bool(enable or (same and current["enable"])),
                         "decisions": [*earlier, dict(decision)], "recorded_at": utc_now_iso()}
        return True

    if not _change_document(drive_root, key, change):
        return "unrecorded"
    publish_consent_outcomes(drive_root, key)
    return "recorded"


def claim_consent(drive_root: pathlib.Path, skill: Any, installation: str) -> Tuple[str, Dict[str, Any]]:
    """Claim exact admitted files under the lane. Never reclaim a started step; other admitted files end the wait."""
    key, out = canonical_skill_name(skill.name), {"status": "", "consent": {}}
    waiting = consent_of(drive_root, key)
    readers = _readers_now(drive_root, waiting) if waiting and not waiting.get("claimed_at") and (
        waiting["digest"] != skill.content_hash or waiting["installation"] != installation) else {}

    def change(_skills, consents, settlements):
        current = _consent(consents.get(key))
        if not current:
            return False
        if current.get("claimed_at"):
            out.update(status="interrupted", consent=current)
            return False
        if current["digest"] != skill.content_hash or not installation or current["installation"] != installation:
            reason = "skill_changed" if current["digest"] != skill.content_hash else "installation_changed"
            _remember_settlements(settlements, key, _settlement_rows(key, current, _ended_facts(current, reason),
                                                                     readers if current == waiting else {}))
            del consents[key]
            out.update(status="obsolete", consent=current)
        else:
            claimed = {**current, "claimed_at": utc_now_iso()}
            consents[key] = claimed
            out.update(status="claimed", consent=claimed)
        return True

    changed = _change_document(drive_root, key, change)
    if changed and out["status"] == "obsolete":
        publish_consent_outcomes(drive_root, key)
    return (out["status"], out["consent"]) if changed or out["status"] == "interrupted" else ("", {})


def finish_consent(drive_root: pathlib.Path, name: str, consent: Dict[str, Any], facts: Dict[str, Any]) -> None:
    """Save the claimed step's outcome, with the readers as they are now, before projecting it; a disjoint/new
    consent stays untouched."""
    key = canonical_skill_name(name)
    rows = _settlement_rows(key, consent, facts, _readers_now(drive_root, consent))

    def finish(_skills, consents, settlements):
        same = _consent(consents.get(key)) == consent
        if same:
            del consents[key]
        recorded = _remember_settlements(settlements, key, rows)
        return same or recorded

    if _change_document(drive_root, key, finish):
        publish_consent_outcomes(drive_root, key)
    else:
        # An unavailable receipt writer need not prevent the other writer from recording a known fact.
        # If neither write succeeds, the claim still prevents another grant.
        publish_consent_outcomes(drive_root, key, unstored=rows)


def consents_to_recover(drive_root: pathlib.Path) -> List[str]:
    """Skills whose allow a restart may have left unsettled: any stored consent (claimed or waiting on already-admitted
    files) or a known result not yet on its card."""
    data = read_json_dict(skill_grant_receipts_path(drive_root)) or {}
    if data.get("schema_version") != RECEIPTS_SCHEMA_VERSION:
        return []
    consents = data.get("consents") if isinstance(data.get("consents"), dict) else {}
    settlements = data.get("settlements") if isinstance(data.get("settlements"), dict) else {}
    return sorted({name for name, value in consents.items() if _consent(value)} | set(settlements))


def recover_consent(drive_root: pathlib.Path, name: str) -> None:
    """Inside the lane after a restart: publish any owed result; record a claimed step no process holds unknown and
    never run it again; finish an unclaimed consent only when its exact files are already admitted and no update is
    still pending."""
    from ouroboros.secret_requests import ALLOW_ORPHANED
    from ouroboros.skill_lifecycle_actions import grant_consent_on_admission, update_pending

    publish_consent_outcomes(drive_root, name)
    waiting = consent_of(drive_root, name)
    if not waiting:
        return
    if waiting.get("claimed_at"):
        finish_consent(drive_root, name, waiting, ALLOW_ORPHANED)
        return
    if update_pending(drive_root, name):
        return
    grant_consent_on_admission(drive_root, name)


def _continuing_keys(receipt, grants, requested):
    if grants.get("content_hash") != receipt["digest"]:
        return []
    standing = set(receipt["keys"]) & set(_text_items(grants.get("granted_keys"), upper=True))
    return [key for key in requested if key in standing]


def finish_local_edit_grants(drive_root: pathlib.Path, skill: Any) -> None:
    """Rebind only still-granted, still-requested keys after an ordinary review of the verified tip."""
    gate = skill.review.gate_for(skill.content_hash)
    if (skill.load_error or gate["stale"] or not files_accepted(skill.review, skill.content_hash)
            or gate["status"] not in {"clean", "warnings"}
            or skill.review.review_profile == "owner_attested"
            or not (skill.manifest.is_script() or skill.manifest.is_extension())):
        return
    installation = installation_of(drive_root, skill.name, skill.skill_dir)
    receipt = load_receipt(drive_root, skill.name)
    if not installation or receipt.get("installation") != installation or receipt.get("tip") != skill.content_hash:
        return
    requested = requested_core_setting_keys(list(skill.manifest.env_from_settings or []))
    permissions = set(requested_skill_permissions(skill.manifest.permissions, skill.manifest.subscribe_events))
    holdable = permissions & _USE_CHECKED_PERMISSIONS if skill.manifest.is_extension() else set()
    form_installation = allow_installation(drive_root, skill)  # a Repair keeps the installation the form granted

    def rebind_receipt(current):
        if current != receipt:
            return None

        keys: List[str] = []
        scoped: List[str] = []
        held: List[str] = []

        def rebind_grants(grants):
            same_ordinary = grants.get("content_hash") == current["digest"]
            keys.extend(_continuing_keys(current, grants, requested))
            form_standing = set(current["keys"]) & set(_form_keys(grants, form_installation, current["digest"]))
            scoped.extend(key for key in requested if key in form_standing)
            if not (keys or scoped):
                return None
            # What the earlier files had (their grant, or a hold carried to them) is held for these files, not granted.
            held.extend(sorted(holdable & {
                *(_text_items(grants.get("granted_permissions")) if same_ordinary else ()),
                *_text_items(current.get("held_permissions"))}))
            new_form = ({"installation": form_installation, "digest": skill.content_hash, "keys": scoped}
                        if scoped else None)
            if not keys:
                return _preserve_ordinary_with_form(grants, new_form)
            covered = set(_text_items(grants.get("requested_keys"), upper=True)) | set(keys)
            return _grant_decision(grants, skill.content_hash, keys, set(requested) & covered, [], permissions,
                                   form=new_form or {})

        # Receipt identity under its lock, and live grant/digest under its own lock.
        # Other writers finish their grant write before taking the receipt lock.
        update_json_locked(_grant_path(drive_root, skill.name), rebind_grants, strict_existing_dict=True)
        rest = {k: v for k, v in current.items() if k not in {"tip", "held_permissions"}}
        rest.update({"held_permissions": held} if held else {})
        carried = sorted({*keys, *scoped})
        return {**rest, "digest": skill.content_hash, "keys": carried} if carried else {}

    _change_receipt(drive_root, skill.name, rebind_receipt)


def held_use_permissions(drive_root: pathlib.Path, skill: Any, grants: Dict[str, Any],
                         requested_permissions: List[str]) -> List[str]:
    """Requested use-checked permissions an extension's files lack that the receipt holds for exactly them in this
    installation. Never a grant. Read only for such a gap; a decision for other files (or none) holds nothing, and a
    script keeps its run gate as it was."""
    if not skill.manifest.is_extension() or not skill.content_hash:
        return []
    same_bytes = grants.get("content_hash") == skill.content_hash
    if not same_bytes and not grants.get("form_granted_keys"):
        return []
    request_same = sorted(_text_items(grants.get("requested_permissions"))) == sorted(requested_permissions)
    granted = set(_text_items(grants.get("granted_permissions"))) if same_bytes and request_same else set()
    lacking = [perm for perm in requested_permissions if perm in _USE_CHECKED_PERMISSIONS and perm not in granted]
    if not lacking:
        return []
    receipt = load_receipt(drive_root, skill.name)
    if (receipt.get("digest") != skill.content_hash
            or installation_of(drive_root, skill.name, skill.skill_dir) != receipt["installation"]):
        return []
    return [perm for perm in lacking if perm in _text_items(receipt.get("held_permissions"))]


def files_accepted(review: Any, content_hash: str) -> bool:
    """Whether these exact files are admitted: a recorded acceptance or a current clean/warnings verdict of their gate
    (``skill_review_gate``), never Cyber Pro authority alone. If it is all the gate names, read the same gate with
    ``ordinary_only=True`` so the author's ordinary acceptance of these files counts on the same terms;
    a source that reviews each skill still needs its review or the owner's Skip."""
    from ouroboros.skill_review_status import ACCEPTANCE_REASONS

    gate = review.gate_for(content_hash)
    if gate.get("blocking_reason") == "cyber_authority":
        gate = review.gate_for(content_hash, ordinary_only=True)
    return bool(gate.get("executable_review")) and not gate.get("preflight_failed") and (
        gate.get("blocking_reason") in ACCEPTANCE_REASONS or (
            not gate.get("stale") and gate.get("status") in {"clean", "warnings"}))


def grant_view(content_hash: str, grants: Dict[str, Any], requested: List[str],
               requested_permissions: List[str], accepted: bool,
               held_permissions: List[str] | Tuple[str, ...] = ()) -> Dict[str, Any]:
    """Exact grants, missing items, and pending custom keys and held permissions for accepted current files
    (``accepted``: ``files_accepted``)."""
    decided_for = grants.get("content_hash")
    same_bytes = bool(content_hash) and content_hash == decided_for
    keys, permissions = set(grants.get("granted_keys") or []), set(grants.get("granted_permissions") or [])
    form_keys = set(grants.get("form_granted_keys") or []) if content_hash else set()
    request_same = sorted(grants.get("requested_permissions") or []) == sorted(requested_permissions)
    granted = [key for key in requested if (key in keys and same_bytes) or key in form_keys]
    granted_permissions = [perm for perm in requested_permissions if perm in permissions and same_bytes and request_same]
    missing = [key for key in requested if key not in granted]
    missing_permissions = [perm for perm in requested_permissions if perm not in granted_permissions]
    earlier = {"keys": [key for key in missing if key in keys],
               "permissions": [perm for perm in missing_permissions if perm in permissions]}
    always_requested = {key.upper() for key in FORBIDDEN_SKILL_SETTINGS} | REQUESTABLE_WHEN_ABSENT
    covered = set(grants.get("requested_keys") or []) | keys | form_keys
    accepted = bool(content_hash) and accepted
    pending = [key for key in missing if key not in always_requested and key not in covered] if accepted else []
    held = [perm for perm in missing_permissions if perm in held_permissions] if accepted else []
    return {"granted_keys": granted, "missing_keys": missing, "granted_permissions": granted_permissions,
            "missing_permissions": missing_permissions, "pending_keys": pending, "pending_permissions": held,
            "earlier_grant": earlier if decided_for and content_hash and not same_bytes and any(earlier.values()) else None,
            "blocks": bool(set(missing) - set(pending) or set(missing_permissions) - set(held))}


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
    """All missing keys and permissions except pending ones."""
    pending, held = set(status.get("pending_keys") or []), set(status.get("pending_permissions") or [])
    return ([key for key in status.get("missing_keys") or [] if key not in pending],
            [perm for perm in status.get("missing_permissions") or [] if perm not in held])


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
    """Readiness notes about pending keys and held permissions, not a claim that the skill runs or is enabled."""
    return [*(f"key {key} is saved in Settings but not granted for these files; it is not passed and does not"
              " by itself stop the skill" for key in status.get("pending_keys") or []),
            *(f"permission {perm} is held from earlier files of this installation but not granted for these; the host"
              " refuses each use until it is, and it does not by itself stop the skill"
              for perm in status.get("pending_permissions") or [])]


__all__ = [
    "RECEIPTS_FILENAME", "REQUESTABLE_WHEN_ABSENT", "allow_installation", "blocking_items", "blocking_lists",
    "claim_consent", "consent_of", "consents_to_recover", "declared_catalog_settings", "end_receipt",
    "end_superseded_consent", "extend_verified_edit", "files_accepted", "finish_consent", "finish_local_edit_grants",
    "grant_notes", "grant_view", "held_use_permissions", "installation_of", "installation_token", "load_receipt",
    "orphan_facts", "pending_fact", "publish_consent_outcomes", "read_consent_outcomes", "rebind_restored_form_grant",
    "receipt_tip", "record_consent", "record_explicit_grant", "recover_consent", "registration_view",
]
