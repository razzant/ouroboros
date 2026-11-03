"""Install record: where an installed skill came from and why its bytes may run.

One host-owned file per installed name in the state plane
(``data/state/skills/<name>/install.json``), never inside the payload: it is not part of
the release digest and a payload edit cannot rewrite it. The record keeps the source row
as it was at install time (a later change of that row's trust never rewrites an existing
install), the exact release, the base digest that "Modified" compares against, and the
basis the acceptance gate reads.

Installs made before sources existed carry a payload sidecar instead. ``convert_legacy_records``
reads those sidecars once (startup) and writes the equivalent record: an OuroborosHub sidecar names
every file of the original release with its SHA-256 and the declaration the installer used, so its
release digest (``skill_identity.release_digest_from_files``) and base are exact; a ClawHub one does
not, so its record says the original version is unknown (``legacy: true`` with an empty base).
Legacy installs keep the basis they ran under (``review``, marked ``converted_from``, which keeps
their earlier Advisory behaviour in the gate): an old "official" label never becomes a trusted
basis by conversion. The sidecars themselves stay where they are.
"""

from __future__ import annotations

import base64
import json
import logging
import pathlib
import re
import shutil
from typing import Any, Dict, List, Optional

from ouroboros.skill_loader import skill_state_dir, skill_state_dir_path
from ouroboros.utils import atomic_write_json, read_json_dict, utc_now_iso

log = logging.getLogger(__name__)

INSTALL_RECORD_FILENAME = "install.json"
INSTALL_RECORD_SCHEMA_VERSION = 1
# trusted: installed from a source row the owner trusts; review: a review decides;
# self_check: Ouroboros read and recorded its check of these bytes;
# owner_skip: the owner skipped review for these bytes; owner_edit: the owner (or
# Ouroboros on the owner's instruction) changed them; cyber: Cyber Pro judgment.
INSTALL_BASES = frozenset({"trusted", "review", "self_check", "owner_skip", "owner_edit", "cyber"})
LEGACY_HUB_SOURCE_ID = "ouroboroshub-legacy"


def install_basis_fields(trust: str, *, check: str = "review", check_notes: str = "", task_id: str = "",
                         skip_review: bool = False, cyber: Optional[bool] = None) -> Dict[str, Any]:
    """Bind the source trust, the owner's Skip, the review, or (Cyber Pro only) Ouroboros's recorded check."""
    if skip_review:
        return {"basis": "owner_skip"}
    if trust == "trusted":
        return {"basis": "trusted"}  # a trusted source needs no additional check
    if check == "review":
        return {"basis": "review"}
    if check != "self":
        raise ValueError("check is review or self.")
    from ouroboros.skill_review_status import cyber_authority

    if not (cyber_authority() if cyber is None else cyber):
        raise ValueError("check self is available only in Cyber Pro; outside it a source that reviews each "
                         "skill needs its review (check review), and only the owner skips one.")
    notes = str(check_notes or "").strip()
    if not 1 <= len(notes) <= 2000:
        raise ValueError("check self needs check_notes of 1..2000 characters.")
    return {"basis": "self_check", "check": {"by": "ouroboros", "notes": notes,
                                             "task_id": task_id, "at": utc_now_iso()}}


def install_listing_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    """Public check/update facts; task context and durable rollback paths stay private."""
    fields = {}
    for field, keys in (("check", ("by", "notes", "at")), ("pending_update", ("version", "digest")),
                        ("update_rejected", ("version", "digest", "reason", "at"))):
        if isinstance(record.get(field), dict):
            fields[field] = {key: record[field][key] for key in keys if key in record[field]}
    if not record.get("base_digest") and record.get("modified") is True:
        fields["modified"] = True
    return fields


def install_record_path(drive_root: pathlib.Path, name: str) -> pathlib.Path:
    return skill_state_dir_path(drive_root, name) / INSTALL_RECORD_FILENAME


def read_install_record(drive_root: pathlib.Path, name: str) -> Optional[Dict[str, Any]]:
    """The record, or ``None`` when the skill was not installed from a source."""
    data = read_json_dict(install_record_path(drive_root, name))
    if not data or data.get("schema_version") != INSTALL_RECORD_SCHEMA_VERSION:
        return None
    if str(data.get("basis") or "") not in INSTALL_BASES:
        return None
    return data


def write_install_record(drive_root: pathlib.Path, name: str, record: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and atomically persist ``record``; returns what was written.

    ``release.digest`` is required, except on a ``legacy`` record whose original release is unknown.
    """
    basis = str(record.get("basis") or "")
    if basis not in INSTALL_BASES:
        raise ValueError(f"unknown install basis {basis!r}")
    release = record.get("release")
    if not isinstance(release, dict) or not (str(release.get("digest") or "") or record.get("legacy")):
        raise ValueError("install record needs release.digest")
    source = record.get("source")
    if not isinstance(source, dict) or not str(source.get("id") or ""):
        raise ValueError("install record needs source.id")
    payload = dict(record)
    payload["schema_version"] = INSTALL_RECORD_SCHEMA_VERSION
    payload.setdefault("entry", name)
    payload.setdefault("base_digest", str(release.get("digest") or ""))
    payload.setdefault("installed_at", utc_now_iso())
    payload["updated_at"] = utc_now_iso()
    atomic_write_json(skill_state_dir(drive_root, name) / INSTALL_RECORD_FILENAME, payload)
    return payload


def update_install_record(drive_root: pathlib.Path, name: str, **changes: Any) -> Optional[Dict[str, Any]]:
    """Merge ``changes`` into an existing record (``None`` removes a key); ``None`` when there is none."""
    current = read_install_record(drive_root, name)
    if current is None:
        return None
    for key, value in changes.items():
        if value is None:
            current.pop(key, None)
        else:
            current[key] = value
    current.pop("updated_at", None)
    return write_install_record(drive_root, name, current)


def clear_install_record(drive_root: pathlib.Path, name: str) -> None:
    try:
        install_record_path(drive_root, name).unlink()
    except FileNotFoundError:
        pass


def update_backup_path(drive_root: pathlib.Path, name: str) -> pathlib.Path:
    return skill_state_dir_path(drive_root, name) / "update_backup.json"


def retain_update_snapshot(snapshot: Any, record: Dict[str, Any]) -> None:
    """Persist the existing rollback handle BEFORE a review-basis update replaces any bytes.

    The payload copy already lives beside its bucket; only the lifecycle bytes and handle
    need durability here. This is the same snapshot/restore contract as a failed landing.
    """
    if record.get("basis") != "review":
        return
    previous = json.loads(snapshot.lifecycle_state.get(INSTALL_RECORD_FILENAME) or b"{}")
    old, new = previous.get("release") or {}, record["release"]
    pending = {"from_version": old.get("version", ""), "from_digest": old.get("digest", ""),
               "version": new.get("version", ""), "digest": new["digest"], "requested_at": utc_now_iso()}
    data = {"pending_update": pending, "target_dir": str(snapshot.target_dir),
            "backup_dir": str(snapshot.backup_dir) if snapshot.backup_dir else "",
            "state_provenance": snapshot.state_provenance,
            "lifecycle_state": {key: base64.b64encode(value).decode("ascii") if value is not None else None
                                for key, value in snapshot.lifecycle_state.items()},
            "payload_restored": False, "restore_started": False}
    atomic_write_json(update_backup_path(snapshot.drive_root, snapshot.skill_name), data)
    record["pending_update"] = pending


def read_update_snapshot(drive_root: pathlib.Path, name: str) -> tuple[Any, Dict[str, Any]]:
    from ouroboros.marketplace.install import PayloadRollbackSnapshot

    data = read_json_dict(update_backup_path(drive_root, name)) or {}
    if not data:
        return None, {}
    try:
        lifecycle = {key: base64.b64decode(value) if value is not None else None
                     for key, value in data["lifecycle_state"].items()}
        previous = json.loads(lifecycle.get(INSTALL_RECORD_FILENAME) or b"{}")
        kind = (previous.get("source") or {}).get("kind")
        if not kind and (data.get("state_provenance") or {}).get("source") == "clawhub":
            kind = "registry"  # sidecar-era registry install, before install.json
        bucket = {"builtin": "native", "hub": "ouroboroshub", "git": "ouroboroshub", "registry": "clawhub"}[kind]
        target = pathlib.Path(drive_root) / "skills" / bucket / name
        backup = None
        if data.get("backup_dir"):
            leaf = pathlib.Path(data["backup_dir"]).name
            if not re.fullmatch(re.escape(name) + r"\.[0-9a-f]{32}", leaf):
                raise ValueError("invalid backup name")
            backup = target.parent / ".rollback" / f"{name}.{leaf.rsplit('.', 1)[1]}"
        if pathlib.Path(data["target_dir"]).resolve() != target.resolve() or (
                backup is not None and pathlib.Path(data["backup_dir"]).resolve() != backup.resolve()):
            raise ValueError("mismatched stored path")
    except (KeyError, TypeError, ValueError, OSError):
        log.warning("Update snapshot paths do not match the bucket and name for %s; kept evidence at %s",
                    name, update_backup_path(drive_root, name), exc_info=True)
        return None, data
    snapshot = PayloadRollbackSnapshot(
        drive_root=pathlib.Path(drive_root), skill_name=name, target_dir=target, backup_dir=backup,
        state_provenance=data.get("state_provenance"),
        lifecycle_state=lifecycle,
        payload_restored=bool(data.get("payload_restored")),
    )
    return snapshot, data


def discard_update_snapshot(drive_root: pathlib.Path, name: str) -> None:
    """Uninstall cancels the retained update, so startup cannot resurrect removed bytes."""
    from ouroboros.marketplace.install import discard_payload_snapshot

    snapshot, _ = read_update_snapshot(drive_root, name)
    if snapshot is not None:
        discard_payload_snapshot(snapshot)
        update_backup_path(drive_root, name).unlink()


def restore_update_snapshot(drive_root: pathlib.Path, name: str, reason: str) -> bool:
    """Restore once, retry state-only failures, and retain evidence on an ambiguous partial restore."""
    from ouroboros.marketplace.install import restore_payload_state

    snapshot, data = read_update_snapshot(drive_root, name)
    if snapshot is None:
        return False
    if data.get("restore_started") and not snapshot.payload_restored:
        log.error("Update restore for %s was interrupted; kept evidence at %s; payload will not be replaced again",
                  name, update_backup_path(drive_root, name))
        return False
    data.update(restore_started=True, rejection_reason=data.get("rejection_reason") or reason)
    path = update_backup_path(drive_root, name)
    atomic_write_json(path, data)
    try:
        restore_payload_state(snapshot, restore_state_provenance=True)
        pending = data["pending_update"]
        update_install_record(drive_root, name, pending_update=None, update_rejected={
            "version": pending["version"], "digest": pending["digest"],
            "reason": data["rejection_reason"], "at": utc_now_iso(),
        })
    except Exception:
        data["payload_restored"] = snapshot.payload_restored
        atomic_write_json(path, data)
        log.exception("Update restore failed for %s; kept payload and state evidence at %s", name, path)
        return False
    path.unlink()
    return True


def keep_previous_payload(path: Optional[pathlib.Path], name: str) -> None:
    """Retain one previous payload per name beside its bucket (``.rollback/<name>.previous``)."""
    if path is None or not path.exists():
        return
    keep = path.parent / f"{name}.previous"
    try:
        if keep.exists():
            shutil.rmtree(keep)
        path.rename(keep)
    except OSError:
        log.warning("Could not retain the previous payload of %s at %s", name, path, exc_info=True)


def settle_pending_update(drive_root: pathlib.Path, name: str, *, expected_digest: str = "",
                          force: bool = False, landing_failed: bool = False) -> Dict[str, Any]:
    """Settle the kept update snapshot inside the existing lifecycle lane, regardless of review caller."""
    from ouroboros.config import get_skills_repo_path, load_settings
    from ouroboros.skill_loader import find_skill
    from ouroboros.skill_review_runner import _running_job_interrupt_reason, review_job_state_path
    from ouroboros.skill_review_status import preflight_failed
    from ouroboros.utils import read_json_dict

    snapshot, data = read_update_snapshot(drive_root, name)
    if snapshot is None:
        return {"status": "restore_failed" if data else "none"}
    pending = data["pending_update"]
    if expected_digest and expected_digest != pending["digest"]:
        return {"status": "superseded"}
    record = read_install_record(drive_root, name) or {}
    release = record.get("release") or {}
    previous = json.loads(snapshot.lifecycle_state.get(INSTALL_RECORD_FILENAME) or b"{}")
    # The durable snapshot precedes both the payload swap and the new record. If
    # the old record is still present, landing was interrupted, not superseded.
    interrupted_landing = bool(record) and record == previous
    if not data.get("restore_started") and not interrupted_landing and (
            record.get("pending_update") != pending or release.get("digest") != pending["digest"]):
        discard_update_snapshot(drive_root, name)
        return {"status": "superseded"}
    loaded = find_skill(drive_root, name, repo_path=get_skills_repo_path())
    current = not interrupted_landing and loaded is not None and release.get("digest") == pending["digest"]
    if not landing_failed and not data.get("restore_started") and current and not loaded.load_error \
            and loaded.review.gate_for(loaded.content_hash)["executable_review"]:
        update_install_record(drive_root, name, pending_update=None, update_rejected=None)
        keep_previous_payload(snapshot.backup_dir, name)
        update_backup_path(drive_root, name).unlink()
        return {"status": "accepted"}
    job = read_json_dict(review_job_state_path(drive_root, name)) or {}
    if not force and not interrupted_landing and not data.get("restore_started") and job.get("status") == "running" \
            and _running_job_interrupt_reason(job) is None:
        return {"status": "review_running"}
    reason = "landing_failed" if landing_failed else "review_unfinished"
    if not landing_failed and current and not loaded.review.is_stale_for(loaded.content_hash):
        if preflight_failed(loaded.review.findings):
            reason = "preflight_failed"
        elif loaded.review.status == "blockers":
            reason = "review_blockers"
    try:
        from ouroboros.extension_loader import unload_extension

        unload_extension(name)
    except Exception:
        log.debug("unload before restoring %s failed", name, exc_info=True)
    restored = restore_update_snapshot(drive_root, name, reason)
    if restored:
        try:
            from ouroboros.extension_loader import reconcile_extension

            reconcile_extension(name, drive_root, load_settings)
        except Exception:
            log.warning("Restored update of %s needs extension reconciliation", name, exc_info=True)
    return {"status": "restored" if restored else "restore_failed", "reason": reason}


def _legacy_hub_source(catalog_url: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    from ouroboros.skill_sources import repo_from_catalog_url

    repo, _ref = repo_from_catalog_url(catalog_url)
    matches = [row for row in rows if row["kind"] == "hub" and row.get("url") == repo]
    row = next((r for r in matches if r.get("part") != "official"), matches[0] if matches else None)
    if row is None:
        return {"id": LEGACY_HUB_SOURCE_ID, "kind": "hub", "name": "OuroborosHub (earlier address)", "url": repo}
    return {key: row[key] for key in ("id", "kind", "name", "url", "part") if row.get(key)}


def legacy_hub_record(sidecar: Dict[str, Any], rows: List[Dict[str, Any]],
                      payload: pathlib.Path) -> Optional[Dict[str, Any]]:
    """The install record an OuroborosHub payload sidecar stands for, or ``None`` when it is unusable."""
    from ouroboros.skill_identity import legacy_release_declaration, release_digest_from_files

    files = sidecar.get("files")
    if sidecar.get("source") != "ouroboroshub" or not isinstance(files, list) or not files:
        return None
    try:
        digest = release_digest_from_files([item for item in files if isinstance(item, dict)],
                                           legacy_release_declaration(payload, sidecar))
    except (ValueError, TypeError):
        return None
    return {
        "source": _legacy_hub_source(str(sidecar.get("catalog_url") or ""), rows),
        "entry": str(sidecar.get("slug") or ""),
        "release": {"version": str(sidecar.get("version") or ""), "digest": digest},
        "base_digest": digest, "installed_at": str(sidecar.get("installed_at") or utc_now_iso()),
        "basis": "review", "official": False, "installed_by": "owner", "turn_on_when_ready": False,
        "converted_from": ".ouroboroshub.json",
    }


def _legacy_clawhub_record(drive_root: pathlib.Path, name: str, rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    from ouroboros.marketplace.provenance import read_provenance

    provenance = read_provenance(drive_root, name) or {}
    if provenance.get("source") != "clawhub" or not provenance.get("slug"):
        return None
    row = next((r for r in rows if r["kind"] == "registry"), None)
    source = {key: row[key] for key in ("id", "kind", "name", "url") if row and row.get(key)} or {
        "id": "clawhub", "kind": "registry", "name": "ClawHub"}
    return {
        "source": source, "entry": str(provenance["slug"]), "legacy": True,
        "release": {"version": str(provenance.get("version") or ""), "digest": ""}, "base_digest": "",
        "installed_at": str(provenance.get("installed_at") or utc_now_iso()), "basis": "review",
        "official": False, "installed_by": "owner", "turn_on_when_ready": False, "converted_from": ".clawhub.json",
    }


def convert_legacy_records(drive_root: pathlib.Path, rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Write install records for sidecar-era installs that have none. Idempotent; never raises."""
    report: Dict[str, Any] = {"converted": [], "skipped": []}
    try:
        if rows is None:
            from ouroboros.skill_sources import load_sources

            rows = load_sources()
        skills_root = pathlib.Path(drive_root) / "skills"
        for bucket, marker in (("ouroboroshub", ".ouroboroshub.json"), ("clawhub", ".clawhub.json")):
            root = skills_root / bucket
            for payload in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")) \
                    if root.is_dir() else []:
                name = payload.name
                if not (payload / marker).is_file() or install_record_path(drive_root, name).is_file():
                    continue
                if bucket == "ouroboroshub":
                    record = legacy_hub_record(read_json_dict(payload / marker) or {}, rows, payload)
                else:
                    record = _legacy_clawhub_record(pathlib.Path(drive_root), name, rows)
                if record is None:
                    report["skipped"].append(name)
                    continue
                write_install_record(pathlib.Path(drive_root), name, record)
                report["converted"].append(name)
    except Exception:
        log.warning("Legacy install record conversion failed", exc_info=True)
    return report


__all__ = [
    "INSTALL_BASES", "INSTALL_RECORD_FILENAME", "LEGACY_HUB_SOURCE_ID", "clear_install_record",
    "convert_legacy_records", "install_record_path", "legacy_hub_record", "read_install_record",
    "update_install_record", "write_install_record",
]
