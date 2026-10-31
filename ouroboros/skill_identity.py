"""Skill release identity: one digest of a payload and its dependency declaration.

A release digest names "these payload bytes with this dependency declaration" on every
machine. Per-install notes (launcher markers, legacy marketplace sidecars, local lineage
notes) are not part of a release, so the same release installed twice has one digest, and
a review, an endorsement or an owner decision about it is recognized wherever it lands.
It is the staleness hash of every review, grant and acceptance record:
``skill_loader.compute_content_hash`` returns it.

Version marker: the preimage starts with ``DIGEST_DOMAIN``, so the previous algorithm
(version 1: sidecars inside, the native launcher marker exempt) never yields the value of
the same bytes under this one. The text stays 64 lowercase hex because Presence ceilings,
publication receipts and request validators already require that shape. Version-1 hashes
survive only in ``legacy_content_hashes`` for the one-time conversion of persisted records
(``convert_skill_identities``): a record moves to the new identity only when it covered
the unchanged bytes; stale records stay stale.

The dependency declaration is part of the release because it decides what the installer
puts beside the payload: a legacy install sidecar's declaration when the payload carries
one (that is what the installer used), else the manifest's own ``install_specs`` /
``install`` / ``dependencies``. A source index computes the same value from the same
manifest, so a source digest and an installed digest agree byte for byte.

The file walk, confinement and sensitive-file policy are the loader's own
(``skill_loader._iter_payload_files``); this module only decides what a release is.
"""

from __future__ import annotations

import collections
import hashlib
import json
import logging
import os
import pathlib
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

log = logging.getLogger(__name__)

DIGEST_DOMAIN = b"ouroboros-skill-release/2\0"
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")

# Top-level per-install notes the host writes into payloads of ONE bucket. There they never
# belong to the release (legacy names stay listed so older installs keep the digest of the
# bytes they run); anywhere else a file of that name is skill content and is hashed, so it
# cannot carry unreviewed bytes past the review binding.
PER_INSTALL_NOTE_BUCKETS: Dict[str, str] = {
    ".seed-origin": "native",
    ".ouroboroshub.json": "ouroboroshub",
    ".clawhub.json": "clawhub",
    ".self_authored.json": "external",
}
PER_INSTALL_NOTE_FILENAMES = frozenset(PER_INSTALL_NOTE_BUCKETS)
# Legacy install sidecars whose recorded declaration the installer used (same order as
# ``skill_dependencies._payload_sidecar_specs``).
_DECLARATION_SIDECARS = (".ouroboroshub.json", ".clawhub.json")
_MANIFEST_DECLARATION_KEYS = ("install_specs", "install", "dependencies")
_MANIFEST_NAMES = ("SKILL.md", "skill.json")


def is_release_digest(value: Any) -> bool:
    return bool(_DIGEST_RE.fullmatch(str(value or "")))


def payload_bucket(skill_dir: pathlib.Path) -> str:
    """The data bucket a payload directory sits in (its parent's name; a frozen review view keeps it)."""
    return pathlib.Path(skill_dir).resolve().parent.name


def is_per_install_note(rel: str, bucket: str) -> bool:
    """A top-level note the host writes into payloads of ``bucket``."""
    rel = str(rel)
    return bool(bucket) and "/" not in rel and PER_INSTALL_NOTE_BUCKETS.get(rel) == bucket


def _empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def declaration_digest(declaration: Any) -> str:
    """SHA-256 of the canonical JSON of a raw declaration, or "" when there is none."""
    if _empty(declaration):
        return ""
    text = json.dumps(declaration, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def manifest_declaration(manifest: Any) -> Any:
    """The manifest's raw dependency declaration (first non-empty key), or None."""
    extras = getattr(manifest, "raw_extra", None)
    if not isinstance(extras, dict):
        return None
    for key in _MANIFEST_DECLARATION_KEYS:
        if not _empty(extras.get(key)):
            return extras[key]
    return None


def _parse_manifest(skill_dir: pathlib.Path) -> Any:
    from ouroboros.contracts.skill_manifest import parse_skill_manifest_text

    for name in _MANIFEST_NAMES:
        path = skill_dir / name
        if not path.is_file():
            continue
        try:
            return parse_skill_manifest_text(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def sidecar_declaration(sidecar: Dict[str, Any]) -> Any:
    """The raw declaration the legacy installer used, if it recorded one."""
    specs = sidecar.get("install_specs")
    if isinstance(specs, dict):
        for key in ("raw", "auto"):
            if not _empty(specs.get(key)):
                return specs[key]
    return None


def legacy_release_declaration(skill_dir: pathlib.Path, sidecar: Dict[str, Any]) -> Any:
    """Reconstruct the original declaration; a manifest fallback needs its recorded byte hash."""
    declaration = sidecar_declaration(sidecar)
    if declaration is not None:
        return declaration
    files = {item.get("path"): item.get("sha256") for item in sidecar.get("files", []) if isinstance(item, dict)}
    for name in _MANIFEST_NAMES:
        path = pathlib.Path(skill_dir) / name
        if path.is_file():
            if _file_sha256(path, name).hex() == files.get(name):
                return manifest_declaration(_parse_manifest(pathlib.Path(skill_dir)))
            break  # only the manifest the loader selects can supply a declaration
    return None


def release_declaration(skill_dir: pathlib.Path, manifest: Any = None, *, bucket: Optional[str] = None) -> Any:
    """Raw dependency declaration of the release in ``skill_dir``.

    ``manifest`` is the already parsed manifest of the same directory, when the caller
    has it; otherwise it is parsed here (SKILL.md first, as the loader selects it).
    """
    from ouroboros.utils import read_json_dict

    root = pathlib.Path(skill_dir)
    bucket = payload_bucket(root) if bucket is None else bucket
    for name in _DECLARATION_SIDECARS:
        sidecar = root / name
        if not is_per_install_note(name, bucket) or not sidecar.is_file():
            continue
        declaration = sidecar_declaration(read_json_dict(sidecar) or {})
        if declaration is not None:
            return declaration
    if manifest is None:
        manifest = _parse_manifest(root)
    return manifest_declaration(manifest) if manifest is not None else None


def release_digest_from_file_digests(file_digests: Iterable[Tuple[str, bytes]], declaration: Any = None, *,
                                     bucket: str = "") -> str:
    """The release digest over ``(relative POSIX path, raw SHA-256)`` pairs.

    A caller that captured every file of a payload names its ``bucket`` so the host's
    per-install notes of that bucket are skipped; an index file list names none. Paths are
    ordered as plain strings, which any index generator reproduces.
    """
    entries: Dict[str, bytes] = {}
    for rel, file_digest in file_digests:
        rel = str(rel)
        if is_per_install_note(rel, bucket):
            continue
        entries[rel] = bytes(file_digest)
    digest = hashlib.sha256(DIGEST_DOMAIN)
    for rel in sorted(entries):
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(entries[rel])
    declared = declaration_digest(declaration)
    if declared:
        digest.update(b"\0install_specs\0")
        digest.update(declared.encode("ascii"))
    return digest.hexdigest()


def release_digest_from_files(files: Iterable[Dict[str, Any]], declaration: Any = None) -> str:
    """The release digest of an index-shaped file list ``[{"path", "sha256"}]``."""
    pairs = []
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("file entries must be objects with path and sha256")
        pairs.append((str(item.get("path") or ""), bytes.fromhex(str(item.get("sha256") or ""))))
    return release_digest_from_file_digests(pairs, declaration)


# (device, inode) -> ((size, mtime_ns, ctime_ns), the file's SHA-256) of payload files hashed in this process.
_DIGESTS: "collections.OrderedDict[Tuple[int, int], Tuple[Tuple[int, int, int], bytes]]" = collections.OrderedDict()
_DIGESTS_KEPT = 8192  # memory: about half a kilobyte per kept digest; the oldest goes first
_LOCAL_DEVICES: Dict[int, bool] = {}  # st_dev -> the store's local-filesystem verdict, probed once
_RACY_NS = 2_000_000_000  # a change time this close to the hash may share a coarse clock tick with a later write


def _change_key(path: pathlib.Path) -> Optional[Tuple[Tuple[int, int], Tuple[int, int, int]]]:
    """The file's identity and change key on a filesystem the store calls local, else ``None``: Windows
    (its ``st_ctime`` is the creation time), network, FUSE and unknown filesystems hash every time."""
    if os.name == "nt":
        return None
    info = os.stat(path)
    if info.st_dev not in _LOCAL_DEVICES:
        from ouroboros import store

        _LOCAL_DEVICES[info.st_dev] = store._filesystem_type(path) in store._LOCAL_FILESYSTEMS
    if not _LOCAL_DEVICES[info.st_dev]:
        return None
    return (info.st_dev, info.st_ino), (info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _file_sha256(path: pathlib.Path, rel: str) -> bytes:
    """The file's SHA-256, the review-freshness authority (P3), reused while its change key stays.

    On a local POSIX filesystem every change of a file's bytes or times (write, truncate, ``utime``, rename)
    sets its change time to the kernel's clock, which no unprivileged call sets back, so an unchanged
    (device, inode, size, mtime, ctime) key names the bytes hashed under it. A digest is kept only when that
    change time lies ``_RACY_NS`` before the hash began, so a later write in the same coarse clock tick (one
    second on HFS+ or ext3) cannot leave the key unchanged."""
    from ouroboros.skill_loader import SkillPayloadUnreadable

    file_digest = hashlib.sha256()
    try:
        key, started = _change_key(path), time.time_ns()
        if key is not None and (kept := _DIGESTS.get(key[0])) is not None and kept[0] == key[1]:
            return kept[1]
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(64 * 1024), b""):
                file_digest.update(chunk)
    except OSError as exc:
        log.warning("Failed to read skill payload file %s", path, exc_info=True)
        raise SkillPayloadUnreadable(rel, exc) from exc
    if key is not None and key[1][2] < started - _RACY_NS:
        _DIGESTS[key[0]] = (key[1], file_digest.digest())
        if len(_DIGESTS) > _DIGESTS_KEPT:
            _DIGESTS.popitem(last=False)
    return file_digest.digest()


def _payload_file_digests(
    skill_dir: pathlib.Path,
    *,
    manifest_entry: str = "",
    manifest_scripts: Optional[List[Dict[str, Any]]] = None,
    include_notes: bool = False,
    bucket: Optional[str] = None,
) -> List[Tuple[str, bytes]]:
    from ouroboros.skill_loader import _iter_payload_files

    pairs = []
    if include_notes:
        bucket = ""
    elif bucket is None:
        bucket = payload_bucket(skill_dir)
    for file_path in _iter_payload_files(skill_dir, manifest_entry=manifest_entry, manifest_scripts=manifest_scripts):
        rel = file_path.relative_to(skill_dir).as_posix()
        if is_per_install_note(rel, bucket):
            continue
        pairs.append((rel, _file_sha256(file_path, rel)))
    return pairs


def payload_digest(
    skill_dir: pathlib.Path,
    *,
    manifest_entry: str = "",
    manifest_scripts: Optional[List[Dict[str, Any]]] = None,
    manifest: Any = None,
    bucket: Optional[str] = None,
) -> str:
    """Release digest of an installed or staged payload directory.

    ``bucket`` names the data bucket whose per-install notes are skipped; by default the
    directory's own (a staged payload names its destination). Unreadable files fail closed
    through ``SkillPayloadUnreadable`` so no verdict is ever bound to a partial runtime surface.
    """
    root = pathlib.Path(skill_dir).resolve()
    pairs = _payload_file_digests(root, manifest_entry=manifest_entry, manifest_scripts=manifest_scripts,
                                  bucket=bucket)
    return release_digest_from_file_digests(pairs, release_declaration(root, manifest, bucket=bucket))


def _legacy_reduce(file_digests: Iterable[Tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    for rel, file_digest in file_digests:
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest)
    return digest.hexdigest()


def legacy_content_hashes(
    skill_dir: pathlib.Path,
    *,
    manifest_entry: str = "",
    manifest_scripts: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, ...]:
    """Version-1 hashes of the CURRENT bytes, for conversion only.

    Version 1 hashed every payload file including install sidecars; in the native bucket
    it exempted the top-level launcher marker (and before 6.31 it did not).
    """
    root = pathlib.Path(skill_dir).resolve()
    pairs = _payload_file_digests(
        root, manifest_entry=manifest_entry, manifest_scripts=manifest_scripts, include_notes=True,
    )
    every = _legacy_reduce(pairs)
    if root.parent.name != "native":
        return (every,)
    exempt = _legacy_reduce(pair for pair in pairs if pair[0] != ".seed-origin")
    return (exempt, every) if exempt != every else (exempt,)


# --- one-time conversion of persisted records --------------------------------------

CONVERSION_MARKER = ".identity-v2.json"


def _rebind(record: Dict[str, Any], key: str, legacy: frozenset, digest: str) -> bool:
    if str(record.get(key) or "") in legacy:
        record[key] = digest
        return True
    return False


def _rebind_review(data: Dict[str, Any], legacy: frozenset, digest: str) -> bool:
    changed = _rebind(data, "content_hash", legacy, digest)
    changed = _rebind(data, "reviewed_content_hash", legacy, digest) or changed
    author = data.get("author_disposition")
    if isinstance(author, dict):
        changed = _rebind(author, "subject_hash", legacy, digest) or changed
        reference = author.get("review_reference")
        if isinstance(reference, dict):
            changed = _rebind(reference, "content_hash", legacy, digest) or changed
            preflight = reference.get("preflight")
            if isinstance(preflight, dict):
                changed = _rebind(preflight, "content_hash", legacy, digest) or changed
    if changed:
        # History rows of this snapshot keep their version-1 hash; free replay reads
        # this bounded association while review.json still describes these bytes.
        data["legacy_content_hashes"] = sorted(legacy)
    return changed


def convert_skill_records(drive_root: pathlib.Path, name: str, legacy: Iterable[str], digest: str) -> List[str]:
    """Rebind the authority records of one skill from a version-1 hash of its unchanged
    bytes to ``digest``. Returns the converted file names. Records bound to other bytes
    are left alone (stale stays stale); publication history keeps its facts with the current identity."""
    from ouroboros.skill_loader import skill_state_dir_path
    from ouroboros.utils import atomic_write_json, read_json_dict

    legacy_set = frozenset(str(item) for item in legacy if item and item != digest)
    if not legacy_set or not digest:
        return []
    state_dir = skill_state_dir_path(drive_root, name)
    converted: List[str] = []
    review = read_json_dict(state_dir / "review.json")
    if review is not None and _rebind_review(review, legacy_set, digest):
        atomic_write_json(state_dir / "review.json", review)
        converted.append("review.json")
    for filename in ("grants.json", "owner_attestation.json"):
        data = read_json_dict(state_dir / filename)
        if data is not None and _rebind(data, "content_hash", legacy_set, digest):
            atomic_write_json(state_dir / filename, data)
            converted.append(filename)
    from ouroboros.marketplace.provenance import PUBLICATION_FILENAME, merge_state_record

    receipt = read_json_dict(state_dir / PUBLICATION_FILENAME) or {}
    published = receipt.get("published")
    if isinstance(published, dict) and _rebind(published, "content_hash", legacy_set, digest):
        merge_state_record(drive_root, name, PUBLICATION_FILENAME, {"published": published})
        converted.append(PUBLICATION_FILENAME)
    return converted


def convert_skill_identities(drive_root: pathlib.Path, *, log_obj: Any = None,
                             repo_path: Optional[str] = None,
                             on_skill: Any = None) -> Dict[str, Any]:
    """One-time, idempotent move of persisted records to the release digest.

    Runs once per data root (``state/skills/.identity-v2.json`` records it); a second
    run is a no-op, and an interrupted run converts the rest next time because a
    converted record no longer equals a version-1 hash. ``on_skill(skill, legacy)``
    lets the launcher record built-in install facts during the same pass. No payload
    is moved and nothing is re-reviewed.
    """
    from ouroboros.skill_loader import discover_skills
    from ouroboros.utils import atomic_write_json, utc_now_iso

    logger = log_obj or log
    marker = pathlib.Path(drive_root) / "state" / "skills" / CONVERSION_MARKER
    if marker.is_file():
        return {"skipped": True}
    summary: Dict[str, Any] = {"converted": {}, "errors": []}
    for skill in discover_skills(pathlib.Path(drive_root), repo_path=repo_path):
        if skill.identity_collision or skill.load_error or not skill.content_hash:
            continue
        try:
            legacy = legacy_content_hashes(
                skill.skill_dir, manifest_entry=skill.manifest.entry, manifest_scripts=skill.manifest.scripts,
            )
            if on_skill is not None:
                on_skill(skill, legacy)
            files = convert_skill_records(pathlib.Path(drive_root), skill.name, legacy, skill.content_hash)
        except Exception as exc:
            logger.warning("Skill identity conversion failed for %s", skill.name, exc_info=True)
            summary["errors"].append(f"{skill.name}: {type(exc).__name__}: {exc}")
            continue
        if files:
            summary["converted"][skill.name] = files
    if not summary["errors"]:
        marker.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(marker, {"schema_version": 1, "converted_at": utc_now_iso(),
                                   "converted": summary["converted"], "errors": []})
    if summary["converted"]:
        logger.info("Skill identity v2: converted records for %d skill(s)", len(summary["converted"]))
    return summary


__all__ = [
    "CONVERSION_MARKER",
    "DIGEST_DOMAIN",
    "PER_INSTALL_NOTE_BUCKETS",
    "PER_INSTALL_NOTE_FILENAMES",
    "convert_skill_identities",
    "convert_skill_records",
    "declaration_digest",
    "is_release_digest",
    "legacy_content_hashes",
    "manifest_declaration",
    "payload_digest",
    "release_declaration",
    "release_digest_from_file_digests",
    "release_digest_from_files",
]
