"""Task-scoped artifact helpers shared by tools and outcome finalization."""

from __future__ import annotations

import logging
import mimetypes
import pathlib
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Dict, Iterable, List, Union

from ouroboros.utils import atomic_write_json, read_json_dict
from ouroboros.headless import ARTIFACT_STATUS_READY, SCRATCH_MANIFEST_NAME, task_artifacts_dir
from ouroboros.task_results import validate_task_id

log = logging.getLogger(__name__)

_ARTIFACT_MANIFEST = ".artifact_manifest.json"
_ARTIFACT_VERSION_RETENTION = 5
_ARTIFACT_VERSIONS_DIR = "artifact_versions"

# Ephemeral verification scratch (v6.52.2): the task-scoped manifest of {ABSOLUTE_path: sha256}
# FINGERPRINTS for files the agent declared via run_command/run_script `scratch=[...]` — transient
# in-workspace files (e.g. a throwaway test it writes, runs, and deletes) that are NOT deliverables.
# Workspace patch capture (headless.write_workspace_patch_artifacts) reads this and EXCLUDES an
# untracked file ONLY while its CURRENT content still matches the recorded sha — so a LATER real file
# at the same path is never dropped. Empty manifest => no effect. Filename SSOT: ouroboros.headless.
_MAX_SCRATCH_PATHS = 1000

# Input-attachment staging (v6.52.0, P1 first-class attachment access): the
# subdir under the task artifact store that holds STAGED INPUT files (never task
# deliverables — collect_task_artifact_records excludes it). Bounds keep one task
# from importing an unbounded amount of host data.
_ATTACHMENTS_SUBDIR = "attachments"
_MAX_STAGED_ATTACHMENTS = 25
_MAX_STAGED_ATTACHMENT_BYTES = 50 * 1024 * 1024  # ~50 MB per file


def _safe_attachment_name(raw_name: str) -> str:
    """Sanitize an attachment basename (mirrors gateway/files._sanitize_upload_filename)."""

    cleaned = str(raw_name or "").replace("\\", "/").strip()
    name = pathlib.PurePosixPath(cleaned).name.strip()
    if not name or name in {".", ".."} or "/" in name:
        name = "attachment"
    # Restrict to safe filename chars (alnum + . _ -) and bound length, so the rendered
    # read_file(root='artifact_store', path='attachments/<name>') manifest line cannot be broken
    # by apostrophes / quotes / newlines / backticks in the original filename.
    name = "".join(c if (c.isalnum() or c in "._-") else "_" for c in name)[:200] or "attachment"
    # A staged name must NOT start with '.': artifact_store_path_block_reason blocks leading-dot
    # components, which would make the advertised read_file(root='artifact_store',
    # path='attachments/<name>') unreadable for an attached dotfile (e.g. .gitignore).
    if name.startswith("."):
        name = "_" + name
    return name


def stage_task_attachments(
    drive_root: Union[pathlib.Path, str],
    task_id: str,
    attachments: Any,
) -> List[Dict[str, Any]]:
    """Stage input attachments into the task artifact store and return a manifest.

    Every task surface (CLI/API, GAIA solver, desktop chat) routes its attachments
    through here so they land in ONE agent-readable root (``artifact_store``) and
    become reachable via ``read_file(root='artifact_store', path='attachments/...')``
    instead of a bare absolute host path. Secret SOURCES are skipped (SSOT: the
    ``ouroboros.tool_access`` secret blocklist). Never raises — a per-file error
    skips just that file.

    Returns a list of manifest entries::

        {"label", "root": "artifact_store", "relpath": "attachments/<safe>",
         "mime", "is_image"}
    """

    items: List[Dict[str, Any]] = []
    if isinstance(attachments, list):
        for item in attachments:
            if isinstance(item, dict):
                path = str(item.get("path") or "").strip()
                label = str(item.get("label") or item.get("display_name") or "").strip()
            else:
                path = str(item or "").strip()
                label = ""
            if path:
                items.append({"path": path, "label": label})
    if not items:
        return []

    # SSOT secret detection: reuse the user_files secret blocklist so a credential
    # SOURCE (e.g. ~/.ssh/id_rsa, credentials.json, *.pem) is never copied in.
    from ouroboros.tool_access import (
        _USER_FILES_ALLOWED_DOTNAMES,
        _USER_FILES_SECRET_COMPONENTS,
        _USER_FILES_SECRET_NAMES,
        _USER_FILES_SECRET_RE,
    )

    def _is_secret_source(src: pathlib.Path) -> bool:
        for part in src.parts:
            part_lower = part.lower()
            if part_lower in _USER_FILES_SECRET_COMPONENTS:
                return True
            # Parity with user_files_path_block_reason (DEFAULT-DENY dotted): a non-allowlisted
            # dotted SOURCE component is potentially credential-bearing, so an enumerated-blocklist
            # gap (e.g. ~/.terraform.d/credentials.tfrc.json) can't auto-stage a secret. Owner-
            # supplied attachments only — defense-in-depth parity, not a live agent-exfil path.
            if part.startswith(".") and part_lower not in _USER_FILES_ALLOWED_DOTNAMES:
                return True
        name = src.name
        name_lower = name.lower()
        return bool(
            name_lower in _USER_FILES_SECRET_NAMES
            or _USER_FILES_SECRET_RE.search(name)
            or name_lower.endswith((".key", ".pem", ".p12", ".pfx"))
        )

    try:
        attach_dir = task_artifact_dir_path(drive_root, task_id, create=True) / _ATTACHMENTS_SUBDIR
    except Exception:
        log.debug("stage_task_attachments: could not resolve attachment dir", exc_info=True)
        return []

    manifest: List[Dict[str, Any]] = []
    staged = 0
    for item in items:
        if staged >= _MAX_STAGED_ATTACHMENTS:
            log.info("stage_task_attachments: hit max staged attachments (%d); skipping rest", _MAX_STAGED_ATTACHMENTS)
            break
        try:
            source = pathlib.Path(item["path"]).expanduser().resolve(strict=False)
            if not source.is_file():
                continue
            if _is_secret_source(source):
                log.info("stage_task_attachments: skipped secret source %s", source.name)
                continue
            try:
                if source.stat().st_size > _MAX_STAGED_ATTACHMENT_BYTES:
                    log.info("stage_task_attachments: skipped oversized source %s", source.name)
                    continue
            except OSError:
                continue
            attach_dir.mkdir(parents=True, exist_ok=True)
            # The stored filename derives from the SOURCE basename (it carries the
            # real extension, which mime detection needs); the human label is for
            # display only and is kept in the manifest entry.
            safe_name = _safe_attachment_name(source.name)
            dest = attach_dir / safe_name
            # Collision-safe destination: distinct sources never clobber each other.
            if dest.exists() and dest.resolve(strict=False) != source.resolve(strict=False):
                suffix = pathlib.Path(safe_name).suffix
                stem = safe_name[: -len(suffix)] if suffix else safe_name
                dest = attach_dir / f"{stem}.{uuid.uuid4().hex[:8]}{suffix}"
            if dest.resolve(strict=False) != source.resolve(strict=False):
                shutil.copy2(source, dest)
            mime = mimetypes.guess_type(str(dest))[0] or "application/octet-stream"
            # Sanitize the label rendered verbatim into the [ATTACHMENTS] manifest line / image
            # caption: drop control chars + collapse whitespace (incl. newlines) + bound, so a
            # crafted filename cannot inject extra prompt lines or break the rendered read_file line.
            _raw_label = str(item.get("label") or "").strip() or source.name
            label = " ".join("".join(c for c in _raw_label if c.isprintable()).split())[:120] or "attachment"
            manifest.append({
                "label": label,
                "root": "artifact_store",
                "relpath": f"{_ATTACHMENTS_SUBDIR}/{dest.name}",
                # v6.54.3: the REAL staged path, for process tools (a python/audio
                # script must open its own staged attachment directly — GAIA showed
                # models GUESSING a wrong absolute path and hitting light-mode
                # blocks when only the read_file() form was given). The path is
                # inside the task's own artifact_store, so scripts reach it under
                # every runtime mode.
                "abs_path": str(dest),
                "mime": mime,
                "is_image": mime.startswith("image/"),
            })
            staged += 1
        except Exception:
            log.debug("stage_task_attachments: skipped a file on error", exc_info=True)
            continue
    return manifest


def artifact_store_path_block_reason(path: pathlib.Path) -> str:
    """Return a block reason for task-artifact control/provenance paths."""

    try:
        parts = pathlib.Path(path).parts
    except TypeError:
        parts = (str(path),)
    for part in parts:
        if part.startswith("."):
            return "artifact_store hidden/control metadata paths are reserved"
    return ""


def task_artifact_dir_path(drive_root: Union[pathlib.Path, str], task_id: str, *, create: bool = False) -> pathlib.Path:
    """Return the task artifact directory without creating it unless requested."""

    return task_artifacts_dir(pathlib.Path(drive_root), validate_task_id(task_id), create=create)


def task_id_for_artifacts(ctx: Any) -> str:
    """Return a stable task id for artifact storage."""

    for value in (
        getattr(ctx, "task_id", None),
        (getattr(ctx, "task_metadata", {}) or {}).get("task_id")
        if isinstance(getattr(ctx, "task_metadata", {}), dict)
        else "",
        (getattr(ctx, "task_metadata", {}) or {}).get("id")
        if isinstance(getattr(ctx, "task_metadata", {}), dict)
        else "",
    ):
        try:
            return validate_task_id(value)
        except ValueError:
            continue
    return "interactive"


def record_task_scratch(ctx: Any, fingerprints: Dict[str, str]) -> None:
    """Record declared ephemeral-scratch FINGERPRINTS {abs_path: sha256} (task-scoped, additive
    union across calls; a newer sha for a path wins) so workspace patch capture can EXCLUDE a file
    ONLY while it still matches the recorded scratch content. Recording the sha (not just the path)
    is what keeps the manifest from being stale-authoritative: a LATER real file written to the same
    path has a different sha and is therefore NOT dropped from the patch. Fail-soft and bounded;
    written to BOTH the canonical ``budget_drive_root`` (where the supervisor finalizes the patch)
    AND the live ``drive_root`` (the child drive for forked/workspace tasks)."""
    fps = {
        str(k).strip(): str(v).strip()
        for k, v in (fingerprints or {}).items()
        if str(k or "").strip() and str(v or "").strip()
    }
    if not fps:
        return
    roots: List[str] = []
    for attr in ("budget_drive_root", "drive_root"):
        value = str(getattr(ctx, attr, "") or "").strip()
        if value and value not in roots:
            roots.append(value)
    if not roots:
        return
    task_id = task_id_for_artifacts(ctx)
    for root in roots:
        try:
            artifact_dir = task_artifact_dir_path(pathlib.Path(root), task_id, create=True)
            manifest = artifact_dir / SCRATCH_MANIFEST_NAME
            data = read_json_dict(manifest) or {}
            existing = data.get("scratch") if isinstance(data.get("scratch"), dict) else {}
            merged = {**{str(k): str(v) for k, v in existing.items()}, **fps}
            if len(merged) > _MAX_SCRATCH_PATHS:  # keep the most recent entries
                merged = dict(list(merged.items())[-_MAX_SCRATCH_PATHS:])
            atomic_write_json(manifest, {"schema_version": 2, "scratch": merged}, trailing_newline=True)
        except Exception:  # noqa: BLE001 — scratch manifest is advisory leak-hygiene, never load-bearing
            log.debug("record_task_scratch failed for root=%s", root, exc_info=True)


def read_task_scratch_fingerprints(drive_root: Union[pathlib.Path, str], task_id: str) -> Dict[str, str]:
    """Return the recorded ephemeral-scratch fingerprints {abs_path: sha256} for a task (empty when
    none). Patch capture excludes an untracked file only when its CURRENT sha matches the value here."""
    try:
        artifact_dir = task_artifact_dir_path(pathlib.Path(drive_root), validate_task_id(task_id), create=False)
        data = read_json_dict(artifact_dir / SCRATCH_MANIFEST_NAME) or {}
    except Exception:  # noqa: BLE001
        return {}
    vals = data.get("scratch")
    return {str(k): str(v) for k, v in vals.items()} if isinstance(vals, dict) else {}


def artifact_record(path: pathlib.Path, *, kind: str = "task_artifact", source_path: str = "") -> Dict[str, Any]:
    raw = pathlib.Path(path).read_bytes()
    record: Dict[str, Any] = {
        "kind": kind,
        "name": pathlib.Path(path).name,
        "path": str(path),
        "size": len(raw),
        "sha256": sha256(raw).hexdigest(),
        "status": ARTIFACT_STATUS_READY,
        "errors": [],
    }
    if source_path:
        record["source_path"] = source_path
    return record


def _artifact_versions_dir(drive_root: pathlib.Path, task_id: str, artifact_name: str) -> pathlib.Path:
    safe_name = pathlib.Path(artifact_name).name.replace("/", "_").replace("\\", "_")
    if not safe_name or safe_name in {".", ".."}:
        safe_name = "artifact"
    return pathlib.Path(drive_root) / "task_results" / _ARTIFACT_VERSIONS_DIR / validate_task_id(task_id) / safe_name


def _archive_previous_artifact_version(drive_root: pathlib.Path, task_id: str, dest: pathlib.Path, source: pathlib.Path) -> None:
    if not dest.is_file() or not source.is_file():
        return
    try:
        previous = dest.read_bytes()
        current = source.read_bytes()
    except OSError:
        return
    if previous == current:
        return
    version_dir = _artifact_versions_dir(drive_root, task_id, dest.name)
    version_dir.mkdir(parents=True, exist_ok=True)
    suffix = dest.suffix
    stem = dest.name[: -len(suffix)] if suffix else dest.name
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    digest = sha256(previous).hexdigest()[:12]
    version_path = version_dir / f"{stamp}.{digest}.{stem}{suffix}"
    version_path.write_bytes(previous)
    versions = sorted((p for p in version_dir.iterdir() if p.is_file()), key=lambda p: p.name)
    for stale in versions[:-_ARTIFACT_VERSION_RETENTION]:
        try:
            stale.unlink()
        except OSError:
            continue


def copy_file_to_task_artifacts(ctx: Any, source_path: Union[pathlib.Path, str], *, kind: str = "user_file") -> Dict[str, Any] | None:
    """Copy a generated file into this task's canonical artifact store."""

    source = pathlib.Path(source_path).expanduser().resolve(strict=False)
    if not source.is_file():
        return None
    task_id = task_id_for_artifacts(ctx)
    artifact_dir = task_artifact_dir_path(pathlib.Path(getattr(ctx, "drive_root")), task_id, create=True)
    data = read_json_dict(artifact_dir / _ARTIFACT_MANIFEST) or {}
    manifest = data.get("artifacts") if isinstance(data.get("artifacts"), dict) else {}
    manifest = {str(key): dict(value) for key, value in manifest.items() if isinstance(value, dict)}
    dest = artifact_dir / source.name
    reused_existing_source = False
    for existing in manifest.values():
        existing_source = str(existing.get("source_path") or "")
        existing_path = str(existing.get("path") or "")
        if existing_source == str(source) and existing_path:
            candidate = pathlib.Path(existing_path).resolve(strict=False)
            if candidate.parent == artifact_dir.resolve(strict=False):
                dest = candidate
                reused_existing_source = True
                break
    if dest.exists() and dest.resolve(strict=False) != source.resolve(strict=False) and not reused_existing_source:
        suffix = source.suffix
        stem = source.name[: -len(suffix)] if suffix else source.name
        digest = sha256(str(source.resolve(strict=False)).encode("utf-8", errors="replace")).hexdigest()[:8]
        dest = artifact_dir / f"{stem}.{digest}{suffix}"
    if kind == "user_file" and reused_existing_source and dest.resolve(strict=False) != source.resolve(strict=False):
        _archive_previous_artifact_version(pathlib.Path(getattr(ctx, "drive_root")), task_id, dest, source)
    if dest.resolve(strict=False) != source.resolve(strict=False):
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    record = artifact_record(dest, kind=kind, source_path=str(source))
    manifest[pathlib.Path(str(record.get("path") or record.get("name") or "")).name] = dict(record)
    atomic_write_json(artifact_dir / _ARTIFACT_MANIFEST, {"schema_version": 1, "artifacts": manifest}, trailing_newline=True)
    return record


def copy_directory_to_task_artifacts(
    ctx: Any,
    source_path: Union[pathlib.Path, str],
    *,
    kind: str = "process_output_directory",
    member_paths: Iterable[pathlib.Path] | None = None,
) -> List[Dict[str, Any]]:
    """Package a generated directory as a manifest ledger plus zip artifact."""

    source = pathlib.Path(source_path).expanduser().resolve(strict=False)
    if not source.is_dir():
        return []
    task_id = task_id_for_artifacts(ctx)
    artifact_dir = task_artifact_dir_path(pathlib.Path(getattr(ctx, "drive_root")), task_id, create=True)
    data = read_json_dict(artifact_dir / _ARTIFACT_MANIFEST) or {}
    manifest = data.get("artifacts") if isinstance(data.get("artifacts"), dict) else {}
    manifest = {str(key): dict(value) for key, value in manifest.items() if isinstance(value, dict)}
    root = source.resolve(strict=False)
    if member_paths is None:
        members = sorted(p for p in source.rglob("*") if p.is_file() and not p.is_symlink())
    else:
        members = sorted(pathlib.Path(p).resolve(strict=False) for p in member_paths)
    file_records: List[Dict[str, Any]] = []
    member_blobs: List[tuple[str, bytes, str]] = []
    tree_hasher = sha256()
    tree_hasher.update(str(source).encode("utf-8", errors="replace"))
    tree_hasher.update(b"\0")
    for path in members:
        if not path.is_file() or path.is_symlink():
            continue
        try:
            rel = path.resolve(strict=False).relative_to(root).as_posix()
        except ValueError:
            continue
        raw = path.read_bytes()
        digest = sha256(raw).hexdigest()
        tree_hasher.update(rel.encode("utf-8", errors="replace"))
        tree_hasher.update(b"\0")
        tree_hasher.update(digest.encode("ascii"))
        tree_hasher.update(b"\0")
        member_blobs.append((rel, raw, digest))
        file_records.append({
            "path": rel,
            "size": len(raw),
            "sha256": digest,
        })
    safe_stem = source.name.replace("/", "_").replace("\\", "_") or "directory"
    tree_digest = tree_hasher.hexdigest()[:8]
    ledger_path = artifact_dir / f"{safe_stem}.{tree_digest}.manifest.json"
    zip_path = artifact_dir / f"{safe_stem}.{tree_digest}.zip"
    tmp_zip_path = artifact_dir / f".{zip_path.name}.{uuid.uuid4().hex}.tmp"
    tmp_ledger_path = artifact_dir / f".{ledger_path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with zipfile.ZipFile(tmp_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for rel, raw, _digest in member_blobs:
                archive.writestr(rel, raw)
        atomic_write_json(
            tmp_ledger_path,
            {
                "schema_version": 1,
                "kind": kind,
                "source_path": str(source),
                "file_count": len(file_records),
                "files": file_records,
                "zip_name": zip_path.name,
            },
            trailing_newline=True,
        )
        tmp_zip_path.replace(zip_path)
        tmp_ledger_path.replace(ledger_path)
    except Exception:
        for tmp_path in (tmp_zip_path, tmp_ledger_path):
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
        raise
    records = [
        artifact_record(ledger_path, kind=f"{kind}_manifest", source_path=str(source)),
        artifact_record(zip_path, kind=kind, source_path=str(source)),
    ]
    for record in records:
        manifest[pathlib.Path(str(record.get("path") or record.get("name") or "")).name] = dict(record)
    atomic_write_json(artifact_dir / _ARTIFACT_MANIFEST, {"schema_version": 1, "artifacts": manifest}, trailing_newline=True)
    return records


def collect_task_artifact_records(drive_root: Union[pathlib.Path, str], task_id: str) -> List[Dict[str, Any]]:
    """Return records for files already present in the task artifact store."""

    try:
        artifact_dir = task_artifact_dir_path(pathlib.Path(drive_root), validate_task_id(task_id), create=False)
    except ValueError:
        return []
    records: List[Dict[str, Any]] = []
    if not artifact_dir.exists():
        return records
    data = read_json_dict(artifact_dir / _ARTIFACT_MANIFEST) or {}
    raw_manifest = data.get("artifacts") if isinstance(data.get("artifacts"), dict) else {}
    manifest = {str(key): dict(value) for key, value in raw_manifest.items() if isinstance(value, dict)}
    artifact_root = artifact_dir.resolve(strict=False)
    for path in sorted(p for p in artifact_dir.rglob("*") if p.is_file() and not p.is_symlink()):
        # Internal task-metadata files (the artifact manifest and the v6.52.2 scratch manifest)
        # are NOT deliverables — never record them as produced artifacts.
        if path.name in (_ARTIFACT_MANIFEST, SCRATCH_MANIFEST_NAME):
            continue
        try:
            rel_parts = path.resolve(strict=False).relative_to(artifact_root).parts
        except (OSError, ValueError):
            continue
        # v6.52.0 (P1): staged INPUT attachments live under attachments/ and are NOT
        # task deliverables — never record them as produced artifacts.
        if rel_parts and rel_parts[0] == _ATTACHMENTS_SUBDIR:
            continue
        try:
            record = artifact_record(path)
            manifest_record = manifest.get(path.name)
            if manifest_record:
                record.update({
                    key: value
                    for key, value in manifest_record.items()
                    if key not in {"path", "size", "sha256", "status", "errors"} and value
                })
            records.append(record)
        except OSError:
            continue
    return records


def merge_artifact_records(*groups: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for group in groups:
        for item in group:
            if not isinstance(item, dict):
                continue
            key = str(item.get("path") or item.get("name") or "")
            if not key:
                continue
            if key not in merged:
                order.append(key)
                merged[key] = dict(item)
                continue
            existing = merged[key]
            fresh = dict(item)
            merged[key] = {**existing, **fresh}
            if existing.get("kind") and fresh.get("kind") == "task_artifact" and existing.get("kind") != "task_artifact":
                merged[key]["kind"] = existing["kind"]
            for meta_key in ("kind", "source_path", "name"):
                if existing.get(meta_key) and not fresh.get(meta_key):
                    merged[key][meta_key] = existing[meta_key]
            if existing.get("name") and fresh.get("kind") == "task_artifact":
                merged[key]["name"] = existing["name"]
    return [merged[key] for key in order]
