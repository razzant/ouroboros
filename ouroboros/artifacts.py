"""Task-scoped artifact helpers shared by tools and outcome finalization."""

from __future__ import annotations

import errno
import json
import logging
import os
import stat
import pathlib
import re
import shutil
from contextlib import nullcontext, suppress
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Dict, Iterable, List, Optional, Union

from ouroboros.artifact_storage import clone_file, temporary_sibling
from ouroboros.utils import atomic_write_json, read_json_dict, update_json_locked, write_bytes_atomic
from ouroboros.headless import ARTIFACT_STATUS_FAILED, ARTIFACT_STATUS_READY, SCRATCH_MANIFEST_NAME, task_artifacts_dir
from ouroboros.outcome_receipt_store import is_verification_receipts_path
from ouroboros.task_custody import fence_publication
from ouroboros.task_results import validate_task_id

log = logging.getLogger(__name__)

_ARTIFACT_MANIFEST = ".artifact_manifest.json"
_ARTIFACT_VERSION_RETENTION = 5
_ARTIFACT_VERSIONS_DIR = "artifact_versions"


def text_source_range_projection(
    text: str, kind: str, start_char: Any = None, end_char: Any = None,
) -> tuple[dict[str, Any] | None, str]:
    """Project an explicit character range while retaining complete-source identity."""
    projection = {"schema": 1, "kind": kind, "complete_chars": len(text),
                  "complete_sha256": sha256(text.encode("utf-8")).hexdigest()}
    if start_char is None and end_char is None:
        projection["range_required"] = True
        return projection, "source_range_required"
    if (type(start_char) is not int or type(end_char) is not int
            or start_char < 0 or end_char <= start_char or end_char > len(text)):
        # Still no text, but the caller learns the length it must fit a range into:
        # a bare "invalid" was retried blind (and read as "unavailable").
        projection["requested_range"] = [start_char, end_char]
        return projection, "source_range_invalid"
    part = text[start_char:end_char]
    projection.update(start_char=start_char, end_char=end_char, text=part,
                      text_chars=len(part), text_sha256=sha256(part.encode("utf-8")).hexdigest())
    return projection, ""

# Ephemeral verification scratch (v6.52.2): the task-scoped manifest of {ABSOLUTE_path: sha256}
# FINGERPRINTS for files the agent declared via run_command/run_script `scratch=[...]` — transient
# in-workspace files (e.g. a throwaway test it writes, runs, and deletes) that are NOT deliverables.
# Workspace patch capture (headless.write_workspace_patch_artifacts) reads this and EXCLUDES an
# untracked file ONLY while its CURRENT content still matches the recorded sha — so a LATER real file
# at the same path is never dropped. Empty manifest => no effect. Filename SSOT: ouroboros.headless.
_MAX_SCRATCH_PATHS = 1000

# Input-attachment staging (v6.52.0, P1 first-class attachment access): the
# subdir under the task artifact store that holds STAGED INPUT files (never task
# deliverables — collect_task_artifact_records excludes it). Only the inline
# projection is bounded; complete captured inputs stay behind a manifest ref.
_ATTACHMENTS_SUBDIR = "attachments"
_CHAT_MEDIA_SUBDIR = "chat_media"
_SOURCE_HANDLES_SUBDIR = "source_handles"
_SOURCE_HANDLE_CATEGORIES = frozenset({"tool_results", "context_checkpoints", "delegated_activity"})
_LEGACY_TOOL_RESULT_TRUNCATION_RE = re.compile(
    r"\n\.\.\. \(truncated from (?P<original>[1-9][0-9]*) chars, "
    r"limit=(?P<limit>[1-9][0-9]*)\)"
    r"(?:\nFULL_RESULT_SOURCE_UNAVAILABLE=true"
    r"\nDo not treat this partial result as complete; exact source persistence failed\.)?\Z"
)
_CHAT_MEDIA_EXTENSIONS = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/gif": "gif",
    "image/webp": "webp",
    "video/mp4": "mp4",
    "video/webm": "webm",
}
_CHAT_MEDIA_NAME_RE = re.compile(
    r"^chat-media-[0-9a-f]{64}\.(png|jpg|gif|webp|mp4|webm)$"
)
_MAX_STAGED_ATTACHMENTS = 25


class _StagedAttachmentManifest(list):
    """List-compatible manifest with call-private cleanup ownership."""

    def __init__(self) -> None:
        super().__init__()
        self._cleanup_owned_paths: set[pathlib.Path] = set()


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
    ``ouroboros.credential_shapes`` vocabulary for host paths; the workspace
    patch rules for /api/chat/upload byte uploads, judged on the ORIGINAL
    basename). Never raises. Every declared
    input produces exactly one ordinal-preserving ``staged`` or ``rejected``
    row, so callers never confuse a partial staging result with the complete
    attachment set.

    Returns a list of manifest entries::

        {"ordinal", "status": "staged", "reason": "", "label",
         "root": "artifact_store", "relpath": "attachments/<safe>",
         "mime", "is_image"}

        {"ordinal", "status": "rejected", "reason": "source_missing",
         "label"}
    """

    declared = list(attachments) if isinstance(attachments, list) else []
    if not declared:
        return []
    try:
        from ouroboros.config import get_runtime_mode
        from ouroboros.runtime_mode_policy import mode_has_unrestricted_agency

        allow_owner_sensitive = mode_has_unrestricted_agency(get_runtime_mode())
    except Exception:
        allow_owner_sensitive = False

    def _display_label(item: Any, raw_path: str, ordinal: int) -> str:
        if isinstance(item, dict):
            raw = item.get("label") or item.get("display_name") or ""
        else:
            raw = ""
        if not str(raw or "").strip() and raw_path:
            raw = pathlib.Path(raw_path).name
        cleaned = " ".join(
            "".join(c for c in str(raw or "") if c.isprintable()).split()
        )[:120]
        return cleaned or f"attachment {ordinal + 1}"

    def _rejected(ordinal: int, label: str, reason: str) -> Dict[str, Any]:
        return {
            "ordinal": ordinal,
            "status": "rejected",
            "reason": str(reason),
            "label": label,
        }

    def _same_bytes(path: pathlib.Path, expected: Dict[str, Any]) -> bool:
        try:
            return not path.is_symlink() and stream_artifact_file(path) == expected
        except OSError:
            return False

    # SSOT secret detection: reuse the shared credential-shape vocabulary so a
    # credential SOURCE (e.g. ~/.ssh/id_rsa, credentials.json) is never copied in.
    from ouroboros.credential_shapes import user_files_mutation_shape_reason

    # G10 (capinv-447): BOTH attachment routes get ONE policy. A path-selected
    # attachment is judged on its host path; a /api/chat/upload byte upload sits
    # under data/uploads with a server-generated "<32hex>_<original>" basename
    # whose transport path is meaningless (its uuid prefix used to defeat the
    # name rules entirely, and a dotted DATA_DIR parent — ~/.local, ~/Library —
    # used to reject EVERY upload), so it is judged on the ORIGINAL basename.
    try:
        from ouroboros.config import DATA_DIR as _DATA_DIR

        _uploads_root = (pathlib.Path(_DATA_DIR) / "uploads").resolve(strict=False)
    except Exception:
        _uploads_root = None
    _upload_name_re = re.compile(r"^[0-9a-f]{32}_")

    def _secret_source_reason(src: pathlib.Path) -> str:
        """Rule-named reason the source must not be staged, or ``""``."""
        from ouroboros.workspace_patch_rules import _sensitive_untracked_reason

        if _uploads_root is not None:
            try:
                src.relative_to(_uploads_root)
            except ValueError:
                pass
            else:
                original = _upload_name_re.sub("", src.name)
                reason = _sensitive_untracked_reason(original)
                return f"uploaded file name {original!r}: {reason}" if reason else ""
        physical_reason = user_files_mutation_shape_reason(src, pathlib.Path.home())
        if physical_reason:
            return physical_reason
        name = src.name
        reason = _sensitive_untracked_reason(name)
        return f"file name {name!r}: {reason}" if reason else ""

    try:
        artifact_root = task_artifact_dir_path(drive_root, task_id, create=False).resolve(strict=False)
        attach_dir = artifact_root / _ATTACHMENTS_SUBDIR
        if not attach_dir.resolve(strict=False).is_relative_to(artifact_root):
            raise ValueError("attachment directory escapes its task owner")
    except Exception:
        log.debug("stage_task_attachments: could not resolve attachment dir", exc_info=True)
        return [
            _rejected(
                ordinal,
                _display_label(
                    item,
                    str(item.get("path") or "") if isinstance(item, dict) else str(item or ""),
                    ordinal,
                ),
                "staging_unavailable",
            )
            for ordinal, item in enumerate(declared)
        ]

    manifest = _StagedAttachmentManifest()
    for ordinal, raw_item in enumerate(declared):
        if isinstance(raw_item, dict):
            raw_path = str(raw_item.get("path") or "").strip()
        else:
            raw_path = str(raw_item or "").strip()
        label = _display_label(raw_item, raw_path, ordinal)
        if not raw_path:
            manifest.append(_rejected(ordinal, label, "invalid_path"))
            continue
        try:
            source = pathlib.Path(raw_path).expanduser().resolve(strict=False)
            if not source.exists():
                manifest.append(_rejected(ordinal, label, "source_missing"))
                continue
            if not source.is_file():
                manifest.append(_rejected(ordinal, label, "source_not_file"))
                continue
            if (secret_rule := _secret_source_reason(source)) and not allow_owner_sensitive:
                log.info("stage_task_attachments: skipped secret source %s (%s)", source.name, secret_rule)
                # Reason stays a closed vocabulary; the RULE that fired is named
                # separately so the owner sees exactly why (G10, capinv-447).
                row = _rejected(ordinal, label, "secret_source")
                row["rule"] = secret_rule
                manifest.append(row)
                continue
            source_identity = stream_artifact_file(source, expected=raw_item)
            attach_dir.mkdir(parents=True, exist_ok=True)
            # The stored filename derives from the SOURCE basename (it carries the
            # real extension, which mime detection needs); the human label is for
            # display only and is kept in the manifest entry.
            safe_name = _safe_attachment_name(source.name)
            dest = attach_dir / safe_name
            # Collision-safe destination: distinct sources never clobber each other.
            if dest.is_symlink() or (dest.exists() and dest.resolve(strict=False) != source.resolve(strict=False)):
                same_bytes = _same_bytes(dest, source_identity)
                if not same_bytes:
                    suffix = pathlib.Path(safe_name).suffix
                    stem = safe_name[: -len(suffix)] if suffix else safe_name
                    content_key = source_identity["sha256"][:12]
                    dest = attach_dir / f"{stem}.{content_key}{suffix}"
                    collision = 1
                    while dest.exists() and not _same_bytes(dest, source_identity):
                        collision += 1
                        dest = attach_dir / f"{stem}.{content_key}.{collision}{suffix}"
            if dest.resolve(strict=False) != source.resolve(strict=False):
                existed = dest.exists()
                try:
                    if not existed or not _same_bytes(dest, source_identity) or dest.samefile(source):
                        copy_artifact_file(source, dest, expected=source_identity)
                        if not existed:
                            manifest._cleanup_owned_paths.add(dest)
                except OSError:
                    if not existed:
                        dest.unlink(missing_ok=True)
                    manifest.append(_rejected(ordinal, label, "copy_failed"))
                    continue
            measured = stream_artifact_file(dest, expected=source_identity)
            from ouroboros.chat_uploads import file_media
            mime, kind = file_media(dest)
            manifest.append({
                **measured,
                "ordinal": ordinal,
                "status": "staged",
                "reason": "",
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
                "is_image": kind == "image",
            })
        except Exception:
            log.debug("stage_task_attachments: rejected a file on error", exc_info=True)
            manifest.append(_rejected(ordinal, label, "copy_failed"))
    return manifest


def attachment_manifest_projection(drive_root: Any, task_id: str, manifest: Any) -> Dict[str, Any]:
    """Publish the complete input manifest, returning only its bounded inline view."""
    from ouroboros.contracts.task_contract import normalize_attachment_manifest

    rows = normalize_attachment_manifest(manifest)
    projection = {"attachment_manifest": rows[:_MAX_STAGED_ATTACHMENTS]}
    if len(rows) > _MAX_STAGED_ATTACHMENTS:
        data = json.dumps({"schema_version": 1, "task_id": validate_task_id(task_id), "attachments": rows},
                          ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        projection["attachment_manifest_ref"] = {
            **store_actor_source_bytes(drive_root, task_id, category="context_checkpoints",
                                       source_id="attachments", data=data, extension="json"),
            "count": len(rows),
        }
    return projection


def resolve_attachment_manifest(drive_root: Any, task_id: str, authority: Any) -> List[Dict[str, Any]]:
    """Resolve complete captured inputs under the actual owner, never preview fallback.

    Only old inline manifests retain their historical absolute-path contract.
    Referenced rows resolve from this task's artifact root; serialized host paths
    cannot redirect the reader. File bytes are verified by the materializer.
    """
    from ouroboros.contracts.task_contract import normalize_attachment_manifest

    authority = authority if isinstance(authority, dict) else {}
    ref = authority.get("attachment_manifest_ref")
    if "attachment_manifest_ref" not in authority:
        return normalize_attachment_manifest(authority.get("attachment_manifest"))
    document = json.loads(read_actor_source_bytes(drive_root, task_id, ref))
    if not isinstance(document, dict) or document.get("schema_version") != 1 or document.get("task_id") != task_id:
        raise ValueError("attachment manifest owner does not match the task")
    raw = document.get("attachments")
    rows = normalize_attachment_manifest(raw)
    if not isinstance(raw, list) or len(rows) != len(raw) or len(rows) != ref.get("count"):
        raise ValueError("attachment manifest count is invalid")
    root = task_artifact_dir_path(drive_root, task_id).resolve(strict=False)
    for row in rows:
        if row["status"] == "rejected":
            row.pop("abs_path", None)
            continue
        rel = pathlib.PurePosixPath(str(row.get("relpath") or ""))
        if rel.is_absolute() or len(rel.parts) != 2 or rel.parts[0] != _ATTACHMENTS_SUBDIR or ".." in rel.parts:
            raise ValueError("attachment manifest contains an invalid file reference")
        path = root.joinpath(*rel.parts)
        if path.is_symlink() or not path.resolve(strict=False).is_relative_to(root):
            raise ValueError("attachment manifest file escapes its task owner")
        if row.get("root") != "artifact_store" or type(row.get("size")) is not int or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256") or "")):
            raise ValueError("attachment manifest file has no captured identity")
        row["abs_path"] = str(path)
    return rows


def promote_task_attachment_refs(parent: Any, child: Any, task_id: str, result: Dict[str, Any], state: Dict[str, Any]) -> None:
    """Carry the full input closure under existing child-ref custody before GC."""
    from ouroboros.owner_mailbox import promote_owner_attachments

    promote_owner_attachments(parent, child, task_id, state)
    contract = result.get("task_contract")
    if not isinstance(contract, dict):
        return
    rows = contract.get("attachment_manifest") or []
    ref = contract.get("attachment_manifest_ref")
    if not rows and "attachment_manifest_ref" not in contract:
        return
    try:
        if "attachment_manifest_ref" in contract:
            try:
                rows = resolve_attachment_manifest(parent, task_id, contract)
                # A previous successful publication is already entirely canonical.
                for row in rows:
                    if row["status"] == "staged":
                        stream_artifact_file(pathlib.Path(row["abs_path"]), expected=row)
            except (OSError, ValueError, TypeError):
                rows = resolve_attachment_manifest(child, task_id, contract)
        copied, error = materialize_inherited_attachment_manifest(rows, parent, task_id)
        if error:
            raise OSError(error)
        authority = attachment_manifest_projection(parent, task_id, copied)
        result["task_contract"] = {key: value for key, value in contract.items()
                                   if key != "attachment_manifest_ref"} | authority
        result.update(authority)
        if isinstance(result.get("metadata"), dict) and isinstance(result["metadata"].get("task_contract"), dict):
            result["metadata"] = {**result["metadata"], "task_contract": result["task_contract"]}
    except (OSError, ValueError, TypeError) as exc:
        paths = [str(row.get("abs_path") or "") for row in rows if isinstance(row, dict)]
        state["unavailable_refs"].append({"kind": "task_attachment", "reason": f"{type(exc).__name__}: {exc}"})
        if isinstance(ref, dict) and ref.get("path"):
            paths.append(str(task_artifact_dir_path(child, task_id) / str(ref["path"])))
        for path in paths:
            if path:
                state["pending_refs"].append({"kind": "task_attachment", "path": path,
                                               "reason": f"{type(exc).__name__}: {exc}"})


def attachment_manifest_all_rejected(manifest: Any) -> bool:
    """Whether staging rejected EVERY declared attachment (none staged).

    Partial staging is the default (В25c, capinv-447): good rows ride, rejected
    rows are disclosed. A fully-rejected set is different — the task would start
    with NONE of the material it declared it needs — so the flagless ingresses
    (UI chat, presence, promote) stay atomic for exactly that case."""

    return bool(
        isinstance(manifest, list) and manifest
        and all(
            isinstance(item, dict) and str(item.get("status") or "staged") == "rejected"
            for item in manifest
        )
    )


def attachment_manifest_has_rejections(manifest: Any) -> bool:
    """Whether a complete staging manifest contains any rejected declaration."""

    return bool(
        isinstance(manifest, list)
        and any(
            isinstance(item, dict) and str(item.get("status") or "staged") == "rejected"
            for item in manifest
        )
    )


def remove_staged_attachments(manifest: Any) -> int:
    """Unlink files a ``stage_task_attachments`` call just staged (GR2-9).

    Used when the admission that motivated the staging is REFUSED after the
    fact (the transactional cancel-pending re-check): the inputs must not
    linger in the artifact store of a task the supervisor is tearing down.
    Only entries carrying the staged ``abs_path`` this module wrote are
    touched. Never raises; returns the number of files removed.
    """
    removed = 0
    owned = {
        pathlib.Path(path)
        for path in getattr(manifest, "_cleanup_owned_paths", set())
    }
    attachment_dirs: set[pathlib.Path] = set()
    if not isinstance(manifest, list):
        return removed
    for entry in manifest:
        if not isinstance(entry, dict):
            continue
        staged = str(entry.get("abs_path") or "").strip()
        if not staged:
            continue
        try:
            path = pathlib.Path(staged)
            if path in owned and path.is_file() and _ATTACHMENTS_SUBDIR in path.parts:
                path.unlink(missing_ok=True)
                removed += 1
                attachment_dirs.add(path.parent)
        except Exception:
            log.debug("remove_staged_attachments: could not remove %s", staged, exc_info=True)
    for directory in attachment_dirs:
        try:
            directory.rmdir()
            directory.parent.rmdir()
        except OSError:
            # Another admitted attachment or artifact still owns the directory.
            pass
    return removed


def rebase_staged_attachment_manifest(
    manifest: Any,
    old_dir: Union[pathlib.Path, str],
    new_dir: Union[pathlib.Path, str],
) -> Any:
    """Rebase generated absolute paths and private cleanup ownership in place."""

    old, new = pathlib.Path(old_dir), pathlib.Path(new_dir)
    if not isinstance(manifest, list):
        return manifest
    for row in manifest:
        if not isinstance(row, dict) or not str(row.get("abs_path") or ""):
            continue
        path = pathlib.Path(str(row["abs_path"]))
        try:
            row["abs_path"] = str(new / path.relative_to(old))
        except ValueError:
            continue
    owned = getattr(manifest, "_cleanup_owned_paths", None)
    if isinstance(owned, set):
        rebased: set[pathlib.Path] = set()
        for path in owned:
            try:
                rebased.add(new / pathlib.Path(path).relative_to(old))
            except ValueError:
                rebased.add(pathlib.Path(path))
        manifest._cleanup_owned_paths = rebased
    return manifest


def materialize_inherited_attachment_manifest(
    manifest: Any,
    target_drive: Union[pathlib.Path, str],
    target_task_id: str,
) -> tuple[List[Dict[str, Any]], str]:
    """Copy inherited staged inputs into a child's own artifact store."""

    if not isinstance(manifest, list) or not manifest:
        return [], ""
    copied = _StagedAttachmentManifest()
    target_dir = task_artifact_dir_path(target_drive, target_task_id) / _ATTACHMENTS_SUBDIR
    try:
        for index, raw in enumerate(manifest):
            if not isinstance(raw, dict):
                continue
            row = dict(raw)
            if str(row.get("status") or "staged") == "rejected":
                row.pop("abs_path", None)
                copied.append(row)
                continue
            source = pathlib.Path(str(row.get("abs_path") or ""))
            if not source.is_file():
                raise FileNotFoundError(f"inherited attachment {index} is unavailable")
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / _safe_attachment_name(
                pathlib.Path(str(row.get("relpath") or source.name)).name
            )
            if target.is_symlink():
                raise OSError(f"inherited attachment collision at {target.name}")
            if not target.exists():
                measured = copy_artifact_file(source, target, expected=row)
                copied._cleanup_owned_paths.add(target)
            else:
                measured = stream_artifact_file(source, expected=row)
                stream_artifact_file(target, expected=measured)
                if target.samefile(source):
                    copy_artifact_file(source, target, expected=measured)
            row.update({
                **measured,
                "root": "artifact_store",
                "relpath": f"{_ATTACHMENTS_SUBDIR}/{target.name}",
                "abs_path": str(target),
            })
            copied.append(row)
        return copied, ""
    except Exception as exc:
        remove_staged_attachments(copied)
        return [], f"{type(exc).__name__}: {exc}"


def artifact_store_path_block_reason(
    path: pathlib.Path,
    *,
    base_path: pathlib.Path | None = None,
) -> str:
    """Return a block reason for task-artifact control/provenance paths."""

    from ouroboros.config import get_runtime_mode
    from ouroboros.runtime_mode_policy import mode_has_unrestricted_agency

    if mode_has_unrestricted_agency(get_runtime_mode()):
        return ""

    try:
        candidate = pathlib.Path(path)
        if base_path is not None:
            try:
                candidate = candidate.resolve(strict=False).relative_to(
                    pathlib.Path(base_path).resolve(strict=False)
                )
            except ValueError:
                pass
        parts = candidate.parts
    except TypeError:
        parts = (str(path),)
    from ouroboros.headless import SCRATCH_MANIFEST_NAME

    if any(part in {_ARTIFACT_MANIFEST, _ARTIFACT_MANIFEST + ".lock", SCRATCH_MANIFEST_NAME}
           for part in parts):
        return "artifact_store task metadata paths are reserved"
    if parts == ("verification_receipts.jsonl",):
        return "artifact_store verification receipt authority path is reserved"
    return ""


def task_artifact_dir_path(drive_root: Union[pathlib.Path, str], task_id: str, *, create: bool = False) -> pathlib.Path:
    """Return the task artifact directory without creating it unless requested."""

    return task_artifacts_dir(pathlib.Path(drive_root), validate_task_id(task_id), create=create)


def actor_source_reference(*, category: str, source_id: str, data: bytes, extension: str) -> Dict[str, Any]:
    """Price an exact actor-source address without creating a directory or writing bytes."""
    normalized_category = str(category or "").strip()
    if normalized_category not in _SOURCE_HANDLE_CATEGORIES:
        raise ValueError(f"unsupported source-handle category: {normalized_category}")
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(source_id or "source")).strip("._")
    safe_id = safe_id[:160] or "source"
    safe_extension = re.sub(r"[^A-Za-z0-9]+", "", str(extension or "bin"))[:12] or "bin"
    digest = sha256(data).hexdigest()
    relative = pathlib.PurePosixPath(
        _SOURCE_HANDLES_SUBDIR,
        normalized_category,
        f"{safe_id}-{digest}.{safe_extension}",
    )
    read = {"root": "artifact_store", "path": relative.as_posix(), "start_line": 1, "max_lines": 2000, "start_char": 0}
    return {"kind": "task_source", "root": "artifact_store", "path": relative.as_posix(), "size": len(data),
            "sha256": digest, "read": {"tool": "read_file", "arguments": read}}



def store_actor_source_bytes(
    drive_root: Union[pathlib.Path, str],
    task_id: str,
    *,
    category: str,
    source_id: str,
    data: bytes,
    extension: str,
) -> Dict[str, Any]:
    """Persist exact bytes inside this task's existing actor-readable artifact root."""

    ref = actor_source_reference(category=category, source_id=source_id, data=data, extension=extension)
    relative = pathlib.PurePosixPath(ref["path"])
    artifact_dir = task_artifact_dir_path(drive_root, task_id, create=True)
    target = artifact_dir.joinpath(*relative.parts)
    # WRITE-ONCE. The name carries the digest, so an existing file (loose or a pack member) with
    # these bytes IS this handle: rewriting would republish identical content while racing another
    # writer for the path (a Windows os.replace over a file a reader holds open is a sharing violation).
    try:
        already_stored = not target.is_symlink() and target.read_bytes() == bytes(data)
    except OSError:
        from ouroboros.evidence_packs import handle_info
        already_stored = handle_info(drive_root, task_id, relative.as_posix()) is not None
    if not already_stored:
        fence_publication()  # a closed publication generation writes no new source
        write_bytes_atomic(target, bytes(data))
    return ref


def read_actor_source_bytes(
    drive_root: Union[pathlib.Path, str], task_id: str, ref: Any, *, retained: bool = True, anchor: Any = None,
) -> bytes:
    """Verify a task source: loose, packed, then (unless disabled) retained; errors stay explicit."""

    if not isinstance(ref, dict) or ref.get("kind") != "task_source":
        raise ValueError("actor source ref has an unexpected kind")
    if ref.get("root") != "artifact_store":
        raise ValueError("actor source ref has an unexpected root")
    from ouroboros.confined_files import open_confined_file, physical_path, relative_parts, root_spellings

    parts = relative_parts(ref.get("path"))
    rel = pathlib.PurePosixPath(*parts)
    if not parts or parts[0] != _SOURCE_HANDLES_SUBDIR:
        raise ValueError("actor source ref has an invalid path")
    placement, root = pathlib.Path(drive_root), None
    if anchor is not None:  # placement readers below receive the configured anchor
        spellings = root_spellings(anchor)
        root, placement = spellings[0], physical_path(spellings, placement.absolute())
        target = task_artifact_dir_path(placement, task_id, create=False).joinpath(*rel.parts)
    else:  # Existing internal callers retain their placement-cache and scope/error contracts.
        base = task_artifact_dir_path(placement, task_id, create=False).resolve(strict=False)
        target = base.joinpath(*rel.parts)
        if target.is_symlink():
            raise ValueError("actor source ref may not be a symlink")
        target = target.resolve(strict=False)
        if not target.is_relative_to(base):
            raise ValueError("actor source ref escapes its task artifact root")
    from ouroboros.evidence_packs import handle_bytes
    try:
        if rel.parts[:2] == ("source_handles", "context_checkpoints"):
            raw = handle_bytes(placement, task_id, rel.as_posix(), anchor=anchor)
        elif root is not None:
            handle, _ = open_confined_file(root, target.relative_to(root).parts)
            with handle:
                raw = handle.read()
        else:
            raw = target.read_bytes()
    except FileNotFoundError:
        if not retained:
            raise
        from ouroboros.source_retention import read_retained_task_source
        return read_retained_task_source(placement, task_id, ref, anchor=anchor)
    try:
        expected_size = int(ref["size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("actor source ref has no valid size") from exc
    if len(raw) != expected_size:
        raise ValueError("actor source ref failed size verification")
    if sha256(raw).hexdigest() != str(ref.get("sha256") or ""):
        raise ValueError("actor source ref failed sha256 verification")
    return raw


def read_task_result_source_bytes(
    drive_root: Any, result: Dict[str, Any], name: str, source_path: str,
) -> bytes:
    """Read an exact published source ref, never a caller-selected filesystem path."""
    review = result.get("review_projection")
    panels = review.get("panels") if isinstance(review, dict) else []
    refs = [row.get("applied_source_ref") for row in (panels if isinstance(panels, list) else []) if isinstance(row, dict)]
    # completion observations, and the historical source an older record's acceptance debt pinned (read-only)
    for key in ("completion_observations", "acceptance_debt"):
        if isinstance(value := result.get(key), dict):
            refs.append(value.get("source_ref"))
    for ref in refs:
        if isinstance(ref, dict) and ref.get("path") == source_path and pathlib.PurePosixPath(source_path).name == name:
            return read_actor_source_bytes(drive_root, validate_task_id(result.get("task_id")), ref, anchor=drive_root)
    raise ValueError("the requested source is not published by this task result")


def persist_exact_text_source(
    drive_root: Union[pathlib.Path, str], task_id: str, *,
    source_id: str, text: str,
) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Persist and immediately verify one exact redacted text source."""
    ref: Dict[str, Any] = {}
    try:
        ref = store_actor_source_bytes(
            drive_root, task_id, category="tool_results", source_id=source_id,
            data=str(text).encode("utf-8"), extension="txt",
        )
        exact = read_actor_source_bytes(drive_root, task_id, ref).decode("utf-8")
        return exact, ref, {}
    except Exception as exc:
        return "", ref, {
            "status": "source_unavailable",
            "reason": f"{type(exc).__name__}: {exc}",
        }


def _matching_projection(drive_root: Any, call: Dict[str, Any]) -> Dict[str, Any]:
    """Read the call's redacted observability projection, verified to be the same call."""
    from ouroboros.observability import read_blob_ref

    trace = call.get("trace_ref") if isinstance(call.get("trace_ref"), dict) else {}
    payload = read_blob_ref(pathlib.Path(drive_root), trace.get("redacted_projection_ref") or {})
    # An absent tool_call_id cannot identify a call, so it never matches. The
    # argument gap publishes this text; the result fallback swallows it.
    if (not isinstance(payload, dict) or not call.get("tool_call_id")
            or payload.get("tool_call_id") != call["tool_call_id"]
            or payload.get("tool") != call.get("tool")):
        raise ValueError("argument source does not match the tool call")
    return payload


# sanitize_tool_args_for_log's transport markers — the ONE list, so a marker shape
# added there is never missed here (the colons keep a literal value like
# "_truncated" out).
SANITIZER_OMISSION_MARKERS = (
    "<TRUNCATED:", '"_depth_limit":', '"_truncated":', '"_repr":', '"_error":',
)


def materialize_tool_args_source(drive_root: Any, call: Dict[str, Any]) -> tuple[Any, bool, Dict[str, Any]]:
    """Recover logging-sanitizer omissions from the existing redacted call blob."""
    args = call.get("args")
    rendered = json.dumps(args, ensure_ascii=False, default=str)
    # These are sanitize_tool_args_for_log's transport markers, not a judgment
    # about the command's meaning. Legacy rows without them retain their view.
    if not any(marker in rendered for marker in SANITIZER_OMISSION_MARKERS):
        return args, True, {}
    trace = call.get("trace_ref") if isinstance(call.get("trace_ref"), dict) else {}
    ref = trace.get("redacted_projection_ref") or {}
    try:
        payload = _matching_projection(drive_root, call)
        if "args" not in payload:
            raise ValueError("argument source does not match the tool call")
        return payload["args"], True, {}
    except (OSError, TypeError, ValueError) as exc:
        return args, False, {"tool": str(call.get("tool") or ""), "status": "source_unavailable",
                             "reason": f"argument_source_unavailable: {type(exc).__name__}: {exc}",
                             "source_ref": ref}


def materialize_tool_result_source(
    drive_root: Union[pathlib.Path, str], task_id: str, call: Dict[str, Any],
) -> tuple[Any, bool, Dict[str, Any]]:
    """Materialize a result under existing redaction; metadata carries its source or gap. A partial result reads its verified actor source, then a matching redacted projection (re-persisted best-effort), else a source_unavailable gap."""
    result = call.get("result")
    legacy_match = (
        _LEGACY_TOOL_RESULT_TRUNCATION_RE.search(result)
        if "result_partial" not in call and isinstance(result, str) else None
    )
    legacy_partial = bool(
        legacy_match
        and int(legacy_match.group("limit")) == legacy_match.start()
        and int(legacy_match.group("original")) > int(legacy_match.group("limit"))
    )
    if not call.get("result_partial") and not legacy_partial:
        return result, True, {}
    ref = call.get("result_source_ref") if isinstance(call.get("result_source_ref"), dict) else {}
    gap = {
        "tool_call_id": str(call.get("tool_call_id") or ""), "tool": str(call.get("tool") or ""),
        "status": "source_unavailable", "source_ref": ref,
    }
    if legacy_partial:
        gap.update(reason="legacy_actor_truncation_without_source_ref", source_ref={})
    else:
        try:
            return read_actor_source_bytes(drive_root, task_id, ref).decode("utf-8"), True, {}
        except (OSError, UnicodeError, TypeError, ValueError) as exc:
            gap.update(reason=f"{type(exc).__name__}: {exc}",
                       declared_status=str(call.get("result_source_status") or ""))
    try:
        payload = _matching_projection(drive_root, call)
        if isinstance(payload.get("result"), str):
            text = payload["result"]
            _, recovered_ref, issue = persist_exact_text_source(
                drive_root, task_id, source_id=call["tool_call_id"], text=text,
            )
            return text, True, {"source_ref": {} if issue else recovered_ref}
    except Exception:
        pass  # A missing or corrupt projection preserves the primary-source failure.
    return result, False, gap


def store_chat_media_bytes(
    drive_root: Union[pathlib.Path, str], task_id: str, data: bytes, mime: str,
) -> Optional[Dict[str, Any]]:
    """Store reloadable outbound chat media outside the deliverable inventory."""

    normalized_mime = str(mime or "").lower()
    extension = _CHAT_MEDIA_EXTENSIONS.get(normalized_mime)
    if not extension:
        return None
    try:
        artifact_dir = task_artifact_dir_path(drive_root, task_id, create=True)
    except ValueError:
        return None
    digest = sha256(data).hexdigest()
    name = f"chat-media-{digest}.{extension}"
    media_dir = artifact_dir / _CHAT_MEDIA_SUBDIR
    media_dir.mkdir(parents=True, exist_ok=True)
    if media_dir.is_symlink():
        return None
    try:
        media_dir.resolve(strict=False).relative_to(artifact_dir.resolve(strict=False))
    except (OSError, ValueError):
        return None
    path = media_dir / name
    if path.is_symlink() or not path.is_file():
        write_bytes_atomic(path, data)
    # ``path`` is reported so a caller can derive a second, launcher-compatible
    # URL for the same bytes (see supervisor.message_bus).
    return {
        "name": name,
        "mime": normalized_mime,
        "sha256": digest,
        "size": len(data),
        "path": str(path),
    }


def resolve_chat_media_path(
    drive_root: Union[pathlib.Path, str], task_id: str, name: str,
) -> Optional[pathlib.Path]:
    """Resolve an exact content-addressed chat-media filename, or return ``None``."""

    match = _CHAT_MEDIA_NAME_RE.fullmatch(str(name or ""))
    if not match:
        return None
    from ouroboros.confined_files import open_confined_file

    try:
        root = pathlib.Path(drive_root).resolve()
        path = task_artifact_dir_path(root, task_id, create=False) / _CHAT_MEDIA_SUBDIR / name
        handle, _ = open_confined_file(root, path.relative_to(root).parts)
        handle.close()
    except (OSError, ValueError):
        return None
    return path


# The artifact-store subdir delegated-run captures live in (the naming SSOT
# `delegate_custody.delegated_capture_dir` builds on this dir).
DELEGATED_CAPTURE_PREFIX = "delegated_runs"


def delegated_capture_read_target(
    canonical_root: Any, task_id: str, rel_text: str, resolved_base: pathlib.Path,
) -> Optional[pathlib.Path]:
    """Canonical-drive anchor for READS of own delegated captures and activity.

    Writers retain captures and activity under ``delegate_custody.custody_root``
    (the canonical budget drive) so they survive child-drive pruning. The child's
    ``artifact_store`` otherwise resolves on its execution drive and misses them.
    Only these two prefixes in the caller's OWN task store re-anchor here; other
    paths and already-canonical bases return None. Writes keep their original base.

    Another task's capture requires ``delegate_shared.orphan_capture_read_target``
    and its ``orphan_disposition_status`` proof; this binding grants no orphan access.
    """
    activity_prefix = f"{_SOURCE_HANDLES_SUBDIR}/delegated_activity"
    prefix = activity_prefix if rel_text == activity_prefix or rel_text.startswith(activity_prefix + "/") else DELEGATED_CAPTURE_PREFIX
    if rel_text != prefix and not rel_text.startswith(prefix + "/"):
        return None
    canonical_base = task_artifact_dir_path(
        canonical_root, task_id, create=False,
    ).resolve(strict=False)
    if canonical_base == pathlib.Path(resolved_base):
        return None
    anchored = (canonical_base / rel_text).resolve(strict=False)
    # The new source binding grants only this subtree, including after symlink
    # resolution; it cannot expose sibling source categories on the canonical drive.
    allowed_base = canonical_base / prefix if prefix == activity_prefix else canonical_base
    try:
        anchored.relative_to(allowed_base)
    except ValueError as exc:
        raise ValueError(f"path escapes {allowed_base}") from exc
    return anchored


def task_id_for_artifacts(ctx: Any) -> str:
    """Return a stable task id for artifact storage."""

    metadata = getattr(ctx, "task_metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    for value in (getattr(ctx, "task_id", None), metadata.get("task_id"), metadata.get("id")):
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


class ArtifactIdentityError(OSError):
    """Stable captured identity mismatch after a complete, unchanged file read."""
    def __init__(self, message: str, *, source_path: Any, source_stamp: tuple):
        super().__init__(message)
        self.source_path, self.source_stamp = source_path, source_stamp


def stream_artifact_file(path: Any, sink: Any = None, *, expected: Any = None) -> Dict[str, Any]:
    """Hash/copy regular bytes in bounded chunks; prove descriptor/path stability
    through EOF and ``expected`` identity before publication. Missing/changing bytes
    fail. Borrowed binary regular handles (completed spools included) read from zero,
    stay open and prove only their descriptor; network/pipe streams are not inputs.
    """
    source_path = pathlib.Path(path) if isinstance(path, (str, os.PathLike)) else None
    digest, size = sha256(), 0
    def identity(observation: os.stat_result) -> tuple:
        return (observation.st_dev, observation.st_ino, observation.st_size,
                observation.st_mtime_ns, observation.st_ctime_ns)
    from ouroboros.store import open_for_read  # a held store file keeps its locks (no new descriptor to close)

    # Reject pipes/devices before opening: the descriptor check alone comes too
    # late for a FIFO with no writer. Keep the post-open checks for path changes.
    if source_path is not None and not stat.S_ISREG(source_path.stat().st_mode):
        raise OSError(f"artifact source is not a regular file: {path}")
    with open_for_read(source_path) if source_path is not None else nullcontext(path) as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise OSError(f"artifact source is not a regular file: {path}")
        if not isinstance(handle.read(0), bytes):
            raise OSError("artifact source must be a binary file")
        handle.seek(0)
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            if sink is not None:
                sink.write(chunk)
            digest.update(chunk)
            size += len(chunk)
            if size > before.st_size:
                raise OSError(f"artifact source size changed during read: {path}")
        if (identity(before) != identity(os.fstat(handle.fileno()))
                or (source_path is not None and identity(before) != identity(source_path.stat()))):
            raise OSError(f"artifact source changed during read: {path}")
    result = {"size": size, "sha256": digest.hexdigest()}
    if size != before.st_size:
        raise OSError(f"artifact source size changed during read: {path}")
    expected = expected if isinstance(expected, dict) else {}
    if expected.get("immutable") and (not isinstance(expected.get("size"), int) or not expected.get("sha256")):
        raise ArtifactIdentityError(f"immutable artifact is missing its capture identity: {path}",
                                    source_path=source_path, source_stamp=identity(before))
    for key in ("size", "sha256"):
        if expected.get(key) not in (None, "") and expected[key] != result[key]:
            raise ArtifactIdentityError(f"artifact source failed {key} verification: {path}",
                                        source_path=source_path, source_stamp=identity(before))
    return result


def copy_artifact_file(source: Any, destination: pathlib.Path, *, expected: Any = None) -> Dict[str, Any]:
    """Publish a verified file copy atomically; preserve any prior bytes on failure. Inside a
    ``task_custody.publication_fence`` a closed generation starts no copy. A block-sharing clone
    (``artifact_storage``) is taken where the filesystem gives one, its own bytes verified; else streamed."""
    source_path, destination = (pathlib.Path(source) if isinstance(source, (str, os.PathLike)) else None), pathlib.Path(destination)
    if source_path is not None and not destination.is_symlink() and source_path.resolve(strict=False) == destination.resolve(strict=False):
        return stream_artifact_file(source, expected=expected)
    fence_publication()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_sibling(destination)  # the atomic signature the temp sweep reaps after a hard kill
    from ouroboros.store import is_sqlite_file, is_wal_index  # a store file streams: a clone opens and closes it

    try:
        if (source_path is not None and not is_sqlite_file(source_path) and not is_wal_index(source_path)
                and clone_file(source_path, temporary)):
            try:
                measured = stream_artifact_file(temporary, expected=expected)
            except ArtifactIdentityError:
                # Retention watches the source's stable failure, never a disposable clone's inode.
                stream_artifact_file(source_path, expected=expected)
                raise
        else:
            with temporary.open("xb") as sink:
                measured = stream_artifact_file(source, sink, expected=expected)
        if source_path is not None:
            shutil.copystat(source_path, temporary)
        temporary.replace(destination)
        return measured
    finally:
        temporary.unlink(missing_ok=True)


def artifact_record(path: pathlib.Path, *, kind: str = "task_artifact", source_path: str = "") -> Dict[str, Any]:
    return {"kind": kind, "name": pathlib.Path(path).name, "path": str(path),
            **stream_artifact_file(path), "status": ARTIFACT_STATUS_READY, "errors": [],
            **({"source_path": source_path} if source_path else {})}


def store_task_artifact_bytes(
    drive_root: Union[pathlib.Path, str], task_id: str, name: str, data: bytes,
    *, kind: str = "task_artifact",
) -> Dict[str, Any]:
    """Publish immutable actor-readable bytes; reuse equal bytes, refuse collisions.
    Byte-oriented twin of ``copy_file_to_task_artifacts`` for canonical bytes."""
    safe_name = pathlib.Path(str(name or "")).name
    if not safe_name or safe_name in {".", ".."} or safe_name != str(name):
        raise ValueError("task artifact name must be one plain filename")
    if artifact_store_path_block_reason(pathlib.Path(safe_name)):
        raise ValueError("task artifact name is reserved")
    artifact_dir = task_artifact_dir_path(drive_root, task_id, create=True)
    path = artifact_dir / safe_name
    temporary = temporary_sibling(path)

    def publish(_current):
        try:
            os.link(temporary, path)
        except OSError as exc:
            if (exc.errno not in {errno.EEXIST, errno.EOPNOTSUPP, errno.ENOTSUP, errno.EPERM, errno.EXDEV, errno.ENOSYS}
                    and getattr(exc, "winerror", None) != 50):
                raise
            if not path.exists():
                temporary.replace(path)
            elif path.read_bytes() != data:
                raise ValueError(f"task artifact collision: {safe_name}")
    try:
        write_bytes_atomic(temporary, data)
        update_json_locked(artifact_dir / _ARTIFACT_MANIFEST, publish)
    finally:
        temporary.unlink(missing_ok=True)
    record = {**artifact_record(path, kind=kind), "immutable": True}
    _register_task_artifact_records(artifact_dir, [record])
    return {"root": "artifact_store", "path": safe_name, "sha256": record["sha256"], "bytes": record["size"], "kind": kind}


def _register_task_artifact_records(artifact_dir: pathlib.Path, records: Iterable[Dict[str, Any]]) -> None:
    """Merge only these registrations under the existing manifest-file lock.

    File copying/hashing finishes before this short metadata transaction. No
    task-result lock is acquired here, so a caller already holding one cannot
    invert the publication path's artifact-then-result ordering.
    """
    additions = {pathlib.Path(str(row.get("path") or row.get("name") or "")).name: dict(row)
                 for row in records}

    def merge(current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        previous = current.get("artifacts") if isinstance(current.get("artifacts"), dict) else {}
        merged = merge_artifact_records(previous.values(), additions.values())
        document = {**current, "schema_version": 1, "artifacts": {
            pathlib.Path(str(row.get("path") or row.get("name") or "")).name: row for row in merged
        }}
        return None if document == current else document  # identical registration: no rewrite

    update_json_locked(artifact_dir / _ARTIFACT_MANIFEST, merge)


def registered_task_artifact(drive_root: Any, task_id: str, name: str) -> Optional[Dict[str, Any]]:
    """Read one host registration, including an output published before task finality."""
    from ouroboros.confined_files import read_confined_json

    root = pathlib.Path(drive_root).resolve()
    path = task_artifact_dir_path(root, task_id) / _ARTIFACT_MANIFEST
    manifest = (read_confined_json(root, path.relative_to(root).parts) or {}).get("artifacts")
    row = manifest.get(name) if isinstance(manifest, dict) else None
    return dict(row) if isinstance(row, dict) else None


def _archive_previous_artifact_version(drive_root: pathlib.Path, task_id: str, dest: pathlib.Path, source: pathlib.Path) -> None:
    if not dest.is_file() or not source.is_file():
        return
    previous = stream_artifact_file(dest)
    if previous == stream_artifact_file(source):
        return
    safe_name = pathlib.Path(dest.name).name.replace("/", "_").replace("\\", "_")
    if not safe_name or safe_name in {".", ".."}:
        safe_name = "artifact"
    version_dir = pathlib.Path(drive_root) / "task_results" / _ARTIFACT_VERSIONS_DIR / validate_task_id(task_id) / safe_name
    version_dir.mkdir(parents=True, exist_ok=True)
    suffix = dest.suffix
    stem = dest.name[: -len(suffix)] if suffix else dest.name
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    version_path = version_dir / f"{stamp}.{previous['sha256'][:12]}.{stem}{suffix}"
    copy_artifact_file(dest, version_path, expected=previous)
    versions = sorted((p for p in version_dir.iterdir() if p.is_file()), key=lambda p: p.name)
    for stale in versions[:-_ARTIFACT_VERSION_RETENTION]:
        fence_publication()  # a closed generation deletes no retained version, even after its backup landed
        try:
            stale.unlink()
        except OSError:
            continue


def copy_file_to_task_artifacts(ctx: Any, source_path: Union[pathlib.Path, str], *, kind: str = "user_file", immutable: bool = False, expected: Any = None) -> Dict[str, Any] | None:
    """Copy a generated file, preserving an immutable rebase's original identity."""

    source = pathlib.Path(source_path).expanduser().resolve(strict=False)
    if not source.is_file() and not (immutable and isinstance(expected, dict) and expected.get("name")):
        return None
    task_id = task_id_for_artifacts(ctx)
    drive_root = pathlib.Path(getattr(ctx, "drive_root"))
    if is_verification_receipts_path(drive_root, task_id, source):
        return None
    artifact_dir = task_artifact_dir_path(drive_root, task_id, create=True)
    data = read_json_dict(artifact_dir / _ARTIFACT_MANIFEST) or {}
    manifest = data.get("artifacts") if isinstance(data.get("artifacts"), dict) else {}
    manifest = {str(key): dict(value) for key, value in manifest.items() if isinstance(value, dict)}
    captured = manifest.get(source.name) if source.parent == artifact_dir.resolve(strict=False) else None
    if captured and captured.get("immutable"):
        immutable, expected = True, expected or captured
    dest = artifact_dir / source.name
    reused_existing_source = False
    source_identity = expected
    if immutable and isinstance(expected, dict) and expected.get("name"):
        name = str(expected["name"])
        if pathlib.Path(name).name != name or name in {".", ".."}:
            raise ValueError("immutable artifact name must be a plain filename")
        dest = artifact_dir / name
    elif immutable:
        source_identity = stream_artifact_file(source, expected=expected)
        stem = _safe_attachment_name(source.stem).removesuffix("-" + source_identity["sha256"])
        stem = stem.encode("utf-8")[:120].decode("utf-8", errors="ignore")
        suffix = source.suffix.encode("utf-8")[:20].decode("utf-8", errors="ignore")
        dest = artifact_dir / f"{stem}-{source_identity['sha256']}{suffix}"
    for existing in ([] if immutable else manifest.values()):
        if existing.get("immutable"):
            continue
        existing_source = str(existing.get("source_path") or "")
        existing_path = str(existing.get("path") or "")
        if existing_source == str(source) and existing_path:
            candidate = pathlib.Path(existing_path).resolve(strict=False)
            if (
                candidate.parent == artifact_dir.resolve(strict=False)
                and not is_verification_receipts_path(drive_root, task_id, candidate)
            ):
                dest = candidate
                reused_existing_source = True
                break
    if (
        is_verification_receipts_path(drive_root, task_id, dest)
        or (
            dest.exists()
            and dest.resolve(strict=False) != source.resolve(strict=False)
            and not reused_existing_source and not immutable
        )
    ):
        suffix = source.suffix
        stem = source.name[: -len(suffix)] if suffix else source.name
        digest = sha256(str(source.resolve(strict=False)).encode("utf-8", errors="replace")).hexdigest()[:8]
        dest = artifact_dir / f"{stem}.{digest}{suffix}"
    if kind == "user_file" and reused_existing_source and dest.resolve(strict=False) != source.resolve(strict=False):
        _archive_previous_artifact_version(pathlib.Path(getattr(ctx, "drive_root")), task_id, dest, source)
    if immutable and dest.exists() and not dest.is_symlink() and (not source.exists() or not dest.samefile(source)):
        measured = stream_artifact_file(dest, expected=source_identity)
    elif dest.is_symlink() or dest.resolve(strict=False) != source.resolve(strict=False):
        measured = copy_artifact_file(source, dest, expected=source_identity)
    else:
        measured = stream_artifact_file(dest, expected=source_identity)
    record = {"kind": kind, "name": dest.name, "path": str(dest), **measured,
              "status": ARTIFACT_STATUS_READY, "errors": [], "source_path": str(source),
              **({"immutable": True} if immutable else {})}
    _register_task_artifact_records(artifact_dir, [record])
    return record


def collect_task_artifact_records(
    drive_root: Union[pathlib.Path, str], task_id: str, *, measure: bool = True, strict: bool = False,
    require_registered: bool = False, anchor: Any = None,
) -> List[Dict[str, Any]]:
    """List deliverables by handle, excluding metadata, inputs and SQLite WAL indexes (refused unread:
    ``ConfinedDirectory.open_for_read``). Unmeasured rows keep registered identities; measurement
    hashes the open file. Strict reads raise on unreadable files/metadata; ``anchor`` confines child stores."""
    from ouroboros.confined_files import confined_directory, read_confined_json
    from ouroboros.outcome_receipt_store import verification_receipts_path
    from ouroboros.store import WalIndexRefused

    try:
        store = task_artifact_dir_path(drive_root, validate_task_id(task_id), create=False)
    except ValueError:
        return []
    root = pathlib.Path(anchor if anchor is not None else drive_root).absolute()
    parts = store.absolute().relative_to(root).parts
    receipts = verification_receipts_path(drive_root, task_id, create=False)
    document = read_confined_json(root, (*parts, _ARTIFACT_MANIFEST), strict=strict) or {}
    entries = document.get("artifacts") or {}
    manifest = {str(k): v for k, v in entries.items() if isinstance(v, dict)} if isinstance(entries, dict) else {}
    rows, opened = [], False

    def walk(directory, relative=()):
        for name, is_directory in sorted(directory.entries()):
            path = store.joinpath(*relative, name)
            if (name in {_ARTIFACT_MANIFEST, SCRATCH_MANIFEST_NAME}
                    or path in {receipts, store / (_ARTIFACT_MANIFEST + ".lock")}
                    or not relative and name in {_ATTACHMENTS_SUBDIR, _CHAT_MEDIA_SUBDIR, _SOURCE_HANDLES_SUBDIR}):
                continue
            try:
                if is_directory:
                    with directory.child(name) as child:
                        walk(child, (*relative, name))
                    continue
                handle, observed = directory.open_for_read(name)
                registered = manifest.get(name) if not relative else None
                nested = {"relpath": "/".join((*relative, name))} if relative else {}
                with handle:
                    if measure:
                        row = {"kind": "task_artifact", "name": name, "path": str(path), **nested,
                               **stream_artifact_file(handle), "status": ARTIFACT_STATUS_READY, "errors": []}
                        if registered:
                            row = merge_artifact_records([{**registered, "path": str(path)}], [row])[0]
                    else:
                        registered = registered or {}
                        row = {"kind": str(registered.get("kind") or "task_artifact"), "name": name,
                               "path": str(path), **nested, "size": observed.st_size, "measured": False,
                               **({key: registered.get(key) for key in ("immutable", "size", "sha256")}
                                  if registered.get("immutable") else {})}
                rows.append(row)
            except OSError as exc:  # a WAL index, refused unread, is left out as copies and settlement leave it
                if strict and not isinstance(exc, WalIndexRefused):
                    raise

    try:
        with confined_directory(root, parts) as directory:
            opened = True
            if require_registered:
                for name in manifest:
                    with suppress(WalIndexRefused):  # a registered index is present, and never read
                        directory.open_for_read(name)[0].close()
            walk(directory)
    except FileNotFoundError:
        if opened and strict:
            raise
    except OSError:
        if strict:
            raise
    return sorted(rows, key=lambda row: row["path"])


def merge_artifact_records(*groups: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for group in groups:
        for item in group:
            if not isinstance(item, dict):
                continue
            key = str(item.get("path") or item.get("name") or "")
            if not key:
                continue
            if key not in merged:
                merged[key] = dict(item)
                continue
            existing = merged[key]
            fresh = dict(item)
            merged[key] = {**existing, **fresh}
            if existing.get("immutable"):
                changed = any(fresh.get(field) not in (None, "") and fresh[field] != existing.get(field)
                              for field in ("size", "sha256"))
                merged[key].update({field: existing.get(field) for field in ("immutable", "size", "sha256")})
                if changed:
                    merged[key].update(status=ARTIFACT_STATUS_FAILED, errors=["immutable artifact bytes changed after capture"])
            if existing.get("kind") and fresh.get("kind") == "task_artifact" and existing.get("kind") != "task_artifact":
                merged[key]["kind"] = existing["kind"]
            for meta_key in ("kind", "source_path", "name"):
                if existing.get(meta_key) and not fresh.get(meta_key):
                    merged[key][meta_key] = existing[meta_key]
            if existing.get("name") and fresh.get("kind") == "task_artifact":
                merged[key]["name"] = existing["name"]
    return list(merged.values())


# Public artifact API; directory packaging owns traversal and virtual member names.
from ouroboros.artifact_directories import iter_artifact_tree, copy_directory_to_task_artifacts  # noqa: E402,F401
