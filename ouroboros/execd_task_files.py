"""Execd-side task-file contract: canonical attachment manifest + private cache.

Extracted verbatim from the donor's ``remote_task_files.py`` so that the execd
import closure carries only the attachment CONTRACT (constants, canonical
manifest/blob validation, the typed error) and the execd-owned
``RemoteTaskFileCache`` — never the Home admission/import half (staging RPCs,
envelope validation, media-cache cleanup), which imports the Home broker.  The
cache is execution transport, never workspace content or durable Home memory.
Execd owns its paths and accepts only the canonical attachment manifest plus
content-addressed blobs; callers cannot nominate a remote destination.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import pathlib
import re
import shutil
import stat
import tempfile
from collections.abc import Mapping
from typing import Any

from ouroboros.remote_contracts import refuse_unknown_members
from ouroboros.remote_protocol import canonical_json

ATTACHMENT_STAGE_OPERATION = "_stage_task_attachments"
MEDIA_EXPORT_OPERATION = "_export_task_media"
INTERNAL_TASK_FILE_OPERATIONS = frozenset(
    {ATTACHMENT_STAGE_OPERATION, MEDIA_EXPORT_OPERATION}
)

# The export channel a media import is judged under. Named here rather than derived
# because `_export_task_media` is an INTERNAL operation and is therefore absent from
# `remote_export_policy.OPERATION_EXPORT_CHANNEL` (that table lives on Home and maps
# the MODEL-facing operations); an internal door with no channel would be an unpoliced
# one, which is exactly what the closed registry exists to prevent.
MEDIA_EXPORT_CHANNEL = "media_frames"

# The CLOSED field set of one staged attachment row — the same nine keys the Home
# mirror declares as `remote_task_files._WIRE_FIELDS`. It lives here as a set rather
# than only as the shape of the projection below, because a projection can only DROP a
# field it does not know, and dropping is the one outcome this boundary must not have:
# a newer Home that adds a redaction or policy fact to a row would otherwise have its
# attachment staged by a build that ignored exactly the field which changed what
# staging means.
ATTACHMENT_WIRE_FIELDS = frozenset({
    "attachment_id", "label", "root", "relpath", "mime", "is_image", "size",
    "sha256", "stage_status",
})

MAX_ATTACHMENT_COUNT = 25
MAX_ATTACHMENT_BYTES = 50 * 1024 * 1024
MAX_ATTACHMENT_TOTAL_BYTES = 512 * 1024 * 1024
MAX_MEDIA_EXPORT_BYTES = 25 * 1024 * 1024

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_OPAQUE_RE = re.compile(
    r"^[A-Za-z0-9_:@-](?:[A-Za-z0-9_.:@-]{0,254}[A-Za-z0-9_:@-])?$"
)
_SAFE_SUFFIX_RE = re.compile(r"^\.[A-Za-z0-9]{1,16}$")


class RemoteTaskFileError(RuntimeError):
    """Typed internal cache error translated at the execd boundary."""

    def __init__(self, code: str, message: str) -> None:
        self.code = str(code)
        super().__init__(str(message))

def _opaque(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not _OPAQUE_RE.fullmatch(text):
        raise RemoteTaskFileError(
            "attachment_manifest_invalid",
            f"{field} must be a file-safe opaque ID.",
        )
    return text


def _safe_label(value: Any) -> str:
    label = " ".join(
        "".join(character for character in str(value or "") if character.isprintable()).split()
    )[:120]
    if not label:
        raise RemoteTaskFileError(
            "attachment_manifest_invalid",
            "Attachment label is empty.",
        )
    return label


def _safe_relpath(value: Any) -> str:
    text = str(value or "").replace("\\", "/").strip()
    path = pathlib.PurePosixPath(text)
    if (
        path.is_absolute()
        or len(path.parts) != 2
        or path.parts[0] != "attachments"
        or path.parts[1] in {"", ".", ".."}
    ):
        raise RemoteTaskFileError(
            "attachment_manifest_invalid",
            "Attachment relpath must name one canonical artifact-store attachment.",
        )
    return path.as_posix()


def canonical_attachment_manifest(value: Any) -> list[dict[str, Any]]:
    """Validate and copy the Home-authoritative ready attachment set."""

    if not isinstance(value, list) or len(value) > MAX_ATTACHMENT_COUNT:
        raise RemoteTaskFileError(
            "attachment_manifest_invalid",
            "Attachment manifest count exceeds the admission limit.",
        )
    result: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    total = 0
    for raw in value:
        if not isinstance(raw, Mapping):
            raise RemoteTaskFileError(
                "attachment_manifest_invalid",
                "Attachment manifest entries must be objects.",
            )
        unknown = sorted(set(map(str, raw.keys())) - ATTACHMENT_WIRE_FIELDS)
        if unknown:
            refuse_unknown_members(
                "attachment_stage",
                unknown=unknown,
                understood=ATTACHMENT_WIRE_FIELDS,
                member="attachment fields",
            )
        attachment_id = _opaque(raw.get("attachment_id"), "attachment_id")
        if attachment_id in seen_ids:
            raise RemoteTaskFileError(
                "attachment_manifest_invalid",
                "Attachment IDs must be unique.",
            )
        seen_ids.add(attachment_id)
        digest = str(raw.get("sha256") or "")
        if not _HASH_RE.fullmatch(digest):
            raise RemoteTaskFileError(
                "attachment_manifest_invalid",
                "Attachment SHA-256 is invalid.",
            )
        size = raw.get("size")
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or size > MAX_ATTACHMENT_BYTES
        ):
            raise RemoteTaskFileError(
                "attachment_manifest_invalid",
                "Attachment size exceeds the admission limit.",
            )
        total += size
        if total > MAX_ATTACHMENT_TOTAL_BYTES:
            raise RemoteTaskFileError(
                "attachment_manifest_invalid",
                "Attachment set exceeds the aggregate admission limit.",
            )
        mime = str(raw.get("mime") or "application/octet-stream").strip()[:255]
        if not mime or any(character.isspace() for character in mime):
            raise RemoteTaskFileError(
                "attachment_manifest_invalid",
                "Attachment MIME is invalid.",
            )
        is_image = raw.get("is_image")
        if not isinstance(is_image, bool) or is_image != mime.startswith("image/"):
            raise RemoteTaskFileError(
                "attachment_manifest_invalid",
                "Attachment image fact does not match its MIME.",
            )
        if str(raw.get("root") or "") != "artifact_store":
            raise RemoteTaskFileError(
                "attachment_manifest_invalid",
                "Attachment root must remain artifact_store.",
            )
        if str(raw.get("stage_status") or "") != "ready":
            raise RemoteTaskFileError(
                "attachment_manifest_invalid",
                "Only ready Home-staged attachments may be admitted.",
            )
        result.append(
            {
                "attachment_id": attachment_id,
                "label": _safe_label(raw.get("label")),
                "root": "artifact_store",
                "relpath": _safe_relpath(raw.get("relpath")),
                "mime": mime,
                "is_image": is_image,
                "size": size,
                "sha256": digest,
                "stage_status": "ready",
            }
        )
    return result


def attachment_blob_map(
    manifest: Any,
    blobs: Mapping[str, bytes],
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    """Require one exact content-addressed blob for every authoritative entry."""

    canonical = canonical_attachment_manifest(manifest)
    required = {entry["sha256"] for entry in canonical}
    if set(str(key) for key in blobs) != required:
        raise RemoteTaskFileError(
            "attachment_blob_set_mismatch",
            "Remote attachment upload must contain every and only authoritative blob.",
        )
    verified: dict[str, bytes] = {}
    for digest in required:
        payload = bytes(blobs[digest])
        if hashlib.sha256(payload).hexdigest() != digest:
            raise RemoteTaskFileError(
                "attachment_hash_mismatch",
                "Remote attachment upload failed SHA-256 verification.",
            )
        expected_sizes = {
            int(entry["size"]) for entry in canonical if entry["sha256"] == digest
        }
        if expected_sizes != {len(payload)}:
            raise RemoteTaskFileError(
                "attachment_size_mismatch",
                "Remote attachment upload failed exact-size verification.",
            )
        verified[digest] = payload
    return canonical, verified


def media_export_policy_facts(args: Mapping[str, Any]) -> dict[str, Any]:
    """The policy a media export was PREPARED under, as facts, or ``{}``.

    Kept as facts so the post-authorization revalidation compares the same document the
    first prepare applied, and so the hash Home checks is the hash of the rules that
    actually ran rather than one recomputed later.
    """

    from ouroboros.export_policy_contract import export_policy_hash, normalize_export_policy

    raw = args.get("_export_policy")
    if raw is None:
        return {}
    document = normalize_export_policy(raw)
    return {"export_policy": document, "export_policy_hash": export_policy_hash(document)}


def media_export_execution_args(
    args: Mapping[str, Any], facts: Mapping[str, Any]
) -> dict[str, Any]:
    """The target-canonical arguments of one media export.

    Built from the RESOLVED facts rather than echoed from the request, so the token
    binds the source the target actually opened — a request naming an absolute path and
    the resolution naming a workspace-relative one are the same file, and only the
    second one is checkable."""

    max_bytes = int(args.get("max_bytes") or 0)
    if str(args.get("attachment_id") or ""):
        return {"attachment_id": str(facts["attachment_id"]), "max_bytes": max_bytes}
    return {"path": str(facts["relative_path"]), "max_bytes": max_bytes}


def media_export_artifact_row(facts: Mapping[str, Any]) -> dict[str, Any]:
    """Declare one exported media file as the DECLARED OUTPUT it actually is.

    Home named exactly one path and these are that path's bytes, so declaring it as a
    declared output is not a convenience — it is what makes the transport prefetch and
    verify the blob and the transfer service publish it through the artifact authority.
    An artifact row carrying a blob id and no declared kind was fetched by nobody, so
    the import produced a manifest describing bytes that never moved.

    ``mime`` is the wire-level ``application/octet-stream`` the declared-output ref
    contract requires; the REAL media type travels in the trace, where Home reads it.
    """

    return {
        **dict(facts),
        "blob_id": str(facts["sha256"]),
        "kind": "declared_output",
        "mime": "application/octet-stream",
        "declared_as": str(facts.get("relative_path") or facts.get("attachment_id") or ""),
        "member_path": "",
    }


class RemoteTaskFileCache:
    """Execd-owned, private cache for one connection/server generation."""

    def __init__(
        self,
        state_root: pathlib.Path,
        *,
        connection_id: str,
        server_generation: str,
    ) -> None:
        connection = _opaque(connection_id, "connection_id")
        generation = _opaque(server_generation, "server_generation")
        self.connection_root = (
            pathlib.Path(state_root) / "task_files" / connection
        )
        self.generation_root = self.connection_root / generation
        self.generation_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.connection_root.parent, 0o700)
        os.chmod(self.connection_root, 0o700)
        os.chmod(self.generation_root, 0o700)
        self._prune_stale_generations(generation)

    def stage_attachments(
        self,
        task_id: str,
        manifest: Any,
        blobs: Mapping[str, bytes],
    ) -> list[dict[str, Any]]:
        """Atomically publish the entire verified task attachment set."""

        task = _opaque(task_id, "task_id")
        canonical, verified = attachment_blob_map(manifest, blobs)
        task_root = self.generation_root / task
        expected_identity = hashlib.sha256(canonical_json(canonical)).hexdigest()
        existing = self._existing_manifest(task_root, expected_identity)
        if existing is not None:
            return existing
        if task_root.exists():
            raise RemoteTaskFileError(
                "attachment_task_cache_conflict",
                "Task attachment cache already contains a different manifest.",
            )
        temporary = pathlib.Path(
            tempfile.mkdtemp(prefix=f".{task}.", dir=str(self.generation_root))
        )
        try:
            os.chmod(temporary, 0o700)
            published: list[dict[str, Any]] = []
            for entry in canonical:
                digest = entry["sha256"]
                suffix = pathlib.PurePosixPath(entry["relpath"]).suffix.lower()
                safe_suffix = suffix if _SAFE_SUFFIX_RE.fullmatch(suffix) else ""
                target = temporary / f"{digest}{safe_suffix}"
                if not target.exists():
                    self._write_private_file(target, verified[digest])
                remote = {
                    **entry,
                    "execution_path": str(target),
                    "abs_path": str(target),
                }
                published.append(remote)
            self._write_private_file(
                temporary / "manifest.json",
                canonical_json(
                    {
                        "_schema_version": 1,
                        "identity_sha256": expected_identity,
                        "attachments": published,
                    }
                ),
            )
            os.replace(temporary, task_root)
            self._fsync_directory(self.generation_root)
            # The temp absolute prefix changed after rename; publish canonical
            # paths from the final execd-owned root.
            return self._existing_manifest(task_root, expected_identity) or []
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def cleanup_task(self, task_id: str) -> bool:
        task = _opaque(task_id, "task_id")
        target = self.generation_root / task
        if not target.exists():
            return False
        shutil.rmtree(target)
        self._fsync_directory(self.generation_root)
        return True

    def export_media(
        self,
        workspace_root: pathlib.Path,
        args: Mapping[str, Any],
        *,
        task_id: str,
        expected_sha256: str = "",
        expected_size: int | None = None,
    ) -> tuple[dict[str, Any], bytes]:
        """ONE door for both media sources, with the policy applied before the read.

        A workspace file is judged by the export policy the operation was prepared
        under, exactly as every other export door judges its paths, and a single named
        source the policy excludes REFUSES rather than discloses: once the one source
        is out there is nothing left to deliver, and an empty success would read as
        "the file had no content".

        A task-cache ATTACHMENT skips that judgement on purpose — Home already filtered
        that set on the way out, and re-judging a file Home itself admitted would refuse
        the owner's own input on its way back to the model that was given it.

        Both branches go through one door so prepare and execute cannot resolve the
        source differently; the caller's only difference is whether it passes the
        expected hash/size, which turns the second read into a change detector.
        """

        max_bytes = int(args.get("max_bytes") or 0)
        attachment_id = str(args.get("attachment_id") or "")
        if attachment_id:
            return self.export_task_attachment(
                task_id,
                attachment_id,
                max_bytes=max_bytes,
                expected_sha256=expected_sha256,
                expected_size=expected_size,
            )
        return self.export_workspace_file(
            workspace_root,
            str(args.get("path") or ""),
            max_bytes=max_bytes,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
            # The judgement moves INSIDE the reader, where the RESOLVED target exists.
            # Asked out here on the requested spelling it could not see that `clip.mp4`
            # was a link to `.env`; the reader then resolved that link and read it.
            policy_facts=media_export_policy_facts(args),
        )

    def export_workspace_file(
        self,
        workspace_root: pathlib.Path,
        relative_path: str,
        *,
        max_bytes: int,
        expected_sha256: str = "",
        expected_size: int | None = None,
        policy_facts: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], bytes]:
        """Read one symlink-confined regular workspace file with exact facts.

        ``policy_facts`` is the bound export document, applied here rather than by the
        caller: this method RESOLVES the path and then reads it, so it is the only place
        the policy can be asked about the file that will really be opened. ``None`` is
        the attachment branch, which Home already filtered on the way out.
        """

        from ouroboros.export_policy_contract import QUESTION_EXPORT
        from ouroboros.workspace_native_paths import open_confined_source

        root = pathlib.Path(workspace_root).resolve(strict=True)
        relative = str(relative_path or "").replace("\\", "/").strip()
        pure = pathlib.PurePosixPath(relative)
        if not relative or pure.is_absolute() or any(
            part in {"", ".", ".."} for part in pure.parts
        ):
            raise RemoteTaskFileError(
                "remote_media_path_invalid",
                "Remote media path must be a non-traversing workspace-relative file.",
            )
        limit = int(max_bytes)
        if limit <= 0 or limit > MAX_MEDIA_EXPORT_BYTES:
            raise RemoteTaskFileError(
                "remote_media_limit_invalid",
                "Remote media import limit is invalid.",
            )
        # ONE door, and it hands back a DESCRIPTOR. The old shape resolved the path,
        # judged the path, then stat'ed and read the path again — so a reviewer replaced
        # `frame.png` with a symlink to `.env` between the check and the read and this
        # method returned `b'SECRET_TOKEN=hunter2\n'` labelled `mime: image/png`. The fd
        # is opened `O_NOFOLLOW` and the policy is applied to its `fstat`, so the identity
        # judged is the identity read, and nothing the NAME comes to mean afterwards can
        # change the bytes.
        try:
            target, handle = open_confined_source(
                root,
                pure.as_posix(),
                facts=policy_facts,
                question=QUESTION_EXPORT if policy_facts is not None else "none",
                channel=MEDIA_EXPORT_CHANNEL if policy_facts is not None else "",
            )
        except PermissionError:
            raise
        except (OSError, ValueError) as exc:
            raise RemoteTaskFileError(
                "remote_media_path_escape",
                "Remote media path escapes the admitted workspace.",
            ) from exc
        try:
            info = os.fstat(handle)
            if not stat.S_ISREG(info.st_mode):
                raise RemoteTaskFileError(
                    "remote_media_not_file",
                    "Remote media source is not a regular file.",
                )
            size = info.st_size
            if size > limit:
                raise RemoteTaskFileError(
                    "remote_media_too_large",
                    "Remote media source exceeds the Home import limit.",
                )
            payload = os.read(handle, size + 1)
        finally:
            os.close(handle)
        digest = hashlib.sha256(payload).hexdigest()
        if len(payload) != size:
            raise RemoteTaskFileError(
                "remote_media_changed",
                "Remote media source changed while it was imported.",
            )
        if expected_size is not None and size != int(expected_size):
            raise RemoteTaskFileError(
                "remote_media_changed",
                "Remote media source size changed after preparation.",
            )
        if expected_sha256 and digest != expected_sha256:
            raise RemoteTaskFileError(
                "remote_media_changed",
                "Remote media source hash changed after preparation.",
            )
        return (
            {
                "relative_path": pure.as_posix(),
                "size": size,
                "sha256": digest,
                "mime": mimetypes.guess_type(target.name)[0]
                or "application/octet-stream",
                "name": target.name,
            },
            payload,
        )

    def export_task_attachment(
        self,
        task_id: str,
        attachment_id: str,
        *,
        max_bytes: int,
        expected_sha256: str = "",
        expected_size: int | None = None,
    ) -> tuple[dict[str, Any], bytes]:
        """Read one exact manifest-bound attachment from the current task cache."""

        task = _opaque(task_id, "task_id")
        wanted = _opaque(attachment_id, "attachment_id")
        manifest_path = self.generation_root / task / "manifest.json"
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RemoteTaskFileError(
                "attachment_task_cache_unavailable",
                "Remote task attachment cache is unavailable.",
            ) from exc
        entries = raw.get("attachments") if isinstance(raw, dict) else None
        entry = next(
            (
                item
                for item in entries
                if isinstance(item, dict)
                and str(item.get("attachment_id") or "") == wanted
            ),
            None,
        ) if isinstance(entries, list) else None
        if entry is None:
            raise RemoteTaskFileError(
                "attachment_not_found",
                "Attachment is not present in the current task manifest.",
            )
        size = int(entry.get("size") or 0)
        limit = int(max_bytes)
        if limit <= 0 or limit > MAX_MEDIA_EXPORT_BYTES or size > limit:
            raise RemoteTaskFileError(
                "remote_media_too_large",
                "Remote attachment exceeds the Home media import limit.",
            )
        digest = str(entry.get("sha256") or "")
        suffix = pathlib.PurePosixPath(str(entry.get("relpath") or "")).suffix.lower()
        safe_suffix = suffix if _SAFE_SUFFIX_RE.fullmatch(suffix) else ""
        target = self.generation_root / task / f"{digest}{safe_suffix}"
        try:
            payload = target.read_bytes()
        except OSError as exc:
            raise RemoteTaskFileError(
                "attachment_task_cache_unavailable",
                "Remote task attachment blob is unavailable.",
            ) from exc
        observed = hashlib.sha256(payload).hexdigest()
        if (
            len(payload) != size
            or observed != digest
            or (expected_size is not None and len(payload) != int(expected_size))
            or (expected_sha256 and observed != expected_sha256)
        ):
            raise RemoteTaskFileError(
                "attachment_task_cache_corrupt",
                "Remote task attachment failed exact size/hash verification.",
            )
        return (
            {
                "attachment_id": wanted,
                "size": size,
                "sha256": digest,
                "mime": str(entry.get("mime") or "application/octet-stream"),
                "name": pathlib.PurePosixPath(
                    str(entry.get("relpath") or "")
                ).name,
            },
            payload,
        )

    def _existing_manifest(
        self,
        task_root: pathlib.Path,
        identity: str,
    ) -> list[dict[str, Any]] | None:
        path = task_root / "manifest.json"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RemoteTaskFileError(
                "attachment_task_cache_corrupt",
                "Existing remote task attachment cache is corrupt.",
            ) from exc
        if (
            not isinstance(raw, dict)
            or raw.get("identity_sha256") != identity
            or not isinstance(raw.get("attachments"), list)
        ):
            return None
        result: list[dict[str, Any]] = []
        for entry in raw["attachments"]:
            if not isinstance(entry, dict):
                raise RemoteTaskFileError(
                    "attachment_task_cache_corrupt",
                    "Existing remote task attachment manifest is corrupt.",
                )
            digest = str(entry.get("sha256") or "")
            suffix = pathlib.PurePosixPath(str(entry.get("relpath") or "")).suffix.lower()
            safe_suffix = suffix if _SAFE_SUFFIX_RE.fullmatch(suffix) else ""
            target = task_root / f"{digest}{safe_suffix}"
            try:
                payload = target.read_bytes()
            except OSError as exc:
                raise RemoteTaskFileError(
                    "attachment_task_cache_corrupt",
                    "Existing remote task attachment blob is unavailable.",
                ) from exc
            if (
                len(payload) != int(entry.get("size") or 0)
                or hashlib.sha256(payload).hexdigest() != digest
            ):
                raise RemoteTaskFileError(
                    "attachment_task_cache_corrupt",
                    "Existing remote task attachment blob failed verification.",
                )
            result.append(
                {
                    **entry,
                    "execution_path": str(target),
                    "abs_path": str(target),
                }
            )
        return result

    def _prune_stale_generations(self, current: str) -> None:
        for child in self.connection_root.iterdir():
            if child.name == current or not child.is_dir() or child.is_symlink():
                continue
            shutil.rmtree(child, ignore_errors=True)

    @staticmethod
    def _write_private_file(path: pathlib.Path, payload: bytes) -> None:
        descriptor = os.open(
            str(path),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(path, 0o600)

    @staticmethod
    def _fsync_directory(path: pathlib.Path) -> None:
        descriptor = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
