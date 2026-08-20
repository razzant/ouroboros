"""Immutable byte capture for one reviewed skill publication attempt."""

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass
from typing import Any

from ouroboros.contracts.skill_manifest import (
    SkillManifestError,
    parse_skill_manifest_text,
)
from ouroboros.contracts.skill_payload_policy import (
    SKILL_PAYLOAD_CONTROL_FILENAMES,
)
from ouroboros.skill_loader import (
    _MANIFEST_NAMES,
    LoadedSkill,
    SkillPayloadUnreadable,
    _iter_payload_files,
    reduce_skill_content_hash,
)

MAX_PUBLIC_PAYLOAD_BYTES = 5 * 1024 * 1024


class SkillPublishSnapshotError(RuntimeError):
    """Closed, payload-free failure raised while capturing reviewed bytes."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


@dataclass(frozen=True)
class CapturedSkillFile:
    """One canonical payload path and its exact immutable bytes."""

    path: str
    content: bytes
    byte_count: int
    sha256: str

    @classmethod
    def from_bytes(cls, path: str, content: bytes) -> "CapturedSkillFile":
        data = bytes(content)
        return cls(
            path=path,
            content=data,
            byte_count=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
        )


@dataclass(frozen=True)
class CapturedPublishManifest:
    """Frozen projection of the manifest fields used by publication."""

    path: str
    name: str
    description: str
    version: str
    skill_type: str
    when_to_use: str
    install_specs_json: str = ""

    def install_specs(self) -> Any:
        """Return a fresh JSON value, never mutable state held by the snapshot."""
        return json.loads(self.install_specs_json) if self.install_specs_json else None


@dataclass(frozen=True)
class SkillPublishSnapshot:
    """Full reviewed view plus the exact public outbound subset."""

    skill: str
    source: str
    manifest_file: CapturedSkillFile
    manifest: CapturedPublishManifest
    content_hash: str
    full_files: tuple[CapturedSkillFile, ...]
    public_files: tuple[CapturedSkillFile, ...]
    control_files: tuple[CapturedSkillFile, ...]

    @property
    def manifest_bytes(self) -> bytes:
        return self.manifest_file.content

    @property
    def public_byte_count(self) -> int:
        return sum(item.byte_count for item in self.public_files)

    def file(self, path: str) -> CapturedSkillFile | None:
        """Return a captured file by canonical path without exposing a live handle."""
        return next((item for item in self.full_files if item.path == path), None)


def _read_captured_file(
    path: pathlib.Path,
    relpath: str,
    *,
    max_bytes: int | None = None,
) -> CapturedSkillFile:
    try:
        with path.open("rb") as stream:
            content = stream.read() if max_bytes is None else stream.read(max_bytes + 1)
    except (OSError, RuntimeError) as exc:
        raise SkillPublishSnapshotError("snapshot_payload_unreadable") from exc
    if max_bytes is not None and len(content) > max_bytes:
        raise SkillPublishSnapshotError("snapshot_payload_too_large")
    return CapturedSkillFile.from_bytes(relpath, content)


def _select_manifest_path(skill_dir: pathlib.Path) -> tuple[pathlib.Path, str]:
    for name in _MANIFEST_NAMES:
        candidate = skill_dir / name
        try:
            if not candidate.is_file():
                continue
            candidate.resolve().relative_to(skill_dir)
        except (OSError, RuntimeError, ValueError) as exc:
            raise SkillPublishSnapshotError("snapshot_manifest_unreadable") from exc
        return candidate, name
    raise SkillPublishSnapshotError("snapshot_manifest_missing")


def _manifest_projection(
    *,
    path: str,
    manifest_text: str,
    fallback_name: str,
) -> tuple[CapturedPublishManifest, Any]:
    try:
        parsed = parse_skill_manifest_text(manifest_text)
    except SkillManifestError as exc:
        raise SkillPublishSnapshotError("snapshot_manifest_invalid") from exc
    if not parsed.name:
        parsed.name = fallback_name

    raw_extra = parsed.raw_extra or {}
    install_specs = raw_extra.get("install_specs") or raw_extra.get("install") or raw_extra.get("dependencies")
    install_specs_json = ""
    if install_specs:
        try:
            install_specs_json = json.dumps(
                install_specs,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise SkillPublishSnapshotError("snapshot_manifest_invalid") from exc

    return (
        CapturedPublishManifest(
            path=path,
            name=str(parsed.name or fallback_name),
            description=str(parsed.description or ""),
            version=str(parsed.version or ""),
            skill_type=str(parsed.type or ""),
            when_to_use=str(parsed.when_to_use or ""),
            install_specs_json=install_specs_json,
        ),
        parsed,
    )


def capture_skill_publish_snapshot(loaded: LoadedSkill) -> SkillPublishSnapshot:
    """Capture and bind the exact reviewed bytes used by a publish transaction.

    The manifest is selected with loader precedence and read once. Its captured
    text drives the existing payload inventory; each remaining file is then read
    once. Only the resulting immutable hash is compared with durable review
    authority, so later live-tree edits cannot change this transaction.
    """
    if loaded.load_error:
        raise SkillPublishSnapshotError("snapshot_skill_unreadable")
    try:
        skill_dir = pathlib.Path(loaded.skill_dir).resolve()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SkillPublishSnapshotError("snapshot_skill_unreadable") from exc

    manifest_path, manifest_rel = _select_manifest_path(skill_dir)
    manifest_file = _read_captured_file(
        manifest_path,
        manifest_rel,
        max_bytes=MAX_PUBLIC_PAYLOAD_BYTES,
    )
    try:
        manifest_text = manifest_file.content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SkillPublishSnapshotError("snapshot_manifest_unreadable") from exc
    manifest, parsed = _manifest_projection(
        path=manifest_rel,
        manifest_text=manifest_text,
        fallback_name=skill_dir.name,
    )

    try:
        inventory = _iter_payload_files(
            skill_dir,
            manifest_entry=parsed.entry,
            manifest_scripts=parsed.scripts,
        )
    except SkillPayloadUnreadable as exc:
        raise SkillPublishSnapshotError("snapshot_payload_unreadable") from exc

    captured: list[CapturedSkillFile] = []
    public: list[CapturedSkillFile] = []
    control: list[CapturedSkillFile] = []
    public_bytes = manifest_file.byte_count
    for file_path in inventory:
        try:
            rel = file_path.relative_to(skill_dir).as_posix()
        except ValueError as exc:
            raise SkillPublishSnapshotError("snapshot_payload_unreadable") from exc
        is_control = pathlib.PurePosixPath(rel).name.lower() in SKILL_PAYLOAD_CONTROL_FILENAMES
        if rel == manifest_rel:
            item = manifest_file
        elif is_control:
            item = _read_captured_file(file_path, rel)
        else:
            item = _read_captured_file(
                file_path,
                rel,
                max_bytes=MAX_PUBLIC_PAYLOAD_BYTES - public_bytes,
            )
            public_bytes += item.byte_count
        captured.append(item)
        (control if is_control else public).append(item)

    if not any(item.path == manifest_rel for item in captured):
        raise SkillPublishSnapshotError("snapshot_manifest_unreadable")

    content_hash = reduce_skill_content_hash((item.path, bytes.fromhex(item.sha256)) for item in captured)
    stored_review_hash = str(getattr(loaded.review, "content_hash", "") or "")
    if not stored_review_hash or content_hash != stored_review_hash:
        raise SkillPublishSnapshotError("snapshot_review_stale")

    return SkillPublishSnapshot(
        skill=str(loaded.name or skill_dir.name),
        source=str(loaded.source or ""),
        manifest_file=manifest_file,
        manifest=manifest,
        content_hash=content_hash,
        full_files=tuple(captured),
        public_files=tuple(public),
        control_files=tuple(control),
    )


__all__ = [
    "MAX_PUBLIC_PAYLOAD_BYTES",
    "CapturedPublishManifest",
    "CapturedSkillFile",
    "SkillPublishSnapshot",
    "SkillPublishSnapshotError",
    "capture_skill_publish_snapshot",
]
