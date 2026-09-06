"""Fail-closed ClawHub archive staging with size, path, symlink, and file-policy guards."""

from __future__ import annotations

import hashlib
import io
import logging
import pathlib
import shutil
import tempfile
import uuid
import zipfile
from dataclasses import dataclass, field
from typing import List, Optional

log = logging.getLogger(__name__)


_MAX_TOTAL_BYTES = 50 * 1024 * 1024  # 50 MB total uncompressed
_MAX_FILE_COUNT = 200
_MAX_PER_FILE_BYTES = 8 * 1024 * 1024  # 8 MB per individual file

# Mirrors ClawHub text allowlist plus inert widget assets (images, audio, video,
# fonts) and WebAssembly, which runs only inside the sandboxed widget frame and
# reviews as a content-hash-bound descriptor; native loadable binaries stay denied.
_ALLOWED_EXTENSIONS = frozenset({
    ".md", ".markdown", ".txt", ".rst", ".org", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".py", ".sh", ".bash", ".zsh", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx",
    ".html", ".htm", ".css", ".scss", ".sass", ".svg", ".csv", ".tsv", ".sql", ".graphql", ".gql",
    ".lock", ".license", ".png", ".jpg", ".jpeg", ".gif", ".webp",
    ".wasm", ".mp3", ".ogg", ".wav", ".mp4", ".webm", ".woff", ".woff2", ".ttf", ".otf",
})

_ALLOWED_BARE_BASENAMES = frozenset({
    "LICENSE", "COPYING", "NOTICE", "README", "CHANGELOG", "AUTHORS", "CONTRIBUTORS", "AGENTS",
    ".gitignore", ".npmignore", ".editorconfig", ".gitattributes", ".eslintrc", ".prettierrc", ".nvmrc",
})


class FetchError(RuntimeError):
    """Archive validation failed closed."""


def _reject_if(condition: bool, message: str) -> None:
    if condition:
        raise FetchError(message)


@dataclass
class StagedSkill:
    """Validated archive staged in a temporary tree the caller must move or clean up."""

    slug: str
    version: str
    sha256: str
    staging_dir: pathlib.Path
    file_count: int = 0
    total_bytes: int = 0
    file_list: List[str] = field(default_factory=list)
    has_skill_md: bool = False
    has_plugin_manifest: bool = False

    def cleanup(self) -> None:
        """Best-effort staging-tree removal."""
        try:
            if self.staging_dir.exists():
                shutil.rmtree(self.staging_dir, ignore_errors=True)
        except OSError:
            log.warning(
                "Failed to clean up staging dir %s", self.staging_dir, exc_info=True
            )


def _is_sensitive(path: pathlib.PurePosixPath) -> bool:
    """Use the review denylist so marketplace cannot import sensitive filenames."""
    try:
        from ouroboros.tools.review_helpers import (
            _SENSITIVE_EXTENSIONS,
            _SENSITIVE_NAMES,
        )
    except Exception:  # pragma: no cover — defensive, the module exists
        return False
    name_lower = path.name.lower()
    if name_lower in _SENSITIVE_NAMES:
        return True
    return any(name_lower.endswith(ext) for ext in _SENSITIVE_EXTENSIONS)


def _is_loadable_binary(path: pathlib.PurePosixPath) -> bool:
    try:
        from ouroboros.skill_review import _LOADABLE_BINARY_EXTENSIONS
    except Exception:  # pragma: no cover
        return False
    name_lower = path.name.lower()
    return any(name_lower.endswith(ext) for ext in _LOADABLE_BINARY_EXTENSIONS)


def _has_review_opaque_dir(path: pathlib.PurePosixPath) -> bool:
    return any(part in {"node_modules", ".ouroboros_env"} for part in path.parts)


def _validate_member_path(name: str) -> pathlib.PurePosixPath:
    """Normalize and reject absolute/traversal zip members before extraction."""
    cleaned = name.replace("\\", "/").lstrip("/")
    _reject_if(not cleaned, f"Archive member has empty path: {name!r}")
    posix = pathlib.PurePosixPath(cleaned)
    _reject_if(bool(posix.is_absolute() or posix.anchor), f"Archive member uses absolute path: {name!r}")
    parts = posix.parts
    _reject_if(any(part == ".." for part in parts), f"Archive member uses '..' traversal: {name!r}")
    _reject_if(any(part.startswith("/") for part in parts), f"Archive member has malformed segment: {name!r}")
    # Windows drive/ADS escapes: a "C:" segment re-roots pathlib joins on
    # Windows and ":" opens NTFS alternate data streams.
    _reject_if(any(":" in part or "\\" in part for part in parts),
               f"Archive member contains ':' or backslash segment: {name!r}")
    return posix


def _classify_member(member: zipfile.ZipInfo) -> str:
    """Classify zip member; symlinks are rejected before path writes."""
    is_dir = member.is_dir() or member.filename.endswith("/")
    if is_dir:
        return "dir"
    # zipfile stores Unix mode in the high 16 external_attr bits.
    mode = (member.external_attr >> 16) & 0xFFFF
    if mode and (mode & 0xF000) == 0xA000:  # S_IFLNK
        return "symlink"
    return "file"


def _extension_allowed(path: pathlib.PurePosixPath) -> bool:
    name_lower = path.name.lower()
    if path.name in _ALLOWED_BARE_BASENAMES or name_lower in {
        bn.lower() for bn in _ALLOWED_BARE_BASENAMES
    }:
        return True
    suffix = path.suffix.lower()
    return suffix in _ALLOWED_EXTENSIONS


def stage(
    archive_bytes: bytes,
    *,
    slug: str,
    version: str = "",
    expected_sha256: Optional[str] = None,
    staging_root: Optional[pathlib.Path] = None,
) -> StagedSkill:
    """Validate and extract into a private staging dir, cleaning up on failure."""
    _reject_if(
        not isinstance(archive_bytes, (bytes, bytearray)),
        f"archive_bytes must be bytes, got {type(archive_bytes).__name__}",
    )
    _reject_if(not archive_bytes, "archive_bytes is empty")
    _reject_if(len(archive_bytes) > _MAX_TOTAL_BYTES, f"Archive size {len(archive_bytes)} bytes exceeds {_MAX_TOTAL_BYTES} cap")

    actual_sha = hashlib.sha256(archive_bytes).hexdigest()
    _reject_if(
        bool(expected_sha256 and expected_sha256.strip() and expected_sha256 != actual_sha),
        f"Archive sha256 mismatch: expected {expected_sha256}, got {actual_sha}",
    )

    if staging_root is None:
        # mkdtemp avoids attacker pre-created shared staging roots/symlinks.
        staging_dir = pathlib.Path(
            tempfile.mkdtemp(
                prefix=f"ouroboros_marketplace_{slug.replace('/', '__')}_"
            )
        )
    else:
        try:
            staging_root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise FetchError(f"Cannot create staging root {staging_root}: {exc}") from exc
        staging_dir = staging_root / f"{slug.replace('/', '__')}-{actual_sha[:12]}-{uuid.uuid4().hex[:8]}"
        staging_dir.mkdir(parents=True, exist_ok=False)

    file_list: List[str] = []
    total_bytes = 0
    file_count = 0
    has_skill_md = False
    has_plugin_manifest = False

    try:
        try:
            with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as zf:
                members = zf.infolist()
                _reject_if(not members, "Archive contains no entries")
                _reject_if(
                    len(members) > _MAX_FILE_COUNT * 2,
                    f"Archive has {len(members)} entries (cap is {_MAX_FILE_COUNT * 2} including directories)",
                )
                # Strip a common top-level wrapper so SKILL.md lands at root.
                stripped_prefix = _common_top_prefix(members)
                for member in members:
                    classification = _classify_member(member)
                    if classification == "dir":
                        continue
                    _reject_if(classification == "symlink", f"Archive member {member.filename!r} is a symlink (rejected)")
                    rel_path = _validate_member_path(member.filename)
                    if stripped_prefix:
                        parts = rel_path.parts
                        if parts and parts[0] == stripped_prefix:
                            rel_path = pathlib.PurePosixPath(*parts[1:])
                            if not rel_path.parts:
                                continue
                    _reject_if(_is_sensitive(rel_path), f"Archive contains sensitive-shape filename {rel_path}")
                    _reject_if(_has_review_opaque_dir(rel_path), f"Archive contains review-opaque dependency directory {rel_path}")
                    _reject_if(
                        _is_loadable_binary(rel_path),
                        f"Archive contains loadable-binary file {rel_path} (.so/.dll/.pyc/.exe etc. are not permitted)",
                    )
                    _reject_if(not _extension_allowed(rel_path), f"Archive contains disallowed extension: {rel_path}")
                    _reject_if(
                        member.file_size > _MAX_PER_FILE_BYTES,
                        f"Archive member {rel_path} is {member.file_size} bytes (cap {_MAX_PER_FILE_BYTES})",
                    )
                    file_count += 1
                    _reject_if(file_count > _MAX_FILE_COUNT, f"Archive exceeds file count cap {_MAX_FILE_COUNT}")
                    target_path = staging_dir / pathlib.Path(*rel_path.parts)
                    # Post-join containment: belt-and-braces against any
                    # platform-specific join surprise the lexical checks missed.
                    _reject_if(
                        not target_path.resolve(strict=False).is_relative_to(
                            staging_dir.resolve(strict=False)
                        ),
                        f"Archive member escapes staging dir after join: {rel_path}",
                    )
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member, "r") as src:
                        # Zip-bomb defense: trust actual cap+1 read, not file_size metadata.
                        data = src.read(_MAX_PER_FILE_BYTES + 1)
                    if len(data) > _MAX_PER_FILE_BYTES:
                        raise FetchError(f"Archive member {rel_path} actual size > cap {_MAX_PER_FILE_BYTES} (forged file_size header?)")
                    if len(data) != member.file_size:
                        # Recheck actual bytes; the read above already bounded memory.
                        _reject_if(
                            len(data) > _MAX_PER_FILE_BYTES,
                            f"Archive member {rel_path} actual size {len(data)} > cap {_MAX_PER_FILE_BYTES}",
                        )
                    total_bytes += len(data)
                    _reject_if(total_bytes > _MAX_TOTAL_BYTES, f"Archive uncompressed size exceeds {_MAX_TOTAL_BYTES} bytes")
                    target_path.write_bytes(data)
                    file_list.append(rel_path.as_posix())
                    if rel_path.name in ("SKILL.md", "skill.json"):
                        has_skill_md = True
                    if rel_path.name == "openclaw.plugin.json":
                        has_plugin_manifest = True
        except zipfile.BadZipFile as exc:
            raise FetchError(f"Archive is not a valid zip: {exc}") from exc

        _reject_if(not has_skill_md, "Archive does not contain SKILL.md / skill.json — not a recognisable skill package")

        return StagedSkill(
            slug=slug,
            version=(version or "").strip(),
            sha256=actual_sha,
            staging_dir=staging_dir,
            file_count=file_count,
            total_bytes=total_bytes,
            file_list=sorted(file_list),
            has_skill_md=has_skill_md,
            has_plugin_manifest=has_plugin_manifest,
        )
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def land_staged_tree(staging: pathlib.Path, target_dir: pathlib.Path, *, replacement_suffix: str) -> None:
    """Atomically swap a staged tree into place, restoring old content on move failure."""
    staging = pathlib.Path(staging)
    target_dir = pathlib.Path(target_dir)
    if target_dir.exists():
        sibling = target_dir.with_name(f"{target_dir.name}.{replacement_suffix}")
        if sibling.exists():
            shutil.rmtree(sibling, ignore_errors=True)
        target_dir.rename(sibling)
        try:
            shutil.move(str(staging), str(target_dir))
        except OSError:
            try:
                sibling.rename(target_dir)
            except OSError:
                log.error("Failed to restore previous skill after landing failure: %s, %s", target_dir, sibling)
            raise
        shutil.rmtree(sibling, ignore_errors=True)
        return
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staging), str(target_dir))


def _common_top_prefix(members: List[zipfile.ZipInfo]) -> str:
    """Return the single common top-level directory, or ''."""
    prefixes: set[str] = set()
    for member in members:
        cleaned = member.filename.replace("\\", "/").lstrip("/")
        if not cleaned:
            continue
        head, _, _ = cleaned.partition("/")
        if not head or head == cleaned:
            return ""  # at least one entry is a top-level file -> no wrapper dir
        prefixes.add(head)
        if len(prefixes) > 1:
            return ""
    if len(prefixes) == 1:
        return next(iter(prefixes))
    return ""


__all__ = [
    "FetchError",
    "StagedSkill",
    "land_staged_tree",
    "stage",
]
