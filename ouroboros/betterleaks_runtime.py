"""Pinned Betterleaks runtime resolution and explicit installation.

Publishing never downloads an executable.  The read-only resolver selects an
exact bundled runtime first, then the exact managed runtime under ``DATA_DIR``.
Network acquisition is available only through the explicit module CLI::

    python -m ouroboros.betterleaks_runtime install

Release builds use the same command with ``--build-output
betterleaks-standalone``.  The six-platform pin table below is the single
source of truth for both paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping, Optional, Sequence

from ouroboros import platform_layer

BETTERLEAKS_VERSION = "1.8.1"
BETTERLEAKS_RELEASE_COMMIT = "5eab48332cc48565864514e3bc6de89df091a7c4"
BETTERLEAKS_LICENSE_SHA256 = (
    "caea114592a8f8e5e05a116d63a99e0ccd79a3ff74f4ddf270bcf4c929eb021e"
)
BETTERLEAKS_RELEASE_BASE_URL = (
    "https://github.com/betterleaks/betterleaks/releases/download/v1.8.1"
)
BETTERLEAKS_INSTALL_COMMAND = "python -m ouroboros.betterleaks_runtime install"

STANDALONE_DIRNAME = "betterleaks-standalone"
INSTALL_METADATA_FILENAME = "install.json"
LICENSE_MEMBER = "LICENSE"
README_MEMBER = "README.md"
_MANAGED_DIRNAME = "betterleaks"
_DOWNLOAD_TIMEOUT_SEC = 300.0
_PROBE_TIMEOUT_SEC = 15.0
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class BetterleaksRuntimeError(RuntimeError):
    """Typed Betterleaks delivery failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = str(code or "betterleaks_runtime_error")


@dataclass(frozen=True)
class BetterleaksArtifact:
    """Immutable identity of one official Betterleaks release asset."""

    platform_key: str
    version: str
    release_commit: str
    archive_url: str
    archive_name: str
    archive_sha256: str
    archive_kind: str
    binary_member: str
    license_member: str
    license_sha256: str

    @property
    def expected_members(self) -> frozenset[str]:
        return frozenset((self.binary_member, self.license_member, README_MEMBER))

    def validate(self) -> None:
        errors: list[str] = []
        if self.version != BETTERLEAKS_VERSION:
            errors.append("version does not match the selected Betterleaks release")
        if self.release_commit != BETTERLEAKS_RELEASE_COMMIT:
            errors.append("release commit does not match the selected Betterleaks release")
        if not self.archive_url.startswith("https://"):
            errors.append("archive URL must use HTTPS")
        if pathlib.PurePosixPath(self.archive_url).name != self.archive_name:
            errors.append("archive URL and archive name differ")
        if self.archive_kind not in {"tar.gz", "zip"}:
            errors.append("archive kind must be tar.gz or zip")
        if not _SHA256_RE.fullmatch(self.archive_sha256):
            errors.append("archive SHA-256 must be 64 lowercase hex characters")
        if not _SHA256_RE.fullmatch(self.license_sha256):
            errors.append("license SHA-256 must be 64 lowercase hex characters")
        if not _safe_archive_member(self.binary_member):
            errors.append("binary member is not a safe canonical archive path")
        if not _safe_archive_member(self.license_member):
            errors.append("license member is not a safe canonical archive path")
        if errors:
            raise BetterleaksRuntimeError("pin_invalid", "; ".join(errors))


def _artifact(
    platform_key: str,
    archive_name: str,
    archive_sha256: str,
    archive_kind: str,
    binary_member: str,
) -> BetterleaksArtifact:
    return BetterleaksArtifact(
        platform_key=platform_key,
        version=BETTERLEAKS_VERSION,
        release_commit=BETTERLEAKS_RELEASE_COMMIT,
        archive_url=f"{BETTERLEAKS_RELEASE_BASE_URL}/{archive_name}",
        archive_name=archive_name,
        archive_sha256=archive_sha256,
        archive_kind=archive_kind,
        binary_member=binary_member,
        license_member=LICENSE_MEMBER,
        license_sha256=BETTERLEAKS_LICENSE_SHA256,
    )


BETTERLEAKS_ARTIFACTS: Mapping[str, BetterleaksArtifact] = MappingProxyType(
    {
        "darwin-arm64": _artifact(
            "darwin-arm64",
            "betterleaks_1.8.1_darwin_arm64.tar.gz",
            "8e80f33b5f2a7426b390347b9fd466033723cb94b6bdffa7572632e2eaec964e",
            "tar.gz",
            "betterleaks",
        ),
        "darwin-x64": _artifact(
            "darwin-x64",
            "betterleaks_1.8.1_darwin_x64.tar.gz",
            "6abc37df76f881cffae406aa2cec72bea6e6ae64b4e771b3ed21b4aac472ed10",
            "tar.gz",
            "betterleaks",
        ),
        "linux-arm64": _artifact(
            "linux-arm64",
            "betterleaks_1.8.1_linux_arm64.tar.gz",
            "bbb578b12a2f65d7082ab436abf37724232bc71d8a078e3c41336574420f1b48",
            "tar.gz",
            "betterleaks",
        ),
        "linux-x64": _artifact(
            "linux-x64",
            "betterleaks_1.8.1_linux_x64.tar.gz",
            "efa407244e1ea8e35f582b8a42becdeac08bdead04f68eb752adda722d583c2a",
            "tar.gz",
            "betterleaks",
        ),
        "windows-arm64": _artifact(
            "windows-arm64",
            "betterleaks_1.8.1_windows_arm64.zip",
            "aa12beb9ce1f6a911da91e1d0d8a72d7e68daf56a52a53f930038fd81f10f0ba",
            "zip",
            "betterleaks.exe",
        ),
        "windows-x64": _artifact(
            "windows-x64",
            "betterleaks_1.8.1_windows_x64.zip",
            "94310d028285a1bcce7f160bc19eb62f87de6460c95bfd4319151ef5b501ed3f",
            "zip",
            "betterleaks.exe",
        ),
    }
)


@dataclass(frozen=True)
class BetterleaksRuntimeState:
    """Read-only runtime resolution result."""

    status: str
    platform_key: str
    version: str = BETTERLEAKS_VERSION
    binary_path: str = ""
    license_path: str = ""
    source: str = ""
    reason_code: str = ""
    repair_hint: str = BETTERLEAKS_INSTALL_COMMAND

    @property
    def ready(self) -> bool:
        return self.status == "ready"


_VALIDATION_CACHE: dict[tuple[object, ...], BetterleaksRuntimeState] = {}
_VALIDATION_CACHE_LOCK = threading.Lock()


def platform_key(
    *, system: Optional[str] = None, machine: Optional[str] = None
) -> str:
    """Return the six-platform release key for this host, or ``""``."""
    if system is None:
        if platform_layer.IS_WINDOWS:
            system_name = "windows"
        elif platform_layer.IS_MACOS:
            system_name = "darwin"
        elif platform_layer.IS_LINUX:
            system_name = "linux"
        else:
            system_name = platform.system().strip().lower()
    else:
        system_name = str(system).strip().lower()
    machine_name = str(machine or platform.machine()).strip().lower()
    os_name = {
        "darwin": "darwin",
        "linux": "linux",
        "windows": "windows",
    }.get(system_name, "")
    architecture = {
        "aarch64": "arm64",
        "arm64": "arm64",
        "amd64": "x64",
        "x64": "x64",
        "x86_64": "x64",
    }.get(machine_name, "")
    key = f"{os_name}-{architecture}" if os_name and architecture else ""
    return key if key in BETTERLEAKS_ARTIFACTS else ""


def current_artifact() -> BetterleaksArtifact:
    key = platform_key()
    artifact = BETTERLEAKS_ARTIFACTS.get(key)
    if artifact is None:
        raise BetterleaksRuntimeError(
            "platform_unsupported",
            "no pinned Betterleaks asset exists for this operating system and architecture",
        )
    artifact.validate()
    return artifact


def managed_runtime_root(data_root: "str | pathlib.Path | None" = None) -> pathlib.Path:
    if data_root is None:
        from ouroboros.config import DATA_DIR

        data_root = DATA_DIR
    return pathlib.Path(data_root) / "state" / _MANAGED_DIRNAME


def managed_runtime_dir(
    artifact: Optional[BetterleaksArtifact] = None,
    *,
    data_root: "str | pathlib.Path | None" = None,
) -> pathlib.Path:
    selected = artifact or current_artifact()
    selected.validate()
    return managed_runtime_root(data_root) / f"v{selected.version}" / selected.platform_key


def _default_build_cache_root() -> pathlib.Path:
    if platform_layer.IS_WINDOWS:
        base = os.environ.get("LOCALAPPDATA", "").strip()
        if base:
            return pathlib.Path(base) / "Ouroboros" / "cache" / _MANAGED_DIRNAME
    base = os.environ.get("XDG_CACHE_HOME", "").strip()
    return (
        pathlib.Path(base) / "ouroboros" / _MANAGED_DIRNAME
        if base
        else pathlib.Path.home() / ".cache" / "ouroboros" / _MANAGED_DIRNAME
    )


def _runtime_binary(root: pathlib.Path, artifact: BetterleaksArtifact) -> pathlib.Path:
    name = "betterleaks.exe" if artifact.platform_key.startswith("windows-") else "betterleaks"
    return root / "bin" / name


def _runtime_license(root: pathlib.Path) -> pathlib.Path:
    return root / LICENSE_MEMBER


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise BetterleaksRuntimeError(
            "runtime_unreadable", "Betterleaks runtime file could not be read"
        ) from exc
    return digest.hexdigest()


def _stat_fingerprint(path: pathlib.Path) -> tuple[int, int, int, int, int]:
    info = path.stat()
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_mode),
    )


def _probe_version(binary: pathlib.Path) -> str:
    try:
        completed = subprocess.run(
            [str(binary), "version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=_PROBE_TIMEOUT_SEC,
            check=False,
            **platform_layer.subprocess_hidden_kwargs(),
        )
    except (OSError, UnicodeError, subprocess.SubprocessError) as exc:
        raise BetterleaksRuntimeError(
            "runtime_probe_failed", "Betterleaks version probe failed"
        ) from exc
    if completed.returncode != 0:
        raise BetterleaksRuntimeError(
            "runtime_probe_failed", "Betterleaks version probe returned a nonzero exit"
        )
    return str(completed.stdout or "").strip()


def _read_metadata(path: pathlib.Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise BetterleaksRuntimeError(
            "runtime_metadata_invalid", "Betterleaks install metadata is invalid"
        ) from exc
    if not isinstance(raw, dict):
        raise BetterleaksRuntimeError(
            "runtime_metadata_invalid", "Betterleaks install metadata is not an object"
        )
    return raw


def _validate_runtime_root(
    root: pathlib.Path,
    artifact: BetterleaksArtifact,
    *,
    source: str,
    allow_signed_macos_digest: bool = False,
) -> BetterleaksRuntimeState:
    """Validate one exact bundled or managed root without acquiring anything."""
    artifact.validate()
    binary = _runtime_binary(root, artifact)
    license_path = _runtime_license(root)
    metadata_path = root / INSTALL_METADATA_FILENAME
    try:
        if root.is_symlink():
            raise BetterleaksRuntimeError(
                "runtime_layout_invalid", "Betterleaks runtime root must not be a symlink"
            )
        if any(path.is_symlink() for path in (binary, license_path, metadata_path)):
            raise BetterleaksRuntimeError(
                "runtime_layout_invalid", "Betterleaks runtime files must not be symlinks"
            )
        if not binary.is_file() or not license_path.is_file() or not metadata_path.is_file():
            raise BetterleaksRuntimeError(
                "runtime_layout_invalid", "Betterleaks runtime files are incomplete"
            )
        fingerprint = (
            str(root.resolve()),
            source,
            allow_signed_macos_digest,
            artifact.platform_key,
            _stat_fingerprint(binary),
            _stat_fingerprint(license_path),
            _stat_fingerprint(metadata_path),
        )
    except BetterleaksRuntimeError:
        raise
    except OSError as exc:
        raise BetterleaksRuntimeError(
            "runtime_layout_invalid", "Betterleaks runtime layout could not be inspected"
        ) from exc

    with _VALIDATION_CACHE_LOCK:
        cached = _VALIDATION_CACHE.get(fingerprint)
    if cached is not None:
        return cached

    metadata = _read_metadata(metadata_path)
    expected = {
        "schema_version": 1,
        "version": artifact.version,
        "release_commit": artifact.release_commit,
        "platform": artifact.platform_key,
        "archive_url": artifact.archive_url,
        "archive_name": artifact.archive_name,
        "archive_sha256": artifact.archive_sha256,
        "archive_kind": artifact.archive_kind,
        "binary_member": artifact.binary_member,
        "license_member": artifact.license_member,
        "license_sha256": artifact.license_sha256,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise BetterleaksRuntimeError(
            "runtime_metadata_invalid", "Betterleaks install metadata does not match the pin"
        )
    if metadata.get("install_kind") not in {"managed", "build"}:
        raise BetterleaksRuntimeError(
            "runtime_metadata_invalid", "Betterleaks install kind is invalid"
        )
    if source == "managed" and metadata.get("install_kind") != "managed":
        raise BetterleaksRuntimeError(
            "runtime_metadata_invalid", "managed Betterleaks metadata has the wrong install kind"
        )
    installed_binary_sha = str(metadata.get("binary_sha256") or "")
    if not _SHA256_RE.fullmatch(installed_binary_sha):
        raise BetterleaksRuntimeError(
            "runtime_metadata_invalid", "Betterleaks binary digest metadata is invalid"
        )
    if _sha256_file(license_path) != artifact.license_sha256:
        raise BetterleaksRuntimeError(
            "runtime_license_mismatch", "Betterleaks license digest does not match the pin"
        )

    # Nested codesign legitimately changes the staged Mach-O bytes.  A packaged
    # macOS runtime is instead bound by pinned build acquisition, this metadata,
    # exact version/license/ruleset probes, and the final app's codesign seal.
    # Managed installs and every non-macOS package still verify installed bytes.
    skip_signed_macos_digest = (
        source == "bundled"
        and platform_layer.IS_MACOS
        and allow_signed_macos_digest
    )
    if not skip_signed_macos_digest and _sha256_file(binary) != installed_binary_sha:
        raise BetterleaksRuntimeError(
            "runtime_binary_mismatch", "Betterleaks binary digest does not match its install metadata"
        )
    observed_version = _probe_version(binary)
    if observed_version != artifact.version:
        raise BetterleaksRuntimeError(
            "runtime_version_mismatch", "Betterleaks binary version does not match the pin"
        )

    state = BetterleaksRuntimeState(
        status="ready",
        platform_key=artifact.platform_key,
        binary_path=str(binary),
        license_path=str(license_path),
        source=source,
    )
    with _VALIDATION_CACHE_LOCK:
        if len(_VALIDATION_CACHE) >= 64:
            _VALIDATION_CACHE.clear()
        _VALIDATION_CACHE[fingerprint] = state
    return state


def resolve_betterleaks(
    *,
    data_root: "str | pathlib.Path | None" = None,
    bundle_bases: "Optional[Iterable[str | pathlib.Path]]" = None,
    include_managed: bool = True,
) -> BetterleaksRuntimeState:
    """Resolve bundled then exact managed Betterleaks without downloading."""
    try:
        artifact = current_artifact()
    except BetterleaksRuntimeError as exc:
        return BetterleaksRuntimeState(
            status="missing",
            platform_key="",
            reason_code=exc.code,
        )

    corrupt_code = ""
    bases = (
        platform_layer.bundled_resource_bases()
        if bundle_bases is None
        else [pathlib.Path(base) for base in bundle_bases]
    )
    source_repo = pathlib.Path(__file__).resolve().parent.parent
    for base in bases:
        root = pathlib.Path(base) / STANDALONE_DIRNAME
        try:
            present = os.path.lexists(root)
        except OSError:
            present = False
        if not present:
            continue
        try:
            try:
                is_source_staging = base.resolve() == source_repo
            except OSError:
                is_source_staging = False
            return _validate_runtime_root(
                root,
                artifact,
                source="bundled",
                # Explicit roots are extracted/installed package evidence. The
                # default source-repo base is only pre-sign build staging and
                # must retain the ordinary installed-byte digest check.
                allow_signed_macos_digest=(
                    bundle_bases is not None or not is_source_staging
                ),
            )
        except BetterleaksRuntimeError as exc:
            corrupt_code = corrupt_code or exc.code

    if include_managed:
        root = managed_runtime_dir(artifact, data_root=data_root)
        try:
            present = os.path.lexists(root)
        except OSError:
            present = False
        if present:
            try:
                return _validate_runtime_root(root, artifact, source="managed")
            except BetterleaksRuntimeError as exc:
                corrupt_code = corrupt_code or exc.code

    return BetterleaksRuntimeState(
        status="corrupt" if corrupt_code else "missing",
        platform_key=artifact.platform_key,
        reason_code=corrupt_code or "runtime_missing",
    )


def _safe_archive_member(value: str) -> bool:
    text = str(value or "")
    if not text or "\\" in text or "\x00" in text:
        return False
    path = pathlib.PurePosixPath(text)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and ":" not in path.parts[0]
        and path.as_posix() == text
    )


def verify_archive(
    path: "str | pathlib.Path", artifact: Optional[BetterleaksArtifact] = None
) -> pathlib.Path:
    selected = artifact or current_artifact()
    selected.validate()
    archive = pathlib.Path(path)
    if not archive.is_file():
        raise BetterleaksRuntimeError(
            "archive_missing", "the pinned Betterleaks archive is unavailable"
        )
    if _sha256_file(archive) != selected.archive_sha256:
        raise BetterleaksRuntimeError(
            "archive_checksum_mismatch", "Betterleaks archive checksum does not match the pin"
        )
    return archive


def _validate_archive_names(names: Sequence[str], artifact: BetterleaksArtifact) -> None:
    seen: set[str] = set()
    for name in names:
        if not _safe_archive_member(name):
            raise BetterleaksRuntimeError(
                "archive_unsafe", "Betterleaks archive contains an unsafe member path"
            )
        if name in seen:
            raise BetterleaksRuntimeError(
                "archive_duplicate", "Betterleaks archive contains a duplicate member"
            )
        seen.add(name)
    if seen != set(artifact.expected_members):
        raise BetterleaksRuntimeError(
            "archive_members_mismatch", "Betterleaks archive member set does not match the pin"
        )


def _copy_member(source, destination: pathlib.Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as sink:
        shutil.copyfileobj(source, sink, length=1024 * 1024)
        sink.flush()
        os.fsync(sink.fileno())


def _extract_archive(
    archive: pathlib.Path,
    destination: pathlib.Path,
    artifact: BetterleaksArtifact,
) -> None:
    """Inspect every member and extract only the exact binary and license."""
    binary_destination = _runtime_binary(destination, artifact)
    license_destination = _runtime_license(destination)
    try:
        if artifact.archive_kind == "tar.gz":
            with tarfile.open(archive, "r:gz") as bundle:
                members = bundle.getmembers()
                _validate_archive_names([str(member.name or "") for member in members], artifact)
                for member in members:
                    if not member.isreg():
                        raise BetterleaksRuntimeError(
                            "archive_unsafe",
                            "Betterleaks archive contains a link or non-regular member",
                        )
                for member_name, target in (
                    (artifact.binary_member, binary_destination),
                    (artifact.license_member, license_destination),
                ):
                    member = bundle.getmember(member_name)
                    source = bundle.extractfile(member)
                    if source is None:
                        raise BetterleaksRuntimeError(
                            "archive_invalid", "Betterleaks archive member could not be read"
                        )
                    with source:
                        _copy_member(source, target)
        elif artifact.archive_kind == "zip":
            with zipfile.ZipFile(archive) as bundle:
                infos = bundle.infolist()
                _validate_archive_names([str(info.filename or "") for info in infos], artifact)
                for info in infos:
                    mode = int(info.external_attr >> 16)
                    member_type = stat.S_IFMT(mode)
                    if info.is_dir() or member_type == stat.S_IFLNK or member_type not in {
                        0,
                        stat.S_IFREG,
                    }:
                        raise BetterleaksRuntimeError(
                            "archive_unsafe",
                            "Betterleaks archive contains a link or non-regular member",
                        )
                for member_name, target in (
                    (artifact.binary_member, binary_destination),
                    (artifact.license_member, license_destination),
                ):
                    with bundle.open(member_name) as source:
                        _copy_member(source, target)
        else:
            raise BetterleaksRuntimeError(
                "archive_kind_unsupported", "Betterleaks archive kind is unsupported"
            )
    except BetterleaksRuntimeError:
        raise
    except (KeyError, OSError, RuntimeError, tarfile.TarError, zipfile.BadZipFile) as exc:
        raise BetterleaksRuntimeError(
            "archive_invalid", "Betterleaks archive could not be extracted"
        ) from exc


def _download_archive(artifact: BetterleaksArtifact, target: pathlib.Path) -> pathlib.Path:
    """Download to a sibling temporary file and publish only verified bytes."""
    try:
        return verify_archive(target, artifact)
    except BetterleaksRuntimeError:
        pass
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        request = urllib.request.Request(
            artifact.archive_url,
            headers={"User-Agent": "Ouroboros-Betterleaks-runtime/1"},
        )
        with urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT_SEC) as response:
            with temporary.open("xb") as sink:
                shutil.copyfileobj(response, sink, length=1024 * 1024)
                sink.flush()
                os.fsync(sink.fileno())
        verify_archive(temporary, artifact)
        try:
            return verify_archive(target, artifact)
        except BetterleaksRuntimeError:
            os.replace(temporary, target)
        return verify_archive(target, artifact)
    except BetterleaksRuntimeError:
        raise
    except Exception as exc:
        raise BetterleaksRuntimeError(
            "archive_download_failed", "Betterleaks archive download failed"
        ) from exc
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _write_metadata(
    staging: pathlib.Path,
    artifact: BetterleaksArtifact,
    *,
    install_kind: str,
    binary_sha256: str,
) -> None:
    metadata = {
        "schema_version": 1,
        "version": artifact.version,
        "release_commit": artifact.release_commit,
        "platform": artifact.platform_key,
        "archive_url": artifact.archive_url,
        "archive_name": artifact.archive_name,
        "archive_sha256": artifact.archive_sha256,
        "archive_kind": artifact.archive_kind,
        "binary_member": artifact.binary_member,
        "binary_sha256": binary_sha256,
        "license_member": artifact.license_member,
        "license_sha256": artifact.license_sha256,
        "install_kind": install_kind,
    }
    path = staging / INSTALL_METADATA_FILENAME
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(metadata, sort_keys=True, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _path_exists(path: pathlib.Path) -> bool:
    return os.path.lexists(path)


def _remove_path(path: pathlib.Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


def _promote_directory(
    staging: pathlib.Path,
    target: pathlib.Path,
    *,
    validate,
) -> None:
    displaced = target.parent / f".{target.name}.old.{uuid.uuid4().hex[:10]}"
    had_old = _path_exists(target)
    if had_old:
        os.replace(target, displaced)
    try:
        os.replace(staging, target)
        validate(target)
    except Exception:
        if _path_exists(target):
            _remove_path(target)
        if had_old and _path_exists(displaced):
            os.replace(displaced, target)
        raise
    else:
        if _path_exists(displaced):
            _remove_path(displaced)


def install_betterleaks(
    *,
    build_output: "str | pathlib.Path | None" = None,
    data_root: "str | pathlib.Path | None" = None,
    cache_dir: "str | pathlib.Path | None" = None,
    archive_path: "str | pathlib.Path | None" = None,
    artifact: Optional[BetterleaksArtifact] = None,
) -> BetterleaksRuntimeState:
    """Explicitly install the pinned runtime, preserving any prior target."""
    selected = artifact or current_artifact()
    selected.validate()
    install_kind = "build" if build_output is not None else "managed"
    target = (
        pathlib.Path(build_output)
        if build_output is not None
        else managed_runtime_dir(selected, data_root=data_root)
    )
    validation_source = "bundled" if install_kind == "build" else "managed"

    try:
        return _validate_runtime_root(target, selected, source=validation_source)
    except BetterleaksRuntimeError:
        pass

    if archive_path is not None:
        archive = verify_archive(archive_path, selected)
    else:
        if cache_dir is not None:
            cache_base = pathlib.Path(cache_dir)
        elif build_output is not None:
            cache_base = _default_build_cache_root()
        else:
            cache_base = managed_runtime_root(data_root) / "cache"
        cache = cache_base / f"v{selected.version}" / selected.archive_name
        archive = _download_archive(selected, cache)

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = pathlib.Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp.", dir=str(target.parent))
    )
    try:
        _extract_archive(archive, staging, selected)
        binary = _runtime_binary(staging, selected)
        license_path = _runtime_license(staging)
        try:
            binary.chmod(0o755)
        except OSError as exc:
            raise BetterleaksRuntimeError(
                "runtime_install_failed", "Betterleaks binary could not be made executable"
            ) from exc
        if _sha256_file(license_path) != selected.license_sha256:
            raise BetterleaksRuntimeError(
                "runtime_license_mismatch", "Betterleaks archive license does not match the pin"
            )
        binary_sha256 = _sha256_file(binary)
        if _probe_version(binary) != selected.version:
            raise BetterleaksRuntimeError(
                "runtime_version_mismatch", "Betterleaks archive binary version does not match the pin"
            )
        _write_metadata(
            staging,
            selected,
            install_kind=install_kind,
            binary_sha256=binary_sha256,
        )
        _validate_runtime_root(staging, selected, source=validation_source)
        _promote_directory(
            staging,
            target,
            validate=lambda root: _validate_runtime_root(
                root, selected, source=validation_source
            ),
        )
        return _validate_runtime_root(target, selected, source=validation_source)
    except BetterleaksRuntimeError:
        raise
    except Exception as exc:
        raise BetterleaksRuntimeError(
            "runtime_install_failed", "Betterleaks installation failed"
        ) from exc
    finally:
        if _path_exists(staging):
            _remove_path(staging)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    install = commands.add_parser("install", help="install the exact pinned Betterleaks runtime")
    install.add_argument(
        "--build-output",
        type=pathlib.Path,
        help="stage the normalized standalone resource instead of a managed runtime",
    )
    install.add_argument(
        "--data-root",
        type=pathlib.Path,
        help="override the active Ouroboros data root",
    )
    install.add_argument(
        "--cache-dir",
        type=pathlib.Path,
        help="durable cache base (the pinned version directory is appended)",
    )
    install.add_argument(
        "--archive",
        type=pathlib.Path,
        help="use a local archive after exact pin verification instead of downloading",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command != "install":
            raise BetterleaksRuntimeError("command_invalid", "unsupported command")
        state = install_betterleaks(
            build_output=args.build_output,
            data_root=args.data_root,
            cache_dir=args.cache_dir,
            archive_path=args.archive,
        )
    except BetterleaksRuntimeError as exc:
        print(f"{exc.code}: {exc}", file=sys.stderr)
        return 1
    print(f"Betterleaks {state.version} ready ({state.source})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BETTERLEAKS_ARTIFACTS",
    "BETTERLEAKS_INSTALL_COMMAND",
    "BETTERLEAKS_LICENSE_SHA256",
    "BETTERLEAKS_RELEASE_COMMIT",
    "BETTERLEAKS_VERSION",
    "BetterleaksArtifact",
    "BetterleaksRuntimeError",
    "BetterleaksRuntimeState",
    "current_artifact",
    "install_betterleaks",
    "managed_runtime_dir",
    "managed_runtime_root",
    "platform_key",
    "resolve_betterleaks",
    "verify_archive",
]
