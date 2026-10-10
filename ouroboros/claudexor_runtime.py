"""Managed Claudexor runtime delivery.

The reviewed pin in :mod:`ouroboros.claudexor_runtime_pin.json` selects one
immutable runtime tree.  Installation is deliberately separate from the
Claudexor auth/config home: executable bytes live under ``DATA_DIR/state/cx``
while credentials, daemon state, and runs remain under ``DATA_DIR/claudexor``.

There is no mutable ``current`` pointer.  The code's exact pin is the next-spawn
selection.  Installing a newer pin while an older daemon is alive therefore
stages the immutable directory without hot-swapping the process; the next
daemon start selects it, and a planned restart whose landed checkout changed the
pin ends the old daemon so that start follows (``server_restart``).  Rolling
Ouroboros back selects the older pin and its preserved directory again.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import re
import shutil
import stat
import subprocess
import tarfile
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass
from typing import Any, Mapping, Optional
from urllib.parse import urlparse

from ouroboros.utils import replace_atomic
from ouroboros.verified_download import fetch_exact_file as _fetch_exact_file
from ouroboros.verified_download import verify_exact_file as _verify_exact_file

log = logging.getLogger(__name__)

_PIN_SCHEMA_VERSION = 1
_PIN_FILENAME = "claudexor_runtime_pin.json"
_SEED_DIR_NAME = "claudexor-runtime"
_RUNTIME_META_FILENAME = "managed-runtime.json"
_NODE_META_FILENAME = "managed-node.json"
_NODE_META_SCHEMA_VERSION = 2
_INSTALL_LOCK_FILENAME = "install.lock"
_INSTALL_LOCK_STALE_SEC = 1800.0
_INSTALL_LOCK_WAIT_SEC = 4.0
_DOWNLOAD_TIMEOUT_SEC = 300.0
_PROBE_TIMEOUT_SEC = 20.0
_SEMVER = re.compile(r"^\d+\.\d+\.\d+$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PIN_FIELDS = frozenset({
    "version",
    "build_sha",
    "protocol_major",
    "archive_url",
    "sha256",
    "size_bytes",
    "node_version",
    "node_artifacts",
    "entrypoint",
    "cli_entrypoint",
})
_NODE_ARTIFACT_FIELDS = frozenset({"archive_url", "sha256", "size_bytes", "executable"})
_NODE_PLATFORM_KEYS = frozenset({
    "darwin-arm64",
    "darwin-x64",
    "linux-arm64",
    "linux-x64",
    "win32-x64",
})
_INSTALL_SHA_PREFIX_LENGTH = 12
_DEFAULT_PIN = object()


class ClaudexorRuntimeError(RuntimeError):
    """Typed runtime delivery failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = str(code or "claudexor_runtime_error")


@dataclass(frozen=True)
class NodeRuntimeArtifact:
    """Exact official Node archive used when a package carries no suitable Node."""

    archive_url: str
    sha256: str
    size_bytes: int
    executable: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "NodeRuntimeArtifact":
        unknown = sorted(set(raw) - _NODE_ARTIFACT_FIELDS)
        missing = sorted(_NODE_ARTIFACT_FIELDS - set(raw))
        if unknown or missing:
            details = []
            if missing:
                details.append(f"missing fields: {', '.join(missing)}")
            if unknown:
                details.append(f"unknown fields: {', '.join(unknown)}")
            raise ClaudexorRuntimeError("runtime_pin_invalid", "; ".join(details))
        try:
            artifact = cls(
                archive_url=str(raw["archive_url"]),
                sha256=str(raw["sha256"]),
                size_bytes=int(raw["size_bytes"]),
                executable=str(raw["executable"]),
            )
        except (TypeError, ValueError) as exc:
            raise ClaudexorRuntimeError(
                "runtime_pin_invalid", f"Node artifact field has the wrong type: {exc}"
            ) from exc
        artifact.validate()
        return artifact

    @property
    def archive_name(self) -> str:
        return pathlib.PurePosixPath(urlparse(self.archive_url).path).name

    def validate(self) -> None:
        parsed = urlparse(self.archive_url)
        errors = []
        if parsed.scheme != "https" or not parsed.netloc or not self.archive_name:
            errors.append("Node archive_url must be an absolute https URL with a filename")
        if not (self.archive_name.endswith(".tar.gz") or self.archive_name.endswith(".zip")):
            errors.append("Node archive_url must name a .tar.gz or .zip archive")
        if not _SHA256.fullmatch(self.sha256):
            errors.append("Node sha256 must be 64 lowercase hex characters")
        if self.size_bytes <= 0:
            errors.append("Node size_bytes must be positive")
        if not _safe_archive_relative_path(self.executable):
            errors.append("Node executable must be a safe relative archive path")
        if errors:
            raise ClaudexorRuntimeError("runtime_pin_invalid", "; ".join(errors))


@dataclass(frozen=True)
class ClaudexorRuntimePin:
    """Exact reviewed identity of one runtime archive."""

    version: str
    build_sha: str
    protocol_major: int
    archive_url: str
    sha256: str
    size_bytes: int
    # Lockstep pair: the live CI lane's setup-node version in
    # .github/workflows/claudexor-platform-gate.yml must match node_version.
    node_version: str
    node_artifacts: Mapping[str, NodeRuntimeArtifact]
    entrypoint: str
    cli_entrypoint: Optional[str] = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ClaudexorRuntimePin":
        unknown = sorted(set(raw) - _PIN_FIELDS)
        missing = sorted(_PIN_FIELDS - set(raw))
        if unknown or missing:
            details = []
            if missing:
                details.append(f"missing fields: {', '.join(missing)}")
            if unknown:
                details.append(f"unknown fields: {', '.join(unknown)}")
            raise ClaudexorRuntimeError("runtime_pin_invalid", "; ".join(details))
        try:
            cli_entrypoint = raw["cli_entrypoint"]
            if cli_entrypoint is not None and not isinstance(cli_entrypoint, str):
                raise TypeError("cli_entrypoint must be a string or null")
            pin = cls(
                version=str(raw["version"]),
                build_sha=str(raw["build_sha"]),
                protocol_major=int(raw["protocol_major"]),
                archive_url=str(raw["archive_url"]),
                sha256=str(raw["sha256"]),
                size_bytes=int(raw["size_bytes"]),
                node_version=str(raw["node_version"]),
                node_artifacts={
                    str(key): NodeRuntimeArtifact.from_mapping(value)
                    for key, value in dict(raw["node_artifacts"]).items()
                },
                entrypoint=str(raw["entrypoint"]),
                cli_entrypoint=cli_entrypoint,
            )
        except (TypeError, ValueError) as exc:
            raise ClaudexorRuntimeError(
                "runtime_pin_invalid", f"runtime pin field has the wrong type: {exc}"
            ) from exc
        pin.validate()
        return pin

    @property
    def archive_name(self) -> str:
        return pathlib.PurePosixPath(urlparse(self.archive_url).path).name

    @property
    def install_name(self) -> str:
        return f"{self.version}-{self.build_sha[:_INSTALL_SHA_PREFIX_LENGTH]}"

    def validate(self) -> None:
        from ouroboros.config import CLAUDEXOR_PROTOCOL_MAJOR

        errors = []
        if not _SEMVER.fullmatch(self.version):
            errors.append("version must be x.y.z")
        if not _GIT_SHA.fullmatch(self.build_sha):
            errors.append("build_sha must be 40 lowercase hex characters")
        if self.protocol_major != CLAUDEXOR_PROTOCOL_MAJOR:
            errors.append(
                f"protocol_major must equal Ouroboros protocol {CLAUDEXOR_PROTOCOL_MAJOR}"
            )
        parsed = urlparse(self.archive_url)
        if parsed.scheme != "https" or not parsed.netloc or not self.archive_name:
            errors.append("archive_url must be an absolute https URL with a filename")
        if not self.archive_name.endswith(".tar.gz"):
            errors.append("archive_url must name a .tar.gz archive")
        if not _SHA256.fullmatch(self.sha256):
            errors.append("sha256 must be 64 lowercase hex characters")
        if self.size_bytes <= 0:
            errors.append("size_bytes must be positive")
        if not _SEMVER.fullmatch(self.node_version):
            errors.append("node_version must be x.y.z")
        if not isinstance(self.node_artifacts, Mapping):
            errors.append("node_artifacts must be an object")
        elif set(self.node_artifacts) != _NODE_PLATFORM_KEYS:
            errors.append(
                "node_artifacts must contain exactly " + ", ".join(sorted(_NODE_PLATFORM_KEYS))
            )
        else:
            for artifact in self.node_artifacts.values():
                if not isinstance(artifact, NodeRuntimeArtifact):
                    errors.append("node_artifacts entries must be NodeRuntimeArtifact objects")
                    break
                artifact.validate()
        if not _safe_archive_relative_path(self.entrypoint):
            errors.append("entrypoint must be a safe relative archive path")
        if self.cli_entrypoint is not None and not _safe_archive_relative_path(self.cli_entrypoint):
            errors.append("cli_entrypoint must be null or a safe relative archive path")
        if errors:
            raise ClaudexorRuntimeError("runtime_pin_invalid", "; ".join(errors))


def load_runtime_pin(path: "str | pathlib.Path | None" = None) -> Optional[ClaudexorRuntimePin]:
    """Load the tracked exact pin; ``release: null`` means not published yet.

    A partial or placeholder release is a typed error, never an implicit latest
    version.  Tests and release tooling may pass an alternate file explicitly.
    """
    pin_path = pathlib.Path(path) if path is not None else pathlib.Path(__file__).with_name(_PIN_FILENAME)
    try:
        raw = json.loads(pin_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ClaudexorRuntimeError(
            "runtime_pin_missing", f"managed runtime pin not found at {pin_path}"
        ) from exc
    except (OSError, ValueError) as exc:
        raise ClaudexorRuntimeError(
            "runtime_pin_invalid", f"managed runtime pin is unreadable: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "release"}:
        raise ClaudexorRuntimeError(
            "runtime_pin_invalid", "pin root must contain exactly schema_version and release"
        )
    if raw.get("schema_version") != _PIN_SCHEMA_VERSION:
        raise ClaudexorRuntimeError(
            "runtime_pin_invalid", f"pin schema_version must be {_PIN_SCHEMA_VERSION}"
        )
    release = raw.get("release")
    if release is None:
        return None
    if not isinstance(release, dict):
        raise ClaudexorRuntimeError("runtime_pin_invalid", "pin release must be an object or null")
    return ClaudexorRuntimePin.from_mapping(release)




def verify_runtime_archive(path: "str | pathlib.Path", pin: ClaudexorRuntimePin) -> pathlib.Path:
    """Verify exact size and SHA-256, returning the validated path."""
    pin.validate()
    return _verify_exact_file(
        path,
        size_bytes=pin.size_bytes,
        sha256=pin.sha256,
        code_prefix="runtime_archive",
        label="runtime archive",
        error_type=ClaudexorRuntimeError,
    )


def verify_node_archive(
    path: "str | pathlib.Path", artifact: NodeRuntimeArtifact
) -> pathlib.Path:
    """Verify one exact platform Node archive from the reviewed pin."""
    artifact.validate()
    return _verify_exact_file(
        path,
        size_bytes=artifact.size_bytes,
        sha256=artifact.sha256,
        code_prefix="runtime_node_archive",
        label="Node archive",
        error_type=ClaudexorRuntimeError,
    )




def fetch_runtime_archive(
    pin: ClaudexorRuntimePin, destination: "str | pathlib.Path"
) -> pathlib.Path:
    """Atomically fetch the exact archive; an existing valid file wins.

    ``destination`` is the final file path, not a directory.  The response may
    follow the release CDN's redirects because exact size and SHA-256 bind the
    admitted bytes.  A failed download never replaces a valid cache entry.
    """
    pin.validate()
    return _fetch_exact_file(
        url=pin.archive_url,
        destination=destination,
        verify=lambda path: verify_runtime_archive(path, pin),
        size_bytes=pin.size_bytes,
        overflow_code="runtime_archive_size_mismatch",
        failure_code="runtime_download_failed",
        label="managed runtime",
        timeout_sec=_DOWNLOAD_TIMEOUT_SEC,
        error_type=ClaudexorRuntimeError,
    )


def fetch_node_archive(
    artifact: NodeRuntimeArtifact, destination: "str | pathlib.Path"
) -> pathlib.Path:
    """Atomically fetch one exact platform Node archive."""
    artifact.validate()
    return _fetch_exact_file(
        url=artifact.archive_url,
        destination=destination,
        verify=lambda path: verify_node_archive(path, artifact),
        size_bytes=artifact.size_bytes,
        overflow_code="runtime_node_archive_size_mismatch",
        failure_code="runtime_node_download_failed",
        label="managed Node",
        timeout_sec=_DOWNLOAD_TIMEOUT_SEC,
        error_type=ClaudexorRuntimeError,
    )


def managed_runtime_root() -> pathlib.Path:
    from ouroboros.config import DATA_DIR

    # Keep this path deliberately compact.  The public runtime closure contains
    # package-manager paths up to roughly 170 characters, while legacy Windows
    # consumers may still enforce MAX_PATH for extraction and Node module loads.
    # Exact identity is not weakened by the short directory name: the full SHA,
    # archive digest, size and entrypoint are checked in managed-runtime.json and
    # the candidate's own side-effect-free probe before promotion.
    return pathlib.Path(DATA_DIR) / "state" / "cx"


def managed_runtime_dir(pin: ClaudexorRuntimePin) -> pathlib.Path:
    pin.validate()
    return managed_runtime_root() / pin.install_name


def managed_node_dir(pin: ClaudexorRuntimePin, platform_key: str) -> pathlib.Path:
    """Immutable host-Node directory for source and pre-Node package upgrades."""
    pin.validate()
    return managed_runtime_root() / "node" / f"{pin.node_version}-{platform_key}"


def _safe_archive_relative_path(value: str) -> bool:
    text = str(value or "")
    if not text or "\\" in text or "\x00" in text:
        return False
    path = pathlib.PurePosixPath(text)
    # A nested Windows drive-relative path (or alternate data stream) is unsafe
    # too: npm_root.joinpath("D:entry.js") need not stay beneath npm_root.
    return bool(path.parts) and not path.is_absolute() and ".." not in path.parts and all(
        ":" not in part for part in path.parts
    )


def _safe_zip_member_type(info: zipfile.ZipInfo) -> bool:
    """Accept directories and regular files, including ZIPs without Unix type bits."""
    file_type = stat.S_IFMT(info.external_attr >> 16)
    if info.is_dir():
        return file_type in (0, stat.S_IFDIR)
    return file_type in (0, stat.S_IFREG)


def _explicit_binary() -> tuple[str, str]:
    raw = str(os.environ.get("OUROBOROS_CLAUDEXOR_BIN", "") or "").strip()
    if not raw:
        return "", ""
    try:
        if pathlib.Path(raw).is_file():
            return raw, ""
    except OSError:
        pass
    return "", f"OUROBOROS_CLAUDEXOR_BIN does not name a file: {raw}"


def _compatibility_binary() -> str:
    found = shutil.which("claudexord")
    if found:
        return found
    from ouroboros.platform_layer import executable_name_candidates

    managed_bin = pathlib.Path.home() / ".claudexor" / "node" / "bin"
    for name in executable_name_candidates("claudexord"):
        candidate = managed_bin / name
        try:
            if candidate.is_file():
                return str(candidate)
        except OSError:
            continue
    return ""


def resolve_external_claudexord() -> str:
    """Compatibility resolver retained for callers of the old binary seam."""
    explicit, error = _explicit_binary()
    if explicit or error:
        return explicit
    return _compatibility_binary()


class ClaudexorRuntimeManager:
    """Single authority for next-spawn runtime selection and delivery."""

    def __init__(self, pin: Any = _DEFAULT_PIN) -> None:
        self._lock = threading.Lock()
        self._installing = False
        self._last_error = ""
        self._launch_probe: dict[str, Any] = {}
        self._pin_error = ""
        self._pin_error_code = ""
        if pin is _DEFAULT_PIN:
            try:
                self._pin = load_runtime_pin()
            except ClaudexorRuntimeError as exc:
                self._pin = None
                self._pin_error = f"{exc.code}: {exc}"
                self._pin_error_code = exc.code
        else:
            self._pin = pin
            if self._pin is not None and not isinstance(self._pin, ClaudexorRuntimePin):
                raise TypeError("pin must be ClaudexorRuntimePin or None")
            if self._pin is not None:
                self._pin.validate()

    @property
    def pin(self) -> Optional[ClaudexorRuntimePin]:
        return self._pin

    def resolve_command(self) -> list[str]:
        """Read-only next-spawn selection: override -> managed -> compatibility."""
        explicit, override_error = _explicit_binary()
        if explicit:
            return [explicit]
        if override_error:
            return []
        if self._pin_error:
            return []
        if self._pin is not None:
            return self._managed_command()
        external = _compatibility_binary()
        return [external] if external else []

    def resolve_cli_command(self, *, require_npm: bool = True) -> list[str]:
        """Resolve the installed CLI; pure operator commands need Node, not npm."""
        if self._pin_error or self._pin is None or self._pin.cli_entrypoint is None:
            return []
        return self._managed_cli_command(require_npm=require_npm)

    def ensure_cli_command(self) -> list[str]:
        """Provision the pin-bound closure and managed Node/npm CLI toolchain."""
        with self._lock:
            pin = self._pin
            if self._pin_error:
                raise ClaudexorRuntimeError(
                    self._pin_error_code or "runtime_pin_invalid", self._pin_error
                )
            if pin is None or pin.cli_entrypoint is None:
                raise ClaudexorRuntimeError(
                    "runtime_cli_unavailable",
                    "the reviewed Claudexor closure has no managed CLI entrypoint",
                )
            try:
                self._cli_node_artifact(pin)
                command = self._managed_cli_command()
                if not command:
                    self._install(pin, require_npm=True)
                    command = self._managed_cli_command()
                if not command:
                    raise ClaudexorRuntimeError(
                        "runtime_cli_unavailable",
                        "the exact managed Claudexor CLI/toolchain is not selectable",
                    )
                self._last_error = ""
                return command
            except ClaudexorRuntimeError as exc:
                self._last_error = f"{exc.code}: {exc}"
                raise

    def ensure(self) -> list[str]:
        """Foreground install/repair/update intent, returning the next command."""
        with self._lock:
            explicit, override_error = _explicit_binary()
            if explicit:
                self._last_error = ""
                return [explicit]
            if override_error:
                self._last_error = override_error
                raise ClaudexorRuntimeError("runtime_override_invalid", override_error)
            if self._pin is not None:
                try:
                    command = self._managed_command()
                    if command:
                        try:
                            self._probe(command, self._pin)
                        except ClaudexorRuntimeError:
                            self._install(self._pin)
                        else:
                            self._last_error = ""
                            return self._daemon_command(command)
                    else:
                        self._install(self._pin)
                    command = self._managed_command(probe=True)
                    if command:
                        self._last_error = ""
                        return self._daemon_command(command)
                    raise ClaudexorRuntimeError(
                        "runtime_install_failed",
                        "the exact managed runtime was not selectable after installation",
                    )
                except ClaudexorRuntimeError as exc:
                    self._last_error = f"{exc.code}: {exc}"
                    raise
            if self._pin_error:
                self._last_error = self._pin_error
                raise ClaudexorRuntimeError(
                    self._pin_error_code or "runtime_pin_invalid", self._pin_error
                )
            external = _compatibility_binary()
            if external:
                self._last_error = self._pin_error
                return [external]
            message = self._pin_error or (
                "managed Claudexor runtime release is not pinned yet and no compatible "
                "external claudexord was found"
            )
            self._last_error = message
            raise ClaudexorRuntimeError("claudexord_not_installed", message)

    def status(
        self, *, running: bool = False, engine_version: str = "", engine_build_sha: str = ""
    ) -> dict[str, Any]:
        """Read-only browser/diagnostic projection; never downloads or extracts."""
        explicit, override_error = _explicit_binary()
        target_meta = self._managed_metadata()
        installed_meta = target_meta or self._other_installed_metadata(
            engine_version=engine_version,
            engine_build_sha=engine_build_sha,
        )
        external = (
            _compatibility_binary()
            if self._pin is None and not self._pin_error and not explicit and not override_error
            else ""
        )
        pin = self._pin
        target_root_present = bool(pin is not None and managed_runtime_dir(pin).exists())
        target_version = pin.version if pin is not None else ""
        version = str(installed_meta.get("version") or (engine_version if running else ""))
        build_sha = str(installed_meta.get("build_sha") or (engine_build_sha if running else ""))
        node_version = str(installed_meta.get("node_version") or "")
        source = "override" if explicit else (
            str(installed_meta.get("archive_source") or "") if installed_meta else (
                "external" if external else ""
            )
        )
        staged_version = ""
        last_error = self._last_error or self._pin_error or override_error
        if self._install_in_progress() and not target_meta:
            state = "installing"
        elif explicit or external:
            state = "ready"
        elif target_meta:
            target_is_running = (
                running
                and engine_version == str(target_meta.get("version") or "")
                and engine_build_sha == str(target_meta.get("build_sha") or "")
            )
            if running and not target_is_running:
                state = "update_staged"
                staged_version = str(target_meta.get("version") or "")
            else:
                state = "ready"
        elif target_root_present:
            # A first install has no target directory at all. If the immutable
            # target exists but fails metadata/entrypoint identity checks, it is
            # a damaged installation and the UI must offer Fix, not Install.
            state = "error"
            last_error = last_error or "managed runtime files are incomplete or fail identity checks"
        elif last_error and (pin is not None or self._pin_error or override_error):
            state = "error"
        elif pin is not None and (
            installed_meta
            or (
                running
                and (
                    engine_version != pin.version
                    or engine_build_sha != pin.build_sha
                )
            )
        ):
            state = "update_available"
        else:
            state = "missing"
        return {
            "state": state,
            "version": version,
            "target_version": target_version,
            "staged_version": staged_version,
            "build_sha": build_sha,
            "source": source,
            "node_version": node_version,
            "last_error": last_error or None,
        }

    def _other_installed_metadata(
        self, *, engine_version: str = "", engine_build_sha: str = ""
    ) -> dict[str, Any]:
        """Select a valid preserved non-target tree without mutating it."""
        versions = managed_runtime_root()
        try:
            candidates = list(versions.iterdir())
        except OSError:
            return {}
        rows = []
        target = managed_runtime_dir(self._pin) if self._pin is not None else None
        for candidate in candidates:
            if target is not None and candidate == target:
                continue
            metadata = self._read_metadata(candidate)
            if not metadata:
                continue
            exact_build = bool(
                engine_build_sha and metadata.get("build_sha") == engine_build_sha
            )
            exact_version = bool(
                engine_version and metadata.get("version") == engine_version
            )
            try:
                modified = candidate.stat().st_mtime
            except OSError:
                modified = 0.0
            rows.append((exact_build, exact_version, modified, metadata))
        if not rows:
            return {}
        rows.sort(key=lambda item: item[:3], reverse=True)
        return rows[0][3]

    def resolve_serving_role_command(
        self,
        *,
        engine_version: str,
        engine_build_sha: str,
        engine_entry: str,
        role: str,
    ) -> list[str]:
        """Resolve one role on the exact immutable tree serving the daemon.

        The reviewed pin selects the *next* daemon spawn.  A live daemon may
        keep serving an older preserved tree while that pin is already staged,
        so a recovery command binds instead to all three handshake identity
        fields.  Supported schema-1/2 metadata locates the preserved Node and
        entry; a fresh side-effect-free probe establishes the additive role
        without rewriting metadata underneath the live process.
        """
        version = str(engine_version or "").strip()
        build_sha = str(engine_build_sha or "").strip()
        entry_text = str(engine_entry or "").strip()
        required_role = str(role or "").strip()
        if (
            not _SEMVER.fullmatch(version)
            or not _GIT_SHA.fullmatch(build_sha)
            or not entry_text
            or not required_role
        ):
            raise ClaudexorRuntimeError(
                "runtime_serving_identity_invalid",
                "the serving daemon did not disclose a complete version/build/entry identity",
            )
        try:
            disclosed_entry = pathlib.Path(entry_text).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ClaudexorRuntimeError(
                "runtime_serving_entry_unavailable",
                f"the serving daemon entry is unavailable: {type(exc).__name__}",
            ) from exc
        if not disclosed_entry.is_file():
            raise ClaudexorRuntimeError(
                "runtime_serving_entry_unavailable",
                "the serving daemon entry is not a regular file",
            )

        metadata: dict[str, Any] = {}
        try:
            candidates = sorted(managed_runtime_root().iterdir(), key=lambda path: path.name)
        except OSError:
            candidates = []
        for candidate in candidates:
            current = self._read_metadata(candidate)
            if (
                current.get("version") != version
                or current.get("build_sha") != build_sha
            ):
                continue
            candidate_entry = candidate / pathlib.PurePosixPath(
                str(current.get("entrypoint") or "")
            )
            try:
                if candidate_entry.resolve(strict=True) != disclosed_entry:
                    continue
            except (OSError, RuntimeError):
                continue
            metadata = current
            break
        if not metadata:
            raise ClaudexorRuntimeError(
                "runtime_serving_tree_unavailable",
                "the exact runtime tree reported by the serving daemon is not preserved",
            )

        node_version = str(metadata.get("node_version") or "")
        node = self._resolve_preserved_node(node_version)
        if not node:
            raise ClaudexorRuntimeError(
                "runtime_serving_node_unavailable",
                f"the preserved Node {node_version or 'unknown'} for the serving daemon is unavailable",
            )
        command = [node, str(disclosed_entry)]
        probe, _ = self._probe_payload(command, expected_node_version=node_version)
        if (
            str(probe.get("version") or "") != version
            or str(probe.get("buildSha") or "") != build_sha
        ):
            raise ClaudexorRuntimeError(
                "runtime_probe_identity_mismatch",
                "serving runtime probe identity does not match the daemon handshake",
            )
        roles = probe.get("roles")
        advertised = (
            {str(value) for value in roles if isinstance(value, str) and value}
            if isinstance(roles, list)
            else set()
        )
        if required_role not in advertised:
            raise ClaudexorRuntimeError(
                "runtime_role_unavailable",
                f"the serving runtime does not advertise the {required_role} role",
            )
        return command

    @staticmethod
    def _read_metadata(root: pathlib.Path) -> dict[str, Any]:
        try:
            raw = json.loads((root / _RUNTIME_META_FILENAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(raw, dict):
            return {}
        version = str(raw.get("version") or "")
        build_sha = str(raw.get("build_sha") or "")
        node_version = str(raw.get("node_version") or "")
        entrypoint = str(raw.get("entrypoint") or "")
        cli_entrypoint = raw.get("cli_entrypoint")
        if (
            raw.get("schema_version") != 1
            or not _SEMVER.fullmatch(version)
            or not _GIT_SHA.fullmatch(build_sha)
            or not _SEMVER.fullmatch(node_version)
            or not _safe_archive_relative_path(entrypoint)
            or (cli_entrypoint is not None and (
                not isinstance(cli_entrypoint, str)
                or not _safe_archive_relative_path(cli_entrypoint)
            ))
        ):
            return {}
        try:
            if not (root / pathlib.PurePosixPath(entrypoint)).is_file():
                return {}
            if cli_entrypoint is not None and not (
                root / pathlib.PurePosixPath(cli_entrypoint)
            ).is_file():
                return {}
        except OSError:
            return {}
        return raw

    def _managed_metadata(self) -> dict[str, Any]:
        pin = self._pin
        if pin is None:
            return {}
        root = managed_runtime_dir(pin)
        raw = self._read_metadata(root)
        if not raw:
            return {}
        expected = {
            "schema_version": 1,
            "version": pin.version,
            "build_sha": pin.build_sha,
            "protocol_major": pin.protocol_major,
            "archive_sha256": pin.sha256,
            "archive_size": pin.size_bytes,
            "node_version": pin.node_version,
            "entrypoint": pin.entrypoint,
        }
        if pin.cli_entrypoint is not None:
            expected["cli_entrypoint"] = pin.cli_entrypoint
        if any(raw.get(key) != value for key, value in expected.items()):
            return {}
        entrypoint = root / pathlib.PurePosixPath(pin.entrypoint)
        try:
            if not entrypoint.is_file():
                return {}
            if pin.cli_entrypoint is not None and not (
                root / pathlib.PurePosixPath(pin.cli_entrypoint)
            ).is_file():
                return {}
        except OSError:
            return {}
        return raw

    @staticmethod
    def _resolve_preserved_node(node_version: str) -> str:
        """Find an already-admitted Node without consulting the next pin."""
        from ouroboros.platform_layer import (
            bundled_resource_bases,
            embedded_node_candidates,
            node_distribution_platform,
            probe_node_version,
        )

        version = str(node_version or "").strip()
        if not _SEMVER.fullmatch(version):
            return ""
        for base in bundled_resource_bases():
            for candidate in embedded_node_candidates(base):
                try:
                    if candidate.is_file() and probe_node_version(str(candidate)) == version:
                        return str(candidate.resolve())
                except (OSError, RuntimeError):
                    continue
        platform_key = node_distribution_platform()
        if not platform_key:
            return ""
        root = managed_runtime_root() / "node" / f"{version}-{platform_key}"
        try:
            raw = json.loads((root / _NODE_META_FILENAME).read_text(encoding="utf-8"))
        except OSError:
            return ""
        except ValueError as exc:
            raise ClaudexorRuntimeError(
                "runtime_serving_node_metadata_invalid",
                "the preserved Node metadata is not valid JSON",
            ) from exc
        schema = raw.get("schema_version") if isinstance(raw, dict) else None
        if (
            not isinstance(raw, dict)
            or type(schema) is not int
            or schema not in (1, _NODE_META_SCHEMA_VERSION)
            or raw.get("version") != version
            or raw.get("platform") != platform_key
        ):
            raise ClaudexorRuntimeError(
                "runtime_serving_node_metadata_invalid",
                "the preserved Node metadata has an unsupported or malformed schema",
            )
        candidate = embedded_node_candidates(root)[0]
        try:
            if candidate.is_file() and probe_node_version(str(candidate)) == version:
                return str(candidate.resolve())
        except (OSError, RuntimeError):
            pass
        return ""

    def _resolve_node(self, pin: ClaudexorRuntimePin) -> str:
        """Select an exact packaged/source Node, then an exact managed copy."""
        from ouroboros.platform_layer import (
            bundled_resource_bases,
            embedded_node_candidates,
            node_distribution_platform,
            probe_node_version,
        )

        for base in bundled_resource_bases():
            for candidate in embedded_node_candidates(base):
                try:
                    if candidate.is_file() and probe_node_version(str(candidate)) == pin.node_version:
                        return str(candidate)
                except OSError:
                    continue
        platform_key = node_distribution_platform()
        if not platform_key:
            return ""
        return self._resolve_managed_node(pin, platform_key)

    @staticmethod
    def _archive_npm_cli(artifact: NodeRuntimeArtifact, platform_key: str) -> str:
        executable = pathlib.PurePosixPath(artifact.executable)
        if platform_key.startswith("win32-"):
            if executable.name != "node.exe":
                return ""
            return str(executable.parent / "node_modules/npm/bin/npm-cli.js")
        if executable.parts[-2:] != ("bin", "node"):
            return ""
        return str(executable.parent.parent / "lib/node_modules/npm/bin/npm-cli.js")

    @staticmethod
    def _managed_npm_cli(node: pathlib.Path) -> pathlib.Path:
        if node.suffix.lower() == ".exe":
            return node.parent / "node_modules/npm/bin/npm-cli.js"
        return node.parent.parent / "lib/node_modules/npm/bin/npm-cli.js"

    def _resolve_managed_node(
        self, pin: ClaudexorRuntimePin, platform_key: str, *, require_npm: bool = False
    ) -> str:
        from ouroboros.platform_layer import embedded_node_candidates, probe_node_version

        root = managed_node_dir(pin, platform_key)
        try:
            raw = json.loads((root / _NODE_META_FILENAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ""
        artifact = pin.node_artifacts.get(platform_key)
        if artifact is None:
            return ""
        archive_npm_cli = self._archive_npm_cli(artifact, platform_key)
        expected = {
            "version": pin.node_version,
            "platform": platform_key,
            "archive_url": artifact.archive_url,
            "archive_sha256": artifact.sha256,
            "archive_size": artifact.size_bytes,
            "archive_executable": artifact.executable,
        }
        if not isinstance(raw, dict) or any(raw.get(key) != value for key, value in expected.items()):
            return ""
        schema = raw.get("schema_version")
        if schema not in (1, _NODE_META_SCHEMA_VERSION):
            return ""
        if require_npm and (
            schema != _NODE_META_SCHEMA_VERSION
            or not archive_npm_cli
            or raw.get("archive_npm_cli") != archive_npm_cli
        ):
            return ""
        candidate = embedded_node_candidates(root)[0]
        try:
            if not candidate.is_file() or probe_node_version(str(candidate)) != pin.node_version:
                return ""
            if require_npm and not self._managed_npm_cli(candidate).is_file():
                return ""
            return str(candidate)
        except OSError:
            return ""

    def _resolve_node_toolchain(self, pin: ClaudexorRuntimePin) -> str:
        from ouroboros.platform_layer import node_distribution_platform

        platform_key = node_distribution_platform()
        if not platform_key:
            return ""
        return self._resolve_managed_node(pin, platform_key, require_npm=True)

    def _cli_node_artifact(
        self, pin: ClaudexorRuntimePin
    ) -> tuple[str, NodeRuntimeArtifact]:
        from ouroboros.platform_layer import node_distribution_platform

        platform_key = node_distribution_platform()
        artifact = pin.node_artifacts.get(platform_key)
        if (
            not platform_key
            or artifact is None
            or not self._archive_npm_cli(artifact, platform_key)
        ):
            raise ClaudexorRuntimeError(
                "runtime_cli_platform_unsupported",
                "the reviewed platform has no managed Node/npm CLI toolchain contract",
            )
        return platform_key, artifact

    def _ensure_node(self, pin: ClaudexorRuntimePin) -> str:
        """Provision the pin's exact Node only when no exact packaged copy exists."""
        node = self._resolve_node(pin)
        if node:
            return node
        from ouroboros.platform_layer import node_distribution_platform

        platform_key = node_distribution_platform()
        artifact = pin.node_artifacts.get(platform_key)
        if not platform_key or artifact is None:
            raise ClaudexorRuntimeError(
                "runtime_node_platform_unsupported",
                "no reviewed Node artifact exists for this operating system and architecture",
            )
        cache = managed_runtime_root() / "cache" / artifact.archive_name
        archive = fetch_node_archive(artifact, cache)
        self._promote_node(
            pin, platform_key, artifact, archive,
            include_npm=pin.cli_entrypoint is not None,
        )
        node = self._resolve_node(pin)
        if not node:
            raise ClaudexorRuntimeError(
                "runtime_node_install_failed",
                "the exact managed Node was not selectable after installation",
            )
        return node

    def _ensure_node_toolchain(self, pin: ClaudexorRuntimePin) -> str:
        node = self._resolve_node_toolchain(pin)
        if node:
            return node
        platform_key, artifact = self._cli_node_artifact(pin)
        cache = managed_runtime_root() / "cache" / artifact.archive_name
        archive = fetch_node_archive(artifact, cache)
        self._promote_node(pin, platform_key, artifact, archive, include_npm=True)
        node = self._resolve_node_toolchain(pin)
        if not node:
            raise ClaudexorRuntimeError(
                "runtime_node_install_failed",
                "the exact managed Node/npm toolchain was not selectable after installation",
            )
        return node

    def _promote_node(
        self,
        pin: ClaudexorRuntimePin,
        platform_key: str,
        artifact: NodeRuntimeArtifact,
        archive: pathlib.Path,
        *,
        include_npm: bool = False,
    ) -> None:
        from ouroboros.platform_layer import embedded_node_candidates, probe_node_version
        from ouroboros.utils import atomic_write_json

        root = managed_node_dir(pin, platform_key)
        parent = root.parent
        parent.mkdir(parents=True, exist_ok=True)
        nonce = uuid.uuid4().hex[:8]
        staging = parent / f".tmp-{os.getpid()}-{nonce}"
        displaced = parent / f".old-{nonce}"
        try:
            staging.mkdir(parents=False, exist_ok=False)
            node = embedded_node_candidates(staging)[0]
            node.parent.mkdir(parents=True, exist_ok=True)
            archive_npm_cli = self._archive_npm_cli(artifact, platform_key) if include_npm else ""
            npm_root = self._managed_npm_cli(node).parent.parent if archive_npm_cli else None
            self._extract_node_archive(
                archive, artifact, node,
                archive_npm_cli=archive_npm_cli, npm_root=npm_root,
            )
            try:
                node.chmod(0o755)
            except OSError:
                pass
            actual = probe_node_version(str(node))
            if actual != pin.node_version:
                raise ClaudexorRuntimeError(
                    "runtime_node_version_mismatch",
                    f"managed Node {actual or 'unknown'} does not match the reviewed {pin.node_version}",
                )
            metadata = {
                "schema_version": _NODE_META_SCHEMA_VERSION if archive_npm_cli else 1,
                "version": pin.node_version,
                "platform": platform_key,
                "archive_url": artifact.archive_url,
                "archive_sha256": artifact.sha256,
                "archive_size": artifact.size_bytes,
                "archive_executable": artifact.executable,
            }
            if archive_npm_cli:
                metadata["archive_npm_cli"] = archive_npm_cli
            atomic_write_json(
                staging / _NODE_META_FILENAME, metadata,
                trailing_newline=True,
                fsync=True,
            )
            had_old = root.exists()
            if had_old:
                replace_atomic(root, displaced)
            try:
                replace_atomic(staging, root)
            except Exception:
                if had_old and displaced.exists() and not root.exists():
                    replace_atomic(displaced, root)
                raise
            if displaced.exists():
                shutil.rmtree(displaced, ignore_errors=True)
        except ClaudexorRuntimeError:
            raise
        except Exception as exc:
            raise ClaudexorRuntimeError(
                "runtime_node_install_failed",
                f"managed Node promotion failed: {type(exc).__name__}: {exc}",
            ) from exc
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            if displaced.exists() and root.exists():
                shutil.rmtree(displaced, ignore_errors=True)

    @staticmethod
    def _extract_node_archive(
        archive: pathlib.Path, artifact: NodeRuntimeArtifact, destination: pathlib.Path,
        *, archive_npm_cli: str = "", npm_root: Optional[pathlib.Path] = None,
    ) -> None:
        """Copy exact Node and, when requested, a regular-file npm tree."""
        try:
            if artifact.archive_name.endswith(".zip"):
                if bool(archive_npm_cli) != (npm_root is not None):
                    raise ClaudexorRuntimeError(
                        "runtime_node_archive_invalid", "reviewed npm path is invalid"
                    )
                with zipfile.ZipFile(archive) as bundle:
                    info = bundle.getinfo(artifact.executable)
                    if info.is_dir() or not _safe_zip_member_type(info):
                        raise ClaudexorRuntimeError(
                            "runtime_node_archive_invalid", "Node executable is not a regular file"
                        )
                    with bundle.open(info) as source, destination.open("xb") as sink:
                        shutil.copyfileobj(source, sink)
                    if archive_npm_cli:
                        if npm_root is None or not _safe_archive_relative_path(archive_npm_cli):
                            raise ClaudexorRuntimeError(
                                "runtime_node_archive_invalid", "reviewed npm path is invalid"
                            )
                        npm_prefix_path = pathlib.PurePosixPath(archive_npm_cli).parent.parent
                        npm_prefix = str(npm_prefix_path)
                        selected, seen = [], set()
                        for npm_member in bundle.infolist():
                            name = str(npm_member.filename or "")
                            if name != npm_prefix and not name.startswith(npm_prefix + "/"):
                                continue
                            if name in seen or not _safe_archive_relative_path(name):
                                raise ClaudexorRuntimeError(
                                    "runtime_node_archive_invalid",
                                    "Node archive has a duplicate or unsafe npm path",
                                )
                            seen.add(name)
                            if not _safe_zip_member_type(npm_member):
                                raise ClaudexorRuntimeError(
                                    "runtime_node_archive_invalid",
                                    f"Node npm entry {name!r} is a link or special file",
                                )
                            selected.append(npm_member)
                        if archive_npm_cli not in seen:
                            raise ClaudexorRuntimeError(
                                "runtime_node_archive_invalid",
                                "Node archive lacks the reviewed npm entrypoint",
                            )
                        for npm_member in selected:
                            relative = pathlib.PurePosixPath(npm_member.filename).relative_to(npm_prefix_path)
                            target = npm_root.joinpath(*relative.parts)
                            if npm_member.is_dir():
                                target.mkdir(parents=True, exist_ok=True)
                                continue
                            target.parent.mkdir(parents=True, exist_ok=True)
                            with bundle.open(npm_member) as npm_source, target.open("xb") as sink:
                                shutil.copyfileobj(npm_source, sink)
            else:
                with tarfile.open(archive, "r:gz") as bundle:
                    member = bundle.getmember(artifact.executable)
                    if not member.isreg():
                        raise ClaudexorRuntimeError(
                            "runtime_node_archive_invalid", "Node executable is not a regular file"
                        )
                    source = bundle.extractfile(member)
                    if source is None:
                        raise ClaudexorRuntimeError(
                            "runtime_node_archive_invalid", "Node executable could not be read"
                        )
                    with source, destination.open("xb") as sink:
                        shutil.copyfileobj(source, sink)
                    if archive_npm_cli:
                        if npm_root is None or not _safe_archive_relative_path(archive_npm_cli):
                            raise ClaudexorRuntimeError(
                                "runtime_node_archive_invalid", "reviewed npm path is invalid"
                            )
                        npm_prefix_path = pathlib.PurePosixPath(archive_npm_cli).parent.parent
                        npm_prefix = str(npm_prefix_path)
                        selected, seen = [], set()
                        for npm_member in bundle.getmembers():
                            name = str(npm_member.name or "")
                            if name != npm_prefix and not name.startswith(npm_prefix + "/"):
                                continue
                            if name in seen or not _safe_archive_relative_path(name):
                                raise ClaudexorRuntimeError(
                                    "runtime_node_archive_invalid",
                                    "Node archive has a duplicate or unsafe npm path",
                                )
                            seen.add(name)
                            if not (npm_member.isdir() or npm_member.isreg()):
                                raise ClaudexorRuntimeError(
                                    "runtime_node_archive_invalid",
                                    f"Node npm entry {name!r} is a link or special file",
                                )
                            selected.append(npm_member)
                        if archive_npm_cli not in seen:
                            raise ClaudexorRuntimeError(
                                "runtime_node_archive_invalid",
                                "Node archive lacks the reviewed npm entrypoint",
                            )
                        for npm_member in selected:
                            relative = pathlib.PurePosixPath(npm_member.name).relative_to(npm_prefix_path)
                            target = npm_root.joinpath(*relative.parts)
                            if npm_member.isdir():
                                target.mkdir(parents=True, exist_ok=True)
                                continue
                            target.parent.mkdir(parents=True, exist_ok=True)
                            npm_source = bundle.extractfile(npm_member)
                            if npm_source is None:
                                raise ClaudexorRuntimeError(
                                    "runtime_node_archive_invalid",
                                    f"Node npm entry {npm_member.name!r} could not be read",
                                )
                            with npm_source, target.open("xb") as sink:
                                shutil.copyfileobj(npm_source, sink)
                            try:
                                target.chmod(npm_member.mode & 0o777)
                            except OSError:
                                pass
        except ClaudexorRuntimeError:
            raise
        except (KeyError, OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
            raise ClaudexorRuntimeError(
                "runtime_node_archive_invalid",
                f"Node archive extraction failed: {type(exc).__name__}: {exc}",
            ) from exc

    def _managed_command(self, *, probe: bool = False) -> list[str]:
        pin = self._pin
        metadata = self._managed_metadata()
        if pin is None or not metadata:
            return []
        node = self._resolve_node(pin)
        if not node:
            return []
        command = [node, str(managed_runtime_dir(pin) / pathlib.PurePosixPath(pin.entrypoint))]
        if probe:
            self._probe(command, pin)
        return command

    def _daemon_command(self, command: list[str]) -> list[str]:
        """Apply the engine's heap recommendation only to the actual spawn.

        The daemon inherits NODE_OPTIONS unchanged from this same environment;
        an operator's heap option wins. Explicit/external commands bypass this.
        """
        if "launch" not in self._launch_probe:
            return command
        launch = self._launch_probe["launch"]
        args = launch.get("nodeArgs") if isinstance(launch, dict) else None
        if not isinstance(args, list) or not all(
            isinstance(arg, str) and re.fullmatch(r"--max-old-space-size=\d{3,6}", arg, flags=re.ASCII)
            for arg in args
        ):
            self._last_error = "runtime_launch_invalid: engine heap arguments ignored"
            return command
        if "--max-old-space-size" in os.environ.get("NODE_OPTIONS", ""):
            return command
        return [command[0], *args, command[-1]]

    def _managed_cli_command(self, *, require_npm: bool = True) -> list[str]:
        pin = self._pin
        if pin is None or pin.cli_entrypoint is None or not self._managed_metadata():
            return []
        node = self._resolve_node_toolchain(pin) if require_npm else self._resolve_node(pin)
        cli = managed_runtime_dir(pin) / pathlib.PurePosixPath(pin.cli_entrypoint)
        try:
            return [node, str(cli)] if node and cli.is_file() else []
        except OSError:
            return []

    def _install(self, pin: ClaudexorRuntimePin, *, require_npm: bool = False) -> None:
        from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock

        root = managed_runtime_root()
        lock_path = root / _INSTALL_LOCK_FILENAME
        lock_fd = acquire_exclusive_file_lock(
            lock_path,
            timeout_sec=_INSTALL_LOCK_WAIT_SEC,
            stale_sec=_INSTALL_LOCK_STALE_SEC,
            metadata=f"pid={os.getpid()} version={pin.version} build_sha={pin.build_sha}\n",
        )
        if lock_fd is None:
            command = self._managed_cli_command() if require_npm else self._managed_command()
            if command:
                return
            raise ClaudexorRuntimeError(
                "runtime_install_in_progress", "another process is installing the managed runtime"
            )
        self._installing = True
        try:
            if require_npm:
                self._ensure_node_toolchain(pin)
            else:
                self._ensure_node(pin)
            try:
                command = (
                    self._managed_cli_command()
                    if require_npm
                    else self._managed_command(probe=True)
                )
                if command:
                    return
            except ClaudexorRuntimeError:
                pass
            cache = root / "cache" / pin.archive_name
            archive, archive_source = self._obtain_archive(pin, cache)
            self._promote_archive(pin, archive, archive_source)
        finally:
            self._installing = False
            release_exclusive_file_lock(lock_path, lock_fd)

    def _obtain_archive(
        self, pin: ClaudexorRuntimePin, cache: pathlib.Path
    ) -> tuple[pathlib.Path, str]:
        try:
            return verify_runtime_archive(cache, pin), "cache"
        except ClaudexorRuntimeError:
            pass
        from ouroboros.platform_layer import bundled_resource_bases

        for base in bundled_resource_bases():
            seed = base / _SEED_DIR_NAME / pin.archive_name
            try:
                verify_runtime_archive(seed, pin)
            except ClaudexorRuntimeError:
                continue
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache.with_name(f".{cache.name}.seed.{os.getpid()}.{uuid.uuid4().hex}")
            try:
                shutil.copyfile(seed, temporary)
                verify_runtime_archive(temporary, pin)
                replace_atomic(temporary, cache)
                return verify_runtime_archive(cache, pin), "bundle_seed"
            finally:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
        return fetch_runtime_archive(pin, cache), "download"

    def _promote_archive(
        self, pin: ClaudexorRuntimePin, archive: pathlib.Path, archive_source: str
    ) -> None:
        root = managed_runtime_dir(pin)
        versions = root.parent
        versions.mkdir(parents=True, exist_ok=True)
        nonce = uuid.uuid4().hex[:8]
        staging = versions / f".tmp-{os.getpid()}-{nonce}"
        displaced = versions / f".old-{nonce}"
        try:
            staging.mkdir(parents=False, exist_ok=False)
            self._extract_archive(archive, staging)
            node = self._resolve_node(pin)
            if not node:
                raise ClaudexorRuntimeError(
                    "runtime_node_missing", "the exact packaged or managed Node runtime is unavailable"
                )
            command = [node, str(staging / pathlib.PurePosixPath(pin.entrypoint))]
            self._probe(command, pin)
            if pin.cli_entrypoint is not None and not (
                staging / pathlib.PurePosixPath(pin.cli_entrypoint)
            ).is_file():
                raise ClaudexorRuntimeError(
                    "runtime_archive_invalid", "runtime archive lacks its reviewed CLI entrypoint"
                )
            from ouroboros.utils import atomic_write_json

            metadata = {
                "schema_version": 1,
                "version": pin.version,
                "build_sha": pin.build_sha,
                "protocol_major": pin.protocol_major,
                "archive_sha256": pin.sha256,
                "archive_size": pin.size_bytes,
                "node_version": pin.node_version,
                "entrypoint": pin.entrypoint,
                "archive_source": archive_source,
            }
            if pin.cli_entrypoint is not None:
                metadata["cli_entrypoint"] = pin.cli_entrypoint
            atomic_write_json(
                staging / _RUNTIME_META_FILENAME, metadata,
                trailing_newline=True,
                fsync=True,
            )
            had_old = root.exists()
            if had_old:
                os.replace(root, displaced)
            try:
                os.replace(staging, root)
            except Exception:
                if had_old and displaced.exists() and not root.exists():
                    os.replace(displaced, root)
                raise
            if displaced.exists():
                shutil.rmtree(displaced, ignore_errors=True)
        except ClaudexorRuntimeError:
            raise
        except Exception as exc:
            raise ClaudexorRuntimeError(
                "runtime_install_failed", f"managed runtime promotion failed: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            if displaced.exists() and root.exists():
                shutil.rmtree(displaced, ignore_errors=True)

    def _probe_payload(
        self, command: list[str], *, expected_node_version: str
    ) -> tuple[dict[str, Any], str]:
        from ouroboros.platform_layer import probe_node_version, subprocess_hidden_kwargs

        node_version = probe_node_version(command[0])
        if node_version != expected_node_version:
            raise ClaudexorRuntimeError(
                "runtime_node_version_mismatch",
                "packaged Node "
                f"{node_version or 'unknown'} does not match the reviewed "
                f"{expected_node_version}",
            )
        env = dict(os.environ)
        # The probe must report the identity stamped into the bundle. An
        # inherited build-SHA override would let the candidate echo the answer
        # it is being checked against.
        env.pop("CLAUDEXOR_BUILD_SHA", None)
        try:
            completed = subprocess.run(
                [command[0], command[-1], "--probe"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=_PROBE_TIMEOUT_SEC,
                env=env,
                **subprocess_hidden_kwargs(),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ClaudexorRuntimeError(
                "runtime_probe_failed", f"runtime probe failed: {type(exc).__name__}: {exc}"
            ) from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()[-500:]
            raise ClaudexorRuntimeError(
                "runtime_probe_failed",
                f"runtime probe exited {completed.returncode}" + (f": {detail}" if detail else ""),
            )
        probe = None
        for line in reversed((completed.stdout or "").splitlines()):
            try:
                candidate = json.loads(line)
            except ValueError:
                continue
            if isinstance(candidate, dict):
                probe = candidate
                break
        if not isinstance(probe, dict):
            raise ClaudexorRuntimeError("runtime_probe_failed", "runtime probe returned no JSON object")
        return probe, node_version

    def _probe(self, command: list[str], pin: ClaudexorRuntimePin) -> str:
        probe, node_version = self._probe_payload(
            command, expected_node_version=pin.node_version
        )
        if str(probe.get("version") or "") != pin.version or str(probe.get("buildSha") or "") != pin.build_sha:
            raise ClaudexorRuntimeError(
                "runtime_probe_identity_mismatch",
                "runtime probe identity does not match the reviewed version/build SHA",
            )
        self._launch_probe = probe
        return node_version

    @staticmethod
    def _extract_archive(archive: pathlib.Path, destination: pathlib.Path) -> None:
        try:
            with tarfile.open(archive, "r:gz") as bundle:
                members = bundle.getmembers()
                for member in members:
                    name = str(member.name or "")
                    if not _safe_archive_relative_path(name) and name not in (".", "./"):
                        raise ClaudexorRuntimeError(
                            "runtime_archive_unsafe", f"runtime archive has unsafe path {name!r}"
                        )
                    if not (member.isdir() or member.isreg()):
                        raise ClaudexorRuntimeError(
                            "runtime_archive_unsafe",
                            f"runtime archive entry {name!r} is a link or special file",
                        )
                bundle.extractall(destination, members=members)
        except ClaudexorRuntimeError:
            raise
        except (OSError, tarfile.TarError) as exc:
            raise ClaudexorRuntimeError(
                "runtime_archive_invalid", f"runtime archive extraction failed: {type(exc).__name__}: {exc}"
            ) from exc

    def _install_in_progress(self) -> bool:
        if self._installing:
            return True
        lock = managed_runtime_root() / _INSTALL_LOCK_FILENAME
        try:
            return lock.is_file() and (time.time() - lock.stat().st_mtime) < _INSTALL_LOCK_STALE_SEC
        except OSError:
            return False


_MANAGER: Optional[ClaudexorRuntimeManager] = None
_MANAGER_LOCK = threading.Lock()


def get_runtime_manager() -> ClaudexorRuntimeManager:
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = ClaudexorRuntimeManager()
        return _MANAGER


__all__ = [
    "ClaudexorRuntimeError",
    "ClaudexorRuntimeManager",
    "ClaudexorRuntimePin",
    "NodeRuntimeArtifact",
    "fetch_node_archive",
    "fetch_runtime_archive",
    "get_runtime_manager",
    "load_runtime_pin",
    "managed_node_dir",
    "managed_runtime_dir",
    "managed_runtime_root",
    "resolve_external_claudexord",
    "verify_node_archive",
    "verify_runtime_archive",
]
