"""Concrete, adapter-owned CyberGym execution lifecycle.

The benchmark launcher keeps its pre-admission path dependency free.  This
module is loaded only after admission and provides the small amount of runtime
wiring that the pure protocol helpers cannot provide: a pinned CyberGym
generator, a rootless Docker sidecar/workspace pair, an Ouroboros task gateway,
and the official final PoC verification path.  All effects are behind injected
command and HTTP callables, which keeps the contract unit-testable without
Docker, the upstream package, or provider credentials on CI workers.

The executor deliberately does not implement scoring or a second scheduler.
The upstream server remains the source of truth for vulnerable/fixed exits;
``run_campaign`` owns the campaign budget and result rows.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import os
import pathlib
import posixpath
import re
import stat
import tarfile
import tempfile
import threading
import time
import urllib.parse
import uuid
from collections.abc import Callable, Mapping
from typing import Any

from devtools.benchmarks.cybergym.cybergym_adapter import (
    DEFAULT_DISABLED_TOOLS,
    DEFAULT_LEVEL,
    MAX_TASK_TIMEOUT_SEC,
    TaskSpec,
    build_generate_task_argv,
    safe_task_path,
)
from devtools.benchmarks.cybergym.cybergym_custody import (  # noqa: F401
    _CustodyMixin,
)
from devtools.benchmarks.cybergym.cybergym_dispatch import WorkspaceCustodyPending
from devtools.benchmarks.cybergym.cybergym_docker import (  # noqa: F401
    _EXPECTED_MODEL,
    _GATEWAY_TASK_ID,
    _GENERATED_INPUT_EXCLUDES,
    _GENERATED_TRACKED_INPUTS,
    _OPENROUTER_KEY_ENV,
    _OPENROUTER_URL,
    _WORKSPACE_BACKEND_ALIAS_NAME,
    _WORKSPACE_BACKEND_ALIAS_SCHEMA,
    _WORKSPACE_BACKEND_ALIAS_TARGET,
    CommandResult,
    CommandRunner,
    HttpRunner,
    _bind_container_image,
    _DockerRuntimeMixin,
    _image_digest,
    _initialize_generated_workspace_git,
    _inside,
    _install_workspace_backend_alias,
    _minimal_child_env,
    _paths_overlap,
    _pid_from_observation,
    _safe_abs,
    _write_json,
    run_command,
)
from devtools.benchmarks.cybergym.cybergym_lifecycle import (  # noqa: F401
    _deadline_guidance,
    _LifecycleMixin,
    _parse_json_stdout,
    _record_matches,
    _response_poc_id,
    _reuse_directory_observation,
    _validate_verify_response,
)
from devtools.benchmarks.cybergym.cybergym_reconcile import (  # noqa: F401
    _ReconcileMixin,
)
from devtools.benchmarks.cybergym.cybergym_sidecar import (
    API_KEY_ENV,
    EXECUTOR_NETWORK_DECLARATION,
    DockerHostRef,
    NetworkPlan,
    SidecarExpectation,
    attest_sidecar_runtime,
    make_opaque_agent_id,
    resolve_rootless_docker_host,
)
from devtools.benchmarks.cybergym.cybergym_wire import (  # noqa: F401
    _HEX64,
    _PROVIDER_ID,
    ExecutorFailure,
    GatewayAdmissionRejected,
    GatewayTransportError,
    HttpStatusError,
    _cost_final_marker,
    _cost_is_pending,
    _definitive_admission_rejection,
    _gateway_execution_status,
    _gateway_has_tool_markup,
    _gateway_path,
    _nonnegative_number,
    _path_under_any_root,
    _positive_int,
    _read_json_ref,
    _require_exact_effort,
    _response_status,
    _response_wire_telemetry,
    _runtime_value,
    _served_telemetry,
    _strict_flag,
    _unwrap_http_json,
    _unwrap_http_payload,
    urllib_json,
)

_ARCHIVE_RENAME_DIR_FD = os.rename in os.supports_dir_fd
# Rollback/cleanup needs a smaller, independent capability set than publish:
# stat/open/unlink/rmdir must all accept a directory descriptor and stat must
# support no-follow inspection.  In particular, do not infer this from rename
# support on platforms where one of the primitives is path-only.


_ARCHIVE_CLEANUP_DIR_FD = (
    all(
        operation in os.supports_dir_fd
        for operation in (os.open, os.stat, os.unlink, os.rmdir)
    )
    and os.listdir in os.supports_fd
    and os.stat in os.supports_follow_symlinks
    and bool(getattr(os, "O_NOFOLLOW", 0))
)


@dataclasses.dataclass(frozen=True)
class ExecutorConfig:
    """Immutable inputs for one campaign executor.

    ``server_root`` is mounted at the identical absolute path in the server
    container.  The upstream verifier launches nested containers through the
    mounted Docker socket and bind-mounts these paths, so translating it to a
    cosmetic ``/cybergym-data`` path would make every verification fail.
    """

    campaign_id: str
    source_root: pathlib.Path
    data_root: pathlib.Path
    mask_map: pathlib.Path
    run_root: pathlib.Path
    server_root: pathlib.Path
    server_image: str
    server_image_digest: str
    workspace_image: str
    workspace_image_digest: str
    ouroboros_url: str
    docker_host: str | DockerHostRef
    model: str = _EXPECTED_MODEL
    settings_path: pathlib.Path | None = None
    server_port: int = 8666
    verifier_host_port: int = 0
    task_timeout_sec: int = 21_600
    difficulty: str = DEFAULT_LEVEL
    api_key_env: str = API_KEY_ENV
    provider_key_env: str = _OPENROUTER_KEY_ENV
    provider_url: str = _OPENROUTER_URL
    provider_probe: bool = True
    provider_only: tuple[str, ...] = ()
    provider_order: tuple[str, ...] = ()
    provider_allow_fallbacks: bool = True
    provider_inventory_probe: bool = True
    binary_dir: pathlib.Path | None = None
    expose_vulnerable_runtime: bool = False
    expected_data_sha256: str = ""
    expected_binary_sha256: str = ""
    preverified_data_observation: Mapping[str, Any] | None = None
    preverified_binary_observation: Mapping[str, Any] | None = None
    log_dir: pathlib.Path | None = None
    db_path: pathlib.Path | None = None
    # When the campaign-owned Ouroboros server keeps its mutable state on an
    # external disk (``--state-dir``), its wire-evidence refs resolve below
    # that data root rather than below ``run_root``.  Telemetry verification
    # must accept exactly that extra root, never a broader one.
    isolate_data_root: pathlib.Path | None = None
    python_executable: str = "python"
    command: tuple[str, ...] = ("tail", "-f", "/dev/null")
    disabled_tools: tuple[str, ...] = DEFAULT_DISABLED_TOOLS
    poll_interval_sec: float = 3.0
    command_runner: CommandRunner = run_command
    http_runner: HttpRunner = urllib_json
    sleep: Callable[[float], None] = time.sleep

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", self.campaign_id):
            raise ExecutorFailure("campaign_id is unsafe")
        source = _safe_abs(self.source_root, "source_root")
        data = _safe_abs(self.data_root, "data_root")
        mask = _safe_abs(self.mask_map, "mask_map")
        run = _safe_abs(self.run_root, "run_root")
        server_root = _safe_abs(self.server_root, "server_root")
        for other, name in ((source, "source_root"), (data, "data_root"), (run, "run_root")):
            try:
                server_root.relative_to(other)
                overlaps = True
            except ValueError:
                try:
                    other.relative_to(server_root)
                    overlaps = True
                except ValueError:
                    overlaps = False
            if overlaps:
                raise ExecutorFailure(f"server_root must not overlap {name}")
        # Keep the reusable executor safe even when a caller bypasses the
        # launcher admission helper.  The common run-root module is the SSOT
        # for the operator's live repo/data roots; importing it here is lazy and
        # therefore does not weaken the dependency-free pre-admission path.
        try:
            from devtools.benchmarks.common.run_roots import live_data_roots, live_repo_roots

            forbidden = [
                pathlib.Path(item).expanduser().resolve(strict=False)
                for item in (*live_data_roots(), *live_repo_roots())
            ]
        except (ImportError, OSError, ValueError):
            forbidden = []
        for candidate, name in (
            (source, "source_root"),
            (data, "data_root"),
            (mask, "mask_map"),
            (server_root, "server_root"),
            (run, "run_root"),
        ):
            if any(_paths_overlap(candidate, root) for root in forbidden):
                raise ExecutorFailure(f"{name} overlaps a live Ouroboros root")
        # The map may live in an immutable dataset cache.  ``start`` stages a
        # byte-for-byte copy below ``server_root`` before the sidecar starts;
        # requiring it to already be there would make a clean source/data
        # checkout impossible to use without mutating it.
        if mask == pathlib.Path("/"):
            raise ExecutorFailure("mask_map cannot be the filesystem root")
        if not mask.name or mask.name in {".", ".."}:
            raise ExecutorFailure("mask_map must name a file")
        if self.difficulty != DEFAULT_LEVEL:
            raise ExecutorFailure("only Level-1 CyberGym is supported")
        if str(self.model or "").strip() != _EXPECTED_MODEL:
            raise ExecutorFailure(f"model must be exactly {_EXPECTED_MODEL!r}")
        if (
            self.task_timeout_sec <= 0
            or self.task_timeout_sec != int(self.task_timeout_sec)
            or int(self.task_timeout_sec) > MAX_TASK_TIMEOUT_SEC
        ):
            raise ExecutorFailure(
                f"task_timeout_sec must be a positive integer <= {MAX_TASK_TIMEOUT_SEC}"
            )
        if not str(self.ouroboros_url).startswith(("http://", "https://")):
            raise ExecutorFailure("ouroboros_url must be an HTTP URL")
        if self.api_key_env != API_KEY_ENV:
            raise ExecutorFailure(f"api_key_env must be {API_KEY_ENV}")
        if self.provider_key_env != _OPENROUTER_KEY_ENV:
            raise ExecutorFailure(f"provider_key_env must be {_OPENROUTER_KEY_ENV}")
        parsed_provider = urllib.parse.urlsplit(str(self.provider_url))
        if str(self.provider_url).rstrip("/") != _OPENROUTER_URL or parsed_provider.scheme != "https":
            raise ExecutorFailure("provider_url must be the pinned OpenRouter chat-completions route")
        if not self.server_port or not 1 <= int(self.server_port) <= 65535:
            raise ExecutorFailure("server_port must be a TCP port")
        if self.verifier_host_port and not 1 <= int(self.verifier_host_port) <= 65535:
            raise ExecutorFailure("verifier_host_port must be zero or a TCP port")
        try:
            poll_interval = float(self.poll_interval_sec)
        except (TypeError, ValueError) as exc:
            raise ExecutorFailure("poll_interval_sec must be finite and non-negative") from exc
        if not math.isfinite(poll_interval) or poll_interval < 0:
            raise ExecutorFailure("poll_interval_sec must be finite and non-negative")
        object.__setattr__(self, "poll_interval_sec", poll_interval)
        for field_name in ("provider_only", "provider_order"):
            raw_values = getattr(self, field_name)
            if isinstance(raw_values, str):
                raw_values = tuple(item.strip() for item in raw_values.split(",") if item.strip())
            else:
                raw_values = tuple(raw_values or ())
            if any(not isinstance(item, str) or not _PROVIDER_ID.fullmatch(item) for item in raw_values):
                raise ExecutorFailure(f"{field_name} contains an unsafe provider id")
            object.__setattr__(self, field_name, tuple(dict.fromkeys(raw_values)))
        if self.provider_only and self.provider_order:
            overlap = set(self.provider_only) - set(self.provider_order)
            if overlap:
                raise ExecutorFailure("provider_only must be contained in provider_order")
        try:
            host = resolve_rootless_docker_host(self.docker_host)
        except Exception as exc:
            # Keep the public executor boundary typed even though the pure
            # sidecar validator uses ValueError for CI-friendly callers.
            raise ExecutorFailure("invalid rootless Docker host") from exc
        object.__setattr__(self, "source_root", source)
        object.__setattr__(self, "data_root", data)
        object.__setattr__(self, "mask_map", mask)
        object.__setattr__(self, "run_root", run)
        object.__setattr__(self, "server_root", server_root)
        object.__setattr__(self, "model", _EXPECTED_MODEL)
        if self.settings_path is not None:
            settings = _safe_abs(self.settings_path, "settings_path")
            if not settings.is_file():
                raise ExecutorFailure("settings_path must name an applied settings file")
            object.__setattr__(self, "settings_path", settings)
        for field_name in ("binary_dir", "log_dir", "db_path"):
            value = getattr(self, field_name)
            if value is not None:
                resolved = _safe_abs(value, field_name)
                # binary_dir may live outside server_root: the lifecycle then
                # bind-mounts it into the sidecar read-only at the identical
                # absolute path (see _server external_binary_mounts). log_dir
                # and db_path stay confined because the verifier writes them.
                if field_name != "binary_dir":
                    _inside(resolved, server_root, field_name)
                object.__setattr__(self, field_name, resolved)
        if self.isolate_data_root is not None:
            isolate = _safe_abs(self.isolate_data_root, "isolate_data_root")
            if isolate == pathlib.Path("/"):
                raise ExecutorFailure("isolate_data_root cannot be the filesystem root")
            if any(_paths_overlap(isolate, root) for root in forbidden):
                raise ExecutorFailure("isolate_data_root overlaps a live Ouroboros root")
            object.__setattr__(self, "isolate_data_root", isolate)
        object.__setattr__(self, "docker_host", host)
        object.__setattr__(self, "server_image_digest", _image_digest(self.server_image_digest, "server_image_digest"))
        object.__setattr__(self, "workspace_image_digest", _image_digest(self.workspace_image_digest, "workspace_image_digest"))
        if self.provider_probe:
            for value, name in (
                (self.expected_data_sha256, "expected_data_sha256"),
                (self.expected_binary_sha256, "expected_binary_sha256"),
            ):
                if not re.fullmatch(r"[0-9a-fA-F]{64}", str(value or "").strip()):
                    raise ExecutorFailure(f"{name} is required for a paid immutable input")
        if (self.preverified_data_observation is None) != (
            self.preverified_binary_observation is None
        ):
            raise ExecutorFailure(
                "preverified data and binary observations must be supplied together"
            )


def _archive_relative(value: Any) -> str:
    """Normalize one POSIX member name and keep it inside the destination.

    Any non-NUL relative POSIX name is valid, including the colon, backslash
    and newline of pinned AFL and Capstone inputs; extraction refuses on
    platforms without POSIX descriptor primitives before names matter.
    """
    if not isinstance(value, str) or not value:
        raise ExecutorFailure("task archive member name is empty")
    if "\x00" in value:
        raise ExecutorFailure("task archive member name contains NUL")
    if value.startswith("/"):
        raise ExecutorFailure("task archive member name must be relative")
    normalized = posixpath.normpath(value)
    if normalized in {"", "."}:
        return "."
    if normalized.startswith("../") or normalized == "..":
        raise ExecutorFailure("task archive member name escapes its workspace")
    return normalized


def _archive_path(root: pathlib.Path, relative: str) -> pathlib.Path:
    """Join a validated POSIX path without host separator tricks."""
    return root if relative == "." else root.joinpath(*pathlib.PurePosixPath(relative).parts)


def _assert_archive_parent_is_directory(root: pathlib.Path, relative: str) -> None:
    """Reject an existing symlink or non-directory in a member's parents."""
    current = root
    for part in pathlib.PurePosixPath(relative).parts[:-1]:
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ExecutorFailure("task archive destination cannot be inspected") from exc
        if stat.S_ISLNK(info.st_mode):
            raise ExecutorFailure("task archive destination contains a symlinked parent")
        if not stat.S_ISDIR(info.st_mode):
            raise ExecutorFailure("task archive destination parent is not a directory")


def _assert_archive_root_is_not_symlink(path: pathlib.Path) -> None:
    """Reject a destination whose lexical path resolves through a link."""
    absolute = pathlib.Path(os.path.abspath(os.fspath(path)))
    try:
        resolved = absolute.resolve(strict=False)
    except OSError as exc:
        raise ExecutorFailure("task archive destination cannot be inspected") from exc
    if resolved != absolute:
        raise ExecutorFailure("task archive destination must not traverse a symlink")


def _remove_archive_entry_at(
    dir_fd: int,
    name: str,
    expected_identity: tuple[int, int] | None = None,
) -> None:
    """Remove one archive entry relative to an already-open directory.

    A lexical ``Path`` is unsafe during rollback: the destination's parent may
    have been renamed and replaced while the publish loop was running.  The
    descriptor keeps the operation anchored to the directory that received the
    entry, and ``O_NOFOLLOW`` prevents a replaced directory from redirecting a
    recursive walk.
    """
    try:
        info = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if expected_identity is not None and (
        int(info.st_dev), int(info.st_ino)
    ) != expected_identity:
        raise RuntimeError("published archive entry was replaced")
    if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        child_fd = os.open(name, flags, dir_fd=dir_fd)
        try:
            child_info = os.fstat(child_fd)
            if (
                int(child_info.st_dev), int(child_info.st_ino)
            ) != (int(info.st_dev), int(info.st_ino)):
                raise RuntimeError("published archive entry was replaced")
            for child in os.listdir(child_fd):
                _remove_archive_entry_at(child_fd, child)
        finally:
            os.close(child_fd)
        current = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
        if expected_identity is not None and (
            int(current.st_dev), int(current.st_ino)
        ) != expected_identity:
            raise RuntimeError("published archive entry was replaced")
        os.rmdir(name, dir_fd=dir_fd)
    else:
        current = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
        if expected_identity is not None and (
            int(current.st_dev), int(current.st_ino)
        ) != expected_identity:
            raise RuntimeError("published archive entry was replaced")
        os.unlink(name, dir_fd=dir_fd)


def _safe_extract(archive: pathlib.Path, destination: pathlib.Path) -> None:
    """Extract a task tree with GNU-tar parity for files, directories and links.

    Python 3.10 has no dependable extraction filter.  Every member name is
    validated first (relative, no NUL, no lexical escape, no duplicate, no
    hardlink/FIFO/device, no member below a link or file).  Directories and
    files are written into private staging, then every symlink is created last
    with its archive target verbatim: a dangling, absolute or outside-pointing
    target is data, exactly as GNU tar keeps it, and extraction never follows
    or writes through one.  Validated top-level entries are published by
    descriptor-relative rename with identity-checked rollback.
    """

    destination = pathlib.Path(destination).expanduser()
    _assert_archive_root_is_not_symlink(destination)
    try:
        destination.mkdir(parents=True, exist_ok=True)
        destination = destination.resolve(strict=True)
        root_info = destination.lstat()
    except OSError as exc:
        raise ExecutorFailure("task archive destination is unavailable") from exc
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise ExecutorFailure("task archive destination must be a regular directory")
    destination_identity = (int(root_info.st_dev), int(root_info.st_ino))

    try:
        destination_parent_info = destination.parent.lstat()
    except OSError as exc:
        raise ExecutorFailure("task archive destination parent is unavailable") from exc
    if stat.S_ISLNK(destination_parent_info.st_mode) or not stat.S_ISDIR(destination_parent_info.st_mode):
        raise ExecutorFailure("task archive destination parent must be a regular directory")
    destination_parent_identity = (
        int(destination_parent_info.st_dev),
        int(destination_parent_info.st_ino),
    )

    # The CyberGym task workspace is an untrusted boundary.  On a platform
    # without both descriptor-relative rename and descriptor-safe cleanup there
    # is no race-free way to publish several top-level entries and roll them
    # back. Refuse before creating staging rather than silently leaving a
    # partially published tree or following a replaced path.
    if not (_ARCHIVE_RENAME_DIR_FD and _ARCHIVE_CLEANUP_DIR_FD):
        raise ExecutorFailure(
            "task archive requires descriptor-safe publish and cleanup primitives"
        )

    staging: pathlib.Path | None = None
    staging_identity: tuple[int, int] | None = None
    # Entries published through a directory descriptor carry that descriptor
    # into rollback. The source inode is recorded before rename, avoiding a
    # post-rename stat window in which a replacement could be mistaken for our
    # entry.
    published: list[tuple[pathlib.Path, int | None, str | None, tuple[int, int] | None]] = []
    publish_dir_fd: int | None = None
    publish_parent_fd: int | None = None

    def rollback() -> None:
        rollback_error: Exception | None = None
        for path, dir_fd, name, identity in reversed(published):
            try:
                if dir_fd is None or name is None:
                    # A path-only rollback cannot be made race-safe: a parent
                    # can change after any identity check and redirect the
                    # unlink/rmtree.  Refuse the destructive operation when
                    # no descriptor anchor was retained.
                    raise RuntimeError("task archive rollback requires descriptor-safe cleanup")
                else:
                    # Refuse to unlink a replacement entry.  Leaving it in
                    # place is safer than deleting an object not authored by
                    # this extraction attempt.
                    if identity is None:
                        raise RuntimeError("published archive entry identity is unavailable")
                    _remove_archive_entry_at(dir_fd, name, expected_identity=identity)
            except Exception as exc:  # pragma: no cover - filesystem failure
                rollback_error = exc
        published.clear()
        if rollback_error is not None:
            raise ExecutorFailure("task archive publish rollback failed") from rollback_error

    try:
        staging = pathlib.Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.extract-", dir=destination.parent)
        )
        try:
            staging_info = staging.lstat()
        except OSError as exc:
            raise ExecutorFailure("task archive staging directory is unavailable") from exc
        staging_identity = (int(staging_info.st_dev), int(staging_info.st_ino))
        root = staging
        with tarfile.open(archive, "r:*") as tar:
            members: dict[str, tarfile.TarInfo] = {}
            member_types: dict[str, str] = {}
            implicit_dirs: set[str] = {"."}
            link_targets: dict[str, str] = {}
            for member in tar.getmembers():
                relative = _archive_relative(member.name)
                if relative in members:
                    raise ExecutorFailure("task archive contains duplicate member paths")
                if member.isdir():
                    kind = "dir"
                elif member.isreg():
                    kind = "file"
                elif member.issym():
                    kind = "link"
                else:
                    raise ExecutorFailure("task archive contains a special member")
                if relative == "." and kind != "dir":
                    raise ExecutorFailure("task archive root member must be a directory")
                members[relative] = member
                member_types[relative] = kind
                for parent in pathlib.PurePosixPath(relative).parents:
                    parent_text = parent.as_posix()
                    if parent_text != ".":
                        implicit_dirs.add(parent_text)
                if kind == "link":
                    linkname = member.linkname
                    if not isinstance(linkname, str) or not linkname or "\x00" in linkname:
                        raise ExecutorFailure("task archive symlink target is empty or contains NUL")
                    link_targets[relative] = linkname

            # Reject file/link parents before any filesystem write, so no
            # member is ever placed through a link.  Archive contents are
            # published only after every member is valid.
            for relative in member_types:
                for parent in pathlib.PurePosixPath(relative).parents:
                    parent_text = parent.as_posix()
                    if parent_text != "." and member_types.get(parent_text) not in {None, "dir"}:
                        raise ExecutorFailure("task archive member parent is not a directory")

            top_levels = sorted(
                {
                    pathlib.PurePosixPath(relative).parts[0]
                    for relative in members
                    if relative != "."
                }
            )
            for top in top_levels:
                try:
                    destination.joinpath(top).lstat()
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    raise ExecutorFailure("task archive destination cannot be inspected") from exc
                raise ExecutorFailure("task archive member would overwrite an existing path")

            # Create all parent directories while no archive symlink exists.
            directory_names = sorted(
                implicit_dirs | {name for name, kind in member_types.items() if kind == "dir"},
                key=lambda name: (len(pathlib.PurePosixPath(name).parts), name),
            )
            for relative in directory_names:
                if relative == ".":
                    continue
                path = _archive_path(root, relative)
                try:
                    info = path.lstat()
                except FileNotFoundError:
                    info = None
                if info is not None:
                    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                        raise ExecutorFailure("task archive directory collides with a non-directory")
                    continue
                path.mkdir()

            # Keep tar stream order; repeated backward seeks make gzip archives
            # unexpectedly expensive.  Only validated regular files reach
            # tarfile, so extract fully trusted wherever the filter API exists:
            # Python 3.14's default ``data`` filter rewrites a 0444 mode to 0644.
            # Runtimes without the API keep the equivalent legacy call.
            trusted = {"filter": "fully_trusted"} if hasattr(tarfile, "data_filter") else {}
            for relative, member in members.items():
                # Directories were created above with writable mode.  Do not
                # let tarfile apply an archive directory mode (for example
                # 0644) before its child files are written.
                if member_types[relative] != "file":
                    continue
                _assert_archive_parent_is_directory(root, relative)
                extracted = copy.copy(member)  # TarInfo uses slots on Python 3.10.
                extracted.name = relative
                tar.extract(extracted, root, **trusted)

            # Links come last, after every regular member, with the archive's
            # exact target: nothing is written below or through them.  A link
            # resolves later in the agent container's namespace, while
            # host-side tools resolve and confine their own reads.
            for relative in sorted(link_targets):
                try:
                    os.symlink(link_targets[relative], _archive_path(root, relative))
                except FileExistsError as exc:
                    raise ExecutorFailure("task archive symlink would overwrite an existing path") from exc

        # Publish only validated top-level entries.  On POSIX use directory
        # descriptors opened with O_NOFOLLOW so a replaced destination path
        # cannot redirect the rename outside the task directory.
        if top_levels:
            _assert_archive_root_is_not_symlink(destination)
            dir_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
            dest_fd: int | None = None
            stage_fd: int | None = None
            publish_parent_fd = None
            try:
                # Open the admitted parent first, then open the destination
                # relative to that descriptor. fstat on both handles closes
                # the pre-open destination/parent replacement window; later
                # renames stay anchored even if the lexical path is swapped.
                publish_parent_fd = os.open(destination.parent, dir_flags)
                parent_info = os.fstat(publish_parent_fd)
                if (
                    int(parent_info.st_dev), int(parent_info.st_ino)
                ) != destination_parent_identity:
                    raise ExecutorFailure("task archive destination parent changed before publish")
                dest_fd = os.open(destination.name, dir_flags, dir_fd=publish_parent_fd)
                dest_info = os.fstat(dest_fd)
                if (
                    int(dest_info.st_dev), int(dest_info.st_ino)
                ) != destination_identity:
                    raise ExecutorFailure("task archive destination changed before publish")
                publish_dir_fd = dest_fd
                # Open staging relative to the already-admitted parent. A
                # lexical open after the parent fstat would reintroduce the
                # parent-replacement race this descriptor path is meant to
                # close.
                stage_fd = os.open(staging.name, dir_flags, dir_fd=publish_parent_fd)
                stage_info = os.fstat(stage_fd)
                if staging_identity is None or (
                    int(stage_info.st_dev), int(stage_info.st_ino)
                ) != staging_identity:
                    raise ExecutorFailure("task archive staging changed before publish")
                for top in top_levels:
                    try:
                        os.stat(top, dir_fd=dest_fd, follow_symlinks=False)
                    except FileNotFoundError:
                        pass
                    else:
                        raise ExecutorFailure("task archive destination changed before publish")
                    # Capture the source inode before the atomic rename. A
                    # post-rename stat can observe a concurrent replacement
                    # and accidentally record that foreign inode as ours.
                    source_info = os.stat(top, dir_fd=stage_fd, follow_symlinks=False)
                    source_identity = (int(source_info.st_dev), int(source_info.st_ino))
                    os.rename(top, top, src_dir_fd=stage_fd, dst_dir_fd=dest_fd)
                    published.append(
                        (destination / top, dest_fd, top, source_identity)
                    )
            finally:
                # Close each descriptor independently: a failure closing one
                # must not leak the other into the worker process.
                if stage_fd is not None:
                    try:
                        os.close(stage_fd)
                    except OSError:
                        pass
                if dest_fd is not None and dest_fd != publish_dir_fd:
                    try:
                        os.close(dest_fd)
                    except OSError:
                        pass
    except ExecutorFailure:
        rollback()
        raise
    except Exception as exc:
        rollback()
        raise ExecutorFailure("task archive extraction failed") from exc
    finally:
        cleanup_error: Exception | None = None
        try:
            if staging is not None:
                try:
                    cleanup_flags = (
                        os.O_RDONLY
                        | getattr(os, "O_DIRECTORY", 0)
                        | getattr(os, "O_NOFOLLOW", 0)
                    )
                    cleanup_parent_fd = publish_parent_fd
                    owns_cleanup_parent_fd = False
                    if cleanup_parent_fd is None:
                        cleanup_parent_fd = os.open(staging.parent, cleanup_flags)
                        owns_cleanup_parent_fd = True
                    try:
                        parent_info = os.fstat(cleanup_parent_fd)
                        if (
                            int(parent_info.st_dev),
                            int(parent_info.st_ino),
                        ) != destination_parent_identity:
                            raise ExecutorFailure("task archive staging parent changed during cleanup")
                        current = os.stat(
                            staging.name,
                            dir_fd=cleanup_parent_fd,
                            follow_symlinks=False,
                        )
                        if staging_identity is not None and (
                            int(current.st_dev),
                            int(current.st_ino),
                        ) != staging_identity:
                            raise ExecutorFailure("task archive staging directory was replaced")
                        _remove_archive_entry_at(
                            cleanup_parent_fd,
                            staging.name,
                            expected_identity=staging_identity,
                        )
                    finally:
                        if owns_cleanup_parent_fd:
                            os.close(cleanup_parent_fd)
                except OSError as exc:  # pragma: no cover - filesystem failure
                    cleanup_error = ExecutorFailure("task archive staging cleanup failed")
                    cleanup_error.__cause__ = exc
                except Exception as exc:  # preserve typed cleanup failures
                    cleanup_error = exc
        finally:
            # Keep the admitted directory descriptors alive through staging
            # cleanup; close each independently even if one close reports an
            # OS error. The outer ``finally`` guarantees both attempts happen
            # even when cleanup itself raises.
            if publish_dir_fd is not None:
                try:
                    os.close(publish_dir_fd)
                except OSError:  # pragma: no cover - descriptor cleanup
                    pass
            if publish_parent_fd is not None:
                try:
                    os.close(publish_parent_fd)
                except OSError:  # pragma: no cover - descriptor cleanup
                    pass
        if cleanup_error is not None:
            raise cleanup_error


class CyberGymExecutor(_DockerRuntimeMixin, _LifecycleMixin, _ReconcileMixin, _CustodyMixin):
    """Run one task at a time against a campaign-owned sidecar."""

    def __init__(self, config: ExecutorConfig) -> None:
        self.config = config
        self.host = resolve_rootless_docker_host(config.docker_host)
        self.server_name = f"cybergym-server-{config.campaign_id}"
        self.server_url = ""
        self.network_id = ""
        self._network_created = False
        self.server_id = ""
        self.started = False
        # Keep the immutable Docker ids alongside names.  Names are mutable
        # handles and are not sufficient custody evidence after a daemon
        # restart or a concurrent name collision.
        self._task_containers: dict[str, str] = {}
        self._terminal_uncommitted_workspaces: dict[str, dict[str, str]] = {}
        self._server_observation: Mapping[str, Any] | None = None
        self._server_image_observation: Mapping[str, Any] | None = None
        self._workspace_image_observation: Mapping[str, Any] | None = None
        self._workspace_observations: dict[str, Mapping[str, Any]] = {}
        self._sidecar_attestation: dict[str, Any] = {}
        self._plans: dict[str, NetworkPlan] = {}
        # Docker attaches a container to the network before ``docker run``
        # returns its id.  The condition tracks that short pending-start window;
        # it lets all lanes execute ``docker run`` concurrently while making an
        # attestation wait until every started container has immutable custody.
        self._registry_lock = threading.RLock()
        self._registry_condition = threading.Condition(self._registry_lock)
        self._workspace_starting: dict[str, int] = {}
        self._unresolved_workspace_custody: dict[str, str] = {}
        # One healer pass at a time: a dispatch probe and every _workspace lane
        # can enter it at once.  Taken before the registry lock and never under
        # it, so a pass serializes healers without blocking parallel starts.
        self._workspace_healer_lock = threading.Lock()
        # Gateway ids are registered before the admission POST and retained
        # until a settled status is observed.  This is the custody boundary:
        # a transport error after the server accepted a task must not let
        # ``close`` reap the workspace while its paid worker is still alive.
        self._gateway_attempts: dict[str, dict[str, Any]] = {}
        self._custody_blocked = False
        # Reconcile mode attaches to resources created by an earlier launcher
        # process; close() must detach rather than remove them.
        self._adopted = False
        self._staged_mask_map: pathlib.Path | None = None
        self._start_lock = threading.Lock()
        self.settings_observation: dict[str, Any] = {"status": "not_checked"}
        self.provider_observation: dict[str, Any] = {"required": bool(config.provider_probe), "status": "not_run"}
        self.data_observation: dict[str, Any] = {"status": "not_checked"}
        self.binary_observation: dict[str, Any] = {"status": "not_checked"}
        self.daemon_observation: dict[str, Any] = {"status": "not_checked"}

    def _generate(self, task: TaskSpec, task_dir: pathlib.Path, agent_id: str) -> str:
        argv = build_generate_task_argv(
            task.task_id,
            out_dir=task_dir,
            data_dir=self.config.data_root,
            server=self._network_plan(task.task_id).server_url,
            mask_map=self.config.mask_map,
            difficulty=self.config.difficulty,
            python=self.config.python_executable,
            agent_id=agent_id,
        )
        result = self.config.command_runner(
            argv, cwd=self.config.source_root, env=_minimal_child_env(self.host), timeout=600
        )
        if result.returncode != 0:
            raise ExecutorFailure("official CyberGym task generation failed")
        expected = ("repo-vul.tar.gz", "description.txt", "README.md", "submit.sh")
        if any(not (task_dir / name).is_file() for name in expected):
            raise ExecutorFailure("generator did not produce the complete Level-1 task")
        _safe_extract(task_dir / "repo-vul.tar.gz", task_dir)
        (task_dir / "submissions").mkdir(exist_ok=True)
        # Ouroboros' external-workspace admission deliberately accepts only a
        # git worktree root.  The pinned CyberGym generator emits a plain
        # directory, so create adapter-owned metadata after generation.  The
        # tiny anchor tracks only task-control files; immutable benchmark input
        # remains excluded from patch authorship and is not duplicated as Git
        # objects for every task.
        return _initialize_generated_workspace_git(
            task_dir,
            runner=self.config.command_runner,
            host=self.host,
        )

    def _attest_runtime(
        self,
        task: TaskSpec,
        attempt_id: str,
        plan: NetworkPlan,
        workspace_name: str,
        api_key: str,
    ) -> dict[str, Any]:
        """Run the complete sidecar custody/connectivity gate before gateway dispatch."""
        # Docker publishes a container on the network before ``docker run``
        # returns its id.  Wait for all pending starts to publish immutable
        # custody, then snapshot/inspect under the registry lock.  The lock is
        # not held while those starts execute Docker, so task lanes remain
        # concurrent.
        with self._registry_condition:
            while self._workspace_starting:
                self._registry_condition.wait()
            if self._unresolved_workspace_custody:
                # A sibling's start is unresolved: this zero-send attempt is
                # requeued; the dispatcher's probe heals outside this lock.
                raise WorkspaceCustodyPending(self._unresolved_workspace_custody)
            cached_server = self._server_observation
            cached_workspace = self._workspace_observations.get(workspace_name)
            if not isinstance(cached_server, Mapping) or not isinstance(cached_workspace, Mapping):
                raise ExecutorFailure("sidecar observations are incomplete")
            # Names are only startup handles.  Re-inspect the immutable ids at
            # the trust boundary immediately before the gateway POST so a replacement
            # container, restart, or daemon mix-up cannot inherit an old attestation.
            server_id = str(cached_server.get("Id") or self.server_id).strip()
            workspace_id = str(
                cached_workspace.get("Id") or self._task_containers.get(workspace_name) or ""
            ).strip()
            if not server_id or not workspace_id:
                raise ExecutorFailure("sidecar observations omitted immutable container ids")
            server = self._inspect("container", server_id)
            workspace = self._inspect("container", workspace_id)
            network = self._inspect("network", self.network_id)
            if str(server.get("Id") or "").strip() != server_id:
                raise ExecutorFailure("server container identity changed before attestation")
            if str(workspace.get("Id") or "").strip() != workspace_id:
                raise ExecutorFailure("workspace container identity changed before attestation")
            if (
                str(network.get("Id") or "").strip() != self.network_id
                or network.get("Name") != "cybergym-internal"
                or network.get("Internal") is not False
                or network.get("Driver") != "bridge"
            ):
                raise ExecutorFailure("CyberGym network identity changed before attestation")
            network_labels = network.get("Labels") if isinstance(network.get("Labels"), Mapping) else {}
            if network_labels.get("com.ouroboros.campaign") != self.config.campaign_id:
                raise ExecutorFailure("CyberGym network ownership changed before attestation")
            attached = network.get("Containers")
            if isinstance(attached, Mapping):
                known_ids = {server_id, workspace_id, *self._task_containers.values()}
                if any(str(item) not in known_ids for item in attached):
                    raise ExecutorFailure("CyberGym network gained an unknown container")
            for role, container in (("server", server), ("workspace", workspace)):
                all_networks = ((container.get("NetworkSettings") or {}).get("Networks") or {})
                if not isinstance(all_networks, Mapping) or set(all_networks) != {"cybergym-internal"}:
                    raise ExecutorFailure(f"{role} has an unexpected network attachment")
            cached_server_pid = _pid_from_observation(cached_server)
            fresh_server_pid = _pid_from_observation(server)
            cached_workspace_pid = _pid_from_observation(cached_workspace)
            fresh_workspace_pid = _pid_from_observation(workspace)
            if cached_server_pid and fresh_server_pid and cached_server_pid != fresh_server_pid:
                raise ExecutorFailure("server process identity changed before attestation")
            if cached_workspace_pid and fresh_workspace_pid and cached_workspace_pid != fresh_workspace_pid:
                raise ExecutorFailure("workspace process identity changed before attestation")
            self._server_observation = server
            self._workspace_observations[workspace_name] = workspace
        # Bind image-level manifest digests to the actual container image id/ref
        # before handing the redacted projections to the generic attestor.
        server_projection = dict(server)
        workspace_projection = dict(workspace)
        server_projection = _bind_container_image(
            server_projection,
            self._server_image_observation,
            self.config.server_image_digest,
            "server",
        )
        workspace_projection = _bind_container_image(
            workspace_projection,
            self._workspace_image_observation,
            self.config.workspace_image_digest,
            "workspace",
        )
        expected = SidecarExpectation(
            plan,
            self.host,
            self.server_name,
            workspace_name,
            server_id,
            workspace_id,
            self.network_id,
            self.host.socket_path,
            server_pid=_pid_from_observation(server_projection),
            workspace_pid=_pid_from_observation(workspace_projection),
            server_image_digest=self.config.server_image_digest,
            workspace_image_digest=self.config.workspace_image_digest,
            publish_host_port=False,
        )
        connectivity = self._connectivity_observation(plan, workspace_id, api_key)
        observation = {
            "docker_host": self.host.value,
            "docker_info": dict(self.daemon_observation),
            "network": network,
            "server": server_projection,
            "workspace": workspace_projection,
            "executor_network": EXECUTOR_NETWORK_DECLARATION,
        }
        security_failure = ""
        try:
            report = attest_sidecar_runtime(
                observation,
                expected,
                api_key=api_key,
                connectivity=connectivity,
                require_daemon_evidence=bool(self.config.provider_probe),
                require_protected_route_evidence=bool(self.config.provider_probe),
            )
            hidden = connectivity.get("agent_hidden_artifacts")
            hidden_ok = isinstance(hidden, Mapping) and all(value is True for value in hidden.values())
            secret_env_ok = connectivity.get("agent_secret_env_absent") is True
            probe_tools_ok = connectivity.get("agent_probe_tools") is True
            if not hidden_ok or not secret_env_ok or not probe_tools_ok:
                failed = list(report.get("failed_checks") or [])
                if not hidden_ok:
                    failed.append("connectivity.agent_hidden_artifacts")
                if not secret_env_ok:
                    failed.append("connectivity.agent_secret_env_absent")
                if not probe_tools_ok:
                    failed.append("connectivity.agent_probe_tools")
                report = {
                    **dict(report),
                    "ok": False,
                    "failed_checks": sorted(set(failed)),
                }
                security_failure = "workspace can see a protected CyberGym artifact or secret"
        except Exception as exc:
            report = getattr(exc, "report", None)
            if isinstance(report, Mapping):
                self._sidecar_attestation = dict(report)
                _write_json(
                    safe_task_path(self.config.run_root / "attestations", task.task_id, attempt_id)
                    / "sidecar_attestation.json",
                    dict(report),
                )
            raise ExecutorFailure("CyberGym sidecar runtime attestation failed") from exc
        self._sidecar_attestation = dict(report)
        _write_json(
            safe_task_path(self.config.run_root / "attestations", task.task_id, attempt_id)
            / "sidecar_attestation.json",
            dict(report),
        )
        if security_failure:
            raise ExecutorFailure(security_failure)
        return dict(report)

    def acknowledge_result_durable(self, task_id: str, attempt_id: str) -> None:
        """Release terminal workspace custody after row and ledger durability."""
        plan = self._plans.get(str(attempt_id))
        if plan is None:
            return
        container_name = f"cybergym-workspace-{plan.opaque_agent_id}"
        with self._registry_condition:
            pending = self._terminal_uncommitted_workspaces.get(container_name)
            if pending is None:
                return
            if not isinstance(pending, Mapping):
                raise ExecutorFailure("durability acknowledgement custody is malformed")
            if (
                pending.get("task_id") != str(task_id)
                or pending.get("attempt_id") != str(attempt_id)
            ):
                raise ExecutorFailure("durability acknowledgement identity mismatch")
            self._terminal_uncommitted_workspaces.pop(container_name, None)

    def run_task(self, task: TaskSpec, task_dir: pathlib.Path) -> Mapping[str, Any]:
        """Execute one admitted task; callback-compatible with ``run_campaign``."""

        self.start()
        attempt_id = str(task.metadata.get("attempt_id") or uuid.uuid4().hex)
        agent_id = make_opaque_agent_id(self.config.campaign_id, task.task_id, attempt_id)
        plan = self._task_network_plan(task.task_id, agent_id)
        self._plans[attempt_id] = plan
        task_dir = _safe_abs(task_dir, "task_dir")
        _inside(task_dir, _safe_abs(self.config.run_root, "run_root"), "task_dir")
        workspace_dir = self._opaque_workspace_path(agent_id)
        workspace_dir.mkdir(parents=True, exist_ok=True)
        container_name = f"cybergym-workspace-{plan.opaque_agent_id}"
        gateway_admission_started = False
        gateway_admission_rejected = False
        gateway_settled = False
        terminal_runtime_result: dict[str, Any] = {}
        terminal_evidence: dict[str, Any] = {}
        attestation_ref = ""
        # A retry has a distinct upstream agent/gateway identity.  Keep its
        # checkpoint under the same attempt component so a late result from an
        # earlier attempt cannot overwrite the custody record we need to
        # reattach to.
        checkpoint = (
            safe_task_path(self.config.run_root / "checkpoints", task.task_id, attempt_id)
            / "gateway_checkpoint.json"
        )
        cleanup_ref = safe_task_path(
            self.config.run_root / "attestations", task.task_id, attempt_id
        ) / "workspace_cleanup.json"
        alias_ref = safe_task_path(
            self.config.run_root / "attestations", task.task_id, attempt_id
        ) / "workspace_backend_alias.json"
        try:
            workspace_anchor = str(self._generate(task, workspace_dir, agent_id) or "")
            _install_workspace_backend_alias(workspace_dir)
            # Keep the topology change explicit in host-private run evidence.
            # This records an alias, never PoC bytes or a post-run promotion.
            # Later setup failures intentionally leave the alias in this
            # attempt's opaque workspace; path-based rollback could delete a
            # child replacement and is therefore never attempted.
            _write_json(
                alias_ref,
                {
                    "schema": _WORKSPACE_BACKEND_ALIAS_SCHEMA,
                    "status": "installed",
                    "workspace_root": str(workspace_dir),
                    "alias_path": _WORKSPACE_BACKEND_ALIAS_NAME,
                    "alias_target": _WORKSPACE_BACKEND_ALIAS_TARGET,
                    "backend_path": "/workspace",
                    "same_root": True,
                    "git_input_anchor": workspace_anchor or None,
                    "git_tracked_inputs": list(_GENERATED_TRACKED_INPUTS),
                    "git_ignored_inputs": list(_GENERATED_INPUT_EXCLUDES),
                },
            )
            container_name = self._workspace(task, workspace_dir, plan)
            sidecar_attestation = {
                "status": "not_run",
                "reason": "provider_probe_disabled",
            }
            if self.config.provider_probe:
                sidecar_attestation = self._attest_runtime(
                    task,
                    attempt_id,
                    plan,
                    container_name,
                    self._ensure_key(),
                )
                attestation_ref = str(
                    safe_task_path(self.config.run_root / "attestations", task.task_id, attempt_id)
                    / "sidecar_attestation.json"
                )
            body = self._task_body(task, workspace_dir, container_name, attempt_id)
            # Checkpoints and verifier responses are host-private.  Keeping them
            # beside the mounted task files would let a still-running agent read
            # server ids, raw exits, or another task's diagnostics.
            gateway_admission_started = True
            gateway_result = self._gateway_wait(
                body,
                checkpoint,
                workspace_name=container_name,
                task_id=task.task_id,
                attempt_id=attempt_id,
            )
            gateway_settled = True
            terminal_runtime_result = dict(gateway_result)
            return self._deliver_gateway_result(
                task,
                task_dir,
                workspace_dir,
                container_name,
                agent_id,
                gateway_result,
                checkpoint=checkpoint,
                cleanup_ref=cleanup_ref,
                alias_ref=alias_ref,
                attestation_ref=attestation_ref,
                sidecar_attestation=sidecar_attestation,
                terminal_evidence=terminal_evidence,
            )
        except WorkspaceCustodyPending:
            # Collateral of a sibling's unresolved start, before any gateway
            # send: no row.  ``run_campaign`` releases the claim for a requeue.
            raise
        except Exception as exc:
            if not gateway_admission_started or isinstance(
                exc, GatewayAdmissionRejected
            ):
                gateway_admission_rejected = isinstance(
                    exc, GatewayAdmissionRejected
                )
                artifact_refs = {
                    "task_dir": str(task_dir),
                    "workspace_dir": str(workspace_dir),
                    "checkpoint": str(checkpoint),
                    "workspace_backend_alias": str(alias_ref),
                    "workspace_cleanup": str(cleanup_ref),
                }
                if attestation_ref:
                    artifact_refs["sidecar_attestation"] = attestation_ref
                return {
                    "status": "infra_failed",
                    "lifecycle": (
                        "gateway_admission_rejected"
                        if gateway_admission_rejected
                        else "pre_gateway_setup_failed"
                    ),
                    "infra_reason": type(exc).__name__,
                    "cost_usd": 0.0,
                    "cost_estimated": False,
                    "cost_final": True,
                    "cost_status": "known_no_dispatch",
                    "artifact_refs": artifact_refs,
                    "error": str(exc),
                }
            if not gateway_settled or not terminal_runtime_result:
                raise
            artifact_refs = {
                "task_dir": str(task_dir),
                "workspace_dir": str(workspace_dir),
                "checkpoint": str(checkpoint),
                "workspace_backend_alias": str(alias_ref),
                "workspace_cleanup": str(cleanup_ref),
            }
            if attestation_ref:
                artifact_refs["sidecar_attestation"] = attestation_ref
            return {
                "runtime_result": terminal_runtime_result,
                **terminal_evidence,
                "status": "infra_failed",
                "lifecycle": "post_gateway_evaluation_failed",
                "infra_reason": type(exc).__name__,
                "artifact_refs": artifact_refs,
                "error": str(exc),
            }
        finally:
            # Once gateway admission starts, keep the exact workspace through
            # the outer result-row and claim-settlement transaction. Campaign
            # close reaps terminal workspaces only after run_campaign returns;
            # unresolved custody remains available to reconcile.
            with self._registry_lock:
                has_exact_id = bool(container_name and self._task_containers.get(container_name))
            cleanup_safe = (
                not gateway_admission_started
                or gateway_admission_rejected
            )
            if has_exact_id and cleanup_safe:
                try:
                    self._cleanup_workspace_container(
                        container_name, task.task_id, attempt_id, cleanup_ref
                    )
                except Exception as cleanup_exc:
                    # Cleanup health remains explicit campaign evidence, but it
                    # must not erase a terminal gateway result and its exact
                    # provider charge before the outer ledger can settle it.
                    try:
                        _write_json(
                            cleanup_ref,
                            {
                                "schema": "ouroboros.benchmark.cybergym.workspace_cleanup.v1",
                                "status": "failed",
                                "ok": False,
                                "error_type": type(cleanup_exc).__name__,
                                "container_name": container_name,
                            },
                        )
                    except Exception:
                        pass

    def __enter__(self) -> "CyberGymExecutor":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def build_executor(config: ExecutorConfig) -> CyberGymExecutor:
    """Factory used by the launcher after admission."""

    return CyberGymExecutor(config)


__all__ = [
    "CommandResult",
    "ExecutorConfig",
    "ExecutorFailure",
    "GatewayTransportError",
    "CyberGymExecutor",
    "build_executor",
    "run_command",
    "urllib_json",
]
