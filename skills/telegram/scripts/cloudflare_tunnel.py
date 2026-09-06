"""Pinned, state-local Cloudflare Quick Tunnel bootstrap for Telegram.

The public API is deliberately small:

* ``ensure_cloudflared(state_dir)`` installs or verifies the pinned binary.
* ``QuickTunnel(binary, state_dir, sidecar_port)`` starts a Quick Tunnel whose
  origin is constructed internally as ``http://127.0.0.1:<sidecar_port>``.

No caller-supplied origin is accepted.  The module never consults ``PATH``, a
package manager, a Cloudflare account, or an existing cloudflared config.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import math
import os
import platform
import re
import secrets
import time
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from platform_support import (
    IS_WINDOWS,
    PlatformSupportError,
    WindowsKillJob,
    acquire_file_lock,
    fsync_directory,
    minimal_process_environment,
    path_is_link_or_reparse,
    release_file_lock,
    secure_executable,
)


CLOUDFLARED_VERSION = "2026.7.2"
CLOUDFLARED_BUILD = "2026-07-15-13:30 UTC"
DOWNLOAD_DEADLINE_SEC = 300  # whole-download bound (see _download_asset)
MAX_ASSET_BYTES = 70 * 1024 * 1024
MAX_LOG_BYTES = 64 * 1024
_DOWNLOAD_HOSTS = frozenset({"github.com", "release-assets.githubusercontent.com"})
_URL_CANDIDATE_RE = re.compile(r"https://[^\s\"'<>\\]+")
_LOG_URL_RE = re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s\"'<>\\]+", re.IGNORECASE)
_LOG_CREDENTIAL_RE = re.compile(
    r"(?i)([\"']?[a-z0-9_.-]*(?:token|secret|password|authorization|cookie|"
    r"api[_-]?key|credential)[a-z0-9_.-]*[\"']?\s*[:=]\s*)"
    r"(?:\"[^\"\r\n]*\"|'[^'\r\n]*'|[^,;}\r\n]+)"
)
_QUICK_HOST_RE = re.compile(
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.trycloudflare\.com"
)
_VERSION_RE = re.compile(
    rf"cloudflared version {re.escape(CLOUDFLARED_VERSION)}"
    rf"(?: \(built [^\r\n]{{1,96}}\))?\r?\n?\Z"
)


class CloudflaredError(RuntimeError):
    """A bootstrap, validation, or tunnel lifecycle failure."""


@dataclass(frozen=True)
class _AssetSpec:
    platform_id: str
    arch: str
    filename: str
    kind: str
    download_size: int
    url: str
    archive_sha256: str
    binary_sha256: str


_ASSETS = {
    ("Darwin", "arm64"): _AssetSpec(
        platform_id="darwin-arm64",
        arch="arm64",
        filename="cloudflared",
        kind="tgz",
        download_size=18_959_119,
        url=(
            "https://github.com/cloudflare/cloudflared/releases/download/"
            "2026.7.2/cloudflared-darwin-arm64.tgz"
        ),
        archive_sha256="2086e51c61d6565781d84117a5007d0c826d03ffdc74acb91c08c167f9f8cd7c",
        binary_sha256="0588df58494a6cadd38b9deb6078908a5054063c80784d92fdb8d4a5f3de1c67",
    ),
    ("Darwin", "amd64"): _AssetSpec(
        platform_id="darwin-amd64",
        arch="amd64",
        filename="cloudflared",
        kind="tgz",
        download_size=20_841_457,
        url=(
            "https://github.com/cloudflare/cloudflared/releases/download/"
            "2026.7.2/cloudflared-darwin-amd64.tgz"
        ),
        archive_sha256="4ee0d3b48a990a2f9b5faec5838f73ec1f400aa8e0a4864be576adfafec406cb",
        binary_sha256="a5afb0ba3da859da47bebc9a918d5b196bf7e4aec23589419b46356731bcc75f",
    ),
    ("Linux", "amd64"): _AssetSpec(
        platform_id="linux-amd64",
        arch="amd64",
        filename="cloudflared",
        kind="raw",
        download_size=39_261_733,
        url=(
            "https://github.com/cloudflare/cloudflared/releases/download/"
            "2026.7.2/cloudflared-linux-amd64"
        ),
        archive_sha256="ec905ea7b7e327ff8abdde8cb64697a2152de74dbcdbf6aec9db8364eb3886cd",
        binary_sha256="ec905ea7b7e327ff8abdde8cb64697a2152de74dbcdbf6aec9db8364eb3886cd",
    ),
    ("Linux", "arm64"): _AssetSpec(
        platform_id="linux-arm64",
        arch="arm64",
        filename="cloudflared",
        kind="raw",
        download_size=36_984_601,
        url=(
            "https://github.com/cloudflare/cloudflared/releases/download/"
            "2026.7.2/cloudflared-linux-arm64"
        ),
        archive_sha256="405df476437e027fc6d18729a5a77155c0a33a6082aeee60a799a688f3052e66",
        binary_sha256="405df476437e027fc6d18729a5a77155c0a33a6082aeee60a799a688f3052e66",
    ),
    ("Windows", "amd64"): _AssetSpec(
        platform_id="windows-amd64",
        arch="amd64",
        filename="cloudflared.exe",
        kind="raw",
        download_size=54_159_760,
        url=(
            "https://github.com/cloudflare/cloudflared/releases/download/"
            "2026.7.2/cloudflared-windows-amd64.exe"
        ),
        archive_sha256="cdb5d4432f6ae1595654a692a51308b69d2bf7af961f5578d9391837cf072df9",
        binary_sha256="cdb5d4432f6ae1595654a692a51308b69d2bf7af961f5578d9391837cf072df9",
    ),
}


def _current_asset() -> _AssetSpec:
    system = platform.system()
    machine = platform.machine().lower()
    arch = {
        "arm64": "arm64",
        "aarch64": "arm64",
        "x86_64": "amd64",
        "amd64": "amd64",
    }.get(machine)
    spec = _ASSETS.get((system, arch or ""))
    if spec is None:
        raise CloudflaredError(
            f"Unsupported cloudflared platform: {system or 'unknown'}/{machine or 'unknown'}."
        )
    return spec


def _sha256_file(path: Path, *, maximum: int = MAX_ASSET_BYTES) -> str:
    digest = hashlib.sha256()
    total = 0
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                total += len(chunk)
                if total > maximum:
                    raise CloudflaredError("cloudflared file exceeds the 70 MiB safety limit.")
                digest.update(chunk)
    except OSError as exc:
        raise CloudflaredError("Could not read the cloudflared file.") from exc
    return digest.hexdigest()


def _safe_state_root(state_dir: Path) -> Path:
    raw = Path(state_dir).expanduser()
    try:
        raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as exc:
        raise CloudflaredError("Could not create the skill state directory.") from exc
    if path_is_link_or_reparse(raw) or not raw.is_dir():
        raise CloudflaredError("The skill state directory must be a real directory, not a link.")
    try:
        return raw.resolve(strict=True)
    except OSError as exc:
        raise CloudflaredError("Could not resolve the skill state directory.") from exc


def _safe_child_dir(root: Path, *parts: str) -> Path:
    current = root
    for part in parts:
        if not part or part in {".", ".."} or "/" in part:
            raise CloudflaredError("Invalid internal cloudflared state path.")
        candidate = current / part
        try:
            candidate.mkdir(mode=0o700, exist_ok=True)
        except OSError as exc:
            raise CloudflaredError("Could not create private cloudflared state.") from exc
        if path_is_link_or_reparse(candidate) or not candidate.is_dir():
            raise CloudflaredError("Private cloudflared state contains an unsafe link.")
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise CloudflaredError("Private cloudflared state escaped the skill state directory.") from exc
        current = resolved
    return current


def _install_dir(root: Path, spec: _AssetSpec) -> Path:
    return _safe_child_dir(root, "cloudflared", CLOUDFLARED_VERSION, spec.platform_id)


def _installed_path(root: Path, spec: _AssetSpec) -> Path:
    return _install_dir(root, spec) / spec.filename


def _validate_download_url(url: str) -> None:
    try:
        parsed = urllib.parse.urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise CloudflaredError("Cloudflared download redirected to an invalid URL.") from exc
    host = (parsed.hostname or "").lower().rstrip(".")
    if (
        parsed.scheme != "https"
        or host not in _DOWNLOAD_HOSTS
        or parsed.username
        or parsed.password
        or port not in (None, 443)
    ):
        raise CloudflaredError("Cloudflared download left the pinned HTTPS host allowlist.")


class _StrictRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: BinaryIO,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        _validate_download_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _download_asset(spec: _AssetSpec, destination: Path) -> None:
    _validate_download_url(spec.url)
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _StrictRedirectHandler(),
    )
    request = urllib.request.Request(
        spec.url,
        headers={
            "Accept": "application/octet-stream",
            "User-Agent": "Ouroboros-Telegram/1",
        },
        method="GET",
    )
    digest = hashlib.sha256()
    total = 0
    # The per-operation socket timeout bounds one stalled read, not a trickling
    # server; the whole download is bounded too, so the private install lock
    # (and the worker thread an async caller cannot cancel) is held at most
    # DOWNLOAD_DEADLINE_SEC.
    deadline = time.monotonic() + DOWNLOAD_DEADLINE_SEC
    try:
        with opener.open(request, timeout=30) as response, destination.open("xb") as output:
            _validate_download_url(response.geturl())
            length_header = response.headers.get("Content-Length")
            if length_header:
                try:
                    announced = int(length_header)
                except ValueError as exc:
                    raise CloudflaredError("Cloudflared download sent an invalid size.") from exc
                if announced != spec.download_size:
                    raise CloudflaredError("Cloudflared download size did not match the pinned asset.")
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_ASSET_BYTES:
                    raise CloudflaredError("Cloudflared download exceeds the 70 MiB safety limit.")
                if time.monotonic() > deadline:
                    raise CloudflaredError(
                        f"Cloudflared download exceeded the {DOWNLOAD_DEADLINE_SEC}s deadline."
                    )
                output.write(chunk)
                digest.update(chunk)
            output.flush()
            os.fsync(output.fileno())
    except CloudflaredError:
        raise
    except (OSError, urllib.error.URLError) as exc:
        raise CloudflaredError(f"Cloudflared download failed ({type(exc).__name__}).") from exc
    if total != spec.download_size or digest.hexdigest() != spec.archive_sha256:
        raise CloudflaredError("Cloudflared archive SHA-256 did not match the pinned release asset.")


def _extract_verified_binary(spec: _AssetSpec, archive: Path, destination: Path) -> None:
    try:
        with tarfile.open(archive, mode="r:gz") as bundle:
            members = bundle.getmembers()
            if len(members) != 1:
                raise CloudflaredError("Cloudflared archive must contain exactly one file.")
            member = members[0]
            if member.name != "cloudflared" or not member.isfile():
                raise CloudflaredError("Cloudflared archive contains an unsafe member.")
            if member.size < 1 or member.size > MAX_ASSET_BYTES:
                raise CloudflaredError("Cloudflared binary exceeds the 70 MiB safety limit.")
            source = bundle.extractfile(member)
            if source is None:
                raise CloudflaredError("Cloudflared archive member could not be read.")
            digest = hashlib.sha256()
            total = 0
            with destination.open("xb") as output:
                while chunk := source.read(1024 * 1024):
                    total += len(chunk)
                    if total > member.size or total > MAX_ASSET_BYTES:
                        raise CloudflaredError("Cloudflared binary exceeded its declared size.")
                    output.write(chunk)
                    digest.update(chunk)
                output.flush()
                os.fsync(output.fileno())
    except CloudflaredError:
        raise
    except (OSError, EOFError, tarfile.TarError) as exc:
        raise CloudflaredError("Cloudflared archive is corrupt or unreadable.") from exc
    if total != member.size or digest.hexdigest() != spec.binary_sha256:
        raise CloudflaredError("Extracted cloudflared SHA-256 did not match the pinned binary.")
    try:
        destination.chmod(0o700)
    except OSError as exc:
        raise CloudflaredError("Could not make the verified cloudflared binary executable.") from exc


def _version_environment(home: Path) -> dict[str, str]:
    return minimal_process_environment(home)


def _verify_version(binary: Path, home: Path) -> None:
    try:
        completed = subprocess.run(
            [str(binary), "--version"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(home),
            env=_version_environment(home),
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CloudflaredError("Could not execute the pinned cloudflared version check.") from exc
    if len(completed.stdout) > 4096 or len(completed.stderr) > 4096:
        raise CloudflaredError("Cloudflared version check produced oversized output.")
    try:
        output = completed.stdout.decode("utf-8", "strict")
    except UnicodeDecodeError as exc:
        raise CloudflaredError("Cloudflared version output was not UTF-8.") from exc
    if completed.returncode != 0 or completed.stderr or not _VERSION_RE.fullmatch(output):
        raise CloudflaredError("Cloudflared did not report the exact pinned version.")


def _verify_binary(binary: Path, spec: _AssetSpec, home: Path) -> None:
    try:
        metadata = binary.lstat()
    except OSError as exc:
        raise CloudflaredError("Pinned cloudflared binary is missing.") from exc
    if path_is_link_or_reparse(binary) or not stat.S_ISREG(metadata.st_mode):
        raise CloudflaredError("Pinned cloudflared path must be a regular file, not a link.")
    if _sha256_file(binary) != spec.binary_sha256:
        raise CloudflaredError("Cached cloudflared SHA-256 did not match the pinned binary.")
    try:
        secure_executable(binary)
    except PlatformSupportError as exc:
        raise CloudflaredError(str(exc)) from exc
    _verify_version(binary, home)


@contextlib.contextmanager
def _install_lock(lock_path: Path):
    try:
        lock_fd = acquire_file_lock(lock_path, timeout_sec=30.0)
    except PlatformSupportError as exc:
        raise CloudflaredError("Could not lock the private cloudflared installation.") from exc
    try:
        yield
    finally:
        release_file_lock(lock_fd)


def _verified_install(target: Path, spec: _AssetSpec, check_home: Path) -> bool:
    """True when the installed path already passes every check (call under the lock)."""
    if not (target.exists() or path_is_link_or_reparse(target)):
        return False
    try:
        _verify_binary(target, spec, check_home)
    except CloudflaredError:
        # Preserve the old path until a complete replacement passes every check.
        return False
    return True


def _fetch_verified_candidate(spec: _AssetSpec, directory: Path, check_home: Path) -> Path:
    """Download, extract and verify the pinned asset into a private candidate file.

    Runs OUTSIDE the install lock on purpose: the network stages are bounded by
    DOWNLOAD_DEADLINE_SEC, and holding the lock across them made every
    concurrent installer wait out somebody else's download. The caller promotes
    the returned path under the lock and unlinks it in every outcome.
    """
    archive = directory / f".asset-{secrets.token_hex(16)}"
    candidate = directory / f".binary-{secrets.token_hex(16)}"
    try:
        if spec.kind == "tgz":
            _download_asset(spec, archive)
            if _sha256_file(archive) != spec.archive_sha256:
                raise CloudflaredError("Downloaded cloudflared archive changed before extraction.")
            _extract_verified_binary(spec, archive, candidate)
        elif spec.kind == "raw":
            _download_asset(spec, candidate)
        else:
            raise CloudflaredError("Pinned cloudflared asset kind is unsupported.")
        _verify_binary(candidate, spec, check_home)
    except BaseException:
        with contextlib.suppress(OSError):
            candidate.unlink(missing_ok=True)
        raise
    finally:
        with contextlib.suppress(OSError):
            archive.unlink(missing_ok=True)
    return candidate


def _ensure_cloudflared_sync(state_dir: Path) -> Path:
    spec = _current_asset()
    root = _safe_state_root(state_dir)
    directory = _install_dir(root, spec)
    target = _installed_path(root, spec)
    check_home = _safe_child_dir(directory, "version-check-home")
    lock_path = directory / ".install.lock"
    with _install_lock(lock_path):
        if _verified_install(target, spec, check_home):
            return target
    # The lock is released across the download (see _fetch_verified_candidate);
    # only the atomic promote below holds it.
    candidate = _fetch_verified_candidate(spec, directory, check_home)
    try:
        with _install_lock(lock_path):
            # A concurrent installer may have promoted its own verified copy in
            # the meantime; a verified install is never replaced.
            if _verified_install(target, spec, check_home):
                return target
            os.replace(candidate, target)
            try:
                fsync_directory(directory)
            except (OSError, PlatformSupportError) as exc:
                raise CloudflaredError("Could not persist the verified cloudflared install.") from exc
            _verify_binary(target, spec, check_home)
            return target
    finally:
        with contextlib.suppress(OSError):
            candidate.unlink(missing_ok=True)


async def ensure_cloudflared(state_dir: Path) -> Path:
    """Install or re-verify cloudflared 2026.7.2 in private skill state."""

    return await asyncio.to_thread(_ensure_cloudflared_sync, Path(state_dir))


def _quick_tunnel_url(text: str) -> str | None:
    for match in _URL_CANDIDATE_RE.finditer(text):
        candidate = match.group(0).rstrip(".,;:!?)]")
        try:
            parsed = urllib.parse.urlsplit(candidate)
            port = parsed.port
        except ValueError:
            continue
        host = (parsed.hostname or "").lower().rstrip(".")
        if (
            parsed.scheme == "https"
            and _QUICK_HOST_RE.fullmatch(host)
            and parsed.netloc == host
            and not parsed.username
            and not parsed.password
            and port is None
            and parsed.path in ("", "/")
            and not parsed.query
            and not parsed.fragment
        ):
            return f"https://{host}/"
    return None


def _safe_log_detail(text: str, maximum: int = 180) -> str:
    """Return one bounded cloudflared line without URLs or credential values."""

    lines = [
        " ".join(line.split())
        for line in str(text or "").splitlines()
        if line.strip()
    ]
    if not lines:
        return ""
    detail = _LOG_URL_RE.sub("[url]", lines[-1])
    detail = _LOG_CREDENTIAL_RE.sub(r"\1[redacted]", detail)
    if len(detail) <= maximum:
        return detail
    return detail[: maximum - 1] + "…"


class _BoundedLog:
    def __init__(self, maximum: int = MAX_LOG_BYTES) -> None:
        self.maximum = maximum
        self._data = bytearray()
        self._scan_tail = ""

    def add(self, chunk: bytes) -> str | None:
        self._data.extend(chunk)
        if len(self._data) > self.maximum:
            del self._data[: len(self._data) - self.maximum]
        decoded = chunk.decode("utf-8", "replace")
        scan = (self._scan_tail + decoded)[-16384:]
        self._scan_tail = scan
        return _quick_tunnel_url(scan)

    def text(self) -> str:
        return bytes(self._data).decode("utf-8", "replace")


class QuickTunnel:
    """Lifecycle wrapper for one Quick Tunnel to a loopback-only sidecar port."""

    def __init__(self, binary: Path, state_dir: Path, sidecar_port: int) -> None:
        if isinstance(sidecar_port, bool) or not isinstance(sidecar_port, int):
            raise CloudflaredError("Sidecar port must be an integer.")
        if not 1 <= sidecar_port <= 65535:
            raise CloudflaredError("Sidecar port is outside the valid TCP range.")
        self.binary = Path(binary)
        self.state_dir = Path(state_dir)
        self.sidecar_port = sidecar_port
        self._process: asyncio.subprocess.Process | None = None
        self._stdout_log = _BoundedLog()
        self._stderr_log = _BoundedLog()
        self._url_future: asyncio.Future[str] | None = None
        self._reader_tasks: list[asyncio.Task[None]] = []
        self._watch_task: asyncio.Task[None] | None = None
        self._watchdog_process: asyncio.subprocess.Process | None = None
        self._watchdog_task: asyncio.Task[None] | None = None
        self._watchdog_write_fd: int | None = None
        self._watchdog_expected = False
        self._watchdog_failed = False
        self._windows_job: WindowsKillJob | None = None
        self._home: Path | None = None
        self._run_root: Path | None = None

    @property
    def stdout_tail(self) -> str:
        return self._stdout_log.text()

    @property
    def stderr_tail(self) -> str:
        return self._stderr_log.text()

    @property
    def returncode(self) -> int | None:
        """Current child exit code, or ``None`` before start and while running."""

        return self._process.returncode if self._process is not None else None

    def _argv(self) -> list[str]:
        return [
            str(self.binary),
            "tunnel",
            "--no-autoupdate",
            "--url",
            f"http://127.0.0.1:{self.sidecar_port}",
        ]

    def _watchdog_script(self) -> Path:
        path = Path(__file__).with_name("tunnel_watchdog.py")
        if path.is_symlink() or not path.is_file():
            raise CloudflaredError("Reviewed tunnel watchdog is missing or unsafe.")
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(Path(__file__).resolve(strict=True).parent)
        except (OSError, ValueError) as exc:
            raise CloudflaredError("Reviewed tunnel watchdog escaped the skill tree.") from exc
        return resolved

    async def _start_watchdog(self, env: dict[str, str]) -> None:
        if IS_WINDOWS:
            raise CloudflaredError("The POSIX tunnel watchdog cannot run on Windows.")
        if self._watchdog_process is not None or self._watchdog_write_fd is not None:
            raise CloudflaredError("Tunnel watchdog has already been started.")
        if self._home is None:
            raise CloudflaredError("Tunnel watchdog has no private runtime home.")
        read_fd, write_fd = os.pipe()
        os.set_inheritable(write_fd, False)
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(self._watchdog_script()),
                str(os.getpid()),
                str(read_fd),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                cwd=str(self._home),
                env=env,
                pass_fds=(read_fd,),
            )
        except BaseException as exc:
            os.close(write_fd)
            if isinstance(exc, OSError):
                raise CloudflaredError("Could not start the tunnel hard-death watchdog.") from exc
            raise
        finally:
            os.close(read_fd)
        self._watchdog_process = process
        self._watchdog_write_fd = write_fd
        self._watchdog_task = asyncio.create_task(self._monitor_watchdog())
        await asyncio.sleep(0)
        if process.returncode is not None:
            await self._disarm_watchdog()
            raise CloudflaredError("Tunnel hard-death watchdog exited during startup.")

    async def _monitor_watchdog(self) -> None:
        assert self._watchdog_process is not None
        returncode = await self._watchdog_process.wait()
        if self._watchdog_expected:
            return
        self._watchdog_failed = True
        if self._url_future is not None and not self._url_future.done():
            self._url_future.set_exception(
                CloudflaredError(f"Tunnel hard-death watchdog stopped unexpectedly (exit {returncode}).")
            )
        process = self._process
        if process is not None and process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.terminate()

    async def _disarm_watchdog(self) -> None:
        if IS_WINDOWS:
            job = self._windows_job
            self._windows_job = None
            if job is not None:
                job.close()
            return
        self._watchdog_expected = True
        write_fd = self._watchdog_write_fd
        self._watchdog_write_fd = None
        if write_fd is not None:
            with contextlib.suppress(OSError):
                os.write(write_fd, b"S")
            with contextlib.suppress(OSError):
                os.close(write_fd)
        process = self._watchdog_process
        if process is not None and process.returncode is None:
            try:
                await asyncio.wait_for(process.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=0.5)
                except asyncio.TimeoutError:
                    with contextlib.suppress(ProcessLookupError):
                        process.kill()
                    await process.wait()
        task = self._watchdog_task
        if task is not None and task is not asyncio.current_task():
            await asyncio.gather(task, return_exceptions=True)

    async def start(self) -> None:
        """Verify the pinned binary and start cloudflared exactly once."""

        if self._process is not None:
            raise CloudflaredError("Quick Tunnel has already been started.")
        spec = _current_asset()
        root = _safe_state_root(self.state_dir)
        expected = _installed_path(root, spec)
        try:
            actual = self.binary.resolve(strict=True)
            canonical = expected.resolve(strict=True)
        except OSError as exc:
            raise CloudflaredError("Pinned cloudflared binary is missing.") from exc
        if actual != canonical or self.binary.is_symlink():
            raise CloudflaredError("Quick Tunnel accepts only the pinned state-local cloudflared binary.")
        self._run_root = _safe_child_dir(root, "cloudflared", "quick-tunnel-runs")
        self._home = Path(
            tempfile.mkdtemp(prefix="home-", dir=self._run_root)
        ).resolve(strict=True)
        try:
            await asyncio.to_thread(_verify_binary, canonical, spec, self._home)
        except BaseException:
            await asyncio.to_thread(self._cleanup_home)
            raise
        self.binary = canonical
        env = _version_environment(self._home)
        try:
            if IS_WINDOWS:
                self._windows_job = WindowsKillJob()
            else:
                await self._start_watchdog(env)
            process_kwargs: dict[str, Any] = {}
            if IS_WINDOWS:
                process_kwargs["creationflags"] = getattr(
                    subprocess,
                    "CREATE_NO_WINDOW",
                    0x08000000,
                )
            self._process = await asyncio.create_subprocess_exec(
                *self._argv(),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._home),
                env=env,
                **process_kwargs,
            )
            if self._windows_job is not None:
                self._windows_job.assign(self._process.pid)
        except BaseException as exc:
            if self._process is not None and self._process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    self._process.terminate()
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(self._process.wait(), timeout=1.0)
            await self._disarm_watchdog()
            await asyncio.to_thread(self._cleanup_home)
            if isinstance(exc, (OSError, PlatformSupportError)):
                raise CloudflaredError("Could not start the pinned cloudflared process.") from exc
            raise
        if not IS_WINDOWS and self._watchdog_failed:
            if self._process.returncode is None:
                self._process.terminate()
                await self._process.wait()
            await self._disarm_watchdog()
            await asyncio.to_thread(self._cleanup_home)
            raise CloudflaredError("Tunnel hard-death watchdog failed before cloudflared startup.")
        loop = asyncio.get_running_loop()
        self._url_future = loop.create_future()
        self._url_future.add_done_callback(self._consume_future_exception)
        assert self._process.stdout is not None and self._process.stderr is not None
        self._reader_tasks = [
            asyncio.create_task(self._read_stream(self._process.stdout, self._stdout_log)),
            asyncio.create_task(self._read_stream(self._process.stderr, self._stderr_log)),
        ]
        self._watch_task = asyncio.create_task(self._watch_process())

    async def _read_stream(self, stream: asyncio.StreamReader, log: _BoundedLog) -> None:
        while chunk := await stream.read(4096):
            url = log.add(chunk)
            if url and self._url_future is not None and not self._url_future.done():
                self._url_future.set_result(url)

    async def _watch_process(self) -> None:
        assert self._process is not None
        returncode = await self._process.wait()
        await asyncio.gather(*self._reader_tasks, return_exceptions=True)
        await self._disarm_watchdog()
        if self._url_future is not None and not self._url_future.done():
            detail = _safe_log_detail(self.stderr_tail) or _safe_log_detail(self.stdout_tail)
            suffix = f" Last output: {detail}" if detail else ""
            self._url_future.set_exception(
                CloudflaredError(
                    f"Cloudflared exited before publishing a URL (exit {returncode}).{suffix}"
                )
            )
        await asyncio.to_thread(self._cleanup_home)

    @staticmethod
    def _consume_future_exception(future: asyncio.Future[str]) -> None:
        if not future.cancelled():
            future.exception()

    def _cleanup_home(self) -> None:
        home = self._home
        run_root = self._run_root
        if home is None or run_root is None:
            return
        try:
            if home.is_symlink() or not home.name.startswith("home-"):
                return
            resolved_home = home.resolve(strict=True)
            resolved_root = run_root.resolve(strict=True)
            resolved_home.relative_to(resolved_root)
            if resolved_home.parent != resolved_root:
                return
            shutil.rmtree(resolved_home)
            self._home = None
        except (OSError, ValueError):
            # Cleanup is best effort and is restricted to the unique state-local run home.
            return

    async def wait_url(self, timeout_sec: float = 30.0) -> str:
        """Wait for one strictly validated ``https://<slug>.trycloudflare.com/`` URL."""

        if self._url_future is None:
            raise CloudflaredError("Quick Tunnel has not been started.")
        if (
            isinstance(timeout_sec, bool)
            or not isinstance(timeout_sec, (int, float))
            or not math.isfinite(float(timeout_sec))
            or timeout_sec <= 0
        ):
            raise CloudflaredError("Quick Tunnel URL timeout must be a positive finite number.")
        try:
            return await asyncio.wait_for(asyncio.shield(self._url_future), float(timeout_sec))
        except asyncio.TimeoutError as exc:
            raise CloudflaredError("Timed out waiting for the Cloudflare Quick Tunnel URL.") from exc

    async def wait(self) -> int:
        """Wait for cloudflared to exit and return its process exit code."""

        if self._process is None:
            raise CloudflaredError("Quick Tunnel has not been started.")
        returncode = await self._process.wait()
        if self._watch_task is not None:
            await self._join_watcher()
        return returncode

    async def _join_watcher(self) -> None:
        """Observe the owned watcher without letting a wrapper cancel it."""

        watch_task = self._watch_task
        if watch_task is None:
            await self._disarm_watchdog()
            return
        # asyncio.wait never propagates the child task's cancellation or
        # exception.  Cancellation of this await is therefore unambiguously
        # caller cancellation, even if caller and watcher are cancelled in the
        # same event-loop tick on the bundled Python 3.10 runtime.
        await asyncio.wait({watch_task})
        if watch_task.cancelled():
            # Recover a watcher cancelled by an older/foreign wrapper.  The
            # process is already stopped here, so only local hard-death and
            # private-home cleanup remain.
            await self._disarm_watchdog()
            await asyncio.to_thread(self._cleanup_home)
            return
        error = watch_task.exception()
        if error is not None:
            raise error

    async def stop(self) -> None:
        """Idempotently terminate cloudflared and disarm its hard-death watchdog."""

        process = self._process
        if process is None:
            return
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=0.75)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        if self._watch_task is not None:
            await self._join_watcher()
        else:
            await self._disarm_watchdog()
        if self._url_future is not None and not self._url_future.done():
            self._url_future.cancel()


__all__ = [
    "CLOUDFLARED_VERSION",
    "CloudflaredError",
    "QuickTunnel",
    "ensure_cloudflared",
]
