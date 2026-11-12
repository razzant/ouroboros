"""The browser tools' one Chromium: a review-pinned Chrome for Testing headless shell.

``tools/browser.py`` launches only this exact build, by its executable path, inside
Chromium's own sandbox unless an administrator sets ``OUROBOROS_CHROMIUM_SANDBOX=off``.
The desktop build scripts and the Docker image install it with
``python -m ouroboros.chromium_runtime install``; a source checkout installs it on the
first Chromium use. Playwright's own Chromium build is never launched for browser
tools, and nothing falls back to an older or unsandboxed browser: a missing,
unverifiable, unsupported or unsandboxable Chromium is a typed ``ChromiumRefusal``.

A browsers root has Playwright's ``PLAYWRIGHT_BROWSERS_PATH`` meaning (a directory,
or ``0`` for the installed Playwright package's ``.local-browsers``). It holds the pin
as ``ouroboros-cft-<version>-<platform>/``, published by one directory rename with its
``install.json`` inside; Playwright's browser garbage collector deletes only its own
name prefixes, never this one. Standard library only at import, so the Docker build
stage can run it beside ``verified_download.py`` alone.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import pathlib
import platform
import shutil
import stat
import sys
import uuid
import zipfile
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping, Optional

CHROMIUM_VERSION = "157.0.8103.50"
SANDBOX_ENV = "OUROBOROS_CHROMIUM_SANDBOX"
UNAVAILABLE = "BROWSER_CHROMIUM_UNAVAILABLE"
PREPARING = "BROWSER_PREPARING"  # a browser engine is still being installed beside the calls
SANDBOX_UNAVAILABLE = "BROWSER_CHROMIUM_SANDBOX_UNAVAILABLE"
RECEIPT_NAME = "install.json"
_DOWNLOAD_BASE = "https://storage.googleapis.com/chrome-for-testing-public"
_DOWNLOAD_TIMEOUT_SEC = 120.0
_NO_FALLBACK = ("Ouroboros never substitutes an older or unsandboxed Chromium; "
                "engine='webkit' stays available where it is installed.")
_SANDBOX_ON = frozenset({"", "1", "true", "yes", "on"})
_SANDBOX_OFF = frozenset({"0", "false", "no", "off"})
# Fatal lines Chromium prints when its sandbox cannot start: Linux as root, without
# usable user namespaces or a configured setuid helper; macOS seatbelt; Windows broker.
_SANDBOX_FAILURE_MARKERS = (
    "running as root without --no-sandbox",
    "no usable sandbox",
    "failed to move to new namespace",
    "suid sandbox",
    "credentials.cc",
    "sandbox initialization failed",
    "seatbelt",
    "sandbox cannot access executable",
)


class ChromiumRefusal(RuntimeError):
    """Typed refusal of a Chromium launch: nothing is left running or substituted."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code

    def tool_text(self) -> str:
        return f"⚠️ {self.code}: {self}"


@dataclass(frozen=True)
class ChromiumPin:
    """One official Chrome for Testing headless-shell archive of ``CHROMIUM_VERSION``."""

    platform: str
    archive_size: int
    archive_sha256: str
    macos_minimum: tuple[int, ...] = ()  # the executable's LC_BUILD_VERSION minos

    @property
    def folder(self) -> str:
        """The archive's single top-level folder."""
        return f"chrome-headless-shell-{self.platform}"

    @property
    def url(self) -> str:
        return f"{_DOWNLOAD_BASE}/{CHROMIUM_VERSION}/{self.platform}/{self.folder}.zip"

    @property
    def directory(self) -> str:
        return f"ouroboros-cft-{CHROMIUM_VERSION}-{self.platform}"

    @property
    def executable(self) -> str:
        return "chrome-headless-shell.exe" if self.platform.startswith("win") else "chrome-headless-shell"

    def receipt(self) -> dict:
        return {"product": "chrome-headless-shell", "version": CHROMIUM_VERSION, "platform": self.platform,
                "url": self.url, "archive_size": self.archive_size, "archive_sha256": self.archive_sha256}


PINS: Mapping[str, ChromiumPin] = MappingProxyType({
    "linux-arm64": ChromiumPin("linux-arm64", 128169705,
                               "55add1fa634756811fd644094ec226fc7e405be698fb07dbf0634bca689e293a"),
    "linux64": ChromiumPin("linux64", 128166087,
                           "7bf0d41e424efd1d6281c4162993ab9dc5de92fac213c211614905a71c057d0f"),
    "mac-arm64": ChromiumPin("mac-arm64", 104775379,
                             "082e3fe8fcfac75aec2ef0949d70e8d2254a99ea465450e287e6c06a3f1863ed", (13, 0)),
    "mac-x64": ChromiumPin("mac-x64", 110442622,
                           "886beb07d8237b8acc2b879843a3506b3a56284264b82c806c7d80546e24e0ad", (13, 0)),
    "win32": ChromiumPin("win32", 120390768,
                         "7622162c88977542696559825d7611bb305f6236bb8505dfd5b449316ee68fe2"),
    "win64": ChromiumPin("win64", 126512350,
                         "9e59432baf484387bdc9cc4a8efca499fb558aeda93e536eb8b859a065ddb884"),
})


def host_pin(*, system: Optional[str] = None, machine: Optional[str] = None) -> ChromiumPin:
    """The pin for this operating system and architecture (Windows on Arm runs win64)."""
    system = str(system or sys.platform)
    machine = str(machine or platform.machine()).strip().lower()
    arm, x64 = machine in {"arm64", "aarch64"}, machine in {"x86_64", "amd64", "x64"}
    if system == "darwin":
        key = "mac-arm64" if arm else "mac-x64" if x64 else ""
    elif system.startswith("linux"):
        key = "linux-arm64" if arm else "linux64" if x64 else ""
    elif system in {"win32", "cygwin"}:
        key = "win64" if arm or x64 else "win32" if machine in {"x86", "i386", "i686"} else ""
    else:
        key = ""
    if not key:
        raise ChromiumRefusal(UNAVAILABLE, f"no Chrome for Testing {CHROMIUM_VERSION} build is pinned for "
                                           f"{system}/{machine or 'unknown'}. {_NO_FALLBACK}")
    return PINS[key]


def require_supported_host(pin: ChromiumPin, *, mac_release: Optional[str] = None) -> None:
    """Refuse a macOS older than the build's own minimum; an unreadable version is not a refusal."""
    if not pin.macos_minimum:
        return
    release = platform.mac_ver()[0] if mac_release is None else mac_release
    try:
        current = tuple(int(part) for part in (release.split(".") + ["0"])[:2])
    except ValueError:
        return
    if current < pin.macos_minimum:
        needed = ".".join(str(part) for part in pin.macos_minimum)
        raise ChromiumRefusal(UNAVAILABLE, f"the pinned Chromium {CHROMIUM_VERSION} needs macOS {needed} or "
                                           f"later; this Mac runs {release}. {_NO_FALLBACK}")


def browsers_root(value: str) -> Optional[pathlib.Path]:
    """A ``PLAYWRIGHT_BROWSERS_PATH`` value as a directory; None when unset or unresolvable."""
    value = str(value or "").strip()
    if value != "0":
        return pathlib.Path(value) if value else None
    try:
        import playwright
    except ImportError:
        return None
    return pathlib.Path(playwright.__file__).parent / "driver" / "package" / ".local-browsers"


def installed_executable(root: "str | pathlib.Path", pin: ChromiumPin) -> Optional[pathlib.Path]:
    """The exact pin's executable under one browsers root, or None: anything else is not it."""
    directory = pathlib.Path(root) / pin.directory
    try:
        receipt = json.loads((directory / RECEIPT_NAME).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None
    executable = directory / pin.executable
    return executable if receipt == pin.receipt() and executable.is_file() else None


def find_installed(roots: Iterable["str | pathlib.Path"], pin: ChromiumPin) -> Optional[pathlib.Path]:
    for root in roots:
        found = installed_executable(root, pin)
        if found is not None:
            return found
    return None


def _verifier(pin: ChromiumPin):
    from ouroboros.verified_download import verify_exact_file

    return functools.partial(verify_exact_file, size_bytes=pin.archive_size, sha256=pin.archive_sha256,
                             code_prefix="chromium_archive",
                             label=f"Chrome for Testing {CHROMIUM_VERSION} {pin.platform} archive")


def _acquire(pin: ChromiumPin, archive: Optional[pathlib.Path], download_to: pathlib.Path,
             progress: Optional[dict] = None) -> pathlib.Path:
    from ouroboros.verified_download import VerifiedDownloadError, fetch_exact_file

    try:
        if archive is not None:
            return _verifier(pin)(archive)
        return fetch_exact_file(url=pin.url, destination=download_to, verify=_verifier(pin),
                                size_bytes=pin.archive_size, overflow_code="chromium_archive_oversize",
                                failure_code="chromium_archive_download_failed",
                                label=f"Chrome for Testing {CHROMIUM_VERSION} {pin.platform} archive",
                                timeout_sec=_DOWNLOAD_TIMEOUT_SEC, progress=progress)
    except VerifiedDownloadError as exc:
        raise ChromiumRefusal(UNAVAILABLE, f"{exc} ({exc.code}). {_NO_FALLBACK}") from exc


def _extract(pin: ChromiumPin, archive: pathlib.Path, staging: pathlib.Path) -> None:
    """Unpack the archive's one folder into ``staging``; links and escaping names refuse."""
    prefix = pin.folder + "/"
    staging.mkdir(parents=True)
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            relative = info.filename[len(prefix):] if info.filename.startswith(prefix) else None
            mode = (info.external_attr >> 16) & 0xFFFF
            parts = pathlib.PurePosixPath(relative or "").parts
            if (relative is None or "\\" in info.filename or relative.startswith("/")
                    or any(part in {".", ".."} for part in parts) or stat.S_ISLNK(mode)):
                raise ChromiumRefusal(UNAVAILABLE, f"the verified archive holds an unexpected member "
                                                   f"{info.filename!r}; nothing was installed")
            if not parts:
                continue
            target = staging.joinpath(*parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as source, open(target, "xb") as sink:
                shutil.copyfileobj(source, sink, 1024 * 1024)
            if os.name != "nt":
                os.chmod(target, 0o755 if mode & 0o111 else 0o644)
    if not (staging / pin.executable).is_file():
        raise ChromiumRefusal(UNAVAILABLE, f"the verified archive lacks {pin.folder}/{pin.executable}")


def _publish(staging: pathlib.Path, final: pathlib.Path, pin: ChromiumPin) -> None:
    """One rename makes the complete pin visible; a concurrent identical install may win."""
    try:
        os.rename(staging, final)
        return
    except OSError:
        if installed_executable(final.parent, pin) is not None:
            return
    # An incomplete directory under the pin's own name: replace it, never use it.
    stale = final.with_name(f".{final.name}.{uuid.uuid4().hex}.stale")
    os.rename(final, stale)
    try:
        os.rename(staging, final)
    finally:
        shutil.rmtree(stale, ignore_errors=True)


def install(root: "str | pathlib.Path", pin: Optional[ChromiumPin] = None, *,
            archive: "str | pathlib.Path | None" = None, progress: Optional[dict] = None) -> pathlib.Path:
    """Install the exact pin under ``root`` (verified ``archive``, else a download) and return it."""
    pin = pin or host_pin()
    root = pathlib.Path(root)
    found = installed_executable(root, pin)
    if found is not None:
        return found
    attempt = f".{pin.directory}.{os.getpid()}.{uuid.uuid4().hex}"
    download_to, staging = root / f"{attempt}.zip", root / f"{attempt}.partial"
    try:
        root.mkdir(parents=True, exist_ok=True)
        try:
            verified = _acquire(pin, pathlib.Path(archive) if archive is not None else None, download_to, progress)
            _extract(pin, verified, staging)
            (staging / RECEIPT_NAME).write_text(json.dumps(pin.receipt(), indent=1, sort_keys=True) + "\n",
                                                encoding="utf-8")
            _publish(staging, root / pin.directory, pin)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            if archive is None:
                download_to.unlink(missing_ok=True)
    except OSError as exc:
        raise ChromiumRefusal(UNAVAILABLE, f"installing Chromium {CHROMIUM_VERSION} into {root} failed: "
                                           f"{type(exc).__name__}: {exc}. {_NO_FALLBACK}") from exc
    found = installed_executable(root, pin)
    if found is None:
        raise ChromiumRefusal(UNAVAILABLE, f"Chromium {CHROMIUM_VERSION} is still not usable under {root}")
    return found


def sandbox_enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    """On unless the administrator's environment explicitly opts out; unknown values refuse."""
    raw = str((os.environ if environ is None else environ).get(SANDBOX_ENV, "")).strip().lower()
    if raw in _SANDBOX_ON:
        return True
    if raw in _SANDBOX_OFF:
        return False
    raise ChromiumRefusal(SANDBOX_UNAVAILABLE, f"{SANDBOX_ENV}={raw!r} is not a recognized value: leave it unset "
                                               "for the sandbox, or set it to 'off' only as an explicit "
                                               "administrator opt-out.")


def sandbox_refusal(error: BaseException) -> Optional[ChromiumRefusal]:
    """The typed refusal for a launch that failed because the sandbox could not start."""
    text = str(error)
    line = next((row.strip() for row in text.splitlines()
                 if any(marker in row.lower() for marker in _SANDBOX_FAILURE_MARKERS)), "")
    if not line:
        return None
    return ChromiumRefusal(SANDBOX_UNAVAILABLE, (
        f"Chromium could not start inside its sandbox ({line[:300]}). Run Ouroboros as a non-root user "
        "on a host that permits the sandbox (Linux: unprivileged user namespaces, which Ubuntu 23.10+ "
        "AppArmor and Docker's default seccomp profile restrict), or, accepting the risk, an "
        f"administrator sets {SANDBOX_ENV}=off and restarts; launch warnings and browser-condition "
        f"annotations then disclose the opt-out. {_NO_FALLBACK}"))


def conditions(sandbox: bool) -> str:
    """The launch facts a browser result discloses: what was requested, not a per-process audit."""
    state = ("requested at launch (no per-process isolation check)" if sandbox
             else f"disabled by administrator opt-out {SANDBOX_ENV}=off")
    return f"pinned Chrome for Testing headless shell, Chromium sandbox {state}"


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ouroboros.chromium_runtime",
                                     description="Install or locate the pinned Chrome for Testing headless shell.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("url", help="print this host's pinned archive URL")
    installer = commands.add_parser("install", help="install the pin into a browsers root")
    installer.add_argument("--browsers-path", default=None,
                           help="browsers root (default: PLAYWRIGHT_BROWSERS_PATH; 0 = Playwright's .local-browsers)")
    installer.add_argument("--archive", default=None, help="an already downloaded archive, verified against the pin")
    args = parser.parse_args(argv)
    try:
        pin = host_pin()
        if args.command == "url":
            print(pin.url)
            return 0
        selected = args.browsers_path if args.browsers_path is not None else os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "")
        root = browsers_root(selected)
        if root is None:
            parser.error("name a browsers root with --browsers-path or PLAYWRIGHT_BROWSERS_PATH")
        executable = install(root, pin, archive=args.archive)
    except ChromiumRefusal as refusal:
        print(refusal.tool_text(), file=sys.stderr)
        return 1
    print(json.dumps({"version": CHROMIUM_VERSION, "platform": pin.platform, "executable": str(executable)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
