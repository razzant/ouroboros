"""Packaged desktop CLI bridge.

This module is invoked by tiny shell/cmd wrappers shipped inside desktop
artifacts. It bootstraps the launcher-managed repo from the packaged bundle,
then delegates to the normal gateway-backed ``ouroboros.cli`` entrypoint.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Iterable, Sequence

# WA6: stop the current process from writing __pycache__/*.pyc into a signed macOS
# bundle BEFORE importing any project module. os.environ alone is insufficient for
# this process (PYTHONDONTWRITEBYTECODE is read only at interpreter startup); only
# sys.dont_write_bytecode stops the current process's own imports. _set_global_
# bytecode_suppression() below additionally sets the env vars for child spawns.
sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

from ouroboros.launch_roots import TUPLE_KEYS, WORK_ROOT_KEYS, configured_roots
from ouroboros.launcher_bootstrap import (
    BootstrapContext, appimage_extracts_and_runs, bootstrap_repo, check_git, embedded_python_env,
)
from ouroboros.platform_layer import (
    BUNDLE_DIR_ENV,
    IS_MACOS,
    IS_WINDOWS,
    embedded_python_candidates,
    git_install_hint,
)
from ouroboros.unindexed import bytecode_prefix


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
START_TIMEOUT_SEC = 90.0


class PackagedCLIError(RuntimeError):
    pass


@dataclass(frozen=True)
class PackagedRuntime:
    """The bundle's own identity, plus the launch tuple this CLI and its desktop app share."""

    bundle_root: pathlib.Path
    embedded_python: pathlib.Path
    app_root: pathlib.Path
    repo_dir: pathlib.Path
    data_dir: pathlib.Path
    app_version: str
    settings_path: pathlib.Path | None = None  # absent: derived from the roots as config derives it
    pid_file: pathlib.Path | None = None
    port_file: pathlib.Path | None = None

    def __post_init__(self) -> None:
        for name, derived in (("settings_path", self.data_dir / "settings.json"),
                              ("pid_file", self.app_root / "ouroboros.pid"),
                              ("port_file", self.data_dir / "state" / "server_port")):
            if getattr(self, name) is None:
                object.__setattr__(self, name, derived)


class _StdLogger:
    def info(self, message: str, *args: object) -> None:
        print("[ouroboros-cli] " + (message % args if args else message), file=sys.stderr)

    def warning(self, message: str, *args: object, **_kwargs: object) -> None:
        print("[ouroboros-cli] warning: " + (message % args if args else message), file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    try:
        runtime = resolve_packaged_runtime()
        _set_global_bytecode_suppression(runtime.data_dir)
        args = _prepare_start_if_requested(raw_args, runtime)
        command_idx, command = _find_command(args)
        if command == "server":
            raise PackagedCLIError(
                "packaged 'ouroboros server' is not supported; start the desktop app "
                "or use 'ouroboros run --start ...' so the launcher owns the runtime"
            )
        _bootstrap_runtime(runtime)
        return _run_inner_cli(runtime, args)
    except PackagedCLIError as exc:
        print(f"ouroboros: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


def _set_global_bytecode_suppression(data_dir: pathlib.Path) -> None:
    """WA6: globally suppress bytecode writes for the packaged CLI process itself.

    Parity with ``_inner_cli_env``'s ``embedded_python_env`` call so the CLI entry
    process and any naive ``os.environ.copy()`` child inherit the suppression. A
    signed+notarized macOS .app must never write ``__pycache__/*.pyc`` into its own
    bundle at runtime — that breaks the codesign seal. Reuses the data dir's
    ``unindexed.bytecode_prefix``, which keeps an explicit override.

    The module top already set ``sys.dont_write_bytecode`` (the only thing that
    stops THIS process's own imports); we re-assert it and add the cache-prefix env
    so child spawns inherit the policy too.
    """
    sys.dont_write_bytecode = True
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    os.environ["PYTHONPYCACHEPREFIX"] = bytecode_prefix(data_dir, os.environ.get("PYTHONPYCACHEPREFIX", ""))


def resolve_packaged_runtime() -> PackagedRuntime:
    bundle_root = _find_bundle_root(_candidate_start_paths())
    embedded_python = _find_embedded_python(bundle_root)
    app_version = _read_version(bundle_root)
    roots = configured_roots()  # the desktop launcher's own tuple: explicit values, else APP_ROOT's
    return PackagedRuntime(
        bundle_root=bundle_root,
        embedded_python=embedded_python,
        app_root=roots.app_root,
        repo_dir=roots.repo_dir,
        data_dir=roots.data_dir,
        app_version=app_version,
        settings_path=roots.settings_path,
        pid_file=roots.pid_file,
        port_file=roots.port_file,
    )


def _candidate_start_paths() -> list[pathlib.Path]:
    paths: list[pathlib.Path] = []
    env_root = os.environ.get("OUROBOROS_PACKAGED_BUNDLE_ROOT", "").strip()
    if env_root:
        paths.append(pathlib.Path(env_root))
    try:
        paths.append(pathlib.Path(__file__).resolve())
    except OSError:
        pass
    if sys.argv:
        try:
            paths.append(pathlib.Path(sys.argv[0]).resolve())
        except OSError:
            pass
    return paths


def _find_bundle_root(start_paths: Iterable[pathlib.Path]) -> pathlib.Path:
    seen: set[pathlib.Path] = set()
    for raw in start_paths:
        path = pathlib.Path(raw)
        current = path if path.is_dir() else path.parent
        candidates = [current, *current.parents]
        for parent in list(candidates):
            candidates.extend((parent / "Resources", parent / "Frameworks", parent / "_internal"))
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError:
                resolved = candidate
            if resolved in seen:
                continue
            seen.add(resolved)
            if _looks_like_bundle_root(resolved):
                return resolved
    raise PackagedCLIError(
        "could not locate packaged bundle root with repo.bundle, manifest, and python-standalone"
    )


def _looks_like_bundle_root(path: pathlib.Path) -> bool:
    return (
        (path / "repo.bundle").is_file()
        and (path / "repo_bundle_manifest.json").is_file()
        and any(candidate.exists() for candidate in embedded_python_candidates(path))
    )


def _find_embedded_python(bundle_root: pathlib.Path) -> pathlib.Path:
    for candidate in embedded_python_candidates(bundle_root):
        if candidate.exists():
            return candidate
    raise PackagedCLIError(f"embedded python-standalone was not found under {bundle_root}")


def _read_version(bundle_root: pathlib.Path) -> str:
    try:
        return (bundle_root / "VERSION").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _bootstrap_runtime(runtime: PackagedRuntime) -> None:
    if not check_git(IS_WINDOWS):
        raise PackagedCLIError(f"Git is required before first CLI use. {git_install_hint()}")
    runtime.data_dir.mkdir(parents=True, exist_ok=True)
    context = BootstrapContext(
        bundle_dir=runtime.bundle_root,
        repo_dir=runtime.repo_dir,
        data_dir=runtime.data_dir,
        settings_path=runtime.settings_path,
        embedded_python=str(runtime.embedded_python),
        app_version=runtime.app_version,
        hidden_run=_hidden_run,
        save_settings=lambda settings: _save_settings(runtime.settings_path, settings),
        log=_StdLogger(),
    )
    bootstrap_repo(context)


def _hidden_run(command: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess:
    return subprocess.run(list(command), **kwargs)  # type: ignore[arg-type]


def _save_settings(path: pathlib.Path, settings: dict) -> None:
    """The packaged bootstrap's settings writer — same prologue, same bytes.

    It is a THIRD writer only in the sense that it owns its own path; what it persists
    goes through the one persistence prologue (which proves the owner-only context/safety
    ratchets against the value on disk), the one serializer and the one byte-exact commit
    helper, so a bootstrap save cannot be the save that skips them. The path
    it writes is the path the prologue reads: the packaged runtime resolves the same
    launch tuple ``config`` freezes (``launch_roots.configured_roots``, overrides
    included), pinned by tests/test_settings_read_seam.py. Dormant today —
    ``BootstrapContext.save_settings`` has no caller.

    The write GUARDS are taken on the path this saver actually writes, not on
    ``config.SETTINGS_PATH``: under a pinned benchmark snapshot, or from pytest against the
    live install, a bootstrap save must refuse for the same reason the other two writers
    refuse. Passing its own path is what keeps that honest while the disclosed override
    split above exists."""
    from ouroboros.config import prepare_settings_for_persist, serialize_settings
    from ouroboros.settings_integrity import guard_live_settings_write
    from ouroboros.utils import write_text_atomic

    guard_live_settings_write(path, pathlib.Path.home())
    write_text_atomic(path, serialize_settings(prepare_settings_for_persist(dict(settings))))


def _run_inner_cli(runtime: PackagedRuntime, args: Sequence[str]) -> int:
    env = _inner_cli_env(runtime)
    proc = subprocess.run(
        [str(runtime.embedded_python), "-m", "ouroboros.cli", *args],
        cwd=str(runtime.repo_dir),
        env=env,
    )
    return int(proc.returncode)


def _inner_cli_env(runtime: PackagedRuntime) -> dict[str, str]:
    keep = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", *TUPLE_KEYS, BUNDLE_DIR_ENV}}
    keep.update(_roots_env(runtime))
    keep.update(
        {
            "PYTHONPATH": str(runtime.repo_dir),
            "OUROBOROS_APP_VERSION": runtime.app_version,
            "OUROBOROS_PACKAGED_CLI": "1",
            # The inner CLI runs out of the managed repo; bundled payloads
            # (node, ripgrep) are only reachable through this root.
            BUNDLE_DIR_ENV: str(runtime.bundle_root),
        }
    )
    return embedded_python_env(runtime.data_dir, keep)


def _roots_env(runtime: PackagedRuntime) -> dict[str, str]:
    """The launch tuple the inner CLI and an app opened for this CLI receive: every resolved path,
    the port file only when one was given (otherwise it follows the data root)."""
    env = {"OUROBOROS_APP_ROOT": str(runtime.app_root), "OUROBOROS_REPO_DIR": str(runtime.repo_dir),
           "OUROBOROS_DATA_DIR": str(runtime.data_dir), "OUROBOROS_SETTINGS_PATH": str(runtime.settings_path),
           "OUROBOROS_PID_FILE": str(runtime.pid_file)}
    if "OUROBOROS_PORT_FILE" in os.environ:  # a relocated endpoint stays this deployment's
        env["OUROBOROS_PORT_FILE"] = str(runtime.port_file)
    return env


def _app_env(runtime: PackagedRuntime) -> dict[str, str]:
    """What an app opened through macOS must be handed, since LaunchServices starts it with its own
    environment: this CLI's resolved tuple, including defaults, and each explicit work root."""
    work_roots = {key: os.environ[key] for key in WORK_ROOT_KEYS if os.environ.get(key)}
    return {**_roots_env(runtime), **work_roots}


def _prepare_start_if_requested(args: list[str], runtime: PackagedRuntime) -> list[str]:
    command_idx, command = _find_command(args)
    start_idx = _run_start_index(args, command_idx) if command == "run" else None
    if start_idx is None:
        return args
    base_url = _base_url_from_args(args, runtime.port_file)
    if not _is_loopback_url(base_url):
        raise PackagedCLIError("--start can only launch a local loopback Ouroboros desktop runtime")
    stripped = _strip_arg_at(args, start_idx)
    explicit = bool(_explicit_url(args) or os.environ.get("OUROBOROS_URL", "").strip())
    if not _selected_ready(base_url, runtime, explicit=explicit):
        _launch_desktop_app(runtime)
    ready_url = _wait_for_ready(base_url, runtime, explicit_url=explicit)
    os.environ["OUROBOROS_URL"] = ready_url
    return stripped


def _find_command(args: Sequence[str]) -> tuple[int, str]:
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--url":
            i += 2
            continue
        if arg.startswith("--url="):
            i += 1
            continue
        if arg.startswith("-"):
            i += 1
            continue
        return i, arg
    return len(args), ""


def _run_start_index(args: Sequence[str], command_idx: int) -> int | None:
    value_options = {
        "--workspace",
        "--memory-mode",
        "--attach",
        "--patch-out",
        "--timeout",
        "--execution-timeout",
        "--actor-id",
        "--delegation-role",
    }
    idx = command_idx + 1
    while idx < len(args):
        arg = args[idx]
        if arg == "--":
            return None
        if arg == "--start":
            return idx
        if not arg.startswith("-"):
            return None
        if arg in value_options:
            idx += 2
            continue
        if any(arg.startswith(option + "=") for option in value_options):
            idx += 1
            continue
        idx += 1
    return None


def _strip_arg_at(args: Sequence[str], remove_idx: int) -> list[str]:
    result: list[str] = []
    for idx, arg in enumerate(args):
        if idx == remove_idx:
            continue
        result.append(arg)
    return result


def _explicit_url(args: Sequence[str]) -> str:
    for idx, arg in enumerate(args):
        if arg == "--url" and idx + 1 < len(args):
            return args[idx + 1]
        if arg.startswith("--url="):
            return arg.split("=", 1)[1]
    return ""


def _base_url_from_args(args: Sequence[str], port_file: pathlib.Path) -> str:
    explicit = _explicit_url(args) or os.environ.get("OUROBOROS_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")
    # The port file of the deployment this CLI resolved (resolve_packaged_runtime).
    port = DEFAULT_PORT
    try:
        raw = pathlib.Path(port_file).read_text(encoding="utf-8").strip()
        if raw:
            port = int(raw)
    except Exception:
        pass
    return f"http://{DEFAULT_HOST}:{port}"


def _is_loopback_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    return (parsed.hostname or DEFAULT_HOST).lower() in {"127.0.0.1", "localhost", "::1"}


def _gateway_supervisor_ready(base_url: str) -> bool:
    try:
        _request_json(base_url.rstrip("/") + "/api/health", timeout=2)
        state = _request_json(base_url.rstrip("/") + "/api/state", timeout=2)
        return bool(state.get("supervisor_ready"))
    except Exception:
        return False


def _selected_ready(url: str, runtime: PackagedRuntime, *, explicit: bool) -> bool:
    """A ready supervisor at URL that is this CLI's: one the caller named, or the main server of the
    data root it resolved, never another installation's on a shared or fallback port."""
    from ouroboros.server_process import serves_data_root

    return (explicit or serves_data_root(url, runtime.data_dir)) and _gateway_supervisor_ready(url)


def _wait_for_ready(base_url: str, runtime: PackagedRuntime, *, explicit_url: bool) -> str:
    deadline = time.time() + float(os.environ.get("OUROBOROS_CLI_START_TIMEOUT", START_TIMEOUT_SEC))
    current = base_url.rstrip("/")
    while time.time() < deadline:
        if not explicit_url:
            current = _base_url_from_args([], runtime.port_file)
        if _selected_ready(current, runtime, explicit=explicit_url):
            return current
        time.sleep(0.5)
    if not explicit_url and _gateway_supervisor_ready(current):
        raise PackagedCLIError(f"the Ouroboros answering at {current} is not the one for {runtime.data_dir}; "
                               "pass --url to use it")
    raise PackagedCLIError(f"desktop runtime did not become ready at {current}")


def _request_json(url: str, *, timeout: float) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        raise PackagedCLIError(str(exc)) from exc
    data = json.loads(raw or "{}")
    return data if isinstance(data, dict) else {}


def _launch_desktop_app(runtime: PackagedRuntime) -> None:
    app = _desktop_app_path(runtime.bundle_root)
    if not app:
        raise PackagedCLIError("could not locate packaged desktop launcher for --start")
    if IS_MACOS:
        # LaunchServices keeps the app's identity and lifetime and starts it with its own environment:
        # the resolved roots ride `open --env` into a new instance (`-n`), even for defaults: one
        # running with other roots must not be activated instead. No global launchd variable is set.
        carried = [arg for key, value in _app_env(runtime).items() for arg in ("--env", f"{key}={value}")]
        subprocess.Popen(["open", "-n", *carried, str(app)])
    else:  # never with the CLI shim's bytecode prefix: outside the data dir it leaves every worker boot recompiling
        launch_env = (None if IS_WINDOWS else _linux_appimage_relaunch_env(app)) or os.environ
        subprocess.Popen([str(app)], cwd=str(app.parent),
                         env={key: value for key, value in launch_env.items() if key.upper() != "PYTHONPYCACHEPREFIX"})


def _linux_appimage_relaunch_env(app: pathlib.Path) -> dict[str, str] | None:
    """Give a nested extract-and-run AppImage its own extraction namespace."""
    if IS_MACOS or IS_WINDOWS:
        return None
    appimage = os.environ.get("APPIMAGE", "").strip()
    if not appimage or pathlib.Path(appimage) != app:
        return None
    if not appimage_extracts_and_runs():
        return None

    env = os.environ.copy()
    env["APPIMAGE_EXTRACT_AND_RUN"] = "1"
    env["OUROBOROS_APPIMAGE_RESTORE_TMPDIR"] = "1"
    env["OUROBOROS_APPIMAGE_ORIGINAL_TMPDIR_SET"] = "1" if "TMPDIR" in os.environ else "0"
    env["OUROBOROS_APPIMAGE_ORIGINAL_TMPDIR"] = os.environ.get("TMPDIR", "")
    # The type-2 runtime keys extract-and-run paths only by TMPDIR + AppImage
    # digest. A nested launch of the same AppImage must therefore use a private
    # temp base or the short-lived CLI runtime removes the desktop payload.
    env["TMPDIR"] = tempfile.mkdtemp(prefix="ouroboros-appimage-runtime-")
    return env


def _desktop_app_path(bundle_root: pathlib.Path) -> pathlib.Path | None:
    if not IS_MACOS and not IS_WINDOWS:
        # Inside an AppImage, the inner PyInstaller launcher lives on a mount (or
        # temporary extract tree) owned by the outer runtime.  Starting that raw
        # binary would let the CLI exit and release the payload underneath the
        # detached desktop process. Relaunch the stable AppImage path instead so
        # its own runtime owns the payload for the launcher's whole lifetime.
        from ouroboros.desktop_autostart import own_appimage

        appimage = own_appimage(bundle_root)  # not one inherited from another AppImage app's terminal
        if appimage and pathlib.Path(appimage).is_file():
            return pathlib.Path(appimage)
    for parent in (bundle_root, *bundle_root.parents):
        if IS_MACOS and parent.suffix == ".app":
            return parent
        if IS_WINDOWS:
            candidate = parent / "Ouroboros.exe"
            if candidate.is_file():
                return candidate
        else:
            candidate = parent / "Ouroboros"
            if candidate.is_file():
                return candidate
    return None


if __name__ == "__main__":
    sys.exit(main())
