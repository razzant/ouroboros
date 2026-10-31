"""Managed source and native-host bootstrap helpers for the common launcher."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import logging
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from ouroboros.desktop_media import host_environment
from ouroboros.unindexed import bytecode_prefix


BUNDLE_REPO_NAME = "repo.bundle"
BUNDLE_MANIFEST_NAME = "repo_bundle_manifest.json"
MANAGED_REPO_META_NAME = "ouroboros-managed.json"
BOOTSTRAP_PIN_MARKER_NAME = "ouroboros-bootstrap-pending"
MANIFEST_SCHEMA_VERSION = 1
DEFAULT_MANAGED_REMOTE_NAME = "managed"
DEFAULT_MANAGED_LOCAL_BRANCH = "ouroboros"
DEFAULT_MANAGED_LOCAL_STABLE_BRANCH = "ouroboros-stable"
DEFAULT_MANAGED_REMOTE_STABLE_BRANCH = "ouroboros-stable"


_BOOTSTRAP_COMMANDS = []  # live invocation owners only; never a scan of historical work


class _SourceHandoff:
    """Keep a Windows successor owned until transfer or Panic has won termination.

    The lock covers only publication and the terminal decision, never Popen or
    stopping a child. Panic never waits for spawn: a compatible successor observes
    the inherited parent handle before startup, even before Popen returns here.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.stopped = False
        self.successor = None

    def claim_stop(self):
        with self.lock:
            self.stopped = True

    def begin(self, cancelled):
        with self.lock:
            if self.stopped or cancelled():
                return False
            return True

    def stop(self):
        from ouroboros.platform_layer import request_process_tree_kill

        with self.lock:
            self.stopped = True
            proc = self.successor
        return request_process_tree_kill(proc) if proc is not None else None

    def spawned(self, proc, cancelled):
        with self.lock:
            self.successor = proc
            stopping = self.stopped or cancelled()
        if stopping:
            self.stop()

    def leave(self, code, cancelled):
        with self.lock:
            if not self.stopped and not cancelled():
                os._exit(code)  # terminal commit, with no intervening spawn or I/O


source_handoff = _SourceHandoff()


def stop_bootstrap_commands(*, include_interactive=True) -> list[dict]:
    """Request captured bootstrap children before early tray Panic hard-exits."""
    from ouroboros.platform_layer import request_process_tree_kill

    successor_stop = source_handoff.stop()
    return ([successor_stop] if successor_stop is not None else []) + [
            request_process_tree_kill(owner["proc"], job_handle=getattr(owner["container"], "_job", None))
            for owner in list(_BOOTSTRAP_COMMANDS) if owner["proc"] is not None
            and (include_interactive or owner["container"] is not None)]


def spawn_bootstrap_command(command, shutdown, *, interactive=False, **kwargs):
    """Capture a bootstrap command in the existing Job/lifeline owner before exposing its handle."""
    from ouroboros.process_containment import ProcessContainer
    from ouroboros.platform_layer import merge_hidden_kwargs

    if shutdown.is_set():
        raise InterruptedError("Launcher shutdown stopped bootstrap admission")
    # Interactive OS installers keep their terminal and may finish after an
    # ordinary setup-window close; Panic still has their captured Popen to stop.
    # Frozen sys.executable is the app bootloader, not a Python CLI for the
    # lifeline guardian. Keep group/token (Windows: Job) custody and Panic stops;
    # frozen POSIX has no guardian for an abrupt launcher death during bootstrap.
    owner = {"proc": None, "container": None if interactive else
             ProcessContainer(lifeline=not getattr(sys, "frozen", False))}
    _BOOTSTRAP_COMMANDS.append(owner)
    try:
        owner["proc"] = (subprocess.Popen(command, **merge_hidden_kwargs(kwargs)) if interactive
                         else owner["container"].spawn(command, **kwargs))
        if shutdown.is_set():
            stop_bootstrap_commands()
        return owner
    except BaseException:
        if owner["container"] is not None:
            owner["container"].reap()
            owner["container"].close()
        _BOOTSTRAP_COMMANDS.remove(owner)
        raise


def run_bootstrap_command(command, shutdown, **kwargs):
    """subprocess.run's result contract with early Quit/Panic and owned-tree cleanup."""
    data, check, timeout = kwargs.pop("input", None), kwargs.pop("check", False), kwargs.pop("timeout", None)
    if data is not None:
        if "stdin" in kwargs:
            raise ValueError("stdin and input arguments may not both be used")
        kwargs["stdin"] = subprocess.PIPE
    if kwargs.pop("capture_output", False):
        if "stdout" in kwargs or "stderr" in kwargs:
            raise ValueError("stdout and stderr arguments may not be used with capture_output")
        kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    owner = spawn_bootstrap_command(command, shutdown, **kwargs)
    proc, container = owner["proc"], owner["container"]
    deadline = None if timeout is None else time.monotonic() + timeout
    partial = None
    try:
        while True:
            if shutdown.is_set():
                stop_bootstrap_commands()
                raise InterruptedError("Launcher shutdown interrupted bootstrap")
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout,
                    output=partial.output if partial else None, stderr=partial.stderr if partial else None)
            try:
                stdout, stderr = proc.communicate(data, timeout=min(0.2, remaining) if remaining is not None else 0.2)
                break
            except subprocess.TimeoutExpired as exc:
                partial = exc
                data = None  # communicate retains the pending input/output across retries
        if check and proc.returncode:
            raise subprocess.CalledProcessError(proc.returncode, command, stdout, stderr)
        return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)
    finally:
        error = container.reap()
        container.close()
        _BOOTSTRAP_COMMANDS.remove(owner)
        proc.wait(timeout=5)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()
        if error:
            logging.getLogger("launcher").error("Bootstrap command cleanup unconfirmed: %s", error)


@dataclass(frozen=True)
class BootstrapContext:
    bundle_dir: pathlib.Path
    repo_dir: pathlib.Path
    data_dir: pathlib.Path
    settings_path: pathlib.Path
    embedded_python: str
    app_version: str
    hidden_run: Callable[..., Any]
    save_settings: Callable[[dict], None]
    log: Any
    deps_timeout_sec: float = 240.0  # a new runtime venv installs the whole lock at once (runtime_venv)


def checkout_sha(repo_dir: pathlib.Path) -> str:
    """HEAD of the checkout, or "" when Git or the repository is unavailable (never a guess)."""
    from ouroboros.platform_layer import merge_hidden_kwargs

    try:
        shown = subprocess.run(["git", "rev-parse", "HEAD"], **merge_hidden_kwargs(
            {"cwd": str(repo_dir), "capture_output": True, "text": True, "encoding": "utf-8", "check": False,
             **({"env": host_environment()} if sys.platform == "linux" else {})}))
    except OSError:
        return ""
    return shown.stdout.strip() if shown.returncode == 0 else ""


_LOADED_CHECKOUT_SHA = ""  # the commit whose modules this launcher process imported (remember_loaded_checkout)


def remember_loaded_checkout(repo_dir: pathlib.Path, *, bundle_dir: pathlib.Path | None = None) -> str:
    """Record once, before bootstrap can advance the checkout, which commit the launcher's modules came
    from: a frozen app or a seed copy imported them from ``bundle_dir`` (its manifest's ``source_sha``),
    the checkout's own launcher from ``repo_dir`` (its HEAD)."""
    global _LOADED_CHECKOUT_SHA
    if _LOADED_CHECKOUT_SHA:
        return _LOADED_CHECKOUT_SHA
    # Frozen helpers came from the seed, even when the managed checkout is newer.
    loaded = (_read_json_file(bundle_dir / BUNDLE_MANIFEST_NAME).get("source_sha", "")
              if bundle_dir is not None else checkout_sha(repo_dir))
    _LOADED_CHECKOUT_SHA = str(loaded or "")
    return _LOADED_CHECKOUT_SHA


def launcher_sources_changed(repo_dir: pathlib.Path, loaded_sha: str | None = None, *,
                             bundle_dir: pathlib.Path | None = None, packaged: bool = False,
                             source_root: pathlib.Path | None = None) -> bool:
    """True when the landed checkout changed a module the launcher process has loaded.

    The launcher imports its helpers once; a cold start or body adoption that
    finds another version in the checkout hands off through ``exec_source_launcher``.
    A server-only update leaves the launcher running. The comparison is the
    checkout's own history, not the file system: the commit the launcher imported
    from (``loaded_sha``, "" when unknown: never a relaunch) against the commit
    that now serves, over the loaded modules' repository paths plus ``launcher.py``.
    A packaged launcher loaded its modules from ``bundle_dir`` (PyInstaller names
    them ``<pkg>/<module>.pyc`` or ``.py`` there); they count under their repository
    paths. A path counts as named or resolved: a macOS app's bundle is
    ``Contents/Frameworks``, whose ``<pkg>`` links into ``Contents/Resources``, so
    there only the named path lies inside the bundle. ``source_root`` also covers
    seed hosts that import from a separate image directory (the container's /app).

    ``packaged`` (the cold check) counts the whole packaged trees,
    ``ouroboros`` and ``supervisor``, not only what is loaded so far: later imports
    (Panic sweep, restart acknowledgement, server records) would come from the seed
    too. Conservative by design: a body-only change there hands off as well, while a
    ``server.py``-only update keeps the launcher.
    """
    from ouroboros.platform_layer import merge_hidden_kwargs

    loaded_sha = _LOADED_CHECKOUT_SHA if loaded_sha is None else loaded_sha
    head = checkout_sha(repo_dir)
    if not loaded_sha or not head or head == loaded_sha:
        return False
    roots = [pathlib.Path(os.path.abspath(root)) for root in (repo_dir, bundle_dir, source_root) if root]
    roots += [root.resolve() for root in roots]
    loaded = {"launcher.py", *(("ouroboros", "supervisor") if packaged else ())}
    for module in list(sys.modules.values()):
        source = getattr(module, "__file__", None)
        if not source:
            continue
        try:
            named = pathlib.Path(os.path.abspath(source))
            paths = (named, named.resolve())
        except OSError:
            continue
        relative = next((path.relative_to(root) for path in paths for root in roots if path.is_relative_to(root)), None)
        if relative is not None:
            loaded.add((relative.with_suffix(".py") if relative.suffix == ".pyc" else relative).as_posix())
    probe = subprocess.run(["git", "diff", "--quiet", loaded_sha, head, "--", *sorted(loaded)],
                           **merge_hidden_kwargs({"cwd": str(repo_dir), "capture_output": True, "check": False,
                               **({"env": host_environment()} if sys.platform == "linux" else {})}))
    return probe.returncode == 1


def embedded_python_env(data_dir: pathlib.Path, base: dict[str, str] | None = None) -> dict[str, str]:
    """Env for running the EMBEDDED interpreter without writing into the bundle.

    The packaged interpreter lives INSIDE the signed application bundle, which is
    read-only in principle (and whose signature any write invalidates). Python
    has exactly two default write targets there and BOTH are redirected under the
    data dir, in one place, so a new caller cannot pick up one and miss the other:

    - bytecode caches -> ``PYTHONPYCACHEPREFIX``;
    - ``pip install`` -> the user site under ``PYTHONUSERBASE`` (paired with the
      ``--user`` flag at the pip call sites; ``site`` puts that directory on
      ``sys.path`` automatically, so every child launched through this env
      IMPORTS what was installed through it).

    ``PYTHONNOUSERSITE`` is cleared for the same reason: an inherited value would
    install to the user site and then refuse to import from it.

    DISCLOSED, not fixed: ``site`` puts the user site AHEAD of the bundle's own
    site-packages, and nothing prunes it. So a package installed here by an older
    version keeps OUTRANKING the signed bundle's copy for as long as the data dir
    survives — an upgrade can ship a newer dependency and still import the old one.
    Making that impossible means either pruning logic or versioning this directory,
    and versioning it silently abandons every dependency an existing install already
    paid to download. Both are product decisions with real cost, so the behaviour is
    stated here rather than changed unilaterally. Recovery is manual and total:
    delete ``<data>/state/python-userbase`` and relaunch.
    """
    # Defaults are for bootstrap host commands. Source-launcher probe/exec pass
    # their explicit GUI base and must retain it for GI and WebKit helpers.
    env = dict((host_environment() if sys.platform == "linux" else os.environ) if base is None else base)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    user_base = pathlib.Path(data_dir) / "state" / "python-userbase"
    try:
        user_base.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    env["PYTHONPYCACHEPREFIX"] = bytecode_prefix(data_dir, env.get("PYTHONPYCACHEPREFIX", ""))  # a custom one stays
    env["PYTHONUSERBASE"] = str(user_base)
    env.pop("PYTHONNOUSERSITE", None)
    return env


def check_git(is_windows: bool) -> bool:
    if shutil.which("git") is not None:
        return True
    if is_windows:
        for candidate in (
            os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Git", "cmd", "git.exe"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Git", "cmd", "git.exe"),
        ):
            if os.path.isfile(candidate):
                git_dir = os.path.dirname(candidate)
                os.environ["PATH"] = git_dir + ";" + os.environ.get("PATH", "")
                return True
    return False


def _bundle_manifest_path(context: BootstrapContext) -> pathlib.Path:
    return context.bundle_dir / BUNDLE_MANIFEST_NAME


def _managed_meta_path(repo_dir: pathlib.Path) -> pathlib.Path:
    return repo_dir / ".git" / MANAGED_REPO_META_NAME


def _bootstrap_pin_marker_path(repo_dir: pathlib.Path) -> pathlib.Path:
    return repo_dir / ".git" / BOOTSTRAP_PIN_MARKER_NAME


def _read_json_file(path: pathlib.Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _normalize_bundle_manifest(raw: dict[str, Any], *, app_version: str) -> dict[str, Any]:
    manifest = dict(raw)
    return {
        "schema_version": int(manifest.get("schema_version") or MANIFEST_SCHEMA_VERSION),
        "bundle_file": str(manifest.get("bundle_file") or BUNDLE_REPO_NAME),
        "app_version": str(manifest.get("app_version") or app_version),
        "source_sha": str(manifest.get("source_sha") or ""),
        "release_tag": str(manifest.get("release_tag") or ""),
        "bundle_sha256": str(manifest.get("bundle_sha256") or ""),
        "source_branch": str(manifest.get("source_branch") or ""),
        "managed_remote_name": str(manifest.get("managed_remote_name") or DEFAULT_MANAGED_REMOTE_NAME),
        "managed_remote_url": str(manifest.get("managed_remote_url") or ""),
        "managed_remote_branch": str(manifest.get("managed_remote_branch") or manifest.get("source_branch") or ""),
        "managed_local_branch": str(manifest.get("managed_local_branch") or DEFAULT_MANAGED_LOCAL_BRANCH),
        "managed_local_stable_branch": str(
            manifest.get("managed_local_stable_branch") or DEFAULT_MANAGED_LOCAL_STABLE_BRANCH
        ),
        "managed_remote_stable_branch": str(
            manifest.get("managed_remote_stable_branch") or DEFAULT_MANAGED_REMOTE_STABLE_BRANCH
        ),
    }


def load_bundle_manifest(context: BootstrapContext) -> dict[str, Any]:
    manifest_path = _bundle_manifest_path(context)
    if not manifest_path.is_file():
        raise RuntimeError(
            f"Embedded managed repo manifest is missing: {manifest_path}. "
            "Rebuild the app bundle with scripts/build_repo_bundle.py."
        )
    manifest = _normalize_bundle_manifest(_read_json_file(manifest_path), app_version=context.app_version)
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported managed repo manifest schema {manifest['schema_version']} "
            f"(expected {MANIFEST_SCHEMA_VERSION})."
        )
    if not manifest["source_sha"]:
        raise RuntimeError("Managed repo manifest is missing source_sha.")
    if not manifest["bundle_sha256"]:
        raise RuntimeError("Managed repo manifest is missing bundle_sha256.")
    if not manifest["managed_remote_branch"]:
        raise RuntimeError("Managed repo manifest is missing managed_remote_branch.")
    if manifest["app_version"] != context.app_version:
        raise RuntimeError(
            f"Managed repo manifest app_version {manifest['app_version']!r} does not "
            f"match launcher app version {context.app_version!r}."
        )
    expected_tag = f"v{manifest['app_version']}"
    if manifest["release_tag"] and manifest["release_tag"] != expected_tag:
        raise RuntimeError(
            f"Managed repo manifest release_tag {manifest['release_tag']!r} does not "
            f"match app_version {manifest['app_version']!r}."
        )
    _assert_bundle_integrity(context, manifest)
    return manifest


def load_repo_manifest(repo_dir: pathlib.Path) -> dict[str, Any]:
    meta_path = _managed_meta_path(repo_dir)
    if not meta_path.is_file():
        return {}
    return _read_json_file(meta_path)


def _write_repo_manifest(repo_dir: pathlib.Path, manifest: dict[str, Any]) -> None:
    # Atomic: a torn manifest makes the next launch misdetect the installed
    # repo generation and re-bootstrap over a healthy checkout.
    from ouroboros.utils import atomic_write_json

    atomic_write_json(_managed_meta_path(repo_dir), manifest, trailing_newline=True)


def _mark_bootstrap_pin_pending(repo_dir: pathlib.Path) -> None:
    marker = _bootstrap_pin_marker_path(repo_dir)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("pending\n", encoding="utf-8")


def _repo_manifest_matches(repo_dir: pathlib.Path, bundle_manifest: dict[str, Any]) -> bool:
    installed = load_repo_manifest(repo_dir)
    if not installed:
        return False
    tracked_keys = (
        "schema_version",
        "app_version",
        "source_sha",
        "release_tag",
        "bundle_sha256",
        "managed_remote_name",
        "managed_remote_url",
        "managed_remote_branch",
        "managed_local_branch",
        "managed_local_stable_branch",
        "managed_remote_stable_branch",
    )
    return all(str(installed.get(key) or "") == str(bundle_manifest.get(key) or "") for key in tracked_keys)


def _run_git(context: BootstrapContext, args: list[str], *, cwd: pathlib.Path, check: bool = True) -> Any:
    return context.hidden_run(
        args,
        cwd=str(cwd),
        check=check,
        capture_output=True,
        text=True,
    )


def _remote_url(context: BootstrapContext, repo_dir: pathlib.Path, remote_name: str) -> str:
    result = _run_git(context, ["git", "remote", "get-url", remote_name], cwd=repo_dir, check=False)
    return str(getattr(result, "stdout", "") or "").strip()


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_bundle_integrity(context: BootstrapContext, manifest: dict[str, Any]) -> None:
    bundle_path = context.bundle_dir / manifest["bundle_file"]
    if not bundle_path.is_file():
        raise RuntimeError(
            f"Embedded managed repo bundle is missing: {bundle_path}. "
            "Rebuild the app bundle with scripts/build_repo_bundle.py."
        )
    actual_sha = _sha256_file(bundle_path)
    expected_sha = str(manifest.get("bundle_sha256") or "").strip()
    if expected_sha and actual_sha != expected_sha:
        raise RuntimeError(
            f"Embedded managed repo bundle hash mismatch for {bundle_path}: "
            f"expected {expected_sha}, got {actual_sha}."
        )


def _archive_existing_repo(context: BootstrapContext, reason: str) -> pathlib.Path | None:
    if not context.repo_dir.exists():
        return None
    archive_root = context.data_dir / "archive" / "managed_repo"
    archive_root.mkdir(parents=True, exist_ok=True)
    archive_dir = archive_root / f"{int(time.time())}-{uuid.uuid4().hex[:8]}-{reason}"
    shutil.move(str(context.repo_dir), str(archive_dir))
    context.log.info("Archived existing repo to %s (%s)", archive_dir, reason)
    return archive_dir


def _remove_if_exists(path: pathlib.Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _configure_managed_clone(context: BootstrapContext, repo_dir: pathlib.Path, manifest: dict[str, Any]) -> None:
    source_sha = str(manifest.get("source_sha") or "").strip()
    local_branch = manifest["managed_local_branch"]
    local_stable_branch = manifest["managed_local_stable_branch"]
    remote_name = manifest["managed_remote_name"]
    remote_url = manifest["managed_remote_url"]

    source_sha_check = _run_git(
        context,
        ["git", "rev-parse", "--verify", source_sha],
        cwd=repo_dir,
        check=False,
    )
    if getattr(source_sha_check, "returncode", 1) != 0:
        raise RuntimeError(
            f"Embedded managed repo bundle does not contain manifest source_sha {source_sha}."
        )
    _run_git(context, ["git", "checkout", "-B", local_branch, source_sha], cwd=repo_dir)
    head = _run_git(context, ["git", "rev-parse", "HEAD"], cwd=repo_dir)
    head_sha = str(getattr(head, "stdout", "") or "").strip()
    if head_sha != source_sha:
        raise RuntimeError(
            f"Managed repo bootstrap checked out {head_sha or '(unknown)'} but manifest "
            f"requires {source_sha}."
        )
    if local_stable_branch and local_stable_branch != local_branch:
        if head_sha:
            _run_git(context, ["git", "branch", "-f", local_stable_branch, head_sha], cwd=repo_dir)

    origin = _run_git(context, ["git", "remote"], cwd=repo_dir, check=False)
    existing_remotes = {
        line.strip() for line in str(getattr(origin, "stdout", "") or "").splitlines() if line.strip()
    }
    if "origin" in existing_remotes:
        _run_git(context, ["git", "remote", "remove", "origin"], cwd=repo_dir, check=False)
    if remote_name in existing_remotes:
        _run_git(context, ["git", "remote", "remove", remote_name], cwd=repo_dir, check=False)
    if remote_url:
        _run_git(context, ["git", "remote", "add", remote_name, remote_url], cwd=repo_dir)

    _run_git(context, ["git", "config", "user.name", "Ouroboros"], cwd=repo_dir, check=False)
    _run_git(context, ["git", "config", "user.email", "ouroboros@local.mac"], cwd=repo_dir, check=False)
    _write_repo_manifest(repo_dir, manifest)
    _mark_bootstrap_pin_pending(repo_dir)


def _ensure_managed_remote(context: BootstrapContext, repo_dir: pathlib.Path, manifest: dict[str, Any]) -> None:
    remote_name = manifest["managed_remote_name"]
    remote_url = manifest["managed_remote_url"]

    remotes = _run_git(context, ["git", "remote"], cwd=repo_dir, check=False)
    existing_remotes = {
        line.strip() for line in str(getattr(remotes, "stdout", "") or "").splitlines() if line.strip()
    }
    if remote_url:
        if remote_name in existing_remotes:
            _run_git(context, ["git", "remote", "set-url", remote_name, remote_url], cwd=repo_dir)
        else:
            _run_git(context, ["git", "remote", "add", remote_name, remote_url], cwd=repo_dir)

    _run_git(context, ["git", "config", "user.name", "Ouroboros"], cwd=repo_dir, check=False)
    _run_git(context, ["git", "config", "user.email", "ouroboros@local.mac"], cwd=repo_dir, check=False)
    _write_repo_manifest(repo_dir, manifest)


def _clone_repo_from_bundle(context: BootstrapContext, manifest: dict[str, Any]) -> pathlib.Path:
    bundle_path = context.bundle_dir / manifest["bundle_file"]
    if not bundle_path.is_file():
        raise RuntimeError(
            f"Embedded managed repo bundle is missing: {bundle_path}. "
            "Rebuild the app bundle with scripts/build_repo_bundle.py."
        )

    temp_repo = context.repo_dir.parent / f".repo-bootstrap-{uuid.uuid4().hex[:8]}"
    _remove_if_exists(temp_repo)
    try:
        _run_git(context, ["git", "clone", str(bundle_path), str(temp_repo)], cwd=context.bundle_dir)
        _configure_managed_clone(context, temp_repo, manifest)
        return temp_repo
    except Exception:
        _remove_if_exists(temp_repo)
        raise


def _install_managed_repo(context: BootstrapContext, manifest: dict[str, Any], *, reason: str) -> str:
    preserved_origin_url = _remote_url(context, context.repo_dir, "origin") if (context.repo_dir / ".git").exists() else ""
    archived_repo = _archive_existing_repo(context, reason)
    temp_repo = _clone_repo_from_bundle(context, manifest)
    try:
        shutil.move(str(temp_repo), str(context.repo_dir))
        if preserved_origin_url:
            _run_git(context, ["git", "remote", "add", "origin", preserved_origin_url], cwd=context.repo_dir, check=False)
    except Exception:
        _remove_if_exists(temp_repo)
        if archived_repo is not None and not context.repo_dir.exists():
            shutil.move(str(archived_repo), str(context.repo_dir))
        raise
    return "replaced" if archived_repo is not None else "created"


def ensure_managed_repo(context: BootstrapContext) -> str:
    """Ensure REPO_DIR is a managed git clone backed by the embedded bundle.

    An existing directory without ``.git`` is archived under
    ``data/archive/managed_repo`` before the first clone. A manifest mismatch on
    a git checkout only refreshes the managed remote metadata; the checkout stays.
    """
    manifest = load_bundle_manifest(context)
    if not context.repo_dir.exists():
        return _install_managed_repo(context, manifest, reason="missing")
    if not (context.repo_dir / ".git").exists():
        from ouroboros.platform_layer import is_mount_point

        try:
            mounted = is_mount_point(context.repo_dir)
        except OSError as exc:
            raise RuntimeError(f"Cannot determine whether {context.repo_dir} is a mount point; "
                               f"nothing was moved: {exc}") from exc
        if mounted:
            # Archiving renames the directory away, which a mount point refuses
            # halfway through a cross-device copy: refuse before touching it.
            raise RuntimeError(
                f"{context.repo_dir} is a mount point without a checkout; nothing was moved. "
                "Mount the volume at its parent directory so the first bootstrap can create "
                "the repository inside it (docs/DEPLOYMENT.md).")
        return _install_managed_repo(context, manifest, reason="legacy-no-git")
    if not _repo_manifest_matches(context.repo_dir, manifest):
        _ensure_managed_remote(context, context.repo_dir, manifest)
        context.log.info(
            "Updated managed repo metadata for embedded bundle without replacing local checkout."
        )
        return "metadata-updated"

    _ensure_managed_remote(context, context.repo_dir, manifest)
    return "unchanged"


def sync_existing_repo_from_bundle(context: BootstrapContext) -> None:
    """Reconcile the managed repo against the embedded bundle metadata."""
    outcome = ensure_managed_repo(context)
    context.log.info("Managed repo sync outcome: %s", outcome)


def _pip_output_tail(result: Any, limit: int = 800) -> str:
    """Last chars of a captured pip run, whether it captured bytes or text."""
    parts = []
    for stream in (getattr(result, "stdout", None), getattr(result, "stderr", None)):
        if isinstance(stream, bytes):
            stream = stream.decode("utf-8", errors="replace")
        if isinstance(stream, str) and stream.strip():
            parts.append(stream.strip())
    return ("\n".join(parts))[-limit:]


def install_deps(context: BootstrapContext) -> bool:
    """Install/update Python deps for the embedded interpreter.

    ``pip_install_target_args`` keeps a packaged install OUT of the signed bundle
    the embedded interpreter lives in; the return value reports whether pip
    actually succeeded.
    """
    try:
        requirements = context.repo_dir / "requirements-runtime.lock"
        if not requirements.exists():
            # Managed repositories from before the uv migration may still use
            # the legacy filename during an in-place upgrade.
            requirements = context.repo_dir / "requirements.txt"
        if not requirements.exists():
            return True
        from ouroboros.platform_layer import pip_install_target_args

        result = context.hidden_run(
            [context.embedded_python, "-m", "pip", "install",
             *pip_install_target_args(context.embedded_python), "-r", str(requirements)],
            env=embedded_python_env(context.data_dir),
            timeout=context.deps_timeout_sec,
            capture_output=True,
        )
    except Exception as exc:
        context.log.warning("Dependency install/update failed: %s", exc)
        return False
    # A non-zero pip exit is a real, actionable failure (unreachable index, an
    # unbuildable wheel, a read-only target): silently discarding it leaves the
    # app to fail much later with an unexplained ImportError.
    returncode = int(getattr(result, "returncode", 0) or 0)
    if returncode != 0:
        context.log.warning(
            "Dependency install/update pip exited %d; the runtime may be missing packages:\n%s",
            returncode, _pip_output_tail(result),
        )
        return False
    return True


def ensure_data_skills_seeded() -> int:
    """Startup work of the built-in skill source on the data root (``skill_builtin``):
    the one-time identity conversion, re-seeding built-ins whose bundled version changed
    and releasing built-ins the app no longer bundles. A new install gets its built-ins
    from the starter selection, never from this hook."""
    from ouroboros.skill_builtin import ensure_builtin_skills

    return ensure_builtin_skills()


def bootstrap_repo(context: BootstrapContext) -> bool:
    """Ensure the launcher-managed git repo exists and matches the embedded bundle.

    Returns whether the dependency install (when one was needed) succeeded —
    a freshly created/synced checkout that could not land its requirements is
    a fact the caller must be able to disclose, not a silently swallowed pip
    exit code that resurfaces later as an unexplained ImportError.
    """
    context.data_dir.mkdir(parents=True, exist_ok=True)
    outcome = ensure_managed_repo(context)
    context.log.info("Bootstrapping managed repository to %s (outcome=%s)", context.repo_dir, outcome)

    try:
        memory_dir = context.data_dir / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        world_path = memory_dir / "WORLD.md"
        if not world_path.exists():
            # Roots ride the environment and the path argv (a quote in it cannot break the
            # program text); the one-argument call also suits an older seed's profiler.
            env = {**(host_environment() if sys.platform == "linux" else os.environ), "PYTHONPATH": str(context.repo_dir),
                   "OUROBOROS_REPO_DIR": str(context.repo_dir), "OUROBOROS_DATA_DIR": str(context.data_dir)}
            context.hidden_run(
                [
                    context.embedded_python,
                    "-c",
                    "import sys; from ouroboros.world_profiler import generate_world_profile; "
                    "generate_world_profile(sys.argv[1])",
                    str(world_path),
                ],
                env=embedded_python_env(context.data_dir, env),
                timeout=30,
                capture_output=True,
            )
    except Exception as exc:
        context.log.warning("World profile generation failed: %s", exc)

    deps_ok = True
    if outcome != "unchanged":
        deps_ok = install_deps(context)
        if not deps_ok:
            context.log.warning(
                "Bootstrap finished with a FAILED dependency install (repo outcome=%s): "
                "the checkout changed but its requirements did not land, so the server "
                "may be missing packages. See the pip output above for the cause.",
                outcome,
            )
    context.log.info("Bootstrap complete.")
    return deps_ok


def update_external_host(hook: pathlib.Path | None, python: str, log: Any, abort_event=None) -> dict:
    """Synchronize a selected native host while preserving the available core."""
    if hook is None:
        return {}
    from ouroboros.process_containment import ProcessContainer
    from ouroboros.config import EXTERNAL_PLATFORM_UPDATE_TIMEOUT_SEC

    container = ProcessContainer()
    proc = None
    returncode = None
    detail = ""
    cancelled = False
    try:
        proc = container.spawn(
            [python, str(hook)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        deadline = time.monotonic() + EXTERNAL_PLATFORM_UPDATE_TIMEOUT_SEC
        while True:
            if abort_event is not None and abort_event.is_set():
                cancelled = True
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(proc.args, EXTERNAL_PLATFORM_UPDATE_TIMEOUT_SEC)
            try:
                # communicate retains partial pipe output across timeout retries.
                # This is a shutdown observation interval, not a shorter build budget.
                stdout, stderr = proc.communicate(timeout=min(remaining, 0.2) if abort_event is not None
                                                  else EXTERNAL_PLATFORM_UPDATE_TIMEOUT_SEC)
                returncode = proc.returncode
                detail = stdout.strip() or stderr.strip()
                break
            except subprocess.TimeoutExpired:
                if abort_event is None:
                    raise
    except (OSError, subprocess.SubprocessError) as error:
        detail = str(error)
    finally:
        try:
            reap_error = container.reap()
        except Exception as error:
            reap_error = str(error)
        finally:
            container.close()
        if proc is not None:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                reap_error = reap_error or "Native update process termination was not confirmed."
            for stream in (proc.stdout, proc.stderr):
                if stream is not None:
                    stream.close()
    if cancelled:
        log.info("Native host preparation cancelled by launcher shutdown.")
        if reap_error:
            log.error("Native host preparation cleanup was not confirmed: %s", reap_error)
        return {"status": "cancelled"}
    if returncode == 0 and not reap_error:
        try:
            payload = json.loads(detail)
            installed = payload.get("installed") or {}
            proof = {"source_commit": payload.get("source_commit"),
                     "input_sha256": payload.get("input_sha256"),
                     "apk_sha256": payload.get("apk_sha256") or installed.get("sha256"),
                     "signer_sha256": payload.get("signer_sha256")}
            if (payload.get("status") in {"current", "installed"}
                    and all(isinstance(value, str) and len(value) == (40 if key == "source_commit" else 64)
                            and all(char in "0123456789abcdef" for char in value)
                            for key, value in proof.items())):
                log.info("External host update verified: %s", json.dumps(proof))
                return {"status": "verified", **proof}
            detail = "Native installer did not return a complete artifact proof."
        except (ValueError, TypeError, AttributeError):
            detail = "Native installer did not return a valid artifact proof."
    if reap_error:
        detail += "; " + reap_error
    # Core health remains independent; managed-update boot verification checks
    # the desired host artifact before reporting that the full update landed.
    log.error("native_update_failed: %s; continuing with the available core", detail)
    return {"status": "failed"}


def appimage_extracts_and_runs() -> bool:
    """Whether this AppImage process runs from an extracted tree instead of a FUSE mount."""
    if "APPIMAGE_EXTRACT_AND_RUN" in os.environ:
        return True
    # The type-2 runtime removes the explicit --appimage-extract-and-run
    # argument before execing AppRun. Its APPDIR is then a regular directory;
    # a normal FUSE-backed APPDIR is a mount point.
    appdir = os.environ.get("APPDIR", "").strip()
    return bool(appdir) and not os.path.ismount(appdir)


def parse_launch_options(argv):
    """Parse presentation/seed inputs while tolerating ordinary platform launch arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-ui", action="store_true", help="Use an external UI host.")
    parser.add_argument("--launch-intent", choices=("owner", "automatic"), default="owner",
                        help="Automatic boot entry preserves an explicit Panic stop.")
    parser.add_argument("--host-update", type=pathlib.Path,
                        help="External host Python installer; also supports read-only --check.")
    parser.add_argument("--seed-bundle", type=pathlib.Path, help="Keep immutable seed provenance when relaunching from source; presentation is unchanged.")
    parser.add_argument("--runtime-venv", type=pathlib.Path,
                        help="Absolute path of a persistent venv the body and its dependencies run in.")
    # Desktop launchers may supply platform arguments (for example Finder's
    # process serial number). Retain their previously ignored behavior.
    options, _ = parser.parse_known_args(argv)
    if options.host_update is not None and not options.no_ui:
        parser.error("--host-update requires --no-ui")
    return options


def macos_source_entry(bundle_dir: pathlib.Path) -> pathlib.Path | None:
    """The installed, signed Python host in Contents/MacOS; never a python symlink.

    Its location keeps the ordinary app's WebKit profile. Its explicit signing
    identifier keeps notification authorization. An older bundle has no entry;
    updating source alone cannot add one to a sealed application.
    """
    if bundle_dir.name != "Frameworks" or bundle_dir.parent.name != "Contents" \
            or bundle_dir.parent.parent.suffix != ".app":
        return None
    entry = bundle_dir.parent / "MacOS" / "ouroboros-shell"
    return entry if entry.is_file() and not entry.is_symlink() and os.access(entry, os.X_OK) else None


def find_embedded_python(bundle_dir: pathlib.Path, *, macos: bool) -> str:
    """Select plain Python for the body/probes; an app entry must never replace it."""
    from ouroboros.platform_layer import embedded_python_candidates

    for path in embedded_python_candidates(bundle_dir):
        if path.exists():
            return str(path)
    if macos and macos_source_entry(bundle_dir) is not None:
        raise RuntimeError("The app's bundled Python is missing; the source entry cannot run the server.")
    return sys.executable


def shell_runtime_facts(repo_dir: pathlib.Path, seed: pathlib.Path | None, shell_version: str, *, macos: bool) -> dict:
    """Installed seed and loaded source provenance; neither asserts native permission."""
    frozen = getattr(sys, "frozen", False)
    try:
        bundle_version = (seed / "VERSION").read_text().strip() if seed else ""
    except OSError:
        bundle_version = ""
    entry = macos_source_entry(seed) if macos and seed else None
    return {"shell_version": shell_version, "bundle_version": bundle_version,
            "source_launcher": not frozen,
            "source_revision": remember_loaded_checkout(repo_dir, bundle_dir=seed if frozen else None),
            "macos_source_entry": bool(entry and pathlib.Path(sys.executable) == entry) if macos and seed else None}


# Consumed by launcher.py's stdlib prologue before any package import.
LAUNCHER_HANDOFF_ENV = "OUROBOROS_LAUNCHER_HANDOFF"


def exec_source_launcher(context: BootstrapContext, argv: list[str], *, release_lock: Callable[[], None],
                         acquire_lock: Callable[[], bool], before_release: Callable[[], None],
                         on_failure: Callable[[], None], desktop: bool, macos: bool, cancelled=lambda: False) -> None:
    """Hand off to compatible managed source; keep the shell on probe or spawn refusal.

    macOS selects its signed entry and probes imports/protocol with plain Python before
    teardown or unlocking. POSIX replaces the process image with that exact command.
    Windows spawns a quoted argv, preserving an attached CLI console and hiding only
    when there is none. Remove the old icon while holding the lock; a definite refusal
    reacquires it before restoring the shell, or exits if another launcher took it.
    A successor that starts and then fails is not a spawn refusal; no retry state is kept.
    """
    from ouroboros.platform_layer import IS_WINDOWS, inherited_console_kwargs

    entry = macos_source_entry(context.bundle_dir) if macos else None
    frozen = getattr(sys, "frozen", False)
    if not source_launcher_importable(
            context, desktop=desktop, python=(find_embedded_python(context.bundle_dir, macos=macos)
                if entry is not None or frozen else context.embedded_python),
            require_macos_entry=entry is not None):
        return  # old/rollback source or a missing dependency must not lose the working shell
    handoff = source_handoff_argv(argv, keep_runtime_venv=entry is not None)
    if frozen:
        handoff += ["--seed-bundle", str(context.bundle_dir)]
    command = [str(entry) if entry else context.embedded_python, str(context.repo_dir / "launcher.py"), *handoff]
    context.log.info("Loading managed source launcher through %s", command[0])
    # The source shell needs the dependencies bootstrap installed in the userbase,
    # too. Do not export PYTHONHOME or change any child's process/bundle identity.
    env = embedded_python_env(context.data_dir, {**os.environ, LAUNCHER_HANDOFF_ENV: "1"})
    if cancelled():
        return
    before_release()
    release_lock()
    try:
        if cancelled():
            if acquire_lock():
                on_failure()
            return  # the Panic owner must finish even if another launcher took the lock
        if not IS_WINDOWS:
            os.execve(command[0], command, env)  # returns only by raising
        from ouroboros.server_control import _windows_restart_handoff

        # Use the direct-restart parent handle/stdio contract. A current source
        # launcher refuses its parent's Panic before startup; old source still
        # imports/runs, but cannot guarantee this pre-Popen-return stop boundary.
        with _windows_restart_handoff(env, context.log) as (_, inheritance):
            if not source_handoff.begin(cancelled):
                return
            successor = None
            try:
                successor = subprocess.Popen(command, env=env, **inherited_console_kwargs(),
                                             **{"close_fds": True, **inheritance})
            finally:
                source_handoff.spawned(successor, cancelled)
    except Exception:
        context.log.error("The checkout's launcher did not start", exc_info=True)
        if acquire_lock():
            on_failure()
            return
        if cancelled():
            return  # a failed spawn cannot hard-exit over Panic's pending stop/writes
        context.log.error("Another launcher took the single-instance lock meanwhile; this one exits")
        source_handoff.leave(1, cancelled)
        return
    source_handoff.leave(0, cancelled)  # no atexit release over the successor's PID lock


def source_launcher_importable(context: BootstrapContext, *, desktop: bool, python: str | None = None,
                               require_macos_entry: bool = False) -> bool:
    """Bounded handoff check using plain Python, before losing the running shell.

    Import only: no main(), window, server or notification center. A source update
    with missing dependencies can keep using the running shell until repaired.
    Entry compatibility is explicit, not inferred from version or Git ancestry:
    an older launcher can import yet use sys.executable (the entry) for its body.
    """
    code = "import launcher" + ("; import webview" if desktop else "")
    if require_macos_entry:
        code += ("\nif getattr(launcher, 'MACOS_SOURCE_ENTRY_PROTOCOL', None) != 1:\n"
                 "    raise SystemExit('Source launcher does not support macOS source entry protocol 1')\n")
    env = embedded_python_env(context.data_dir, {**os.environ, "PYTHONPATH": str(context.repo_dir)})
    try:
        result = context.hidden_run([python or context.embedded_python, "-c", code], cwd=str(context.repo_dir),
                                    env=env, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return True
        reason = _pip_output_tail(result)
    except (OSError, subprocess.SubprocessError) as exc:
        reason = str(exc)
    context.log.warning("Source launcher probe failed; keeping the running shell: %s", reason)
    return False


def source_handoff_argv(argv, *, keep_runtime_venv: bool = False) -> list[str]:
    """The argv a relaunch hands to the checkout's own launcher.

    ``--runtime-venv`` was consumed here: the relaunch interpreter IS that venv, and
    an older rollback launcher must not receive an option it never knew. The macOS
    app entry instead hosts the shell on bundled Python, so it must retain the
    option selecting the body's separate interpreter."""
    if keep_runtime_venv:
        return list(argv)
    handoff, items = [], iter(argv)
    for item in items:
        if item == "--runtime-venv":
            next(items, None)  # its value
        elif not str(item).startswith("--runtime-venv="):
            handoff.append(item)
    return handoff


def automatic_launch_allowed(intent: str, data_dir: pathlib.Path, log: Any) -> bool:
    if intent == "automatic":
        from ouroboros.installation_stop import installation_panic

        stopped = installation_panic(data_dir)
        if stopped:
            log.info("Automatic start skipped: Ouroboros was stopped with Panic.")
            print("Ouroboros is stopped. Use Start to resume.", flush=True)
            return False
    return True
