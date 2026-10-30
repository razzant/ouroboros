"""Stdlib-only environment boundary for local pytest, preflight and UI servers.

May be loaded by file path before importing the package. These are disposable
filesystem defaults, not an OS sandbox. Tests can still explicitly select their
own synthetic roots. No dependency installation happens here.
"""
from __future__ import annotations

import ast
import os
import sys
import tempfile
from pathlib import Path


def default_temp_parent() -> Path:
    """A short root keeps nested multiprocessing sockets inside AF_UNIX's path limit (104 bytes
    on macOS, 108 on Linux): rooted in an inherited, already nested TMPDIR, a nested forkserver
    or manager address outgrew it (#2110). Other platforms keep their temporary directory."""
    short = sys.platform == "darwin" or sys.platform.startswith("linux")
    return Path("/tmp") if short else Path(tempfile.gettempdir())


def git_ancestor(path: Path):
    """The nearest directory at or above ``path`` that Git would treat as a checkout.

    safe_test refuses a disposable parent with one; delegated-session facts name the same refusal."""
    for directory in (path, *path.parents):
        if (directory / ".git").exists():
            return directory
    return None


def settings_keys() -> set[str]:
    """Read the settings vocabulary without importing the settings runtime."""
    keys = set()
    for name in ("settings_defaults.py", "update_channels.py"):
        tree = ast.parse(Path(__file__).with_name(name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                keys.update(key.value for key in node.keys
                            if isinstance(key, ast.Constant) and isinstance(key.value, str)
                            and key.value.isupper())
    return keys


def scrub_environment(source, *, keep=()) -> dict[str, str]:
    """Drop owner configuration, credentials and process/import overrides.

    The caller names any test controls it wants to retain. Containment membership
    (OURO_PROC_CONTAINER_*) is deliberately inherited by every descendant.
    """
    projected = settings_keys()
    suffixes = ("_API_KEY", "_TOKEN", "_PASSWORD", "_CREDENTIALS", "_SECRET")
    prefixes = ("OUROBOROS_", "GH_", "GIT_", "PYTEST_", "PYTHON", "UV_", "PIP_")
    return {key: value for key, value in source.items()
            if key in keep or not (key.startswith(prefixes) or key.endswith(suffixes)
                                  or key in projected or key in {"NODE_OPTIONS", "VIRTUAL_ENV"})}


# Launch descriptors: launchd/XPC session ids, PyInstaller bootloader state and the
# shell's location/session bookkeeping. They change with every launch and configure
# nothing a test may rely on, so the hermetic gate neither passes nor fingerprints
# them (a durable test receipt must survive a Restart); every other key stays.
LAUNCH_ONLY_PREFIXES = ("XPC_", "_PYI_")
LAUNCH_ONLY_KEYS = frozenset({"PWD", "OLDPWD", "SHLVL", "_", "TERM_SESSION_ID", "SECURITYSESSIONID",
                              "LaunchInstanceID", "LAUNCHINSTANCEID"})


def without_launch_descriptors(env: dict) -> dict[str, str]:
    return {key: value for key, value in env.items()
            if key not in LAUNCH_ONLY_KEYS and not key.startswith(LAUNCH_ONLY_PREFIXES)}


# A disposable tree whose processes were not PROVEN gone keeps this file at its
# top. Every enclosing cleanup layer (candidate checkout, pytest session root,
# safe_test launcher) refuses to delete a tree that contains one: retention is
# in place, so a surviving process never loses the directory it is writing into.
RETENTION_MARKER = "OUROBOROS_RETAINED.txt"
SCRATCH_MARKER = "OUROBOROS_TEST_SCRATCH.txt"
# Explicit diagnostic destinations accepted by existing test producers. They
# carry output paths, not provider/configuration or test-selection authority.
DIAGNOSTIC_OUTPUT_KEYS = (
    "OUROBOROS_UI_EVIDENCE_DIR", "OUROBOROS_BROWSER_EVIDENCE_OUT",
    "OUROBOROS_UI_EVIDENCE_OUT", "HISTORY_UI_EVIDENCE_DIR",
    "OUROBOROS_UI_SCREENSHOT_DIR", "OUROBOROS_PUBLICATION_SCREENSHOTS", "OUROBOROS_CONTAINER_EVIDENCE",
    "OUROBOROS_UI_SHOTS", "OUROBOROS_UI_SMOKE_SHOT_DIR", "QUIZ_BROWSER_EVIDENCE",
)
# Where a run writes, not what it tests: never part of a reusable proof identity.
OUTPUT_LOCATION_KEYS = ("OUROBOROS_TEST_TEMP_ROOT", "OUROBOROS_TEST_EVIDENCE_ROOT", *DIAGNOSTIC_OUTPUT_KEYS)


def declare_scratch(root, role: str) -> None:
    """A fresh producer declares reproducibility, never liveness or deletion authority."""
    if role not in {"pytest-session", "pytest-basetemp", "body-process"}:
        raise ValueError(f"unknown test scratch role: {role}")
    with (Path(root) / SCRATCH_MARKER).open("x", encoding="utf-8") as stream:
        stream.write(role + "\n")


def retain_tree(root, reason: str) -> bool:
    """Mark ``root`` retained; False when the marker itself could not be written."""
    try:
        with (Path(root) / RETENTION_MARKER).open("a", encoding="utf-8") as stream:
            stream.write(reason.rstrip() + "\n")
        # A quiescent body environment may have recorded an empty walk. Publish
        # this nested retention to its addressed declaration before returning.
        for parent in Path(root).resolve().parents:
            declaration = parent / SCRATCH_MARKER
            try:
                os.utime(declaration, None)
            except FileNotFoundError:
                continue
    except OSError:
        return False
    return True


def retention_markers(root) -> list[Path]:
    """Markers beneath ``root``. A subtree that cannot be walked counts as one."""
    found: list[Path] = []
    for directory, _dirs, files in os.walk(
            root, followlinks=False,
            onerror=lambda error: found.append(Path(error.filename or root))):
        if RETENTION_MARKER in files:
            found.append(Path(directory) / RETENTION_MARKER)
    return found


def _retention_identity(path):
    try:
        seen = path.lstat()
        return [seen.st_dev, seen.st_ino, seen.st_mode, seen.st_size, seen.st_mtime_ns, seen.st_ctime_ns]
    except FileNotFoundError:
        return None
    except OSError:
        return "unreadable"


def scratch_retention(root, *, previous=None, classify=False) -> dict:
    """One walk per quiescent tree; later checks read only its recorded witness.

    A declaration proves reproducibility, not absence of nested retained readers.
    Unclassified/changed trees stay retained until an unlocked classification;
    undeclared legacy trees and unreadable subtrees never certify deletion.
    """
    root = Path(root)
    observed = {"root": _retention_identity(root),
                "declaration": _retention_identity(root / SCRATCH_MARKER)}
    empty = {**observed, "marker": "", "identity": None, "retained": False}
    if observed["root"] is None:
        return empty
    marker = root / RETENTION_MARKER
    identity = _retention_identity(marker)
    if identity is not None:
        return {**observed, "marker": RETENTION_MARKER, "identity": identity, "retained": True}
    if isinstance(previous, dict) and all(previous.get(key) == value for key, value in observed.items()):
        relative = previous.get("marker")
        if isinstance(relative, str) and isinstance(previous.get("retained"), bool) and (
                not relative or previous.get("identity") is not None
                and _retention_identity(root / relative) == previous["identity"]):
            return previous
    if (not classify or observed["root"] == "unreadable" or observed["declaration"] in (None, "unreadable")
            or root.is_symlink() or not root.is_dir()
            or (root / SCRATCH_MARKER).is_symlink() or not (root / SCRATCH_MARKER).is_file()):
        return {**empty, "retained": True}
    markers = retention_markers(root)
    if not markers:
        return empty
    marker = markers[0]
    return {**observed, "marker": str(marker.relative_to(root)),
            "identity": _retention_identity(marker), "retained": True}


def retain_diagnostic_outputs(root, env, *, cwd=None) -> None:
    """An explicit output inside this owner's scratch keeps that tree, not a dangling path."""
    root = Path(root).resolve()
    child_cwd = Path(cwd or Path.cwd())
    for key in DIAGNOSTIC_OUTPUT_KEYS:
        value = env.get(key)
        if value and (child_cwd / value).resolve().is_relative_to(root):
            if not retain_tree(root, f"explicit diagnostic destination inside scratch: {key}"):
                raise OSError(f"cannot retain selected diagnostic destination: {root}")
            return


def isolated_environment(root: Path, repo: Path, *, source=None, keep=(), create=True) -> dict[str, str]:
    """Create all writable defaults beneath a caller-owned disposable directory."""
    root, repo = Path(root).resolve(), Path(repo).resolve()
    env = scrub_environment(os.environ if source is None else source, keep=keep)
    paths = {
        "HOME": root / "home", "USERPROFILE": root / "home",
        "OUROBOROS_APP_ROOT": root / "app",
        "OUROBOROS_DATA_DIR": root / "data",
        "OUROBOROS_SUBAGENT_PROJECTS_ROOT": root / "projects",
        "OUROBOROS_SUBAGENT_WORKTREE_ROOT": root / "worktrees",
        "OUROBOROS_DELIVERABLES_ROOT": root / "Deliverables",
        "OUROBOROS_BENCH_RUNS_ROOT": root / "bench_runs",
        "PYTHONPYCACHEPREFIX": root / "pycache.noindex", "PYTHONUSERBASE": root / "userbase",
        "XDG_CACHE_HOME": root / "cache", "XDG_CONFIG_HOME": root / "config",
        "XDG_DATA_HOME": root / "xdg-data", "XDG_STATE_HOME": root / "xdg-state",
        "APPDATA": root / "config", "LOCALAPPDATA": root / "cache",
        "TMPDIR": root / "tmp", "TEMP": root / "tmp", "TMP": root / "tmp",
        # Chromium on macOS ignores TMPDIR for initial download staging.
        "MAC_CHROMIUM_TMPDIR": root / "tmp",
        "PYTEST_DEBUG_TEMPROOT": root / "tmp",
        "UV_CACHE_DIR": root / "cache" / "uv",
        "UV_PYTHON_INSTALL_DIR": root / "python",
        "PIP_CACHE_DIR": root / "cache" / "pip",
    }
    for key, path in paths.items():
        if create:
            path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env.update({
        "OUROBOROS_REPO_DIR": str(repo),
        "OUROBOROS_SETTINGS_PATH": str(root / "data" / "settings.json"),
        "OUROBOROS_PYTEST_ACTIVE": "1",
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        # No PYTHONUTF8 / PYTHONIOENCODING: forcing UTF-8 mode on every test and
        # commit-gate child would hide the cp1252 bugs Windows CI exists to catch.
        "LC_ALL": "C", "LANG": "C",
        "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_OPTIONAL_LOCKS": "0", "GIT_CEILING_DIRECTORIES": str(root),
        "PIP_REQUIRE_VIRTUALENV": "true",
    })
    return env
