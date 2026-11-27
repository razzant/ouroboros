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


def _short_temp_platform() -> bool:
    return sys.platform == "darwin" or sys.platform.startswith("linux")


def default_temp_parent() -> Path:
    """A short root keeps nested multiprocessing sockets inside AF_UNIX's path limit (104 bytes
    on macOS, 108 on Linux): rooted in an inherited, already nested TMPDIR, a nested forkserver
    or manager address outgrew it (#2110). Other platforms keep their temporary directory.

    The parent of preflight, commit-admission and safe_test roots. New task-owned kept preflight
    containers have an ownership record; unrecorded roots still have only the OS's temp cleaners."""
    return Path("/tmp") if _short_temp_platform() else Path(tempfile.gettempdir())


def session_scratch_parent() -> Path:
    """Where a writable session's recorded scratch (``workspace_preflight._session_scratch``) lives:
    ``/var/tmp`` on macOS and Linux, four bytes longer than ``default_temp_parent()`` and still
    inside the socket bound (a nested ``w/`` layout binds at 97 of 104 bytes under
    ``/private/var/tmp``), outside the macOS ``/tmp`` cleaners and Linux ``tmpfs``. Host release
    follows its holder, not age; system policy can still age ``/var/tmp`` (see PERSISTENCE).
    Other platforms keep their temporary directory."""
    return Path("/var/tmp") if _short_temp_platform() else Path(tempfile.gettempdir())


def git_ancestor(path: Path):
    """The nearest directory at or above ``path`` that Git would treat as a checkout.

    safe_test refuses a disposable parent with one; delegated-session facts name the same refusal."""
    for directory in (path, *path.parents):
        if (directory / ".git").exists():
            return directory
    return None


# A strict socket consumer (#4093) refuses a socket below any directory others may write, which ``/var/tmp`` is,
# sticky or not. A full in-place run's NEW scratch is therefore one unambiguously named child of the physical home,
# ``<home>/.obs-<8 chars>`` (``session_scratch_allocation.prepare_session_scratch``, made atomically 0700 by ``mkdtemp``, so no
# shared parent is ever published or judged), when the home qualifies (``private_scratch_problem``).
PRIVATE_SCRATCH_PREFIX = ".obs-"
SUN_PATH = 104 if sys.platform == "darwin" else 108  # AF_UNIX path bytes, the NUL included
# The deepest real nested manager socket below the home: ``/.obs-<8>/w`` (16 B), safe_test's ``/ob-<8>.noindex``
# (20 B, ``scripts/safe_test.py``), pytest's ``/p<8>/tmp`` (14 B, ``tests/conftest.py``), and CPython's
# ``/pymp-<8>/listener-<8>`` (32 B): 82 bytes total, so AF_UNIX's final bind limit (``SUN_PATH - 1``) leaves at most
# 21 bytes for the home on macOS (104) and 25 on Linux (108). Separately, CPython on macOS can enforce its own
# stricter ``TMPDIR`` check in ``multiprocessing`` and fall back to another temp dir below the raw AF_UNIX limit.
SOCKET_LAYOUT = "/.obs-12345678/w/ob-12345678.noindex/p12345678/tmp/pymp-12345678/listener-12345678"

# Known OS-managed path families a private home must not sit under: the shared temp cleaners
# (``/tmp``, systemd-tmpfiles ``/var/tmp``), the per-user runtime services (Linux ``/run``, with
# ``/run/user/<uid>`` under it; ``/var/run`` its usual physical alias) and macOS's per-user temp
# trees (``/var/folders``), plus the absolute folders this process's ``OS_TEMP_VARIABLES`` name. Judged
# component-wise on the given and symlink-resolved spelling of both sides, then by bounded directory
# identity (``st_dev``, ``st_ino``) across those already-selected existing roots and ancestors so a
# case-insensitive volume alias is also caught while distinct directories on case-sensitive volumes stay
# separate; ``/`` is never a family. The variables only veto a NEW placement: no record's release depends on
# them. There is no general mount/ACL judgment, and a home under some other site-local ephemeral mount is
# judged by its chain alone (a residual).
OS_EPHEMERAL_ROOTS = (Path("/tmp"), Path("/var/tmp"), Path("/run"), Path("/var/run"), Path("/var/folders"))
OS_TEMP_VARIABLES = ("TMPDIR", "TEMP", "TMP", "XDG_RUNTIME_DIR")


def ephemeral_ancestor(path) -> str:
    """Why PATH is, or sits under, a known OS-managed root (``OS_EPHEMERAL_ROOTS``) or an absolute folder this
    process's ``OS_TEMP_VARIABLES`` name, whether matched lexically (including symlink aliases) or by bounded
    directory identity (``st_dev``, ``st_ino``) among existing selected roots and ancestors (covering
    case-insensitive volume aliases without conflating distinct directories on case-sensitive volumes), else "":
    a private home there would be cleaned with the OS's own temp state. Reads only (containment by path
    components, never a string prefix: a ``/tmporary`` is no OS root); ordinary ``OSError`` on stat returns a
    refusal so callers keep shared fallback; nothing is created, made writable or promised here, and ``/`` is
    never a family."""
    try:
        named = [("the OS-managed root", root) for root in OS_EPHEMERAL_ROOTS]
        for variable in OS_TEMP_VARIABLES:
            value = os.environ.get(variable, "")
            if os.path.isabs(value):  # a relative or empty value names no folder to judge
                named.append((f"${variable}", Path(value)))
        resolved_named = [(label, root, Path(os.path.realpath(root))) for label, root in named]
        families = [(label, spelling) for label, root, resolved in resolved_named for spelling in (root, resolved)]
        spellings = (Path(path), Path(os.path.realpath(path)))
    except (OSError, RuntimeError, ValueError):  # a path that cannot even be resolved names no family
        return f"{path} cannot be resolved to judge its OS-managed family"
    for spelling in spellings:
        for label, root in families:
            if root != root.parent and (root == spelling or root in spelling.parents):
                return f"{spelling} is {'' if root == spelling else 'under '}{label} {root}"
    root_ids: dict[tuple[int, int], tuple[str, Path]] = {}
    for label, root, resolved in resolved_named:
        if root == root.parent or resolved == resolved.parent:
            continue
        try:
            info = os.stat(root)
        except (FileNotFoundError, NotADirectoryError):
            continue
        except OSError as exc:
            return f"{root} cannot be read ({type(exc).__name__})"
        if (info.st_mode & 0o170000) == 0o040000 and info.st_ino:
            root_ids.setdefault((info.st_dev, info.st_ino), (label, root))
    if root_ids:
        seen_dirs: set[Path] = set()
        for spelling in (spellings[0],) if spellings[0] == spellings[1] else spellings:
            for directory in (spelling, *spelling.parents):
                if directory == directory.parent or directory in seen_dirs:
                    continue
                seen_dirs.add(directory)
                try:
                    info = os.stat(directory)
                except (FileNotFoundError, NotADirectoryError):
                    continue
                except OSError as exc:
                    return f"{directory} cannot be read ({type(exc).__name__})"
                if (info.st_mode & 0o170000) == 0o040000 and (match := root_ids.get((info.st_dev, info.st_ino))):
                    label, root = match
                    return f"{spelling} is {'' if directory == spelling else 'under '}{label} {root}"
    return ""


def _own_uid() -> int:
    return os.geteuid() if hasattr(os, "geteuid") else -1  # a private home exists only where users have one


def private_scratch_home() -> Path | None:
    """The physical home (HOME honoured) on macOS and Linux, else None. A name only, never judged or made here."""
    home = os.path.expanduser("~") if _short_temp_platform() else ""
    try:
        return Path(home).resolve() if os.path.isabs(home) else None
    except (OSError, RuntimeError):
        return None


def unsafe_ancestor(path, *, absent_ok: bool = True) -> str:
    """Why an existing component from PATH up to ``/`` fails the POSIX owner/mode check for a private ancestor, nearest
    first, else "": each must be a real directory owned by root or this user with no group or other write bit (a
    sticky bit does not excuse). ACLs, mount options and effective access are not examined. A component that does not
    exist is skipped, a leaf not made yet, unless ABSENT_OK is False. Reads only."""
    for directory in (Path(path), *Path(path).parents):
        try:
            seen = os.lstat(directory)
        except FileNotFoundError:
            if absent_ok:
                continue
            return f"{directory} does not exist"
        except OSError as exc:
            return f"{directory} cannot be read ({type(exc).__name__})"
        if (seen.st_mode & 0o170000) != 0o040000 or seen.st_uid not in (0, _own_uid()) or seen.st_mode & 0o022:
            return f"{directory} is not a directory only root or you can write"
    return ""


def issued_scratch_problem(scratch) -> str:
    """Why the ISSUED SCRATCH (``<home>/.obs-<id>/w``) fails the POSIX owner/mode check now, else "": the chain from
    ``/`` through the scratch itself is read as it exists, its container and ``w`` included (``unsafe_ancestor``
    without an absent leaf). A missing, linked, foreign, group/other-writable or unreadable component is the
    problem; a read that failed claims nothing."""
    return f"{scratch} is not an absolute path" if not os.path.isabs(scratch) else unsafe_ancestor(scratch, absent_ok=False)


def private_scratch_problem(home) -> str:
    """Why a NEW private scratch may not be made under HOME, or "" when it may; reads only, nothing is created.
    (f) HOME round-trips strict UTF-8 (no surrogate-escaped filesystem bytes); (d) the deepest nested socket fits
    under AF_UNIX's limit, counted in BYTES; (e) no known OS-managed ephemeral family (``ephemeral_ancestor``);
    (a) no Git ancestor; (b) every existing component from HOME to ``/`` passes ``unsafe_ancestor``. A failed read
    of any is the problem, never an escaped error."""
    room = SUN_PATH - 1 - len(SOCKET_LAYOUT)
    if home is None or home == home.parent:
        return "no home directory"
    text = str(home)
    try:
        encoded = os.fsencode(text)
        if encoded != text.encode("utf-8") or encoded.decode("utf-8") != text:
            return "home path cannot round-trip strict UTF-8"
    except UnicodeError:
        return "home path cannot round-trip strict UTF-8"
    if (size := len(encoded)) > room:
        return f"{home} is {size} bytes, over the {room} a nested socket leaves"
    if problem := ephemeral_ancestor(home):
        return problem
    try:
        ancestor = git_ancestor(home)
    except OSError as exc:  # Python 3.10 ``Path.exists`` raises on an unreadable directory
        return f"{home} cannot be read ({type(exc).__name__})"
    if ancestor is not None:
        return f"{home} is inside the Git working tree {ancestor}"
    return unsafe_ancestor(home)


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
                                  or key in projected or key in {"NODE_OPTIONS", "VIRTUAL_ENV", "PLAYWRIGHT_NODEJS_PATH"})}


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
# safe_test launcher) leaves the folder that contains one (``retained_folders``):
# retention is in place, so a surviving process never loses the directory it writes into.
RETENTION_MARKER = "OUROBOROS_RETAINED.txt"
SCRATCH_MARKER = "OUROBOROS_TEST_SCRATCH.txt"
# Logical ownership is outside the workspace handed to a session or a top-level gate.
SCRATCH_CONTAINER_MARKER = ".ouroboros-scratch"
SCRATCH_WORKSPACE = "w"


def new_scratch_identity(container) -> dict:
    """Identity issued only by a fresh allocator; never infer a nonce from an existing folder."""
    return {"schema_version": 2, "path": str(container), "nonce": os.urandom(16).hex()}


def write_scratch_marker(container, nonce) -> None:
    """Publish a new local allocation's marker, before any workspace is handed out."""
    with (Path(container) / SCRATCH_CONTAINER_MARKER).open("xb") as stream:
        stream.write((nonce + "\n").encode("ascii"))
        stream.flush()
        os.fsync(stream.fileno())


# Explicit diagnostic destinations accepted by existing test producers. They
# carry output paths, not provider/configuration or test-selection authority.
DIAGNOSTIC_OUTPUT_KEYS = (
    "OUROBOROS_UI_EVIDENCE_DIR", "OUROBOROS_BROWSER_EVIDENCE_OUT",
    "OUROBOROS_UI_EVIDENCE_OUT", "HISTORY_UI_EVIDENCE_DIR",
    "OUROBOROS_UI_SCREENSHOT_DIR", "OUROBOROS_PUBLICATION_SCREENSHOTS", "OUROBOROS_CONTAINER_EVIDENCE",
    "OUROBOROS_UI_SHOTS", "OUROBOROS_UI_SMOKE_SHOT_DIR", "QUIZ_BROWSER_EVIDENCE",
)
# Where a run writes, not what it tests: never part of a reusable proof identity.
OUTPUT_LOCATION_KEYS = ("OUROBOROS_TEST_TEMP_ROOT", "OUROBOROS_TEST_EVIDENCE_ROOT", "OUROBOROS_TEST_DURATIONS",
                        *DIAGNOSTIC_OUTPUT_KEYS)


def declare_scratch(root, role: str, *, owner: str = "") -> None:
    """A fresh producer declares its tree disposable state it made: purpose, never liveness, deletion
    authority or a promise that the contents come back byte for byte.
    ``owner``: a one-line fact naming what it was made for (``copy_environments`` reads it)."""
    if role not in {"pytest-session", "pytest-basetemp", "body-process", "process-temp", "bytecode-prefix"}:
        raise ValueError(f"unknown test scratch role: {role}")
    with (Path(root) / SCRATCH_MARKER).open("x", encoding="utf-8") as stream:
        stream.write(role + "\n" + (owner.strip() + "\n" if owner else ""))


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


def marker_text(found) -> str:
    """What a marker says; a subtree that could not be walked has no words of its own."""
    try:
        return Path(found).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return "retention marker or unreadable subtree"


def _kept_folder(root: Path, folder: Path) -> Path:
    """The folder retention at ``folder`` keeps: its test's folder in a declared pytest basetemp
    (``b<N>/t/<test>``, or ``b<N>/t/popen-gw<K>/<test>`` in an xdist worker's own basetemp), else the
    direct child of ``root``. Never deeper: a test's processes work anywhere in its folder."""
    parts = folder.relative_to(root).parts
    try:
        with (root / parts[0] / SCRATCH_MARKER).open(encoding="utf-8") as stream:
            basetemp = stream.readline().strip() == "pytest-basetemp"
    except (IndexError, OSError, UnicodeError):
        basetemp = False
    depth = 1 if not basetemp else 4 if parts[2:3] and parts[2].startswith("popen-") else 3
    return root.joinpath(*parts[:depth])


def retained_folders(root) -> dict[Path, list[Path]]:
    """``{kept folder: [markers or unreadable subtrees]}`` beneath a run root: a marker keeps only its own
    folder (``_kept_folder``), and one at the top of ``root`` all of it (then ``root`` is a key)."""
    root, kept = Path(root), {}
    for found in retention_markers(root):
        folder = found.parent if found.name == RETENTION_MARKER else found
        kept.setdefault(_kept_folder(root, folder), []).append(found)
    return kept


def kept_facts(kept) -> list[dict]:
    """``retained_folders`` as a record: each kept path with each marker's first line."""
    return [{"path": str(folder), "reasons": [marker_text(found).split("\n")[0] for found in markers]}
            for folder, markers in kept.items()]


def release_except_retained(root, kept, remove, *, declared_only=False) -> list[str]:
    """Remove what ``root`` holds beyond ``kept`` (``retained_folders``); the removed paths.

    The caller proved its run's processes gone. ``remove`` takes a directory tree; links go as links,
    never followed. ``declared_only`` leaves direct children no scratch declaration names: unknown
    command data, links included. The folders on the way to a kept one stay, declared as they were."""
    released: list[str] = []
    passed = {parent for folder in kept for parent in folder.parents}

    def release(directory: Path, top: bool) -> None:
        for entry in list(os.scandir(directory)):
            path, is_dir = Path(entry.path), entry.is_dir(follow_symlinks=False)
            if path in kept or entry.name == SCRATCH_MARKER:
                continue
            if path in passed:
                release(path, False)
            elif not (top and declared_only) or is_dir and (path / SCRATCH_MARKER).is_file():
                if is_dir:
                    remove(path)
                else:
                    os.unlink(path)
                released.append(str(path))

    if Path(root) not in kept:
        release(Path(root), True)
    return released


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

    A declaration states purpose, not absence of nested retained readers.
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


def owned_process_temp():
    """A fresh declared temp for one real process tree, directly beneath the run's owner root.

    TMPDIR decides where a child's multiprocessing listens, and AF_UNIX caps that path (104 bytes on
    macOS, 108 on Linux; on macOS, CPython multiprocessing can also reject a long TMPDIR and fall back
    to another temp dir below the raw AF_UNIX bind limit). A case's descriptive tmp_path is too deep, so only its temp moves here: HOME, settings and
    data stay in the case. The owner OUROBOROS_TEST_TEMP_ROOT names (safe_test, the commit gate)
    releases it with its run once every process is proven gone, never at a case's stop or restart.
    An explicit long --temp-parent stays the caller's choice and fails where a socket does not fit.
    None without that owner: bare pytest's ambient temp is nobody's to vouch for.
    """
    owner = os.environ.get("OUROBOROS_TEST_TEMP_ROOT")
    if not owner:
        return None
    temp = Path(tempfile.mkdtemp(prefix="c", dir=owner))
    declare_scratch(temp, "process-temp")
    return temp


def gate_root(*, evidence=None) -> Path:
    """One gate run's fresh root. A gate a test run starts lives beneath that run's owner, whose release
    covers what it leaves, and short: the deepest listener stays inside AF_UNIX's limit. A top-level
    gate has the same ``obs-*`` container and outside marker as session scratch; its per-run evidence
    carries the issued identity to ``test_evidence.finish_gate``, never a later read of an unknown folder.
    Debug retention holds the root before anything runs (a test's folder goes unless its root is held)."""
    nested = owned_process_temp()
    if nested is not None:
        root = nested.resolve()  # its outer owner already holds and releases the entire allocation
    else:
        container = Path(tempfile.mkdtemp(prefix="obs-preflight-", dir=default_temp_parent())).resolve()
        identity = new_scratch_identity(container)
        write_scratch_marker(container, identity["nonce"])
        root = container / SCRATCH_WORKSPACE
        root.mkdir()
        if evidence is not None:
            evidence["scratch_ownership"] = identity
    if os.environ.get("OUROBOROS_PREFLIGHT_KEEP") == "1":
        retain_tree(root, "explicit preflight debug retention")
    return root


def bundled_git_paths(env) -> dict[str, str]:
    """The app's macOS/Linux Git (``git-standalone``, ouroboros/git_selection.py) first on ``env``'s PATH looks for
    its helpers and templates under ``/`` unless told; scrubbing every ``GIT_*`` must not take those two away."""
    for folder in env.get("PATH", "").split(os.pathsep):
        git = Path(folder) / "git"
        if folder and git.is_file() and os.access(git, os.X_OK):  # the Git a bare ``git`` runs
            root = git.resolve().parent.parent
            if root.name != "git-standalone" or not (root / "libexec" / "git-core").is_dir():
                return {}  # the person's Git knows its own paths
            return {"GIT_EXEC_PATH": str(root / "libexec" / "git-core"),
                    "GIT_TEMPLATE_DIR": str(root / "share" / "git-core" / "templates")}
    return {}  # none, or Windows: MinGit finds its own tree


def isolated_environment(root: Path, repo: Path, *, source=None, keep=(), create=True,
                         process_temp=None) -> dict[str, str]:
    """Create all writable defaults beneath a caller-owned disposable directory.

    ``process_temp``, the caller's own (``owned_process_temp``), replaces only the temporary directories."""
    root, repo = Path(root).resolve(), Path(repo).resolve()
    temp = root / "tmp" if process_temp is None else Path(process_temp).resolve()
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
        "TMPDIR": temp, "TEMP": temp, "TMP": temp,
        # Chromium on macOS ignores TMPDIR for initial download staging.
        "MAC_CHROMIUM_TMPDIR": temp,
        "PYTEST_DEBUG_TEMPROOT": temp,
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
        "GIT_OPTIONAL_LOCKS": "0",
        # A git started in a process temp outside ``root`` stops there too.
        "GIT_CEILING_DIRECTORIES": os.pathsep.join(dict.fromkeys((str(root), str(temp.parent)))),
        "PIP_REQUIRE_VIRTUALENV": "true",
        **bundled_git_paths(env),
    })
    return env
