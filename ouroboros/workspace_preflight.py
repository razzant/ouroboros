"""Read-only external workspace preflight helpers, session environment facts and the
service directories a writable session's engine may grant besides its copy."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import pathlib
import platform
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid
from typing import Any, Dict, Iterable, List, NamedTuple, Tuple

from ouroboros.platform_layer import (
    IS_LINUX,
    IS_MACOS,
    IS_WINDOWS,
    bootstrap_process_path,
    node_runtime_health,
    resolve_bundled_node,
)
from ouroboros.utils import utc_now_iso


_CURATED_TOOLS = (
    "git",
    "python",
    "python3",
    "pip",
    "pip3",
    "pytest",
    "uv",
    "node",
    "npm",
    "npx",
    "pnpm",
    "yarn",
    "go",
)
_MACOS_PACKAGE_TOOLS = (
    "brew",
)
_LINUX_PACKAGE_TOOLS = (
    "apt-get",
    "apt",
    "dnf",
    "yum",
    "apk",
    "pacman",
)
_WINDOWS_PACKAGE_TOOLS = (
    "winget",
)
# The exec probe covers ONLY node (the incident class: `which` succeeds but the
# kernel SIGKILLs the binary on launch). npm/npx/pnpm/yarn follow node — they
# get no exec probes of their own; python/git/other tools are out of class.
# The probe timeout is SHORT so the bounded admission path (8s wall-clock cap in
# workspace_admission.bounded_workspace_preflight) stays safe; verdicts are
# memoized in the platform layer.
_NODE_PROBE_TIMEOUT_SEC = 3.0
_NODE_DEPENDENT_TOOLS = ("npm", "npx", "pnpm", "yarn")
_MANIFEST_NAMES = (
    "package.json",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    "requirements-runtime.lock",
    "uv.lock",
    "tox.ini",
    "go.mod",
    "Cargo.toml",
    "Makefile",
)


def collect_workspace_preflight(workspace_root: pathlib.Path) -> Dict[str, Any]:
    """Collect a compact, read-only snapshot of a target workspace."""

    added_dirs = bootstrap_process_path()
    root = pathlib.Path(workspace_root).resolve(strict=False)
    manifests = _collect_manifests(root)
    inferred_tools = sorted(set(_infer_tools_from_manifests(manifests)))
    probe_tools = sorted(set(_platform_curated_tools()).union(inferred_tools))
    tools = _probe_tools(probe_tools)
    git = _git_snapshot(root)
    return {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "workspace_root": str(root),
        "git": git,
        "manifests": manifests,
        "tools": tools,
        "path_bootstrap": {
            "added_dirs": added_dirs,
        },
    }


def summarize_workspace_preflight(preflight: Dict[str, Any]) -> Dict[str, Any]:
    """Return the task-metadata/prompt-safe preflight summary."""

    git = preflight.get("git") if isinstance(preflight.get("git"), dict) else {}
    manifests = preflight.get("manifests") if isinstance(preflight.get("manifests"), list) else []
    tools = preflight.get("tools") if isinstance(preflight.get("tools"), dict) else {}
    available = sorted(k for k, v in tools.items() if isinstance(v, dict) and v.get("available"))
    broken_names = sorted(
        k for k, v in tools.items()
        if isinstance(v, dict)
        and not v.get("available")
        and v.get("status") in ("found_but_broken", "interpreter_broken")
    )
    missing = sorted(
        k for k, v in tools.items()
        if isinstance(v, dict) and not v.get("available") and k not in set(broken_names)
    )
    broken: List[Dict[str, str]] = []
    for name in broken_names:
        entry = tools[name]
        reason = str(entry.get("probe_reason") or entry.get("reason") or entry.get("status") or "broken")
        if entry.get("bundled_fallback"):
            reason += "; bundled fallback available"
        broken.append({"tool": name, "reason": reason})
    manifest_summary = []
    for item in manifests[:20]:
        if not isinstance(item, dict):
            continue
        manifest_summary.append({
            "path": item.get("path", ""),
            "type": item.get("type", ""),
            "scripts": list(item.get("scripts", []) or [])[:20],
        })
    return {
        "schema_version": 1,
        "workspace_root": str(preflight.get("workspace_root") or ""),
        "git": {
            "head": str(git.get("head") or ""),
            "head_state": str(git.get("head_state") or ("known" if git.get("head") else "unknown")),
            "branch": str(git.get("branch") or ""),
            "dirty": bool(git.get("dirty")),
            "status_count": int(git.get("status_count") or 0),
        },
        "manifests": manifest_summary,
        "tools": {
            "available": available,
            "missing": missing,
            # Present only when a which-found tool failed its exec probe (or
            # follows a broken interpreter) — the healthy-path summary stays
            # byte-identical.
            **({"broken": broken} if broken else {}),
        },
    }


def render_workspace_preflight_summary(summary: Dict[str, Any]) -> str:
    """Render a stable short block for the headless task prompt."""

    git = summary.get("git") if isinstance(summary.get("git"), dict) else {}
    tools = summary.get("tools") if isinstance(summary.get("tools"), dict) else {}
    manifests = summary.get("manifests") if isinstance(summary.get("manifests"), list) else []
    lines = [
        "workspace_preflight:",
        f"- git_head: {git.get('head') or '<unknown>'}",
        f"- git_branch: {git.get('branch') or '<unknown>'}",
        f"- git_dirty: {bool(git.get('dirty'))} ({int(git.get('status_count') or 0)} status entries)",
        "- manifests: "
        + (", ".join(str(item.get("path") or "") for item in manifests if isinstance(item, dict)) or "<none detected>"),
        # "on PATH" is what shutil.which actually measured; "available" over-
        # claimed executability (a broken binary on PATH counted as available).
        # The structured summary keys stay unchanged — they ride durable
        # task metadata and old rows must keep replaying.
        "- tools_on_path: " + (", ".join(tools.get("available") or []) or "<none detected>"),
        "- tools_missing_from_path: " + (", ".join(tools.get("missing") or []) or "<none detected>"),
    ]
    broken = [item for item in (tools.get("broken") or []) if isinstance(item, dict)]
    if broken:
        # Which-found but execution-probe-dead tools: NOT available, and the
        # reason (plus a bundled-node fallback fact, when one exists) is what
        # keeps the agent from treating the miss as a syntax/config problem.
        lines.append(
            "- broken_tools: "
            + ", ".join(
                f"{item.get('tool') or '<tool>'} ({item.get('reason') or 'broken'})"
                for item in broken
            )
        )
    if summary.get("error"):
        lines.append(f"- preflight_error: {summary.get('error')}")
    return "\n".join(lines)


# The body documents its dependency-only bootstrap once, in this DEVELOPMENT section;
# session facts point at it inside the candidate and never restate its commands.
_BODY_BOOTSTRAP_SECTION = "Safe local verification and dirty browser candidates"
_SESSION_TOOLS = ("uv", "git", "python3", "node", "npm")
# The Ouroboros-owned download cache that session bootstraps share (owner choice A, quiz 44dd72,
# 2026-10-21: one shared cache, never a personal cache, a shared environment or a toolchain
# location, and no bespoke cleaner). Each package manager locks it, and it is pruned only when
# someone runs that manager's own prune explicitly; the host only names it.
PACKAGE_DOWNLOADS_DIRNAME = "package-downloads"
_CACHE_VARIABLES = (("uv", "UV_CACHE_DIR"), ("pip", "PIP_CACHE_DIR"), ("npm", "npm_config_cache"))
_LISTED_DEPENDENCY_FILES = 16
SNAPSHOT_ENVIRONMENT_NOTE = "inside this copy, where a top-level .venv and any node_modules are never captured as work product"
ENVIRONMENT_FACTS_HEADING = (
    "\n\nPROJECT ENVIRONMENT (host facts observed when this session started; nothing was installed for it)\n")


def package_downloads_root(data_root: Any) -> pathlib.Path:
    """The shared package download cache under the canonical data root (PERSISTENCE)."""
    return pathlib.Path(data_root) / PACKAGE_DOWNLOADS_DIRNAME


# The starting task's drive holds ownership records, never the physical scratch: nesting
# safe_test beneath that long path breaks AF_UNIX consumers on macOS (PERSISTENCE).
SESSION_SCRATCH_DIRNAME = "session_scratch_owners"
# The typed limitation of a full-access start whose scratch could not be made: no grant was ever due.
SCRATCH_UNAVAILABLE = "scratch_unavailable"
# Under the canonical root, ``<task_id>/<key>.json`` names each record's folder while its task may still
# reuse the scratch; the custody pass releases it once that task is over (``release_finished_scratch``).
OPEN_SCRATCH_DIR = pathlib.Path("state") / "session_scratch"


class ServiceDirectories(NamedTuple):
    """What ONE writable session start may write outside its copy, as its engine negotiated (#1775).

    ``roots`` (the scratch, then the download cache's uv/pip/npm folders) ride the start request
    as ``execution.auxiliaryWritableRoots``, and ``scratch`` is the place its instructions name:
    at full access the host still owns and names one, but ``roots`` stays empty, since full
    access needs no grant; ``limitation`` says, typed, why a writable session got none. All
    empty: nothing to make (read-only, or a ``workspace_write`` ``tool_policy`` route, whose
    on-demand behavior needs no engine grant)."""

    roots: Tuple[str, ...] = ()
    scratch: str = ""
    limitation: str = ""


def _route_row(gateway: Any, route_id: str) -> Dict[str, Any]:
    """The route's live catalog row; ``{}`` when it is missing or the catalog cannot be read."""
    try:
        rows = gateway.agent_capabilities().get("harnesses")
    except Exception:
        return {}
    if not isinstance(rows, list):
        return {}
    return next((item for item in rows if isinstance(item, dict) and item.get("id") == route_id), {})


def route_grants_auxiliary_roots(row: Dict[str, Any]) -> bool:
    """The catalog bit ``accessControl.auxiliaryWritableRoots``, never a guess from the route's name,
    model, write mechanism or environment: an engine that predates the row negotiates nothing."""
    control = row.get("accessControl")
    return isinstance(control, dict) and control.get("auxiliaryWritableRoots") is True


def _owned_directory(base: Any, parts: Tuple[str, ...], execution: pathlib.Path) -> pathlib.Path:
    """``base``/``parts`` as real directories, each made empty when missing and never reached through
    a link, outside every Git working tree and apart from the execution tree, or ``ValueError``."""
    from ouroboros.test_environment import git_ancestor

    if not str(base or "").strip():
        raise ValueError("no host root to hold it")
    path = pathlib.Path(base).resolve()  # the host's own root is trusted; nothing below it may be a link
    path.mkdir(parents=True, exist_ok=True)
    for part in parts:
        if pathlib.Path(part).name != part or part in ("", ".", ".."):
            raise ValueError(f"unsafe folder name {part!r}")
        path = path / part
        try:
            os.mkdir(path)
        except FileExistsError:
            pass
        if not stat.S_ISDIR(os.lstat(path).st_mode):  # a link, even to a folder, is never granted
            raise ValueError(f"{path} is not a real directory")
    ancestor = git_ancestor(path)
    if ancestor is not None:
        raise ValueError(f"{path} is inside the Git working tree {ancestor}")
    if path == execution or path in execution.parents or execution in path.parents:
        raise ValueError(f"{path} overlaps the execution tree {execution}")
    return path


def _recorded_scratch(record: pathlib.Path) -> pathlib.Path:
    """Resolve only this host-created short directory, never a link or a replacement inode."""
    from ouroboros.test_environment import default_temp_parent

    if record.is_symlink():
        raise ValueError(f"{record} is not a real ownership record")
    row = json.loads(record.read_text(encoding="utf-8"))
    path = pathlib.Path(row["path"])
    if path.parent != default_temp_parent().resolve() or not path.name.startswith("obs-"):
        raise ValueError(f"{record} does not name a session scratch directory")
    seen = path.lstat()  # a missing temporary directory is a gap, never silently re-created
    if not stat.S_ISDIR(seen.st_mode) or [seen.st_dev, seen.st_ino] != row["identity"]:
        raise ValueError(f"{path} is not the recorded scratch directory")
    return path


def _session_scratch(owner: pathlib.Path, execution: pathlib.Path) -> pathlib.Path:
    """One short physical directory; its record stays in the ungranted task drive until its task is
    over (``release_finished_scratch``) or the drive retires, whichever comes first.

    A same-task/key start reuses it. Literal retries never call this function. Allocation is
    serialized using the existing file lock; no process owns an in-memory cleanup obligation.
    """
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
    from ouroboros.test_environment import default_temp_parent, git_ancestor
    from ouroboros.utils import atomic_write_json

    lock = owner / "prepare.lock"
    fd = acquire_exclusive_file_lock(lock, timeout_sec=5, stale_sec=120, owner_aware_stale=True)
    if fd is None:
        raise OSError(f"session scratch preparation is busy: {owner}")
    try:
        record = owner / "scratch.json"
        if os.path.lexists(record):
            try:
                path = _recorded_scratch(record)
            except FileNotFoundError:
                pass  # a new start may allocate after OS aging; literal retry never comes here
            else:
                if git_ancestor(path) is not None or path == execution or path in execution.parents or execution in path.parents:
                    raise ValueError(f"{path} overlaps a Git working tree or the execution root")
                return path  # never recreate a path between identity validation and granting it
        parent = default_temp_parent().resolve()
        if git_ancestor(parent) is not None:
            raise ValueError(f"{parent} is inside a Git working tree")
        path = pathlib.Path(tempfile.mkdtemp(prefix="obs-", dir=parent))
        try:
            _owned_directory(parent, (path.name,), execution)
            seen = path.lstat()
            atomic_write_json(record, {"schema_version": 1, "path": str(path), "identity": [seen.st_dev, seen.st_ino]})
        except Exception:
            path.rmdir()  # not yet granted; only this allocation's empty directory
            raise
        return path
    finally:
        release_exclusive_file_lock(lock, fd)


def release_session_scratch(drive: pathlib.Path) -> List[str]:
    """Release short scratch whose ownership task custody retired: a drive's, or a finished task's.

    Called by the trash remover, including nested child-drive layouts. A producer retention marker
    keeps that scratch where it is, and its path is returned: from then on nothing owns it, like any
    other retained test root, so a custodied drive is not held for it. An unreadable record, changed
    identity or incomplete removal keeps the trash and its pointer for the next pass. No temp-root
    scan, arbitrary symlink following or independent sweeper.
    """
    from ouroboros.force_remove import force_rmtree
    from ouroboros.test_environment import retention_markers

    if drive.is_symlink():
        raise OSError(f"linked scratch ownership drive: {drive}")
    owners = drive / SESSION_SCRATCH_DIRNAME
    if owners.is_symlink():
        raise OSError(f"linked scratch ownership directory: {owners}")
    retained: List[str] = []
    if owners.is_dir():
        for owner in owners.iterdir():
            if owner.is_symlink() or not owner.is_dir():
                raise OSError(f"invalid scratch owner: {owner}")
            record = owner / "scratch.json"
            if not os.path.lexists(record):  # an uncompleted preparation
                continue
            try:
                path = _recorded_scratch(record)
            except FileNotFoundError:  # the OS or a prior pass already removed this scratch
                continue
            except (ValueError, KeyError, TypeError) as exc:
                raise OSError(f"invalid scratch ownership record: {record}") from exc
            if retention_markers(path):
                retained.append(str(path))  # the producer's to keep; never deleted here
                continue
            force_rmtree(path)
            if os.path.lexists(path):
                raise OSError(f"session scratch removal incomplete: {path}")
    # These are the existing child layouts, never arbitrary folders of a drive.
    for relative, suffix in (("task_drives", ()), ("state/headless_tasks", ("data",))):
        children = drive / relative
        if (children.is_dir() and not children.is_symlink()
                and all(not parent.is_symlink() for parent in children.parents if parent != drive and drive in parent.parents)):
            for child in children.iterdir():
                nested = child.joinpath(*suffix)
                if child.is_dir() and not child.is_symlink() and nested.is_dir() and not nested.is_symlink():
                    retained += release_session_scratch(nested)
    return retained


def _note_open_scratch(data_root: Any, task_id: str, key: str, owner: pathlib.Path) -> None:
    """Name ``owner`` under its task in the canonical root (``OPEN_SCRATCH_DIR``). Best-effort: a
    missing note only leaves the scratch to its drive's retirement, as before it existed."""
    from ouroboros.utils import atomic_write_json

    if not str(data_root or "").strip():
        return
    note = pathlib.Path(data_root) / OPEN_SCRATCH_DIR / task_id / f"{key}.json"
    try:
        if not note.is_file():
            note.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(note, {"schema_version": 1, "owner": str(owner)})
    except OSError:
        logging.getLogger(__name__).warning("Open session scratch note not written: %s", note, exc_info=True)


def release_finished_scratch(data_root: Any, live: Any, *, stop: Any = None) -> Dict[str, Any]:
    """Hand the session scratch of every task proven over to the custody trash (#2586).

    A same-task start reuses its scratch while the task lives (``_session_scratch``), so the scratch
    waits for its task, not for its drive's retention term: terminal, not live by the supervisor's
    probe ``live``, no lineage process (``subagent_worktrees._task_proven_over``) and not kept by the
    owner (``owner_keep.gate``, the recorded facts read again inside). Each noted record's folder
    moves, cheaply, into ``state/custody_trash``, whose remover deletes the scratch in the maintenance
    child, off the serving process (``release_session_scratch``). A note whose folder already retired
    with its drive, or that names no record of its task, is dropped; unknown liveness, a busy custody
    lock or a failed move keeps the note for the next pass."""
    from ouroboros.body_candidate import _recorded_terminal
    from ouroboros.owner_keep import gate
    from ouroboros.subagent_worktrees import _task_proven_over
    from ouroboros.task_custody import _TRASH_DIR
    from ouroboros.task_results import validate_task_id

    root = pathlib.Path(data_root)
    report: Dict[str, Any] = {"released": [], "kept": 0, "errors": []}
    try:
        folders = [] if (root / OPEN_SCRATCH_DIR).is_symlink() else sorted((root / OPEN_SCRATCH_DIR).iterdir())
    except (FileNotFoundError, NotADirectoryError):
        return report
    base = root.resolve()
    for folder in folders:
        if stop is not None and stop():
            break
        task_id = folder.name
        try:
            validate_task_id(task_id)
        except ValueError:
            continue  # no note of a task: nothing in it is read
        if not _task_proven_over(task_id, root, live):
            report["kept"] += 1
            continue
        with gate(root, task_id, timeout_sec=0) as held:
            if held or not _recorded_terminal(task_id, root):
                report["kept"] += 1
                continue
            for note in sorted(folder.glob("*.json")):
                try:
                    owner = pathlib.Path(str(json.loads(note.read_text(encoding="utf-8"))["owner"]))
                    if (owner.name != note.stem or owner.parent.name != SESSION_SCRATCH_DIRNAME
                            or owner.parent.parent.name != task_id or base not in owner.parents
                            or owner.resolve() != owner):
                        raise ValueError(f"names no scratch record of {task_id}: {owner}")
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    report["errors"].append(f"{note}: {type(exc).__name__}: {exc}")
                    with contextlib.suppress(OSError):
                        note.unlink()  # its record, if any, still retires with its drive
                    continue
                try:
                    if owner.is_dir() and not owner.is_symlink():
                        entry = root / _TRASH_DIR / f"{task_id}.scratch-{uuid.uuid4().hex[:12]}"
                        (entry / SESSION_SCRATCH_DIRNAME).mkdir(parents=True)
                        os.replace(owner, entry / SESSION_SCRATCH_DIRNAME / owner.name)
                        report["released"].append(str(owner))
                    note.unlink()
                except OSError as exc:
                    report["errors"].append(f"{owner}: {type(exc).__name__}: {exc}")
            with contextlib.suppress(OSError):
                folder.rmdir()  # empty once every note went
    return report


def session_service_directories(gateway: Any, route_id: str, *, access: str, mode: str, data_root: Any,
                                drive_root: Any, task_id: str, key: str, execution_root: Any) -> ServiceDirectories:
    """The directories one writable session may write besides its copy, made and named once (#1775).

    An Agent start gets a short physical scratch under ``default_temp_parent()`` (macOS /tmp),
    recorded under ``task_drives/<task_id>/session_scratch_owners/<key>`` in the starting task's own
    drive, when it is ``workspace_write`` on a route that negotiates auxiliary writable roots, or
    ``full``. Only the former is granted it, with the shared download cache's ``uv``/``pip``/``npm``
    folders: these exact directories, never the temp parent or ownership records. Full access
    reaches the scratch already, so no catalog is read and nothing is sent: the host only owns and
    names it. No install, no links, and no Git ancestor. The engine never creates, captures or
    deletes them; the instructions declare the scratch disposable before use
    (``DISPOSABLE_SCRATCH``); the record is noted under its task (``OPEN_SCRATCH_DIR``), the custody
    pass retires it once that task is over (``release_finished_scratch``), or its drive's settlement
    does, unless its producer retained it; the cache stays (PERSISTENCE). A route without that
    capability prepares nothing, and a folder that cannot be made grants nothing either: both are a
    typed ``limitation`` and the start goes on without the field. Read-only, and a
    ``workspace_write`` ``tool_policy`` route, keep their on-demand behavior: nothing is made. A
    retry replays its recorded request and never comes here.
    """
    if mode != "agent" or access not in ("workspace_write", "full"):
        return ServiceDirectories()
    granted = access == "workspace_write"
    row = _route_row(gateway, route_id) if granted else {}
    if row.get("writeMechanism") == "tool_policy":  # relevance read from confinement, never support
        return ServiceDirectories()
    if granted and not route_grants_auxiliary_roots(row):
        return ServiceDirectories(limitation="auxiliary_roots_unsupported: this route's engine negotiates no "
                                             "writable directory outside the copy")
    try:
        from ouroboros.task_results import validate_task_id

        execution = pathlib.Path(execution_root).resolve()
        task_id = validate_task_id(task_id)
        owner = _owned_directory(drive_root, ("task_drives", task_id, SESSION_SCRATCH_DIRNAME, key), execution)
        caches = [_owned_directory(data_root, (PACKAGE_DOWNLOADS_DIRNAME, name), execution)
                  for name, _variable in (_CACHE_VARIABLES if granted else ())]
        scratch = _session_scratch(owner, execution)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return ServiceDirectories(limitation=f"{'service_directories_unavailable' if granted else SCRATCH_UNAVAILABLE}: "
                                             f"{type(exc).__name__}: {exc}")
    _note_open_scratch(data_root, task_id, key, owner)
    return ServiceDirectories(roots=tuple(str(path) for path in (scratch, *caches)) if granted else (),
                              scratch=str(scratch))


def _dependency_files(root: pathlib.Path) -> List[Tuple[str, str]]:
    """``(relative path, sha256)`` of the copy's own manifests and locks, top level and one folder down."""
    from ouroboros.workspace_patch_rules import _ANY_SEGMENT_EXCLUDE_DIRS, _LOCKFILE_MANIFESTS

    names = sorted(set(_MANIFEST_NAMES) | set(_LOCKFILE_MANIFESTS) | set(_LOCKFILE_MANIFESTS.values()))
    try:
        folders = [root, *sorted(path for path in root.iterdir() if path.is_dir() and not path.is_symlink()
                                 and not path.name.startswith(".") and path.name not in _ANY_SEGMENT_EXCLUDE_DIRS)]
    except OSError:
        folders = [root]
    found = []
    for folder in folders:
        for name in names:
            path = folder / name
            if path.is_file() and not path.is_symlink():
                try:
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                except OSError as exc:
                    digest = f"unreadable ({type(exc).__name__})"
                found.append((path.relative_to(root).as_posix(), digest))
    return found


def _body_bootstrap_reference(body_root: pathlib.Path) -> str:
    """Where the candidate body documents its bootstrap, by physical path, lines and digest."""
    from ouroboros.reference_books import load_reference_book

    try:
        book = load_reference_book(body_root, "development")
    except (OSError, ValueError) as exc:
        return (f"the body's DEVELOPMENT book could not be read in this copy ({type(exc).__name__}: {exc}); "
                "report that gap rather than guess commands")
    for source in book.chapters or (book.entrypoint,):
        for heading in source.headings:
            if heading.title == _BODY_BOOTSTRAP_SECTION:
                return (f'the dependency-only setup this candidate documents, "{heading.title}" in '
                        f"{body_root / source.source_path} lines {heading.section.start_line}-"
                        f"{heading.section.end_line} (sha256 {source.sha256})")
    return (f'this copy\'s DEVELOPMENT book has no "{_BODY_BOOTSTRAP_SECTION}" section; report that gap '
            "rather than guess commands")


def _host_tools() -> str:
    """What THIS process's PATH resolves, through the preflight's own probe (node exec-probed)."""
    tools = _probe_tools(_SESSION_TOOLS)
    parts = []
    for name in _SESSION_TOOLS:
        entry = tools.get(name) or {}
        if entry.get("available"):
            parts.append(f"{name} {entry['path']}")
        elif entry.get("path"):
            reason = entry.get("probe_reason") or entry.get("reason") or entry.get("status") or "broken"
            parts.append(f"{name} {entry['path']} (not runnable: {reason})")
        else:
            parts.append(f"{name} not found")
    bundled = (tools.get("node") or {}).get("bundled_fallback")
    if bundled:
        parts.append(f"bundled node {bundled}")
    parts.append(f"host interpreter {sys.executable} (Python {platform.python_version()})")
    return ", ".join(parts)


def session_environment_facts(copy_root: Any, *, data_root: Any, body: bool = False,
                              environment_note: str = SNAPSHOT_ENVIRONMENT_NOTE, write_root: Any = None,
                              service: ServiceDirectories = ServiceDirectories()) -> str:
    """The PROJECT ENVIRONMENT facts of one writable session's private copy (#1775).

    Facts, never commands or an install (owner 3B): the copy's own dependency files
    by digest, whether an environment already exists there, the body's documented
    bootstrap and own verification launcher by reference (a project without one is a
    named gap), the shared download cache and what this host process's PATH resolves.
    The session decides whether a check needs an environment; its native access and
    PATH may differ. Executable environments stay private per copy; only downloads are
    shared. A session whose ``write_root`` is a folder below the copy's root still reads
    the root's dependency files, but its environment is disposable state outside every
    Git working tree. ``service`` is what the host made for it outside the copy: its
    disposable scratch, granted by its engine or owned at full access, is then the one such
    place named, and a typed limitation is said as one. A fact that cannot be gathered is a
    named gap, never a reason the session does not start.
    """
    try:
        return _environment_facts(pathlib.Path(copy_root), data_root, body, environment_note,
                                  pathlib.Path(write_root if write_root is not None else copy_root), service)
    except Exception as exc:  # e.g. an unreadable folder in the copy
        logging.getLogger(__name__).warning("Session environment facts unavailable", exc_info=True)
        granted = ("\n" + session_service_facts(service, data_root).rstrip("\n")) if service.scratch else ""
        return (f"\n\nPROJECT ENVIRONMENT facts unavailable ({type(exc).__name__}: {exc}); nothing was "
                "installed for this session: read the copy's own files and report that gap rather than guess."
                + granted)


def _environment_facts(root: pathlib.Path, data_root: Any, body: bool, environment_note: str,
                       start: pathlib.Path, service: ServiceDirectories) -> str:
    from ouroboros.delegate_start_instructions import EVIDENCE_DIRNAME, scratch_fact

    if start.resolve() != root.resolve():  # started in a folder of its copy, it may write only there
        environment_note = (f"outside every Git working tree, as disposable process state "
                            f"({scratch_fact(granted=service.scratch)}), "
                            f"since this session writes only in {start}, a folder of the copy")
    files = _dependency_files(root)
    shown = "; ".join(f"{path} {digest}" for path, digest in files[:_LISTED_DEPENDENCY_FILES])
    more = f"; +{len(files) - _LISTED_DEPENDENCY_FILES} more" if len(files) > _LISTED_DEPENDENCY_FILES else ""
    folders = sorted({str(pathlib.PurePosixPath(path).parent) for path, _digest in files} | {"."})
    existing = [str(root / folder / name) for folder in folders for name in (".venv", "node_modules")
                if (root / folder / name).is_dir()]
    state = (f"present: {', '.join(existing)} (made earlier in this copy, not verified against those files)"
             if existing else "none")
    recipe = (_body_bootstrap_reference(root) if body else
              "none known to the host for this project: use its own documentation, and report a gap "
              "rather than guess commands")
    launcher = (f"- Verification launcher: this copy's own {root / 'scripts' / 'safe_test.py'} "
                f"({scratch_fact(launcher=True, granted=service.scratch)}); its --evidence-dir for evidence the "
                f"host retains is {start / EVIDENCE_DIRNAME}.\n") if body else ""
    return (
        ENVIRONMENT_FACTS_HEADING +
        f"- Dependency files in {root} (sha256): {shown + more if files else 'none at its top level or one folder down'}.\n"
        f"- Environment in this copy: {state}. Build one only when a check needs it, from this copy's own "
        f"files and {environment_note}; never from or into another checkout's environment; then give the "
        "verification command its absolute interpreter.\n"
        f"- Bootstrap recipe: {recipe}.\n" + launcher + session_service_facts(service, data_root) +
        "- Tools on the host process's PATH at start (your session's PATH may differ; check before relying "
        f"on one): {_host_tools()}."
    )


def session_service_facts(service: ServiceDirectories, data_root: Any) -> str:
    """The lines naming what a writable session may write outside its copy: its engine's grant, the
    scratch the host owns at full access, or the typed limitation, each scratch declared disposable
    before use, then the shared download cache, by the very folders a grant permits."""
    from ouroboros.delegate_start_instructions import DISPOSABLE_SCRATCH, scratch_fact

    cache = package_downloads_root(data_root)
    caches = service.roots[1:] if service.roots else [str(cache / name) for name, _variable in _CACHE_VARIABLES]
    links = "" if sys.platform == "darwin" else "; with it set UV_LINK_MODE=copy so no installed file is shared"
    held = (f"the scratch {service.scratch} for disposable process state (test temporaries, package-manager "
            f"temporaries through TMPDIR, a folder session's environment), {DISPOSABLE_SCRATCH}.")
    grant = (f"- Writes your engine grants this run outside this copy (execution.auxiliaryWritableRoots): {held} "
             "The three download-cache folders are listed below.\n" if service.roots else
             "- Outside this copy the host made this run's own scratch; no engine grant is sent, since full "
             f"access already reaches it: {held}\n" if service.scratch else
             f"- Outside this copy: no disposable scratch could be made for this run ({service.limitation}); keep "
             f"disposable process state outside every Git working tree ({scratch_fact()}), where nothing owns "
             "or removes it, and report that gap.\n" if service.limitation.startswith(SCRATCH_UNAVAILABLE) else
             f"- Writes outside this copy: your engine grants this run none ({service.limitation}), so your "
             "harness may refuse the scratch and download cache named here; when a check needs one and it is "
             "refused, report that limitation rather than widen access or work around it.\n"
             if service.limitation else "")
    return (grant +
            "- Shared download cache (Ouroboros-owned, shared across sessions; the one durable place outside "
            "this copy your package manager may write, through it only; that manager locks it, and it is pruned "
            "only explicitly, with the package manager; never an environment or toolchain location; "
            + ("granted to this run; " if service.roots else "your full access reaches it; " if service.scratch
               else "your harness may not permit writing it; ") +
            f"{'present' if cache.is_dir() else 'created on first use'}): "
            + ", ".join(f"{variable}={path}" for (_name, variable), path in zip(_CACHE_VARIABLES, caches))
            + f"{links}.\n")


def git_head_basis(root: pathlib.Path) -> Tuple[str, str]:
    """A readable commit, positively unborn branch, or unknown; a failed probe is never unborn."""
    def probe(*args):
        return subprocess.run(["git", *args], cwd=str(root), capture_output=True,
                              text=True, timeout=5, check=False)

    try:
        head = probe("rev-parse", "--verify", "HEAD^{commit}")
        if head.returncode == 0 and head.stdout.strip():
            return head.stdout.strip(), "known"
        branch = probe("symbolic-ref", "--quiet", "HEAD")
        if branch.returncode or not branch.stdout.strip().startswith("refs/heads/"):
            return "", "unknown"
        missing = probe("show-ref", "--verify", "--quiet", branch.stdout.strip())
        history = probe("rev-parse", "--git-path", "logs/HEAD")
        if missing.returncode == 1 and history.returncode == 0 and history.stdout.strip():
            log_path = root / history.stdout.strip()
            try:
                size = log_path.stat().st_size
            except FileNotFoundError:
                size = 0
            if size == 0:
                return "", "unborn"
    except (OSError, subprocess.SubprocessError):
        pass
    return "", "unknown"


def _git_snapshot(root: pathlib.Path) -> Dict[str, Any]:
    head, head_state = git_head_basis(root)
    branch = _git_stdout(root, ["git", "rev-parse", "--abbrev-ref", "HEAD"], timeout=5).strip()
    status = _git_stdout(root, ["git", "status", "--porcelain=v1", "--untracked-files=all"], timeout=8)
    entries = [line for line in status.splitlines() if line.strip()]
    return {
        "head": head,
        "head_state": head_state,
        "branch": branch,
        "dirty": bool(entries),
        "status_count": len(entries),
        "status_porcelain": entries[:200],
        "status_truncated": len(entries) > 200,
    }


def _collect_manifests(root: pathlib.Path) -> List[Dict[str, Any]]:
    manifests: List[Dict[str, Any]] = []
    for name in _MANIFEST_NAMES:
        path = root / name
        if not path.is_file():
            continue
        entry: Dict[str, Any] = {"path": name, "type": _manifest_type(name), "scripts": []}
        if name == "package.json":
            scripts, script_commands = _package_json_scripts(path)
            entry["scripts"] = scripts
            entry["script_commands"] = script_commands
        elif name == "Makefile":
            entry["scripts"] = _makefile_targets(path)
        manifests.append(entry)
    return manifests


def _manifest_type(name: str) -> str:
    if name == "package.json":
        return "node"
    if name == "go.mod":
        return "go"
    if name in {"pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "tox.ini"}:
        return "python"
    if name == "Cargo.toml":
        return "rust"
    if name == "Makefile":
        return "make"
    return "manifest"


def _platform_curated_tools() -> List[str]:
    tools = list(_CURATED_TOOLS)
    if IS_MACOS:
        tools.extend(_MACOS_PACKAGE_TOOLS)
    elif IS_LINUX:
        tools.extend(_LINUX_PACKAGE_TOOLS)
    elif IS_WINDOWS:
        tools.extend(_WINDOWS_PACKAGE_TOOLS)
    return tools


def _package_json_scripts(path: pathlib.Path) -> Tuple[List[str], Dict[str, str]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return [], {}
    scripts = data.get("scripts") if isinstance(data, dict) else {}
    if not isinstance(scripts, dict):
        return [], {}
    names = sorted(str(key) for key in scripts.keys())[:50]
    commands = {str(key): str(value) for key, value in scripts.items() if isinstance(value, str)}
    return names, {key: commands[key] for key in names if key in commands}


def _makefile_targets(path: pathlib.Path) -> List[str]:
    out: List[str] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return out
    for line in lines:
        if line.startswith(("\t", " ", ".", "#")) or ":" not in line:
            continue
        target = line.split(":", 1)[0].strip()
        if target and all(ch not in target for ch in " $/\\"):
            out.append(target)
    return sorted(set(out))[:50]


def _infer_tools_from_manifests(manifests: Iterable[Dict[str, Any]]) -> List[str]:
    tools: set[str] = set()
    for entry in manifests:
        kind = str(entry.get("type") or "")
        if kind == "node":
            tools.update({"node", "npm", "npx"})
        elif kind == "go":
            tools.add("go")
        elif kind == "python":
            tools.update({"python", "python3", "pytest", "pip", "uv"})
        elif kind == "rust":
            tools.update({"cargo", "rustc"})
        elif kind == "make":
            tools.add("make")
        script_commands = entry.get("script_commands") if isinstance(entry.get("script_commands"), dict) else {}
        for script in script_commands.values():
            binary = _first_script_binary(str(script))
            if binary:
                tools.add(binary)
    return sorted(tools)


def _first_script_binary(script: str) -> str:
    try:
        parts = shlex.split(str(script))
    except ValueError:
        parts = str(script).split()
    skip_next = False
    for token in parts:
        if skip_next:
            skip_next = False
            continue
        text = str(token)
        if not text:
            continue
        if "=" in text and not text.startswith(("=", "-", "./", "../", "/")):
            continue
        name = pathlib.PurePath(text).name
        if name in {"cd", "set", "export", "source", ".", "&&", "||", ";"}:
            if name == "cd":
                skip_next = True
            continue
        if name.startswith("-"):
            continue
        return name
    return ""


def _probe_tools(names: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for name in names:
        clean = str(name or "").strip()
        if not clean:
            continue
        path = shutil.which(clean)
        out[clean] = {"available": bool(path), "path": path or ""}
    _apply_node_exec_probe(out)
    return out


def _apply_node_exec_probe(tools: Dict[str, Dict[str, Any]]) -> None:
    """Exec-probe node in place; a dead node also downgrades its dependents.

    ``shutil.which`` proves existence, not runnability. Statuses on the node
    entry: ``available`` / ``found_but_broken`` / ``missing``. When node is
    execution-probed broken OR absent entirely, npm/npx/pnpm/yarn that
    ``which`` found are marked unavailable with reason ``interpreter_broken``
    (no exec probes on them — their `#!/usr/bin/env node` launchers cannot run
    without node), and a healthy bundled node is disclosed as a fallback fact
    for the agent-facing summary.
    """
    node = tools.get("node")
    if not isinstance(node, dict):
        return
    if node.get("path"):
        health = node_runtime_health(str(node["path"]), timeout_sec=_NODE_PROBE_TIMEOUT_SEC)
        if health.healthy:
            node["status"] = "available"
            node["version"] = health.version
            return
        node["available"] = False
        # A which-hit that stat/access-fails mid-probe is missing, not broken.
        node["status"] = "found_but_broken" if health.status == "broken" else "missing"
        node["probe_reason"] = health.reason or health.status
    else:
        node["status"] = "missing"
    bundled = resolve_bundled_node()
    if bundled and node_runtime_health(bundled, timeout_sec=_NODE_PROBE_TIMEOUT_SEC).healthy:
        node["bundled_fallback"] = bundled
    # npm/npx/pnpm/yarn start through `#!/usr/bin/env node`: with node broken OR
    # absent they are equally dead, however present their own files are.
    for name in _NODE_DEPENDENT_TOOLS:
        entry = tools.get(name)
        if isinstance(entry, dict) and entry.get("path"):
            entry["available"] = False
            entry["status"] = "interpreter_broken"
            entry["reason"] = "interpreter_broken"


def _git_stdout(root: pathlib.Path, cmd: List[str], *, timeout: int) -> str:
    from ouroboros.git_observation import host_git_env
    try:
        result = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=host_git_env(),
        )
    except Exception:
        return ""
    return result.stdout if result.returncode == 0 else ""


__all__ = [
    "collect_workspace_preflight",
    "render_workspace_preflight_summary",
    "summarize_workspace_preflight",
]
