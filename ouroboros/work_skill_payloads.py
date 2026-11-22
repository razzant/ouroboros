"""Skill payloads a managed deployment's work domain runs and develops (``work_domain``).

Two flows put reviewed payload bytes the service keeps under its data root where the work domain can use them,
and bring results back as bytes the service checks; neither mounts, links or reopens a service path in the work:

- A native ``skill_payload`` run (``tools.delegate_integration``): the service builds the standalone baseline
  snapshot exactly as unmanaged (``subagent_worktrees.provision_payload_snapshot``), then the snapshot moves into
  ``<staging_root>/copies`` as the run's private copy (``materialize_payload_snapshot``) under the same snapshot
  record. Its capture exports the copy's tree into a private service folder (``capture_work_payload``) and runs
  the unchanged payload capture there; the baseline is trusted only by the content address the service recorded.
  Applying a result to the live payload stays the service's own apply, with its own refusals.
- An installed script skill (``tools.skill_exec``, ``skill_dependencies``): its reviewed inventory is copied into
  ``<staging_root>/skills/<name>/<content hash>`` only after the copy hashes to the reviewed content hash
  (``materialize_skill``), with each file's digest kept on the service; every later use verifies the work copy
  against those digests and replaces what changed. Its dependencies are built there, in the work domain
  (``marketplace.isolated_deps``), and the launch names that copy, its environment and the container's runtime.

A skills checkout already inside the mappings, and every unmanaged installation, keep their own folder as before.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import posixpath
import shutil
import stat
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from ouroboros import work_domain

# The interpreter names a work container resolves on its own PATH for a skill's runtime and its builds.
CONTAINER_PYTHON = "python3"
_CONTAINER_RUNTIMES = {"python": "python3", "python3": "python3", "node": "node", "deno": "deno", "bash": "bash",
                       "sh": "sh", "go": "go", "ruby": "ruby", "perl": "perl"}
_RECORD = "work_payload.json"


def _tree_members(source: pathlib.Path) -> List[Tuple[str, Any, int]]:
    """``import_tree`` members of a service folder: its folders, regular files (with their modes) and links as
    links; nothing else."""
    members: List[Tuple[str, Any, int]] = []
    for folder, dirs, files in os.walk(source, followlinks=False):
        for name in sorted(dirs + files):
            path = pathlib.Path(folder) / name
            rel, info = path.relative_to(source).as_posix(), path.lstat()
            if stat.S_ISLNK(info.st_mode):
                members.append((rel, work_domain.Link(os.readlink(path)), 0o777))
            elif stat.S_ISDIR(info.st_mode):
                members.append((rel, None, stat.S_IMODE(info.st_mode)))
            elif stat.S_ISREG(info.st_mode):
                members.append((rel, path, stat.S_IMODE(info.st_mode)))
    return members


# --- a native run's private copy of a skill payload ----------------------------------------------------------

def materialize_payload_snapshot(handle: Any, data_dir: Any = None) -> Any:
    """Move the service-built standalone snapshot ``handle`` into the work domain as the run's private copy:
    ``<staging_root>/copies/<the snapshot's own name>``, the whole tree (its ``.git`` and the committed baseline)
    unpacked there by the work domain; the snapshot's record then names that copy, and the service folder goes.
    Raises before any start, leaving neither copy nor record, when the move does not complete."""
    from ouroboros import subagent_worktrees as trees

    source = pathlib.Path(handle.path)
    target = pathlib.Path(work_domain.staging_path("copies", source.name))
    try:
        work_domain.import_tree(target, _tree_members(source))
        row = trees._record_at(source, data_dir, op="materialize_payload_snapshot")
        if row is None:
            raise RuntimeError(f"the payload snapshot {handle.snapshot_id} lost its record while it was moved")
        trees._put_record({**row, "path": str(target)}, data_dir)
        trees._drop_record(source, data_dir)
    except BaseException:
        work_domain.discard(target)
        trees._drop_record(target, data_dir)
        trees._drop_record(source, data_dir)
        trees._force_rmtree(source)
        raise
    trees._force_rmtree(source)
    return dataclasses.replace(handle, path=str(target))


def capture_work_payload(exec_root: pathlib.Path, cap_dir: pathlib.Path, entry: Any, capture: Any) -> Dict[str, Any]:
    """A managed payload run's capture: its copy's tree — every regular file, a link whose text stays inside the copy
    placed as that link, its ``.git`` included — arrives in one checked export (``work_domain.export_tree``) into a
    private service folder, where the unchanged payload capture (``capture``) reads only the service's own bytes.
    A link leaving the copy and any special file stay out, as the loader inventory leaves them out (a baseline
    path replaced by one rides as a deletion); a folder the service cannot list fails the capture."""
    rels = []
    for folder, listed in work_domain.walk_entries(exec_root, strict=True):
        for name, kind in listed:
            rel = (pathlib.Path(folder) / name).relative_to(exec_root).as_posix()
            if kind == "file" or (kind == "link" and work_domain._confined_link(rel, work_domain.readlink(f"{folder}/{name}"))):
                rels.append(rel)
    private = pathlib.Path(tempfile.mkdtemp(prefix="ouroboros-payload-capture-")).resolve()
    try:
        work_domain.export_tree(exec_root, rels, private / "copy", links=True)
        return capture(private / "copy", cap_dir, entry)
    finally:
        shutil.rmtree(private, ignore_errors=True)


# --- an installed script skill in the work domain ------------------------------------------------------------

def _home(name: str, content_hash: str) -> pathlib.Path:
    from ouroboros.contracts.skill_manifest import canonical_skill_name

    if not content_hash:
        raise work_domain.WorkDomainRefused(f"skill {name!r} has no reviewed content hash to copy; nothing was started")
    return pathlib.Path(work_domain.staging_path("skills", canonical_skill_name(name), content_hash))


def _record_path(drive_root: Any, name: str) -> pathlib.Path:
    from ouroboros.skill_loader import skill_state_dir

    return skill_state_dir(pathlib.Path(drive_root), name) / _RECORD


# What a dependency build adds beside a work copy's payload (``isolated_deps``): never part of the payload.
_BUILT = (".ouroboros_env", "node_modules")


def _expected(members: List[Tuple[str, Any, int]]) -> Dict[str, str]:
    return {rel: (f"link:{data}" if isinstance(data, work_domain.Link) else _digest(data))
            for rel, data, _mode in members if data is not None}


def _verified(home: pathlib.Path, expected: Dict[str, str]) -> bool:
    """Whether the work copy holds exactly the recorded payload: each file's digest, each link's text, nothing more
    (its dependency build left out), read by held descriptors."""
    found: Dict[str, str] = {}
    try:
        for folder, listed, dirs in work_domain._descend(home, True, None):
            if folder == str(home):
                dirs[:] = [name for name in dirs if name not in _BUILT]
            for name, kind in listed:
                rel = pathlib.Path(folder, name).relative_to(home).as_posix()
                if folder == str(home) and name in _BUILT:
                    continue
                if kind == "file":
                    found[rel] = work_domain.measure(home / rel, strict=True)["sha256"]
                elif kind == "link":
                    found[rel] = f"link:{work_domain.readlink(home / rel)}"
                elif kind != "dir":
                    return False
    except OSError:
        return False
    return bool(expected) and found == expected


def materialize_skill(drive_root: Any, loaded: Any, content_hash: str) -> pathlib.Path:
    """The work copy of an installed skill's reviewed payload, ``<staging_root>/skills/<name>/<content_hash>``.

    The loader inventory is copied into a private service folder and must hash to ``content_hash`` (the reviewed
    bytes) before one tar stream unpacks it there as the work domain; each file's digest and each link's text are
    kept in the skill's service state. A copy that still holds exactly those is used as it is; one that changed
    (the work domain may write it) loses its payload entries and is unpacked again, its built dependencies
    (``.ouroboros_env``) left in place."""
    from ouroboros.skill_loader import compute_content_hash
    from ouroboros.subagent_worktrees import _copy_payload_inventory
    from ouroboros.utils import atomic_write_json, read_json_dict

    home, record = _home(loaded.name, content_hash), _record_path(drive_root, loaded.name)
    known = read_json_dict(record) or {}
    files = known.get("files") if known.get("content_hash") == content_hash and known.get("path") == str(home) else None
    if isinstance(files, dict) and _verified(home, files):
        return home
    if work_domain.is_dir(home):  # a changed copy: its payload entries go, its dependency build stays
        for name, _kind in work_domain.entries(home):
            if name not in _BUILT:
                work_domain.remove_tree(home / name)
    private = pathlib.Path(tempfile.mkdtemp(prefix="ouroboros-skill-copy-")).resolve()
    try:
        snapshot = private / "payload"
        _copy_payload_inventory(pathlib.Path(loaded.skill_dir), snapshot, manifest_entry=loaded.manifest.entry,
                                manifest_scripts=loaded.manifest.scripts)
        copied = compute_content_hash(snapshot, manifest_entry=loaded.manifest.entry,
                                      manifest_scripts=loaded.manifest.scripts)
        if copied != content_hash:
            raise work_domain.WorkDomainRefused(f"skill {loaded.name!r} changed while it was copied for the work "
                                                "domain; re-run skill_review. Nothing was started")
        members = _tree_members(snapshot)
        work_domain.import_tree(home, members)
        files = _expected(members)
    finally:
        shutil.rmtree(private, ignore_errors=True)
    record.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(record, {"schema_version": 1, "content_hash": content_hash, "path": str(home), "files": files},
                      trailing_newline=True)
    if not _verified(home, files):
        raise OSError(f"the work copy of skill {loaded.name!r} did not verify after it was unpacked")
    return home


def _digest(path: pathlib.Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deps_home(drive_root: Any, loaded: Any, *, materialize: bool = False) -> pathlib.Path:
    """Where an installed skill's dependencies live and are built: its own folder unmanaged, or for a skills checkout
    inside the mappings; managed, the work copy of its reviewed payload, made first when ``materialize``. An
    extension keeps its own folder: only a trusted one runs, in the service, with the packages its administrator
    delivered there (``deployment_extensions.dependency_refusal``); nothing of it is copied to the work."""
    source = pathlib.Path(loaded.skill_dir)
    if not work_domain.managed() or work_domain.is_work(source) or _is_extension(loaded):
        return source
    content_hash = str(getattr(loaded, "content_hash", "") or "")
    return materialize_skill(drive_root, loaded, content_hash) if materialize else _home(loaded.name, content_hash)


def _is_extension(loaded: Any) -> bool:
    manifest = getattr(loaded, "manifest", None)
    return bool(manifest is not None and callable(getattr(manifest, "is_extension", None)) and manifest.is_extension())


def launch(drive_root: Any, loaded: Any, content_hash: str, cmd: List[str], env: Dict[str, str],
           runtime: str) -> Tuple[List[str], str, Dict[str, str]]:
    """``(cmd, cwd, env)`` for one run of an installed script skill. Unmanaged, or for a checkout inside the mappings,
    exactly as the caller built them. Managed, the reviewed payload's verified work copy (``materialize_skill``):
    every argument naming the installed payload names that copy, the interpreter is the copy's own environment's or
    the container's (``_CONTAINER_RUNTIMES``), and the dependency variables name the copy's environment. A refusal
    before the start is the tool's ``PermissionError``, as any managed path refusal is."""
    from ouroboros.marketplace.isolated_deps import augment_env_for_skill_deps, python_runtime_binary

    source = pathlib.Path(loaded.skill_dir)
    if not work_domain.managed() or work_domain.is_work(source):
        return cmd, str(source), env
    try:
        home = materialize_skill(drive_root, loaded, content_hash)
    except work_domain.WorkDomainRefused as refused:
        raise PermissionError(str(refused)) from refused
    installed = source.resolve()

    def named_here(word: str) -> str:
        try:
            return str(home / pathlib.Path(word).resolve(strict=False).relative_to(installed))
        except (ValueError, OSError):
            return word

    words = [named_here(str(word)) for word in cmd]
    if words and not words[0].startswith(f"{home}/"):  # a service interpreter: the container names its own
        python = python_runtime_binary(home) if runtime in {"python", "python3"} else None
        words[0] = str(python) if python else _CONTAINER_RUNTIMES.get(runtime, posixpath.basename(words[0]))
    kept = {}
    for key, value in env.items():  # the installed copy's dependency variables are the service's, never sent
        parts = [part for part in str(value).split(os.pathsep) if not _under(part, installed)]
        if parts or not value:
            kept[key] = os.pathsep.join(parts) if key == "PATH" else value
    return words, str(home), augment_env_for_skill_deps(kept, home)


def _under(part: str, installed: pathlib.Path) -> bool:
    try:
        return bool(part) and os.path.isabs(part) and pathlib.Path(part).resolve(strict=False).is_relative_to(installed)
    except (OSError, ValueError):
        return False


def recorded_copy(drive_root: Any, name: str) -> Optional[Dict[str, Any]]:
    """The service's record of a skill's work copy (``materialize_skill``), or None."""
    from ouroboros.utils import read_json_dict

    row = read_json_dict(_record_path(drive_root, name))
    return row if isinstance(row, dict) and row.get("path") else None


def _json_read(path: Any) -> Optional[Dict[str, Any]]:
    """``read_json_dict`` for a service path; a work file read through the confined intake."""
    from ouroboros.utils import read_json_dict

    if not work_domain.is_work(path):
        return read_json_dict(path)
    try:
        value = json.loads(work_domain.read_text(path))
    except (OSError, ValueError, UnicodeError):
        return None
    return value if isinstance(value, dict) else None
