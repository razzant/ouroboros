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
  against its digests and unpacked execute bits; mode-only refresh replaces leaves without moving their parents.
  Dependencies are built in the work domain (``marketplace.isolated_deps``); launch names the copy, its environment
  and the container's runtime.

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
import subprocess
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


def _admission(record: pathlib.Path, home: pathlib.Path,
               content_hash: str) -> Tuple[Dict[str, Any], Optional[Dict[str, str]]]:
    """``(the record, its digest map for exactly this copy and hash)``; the map is None whenever the record
    does not name this copy as it stands, or names a member anything but a digest or a link text (a
    hand-edited record never admits what it cannot name)."""
    from ouroboros.utils import read_json_dict

    known = read_json_dict(record) or {}
    files = known.get("files") if known.get("content_hash") == content_hash and known.get("path") == str(home) else None
    if not (isinstance(files, dict) and all(isinstance(value, str) and value for value in files.values())):
        files = None
    return known, files


def _admission_state(known: Dict[str, Any], files: Dict[str, str], modes: Dict[str, int],
                     home: pathlib.Path) -> Tuple[bool, Optional[Dict[str, int]]]:
    """``(admitted, legacy actual modes)``: whether the record admits the copy at ``home`` exactly as it
    stands — one authority for the fast path and the retained-stage bypass. A modern record admits on its
    recorded unpacked bits; a legacy one only on an exact installed-bit match, and then names its actual."""
    unpacked = known.get("unpacked_exec")
    if known.get("installed_exec") == modes and isinstance(unpacked, dict):
        return _verified_modes(home, files, modes) == unpacked, None
    if "installed_exec" not in known and "unpacked_exec" not in known:
        actual = _verified_modes(home, files, modes)
        return actual == modes, actual
    return False, None


# What a dependency build adds beside a work copy's payload (``isolated_deps``): never part of the payload.
_BUILT = (".ouroboros_env", "node_modules")


def _expected(members: List[Tuple[str, Any, int]]) -> Dict[str, str]:
    return {rel: (f"link:{data}" if isinstance(data, work_domain.Link) else _digest(data))
            for rel, data, _mode in members if data is not None}


def _modes(loaded: Any) -> Dict[str, int]:
    """Installed execute bits. A same-byte replace or rollback can change them without changing the digest."""
    from ouroboros.skill_loader import _iter_payload_files

    root = pathlib.Path(loaded.skill_dir).resolve()
    modes: Dict[str, int] = {}
    for path in _iter_payload_files(root, manifest_entry=loaded.manifest.entry,
                                     manifest_scripts=loaded.manifest.scripts):
        info = os.lstat(path)
        if stat.S_ISREG(info.st_mode) and (bits := info.st_mode & 0o111):
            modes[path.relative_to(root).as_posix()] = bits
    return modes


def _inventory(home: pathlib.Path) -> Tuple[Dict[str, Optional[str]], Dict[str, int]]:
    """The copy's own inventory — each file's digest, each link's text, each file's execute bits (``0`` kept) —
    read by held descriptors, one confined read per regular leaf: its digest and its execute bits come from the
    same opened object (``measure``'s stamp), never from a second look the leaf could have missed. A special file
    is read as one (a ``None`` digest: the tree was read; the entry is not a payload member); a read that fails
    raises its ``OSError`` and proves nothing about the bytes."""
    found: Dict[str, Optional[str]] = {}
    exec_bits: Dict[str, int] = {}
    for folder, listed, dirs in work_domain._descend(home, True, None):
        if folder == str(home):
            dirs[:] = [name for name in dirs if name not in _BUILT]
        for name, kind in listed:
            rel = pathlib.Path(folder, name).relative_to(home).as_posix()
            if folder == str(home) and name in _BUILT:
                continue
            if kind == "file":
                measured = work_domain.measure(home / rel, strict=True, stamp=True)
                exec_bits[rel] = measured["mode"] & 0o111
                found[rel] = measured["sha256"]
            elif kind == "link":
                found[rel] = f"link:{work_domain.readlink(home / rel)}"
            elif kind != "dir":
                found[rel] = None
    return found, exec_bits


def _verified_modes(home: pathlib.Path, expected: Dict[str, str],
                    installed: Optional[Dict[str, int]]) -> Optional[Dict[str, int]]:
    """Read execute bits only on a matching inventory. With installed modes, allow umask to narrow group/other
    access, but never add execute rights or change the owner's executable bit. None as input checks bytes and
    links without disqualifying the previous modes during a refresh."""
    try:
        found, exec_bits = _inventory(home)
    except OSError:
        return None
    if installed is not None:
        for rel, bits in exec_bits.items():
            source = installed.get(rel, 0)
            if bits & ~source or (bits ^ source) & stat.S_IXUSR:
                return None
    return {rel: bits for rel, bits in exec_bits.items() if bits} if expected and found == expected else None


class _RefreshUncertain(OSError):
    """The observed paths do not prove the move or its reversal; keep the staging tree."""


class _UnreadableCopy(OSError):
    """The copy could not be read to compare: its bytes are unknown — not a proven change, and no live entry
    moves on them."""


def _entry_identity(path: pathlib.Path, *, regular: bool = False) -> Optional[Tuple[int, int]]:
    try:
        info = work_domain.lstat(path)
    except FileNotFoundError:
        return None
    if regular and not stat.S_ISREG(info.st_mode):
        raise OSError(f"expected a regular work-copy leaf at {path}")
    return info.st_dev, info.st_ino


def _move_payload_entry(source, target, undo=None, *, expected=None, replacing=None, backup=None):
    """A same-filesystem move to an unused name or over a measured regular leaf, with readback for rollback."""
    leaf = replacing is not None
    before = _entry_identity(source, regular=leaf)
    if (before is None or (expected is not None and before != expected)
            or _entry_identity(target, regular=leaf) != replacing
            or work_domain.lstat(target.parent).st_dev != before[0]
            or (leaf and (source.name != target.name or replacing[0] != before[0]))):
        raise OSError(f"cannot move {source} to the expected entry {target}; no move attempted")
    if backup is not None and _entry_identity(backup) != replacing:
        raise _RefreshUncertain(f"cannot confirm the previous leaf at {backup}")
    failure = None
    try:
        # With equal basenames, giving mv the existing parent requests exactly parent/basename.
        # Giving it the leaf itself could instead nest the source if the leaf had become a directory.
        work_domain.rename(source, target.parent if leaf else target)
    except (work_domain.WorkDomainRefused, work_domain.WorkDomainUnsettled):
        raise
    except (OSError, subprocess.TimeoutExpired) as exc:
        failure = exc  # a settled timeout or nonzero exit is not proof the rename did not land
    try:
        observed = _entry_identity(source), _entry_identity(target)
    except OSError as exc:
        raise _RefreshUncertain(f"cannot read back move from {source} to {target}: {exc}") from exc
    if observed == (None, before):
        if undo is not None:
            undo.append((target, source, before, None) if backup is None else (backup, target, replacing, before))
    elif observed != (before, replacing):
        raise _RefreshUncertain(f"cannot establish the outcome of moving {source} to {target}") from failure
    elif failure is None:
        raise OSError(f"move from {source} to {target} did not land")
    if failure is not None:
        raise failure


def _replace_payload_leaf(source, target, backup, undo):
    """Keep the old inode without taking its live name away; restore it over the new leaf if needed."""
    old = _entry_identity(target, regular=True)
    if old is None:
        raise OSError(f"work-copy leaf {target} disappeared before refresh")
    work_domain.make_dirs(backup.parent)
    work_domain._effect("retain", str(target), ["ln", "--", work_domain.container_path(target),
                                               work_domain.container_path(backup.parent)])
    _move_payload_entry(source, target, undo, replacing=old, backup=backup)


def _refresh_copy(home, hold, loaded, members, files, modes, save):
    """Publish only a verified stage. Preserve live paths on a mode-only change; keep builds even on byte repair.
    A copy that cannot be read is not a proven change: no live entry moves on unreadable bytes."""
    work_domain.make_dirs(home.parent)
    work_domain.make_dir(hold)  # exclusive name; a failed/unknown creation never authorizes cleanup here
    candidate, previous = hold / "new", hold / "previous"
    undo = []
    discard = False
    try:
        work_domain.import_tree(candidate, members)
        unpacked = _verified_modes(candidate, files, modes)
        if unpacked is None:
            raise OSError(f"the staged work copy of skill {loaded.name!r} did not verify after it was unpacked")
        if _modes(loaded) != modes:
            raise OSError(f"installed execute bits for skill {loaded.name!r} changed during work-copy preparation")
        actual: Optional[Dict[str, int]] = None
        if _entry_identity(home) is None:
            _move_payload_entry(candidate, home, undo)
        else:
            try:
                actual_found, actual_bits = _inventory(home)
            except OSError as exc:
                raise _UnreadableCopy(f"the work copy of skill {loaded.name!r} at {home} could not be verified "
                                      f"against its record ({exc}); it is not proven changed, so nothing was "
                                      "moved or replaced; make it readable, or inspect it after the work domain "
                                      "settles") from exc
            actual = ({rel: bits for rel, bits in actual_bits.items() if bits}
                      if files and actual_found == files and None not in actual_found.values() else None)
            if actual is None:
                old = {name for name, _kind in work_domain.entries(home) if name not in _BUILT}
                new = {name for name, _kind in work_domain.entries(candidate)}
                work_domain.make_dir(previous)
                for name in sorted(old | new):
                    if name in old:
                        _move_payload_entry(home / name, previous / name, undo)
                    if name in new:
                        _move_payload_entry(candidate / name, home / name, undo)
            elif actual != unpacked:
                for rel in sorted(actual.keys() | unpacked.keys()):
                    if actual.get(rel, 0) != unpacked.get(rel, 0):
                        _replace_payload_leaf(candidate / rel, home / rel, previous / rel, undo)
        if actual != unpacked and _verified_modes(home, files, modes) != unpacked:
            raise OSError(f"the replacement work copy of skill {loaded.name!r} did not verify at {home}")
        save(files, unpacked)
    except (work_domain.WorkDomainUnsettled, _RefreshUncertain):
        raise  # no rollback, deletion or retry of an ambiguous operation
    except (OSError, ValueError, subprocess.TimeoutExpired) as failure:
        try:
            for source, target, identity, replacing in reversed(undo):
                _move_payload_entry(source, target, expected=identity, replacing=replacing)
        except work_domain.WorkDomainUnsettled:
            raise
        except (OSError, ValueError, subprocess.TimeoutExpired) as rollback:
            raise _RefreshUncertain(f"work-copy refresh failed ({failure}); rollback not confirmed ({rollback}); "
                                    f"retained {hold}") from rollback
        discard = True
        raise
    else:
        discard = True
    finally:
        if discard:
            work_domain.discard(hold)


def _materialize_skill(home, hold, record, loaded, content_hash):
    from ouroboros.skill_identity import payload_bucket, payload_digest
    from ouroboros.subagent_worktrees import _copy_payload_inventory
    from ouroboros.utils import atomic_write_json

    modes = _modes(loaded)

    def save(files, unpacked):
        atomic_write_json(record, {"schema_version": 1, "content_hash": content_hash, "path": str(home), "files": files,
                                   "installed_exec": modes, "unpacked_exec": unpacked}, trailing_newline=True)

    known, files = _admission(record, home, content_hash)
    if files is not None:
        admitted, actual = _admission_state(known, files, modes, home)
        if admitted:
            if actual is not None:  # an exact legacy match: no need to infer a historical umask
                save(files, actual)
            return

    with tempfile.TemporaryDirectory(prefix="ouroboros-skill-copy-") as private:
        snapshot, installed = pathlib.Path(private).resolve() / "payload", pathlib.Path(loaded.skill_dir)
        entry, scripts = loaded.manifest.entry, loaded.manifest.scripts
        _copy_payload_inventory(installed, snapshot, manifest_entry=entry, manifest_scripts=scripts)
        if payload_digest(snapshot, manifest_entry=entry, manifest_scripts=scripts,
                          bucket=payload_bucket(installed)) != content_hash:
            raise work_domain.WorkDomainRefused(f"skill {loaded.name!r} changed while it was copied for the work "
                                                "domain; re-run skill_review. Nothing was started")
        members = _tree_members(snapshot)
        if any(rel.split("/", 1)[0] in _BUILT for rel, _data, _mode in members):
            raise work_domain.WorkDomainRefused(f"skill {loaded.name!r} uses a reserved work-copy dependency path")
        snapshot_modes = {rel: mode & 0o111 for rel, data, mode in members
                          if data is not None and not isinstance(data, work_domain.Link) and mode & 0o111}
        if snapshot_modes != modes:
            raise OSError(f"installed execute bits for skill {loaded.name!r} changed while it was copied")
        _refresh_copy(home, hold, loaded, members, _expected(members), modes, save)


def _stage_without_moves(hold: pathlib.Path) -> bool:
    """Whether a retained refresh stage is exactly its own staged candidate — one ``new`` entry, nothing
    else. Every move out of the live copy creates the ``previous`` stage first, so such a stage never moved
    a live entry, and an unsettled import inside it reaches nothing a materialization reads or writes. An
    empty stage (a fresh move's rollback can empty it and still land), an unlistable one and any other
    shape prove nothing."""
    try:
        return [name for name, _kind in work_domain.entries(hold)] == ["new"]
    except OSError:
        return False


def _admitted_as_is(record: pathlib.Path, home: pathlib.Path, loaded: Any, content_hash: str) -> bool:
    """Whether the record still admits the work copy exactly as it stands — a read-only question, so a
    retained stage that provably never touched the copy does not block the runs its record vouches for. A
    comparison that fails decides nothing (False): never a repair, never a run on unproven bytes; the
    legacy record's own upgrade stays with a normal materialize."""
    known, files = _admission(record, home, content_hash)
    if files is None:
        return False
    admitted, _actual = _admission_state(known, files, _modes(loaded), home)
    return admitted


def materialize_skill(drive_root: Any, loaded: Any, content_hash: str) -> pathlib.Path:
    """Verify or refresh a reviewed script skill at its fixed work path.

    Cooperating service materializers share one lock per skill, outside deletable skill state; contention is
    a bounded wait and then a refusal that says what to do, never an unbounded block. Stage beside the home
    so moves need no cross-filesystem copy. An interrupted or indeterminate refresh keeps that named stage
    and blocks a later refresh until its custody is resolved; never recover by deleting the old payload. The
    runs its record still admits as-is proceed past a stage that provably never touched the live copy, and
    the stage stays for inspection.
    """
    from ouroboros.execution_domain import _data_root
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock

    home, record = _home(loaded.name, content_hash), _record_path(drive_root, loaded.name)
    hold = home.with_name(f".{home.name}.refresh")
    lock = _data_root() / "state" / "skills" / f".{home.parent.name}-work-payload.lock"
    held: Dict[str, Any] = {}
    fd = acquire_exclusive_file_lock(lock, owner_aware_stale=True, outcome=held)
    if fd is None:
        reason = str(held.get("reason") or "unknown")
        advice = ("a concurrent preparation may hold it — retry after it finishes" if reason == "contention"
                  else "retrying now will not help; inspect the lock")
        raise OSError(f"work-copy lock unavailable for skill {loaded.name!r} ({reason}); {advice}; "
                      "nothing was changed")
    try:
        if _entry_identity(hold) is not None:
            if _stage_without_moves(hold) and _admitted_as_is(record, home, loaded, content_hash):
                return home
            raise OSError(f"retained work-copy refresh at {hold}; it may hold entries moved out of the live "
                          "copy, or a work-domain operation that has not settled; inspect it after any "
                          "executor operation settles; nothing was changed")
        _materialize_skill(home, hold, record, loaded, content_hash)
    except work_domain.WorkDomainUnsettled as exc:
        raise work_domain.WorkDomainUnsettled(f"{exc}; work-copy refresh not confirmed; inspect {hold} before retrying",
                                              exc.result) from exc
    except subprocess.TimeoutExpired as exc:
        raise OSError(f"timed out preparing the work copy at {home}: {exc}") from exc
    finally:
        release_exclusive_file_lock(lock, fd)
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
        try:
            python = python_runtime_binary(home, conclusive_absence=True) if runtime in {"python", "python3"} else None
        except (OSError, work_domain.WorkDomainRefused, subprocess.TimeoutExpired) as refused:
            raise PermissionError(f"the managed skill's own Python is unobserved: {refused}") from refused
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


def _json_read(path: Any, *, conclusive_absence: bool = False) -> Optional[Dict[str, Any]]:
    """Service JSON or confined work JSON. Strict work reads return empty only on confirmed absence."""
    from ouroboros.utils import read_json_dict

    if not work_domain.is_work(path):
        return read_json_dict(path)
    try:
        value = json.loads(work_domain.read_text(path))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        if conclusive_absence:
            raise
        return None
    if not isinstance(value, dict) and conclusive_absence:
        raise ValueError(f"{path} is present and holds no dependency record")
    return value if isinstance(value, dict) else None
