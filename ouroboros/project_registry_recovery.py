"""Exact local recovery of the Project registry at the writer's existing lock.

projects.json is the commit. Its .committed sibling starts as an existence witness
and can hold the last qualified bytes. Its presence alone never establishes freshness.
The released projects.json.lock carries the certificate. On POSIX an atomic name
exchange takes that exact file out of the legacy writer's way and puts a
directory in its place: even a pending name-only unlink cannot remove the live
writer's directory. Current writers serialize these exchanges with a directory
flock. A quiescent body switch returns an interrupted exchange to its
released file name. On Windows a stable file lock serializes current writers and an open
handle pins the legacy name against deletion. Without the
qualified primitive ordinary writes use the platform lock and keep only an
existence witness. The source stamp also detects an in-place downgrade to a
pre-lock body and back. Independent writers bypassing the protocol and
whole-volume rollback are outside it.
First use may coexist with independent workspace shelves. A recorded owner
Reset starts from an empty registry. Only reconciliation restores a lost one.
"""
from __future__ import annotations

import contextlib
import contextvars
import errno
import hashlib
import io
import json
import logging
import os
import stat
import time
from pathlib import Path
from typing import Any

from ouroboros import platform_layer as platform, utils
from ouroboros.contracts.chat_id_policy import project_chat_id
from ouroboros.install_identity import recorded_id
from ouroboros.platform_file_rename import rename_exclusive
from ouroboros.installation_stop import confirm_absent
from ouroboros.project_facts import can_be_workspace_shelf, sanitize_project_id
from ouroboros.temp_sweep import atomic_temp_path

log = logging.getLogger(__name__)
_MAGIC = b"ouroboros-project-registry-v1\n"
_NO_LINK = {errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EPERM}
_NO_DIRECTORY_LOCK = _NO_LINK | {errno.ENOLCK, errno.EBADF, errno.EACCES, errno.EISDIR, errno.EINVAL}
_EXCHANGE_PARTNER = ".projects.registry-writer"
_EXCHANGE_MARKER = ".project-registry-writer"
_HELD = contextvars.ContextVar("project_registry_writer", default=None)
SCHEMA_VERSION = 2


PROJECT_ACTIVE = "active"
PROJECT_DELETING = "deleting"
PROJECT_TOMBSTONED = "tombstoned"
PROJECT_LIFECYCLES = frozenset({PROJECT_ACTIVE, PROJECT_DELETING, PROJECT_TOMBSTONED})


def read_document(source) -> dict:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate registry member: {key}")
            result[key] = value
        return result

    data = json.load(source, object_pairs_hook=unique_object)
    if not isinstance(data, dict) or not isinstance(data.get("projects"), list):
        raise ValueError("Project registry must contain a projects array")
    return data


def read_bytes(raw: bytes) -> dict:
    return read_document(io.StringIO(raw.decode("utf-8")))


def routing_row(raw: Any, *, identity_only: bool = False) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("Project registry contains a non-object row")
    pid = raw.get("id")
    if not isinstance(pid, str) or not pid or sanitize_project_id(pid) != pid:
        raise ValueError("Project registry contains an invalid id")
    row = {"lifecycle": PROJECT_ACTIVE, "routing_generation": 0,
           "chat_id": project_chat_id(pid), "working_dir": "", **raw}
    if type(row["chat_id"]) is not int or row["chat_id"] <= 0 or not identity_only and (
            not isinstance(row["lifecycle"], str) or row["lifecycle"] not in PROJECT_LIFECYCLES
            or type(row["routing_generation"]) is not int or row["routing_generation"] < 0
            or not isinstance(row["working_dir"], str)
            or any(key in row and (not isinstance(row[key], str) or not row[key])
                   for key in ("routing_incarnation", "created_at"))):
        raise ValueError(f"Project {pid!r} has malformed routing authority")
    return row


def validated_registry(data: dict, *, identity_only: bool = False, preserve_raw: bool = False) -> dict:
    version = data.get("_schema_version", 0)
    if type(version) is not int or not 0 <= version <= SCHEMA_VERSION:
        raise ValueError("Unknown Project registry schema")
    rows, seen, chats = [], set(), set()
    for raw in data["projects"]:
        row = routing_row(raw, identity_only=identity_only)
        if row["id"] in seen or row["chat_id"] in chats:
            raise ValueError("Project registry contains a duplicate id or chat reservation")
        seen.add(row["id"])
        chats.add(row["chat_id"])
        rows.append(raw if preserve_raw else row)
    return {**data, "projects": rows}


def registry_path(root) -> Path:
    return Path(root) / "state" / "projects.json"


def witness_path(root) -> Path:
    return Path(root) / "state" / "projects.json.committed"


def history_generation(root) -> str:
    """On loss, distinguish the existing profile from a pending Reset that must not read its predecessor."""
    from ouroboros.launch_roots import RESETTING, read_witness

    status, _ = read_witness(root)
    if status == RESETTING:
        return "reset_pending"
    if status not in {"missing", "ok"}:
        return "unknown"
    return "current"


def could_be_first_registry(root, *, initializing=False) -> bool:
    """Only consulted on ENOENT. Preserve positive Project history, including folderless rooms."""
    path = registry_path(root)
    try:
        path.lstat()
    except FileNotFoundError:
        confirm_absent(path)
    else:
        return False
    if _registry_evidence(root):
        return False
    # Reset's cleanup (``state_initialization.accept_runtime_reset``) publishes its own
    # empty registry and records ``reset_receipt`` before any witness reads ``ok``: losing
    # that registry is a loss, never another first use. An older Reset (``mark_pending``,
    # then ``complete``) wrote no such field and no registry; only the field's absence
    # leaves its profile to the ordinary evidence below. A field present but unreadable is unknown.
    from ouroboros.launch_roots import read_witness
    status, initialization = read_witness(root)
    if status not in {"missing", "ok"} or (
            initialization.get("origin") == "owner_reset" and "reset_receipt" in initialization):
        return False
    from ouroboros.dialogue_history_evidence import ProjectHistoryImportPending, could_lack_project_history

    try:
        return could_lack_project_history(root, initializing=initializing)
    except ProjectHistoryImportPending:
        if initializing:
            return False  # the boot owner retries first use after the dialogue import
        raise


def _registry_evidence(root) -> list:
    """The durable traces, beside an absent registry, that this root's profile had Projects: the
    registry's own committed witness or restore receipt, its task bindings, a writer lock a
    commit still shapes, and any shelf a marker or an older store name still claims. One walk
    decides first use (``could_be_first_registry``) and what the Reset's prior-profile reading may
    treat as no-Projects (``prior_working_folders``): the proofs are these, never a second copy."""
    path = registry_path(root)
    found = []
    for candidate in (witness_path(root), _restore_receipt(path),
                      path.with_name("project_task_bindings.json")):
        try:
            candidate.lstat()
        except FileNotFoundError:
            confirm_absent(candidate)
            continue
        found.append(candidate.name)
    held = _HELD.get()
    if held is not None and held.path == path:
        if held.certificate:
            found.append(_lock_path(path).name)
    else:
        lock = _lock_path(path)
        try:
            if not stat.S_ISREG(lock.lstat().st_mode):
                found.append(lock.name)
            else:
                with lock.open("rb") as source:
                    if source.read(len(_MAGIC)) == _MAGIC:
                        found.append(lock.name)
        except FileNotFoundError:
            confirm_absent(lock)
    found.extend(f"projects/{name}" for name in _marked_or_named_shelves(root))
    return found


def _marked_or_named_shelves(root) -> list:
    """Shelf names that rule out first use: a marker still claims the shelf, or an older named
    store holds the namespace. ``_only_workspace_shelves`` is this walk's boolean."""
    projects = Path(root) / "projects"
    found = []
    try:
        with os.scandir(projects) as stores:
            for entry in stores:
                if entry.name.startswith(".") or not entry.is_dir():
                    continue
                # A bare workspace or explicit shelf-only scope can use this namespace.
                # A marker in such a shelf, or an older named store, still rules out first use.
                if not can_be_workspace_shelf(entry.name):
                    found.append(entry.name)
                    continue
                marker = Path(entry.path) / ".project.json"
                try:
                    marker.lstat()
                except FileNotFoundError:
                    confirm_absent(marker)
                else:
                    found.append(entry.name)
    except FileNotFoundError:
        confirm_absent(projects)
    return found


def _only_workspace_shelves(root):
    return not _marked_or_named_shelves(root)


def _lock_path(path):
    return path.with_name(path.name + ".lock")


def _read_certificate(source):
    if source.read(len(_MAGIC)) == _MAGIC:
        raw = source.read(2049)
        if len(raw) > 2048:
            return None
        try:
            value = json.loads(raw)
        except (ValueError, UnicodeError):
            return None
        if isinstance(value, dict) and isinstance(value.get("digest"), str):
            return value
    return None


def _released_proof(path):
    """Read only. An idle certificate is never a writer lock, even on a kernel-lock volume."""
    lock = _lock_path(path)
    try:
        if not stat.S_ISREG(lock.lstat().st_mode):
            return None
        with lock.open("rb") as source:
            proof = _read_certificate(source)
            if proof and os.path.samestat(os.fstat(source.fileno()), lock.lstat()):
                local = _local_identity(path, source.fileno())
                identity = proof.get("identity")
                if local and isinstance(identity, list) and local[:-1] == identity[:-1]:
                    return proof
    except FileNotFoundError:
        pass
    return None


class _Writer:
    def __init__(self, path, fd, enforced, proof=None):
        self.path, self.fd, self.enforced = path, fd, enforced
        self.certificate = proof
        self.initial_proof = proof
        self.new_commit = False
        self.keep_name = False

    def certify(self, proof):
        self.certificate = proof
        self.new_commit = True

    def revoke(self):
        if self.certificate is not None:
            os.ftruncate(self.fd, 0)
            os.fsync(self.fd)
            self.certificate = None

    def retire(self, private):
        """Age the certificate only when the protected operation is over."""
        if self.certificate is not None and self.certificate is self.initial_proof:
            if os.name == "nt":
                os.utime(private, ns=(0, 0))
            return
        try:
            body = (_MAGIC + utils.machine_json(self.certificate).encode("utf-8")
                    if self.certificate else b"released")
            os.lseek(self.fd, 0, os.SEEK_SET)
            utils._write_fd_fully(self.fd, body, _lock_path(self.path))
            os.ftruncate(self.fd, len(body))
            if self.new_commit:
                os.fsync(self.fd)
        except OSError:
            os.ftruncate(self.fd, 0)
            os.fsync(self.fd)
            log.warning("Project registry certificate unavailable", exc_info=True)
        os.utime(private, ns=(0, 0))


@contextlib.contextmanager
def _platform_writer(path, timeout):
    lock = _lock_path(path)
    if not path.exists() and not witness_path(path.parent.parent).exists() and lock.is_file():
        # An unqualified writer would evict the last evidence of this registry.
        try:
            with lock.open("rb") as source:
                was_registry = source.read(len(_MAGIC)) == _MAGIC
        except FileNotFoundError:
            was_registry = False
        if was_registry:
            raise FileNotFoundError(str(path))
    fd = platform.acquire_exclusive_file_lock(lock, timeout_sec=timeout, writable_stale_probe=os.name != "nt")
    if fd is None:
        raise TimeoutError(f"Project registry writer lock busy: {lock}")
    writer = _Writer(path, fd, False)
    try:
        yield writer
    finally:
        if writer.keep_name:
            platform.file_unlock(fd)
            os.close(fd)
        else:
            platform.release_exclusive_file_lock(lock, fd)


def _open_proof(lock):
    """Pin the observed file. A stale-name read alone never authorizes the next writer."""
    flags = os.O_RDWR | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(lock, flags)
    except (FileNotFoundError, PermissionError, IsADirectoryError):
        return None
    try:
        if os.read(fd, len(_MAGIC)) == _MAGIC and os.path.samestat(os.fstat(fd), lock.lstat()):
            return fd
    except FileNotFoundError:
        pass
    except BaseException:
        os.close(fd)
        raise
    os.close(fd)
    return None


def _wait_for_kernel_lock(fd, lock, timeout):
    deadline = time.monotonic() + timeout
    while True:
        try:
            platform.file_lock_exclusive_nb(fd)
            return
        except OSError as error:
            if error.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                raise
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Project registry writer lock busy: {lock}") from error
            time.sleep(min(0.05, max(0, deadline - time.monotonic())))


@contextlib.contextmanager
def _writer_gate(lock, timeout):
    if not platform.kernel_file_locks_enforced(lock):
        yield False
        return
    with contextlib.ExitStack() as stack:
        try:
            gate = (os.open(lock.with_name(".projects.registry-gate"), os.O_RDWR | os.O_CREAT, 0o600)
                    if os.name == "nt" else os.open(lock.parent, os.O_RDONLY))
            stack.callback(os.close, gate)
            _wait_for_kernel_lock(gate, lock, timeout)
        except OSError as error:
            unavailable = _NO_DIRECTORY_LOCK if os.name != "nt" else _NO_LINK | {errno.ENOLCK}
            if error.errno not in unavailable:
                raise
            yield False
            return
        stack.callback(platform.file_unlock, gate)
        yield True


def _lock_type(lock):
    try:
        mode = stat.S_IFMT(lock.lstat().st_mode)
    except FileNotFoundError:
        return None
    if mode not in (stat.S_IFDIR, stat.S_IFREG):
        raise OSError(errno.EINVAL, "Invalid Project writer lock", str(lock))
    return mode


def _return_exchange(lock, partner):
    """Under the directory lock, return only the protocol's own exchange to its idle position."""
    utils.assert_test_data_path(partner)
    if (not stat.S_ISREG((lock / _EXCHANGE_MARKER).lstat().st_mode)
            or not stat.S_ISREG(partner.lstat().st_mode)):
        raise OSError(errno.EINVAL, "Invalid Project writer exchange", str(lock))
    # Revocation precedes a new commit. Preserve this inode and its contents.
    # The old writer can reclaim the aged name immediately.
    os.utime(partner, ns=(0, 0))
    rename_exclusive(partner, lock, exchange=True)


def hand_off(root):
    """After writers quiesce, leave the legacy writer its ordinary lock name."""
    path = registry_path(root)
    lock = _lock_path(path)
    mode = _lock_type(lock)
    if mode is None or (os.name == "nt" and mode == stat.S_IFREG):
        return
    utils.assert_test_data_path(lock)
    if os.name == "nt":
        raise OSError(errno.EISDIR, "Unexpected Project writer directory", str(lock))
    with _writer_gate(lock, 0) as qualified:
        mode = _lock_type(lock)
        if mode == stat.S_IFDIR:
            if not qualified:
                raise OSError(errno.ENOTSUP, "Cannot establish ownership of the Project writer directory", str(lock))
            _return_exchange(lock, lock.with_name(_EXCHANGE_PARTNER))
            utils.fsync_folder(lock.parent)
        elif not qualified and mode == stat.S_IFREG:
            # A previously released file can outlive the volume's directory-lock support.
            # The old writer only tries LOCK_EX on a read-only descriptor to reclaim it.
            with lock.open("rb") as source:
                released = source.read(len(_MAGIC)) in (_MAGIC, b"released")
            if released:
                with _platform_writer(path, 0):
                    pass
                confirm_absent(lock)
                utils.fsync_folder(lock.parent)


def _prepare_exchange(lock, partner):
    """Prepare or resume the private directory while holding its parent's kernel lock."""
    utils.assert_test_data_path(partner)
    if _lock_type(lock) == stat.S_IFDIR:
        _return_exchange(lock, partner)
    if not partner.exists():
        partner.mkdir(mode=0o700)
    if not stat.S_ISDIR(partner.lstat().st_mode):
        raise ValueError("Project writer exchange requires a local directory")
    if not (partner / _EXCHANGE_MARKER).is_file():
        if any(partner.iterdir()):
            raise ValueError("Project writer exchange directory is already in use")
        (partner / _EXCHANGE_MARKER).touch(exist_ok=False)
        utils.fsync_folder(partner)


@contextlib.contextmanager
def _qualified_writer(path, timeout):
    lock = _lock_path(path)
    utils.assert_test_data_path(lock)
    with contextlib.ExitStack() as stack:
        # Serialize the decision to reuse a certificate or acquire the legacy
        # name. A waiter must inspect the name AFTER the previous writer retires.
        if not stack.enter_context(_writer_gate(lock, timeout)):
            yield stack.enter_context(_platform_writer(path, timeout))
            return
        partner = lock if os.name == "nt" else lock.with_name(_EXCHANGE_PARTNER)
        if os.name != "nt":
            _prepare_exchange(lock, partner)
        pinned = _open_proof(lock)
        if pinned is None:
            acquired = stack.enter_context(_platform_writer(path, timeout))
            fd, proof = acquired.fd, None
            if not platform.kernel_file_locks_enforced(lock):
                yield acquired
                return
        else:
            fd = pinned
            pin = stack.enter_context(contextlib.ExitStack())
            pin.callback(os.close, fd)
            _wait_for_kernel_lock(fd, lock, timeout)
            pin.callback(platform.file_unlock, fd)
            with os.fdopen(fd, "rb", closefd=False) as source:
                source.seek(0)
                proof = _read_certificate(source)
            if proof is None and not path.exists():
                _record_existence(witness_path(path.parent.parent))
        if os.name == "nt":
            if pinned is not None:
                # The open handle prevents both a kernel peer and a name-only
                # peer from removing the name while this lock is active.
                os.utime(lock)
        else:
            try:
                rename_exclusive(partner, lock, exchange=True)
            except OSError as error:
                if error.errno not in _NO_LINK | {errno.EINVAL, errno.EXDEV, errno.EISDIR, errno.ENOTDIR}:
                    raise
                if pinned is None:
                    yield acquired
                else:
                    pin.close()
                    yield stack.enter_context(_platform_writer(path, timeout))
                return
            if not os.path.samestat(os.fstat(fd), partner.lstat()):
                rename_exclusive(partner, lock, exchange=True)
                raise TimeoutError(f"Project registry writer changed during acquisition: {lock}")
        # The platform normally unlinks an acquired name on exit. We own the FD
        # now and release it ourselves after publishing the retired name.
        if pinned is None:
            acquired.keep_name = True
        writer = _Writer(path, fd, True, proof)
        try:
            yield writer
        finally:
            writer.retire(partner)
            if os.name != "nt":
                rename_exclusive(partner, lock, exchange=True)
            if writer.new_commit:
                utils.fsync_folder(path.parent)


@contextlib.contextmanager
def writer_lock(path, timeout=4.0):
    held = _HELD.get()
    if held is not None and held.path == path:
        yield held
        return
    utils.assert_test_data_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _qualified_writer(path, timeout) as writer:
        token = _HELD.set(writer)
        try:
            yield writer
        finally:
            _HELD.reset(token)


def _local_identity(path, fd):
    directory, lock = path.parent.stat(), os.fstat(fd)
    source = Path(__file__)
    try:
        body = source.lstat()
    except OSError:
        return None
    if (source.suffix != ".py" or not stat.S_ISREG(body.st_mode)
            or not all((directory.st_ino, lock.st_ino, body.st_ino))):
        return None
    return [directory.st_dev, directory.st_ino, lock.st_dev, lock.st_ino,
            recorded_id(path.parent.parent), list(utils.file_identity(body))]


def _stamp(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Project registry requires a regular file at {path}")
    return list(utils.file_identity(info))


@contextlib.contextmanager
def _staged_bytes(path, raw, mode):
    """Stage a complete sibling. A new empty primary uses the regular writer's umask."""
    utils.assert_test_data_path(path)
    temporary = atomic_temp_path(path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd = os.open(temporary, flags, 0o644 if mode is None else 0o600)
    try:
        try:
            utils._write_fd_fully(fd, raw, temporary)
            if mode is not None:
                os.chmod(temporary, mode)
            os.fsync(fd)
        finally:
            os.close(fd)
        yield temporary
    finally:
        temporary.unlink(missing_ok=True)


def read_primary(path):
    """Capture one snapshot. Observing a different commit invalidates the old proof even if parsing fails."""
    with path.open("rb") as source:
        info = os.fstat(source.fileno())
        raw = source.read()
        writer = _HELD.get()
        if writer is not None and writer.path == path and writer.certificate:
            proof = writer.certificate
            primary = list(utils.file_identity(info))
            digest = hashlib.sha256(raw).hexdigest()
            if digest != proof.get("digest"):
                writer.revoke()
            elif primary != proof.get("primary"):
                try:
                    restored = _restore_status(path, primary, raw) != "ordinary"
                except (OSError, ValueError):
                    writer.revoke()
                    raise
                if not restored:
                    writer.revoke()
        return info, raw


def prior_working_folders(root) -> tuple[list[str], str]:
    """The working folders the PRIOR profile's registry named, and why they may be unknown.

    Before this Reset's own passes began clearing the state folder, the registry on disk is the
    prior profile's. Once clearing began, a registry that is still there is either the successor's
    own — published by this Reset after the clear, empty while no new profile serves, and proof
    the clear finished, so the prior rows it lacks cannot name anything a repeat can still touch —
    or the prior's own bytes that a cut inside the clear left behind, whose rows are then exactly
    the facts to check against what the repeat would still remove. Neither is ever read as proof
    the prior profile had no Projects. A registry that vanished WITH the clear, nothing published
    after it, leaves the prior folders unknown: that repeat stays interrupted. Before any clearing,
    an absent registry reads as no Projects only where no trace survives (``_registry_evidence``);
    any trace, any error leaves the folders unknown, never an empty list.
    """
    from ouroboros.launch_roots import RESETTING, read_witness

    path = registry_path(root)
    status, witness = read_witness(root)
    clearing = status == RESETTING and witness.get("reset_clearing_state") is True
    proof = (witness.get("reset_working_folders")
             if clearing and isinstance(witness.get("reset_working_folders"), list) else None)
    try:
        _info, raw = read_primary(path)
    except FileNotFoundError:
        confirm_absent(path)
        if clearing:
            if proof is not None:
                return [str(f) for f in proof if f], ""  # the saved proof predates the clear
            return [], ("this Reset already cleared the state folder and published no registry "
                        "after it, so the prior profile's working folders are unknown")
        return [], _absent_registry_reason(root)
    except OSError as exc:
        return [], f"the prior Project registry could not be read ({type(exc).__name__})"
    if clearing and proof is not None:
        return [str(f) for f in proof if f], ""  # the proof is the prior's facts; the registry may be the successor's
    try:
        rows = validated_registry(read_bytes(raw)).get("projects", [])
    except (UnicodeDecodeError, ValueError) as exc:
        return [], f"the prior Project registry is unreadable ({type(exc).__name__})"
    folders = [str(row.get("working_dir") or "").strip() for row in rows]
    found = [folder for folder in folders if folder]
    if clearing and not found:
        return [], ("this Reset already began clearing the state folder and the registry holds "
                    "no prior working folders; without saved proof they are unknown")
    return found, ""


def _absent_registry_reason(root) -> str:
    """Why this profile's absent registry leaves its Projects' folders unknown, else "": only a
    root no Project trace ever touched reads as no-Projects. The traces are the first-use proofs
    (``_registry_evidence``) — the registry's own witness, its restore receipt, its task bindings,
    a writer lock a commit still shapes, a claimed shelf — the same walk, not a second copy."""
    evidence = _registry_evidence(root)
    if not evidence:
        return ""
    return ("the prior Project registry is missing while " + ", ".join(evidence)
            + " still point at Projects here, so their working folders are unknown")


def _restore_receipt(path):
    return path.with_name("projects.json.restore")


def _restore_status(path, primary, raw):
    """A receipt identifies the file we published. It never authorizes a missing primary."""
    receipt = utils.json_snapshot(_restore_receipt(path), strict=True) or {}
    if receipt.get("primary") == primary:
        return "exact"
    if receipt.get("staged") == primary[:4] and receipt.get("digest") == hashlib.sha256(raw).hexdigest():
        return "unconfirmed"
    return "ordinary"


def _qualify(writer, witness, snapshot, restore_status="ordinary", *, identity=None) -> bool:
    """The caller validated these bytes and observed or published the file containing them."""
    if identity is None:
        identity = _local_identity(writer.path, writer.fd)
    if identity is None:
        return True
    info, raw = snapshot
    primary = list(utils.file_identity(info))
    if _stamp(writer.path) != primary:
        return True
    with _staged_bytes(witness, raw, 0o600) as temporary:
        utils.replace_atomic(temporary, witness)
    if _stamp(writer.path) != primary:
        return True
    writer.certify({"identity": identity, "digest": hashlib.sha256(raw).hexdigest(),
                    "primary": primary, "mode": stat.S_IMODE(info.st_mode),
                    "copy": _stamp(witness), "restore_status": restore_status})
    return restore_status != "ordinary"


def _publish_new_file(temporary, path):
    """Publish a staged primary only if the original name is still absent."""
    if os.name != "nt":
        try:
            os.link(temporary, path)
            return
        except OSError as error:
            if error.errno not in _NO_LINK:
                raise
    rename_exclusive(temporary, path)


def _empty_registry_bytes():
    return utils.machine_json({"_schema_version": SCHEMA_VERSION, "projects": []}).encode("utf-8")


def _publish_initial_file(temporary, path, *, context_still_valid):
    """Publish fully staged first-use bytes under the registry writer, never a restored copy."""
    try:
        _publish_new_file(temporary, path)
    except FileExistsError:
        return False
    except OSError as error:
        if error.errno not in {errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EINVAL}:
            raise

        def still_owns_name():
            if not context_still_valid():
                return False
            writer = _HELD.get()  # both first-use callers hold writer_lock(path)
            if not writer.enforced and not platform.refresh_exclusive_file_lock(_lock_path(path), writer.fd):
                raise TimeoutError(f"Project registry writer changed during initialization: {path}")
            try:
                confirm_absent(path)
            except FileExistsError:
                return False
            return True

        # An atomic publication among the profile's writers, not no-clobber against
        # non-participants on this filesystem. Recovery never uses this fallback.
        return utils.replace_atomic(temporary, path, precondition=still_owns_name)
    return True


def _start_empty(root, path):
    """Publish the initial empty set under the writer lock, before its existence witness."""
    if not could_be_first_registry(root, initializing=True):
        return False
    with _staged_bytes(path, _empty_registry_bytes(), None) as temporary:
        def first_use_current():
            if history_generation(root) != "current":
                raise OSError("Project registry first use has no current profile")
            return could_be_first_registry(root, initializing=True)

        if not _publish_initial_file(temporary, path, context_still_valid=first_use_current):
            return False
    utils.fsync_folder(path.parent)
    return True


def _record_existence(witness):
    if not witness.exists():
        utils.assert_test_data_path(witness)
        os.close(os.open(witness, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
        utils.fsync_folder(witness.parent)


def _certify_commit(writer, witness, *, staged=None, raw=None, data=None):
    if not writer.enforced:
        return
    try:
        if staged is None or not staged.st_ino:
            snapshot = read_primary(writer.path)
            if raw is not None and raw != snapshot[1]:
                return
            validated_registry(read_bytes(snapshot[1]))
        else:
            published = writer.path.lstat()
            if utils.file_identity(published)[:4] != utils.file_identity(staged)[:4]:
                return
            validated_registry(data)
            snapshot = published, raw
        _qualify(writer, witness, snapshot)
    except (OSError, ValueError):
        # Targeted writes can preserve an invalid unrelated row.
        log.warning("Project registry committed without a recovery certificate", exc_info=True)


def initialize_new_root(root, *, after_reset=False):
    """A Reset's exclusive owner calls this only after recording same-operation cleanup."""
    path = registry_path(root)
    if not after_reset and (path.exists() or witness_path(root).exists()):
        return
    with writer_lock(path) as writer:
        if after_reset:
            raw = _empty_registry_bytes()
            try:
                stamp = _stamp(path)
            except FileNotFoundError:
                confirm_absent(path)
                confirm_absent(witness_path(root))
                with _staged_bytes(path, raw, None) as temporary:
                    def reset_still_owns_empty_root():
                        # The admitting server proved cleanup for this operation and still
                        # holds exclusive data-root admission; no profile writer has started.
                        confirm_absent(witness_path(root))
                        return True

                    if not _publish_initial_file(temporary, path, context_still_valid=reset_still_owns_empty_root):
                        raise FileExistsError(f"Project registry appeared during Reset at {path}")
                stamp = _stamp(path)
            # No other profile writer is admitted until the receipt. An interrupted publish may already own
            # this name; accept only its exact complete bytes under the same writer, and never replace it.
            info, observed = read_primary(path)
            if list(utils.file_identity(info)) != stamp or observed != raw:
                raise OSError(f"Reset could not confirm the empty registry at {path}")
            utils.fsync_folder(path.parent)
        if after_reset or _start_empty(root, path):
            witness = witness_path(root)
            _record_existence(witness)
            _certify_commit(writer, witness)


def require_primary(root, path):
    """Under the writer lock, publish an eligible first registry before loading or saving its rows."""
    if not path.exists() and not _start_empty(root, path):
        raise FileNotFoundError(str(path))


def save(root, data):
    path, witness = registry_path(root), witness_path(root)
    with writer_lock(path) as writer:
        require_primary(root, path)
        # Even the first non-empty commit starts from a real primary. A cut
        # before its replacement leaves an empty registry that can be retried.
        _record_existence(witness)
        raw = utils.machine_json(data).encode("utf-8")
        writer.revoke()
        staged = utils.write_bytes_atomic(path, raw, fsync=True, attest=writer.enforced)
        _certify_commit(writer, witness, staged=staged, raw=raw, data=data)


def _restore_missing(writer, witness, local):
    path, proof = writer.path, writer.certificate or {}
    if not local or local != proof.get("identity") or _stamp(witness) != proof.get("copy"):
        return
    raw = witness.read_bytes()
    if hashlib.sha256(raw).hexdigest() != proof.get("digest"):
        return
    validated_registry(read_bytes(raw))
    mode = proof.get("mode")
    if type(mode) is not int or not 0 <= mode <= 0o7777:
        return
    receipt = _restore_receipt(path)
    with _staged_bytes(path, raw, mode) as temporary:
        staged = _stamp(temporary)[:4]  # dev, inode, size and mtime survive publication
        if not staged[1]:
            return
        utils.atomic_write_json(receipt, {"staged": staged, "digest": proof["digest"]}, fsync=True)
        utils.fsync_folder(path.parent)
        try:
            _publish_new_file(temporary, path)
        except FileExistsError:
            writer.revoke()
            return
    primary = _stamp(path)
    if primary[:4] != staged:
        writer.revoke()
        return
    # Restoring is not a new registry commit. Keep its certificate intact even
    # if the process stops during publication or while recording the notice.
    utils.atomic_write_json(receipt, {"primary": primary}, fsync=True)
    utils.fsync_folder(path.parent)


def reconcile(root, *, snapshot=None) -> bool:
    """Return a hold on legacy registration while a restored revision is current."""
    path, witness = registry_path(root), witness_path(root)
    if not path.exists() and not witness.exists():
        return True
    with writer_lock(path) as writer:
        local = _local_identity(path, writer.fd) if writer.enforced else None
        try:
            primary = _stamp(path)
        except FileNotFoundError:
            _restore_missing(writer, witness, local)
            return True
        except ValueError:
            writer.revoke()
            raise
        if snapshot is None:
            snapshot = read_primary(path)
            validated_registry(read_bytes(snapshot[1]))
        if primary != list(utils.file_identity(snapshot[0])):
            writer.revoke()
            return True
        proof = writer.certificate or {}
        identity = proof.get("identity")
        same_writer = local and isinstance(identity, list) and local[:-1] == identity[:-1]
        same_copy = same_writer and witness.exists() and _stamp(witness) == proof.get("copy")
        if same_copy and primary == proof.get("primary"):
            if local != identity:
                writer.certify({**proof, "identity": local})
            return proof.get("restore_status") != "ordinary"
        status = _restore_status(path, primary, snapshot[1])
        if same_copy and status != "ordinary" and local == identity:
            return True
        writer.revoke()
        if not local:
            _record_existence(witness)
            return status != "ordinary"
        return _qualify(writer, witness, snapshot, status, identity=local)


def recovery_notice(drive_root: Path) -> dict | None:
    """Passive. Only a confirmed restoration of this primary can carry a notice."""
    path = registry_path(drive_root)
    try:
        receipt = utils.json_snapshot(_restore_receipt(path), strict=True) or {}
        published = receipt.get("primary")
        if published is None:
            # Also recognize an exact restoration recorded by the previous writer.
            proof = _released_proof(path) or {}
            if proof.get("restore_status") == "exact":
                published = proof.get("primary")
        if published is not None and _stamp(path) == published:
            return {"kind": "exact_registry_restored"}
    except (OSError, ValueError):
        pass
    return None
