"""Captured body-switch helper: moves an ARMED adoption into the serving checkout.

Standard library plus the Git CLI only; it never imports the product. The
package-init hook (``ouroboros/__init__.py``) executes the COPY captured outside
the tree at authorize time, before any other body module is imported, with
``SERVING_ROOT`` injected — never the tree's own copy, which may be mid-switch.

It acts on two phases of ``handoff.json`` and nothing else:

* ``armed`` — the old generation's readers were confirmed stopped. Re-verify
  (base, branch, candidate object, clean switch set, no Panic), then switch.
* ``switching`` — an interrupted switch. Resume it by content.

A ``stuck`` transition refuses the boot (exit 3): a mixed tree is never imported.

Order: files first, each changed path replaced atomically (temp + fsync +
``os.replace``), so a crash leaves old-or-new bytes and never a torn file; then
the index entries; the branch compare-and-swap LAST. Recovery reads what is on
disk, so it understands every boundary. Bytes that are neither side — a later
owner edit — are never overwritten: the paths already switched are returned to
the old side, the foreign file stays as found and the transition is abandoned
onto the coherent old tree. A HEAD that is neither side stops before any write.
The same rule holds for the index: an entry that is neither side (a later
``git add``) is never rewritten, in either direction. A file's state is its
content AND its executable bit (where the checkout tracks one), so a
permission-only change is switched like any other.

A process that starts on a ``switched`` tree verifies, BEFORE its imports, that
the checkout is at the candidate and every switch path holds the candidate's
bytes, and leaves that fact on ``sys`` for ``body_adoption.finalize_on_boot``:
the adoption is confirmed from what this process loaded, never from a later
read of HEAD.

A managed Update (``kind=update``, schema 2) carries the owner's uncommitted
work: each switch row names the commit side, the INDEX side and the FILE side
separately, so one path may differ in all three. Its switch set may be dirty by
design; a path is later work only when its file or index entry is neither
recorded side (``_foreign_paths``). After the branch moves, its own packages (a
changed lock, a new app) are selected at the one path the body runs with
(``_select_environment``), then an import probe of the new tree runs in a scratch
child (``_probe``); either failing returns the whole transition before any entry.
Entries are recorded before their imports (``_update_entry``), and a later entry
after an unready one asks ``qualify_return`` whether the previous generation may
run on what is there now: data is never rolled back, and a Panic forbids every
automatic step. A bare interpreter's private environment (``kind=private``)
names the interpreter the probe and the hand-over run.

An Update that also replaces the official macOS app (a ``native`` record) is
switched by the captured app installer (``app_install.py``, ``native_switch``),
never by a hook: the new app's frozen bootstrap installs packages and imports
the tree before any hooked entry. An entry that finds such a transition armed
leaves it to a live installer and abandons it when that process is provably
gone; an entry that finds the tree switched while the installed app is still
the previous one (the installer died before its exchange) returns the tree
(``_app_not_exchanged``). Once the new app may have run, no return is automatic.
"""
import contextlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time

HANDOFF_NAME = "handoff.json"
LOCK_NAME = "lock"
POINTER_NAME = "ouroboros-body-adoption"
RESTART_EXIT_CODE = 42
STUCK_EXIT_CODE = 3
_TMP_PREFIX = ".body-switch-"
_CHUNK = 200
_DEPS_TIMEOUT_SEC = 900
# Wire names shared with delegate_recovery; the captured helper cannot import the body.
_RESTART_PARENT = "OUROBOROS_PLANNED_RESTART_PARENT_HANDLE"
_RESTART_TRANSACTION = "OUROBOROS_PLANNED_RESTART_TRANSACTION_ID"
_RESTART_BINDING = "OUROBOROS_PLANNED_RESTART_BINDING_HANDLE"
# Test seam only (fault injection at a named boundary); production never sets it.
_FAIL_AT = os.environ.get("OUROBOROS_BODY_SWITCH_FAIL_AT", "")
# The import probe's child names the handoff it probes: its own hook must not enter the switch again.
_PROBE_ENV = "OUROBOROS_BODY_SWITCH_PROBE"
_PROBE_TIMEOUT_SEC = 600
# The installation's scope marker (``process_census.INSTALL_SCOPE_PREFIX``, restated: stdlib only before imports).
_INSTALL_SCOPE_PREFIX = "OURO_INSTALL_SCOPE_"
# Source paths whose code a launcher process runs before any server entry of the new tree.
_LAUNCHER_PATHS = ("launcher.py", "ouroboros/launcher")
# The environment state this process's entry found (``main``); ``_return_to_old`` says why a return reads it.
_ENTERED_ON = []


def installation_panic(root):
    # Both the serving helper and its captured copy use this stdlib-only reader.
    import runpy
    reader = runpy.run_path(os.path.join(os.path.dirname(__file__), "installation_stop.py"))
    return reader["installation_panic"](root)


class _Stuck(Exception):
    """The transition cannot continue without the owner; the tree is left as found."""


class _Foreign(Exception):
    """A switch path holds bytes that are neither the old nor the new side."""


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_handoff(helper_dir):
    with open(os.path.join(helper_dir, HANDOFF_NAME), encoding="utf-8") as fh:
        return json.load(fh)


def write_handoff(helper_dir, handoff):
    _write_json(os.path.join(helper_dir, HANDOFF_NAME), handoff)


def _write_json(path, value):
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(value, fh, indent=1, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp)


def record(helper_dir, handoff, phase, detail=""):
    handoff["phase"] = phase
    handoff.setdefault("events", []).append({"ts": _now(), "pid": os.getpid(), "phase": phase, "detail": detail})
    write_handoff(helper_dir, handoff)


def _git(root, *args, data=None, check=True, index=None):
    env = dict(os.environ, GIT_LITERAL_PATHSPECS="1", GIT_OPTIONAL_LOCKS="0")
    for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        env.pop(key, None)
    if index:
        env["GIT_INDEX_FILE"] = index  # a scratch index outside the work tree; the real one is never written
    proc = subprocess.run(["git", *args], cwd=root, env=env, input=data,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and proc.returncode != 0:
        raise RuntimeError("git %s failed rc=%d: %s" % (
            " ".join(args[:2]), proc.returncode, proc.stderr.decode("utf-8", "replace").strip()))
    return proc.returncode, proc.stdout


def git_dir(root):
    return _git(root, "rev-parse", "--absolute-git-dir")[1].decode("utf-8", "surrogateescape").strip()


def pointer_path(root):
    return os.path.join(git_dir(root), POINTER_NAME)


def _remove_pointer(root):
    with contextlib.suppress(OSError):
        os.unlink(pointer_path(root))


def _fail_point(name):
    if _FAIL_AT == name:
        sys.stderr.write("[body_switch] fault injection at %s\n" % name)
        sys.stderr.flush()
        os._exit(137)


def _head(root):
    return _git(root, "rev-parse", "--verify", "HEAD")[1].decode().strip()


def _branch(root):
    rc, out = _git(root, "symbolic-ref", "-q", "--short", "HEAD", check=False)
    return out.decode("utf-8", "replace").strip() if rc == 0 else ""


def _tracks_filemode(root):
    """Whether this checkout records the executable bit (``core.filemode``; false on Windows)."""
    rc, out = _git(root, "config", "--bool", "core.filemode", check=False)
    return not (rc == 0 and out.decode("utf-8", "replace").strip() == "false")


def _file_mode(mode, tracked):
    """A tree mode as a comparable on-disk kind: the executable bit only where it is tracked."""
    if mode in ("100644", "100755"):
        return mode if tracked else "file"
    return mode


def _file_target(row, side):
    """``(blob id, tree mode)`` ``side`` puts ON DISK; schema 2 rows keep it apart from the commit side."""
    key = side + "_file" if side + "_file_sha" in row else side
    return row[key + "_sha"], row[key + "_mode"]


def _side(row, side, tracked):
    """What ``side`` ("old"/"new") wants ON DISK at this path: ``(blob id, kind)``, ``(None, None)`` = absent."""
    sha, mode = _file_target(row, side)
    return (None, None) if sha is None else (sha, _file_mode(mode, tracked))


def _index_side(row, side):
    key = side + "_index" if side + "_index_sha" in row else side
    sha = row[key + "_sha"]
    return (None, None) if sha is None else (sha, row[key + "_mode"])


def _states(root, rows, tracked):
    """``(blob id, kind)`` of what is on disk at each switch path (``(None, None)`` absent).

    Regular files go through ``git hash-object`` so the repository's own clean
    filters apply: an untouched CRLF checkout still equals its blob.
    """
    states, regular = {}, []
    switch_paths = {row["path"] for row in rows}
    for row in rows:
        full = os.path.join(root, row["path"])
        try:
            mode = os.lstat(full).st_mode
        except (FileNotFoundError, NotADirectoryError):
            states[row["path"]] = (None, None)
            continue
        if stat.S_ISLNK(mode):
            target = os.readlink(full).encode("utf-8", "surrogateescape")
            states[row["path"]] = (_git(root, "hash-object", "--stdin", data=target)[1].decode().strip(), "120000")
        elif stat.S_ISDIR(mode):
            states[row["path"]] = _dir_state(root, row["path"], switch_paths)
        else:
            regular.append((row["path"], _file_mode("100755" if mode & stat.S_IXUSR else "100644", tracked)))
    for start in range(0, len(regular), _CHUNK):
        chunk = [path for path, _kind in regular[start:start + _CHUNK]]
        if any("\n" in path for path in chunk):
            hashes = [_git(root, "hash-object", "--", path)[1].decode().strip() for path in chunk]
        else:
            data = "\n".join(chunk).encode("utf-8", "surrogateescape") + b"\n"
            hashes = _git(root, "hash-object", "--stdin-paths", data=data)[1].decode().split()
        for (path, kind), sha in zip(regular[start:start + _CHUNK], hashes):
            states[path] = (sha, kind)
    return states


def _dir_state(root, path, switch_paths):
    """What a DIRECTORY at a switch path means: absent when it holds nothing but other switch
    paths (the other side's own files, which the deletions ordered ahead of this path remove) and
    bytecode caches; foreign ("directory") when anything else lives there."""
    for dirpath, dirnames, filenames in os.walk(os.path.join(root, path)):
        for name in filenames + [name for name in dirnames if os.path.islink(os.path.join(dirpath, name))]:
            full = os.path.join(dirpath, name)
            bytecode = (os.path.basename(dirpath) == "__pycache__" and name.endswith(".pyc")
                        and not os.path.islink(full))
            if not bytecode and os.path.relpath(full, root).replace(os.sep, "/") not in switch_paths:
                return ("directory", "directory")
    return (None, None)


def _drop_bytecode(directory, path):
    """A stale bytecode file must not outlive (or stand in for) its source."""
    if not path.endswith(".py"):
        return
    cache = os.path.join(directory, "__pycache__")
    stem = os.path.basename(path)[:-3] + "."
    with contextlib.suppress(OSError):
        for name in os.listdir(cache):
            if name.startswith(stem) and name.endswith(".pyc"):
                with contextlib.suppress(OSError):
                    os.unlink(os.path.join(cache, name))
    with contextlib.suppress(OSError):
        os.rmdir(cache)  # only when it is empty now: a file may become a directory here


def _index_states(root, rows):
    """``(blob id, mode)`` of the index entry at each switch path (``(None, None)`` absent).

    An unmerged path (any stage but 0) reads ``("unmerged", ...)``: it is never either side.
    """
    states = {row["path"]: (None, None) for row in rows}
    paths = list(states)
    for start in range(0, len(paths), _CHUNK):
        out = _git(root, "ls-files", "-s", "-z", "--", *paths[start:start + _CHUNK])[1]
        for chunk in out.split(b"\0"):
            if not chunk:
                continue
            meta, _tab, raw = chunk.partition(b"\t")
            mode, sha, stage = meta.decode().split(" ")
            path = raw.decode("utf-8", "surrogateescape")
            if path in states:
                states[path] = (sha, mode) if stage == "0" else ("unmerged", stage)
    return states


def _foreign_paths(rows, files, index, tracked):
    """Switch paths whose file or index entry is neither the old nor the new side: later work."""
    return sorted(
        row["path"] for row in rows
        if files[row["path"]] not in (_side(row, "old", tracked), _side(row, "new", tracked))
        or index[row["path"]] not in (_index_side(row, "old"), _index_side(row, "new")))


def _prune_empty_parents(root, path):
    parent = os.path.dirname(os.path.join(root, path))
    while os.path.realpath(parent) != os.path.realpath(root) and parent.startswith(root):
        try:
            os.rmdir(parent)
        except OSError:
            return
        parent = os.path.dirname(parent)


def _put(root, path, mode, sha):
    """Make ``path`` hold exactly ``sha``/``mode``; one atomic replace per file."""
    full = os.path.join(root, path)
    if sha is None:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(full)
        _drop_bytecode(os.path.dirname(full), path)
        _prune_empty_parents(root, path)
        return
    directory = os.path.dirname(full)
    os.makedirs(directory, exist_ok=True)
    if os.path.isdir(full) and not os.path.islink(full):
        # Other switch paths were deleted first. Historical caches have no current
        # source row; remove only bytecode and empty directories, never foreign files.
        for parent, _dirs, names in os.walk(full, topdown=False):
            if os.path.basename(parent) == "__pycache__":
                for name in names:
                    cache = os.path.join(parent, name)
                    if name.endswith(".pyc") and not os.path.islink(cache):
                        os.unlink(cache)
            os.rmdir(parent)
    data = _git(root, "cat-file", "blob", sha)[1]
    if mode == "120000":
        tmp = os.path.join(directory, "%s%d.link" % (_TMP_PREFIX, os.getpid()))
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        os.symlink(data.decode("utf-8", "surrogateescape"), tmp)
        os.replace(tmp, full)
        return
    fd, tmp = tempfile.mkstemp(prefix=_TMP_PREFIX, dir=directory)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o755 if mode == "100755" else 0o644)
        os.replace(tmp, full)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    _drop_bytecode(directory, path)


def _converge(root, rows, side, files, tracked, keep=()):
    """Bring every switch path that holds the OTHER side to ``side`` ("new" or "old").

    Only our own bytes move: a path already at ``side`` is left, and a path in
    ``keep`` (foreign work the caller found before any write) is never touched.
    Deletions run first so a file can become a directory and back.
    """
    other = "old" if side == "new" else "new"
    ordered = sorted(rows, key=lambda row: _file_target(row, side)[0] is not None)
    for index, row in enumerate(ordered):
        have, want = files[row["path"]], _side(row, side, tracked)
        if have == want or row["path"] in keep or have != _side(row, other, tracked):
            continue
        sha, mode = _file_target(row, side)
        _put(root, row["path"], mode, sha)
        if side == "new" and index == len(ordered) // 2:
            _fail_point("mid_files")


def _set_index(root, rows, side, index):
    """Move the index entries that hold the OTHER side to ``side``; a later ``git add`` stays."""
    other = "old" if side == "new" else "new"
    zero = "0" * len(next((sha for row in rows for name in ("old", "new")
                           for sha in (_index_side(row, name)[0], _file_target(row, name)[0]) if sha), "0" * 40))
    lines = []
    for row in rows:
        if index[row["path"]] != _index_side(row, other) or _index_side(row, side) == _index_side(row, other):
            continue
        sha, mode = _index_side(row, side)
        entry = ("0 %s\t%s" % (zero, row["path"])) if sha is None else ("%s %s\t%s" % (mode, sha, row["path"]))
        lines.append(entry.encode("utf-8", "surrogateescape"))
    if lines:
        _git(root, "update-index", "-z", "--index-info", data=b"\0".join(lines) + b"\0")


def _dirty(root, paths):
    out = b""
    for start in range(0, len(paths), _CHUNK):
        out += _git(root, "status", "--porcelain", "--untracked-files=all", "--", *paths[start:start + _CHUNK])[1]
    return out.decode("utf-8", "replace").strip()


def bundle_identity(path):
    """``[device, inode, Contents inode]`` of a bundle directory, ``None`` when absent.

    A rename on its volume keeps all three (the exchange, the backup). The ``Contents`` inode
    tells apart an installer that kept the folder and replaced what is in it: Homebrew with
    app-management permission moves (or copies) a new ``Contents`` into the same folder.
    """
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return None
    if not stat.S_ISDIR(st.st_mode):
        return ["not_a_directory"]
    try:
        contents = os.lstat(os.path.join(path, "Contents")).st_ino
    except OSError:
        contents = None
    return [st.st_dev, st.st_ino, contents]


def identity_stale(recorded):
    """A ``[device, inode]`` identity, recorded before ``Contents`` was part of it: it proves no bundle.

    A decision on it is refused (a candidate is prepared again, a readback fails); only the move
    that keeps a previous app as its backup reads it as written (``same_bundle``).
    """
    return isinstance(recorded, list) and len(recorded) == 2


def same_bundle(path, recorded):
    """Whether ``path`` is the bundle ``recorded`` names, for a move that only keeps it (the backup).

    A stale identity (``identity_stale``) is compared as it was written, by its folder alone, so a
    previous app it names is moved to its backup rather than left where retirement deletes it.
    """
    identity = bundle_identity(path)
    if identity_stale(recorded) and isinstance(identity, list) and len(identity) == 3:
        return identity[:2] == recorded
    return identity == recorded


def process_gone(identity):
    """True only when the recorded process (pid and its ``ps`` start time) provably no longer runs.

    An identity that was never recorded names nothing that runs. A pid that is still
    signalable with an unreadable start time is not gone: unknown is never gone.
    """
    pid = int((identity or {}).get("pid") or 0)
    if pid <= 0:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        pass  # it exists and refuses us
    start = str((identity or {}).get("start_time") or "")
    try:
        shown = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False
    now = shown.stdout.decode("utf-8", "replace").strip()
    if now:
        return bool(start) and now != start  # another process holds that pid now
    try:  # no row: it may have just exited
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        pass
    return False


def _app_not_exchanged(handoff):
    """The installer died after the switch and before its exchange: the previous app is still installed.

    Only then, and only while no entry other than the installer's own intent ran the
    switched tree, is returning it positively safe (nothing of the new tree ran outside
    the scratch probe).
    """
    native = handoff.get("native") or {}
    return (bool(native.get("previous_identity"))
            and bundle_identity(native.get("installed") or "") == native["previous_identity"]
            and process_gone(native.get("installer"))
            and not any(entry.get("kind") != "app" for entry in handoff.get("entries") or []))


def _armed_refusal(root, handoff):
    """Typed reason to abandon an ARMED handoff with the tree untouched, or ``""``."""
    if _head(root) != handoff["old"]:
        return "base_moved:%s" % _head(root)[:12]
    if _branch(root) != handoff["branch"]:
        return "branch_changed:%s" % (_branch(root) or "detached")
    if _git(root, "cat-file", "-e", "%s^{commit}" % handoff["cand"], check=False)[0] != 0:
        return "candidate_missing"
    if installation_panic(handoff.get("data_dir") or ""):
        return "panic_stop"
    if handoff.get("kind") == "update":
        return ""  # its old side is the owner's dirty work; later work is any other bytes (``_switch``)
    dirt = _dirty(root, [row["path"] for row in handoff["switch"]])
    return ("switch_set_dirty:%s" % dirt.splitlines()[0]) if dirt else ""


@contextlib.contextmanager
def _locked(helper_dir, wait=True):
    """One switcher at a time; the kernel releases the lock when the process ends or execs.

    ``wait=False`` never waits: another holder raises ``BlockingIOError`` at once (``body_adoption.arm``)."""
    fh = open(os.path.join(helper_dir, LOCK_NAME), "a+")
    try:
        if sys.platform == "win32":
            import msvcrt

            deadline = time.time() + 60
            while True:
                try:
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if not wait:
                        raise BlockingIOError(exc.errno, "the helper's lock is held") from exc
                    if time.time() > deadline:
                        raise
                    time.sleep(0.1)
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX if wait else fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        fh.close()


def _windows_identity(handle):
    """PID and creation FILETIME from an owned HANDLE, matching the runtime's birth token."""
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetProcessId.argtypes, kernel.GetProcessId.restype = (wintypes.HANDLE,), wintypes.DWORD
    kernel.GetProcessTimes.argtypes = (wintypes.HANDLE, *(ctypes.POINTER(wintypes.FILETIME),) * 4)
    kernel.GetProcessTimes.restype = wintypes.BOOL
    created, exited, system, user = (wintypes.FILETIME() for _ in range(4))
    pid = kernel.GetProcessId(handle)
    if not pid or not kernel.GetProcessTimes(
        handle, ctypes.byref(created), ctypes.byref(exited), ctypes.byref(system), ctypes.byref(user),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return {"pid": int(pid), "birth": "win-filetime:%d" % ((created.dwHighDateTime << 32) | created.dwLowDateTime)}


def _bind_windows_successor(parent, proc):
    """Transfer only this helper's exact binding, after the original writer has exited."""
    import _winapi

    if _winapi.WaitForSingleObject(parent, _winapi.INFINITE) != _winapi.WAIT_OBJECT_0:
        raise OSError("restart parent did not signal")
    if _winapi.GetExitCodeProcess(parent) != RESTART_EXIT_CODE:
        return  # final main still observes the original handle, including Panic's exit 99
    original, current = _windows_identity(parent), _windows_identity(_winapi.GetCurrentProcess())
    transaction = os.environ.get(_RESTART_TRANSACTION, "")
    if not transaction:
        return
    data_dir = read_handoff(os.path.dirname(os.path.abspath(__file__)))["data_dir"]
    path = os.path.join(data_dir, "state", "delegate_recovery_transactions", transaction + ".json")
    with open(path, encoding="utf-8") as fh:
        row = json.load(fh)
    if (row.get("status") != "prepared" or row.get("transaction_id") != transaction
            or (row.get("supervisor_pid"), row.get("supervisor_birth")) != (original["pid"], original["birth"])
            or (row.get("successor_pid"), row.get("successor_birth")) != (current["pid"], current["birth"])):
        raise ValueError("restart transaction does not bind this bootstrap generation")
    child = _windows_identity(proc._handle)
    row.update(successor_pid=child["pid"], successor_birth=child["birth"], updated_at=_now())
    _write_json(path, row)


def _windows_handoff(argv, venv=None):
    """Forward original-parent proof and release final main only after rebinding.

    A venv's interpreter (``venv``, else this process's own when it runs in one) is started as its
    base Python with ``__PYVENV_LAUNCHER__``, as CPython's redirector would, minus the redirector.

    EOF is a gate, never proof: even a failed write or this helper's death releases
    the child, whose normal consumer still checks the exact transaction. Only the
    read end is inherited, so our death cannot leave a writer keeping it blocked.
    """
    import _winapi
    import msvcrt

    env = dict(os.environ)
    argv = list(argv)
    base = getattr(sys, "_base_executable", None) or sys.executable
    venv = venv or (sys.executable if base != sys.executable else None)
    if venv:
        argv[0], env["__PYVENV_LAUNCHER__"] = base, venv
    if not env.get(_RESTART_PARENT):
        return subprocess.Popen(argv, env=env)
    with contextlib.ExitStack() as resources:
        parent = int(env[_RESTART_PARENT])
        resources.callback(_winapi.CloseHandle, parent)
        reader, writer = os.pipe()
        resources.callback(os.close, reader)
        resources.callback(os.close, writer)
        os.set_inheritable(reader, True)
        gate = msvcrt.get_osfhandle(reader)
        env[_RESTART_BINDING] = str(gate)
        startup = subprocess.STARTUPINFO(dwFlags=subprocess.STARTF_USESTDHANDLES,
                                        lpAttributeList={"handle_list": [parent, gate]})
        current = _winapi.GetCurrentProcess()
        # Same STARTUPINFO contract as platform_layer.inheritable_self_handle:
        # duplicate present streams; preserve absent ones; never synthesize pipes.
        for name, field in (("stdin", "hStdInput"), ("stdout", "hStdOutput"), ("stderr", "hStdError")):
            try:
                raw = msvcrt.get_osfhandle(getattr(sys, "__%s__" % name).fileno())
            except (AttributeError, OSError, ValueError):
                raw = 0
            duplicate = 0
            if raw not in (0, -1, -2):
                duplicate = subprocess.Handle(_winapi.DuplicateHandle(
                    current, raw, current, 0, True, _winapi.DUPLICATE_SAME_ACCESS))
                resources.callback(duplicate.Close)
                startup.lpAttributeList["handle_list"].append(duplicate)
            setattr(startup, field, duplicate)
        proc = subprocess.Popen(argv, env=env, startupinfo=startup, close_fds=True)
        try:
            _bind_windows_successor(parent, proc)
        except (Exception, KeyboardInterrupt) as exc:
            # The waited-on child stays available. An unproved continuation stays held.
            with contextlib.suppress(AttributeError, OSError, ValueError):
                sys.stderr.write("[body_switch] restart binding unproven; successor remains available: %s\n" % exc)
        return proc  # close the gate BEFORE waiting for the final process


def _wait_as_parent(argv, venv=None):
    """Windows has no process-image replacement: run the fresh entry and stay its waited-on parent.

    The child keeps this console group, as an exec would, so the console's
    CTRL+C / CTRL+BREAK reach it directly and ITS OWN stop runs: this parent
    never kills it and never answers the interrupt itself. The child's exit
    status (a Panic, a stop, the restart code of a supervising launcher) is
    returned unchanged. ``venv``: the entry runs in that venv (``_windows_handoff``).
    """
    proc = _windows_handoff(argv, venv)
    while True:
        try:
            return proc.wait()
        except KeyboardInterrupt:
            continue


def _venv_python(root):
    """The interpreter of the venv at ``root`` (``runtime_venv.interpreter``; the helper cannot import it)."""
    return os.path.join(root, *(("Scripts", "python.exe") if sys.platform == "win32" else ("bin", "python")))


def _handover(handoff=None):
    """Give the switched tree to a FRESH process through the owning lifecycle.

    A bare interpreter's private environment (``kind=private``) names the interpreter: the
    selected environment's own, or after a return the one the previous generation ran on.
    """
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, OSError, ValueError):
            stream.flush()
    if os.environ.get("OUROBOROS_MANAGED_BY_LAUNCHER") == "1":
        os._exit(RESTART_EXIT_CODE)  # the launcher's own cycle: dependencies, native host, relaunch
    environment, python = (handoff or {}).get("environment") or {}, sys.executable
    if environment.get("kind") == "private":
        python = (_venv_python(environment["current"]) if environment.get("state") == "selected"
                  else environment.get("python") or python)
    argv = [python, *sys.orig_argv[1:]]
    if sys.platform == "win32":  # a venv's interpreter runs as its base Python: the waited-on child is the entry
        in_venv = os.path.isfile(os.path.join(os.path.dirname(os.path.dirname(python)), "pyvenv.cfg"))
        os._exit(_wait_as_parent(argv, python if in_venv else None))
    os.execv(python, argv)


def _return_to_old(root, helper_dir, handoff, reason, files_at_start, phase="abandoned"):
    """Undo OUR writes only, then abandon (or, after a switch, return): the old tree is coherent again.

    Boot continues in this process only when it began on a wholly old tree (it
    parsed its entry and the hook from the old side). A process that resumed a
    half-switched tree may already hold the candidate's compiled entry, and one whose
    interpreter started on a switched-in environment may have run its startup files, so
    it returns True: the caller hands the returned tree to a FRESH process instead.
    """
    rows, tracked = handoff["switch"], _tracks_filemode(root)
    _converge(root, rows, "old", _states(root, rows, tracked), tracked)
    _set_index(root, rows, "old", _index_states(root, rows))  # a foreign file then reads as an unstaged edit
    if _head(root) == handoff["cand"] != handoff["old"]:
        _git(root, "update-ref", "refs/heads/%s" % handoff["branch"], handoff["old"], handoff["cand"])
    _restore_environment(helper_dir, handoff)
    environment = handoff.get("environment") or {}
    if environment.get("state") == "unrestored":
        # The previous tree without its own packages is no coherent generation: nothing starts on it.
        raise _Stuck("the previous Python packages could not be put back at %s (%s); they are kept at %s" % (
            environment.get("current"), environment.get("restore_error") or "another directory is there",
            environment.get("previous")))
    record(helper_dir, handoff, phase, reason)
    _remove_pointer(root)
    started_on_new = bool(_ENTERED_ON) and _ENTERED_ON[0] in ("selecting", "selected")
    return started_on_new or any(files_at_start[row["path"]] == _side(row, "new", tracked) != _side(row, "old", tracked)
                                 for row in rows)


def _returned(root, helper_dir, handoff, trigger, files_at_start=None, reason="", **facts):
    """Record the decision ``returned`` and write the previous composition back (``_return_to_old``, ``reason``
    defaulting to ``trigger``); ``files_at_start`` (what was there before this process wrote) defaults to now."""
    handoff["return"] = {"outcome": "returned", "trigger": trigger, "reasons": [], "at": _now(), **facts}
    if files_at_start is None:
        files_at_start = _states(root, handoff["switch"], _tracks_filemode(root))
    return _return_to_old(root, helper_dir, handoff, reason or trigger, files_at_start, phase="returned")


def _path_identity(path):
    """``[device, inode]`` of what is at ``path`` itself (a link is not followed), ``None`` when nothing is.

    A rename on its volume keeps it. A directory made again at that path may get the same one
    (ext4 hands out a freed inode at once), and one emptied in place keeps it everywhere.
    """
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return None
    return [st.st_dev, st.st_ino]


GENERATION_WITNESS = "ouroboros-update-generation.json"


def mark_generation(path):
    """Preparation's ``[device, inode, token]`` of a generation it made, a fresh token written inside it.

    Authorization and the switch prove the generation by it (``_is_prepared``). The renames of a
    selection and a return keep the token with the directory; a directory made again at the path,
    or the generation emptied in place, has none, whatever inode it keeps.
    """
    token = os.urandom(16).hex()
    _write_json(os.path.join(path, GENERATION_WITNESS), token)
    return _path_identity(path) + [token]


def _is_prepared(path, prepared):
    """Is what is at ``path`` the generation its preparation marked (``mark_generation``)?

    By its token, the directory itself at ``path`` (a link is not followed): the device and inode
    recorded beside it neither prove it (a directory made again may get them) nor disprove it (a
    reboot may renumber the device: btrfs's anonymous device numbers, a renumbered disk).
    An identity recorded before the token was part of it (``[device, inode]``) is compared as it was written.
    """
    if not prepared:
        return False
    if len(prepared) < 3:
        return _path_identity(path) == prepared[:2]
    if os.path.islink(path) or not os.path.isdir(path):
        return False
    try:
        with open(os.path.join(path, GENERATION_WITNESS), encoding="utf-8") as fh:
            return json.load(fh) == prepared[2]
    except (OSError, ValueError):
        return False


def _set_aside(helper_dir, handoff, path):
    """Keep what something else made at the fixed path: moved to a free ``<path>.foreign-<id>``, named in the record."""
    target = base = "%s.foreign-%s" % (path, handoff.get("id") or "update")
    count = 1
    while os.path.lexists(target):
        count += 1
        target = "%s-%d" % (base, count)
    os.rename(path, target)
    handoff["environment"].setdefault("foreign", []).append(target)
    write_handoff(helper_dir, handoff)


def _select_environment(helper_dir, handoff):
    """Put the new generation's prepared packages at the one path every launcher runs the body with.

    Two renames on one volume, recorded first and resumed by what is where: the serving
    packages move aside (kept for a return), the prepared ones take their place. Each is
    told by what authorization recorded (``identity``), never by a name alone: the serving
    one by its device and inode, the prepared one by the token its preparation wrote into it (``mark_generation``): a
    directory something else made at the fixed path between the renames (a launcher's user
    base, a fresh runtime venv), even one that got the prepared inode again, is neither, so it
    is kept aside (``_set_aside``), never selected or deleted. A bare interpreter's first
    private environment has no serving one (``None``). Before any import of the new tree; a
    launcher's dependency sync then finds the new lock already satisfied.
    """
    environment = handoff.get("environment")
    if not environment or environment.get("state") == "selected":
        return
    current, generation, previous = environment["current"], environment["generation"], environment["previous"]
    serving, prepared = environment["identity"]["current"], environment["identity"]["generation"]
    environment["state"] = "selecting"
    write_handoff(helper_dir, handoff)
    if serving and _path_identity(current) == serving and not os.path.lexists(previous):
        _rename(current, previous, "serving_aside")  # refused (on Windows, say, a process inside it): nothing moved
    _fail_point("mid_environment")
    if _path_identity(current) not in (None, serving) and not _is_prepared(current, prepared):
        _set_aside(helper_dir, handoff, current)
    if _is_prepared(generation, prepared) and not os.path.lexists(current):
        _rename(generation, current, "prepared_in")  # refused: the return puts the serving packages back
    if not _is_prepared(current, prepared) or (serving and _path_identity(previous) != serving):
        raise _Stuck("the prepared dependency environment could not be selected at %s" % current)
    if environment.get("kind") in ("venv", "private"):
        _retarget_scripts(current, generation, current)  # resumable: a second pass finds nothing to change
    environment["state"] = "selected"
    write_handoff(helper_dir, handoff)


def _rename(source, target, step):
    """``os.rename`` whose refusal names the selection step it stopped (``OSError.step``)."""
    try:
        os.rename(source, target)
    except OSError as exc:
        exc.step = step
        raise


def _os_error(exc):
    """What the system said, as recorded for the owner: never more than the error itself proves."""
    return {key: value for key, value in (
        ("type", type(exc).__name__), ("errno", getattr(exc, "errno", None)),
        ("winerror", getattr(exc, "winerror", None)), ("strerror", getattr(exc, "strerror", None)),
        ("filename", getattr(exc, "filename", None)), ("filename2", getattr(exc, "filename2", None)),
        ("step", getattr(exc, "step", None))) if value is not None}


def _retarget_scripts(root, old, new, windows=None):
    """Point a venv's console scripts at the path it now runs from.

    pip names a script's interpreter by the path the environment had when it wrote
    the script. In ``bin/``: a ``#!<old>/bin/python`` line, or for a long path distlib's
    ``/bin/sh`` trampoline, whose second line execs ``"<old>/bin/python"``; only those two
    lines of a regular ``#!`` file are rewritten. In Windows' ``Scripts/``, ``<name>.exe``
    is distlib's launcher: a stub, one ``#!<interpreter>`` line (quoted when the path holds
    a space) and a zip archive that the launcher and Python both locate from the end of the
    file, so only that line is rewritten and the stub and archive stay byte for byte; the
    interpreter is matched there as a Windows path (any case, either slash). Each rewrite
    goes through a temporary file and an atomic replace that keeps its mode; links stay.
    """
    windows = sys.platform == "win32" if windows is None else windows
    for folder in (os.path.join(root, "bin"), os.path.join(root, "Scripts")):
        try:
            names = sorted(os.listdir(folder))
        except OSError:
            continue
        for name in names:
            path = os.path.join(folder, name)
            if os.path.islink(path) or not os.path.isfile(path):
                continue
            with open(path, "rb") as fh:
                data = fh.read()
            moved = (_moved_launcher(data, old, new, windows) if name.lower().endswith(".exe")
                     else _moved_script(data, old, new, windows))
            if moved is None:
                continue
            temporary = path + ".retarget"
            with open(temporary, "wb") as fh:
                fh.write(moved)
            os.chmod(temporary, os.stat(path).st_mode)
            os.replace(temporary, path)


def _moved_script(data, old, new, windows):
    """A ``#!`` text script naming an interpreter under ``old``, rewritten to ``new``; ``None`` when it names none."""
    if not data.startswith(b"#!"):
        return None
    if windows:
        end = min([at for at in (data.find(b"\r"), data.find(b"\n")) if at >= 0] or [len(data)])
        return _moved_line(data, 0, end, old, new, windows)
    old_bytes, new_bytes = old.encode(), new.encode()
    first, sep, rest = data.partition(b"\n")
    second, sep2, tail = rest.partition(b"\n")
    if old_bytes not in first + second:
        return None
    return first.replace(old_bytes, new_bytes) + sep + second.replace(old_bytes, new_bytes) + sep2 + tail


def _moved_launcher(data, old, new, windows):
    """distlib's Windows launcher naming an interpreter under ``old``, its ``#!`` line rewritten; else ``None``.

    The launcher finds its line as Python finds the archive: the end-of-central-directory
    record from the end, the archive's start from its sizes, the last ``#!`` before it.
    """
    import struct

    eocd = data.rfind(b"PK\x05\x06", max(0, len(data) - 65557))
    if eocd < 0 or len(data) < eocd + 22:
        return None
    size, offset = struct.unpack("<II", data[eocd + 12:eocd + 20])
    archive = eocd - size - offset
    if archive <= 0 or data[archive:archive + 4] != b"PK\x03\x04":
        return None
    start = data.rfind(b"#!", 0, archive)
    ends = [at for at in (data.find(b"\r", start, archive), data.find(b"\n", start, archive)) if at >= 0]
    return _moved_line(data, start, min(ends), old, new, windows) if start >= 0 and ends else None


def _moved_line(data, start, end, old, new, windows):
    """``data`` with the ``#!`` line at ``start:end`` naming the same interpreter under ``new``, or ``None``."""
    try:
        line = data[start + 2:end].decode("utf-8")
    except UnicodeDecodeError:
        return None
    quoted = line.startswith('"')
    if quoted:
        close = line.find('"', 1)
        if close < 0:
            return None
        interpreter, rest = line[1:close], line[close + 1:]
    else:
        interpreter, sep, rest = line.partition(" ")
        rest = sep + rest
    if windows:
        import ntpath

        have, base = ntpath.normpath(interpreter), ntpath.normpath(old)
        if not ntpath.normcase(have).startswith(ntpath.normcase(base) + "\\"):
            return None
        target = ntpath.normpath(new) + have[len(base):]
    elif interpreter.startswith(old + "/"):
        target = new + interpreter[len(old):]
    else:
        return None
    named = '"%s"' % target if quoted or " " in target else target
    return data[:start] + b"#!" + named.encode("utf-8") + rest.encode("utf-8") + data[end:]


def _restore_environment(helper_dir, handoff):
    """A return puts the previous generation's own packages back where it runs them.

    By identity, as the selection moved them: the prepared generation (by its token) goes
    back to its own name, anything else at the fixed path is kept aside, and the serving
    packages return. A bare interpreter's first private environment leaves the fixed path
    empty: that interpreter's own packages serve again.
    """
    environment = handoff.get("environment")
    if not environment or environment.get("state") not in ("selecting", "selected"):
        return
    current, generation, previous = environment["current"], environment["generation"], environment["previous"]
    serving, prepared = environment["identity"]["current"], environment["identity"]["generation"]
    try:
        if _is_prepared(current, prepared) and not os.path.lexists(generation):
            os.rename(current, generation)
        if _path_identity(current) not in (None, serving):
            _set_aside(helper_dir, handoff, current)
        if serving and _path_identity(previous) == serving and not os.path.lexists(current):
            os.rename(previous, current)
    except OSError as exc:  # refused like a selection's rename: what is where stays recorded
        environment["restore_error"] = "%s: %s" % (type(exc).__name__, exc)
    environment["state"] = "restored" if _path_identity(current) == serving else "unrestored"
    write_handoff(helper_dir, handoff)


def runtime_identity():
    """The interpreter and SQLite library this process runs: what an old generation needs to find again."""
    import sqlite3

    return {"executable": os.path.realpath(sys.executable), "version": sys.version.split()[0],
            "sqlite": sqlite3.sqlite_version}


def _missing_objects(root, rows):
    """Old-side blobs Git no longer has: the previous composition could not be written back."""
    shas = sorted({sha for row in rows for sha in (_file_target(row, "old")[0], _index_side(row, "old")[0]) if sha})
    if not shas:
        return []
    out = _git(root, "cat-file", "--batch-check", data=("\n".join(shas) + "\n").encode())[1]
    return [line.split()[0] for line in out.decode("utf-8", "replace").splitlines() if line.endswith(" missing")]


def data_census(data_dir):
    """Stat facts of the stores the previous generation's boot reads first, as this process finds them.

    ``settings.json``, the state and migration records and every top-level SQLite
    store of ``state/`` with its WAL (an empty or absent WAL is no write). Size,
    mtime and inode change with every write through the normal file APIs, atomic
    replaces included; a read leaves them. Taken before the successor's first
    imports, compared before a return: equal means the successor wrote none of them.
    """
    state = os.path.join(data_dir, "state")
    try:
        stores = sorted(name for name in os.listdir(state) if name.endswith(".sqlite"))
    except OSError:
        stores = []
    names = ["settings.json", "state/migrations.json", "state/state.json", "state/state.last_good.json"]
    names += ["state/%s%s" % (name, suffix) for name in stores for suffix in ("", "-wal")]
    facts = {}
    for name in names:
        try:
            st = os.stat(os.path.join(data_dir, name))
        except FileNotFoundError:
            facts[name] = None
            continue
        facts[name] = None if name.endswith("-wal") and st.st_size == 0 else [st.st_size, st.st_mtime_ns, st.st_ino]
    return facts


def _census_changes(handoff):
    """The first-read stores the successor changed since its first entry (``["census_missing"]`` without one)."""
    before = handoff.get("census")
    if not isinstance(before, dict):
        return ["census_missing"]
    now = data_census(handoff.get("data_dir") or "")
    return sorted(name for name in set(before) | set(now) if before.get(name) != now.get(name))


_SIDE_KEYS = ("", "_index", "_file")


def _passive_presentation(rows, web_root):
    """``""`` when every switch path is a passive stylesheet, else the first path that is not.

    Passive means: inside the gateway's served static root (``web_root``, recorded at
    authorization from ``server_web.resolve_web_dir``), served as ``text/css`` (the media
    type the static route gives that name), and a regular non-executable blob on every
    side that has it. Everything else the generation runs (Python, scripts, markup,
    images that can carry script, prompts, skills, configuration, the launcher, the
    dependency input) is then byte-identical to the previous composition.
    """
    import mimetypes

    if not web_root:
        return "web_root_unknown"
    prefix = web_root.rstrip("/") + "/"
    for row in rows:
        path = row["path"]
        if not path.startswith(prefix) or mimetypes.guess_type(path)[0] != "text/css":
            return "not_passive:%s" % path
        if any(row.get(side + key + "_sha") is not None and row.get(side + key + "_mode") != "100644"
               for side in ("old", "new") for key in _SIDE_KEYS):
            return "not_regular:%s" % path
    return ""


def owned_engine(data_dir):
    """``"absent"`` only when this data root holds no owned engine home at all, else ``"present"``/``"unreadable"``.

    The home is ``<data>/claudexor`` (``claudexor_daemon.owned_config_dir``), created only
    when an owned engine is provisioned or spawned. Only its observed absence says no
    engine's forward-only floor exists here; a present home (its descriptor's stat
    included, changed or not) proves nothing about that floor, so it is never evidence.
    """
    if not data_dir or not os.path.isdir(data_dir):
        return "unreadable"
    try:
        os.lstat(os.path.join(data_dir, "claudexor"))
    except FileNotFoundError:
        return "absent"
    except OSError:
        return "unreadable"
    return "present"


def _composition(root):
    """The checkout's index tree and working-files tree, or ``None`` when either cannot be measured.

    Measured as preparation's snapshot measures the layers it records (``update_prepare.snapshot`` says
    why): a byte copy of the real index with its own time, written whole (``core.splitIndex`` off), gives
    the index tree, then ``add -A`` (untracked included, ignored files excluded) gives the files tree.
    The real index is only copied.
    """
    rc, path = _git(root, "rev-parse", "--git-path", "index", check=False)
    if rc != 0:
        return None
    with tempfile.TemporaryDirectory(prefix=_TMP_PREFIX) as scratch:
        index = os.path.join(scratch, "index")
        trees = []
        steps = [("write-tree",), ("add", "-A", "--", "."), ("write-tree",)]
        try:
            with open(os.path.join(root, path.decode("utf-8", "surrogateescape").strip()), "rb") as source, \
                    open(index, "wb") as sink:
                held = os.fstat(source.fileno())  # the bytes and the time of the one file read
                shutil.copyfileobj(source, sink)
            os.utime(index, ns=(held.st_atime_ns, held.st_mtime_ns))
        except FileNotFoundError:
            steps.insert(0, ("read-tree", "--empty"))
        except OSError:
            return None
        for step in steps:
            rc, out = _git(root, "-c", "core.splitIndex=false", *step, check=False, index=index)
            if rc != 0:
                return None  # an unmerged entry, an unreadable file: not measured, so never the same
            if step[0] == "write-tree":
                trees.append(out.decode().strip())
    return {"index": trees[0], "files": trees[1]}


def _composition_drift(root, layers):
    """Paths where the checkout no longer holds the switched composition (its files, then its index).

    The recorded ``new_files``/``new_index`` trees are the candidate with the owner's committed,
    staged, unstaged and untracked layers carried, so those layers never count; anything else does.
    """
    now = _composition(root)
    if now is None:
        return ["composition_unread"]
    drift = []
    for want, have in ((layers.get("new_files"), now["files"]), (layers.get("new_index"), now["index"])):
        if not want:
            return ["composition_unrecorded"]
        if want != have:
            rc, out = _git(root, "diff-tree", "-r", "-z", "--name-only", want, have, check=False)
            names = [name for name in out.decode("utf-8", "surrogateescape").split("\0") if name] if rc == 0 else []
            drift += [name for name in names or [have] if name not in drift]
    return drift


def _previous_environment_kept(handoff):
    """The previous generation's own installed dependencies are still where it can run them again.

    By the identity recorded at authorization; ``None`` there (a bare interpreter's first private
    environment) moved nothing: that interpreter's own packages were never touched.
    """
    environment = handoff.get("environment") or {}
    previous, identity = environment.get("previous"), environment.get("identity")
    if environment.get("state") != "selected" or not previous:
        return False
    if not isinstance(identity, dict):  # recorded before identities were: the name alone
        return os.path.isdir(previous)
    return identity.get("current") is None or _path_identity(previous) == identity.get("current")


def qualify_return(root, handoff):
    """May the previous generation run on what is here NOW? ``(outcome, reasons)``.

    The boundary is this helper's own durable entry record (``_update_entry``), written
    under the lock before an entry of the switched tree imports any body module; an entry
    that cannot make the first one refuses its boot (``main``). Before it nothing of the
    new tree ran outside the scratch import probe. After it, data
    owners may have written, so the outcome is ``unknown`` (recovery, data and local
    work kept) unless the switch moved passive stylesheets only (``_passive_presentation``):
    then the successor ran byte-identical code, so every record it wrote has a form the
    previous generation's own writers produce and its own readers accept, provided the
    root held no owned engine before the switch and holds none now (``owned_engine``:
    no forward-only engine floor exists to have moved) and no environment was switched
    in. Either way ``compatible`` (no reasons) also needs: no Panic; the previous objects
    present; the same interpreter and SQLite, and its own dependencies (an Update that
    changed the lock keeps the previous environment beside the new one); an unchanged
    engine pin; the whole checkout still the switched composition, files and index
    (``_composition_drift``: an edit anywhere since, not only at a switch path, is later
    work); no changed launcher that ran. For any other successor, what it changed of the
    stores ``data_census`` watches is named for recovery. Panic: ``refused``.
    """
    if installation_panic(handoff.get("data_dir") or ""):
        return "refused", ["panic_stop"]
    rows, quality = handoff["switch"], handoff.get("qualification") or {}
    entries = handoff.get("entries") or []
    reasons = []
    passive = _passive_presentation(rows, quality.get("web_root") or "") if entries else ""
    app = any(entry.get("kind") == "app" for entry in entries)  # another app's frozen code ran
    if entries and (passive or app):  # the boundary: this record precedes the entry's first body import
        reasons.append("successor_entered:%s" % ("app" if app else entries[0].get("kind")))
        changed = _census_changes(handoff)  # named for recovery: what the successor changed of the first reads
        if changed == ["census_missing"]:
            reasons.append("data_unobserved")
        elif changed:
            reasons.append("successor_changed_data:%s" % changed[0])
    elif entries:  # stylesheets only: the successor's data forms are the previous generation's own
        if quality.get("owned_engine") != "absent" or owned_engine(handoff.get("data_dir") or "") != "absent":
            reasons.append("engine_floor_unproven")
        if handoff.get("environment"):
            reasons.append("environment_switched")
    if quality.get("dependencies_same") is not True and not _previous_environment_kept(handoff):
        reasons.append("dependencies_changed")
    if quality.get("runtime") != runtime_identity():
        reasons.append("runtime_changed")
    if quality.get("engine_pin_same") is not True:
        reasons.append("engine_pin_changed")
    if any(entry.get("kind") == "launcher" for entry in entries) and any(
            row["path"].startswith(_LAUNCHER_PATHS) for row in rows):
        reasons.append("successor_launcher_ran")
    if _head(root) != handoff["cand"] or _branch(root) != handoff["branch"]:
        reasons.append("checkout_moved")
    else:  # at the candidate, the whole checkout must still be the switched composition
        drift = _composition_drift(root, handoff.get("layers") or {})
        if drift:
            reasons.append(drift[0] if drift[0].startswith("composition_") else "later_edits:%s" % drift[0])
    missing = _missing_objects(root, rows)
    if missing:
        reasons.append("previous_objects_missing:%s" % missing[0])
    return ("compatible" if not reasons else "unknown"), reasons


def _probe_env(handoff, scratch):
    """An allowlisted environment: the interpreter's own search paths and packages, nothing else.

    No credential, provider, setting, test marker or repository-location variable
    crosses; HOME, temp and the data root are scratch, so no real root is reachable
    through them. The user site the serving generation imports from is named
    explicitly, because a scratch HOME would otherwise move it. The installation's
    scope marker alone crosses (no task or engine-run marker): Panic's census finds
    the probe by it where no process group of this entry holds it (the detached app
    installer runs the switch in its own session).
    """
    import site

    env = {key: os.environ[key] for key in ("PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "LANG",
                                            "LC_ALL", "LC_CTYPE", "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")
           if key in os.environ}
    env.update({key: "1" for key, value in os.environ.items() if value == "1" and key.startswith(_INSTALL_SCOPE_PREFIX)
                and len(key) == len(_INSTALL_SCOPE_PREFIX) + 32
                and all(char in "0123456789abcdef" for char in key[len(_INSTALL_SCOPE_PREFIX):])})
    tmp = os.path.join(scratch, "tmp")
    os.makedirs(tmp, exist_ok=True)
    env.update(PYTHONUSERBASE=os.environ.get("PYTHONUSERBASE") or site.getuserbase(), HOME=scratch,
               USERPROFILE=scratch, TMPDIR=tmp, TEMP=tmp, TMP=tmp, PYTHONDONTWRITEBYTECODE="1",
               OUROBOROS_DATA_DIR=os.path.join(scratch, "data"))
    env[_PROBE_ENV] = str(handoff.get("id") or "probe")
    return env


def _probe_command(handoff, scratch):
    """``probe_command`` on the Python the switched environment runs once the new app is in place.

    A venv of the installed app's Python runs the staged app's after the exchange, and the probe
    comes before it: it then runs a venv in ``scratch`` that names that Python
    (``environment.staged_python``, the one the generation's clone ran its install and checks on)
    over the selected venv's own packages, linked in. Nothing of the selected venv is changed.
    """
    command = list(handoff["probe_command"])
    environment = handoff.get("environment") or {}
    python, venv = environment.get("staged_python"), environment.get("current")
    if not python or not venv:
        return command
    root = os.path.join(scratch, "venv")
    os.makedirs(os.path.join(root, "bin"))
    for name in os.listdir(venv):
        if name not in ("bin", "pyvenv.cfg"):
            os.symlink(os.path.join(venv, name), os.path.join(root, name))
    os.symlink(python, os.path.join(root, "bin", "python"))
    config = os.path.join(venv, "pyvenv.cfg")
    if os.path.isfile(config):  # where that Python lives; every other setting as the venv has it
        named = {"home": os.path.dirname(python), "executable": python}
        with open(config, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        with open(os.path.join(root, "pyvenv.cfg"), "w", encoding="utf-8") as fh:
            for line in lines:
                key, sep, _value = line.partition("=")
                key = key.strip().lower()
                fh.write("%s = %s\n" % (key, named[key]) if sep and key in named else line + "\n")
    return [os.path.join(root, "bin", "python")] + command[1:]


def _probe(root, handoff):
    """Import the switched tree in a child whose data root is scratch: ``""`` or why it failed.

    The child stays in this entry's process group (a Panic or launcher stop that ends
    the entry ends it too) and is bounded by ``_PROBE_TIMEOUT_SEC``; it imports only, on
    the Python the new generation runs (``_probe_command``).
    """
    scratch = tempfile.mkdtemp(prefix="ouroboros-switch-probe-")
    env = _probe_env(handoff, scratch)
    try:
        proc = subprocess.run(_probe_command(handoff, scratch), cwd=root, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=_PROBE_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        return "the import probe did not finish within %ds" % _PROBE_TIMEOUT_SEC
    except OSError as exc:
        return "the import probe could not start: %s" % exc
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if proc.returncode == 0:
        return ""
    tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-4:]
    return "rc=%d: %s" % (proc.returncode, " | ".join(line.strip() for line in tail) or "no output")


def _entry_kind():
    """Which process of a generation this entry is: its server, its source launcher, or another.

    A multiprocessing spawn child (the supervisor's event-queue manager, a worker) re-imports its
    parent's script after taking the parent's ``sys.argv``; its own command line says what it is.
    """
    if "--multiprocessing-fork" in (getattr(sys, "orig_argv", None) or ()):
        return ""
    name = os.path.basename(str(sys.argv[0] if sys.argv else ""))
    return {"server.py": "server", "launcher.py": "launcher"}.get(name, "")


def _update_entry(root, helper_dir, handoff):
    """An entry of a SWITCHED Update, under the lock: recorded before its imports (the data boundary).

    Servers and launchers are each recorded (by script name); any other entry, a module
    CLI or a worker, once. A server entry after an unready one (the successor did not
    start) or after a recorded return request asks ``qualify_return`` once;
    ``compatible`` writes the previous files, index and branch back and returns True: the
    caller hands the tree to a fresh process. Any other outcome is recorded and this entry
    starts the new tree. An app transition whose installer died before its exchange
    (``_app_not_exchanged``) is returned at the first entry, before anything records it.
    """
    if handoff.get("native") and _app_not_exchanged(handoff):
        if installation_panic(handoff.get("data_dir") or ""):
            raise _Stuck("the app installer stopped before its exchange and a Panic is recorded; nothing was returned")
        return _returned(root, helper_dir, handoff, "app_not_exchanged")
    entries = handoff.setdefault("entries", [])
    kind = _entry_kind()
    servers = [entry for entry in entries if entry.get("kind") == "server"]
    if kind == "server" and not handoff.get("return") and (
            handoff.get("return_requested") or (servers and not servers[-1].get("ready"))):
        trigger = "return_requested" if handoff.get("return_requested") else "successor_start_failed"
        outcome, reasons = qualify_return(root, handoff)
        if outcome == "compatible":  # no reasons
            return _returned(root, helper_dir, handoff, trigger)
        handoff["return"] = {"outcome": outcome, "trigger": trigger, "reasons": reasons, "at": _now()}
    if "census" not in handoff:
        handoff["census"] = data_census(handoff.get("data_dir") or "")  # what the successor has not touched yet
    if kind or not any(entry.get("kind") == "other" for entry in entries):  # one record says another ran
        entries.append({"kind": kind or "other", "argv0": os.path.basename(str(sys.argv[0] if sys.argv else "")),
                        "pid": os.getpid(), "ts": _now(), "ready": False})
        write_handoff(helper_dir, handoff)
    return False


def _switch(root, helper_dir, handoff):
    """Files, index, branch CAS, environment and probe; True when a fresh process must take the tree."""
    rows = handoff["switch"]
    head = _head(root)
    if head not in (handoff["old"], handoff["cand"]) or _branch(root) != handoff["branch"]:
        raise _Stuck("HEAD %s on %s is neither the recorded base nor the candidate" % (
            head[:12], _branch(root) or "detached"))
    tracked = _tracks_filemode(root)
    files, index = _states(root, rows, tracked), _index_states(root, rows)
    foreign = _foreign_paths(rows, files, index, tracked)  # found BEFORE this attempt writes anything
    if foreign and head != handoff["cand"]:
        return _return_to_old(root, helper_dir, handoff, "foreign_content:%s" % foreign[0], files)
    if foreign:
        handoff["later_edit_kept"] = foreign  # the branch already moved: this is work on the adopted tree
    _converge(root, rows, "new", files, tracked, keep=foreign)
    _fail_point("after_files")
    if head == handoff["old"]:
        _set_index(root, rows, "new", index)
        _fail_point("before_ref")
        rc, _out = _git(root, "update-ref", "-m", "body adoption %s" % handoff.get("id", ""),
                        "refs/heads/%s" % handoff["branch"], handoff["cand"], handoff["old"], check=False)
        if rc != 0:
            raise _Stuck("the branch moved during the switch; the compare-and-swap was refused")
    _fail_point("after_ref")
    command = handoff.get("deps_command")
    if command and os.environ.get("OUROBOROS_MANAGED_BY_LAUNCHER") != "1":
        # No launcher owns dependencies here; the old readers are gone, so sync now.
        try:
            rc = subprocess.run(command, cwd=root, timeout=_DEPS_TIMEOUT_SEC).returncode
        except (OSError, subprocess.TimeoutExpired):
            rc = -1
        if rc != 0:
            handoff["interpreter_state"] = "may_have_changed"  # Git returns; installed packages do not
            return _return_to_old(root, helper_dir, handoff, "dependency_sync_failed:rc=%d" % rc, files)
    if handoff.get("kind") == "update":
        try:
            _select_environment(helper_dir, handoff)
        except (_Stuck, OSError) as exc:
            # Before the probe and any entry: nothing of the new tree ran, so files, index, branch and packages
            # go back together rather than leave a switched tree no start can finish (exit 3 at every start).
            # A refused rename is recorded as the system said it, never classified further: an error code alone
            # does not name what holds the directory.
            decision = {"trigger": "environment_selection_failed", "detail": str(exc)[:2000], "at": _now(),
                        **({"error": _os_error(exc)} if isinstance(exc, OSError) else {})}
            if installation_panic(handoff.get("data_dir") or ""):
                handoff["return"] = {"outcome": "refused", "reasons": ["panic_stop"], **decision}
                raise _Stuck("the environment selection failed and a Panic is recorded; nothing was returned")
            # Reaching "returned" proves the packages are back too (``_return_to_old``): the candidate may try again.
            handoff["return"] = {"outcome": "returned", "reasons": [], "retry": True, **decision}
            return _return_to_old(root, helper_dir, handoff, "environment_selection_failed", files, phase="returned")
    if handoff.get("kind") == "update" and handoff.get("probe_command"):
        failure = _probe(root, handoff)
        _fail_point("after_probe")
        if failure and installation_panic(handoff.get("data_dir") or ""):
            handoff["return"] = {"outcome": "refused", "trigger": "import_probe_failed", "reasons": ["panic_stop"],
                                 "detail": failure[:2000], "at": _now()}
            record(helper_dir, handoff, "switched", "import probe failed after a Panic: nothing automatic follows")
            raise _Stuck("the new tree failed its import probe and a Panic is recorded; nothing was returned")
        if failure:
            # Nothing of the new tree ran outside the scratch probe: positive evidence for this return.
            return _returned(root, helper_dir, handoff, "import_probe_failed", files, detail=failure[:2000])
    record(helper_dir, handoff, "switched", "files, index and branch are at the candidate")
    return True


def _attest_loaded(root, handoff):
    """This process is about to import a ``switched`` tree: record what it actually finds.

    Left on ``sys`` (in-process, never inherited through the environment) for
    ``body_adoption.finalize_on_boot``. ``sha`` is set only when the checkout is at
    the candidate; ``later_edits`` names switch paths that hold other bytes.
    """
    try:
        rows, tracked = handoff["switch"], _tracks_filemode(root)
        files = _states(root, rows, tracked)
        fact = {"adoption_id": handoff.get("id"),
                "sha": handoff["cand"] if _head(root) == handoff["cand"] else "",
                "later_edits": sorted(row["path"] for row in rows
                                      if files[row["path"]] != _side(row, "new", tracked))}
    except Exception as exc:
        fact = {"adoption_id": handoff.get("id"), "sha": "", "error": "%s: %s" % (type(exc).__name__, exc)}
    sys._ouroboros_body_generation = fact


def _refuse_boot(root, helper_dir, message):
    """Stop this process before any body import; name the record and the one file that ends the hold."""
    try:
        pointer = os.path.normpath(pointer_path(root))
    except Exception:
        pointer = "%s in this checkout's Git dir" % POINTER_NAME
    sys.stderr.write("[body_switch] %s. Record: %s. After deciding what the tree should be, removing %s "
                     "lets it start as it is.\n" % (message, os.path.join(helper_dir, HANDOFF_NAME), pointer))
    sys.stderr.flush()
    sys.exit(STUCK_EXIT_CODE)


def main(serving_root):
    if os.environ.get(_PROBE_ENV):
        return  # this process IS the import probe of a switch that holds the lock
    helper_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.realpath(serving_root)
    try:
        handoff = read_handoff(helper_dir)
        phase = handoff.get("phase")
        elsewhere = os.path.realpath(str(handoff.get("repo_dir") or "")) != root
    except (OSError, ValueError, AttributeError) as exc:
        # The pointer says a transition exists; without its record nothing proves the tree coherent.
        _refuse_boot(root, helper_dir, "a body adoption is pending but its record is unreadable (%s)" % exc)
    if phase == "stuck" or (elsewhere and phase == "switching"):
        _refuse_boot(root, helper_dir, "an interrupted body adoption needs a decision before this tree can be imported")
    if elsewhere:
        if phase == "armed":  # recorded for another path of this checkout: nothing was written
            record(helper_dir, handoff, "abandoned", "checkout_path_changed")
            _remove_pointer(root)
        return
    _ENTERED_ON[:] = [str((handoff.get("environment") or {}).get("state") or "")]
    if phase == "switched" and handoff.get("kind") == "update":
        handover = False
        with _locked(helper_dir):
            handoff = read_handoff(helper_dir)
            if handoff.get("phase") == "switched":
                # What the record proves now: the attempt below changes this copy before it writes.
                entered = any(entry.get("kind") != "app" for entry in handoff.get("entries") or ())
                try:
                    handover = _update_entry(root, helper_dir, handoff)
                except _Stuck as exc:
                    record(helper_dir, handoff, "stuck", str(exc))
                    _refuse_boot(root, helper_dir, "the update's return stopped: %s" % exc)
                except Exception as exc:
                    if (handoff.get("return") or {}).get("outcome") == "returned":
                        # A return interrupted midway leaves old and new bytes: never import that tree.
                        record(helper_dir, handoff, "stuck", "return interrupted: %s: %s" % (type(exc).__name__, exc))
                        _refuse_boot(root, helper_dir, "the update's return was interrupted (%s: %s)" % (
                            type(exc).__name__, exc))
                    if not entered:
                        # Every return rests on this record (``qualify_return``, ``_app_not_exchanged``): imports with
                        # no entry on disk could write data that later reads as "nothing of the new tree ran".
                        _refuse_boot(root, helper_dir, "the update's entry could not be recorded (%s: %s), so nothing "
                                     "of it was imported; the next start retries it" % (type(exc).__name__, exc))
                    # An entry already on record holds that boundary; this one adds nothing a return reads.
                    sys.stderr.write("[body_switch] update entry not recorded: %s: %s\n" % (type(exc).__name__, exc))
            phase = handoff.get("phase")
        if handover:
            _handover(handoff)
    if phase == "switched":
        _attest_loaded(root, handoff)
    if phase not in ("armed", "switching"):
        return
    handover = False
    with _locked(helper_dir):
        handoff = read_handoff(helper_dir)
        phase = handoff.get("phase")
        if phase == "switched":  # another entry finished it while this one waited for the lock
            handover = True
        elif phase in ("armed", "switching"):
            handover = _hook_switch(root, helper_dir, handoff, phase)
    if handover:
        _handover(handoff)


def _hook_switch(root, helper_dir, handoff, phase):
    """An entry's own switch of an armed or interrupted transition, under the lock; True: hand over."""
    try:
        native = handoff.get("native")
        if phase == "armed":
            reason = _armed_refusal(root, handoff)
            if native and not reason:
                # The app installer switches this transition once the old app is gone; never an entry.
                if not process_gone(native.get("installer")):
                    return False
                reason = "installer_gone: nothing was switched"
            if reason:
                record(helper_dir, handoff, "abandoned", reason)
                _remove_pointer(root)
                return False
        if native:  # the installer died mid-switch, before any app exchange: back by content
            return _returned(root, helper_dir, handoff, "native_switch_interrupted")
        handoff["attempts"] = int(handoff.get("attempts") or 0) + 1
        record(helper_dir, handoff, "switching", "attempt %d" % handoff["attempts"])
        return _switch(root, helper_dir, handoff)
    except _Stuck as exc:
        record(helper_dir, handoff, "stuck", str(exc))
        _refuse_boot(root, helper_dir, "body adoption stopped: %s. Nothing further was changed" % exc)
    except Exception as exc:
        # A failure that may pass (a busy index, a full disk): the phase is kept, so the next
        # entry resumes by content. Either way this process must not go on to import the tree.
        record(helper_dir, handoff, handoff.get("phase") or "switching",
               "interrupted: %s: %s" % (type(exc).__name__, exc))
        _refuse_boot(root, helper_dir, "body adoption was interrupted (%s: %s); the next start retries it" % (
            type(exc).__name__, exc))


# --------------------------------------------------------------------------- #
# The app installer's entries (``app_install.py``), each under the caller's ``_locked``
# --------------------------------------------------------------------------- #
def native_switch(serving_root, helper_dir):
    """Switch an ARMED app transition for the installer, without a hand-over: ``(phase, detail)``.

    The installer calls this once the old app's launcher and server are provably gone and
    it holds the launcher's single-instance lock, before the new app opens. Files, index,
    branch, the generation environment and the import probe (the new app's interpreter)
    are the hook's own steps; a failed probe or any failure that wrote returns the tree
    by content at once, so the previous app can reopen on its own composition.
    """
    root = os.path.realpath(serving_root)
    handoff = read_handoff(helper_dir)
    if handoff.get("phase") != "armed" or not handoff.get("native"):
        return str(handoff.get("phase") or ""), "the transition is not armed for the app installer"
    reason = _armed_refusal(root, handoff)
    if reason:
        return native_abandon(root, helper_dir, reason), reason
    handoff["attempts"] = int(handoff.get("attempts") or 0) + 1
    record(helper_dir, handoff, "switching", "attempt %d by the app installer" % handoff["attempts"])
    try:
        _switch(root, helper_dir, handoff)
    except _Stuck as exc:
        record(helper_dir, handoff, "stuck", str(exc))
        return "stuck", str(exc)
    except Exception as exc:
        detail = "%s: %s" % (type(exc).__name__, exc)
        try:
            native_return(root, helper_dir, "native_switch_failed: " + detail, handoff)
        except Exception as again:
            record(helper_dir, handoff, "stuck", "switch failed (%s), its return too: %s" % (detail, again))
            return "stuck", detail
    return str(handoff.get("phase") or ""), str((handoff.get("return") or {}).get("detail") or "")


def native_return(serving_root, helper_dir, reason, handoff=None):
    """Write the previous composition back after the installer's switch (its exchange failed, or the
    switch itself did): nothing of the new tree ran outside the scratch probe. Returns the phase."""
    root = os.path.realpath(serving_root)
    handoff = handoff or read_handoff(helper_dir)
    if handoff.get("phase") not in ("switching", "switched"):
        return str(handoff.get("phase") or "")
    _returned(root, helper_dir, handoff, reason.split(":", 1)[0], reason=reason, detail=reason[:2000])
    return "returned"


def native_abandon(serving_root, helper_dir, reason):
    """The installer ends an ARMED transition it will not switch; nothing was written. Returns the phase."""
    handoff = read_handoff(helper_dir)
    if handoff.get("phase") != "armed":
        return str(handoff.get("phase") or "")
    record(helper_dir, handoff, "abandoned", reason)
    _remove_pointer(os.path.realpath(serving_root))
    return "abandoned"


def native_settle_backup(native):
    """After the exchange the previous app sits at the staged path; it becomes the backup beside it.

    True once the previous app (by identity) is at the backup path. An occupied backup path
    is never overwritten and another bundle at the staged path is never moved.
    """
    staged, backup, previous = native.get("staged") or "", native.get("backup") or "", native.get("previous_identity")
    if not previous or same_bundle(backup, previous):
        return bool(previous)
    if not same_bundle(staged, previous) or os.path.lexists(backup):
        return False
    os.rename(staged, backup)
    return True


def native_journal(helper_dir, state, entry=False, **facts):
    """One step of the app transition into its record (``native.state``, ``native.steps``).

    ``entry``: the new app may run from here on, so its frozen code is recorded as an
    entry of the switched tree BEFORE the step that makes that possible.
    """
    handoff = read_handoff(helper_dir)
    native = handoff.setdefault("native", {})
    native["state"] = state
    native.setdefault("steps", []).append({"ts": _now(), "pid": os.getpid(), "state": state, **facts})
    if entry:
        handoff.setdefault("entries", []).append({"kind": "app", "argv0": os.path.basename(native.get("installed")
                                                                                            or ""),
                                                  "pid": os.getpid(), "ts": _now(), "ready": False})
    write_handoff(helper_dir, handoff)
    return handoff


if __name__ == "__ouroboros_body_switch__":
    main(SERVING_ROOT)  # noqa: F821 -- injected by the package-init hook
