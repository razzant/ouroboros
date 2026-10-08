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
"""
import contextlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import time

HANDOFF_NAME = "handoff.json"
POINTER_NAME = "ouroboros-body-adoption"
RESTART_EXIT_CODE = 42
STUCK_EXIT_CODE = 3
_TMP_PREFIX = ".body-switch-"
_CHUNK = 200
_DEPS_TIMEOUT_SEC = 900
# Test seam only (fault injection at a named boundary); production never sets it.
_FAIL_AT = os.environ.get("OUROBOROS_BODY_SWITCH_FAIL_AT", "")


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
    path = os.path.join(helper_dir, HANDOFF_NAME)
    fd, tmp = tempfile.mkstemp(prefix=HANDOFF_NAME + ".", dir=helper_dir)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(handoff, fh, indent=1, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def record(helper_dir, handoff, phase, detail=""):
    handoff["phase"] = phase
    handoff.setdefault("events", []).append({"ts": _now(), "pid": os.getpid(), "phase": phase, "detail": detail})
    write_handoff(helper_dir, handoff)


def _git(root, *args, data=None, check=True):
    env = dict(os.environ, GIT_LITERAL_PATHSPECS="1", GIT_OPTIONAL_LOCKS="0")
    for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        env.pop(key, None)
    proc = subprocess.run(["git", *args], cwd=root, env=env, input=data,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and proc.returncode != 0:
        raise RuntimeError("git %s failed rc=%d: %s" % (
            " ".join(args[:2]), proc.returncode, proc.stderr.decode("utf-8", "replace").strip()))
    return proc.returncode, proc.stdout


def git_dir(root):
    out = _git(root, "rev-parse", "--absolute-git-dir")[1].decode("utf-8", "surrogateescape").strip()
    return out


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


def _side(row, side, tracked):
    """What ``side`` ("old"/"new") wants ON DISK at this path: ``(blob id, kind)``, ``(None, None)`` = absent."""
    sha = row[side + "_sha"]
    return (None, None) if sha is None else (sha, _file_mode(row[side + "_mode"], tracked))


def _index_side(row, side):
    sha = row[side + "_sha"]
    return (None, None) if sha is None else (sha, row[side + "_mode"])


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
    ordered = sorted(rows, key=lambda row: row[side + "_sha"] is not None)
    for index, row in enumerate(ordered):
        have, want = files[row["path"]], _side(row, side, tracked)
        if have == want or row["path"] in keep or have != _side(row, other, tracked):
            continue
        _put(root, row["path"], row[side + "_mode"], row[side + "_sha"])
        if side == "new" and index == len(ordered) // 2:
            _fail_point("mid_files")


def _set_index(root, rows, side, index):
    """Move the index entries that hold the OTHER side to ``side``; a later ``git add`` stays."""
    other = "old" if side == "new" else "new"
    zero = "0" * len(next((row[key] for row in rows for key in ("old_sha", "new_sha") if row[key]), "0" * 40))
    lines = []
    for row in rows:
        if index[row["path"]] != _index_side(row, other) or _index_side(row, side) == _index_side(row, other):
            continue
        sha = row[side + "_sha"]
        entry = ("0 %s\t%s" % (zero, row["path"])) if sha is None else (
            "%s %s\t%s" % (row[side + "_mode"], sha, row["path"]))
        lines.append(entry.encode("utf-8", "surrogateescape"))
    if lines:
        _git(root, "update-index", "-z", "--index-info", data=b"\0".join(lines) + b"\0")


def _dirty(root, paths):
    out = b""
    for start in range(0, len(paths), _CHUNK):
        out += _git(root, "status", "--porcelain", "--untracked-files=all", "--", *paths[start:start + _CHUNK])[1]
    return out.decode("utf-8", "replace").strip()


def _armed_refusal(root, handoff):
    """Typed reason to abandon an ARMED handoff with the tree untouched, or ``""``."""
    if _head(root) != handoff["old"]:
        return "base_moved:%s" % _head(root)[:12]
    if _branch(root) != handoff["branch"]:
        return "branch_changed:%s" % (_branch(root) or "detached")
    if _git(root, "cat-file", "-e", "%s^{commit}" % handoff["cand"], check=False)[0] != 0:
        return "candidate_missing"
    if os.path.exists(os.path.join(handoff.get("data_dir") or "", "state", "panic_stop.flag")):
        return "panic_stop"
    dirt = _dirty(root, [row["path"] for row in handoff["switch"]])
    return ("switch_set_dirty:%s" % dirt.splitlines()[0]) if dirt else ""


@contextlib.contextmanager
def _locked(helper_dir):
    """One switcher at a time; the kernel releases the lock when the process ends or execs."""
    fh = open(os.path.join(helper_dir, "lock"), "a+")
    try:
        if sys.platform == "win32":
            import msvcrt

            deadline = time.time() + 60
            while True:
                try:
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.time() > deadline:
                        raise
                    time.sleep(0.1)
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fh.close()


def _wait_as_parent(argv):
    """Windows has no process-image replacement: run the fresh entry and stay its waited-on parent.

    The child keeps this console group, as an exec would, so the console's
    CTRL+C / CTRL+BREAK reach it directly and ITS OWN stop runs: this parent
    never kills it and never answers the interrupt itself. The child's exit
    status (a Panic, a stop, the restart code of a supervising launcher) is
    returned unchanged.
    """
    proc = subprocess.Popen(argv)
    while True:
        try:
            return proc.wait()
        except KeyboardInterrupt:
            continue


def _handover():
    """Give the switched tree to a FRESH process through the owning lifecycle."""
    sys.stdout.flush()
    sys.stderr.flush()
    if os.environ.get("OUROBOROS_MANAGED_BY_LAUNCHER") == "1":
        os._exit(RESTART_EXIT_CODE)  # the launcher's own cycle: dependencies, native host, relaunch
    argv = [sys.executable, *sys.orig_argv[1:]]
    if sys.platform == "win32":
        os._exit(_wait_as_parent(argv))
    os.execv(sys.executable, argv)


def _return_to_old(root, helper_dir, handoff, reason, files_at_start):
    """Undo OUR writes only, then abandon: the old tree is coherent again.

    Boot continues in this process only when it began on a wholly old tree (it
    parsed its entry and the hook from the old side). A process that resumed a
    half-switched tree may already hold the candidate's compiled entry, so the
    returned tree is handed to a FRESH process instead.
    """
    rows, tracked = handoff["switch"], _tracks_filemode(root)
    _converge(root, rows, "old", _states(root, rows, tracked), tracked)
    _set_index(root, rows, "old", _index_states(root, rows))  # a foreign file then reads as an unstaged edit
    if _head(root) == handoff["cand"]:
        _git(root, "update-ref", "refs/heads/%s" % handoff["branch"], handoff["old"], handoff["cand"])
    record(helper_dir, handoff, "abandoned", reason)
    _remove_pointer(root)
    if any(files_at_start[row["path"]] == _side(row, "new", tracked) != _side(row, "old", tracked) for row in rows):
        _handover()


def _switch(root, helper_dir, handoff):
    rows = handoff["switch"]
    head = _head(root)
    if head not in (handoff["old"], handoff["cand"]) or _branch(root) != handoff["branch"]:
        raise _Stuck("HEAD %s on %s is neither the recorded base nor the candidate" % (
            head[:12], _branch(root) or "detached"))
    tracked = _tracks_filemode(root)
    files, index = _states(root, rows, tracked), _index_states(root, rows)
    foreign = _foreign_paths(rows, files, index, tracked)  # found BEFORE this attempt writes anything
    if foreign and head != handoff["cand"]:
        _return_to_old(root, helper_dir, handoff, "foreign_content:%s" % foreign[0], files)
        return
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
            _return_to_old(root, helper_dir, handoff, "dependency_sync_failed:rc=%d" % rc, files)
            return
    record(helper_dir, handoff, "switched", "files, index and branch are at the candidate")
    _handover()


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
    if phase == "switched":
        _attest_loaded(root, handoff)
    if phase not in ("armed", "switching"):
        return
    with _locked(helper_dir):
        handoff = read_handoff(helper_dir)
        phase = handoff.get("phase")
        if phase == "switched":  # another entry finished it while this one waited for the lock
            _handover()
        if phase not in ("armed", "switching"):
            return
        try:
            if phase == "armed":
                reason = _armed_refusal(root, handoff)
                if reason:
                    record(helper_dir, handoff, "abandoned", reason)
                    _remove_pointer(root)
                    return
            handoff["attempts"] = int(handoff.get("attempts") or 0) + 1
            record(helper_dir, handoff, "switching", "attempt %d" % handoff["attempts"])
            _switch(root, helper_dir, handoff)
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


if __name__ == "__ouroboros_body_switch__":
    main(SERVING_ROOT)  # noqa: F821 -- injected by the package-init hook
