"""The cost stand: each size of tests/_scale_root.py's root built once per run, restored where it was built.

Three modules measure the same roots (tests/test_cost_contract.py, tests/test_cost_live_set.py, tests/test_stand_counts.py).
A k=30 root takes 80 s and more to build, so each module's own build under the 300 s per-test bound was what that bound killed
under load. Here a root is built once, under its own child bound (a stand module's timer covers the test body, not its
fixtures: ``STAND_MARKS``), and every measurement finds it pristine.

A root is no tests/_worlds.py world. It records absolute paths (delegated checkouts, custody) and file identities (the archive's
stamp, the custody chain's cursor, each child drive's boot stamp), so a copy, even at the same path, sends the next boot down the
history listing the contract exists to forbid. A consumer therefore holds the stand's one path for as long as its fixture lives,
and the next one finds the root put back in place (``_restore``): beside it are kept the built bytes (``kept/bytes``, a clone) and
a hard link to each built file (``kept/inodes``), which keeps an inode alive after a rotation or an atomic write took its name.
The modules share that path as a host resource; a consumer that still meets it in use (a focused run that spreads a module over
two workers) builds a private root, as if the stand did not exist, and waits for nobody. The owner's release takes the stand with
the run.
"""
from __future__ import annotations

import contextlib
import json
import os
import pathlib
import shutil
import stat
import sys
import tempfile
from typing import Any, Dict, Iterator

import pytest

from ouroboros.platform_layer import file_lock_exclusive_nb
from ouroboros.test_evidence import remove_scratch_tree
from tests._scale_root import worktree_root
from tests._worlds import clone, run_folder

REPO = pathlib.Path(__file__).resolve().parents[1]
CHILD_TIMEOUT_SEC = 900  # a transport bound for one child, not a judgment about the boot
# The stand's one path is the host resource the three modules share (``serial("cost-stand")``: one worker at a time restores
# and measures it); a build answers to ``child``'s bound, so the per-test timer leaves fixtures alone.
STAND_MARKS = [pytest.mark.serial("cost-stand"), pytest.mark.timeout(func_only=True)]


def child(module: str, args: list, env: Dict[str, str], log: pathlib.Path) -> None:
    """One fresh interpreter; a timeout or an interrupt stops its whole process tree (tests/group_run.py)."""
    from tests.group_run import run_in_group

    done = run_in_group([sys.executable, "-m", module, *args], timeout=CHILD_TIMEOUT_SEC, cwd=REPO, env=env)
    log.write_text(done.stdout + done.stderr, encoding="utf-8")
    if done.returncode != 0:
        raise RuntimeError(f"{module} exited {done.returncode}:\n{(done.stdout + done.stderr)[-6000:]}")


def _environment(home: pathlib.Path, journal: str) -> Dict[str, str]:
    root = home / "data"
    # The delegated copies live under the worktree root the build and the boot both name (tests/_scale_root.py).
    return {**os.environ, "OUROBOROS_DATA_DIR": str(root), "OUROBOROS_SETTINGS_PATH": str(root / "settings.json"),
            "OUROBOROS_REPO_DIR": str(home / "repo"), "OUROBOROS_HOST_SERVICE_PORT": "0",
            "OUROBOROS_SUBAGENT_WORKTREE_ROOT": str(worktree_root(root)), "COST_STORE_JOURNAL": journal}


def build(home: pathlib.Path, k: int, journal: str = "") -> None:
    """Lay the k-root out in one interpreter."""
    (home / "data").mkdir(parents=True)
    (home / "repo").mkdir()
    child("tests._scale_root", [str(home / "data"), str(k), str(home / "facts.json")], _environment(home, journal),
          home / "build.log")


def drive(home: pathlib.Path, **plan: Any) -> Dict[str, Any]:
    """Boot and query the root in a fresh interpreter (cold); ``plan``: what the driver measures instead of the contract's
    boundaries (the live-set axis, tests/test_cost_live_set.py; the stand, tests/test_stand_counts.py)."""
    root = home / "data"
    facts = json.loads((home / "facts.json").read_text(encoding="utf-8"))
    spec_path, out_path = home / "spec.json", home / "measured.json"
    spec_path.write_text(json.dumps({"root": str(root), "facts": facts, **plan}), encoding="utf-8")
    child("tests._cost_driver", [str(spec_path), str(out_path)], _environment(home, str(plan.get("journal") or "")),
          home / "drive.log")
    measured = json.loads(out_path.read_text(encoding="utf-8"))
    if measured["errors"]:  # a broken measurement is not an expected performance assertion
        raise RuntimeError(f"k={facts['k']}: the driver failed:\n" + "\n".join(measured["errors"]))
    measured["root"] = str(root)
    return measured


def _remove(path: pathlib.Path) -> None:
    """One entry gone. Windows refuses to delete a read-only file, and Git writes its objects read-only, so there the
    bit goes first."""
    def remove() -> None:
        path.rmdir() if path.is_dir() and not path.is_symlink() else path.unlink(missing_ok=True)
    try:
        remove()
    except PermissionError:
        if os.name != "nt":
            raise
        os.chmod(path, stat.S_IWRITE)
        remove()


def _restore_file(source: pathlib.Path, inode: pathlib.Path, target: pathlib.Path) -> None:
    """One file back: its inode first (a rotation moved it away, an atomic write replaced it), then its bytes and time."""
    if source.is_symlink():
        if not target.is_symlink() or os.readlink(target) != os.readlink(source):
            _remove(target)
            os.symlink(os.readlink(source), target)
        return
    held = os.lstat(inode)
    if not os.path.lexists(target) or os.lstat(target).st_ino != held.st_ino:
        _remove(target)
        os.link(inode, target)
    seen = os.lstat(source)
    if (held.st_size, held.st_mtime_ns) != (seen.st_size, seen.st_mtime_ns):
        if inode.read_bytes() != source.read_bytes():  # Git freshens a read-only object's time and nothing else
            with source.open("rb") as reading, inode.open("r+b") as writing:
                shutil.copyfileobj(reading, writing)
                writing.truncate()
        os.utime(inode, ns=(seen.st_atime_ns, seen.st_mtime_ns))


def _restore(kept: pathlib.Path, home: pathlib.Path) -> None:
    """Put the built root back in ``home`` in place: what a consumer added goes, each file gets its inode, bytes and time
    back, a folder its time (the archive's stamp reads it), and nothing else is touched."""
    built = kept / "bytes"
    for directory, folders, files in os.walk(home, topdown=False):
        for name in (*files, *folders):
            path = pathlib.Path(directory, name)
            if not os.path.lexists(built / path.relative_to(home)):
                _remove(path)
    for directory, folders, files in os.walk(built, topdown=False):
        relative = pathlib.Path(directory).relative_to(built)
        if os.path.lexists(home / relative) and (os.path.islink(home / relative) or not (home / relative).is_dir()):
            _remove(home / relative)  # a consumer put a file or a link where the stand has a folder
        (home / relative).mkdir(parents=True, exist_ok=True)
        for name in (*files, *(name for name in folders if os.path.islink(pathlib.Path(directory, name)))):
            _restore_file(pathlib.Path(directory, name), kept / "inodes" / relative / name, home / relative / name)
        times = os.stat(directory)
        os.utime(home / relative, ns=(times.st_atime_ns, times.st_mtime_ns))


@contextlib.contextmanager
def stand(k: int, journal: str = "") -> Iterator[pathlib.Path]:
    """The stand of ``k`` and ``journal`` at its one path, pristine and this consumer's until it leaves."""
    folder = run_folder("stands") / f"k{k:03d}-{journal}"
    folder.mkdir(exist_ok=True)
    lock = os.open(folder / "lock", os.O_CREAT | os.O_RDWR, 0o600)  # the kernel lets go of a killed holder
    try:
        try:
            file_lock_exclusive_nb(lock)
        except OSError:
            home = pathlib.Path(tempfile.mkdtemp(dir=folder.parent)) / "home"  # as deep: a boot's stats follow the depth
            build(home, k, journal)
            yield home
            return
        home, kept = folder / "home", folder / "kept"
        if kept.is_dir():
            _restore(kept, home)
        else:
            if os.path.lexists(home):
                remove_scratch_tree(home)  # a build that was killed
            build(home, k, journal)
            staging = pathlib.Path(tempfile.mkdtemp(dir=folder))
            clone(home, staging / "bytes")
            shutil.copytree(home, staging / "inodes", symlinks=True, copy_function=os.link)
            os.replace(staging, kept)  # published whole
        yield home
    finally:
        os.close(lock)


@contextlib.contextmanager
def measuring(k: int, **plan: Any) -> Iterator[Dict[str, Any]]:
    """What the driver measured over the stand of ``k``; the root stays the caller's while it runs tests on it."""
    with stand(k, str(plan.get("journal") or "")) as home:
        yield drive(home, **plan)
