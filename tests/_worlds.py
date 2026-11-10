"""Worlds: what a test only reads or starts from, built once per run and cloned per test.

A world is a folder: a Git checkout, a copy of the source, a small installation. Laying one out
is real work (13 git processes for an installed body) that says nothing about the test using it,
so the run's first consumer builds it, inside its own timeout and behind a lock every process of
the run shares, and each test works in a private copy: ``clone(world(key, build), tmp_path / name)``.

The template is immutable: nothing writes it once it is in place, and a build leaves no store
open that could checkpoint into it later. A copy shares no bytes a write could reach: an APFS clone
or a reflink copies on write, a plain copy is a copy, and no hardlink is ever made.

A world must not know its own path, because every copy lives somewhere else: a build that writes
it is refused. One that cannot avoid absolute paths (a linked worktree, a custody record) is
restored at its one path instead (tests/_stand_cache.py). Nor may the first consumer's state reach
it: the first test's fixtures run before the build.

Templates live beneath the owner root of the run (``OUROBOROS_TEST_TEMP_ROOT``: safe_test, the
commit gate), declared, so the owner's release takes them with the run (PERSISTENCE). Bare pytest
has no owner to share a root through: each of its processes keeps its own until it exits.
"""
from __future__ import annotations

import contextlib
import ctypes
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from typing import Callable, Iterator

_bare_root = None  # tempfile.TemporaryDirectory: its finalizer is the remover


def _run_root() -> pathlib.Path:
    global _bare_root
    owner = os.environ.get("OUROBOROS_TEST_TEMP_ROOT")
    if owner:
        return pathlib.Path(owner).resolve()
    if _bare_root is None:
        _bare_root = tempfile.TemporaryDirectory(prefix="ouroboros-worlds-")
    return pathlib.Path(_bare_root.name).resolve()


def run_folder(name: str) -> pathlib.Path:
    """``name`` directly beneath the run's root, one folder for all its processes, declared for its release."""
    from ouroboros.test_environment import declare_scratch

    folder = _run_root() / name
    folder.mkdir(exist_ok=True)
    with contextlib.suppress(FileExistsError):  # a sibling worker declared it first
        declare_scratch(folder, "process-temp", owner=f"shared test {name} of this run")
    return folder


@contextlib.contextmanager
def _exclusive(path: pathlib.Path) -> Iterator[None]:
    """Across the run's processes; a holder that dies releases it, so a killed build blocks nobody."""
    from ouroboros.platform_layer import file_lock_exclusive

    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        file_lock_exclusive(descriptor)
        yield
    finally:
        os.close(descriptor)


def _refuse_own_path(folder: pathlib.Path) -> None:
    """A copy lives elsewhere, so a path to the template inside it leads a test back to the template."""
    needle = os.fsencode(str(folder).removeprefix("/private"))  # macOS: /tmp is a link to /private/tmp
    for directory, folders, files in os.walk(folder):
        for name in (*files, *(name for name in folders if os.path.islink(os.path.join(directory, name)))):
            path = pathlib.Path(directory, name)  # a link to a folder is listed with the folders and never walked
            if needle in (os.fsencode(os.readlink(path)) if path.is_symlink() else path.read_bytes()):
                raise ValueError(f"{path.relative_to(folder)} holds the path it was built at: every copy would lead back here")


def world(key: str, build: Callable[[pathlib.Path], None]) -> pathlib.Path:
    """The template of ``key``, which ``build(folder)`` lays out in an empty folder once per run.

    A rename publishes it, so a template that exists is complete; a build that fails or is killed
    publishes nothing and the next consumer builds again. ``key`` names everything the build depends on."""
    from ouroboros import store

    folder = run_folder("worlds")
    template = folder / key
    if template.is_dir():
        return template
    with _exclusive(folder / f"{key}.lock"):
        if not template.is_dir():
            building = pathlib.Path(tempfile.mkdtemp(prefix=f"{key}.", dir=folder))
            try:
                build(building)
                store.close_connections_under(building)
                _refuse_own_path(building)
                os.replace(building, template)
            except BaseException:  # pytest-timeout's failure is one
                shutil.rmtree(building, ignore_errors=True)
                raise
    return template


def _copy_on_write(template: pathlib.Path, destination: pathlib.Path) -> bool:
    """False leaves the copy to the caller. ``clonefile`` is what ``cp -c`` calls per file; given the
    folder it clones the whole tree in one call (1,046 files: 0.02 s against 0.36 s, and no child),
    stamping only the folders it makes with the time of the call."""
    if sys.platform == "darwin":
        if ctypes.CDLL(None).clonefile(os.fsencode(template), os.fsencode(destination), 0) != 0:
            return False
        for directory, _folders, _files in os.walk(template):
            seen = os.stat(directory)
            os.utime(destination / pathlib.Path(directory).relative_to(template), ns=(seen.st_atime_ns, seen.st_mtime_ns))
        return True
    if sys.platform.startswith("linux"):
        try:
            done = subprocess.run(["cp", "-a", "--reflink=auto", str(template), str(destination)], capture_output=True)
        except OSError:
            return False
        return done.returncode == 0
    return False


def clone(template: pathlib.Path, destination: pathlib.Path) -> pathlib.Path:
    """A private copy of ``template`` at ``destination``, which must not exist. Links stay links, times stay."""
    destination = pathlib.Path(destination)
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not _copy_on_write(template, destination):
        shutil.rmtree(destination, ignore_errors=True)  # a copy that stopped half way
        shutil.copytree(template, destination, symlinks=True)
    return destination


def clone_repository(template: pathlib.Path, destination: pathlib.Path) -> pathlib.Path:
    """``clone`` of a Git checkout. The copy's inodes and ctimes differ from what its index records, and
    plumbing (`diff-files`, `diff-index`) reads that as every file changed, so one refresh makes the
    copy what the build left: one process where a fresh repository took 4 to 13."""
    destination = clone(template, destination)
    subprocess.run(["git", "update-index", "-q", "--refresh"], cwd=destination, capture_output=True, check=False)
    return destination
