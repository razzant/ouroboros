"""A passed test's ``tmp_path`` goes with the test (owner 05.11); ``tests/conftest.py`` drives this.

Each test's folder used to wait for the run's end: 16-17 GiB per full pass, all of it left behind by
one marker or a stopped run. It waits now only while something still needs it (``release_passed``).
"""
from __future__ import annotations

import collections
import multiprocessing
import os
import pathlib
import types
import uuid

# The test in progress in this process: ``token`` marks every child it starts, ``children`` holds
# (Popen, started detached), ``before`` the multiprocessing children it found.
custody = types.SimpleNamespace(token="", children=[], before=frozenset(), failed=False)
counts: collections.Counter = collections.Counter()  # released, waited, censuses; merged onto the controller


def custody_prefix() -> str:
    """A test's own token: a container's spelling, so explicit child environments keep it as they keep a
    container's (``process_containment.ProcessContainer.spawn``), but no container reaps by it."""
    from ouroboros.process_containment import CONTAINMENT_ENV_PREFIX

    return CONTAINMENT_ENV_PREFIX + "TEST_"


def begin() -> None:
    custody.token = custody_prefix() + uuid.uuid4().hex
    custody.children = []
    custody.before = frozenset(multiprocessing.active_children())
    custody.failed = False


def child_env() -> dict:
    return {custody.token: "1"} if custody.token else {}


def note_child(process, positional, kwargs) -> None:
    """A started child, detached when it may outlive what ends with it: a new session or group, or
    whatever a ``preexec_fn`` does (Popen takes all of these at position 6 or later)."""
    custody.children.append((process, bool(
        len(positional) > 6 or kwargs.get("start_new_session") or kwargs.get("preexec_fn")
        or kwargs.get("process_group") is not None
        or int(kwargs.get("creationflags") or 0) & 0x208)))  # CREATE_NEW_PROCESS_GROUP, DETACHED_PROCESS


def _processes_gone() -> bool:
    """Whether what the test started has ended: every Popen child and multiprocessing child, and,
    only if a child was started detached, every carrier of the test's token by one census (about
    0.3 s; Windows has none, so such a folder waits). A descendant that double-forks inside its
    parent's group is not seen here; the run's containment still finds it and goes red."""
    token, custody.token = custody.token, ""  # the census's own `ps` must not carry it
    if any(child.poll() is None for child, _detached in custody.children):
        return False
    if set(multiprocessing.active_children()) - custody.before:
        return False
    if not any(detached for _child, detached in custody.children):
        return True
    from ouroboros.process_containment import pids_with_env_marker

    counts["censuses"] += 1
    return os.name != "nt" and pids_with_env_marker(token) == []


def release_passed(tmp, *, run_root: str, child_seam: bool) -> None:
    """Remove a passed test's ``tmp_path`` once its processes are proven gone.

    Anything else waits for the run's own containment proof and release: a failed phase or
    subtest (its folder is the diagnosis), retention in the folder or at the run's root (a debug
    Keep writes one there first), a run with no child seam (live data), a folder the OS will not
    let go (a Windows sharing violation). ``tmp_path_factory`` folders are never touched."""
    from ouroboros.test_environment import RETENTION_MARKER, retention_markers
    from ouroboros.test_evidence import remove_scratch_tree

    if not isinstance(tmp, pathlib.Path):
        return
    if (custody.failed or not child_seam or (run_root and os.path.lexists(os.path.join(run_root, RETENTION_MARKER)))
            or retention_markers(tmp) or not _processes_gone()):
        counts["waited"] += 1
        return
    try:
        remove_scratch_tree(tmp)
        counts["released"] += 1
    except OSError:
        counts["waited"] += 1


def number_once(config) -> None:
    """A released test's number stays spent, so no path comes back within a process.

    pytest numbers a folder from what the basetemp still holds. With passed tests' folders gone, the
    next test of that name would get the same path, and process-global state keyed by path
    (``install_identity``, ``store``, ``queue_observation``) would take it for the earlier test's.
    """
    factory, spent = config._tmp_path_factory, collections.Counter()
    mktemp = factory.mktemp

    def numbered_once(basename, numbered=True):
        if not numbered:
            return mktemp(basename, numbered=False)
        spent[basename] += 1
        return mktemp(f"{basename}{spent[basename] - 1}", numbered=False)

    factory.mktemp = numbered_once


def summary() -> str:
    return "test folders: " + ", ".join(f"{counts[name]} {name}" for name in ("released", "waited", "censuses"))
