"""Per-command memory leaves for sessions started by the packaged systemd unit (Linux, cgroup v2).

The unit runs the launcher, server and workers in its ``runtime`` subgroup (``DelegateSubgroup``) and
delegates the memory controller. Each gated local command gets its own leaf under ``<unit>/commands``:
an OOM at the leaf's limit kills that command (``memory.oom.group``), while ``OOMPolicy=continue`` keeps
the service alive. Limits come from one snapshot of the host (``host_memory.physical_memory``): the parent is the
total less ``HOST_MEMORY_RESERVE_FRACTION`` and the nonreclaimable memory held outside the commands, a new leaf all
the room the parent leaves free, a running leaf is only raised (``_compute``). An OOM at the shared ``commands``
limit (two commands growing together) takes a command, even a healthy one, never the service; best effort, it is the
LARGEST whole one: every process of the command holding most gets ``oom_score_adj`` 1000 (``_bias``, only raised),
refreshed at every start, pressure refresh, finish and wait slice. The kernel still picks a process, so a command
that shrank keeps its 1000 and the smaller of two such commands can die (an accepted residual). No new setting. The isolation covers OOMs within a leaf or the
``commands`` subtree only: memory that grows outside it after a snapshot (the runtime, a ``start_service`` service,
another program) can still bring an ancestor's or the host's OOM.

``probe()`` decides once per process (``Enabled`` or ``Unavailable``; a transient failure is reused for 30 s
only), ``begin()`` prepares a leaf under a flock shared by the unit's workers, ``Lease.place()`` moves the
gate shim into it and ``Lease.finish()`` reads the typed OOM facts and removes the leaf. Layouts without
support (older package, systemd below 254, desktop, AppImage, macOS) get None from ``begin()``: the old
path runs. On Linux ``status_note()`` always names the state (on, off outside the packaged unit, or the reason the
unit cannot isolate); it serves the last observed verdict and refreshes it off the caller's thread. Only
``systemctl show`` is ever run.
"""
from __future__ import annotations

import contextlib
import errno
import os
import re
import secrets
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Union

_ROOT = Path("/")  # where /proc and the cgroup mount are read (a fake tree in tests)
_ATTEMPTS = 3
_SHOW_TIMEOUT_SEC = 5
_LOCK_WAIT_SEC = 30.0  # the longest a start or finish waits for the unit's other workers
_TRANSIENT_TTL_SEC = 30.0  # a transient verdict is reused this long, so /status does not spawn systemctl each time
_now = time.monotonic
_USER_UNIT_DIR = "/.config/systemd/user"  # a user's own unit or drop-in shadows the package's
PACKAGED_UNIT = "ouroboros.service"  # the only unit whose cgroup is ever rewritten


@dataclass(frozen=True)
class Enabled:
    unit: str
    unit_cgroup: str  # e.g. /user.slice/.../ouroboros.service
    unit_root: Path
    commands: Path


@dataclass(frozen=True)
class Unavailable:
    code: str
    reason: str
    transient: bool = False  # not cached: the next call asks again


class CommandMemoryUnavailable(RuntimeError):
    """Support turned out absent while preparing: the old path runs; nothing was changed for the command."""

    def __init__(self, state: Unavailable) -> None:
        super().__init__(f"{state.code}: {state.reason}")
        self.state = state


class CommandMemoryRefused(RuntimeError):
    """The command was refused before its argv ran (confirmed support, but no room or a leaf step failed)."""

    process_not_started = True

    def __init__(self, code: str, reason: str) -> None:
        super().__init__(f"command not started ({code}): {reason}")
        self.code, self.reason = code, reason


_PROBED: Optional[Union[Enabled, Unavailable]] = None
_TRANSIENT: Optional[tuple] = None  # (Unavailable, deadline on ``_now``)
_PROBE_LOCK = threading.Lock()  # guards only the two verdict variables; never held across a systemctl call
_SLOW_LOCK = threading.Lock()  # serialises the slow ``_probe()`` itself


def _show(*argv: str) -> dict:
    # A packaged user service inherits no login session: give --user the bus it always has,
    # so the probe reads the manager from inside the service exactly as from a shell.
    env = os.environ.copy()
    if argv and argv[0] == "--user":
        env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{_uid()}")
        env.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path={env['XDG_RUNTIME_DIR']}/bus")
    out = subprocess.run(["systemctl", *argv], check=True, capture_output=True, text=True,
                         encoding="utf-8", timeout=_SHOW_TIMEOUT_SEC, env=env).stdout
    return dict(line.split("=", 1) for line in out.splitlines() if "=" in line)


def _uid() -> int:
    return os.getuid() if hasattr(os, "getuid") else 0  # cgroups are Linux; the probe still reads on any platform


def _cgroup_text() -> str:
    return (_ROOT / "proc/self/cgroup").read_text(encoding="utf-8")


def _cgroup_parts(text: str) -> list:
    path = next((line[3:] for line in text.splitlines() if line.startswith("0::")), None)
    return [part for part in (path or "").split("/") if part]


def _own_cgroup(text: str) -> Optional[tuple]:
    """``(unit cgroup path, unit name)`` when this process (``/proc/self/cgroup`` text, read once by the caller)
    lives in the packaged unit's ``runtime`` subgroup: only ``ouroboros.service`` is ever rewritten, never
    another delegated user service's ``commands``."""
    parts = _cgroup_parts(text)
    if len(parts) < 2 or parts[-1] != "runtime" or parts[-2] != PACKAGED_UNIT:
        return None
    return "/" + "/".join(parts[:-1]), parts[-2]


def _unit_dir(cgroup: str) -> Optional[Path]:
    """The unit's cgroup directory; None when the mount table is read and shows none. An unreadable table raises."""
    from ouroboros.host_memory import _beneath, _cgroup_mounts

    for top, point in _cgroup_mounts(_ROOT)[2]:
        if (below := _beneath(cgroup, top)) is not None:
            return (_ROOT / point.lstrip("/")).joinpath(*below)
    return None


def _check_unit(props: Mapping[str, str], what: str, *, subgroup: bool, cgroup: str) -> Optional[Unavailable]:
    """Why ``props`` of a ``systemctl show`` is not the delegated-memory layout, else None."""
    if props.get("LoadState") != "loaded":
        return Unavailable("unit_not_loaded", f"{what} is not loaded")
    control = props.get("ControlGroup", "")
    if not (control == cgroup if subgroup else cgroup == control or cgroup.startswith(control.rstrip("/") + "/")):
        return Unavailable("unit_cgroup_mismatch", f"{what} runs in {control or 'no cgroup'}, not this process's")
    wrong = None
    if props.get("OOMPolicy") != "continue":
        wrong = f"OOMPolicy={props.get('OOMPolicy') or 'unset'}"
    elif "memory" not in props.get("DelegateControllers", "").split():
        wrong = "the memory controller is not delegated"
    elif subgroup and props.get("DelegateSubgroup") != "runtime":
        wrong = "DelegateSubgroup is not runtime"
    if wrong is None:
        return None
    shadow = [p for p in (props.get("FragmentPath", ""), *props.get("DropInPaths", "").split())
              if _USER_UNIT_DIR in p]
    reason = f"{what}: {wrong}" + (f" (set by {shadow[0]}; not overridden)" if shadow else "")
    if props.get("NeedDaemonReload") == "yes":  # a fresh unit is inert until reload and restart
        return Unavailable("needs_reload", reason + "; the unit changed: daemon-reload and restart apply it", True)
    return Unavailable("unit_policy", reason)


def _without_runtime(unit: str) -> Unavailable:
    """The packaged unit runs without a ``runtime`` subgroup: a reload not yet applied, a systemd that ignores
    ``DelegateSubgroup`` (below 254), or an older package, in that order of what the manager says."""
    try:
        reload = _show("--user", "show", unit, "--property=NeedDaemonReload").get("NeedDaemonReload") == "yes"
        major = re.match(r"\D*(\d+)", _show("--user", "show", "--property=Version").get("Version", ""))
    except (OSError, subprocess.SubprocessError):
        reload, major = False, None
    if reload:
        return Unavailable("needs_reload", f"{unit} has no runtime subgroup yet: daemon-reload and restart apply the new unit", True)
    if major is not None and int(major.group(1)) < 254:
        return Unavailable("unit_without_runtime", f"systemd {major.group(1)} ignores DelegateSubgroup (254 or later is needed)")
    return Unavailable("unit_without_runtime", f"{unit} has no runtime subgroup (older package or DelegateSubgroup unset)")


def _kernel_props(unit_root: Path, cgroup: str) -> tuple:
    """The user bus can be unreachable from inside the packaged service (observed on a hosted kernel):
    the same properties then come from the unit's and the manager's cgroup files, which the kernel
    writes for the layout actually in force. Only what those files show is asserted."""
    def props(directory: Path, control: str, subgroup: bool) -> dict:
        controllers = (directory / "cgroup.controllers").read_text(encoding="ascii").split()
        oom = (directory / "memory.oom.group").read_text(encoding="ascii").strip()
        out = {"LoadState": "loaded", "ControlGroup": control, "DelegateControllers": " ".join(controllers),
               "OOMPolicy": "continue" if oom == "0" else "kill"}
        if subgroup:
            out["DelegateSubgroup"] = "runtime"  # this process's own cgroup already ends in it
        return out
    return props(unit_root, cgroup, True), props(unit_root.parent, cgroup.rsplit("/", 1)[0], False)


def _probe() -> Union[Enabled, Unavailable]:
    if not sys.platform.startswith("linux"):
        return Unavailable("platform", "not Linux")
    from ouroboros.host_memory import physical_memory

    try:
        text = _cgroup_text()
    except OSError:
        return Unavailable("no_cgroup_v2", "/proc/self/cgroup is unreadable", True)
    if "0::" not in text:  # v1 rows are ``N:controllers:/path``: the unit's identity is still readable there
        if any(PACKAGED_UNIT in line.split(":", 2)[-1].split("/") for line in text.splitlines()):
            return Unavailable("cgroup_v1", f"{PACKAGED_UNIT} runs on cgroup v1; per-command memory needs cgroup v2")
        return Unavailable("no_cgroup_v2", "cgroup v2 is not in use")
    own = _own_cgroup(text)  # the one read of the membership: a second could fail into a cached "no subgroup"
    if own is None:
        parts = _cgroup_parts(text)
        if parts and parts[-1] == PACKAGED_UNIT:  # our unit, old layout
            return _without_runtime(PACKAGED_UNIT)
        return Unavailable("no_runtime_subgroup", "this process does not run in the packaged unit's runtime subgroup")
    cgroup, unit = own
    try:
        unit_root = _unit_dir(cgroup)
    except (OSError, ValueError):  # the unit is identified; only the mount table could not be read
        return Unavailable("mounts_unreadable", "the cgroup mount table could not be read", True)
    if unit_root is None:
        return Unavailable("cgroup_not_visible", "the unit's cgroup directory is not visible")
    props = ("LoadState", "ControlGroup", "OOMPolicy", "DelegateControllers", "DelegateSubgroup",
             "FragmentPath", "DropInPaths", "NeedDaemonReload")
    try:
        service = _show("--user", "show", unit, *(f"--property={name}" for name in props))
        manager = _show("show", f"user@{_uid()}.service", *(f"--property={name}" for name in props))
    except (subprocess.CalledProcessError, OSError, subprocess.SubprocessError) as exc:
        try:  # a denied or absent bus is not a missing layout: read the same facts from the kernel
            service, manager = _kernel_props(unit_root, cgroup)
        except OSError:
            tail = " ".join((getattr(exc, "stderr", "") or str(exc)).split())[-200:]
            return Unavailable("systemctl_unreadable", f"systemctl show failed: {tail}", True)
    for found in (_check_unit(service, unit, subgroup=True, cgroup=cgroup),
                  _check_unit(manager, f"user@{_uid()}.service", subgroup=False, cgroup=cgroup)):
        if found is not None:
            return found
    try:
        if "memory" not in (unit_root / "cgroup.controllers").read_text(encoding="ascii").split():
            return Unavailable("memory_controller_missing", "the memory controller is not available in the unit's cgroup")
    except OSError:
        pass  # unreadable proves nothing: the first leaf says
    if physical_memory() is None:
        return Unavailable("memory_unreadable", "host memory could not be read", True)
    try:  # the same swap verdict ``begin()`` reaches, read-only, so /status and the workers agree up front
        if _meminfo("SwapTotal") > 0 and not (unit_root / "memory.swap.max").exists():
            return Unavailable("swap_unmanaged", "memory.swap.max is absent while swap exists")
    except (OSError, ValueError):
        pass
    return Enabled(unit, cgroup, unit_root, unit_root / "commands")


def _cached() -> Optional[Union[Enabled, Unavailable]]:
    """The verdict in force: a definite one, or a transient one not yet aged out. The caller holds ``_PROBE_LOCK``."""
    if _PROBED is not None:
        return _PROBED
    return _TRANSIENT[0] if _TRANSIENT is not None and _now() < _TRANSIENT[1] else None


def probe(*, refresh: bool = False) -> Union[Enabled, Unavailable]:
    """The cached verdict for this process; a transient failure is reused for ``_TRANSIENT_TTL_SEC`` and then
    asked again. Sweeps dead creators' leaves. The slow read runs under its own lock, so a caller that only
    needs the cached verdict (``begin``'s store, ``status_note``) never waits for ``systemctl``."""
    global _PROBED, _TRANSIENT
    with _PROBE_LOCK:
        result = None if refresh else _cached()
    if result is None:
        with _SLOW_LOCK:
            with _PROBE_LOCK:
                result = None if refresh else _cached()  # another thread may have just filled it
            if result is None:
                result = _probe()
                with _PROBE_LOCK:
                    if isinstance(result, Unavailable) and result.transient:
                        _PROBED, _TRANSIENT = None, (result, _now() + _TRANSIENT_TTL_SEC)
                    else:
                        _PROBED, _TRANSIENT = result, None
    if isinstance(result, Enabled):
        _sweep(result)
    return result


_IDENTIFIED_TRANSIENT = ("systemctl_unreadable", "memory_unreadable", "mounts_unreadable")  # the unit is identified; only a reading failed
_REFRESHING = threading.Event()


def _after_fork_in_child() -> None:
    """Only the forking thread survives a fork (``OUROBOROS_WORKER_START_METHOD=fork``): a lock the status refresh
    thread held, its running flag and the parent's verdicts mean nothing here, so the child starts them afresh."""
    global _PROBE_LOCK, _SLOW_LOCK, _REFRESHING, _PROBED, _TRANSIENT
    _PROBE_LOCK, _SLOW_LOCK, _REFRESHING, _PROBED, _TRANSIENT = threading.Lock(), threading.Lock(), threading.Event(), None, None
    _FINISHED_BUSY.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


def _observed() -> Optional[Union[Enabled, Unavailable]]:
    """The last verdict still in force: a definite one, or a transient one not yet aged out."""
    with _PROBE_LOCK:
        return _cached()


def isolating() -> bool:
    """Whether commands here run in their own memory group, by the last verdict (never probes)."""
    return isinstance(_observed(), Enabled)


def _in_packaged_unit() -> bool:
    """One cheap read: does this process's cgroup path (v2 or v1 row) name ``ouroboros.service``? Anywhere else
    there is nothing for a probe to find out, so no thread is started for it."""
    try:
        return any(PACKAGED_UNIT in line.split(":", 2)[-1].split("/") for line in _cgroup_text().splitlines())
    except OSError:
        return False


def _refresh() -> None:
    try:
        probe()
    finally:
        _REFRESHING.clear()


def status_note() -> str:
    """One /status line on Linux, empty elsewhere: ``on`` when commands get their own memory group, ``off`` when this
    process does not run in the packaged unit (a desktop or AppImage session, a terminal: commands share the runtime's
    memory, as before), else the reason the unit cannot isolate. Never probes on the caller's thread (``systemctl`` can
    take seconds): it serves the last verdict and, while there is none, asks for one from a daemon thread."""
    if not sys.platform.startswith("linux"):
        return ""
    state = _observed()
    if state is None:
        if not _in_packaged_unit():  # the one cheap membership read: nothing for a probe to find out
            return f"command_memory: off (not started by the packaged {PACKAGED_UNIT}; commands share the runtime's memory)"
        if not _REFRESHING.is_set():
            _REFRESHING.set()
            threading.Thread(target=_refresh, name="command-memory-probe", daemon=True).start()
        return "command_memory: checking"
    if isinstance(state, Enabled):
        return "command_memory: on (each local command has its own memory group)"
    return f"command_memory: off ({state.code}: {state.reason})"


# --------------------------------------------------------------------------------------- cgroup files

def _read(path: Path) -> str:
    return path.read_text(encoding="ascii").strip()


def _write(path: Path, value: str) -> None:
    path.write_text(value, encoding="ascii")


def _page_size() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return 4096


def _floor(value: int) -> int:
    """The kernel stores a byte limit in whole pages; write (and compare) exactly what it will keep."""
    page = _page_size()
    return value // page * page


def _ceil(value: int) -> int:
    page = _page_size()
    return -(-value // page) * page


def _write_max(path: Path, value: int) -> int:
    value = _floor(value)
    _write(path, str(value))
    return value


def _numeric(path: Path) -> Optional[int]:
    try:
        return int(_read(path))
    except (OSError, ValueError):
        return None  # absent, or ``max``


def _keyed(path: Path) -> dict:
    out = {}
    for line in _read(path).splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("-").isdigit():
            out[parts[0]] = int(parts[1])
    return out


def _meminfo(name: str) -> int:
    for line in (_ROOT / "proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith(name + ":"):
            return int(line.split()[1]) * 1024
    return 0


def _enable_memory(directory: Path) -> None:
    control = directory / "cgroup.subtree_control"
    if "memory" in _read(control).split():
        return
    try:
        _write(control, "+memory")
    except (BlockingIOError, FileExistsError):
        pass
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.EINVAL):  # the controller is not there to enable: a layout, not a fault
            raise CommandMemoryUnavailable(Unavailable("memory_controller_missing", "the memory controller cannot be enabled here")) from exc
        if exc.errno not in (errno.EBUSY, errno.EEXIST):  # a concurrent worker won the same write
            raise


@contextlib.contextmanager
def _lock(wait: Optional[float] = None):
    """The flock shared by the unit's workers, waited for at most ``wait`` (default ``_LOCK_WAIT_SEC``; 0 tries
    once), then a ``lock_timeout`` refusal."""
    from ouroboros import platform_layer

    base = os.environ.get("XDG_RUNTIME_DIR")  # per user already; the /tmp fallback is per UID
    uid = _uid()
    path = Path(base) / "ouroboros-command-memory.lock" if base else Path(tempfile.gettempdir()) / f"ouroboros-command-memory-{uid}.lock"
    fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        deadline = _now() + (_LOCK_WAIT_SEC if wait is None else wait)
        while True:
            try:
                platform_layer.file_lock_exclusive_nb(fd)
                break
            except OSError:
                if _now() >= deadline:
                    raise CommandMemoryRefused("lock_timeout", "another start or finish held the memory lock too long")
                time.sleep(0.05)
        try:
            yield
        finally:
            platform_layer.file_unlock(fd)
    finally:
        os.close(fd)


def _held(group: Path) -> tuple:
    """``(memory.current, nonreclaimable)`` of a group: its charge less the file cache and reclaimable slab the
    kernel gives back before an OOM (``memory.stat`` ``inactive_file`` + ``active_file`` + ``slab_reclaimable``,
    consistent across the commands and their siblings; unreadable counts nothing as reclaimable)."""
    current = int(_read(group / "memory.current"))
    try:
        stat = _keyed(group / "memory.stat")
        reclaimable = sum(stat.get(key, 0) for key in ("inactive_file", "active_file", "slab_reclaimable"))
    except (OSError, ValueError):
        reclaimable = 0
    return current, max(0, current - reclaimable)


def _ancestor_budget(commands: Path, reserve: int, held: int) -> Optional[int]:
    """A finite ancestor must keep room for its own non-command charges, including other applications in its slice."""
    path, corridor = commands, []
    while (ancestor := path.parent) != path and (ancestor / "cgroup.controllers").is_file():
        ceiling = _read(ancestor / "memory.max") if (ancestor / "memory.max").exists() else "max"
        corridor.append((path, ancestor, None if ceiling == "max" else int(ceiling)))
        path = ancestor
    last = max((idx for idx, (_, _, cap) in enumerate(corridor) if cap is not None), default=-1)
    outside, bound = 0, None
    for child, ancestor, cap in corridor[:last + 1]:
        # Read sibling usage directly: the build may grow, shrink, and regrow around an ancestor reading.
        for sibling in ancestor.iterdir():
            if sibling != child and (sibling / "memory.current").is_file():
                outside += _held(sibling)[1]
        if cap is not None:
            # Also account for charges off the measured child/sibling paths (e.g. dead cgroups at this level).
            other = max(outside, _held(ancestor)[1] - held)
            bound = min(cap - other - reserve, bound) if bound is not None else cap - other - reserve
    return bound


def _compute(commands: Path):
    """``(leaf_quota, parent_max, host_snapshot, free)`` from one host snapshot, every byte count a whole number of pages.

    The parent is what commands may hold without exposing the reserve: ``total - reserve - outside``, ``outside`` the
    nonreclaimable memory everything else holds. It is the snapshot's used memory less the commands' own, read just
    before and just after it (the smaller of the two, so a command that moved between the reads never shrinks
    ``outside``), and never less than the ``runtime`` subgroup's own charge, read directly (a build cycling between the
    reads cannot hide the service). No floor at the commands' current charge: a parent under it makes the kernel
    reclaim their cache first, then end the largest command. ``free`` is what the parent leaves above the commands'
    larger reading: a new leaf gets all of it, a running leaf is raised by it. A finite ancestor subtracts
    non-command charges using that ancestor's own accounting, separately from the physical host calculation."""
    from ouroboros.host_memory import physical_memory

    before = _held(commands)[1]
    snap = physical_memory()
    after = _held(commands)[1]
    if snap is None:
        raise CommandMemoryRefused("memory_unreadable", "host memory could not be read")
    try:
        runtime = _held(commands.parent / "runtime")[1]
    except (OSError, ValueError):
        runtime = 0
    try:
        bound = _ancestor_budget(commands, snap.reserve_bytes, max(before, after))
        total, available = ((snap.total_bytes, snap.available_bytes) if bound is None
                            else (_meminfo("MemTotal"), _meminfo("MemAvailable")))
    except ValueError as exc:
        raise CommandMemoryRefused("memory_unreadable", "a cgroup memory charge or limit could not be read") from exc
    outside = max(runtime, total - available - min(before, after))
    parent = total - snap.reserve_bytes - outside
    parent = _floor(max(0, parent if bound is None else min(parent, bound)))
    free = _floor(max(0, parent - max(before, after)))
    return free, parent, snap, free


def _raise_member_score(pid: str, cgroup: Path, score: int) -> None:
    """Read membership and raise the score through one proc directory, never addressing a recycled PID by name."""
    # Windows has neither flag; this reader only runs where /proc does, and the module must still import there.
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(_ROOT / "proc" / pid, flags)
    try:
        def open_member(name, flags):
            return os.open(name, flags, dir_fd=fd)

        with open("cgroup", encoding="ascii", opener=open_member) as handle:
            if f"0::{cgroup}" not in handle.read().splitlines():
                return
        with open("oom_score_adj", encoding="ascii", opener=open_member) as handle:
            raised = score > int(handle.read())
        if raised:
            with open("oom_score_adj", "w", encoding="ascii", opener=open_member) as handle:
                handle.write(str(score))
    finally:
        os.close(fd)


def _bias(commands: Path, total: int = 0) -> None:
    """Best effort: make the LARGEST whole command the victim when the shared ``commands`` limit (or the host) runs out.
    The kernel scores each process by its own memory plus ``oom_score_adj`` thousandths of the memory in play, so a
    build of many small processes would lose to a smaller single one if each process were scored by itself. Every
    process of the command holding the most nonreclaimable memory gets 1000, a whole limit above any process of
    another command; every process of another command gets its command's share, ``held * 1000 // limit`` (the parent's
    limit, else ``total``). Only raised, never below 0 (lowering needs privilege); an exited pid or an unreadable
    file is skipped."""
    held = {}
    for leaf in commands.glob("cmd-*"):
        try:
            held[leaf] = _held(leaf)[1]
        except (OSError, ValueError):
            continue
    limit = _numeric(commands / "memory.max") or total
    if not held or limit <= 0:
        return
    from ouroboros.host_memory import _cgroup_mounts

    try:
        group = next((Path(top) / commands.relative_to(mount) for top, point in _cgroup_mounts(_ROOT)[2]
                      if commands.is_relative_to(mount := _ROOT / point.lstrip("/"))), None)
    except (OSError, ValueError):
        return
    if group is None:
        return
    largest = max(held, key=held.get)
    for leaf, value in held.items():
        score = 1000 if leaf == largest and value > 0 else min(1000, value * 1000 // limit)
        try:
            pids = _read(leaf / "cgroup.procs").split() if score > 0 else ()
        except OSError:
            continue
        for pid in filter(str.isdecimal, pids):
            try:
                _raise_member_score(pid, group / leaf.name, score)
            except (OSError, ValueError):
                continue


def _raise_live_leaves(commands: Path, parent: int, room: int) -> None:
    """Running commands only gain room: each leaf up to what it holds itself plus this snapshot's free room, within
    the parent (never lowered: a lower value could OOM a command that was granted more)."""
    for leaf in commands.glob("cmd-*"):
        try:
            existing = _read(leaf / "memory.max")
            target = _floor(min(parent, _held(leaf)[1] + room))
            if existing == "max" or int(existing) >= target:
                continue
            _write_max(leaf / "memory.max", target)
        except (OSError, ValueError):
            continue  # it ended meanwhile


def _set_parent(commands: Path, *, refuse: bool):
    """Write ``commands/memory.max`` (bounded retry on EBUSY) and raise the live leaves; ``refuse``: a start whose
    leaf would get no room raises. Returns ``(leaf_quota, parent)``."""
    for attempt in range(_ATTEMPTS):
        quota, parent, snap, room = _compute(commands)
        if refuse and quota <= 0:
            host = _meminfo("MemAvailable")
            raise CommandMemoryRefused(*(
                ("under_reserve", "free memory is under the reserve; swap is not counted") if host <= snap.reserve_bytes
                else ("budget_exhausted", "the unit's memory limit leaves no room above the reserve")))
        try:
            _write_max(commands / "memory.max", parent)
        except OSError as exc:
            if exc.errno != errno.EBUSY or attempt == _ATTEMPTS - 1:
                raise
            continue
        _raise_live_leaves(commands, parent, room)
        _bias(commands, snap.total_bytes)  # at every start, pressure refresh and finish
        return min(quota, parent), parent
    raise AssertionError("unreachable")


_FINISHED_BUSY: set = set()  # leaves of finished commands a background child still held when they ended


def _sweep(state: Enabled) -> None:
    """Remove empty leaves whose creator is gone, and finished commands' leaves a background child has since
    released (a busy one is kept)."""
    for leaf in list(_FINISHED_BUSY):
        try:
            leaf.rmdir()
        except FileNotFoundError:
            pass
        except OSError:
            continue
        _FINISHED_BUSY.discard(leaf)
    try:
        leaves = list(state.commands.glob("cmd-*"))
    except OSError:
        return
    for leaf in leaves:
        match = re.fullmatch(r"cmd-(\d+)-[0-9a-f]+", leaf.name)
        if match is None or (_ROOT / "proc" / match.group(1)).exists():
            continue
        try:
            leaf.rmdir()
        except OSError:
            pass


@dataclass
class Lease:
    state: Enabled
    leaf: Path
    baseline: int  # commands/memory.events.local oom at creation
    seen_max: int = 0  # the leaf's ``memory.events`` ``max`` count when its room was last recomputed

    def place(self, pid: int) -> None:
        """Move ``pid`` into the leaf and read its membership back; any failure refuses the command."""
        try:
            _write(self.leaf / "cgroup.procs", str(pid))
            member = (_ROOT / "proc" / str(pid) / "cgroup").read_text(encoding="utf-8")
        except OSError as exc:
            raise CommandMemoryRefused("move_failed", f"the command could not enter its memory group: {exc}") from exc
        if f"0::{self.state.unit_cgroup}/commands/{self.leaf.name}" not in member.splitlines():
            raise CommandMemoryRefused("move_failed", "the command is not in its memory group after the move")

    def pressure(self) -> None:
        """A cheap trigger for a command that started low and has since hit its limit: when the leaf's ``max`` event
        count moved, the room is recomputed (memory other commands freed since the start is raised into, never lowered).
        The lock is tried once: a busy one leaves the count unseen, so a later wait slice tries again."""
        try:
            hits = _keyed(self.leaf / "memory.events").get("max", 0)
        except (OSError, ValueError):
            return
        if hits <= self.seen_max:
            return
        try:
            with _lock(0):
                _set_parent(self.state.commands, refuse=False)
            self.seen_max = hits
        except (OSError, CommandMemoryRefused):
            pass  # a later start, finish or slice recomputes it

    def bias(self) -> None:
        """Re-rank every command's victim score from what each holds now (``_bias``): a growing command is not
        waiting for another start or finish to count as the largest."""
        _bias(self.state.commands)

    def watch(self, stop=None):
        """``stop`` (a Stop-now reader, or None) widened to run ``pressure()`` and ``bias()`` at every wait slice that is not a stop."""
        def read():
            stopped = stop() if stop is not None else None
            if stopped is not True:  # Stop first: a pressure refresh never stands between a command and its Stop
                self.pressure()
                self.bias()
            return stopped
        return read

    def finish(self, pid: int) -> dict:
        """Typed OOM facts, then the leaf is removed when empty and the parent room recomputed."""
        facts = outcome(self.leaf, pid, self.baseline)
        try:
            self.leaf.rmdir()  # a busy leaf stays; the next probe sweeps it once it is empty or its creator is gone
        except FileNotFoundError:
            pass
        except OSError:
            _FINISHED_BUSY.add(self.leaf)
        try:
            with _lock():
                _set_parent(self.state.commands, refuse=False)
        except (OSError, CommandMemoryRefused):
            pass  # the parent keeps its last value; a later start recomputes it
        return facts


def begin(deadline: Optional[float] = None) -> Optional[Lease]:
    """A prepared leaf for one command, or None when this layout has no support (the old path runs). ``deadline``
    (``_now`` clock, the command's absolute bound) narrows the wait for the workers' lock to what remains."""
    global _PROBED, _TRANSIENT
    state = probe()
    if isinstance(state, Unavailable) and state.code in _IDENTIFIED_TRANSIENT:  # the packaged unit is known: ask once more
        state = probe(refresh=True)
        if isinstance(state, Unavailable) and state.code in _IDENTIFIED_TRANSIENT:
            raise CommandMemoryRefused(state.code, state.reason)  # not an unisolated run; the next sample may admit
    if not isinstance(state, Enabled):
        return None
    try:
        with _lock(None if deadline is None else max(0.0, min(_LOCK_WAIT_SEC, deadline - _now()))):
            state.commands.mkdir(exist_ok=True)
            _enable_memory(state.unit_root)
            _enable_memory(state.commands)
            swap = state.commands / "memory.swap.max"  # before the parent: an unsupported layout changes nothing
            swap_managed = swap.exists()
            if not swap_managed:
                if _meminfo("SwapTotal") > 0:
                    raise CommandMemoryUnavailable(Unavailable("swap_unmanaged", "memory.swap.max is absent while swap exists"))
            else:  # 0 always, even without swap
                _write(swap, "0")
                if _read(swap) != "0":
                    raise CommandMemoryRefused("prepare_failed", "memory.swap.max did not read back")
            quota, _ = _set_parent(state.commands, refuse=True)
            leaf = state.commands / f"cmd-{os.getpid()}-{secrets.token_hex(4)}"
            leaf.mkdir()
            try:
                for name, value in (("memory.max", quota), ("memory.swap.max", "0"), ("memory.oom.group", "1")):
                    if name == "memory.swap.max" and not swap_managed:
                        continue
                    if name == "memory.max":
                        _write_max(leaf / name, value)
                        kept = _numeric(leaf / name)
                        if kept is None or kept != _floor(value):
                            raise CommandMemoryRefused("prepare_failed", f"{name} did not read back")
                        continue
                    _write(leaf / name, value)
                    if _read(leaf / name) != value:
                        raise CommandMemoryRefused("prepare_failed", f"{name} did not read back")
                baseline = _keyed(state.commands / "memory.events.local").get("oom", 0)
            except BaseException:
                try:
                    leaf.rmdir()
                except OSError:
                    pass
                raise
    except CommandMemoryUnavailable as exc:
        with _PROBE_LOCK:
            _PROBED, _TRANSIENT = exc.state, None  # a definite absence: the old path runs and /status says why
        return None
    except CommandMemoryRefused:
        raise
    except OSError as exc:
        raise CommandMemoryRefused("prepare_failed", f"the memory group could not be prepared: {exc}") from exc
    return Lease(state, leaf, baseline)


def outcome(leaf_path: Path, pid: int, baseline: int = 0) -> dict:
    """What the kernel recorded for one finished command: facts, or ``None`` where it could not be read.

    ``victim_in_leaf``: a process of this command's leaf was OOM-killed; ``leaf_local_oom`` and
    ``commands_local_oom`` count OOM events at the leaf and (since launch) the whole commands group. They
    state what was observed; no verdict about the service is drawn here."""
    leaf_path = Path(leaf_path)

    def events(path: Path):
        try:
            return _keyed(path)
        except (OSError, ValueError):
            return None

    leaf = events(leaf_path / "memory.events.local")
    commands = events(leaf_path.parent / "memory.events.local")
    kills = None if leaf is None else leaf.get("oom_kill", 0) + leaf.get("oom_group_kill", 0)
    return {
        "victim_in_leaf": None if kills is None else kills > 0,
        "leaf_local_oom": None if leaf is None else leaf.get("oom", 0),
        "commands_local_oom": None if commands is None else max(0, commands.get("oom", 0) - baseline),
        "unknown": leaf is None or commands is None,
        "pid": int(pid),
    }


def facts_worth_publishing(facts: Optional[Mapping], returncode: Optional[int] = None) -> Optional[dict]:
    """The outcome only when something happened (an OOM event, or a SIGKILL with unreadable events)."""
    if not facts:
        return None
    happened = (facts.get("victim_in_leaf") or facts.get("leaf_local_oom") or facts.get("commands_local_oom"))
    return dict(facts) if happened or (facts.get("unknown") and returncode == -9) else None


# ------------------------------------------------------------------------------------------------ gate
# No preexec_fn: the gate is a Python helper started with a minimal environment (its scope markers and a default
# PATH, so nothing the command selected, such as a preloaded library, runs in the service's group) in isolated mode
# (``-I -S``). It blocks on its stdin until the parent has moved it into its leaf and registered it. The parent then
# sends the command's whole environment over that pipe, never in argv (an argv is world-readable in /proc and audit
# records): a length line, every entry as raw NUL-separated ``NAME=value`` bytes and ``GATE_SENTINEL``. The helper
# checks the exact byte count and the sentinel, else exits 125 before the argv runs (EOF included: the parent died,
# refused or cut the stream), then ``execvpe``s the ORIGINAL argv with exactly that environment, /dev/null on stdin:
# every admitted name reaches the target (a bash-readonly ``UID``, a ``build.profile``), ``$0`` stays the argv's own,
# and the PATH search runs in the child's cwd as ``Popen``'s would (relative entries, shebangs, ``ENOEXEC``). The
# helper's own ``-I -S`` startup leaves SIGPIPE and SIGXFSZ ignored; they are reset to the default first, so a
# pipeline (``yes | head``) behaves as under ``Popen``; that reset follows the ``x`` below, so a parent that stopped
# waiting for the verdict (``EXEC_TIMEOUT``) never turns it into a SIGPIPE death. Any failure while preparing (a
# ``MemoryError`` in its leaf) exits 125 before the argv. Only once prepared does it write ``x`` to a close-on-exec
# status pipe, then exec: an exec failure follows with its errno, a typed not-started refusal, never a run's exit
# 126/127; EOF after ``x`` is the exec; EOF without ``x`` is a helper that ended before the argv (``exec_verdict``).
_ENV_HELPER = "\n".join((
    "import os,sys,signal", "s=int(sys.argv[1]);os.set_inheritable(s,False);r=sys.stdin.buffer",
    "try:", " n=int(r.readline());b=r.read(n);t=r.read(%d)" % len(b"\0ouroboros-gate-release\n"),
    " if len(b)!=n or t!=b\"\\0ouroboros-gate-release\\n\":os._exit(125)",
    ' e=dict(x.partition(b"=")[::2] for x in b.split(b"\\0") if x)',
    " os.dup2(os.open(os.devnull,os.O_RDONLY),0);a=[os.fsencode(x) for x in sys.argv[2:]]",
    "except BaseException:", " os._exit(125)",
    "try:", " os.write(s,b'x')",
    "except OSError:", " pass",
    "for n in('SIGPIPE','SIGXFSZ'):hasattr(signal,n) and signal.signal(getattr(signal,n),signal.SIG_DFL)",
    "try:", " os.execvpe(a[0],a,e)",
    "except BaseException as x:", " os.write(s,b\"%d\"%(getattr(x,'errno',0) or 0))",
    "os._exit(125)", ""))
GATE_SENTINEL = b"\0ouroboros-gate-release\n"
GATE_NOT_STARTED = 125  # the gate's own exit; a completed run's 125, 126 or 127 is never read as "not started"
_STATUS_WAIT_SEC = 30.0  # the longest the parent waits for the exec verdict without a deadline


def _interpreter() -> Optional[str]:
    """A Python that can run ``_ENV_HELPER`` (not a frozen launcher, which is no interpreter)."""
    import shutil
    from ouroboros.platform_layer import bundled_resource_bases, embedded_python_candidates

    if not getattr(sys, "frozen", False) and sys.executable and os.access(sys.executable, os.X_OK):
        return sys.executable
    for base in bundled_resource_bases():
        for candidate in embedded_python_candidates(base):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return os.path.abspath(candidate)
    for name in ("python3", "python"):
        if found := shutil.which(name):
            return found
    return None


def gate_argv(argv, status_fd: int) -> list:
    """The helper's argv: ``status_fd`` (inherited, close-on-exec in the helper) and the command's own argv."""
    interpreter = _interpreter()
    if interpreter is None:
        raise CommandMemoryRefused("env_unsupported", "the memory gate needs a Python interpreter")
    return [interpreter, "-I", "-S", "-c", _ENV_HELPER, str(int(status_fd)), *(str(part) for part in argv)]


def gate_release(env: Mapping[str, str]) -> bytes:
    """The command's environment as the helper reads it, short of the one byte that releases it: a length line, raw
    ``NAME=value`` entries and the sentinel less its last byte (the helper blocks for that byte)."""
    blob = b"\0".join(os.fsencode(key) + b"=" + os.fsencode(value) for key, value in env.items()
                      if key and "=" not in key and "\0" not in key and "\0" not in value)
    return b"%d\n" % len(blob) + blob + GATE_SENTINEL[:-1]


def _write_all(proc, data) -> None:
    """Raw fd writes: the pipe may be wrapped as text."""
    data, fd = memoryview(data), proc.stdin.fileno()
    while data:
        data = data[os.write(fd, data):]


def stage_gate(proc, env: Mapping[str, str]) -> None:
    """Hand the helper the whole environment, built and written outside any admission lock; it still waits for the last byte."""
    _write_all(proc, gate_release(env))


def release_gate(proc, env: Optional[Mapping[str, str]] = None) -> None:
    """Let the helper exec its argv: the one final byte (after ``stage_gate``, or ``env`` staged here). A failure
    leaves the helper short of its sentinel, which it reads as a refusal."""
    if env is not None:
        stage_gate(proc, env)
    _write_all(proc, GATE_SENTINEL[-1:])


EXEC_TIMEOUT = -1  # ``exec_verdict``: neither an exec errno nor the close-on-exec EOF within the wait
GATE_ENDED = -2  # ``exec_verdict``: EOF before the helper's ``x``: it ended before the argv could run


def exec_verdict(status_fd: int, wait: float) -> Optional[int]:
    """The helper's exec errno; None once the exec succeeded (``x``, then the close-on-exec end reached EOF);
    ``GATE_ENDED`` when the helper ended before it was ready to exec (killed, or a failure while preparing);
    ``EXEC_TIMEOUT`` when ``wait`` ran out first (unknown: the released command may be running)."""
    import selectors

    deadline, data = _now() + max(0.0, wait), b""
    with selectors.DefaultSelector() as sel:  # no FD_SETSIZE ceiling, unlike select.select
        sel.register(status_fd, selectors.EVENT_READ)
        while True:
            if not sel.select(max(0.0, deadline - _now())):
                return EXEC_TIMEOUT
            chunk = os.read(status_fd, 64)
            if not chunk:
                if not data.startswith(b"x"):
                    return GATE_ENDED
                return int(data[1:]) if data[1:].isdigit() else None
            data += chunk


def check_deadline(deadline: Optional[float]) -> None:
    """The command's one absolute bound (``_now`` clock), read in its final admission just before the launch: none
    left is a typed not-started refusal. Both the gated start and the ungated memory-path fallback call it."""
    if deadline is not None and _now() >= deadline:
        raise CommandMemoryRefused("deadline", "the command's time ran out before it could start")


def refuse_gate(proc) -> None:
    """Close the write end without writing: EOF makes the helper exit 125 before it runs the argv."""
    try:
        proc.stdin.close()
    except (OSError, ValueError):
        pass
    try:
        proc.wait(timeout=5)
    except (subprocess.SubprocessError, OSError):
        pass


def gated_start(lease, argv, env, cwd, *, spawn, confirmed, drive_root, extra, target, owed, deadline=None, stop=None):
    """The memory-gated start both local entry points share (``shell_process._tracked_subprocess_run``,
    ``shell_process.execute_local_gated``).

    ``spawn(gate_argv, gate_env, pass_fds)`` starts the helper, which keeps no environment but its scope markers
    until it is released, and returns its Popen. Then, in order: the helper is placed in its leaf; its launch-identity
    record (marker, birth) is written and indexed (``already_started=False``: a failed write leaves nothing indexed);
    the turn's Stop-now reader ``stop`` is read (True refuses; False or unknown admits); ``operation_start`` takes the
    final Stop/Panic check, the command's absolute ``deadline`` (``_now`` clock, ``check_deadline``) is checked and the
    release is sent with the command's whole environment. Returns ``(proc, record_path)`` with the argv exec'd;
    ``record_path`` is None when custody was owed and could not be written: the placed command is then released with
    the ordinary ``host_unconfirmed`` fact the ungated path latches (it may run), so ``rm`` and ``du`` still work on a
    full disk. Any earlier failure refuses the gate (EOF, exit 125), settles an exited helper's receipt and re-raises
    typed ``process_not_started``, except Pause and store refusals, which keep their own typed results; so does an exec
    the kernel refused (its errno, as ``Popen`` raises it) and a helper that ended before it was ready to exec
    (``GATE_ENDED``: the argv never ran, whatever its exit status says). An exec verdict that did not arrive in time (``EXEC_TIMEOUT``)
    is neither: the released command keeps running, the ordinary wait settles it, and the ``host_unconfirmed`` fact
    ``exec_unconfirmed`` records that it may have run. The caller finishes the lease. After the exec the argv runs:
    a failure then stops it."""
    from ouroboros import workspace_executor as we
    from ouroboros.owner_pause import OwnerPauseRefused, operation_start
    from ouroboros.platform_layer import kill_process_tree
    from ouroboros.process_census import LAUNCH_SCOPE_PREFIX, SCOPE_VALUE, birth_text, mint_marker, scope_entries
    from ouroboros.process_scope import record_task_group
    from ouroboros.store import StoreError
    from ouroboros.tool_custody import retain_unconfirmed_host_operation
    from ouroboros.tools.shell_process import CommandAdmissionClosed, _admit_gate

    proc = record = None
    released = executed = False
    status_r, status_w = os.pipe()
    try:
        marker = mint_marker(LAUNCH_SCOPE_PREFIX)
        try:
            proc = spawn(gate_argv(argv, status_w), {**scope_entries(env), marker: SCOPE_VALUE, "PATH": os.defpath},
                         (status_w,))
        finally:
            os.close(status_w)
        lease.place(proc.pid)
        if target is not None:
            record_task_group(*target, proc.pid)
        if drive_root is not None or owed:
            record = we._register_process(drive_root, {**extra, "record_type": "foreground", "host_pid": proc.pid,
                                                       "scope_marker": marker, "host_birth": birth_text(proc.pid)}, already_started=False)
        if record is None and owed:  # custody was owed and not written: the leaf stays, the fact is the ungated path's
            retain_unconfirmed_host_operation("executor_custody_write_failed")
        if record is not None:
            confirmed()
        stage_gate(proc, {**env, marker: SCOPE_VALUE})  # the bulk, outside the Pause admission lock
        if stop is not None and stop() is True:  # a turn's Stop now, read outside the Pause lock (it reads a file)
            raise CommandAdmissionClosed("Stop now was already effective: the command was not started")
        with operation_start():  # the one start point: the final Stop/Panic check, the deadline, then the last byte
            _admit_gate()
            check_deadline(deadline)
            release_gate(proc)
            released = True
        failed = exec_verdict(status_r, _STATUS_WAIT_SEC if deadline is None else deadline - _now())
        if failed == GATE_ENDED:
            raise CommandMemoryRefused("gate_ended", "the memory gate's helper ended before the command could start")
        if failed == EXEC_TIMEOUT:  # unknown, not failed: the released command may be running; the caller's wait settles it
            retain_unconfirmed_host_operation("exec_unconfirmed")
        elif failed is not None:
            exc = OSError(failed, os.strerror(failed), str(argv[0]))
            exc.process_not_started = True
            raise exc
        executed = True
        return proc, record
    except BaseException as exc:
        if proc is not None:
            kill_process_tree(proc) if executed or (released and not getattr(exc, "process_not_started", False)) else refuse_gate(proc)
            if proc.poll() is not None:  # the helper's receipt and custody settle now
                confirmed()
                we._forget_process(record)
        if not released and not isinstance(exc, (OwnerPauseRefused, StoreError)) and not hasattr(exc, "process_not_started"):
            exc.process_not_started = True
        raise
    finally:
        os.close(status_r)
