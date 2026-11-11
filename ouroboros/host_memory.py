"""This host's physical memory: total and reclaimable bytes, and the kernel's own critical verdict.

The one reader the pool's admission of new processes (``supervisor.pool_admission``), the test runner's
worker budget (``test_parallelism``) and the overload stop of an owned test tree (``process_overload``)
share. A reading carries the host reserve (``config.HOST_MEMORY_RESERVE_FRACTION`` of the total) and the
room above it; each caller keeps its decision. Only macOS's fixed physical total is cached per process.

macOS: ``hw.memsize``, free + inactive + speculative pages from ``vm_stat``, and the kernel's
pressure level, of which only critical counts. Linux: ``/proc/meminfo`` within every memory limit
on this process's own cgroup path, v2 and v1 alike (inside a container MemTotal and MemAvailable
are the host's). Windows: ``psutil``; Job object and commit limits are not read. ``None`` means
unknown: an unsupported platform, a failed probe, or a limit that exists but cannot be read.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Optional, Sequence

_LINUX_ROOT = Path("/")  # where the Linux probe finds /proc and the cgroup mounts (a fake tree in tests)
# Per cgroup version: the limit file, the usage file, and the memory.stat key of its inactive file pages.
_FILES = {2: ("memory.max", "memory.current", "inactive_file"),
          1: ("memory.limit_in_bytes", "memory.usage_in_bytes", "total_inactive_file")}
# v2 spells "no limit" as "max"; v1 as LONG_MAX rounded down to the kernel page (LONG_MAX itself
# before Linux 3.19). No machine holds 4 EiB, so any limit from here up is that spelling.
_UNLIMITED = 2 ** 62


@dataclass(frozen=True)
class MemorySnapshot:
    total_bytes: int
    available_bytes: int
    pressure: bool = False  # the kernel itself reports memory pressure (macOS)

    @property
    def reserve_bytes(self) -> int:
        """What every admission leaves to the OS and other work: ``HOST_MEMORY_RESERVE_FRACTION`` of the total."""
        from ouroboros.config import HOST_MEMORY_RESERVE_FRACTION

        return int(self.total_bytes * HOST_MEMORY_RESERVE_FRACTION)

    @property
    def room_bytes(self) -> int:
        """Available memory above the reserve; 0 at or under it."""
        return max(0, self.available_bytes - self.reserve_bytes)


def _native_output(argv: Sequence[str]) -> str:
    # Bounded, same-call OS probes; no test process or daemon is started here.
    return subprocess.run(argv, check=True, capture_output=True, text=True,
                          encoding="utf-8", timeout=2).stdout


def _darwin_pressure() -> bool:
    """The kernel's own critical verdict (levels: 1 normal, 2 warn, 4 critical). Free + inactive
    + speculative barely moves under pressure: XNU keeps refilling the inactive queue with
    anonymous pages that it then compresses. Warn is routine on a busy 8 GB Mac running a
    healthy battery, so only critical counts. An unreadable level reads as normal, silently;
    the free-memory reading still applies."""
    try:
        return int(_native_output(["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"]).strip()) >= 4
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


@lru_cache(maxsize=1)
def _darwin_total() -> int:
    """Physical RAM is fixed for this process; a failed probe remains retryable."""
    return int(_native_output(["/usr/sbin/sysctl", "-n", "hw.memsize"]).strip())


def _cgroup_mounts(root: Path) -> dict:
    """``{version: [(mount root, mount point)]}`` of every cgroup v2 mount and every v1 mount that carries
    the memory controller, from ``/proc/self/mountinfo`` (its octal escapes undone, proc(5))."""
    mounts = {1: [], 2: []}
    for line in (root / "proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
        head, _, tail = line.partition(" - ")
        fields, kind = head.split(), tail.split()
        if len(fields) >= 5 and len(kind) >= 3 and (
                kind[0] == "cgroup2" or (kind[0] == "cgroup" and "memory" in kind[2].split(","))):
            mounts[2 if kind[0] == "cgroup2" else 1].append(tuple(
                re.sub(r"\\([0-7]{3})", lambda code: chr(int(code.group(1), 8)), field) for field in fields[3:5]))
    return mounts


def _beneath(path: str, top: str) -> Optional[tuple]:
    """The components of cgroup ``path`` below a mount's root ``top``; None when that mount does not show it."""
    parts, prefix = PurePosixPath(path).parts, PurePosixPath(top).parts
    return parts[len(prefix):] if parts[:len(prefix)] == prefix and ".." not in parts else None


def _limit_room(directory: Path, version: int) -> Optional[tuple]:
    """``(limit, room)`` when this cgroup sets a finite memory limit; None when its limit file is absent
    or says unlimited. Room is the limit less the usage beyond inactive file pages, which reclaim as
    MemAvailable counts them. An existing limit or usage that cannot be read raises."""
    limit_name, usage_name, inactive = _FILES[version]
    try:
        text = (directory / limit_name).read_text(encoding="ascii").strip()
    except FileNotFoundError:
        return None
    if text == "max" or (limit := int(text)) >= _UNLIMITED:
        return None
    used = int((directory / usage_name).read_text(encoding="ascii"))
    try:
        stat = dict(line.split() for line in (directory / "memory.stat").read_text(
            encoding="ascii").splitlines() if len(line.split()) == 2)
    except OSError:
        stat = {}  # without the stat file the cache is not credited
    return limit, max(0, limit - max(0, used - int(stat.get(inactive, 0))))


def _cgroup_limits(root: Path) -> list:
    """``(limit, room)`` of every finite memory limit from this process's own cgroup up to the top its
    mount shows, in each hierarchy that carries memory. A hierarchy not mounted here shows no limit;
    one mounted where this cgroup is not visible raises, as an unreadable limit does."""
    try:
        membership = (root / "proc/self/cgroup").read_text(encoding="utf-8")
    except FileNotFoundError:
        return []  # a kernel without cgroups limits nothing
    mounts, limits = _cgroup_mounts(root), []
    for line in membership.splitlines():
        _, controllers, path = line.split(":", 2)
        version = 1 if controllers else 2
        if version == 1 and "memory" not in controllers.split(","):
            continue
        shown = [(point, below) for top, point in mounts[version] if (below := _beneath(path, top)) is not None]
        if not shown:
            if mounts[version]:
                raise ValueError(f"cgroup {path!r} lies outside every mount of its hierarchy")
            continue
        point, below = shown[0]
        for depth in range(len(below), -1, -1):
            if (bound := _limit_room((root / point.lstrip("/")).joinpath(*below[:depth]), version)) is not None:
                limits.append(bound)
    return limits


def physical_memory() -> Optional[MemorySnapshot]:
    """Read total and reclaimable physical memory using native OS facilities."""
    pressure = False
    try:
        if sys.platform == "win32":
            import psutil  # The packaged Windows runtime includes psutil.
            memory = psutil.virtual_memory()
            total, available = memory.total, memory.available
        elif sys.platform == "darwin":
            total = _darwin_total()
            vm = _native_output(["/usr/bin/vm_stat"])
            page_size = int(re.search(r"page size of (\d+) bytes", vm).group(1))
            pages = {name.strip(): int(value.strip().rstrip("."))
                     for name, value in (line.split(":", 1) for line in vm.splitlines()[1:] if ":" in line)}
            # Admission counts inactive pages as room, anonymous ones included, which
            # overstates it under pressure; the overload stop also reads the kernel's
            # verdict. Purgeable is already inside active/inactive, so it is not added.
            available = page_size * sum(pages[name] for name in
                                        ("Pages free", "Pages inactive", "Pages speculative"))
            pressure = _darwin_pressure()
        elif sys.platform.startswith("linux"):
            fields = {name: int(value.split()[0]) * 1024 for name, value in (line.split(":", 1) for line in (
                _LINUX_ROOT / "proc/meminfo").read_text(encoding="utf-8").splitlines())}
            limits = _cgroup_limits(_LINUX_ROOT)
            total = min([fields["MemTotal"], *(limit for limit, _ in limits)])
            available = min([fields["MemAvailable"], *(room for _, room in limits)])
        else:
            return None
        if total <= 0 or available < 0:
            return None
        return MemorySnapshot(total, min(total, available), pressure)
    except (OSError, ValueError, LookupError, AttributeError, ImportError, subprocess.SubprocessError):
        return None
