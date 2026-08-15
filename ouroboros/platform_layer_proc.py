"""The Linux `/proc` surface and the boot-anchored clock.

The SECOND file of the platform layer, and it exists for a reason narrower than size:
everything here is Linux's process pseudo-filesystem and the boot identity read out of
it. `/proc` is not an API that exists everywhere and then behaves differently — on
macOS and Windows it is simply absent — so a caller that spells the path itself has
written a Linux-only assumption that looks like ordinary file access.

The confinement rule is UNCHANGED: no platform-exclusive API outside the platform
layer. The layer is now two named files rather than one, and
`tests/test_platform_guard.py` names both. What must NOT become two is the number of
readers of `/proc/sys/kernel/random/boot_id` — two readers means two places deciding
what an unreadable boot id means, and the last time that happened the copy decided
wrong (it kept building a fingerprint whose other fields are meaningless without the
anchor it had just failed to get). That literal appears exactly once, here, and the
guard asserts it.
"""

from __future__ import annotations

from typing import Any

from ouroboros.platform_flags import BOOT_IDENTITY_UNKNOWN, IS_LINUX, IS_WINDOWS

import os
import pathlib
import re
import subprocess
import sys
import time


def process_group_status(pgid: int) -> str:
    """Return ``alive``, ``gone``, or ``unknown`` for a process group."""

    if IS_WINDOWS:
        return "unknown"
    try:
        group_id = int(pgid)
        if group_id <= 0:
            return "unknown"
        os.killpg(group_id, 0)
        return "alive"
    except ProcessLookupError:
        return "gone"
    except (PermissionError, OSError, ValueError):
        return "unknown"


LINUX_PROC_ROOT = "/proc"


def has_proc_filesystem() -> bool:
    """Whether Linux's process pseudo-filesystem is present on this host.

    The one place `/proc` is named. Callers ask THIS instead of testing the literal
    themselves: a path that exists on one platform and not another is a platform
    API, and spelling it inline is how a Linux-only assumption gets to look like
    ordinary file access (the Platform Abstraction Rule, docs/DEVELOPMENT.md).
    """
    return os.path.isdir(LINUX_PROC_ROOT)


def proc_pid_path(pid: int, leaf: str) -> str:
    """Path to one `/proc/<pid>/<leaf>` node. Only meaningful where /proc exists."""
    return f"{LINUX_PROC_ROOT}/{int(pid)}/{leaf}"


def proc_entries() -> "list[str]":
    """Raw `/proc` directory entries (pid dirs plus kernel nodes); [] when absent."""
    try:
        return os.listdir(LINUX_PROC_ROOT)
    except OSError:
        return []


def _proc_start_ticks(pid: int) -> int:
    """Boot-relative start time (``/proc/<pid>/stat`` field 22), or 0 when it cannot be read."""
    try:
        with open(f"/proc/{int(pid)}/stat", "rb") as handle:
            fields = handle.read().rpartition(b")")[2].split()
        return int(fields[19]) if len(fields) >= 20 else 0  # rpartition dropped fields 1-2
    except (OSError, ValueError):
        return 0


_LINUX_BOOT_ID = pathlib.Path("/proc/sys/kernel/random/boot_id")


def boot_anchored_monotonic_ms() -> "tuple[str, int]":
    """`(boot identity, milliseconds)` on a clock a wall-clock step cannot move.

    Two properties are needed together and neither alone is enough. The clock must
    be MONOTONIC, so that an NTP correction or a manual date change cannot make a
    deadline fire early or late — a deadline written as `time.time() * 1000` and
    compared against a fresh `time.time()` does both. And it must be shared ACROSS
    PROCESSES on the same host, because the custodian lease deadline is written by
    execd and read by an independent watchdog process; a per-process epoch would make
    the two disagree about the same instant.

    On Linux that is `CLOCK_BOOTTIME` — which, unlike `CLOCK_MONOTONIC`, keeps
    counting across suspend, and a suspended host has not paused its own failure
    detection — paired with the kernel's boot id, which is what makes the anchor
    CHECKABLE: a value from a previous boot is meaningless on this one, and the
    identity says so instead of leaving a stale number to be believed. Elsewhere the
    identity is the literal `"non-linux"`, the convention `execd_state`'s process
    fingerprint already uses for a host it cannot identify that precisely, over
    `time.monotonic()` — per-boot and cross-process on the platforms Ouroboros runs
    Home on. execd targets are Linux in v1.
    """

    if sys.platform.startswith("linux"):
        try:
            boot_id = _LINUX_BOOT_ID.read_text(encoding="ascii").strip()
            if boot_id:
                return boot_id, int(time.clock_gettime(time.CLOCK_BOOTTIME) * 1000)
        except (OSError, ValueError, AttributeError):
            pass
    return BOOT_IDENTITY_UNKNOWN, int(time.monotonic() * 1000)


def process_fingerprint(pid: int) -> "dict[str, Any] | None":
    """Non-secret process-leader identity that survives an execd restart.

    Lives here, not with the custody state that consumes it, because every line of it
    is platform-specific: on Linux the identity comes out of `/proc` — the leader's
    `stat` row, the kernel boot id and the pid-namespace symlink — and everywhere else
    POSIX out of `ps`, with no answer at all on Windows. It was written beside its
    caller and the platform guard did not object, because a kernel pseudo-filesystem is
    an API reached by PATH and every rule that gate held was a list of forbidden NAMES.

    `None` means "cannot identify", never "not the same process": custody treats an
    absent fingerprint as unknown and falls back to a liveness check.
    """

    if IS_LINUX:
        try:
            stat_text = pathlib.Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
            fields = stat_text[stat_text.rfind(")") + 2 :].split()
            boot_id = _LINUX_BOOT_ID.read_text(encoding="ascii").strip()
            if not boot_id:
                # Without a boot anchor the rest is not a fingerprint: `start_ticks` is
                # measured FROM boot and a pid namespace inode can repeat across one, so
                # a match would claim a reused pgid is the original process.
                return None
            return {
                "boot_id": boot_id,
                "pid_namespace": os.readlink(f"/proc/{int(pid)}/ns/pid"),
                "leader_pid": int(pid),
                "pgrp": int(fields[2]),
                "session": int(fields[3]),
                "start_ticks": int(fields[19]),
            }
        except (OSError, ValueError, IndexError):
            return None
    if os.name != "posix":
        return None
    try:
        row = subprocess.run(
            ["ps", "-o", "lstart=", "-o", "pgid=", "-o", "sess=", "-p", str(pid)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        ).stdout.strip()
        match = re.fullmatch(r"(.+?)\s+(\d+)\s+(\d+)", row)
        if match is None:
            return None
        return {
            "boot_id": BOOT_IDENTITY_UNKNOWN,
            "pid_namespace": BOOT_IDENTITY_UNKNOWN,
            "leader_pid": int(pid),
            "pgrp": int(match.group(2)),
            "session": int(match.group(3)),
            "start_ticks": match.group(1),
        }
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


__all__ = [
    "LINUX_PROC_ROOT",
    "has_proc_filesystem",
    "proc_pid_path",
    "proc_entries",
    "_proc_start_ticks",
    "boot_anchored_monotonic_ms",
    "process_fingerprint",
    "process_group_status",
]
