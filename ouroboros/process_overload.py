"""Sample an owned test tree and stop it when the machine runs short of physical memory.

RSS is summed (shared pages may be counted more than once), so tree size alone
never stops a run: the machine's free memory, and on macOS its kernel's critical
pressure verdict, is the signal. This is supervision, not an OS memory/PID/
file-descriptor quota. Rapid growth between samples can outrun it, a host too
loaded to run its probes skips the stop loudly, and an unreadable macOS pressure
level reads as normal.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from ouroboros import platform_layer as pl


@dataclass(frozen=True)
class MemberMemory:
    pid: int
    rss_bytes: int
    name: str


def _in_job(pid: int, job) -> bool:
    """Windows membership comes from the held Job, including reparented members."""
    import ctypes
    from ctypes import wintypes

    def read(handle):
        answer = wintypes.BOOL()
        if not pl._kernel32.IsProcessInJob(handle, job, ctypes.byref(answer)):
            raise OSError(ctypes.get_last_error(), "IsProcessInJob failed")
        return bool(answer.value)

    error, member = pl._windows_query(pid, read)
    if error == 87:  # The process exited while the process table was read.
        return False
    if error:
        return False  # An inaccessible stranger is not an attributed member.
    return bool(member)


def token_members(container) -> set[int]:
    """Members that carry the container's token, detached ones included: one whole-table scan."""
    from ouroboros.process_containment import pids_with_env_marker

    pids = pids_with_env_marker(container._token)
    if pids is None:
        raise RuntimeError("contained process membership is unavailable")
    return set(pids)


def member_memory(container, detached=None) -> list[MemberMemory]:
    """Read live RSS only for the container's Job, token or measured tree.

    ``detached`` reuses the last token scan; the measured tree is re-read every sample.
    """
    if pl.IS_WINDOWS:
        import psutil

        if container._job is None:
            return []
        members = []
        for proc in psutil.process_iter(["pid", "name", "memory_info"]):
            try:
                if _in_job(proc.pid, container._job):
                    info = proc.info["memory_info"]
                    if info is None:
                        raise RuntimeError(f"RSS unavailable for Job member {proc.pid}")
                    members.append(MemberMemory(proc.pid, info.rss, proc.info["name"] or "unknown"))
            except psutil.NoSuchProcess:
                continue
        return members

    owned = set(token_members(container) if detached is None else detached)
    if container._birth and pl.process_start_time(container._root) == container._birth:
        owned.update([container._root, *pl.collect_descendant_pids(container._root)])
    owned.discard(os.getpid())
    if os.path.isdir("/proc"):  # Linux: slim images ship no ps; the kernel table is authoritative
        page, members = os.sysconf("SC_PAGE_SIZE"), []
        for pid in owned:
            try:
                resident = int(Path(f"/proc/{pid}/statm").read_text(encoding="ascii").split()[1])
                name = Path(f"/proc/{pid}/comm").read_text(encoding="utf-8", errors="replace").strip()
            except (OSError, ValueError, IndexError):
                continue  # exited between enumeration and the read
            members.append(MemberMemory(pid, resident * page, name))
        return members
    # One ps RSS table per sample; psutil is intentionally a Windows-only dependency.
    result = subprocess.run(["ps", "-A", "-o", "pid=,rss=,comm="],
                            capture_output=True, text=True, timeout=5, check=True)
    members = []
    for line in result.stdout.splitlines():
        fields = line.split(maxsplit=2)
        if len(fields) == 3 and int(fields[0]) in owned:
            members.append(MemberMemory(int(fields[0]), int(fields[1]) * 1024, fields[2]))
    return members


def stop_reason(total: int, memory, *, floor: int, reserve: int) -> str:
    """Why an owned tree must stop now, or "".

    Only real pressure stops a tree that outgrew what admission gave it: the machine
    is under its physical reserve, or its kernel reports critical memory pressure. A tree
    within its admitted floor is never killed for another program's pressure, and a
    large tree on a machine that keeps its reserve runs on (a start-of-run
    free-memory snapshot and summed RSS both misjudge room, which killed healthy
    two-worker runs on 7 GB hosts).
    """
    if memory is None or total <= floor:
        return ""
    if memory.available_bytes < reserve:
        return "machine under its reserve and tree past its admitted size"
    if memory.pressure:
        return "kernel reports critical memory pressure and tree past its admitted size"
    return ""


class OverloadWatch:
    """A joined owner thread; a stop always survives as a nonempty reap error."""

    def __init__(self, container, slots: int = 1):
        from ouroboros.config import TEST_PROCESS_SAMPLE_SEC, TEST_WORKER_PEAK_RSS_BYTES

        # The admitted workers and their controller always fit, so admission never
        # starts a tree that this watch would then kill.
        self.floor_bytes = (max(1, slots) + 1) * TEST_WORKER_PEAK_RSS_BYTES
        self.degraded = ""
        self._container = container
        self._interval = TEST_PROCESS_SAMPLE_SEC
        self._done = threading.Event()
        self._thread = None
        self.diagnostic = ""

    def start(self) -> None:
        thread = threading.Thread(target=self._run, name="test-overload-watch", daemon=True)
        thread.start()
        self._thread = thread  # A failed start must never leave an unjoinable thread.

    def _run(self) -> None:
        from ouroboros.config import TEST_MEMORY_RESERVE_FRACTION, TEST_PROCESS_TOKEN_SCAN_SEC
        from ouroboros.test_parallelism import physical_memory

        detached, scanned = None, None
        while not self._done.is_set():
            try:
                if not pl.IS_WINDOWS and (scanned is None or time.monotonic() - scanned >= TEST_PROCESS_TOKEN_SCAN_SEC):
                    detached, scanned = token_members(self._container), time.monotonic()
                members = member_memory(self._container, detached)
                memory = physical_memory()
                if memory is None:  # without the machine's free memory there is no pressure signal
                    raise RuntimeError("the machine's physical memory is unreadable")
            except Exception as exc:
                # A sample this host cannot take is skipped (said once); it never fails a
                # healthy run or ends the watch. The tree is still owned and reaped.
                if not self.degraded:
                    self.degraded = f"overload watch sample failed, the memory stop skips it: {exc}"
                    print(self.degraded, file=sys.stderr, flush=True)
            else:
                total = sum(member.rss_bytes for member in members)
                reserve = int(memory.total_bytes * TEST_MEMORY_RESERVE_FRACTION)
                why = stop_reason(total, memory, floor=self.floor_bytes, reserve=reserve)
                if members and why:
                    largest = max(members, key=lambda member: member.rss_bytes)
                    self.diagnostic = (
                        f"overload_stop: {why}; tree_rss={total} floor={self.floor_bytes} "
                        f"available={memory.available_bytes} reserve={reserve}; "
                        f"largest pid={largest.pid} name={largest.name!r} rss={largest.rss_bytes}"
                    )
                    try:
                        self._container.stop_tree()
                    except Exception as exc:
                        self.diagnostic += f"; tree stop unconfirmed: {exc}"
                    return
            if self._done.wait(self._interval):
                return

    def stop(self) -> str:
        self._done.set()
        if self._thread is not None:
            self._thread.join()  # all process-table/memory probes have bounded waits
        return self.diagnostic
