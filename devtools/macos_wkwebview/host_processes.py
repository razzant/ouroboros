"""Measured process and connection facts for the window-renewal instrument.

Identity follows the process-custody rule: a pid, its birth (``ps`` lstart) and the
measured parent chain — never a command line. Every helper returns None or "" when
its read fails: an unreadable fact stays missing, never a value.
"""
from __future__ import annotations

import os
import subprocess


def command(*argv: str) -> str | None:
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=15, check=False, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def ps(pid: int, column: str) -> str:
    return command("/bin/ps", "-o", f"{column}=", "-p", str(pid)) or ""


def alive(generation: dict) -> bool | None:
    """A zombie, a missing PID or a different birth ended this generation. A failed read is unknown."""
    done = subprocess.run(["/bin/ps", "-o", "stat=,lstart=", "-p", str(generation["pid"])],
                          capture_output=True, text=True, timeout=5)
    if done.returncode == 1 and not done.stdout.strip() and not done.stderr.strip():
        return False
    parts = done.stdout.strip().split(None, 1)
    if done.returncode or len(parts) != 2:
        return None
    return not parts[0].startswith("Z") and parts[1].strip() == generation["birth"]


def connection_owner(port: int) -> int | None:
    """The PID holding the client end of one loopback connection, measured with lsof.

    Only the exact local endpoint matches, so the stub's own accepted socket of the same
    connection is never the answer, and this process is excluded. None when unresolved:
    an unattributable request stays unattributed, never guessed from arrival order.
    """
    found = command("/usr/sbin/lsof", "-nP", "-Fpn", "-a", f"-iTCP@127.0.0.1:{port}", "-sTCP:ESTABLISHED")
    if found is None:
        return None
    pids, pid = set(), None
    for line in found.splitlines():
        if line.startswith("p"):
            try:
                pid = int(line[1:])
            except ValueError:
                pid = None
        elif pid is not None and pid != os.getpid() and line.startswith(f"n127.0.0.1:{port}->"):
            pids.add(pid)
    return pids.pop() if len(pids) == 1 else None


def process_ancestry(pid: int, limit: int = 12) -> list:
    """``[pid, parent, ...]`` up to pid 1 or the first unreadable hop: the measured chain."""
    chain, current = [], pid
    for _ in range(limit):
        chain.append(current)
        parent = ps(current, "ppid")
        if not parent:
            break
        try:
            current = int(parent.split()[0])
        except ValueError:
            break
        if current <= 1:
            break
    return chain
