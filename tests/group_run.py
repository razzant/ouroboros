"""``subprocess.run`` for test commands whose timeout must stop every process they started.

``subprocess.run(timeout=...)`` kills only its direct child. ``node --test`` runs each
test file in a child process of its own, so a timed-out runner leaves that child alive
with ppid 1; on 16.10.2026 four such orphans of a hung test grew to 77-456 GB within
minutes and forced a hardware reset. Here the command starts in a new process group, and
a wait that ends in a timeout or any other exception kills the group and every
descendant before the exception propagates.

Outer supervisors reach these processes through the process tree, not through their
own group: ``kill_process_tree`` and ``scripts/safe_test.py --timeout`` both do.
"""
from __future__ import annotations

import subprocess
import time

from ouroboros.platform_layer import (
    force_kill_pid, kill_process_tree, pid_provably_gone, subprocess_new_group_kwargs,
)

# After the kill only an escaped descendant can still hold a pipe; never wait on it forever.
_DRAIN_SECONDS = 10


def run_in_group(args, *, timeout: float, cwd=None, env=None) -> subprocess.CompletedProcess:
    """Run ``args`` capturing UTF-8 text; on timeout kill the whole tree, then re-raise."""
    with subprocess.Popen(args, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          encoding="utf-8", errors="replace", **subprocess_new_group_kwargs()) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as expired:
            kill_process_tree(process)
            try:
                expired.stdout, expired.stderr = process.communicate(timeout=_DRAIN_SECONDS)
            except subprocess.TimeoutExpired:
                pass
            raise
        except BaseException:  # an interrupt or pytest-timeout's signal must not orphan the tree either
            kill_process_tree(process)
            raise
    return subprocess.CompletedProcess(args, process.returncode, stdout, stderr)


def kill_survivors(pids, *, within: float = 10.0) -> list[int]:
    """Give each PID ``within`` seconds to be provably gone; kill the rest by exact PID and return them."""
    deadline = time.monotonic() + within
    survivors = []
    for pid in pids:
        while not pid_provably_gone(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        if not pid_provably_gone(pid):
            survivors.append(pid)
            force_kill_pid(pid)
    return survivors
