"""Invocation-scoped POSIX cleanup after a ProcessContainer owner dies.

The guardian is outside the workload's session. Only the owner holds the pipe's
write end; EOF survives SIGKILL and never depends on a Python finally block.
Windows needs no guardian: ProcessContainer's non-inheritable kill-on-close Job
handle is closed by the kernel on owner death, including TerminateProcess.
This is supervision, not a sandbox: a detached child that discards its token
before observation, or killing both owner and guardian, can defeat POSIX cleanup.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import select
import subprocess
import sys
from types import SimpleNamespace

from ouroboros import platform_layer as pl

# Bounds on the guardian protocol/exit, not on the workload's useful lifetime.
_HANDSHAKE_SEC = 10.0
_GUARDIAN_EXIT_SEC = 30.0


def kill_attributed_tree(token: str, root: int = 0, birth: str = "", *, graceful: bool = False) -> None:
    """End the measured root's tree, then token-bearing detached descendants.

    Neither a stale pid/pgid nor a command-name match grants signal authority.
    POSIX still has the unavoidable exit/reuse race between validation and kill;
    the existing container's survivor scan reports unconfirmed cleanup.
    ``graceful`` sends SIGTERM to the same targets instead: a request, never an
    exit, so the caller still forces and verifies (Windows: no-op).
    """
    from ouroboros.process_containment import (
        ENV_ASSIGNMENT_PRESENT, pid_environment_assignment_state, pids_with_env_marker,
    )

    if root > 0 and birth and pl.process_start_time(root) == birth:
        if not graceful:
            pl.kill_process_tree(SimpleNamespace(pid=root))
        else:
            if pl.process_group_id(root) == root:  # spawn made it a session leader
                pl.terminate_process_group_id(root)
            for pid in (*pl.collect_descendant_pids(root), root):
                pl.deliver_signal(pid, graceful=True)
    members = pids_with_env_marker(token)
    if members is None:
        raise RuntimeError("lifeline could not enumerate the contained process tree")
    for pid in members:
        if pid != os.getpid() and pid_environment_assignment_state(pid, token, "1") == ENV_ASSIGNMENT_PRESENT:
            if graceful:
                pl.deliver_signal(pid, graceful=True)
            else:
                pl.force_kill_pid(pid)


class ProcessLifeline:
    """An owner-held pipe; release only after the workload has been reaped."""

    def __init__(self):
        from ouroboros.process_containment import ProcessContainer

        # This container owns the guardian itself, with a DIFFERENT token. The
        # watched token travels over stdin, never argv/environment (which would
        # make the guardian attribute itself or its ps helpers to the workload).
        self._container = ProcessContainer()
        self._proc = None

    def start(self, token: str) -> None:
        """Raises with the guardian possibly running: the owner's reap ``finish``es either way,
        so what that teardown found reaches the owner instead of dying with this frame."""
        self._proc = self._container.spawn(
            [sys.executable, "-I", "-B", "-c",
             "import sys; sys.path.insert(0, sys.argv[1]); "
             "from ouroboros.process_lifeline import _guardian_main; _guardian_main()",
             str(Path(__file__).resolve().parents[1])],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        )
        ready, _, _ = select.select([self._proc.stdout], [], [], _HANDSHAKE_SEC)
        if not ready or self._proc.stdout.read(1) != b"R":
            raise RuntimeError("process lifeline failed to start")
        self._proc.stdout.close()
        self._send({"token": token})

    def _send(self, message: dict) -> None:
        self._proc.stdin.write((json.dumps(message) + "\n").encode("utf-8"))
        self._proc.stdin.flush()

    def attach(self, pid: int, birth: str, pgid: int, since: int) -> None:
        self._send({"root": pid, "birth": birth, "pgid": pgid, "since": since})

    def finish(self, *, released: bool) -> str:
        proc, self._proc = self._proc, None
        if proc is None:
            return self._container.reap()  # a guardian spawn that raised answered there
        error = ""
        gone = False  # EPIPE: the guardian's read end is closed, so it has already exited
        try:
            if proc.stdin is not None and not proc.stdin.closed:
                try:
                    if released:
                        proc.stdin.write(b'{"release":true}\n')
                        proc.stdin.flush()
                except BrokenPipeError:
                    gone = True
                except OSError as exc:
                    error = f"process lifeline lost: {exc}"
                finally:
                    try:
                        proc.stdin.close()  # retries a release still buffered after EPIPE
                    except BrokenPipeError:
                        gone = True
            # EOF asks for cleanup; normal release asks for stand-down. Wait
            # either way, so successful test runs never accumulate guardians.
            try:
                code = proc.wait(timeout=_GUARDIAN_EXIT_SEC)
                # A guardian stopped from outside (a harness ending the whole tree)
                # before a release protected nothing more: the owner has itself
                # reaped and confirmed the workload. Its own container is reaped below.
                if code and not (gone and released):
                    error = error or f"process lifeline exited {code}"
            except subprocess.TimeoutExpired:
                error = "process lifeline did not exit after pipe closure"
        finally:
            error = self._container.reap() or error
            self._container.close()
            if proc.stdout is not None:
                proc.stdout.close()
            proc.wait(timeout=5)
        return error


def _guardian_main() -> None:
    """Private pipe protocol: initialize, optionally identify root, release/EOF."""
    token, root, birth, pgid, since = "", 0, "", 0, 0
    sys.stdout.buffer.write(b"R")
    sys.stdout.buffer.flush()
    for line in sys.stdin.buffer:
        try:
            record = json.loads(line)
        except (ValueError, UnicodeError):
            # Owner may have died during its final write. Keep the last complete
            # ownership record; the token was delivered before workload spawn.
            continue
        if record.get("release"):
            return
        token = record.get("token", token)
        root, birth = record.get("root", root), record.get("birth", birth)
        pgid, since = record.get("pgid", pgid), record.get("since", since)
    if token:
        kill_attributed_tree(token, root, birth)
        # Reuse the same bounded verification as a live owner, including
        # detached grandchildren created while the first tree sweep ran.
        from ouroboros.process_containment import ProcessContainer

        container = ProcessContainer()
        container._token, container._root = token, root
        container._pgid, container._start_ticks = pgid, since
        error = container.reap()
        if error:
            print(f"PROCESS_LIFELINE_UNCONFIRMED {error}", file=sys.stderr, flush=True)
            raise SystemExit(1)
