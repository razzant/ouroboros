"""Native process targets retained by the public platform layer.

Capture pins an authorized process for later requests. A tracked POSIX child
also retains its waitable status so its group number cannot be recycled.
Signal operations stay in platform_layer; these witnesses own their lifetime.
"""

from __future__ import annotations

import os
import subprocess


def capture_process_stop_target(pid: int) -> dict:
    """Pin an already authorized process for a later wait-free stop request. The caller proves custody before
    AND after capture; a PID is not authority. Handles own their lifetime; Darwin's exit watch rejects a departed
    target. No attached-daemon Job is created: ordinary client close must preserve it."""
    from ouroboros import platform_layer as platform

    if pid <= 0 or pid == os.getpid():
        raise ValueError("invalid stop target")
    if platform.IS_WINDOWS:
        import _winapi
        handle = subprocess.Handle(_winapi.OpenProcess(0x1001, False, pid))
    elif platform.IS_MACOS:
        import select
        handle = select.kqueue()
        handle.control([select.kevent(pid, filter=select.KQ_FILTER_PROC,
                                     flags=select.KQ_EV_ADD | select.KQ_EV_CLEAR,
                                     fflags=select.KQ_NOTE_EXIT | select.KQ_NOTE_EXEC)], 0, 0)
    else:
        import io
        handle = io.FileIO(os.pidfd_open(pid), mode="rb", closefd=True)
    return {"pid": pid, "handle": handle, "pgid": platform.process_group_id(pid)}


def observe_child_exit(pid: int, flags: int = 0) -> tuple[int, int]:
    """waitpid-shaped status without reaping: keep the child's PID reserved.

    A tracked session leader stays waitable until its entire group is empty.
    ECHILD means that ownership witness was lost, never authority for killpg.
    """
    from ouroboros import platform_layer as platform

    if platform.IS_MACOS:
        from ouroboros.platform_darwin_abi import wait_child_status
        return wait_child_status(pid, nohang=bool(flags & os.WNOHANG))
    result = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT | (flags & os.WNOHANG))
    if result is None:
        return 0, 0
    return result.si_pid, (result.si_status << 8 if result.si_code == os.CLD_EXITED else result.si_status)


def retained_child_group(proc, *, release: bool = False, kill: bool = False) -> bool:
    """Use a retained, waitable session leader, never a saved numeric group alone.

    Request-only callers never wait for the retirement lock. Only this owner
    consumes status; an unexpected external reap invalidates the witness.
    A successful kill is a request, and a successful release consumes custody.
    """
    from ouroboros import platform_layer as platform

    if not proc._ouroboros_group_lock.acquire(blocking=False):
        return False
    try:
        if not proc._ouroboros_group_held:
            return False
        try:
            from ouroboros.process_identity import process_kernel_birth
            birth = proc._ouroboros_group_birth
            # Request-only: no ps fallback on an unreadable Linux /proc row.
            matches = (process_kernel_birth(proc.pid) == birth if platform.IS_MACOS else
                       str(platform._proc_start_ticks(proc.pid)) == str(birth).split(".", 1)[0])
            if not birth or not matches:
                return False  # external reaping/identity read failure never grants a new group
            pid, _ = platform.observe_child_exit(proc.pid, os.WNOHANG)
            if release:
                if pid != proc.pid:
                    return False
                os.waitpid(proc.pid, os.WNOHANG)
                proc._ouroboros_group_held = False
            elif kill:
                return platform.kill_process_group_id(proc.pid)
            return True
        except ChildProcessError:
            proc._ouroboros_group_held = False
            return False
        except OSError:
            return False
    finally:
        proc._ouroboros_group_lock.release()
