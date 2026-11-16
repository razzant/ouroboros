"""Native process targets retained by the public platform layer.

Capture pins an authorized process for later requests. A tracked POSIX child
also retains its waitable status so its group number cannot be recycled. A
retiring Windows Job's members stay held until each process has signalled.
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


def job_members_running(job_handle, held: dict) -> bool | None:
    """Windows: whether a member of the Job has not exited yet; None when the Job or a listed member is unreadable,
    even beside a held member still running: another member's state never answers for one this read could not hold.

    Each process the Job lists stays open in ``held`` (pid -> handle, closed by ``release_job_members``) once
    ``IsProcessInJob`` proves its PID still names a member, and runs until its process object signals: termination is
    asynchronous and only that signal proves its end (TerminateProcess's contract), never an exit code or the Job's
    list. So the caller reads once before terminating the Job (``QuitEngineStop``): every member held then is
    awaited, and one it could not hold stays unconfirmed however the others end."""
    from ouroboros import platform_layer as platform

    ctypes, kernel = platform.ctypes, platform._kernel32
    listed = platform._JOBOBJECT_BASIC_PROCESS_ID_LIST()
    if not kernel.QueryInformationJobObject(job_handle, platform._JOBOBJECTINFOCLASS_PROCESS_ID_LIST,
                                            ctypes.byref(listed), ctypes.sizeof(listed), None):
        return None
    unknown = False
    for pid in map(int, listed.ProcessIdList[:listed.NumberOfProcessIdsInList]):
        if pid in held:
            continue
        if not (handle := kernel.OpenProcess(0x00101000, False, pid)):  # SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFO
            unknown |= ctypes.get_last_error() != 87  # 87: the PID is already gone, nothing left to await
            continue
        member = ctypes.wintypes.BOOL()
        if (read := kernel.IsProcessInJob(handle, job_handle, ctypes.byref(member))) and member.value:
            held[pid] = handle
        else:  # unreadable, or the PID now names a process outside the Job: the member it named is gone
            unknown |= not read
            kernel.CloseHandle(handle)
    waits = {kernel.WaitForSingleObject(handle, 0) for handle in held.values()}  # 0: signalled; 0x102: timeout
    return None if unknown else True if 0x102 in waits else None if waits - {0} else False


def release_job_members(held: dict) -> None:
    """Close every handle ``job_members_running`` keeps in ``held`` (nothing held: no native call)."""
    if held:
        from ouroboros import platform_layer as platform

        while held:
            platform._kernel32.CloseHandle(held.popitem()[1])
