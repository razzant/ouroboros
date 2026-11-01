"""The Win32 calls the qualification needs (ctypes, stdlib only; imported on Windows only).

Adapted from the parent-reviewed NTFS probe (``devtools/update_ntfs_probe/winapi.py`` at 49168beac):
its kill-on-close Job, accounting, process list and volume reads, plus process identity by an open
handle (creation FILETIME, image, command line, parent, exit code). Identity is always read through
one handle that is held across the reads, so a recycled PID cannot answer for the process we named.
"""
import ctypes as c
from ctypes import wintypes as w

PROCESS_QUERY_LIMITED_INFORMATION, SYNCHRONIZE, PROCESS_TERMINATE = 0x1000, 0x00100000, 0x0001
KILL_ON_JOB_CLOSE = 0x2000  # no BREAKAWAY_OK, no SILENT_BREAKAWAY_OK: nothing leaves this Job
STILL_ACTIVE = 259
STATUS_INFO_LENGTH_MISMATCH = 0xC0000004
# OpenProcess answers ERROR_INVALID_PARAMETER for a PID that no process holds: the only "absent".
# ERROR_ACCESS_DENIED (a protected process) or any other refusal is unobserved, never "dead".
ERROR_ACCESS_DENIED, ERROR_INVALID_PARAMETER, ERROR_NO_MORE_FILES = 5, 87, 18
CREATE_SUSPENDED, CREATE_NO_WINDOW = 0x4, 0x08000000


class Limits(c.Structure):
    _fields_ = [("process_time", c.c_int64), ("job_time", c.c_int64), ("flags", w.DWORD),
                ("min_ws", c.c_size_t), ("max_ws", c.c_size_t), ("active_limit", w.DWORD),
                ("affinity", c.c_size_t), ("priority", w.DWORD), ("scheduling", w.DWORD)]


class ExtendedLimits(c.Structure):
    _fields_ = [("basic", Limits), ("io", c.c_uint64 * 6), ("memory", c.c_size_t * 4)]


class Accounting(c.Structure):
    _fields_ = [("times", c.c_int64 * 4), ("faults", w.DWORD), ("total", w.DWORD), ("active", w.DWORD),
                ("terminated", w.DWORD)]


SLOTS = 1024


class ProcessList(c.Structure):
    _fields_ = [("assigned", w.DWORD), ("count", w.DWORD), ("pids", c.c_size_t * SLOTS)]


class UnicodeString(c.Structure):
    _fields_ = [("length", w.USHORT), ("maximum", w.USHORT), ("buffer", c.c_void_p)]


class ProcessEntry(c.Structure):
    _fields_ = [("size", w.DWORD), ("usage", w.DWORD), ("pid", w.DWORD), ("heap", c.c_size_t),
                ("module", w.DWORD), ("threads", w.DWORD), ("ppid", w.DWORD), ("priority", w.LONG),
                ("flags", w.DWORD), ("exe", w.WCHAR * 260)]


class Win:
    def __init__(self):
        k = c.WinDLL("kernel32", use_last_error=True)
        n = c.WinDLL("ntdll")
        H, D, P, S, B = w.HANDLE, w.DWORD, c.c_void_p, w.LPCWSTR, w.BOOL
        signatures = {
            "CloseHandle": (B, H), "CreateJobObjectW": (H, P, S),
            "SetInformationJobObject": (B, H, c.c_int, P, D),
            "QueryInformationJobObject": (B, H, c.c_int, P, D, P),
            "AssignProcessToJobObject": (B, H, H), "TerminateJobObject": (B, H, w.UINT),
            "IsProcessInJob": (B, H, H, c.POINTER(B)), "OpenProcess": (H, D, B, D),
            "QueryFullProcessImageNameW": (B, H, D, w.LPWSTR, c.POINTER(D)),
            "GetProcessTimes": (B, H, *(c.POINTER(w.FILETIME),) * 4),
            "GetExitCodeProcess": (B, H, c.POINTER(D)),
            "CreateToolhelp32Snapshot": (H, D, D), "Process32FirstW": (B, H, P), "Process32NextW": (B, H, P),
            "GetVolumePathNameW": (B, S, w.LPWSTR, D),
            "GetVolumeInformationW": (B, S, w.LPWSTR, D, P, P, P, w.LPWSTR, D),
        }
        for name, signature in signatures.items():
            fn = getattr(k, name)
            fn.restype, fn.argtypes = signature[0], signature[1:]
            setattr(self, name, fn)
        self.NtResumeProcess = n.NtResumeProcess
        self.NtResumeProcess.restype, self.NtResumeProcess.argtypes = c.c_int32, (H,)
        self.NtQueryInformationProcess = n.NtQueryInformationProcess
        self.NtQueryInformationProcess.restype = c.c_int32
        self.NtQueryInformationProcess.argtypes = (H, c.c_int, P, w.ULONG, c.POINTER(w.ULONG))

    @staticmethod
    def check(result):
        if not result:
            raise c.WinError(c.get_last_error())
        return result

    def close(self, handle):
        self.check(self.CloseHandle(handle))

    # ----------------------------------------------------------------- the harness Job
    def job(self):
        handle = self.check(self.CreateJobObjectW(None, None))
        limits = ExtendedLimits()
        limits.basic.flags = KILL_ON_JOB_CLOSE
        try:
            self.check(self.SetInformationJobObject(handle, 9, c.byref(limits), c.sizeof(limits)))
        except BaseException:
            self.close(handle)
            raise
        return handle

    def assign(self, job, process_handle):
        self.check(self.AssignProcessToJobObject(job, process_handle))

    def resume(self, process_handle):
        status = self.NtResumeProcess(process_handle)
        if status != 0:
            raise OSError(f"NtResumeProcess returned NTSTATUS 0x{status & 0xFFFFFFFF:08x}")

    def active(self, job):
        info = Accounting()
        self.check(self.QueryInformationJobObject(job, 1, c.byref(info), c.sizeof(info), None))
        return int(info.active)

    def pids(self, job):
        info = ProcessList()
        self.check(self.QueryInformationJobObject(job, 3, c.byref(info), c.sizeof(info), None))
        if info.assigned != info.count or info.count > SLOTS:
            raise RuntimeError(f"job process list incomplete ({info.count}/{info.assigned})")
        return [int(pid) for pid in info.pids[:info.count]]

    def terminate(self, job, code):
        self.check(self.TerminateJobObject(job, code))

    # ----------------------------------------------------------------- process identity
    def open(self, pid, access=PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE):
        """``(handle, 0)`` or ``(None, Win32 error)``; the error is read at once, before any other call."""
        handle = self.OpenProcess(access, False, int(pid))
        return (handle, 0) if handle else (None, c.get_last_error())

    def birth(self, handle):
        times = [w.FILETIME() for _ in range(4)]
        self.check(self.GetProcessTimes(handle, *[c.byref(t) for t in times]))
        return str((times[0].dwHighDateTime << 32) | times[0].dwLowDateTime)

    def image(self, handle):
        buf, size = c.create_unicode_buffer(32768), w.DWORD(32768)
        self.check(self.QueryFullProcessImageNameW(handle, 0, buf, c.byref(size)))
        return buf.value

    def exit_code(self, handle):
        code = w.DWORD()
        self.check(self.GetExitCodeProcess(handle, c.byref(code)))
        return None if code.value == STILL_ACTIVE else int(code.value)

    def cmdline(self, handle):
        """ProcessCommandLineInformation (60, Windows 8.1+); ``None`` when the system refuses it."""
        size = w.ULONG(65536)
        for _ in range(3):
            buf = c.create_string_buffer(size.value)
            status = self.NtQueryInformationProcess(handle, 60, buf, size.value, c.byref(size)) & 0xFFFFFFFF
            if status == STATUS_INFO_LENGTH_MISMATCH:
                continue
            if status != 0:
                return None
            text = UnicodeString.from_buffer(buf)
            return c.wstring_at(text.buffer, text.length // 2) if text.buffer else ""
        return None

    def in_job(self, handle, job):
        member = w.BOOL()
        self.check(self.IsProcessInJob(handle, job, c.byref(member)))
        return bool(member.value)

    def identity(self, pid, job=None, parents=None):
        """Everything one handle says about ``pid``. ``alive`` is ``True``/``False`` only when measured: an
        absent PID or an exit code read from the handle. A refused open or a failed read is ``None``. The exit
        code comes before the image: an exited process kept open by any handle (a holder's own ``Popen``) still
        opens by PID but no longer answers for its image, and that refusal must not hide the exit it does answer."""
        handle, error = self.open(pid)
        if handle is None:
            return {"pid": int(pid), "alive": False if error == ERROR_INVALID_PARAMETER else None,
                    "absent": error == ERROR_INVALID_PARAMETER, "open_error": error}
        try:
            row = {"pid": int(pid), "birth": self.birth(handle), "image": None, "exit_code": self.exit_code(handle),
                   "cmdline": None}
            row["alive"] = row["exit_code"] is None
            if row["alive"]:
                row.update(image=self.image(handle), cmdline=self.cmdline(handle))
                if job is not None:
                    row["in_job"] = self.in_job(handle, job)
            if parents is not None:
                row["ppid"] = parents.get(int(pid))
            return row
        except OSError as exc:
            return {"pid": int(pid), "alive": None, "read_error": repr(exc)}
        finally:
            self.CloseHandle(handle)

    def snapshot(self):
        """Every process of the system: ``{pid: (ppid, exe name)}`` (Toolhelp; names only)."""
        snap = self.CreateToolhelp32Snapshot(0x2, 0)
        if snap in (None, c.c_void_p(-1).value):
            raise c.WinError(c.get_last_error())
        try:
            entry, rows = ProcessEntry(), {}
            entry.size = c.sizeof(ProcessEntry)
            step = self.Process32FirstW
            while step(snap, c.byref(entry)):
                rows[int(entry.pid)] = (int(entry.ppid), entry.exe)
                step = self.Process32NextW
            error = c.get_last_error()
            if error != ERROR_NO_MORE_FILES:  # the walk failed: an incomplete census is no census
                raise c.WinError(error)
            return rows
        finally:
            self.close(snap)

    def filesystem(self, path):
        volume, fs = c.create_unicode_buffer(32768), c.create_unicode_buffer(261)
        self.check(self.GetVolumePathNameW(str(path), volume, len(volume)))
        self.check(self.GetVolumeInformationW(volume.value, None, 0, None, None, None, fs, len(fs)))
        return {"volume": volume.value, "filesystem": fs.value}
