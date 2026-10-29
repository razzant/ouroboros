"""Only the Win32 calls needed by the disposable NTFS experiment (Python 3.10)."""
import ctypes as c
from ctypes import wintypes as w
import ntpath
import struct

MOUNT_POINT = 0xA0000003


def mount_buffer(target):
    target = str(target)
    if not ntpath.isabs(target) or len(ntpath.splitdrive(target)[0]) != 2:
        raise ValueError("junction target must be an absolute local drive path")
    substitute = ("\\??\\" + target).encode("utf-16-le")
    printed = target.encode("utf-16-le")
    body = struct.pack("<HHHH", 0, len(substitute), len(substitute) + 2, len(printed))
    body += substitute + b"\0\0" + printed + b"\0\0"
    if len(body) + 8 > 16384:
        raise ValueError("reparse buffer too long")
    return struct.pack("<IHH", MOUNT_POINT, len(body), 0) + body


class Limits(c.Structure):
    _fields_ = [("process_time", c.c_int64), ("job_time", c.c_int64),
                ("flags", w.DWORD), ("min_ws", c.c_size_t), ("max_ws", c.c_size_t),
                ("active_limit", w.DWORD), ("affinity", c.c_size_t),
                ("priority", w.DWORD), ("scheduling", w.DWORD)]


class ExtendedLimits(c.Structure):
    _fields_ = [("basic", Limits), ("io", c.c_uint64 * 6),
                ("memory", c.c_size_t * 4)]


class Accounting(c.Structure):
    _fields_ = [("times", c.c_int64 * 4), ("faults", w.DWORD),
                ("total", w.DWORD), ("active", w.DWORD), ("terminated", w.DWORD)]


class ProcessList(c.Structure):
    _fields_ = [("assigned", w.DWORD), ("count", w.DWORD), ("pids", c.c_size_t * 64)]


class Win:
    def __init__(self):
        k = c.WinDLL("kernel32", use_last_error=True)
        H, D, P, S, B = w.HANDLE, w.DWORD, c.c_void_p, w.LPCWSTR, w.BOOL
        signatures = {
            "CreateFileW": (H, S, D, D, P, D, D, H),
            "CloseHandle": (B, H),
            "DeviceIoControl": (B, H, D, P, D, P, D, c.POINTER(D), P),
            "GetFinalPathNameByHandleW": (D, H, w.LPWSTR, D, D),
            "GetModuleHandleW": (H, S),
            "GetModuleFileNameW": (D, H, w.LPWSTR, D),
            "GetCurrentProcess": (H,),
            "K32GetMappedFileNameW": (D, H, P, w.LPWSTR, D),
            "CreateJobObjectW": (H, P, S),
            "SetInformationJobObject": (B, H, c.c_int, P, D),
            "QueryInformationJobObject": (B, H, c.c_int, P, D, P),
            "AssignProcessToJobObject": (B, H, H),
            "TerminateJobObject": (B, H, w.UINT),
            "IsProcessInJob": (B, H, H, c.POINTER(B)),
            "OpenProcess": (H, D, B, D),
            "QueryFullProcessImageNameW": (B, H, D, w.LPWSTR, c.POINTER(D)),
            "GetVolumePathNameW": (B, S, w.LPWSTR, D),
            "GetVolumeInformationW": (B, S, w.LPWSTR, D, P, P, P, w.LPWSTR, D),
        }
        for name, signature in signatures.items():
            fn = getattr(k, name)
            fn.restype, fn.argtypes = signature[0], signature[1:]
            setattr(self, name, fn)

    def check(self, result):
        if not result:
            raise c.WinError(c.get_last_error())
        return result

    def close(self, handle):
        self.check(self.CloseHandle(handle))

    def open(self, path, access=0, reparse=False):
        # SHARE_READ | SHARE_WRITE | SHARE_DELETE; the handle is short-lived.
        h = self.CreateFileW(str(path), access, 7, None, 3,
                             0x02000000 | (0x00200000 if reparse else 0), None)
        if h == c.c_void_p(-1).value:
            raise c.WinError(c.get_last_error())
        return self.check(h)

    def final_path(self, path, device=False):
        h = self.open(path)
        try:
            buf = c.create_unicode_buffer(32768)
            n = self.check(self.GetFinalPathNameByHandleW(h, buf, len(buf), 2 if device else 0))
            if n >= len(buf):
                raise ValueError("final path truncated")
            return buf.value
        finally:
            self.close(h)

    def reparse(self, path, target=None):
        h = self.open(path, 0x40000000 if target is not None else 0, reparse=True)
        try:
            got = w.DWORD()
            if target is not None:
                data = mount_buffer(target)
                buf = c.create_string_buffer(data, len(data))
                self.check(self.DeviceIoControl(h, 0x900A4, buf, len(data), None, 0, c.byref(got), None))
                return None
            buf = c.create_string_buffer(16384)
            self.check(self.DeviceIoControl(h, 0x900A8, None, 0, buf, len(buf), c.byref(got), None))
            return buf.raw[:got.value].hex()
        finally:
            self.close(h)

    def junction(self, path, target):
        path.mkdir()  # must be absent; never reuse an arbitrary directory
        self.reparse(path, target)

    def module(self, path=None):
        handle = None if path is None else self.check(self.GetModuleHandleW(str(path)))
        buf = c.create_unicode_buffer(32768)
        n = self.check(self.GetModuleFileNameW(handle, buf, len(buf)))
        if n >= len(buf):
            raise ValueError("module path truncated")
        result = {"loader_path": buf.value}
        if handle:
            n = self.check(self.K32GetMappedFileNameW(self.GetCurrentProcess(), handle, buf, len(buf)))
            if n >= len(buf):
                raise ValueError("mapped path truncated")
            result["mapped_device"] = buf.value
        return result

    def filesystem(self, path):
        volume, fs = c.create_unicode_buffer(32768), c.create_unicode_buffer(261)
        self.check(self.GetVolumePathNameW(str(path), volume, len(volume)))
        self.check(self.GetVolumeInformationW(volume.value, None, 0, None, None, None, fs, len(fs)))
        return {"volume": volume.value, "filesystem": fs.value}

    def job(self):
        h = self.check(self.CreateJobObjectW(None, None))
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # KILL_ON_JOB_CLOSE, no breakaway
        try:
            self.check(self.SetInformationJobObject(h, 9, c.byref(limits), c.sizeof(limits)))
        except BaseException:
            self.close(h)
            raise
        return h

    def assign(self, job, process):
        self.check(self.AssignProcessToJobObject(job, int(process._handle)))

    def active(self, job):
        info = Accounting()
        self.check(self.QueryInformationJobObject(job, 1, c.byref(info), c.sizeof(info), None))
        return int(info.active)

    def images(self, job):
        info = ProcessList()
        self.check(self.QueryInformationJobObject(job, 3, c.byref(info), c.sizeof(info), None))
        if info.count > 64 or info.assigned != info.count:
            raise RuntimeError("job process list incomplete")
        result = []
        for pid in info.pids[:info.count]:
            h = self.check(self.OpenProcess(0x1000, False, pid))
            try:
                member = w.BOOL()
                self.check(self.IsProcessInJob(h, job, c.byref(member)))
                if not member.value:
                    raise RuntimeError("PID no longer belongs to owned job")
                buf, n = c.create_unicode_buffer(32768), w.DWORD(32768)
                self.check(self.QueryFullProcessImageNameW(h, 0, buf, c.byref(n)))
                result.append({"pid": int(pid), "image": buf.value})
            finally:
                self.close(h)
        return result

    def terminate(self, job):
        self.check(self.TerminateJobObject(job, 125))
