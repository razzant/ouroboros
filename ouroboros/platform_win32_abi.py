"""Private Win32 ABI declarations for platform_layer's process and lock primitives.

Signatures and ctypes layouts share one identity; operations remain in the public
platform layer. Importing this module on another platform loads no native DLL.
"""

import sys

if sys.platform == "win32":
    import ctypes
    import ctypes.wintypes

    # Snapshot the actual call error; declare full-width HANDLE ABI (ARCHITECTURE §1).
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]

    class _OVERLAPPED(ctypes.Structure):
        _fields_ = [
            ("Internal", ctypes.c_void_p),
            ("InternalHigh", ctypes.c_void_p),
            ("Offset", ctypes.wintypes.DWORD),
            ("OffsetHigh", ctypes.wintypes.DWORD),
            ("hEvent", ctypes.wintypes.HANDLE),
        ]

    # One complete ABI table for process presence, locks and Job ownership.
    for _api_name, _result_type, _argument_types in (
        ("CreateJobObjectW", ctypes.wintypes.HANDLE, (ctypes.wintypes.LPVOID, ctypes.wintypes.LPCWSTR)),
        ("SetInformationJobObject", ctypes.wintypes.BOOL,
         (ctypes.wintypes.HANDLE, ctypes.c_int, ctypes.wintypes.LPVOID, ctypes.wintypes.DWORD)),
        ("OpenProcess", ctypes.wintypes.HANDLE, (ctypes.wintypes.DWORD, ctypes.wintypes.BOOL, ctypes.wintypes.DWORD)),
        ("GetExitCodeProcess", ctypes.wintypes.BOOL, (ctypes.wintypes.HANDLE, ctypes.POINTER(ctypes.wintypes.DWORD))),
        ("GetProcessTimes", ctypes.wintypes.BOOL, (ctypes.wintypes.HANDLE, *(ctypes.POINTER(ctypes.wintypes.FILETIME),) * 4)),
        ("GetProcessId", ctypes.wintypes.DWORD, (ctypes.wintypes.HANDLE,)),
        ("GetCurrentProcess", ctypes.wintypes.HANDLE, ()),
        ("IsProcessInJob", ctypes.wintypes.BOOL,
         (ctypes.wintypes.HANDLE, ctypes.wintypes.HANDLE, ctypes.POINTER(ctypes.wintypes.BOOL))),
        ("QueryInformationJobObject", ctypes.wintypes.BOOL,
         (ctypes.wintypes.HANDLE, ctypes.c_int, ctypes.wintypes.LPVOID, ctypes.wintypes.DWORD, ctypes.POINTER(ctypes.wintypes.DWORD))),
        ("AssignProcessToJobObject", ctypes.wintypes.BOOL, (ctypes.wintypes.HANDLE, ctypes.wintypes.HANDLE)),
        ("TerminateJobObject", ctypes.wintypes.BOOL, (ctypes.wintypes.HANDLE, ctypes.wintypes.UINT)),
        ("CloseHandle", ctypes.wintypes.BOOL, (ctypes.wintypes.HANDLE,)),
        ("LockFileEx", ctypes.wintypes.BOOL,
         (ctypes.wintypes.HANDLE, ctypes.wintypes.DWORD, ctypes.wintypes.DWORD,
          ctypes.wintypes.DWORD, ctypes.wintypes.DWORD, ctypes.POINTER(_OVERLAPPED))),
        ("UnlockFileEx", ctypes.wintypes.BOOL,
         (ctypes.wintypes.HANDLE, ctypes.wintypes.DWORD, ctypes.wintypes.DWORD,
          ctypes.wintypes.DWORD, ctypes.POINTER(_OVERLAPPED))),
    ):
        _api = getattr(_kernel32, _api_name)
        _api.restype, _api.argtypes = _result_type, _argument_types

    # .value, not the HANDLE instance: with restype=HANDLE the calls return plain
    # ints (or None for NULL), and an int never equals a ctypes instance.
    _INVALID_HANDLE_VALUE = ctypes.wintypes.HANDLE(-1).value
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    _JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x800
    _JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK = 0x1000
    _JOBOBJECTINFOCLASS_EXTENDED = 9
    _PROCESS_SET_QUOTA = 0x0100
    _PROCESS_TERMINATE = 0x0001
    _PROCESS_SUSPEND_RESUME = 0x0800
    _CREATE_SUSPENDED = 0x4

    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", ctypes.wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", ctypes.wintypes.DWORD),
            ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
            ("PriorityClass", ctypes.wintypes.DWORD),
            ("SchedulingClass", ctypes.wintypes.DWORD),
        ]

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class _ExtendedLimitInfo(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]
