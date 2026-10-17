"""Windows meeting capture: the meeting application's process tree, else the whole device.

Preferred: WASAPI process loopback (Windows 10 build 20348 / Windows 11 and later)
in INCLUDE mode on the meeting application's root process, so only that program's
sound is captured and the whisper the owner's surface plays stays out. It is
reached through ``ActivateAudioInterfaceAsync`` with ctypes COM calls, no native
build. Anything that fails here (an older build, a refused activation) falls back
to PyAudioWPatch's endpoint loopback of the default output device; that capture
contains everything the computer plays, including Ouroboros's own whispered hints,
and the companion says so to the call and in its status. No timing or loudness
rule tries to cut the whisper out: the mind is told what the audio contains.

A crash inside a COM call cannot be caught in Python, so an attempt marker is
written before activation and cleared after it; a marker left by a dead companion
turns process loopback off on this machine (recorded with the reason) and the
endpoint loopback is used from then on.
"""

from __future__ import annotations

import ctypes
import json
import pathlib
import sys
import threading
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from capture_common import CHUNK_MS, CaptureFailed, OnError, OnPcm, int16_mono
from meeting_apps import RunningApp, Selection

CAPTURE_RATE = 48000
CAPTURE_CHANNELS = 2
PROCESS_LOOPBACK_MIN_BUILD = 20348
ACTIVATION_TIMEOUT_SEC = 5.0
ATTEMPT_FILE = "process_loopback.attempt"
STATE_FILE = "process_loopback.json"

VT_BLOB = 65
AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK = 1
PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE = 0
AUDCLNT_SHAREMODE_SHARED = 0
AUDCLNT_STREAMFLAGS_LOOPBACK = 0x00020000
AUDCLNT_STREAMFLAGS_EVENTCALLBACK = 0x00040000
AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY = 0x08000000
AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM = 0x80000000
AUDCLNT_BUFFERFLAGS_SILENT = 0x2
VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK = "VAD\\Process_Loopback"
S_OK = 0
E_NOINTERFACE = -2147467262
TH32CS_SNAPPROCESS = 0x2
# Completion handlers of activations that timed out: Windows still holds their pointers and may
# call them later, so Python must never free them. One small object per timed-out attempt.
_ABANDONED_ACTIVATIONS: List[Any] = []


class GUID(ctypes.Structure):
    _fields_ = [("Data1", ctypes.c_uint32), ("Data2", ctypes.c_uint16), ("Data3", ctypes.c_uint16),
                ("Data4", ctypes.c_ubyte * 8)]


def guid(text: str) -> GUID:
    value = uuid.UUID(text)
    return GUID(value.time_low, value.time_mid, value.time_hi_version,
                (ctypes.c_ubyte * 8)(*value.bytes[8:]))


IID_IUNKNOWN = guid("00000000-0000-0000-C000-000000000046")
IID_IAGILE_OBJECT = guid("94EA2B94-E9CC-49E0-C0FF-EE64CA8F5B90")
IID_COMPLETION_HANDLER = guid("41D949AB-9862-444A-80F6-C261334DA5EB")
IID_IAUDIO_CLIENT = guid("1CB9AD4C-DBFA-4C32-B178-C2F568A703B2")
IID_IAUDIO_CAPTURE_CLIENT = guid("C8ADBD64-E71E-48A0-A4DE-185C395CD317")


class AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS(ctypes.Structure):
    _fields_ = [("TargetProcessId", ctypes.c_uint32), ("ProcessLoopbackMode", ctypes.c_int)]


class AUDIOCLIENT_ACTIVATION_PARAMS(ctypes.Structure):
    _fields_ = [("ActivationType", ctypes.c_int),
                ("ProcessLoopbackParams", AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS)]


class BLOB(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_uint32), ("pBlobData", ctypes.c_void_p)]


class PROPVARIANT(ctypes.Structure):
    _fields_ = [("vt", ctypes.c_ushort), ("wReserved1", ctypes.c_ushort), ("wReserved2", ctypes.c_ushort),
                ("wReserved3", ctypes.c_ushort), ("blob", BLOB)]


class WAVEFORMATEX(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("wFormatTag", ctypes.c_uint16), ("nChannels", ctypes.c_uint16),
                ("nSamplesPerSec", ctypes.c_uint32), ("nAvgBytesPerSec", ctypes.c_uint32),
                ("nBlockAlign", ctypes.c_uint16), ("wBitsPerSample", ctypes.c_uint16), ("cbSize", ctypes.c_uint16)]


def activation_params(pid: int) -> Tuple[AUDIOCLIENT_ACTIVATION_PARAMS, PROPVARIANT]:
    """The activation blob asking for the audio of ``pid``'s process tree only."""
    params = AUDIOCLIENT_ACTIVATION_PARAMS(
        AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK,
        AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS(int(pid), PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE))
    variant = PROPVARIANT()
    variant.vt = VT_BLOB
    variant.blob.cbSize = ctypes.sizeof(params)
    variant.blob.pBlobData = ctypes.cast(ctypes.pointer(params), ctypes.c_void_p)
    return params, variant


def capture_format() -> WAVEFORMATEX:
    block = CAPTURE_CHANNELS * 2
    return WAVEFORMATEX(1, CAPTURE_CHANNELS, CAPTURE_RATE, CAPTURE_RATE * block, block, 16, 0)


def root_processes(rows: Sequence[Tuple[int, int, str]]) -> List[RunningApp]:
    """One entry per process-tree root of each executable (a browser's many children fold in)."""
    by_pid = {pid: (ppid, exe.lower()) for pid, ppid, exe in rows}
    roots = []
    for pid, ppid, exe in rows:
        if pid in (0, 4) or not exe:
            continue
        parent = by_pid.get(ppid)
        if parent is None or parent[1] != exe.lower():
            roots.append(RunningApp(name=exe, ident=exe.lower(), pid=int(pid)))
    return roots


def windows_processes() -> List[Tuple[int, int, str]]:  # pragma: no cover - Windows only
    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [("dwSize", ctypes.c_uint32), ("cntUsage", ctypes.c_uint32),
                    ("th32ProcessID", ctypes.c_uint32), ("th32DefaultHeapID", ctypes.c_size_t),
                    ("th32ModuleID", ctypes.c_uint32), ("cntThreads", ctypes.c_uint32),
                    ("th32ParentProcessID", ctypes.c_uint32), ("pcPriClassBase", ctypes.c_long),
                    ("dwFlags", ctypes.c_uint32), ("szExeFile", ctypes.c_wchar * 260)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
    kernel32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    kernel32.Process32FirstW.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESSENTRY32W)]
    kernel32.Process32NextW.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESSENTRY32W)]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if not snapshot or snapshot == ctypes.c_void_p(-1).value:
        raise CaptureFailed("MEETING_CAPTURE_FAILED", "Windows did not list running processes")
    rows = []
    entry = PROCESSENTRY32W()
    entry.dwSize = ctypes.sizeof(entry)
    try:
        ok = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while ok:
            rows.append((int(entry.th32ProcessID), int(entry.th32ParentProcessID), str(entry.szExeFile)))
            ok = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return rows


class _Com:  # pragma: no cover - Windows only
    """The few COM entry points process loopback needs, as raw vtable calls."""

    def __init__(self) -> None:
        self.ole32 = ctypes.OleDLL("ole32")
        self.mmdevapi = ctypes.OleDLL("Mmdevapi")
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel32.CreateEventW.restype = ctypes.c_void_p
        self.kernel32.CreateEventW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_wchar_p]
        self.kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self.kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        self.ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self.mmdevapi.ActivateAudioInterfaceAsync.argtypes = [
            ctypes.c_wchar_p, ctypes.POINTER(GUID), ctypes.POINTER(PROPVARIANT), ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p)]
        self.QueryInterface = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(GUID),
                                                 ctypes.POINTER(ctypes.c_void_p))
        self.RefCount = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)
        self.Completed = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)

    @staticmethod
    def call(pointer: Any, index: int, *args: Any, argtypes: Sequence[Any] = (), restype: Any = None) -> Any:
        vtable = ctypes.cast(pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        prototype = ctypes.WINFUNCTYPE(restype or ctypes.HRESULT, ctypes.c_void_p, *argtypes)
        return prototype(vtable[index])(pointer, *args)

    def release(self, pointer: Any) -> None:
        if pointer:
            self.call(pointer, 2, restype=ctypes.c_ulong)


class _CompletionHandler:  # pragma: no cover - Windows only
    """An agile IActivateAudioInterfaceCompletionHandler whose lifetime Python holds."""

    def __init__(self, com: _Com):
        self.com = com
        self.done = threading.Event()
        self.result: Tuple[int, Any] = (-1, None)
        self.abandoned = False

        class Vtbl(ctypes.Structure):
            _fields_ = [("QueryInterface", com.QueryInterface), ("AddRef", com.RefCount),
                        ("Release", com.RefCount), ("ActivateCompleted", com.Completed)]

        class Object(ctypes.Structure):
            _fields_ = [("lpVtbl", ctypes.POINTER(Vtbl))]

        accepted = {bytes(IID_IUNKNOWN), bytes(IID_IAGILE_OBJECT), bytes(IID_COMPLETION_HANDLER)}

        def query(this: Any, riid: Any, out: Any) -> int:
            if bytes(riid.contents) in accepted:
                out[0] = this
                return S_OK
            out[0] = None
            return E_NOINTERFACE

        def completed(this: Any, operation: Any) -> int:
            status, unknown = ctypes.c_long(-1), ctypes.c_void_p()
            try:
                com.call(operation, 3, ctypes.byref(status), ctypes.byref(unknown),
                         argtypes=(ctypes.POINTER(ctypes.c_long), ctypes.POINTER(ctypes.c_void_p)))
                self.result = (int(status.value), unknown)
            except OSError as exc:
                self.result = (int(getattr(exc, "winerror", -1) or -1), None)
            if self.abandoned and unknown:  # nobody waits for this late client any more
                try:
                    com.release(unknown)
                except OSError:
                    pass
            self.done.set()
            return S_OK

        self._callbacks = (com.QueryInterface(query), com.RefCount(lambda this: 1),
                           com.RefCount(lambda this: 1), com.Completed(completed))
        self._vtbl = Vtbl(*self._callbacks)
        self._object = Object(ctypes.pointer(self._vtbl))
        self.pointer = ctypes.cast(ctypes.pointer(self._object), ctypes.c_void_p)


class _ProcessLoopbackStream:  # pragma: no cover - Windows only (driven through ProcessLoopbackCapture)
    """One process-loopback capture, all COM calls on its own thread."""

    def __init__(self, pid: int, stream: int):
        self.pid, self.stream = pid, stream
        self.ready = threading.Event()
        self.failure = ""
        self._go = threading.Event()
        self._stop = threading.Event()
        self._on_pcm: Optional[OnPcm] = None
        self._on_error: Optional[OnError] = None
        self._thread = threading.Thread(target=self._run, name=f"meeting-loopback-{pid}", daemon=True)

    def prepare(self) -> None:
        self._thread.start()
        if not self.ready.wait(ACTIVATION_TIMEOUT_SEC + 2):
            self.failure = self.failure or "Windows did not activate process loopback in time"
        if self.failure:
            self.stop()  # a failed stream never waits for a start that will not come

    def start(self, on_pcm: OnPcm, on_error: OnError) -> None:
        self._on_pcm, self._on_error = on_pcm, on_error
        self._go.set()

    def stop(self) -> None:
        self._stop.set()
        self._go.set()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        com = _Com()
        client = capture = event = None
        try:
            com.ole32.CoInitializeEx(None, 0)
            handler = _CompletionHandler(com)
            _params, variant = activation_params(self.pid)
            operation = ctypes.c_void_p()
            com.mmdevapi.ActivateAudioInterfaceAsync(VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK,
                                                     ctypes.byref(IID_IAUDIO_CLIENT), ctypes.byref(variant),
                                                     handler.pointer, ctypes.byref(operation))
            if not handler.done.wait(ACTIVATION_TIMEOUT_SEC):
                handler.abandoned = True
                _ABANDONED_ACTIVATIONS.append((handler, operation))  # Windows may still call it
                raise OSError("activation did not complete")
            status, unknown = handler.result
            com.release(operation)
            if status != 0 or not unknown:
                raise OSError(f"activation refused (HRESULT {status & 0xFFFFFFFF:#010x})")
            client = ctypes.c_void_p()
            com.call(unknown, 0, ctypes.byref(IID_IAUDIO_CLIENT), ctypes.byref(client),
                     argtypes=(ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)))
            com.release(unknown)
            fmt = capture_format()
            flags = (AUDCLNT_STREAMFLAGS_LOOPBACK | AUDCLNT_STREAMFLAGS_EVENTCALLBACK
                     | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY)
            com.call(client, 3, AUDCLNT_SHAREMODE_SHARED, flags, CHUNK_MS * 10000, 0, ctypes.byref(fmt), None,
                     argtypes=(ctypes.c_int, ctypes.c_uint32, ctypes.c_longlong, ctypes.c_longlong,
                               ctypes.POINTER(WAVEFORMATEX), ctypes.c_void_p))
            event = com.kernel32.CreateEventW(None, 0, 0, None)
            com.call(client, 13, event, argtypes=(ctypes.c_void_p,))
            capture = ctypes.c_void_p()
            com.call(client, 14, ctypes.byref(IID_IAUDIO_CAPTURE_CLIENT), ctypes.byref(capture),
                     argtypes=(ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)))
        except OSError as exc:
            self.failure = f"process loopback for pid {self.pid} failed: {exc}"
            self.ready.set()
            self._cleanup(com, client, capture, event)
            return
        self.ready.set()
        self._go.wait()
        started = False
        try:
            if not self._stop.is_set():
                com.call(client, 10)
                started = True
            while not self._stop.is_set():
                com.kernel32.WaitForSingleObject(event, 200)
                self._drain(com, capture)
        except OSError as exc:
            if self._on_error is not None and not self._stop.is_set():
                self._on_error(f"Windows stopped the process loopback capture: {exc}")
        finally:
            if started:
                try:
                    com.call(client, 11)
                except OSError:
                    pass
            self._cleanup(com, client, capture, event)

    def _drain(self, com: _Com, capture: Any) -> None:
        packet = ctypes.c_uint32(0)
        while True:
            com.call(capture, 5, ctypes.byref(packet), argtypes=(ctypes.POINTER(ctypes.c_uint32),))
            if not packet.value:
                return
            data, frames, flags = ctypes.POINTER(ctypes.c_ubyte)(), ctypes.c_uint32(0), ctypes.c_uint32(0)
            com.call(capture, 3, ctypes.byref(data), ctypes.byref(frames), ctypes.byref(flags), None, None,
                     argtypes=(ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)), ctypes.POINTER(ctypes.c_uint32),
                               ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p, ctypes.c_void_p))
            size = frames.value * CAPTURE_CHANNELS * 2
            chunk = bytes(size) if flags.value & AUDCLNT_BUFFERFLAGS_SILENT else ctypes.string_at(data, size)
            com.call(capture, 4, frames.value, argtypes=(ctypes.c_uint32,))
            if self._on_pcm is not None and chunk:
                self._on_pcm(int16_mono(chunk, CAPTURE_CHANNELS), self.stream)

    @staticmethod
    def _cleanup(com: _Com, client: Any, capture: Any, event: Any) -> None:
        for pointer in (capture, client):
            try:
                com.release(pointer)
            except OSError:
                pass
        if event:
            com.kernel32.CloseHandle(event)


class ProcessLoopbackCapture:
    """Process-tree audio of each selected application; one hub stream per application."""

    rate = CAPTURE_RATE

    def __init__(self, backend: "WindowsBackend", selection: Selection,
                 stream_factory: Callable[[int, int], Any] = _ProcessLoopbackStream):
        self.backend = backend
        self.selection = selection
        pids = list(dict.fromkeys(app.pid for app in selection.apps if app.pid))
        self.streams = [stream_factory(pid, index) for index, pid in enumerate(pids)]
        self._closed = False

    def describe(self) -> Dict[str, Any]:
        return {"backend": "wasapi-process-loopback", "source": "application process tree", "device": "",
                "includes_own_voice": self.selection.browser}

    def disclosures(self) -> List[str]:
        return []

    def prepare(self) -> None:
        if not self.streams:
            raise CaptureFailed("MEETING_CAPTURE_FAILED", "no process to capture")
        self.backend.mark_attempt()
        for stream in self.streams:
            stream.prepare()
        self.backend.clear_attempt()
        failures = [stream.failure for stream in self.streams if stream.failure]
        for stream in self.streams:
            if stream.failure:
                stream.stop()
        self.streams = [stream for stream in self.streams if not stream.failure]
        if not self.streams:
            raise CaptureFailed("MEETING_CAPTURE_FAILED", "; ".join(failures))

    def start(self, on_pcm: OnPcm, on_error: OnError) -> None:
        if self._closed:
            raise CaptureFailed("MEETING_CAPTURE_STOPPED", "listening stopped before the capture started")
        for stream in self.streams:
            stream.start(on_pcm, on_error)

    def stop(self) -> None:
        self._closed = True  # a stopped stream never starts, even if start() races with this
        for stream in self.streams:
            stream.stop()


class EndpointLoopbackCapture:
    """PyAudioWPatch loopback of the default output device: everything the computer plays."""

    def __init__(self, selection: Selection, pyaudio: Any, reason: str = ""):
        self.selection = selection
        self.reason = reason
        self._pyaudio = pyaudio
        self._lifecycle = threading.Lock()
        self._closed = False
        self._audio = pyaudio.PyAudio()
        try:
            self.device = dict(self._audio.get_default_wasapi_loopback())
        except Exception as exc:
            self._audio.terminate()
            raise CaptureFailed("MEETING_CAPTURE_FAILED", f"Windows reports no loopback device: {exc}") from None
        self.rate = int(float(self.device.get("defaultSampleRate") or 48000))
        self.channels = max(1, int(self.device.get("maxInputChannels") or 2))
        self._stream: Any = None

    def describe(self) -> Dict[str, Any]:
        return {"backend": "wasapi-endpoint-loopback", "source": "device loopback",
                "device": str(self.device.get("name") or ""), "includes_own_voice": True,
                **({"fallback_reason": self.reason} if self.reason else {})}

    def disclosures(self) -> List[str]:
        where = str(self.device.get("name") or "the default output device")
        return [f"This meeting audio is everything this computer plays on {where}, not only "
                f"{self.selection.label}; Ouroboros's own whispered hints to the owner are in it too, and "
                "speech in Ouroboros's own voice here is not a meeting participant."]

    def start(self, on_pcm: OnPcm, on_error: OnError) -> None:
        channels = self.channels

        def callback(in_data: bytes, frame_count: int, time_info: Any, status: int) -> tuple:
            on_pcm(int16_mono(in_data or b"", channels), 0)
            return None, self._pyaudio.paContinue

        with self._lifecycle:
            if self._closed:
                raise CaptureFailed("MEETING_CAPTURE_STOPPED", "listening stopped before the capture started")
            try:
                self._stream = self._audio.open(
                    format=self._pyaudio.paInt16, channels=channels, rate=self.rate, input=True,
                    input_device_index=int(self.device["index"]), frames_per_buffer=self.rate * CHUNK_MS // 1000,
                    stream_callback=callback, start=False)
                self._stream.start_stream()
            except Exception as exc:
                raise CaptureFailed("MEETING_CAPTURE_FAILED", f"Windows loopback capture failed: {exc}") from None

    def stop(self) -> None:
        with self._lifecycle:
            if self._closed:
                return
            self._closed = True
            stream, self._stream = self._stream, None
            try:
                if stream is not None:
                    stream.stop_stream()
                    stream.close()
            except Exception:
                pass
            finally:
                self._audio.terminate()


def _load_pyaudio() -> Any:
    try:
        import pyaudiowpatch
    except ImportError as exc:
        raise CaptureFailed("MEETING_DEPENDENCY_MISSING",
                            f"PyAudioWPatch is not installed for this skill ({exc}); its dependency install "
                            "has not finished") from None
    return pyaudiowpatch


class WindowsBackend:
    """Capture targets and capture on Windows; speaking needs a driver and is not offered."""

    platform = "win32"
    speaking_supported = False

    def __init__(self, state_dir: pathlib.Path, *, processes: Callable[[], List[Tuple[int, int, str]]] = None,
                 pyaudio_loader: Callable[[], Any] = _load_pyaudio, build: Optional[int] = None,
                 stream_factory: Optional[Callable[[int, int], Any]] = None):
        self.state_dir = pathlib.Path(state_dir)
        self._processes = processes or windows_processes
        self._pyaudio_loader = pyaudio_loader
        self._stream_factory = stream_factory or _ProcessLoopbackStream
        self.build = build if build is not None else (sys.getwindowsversion().build  # type: ignore[attr-defined]
                                                      if sys.platform == "win32" else 0)
        self._settle_crashed_attempt()

    def _settle_crashed_attempt(self) -> None:
        marker = self.state_dir / ATTEMPT_FILE
        if marker.exists():
            self._record(False, "an earlier process loopback attempt ended the companion; using the "
                                "device loopback on this computer")
            try:
                marker.unlink()
            except OSError:
                pass

    def _record(self, usable: bool, reason: str) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / STATE_FILE).write_text(json.dumps({"usable": usable, "reason": reason}),
                                                 encoding="utf-8")

    def process_loopback_state(self) -> Tuple[bool, str]:
        if self.build < PROCESS_LOOPBACK_MIN_BUILD:
            return False, f"Windows build {self.build} has no process loopback (needs {PROCESS_LOOPBACK_MIN_BUILD}+)"
        try:
            state = json.loads((self.state_dir / STATE_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return True, ""
        return bool(state.get("usable", True)), str(state.get("reason") or "")

    def mark_attempt(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / ATTEMPT_FILE).write_text("attempt", encoding="utf-8")

    def clear_attempt(self) -> None:
        try:
            (self.state_dir / ATTEMPT_FILE).unlink()
        except OSError:
            pass

    def permission(self) -> tuple:
        return "granted", ""

    def request_permission(self) -> None:
        return None

    def running_apps(self) -> List[RunningApp]:
        return root_processes(self._processes())

    def open_capture(self, selection: Selection) -> Any:
        usable, reason = self.process_loopback_state()
        if usable:
            capture = ProcessLoopbackCapture(self, selection, self._stream_factory)
            try:
                capture.prepare()
                return capture
            except CaptureFailed as exc:
                reason = exc.message
        return EndpointLoopbackCapture(selection, self._pyaudio_loader(), reason)

    def cleanup_leftovers(self) -> List[int]:
        return []

    def open_voice(self, rate: int) -> Any:
        raise CaptureFailed("MEETING_SPEAK_UNSUPPORTED", "Speaking mode is not available on Windows.")

    def run_main_loop(self, stop_event: threading.Event) -> None:
        stop_event.wait()
