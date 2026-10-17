"""What every capture backend shares: typed failures, PCM conversion, a pipe reader.

A backend turns the meeting application's sound into PCM16 mono at one true
sample rate (the rate the peer declares in ``hello``; the call's hub resamples).
Conversion here is arithmetic only: float samples are scaled and clipped,
several channels are averaged.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from array import array
from typing import Callable, Optional

# Read size of one capture chunk: a latency/transport bound, not a judgement.
CHUNK_MS = 20

OnPcm = Callable[[bytes, int], None]      # (pcm16 mono, stream id)
OnError = Callable[[str], None]           # plain reason


class CaptureFailed(Exception):
    """Capture could not start or stopped; ``code`` is typed, ``message`` plain words."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def chunk_bytes(rate: int) -> int:
    return max(2, int(rate) * CHUNK_MS // 1000 * 2)


def _to_bytes(samples: array) -> bytes:
    if sys.byteorder != "little":
        samples = array(samples.typecode, samples)
        samples.byteswap()
    return samples.tobytes()


def int16_mono(pcm: bytes, channels: int) -> bytes:
    """Average interleaved PCM16 channels into one."""
    if channels <= 1:
        return pcm[: len(pcm) - len(pcm) % 2]
    samples = array("h")
    usable = len(pcm) - len(pcm) % (2 * channels)
    samples.frombytes(pcm[:usable])
    if sys.byteorder != "little":
        samples.byteswap()
    mono = array("h", (sum(samples[index:index + channels]) // channels
                       for index in range(0, len(samples), channels)))
    return _to_bytes(mono)


def float32_mono(data: bytes, channels: int = 1, planar: bool = True) -> bytes:
    """Scale native-endian float32 samples (planar or interleaved) to PCM16 mono."""
    floats = array("f")
    floats.frombytes(data[: len(data) - len(data) % 4])
    channels = max(1, int(channels))
    frames = len(floats) // channels
    if channels == 1:
        mixed = floats
    elif planar:
        mixed = [sum(floats[channel * frames + index] for channel in range(channels)) / channels
                 for index in range(frames)]
    else:
        mixed = [sum(floats[index * channels:(index + 1) * channels]) / channels
                 for index in range(frames)]
    return _to_bytes(array("h", (max(-32768, min(32767, int(round(value * 32767.0))))
                                 for value in mixed)))


def stop_process(proc: Optional[subprocess.Popen], timeout_sec: float = 2.0) -> None:
    """Terminate a helper process and observe its exit."""
    if proc is None:
        return
    if proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                pass
        except OSError:
            pass
    for stream in (proc.stdin, proc.stdout, proc.stderr):
        try:
            if stream is not None:
                stream.close()
        except OSError:
            pass


class Helper:
    """A long-running audio helper (recorder or player) whose stderr is kept, bounded.

    Draining stderr on a thread keeps a chatty helper from blocking on a full pipe;
    only the newest ``STDERR_KEEP`` bytes are kept, because only the reason for an
    exit is ever reported.
    """

    STDERR_KEEP = 4096

    def __init__(self, argv, *, popen=subprocess.Popen, env=None, stdin: bool = False):
        self.argv = list(argv)
        self.proc = popen(self.argv, stdin=subprocess.PIPE if stdin else subprocess.DEVNULL,
                          stdout=subprocess.DEVNULL if stdin else subprocess.PIPE,
                          stderr=subprocess.PIPE, env=env)
        self._stderr = bytearray()
        self._drain = threading.Thread(target=self._drain_stderr, name="meeting-helper-stderr",
                                       daemon=True)
        self._drain.start()

    def _drain_stderr(self) -> None:
        stream = self.proc.stderr
        while stream is not None:
            try:
                chunk = stream.read(512)
            except (OSError, ValueError):
                return
            if not chunk:
                return
            self._stderr.extend(chunk if isinstance(chunk, bytes) else str(chunk).encode("utf-8"))
            del self._stderr[:-self.STDERR_KEEP]

    def tail(self, limit: int = 600) -> str:
        text = bytes(self._stderr).decode("utf-8", errors="replace").strip()
        return text[-limit:]

    def stop(self) -> None:
        stop_process(self.proc)


class PipeReader:
    """Read fixed PCM chunks from a capture helper's stdout on a thread."""

    def __init__(self, helper: Helper, rate: int, on_pcm: OnPcm, on_error: OnError,
                 *, channels: int = 1, stream: int = 0, name: str = "capture"):
        self._helper = helper
        self._proc = helper.proc
        self._size = chunk_bytes(rate) * max(1, channels)
        self._channels = max(1, channels)
        self._on_pcm = on_pcm
        self._on_error = on_error
        self._stream = stream
        self._stopping = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"meeting-{name}-reader", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stopping.set()

    def join(self, timeout: float = 2.0) -> None:
        if self._thread is not threading.current_thread():
            self._thread.join(timeout)

    def _run(self) -> None:
        stdout = self._proc.stdout
        while not self._stopping.is_set():
            try:
                data = stdout.read(self._size) if stdout is not None else b""
            except (OSError, ValueError):
                data = b""
            if not data:
                break
            self._on_pcm(int16_mono(data, self._channels), self._stream)
        if not self._stopping.is_set():
            try:
                code = self._proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                code = None
            tail = self._helper.tail()
            self._on_error(f"the capture helper stopped (exit {code}){': ' + tail if tail else ''}")
