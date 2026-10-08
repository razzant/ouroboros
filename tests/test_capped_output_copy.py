"""#1569: the launcher's pipe copy is byte-capped and keeps draining after storage errors.

Owner decision of 8 October 2026 (quiz 9928636c): keep the size limit; while storage
refuses writes, keep draining and drop new chunks with an explicit gap, never truncate
the old live file after a failed rename and never append past the cap.
"""
from __future__ import annotations

import errno
import io
import os
import pathlib
import subprocess
import sys
import threading

import pytest

from ouroboros import process_logging
from ouroboros.process_logging import OUTPUT_CHUNK_BYTES, SERVER_LOG_MAX_BYTES, copy_capped_output


class Report:
    """The launcher's own log, recorded."""

    def __init__(self):
        self.lines = []

    def warning(self, message, *args):
        self.lines.append(("warning", message % args))

    def info(self, message, *args):
        self.lines.append(("info", message % args))


class Reads(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.sizes = []

    def readline(self, limit=-1):
        chunk = super().readline(limit)
        self.sizes.append(len(chunk))
        return chunk


def generations(path):
    """Oldest first: .3, .2, .1, live."""
    names = [path.with_name(f"{path.name}.{index}") for index in (3, 2, 1)] + [path]
    return [name for name in names if name.exists()]


def stored(path):
    return b"".join(name.read_bytes() for name in generations(path))


def mixed_lines(total):
    """Multibyte, CRLF and malformed UTF-8 lines of varied length."""
    pieces, size, index = [], 0, 0
    while size < total:
        body = ("é漢" * (index % 400)).encode("utf-8") + b"\xff\xfe" * (index % 3)
        line = b"%06d " % index + body + (b"\r\n" if index % 2 else b"\n")
        pieces.append(line)
        size += len(line)
        index += 1
    return b"".join(pieces)


def test_raw_bytes_are_copied_exactly_and_every_file_stays_within_the_cap(tmp_path):
    path, cap, data = tmp_path / "agent_stdout.log", 100_000, mixed_lines(1_000_000)
    stream = Reads(data)
    copy_capped_output(stream, path, report=Report(), max_bytes=cap)

    files = generations(path)
    assert len(files) == 4 and all(name.stat().st_size <= cap for name in files)
    kept = stored(path)
    assert data.endswith(kept) and len(kept) > 3 * cap - OUTPUT_CHUNK_BYTES  # No decoding, no newline translation.
    assert all(name.read_bytes().startswith(b"0") for name in files)  # Rotation falls between lines.
    assert stream.tell() == len(data)


def test_a_newline_free_flood_is_read_in_bounded_chunks(tmp_path):
    path, data = tmp_path / "agent_stdout.log", b"x" * (5 * 1024 * 1024)
    stream = Reads(data)
    copy_capped_output(stream, path, report=Report())

    assert max(stream.sizes) == OUTPUT_CHUNK_BYTES
    assert all(name.stat().st_size <= SERVER_LOG_MAX_BYTES for name in generations(path))
    assert stored(path) == data  # Three generations hold 5 MiB; nothing older existed.


def test_an_inherited_oversized_file_rotates_whole_and_is_not_cleaned(tmp_path):
    path, cap = tmp_path / "agent_stdout.log", 4096
    inherited = b"old writer line\n" * 1000  # 16 000 bytes, written by someone else
    path.write_bytes(inherited)
    copy_capped_output(Reads(b"new line\n" * 10), path, report=Report(), max_bytes=cap)

    assert path.with_name("agent_stdout.log.1").read_bytes() == inherited  # Disclosed, aged out later.
    assert path.read_bytes() == b"new line\n" * 10


@pytest.mark.parametrize("refused", ["live_rename", "every_rename"])
def test_a_failed_rotation_never_truncates_never_passes_the_cap_and_keeps_draining(tmp_path, monkeypatch, refused):
    path, cap = tmp_path / "agent_stdout.log", 4096
    live = b"L" * (cap - 5)
    path.write_bytes(live)
    olds = {index: bytes([48 + index]) * 100 for index in (1, 2, 3)}
    for index, body in olds.items():
        path.with_name(f"agent_stdout.log.{index}").write_bytes(body)
    real_replace, attempts = os.replace, []

    def replace(src, dst):
        attempts.append(pathlib.Path(src).name)
        if refused == "every_rename" or pathlib.Path(src) == path:
            raise PermissionError(errno.EACCES, "held by another process")
        real_replace(src, dst)

    monkeypatch.setattr(process_logging.os, "replace", replace)
    data = b"0123456789\n" * 200
    stream, report = Reads(data), Report()
    copy_capped_output(stream, path, report=report, max_bytes=cap)

    assert stream.tell() == len(data)  # Every chunk was drained.
    assert path.read_bytes() == live  # Never truncated, never appended past the cap.
    backups = {index: path.with_name(f"agent_stdout.log.{index}").read_bytes()
               for index in (1, 2, 3) if path.with_name(f"agent_stdout.log.{index}").exists()}
    if refused == "every_rename":
        assert backups == olds
    else:
        # The first attempt shifted the chain once (the oldest generation is what any
        # rotation discards); every retry found .1 free and moved nothing else.
        assert backups == {2: olds[1], 3: olds[2]}
        assert attempts[:3] == ["agent_stdout.log.2", "agent_stdout.log.1", "agent_stdout.log"]
        assert set(attempts[3:]) == {"agent_stdout.log"}
    assert report.lines == [
        ("warning", "agent_stdout.log storage failed (rotate:PermissionError:EACCES); output keeps draining "
                    "and is dropped until a write succeeds"),
        ("warning", f"agent_stdout.log output ended with storage failing (rotate:PermissionError:EACCES): "
                    f"{len(data)} bytes not written, 0 bytes of uncertain durability")]


def test_storage_that_recovers_writes_one_bounded_marker_counted_toward_the_cap(tmp_path, monkeypatch):
    path, cap = tmp_path / "agent_stdout.log", 200
    path.write_bytes(b"p" * (cap - 15))
    real_open, refusals = open, [3]

    def flaky_open(*args, **kwargs):
        if refusals[0]:
            refusals[0] -= 1
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_open(*args, **kwargs)

    monkeypatch.setattr(process_logging, "open", flaky_open, raising=False)
    lines = [b"line-%d....\n" % index for index in range(5)]  # 11 bytes each
    report = Report()
    copy_capped_output(Reads(b"".join(lines)), path, report=report, max_bytes=cap)

    marker = (b"[output gap: 33 bytes not written, 0 bytes of uncertain durability "
              b"(open:OSError:ENOSPC)]\n")
    assert path.with_name("agent_stdout.log.1").read_bytes() == b"p" * (cap - 15)
    assert path.read_bytes() == marker + lines[3] + lines[4]  # The marker rotated the file first.
    assert all(name.stat().st_size <= cap for name in generations(path))
    assert report.lines == [
        ("warning", "agent_stdout.log storage failed (open:OSError:ENOSPC); output keeps draining and is "
                    "dropped until a write succeeds"),
        ("info", "agent_stdout.log storage recovered: 33 bytes not written, 0 bytes of uncertain durability")]


def test_a_failed_write_is_uncertain_and_kept_apart_from_unwritten_bytes(tmp_path, monkeypatch):
    path = tmp_path / "agent_stdout.log"
    real_open, plan = open, ["partial", "refuse", "ok"]

    class PartialWrite:
        """Four bytes reach the file, then the device fails."""

        def __init__(self, handle):
            self.handle, self.calls = handle, 0

        def write(self, data):
            self.calls += 1
            if self.calls == 1:
                return self.handle.write(data)  # The first chunk, whole
            if self.calls == 2:
                return self.handle.write(bytes(data[:4]))
            raise OSError(errno.EIO, "Input/output error")

        def fileno(self):
            return self.handle.fileno()

        def close(self):
            self.handle.close()

    def opener(*args, **kwargs):
        step = plan.pop(0)
        if step == "refuse":
            raise PermissionError(errno.EACCES, "denied")
        handle = real_open(*args, **kwargs)
        return PartialWrite(handle) if step == "partial" else handle

    monkeypatch.setattr(process_logging, "open", opener, raising=False)
    chunks = [b"first-ok.\n", b"second-10\n", b"third-10.\n", b"fourth-ok\n"]
    report = Report()
    copy_capped_output(Reads(b"".join(chunks)), path, report=report)

    # The healthy handle writes chunk 1; 4 bytes of chunk 2 reach the file and its other
    # 6 are uncertain; chunk 3 meets a refused open and is known not written.
    marker = b"[output gap: 10 bytes not written, 6 bytes of uncertain durability (write:OSError:EIO)]\n"
    assert path.read_bytes() == chunks[0] + chunks[1][:4] + marker + chunks[3]
    assert report.lines[-1] == (
        "info", "agent_stdout.log storage recovered: 10 bytes not written, 6 bytes of uncertain durability")


def test_a_failing_report_never_stops_the_copy(tmp_path):
    class Broken:
        def warning(self, *_args):
            raise RuntimeError("launcher log is gone")

        info = warning

    blocker = tmp_path / "not-a-directory"
    blocker.write_text("file", encoding="utf-8")
    data = b"line\n" * 1000
    stream = Reads(data)
    copy_capped_output(stream, blocker / "agent_stdout.log", report=Broken())
    assert stream.tell() == len(data)


@pytest.mark.parametrize("failure", [OSError(errno.EIO, "Input/output error"), ValueError("I/O on closed file")])
def test_a_broken_input_ends_the_copy_instead_of_spinning(tmp_path, failure):
    class Failing:
        calls = 0

        def readline(self, _limit):
            self.calls += 1
            if self.calls == 1:
                return b"last words\n"
            raise failure

    stream, report = Failing(), Report()
    copy_capped_output(stream, tmp_path / "agent_stdout.log", report=report)
    assert stream.calls == 2 and (tmp_path / "agent_stdout.log").read_bytes() == b"last words\n"
    assert report.lines == [("warning", f"Output pipe read failed ({type(failure).__name__}"
                                        f"{':EIO' if isinstance(failure, OSError) else ''}); "
                                        "the copy into agent_stdout.log stopped")]


PRODUCER = ("import sys\nout = sys.stdout.buffer\n"
            "for index in range(10240):\n    out.write(b'%07d ' % index + b'x' * 1015 + b'\\n')\nout.flush()\n")


@pytest.mark.serial
@pytest.mark.parametrize("sink", ["unwritable", "writable"])
def test_a_real_child_writing_past_pipe_capacity_never_blocks(tmp_path, sink):
    """10 MiB through a real pipe: the child exits with a working sink or immediate storage errors."""
    blocker = tmp_path / "blocker"
    if sink == "unwritable":
        blocker.write_text("a file where the logs directory should be", encoding="utf-8")
    path = blocker / "agent_stdout.log"
    child = subprocess.Popen([sys.executable, "-c", PRODUCER], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    report = Report()
    copier = threading.Thread(target=copy_capped_output, args=(child.stdout, path), kwargs={"report": report})
    copier.start()
    try:
        assert child.wait(timeout=120) == 0
    finally:
        copier.join(timeout=120)
        child.stdout.close()
    assert not copier.is_alive()
    total = 10240 * 1024
    if sink == "unwritable":
        assert [level for level, _ in report.lines] == ["warning", "warning"]
        assert report.lines[1][1].endswith(f"{total} bytes not written, 0 bytes of uncertain durability")
    else:
        assert report.lines == []
        kept = stored(path)
        assert len(generations(path)) == 4 and all(name.stat().st_size <= SERVER_LOG_MAX_BYTES
                                                    for name in generations(path))
        assert kept.endswith(b"%07d " % 10239 + b"x" * 1015 + b"\n") and len(kept) % 1024 == 0
        assert len(kept) > 3 * SERVER_LOG_MAX_BYTES
