"""Phase 4 (v6.39) G: crash-safe atomic full-file overwrite."""

from __future__ import annotations

import json

import pytest

from ouroboros import utils
from ouroboros.utils import atomic_write_json, write_bytes_atomic, write_text, write_text_atomic


def test_write_text_atomic_writes_content(tmp_path):
    target = tmp_path / "f.txt"
    write_text_atomic(target, "hello world")
    assert target.read_text(encoding="utf-8") == "hello world"


@pytest.mark.parametrize("fsync", [False, True])
def test_write_bytes_atomic_writes_exact_bytes(tmp_path, fsync):
    target = tmp_path / "media.bin"
    content = b"\x00\xff\x10media"
    write_bytes_atomic(target, content, fsync=fsync)
    assert target.read_bytes() == content


def test_write_text_atomic_preserves_old_file_on_failure(tmp_path, monkeypatch):
    target = tmp_path / "f.txt"
    target.write_text("OLD CONTENT", encoding="utf-8")

    def _boom(*a, **k):
        raise OSError("simulated crash during replace")

    # Fail the atomic swap AFTER the temp is written: the EXISTING file must stay fully
    # intact (never a truncated/partial file) and the orphan temp must be cleaned up.
    monkeypatch.setattr(utils.os, "replace", _boom)
    with pytest.raises(OSError):
        write_text_atomic(target, "NEW CONTENT THAT NEVER LANDS")

    assert target.read_text(encoding="utf-8") == "OLD CONTENT"
    assert not list(tmp_path.glob(".f.txt.tmp.*"))  # no orphaned temp left behind


@pytest.mark.skipif(__import__("sys").platform.startswith("win"),
                    reason="POSIX execute bits are not preserved/reported on Windows")
def test_write_text_atomic_preserves_full_mode(tmp_path):
    import os
    target = tmp_path / "script.sh"
    target.write_text("#!/bin/sh\necho old\n", encoding="utf-8")
    # setgid + rwxr-x--- exercises the FULL 0o7777 mask (special bits, not just rwx). Use
    # whatever the filesystem actually stored as the baseline so the test is fs-robust.
    os.chmod(target, 0o2750)
    expected = os.stat(target).st_mode & 0o7777
    # os.replace creates a new inode; the existing mode (incl any special bits) must survive.
    write_text_atomic(target, "#!/bin/sh\necho new\n")
    assert target.read_text(encoding="utf-8") == "#!/bin/sh\necho new\n"
    assert (os.stat(target).st_mode & 0o7777) == expected
    assert (os.stat(target).st_mode & 0o111)  # still executable


def test_write_text_helper_is_atomic(tmp_path):
    # utils.write_text (the shared overwrite helper used by git.py et al.) now routes
    # through the atomic primitive.
    target = tmp_path / "g.txt"
    target.write_text("OLD", encoding="utf-8")
    write_text(target, "NEW")
    assert target.read_text(encoding="utf-8") == "NEW"


def test_atomic_write_json_still_works(tmp_path):
    target = tmp_path / "d.json"
    atomic_write_json(target, {"a": 1, "b": [2, 3]})
    import json
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1, "b": [2, 3]}


def test_replace_retries_windows_sharing_violation_then_succeeds(tmp_path, monkeypatch):
    """Transient PermissionError (Windows winerror 5/32 sharing violation: a
    reader holds the destination open without FILE_SHARE_DELETE) must be
    absorbed by a bounded retry — the write lands and the file is intact."""
    target = tmp_path / "job.json"
    target.write_text("OLD", encoding="utf-8")
    real_replace = utils.os.replace
    calls = {"n": 0}

    def _flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise PermissionError(13, "The process cannot access the file")
        return real_replace(src, dst)

    monkeypatch.setattr(utils.os, "replace", _flaky_replace)
    monkeypatch.setattr(utils.time, "sleep", lambda _s: None)
    write_text_atomic(target, "NEW")
    assert calls["n"] == 4
    assert target.read_text(encoding="utf-8") == "NEW"
    assert not list(tmp_path.glob(".job.json.tmp.*"))


def test_replace_raises_permission_error_after_bounded_retries(tmp_path, monkeypatch):
    """A persistent PermissionError (a genuinely locked file) must surface
    honestly after the retry bound — never swallowed, never unbounded."""
    target = tmp_path / "job.json"
    target.write_text("OLD", encoding="utf-8")
    calls = {"n": 0}

    def _always_denied(src, dst):
        calls["n"] += 1
        raise PermissionError(13, "The process cannot access the file")

    monkeypatch.setattr(utils.os, "replace", _always_denied)
    monkeypatch.setattr(utils.time, "sleep", lambda _s: None)
    with pytest.raises(PermissionError):
        write_text_atomic(target, "NEW CONTENT THAT NEVER LANDS")
    assert calls["n"] == utils._REPLACE_RETRY_ATTEMPTS
    assert target.read_text(encoding="utf-8") == "OLD"
    assert not list(tmp_path.glob(".job.json.tmp.*"))


def test_replace_atomic_does_not_retry_other_oserrors(tmp_path, monkeypatch):
    """Only the Windows sharing-violation class retries; any other OSError
    (POSIX-visible failures included) propagates on the first attempt."""
    calls = {"n": 0}

    def _boom(src, dst):
        calls["n"] += 1
        raise OSError("disk detached")

    monkeypatch.setattr(utils.os, "replace", _boom)
    with pytest.raises(OSError):
        utils.replace_atomic(tmp_path / "a", tmp_path / "b")
    assert calls["n"] == 1


@pytest.mark.parametrize("writer,payload,read", [
    (write_text_atomic, "short-write survivor ✓ text", lambda p: p.read_text(encoding="utf-8")),
    (write_bytes_atomic, b"short-write survivor bytes", lambda p: p.read_bytes()),
])
def test_fsync_path_survives_short_os_writes(tmp_path, monkeypatch, writer, payload, read):
    """External-audit correction lane (base 8827fd2c), item 2: the fsync lane
    of write_text_atomic issued ONE bare ``os.write`` and trusted its return —
    a partial write published a truncated file behind a successful rename.
    Both fsync lanes must loop until every byte lands (``_write_fd_fully``)."""
    import os as _os

    target = tmp_path / "out.dat"
    real_write = _os.write

    def one_byte_at_a_time(fd, data):
        return real_write(fd, bytes(data)[:1])

    monkeypatch.setattr(utils.os, "write", one_byte_at_a_time)
    writer(target, payload, fsync=True)
    monkeypatch.undo()
    assert read(target) == payload


def test_append_jsonl_survives_short_os_writes(tmp_path, monkeypatch):
    """Audit #15-11/12 corrective lane: ``append_jsonl`` issued ONE bare
    ``os.write`` and returned success without checking the byte count — the
    same short-write class the atomic writers already fixed, left in the
    append SSOT every authority JSONL stream goes through. A torn line here is
    a lost record, not a truncated file."""
    import os as _os

    target = tmp_path / "logs" / "events.jsonl"
    real_write = _os.write

    def one_byte_at_a_time(fd, data):
        if bytes(data).startswith(b"pid="):
            return real_write(fd, data)  # the lock primitive's own metadata
        return real_write(fd, bytes(data)[:1])

    monkeypatch.setattr(utils.os, "write", one_byte_at_a_time)
    assert utils.append_jsonl(target, {"type": "llm_usage", "cost": 1.5}) is True
    monkeypatch.undo()
    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"type": "llm_usage", "cost": 1.5}


def test_append_jsonl_reports_failure_and_never_retries_a_torn_record(tmp_path, monkeypatch):
    """A write that dies MID-record must report False, not retry the whole
    line: the retry would duplicate the prefix already on disk. Only the open
    is retried."""
    import os as _os

    target = tmp_path / "logs" / "events.jsonl"
    real_write = _os.write
    calls = {"n": 0}

    def half_then_die(fd, data):
        if bytes(data).startswith(b"pid="):
            return real_write(fd, data)  # the lock primitive's own metadata
        calls["n"] += 1
        if calls["n"] == 1:
            return real_write(fd, bytes(data)[:4])
        raise OSError("device gone")

    monkeypatch.setattr(utils.os, "write", half_then_die)
    assert utils.append_jsonl(target, {"type": "task_done"}) is False
    monkeypatch.undo()
    assert calls["n"] == 2  # one short write, one failure — no whole-record replay
    assert target.read_bytes() == b'{"ty'


@pytest.mark.parametrize("fsync", [False, True])
def test_write_text_atomic_is_byte_exact_on_every_platform(tmp_path, monkeypatch, fsync):
    """Audit #14-5: the text lane used to translate newlines on Windows — the
    non-fsync lane through ``Path.write_text``'s text mode, the fsync lane
    through an ``os.open`` without ``O_BINARY``. Byte-exact consumers (run
    manifests, hashed receipts, agent file writes that round-trip LF source)
    were rewritten silently by it.

    POSIX has no translation to observe, so the flag is simulated: give ``os``
    an ``O_BINARY`` bit the way Windows has one, pin that the fsync lane passes
    it, and pin the exact bytes on both lanes."""
    import os as _os

    # Windows has the real bit: pin it as-is (stripping it would re-enable the
    # text-mode translation this test forbids). POSIX has none, so simulate one.
    real_o_binary = getattr(_os, "O_BINARY", None)
    fake_o_binary = real_o_binary if real_o_binary is not None else 1 << 26
    target = tmp_path / "run_manifest.json"
    content = '{\n  "seed": "v7next\\r",\n  "lines": "a\\nb"\n}\n'
    flags_seen: list[int] = []
    real_open = _os.open

    def spy_open(path, flags, *args, **kwargs):
        flags_seen.append(flags)
        if real_o_binary is None:
            flags &= ~fake_o_binary
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(_os, "O_BINARY", fake_o_binary, raising=False)
    monkeypatch.setattr(utils.os, "open", spy_open)
    write_text_atomic(target, content, fsync=fsync)
    monkeypatch.undo()

    assert target.read_bytes() == content.encode("utf-8")
    if fsync:
        assert flags_seen and all(flags & fake_o_binary for flags in flags_seen)


def test_atomic_write_json_lands_exact_bytes(tmp_path):
    """The JSON SSOT inherits the byte-exact contract: durable state hashes the
    same on every platform."""
    target = tmp_path / "state.json"
    atomic_write_json(target, {"a": [1, 2], "b": "x"}, trailing_newline=True)
    assert target.read_bytes() == b'{\n  "a": [\n    1,\n    2\n  ],\n  "b": "x"\n}\n'


def test_supervisor_state_atomic_write_text_lands_every_byte_on_short_writes(tmp_path, monkeypatch):
    """Audit #16-6: ``supervisor.state.atomic_write_text`` used to publish one
    ``os.write`` behind the rename — a short write became a truncated
    ``state.json``. It rides the utils write loop now."""
    import os

    from supervisor import state as sup_state

    real_write = os.write
    monkeypatch.setattr(utils.os, "write", lambda fd, data: real_write(fd, bytes(data[:1])))
    target = tmp_path / "state.json"
    payload = '{"a": 1, "b": "' + "x" * 300 + '"}\n'
    sup_state.atomic_write_text(target, payload)
    assert target.read_bytes() == payload.encode("utf-8")
