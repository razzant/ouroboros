"""Transparent patch storage keeps every existing reader and artifact identity."""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import pathlib
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ouroboros import artifact_storage as storage, artifacts, workspace_patch_capture


def _identity(raw):
    return {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _fake_ditto(monkeypatch, run):
    # Replace this module's binding, not subprocess.run used by Git and fixtures; a private file shares no block.
    monkeypatch.setattr(storage, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(storage, "subprocess", SimpleNamespace(
        run=run, DEVNULL=subprocess.DEVNULL, SubprocessError=subprocess.SubprocessError))
    monkeypatch.setattr(storage, "_may_share", lambda path: False)


@pytest.mark.parametrize("failure", ["unavailable", "timeout", "size", "sha256"])
def test_failed_compression_preserves_original_and_removes_temporary(tmp_path, monkeypatch, failure):
    from ouroboros.config import ARTIFACT_COMPRESSION_TIMEOUT_SEC

    path = tmp_path / "workspace.patch"
    raw = b"diff --git a/result.txt b/result.txt\n" * 4096
    path.write_bytes(raw)
    before, calls = path.stat(), []

    def refused(command, **kwargs):
        calls.append(command)
        assert command[:2] == ["/usr/bin/ditto", "--hfsCompression"]
        assert kwargs["timeout"] == ARTIFACT_COMPRESSION_TIMEOUT_SEC and kwargs["check"]
        assert not kwargs.get("shell", False)
        temporary = pathlib.Path(command[-1])
        assert temporary.parent == path.parent and temporary != path
        temporary.write_bytes(raw[:-1] if failure == "size" else b"X" + raw[1:])
        if failure == "unavailable":
            raise OSError("compression unavailable")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    _fake_ditto(monkeypatch, refused)
    assert storage.compress_in_place(path) is False
    assert len(calls) == 1
    assert path.read_bytes() == raw and path.stat().st_ino == before.st_ino
    assert artifacts.stream_artifact_file(path, expected=_identity(raw)) == _identity(raw)
    assert list(tmp_path.iterdir()) == [path]
    _assert_swept_after_a_hard_kill(tmp_path, pathlib.Path(calls[0][-1]).name)


def _assert_swept_after_a_hard_kill(tmp_path, temporary_name):
    """A hard kill between the sibling's write and its replace leaves it in the artifact folder: the
    maintenance child's ``temp_sweep`` reaps that name (the atomic signature), and the artifact
    listing never offers it as a deliverable; the former ``.<hex>.tmp`` name met neither."""
    import os
    import time

    from ouroboros.temp_sweep import sweep_stale_temp_files

    root = tmp_path / "data"
    folder = root / "task_results" / "artifacts" / "task-1"
    folder.mkdir(parents=True)
    leftover, former = folder / temporary_name, folder / ".0123456789abcdef0123456789abcdef.tmp"
    for path in (leftover, former):
        path.write_bytes(b"half a patch")
        os.utime(path, (time.time() - 7200, time.time() - 7200))
    assert temporary_name.startswith(".") and ".tmp." in temporary_name
    assert sweep_stale_temp_files(root, scripts=False) == 1
    assert not leftover.exists() and former.exists()


@pytest.mark.parametrize("platform", ["linux", "freebsd", "unknown"])
def test_unsupported_platform_keeps_ordinary_bytes(tmp_path, monkeypatch, platform):
    path = tmp_path / "workspace.patch"
    path.write_bytes(b"ordinary patch bytes\n")
    before = path.read_bytes()
    monkeypatch.setattr(storage, "sys", SimpleNamespace(platform=platform))
    assert storage.compress_in_place(path) is False
    assert path.read_bytes() == before


def _write_patch(tmp_path, monkeypatch, writer):
    content = b"The generated result retains the complete original source bytes.\n" * 8192
    if writer == "workspace":
        root, output = tmp_path / "repo", tmp_path / "artifacts"
        root.mkdir()
        subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True)
        (root / "result.txt").write_bytes(content)
        rows, manifest = workspace_patch_capture.write_workspace_patch_artifacts(root, output, task={})
        row = next(row for row in rows if row["kind"] == "workspace_patch")
        return pathlib.Path(row["path"]), manifest, row

    from ouroboros import delegate_custody
    from tests.test_delegated_skill_payload import _captured

    try:
        _ctx, _skill, _handle, _entry, capture = _captured(
            tmp_path, monkeypatch, edit=lambda root: (root / "notes.txt").write_bytes(content))
        assert capture["status"] == "ready_with_changes", capture
        manifest = json.loads(pathlib.Path(capture["manifest_artifact"]).read_text())
        return pathlib.Path(capture["patch_artifact"]), manifest, {
            "size": manifest["patch_size"], "sha256": manifest["sha256"]}
    finally:
        delegate_custody._CUSTODY.clear()


@pytest.mark.parametrize("writer", ["workspace", "payload"])
@pytest.mark.parametrize("mode", ["native", "refused", "corrupt", "unsupported"])
def test_both_patch_writers_publish_measured_final_bytes(tmp_path, monkeypatch, writer, mode):
    if mode == "native" and sys.platform != "darwin":
        pytest.skip("native APFS compression proof")
    if mode in {"refused", "corrupt"}:
        def refused(command, **_kwargs):
            if mode == "refused":
                raise OSError("compression unavailable")
            raw = pathlib.Path(command[-2]).read_bytes()
            pathlib.Path(command[-1]).write_bytes(b"X" + raw[1:])
        _fake_ditto(monkeypatch, refused)
    elif mode == "unsupported":
        monkeypatch.setattr(storage, "sys", SimpleNamespace(platform="linux"))

    events, captured = [], []
    compress, measure = storage.compress_in_place, artifacts.stream_artifact_file

    def observed_compression(path):
        raw = path.read_bytes()
        compressed = compress(path)
        captured.append((raw, compressed))
        events.append("compressed")
        return compressed

    def observed_measure(path, *args, **kwargs):
        if isinstance(path, pathlib.Path) and path.name == "workspace.patch":
            events.append("measured")
        return measure(path, *args, **kwargs)

    monkeypatch.setattr(storage, "compress_in_place", observed_compression)
    monkeypatch.setattr(workspace_patch_capture, "compress_in_place", observed_compression)
    monkeypatch.setattr(artifacts, "stream_artifact_file", observed_measure)
    path, manifest, row = _write_patch(tmp_path, monkeypatch, writer)
    assert len(captured) == 1 and events[-2:] == ["compressed", "measured"]
    raw, compressed = captured[0]
    assert compressed is (mode == "native")
    assert path.read_bytes() == raw
    assert measure(path, expected=row) == _identity(raw)
    assert (manifest["patch_size"], manifest["sha256"]) == (len(raw), _identity(raw)["sha256"])
    assert manifest["status"] == "ready_with_changes"
    # Read from a directory outside any repository: Git must actually see the patch paths.
    result = subprocess.run(["git", "apply", "--numstat", str(path)], cwd=tmp_path,
                            capture_output=True, check=True)
    assert result.stdout.strip()
    assert not list(path.parent.glob(".*.tmp"))
    if mode == "native":
        info = path.stat()
        assert info.st_flags & stat.UF_COMPRESSED
        assert info.st_blocks * 512 < info.st_size // 2


@pytest.mark.skipif(sys.platform != "darwin", reason="APFS compressed clone")
def test_compressed_patch_clone_and_streamed_fallback_keep_identity(tmp_path, monkeypatch):
    source = tmp_path / "workspace.patch"
    raw = b"+preserve the source and every byte of the patch\n" * 32768
    source.write_bytes(raw)
    assert storage.compress_in_place(source)
    expected = _identity(raw)
    cloned = tmp_path / "cloned.patch"
    assert storage.clone_file(source, cloned)
    assert cloned.stat().st_flags & stat.UF_COMPRESSED
    assert (source.stat().st_dev, source.stat().st_ino) != (cloned.stat().st_dev, cloned.stat().st_ino)
    assert artifacts.stream_artifact_file(cloned, expected=expected) == expected
    published = tmp_path / "published.patch"
    assert artifacts.copy_artifact_file(source, published, expected=expected) == expected
    assert published.stat().st_flags & stat.UF_COMPRESSED
    monkeypatch.setattr(artifacts, "clone_file", lambda *_: False)
    streamed = tmp_path / "streamed.patch"
    assert artifacts.copy_artifact_file(source, streamed, expected=expected) == expected
    assert streamed.read_bytes() == raw


@pytest.mark.parametrize("outcome", [True, False, "error"])
def test_ntfs_control_uses_a_live_handle_and_preserves_bytes(tmp_path, monkeypatch, outcome):
    path = tmp_path / "workspace.patch"
    raw = b"complete patch\n" * 8192  # past the 64 KiB floor
    path.write_bytes(raw)
    handles = []

    def control(handle, code, data, size, output, output_size, returned, overlapped):
        handles.append(handle)
        assert os.fstat(handle).st_size == len(raw)
        assert code == 0x9C040 and size == ctypes.sizeof(ctypes.c_ushort)
        assert ctypes.cast(data, ctypes.POINTER(ctypes.c_ushort)).contents.value == 1
        assert output is None and output_size == 0 and overlapped is None
        if outcome == "error":
            raise OSError("filesystem cannot compress")
        return outcome

    monkeypatch.setattr(storage, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setitem(sys.modules, "msvcrt", SimpleNamespace(get_osfhandle=lambda fd: fd))
    monkeypatch.setattr(ctypes, "WinDLL", lambda *a, **kw: SimpleNamespace(DeviceIoControl=control), raising=False)
    assert storage.compress_in_place(path) is (outcome is True)
    assert len(handles) == 1 and path.read_bytes() == raw
    with pytest.raises(OSError):
        os.fstat(handles[0])  # even an ioctl exception releases the handle


@pytest.mark.skipif(sys.platform != "win32", reason="native NTFS compression")
def test_native_ntfs_compression_keeps_identity(tmp_path):
    path = tmp_path / "workspace.patch"
    raw = b"+complete patch source\n" * 32768
    path.write_bytes(raw)
    assert storage.compress_in_place(path)
    assert path.stat().st_file_attributes & stat.FILE_ATTRIBUTE_COMPRESSED
    assert artifacts.stream_artifact_file(path, expected=_identity(raw)) == _identity(raw)
