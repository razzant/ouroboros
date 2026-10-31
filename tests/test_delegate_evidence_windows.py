"""Native filesystem controls for immutable delegated evidence; run on Windows CI.

The copy stays because a Git repository the run made inside it cannot be saved in its
result (``tests/test_delegate_capture_generations._prepared``), so later evidence
generations keep happening on it."""
from __future__ import annotations

import os
import pathlib
import stat

import pytest

from ouroboros import delegate_collection as body, delegate_named_evidence as named, review_checking as checking
from tests.test_delegate_capture_generations import _captured
from tests.test_recovered_capture import EVIDENCE

pytestmark = pytest.mark.skipif(os.name != "nt", reason="requires native Windows attributes, sharing and junctions")


@pytest.mark.parametrize("obstacle", ["readonly_attribute", "non_delete_sharing_handle", "junction_parent"])
def test_new_generation_preserves_a_protected_old_destination(tmp_path, monkeypatch, obstacle):
    import ctypes
    from ctypes import wintypes
    import _winapi

    entry, area = _captured(tmp_path, monkeypatch)
    receipt, old = body.saved_named_evidence(tmp_path, entry)
    old_file = receipt.parent / "files" / checking.EVIDENCE_DIRNAME / "result.bin"
    original = receipt.read_bytes()
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "result.bin"
    victim.write_bytes(EVIDENCE)
    abandoned = receipt.parent.parent / ".evidence-abandoned.pending"
    handle = None
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p]
    kernel.CreateFileW.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.restype = wintypes.BOOL
    try:
        if obstacle == "readonly_attribute":
            old_file.chmod(0o444)
            assert old_file.stat().st_file_attributes & stat.FILE_ATTRIBUTE_READONLY
            with pytest.raises(PermissionError):
                old_file.write_bytes(b"raw overwrite must fail")
        elif obstacle == "non_delete_sharing_handle":
            handle = kernel.CreateFileW(str(old_file), 0x80000000, 3, None, 3, 0x80, None)
            assert handle not in (None, ctypes.c_void_p(-1).value), ctypes.get_last_error()
            with pytest.raises(OSError) as failure:
                old_file.rename(old_file.with_name("raw-rename"))
            assert failure.value.winerror in (32, 33)
        else:
            old_file.unlink()
            old_file.parent.rmdir()
            _winapi.CreateJunction(str(outside), str(old_file.parent))
            _winapi.CreateJunction(str(outside), str(abandoned))
            assert named.linked(old_file.parent)
        source = area / checking.EVIDENCE_DIRNAME / "result.bin"
        source.chmod(0o644)
        source.write_bytes(b"changed source")
        with monkeypatch.context() as patch:
            patch.setattr(os, "chmod", lambda *a, **k: pytest.fail("retention changed old attributes"))
            latest = named.run_named_evidence(tmp_path, entry)
            assert named.run_named_evidence(tmp_path, entry) == latest
        assert latest["status"] == "retained" and latest["evidence_dir"] != old["evidence_dir"]
        assert receipt.read_bytes() == original and old_file.read_bytes() == EVIDENCE
        assert victim.read_bytes() == EVIDENCE
        assert not abandoned.exists()
        fresh = pathlib.Path(latest["evidence_dir"]) / "files" / checking.EVIDENCE_DIRNAME / "result.bin"
        assert fresh.read_bytes() == b"changed source"
    finally:
        if handle not in (None, ctypes.c_void_p(-1).value):
            assert kernel.CloseHandle(handle)
        if obstacle == "readonly_attribute":
            old_file.chmod(0o644)


def test_capture_parent_junction_is_refused_without_touching_its_target(tmp_path):
    import _winapi

    outside = tmp_path / "outside"
    outside.mkdir()
    alias = tmp_path / "alias"
    _winapi.CreateJunction(str(outside), str(alias))
    with pytest.raises(OSError, match="private directory"):
        named.private_directory(alias / "files", tmp_path)
    assert not (outside / "files").exists()
    named.private_directory(tmp_path / "ordinary" / "files", tmp_path)
    assert (tmp_path / "ordinary" / "files").is_dir()
