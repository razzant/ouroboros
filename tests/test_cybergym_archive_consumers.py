"""Synthetic source/output/telemetry consumers; no upstream data or paid calls."""
from __future__ import annotations

import gzip
import hashlib
import io
import os
import tarfile
from types import SimpleNamespace

import pytest

from devtools.benchmarks.cybergym import cybergym_executor as executor
from devtools.benchmarks.cybergym import cybergym_lifecycle as lifecycle
from devtools.benchmarks.cybergym import cybergym_wire as wire
from devtools.benchmarks.cybergym.cybergym_adapter import FinalPocRefused, TaskSpec, final_poc_record
from tests.test_cybergym_executor import _config, _write_archive, _DESCRIPTOR_SAFE_EXTRACT


@pytest.mark.skipif(not _DESCRIPTOR_SAFE_EXTRACT, reason="requires descriptor-safe archive primitives")
@pytest.mark.parametrize("limit,value,entries,message", [
    ("_MAX_ARCHIVE_MEMBERS", 1, [("src/one", "file", "a"), ("src/two", "file", "b")], "member count"),
    ("_MAX_ARCHIVE_FILE_BYTES", 3, [("src/large", "file", "four")], "member size"),
    ("_MAX_ARCHIVE_TOTAL_BYTES", 3, [("src/one", "file", "ab"), ("src/two", "file", "cd")], "total size"),
])
def test_archive_limits_refuse_before_publication(tmp_path, monkeypatch, limit, value, entries, message):
    archive = tmp_path / "repo-vul.tar.gz"
    _write_archive(archive, entries)
    monkeypatch.setattr(executor, limit, value)
    destination = tmp_path / "workspace"
    with pytest.raises(executor.ExecutorFailure, match=message):
        executor._safe_extract(archive, destination)
    assert list(destination.iterdir()) == []
    assert not list(tmp_path.glob(".workspace.extract-*"))


@pytest.mark.skipif(not _DESCRIPTOR_SAFE_EXTRACT, reason="requires descriptor-safe archive primitives")
@pytest.mark.parametrize("kind", [tarfile.XHDTYPE, tarfile.GNUTYPE_LONGNAME])
def test_archive_rejects_metadata_size_before_first_member(tmp_path, monkeypatch, kind):
    # Header only: refusal must happen before tarfile tries to allocate/read its payload.
    member = tarfile.TarInfo("extended-header")
    member.type, member.size = kind, 2048
    archive = tmp_path / "repo-vul.tar.gz"
    archive.write_bytes(gzip.compress(member.tobuf()))
    monkeypatch.setattr(executor, "_MAX_ARCHIVE_READ_BYTES", 1024)
    with pytest.raises(executor.ExecutorFailure, match="metadata read"):
        executor._safe_extract(archive, tmp_path / "workspace")
    assert list((tmp_path / "workspace").iterdir()) == []


@pytest.mark.skipif(not _DESCRIPTOR_SAFE_EXTRACT, reason="requires descriptor-safe archive primitives")
def test_archive_sparse_logical_size_is_bounded(tmp_path, monkeypatch):
    archive = tmp_path / "repo-vul.tar.gz"
    with tarfile.open(archive, "w:gz", format=tarfile.PAX_FORMAT) as tar:
        member = tarfile.TarInfo("src/sparse")
        member.size = 1
        member.pax_headers = {"GNU.sparse.map": "0,1", "GNU.sparse.size": "1000000"}
        tar.addfile(member, io.BytesIO(b"x"))
    monkeypatch.setattr(executor, "_MAX_ARCHIVE_FILE_BYTES", 1024)
    with pytest.raises(executor.ExecutorFailure, match="member size"):
        executor._safe_extract(archive, tmp_path / "workspace")
    assert list((tmp_path / "workspace").iterdir()) == []


@pytest.mark.skipif(not _DESCRIPTOR_SAFE_EXTRACT, reason="requires descriptor-safe archive primitives")
def test_archive_small_legitimate_input_and_links_survive_bounds(tmp_path, monkeypatch):
    archive = tmp_path / "repo-vul.tar.gz"
    _write_archive(archive, [("src/wiki:page\n1", "file", "okay"), ("src/link", "symlink", "/missing/target")])
    monkeypatch.setattr(executor, "_MAX_ARCHIVE_MEMBERS", 2)
    monkeypatch.setattr(executor, "_MAX_ARCHIVE_FILE_BYTES", 4)
    monkeypatch.setattr(executor, "_MAX_ARCHIVE_TOTAL_BYTES", 4)
    executor._safe_extract(archive, tmp_path / "workspace")
    assert (tmp_path / "workspace/src/wiki:page\n1").read_bytes() == b"okay"
    assert os.readlink(tmp_path / "workspace/src/link") == "/missing/target"


@pytest.mark.skipif(not _DESCRIPTOR_SAFE_EXTRACT, reason="requires descriptor-safe archive primitives")
@pytest.mark.parametrize("compressed_bytes", [2_267_326_222, 2_164_985_032])
def test_archive_compressed_length_does_not_limit_streamed_input(tmp_path, monkeypatch, compressed_bytes):
    # Compressed lengths of two real archives at the pinned dataset revision.
    # Model their fstat metadata while exercising extraction with a small payload.
    archive = tmp_path / "repo-vul.tar.gz"
    _write_archive(archive, [("src/input", "file", "okay")])
    source = archive.stat()
    original_fstat = os.fstat

    def reported_size(fd):
        info = original_fstat(fd)
        if (info.st_dev, info.st_ino) == (source.st_dev, source.st_ino):
            return SimpleNamespace(st_mode=info.st_mode, st_size=compressed_bytes)
        return info

    monkeypatch.setattr(executor.os, "fstat", reported_size)
    executor._safe_extract(archive, tmp_path / "workspace")
    assert (tmp_path / "workspace/src/input").read_bytes() == b"okay"


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="POSIX workspace contract")
@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_host_consumers_refuse_links_and_fifos(tmp_path, kind):
    target = tmp_path / "outside"
    target.write_text("outside sentinel", encoding="utf-8")
    for name in ("repo-vul.tar.gz", "submit.sh", "final.poc"):
        path = tmp_path / name
        if kind == "symlink":
            path.symlink_to(target)
        else:
            os.mkfifo(path)
    with pytest.raises(executor.ExecutorFailure):
        executor._safe_extract(tmp_path / "repo-vul.tar.gz", tmp_path / "workspace")
    with pytest.raises(executor.ExecutorFailure):
        lifecycle._masked_id_from_submit_script(tmp_path / "submit.sh")
    with pytest.raises(FinalPocRefused, match="regular|cannot be opened"):
        final_poc_record(tmp_path)
    assert target.read_text(encoding="utf-8") == "outside sentinel"


def test_workspace_text_is_complete_and_bounded(tmp_path):
    path = tmp_path / "description.txt"
    path.write_bytes(b"abcdef")
    assert lifecycle._read_text(path, "description", limit=6) == "abcdef"
    with pytest.raises(executor.ExecutorFailure, match="exceeds"):
        lifecycle._read_text(path, "description", limit=5)


def test_final_copy_publishes_only_matching_hashed_bytes(tmp_path):
    source = tmp_path / "source/final.poc"
    source.parent.mkdir()
    source.write_bytes(b"chosen input")
    digest = final_poc_record(source).sha256
    destination = tmp_path / "final.poc"
    lifecycle._copy_final_poc(source, destination, digest)
    assert destination.read_bytes() == b"chosen input"
    source.write_bytes(b"replacement")
    with pytest.raises(executor.ExecutorFailure, match="changed before copy"):
        lifecycle._copy_final_poc(source, destination, digest)
    assert destination.read_bytes() == b"chosen input"
    assert not list(tmp_path.glob(".final-poc-*"))


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="POSIX workspace contract")
def test_final_copy_refuses_replaced_symlink_and_regrade_never_resolves_it(tmp_path, monkeypatch):
    source = tmp_path / "source/final.poc"
    source.parent.mkdir()
    source.write_bytes(b"chosen")
    digest = final_poc_record(source).sha256
    outside = tmp_path / "outside/final.poc"
    outside.parent.mkdir()
    outside.write_bytes(b"must not copy")
    source.unlink()
    source.symlink_to(outside)
    destination = tmp_path / "final.poc"
    with pytest.raises(FinalPocRefused):
        lifecycle._copy_final_poc(source, destination, digest)
    assert not destination.exists()
    runner = executor.CyberGymExecutor(_config(tmp_path))
    monkeypatch.setattr(runner, "start", lambda: pytest.fail("invalid marker must fail before startup"))
    with pytest.raises(FinalPocRefused):
        runner.regrade_final_poc(TaskSpec("arvo:1", "arvo"), source, tmp_path / "result")


def test_final_copy_uses_one_open_inode(tmp_path, monkeypatch):
    source = tmp_path / "source/final.poc"
    source.parent.mkdir()
    source.write_bytes(b"chosen")
    digest = final_poc_record(source).sha256
    outside = tmp_path / "outside"
    outside.write_bytes(b"private sentinel")
    original = lifecycle.final_poc_record

    def replace_after_read(*args, **kwargs):
        result = original(*args, **kwargs)
        source.unlink()
        source.symlink_to(outside)
        return result

    monkeypatch.setattr(lifecycle, "final_poc_record", replace_after_read)
    destination = tmp_path / "final.poc"
    lifecycle._copy_final_poc(source, destination, digest)
    assert destination.read_bytes() == b"chosen"


@pytest.mark.parametrize("variant", ["valid", "corrupt", "escaped"])
def test_compressed_telemetry_consumer_is_bounded(tmp_path, monkeypatch, variant):
    cap = 512
    monkeypatch.setattr(wire, "_MAX_TELEMETRY_REF_BYTES", cap)
    raw = b'{"evidence": "' + b"a" * (cap if variant == "expanded" else 10) + b'"}'
    path = tmp_path / "payload.json.gz"
    path.write_bytes(gzip.compress(raw))
    if variant == "compressed":
        path.write_bytes(b"x" * (cap + 1))
    if variant == "corrupt":
        path.write_bytes(b"bad gzip")
    if variant == "deflate":
        path.write_bytes(bytes.fromhex("1f8b0800000000000000") + b"\xff" * 32)
    reads = []
    original_read = gzip.GzipFile.read

    def bounded_read(handle, size=-1):
        reads.append(size)
        assert 0 <= size <= cap + 1
        return original_read(handle, size)

    monkeypatch.setattr(gzip.GzipFile, "read", bounded_read)
    roots = [tmp_path / "approved"] if variant == "escaped" else [tmp_path]
    ref = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw), "kind": "json", "encoding": "gzip"}
    result = wire._read_json_ref(ref, roots, compressed=True)
    assert result == ({"evidence": "a" * 10} if variant == "valid" else None)
    if variant in {"valid", "expanded"}:
        assert reads == [cap + 1]
