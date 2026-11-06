"""Direct codec vetoes and reader regressions for the S5 reduction."""
import hashlib
import json
import lzma
import os
import subprocess

import pytest

from ouroboros import evidence_packer as packer, evidence_packs as packs
from tests.test_evidence_packs import _task
from tests.test_evidence_packs import root as root


def test_pack_uses_stdlib_compressor_and_exactly_one_xz_stream(root, monkeypatch):
    _task(root, "stream", blobs=3)
    # Prime only the platform tier probe: no codec or external helper may run below.
    packs.index_tier(root)
    calls = []
    compressor = lzma.LZMACompressor

    def compress(*args, **kwargs):
        calls.append(kwargs)
        return compressor(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(lzma, "LZMACompressor", compress)
        patch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("codec subprocess"))
        assert packer.run_pass(root)["faults"] == 0
    assert len(calls) == 1
    assert calls[0]["filters"][0]["preset"] == 1
    path = packs.packs_dir(root) / "stream.xz"
    decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ)
    raw = decoder.decompress(path.read_bytes())
    assert decoder.eof and decoder.unused_data == b""
    sidecar = json.loads(path.with_suffix(".members.json").read_text())
    assert len(raw) == sidecar["raw_total"]
    for member in sidecar["members"]:
        body = raw[member["offset"]:member["offset"] + member["size"]]
        assert len(body) == member["size"] and hashlib.sha256(body).hexdigest() == member["sha256"]


def test_reader_ascending_backward_and_replaced_pack_cursors(root):
    _, _ = _task(root, "cursor", blobs=2)
    packer.run_pass(root)
    sidecar = packs.read_sidecar(packs.packs_dir(root) / "cursor.members.json")
    path = packs.packs_dir(root) / "cursor.xz"
    expected = lzma.decompress(path.read_bytes())
    reader = packs._PackReader(path)

    def read(offset, size):
        chunks = []
        reader.read_at(offset, size, chunks.append)
        return b"".join(chunks)

    for member in [*sidecar["members"], sidecar["members"][0]]:
        assert read(member["offset"], member["size"]) == expected[member["offset"]:member["offset"] + member["size"]]
    replacement = path.with_suffix(".replacement")
    replacement.write_bytes(lzma.compress(b"replacement"))
    if os.name == "nt":
        # Windows cannot replace our open file. Keep the cached identity so the
        # next read must still notice replacement and reopen its stale cursor.
        reader._stream.close()
    replacement.replace(path)
    assert read(0, 11) == b"replacement"
    with pytest.raises(packs.EvidenceCorrupt, match="evidence_corrupt"):
        read(0, 12)


@pytest.mark.parametrize("damage", ["wrong_total", "escaped_handle", "overlap"])
def test_invalid_sidecar_structure_refuses_restore_and_preflight(root, damage):
    _task(root, "shape", blobs=1)
    packer.run_pass(root)
    path = packs.packs_dir(root) / "shape.members.json"
    original = path.read_bytes()
    sidecar = json.loads(original)
    if damage == "missing_header":
        del sidecar["pack_sha256"]
    elif damage == "wrong_total":
        sidecar["raw_total"] += 1
    elif damage == "escaped_handle":
        sidecar["members"][0]["path"] = "../../outside.md"
    else:
        sidecar["members"][1]["offset"] = 0
    path.write_text(json.dumps(sidecar), encoding="utf-8")
    for operation in (packs.pending_restore_bytes, packs.restore_checkpoint_handles):
        with pytest.raises(packs.SidecarCorrupt, match="evidence_corrupt"):
            operation(root)
    path.write_bytes(original)
    assert packs.pending_restore_bytes(root) > 0
    assert packs.restore_checkpoint_handles(root)["handles"] == 1
