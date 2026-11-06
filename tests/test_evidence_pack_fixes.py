"""Regressions for the dispositioned S5 pack and reader findings."""
from __future__ import annotations

import gzip
import json
import os
import pathlib

import pytest

from ouroboros import evidence_packer as packer
from ouroboros import evidence_packs as packs
from ouroboros import observability as obs
from ouroboros.artifacts import store_actor_source_bytes
from tests.test_evidence_packs import _handle_path, _task
from tests.test_evidence_packs import root as root


@pytest.mark.parametrize("damage", ["{", '{"format_version": 1}', "[]"])
def test_bad_sidecar_refuses_restore_and_packing_without_hiding_neighbour(root, damage):
    refs, handle = _task(root, "broken", blobs=1)
    other, _ = _task(root, "healthy", blobs=1)
    packer.run_pass(root)
    directory = packs.packs_dir(root)
    original = (directory / "broken.xz").read_bytes()
    sidecar = directory / "broken.members.json"
    intact = sidecar.read_bytes()
    sidecar.write_text(damage, encoding="utf-8")
    for operation in (packs.pending_restore_bytes, packs.restore_checkpoint_handles):
        with pytest.raises(OSError, match="evidence_corrupt"):
            operation(root)
    _task(root, "broken", blobs=1, payloads=[b"new evidence"])
    with pytest.raises(OSError, match="evidence_corrupt"):
        packs.pack_task(root, "broken", keep_shas=set())
    assert (directory / "broken.xz").read_bytes() == original
    (directory / packs.INDEX_NAME).unlink()
    assert packs.blob_bytes(root, other[0]).startswith(b'{"message"')
    with pytest.raises(OSError, match="evidence_corrupt"):
        packs.blob_bytes(root, refs[0])

    sidecar.write_bytes(intact)
    assert packs.pending_restore_bytes(root) > 0
    assert packs.restore_checkpoint_handles(root)["handles"] >= 1
    assert _handle_path(root, "broken", handle).exists()


@pytest.mark.parametrize("occupied", ["pack", "sidecar"])
def test_generation_is_never_reused_while_either_publication_file_exists(root, occupied):
    _task(root, "orphan", blobs=1)
    directory = packs.packs_dir(root)
    directory.mkdir(parents=True)
    path = directory / ("orphan.xz" if occupied == "pack" else "orphan.members.json")
    path.write_bytes(b"occupied generation")
    before = path.read_bytes()
    if occupied == "pack":
        report = packs.pack_task(root, "orphan", keep_shas=set())
        assert report["generation"] == 2
    else:
        with pytest.raises(OSError, match="evidence_corrupt"):
            packs.pack_task(root, "orphan", keep_shas=set())
    assert path.read_bytes() == before
    with pytest.raises(FileExistsError):
        packs.write_pack(root, "orphan", 1, packs.loose_members(root, "orphan"))
    assert path.read_bytes() == before


@pytest.mark.parametrize("placement", ["missing", "valid", "corrupt_short", "corrupt_same_size"])
def test_presence_reentry_projection_reads_loose_and_packed_source(root, monkeypatch, placement):
    from ouroboros import artifacts
    from ouroboros.presence_continuation import reentry_source_projection

    _task(root, "presence", blobs=1)
    raw = json.dumps({"kind": "presence_reentry", "task_id": "presence", "conversation_key": "room",
                      "history": {"status": "ok", "start": 0, "end": 1}}).encode()
    ref = store_actor_source_bytes(root, "presence", category="context_checkpoints",
                                  source_id="presence-reentry", data=raw, extension="json")
    before = reentry_source_projection(root, "presence", ref["sha256"])
    assert before["complete_sha256"] == ref["sha256"]
    assert packer.run_pass(root)["faults"] == 0

    loose = _handle_path(root, "presence", ref)
    assert not loose.exists()
    if placement != "missing":
        loose.write_bytes(raw if placement == "valid" else b"bad" if placement == "corrupt_short" else b"x" * len(raw))
    assert reentry_source_projection(root, "presence", ref["sha256"]) == before
    assert packs.handle_info(root, "presence", ref["path"]) == {
        "size": len(raw), "sha256": ref["sha256"], "loose": placement == "valid"}
    assert reentry_source_projection(root, "presence", "0" * 64)["reason"] == "source_unavailable"
    monkeypatch.setattr(artifacts, "read_actor_source_bytes", lambda *a, **k: (_ for _ in ()).throw(OSError("read failed")))
    assert reentry_source_projection(root, "presence", ref["sha256"])["reason"] == "source_unreadable"


def test_blob_byte_cap_bounds_loose_reads_and_refuses_oversize_pack_before_decode(root, monkeypatch):
    raw = b'{"body": "' + b"x" * 10000 + b'"}'
    refs, _ = _task(root, "bounded", blobs=1, payloads=[raw])
    no_size = {key: value for key, value in refs[0].items() if key != "size"}
    sizes = []
    original_read = gzip.GzipFile.read

    def read(stream, size=-1):
        sizes.append(size)
        return original_read(stream, size)

    with monkeypatch.context() as patch:
        patch.setattr(gzip.GzipFile, "read", read)
        with pytest.raises(ValueError, match="byte limit"):
            obs.read_blob_ref(root, no_size, max_bytes=100, raw_bytes=True)
    assert sizes and all(0 <= size <= 101 for size in sizes)
    assert obs.read_blob_ref(root, no_size, max_bytes=len(raw), raw_bytes=True) == raw
    with pytest.raises(ValueError, match="size"):
        obs.read_blob_ref(root, no_size)
    packer.run_pass(root)
    with monkeypatch.context() as patch:
        patch.setattr(packs, "member_bytes", lambda *a: pytest.fail("oversize member decoded"))
        with pytest.raises(ValueError, match="byte limit"):
            obs.read_blob_ref(root, no_size, max_bytes=100, raw_bytes=True)
    assert obs.read_blob_ref(root, no_size, max_bytes=len(raw), raw_bytes=True) == raw


@pytest.mark.parametrize("kind", ["task", "archive"])
def test_unpacker_is_published_before_unlink_and_write_failure_keeps_loose(root, monkeypatch, kind):
    if kind == "task":
        refs, handle = _task(root, "manual", blobs=1)
        loose = {pathlib.Path(refs[0]["path"]), _handle_path(root, "manual", handle)}
    else:
        source = root / "archive/usage_ledger/segment_ep1.jsonl"
        source.parent.mkdir(parents=True)
        source.write_text('{"attempt_id":"example"}\n', encoding="utf-8")
        loose = {source}
    original_write, original_unlink = packs.write_bytes_atomic, os.unlink

    def fail_readme(path, *args, **kwargs):
        if pathlib.Path(path).name == packs.README_NAME:
            raise PermissionError("read-only unpacker destination")
        return original_write(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(packs, "write_bytes_atomic", fail_readme)
        report = packer.run_pass(root)
    assert report["faults"] >= 1 and all(path.exists() for path in loose)

    def unlink(path, *args, **kwargs):
        if pathlib.Path(path) in loose:
            assert (packs.packs_dir(root) / packs.UNPACKER_NAME).read_text() == packs._UNPACKER
            assert (packs.packs_dir(root) / packs.README_NAME).read_text() == packs._README
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", unlink)
    assert packer.run_pass(root)["faults"] == 0
    assert not any(path.exists() for path in loose)


def test_switch_preflight_refuses_corrupt_sidecar_before_export_or_owner(root, monkeypatch):
    from ouroboros import store, store_import, store_switch

    _task(root, "switch", blobs=1)
    packer.run_pass(root)
    sidecar = packs.packs_dir(root) / "switch.members.json"
    original = sidecar.read_bytes()
    sidecar.write_text("{", encoding="utf-8")
    store_import.ensure_current_store(root)
    monkeypatch.setattr(store_switch, "switch_needs", lambda *a: {"handles": True, "money": False, "facts": True})
    monkeypatch.setattr(store_import, "export_facts", lambda *a: pytest.fail("export before corrupt-sidecar refusal"))
    store_switch.record_switch_intent(root, target_sha="example-target", target_ref="example", reason="test", door="ui_rollback")
    report = store_switch.run_switch_seam(root, repo_dir=root, owner=lambda *a: pytest.fail("owner called"))
    assert report["status"] == "refused" and "SidecarCorrupt" in report["reason"]
    with store.open_store(root, write=False) as txn:
        assert txn.phase() == store.PUBLISHED
    sidecar.write_bytes(original)
    assert packs.pending_restore_bytes(root) > 0
