"""History retention scales with immutable sources, not paths through them."""

import json
import shutil
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import artifacts, context_compaction as cc, headless, observability
from ouroboros.task_results import write_task_result
from tests._store_shared import remove_store_tree
from tests.test_context_reclaim_materializer import _request, _unit


def test_shared_checkpoint_history_has_bounded_parsing_and_survives_gc(tmp_path, monkeypatch):
    parent = tmp_path / "canonical"
    parent.mkdir()
    task_id = "shared-checkpoint-history"
    child = headless.prepare_task_drive(parent, task_id, "empty")
    capsules, refs, originals = [], [], {}
    for index in range(90):
        messages = [*capsules, *_unit(f"read-{index}", "x")]
        request = _request(messages, 100)
        unit = cc._atomic_units(messages)[-1]
        ref = cc._persist_reclaim_checkpoint(
            messages, request,
            SimpleNamespace(fingerprint=f"{index:064x}", units=[]),
            drive_root=child, task_id=task_id,
        )
        capsule, _ = cc._capsule_message(
            unit, f"summary {index}",
            [cc._part(unit.unit_id, unit.source_text)], ref, request,
        )
        refs.append(ref)
        originals[ref["path"]] = artifacts.read_actor_source_bytes(child, task_id, ref)
        capsules.append(capsule)

    child_result = write_task_result(
        child, task_id, "completed", result="saved answer", artifact_status="ready",
        review_evidence={"source_refs": [refs[-1], refs[-1]]},
    )
    write_task_result(parent, task_id, "running", child_drive_root=str(child))
    checkpoint_bodies = {*originals.values(), *(raw.decode("utf-8") for raw in originals.values())}
    parsed = Counter()
    original_loads = json.loads

    def counted_loads(value, *args, **kwargs):
        if isinstance(value, (str, bytes)) and value in checkpoint_bodies:
            parsed[value] += 1
            if sum(parsed.values()) > 4 * len(refs):
                pytest.fail("shared ancestors were parsed per path instead of per immutable node")
        return original_loads(value, *args, **kwargs)

    with monkeypatch.context() as counting:
        counting.setattr(json, "loads", counted_loads)
        _result, promotion = observability.promote_child_task_refs(parent, child, task_id, child_result)

    assert promotion["status"] == "complete"
    # Allow separate discovery/verification reads, but never expansion once per
    # path through shared ancestry. This bound fails on the pre-fix walker.
    assert sum(parsed.values()) <= 4 * len(refs), {
        "parses": sum(parsed.values()), "sources": len(refs),
        "per_body": sorted(parsed.values()),
    }
    remove_store_tree(child)
    for ref in refs:
        assert artifacts.read_actor_source_bytes(parent, task_id, ref) == originals[ref["path"]]


def test_exact_checkpoint_keeps_old_manifest_identity_beside_derived_version(tmp_path):
    parent, task = tmp_path / "canonical", "checkpoint-versions"
    parent.mkdir()
    child = headless.prepare_task_drive(parent, task, "empty")
    call = observability.persist_call(child, task_id=task, call_id="tool-original", call_type="tool_call",
                                     payload={"tool": "read_file", "result": "original physical result"})
    old = call["manifest_ref"]
    old_bytes = Path(old["path"]).read_bytes()
    messages = _unit("read", "x")
    request = _request(messages, 10)
    first = cc._persist_reclaim_checkpoint(messages, request,
        SimpleNamespace(fingerprint="first", units=[]), drive_root=child, task_id=task)
    unit = cc._atomic_units(messages, trace_refs_by_tool_call_id={"read": call})[-1]
    capsule, _ = cc._capsule_message(unit, "summary",
                                    [cc._part(unit.unit_id, unit.source_text)], first, request)
    later_messages = [capsule, *_unit("next", "y")]
    later = cc._persist_reclaim_checkpoint(later_messages, _request(later_messages, 10),
        SimpleNamespace(fingerprint="later", units=[]), drive_root=child, task_id=task)
    exact = artifacts.read_actor_source_bytes(child, task, later)
    original_result = {"review_evidence": {"source_refs": [later]},
                       "trace_refs": {"tool_call_refs": [{"manifest_ref": old}]}}
    copied, state = observability.promote_child_task_refs(parent, child, task, original_result)
    assert state["status"] == "complete" and not state["unavailable_refs"]
    new = copied["trace_refs"]["tool_call_refs"][0]["manifest_ref"]
    assert new["sha256"] != old["sha256"]
    shutil.rmtree(child)
    assert artifacts.read_actor_source_bytes(parent, task, later) == exact
    from ouroboros.evidence_packs import blob_bytes
    from ouroboros.source_retention import manifest_version_ref
    assert blob_bytes(parent, manifest_version_ref(parent, old["sha256"])) == old_bytes
    for ref in (old, new):
        manifest = observability.read_call_manifest_ref(parent, ref, task_id=task)
        assert observability.read_blob_ref(parent, manifest["full_payload_ref"])["result"] == "original physical result"
    assert observability.read_call_manifest_ref(parent, new, task_id=task)["promoted_call_manifest"] is True


def test_inventory_holds_unreferenced_calls_and_stop_interrupts_work(tmp_path):
    from ouroboros.task_custody import PublicationClosed, publication_fence

    parent, task = tmp_path / "canonical", "inventory"
    parent.mkdir()
    child = headless.prepare_task_drive(parent, task, "empty")
    call = observability.persist_call(child, task_id=task, call_id="unreferenced", call_type="context_compaction_map",
                                     payload={"answer": "forensic evidence outside terminal refs"})
    with publication_fence(lambda: True), pytest.raises(PublicationClosed):
        observability.promote_child_task_refs(parent, child, task, {})
    _, state = observability.promote_child_task_refs(parent, child, task, {})
    assert state["call_inventory_preserved"] is True
    assert state["call_inventory_mtime_ns"] == (child / "observability/calls" / task).stat().st_mtime_ns
    shutil.rmtree(child)
    manifest = observability.read_call_manifest_ref(parent, call["manifest_ref"], task_id=task)
    assert observability.read_blob_ref(parent, manifest["full_payload_ref"])["answer"].startswith("forensic evidence")


def test_deferred_sources_read_only_from_own_retained_child(tmp_path):
    parent, task = tmp_path / "canonical", "deferred"
    parent.mkdir()
    child = headless.prepare_task_drive(parent, task, "empty")
    ref = artifacts.store_actor_source_bytes(child, task, category="tool_results", source_id="answer",
                                            data=b"retained exact bytes", extension="txt")
    call = observability.persist_call(child, task_id=task, call_id="deferred-call", call_type="llm_response",
                                     payload={"answer": "before background placement"})
    assert artifacts.read_actor_source_bytes(parent, task, ref) == b"retained exact bytes"
    manifest = observability.read_call_manifest_ref(parent, call["manifest_ref"], task_id=task)
    assert observability.read_blob_ref(parent, manifest["full_payload_ref"])["answer"] == "before background placement"
    _manifest, payload, _call_ref = observability.read_call_payload(parent, task_id=task, call_id="deferred-call")
    assert payload["answer"] == "before background placement"
    foreign = tmp_path / "foreign"
    foreign_call = observability.persist_call(foreign, task_id=task, call_id="foreign", call_type="llm_response",
                                             payload={"answer": "foreign unreadable"})
    with pytest.raises(ValueError, match="outside"):
        observability.read_call_manifest_ref(parent, foreign_call["manifest_ref"], task_id=task)


def test_shared_missing_dependency_stays_unavailable_and_retry_can_recover(tmp_path):
    parent, task = tmp_path / "canonical", "missing"
    parent.mkdir()
    child = headless.prepare_task_drive(parent, task, "empty")
    ref = artifacts.store_actor_source_bytes(child, task, category="tool_results", source_id="source",
                                            data=b"exact source", extension="txt")
    path = artifacts.task_artifact_dir_path(child, task) / ref["path"]
    path.unlink()
    body = {"review_evidence": {"source_refs": [ref, ref]}}
    copied, state = observability.promote_child_task_refs(parent, child, task, body)
    assert len(state["unavailable_refs"]) == 1 and state["unavailable_refs"][0]["path"] == ref["path"]
    assert copied == body  # both carriers keep their reference; the failure is a promotion fact
    path.write_bytes(b"exact source")
    # The live retry walks the row the previous pass published, not the original body.
    copied, state = observability.promote_child_task_refs(parent, child, task, copied)
    assert not state["unavailable_refs"] and not state["pending_refs"]
    shutil.rmtree(child)
    assert artifacts.read_actor_source_bytes(parent, task, copied["review_evidence"]["source_refs"][0]) == b"exact source"


def test_deep_producer_checkpoint_chain_has_no_graph_recursion(tmp_path):
    parent, task = tmp_path / "canonical", "deep-history"
    parent.mkdir()
    child = headless.prepare_task_drive(parent, task, "empty")
    capsule, refs = None, []
    for index in range(160):
        messages = ([capsule] if capsule else []) + _unit(f"read-{index}", "x")
        request = _request(messages, 10)
        unit = cc._atomic_units(messages)[-1]
        ref = cc._persist_reclaim_checkpoint(messages, request,
            SimpleNamespace(fingerprint=f"deep-{index}", units=[]), drive_root=child, task_id=task)
        capsule, _ = cc._capsule_message(unit, "summary",
                                        [cc._part(unit.unit_id, unit.source_text)], ref, request)
        refs.append(ref)
    _, state = observability.promote_child_task_refs(parent, child, task,
        {"review_evidence": {"source_refs": [refs[-1]]}})
    assert state["status"] == "complete" and not state["unavailable_refs"]
    shutil.rmtree(child)
    for ref in refs:
        assert len(artifacts.read_actor_source_bytes(parent, task, ref)) == ref["size"]


def test_payload_memory_does_not_grow_with_retained_history(tmp_path):
    import gc
    import tracemalloc

    def peak(name, count, *, measured=True):
        parent, task = tmp_path / name, f"memory-{name}"
        parent.mkdir()
        child = headless.prepare_task_drive(parent, task, "empty")
        refs = [observability.write_blob(child, {"text": (str(index) + "x" * 262_144)}) for index in range(count)]
        gc.collect()
        enabled, high = gc.isenabled(), None
        try:
            if measured:  # no GC pass inside: payloads a reference cycle holds count, whatever the thresholds
                gc.disable()
                tracemalloc.start()
            with observability.child_ref_promotion_scope():
                _, state = observability.promote_child_task_refs(
                    parent, child, task, {"review_evidence": {"source_refs": refs}})
                if measured:
                    _current, high = tracemalloc.get_traced_memory()
        finally:
            if measured:
                tracemalloc.stop()
                if enabled:
                    gc.enable()
        assert state["status"] == "complete" and not state["unavailable_refs"], state
        remove_store_tree(child)
        for index, ref in enumerate(refs):
            assert observability.read_blob_ref(parent, ref) == {"text": str(index) + "x" * 262_144}
        return high

    peak("warmup", 16, measured=False)  # one-time imports and caches land in neither measurement
    small, large = peak("small", 16), peak("large", 64)
    # Metadata grows with references; another 48 x 256 KiB of payload must not
    # remain live, in the operation memo or in a reference cycle. This is a test
    # allowance, no runtime cap.
    assert large - small < 4 * 1024 * 1024, (small, large)


def test_imported_version_never_hides_native_seal_under_same_call_id(tmp_path, monkeypatch):
    import hashlib
    from ouroboros import model_send_seal
    from ouroboros.llm import _canonical_candidate_bytes

    parent, task = tmp_path / "canonical", "seal-collision"
    parent.mkdir()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(parent))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(parent / "settings.json"))
    child = headless.prepare_task_drive(parent, task, "empty")

    def capture(root, text):
        payload = {"model": "gpt-x", "messages": [{"role": "user", "content": text}]}
        raw = _canonical_candidate_bytes(payload)
        return observability.persist_physical_candidate(root, task_id=task, attempt_id="a" * 32,
            candidate=payload, candidate_facts={"candidate_raw_sha256": hashlib.sha256(raw).hexdigest(),
                "candidate_raw_size_bytes": len(raw), "candidate_measurement_kind": "canonical_json_v1"})["manifest_ref"]

    native = capture(parent, "native physical request")
    imported = capture(child, "different imported request")
    before = Path(native["path"]).read_bytes()
    transferred = observability.promote_call_manifest_ref(child, parent, imported, task_id=task)
    assert Path(native["path"]).read_bytes() == before
    shutil.rmtree(child)
    for ref, expected in ((native, "native physical request"), (transferred, "different imported request")):
        manifest = observability.read_call_manifest_ref(parent, ref, task_id=task)
        payload = observability.read_blob_ref(parent, manifest["full_payload_ref"])
        assert payload["messages"][0]["content"] == expected
    report = model_send_seal.reconcile_model_send_seals(parent)
    assert report["seals"] == 1 and report["orphan_seals"] == 1, report


def test_exact_local_blob_wins_old_locator_and_own_retained_copy_repairs_corruption(tmp_path):
    from ouroboros.utils import write_bytes_atomic

    parent, task = tmp_path / "canonical", "blob-resolution"
    parent.mkdir()
    child = headless.prepare_task_drive(parent, task, "empty")
    original = observability.write_blob(child, {"text": "exact retained payload"})
    source_path = Path(original["path"])
    compressed = source_path.read_bytes()
    body = {"review_evidence": {"source_ref": original}}
    copied, _state = observability.promote_child_task_refs(parent, child, task, body)
    target = Path(copied["review_evidence"]["source_ref"]["path"])
    # Runtime repairs replace a placement; in-place writes would damage every
    # hardlink to this immutable inode, not just the selected locator.
    write_bytes_atomic(source_path, b"corrupt former locator")
    assert observability.read_blob_ref(parent, original) == {"text": "exact retained payload"}
    write_bytes_atomic(source_path, compressed)
    write_bytes_atomic(target, b"corrupt canonical copy")
    assert observability.read_blob_ref(parent, original) == {"text": "exact retained payload"}
    _copied, state = observability.promote_child_task_refs(parent, child, task, body)
    assert state["status"] == "complete" and not state["unavailable_refs"]
    shutil.rmtree(child)
    assert observability.read_blob_ref(parent, original) == {"text": "exact retained payload"}


def _checkpoint_chain(child, task, count):
    capsule, refs = None, []
    for index in range(count):
        messages = ([capsule] if capsule else []) + _unit(f"read-{index}", "x")
        request = _request(messages, 10)
        unit = cc._atomic_units(messages)[-1]
        ref = cc._persist_reclaim_checkpoint(messages, request,
            SimpleNamespace(fingerprint=f"chain-{index}", units=[]), drive_root=child, task_id=task)
        capsule, _ = cc._capsule_message(unit, "summary", [cc._part(unit.unit_id, unit.source_text)], ref, request)
        refs.append(ref)
    return refs


def test_a_child_checkpoint_held_only_in_its_pack_is_copied_without_a_second_decode(tmp_path, monkeypatch):
    """The link before a copy compares the child's own loose file: a checkpoint the child holds only as a pack
    member, or beside a damaged loose name, has no inode to share, so its verified bytes are copied without
    decoding the member again and nothing damaged is linked into the parent."""
    from ouroboros import evidence_packer, source_retention
    from ouroboros import evidence_packs as packs
    from tests._packed_read_probe import count_packed_reads, release_pack_readers

    parent, task = tmp_path / "canonical", "packed-child"
    parent.mkdir()
    child = headless.prepare_task_drive(parent, task, "empty")
    refs = _checkpoint_chain(child, task, 6)
    originals = {ref["path"]: artifacts.read_actor_source_bytes(child, task, ref) for ref in refs}
    body = {"review_evidence": {"source_refs": [refs[-1]]}}
    write_task_result(child, task, "completed", result="saved answer", artifact_status="ready", **body)
    assert evidence_packer.run_pass(child)["packed"] == 1  # the child's own root holds its checkpoints packed
    damaged = artifacts.task_artifact_dir_path(child, task) / refs[0]["path"]
    assert not damaged.exists()
    damaged.write_bytes(b"a damaged loose name beside the healthy pack member")
    links, link = [], source_retention.os.link

    def recorded(source, destination, *args, **kwargs):
        links.append(Path(source))
        return link(source, destination, *args, **kwargs)

    monkeypatch.setattr(source_retention.os, "link", recorded)
    reads = count_packed_reads(monkeypatch)
    _rewritten, state = observability.promote_child_task_refs(parent, child, task, body)
    assert state["status"] == "complete" and not state["unavailable_refs"] and not state["pending_refs"], state
    checkpoints = {(packs.pack_name(sidecar["task_id"], sidecar["generation"]), member["offset"])
                   for _path, sidecar in packs.list_sidecars(child) for member in sidecar["members"]
                   if member["kind"] == packs.CHECKPOINT}
    assert len(checkpoints) == len(refs) and {reads[member] for member in checkpoints} == {1}, reads
    assert damaged not in links, "a damaged loose file is never linked into the parent"
    assert release_pack_readers(child) == 1  # the child's own pack, held open by these reads (WinError 32 below)
    remove_store_tree(child)
    for ref in refs:
        assert artifacts.read_actor_source_bytes(parent, task, ref) == originals[ref["path"]]


def test_a_loose_child_checkpoint_is_still_linked_once_its_own_bytes_compare_equal(tmp_path):
    parent, task = tmp_path / "canonical", "loose-child"
    parent.mkdir()
    child = headless.prepare_task_drive(parent, task, "empty")
    refs = _checkpoint_chain(child, task, 3)
    _rewritten, state = observability.promote_child_task_refs(
        parent, child, task, {"review_evidence": {"source_refs": [refs[-1]]}})
    assert state["status"] == "complete" and state["promoted_source_handle_count"] == len(refs), state
    for ref in refs:
        source = artifacts.task_artifact_dir_path(child, task) / ref["path"]
        assert (artifacts.task_artifact_dir_path(parent, task) / ref["path"]).stat().st_ino == source.stat().st_ino


def test_historical_manifest_keeps_exact_bytes_without_a_second_packed_read(tmp_path, monkeypatch):
    from ouroboros import evidence_packer, source_retention
    from ouroboros import evidence_packs as packs
    from tests._packed_read_probe import count_packed_reads, release_pack_readers

    parent, task = tmp_path / "canonical", "packed-manifest"
    child = headless.prepare_task_drive(parent, task, "empty")
    call = observability.persist_call(child, task_id=task, call_id="original", call_type="tool_call",
                                     payload={"result": "exact original"})
    # An exact historical version need not have the current writer's JSON spacing.
    raw = Path(call["manifest_ref"]["path"]).read_bytes() + b"\n" * (24 * 1024)
    version = observability.write_blob(child, raw.decode("utf-8"), kind="txt")
    old = {**call["manifest_ref"], "sha256": version["sha256"]}
    observability.update_call_manifest_metadata(child, task_id=task, call_id="original",
                                               metadata={"previous_version": version})
    write_task_result(child, task, "completed", result="answer", artifact_status="ready")
    assert evidence_packer.run_pass(child)["packed"] == 1
    assert not Path(version["path"]).exists()
    member = next((packs.pack_name(sidecar["task_id"], sidecar["generation"]), item["offset"])
                  for _path, sidecar in packs.list_sidecars(child) for item in sidecar["members"]
                  if item["kind"] == "txt" and item["sha256"] == old["sha256"])
    reads = count_packed_reads(monkeypatch)
    body = {"trace_refs": {"tool_call_refs": [{"manifest_ref": old}]}}
    _rewritten, state = observability.promote_child_task_refs(parent, child, task, body)
    assert state["status"] == "complete", state
    # The historical manifest is decoded once; its exact version is then available in the parent's CAS.
    assert reads[member] == 1, reads
    assert release_pack_readers(child) == 1  # the child's own pack, held open by this read (WinError 32 below)
    remove_store_tree(child)
    assert packs.blob_bytes(parent, source_retention.manifest_version_ref(parent, old["sha256"])) == raw
    manifest = observability.read_call_manifest_ref(parent, old, task_id=task)
    assert observability.read_blob_ref(parent, manifest["full_payload_ref"]) == {"result": "exact original"}
