"""Regressions for the dispositioned S5 pack and reader findings."""
from __future__ import annotations

import builtins
import errno
import gzip
import hashlib
import json
import lzma
import os
import pathlib
import sqlite3
from contextlib import closing

import pytest

from ouroboros import evidence_packer as packer
from ouroboros import evidence_packs as packs
from ouroboros import observability as obs
from ouroboros.artifacts import read_actor_source_bytes, store_actor_source_bytes
from tests.test_evidence_packs import _blob, _handle_path, _loose, _result, _task
from tests.test_evidence_packs import root as root


def test_windows_busy_retry_reverifies_disk_before_unlink(root, monkeypatch):
    refs, handle = _task(root, "busy", blobs=1)
    loose_paths = {pathlib.Path(refs[0]["path"]), _handle_path(root, "busy", handle)}
    original_unlink = os.unlink

    def busy(path, *args, **kwargs):
        if pathlib.Path(path) in loose_paths:
            raise PermissionError("simulated Windows sharing violation")
        return original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(os, "unlink", busy)
        assert packer.run_pass(root)["busy"] == 2
    packed = packs.packs_dir(root) / "busy.xz"
    intact = packed.read_bytes()
    packed.write_bytes(b"damaged pack")
    failed = packer.run_pass(root)
    assert failed["faults"] == 1 and failed["unlinked"] == 0
    assert _loose(refs[0]) and _handle_path(root, "busy", handle).exists()

    packed.write_bytes(intact)
    verified = []
    original_verify = packs.verify_pack

    def verify(path, sidecar):
        original_verify(path, sidecar)
        verified.append(path)

    monkeypatch.setattr(packs, "verify_pack", verify)
    passed = packer.run_pass(root)
    assert passed["faults"] == 0 and passed["unlinked"] == 2
    assert verified == [packed]
    assert not any(path.exists() for path in loose_paths)


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


@pytest.mark.parametrize("placement", ["valid", "missing", "corrupt"])
def test_historical_focus_distinguishes_missing_and_corrupt_handles(root, placement):
    from ouroboros.task_finalization import focus_source_projection

    _, ref = _task(root, "focus", blobs=0)
    loose = _handle_path(root, "focus", ref)
    if placement == "missing":
        loose.unlink()
    elif placement == "corrupt":
        loose.write_bytes(b"damaged")
    result = focus_source_projection(root, "focus", {}, sha256=ref["sha256"])
    if placement == "valid":
        assert result["historical"] and result["complete_sha256"] == ref["sha256"]
    else:
        assert result["reason"] == ("source_unavailable" if placement == "missing" else "source_identity_mismatch")


def test_host_acceptance_distinguishes_verified_and_corrupt_retained_source(root):
    from ouroboros.task_finalization import host_acceptance_source

    child = root / "task_drives" / "author"
    _result(root, "author", child_drive_root=str(child))
    raw = json.dumps({"authority": "host_root", "request": {
        "surface": "task_acceptance", "task_id": "author"}}).encode()
    ref = store_actor_source_bytes(child, "author", category="context_checkpoints",
                                   source_id="acceptance", data=raw, extension="json")
    stored, actual = host_acceptance_source(root, "author", ref["sha256"])
    assert stored["sha256"] == ref["sha256"] and actual == raw
    _handle_path(child, "author", ref).write_bytes(b"damaged")
    with pytest.raises(packs.EvidenceCorrupt, match="verification"):
        host_acceptance_source(root, "author", ref["sha256"])


def test_kept_shared_blob_rests_until_the_last_live_sharer_settles(root):
    raw = b'{"shared":true}'
    refs, _ = _task(root, "settled", blobs=1, payloads=[raw])
    _task(root, "live", status="running", blobs=1, payloads=[raw])
    first = packer.run_pass(root)
    assert (first["kept"], first["packed"], first["swept"]) == (1, 1, 2) and _loose(refs[0])
    second = packer.run_pass(root)  # the pass recorded the directories as its own unlinks left them
    assert (second["examined"], second["kept"], second["swept"]) == (0, 0, 0)
    for _ in range(2):  # then no treadmill: the settled task is not re-examined, re-verified or re-kept
        assert packer.changed_tasks(root) == []
        rest = packer.run_pass(root)
        assert (rest["candidates"], rest["examined"], rest["kept"], rest["verified_bytes"]) == (0, 0, 0, 0)
        assert _loose(refs[0])
    _result(root, "live")  # the sharer's own stamp moves: ITS pass packs and unlinks the shared blob
    assert packer.changed_tasks(root) == ["live"]
    assert packer.run_pass(root)["faults"] == 0
    assert not _loose(refs[0]) and packs.blob_bytes(root, refs[0]) == raw
    packer.run_pass(root)
    assert packer.changed_tasks(root) == []


def test_identical_checkpoint_and_blob_bytes_keep_both_index_kinds(root):
    raw = b'{"same": true}'
    refs, handle = _task(root, "same", blobs=1, payloads=[raw], handle=raw)
    other = _blob(root, raw, kind="bin")
    (root / "observability" / "calls" / "same" / "other.json").write_text(json.dumps(other), encoding="utf-8")
    assert packer.run_pass(root)["faults"] == 0
    assert not _loose(refs[0]) and not _handle_path(root, "same", handle).exists()
    assert read_actor_source_bytes(root, "same", handle) == raw
    assert packs.blob_bytes(root, refs[0]) == packs.blob_bytes(root, other) == raw
    with packs.open_index(root) as conn:
        assert {row[0] for row in conn.execute("SELECT kind FROM members")} == {"json", "bin", "checkpoint"}


def test_old_index_schema_rebuilds_without_rewriting_sidecars(root):
    refs, handle = _task(root, "indexed", blobs=1)
    packer.run_pass(root)
    directory = packs.packs_dir(root)
    sidecar = (directory / "indexed.members.json").read_bytes()
    with closing(sqlite3.connect(directory / packs.INDEX_NAME)) as conn, conn:
        conn.execute("DROP TABLE members")
        conn.execute("CREATE TABLE members (sha256 TEXT, task_id TEXT, generation INTEGER, offset INTEGER, "
                     "size INTEGER, kind TEXT, PRIMARY KEY (sha256, task_id, generation))")
        conn.execute("DELETE FROM meta WHERE key='schema_version'")
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('format_version', '1')")
    packs._VALIDATED.clear()
    with packs.open_index(root) as conn:
        key = {row["name"] for row in conn.execute("PRAGMA table_info(members)") if row["pk"]}
        assert key == {"sha256", "kind", "task_id", "generation"}
        assert conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == str(packs.INDEX_SCHEMA_VERSION)
    assert (directory / "indexed.members.json").read_bytes() == sidecar
    assert json.loads(sidecar)["format_version"] == 1
    assert read_actor_source_bytes(root, "indexed", handle).startswith(b"# focus")
    assert packs.blob_bytes(root, refs[0]).startswith(b'{"message"')


def test_blob_resolver_tries_all_local_placements_then_retained_root(root):
    raw = b'{"shared": "exact bytes"}'
    refs, _ = _task(root, "a", blobs=1, payloads=[raw])
    _task(root, "b", blobs=1, payloads=[raw])
    packer.run_pass(root)
    (packs.packs_dir(root) / "a.xz").write_bytes(b"bad a")
    assert packs.blob_bytes(root, refs[0]) == raw
    child = root / "task_drives" / "a"
    child.mkdir(parents=True)
    retained, _ = _task(child, "a", blobs=1, payloads=[raw])
    packer.run_pass(child)
    (packs.packs_dir(root) / "b.xz").write_bytes(b"bad b")
    assert packs.blob_bytes(root, retained[0]) == raw
    (packs.packs_dir(child) / "a.xz").write_bytes(b"bad child")
    with pytest.raises(packs.EvidenceCorrupt, match="evidence_corrupt"):
        packs.blob_bytes(root, retained[0])


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


@pytest.mark.parametrize("reader", ["handle", "actor"])
def test_corrupt_loose_handle_falls_back_to_verified_pack(root, reader):
    raw = b"the retained checkpoint"
    _, handle = _task(root, "handle", blobs=1, handle=raw)
    packer.run_pass(root)
    loose = _handle_path(root, "handle", handle)
    loose.write_bytes(b"damaged loose")

    def read():
        return (packs.handle_bytes(root, "handle", handle["path"]) if reader == "handle"
                else read_actor_source_bytes(root, "handle", handle))

    assert read() == raw
    (packs.packs_dir(root) / "handle.xz").write_bytes(b"damaged pack")
    with pytest.raises(packs.EvidenceCorrupt, match="evidence_corrupt"):
        read()
    loose.write_bytes(raw)
    assert read() == raw


@pytest.mark.parametrize("change", ["resume", "new_reference"])
def test_unlink_rereads_facts_and_references_after_compression(root, monkeypatch, change):
    refs, handle = _task(root, "packing", blobs=1)
    _task(root, "other", status="running", blobs=1)
    write = packs.write_pack

    def publish(*args, **kwargs):
        result = write(*args, **kwargs)
        if change == "resume":
            _result(root, "packing", "running")
        else:
            obs.write_call_manifest(root, task_id="other", call_id="new", manifest={"ref": refs[0]})
        return result

    with monkeypatch.context() as patch:
        patch.setattr(packs, "write_pack", publish)
        report = packer.run_pass(root)
    assert report["faults"] == 0 and _loose(refs[0])
    if change == "resume":
        assert _handle_path(root, "packing", handle).exists()
        assert "packing" in packer.changed_tasks(root)
    _result(root, "packing")
    _result(root, "other")
    assert packer.run_pass(root)["faults"] == 0
    assert not _loose(refs[0]) and not _handle_path(root, "packing", handle).exists()


def test_compression_is_outside_result_and_publication_locks(root, monkeypatch):
    from ouroboros import task_custody, store

    _task(root, "unlocked", blobs=1)
    write = packs.write_pack

    def compress(*args, **kwargs):
        with task_custody.task_custody_lock(root, "unlocked", timeout_sec=0) as acquired:
            assert acquired
            with store.open_store(root, write=True, wait_sec=0) as txn:
                assert txn.load("unlocked")
                with store.name_lock(root, packs.PUBLICATION_LOCK_NAME, timeout_sec=0):
                    pass
        return write(*args, **kwargs)

    monkeypatch.setattr(packs, "write_pack", compress)
    assert packer.run_pass(root)["faults"] == 0


def test_stop_latches_and_signals_without_waiting_for_the_job_facts_read(root, monkeypatch):
    import threading
    from tests.test_evidence_packs import REPO, _FakeChild
    from ouroboros import maintenance, process_custody

    entered, release, stopped = threading.Event(), threading.Event(), threading.Event()
    unit, child = maintenance.MaintenanceChild(), _FakeChild(b"")
    unit._process = child

    def facts(_root):
        entered.set()
        assert release.wait(5)
        return ["evidence_packing"]

    monkeypatch.setattr(maintenance, "due", facts)
    spawns = []
    monkeypatch.setattr(process_custody, "spawn_supervised", lambda *a, **k: spawns.append(a) or _FakeChild(b""))
    tick = threading.Thread(target=unit.tick, args=(root, REPO))
    stop = threading.Thread(target=lambda: (unit.stop(), stopped.set()))
    tick.start()
    try:
        assert entered.wait(2)
        stop.start()
        assert stopped.wait(0.3), "Stop waited for the mutex held by the facts read"
        assert child.killed.is_set() and unit._stopped.is_set()
    finally:
        release.set()
        tick.join(3)
        stop.join(3)
        if unit._thread is not None:
            unit._thread.join(3)
    assert not tick.is_alive() and not stop.is_alive()
    assert spawns == [], "spawned after Stop"


def test_restore_and_preflight_cover_all_descendants_but_not_symlink_drives(root):
    outer = root / "task_drives/outer"
    inner = outer / "state/headless_tasks/inner/data"
    raw = b"nested retained checkpoint"
    _, handle = _task(inner, "inner", blobs=1, handle=raw)
    assert packer.run_pass(inner)["faults"] == 0
    loose = _handle_path(inner, "inner", handle)
    assert not loose.exists()
    outside = root.parent / "outside"
    _, outside_handle = _task(outside, "outside", blobs=1, handle=b"outside bytes")
    packer.run_pass(outside)
    (root / "task_drives/linked").symlink_to(outside, target_is_directory=True)
    assert packs.pending_restore_bytes(root) == len(raw)
    assert packs.restore_checkpoint_handles(root)["handles"] == 1
    assert loose.read_bytes() == raw
    assert not _handle_path(outside, "outside", outside_handle).exists()
    assert packs.pending_restore_bytes(root) == 0
    assert packs.restore_checkpoint_handles(root)["handles"] == 0


def test_first_backfill_by_size_then_new_settlements_before_backlog(root):
    _task(root, "large", blobs=1, payloads=[b"x" * 10000], updated_at="2026-01-01T00:00:00Z")
    _task(root, "small", blobs=1, payloads=[b"small"], updated_at="2026-01-02T00:00:00Z")
    assert [task for task, _ in packer.ordered_candidates(root, ["small", "large"])] == ["large", "small"]
    _task(root, "published", blobs=1)
    packs.pack_task(root, "published", keep_shas=set())
    _task(root, "new", blobs=1, payloads=[b"new"])
    assert [task for task, _ in packer.ordered_candidates(root, ["large", "small", "new"])] == ["new", "large", "small"]


def test_manifest_members_follow_first_appearance_with_name_ties(root):
    _task(root, "ordered", blobs=0)
    directory = root / "observability/calls/ordered"
    first, second, third = [_blob(root, value) for value in (b"first", b"second", b"third")]
    for name, ref, created in [("z", first, "2026-01-01T00:00:01Z"), ("b", third, "2026-01-01T00:00:02Z"),
                               ("a", second, "2026-01-01T00:00:02Z")]:
        path = directory / (name + ".json")
        path.write_text(json.dumps({"created_at": created, "ref": ref}), encoding="utf-8")
    expected = [ref["sha256"] for ref in (first, second, third)]
    assert [sha for _, sha, _ in packs.manifest_blob_refs(root, "ordered")] == expected
    # A later atomic rewrite preserves the original created_at order.
    obs.write_call_manifest(root, task_id="ordered", call_id="z", manifest={"created_at": "2026-01-01T00:00:01Z", "ref": first})
    assert [sha for _, sha, _ in packs.manifest_blob_refs(root, "ordered")] == expected


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


def test_archive_fsync_uses_a_writable_handle(root, monkeypatch):
    source = root / "archive/usage_ledger/segment_ep1.jsonl"
    source.parent.mkdir(parents=True)
    raw = b'{"attempt_id":"example"}\r\n'
    source.write_bytes(raw)
    opened, synced = {}, []
    original_fsync = os.fsync

    def tracked_open(*args, **kwargs):
        handle = builtins.open(*args, **kwargs)
        opened[handle.fileno()] = handle
        return handle

    def windows_fsync(fd):
        handle = opened.get(fd)
        if handle is not None and not handle.closed:
            # Windows FlushFileBuffers requires GENERIC_WRITE; POSIX accepts rb.
            if not handle.writable():
                raise OSError(errno.EBADF, "fsync requires a writable handle")
            synced.append(pathlib.Path(handle.name))
        return original_fsync(fd)

    monkeypatch.setattr(packs, "open", tracked_open, raising=False)
    monkeypatch.setattr(os, "fsync", windows_fsync)
    assert packs.archive_usage_segments(root)["archived"] == 1
    assert synced and not source.exists()
    assert lzma.decompress(source.with_suffix(".jsonl.xz").read_bytes()) == raw
    meta = json.loads(source.with_suffix(".jsonl.xz.json").read_text(encoding="utf-8"))
    assert (meta["size"], meta["sha256"]) == (len(raw), hashlib.sha256(raw).hexdigest())


def test_exporting_phase_keeps_loose_evidence_until_published_again(root):
    from ouroboros import store, store_import

    refs, handle = _task(root, "phase", blobs=1)
    store_import.ensure_current_store(root)
    with store.open_store(root, write=True) as txn:
        txn.set_meta("phase", store.EXPORTING)
    report = packer.run_pass(root)
    assert report["faults"] == 0 and report["unlinked"] == 0
    assert _loose(refs[0]) and _handle_path(root, "phase", handle).exists()
    with store.open_store(root, write=True, service=True) as txn:  # the seam's own mode: the gate refuses a public write here
        txn.set_meta("phase", store.PUBLISHED)
    assert packer.run_pass(root)["unlinked"] == 2


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


@pytest.mark.parametrize("leader", [True, False])
def test_maintenance_child_is_custody_blocker_until_exit_is_confirmed(root, monkeypatch, leader):
    from ouroboros import process_custody

    entry = {"pid": 4242, "pgid": 4242, "purpose": "maintenance", "scope": "session"}
    alive, signals = [True], []
    monkeypatch.setattr(process_custody, "_read_ledger_records", lambda *a, **k: (True, [entry], b""))
    monkeypatch.setattr(process_custody, "_fingerprint_matches", lambda e: alive[0] and leader)
    monkeypatch.setattr(process_custody, "process_group_has_live_members", lambda pgid: alive[0])
    monkeypatch.setattr(process_custody, "_rewrite_ledger", lambda *a, **k: None)
    monkeypatch.setattr(process_custody, "kill_process_group_id", lambda pid: signals.append(pid))
    assert process_custody.quiesce_custodied_services(root, timeout_sec=0) == (False, ["custody_service:4242"])
    assert signals == [4242]
    alive[0] = False
    assert process_custody.quiesce_custodied_services(root, timeout_sec=0) == (True, [])


@pytest.mark.parametrize("door", ["run_switch_seam", "resume_pending_switch"])
def test_switch_doors_restore_handles_by_default_before_owner(root, monkeypatch, door):
    from ouroboros import store_switch

    raw = b"checkpoint before the old body starts"
    _, handle = _task(root, "restore", blobs=1, handle=raw)
    packer.run_pass(root)
    target = _handle_path(root, "restore", handle)
    assert not target.exists()
    monkeypatch.setattr(store_switch, "switch_needs", lambda *a: {"handles": True, "money": False, "facts": False})
    monkeypatch.setattr(store_switch, "rev_parse", lambda *a: "example-target")
    store_switch.record_switch_intent(root, target_sha="example-target", target_ref="example", reason="test", door="ui_rollback")

    def owner(_):
        assert target.read_bytes() == raw
        return True, "example switched"

    assert getattr(store_switch, door)(root, repo_dir=root, owner=owner)["status"] == "switched"
