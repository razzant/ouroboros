"""Evidence packs (S5): format, index cache, resolver chain, packer steps, A+ restore, unpacker, archive.

Fixture roots are real-shaped: stamped task results, call manifests with gzip CAS refs, context
checkpoint handles stored through ``store_actor_source_bytes``.
"""
from __future__ import annotations

import contextlib
import gc
import gzip
import hashlib
import json
import lzma
import os
import pathlib
import sqlite3
import subprocess
import sys
import threading
import traceback
import weakref

import pytest

from ouroboros import evidence_packer as packer
from ouroboros import evidence_packs as packs
from ouroboros import observability as obs
from ouroboros.artifacts import read_actor_source_bytes, store_actor_source_bytes, task_artifact_dir_path

REPO = pathlib.Path(__file__).resolve().parents[1]
CHECKPOINTS = "source_handles/context_checkpoints"


def _blob(root, raw, kind="json"):
    sha = hashlib.sha256(raw).hexdigest()
    path = root / "observability" / "blobs" / f"{sha}.{kind}.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb", compresslevel=1) as fh:
        fh.write(raw)
    return {"kind": kind, "encoding": "gzip", "sha256": sha, "size": len(raw), "path": str(path)}


def _result(root, task_id, status="completed", **fields):
    from ouroboros import store
    from ouroboros.utils import utc_now_iso

    row = {"task_id": task_id, "status": status, "_schema_version": 1, "updated_at": utc_now_iso(), **fields}
    with store.open_store(root, write=True) as txn:
        txn.store_task(row)


def _task(root, task_id, *, status="completed", blobs=3, payloads=None, handle=b"# focus\n" + b"y" * 3000, **fields):
    """A settled-shaped task: manifests referencing gzip blobs plus one context-checkpoint handle."""
    calls = root / "observability" / "calls" / task_id
    calls.mkdir(parents=True, exist_ok=True)
    refs = []
    for i in range(blobs):
        raw = payloads[i] if payloads else json.dumps({"message": {"content": f"answer {task_id} {i} " + "x" * 4000 * (i + 1)}}).encode()
        ref = _blob(root, raw)
        (calls / f"llm_{i}_response.json").write_text(json.dumps({"call_id": f"llm_{i}", "full_payload_ref": ref}), encoding="utf-8")
        refs.append(ref)
    handle_ref = store_actor_source_bytes(root, task_id, category="context_checkpoints",
                                          source_id=f"focus_source_{task_id}", data=handle, extension="md")
    _result(root, task_id, status, **fields)
    return refs, handle_ref


def _run(root):
    return packer.run_pass(root)


def _loose(ref):
    return pathlib.Path(ref["path"]).exists()


def _handle_path(root, task_id, handle_ref):
    return task_artifact_dir_path(root, task_id, create=False) / handle_ref["path"]


@pytest.fixture
def root(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    packs._READERS.clear()
    yield data
    packs._READERS.clear()


# ---- format, sidecar, vetoes ----------------------------------------------------

def test_pack_format_sidecar_prefix_and_shared_sha_duplicated(root):
    refs_a, handle_a = _task(root, "task-a")
    refs_b, handle_b = _task(root, "task-b", blobs=1, payloads=[gzip.decompress(pathlib.Path(refs_a[0]["path"]).read_bytes())])
    assert refs_b[0]["sha256"] == refs_a[0]["sha256"]  # the same bytes referenced by two settled tasks
    report = _run(root)
    assert (report["packed"], report["faults"]) == (2, 0)
    pack_a = root / "observability" / "packs" / "task-a.xz"
    sidecar = json.loads((root / "observability" / "packs" / "task-a.members.json").read_text(encoding="utf-8"))
    assert pack_a.read_bytes()[:6] == b"\xfd7zXZ\x00"  # one xz stream, stdlib lzma decodes it whole (veto: codec)
    members = sidecar["members"]
    whole = lzma.decompress(pack_a.read_bytes())
    assert len(whole) == sidecar["raw_total"] and members[0]["kind"] == "checkpoint" and members[0]["offset"] == 0
    assert [m["offset"] for m in members] == sorted(m["offset"] for m in members)
    for member in members:  # every member complete inside its own pack; no independent frames anywhere
        raw = whole[member["offset"]:member["offset"] + member["size"]]
        assert hashlib.sha256(raw).hexdigest() == member["sha256"]
    assert {sidecar[k] for k in ("format_version", "codec", "preset")} == {1, "xz"}
    assert sidecar["dict_size"] == packs.dict_size_for(sidecar["raw_total"]) and sidecar["dict_size"] % 2 == 0
    assert sidecar["pack_sha256"] == hashlib.sha256(pack_a.read_bytes()).hexdigest()
    assert sidecar["members"][0]["path"] == handle_a["path"]
    shared = refs_a[0]["sha256"]
    sidecar_b = json.loads((root / "observability" / "packs" / "task-b.members.json").read_text(encoding="utf-8"))
    assert shared in {m["sha256"] for m in sidecar["members"]} and shared in {m["sha256"] for m in sidecar_b["members"]}
    assert not any(_loose(ref) for ref in refs_a + refs_b) and not _handle_path(root, "task-a", handle_a).exists()
    from ouroboros import store
    with store.open_store(root, write=False) as txn:
        tables = {row[0] for row in txn.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert not tables.intersection({"packs", "members", "handles", "swept"})  # separate cache, never in current.sqlite
    assert (root / "observability" / "packs" / "index.sqlite").exists()
    assert len(set(sidecar["members"][0].keys()) & {"sha256", "kind", "size", "offset", "path"}) == 5


def test_dict_size_adapts_between_bounds():
    assert packs.dict_size_for(0) == 4096 and packs.dict_size_for(5000) == 8192
    assert packs.dict_size_for(10 ** 12) == 64 << 20


# ---- the resolver chain ----------------------------------------------------------

def test_readers_resolve_loose_then_pack_and_blob_shared_with_active_task_stays_loose(root):
    refs_a, handle_a = _task(root, "task-a")
    refs_c, _ = _task(root, "task-c", status="running", blobs=1)
    calls_c = root / "observability" / "calls" / "task-c"
    (calls_c / "llm_9_response.json").write_text(json.dumps({"call_id": "llm_9", "full_payload_ref": refs_a[0]}), encoding="utf-8")
    report = _run(root)
    assert report["packed"] == 1 and report["kept"] == 1
    assert _loose(refs_a[0]) and not _loose(refs_a[1])  # a blob an active task still references stays loose
    assert _loose(refs_c[0])  # an active task is never packed
    for ref in refs_a:
        assert obs.read_blob_ref(root, ref)["message"]["content"].startswith("answer task-a")
    packed_raw = packs.blob_bytes(root, refs_a[2], expected_kind="json")
    assert hashlib.sha256(packed_raw).hexdigest() == refs_a[2]["sha256"]
    with pytest.raises(ValueError, match="unexpected kind"):
        packs.blob_bytes(root, refs_a[2], expected_kind="bin")
    raw = read_actor_source_bytes(root, "task-a", handle_a)
    assert raw.startswith(b"# focus") and len(raw) == handle_a["size"]
    assert packs.handle_info(root, "task-a", handle_a["path"]) == {"size": handle_a["size"], "sha256": handle_a["sha256"], "loose": False}
    from ouroboros.review_ledger import source_ref_resolvable
    assert source_ref_resolvable(root, "task-a", {"status": "retained", "ref": handle_a})
    assert not source_ref_resolvable(root, "task-a", {"status": "retained", "ref": {**handle_a, "size": handle_a["size"] + 1}})
    from ouroboros.model_send_seal import _read_blob_bytes
    payload, intact = _read_blob_bytes({"full_payload_ref": refs_a[1]})
    assert intact and json.loads(payload)["message"]["content"].startswith("answer task-a 1")
    assert _read_blob_bytes({"full_payload_ref": {**refs_a[1], "sha256": "0" * 64}}) == (None, False)
    txt = _blob(root, b"manifest version bytes", kind="txt")
    (root / "observability" / "calls" / "task-a" / "llm_8_response.json").write_text(
        json.dumps({"call_id": "llm_8", "manifest_version_ref": txt}), encoding="utf-8")
    _run(root)
    assert not _loose(txt)
    from ouroboros.source_retention import read_manifest_version
    assert read_manifest_version(root, txt["sha256"]) == b"manifest version bytes"
    # directory listing and read_file see the packed handle beside loose ones
    from ouroboros.tools.core_file_tools import _list_dir, _read_source_text
    store_dir = task_artifact_dir_path(root, "task-a", create=False) / CHECKPOINTS
    listed = _list_dir(task_artifact_dir_path(root, "task-a", create=False), CHECKPOINTS)
    assert listed == [str(pathlib.PurePosixPath(CHECKPOINTS) / pathlib.PurePosixPath(handle_a['path']).name)]  # slashes on any OS
    extent = {}
    assert _read_source_text(store_dir / pathlib.PurePosixPath(handle_a["path"]).name, extent).startswith("# focus")
    assert extent["source_revision"] == handle_a["sha256"] and _handle_path(root, "task-a", handle_a).exists()


def test_salvage_and_task_finalization_selectors_read_packed_evidence(root):
    refs, handle = _task(root, "task-a", blobs=2)
    _run(root)
    assert not _loose(refs[1])
    from ouroboros.observability import latest_llm_response_text
    assert latest_llm_response_text(root, "task-a").startswith("answer task-a")
    from ouroboros.task_finalization import focus_source_projection
    projection = focus_source_projection(root, "task-a", {"focus": None}, sha256=handle["sha256"])
    assert projection.get("status") != "unavailable", projection
    assert projection["complete_sha256"] == handle["sha256"] and projection["complete_chars"] == handle["size"]


def test_store_actor_source_bytes_is_write_once_against_a_packed_handle(root, monkeypatch):
    _, handle = _task(root, "task-a", blobs=1)
    _run(root)
    assert not _handle_path(root, "task-a", handle).exists()
    from ouroboros import artifacts
    monkeypatch.setattr(artifacts, "fence_publication", lambda: pytest.fail("a packed handle was rewritten"))
    again = store_actor_source_bytes(root, "task-a", category="context_checkpoints", source_id="focus_source_task-a",
                                     data=b"# focus\n" + b"y" * 3000, extension="md")
    assert again["sha256"] == handle["sha256"] and not _handle_path(root, "task-a", handle).exists()


def test_retained_child_root_packs_serve_predecessor_lookups(root):
    child = root / "task_drives" / "task-a"
    child.mkdir(parents=True)
    refs, handle = _task(child, "task-a", blobs=1)
    _result(root, "task-a", child_drive_root=str(child))
    _run(child)
    assert not _loose(refs[0]) and not _handle_path(child, "task-a", handle).exists()
    assert obs.read_blob_ref(root, refs[0])["message"]["content"].startswith("answer")  # the ref names the child root
    assert read_actor_source_bytes(root, "task-a", handle).startswith(b"# focus")  # the retained walk reaches the pack
    from ouroboros.source_retention import retained_actor_source_read_target
    target = retained_actor_source_read_target(root, "task-a", handle["path"])
    assert target is not None and target.read_bytes().startswith(b"# focus")


def test_admission_sees_every_physical_blob_source_and_its_refusal_is_never_a_miss(root):
    child = root / "task_drives" / "task-a"
    child.mkdir(parents=True)
    refs, _ = _task(child, "task-a", blobs=1)
    _result(root, "task-a", child_drive_root=str(child))
    _run(child)
    seen = []
    assert obs.read_blob_ref(root, refs[0], admit=seen.append)["message"]["content"].startswith("answer")
    loose = pathlib.Path("observability", "blobs", f"{refs[0]['sha256']}.json.gz")
    assert seen == [root / loose, child / loose, child / "observability" / "packs" / "task-a.xz"]
    for refused in seen:  # raised outside the OSError handlers: never absence, corruption or a later candidate

        def admit(path, refused=refused):
            if path == refused:
                raise PermissionError(f"refused {path}")

        with pytest.raises(PermissionError, match="refused"):
            obs.read_blob_ref(root, refs[0], admit=admit)
    assert obs.read_blob_ref(root, refs[0])["message"]["content"].startswith("answer")  # no callback: unchanged


@pytest.mark.parametrize("case", ["missing", "packed", "retained", "damaged"])
def test_a_kept_miss_or_damage_never_keeps_the_read_alive(root, case):
    """A miss or a damaged loose copy is kept while later candidates are tried, and its traceback holds
    the reader's frame; a frame still holding it is a cycle that keeps the caller's ``admit`` callback
    (and, through the callers' frames, their payloads) until a GC pass. With the collector off, every
    exit frees the callback with its last reference (the final raise; a pack or a retained root after a
    miss; a pack after a damaged copy), and the raise is still the reader's own error with its traceback."""
    drive = root / "task_drives" / "task-a" if case == "retained" else root
    drive.mkdir(parents=True, exist_ok=True)
    refs, _ = _task(drive, "task-a", blobs=1)
    raw = gzip.decompress(pathlib.Path(refs[0]["path"]).read_bytes())
    loose = root / "observability" / "blobs" / f"{refs[0]['sha256']}.json.gz"
    if case == "missing":
        loose.unlink()  # nothing loose and no pack anywhere
        expected = [loose]
    elif case == "retained":
        _result(root, "task-a", child_drive_root=str(drive))  # this root misses; its retained child holds it
        expected = [loose, pathlib.Path(refs[0]["path"])]
    else:
        _run(root)  # the pack holds the bytes, the loose copy is gone
        if case == "damaged":
            loose.write_bytes(b"damaged")  # not gzip: damage, never a miss
        expected = [loose, root / "observability" / "packs" / "task-a.xz"]
    admitted = []

    def admit(path):
        admitted.append(path)

    watched, enabled = weakref.ref(admit), gc.isenabled()
    gc.disable()  # only reference counting frees now, so a cycle stays alive and visible
    try:
        try:
            read = packs.blob_bytes(root, refs[0], admit=admit)
        except FileNotFoundError as error:  # facts only: the error must not outlive this block
            read = (os.fspath(error.filename), {entry.name for entry in traceback.extract_tb(error.__traceback__)})
        del admit
        freed = watched() is None
    finally:
        if enabled:
            gc.enable()
    assert admitted == expected
    if case == "missing":  # the first miss itself, its traceback down to the reader intact
        assert read[0] == os.fspath(loose) and "read_loose_gzip" in read[1]
    else:
        assert read == raw
    assert freed, "a kept error held the reader's frame and with it the caller's callback"


# ---- damage, index cache, kill matrix --------------------------------------------

def test_damaged_pack_is_explicit_evidence_corrupt_and_neighbours_unaffected(root):
    refs_a, handle_a = _task(root, "task-a")
    refs_b, _ = _task(root, "task-b", blobs=1)
    _run(root)
    pack = root / "observability" / "packs" / "task-a.xz"
    data = bytearray(pack.read_bytes())
    data[len(data) // 2] ^= 0xFF
    pack.write_bytes(bytes(data))
    packs._READERS.clear()
    with pytest.raises(packs.EvidenceCorrupt, match="evidence_corrupt"):
        obs.read_blob_ref(root, refs_a[2])
    with pytest.raises(ValueError, match="evidence_corrupt"):
        read_actor_source_bytes(root, "task-a", handle_a)
    assert obs.read_blob_ref(root, refs_b[0])["message"]["content"].startswith("answer task-b")
    pack.unlink()  # a missing pack is absence, not corruption
    packs._READERS.clear()
    with pytest.raises(FileNotFoundError):
        obs.read_blob_ref(root, refs_a[2])


def test_index_is_a_cache_rebuilt_from_sidecars(root):
    refs, handle = _task(root, "task-a", blobs=2)
    _run(root)
    index = root / "observability" / "packs" / "index.sqlite"
    index.unlink()
    assert obs.read_blob_ref(root, refs[1])["message"]["content"].startswith("answer")
    assert index.exists()
    with contextlib.closing(sqlite3.connect(index)) as conn:
        assert conn.execute("PRAGMA application_id").fetchone()[0] in packs._APPLICATION_ID.values()
        assert conn.execute("SELECT count(*) FROM members").fetchone()[0] == 3
        assert conn.execute("SELECT count(*) FROM handles").fetchone()[0] == 1
    index.write_bytes(b"not a database at all")
    packs._VALIDATED.clear()
    # A handle is addressed by its task: its own sidecar answers, and the damaged cache is
    # left to the readers that need every pack (a blob lookup here, the packer's pass).
    assert read_actor_source_bytes(root, "task-a", handle).startswith(b"# focus")
    assert index.read_bytes() == b"not a database at all"
    assert obs.read_blob_ref(root, refs[1])["message"]["content"].startswith("answer")
    with contextlib.closing(sqlite3.connect(index)) as conn:
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    # a stale index schema is rebuilt without changing the sidecar format
    with contextlib.closing(sqlite3.connect(index)) as conn, conn:
        conn.execute("UPDATE meta SET value='0' WHERE key='schema_version'")
    packs._VALIDATED.clear()
    with packs.open_index(root) as conn:
        assert conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == str(packs.INDEX_SCHEMA_VERSION)


def test_no_unlink_before_on_disk_reverify(root, monkeypatch):
    refs, handle = _task(root, "task-a", blobs=2)
    order = []
    real_verify, real_unlink = packs.verify_pack, os.unlink

    def verify(pack_path, sidecar):
        order.append("verify")
        real_verify(pack_path, sidecar)

    def unlink(path, *args, **kwargs):
        if pathlib.Path(path).name.endswith(".gz") or "context_checkpoints" in str(path):
            order.append("unlink")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(packs, "verify_pack", verify)
    monkeypatch.setattr(os, "unlink", unlink)
    _run(root)
    assert order and order[0] == "verify" and order.count("unlink") == 3
    _run(root)  # acknowledge our handle-directory deletion, without stamping an unseen revision
    # and a failing verification keeps every loose file
    refs2, handle2 = _task(root, "task-b", blobs=1)
    monkeypatch.setattr(packs, "verify_pack", lambda *a: (_ for _ in ()).throw(packs.EvidenceCorrupt("evidence_corrupt: test")))
    report = _run(root)
    assert report["faults"] == 1 and _loose(refs2[0]) and _handle_path(root, "task-b", handle2).exists()


@pytest.mark.parametrize("step", ["pack_published", "sidecar_published", "indexed"])
def test_kill_at_each_packer_step_leaves_readable_state_and_rerun_completes(root, monkeypatch, step):
    refs, handle = _task(root, "task-a", blobs=2)
    real_replace, real_upsert = packs.replace_atomic, packs.index_upsert
    calls = {"replace": 0}

    def replace(src, dst):
        real_replace(src, dst)
        calls["replace"] += 1
        if step == "pack_published" and calls["replace"] == 1:
            raise KeyboardInterrupt("kill -9 after the pack rename")
        if step == "sidecar_published" and calls["replace"] == 2:
            raise KeyboardInterrupt("kill -9 after the sidecar rename")

    def upsert(root_, pack_path, sidecar):
        real_upsert(root_, pack_path, sidecar)
        raise KeyboardInterrupt("kill -9 after the index upsert")

    with monkeypatch.context() as patch:
        patch.setattr(packs, "replace_atomic", replace)
        if step == "indexed":
            patch.setattr(packs, "index_upsert", upsert)
        with pytest.raises(KeyboardInterrupt):
            packs.pack_task(root, "task-a", keep_shas=set())
    packs._READERS.clear()
    # loose files are still there (nothing unlinked before the verified index), every reader works
    assert all(_loose(ref) for ref in refs) and _handle_path(root, "task-a", handle).exists()
    assert obs.read_blob_ref(root, refs[1])["message"]["content"].startswith("answer")
    assert read_actor_source_bytes(root, "task-a", handle).startswith(b"# focus")
    stray = [p.name for p in (root / "observability" / "packs").iterdir() if p.name.startswith(".")]
    assert not stray or step != "indexed"
    report = _run(root)  # ``packed`` counts a new generation; a published one is only re-verified
    assert (report["examined"], report["packed"], report["faults"]) == (1, int(step == "pack_published"), 0)
    names = sorted(p.name for p in (root / "observability" / "packs").iterdir() if p.name.startswith("task-a"))
    assert names == (["task-a.2.members.json", "task-a.2.xz", "task-a.xz"] if step == "pack_published"
                     else ["task-a.members.json", "task-a.xz"])  # an orphan's generation is never overwritten
    assert not any(_loose(ref) for ref in refs)
    assert obs.read_blob_ref(root, refs[0])["message"]["content"].startswith("answer")


def test_second_generation_packs_only_new_members(root):
    refs, handle = _task(root, "task-a", blobs=1)
    _run(root)
    late = _blob(root, json.dumps({"message": {"content": "late answer"}}).encode())
    (root / "observability" / "calls" / "task-a" / "llm_7_response.json").write_text(
        json.dumps({"call_id": "llm_7", "full_payload_ref": late}), encoding="utf-8")
    report = _run(root)
    assert report["packed"] == 1
    sidecar = json.loads((root / "observability" / "packs" / "task-a.2.members.json").read_text(encoding="utf-8"))
    assert [m["sha256"] for m in sidecar["members"]] == [late["sha256"]] and sidecar["generation"] == 2
    assert not _loose(late) and obs.read_blob_ref(root, late)["message"]["content"] == "late answer"
    assert obs.read_blob_ref(root, refs[0])["message"]["content"].startswith("answer")


# ---- A+ restore and the unpacker ---------------------------------------------------

def test_restore_checkpoint_handles_decodes_only_the_prefix(root, monkeypatch):
    big = [os.urandom(3 << 20)]  # incompressible: the pack is as large as its blob
    refs, handle = _task(root, "task-a", blobs=1, payloads=big)
    child = root / "task_drives" / "task-a"
    child.mkdir(parents=True)
    _, child_handle = _task(child, "task-a", blobs=1, handle=b"# child focus\n")
    _run(root)
    _run(child)
    assert packs.pending_restore_bytes(root) == handle["size"] + child_handle["size"]
    import builtins

    consumed = []
    real_open = builtins.open

    class Meter:
        def __init__(self, handle):
            self.handle = handle

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def read(self, size=-1):
            raw = self.handle.read(size)
            consumed.append(len(raw))
            return raw

    def opened(path, *args, **kwargs):
        handle = real_open(path, *args, **kwargs)
        return Meter(handle) if str(path).endswith(".xz") else handle

    monkeypatch.setattr(builtins, "open", opened)
    report = packs.restore_checkpoint_handles(root)
    assert report["status"] == "restored" and report["handles"] == 2 and report["packs"] == 2
    assert _handle_path(root, "task-a", handle).read_bytes().startswith(b"# focus")
    assert _handle_path(child, "task-a", child_handle).read_bytes() == b"# child focus\n"
    assert 0 < sum(consumed) < 64 << 10  # bounded compressed prefix, not the 3 MiB blob
    assert not _loose(refs[0])  # blobs stay packed
    assert packs.pending_restore_bytes(root) == 0
    assert packs.restore_checkpoint_handles(root)["handles"] == 0  # skip-if-present
    pack = root / "observability" / "packs" / "task-a.xz"
    pack.write_bytes(pack.read_bytes()[:20])  # the stream header only: no member decodes
    _handle_path(root, "task-a", handle).unlink()
    packs._READERS.clear()
    with pytest.raises(OSError):
        packs.restore_checkpoint_handles(root)


def test_unpacker_script_restores_a_task_and_the_v760_reader_reads_it(root):
    refs, handle = _task(root, "task-a", blobs=2)
    _run(root)
    script = root / "observability" / "packs" / packs.UNPACKER_NAME
    assert script.exists() and (root / "observability" / "packs" / packs.README_NAME).exists()
    done = subprocess.run([sys.executable, str(script), "task-a"], capture_output=True, text=True, cwd=str(root), check=True)
    assert "restored 3 file" in done.stdout
    assert all(_loose(ref) for ref in refs) and _handle_path(root, "task-a", handle).exists()
    v760 = subprocess.run(["git", "show", "v7.6.0:ouroboros/observability.py"], capture_output=True, text=True, cwd=str(REPO))
    if v760.returncode != 0:
        pytest.skip("v7.6.0 is not reachable from this checkout")
    source = v760.stdout
    start = source.index("def read_blob_ref(")
    body = source[start:source.index("\ndef ", start + 1)]
    namespace = {"pathlib": pathlib, "gzip": gzip, "hashlib": hashlib, "json": json, "Dict": dict, "Any": object,
                 "_blob_ref_path": obs._blob_ref_path, "OBSERVABILITY_DIR": "observability"}
    exec(compile("from __future__ import annotations\n" + body, "v7.6.0", "exec"), namespace)
    assert namespace["read_blob_ref"](root, refs[1])["message"]["content"].startswith("answer task-a 1")
    again = subprocess.run([sys.executable, str(script), "--all"], capture_output=True, text=True, cwd=str(root), check=True)
    assert "restored 0 file" in again.stdout  # idempotent


def test_usage_archive_segments_become_xz_and_attempt_ids_stay_findable(root):
    directory = root / "archive" / "usage_ledger"
    directory.mkdir(parents=True)
    rows = [json.dumps({"attempt_id": f"att-{i}", "state": "settled"}) for i in range(50)]
    (directory / "segment_ep0001.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    report = packs.archive_usage_segments(root)
    assert report["archived"] == 1 and not (directory / "segment_ep0001.jsonl").exists()
    meta = json.loads((directory / "segment_ep0001.jsonl.xz.json").read_text(encoding="utf-8"))
    assert meta["size"] == report["bytes"] and len(meta["sha256"]) == 64
    with lzma.open(directory / "segment_ep0001.jsonl.xz", "rb") as handle:
        assert handle.read().count(b"\n") == 50
    from ouroboros.model_send_seal import _retained_attempt_ids
    assert _retained_attempt_ids(root, {"att-7", "att-49"}) == {"att-7", "att-49"}
    assert _retained_attempt_ids(root, {"att-999"}) == set()
    archived = directory / "segment_ep0001.jsonl.xz"
    archived.write_bytes(archived.read_bytes()[:-9])  # truncated: unknown, never absence
    assert _retained_attempt_ids(root, {"att-999"}) is None
    done = subprocess.run([sys.executable, str(root / "observability" / "packs" / packs.UNPACKER_NAME), "--usage-archive"],
                          capture_output=True, text=True, cwd=str(root))
    assert done.returncode != 0 and "EOFError" in done.stderr


def test_unpacker_restores_usage_archive(root):
    directory = root / "archive" / "usage_ledger"
    directory.mkdir(parents=True)
    (directory / "segment_ep0002.jsonl").write_text('{"attempt_id": "a"}\n', encoding="utf-8")
    _task(root, "task-a", blobs=1)
    _run(root)
    assert not (directory / "segment_ep0002.jsonl").exists()
    done = subprocess.run([sys.executable, str(root / "observability" / "packs" / packs.UNPACKER_NAME), "--usage-archive"],
                          capture_output=True, text=True, cwd=str(root), check=True)
    assert (directory / "segment_ep0002.jsonl").read_text(encoding="utf-8") == '{"attempt_id": "a"}\n', done.stdout


# ---- the packable predicate --------------------------------------------------------

@pytest.mark.parametrize("review", ["none", "closed", "open", "unclassified"])
@pytest.mark.parametrize("status", ["completed", "failed", "cancelled", "rejected_duplicate",
                                    "requested", "scheduled", "running", "interrupted"])
def test_packable_is_a_terminal_row_with_no_open_review_and_nothing_else(root, status, review):
    """Both directions: a terminal row packs whatever duty, projection or acceptance debt it carries, unless a
    review of it is still open (its agent-session reader holds a retained handle's absolute path); a row no
    body with the selector wrote (NULL) has no such reader and packs; a live row never packs."""
    from ouroboros import store
    from ouroboros.task_status import SETTLED_STATUSES

    pointers = {"none": {}, "closed": {"op": {"surface": "multi_model_review", "state": "closed"}},
                "open": {"op": {"surface": "multi_model_review", "state": "dispatched"}}, "unclassified": {}}
    _task(root, "t", blobs=1, status=status, canonical_terminal_projection_origin="terminal_transition",
          review_operations=pointers[review], acceptance_debt={"debt_id": "d1"})
    if review == "unclassified":
        with store.open_store(root, write=True) as txn:
            txn.conn.execute("UPDATE tasks SET review_collection_pending=NULL WHERE task_id='t'")
    with store.open_store(root, write=False) as txn:
        assert packs.task_packable(txn, "t") == (status in SETTLED_STATUSES and review != "open")
        assert not packs.task_packable(txn, "missing")


# ---- the supervisor side ----------------------------------------------------------

class _FakeChild:
    def __init__(self, payload: bytes):
        import io

        self.pid = 4242
        self.stdout = io.BytesIO(payload)
        self.killed = threading.Event()

    def kill(self):
        self.killed.set()

    def wait(self):
        return -9 if self.killed.is_set() else 0


def test_maintenance_tick_spawns_one_supervised_child_then_rests(root, monkeypatch):
    from ouroboros import maintenance, process_custody

    seen = []
    report = json.dumps({"status": "completed", "jobs": {"evidence_packing": {"state": "done", "items": 1}}}).encode()
    gate = threading.Event()

    def fake_spawn(cmd, **kwargs):
        seen.append((cmd, kwargs))
        assert gate.wait(10)
        maintenance.run(root)  # the child's pass, in process
        return _FakeChild(report)

    monkeypatch.setattr(process_custody, "spawn_supervised", fake_spawn)
    monkeypatch.setattr(process_custody, "current_custody_session_id", lambda: "sess-1")
    (root / "logs").mkdir()
    _task(root, "task-a", blobs=1)
    unit = maintenance.MaintenanceChild()
    assert unit.tick(root, REPO) is True
    assert unit.tick(root, REPO) is False  # one child at a time
    gate.set()
    unit._thread.join(10)
    assert not unit._thread.is_alive() and len(seen) == 1
    cmd, kwargs = seen[0]
    assert cmd[1:3] == ["-m", "ouroboros.maintenance"] and "--custody-session" in cmd
    assert kwargs["purpose"] == "maintenance" and kwargs["scope"] == "session"
    assert kwargs["env"]["OUROBOROS_DATA_DIR"] == str(root) and "OUROBOROS_UNRELATED" not in kwargs["env"]
    rows = [json.loads(line) for line in (root / "logs" / "supervisor.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["phase"] for row in rows if row.get("type") == "maintenance"] == ["started", "completed"]
    assert [row["phase"] for row in rows if row.get("type") == "storage_pack"][-1:] == ["completed"]
    # The pass unlinked: one more look is owed (it records the moved directories' stamps), then quiet.
    assert maintenance.due(root) == ["evidence_packing"]
    assert unit.tick(root, REPO) is True
    unit._thread.join(10)
    assert len(seen) == 2 and maintenance.due(root) == []
    # Quiet: the facts the look recorded hold, so the next tick spawns nothing and writes no row.
    log = (root / "logs" / "supervisor.jsonl").read_bytes()
    monkeypatch.setattr(process_custody, "spawn_supervised", lambda *a, **k: pytest.fail("quiet root spawned"))
    assert unit.tick(root, REPO) is True
    unit._thread.join(10)
    assert (root / "logs" / "supervisor.jsonl").read_bytes() == log
    unit.stop()
    assert unit.tick(root, REPO) is False


def test_packing_job_wakes_for_a_task_closing_never_for_a_live_tasks_writes(root, monkeypatch):
    from ouroboros import maintenance

    _task(root, "live", blobs=1, status="running")
    assert maintenance.run_job(root, "evidence_packing")["state"] == "done"  # the first look records its facts
    assert "evidence_packing" not in maintenance.due(root) and packer.changed_tasks(root) == []
    _result(root, "live", "running", step=2)
    _task(root, "later", blobs=1, status="running")
    assert "evidence_packing" not in maintenance.due(root)  # a live task's start and writes: no reason to spawn
    assert maintenance.run_job(root, "evidence_packing") is None
    assert packer.changed_tasks(root) == ["later"]  # a pass run for another reason looks once, records, rests
    _run(root)
    assert packer.changed_tasks(root) == []
    _result(root, "live", "completed")
    assert "evidence_packing" in maintenance.due(root)  # a terminal row moved: a pass may be owed
    assert packer.changed_tasks(root) == ["live"]
    assert maintenance.run_job(root, "evidence_packing")["state"] == "done"
    assert packs.published_generations(root, "live")


def test_child_main_runs_the_due_jobs_and_prints_the_report(root, monkeypatch):
    from ouroboros import maintenance

    _task(root, "task-a", blobs=1)
    from ouroboros import process_custody
    monkeypatch.setattr(process_custody, "adopt_session_id", lambda *a, **k: None)
    monkeypatch.setattr(process_custody, "start_parent_lifeline", lambda *a, **k: None)
    monkeypatch.setattr(maintenance, "_lower_priority", lambda: None)
    import io
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    assert maintenance.main(["--data-root", str(root), "--custody-session", "s"]) == 0
    report = json.loads(out.getvalue())
    assert report["status"] == "completed" and report["jobs"]["evidence_packing"]["state"] == "done"
    assert packs.published_generations(root, "task-a")
    fields = maintenance._report_fields(out.getvalue().encode())
    assert fields["jobs"]["evidence_packing"] == {"state": "done", "items": 1}
    assert maintenance._report_fields(b"garbage") == {}
    assert maintenance._report_fields(json.dumps({"status": "completed", "jobs": {"unknown_job": {}}}).encode()) == {}
    assert maintenance._report_fields(json.dumps(
        {"status": "completed", "jobs": {"evidence_packing": {"state": "done", "items": -1}}}).encode()) == {}
