"""A retention walk decodes each XZ pack forward once, however its references are ordered."""

import hashlib
import json
from collections import deque

import pytest

from ouroboros import evidence_packer as packer
from ouroboros import evidence_packs as packs
from ouroboros import headless, observability as obs
from ouroboros import source_retention
from ouroboros.task_results import write_task_result
from tests._packed_read_probe import count_decoding, count_packed_reads, release_pack_readers
from tests.test_evidence_packs import _blob

MEMBERS = 24


@pytest.fixture
def root(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    packs._READERS.clear()
    yield data
    release_pack_readers(tmp_path)
    packs._READERS.clear()


def _packed_task(root, task="packed-walk"):
    """A retained child whose calls name members in the opposite of pack order (the packer stores smallest first)."""
    child = headless.prepare_task_drive(root, task, "empty")
    calls = []
    for index in range(MEMBERS):
        body = hashlib.sha256(str(index).encode()).hexdigest() * (900 - 30 * index)
        calls.append(obs.persist_call(child, task_id=task, call_id=f"call_{index:02d}", call_type="tool_call",
                                      payload={"result": body}))
    write_task_result(child, task, "completed", result="saved answer", artifact_status="ready")
    assert packer.run_pass(child)["packed"] == 1
    assert not any((child / "observability" / "blobs").glob("*.gz"))
    return task, child, calls


def _walk(root, child, task):
    state = {"schema_version": 1, "status": "complete", "promoted_ref_count": 0, "promoted_source_handle_count": 0,
             "pending_refs": [], "unavailable_refs": []}
    with obs.child_ref_promotion_scope(), source_retention.retention_walk(root, child, state) as walk:
        walk.inventory(task)
    return state


def _raw_total(root, task):
    return next(sidecar["raw_total"] for _path, sidecar in packs.list_sidecars(root) if sidecar["task_id"] == task)


def _proven(state):
    assert state["status"] == "complete" and not state["unavailable_refs"] and not state["pending_refs"], state


def test_many_members_cost_one_pass_over_the_pack(root, tmp_path, monkeypatch):
    task, child, calls = _packed_task(root)
    decoding = count_decoding(monkeypatch)
    state = _walk(root, child, task)
    _proven(state)
    assert state["promoted_ref_count"] >= MEMBERS
    assert decoding["starts"] <= 2, decoding  # the first open and at most one wrap, never one per member
    assert decoding["decoded"] <= 2 * _raw_total(child, task), decoding
    for call in calls:
        manifest = obs.read_call_manifest_ref(root, call["manifest_ref"], task_id=task)
        assert obs.read_blob_ref(root, manifest["full_payload_ref"])


def test_the_same_walk_in_reference_order_rewinds_for_every_member(root, tmp_path, monkeypatch):
    """Control: the fixture must be able to show the cost the ordering removes."""
    task, child, _calls = _packed_task(root)
    monkeypatch.setattr(source_retention.RetentionWalk, "_position", lambda self, node: ())
    decoding = count_decoding(monkeypatch)
    _proven(_walk(root, child, task))
    assert decoding["starts"] >= MEMBERS / 2, decoding
    assert decoding["decoded"] >= 4 * _raw_total(child, task), decoding


def test_damaged_pack_stays_unavailable_and_is_not_served_to_the_next_carrier(root, tmp_path):
    task, child, _calls = _packed_task(root)
    pack = packs.packs_dir(child) / packs.pack_name(task, 1)
    data = bytearray(pack.read_bytes())
    data[len(data) // 2] ^= 0xFF
    pack.write_bytes(bytes(data))
    state = _walk(root, child, task)
    assert state["unavailable_refs"], state
    assert state["status"] != "complete" or state["unavailable_refs"]


def test_read_position_names_only_what_a_read_would_decode(root, tmp_path):
    task, child, _calls = _packed_task(root)
    _path, sidecar = next(iter(packs.list_sidecars(child)))
    member = next(item for item in sidecar["members"] if item["kind"] == "json")
    assert packs.read_position(child, sha256=member["sha256"], kind="json") == (
        str(packs.packs_dir(child) / packs.pack_name(task, 1)), int(member["offset"]))
    assert packs.read_position(child, sha256="0" * 64, kind="json") is None
    loose = _blob(child, b'{"loose": true}')
    assert packs.read_position(child, sha256=loose["sha256"], kind="json") == ()
    assert packs.read_position(tmp_path, sha256=member["sha256"], kind="json") is None  # another root holds nothing


def test_discovery_behind_the_sweep_waits_for_the_next_forward_pass(tmp_path):
    walk = source_retention.RetentionWalk(tmp_path / "parent", tmp_path / "child", {})
    positions = {}
    walk._position = lambda node: positions[node["key"]]

    def queue(key, position):
        positions[key] = position
        walk.queue.append({"key": key, "kind": "blob", "ref": {}})

    queue("high", ("pack", 300))
    queue("low", ("pack", 100))
    assert walk._next_node()[0]["key"] == "low"
    queue("behind", ("pack", 50))  # the cursor is past it: decoding it now would rewind
    queue("ahead", ("pack", 200))
    assert [walk._next_node()[0]["key"] for _ in range(3)] == ["ahead", "high", "behind"]
    assert walk._next_node() == (None, ())


def test_a_view_skeleton_still_follows_its_packed_parts(tmp_path):
    walk = source_retention.RetentionWalk(tmp_path / "parent", tmp_path / "child", {})
    positions = {"skeleton": (), "part": ("pack", 10)}
    walk._position = lambda node: positions[node["key"]]
    walk.queue = deque([{"key": "skeleton", "kind": "blob", "ref": {}}, {"key": "part", "kind": "blob", "ref": {}}])
    assert [walk._next_node()[0]["key"] for _ in range(2)] == ["part", "skeleton"]


def test_two_carriers_of_one_member_read_it_once(root, tmp_path, monkeypatch):
    task, child, calls = _packed_task(root)
    # The same blob, reached through a second carrier (a shared part), sorts next to its first read.
    path = child / "observability" / "calls" / task / "call_00.json"
    manifest = json.loads(path.read_text())
    manifest["shared_parts"] = [manifest["full_payload_ref"]]
    path.write_text(json.dumps(manifest))
    reads = count_packed_reads(monkeypatch)
    _proven(_walk(root, child, task))
    assert reads and max(reads.values()) == 1, reads


def test_views_reassembled_after_the_sweep_decode_each_shared_part_once(root, tmp_path, monkeypatch):
    """A retained request view names many parts; hundreds of views share them. Reassembling them part by part was the
    bulk of the cost on a real 4.5 GB pack: every repeat of a small part rewinds past the checkpoint prefix."""
    settled = [{"role": "user", "content": f"Shared turn {index} " + "x" * 2000} for index in range(MEMBERS)]
    view = {"messages": settled}
    obs.persist_call(root, task_id="settled", call_id="llm_1_request", call_type="llm_request", payload=view)
    write_task_result(root, "settled", "completed", result="saved answer", artifact_status="ready")
    assert packer.run_pass(root)["packed"] == 1
    task = "promoted"
    child = headless.prepare_task_drive(root, task, "empty")
    views = [{"messages": [*settled, {"role": "user", "content": f"Question {index}"}]} for index in range(8)]
    for index, payload in enumerate(views):
        obs.persist_call(child, task_id=task, call_id=f"llm_{index}_request", call_type="llm_request", payload=payload)
    reads = count_packed_reads(monkeypatch)
    decoding = count_decoding(monkeypatch)
    _proven(_walk(root, child, task))
    assert reads and set(reads.values()) == {1}, reads
    assert decoding["starts"] <= 2, decoding


def test_a_part_cached_for_one_root_never_answers_for_another(root, tmp_path):
    """A child's copy of a part must not satisfy a check of whether the parent holds it."""
    from ouroboros.shared_values import token_digests

    other = tmp_path / "other"
    other.mkdir()
    messages = [{"role": "user", "content": f"turn {index} " + "x" * 300} for index in range(4)]
    ref = obs.write_shared_blob(root, {"messages": messages}, {})
    digest = token_digests(packs.read_loose_gzip(packs.shared_placement_path(root, ref["sha256"])))[0]
    lost = root / "observability" / "blobs" / f"{digest}.json.gz"
    raw = packs.read_loose_gzip(lost)
    lost.unlink()
    whole = {**ref, "path": ""}
    with packs.part_cache_scope():
        packs.remember_part(other, digest, raw)
        with pytest.raises(FileNotFoundError):
            packs.blob_bytes(root, whole)
        packs.remember_part(root, digest, raw)
        assert packs.blob_bytes(root, whole)


def test_a_carrier_claiming_another_size_is_checked_by_its_own_read(root, tmp_path):
    """Two carriers name one packed member and the later one claims a size the bytes lack. The read that verified
    the member for the first carrier never vouches for the second: it is read and checked itself, so it stays
    unavailable, as when every carrier was read alone."""
    task = "packed-claims"
    child = headless.prepare_task_drive(root, task, "empty")
    named = obs.persist_call(child, task_id=task, call_id="call_00", call_type="tool_call",
                             payload={"result": "y" * 4000})
    ref = obs.read_call_manifest_ref(child, named["manifest_ref"], task_id=task)["full_payload_ref"]
    wrong = {**ref, "size": int(ref["size"]) + 1}
    # Smaller, so it packs first: the sweep discovers the wrong carrier there, and it sorts right after the member's
    # own carrier (queued earlier), the order a forward sweep reads them in.
    namer = obs.persist_call(child, task_id=task, call_id="call_01", call_type="tool_call",
                             payload={"source_ref": wrong})
    namer_ref = obs.read_call_manifest_ref(child, namer["manifest_ref"], task_id=task)["full_payload_ref"]
    assert obs.read_blob_ref(child, namer_ref) == {"source_ref": wrong}
    write_task_result(child, task, "completed", result="saved answer", artifact_status="ready")
    assert packer.run_pass(child)["packed"] == 1
    state = _walk(root, child, task)
    assert [fact.get("size") for fact in state["unavailable_refs"]] == [wrong["size"]], state
    assert json.loads(packs.blob_bytes(root, {**ref, "path": ""})) == {"result": "y" * 4000}


def test_members_sharing_an_offset_are_each_read_as_themselves(root, tmp_path):
    """An empty member takes no bytes, so the member after it starts at the same offset: one (pack, offset) is not
    one member. A carrier of each gets its own member's bytes."""
    task = "packed-empty"
    child = headless.prepare_task_drive(root, task, "empty")
    empty = obs.write_blob_bytes(child, b"", kind="txt")
    # The manifest keeps the empty member in the pack; the payload names it, so the walk meets it right after
    # reading the payload, the smallest other member, at the empty member's offset.
    call = obs.persist_call(child, task_id=task, call_id="call_00", call_type="tool_call",
                            payload={"source_ref": empty}, manifest={"empty_ref": empty})
    payload = obs.read_call_manifest_ref(child, call["manifest_ref"], task_id=task)["full_payload_ref"]
    assert obs.read_blob_ref(child, payload) == {"source_ref": empty}
    write_task_result(child, task, "completed", result="saved answer", artifact_status="ready")
    assert packer.run_pass(child)["packed"] == 1
    _path, sidecar = next(iter(packs.list_sidecars(child)))
    offsets = {(member["sha256"], member["kind"]): member["offset"] for member in sidecar["members"]}
    assert offsets[(empty["sha256"], "txt")] == offsets[(payload["sha256"], "json")], sidecar["members"]
    _proven(_walk(root, child, task))
    assert packs.blob_bytes(root, {**empty, "path": ""}) == b""


def test_a_part_offered_again_is_counted_once():
    """A part offered again replaces its copy: the bound counts its bytes once and evicts nothing that fits."""
    cache = packs._PartCache(10)
    cache.put(("root", "a"), b"12345")
    cache.put(("root", "b"), b"1234")
    cache.put(("root", "a"), b"12345")
    assert cache.size == sum(len(raw) for raw in cache.parts.values()) == 9
    assert (cache.get(("root", "a")), cache.get(("root", "b"))) == (b"12345", b"1234")


def test_a_response_two_carriers_read_apart_is_counted_once_by_the_part_cache(root, tmp_path, monkeypatch):
    """Two carriers name one packed response and another read comes between them, so each reads it and offers it to
    the walk's part cache. The second offer replaces the first: the cache's bound keeps counting its real bytes."""
    settled = obs.persist_call(root, task_id="settled", call_id="call_00", call_type="tool_call",
                               payload={"answer": "z" * 3000})
    write_task_result(root, "settled", "completed", result="saved answer", artifact_status="ready")
    assert packer.run_pass(root)["packed"] == 1
    ref = obs.read_call_manifest_ref(root, settled["manifest_ref"], task_id="settled")["full_payload_ref"]
    task = "two-carriers"
    child = headless.prepare_task_drive(root, task, "empty")
    # The same response at the canonical address and at the one a child drive writes: two carriers, one member.
    at_child = {**ref, "path": str(child / "observability" / "blobs" / f"{ref['sha256']}.json.gz")}
    for index, carried in enumerate((ref, at_child)):
        obs.persist_call(child, task_id=task, call_id=f"call_{index:02d}", call_type="tool_call",
                         payload={"response_ref": carried})
    offers, put = [], packs._PartCache.put

    def accounted(cache, key, raw):
        offers.append(key in cache.parts)
        put(cache, key, raw)
        assert cache.size == sum(len(part) for part in cache.parts.values()), (cache.size, offers)

    monkeypatch.setattr(packs._PartCache, "put", accounted)
    _proven(_walk(root, child, task))
    assert True in offers, offers  # the second carrier's read offered the cached response again
