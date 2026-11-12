"""Request views stored once: skeleton placement plus parts, exact wholes, public parts, packer reach."""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import pathlib
import shutil

import pytest

from ouroboros import evidence_packer as packer
from ouroboros import evidence_packs as packs
from ouroboros import observability as obs
from ouroboros.observability import _json_bytes
from ouroboros.shared_values import SharedValueMissing, token_digests

TOOLS = [{"type": "function", "function": {"name": f"tool_{i}", "description": "Ω " * 40 + str(i),
                                             "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}
         for i in range(20)]


def _transcript(rounds: int = 20, start: int = 0) -> list:
    messages = [{"role": "system", "content": "SYSTEM Ω " * 2000}, {"role": "user", "content": "Inspect the project."}]
    for index in range(start, start + rounds):
        call = f"call_{index}"
        messages += [
            {"role": "assistant", "content": f"Round {index}", "tool_calls": [
                {"id": call, "type": "function", "function": {"name": "tool_1", "arguments": json.dumps({"path": f"m{index}.py"})}}]},
            {"role": "tool", "tool_call_id": call, "content": f"File {index}: café / состояние / 😀\n" + "x" * 1500},
        ]
    return messages


def _view(messages, tools=TOOLS, **extra) -> dict:
    return {"messages": messages, "send_messages": messages, "tools": tools, "model": "m", "reasoning_effort": "high", **extra}


def _files(root: pathlib.Path) -> dict:
    base = root / "observability"
    return {p: p.stat().st_size for p in base.rglob("*") if p.is_file() and not p.name.startswith(".")}


def _persist(root, call_id, payload, call_type="llm_request", task_id="task", **kw):
    before = _files(root)
    refs = obs.persist_call(root, task_id=task_id, call_id=call_id, call_type=call_type, payload=payload, **kw)
    after = _files(root)
    manifest = json.loads(pathlib.Path(refs["manifest_ref"]["path"]).read_text(encoding="utf-8"))
    written = sum(size for path, size in after.items() if path not in before)
    return refs, manifest, written


def _skeleton(root, ref) -> bytes:
    return packs.read_loose_gzip(packs.shared_placement_path(root, ref["sha256"]))


def _task_row(root, task_id, status="completed"):
    from ouroboros import store
    from ouroboros.utils import utc_now_iso

    with store.open_store(root, write=True) as txn:
        txn.store_task({"task_id": task_id, "status": status, "_schema_version": 1, "updated_at": utc_now_iso()})


@pytest.fixture
def root(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    monkeypatch.delenv("OUROBOROS_OBSERVABILITY_KEEP_RAW", raising=False)
    packs._READERS.clear()
    obs._TASK_PARTS.clear()
    yield data
    packs._READERS.clear()
    obs._TASK_PARTS.clear()


def test_a_request_view_stores_each_element_once_and_keeps_the_whole_digest(root):
    first = _view(_transcript())
    refs, manifest, written_first = _persist(root, "llm_1_request", first)
    ref = manifest["full_payload_ref"]
    redacted = obs.redact_projection(first).value
    assert ref["sha256"] == hashlib.sha256(_json_bytes(redacted)).hexdigest() and ref["size"] == len(_json_bytes(redacted))
    assert not pathlib.Path(ref["path"]).exists() and packs.shared_placement_path(root, ref["sha256"]).is_file()
    assert ref["path"].endswith(f"{ref['sha256']}.json.gz"), "the ref names the canonical loose address"
    elements = {hashlib.sha256(_json_bytes(e)).hexdigest() for e in first["messages"] + first["tools"]}
    assert {part["sha256"] for part in manifest["shared_parts"]} == elements
    assert set(token_digests(_skeleton(root, ref))) == elements
    assert all(set(part) == {"sha256", "kind", "encoding", "size"} for part in manifest["shared_parts"])
    for part in manifest["shared_parts"]:
        assert hashlib.sha256(packs.blob_bytes(root, part)).hexdigest() == part["sha256"]
    _manifest, payload, _ = obs.read_call_payload(root, task_id="task", call_id="llm_1_request")
    assert payload == json.loads(_json_bytes(redacted))
    assert obs.read_blob_ref(root, ref, raw_bytes=True) == _json_bytes(redacted)

    # The next call: an edit near the beginning (compaction replaced message 2) and three new messages.
    second = _view(copy.deepcopy(first["messages"]))
    second["messages"][2] = {"role": "assistant", "content": "[compacted] rounds 0-3 folded"}
    second["messages"] += _transcript(2, start=100)[2:5]
    _refs2, manifest2, written_second = _persist(root, "llm_2_request", second)
    ref2 = manifest2["full_payload_ref"]
    assert ref2["sha256"] == hashlib.sha256(_json_bytes(obs.redact_projection(second).value)).hexdigest()
    new_elements = {hashlib.sha256(_json_bytes(e)).hexdigest() for e in second["messages"]} - elements
    assert {part["sha256"] for part in manifest2["shared_parts"]} == new_elements and len(new_elements) == 4
    assert written_second * 10 < ref2["size"], f"call 2 wrote {written_second} B for a {ref2['size']} B view"
    assert written_first < ref["size"] * 1.5, "the first call writes the elements once plus the skeleton"
    _, payload2, _ = obs.read_call_payload(root, task_id="task", call_id="llm_2_request")
    assert payload2 == json.loads(_json_bytes(obs.redact_projection(second).value))


def test_the_physical_candidate_and_the_claudexor_request_share_parts_and_reconstruct_exactly(root):
    from ouroboros.model_send_seal import _read_blob_bytes, _reconstruct_blob_bytes, persist_physical_candidate
    from ouroboros.request_wire_contract import physical_candidate_bytes

    messages = _transcript(5)
    candidate = {"model": "m", "messages": messages, "tools": TOOLS, "max_tokens": 10}
    raw = physical_candidate_bytes(candidate)
    persisted = persist_physical_candidate(root, task_id="task", attempt_id="attempt-1", candidate=candidate,
                                           candidate_facts={"candidate_raw_sha256": hashlib.sha256(raw).hexdigest(),
                                                            "candidate_raw_size_bytes": len(raw),
                                                            "candidate_measurement_kind": "canonical_json_v1"})
    manifest = json.loads(pathlib.Path(persisted["manifest_ref"]["path"]).read_text())
    stored, intact = _read_blob_bytes(manifest)
    assert intact and stored == _reconstruct_blob_bytes(candidate), "the seal's reconstruction equals the spliced whole"
    assert manifest["model_send_seal"]["attempt_id"] == "attempt-1" and manifest["shared_parts"]
    physical_parts = {p["sha256"] for p in manifest["shared_parts"]}

    request = {"account": {"profileId": "p"}, "messages": messages, "tools": TOOLS, "options": {"effort": "high"}}
    refs, request_manifest, _ = _persist(root, "inv_model_request", request, call_type="llm_claudexor_request", keep_raw=True)
    assert request_manifest["full_payload_redacted"] is False
    assert request_manifest["full_payload_ref"]["sha256"] == hashlib.sha256(_json_bytes(request)).hexdigest()
    assert request_manifest["redacted_projection_ref"]["sha256"] == hashlib.sha256(_json_bytes(obs.redact_projection(request).value)).hexdigest()
    assert "shared_parts" not in request_manifest, "every element was listed by the candidate's manifest already"
    assert set(token_digests(_skeleton(root, request_manifest["full_payload_ref"]))) <= physical_parts
    assert obs.read_blob_ref(root, request_manifest["full_payload_ref"]) == json.loads(_json_bytes(request))
    assert obs.read_blob_ref(root, request_manifest["redacted_projection_ref"]) == json.loads(_json_bytes(request))


def test_the_public_skeleton_names_only_public_parts(root):
    secret = "ghp_" + "abcdefghijklmnopqrstuvwxyz123456"
    messages = _transcript(3) + [{"role": "tool", "tool_call_id": "x", "content": f"token: {secret}", "api_key": secret}]
    payload = _view(messages)
    _refs, manifest, _ = _persist(root, "inv_model_request", payload, call_type="llm_claudexor_request", keep_raw=True)
    private = set(token_digests(_skeleton(root, manifest["full_payload_ref"])))
    public = set(token_digests(_skeleton(root, manifest["redacted_projection_ref"])))
    redacted_elements = {hashlib.sha256(_json_bytes(e)).hexdigest()
                         for e in obs.redact_projection(payload).value["messages"] + TOOLS}
    assert public <= redacted_elements, "a public skeleton names only parts whose bytes are redacted"
    private_only = private - public
    assert len(private_only) == 1 and len(public - private) == 1, "the secret-bearing element: one private, one public part"
    for digest in private_only:
        assert secret.encode() in packs.blob_bytes(root, {"sha256": digest, "kind": "json", "encoding": "gzip"})
    for digest in public:
        assert secret.encode() not in packs.blob_bytes(root, {"sha256": digest, "kind": "json", "encoding": "gzip"})
    assert {p["sha256"] for p in manifest["shared_parts"]} == private | public
    assert manifest["redaction"]["redacted"] is True
    assert secret not in json.dumps(obs.read_blob_ref(root, manifest["redacted_projection_ref"]))


def test_other_records_and_views_without_elements_are_written_whole(root):
    for call_type, payload in (("tool_call", {"tool": "run", "messages": [{"role": "user", "content": "x"}]}),
                               ("llm_request", {"messages": [], "tools": []}),
                               ("llm_response", {"message": {"role": "assistant", "content": "ok", "tool_calls": [{"id": "c"}]}})):
        _refs, manifest, _ = _persist(root, f"{call_type}_1", payload, call_type=call_type)
        ref = manifest["full_payload_ref"]
        assert pathlib.Path(ref["path"]).is_file() and not packs.shared_placement_path(root, ref["sha256"]).exists()
        assert "shared_parts" not in manifest
        assert obs.read_blob_ref(root, ref) == json.loads(_json_bytes(obs.redact_projection(payload).value))


def test_a_whole_that_already_lies_loose_is_kept_and_an_old_whole_blob_reads_unchanged(root):
    payload = _view(_transcript(2))
    legacy = obs.write_blob(root, obs.redact_projection(payload).value)  # an older body's whole view
    _refs, manifest, written = _persist(root, "llm_old_request", payload)
    assert manifest["full_payload_ref"]["sha256"] == legacy["sha256"] and "shared_parts" not in manifest
    assert not packs.shared_placement_path(root, legacy["sha256"]).exists() and written < 4096, "only the manifest"
    assert obs.read_blob_ref(root, manifest["full_payload_ref"]) == json.loads(_json_bytes(obs.redact_projection(payload).value))


def test_the_reader_admits_every_physical_file_bounds_bytes_and_types_missing_and_damage(root):
    _refs, manifest, _ = _persist(root, "llm_1_request", _view(_transcript(2)))
    ref = manifest["full_payload_ref"]
    seen = []
    assert obs.read_blob_ref(root, ref, admit=seen.append) == json.loads(_json_bytes(obs.redact_projection(_view(_transcript(2))).value))
    names = {p.name for p in seen}
    assert f"{ref['sha256']}.json.gz" in names and f"{ref['sha256']}.json{packs.SHARED_PLACEMENT_SUFFIX}" in names
    assert all(f"{p['sha256']}.json.gz" in names for p in manifest["shared_parts"]), "each part is admitted before it is opened"
    with pytest.raises(ValueError, match="byte limit"):
        obs.read_blob_ref(root, ref, max_bytes=ref["size"] - 1)
    with pytest.raises(ValueError, match="byte limit"):
        packs.blob_bytes(root, {**ref, "size": None}, max_bytes=ref["size"] - 1)
    with pytest.raises(PermissionError):
        obs.read_blob_ref(root, ref, admit=lambda path: (_ for _ in ()).throw(PermissionError(path.name))
                          if path.name.endswith(".json.gz") and path.name != f"{ref['sha256']}.json.gz" else None)

    part = manifest["shared_parts"][0]
    part_path = root / "observability" / "blobs" / f"{part['sha256']}.json.gz"
    kept = part_path.read_bytes()
    with gzip.open(part_path, "wb") as handle:
        handle.write(b'{"damaged": true}')
    with pytest.raises(packs.EvidenceCorrupt):
        obs.read_blob_ref(root, ref)
    part_path.unlink()
    with pytest.raises(FileNotFoundError, match=part["sha256"]):
        obs.read_blob_ref(root, ref)
    part_path.write_bytes(kept)
    assert obs.read_blob_ref(root, ref, raw_bytes=True)
    with gzip.open(packs.shared_placement_path(root, ref["sha256"]), "wb") as handle:
        handle.write(b'{"tampered": true}')
    with pytest.raises(packs.EvidenceCorrupt):
        obs.read_blob_ref(root, ref)
    with pytest.raises(SharedValueMissing):
        from ouroboros.shared_values import splice
        splice(_json_bytes({"messages": [{"$shared": "f" * 64}]}), {})


def test_the_packer_keeps_the_whole_and_every_part_and_retires_the_placement(root):
    first, second = _view(_transcript(4)), _view(_transcript(6))
    _r1, manifest1, _ = _persist(root, "llm_1_request", first, task_id="settled")
    _r2, manifest2, _ = _persist(root, "llm_2_request", second, task_id="settled")
    _task_row(root, "settled")
    wholes = {manifest1["full_payload_ref"]["sha256"], manifest2["full_payload_ref"]["sha256"]}
    parts = {p["sha256"] for m in (manifest1, manifest2) for p in m.get("shared_parts", [])}
    report = packer.run_pass(root)
    assert report["packed"] == 1
    sidecar = packs.published_generations(root, "settled")[0]
    members = {m["sha256"] for m in sidecar["members"]}
    assert wholes <= members and parts <= members and len(members) == len(wholes | parts)
    assert not any(packs.shared_placement_path(root, sha).exists() for sha in wholes), "placements retired"
    assert not any((root / "observability" / "blobs" / f"{sha}.json.gz").exists() for sha in parts), "parts retired"
    for call_id, view in (("llm_1_request", first), ("llm_2_request", second)):
        _m, payload, _ = obs.read_call_payload(root, task_id="settled", call_id=call_id)
        assert payload == json.loads(_json_bytes(obs.redact_projection(view).value))
    # The standalone unpacker restores whole views an older body reads as plain loose blobs.
    import subprocess
    import sys

    result = subprocess.run([sys.executable, str(packs.packs_dir(root) / packs.UNPACKER_NAME), "settled"],
                            capture_output=True, text=True, cwd=str(packs.packs_dir(root)))
    assert result.returncode == 0, result.stdout + result.stderr
    for sha in wholes:
        loose = root / "observability" / "blobs" / f"{sha}.json.gz"
        assert hashlib.sha256(packs.read_loose_gzip(loose)).hexdigest() == sha
    assert packer.run_pass(root)["packed"] == 0


def test_a_part_shared_with_a_live_task_stays_loose_and_each_view_still_reads(root):
    shared_messages = _transcript(3)
    _r1, manifest_settled, _ = _persist(root, "llm_1_request", _view(shared_messages), task_id="settled")
    _r2, manifest_live, _ = _persist(root, "llm_1_request", _view(shared_messages + [{"role": "user", "content": "more"}]),
                                     task_id="live")
    _task_row(root, "settled")
    _task_row(root, "live", status="running")
    shared_parts = {p["sha256"] for p in manifest_settled["shared_parts"]}
    live_parts = {p["sha256"] for p in manifest_live["shared_parts"]}
    assert shared_parts < live_parts, "each task lists every part it uses once: reachability is per task"
    report = packer.run_pass(root)
    assert report["packed"] == 1 and report["kept"] == len(shared_parts), "parts the live task lists stay loose"
    settled_ref = manifest_settled["full_payload_ref"]
    assert not packs.shared_placement_path(root, settled_ref["sha256"]).exists(), "the settled view's placement retired"
    assert all((root / "observability" / "blobs" / f"{sha}.json.gz").is_file() for sha in shared_parts)
    assert packs.shared_placement_path(root, manifest_live["full_payload_ref"]["sha256"]).is_file()
    assert packs.blob_bytes(root, settled_ref) == _json_bytes(obs.redact_projection(_view(shared_messages)).value)
    live_whole = packs.blob_bytes(root, manifest_live["full_payload_ref"])
    assert hashlib.sha256(live_whole).hexdigest() == manifest_live["full_payload_ref"]["sha256"]
    # The sharer settles later: its own pass packs its whole and its parts (a shared sha is duplicated per pack).
    for sha in shared_parts:  # as if the settled task's parts had been retired first: the pack still serves them
        (root / "observability" / "blobs" / f"{sha}.json.gz").unlink()
    assert packs.blob_bytes(root, manifest_live["full_payload_ref"]) == live_whole, "parts come from the other task's pack"
    _task_row(root, "live")
    members = packs.loose_members(root, "live", missing=(missing := []))
    assert not missing and {m["sha256"] for m in members} == live_parts | {manifest_live["full_payload_ref"]["sha256"]}
    assert packer.run_pass(root)["packed"] == 1
    assert not packs.shared_placement_path(root, manifest_live["full_payload_ref"]["sha256"]).exists()
    assert packs.blob_bytes(root, manifest_live["full_payload_ref"]) == live_whole


def test_a_missing_part_is_reported_missing_and_a_damaged_one_refuses_the_pack(root):
    _refs, manifest, _ = _persist(root, "llm_1_request", _view(_transcript(2)), task_id="gap")
    _task_row(root, "gap")
    part = manifest["shared_parts"][-1]
    part_path = root / "observability" / "blobs" / f"{part['sha256']}.json.gz"
    kept = part_path.read_bytes()
    with gzip.open(part_path, "wb") as handle:
        handle.write(b"damaged")
    with pytest.raises(packs.EvidenceCorrupt):
        packs.write_pack(root, "gap", 1, packs.loose_members(root, "gap"))
    part_path.unlink()
    members = packs.loose_members(root, "gap", missing=(missing := []))
    assert missing == [("json", part["sha256"], part["size"])]
    assert any(m["source"].name.endswith(packs.SHARED_PLACEMENT_SUFFIX) for m in members)
    with pytest.raises(FileNotFoundError):
        packs.write_pack(root, "gap", 1, members)  # the whole cannot be spliced without the part
    part_path.write_bytes(kept)
    assert packer.run_pass(root)["packed"] == 1


def test_a_child_root_promotion_leaves_a_readable_whole_in_the_parent(tmp_path):
    from ouroboros import headless

    parent, task = tmp_path / "canonical", "promoted"
    child = headless.prepare_task_drive(parent, task, "empty")
    payload = _view(_transcript(2))
    obs.persist_call(child, task_id=task, call_id="llm_1_request", call_type="llm_request", payload=payload)
    _rewritten, state = obs.promote_child_task_refs(parent, child, task, {})
    assert state["status"] == "complete" and state["call_inventory_preserved"]
    shutil.rmtree(child)
    manifest, restored, _ = obs.read_call_payload(parent, task_id=task, call_id="llm_1_request")
    assert restored == json.loads(_json_bytes(obs.redact_projection(payload).value))
    ref = manifest["full_payload_ref"]
    assert ref["path"].endswith(f"{ref['sha256']}.json.gz") and not pathlib.Path(ref["path"]).exists()
    assert packs.shared_placement_path(parent, ref["sha256"]).is_file(), "the parent holds the placement, not a materialized whole"
    members = packs.loose_members(parent, task, missing=(missing := []))
    assert not missing and {p["sha256"] for p in manifest["shared_parts"]} < {m["sha256"] for m in members}, "every part came along"


def test_a_child_whose_later_call_sorts_first_is_promoted_complete_in_one_pass(root, tmp_path):
    """A later call's view reuses parts only the first call's manifest listed (``persist_call`` lists a part once per
    process); the inventory walks manifests by call id, which sorts in no time order. Every manifest's parts still
    land before any skeleton verifies, so one pass promotes the child complete and its drive can go."""
    from ouroboros import headless

    parent, task = tmp_path / "canonical", "promoted"
    child = headless.prepare_task_drive(parent, task, "empty")
    first, later = _view(_transcript(2)), _view(_transcript(3))
    obs.persist_call(child, task_id=task, call_id="llm_z_request", call_type="llm_request", payload=first)
    obs.persist_call(child, task_id=task, call_id="llm_a_request", call_type="llm_request", payload=later)
    manifests = {call: json.loads(obs.call_manifest_path(child, task, call).read_text(encoding="utf-8"))
                 for call in ("llm_z_request", "llm_a_request")}
    listed = {call: {p["sha256"] for p in manifest.get("shared_parts") or []} for call, manifest in manifests.items()}
    skeleton = _skeleton(child, manifests["llm_a_request"]["full_payload_ref"])
    assert set(token_digests(skeleton)) & listed["llm_z_request"] - listed["llm_a_request"], \
        "the later view names parts only the earlier manifest lists"
    _rewritten, state = obs.promote_child_task_refs(parent, child, task, {})
    assert (state["status"], state["call_inventory_preserved"], state["pending_refs"]) == ("complete", True, [])
    shutil.rmtree(child)
    for call, payload in (("llm_z_request", first), ("llm_a_request", later)):
        _manifest, restored, _ = obs.read_call_payload(parent, task_id=task, call_id=call)
        assert restored == json.loads(_json_bytes(obs.redact_projection(payload).value)), call


def test_a_view_named_outside_a_manifest_brings_its_parts_and_the_child_can_go(tmp_path):
    """A request view that a trace or usage fact names directly, with no call manifest listing its parts, is
    promoted with them: the parent splices the exact whole and the child's drive holds nothing pending."""
    from ouroboros import headless

    parent, task = tmp_path / "canonical", "promoted"
    child = headless.prepare_task_drive(parent, task, "empty")
    payload = _view(_transcript(2))
    ref = obs.write_shared_blob(child, payload, {})
    assert packs.shared_placement_path(child, ref["sha256"]).is_file() and token_digests(_skeleton(child, ref))
    row = {"trace_refs": {"timeout_repeat": {"first": ref}}}  # the task result names the view itself
    promoted, state = obs.promote_child_task_refs(parent, child, task, row)
    assert state["status"] == "complete" and state["pending_refs"] == [], state
    assert promoted["trace_refs"]["timeout_repeat"]["first"]["path"].startswith(str(parent))
    shutil.rmtree(child)
    assert packs.blob_bytes(parent, {**ref, "path": ""}) == _json_bytes(payload)


def test_a_child_view_beside_a_packed_task_proves_each_shared_part_once_then_checks_each_new_whole(root, monkeypatch):
    """The parts a child's views share with an already packed task are each decoded once, by their own node: every
    NEW skeleton is then checked as the whole it splices to from the walk's bounded part cache (a token repeated in
    ``send_messages`` included), never decoding a part again only to learn the parent already holds it."""
    from ouroboros import headless
    from tests._packed_read_probe import count_packed_reads

    settled = _transcript(2)
    _persist(root, "llm_1_request", _view(settled), task_id="settled")
    _task_row(root, "settled")
    assert packer.run_pass(root)["packed"] == 1  # the shared parts are pack members only now
    task = "promoted"
    child = headless.prepare_task_drive(root, task, "empty")
    views = [_view(settled[:2] + [{"role": "user", "content": f"Question {index}"}]) for index in range(3)]
    for index, view in enumerate(views):
        obs.persist_call(child, task_id=task, call_id=f"llm_{index}_request", call_type="llm_request", payload=view)
    reads = count_packed_reads(monkeypatch)
    _rewritten, state = obs.promote_child_task_refs(root, child, task, {})
    assert (state["status"], state["pending_refs"]) == ("complete", []), state
    assert reads and set(reads.values()) == {1}, reads
    shutil.rmtree(child)
    for index, view in enumerate(views):
        _manifest, restored, _ = obs.read_call_payload(root, task_id=task, call_id=f"llm_{index}_request")
        assert restored == json.loads(_json_bytes(obs.redact_projection(view).value))


def test_a_damaged_child_loose_whole_beside_its_healthy_skeleton_keeps_the_child(tmp_path):
    """The verified bytes come from the child's skeleton, but the loose name the walk links is damaged: the
    read-back after that write refuses the copy, so the promotion stays pending and the child is kept."""
    from ouroboros import headless

    parent, task = tmp_path / "canonical", "damaged"
    child = headless.prepare_task_drive(parent, task, "empty")
    payload = _view(_transcript(2))
    ref = obs.write_shared_blob(child, payload, {})
    with gzip.open(ref["path"], "wb") as handle:
        handle.write(b'{"damaged": true}')
    _promoted, state = obs.promote_child_task_refs(parent, child, task, {"trace_refs": {"timeout_repeat": {"first": ref}}})
    assert state["status"] == "incomplete", state
    assert any("failed verification" in str(row.get("reason")) for row in state["pending_refs"]), state
    assert packs.blob_bytes(child, {**ref, "path": ""}) == _json_bytes(payload)


def test_a_parent_skeleton_missing_a_part_regains_it_from_the_child(tmp_path):
    """The parent holds a view's skeleton but lost one part: the walk carries the part, checks the whole again
    after that write and completes without rewriting the skeleton."""
    from ouroboros import headless

    parent, task = tmp_path / "canonical", "regained"
    child = headless.prepare_task_drive(parent, task, "empty")
    payload = _view(_transcript(2))
    ref = obs.write_shared_blob(child, payload, {})
    row = {"trace_refs": {"timeout_repeat": {"first": ref}}}
    assert obs.promote_child_task_refs(parent, child, task, row)[1]["status"] == "complete"
    skeleton = packs.shared_placement_path(parent, ref["sha256"])
    inode = skeleton.stat().st_ino
    lost = parent / "observability" / "blobs" / f"{token_digests(_skeleton(parent, ref))[-1]}.json.gz"
    lost.unlink()
    _promoted, state = obs.promote_child_task_refs(parent, child, task, row)
    assert (state["status"], state["pending_refs"]) == ("complete", []), state
    assert lost.is_file() and skeleton.stat().st_ino == inode, "the part is carried; the parent's skeleton stays"
    shutil.rmtree(child)
    assert packs.blob_bytes(parent, {**ref, "path": ""}) == _json_bytes(payload)


# ---- a part's redaction is worked out once (``redaction.part_fact``; K3) ---------------------------------------

SECRET = "ghp_" + "syntheticfixture0123456789" * 2


def _secret_view() -> dict:
    """Redaction fires inside parts (a token, a secret-named key, a nested block), in the skeleton and in a tool's
    schema; one secret is split across two messages, which no rule reads across."""
    messages = _transcript(2) + [
        {"role": "tool", "tool_call_id": "s", "content": f"token: {SECRET}", "api_key": SECRET},
        {"role": "assistant", "content": [{"type": "text", "text": f"Authorization: Bearer {SECRET}"},
                                          {"type": "text", "text": "plain"}]},
        {"role": "user", "content": "the token is ghp_"}, {"role": "user", "content": "syntheticfixture0123456789" * 2},
    ]
    login = {"type": "function", "function": {"name": "login", "parameters": {"type": "object", "properties": {
        "api_key": {"type": "string", "default": "sk-" + "a" * 30}, "page_token": {"type": "string"}}}}}
    return {**_view(messages, tools=TOOLS[:3] + [login]), "authorization": f"Bearer {SECRET}"}


def _scanned(payload, facts) -> tuple:
    from ouroboros.context_timing import measured_sections, section_spans

    with measured_sections():
        result = obs.redact_projection(payload, facts=facts)
        return result, section_spans()["work"].get("secret_scan", 0)


@pytest.mark.parametrize("payload", [_secret_view(), _view(_transcript(3))], ids=["secrets", "public"])
def test_a_view_redacted_by_parts_is_the_whole_walk_byte_for_byte_cold_and_warm(payload):
    whole = obs.redact_projection(payload)
    facts: dict = {}
    cold, scanned = _scanned(payload, facts)
    warm, rescanned = _scanned(payload, facts)
    for parts in (cold, warm):
        assert _json_bytes(parts.value) == _json_bytes(whole.value)
        assert parts.manifest() == whole.manifest(), "the same rules fired at the same paths, in walk order"
    elements = {_json_bytes(e) for e in payload["messages"] + payload["tools"]}
    assert scanned == sum(map(len, elements)) and rescanned == 0, "each distinct part once, then never again"
    assert whole.value["messages"][-2:] == payload["messages"][-2:], "a secret split across two parts stays unread"


def test_the_same_bytes_under_a_credential_named_parent_are_another_fact():
    element = {"note": "plain example value 12345"}
    payload = {"type": "object", "properties": {"api_key": {"type": "array", "examples": [element]}},
               "messages": [element]}
    facts: dict = {}
    parts = obs.redact_projection(payload, facts=facts)
    whole = obs.redact_projection(payload)
    assert _json_bytes(parts.value) == _json_bytes(whole.value) and parts.manifest() == whole.manifest()
    assert parts.value["messages"] == [element], "public where its parent is ordinary"
    assert parts.value["properties"]["api_key"]["examples"][0]["note"].startswith("***REDACTED["), "masked here"
    assert len({key[0] for key in facts}) == 1 and {key[1] for key in facts} == {False, True}


def test_changed_bytes_are_worked_out_again():
    payload = _view(_transcript(2))
    facts: dict = {}
    obs.redact_projection(payload, facts=facts)
    known = len(facts)
    edited = copy.deepcopy(payload)
    edited["messages"][1]["content"] += "!"  # send_messages is the same list: one changed part
    result, scanned = _scanned(edited, facts)
    assert len(facts) == known + 1 and scanned == len(_json_bytes(edited["messages"][1]))
    assert _json_bytes(result.value) == _json_bytes(obs.redact_projection(edited).value)


@pytest.mark.parametrize("names", [["api_key", "password"], ["password", "api_key"]])
def test_a_schema_in_another_key_order_keeps_the_whole_walk_s_records(names):
    """A part's fact is held per its bytes in their own order: a schema with secret defaults, met first in one order of
    its properties and then in the other, is reported in the order of the walk that met it, as the whole walk does."""
    def request(order):
        properties = {name: {"type": "string", "default": "sk-" + name * 6} for name in order}
        return {"tools": [{"type": "function", "function": {"name": "connect", "parameters": {"type": "object",
                                                                                              "properties": properties}}}]}

    facts: dict = {}
    obs.redact_projection(request(list(reversed(names))), facts=facts)  # the other order is met first
    current = request(names)
    reused, whole = obs.redact_projection(current, facts=facts), obs.redact_projection(current)
    assert reused.value == whole.value and reused.manifest() == whole.manifest()


@pytest.mark.parametrize("odd", ["mixed", object()], ids=["json_value", "no_json_form"])
def test_a_part_with_keys_of_mixed_types_is_redacted_as_the_whole_walk(tmp_path, odd):
    """A part whose keys have no sorted form, even with a value that has no JSON form at all, is walked as the whole
    is: the redaction by parts never serializes it to count, and writing its request record raises nothing."""
    payload = {"messages": [{"role": "user", "content": "plain", 1: odd, "token": "sk-" + "a" * 40}], "tools": []}
    facts: dict = {}
    assert obs.redact_projection(payload, facts=facts).manifest() == obs.redact_projection(payload).manifest()
    assert obs.redact_projection(payload, facts=facts).value == obs.redact_projection(payload).value
    assert obs.persist_call(tmp_path, task_id="odd", call_id="c", call_type="llm_request", payload=payload)


@pytest.mark.parametrize("count", [2, 80])
def test_shared_publication_prepares_directories_once_per_operation(tmp_path, monkeypatch, count):
    """Actual mkdir/chmod traffic stays constant; existing files are still checked and moded."""
    mkdir, chmod = pathlib.Path.mkdir, obs.os.chmod
    directories, modes = [], []

    def make_directory(path, *args, **kwargs):
        directories.append(path)
        return mkdir(path, *args, **kwargs)

    def private_mode(path, mode, *args, **kwargs):
        modes.append((pathlib.Path(path), mode))
        return chmod(path, mode, *args, **kwargs)

    # Precreate the test root so recursive mkdir setup does not enter the comparison.
    (tmp_path / "observability" / "blobs").mkdir(parents=True)
    monkeypatch.setattr(pathlib.Path, "mkdir", make_directory)
    monkeypatch.setattr(obs.os, "chmod", private_mode)
    payload = {"messages": [{"role": "user", "content": f"message {i} Ω"} for i in range(count)]}
    for _ in range(2):
        directories.clear()
        modes.clear()
        parts = {}
        ref = obs.write_shared_blob(tmp_path, payload, parts)
        assert len(directories) == 2, directories
        assert sum(mode == obs._PRIVATE_DIR_MODE for _path, mode in modes) == 2
        assert len(parts) == count
        assert all(any(path.name == f"{digest}.json.gz" and mode == obs._PRIVATE_FILE_MODE
                       for path, mode in modes) for digest in parts)
        assert packs.blob_bytes(tmp_path, ref) == obs._json_bytes(payload)


def test_part_batches_prepare_each_root_and_count_only_new_bytes(tmp_path):
    from ouroboros.context_timing import measured_sections, section_spans

    raw = obs._json_bytes({"content": "same bytes Ω"})
    digest = hashlib.sha256(raw).hexdigest()
    for root in (tmp_path / "one", tmp_path / "two"):
        with measured_sections():
            ref = obs.write_parts(root, {digest: raw})[digest]
            assert section_spans()["work"]["new_parts"] == len(raw)
        assert packs.blob_bytes(root, ref) == raw
        with measured_sections():
            assert obs.write_parts(root, {digest: raw})[digest] == ref
            assert section_spans()["work"].get("new_parts", 0) == 0
    assert obs.write_parts(tmp_path / "unused", {}) == {}
    assert not (tmp_path / "unused").exists()
