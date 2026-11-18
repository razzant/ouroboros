"""Claim adoption and its receipts through the ordinary knowledge tools."""
from __future__ import annotations

import json
from pathlib import Path

from ouroboros import dialogue_store, knowledge, knowledge_versions
from ouroboros.places import connect_place, conversation_chat_id
from ouroboros.presence_bindings import conversation_key
from ouroboros.tools.registry import ToolContext, ToolRegistry


def _registry(root: Path, **kwargs) -> ToolRegistry:
    registry = ToolRegistry(repo_dir=root, drive_root=root)
    registry.set_context(ToolContext(repo_dir=root, drive_root=root, task_id="claim-writer", **kwargs))
    return registry


def _write(registry, address, content, **kwargs):
    return registry.execute("knowledge_write", {"topic": address.topic, "scope": address.scope,
                                                "mode": "overwrite", "content": content, **kwargs})


def _markdown(claims, body="Project notes.\n"):
    return "---\ntype: note\nclaims: " + json.dumps(claims) + "\n---\n" + body


def test_only_recorded_claims_hold_a_note_and_the_start_reads_just_its_inventory_and_record(tmp_path, monkeypatch):
    registry = _registry(tmp_path)
    ordinary = knowledge.resolve_knowledge_address(tmp_path, "ordinary", "global")
    reply = _write(registry, ordinary, "Ordinary note.\n")
    assert reply.startswith("✅") and "This situation's names: room:1 (Main)" in reply
    assert "holding this note" not in reply and "Claims with no matching" not in reply
    assert not knowledge_versions.claim_record_path(ordinary.shelf, ordinary.topic).parent.exists()
    assert knowledge.active_claims(tmp_path) == {}

    target = knowledge.resolve_knowledge_address(tmp_path, "notes/nova", "global")
    raw = _markdown(["room:1"])
    target.path.parent.mkdir()
    target.path.write_bytes(raw.encode())
    assert knowledge.active_claims(tmp_path) == {}
    earlier = knowledge.read_knowledge_note(target)
    assert knowledge.write_knowledge_note(target, raw, expected_revision=earlier.revision).ok
    assert knowledge.active_claims(tmp_path) == {}
    receipt = _write(registry, target, raw, expected_revision=earlier.revision)
    assert receipt.startswith("✅") and target.path.read_bytes() == earlier.raw
    assert "Present situations holding this note: room:1" in receipt
    history = list(knowledge_versions.NoteHistory(target))
    assert history[0][1]["active_claims"] == ["room:1"]
    assert history[0][1]["old_sha256"] == history[0][1]["new_sha256"] == earlier.revision

    rows = knowledge.inventory_knowledge(target, navigation=True)
    reads = []
    read_record = knowledge_versions.read_claim_record_names

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a start already has the inventory and needs no history")

    def record(path):
        reads.append(path)
        return read_record(path)

    with monkeypatch.context() as spy:
        spy.setattr(knowledge, "inventory_knowledge", forbidden)
        spy.setattr(knowledge_versions, "NoteHistory", forbidden)
        spy.setattr(knowledge_versions, "read_claim_record_names", record)
        assert knowledge.active_claims(tmp_path, inventory=rows) == {"room:1": [target]}
    assert reads == [knowledge_versions.claim_record_path(target.shelf, target.topic)]
    archived = _write(registry, target, "", mode="archive", reason="For reference", expected_revision=earlier.revision)
    assert archived.startswith("✅") and knowledge.active_claims(tmp_path) == {}
    current = knowledge.read_knowledge_note(target)
    assert _write(registry, target, "", mode="restore", expected_revision=current.revision).startswith("✅")
    assert knowledge.active_claims(tmp_path) == {"room:1": [target]}


def test_body_only_append_keeps_recorded_names_without_adopting_names_from_yaml(tmp_path):
    registry = _registry(tmp_path)
    target = knowledge.resolve_knowledge_address(tmp_path, "nova", "global")
    raw = _markdown(["room:1", "place:unrecorded"])
    target.path.parent.mkdir(parents=True)
    target.path.write_bytes(raw.encode())
    assert _write(registry, target, "First body append.", mode="append").startswith("✅")
    assert knowledge.active_claims(tmp_path) == {}

    before = knowledge.read_knowledge_note(target)
    assert _write(registry, target, "Body edit.", mode="edit", old_str="Project notes.",
                  expected_revision=before.revision).startswith("✅")
    assert knowledge.active_claims(tmp_path) == {}

    earlier = knowledge.read_knowledge_note(target)
    assert _write(registry, target, _markdown(["room:1"]), expected_revision=earlier.revision).startswith("✅")
    target.path.write_bytes(raw.encode())  # a second name arrives outside the writer
    receipt = _write(registry, target, "Second body append.", mode="append")
    assert receipt.startswith("✅") and "Claims with no matching" not in receipt
    assert knowledge.active_claims(tmp_path) == {"room:1": [target]}
    assert next(iter(knowledge_versions.NoteHistory(target)))[1]["active_claims"] == ["room:1"]
    assert knowledge.read_knowledge_note(target).metadata["claims"] == ["room:1", "place:unrecorded"]

    target.path.write_bytes(_markdown(["place:unrecorded"]).encode())
    assert knowledge.active_claims(tmp_path) == {}
    removed = _write(registry, target, "Third body append.", mode="append")
    assert "Removing a claim clears no arrival, releases no mark, and withdraws no doubt." in removed
    assert next(iter(knowledge_versions.NoteHistory(target)))[1]["active_claims"] == []
    target.path.write_bytes(raw.encode())
    assert knowledge.active_claims(tmp_path) == {}


def test_a_failed_claim_capture_cannot_be_rebuilt_as_a_publication(tmp_path, monkeypatch):
    registry = _registry(tmp_path)
    target = knowledge.resolve_knowledge_address(tmp_path, "nova", "global")
    raw = _markdown(["room:1"])
    target.path.parent.mkdir(parents=True)
    target.path.write_bytes(raw.encode())
    revision = knowledge.read_knowledge_note(target).revision

    def index_unavailable(_address):
        raise OSError("index could not be replaced")

    with monkeypatch.context() as failure:
        failure.setattr(knowledge, "rebuild_knowledge_index", index_unavailable)
        assert "publication_incomplete" in _write(registry, target, raw, expected_revision=revision)
    assert next(iter(knowledge_versions.NoteHistory(target)))[1]["active_claims"] == ["room:1"]
    with knowledge.knowledge_write_lock(target.shelf):
        knowledge_versions.rebuild_claim_record(target)
    assert knowledge.active_claims(tmp_path) == {}

    assert _write(registry, target, raw, expected_revision=revision).startswith("✅")
    record = knowledge_versions.claim_record_path(target.shelf, target.topic)
    record.unlink()
    assert knowledge.active_claims(tmp_path) == {}
    with knowledge.knowledge_write_lock(target.shelf):
        knowledge_versions.rebuild_claim_record(target)
    assert knowledge.active_claims(tmp_path) == {"room:1": [target]}


def test_a_later_write_restores_a_committed_claim_after_projection_failure(tmp_path, monkeypatch):
    registry = _registry(tmp_path)
    target = knowledge.resolve_knowledge_address(tmp_path, "nova", "global")

    def projection_unavailable(*_args):
        raise OSError("record could not be replaced")

    with monkeypatch.context() as failure:
        failure.setattr(knowledge_versions, "_project_claim_record", projection_unavailable)
        assert "publication_incomplete" in _write(registry, target, _markdown(["room:1"]))
    assert knowledge.active_claims(tmp_path) == {}
    assert _write(registry, target, "An ordinary append.\n", mode="append").startswith("✅")
    assert knowledge.active_claims(tmp_path) == {"room:1": [target]}


def test_host_claim_adds_only_its_name_and_seed_is_create_only(tmp_path):
    registry = _registry(tmp_path)
    target = knowledge.resolve_knowledge_address(tmp_path, "nova", "global")
    assert _write(registry, target, _markdown(["room:1"])).startswith("✅")
    target.path.write_bytes(_markdown(["room:1", "place:handwritten"]).encode())
    before = knowledge.read_knowledge_note(target)
    refusal = knowledge.write_host_claim(target, "place:carried", expected_revision="", reason="Binding migration")
    assert refusal.reason == "revision_conflict" and target.path.read_bytes() == before.raw
    added = knowledge.write_host_claim(target, "place:carried", expected_revision=before.revision, reason="Binding migration")
    assert added.ok and added.current.metadata["claims"] == ["room:1", "place:handwritten", "place:carried"]
    assert added.current.source.text_at(added.current.source.body_span) == before.source.text_at(before.source.body_span)
    assert knowledge.active_claims(tmp_path) == {"room:1": [target], "place:carried": [target]}
    change = next(iter(knowledge_versions.NoteHistory(target)))[1]
    assert (change["writer"], change["reason"], change["old_sha256"], change["new_sha256"]) == (
        "host", "Binding migration", before.revision, added.current.revision)

    seed = knowledge.resolve_knowledge_address(tmp_path, "seed", "global")
    result = knowledge.write_host_claim(seed, "wake", expected_revision="", reason="First Start", content="Seed body.\n")
    assert result.ok and result.current.text.endswith("Seed body.\n")
    assert knowledge.active_claims(tmp_path)["wake"] == [seed]
    assert knowledge.write_host_claim(seed, "wake", expected_revision="", reason="First Start").reason == "revision_conflict"


def test_claim_receipt_reports_unmatched_names_and_nonstrings_and_project_claims_hold_nowhere(tmp_path):
    registry = _registry(tmp_path)
    place = connect_place(tmp_path, connector="fixture", provider="chat", account_id="T", conversation_id="C",
                          scope="conversation", place_id="team/room")
    target = knowledge.resolve_knowledge_address(tmp_path, "nova", "global")
    route = {"provider": "chat", "account_id": "T", "conversation_id": "C", "thread_id": "inbox/1"}
    chat = conversation_chat_id(conversation_key(**route))
    dialogue_store.append(tmp_path, "chat", {"type": "place_arrival", "chat_id": chat, "direction": "in",
                                           "place_id": place.place_id, "transport": route, "text": "An arrival"})
    matching = ["room:1", "place:team%2Froom", "conversation:chat/T/C", f"room:{chat}", "thread:chat/T/C/inbox%2F1"]
    unmatched = ["room:12345679", "place:unconnected", "place:team/room", " room:1", "Room:1", "thread:chat/T/C/missing"]
    receipt = _write(registry, target, _markdown([*matching, *unmatched, 19, False, None]))
    assert receipt.startswith("✅") and place.place_id == "team/room"
    assert f"Present situations holding this note: {', '.join(matching)}" in receipt
    assert "Ignored claims entries (expected a list of strings): 19, False, None" in receipt
    line = next(line for line in receipt.splitlines() if line.startswith("Claims with no matching"))
    assert line == "Claims with no matching situation found: " + ", ".join(repr(name) for name in unmatched)
    assert "Removing a claim" not in receipt

    project = knowledge.resolve_knowledge_address(tmp_path, "project-only", "project:nova")
    local = _write(registry, project, _markdown(matching))
    assert local.startswith("✅") and "Claims on a project-shelf note hold nowhere." in local
    assert "Claims with no matching" not in local and "Present situations holding this note" not in local
    assert knowledge.active_claims(tmp_path)["room:1"] == [target]
    assert not knowledge_versions.claim_record_path(project.shelf, project.topic).exists()

    current = knowledge.read_knowledge_note(target)
    cleared = _write(registry, target, _markdown([]), expected_revision=current.revision)
    assert "Present situations holding this note: none" in cleared
    assert "Removing a claim clears no arrival, releases no mark, and withdraws no doubt." in cleared
    assert "Claims with no matching" not in cleared and knowledge.active_claims(tmp_path) == {}
