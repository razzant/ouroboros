"""Claim adoption and its receipts through the ordinary knowledge tools."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

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
    read_record = knowledge_versions.utils.json_snapshot

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a start already has the inventory and needs no history")

    def record(path, **kwargs):
        reads.append(path)
        return read_record(path, **kwargs)

    with monkeypatch.context() as spy:
        spy.setattr(knowledge, "inventory_knowledge", forbidden)
        spy.setattr(knowledge_versions, "_journal_lines", forbidden)
        spy.setattr(knowledge_versions.utils, "json_snapshot", record)
        assert knowledge.active_claims(tmp_path, inventory=rows) == {"room:1": [target]}
    assert reads == [knowledge_versions.claim_record_path(target.shelf, target.topic)]
    archived = _write(registry, target, "", mode="archive", reason="For reference", expected_revision=earlier.revision)
    assert archived.startswith("✅") and knowledge.active_claims(tmp_path) == {}
    current = knowledge.read_knowledge_note(target)
    assert _write(registry, target, "", mode="restore", expected_revision=current.revision).startswith("✅")
    assert knowledge.active_claims(tmp_path) == {"room:1": [target]}


def test_inventory_never_uses_a_global_record_for_its_project_namesake(tmp_path):
    global_note = knowledge.resolve_knowledge_address(tmp_path, "nova", "global")
    project_note = knowledge.resolve_knowledge_address(tmp_path, "nova", "project:river")
    registry = _registry(tmp_path)
    for note in (global_note, project_note):
        assert _write(registry, note, _markdown(["room:1"])).startswith("✅")
    global_rows = knowledge.inventory_knowledge(global_note, navigation=True)
    project_rows = knowledge.inventory_knowledge(project_note, navigation=True)
    assert project_rows[0]["claims"] == global_rows[0]["claims"] == ["room:1"]
    assert knowledge.active_claims(tmp_path, inventory=project_rows) == {}
    assert knowledge.active_claims(tmp_path, inventory=(*project_rows, *global_rows)) == {"room:1": [global_note]}


def test_claim_records_use_the_shelf_spelling_and_advertise_that_versions_address(tmp_path):
    canonical = knowledge.resolve_knowledge_address(tmp_path, "Field/CaseNote", "global")
    alias = knowledge.resolve_knowledge_address(tmp_path, "field/casenote", "global")
    written = knowledge.write_knowledge_note(canonical, _markdown(["room:1"]))
    assert written.ok and knowledge.active_claims(tmp_path) == {}
    registry = _registry(tmp_path)
    case_alias = alias.path.exists()
    assert _write(registry, alias, _markdown(["room:1"], "Revised case note.\n"),
                  expected_revision=written.current.revision if case_alias else "").startswith("✅")
    target, other = (canonical, alias) if case_alias else (alias, canonical)
    assert knowledge.active_claims(tmp_path) == {"room:1": [target]}
    assert not knowledge_versions.claim_record_path(other.shelf, other.topic).exists()
    assert knowledge_versions.claim_record_path(target.shelf, target.topic).is_file()

    revision = knowledge.read_knowledge_note(alias).revision
    pinned = registry.execute_result("knowledge_read", {"topic": alias.topic, "scope": "global", "version": revision})
    reader = pinned.meta["knowledge_source"]["read"]
    assert reader["arguments"]["topic"] == target.topic
    target.path.rename(target.path.with_name("Moved.md"))
    exact = registry.execute_result(reader["tool"], reader["arguments"])
    assert exact.status == "ok" and exact.text.endswith("Revised case note.\n")


def test_a_byte_identical_move_keeps_claims_and_rekeys_only_on_write(tmp_path, monkeypatch):
    registry = _registry(tmp_path)
    old = knowledge.resolve_knowledge_address(tmp_path, "nova", "global")
    moved = knowledge.resolve_knowledge_address(tmp_path, "renamed", "global")
    unrelated = knowledge.resolve_knowledge_address(tmp_path, "unrelated", "global")
    raw = _markdown(["room:1", "place:handwritten"])
    assert _write(registry, old, _markdown(["room:1"])).startswith("✅")
    old.path.write_bytes(raw.encode())
    assert _write(registry, old, "Latest authored body.\n", mode="append").startswith("✅")
    old.path.rename(moved.path)
    unrelated.path.write_bytes(raw.encode())
    old_record = knowledge_versions.claim_record_path(old.shelf, old.topic)
    new_record = knowledge_versions.claim_record_path(moved.shelf, moved.topic)
    before = old_record.read_bytes()
    rows = knowledge.inventory_knowledge(moved, navigation=True)

    def forbidden(*_args, **_kwargs):
        pytest.fail("claim lookup must use the supplied inventory and projections")

    with monkeypatch.context() as spy:
        spy.setattr(knowledge, "inventory_knowledge", forbidden)
        spy.setattr(knowledge_versions, "_journal_lines", forbidden)
        assert knowledge.active_claims(tmp_path, inventory=rows) == {"room:1": [moved]}
    assert old_record.read_bytes() == before and not new_record.exists()

    # A different source at the old topic uses only the record names still in its YAML.
    old.path.write_bytes(raw.encode())
    assert knowledge.active_claims(tmp_path) == {"room:1": [old, moved]}
    original = knowledge.read_knowledge_note(moved)
    same_bytes = _write(registry, moved, "", mode="append")
    assert "unchanged" in same_bytes and not old_record.exists()
    assert json.loads(new_record.read_bytes())["revision"] == original.revision
    assert knowledge.active_claims(tmp_path) == {"room:1": [moved]}
    new_record.unlink()
    with knowledge.knowledge_write_lock(moved.shelf):
        knowledge_versions.rebuild_claim_record(moved)
        knowledge_versions.rebuild_claim_record(old)
    assert not old_record.exists() and knowledge.active_claims(tmp_path) == {"room:1": [moved]}


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


@pytest.mark.parametrize(("prefix", "insertion", "suffix"), [
    ("---\r\ntype: 'note' # май\r\nclaims: # owned\r\n  - 'room:1' # keep\r\n",
     '  - "place:carried"\r\n', "summary: 'keep me'\r\n...\r\n# Body\r\n"),
    ("---\r\ntype: 'note' # keep\r\nclaims: ['room:1', # authored punctuation\r\n  ",
     ' "place:carried"', "]\r\n---\r\n# Body\r\n"),
    ("---\r\ntype: 'note' # keep\r\nsummary: 'important'\r\n",
     'claims: ["place:carried"]\r\n', "---\r\n# Body\r\n"),
    ("---\r\ntype: 'note' # keep\r\nclaims: ['room:1', 'place:carried']\r\n",
     "", "---\r\n# Body\r\n"),
    ("", '---\r\nclaims: ["place:carried"]\r\n---\r\n', "# Plain Markdown\r\n\r\nNo preamble.\r\n"),
], ids=["block", "flow", "absent", "already-present", "plain"])
def test_host_claim_splices_only_its_name_in_source_and_by_name_readers(tmp_path, prefix, insertion, suffix):
    from ouroboros.improvement_backlog import authored_backlog_text
    from ouroboros.memory_fallback import _remembering
    from ouroboros.presence_context import focus_topics

    raw, expected = (prefix + suffix).encode(), (prefix + insertion + suffix).encode()
    for topic in ("improvement-backlog", "remembering"):
        target = knowledge.resolve_knowledge_address(tmp_path, topic, "global")
        target.path.parent.mkdir(parents=True, exist_ok=True)
        target.path.write_bytes(raw)
        host = knowledge.write_host_claim(target, "place:carried", expected_revision=knowledge.read_knowledge_note(target).revision,
                                          reason="Binding migration")
        assert host.ok and target.path.read_bytes() == expected
        assert target in knowledge.active_claims(tmp_path)["place:carried"]
        assert "room:1" not in knowledge.active_claims(tmp_path)
        bound = focus_topics(tmp_path, {"context_topics": [topic]})
        assert bound == f"### Knowledge topic: {topic}\n\n" + expected.decode().replace("\r\n", "\n")
    assert authored_backlog_text(tmp_path).split("\n\n", 1)[1].encode() == expected
    assert _remembering(tmp_path) == expected.decode().strip()


@pytest.mark.parametrize("preamble", [
    "claims: room:1\n",
    "source: &names ['room:1']\nclaims: *names\n",
    "claims: &names ['room:1']\nshared: *names\n",
], ids=["scalar", "alias", "shared-anchor"])
def test_host_claim_keeps_an_unextendable_note_pending_with_no_write(tmp_path, preamble):
    target = knowledge.resolve_knowledge_address(tmp_path, "bound", "global")
    target.path.parent.mkdir(parents=True)
    unavailable = f"---\n{preamble}---\nOriginal body.\n".encode()
    target.path.write_bytes(unavailable)

    def add():
        return knowledge.write_host_claim(target, "place:carried", reason="Binding migration",
                                          expected_revision=knowledge.read_knowledge_note(target).revision)

    result = add()
    assert not result.ok and result.reason.startswith("host_claim_pending: ")
    assert target.path.read_bytes() == unavailable and not knowledge_versions.journal_path(target).exists()
    assert knowledge.active_claims(tmp_path) == {}
    target.path.write_bytes(b"---\nclaims: ['room:1']\n---\nOriginal body.\n")
    assert add().ok and knowledge.active_claims(tmp_path) == {"place:carried": [target]}


def test_revert_keeps_the_continuously_present_claim_after_an_outside_removal(tmp_path):
    registry = _registry(tmp_path)
    target = knowledge.resolve_knowledge_address(tmp_path, "nova", "global")
    names = ["room:1", "place:continues"]
    assert _write(registry, target, _markdown(names)).startswith("✅")
    assert knowledge.active_claims(tmp_path) == dict.fromkeys(names, [target])
    first = knowledge.read_knowledge_note(target)
    target.path.write_bytes(_markdown(names[1:], "Outside edit.\n").encode())
    current = knowledge.read_knowledge_note(target)
    assert _write(registry, target, "", mode="revert", version=first.revision, expected_revision=current.revision).startswith("✅")
    assert target.path.read_bytes() == first.raw
    assert knowledge.active_claims(tmp_path) == {names[1]: [target]}


def test_revert_between_explicit_claim_lists_carries_only_their_intersection(tmp_path):
    registry = _registry(tmp_path)
    target = knowledge.resolve_knowledge_address(tmp_path, "nova", "global")
    assert _write(registry, target, _markdown(["room:1", "place:first"])).startswith("✅")
    first = knowledge.read_knowledge_note(target)
    assert _write(registry, target, _markdown(["room:1", "place:second"]), expected_revision=first.revision).startswith("✅")
    current = knowledge.read_knowledge_note(target)
    assert knowledge.active_claims(tmp_path) == {"room:1": [target], "place:second": [target]}
    reverted = _write(registry, target, "", mode="revert", version=first.revision, expected_revision=current.revision)
    assert reverted.startswith("✅") and target.path.read_bytes() == first.raw
    assert "Present situations holding this note: room:1" in reverted and "Removing a claim clears no arrival" in reverted
    assert knowledge.active_claims(tmp_path) == {"room:1": [target]}


def test_revert_reports_a_change_between_typed_nonstring_claim_entries(tmp_path):
    registry = _registry(tmp_path)
    target = knowledge.resolve_knowledge_address(tmp_path, "nova", "global")
    assert _write(registry, target, _markdown(["room:1", True])).startswith("✅")
    first = knowledge.read_knowledge_note(target)
    assert _write(registry, target, _markdown(["room:1", 1]), expected_revision=first.revision).startswith("✅")
    append = _write(registry, target, "Unrelated body line.\n", mode="append")
    assert append.startswith("✅") and "Ignored claims" not in append and "holding this note" not in append
    current = knowledge.read_knowledge_note(target)
    reverted = _write(registry, target, "", mode="revert", version=first.revision, expected_revision=current.revision)
    assert reverted.startswith("✅") and target.path.read_bytes() == first.raw
    assert "Ignored claims entries (expected a list of strings): True" in reverted
    assert "Present situations holding this note: room:1" in reverted


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


def test_authored_history_and_return_facts_count_claimed_writes_once(tmp_path, monkeypatch):
    from ouroboros import return_facts
    from ouroboros.projects_registry import create_project

    project = create_project(tmp_path, "writer", name="Writer")
    registry = _registry(tmp_path, current_chat_id=project["chat_id"])
    target = knowledge.resolve_knowledge_address(tmp_path, "nova", "global")
    journal = knowledge_versions.journal_path(target)

    def stamp(index):
        return f"2025-11-18T20:00:{index:02}+00:00"

    def owner_word(index):
        identity = f"word-{index}"
        dialogue_store.append(tmp_path, "chat", {"ts": stamp(index), "direction": "in", "chat_id": 1,
                                                "user_id": 1, "text": identity, "source": "web",
                                                "client_message_id": identity, "ingress_accepted": True})
        return dialogue_store.accepted_row(tmp_path, 1, identity)

    def counts(index, expected):
        facts = return_facts.facts(tmp_path, owner_word(index), return_facts.Own(room=1))
        full = return_facts.render(facts)[0]
        assert f"- knowledge global/nova: {expected} {'write' if expected == 1 else 'writes'} by" in full

    def edit(index, content, **kwargs):
        monkeypatch.setattr(knowledge, "utc_now_iso", lambda: stamp(index))
        assert _write(registry, target, content, **kwargs).startswith("✅")

    def history(**kwargs):
        result = json.loads(registry.execute("query_data", {"kind": "knowledge_history", "op": "jsonl", **kwargs}))
        assert result["status"] == "ok"
        return result["data"]

    owner_word(0)
    edit(1, _markdown(["room:1"]), reason="Adopt the claim")
    counts(2, 1)
    [first] = history(tail=1)
    assert first["reason"] == "Adopt the claim" and first["delta"]["body_changed"] is True
    edit(3, "Second.\n", mode="append", reason="Second write")
    edit(4, "Third.\n", mode="append", reason="Third write")
    counts(5, 2)
    assert [row["reason"] for row in history(tail=2)] == ["Second write", "Third write"]
    edit(6, _markdown([]), expected_revision=knowledge.read_knowledge_note(target).revision, reason="Remove the claim")
    counts(7, 1)
    assert [row["reason"] for row in history()] == ["Adopt the claim", "Second write", "Third write", "Remove the claim"]
    assert history(filter={"field": "publication", "op": "eq", "value": "claims_published"}) == []
    assert json.loads(journal.read_bytes().splitlines()[-1])["publication"] == "claims_published"
