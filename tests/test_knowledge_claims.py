"""Claim adoption and its receipts through the ordinary knowledge tools."""
from __future__ import annotations

import hashlib
import json
import re
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


REMOVED = ("A removed name stops holding this head in works that start from now on; a work already running keeps "
           "the heads it started with.")


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


@pytest.mark.parametrize("copy", [False, True], ids=["move", "copy"])
def test_a_new_address_holds_nothing_until_an_explicit_write_there(tmp_path, copy):
    registry = _registry(tmp_path)
    old = knowledge.resolve_knowledge_address(tmp_path, "nova", "global")
    moved = knowledge.resolve_knowledge_address(tmp_path, "renamed", "global")
    assert _write(registry, old, _markdown(["room:1"])).startswith("✅")
    assert knowledge.active_claims(tmp_path) == {"room:1": [old]}
    original = knowledge.read_knowledge_note(old)
    old_record = knowledge_versions.claim_record_path(old.shelf, old.topic)
    new_record = knowledge_versions.claim_record_path(moved.shelf, moved.topic)
    before = old_record.read_bytes()
    if copy:
        moved.path.write_bytes(original.raw)
    else:
        old.path.rename(moved.path)
    before_adoption = {"room:1": [old]} if copy else {}
    assert knowledge.active_claims(tmp_path) == before_adoption
    assert "unchanged" in _write(registry, moved, "", mode="append")
    assert not new_record.exists() and knowledge.active_claims(tmp_path) == before_adoption
    assert _write(registry, moved, "Later body line.\n", mode="append").startswith("✅")
    assert not new_record.exists() and knowledge.active_claims(tmp_path) == before_adoption

    current = knowledge.read_knowledge_note(moved)
    adoption = _write(registry, moved, current.text, expected_revision=current.revision)
    assert "Present situations holding this note: room:1" in adoption
    assert moved.path.read_bytes() == current.raw and new_record.is_file()
    assert knowledge.active_claims(tmp_path) == {"room:1": [old, moved] if copy else [moved]}
    assert old_record.read_bytes() == before
    pin = registry.execute_result("knowledge_read", {"topic": old.topic, "scope": "global", "version": original.revision})
    assert pin.status == "ok" and pin.text.endswith(original.text)


def test_a_claimless_append_can_use_a_known_file_when_its_directory_cannot_be_listed(tmp_path, monkeypatch):
    target = knowledge.resolve_knowledge_address(tmp_path, "restricted/known", "global")
    first = knowledge.write_knowledge_note(target, "Known text.\n").current
    journal = knowledge_versions.journal_path(target)
    before = journal.read_bytes()
    listdir = knowledge.os.listdir

    def restricted(path):
        if path == target.path.parent:
            raise PermissionError("directory has write and search permission only")
        return listdir(path)

    with monkeypatch.context() as patch:
        patch.setattr(knowledge.os, "listdir", restricted)
        without_listing = knowledge.write_knowledge_note(target, "", mode="append")
    with_listing = knowledge.write_knowledge_note(target, "", mode="append")
    assert with_listing == without_listing
    assert without_listing.ok and without_listing.reason == "unchanged" and without_listing.current.raw == first.raw
    assert journal.read_bytes() == before


def test_body_only_append_keeps_recorded_names_without_adopting_names_from_yaml(tmp_path):
    registry = _registry(tmp_path)
    target = knowledge.resolve_knowledge_address(tmp_path, "nova", "global")
    raw = _markdown(["room:1", "place:unrecorded"])
    target.path.parent.mkdir(parents=True)
    target.path.write_bytes(raw.encode())
    plain = knowledge.resolve_knowledge_address(tmp_path, "plain", "global")
    plain.path.write_bytes(b"---\ntype: note\n---\nProject notes.\n")

    def newest(address=target):
        return next(iter(knowledge_versions.NoteHistory(address)))[1]

    assert _write(registry, target, "First body append.", mode="append").startswith("✅")
    assert _write(registry, plain, "First body append.", mode="append").startswith("✅")
    assert knowledge.active_claims(tmp_path) == {}
    # A YAML field nobody adopted adds nothing to the write's row: the fields of a note that never had one.
    assert "active_claims" not in newest() and set(newest()) == set(newest(plain))

    before = knowledge.read_knowledge_note(target)
    assert _write(registry, target, "Body edit.", mode="edit", old_str="Project notes.",
                  expected_revision=before.revision).startswith("✅")
    assert knowledge.active_claims(tmp_path) == {} and "active_claims" not in newest()

    earlier = knowledge.read_knowledge_note(target)
    assert _write(registry, target, _markdown(["room:1"]), expected_revision=earlier.revision).startswith("✅")
    assert newest()["active_claims"] == ["room:1"]
    target.path.write_bytes(raw.encode())  # a second name arrives outside the writer
    receipt = _write(registry, target, "Second body append.", mode="append")
    assert receipt.startswith("✅") and "Claims with no matching" not in receipt
    assert knowledge.active_claims(tmp_path) == {"room:1": [target]}
    assert newest()["active_claims"] == ["room:1"]
    assert knowledge.read_knowledge_note(target).metadata["claims"] == ["room:1", "place:unrecorded"]

    target.path.write_bytes(_markdown(["place:unrecorded"]).encode())
    assert knowledge.active_claims(tmp_path) == {}
    removed = _write(registry, target, "Third body append.", mode="append")
    assert REMOVED in removed and newest()["active_claims"] == []  # the removal is the row's fact
    target.path.write_bytes(raw.encode())
    assert knowledge.active_claims(tmp_path) == {}
    later = _write(registry, target, "Fourth body append.", mode="append")
    assert later.startswith("✅") and REMOVED not in later and "active_claims" not in newest()


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
    assert "Present situations holding this note: room:1" in reverted and REMOVED in reverted
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
    assert "A removed name" not in receipt

    project = knowledge.resolve_knowledge_address(tmp_path, "project-only", "project:nova")
    local = _write(registry, project, _markdown(matching))
    assert local.startswith("✅") and "Claims on a project-shelf note hold nowhere." in local
    assert "Claims with no matching" not in local and "Present situations holding this note" not in local
    assert knowledge.active_claims(tmp_path)["room:1"] == [target]
    assert not knowledge_versions.claim_record_path(project.shelf, project.topic).exists()

    current = knowledge.read_knowledge_note(target)
    cleared = _write(registry, target, _markdown([]), expected_revision=current.revision)
    assert "Present situations holding this note: none" in cleared
    assert REMOVED in cleared
    assert "Claims with no matching" not in cleared and knowledge.active_claims(tmp_path) == {}


def test_a_claims_overwrite_that_carries_no_body_is_refused_and_writes_nothing(tmp_path):
    """Adding a name is an overwrite, and an overwrite replaces the whole note. Claims sent alone would save the note
    without its body, so that one write is refused; every other write is what it was."""
    registry = _registry(tmp_path)
    target = knowledge.resolve_knowledge_address(tmp_path, "nova", "global")
    body = "What is true of Nova now.\n\n## History\n\nOlder facts.\n"
    assert _write(registry, target, _markdown(["room:1"], body)).startswith("✅")
    journal = knowledge_versions.journal_path(target)
    record = knowledge_versions.claim_record_path(target.shelf, target.topic)
    before = knowledge.read_knowledge_note(target)
    stored = target.path.read_bytes(), journal.read_bytes(), record.read_bytes()
    said = ("⚠️ TOOL_ERROR: Knowledge write not completed: invalid_note: this overwrite sets claims and carries no body, so it "
            f"would erase the note's body ({len(body)} characters); an overwrite replaces the whole note: send its whole text "
            "with the claims list in its YAML.")
    alone = "---\nclaims: ['room:1', 'room:2']\n---"
    for content in (alone, alone + "\n", alone + "\n\n  \n", "---\nclaims: []\n---\n"):
        refused = _write(registry, target, content, expected_revision=before.revision)
        assert refused.startswith(said) and refused.endswith(body)  # the reply carries the text to send back
        assert (target.path.read_bytes(), journal.read_bytes(), record.read_bytes()) == stored
    assert knowledge.active_claims(tmp_path) == {"room:1": [target]}

    # The same names with the whole text are saved, and the new name holds.
    saved = _write(registry, target, _markdown(["room:1", "room:2"], body), expected_revision=before.revision)
    assert saved.startswith("✅") and target.path.read_bytes().endswith(body.encode("utf-8"))
    assert knowledge.active_claims(tmp_path) == {"room:1": [target], "room:2": [target]}

    # An overwrite whose YAML sets no claims is untouched: its fields merge and the body it carries, none, is the note's.
    plain = knowledge.resolve_knowledge_address(tmp_path, "plain", "global")
    assert _write(registry, plain, "---\ntype: note\n---\nA body.\n").startswith("✅")
    fields = _write(registry, plain, "---\nsummary: Only fields.\n---",
                    expected_revision=knowledge.read_knowledge_note(plain).revision)
    assert fields.startswith("✅") and plain.path.read_bytes() == b"---\ntype: note\nsummary: Only fields.\n---\n"
    held = knowledge.read_knowledge_note(target)  # the same on a note that holds names, while its YAML sends none
    kept = _write(registry, target, "---\nsummary: Only fields.\n---", expected_revision=held.revision)
    assert kept.startswith("✅") and knowledge_versions.body_chars(knowledge.read_knowledge_note(target)) == 0
    assert knowledge.active_claims(tmp_path) == {"room:1": [target], "room:2": [target]}

    # A new note may be created with claims and no body.
    fresh = knowledge.resolve_knowledge_address(tmp_path, "fresh", "global")
    assert _write(registry, fresh, "---\nclaims: ['room:3']\n---").startswith("✅")
    assert knowledge.active_claims(tmp_path)["room:3"] == [fresh]


def test_a_write_of_a_held_note_says_how_large_its_head_is(tmp_path):
    """Where a name is present the note's head stands whole in every start, and a note with no `## ` line is all
    head. So each completed write of a note that holds names states the head's size and what ends it; a write of a
    note that holds none says nothing."""
    registry = _registry(tmp_path)

    def said(receipt):
        return [line for line in receipt.splitlines() if line.startswith("Held head:")]

    def write(address, content=None, **kwargs):
        revision = knowledge.read_knowledge_note(address).revision
        return registry.execute("knowledge_write", {"topic": address.topic, "scope": address.scope,
                                                    "expected_revision": revision, "content": content, **kwargs})

    # A head that a `## ` line ends: its size, and a later write below that line leaves it as it was.
    target = knowledge.resolve_knowledge_address(tmp_path, "nova", "global")
    head = "Fact. " * 250 + "\n\n"
    saved = _write(registry, target, _markdown(["room:1"], head + "## History\n\nOlder facts.\n"))
    ended = [f"Held head: {len(head):,} characters, the body above the first `## ` line."]
    assert saved.startswith("✅") and len(head) == 1_502 and said(saved) == ended
    assert said(write(target, "A later fact.\n", mode="append")) == ended

    # No `## ` line, a fenced one included: the whole body is the head, and an append grows it.
    whole = knowledge.resolve_knowledge_address(tmp_path, "tools", "global")
    body = "Whole. " * 200 + "\n\n```\n## not a heading\n```\n"
    created = _write(registry, whole, _markdown(["room:1"], body))
    assert said(created) == [f"Held head: {len(body):,} characters: this note has no `## ` line, "
                             "so its whole body is the head."]
    grown = write(whole, "More.\n", mode="append")
    assert said(grown) == [f"Held head: {len(body) + len('More.') + 1:,} characters: this note has no `## ` line, "
                           "so its whole body is the head."]

    # A note that holds no name says nothing: never adopted, on a project shelf, archived, cleared, or refused.
    plain = knowledge.resolve_knowledge_address(tmp_path, "plain", "global")
    assert _write(registry, plain, "---\ntype: note\n---\nA body.\n").startswith("✅")
    plain.path.write_bytes(_markdown(["room:1"], "A body.\n").encode("utf-8"))  # a field the writer never took
    unadopted = write(plain, "More.\n", mode="append")
    assert unadopted.startswith("✅") and not said(unadopted)
    project = knowledge.resolve_knowledge_address(tmp_path, "project-only", "project:nova")
    local = _write(registry, project, _markdown(["room:1"], body))
    assert local.startswith("✅") and not said(local)
    archived = write(whole, mode="archive", reason="Out of use.")
    assert archived.startswith("✅") and not said(archived)
    restored = write(whole, mode="restore")
    assert restored.startswith("✅") and said(restored) == said(grown)
    refused = write(whole, "---\nclaims: ['room:2']\n---", mode="overwrite")
    assert refused.startswith("⚠️") and not said(refused)
    cleared = write(whole, _markdown([], body), mode="overwrite")
    assert cleared.startswith("✅") and REMOVED in cleared and not said(cleared)


def test_history_readers_skip_only_claim_commits_and_keep_the_index_backup_row(tmp_path, monkeypatch):
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

    def pages(index):
        """Each page written since my human's previous word, to its count of writes."""
        facts = return_facts.facts(tmp_path, owner_word(index), return_facts.Own(room=1))
        lines = [line for line in return_facts.render(facts)[0].splitlines() if line.startswith("- knowledge ")]
        found = {match[1]: int(match[2]) for line in lines if (match := re.match(r"- knowledge (\S+): (\d+) writes? by ", line))}
        assert len(found) == len(lines)
        return found

    def edit(index, content, **kwargs):
        monkeypatch.setattr(knowledge, "utc_now_iso", lambda: stamp(index))
        assert _write(registry, target, content, **kwargs).startswith("✅")

    def history(**kwargs):
        result = json.loads(registry.execute("query_data", {"kind": "knowledge_history", "op": "jsonl", **kwargs}))
        assert result["status"] == "ok"
        return result["data"]

    owner_word(0)
    target.shelf.mkdir(parents=True)
    legacy = {"ts": stamp(1), "topic": target.topic, "mode": "overwrite", "reason": "Legacy write"}
    journal.write_bytes((json.dumps(legacy) + "\n").encode("utf-8"))
    assert pages(2) == {"global/nova": 1}
    assert history(tail=1) == [legacy]

    legacy_index = "# Authored index before upgrading\n"
    (target.shelf / knowledge.INDEX_FILE).write_bytes(legacy_index.encode("utf-8"))
    edit(3, _markdown(["room:1"]), reason="Adopt the claim")
    # The claim's commit is no write: the note counts once. The one backup of a hand-written index is a row of the
    # journal as before claims existed: both readers return it, its text left out.
    assert pages(4) == {"global/nova": 1, "global/index-full": 1}
    adopted, backup = history(tail=2)
    assert adopted["reason"] == "Adopt the claim" and adopted["delta"]["body_changed"] is True
    assert backup == {"ts": stamp(3), "type": "knowledge_index_source", "topic": "index-full", "scope": "global",
                      "old_sha256": hashlib.sha256(legacy_index.encode("utf-8")).hexdigest()}
    edit(5, "Second.\n", mode="append", reason="Second write")
    edit(6, "Third.\n", mode="append", reason="Third write")
    assert pages(7) == {"global/nova": 2}
    assert [row["reason"] for row in history(tail=2)] == ["Second write", "Third write"]
    edit(8, _markdown([]), expected_revision=knowledge.read_knowledge_note(target).revision, reason="Remove the claim")
    assert pages(9) == {"global/nova": 1}
    assert [row.get("reason") for row in history()] == [
        "Legacy write", "Adopt the claim", None, "Second write", "Third write", "Remove the claim",
    ]
    assert history(filter={"field": "publication", "op": "eq", "value": "claims_published"}) == []
    assert history(filter={"field": "type", "op": "eq", "value": "knowledge_index_source"}) == [backup]
    raw = [json.loads(line) for line in journal.read_bytes().splitlines()]
    assert [row.get("publication") for row in raw].count("claims_published") == 4  # one beside each claimed write
    [stored] = [row for row in raw if row.get("type") == "knowledge_index_source"]
    assert stored["old_content"] == legacy_index and "publication" not in stored
    assert raw[-1]["publication"] == "claims_published"
