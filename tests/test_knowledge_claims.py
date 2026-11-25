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
    append = _write(registry, target, "Unrelated body line.\n", mode="append")  # the head's cost, no claim diagnostics
    assert append.startswith("✅") and "Ignored claims" not in append and "room:1 holds 1 head" in append
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
    # Situations that hold the same heads are named together: here each holds this one ("Project notes.\n", 5 tokens).
    assert f"Present situations holding this note: {', '.join(matching)} each hold 1 head, 5 tokens\n" in receipt
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
            "with the claims list in its YAML, or the YAML alone with mode=edit, which keeps the body.")
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


def test_a_yaml_only_edit_sets_names_and_keys_and_keeps_every_body_byte(tmp_path):
    """A name or a key is set without resending the body: an edit whose content is only a YAML block merges its
    fields as an overwrite does and keeps every body byte. Its claims are the turn's explicit write, adopted as an
    overwrite's, so the name holds at once; body text that an edit or an append writes adopts no name, as before."""
    registry = _registry(tmp_path)
    target = knowledge.resolve_knowledge_address(tmp_path, "diary", "global")
    body = "First entry.\r\n" + "An entry under no heading.\r\n" * 400 + "Last entry.\r\n"
    target.path.parent.mkdir(parents=True)
    target.path.write_bytes(("---\ntype: note\nsummary: A diary.\nclaims: ['room:1']\nseen: {by: hand}\n---\n"
                             + body).encode("utf-8"))  # a claims field written by hand, which holds nothing yet

    def edit(**arguments):
        revision = knowledge.read_knowledge_note(target).revision
        return registry.execute("knowledge_write", {"topic": "diary", "scope": "global", "mode": "edit",
                                                    "expected_revision": revision, **arguments})

    def body_of(note):
        return note.raw[note.source.body_span.start_byte:]

    assert edit(old_str="Last entry.", content="Last entry, revised.").startswith("✅")
    assert _write(registry, target, "Appended entry.\r\n", mode="append").startswith("✅")
    assert knowledge.active_claims(tmp_path) == {}  # body text adopted nothing
    before = knowledge.read_knowledge_note(target)

    # The name the YAML already lists by hand: the bytes stay, and the explicit write adopts it.
    named = edit(content="---\nclaims: ['room:1']\n---\n")
    assert named.startswith("✅") and "Present situations holding this note: room:1" in named
    assert target.path.read_bytes() == before.raw and knowledge.active_claims(tmp_path) == {"room:1": [target]}
    assert next(iter(knowledge_versions.NoteHistory(target)))[1]["active_claims"] == ["room:1"]

    # A key: the YAML is merged and re-rendered, every body byte stays, and a place turn's speaker key joins the note.
    key = {"provider": "chat", "namespace": "T-ONE", "id": "U-VELA"}
    keyed = edit(content="---\nkeys: " + json.dumps([key]) + "\n---\n")
    after = knowledge.read_knowledge_note(target)
    assert keyed.startswith("✅") and body_of(after) == body_of(before) and after.raw != before.raw
    assert after.metadata == {**before.metadata, "keys": [key]}  # the summary and the unknown field survive
    assert knowledge.speaker_pages(tmp_path, "chat", "T-ONE", "U-VELA")["page"] == "knowledge_read(topic='diary', scope='global')"
    assert knowledge.active_claims(tmp_path) == {"room:1": [target]}


def test_a_yaml_only_edit_is_revision_checked_keeps_the_archive_field_and_takes_no_body_text(tmp_path):
    """A YAML-only edit replaces an existing note, so it needs the note's current revision; the archive field stays
    with archive and restore; content that carries body text and no old_str is still refused, by the tool and by the
    writer. Each refusal writes nothing, and the same edit with the current revision lands."""
    registry = _registry(tmp_path)
    target = knowledge.resolve_knowledge_address(tmp_path, "diary", "global")
    assert _write(registry, target, "---\ntype: note\n---\nA body under no heading.\n").startswith("✅")
    current = knowledge.read_knowledge_note(target)
    journal = knowledge_versions.journal_path(target)
    stored = target.path.read_bytes(), journal.read_bytes()
    names = "---\nclaims: ['room:1']\n---\n"

    def edit(content, **arguments):
        return registry.execute("knowledge_write", {"topic": "diary", "scope": "global", "mode": "edit",
                                                    "content": content, **arguments})

    for content, arguments, said in (
            (names, {}, "revision_required"),
            (names, {"expected_revision": "0" * 64}, "revision_conflict"),
            ("---\narchive: {at: '2026-11-19', reason: done}\n---\n", {"expected_revision": current.revision},
             "invalid_note: archive is owned by archive/restore modes"),
            (names + "Body text sent beside it.\n", {"expected_revision": current.revision},
             "mode=edit requires edits, or a non-empty old_str, or a summary, or YAML-only content"),
            ("---\nclaims: [room:1\n---\n", {"expected_revision": current.revision}, "TOOL_ARG_ERROR: content: ")):
        reply = edit(content, **arguments)
        assert reply.startswith("⚠️") and said in reply, reply
        assert (target.path.read_bytes(), journal.read_bytes()) == stored
    with pytest.raises(ValueError, match="an edit's content alone is a YAML block"):
        knowledge.write_knowledge_note(target, names + "Body.\n", "edit", expected_revision=current.revision)
    assert (target.path.read_bytes(), journal.read_bytes()) == stored and knowledge.active_claims(tmp_path) == {}
    assert edit(names, expected_revision=current.revision).startswith("✅")
    assert knowledge.active_claims(tmp_path) == {"room:1": [target]}


HELD_HEAD = "A held head takes its space ahead of the memory of other rooms and yields to the window last."
# A head only a link holds in every present situation, or only in some of them, yields before every claimed head there.
CARRIED_HEAD = ("A head a held head links takes its space ahead of the memory of other rooms and yields to the window "
                "before every claimed head.")
MIXED_HEAD = ("A held head takes its space ahead of the memory of other rooms and yields to the window last; where only "
              "a link holds it, before every claimed head.")


def test_a_write_of_a_held_note_says_what_its_head_costs(tmp_path):
    """Where a present situation holds a note, its head stands whole in every start, ahead of the memory of other
    rooms; a note with no `## ` line is all head. So each completed write of a note that holds names states, per
    present situation holding it, the heads that situation holds and their host tokens, then the head's characters
    and tokens and what ends it. A note no present situation holds says so and gives no size; a note that holds no
    name says nothing of a head."""
    registry = _registry(tmp_path)

    def said(receipt):
        return [line for line in receipt.splitlines()
                if line.startswith(("Present situations holding this note:", "Held head:")) or line == HELD_HEAD]

    def write(address, content=None, **kwargs):
        revision = knowledge.read_knowledge_note(address).revision
        return registry.execute("knowledge_write", {"topic": address.topic, "scope": address.scope,
                                                    "expected_revision": revision, "content": content, **kwargs})

    # A head that a `## ` line ends: its size, and a later write below that line leaves it as it was.
    target = knowledge.resolve_knowledge_address(tmp_path, "nova", "global")
    head = "Fact. " * 250 + "\n\n"  # 1,504 characters inside a request: 377 host tokens
    saved = _write(registry, target, _markdown(["room:1"], head + "## History\n\nOlder facts.\n"))
    ended = ["Present situations holding this note: room:1 holds 1 head, 377 tokens",
             "Held head: 1,502 characters, 377 tokens, the body above the first `## ` line.", HELD_HEAD]
    assert saved.startswith("✅") and len(head) == 1_502 and said(saved) == ended
    assert said(write(target, "A later fact.\n", mode="append")) == ended

    # No `## ` line, a fenced one included: the whole body is the head; room:1 now holds both heads, and an append grows it.
    whole = knowledge.resolve_knowledge_address(tmp_path, "tools", "global")
    body = "Whole. " * 200 + "\n\n```\n## not a heading\n```\n"
    created = _write(registry, whole, _markdown(["room:1"], body))
    all_head = "this note has no `## ` line outside a fence, so its whole body is the head."
    assert said(created) == ["Present situations holding this note: room:1 holds 2 heads, 736 tokens",
                             f"Held head: 1,427 characters, 359 tokens: {all_head}", HELD_HEAD]
    grown = write(whole, "More.\n", mode="append")
    assert said(grown) == ["Present situations holding this note: room:1 holds 2 heads, 738 tokens",
                           f"Held head: 1,433 characters, 361 tokens: {all_head}", HELD_HEAD]

    # A note whose names no present situation matches: it says so, and no size as if its head stood somewhere.
    far = knowledge.resolve_knowledge_address(tmp_path, "far", "global")
    unheld = _write(registry, far, _markdown(["room:99999"], body))
    assert unheld.startswith("✅") and said(unheld) == ["Present situations holding this note: none"]
    assert knowledge.active_claims(tmp_path)["room:99999"] == [far]

    # A note that holds no name says nothing of a head: never adopted, on a project shelf, archived, cleared, or refused.
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
    assert cleared.startswith("✅") and REMOVED in cleared and said(cleared) == ["Present situations holding this note: none"]


def test_a_write_names_the_heads_a_held_head_newly_carries_or_releases_and_where_a_carried_note_stands(tmp_path,
                                                                                                    monkeypatch):
    """A held note's head links bring the heads they link: its write lists each one newly carried or released with
    its head size, the total and where they stand, and the links that hold nothing, and its totals count the heads
    it carries. A carried note's own write says where it stands, with those totals and by which carrier, then how
    large its head is and what that head takes. A note neither claimed nor carried hears none of it."""
    registry = _registry(tmp_path)
    prefixes = ("Newly carried", "Released", "Linked from this head", "Present situations", "Held head", HELD_HEAD,
                CARRIED_HEAD, MIXED_HEAD)

    def said(receipt):
        assert receipt.startswith("✅"), receipt
        return [line for line in receipt.splitlines() if line.startswith(prefixes)]

    def write(address, content, **kwargs):
        revision = knowledge.read_knowledge_note(address).revision
        return registry.execute("knowledge_write", {"topic": address.topic, "scope": address.scope, "mode": "overwrite",
                                                    "content": content, "expected_revision": revision, **kwargs})

    sasha, tools, nova, plain = (knowledge.resolve_knowledge_address(tmp_path, topic, "global")
                                 for topic in ("people/sasha", "tools", "work/nova", "plain"))
    head = "How to work with Sasha.\n\n"
    assert not said(_write(registry, sasha, "---\ntype: note\n---\n" + head + "## History\n\nOlder words.\n"))
    assert not said(_write(registry, tools, "---\ntype: note\n---\nThe instruments.\n"))
    links = "Nova. [Sasha](../people/sasha.md) [Tools](../tools.md) [gone](../gone.md)\n"
    # room:1 holds this head (20 tokens) and the two it carries: sasha's (8) and tools' (5).
    assert said(_write(registry, nova, _markdown(["room:1"], links))) == [
        "Present situations holding this note: room:1 holds 3 heads, 33 tokens",
        f"Newly carried by this head: people/sasha ({len(head)} characters), tools (17 characters; no `## ` line "
        f"outside a fence: the whole body); {len(head) + 17} characters in all, standing where this note's own names "
        "hold it: room:1.",
        "Linked from this head, not held: memory/knowledge/gone.md (no such note).",
        f"Held head: {len(links)} characters, 20 tokens: this note has no `## ` line outside a fence, so its whole body "
        "is the head.", HELD_HEAD]

    edited = write(sasha, "---\ntype: note\n---\n" + head + "More.\n\n## History\n\nOlder words.\n")
    assert said(edited) == ["Present situations holding this note: room:1 holds 3 heads, 35 tokens (carried by "
                            "work/nova)",
                            f"Held head: {len(head) + 7} characters, 10 tokens, the body above the first `## ` line.",
                            CARRIED_HEAD]  # only a link holds it: it yields before every claimed head
    claimed = write(tools, _markdown(["room:1"], "The instruments.\n"))  # its own name and nova's link: one head
    assert said(claimed)[0] == ("Present situations holding this note: room:1 holds 3 heads, 35 tokens "
                                "(also carried by work/nova)")
    assert said(claimed)[-1] == HELD_HEAD  # its own name holds it where a link does too: it yields last
    assert not said(_write(registry, plain, "---\ntype: note\n---\nNeither claimed nor carried.\n"))

    dropped = write(nova, _markdown(["room:1"], "Nova. [Tools](../tools.md)\n"))
    assert said(dropped) == ["Present situations holding this note: room:1 holds 2 heads, 13 tokens",
                             f"Released by this head: people/sasha ({len(head) + 7} characters; now held: none); "
                             f"{len(head) + 7} characters in all.",
                             "Held head: 27 characters, 8 tokens: this note has no `## ` line outside a fence, so its "
                             "whole body is the head.", HELD_HEAD]
    assert not said(write(sasha, "---\ntype: note\n---\nNo longer carried.\n"))

    def unreadable(*_args, **_kwargs):  # the journal cannot give the version the write replaced
        raise knowledge_versions.VersionUnavailable("no row holds it")

    monkeypatch.setattr(knowledge_versions, "find_version", unreadable)
    unknown = write(nova, _markdown(["room:1"], "Nova. [Sasha](../people/sasha.md)\n"))
    assert "What this head carried before the write is unknown: no row holds it." in unknown
    assert not any(line.startswith(("Newly carried", "Released")) for line in said(unknown))


def test_one_present_line_names_why_a_note_stands_in_each_situation_and_the_shelf_is_read_once(tmp_path, monkeypatch):
    """One line names every present situation where a note stands, by its own names or carried by the held heads that
    link it, each once: the totals count every head standing there, and situations alike in heads and in why are
    named together. Where no present situation holds a carried note it says none and names the carriers, with no
    size. Who holds what is read once per write, never before a claim was recorded; a note neither claimed nor
    carried gets nothing after its situation line."""
    from ouroboros import held_notes
    from ouroboros.projects_registry import create_project

    registry = _registry(tmp_path)
    rooms = ["room:1", f"room:{create_project(tmp_path, 'writer', name='Writer')['chat_id']}"]
    both = ", ".join(rooms)
    reads, standing = [], held_notes.standing
    monkeypatch.setattr(held_notes, "standing", lambda root: reads.append(root) or standing(root))

    def write(address, content):
        """The lines after the situation line, and how many times this write read who holds what."""
        current = {"expected_revision": knowledge.read_knowledge_note(address).revision} if address.path.is_file() else {}
        receipt = registry.execute("knowledge_write", {"topic": address.topic, "scope": address.scope,
                                                       "mode": "overwrite", "content": content, **current})
        assert receipt.startswith("✅") and receipt.splitlines()[2] == "This situation's names: room:1 (Main)", receipt
        said, count = receipt.splitlines()[3:], len(reads)
        reads.clear()
        return said, count

    sasha, tools, nova, aika, far, lena, plain, ivan = (
        knowledge.resolve_knowledge_address(tmp_path, topic, "global")
        for topic in ("people/sasha", "tools", "work/nova", "work/aika", "far", "people/lena", "plain", "people/ivan"))
    head, all_head = ("How to work with Sasha.\n\n",
                      "this note has no `## ` line outside a fence, so its whole body is the head.")
    # Before any claim was recorded on the shelf: nothing is read and nothing is said.
    assert write(sasha, "---\ntype: note\n---\n" + head + "## History\n\nOlder words.\n") == ([], 0)
    assert write(tools, "---\ntype: note\n---\nThe instruments.\n") == ([], 0)
    assert not knowledge_versions.claim_record_path(nova.shelf, nova.topic).parent.exists()

    # Both rooms hold nova's head and the two it carries; then aika's, which carries sasha too: still one sasha.
    assert write(nova, _markdown(rooms, "Nova. [Sasha](../people/sasha.md) [Tools](../tools.md)\n")) == ([
        f"Present situations holding this note: {both} each hold 3 heads, 28 tokens",
        f"Newly carried by this head: people/sasha (25 characters), tools (17 characters; no `## ` line outside a "
        f"fence: the whole body); 42 characters in all, standing where this note's own names hold it: {both}.",
        f"Held head: 55 characters, 15 tokens: {all_head}", HELD_HEAD], 1)
    assert write(aika, _markdown(rooms, "Aika. [Sasha](../people/sasha.md)\n"))[0][0] == (
        f"Present situations holding this note: {both} each hold 4 heads, 38 tokens")
    assert write(sasha, "---\ntype: note\n---\n" + head + "More.\n\n## History\n\nOlder words.\n") == ([
        f"Present situations holding this note: {both} each hold 4 heads, 40 tokens (carried by work/aika, work/nova)",
        "Held head: 32 characters, 10 tokens, the body above the first `## ` line.", CARRIED_HEAD], 1)

    # Its own names and a link: one head per situation, the reason once per group; a group per distinct reason. Where
    # its own names hold it in every present situation it yields last; where only the link holds it, first.
    tools_head = f"Held head: 17 characters, 5 tokens: {all_head}"
    assert write(tools, _markdown(rooms, "The instruments.\n")) == ([
        f"Present situations holding this note: {both} each hold 4 heads, 40 tokens (also carried by work/nova)",
        tools_head, HELD_HEAD], 1)
    assert write(tools, _markdown(["room:1"], "The instruments.\n")) == ([
        "Present situations holding this note: room:1 holds 4 heads, 40 tokens (also carried by work/nova); "
        f"{rooms[1]} holds 4 heads, 40 tokens (carried by work/nova)", REMOVED, tools_head, MIXED_HEAD], 1)
    # The heads its own head links stand only where its own names hold it, not where a link holds it (one hop).
    assert write(ivan, "---\ntype: note\n---\nIvan.\n") == ([], 1)
    linking = write(tools, _markdown(["room:1"], "The instruments. [Ivan](people/ivan.md)\n"))[0]
    assert rooms[1] in linking[0] and linking[1] == (
        "Newly carried by this head: people/ivan (6 characters; no `## ` line outside a fence: the whole body); "
        "6 characters in all, standing where this note's own names hold it: room:1.")

    # A carrier whose names match no present situation: none, with its carrier named, and no size.
    assert write(lena, "---\ntype: note\n---\nLena.\n") == ([], 1)
    assert write(far, _markdown(["room:99999"], "Far. [Lena](people/lena.md)\n")) == ([
        "Claims with no matching situation found: 'room:99999'", "Present situations holding this note: none",
        "Newly carried by this head: people/lena (6 characters; no `## ` line outside a fence: the whole body); "
        "6 characters in all, standing where this note's own names hold it: none."], 1)
    assert write(lena, "---\ntype: note\n---\nLena, again.\n") == (
        ["Present situations holding this note: none (carried by far)"], 1)
    assert write(plain, "---\ntype: note\n---\nNeither claimed nor carried.\n") == ([], 1)


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
